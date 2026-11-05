"""
MCP Assistant: web version.

    python web.py                      opens http://127.0.0.1:8000 in your browser
    python web.py --brain ollama       pick the brain (auto | claude | ollama | demo)
    python web.py --with-evil          include the evil demo server (try the attack in the UI)
    python web.py --config config.remote.json

Same hub, brain, agent and safety layer as app.py. Only the "front door" changes:
instead of input() in a terminal, the browser talks to these endpoints:

    GET  /              the page
    GET  /api/info      brain, servers, tools, safety warnings
    GET  /api/events    a live stream of what the agent is doing (Server-Sent Events)
    POST /api/ask       send a message
    POST /api/decide    approve or deny a pending write action
    POST /api/reset     start a new conversation

Runs on 127.0.0.1 only, so other computers on your network can't reach it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import threading
import webbrowser
from pathlib import Path

import uvicorn
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from assistant.agent import Agent
from assistant.brains import make_brain
from assistant.hub import ServerHub
from assistant.safety import ToolLog, scan_tools
from assistant.settings import build_config

ROOT = Path(__file__).resolve().parent


class WebSession:
    """Holds the one running assistant and connects it to browsers."""

    def __init__(self, args):
        self.args = args
        self.hub: ServerHub | None = None
        self.agent: Agent | None = None
        self.brain = None
        self.warnings: list[str] = []
        self.listeners: set[asyncio.Queue] = set()
        self.pending: dict[str, asyncio.Future] = {}   # call_id -> waiting for Approve/Deny
        self.busy = False

    # -- events out to every open browser tab ---------------------------------
    def publish(self, kind: str, data: dict) -> None:
        for q in list(self.listeners):
            q.put_nowait({"event": kind, "data": json.dumps(data)})

    # -- the approval gate: wait until the user clicks Approve or Deny --------
    async def confirm(self, tool: str, args: dict, call_id: str) -> bool:
        future = asyncio.get_running_loop().create_future()
        self.pending[call_id] = future
        self.publish("approval_needed", {"id": call_id, "tool": tool, "args": args})
        try:
            return await future
        finally:
            self.pending.pop(call_id, None)

    async def run_request(self, text: str) -> None:
        self.busy = True
        self.publish("status", {"busy": True})
        try:
            answer = await self.agent.ask(text)
            self.publish("answer", {"text": answer})
        except Exception as exc:
            self.publish("error", {"text": f"{type(exc).__name__}: {exc}"})
        finally:
            self.busy = False
            self.publish("status", {"busy": False})

    # -- startup / shutdown ---------------------------------------------------
    @contextlib.asynccontextmanager
    async def lifespan(self, app):
        a = self.args
        config = build_config(a.config, a.with_evil, a.no_safety)
        async with ServerHub(config) as hub:
            self.hub = hub
            self.warnings = scan_tools(list(hub.tools.values()),
                                       quarantine=hub.config["safety"]["quarantine_suspicious_tools"])
            self.brain = make_brain(a.brain)
            self.agent = Agent(hub, self.brain, ToolLog(ROOT / "logs" / "tool_calls.jsonl"),
                               self.confirm, self.publish)
            print(f"Brain: {self.brain.name}  ·  servers: {', '.join(hub.clients)}")
            for w in self.warnings:
                print(f"  SAFETY WARNING: {w}")
            yield
            for f in self.pending.values():  # unblock anything still waiting
                f.cancel()


def create_app(args) -> Starlette:
    s = WebSession(args)

    async def page(request: Request):
        return FileResponse(ROOT / "static" / "index.html")

    async def info(request: Request):
        return JSONResponse({
            "brain": s.brain.name,
            "servers": list(s.hub.clients),
            "warnings": s.warnings,
            "busy": s.busy,
            "tools": [{"name": t.full_name, "server": t.server, "description": t.tool.description,
                       "writes": not t.read_only, "quarantined": t.quarantined}
                      for t in s.hub.tools.values()],
        })

    async def events(request: Request):
        q: asyncio.Queue = asyncio.Queue()
        s.listeners.add(q)

        async def stream():
            try:
                while True:
                    yield await q.get()
            finally:
                s.listeners.discard(q)

        return EventSourceResponse(stream(), ping=15)

    async def ask(request: Request):
        body = await request.json()
        text = str(body.get("text", "")).strip()
        if not text:
            return JSONResponse({"error": "Type a request first."}, status_code=400)
        if s.busy:
            return JSONResponse({"error": "Still working on the last request."}, status_code=409)
        s.publish("user", {"text": text})
        asyncio.create_task(s.run_request(text))
        return JSONResponse({"ok": True}, status_code=202)

    async def decide(request: Request):
        body = await request.json()
        future = s.pending.get(str(body.get("id")))
        if future is None or future.done():
            return JSONResponse({"error": "That action is no longer waiting."}, status_code=404)
        allow = bool(body.get("allow"))
        future.set_result(allow)
        s.publish("decision", {"id": body["id"], "allow": allow})
        return JSONResponse({"ok": True})

    async def reset(request: Request):
        if s.busy:
            return JSONResponse({"error": "Wait for the current request to finish."}, status_code=409)
        s.agent.history.clear()
        s.publish("reset", {})
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/", page),
            Route("/api/info", info),
            Route("/api/events", events),
            Route("/api/ask", ask, methods=["POST"]),
            Route("/api/decide", decide, methods=["POST"]),
            Route("/api/reset", reset, methods=["POST"]),
        ],
        lifespan=s.lifespan,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--brain", choices=["auto", "claude", "ollama", "demo"], default="auto")
    p.add_argument("--config", default="config.json")
    p.add_argument("--with-evil", action="store_true")
    p.add_argument("--no-safety", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()

    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    print(f"Open {url}   (Control+C to stop)")
    try:
        uvicorn.run(create_app(a), host="127.0.0.1", port=a.port, log_level="warning")
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
