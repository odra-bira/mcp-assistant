"""
MCP Assistant: command-line version.

    python app.py                     chat (Claude if ANTHROPIC_API_KEY is set, else Ollama if running, else demo)
    python app.py --brain ollama      free local model (set OLLAMA_MODEL to change it, default qwen3:8b)
    python app.py --brain claude      Claude through the API
    python app.py --demo              offline rule-based demo brain (no model at all)
    python app.py "find python jobs"  ask one question and exit
    python app.py --with-evil         also connect the evil demo server
    python app.py --with-evil --no-safety   ...and switch the scanner off to watch the attack work
    python app.py --yes               auto-approve write actions (for tests only)

In chat:  /tools   list tools     /log   show tool calls     /quit   exit
"""

import argparse
import asyncio
import json
from pathlib import Path

from assistant.agent import Agent
from assistant.brains import make_brain
from assistant.hub import ServerHub
from assistant.safety import ToolLog, scan_tools
from assistant.settings import build_config

ROOT = Path(__file__).resolve().parent


def print_event(kind: str, data: dict) -> None:
    if kind == "thinking":
        print(f"  [brain] {data['text']}")
    elif kind == "tool_call":
        print(f"  [tool call]  {data['tool']}  {json.dumps(data['args'])}")
    elif kind == "tool_result":
        status = "ERROR" if data["is_error"] else "ok"
        preview = data["result_preview"].replace("\n", " ")[:110]
        print(f"  [result {status}]  {preview}")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", help="ask one question and exit")
    p.add_argument("--brain", choices=["auto", "claude", "ollama", "demo"], default="auto")
    p.add_argument("--demo", action="store_true", help="same as --brain demo")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--with-evil", action="store_true")
    p.add_argument("--no-safety", action="store_true")
    p.add_argument("--config", default="config.json",
                   help="config file in assistant/ (config.remote.json = remote job search)")
    a = p.parse_args()

    async def confirm(tool: str, args: dict, call_id: str = "") -> bool:
        if a.yes:
            print(f"  [auto-approved] {tool}")
            return True
        answer = await asyncio.to_thread(input, f"  Allow {tool} {json.dumps(args)}? [y/N] ")
        return answer.strip().lower() in ("y", "yes")

    config_path = build_config(a.config, a.with_evil, a.no_safety)
    async with ServerHub(config_path) as hub:
        quarantine = hub.config["safety"]["quarantine_suspicious_tools"]
        warnings = scan_tools(list(hub.tools.values()), quarantine=quarantine)

        brain = make_brain("demo" if a.demo else a.brain)
        print(f"Brain: {brain.name}")
        print(f"Connected servers: {', '.join(hub.clients)}")
        print(f"Tools available: {len(hub.active_tools())}")
        for w in warnings:
            print(f"  SAFETY WARNING: {w}")

        agent = Agent(hub, brain, ToolLog(ROOT / "logs" / "tool_calls.jsonl"), confirm, print_event)

        if a.question:
            print(f"\nYou: {a.question}")
            print(f"\nAssistant: {await agent.ask(a.question)}")
            return

        print("\nType a request, or /tools, /log, /quit\n")
        while True:
            try:
                text = (await asyncio.to_thread(input, "You: ")).strip()
            except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
                print()
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/tools":
                for t in hub.tools.values():
                    flag = " (QUARANTINED)" if t.quarantined else ""
                    kind = "read" if t.read_only else "WRITE"
                    print(f"  {t.full_name} [{kind}]{flag}: {t.tool.description}")
                continue
            if text == "/log":
                for e in agent.log.entries:
                    print(f"  {e['time']} {e['tool']} approved={e['approved']} error={e['is_error']}")
                continue
            try:
                print(f"\nAssistant: {await agent.ask(text)}\n")
            except RuntimeError as e:  # e.g. Ollama not running
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:  # e.g. a remote server isn't running
        print(f"Error: {e}")
