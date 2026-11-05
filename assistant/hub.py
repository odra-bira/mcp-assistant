"""
MCP client hub.

Connects to every server in the config (local servers over stdio, remote servers over
Streamable HTTP with bearer-token auth), aggregates their tools into one namespaced
catalog, and routes tool calls to the owning server.

Tools are namespaced as "<server>__<tool>" (e.g. "jobsdb__save_application"), so two
servers exposing the same tool name can never be confused for one another
(tool-name collision, Hou et al. 2025, Section 5.2.1).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import Tool

ROOT = Path(__file__).resolve().parent.parent
SEP = "__"


@dataclass
class HubTool:
    server: str           # which server owns it, e.g. "jobsdb"
    tool: Tool            # the original MCP tool definition
    quarantined: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.server}{SEP}{self.tool.name}"

    @property
    def read_only(self) -> bool:
        ann = self.tool.annotations
        # No annotation = we don't know, so treat it as a write (safe default).
        return bool(ann and ann.read_only_hint)


class ServerHub:
    def __init__(self, config_path: Path):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.clients: dict[str, Client] = {}
        self.tools: dict[str, HubTool] = {}
        self._stack = AsyncExitStack()

    # -- 1. CONNECT ---------------------------------------------------------
    async def _connect(self, spec: dict) -> Client:
        """Supported transports (Hou et al. 2025, Section 3.2):
        - local:  "command" -> start the server as a subprocess, talk over stdin/stdout
        - remote: "url"     -> talk HTTP to a server that's already running somewhere
        """
        if spec.get("url"):
            headers = {}
            token_env = spec.get("token_env")
            if token_env:
                token = os.environ.get(token_env)
                if not token:
                    raise RuntimeError(f"Server '{spec['name']}' needs the {token_env} environment variable.")
                headers["Authorization"] = f"Bearer {token}"
            http = await self._stack.enter_async_context(create_mcp_http_client(headers=headers))
            transport = streamable_http_client(spec["url"], http_client=http)
            try:
                return await self._stack.enter_async_context(Client(transport))
            except Exception as exc:
                raise RuntimeError(
                    f"Couldn't connect to remote server '{spec['name']}' at {spec['url']}. "
                    "Is it running, and is the token correct?") from exc

        command = sys.executable if spec["command"] == "python" else spec["command"]
        args = [str(ROOT / a) if a.endswith(".py") else a for a in spec.get("args", [])]
        params = StdioServerParameters(command=command, args=args, env=spec.get("env"))
        return await self._stack.enter_async_context(Client(params))

    async def __aenter__(self) -> "ServerHub":
        await self._stack.__aenter__()
        try:
            for spec in self.config["servers"]:
                if not spec.get("enabled", True):
                    continue
                client = await self._connect(spec)
                self.clients[spec["name"]] = client

                # -- 2. COLLECT (the "what can you do?" step of the handshake) --
                listed = await client.list_tools()
                for tool in listed.tools:
                    hub_tool = HubTool(server=spec["name"], tool=tool)
                    self.tools[hub_tool.full_name] = hub_tool
        except BaseException:
            await self._stack.aclose()  # stop any servers we already started
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.__aexit__(*exc)

    def active_tools(self) -> list[HubTool]:
        return [t for t in self.tools.values() if not t.quarantined]

    def tools_for_llm(self) -> list[dict]:
        """Tool list in the format LLM APIs expect (name, description, JSON schema)."""
        return [
            {
                "name": t.full_name,
                "description": t.tool.description or "",
                "input_schema": t.tool.input_schema or {"type": "object", "properties": {}},
            }
            for t in self.active_tools()
        ]

    # -- 3. ROUTE -----------------------------------------------------------
    async def call(self, full_name: str, args: dict) -> tuple[str, bool]:
        """Run a tool. Returns (text result, is_error)."""
        hub_tool = self.tools.get(full_name)
        if hub_tool is None or hub_tool.quarantined:
            return f"Tool '{full_name}' is not available.", True
        client = self.clients[hub_tool.server]
        result = await client.call_tool(hub_tool.tool.name, args)
        parts = [getattr(block, "text", str(block)) for block in result.content]
        return "\n".join(parts), bool(result.is_error)
