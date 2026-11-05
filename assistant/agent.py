"""
Agent loop with a human-approval gate.

    user message
       |
       v
    brain.think(history, tools) --> no tool calls --> return final text
       |
       | tool calls requested
       v
    per call: availability check -> approval (non-read-only tools) -> hub.call() -> audit log
       |
       v
    append results to history, repeat (bounded by max_steps_per_turn)
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from .hub import ServerHub
from .safety import ToolLog, needs_confirmation

# confirm(tool_name, args, call_id) -> True to allow, False to block
ConfirmFn = Callable[[str, dict, str], Awaitable[bool]]
# on_event(kind, data): lets a UI show progress ("tool_call", "tool_result", ...)
EventFn = Callable[[str, dict], None]


class Agent:
    def __init__(self, hub: ServerHub, brain, log: ToolLog, confirm: ConfirmFn,
                 on_event: EventFn | None = None):
        self.hub = hub
        self.brain = brain
        self.log = log
        self.confirm = confirm
        self.on_event = on_event or (lambda kind, data: None)
        self.history: list[dict] = []
        self.max_steps = hub.config.get("max_steps_per_turn", 8)
        self.confirm_writes = hub.config.get("safety", {}).get("confirm_write_tools", True)

    async def ask(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        tools = self.hub.tools_for_llm()

        for _ in range(self.max_steps):
            reply = await self.brain.think(self.history, tools)
            self.history.append({"role": "assistant", "content": reply.as_blocks()})

            if not reply.tool_calls:          # the brain is finished
                return reply.text

            if reply.text:
                self.on_event("thinking", {"text": reply.text})

            results = []
            for call in reply.tool_calls:  # the brain may ask for several at once
                result_text, is_error = await self._run_one(call.id, call.name, call.args)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result_text,
                    "is_error": is_error,
                })
            # Tool results go back to the brain as a "user" message (API convention).
            self.history.append({"role": "user", "content": results})

        return "Stopped: too many tool steps for one request."

    async def _run_one(self, call_id: str, name: str, args: dict) -> tuple[str, bool]:
        hub_tool = self.hub.tools.get(name)
        ask_first = bool(hub_tool and not hub_tool.quarantined
                         and needs_confirmation(hub_tool, self.confirm_writes))
        self.on_event("tool_call", {"id": call_id, "tool": name, "args": args,
                                    "server": name.split("__")[0], "needs_approval": ask_first})
        start = time.perf_counter()

        if hub_tool is None or hub_tool.quarantined:
            result, is_error, approved = f"Tool '{name}' is not available.", True, False
        elif ask_first and not await self.confirm(name, args, call_id):
            result, is_error, approved = "The user declined this action.", True, False
        else:
            approved = True
            try:
                result, is_error = await self.hub.call(name, args)
            except Exception as exc:  # a broken server shouldn't crash the app
                result, is_error = f"Tool failed: {exc}", True

        entry = self.log.record(tool=name, args=args, approved=approved, result=result,
                                is_error=is_error, seconds=time.perf_counter() - start)
        self.on_event("tool_result", {**entry, "id": call_id, "result": result[:4000]})
        return result, is_error
