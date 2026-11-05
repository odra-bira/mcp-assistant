"""Tests for the agent loop, using a scripted brain and a fake hub (no servers, no AI)."""

import pytest
from mcp.types import Tool, ToolAnnotations

from assistant.agent import Agent
from assistant.brains import OllamaBrain, Reply, ToolCall
from assistant.hub import HubTool
from assistant.safety import ToolLog

pytestmark = pytest.mark.anyio


class FakeHub:
    def __init__(self):
        self.config = {"safety": {"confirm_write_tools": True}, "max_steps_per_turn": 4}
        self.calls = []
        self.tools = {}
        for server, name, ro in [("jobsearch", "search_jobs", True), ("jobsdb", "save_application", False),
                                 ("evil", "steal", False)]:
            t = HubTool(server, Tool(name=name, description="", input_schema={"type": "object"},
                                     annotations=ToolAnnotations(read_only_hint=ro)))
            self.tools[t.full_name] = t
        self.tools["evil__steal"].quarantined = True

    def tools_for_llm(self):
        return [{"name": n, "description": "", "input_schema": {}} for n, t in self.tools.items()
                if not t.quarantined]

    async def call(self, name, args):
        self.calls.append(name)
        return f"result of {name}", False


class ScriptedBrain:
    """Returns pre-written replies in order, like a fake LLM."""
    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen_tools = None

    async def think(self, history, tools):
        self.seen_tools = [t["name"] for t in tools]
        return self.replies.pop(0)


def agent_with(brain, hub, tmp_path, approve=True):
    asked = []

    async def confirm(tool, args, call_id):
        asked.append(tool)
        return approve

    agent = Agent(hub, brain, ToolLog(tmp_path / "log.jsonl"), confirm)
    return agent, asked


async def test_read_tool_runs_without_asking(tmp_path):
    hub = FakeHub()
    brain = ScriptedBrain(Reply(tool_calls=[ToolCall("1", "jobsearch__search_jobs", {"query": "py"})]),
                          Reply(text="Found jobs."))
    agent, asked = agent_with(brain, hub, tmp_path)
    assert await agent.ask("find jobs") == "Found jobs."
    assert hub.calls == ["jobsearch__search_jobs"] and asked == []


async def test_write_tool_asks_and_respects_deny(tmp_path):
    hub = FakeHub()
    brain = ScriptedBrain(Reply(tool_calls=[ToolCall("1", "jobsdb__save_application", {})]),
                          Reply(text="OK, not saved."))
    agent, asked = agent_with(brain, hub, tmp_path, approve=False)
    await agent.ask("save it")
    assert asked == ["jobsdb__save_application"]
    assert hub.calls == []                                   # denied -> never reached the server
    result_block = agent.history[2]["content"][0]
    assert result_block["is_error"] and "declined" in result_block["content"]


async def test_quarantined_tool_is_hidden_and_blocked(tmp_path):
    hub = FakeHub()
    brain = ScriptedBrain(Reply(tool_calls=[ToolCall("1", "evil__steal", {})]), Reply(text="done"))
    agent, _ = agent_with(brain, hub, tmp_path)
    await agent.ask("anything")
    assert "evil__steal" not in brain.seen_tools             # the AI never even sees it
    assert hub.calls == []                                   # and calling it anyway fails


async def test_stops_after_max_steps(tmp_path):
    hub = FakeHub()
    loop = Reply(tool_calls=[ToolCall("x", "jobsearch__search_jobs", {})])
    agent, _ = agent_with(ScriptedBrain(*[loop] * 10), hub, tmp_path)
    assert "too many tool steps" in await agent.ask("loop forever")
    assert len(hub.calls) == 4


async def test_every_call_is_logged(tmp_path):
    hub = FakeHub()
    brain = ScriptedBrain(Reply(tool_calls=[ToolCall("1", "jobsearch__search_jobs", {}),
                                            ToolCall("2", "jobsdb__save_application", {})]),
                          Reply(text="done"))
    agent, _ = agent_with(brain, hub, tmp_path)
    await agent.ask("go")
    assert [e["tool"] for e in agent.log.entries] == ["jobsearch__search_jobs", "jobsdb__save_application"]
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


def test_ollama_translation_keeps_tool_names_and_results():
    history = [
        {"role": "user", "content": "find jobs"},
        {"role": "assistant", "content": [{"type": "text", "text": "Searching."},
                                          {"type": "tool_use", "id": "a", "name": "jobsearch__search_jobs",
                                           "input": {"query": "py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "[]"}]},
    ]
    msgs = OllamaBrain._to_ollama(history)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "jobsearch__search_jobs"
    assert msgs[3]["tool_name"] == "jobsearch__search_jobs" and msgs[3]["content"] == "[]"
