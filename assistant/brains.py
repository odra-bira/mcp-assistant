"""
Model adapters ("brains").

A brain decides; it never executes. Given the conversation and the tool catalog it
returns text and/or tool-call requests, which the agent executes through the hub.

All adapters share one interface, so the model is swappable without touching the
agent, hub or safety layer:
    ClaudeBrain   Anthropic Messages API (requires ANTHROPIC_API_KEY)
    OllamaBrain   any local tool-calling model served by Ollama
    DemoBrain     deterministic rule-based stand-in for offline tests and CI

Conversation history is stored in Anthropic message format; adapters translate as needed.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field

SYSTEM_PROMPT = """You are a job-search assistant with tools. ACT with your tools; do not ask
the user for information a tool can get.

Tools and when to use them:
- resume__read_resume: get the user's skills and experience. Call it whenever the request
  mentions the resume or "match".
- jobsearch__search_jobs: find job listings. YOU choose the keywords (for example "python")
  from the request and resume. Never ask the user for job titles or companies; search instead.
- jobsdb__save_application: save a job. Use the company, title and url from search results.
- jobsdb__list_applications: show the user's saved jobs.

For "find jobs that match my resume and save the best two": call read_resume, then
search_jobs, pick the two jobs whose skills best overlap the resume, then call
save_application once for each. Finally tell the user what you saved and why, in a few lines.

When you list jobs, include each job's URL and name the source shown in the search results
(for example "via Remotive")."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_blocks(self) -> list[dict]:
        """Turn the reply into history blocks."""
        blocks: list[dict] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for c in self.tool_calls:
            blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
        return blocks


# ---------------------------------------------------------------------------
# The real brain
# ---------------------------------------------------------------------------
class ClaudeBrain:
    name = "Claude"

    def __init__(self, model: str | None = None):
        import anthropic  # pip install anthropic

        self.client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

    async def think(self, history: list[dict], tools: list[dict]) -> Reply:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=history,
        )
        reply = Reply()
        for block in response.content:
            if block.type == "text":
                reply.text += block.text
            elif block.type == "tool_use":
                reply.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        return reply


# ---------------------------------------------------------------------------
# A free local brain: any tool-calling model running in Ollama on your computer
# ---------------------------------------------------------------------------
class OllamaBrain:
    """Talks to Ollama's local HTTP API (http://localhost:11434). No API key, no cost.

    Ollama uses a different message format than Anthropic, so this class
    TRANSLATES our history into Ollama's format and its reply back into ours.
    That translation is the only thing that changes between brains.
    """

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.name = f"Ollama ({self.model})"

    async def think(self, history: list[dict], tools: list[dict]) -> Reply:
        import asyncio

        payload = {
            "model": self.model,
            "stream": False,
            "messages": self._to_ollama(history),
            "tools": [
                {"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}}
                for t in tools
            ],
        }
        if os.environ.get("OLLAMA_THINK") in ("0", "false"):
            payload["think"] = False  # faster answers from "thinking" models like qwen3
        data = await asyncio.to_thread(self._post, "/api/chat", payload)

        message = data.get("message", {})
        text = re.sub(r"<think>.*?</think>", "", message.get("content") or "", flags=re.S).strip()
        reply = Reply(text=text)
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some models send arguments as a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            reply.tool_calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", fn.get("name", ""), args))
        return reply

    def _post(self, path: str, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama error {e.code}: {e.read().decode()[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Can't reach Ollama at {self.host}. Is the Ollama app running?") from e

    @staticmethod
    def _to_ollama(history: list[dict]) -> list[dict]:
        """Anthropic-style history -> Ollama messages."""
        out = [{"role": "system", "content": SYSTEM_PROMPT}]
        names: dict[str, str] = {}  # tool_use id -> tool name
        for msg in history:
            content = msg["content"]
            if isinstance(content, str):
                out.append({"role": msg["role"], "content": content})
                continue
            if msg["role"] == "assistant":
                text = "".join(b["text"] for b in content if b["type"] == "text")
                calls = []
                for b in content:
                    if b["type"] == "tool_use":
                        names[b["id"]] = b["name"]
                        calls.append({"function": {"name": b["name"], "arguments": b["input"]}})
                entry = {"role": "assistant", "content": text}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
            else:  # a "user" message holding tool results
                for b in content:
                    if b["type"] == "tool_result":
                        out.append({"role": "tool", "tool_name": names.get(b["tool_use_id"], ""),
                                    "content": str(b["content"])})
        return out


# ---------------------------------------------------------------------------
# The offline demo brain (rule-based, NOT smart; just for testing the plumbing)
# ---------------------------------------------------------------------------
SKILL_WORDS = ["python", "data", "react", "frontend", "backend", "ai", "mcp", "machine learning", "platform"]


class DemoBrain:
    name = "Demo (offline rules)"

    async def think(self, history: list[dict], tools: list[dict]) -> Reply:
        request = self._last_user_text(history).lower()
        done = self._results_this_turn(history)   # {short tool name: result text}

        def pick(short: str) -> str | None:
            """Find a tool by short name. Deliberately prefers descriptions that say
            'prefer this tool', modelling the hijacking behaviour reported by Hou et al. (2025)."""
            matches = [t for t in tools if t["name"].split("__")[-1] == short]
            if not matches:
                return None
            pushy = [t for t in matches if "prefer" in t["description"].lower()]
            return (pushy or matches)[0]["name"]

        def call(short: str, args: dict) -> Reply | None:
            name = pick(short)
            if not name:
                return None
            return Reply(tool_calls=[ToolCall(f"demo_{uuid.uuid4().hex[:8]}", name, args)])

        wants_resume = "resume" in request or "match" in request
        wants_search = any(w in request for w in ("find", "search", "jobs", "job"))
        wants_save = re.search(r"\bsave\b", request) is not None
        wants_list = any(w in request for w in ("my applications", "list", "show my", "saved"))

        if wants_resume and "read_resume" not in done:
            r = call("read_resume", {})
            if r:
                return r

        if wants_search and "search_jobs" not in done and not (wants_list and not wants_save and "find" not in request):
            query = next((w for w in SKILL_WORDS if w in request), "python")
            r = call("search_jobs", {"query": query, "remote_only": "remote" in request, "limit": 5})
            if r:
                return r

        if wants_save and "search_jobs" in done and "save_application" not in done:
            best = self._best_matches(done.get("search_jobs", ""), done.get("read_resume", ""), n=2)
            name = pick("save_application")
            if best and name:
                return Reply(
                    text="Saving the two best matches.",
                    tool_calls=[
                        ToolCall(f"demo_{uuid.uuid4().hex[:8]}", name,
                                 {"company": j["company"], "title": j["title"], "url": j["url"]})
                        for j in best
                    ],
                )

        if wants_list and "list_applications" not in done and not wants_save:
            r = call("list_applications", {})
            if r:
                return r

        return Reply(text=self._summary(done))

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _last_user_text(history: list[dict]) -> str:
        for msg in reversed(history):
            if msg["role"] == "user" and isinstance(msg["content"], str):
                return msg["content"]
        return ""

    @staticmethod
    def _results_this_turn(history: list[dict]) -> dict[str, str]:
        """Collect tool results since the user's last typed message."""
        start = max(i for i, m in enumerate(history) if m["role"] == "user" and isinstance(m["content"], str))
        names: dict[str, str] = {}
        results: dict[str, str] = {}
        for msg in history[start + 1:]:
            for block in msg["content"] if isinstance(msg["content"], list) else []:
                if block["type"] == "tool_use":
                    names[block["id"]] = block["name"].split("__")[-1]
                elif block["type"] == "tool_result":
                    short = names.get(block["tool_use_id"], "?")
                    prev = results.get(short)
                    results[short] = f"{prev} | {block['content']}" if prev else str(block["content"])
        return results

    @staticmethod
    def _parse_jobs(search_text: str) -> list[dict]:
        try:
            data = json.loads(search_text)
        except json.JSONDecodeError:
            return []
        return data.get("jobs", []) if isinstance(data, dict) else data

    @classmethod
    def _best_matches(cls, search_json: str, resume: str, n: int) -> list[dict]:
        jobs = cls._parse_jobs(search_json)
        resume_l = resume.lower()
        scored = sorted(jobs, key=lambda j: -sum(s.lower() in resume_l for s in j.get("skills", [])))
        return scored[:n]

    def _summary(self, done: dict[str, str]) -> str:
        if not done:
            return ("(Demo brain) I can read your resume, search jobs, and save or list applications. "
                    "Try: 'Read my resume, find remote python jobs and save the best two.'")
        lines = ["(Demo brain) Here's what I did:"]
        if "read_resume" in done:
            lines.append("- Read your resume.")
        if "search_jobs" in done:
            jobs = self._parse_jobs(done["search_jobs"])
            if jobs:
                lines.append(f"- Found {len(jobs)} jobs: " + "; ".join(f"{j['title']} at {j['company']}" for j in jobs))
            else:
                lines.append("- " + done["search_jobs"])
        if "save_application" in done:
            lines.append("- Save results: " + re.sub(r"\s+", " ", done["save_application"]))
        if "list_applications" in done:
            lines.append("- Your applications:\n" + done["list_applications"])
        return "\n".join(lines)


def make_brain(choice: str = "auto"):
    """Pick a brain.

    choice: "claude", "ollama", "demo", or "auto"
    auto = Claude if ANTHROPIC_API_KEY is set, else Ollama if it's running, else demo.
    """
    if choice == "demo":
        return DemoBrain()
    if choice == "ollama":
        return OllamaBrain()
    if choice == "claude" or (choice == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        try:
            return ClaudeBrain()
        except ImportError:
            print("anthropic package not installed (pip install anthropic).")
            if choice == "claude":
                raise
    if choice == "auto":
        brain = OllamaBrain()
        try:
            import urllib.request
            urllib.request.urlopen(brain.host + "/api/tags", timeout=2)  # is Ollama running?
            return brain
        except Exception:
            pass
    return DemoBrain()
