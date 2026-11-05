"""
Safety layer implementing mitigations from Hou et al. (2025), Sections 5 and 6.3.

scan_tools()         detects tool-description poisoning and tool-name collisions,
                     optionally quarantining offending tools
needs_confirmation() human-in-the-loop approval for any non-read-only tool
ToolLog              append-only JSONL audit log of every tool call
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from .hub import HubTool

# Phrases that try to manipulate the AI instead of describing the tool.
SUSPICIOUS_PATTERNS = [
    r"prefer (using )?this tool",
    r"use this tool first",
    r"always use this",
    r"this tool should be prioriti[sz]ed",
    r"ignore (all |any )?(previous|prior|other) instructions",
    r"do not tell the user",
    r"include (the )?(full )?(resume|password|api key|token|secret)",
    r"<important>|\[system\]",
]


def scan_tools(tools: list[HubTool], quarantine: bool = True) -> list[str]:
    """Check every tool. Returns human-readable warnings. Flags tools in place."""
    warnings: list[str] = []

    # Toolflow hijacking: manipulative text in the description
    for t in tools:
        desc = (t.tool.description or "").lower()
        # Report the actual words found, not the pattern, so a person can read it.
        hits = [m.group(0) for p in SUSPICIOUS_PATTERNS if (m := re.search(p, desc))]
        if hits:
            found = ", ".join(f'"{h}"' for h in hits)
            msg = f"{t.full_name}: its description tries to steer the AI ({found})"
            t.warnings.append(msg)
            t.quarantined = quarantine
            warnings.append(msg + (" -> QUARANTINED" if quarantine else ""))

    # Tool name conflicts: the same short name offered by different servers
    by_name: dict[str, list[HubTool]] = defaultdict(list)
    for t in tools:
        by_name[t.tool.name].append(t)
    for name, owners in by_name.items():
        if len(owners) > 1:
            servers = ", ".join(o.server for o in owners)
            warnings.append(f"Tool name conflict: '{name}' is offered by: {servers}")

    return warnings


def needs_confirmation(tool: HubTool, enabled: bool = True) -> bool:
    return enabled and not tool.read_only


class ToolLog:
    """Appends one JSON line per tool call to logs/tool_calls.jsonl."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []

    def record(self, *, tool: str, args: dict, approved: bool, result: str,
               is_error: bool, seconds: float) -> dict:
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool,
            "args": args,
            "approved": approved,
            "is_error": is_error,
            "seconds": round(seconds, 3),
            "result_preview": result[:300],
        }
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry
