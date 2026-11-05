"""
Application tracker MCP server (SQLite).

Exposes all three MCP server primitives:
    tools     save_application, list_applications, update_status
    resource  jobs://applications   (all applications as JSON)
    prompt    cover_letter          (reusable cover-letter template)

Tool annotations declare which tools are read-only; the client uses them to require
human approval for writes. Set JOBS_DB_PATH to use a different database file.

Run standalone (stdio transport):  python servers/jobs_db_server.py
"""

import json
import os
import sqlite3
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

# ---------------------------------------------------------------------------
# METADATA: the server's name. Clients see this during the handshake.
# ---------------------------------------------------------------------------
server = MCPServer("jobs-db")

# JOBS_DB_PATH lets the benchmark use its own throwaway database.
DB_PATH = Path(os.environ.get("JOBS_DB_PATH") or Path(__file__).resolve().parent.parent / "data" / "jobs.db")
VALID_STATUSES = ("saved", "applied", "interview", "offer", "rejected")


def get_db() -> sqlite3.Connection:
    """Open the database and create the table the first time."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute(
        """CREATE TABLE IF NOT EXISTS applications (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               company TEXT NOT NULL,
               title TEXT NOT NULL,
               url TEXT,
               status TEXT NOT NULL DEFAULT 'saved',
               notes TEXT,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    return db


# ---------------------------------------------------------------------------
# TOOLS
# The function's docstring becomes the tool DESCRIPTION. The LLM reads that
# description to decide when to use the tool, so write it clearly and honestly.
#
# Annotations are hints for the client. Our client uses read_only_hint to
# decide whether to ask the user "Are you sure?" before running a tool.
# ---------------------------------------------------------------------------
@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
def save_application(company: str, title: str, url: str = "", notes: str = "") -> str:
    """Save a job the user wants to apply to. Use only after the user asks to save it."""
    with get_db() as db:
        dup = db.execute(
            "SELECT id FROM applications WHERE company = ? AND title = ?", (company, title)
        ).fetchone()
        if dup:
            return f"Already saved: {title} at {company} (id {dup['id']})."
        cur = db.execute(
            "INSERT INTO applications (company, title, url, notes) VALUES (?, ?, ?, ?)",
            (company, title, url, notes),
        )
        return f"Saved: {title} at {company} (id {cur.lastrowid})."


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def list_applications(status: str = "") -> str:
    """List saved job applications. Optionally filter by status: saved, applied, interview, offer, rejected."""
    with get_db() as db:
        if status:
            rows = db.execute(
                "SELECT * FROM applications WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        else:
            rows = db.execute("SELECT * FROM applications ORDER BY id").fetchall()
    if not rows:
        return "No applications saved yet."
    return json.dumps([dict(r) for r in rows], indent=2)


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
def update_status(application_id: int, status: str) -> str:
    """Change the status of a saved application (saved, applied, interview, offer, rejected)."""
    if status not in VALID_STATUSES:
        return f"Invalid status '{status}'. Use one of: {', '.join(VALID_STATUSES)}."
    with get_db() as db:
        cur = db.execute(
            "UPDATE applications SET status = ? WHERE id = ?", (status, application_id)
        )
    if cur.rowcount == 0:
        return f"No application with id {application_id}."
    return f"Application {application_id} is now '{status}'."


# ---------------------------------------------------------------------------
# RESOURCE: read-only data, addressed by a URI (like a web address).
# ---------------------------------------------------------------------------
@server.resource("jobs://applications", mime_type="application/json")
def all_applications() -> str:
    """Every saved application as JSON."""
    with get_db() as db:
        rows = db.execute("SELECT * FROM applications ORDER BY id").fetchall()
    return json.dumps([dict(r) for r in rows], indent=2)


# ---------------------------------------------------------------------------
# PROMPT: a reusable template the user (or app) can pick, like a slash command.
# ---------------------------------------------------------------------------
@server.prompt()
def cover_letter(company: str, title: str) -> str:
    """Draft a short cover letter for a saved job."""
    return (
        f"Write a short, specific cover letter for the {title} role at {company}. "
        "First read my resume with the resume tool, then mention two concrete "
        "skills from it that match this role. Keep it under 200 words."
    )


if __name__ == "__main__":
    server.run()  # default transport is stdio: talk over stdin/stdout
