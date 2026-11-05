"""
Job search MCP server

Two data sources, same tool:
  JOBS_SOURCE=remotive  -> real remote jobs from Remotive's free public API (default)
  JOBS_SOURCE=sample    -> the fixed sample file (used by the benchmark and tests,
                           so results are repeatable)

Remotive asks API users to call at most ~4 times a day. So we download ALL
software jobs in one request, CACHE them for 6 hours in data/remotive_cache.json,
and search the cached copy. If the API can't be reached, we fall back to the cache
(even if old), then to the sample file. The AI never sees any of this: the tool's
name and input schema are identical regardless of the backing data source.

Run as a LOCAL server (stdio, the default):
    python servers/job_search_server.py
Run as a REMOTE server over HTTP (see README, "Remote server"):
    MCP_AUTH_TOKEN=some-secret python servers/job_search_server.py --http --port 8765
"""

import html
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("job-search")
DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLE_FILE = DATA / "sample_jobs.json"
# v2: the cache now stores each job's full text for searching (v1 only kept 300 characters).
CACHE_FILE = DATA / "remotive_cache_v2.json"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs?category=software-dev"
CACHE_HOURS = 6


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
def load_sample() -> list[dict]:
    jobs = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    for j in jobs:
        j.setdefault("source", "sample data")
    return jobs


def clean_html(text: str, limit: int | None = 300) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if limit is None:
        return text
    return text[:limit] + ("..." if len(text) > limit else "")


def normalize_remotive(raw: dict) -> list[dict]:
    """Turn Remotive's format into ours, so the AI always sees the same shape."""
    jobs = []
    for j in raw.get("jobs", []):
        jobs.append({
            "id": f"R{j.get('id')}",
            "title": j.get("title", ""),
            "company": j.get("company_name", ""),
            "remote": True,
            "location": j.get("candidate_required_location", ""),
            "salary": j.get("salary", ""),
            "skills": [],  # Remotive doesn't list skills; the description has them
            # Short version for the AI to read; keeps responses small.
            "description": clean_html(j.get("description", "")),
            "url": j.get("url", ""),
            "posted": (j.get("publication_date") or "")[:10],
            "source": "Remotive",
            # Full text, used ONLY for searching. "Python" is often far down a job post,
            # so searching the short version misses most matches. Never sent to the AI.
            "_search_text": " ".join([j.get("title", ""), j.get("company_name", ""),
                                      clean_html(j.get("description", ""), limit=None),
                                      " ".join(j.get("tags") or [])]).lower(),
        })
    return jobs


def fetch_remotive() -> list[dict]:
    req = urllib.request.Request(REMOTIVE_URL, headers={"User-Agent": "mcp-assistant"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read())
    jobs = normalize_remotive(raw)
    CACHE_FILE.write_text(json.dumps({"fetched_at": time.time(), "jobs": jobs}), encoding="utf-8")
    return jobs


def load_remotive() -> tuple[list[dict], str]:
    """Returns (jobs, note about where they came from)."""
    cache = None
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        age_hours = (time.time() - cache["fetched_at"]) / 3600
        if age_hours < CACHE_HOURS:
            return cache["jobs"], f"Remotive (cached {age_hours:.1f}h ago)"
    try:
        return fetch_remotive(), "Remotive (just updated)"
    except Exception as exc:  # offline, blocked, API changed...
        if cache:
            return cache["jobs"], f"Remotive (old cache; update failed: {type(exc).__name__})"
        return load_sample(), f"sample data (Remotive unavailable: {type(exc).__name__})"


def load_jobs() -> tuple[list[dict], str]:
    if os.environ.get("JOBS_SOURCE", "remotive").lower() == "sample":
        return load_sample(), "sample data"
    return load_remotive()


def search(jobs: list[dict], query: str, remote_only: bool) -> list[dict]:
    words = [w.lower() for w in query.split() if w.strip()]
    results = []
    for job in jobs:
        text = job.get("_search_text") or " ".join(
            [job["title"], job["company"], job.get("description", ""), " ".join(job.get("skills", []))]).lower()
        if remote_only and not job.get("remote"):
            continue
        if all(w in text for w in words):
            results.append(job)
    # Title matches first: "python developer" beats a description that mentions python once.
    results.sort(key=lambda j: -sum(w in j["title"].lower() for w in words))
    return results


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------
@server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
def search_jobs(query: str, remote_only: bool = False, limit: int = 5) -> str:
    """Search job listings by keywords (for example 'python' or 'data engineer'). Returns title, company, description and URL."""
    jobs, source = load_jobs()
    results = search(jobs, query, remote_only)[: max(1, min(limit, 20))]
    if not results:
        return f"No jobs found for '{query}' (source: {source})."
    public = [{k: v for k, v in j.items() if not k.startswith("_")} for j in results]
    return json.dumps({"source": source, "jobs": public}, indent=2)


# ---------------------------------------------------------------------------
# Remote mode: HTTP + a bearer token
# ---------------------------------------------------------------------------
def run_http(port: int) -> None:
    import uvicorn

    from servers.http_auth import BearerTokenMiddleware

    token = os.environ.get("MCP_AUTH_TOKEN")
    if not token:
        sys.exit("Set MCP_AUTH_TOKEN to a secret value before running in --http mode.")
    app = BearerTokenMiddleware(server.streamable_http_app(), token)
    print(f"Job search MCP server on http://127.0.0.1:{port}/mcp (token required)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    if "--http" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8765
        run_http(port)
    else:
        server.run()
