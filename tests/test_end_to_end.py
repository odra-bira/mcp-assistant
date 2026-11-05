"""End to end: real servers as subprocesses, the real hub and agent, the demo brain."""

import json
import sqlite3

import pytest

from assistant.agent import Agent
from assistant.brains import DemoBrain
from assistant.hub import ServerHub
from assistant.safety import ToolLog, scan_tools
from tests.conftest import ROOT

pytestmark = pytest.mark.anyio


def write_config(tmp_path, evil=False):
    config = json.loads((ROOT / "assistant" / "config.json").read_text())
    for s in config["servers"]:
        s["enabled"] = s["name"] != "evil" or evil
        if s["name"] == "jobsdb":
            s["env"] = {"JOBS_DB_PATH": str(tmp_path / "e2e.db")}
        if s["name"] == "jobsearch":
            s["env"] = {"JOBS_SOURCE": "sample"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


async def approve(tool, args, call_id):
    return True


async def test_full_request_saves_two_jobs(tmp_path):
    async with ServerHub(write_config(tmp_path)) as hub:
        assert set(hub.clients) == {"resume", "jobsearch", "jobsdb"}
        agent = Agent(hub, DemoBrain(), ToolLog(tmp_path / "log.jsonl"), approve)
        await agent.ask("Read my resume, find remote python jobs that match it, and save the best two.")
    rows = sqlite3.connect(tmp_path / "e2e.db").execute("SELECT company FROM applications").fetchall()
    assert len(rows) == 2


async def test_scanner_blocks_the_evil_server(tmp_path):
    async with ServerHub(write_config(tmp_path, evil=True)) as hub:
        scan_tools(list(hub.tools.values()), quarantine=True)
        agent = Agent(hub, DemoBrain(), ToolLog(tmp_path / "log.jsonl"), approve)
        await agent.ask("find remote python jobs and save the best two")
    used = [e["tool"] for e in agent.log.entries]
    assert "evil__save_application" not in used
    assert "jobsdb__save_application" in used
