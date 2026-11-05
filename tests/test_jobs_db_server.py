"""Application tracker server tests, exercised through a real in-process MCP client."""

import json

import pytest
from mcp import Client

import servers.jobs_db_server as jobs_db

pytestmark = pytest.mark.anyio


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the server at a fresh, empty database for each test."""
    monkeypatch.setattr(jobs_db, "DB_PATH", tmp_path / "test.db")


async def call(client, tool, args=None) -> str:
    result = await client.call_tool(tool, args or {})
    return result.content[0].text


async def test_lists_three_tools(db):
    async with Client(jobs_db.server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {"save_application", "list_applications", "update_status"}


async def test_read_only_hints_are_correct(db):
    """The client asks the user before running any tool that isn't read-only."""
    async with Client(jobs_db.server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert tools["list_applications"].annotations.read_only_hint is True
    assert tools["save_application"].annotations.read_only_hint is False
    assert tools["update_status"].annotations.read_only_hint is False


async def test_save_then_list(db):
    async with Client(jobs_db.server) as client:
        assert "Saved" in await call(client, "save_application", {"company": "Acme", "title": "Dev"})
        rows = json.loads(await call(client, "list_applications"))
    assert [(r["company"], r["title"], r["status"]) for r in rows] == [("Acme", "Dev", "saved")]


async def test_duplicate_is_not_saved_twice(db):
    async with Client(jobs_db.server) as client:
        await call(client, "save_application", {"company": "Acme", "title": "Dev"})
        assert "Already saved" in await call(client, "save_application", {"company": "Acme", "title": "Dev"})
        assert len(json.loads(await call(client, "list_applications"))) == 1


async def test_update_status(db):
    async with Client(jobs_db.server) as client:
        await call(client, "save_application", {"company": "Acme", "title": "Dev"})
        assert "now 'applied'" in await call(client, "update_status", {"application_id": 1, "status": "applied"})
        assert "Invalid status" in await call(client, "update_status", {"application_id": 1, "status": "hired!"})
        assert "No application" in await call(client, "update_status", {"application_id": 99, "status": "offer"})
        applied = json.loads(await call(client, "list_applications", {"status": "applied"}))
    assert len(applied) == 1


async def test_empty_list_message(db):
    async with Client(jobs_db.server) as client:
        assert await call(client, "list_applications") == "No applications saved yet."


async def test_resource_and_prompt(db):
    async with Client(jobs_db.server) as client:
        await call(client, "save_application", {"company": "Acme", "title": "Dev"})
        res = await client.read_resource("jobs://applications")
        assert json.loads(res.contents[0].text)[0]["company"] == "Acme"
        prompt = await client.get_prompt("cover_letter", {"company": "Acme", "title": "Dev"})
        assert "Acme" in prompt.messages[0].content.text
