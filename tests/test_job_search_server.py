"""Tests for job search: sample data, the real-API adapter, caching and fallbacks."""

import json
import time

import pytest

import servers.job_search_server as js


def test_sample_search_and_remote_filter():
    jobs = js.load_sample()
    assert all("python" in (j["title"] + j["description"] + " ".join(j["skills"])).lower()
               for j in js.search(jobs, "python", remote_only=False))
    assert all(j["remote"] for j in js.search(jobs, "python", remote_only=True))
    assert js.search(jobs, "cobol mainframe", remote_only=False) == []


def test_title_matches_rank_first():
    jobs = [{"title": "Designer", "company": "A", "description": "some python", "skills": [], "remote": True},
            {"title": "Python Developer", "company": "B", "description": "", "skills": [], "remote": True}]
    assert js.search(jobs, "python", remote_only=False)[0]["company"] == "B"


def test_normalize_remotive_cleans_html():
    raw = {"jobs": [{"id": 7, "title": "Backend Engineer", "company_name": "Foo",
                     "description": "<p>Use <b>Python</b> &amp; SQL</p>", "url": "https://x",
                     "publication_date": "2026-11-01T10:00:00", "candidate_required_location": "USA"}]}
    job = js.normalize_remotive(raw)[0]
    assert job["id"] == "R7" and job["company"] == "Foo" and job["remote"] is True
    assert job["description"] == "Use Python & SQL"
    assert job["posted"] == "2026-11-01" and job["source"] == "Remotive"


def test_search_finds_words_beyond_the_short_description(monkeypatch):
    """Regression test: 'python' at the END of a long post must still match,
    and the full search text must never be sent to the AI."""
    long_post = "<p>" + "We build great things. " * 40 + "Stack: Python, Postgres.</p>"
    raw = {"jobs": [{"id": 1, "title": "Backend Engineer", "company_name": "Foo", "description": long_post}]}
    jobs = js.normalize_remotive(raw)
    assert "python" not in jobs[0]["description"].lower()        # cut from the short version...
    assert js.search(jobs, "python", remote_only=True)           # ...but still found
    monkeypatch.setattr(js, "load_jobs", lambda: (jobs, "Remotive (test)"))
    out = json.loads(js.search_jobs("python"))
    assert out["jobs"][0]["title"] == "Backend Engineer"
    assert "_search_text" not in out["jobs"][0]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.delenv("JOBS_SOURCE", raising=False)
    return js.CACHE_FILE


def offline(*_a, **_k):
    raise OSError("no network")


def test_fresh_cache_is_used_without_calling_api(cache, monkeypatch):
    cache.write_text(json.dumps({"fetched_at": time.time(), "jobs": [{"title": "Cached"}]}))
    monkeypatch.setattr(js, "fetch_remotive", lambda: pytest.fail("should not call the API"))
    jobs, source = js.load_jobs()
    assert jobs[0]["title"] == "Cached" and "cached" in source


def test_old_cache_used_when_api_is_down(cache, monkeypatch):
    cache.write_text(json.dumps({"fetched_at": time.time() - 24 * 3600, "jobs": [{"title": "Old"}]}))
    monkeypatch.setattr(js, "fetch_remotive", offline)
    jobs, source = js.load_jobs()
    assert jobs[0]["title"] == "Old" and "old cache" in source


def test_sample_used_when_no_cache_and_api_down(cache, monkeypatch):
    monkeypatch.setattr(js, "fetch_remotive", offline)
    jobs, source = js.load_jobs()
    assert source.startswith("sample data") and len(jobs) == 8


def test_jobs_source_sample_skips_the_api(monkeypatch):
    monkeypatch.setenv("JOBS_SOURCE", "sample")
    monkeypatch.setattr(js, "fetch_remotive", lambda: pytest.fail("should not call the API"))
    assert js.load_jobs()[1] == "sample data"
