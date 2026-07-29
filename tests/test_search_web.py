"""Tests for the cached web search tool.

Two properties carry the weight here. The cache must be a determinism mechanism — a graded
re-run has to replay the earlier run's evidence rather than whatever the web says today — and
every failure must arrive as `ToolInfraError`, because a tool we could not run is our gap and
booking it as the model's would let an outage look like a finding.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent.tools import ToolInfraError, registry
from agent.tools.search_web import (
    SearchResult,
    cache_key,
    read_cache,
    search_web,
    write_cache,
)

HITS = [
    SearchResult(title="First", url="https://a.example", snippet="alpha"),
    SearchResult(title="Second", url="https://b.example", snippet="beta"),
]


def tavily_ok(*titles: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "results": [
                {"title": t, "url": f"https://{t}.example", "content": f"snippet for {t}"}
                for t in titles
            ]
        },
    )


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


# --------------------------------------------------------------------------------------
# Cache key
# --------------------------------------------------------------------------------------


def test_cache_key_is_stable_for_the_same_query():
    assert cache_key("how much water", 5) == cache_key("how much water", 5)


@pytest.mark.parametrize(
    "variant",
    ["How Much Water", "  how much water  ", "how   much\twater"],
)
def test_cache_key_normalises_case_and_whitespace(variant):
    """Trivially different spellings of one query must share an entry, or a re-run pays twice
    for the same evidence and the determinism claim weakens."""
    assert cache_key(variant, 5) == cache_key("how much water", 5)


def test_cache_key_changes_with_max_results():
    """An entry holding three results cannot serve a request for five."""
    assert cache_key("q", 3) != cache_key("q", 5)


def test_cache_key_does_not_depend_on_the_credential(monkeypatch):
    """Rotating a key must not discard the cache, nor put a secret in a filename."""
    first = cache_key("q", 5)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-rotated")
    assert cache_key("q", 5) == first


# --------------------------------------------------------------------------------------
# Read and write
# --------------------------------------------------------------------------------------


def test_a_written_entry_reads_back_identically(tmp_path):
    key = cache_key("q", 5)
    write_cache(key, HITS, tmp_path, query="q")
    assert read_cache(key, tmp_path) == HITS


def test_a_missing_entry_is_a_miss(tmp_path):
    assert read_cache(cache_key("never fetched", 5), tmp_path) is None


def test_a_corrupt_entry_is_a_miss_not_an_error(tmp_path):
    """One bad file should cost a single API call, not abort a run."""
    key = cache_key("q", 5)
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert read_cache(key, tmp_path) is None


def test_an_entry_missing_a_field_is_a_miss(tmp_path):
    key = cache_key("q", 5)
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"results": [{"title": "T", "url": "u"}]}), encoding="utf-8"
    )
    assert read_cache(key, tmp_path) is None


def test_the_entry_records_the_query_and_a_timestamp(tmp_path):
    """The point of recording the query is that a reader can see what was actually asked, so
    it is the raw text rather than the normalised form the key hashes."""
    key = cache_key("  How Much Water ", 5)
    write_cache(key, HITS, tmp_path, query="  How Much Water ")

    entry = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert entry["query"] == "  How Much Water "
    assert entry["fetched_at"]
    assert entry["cache_version"] == 1


def test_writing_leaves_no_temporary_files_behind(tmp_path):
    write_cache(cache_key("q", 5), HITS, tmp_path, query="q")
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


# --------------------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------------------


def test_a_miss_calls_the_provider_and_reports_cached_false(tmp_path, monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return tavily_ok("one", "two")

    monkeypatch.setattr(httpx, "post", _transport(handler))
    result = search_web("hydration guidance", cache_dir=tmp_path)

    assert result["cached"] is False
    assert [hit.title for hit in result["results"]] == ["one", "two"]
    assert len(calls) == 1


def test_a_second_identical_call_is_served_from_disk(tmp_path, monkeypatch):
    """The determinism requirement: a graded re-run replays the first run's evidence."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return tavily_ok("one")

    monkeypatch.setattr(httpx, "post", _transport(handler))
    first = search_web("hydration guidance", cache_dir=tmp_path)
    second = search_web("hydration guidance", cache_dir=tmp_path)

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["results"] == first["results"]
    assert len(calls) == 1, "a cache hit must not touch the network"


def test_the_credential_never_reaches_the_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "post", _transport(lambda r: tavily_ok("one")))
    search_web("q", cache_dir=tmp_path)

    written = "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json"))
    assert "tvly-test" not in written


def test_web_cache_dir_is_read_from_the_environment(tmp_path, monkeypatch):
    """The variable `.env.example` advertises has to actually do something."""
    monkeypatch.setenv("WEB_CACHE_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(httpx, "post", _transport(lambda r: tavily_ok("one")))

    search_web("q")
    assert list((tmp_path / "elsewhere").glob("*.json"))


def test_an_explicit_cache_dir_beats_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_CACHE_DIR", str(tmp_path / "ignored"))
    monkeypatch.setattr(httpx, "post", _transport(lambda r: tavily_ok("one")))

    search_web("q", cache_dir=tmp_path / "chosen")
    assert list((tmp_path / "chosen").glob("*.json"))
    assert not (tmp_path / "ignored").exists()


def test_max_results_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "post", _transport(lambda r: tavily_ok("a", "b", "c", "d")))
    result = search_web("q", max_results=2, cache_dir=tmp_path)
    assert len(result["results"]) == 2


# --------------------------------------------------------------------------------------
# Every failure is ours, not the model's
# --------------------------------------------------------------------------------------


def test_a_missing_credential_is_an_infrastructure_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ToolInfraError, match="TAVILY_API_KEY"):
        search_web("q", cache_dir=tmp_path)


def test_a_blank_credential_counts_as_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "   ")
    with pytest.raises(ToolInfraError, match="TAVILY_API_KEY"):
        search_web("q", cache_dir=tmp_path)


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_a_provider_error_status_is_an_infrastructure_failure(status, tmp_path, monkeypatch):
    """Including 401: a credential we got wrong is still our side of the contract."""
    monkeypatch.setattr(httpx, "post", _transport(lambda r: httpx.Response(status, text="no")))
    with pytest.raises(ToolInfraError, match=str(status)):
        search_web("q", cache_dir=tmp_path)


def test_a_transport_failure_is_an_infrastructure_failure(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(ToolInfraError, match="request failed"):
        search_web("q", cache_dir=tmp_path)


def test_an_unexpected_payload_is_an_infrastructure_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "post", _transport(lambda r: httpx.Response(200, json={"x": 1})))
    with pytest.raises(ToolInfraError, match="unexpected response shape"):
        search_web("q", cache_dir=tmp_path)


def test_a_failed_fetch_writes_nothing(tmp_path, monkeypatch):
    """A cache entry for a failed call would make the failure permanent."""
    monkeypatch.setattr(httpx, "post", _transport(lambda r: httpx.Response(500, text="no")))
    with pytest.raises(ToolInfraError):
        search_web("q", cache_dir=tmp_path)
    assert list(tmp_path.glob("*.json")) == []


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_the_tool_is_registered_and_callable():
    """`registry()` binds the module's four exports, so a rename here must not silently drop
    the tool from the prompt the model is shown."""
    tool = registry()["search_web"]
    assert tool.description
    assert set(tool.schema["properties"]) == {"query", "max_results"}


def test_the_run_level_arguments_are_not_model_facing():
    """`cache_dir` and `timeout_s` are properties of the run. Absent from `schema`, so
    `core.validate_arguments` rejects a model that names one."""
    assert "cache_dir" not in registry()["search_web"].schema["properties"]
    assert "timeout_s" not in registry()["search_web"].schema["properties"]


def _transport(handler):
    """Return a stand-in for `httpx.post` that routes through a mock transport."""

    def post(url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.post(url, **kwargs)

    return post
