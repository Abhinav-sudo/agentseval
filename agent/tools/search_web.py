"""Web search via Tavily, with results cached to disk.

The cache is a determinism requirement, not an optimisation (PROJECT.md): a graded run
must not depend on what the live web returned that afternoon. On a cache hit the tool
returns the stored payload and never touches the network, so re-running an eval reproduces
the earlier run's evidence.

Cache keys are derived from the normalised query, so the same query is stable across runs.
Every call logs whether it was a hit or a miss; a miss during a graded re-run means the
evidence changed and the comparison needs re-basing.

The on-disk shape deliberately mirrors `models.base.ResponseCache`: a readable JSON entry
holding the request beside the response, written to a temporary sibling and renamed into
place so two concurrent workers cannot read a half-written file. The credential is never part
of the key or the entry, so rotating it neither invalidates the cache nor puts a secret on
disk.

`cached` rides back in the result, which is what keeps a replay out of the timing figures.
It needs no filter in `metrics.latency_aggregates`: that function averages over
`MODEL_CALL_ROLES`, and a tool result is logged under `core.ROLE_TOOL`, which is not one of
them — the same role partitioning that keeps guardrail stages out of the candidate's latency.
So a cache hit here can no more inflate measured model speed than a rule-based screen can.

Every failure is `ToolInfraError`, including a missing credential. A tool we could not run is
our gap, not the model's: the loop retries it and, if it persists, excludes the item from
scoring rather than charging the model for our outage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent.tools.errors import ToolInfraError
from agent.trace import utc_now_iso

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_CACHE_DIR = Path(".cache/web")
DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_S = 20.0

#: Bumped when the cached entry shape changes, so stale entries miss instead of
#: deserialising into something this code misreads. Mirrors `base.CACHE_VERSION`.
CACHE_VERSION = 1

_WHITESPACE = re.compile(r"\s+")


def resolve_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Resolve the cache directory: explicit argument, then `WEB_CACHE_DIR`, then default.

    Mirrors `models.base.resolve_cache_dir` so the two caches are configured the same way.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.environ.get("WEB_CACHE_DIR", "").strip() or DEFAULT_CACHE_DIR)


@dataclass(frozen=True)
class SearchResult:
    """One search hit."""

    title: str
    url: str
    snippet: str


def cache_key(query: str, max_results: int) -> str:
    """Return the cache key for a query: hash of the normalised query and result count.

    Normalisation is case-folding plus whitespace collapse, so trivially different spellings
    of one query share an entry. `max_results` is part of the key because an entry holding
    three results cannot serve a request for five.
    """
    normalised = _WHITESPACE.sub(" ", query).strip().casefold()
    payload = {
        "cache_version": CACHE_VERSION,
        "query": normalised,
        "max_results": max_results,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_cache(key: str, cache_dir: Path | str | None = None) -> list[SearchResult] | None:
    """Return cached results for `key`, or None on a miss.

    An unreadable or malformed entry is a miss rather than an error, as in
    `ResponseCache.get`: a corrupt file should cost one API call, not abort a run.
    """
    try:
        path = resolve_cache_dir(cache_dir) / f"{key}.json"
        entry = json.loads(path.read_text(encoding="utf-8"))
        return [
            SearchResult(title=hit["title"], url=hit["url"], snippet=hit["snippet"])
            for hit in entry["results"]
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def write_cache(
    key: str,
    results: list[SearchResult],
    cache_dir: Path | str | None = None,
    *,
    query: str | None = None,
) -> None:
    """Persist results, including the query and fetch timestamp for auditability.

    `query` is the raw text rather than the normalised form the key hashes, since the point of
    recording it is to let a reader see what was actually asked. Written via a temporary
    sibling and an atomic rename, so a concurrent reader sees either the previous entry or the
    complete new one.
    """
    directory = resolve_cache_dir(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    entry = {
        "cache_version": CACHE_VERSION,
        "key": key,
        "query": query,
        "fetched_at": utc_now_iso(),
        "results": [
            {"title": hit.title, "url": hit.url, "snippet": hit.snippet} for hit in results
        ],
    }
    payload = json.dumps(entry, indent=2, ensure_ascii=False) + "\n"

    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(directory), suffix=".web.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, directory / f"{key}.json")
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _fetch(query: str, max_results: int, timeout_s: float) -> list[SearchResult]:
    """Call Tavily once and map its payload onto `SearchResult`.

    Raises:
        ToolInfraError: no credential, the request failed, the provider returned an error
            status, or the payload was not the documented shape. All of these are our side of
            the contract, so none is charged to the model.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ToolInfraError(f"{name}: TAVILY_API_KEY is not set")

    try:
        response = httpx.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": query, "max_results": max_results},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise ToolInfraError(f"{name}: request failed: {exc}") from exc

    if response.status_code >= 300:
        raise ToolInfraError(
            f"{name}: provider returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        hits = response.json()["results"]
        return [
            SearchResult(
                title=str(hit.get("title", "")),
                url=str(hit.get("url", "")),
                snippet=str(hit.get("content", "")),
            )
            for hit in hits[:max_results]
        ]
    except (ValueError, KeyError, TypeError) as exc:
        raise ToolInfraError(f"{name}: unexpected response shape: {exc}") from exc


def search_web(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    cache_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Tool entry point: cached web search.

    Checks the cache first and only calls the provider on a miss, then stores the result.

    Args:
        query: Search query supplied by the agent.
        max_results: Maximum number of results to return.
        cache_dir: Where entries live; defaults to `WEB_CACHE_DIR` then `.cache/web`.
            Keyword-only, and absent from `schema`, so it is a property of the run rather
            than something the model can redirect.
        timeout_s: Per-request timeout.

    Returns:
        A JSON-serialisable dict including a `cached` flag, so the trace records whether
        this call hit the network.

    Raises:
        ToolInfraError: the provider was unreachable, refused the request, or no credential
            was configured. Never `ToolInputError`: nothing the model wrote causes these.
    """
    key = cache_key(query, max_results)
    hits = read_cache(key, cache_dir)
    cached = hits is not None
    if hits is None:
        hits = _fetch(query, max_results, timeout_s)
        write_cache(key, hits, cache_dir, query=query)
    logger.info("%s: %s for %r (%d results)", name, "hit" if cached else "miss", query, len(hits))
    return {"query": query, "cached": cached, "results": hits}


name = "search_web"
description = "Search the web for current information not present in the knowledge base."
schema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query. Keep it specific; results are cached per query.",
        },
        "max_results": {
            "type": "integer",
            "description": f"How many results to return (default {DEFAULT_MAX_RESULTS}).",
            "minimum": 1,
            "maximum": 10,
        },
    },
    "required": ["query"],
}
