"""One interface for every model, and the machinery all of them share.

The frontier agent, the OSS agent, and the judge are reached through `ModelAdapter`. The
interface deliberately offers *no* way to pass native tool or function definitions to a
provider: tool calling is a prompt-based JSON protocol (see `agent.prompts`), identical for
both agents, because giving the frontier model native structured output while the OSS model
has JSON parsed out of prose would confound model quality with harness quality. See
PROJECT.md.

Everything that could differ between the two arms lives here rather than in the individual
adapters — the retry policy, the response cache, timing, cost, and error mapping — so that
"both agents ran through the same harness" is a property of the code rather than a promise.
A subclass supplies only the wire format of its provider: endpoint, headers, request body,
and response parsing.

Requests go out over `httpx` directly rather than through provider SDKs. The SDKs retry
internally on their own schedules, which would mean the arms had different retry semantics
unless each was individually disabled; one transport keeps that impossible and gives tests a
single mocking seam.
"""

from __future__ import annotations

import abc
import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import httpx
from dotenv import find_dotenv, load_dotenv

ChatMessage = dict[str, str]

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_CACHE_DIR = Path(".cache/models")

#: Bumped when the cached payload shape changes, so stale entries miss instead of
#: deserialising into something the current code misreads. 2 added `finish_reason`;
#: 3 added `reasoning_tokens`, whose absence would otherwise read as "no thinking" on an
#: entry written when we were not counting it — the one reading that is never safe here,
#: since it is what understated the frontier arm's cost in the first place.
CACHE_VERSION = 3

#: Request-body keys that would hand a provider native tool calling. Forbidden for every
#: adapter (PROJECT.md); `tests/test_models.py` asserts no adapter emits them.
FORBIDDEN_BODY_KEYS = frozenset({"tools", "functions", "tool_choice", "function_call"})


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class ModelError(RuntimeError):
    """A provider call failed. Never swallow this; log it into the trace."""


class ConfigError(ModelError):
    """The adapter is misconfigured — missing credentials, unknown provider."""


class RequestError(ModelError):
    """The provider rejected the request (4xx). Not retried: it will fail again."""


class RateLimitError(ModelError):
    """Rate limited (429). Retried with backoff."""


class ServerError(ModelError):
    """The provider failed (5xx) or the connection broke. Retried with backoff."""


# --------------------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------------------


class FinishReason(StrEnum):
    """Why a provider stopped generating, normalised across wire formats.

    Providers name these differently — Anthropic's `stop_reason`, OpenAI's `finish_reason` —
    and one enum here is what lets `agent.core` tell a model that broke the protocol from one
    that was cut off mid-object by our own `max_tokens`. Those are opposite findings: the
    first is a model failure, the second is a harness failure, and a loop that could not
    distinguish them would report ours as theirs.

    `UNKNOWN` is the value for a provider that said nothing, and is deliberately not folded
    into `COMPLETE`: assuming a response finished cleanly is exactly the assumption that
    would misattribute a truncation.
    """

    COMPLETE = "complete"
    LENGTH = "length"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"
    UNKNOWN = "unknown"

    @classmethod
    def normalise(cls, raw: object, mapping: Mapping[str, FinishReason]) -> FinishReason:
        """Map one provider's stop value onto this enum.

        An unrecognised value becomes `OTHER` rather than raising: a provider adding a new
        stop reason should not fail a run, and the raw payload is on `ModelResponse.raw` for
        whoever investigates.
        """
        if raw is None:
            return cls.UNKNOWN
        return mapping.get(str(raw), cls.OTHER)


@dataclass
class ModelResponse:
    """The result of one model call.

    `text` is the raw assistant content, unparsed. Extracting a tool call from it is the job
    of `agent.core`, not of an adapter — that keeps parsing uniform across providers.

    Attributes:
        latency_ms: Wall-clock time for the call that produced this text. On a cache hit
            this is the latency of the *original* call, replayed.
        usd_cost: Computed from `PRICING`, or None when the model is unpriced or the
            provider reported no token counts.
        reasoning_tokens: Output tokens the provider billed but left out of
            `completion_tokens` — see `derive_reasoning_tokens`. Bill from
            `billed_completion_tokens`, never from `completion_tokens` alone.
        raw: The provider's full JSON response, kept for auditing.
        cached: True when this came from the on-disk cache. Anything that aggregates
            latency or cost must account for this: a cache hit is a replay, not a fresh
            measurement of the model.
        finish_reason: Why generation stopped, normalised. Read by `agent.core` before it
            classifies malformed output, so a response cut off at `max_tokens` is booked
            against the harness rather than against the model's protocol compliance.
    """

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    usd_cost: float | None
    raw: dict[str, Any]
    reasoning_tokens: int | None = None
    cached: bool = False
    finish_reason: FinishReason = FinishReason.UNKNOWN

    @property
    def truncated(self) -> bool:
        """True when the provider stopped because it hit the token ceiling we set."""
        return self.finish_reason is FinishReason.LENGTH

    @property
    def billed_completion_tokens(self) -> int | None:
        """Output tokens the provider charged for: visible plus reasoning.

        The figure every cost and tokens-per-second calculation wants. `completion_tokens`
        is the *visible* reply on a provider that meters thinking separately, so dividing
        latency by it reports a speed the model never achieved.
        """
        if self.completion_tokens is None:
            return None
        return self.completion_tokens + (self.reasoning_tokens or 0)


class ParsedPayload(NamedTuple):
    """What an adapter reads out of a provider's response body.

    A named tuple rather than four positional values, because `prompt_tokens` and
    `completion_tokens` are the same type and adjacent.

    `reasoning_tokens` defaults to None so that an adapter which has not been taught about
    thinking tokens is *unknown* rather than asserting zero. Every adapter here sets it.
    """

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: FinishReason
    reasoning_tokens: int | None = None


def derive_reasoning_tokens(
    total_tokens: int | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> int | None:
    """Billed output tokens the provider left out of `completion_tokens`.

    Providers disagree about what `completion_tokens` means, and the disagreement is silent.
    Gemini's OpenAI-compatible layer reports the *visible* reply there and folds thinking
    into `total_tokens` only, so `prompt + completion` falls short of `total` by exactly
    `thoughtsTokenCount` — verified against the native endpoint, where the residual and the
    named field agree to the token. OpenAI does the opposite: reasoning is already inside
    `completion_tokens`, and `completion_tokens_details.reasoning_tokens` is a breakdown of
    it rather than an addition to it.

    Taking the residual gets both right, which reading a vendor field would not: it measures
    what is billed but unreported, so a provider that already counts reasoning in
    `completion_tokens` yields 0 here and is not charged twice. That makes
    `completion + reasoning` the billed total under either convention.

    Returns None when the provider reported no `total_tokens` — unknown, not zero, since a
    zero would reinstate exactly the understatement this exists to catch. Adapters for
    providers that publish no total (Anthropic) pass 0 themselves, which is a claim about
    that wire format rather than a guess.

    A negative residual is clamped to 0: it means the provider's own numbers disagree, and
    inventing negative billable tokens would corrupt a run's cost rather than one call's.
    """
    if total_tokens is None or prompt_tokens is None or completion_tokens is None:
        return None
    return max(0, total_tokens - prompt_tokens - completion_tokens)


@runtime_checkable
class ModelAdapter(Protocol):
    """A plain chat-completion endpoint.

    Attributes:
        name: Role in the harness: "frontier", "oss", or "judge".
        model_id: Provider-side model string, recorded in the run manifest.
        family: Model family, used to keep the judge distinct from both agents.
    """

    name: str
    model_id: str
    family: str

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop: list[str] | None = None,
    ) -> ModelResponse:
        """Return one completion for `messages`.

        Raises:
            ModelError: on any provider failure, including retries exhausted.
        """
        ...


# --------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------

#: USD per 1M tokens, as (prompt, completion), keyed by model id prefix so that dated model
#: ids match. The single price table for the project: cost is computed here or not at all.
#: Approximate and provider-published — verify against current pricing before quoting a
#: figure. Unpriced models yield None rather than a misleading zero.
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Standard paid tier. Output includes thinking tokens, which is why a Gemini 3 completion
    # can cost more than its visible reply length suggests — and why output is 5x input here.
    # Verified against ai.google.dev/gemini-api/docs/pricing, 2026-07-29.
    "gemini-3.6-flash": (1.50, 7.50),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "qwen-2.5-7b-instruct": (0.20, 0.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
}


def estimate_usd_cost(
    model_id: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    reasoning_tokens: int | None = None,
) -> float | None:
    """Estimate the USD cost of one call from token counts.

    `reasoning_tokens` is charged at the completion rate, because that is how providers
    charge it: thinking is output. Omitting it understated this project's frontier arm by
    about 40% before it was counted, so it is a parameter rather than an optional refinement.

    Returns None when the model is unpriced or either required token count is missing. Never
    0.0 in those cases: a zero would silently understate the cost of a run rather than
    flagging that it is unknown. A None `reasoning_tokens` is the one exception and is
    treated as 0 — it means the provider folds reasoning into `completion_tokens`, so the
    tokens are already charged on the line above.
    """
    if prompt_tokens is None or completion_tokens is None:
        return None

    matches = [prefix for prefix in PRICING if model_id.startswith(prefix)]
    if not matches:
        return None
    prompt_rate, completion_rate = PRICING[max(matches, key=len)]

    billed_completion = completion_tokens + (reasoning_tokens or 0)
    return (prompt_tokens * prompt_rate + billed_completion * completion_rate) / 1_000_000


# --------------------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------------------


def is_retryable(status: int) -> bool:
    """Return True for statuses worth trying again: 429 and any 5xx.

    Everything else — 400, 401, 403, 404 — is a fact about the request, not a transient
    condition, and retrying only delays the error by a minute while burying its cause.
    """
    return status == 429 or status >= 500


@dataclass
class RetryPolicy:
    """Exponential backoff for transient provider failures.

    Only 429, 5xx, and connection failures are retried. A 400 or 401 is not: the request
    is wrong or the key is, and five more attempts would waste a minute to reach the same
    conclusion while hiding the real error behind a timeout.

    Attributes:
        max_retries: Retries *after* the first attempt, so 5 means at most 6 calls.
        jitter: Randomises each delay across [0, delay]. Two arms rate-limited at the same
            moment would otherwise retry in lockstep and collide again.
        sleep: Injectable so tests exercise the schedule without waiting.
    """

    max_retries: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: bool = True
    sleep: Callable[[float], None] = time.sleep

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Return the delay before `attempt` (1-based retry number).

        A provider-supplied `Retry-After` wins over the computed backoff: it is the only
        party that knows when the limit actually resets.
        """
        if retry_after is not None:
            return min(retry_after, self.max_delay_s)
        delay = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return random.uniform(0, delay) if self.jitter else delay


def parse_retry_after(headers: httpx.Headers) -> float | None:
    """Return the `Retry-After` delay in seconds, or None if absent or unparseable."""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        # The HTTP-date form is legal but rare from these providers; fall back to backoff.
        return None


# --------------------------------------------------------------------------------------
# Response cache
# --------------------------------------------------------------------------------------


def resolve_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Resolve the cache directory: explicit argument, then `MODEL_CACHE_DIR`, then default."""
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.environ.get("MODEL_CACHE_DIR", "").strip() or DEFAULT_CACHE_DIR)


def cache_enabled(no_cache: bool | None = None) -> bool:
    """Resolve whether the response cache is active.

    An explicit `no_cache` argument (from `--no-cache`) wins; otherwise
    `AGENTSEVAL_NO_CACHE` decides, defaulting to caching on.
    """
    if no_cache is not None:
        return not no_cache
    raw = os.environ.get("AGENTSEVAL_NO_CACHE", "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def add_cache_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add `--no-cache` to an `argparse` parser.

    Defined once so every entry point spells the escape hatch the same way. Callers pass
    the parsed value straight through: `cache_enabled(args.no_cache)`.

    The default is `None` rather than `False`, because `cache_enabled` reads `None` as "no flag
    was given, ask the environment" and a `False` as "the caller asked for caching". With
    `store_true`'s usual default the env var documented in `.env.example` could never be
    reached from a CLI: absence of the flag would have looked like an explicit request.
    """
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=None,
        help="Bypass the on-disk model response cache and call providers directly.",
    )
    return parser


class ResponseCache:
    """On-disk cache of provider responses, keyed by the full request.

    Evals get re-run many times over an unchanged dataset, and a cached run is both free
    and exactly reproducible — the same reason web search results are cached (PROJECT.md).

    The key covers everything that changes the output: model, messages, temperature,
    max_tokens, and stop sequences. It deliberately excludes the API key, so rotating
    credentials does not throw away the cache and no secret reaches the filesystem.

    Entries are readable JSON holding the request alongside the response, so a cache hit
    can be audited rather than taken on trust.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = resolve_cache_dir(cache_dir)

    @staticmethod
    def key(
        model_id: str,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> str:
        """Return the cache key for one request."""
        payload = {
            "cache_version": CACHE_VERSION,
            "model_id": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        """Return the cached response for `key`, or None on a miss.

        An unreadable or malformed entry is a miss, not an error: a corrupt cache file
        should cost one API call, not abort a run.

        `finish_reason` is read strictly, like every other field: entries written before it
        existed are unreachable anyway, since `CACHE_VERSION` is part of the key.
        """
        path = self.path_for(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            response = entry["response"]
            return ModelResponse(
                text=response["text"],
                prompt_tokens=response["prompt_tokens"],
                completion_tokens=response["completion_tokens"],
                latency_ms=response["latency_ms"],
                usd_cost=response["usd_cost"],
                raw=response["raw"],
                reasoning_tokens=response["reasoning_tokens"],
                cached=True,
                finish_reason=FinishReason(response["finish_reason"]),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def put(self, key: str, request: dict[str, Any], response: ModelResponse) -> Path:
        """Store `response` under `key`, echoing the request for auditability.

        Written to a temporary sibling and renamed into place, as the retrieval index and
        the run manifest are. Two workers evaluating a dataset concurrently can reach the
        same key — identical messages at temperature 0 are exactly what the cache is for —
        and a plain write would let one read the other's half-written file. A rename is
        atomic, so a reader sees either the previous entry or the complete new one.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        entry = {
            "cache_version": CACHE_VERSION,
            "key": key,
            "request": request,
            "response": {
                "text": response.text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
                "usd_cost": response.usd_cost,
                "raw": response.raw,
                "reasoning_tokens": response.reasoning_tokens,
                "finish_reason": str(response.finish_reason),
            },
        }
        payload = json.dumps(entry, indent=2, default=str) + "\n"

        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(self.cache_dir), suffix=".cache.tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path


# --------------------------------------------------------------------------------------
# Shared call pipeline
# --------------------------------------------------------------------------------------


class ChatAdapter(abc.ABC):
    """The one call pipeline: cache lookup, retries, timing, parsing, cost, cache write.

    Subclasses implement only their provider's wire format. Nothing in this class may
    branch on `self.name`; a frontier-only path would make the harness a variable and void
    the comparison (PROJECT.md).
    """

    name: str = ""
    family: str = ""
    provider: str = ""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        http_client: httpx.Client | None = None,
        retry_policy: RetryPolicy | None = None,
        no_cache: bool | None = None,
        cache_dir: Path | str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._client = http_client or httpx.Client(timeout=timeout_s)
        self._owns_client = http_client is None
        self.retry_policy = retry_policy or RetryPolicy()
        self.use_cache = cache_enabled(no_cache)
        self.cache = ResponseCache(cache_dir)

    # -- provider hooks ----------------------------------------------------------------

    @abc.abstractmethod
    def _endpoint(self) -> str:
        """Return the full URL to POST to."""

    @abc.abstractmethod
    def _headers(self) -> dict[str, str]:
        """Return request headers, including authentication."""

    @abc.abstractmethod
    def _body(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        """Return the JSON request body.

        Must never include a native tool-calling key (`FORBIDDEN_BODY_KEYS`).
        """

    @abc.abstractmethod
    def _parse(self, payload: dict[str, Any]) -> ParsedPayload:
        """Extract the fields `generate` needs from a provider response.

        Mapping the provider's stop value onto `FinishReason` is part of this: it is the one
        place that knows the wire format, and every caller above it works in the enum.
        """

    # -- public API --------------------------------------------------------------------

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop: list[str] | None = None,
    ) -> ModelResponse:
        """Return one completion, from cache when available.

        Raises:
            ModelError: the provider rejected the request, or retries were exhausted.
        """
        key = ResponseCache.key(
            self.model_id,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        if self.use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        body = self._body(messages, temperature, max_tokens, stop)
        payload, latency_ms = self._post_with_retries(body)
        parsed = self._parse(payload)

        response = ModelResponse(
            text=parsed.text,
            prompt_tokens=parsed.prompt_tokens,
            completion_tokens=parsed.completion_tokens,
            latency_ms=latency_ms,
            usd_cost=estimate_usd_cost(
                self.model_id,
                parsed.prompt_tokens,
                parsed.completion_tokens,
                parsed.reasoning_tokens,
            ),
            raw=payload,
            reasoning_tokens=parsed.reasoning_tokens,
            cached=False,
            finish_reason=parsed.finish_reason,
        )

        if self.use_cache:
            self.cache.put(key, {"endpoint": self._endpoint(), "body": body}, response)
        return response

    def manifest(self) -> dict[str, Any]:
        """Describe this adapter for a run manifest."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "family": self.family,
            "provider": self.provider,
        }

    def close(self) -> None:
        """Close the HTTP client, if this adapter created it."""
        if self._owns_client:
            self._client.close()

    # -- transport ---------------------------------------------------------------------

    def _post_with_retries(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """POST `body`, retrying transient failures, and return `(payload, latency_ms)`.

        Latency covers only the attempt that succeeded, not the backoff sleeps: it is meant
        to describe how long the model took, not how long the provider was unavailable.
        """
        policy = self.retry_policy
        last_error: ModelError = ServerError(f"{self.name}: no attempt was made")

        for attempt in range(policy.max_retries + 1):
            retry_after: float | None = None
            started = time.perf_counter()
            try:
                http_response = self._client.post(
                    self._endpoint(), headers=self._headers(), json=body
                )
            except httpx.HTTPError as exc:
                last_error = ServerError(f"{self.name}: request to {self.model_id} failed: {exc}")
            else:
                latency_ms = (time.perf_counter() - started) * 1000
                status = http_response.status_code
                if status < 300:
                    return self._decode(http_response), latency_ms
                if not is_retryable(status):
                    raise self._error_for(http_response)
                last_error = self._error_for(http_response)
                retry_after = parse_retry_after(http_response.headers)

            if attempt == policy.max_retries:
                break
            policy.sleep(policy.delay_for(attempt + 1, retry_after))

        raise ModelError(
            f"{self.name}: giving up on {self.model_id} after "
            f"{policy.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _decode(self, http_response: httpx.Response) -> dict[str, Any]:
        try:
            payload = http_response.json()
        except ValueError as exc:
            raise ModelError(f"{self.name}: {self.model_id} returned non-JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ModelError(f"{self.name}: {self.model_id} returned {type(payload).__name__}")
        return payload

    def _error_for(self, http_response: httpx.Response) -> ModelError:
        """Map an HTTP failure onto the error hierarchy.

        The body is truncated into the message because provider errors carry the actionable
        detail, and it must reach the trace rather than being reduced to a status code.
        """
        status = http_response.status_code
        detail = http_response.text[:500]
        message = f"{self.name}: {self.model_id} returned HTTP {status}: {detail}"
        if status == 429:
            return RateLimitError(message)
        if status >= 500:
            return ServerError(message)
        return RequestError(message)


#: OpenAI's `choices[0].finish_reason` vocabulary. `tool_calls` and `function_call` are
#: absent deliberately: the body never requests native tool calling, so seeing one would be a
#: provider surprise worth surfacing as `OTHER` rather than quietly accepting.
OPENAI_FINISH_REASONS: Mapping[str, FinishReason] = {
    "stop": FinishReason.COMPLETE,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAICompatibleAdapter(ChatAdapter):
    """Adapter for the OpenAI chat-completions wire format.

    Shared by the OSS agent (Groq/Together) and the judge (OpenAI), which speak the same
    protocol at different base URLs. One implementation means the two cannot drift.
    """

    base_url: str = ""

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _body(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        """Build a plain chat-completions body.

        No `tools`, `functions`, `tool_choice`, or `response_format`: tool calling is the
        prompt-based JSON protocol, identically for every model (PROJECT.md).
        """
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            body["stop"] = stop
        return body

    def _parse(self, payload: dict[str, Any]) -> ParsedPayload:
        try:
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(
                f"{self.name}: unexpected response shape from {self.model_id}: {exc}"
            ) from exc
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return ParsedPayload(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=FinishReason.normalise(
                choice.get("finish_reason"), OPENAI_FINISH_REASONS
            ),
            reasoning_tokens=derive_reasoning_tokens(
                usage.get("total_tokens"), prompt_tokens, completion_tokens
            ),
        )


# --------------------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------------------


def load_env() -> Path | None:
    """Load `.env` into the process environment, returning the file read or `None`.

    Called first by every entry point that resolves configuration from the environment —
    `agentseval-run`, `agentseval-judge`, `agentseval-validate-judge`, and `app.py` — so a
    credential in `.env` reaches a CLI exactly as it reaches the chat surface. It is called
    there rather than at import time because a library that mutates the environment when it is
    imported is a library no test can isolate. An entry point that reads no environment
    variable does not call it, which is why `agentseval-report` and `agentseval-index` do not.

    Variables already in the environment win, since an exported variable is a deliberate
    override for one command whereas `.env` is the project default. That direction is what
    makes `AGENTSEVAL_NO_CACHE=1 agentseval-run ...` mean what it says.

    The search walks up from the working directory, so a CLI invoked from a subdirectory finds
    the project's file. A missing `.env` is not an error — CI and shell profiles set the
    environment directly — and `require_env` is what reports an absent credential, naming the
    variable rather than the file it was not in.
    """
    found = find_dotenv(usecwd=True)
    if not found:
        return None
    load_dotenv(found, override=False)
    return Path(found)


def require_env(name: str) -> str:
    """Return environment variable `name`, or raise.

    The error names the variable and never echoes a value, so a stack trace in a log cannot
    leak a credential.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


def load_agent_model(
    which: Literal["frontier", "oss"],
    **kwargs: Any,
) -> ModelAdapter:
    """Build the frontier or OSS agent adapter from environment configuration."""
    if which == "frontier":
        from agent.models.frontier import load_frontier_adapter

        return load_frontier_adapter(**kwargs)
    if which == "oss":
        from agent.models.oss import OSSAdapter

        return OSSAdapter(**kwargs)
    raise ConfigError(f"Unknown agent model {which!r}; expected 'frontier' or 'oss'")


def assert_distinct_families(agents: list[ModelAdapter], judge: ModelAdapter) -> None:
    """Fail loudly if the judge shares a model family with any agent under test.

    A judge from the same family as a candidate exhibits self-preference bias, which
    silently invalidates every comparison built on top of it. This check is a guardrail
    against a misconfiguration that produces plausible-looking but worthless numbers.

    Raises:
        ConfigError: the judge's family matches an agent's.
    """
    clashing = [agent for agent in agents if agent.family == judge.family]
    if clashing:
        names = ", ".join(f"{agent.name} ({agent.model_id})" for agent in clashing)
        raise ConfigError(
            f"Judge {judge.model_id!r} is from family {judge.family!r}, shared with {names}. "
            "A judge scoring its own family exhibits self-preference bias; configure a "
            "third family (PROJECT.md)."
        )
