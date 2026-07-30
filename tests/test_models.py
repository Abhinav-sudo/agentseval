"""Tests for the model adapters, with the HTTP layer mocked.

Every request is served by an `httpx.MockTransport`, so no test touches the network or needs
an API key, and `RetryPolicy.sleep` is injected so the backoff schedule is asserted without
waiting for it.

The tests that matter most are the ones guarding decisions locked in PROJECT.md: that no
adapter sends native tool-calling parameters, that the judge is a different family from both
agents, and that the retry and cache behaviour is identical across arms because it lives in
one place.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from dotenv import find_dotenv

from agent.models import base
from agent.models.base import (
    FORBIDDEN_BODY_KEYS,
    ChatAdapter,
    ConfigError,
    FinishReason,
    ModelAdapter,
    ModelError,
    RequestError,
    ResponseCache,
    RetryPolicy,
    assert_distinct_families,
    cache_enabled,
    estimate_usd_cost,
    is_retryable,
    load_agent_model,
    load_env,
)
from agent.models.frontier import (
    FrontierAdapter,
    GeminiFrontierAdapter,
    load_frontier_adapter,
    resolved_frontier_model,
)
from agent.models.judge_model import JudgeAdapter, load_judge_model
from agent.models.oss import OSSAdapter, resolved_oss_model

MESSAGES = [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "What is in the knowledge base?"},
]


# --------------------------------------------------------------------------------------
# Mock transport helpers
# --------------------------------------------------------------------------------------


class Recorder:
    """Serves canned responses and records every request that was sent."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.queue = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # The last queued response repeats, so a test needing "always 500" queues one.
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]

    @property
    def count(self) -> int:
        return len(self.requests)

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def anthropic_ok(
    text: str = "hello",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    stop_reason: str | None = "end_turn",
):
    payload: dict = {
        "id": "msg_1",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    if stop_reason is not None:
        payload["stop_reason"] = stop_reason
    return httpx.Response(200, json=payload)


def openai_ok(
    text: str = "hello",
    prompt_tokens: int = 1000,
    completion_tokens: int = 500,
    finish_reason: str | None = "stop",
    total_tokens: int | None = None,
):
    """An OpenAI-compatible success.

    `total_tokens` defaults to `prompt + completion`, which is what a provider that does not
    meter thinking separately returns. Passing a larger value is how a Gemini-style response
    is built: the surplus is what `derive_reasoning_tokens` recovers.
    """
    choice: dict = {"index": 0, "message": {"role": "assistant", "content": text}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_1",
            "model": "gpt-4o-2024-11-20",
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    prompt_tokens + completion_tokens if total_tokens is None else total_tokens
                ),
            },
        },
    )


class FakeSleep:
    """Records requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def no_jitter(sleep: FakeSleep | None = None, **kwargs) -> RetryPolicy:
    """A deterministic retry policy: no jitter, no real sleeping."""
    return RetryPolicy(jitter=False, sleep=sleep or FakeSleep(), **kwargs)


def common_kwargs(recorder: Recorder, tmp_path, kwargs: dict) -> dict:
    """Adapter options shared by the builders.

    Caching is off unless the test supplies a `tmp_path` to cache into, so no test can
    write into the real `.cache/` directory. Explicit kwargs win.
    """
    return {
        "api_key": "test-key",
        "http_client": recorder.client(),
        "no_cache": tmp_path is None,
        "cache_dir": tmp_path or base.DEFAULT_CACHE_DIR,
        **kwargs,
    }


def frontier(recorder: Recorder, tmp_path=None, **kwargs) -> FrontierAdapter:
    return FrontierAdapter(
        model_id="claude-sonnet-4-20250514", **common_kwargs(recorder, tmp_path, kwargs)
    )


def gemini(recorder: Recorder, tmp_path=None, **kwargs) -> GeminiFrontierAdapter:
    return GeminiFrontierAdapter(
        model_id="gemini-3.6-flash", **common_kwargs(recorder, tmp_path, kwargs)
    )


def oss(recorder: Recorder, tmp_path=None, **kwargs) -> OSSAdapter:
    return OSSAdapter(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        **common_kwargs(recorder, tmp_path, kwargs),
    )


def judge(recorder: Recorder, tmp_path=None, **kwargs) -> JudgeAdapter:
    return JudgeAdapter(model_id="gpt-4o-2024-11-20", **common_kwargs(recorder, tmp_path, kwargs))


#: (builder, canned success response) for tests that must hold across every adapter, including
#: both frontier providers: the invariants that keep the two arms on one harness are exactly the
#: ones a newly added provider is most likely to break.
ALL_ADAPTERS = [
    pytest.param(frontier, anthropic_ok, id="frontier"),
    pytest.param(gemini, openai_ok, id="frontier-gemini"),
    pytest.param(oss, openai_ok, id="oss"),
    pytest.param(judge, openai_ok, id="judge"),
]


# --------------------------------------------------------------------------------------
# Interface conformance
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("build", "ok"), ALL_ADAPTERS)
def test_adapter_satisfies_the_protocol(build, ok):
    adapter = build(Recorder(ok()))
    assert isinstance(adapter, ModelAdapter)
    assert isinstance(adapter, ChatAdapter)
    assert adapter.name and adapter.model_id and adapter.family and adapter.provider


def test_the_three_roles_have_distinct_names_and_families():
    """The judge must not share a family with either agent (PROJECT.md)."""
    adapters = [
        frontier(Recorder(anthropic_ok())),
        oss(Recorder(openai_ok())),
        judge(Recorder(openai_ok())),
    ]
    assert [a.name for a in adapters] == ["frontier", "oss", "judge"]
    assert len({a.family for a in adapters}) == 3


# --------------------------------------------------------------------------------------
# No native tool calling — the locked decision
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("build", "ok"), ALL_ADAPTERS)
def test_no_native_tool_calling_parameters_are_sent(build, ok):
    """Native function calling on one arm only would confound model with harness quality.

    Tool docs reach the model through the system prompt; nothing may reach the provider as
    a structured tool definition.
    """
    recorder = Recorder(ok())
    build(recorder).generate(MESSAGES)

    body = recorder.body()
    assert FORBIDDEN_BODY_KEYS.isdisjoint(body)
    assert "response_format" not in body


@pytest.mark.parametrize(("build", "ok"), ALL_ADAPTERS)
def test_request_body_is_a_plain_chat_completion(build, ok):
    """Only model, messages/system, temperature, max_tokens, and stop are ever sent."""
    recorder = Recorder(ok())
    build(recorder).generate(MESSAGES, temperature=0.0, max_tokens=256, stop=["END"])

    allowed = {"model", "messages", "system", "temperature", "max_tokens", "stop", "stop_sequences"}
    assert set(recorder.body()) <= allowed


# --------------------------------------------------------------------------------------
# Wire format per provider
# --------------------------------------------------------------------------------------


def test_frontier_hoists_the_system_prompt_and_maps_stop():
    recorder = Recorder(anthropic_ok())
    frontier(recorder).generate(MESSAGES, temperature=0.0, max_tokens=256, stop=["END"])

    request = recorder.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"

    body = recorder.body()
    assert body["system"] == "You are a careful assistant."
    assert body["messages"] == [MESSAGES[1]]
    assert body["stop_sequences"] == ["END"]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 256


def test_frontier_joins_multiple_system_messages():
    """Dropping one would change what the model saw without showing up anywhere."""
    recorder = Recorder(anthropic_ok())
    frontier(recorder).generate(
        [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert recorder.body()["system"] == "first\n\nsecond"


def test_frontier_omits_system_when_there_is_none():
    recorder = Recorder(anthropic_ok())
    frontier(recorder).generate([{"role": "user", "content": "hi"}])
    assert "system" not in recorder.body()


@pytest.mark.parametrize(
    ("build", "expected_url"),
    [
        (oss, "https://api.groq.com/openai/v1/chat/completions"),
        (judge, "https://api.openai.com/v1/chat/completions"),
    ],
)
def test_openai_compatible_wire_format(build, expected_url):
    recorder = Recorder(openai_ok())
    build(recorder).generate(MESSAGES, stop=["END"])

    request = recorder.requests[0]
    assert str(request.url) == expected_url
    assert request.headers["authorization"] == "Bearer test-key"

    body = recorder.body()
    assert body["messages"] == MESSAGES  # system stays in the message list here
    assert body["stop"] == ["END"]


def test_stop_is_omitted_when_not_requested():
    recorder = Recorder(openai_ok())
    oss(recorder).generate(MESSAGES)
    assert "stop" not in recorder.body()


def test_together_provider_changes_the_base_url():
    recorder = Recorder(openai_ok())
    adapter = OSSAdapter(
        model_id="Qwen/Qwen2.5-7B-Instruct-Turbo",
        provider="together",
        api_key="k",
        http_client=recorder.client(),
        no_cache=True,
    )
    adapter.generate(MESSAGES)
    assert str(recorder.requests[0].url).startswith("https://api.together.xyz/v1")
    assert adapter.provider == "together"


def test_gemini_frontier_uses_the_openai_compatible_endpoint():
    """The Gemini arm rides the shared OpenAI-compatible wire format, bearer auth included.

    Sending Gemini's native `generateContent` instead would mean the frontier arm had its own
    body builder, and the ban on native tool calling would stop holding by construction.
    """
    recorder = Recorder(openai_ok())
    adapter = gemini(recorder)
    adapter.generate(MESSAGES)

    request = recorder.requests[0]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert request.headers["Authorization"] == "Bearer test-key"
    # The system turn stays in `messages` rather than being hoisted, unlike the Anthropic arm.
    assert [m["role"] for m in recorder.body()["messages"]] == ["system", "user"]


def test_gemini_frontier_keeps_the_judge_a_third_family():
    """Gemini is the frontier arm precisely because gpt-4o judging it is not self-preference."""
    agents = [gemini(Recorder(openai_ok())), oss(Recorder(openai_ok()))]
    assert_distinct_families(agents, judge(Recorder(openai_ok())))
    assert len({a.family for a in agents}) == 2


# --------------------------------------------------------------------------------------
# Response parsing, timing, cost
# --------------------------------------------------------------------------------------


def test_frontier_parses_text_tokens_and_cost():
    recorder = Recorder(anthropic_ok("the answer", input_tokens=1000, output_tokens=500))
    response = frontier(recorder).generate(MESSAGES)

    assert response.text == "the answer"
    assert (response.prompt_tokens, response.completion_tokens) == (1000, 500)
    assert response.latency_ms >= 0
    assert response.usd_cost == pytest.approx(0.0105)  # $3/$15 per 1M
    assert response.raw["id"] == "msg_1"
    assert response.cached is False


def test_oss_parses_text_tokens_and_cost():
    recorder = Recorder(openai_ok("the oss answer", prompt_tokens=1000, completion_tokens=1000))
    response = oss(recorder).generate(MESSAGES)

    assert response.text == "the oss answer"
    assert response.usd_cost == pytest.approx(0.00013)  # $0.05/$0.08 per 1M
    assert response.cached is False


def test_frontier_concatenates_text_blocks():
    recorder = Recorder(
        httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    assert frontier(recorder).generate(MESSAGES).text == "ab"


def test_null_content_becomes_empty_string_not_an_error():
    """An empty completion is a fact about the model, recorded rather than raised."""
    recorder = Recorder(
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": None}}], "usage": {}},
        )
    )
    response = oss(recorder).generate(MESSAGES)
    assert response.text == ""
    assert response.usd_cost is None  # no token counts reported


@pytest.mark.parametrize(("build", "ok"), ALL_ADAPTERS)
def test_unexpected_response_shape_raises_model_error(build, ok):
    recorder = Recorder(httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ModelError):
        build(recorder).generate(MESSAGES)


@pytest.mark.parametrize(("build", "ok"), ALL_ADAPTERS)
def test_non_json_response_raises_model_error(build, ok):
    recorder = Recorder(httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(ModelError, match="non-JSON"):
        build(recorder).generate(MESSAGES)


# --------------------------------------------------------------------------------------
# Finish reasons, normalised
# --------------------------------------------------------------------------------------
#
# Two providers, two vocabularies, one enum. `agent.core` reads it to tell a model that broke
# the tool protocol from a reply our own `max_tokens` cut in half — opposite findings, one a
# model failure and the other ours — so a mapping that quietly reports every stop as
# "complete" would file our truncations as the model's violations.


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", FinishReason.COMPLETE),
        ("max_tokens", FinishReason.LENGTH),
        ("stop_sequence", FinishReason.STOP_SEQUENCE),
        ("refusal", FinishReason.CONTENT_FILTER),
        ("tool_use", FinishReason.OTHER),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_anthropic_stop_reasons_are_normalised(stop_reason, expected):
    recorder = Recorder(anthropic_ok("hi", stop_reason=stop_reason))
    assert frontier(recorder).generate(MESSAGES).finish_reason is expected


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", FinishReason.COMPLETE),
        ("length", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("tool_calls", FinishReason.OTHER),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_openai_finish_reasons_are_normalised(finish_reason, expected):
    recorder = Recorder(openai_ok("hi", finish_reason=finish_reason))
    assert oss(recorder).generate(MESSAGES).finish_reason is expected


@pytest.mark.parametrize(
    ("build", "ok", "truncating"),
    [
        (frontier, anthropic_ok, {"stop_reason": "max_tokens"}),
        (oss, openai_ok, {"finish_reason": "length"}),
    ],
)
def test_truncated_is_true_only_for_the_length_stop(build, ok, truncating):
    assert build(Recorder(ok("cut off", **truncating))).generate(MESSAGES).truncated is True
    assert build(Recorder(ok("all of it"))).generate(MESSAGES).truncated is False


# --------------------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------------------


def test_is_retryable_covers_429_and_5xx_only():
    assert is_retryable(429)
    assert all(is_retryable(s) for s in (500, 502, 503, 504))
    assert not any(is_retryable(s) for s in (400, 401, 403, 404, 200))


def test_rate_limit_then_success():
    recorder = Recorder(httpx.Response(429), openai_ok("recovered"))
    sleep = FakeSleep()
    response = oss(recorder, retry_policy=no_jitter(sleep)).generate(MESSAGES)

    assert response.text == "recovered"
    assert recorder.count == 2
    assert len(sleep.delays) == 1


def test_five_server_errors_then_success_is_within_budget():
    recorder = Recorder(*[httpx.Response(500)] * 5, openai_ok("recovered"))
    sleep = FakeSleep()
    response = oss(recorder, retry_policy=no_jitter(sleep)).generate(MESSAGES)

    assert response.text == "recovered"
    assert recorder.count == 6  # one attempt plus five retries
    assert len(sleep.delays) == 5


def test_six_failures_exhausts_retries():
    recorder = Recorder(httpx.Response(500, text="upstream exploded"))
    with pytest.raises(ModelError, match="after 6 attempts"):
        oss(recorder, retry_policy=no_jitter()).generate(MESSAGES)
    assert recorder.count == 6


def test_client_errors_are_not_retried():
    """Retrying a bad key five times wastes a minute and buries the real error."""
    recorder = Recorder(httpx.Response(401, text="invalid api key"))
    sleep = FakeSleep()
    with pytest.raises(RequestError, match="401"):
        oss(recorder, retry_policy=no_jitter(sleep)).generate(MESSAGES)

    assert recorder.count == 1
    assert sleep.delays == []


def test_connection_errors_are_retried():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(explode))
    adapter = OSSAdapter(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        api_key="k",
        http_client=client,
        no_cache=True,
        retry_policy=no_jitter(max_retries=2),
    )
    with pytest.raises(ModelError, match="after 3 attempts"):
        adapter.generate(MESSAGES)


def test_backoff_grows_exponentially():
    policy = no_jitter()
    delays = [policy.delay_for(attempt) for attempt in range(1, 6)]
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_backoff_is_capped():
    policy = RetryPolicy(jitter=False, base_delay_s=10.0, max_delay_s=15.0)
    assert policy.delay_for(5) == 15.0


def test_jitter_stays_within_the_computed_delay():
    policy = RetryPolicy(jitter=True, base_delay_s=1.0)
    assert all(0.0 <= policy.delay_for(3) <= 4.0 for _ in range(50))


def test_retry_after_header_wins_over_backoff():
    """Only the provider knows when the limit actually resets."""
    recorder = Recorder(httpx.Response(429, headers={"retry-after": "7"}), openai_ok())
    sleep = FakeSleep()
    oss(recorder, retry_policy=no_jitter(sleep)).generate(MESSAGES)
    assert sleep.delays == [7.0]


def test_unparseable_retry_after_falls_back_to_backoff():
    recorder = Recorder(
        httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        openai_ok(),
    )
    sleep = FakeSleep()
    oss(recorder, retry_policy=no_jitter(sleep)).generate(MESSAGES)
    assert sleep.delays == [0.5]


def test_error_message_carries_provider_detail():
    """The provider's own words are the actionable part and must reach the trace."""
    recorder = Recorder(httpx.Response(400, text="max_tokens too large"))
    with pytest.raises(RequestError, match="max_tokens too large"):
        oss(recorder).generate(MESSAGES)


# --------------------------------------------------------------------------------------
# Response cache
# --------------------------------------------------------------------------------------


def test_second_identical_call_is_served_from_cache(tmp_path):
    recorder = Recorder(openai_ok("cached answer"))
    adapter = oss(recorder, tmp_path)

    first = adapter.generate(MESSAGES)
    second = adapter.generate(MESSAGES)

    assert recorder.count == 1  # no second HTTP request
    assert second.text == first.text == "cached answer"
    assert first.cached is False
    assert second.cached is True
    assert second.latency_ms == first.latency_ms  # replayed, not re-measured
    assert second.usd_cost == first.usd_cost


def test_cache_entry_is_readable_json_with_the_request(tmp_path):
    recorder = Recorder(openai_ok("stored"))
    adapter = oss(recorder, tmp_path)
    adapter.generate(MESSAGES)

    (entry_path,) = list(tmp_path.glob("*.json"))
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    assert entry["response"]["text"] == "stored"
    assert entry["request"]["body"]["messages"] == MESSAGES
    assert entry["request"]["endpoint"].endswith("/chat/completions")


def test_a_fresh_adapter_reuses_the_cache_on_disk(tmp_path):
    """The point of the cache: re-running an eval does not re-pay for it."""
    first_recorder = Recorder(openai_ok("first run"))
    oss(first_recorder, tmp_path).generate(MESSAGES)

    second_recorder = Recorder(openai_ok("should not be reached"))
    response = oss(second_recorder, tmp_path).generate(MESSAGES)

    assert second_recorder.count == 0
    assert response.text == "first run"
    assert response.cached is True


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"messages": [{"role": "user", "content": "different"}]}, id="messages"),
        pytest.param({"temperature": 0.7}, id="temperature"),
        pytest.param({"max_tokens": 2048}, id="max_tokens"),
        pytest.param({"stop": ["END"]}, id="stop"),
    ],
)
def test_changing_any_request_parameter_misses_the_cache(tmp_path, changed):
    """Anything that changes the output must change the key, or a run reuses wrong data."""
    call = {"messages": MESSAGES, "temperature": 0.0, "max_tokens": 1024, "stop": None}
    recorder = Recorder(openai_ok())
    adapter = oss(recorder, tmp_path)

    adapter.generate(call["messages"], temperature=0.0, max_tokens=1024, stop=None)
    updated = {**call, **changed}
    adapter.generate(
        updated["messages"],
        temperature=updated["temperature"],
        max_tokens=updated["max_tokens"],
        stop=updated["stop"],
    )
    assert recorder.count == 2


def test_changing_the_model_misses_the_cache(tmp_path):
    recorder = Recorder(openai_ok())
    oss(recorder, tmp_path).generate(MESSAGES)
    OSSAdapter(
        model_id="a-different-model",
        provider="groq",
        api_key="test-key",
        http_client=recorder.client(),
        no_cache=False,
        cache_dir=tmp_path,
    ).generate(MESSAGES)
    assert recorder.count == 2


def test_cache_key_ignores_the_api_key(tmp_path):
    """Rotating a credential must not discard the cache, or leak into the key."""
    recorder = Recorder(openai_ok("shared"))
    oss(recorder, tmp_path).generate(MESSAGES)

    rotated = OSSAdapter(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        api_key="a-totally-different-key",
        http_client=recorder.client(),
        no_cache=False,
        cache_dir=tmp_path,
    )
    response = rotated.generate(MESSAGES)

    assert recorder.count == 1
    assert response.cached is True


def test_no_cache_bypasses_read_and_write(tmp_path):
    recorder = Recorder(openai_ok())
    adapter = oss(recorder, tmp_path, no_cache=True)

    adapter.generate(MESSAGES)
    adapter.generate(MESSAGES)

    assert recorder.count == 2
    assert list(tmp_path.glob("*.json")) == []


def test_env_var_disables_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTSEVAL_NO_CACHE", "1")
    recorder = Recorder(openai_ok())
    adapter = OSSAdapter(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        api_key="k",
        http_client=recorder.client(),
        cache_dir=tmp_path,
    )
    adapter.generate(MESSAGES)
    adapter.generate(MESSAGES)
    assert recorder.count == 2


def test_explicit_flag_overrides_the_env_var(monkeypatch):
    monkeypatch.setenv("AGENTSEVAL_NO_CACHE", "1")
    assert cache_enabled(no_cache=False) is True
    monkeypatch.delenv("AGENTSEVAL_NO_CACHE")
    assert cache_enabled(no_cache=True) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_env_values_disable_the_cache(monkeypatch, value):
    monkeypatch.setenv("AGENTSEVAL_NO_CACHE", value)
    assert cache_enabled() is False


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_other_env_values_leave_the_cache_on(monkeypatch, value):
    monkeypatch.setenv("AGENTSEVAL_NO_CACHE", value)
    assert cache_enabled() is True


def test_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    """A damaged cache file should cost one API call, not abort a run."""
    recorder = Recorder(openai_ok("refetched"))
    adapter = oss(recorder, tmp_path)
    adapter.generate(MESSAGES)

    (entry_path,) = list(tmp_path.glob("*.json"))
    entry_path.write_text("{ truncated", encoding="utf-8")

    response = adapter.generate(MESSAGES)
    assert recorder.count == 2
    assert response.text == "refetched"


def test_a_cached_response_replays_its_finish_reason(tmp_path):
    """Otherwise a re-run would read every cached truncation as an unexplained parse failure,
    and the format-violation rate would depend on whether the cache was warm."""
    recorder = Recorder(openai_ok("cut off", finish_reason="length"))
    adapter = oss(recorder, tmp_path)
    adapter.generate(MESSAGES)
    replayed = adapter.generate(MESSAGES)

    assert recorder.count == 1
    assert replayed.cached is True
    assert replayed.finish_reason is FinishReason.LENGTH
    assert replayed.truncated is True


def test_the_cache_key_changes_with_the_payload_version(monkeypatch):
    """How entries written before `finish_reason` existed are made unreachable: they hash to a
    different key, so nothing deserialises them into a response whose stop reason is a guess."""
    args = {"temperature": 0.0, "max_tokens": 1024, "stop": None}
    before = ResponseCache.key("m", MESSAGES, **args)
    monkeypatch.setattr(base, "CACHE_VERSION", base.CACHE_VERSION + 1)

    assert ResponseCache.key("m", MESSAGES, **args) != before


def test_an_entry_missing_a_field_is_a_miss_rather_than_a_default(tmp_path):
    """The second line of defence, in case an entry is reachable but incomplete: read strictly
    and refetch, because a defaulted `finish_reason` is a guess about whether we truncated."""
    recorder = Recorder(openai_ok("first"), openai_ok("second"))
    adapter = oss(recorder, tmp_path)
    adapter.generate(MESSAGES)

    (entry,) = list(tmp_path.glob("*.json"))
    stale = json.loads(entry.read_text(encoding="utf-8"))
    del stale["response"]["finish_reason"]
    entry.write_text(json.dumps(stale), encoding="utf-8")

    assert adapter.generate(MESSAGES).text == "second"
    assert recorder.count == 2


def test_cache_key_is_stable_and_order_independent():
    args = {"temperature": 0.0, "max_tokens": 1024, "stop": None}
    first = ResponseCache.key("m", MESSAGES, **args)
    second = ResponseCache.key("m", [dict(reversed(list(m.items()))) for m in MESSAGES], **args)
    assert first == second  # key sensitivity is about values, not dict ordering


def test_cache_dir_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "from-env"))
    recorder = Recorder(openai_ok())
    OSSAdapter(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        api_key="k",
        http_client=recorder.client(),
        no_cache=False,
    ).generate(MESSAGES)

    assert list((tmp_path / "from-env").glob("*.json"))


def test_explicit_cache_dir_beats_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "ignored"))
    assert base.resolve_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert base.resolve_cache_dir() == tmp_path / "ignored"


def test_no_cache_arguments_flag_is_defined_once():
    import argparse

    parser = base.add_cache_arguments(argparse.ArgumentParser())
    assert parser.parse_args([]).no_cache is None
    assert parser.parse_args(["--no-cache"]).no_cache is True


def test_omitting_the_flag_leaves_the_env_var_reachable(monkeypatch):
    """`store_true`'s usual False default would make AGENTSEVAL_NO_CACHE unreachable from a CLI."""
    import argparse

    parser = base.add_cache_arguments(argparse.ArgumentParser())
    monkeypatch.setenv("AGENTSEVAL_NO_CACHE", "1")
    assert cache_enabled(parser.parse_args([]).no_cache) is False
    assert cache_enabled(parser.parse_args(["--no-cache"]).no_cache) is False


# --------------------------------------------------------------------------------------
# Cost table
# --------------------------------------------------------------------------------------


def test_estimate_usd_cost_matches_the_price_table():
    # 1000 prompt + 500 completion tokens at $3 / $15 per 1M.
    assert estimate_usd_cost("claude-sonnet-4-20250514", 1000, 500) == pytest.approx(0.0105)


def test_estimate_usd_cost_is_none_when_unknowable():
    """None rather than 0.0: a zero would silently understate the cost of a run."""
    assert estimate_usd_cost("some-unpriced-model", 1000, 500) is None
    assert estimate_usd_cost("gpt-4o", None, 500) is None
    assert estimate_usd_cost("gpt-4o", 1000, None) is None


def test_estimate_usd_cost_prefers_the_longest_matching_prefix():
    """gpt-4o-mini must not be priced as gpt-4o."""
    assert estimate_usd_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_usd_cost("gpt-4o-2024-11-20", 1_000_000, 0) == pytest.approx(2.50)


def test_every_default_model_is_priced():
    """An unpriced default would silently report None cost for a whole run."""
    from agent.models.frontier import DEFAULT_FRONTIER_MODEL, DEFAULT_GEMINI_MODEL
    from agent.models.judge_model import DEFAULT_JUDGE_MODEL
    from agent.models.oss import DEFAULT_OSS_MODEL

    for model_id in (
        DEFAULT_FRONTIER_MODEL,
        DEFAULT_GEMINI_MODEL,
        DEFAULT_OSS_MODEL,
        DEFAULT_JUDGE_MODEL,
    ):
        assert estimate_usd_cost(model_id, 1, 1) is not None, model_id


def test_gemini_flash_is_priced_with_output_dearer_than_input():
    """$1.50 / $7.50 per 1M (ai.google.dev, 2026-07-29). Output is 5x input on this model, so
    a thinking-heavy reply is the expensive case and must not be priced at the input rate."""
    assert estimate_usd_cost("gemini-3.6-flash", 1_000_000, 0) == pytest.approx(1.50)
    assert estimate_usd_cost("gemini-3.6-flash", 0, 1_000_000) == pytest.approx(7.50)


def test_reasoning_tokens_are_billed_at_the_completion_rate():
    """Thinking is output, and providers charge it as output."""
    visible_only = estimate_usd_cost("gemini-3.6-flash", 0, 1_000)
    with_thinking = estimate_usd_cost("gemini-3.6-flash", 0, 1_000, 9_000)

    assert visible_only == pytest.approx(1_000 * 7.50 / 1_000_000)
    assert with_thinking == pytest.approx(10_000 * 7.50 / 1_000_000)
    assert with_thinking == pytest.approx(visible_only * 10)


def test_omitting_reasoning_tokens_prices_a_call_exactly_as_before():
    """The parameter is additive: every existing caller and every non-thinking provider must
    get the same number it got before reasoning tokens were counted at all."""
    for model_id in ("claude-sonnet-4-20250514", "gpt-4o-2024-11-20", "llama-3.1-8b-instant"):
        baseline = estimate_usd_cost(model_id, 1_000, 500)
        assert estimate_usd_cost(model_id, 1_000, 500, None) == baseline
        assert estimate_usd_cost(model_id, 1_000, 500, 0) == baseline


# --------------------------------------------------------------------------------------
# Reasoning-token accounting
# --------------------------------------------------------------------------------------

#: Usage objects copied verbatim from live responses on 2026-07-29, one per provider the
#: harness talks to. The Gemini rows are the interesting ones: `completion_tokens` there is the
#: *visible* reply and the thinking is only in `total_tokens`, which is why the residual is the
#: quantity to take. `gemini-truncated` is the hall-023 failure — 36 visible tokens against a
#: 1024 ceiling, because 984 tokens of thinking were charged against the same budget.
LIVE_USAGE: dict[str, tuple[dict[str, int], int]] = {
    "gpt-4o": ({"prompt_tokens": 1514, "completion_tokens": 148, "total_tokens": 1662}, 0),
    "gpt-4o-cached-prompt": (
        {"prompt_tokens": 1511, "completion_tokens": 160, "total_tokens": 1671},
        0,
    ),
    "groq-llama": ({"prompt_tokens": 3344, "completion_tokens": 187, "total_tokens": 3531}, 0),
    "groq-llama-short": (
        {"prompt_tokens": 2181, "completion_tokens": 30, "total_tokens": 2211},
        0,
    ),
    "gemini-no-thinking": (
        {"prompt_tokens": 3351, "completion_tokens": 236, "total_tokens": 3587},
        0,
    ),
    "gemini-thinking": (
        {"prompt_tokens": 3351, "completion_tokens": 32, "total_tokens": 3840},
        457,
    ),
    "gemini-truncated": (
        {"prompt_tokens": 3351, "completion_tokens": 36, "total_tokens": 4371},
        984,
    ),
}


@pytest.mark.parametrize(("label", "case"), sorted(LIVE_USAGE.items()))
def test_the_reasoning_residual_matches_what_each_provider_actually_returned(label, case):
    """The guard on the derivation rule.

    A provider swap, or a provider changing which fields it fills, must not silently move
    thinking tokens out of the cost basis — that is precisely how the frontier arm came to be
    billed at a quarter of its true output. Non-thinking providers must keep coming out at
    exactly 0 so that counting reasoning costs them nothing.
    """
    usage, expected = case
    assert (
        base.derive_reasoning_tokens(
            usage["total_tokens"], usage["prompt_tokens"], usage["completion_tokens"]
        )
        == expected
    ), label


def test_no_non_thinking_provider_gains_reasoning_tokens():
    """Stated as its own assertion because it is the half that protects the old numbers."""
    non_thinking = [case for label, case in LIVE_USAGE.items() if not label.startswith("gemini")]
    assert non_thinking
    for usage, _ in non_thinking:
        residual = base.derive_reasoning_tokens(
            usage["total_tokens"], usage["prompt_tokens"], usage["completion_tokens"]
        )
        assert residual == 0


def test_a_provider_reporting_no_total_leaves_reasoning_unknown():
    """None, not 0: a zero would reinstate the understatement. Adapters for wire formats with
    no total say 0 for themselves, which is a claim rather than a default."""
    assert base.derive_reasoning_tokens(None, 100, 20) is None
    assert base.derive_reasoning_tokens(120, None, 20) is None
    assert base.derive_reasoning_tokens(120, 100, None) is None


def test_an_inconsistent_total_clamps_instead_of_billing_negative_tokens():
    assert base.derive_reasoning_tokens(100, 100, 20) == 0


def test_the_gemini_adapter_records_the_residual_and_bills_it():
    """3351 prompt + 36 visible against a 4371 total: 984 tokens of thinking, charged.

    These are the hall-023 numbers, and this is the assertion that would have caught the
    understatement when it was introduced.
    """
    recorder = Recorder(
        openai_ok(text="cut off", prompt_tokens=3351, completion_tokens=36, total_tokens=4371)
    )
    response = gemini(recorder).generate(MESSAGES)

    assert response.completion_tokens == 36
    assert response.reasoning_tokens == 984
    assert response.billed_completion_tokens == 1020
    assert response.usd_cost == pytest.approx((3351 * 1.50 + 1020 * 7.50) / 1_000_000)


def test_the_anthropic_adapter_claims_zero_rather_than_unknown():
    """Anthropic publishes no total, and needs none: thinking is already in `output_tokens`.
    Left as None the residual would be unknown, and the arm's cost would still be right —
    but the claim has to be explicit, or a later reader cannot tell which it is."""
    response = frontier(Recorder(anthropic_ok(input_tokens=1000, output_tokens=500))).generate(
        MESSAGES
    )

    assert response.reasoning_tokens == 0
    assert response.billed_completion_tokens == 500
    assert response.usd_cost == pytest.approx((1000 * 3.00 + 500 * 15.00) / 1_000_000)


def test_reasoning_tokens_survive_a_cache_round_trip(tmp_path):
    """A replayed call must not lose the thinking it was billed for; if it did, a re-run would
    report a cheaper number than the run it replays."""
    recorder = Recorder(
        openai_ok(prompt_tokens=3351, completion_tokens=32, total_tokens=3840),
        openai_ok(prompt_tokens=1, completion_tokens=1),
    )
    adapter = gemini(recorder, tmp_path)
    fresh = adapter.generate(MESSAGES)
    replayed = adapter.generate(MESSAGES)

    assert replayed.cached is True
    assert replayed.reasoning_tokens == fresh.reasoning_tokens == 457
    assert replayed.usd_cost == fresh.usd_cost


def test_billed_completion_tokens_is_unknown_when_the_visible_count_is():
    response = base.ModelResponse(
        text="", prompt_tokens=None, completion_tokens=None, latency_ms=0.0, usd_cost=None, raw={}
    )
    assert response.billed_completion_tokens is None


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Real `.env` discovery, into a throwaway environment, from an empty directory.

    Three things, all of which these tests need and no other test wants. Discovery is restored
    because `conftest.py` disables it for the suite. `os.environ` is swapped for a copy because
    `load_env` writes into it through `python-dotenv`, which `monkeypatch.setenv` cannot undo —
    it restores only keys it recorded itself, and the keys here are absent beforehand. And the
    chdir gives the walk-up somewhere empty to start, so a test asserting "no file" does not
    depend on the repository having no `.env`.
    """
    monkeypatch.setattr(base, "find_dotenv", find_dotenv)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.chdir(tmp_path)
    return os.environ


def test_load_env_populates_the_process_environment(env, tmp_path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8")
    env.pop("ANTHROPIC_API_KEY", None)

    assert load_env() == tmp_path / ".env"
    assert env["ANTHROPIC_API_KEY"] == "from-dotenv"


def test_load_env_does_not_override_an_exported_variable(env, tmp_path):
    """An export is a deliberate override for one command; `.env` is the project default."""
    (tmp_path / ".env").write_text("OSS_PROVIDER=groq\n", encoding="utf-8")
    env["OSS_PROVIDER"] = "together"

    load_env()

    assert env["OSS_PROVIDER"] == "together"


def test_load_env_is_found_from_a_subdirectory(env, tmp_path, monkeypatch):
    """A CLI run from `evals/` must configure the same way as one run from the root."""
    (tmp_path / ".env").write_text("JUDGE_MODEL=gpt-4o-2024-11-20\n", encoding="utf-8")
    nested = tmp_path / "evals" / "datasets"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    env.pop("JUDGE_MODEL", None)

    assert load_env() == tmp_path / ".env"
    assert env["JUDGE_MODEL"] == "gpt-4o-2024-11-20"


def test_load_env_reports_no_file_rather_than_raising(env):
    """CI sets the environment directly; `require_env` is what reports an absent key."""
    assert load_env() is None


def test_missing_api_key_raises_without_echoing_secrets(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError) as exc:
        FrontierAdapter(model_id="claude-sonnet-4-20250514")
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)


def test_blank_api_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        JudgeAdapter()


def test_unknown_oss_provider_raises(monkeypatch):
    monkeypatch.setenv("OSS_PROVIDER", "vllm-on-my-laptop")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    with pytest.raises(ConfigError, match="vllm-on-my-laptop"):
        OSSAdapter()


def test_unknown_frontier_provider_raises(monkeypatch):
    """Naming an unconfigured host must fail rather than fall back to the default.

    A silent fallback would write a manifest claiming a provider that did not serve the run.
    """
    monkeypatch.setenv("FRONTIER_PROVIDER", "bedrock")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ConfigError) as exc:
        load_frontier_adapter()
    assert "bedrock" in str(exc.value)
    assert "anthropic, gemini" in str(exc.value)


def test_frontier_provider_defaults_to_anthropic(monkeypatch):
    """An existing .env naming no provider keeps resolving to the arm it always did."""
    monkeypatch.delenv("FRONTIER_PROVIDER", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    adapter = load_frontier_adapter()
    assert isinstance(adapter, FrontierAdapter)
    assert (adapter.family, adapter.provider) == ("anthropic", "anthropic")


def test_adapters_read_their_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("FRONTIER_MODEL", "claude-3-5-sonnet-20241022")
    monkeypatch.setenv("OSS_PROVIDER", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setenv("OSS_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")

    assert load_agent_model("frontier").model_id == "claude-3-5-sonnet-20241022"
    assert load_agent_model("oss").provider == "together"
    assert load_judge_model().model_id == "gpt-4o-mini"


def test_frontier_provider_gemini_resolves_through_load_agent_model(monkeypatch):
    """With no `FRONTIER_MODEL` set, the provider switch alone reaches Gemini's own default."""
    monkeypatch.setenv("FRONTIER_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)

    adapter = load_agent_model("frontier")
    assert adapter.model_id == "gemini-3.6-flash"
    assert (adapter.name, adapter.family, adapter.provider) == ("frontier", "gemini", "gemini")


def test_switching_only_the_provider_is_refused_rather_than_sent_to_the_vendor(monkeypatch):
    """The real contract, and the one this pair of variables gets wrong in practice.

    `FRONTIER_PROVIDER` is *not* the whole switch: `FRONTIER_MODEL` names the selected
    provider's own model, so flipping one without the other is a misconfiguration. Left
    unchecked it becomes `404 ... not found for API version v1main` from Google, which reads
    like a retired model rather than a stale variable. The error has to name both variables.
    """
    monkeypatch.setenv("FRONTIER_PROVIDER", "gemini")
    monkeypatch.setenv("FRONTIER_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    with pytest.raises(ConfigError) as exc:
        load_frontier_adapter()

    message = str(exc.value)
    assert "FRONTIER_PROVIDER" in message
    assert "FRONTIER_MODEL" in message
    assert "claude-sonnet-4-20250514" in message


def test_the_mismatch_is_refused_in_both_directions(monkeypatch):
    """Symmetry matters: switching back to Anthropic over a Gemini model id is the same bug."""
    monkeypatch.setenv("FRONTIER_PROVIDER", "anthropic")
    monkeypatch.setenv("FRONTIER_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    with pytest.raises(ConfigError, match="gemini-3.6-flash"):
        load_frontier_adapter()


def test_the_mismatch_error_says_which_vendor_the_model_belongs_to(monkeypatch):
    """So the fix is readable from the message without consulting the table."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(ConfigError) as exc:
        GeminiFrontierAdapter(model_id="claude-sonnet-4-20250514")
    assert "that is a anthropic model id" in str(exc.value)


def test_an_unrecognised_model_id_is_allowed_through(monkeypatch):
    """A prefix table cannot keep up with model releases, so it only rejects a positively
    wrong vendor. An unfamiliar id reaches the provider, which is the authority on it."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    adapter = GeminiFrontierAdapter(model_id="gemini-4.0-something-unreleased")
    assert adapter.model_id == "gemini-4.0-something-unreleased"


def test_gemini_frontier_reports_its_missing_key_by_name(monkeypatch):
    monkeypatch.setenv("FRONTIER_PROVIDER", "gemini")
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError) as exc:
        load_frontier_adapter()
    assert "GEMINI_API_KEY" in str(exc.value)


def test_resolving_the_frontier_model_needs_no_credential(monkeypatch):
    """A surface that names the model it is about to use has to read the id without a key.

    `app.py` labels its arm selector before any session exists, and building an adapter to
    read `model_id` off it would demand a credential for a label — so the label would be a
    second place a missing key can fail, ahead of the one that reports it properly.
    """
    monkeypatch.setenv("FRONTIER_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)

    assert resolved_frontier_model() == ("gemini", "gemini-3.6-flash")


def test_resolving_the_frontier_model_reads_both_variables(monkeypatch):
    monkeypatch.setenv("FRONTIER_PROVIDER", "anthropic")
    monkeypatch.setenv("FRONTIER_MODEL", "claude-3-5-sonnet-20241022")

    assert resolved_frontier_model() == ("anthropic", "claude-3-5-sonnet-20241022")


def test_resolving_the_frontier_model_rejects_an_unknown_provider(monkeypatch):
    """Same refusal as `load_frontier_adapter`: a label naming a host we cannot reach would
    describe a run that is about to fail."""
    monkeypatch.setenv("FRONTIER_PROVIDER", "bedrock")

    with pytest.raises(ConfigError) as exc:
        resolved_frontier_model()
    assert "bedrock" in str(exc.value)


def test_the_resolved_frontier_id_is_the_one_the_adapter_sends(monkeypatch):
    """The regression this resolver exists for, held for every provider.

    A display path that derived the default model id separately from the adapter is how
    `app.py` came to label a Gemini run "Frontier (Claude)". Asserting the two agree per
    provider is what makes the label evidence about the request rather than a caption.
    """
    from agent.models.frontier import PROVIDERS

    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    for provider in PROVIDERS:
        resolved = resolved_frontier_model(provider)
        assert resolved == (provider, load_frontier_adapter(provider).model_id)


def test_resolving_the_oss_model_needs_no_credential(monkeypatch):
    monkeypatch.setenv("OSS_PROVIDER", "together")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("OSS_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")

    assert resolved_oss_model() == ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo")


def test_the_resolved_oss_id_is_the_one_the_adapter_sends(monkeypatch):
    monkeypatch.delenv("OSS_MODEL", raising=False)
    monkeypatch.delenv("OSS_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")

    assert resolved_oss_model() == ("groq", OSSAdapter().model_id)


def test_resolving_the_oss_model_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("OSS_PROVIDER", "vllm-on-my-laptop")

    with pytest.raises(ConfigError, match="vllm-on-my-laptop"):
        resolved_oss_model()


def test_load_agent_model_rejects_the_judge_role():
    with pytest.raises(ConfigError, match="judge"):
        load_agent_model("judge")


# --------------------------------------------------------------------------------------
# Cross-module wiring
# --------------------------------------------------------------------------------------


def test_every_module_imports():
    """Catch dangling imports left by a rename.

    Most modules here are stubs no test exercises yet, so a renamed symbol can break their
    imports without a single test failing — which is how `JudgeModel` survived becoming
    `JudgeAdapter` until someone ran the CLI.
    """
    import importlib
    import importlib.util
    import pkgutil

    import agent
    import evals

    modules = ["agent", "evals"]
    packages = [agent, evals]
    if importlib.util.find_spec("streamlit") is not None:
        # `app.py` and `ui/` need the optional [app] extra. Whether it is installed is a local
        # choice, so its absence must not fail the suite; a dangling import inside either is still
        # a bug, which is why they are covered whenever the extra is there. `ui/pages/*.py` are
        # scripts Streamlit runs rather than modules anything imports, so this walk is the only
        # thing that would notice a rename breaking one of their imports.
        import ui

        modules.append("app")
        packages.append(ui)
    for package in packages:
        modules += [
            name for _, name, _ in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
        ]

    for name in modules:
        importlib.import_module(name)


# --------------------------------------------------------------------------------------
# Judge family guard
# --------------------------------------------------------------------------------------


def test_distinct_families_pass():
    agents = [frontier(Recorder(anthropic_ok())), oss(Recorder(openai_ok()))]
    assert_distinct_families(agents, judge(Recorder(openai_ok())))


def test_judge_sharing_a_family_with_an_agent_raises():
    """A judge scoring its own family inflates that arm for reasons unrelated to quality."""
    agents = [frontier(Recorder(anthropic_ok())), oss(Recorder(openai_ok()))]
    same_family_judge = judge(Recorder(openai_ok()))
    same_family_judge.family = "llama"  # as if someone pointed the judge at the OSS family

    with pytest.raises(ConfigError) as exc:
        assert_distinct_families(agents, same_family_judge)
    assert "self-preference" in str(exc.value)
    assert "oss" in str(exc.value)
