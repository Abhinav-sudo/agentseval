"""Frontier agent adapter: Gemini via Google, or Claude Sonnet via Anthropic.

Locked decision (PROJECT.md): the frontier arm is one current-generation model from a frontier
lab — currently Gemini 3.6 Flash, with Claude Sonnet and a GPT-4o-class model as the permitted
alternatives. `FRONTIER_PROVIDER` selects the host and `FRONTIER_MODEL` that host's own model
string.

No adapter here may send a `tools` array, even though both APIs support it and would parse
tool calls far more reliably than our prompt protocol does. The OSS agent has no equivalent,
so using it here would mean the two arms no longer share a harness and the score gap would
mix model quality with harness quality. See PROJECT.md.

Only the wire format lives here. Retries, caching, timing, and cost come from
`ChatAdapter`, so the frontier arm cannot acquire behaviour the OSS arm lacks.

Unlike `oss.PROVIDERS`, which maps a name onto a base URL because Groq and Together speak one
wire format, this maps a name onto an *adapter class*: Anthropic's messages API and the
OpenAI-compatible surface Google exposes for Gemini are different protocols, so a shared
`base_url` swap could not reach both. The selection is still one env var resolved in one
place, and the resolved provider still reaches the run manifest.

The Gemini arm deliberately rides `OpenAICompatibleAdapter` unmodified rather than Gemini's
native `generateContent`. The compatibility surface is the same body builder the OSS arm and
the judge already use, so the prompt-based JSON protocol and the ban on native tool calling
hold by construction here instead of by remembering to re-implement them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from agent.models.base import (
    ChatAdapter,
    ChatMessage,
    ConfigError,
    FinishReason,
    ModelError,
    OpenAICompatibleAdapter,
    ParsedPayload,
    require_env,
)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_FRONTIER_MODEL = "claude-sonnet-4-20250514"

#: Google's OpenAI-compatibility base. `OpenAICompatibleAdapter._endpoint` appends
#: `/chat/completions`, giving the documented compat path.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

#: Current Flash generation. Not `gemini-2.5-flash`: Google returns 404 "no longer available to
#: new users" for it, and `/v1beta/openai/models` still lists it, so the model listing cannot be
#: trusted to say what a new key may actually call.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

Provider = Literal["anthropic", "gemini"]

#: Anthropic stays the default so an existing `.env` that names only `ANTHROPIC_API_KEY` and
#: `FRONTIER_MODEL` keeps resolving to the arm it always did.
DEFAULT_PROVIDER = "anthropic"

#: Each provider's own default model, for an `.env` that names a provider and no model. Read by
#: the adapters and by `resolved_frontier_model` alike, so a surface that wants to *name* the
#: model without building an adapter cannot derive a different id from the one that is sent.
DEFAULT_MODELS: Mapping[str, str] = MappingProxyType(
    {
        "anthropic": DEFAULT_FRONTIER_MODEL,
        "gemini": DEFAULT_GEMINI_MODEL,
    }
)

#: Model-id prefixes each provider will actually serve.
#:
#: `FRONTIER_PROVIDER` and `FRONTIER_MODEL` have to move together, and nothing about setting one
#: reminds you to set the other: flipping the provider to `gemini` over an `.env` still naming
#: `claude-sonnet-4-20250514` sends a Claude id to Google, which answers `404 not found for API
#: version v1main` — a message that reads like the model was retired rather than like a
#: misconfiguration, and costs a debugging session to tell apart. Checking the pair here turns
#: that into one sentence naming both variables, before any request is sent.
MODEL_PREFIXES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "anthropic": ("claude-",),
        "gemini": ("gemini-",),
    }
)


def check_model_matches_provider(provider: str, model_id: str) -> None:
    """Raise unless `model_id` is one `provider` can serve.

    A prefix check and deliberately not a list of known model ids: models are released faster
    than this table would be updated, and a guard that rejected a valid new model would be
    worse than the 404 it replaces. It only has to catch the pair being from different vendors.

    Raises:
        ConfigError: the model id belongs to a different provider.
    """
    prefixes = MODEL_PREFIXES.get(provider)
    if prefixes is None or model_id.startswith(prefixes):
        return
    owner = next(
        (name for name, known in MODEL_PREFIXES.items() if model_id.startswith(known)), None
    )
    belongs = f" — that is a {owner} model id" if owner else ""
    raise ConfigError(
        f"FRONTIER_PROVIDER={provider!r} cannot serve FRONTIER_MODEL={model_id!r}{belongs}. "
        f"Set FRONTIER_MODEL to a {provider} model (expected prefix: "
        f"{' or '.join(prefixes)}), or change FRONTIER_PROVIDER to match the model."
    )


#: Anthropic's `stop_reason` vocabulary. `tool_use` is absent deliberately: `_body` never
#: sends a `tools` array, so it would be a provider surprise, and `OTHER` surfaces it.
ANTHROPIC_STOP_REASONS: Mapping[str, FinishReason] = {
    "end_turn": FinishReason.COMPLETE,
    "max_tokens": FinishReason.LENGTH,
    "stop_sequence": FinishReason.STOP_SEQUENCE,
    "refusal": FinishReason.CONTENT_FILTER,
}


class FrontierAdapter(ChatAdapter):
    """Anthropic messages API, plain text completions only."""

    name = "frontier"
    family = "anthropic"
    provider = "anthropic"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        *,
        base_url: str = ANTHROPIC_BASE_URL,
        **kwargs: Any,
    ) -> None:
        """Read `FRONTIER_MODEL` / `ANTHROPIC_API_KEY` when arguments are omitted.

        Raises:
            ConfigError: the resolved model id belongs to another provider.
        """
        self.base_url = base_url
        resolved = model_id or resolved_frontier_model(self.provider)[1]
        check_model_matches_provider(self.provider, resolved)
        super().__init__(
            model_id=resolved,
            api_key=api_key or require_env("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _body(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        """Build a plain messages-API body.

        Anthropic takes the system prompt as a top-level field rather than a message, so
        system turns are hoisted out here. Multiple system messages are joined rather than
        dropped: silently discarding prompt text would change what the model saw without
        showing up anywhere.

        No `tools` or `tool_choice` key is ever set.
        """
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        turns = [m for m in messages if m.get("role") != "system"]

        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if stop:
            body["stop_sequences"] = stop
        return body

    def _parse(self, payload: dict[str, Any]) -> ParsedPayload:
        """Extract text, token counts, and stop reason from a messages-API response.

        Text blocks are concatenated; a response whose only block is a non-text type yields
        an empty string, which the trace records as-is rather than treating as a failure.
        """
        try:
            blocks = payload["content"]
            text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        except (KeyError, TypeError) as exc:
            raise ModelError(
                f"{self.name}: unexpected response shape from {self.model_id}: {exc}"
            ) from exc
        usage = payload.get("usage") or {}
        return ParsedPayload(
            text=text,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            finish_reason=FinishReason.normalise(
                payload.get("stop_reason"), ANTHROPIC_STOP_REASONS
            ),
            # Anthropic publishes no total to take a residual from, and needs none: extended
            # thinking is billed as output and already counted in `output_tokens`. A literal
            # 0 is therefore a statement about this wire format, not the absence of one —
            # `derive_reasoning_tokens` would return None here and lose the cost.
            reasoning_tokens=0,
        )


class GeminiFrontierAdapter(OpenAICompatibleAdapter):
    """Gemini through Google's OpenAI-compatible endpoint.

    `_body`, `_parse`, `_endpoint`, and `_headers` are all inherited: the compat surface takes
    a bearer token and returns `choices[0].message.content` with an OpenAI `finish_reason`, so
    there is no wire format to describe beyond a base URL and a credential.

    One hazard worth knowing when reading a trace from this arm. Gemini 2.5 models spend
    *thinking* tokens, which the compat layer bills and counts as completion tokens, and
    nothing here raises `max_tokens` or sends `reasoning_effort` to compensate — the body must
    stay the plain chat completion the OSS arm sends, or the arms stop sharing a harness. A
    reply whose thinking consumed the ceiling therefore arrives as a `LENGTH` finish, which
    `agent.core` books as `TRUNCATED`/`budget_induced` and keeps out of the contract-violation
    rate. That is the correct accounting — our ceiling, not the model's formatting — but it
    means `budget_induced_truncations` is the field to read first if this arm looks unusually
    bad at the protocol.
    """

    name = "frontier"
    family = "gemini"
    provider = "gemini"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        *,
        base_url: str = GEMINI_BASE_URL,
        **kwargs: Any,
    ) -> None:
        """Read `FRONTIER_MODEL` / `GEMINI_API_KEY` when arguments are omitted.

        Raises:
            ConfigError: the resolved model id belongs to another provider — the likely case
                being a `FRONTIER_MODEL` left at its Claude value while only the provider was
                switched.
        """
        self.base_url = base_url
        resolved = model_id or resolved_frontier_model(self.provider)[1]
        check_model_matches_provider(self.provider, resolved)
        super().__init__(
            model_id=resolved,
            api_key=api_key or require_env("GEMINI_API_KEY"),
            **kwargs,
        )


#: Provider name to adapter class. A class rather than a `(base_url, key_var)` pair because
#: the two providers do not share a wire format; see the module docstring.
PROVIDERS: dict[str, type[ChatAdapter]] = {
    "anthropic": FrontierAdapter,
    "gemini": GeminiFrontierAdapter,
}


def resolved_frontier_model(provider: Provider | str | None = None) -> tuple[str, str]:
    """Return the `(provider, model_id)` this environment selects, without building anything.

    What `load_frontier_adapter` is about to do, answered without a credential — which is the
    point. A surface that displays the model under test has to read the id before a session
    exists (`app.py` labels its arm selector above the code that builds one), and constructing
    an adapter for a label would make the label a second place a missing key fails, ahead of
    the one that reports it properly. Deriving the id independently instead is how a Gemini run
    came to be labelled "Frontier (Claude)".

    Raises:
        ConfigError: `FRONTIER_PROVIDER` names a host we have no configuration for. Refused
            here as well as in the loader, since a label naming an unreachable host would
            describe a run that cannot happen.
    """
    resolved = (provider or os.environ.get("FRONTIER_PROVIDER") or DEFAULT_PROVIDER).lower()
    if resolved not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"Unknown FRONTIER_PROVIDER {resolved!r}; expected one of: {known}")
    return resolved, os.environ.get("FRONTIER_MODEL") or DEFAULT_MODELS[resolved]


def load_frontier_adapter(
    provider: Provider | str | None = None,
    **kwargs: Any,
) -> ChatAdapter:
    """Build the frontier adapter named by `FRONTIER_PROVIDER`.

    `FRONTIER_MODEL` is the *selected provider's own* model string, exactly as `OSS_MODEL` is
    for `OSS_PROVIDER`, so it has to change when the provider does. Each adapter falls back to
    its own default when the variable is unset, and rejects a model id belonging to the other
    provider — so `FRONTIER_PROVIDER` alone is **not** the whole switch, and setting only it is
    an error with a message saying so rather than a 404 from the vendor.

    Raises:
        ConfigError: `FRONTIER_PROVIDER` names a host we have no configuration for, or names one
            that cannot serve `FRONTIER_MODEL`. Failing here beats defaulting to Anthropic,
            which would silently produce a run whose manifest disagreed with what served it.
    """
    resolved, _ = resolved_frontier_model(provider)
    return PROVIDERS[resolved](**kwargs)
