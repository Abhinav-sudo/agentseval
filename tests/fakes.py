"""Test doubles shared across test modules.

`FakeAdapter` is a scripted `ModelAdapter` for testing the loop without a provider, used by
`test_core.py` and `test_memory.py`, which both need to drive the agent through an exact
sequence of completions and then inspect the messages it sent.

`refuse_env_load` stands in for `base.load_env` in the CLI tests, which each assert the same
property about a different entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.models.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ChatMessage,
    FinishReason,
    ModelResponse,
)


class EnvLoaded(Exception):
    """Raised by `refuse_env_load`; carries no message because nothing reads one."""


def refuse_env_load() -> None:
    """Stand in for `base.load_env`, raising instead of returning.

    Patched over the `load_env` an entry point imported, it proves both that `main` calls it and
    that it calls it before anything else — argv parsing included, so a future argparse default
    that reads the environment would see a loaded `.env` rather than an empty one. Asserting the
    ordering needs the stub to abort; asserting only the call would pass on a `main` that read
    its configuration first and loaded `.env` afterwards.
    """
    raise EnvLoaded


@dataclass(frozen=True)
class Reply:
    """A scripted completion whose stop reason is not the boring one.

    Tests that only care about the text pass a bare string; this exists for the ones that
    need to distinguish a model that broke the protocol from one we cut off at `max_tokens`.
    """

    text: str
    finish_reason: FinishReason = FinishReason.LENGTH


ScriptEntry = str | Reply | Exception


@dataclass
class FakeAdapter:
    """Replays scripted completions and records every request it was handed.

    An entry that is an exception is raised instead of returned, which is how a provider
    failure is simulated. The last entry repeats once the script runs out, so "the model keeps
    calling tools" is one entry rather than a padded list — the same convention as the mock
    transport in `test_models.py`.
    """

    completions: list[ScriptEntry]
    name: str = "fake"
    model_id: str = "fake-model-1"
    family: str = "fake"
    provider: str = "fake"
    latency_ms: float = 12.0
    prompt_tokens: int | None = 100
    completion_tokens: int | None = 20
    # 0 rather than None, so the default fake behaves like a provider that does not meter
    # thinking separately. A test about reasoning tokens sets it.
    reasoning_tokens: int | None = 0
    usd_cost: float | None = 0.001
    cached: bool = False
    # False by default because nothing here reads a cache: replies come from the script. It is
    # here so a caller that inspects `use_cache` before trusting a measurement — as
    # `judge.sample_verdicts` does, since identical requests share one cache key — sees an
    # honest answer, and so a test can set it True to prove that refusal fires.
    use_cache: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop: list[str] | None = None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
            }
        )
        script = self.completions
        entry = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(entry, Exception):
            raise entry
        reply = entry if isinstance(entry, Reply) else Reply(entry, FinishReason.COMPLETE)
        return ModelResponse(
            text=reply.text,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_ms=self.latency_ms,
            usd_cost=self.usd_cost,
            raw={"fake": True},
            reasoning_tokens=self.reasoning_tokens,
            cached=self.cached,
            finish_reason=reply.finish_reason,
        )

    @property
    def count(self) -> int:
        return len(self.calls)

    def messages(self, index: int = -1) -> list[ChatMessage]:
        """Messages sent on one call, defaulting to the most recent."""
        return self.calls[index]["messages"]

    def prompt(self, index: int = -1) -> str:
        """Every message from one call, flattened, for substring assertions."""
        return "\n".join(message["content"] for message in self.messages(index))
