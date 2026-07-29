"""OSS agent adapter: Llama 3.1 8B Instant via a hosted provider (Groq/Together).

Locked decision (PROJECT.md). `OSS_PROVIDER` selects the host (`groq` or `together`) and
`OSS_MODEL` the model string; both expose an OpenAI-compatible chat endpoint, so the wire
format comes from `OpenAICompatibleAdapter` and only the base URL and credential differ.

This arm shares the prompt-based JSON tool protocol with the frontier arm. Expect this model
to emit malformed or fenced JSON more often. That is a genuine finding about the model, and
it surfaces in the trace as a parse failure rather than being patched over here — the
comparison is only meaningful if both arms get identical treatment.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from agent.models.base import ConfigError, OpenAICompatibleAdapter, require_env

Provider = Literal["groq", "together"]

#: Base URL and credential per hosted provider. The same weights can differ in quantisation
#: between hosts, which is why the resolved provider goes into the run manifest.
PROVIDERS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
}

DEFAULT_OSS_MODEL = "llama-3.1-8b-instant"
DEFAULT_PROVIDER = "groq"


class OSSAdapter(OpenAICompatibleAdapter):
    """Llama 3.1 8B Instant behind an OpenAI-compatible hosted endpoint."""

    name = "oss"
    family = "llama"

    def __init__(
        self,
        model_id: str | None = None,
        provider: Provider | str | None = None,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Read `OSS_MODEL` / `OSS_PROVIDER` and the provider's key when omitted.

        Raises:
            ConfigError: `OSS_PROVIDER` names a host we have no configuration for. Failing
                here beats defaulting to Groq, which would silently produce a run whose
                manifest disagreed with what actually served it.
        """
        resolved = (provider or os.environ.get("OSS_PROVIDER") or DEFAULT_PROVIDER).lower()
        if resolved not in PROVIDERS:
            known = ", ".join(sorted(PROVIDERS))
            raise ConfigError(f"Unknown OSS_PROVIDER {resolved!r}; expected one of: {known}")

        default_url, key_var = PROVIDERS[resolved]
        self.provider = resolved
        self.base_url = base_url or default_url
        super().__init__(
            model_id=model_id or os.environ.get("OSS_MODEL") or DEFAULT_OSS_MODEL,
            api_key=api_key or require_env(key_var),
            **kwargs,
        )
