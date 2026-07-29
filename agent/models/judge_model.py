"""Judge adapter: a third model family, different from both agents.

Locked decision (PROJECT.md). If the agents are Claude and Llama, the judge is a third family
— by default a GPT-4o-class model via `JUDGE_MODEL` and `OPENAI_API_KEY`. LLM judges reliably
favour text from their own family, so a judge sharing a family with a candidate would inflate
that candidate's scores for reasons unrelated to answer quality.

`agent.models.base.assert_distinct_families` enforces this at run start and reads `family`
below, so that attribute must name the real provider family rather than a label.

The judge is scoring machinery, not an agent: it has no tools, no memory, and no retrieval,
and it runs at `JUDGE_TEMPERATURE` (0 by default) so that re-scoring a fixed set of
(prompt, response) pairs reproduces.
"""

from __future__ import annotations

import os
from typing import Any

from agent.models.base import OpenAICompatibleAdapter, require_env

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-11-20"


class JudgeAdapter(OpenAICompatibleAdapter):
    """Chat adapter for the judge, deliberately distinct from both agents' families."""

    name = "judge"
    family = "openai"
    provider = "openai"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        *,
        base_url: str = OPENAI_BASE_URL,
        **kwargs: Any,
    ) -> None:
        """Read `JUDGE_MODEL` / `OPENAI_API_KEY` when arguments are omitted."""
        self.base_url = base_url
        super().__init__(
            model_id=model_id or os.environ.get("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
            api_key=api_key or require_env("OPENAI_API_KEY"),
            **kwargs,
        )


def load_judge_model(**kwargs: Any) -> JudgeAdapter:
    """Build the judge adapter from environment configuration."""
    return JudgeAdapter(**kwargs)
