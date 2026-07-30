"""Model adapters.

Three distinct families behind one interface (`base.ModelAdapter`):

* `frontier` — Gemini via Google's OpenAI-compatible endpoint, or Claude Sonnet via
  Anthropic; selected by `FRONTIER_PROVIDER`.
* `oss`      — Llama 3.1 8B Instant via a hosted provider (Groq/Together).
* `judge_model` — a third family, different from both agents, to avoid self-preference
  bias.

No adapter may expose native function-calling parameters. Tool calling is a prompt-based
JSON protocol, uniform across both agents. Retries, response caching, timing, and cost all
live in `base` rather than in the individual adapters, so neither arm can acquire harness
behaviour the other lacks. See PROJECT.md.
"""
