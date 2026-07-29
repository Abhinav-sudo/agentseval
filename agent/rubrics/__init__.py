"""Judge rubric text: `default.md` plus one file per `evals.schema.Axis`.

A package rather than a bare data directory so `importlib.resources` reaches these files in an
installed wheel exactly as it does in a source checkout — a judge whose rubrics did not ship
would fail at the point of scoring, which is the worst place to discover it.

Every file here is prose. Nothing executable lives in this package: the loader, the refusals,
the rendered JSON schema, the anchor validation, and the digest are all in `agent.prompts`
(PROJECT.md, "The judge rubric lives on disk").

Adding a rubric means adding two files, `{name}.md` and `{name}.anchors.json`, and the loader
will refuse either one on its own. Read `agent.prompts.judge_rubric_prompt` before writing one:
it enforces the placeholders, the four dimensions, and the sentences every rubric shares.
"""
