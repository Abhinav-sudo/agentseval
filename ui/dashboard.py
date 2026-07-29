"""Entry script for the platform's read-only views: `streamlit run ui/dashboard.py`.

A second entry script rather than more pages on `app.py`, because PROJECT.md says the chat app is
a demo surface and not the product, and eval views bolted onto it would make that false. Streamlit
looks for `pages/` beside its entry script, so both live under `ui/`: a repo-root `pages/` would be
picked up by `app.py` too, which is the thing being avoided.

This page is a landing page and holds no figures of its own. The views are in `ui/pages/`.
"""

from __future__ import annotations

import streamlit as st

from evals.metrics import THRESHOLD_CUTS
from ui.data import eval_runs
from ui.layout import READ_ONLY_NOTE, configure_page, dirty_runs, load_runs


def main() -> None:
    """Render the landing page: what this is, what it found, and what it will not do."""
    configure_page("Runs")
    st.title("AgentsEval")
    st.caption(
        "Read-only views over the evaluation platform. The chat surface is `app.py`, which is a "
        "demo; this is the platform."
    )
    st.info(READ_ONLY_NOTE)

    runs, root = load_runs()
    evals_found = eval_runs(runs)

    columns = st.columns(3)
    columns[0].metric("Runs found", len(runs), border=True)
    columns[1].metric("Eval runs", len(evals_found), border=True)
    columns[2].metric(
        "With a judge run", sum(1 for run in evals_found if run.judge_run_id), border=True
    )
    st.caption(f"searched `{root}` recursively for `*.manifest.json`")

    if not runs:
        st.warning(
            f"No manifest under `{root}`. Point the box in the sidebar at a runs directory, or "
            "execute a run with `agentseval-run`."
        )

    dirty = dirty_runs(runs)
    if dirty:
        st.warning(
            f"{len(dirty)} run(s) were executed from a dirty working tree, so their `git_sha` "
            f"does not identify the code that produced them: {', '.join(sorted(dirty))}"
        )

    st.subheader("Pages")
    st.markdown(
        "- **Browse runs** — every manifest found, of every kind, with the conditions each run "
        "was executed under.\n"
        "- **Run detail** — one eval run: judge dimensions, deterministic checks, the axis rates "
        f"at all {len(THRESHOLD_CUTS)} pre-registered cuts, and what the run cost.\n"
        "- **Chat history** — a past conversation from the chat surface, read back out of its "
        "trace. One segment per run, since a conversation that crossed a model switch was "
        "recorded under a manifest each."
    )

    st.subheader("What is deliberately not here")
    st.markdown(
        "- **Labelling.** It stays in the terminal (`agentseval-label`), which keeps "
        "counterfactual variants apart. A side-by-side web form would have an annotator label the "
        "comparison rather than the response, which is the judgement the within-pair delta exists "
        "to make independently.\n"
        "- **Launching a run.** These pages spend nothing; a run spends credits and mints a "
        "manifest.\n"
        "- **Annotator fields.** `expected_behavior` and `notes` are instructions to a human and "
        "are shown to nobody here, for the same reason they are never shown to a model."
    )


if __name__ == "__main__":
    main()
