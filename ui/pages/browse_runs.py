"""Every run found on disk, of every kind, with the conditions it was executed under.

A table of manifests rather than of results, and all three `run_kind`s rather than the eval runs
alone. A chat session and a judge run are runs someone paid for and can be asked about later; a
browser that listed only the ones this UI can summarise would make the others hard to find and
easy to mistake for missing.

The columns are provenance. A run id on its own says nothing about whether two rows can be
compared, so the model, the provider, the dataset, the item count, and the commit are here beside
it — and `git_dirty` is spelled out rather than symbolised, because a dirty tree means the `git_sha`
in the row does not identify the code that ran.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from ui.data import RunRef
from ui.layout import (
    DETAIL_RUN_KEY,
    READ_ONLY_NOTE,
    configure_page,
    dirty_runs,
    fmt_optional,
    git_marker,
    load_runs,
)

#: The table's columns, in display order, as `(heading, how to read one run)`. Named here so the
#: columns a test asks for are the columns the page has, rather than both sides listing them.
COLUMNS: tuple[tuple[str, Callable[[RunRef], str]], ...] = (
    ("run_id", lambda run: run.run_id),
    ("run_kind", lambda run: run.manifest.run_kind),
    ("model", lambda run: run.manifest.model_name),
    ("provider", lambda run: run.manifest.provider),
    ("dataset", lambda run: fmt_optional(run.manifest.dataset_path)),
    ("n_items", lambda run: fmt_optional(run.manifest.n_items)),
    ("started_at", lambda run: run.manifest.started_at),
    ("judge_run", lambda run: fmt_optional(run.judge_run_id)),
    ("git_sha", lambda run: fmt_optional(run.manifest.git_sha)),
    ("git_dirty", lambda run: git_marker(run.manifest)),
    ("runs_dir", lambda run: str(run.runs_dir)),
)


#: How those columns are drawn. Display only, and the headings stay the field names: these are
#: manifest fields, and a reader who has one open beside the table should not have to translate. The
#: `help` text is what the columns mean, which was otherwise only in this module's docstring.
COLUMN_CONFIG: dict[str, Any] = {
    "run_id": st.column_config.TextColumn("run_id", width="medium", pinned=True),
    "run_kind": st.column_config.TextColumn(
        "run_kind", width="small", help="eval, judge, or chat. Only an eval run has a summary."
    ),
    "model": st.column_config.TextColumn("model", width="medium"),
    "dataset": st.column_config.TextColumn(
        "dataset", width="medium", help="Digested in the manifest as dataset_sha256."
    ),
    "n_items": st.column_config.TextColumn("n_items", width="small"),
    "judge_run": st.column_config.TextColumn(
        "judge_run",
        width="medium",
        help="Joined through the judge manifest's recorded pairs_path, not by matching filenames.",
    ),
    "git_sha": st.column_config.TextColumn("git_sha", width="small"),
    "git_dirty": st.column_config.TextColumn(
        "git_dirty",
        width="medium",
        help="A dirty tree means the git_sha in this row does not identify the code that ran, so "
        "nothing measured under it can be reproduced from that commit alone.",
    ),
}


def run_table(runs: list[RunRef]) -> list[dict[str, str]]:
    """The rows `st.dataframe` draws, as plain dicts.

    Plain dicts rather than a DataFrame of our own: Streamlit builds one either way, and importing
    pandas here would add a dependency to read a manifest.
    """
    return [{heading: read(run) for heading, read in COLUMNS} for run in runs]


def main() -> None:
    """Render the table, with the provenance caveats above it rather than beneath."""
    configure_page("Browse runs")
    st.title("Runs")
    st.caption(READ_ONLY_NOTE)

    runs, root = load_runs()
    if not runs:
        st.warning(f"No manifest under `{root}`.")
        return

    dirty = dirty_runs(runs)
    if dirty:
        st.warning(
            "Executed from a dirty working tree, so the `git_sha` in the row does not identify "
            f"the code that produced the run: {', '.join(sorted(dirty))}. Anything measured here "
            "cannot be reproduced from that commit alone."
        )

    st.dataframe(run_table(runs), width="stretch", hide_index=True, column_config=COLUMN_CONFIG)
    st.caption(
        f"{len(runs)} run(s) under `{root}`, newest first. `judge_run` is joined through the judge "
        "manifest's recorded `pairs_path`, not by matching filenames."
    )

    evaluated = [run for run in runs if run.is_eval]
    st.subheader("Open one in detail")
    st.caption(
        "Eval runs only. A chat session was scored against no dataset and a judge run has no "
        "agent under test, so neither has a summary to show."
    )
    if not evaluated:
        st.info("None of these runs is an eval run.")
        return
    chosen = st.selectbox(
        "Eval run",
        [run.run_id for run in evaluated],
        format_func=lambda run_id: _describe(evaluated, run_id),
        key=DETAIL_RUN_KEY,
    )
    # A shared session-state key rather than a link: the same key backs the selector on the detail
    # page, so choosing here and navigating there shows this run rather than resetting to the first
    # one — and one key means the two pages cannot disagree about which run is open.
    st.info(f"**{chosen}** is selected. Open **Run detail** in the sidebar to see it.")


def _describe(runs: list[RunRef], run_id: str) -> str:
    """`run_id — model, dataset` for the selector, so the ids are not chosen blind."""
    run = next(one for one in runs if one.run_id == run_id)
    return f"{run_id} — {run.manifest.model_name}, {run.manifest.dataset_path}"


if __name__ == "__main__":
    main()
