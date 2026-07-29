"""Chrome shared by the pages: page setup, the runs-root control, and the discovery banners.

Widgets only. Anything that reads a file is in `ui.data`, and anything that decides what a number
means is in `evals/` — this module exists so that three scripts do not each grow their own copy of
the sidebar and their own idea of what to do about a manifest that would not parse.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent.manifest import RunManifest
from ui.data import DEFAULT_RUNS_ROOT, RunRef, discover_runs

#: Session-state key behind the runs-root box, so the three pages share one value and switching
#: page does not silently switch which runs are being read.
RUNS_ROOT_KEY = "runs_root"

#: Session-state key naming the run the detail page is showing. Shared so a test — or a later
#: revision with a link from the table — can say which run to open without a URL scheme.
DETAIL_RUN_KEY = "detail_run_id"

#: What a dirty working tree is called in the table. Spelled out rather than a symbol: `git_sha`
#: identifies a commit, and an uncommitted edit means the code that ran was not that commit. A
#: reader who does not already know that has to be told it here, where the number is.
DIRTY_MARKER = "DIRTY — uncommitted changes, git_sha does not identify the code that ran"
CLEAN_MARKER = "clean"

#: Printed on every page. These views spend nothing and can be pointed at an old trace safely,
#: which is worth saying once per page rather than leaving to be inferred.
READ_ONLY_NOTE = (
    "Reads `runs/` only: no model calls, no API keys, no cost, and nothing written. Every figure "
    "is recomputed from the trace, the judgements, and the dataset on demand — there is no "
    "derived results file to go stale."
)


def configure_page(title: str) -> None:
    """Page title, icon, mark, and layout. Called by each script, as Streamlit expects.

    An emoji favicon rather than a Material icon, which Streamlit renders black for a favicon
    whatever the browser's theme is and which therefore disappears into a dark tab strip. The
    sidebar mark is a Material icon, since that one is drawn inside the app and takes its colour
    from the theme.
    """
    st.set_page_config(page_title=f"AgentsEval — {title}", page_icon="📊", layout="wide")
    st.logo(":material/monitoring:", size="large")


def runs_root() -> Path:
    """The directory the pages read, from a sidebar box shared across them.

    A box rather than a fixed constant because `--runs-dir` exists on the CLIs and a run can
    therefore be anywhere; the default is the same `runs/` those CLIs default to.
    """
    with st.sidebar:
        st.header("Runs")
        value = st.text_input(
            "Runs directory",
            value=str(DEFAULT_RUNS_ROOT),
            key=RUNS_ROOT_KEY,
            help="Searched recursively, so runs in subdirectories such as runs/pilot/ are found.",
        )
    return Path(value or DEFAULT_RUNS_ROOT)


def load_runs() -> tuple[list[RunRef], Path]:
    """Discover runs under the current root, rendering whatever went wrong on the way.

    The problems are drawn rather than returned, because every caller does the same thing with
    them and a caller that forgot would be hiding a file someone meant to be a run.
    """
    root = runs_root()
    runs, problems = discover_runs(root)
    for problem in problems:
        st.warning(f"Unreadable manifest — {problem}")
    for run in runs:
        if run.judge_error is not None:
            st.warning(f"{run.run_id}: {run.judge_error}")
    return runs, root


def git_marker(manifest: RunManifest) -> str:
    """`DIRTY_MARKER` or `CLEAN_MARKER` for a manifest, or the unknown case spelled out.

    None means the manifest predates the field, which is not the same as a clean tree and is not
    reported as one.
    """
    if manifest.git_dirty is None:
        return "unknown — this manifest predates the field"
    return DIRTY_MARKER if manifest.git_dirty else CLEAN_MARKER


def dirty_runs(runs: list[RunRef]) -> list[str]:
    """The ids of runs whose working tree was dirty. Empty is the ordinary case."""
    return [run.run_id for run in runs if run.manifest.git_dirty]


def fmt_optional(value: object, *, missing: str = "—") -> str:
    """A manifest field for a table cell, with None shown as absent rather than as a value."""
    return missing if value is None else str(value)
