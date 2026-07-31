"""The committed reports, rendered as they were written.

Every other page here computes: it joins a trace to its judgements and its dataset and draws the
result, so what it shows is true at the moment it is read. This page does the opposite and shows a
file — a report `agentseval-report` wrote and somebody committed — and the distinction is the reason
the page exists at all. `runs/` is gitignored, so a trace never leaves this repository; a report is
the one artifact of a run that does, which makes it the only thing a reader without the runs on
their disk can be shown. That includes the reader looking at this repository on GitHub, where none
of the other four pages exist.

Which also makes it the only page that can be wrong. A committed report was true of a run at the
commit that wrote it, and nothing keeps it in step with a trace that has since been re-judged. So
the page says so, beside the report rather than in a docstring, and names Run detail as the view
that recomputes. Nothing here re-derives a figure from the markdown: the numbers are text by the
time they arrive, and a page that parsed them back into figures would be a second answer to what
the run said, computed from a worse copy of the data than the trace it came from.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from evals.report import DEFAULT_REPORTS_DIR
from ui.data import discover_reports, report_text
from ui.layout import configure_page, reports_root

#: What this page is and is not, printed above the report. `READ_ONLY_NOTE` is the other pages'
#: version of this and would be wrong here in its second half: they recompute from the trace on
#: every read, and this one reads a file that was computed once.
SNAPSHOT_NOTE = (
    "A report is a snapshot. It was recomputed from a trace, the judgements, and the dataset at "
    "the commit that wrote it, and nothing keeps it in step with a run that has since been "
    "re-judged. **Run detail** recomputes on every read and is the authority where they disagree."
)

#: Shown when the directory holds nothing, with the command that fixes it. A page that said only
#: "no reports found" would leave a reader to discover that the CLI exists.
EMPTY_NOTE = (
    "No `*.md` under `{root}`. Write one with `agentseval-report <run_id>`, which reports on a run "
    "in `runs/` and writes to `{default}/<run_id>.md` — the short report by default, every "
    "breakdown with `--full`."
)


def report_label(path: Path) -> str:
    """How a report is named in the picker: its filename, which is a run id by default.

    The filename and not a heading parsed out of the file. `--out` accepts any path, so the name is
    whatever the person who wrote it chose, and showing them something else — a title lifted from
    inside — would make the file they are looking for hard to find in a directory listing.
    """
    return path.name


def main() -> None:
    """Pick a committed report and render it."""
    configure_page("Report")
    st.title("Report")
    st.caption(
        "Markdown reports written by `agentseval-report` and committed to the repository. This is "
        "the form a result takes outside these pages, since `runs/` is not tracked by git."
    )
    st.info(SNAPSHOT_NOTE)

    root = reports_root()
    reports = discover_reports(root)
    if not reports:
        st.warning(EMPTY_NOTE.format(root=root, default=DEFAULT_REPORTS_DIR))
        return

    # A picker even for one report, so the filename of what is being read is always on the page. A
    # report renders under its own `#` heading, which names a run and not the file it lives in.
    chosen = st.selectbox(
        "Report",
        reports,
        format_func=report_label,
        help="One file per report. The default filename is the run id it reports on.",
    )
    st.caption(f"`{chosen}`")

    try:
        markdown = report_text(chosen)
    except OSError as exc:
        # Most likely deleted or regenerated between the listing and the read, which a rerun fixes.
        st.error(f"Cannot read {chosen}: {exc}")
        return

    st.divider()
    st.markdown(markdown)


if __name__ == "__main__":
    main()
