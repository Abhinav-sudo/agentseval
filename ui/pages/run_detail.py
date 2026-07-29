"""One eval run in full: what invalidates it, what it measured, and what it cost.

The order of this page is an argument. The three figures that decide whether anything below them
can be read at all come first — items our tooling lost, items our token ceiling cut off, items the
run never reached — then the warnings `summarise_run` recorded, then the conditions the run was
executed under, and only then the metrics. A page that opened with a judge mean would be inviting
someone to quote a number over three of sixty items without ever seeing the sixty.

Every row comes from `report.summary_rows`, unfiltered. That is what makes the four threshold cuts,
the empty buckets, and the over-refusal control appear here without this page deciding to include
them: they are properties of the shaper, and a view that re-derived them could re-derive them
differently.

No item text is rendered anywhere on this page. `expected_behavior` and `notes` are instructions
written for a human annotator, and PROJECT.md's rule about them is not a rule about models only.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from evals.metrics import BUDGET_INDUCED_WARNING_THRESHOLD, RunSummary
from evals.report import SECTION_ORDER, SummaryRow, summary_rows
from ui.data import RunRef, eval_runs, summary_for
from ui.layout import (
    DETAIL_RUN_KEY,
    READ_ONLY_NOTE,
    configure_page,
    fmt_optional,
    git_marker,
    load_runs,
)

#: The conditions block, as `(heading, how to read one run)`. The same fields
#: `report.CONDITION_FIELDS` prints above a comparison, minus the digests: a reader comparing two
#: runs by hand needs those, and a reader of one run has nothing to compare them against.
#: How the metric table is drawn: headings, number formats, and the per-column notes that were
#: otherwise a paragraph above the table. Display only — the keys are the ones `section_table`
#: emits, because those are what a reader exports and what the tests ask for, and a heading is not
#: a column name. No progress bars: these rows mix rates on 0–1 with judge means on 1–5, and a bar
#: needs one scale for the column.
METRIC_COLUMNS: dict[str, Any] = {
    "metric": st.column_config.TextColumn(
        "metric key",
        width="medium",
        help="Names a bucket rather than describing one, so it can be filtered and joined back to "
        "the items behind it.",
    ),
    "reading": st.column_config.TextColumn("what it measures", width="medium"),
    "cut": st.column_config.NumberColumn(
        "cut",
        format="%d",
        help="The pre-registered threshold this row is at. Rates with a curve appear at every cut, "
        "so a conclusion that only holds at one of them is visible as one.",
    ),
    "mean": st.column_config.NumberColumn(
        "mean",
        format="%.3f",
        help="Blank where the bucket has no items: a zero with an interval around it would read as "
        "a measurement, and an empty bucket is not one.",
    ),
    "ci_low": st.column_config.NumberColumn("CI low", format="%.3f"),
    "ci_high": st.column_config.NumberColumn("CI high", format="%.3f"),
    "n": st.column_config.NumberColumn(
        "items", format="%d", help="The denominator this figure was computed over."
    ),
    "well-formed only": st.column_config.CheckboxColumn(
        "well-formed only",
        help="Conditioned on responses that parsed, which conditions on the model's own success.",
    ),
    "note": st.column_config.TextColumn("note", width="large"),
}

#: The same idea for the conditions block: three columns, the last of which is prose.
CONDITION_COLUMNS: dict[str, Any] = {
    "condition": st.column_config.TextColumn("condition", width="small"),
    "value": st.column_config.TextColumn("value", width="medium"),
    "what it is": st.column_config.TextColumn("what it is", width="large"),
}

CONDITIONS: tuple[tuple[str, str], ...] = (
    ("run_id", "which run these figures are from"),
    ("model", "the model under test"),
    ("judge_run_id", "the judge run that scored it, joined through its recorded pairs_path"),
    ("n_cases", "items in the eval set"),
    ("n_scored", "items that entered scoring"),
    ("n_wellformed", "items whose response parsed under the protocol"),
    ("n_substituted", "items whose answer a guardrail replaced"),
    ("n_unjudged", "items with no parsed judgement"),
)


def promoted_figures(summary: RunSummary) -> None:
    """The three figures that decide whether anything below them can be read.

    Above the metrics rather than beside them. Each is a way for a table of rates to be
    arithmetically correct and still not describe the eval set someone thinks they are reading
    about.
    """
    st.subheader("Read these first")
    # Bordered, so each figure and the sentence explaining it read as one card. Without the border
    # the captions run together into a paragraph under three unrelated numbers, which is the
    # opposite of what putting them above the metrics was for.
    columns = st.columns(3, border=True)

    columns[0].metric("Infrastructure failures", summary.infrastructure_failed)
    columns[0].caption(
        "Items excluded from every denominator because our tooling failed, not the model. A count "
        "that differs sharply between two runs is itself a finding."
    )

    truncation = summary.budget_induced_truncation_rate
    rate = 0.0 if truncation is None else truncation.mean
    columns[1].metric("Budget-induced truncation", f"{rate:.1%}")
    columns[1].caption(
        f"Model calls our own max_tokens cut off, against the pre-registered "
        f"{BUDGET_INDUCED_WARNING_THRESHOLD:.0%} threshold. Above it, a format-violation figure is "
        "measuring our ceiling rather than the model."
    )

    columns[2].metric("Items missing from the trace", summary.n_missing_from_trace)
    columns[2].caption(
        "Items the run never reached. A partial run is not a smaller run: the items it skipped are "
        "not a random sample of the eval set."
    )


def section_table(rows: list[SummaryRow]) -> list[dict[str, object]]:
    """One section's rows, with the interval and the denominator as separate numeric columns.

    Separate columns rather than one formatted `mean [low, high] n=` cell, so a later revision can
    draw an error bar or sort by effect size without going back around the shaper — which is the
    reason `SummaryRow` carries an `Aggregate` rather than a string.

    None where a bucket is empty, never a zero. A zero with an interval around it reads as a
    measurement, and an empty bucket is not one; the note column says why it is empty.
    """
    table: list[dict[str, object]] = []
    for row in rows:
        measured = None if row.is_empty else row.aggregate
        table.append(
            {
                "metric": row.metric,
                "reading": row.label,
                "cut": row.cut,
                "mean": None if measured is None else measured.mean,
                "ci_low": None if measured is None else measured.ci_low,
                "ci_high": None if measured is None else measured.ci_high,
                "n": row.n,
                "well-formed only": row.wellformed,
                "note": row.note,
            }
        )
    return table


def render_metrics(summary: RunSummary) -> None:
    """Every row `summary_rows` emits, grouped into its sections and in their order."""
    rows = summary_rows(summary)
    by_section: dict[str, list[SummaryRow]] = {}
    for row in rows:
        by_section.setdefault(row.section, []).append(row)

    st.subheader("Metrics")
    st.caption(
        "Every bucket the platform measures, including the ones with no items — an omitted row "
        "cannot be told apart from a vocabulary value nobody authored. Thresholded rates appear at "
        "all four pre-registered cuts. Rows marked well-formed are conditioned on responses that "
        "parsed, which conditions on the model's own success: read the unconditioned figure for a "
        "comparison and the conditioned one for answer quality given a well-formed reply."
    )
    for section in SECTION_ORDER:
        section_rows = by_section.get(section)
        if not section_rows:
            continue
        with st.expander(f"{section} ({len(section_rows)} rows)", expanded=True):
            st.dataframe(
                section_table(section_rows),
                width="stretch",
                hide_index=True,
                column_config=METRIC_COLUMNS,
            )


def operational_figures(summary: RunSummary) -> None:
    """What the run cost and how long it took.

    Scalars with no interval and no denominator, which is why they are `st.metric`s here and not
    rows in the tables above: giving a token count an empty `Aggregate` to fit one shape would make
    a bookkeeping figure look like a measurement with no data behind it.
    """
    st.subheader("What it cost")
    latency, spend = st.columns(2, border=True)
    latency.metric("Mean latency, uncached calls", f"{summary.mean_latency_ms:.0f} ms")
    latency.metric("p95 latency, uncached calls", f"{summary.p95_latency_ms:.0f} ms")
    latency.metric("Served from cache", f"{summary.cached_fraction:.1%}")
    latency.caption(
        "Latency is over uncached calls only: a cache hit replays the original call's latency, so "
        "averaging over hits would report a disk read as model speed."
    )

    spend.metric("Total tokens", f"{summary.total_tokens:,}")
    # None rather than zero when the model has no price-table entry: PROJECT.md's one-price-table
    # rule. A 0.00 here would be a claim that the run was free.
    unpriced = "unpriced"
    cost = summary.total_usd_cost
    per_1k = summary.usd_per_1k_queries
    spend.metric("Total cost", unpriced if cost is None else f"${cost:.4f}")
    spend.metric("Cost per 1k queries", unpriced if per_1k is None else f"${per_1k:.2f}")
    spend.metric("Mean model calls per item", f"{summary.mean_model_calls:.2f}")
    if cost is None:
        spend.caption(
            "This model has no entry in the price table, so its cost is unknown rather than zero."
        )


def conditions_table(run: RunRef, summary: RunSummary) -> list[dict[str, str]]:
    """The run's identity and denominators, each row saying what it is.

    Printed rather than assumed. A score quoted without the conditions it was measured under
    invites comparison against a number from different conditions, which is the failure the
    manifest exists to prevent.
    """
    values: dict[str, object] = {
        "run_id": summary.run_id,
        "model": summary.model,
        "judge_run_id": summary.judge_run_id,
        "n_cases": summary.n_cases,
        "n_scored": summary.n_scored,
        "n_wellformed": summary.n_wellformed,
        "n_substituted": summary.n_substituted,
        "n_unjudged": summary.n_unjudged,
    }
    rows = [
        {"condition": name, "value": fmt_optional(values[name]), "what it is": explanation}
        for name, explanation in CONDITIONS
    ]
    manifest = run.manifest
    return [
        *rows,
        {
            "condition": "provider",
            "value": manifest.provider,
            "what it is": "who served the model",
        },
        {
            "condition": "dataset_path",
            "value": fmt_optional(manifest.dataset_path),
            "what it is": "the eval set, digested in the manifest as dataset_sha256",
        },
        {
            "condition": "started_at",
            "value": manifest.started_at,
            "what it is": "when the run began",
        },
        {
            "condition": "git_sha",
            "value": fmt_optional(manifest.git_sha),
            "what it is": "the commit the harness was at",
        },
        {
            "condition": "git_dirty",
            "value": git_marker(manifest),
            "what it is": "whether that commit identifies the code that ran",
        },
    ]


def main() -> None:
    """Pick an eval run and render it, in the order the module docstring argues for."""
    configure_page("Run detail")
    st.title("Run detail")
    st.caption(READ_ONLY_NOTE)

    runs, root = load_runs()
    evaluated = eval_runs(runs)
    if not evaluated:
        st.warning(
            f"No eval run under `{root}`. Chat sessions and judge runs have no summary to show: "
            "one was scored against no dataset, the other has no agent under test."
        )
        return

    run_id = st.selectbox(
        "Eval run",
        [run.run_id for run in evaluated],
        key=DETAIL_RUN_KEY,
        help="Shared with the Browse runs page, so a run chosen there opens here.",
    )
    run = next(one for one in evaluated if one.run_id == run_id)

    try:
        summary = summary_for(run)
    except (OSError, ValueError) as exc:
        # Most often a dataset that has changed since the run, which `summarise_run` refuses to
        # score against rather than reporting figures for two different eval sets under one id.
        st.error(f"Cannot summarise {run_id}: {exc}")
        return

    # The heading names the run whose numbers follow, so a screenshot carries its own provenance.
    st.header(f"{summary.run_id} — {summary.model}")

    promoted_figures(summary)

    if summary.warnings:
        st.subheader("Warnings recorded while summarising")
        for warning in summary.warnings:
            st.warning(warning)

    st.subheader("Conditions")
    st.dataframe(
        conditions_table(run, summary),
        width="stretch",
        hide_index=True,
        column_config=CONDITION_COLUMNS,
    )

    render_metrics(summary)
    operational_figures(summary)

    st.caption(
        "No item text, expected behaviour, or annotator note is shown on this page: those are "
        "instructions written for a human labeller, and showing them here would be the same "
        "mistake as showing them to a model."
    )


if __name__ == "__main__":
    main()
