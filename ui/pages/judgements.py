"""One eval run's judgements, item by item: the scores, and the reasoning behind them.

Run detail answers what a run measured. This page answers where one of those figures came from —
a judge mean over sixty items is one number, and the sixty rationales it averaged are what say
whether it should be trusted. So there is a row per dataset item and no aggregation at all.

Three things this page is careful about, each a way an honest-looking table would mislead:

**A blank score is an absence, never a zero.** `ItemResult.judge` is None when nothing scored the
item and `parse_ok` is False when the judge's reply did not parse. `metrics.py` puts the rule as "a
failed parse cannot be averaged in as a zero": an unparsed judgement and a genuinely bad response
are different findings, and a 0.0 in this table would merge them. Every unscored row therefore
carries blanks and, in the last column, the reason it is blank — an item with no row at all would
be the same omission in a more deniable form.

**`axis` and `rubric` are two different things and one of them is not the judge's.** An axis groups
items and is a property of the question, which is why it is read from `ItemResult.item`; a rubric is
the text the judge was given, which is why it is read from the judgement. The pilot judge runs are
the case that makes this matter: they record `axis: null` under the default rubric, because they
were run without per-axis rubrics, while the metrics elsewhere show `axis:hallucination:*` rows
sourced from the dataset. Reading "axis" off the judgement would print None on every row of a run
whose items all have one.

**No item text.** Not the prompt, not `expected_behavior`, not `notes`. A rationale is the judge's
writing about a response and is a result; the item is an instruction written for a human annotator,
and PROJECT.md's rule about those is not a rule about models only.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent.prompts import JUDGE_DIMENSIONS, JUDGE_SCALE_MAX
from evals.metrics import ItemResult
from ui.data import RunRef, eval_runs, judgements_for
from ui.layout import DETAIL_RUN_KEY, READ_ONLY_NOTE, configure_page, load_runs

#: The reading `overall` is, kept beside the four dimensions rather than folded into them: the
#: rubric returns it as its own holistic score and `ItemResult.dimension` accepts it as such.
OVERALL = "overall"

#: Said in the `why blank` column of a row nothing scored, rather than left empty. A blank score
#: with a blank reason is indistinguishable from a rendering bug.
UNJUDGED_NOTE = "no judgement for this item in this judge run"

#: Row height in pixels, set above the default so that Streamlit wraps the rationale over several
#: lines instead of clipping it to one. A number rather than a default because the rationales this
#: page exists to show are long: over the pilot judge runs they are 248 to 688 characters, median
#: 443, and one clipped line of a 443-character paragraph is a preview of the judge's reasoning
#: rather than the reasoning. Three lines and a click for the rest is the trade — taller rows would
#: read better one at a time and make the scores, which are the other half of this table, impossible
#: to compare down the column.
RATIONALE_ROW_HEIGHT = 88

#: The four dimension columns, all on one scale and formatted alike. Built from
#: `prompts.JUDGE_DIMENSIONS` so this page cannot come to disagree with the rubric about what was
#: scored — a dimension added there appears here without an edit.
DIMENSION_COLUMNS: dict[str, Any] = {
    dimension: st.column_config.NumberColumn(
        dimension,
        format="%.1f",
        help=f"1-{JUDGE_SCALE_MAX}. Blank where the item was not scored.",
    )
    for dimension in JUDGE_DIMENSIONS
}

#: How the table is drawn, in display order. Display only, and the score columns keep their
#: dimension names, because those are the keys the rubric asks for and the ones every other surface
#: prints.
COLUMNS: dict[str, Any] = {
    "item_id": st.column_config.TextColumn(
        "item_id",
        width="small",
        pinned=True,
        help="The dataset item this judgement is of. The judge saw it as a pair_id.",
    ),
    "axis": st.column_config.TextColumn(
        "item axis",
        width="small",
        help="What the item measures, from the dataset. An axis groups items; it is not a judge "
        "dimension, and it is not read from the judgement.",
    ),
    "rubric": st.column_config.TextColumn(
        "rubric read",
        width="small",
        help="The rubric file this judgement was made under, as the judgement recorded it. A run "
        "judged without per-axis rubrics scored every axis under `default`.",
    ),
    **DIMENSION_COLUMNS,
    OVERALL: st.column_config.NumberColumn(
        OVERALL,
        format="%.1f",
        help="The judge's holistic score. Blank where there is none — an unparsed or absent "
        "judgement is not a zero.",
    ),
    "why blank": st.column_config.TextColumn(
        "why blank",
        width="medium",
        help="Empty on a scored row. Otherwise the judge's own parse error, or the absence of a "
        "judgement, so no blank row is unexplained.",
    ),
    "rationale": st.column_config.TextColumn(
        "rationale",
        width="large",
        help="What the judge wrote before it scored. Click a cell to read it in full.",
    ),
}


def judgement_rows(results: list[ItemResult]) -> list[dict[str, object]]:
    """One row per item, with None for every score there is not.

    `ItemResult.dimension` is what decides that: it answers None when the judgement is absent or
    did not parse, so this function does not get to decide it a second way.
    """
    rows: list[dict[str, object]] = []
    for result in results:
        judge = result.judge
        rows.append(
            {
                "item_id": result.item_id,
                # From the item, not the judgement. See the module docstring.
                "axis": str(result.item.axis),
                "rubric": "" if judge is None else judge.rubric,
                **{name: result.dimension(name) for name in JUDGE_DIMENSIONS},
                OVERALL: result.dimension(OVERALL),
                "why blank": _why_blank(result),
                "rationale": "" if judge is None else judge.rationale,
            }
        )
    return rows


def _why_blank(result: ItemResult) -> str:
    """Why this row has no scores, or empty when it has them."""
    judge = result.judge
    if judge is None:
        return UNJUDGED_NOTE
    if not judge.parse_ok:
        # The judge's own message, which says what it emitted instead of a verdict. A judge failure
        # is ours rather than the model's, and this column is where that is visible per item.
        return judge.error or "the judgement did not parse"
    return ""


def main() -> None:
    """Pick an eval run and render every judgement made of it."""
    configure_page("Judgements")
    st.title("Judgements")
    st.caption(READ_ONLY_NOTE)

    runs, root = load_runs()
    evaluated = eval_runs(runs)
    if not evaluated:
        st.warning(
            f"No eval run under `{root}`. Only an eval run has judgements: a chat session was "
            "scored against no dataset, and a judge run is the scoring."
        )
        return

    run_id = st.selectbox(
        "Eval run",
        [run.run_id for run in evaluated],
        key=DETAIL_RUN_KEY,
        help="Shared with the Browse runs and Run detail pages, so a run chosen there opens here.",
    )
    run = next(one for one in evaluated if one.run_id == run_id)

    if run.judge_run_id is None:
        st.warning(
            f"No judge run scored `{run_id}`, so there are no judgements to show — the "
            "deterministic checks on Run detail do not need one, but every judge-derived figure "
            f"there is reported with `n=0`. Score it with `agentseval-judge --run {run_id}`."
        )
        return

    try:
        results = judgements_for(run)
    except (OSError, ValueError) as exc:
        # Most often a dataset that has changed since the run, which the join refuses rather than
        # attaching scores to items that are no longer the ones the model answered.
        st.error(f"Cannot read the judgements for {run_id}: {exc}")
        return

    _render(run, results)


def _render(run: RunRef, results: list[ItemResult]) -> None:
    """Draw the table, with what it is over stated above it."""
    unscored = [result for result in results if not result.judged]
    # The heading names both runs, so a screenshot says which judge produced these numbers as well
    # as which candidate they are about.
    st.header(f"{run.run_id} — judged by {run.judge_run_id}")
    st.caption(
        f"{len(results)} item(s) with records in this run's trace, {len(unscored)} of them without "
        "a parsed judgement. An unscored item keeps its row and its scores stay blank: an unparsed "
        "judgement is a judge-side failure, and entering it as a zero would move a candidate-side "
        "figure for a judge-side reason."
    )

    st.dataframe(
        judgement_rows(results),
        width="stretch",
        hide_index=True,
        column_config=COLUMNS,
        row_height=RATIONALE_ROW_HEIGHT,
    )
    st.caption(
        "Click a rationale to read the rest of it. Rationales are the judge's own text about a "
        "response; no prompt, expected behaviour, or annotator note is shown, because those are "
        "written for a human labeller and showing them here would be the same mistake as showing "
        "them to a model."
    )


if __name__ == "__main__":
    main()
