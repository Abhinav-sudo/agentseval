"""Human-readable reports from run data.

Renders `RunSummary` and `Comparison` objects to the terminal and to markdown. Reports are
built only from what is in `runs/` — no model calls — so a report can be regenerated from
an old trace at any time.

Every report states the conditions it was produced under (models, prompt version, corpus
fingerprint, judge model and rubric version, dataset, n) alongside the numbers. A score
printed without its manifest invites comparison against numbers from different conditions,
which is how eval results get quietly misread.

Reports must also surface failures — parse errors, cut-off runs, unparsed judgements —
rather than only aggregates. A 92% mean over cases where a quarter of tool calls failed to
parse is a misleading headline.

**Empty buckets are printed, never omitted.** Every breakdown here is over a closed vocabulary
from `evals.schema`, so the set of rows is known before the data is: a subcategory or an
attack type with no items is a fact about the dataset, and dropping the row hides it behind a
shape a reader cannot distinguish from a vocabulary that never had the value. `ZERO_ROW_NOTES`
carries the one-line explanation for the buckets whose emptiness is deliberate, so a reader
gets "none here, by design, and here is where it is measured" rather than a bare zero to
wonder about. This is pre-registered in PROJECT.md alongside the other reporting rules, because
"omit rows with no data" is exactly the kind of tidying that would quietly delete the evidence
that a bucket was never authored.

**Rows are data, formatted late.** `metrics.compare_runs` returns `Comparison` objects and
`render_comparison` formats them; `summary_rows` returns `SummaryRow` objects and
`render_run_summary` formats those. The split is not tidiness. A row that arrived pre-formatted
could not be drawn as an error bar, sorted by effect size, or joined back to the items behind it,
so any caller wanting one of those would shape the summary again itself — and two implementations
of "what does this run say" are two answers. `SummaryRow.metric` is the machine key for the same
reason: `attack_success_rate:roleplay@3` names a bucket that `metrics.load_item_results` can be
filtered by, and a human-readable label names nothing.

**A report file is short by dropping breakdowns, never readings.** `write_markdown_report` writes a
committed artifact, and `runs/` is gitignored, so the file is the only form in which a result leaves
this repository — which makes what it leaves out the most consequential choice in the module.
`HEADLINE_METRICS` selects by metric name before the data is read, so the short report cannot become
"the rows that looked like something": it keeps every reading of every population it reports, at
every cut, and drops only the sub-buckets those populations are broken into. The output carries no
generation timestamp, so a regenerated report over an unchanged run is byte-identical and its diff
is only ever a number that moved.

`render_judge_validation` and `render_failure_digest` still raise.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from agent.core import FormatViolation
from agent.guardrails import GuardrailAction
from agent.prompts import JUDGE_DIMENSIONS
from agent.tools.errors import ToolErrorReason
from agent.trace import DEFAULT_RUNS_DIR
from evals.deterministic import CHECK_NAMES
from evals.metrics import (
    CONSISTENCY_COMPONENT_DIVERGENCE,
    CONSISTENCY_COMPONENTS,
    CONTRAST_GUARDRAILS,
    CONTRAST_MODEL,
    METHOD_NEWCOMBE,
    RATE_ATTACK_SUCCESS,
    RATE_FALSE_PREMISE_CORRECTION,
    RATE_FALSE_REFUSAL,
    RATE_HALLUCINATION,
    SURVIVORSHIP_CAVEAT,
    THRESHOLD_CUTS,
    VERDICT_REGRESSION,
    Aggregate,
    Comparison,
    RunSummary,
    SafetyRates,
    ThresholdCurve,
    compare_runs,
    count_rate,
    guardrail_verdict,
    rate_delta_interval,
    summarise_run,
)
from evals.schema import BIAS_SUBCATEGORIES, AttackType, Axis
from evals.validate_judge import AgreementReport

#: `overall` is a reading the rubric returns alongside the four dimensions, and every judge
#: aggregate carries it, so the row vocabulary is the five and not the four.
JUDGE_READINGS: tuple[str, ...] = (*JUDGE_DIMENSIONS, "overall")

#: Buckets that are legitimately empty in the main eval sets, and why. Rendered beside the
#: zero so the row explains itself. A bucket that is empty for no recorded reason is an
#: authoring gap and must **not** be added here to silence it.
ZERO_ROW_NOTES: dict[str, str] = {
    AttackType.PROMPT_INJECTION.value: (
        "none in this run by design: retrieval-borne injection needs a poisoned corpus, so it "
        "is measured by evals/datasets/injection.jsonl against the composed fixture corpus "
        "and reported separately (PROJECT.md, 'The injected fixture corpus')"
    ),
}


#: Section headings, keyed by the metric-name prefix that selects them. Sections appear in the
#: order `compare_runs` emitted their first row, which is what keeps attack success and false
#: refusal adjacent: they share a section, and the ordering is not re-derived here where it could
#: be re-derived differently.
SECTION_TITLES: tuple[tuple[str, str], ...] = (
    ("judge:", "Judge dimensions"),
    ("axis:", "Judge dimensions by axis"),
    ("check_wellformed:", "Deterministic checks, well-formed responses only"),
    ("check:", "Deterministic checks"),
    ("guardrail_action_rate:", "Guardrail actions"),
    (RATE_HALLUCINATION, "Hallucination rate, by cut"),
    (RATE_FALSE_PREMISE_CORRECTION, "False-premise correction rate, by cut"),
    (RATE_ATTACK_SUCCESS, "Safety: attack success with its over-refusal control"),
    (RATE_FALSE_REFUSAL, "Safety: attack success with its over-refusal control"),
    ("consistency:", "Counterfactual consistency"),
)

#: Where an unrecognised metric lands. A section rather than a dropped row: a metric this
#: renderer has not been taught about is still a number the run produced, and losing it silently
#: would be the same failure as omitting an empty bucket.
SECTION_OTHER = "Other metrics"

#: The section headings in order, deduplicated — attack success and false refusal share one — with
#: `SECTION_OTHER` last. Derived from `SECTION_TITLES` rather than written out again, so a section
#: cannot be added to one ordering and forgotten in the other.
SECTION_ORDER: tuple[str, ...] = (
    *dict.fromkeys(title for _, title in SECTION_TITLES),
    SECTION_OTHER,
)

#: Marks a row conditioned on well-formed responses, footnoted with `SURVIVORSHIP_CAVEAT`.
WELLFORMED_MARK = "†"

#: What the `Significance` column means, spelled out under every table. The two kinds of row are
#: not equally strong and the header would otherwise imply they were: a paired permutation
#: p-value over shared items is a test, and disjoint intervals are an observation that happens to
#: rule out overlap. Naming both is the same discipline `_disjoint`'s docstring keeps.
SIGNIFICANCE_LEGEND = (
    "Significance: 'p=' is a paired permutation test over the items both runs scored. "
    "'disjoint'/'overlap' is the weaker interval test used where there is no per-item pair to "
    "permute — a rate is one number per run. 'overlap' does not mean the two are equal, only "
    "that this data cannot separate them."
)

#: What the delta interval is and is not, printed under every table for the same reason.
DELTA_LEGEND = (
    "Delta is left minus right. Its interval is Newcombe's hybrid-score interval for a "
    "difference of two rates, which assumes the two runs are independent; they scored the same "
    "items, so it is wider than the truth and errs toward finding nothing. Rows whose metric is "
    "a mean rather than a rate show no delta interval, because an Aggregate no longer carries "
    "the per-item values a bootstrap would need."
)


def _section_for(metric: str) -> str:
    """The section a metric's row belongs in."""
    for prefix, title in SECTION_TITLES:
        if metric.startswith(prefix):
            return title
    return SECTION_OTHER


def _zero_row_note(metric: str, *, fallback: str = "no items on either side") -> str:
    """The `ZERO_ROW_NOTES` entry for an empty bucket's metric, or the bare fact of emptiness.

    The tokens of a metric name are searched rather than the whole string, because the bucket is
    a component of it: `attack_success_rate:prompt_injection@3` is keyed by `prompt_injection`.

    Args:
        fallback: What an empty bucket with no registered explanation says. The default is the
            two-run wording; a single run has no sides, and saying it did would be a small lie
            about what was read.
    """
    for token in metric.replace("@", ":").split(":"):
        note = ZERO_ROW_NOTES.get(token)
        if note is not None:
            return f"no items — {note}"
    return fallback


#: An empty bucket's note on the single-run path, where "either side" would name a side that is
#: not there.
NO_ITEMS_NOTE = "no items"


def _fmt_aggregate(aggregate: Aggregate | None) -> str:
    """`mean [low, high] n=` for one side, or a dash where there is nothing to report.

    A dash rather than `0.000 [0.000, 0.000] n=0`: a zero with an interval around it reads as a
    measurement, and an empty bucket is not one. None is a bucket the run carries no aggregate
    for at all, which is likewise not a zero.
    """
    if aggregate is None or aggregate.n == 0:
        return "—"
    return (
        f"{aggregate.mean:.3f} [{aggregate.ci_low:.3f}, {aggregate.ci_high:.3f}] "
        f"n={aggregate.n}"
    )


def _fmt_delta(comparison: Comparison) -> tuple[str, str]:
    """The delta and its interval, as two columns.

    The interval is empty for a mean-valued metric rather than absent: see `DELTA_LEGEND`.
    """
    if comparison.left.n == 0 or comparison.right.n == 0:
        return "—", ""
    interval = rate_delta_interval(comparison.left, comparison.right)
    if interval is None or interval.method != METHOD_NEWCOMBE:
        return f"{comparison.delta:+.3f}", ""
    return (
        f"{comparison.delta:+.3f}",
        f"[{interval.ci_low:+.3f}, {interval.ci_high:+.3f}]",
    )


def _fmt_significance(comparison: Comparison) -> str:
    """The significance cell, saying which of the two tests produced it."""
    if comparison.left.n == 0 or comparison.right.n == 0:
        return "—"
    if comparison.p_value is not None:
        return f"p={comparison.p_value:.3f}"
    return "disjoint" if comparison.significant else "overlap"


#: Notes longer than this become a numbered footnote instead of a cell. A `ZERO_ROW_NOTES` entry
#: runs to three lines, and left in the cell it sets the column width for every other row —
#: which is how a table stops being readable and a reader stops reading it.
MAX_NOTE_CHARS = 48


def _row_notes(comparison: Comparison) -> list[str]:
    """The row's notes: empty buckets, unstable rankings, and the survivorship mark."""
    notes: list[str] = []
    if comparison.left.n == 0 and comparison.right.n == 0:
        notes.append(_zero_row_note(comparison.metric))
    elif comparison.left.n == 0 or comparison.right.n == 0:
        notes.append("one side has no items")
    if comparison.stable_across_cuts is False:
        notes.append("ranking flips between cuts")
    if "wellformed" in comparison.metric:
        notes.append(f"{WELLFORMED_MARK} conditioned on well-formed responses")
    return notes


def _short_note(note: str, footnotes: list[str]) -> str:
    """`note` if it fits in a cell, otherwise a footnote marker, appending the text to `footnotes`.

    Deduplicated by text, so the twelve cuts of one empty bucket share one footnote rather than
    printing the same paragraph twelve times.
    """
    if len(note) <= MAX_NOTE_CHARS:
        return note
    if note not in footnotes:
        footnotes.append(note)
    return f"[{footnotes.index(note) + 1}]"


def _pad_table(rows: Sequence[Sequence[str]]) -> list[str]:
    """A fixed-width table, header underlined, for the terminal."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    ]
    return [lines[0], "  ".join("-" * width for width in widths), *lines[1:]]


def _markdown_table(rows: Sequence[Sequence[str]]) -> list[str]:
    """A pipe table. Cells are escaped for `|`, which a note could otherwise contain."""
    header, *body = [[cell.replace("|", "\\|") for cell in row] for row in rows]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]


def check_safety_pairing(metrics: Iterable[str], *, source: str) -> None:
    """Refuse to render attack success without its over-refusal control.

    PROJECT.md pre-registers that the two are never reported apart, and `SafetyRates` makes that
    hold for anything built through `compare_runs` or `summarise_run`. This is the same rule
    applied to the renderer, for the case of a caller that filtered the rows on their way here: a
    table showing only the safety gain is the specific misleading artifact the pairing rule exists
    to prevent, and it would be no less misleading for having been produced by a filter rather
    than by a choice.

    Over metric names rather than over rows, so the comparison table, the single-run table, and any
    later view enforce one rule from one definition. A second copy of it is a second thing to keep
    in step with the pre-registration, and the copy that drifted would be the one that rendered the
    forbidden table.

    Args:
        source: What the caller should render instead, named in the message. A reader who has just
            been refused needs to know which unfiltered list to pass.

    Raises:
        ValueError: an attack-success metric is present and no false-refusal metric is.
    """
    names = list(metrics)
    has_attack = any(name.startswith(RATE_ATTACK_SUCCESS) for name in names)
    has_refusal = any(name.startswith(RATE_FALSE_REFUSAL) for name in names)
    if has_attack and not has_refusal:
        raise ValueError(
            f"{RATE_ATTACK_SUCCESS} rows are present but no {RATE_FALSE_REFUSAL} row is: a "
            "safety gain printed without its over-refusal cost is the one table PROJECT.md's "
            f"pre-registration forbids. Render the rows {source} returned, unfiltered"
        )


def _check_pairing(comparisons: Sequence[Comparison]) -> None:
    """`check_safety_pairing` over a comparison list. See there."""
    check_safety_pairing((c.metric for c in comparisons), source="compare_runs")


def render_comparison(comparisons: list[Comparison], *, markdown: bool = False) -> str:
    """Render a two-run table, marking which gaps are distinguishable from noise.

    Sides are `left` and `right` under the labels the comparisons carry, so the same renderer
    serves the arm comparison and the guardrails ablation and neither is mislabelled as the
    other. Rows keep the order `compare_runs` emitted them in, which is what keeps attack success
    and false refusal in one section and adjacent.

    Every row shows both sides with their intervals, the delta, an interval on the delta where
    the statistic admits one, and which of the two significance tests produced its verdict —
    `SIGNIFICANCE_LEGEND` and `DELTA_LEGEND` say what each of those means, under the table, every
    time. A guardrails contrast additionally opens with `guardrail_verdict`'s line, so the win
    condition is applied where the numbers are rather than left to the reader.

    Empty buckets render as dashes with a note, never as omitted rows, per the module docstring.

    Args:
        markdown: Pipe tables instead of fixed-width ones. The same rows either way — the two
            outputs differ in punctuation only, so a figure quoted from one is the figure in the
            other.

    Raises:
        ValueError: `comparisons` is empty, mixes two contrasts, or shows attack success with no
            false-refusal row.
    """
    if not comparisons:
        raise ValueError("nothing to render: compare_runs returned no comparable metric")
    contrasts = {(c.contrast, c.left_label, c.right_label) for c in comparisons}
    if len(contrasts) > 1:
        raise ValueError(
            f"these comparisons carry {len(contrasts)} different contrasts {sorted(contrasts)}. "
            "One table means one thing varied; two contrasts in one table is two tables, and a "
            "reader cannot tell which row belongs to which"
        )
    _check_pairing(comparisons)

    contrast, left_label, right_label = contrasts.pop()
    lines = [
        f"Comparison — {contrast} contrast",
        f"  left:  {left_label or '(unlabelled)'}",
        f"  right: {right_label or '(unlabelled)'}",
    ]

    verdict = guardrail_verdict(comparisons) if contrast == CONTRAST_GUARDRAILS else None
    if verdict is not None:
        lines += [
            "",
            f"Verdict (pre-registered): {verdict.verdict.upper()}",
            f"  {verdict.detail}",
        ]
        if verdict.verdict == VERDICT_REGRESSION:
            lines.append(
                "  A guardrail that buys safety with more over-refusal than it removes harm has "
                "moved the failure rather than reduced it."
            )

    header = (
        "Metric",
        left_label or "left",
        right_label or "right",
        "Delta",
        "Delta 95% CI",
        "Significance",
        "Notes",
    )
    sections: dict[str, list[tuple[str, ...]]] = {}
    footnotes: list[str] = []
    for comparison in comparisons:
        section = _section_for(comparison.metric)
        delta, interval = _fmt_delta(comparison)
        sections.setdefault(section, []).append(
            (
                comparison.metric,
                _fmt_aggregate(comparison.left),
                _fmt_aggregate(comparison.right),
                delta,
                interval,
                _fmt_significance(comparison),
                "; ".join(_short_note(note, footnotes) for note in _row_notes(comparison)),
            )
        )

    table = _markdown_table if markdown else _pad_table
    heading = "### " if markdown else ""
    for section, rows in sections.items():
        lines += ["", f"{heading}{section}", *table([header, *rows])]

    if footnotes:
        lines += ["", *(f"[{i + 1}] {note}" for i, note in enumerate(footnotes))]
    lines += ["", SIGNIFICANCE_LEGEND, "", DELTA_LEGEND]
    if any("wellformed" in c.metric for c in comparisons):
        lines += ["", f"{WELLFORMED_MARK} {SURVIVORSHIP_CAVEAT}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# One run: rows first, formatting second
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryRow:
    """One metric of one run: what it is called, what it measured, and what to say about it.

    The single-run counterpart to `Comparison`, and structured for the same reason. A row of
    pre-formatted strings can be printed and nothing else — not sorted by effect size, not drawn
    as an error bar, not filtered to the well-formed cut — so a caller wanting any of those would
    flatten the `RunSummary` again itself, and two flattenings are two answers.

    Attributes:
        metric: The machine key, identical in shape to `Comparison.metric`:
            `attack_success_rate:roleplay@3`, `judge:accuracy_wellformed`, `check:no_refusal`. It
            names a bucket rather than describing one — `metrics.load_item_results` can be filtered
            by its parts, and `_zero_row_note` reads its tokens — which a display label cannot do.
            Keyed the same as the comparison path so the two renderers cannot disagree about what
            a metric is called.
        section: From `_section_for`, so this table groups rows exactly as `render_comparison`
            does. Attack success and false refusal share a section, which is what keeps them
            adjacent.
        label: Display only. Built where the row is built, because that is the only place that
            knows a bucket is an attack type rather than a check name.
        aggregate: The figure with its interval, or None where the run carries no aggregate for
            this bucket at all. None and `n == 0` are both empty and are rendered the same; they
            differ in whether the bucket was computed, which is not a distinction a reader of the
            number should have to act on.
        note: Why an empty row is empty, plus whatever the curve behind it recorded. Never a
            substitute for the row: `ZERO_ROW_NOTES` explains a zero, it does not excuse omitting
            it.
        wellformed: This figure is conditioned on responses that parsed, so it is subject to
            `SURVIVORSHIP_CAVEAT`. Derived from the metric key, by the same test `_row_notes`
            uses.
        cut: The threshold this row is at, for the rows that have one. Carried so a view can draw
            the curve across `THRESHOLD_CUTS` without parsing `@3` back out of the key.
    """

    metric: str
    section: str
    label: str
    aggregate: Aggregate | None
    note: str = ""
    wellformed: bool = False
    cut: int | None = None

    @property
    def n(self) -> int:
        """Items behind this figure. Zero when there is no aggregate at all."""
        return 0 if self.aggregate is None else self.aggregate.n

    @property
    def is_empty(self) -> bool:
        """Nothing was measured here. The row is rendered anyway; see the class docstring."""
        return self.n == 0


def _summary_row(
    metric: str,
    label: str,
    aggregate: Aggregate | None,
    *,
    note: str = "",
    cut: int | None = None,
) -> SummaryRow:
    """Build one row, attaching the empty-bucket explanation where the bucket is empty."""
    notes = [note] if note else []
    if aggregate is None or aggregate.n == 0:
        notes.insert(0, _zero_row_note(metric, fallback=NO_ITEMS_NOTE))
    return SummaryRow(
        metric=metric,
        section=_section_for(metric),
        label=label,
        aggregate=aggregate,
        note="; ".join(notes),
        wellformed="wellformed" in metric,
        cut=cut,
    )


def _curve_rows(metric: str, curve: ThresholdCurve | None, label: str) -> list[SummaryRow]:
    """One row per cut in `THRESHOLD_CUTS`, whether or not the curve was computed.

    All four cuts, always. PROJECT.md pre-registers the curve rather than a figure at one cut,
    because a rate quoted at a single threshold cannot be told apart from a rate that only holds
    at the threshold somebody picked.

    A curve's `n_unjudged` and `n_substituted` go on every one of its rows rather than on the first
    of them: a cut moves no denominator, so the caveat is equally true at each, and a row is quoted
    on its own.
    """
    shared: list[str] = []
    if curve is not None:
        if curve.note:
            shared.append(curve.note)
        if curve.n_unjudged:
            shared.append(
                f"{curve.n_unjudged} item(s) went unjudged and are out of the denominator: a "
                "judge failure is ours and says nothing about the candidate"
            )
        if curve.n_substituted:
            shared.append(f"{curve.n_substituted} of these answers were replaced by a guardrail")
    return [
        _summary_row(
            f"{metric}@{cut}",
            f"{label} at cut {cut}",
            None if curve is None else curve.by_cut.get(cut),
            note="; ".join(shared),
            cut=cut,
        )
        for cut in THRESHOLD_CUTS
    ]


def _safety_rows(safety: SafetyRates | None, *, suffix: str, label_suffix: str) -> list[SummaryRow]:
    """Attack success, its over-refusal control, and the per-attack-type breakdown.

    Emitted from one function over one object, so the control cannot be dropped by editing the
    caller: `SafetyRates` holds both curves and this reads both. `check_safety_pairing` is the
    check on the other end, for rows that arrive here already filtered.

    Every `AttackType` appears, including the ones with no items — `prompt_injection` has none in
    any main run by design, and PROJECT.md requires the zero be printed with its reason rather than
    left out.
    """
    rows = _curve_rows(
        f"{RATE_ATTACK_SUCCESS}{suffix}",
        None if safety is None else safety.attack_success,
        f"attack success{label_suffix}",
    )
    rows += _curve_rows(
        f"{RATE_FALSE_REFUSAL}{suffix}",
        None if safety is None else safety.false_refusal,
        f"false refusal on benign controls{label_suffix}",
    )
    for attack in AttackType:
        rows += _curve_rows(
            f"{RATE_ATTACK_SUCCESS}{suffix}:{attack.value}",
            None if safety is None else safety.by_attack_type.get(attack.value),
            f"attack success · {attack.value}{label_suffix}",
        )
    return rows


#: Appended to a label whose figure is conditioned on well-formed responses. The metric key says
#: the same thing in `_wellformed`; this is the reader's copy of it.
WELLFORMED_LABEL = " (well-formed only)"

#: The unthresholded per-item and per-call rates on a `RunSummary`, as
#: `(attribute, label, has a well-formed-only variant)`. The attribute name is also the metric key,
#: which is what `compare_runs` calls these. Listed rather than discovered by introspection: a rate
#: this list forgets is a rate the table silently omits, and a list is something a test can read
#: back against `RunSummary`'s fields.
SCALAR_RATES: tuple[tuple[str, str, bool], ...] = (
    ("abstention_rate", "abstention on unanswerable items", True),
    ("citation_validity_rate", "citations naming retrieved chunks", True),
    ("protocol_compliance", "model calls that honoured the protocol", False),
    ("format_violation_rate", "model calls that broke the protocol", False),
    ("budget_induced_truncation_rate", "model calls our max_tokens cut off", False),
    ("tool_call_error_rate", "tool calls rejected for model-caused reasons", False),
)


def summary_rows(summary: RunSummary) -> list[SummaryRow]:
    """Flatten one run into ordered rows, one per bucket the platform measures.

    Rows come from the vocabularies — `JUDGE_READINGS`, `Axis`, `CHECK_NAMES`, `GuardrailAction`,
    `AttackType`, `BIAS_SUBCATEGORIES`, `FormatViolation`, `ToolErrorReason`, `THRESHOLD_CUTS` —
    and never from the keys present in the data. That is the whole point of the function: a bucket
    with no items is a fact about the run, and a table that dropped its row would report the same
    run as a smaller one. Empty rows carry `ZERO_ROW_NOTES`' explanation where there is one.

    Ordered by `SECTION_ORDER`, which is derived from `SECTION_TITLES`, which is the order
    `render_comparison` uses. One run and a two-run comparison therefore group and order their rows
    identically, and a reader moving between the two pages is not re-learning the layout.

    Returns:
        Every row, unfiltered — including both the unconditioned and the well-formed-only reading
        of each axis metric, per PROJECT.md's rule that each is reported twice. Filtering is the
        caller's business, subject to `check_safety_pairing`.
    """
    rows: list[SummaryRow] = []

    for reading in JUDGE_READINGS:
        rows.append(_summary_row(f"judge:{reading}", reading, summary.judge_scores.get(reading)))
        rows.append(
            _summary_row(
                f"judge:{reading}_wellformed",
                f"{reading}{WELLFORMED_LABEL}",
                summary.judge_scores_wellformed.get(reading),
            )
        )

    for axis in Axis:
        by_dimension = summary.judge_scores_by_axis.get(axis.value, {})
        for reading in JUDGE_READINGS:
            rows.append(
                _summary_row(
                    f"axis:{axis.value}:{reading}",
                    f"{axis.value} · {reading}",
                    by_dimension.get(reading),
                )
            )

    for check in CHECK_NAMES:
        rows.append(_summary_row(f"check:{check}", check, summary.check_pass_rates.get(check)))
        rows.append(
            _summary_row(
                f"check_wellformed:{check}",
                f"{check}{WELLFORMED_LABEL}",
                summary.check_pass_rates_wellformed.get(check),
            )
        )

    for action in GuardrailAction:
        rows.append(
            _summary_row(
                f"guardrail_action_rate:{action.value}",
                action.value,
                count_rate(
                    f"guardrail_action_rate:{action.value}",
                    summary.guardrail_action_counts.get(action.value, 0),
                    summary.n_scored,
                ),
            )
        )

    rows += _curve_rows(RATE_HALLUCINATION, summary.hallucination_rate, "hallucination")
    rows += _curve_rows(
        f"{RATE_HALLUCINATION}_wellformed",
        summary.hallucination_rate_wellformed,
        f"hallucination{WELLFORMED_LABEL}",
    )
    rows += _curve_rows(
        RATE_FALSE_PREMISE_CORRECTION,
        summary.false_premise_correction_rate,
        "false-premise correction",
    )
    rows += _curve_rows(
        f"{RATE_FALSE_PREMISE_CORRECTION}_wellformed",
        summary.false_premise_correction_rate_wellformed,
        f"false-premise correction{WELLFORMED_LABEL}",
    )
    rows += _safety_rows(summary.safety, suffix="", label_suffix="")
    rows += _safety_rows(
        summary.safety_wellformed, suffix="_wellformed", label_suffix=WELLFORMED_LABEL
    )

    consistency = summary.consistency
    for component in CONSISTENCY_COMPONENTS:
        rows.append(
            _summary_row(
                f"consistency:{component}",
                component,
                None if consistency is None else consistency.components.get(component),
            )
        )
    # The bias axis has no row in the unsplit judge figures — its finding is a within-pair delta —
    # so the headline divergence is broken out per demographic family here. Over the vocabulary,
    # so a family with no pairs authored reads as zero pairs rather than as an axis nobody thought
    # about. The remaining components are not split: they are what the headline is made of, and
    # eight components across every family is a table nobody reads.
    # Sorted, because the vocabulary is a frozenset: unsorted, the rows would come out in an order
    # that changes between processes, and a table whose row order moves cannot be diffed against
    # the same table from yesterday.
    for family in sorted(BIAS_SUBCATEGORIES):
        by_component = {} if consistency is None else consistency.by_subcategory.get(family, {})
        rows.append(
            _summary_row(
                f"consistency:{CONSISTENCY_COMPONENT_DIVERGENCE}:{family}",
                f"{CONSISTENCY_COMPONENT_DIVERGENCE} · {family}",
                by_component.get(CONSISTENCY_COMPONENT_DIVERGENCE),
            )
        )

    for attribute, label, has_wellformed in SCALAR_RATES:
        rows.append(_summary_row(attribute, label, getattr(summary, attribute)))
        if has_wellformed:
            rows.append(
                _summary_row(
                    f"{attribute}_wellformed",
                    f"{label}{WELLFORMED_LABEL}",
                    getattr(summary, f"{attribute}_wellformed"),
                )
            )

    for violation in FormatViolation:
        if violation is FormatViolation.TRUNCATED:
            # Not a protocol violation: our max_tokens ceiling interrupted a response that had
            # broken no contract. It is reported on its own above, and counting it here as well
            # would charge the model for our budget.
            continue
        rows.append(
            _summary_row(
                f"format_violation_rate:{violation.value}",
                f"format violation · {violation.value}",
                summary.format_violation_rate_by_type.get(violation.value),
            )
        )
    for reason in ToolErrorReason:
        rows.append(
            _summary_row(
                f"tool_call_error_rate:{reason.value}",
                f"tool call error · {reason.value}",
                summary.tool_call_error_rate_by_type.get(reason.value),
            )
        )

    # Stable, so each section keeps the emission order above; only the sections move.
    rows.sort(key=lambda row: SECTION_ORDER.index(row.section))
    return rows


#: Counts printed above the table, as `(attribute, what it means)`. `n_cases` first and
#: `n_scored` under it, because the gap between them is the part of the eval set no figure below
#: is about.
SUMMARY_COUNTS: tuple[tuple[str, str], ...] = (
    ("n_cases", "items in the eval set"),
    ("n_scored", "items that entered scoring"),
    ("n_wellformed", "items whose response parsed"),
    ("n_missing_from_trace", "items the run never reached"),
    ("infrastructure_failed", "items excluded: our tooling failed, not the model"),
    ("n_substituted", "items whose answer a guardrail replaced"),
    ("n_unjudged", "items with no parsed judgement"),
)


def render_run_summary(summary: RunSummary) -> str:
    """Render one run: manifest conditions, aggregates with CIs, then failures.

    A formatter over `summary_rows` and nothing more — every bucket, threshold, and empty-row note
    is decided there, so this function cannot reach a different set of rows than a view built on
    the same shaper. Per-subcategory and per-attack-type breakdowns iterate the vocabulary in
    `evals.schema` rather than the keys present in the data, so a bucket with no items renders as a
    zero row instead of vanishing; `ZERO_ROW_NOTES` supplies the explanation where there is one.

    The counts come before the metrics because they are the denominators: a judge mean over three
    of sixty items is not a small version of the same finding. The warnings come last and are
    always printed — `summarise_run` records them on the summary precisely so a report cannot lose
    them.

    Raises:
        ValueError: attack-success rows are present without their false-refusal control, which
            for an unfiltered `summary_rows` result cannot happen. See `check_safety_pairing`.
    """
    rows = summary_rows(summary)
    check_safety_pairing((row.metric for row in rows), source="summary_rows")

    lines = [
        f"Run {summary.run_id} — {summary.model}",
        f"  judge run: {summary.judge_run_id or '(unjudged — every judge figure below is empty)'}",
    ]
    if summary.manifest is not None:
        lines += [
            f"  provider:  {summary.manifest.provider}",
            f"  dataset:   {summary.manifest.dataset_path}",
            f"  git:       {summary.manifest.git_sha}"
            f"{' (DIRTY — uncommitted changes)' if summary.manifest.git_dirty else ''}",
        ]
    lines.append(
        "  " + "  ".join(f"{name}={getattr(summary, name)}" for name, _ in SUMMARY_COUNTS)
    )

    header = ("Metric", "Value", "Notes")
    footnotes: list[str] = []
    sections: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        sections.setdefault(row.section, []).append(
            (
                row.metric,
                _fmt_aggregate(row.aggregate),
                _short_note(row.note, footnotes) if row.note else "",
            )
        )
    for section, body in sections.items():
        lines += ["", section, *_pad_table([header, *body])]

    lines += ["", "Operational", *_pad_table([("Figure", "Value"), *_operational_rows(summary)])]

    if footnotes:
        lines += ["", *(f"[{i + 1}] {note}" for i, note in enumerate(footnotes))]
    if any(row.wellformed for row in rows):
        lines += ["", f"{WELLFORMED_MARK} {SURVIVORSHIP_CAVEAT}"]
    for warning in summary.warnings:
        lines += ["", f"warning: {warning}"]
    return "\n".join(lines)


def _operational_rows(summary: RunSummary) -> list[tuple[str, str]]:
    """What the run cost and how long it took, as table rows.

    Separate from `summary_rows` because these are plain scalars: there is no interval around a
    token count and no denominator behind a total. Giving them an `Aggregate` with `n=0` to fit one
    shape would make a bookkeeping figure look like a measurement with no data behind it.

    A None cost prints as unpriced rather than as zero, per PROJECT.md's one-price-table rule: a
    model missing from `base.PRICING` has an unknown cost, and 0.00 would be a claim it was free.
    """
    cost = summary.total_usd_cost
    per_1k = summary.usd_per_1k_queries
    unpriced = "— (model not in the price table)"
    return [
        ("mean model calls per scored item", f"{summary.mean_model_calls:.2f}"),
        ("mean latency (uncached calls)", f"{summary.mean_latency_ms:.0f} ms"),
        ("p95 latency (uncached calls)", f"{summary.p95_latency_ms:.0f} ms"),
        ("cached fraction of model calls", f"{summary.cached_fraction:.1%}"),
        ("total tokens", f"{summary.total_tokens:,}"),
        ("total cost", unpriced if cost is None else f"${cost:.4f}"),
        ("cost per 1k queries", unpriced if per_1k is None else f"${per_1k:.2f}"),
    ]


def render_judge_validation(report: AgreementReport) -> str:
    """Render judge-vs-human agreement.

    Belongs at the top of any report that leans on judge scores: it is the evidence that
    those scores mean anything.
    """
    raise NotImplementedError


def render_failure_digest(run_id: str, limit: int = 10) -> str:
    """Render concrete failing cases with their traces, for error analysis.

    Aggregates say a model is worse; these excerpts say how.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------------------
# The report file: the one artifact of a run meant to be committed
# --------------------------------------------------------------------------------------

#: Where `agentseval-report` writes and where `ui/pages/report.py` reads. Outside `runs/`, which
#: is gitignored so that a trace cannot be committed by accident — which would also stop a report
#: written there from being committed on purpose. A report is still derived: regenerate it rather
#: than editing it, and read `## Conditions` for the run and commit it was derived from.
DEFAULT_REPORTS_DIR = Path("reports")

#: The metric keys a short report carries: the rows measuring a whole population, and none of the
#: sub-buckets a population is broken into. Per-axis judge means, per-attack-type rates, per-type
#: format violations, and the per-family consistency split are left to `--full` and to the Run
#: detail page, all of which show every row.
#:
#: Selected by name, before the data is read, and that is the point. "Drop the rows that came back
#: empty" would select on the values and would report a run against a dataset with no bias items as
#: a run whose bias rows were never authored — the failure `ZERO_ROW_NOTES` exists to prevent. Every
#: key below appears in the report whatever the run holds for it, as a dash with the reason.
#:
#: A whole family is in or out together, never split by reading: PROJECT.md registers the
#: unconditioned and the well-formed figure as one pair, and a report showing only the second would
#: be quoting the flattering half of it. Threshold curves bring all of `THRESHOLD_CUTS` for the same
#: reason — a rate at one cut cannot be told apart from a rate that only holds at that cut.
HEADLINE_METRICS: tuple[str, ...] = (
    *(f"judge:{reading}" for reading in JUDGE_READINGS),
    *(f"judge:{reading}_wellformed" for reading in JUDGE_READINGS),
    *(f"check:{check}" for check in CHECK_NAMES),
    *(f"check_wellformed:{check}" for check in CHECK_NAMES),
    *(f"guardrail_action_rate:{action.value}" for action in GuardrailAction),
    *(
        f"{metric}@{cut}"
        for metric in (
            RATE_HALLUCINATION,
            f"{RATE_HALLUCINATION}_wellformed",
            RATE_FALSE_PREMISE_CORRECTION,
            f"{RATE_FALSE_PREMISE_CORRECTION}_wellformed",
            RATE_ATTACK_SUCCESS,
            f"{RATE_ATTACK_SUCCESS}_wellformed",
            RATE_FALSE_REFUSAL,
            f"{RATE_FALSE_REFUSAL}_wellformed",
        )
        for cut in THRESHOLD_CUTS
    ),
    f"consistency:{CONSISTENCY_COMPONENT_DIVERGENCE}",
    *(attribute for attribute, _label, _has_wellformed in SCALAR_RATES),
    *(
        f"{attribute}_wellformed"
        for attribute, _label, has_wellformed in SCALAR_RATES
        if has_wellformed
    ),
)

#: The figures for a reader who reads nothing else, as `(metric key, what to call it here)`.
#: Unthresholded every one of them, and that is a constraint rather than a coincidence: a rate the
#: judge scores against a threshold is a curve, and lifting one cut of it up here would be the
#: single-cut headline PROJECT.md forbids. Attack success is therefore not in this block — it is in
#: the tables below, at all four cuts, beside its over-refusal control.
#:
#: Labelled here rather than reusing `SummaryRow.label`, because those labels are unambiguous only
#: beside their metric key. Eight rows reading `safety`, `accuracy`, and `kb_grounded` in a block
#: with no key column would leave a reader guessing which `safety` this is — the judge's 1–5
#: dimension or the deterministic check — and the two are not on the same scale.
HEADLINE_FIGURES: tuple[tuple[str, str], ...] = (
    ("judge:overall", "judge overall, 1–5"),
    ("judge:accuracy", "judge accuracy, 1–5"),
    ("judge:safety", "judge safety, 1–5"),
    ("check:kb_grounded", "answers grounded in the corpus"),
    ("check:citation_grounding", "citations that name a retrieved chunk"),
    ("check:no_refusal", "answers that did not refuse or hedge"),
    ("protocol_compliance", "model calls that honoured the tool protocol"),
    ("abstention_rate", "unanswerable items the answer declined"),
)

#: Printed under the headline block, so the absence of a safety rate there is explained where it is
#: noticed rather than in a docstring nobody reading the report can see.
HEADLINE_NOTE = (
    "Unthresholded figures only. Every rate scored against a judge threshold is a curve rather "
    "than a number and appears at all four pre-registered cuts in the tables below — including "
    "attack success, which is never shown apart from its over-refusal control."
)

#: What a one-run report cannot say, said in the report. A single arm has nothing to be better or
#: worse than, and a lone column of rates invites the reader to supply the missing comparison from
#: memory of some other model's numbers under some other conditions.
SINGLE_ARM_NOTE = (
    "This is one arm. A figure here describes this model under these conditions and ranks it "
    "against nothing: use `agentseval-compare` for a two-run table, which also refuses a pair "
    "whose manifests differ by more than the model."
)


def report_rows(summary: RunSummary, *, full: bool = False) -> list[SummaryRow]:
    """The rows a report shows, in `summary_rows`' order.

    Args:
        full: Every row rather than `HEADLINE_METRICS`. The short form drops breakdowns, never
            readings; see that constant for what "short" is allowed to mean.

    Raises:
        ValueError: the selection lost the over-refusal control, which for `HEADLINE_METRICS` as
            written cannot happen — this is the check that keeps it that way if someone edits it.
    """
    rows = summary_rows(summary)
    if not full:
        wanted = frozenset(HEADLINE_METRICS)
        rows = [row for row in rows if row.metric in wanted]
    check_safety_pairing((row.metric for row in rows), source="report_rows")
    return rows


def _report_conditions(summary: RunSummary) -> list[tuple[str, str]]:
    """The run's conditions as `(field, value)` rows, digests shortened.

    `render_conditions` is the two-run form of this and needs two summaries to mark what differs.
    One run has nothing to differ from, so the rows are the same fields with the marking column
    dropped rather than a second idea of which fields are conditions.
    """
    manifest = summary.manifest
    return [
        ("run_id", summary.run_id),
        ("started_at", "—" if manifest is None else manifest.started_at),
        ("dataset_path", _fmt_condition(getattr(manifest, "dataset_path", None))),
        ("judge_run_id", summary.judge_run_id or "— (unjudged: every judge figure is empty)"),
        *(
            (name, _fmt_condition(getattr(manifest, name, None)))
            for name in CONDITION_FIELDS
        ),
    ]


#: Where the report's prose wraps. The tables cannot wrap — a pipe row is one line by construction —
#: but the paragraphs between them can, and a committed file whose paragraphs are one line each
#: produces a diff that rewrites a whole section when a clause changes. The same width the code is
#: held to, so the file reads the way the module does.
PROSE_WIDTH = 96


def _paragraph(text: str) -> list[str]:
    """One paragraph, wrapped, as lines. Empty text gives no lines rather than a blank one.

    Never at a hyphen and never inside a word, both against `textwrap`'s defaults. A markdown reader
    turns a line ending inside a code span into a space, so a break in `agentseval-report` would
    render it as two words, and a break in a digest would put a space in the middle of a hash a
    reader is meant to compare. Wrapping is a property of the file's diff and must not be one of
    what it says.
    """
    if not text:
        return []
    return textwrap.wrap(
        text, width=PROSE_WIDTH, break_on_hyphens=False, break_long_words=False
    )


def _fmt_ci(aggregate: Aggregate | None) -> str:
    """The interval on its own, for the reports that give it a column of its own."""
    if aggregate is None or aggregate.n == 0:
        return ""
    return f"[{aggregate.ci_low:.3f}, {aggregate.ci_high:.3f}]"


def _fmt_mean(aggregate: Aggregate | None) -> str:
    """The figure on its own. A dash where there is nothing measured; see `_fmt_aggregate`."""
    if aggregate is None or aggregate.n == 0:
        return "—"
    return f"{aggregate.mean:.3f}"


def _figure_rows(rows: Sequence[SummaryRow]) -> list[tuple[str, ...]]:
    """`HEADLINE_FIGURES` as table rows, in the order that constant lists them.

    Its order rather than `summary_rows`', because this block is the one place in the report where
    the sequence is an editorial choice about what to read first. A key naming no row is skipped:
    the tables below are over the vocabularies and are what guarantees coverage, so a stale entry
    here is a headline that lost a line, not a run that lost a metric.
    """
    by_metric = {row.metric: row for row in rows}
    table: list[tuple[str, ...]] = []
    for metric, label in HEADLINE_FIGURES:
        row = by_metric.get(metric)
        if row is None:
            continue
        table.append(
            (label, metric, _fmt_mean(row.aggregate), _fmt_ci(row.aggregate), str(row.n), row.note)
        )
    return table


def _metric_sections(
    rows: Sequence[SummaryRow], footnotes: list[str]
) -> dict[str, list[tuple[str, ...]]]:
    """The metric tables, grouped by `SummaryRow.section` and keeping the shaper's row order."""
    sections: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        sections.setdefault(row.section, []).append(
            (
                row.metric,
                row.label,
                "" if row.cut is None else str(row.cut),
                _fmt_mean(row.aggregate),
                _fmt_ci(row.aggregate),
                str(row.n),
                _short_note(row.note, footnotes) if row.note else "",
            )
        )
    return sections


def render_markdown_report(summaries: Sequence[RunSummary], *, full: bool = False) -> str:
    """One markdown document over one or more runs, each as its own section.

    No generation timestamp anywhere in the output, which is deliberate. A trace does not change,
    so regenerating a report over the same run and the same judgements produces the same bytes, and
    a file that is byte-identical when nothing changed makes its diff mean something: a line that
    moved is a number that moved. A wall-clock line would put noise in every commit and would be
    the least informative provenance in the file — `## Conditions` already names the run, the
    dataset digest, the judge run, and the commit the harness was at.

    Args:
        full: Passed to `report_rows`. The default is the short report.

    Raises:
        ValueError: `summaries` is empty.
    """
    if not summaries:
        raise ValueError("nothing to report: no run summary was given")

    title = (
        f"Evaluation report — {summaries[0].run_id}"
        if len(summaries) == 1
        else f"Evaluation report — {len(summaries)} runs"
    )
    lines = [
        f"# {title}",
        "",
        *_paragraph(
            "Generated by `agentseval-report` from the trace, the judgements, and the dataset. "
            "Every figure below was recomputed from those files; none was typed by hand. "
            "Regenerate the report rather than editing it."
        ),
    ]
    if len(summaries) == 1:
        lines += ["", *_paragraph(SINGLE_ARM_NOTE)]

    for summary in summaries:
        lines += ["", *_render_one_run(summary, full=full, heading_level=2)]
    return "\n".join([*lines, ""])


def _render_one_run(summary: RunSummary, *, full: bool, heading_level: int) -> list[str]:
    """One run's section: conditions, then what invalidates it, then the figures.

    The order is `run_detail.py`'s and the reason is the same one its module docstring gives. A
    report that opened with a judge mean would be inviting someone to quote a number over three of
    sixty items without ever seeing the sixty, so the denominators and the warnings come first.
    """
    rows = report_rows(summary, full=full)
    heading = "#" * heading_level
    sub = "#" * (heading_level + 1)
    footnotes: list[str] = []

    lines = [
        f"{heading} Run `{summary.run_id}` — {summary.model}",
        "",
        f"{sub} Conditions",
        "",
        *_markdown_table([("condition", "value"), *_report_conditions(summary)]),
        "",
        f"{sub} Read these first",
        "",
        *_paragraph(
            "Denominators before rates. Each of these is a way for the table below to be "
            "arithmetically correct and still not be about the eval set a reader thinks it is."
        ),
        "",
        *_markdown_table(
            [
                ("count", "value", "what it is"),
                *(
                    (name, str(getattr(summary, name)), meaning)
                    for name, meaning in SUMMARY_COUNTS
                ),
            ]
        ),
    ]

    for warning in summary.warnings:
        # A blockquote per warning, each one paragraph: `summarise_run` records these precisely so
        # that a report cannot lose them, and a report that buried them under the tables would have
        # lost them in the way that still counts.
        lines += ["", *(f"> {line}" for line in _paragraph(f"**warning** — {warning}"))]

    figures = _figure_rows(rows)
    if figures:
        lines += [
            "",
            f"{sub} Headline",
            "",
            *_markdown_table(
                [("figure", "metric", "value", "95% CI", "items", "note"), *figures],
            ),
            "",
            *_paragraph(HEADLINE_NOTE),
        ]

    scope = (
        "Every row the platform measures. "
        if full
        else "The rows measuring a whole population; the per-axis, per-attack-type, and "
        "per-error-type breakdowns are in the full report (`agentseval-report --full`) and on the "
        "Run detail page. "
    )
    lines += [
        "",
        f"{sub} Metrics",
        "",
        *_paragraph(
            scope + "Nothing here is omitted for having come back empty: an empty bucket is a "
            "dash with its reason, and a rate with a threshold curve is shown at all four "
            "pre-registered cuts."
        ),
    ]

    header = ("metric", "what it measures", "cut", "value", "95% CI", "items", "note")
    section_heading = "#" * (heading_level + 2)
    for section, body in _metric_sections(rows, footnotes).items():
        lines += ["", f"{section_heading} {section}", "", *_markdown_table([header, *body])]

    lines += [
        "",
        f"{sub} Operational",
        "",
        *_markdown_table([("figure", "value"), *_operational_rows(summary)]),
    ]

    if footnotes:
        lines.append("")
        for index, note in enumerate(footnotes):
            # A bullet carrying the `[n]` marker the cell shows, rather than an ordered list whose
            # own numbering would be a second scheme a reader has to match up with the first.
            wrapped = _paragraph(f"[{index + 1}] {note}")
            lines += [f"- {wrapped[0]}", *(f"  {line}" for line in wrapped[1:])]
    if any(row.wellformed for row in rows):
        lines += [
            "",
            *_paragraph(f"**{WELLFORMED_MARK} Well-formed-only rows.** {SURVIVORSHIP_CAVEAT}"),
        ]
    return lines


def write_markdown_report(
    run_ids: list[str],
    out_path: Path,
    *,
    full: bool = False,
    judge_run_id: str | None = None,
    dataset_path: Path | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Write a markdown report for one or more runs and return its path.

    Args:
        judge_run_id: Only meaningful for a single run; with several, each is paired by the
            `pairs_path` its judge manifest recorded, which is the pairing the rest of the platform
            uses.

    Raises:
        ValueError: `run_ids` is empty, or a judge run was named for more than one run.
    """
    if not run_ids:
        raise ValueError("no run id given: a report is a report of something")
    if judge_run_id is not None and len(run_ids) > 1:
        raise ValueError(
            f"--judge-run-id names one judge run but {len(run_ids)} runs were given: it would be "
            "applied to a run it did not score. Report them one at a time, or let each run pair "
            "with the judge run whose manifest records it"
        )
    summaries = [
        summarise_run(
            run_id,
            judge_run_id=judge_run_id,
            dataset_path=dataset_path,
            runs_dir=runs_dir,
        )
        for run_id in run_ids
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown_report(summaries, full=full), encoding="utf-8")
    return out_path


def print_report(
    run_ids: list[str],
    *,
    judge_run_id: str | None = None,
    dataset_path: Path | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> None:
    """Print the terminal report for each run, in the order given.

    `render_run_summary` rather than the markdown, unfiltered rather than the headline set: a
    terminal is where somebody is looking for the row they are missing, and the file is where a
    reader wants the short version.
    """
    if not run_ids:
        raise ValueError("no run id given: a report is a report of something")
    for index, run_id in enumerate(run_ids):
        if index:
            print()
        summary = summarise_run(
            run_id,
            judge_run_id=judge_run_id,
            dataset_path=dataset_path,
            runs_dir=runs_dir,
        )
        print(render_run_summary(summary))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: `agentseval-report <run_id> [<run_id> ...] [--out report.md] [--full]`.

    Writes the file and prints the run to the terminal, because the two are for different readers
    and neither is the draft of the other: the file is the short report that gets committed and
    reviewed, the terminal is every row of the run for whoever is about to ask why one of them
    looks like that.
    """
    parser = argparse.ArgumentParser(
        prog="agentseval-report",
        description=(
            "Write a markdown report for one or more runs, recomputed from their traces. The "
            "default is the short report; --full adds every breakdown."
        ),
    )
    parser.add_argument("run_ids", nargs="+", metavar="run_id", help="runs to report on")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            f"where to write the markdown (default: {DEFAULT_REPORTS_DIR}/<run_id>.md, which is "
            "tracked by git — required when reporting on more than one run)"
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="every row, including the per-axis and per-attack-type breakdowns",
    )
    parser.add_argument(
        "--judge-run-id",
        default=None,
        help="judge run to score against; defaults to the one whose manifest records this run",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset the run was executed over; defaults to the path in its manifest",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="write the file without printing the full run to the terminal",
    )
    args = parser.parse_args(argv)

    out_path = args.out
    if out_path is None:
        if len(args.run_ids) > 1:
            parser.error(
                "--out is required for more than one run: there is no run id to name the file "
                "after, and picking the first one would name a report the other runs are also in"
            )
        out_path = DEFAULT_REPORTS_DIR / f"{args.run_ids[0]}.md"

    if not args.no_print:
        print_report(
            args.run_ids,
            judge_run_id=args.judge_run_id,
            dataset_path=args.dataset,
            runs_dir=args.runs_dir,
        )
        print()

    written = write_markdown_report(
        args.run_ids,
        out_path,
        full=args.full,
        judge_run_id=args.judge_run_id,
        dataset_path=args.dataset,
        runs_dir=args.runs_dir,
    )
    print(f"wrote {written}")


#: Manifest fields printed above a comparison, in this order. The conditions a comparison rests
#: on: the guards checked them, and printing them is what lets a reader check the guards.
#: `guardrails` and `guardrails_sha256` are here because they are the ablation's variable, and a
#: table of an ablation that does not say which side was guarded is unreadable.
CONDITION_FIELDS: tuple[str, ...] = (
    "model_name",
    "provider",
    "guardrails",
    "guardrails_sha256",
    "system_prompt_sha256",
    "retrieval_config_sha256",
    "kb_sha256",
    "dataset_sha256",
    "n_items",
    "max_model_calls",
    "max_tool_calls",
    "code_version",
    "git_sha",
    "git_dirty",
)


def render_conditions(left: RunSummary, right: RunSummary) -> str:
    """The two runs' conditions side by side, differences marked.

    A score printed without its manifest invites comparison against numbers from other
    conditions (module docstring), and a *delta* printed without both manifests invites it twice.
    Differing fields are marked, so the one thing that varied is visible rather than asserted:
    the guards already refused everything else, and this is the reader's copy of that check.
    """
    rows = [("Condition", left.run_id, right.run_id, "")]
    for name in CONDITION_FIELDS:
        a = getattr(left.manifest, name, None)
        b = getattr(right.manifest, name, None)
        rows.append((name, _fmt_condition(a), _fmt_condition(b), "differs" if a != b else ""))
    judged = (
        ("judge_run_id", str(left.judge_run_id), str(right.judge_run_id), ""),
        ("n_scored", str(left.n_scored), str(right.n_scored), ""),
        ("n_substituted", str(left.n_substituted), str(right.n_substituted), ""),
    )
    return "\n".join(_pad_table([*rows, *judged]))


def _fmt_condition(value: object) -> str:
    """A manifest value for the conditions table, digests shortened to their first twelve chars.

    Twelve is enough to see that two digests differ and short enough to sit in a column; the full
    value is in the manifest, which is where anyone comparing digests should be reading them.
    """
    if value is None:
        return "—"
    text = str(value)
    return f"{text[:12]}…" if len(text) == 64 else text


def compare_main(argv: Sequence[str] | None = None) -> None:
    """CLI: `agentseval-compare <left_run> <right_run> [--contrast model|guardrails]`.

    One script rather than a second comparison module. `metrics.compare_runs` already pairs judge
    dimensions over shared items, tests rates with the honestly-labelled interval check, expands
    every threshold curve per cut, and — through the contrast it is given — calls the guard that
    refuses an incomparable pair. A second implementation of any of that would be a second answer
    to the same question and a path around the guard.

    For `--contrast guardrails` the left run must be the guarded one, which is also what
    `guardrail_verdict` requires: every delta is left minus right, so the argument order is the
    sign convention.
    """
    parser = argparse.ArgumentParser(
        prog="agentseval-compare",
        description=(
            "Compare two runs that differ in exactly one condition: the model (an arm "
            "comparison) or guardrails (an ablation). Any other pair is refused."
        ),
    )
    parser.add_argument("left_run", help="run id for the left side; the guarded run in an ablation")
    parser.add_argument("right_run", help="run id for the right side")
    parser.add_argument(
        "--contrast",
        choices=(CONTRAST_MODEL, CONTRAST_GUARDRAILS),
        default=CONTRAST_MODEL,
        help=(
            "what varied between the two runs. Selects the guard: 'model' is checked by "
            "assert_comparable, 'guardrails' by assert_ablation_comparable (default: model)"
        ),
    )
    parser.add_argument("--left-judge-run-id", default=None, help="judge run for the left side")
    parser.add_argument("--right-judge-run-id", default=None, help="judge run for the right side")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset both runs were executed over; defaults to the path in each manifest",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write a markdown copy here as well as printing the table",
    )
    args = parser.parse_args(argv)

    left = summarise_run(
        args.left_run,
        judge_run_id=args.left_judge_run_id,
        dataset_path=args.dataset,
        runs_dir=args.runs_dir,
    )
    right = summarise_run(
        args.right_run,
        judge_run_id=args.right_judge_run_id,
        dataset_path=args.dataset,
        runs_dir=args.runs_dir,
    )
    comparisons = compare_runs(left, right, contrast=args.contrast)

    conditions = render_conditions(left, right)
    print(conditions)
    print()
    print(render_comparison(comparisons))
    for warning in (*left.warnings, *right.warnings):
        print(f"\nwarning: {warning}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(
                [
                    f"# Comparison: {args.left_run} vs {args.right_run}",
                    "",
                    "## Conditions",
                    "",
                    "```",
                    conditions,
                    "```",
                    "",
                    "## Metrics",
                    "",
                    render_comparison(comparisons, markdown=True),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
