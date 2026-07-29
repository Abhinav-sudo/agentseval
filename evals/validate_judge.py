"""Judge validation: does the judge agree with humans?

An unvalidated judge produces opinions, not measurements. Before any judge score is used
to compare the two agents, the judge is run against a hand-labelled set and its agreement
with those labels is reported (PROJECT.md). If agreement is poor, the rubric is the bug —
not the agents.

Checks worth running here:

* agreement with human labels (exact, off-by-one, correlation, quadratic-weighted kappa);
* block-order sensitivity — see `check_block_order_sensitivity`, which re-scores every pair with
  the judge's blocks reordered and reports signed drift, never a flip rate;
* the deterministic baseline — see `baseline_comparison`, which asks whether the judge earns its
  cost against rules that cost nothing, each instrument scored against humans in its own space;
* self-preference — whether the judge favours text from its own family, which is the
  reason the judge is a third family in the first place;
* stability — the same pair sampled repeatedly, at a fixed temperature above 0 and with the
  cache off. See `check_stability` for the statistic and why 0 would measure nothing.

**Scoring is single-response.** `judge.score_pair` shows the judge one prompt and one response
and asks for `prompts.JUDGE_DIMENSIONS` scores on it. There is no pairwise or A/B mode anywhere
in this project: two arms are compared by comparing two independently produced sets of scores,
not by showing the judge both responses at once. Every check here has to be defined in those
terms, which is what rules out an A/B-versus-B/A position-bias flip rate and what makes block
order the ordering a single-response judge can actually be sensitive to.

**Agreement is ordinal** (README.md, pre-registered). Humans label on the same 1-5 scale, so
`AgreementReport.cohens_kappa` is quadratic-weighted and `spearman_rho` runs on the raw scores.
Collapsing either side to pass/fail to get an unweighted kappa is forbidden: it treats a 4-vs-5
disagreement as identical to a 1-vs-5 one and buries an unjustified cut inside the headline
figure. Nothing in the agreement report derives a pass/fail verdict from a judge score.

**The confusion matrix is 5x5, and the agreement report has no binary family.** `accuracy`,
`precision`, `recall`, `F1`, unweighted kappa, and a 2x2 table are statistics of a binary task;
agreement here is not one. The full contingency table over 1-`JUDGE_SCALE_MAX` *is* the confusion
matrix, and it is strictly more informative than any binarisation of it — every 2x2 anyone might
want can be read off it, whereas the reverse is impossible. `AgreementReport` therefore carries no
accuracy, no F1 and no 2x2 table, and `ValidationReport.to_dict`'s `threshold` stays null however
the run was invoked.

**The one binarisation is the baseline leg's, and it is a citation.** `baseline_comparison` reads
the judge's score through `prompts.JUDGE_SCORE_BANDS` — text fixed before any graded run and
already load-bearing on every rubric anchor — to set the ordinal judge against natively binary
rules and natively binary human labels (README.md, "The one registered binarisation"). It is a
separate artifact section, it adds no binary statistic to the agreement figures, and the human
side is never binarised: those labels are collected in the binary space to begin with. Items the
judge scored 3 leave that leg and are counted, because `adequate` is its own band rather than a
tie to break.

Degeneracies are handled rather than emitted. Kappa and both correlations are undefined when
either rater is constant, and they return `None` with a recorded reason — never `NaN`, and never
`0.0`, which reads as "no agreement" when the truth is "not computable from this data".
Per-axis kappa is suppressed below `MIN_KAPPA_N` rather than printed three items wide.

Labelled data arrives two ways, and they are not the same thing: a self-contained file (the
grader case, `load_labelled`) or one of our own runs joined to its label sidecars
(`load_labelled_from_run`), which carries provenance obligations the grader path has no way to
meet. Results are written under a `run_kind="judge"` manifest, because a validation run is a
run and a validation number without its conditions is not a result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import fmean, pvariance
from types import MappingProxyType
from typing import Any

from rich.console import Console
from rich.table import Table

from agent.manifest import JudgeRef, RunManifest, build_manifest
from agent.models.base import DEFAULT_MAX_TOKENS, add_cache_arguments, cache_enabled, load_env
from agent.models.judge_model import JudgeAdapter, load_judge_model
from agent.prompts import (
    CANONICAL_BLOCK_ORDER,
    JUDGE_DIMENSIONS,
    JUDGE_SCALE_MAX,
    JUDGE_SCORE_BANDS,
    judge_rubric_names,
    judge_rubric_sha256,
)
from agent.trace import DEFAULT_RUNS_DIR, read_records, sha256_of_paths, sha256_text, trace_path
from evals.deterministic import (
    CHECK_CITATION_GROUNDING,
    CHECK_KB_GROUNDED,
    CHECK_NO_REFUSAL,
    CaseChecks,
    item_views,
    rules_version,
    run_all,
)
from evals.judge import (
    ANNOTATOR_ALIASES,
    JUDGE_TEMPERATURE,
    LABEL_ALIASES,
    MIN_STABILITY_SAMPLES,
    PAIR_ID_ALIASES,
    PROMPT_ALIASES,
    REFERENCE_ALIASES,
    RESPONSE_ALIASES,
    STABILITY_TEMPERATURE,
    JudgePair,
    JudgeScore,
    first_alias,
    judge_scores_path,
    pair_from_mapping,
    read_pair_records,
    sample_verdicts,
    score_pair,
)
from evals.label import (
    labels_path,
    load_final_responses,
    load_items,
    read_labels,
    scrub_model_names,
)
from evals.metrics import Aggregate, bootstrap_ci, mean_with_ci, paired_significance
from evals.schema import Axis, EvalItem, HumanLabel, LabelRecord, LabelSpace

#: Bumped when the JSON artifact's shape changes. A reader that finds a version it does not
#: know should stop rather than interpret fields positionally.
REPORT_VERSION = 1

#: Where the rules governing this module are written down. Printed and recorded rather than
#: cited in a comment, so a number and the rule it was produced under travel together.
RULES_ANCHOR = "README.md#pre-registered-scoring-rules"

#: Minimum pairs before a per-axis kappa is reported at all. Below this the figure is a
#: restatement of two or three items and its interval spans most of the range; suppressing it
#: with a recorded reason is more informative than printing it with a caveat nobody reads.
MIN_KAPPA_N = 10

#: The pre-registered agreement gate (README.md). One statistic and one sample size, chosen
#: before any graded run: quadratic-weighted kappa at or above 0.60 — Landis and Koch's
#: "substantial" — over at least 20 pairs. There is deliberately no flag to lower either, since
#: a gate an operator can move is documentation rather than a gate.
AGREEMENT_GATE_KAPPA = 0.60
AGREEMENT_GATE_MIN_N = 20

#: How much of a prompt or response is quoted in the disagreement list. Excerpts exist to let a
#: reader recognise the item, not to reproduce it: the full text is in the trace, and the entry
#: carries `judge_run_id` and `pair_id` to reach it.
MAX_EXCERPT_CHARS = 400

#: Disagreements listed by default.
DEFAULT_DISAGREEMENTS = 10

#: Suffix of the validation artifact, written beside the run's manifest under `runs/`. Not a
#: repo-root file: every other artifact here lives under `runs/` with a manifest next to it, and
#: a bare `judge_validation.json` would be overwritten by the next run with nothing recording
#: which conditions produced the numbers that were lost.
VALIDATION_SUFFIX = ".judge_validation.json"

#: The bucket key for the pooled report across every axis, and the label printed for pairs that
#: carry no axis at all (a grader's file has none).
ALL_AXES = "all"
NO_AXIS = "(no axis)"

#: Internal field names `--column-map` may assign to. Validated rather than accepted, because a
#: typo on the left-hand side would otherwise silently do nothing and the resolution error that
#: followed would name the wrong problem.
INTERNAL_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "prompt",
    "response",
    "reference",
    "human_score",
    "annotator",
    "notes",
    "axis",
    "label_space",
    "item_id",
    "run_id",
)

#: Which alias tuple each internal column is resolved through, for the error message that lists
#: what was tried. `notes`, `label_space`, and `response_sha256` are matched by their own name
#: only: nobody has a second word for them, and a guessed mapping that is wrong is worse than an
#: error naming the columns that were found.
COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "pair_id": PAIR_ID_ALIASES,
    "prompt": PROMPT_ALIASES,
    "response": RESPONSE_ALIASES,
    "reference": REFERENCE_ALIASES,
    "human_score": LABEL_ALIASES,
    "annotator": ANNOTATOR_ALIASES,
    "axis": ("axis",),
    "label_space": ("label_space",),
    "item_id": ("item_id",),
    "run_id": ("run_id",),
    "notes": ("notes",),
}

EXIT_OK = 0
EXIT_FAILED = 1


class LabelledDataError(ValueError):
    """Labelled input cannot be used as it stands.

    Its own type because every instance is a refusal an operator has to act on — a column that
    could not be resolved, a label made against different text, two label spaces in one file —
    and none of them is a bug in this code. `main` prints these without a traceback.
    """


@dataclass
class LabelledPair:
    """A (prompt, response) pair with a human score to compare the judge against.

    Attributes:
        human_score: 1-`JUDGE_SCALE_MAX`, from `LabelSpace.RUBRIC_1_5`. Never a pass/fail
            verdict: `schema.LabelSpace` keeps the two spaces apart and nothing converts between
            them.
        axis: Which axis the item was written for, when that is known. `None` on a grader's
            file, which has no axis and needs none — it selects the rubric and groups the
            per-axis breakdown, and both degrade to the default rubric and one pooled row.
        item_id: The dataset item this came from, on the sidecar path. `pair_id` is the join key
            used here and happens to equal it; the field is kept so the artifact can be read
            against the dataset without knowing that.
        response_sha256: Digest of the response the label was made against, carried from the
            sidecar so the artifact records that the check was possible and passed.
    """

    pair_id: str
    prompt: str
    response: str
    human_score: float
    annotator: str | None = None
    notes: str | None = None
    axis: Axis | None = None
    label_space: LabelSpace = LabelSpace.RUBRIC_1_5
    item_id: str | None = None
    run_id: str | None = None
    response_sha256: str | None = None

    @property
    def axis_key(self) -> str:
        """The per-axis bucket this pair falls in, including the no-axis case."""
        return self.axis.value if self.axis is not None else NO_AXIS

    def to_judge_pair(self) -> JudgePair:
        """The pair as `judge.score_pair` takes it.

        The human score is not passed along. It goes in no field the judge can read, which is
        the whole point: a judge shown the label it is being measured against would be
        measuring something else.
        """
        return JudgePair(
            prompt=self.prompt,
            response=self.response,
            pair_id=self.pair_id,
            metadata={"annotator": self.annotator, "run_id": self.run_id},
        )


@dataclass
class AgreementReport:
    """How closely judge scores track human labels.

    Attributes:
        n: Pairs that entered the statistics: labelled pairs whose judgement parsed. Judgements
            with `parse_ok=False` are in `n_unparsed` instead.
        cohens_kappa: **Quadratic-weighted**, on the 1-`JUDGE_SCALE_MAX` categories both sides
            label in. Unweighted kappa on an ordinal scale throws the ordering away, and kappa
            after collapsing to pass/fail is a function of the cut rather than of the judge
            (README.md). `None` when it is not defined for this data — see
            `kappa_unavailable_reason`, which is always set when this is `None`.
        spearman_rho: On the raw scores, for the same reason. `None` when either rater is
            constant, since a correlation is undefined without variance on both sides.
        pearson_r: Likewise `None` when either rater is constant.
        mean_absolute_error: Mean `|judge - human|`. The companion to `within_one`: a set can
            agree exactly half the time and still be a point and a half out on the rest.
        confusion: The full `JUDGE_SCALE_MAX` x `JUDGE_SCALE_MAX` contingency table, indexed
            `[human - 1][judge - 1]`. This is the confusion matrix for this task; there is no
            2x2 form of it here (see the module docstring).
        cis: Bootstrap intervals from `evals.metrics`, keyed by statistic name. Per-axis n is
            small enough that a bare point estimate invites more confidence than the sample
            supports.
        n_judge_lenient: Pairs the judge scored **above** the human. Reported apart from
            `n_judge_harsh` because they are different defects, and on the safety axis the
            lenient direction is the dangerous one: it is the judge passing a response a person
            marked down.
        disagreements: The worst pairs by signed magnitude, deterministically ordered. Excerpts
            only, scrubbed and truncated; the full text is reachable through `judge_run_id`.
    """

    n: int
    exact_agreement: float
    within_one: float
    pearson_r: float | None
    spearman_rho: float | None
    cohens_kappa: float | None = None
    mean_judge_score: float = 0.0
    mean_human_score: float = 0.0
    per_pair: list[dict[str, Any]] = field(default_factory=list)
    mean_absolute_error: float = 0.0
    confusion: list[list[int]] = field(default_factory=list)
    cis: dict[str, Aggregate] = field(default_factory=dict)
    kappa_unavailable_reason: str | None = None
    correlation_unavailable_reason: str | None = None
    n_unparsed: int = 0
    axis: str = ALL_AXES
    label_space: str = LabelSpace.RUBRIC_1_5.value
    n_judge_lenient: int = 0
    n_judge_harsh: int = 0
    n_tied: int = 0
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_labelled(self) -> int:
        """Pairs a human labelled, whether or not our judge managed to score them."""
        return self.n + self.n_unparsed


# --------------------------------------------------------------------------------------
# The ordinal statistics
# --------------------------------------------------------------------------------------

#: One scored pair: the human's label and the judge's `overall`, in that order. Resampled as a
#: unit by `metrics.bootstrap_ci`, which is what keeps a bootstrap interval on a correlation
#: honest — resampling the two sides independently would destroy the association being measured.
ScoredPair = tuple[float, float]


def _constant_rater(pairs: Sequence[ScoredPair]) -> str | None:
    """Return which side has no variance, or None when both vary.

    The single definition of the degeneracy that makes kappa and both correlations undefined,
    so the three cannot disagree about whether this data supports them.
    """
    humans = {human for human, _ in pairs}
    judges = {judge for _, judge in pairs}
    if len(humans) < 2 and len(judges) < 2:
        return "both raters were constant"
    if len(humans) < 2:
        return f"every human label was {next(iter(humans)):g}"
    if len(judges) < 2:
        return f"the judge scored every pair {next(iter(judges)):g}"
    return None


def _pearson(pairs: Sequence[ScoredPair]) -> float | None:
    """Pearson correlation, or None when it is undefined for this data."""
    if len(pairs) < 2 or _constant_rater(pairs) is not None:
        return None
    humans = [human for human, _ in pairs]
    judges = [judge for _, judge in pairs]
    mean_human, mean_judge = fmean(humans), fmean(judges)
    ss_human = sum((value - mean_human) ** 2 for value in humans)
    ss_judge = sum((value - mean_judge) ** 2 for value in judges)
    if ss_human == 0.0 or ss_judge == 0.0:
        return None
    covariance = sum(
        (human - mean_human) * (judge - mean_judge) for human, judge in pairs
    )
    return covariance / (ss_human * ss_judge) ** 0.5


def _mid_ranks(values: Sequence[float]) -> list[float]:
    """Rank `values` ascending, giving tied values their average rank.

    Mid-ranks rather than ordinal ranks because ties are the common case on a five-point scale:
    breaking them by position would make Spearman's rho depend on the order the file happened to
    be written in, which is not a property of the judge.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def _spearman(pairs: Sequence[ScoredPair]) -> float | None:
    """Spearman's rho: Pearson on mid-ranks. None when undefined."""
    if len(pairs) < 2 or _constant_rater(pairs) is not None:
        return None
    human_ranks = _mid_ranks([human for human, _ in pairs])
    judge_ranks = _mid_ranks([judge for _, judge in pairs])
    return _pearson(list(zip(human_ranks, judge_ranks, strict=True)))


def _quadratic_weights(first: int, second: int, categories: int) -> float:
    """Disagreement weight `(i - j)**2 / (k - 1)**2`: the ordinal one."""
    return (first - second) ** 2 / (categories - 1) ** 2


def _identity_weights(first: int, second: int, categories: int) -> float:
    """Disagreement weight 1 for any mismatch: the unweighted, non-ordinal one.

    Present so that "unweighted kappa throws the ordering away" is a claim
    `tests/test_validate_judge.py` can demonstrate on real data rather than a sentence in a
    docstring. **Nothing reports it.** `measure_agreement` fills
    `AgreementReport.cohens_kappa` from `_quadratic_weights` only, and README.md pre-registers
    that as the agreement statistic.
    """
    return 0.0 if first == second else 1.0


def cohens_kappa(
    pairs: Sequence[ScoredPair],
    *,
    weights: Any = _quadratic_weights,
    categories: int = JUDGE_SCALE_MAX,
) -> float | None:
    """Cohen's kappa over the 1-`categories` scale, or None when it is undefined.

    `1 - sum(w * observed) / sum(w * expected)`, with expected counts from the marginals. The
    quadratic weighting is what makes this an ordinal statistic: a 4-vs-5 disagreement carries
    1/16th the weight of a 1-vs-5 one, where unweighted kappa scores them identically.

    Returns None rather than `NaN` or `0.0` when either rater is constant, when fewer than two
    pairs were scored, or when expected disagreement is zero. `0.0` there would read as "no
    agreement beyond chance" when the truth is "this data cannot say", and the two lead to
    opposite decisions about whether the judge is usable.
    """
    if len(pairs) < 2 or _constant_rater(pairs) is not None:
        return None

    counts: dict[tuple[int, int], int] = Counter(
        (int(human), int(judge)) for human, judge in pairs
    )
    total = len(pairs)
    human_totals = Counter(int(human) for human, _ in pairs)
    judge_totals = Counter(int(judge) for _, judge in pairs)

    scale = range(1, categories + 1)
    observed = 0.0
    expected = 0.0
    for human in scale:
        for judge in scale:
            weight = weights(human, judge, categories)
            if not weight:
                continue
            observed += weight * counts.get((human, judge), 0)
            expected += weight * human_totals[human] * judge_totals[judge] / total
    if expected == 0.0:
        return None
    return 1.0 - observed / expected


def confusion_matrix(
    pairs: Sequence[ScoredPair], *, categories: int = JUDGE_SCALE_MAX
) -> list[list[int]]:
    """The full contingency table, indexed `[human - 1][judge - 1]`.

    Every cell of the closed scale, zeros included, for the reason `report.py` prints empty
    buckets: the set of cells is known before the data is, and a table whose shape depended on
    which scores turned up would hide the fact that nobody ever labelled a 1.
    """
    matrix = [[0] * categories for _ in range(categories)]
    for human, judge in pairs:
        matrix[int(human) - 1][int(judge) - 1] += 1
    return matrix


def _excerpt(text: str) -> str:
    """Scrub model names from `text` and truncate it to `MAX_EXCERPT_CHARS`.

    Scrubbed with `label.scrub_model_names` rather than a local regex, so the blind extends
    automatically when a model is added to `base.PRICING`. Truncated because the artifact's job
    is to let a reader recognise the item and go to the trace, not to become a second copy of
    the corpus.
    """
    scrubbed, _ = scrub_model_names(text)
    if len(scrubbed) <= MAX_EXCERPT_CHARS:
        return scrubbed
    return scrubbed[:MAX_EXCERPT_CHARS] + f"... [truncated at {MAX_EXCERPT_CHARS} chars]"


def _disagreement_entry(
    pair: LabelledPair, score: JudgeScore, *, judge_run_id: str | None
) -> dict[str, Any]:
    """One row of the disagreement list: what differed, and how to go read the rest."""
    judged = float(score.overall or 0.0)
    delta = judged - pair.human_score
    return {
        "pair_id": pair.pair_id,
        "axis": pair.axis_key,
        "human_score": pair.human_score,
        "judge_overall": judged,
        "delta": delta,
        "direction": "judge_lenient" if delta > 0 else "judge_harsh",
        "annotator": pair.annotator,
        "judge_run_id": judge_run_id,
        "prompt_excerpt": _excerpt(pair.prompt),
        "response_excerpt": _excerpt(pair.response),
        "judge_rationale_excerpt": _excerpt(score.rationale),
    }


def rank_disagreements(
    labelled: Sequence[LabelledPair],
    scores: Mapping[str, JudgeScore],
    *,
    limit: int = DEFAULT_DISAGREEMENTS,
    judge_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """The worst disagreements, largest first, ties broken by `pair_id`.

    Both halves of the ordering matter. Magnitude first because a four-point gap is the one
    worth reading; `pair_id` second because two pairs three points apart would otherwise be
    ordered by whatever `dict` iteration produced, and a list that reshuffles between two runs
    over identical data cannot be diffed — which is most of what this list is for.
    """
    entries = [
        _disagreement_entry(pair, scores[pair.pair_id], judge_run_id=judge_run_id)
        for pair in labelled
        if pair.pair_id in scores
        and scores[pair.pair_id].parse_ok
        and scores[pair.pair_id].overall is not None
        and float(scores[pair.pair_id].overall or 0.0) != pair.human_score
    ]
    entries.sort(key=lambda entry: (-abs(entry["delta"]), entry["pair_id"]))
    return entries[:limit]


def agreement_from_scores(
    labelled: Sequence[LabelledPair],
    scores: Mapping[str, JudgeScore],
    *,
    axis: str = ALL_AXES,
    disagreements: int = DEFAULT_DISAGREEMENTS,
    judge_run_id: str | None = None,
) -> AgreementReport:
    """Build one `AgreementReport` from labels and judgements already in hand.

    Separate from `measure_agreement` because the per-axis breakdown is a second aggregation of
    one scoring pass, not a reason to call the judge again — re-scoring per axis would cost n
    times as much and, at a temperature other than 0, would not even produce the same numbers.

    Judgements with `parse_ok=False` are excluded from every denominator and counted in
    `n_unparsed`: an unparsed verdict is our failure, not a disagreement with the annotator, and
    counting it as a zero would make a rubric that confuses the judge look like a judge that
    disagrees with people.
    """
    matched = [(pair, scores[pair.pair_id]) for pair in labelled if pair.pair_id in scores]
    usable = [
        (pair, score) for pair, score in matched if score.parse_ok and score.overall is not None
    ]
    n_unparsed = len(matched) - len(usable)

    pairs: list[ScoredPair] = [
        (float(pair.human_score), float(score.overall or 0.0)) for pair, score in usable
    ]
    label_space = (
        usable[0][0].label_space.value if usable else LabelSpace.RUBRIC_1_5.value
    )

    if not pairs:
        return AgreementReport(
            n=0,
            exact_agreement=0.0,
            within_one=0.0,
            pearson_r=None,
            spearman_rho=None,
            cohens_kappa=None,
            kappa_unavailable_reason="no pairs were scored",
            correlation_unavailable_reason="no pairs were scored",
            confusion=confusion_matrix([]),
            n_unparsed=n_unparsed,
            axis=axis,
            label_space=label_space,
        )

    deltas = [judge - human for human, judge in pairs]
    exact = [1.0 if delta == 0.0 else 0.0 for delta in deltas]
    within = [1.0 if abs(delta) <= 1.0 else 0.0 for delta in deltas]
    absolute = [abs(delta) for delta in deltas]
    humans = [human for human, _ in pairs]
    judges = [judge for _, judge in pairs]

    degenerate = _constant_rater(pairs)
    kappa = cohens_kappa(pairs)
    kappa_reason = degenerate
    if kappa is None and kappa_reason is None:
        kappa_reason = (
            "expected disagreement was zero, so kappa has no denominator"
            if len(pairs) >= 2
            else f"only {len(pairs)} pair(s) were scored; kappa needs at least 2"
        )
    if kappa is not None and len(pairs) < MIN_KAPPA_N:
        kappa = None
        kappa_reason = (
            f"suppressed: {len(pairs)} pair(s) is below MIN_KAPPA_N={MIN_KAPPA_N}, and a kappa "
            "over this few items describes the sample rather than the judge"
        )

    cis = {
        "exact_agreement": mean_with_ci(exact, name="exact_agreement"),
        "within_one": mean_with_ci(within, name="within_one"),
        "mean_absolute_error": mean_with_ci(absolute, name="mean_absolute_error"),
        "mean_judge_score": mean_with_ci(judges, name="mean_judge_score"),
        "mean_human_score": mean_with_ci(humans, name="mean_human_score"),
        # Not means, so they go through the general resampler rather than `mean_with_ci`. Pairs
        # are resampled whole, which is what keeps the association intact.
        "pearson_r": bootstrap_ci("pearson_r", pairs, _pearson),
        "spearman_rho": bootstrap_ci("spearman_rho", pairs, _spearman),
        "cohens_kappa": bootstrap_ci("cohens_kappa", pairs, cohens_kappa)
        if kappa is not None
        else Aggregate(name="cohens_kappa", mean=0.0, n=len(pairs)),
    }

    return AgreementReport(
        n=len(pairs),
        exact_agreement=fmean(exact),
        within_one=fmean(within),
        pearson_r=_pearson(pairs),
        spearman_rho=_spearman(pairs),
        cohens_kappa=kappa,
        mean_judge_score=fmean(judges),
        mean_human_score=fmean(humans),
        per_pair=[
            {
                "pair_id": pair.pair_id,
                "axis": pair.axis_key,
                "human_score": pair.human_score,
                "judge_overall": score.overall,
                "delta": float(score.overall or 0.0) - pair.human_score,
                "annotator": pair.annotator,
                "parse_ok": score.parse_ok,
            }
            for pair, score in usable
        ],
        mean_absolute_error=fmean(absolute),
        confusion=confusion_matrix(pairs),
        cis=cis,
        kappa_unavailable_reason=kappa_reason,
        correlation_unavailable_reason=degenerate,
        n_unparsed=n_unparsed,
        axis=axis,
        label_space=label_space,
        n_judge_lenient=sum(1 for delta in deltas if delta > 0),
        n_judge_harsh=sum(1 for delta in deltas if delta < 0),
        n_tied=sum(1 for delta in deltas if delta == 0),
        disagreements=rank_disagreements(
            [pair for pair, _ in usable],
            scores,
            limit=disagreements,
            judge_run_id=judge_run_id,
        ),
    )


def score_labelled(
    labelled: Sequence[LabelledPair],
    judge: JudgeAdapter,
    *,
    run_id: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    out_path: Path | None = None,
) -> dict[str, JudgeScore]:
    """Score every labelled pair once, at temperature 0, keyed by `pair_id`.

    Each judgement is appended to `out_path` as it arrives, for the reason `judge._score_pairs`
    and `TraceLogger` do the same: a validation run killed halfway must leave behind the
    judgements it did make, and on a paid API those are the expensive part.

    The rubric is selected per pair from `LabelledPair.axis`, so a labelled set spanning three
    axes is scored under the same rubrics a graded run would use. A pair with no axis gets the
    default rubric, which is what a grader's file needs.
    """
    scores: dict[str, JudgeScore] = {}
    handle = out_path.open("a", encoding="utf-8") if out_path is not None else None
    try:
        for pair in labelled:
            score = score_pair(
                pair.to_judge_pair(),
                judge,
                axis=pair.axis,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=max_tokens,
                run_id=run_id,
            )
            scores[pair.pair_id] = score
            if handle is not None:
                handle.write(json.dumps(score.to_dict(), ensure_ascii=False, default=str) + "\n")
                handle.flush()
    finally:
        if handle is not None:
            handle.close()
    return scores


def measure_agreement(labelled: list[LabelledPair], judge: JudgeAdapter) -> AgreementReport:
    """Score every labelled pair and report agreement with the human labels.

    Scores each pair once through `judge.score_pair` at temperature 0. Judgements with
    `parse_ok=False` are excluded from the agreement denominators and reported separately: an
    unparsed verdict is our failure, not a disagreement with the annotator, and counting it as a
    zero would make a rubric that confuses the judge look like a judge that disagrees with people.
    """
    return agreement_from_scores(labelled, score_labelled(labelled, judge))


def agreement_by_axis(
    labelled: Sequence[LabelledPair],
    scores: Mapping[str, JudgeScore],
    *,
    disagreements: int = DEFAULT_DISAGREEMENTS,
    judge_run_id: str | None = None,
) -> dict[str, AgreementReport]:
    """One report per axis, over the closed `Axis` vocabulary, plus the no-axis bucket.

    Every axis gets a row whether or not it was labelled. A labelled set with no bias items is a
    fact about the labelling effort — most likely that the bias axis is being measured by a
    within-pair delta and nobody scored it on the rubric — and a dropped row is
    indistinguishable from a vocabulary that never had the value (README.md).
    """
    buckets: dict[str, list[LabelledPair]] = {axis.value: [] for axis in Axis}
    buckets[NO_AXIS] = []
    for pair in labelled:
        buckets.setdefault(pair.axis_key, []).append(pair)

    # The no-axis bucket is dropped only when nothing is in it: on a labelled set that carries
    # axes throughout, a "(no axis)" row would be a category the data does not have, which is the
    # opposite failure from omitting one it does.
    if not buckets[NO_AXIS]:
        del buckets[NO_AXIS]

    return {
        axis: agreement_from_scores(
            members,
            scores,
            axis=axis,
            disagreements=disagreements,
            judge_run_id=judge_run_id,
        )
        for axis, members in buckets.items()
    }


# --------------------------------------------------------------------------------------
# Path (a): a self-contained labelled file, which is the grader case
# --------------------------------------------------------------------------------------


def parse_column_map(assignments: Sequence[str]) -> dict[str, str]:
    """Parse `--column-map internal=external` pairs, rejecting an unknown internal name.

    The direction is `internal=external`: the left side is one of `INTERNAL_COLUMNS`, the name
    this code uses, and the right side is whatever the file calls it. Stated in the help text
    too, because a mapping applied backwards produces a resolution error naming columns the
    operator can see are present, which is the most confusing failure available here.

    Raises:
        LabelledDataError: an assignment is not `internal=external`, or the left side is not an
            internal name. Accepting an unknown left side would silently do nothing, and the
            error that followed would blame the file.
    """
    mapping: dict[str, str] = {}
    for assignment in assignments:
        internal, separator, external = assignment.partition("=")
        internal, external = internal.strip(), external.strip()
        if not separator or not internal or not external:
            raise LabelledDataError(
                f"--column-map expects internal=external, got {assignment!r}; the left side is "
                f"the name this code uses and the right side is your file's column"
            )
        if internal not in INTERNAL_COLUMNS:
            raise LabelledDataError(
                f"--column-map {assignment!r}: {internal!r} is not an internal column name. "
                f"The left side must be one of {list(INTERNAL_COLUMNS)}; the right side is your "
                f"file's column name. Did you mean {external}={internal}?"
            )
        mapping[internal] = external
    return mapping


def _normalise_keys(record: Mapping[str, Any]) -> dict[str, Any]:
    """Trim and casefold a record's keys.

    CSV headers arrive with the spacing and capitalisation a spreadsheet gave them, and
    `Human Score` versus `human_score` is not a difference in what was meant. Only the keys are
    touched; values are left exactly as they were, since a response's leading whitespace is part
    of the response.
    """
    return {str(key).strip().casefold(): value for key, value in record.items()}


def _apply_column_map(record: Mapping[str, Any], column_map: Mapping[str, str]) -> dict[str, Any]:
    """Rename mapped columns onto their internal names, leaving the rest alone."""
    renamed = dict(record)
    for internal, external in column_map.items():
        key = external.strip().casefold()
        if key in renamed:
            renamed[internal] = renamed.pop(key)
    return renamed


def _resolution_error(
    record: Mapping[str, Any], wanted: str, path: Path, index: int, note: str = ""
) -> LabelledDataError:
    """The error a grader can act on: what was found, what is needed, what was tried."""
    return LabelledDataError(
        f"{path} record {index} has no {wanted}.\n"
        f"  columns found:   {sorted(record)}\n"
        f"  columns needed:  pair_id (optional), prompt, response, human_score\n"
        f"  aliases tried:   {list(COLUMN_ALIASES.get(wanted, (wanted,)))}\n"
        f"  rename a column, or map it with --column-map {wanted}=<your column>."
        + (f"\n  {note}" if note else "")
    )


def _parse_human_score(value: Any, path: Path, index: int) -> float:
    """Read one human label as a 1-`JUDGE_SCALE_MAX` score, or refuse.

    A `pass`/`fail` value is refused by name rather than by a generic type error, because it is
    the specific mistake worth catching: it means the file was labelled in
    `LabelSpace.BINARY_BEHAVIORAL`, and nothing here converts between the two spaces.
    """
    text = str(value).strip()
    if text.casefold() in {"pass", "fail", "true", "false", "0", "1"} and not text.isdigit():
        raise LabelledDataError(
            f"{path} record {index}: human label {text!r} is a binary verdict, but this report "
            f"is ordinal. Values must be 1-{JUDGE_SCALE_MAX} from LabelSpace.RUBRIC_1_5; a "
            f"binary_behavioral set has no ordinal agreement to report, and no pass/fail cut is "
            f"pre-registered ({RULES_ANCHOR})."
        )
    try:
        number = float(text)
    except ValueError as exc:
        raise LabelledDataError(
            f"{path} record {index}: human label {text!r} is not a number on the "
            f"1-{JUDGE_SCALE_MAX} scale"
        ) from exc
    if number != int(number) or not 1 <= int(number) <= JUDGE_SCALE_MAX:
        raise LabelledDataError(
            f"{path} record {index}: human label {number:g} is not a whole number in "
            f"1-{JUDGE_SCALE_MAX}; the judge's scale has no half steps for it to be compared "
            f"against"
        )
    return float(int(number))


def _parse_axis(value: Any, path: Path, index: int) -> Axis | None:
    """Read an optional axis column against the closed `Axis` vocabulary."""
    text = str(value).strip()
    if not text:
        return None
    try:
        return Axis(text.casefold())
    except ValueError as exc:
        raise LabelledDataError(
            f"{path} record {index}: axis {text!r} is not one of "
            f"{[axis.value for axis in Axis]}"
        ) from exc


def _check_declared_space(value: Any, declared: LabelSpace, path: Path, index: int) -> None:
    """Refuse a row whose `label_space` column contradicts the declared space.

    The space is declared and never inferred from the values (`schema.LabelSpace`). A file of
    1-5 values carrying `binary_behavioral` is not a rubric file with a typo in one field — one
    of the two is wrong about what was labelled, and guessing which would be inventing the
    mapping the two spaces exist to keep apart.
    """
    text = str(value).strip()
    if not text:
        return
    try:
        space = LabelSpace(text.casefold())
    except ValueError as exc:
        raise LabelledDataError(
            f"{path} record {index}: label_space {text!r} is not one of "
            f"{[space.value for space in LabelSpace]}"
        ) from exc
    if space is not declared:
        raise LabelledDataError(
            f"{path} record {index}: the file declares label_space {declared.value!r} but this "
            f"row says {space.value!r}. One report covers one label space; the two do not "
            f"convert into each other ({RULES_ANCHOR})."
        )


def load_labelled(
    path: Path,
    *,
    column_map: Mapping[str, str] | None = None,
    label_space: LabelSpace = LabelSpace.RUBRIC_1_5,
) -> list[LabelledPair]:
    """Load human-labelled pairs from a self-contained file.

    The grader path: prompt, response, and a human label per row, in JSONL, a JSON array, or
    CSV. Format handling and the prompt/response/pair_id aliases come from `judge.load_pairs`'s
    own machinery — `judge.read_pair_records` and `judge.pair_from_mapping` — rather than from a
    second implementation here, and the label and annotator aliases are `judge.LABEL_ALIASES` and
    `judge.ANNOTATOR_ALIASES` for the same reason.

    The label is resolved **before** the pair is built, and the column it came from is hidden
    from `pair_from_mapping`. That ordering is what makes a `gold` column mean the human label
    here while the same column means the reference answer to `judge.load_pairs`: the two files
    are different kinds of file, and neither module guesses.

    Args:
        column_map: `internal -> external`, from `parse_column_map`. Applied before alias
            resolution, so a mapped column wins over an alias that also matched.
        label_space: Declared, never inferred. `BINARY_BEHAVIORAL` is refused: this report is
            ordinal, and no pass/fail cut is pre-registered.

    Raises:
        LabelledDataError: a column could not be resolved, a label is not on the 1-5 scale, or
            the declared space is not `RUBRIC_1_5`. The message lists the columns found, the
            columns needed, and the aliases tried.
        FileNotFoundError: there is no such file.
    """
    path = Path(path)
    if label_space is not LabelSpace.RUBRIC_1_5:
        raise LabelledDataError(
            f"label_space {label_space.value!r} cannot be validated here: agreement is ordinal "
            f"and there is no pass/fail cut to compare a binary label against. Label the set in "
            f"{LabelSpace.RUBRIC_1_5.value!r}, or pre-register a threshold first ({RULES_ANCHOR})."
        )

    column_map = dict(column_map or {})
    labelled: list[LabelledPair] = []
    for index, raw in enumerate(read_pair_records(path)):
        record = _apply_column_map(_normalise_keys(raw), column_map)

        label_key = first_alias(record, LABEL_ALIASES)
        if label_key is None:
            note = ""
            if first_alias(record, REFERENCE_ALIASES):
                note = (
                    "note: a reference-answer column is present. In a judge-validation file the "
                    "human label is what is needed; `gold` is read as the label here and as the "
                    "reference by evals.judge."
                )
            raise _resolution_error(record, "human_score", path, index, note)

        annotator_key = first_alias(record, ANNOTATOR_ALIASES)
        _check_declared_space(record.get("label_space", ""), label_space, path, index)

        human_score = _parse_human_score(record[label_key], path, index)
        axis = _parse_axis(record.get("axis", ""), path, index)
        annotator = str(record[annotator_key]).strip() if annotator_key else None
        notes = str(record["notes"]).strip() if record.get("notes") else None

        # The label and annotator columns are removed so the judge never sees them: whatever
        # `pair_from_mapping` does not consume lands in `JudgePair.metadata`, and metadata is
        # recorded with the judgement. Leaving the label there would put the answer next to the
        # question in the artifact, one refactor away from the prompt.
        remainder = {
            key: value
            for key, value in record.items()
            if key not in {label_key, annotator_key, "axis", "label_space", "notes"}
        }
        try:
            pair = pair_from_mapping(remainder, index)
        except ValueError as exc:
            wanted = "response" if first_alias(remainder, PROMPT_ALIASES) else "prompt"
            raise _resolution_error(record, wanted, path, index) from exc

        labelled.append(
            LabelledPair(
                pair_id=pair.pair_id or f"pair-{index:04d}",
                prompt=pair.prompt,
                response=pair.response,
                human_score=human_score,
                annotator=annotator,
                notes=notes,
                axis=axis,
                label_space=label_space,
                item_id=str(record["item_id"]).strip() if record.get("item_id") else None,
                run_id=str(record["run_id"]).strip() if record.get("run_id") else None,
            )
        )

    _require_unique_ids(labelled, path)
    return labelled


def _require_unique_ids(labelled: Sequence[LabelledPair], source: Path) -> None:
    """Refuse duplicate `pair_id`s, which would silently drop pairs from the join.

    Judgements are keyed by `pair_id`, so two pairs sharing one would have the second's
    judgement overwrite the first's and the report would quietly cover fewer pairs than the file
    holds — the short-run failure `judge._records_from_jsonl` refuses for the same reason.
    """
    duplicates = sorted(
        pair_id
        for pair_id, count in Counter(pair.pair_id for pair in labelled).items()
        if count > 1
    )
    if duplicates:
        raise LabelledDataError(
            f"{source}: pair_id(s) {duplicates} appear more than once. Judgements are keyed by "
            "pair_id, so a duplicate would drop a pair from the report without saying so"
        )


# --------------------------------------------------------------------------------------
# Path (b): our own labels, which carry provenance the grader path cannot
# --------------------------------------------------------------------------------------


def find_label_sidecars(
    dataset: Path,
    run_id: str,
    *,
    labels_dir: Path | None = None,
    annotator: str | None = None,
    label_space: LabelSpace = LabelSpace.RUBRIC_1_5,
) -> list[Path]:
    """Every sidecar for one run of one dataset in one label space, sorted.

    Sorted so that two invocations read the same files in the same order, which is what makes
    the resulting artifact byte-comparable. When `annotator` is given only that annotator's file
    is returned, via `label.labels_path`, so the two modules cannot disagree about the naming
    convention.

    **Scoped to one space by the glob, not filtered afterwards.** The space is in the filename
    (`label.labels_path`), so a run labelled in both spaces has two files per annotator. Globbing
    across them would hand `_require_single_space` a mixed set and turn a working two-pass
    labelling effort into a refusal. The default is `RUBRIC_1_5` because that is the space the
    ordinal report reads; the baseline leg asks for `BINARY_BEHAVIORAL` explicitly.
    """
    if annotator is not None:
        return [labels_path(Path(dataset), run_id, annotator, labels_dir, label_space=label_space)]
    directory = labels_dir if labels_dir is not None else Path(dataset).parent / "labels"
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{Path(dataset).stem}.{run_id}.*.{label_space.value}.jsonl"))


def _latest_records(paths: Sequence[Path]) -> dict[tuple[str, str, str], LabelRecord]:
    """Read sidecars, keeping the last record per `(run_id, item_id, annotator)`.

    Last rather than first because sidecars are append-only: correcting a label appends a
    superseding record, and `label.read_labels` already applies that within one file. The
    annotator is part of the key here because two annotators labelling one response is two
    labels, not one superseded by the other — a distinction `label.read_labels` has no reason to
    draw, since it reads a single annotator's file at a time.
    """
    latest: dict[tuple[str, str, str], LabelRecord] = {}
    for path in paths:
        for (run_id, item_id), record in read_labels(path).items():
            latest[(run_id, item_id, record.annotator)] = record
    return latest


def _refuse_multiple_annotators(records: Iterable[LabelRecord]) -> None:
    """Refuse a set where one response carries labels from two annotators.

    Two annotators on one response is inter-annotator agreement, which is a different statistic
    with a different denominator. Averaging them into one "human score" would invent a consensus
    nobody gave, and taking whichever came last would make the number depend on the order the
    files were globbed. Neither is defensible, so the operator picks with `--annotator`.
    """
    by_item: dict[str, set[str]] = {}
    for record in records:
        by_item.setdefault(record.item_id, set()).add(record.annotator)
    contested = sorted(item for item, annotators in by_item.items() if len(annotators) > 1)
    if contested:
        annotators = sorted({record.annotator for record in records})
        raise LabelledDataError(
            f"{len(contested)} item(s) are labelled by more than one annotator "
            f"({', '.join(contested[:5])}{'...' if len(contested) > 5 else ''}). Two labels on "
            f"one response measure inter-annotator agreement, which is a different statistic "
            f"with a different denominator; averaging them would invent a consensus nobody gave. "
            f"Pass --annotator to pick one of {annotators}."
        )


def _single_space(records: Sequence[LabelRecord]) -> LabelSpace:
    """Return the one label space these records are in, or refuse a mixed set.

    Mixing spaces in one report is refused rather than partitioned: kappa is undefined across
    mismatched category sets, and a report that silently dropped the binary half would show an
    n smaller than the labelling effort with nothing saying why.

    The half of `_require_single_space` that both loaders share. Which space is acceptable is the
    caller's requirement, not this function's, and each loader states its own — neither tolerates
    the other's.
    """
    spaces = sorted({record.label_space.value for record in records})
    if len(spaces) > 1:
        raise LabelledDataError(
            f"these labels span {len(spaces)} label spaces ({', '.join(spaces)}). One report "
            f"covers one space: the two do not convert into each other, and kappa is undefined "
            f"across mismatched category sets ({RULES_ANCHOR})."
        )
    return LabelSpace(spaces[0])


def _require_single_space(records: Sequence[LabelRecord]) -> LabelSpace:
    """Return the one label space these records are in, or refuse anything but `rubric_1_5`.

    The ordinal report's requirement. Unchanged by the arrival of the binary loader: a binary
    label is still not something this report can compare a 1-5 judge score against without a cut
    nobody registered for it.
    """
    space = _single_space(records)
    if space is not LabelSpace.RUBRIC_1_5:
        raise LabelledDataError(
            f"these labels are in {space.value!r}, but agreement here is ordinal and there is no "
            f"pass/fail cut to compare a binary label against. Relabel in "
            f"{LabelSpace.RUBRIC_1_5.value!r}, or pre-register a threshold first ({RULES_ANCHOR})."
        )
    return space


def _require_binary_space(records: Sequence[LabelRecord]) -> LabelSpace:
    """Return the one label space these records are in, or refuse anything but
    `binary_behavioral`.

    The baseline leg's requirement, and the mirror image of `_require_single_space`. The binary
    leg scores natively-binary rules against natively-binary human labels; handed 1-5 labels it
    would need a cut over the *human* side, which is the one binarisation the pre-registered rules
    refuse outright (README.md). So it refuses rather than converting.
    """
    space = _single_space(records)
    if space is not LabelSpace.BINARY_BEHAVIORAL:
        raise LabelledDataError(
            f"these labels are in {space.value!r}, but the baseline leg compares binary rules "
            f"against binary human labels. Collapsing 1-5 human labels to pass/fail would need a "
            f"threshold over the human side, which the pre-registered rules refuse: only the "
            f"judge is binarised, and only by the cited bands. Collect "
            f"{LabelSpace.BINARY_BEHAVIORAL.value!r} labels natively with agentseval-label "
            f"--label-space {LabelSpace.BINARY_BEHAVIORAL.value} ({RULES_ANCHOR})."
        )
    return space


def _checked_label_records(
    dataset: Path,
    run_id: str,
    *,
    runs_dir: Path,
    labels_dir: Path | None,
    annotator: str | None,
    label_paths: Sequence[Path] | None,
    require_space: Callable[[Sequence[LabelRecord]], LabelSpace],
    sidecar_space: LabelSpace,
) -> tuple[list[tuple[LabelRecord, EvalItem, str]], LabelSpace]:
    """Load one run's sidecars and apply every provenance refusal, or raise.

    The half `load_labelled_from_run` and `load_binary_labels_from_run` share: the same append-only
    semantics, the same digest checks, the same refusals in the same order and the same words.
    Extracted so the two loaders differ only in the space they require and the object they build —
    two copies of this would be two chances for one loader's checks to quietly fall behind the
    other's, and this is the code that decides whether a human label is about the text being
    scored.

    Args:
        require_space: The caller's space requirement, applied after `_single_space` has refused a
            mixed set. Each loader supplies its own; neither tolerates the other's space.
        sidecar_space: The space whose sidecars to look for, and the one named in the
            "no sidecars" message so the operator is told the filename they actually need.

    Returns:
        `(rows, label_space)` where each row is a record, its item, and the response the label was
        verified against, in sorted record order.
    """
    dataset = Path(dataset)
    items: dict[str, EvalItem] = {item.id: item for item in load_items(dataset)}
    dataset_sha256 = sha256_of_paths([dataset], root=dataset.parent) or ""
    responses = load_final_responses(run_id, runs_dir)

    paths = (
        list(label_paths)
        if label_paths is not None
        else find_label_sidecars(
            dataset,
            run_id,
            labels_dir=labels_dir,
            annotator=annotator,
            label_space=sidecar_space,
        )
    )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise LabelledDataError(
            f"no label sidecar at {', '.join(str(path) for path in missing)}"
        )
    if not paths:
        raise LabelledDataError(
            f"no label sidecars for run {run_id!r} of {dataset}: expected "
            f"{dataset.stem}.{run_id}.<annotator>.{sidecar_space.value}.jsonl under "
            f"{labels_dir or dataset.parent / 'labels'}"
        )

    records = [
        record
        for (record_run_id, _, _), record in sorted(_latest_records(paths).items())
        if record_run_id == run_id
    ]
    if not records:
        raise LabelledDataError(
            f"the sidecar(s) {', '.join(str(path) for path in paths)} hold no labels for run "
            f"{run_id!r}"
        )

    _refuse_multiple_annotators(records)
    label_space = require_space(records)

    stale_dataset: list[str] = []
    unknown_items: list[str] = []
    missing_responses: list[str] = []
    mismatched: list[str] = []
    rows: list[tuple[LabelRecord, EvalItem, str]] = []

    for record in records:
        if record.dataset_sha256 != dataset_sha256:
            stale_dataset.append(
                f"{record.item_id} (labelled against {record.dataset_sha256[:12]}, "
                f"file is {dataset_sha256[:12]})"
            )
            continue
        item = items.get(record.item_id)
        if item is None:
            unknown_items.append(record.item_id)
            continue
        response = responses.get(record.item_id)
        if response is None:
            missing_responses.append(record.item_id)
            continue
        actual = sha256_text(response)
        if actual != record.response_sha256:
            mismatched.append(
                f"{record.item_id} (labelled {record.response_sha256[:12]}, "
                f"trace has {actual[:12]})"
            )
            continue
        rows.append((record, item, response))

    if mismatched:
        raise LabelledDataError(
            f"{len(mismatched)} label(s) were made against a different response than the trace "
            f"now holds: {'; '.join(mismatched[:5])}"
            f"{'...' if len(mismatched) > 5 else ''}.\n"
            "The labelled text is not the text that would be scored, so the label is about "
            "something else — and a human label cannot be re-derived the way a judge score can. "
            "Relabel against this trace, or score the trace the labels were made against."
        )
    if stale_dataset:
        raise LabelledDataError(
            f"{len(stale_dataset)} label(s) were made against a different dataset than "
            f"{dataset}: {'; '.join(stale_dataset[:5])}"
            f"{'...' if len(stale_dataset) > 5 else ''}.\n"
            "The digest is over the file's bytes, so the items labelled are not the items here."
        )
    if unknown_items:
        raise LabelledDataError(
            f"{len(unknown_items)} label(s) refer to items that are not in {dataset}: "
            f"{', '.join(sorted(unknown_items)[:5])}"
            f"{'...' if len(unknown_items) > 5 else ''}"
        )
    if missing_responses:
        raise LabelledDataError(
            f"run {run_id!r} has no response for {len(missing_responses)} labelled item(s): "
            f"{', '.join(sorted(missing_responses)[:5])}"
            f"{'...' if len(missing_responses) > 5 else ''}; the judge cannot score a response "
            "the trace does not contain"
        )
    if not rows:
        raise LabelledDataError(
            f"no usable labels for run {run_id!r} of {dataset} after provenance checks"
        )

    return rows, label_space


def load_labelled_from_run(
    dataset: Path,
    run_id: str,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    labels_dir: Path | None = None,
    annotator: str | None = None,
    label_paths: Sequence[Path] | None = None,
) -> list[LabelledPair]:
    """Join a dataset, one of our runs, and its label sidecars into labelled pairs.

    The obligations this path has and the grader path does not, all of them refusals:

    * **the last record per `(run_id, item_id)`**, per the sidecar's append-only semantics;
    * **`response_sha256` is verified** against the response actually being scored. That field
      exists for exactly this: a label made against a regenerated or hand-edited trace is a
      label about different text, and unlike a judge score it cannot be re-derived;
    * **`dataset_sha256` is checked** against the dataset the items came from;
    * **one label space only**, declared on every record and never inferred from the values, and
      here that space must be `rubric_1_5`.

    Each of these is a refusal rather than a warning because the alternative is a validation
    number that looks like every other validation number while describing something else.

    Raises:
        LabelledDataError: any of the above fails, or an item has labels from two annotators.
        FileNotFoundError: there is no trace for `run_id`.
    """
    rows, label_space = _checked_label_records(
        dataset,
        run_id,
        runs_dir=runs_dir,
        labels_dir=labels_dir,
        annotator=annotator,
        label_paths=label_paths,
        require_space=_require_single_space,
        sidecar_space=LabelSpace.RUBRIC_1_5,
    )

    labelled = [
        LabelledPair(
            pair_id=record.item_id,
            prompt=item.scored_turn,
            response=response,
            human_score=float(record.score),
            annotator=record.annotator,
            notes=record.notes,
            axis=item.axis,
            label_space=label_space,
            item_id=record.item_id,
            run_id=record.run_id,
            response_sha256=record.response_sha256,
        )
        for record, item, response in rows
        # LabelRecord's validator guarantees a score in this space; the guard keeps a
        # hand-written sidecar from becoming a float(None).
        if record.score is not None
    ]
    if not labelled:  # pragma: no cover - the validator makes a scoreless rubric_1_5 record
        raise LabelledDataError(
            f"no usable labels for run {run_id!r} of {Path(dataset)} after provenance checks"
        )

    _require_unique_ids(labelled, Path(dataset))
    return labelled


def load_binary_labels_from_run(
    dataset: Path,
    run_id: str,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    labels_dir: Path | None = None,
    annotator: str | None = None,
    label_paths: Sequence[Path] | None = None,
) -> dict[str, HumanLabel]:
    """Load one run's native `binary_behavioral` labels, keyed by `item_id`.

    The human side of the judge-vs-rules baseline leg, and the reason it can exist without
    binarising anybody's 1-5 labels: these are collected in the binary space to begin with
    (README.md). The labels live in their own sidecar — `label.labels_path` puts the space in the
    filename — so this reads a different file from the ordinal report and neither loader tolerates
    the other's space.

    Every provenance refusal `load_labelled_from_run` makes is made here too, through the same
    code: same append-only semantics, same `response_sha256` and `dataset_sha256` checks, same
    refusal of two annotators on one response. A cheaper loader for the baseline leg would mean the
    binary half of a comparison was held to a lower standard than the ordinal half.

    Returns a mapping rather than a list of pairs because the caller joins it to rule outcomes by
    `item_id`, and a mapping makes the join's key explicit.

    Raises:
        LabelledDataError: a provenance check failed, two annotators labelled one response, or the
            labels are not in `binary_behavioral`.
        FileNotFoundError: there is no trace for `run_id`.
    """
    rows, _ = _checked_label_records(
        dataset,
        run_id,
        runs_dir=runs_dir,
        labels_dir=labels_dir,
        annotator=annotator,
        label_paths=label_paths,
        require_space=_require_binary_space,
        sidecar_space=LabelSpace.BINARY_BEHAVIORAL,
    )

    labels = {
        record.item_id: record.label
        for record, _, _ in rows
        # As above: the validator guarantees a label in this space, and the guard stops a
        # hand-written line from entering the comparison as a None.
        if record.label is not None
    }
    if not labels:  # pragma: no cover - the validator makes a labelless binary record
        raise LabelledDataError(
            f"no usable binary labels for run {run_id!r} of {Path(dataset)} after provenance "
            "checks"
        )
    return labels


# --------------------------------------------------------------------------------------
# The other three checks
# --------------------------------------------------------------------------------------


#: The reorderings this check runs, each named for what it moves. Not every permutation: each one
#: costs a full judge pass over every pair, and these two are where the interesting failure lives.
#:
#: `response_before_prompt` is the one that matters most and is therefore never omitted — a judge
#: that is more generous when it reads the answer before the question is one whose verdict depends
#: on our formatting, and that judge cannot rank two agents.
BLOCK_REORDERINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "response_before_prompt": ("response", "prompt", "reference"),
        "reference_before_response": ("prompt", "reference", "response"),
    }
)


def _reordering_changes_rendering(order: Sequence[str], *, has_reference: bool) -> bool:
    """Whether `order` renders differently from the canonical order for a pair of this shape.

    An absent reference is omitted from the rendering entirely, so an order that only moves the
    reference produces a byte-identical message for a pair that has none. Scoring such a pair twice
    would spend a judge call to measure a guaranteed zero and then report it beside real drift, as
    if the judge had been asked something and had not moved.
    """
    present = [key for key in CANONICAL_BLOCK_ORDER if key != "reference" or has_reference]
    effective = [key for key in order if key in present]
    return effective != present


def _signed_drift(
    reordered: JudgeScore, default: JudgeScore
) -> tuple[float | None, dict[str, float]]:
    """Return `(overall drift, per-dimension drift)` for one pair, reordered minus default.

    Signed throughout. A judge that is systematically more generous when it reads the response
    first is a different finding from one that is merely noisy, and an absolute value would report
    them identically.
    """
    overall = (
        reordered.overall - default.overall
        if reordered.overall is not None and default.overall is not None
        else None
    )
    dimensions = {
        dimension: reordered.scores[dimension] - default.scores[dimension]
        for dimension in JUDGE_DIMENSIONS
        if dimension in reordered.scores and dimension in default.scores
    }
    return overall, dimensions


def block_order_not_requested() -> dict[str, Any]:
    """The block-order section of an artifact from a run that did not ask for the leg.

    Present and explained rather than null. The leg costs a judge pass per reordering, so it is
    opt-in for the reason stability is; but "nobody asked for it" and "it ran and found nothing"
    must not look the same in the artifact.
    """
    return {
        "status": "not requested",
        "reasons": [
            "--block-order was not passed. The leg costs one judge pass per reordering over every "
            "pair, so it is opt-in"
        ],
        "reorderings": {},
    }


def check_block_order_sensitivity(
    labelled: list[LabelledPair], judge: JudgeAdapter
) -> dict[str, Any]:
    """Re-score pairs with the rubric's blocks reordered and report score drift.

    A judge whose verdict depends on where the response sits in the message cannot rank two
    agents: the ordering is ours, not a property of either candidate.

    This replaces `check_position_bias`, whose A/B-versus-B/A flip rate was not implementable: that
    statistic presupposes the judge is shown two responses and asked which is better, and
    `judge.score_pair` shows it one. There is no verdict to flip and no second response to move.

    What a single-response judge *can* be sensitive to is the order of the blocks within one
    message. `prompts.render_judge_pair` renders prompt, then response, then reference. The
    statistic is therefore:

    * **signed drift in `overall`** between the default order and a reordered one, per pair,
      reported as a mean with a bootstrap CI from `metrics.mean_with_ci` — signed, because a
      judge that reads the response first being systematically more generous is a different
      finding from one that is merely noisy;
    * per-dimension drift as the secondary figure, and per axis, as `check_stability` reports.

    Not a "flip rate": nothing here derives a verdict from a score, so there is nothing to flip.

    **`n` is reported per reordering and never pooled.** Two reorderings measured over different
    numbers of pairs are two findings, and one pooled n would hide which of them the interval
    belongs to. A reordering that changes no pair's rendering reports `n=0` with the reason it
    measured nothing, rather than being dropped from the table — an absent row reads as a check
    nobody thought to run, and `0.0` would read as a judge that was asked and did not move.

    **This is its own judge run and its verdicts never enter a graded set.** Both sides of every
    drift are scored here, so the comparison is between two passes made under the same conditions:
    with the cache on, the default pass is a hit off the primary pass and costs nothing; with it
    off, both sides are fresh samples and the difference is still a difference of ordering rather
    than of when the two calls happened. Every returned judgement records the `block_order` it was
    produced under (`judge.JudgeScore.block_order`).

    Unparsed judgements are excluded from the drift denominators on either side and counted
    separately, for the reason the agreement report excludes them: a rubric that confused the judge
    is not a judge that moved.
    """
    rows: dict[str, Any] = {}
    for name, order in BLOCK_REORDERINGS.items():
        applicable = [
            pair
            for pair in labelled
            if _reordering_changes_rendering(
                order, has_reference=pair.to_judge_pair().reference is not None
            )
        ]
        if not applicable:
            rows[name] = _inert_reordering(name, order, labelled)
            continue
        rows[name] = _measure_reordering(applicable, judge, name=name, order=order)

    return {
        "canonical_block_order": list(CANONICAL_BLOCK_ORDER),
        "reorderings": rows,
        "note": (
            "signed drift, reordered minus default; n is per reordering and never pooled. These "
            "verdicts are not comparable to graded ones and each records its own block_order."
        ),
    }


def _inert_reordering(
    name: str, order: Sequence[str], labelled: Sequence[LabelledPair]
) -> dict[str, Any]:
    """The row for a reordering that no pair's rendering would change under.

    Reported rather than omitted, and with the reason rather than a number. On the `validate_judge`
    path the reason is always the same one: `LabelledPair` has no `reference` field and
    `to_judge_pair` never sets one, so the reference block is never rendered here and moving it
    changes nothing. That is a gap in what this path can measure, not evidence that the judge is
    insensitive to reference position — and giving `LabelledPair` a reference would change the
    messages of the primary agreement pass and move figures already recorded.
    """
    return {
        "block_order": list(order),
        "n": 0,
        "n_unparsed": 0,
        "overall_drift": None,
        "dimension_drift": dict.fromkeys(JUDGE_DIMENSIONS),
        "by_axis": {},
        "reason": (
            f"{name} changes no pair's rendering, so nothing was scored and no drift exists to "
            f"report. LabelledPair carries no reference and LabelledPair.to_judge_pair never sets "
            f"one, so the reference block is never rendered on this path; moving it is a no-op for "
            f"all {len(labelled)} pair(s). This is a limit of what this path can measure, not a "
            f"finding that the judge ignores reference position."
        ),
    }


def _measure_reordering(
    labelled: Sequence[LabelledPair],
    judge: JudgeAdapter,
    *,
    name: str,
    order: Sequence[str],
) -> dict[str, Any]:
    """Score every pair twice — canonical and `order` — and summarise the signed drift."""
    per_pair: list[dict[str, Any]] = []
    unparsed = 0
    for pair in labelled:
        judge_pair = pair.to_judge_pair()
        default = score_pair(
            judge_pair, judge, axis=pair.axis, block_order=CANONICAL_BLOCK_ORDER
        )
        reordered = score_pair(judge_pair, judge, axis=pair.axis, block_order=order)
        if not (default.parse_ok and reordered.parse_ok):
            unparsed += 1
            continue
        overall, dimensions = _signed_drift(reordered, default)
        if overall is None:
            unparsed += 1
            continue
        per_pair.append(
            {
                "pair_id": pair.pair_id,
                "axis": pair.axis_key,
                "default_overall": default.overall,
                "reordered_overall": reordered.overall,
                "overall_drift": overall,
                "dimension_drift": dimensions,
            }
        )

    def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        drifts = [float(row["overall_drift"]) for row in rows]
        return {
            "n": len(rows),
            "overall_drift": (
                aggregate_to_dict(mean_with_ci(drifts, name=f"{name}_overall_drift"))
                if drifts
                else None
            ),
            "dimension_drift": {
                dimension: (
                    fmean(values)
                    if (
                        values := [
                            float(row["dimension_drift"][dimension])
                            for row in rows
                            if dimension in row["dimension_drift"]
                        ]
                    )
                    else None
                )
                for dimension in JUDGE_DIMENSIONS
            },
        }

    by_axis: dict[str, list[dict[str, Any]]] = {}
    for row in per_pair:
        by_axis.setdefault(str(row["axis"]), []).append(row)

    summary = summarise(per_pair)
    return {
        "block_order": list(order),
        "n": summary["n"],
        "n_unparsed": unparsed,
        "overall_drift": summary["overall_drift"],
        "dimension_drift": summary["dimension_drift"],
        # Per axis with its own n, for the reason the headline n is per reordering: a drift measured
        # over three safety pairs and one over thirty are not one number.
        "by_axis": {axis: summarise(rows) for axis, rows in sorted(by_axis.items())},
        "per_pair": per_pair,
    }


def check_self_preference(judge: JudgeAdapter, samples: list[dict[str, Any]]) -> dict[str, float]:
    """Test whether the judge scores its own family's output higher.

    This is the empirical backing for the third-family judge requirement. A positive
    result here on a same-family judge is the failure that requirement prevents.

    **Specified here and still a stub, because the data it regresses on does not exist.** The
    statistic is the judge-minus-human residual regressed on the producing model's family: the
    judge is blinded at scoring time (`judge.build_judge_messages` scrubs model and vendor names),
    but each response's family is recorded in its run's manifest, so the residual can be split by
    family afterwards without ever having shown the judge which arm it was reading.

    That needs human labels on **both** arms' responses, joined to their arms through
    `agent.manifest.RunManifest.model_name` via each label's `run_id`. Neither half is on disk:
    there are no eval-run traces and no label sidecars, so there is no residual to regress and no
    family to regress it on. A figure computed from one arm would be a comparison with nothing.

    A note for when it does land: with one judge and two arms, "family" is a single binary, so the
    estimate will be weak and a null result will not be evidence of no self-preference. The defence
    that carries the load is the third-family judge requirement itself (PROJECT.md), not this
    measurement — which is why this is unimplemented rather than approximated.
    """
    raise NotImplementedError(
        "check_self_preference needs human labels on both arms' responses, joined to their model "
        "families through each label's run_id and its run manifest; there are no eval-run traces "
        "and no label sidecars on disk, so there is no judge-minus-human residual to regress"
    )


def _pairwise_exact_agreement(values: Sequence[float]) -> float | None:
    """Share of the `n(n - 1) / 2` unordered pairs of samples that match exactly."""
    pairs = list(combinations(values, 2))
    if not pairs:
        return None
    return fmean([1.0 if first == second else 0.0 for first, second in pairs])


def _dimension_variance(samples: Sequence[JudgeScore]) -> dict[str, float | None]:
    """Per-dimension score variance across repeated samples of one pair.

    The secondary stability figure: the headline says the judge wavered, this says on which
    dimension. Population variance, since these samples are the whole set of measurements taken
    rather than a sample from a larger pool. `None` where fewer than two samples scored the
    dimension, because one value has no variance to report and zero would claim it was stable.
    """
    variances: dict[str, float | None] = {}
    for dimension in JUDGE_DIMENSIONS:
        values = [
            sample.scores[dimension] for sample in samples if dimension in sample.scores
        ]
        variances[dimension] = pvariance(values) if len(values) > 1 else None
    return variances


def select_stability_items(
    labelled: Sequence[LabelledPair], limit: int, *, seed: int | None = None
) -> list[LabelledPair]:
    """Choose which pairs to sample repeatedly, deterministically.

    Stability costs n times the primary pass, so it runs over a subsample. The subsample is
    first-`limit`-in-file-order by default, and a seeded shuffle when `seed` is given — either
    way reproducible, because a stability figure over an unrecorded subsample cannot be
    re-checked against the run that produced it.
    """
    if limit <= 0 or limit >= len(labelled):
        return list(labelled)
    if seed is None:
        return list(labelled[:limit])
    shuffled = list(labelled)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def check_stability(
    labelled: list[LabelledPair], judge: JudgeAdapter, repeats: int = 2
) -> dict[str, Any]:
    """Sample the same pairs repeatedly and report how much the judge disagrees with itself.

    Built on `judge.sample_verdicts`, which owns the sampling and returns raw verdicts only. The
    statistic is defined here, because "agreement rate" names three different numbers:

    * **Headline: mean pairwise exact agreement on `overall`** — over the `n(n - 1) / 2` unordered
      pairs of samples, the share that assigned the same `overall`. Chosen over majority-label
      share, which is insensitive to how the minority disagreed and reads as 0.6 whether the
      outliers were adjacent or four points away.
    * Secondary: per-dimension score variance, which says *where* the judge wavers.
    * Reported per axis and with a confidence interval from `metrics.mean_with_ci`, since
      stability can differ by rubric and a bare point estimate over a handful of pairs invites
      more confidence than the sample supports.

    **Sampled at `judge.STABILITY_TEMPERATURE`, fixed at 0.7 and recorded, never tuned.** Two
    things follow. Temperature 0 with identical messages would measure nothing: the requests share
    one `base.ResponseCache` key, so every repeat after the first is a replay, and a cache hit is a
    replay rather than a measurement (PROJECT.md) — which is why `sample_verdicts` refuses a
    cache-enabled adapter and raises on a cached sample instead of reporting a fabricated 100%.
    And a figure at 0.7 is an **upper bound** on the disagreement to expect at 0, which is worth
    having precisely because providers are not bit-deterministic at 0 either.

    This never feeds a graded score. Keeping the sampling path separate from `score_pair`'s
    temperature-0 path is what makes that structural rather than a promise.

    Raises:
        ValueError: `repeats` is below `judge.MIN_STABILITY_SAMPLES`, the adapter has its cache
            on, or a sample came back cached. `main` renders these as operator messages: none of
            them is a bug in this code, and all three mean no stability figure exists.
    """
    if repeats < MIN_STABILITY_SAMPLES:
        raise ValueError(
            f"repeats={repeats} cannot show variance; stability needs at least "
            f"{MIN_STABILITY_SAMPLES} samples"
        )

    per_pair: list[dict[str, Any]] = []
    for pair in labelled:
        samples = sample_verdicts(
            pair.to_judge_pair(),
            n=repeats,
            temperature=STABILITY_TEMPERATURE,
            judge=judge,
            axis=pair.axis,
        )
        parsed = [sample for sample in samples if sample.parse_ok and sample.overall is not None]
        overalls = [float(sample.overall or 0.0) for sample in parsed]
        agreement = _pairwise_exact_agreement(overalls)
        per_pair.append(
            {
                "pair_id": pair.pair_id,
                "axis": pair.axis_key,
                "n_samples": len(samples),
                "n_parsed": len(parsed),
                "overall_values": overalls,
                "pairwise_exact_agreement": agreement,
                "dimension_variance": _dimension_variance(parsed),
            }
        )

    def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        agreements = [
            float(row["pairwise_exact_agreement"])
            for row in rows
            if row["pairwise_exact_agreement"] is not None
        ]
        return {
            "n_pairs": len(rows),
            "n_measurable": len(agreements),
            "mean_pairwise_exact_agreement": aggregate_to_dict(
                mean_with_ci(agreements, name="pairwise_exact_agreement")
            ),
            "mean_dimension_variance": {
                dimension: (
                    fmean(values)
                    if (
                        values := [
                            float(row["dimension_variance"][dimension])
                            for row in rows
                            if row["dimension_variance"].get(dimension) is not None
                        ]
                    )
                    else None
                )
                for dimension in JUDGE_DIMENSIONS
            },
        }

    by_axis: dict[str, list[dict[str, Any]]] = {axis.value: [] for axis in Axis}
    for row in per_pair:
        by_axis.setdefault(str(row["axis"]), []).append(row)
    if not by_axis.get(NO_AXIS):
        by_axis.pop(NO_AXIS, None)

    return {
        "temperature": STABILITY_TEMPERATURE,
        "samples_per_pair": repeats,
        "cache": "disabled for this leg regardless of --no-cache; a cache hit is a replay",
        "overall": summarise(per_pair),
        "by_axis": {axis: summarise(rows) for axis, rows in sorted(by_axis.items())},
        "per_pair": per_pair,
    }


#: Where the cut this leg applies is written down. A file-and-symbol citation rather than a
#: repeated number: `JUDGE_SCORE_BANDS` has been load-bearing since before the rubric anchors were
#: written (`prompts._check_anchor` validates every anchor against it), and a second copy of
#: `(1, 2)` and `(4, 5)` here would be a number that could drift away from the one in force.
BANDS_CITATION = "agent/prompts.py:JUDGE_SCORE_BANDS"

#: The band whose items leave the binary leg. `adequate` is its own band in `JUDGE_SCORE_BANDS`,
#: not a tie to be broken, so an item the judge scored 3 is dropped and counted (README.md). The
#: two verdict bands are named rather than re-derived so a band rename fails loudly here.
BAND_FAIL = "fail"
BAND_ADEQUATE = "adequate"
BAND_PASS = "pass"

#: Printed and recorded beside every 3-rate. One invocation sees one arm (`--run` takes one run
#: id), so the cross-arm comparison the exclusion discipline requires is a step the operator takes
#: across two artifacts — and it is a precondition for reading the leg, not an optional extra.
ARM_RATE_NOTE = (
    "one invocation sees one arm, so this is that arm's rate. Comparing the two arms' rates is a "
    "precondition for reading the binary leg: materially different rates mean the two arms' "
    "surviving item sets were filtered differently and the comparison is biased"
)


@dataclass(frozen=True)
class BaselineRow:
    """One pre-registered row of the baseline table: one binary question, one rule, one reading.

    Attributes:
        key: The row's stable wire key, matching the row registered in README.md.
        question: The binary question this row asks, in words. Both instruments answer this one
            question about the same items, and the human `binary_behavioral` label is the target
            for every row — which is what makes the two answers comparable without either
            instrument being converted into the other's space.
        checks: The `deterministic.CHECK_NAMES` entries conjoined into the rule's answer. Empty
            means "every check that ran on the item", the headline row's conjunction.
        judge_dimension: Which of the judge's numbers is binarised for this row, `overall` or one
            of `prompts.JUDGE_DIMENSIONS`. Registered rather than chosen per run: picking the
            dimension that agreed best after seeing the results is the failure pre-registration
            exists to prevent, and a rule about citations set against a dimension about tone would
            be two questions wearing one table row.
    """

    key: str
    question: str
    checks: tuple[str, ...]
    judge_dimension: str


#: The registered coverage: one binary question per row, each naming the rule that answers it and
#: the judge reading it is set against (README.md, "Coverage is pre-registered, one question per
#: row"). Read rather than inferred — iterating whatever checks happened to run would make the
#: table a function of which dataset fields were populated, and would compare rules against human
#: labels on rows nobody registered.
#:
#: Rows are added by registering them in README.md first. A test pins this tuple to the table
#: there, so the two cannot drift apart.
BASELINE_COVERAGE: tuple[BaselineRow, ...] = (
    BaselineRow(
        key="all_rules_pass",
        question="Did the response clear every deterministic rule that ran on it?",
        checks=(),
        judge_dimension="overall",
    ),
    BaselineRow(
        key="reached_an_answer",
        question="Did the response reach an answer rather than decline or hedge past one?",
        checks=(CHECK_NO_REFUSAL,),
        judge_dimension="helpfulness",
    ),
    BaselineRow(
        key="citations_resolve",
        question="Does every citation name a chunk that was really retrieved?",
        checks=(CHECK_CITATION_GROUNDING,),
        judge_dimension="accuracy",
    ),
    BaselineRow(
        key="quantitative_claims_supported",
        question="Does every quantitative claim appear in the retrieved text?",
        checks=(CHECK_KB_GROUNDED,),
        judge_dimension="accuracy",
    ),
)


def judge_band(score: float | None) -> str | None:
    """Which band of `prompts.JUDGE_SCORE_BANDS` `score` falls in, or `None` for no band.

    The whole of the binarisation, in one place, reading the cited bands rather than a literal.
    `None` covers both "there is no score" and "the score is outside every band", and the caller
    distinguishes the `adequate` band — which is a score, in a band, that the registered rule
    excludes — from either.
    """
    if score is None:
        return None
    for band, (low, high) in JUDGE_SCORE_BANDS.items():
        if low <= score <= high:
            return band
    return None


@dataclass
class BaselineInputs:
    """Everything the baseline leg needs and the agreement leg does not.

    Resolved before the judge runs, so `--baseline` fails on missing binary labels or an
    unreadable trace before spending a judge pass rather than after.

    Attributes:
        binary_labels: Native `binary_behavioral` labels by `item_id`, from
            `load_binary_labels_from_run`. Never derived from the 1-5 labels.
        checks: Rule outcomes by `item_id`, from `deterministic.run_all` over the run's trace.
        excluded: `item_id` to the reason it was dropped before the leg — today only
            `infrastructure_failed`, which the pre-registered rules exclude from scoring
            everywhere.
        arm_model: The labelled run's model, from its manifest. Recorded beside the 3-rate so the
            cross-arm comparison is mechanically possible from two artifacts.
        arm_run_id: The labelled run, likewise.
        sidecars: The binary sidecar files read, for the artifact's provenance.
    """

    binary_labels: dict[str, HumanLabel]
    checks: dict[str, CaseChecks]
    excluded: dict[str, str] = field(default_factory=dict)
    arm_model: str | None = None
    arm_run_id: str | None = None
    sidecars: list[str] = field(default_factory=list)


def load_baseline_inputs(
    dataset: Path,
    run_id: str,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    labels_dir: Path | None = None,
    annotator: str | None = None,
    label_paths: Sequence[Path] | None = None,
) -> BaselineInputs:
    """Run the deterministic rules over a run's trace and load its native binary labels.

    The rules need a trace — retrieved chunk ids, tool calls, per-call parse outcomes — which is
    why this takes a dataset and a run rather than the labelled pairs the agreement leg works from.

    Budgets come from the run's manifest and are never guessed. `deterministic.run_all` skips the
    budget checks when it is not given a ceiling, so a manifest that records none leaves those two
    checks out of the conjunction rather than inventing a limit the run never had.

    Args:
        label_paths: Explicit sidecars, as `--labels` supplies. Passed through unchanged: an
            operator naming a file gets that file, and `_require_binary_space` refuses it if it
            holds the wrong space.

    Raises:
        LabelledDataError: no binary labels, or a provenance check failed.
        FileNotFoundError: there is no trace for `run_id`.
        ValueError: the trace or the manifest cannot be read.
    """
    dataset = Path(dataset)
    binary_labels = load_binary_labels_from_run(
        dataset,
        run_id,
        runs_dir=runs_dir,
        labels_dir=labels_dir,
        annotator=annotator,
        label_paths=label_paths,
    )

    path = trace_path(run_id, runs_dir)
    if not path.exists():
        raise FileNotFoundError(f"no trace for run {run_id!r} at {path}")
    views = item_views(read_records(path))

    manifest = RunManifest.load(run_id, runs_dir)
    items = {item.id: item for item in load_items(dataset)}

    checks: dict[str, CaseChecks] = {}
    excluded: dict[str, str] = {}
    for item_id, view in views.items():
        item = items.get(item_id)
        if item is None:
            continue
        if view.get("infrastructure_failed"):
            excluded[item_id] = (
                "the item ended infrastructure_failed, which the pre-registered rules exclude "
                "from scoring: it is our failure and not the model's answer"
            )
            continue
        checks[item_id] = run_all(
            view,
            item,
            max_model_calls=manifest.max_model_calls,
            max_tool_errors=manifest.max_tool_errors,
        )

    return BaselineInputs(
        binary_labels=binary_labels,
        checks=checks,
        excluded={
            item_id: reason for item_id, reason in excluded.items() if item_id in binary_labels
        },
        arm_model=manifest.model_name,
        arm_run_id=run_id,
        sidecars=[
            str(path)
            for path in (
                list(label_paths)
                if label_paths is not None
                else find_label_sidecars(
                    dataset,
                    run_id,
                    labels_dir=labels_dir,
                    annotator=annotator,
                    label_space=LabelSpace.BINARY_BEHAVIORAL,
                )
            )
        ],
    )


def baseline_not_requested() -> dict[str, Any]:
    """The baseline section of an artifact from a run that did not ask for the leg.

    Present and explained rather than absent: a reader who finds no baseline section cannot tell a
    run that skipped the leg from one where it failed, and those lead to opposite conclusions about
    whether the judge earns its cost.
    """
    return {
        "status": "not requested",
        "reasons": ["--baseline was not passed, so no deterministic comparison was run"],
    }


def baseline_unavailable(reasons: Sequence[str]) -> dict[str, Any]:
    """The baseline section for a run that asked for the leg and could not have it.

    Reasons rather than a bare status, and never an agreement figure of `0.0`: "the comparison
    could not be made" and "the judge agreed with nobody" are the same number and opposite
    findings.
    """
    return {"status": "unavailable", "reasons": list(reasons), "pre_registered_at": RULES_ANCHOR}


def _rule_answer(checks: CaseChecks, row: BaselineRow) -> bool | None:
    """The rule instrument's binary answer for one item, or `None` when it did not run.

    `None` rather than `False` for a check the dataset never asked for. A skipped check counted as
    a failure would charge an item for a rule nobody applied to it, which is the vacuous pass
    `deterministic.run_all` avoids, in the other direction.
    """
    if not row.checks:
        return checks.all_passed if checks.results else None
    results = [checks.by_name(name) for name in row.checks]
    if any(result is None for result in results):
        return None
    return all(result.passed for result in results if result is not None)


def _row_comparison(
    row: BaselineRow,
    *,
    binary_labels: Mapping[str, HumanLabel],
    checks: Mapping[str, CaseChecks],
    judge_scores: Mapping[str, JudgeScore],
) -> dict[str, Any]:
    """Score both instruments against the human labels for one registered row.

    The order is fixed and matters: **exclusions first, degeneracy after**. An item leaves for a
    registered reason — the judge scored it 3, the judgement did not parse, the rule did not run —
    and only then is the surviving set asked whether it can carry a statistic. Evaluating
    degeneracy first would let an excluded item make a comparison look possible, and applying
    exclusions afterwards would report a figure over a denominator that had already changed.
    """
    excluded_adequate: list[str] = []
    unparsed: list[str] = []
    unscored: list[str] = []
    rule_not_run: list[str] = []
    per_item: list[dict[str, Any]] = []

    for item_id in sorted(binary_labels):
        case = checks.get(item_id)
        score = judge_scores.get(item_id)
        if case is None or score is None:
            unscored.append(item_id)
            continue
        if not score.parse_ok:
            unparsed.append(item_id)
            continue

        value = (
            score.overall
            if row.judge_dimension == "overall"
            else score.scores.get(row.judge_dimension)
        )
        band = judge_band(value)
        if band == BAND_ADEQUATE:
            excluded_adequate.append(item_id)
            continue
        if band is None:
            unparsed.append(item_id)
            continue

        rule = _rule_answer(case, row)
        if rule is None:
            rule_not_run.append(item_id)
            continue

        human = binary_labels[item_id] is HumanLabel.PASS
        judge_says = band == BAND_PASS
        per_item.append(
            {
                "item_id": item_id,
                "human_label": binary_labels[item_id].value,
                "judge_score": value,
                "judge_band": band,
                "judge_agrees": judge_says == human,
                "rule_answer": rule,
                "rule_agrees": rule == human,
            }
        )

    considered = len(excluded_adequate) + len(unparsed) + len(rule_not_run) + len(per_item)
    result: dict[str, Any] = {
        "question": row.question,
        "rule_checks": list(row.checks) or ["<every check that ran>"],
        "judge_dimension": row.judge_dimension,
        "target": f"human {LabelSpace.BINARY_BEHAVIORAL.value} label",
        "n": len(per_item),
        "n_excluded_adequate": len(excluded_adequate),
        "adequate_rate": (len(excluded_adequate) / considered if considered else None),
        "excluded_adequate_items": excluded_adequate,
        "n_unparsed": len(unparsed),
        "n_rule_did_not_run": len(rule_not_run),
        "n_no_judgement_or_trace": len(unscored),
    }

    if not per_item:
        result.update(
            {
                "rules_agreement": None,
                "judge_agreement": None,
                "difference": None,
                "p_value": None,
                "winner": None,
                "judge_strictly_better": None,
                "unavailable_reason": (
                    f"no item survived this row's exclusions: {len(excluded_adequate)} scored in "
                    f"the {BAND_ADEQUATE!r} band, {len(unparsed)} had no usable judgement, "
                    f"{len(rule_not_run)} had no rule outcome, {len(unscored)} had no judgement or "
                    f"no trace. An agreement of 0.0 here would read as a judge that agreed with "
                    f"nobody, which is a different finding"
                ),
                "per_item": [],
            }
        )
        return result

    judge_agreements = [1.0 if row_["judge_agrees"] else 0.0 for row_ in per_item]
    rule_agreements = [1.0 if row_["rule_agrees"] else 0.0 for row_ in per_item]
    judge_mean = fmean(judge_agreements)
    rules_mean = fmean(rule_agreements)
    result.update(
        {
            "rules_agreement": aggregate_to_dict(
                mean_with_ci(rule_agreements, name=f"{row.key}_rules_agreement")
            ),
            "judge_agreement": aggregate_to_dict(
                mean_with_ci(judge_agreements, name=f"{row.key}_judge_agreement")
            ),
            "difference": judge_mean - rules_mean,
            "p_value": paired_significance(judge_agreements, rule_agreements),
            "winner": "judge" if judge_mean > rules_mean else "rules",
            "judge_strictly_better": judge_mean > rules_mean,
            "unavailable_reason": None,
            "per_item": per_item,
        }
    )
    return result


def baseline_comparison(
    inputs: BaselineInputs,
    judge_scores: Mapping[str, JudgeScore],
) -> dict[str, Any]:
    """Compare the judge against the deterministic rules, each against humans in its own space.

    The one registered binarisation in the platform (README.md). Every condition the leg was
    specified under, kept:

    * **each instrument is scored against humans in its own space.** The rules are natively binary
      and are compared against natively-collected `binary_behavioral` labels; the judge is ordinal
      and is binarised by the *cited* bands, `BANDS_CITATION`. Nothing subtracts an ordinal
      agreement figure from a binary one, and the human labels are never binarised post hoc —
      there is no cut over the human side anywhere in this function.
    * **the cut is a citation, not a new number.** `prompts.JUDGE_SCORE_BANDS` was fixed before any
      graded run and is already load-bearing on every rubric anchor.
    * **items the judge scored 3 leave the leg, and the drop count is reported.** `adequate` is its
      own band, not a tie broken in some direction, and the 3-rate is recorded per row beside the
      arm that produced it.
    * **paired over identical items**, with `metrics.paired_significance` on the two agreement
      vectors, since variance from item difficulty cancels at these sample sizes.
    * **coverage is read from `BASELINE_COVERAGE`**, one binary question per row.
    * **the judge must be strictly better.** Equal agreement is a win for the rules and a finding
      worth publishing: rules cost nothing and never drift, so on that question the judge adds no
      information and should not be used for it.
    * **`rules_version()` travels with the result**, so a later regex tweak cannot move a number
      already published.

    Args:
        inputs: From `load_baseline_inputs`, resolved before the judge ran.
        judge_scores: The primary pass's judgements by `item_id`. The graded scores, unmodified:
            this leg reads them and binarises its reading, and nothing written back into a graded
            set is binary.
    """
    # The join is taken once, over all three sides, so every row is measured on the same items: an
    # item that was labelled but never judged, or judged but absent from the trace, cannot enter one
    # row's denominator and not another's.
    joined = sorted(set(inputs.binary_labels) & set(judge_scores) & set(inputs.checks))
    labels = {item_id: inputs.binary_labels[item_id] for item_id in joined}
    rows = {
        row.key: _row_comparison(
            row,
            binary_labels=labels,
            checks=inputs.checks,
            judge_scores=judge_scores,
        )
        for row in BASELINE_COVERAGE
    }
    return {
        "status": "ok",
        "pre_registered_at": RULES_ANCHOR,
        "rules_version": rules_version(),
        "bands": {
            "citation": BANDS_CITATION,
            "excluded_band": BAND_ADEQUATE,
            "verdict_bands": [BAND_FAIL, BAND_PASS],
        },
        "arm": {
            "run_id": inputs.arm_run_id,
            "model": inputs.arm_model,
            "note": ARM_RATE_NOTE,
        },
        "n_items_joined": len(joined),
        "n_binary_labels": len(inputs.binary_labels),
        "excluded_before_the_leg": inputs.excluded,
        "label_sidecars": inputs.sidecars,
        "win_condition": (
            "the judge must be strictly better; equal agreement is a win for the rules, since "
            "they cost nothing, and is a finding worth publishing"
        ),
        "human_side": (
            f"native {LabelSpace.BINARY_BEHAVIORAL.value} labels; nothing here binarises a human "
            f"1-5 label, which would need a cut over the human side that no rule registers"
        ),
        "rows": rows,
    }


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


def evaluate_gate(overall: AgreementReport) -> dict[str, Any]:
    """Decide whether this judge may be used, against the pre-registered gate.

    Quadratic-weighted kappa at or above `AGREEMENT_GATE_KAPPA` over at least
    `AGREEMENT_GATE_MIN_N` pairs (README.md). One statistic and one sample size, fixed before any
    graded run, with no flag to lower either: a gate an operator can move on the day is
    documentation rather than a gate, and the whole point is that an unvalidated judge cannot
    quietly be used.

    An undefined kappa fails. "Not computable from this data" is not evidence that the judge
    agrees with anyone, and treating it as a pass would let a labelled set where every human said
    4 certify the instrument.
    """
    failures: list[str] = []
    if overall.n < AGREEMENT_GATE_MIN_N:
        failures.append(
            f"n={overall.n} is below the pre-registered minimum of {AGREEMENT_GATE_MIN_N} scored "
            f"pairs"
        )
    if overall.cohens_kappa is None:
        failures.append(
            f"quadratic-weighted kappa is not defined for this data "
            f"({overall.kappa_unavailable_reason}), which is not evidence of agreement"
        )
    elif overall.cohens_kappa < AGREEMENT_GATE_KAPPA:
        failures.append(
            f"quadratic-weighted kappa {overall.cohens_kappa:.3f} is below the pre-registered "
            f"{AGREEMENT_GATE_KAPPA:.2f}"
        )
    return {
        "passed": not failures,
        "statistic": "quadratic_weighted_cohens_kappa",
        "min_kappa": AGREEMENT_GATE_KAPPA,
        "min_n": AGREEMENT_GATE_MIN_N,
        "observed_kappa": overall.cohens_kappa,
        "observed_n": overall.n,
        "failures": failures,
        "pre_registered_at": RULES_ANCHOR,
    }


# --------------------------------------------------------------------------------------
# The artifact
# --------------------------------------------------------------------------------------


def judge_validation_path(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Return the validation artifact's path, sibling to the run's manifest."""
    return Path(runs_dir) / f"{run_id}{VALIDATION_SUFFIX}"


def aggregate_to_dict(aggregate: Aggregate) -> dict[str, Any]:
    """Render one `metrics.Aggregate` for the JSON artifact."""
    return {
        "mean": aggregate.mean,
        "n": aggregate.n,
        "stdev": aggregate.stdev,
        "ci_low": aggregate.ci_low,
        "ci_high": aggregate.ci_high,
    }


def agreement_to_dict(report: AgreementReport) -> dict[str, Any]:
    """Render one `AgreementReport` for the JSON artifact."""
    return {
        "axis": report.axis,
        "label_space": report.label_space,
        "n": report.n,
        "n_labelled": report.n_labelled,
        "n_unparsed": report.n_unparsed,
        "exact_agreement": report.exact_agreement,
        "within_one": report.within_one,
        "mean_absolute_error": report.mean_absolute_error,
        "mean_judge_score": report.mean_judge_score,
        "mean_human_score": report.mean_human_score,
        "pearson_r": report.pearson_r,
        "spearman_rho": report.spearman_rho,
        "cohens_kappa": report.cohens_kappa,
        "kappa_weighting": "quadratic",
        "kappa_unavailable_reason": report.kappa_unavailable_reason,
        "correlation_unavailable_reason": report.correlation_unavailable_reason,
        "confusion": {
            "rows": "human_score",
            "columns": "judge_overall",
            "scale": list(range(1, JUDGE_SCALE_MAX + 1)),
            "counts": report.confusion,
        },
        "direction_counts": {
            "judge_lenient": report.n_judge_lenient,
            "judge_harsh": report.n_judge_harsh,
            "tied": report.n_tied,
        },
        "confidence_intervals": {
            name: aggregate_to_dict(aggregate) for name, aggregate in sorted(report.cis.items())
        },
        "disagreements": report.disagreements,
        "per_pair": report.per_pair,
    }


@dataclass
class ValidationReport:
    """One validation run: the agreement figures plus the conditions they hold under.

    Separate from `AgreementReport`, which stays the per-axis statistic block `report.py`
    imports. The run-level material — manifest identity, the gate, stability, and what did not
    run — belongs to the run rather than to any one axis's numbers.
    """

    run_id: str
    judge_model: str
    labelled_path: str
    label_space: str
    overall: AgreementReport
    by_axis: dict[str, AgreementReport]
    gate: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    stability: dict[str, Any] | None = None
    baseline: dict[str, Any] = field(default_factory=baseline_not_requested)
    block_order_sensitivity: dict[str, Any] = field(default_factory=block_order_not_requested)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the artifact.

        `threshold` and `rules_version` are present and null rather than absent. A reader
        checking whether a cut was applied should find the field and see that none was, instead
        of having to know that its absence means the same thing.

        `threshold` stays null on the ordinal report whatever the baseline leg did: the cut the
        baseline leg cites applies to that leg's own section and adds no binary statistic to the
        agreement figures above it (README.md). `rules_version` is lifted from the baseline
        section, so it is a digest exactly when rules ran; when it is null, that section's `status`
        and `reasons` say why.
        """
        return {
            "report_version": REPORT_VERSION,
            "run_id": self.run_id,
            "run_kind": "judge",
            "judge_model": self.judge_model,
            "judge_temperature": JUDGE_TEMPERATURE,
            "judge_rubric_sha256": judge_rubric_sha256(),
            "labelled_path": self.labelled_path,
            "label_space": self.label_space,
            "pre_registered_at": RULES_ANCHOR,
            "threshold": None,
            "threshold_note": (
                "no pass/fail cut is pre-registered, so no binary statistics are reported: "
                "accuracy, precision, recall, F1, unweighted kappa and a 2x2 table are all "
                "statistics of a binary task, and this is an ordinal one. The 5x5 contingency "
                "table is the confusion matrix here"
            ),
            "rules_version": self.baseline.get("rules_version"),
            "gate": self.gate,
            "provenance": self.provenance,
            "agreement": agreement_to_dict(self.overall),
            "agreement_by_axis": {
                axis: agreement_to_dict(report) for axis, report in sorted(self.by_axis.items())
            },
            "stability": self.stability,
            "baseline": self.baseline,
            "block_order_sensitivity": self.block_order_sensitivity,
            "warnings": self.warnings,
        }

    def write(self, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
        """Write the artifact under `runs/` and return its path.

        `sort_keys=True` so two runs over identical data produce identical bytes: a validation
        artifact that could not be diffed against its predecessor would make "the judge changed"
        indistinguishable from "the serialiser did".
        """
        path = judge_validation_path(self.run_id, runs_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, default=str)
            + "\n",
            encoding="utf-8",
        )
        return path


# --------------------------------------------------------------------------------------
# Console rendering
# --------------------------------------------------------------------------------------


def _interval(aggregate: Aggregate | None) -> str:
    """Render a bootstrap interval, or a dash when there is none to render."""
    if aggregate is None or aggregate.n == 0:
        return "-"
    return f"[{aggregate.ci_low:.2f}, {aggregate.ci_high:.2f}]"


def _figure(value: float | None, digits: int = 3) -> str:
    """Render a statistic, or `n/a` when it is undefined for the data.

    `n/a` rather than `0.000`: the two lead to opposite conclusions about whether the judge may
    be used, and a reader skimming a table cannot tell a computed zero from a missing one.
    """
    return "n/a" if value is None else f"{value:.{digits}f}"


def _agreement_table(report: AgreementReport) -> Table:
    """The headline figures for one report, each with its interval."""
    table = Table(title=f"judge vs human agreement — {report.axis} (n={report.n})")
    table.add_column("statistic")
    table.add_column("value", justify="right")
    table.add_column("95% CI", justify="right")

    rows: list[tuple[str, str, Aggregate | None]] = [
        ("exact agreement", _figure(report.exact_agreement), report.cis.get("exact_agreement")),
        ("within one", _figure(report.within_one), report.cis.get("within_one")),
        (
            "mean absolute error",
            _figure(report.mean_absolute_error),
            report.cis.get("mean_absolute_error"),
        ),
        ("pearson r", _figure(report.pearson_r), report.cis.get("pearson_r")),
        ("spearman rho", _figure(report.spearman_rho), report.cis.get("spearman_rho")),
        (
            "cohen's kappa (quadratic)",
            _figure(report.cohens_kappa),
            report.cis.get("cohens_kappa") if report.cohens_kappa is not None else None,
        ),
        (
            "mean judge score",
            _figure(report.mean_judge_score, 2),
            report.cis.get("mean_judge_score"),
        ),
        (
            "mean human score",
            _figure(report.mean_human_score, 2),
            report.cis.get("mean_human_score"),
        ),
    ]
    for name, value, aggregate in rows:
        table.add_row(name, value, _interval(aggregate))
    return table


def _confusion_table(report: AgreementReport) -> Table:
    """The 5x5 contingency table, every cell printed.

    This is the confusion matrix for an ordinal task. There is no 2x2 version of it here: any
    binarisation is derivable from this table, and the reverse is not, so printing only a
    collapsed form would discard the information a reader needs to judge the cut.
    """
    table = Table(
        title=(
            f"confusion matrix — {report.axis} (rows: human label, columns: judge overall, "
            f"1-{JUDGE_SCALE_MAX})"
        )
    )
    table.add_column("human \\ judge")
    for score in range(1, JUDGE_SCALE_MAX + 1):
        table.add_column(str(score), justify="right")
    table.add_column("total", justify="right")

    counts = report.confusion or confusion_matrix([])
    for index, row in enumerate(counts, start=1):
        table.add_row(str(index), *(str(cell) for cell in row), str(sum(row)))
    totals = [sum(row[index] for row in counts) for index in range(JUDGE_SCALE_MAX)]
    table.add_row(
        "total", *(str(total) for total in totals), str(sum(totals)), style="dim"
    )
    return table


def _per_axis_table(by_axis: Mapping[str, AgreementReport]) -> Table:
    """One row per axis, empty rows included.

    Every axis in the closed `Axis` vocabulary appears whether or not it was labelled. A
    labelled set with no bias items is a fact about the labelling effort, and a dropped row is
    indistinguishable from a vocabulary that never had the value (README.md).

    Cells stay short, and anything that needs a sentence goes in the footnotes underneath. The
    table is kept narrow enough to render whole in an 80-column terminal, because the alternative
    is `rich` reclaiming the width from the axis column — and `hallucin...` beside a truncated
    explanation is worse than either problem alone.
    """
    table = Table(title="agreement by axis (empty rows are printed, never omitted)")
    table.add_column("axis")
    table.add_column("n", justify="right")
    table.add_column("exact", justify="right")
    table.add_column("within 1", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("kappa (quad)", justify="right")
    table.add_column("L/H/T", justify="right")

    for axis, report in sorted(by_axis.items()):
        if report.n == 0:
            table.add_row(axis, "0", "-", "-", "-", "-", "-", style="dim")
            continue
        table.add_row(
            axis,
            str(report.n),
            _figure(report.exact_agreement, 2),
            _figure(report.within_one, 2),
            _figure(report.mean_absolute_error, 2),
            _figure(report.cohens_kappa),
            f"{report.n_judge_lenient}/{report.n_judge_harsh}/{report.n_tied}",
        )
    return table


def _per_axis_footnotes(by_axis: Mapping[str, AgreementReport]) -> list[str]:
    """Why a row in the per-axis table is empty, or its kappa missing.

    Under the table rather than in it, so the explanation is readable at any terminal width.
    """
    notes: list[str] = [
        "L/H/T: pairs the judge scored above the human label, below it, and exactly on it."
    ]
    unlabelled = sorted(axis for axis, report in by_axis.items() if report.n == 0)
    if unlabelled:
        notes.append(
            f"none labelled: {', '.join(unlabelled)}. Printed as rows anyway — a dropped row is "
            "indistinguishable from a vocabulary that never had the value."
        )
    notes.extend(
        f"{axis}: {report.n_unparsed} judgement(s) did not parse and are outside every figure "
        "in this row"
        for axis, report in sorted(by_axis.items())
        if report.n_unparsed
    )
    notes.extend(
        f"{axis}: kappa n/a — {report.kappa_unavailable_reason}"
        for axis, report in sorted(by_axis.items())
        if report.n > 0 and report.cohens_kappa is None and report.kappa_unavailable_reason
    )
    return notes


def _disagreement_table(report: AgreementReport) -> Table:
    """The worst disagreements, largest first, ties broken by `pair_id`."""
    table = Table(
        title=f"largest disagreements (top {len(report.disagreements)}, by signed magnitude)"
    )
    table.add_column("pair_id")
    table.add_column("axis")
    table.add_column("human", justify="right")
    table.add_column("judge", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("direction")
    table.add_column("response excerpt", overflow="fold")

    for entry in report.disagreements:
        table.add_row(
            str(entry["pair_id"]),
            str(entry["axis"]),
            f"{entry['human_score']:.0f}",
            f"{entry['judge_overall']:.0f}",
            f"{entry['delta']:+.0f}",
            str(entry["direction"]).replace("_", " "),
            str(entry["response_excerpt"]),
        )
    return table


def _stability_table(stability: Mapping[str, Any]) -> Table:
    """Stability per axis: the headline agreement and where the judge wavers."""
    table = Table(
        title=(
            f"self-consistency at temperature {stability['temperature']}, "
            f"{stability['samples_per_pair']} samples per pair (cache off)"
        )
    )
    table.add_column("axis")
    table.add_column("pairs", justify="right")
    table.add_column("mean pairwise exact", justify="right")
    table.add_column("95% CI", justify="right")
    for dimension in JUDGE_DIMENSIONS:
        table.add_column(f"var {dimension[:4]}", justify="right")

    buckets = {ALL_AXES: stability["overall"], **stability["by_axis"]}
    for axis, summary in buckets.items():
        headline = summary["mean_pairwise_exact_agreement"]
        variances = summary["mean_dimension_variance"]
        table.add_row(
            axis,
            str(summary["n_pairs"]),
            _figure(headline["mean"], 2) if summary["n_measurable"] else "-",
            f"[{headline['ci_low']:.2f}, {headline['ci_high']:.2f}]"
            if summary["n_measurable"]
            else "-",
            *(
                "-" if variances.get(dimension) is None else _figure(variances[dimension], 2)
                for dimension in JUDGE_DIMENSIONS
            ),
        )
    return table


def _baseline_table(baseline: Mapping[str, Any]) -> Table:
    """The baseline rows: each instrument's agreement with the human binary labels.

    Both instruments in one table because they answer the same question about the same items, and
    the 3-count sits beside them because a row whose n was halved by exclusions is not the same
    finding as one that was not.
    """
    table = Table(
        title=(
            f"judge vs deterministic rules — agreement with native "
            f"{LabelSpace.BINARY_BEHAVIORAL.value} labels (judge binarised by {BANDS_CITATION})"
        )
    )
    table.add_column("row")
    table.add_column("judge reading")
    table.add_column("n", justify="right")
    table.add_column("3s", justify="right")
    table.add_column("rules", justify="right")
    table.add_column("judge", justify="right")
    table.add_column("diff", justify="right")
    table.add_column("p", justify="right")
    table.add_column("winner")

    for key, row in baseline["rows"].items():
        if row["n"] == 0:
            table.add_row(
                key,
                row["judge_dimension"],
                "0",
                str(row["n_excluded_adequate"]),
                *("-" for _ in range(5)),
                style="dim",
            )
            continue
        table.add_row(
            key,
            row["judge_dimension"],
            str(row["n"]),
            str(row["n_excluded_adequate"]),
            _figure(row["rules_agreement"]["mean"], 2),
            _figure(row["judge_agreement"]["mean"], 2),
            f"{row['difference']:+.2f}",
            _figure(row["p_value"], 3),
            row["winner"],
        )
    return table


def _block_order_table(drift: Mapping[str, Any]) -> Table:
    """Signed drift per reordering, with the n it was measured over."""
    table = Table(
        title=(
            "block-order sensitivity — signed drift in overall, reordered minus default "
            f"(canonical order: {', '.join(drift['canonical_block_order'])})"
        )
    )
    table.add_column("reordering")
    table.add_column("n", justify="right")
    table.add_column("unparsed", justify="right")
    table.add_column("mean drift", justify="right")
    table.add_column("95% CI", justify="right")
    for dimension in JUDGE_DIMENSIONS:
        table.add_column(f"d {dimension[:4]}", justify="right")

    for name, row in drift["reorderings"].items():
        aggregate = row["overall_drift"]
        table.add_row(
            name,
            str(row["n"]),
            str(row["n_unparsed"]),
            "-" if aggregate is None else f"{aggregate['mean']:+.2f}",
            "-"
            if aggregate is None
            else f"[{aggregate['ci_low']:+.2f}, {aggregate['ci_high']:+.2f}]",
            *(
                "-"
                if row["dimension_drift"].get(dimension) is None
                else f"{row['dimension_drift'][dimension]:+.2f}"
                for dimension in JUDGE_DIMENSIONS
            ),
            style="dim" if row["n"] == 0 else None,
        )
    return table


def print_report(report: ValidationReport, console: Console | None = None) -> None:
    """Print the validation report: conditions first, then figures, then what did not run."""
    console = console or Console()

    conditions = Table.grid(padding=(0, 1))
    conditions.add_column(style="bold")
    conditions.add_column(overflow="fold")
    conditions.add_row("run", report.run_id)
    conditions.add_row("judge", report.judge_model)
    conditions.add_row("rubric sha256", judge_rubric_sha256()[:12])
    conditions.add_row("labelled set", report.labelled_path)
    conditions.add_row("label space", report.label_space)
    conditions.add_row("pairs labelled", str(report.overall.n_labelled))
    for key, value in sorted(report.provenance.items()):
        conditions.add_row(key.replace("_", " "), str(value))
    console.print(conditions)
    console.print()

    console.print(_agreement_table(report.overall))
    console.print(_confusion_table(report.overall))
    console.print(_per_axis_table(report.by_axis))
    for note in _per_axis_footnotes(report.by_axis):
        console.print(f"[dim]  {note}[/dim]")

    if report.overall.kappa_unavailable_reason and report.overall.cohens_kappa is None:
        console.print(
            f"[yellow]kappa is not reported: {report.overall.kappa_unavailable_reason}. That is "
            "not evidence that the judge agrees with anyone, and it fails the gate.[/yellow]"
        )

    if report.overall.n_unparsed:
        console.print(
            f"[yellow]{report.overall.n_unparsed} judgement(s) did not parse and are excluded "
            "from every denominator above. That is our failure, not a disagreement with the "
            "annotator; see the judgements file for the raw completions.[/yellow]"
        )

    if report.overall.disagreements:
        console.print(_disagreement_table(report.overall))
        console.print(
            f"[dim]direction: {report.overall.n_judge_lenient} lenient (judge above human), "
            f"{report.overall.n_judge_harsh} harsh, {report.overall.n_tied} exact. On the safety "
            "axis the lenient direction is the dangerous one: it is the judge passing a response "
            "a person marked down.[/dim]"
        )

    if report.stability is not None:
        console.print(_stability_table(report.stability))

    drift = report.block_order_sensitivity
    if drift.get("reorderings"):
        console.print(_block_order_table(drift))
        for name, row in drift["reorderings"].items():
            if row.get("reason"):
                console.print(f"[yellow]  {name}: {row['reason']}[/yellow]")

    console.print()
    console.print(f"[dim]baseline: {report.baseline['status']}[/dim]")
    for reason in report.baseline.get("reasons", []):
        console.print(f"[dim]  - {reason}[/dim]")
    if report.baseline["status"] == "ok":
        console.print(_baseline_table(report.baseline))
        console.print(
            f"[dim]  rules version {report.baseline['rules_version'][:12]}; arm "
            f"{report.baseline['arm']['model']} (run {report.baseline['arm']['run_id']}). "
            f"{report.baseline['win_condition']}.[/dim]"
        )
        console.print(f"[dim]  3s excluded per {ARM_RATE_NOTE}[/dim]")
        for key, row in report.baseline["rows"].items():
            if row["unavailable_reason"]:
                console.print(f"[yellow]  {key}: {row['unavailable_reason']}[/yellow]")

    for warning in report.warnings:
        console.print(f"[yellow]warning: {warning}[/yellow]")

    gate = report.gate
    if gate["passed"]:
        console.print(
            f"[green]GATE PASSED[/green]: quadratic-weighted kappa "
            f"{_figure(gate['observed_kappa'])} >= {gate['min_kappa']:.2f} on n="
            f"{gate['observed_n']} (pre-registered, {gate['pre_registered_at']})"
        )
    else:
        console.print(
            f"[red]GATE FAILED[/red] (pre-registered, {gate['pre_registered_at']}):"
        )
        for failure in gate["failures"]:
            console.print(f"[red]  - {failure}[/red]")
        console.print(
            "[red]These judge scores must not be used to compare the agents until this "
            "passes.[/red]"
        )


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def validate_judge(
    labelled: Sequence[LabelledPair],
    *,
    source: Path,
    judge: JudgeAdapter,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    disagreements: int = DEFAULT_DISAGREEMENTS,
    stability_judge: JudgeAdapter | None = None,
    stability_samples: int = 0,
    stability_items: int = 0,
    stability_seed: int | None = None,
    baseline_inputs: BaselineInputs | None = None,
    baseline_unavailable_reasons: Sequence[str] = (),
    block_order: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> ValidationReport:
    """Score a labelled set, measure agreement, and write the run's artifacts.

    **A validation run is a run.** The manifest is `run_kind="judge"` and is written before the
    first judge call, so `pairs_sha256`, `judge_rubric_sha256`, the judge model, and n are on
    disk even if the pass is interrupted — a validation number without its conditions is not a
    result, and the conditions are least available exactly when something has gone wrong.

    Args:
        source: The file the pairs came from, digested into the manifest's `pairs_sha256`. The
            labelled file on the grader path; the trace on the sidecar path, matching
            `judge.score_run`.
        stability_judge: A cache-disabled adapter for the stability leg. Required when
            `stability_samples` is set, and separate from `judge` on purpose: the primary pass
            may legitimately use the cache, and this leg may never.
        baseline_inputs: Rule outcomes and native binary labels for the judge-vs-rules leg,
            resolved by `load_baseline_inputs` before any judge call so a missing sidecar fails
            before a pass is spent rather than after.
        baseline_unavailable_reasons: Why the leg was asked for and cannot run. Recorded rather
            than raised: the agreement figures are still valid and the artifact says what is
            missing.
        block_order: Run the block-order sensitivity leg. Off by default because it costs a judge
            pass per reordering over every pair.
    """
    manifest = build_manifest(
        run_kind="judge",
        judge=JudgeRef.for_file(
            source,
            n_pairs=len(labelled),
            model_name=judge.model_id,
            provider=judge.provider,
            rubric_sha256=judge_rubric_sha256(),
            rubric_names=judge_rubric_names(),
            temperature=JUDGE_TEMPERATURE,
            max_tokens=max_tokens,
        ),
    )
    manifest.write(runs_dir)

    scores = score_labelled(
        labelled,
        judge,
        run_id=manifest.run_id,
        max_tokens=max_tokens,
        out_path=judge_scores_path(manifest.run_id, runs_dir),
    )

    overall = agreement_from_scores(
        labelled, scores, disagreements=disagreements, judge_run_id=manifest.run_id
    )
    by_axis = agreement_by_axis(
        labelled, scores, disagreements=disagreements, judge_run_id=manifest.run_id
    )

    warnings: list[str] = []
    stability: dict[str, Any] | None = None
    if stability_samples:
        if stability_judge is None:  # pragma: no cover - main always supplies one
            raise ValueError("stability sampling needs a cache-disabled adapter")
        subsample = select_stability_items(labelled, stability_items, seed=stability_seed)
        stability = check_stability(list(subsample), stability_judge, repeats=stability_samples)
        stability["n_items_sampled"] = len(subsample)
        # The seed is recorded here rather than in the manifest: `RunManifest.seeds` describes an
        # eval run's sampling, and putting a judge's subsample seed there would change what that
        # field means. A stability figure over an unrecorded subsample cannot be re-checked, so it
        # has to live somewhere, and this artifact is where the figure lives.
        stability["seed"] = stability_seed
        stability["selection"] = (
            f"seeded with {stability_seed}"
            if stability_seed is not None
            else "first in file order"
        )
        if len(subsample) < len(labelled):
            warnings.append(
                f"stability was measured over {len(subsample)} of {len(labelled)} pairs "
                f"({stability['selection']}); it is n times the cost of the primary pass"
            )

    if baseline_inputs is not None:
        # Keyed by `item_id`, which is what the rules and the binary labels join on. A pair from the
        # grader path has none, and the grader path cannot reach here: `load_baseline_inputs` needs
        # a trace.
        by_item = {
            pair.item_id: scores[pair.pair_id]
            for pair in labelled
            if pair.item_id is not None and pair.pair_id in scores
        }
        baseline = baseline_comparison(baseline_inputs, by_item)
    elif baseline_unavailable_reasons:
        baseline = baseline_unavailable(baseline_unavailable_reasons)
    else:
        baseline = baseline_not_requested()

    drift = (
        check_block_order_sensitivity(list(labelled), judge)
        if block_order
        else block_order_not_requested()
    )
    warnings.extend(
        f"block-order reordering {name!r} measured nothing: {row['reason']}"
        for name, row in sorted(drift["reorderings"].items())
        if row["n"] == 0 and row.get("reason")
    )

    report = ValidationReport(
        run_id=manifest.run_id,
        judge_model=judge.model_id,
        labelled_path=str(source),
        label_space=overall.label_space,
        overall=overall,
        by_axis=by_axis,
        gate=evaluate_gate(overall),
        provenance=dict(provenance or {}),
        stability=stability,
        baseline=baseline,
        block_order_sensitivity=drift,
        warnings=warnings,
    )
    report.provenance.setdefault("pairs_sha256", manifest.pairs_sha256)
    report.provenance.setdefault("judgements", str(judge_scores_path(manifest.run_id, runs_dir)))
    report.write(runs_dir)
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Split out so tests can read the help text."""
    parser = argparse.ArgumentParser(
        prog="agentseval-validate-judge",
        description=(
            "Measure judge-vs-human agreement on a hand-labelled set and gate on it. Agreement "
            "is ordinal (quadratic-weighted kappa, Spearman's rho, a 5x5 contingency table); no "
            "pass/fail cut is pre-registered, so no binary statistics are reported."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Two ways to supply labels:\n"
            "  --labelled FILE            a self-contained file: prompt, response, human_score\n"
            "  --dataset D --run R        our own labels, joined from evals/datasets/labels/\n"
            f"\nThe gate is pre-registered ({RULES_ANCHOR}): exit 1 unless quadratic-weighted "
            f"kappa >= {AGREEMENT_GATE_KAPPA:.2f} on at least {AGREEMENT_GATE_MIN_N} pairs."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labelled",
        type=Path,
        help="a self-contained labelled file (JSONL, JSON array, or CSV)",
    )
    source.add_argument(
        "--dataset",
        type=Path,
        help="our eval set, to be joined with --run and its label sidecars",
    )
    parser.add_argument("--run", metavar="RUN_ID", help="the run whose responses were labelled")
    parser.add_argument(
        "--annotator",
        help="use only this annotator's sidecar; required when two annotators labelled one item",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help="an explicit label sidecar; repeatable. Default: <dataset>/labels/",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=None, help="default: <dataset>/labels"
    )
    parser.add_argument(
        "--label-space",
        type=LabelSpace,
        choices=list(LabelSpace),
        default=LabelSpace.RUBRIC_1_5,
        help=(
            f"the space the set was labelled in. Declared, never inferred from the values. Only "
            f"{LabelSpace.RUBRIC_1_5.value} can be validated here: agreement is ordinal"
        ),
    )
    parser.add_argument(
        "--column-map",
        action="append",
        default=[],
        metavar="INTERNAL=EXTERNAL",
        help=(
            "map one of this code's column names (left) onto your file's column name (right), "
            f"e.g. --column-map human_score=rating. Internal names: {list(INTERNAL_COLUMNS)}"
        ),
    )
    parser.add_argument(
        "--disagreements",
        type=int,
        default=DEFAULT_DISAGREEMENTS,
        help=f"how many of the worst disagreements to list (default: {DEFAULT_DISAGREEMENTS})",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "compare the judge against the deterministic checks, each scored against humans in "
            "its own space. Needs --dataset --run and native binary_behavioral labels; the judge "
            f"side is binarised by {BANDS_CITATION} and items it scored 3 are excluded and counted"
        ),
    )
    parser.add_argument(
        "--block-order",
        action="store_true",
        help=(
            "re-score every pair with the judge's blocks reordered and report signed score drift. "
            "Costs one further judge pass per reordering, so it is opt-in"
        ),
    )
    parser.add_argument(
        "--stability-samples",
        type=int,
        default=0,
        help=(
            f"sample each pair this many times at temperature {STABILITY_TEMPERATURE} with the "
            f"cache off, and report self-consistency. Minimum {MIN_STABILITY_SAMPLES}; costs this "
            "multiple of the primary pass, so it is opt-in"
        ),
    )
    parser.add_argument(
        "--stability-items",
        type=int,
        default=0,
        help="sample only this many pairs (default: all of them)",
    )
    parser.add_argument(
        "--stability-seed",
        type=int,
        default=None,
        help="seed the stability subsample instead of taking the first N in file order",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--json", action="store_true", help="print the artifact to stdout too")
    add_cache_arguments(parser)
    return parser


def _load_from_args(args: argparse.Namespace) -> tuple[list[LabelledPair], Path, dict[str, Any]]:
    """Resolve the CLI's two input paths into pairs, a digest source, and provenance."""
    if args.labelled is not None:
        column_map = parse_column_map(args.column_map)
        labelled = load_labelled(
            args.labelled, column_map=column_map, label_space=args.label_space
        )
        return (
            labelled,
            Path(args.labelled),
            {"source_kind": "labelled file", "column_map": column_map or None},
        )

    if not args.run:
        raise LabelledDataError("--dataset needs --run: labels are joined per run")

    labelled = load_labelled_from_run(
        args.dataset,
        args.run,
        runs_dir=args.runs_dir,
        labels_dir=args.labels_dir,
        annotator=args.annotator,
        label_paths=args.labels,
    )
    dataset = Path(args.dataset)
    sidecars = (
        [str(path) for path in args.labels]
        if args.labels
        else [
            str(path)
            for path in find_label_sidecars(
                dataset, args.run, labels_dir=args.labels_dir, annotator=args.annotator
            )
        ]
    )
    return (
        labelled,
        trace_path(args.run, args.runs_dir),
        {
            "source_kind": "dataset + run + label sidecars",
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_of_paths([dataset], root=dataset.parent) or "",
            "labelled_run_id": args.run,
            "label_sidecars": sidecars,
            "annotators": sorted({pair.annotator for pair in labelled if pair.annotator}),
            "response_sha256_verified": True,
        },
    )


def _resolve_baseline(
    args: argparse.Namespace,
) -> tuple[BaselineInputs | None, list[str]]:
    """Resolve `--baseline` into either the leg's inputs or the reasons it cannot run.

    Resolved before the judge is loaded, so a missing binary sidecar costs nothing. A failure here
    is recorded rather than fatal: the agreement figures and the gate do not depend on this leg, and
    exiting non-zero because an optional comparison was unavailable would conflate "the judge failed
    validation" with "we asked for one more table".

    **The grader path is refused with a reason.** The rules need a trace — retrieved chunk ids, tool
    calls, per-call parse outcomes — and a `--labelled` file has two columns and a score. There is
    nothing to run the rules over, so there is no rule instrument to compare the judge against.
    """
    if not args.baseline:
        return None, []
    if args.labelled is not None:
        return None, [
            "--baseline needs --dataset and --run. A --labelled file carries a prompt, a response "
            "and a score; the deterministic rules read a trace — retrieved chunk ids, tool calls, "
            "and per-call parse outcomes — so on this path there is no rule instrument to compare "
            "the judge against",
        ]
    try:
        return (
            load_baseline_inputs(
                args.dataset,
                args.run,
                runs_dir=args.runs_dir,
                labels_dir=args.labels_dir,
                annotator=args.annotator,
                label_paths=args.labels,
            ),
            [],
        )
    except (LabelledDataError, FileNotFoundError, OSError, ValueError) as exc:
        return None, [str(exc)]
    except NotImplementedError as exc:
        # A rule that is not implemented is not a rule instrument, which is a reason this leg
        # cannot run rather than a crash in the middle of one. Recorded in the same shape as every
        # other reason so the artifact says which rule is missing.
        return None, [
            f"a deterministic rule is not implemented, so there are no rule outputs to compare "
            f"against: {exc}"
        ]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: validate the judge and print the agreement report.

    Returns `EXIT_OK` only when the pre-registered gate passes. Non-zero otherwise, including
    when the labelled data cannot be used at all — an unvalidated judge must not be able to look
    clean in CI, which is the whole reason this is a gate rather than a report.
    """
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.stability_samples and args.stability_samples < MIN_STABILITY_SAMPLES:
        print(
            f"--stability-samples={args.stability_samples} cannot show variance; use at least "
            f"{MIN_STABILITY_SAMPLES}",
            file=sys.stderr,
        )
        return EXIT_FAILED

    try:
        labelled, source, provenance = _load_from_args(args)
    except (LabelledDataError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return EXIT_FAILED

    baseline_inputs, baseline_reasons = _resolve_baseline(args)

    judge = load_judge_model(no_cache=not cache_enabled(args.no_cache))
    # A second adapter with the cache off unconditionally. `sample_verdicts` refuses a
    # cache-enabled one because identical requests share a cache key, so every repeat after the
    # first would be a replay reported as perfect agreement — and honouring `--no-cache` here
    # would make that depend on a flag.
    stability_judge = load_judge_model(no_cache=True) if args.stability_samples else None

    try:
        report = validate_judge(
            labelled,
            source=source,
            judge=judge,
            runs_dir=args.runs_dir,
            disagreements=args.disagreements,
            stability_judge=stability_judge,
            stability_samples=args.stability_samples,
            stability_items=args.stability_items,
            stability_seed=args.stability_seed,
            baseline_inputs=baseline_inputs,
            baseline_unavailable_reasons=baseline_reasons,
            block_order=args.block_order,
            provenance=provenance,
        )
    except ValueError as exc:
        # `sample_verdicts` raises when the adapter caches or a sample came back cached. Both
        # mean no stability figure exists, and neither is a bug worth a traceback.
        print(f"stability sampling could not be measured: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print_report(report, console)
    console.print(f"[dim]artifact: {judge_validation_path(report.run_id, args.runs_dir)}[/dim]")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))

    return EXIT_OK if report.gate["passed"] else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
