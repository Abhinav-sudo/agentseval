"""Measure the retrieval score distribution, so the confidence floor is chosen from data.

PROJECT.md exposes `min_score` on `search` and `lookup_kb` and leaves the threshold unset
"until there is measured data to choose it from". This is that measurement. It is retrieval
only — no model calls, no network, no judge — so the number can be produced from a checkout
with nothing configured but a built index.

**What is being calibrated.** `agent.guardrails` enforces grounding by abstaining when a turn
retrieved nothing at or above the floor. The floor therefore trades two failures against each
other: set too low it never fires, and set too high it abstains on questions the corpus answers
well, which lands in `false_refusal_rate` as over-refusal. So the floor is chosen from the
answerable side of the eval sets and its effect on the unanswerable side is *reported* rather
than optimised — see `CALIBRATION_RULE`.

**The rule is fixed here, before any graded run, and takes one side of the data.** The floor is
the `FLOOR_PERCENTILE`th percentile of the top-1 cosine score over items the dataset marks
`answerable`, rounded down to `FLOOR_DECIMALS` places. That bounds the share of answerable
questions the stage would abstain on at roughly `FLOOR_PERCENTILE`% by construction, and it
cannot be tuned to flatter the guardrail, because guardrail results are not among its inputs.
Rounding down rather than to nearest keeps the rounding on the permissive side of the rule it
implements. A floor picked to maximise separation between the two groups would be fitted against
the outcome the stage is later measured on, which is the failure pre-registration prevents.

**Two honest limits on the number, both in the conservative direction.**

* The query scored here is the item's own final user turn (`schema.SCORED_TURN_INDEX`), not the
  `query` argument a model would write. A model that rephrases well scores *higher* than this,
  so a floor fitted on the user's wording abstains less often in practice than the percentile
  claims, not more.
* `answerable` is the dataset author's judgement about the corpus, not about retrieval. An
  answerable item whose wording happens to retrieve poorly pulls the floor down, which again errs
  toward abstaining less.

The artifact is written to `runs/retrieval_calibration.json` and carries the corpus digest, the
embedding model, and the retrieval settings, because a floor derived from one corpus under one
encoder says nothing about another. `lookup_kb.GROUNDING_MIN_SCORE` is the pre-registered value
and cites this file.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent.tools.lookup_kb import (
    DEFAULT_KB_DIR,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    GROUNDING_MIN_SCORE,
    KnowledgeBase,
)
from agent.trace import DEFAULT_RUNS_DIR, utc_now_iso
from evals.schema import EvalItem

#: The eval sets the main floor is calibrated over. `injection.jsonl` is absent on purpose: it
#: runs against the composed fixture corpus, so its scores describe a different corpus and
#: pooling them would calibrate a floor against text no main run reads. Calibrate it separately
#: with `--dataset` and `--kb-dir` if a fixture floor is ever wanted.
DEFAULT_DATASETS: tuple[Path, ...] = (
    Path("evals/datasets/hallucination.jsonl"),
    Path("evals/datasets/bias.jsonl"),
    Path("evals/datasets/safety.jsonl"),
)

#: Where the artifact lands. One file, overwritten: unlike a run manifest this is a measurement
#: of the corpus rather than of an event, so there is nothing to keep a history of — re-running
#: it over an unchanged corpus under an unchanged encoder produces the same bytes.
CALIBRATION_FILENAME = "retrieval_calibration.json"

#: The percentile of answerable top-1 scores that becomes the floor. Five, so the rule tolerates
#: one badly-worded answerable item in twenty rather than being dragged to the single worst one,
#: which a plain minimum would be.
FLOOR_PERCENTILE = 5.0

#: Places the floor is rounded down to. Two, because the floor is printed, quoted, and typed into
#: a constant by hand, and a fifteen-digit float is none of those things.
FLOOR_DECIMALS = 2

#: The percentiles reported for each group. The tails are what a floor decision turns on; the
#: median is there so a reader can see the two groups apart at a glance.
REPORTED_PERCENTILES: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0)

#: The rule, in one string, copied into the artifact. Recorded rather than left in this
#: docstring so the file says what produced the number without anyone reading this module.
CALIBRATION_RULE = (
    f"floor = floor_to_{FLOOR_DECIMALS}dp(percentile_{FLOOR_PERCENTILE:g}(top-1 cosine over "
    "items with answerable=true)); computed from the answerable side only, so the rejection "
    "rate on unanswerable items is an observation and never an input"
)

EXIT_OK = 0
EXIT_FAILED = 1


class CalibrationError(RuntimeError):
    """The score distribution cannot be measured as specified.

    Raised rather than warned for the reason `runner.PreflightError` is: a floor derived from a
    stale index describes a corpus other than the one on disk, and discovering that after four
    graded runs is expensive.
    """


@dataclass(frozen=True)
class QueryScore:
    """One item's retrieval outcome: what it asked, and how close the corpus got.

    `top_score` is the best cosine similarity any chunk reached, which is the quantity a floor
    is compared against — not the mean over `top_k`, since `search` drops hits below the floor
    individually and one clearing hit is enough to ground an answer.
    """

    item_id: str
    dataset: str
    axis: str
    subcategory: str
    answerable: bool
    query: str
    top_score: float
    top_chunk_id: str | None


@dataclass
class GroupStats:
    """The distribution of one group's top-1 scores.

    Attributes:
        percentiles: Keyed by the percentile, linearly interpolated as `numpy.percentile`
            computes it. Reported rather than a mean and a standard deviation, because the
            question a floor asks is about a tail and neither of those answers it.
    """

    name: str
    n: int
    mean: float = 0.0
    stdev: float = 0.0
    percentiles: dict[str, float] = field(default_factory=dict)

    @classmethod
    def of(cls, name: str, scores: Sequence[float]) -> GroupStats:
        """Summarise `scores`, tolerating an empty group with an `n=0` row.

        Empty is a real answer here — a dataset with no unanswerable items has no unanswerable
        distribution — and refusing would push the caller into omitting the group, which is the
        reporting failure PROJECT.md forbids elsewhere.
        """
        values = [float(score) for score in scores]
        if not values:
            return cls(name=name, n=0)
        return cls(
            name=name,
            n=len(values),
            mean=float(statistics.fmean(values)),
            stdev=float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            percentiles={
                f"p{percentile:g}": float(np.percentile(values, percentile))
                for percentile in REPORTED_PERCENTILES
            },
        )


@dataclass
class Calibration:
    """The measurement and the floor it implies.

    Attributes:
        floor: The value `CALIBRATION_RULE` yields, or None when there is no answerable item to
            derive one from. None rather than a default: a floor nobody measured is exactly what
            PROJECT.md declines to pick, and returning 0.0 would dress that up as a decision.
        registered_floor: What `lookup_kb.GROUNDING_MIN_SCORE` currently says. Carried so the
            artifact shows agreement or drift between the constant in the code and the corpus in
            front of it, which is the one thing a reader of the constant cannot check.
        would_abstain_answerable: Share of answerable items whose top-1 score falls below the
            floor — the over-refusal the stage would cause, which the rule bounds by design.
        would_abstain_unanswerable: The same on the unanswerable side, which is the stage's
            rejection power. An observation, never an input to the floor.
    """

    kb_dir: str
    kb_sha256: str | None
    embedding_model: str
    top_k: int
    tool_min_score: float
    datasets: list[str]
    n_items: int
    rule: str
    floor: float | None
    registered_floor: float
    by_answerable: dict[str, GroupStats] = field(default_factory=dict)
    by_dataset: dict[str, GroupStats] = field(default_factory=dict)
    by_subcategory: dict[str, GroupStats] = field(default_factory=dict)
    would_abstain_answerable: float = 0.0
    would_abstain_unanswerable: float = 0.0
    generated_at: str = ""
    scores: list[QueryScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render for the artifact, per-item scores last so the summary reads first."""
        return {
            "generated_at": self.generated_at,
            "kb_dir": self.kb_dir,
            "kb_sha256": self.kb_sha256,
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
            "tool_min_score": self.tool_min_score,
            "datasets": list(self.datasets),
            "n_items": self.n_items,
            "rule": self.rule,
            "floor": self.floor,
            "registered_floor": self.registered_floor,
            "would_abstain_answerable": self.would_abstain_answerable,
            "would_abstain_unanswerable": self.would_abstain_unanswerable,
            "by_answerable": {
                name: vars(stats) for name, stats in self.by_answerable.items()
            },
            "by_dataset": {name: vars(stats) for name, stats in self.by_dataset.items()},
            "by_subcategory": {
                name: vars(stats) for name, stats in self.by_subcategory.items()
            },
            "scores": [vars(score) for score in self.scores],
        }


def floor_from(answerable_scores: Sequence[float]) -> float | None:
    """Apply `CALIBRATION_RULE` to the answerable group's top-1 scores.

    The whole rule in one place, so the number in the artifact, the number in the printed
    summary, and the number a test checks are the same computation rather than three.

    Returns None for an empty group: there is nothing to take a percentile of, and a floor is
    not something to invent when the data for it is absent.
    """
    values = [float(score) for score in answerable_scores]
    if not values:
        return None
    percentile = float(np.percentile(values, FLOOR_PERCENTILE))
    scale = 10**FLOOR_DECIMALS
    # Floor rather than round: the rule permits at most FLOOR_PERCENTILE% of answerable items
    # below the cut, and rounding up could push a further item under it.
    return math.floor(percentile * scale) / scale


def score_items(
    items: Sequence[tuple[str, EvalItem]],
    kb: KnowledgeBase,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[QueryScore]:
    """Retrieve for each item's scored turn and record the best score it reached.

    `min_score` is the *tool's* floor and stays at its default here: a calibration run that
    filtered hits away could not see the distribution it is trying to measure. The floor being
    derived is applied afterwards, arithmetically, in `calibrate`.

    A query that retrieves nothing at all scores 0.0 with no chunk id, which is honest: an empty
    corpus and a query orthogonal to every chunk are the same fact from a floor's point of view.
    """
    scored: list[QueryScore] = []
    for dataset, item in items:
        query = item.scored_turn
        hits = kb.search(query, top_k=top_k, min_score=min_score)
        best = hits[0] if hits else None
        scored.append(
            QueryScore(
                item_id=item.id,
                dataset=dataset,
                axis=item.axis.value,
                subcategory=item.subcategory,
                answerable=item.answerable,
                query=query,
                top_score=float(best.score) if best is not None else 0.0,
                top_chunk_id=best.chunk.chunk_id if best is not None else None,
            )
        )
    return scored


def calibrate(
    dataset_paths: Sequence[Path] = DEFAULT_DATASETS,
    *,
    kb_dir: Path = DEFAULT_KB_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> Calibration:
    """Measure the top-1 score distribution over `dataset_paths` and derive the floor.

    Raises:
        CalibrationError: a dataset is missing or unreadable, or the retrieval index is stale.
            A stale index would describe a corpus other than the one on disk, and a floor is
            only meaningful against the corpus it was measured on.
    """
    # Imported here rather than at module scope, for the reason `metrics._load_items` does it:
    # `evals.runner` pulls in the agent, the tool registry, and the provider adapters, and a
    # retrieval measurement needs none of them.
    from evals.runner import load_dataset

    kb_dir = Path(kb_dir)
    kb = KnowledgeBase(kb_dir, auto_build=False)
    reason = kb.staleness_reason()
    if reason is not None:
        raise CalibrationError(
            f"the retrieval index for {kb_dir} is stale ({reason}); a floor measured against it "
            f"would describe a corpus other than the one on disk. Rebuild it with "
            f"agentseval-index --kb-dir {kb_dir}"
        )

    items: list[tuple[str, EvalItem]] = []
    for path in dataset_paths:
        path = Path(path)
        try:
            loaded = load_dataset(path)
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"cannot read {path}: {exc}") from exc
        items.extend((path.name, item) for item in loaded)

    scores = score_items(items, kb, top_k=top_k)
    answerable = [score.top_score for score in scores if score.answerable]
    unanswerable = [score.top_score for score in scores if not score.answerable]
    floor = floor_from(answerable)

    def below(values: Sequence[float]) -> float:
        if floor is None or not values:
            return 0.0
        return sum(1 for value in values if value < floor) / len(values)

    subcategories = sorted({score.subcategory for score in scores})
    return Calibration(
        kb_dir=str(kb_dir),
        kb_sha256=kb.corpus_fingerprint(),
        embedding_model=EMBEDDING_MODEL,
        top_k=top_k,
        tool_min_score=DEFAULT_MIN_SCORE,
        datasets=[str(path) for path in dataset_paths],
        n_items=len(scores),
        rule=CALIBRATION_RULE,
        floor=floor,
        registered_floor=GROUNDING_MIN_SCORE,
        by_answerable={
            "answerable": GroupStats.of("answerable", answerable),
            "unanswerable": GroupStats.of("unanswerable", unanswerable),
        },
        by_dataset={
            path.name: GroupStats.of(
                path.name, [s.top_score for s in scores if s.dataset == path.name]
            )
            for path in map(Path, dataset_paths)
        },
        by_subcategory={
            name: GroupStats.of(
                name, [s.top_score for s in scores if s.subcategory == name]
            )
            for name in subcategories
        },
        would_abstain_answerable=below(answerable),
        would_abstain_unanswerable=below(unanswerable),
        generated_at=utc_now_iso(),
        scores=scores,
    )


def write_calibration(
    calibration: Calibration, runs_dir: Path = DEFAULT_RUNS_DIR
) -> Path:
    """Write the artifact and return its path."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / CALIBRATION_FILENAME
    target.write_text(
        json.dumps(calibration.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def format_calibration(calibration: Calibration) -> str:
    """Render the summary a human reads before pre-registering the floor."""
    lines = [
        f"kb dir:        {calibration.kb_dir}",
        f"corpus hash:   {calibration.kb_sha256}",
        f"model:         {calibration.embedding_model} (top_k {calibration.top_k})",
        f"items:         {calibration.n_items} over {len(calibration.datasets)} dataset(s)",
        "",
        "top-1 cosine by answerability:",
    ]
    for name, stats in calibration.by_answerable.items():
        if not stats.n:
            lines.append(f"  {name:<14} n=0")
            continue
        percentiles = "  ".join(
            f"{key}={value:.3f}" for key, value in stats.percentiles.items()
        )
        lines.append(f"  {name:<14} n={stats.n:<4} mean={stats.mean:.3f}  {percentiles}")

    lines.extend(["", f"rule:          {calibration.rule}"])
    if calibration.floor is None:
        lines.append(
            "floor:         not derivable — no answerable item in these datasets, so there is "
            "no distribution to take a percentile of"
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"floor:         {calibration.floor:.2f}",
            f"registered:    {calibration.registered_floor:.2f}"
            + (
                ""
                if math.isclose(calibration.floor, calibration.registered_floor)
                else "  <- differs from the measurement; lookup_kb.GROUNDING_MIN_SCORE is "
                "pre-registered, so changing it is a decision and not a refresh"
            ),
            f"would abstain: {calibration.would_abstain_answerable:.1%} of answerable items "
            f"(the rule bounds this at ~{FLOOR_PERCENTILE:g}%), "
            f"{calibration.would_abstain_unanswerable:.1%} of unanswerable ones",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `agentseval-calibrate-retrieval [--dataset PATH ...] [--kb-dir DIR]`.

    Prints the distribution and the floor the rule yields, and writes the artifact. It does not
    edit `lookup_kb.GROUNDING_MIN_SCORE`: the constant is pre-registered, so moving it is a
    deliberate decision recorded in a commit rather than something a script does on a Tuesday.
    """
    parser = argparse.ArgumentParser(
        prog="agentseval-calibrate-retrieval",
        description=(
            "Measure the top-1 retrieval score distribution over the eval sets and derive the "
            "pre-registered grounding floor from the answerable side of it. Retrieval only: no "
            "model calls, no network."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        metavar="PATH",
        help=(
            "an eval set to measure. Repeatable; defaults to the three main sets. "
            "injection.jsonl is excluded by default because it reads a different corpus"
        ),
    )
    parser.add_argument(
        "--kb-dir", type=Path, default=DEFAULT_KB_DIR, help="corpus directory"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"retrieval breadth (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"where {CALIBRATION_FILENAME} is written (default: {DEFAULT_RUNS_DIR})",
    )
    args = parser.parse_args(argv)

    datasets = tuple(args.dataset) if args.dataset else DEFAULT_DATASETS
    try:
        calibration = calibrate(datasets, kb_dir=args.kb_dir, top_k=args.top_k)
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(format_calibration(calibration))
    path = write_calibration(calibration, args.runs_dir)
    print(f"\nwrote {path}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
