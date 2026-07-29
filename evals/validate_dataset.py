"""Lint an eval dataset, and exit non-zero when it is wrong.

A dataset is the one input to this platform that nothing downstream re-derives: if an item is
mislabelled, every number computed from it is wrong in a way no aggregation can reveal. So the
checks here are deliberately unforgiving, and each one names a specific way a score goes quietly
wrong rather than a style preference.

**Every diagnostic carries a line number.** A message without one is unactionable on a
two-hundred-line file, and the practical effect is that the author fixes the first thing they
can find instead of the thing that was reported.

**The strictness is at the byte level, and that is not fussiness.** `agent.manifest.DatasetRef`
digests the file's *bytes*, and `assert_comparable` refuses two runs whose `dataset_sha256`
differ even when `dataset_path` matches. So a BOM, CRLF endings, or a missing trailing newline
are not cosmetic: they are the difference between a file that compares against an earlier run
and one that does not. This linter therefore reports them and — importantly — **never fixes
them**. There is no formatter here and no `--write`. A tool that tidied a dataset after one arm
had run would silently invalidate the comparison, which is the precise failure the digest exists
to catch and a formatter would defeat.

Duplicate keys get their own pass for a related reason: `json.loads` silently keeps the last
value, so `{"answerable": true, "answerable": false}` parses cleanly and `extra="forbid"` never
sees a problem. Only an `object_pairs_hook` catches it.

Diagnostics carry a stable `code` as well as a message, so CI and tests can assert on the code
while the prose stays free to improve. `--json` emits them machine-readably; `--strict` promotes
warnings to errors for a graded run, where a warning nobody read is a warning that did nothing.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent.trace import sha256_of_paths
from evals.schema import (
    COUNTERFACTUAL_PAIR_SIZE,
    ERROR_ATTACK_TYPE_FORBIDDEN,
    ERROR_ATTACK_TYPE_ON_CONTROL,
    ERROR_ATTACK_TYPE_REQUIRED,
    ERROR_BIAS_UNPAIRED,
    ERROR_COUNTERFACTUAL_INCOMPLETE,
    ERROR_SUBCATEGORY_UNKNOWN,
    ERROR_TURN_EMPTY,
    PAIR_INVARIANT_FIELDS,
    Axis,
    EvalItem,
    LabelRecord,
)

#: Diagnostic codes. Stable identifiers so a CI gate or a test asserts on the code rather than
#: on wording, which is then free to be improved without breaking either.
E_READ = "E-READ"
E_EMPTY = "E-EMPTY"
E_BOM = "E-BOM"
E_CRLF = "E-CRLF"
E_NO_TRAILING_NEWLINE = "E-NO-TRAILING-NEWLINE"
E_BLANK_LINE = "E-BLANK-LINE"
E_ENCODING = "E-ENCODING"
E_NOT_JSON = "E-NOT-JSON"
E_NOT_OBJECT = "E-NOT-OBJECT"
E_DUPLICATE_KEY = "E-DUPLICATE-KEY"
E_SCHEMA = "E-SCHEMA"
E_UNKNOWN_FIELD = "E-UNKNOWN-FIELD"
E_MISSING_FIELD = "E-MISSING-FIELD"
E_VOCABULARY = "E-VOCABULARY"
E_SUBCATEGORY = "E-SUBCATEGORY"
E_ATTACK_TYPE = "E-ATTACK-TYPE"
E_TURN_EMPTY = "E-TURN-EMPTY"
E_COUNTERFACTUAL_INCOMPLETE = "E-COUNTERFACTUAL-INCOMPLETE"
E_ID_DUPLICATE = "E-ID-DUPLICATE"
E_PAIR_SIZE = "E-PAIR-SIZE"
E_PAIR_VARIANT_SAME = "E-PAIR-VARIANT-SAME"
E_PAIR_ATTRIBUTE_MISMATCH = "E-PAIR-ATTRIBUTE-MISMATCH"
E_PAIR_FIELD_MISMATCH = "E-PAIR-FIELD-MISMATCH"
E_PAIR_TURN_COUNT = "E-PAIR-TURN-COUNT"
E_BIAS_UNPAIRED = "E-BIAS-UNPAIRED"
W_PAIR_DIFF_SPAN = "W-PAIR-DIFF-SPAN"
W_PAIR_DIFF_IDENTICAL = "W-PAIR-DIFF-IDENTICAL"
W_LABELS_DEGENERATE = "W-LABELS-DEGENERATE"
W_LABELS_STALE_DATASET = "W-LABELS-STALE-DATASET"
W_LABELS_UNKNOWN_ITEM = "W-LABELS-UNKNOWN-ITEM"
W_LABELS_MALFORMED = "W-LABELS-MALFORMED"

#: Printed above every pair diff. The check cannot verify semantics, and saying so is the
#: difference between a heuristic and a false claim. The adjacency caveat is not hypothetical:
#: "a young man" versus "an old woman" is two attributes in one contiguous span, and this check
#: passes it. That is why the diffs are printed for **every** pair and not only for the ones
#: that trip the warning — a check that is necessary but not sufficient has to put the evidence
#: in front of a human rather than report a verdict.
DIFF_HEURISTIC_NOTE = (
    "heuristic: 'differs in exactly one attribute' cannot be proven from text. This compares "
    "whitespace tokens and reports whether the differences form one contiguous span per turn. "
    "A single span is necessary, not sufficient — two attributes changed side by side read as "
    "one span. Read the diffs below rather than trusting the verdict."
)

#: Exit codes.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


@dataclass(frozen=True)
class Diagnostic:
    """One finding, located.

    `line` is 1-based and `None` only for findings about the file as a whole or about a sidecar
    rather than the dataset.
    """

    code: str
    message: str
    line: int | None = None
    #: Extra lines printed under the message, e.g. a token diff. Not part of the assertion
    #: surface, so tests match on `code` and humans read this.
    detail: list[str] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.code.startswith("E-")

    def format(self, path: Path) -> str:
        where = f"{path}:{self.line}" if self.line is not None else str(path)
        head = f"{where}: {'error' if self.is_error else 'warning'}: [{self.code}] {self.message}"
        return "\n".join([head, *(f"    {line}" for line in self.detail)])


@dataclass
class Report:
    """Everything the linter found, plus the summary it always prints.

    Kept as data rather than printed as it goes so that `--json` and the human rendering are
    the same findings, and so a caller can lint without the output.
    """

    path: Path
    dataset_sha256: str = ""
    diagnostics: list[Diagnostic] = field(default_factory=list)
    items: list[EvalItem] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    #: One entry per counterfactual pair: the token diff between its two variants, printed for
    #: every pair rather than only failing ones. Informational, never affecting the exit code.
    pair_diffs: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.is_error]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if not d.is_error]

    def add(self, code: str, message: str, line: int | None = None, **kw: Any) -> None:
        self.diagnostics.append(Diagnostic(code, message, line, **kw))

    def ok(self, *, strict: bool = False) -> bool:
        return not self.errors and not (strict and self.warnings)

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "dataset_sha256": self.dataset_sha256,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "diagnostics": [asdict(d) for d in self.diagnostics],
            "pair_diffs": self.pair_diffs,
            "summary": self.summary,
        }


# --------------------------------------------------------------------------------------
# Byte and line level
# --------------------------------------------------------------------------------------


def _check_bytes(raw: bytes, report: Report) -> list[str] | None:
    """Check the file's bytes and return its lines, or None if it cannot be read as JSONL.

    Runs before any parsing because these are the properties the digest is taken over. A file
    that fails here may still parse fine, and would still be a different file from the one an
    earlier arm ran.
    """
    if not raw:
        report.add(E_EMPTY, "file is empty")
        return None

    if raw.startswith(b"\xef\xbb\xbf"):
        report.add(
            E_BOM,
            "file starts with a UTF-8 BOM. The digest is over bytes, so this file differs from "
            "the same items saved without one",
            line=1,
        )
        raw = raw[3:]

    if b"\r\n" in raw:
        first_crlf = raw.split(b"\r\n", 1)[0].count(b"\n") + 1
        report.add(E_CRLF, "file has CRLF line endings; JSONL here is LF only", line=first_crlf)
        raw = raw.replace(b"\r\n", b"\n")

    if not raw.endswith(b"\n"):
        report.add(
            E_NO_TRAILING_NEWLINE,
            "file does not end with a newline, so appending an item rewrites the last line too",
            line=raw.count(b"\n") + 1,
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        report.add(E_ENCODING, f"file is not valid UTF-8: {exc}")
        return None

    lines = text.split("\n")
    # A trailing newline yields a final empty element that is not a line; anything else empty is.
    if lines and lines[-1] == "":
        lines.pop()

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            report.add(E_BLANK_LINE, "blank line; JSONL is one object per line", line=number)

    return lines


def _load_object(line: str, number: int, report: Report) -> dict[str, Any] | None:
    """Parse one line into a dict, reporting duplicate keys the stdlib would have hidden."""
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    try:
        parsed = json.loads(line, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        report.add(E_NOT_JSON, f"line is not valid JSON: {exc.msg} at column {exc.colno}", number)
        return None

    if not isinstance(parsed, dict):
        report.add(E_NOT_OBJECT, f"line is a JSON {type(parsed).__name__}, not an object", number)
        return None

    for key in duplicates:
        report.add(
            E_DUPLICATE_KEY,
            f"duplicate key {key!r}; json silently keeps the last value, so the first is lost "
            "without any error",
            number,
        )
    return parsed


#: Maps a Pydantic error `type` onto a diagnostic code. Keyed on the machine-readable type that
#: `evals.schema` raises (and that Pydantic uses for built-in failures), never on the message
#: text, so improving the wording of an error cannot change which check a reader sees fail.
_ERROR_CODES: dict[str, str] = {
    ERROR_TURN_EMPTY: E_TURN_EMPTY,
    ERROR_SUBCATEGORY_UNKNOWN: E_SUBCATEGORY,
    ERROR_ATTACK_TYPE_REQUIRED: E_ATTACK_TYPE,
    ERROR_ATTACK_TYPE_FORBIDDEN: E_ATTACK_TYPE,
    ERROR_ATTACK_TYPE_ON_CONTROL: E_ATTACK_TYPE,
    ERROR_COUNTERFACTUAL_INCOMPLETE: E_COUNTERFACTUAL_INCOMPLETE,
    ERROR_BIAS_UNPAIRED: E_BIAS_UNPAIRED,
    "extra_forbidden": E_UNKNOWN_FIELD,
    "missing": E_MISSING_FIELD,
    "enum": E_VOCABULARY,
}


def _format_validation_error(exc: ValidationError) -> list[str]:
    """One readable line per problem, in the form a linter reader expects."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<item>"
        lines.append(f"{loc}: {err['msg']}")
    return lines


def _report_validation_error(exc: ValidationError, number: int, report: Report) -> None:
    """Turn a `ValidationError` into one diagnostic per problem, each with its own code.

    One diagnostic per problem rather than one per line: an item with a typo'd field name *and*
    a bad subcategory has two things wrong with it, and reporting a single "does not validate"
    would send the author back for a second run to discover the other.
    """
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        code = _ERROR_CODES.get(str(err["type"]), E_SCHEMA)
        prefix = f"{loc}: " if loc else ""
        report.add(code, f"{prefix}{err['msg']}", number)


# --------------------------------------------------------------------------------------
# File-level checks
# --------------------------------------------------------------------------------------


def _check_ids(pairs: list[tuple[int, EvalItem]], report: Report) -> None:
    """Ids must be unique: a label refers to an id, so a duplicate re-points existing labels."""
    first_seen: dict[str, int] = {}
    for number, item in pairs:
        if item.id in first_seen:
            report.add(
                E_ID_DUPLICATE,
                f"id {item.id!r} already used on line {first_seen[item.id]}; ids are what "
                "labels and trace records join on",
                number,
            )
        else:
            first_seen[item.id] = number


def _token_diff(a: str, b: str) -> tuple[int, list[str]]:
    """Return the number of differing spans between two turns, and a readable diff.

    Whitespace tokens rather than characters, because a character diff on "he"/"she" reports one
    span for what a reader would call one word anyway, and on "45-year-old"/"18-year-old"
    reports a span inside a token that is harder to eyeball than the token itself.
    """
    a_tokens, b_tokens = a.split(), b.split()
    matcher = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    spans = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    detail = [
        f"- {' '.join(a_tokens[op[1] : op[2]]) or '(nothing)'}"
        f"  ->  + {' '.join(b_tokens[op[3] : op[4]]) or '(nothing)'}"
        for op in spans
    ]
    return len(spans), detail


def _check_pairs(pairs: list[tuple[int, EvalItem]], report: Report) -> None:
    """Pair integrity, then the one-attribute heuristic.

    A pair whose members differ in more than the varied attribute produces a delta that includes
    that other difference, and nothing downstream can decompose it — the number still looks like
    a bias measurement.
    """
    grouped: dict[str, list[tuple[int, EvalItem]]] = defaultdict(list)
    for number, item in pairs:
        if item.counterfactual_id is not None:
            grouped[item.counterfactual_id].append((number, item))

    for number, item in pairs:
        if item.axis is Axis.BIAS and item.counterfactual_id is None:
            report.add(
                E_BIAS_UNPAIRED,
                f"bias item {item.id!r} has no counterfactual_id. Bias is a within-pair delta, "
                "so a lone item yields no measurement but would be counted as though it did",
                number,
            )

    for pair_id, members in sorted(grouped.items()):
        lines = ", ".join(str(n) for n, _ in members)
        if len(members) != COUNTERFACTUAL_PAIR_SIZE:
            report.add(
                E_PAIR_SIZE,
                f"counterfactual_id {pair_id!r} has {len(members)} item(s) on line(s) {lines}; "
                f"a pair is exactly {COUNTERFACTUAL_PAIR_SIZE}, since a delta needs two sides "
                "and three variants give it no defined direction",
                members[0][0],
            )
            continue

        (line_a, first), (line_b, second) = members

        if first.counterfactual_variant == second.counterfactual_variant:
            report.add(
                E_PAIR_VARIANT_SAME,
                f"both items in pair {pair_id!r} have counterfactual_variant "
                f"{first.counterfactual_variant!r}, so nothing was varied",
                line_b,
            )

        if first.counterfactual_attribute != second.counterfactual_attribute:
            report.add(
                E_PAIR_ATTRIBUTE_MISMATCH,
                f"pair {pair_id!r} varies {first.counterfactual_attribute!r} on line {line_a} "
                f"but {second.counterfactual_attribute!r} on line {line_b}; a pair varies one "
                "attribute",
                line_b,
            )

        for name in PAIR_INVARIANT_FIELDS:
            a_value, b_value = getattr(first, name), getattr(second, name)
            if a_value != b_value:
                report.add(
                    E_PAIR_FIELD_MISMATCH,
                    f"pair {pair_id!r} differs in {name}: {a_value!r} on line {line_a} vs "
                    f"{b_value!r} on line {line_b}. Any difference beyond the varied attribute "
                    "is folded into the delta and cannot be separated from it afterwards",
                    line_b,
                )

        if len(first.turns) != len(second.turns):
            report.add(
                E_PAIR_TURN_COUNT,
                f"pair {pair_id!r} has {len(first.turns)} turn(s) on line {line_a} but "
                f"{len(second.turns)} on line {line_b}; the two arms of a pair must be the "
                "same conversation",
                line_b,
            )
            continue

        multi_span: list[str] = []
        identical = True
        report.pair_diffs.append(
            f"{pair_id} (lines {line_a}, {line_b}): {first.counterfactual_attribute}"
        )
        for index, (a_turn, b_turn) in enumerate(zip(first.turns, second.turns, strict=True)):
            span_count, detail = _token_diff(a_turn, b_turn)
            if span_count:
                identical = False
            report.pair_diffs.extend(
                f"  turn {index}: {line}" for line in (detail or ["(no difference)"])
            )
            if span_count > 1:
                multi_span.append(f"turn {index}: {span_count} differing spans")
                multi_span.extend(f"  {line}" for line in detail)

        if identical:
            report.add(
                W_PAIR_DIFF_IDENTICAL,
                f"pair {pair_id!r} has identical turns, so the two variants ask the same "
                f"question and the delta measures only model nondeterminism",
                line_b,
            )
        elif multi_span:
            report.add(
                W_PAIR_DIFF_SPAN,
                f"pair {pair_id!r} differs in more than one span, which usually means more "
                "than one attribute changed",
                line_b,
                detail=[DIFF_HEURISTIC_NOTE, *multi_span],
            )


# --------------------------------------------------------------------------------------
# Labels sidecar
# --------------------------------------------------------------------------------------


def _check_labels(
    label_paths: Sequence[Path],
    items: Sequence[EvalItem],
    report: Report,
) -> dict[str, Any]:
    """Read label sidecars and report on their coverage and balance.

    Read-only, and it takes the *last* record per `(run_id, item_id)` because the sidecars are
    append-only: an earlier record for the same key was superseded, not duplicated.
    """
    known = {item.id for item in items}
    latest: dict[tuple[str, str], LabelRecord] = {}

    for path in label_paths:
        if not path.exists():
            report.add(W_LABELS_MALFORMED, f"labels file not found: {path}")
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = LabelRecord.model_validate_json(line)
            except ValidationError as exc:
                report.add(
                    W_LABELS_MALFORMED,
                    f"{path}:{number}: not a valid label record: "
                    f"{'; '.join(_format_validation_error(exc))}",
                )
                continue
            if record.dataset_sha256 != report.dataset_sha256:
                report.add(
                    W_LABELS_STALE_DATASET,
                    f"{path}:{number}: label for {record.item_id!r} was made against dataset "
                    f"{record.dataset_sha256[:12]} but this file is "
                    f"{report.dataset_sha256[:12]}. The labelled text is not the text here; "
                    "the label cannot be assumed to still apply",
                )
            if record.item_id not in known:
                report.add(
                    W_LABELS_UNKNOWN_ITEM,
                    f"{path}:{number}: label refers to item {record.item_id!r}, which is not "
                    "in this dataset",
                )
            latest[record.key] = record

    label_counts = Counter(
        record.label.value for record in latest.values() if record.label is not None
    )
    score_counts = Counter(
        str(record.score) for record in latest.values() if record.score is not None
    )

    if len(latest) > 1 and len(label_counts) == 1:
        only = next(iter(label_counts))
        report.add(
            W_LABELS_DEGENERATE,
            f"every one of the {len(latest)} binary labels is {only!r}. Judge validation needs "
            "both classes: agreement statistics are undefined or meaningless on one class, and "
            "a judge that always says the same thing would score perfectly",
        )
    if len(latest) > 1 and len(score_counts) == 1:
        report.add(
            W_LABELS_DEGENERATE,
            f"every one of the {len(latest)} rubric scores is the same value; correlation with "
            "human labels is undefined when one side has no variance",
        )

    labelled_items = {record.item_id for record in latest.values()}
    return {
        "n_label_records": len(latest),
        "n_items_labelled": len(labelled_items & known),
        "coverage": round(len(labelled_items & known) / len(known), 3) if known else 0.0,
        "binary_labels": dict(sorted(label_counts.items())),
        "rubric_scores": dict(sorted(score_counts.items())),
        "runs": sorted({record.run_id for record in latest.values()}),
    }


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------


def _summarise(items: Sequence[EvalItem]) -> dict[str, Any]:
    """Counts that catch the dataset which is 80% one subcategory.

    Cheap, and the alternative is discovering the imbalance while trying to explain a result.
    """
    pair_ids = {item.counterfactual_id for item in items if item.counterfactual_id is not None}
    return {
        "n_items": len(items),
        "by_axis": dict(sorted(Counter(item.axis.value for item in items).items())),
        "by_subcategory": dict(sorted(Counter(item.subcategory for item in items).items())),
        "by_attack_type": dict(
            sorted(
                Counter(
                    item.attack_type.value for item in items if item.attack_type is not None
                ).items()
            )
        ),
        "n_pairs": len(pair_ids),
        "n_paired_items": sum(1 for item in items if item.counterfactual_id is not None),
        "n_multi_turn": sum(1 for item in items if item.is_multi_turn),
        "max_turns": max((len(item.turns) for item in items), default=0),
        "answerable": sum(1 for item in items if item.answerable),
        "unanswerable": sum(1 for item in items if not item.answerable),
        "n_with_expected_tool": sum(1 for item in items if item.expected_tool is not None),
        "n_with_must_include": sum(1 for item in items if item.must_include),
    }


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def validate_dataset(path: Path, *, label_paths: Sequence[Path] = ()) -> Report:
    """Lint `path` and return everything found. Reads only; never writes."""
    path = Path(path)
    report = Report(path=path)

    try:
        raw = path.read_bytes()
    except OSError as exc:
        report.add(E_READ, f"cannot read file: {exc}")
        return report

    # The same call `DatasetRef.for_file` makes, so the linter, the manifest, and the labeler
    # cannot disagree about which file they are talking about.
    report.dataset_sha256 = sha256_of_paths([path], root=path.parent) or ""

    lines = _check_bytes(raw, report)
    if lines is None:
        return report

    parsed: list[tuple[int, EvalItem]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        obj = _load_object(line, number, report)
        if obj is None:
            continue
        try:
            parsed.append((number, EvalItem.model_validate(obj)))
        except ValidationError as exc:
            _report_validation_error(exc, number, report)

    report.items = [item for _, item in parsed]
    _check_ids(parsed, report)
    _check_pairs(parsed, report)

    report.summary = _summarise(report.items)
    if label_paths:
        report.summary["labels"] = _check_labels(label_paths, report.items, report)
    return report


def render_report(report: Report, *, strict: bool = False) -> str:
    """Render diagnostics then the summary, in that order.

    Diagnostics first because they are what the reader has to act on, and a summary above them
    is a summary they scroll past.
    """
    out: list[str] = [d.format(report.path) for d in report.diagnostics]
    if out:
        out.append("")

    if report.pair_diffs:
        out.append("counterfactual pairs, for human review:")
        out.append(f"  {DIFF_HEURISTIC_NOTE}")
        out.extend(f"  {line}" for line in report.pair_diffs)
        out.append("")

    out.append(f"{report.path}  sha256={report.dataset_sha256[:12] or '(none)'}")
    for key, value in report.summary.items():
        if isinstance(value, dict):
            rendered = ", ".join(f"{k}={v}" for k, v in value.items()) or "(none)"
            out.append(f"  {key}: {rendered}")
        else:
            out.append(f"  {key}: {value}")

    n_err, n_warn = len(report.errors), len(report.warnings)
    verdict = "OK" if report.ok(strict=strict) else "FAILED"
    promoted = " (warnings are errors under --strict)" if strict and n_warn else ""
    out.append(f"{verdict}: {n_err} error(s), {n_warn} warning(s){promoted}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Lint a dataset: `agentseval-validate-dataset FILE [--labels F] [--strict] [--json]`.

    Returns 0 when clean, 1 when it is not. Never writes to the dataset — see the module
    docstring on why there is no formatter here.
    """
    parser = argparse.ArgumentParser(
        prog="agentseval-validate-dataset",
        description="Lint an eval dataset (JSONL of evals.schema.EvalItem). Read-only.",
    )
    parser.add_argument("dataset", type=Path, help="the .jsonl dataset to lint")
    parser.add_argument(
        "--labels",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="a label sidecar to check coverage and balance against; repeatable",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as errors; use this on a dataset about to be graded",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output for CI")
    args = parser.parse_args(argv)

    report = validate_dataset(args.dataset, label_paths=args.labels)

    if args.json:
        payload = report.to_json()
        payload["ok"] = report.ok(strict=args.strict)
        payload["strict"] = args.strict
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        stream = sys.stdout if report.ok(strict=args.strict) else sys.stderr
        print(render_report(report, strict=args.strict), file=stream)

    return EXIT_OK if report.ok(strict=args.strict) else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
