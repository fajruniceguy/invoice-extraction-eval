"""Calibration of the confidence and the abstention threshold, on labeled runs.

Does a confidence of 0.9 mean the field is right nine times in ten, and where should the
abstention threshold sit? Both are answered from `failures.FieldCase`, so "right" is the scorer's
own comparator. The abstention policy decides what a threshold can move:

  - a field the policy cannot flip is FIXED: one the model abstained itself, one carrying a HARD
    code (`evidence_not_grounded`), and an `absent` field (the policy never flips absent). Fixed
    abstentions stay abstained at every threshold; absent fields stay answered.
  - every other field is GOVERNED: the model extracted a value, nothing contradicts it, and it is
    returned iff its final (capped) confidence is at least the threshold. Its correctness is the
    correctness of the model's own value (`raw`), because the validators only ever withhold a value,
    never change one.

The threshold is chosen by a rule fixed before any result (`choose_threshold`): the lowest one
whose pooled selective accuracy reaches `TARGET_SELECTIVE`. It is a recommendation for the data it
was fitted on, not a result; a threshold chosen on dev is reported on test.

Pooling several runs of one setting adds points, not documents: the independent sample is the
number of distinct documents, and every report says so. Offline and deterministic.
"""

import math
from dataclasses import dataclass
from typing import Literal

from docextract.validate.codes import HARD_CODES
from evalkit.failures import FieldCase, RunCases

TARGET_SELECTIVE = 0.95
# the first bin collects the raw model confidences below 0.5 (capped ones never go there)
BIN_EDGES: tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)

Kind = Literal["withheld", "answered", "governed"]


@dataclass(frozen=True)
class Point:
    """One field as the threshold sees it."""

    confidence: float
    correct: bool


@dataclass(frozen=True)
class Pool:
    runs: int
    documents: int
    labeled: int  # every field instance, fixed abstentions included
    withheld: int  # fixed abstentions: never answered, whatever the threshold
    answered: list[Point]  # fixed answers (absent): always returned
    governed: list[Point]  # returned iff confidence >= threshold
    governed_raw: list[Point]  # the model's own confidence on those same governed fields


def classify(case: FieldCase) -> Kind:
    if case.raw.predicted_status == "abstained":
        return "withheld"  # the model gave no answer
    if any(code in HARD_CODES for code in case.codes):
        return "withheld"  # contradicted by its own evidence
    if case.raw.predicted_status == "absent":
        return "answered"
    return "governed"


def pool(runs: list[RunCases]) -> Pool:
    withheld = 0
    answered: list[Point] = []
    governed: list[Point] = []
    governed_raw: list[Point] = []
    labeled = 0
    documents: set[str] = set()
    for run in runs:
        for case in run.cases:
            labeled += 1
            documents.add(case.doc_id)
            correct = case.raw.outcome == "correct"
            kind = classify(case)
            if kind == "withheld":
                withheld += 1
            elif kind == "answered":
                answered.append(Point(case.confidence, correct))
            else:
                governed.append(Point(case.confidence, correct))
                governed_raw.append(Point(case.raw_confidence, correct))
    return Pool(len(runs), len(documents), labeled, withheld, answered, governed, governed_raw)


@dataclass(frozen=True)
class Operating:
    """The pooled result when fields below `threshold` are withheld."""

    threshold: float
    answered: int
    correct: int
    labeled: int

    @property
    def accuracy(self) -> float:
        """Abstentions count as errors, like the scorecard's headline."""
        return self.correct / self.labeled

    @property
    def selective(self) -> float | None:
        return self.correct / self.answered if self.answered else None

    @property
    def coverage(self) -> float:
        return self.answered / self.labeled


def operating_point(data: Pool, threshold: float) -> Operating:
    kept = [p for p in data.governed if p.confidence >= threshold]
    points = [*data.answered, *kept]
    return Operating(
        threshold=threshold,
        answered=len(points),
        correct=sum(p.correct for p in points),
        labeled=data.labeled,
    )


def curve(data: Pool) -> list[Operating]:
    """One operating point per distinct governed confidence, plus 0 (answer every governed field),
    in increasing threshold order: coverage falls as the threshold rises."""
    thresholds = sorted({0.0, *(p.confidence for p in data.governed)})
    return [operating_point(data, t) for t in thresholds]


@dataclass(frozen=True)
class Chosen:
    point: Operating
    target: float


def choose_threshold(points: list[Operating], target: float = TARGET_SELECTIVE) -> Chosen | None:
    """The lowest threshold (highest coverage) whose selective accuracy reaches `target`; None
    when no threshold does."""
    for point in sorted(points, key=lambda p: p.threshold):
        if point.selective is not None and point.selective >= target:
            return Chosen(point, target)
    return None


@dataclass(frozen=True)
class Bin:
    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability(points: list[Point], edges: tuple[float, ...] = BIN_EDGES) -> list[Bin]:
    """Equal-rule bins [low, high) (the last one closed at 1.0); empty bins are left out."""
    bins: list[Bin] = []
    for index, (low, high) in enumerate(zip(edges, edges[1:], strict=False)):
        last = index == len(edges) - 2
        inside = [p for p in points if p.confidence >= low and (p.confidence < high or last)]
        if inside:
            bins.append(
                Bin(
                    low,
                    high,
                    len(inside),
                    sum(p.confidence for p in inside) / len(inside),
                    sum(p.correct for p in inside) / len(inside),
                )
            )
    return bins


def ece(bins: list[Bin]) -> float | None:
    """Expected calibration error: the count-weighted gap between confidence and accuracy."""
    total = sum(b.count for b in bins)
    if not total:
        return None
    return sum(b.count * abs(b.accuracy - b.mean_confidence) for b in bins) / total


def brier(points: list[Point]) -> float | None:
    if not points:
        return None
    return sum((p.confidence - p.correct) ** 2 for p in points) / len(points)


def aurc(points: list[Point]) -> float | None:
    """Area under the risk-coverage curve over `points` answered in decreasing confidence (ties
    answered together): the mean selective risk. Lower is better; compare with the error rate of
    answering everything, which is what a confidence that ranks nothing would give."""
    if not points:
        return None
    ordered = sorted(points, key=lambda p: -p.confidence)
    total = 0.0
    errors = 0
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end].confidence == ordered[index].confidence:
            end += 1
        errors += sum(not p.correct for p in ordered[index:end])
        total += (errors / end) * (end - index)
        index = end
    return total / len(ordered)


@dataclass(frozen=True)
class CalibrationReport:
    data: Pool
    in_force: Operating
    in_force_threshold: float
    curve: list[Operating]
    chosen: Chosen | None
    final_bins: list[Bin]  # governed fields only: what the threshold decides
    raw_bins: list[Bin]  # the same fields, by the model's own confidence


def analyse(runs: list[RunCases], *, abstain_below: float) -> CalibrationReport:
    data = pool(runs)
    points = curve(data)
    return CalibrationReport(
        data=data,
        in_force=operating_point(data, abstain_below),
        in_force_threshold=abstain_below,
        curve=points,
        chosen=choose_threshold(points),
        final_bins=reliability(data.governed),
        raw_bins=reliability(data.governed_raw),
    )


# ---------------------------------------------------------------- markdown


def _f(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _bin_rows(bins: list[Bin]) -> list[str]:
    return [
        f"| [{b.low:.2f}, {b.high:.2f}{']' if b.high == 1.0 else ')'} | {b.count} | "
        f"{b.mean_confidence:.3f} | {b.accuracy:.3f} | {b.accuracy - b.mean_confidence:+.3f} |"
        for b in bins
    ]


def to_markdown(report: CalibrationReport, *, run_ids: list[str], notes: str | None = None) -> str:
    data = report.data
    absent_right = sum(p.correct for p in data.answered)
    lines = [
        f"# Calibration: {', '.join(f'`{r}`' for r in run_ids)}",
        "",
        f"{data.labeled} labeled field instances from {data.runs} run(s) of {data.documents} "
        f"documents. The runs repeat the same documents, so the independent sample is "
        f"{data.documents} documents, not {data.labeled} fields. Generated by "
        "`scripts/calibrate.py`; the tables are data, the reading at the end is hand-written.",
        "",
        "## What a threshold can move",
        "",
        f"- {data.withheld} instances are **fixed abstentions** (the model abstained itself, or a "
        "value its own evidence does not support: `evidence_not_grounded`). No threshold changes "
        "them.",
        f"- {len(data.answered)} are `absent` fields, always answered.",
        f"- {len(data.governed)} are **governed**: a value was extracted and nothing contradicts "
        "it, so it is returned iff its final confidence reaches the threshold.",
        "",
        "## Reliability",
        "",
        "Accuracy of the model's value against the confidence it came with, over the governed "
        "fields only (the ones a threshold decides). Final confidence is the model's, lowered "
        "by the caps the validators imply (and never raised). The always-answered `absent` "
        f"fields ({absent_right} of {len(data.answered)} right) are left out: they would flatter "
        "the table.",
        "",
        "**Final confidence**",
        "",
        "| bin | n | mean confidence | accuracy | gap |",
        "| --- | --- | --- | --- | --- |",
        *_bin_rows(report.final_bins),
        "",
        f"ECE {_f(ece(report.final_bins))}, Brier {_f(brier(data.governed))}.",
        "",
        "**The model's own confidence** (the same fields, before the validators)",
        "",
        "| bin | n | mean confidence | accuracy | gap |",
        "| --- | --- | --- | --- | --- |",
        *_bin_rows(report.raw_bins),
        "",
        f"ECE {_f(ece(report.raw_bins))}, Brier {_f(brier(data.governed_raw))}.",
        "",
        "## Risk and coverage",
        "",
        "Fields below the threshold are withheld. Accuracy counts a withheld field as an error "
        "(the scorecard's headline); selective accuracy is over what is returned; coverage is "
        "the share returned.",
        "",
        "| threshold | returned | correct | accuracy | selective | coverage |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {p.threshold:.2f} | {p.answered} | {p.correct} | {p.accuracy:.3f} | "
        f"{_f(p.selective)} | {p.coverage:.3f} |"
        for p in report.curve
    ]
    errors = sum(not p.correct for p in data.governed)
    lines += [
        "",
        f"Ranking power of the final confidence over the governed fields: area under the "
        f"risk-coverage curve {_f(aurc(data.governed))}; a confidence that ranks nothing would "
        f"give the plain error rate, "
        f"{_f(errors / len(data.governed) if data.governed else None)}.",
        "",
        "## The threshold",
        "",
        f"Rule, fixed before any result: the lowest threshold whose pooled selective accuracy is "
        f"at least {TARGET_SELECTIVE:.2f}.",
        "",
        "| | threshold | accuracy | selective | coverage |",
        "| --- | --- | --- | --- | --- |",
        f"| in force (`ABSTAIN_BELOW`) | {report.in_force_threshold:.2f} | "
        f"{report.in_force.accuracy:.3f} | {_f(report.in_force.selective)} | "
        f"{report.in_force.coverage:.3f} |",
    ]
    if report.chosen is None:
        lines += [
            "| chosen by the rule | none | n/a | n/a | n/a |",
            "",
            f"No threshold reaches {TARGET_SELECTIVE:.2f} selective accuracy on these data: the "
            "current `ABSTAIN_BELOW` stays.",
        ]
    else:
        c = report.chosen.point
        lines += [
            f"| chosen by the rule | {c.threshold:.2f} | {c.accuracy:.3f} | {_f(c.selective)} | "
            f"{c.coverage:.3f} |",
            "",
            "A higher threshold trades headline accuracy for selective accuracy; where to sit is "
            "a product decision about the cost of a wrong answer against a withheld one. This is "
            "a recommendation fitted on these documents, to be reported on the test split.",
        ]
    if notes:
        lines += ["", "## Reading", "", notes.strip()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- SVG (no plotting dependency)

_W, _H, _L, _R, _T, _B = 480, 400, 56, 20, 36, 52


def _x(value: float) -> float:
    return _L + value * (_W - _L - _R)


def _y(value: float, low: float = 0.0) -> float:
    """Map `value` in [low, 1] onto the plot height."""
    return _H - _B - (value - low) / (1.0 - low) * (_H - _T - _B)


def _frame(title: str, xlabel: str, ylabel: str, y_low: float = 0.0) -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        'font-family="sans-serif" font-size="11">',
        f'<rect width="{_W}" height="{_H}" fill="white"/>',
        f'<text x="{_W / 2}" y="20" text-anchor="middle" font-size="13">{title}</text>',
        f'<rect x="{_x(0):.1f}" y="{_y(1, y_low):.1f}" width="{_x(1) - _x(0):.1f}" '
        f'height="{_y(y_low, y_low) - _y(1, y_low):.1f}" fill="none" stroke="#444"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{_x(tick):.1f}" y1="{_y(y_low):.1f}" x2="{_x(tick):.1f}" '
            f'y2="{_y(y_low) + 4:.1f}" stroke="#444"/>'
        )
        parts.append(
            f'<text x="{_x(tick):.1f}" y="{_y(y_low) + 16:.1f}" '
            f'text-anchor="middle">{tick:g}</text>'
        )
    for step in range(5):
        tick = y_low + (1.0 - y_low) * step / 4
        parts.append(
            f'<line x1="{_x(0) - 4:.1f}" y1="{_y(tick, y_low):.1f}" x2="{_x(0):.1f}" '
            f'y2="{_y(tick, y_low):.1f}" stroke="#444"/>'
        )
        parts.append(
            f'<text x="{_x(0) - 8:.1f}" y="{_y(tick, y_low) + 4:.1f}" '
            f'text-anchor="end">{tick:.2f}</text>'
        )
    parts.append(
        f'<text x="{(_x(0) + _x(1)) / 2:.1f}" y="{_H - 12}" text-anchor="middle">{xlabel}</text>'
    )
    parts.append(
        f'<text x="14" y="{(_y(y_low, y_low) + _y(1, y_low)) / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {(_y(y_low, y_low) + _y(1, y_low)) / 2:.1f})">{ylabel}</text>'
    )
    return parts


def reliability_svg(report: CalibrationReport) -> str:
    """Accuracy against mean confidence per bin, final and raw, with the diagonal."""
    parts = _frame("Reliability (point area = fields in the bin)", "mean confidence", "accuracy")
    parts.append(
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(1):.1f}" '
        'stroke="#999" stroke-dasharray="4 3"/>'
    )
    for bins, colour, name, row in (
        (report.final_bins, "#1b6ec2", "final confidence", 0),
        (report.raw_bins, "#d9822b", "model's own confidence", 1),
    ):
        if len(bins) > 1:
            path = " ".join(f"{_x(b.mean_confidence):.1f},{_y(b.accuracy):.1f}" for b in bins)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{colour}"/>')
        total = max((b.count for b in bins), default=1)
        for b in bins:
            radius = 3 + 7 * (b.count / total) ** 0.5
            parts.append(
                f'<circle cx="{_x(b.mean_confidence):.1f}" cy="{_y(b.accuracy):.1f}" '
                f'r="{radius:.1f}" fill="{colour}" fill-opacity="0.65"/>'
            )
        parts.append(
            f'<circle cx="{_x(0.05):.1f}" cy="{_y(0.95) + 16 * row:.1f}" r="4" fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{_x(0.05) + 10:.1f}" y="{_y(0.95) + 4 + 16 * row:.1f}">{name}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def risk_coverage_svg(report: CalibrationReport) -> str:
    """Selective accuracy against coverage as the threshold rises, with the points marked."""
    points = [p for p in report.curve if p.selective is not None]
    marked = [report.in_force] + ([report.chosen.point] if report.chosen else [])
    shown = [p.selective for p in [*points, *marked] if p.selective is not None]
    lowest = min([*shown, TARGET_SELECTIVE])
    low = max(0.0, math.floor((lowest - 0.02) * 20) / 20)  # zoomed axis, labelled
    parts = _frame(
        "Risk-coverage (threshold rises as coverage falls)",
        "coverage",
        f"selective accuracy (axis starts at {low:.2f})",
        low,
    )
    if points:
        path = " ".join(f"{_x(p.coverage):.1f},{_y(p.selective or 0.0, low):.1f}" for p in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="#1b6ec2"/>')
        for p in points:
            parts.append(
                f'<circle cx="{_x(p.coverage):.1f}" cy="{_y(p.selective or 0.0, low):.1f}" r="3" '
                'fill="#1b6ec2"/>'
            )
    target_y = _y(TARGET_SELECTIVE, low)
    parts.append(
        f'<line x1="{_x(0):.1f}" y1="{target_y:.1f}" x2="{_x(1):.1f}" y2="{target_y:.1f}" '
        'stroke="#999" stroke-dasharray="4 3"/>'
    )
    parts.append(
        f'<text x="{_x(0) + 6:.1f}" y="{target_y - 4:.1f}">target {TARGET_SELECTIVE:g}</text>'
    )
    marks = [(report.in_force, "#c0392b", "in force")]
    if report.chosen is not None:
        marks.append((report.chosen.point, "#2e8b57", "chosen"))
    for point, colour, name in marks:
        if point.selective is None:
            continue
        cx, cy = _x(point.coverage), _y(point.selective, low)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" stroke="{colour}" '
            'stroke-width="2"/>'
        )
        parts.append(f'<text x="{cx + 9:.1f}" y="{cy + 4:.1f}" fill="{colour}">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
