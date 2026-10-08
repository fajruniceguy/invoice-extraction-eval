"""Run-to-run noise: how much two runs of the SAME setting differ from each other.

A model that samples is not a function. A difference between two runs is a finding only if it is
larger than the difference between two runs that changed nothing, so this is the yardstick the
sweep, the regression gate and every before/after table should be read against. Given N runs of
one setting (fresh samples, `--cache refresh`), it reports the spread of the headline numbers, how
many labeled fields change correctness between any two runs, which fields are unstable, and how far
output tokens and latency move for the same document.

Built on `failures.collect` (the scorer's own `score_field`) and `perf.PerfStats`, so the numbers
mean exactly what the scorecard's mean. Offline and deterministic; dollars are recomputed from
tokens at ONE stated price, because runs made while a price changed are otherwise not comparable.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from evalkit.failures import FailureError, FieldCase, collect, read_record
from evalkit.perf import PerfStats


class NoiseError(Exception):
    """The runs cannot be compared as repeats of one another."""


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    labeled: int
    correct: int
    answered: int  # not abstained
    perf: PerfStats

    @property
    def accuracy(self) -> float:
        return self.correct / self.labeled

    @property
    def selective(self) -> float | None:
        return self.correct / self.answered if self.answered else None

    @property
    def coverage(self) -> float:
        return self.answered / self.labeled

    def cost_at(self, price_in: Decimal, price_out: Decimal) -> Decimal:
        return (self.perf.input_tokens * price_in + self.perf.output_tokens * price_out) / Decimal(
            1_000_000
        )


@dataclass(frozen=True)
class PairFlips:
    first: str
    second: str
    first_only: int  # right in the first run, not in the second
    second_only: int  # right in the second run, not in the first

    @property
    def flips(self) -> int:
        return self.first_only + self.second_only


@dataclass(frozen=True)
class Unstable:
    doc_id: str
    name: str
    outcomes: tuple[str, ...]  # the returned outcome in each run, in run order


@dataclass(frozen=True)
class DocSpread:
    doc_id: str
    tokens: tuple[int, ...]  # billable output tokens in each run
    latency_ms: tuple[float, ...]  # live calls only, in run order


@dataclass(frozen=True)
class NoiseReport:
    runs: list[RunSummary]
    pairs: list[PairFlips]
    unstable: list[Unstable]  # outcome differs in at least one run
    correctness_changed: int  # fields right in some runs and not in others
    docs: list[DocSpread]

    @property
    def labeled(self) -> int:
        return self.runs[0].labeled


def _index(cases: list[FieldCase]) -> dict[tuple[str, str], FieldCase]:
    return {(case.doc_id, case.name): case for case in cases}


def measure(runs: list[Path], *, manifest_path: Path, labels_dir: Path) -> NoiseReport:
    """Compare N runs of one setting. They must label and answer exactly the same fields."""
    if len(runs) < 2:
        raise NoiseError("the noise of a setting needs at least two runs of it")
    collected = [collect(run, manifest_path=manifest_path, labels_dir=labels_dir) for run in runs]
    indexed = [_index(result.cases) for result in collected]
    keys = sorted(indexed[0])
    for result, index in zip(collected[1:], indexed[1:], strict=True):
        if sorted(index) != keys:
            raise NoiseError(
                f"{result.run_id} does not cover the same labeled fields as {collected[0].run_id}"
            )

    summaries: list[RunSummary] = []
    for run, result in zip(runs, collected, strict=True):
        doc_ids = sorted({case.doc_id for case in result.cases})
        records = [read_record(run, doc_id) for doc_id in doc_ids]
        summaries.append(
            RunSummary(
                run_id=result.run_id,
                labeled=len(result.cases),
                correct=sum(case.returned.correct for case in result.cases),
                answered=sum(not case.withheld for case in result.cases),
                perf=PerfStats.from_records(records),
            )
        )

    pairs = [
        PairFlips(
            first=summaries[i].run_id,
            second=summaries[j].run_id,
            first_only=sum(
                indexed[i][k].returned.correct and not indexed[j][k].returned.correct for k in keys
            ),
            second_only=sum(
                indexed[j][k].returned.correct and not indexed[i][k].returned.correct for k in keys
            ),
        )
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    unstable = [
        Unstable(doc, name, tuple(index[(doc, name)].returned.outcome for index in indexed))
        for doc, name in keys
        if len({index[(doc, name)].returned.outcome for index in indexed}) > 1
    ]
    changed = sum(len({index[key].returned.correct for index in indexed}) > 1 for key in keys)

    docs: list[DocSpread] = []
    for doc_id in sorted({doc for doc, _ in keys}):
        tokens: list[int] = []
        latency: list[float] = []
        for run in runs:
            try:
                record = read_record(run, doc_id)
            except FailureError as exc:
                raise NoiseError(str(exc)) from exc
            tokens.append(int((record.usage or {}).get("billable_output_tokens", 0)))
            if not record.cache_hit:
                latency.append(record.latency_ms)
        docs.append(DocSpread(doc_id, tuple(tokens), tuple(latency)))
    return NoiseReport(summaries, pairs, unstable, changed, docs)


# ---------------------------------------------------------------- markdown


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.1f} s"


def to_markdown(
    report: NoiseReport,
    *,
    price_in: Decimal,
    price_out: Decimal,
    notes: str | None = None,
) -> str:
    runs = report.runs
    accuracies = [run.accuracy for run in runs]
    flips = [pair.flips for pair in report.pairs]
    tokens = [run.perf.output_tokens for run in runs]
    lines = [
        f"# Run-to-run noise: {len(runs)} runs of one setting",
        "",
        f"Runs: {', '.join(f'`{run.run_id}`' for run in runs)}. {report.labeled} labeled fields in "
        f"{len(report.docs)} documents, so the independent sample is {len(report.docs)} documents, "
        "not the number of runs. Generated by `scripts/noise_report.py`; the tables are data, the "
        "reading at the end is hand-written.",
        "",
        "## Headline numbers",
        "",
        f"Cost is recomputed from tokens at ${price_in}/${price_out} per million tokens "
        "(input/output) for every run, whatever price each run was billed at when it was made.",
        "",
        "| run | accuracy | selective | coverage | repairs | output tokens | cost | latency p50 "
        "| p95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        selective = "n/a" if run.selective is None else f"{run.selective:.3f}"
        lines.append(
            f"| {run.run_id} | {run.accuracy:.3f} | {selective} | {run.coverage:.3f} | "
            f"{run.perf.repairs} | {run.perf.output_tokens} | "
            f"${run.cost_at(price_in, price_out):.4f} | {_ms(run.perf.latency_p50_ms)} | "
            f"{_ms(run.perf.latency_p95_ms)} |"
        )
    lines += [
        "",
        f"Accuracy ranges from {min(accuracies):.3f} to {max(accuracies):.3f} "
        f"({(max(accuracies) - min(accuracies)) * 100:.1f} points); output tokens from "
        f"{min(tokens)} to {max(tokens)}"
        + (f" ({(max(tokens) / min(tokens) - 1) * 100:.0f}% more)." if min(tokens) else "."),
        "",
        "## Fields that change between two runs of the same setting",
        "",
        "A flip is a labeled field that one run got right and the other did not.",
        "",
        "| run A | run B | right only in A | right only in B | flips |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {p.first} | {p.second} | {p.first_only} | {p.second_only} | {p.flips} |"
        for p in report.pairs
    ]
    lines += [
        "",
        f"Between two runs {min(flips)} to {max(flips)} of {report.labeled} fields flip "
        f"({min(flips) / report.labeled * 100:.1f}% to {max(flips) / report.labeled * 100:.1f}%). "
        f"{report.correctness_changed} fields are right in some runs and not in others; "
        f"{len(report.unstable)} change outcome at least once (including abstained or wrong).",
        "",
        "| document | field | outcome in each run |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {u.doc_id} | {u.name} | {', '.join(u.outcomes)} |" for u in report.unstable]
    lines += [
        "",
        "## One document, several runs",
        "",
        "Billable output tokens and live latency of the same document in each run. Reasoning "
        "models think for a different length each time.",
        "",
        "| document | output tokens (each run) | min..max | latency (each run) | min..max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spread in report.docs:
        latencies = list(spread.latency_ms)
        low, high = (min(latencies), max(latencies)) if latencies else (None, None)
        lines.append(
            f"| {spread.doc_id} | {', '.join(str(t) for t in spread.tokens)} | "
            f"{min(spread.tokens)}..{max(spread.tokens)} | "
            f"{', '.join(_ms(v) for v in latencies)} | {_ms(low)}..{_ms(high)} |"
        )
    p95s = [run.perf.latency_p95_ms for run in runs if run.perf.latency_p95_ms is not None]
    count = len(report.docs)
    if p95s:
        slowest = (95 * count + 99) // 100 == count  # nearest-rank p95 is the maximum
        why = "the single slowest call" if slowest else f"call number {(95 * count + 99) // 100}"
        lines += [
            "",
            f"The run-level p95 latency ranges from {_ms(min(p95s))} to {_ms(max(p95s))}: with "
            f"{count} documents per run it is {why}, so one document decides it.",
        ]
    if notes:
        lines += ["", "## Reading", "", notes.strip()]
    return "\n".join(lines) + "\n"
