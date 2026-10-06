"""Dev/test split assignment and blind-document selection.

The unit of assignment is the `content_group` (documents that carry the same content: two photos
of one invoice, a degraded variant and its source), so the same content can never sit on both
sides of the split. Groups are stratified by (real | constructed, doc_type) so that every kind of
document is represented on both sides, and ~35% of them go to dev.

The assignment is deterministic and does not depend on row order: within a stratum, groups are
ranked by a hash of (seed, stratum, group), so adding a document never reshuffles the others.
Assignments that already exist are never moved (the test set must stay put once people have
looked at dev results); only new groups are filled in, so that each stratum still lands on its
target. `force` starts from scratch.

Some rules that fall out of the data, not of taste:
  * a stratum with one group goes to test (there is nothing to put on the dev side);
  * a stratum with two or more groups gets at least one group on each side;
  * degraded variants (and any `gt_from` duplicate) follow their group, they are never assigned
    on their own;
  * constructed invoices are their own strata: their layouts are known to us and they must not
    stand in for real documents.

Blind documents are hand-labeled from the page alone, with no model draft anywhere near them
(see ground_truth/LABELING.md). 8 to 10 REAL test documents are chosen round-robin across doc
types, whole content groups at a time. Every other hand-labeled document defaults to
label_mode=prelabel. Documents whose ground truth already exists are `locked`: their split and
label_mode cannot change.
A label_mode already in the manifest is a decision and is kept, including the `prelabel` this
module writes by default: a document that may already carry a model-influenced label must never
turn into a blind one. New documents are the ones that can still be picked.
"""

import hashlib
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from docextract.ingest.manifest import ManifestRow, check_manifest

DEFAULT_SEED = "docextract-split-v1"
DEFAULT_DEV_FRACTION = "0.35"
DEFAULT_BLIND_COUNT = 10
MIN_BLIND = 8

Stratum = tuple[str, str]  # ("real" | "constructed", doc_type)


class SplitError(ValueError):
    """The manifest cannot be split yet. `problems` lists every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class Change:
    doc_id: str
    field: str  # "split" or "label_mode"
    old: str | None
    new: str | None


@dataclass(frozen=True)
class StratumSummary:
    stratum: Stratum
    groups: int
    dev: int
    test: int


@dataclass(frozen=True)
class SplitPlan:
    rows: list[ManifestRow]
    strata: list[StratumSummary]
    changes: list[Change]
    blind: list[str]  # every doc_id that is blind after the plan, sorted
    warnings: list[str] = field(default_factory=list)


def _rank(seed: str, stratum: Stratum, group: str) -> str:
    """A stable pseudo-random sort key for a group: independent of every other group."""
    text = f"{seed}|{stratum[0]}|{stratum[1]}|{group}"
    return hashlib.sha256(text.encode()).hexdigest()


def allocate_dev_counts(
    sizes: Mapping[Stratum, int], dev_fraction: str | float | Fraction = DEFAULT_DEV_FRACTION
) -> dict[Stratum, int]:
    """How many groups of each stratum go to dev (the rest go to test).

    Largest remainder over the strata that can be split: those with two or more groups, each
    getting between 1 and size - 1. A single-group stratum gets 0 (it goes to test)."""
    fraction = Fraction(str(dev_fraction)).limit_denominator(10_000)
    if not 0 < fraction < 1:
        raise ValueError(f"dev_fraction must be between 0 and 1, got {dev_fraction}")
    ordered = sorted(sizes)
    eligible = [s for s in ordered if sizes[s] >= 2]
    ideal = {s: fraction * sizes[s] for s in eligible}
    low = {s: 1 for s in eligible}
    high = {s: sizes[s] - 1 for s in eligible}
    total = sum(sizes.values())
    target = int(fraction * total + Fraction(1, 2))  # round half up
    target = max(sum(low.values()), min(sum(high.values()), target))
    alloc = {s: max(low[s], min(high[s], int(ideal[s]))) for s in eligible}  # floor, clamped
    while sum(alloc.values()) < target:
        room = [s for s in eligible if alloc[s] < high[s]]
        best = max(room, key=lambda s: (ideal[s] - alloc[s], -ordered.index(s)))
        alloc[best] += 1
    while sum(alloc.values()) > target:
        room = [s for s in eligible if alloc[s] > low[s]]
        worst = min(room, key=lambda s: (ideal[s] - alloc[s], ordered.index(s)))
        alloc[worst] -= 1
    return {s: alloc.get(s, 0) for s in ordered}


def _hand_labeled(row: ManifestRow) -> bool:
    """Rows a human labels: not one that borrows labels (a variant, a duplicate: `gt_from`), not
    a constructed one (whose ground truth is generated with the document)."""
    return row.gt_from is None and row.capture != "constructed"


def _group_of(row: ManifestRow, by_id: Mapping[str, ManifestRow]) -> str | None:
    if row.content_group:
        return row.content_group
    if row.gt_from and row.gt_from in by_id:
        return by_id[row.gt_from].content_group
    return None


def assign_splits(
    rows: Sequence[ManifestRow],
    *,
    seed: str = DEFAULT_SEED,
    dev_fraction: str | float | Fraction = DEFAULT_DEV_FRACTION,
    blind_count: int = DEFAULT_BLIND_COUNT,
    force: bool = False,
    locked: Collection[str] = (),
) -> SplitPlan:
    """Plan the split and the blind selection. Returns new rows; nothing is mutated."""
    by_id = {row.doc_id: row for row in rows}
    problems: list[str] = []

    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        group = _group_of(row, by_id)
        if group is None:
            problems.append(f"{row.doc_id}: no content_group (fill it in the manifest)")
        else:
            groups[group].append(row)

    stratum_of: dict[str, Stratum] = {}
    for group, members in groups.items():
        kinds = {m.doc_type for m in members if m.doc_type}
        constructed = {m.capture == "constructed" for m in members}
        if not kinds:
            problems.append(f"content_group {group!r}: no doc_type on any of its documents")
        elif len(kinds) > 1:
            problems.append(f"content_group {group!r} mixes doc types {sorted(kinds)}")
        elif len(constructed) > 1:
            problems.append(f"content_group {group!r} mixes constructed and real documents")
        else:
            origin = "constructed" if constructed == {True} else "real"
            stratum_of[group] = (origin, next(iter(kinds)))
    if problems:
        raise SplitError(problems)

    existing: dict[str, str] = {}
    if not force:
        for group, members in groups.items():
            found = {m.split for m in members if m.split}
            if len(found) > 1:
                problems.append(f"content_group {group!r} straddles splits {sorted(found)}")
            elif found:
                existing[group] = next(iter(found))
    if problems:
        raise SplitError(problems)

    members_by_stratum: dict[Stratum, list[str]] = defaultdict(list)
    for group, stratum in stratum_of.items():
        members_by_stratum[stratum].append(group)
    allocation = allocate_dev_counts(
        {s: len(g) for s, g in members_by_stratum.items()}, dev_fraction
    )

    assigned: dict[str, str] = dict(existing)
    for stratum, stratum_groups in members_by_stratum.items():
        fresh = sorted(
            (g for g in stratum_groups if g not in existing), key=lambda g: _rank(seed, stratum, g)
        )
        already_dev = sum(1 for g in stratum_groups if existing.get(g) == "dev")
        need_dev = max(0, min(len(fresh), allocation[stratum] - already_dev))
        for index, group in enumerate(fresh):
            assigned[group] = "dev" if index < need_dev else "test"

    label_mode = _choose_label_modes(
        groups, stratum_of, assigned, seed=seed, blind_count=blind_count, force=force
    )

    warnings: list[str] = []
    new_rows: list[ManifestRow] = []
    changes: list[Change] = []
    for row in rows:
        group = _group_of(row, by_id)
        assert group is not None
        split = assigned[group]
        mode = label_mode.get(row.doc_id, row.label_mode)
        if row.doc_id in locked and (split != row.split or mode != row.label_mode):
            problems.append(
                f"{row.doc_id}: has ground truth, so its split/label_mode cannot change "
                f"({row.split}/{row.label_mode} -> {split}/{mode})"
            )
        if row.split != split:
            changes.append(Change(row.doc_id, "split", row.split, split))
        if row.label_mode != mode:
            changes.append(Change(row.doc_id, "label_mode", row.label_mode, mode))
        new_rows.append(row.model_copy(update={"split": split, "label_mode": mode}))
    problems.extend(check_manifest(new_rows))
    if problems:
        raise SplitError(problems)

    blind = sorted(r.doc_id for r in new_rows if r.label_mode == "blind")
    if len(blind) < min(MIN_BLIND, blind_count):
        warnings.append(
            f"only {len(blind)} blind document(s): there are not enough hand-labeled real "
            f"test documents (wanted {blind_count}, at least {MIN_BLIND})"
        )
    for row in new_rows:
        if row.label_mode == "blind" and row.split != "test":
            warnings.append(f"{row.doc_id}: blind but in the {row.split} split")
    strata = [
        StratumSummary(
            stratum,
            len(stratum_groups),
            sum(1 for g in stratum_groups if assigned[g] == "dev"),
            sum(1 for g in stratum_groups if assigned[g] == "test"),
        )
        for stratum, stratum_groups in sorted(members_by_stratum.items())
    ]
    return SplitPlan(new_rows, strata, changes, blind, warnings)


def _choose_label_modes(
    groups: Mapping[str, list[ManifestRow]],
    stratum_of: Mapping[str, Stratum],
    assigned: Mapping[str, str],
    *,
    seed: str,
    blind_count: int,
    force: bool,
) -> dict[str, str]:
    """label_mode for every hand-labeled document (documents nobody labels are not in the result).

    A choice already in the manifest is kept (a blind group is completed: its blank rows become
    blind). Otherwise a REAL TEST group is a candidate for blind, picked round-robin over doc
    types until `blind_count` rows are blind; everything left defaults to prelabel."""
    modes: dict[str, str] = {}
    blind_rows = 0
    per_type: dict[str, int] = defaultdict(int)
    candidates: dict[str, list[str]] = defaultdict(list)  # doc_type -> groups, in rank order

    for group in sorted(groups, key=lambda g: _rank(seed, stratum_of[g], g)):
        hand = _hand(groups[group])
        if not hand:
            continue
        kind = stratum_of[group][1]
        real_test = assigned[group] == "test"  # constructed groups have no hand-labeled rows
        given = [] if force else [m.label_mode for m in hand]
        if "blind" in given:
            for member in hand:
                modes[member.doc_id] = member.label_mode or "blind"
            if real_test:
                blind_rows += len(hand)
                per_type[kind] += len(hand)
        elif any(given):  # somebody already decided "prelabel" for part of this group
            for member in hand:
                modes[member.doc_id] = member.label_mode or "prelabel"
        elif real_test:
            candidates[kind].append(group)
        else:
            for member in hand:
                modes[member.doc_id] = "prelabel"

    types = sorted(candidates)
    while blind_rows < blind_count and types:
        kind = min(types, key=lambda t: (per_type[t], t))
        fits = [g for g in candidates[kind] if blind_rows + len(_hand(groups[g])) <= blind_count]
        if not fits:
            types.remove(kind)
            continue
        group = fits[0]
        candidates[kind].remove(group)
        for member in _hand(groups[group]):
            modes[member.doc_id] = "blind"
        blind_rows += len(_hand(groups[group]))
        per_type[kind] += len(_hand(groups[group]))
        if not candidates[kind]:
            types.remove(kind)

    for leftover in candidates.values():  # not chosen: a model drafts them, a human verifies
        for group in leftover:
            for member in _hand(groups[group]):
                modes[member.doc_id] = "prelabel"
    return modes


def _hand(members: Iterable[ManifestRow]) -> list[ManifestRow]:
    return [m for m in members if _hand_labeled(m)]


def format_plan(plan: SplitPlan, *, seed: str, dev_fraction: str | float | Fraction) -> str:
    groups = sum(s.groups for s in plan.strata)
    dev = sum(s.dev for s in plan.strata)
    wanted = float(Fraction(str(dev_fraction)).limit_denominator(10_000)) * 100
    lines = [
        f"split: {groups} content group(s) -> {dev} dev / {groups - dev} test "
        f"(wanted {wanted:.0f}% dev, seed {seed!r})"
    ]
    lines.append(f"  {'stratum':<34}{'groups':>7}{'dev':>5}{'test':>6}")
    for summary in plan.strata:
        name = f"{summary.stratum[0]} / {summary.stratum[1]}"
        note = "  (single group: test)" if summary.groups == 1 else ""
        lines.append(f"  {name:<34}{summary.groups:>7}{summary.dev:>5}{summary.test:>6}{note}")
    by_id = {row.doc_id: row for row in plan.rows}
    kinds: dict[str, list[str]] = defaultdict(list)
    for doc_id in plan.blind:
        kinds[str(by_id[doc_id].doc_type)].append(doc_id)
    lines.append(f"blind: {len(plan.blind)} document(s)")
    for kind in sorted(kinds):
        lines.append(f"  {kind:<20}{', '.join(kinds[kind])}")
    counts: dict[str, int] = defaultdict(int)
    for row in plan.rows:
        if _hand_labeled(row):
            counts[str(row.label_mode)] += 1
    lines.append(
        "label_mode of hand-labeled documents: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    lines.append(
        f"changes: {len(plan.changes)} cell(s) in {len({c.doc_id for c in plan.changes})} row(s)"
    )
    lines.extend(f"warning: {w}" for w in plan.warnings)
    return "\n".join(lines)
