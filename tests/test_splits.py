import json
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from docextract.ingest.manifest import ManifestRow, check_manifest, read_manifest
from docextract.ingest.splits import (
    MIN_BLIND,
    SplitError,
    Stratum,
    allocate_dev_counts,
    assign_splits,
    format_plan,
)
from ingest_support import run_script, write_manifest_file

TYPES = ["receipt", "faktur", "invoice", "ecommerce", "bill", "hotel_reservation"]


def mk(
    doc_id: str,
    doc_type: str | None = "receipt",
    *,
    group: str | None = "",
    capture: str | None = None,
    split: str | None = None,
    label_mode: str | None = None,
    gt_from: str | None = None,
    variant: str | None = None,
) -> ManifestRow:
    """A row. `group=""` means content_group = doc_id (what manifest_init pre-fills)."""
    derived = variant is not None
    values: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_files": "" if derived else f"{doc_id}.pdf",
        "doc_type": doc_type,
        "content_group": doc_id if group == "" else group,
        "capture": "synthetic_degraded" if derived else capture,
        "split": split,
        "label_mode": label_mode,
        "gt_from": gt_from,
        "variant": variant,
        "needs_redaction": "n",
    }
    return ManifestRow.model_validate(values)


def docs(count: int, doc_type: str = "receipt", prefix: str | None = None) -> list[ManifestRow]:
    name = prefix or doc_type
    return [mk(f"{name}_{i:02d}", doc_type) for i in range(count)]


def by_id(rows: list[ManifestRow]) -> dict[str, ManifestRow]:
    return {row.doc_id: row for row in rows}


def realistic() -> list[ManifestRow]:
    """~25 documents like the real dataset: some types plentiful, one with a single document."""
    sizes = {
        "receipt": 8,
        "faktur": 6,
        "invoice": 3,
        "ecommerce": 3,
        "bill": 1,
        "hotel_reservation": 3,
    }
    return [row for kind, n in sizes.items() for row in docs(n, kind)]


# ------------------------------------------------------------------ allocation


def real(name: str) -> Stratum:
    return ("real", name)


class TestAllocate:
    def test_a_realistic_spread(self) -> None:
        sizes = {real("receipt"): 8, real("faktur"): 6, real("invoice"): 3}
        sizes.update({real("ecommerce"): 3, real("bill"): 1, real("hotel"): 3})
        # 24 groups x 35% = 8.4 -> 8 dev; the largest remainder (receipt, .8) takes the spare one
        assert allocate_dev_counts(sizes) == {
            real("bill"): 0,
            real("ecommerce"): 1,
            real("faktur"): 2,
            real("hotel"): 1,
            real("invoice"): 1,
            real("receipt"): 3,
        }

    def test_a_single_group_goes_to_test_and_two_split_one_one(self) -> None:
        assert allocate_dev_counts({real("a"): 1}) == {real("a"): 0}
        assert allocate_dev_counts({real("a"): 2}) == {real("a"): 1}

    @pytest.mark.parametrize("size", range(2, 31))
    def test_every_stratum_with_two_or_more_groups_has_both_sides(self, size: int) -> None:
        dev = allocate_dev_counts({real("a"): size})[real("a")]
        assert 1 <= dev <= size - 1

    def test_half_up_rounding_of_the_target(self) -> None:
        assert allocate_dev_counts({real("a"): 10})[real("a")] == 4  # 3.5 -> 4
        assert allocate_dev_counts({real("a"): 20})[real("a")] == 7

    def test_equal_strata_get_equal_shares(self) -> None:
        sizes = {real("a"): 20, real("b"): 20, real("c"): 20}
        assert set(allocate_dev_counts(sizes).values()) == {7}

    def test_ties_go_to_the_alphabetically_first_strata(self) -> None:
        sizes = {real(name): 5 for name in "abcd"}  # ideal 1.75 each, target 7
        assert allocate_dev_counts(sizes) == {
            real("a"): 2,
            real("b"): 2,
            real("c"): 2,
            real("d"): 1,
        }

    def test_the_total_tracks_the_fraction(self) -> None:
        rng = random.Random(7)
        for _ in range(100):
            sizes = {real(f"t{i}"): rng.randint(2, 40) for i in range(rng.randint(1, 8))}
            total = sum(sizes.values())
            dev = sum(allocate_dev_counts(sizes).values())
            lower = max(len(sizes), 0)
            assert (
                max(lower, round(0.35 * total) - len(sizes))
                <= dev
                <= round(0.35 * total) + len(sizes)
            )

    def test_when_the_minimums_exceed_the_target_each_gets_one(self) -> None:
        sizes = {real(name): 2 for name in "abcde"}  # 35% of 10 = 3.5, but 5 strata need one each
        assert set(allocate_dev_counts(sizes).values()) == {1}

    @pytest.mark.parametrize(("size", "fraction"), [(2, "0.9"), (3, "0.95"), (10, "0.99")])
    def test_a_high_fraction_still_leaves_one_group_for_test(
        self, size: int, fraction: str
    ) -> None:
        assert allocate_dev_counts({real("a"): size}, fraction) == {real("a"): size - 1}

    @pytest.mark.parametrize(("size", "fraction"), [(2, "0.1"), (3, "0.05"), (10, "0.01")])
    def test_a_low_fraction_still_leaves_one_group_for_dev(self, size: int, fraction: str) -> None:
        assert allocate_dev_counts({real("a"): size}, fraction) == {real("a"): 1}

    def test_all_singletons(self) -> None:
        assert set(allocate_dev_counts({real("a"): 1, real("b"): 1}).values()) == {0}

    def test_order_of_the_input_does_not_matter(self) -> None:
        sizes = {real("a"): 7, real("b"): 3, real("c"): 9, ("constructed", "x"): 3}
        shuffled = dict(reversed(list(sizes.items())))
        assert allocate_dev_counts(sizes) == allocate_dev_counts(shuffled)

    @pytest.mark.parametrize("fraction", ["0.35", 0.35, Fraction(7, 20)])
    def test_the_fraction_can_be_text_float_or_fraction(self, fraction: Any) -> None:
        assert allocate_dev_counts({real("a"): 20}, fraction) == {real("a"): 7}

    @pytest.mark.parametrize("fraction", ["0", "1", "-0.2", "1.5"])
    def test_a_fraction_outside_zero_one_is_refused(self, fraction: str) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            allocate_dev_counts({real("a"): 4}, fraction)


# ------------------------------------------------------------------ assignment


def straddles(rows: list[ManifestRow]) -> list[str]:
    return [p for p in check_manifest(rows) if "straddles" in p]


class TestAssign:
    def test_everything_gets_a_split_in_roughly_the_wanted_ratio(self) -> None:
        plan = assign_splits(realistic())
        splits = Counter(row.split for row in plan.rows)
        assert set(splits) == {"dev", "test"}
        assert splits["dev"] == 8 and splits["test"] == 16
        summary = {s.stratum: (s.dev, s.test) for s in plan.strata}
        assert summary[("real", "bill")] == (0, 1)  # the single bill goes to test
        assert all(
            dev >= 1 and test >= 1
            for (kind, name), (dev, test) in summary.items()
            if name != "bill"
        )

    def test_no_content_group_straddles_the_split_over_many_random_manifests(self) -> None:
        for seed in range(60):
            rng = random.Random(seed)
            rows: list[ManifestRow] = []
            for i in range(rng.randint(4, 50)):
                kind = rng.choice(TYPES)
                doc_id = f"d{i:03d}"
                leader = rng.choice(
                    [r for r in rows if r.doc_type == kind and r.gt_from is None] or [None]
                )
                roll = rng.random()
                if leader is not None and roll < 0.2:  # a second photo of the same content
                    rows.append(mk(doc_id, kind, group=leader.content_group, gt_from=leader.doc_id))
                elif leader is not None and roll < 0.4:  # a degraded variant
                    rows.append(
                        mk(
                            doc_id,
                            kind,
                            group=leader.content_group,
                            gt_from=leader.doc_id,
                            variant="light",
                        )
                    )
                elif (
                    leader is not None and roll < 0.5
                ):  # a second hand-labeled photo of one content
                    rows.append(mk(doc_id, kind, group=leader.content_group))
                else:
                    rows.append(mk(doc_id, kind))
            plan = assign_splits(rows, seed=str(seed))
            assert straddles(plan.rows) == [], seed
            splits: dict[str, set[str | None]] = defaultdict(set)
            for row in plan.rows:
                splits[str(row.content_group)].add(row.split)
            assert all(len(found) == 1 and None not in found for found in splits.values()), seed

    def test_variants_and_duplicates_follow_their_source(self) -> None:
        rows = docs(8)
        rows.append(
            mk("receipt_00-deg-light", group="receipt_00", gt_from="receipt_00", variant="light")
        )
        rows.append(
            mk("receipt_00-deg-heavy", group="receipt_00", gt_from="receipt_00", variant="heavy")
        )
        rows.append(mk("receipt_01_b", group="receipt_01", gt_from="receipt_01"))
        result = by_id(assign_splits(rows).rows)
        for name in ("receipt_00-deg-light", "receipt_00-deg-heavy"):
            assert result[name].split == result["receipt_00"].split
        assert result["receipt_01_b"].split == result["receipt_01"].split

    def test_a_variant_counts_as_part_of_its_group_not_as_a_group(self) -> None:
        plan = assign_splits(
            [*docs(4), mk("v", group="receipt_00", gt_from="receipt_00", variant="light")]
        )
        assert sum(s.groups for s in plan.strata) == 4

    def test_a_variant_without_its_own_content_group_takes_its_sources(self) -> None:
        rows = [*docs(6), mk("v", group=None, gt_from="receipt_00", variant="light")]
        result = by_id(assign_splits(rows).rows)
        assert result["v"].split == result["receipt_00"].split

    def test_constructed_documents_are_a_separate_stratum(self) -> None:
        rows = [
            *docs(4, "invoice", "real_inv"),
            *[mk(f"c{i}", "invoice", capture="constructed") for i in range(3)],
        ]
        plan = assign_splits(rows)
        summary = {s.stratum: (s.groups, s.dev, s.test) for s in plan.strata}
        assert summary[("real", "invoice")][0] == 4
        assert summary[("constructed", "invoice")] == (3, 1, 2)

    def test_it_is_deterministic_and_the_seed_matters(self) -> None:
        rows = docs(30)
        first = [r.split for r in assign_splits(rows, seed="a").rows]
        assert first == [r.split for r in assign_splits(rows, seed="a").rows]
        assert first != [r.split for r in assign_splits(rows, seed="b").rows]

    def test_the_default_seed_gives_this_exact_split_forever(self) -> None:
        # A golden value: the ranking is part of the dataset's definition. If this fails, the
        # hash scheme changed, and splits already in the manifest no longer match a re-run.
        plan = assign_splits(docs(12))
        assert sorted(r.doc_id for r in plan.rows if r.split == "dev") == [
            "receipt_01",
            "receipt_02",
            "receipt_06",
            "receipt_09",
        ]

    def test_row_order_does_not_change_who_is_where(self) -> None:
        rows = realistic()
        shuffled = rows[:]
        random.Random(3).shuffle(shuffled)
        a = {r.doc_id: r.split for r in assign_splits(rows).rows}
        b = {r.doc_id: r.split for r in assign_splits(shuffled).rows}
        assert a == b

    def test_running_it_again_changes_nothing(self) -> None:
        first = assign_splits(realistic())
        second = assign_splits(first.rows)
        assert second.changes == []
        assert second.rows == first.rows

    def test_existing_assignments_never_move_and_new_rows_fill_in(self) -> None:
        first = assign_splits(docs(10))
        grown = [*first.rows, *docs(10, prefix="more")]
        second = assign_splits(grown)
        before = {r.doc_id: r.split for r in first.rows}
        after = {r.doc_id: r.split for r in second.rows}
        assert all(after[d] == s for d, s in before.items())
        dev = sum(1 for s in after.values() if s == "dev")
        assert dev == 7  # 20 groups x 35%: the new rows were assigned so the total still lands
        assert {c.doc_id for c in second.changes if c.field == "split"} == {
            r.doc_id for r in docs(10, prefix="more")
        }

    def test_a_late_second_document_of_a_type_goes_to_dev(self) -> None:
        first = assign_splits([*docs(5), mk("only_bill", "bill")])
        assert by_id(first.rows)["only_bill"].split == "test"
        second = assign_splits([*first.rows, mk("bill_2", "bill")])
        assert by_id(second.rows)["bill_2"].split == "dev"

    def test_a_hand_set_split_is_respected(self) -> None:
        rows = [*docs(9), mk("pinned", "receipt", split="dev")]
        result = by_id(assign_splits(rows).rows)
        assert result["pinned"].split == "dev"

    def test_force_starts_from_scratch(self) -> None:
        rows = docs(40)
        first = assign_splits(rows, seed="a").rows
        assert assign_splits(first, seed="b").changes == []  # no force: nothing moves
        forced = assign_splits(first, seed="b", force=True)
        assert any(c.field == "split" for c in forced.changes)
        assert Counter(r.split for r in forced.rows) == Counter(r.split for r in first)

    def test_nothing_is_mutated(self) -> None:
        rows = realistic()
        snapshot = [r.model_copy() for r in rows]
        assign_splits(rows)
        assert rows == snapshot

    def test_other_columns_are_untouched(self) -> None:
        plan = assign_splits(realistic())
        for before, after in zip(realistic(), plan.rows, strict=True):
            assert before.model_copy(
                update={"split": None, "label_mode": None}
            ) == after.model_copy(update={"split": None, "label_mode": None})


class TestAssignRefusals:
    def test_every_problem_is_listed_at_once(self) -> None:
        rows = [
            mk("a", None),
            mk("b", "receipt", group=None),
            mk("c", "faktur", group="g"),
            mk("d", "receipt", group="g"),
            mk("e", "faktur", group="h", capture="constructed"),
            mk("f", "faktur", group="h"),
        ]
        with pytest.raises(SplitError) as caught:
            assign_splits(rows)
        text = "\n".join(caught.value.problems)
        assert "b: no content_group" in text
        assert "content_group 'a': no doc_type" in text
        assert "content_group 'g' mixes doc types ['faktur', 'receipt']" in text
        assert "content_group 'h' mixes constructed and real" in text
        assert len(caught.value.problems) == 4

    def test_a_group_that_already_straddles_is_reported_not_papered_over(self) -> None:
        rows = [mk("a", split="dev", group="g"), mk("b", split="test", group="g")]
        with pytest.raises(SplitError, match="straddles splits"):
            assign_splits(rows)

    def test_rows_that_are_not_a_valid_manifest_are_refused(self) -> None:
        with pytest.raises(SplitError, match="doc_id 'a' appears 2 times"):
            assign_splits([mk("a"), mk("a")])
        with pytest.raises(SplitError, match="gt_from 'nowhere' is not in the manifest"):
            assign_splits([mk("a"), mk("b", gt_from="nowhere", group="a")])

    def test_a_locked_document_cannot_get_a_new_label_mode_either(self) -> None:
        rows = [*docs(9), mk("labeled", "receipt", split="test")]  # labeled, blank label_mode
        with pytest.raises(SplitError, match=r"labeled: has ground truth.*test/None -> test/"):
            assign_splits(rows, locked={"labeled"})

    def test_a_locked_document_with_no_split_yet_may_be_given_its_first(self) -> None:
        # a constructed invoice is labeled when it is generated, long before it is split:
        # the label does not record the split, so assigning one is not a change to the label
        constructed = [mk(f"cons_{i}", "invoice", capture="constructed") for i in range(3)]
        locked = {row.doc_id for row in constructed}
        plan = assign_splits(constructed, locked=locked)
        assert all(r.split in ("dev", "test") for r in plan.rows)
        assert all(r.label_mode is None for r in plan.rows)  # and nobody labels them
        assert assign_splits(plan.rows, locked=locked).changes == []  # settled: a re-run is quiet

    def test_but_the_first_split_does_not_open_the_door_to_moving_it(self) -> None:
        constructed = [mk(f"cons_{i}", "invoice", capture="constructed") for i in range(6)]
        locked = {row.doc_id for row in constructed}
        first = assign_splits(constructed, locked=locked).rows
        with pytest.raises(
            SplitError, match="has ground truth, so its split/label_mode cannot change"
        ):
            assign_splits(first, locked=locked, seed="another seed", force=True)

    def test_locked_documents_may_stay_but_not_move(self) -> None:
        rows = docs(20, prefix="r")
        first = assign_splits(rows).rows
        # re-running with everything locked changes nothing and is fine
        assert assign_splits(first, locked={r.doc_id for r in first}).changes == []
        with pytest.raises(
            SplitError, match="has ground truth, so its split/label_mode cannot change"
        ):
            assign_splits(first, seed="other", force=True, locked={r.doc_id for r in first})

    def test_locking_one_document_blocks_only_changes_to_it(self) -> None:
        first = assign_splits(docs(30)).rows
        forced = assign_splits(first, seed="other", force=True)
        moved = [c.doc_id for c in forced.changes if c.field == "split"]
        assert moved
        with pytest.raises(SplitError) as caught:
            assign_splits(first, seed="other", force=True, locked={moved[0]})
        assert len(caught.value.problems) == 1
        assert caught.value.problems[0].startswith(moved[0])


# ------------------------------------------------------------------ blind selection


def hand_rows(plan_rows: list[ManifestRow]) -> list[ManifestRow]:
    return [
        r
        for r in plan_rows
        if not r.is_derived and r.gt_from is None and r.capture != "constructed"
    ]


def plentiful() -> list[ManifestRow]:
    return [row for kind in TYPES[:4] for row in docs(12, kind)]


class TestBlind:
    def test_blind_documents_are_real_hand_labeled_and_in_test(self) -> None:
        rows = [*plentiful(), *[mk(f"c{i}", "invoice", capture="constructed") for i in range(4)]]
        rows.append(mk("v", group="receipt_00", gt_from="receipt_00", variant="light"))
        rows.append(mk("dup", "faktur", group="faktur_00", gt_from="faktur_00"))
        plan = assign_splits(rows, blind_count=10)
        result = by_id(plan.rows)
        assert len(plan.blind) == 10
        for doc_id in plan.blind:
            row = result[doc_id]
            assert row.split == "test"
            assert row.capture != "constructed" and not row.is_derived and row.gt_from is None

    def test_blind_documents_are_spread_over_doc_types(self) -> None:
        plan = assign_splits(plentiful(), blind_count=8)
        counts = Counter(by_id(plan.rows)[d].doc_type for d in plan.blind)
        assert counts == {"receipt": 2, "faktur": 2, "invoice": 2, "ecommerce": 2}

    def test_ten_blind_over_four_types_is_three_three_two_two(self) -> None:
        plan = assign_splits(plentiful(), blind_count=10)
        counts = sorted(Counter(by_id(plan.rows)[d].doc_type for d in plan.blind).values())
        assert counts == [2, 2, 3, 3]

    def test_a_type_that_is_scarce_gives_what_it_has_and_the_others_make_up(self) -> None:
        rows = [*docs(12, "receipt"), *docs(12, "faktur"), *docs(2, "bill")]
        plan = assign_splits(rows, blind_count=9)
        counts = Counter(by_id(plan.rows)[d].doc_type for d in plan.blind)
        assert len(plan.blind) == 9
        assert counts["bill"] <= 1  # two bills: one dev, one test
        assert counts["receipt"] >= 4 and counts["faktur"] >= 4

    def test_everyone_else_who_is_hand_labeled_defaults_to_prelabel(self) -> None:
        rows = [*plentiful(), *[mk(f"c{i}", "invoice", capture="constructed") for i in range(3)]]
        rows.append(mk("v", group="receipt_00", gt_from="receipt_00", variant="light"))
        plan = assign_splits(rows, blind_count=8)
        result = by_id(plan.rows)
        modes = Counter(r.label_mode for r in hand_rows(plan.rows))
        assert modes == {"blind": 8, "prelabel": len(hand_rows(plan.rows)) - 8}
        assert result["v"].label_mode is None
        assert all(result[f"c{i}"].label_mode is None for i in range(3))

    def test_existing_choices_are_kept_and_count_toward_the_target(self) -> None:
        first = assign_splits(plentiful(), blind_count=4)
        kept = set(first.blind)
        newcomers = docs(12, "receipt", prefix="late")
        grown = assign_splits([*first.rows, *newcomers], blind_count=7)
        assert kept <= set(grown.blind)
        assert len(grown.blind) == 7
        # the prelabel this script wrote earlier counts as a decision: only new rows are picked
        assert set(grown.blind) - kept <= {r.doc_id for r in newcomers}

    def test_raising_the_count_alone_does_not_flip_documents_already_marked_prelabel(self) -> None:
        first = assign_splits(plentiful(), blind_count=4)
        again = assign_splits(first.rows, blind_count=9)
        assert again.blind == first.blind
        assert any("only 4 blind" in w for w in again.warnings)

    def test_an_explicit_prelabel_is_never_overridden_and_never_blinded(self) -> None:
        rows = [mk(f"r{i}", "receipt", label_mode="prelabel") for i in range(10)]
        plan = assign_splits(rows, blind_count=5)
        assert plan.blind == []
        assert {r.label_mode for r in plan.rows} == {"prelabel"}
        assert any("only 0 blind" in w for w in plan.warnings)

    def test_whole_content_groups_are_blind_together(self) -> None:
        rows = docs(14, "receipt")
        for i in range(4):  # four contents, each photographed and labeled twice
            rows.append(mk(f"twin_{i}_b", "faktur", group=f"twin_{i}"))
            rows.append(mk(f"twin_{i}", "faktur", group=f"twin_{i}"))
        plan = assign_splits(rows, blind_count=6)
        result = by_id(plan.rows)
        for i in range(4):
            assert result[f"twin_{i}"].label_mode == result[f"twin_{i}_b"].label_mode

    def test_a_partly_marked_blind_group_is_completed(self) -> None:
        rows = [
            mk("p", "faktur", group="g", split="test", label_mode="blind"),
            mk("q", "faktur", group="g", split="test"),  # the same content, not yet marked
            *docs(10, "receipt"),
        ]
        result = by_id(assign_splits(rows, blind_count=3).rows)
        assert result["p"].label_mode == result["q"].label_mode == "blind"

    def test_a_group_that_does_not_fit_the_remaining_quota_is_skipped(self) -> None:
        rows = [mk(f"x{i}", "receipt", group="big") for i in range(3)]  # 3 rows, one group
        rows += [mk(f"s{i}", "faktur") for i in range(8)]
        plan = assign_splits(rows, blind_count=2)
        assert len(plan.blind) == 2
        assert not any(d.startswith("x") for d in plan.blind)

    def test_fewer_candidates_than_wanted_marks_them_all_and_warns(self) -> None:
        plan = assign_splits(docs(5), blind_count=10)
        assert 0 < len(plan.blind) < MIN_BLIND
        assert any(f"only {len(plan.blind)} blind" in w for w in plan.warnings)

    def test_blind_count_zero_blinds_nobody(self) -> None:
        plan = assign_splits(plentiful(), blind_count=0)
        assert plan.blind == []
        assert {r.label_mode for r in plan.rows} == {"prelabel"}
        assert plan.warnings == []

    def test_a_blind_document_in_dev_keeps_its_mode_and_is_reported(self) -> None:
        rows = [mk("odd", "receipt", split="dev", label_mode="blind"), *docs(9)]
        plan = assign_splits(rows, blind_count=0)
        assert by_id(plan.rows)["odd"].label_mode == "blind"
        assert any("odd: blind but in the dev split" in w for w in plan.warnings)

    def test_force_chooses_again(self) -> None:
        first = assign_splits(plentiful(), blind_count=4, seed="a")
        again = assign_splits(first.rows, blind_count=4, seed="b", force=True)
        assert len(again.blind) == 4
        assert set(again.blind) != set(first.blind)

    def test_a_dev_document_is_never_picked(self) -> None:
        plan = assign_splits(plentiful(), blind_count=10)
        assert all(by_id(plan.rows)[d].split == "test" for d in plan.blind)

    def test_the_selection_is_deterministic(self) -> None:
        a = assign_splits(plentiful(), seed="s")
        b = assign_splits(plentiful(), seed="s")
        assert a.blind == b.blind


class TestReport:
    def test_the_report_lists_strata_blind_documents_and_changes(self) -> None:
        plan = assign_splits(plentiful(), blind_count=8)
        text = format_plan(plan, seed="s", dev_fraction="0.35")
        assert "48 content group(s) -> 17 dev / 31 test (wanted 35% dev, seed 's')" in text
        assert "real / receipt" in text
        assert "blind: 8 document(s)" in text
        assert "label_mode of hand-labeled documents: blind 8, prelabel 40" in text
        assert f"changes: {len(plan.changes)} cell(s)" in text

    def test_a_single_group_stratum_is_called_out(self) -> None:
        text = format_plan(
            assign_splits([*docs(5), mk("b", "bill")]), seed="s", dev_fraction="0.35"
        )
        assert "(single group: test)" in text


# ------------------------------------------------------------------ the command line


@pytest.mark.slow
class TestCommandLine:
    def manifest(self, tmp_path: Path, rows: list[ManifestRow]) -> Path:
        return write_manifest_file(tmp_path, rows)

    def run(self, manifest: Path, tmp_path: Path, *extra: str) -> tuple[int, str, str]:
        return run_script(
            "assign_split.py",
            "--manifest",
            str(manifest),
            "--ground-truth-dir",
            str(tmp_path / "gt"),
            *extra,
        )

    def test_dry_run_writes_nothing_and_a_real_run_is_idempotent(self, tmp_path: Path) -> None:
        manifest = self.manifest(tmp_path, plentiful())
        before = manifest.read_bytes()
        code, out, _ = self.run(manifest, tmp_path, "--dry-run")
        assert code == 0
        assert "dry run: nothing written" in out
        assert manifest.read_bytes() == before
        code, out, _ = self.run(manifest, tmp_path)
        assert code == 0
        assert "blind: 10 document(s)" in out
        rows = read_manifest(manifest)
        assert Counter(r.split for r in rows) == {"dev": 17, "test": 31}
        after = manifest.read_bytes()
        code, out, _ = self.run(manifest, tmp_path)
        assert code == 0
        assert "changes: 0 cell(s)" in out
        assert manifest.read_bytes() == after

    def test_an_incomplete_manifest_is_refused_with_every_reason(self, tmp_path: Path) -> None:
        manifest = self.manifest(tmp_path, [mk("a", None), mk("b", None)])
        before = manifest.read_bytes()
        code, _, err = self.run(manifest, tmp_path)
        assert code == 1
        assert err.count("no doc_type") == 2
        assert manifest.read_bytes() == before

    def test_an_unreadable_manifest_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "m.csv"
        bad.write_text("nonsense\n", encoding="utf-8")
        code, _, err = self.run(bad, tmp_path)
        assert code == 2
        assert "cannot be read" in err

    def test_force_is_blocked_for_documents_that_have_ground_truth(self, tmp_path: Path) -> None:
        manifest = self.manifest(tmp_path, docs(30))
        assert self.run(manifest, tmp_path)[0] == 0
        gt = tmp_path / "gt"
        gt.mkdir()
        for row in read_manifest(manifest):
            (gt / f"{row.doc_id}.json").write_text(
                json.dumps({"doc_id": row.doc_id}), encoding="utf-8"
            )
        before = manifest.read_bytes()
        code, _, err = self.run(manifest, tmp_path, "--force", "--seed", "another")
        assert code == 1
        assert "has ground truth" in err
        assert manifest.read_bytes() == before

    def test_a_bad_fraction_exits_1(self, tmp_path: Path) -> None:
        manifest = self.manifest(tmp_path, docs(6))
        code, _, err = self.run(manifest, tmp_path, "--dev-fraction", "1.5")
        assert code == 1
        assert "between 0 and 1" in err
