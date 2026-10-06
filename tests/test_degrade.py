import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from docextract.ingest import degrade
from docextract.ingest.degrade import (
    LEVELS,
    SIDECAR,
    DegradeError,
    add_variant_rows,
    degrade_document,
    degrade_image,
    degrade_opencv,
    derive_seed,
    fit_in_canvas,
    page_homography,
    resolve_backend,
    seed_from_digest,
    variant_id,
    variant_row,
)
from docextract.ingest.manifest import ManifestRow, check_manifest, read_manifest
from docextract.ingest.splits import assign_splits
from files import A4, make_jpeg, make_sized_pdf
from ingest_support import Env, run_script, settings, write_manifest_file

FONT = ImageFont.load_default(size=20)  # ~ the size of 11 pt text at 200 dpi


def text_page(width: int = 900, height: int = 1200) -> Image.Image:
    """A page of text with a frame 20 px from every edge (so that clipping would show)."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, width - 21, height - 21], outline=(0, 0, 0), width=3)
    for i in range(36):
        text = f"{i + 1:02d} Kertas A4 80gr Sinar Dunia  10  45.000,00  450.000,00"
        draw.text((50, 60 + i * 30), text, fill=(10, 10, 10), font=FONT)
    return image


def density(image: Image.Image, cell: int = 32) -> tuple[Any, float]:
    """Where the ink is, as a coarse map (tolerates a few pixels of shift), and how much of it."""
    grey = np.asarray(image.convert("L"), dtype=float)
    ink = (grey < (np.median(grey) + np.percentile(grey, 0.5)) / 2).astype(float)
    h, w = (ink.shape[0] // cell) * cell, (ink.shape[1] // cell) * cell
    cells = ink[:h, :w].reshape(h // cell, cell, w // cell, cell).mean(axis=(1, 3))
    return cells, float(ink.mean())


def similarity(a: Image.Image, b: Image.Image) -> float:
    return float(np.corrcoef(density(a)[0].ravel(), density(b)[0].ravel())[0, 1])


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)).mean())


def source_row(**overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": "doc_001",
        "raw_files": "a.pdf",
        "doc_type": "faktur",
        "capture": "born_digital",
        "content_group": "g1",
        "needs_redaction": "n",
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


# ------------------------------------------------------------------ identity


class TestIdentity:
    def test_variant_ids(self) -> None:
        assert variant_id("doc_007", "light") == "doc_007-deg-light"

    def test_a_seed_depends_on_every_argument_and_on_nothing_else(self) -> None:
        base = derive_seed("m", "doc_1", "light", 1)
        assert base == derive_seed("m", "doc_1", "light", 1)
        others = {
            derive_seed("n", "doc_1", "light", 1),
            derive_seed("m", "doc_2", "light", 1),
            derive_seed("m", "doc_1", "heavy", 1),
            derive_seed("m", "doc_1", "light", 2),
        }
        assert base not in others
        assert len(others) == 4

    @pytest.mark.parametrize(
        ("digest_hex", "seed"),
        [
            ("00000000", 1),  # a zero hash must not become the 'no seed' seed 0
            ("00000001", 2),
            ("7ffffffe", 2**31 - 1),
            ("7fffffff", 1),  # 2**31 - 1 wraps around to 0, then +1
            ("ffffffff", 2),  # 4294967295 = 2 * (2**31 - 1) + 1
            ("ffffffff" + "01" * 28, 2),  # only the first four bytes count
        ],
    )
    def test_seed_from_digest_edges(self, digest_hex: str, seed: int) -> None:
        assert seed_from_digest(bytes.fromhex(digest_hex)) == seed

    def test_seeds_are_never_zero_and_always_fit_a_signed_c_int(self) -> None:
        # Augraphy gives the seed to cv2.setRNGSeed (a signed 32-bit int) and treats 0 as "unset"
        seeds = [derive_seed("m", f"doc_{i}", "light", p) for i in range(2000) for p in (1, 2)]
        assert min(seeds) >= 1 and max(seeds) <= 2**31 - 1

    def test_the_variant_row_inherits_the_facts_and_borrows_the_labels(self) -> None:
        source = source_row(
            template_group="t1",
            language="id",
            split="test",
            label_mode="blind",
            source_url="http://x",
        )
        row = variant_row(source, "medium")
        assert row.doc_id == "doc_001-deg-medium"
        assert (row.doc_type, row.template_group, row.language) == ("faktur", "t1", "id")
        assert (row.content_group, row.split) == ("g1", "test")
        assert (row.capture, row.variant, row.gt_from) == (
            "synthetic_degraded",
            "medium",
            "doc_001",
        )
        assert row.raw_files == () and row.is_derived
        assert row.needs_redaction is False  # its pages come from already-redacted pages
        assert row.label_mode is None and row.source_url is None
        assert "doc_001" in (row.license_note or "")

    def test_a_variant_of_a_duplicate_points_at_the_owner_of_the_labels(self) -> None:
        row = variant_row(source_row(doc_id="doc_002", gt_from="doc_001"), "light")
        assert row.gt_from == "doc_001"

    def test_the_content_group_defaults_to_the_source_itself(self) -> None:
        assert variant_row(source_row(content_group=""), "light").content_group == "doc_001"

    def test_a_variant_cannot_be_degraded_again(self) -> None:
        with pytest.raises(ValueError, match="itself a variant"):
            variant_row(variant_row(source_row(), "light"), "heavy")


# ------------------------------------------------------------------ geometry


class TestGeometry:
    @pytest.mark.parametrize("size", [(1654, 2339), (300, 200), (101, 977)])
    @pytest.mark.parametrize("level", LEVELS)
    def test_the_whole_page_always_stays_inside_the_canvas(
        self, size: tuple[int, int], level: str
    ) -> None:
        import cv2

        recipe = degrade.CV_RECIPES[level]  # type: ignore[index]
        corners = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])
        for seed in range(150):
            homography = page_homography(size, recipe, np.random.default_rng(seed))
            moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
            assert moved.min() >= -1e-3, (seed, moved)
            assert moved[:, 0].max() <= size[0] + 1e-3, (seed, moved)
            assert moved[:, 1].max() <= size[1] + 1e-3, (seed, moved)

    def test_fit_in_canvas_scales_down_only_when_needed(self) -> None:
        import cv2

        size = (400, 300)
        corners = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])
        shift = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], dtype=np.float64)  # fits: no scaling
        fitted = fit_in_canvas(corners, shift, size)
        moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), fitted).reshape(-1, 2)
        assert np.allclose(moved, corners, atol=1e-3)  # recentred, same scale
        grow = np.array([[1.25, 0, 0], [0, 1.25, 0], [0, 0, 1]], dtype=np.float64)
        moved = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), fit_in_canvas(corners, grow, size)
        ).reshape(-1, 2)
        assert np.allclose(moved, corners, atol=1e-3)  # scaled back down to fill the canvas

    def test_fit_in_canvas_never_scales_a_small_page_up(self) -> None:
        import cv2

        size = (400, 300)
        corners = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])
        half = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float64)
        fitted = fit_in_canvas(corners, half, size)
        moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), fitted).reshape(-1, 2)
        assert moved[:, 0].min() == pytest.approx(100, abs=1e-3)  # half size, centred
        assert moved[:, 0].max() == pytest.approx(300, abs=1e-3)
        assert moved[:, 1].min() == pytest.approx(75, abs=1e-3)
        assert moved[:, 1].max() == pytest.approx(225, abs=1e-3)

    def test_the_workdir_is_restored_even_when_augraphy_fails(self) -> None:
        before = os.getcwd()
        with pytest.raises(RuntimeError, match="boom"), degrade._augraphy_workdir():
            assert Path(os.getcwd()).parts[-3:] == ("outputs", "cache", "augraphy")
            raise RuntimeError("boom")
        assert os.getcwd() == before

    def test_a_heavy_variant_keeps_the_whole_frame_inside_the_canvas(self) -> None:
        # the frame is 20 px from every edge: a page that were clipped would run it off the canvas
        page = text_page()
        for seed in range(8):
            grey = np.asarray(degrade_opencv(page, "heavy", seed).convert("L"))
            strips = {
                "top": grey[:3, :],
                "bottom": grey[-3:, :],
                "left": grey[:, :3],
                "right": grey[:, -3:],
            }
            for side, strip in strips.items():
                assert strip.min() > 150, (seed, side)
            ink = np.argwhere(grey < 120)  # the frame is the outermost ink
            assert ink.min() >= 3 and ink[:, 0].max() <= grey.shape[0] - 4, seed
            assert ink[:, 1].max() <= grey.shape[1] - 4, seed


# ------------------------------------------------------------------ the OpenCV backend


class TestOpenCv:
    @pytest.mark.parametrize("size", [(900, 1200), (301, 199), (50, 50)])
    @pytest.mark.parametrize("level", LEVELS)
    def test_size_and_mode_are_preserved(self, size: tuple[int, int], level: str) -> None:
        out = degrade_image(Image.new("RGB", size, "white"), level, 1, "opencv")  # type: ignore[arg-type]
        assert out.size == size
        assert out.mode == "RGB"

    def test_it_is_deterministic_and_the_seed_matters(self) -> None:
        page = text_page(450, 600)
        first = degrade_image(page, "medium", 5, "opencv")
        assert first.tobytes() == degrade_image(page, "medium", 5, "opencv").tobytes()
        assert first.tobytes() != degrade_image(page, "medium", 6, "opencv").tobytes()

    def test_the_input_is_not_modified(self) -> None:
        page = text_page(450, 600)
        before = page.tobytes()
        degrade_image(page, "heavy", 1, "opencv")
        assert page.tobytes() == before

    def test_severity_grows_with_the_level(self) -> None:
        page = text_page()
        means: dict[str, tuple[float, float]] = {}
        for level in LEVELS:
            outs = [degrade_image(page, level, seed, "opencv") for seed in range(5)]
            means[level] = (
                float(np.mean([similarity(page, o) for o in outs])),
                float(np.mean([mean_abs_diff(page, o) for o in outs])),
            )
        assert means["light"][0] > means["medium"][0] > means["heavy"][0]
        assert means["light"][1] < means["medium"][1] < means["heavy"][1]

    @pytest.mark.parametrize(
        ("level", "minimum"), [("light", 0.9), ("medium", 0.7), ("heavy", 0.5)]
    )
    def test_it_is_still_recognizably_the_same_page(self, level: str, minimum: float) -> None:
        page = text_page()
        ink = density(page)[1]
        for seed in range(6):
            out = degrade_image(page, level, seed, "opencv")  # type: ignore[arg-type]
            assert similarity(page, out) > minimum, (level, seed)
            assert 0.7 < density(out)[1] / ink < 1.4, (level, seed)  # nothing erased, nothing added

    def test_each_level_adds_shadow_and_noise_to_a_blank_page(self) -> None:
        blank = Image.new("RGB", (600, 800), "white")
        stats = {}
        for level in LEVELS:
            grey = np.asarray(degrade_image(blank, level, 4, "opencv").convert("L"), dtype=float)  # type: ignore[arg-type]
            inner = grey[150:-150, 150:-150]  # away from the white border the warp may leave
            stats[level] = (float(inner.mean()), float(inner.std()))
        assert stats["light"][0] > stats["medium"][0] > stats["heavy"][0]  # darker and darker
        assert stats["light"][1] < stats["medium"][1] < stats["heavy"][1]  # noisier and noisier
        assert stats["light"][1] > 1.0 and stats["heavy"][0] > 140  # visible, but still paper

    def test_other_modes_are_accepted(self) -> None:
        for mode in ("L", "RGBA", "P"):
            image = Image.new(mode, (80, 60), 255 if mode == "L" else None)
            assert degrade_image(image, "light", 1, "opencv").size == (80, 60)

    def test_a_backend_that_changes_the_size_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            degrade, "degrade_opencv", lambda image, level, seed: image.resize((10, 10))
        )
        with pytest.raises(RuntimeError, match=r"changed the page size \(80, 60\) -> \(10, 10\)"):
            degrade_image(Image.new("RGB", (80, 60), "white"), "light", 1, "opencv")


def neutral(**changes: Any) -> degrade.CvRecipe:
    """A recipe with every effect off (a blur of 0.01 px is as close to none as OpenCV allows)."""
    values: dict[str, Any] = {
        "rotation_deg": 0.0,
        "perspective": 0.0,
        "shadow": 0.0,
        "contrast": 1.0,
        "blur_sigma": 0.01,
        "noise_sigma": 0.0,
        "jpeg_quality": 100,
    }
    return degrade.CvRecipe(**{**values, **changes})


class TestEachEffect:
    """Every parameter of a recipe is shown to do its job, on a recipe where it is the only one."""

    @pytest.fixture
    def only(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        def use(**changes: Any) -> None:
            monkeypatch.setitem(degrade.CV_RECIPES, "light", neutral(**changes))

        return use

    @staticmethod
    def grey(image: Image.Image) -> Any:
        return np.asarray(image.convert("L"), dtype=float)

    def flat(self, value: int = 128) -> Image.Image:
        return Image.new("RGB", (400, 300), (value,) * 3)

    def test_contrast_pulls_the_page_towards_grey(self, only: Any) -> None:
        only(contrast=1.0)
        assert self.grey(degrade_opencv(self.flat(200), "light", 1)).mean() == pytest.approx(
            200, abs=7
        )
        only(contrast=0.5)  # (200 - 128) * 0.5 + 128 = 164, give or take the brightness jitter
        assert self.grey(degrade_opencv(self.flat(200), "light", 1)).mean() == pytest.approx(
            164, abs=7
        )

    def test_shadow_is_a_gradient_across_the_page(self, only: Any) -> None:
        only(shadow=0.0)
        flat = self.grey(degrade_opencv(self.flat(220), "light", 1))
        assert flat.max() - flat.min() < 5
        only(shadow=0.5)
        shaded = self.grey(degrade_opencv(self.flat(220), "light", 1))
        assert shaded.max() - shaded.min() > 80 and shaded.std() > 20

    def test_noise_has_the_requested_strength(self, only: Any) -> None:
        # noise is drawn independently per channel, so the luma std is
        # sqrt(.299^2 + .587^2 + .114^2) = 0.67 of sigma
        only(noise_sigma=0.0)
        assert self.grey(degrade_opencv(self.flat(), "light", 1)).std() < 1
        for sigma in (5.0, 10.0):
            only(noise_sigma=sigma)
            std = self.grey(degrade_opencv(self.flat(), "light", 1)).std()
            assert 0.55 * sigma < std < 0.8 * sigma, (sigma, std)

    def test_blur_softens_an_edge(self, only: Any) -> None:
        edge = Image.new("RGB", (400, 300), "black")
        edge.paste((255, 255, 255), (200, 0, 400, 300))

        def transition(image: Image.Image) -> int:
            row = self.grey(image)[150]
            return int(((row > 25) & (row < 230)).sum())

        only(blur_sigma=0.01)
        assert transition(degrade_opencv(edge, "light", 1)) <= 2
        only(blur_sigma=4.0)
        assert transition(degrade_opencv(edge, "light", 1)) > 8

    def test_jpeg_quality_adds_compression_artefacts(self, only: Any) -> None:
        page = text_page(450, 600)
        only(jpeg_quality=100)
        clean = degrade_opencv(page, "light", 1)
        only(jpeg_quality=10)
        crushed = degrade_opencv(page, "light", 1)
        assert crushed.tobytes() != clean.tobytes()
        assert mean_abs_diff(page, crushed) > mean_abs_diff(page, clean) + 1

    def test_rotation_tilts_the_page(self, only: Any) -> None:
        page = Image.new("RGB", (800, 300), "white")
        ImageDraw.Draw(page).line([(0, 150), (800, 150)], fill=(0, 0, 0), width=3)

        def tilt(image: Image.Image) -> float:
            dark = self.grey(image) < 128
            ys = np.arange(dark.shape[0])[:, None] * np.ones(dark.shape[1])[None, :]
            left, right = dark[:, :200], dark[:, -200:]
            return float(ys[:, -200:][right].mean() - ys[:, :200][left].mean())

        only(rotation_deg=0.0)
        assert all(abs(tilt(degrade_opencv(page, "light", s))) < 1 for s in range(4))
        only(rotation_deg=3.0)  # tan(3 deg) * 600 px between the two sides ~ 31 px at most
        tilts = [abs(tilt(degrade_opencv(page, "light", s))) for s in range(6)]
        assert max(tilts) > 8 and max(tilts) < 40

    def test_perspective_moves_the_corners(self, only: Any) -> None:
        page = text_page(450, 600)
        only(perspective=0.0)
        flat = degrade_opencv(page, "light", 2)
        only(perspective=0.04)
        warped = degrade_opencv(page, "light", 2)
        assert mean_abs_diff(flat, warped) > 3


class TestBackendChoice:
    def test_opencv_can_be_requested(self) -> None:
        backend, note = resolve_backend("opencv")
        assert backend == "opencv" and "requested" in note

    def test_auto_falls_back_to_opencv_when_augraphy_is_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(__import__("sys").modules, "augraphy", None)  # import raises
        backend, note = resolve_backend("auto")
        assert backend == "opencv"
        assert "not usable" in note

    def test_asking_for_augraphy_when_it_is_unusable_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(__import__("sys").modules, "augraphy", None)
        with pytest.raises(RuntimeError, match="Augraphy backend was requested"):
            resolve_backend("augraphy")

    def test_the_backend_version_is_recorded_by_name(self) -> None:
        assert degrade.backend_version("opencv").startswith("cv2-")


# ------------------------------------------------------------------ a document


def ingested(env: Env, pages: int = 2, **row_overrides: Any) -> ManifestRow:
    make_sized_pdf(env.raw / "a.pdf", [A4] * pages)
    row = source_row(**row_overrides)
    env.ingest(row, settings=settings(dpi=40))
    return row


def make_variant(env: Env, row: ManifestRow, level: str = "light", **kwargs: Any) -> Any:
    return degrade_document(row, level, pages_dir=env.pages, backend="opencv", **kwargs)  # type: ignore[arg-type]


class TestDocument:
    def test_it_writes_pages_next_to_the_source_with_the_same_count_size_and_format(
        self, env: Env
    ) -> None:
        row = ingested(env, pages=3)
        result = make_variant(env, row, "medium")
        source_sidecar = env.sidecar(row)
        assert result.status == "created" and result.backend == "opencv"
        assert [p.size for p in result.pages] == [tuple(p["size"]) for p in source_sidecar["pages"]]
        assert [p.file for p in result.pages] == ["p1.png", "p2.png", "p3.png"]
        folder = env.pages / "doc_001-deg-medium"
        for record, original in zip(result.pages, source_sidecar["pages"], strict=True):
            stored = Image.open(folder / record.file)
            assert stored.format == "PNG" and stored.size == tuple(original["size"])
            assert record.source_sha256 == original["sha256"]
            assert stored.tobytes() != Image.open(env.pages / "doc_001" / record.file).tobytes()

    def test_every_page_gets_its_own_seed(self, env: Env) -> None:
        row = ingested(env, pages=2)  # the two source pages are identical
        a, b = make_variant(env, row, "medium").pages
        source = env.sidecar(row)["pages"]
        assert source[0]["sha256"] == source[1]["sha256"]
        assert a.sha256 != b.sha256

    def test_the_sidecar_records_how_it_was_made_and_no_file_names(self, env: Env) -> None:
        row = ingested(env)
        make_variant(env, row, "heavy", seed="my-seed")
        data = json.loads((env.pages / "doc_001-deg-heavy" / SIDECAR).read_text(encoding="utf-8"))
        assert (
            data["source"] == "doc_001" and data["level"] == "heavy" and data["seed"] == "my-seed"
        )
        assert data["backend"] == "opencv" and data["backend_version"].startswith("cv2-")
        assert data["degrade_version"] == degrade.DEGRADE_VERSION
        assert "a.pdf" not in json.dumps(data)
        assert {p["n"] for p in data["pages"]} == {1, 2}

    def test_it_is_idempotent_and_deterministic(self, env: Env) -> None:
        row = ingested(env)
        first = make_variant(env, row)
        assert make_variant(env, row).status == "unchanged"
        again = make_variant(env, row, force=True)
        assert again.status == "created"
        assert [p.sha256 for p in again.pages] == [p.sha256 for p in first.pages]

    @pytest.mark.parametrize(
        "change", [{"seed": "other"}, {"jpeg_quality": 80}], ids=["seed", "quality"]
    )
    def test_a_change_of_inputs_redoes_the_variant(self, env: Env, change: dict[str, Any]) -> None:
        row = ingested(env)
        make_variant(env, row)
        assert make_variant(env, row, **change).status == "created"

    def test_a_damaged_variant_is_redone_not_trusted(self, env: Env) -> None:
        row = ingested(env)
        first = make_variant(env, row)
        page = env.pages / "doc_001-deg-light" / "p1.png"
        page.write_bytes(page.read_bytes() + b"x")
        again = make_variant(env, row)
        assert again.status == "created"
        assert [p.sha256 for p in again.pages] == [p.sha256 for p in first.pages]
        page.unlink()
        assert make_variant(env, row).status == "created"
        assert page.is_file()

    def test_a_sidecar_from_another_version_is_not_trusted(self, env: Env) -> None:
        row = ingested(env)
        make_variant(env, row)
        sidecar = env.pages / "doc_001-deg-light" / SIDECAR
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["degrade_version"] = degrade.DEGRADE_VERSION + 1
        sidecar.write_text(json.dumps(data), encoding="utf-8")
        assert make_variant(env, row).status == "created"
        sidecar.write_text("not json", encoding="utf-8")
        assert make_variant(env, row).status == "created"

    def test_the_levels_are_separate_variants(self, env: Env) -> None:
        row = ingested(env, pages=1)
        light, heavy = make_variant(env, row, "light"), make_variant(env, row, "heavy")
        assert light.doc_id != heavy.doc_id
        assert light.pages[0].sha256 != heavy.pages[0].sha256

    def test_a_check_writes_nothing(self, env: Env) -> None:
        row = ingested(env)
        result = make_variant(env, row, dry_run=True)
        assert result.status == "would_create" and result.pages == ()
        assert not (env.pages / "doc_001-deg-light").exists()
        make_variant(env, row)
        assert make_variant(env, row, dry_run=True).status == "unchanged"

    def test_the_inputs_hash_covers_everything_that_changes_the_output(self) -> None:
        pages = [{"sha256": "aa", "format": "png"}]
        base = degrade._inputs_hash(pages, "light", "s", "opencv", "v1", 92)
        variations = [
            degrade._inputs_hash(
                [{"sha256": "bb", "format": "png"}], "light", "s", "opencv", "v1", 92
            ),
            degrade._inputs_hash(
                [{"sha256": "aa", "format": "jpeg"}], "light", "s", "opencv", "v1", 92
            ),
            degrade._inputs_hash(pages, "heavy", "s", "opencv", "v1", 92),
            degrade._inputs_hash(pages, "light", "t", "opencv", "v1", 92),
            degrade._inputs_hash(pages, "light", "s", "augraphy", "v1", 92),
            degrade._inputs_hash(pages, "light", "s", "opencv", "v2", 92),
            degrade._inputs_hash(pages, "light", "s", "opencv", "v1", 80),
        ]
        assert len({base, *variations}) == 8

    def test_pages_that_the_source_no_longer_has_are_removed(self, env: Env) -> None:
        make_sized_pdf(env.raw / "a.pdf", [A4] * 3)
        row = source_row()
        env.ingest(row, settings=settings(dpi=40))
        make_variant(env, row)
        assert (env.pages / "doc_001-deg-light" / "p3.png").exists()
        shorter = source_row(page_range="1-2")
        env.ingest(shorter, settings=settings(dpi=40))
        make_variant(env, shorter)
        assert not (env.pages / "doc_001-deg-light" / "p3.png").exists()

    def test_photos_pages_keep_their_format(self, env: Env) -> None:
        make_jpeg(env.raw / "a.pdf", size=(120, 90), color=(220, 220, 220))  # a "born digital" jpeg
        row = source_row(raw_files="a.pdf")
        env.ingest(row)
        result = make_variant(env, row)
        assert result.pages[0].format == "jpeg" and result.pages[0].file == "p1.jpg"

    def test_the_redaction_survives_the_degradation(self, env: Env) -> None:
        Image.new("RGB", (200, 160), (240, 240, 240)).save(env.raw / "g.png", "PNG")
        row = source_row(raw_files="g.png", needs_redaction="y")
        env.ingest(row, boxes={1: [[40, 40, 100, 60]]})
        make_variant(env, row, "light")
        page = np.asarray(Image.open(env.pages / "doc_001-deg-light" / "p1.png").convert("L"))
        assert page[50:90, 50:130].mean() < 40  # still a black box
        assert page[:30, :].mean() > 150  # the page around it is still paper

    @pytest.mark.parametrize("capture", [None, "phone_photo", "screen_capture", "constructed"])
    def test_only_born_digital_documents_are_degraded(self, env: Env, capture: str | None) -> None:
        row = ingested(env, capture=capture)
        with pytest.raises(DegradeError, match="only born_digital documents are degraded"):
            make_variant(env, row)

    def test_a_variant_cannot_be_degraded(self, env: Env) -> None:
        ingested(env)
        with pytest.raises(DegradeError, match="is itself a variant"):
            make_variant(env, variant_row(source_row(), "light"))

    def test_the_source_must_have_been_ingested(self, env: Env) -> None:
        with pytest.raises(DegradeError, match="not ingested yet"):
            make_variant(env, source_row())

    def test_a_tampered_or_missing_source_page_is_refused(self, env: Env) -> None:
        row = ingested(env)
        page = env.pages / "doc_001" / "p1.png"
        original = page.read_bytes()
        page.write_bytes(original + b"x")
        with pytest.raises(DegradeError, match="no longer matches its ingest.json"):
            make_variant(env, row)
        page.unlink()
        with pytest.raises(DegradeError, match="stored page p1.png is missing"):
            make_variant(env, row)
        assert not (env.pages / "doc_001-deg-light").exists()

    def test_it_never_reads_the_raw_files(self, env: Env) -> None:
        row = ingested(env)
        (env.raw / "a.pdf").unlink()
        assert make_variant(env, row).status == "created"


# ------------------------------------------------------------------ the manifest rows


class TestManifestRows:
    def sources(self) -> list[ManifestRow]:
        return [
            source_row(doc_id="a", content_group="a"),
            source_row(doc_id="b", content_group="b"),
            source_row(doc_id="c", content_group="c", capture="phone_photo"),
        ]

    def test_rows_are_inserted_right_after_their_source(self) -> None:
        rows = self.sources()
        result = add_variant_rows(
            rows, [(rows[0], "light"), (rows[0], "heavy"), (rows[1], "medium")]
        )
        assert [r.doc_id for r in result] == [
            "a", "a-deg-light", "a-deg-heavy", "b", "b-deg-medium", "c",
        ]  # fmt: skip

    def test_the_result_is_a_valid_manifest(self) -> None:
        rows = self.sources()
        result = add_variant_rows(rows, [(r, level) for r in rows[:2] for level in LEVELS])
        assert check_manifest(result) == []

    def test_adding_again_changes_nothing(self) -> None:
        rows = self.sources()
        wanted = [(rows[0], "light"), (rows[1], "heavy")]
        once = add_variant_rows(rows, wanted)
        assert add_variant_rows(once, wanted) == once

    def test_an_existing_row_that_is_something_else_is_refused(self) -> None:
        rows = self.sources()
        clash = source_row(doc_id="a-deg-light", content_group="a-deg-light")
        with pytest.raises(DegradeError, match="already in the manifest as something else"):
            add_variant_rows([*rows, clash], [(rows[0], "light")])

    def test_variants_land_on_their_sources_side_of_the_split(self) -> None:
        rows = [source_row(doc_id=f"d{i:02d}", content_group=f"d{i:02d}") for i in range(12)]
        with_variants = add_variants(rows)
        plan = assign_splits(with_variants)
        splits = {r.doc_id: r.split for r in plan.rows}
        for row in plan.rows:
            if row.is_derived:
                assert splits[row.doc_id] == splits[str(row.gt_from)]
        assert sum(s.groups for s in plan.strata) == 12  # variants are not groups of their own


def add_variants(rows: list[ManifestRow]) -> list[ManifestRow]:
    return add_variant_rows(rows, [(r, level) for r in rows for level in ("light", "heavy")])


# ------------------------------------------------------------------ the command line


@pytest.mark.slow
class TestCommandLine:
    def setup_dataset(self, tmp_path: Path, rows: list[ManifestRow]) -> tuple[Path, Path, Path]:
        env = Env(tmp_path)
        make_sized_pdf(env.raw / "a.pdf", [A4, A4])
        for row in rows:
            if row.raw_files and (env.raw / row.raw_files[0]).exists() and not row.is_derived:
                env.ingest(row, settings=settings(dpi=40))
        return env.raw, write_manifest_file(tmp_path, rows), env.pages

    def run(self, manifest: Path, pages: Path, *extra: str) -> tuple[int, str, str]:
        return run_script(
            "degrade_docs.py",
            "--manifest",
            str(manifest),
            "--pages-dir",
            str(pages),
            "--backend",
            "opencv",
            *extra,
        )

    def test_it_makes_variants_and_adds_their_rows_then_leaves_them_alone(
        self, tmp_path: Path
    ) -> None:
        _, manifest, pages = self.setup_dataset(tmp_path, [source_row(split="dev")])
        code, out, _ = self.run(manifest, pages, "--doc", "doc_001", "--levels", "light,heavy")
        assert code == 0, out
        assert "created     doc_001-deg-light  2 page(s)" in out
        assert "OpenCV backend (requested)" in out
        rows = read_manifest(manifest)
        assert [r.doc_id for r in rows] == ["doc_001", "doc_001-deg-light", "doc_001-deg-heavy"]
        heavy = rows[2]
        assert (heavy.capture, heavy.variant, heavy.gt_from) == (
            "synthetic_degraded",
            "heavy",
            "doc_001",
        )
        assert (heavy.split, heavy.content_group, heavy.doc_type) == ("dev", "g1", "faktur")
        assert (pages / "doc_001-deg-heavy" / "p2.png").is_file()
        before = manifest.read_bytes()
        code, out, _ = self.run(manifest, pages, "--doc", "doc_001", "--levels", "light,heavy")
        assert code == 0 and "unchanged" in out and "created" not in out
        assert manifest.read_bytes() == before

    def test_check_writes_nothing(self, tmp_path: Path) -> None:
        _, manifest, pages = self.setup_dataset(tmp_path, [source_row()])
        before = manifest.read_bytes()
        code, out, _ = self.run(manifest, pages, "--doc", "doc_001", "--check")
        assert code == 0 and "would_create" in out
        assert manifest.read_bytes() == before
        assert not (pages / "doc_001-deg-light").exists()

    def test_all_selects_only_born_digital_sources(self, tmp_path: Path) -> None:
        rows = [
            source_row(doc_id="doc_001", content_group="doc_001"),
            source_row(doc_id="doc_002", content_group="doc_002", capture="phone_photo"),
        ]
        _, manifest, pages = self.setup_dataset(tmp_path, rows)
        code, out, _ = self.run(manifest, pages, "--all", "--levels", "light")
        assert code == 0, out
        assert [r.doc_id for r in read_manifest(manifest)] == [
            "doc_001", "doc_001-deg-light", "doc_002",
        ]  # fmt: skip

    def test_a_document_that_cannot_be_degraded_fails_alone(self, tmp_path: Path) -> None:
        rows = [
            source_row(doc_id="doc_001", content_group="doc_001"),
            source_row(doc_id="doc_002", content_group="doc_002", capture="phone_photo"),
        ]
        _, manifest, pages = self.setup_dataset(tmp_path, rows)
        code, out, _ = self.run(
            manifest, pages, "--doc", "doc_002", "--doc", "doc_001", "--levels", "light"
        )
        assert code == 1
        assert "FAILED    doc_002: capture is phone_photo" in out
        assert "created     doc_001-deg-light" in out
        assert [r.doc_id for r in read_manifest(manifest)] == [
            "doc_001", "doc_001-deg-light", "doc_002",
        ]  # fmt: skip

    def test_a_problem_with_a_source_is_reported_once_not_once_per_level(
        self, tmp_path: Path
    ) -> None:
        _, manifest, pages = self.setup_dataset(tmp_path, [source_row()])
        for sidecar in pages.glob("*/ingest.json"):
            sidecar.unlink()
        code, out, _ = self.run(manifest, pages, "--doc", "doc_001")  # all three levels
        assert code == 1
        assert out.count("not ingested yet") == 1
        assert "done: 0 variant(s), 1 failed" in out

    def test_a_source_that_was_not_ingested_is_reported(self, tmp_path: Path) -> None:
        _, manifest, pages = self.setup_dataset(tmp_path, [source_row()])
        for sidecar in pages.glob("*/ingest.json"):
            sidecar.unlink()
        code, out, _ = self.run(manifest, pages, "--doc", "doc_001")
        assert code == 1 and "not ingested yet" in out

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ([], "Choose the documents"),
            (["--doc", "doc_001", "--all"], "Choose the documents"),
            (["--doc", "nope"], "Not in the manifest: nope"),
            (["--doc", "doc_001", "--levels", "light,extreme"], "unknown level 'extreme'"),
        ],
    )
    def test_bad_arguments_exit_2(self, tmp_path: Path, args: list[str], message: str) -> None:
        _, manifest, pages = self.setup_dataset(tmp_path, [source_row()])
        code, _, err = self.run(manifest, pages, *args)
        assert code == 2
        assert message in err

    def test_an_unreadable_manifest_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "m.csv"
        bad.write_text("nonsense\n", encoding="utf-8")
        code, _, err = self.run(bad, tmp_path / "pages", "--doc", "x")
        assert code == 2 and "cannot be read" in err


# ------------------------------------------------------------------ Augraphy


@pytest.mark.slow
class TestAugraphy:
    """The real Augraphy pipelines: ~20 s to import it, ~1 s a page."""

    @pytest.fixture(autouse=True)
    def _need_augraphy(self) -> None:
        pytest.importorskip("augraphy")

    @pytest.mark.parametrize("level", LEVELS)
    def test_size_is_preserved_and_the_page_is_still_the_page(self, level: str) -> None:
        page = text_page()
        out = degrade_image(page, level, 3, "augraphy")  # type: ignore[arg-type]
        assert out.size == page.size and out.mode == "RGB"
        assert similarity(page, out) > 0.5
        assert 0.6 < density(out)[1] / density(page)[1] < 1.6

    @pytest.mark.parametrize("level", LEVELS)
    def test_it_is_deterministic_for_every_level(self, level: str) -> None:
        # a full pipeline, several times over: one Augraphy effect (BadPhotoCopy) was reproducible
        # alone but not inside a pipeline, which only repeated runs of the whole thing reveal
        page = text_page()
        runs = {degrade_image(page, level, 11, "augraphy").tobytes() for _ in range(3)}  # type: ignore[arg-type]
        assert len(runs) == 1
        other = degrade_image(page, level, 12, "augraphy").tobytes()  # type: ignore[arg-type]
        assert other not in runs

    def test_severity_grows_with_the_level(self) -> None:
        page = text_page()
        diffs = {
            level: float(
                np.mean(
                    [mean_abs_diff(page, degrade_image(page, level, s, "augraphy")) for s in (1, 2)]  # type: ignore[arg-type]
                )
            )
            for level in LEVELS
        }
        assert diffs["light"] < diffs["medium"] < diffs["heavy"]

    def test_a_document_records_the_augraphy_backend(self, env: Env) -> None:
        row = ingested(env, pages=1)
        result = degrade_document(row, "light", pages_dir=env.pages, backend="augraphy")
        assert result.backend == "augraphy"
        data = json.loads((env.pages / "doc_001-deg-light" / SIDECAR).read_text(encoding="utf-8"))
        assert data["backend"] == "augraphy" and data["backend_version"] != ""
        assert (
            degrade_document(row, "light", pages_dir=env.pages, backend="augraphy").status
            == "unchanged"
        )
        # a variant made with one backend is not mistaken for the other's
        assert make_variant(env, row, "light").status == "created"

    def test_it_does_not_litter_the_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        degrade_image(Image.new("RGB", (120, 160), "white"), "light", 1, "augraphy")
        assert Path(os.getcwd()).resolve() == tmp_path.resolve()  # restored
        assert list(tmp_path.iterdir()) == []  # no ./augraphy_cache

    def test_auto_prefers_augraphy(self) -> None:
        backend, note = resolve_backend("auto")
        assert backend == "augraphy" and note == "Augraphy backend"
