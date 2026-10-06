import base64
import io
import math
import random
from pathlib import Path

import pytest
from PIL import Image

from docextract.config import Settings
from docextract.imaging import (
    ELONGATED_AREA_AT_REFERENCE,
    HARD_MAX_SIDE,
    encode_page,
    prepare_for_model,
    storage_format,
    target_size,
)
from docextract.providers.cache import fingerprint
from docextract.providers.openai_compat import build_image_message


class TestTargetSize:
    def test_a_page_inside_the_cap_is_unchanged(self) -> None:
        assert target_size(1654, 2339, 2400) == (1654, 2339)  # A4 at 200 DPI
        assert target_size(2400, 1800, 2400) == (2400, 1800)  # exactly at the cap

    def test_a_large_landscape_photo_is_capped_on_the_long_edge(self) -> None:
        assert target_size(4000, 3000, 2400) == (2400, 1800)

    def test_a_large_portrait_photo_is_capped_on_the_long_edge(self) -> None:
        assert target_size(3000, 4000, 2400) == (1800, 2400)

    def test_nothing_is_ever_upscaled(self) -> None:
        assert target_size(800, 600, 2400) == (800, 600)
        assert target_size(100, 100, 4000) == (100, 100)

    def test_aspect_ratio_is_preserved_for_normal_pages(self) -> None:
        width, height = target_size(4032, 3024, 2400)
        assert abs(width / height - 4032 / 3024) < 0.002

    def test_no_cap_means_only_the_hard_limit(self) -> None:
        assert target_size(6000, 4000, None) == (6000, 4000)
        assert target_size(12000, 8000, None) == (8000, round(8000 * 8000 / 12000))
        assert target_size(12000, 8000, 0) == target_size(12000, 8000, None)

    def test_an_elongated_page_that_fits_the_area_budget_keeps_every_pixel(self) -> None:
        # 1200 x 4000 is a long thermal receipt: 4.8 MP is under the 6 MP budget, so it is NOT
        # squeezed to a 2400 px long edge (that would cost it a third of its width)
        assert target_size(1200, 4000, 2400) == (1200, 4000)

    def test_a_huge_elongated_page_is_capped_by_area_not_long_edge(self) -> None:
        width, height = target_size(3000, 9000, 2400)  # 27 MP
        assert width * height <= ELONGATED_AREA_AT_REFERENCE * 1.001
        assert width * height >= ELONGATED_AREA_AT_REFERENCE * 0.99
        assert height > 2400  # the long edge is allowed to exceed the cap
        assert abs(width / height - 3000 / 9000) < 0.002

    def test_the_elongated_threshold_is_strictly_greater_than_two_and_a_half(self) -> None:
        # exactly 2.5: a normal page, so the long edge is capped
        assert target_size(1000, 2500, 2400) == (960, 2400)
        # just above 2.5: elongated, so the (generous) area budget applies instead
        assert target_size(1000, 2510, 2400) == (1000, 2510)

    def test_the_area_budget_scales_with_the_cap_squared(self) -> None:
        width, height = target_size(1200, 4000, 1200)  # budget = 6 MP / 4 = 1.5 MP
        assert width * height <= 1_500_000 * 1.001
        assert width * height >= 1_500_000 * 0.99
        assert abs(width / height - 0.3) < 0.002

    def test_the_hard_side_limit_beats_the_area_rule(self) -> None:
        width, height = target_size(600, 30000, 2400)
        assert max(width, height) == HARD_MAX_SIDE
        assert abs(width / height - 600 / 30000) < 0.002

    @pytest.mark.parametrize("size", [(0, 10), (10, 0), (-1, 5)])
    def test_invalid_sizes_are_rejected(self, size: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="invalid image size"):
            target_size(*size, 2400)

    def test_a_degenerate_sliver_never_collapses_to_zero(self) -> None:
        width, height = target_size(1, 20000, 2400)
        assert width >= 1
        assert height >= 1


class TestStorageFormat:
    @pytest.mark.parametrize("source", ["JPEG", "jpeg", "MPO", "HEIF", "HEIC"])
    def test_lossy_photo_sources_are_stored_as_jpeg(self, source: str) -> None:
        assert storage_format(source) == "jpeg"

    @pytest.mark.parametrize("source", ["PNG", "PDF", "BMP", "TIFF", "WEBP", "GIF", "", None])
    def test_everything_else_is_stored_as_png(self, source: str | None) -> None:
        assert storage_format(source) == "png"


def noisy(size: tuple[int, int], seed: int = 1, mode: str = "RGB") -> Image.Image:
    """Incompressible content, so file-size limits actually bite."""
    channels = {"RGB": 3, "L": 1}[mode]
    data = random.Random(seed).randbytes(size[0] * size[1] * channels)
    return Image.frombytes(mode, size, data)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestEncodePage:
    def test_png_is_lossless(self) -> None:
        image = noisy((40, 30))
        decoded = Image.open(io.BytesIO(encode_page(image, "png")))
        assert decoded.format == "PNG"
        assert decoded.tobytes() == image.tobytes()

    def test_jpeg_is_a_jpeg_close_to_the_original(self) -> None:
        image = Image.new("RGB", (64, 64), (200, 30, 30))
        decoded = Image.open(io.BytesIO(encode_page(image, "jpeg", jpeg_quality=92)))
        assert decoded.format == "JPEG"
        red, green, blue = decoded.getpixel((32, 32))
        assert abs(red - 200) < 8
        assert abs(green - 30) < 8
        assert abs(blue - 30) < 8

    @pytest.mark.parametrize("fmt", ["png", "jpeg"])
    def test_encoding_is_deterministic(self, fmt: str) -> None:
        image = noisy((50, 40), seed=3)
        assert encode_page(image, fmt) == encode_page(image, fmt)  # type: ignore[arg-type]

    @pytest.mark.parametrize("fmt", ["png", "jpeg"])
    def test_no_metadata_is_written(self, fmt: str) -> None:
        image = Image.new("RGB", (20, 20), "white")
        image.info["dpi"] = (300, 300)
        image.info["comment"] = b"secret"
        decoded = Image.open(io.BytesIO(encode_page(image, fmt)))  # type: ignore[arg-type]
        assert len(decoded.getexif()) == 0
        assert "icc_profile" not in decoded.info
        assert "comment" not in decoded.info
        assert "dpi" not in decoded.info
        assert "exif" not in decoded.info

    def test_transparency_is_composited_on_white(self) -> None:
        image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))  # fully transparent
        decoded = Image.open(io.BytesIO(encode_page(image, "png"))).convert("RGB")
        assert decoded.getpixel((5, 5)) == (255, 255, 255)

    def test_grayscale_and_palette_inputs_become_rgb(self) -> None:
        for source in (noisy((8, 8), mode="L"), Image.new("P", (8, 8))):
            assert Image.open(io.BytesIO(encode_page(source, "png"))).mode == "RGB"


class TestPrepareForModel:
    def test_a_page_that_already_fits_is_sent_byte_for_byte(self, tmp_path: Path) -> None:
        stored = tmp_path / "p1.png"
        stored.write_bytes(png_bytes(noisy((300, 200))))
        data, mime = prepare_for_model(stored, max_long_edge=2400)
        assert data == stored.read_bytes()
        assert mime == "image/png"

    def test_no_limits_means_no_change(self, tmp_path: Path) -> None:
        raw = png_bytes(noisy((300, 200)))
        assert prepare_for_model(raw) == (raw, "image/png")

    def test_a_larger_page_is_downscaled_in_memory_and_the_file_is_untouched(
        self, tmp_path: Path
    ) -> None:
        stored = tmp_path / "p1.png"
        stored.write_bytes(png_bytes(noisy((1200, 800))))
        before = stored.read_bytes()
        data, mime = prepare_for_model(stored, max_long_edge=600)
        sent = Image.open(io.BytesIO(data))
        assert sent.size == (600, 400)
        assert mime == "image/png"  # a PNG page stays a PNG
        assert len(data) < len(before)
        assert stored.read_bytes() == before  # the stored page is never modified

    def test_a_jpeg_page_stays_a_jpeg(self) -> None:
        buffer = io.BytesIO()
        noisy((1000, 700)).save(buffer, format="JPEG", quality=92)
        data, mime = prepare_for_model(buffer.getvalue(), max_long_edge=500)
        assert Image.open(io.BytesIO(data)).format == "JPEG"
        assert Image.open(io.BytesIO(data)).size == (500, 350)
        assert mime == "image/jpeg"

    def test_the_elongated_area_rule_applies_at_request_time_too(self) -> None:
        raw = png_bytes(Image.new("RGB", (1200, 4000), "white"))
        # E = 1200 gives a 1.5 MP budget; the 4.8 MP page must shrink to fit it
        data, _ = prepare_for_model(raw, max_long_edge=1200)
        width, height = Image.open(io.BytesIO(data)).size
        assert width * height <= 1_500_000 * 1.001
        assert height > 1200  # long edge intentionally larger than E
        # at E = 2400 the same page keeps every pixel and is sent untouched
        assert prepare_for_model(raw, max_long_edge=2400)[0] == raw

    def test_max_bytes_forces_a_smaller_derivative(self) -> None:
        raw = png_bytes(noisy((1000, 1000)))  # incompressible: ~3 MB
        limit = 400_000
        assert len(raw) > limit
        data, mime = prepare_for_model(raw, max_long_edge=None, max_bytes=limit)
        assert len(data) <= limit
        assert mime == "image/jpeg"  # PNG noise cannot be shrunk enough without going lossy

    def test_an_impossible_byte_limit_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="cannot fit"):
            prepare_for_model(png_bytes(noisy((300, 300))), max_bytes=50)

    def test_a_page_under_the_byte_limit_is_untouched(self) -> None:
        raw = png_bytes(Image.new("RGB", (100, 100), "white"))
        assert prepare_for_model(raw, max_bytes=1_000_000) == (raw, "image/png")

    def test_other_formats_become_a_jpeg(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (200, 200), "white").save(buffer, format="WEBP")
        data, mime = prepare_for_model(buffer.getvalue(), max_long_edge=100)
        assert mime == "image/jpeg"
        assert Image.open(io.BytesIO(data)).size == (100, 100)


class TestBuildImageMessageResolution:
    def decode(self, message: dict[str, object]) -> Image.Image:
        content = message["content"]
        assert isinstance(content, list)
        url = content[1]["image_url"]["url"]
        assert isinstance(url, str)
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))

    def test_the_default_sends_the_bytes_exactly_as_given(self) -> None:
        raw = png_bytes(noisy((400, 300)))
        message = build_image_message("read", [raw])
        content = message["content"]
        assert isinstance(content, list)
        assert base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]) == raw

    def test_max_long_edge_shrinks_what_is_sent(self) -> None:
        message = build_image_message("read", [png_bytes(noisy((800, 600)))], max_long_edge=400)
        assert self.decode(message).size == (400, 300)

    def test_the_sent_mime_matches_the_derivative(self) -> None:
        message = build_image_message("read", [png_bytes(noisy((800, 600)))], max_long_edge=400)
        content = message["content"]
        assert isinstance(content, list)
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_changing_the_resolution_changes_the_cache_key(self) -> None:
        raw = png_bytes(noisy((800, 600)))
        common = {
            "model": "m",
            "base_url": "http://x/v1",
            "reasoning_effort": None,
            "temperature": 0.0,
            "max_tokens": None,
            "structured_mode": "auto",
            "schema_hash": "s",
        }
        stored = fingerprint([build_image_message("read", [raw])], **common)  # type: ignore[arg-type]
        smaller = fingerprint(
            [build_image_message("read", [raw], max_long_edge=400)],
            **common,  # type: ignore[arg-type]
        )
        same_again = fingerprint(
            [build_image_message("read", [raw], max_long_edge=400)],
            **common,  # type: ignore[arg-type]
        )
        assert stored.digest != smaller.digest  # a resolution sweep cannot replay a stale answer
        assert smaller.digest == same_again.digest  # and resizing is deterministic


class TestImageSettings:
    def make(self, **kwargs: object) -> Settings:
        return Settings(_env_file=None, llm_base_url="http://x/v1", llm_model="m", **kwargs)  # type: ignore[arg-type]

    def test_unset_by_default(self) -> None:
        assert self.make().llm_image_max_long_edge is None

    def test_read_from_the_environment_and_blank_means_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_IMAGE_MAX_LONG_EDGE", "1536")
        assert self.make().llm_image_max_long_edge == 1536
        monkeypatch.setenv("LLM_IMAGE_MAX_LONG_EDGE", "  ")
        assert self.make().llm_image_max_long_edge is None

    @pytest.mark.parametrize("bad", [0, -5])
    def test_must_be_positive(self, bad: int) -> None:
        with pytest.raises(ValueError):
            self.make(llm_image_max_long_edge=bad)


def test_the_area_budget_constant_matches_a_six_megapixel_page() -> None:
    assert ELONGATED_AREA_AT_REFERENCE == 6_000_000
    assert math.isclose(math.sqrt(ELONGATED_AREA_AT_REFERENCE), 2449.49, abs_tol=0.01)
