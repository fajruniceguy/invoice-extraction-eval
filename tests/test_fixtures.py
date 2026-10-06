from pathlib import Path

from PIL import Image

from docextract.providers.openai_compat import build_image_message

FIXTURE = Path(__file__).parent / "fixtures" / "sample_receipt.png"


def test_sample_receipt_is_a_real_png() -> None:
    assert FIXTURE.is_file(), "regenerate with: uv run python scripts/make_sample_receipt.py"
    with Image.open(FIXTURE) as image:
        assert image.format == "PNG"
        assert image.width >= 400
        assert image.height >= 400


def test_sample_receipt_is_accepted_by_the_image_message_builder() -> None:
    message = build_image_message("read", [FIXTURE])
    assert message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
