"""Generate the synthetic receipt used as a smoke-test fixture.

Usage: uv run python scripts/make_sample_receipt.py [OUTPUT_PNG]

The receipt is fictional and says so on its face. What it states (for eyeballing smoke output):
  vendor        CORNER CAFE DEMO, 12 Example Street, Springfield   tax id 12-3456789
  receipt no    R-0042                                              date  Mar 5, 2024
  buyer / due date / discount: NOT on the document (a model should abstain)
  lines         Latte 2 x 4.50 = 9.00 | Croissant 1 x 3.25 = 3.25 | Orange juice 1 x 3.75 = 3.75
  subtotal      16.00      tax (8%)  1.28      total  $17.28      paid by  VISA ending 1234
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_receipt.png"
WIDTH = 560
MARGIN = 36
INK = (20, 20, 20)
GREY = (120, 120, 120)

ITEMS = [
    ("Latte", 2, "4.50", "9.00"),
    ("Croissant", 1, "3.25", "3.25"),
    ("Orange juice", 1, "3.75", "3.75"),
]


def render() -> Image.Image:
    small = ImageFont.load_default(size=18)
    body = ImageFont.load_default(size=22)
    title = ImageFont.load_default(size=34)

    image = Image.new("RGB", (WIDTH, 700), "white")
    draw = ImageDraw.Draw(image)
    right = WIDTH - MARGIN
    y = 24

    def centered(
        text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int], gap: int
    ) -> None:
        nonlocal y
        draw.text((WIDTH / 2, y), text, font=font, fill=fill, anchor="ma")
        y += gap

    def rule() -> None:
        nonlocal y
        draw.line((MARGIN, y + 6, right, y + 6), fill=GREY, width=1)
        y += 20

    def pair(left: str, value: str, font: ImageFont.FreeTypeFont = body) -> None:
        nonlocal y
        draw.text((MARGIN, y), left, font=font, fill=INK)
        draw.text((right, y), value, font=font, fill=INK, anchor="ra")
        y += 32

    centered("SAMPLE RECEIPT - TEST FIXTURE", small, GREY, 34)
    centered("CORNER CAFE DEMO", title, INK, 48)
    centered("12 Example Street, Springfield", body, INK, 30)
    centered("Tax ID: 12-3456789", body, INK, 36)
    rule()
    pair("Receipt No: R-0042", "Mar 5, 2024")
    rule()

    draw.text((MARGIN, y), "Item", font=small, fill=GREY)
    draw.text((330, y), "Qty", font=small, fill=GREY, anchor="ra")
    draw.text((430, y), "Price", font=small, fill=GREY, anchor="ra")
    draw.text((right, y), "Amount", font=small, fill=GREY, anchor="ra")
    y += 30
    for name, qty, price, amount in ITEMS:
        draw.text((MARGIN, y), name, font=body, fill=INK)
        draw.text((330, y), str(qty), font=body, fill=INK, anchor="ra")
        draw.text((430, y), price, font=body, fill=INK, anchor="ra")
        draw.text((right, y), amount, font=body, fill=INK, anchor="ra")
        y += 32
    rule()

    pair("Subtotal", "16.00")
    pair("Tax (8%)", "1.28")
    pair("TOTAL", "$17.28", title)
    y += 4
    rule()
    centered("Paid by: VISA ending 1234", body, INK, 40)
    centered("Thank you!", body, GREY, 30)

    return image.crop((0, 0, WIDTH, y + 20))


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    render().save(output, optimize=True)
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
