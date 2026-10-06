"""Helpers shared by the ingestion and redaction tests: a raw folder + output folder under tmp_path
and a one-call ingest, manifest-row and settings builders, pixel readers."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from docextract.config import IngestSettings
from docextract.ingest.manifest import ManifestRow, write_manifest
from docextract.ingest.pipeline import SIDECAR, IngestResult, ingest_document


def settings(**overrides: Any) -> IngestSettings:
    return IngestSettings(_env_file=None, **overrides)


def doc(*raw_files: str, **overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "doc_id": "doc_001",
        "raw_files": ";".join(raw_files),
        "needs_redaction": False,
    }
    values.update(overrides)
    return ManifestRow.model_validate(values)


class Env:
    """A raw folder and an output folder under tmp_path, and a one-call ingest."""

    def __init__(self, tmp_path: Path) -> None:
        self.raw = tmp_path / "raw"
        self.raw.mkdir()
        self.pages = tmp_path / "pages"

    def out(self, row: ManifestRow) -> Path:
        return self.pages / row.doc_id

    def ingest(self, row: ManifestRow, **kwargs: Any) -> IngestResult:
        config = kwargs.pop("settings", settings())
        return ingest_document(
            row, raw_dir=self.raw, out_dir=self.out(row), settings=config, **kwargs
        )

    def page(self, row: ManifestRow, name: str) -> Image.Image:
        image = Image.open(self.out(row) / name)
        image.load()
        return image

    def sidecar(self, row: ManifestRow) -> dict[str, Any]:
        return json.loads((self.out(row) / SIDECAR).read_text(encoding="utf-8"))


def pixels(image: Image.Image) -> list[Any]:
    """Every pixel of `image` (Pillow is deprecating getdata in favour of get_flattened_data)."""
    getter = getattr(image, "get_flattened_data", None) or image.getdata
    return list(getter())


def gray_png(env: Env, name: str = "g.png", size: tuple[int, int] = (80, 60)) -> None:
    Image.new("RGB", size, (128, 128, 128)).save(env.raw / name, "PNG")


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *args: str) -> tuple[int, str, str]:
    """Run scripts/<script> in a subprocess (INGEST_* settings scrubbed from the environment)."""
    env_vars = {k: v for k, v in os.environ.items() if not k.startswith("INGEST_")}
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env_vars,
    )
    return done.returncode, done.stdout, done.stderr


def write_manifest_file(tmp_path: Path, rows: list[ManifestRow]) -> Path:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    return manifest
