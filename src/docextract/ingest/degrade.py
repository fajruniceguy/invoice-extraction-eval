"""Synthetic degradation: light / medium / heavy variants of born-digital documents.

A born-digital invoice is a perfect PDF; real inputs are skewed, shadowed, noisy and compressed.
A variant is the SAME page after pixel transforms only, so it inherits the ground truth of its
source (manifest: `gt_from`, same `content_group`, same split, capture=synthetic_degraded):

  * it is made from the source's STORED pages (already redacted), never from the raw file;
  * page count and page size are preserved, and the whole page stays inside the canvas (a
    skewed page is scaled down a little instead of having its corners clipped);
  * nothing is added to or removed from the content: no stamps, no handwriting, no erased lines;
  * it is deterministic: the seed of every page derives from (master seed, source, level, page).

Two backends sit behind one interface. Augraphy (a scanner/photocopier simulator) is used when it
imports; otherwise a small OpenCV/NumPy pipeline with the same kinds of effects: rotation and
perspective, a light gradient, contrast, defocus blur, sensor noise and JPEG artefacts. The
backend and its version are recorded in each variant's `degrade.json` and are part of its
content hash, so a variant made with one backend is not mistaken for the other.

numpy, OpenCV and Augraphy are in the `degrade` dependency group and are imported lazily: nothing
else in docextract needs them (Augraphy alone takes ~20 s to import).
"""

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from docextract.imaging import encode_page, flatten
from docextract.ingest.manifest import ManifestRow, Variant
from docextract.ingest.pipeline import (
    SIDECAR as INGEST_SIDECAR,
)
from docextract.ingest.pipeline import (
    read_sidecar,
    sha256_hex,
    write_atomic,
)
from docextract.paths import find_repo_root

DEGRADE_VERSION = 1  # bump when the output would change for the same inputs
SIDECAR = "degrade.json"
DEFAULT_SEED = "docextract-degrade-v1"
LEVELS: tuple[Variant, ...] = ("light", "medium", "heavy")
Backend = Literal["augraphy", "opencv"]


class DegradeError(Exception):
    def __init__(self, doc_id: str, message: str) -> None:
        super().__init__(f"{doc_id}: {message}")
        self.doc_id = doc_id
        self.message = message


# ---------------------------------------------------------------- identity and seeds


def variant_id(source_id: str, level: str) -> str:
    return f"{source_id}-deg-{level}"


def seed_from_digest(digest: bytes) -> int:
    """Map a hash onto [1, 2**31 - 1]: Augraphy hands the seed to cv2.setRNGSeed (a signed 32-bit
    int) and treats 0 as "no seed", i.e. non-deterministic."""
    return int.from_bytes(digest[:4], "big") % (2**31 - 1) + 1


def derive_seed(master: str, source_id: str, level: str, page: int) -> int:
    """A seed that depends on nothing but its four arguments."""
    return seed_from_digest(
        hashlib.sha256(f"{master}|{source_id}|{level}|{page}".encode()).digest()
    )


def variant_row(source: ManifestRow, level: Variant) -> ManifestRow:
    """The manifest row of `source`'s variant: inherits the facts, borrows the labels."""
    if source.is_derived:
        raise ValueError(f"{source.doc_id} is itself a variant")
    return ManifestRow(
        doc_id=variant_id(source.doc_id, level),
        doc_type=source.doc_type,
        template_group=source.template_group,
        content_group=source.content_group or source.doc_id,
        capture="synthetic_degraded",
        language=source.language,
        license_note=f"synthetic variant of {source.doc_id}",
        split=source.split,
        needs_redaction=False,  # made from pages that are already redacted
        gt_from=source.gt_from or source.doc_id,
        variant=level,
    )


# ---------------------------------------------------------------- recipes


@dataclass(frozen=True)
class CvRecipe:
    rotation_deg: float  # max |angle|
    perspective: float  # max corner shift as a fraction of the page size
    shadow: float  # darkening at the dark end of a linear light gradient
    contrast: float  # 1.0 = unchanged
    blur_sigma: float
    noise_sigma: float
    jpeg_quality: int


CV_RECIPES: dict[Variant, CvRecipe] = {
    "light": CvRecipe(0.5, 0.004, 0.05, 0.98, 0.5, 2.5, 82),
    "medium": CvRecipe(1.3, 0.010, 0.12, 0.94, 0.8, 5.0, 65),
    "heavy": CvRecipe(2.6, 0.022, 0.22, 0.88, 1.2, 9.0, 45),
}


def fit_in_canvas(corners: Any, homography: Any, size: tuple[int, int]) -> Any:
    """`homography` followed by the scale and shift that put the whole transformed page inside
    a canvas of `size` (centred; scaled down only if needed)."""
    import cv2
    import numpy as np

    width, height = size
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    low, high = moved.min(axis=0), moved.max(axis=0)
    scale = min(1.0, width / (high[0] - low[0]), height / (high[1] - low[1]))
    centre = (low + high) / 2
    to_origin = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]], dtype=np.float64)
    scaling = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    to_canvas = np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]], dtype=np.float64)
    return to_canvas @ scaling @ to_origin @ homography


def page_homography(size: tuple[int, int], recipe: CvRecipe, rng: Any) -> Any:
    """Small rotation + perspective jitter of a page, kept inside the canvas."""
    import cv2
    import numpy as np

    width, height = size
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    angle = rng.uniform(-recipe.rotation_deg, recipe.rotation_deg)
    rotation = np.vstack([cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0), [0, 0, 1]])
    jitter = rng.uniform(-recipe.perspective, recipe.perspective, (4, 2)) * [width, height]
    perspective = cv2.getPerspectiveTransform(corners, (corners + jitter).astype(np.float32))
    return fit_in_canvas(corners, perspective @ rotation, size)


def degrade_opencv(image: Image.Image, level: Variant, seed: int) -> Image.Image:
    import cv2
    import numpy as np

    recipe = CV_RECIPES[level]
    rng = np.random.default_rng(seed)
    rgb = np.asarray(flatten(image))
    height, width = rgb.shape[:2]

    warped = cv2.warpPerspective(
        rgb,
        page_homography((width, height), recipe, rng),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    ).astype(np.float32)

    # light falls off along a random direction; paper is never quite white
    theta = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = xs * np.cos(theta) + ys * np.sin(theta)
    ramp = (ramp - ramp.min()) / max(float(ramp.max() - ramp.min()), 1.0)
    warped *= (1.0 - recipe.shadow * ramp)[:, :, None]
    warped = (warped - 128.0) * recipe.contrast + 128.0 + rng.uniform(-6.0, 6.0)

    blurred = cv2.GaussianBlur(warped, (0, 0), recipe.blur_sigma)
    noisy = blurred + rng.normal(0.0, recipe.noise_sigma, blurred.shape).astype(np.float32)
    clipped = np.clip(noisy, 0, 255).astype(np.uint8)

    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(clipped, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, recipe.jpeg_quality],
    )
    if not ok:
        raise RuntimeError("OpenCV could not JPEG-encode the degraded page")
    decoded = cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(decoded)


def _augraphy_pipeline(level: Variant, seed: int) -> Any:
    """Content-preserving Augraphy effects only: nothing that stamps, scribbles, erases ink along
    lines, or changes the page geometry."""
    from augraphy import (
        AugraphyPipeline,
        Brightness,
        DirtyScreen,
        Gamma,
        InkBleed,
        Jpeg,
        LightingGradient,
        NoiseTexturize,
        ShadowCast,
        SubtleNoise,
    )

    # BadPhotoCopy is deliberately not used: in a full pipeline it is not reproducible for a given
    # seed (and takes ~2 s a page). calibrated on a blank page: the mean brightness falls
    # light > medium > heavy, never by much
    # (LightingGradient's `transparency` is the weight of the ORIGINAL: lower is stronger)
    ink: list[Any]
    post: list[Any]
    if level == "light":
        ink = [InkBleed(intensity_range=(0.05, 0.1), kernel_size=(3, 3), severity=(0.1, 0.2), p=1)]
        post = [
            Brightness(brightness_range=(0.95, 1.05), p=1),
            SubtleNoise(subtle_range=4, p=1),
            Jpeg(quality_range=(78, 90), p=1),
        ]
    elif level == "medium":
        ink = [InkBleed(intensity_range=(0.1, 0.2), kernel_size=(5, 5), severity=(0.2, 0.4), p=1)]
        post = [
            LightingGradient(mode="gaussian", transparency=0.92, p=1),
            Gamma(gamma_range=(0.9, 1.15), p=1),
            NoiseTexturize(sigma_range=(1, 2), turbulence_range=(2, 3), p=1),
            Jpeg(quality_range=(55, 75), p=1),
        ]
    else:
        ink = [InkBleed(intensity_range=(0.15, 0.3), kernel_size=(5, 5), severity=(0.3, 0.5), p=1)]
        post = [
            LightingGradient(mode="gaussian", transparency=0.82, p=1),
            ShadowCast(
                shadow_side=None,
                shadow_vertices_range=(2, 3),
                shadow_width_range=(0.5, 0.8),
                shadow_height_range=(0.5, 0.8),
                shadow_opacity_range=(0.12, 0.2),
                shadow_iterations_range=(1, 2),
                shadow_blur_kernel_range=(101, 301),
                p=1,
            ),
            DirtyScreen(p=1),
            NoiseTexturize(sigma_range=(2, 3), turbulence_range=(2, 4), p=1),
            Jpeg(quality_range=(35, 55), p=1),
        ]
    return AugraphyPipeline(ink_phase=ink, paper_phase=[], post_phase=post, random_seed=seed)


@contextmanager
def _augraphy_workdir() -> Iterator[None]:
    """Augraphy writes ~13 MB of generated textures to ./augraphy_cache relative to the CURRENT
    directory: run it inside outputs/cache/augraphy (gitignored) instead of littering whatever
    directory the script was started from."""
    target = find_repo_root() / "outputs" / "cache" / "augraphy"
    target.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


def degrade_augraphy(image: Image.Image, level: Variant, seed: int) -> Image.Image:
    import cv2
    import numpy as np

    rgb = np.asarray(flatten(image))
    with _augraphy_workdir():
        result = _augraphy_pipeline(level, seed)(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2RGB))


def backend_version(backend: Backend) -> str:
    if backend == "augraphy":
        import augraphy

        return str(getattr(augraphy, "__version__", "unknown"))
    import cv2

    return f"cv2-{cv2.__version__}"


def resolve_backend(requested: str = "auto") -> tuple[Backend, str]:
    """The backend to use and a note on why. `auto` prefers Augraphy and falls back to OpenCV."""
    if requested == "opencv":
        return "opencv", "OpenCV backend (requested)"
    try:
        import augraphy  # noqa: F401
    except Exception as exc:
        if requested == "augraphy":
            raise RuntimeError(
                f"the Augraphy backend was requested but cannot be used: {exc}"
            ) from exc
        return "opencv", f"OpenCV backend (Augraphy is not usable here: {type(exc).__name__})"
    return "augraphy", "Augraphy backend"


def degrade_image(image: Image.Image, level: Variant, seed: int, backend: Backend) -> Image.Image:
    """One degraded page, RGB, and exactly the size of `image`."""
    degrade = degrade_augraphy if backend == "augraphy" else degrade_opencv
    result = degrade(image, level, seed)
    if result.size != image.size:
        raise RuntimeError(
            f"the {backend} backend changed the page size {image.size} -> {result.size}"
        )
    return result


# ---------------------------------------------------------------- documents


@dataclass(frozen=True)
class VariantPage:
    n: int
    file: str
    format: str
    size: tuple[int, int]
    sha256: str
    bytes: int
    source_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "file": self.file,
            "format": self.format,
            "size": list(self.size),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "VariantPage":
        return cls(
            n=int(raw["n"]),
            file=str(raw["file"]),
            format=str(raw["format"]),
            size=(int(raw["size"][0]), int(raw["size"][1])),
            sha256=str(raw["sha256"]),
            bytes=int(raw["bytes"]),
            source_sha256=str(raw["source_sha256"]),
        )


@dataclass(frozen=True)
class VariantResult:
    doc_id: str
    status: Literal["created", "unchanged", "would_create"]
    pages: tuple[VariantPage, ...]
    backend: Backend


_EXTENSION = {"png": "png", "jpeg": "jpg"}


def _source_pages(source_id: str, source_dir: Path) -> list[dict[str, Any]]:
    """The recorded pages of an ingested source, verified to be intact on disk."""
    sidecar = read_sidecar(source_dir)
    if not sidecar or not sidecar.get("pages"):
        raise DegradeError(source_id, f"not ingested yet (no {INGEST_SIDECAR} in {source_dir})")
    pages: list[dict[str, Any]] = sidecar["pages"]
    for page in pages:
        try:
            actual = sha256_hex((source_dir / page["file"]).read_bytes())
        except OSError:
            raise DegradeError(source_id, f"stored page {page['file']} is missing") from None
        if actual != page["sha256"]:
            raise DegradeError(
                source_id, f"stored page {page['file']} no longer matches its ingest.json"
            )
    return pages


def _inputs_hash(
    pages: list[dict[str, Any]],
    level: str,
    master: str,
    backend: Backend,
    version: str,
    quality: int,
) -> str:
    recipe = {
        "degrade_version": DEGRADE_VERSION,
        "backend": backend,
        "backend_version": version,
        "level": level,
        "seed": master,
        "jpeg_quality": quality,
        "pages": [[p["sha256"], p["format"]] for p in pages],
    }
    return sha256_hex(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())


def _recorded(out_dir: Path, inputs_hash: str) -> tuple[VariantPage, ...] | None:
    sidecar = read_sidecar_named(out_dir)
    if not sidecar or sidecar.get("degrade_version") != DEGRADE_VERSION:
        return None
    if sidecar.get("inputs_hash") != inputs_hash:
        return None
    try:
        records = tuple(VariantPage.from_json(p) for p in sidecar["pages"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    for record in records:
        try:
            if sha256_hex((out_dir / record.file).read_bytes()) != record.sha256:
                return None
        except OSError:
            return None
    return records


def read_sidecar_named(directory: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads((directory / SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def degrade_document(
    source: ManifestRow,
    level: Variant,
    *,
    pages_dir: Path,
    seed: str = DEFAULT_SEED,
    backend: Backend = "opencv",
    jpeg_quality: int = 92,
    force: bool = False,
    dry_run: bool = False,
) -> VariantResult:
    """Write `<source>-deg-<level>` pages next to the source's, from the source's stored pages."""
    if source.is_derived:
        raise DegradeError(source.doc_id, "is itself a variant; degrade its source instead")
    if source.capture != "born_digital":
        raise DegradeError(
            source.doc_id,
            f"capture is {source.capture or 'blank'}: only born_digital documents are degraded "
            "(a photo is already degraded)",
        )
    pages = _source_pages(source.doc_id, pages_dir / source.doc_id)
    name = variant_id(source.doc_id, level)
    out_dir = pages_dir / name
    inputs_hash = _inputs_hash(pages, level, seed, backend, backend_version(backend), jpeg_quality)
    if not force:
        recorded = _recorded(out_dir, inputs_hash)
        if recorded is not None:
            return VariantResult(name, "unchanged", recorded, backend)
    if dry_run:
        return VariantResult(name, "would_create", (), backend)

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[VariantPage] = []
    for page in pages:
        number = int(page["n"])
        with Image.open(pages_dir / source.doc_id / page["file"]) as opened:
            original = flatten(opened)
        try:
            degraded = degrade_image(
                original, level, derive_seed(seed, source.doc_id, level, number), backend
            )
        except RuntimeError as exc:
            raise DegradeError(source.doc_id, f"page {number}: {exc}") from exc
        fmt = str(page["format"])
        data = encode_page(degraded, fmt, jpeg_quality=jpeg_quality)
        file = f"p{number}.{_EXTENSION[fmt]}"
        write_atomic(out_dir / file, data)
        records.append(
            VariantPage(
                n=number,
                file=file,
                format=fmt,
                size=degraded.size,
                sha256=sha256_hex(data),
                bytes=len(data),
                source_sha256=str(page["sha256"]),
            )
        )
    keep = {record.file for record in records}
    for stale in out_dir.iterdir():
        if stale.name not in keep and stale.name != SIDECAR and stale.suffix in {".png", ".jpg"}:
            stale.unlink()
    sidecar = {
        "degrade_version": DEGRADE_VERSION,
        "inputs_hash": inputs_hash,
        "source": source.doc_id,
        "level": level,
        "seed": seed,
        "backend": backend,
        "backend_version": backend_version(backend),
        "pages": [record.to_json() for record in records],
    }
    write_atomic(out_dir / SIDECAR, (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode())
    return VariantResult(name, "created", tuple(records), backend)


# ---------------------------------------------------------------- the manifest


def add_variant_rows(
    rows: list[ManifestRow], wanted: list[tuple[ManifestRow, Variant]]
) -> list[ManifestRow]:
    """`rows` plus a row for each wanted (source, level), placed right after its source.

    A row that already exists must be the variant it claims to be; it is left as it is."""
    by_id = {row.doc_id: row for row in rows}
    additions: dict[str, list[ManifestRow]] = {}
    for source, level in wanted:
        expected = variant_row(source, level)
        existing = by_id.get(expected.doc_id)
        if existing is None:
            additions.setdefault(source.doc_id, []).append(expected)
        elif (existing.gt_from, existing.variant) != (expected.gt_from, expected.variant):
            raise DegradeError(
                expected.doc_id,
                "already in the manifest as something else "
                f"(gt_from {existing.gt_from!r}, variant {existing.variant!r})",
            )
    result: list[ManifestRow] = []
    for row in rows:
        result.append(row)
        result.extend(additions.get(row.doc_id, []))
    return result
