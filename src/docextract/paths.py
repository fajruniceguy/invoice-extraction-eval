"""Where the dataset lives. One place, so scripts and tests agree and tests can use a tmp root.

data/private/raw/       real originals                    (gitignored, never committed)
data/private/preview/   UNREDACTED normalized pages       (gitignored; for finding redaction boxes)
data/pages/<doc_id>/    normalized + redacted pages       (committed)
data/manifest.csv       one row per logical document      (committed)
data/redactions.yaml    redaction boxes per doc and page  (committed)
ground_truth/           labels, drafts/, CHANGELOG.md, LABELING.md (committed)
"""

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @classmethod
    def default(cls) -> "DatasetPaths":
        return cls(find_repo_root())

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "private" / "raw"

    @property
    def preview_dir(self) -> Path:
        return self.data_dir / "private" / "preview"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def manifest(self) -> Path:
        return self.data_dir / "manifest.csv"

    @property
    def redactions(self) -> Path:
        return self.data_dir / "redactions.yaml"

    @property
    def ground_truth_dir(self) -> Path:
        return self.root / "ground_truth"

    @property
    def drafts_dir(self) -> Path:
        return self.ground_truth_dir / "drafts"

    @property
    def changelog(self) -> Path:
        return self.ground_truth_dir / "CHANGELOG.md"
