"""Make the hallucination probe set (S4): constructed invoices with fields erased.

Usage: python -m uv run python scripts/make_probes.py [--check] [--force] [--seed TEXT]
           [--probes-dir DIR] [--dataset-pages DIR] [--raw-dir DIR]

Writes data/probes/: manifest.csv, probes.json (what was erased, where, the page hashes),
labels/<id>.json (erased fields are null) and pages/<id>/pN.png + ingest.json. Nothing is read from
a real document: the sources are cons_001..cons_003, regenerated from `evalkit.construct`. The
controls' pages and the pages a dropped last page keeps must equal data/pages/ by hash, and the
regenerated PDFs must equal data/private/raw/constructed/ when those exist: otherwise the probes
are not copies of the dataset's documents and nothing is written.

Deterministic: a second run is byte-identical. Existing probes that differ from a regeneration
(another seed, another renderer) are never replaced silently: --force replaces them.
--check writes nothing: it reports whether data/probes/ is what a regeneration gives.
Exit codes: 0 done (or identical), 1 differs / refused / a probe cannot be made, 2 bad arguments.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from docextract.paths import DatasetPaths
from evalkit.construct import DEFAULT_SEED
from evalkit.probes import ProbeError, check_sources, tree_hashes, write_probe_set


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    paths = DatasetPaths.default()
    parser = argparse.ArgumentParser(description="Make the hallucination probe set.")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--probes-dir", type=Path, default=paths.data_dir / "probes")
    parser.add_argument("--dataset-pages", type=Path, default=paths.pages_dir)
    parser.add_argument("--raw-dir", type=Path, default=paths.raw_dir)
    parser.add_argument("--force", action="store_true", help="replace probes that differ")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)

    problems = check_sources(args.raw_dir, args.seed)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="probes-") as work:
        fresh_dir = Path(work) / "probes"
        try:
            document = write_probe_set(fresh_dir, seed=args.seed, dataset_pages=args.dataset_pages)
        except ProbeError as exc:
            print(f"make_probes: {exc}", file=sys.stderr)
            return 1
        fresh = tree_hashes(fresh_dir)
        existing = tree_hashes(args.probes_dir) if args.probes_dir.exists() else {}
        erased = sum(len(entry["erased"]) for entry in document["probes"])
        print(
            f"make_probes: seed {args.seed!r}, {len(document['probes'])} probes "
            f"({sum(1 for e in document['probes'] if e['kind'] != 'control')} with something taken "
            f"away, {erased} erased field(s)), {len(fresh)} file(s)"
        )
        if existing == fresh:
            print(f"identical: {args.probes_dir} is what a regeneration gives")
            return 0
        if args.check:
            changed = sorted(
                set(existing) ^ set(fresh) | {n for n in fresh if existing.get(n) != fresh[n]}
            )
            print(f"DIFFERS from a regeneration in {len(changed)} file(s), e.g. {changed[:3]}")
            return 1
        if existing and not args.force:
            print(
                f"{args.probes_dir} exists and differs from a regeneration: --force replaces it",
                file=sys.stderr,
            )
            return 1
        if args.probes_dir.exists():
            shutil.rmtree(args.probes_dir)
        shutil.copytree(fresh_dir, args.probes_dir)
    print(f"written: {args.probes_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
