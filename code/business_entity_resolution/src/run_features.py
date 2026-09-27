"""CLI runner for the full-scale feature pipeline.

Examples (run from the code/ directory with the project .venv Python):

  # Trial on the first 2 candidate parts (~13M pairs, a few minutes):
  python -m business_entity_resolution.src.run_features --limit-parts 2 --workers 10

  # Full run over all 35 parts (resume-safe; rerun to continue):
  python -m business_entity_resolution.src.run_features --workers 10

  # Revert to the original single-threaded loop:
  python -m business_entity_resolution.src.run_features --workers 1
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from business_entity_resolution.src.features import run_feature_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched TF-IDF feature scoring.")
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        default=Path("business_entity_resolution/src/cache/candidates/test_pairs"),
        help="Long-form candidate ID parts from blocking",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("business_entity_resolution/src/cache/normalized"),
        help="Normalized source Parquet caches",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("business_entity_resolution/src/cache/features/test_features"),
        help="Where to write scored feature parts",
    )
    parser.add_argument(
        "--vectorizer-dir",
        type=Path,
        default=Path("business_entity_resolution/src/cache/vectorizers"),
        help="Where to save the fitted TF-IDF vectorizers",
    )
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--fit-sample-rows", type=int, default=500_000)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Worker count; 1 = original single-threaded loop",
    )
    parser.add_argument(
        "--backend", choices=["threads", "processes"], default="threads",
        help="'threads' shares memory (safe, GIL-capped); 'processes' tokenizes "
        "in parallel (fast, needs memory headroom)",
    )
    parser.add_argument(
        "--limit-parts",
        type=int,
        default=None,
        help="Score only the first N candidate parts (trial runs)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore already-written feature parts and redo them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_dir: Path = args.candidate_cache_dir

    if args.limit_parts is not None:
        # Stage a trial directory with symlinks/copies avoided: copy only
        # the part FILE LIST by pointing the pipeline at a temp dir of links.
        # On Windows without symlink rights, fall back to a real staging dir
        # listing — simplest robust approach: temporarily run against a
        # filtered glob by monkey-patching the glob below.
        trial_dir = Path(tempfile.mkdtemp(prefix="trial_parts_"))
        for part in sorted(candidate_dir.glob("part-*.parquet"))[: args.limit_parts]:
            link = trial_dir / part.name
            try:
                link.symlink_to(part.resolve())
            except OSError:
                shutil.copy2(part, link)
        candidate_dir = trial_dir
        print(f"Trial mode: staging first {args.limit_parts} parts in {trial_dir}")

    t0 = time.time()
    run_feature_pipeline(
        candidate_cache_dir=candidate_dir,
        normalized_dir=args.normalized_dir,
        feature_cache_dir=args.feature_cache_dir,
        vectorizer_dir=args.vectorizer_dir,
        batch_size=args.batch_size,
        fit_sample_rows=args.fit_sample_rows,
        workers=args.workers,
        resume=not args.no_resume,
        backend=args.backend,
    )
    print(f"TOTAL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
