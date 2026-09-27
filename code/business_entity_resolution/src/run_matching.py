"""CLI runner for threshold application and matching_results.tsv output.

Example (run from the code/ directory):
  python -m business_entity_resolution.src.run_matching --threshold 0.65
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from business_entity_resolution.src.matching import run_matching_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply threshold and write matching_results.tsv.")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--feature-cache-dir", type=Path,
        default=Path("business_entity_resolution/src/cache/features/test_features"),
    )
    parser.add_argument(
        "--normalized-dir", type=Path,
        default=Path("business_entity_resolution/src/cache/normalized"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("../output/matching_results.tsv"),
        help="Output path (default assumes running from code/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    run_matching_pipeline(
        feature_cache_dir=args.feature_cache_dir,
        normalized_dir=args.normalized_dir,
        output_path=args.output,
        threshold=args.threshold,
    )
    print(f"TOTAL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
