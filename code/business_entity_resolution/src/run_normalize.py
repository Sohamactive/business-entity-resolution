"""One-command normalization of all six raw TSVs to Parquet caches.

Example (run from the code/ directory):
  python -m business_entity_resolution.src.run_normalize --force
On Kaggle (dataset mounted read-only), point at the mount path:
  python -m business_entity_resolution.src.run_normalize \
      --dataset-dir /kaggle/input/<dataset-name>/dataset --force
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from business_entity_resolution.src.normalization import normalize_tsv_to_parquet

FILES = [
    ("test", "test_source1.tsv", "test_source1.parquet"),
    ("test", "test_source2.tsv", "test_source2.parquet"),
    ("test", "test_source3.tsv", "test_source3.parquet"),
    ("train", "train_source1.tsv", "train_source1.parquet"),
    ("train", "train_source2.tsv", "train_source2.parquet"),
    ("train", "train_source3.tsv", "train_source3.parquet"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize all raw TSVs to Parquet.")
    p.add_argument("--dataset-dir", type=Path, default=Path("../data/student_resource/dataset"))
    p.add_argument("--cache-dir", type=Path,
                   default=Path("business_entity_resolution/src/cache/normalized"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    for split, src, dest in FILES:
        out = normalize_tsv_to_parquet(
            args.dataset_dir / split / src, args.cache_dir / dest, force=args.force)
        print(f"wrote {out}")
    print(f"TOTAL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
