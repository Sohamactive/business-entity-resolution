"""Command-line runner to execute the blocking stage end-to-end.

Usage:
    python -m business_entity_resolution.src.run_blocking \
        --source1 data/student_resource/dataset/test/test_source1.tsv \
        --source2 data/student_resource/dataset/test/test_source2.tsv \
        --source3 data/student_resource/dataset/test/test_source3.tsv \
        --output output/candidate_pairs.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from business_entity_resolution.src.blocking import (
    DEFAULT_MAX_CANDIDATES,
    BlockingIndex,
    iter_blocking_candidate_batches,
    write_candidate_outputs_from_batches,
)

load_dotenv()


def run_blocking_pipeline(
    normalized_dir: Path,
    output_path: Path,
    candidate_cache_dir: Path,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    batch_size: int = 50_000,
    limit: int | None = None,
    force: bool = False,
) -> Path:
    """Execute blocking from normalized Parquet caches."""
    source1_path = normalized_dir / "test_source1.parquet"
    source2_path = normalized_dir / "test_source2.parquet"
    source3_path = normalized_dir / "test_source3.parquet"

    print("=" * 70)
    print("RUNNING BUSINESS ENTITY RESOLUTION BLOCKING PIPELINE")
    print(f"  Source 1:       {source1_path}")
    print(f"  Source 2:       {source2_path}")
    print(f"  Source 3:       {source3_path}")
    print(f"  Output:         {output_path}")
    print(f"  Max Candidates: {max_candidates} (configured via .env or CLI)")
    if limit:
        print(f"  Row Limit:      {limit} (debug mode)")
    print("=" * 70)

    total_start = time.time()

    # 1. Load normalized Parquet caches
    print("1. Loading normalized Parquet caches...")
    t0 = time.time()
    required_columns = [
        "entity_id",
        "country",
        "normalized_name",
        "normalized_address",
    ]
    s1_scan = pl.scan_parquet(source1_path).select(required_columns)
    s2_scan = pl.scan_parquet(source2_path).select(required_columns)
    s3_scan = pl.scan_parquet(source3_path).select(required_columns)

    if limit:
        s1_df = s1_scan.head(limit).collect(engine="streaming")
        s2_df = s2_scan.head(limit * 2).collect(engine="streaming")
        s3_df = s3_scan.head(limit * 2).collect(engine="streaming")
    else:
        s1_df = s1_scan.collect(engine="streaming")
        s2_df = s2_scan.collect(engine="streaming")
        s3_df = s3_scan.collect(engine="streaming")

    print(f"   Loaded {len(s1_df)} S1, {len(s2_df)} S2, {len(s3_df)} S3 in {time.time() - t0:.2f}s")

    # 2. Build inverted index on S2 + S3 pool
    print("2. Building inverted index on Source 2 and Source 3 pool...")
    t0 = time.time()
    index = BlockingIndex()
    index.add_dataframe(s2_df)
    index.add_dataframe(s3_df)
    print(f"   Indexed pool in {time.time() - t0:.2f}s")

    # 3. Generate and persist candidate batches for Source 1
    print("3. Querying blocking keys and generating candidate pairs...")
    t0 = time.time()
    candidate_batches = iter_blocking_candidate_batches(
        s1_df=s1_df,
        index=index,
        max_candidates=max_candidates,
        batch_size=batch_size,
    )
    out, cache_dir, pair_count = write_candidate_outputs_from_batches(
        candidate_batches=candidate_batches,
        s1_df=s1_df,
        pool_df=pl.concat([s2_df, s3_df]),
        candidate_tsv_path=output_path,
        pair_cache_dir=candidate_cache_dir,
        batch_size=batch_size,
        force=force,
    )
    blocking_seconds = time.time() - t0
    print(f"   Blocking and output completed in {blocking_seconds:.2f}s ({len(s1_df) / blocking_seconds:.0f} S1/sec)")
    print(f"   Saved {len(s1_df)} TSV rows to {out}")
    print(f"   Saved {pair_count} long-form pairs to {cache_dir}")

    print("-" * 70)
    print(f"PIPELINE COMPLETE in {time.time() - total_start:.2f}s")
    print("=" * 70)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blocking candidate generation pipeline.")
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("cache/normalized"),
        help="Directory containing normalized test Parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/candidate_pairs.tsv"),
        help="Path to save candidate_pairs.tsv",
    )
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        default=Path("cache/candidates/test_pairs"),
        help="Directory for long-form candidate Parquet parts",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Maximum candidates per Source 1 entity",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Number of compact candidate rows written per Parquet part",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows for testing/prototyping",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing candidate TSV and Parquet parts",
    )

    args = parser.parse_args()

    run_blocking_pipeline(
        normalized_dir=args.normalized_dir,
        output_path=args.output,
        candidate_cache_dir=args.candidate_cache_dir,
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
