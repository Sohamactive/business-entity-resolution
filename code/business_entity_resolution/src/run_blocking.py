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
    block_source1_against_index,
    save_candidate_pairs,
)
from business_entity_resolution.src.normalization import normalize_dataframe

load_dotenv()


def run_blocking_pipeline(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    output_path: Path,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    limit: int | None = None,
) -> Path:
    """Execute end-to-end blocking: load -> normalize -> index -> block -> save."""
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

    # 1. Load data
    print("1. Loading datasets with Polars...")
    t0 = time.time()
    s1_scan = pl.scan_csv(source1_path, separator="\t")
    s2_scan = pl.scan_csv(source2_path, separator="\t")
    s3_scan = pl.scan_csv(source3_path, separator="\t")

    if limit:
        s1_df = s1_scan.head(limit).collect()
        s2_df = s2_scan.head(limit * 2).collect()
        s3_df = s3_scan.head(limit * 2).collect()
    else:
        s1_df = s1_scan.collect()
        s2_df = s2_scan.collect()
        s3_df = s3_scan.collect()

    print(f"   Loaded {len(s1_df)} S1, {len(s2_df)} S2, {len(s3_df)} S3 in {time.time() - t0:.2f}s")

    # 2. Normalize
    print("2. Normalizing textual fields (script-blind NFKC + legal expansion)...")
    t0 = time.time()
    s1_norm = normalize_dataframe(s1_df)
    s2_norm = normalize_dataframe(s2_df)
    s3_norm = normalize_dataframe(s3_df)
    print(f"   Normalized {len(s1_norm) + len(s2_norm) + len(s3_norm)} records in {time.time() - t0:.2f}s")

    # 3. Build inverted index on S2 + S3 pool
    print("3. Building inverted index on Source 2 and Source 3 pool...")
    t0 = time.time()
    index = BlockingIndex()
    index.add_dataframe(s2_norm)
    index.add_dataframe(s3_norm)
    print(f"   Indexed pool in {time.time() - t0:.2f}s")

    # 4. Generate candidates for Source 1
    print("4. Querying blocking keys and generating candidate pairs...")
    t0 = time.time()
    cand_df = block_source1_against_index(
        s1_norm,
        index=index,
        max_candidates=max_candidates,
    )
    print(f"   Blocking completed in {time.time() - t0:.2f}s ({len(s1_norm) / (time.time() - t0):.0f} S1/sec)")

    # 5. Save candidate_pairs.tsv
    print("5. Saving output TSV...")
    t0 = time.time()
    out = save_candidate_pairs(cand_df, output_path)
    print(f"   Saved {len(cand_df)} rows to {out} in {time.time() - t0:.2f}s")

    print("-" * 70)
    print(f"PIPELINE COMPLETE in {time.time() - total_start:.2f}s")
    print("=" * 70)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blocking candidate generation pipeline.")
    parser.add_argument(
        "--source1",
        type=Path,
        default=Path("data/student_resource/dataset/test/test_source1.tsv"),
        help="Path to Source 1 TSV",
    )
    parser.add_argument(
        "--source2",
        type=Path,
        default=Path("data/student_resource/dataset/test/test_source2.tsv"),
        help="Path to Source 2 TSV",
    )
    parser.add_argument(
        "--source3",
        type=Path,
        default=Path("data/student_resource/dataset/test/test_source3.tsv"),
        help="Path to Source 3 TSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/candidate_pairs.tsv"),
        help="Path to save candidate_pairs.tsv",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Maximum candidates per Source 1 entity",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows for testing/prototyping",
    )

    args = parser.parse_args()

    run_blocking_pipeline(
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        output_path=args.output,
        max_candidates=args.max_candidates,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
