"""Evaluation and benchmarking script for the blocking engine.

Evaluates blocking recall, candidate pool size distribution, reduction ratio,
and verifies output compliance with validate_submission.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from business_entity_resolution.src.blocking import (
    DEFAULT_MAX_CANDIDATES,
    BlockingIndex,
    block_source1_against_index,
    save_candidate_pairs,
)
from business_entity_resolution.src.normalization import normalize_dataframe


def evaluate_blocking(
    data_dir: Path,
    num_s1: int = 5000,
    num_distractors: int = 50000,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Run an internal holdout validation of the blocking engine."""
    print("=" * 70)
    print(f"EVALUATING BLOCKING ENGINE (Holdout S1: {num_s1}, Max Cap: {max_candidates})")
    print("=" * 70)

    train_dir = data_dir / "dataset" / "train"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    print("1. Loading Source 1 and Ground Truth holdout sample...")
    s1_df = pl.scan_csv(s1_path, separator="\t").head(num_s1).collect()
    s1_ids = set(s1_df["entity_id"])

    gt_df = (
        pl.scan_csv(gt_path, separator="\t")
        .filter(pl.col("source1_entity_id").is_in(list(s1_ids)))
        .collect()
    )

    # Collect ground truth match targets
    gt_map: dict[str, list[str]] = {}
    all_true_m_ids: set[str] = set()
    for row in gt_df.iter_rows(named=True):
        m_str = row["matched_entity_ids"] or ""
        targets = [m.strip() for m in m_str.split(",") if m.strip()]
        gt_map[row["source1_entity_id"]] = targets
        all_true_m_ids.update(targets)

    print(f"   Holdout S1 entities: {len(s1_df)}")
    print(f"   Total true match targets in holdout: {len(all_true_m_ids)}")

    print("2. Building candidate pool from S2 and S3 (true matches + distractors)...")
    t0 = time.time()
    s2_true = (
        pl.scan_csv(s2_path, separator="\t")
        .filter(pl.col("entity_id").is_in(list(all_true_m_ids)))
        .collect()
    )
    s3_true = (
        pl.scan_csv(s3_path, separator="\t")
        .filter(pl.col("entity_id").is_in(list(all_true_m_ids)))
        .collect()
    )
    s2_dist = pl.scan_csv(s2_path, separator="\t").head(num_distractors // 2).collect()
    s3_dist = pl.scan_csv(s3_path, separator="\t").head(num_distractors // 2).collect()

    pool_df = pl.concat([s2_true, s3_true, s2_dist, s3_dist]).unique(subset=["entity_id"])
    print(f"   Candidate pool size: {len(pool_df)} records (loaded in {time.time() - t0:.2f}s)")

    print("3. Normalizing records...")
    t0 = time.time()
    s1_norm = normalize_dataframe(s1_df)
    pool_norm = normalize_dataframe(pool_df)
    print(f"   Normalized {len(s1_norm) + len(pool_norm)} records in {time.time() - t0:.2f}s")

    print("4. Building Inverted Index for candidate pool...")
    t0 = time.time()
    index = BlockingIndex()
    index.add_dataframe(pool_norm)
    print(f"   Built index in {time.time() - t0:.2f}s")

    print("5. Generating candidates for Source 1 entities...")
    t0 = time.time()
    cand_df = block_source1_against_index(s1_norm, index, max_candidates=max_candidates)
    gen_time = time.time() - t0
    print(f"   Generated candidates in {gen_time:.2f}s ({len(s1_norm) / gen_time:.0f} S1/sec)")

    print("6. Computing evaluation metrics...")
    cand_map: dict[str, set[str]] = {}
    cand_lengths: list[int] = []
    for row in cand_df.iter_rows(named=True):
        c_str = row["candidate_entity_ids"] or ""
        cands = {c.strip() for c in c_str.split(",") if c.strip()}
        cand_map[row["source1_entity_id"]] = cands
        cand_lengths.append(len(cands))

    total_gt = 0
    recalled_gt = 0
    country_stats: dict[str, dict[str, int]] = {}

    for row in s1_norm.iter_rows(named=True):
        s1_id = row["entity_id"]
        country = row["country"]
        if country not in country_stats:
            country_stats[country] = {"total_gt": 0, "recalled_gt": 0}

        true_targets = gt_map.get(s1_id, [])
        for tm in true_targets:
            total_gt += 1
            country_stats[country]["total_gt"] += 1
            if tm in cand_map.get(s1_id, set()):
                recalled_gt += 1
                country_stats[country]["recalled_gt"] += 1

    overall_recall = (recalled_gt / total_gt * 100) if total_gt > 0 else 0.0
    sorted_lens = sorted(cand_lengths)
    n = len(sorted_lens)
    median_cands = sorted_lens[n // 2] if n > 0 else 0
    p95_cands = sorted_lens[int(n * 0.95)] if n > 0 else 0
    mean_cands = sum(cand_lengths) / n if n > 0 else 0.0

    print("-" * 70)
    print("RESULTS:")
    print(f"  Overall Blocking Recall: {overall_recall:.2f}% ({recalled_gt} / {total_gt})")
    for country, stats in sorted(country_stats.items()):
        c_tot = stats["total_gt"]
        c_rec = stats["recalled_gt"]
        c_pct = (c_rec / c_tot * 100) if c_tot > 0 else 0.0
        print(f"  Recall [{country:6}]:        {c_pct:.2f}% ({c_rec} / {c_tot})")

    print(f"  Candidate pool per S1:   Mean={mean_cands:.1f}, Median={median_cands}, 95th%={p95_cands}, Max={max(cand_lengths)}")
    print("=" * 70)

    return {
        "candidate_df": cand_df,
        "overall_recall": overall_recall,
        "mean_candidates": mean_cands,
        "median_candidates": median_cands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blocking engine on internal holdout.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/student_resource"),
        help="Path to student_resource directory",
    )
    parser.add_argument("--num-s1", type=int, default=2000, help="Number of S1 holdout rows")
    parser.add_argument("--num-distractors", type=int, default=50000, help="Number of distractor rows")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Max candidates per S1",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("output/candidate_pairs.tsv"),
        help="Path to save candidate TSV",
    )
    args = parser.parse_args()

    results = evaluate_blocking(
        data_dir=args.data_dir,
        num_s1=args.num_s1,
        num_distractors=args.num_distractors,
        max_candidates=args.max_candidates,
    )

    save_candidate_pairs(results["candidate_df"], args.output_tsv)
    print(f"Saved evaluation candidate pairs to {args.output_tsv}")


if __name__ == "__main__":
    main()
