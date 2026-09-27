"""Threshold tuning on a train holdout with ground truth (Iteration 1).

Pipeline: sample train S1 + ground truth -> block (true matches +
distractors) -> score pairs with the ALREADY-FITTED test vectorizers ->
sweep thresholds against macro F_0.5 overall and per country.

The vectorizers are unsupervised (fit on normalized text, never labels),
so reusing the test-fitted vectorizers on train holdout pairs is valid
and guarantees the tuned threshold applies to identically-scaled scores.

Example (run from the code/ directory):
  python -m business_entity_resolution.src.run_tuning --num-s1 2000 --workers 1
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import polars as pl

from business_entity_resolution.src.blocking import build_candidate_pairs_table
from business_entity_resolution.src.evaluate_blocking import evaluate_blocking
from business_entity_resolution.src.features import load_vectorizers, score_pair_batch
from business_entity_resolution.src.matching import group_matches, macro_f05
from business_entity_resolution.src.normalization import normalize_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the match threshold on a train holdout.")
    parser.add_argument("--num-s1", type=int, default=2000)
    parser.add_argument("--num-distractors", type=int, default=50000)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("../data/student_resource"),
        help="student_resource directory (default assumes running from code/)",
    )
    parser.add_argument(
        "--vectorizer-dir", type=Path,
        default=Path("business_entity_resolution/src/cache/vectorizers"),
    )
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("1. Generating holdout candidates via blocking...")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = evaluate_blocking(
            data_dir=args.data_dir,
            num_s1=args.num_s1,
            num_distractors=args.num_distractors,
        )
    cand_df = result["candidate_df"]
    print(f"   blocking recall on holdout: {result['overall_recall']:.2f}%")

    print("2. Rebuilding holdout pair table with normalized text...")
    train_dir = args.data_dir / "dataset" / "train"
    s1_df = normalize_dataframe(
        pl.scan_csv(train_dir / "train_source1.tsv", separator="\t").head(args.num_s1).collect()
    )
    gt_df = pl.scan_csv(train_dir / "train_ground_truth.tsv", separator="\t").collect()
    gt_map: dict[str, set[str]] = {}
    for row in gt_df.filter(pl.col("source1_entity_id").is_in(s1_df["entity_id"].to_list())).iter_rows(named=True):
        targets = {m.strip() for m in (row["matched_entity_ids"] or "").split(",") if m.strip()}
        gt_map[row["source1_entity_id"]] = targets
    for s1_id in s1_df["entity_id"].to_list():
        gt_map.setdefault(s1_id, set())

    # Restrict the pool to IDs that actually appear as holdout candidates
    # instead of scanning all ~10M train pool rows into memory.
    cand_ids: set[str] = set()
    for cell in cand_df["candidate_entity_ids"].to_list():
        if cell:
            cand_ids.update(c.strip() for c in cell.split(",") if c.strip())
    pool_df = normalize_dataframe(
        pl.concat(
            [
                pl.scan_csv(train_dir / "train_source2.tsv", separator="\t")
                .filter(pl.col("entity_id").is_in(list(cand_ids))).collect(),
                pl.scan_csv(train_dir / "train_source3.tsv", separator="\t")
                .filter(pl.col("entity_id").is_in(list(cand_ids))).collect(),
            ]
        ).unique(subset=["entity_id"])
    )
    print(f"   restricted pool rows: {len(pool_df)}")

    pair_table = build_candidate_pairs_table(cand_df, s1_df=s1_df, pool_df=pool_df, include_normalized=True)
    print(f"   holdout pairs to score: {len(pair_table)}")

    print("3. Scoring holdout pairs with fitted vectorizers...")
    name_vec, address_vec = load_vectorizers(args.vectorizer_dir)
    renamed = pair_table.rename({
        "s1_name": "s1_name_raw", "s1_address": "s1_address_raw",
        "cand_name": "cand_name_raw", "cand_address": "cand_address_raw",
        "s1_norm_name": "s1_norm_name", "s1_norm_address": "s1_norm_address",
        "cand_norm_name": "cand_norm_name", "cand_norm_address": "cand_norm_address",
    })
    scored = score_pair_batch(renamed, name_vec, address_vec)

    countries = dict(zip(s1_df["entity_id"].to_list(), s1_df["country"].to_list()))
    s1_ids = list(gt_map.keys())

    print("4. Sweeping thresholds (overall + per country)...")
    print(f"   {'thr':>5} {'F0.5':>7} {'Prec':>7} {'Rec':>7} | per-country F0.5")
    best_thr, best_f05 = 0.0, -1.0
    for thr in args.thresholds:
        above = scored.filter(pl.col("combined_score") > thr).select("source1_entity_id", "candidate_id")
        grouped = group_matches(above, s1_ids)
        predicted = {
            row["source1_entity_id"]: (set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set())
            for row in grouped.iter_rows(named=True)
        }
        f05, prec, rec = macro_f05(predicted, gt_map)
        by_country: dict[str, float] = {}
        for country in sorted(set(countries.values())):
            sub_gt = {sid: gt_map[sid] for sid in s1_ids if countries.get(sid) == country}
            if sub_gt:
                sub_pred = {sid: predicted[sid] for sid in sub_gt}
                by_country[country] = macro_f05(sub_pred, sub_gt)[0]
        detail = " ".join(f"{c}={v:.3f}" for c, v in by_country.items())
        print(f"   {thr:>5.2f} {f05:>7.4f} {prec:>7.4f} {rec:>7.4f} | {detail}")
        if f05 > best_f05:
            best_f05, best_thr = f05, thr
    print(f"RECOMMENDED threshold: {best_thr:.2f} (holdout macro F_0.5={best_f05:.4f})")


if __name__ == "__main__":
    main()
