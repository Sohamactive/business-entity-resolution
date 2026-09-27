"""Apply the trained LightGBM matcher to test feature parts (Iteration 2).

Streams the cached test feature parts, predicts match probabilities in
bounded batches, keeps pairs above the calibrated threshold, groups by
Source 1 entity, and writes matching_results.tsv.

Example: python -m business_entity_resolution.src.apply_matcher
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl
from tqdm import tqdm

from business_entity_resolution.src.embed_features import EMB_COL, embedding_similarity, load_vectors
from business_entity_resolution.src.matching import group_matches, write_matching_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply LightGBM matcher to test features.")
    p.add_argument("--feature-cache-dir", type=Path,
                   default=Path("business_entity_resolution/src/cache/features/test_features"))
    p.add_argument("--normalized-dir", type=Path,
                   default=Path("business_entity_resolution/src/cache/normalized"))
    p.add_argument("--model-dir", type=Path,
                   default=Path("business_entity_resolution/src/cache/matcher"))
    p.add_argument("--output", type=Path, default=Path("../output/matching_results.tsv"))
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the calibrated threshold from training")
    p.add_argument("--embed-vec-dir", type=Path, default=None,
                   help="Directory with name_vectors.npy + name_index.pkl; "
                   "adds emb_name_similarity before predicting")
    p.add_argument("--save-probas", type=Path, default=None,
                   help="Write (source1_entity_id, candidate_id, proba) for pairs "
                   "above --proba-floor, enabling instant threshold variants later")
    p.add_argument("--proba-floor", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    threshold = args.threshold
    if threshold is None:
        threshold = float((args.model_dir / "threshold.txt").read_text().strip())
    print(f"Threshold: {threshold}")

    model = lgb.Booster(model_file=str(args.model_dir / "matcher.txt"))
    cols_file = args.model_dir / "feature_cols.json"
    feature_cols = json.loads(cols_file.read_text()) if cols_file.exists() else [
        "name_similarity", "address_similarity",
        "name_is_missing", "address_is_missing",
        "country_match", "combined_score",
    ]
    print(f"Predicting with features: {feature_cols}")
    parts = sorted(args.feature_cache_dir.glob("features-*.parquet"))
    print(f"Predicting {len(parts)} feature parts...")

    emb_index = emb_matrix = s1_names_map = pool_names_map = None
    if args.embed_vec_dir is not None:
        emb_index, emb_matrix = load_vectors(args.embed_vec_dir)
        s1 = pl.scan_parquet(args.normalized_dir / "test_source1.parquet").select(
            "entity_id", "normalized_name").collect()
        s1_names_map = dict(zip(s1["entity_id"].to_list(), s1["normalized_name"].fill_null("").to_list()))
        del s1
        pool = pl.concat([
            pl.scan_parquet(args.normalized_dir / "test_source2.parquet").select(
                "entity_id", "normalized_name").collect(),
            pl.scan_parquet(args.normalized_dir / "test_source3.parquet").select(
                "entity_id", "normalized_name").collect(),
        ])
        pool_names_map = dict(zip(pool["entity_id"].to_list(), pool["normalized_name"].fill_null("").to_list()))
        del pool
        print(f"  name maps: s1={len(s1_names_map)} pool={len(pool_names_map)}")

    accepted_chunks: list[pl.DataFrame] = []
    proba_chunks: list[pl.DataFrame] = []
    n_scored = 0
    for part_path in tqdm(parts, desc="predicting", unit="part"):
        batch = pl.read_parquet(part_path)
        n_scored += len(batch)
        if emb_index is not None:
            s1_list = [s1_names_map.get(i, "") for i in batch["source1_entity_id"].to_list()]
            c_list = [pool_names_map.get(i, "") for i in batch["candidate_id"].to_list()]
            batch = batch.with_columns(
                pl.Series(EMB_COL, embedding_similarity(s1_list, c_list, emb_index, emb_matrix)))
        proba = model.predict(batch.select(feature_cols).to_numpy())
        keep = batch.filter(pl.Series("keep", proba > threshold)).select(
            "source1_entity_id", "candidate_id")
        if len(keep):
            accepted_chunks.append(keep)
        if args.save_probas is not None:
            floor = batch.filter(pl.Series("keep", proba > args.proba_floor)).select(
                "source1_entity_id", "candidate_id")
            if len(floor):
                proba_chunks.append(floor.with_columns(pl.Series("proba", proba[proba > args.proba_floor].round(4))))
    if args.save_probas is not None and proba_chunks:
        args.save_probas.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(proba_chunks).write_parquet(args.save_probas)
        print(f"Saved proba cache to {args.save_probas}")

    accepted = pl.concat(accepted_chunks) if accepted_chunks else pl.DataFrame(
        schema={"source1_entity_id": pl.String, "candidate_id": pl.String})
    accepted = accepted.unique(subset=["source1_entity_id", "candidate_id"])
    print(f"Scored {n_scored} pairs, accepted {len(accepted)}")

    s1_ids = (pl.scan_parquet(args.normalized_dir / "test_source1.parquet")
              .select("entity_id").collect()["entity_id"].to_list())
    matching_df = group_matches(accepted, s1_ids)
    write_matching_results(matching_df, args.output)
    n_non = matching_df.filter(pl.col("matched_entity_ids") != "").height
    print(f"Wrote {len(matching_df)} rows ({n_non} non-singleton) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
