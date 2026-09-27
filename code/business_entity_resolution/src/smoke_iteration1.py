"""Smoke test for Iteration 1 on the sample_pairs cache.

Exercises the full features -> matching path on ~630k pairs before
anything touches the 224M-pair full run. Prints score distributions
and acceptance counts across thresholds to preview tuning.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

from business_entity_resolution.src.features import (
    fit_vectorizers,
    load_fit_corpus,
    save_vectorizers,
    score_pair_batch,
)
from business_entity_resolution.src.matching import group_matches

BASE = Path(__file__).resolve().parent
CANDIDATE_DIR = BASE / "cache" / "candidates" / "sample_pairs"
NORMALIZED_DIR = BASE / "cache" / "normalized"
SMOKE_DIR = BASE / "cache" / "smoke"
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def main() -> None:
    t0 = time.time()
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Fitting vectorizers on a small sample...")
    names, addresses = load_fit_corpus(NORMALIZED_DIR, sample_rows=30_000)
    name_vec, address_vec = fit_vectorizers(names, addresses)
    save_vectorizers(name_vec, address_vec, SMOKE_DIR / "vectorizers")
    print(f"   name vocab={len(name_vec.vocabulary_)} address vocab={len(address_vec.vocabulary_)}")

    print("2. Loading normalized caches...")
    s1 = pl.scan_parquet(NORMALIZED_DIR / "test_source1.parquet").select(
        "entity_id", "country", "normalized_name", "normalized_address"
    ).collect()
    pool = pl.concat(
        [
            pl.scan_parquet(NORMALIZED_DIR / "test_source2.parquet").select(
                "entity_id", "country", "normalized_name", "normalized_address"
            ).collect(),
            pl.scan_parquet(NORMALIZED_DIR / "test_source3.parquet").select(
                "entity_id", "country", "normalized_name", "normalized_address"
            ).collect(),
        ]
    )
    print(f"   S1={len(s1)} pool={len(pool)}")

    print("3. Scoring sample parts...")
    scored_parts: list[pl.DataFrame] = []
    for part_path in sorted(CANDIDATE_DIR.glob("part-*.parquet")):
        pairs = pl.read_parquet(part_path).select("source1_entity_id", "candidate_id")
        joined = (
            pairs.join(
                s1.select(
                    pl.col("entity_id").alias("source1_entity_id"),
                    pl.col("country").alias("s1_country"),
                    pl.col("normalized_name").alias("s1_norm_name"),
                    pl.col("normalized_address").alias("s1_norm_address"),
                ),
                on="source1_entity_id",
                how="left",
            )
            .join(
                pool.select(
                    pl.col("entity_id").alias("candidate_id"),
                    pl.col("country").alias("cand_country"),
                    pl.col("normalized_name").alias("cand_norm_name"),
                    pl.col("normalized_address").alias("cand_norm_address"),
                ),
                on="candidate_id",
                how="left",
            )
            .fill_null("")
        )
        miss_s1 = joined.filter(pl.col("s1_norm_name") == "").height
        miss_c = joined.filter(pl.col("cand_norm_name") == "").height
        scored = score_pair_batch(joined, name_vec, address_vec)
        scored_parts.append(scored)
        print(f"   {part_path.name}: pairs={len(pairs)} join-miss s1={miss_s1} cand={miss_c}")

    scored_all = pl.concat(scored_parts)
    print(f"4. Total scored pairs: {len(scored_all)}")
    for col in ("combined_score", "name_similarity", "address_similarity"):
        qs = [scored_all[col].quantile(q) for q in (0.5, 0.9, 0.95, 0.99)]
        print(
            f"   {col}: p50={qs[0]:.4f} p90={qs[1]:.4f} p95={qs[2]:.4f} p99={qs[3]:.4f} "
            f"mean={scored_all[col].mean():.4f}"
        )

    print("5. Acceptance preview across thresholds...")
    s1_ids = scored_all["source1_entity_id"].unique().to_list()
    for threshold in THRESHOLDS:
        above = scored_all.filter(pl.col("combined_score") > threshold).select(
            "source1_entity_id", "candidate_id"
        )
        grouped = group_matches(above, s1_ids)
        n_non_single = grouped.filter(pl.col("matched_entity_ids") != "").height
        avg_cands = len(above) / len(s1_ids)
        print(
            f"   threshold>{threshold:.2f}: accepted={len(above)} "
            f"non-singleton S1={n_non_single}/{len(s1_ids)} avg-cands/S1={avg_cands:.2f}"
        )

    print("6. Top-10 accepted pair scores at threshold>0.50...")
    for row in (
        scored_all.filter(pl.col("combined_score") > 0.50)
        .sort("combined_score", descending=True)
        .head(10)
        .select("name_similarity", "address_similarity", "combined_score")
        .iter_rows(named=True)
    ):
        print(f"   name={row['name_similarity']:.4f} addr={row['address_similarity']:.4f} combined={row['combined_score']:.4f}")
    print(f"SMOKE COMPLETE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
