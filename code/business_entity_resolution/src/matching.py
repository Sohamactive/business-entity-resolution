"""Threshold decision and grouping into matching_results.tsv (Iteration 1).

Reads the cached feature parts written by `features.py`, keeps every
candidate pair whose combined_score is STRICTLY greater than the tuned
threshold, groups survivors by Source 1 entity, and writes the scored
submission file. An S1 entity may match zero, one, or many candidates;
this step never keeps only the top candidate.

Output guarantees (competition rules):
  - Exactly one row per test Source 1 entity; entities with no
    surviving candidates get a true empty match field.
  - No duplicate IDs within a list; matched IDs are S2-/S3- only.
  - Matched IDs always come from the blocking candidate set, because
    the feature parts were built from exactly that set.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _is_pool_id(candidate_id: str) -> bool:
    """Only Source 2/3 records may appear as matches (never S1- self-matches)."""
    return candidate_id.startswith("S2-") or candidate_id.startswith("S3-")


def load_accepted_pairs(feature_cache_dir: Path, threshold: float) -> pl.DataFrame:
    """Scan all feature parts and return pairs above the strict threshold."""
    accepted = (
        pl.scan_parquet(feature_cache_dir / "features-*.parquet")
        .filter(pl.col("combined_score") > threshold)
        .filter(
            pl.col("candidate_id").str.starts_with("S2-")
            | pl.col("candidate_id").str.starts_with("S3-")
        )
        .select("source1_entity_id", "candidate_id")
        .collect()
    )
    return accepted.unique(subset=["source1_entity_id", "candidate_id"])


def group_matches(
    accepted: pl.DataFrame, all_s1_ids: list[str]
) -> pl.DataFrame:
    """Group accepted pairs, guaranteeing one row per Source 1 entity."""
    if len(accepted) == 0:
        grouped = pl.DataFrame(
            {"source1_entity_id": [], "matched_entity_ids": []},
            schema={"source1_entity_id": pl.String, "matched_entity_ids": pl.String},
        )
    else:
        grouped = (
            accepted.group_by("source1_entity_id")
            .agg(
                pl.col("candidate_id").unique().sort().str.join(",").alias("matched_entity_ids")
            )
        )
    full = pl.DataFrame({"source1_entity_id": all_s1_ids}).join(
        grouped, on="source1_entity_id", how="left"
    )
    return full.select(
        "source1_entity_id",
        pl.col("matched_entity_ids").fill_null(""),
    )


def write_matching_results(matching_df: pl.DataFrame, output_path: Path) -> Path:
    """Write matching_results.tsv with true empty fields for singletons."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matching_df.write_csv(
        output_path,
        separator="\t",
        include_header=True,
        quote_style="never",
        null_value="",
    )
    return output_path


def run_matching_pipeline(
    feature_cache_dir: Path,
    normalized_dir: Path,
    output_path: Path,
    threshold: float,
) -> Path:
    """Apply the threshold and write the leaderboard-scored file."""
    print(f"Loading pairs with combined_score > {threshold}...")
    accepted = load_accepted_pairs(feature_cache_dir, threshold)
    print(f"  accepted pairs: {len(accepted)}")

    s1_ids = (
        pl.scan_parquet(normalized_dir / "test_source1.parquet")
        .select("entity_id")
        .collect()["entity_id"]
        .to_list()
    )
    print(f"  S1 entities: {len(s1_ids)}")

    matching_df = group_matches(accepted, s1_ids)
    write_matching_results(matching_df, output_path)
    n_matched = matching_df.filter(pl.col("matched_entity_ids") != "").height
    print(f"Wrote {len(matching_df)} rows to {output_path} ({n_matched} non-singleton)")
    return output_path


def macro_f05(
    predicted: dict[str, set[str]], ground_truth: dict[str, set[str]]
) -> tuple[float, float, float]:
    """Macro-averaged F_0.5 with its precision/recall components.

    Every Source 1 entity counts equally. A correctly predicted
    singleton (both empty) scores 1.0.
    """
    beta2 = 0.25
    f_scores, precisions, recalls = [], [], []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predicted.get(s1_id, set())
        if not true_ids and not pred_ids:
            f_scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
            continue
        tp = len(pred_ids & true_ids)
        precision = tp / len(pred_ids) if pred_ids else 0.0
        recall = tp / len(true_ids) if true_ids else 0.0
        denom = beta2 * precision + recall
        f_scores.append(
            (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
    n = len(ground_truth)
    return sum(f_scores) / n, sum(precisions) / n, sum(recalls) / n


def tune_threshold(
    feature_cache_dir: Path,
    ground_truth: dict[str, set[str]],
    thresholds: list[float],
) -> tuple[float, dict[float, tuple[float, float, float]]]:
    """Sweep candidate thresholds against macro F_0.5 (no re-scoring needed).

    Reads the cached feature parts once; each threshold is just a
    filter plus grouping, so sweeping is cheap.
    """
    scored = (
        pl.scan_parquet(feature_cache_dir / "features-*.parquet")
        .select("source1_entity_id", "candidate_id", "combined_score")
        .collect()
    )
    s1_ids = list(ground_truth.keys())
    results: dict[float, tuple[float, float, float]] = {}
    for threshold in thresholds:
        above = scored.filter(pl.col("combined_score") > threshold).select(
            "source1_entity_id", "candidate_id"
        )
        grouped = group_matches(above, s1_ids)
        predicted = {
            row["source1_entity_id"]: (
                set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
            )
            for row in grouped.iter_rows(named=True)
        }
        results[threshold] = macro_f05(predicted, ground_truth)
        f05, prec, rec = results[threshold]
        print(f"  threshold={threshold:.2f} F_0.5={f05:.4f} P={prec:.4f} R={rec:.4f}")
    best = max(thresholds, key=lambda t: results[t][0])
    print(f"Best threshold: {best:.2f} (F_0.5={results[best][0]:.4f})")
    return best, results
