"""Train LightGBM matcher on a hard train holdout (Iteration 2).

Uses blocking candidates (hard negatives) with the ALREADY-FITTED
vectorizers, training only on the 8 feature columns already stored in
the test feature parts so no test re-scoring is needed. Calibrates the
decision threshold against macro F_0.5 with per-country breakdowns.

Example: python -m business_entity_resolution.src.train_matcher --num-s1 5000
"""

from __future__ import annotations

import argparse
import pickle
import tempfile
from pathlib import Path

import json

import lightgbm as lgb
import polars as pl

from business_entity_resolution.src.blocking import build_candidate_pairs_table
from business_entity_resolution.src.embed_features import EMB_COL, embedding_similarity, load_vectors
from business_entity_resolution.src.evaluate_blocking import evaluate_blocking
from business_entity_resolution.src.features import load_vectorizers, score_pair_batch
from business_entity_resolution.src.matching import group_matches, macro_f05
from business_entity_resolution.src.normalization import normalize_dataframe

FEATURE_COLS = [
    "name_similarity", "address_similarity",
    "name_is_missing", "address_is_missing",
    "country_match", "combined_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM matcher on hard holdout.")
    p.add_argument("--num-s1", type=int, default=5000)
    p.add_argument("--num-distractors", type=int, default=200000)
    p.add_argument("--data-dir", type=Path, default=Path("../data/student_resource"))
    p.add_argument("--vectorizer-dir", type=Path,
                   default=Path("business_entity_resolution/src/cache/vectorizers"))
    p.add_argument("--model-dir", type=Path,
                   default=Path("business_entity_resolution/src/cache/matcher"))
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--embed-vec-dir", type=Path, default=None,
                   help="Directory with name_vectors.npy + name_index.pkl; "
                   "adds emb_name_similarity to the trained features")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print("1. Hard holdout candidates via blocking...")
    with tempfile.TemporaryDirectory():
        result = evaluate_blocking(
            data_dir=args.data_dir, num_s1=args.num_s1,
            num_distractors=args.num_distractors)
    cand_df = result["candidate_df"]
    print(f"   blocking recall: {result['overall_recall']:.2f}%")

    print("2. Labels + pair table...")
    train_dir = args.data_dir / "dataset" / "train"
    s1_df = normalize_dataframe(
        pl.scan_csv(train_dir / "train_source1.tsv", separator="\t").head(args.num_s1).collect())
    gt_df = pl.scan_csv(train_dir / "train_ground_truth.tsv", separator="\t").collect()
    gt_map: dict[str, set[str]] = {}
    for row in gt_df.filter(pl.col("source1_entity_id").is_in(s1_df["entity_id"].to_list())).iter_rows(named=True):
        gt_map[row["source1_entity_id"]] = {
            m.strip() for m in (row["matched_entity_ids"] or "").split(",") if m.strip()}
    for sid in s1_df["entity_id"].to_list():
        gt_map.setdefault(sid, set())

    cand_ids: set[str] = set()
    for cell in cand_df["candidate_entity_ids"].to_list():
        if cell:
            cand_ids.update(c.strip() for c in cell.split(",") if c.strip())
    pool_df = normalize_dataframe(pl.concat([
        pl.scan_csv(train_dir / "train_source2.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(cand_ids))).collect(),
        pl.scan_csv(train_dir / "train_source3.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(cand_ids))).collect(),
    ]).unique(subset=["entity_id"]))

    pair_table = build_candidate_pairs_table(cand_df, s1_df=s1_df, pool_df=pool_df, include_normalized=True)
    print(f"   pairs: {len(pair_table)}")

    print("3. Scoring with fitted vectorizers...")
    name_vec, address_vec = load_vectorizers(args.vectorizer_dir)
    scored = score_pair_batch(pair_table, name_vec, address_vec)
    feature_cols = list(FEATURE_COLS)
    if args.embed_vec_dir is not None:
        print("   adding embedding similarity...")
        emb_index, emb_matrix = load_vectors(args.embed_vec_dir)
        emb_sim = embedding_similarity(
            pair_table["s1_norm_name"].fill_null("").to_list(),
            pair_table["cand_norm_name"].fill_null("").to_list(),
            emb_index, emb_matrix)
        scored = scored.with_columns(pl.Series(EMB_COL, emb_sim))
        feature_cols.append(EMB_COL)
    labels = [
        1 if cid in gt_map.get(sid, set()) else 0
        for sid, cid in zip(scored["source1_entity_id"].to_list(), scored["candidate_id"].to_list())
    ]
    scored = scored.with_columns(pl.Series("label", labels))
    print(f"   positives: {sum(labels)}/{len(labels)} ({100*sum(labels)/len(labels):.2f}%)")

    print("4. Training LightGBM (MIT-licensed, tiny)...")
    X = scored.select(feature_cols).to_numpy()
    y = scored["label"].to_numpy()
    train_set = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
        "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.9,
        "bagging_fraction": 0.8, "bagging_freq": 5, "num_threads": 0,
    }
    model = lgb.train(params, train_set, num_boost_round=500)
    model.save_model(str(args.model_dir / "matcher.txt"))
    with open(args.model_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"   saved matcher.txt with features: {feature_cols}")

    print("5. Calibrating threshold on macro F_0.5...")
    proba = model.predict(X)
    countries = dict(zip(s1_df["entity_id"].to_list(), s1_df["country"].to_list()))
    s1_ids = list(gt_map.keys())
    pairs = list(zip(scored["source1_entity_id"].to_list(), scored["candidate_id"].to_list()))
    best_thr, best_f05 = 0.5, -1.0
    for thr in args.thresholds:
        accepted = pl.DataFrame({
            "source1_entity_id": [p[0] for p, pr in zip(pairs, proba) if pr > thr],
            "candidate_id": [p[1] for p, pr in zip(pairs, proba) if pr > thr],
        })
        grouped = group_matches(accepted, s1_ids)
        predicted = {
            r["source1_entity_id"]: (set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set())
            for r in grouped.iter_rows(named=True)}
        f05, prec, rec = macro_f05(predicted, gt_map)
        detail = ""
        for c in sorted(set(countries.values())):
            sub = {s: gt_map[s] for s in s1_ids if countries.get(s) == c}
            if sub:
                detail += f" {c}={macro_f05({s: predicted[s] for s in sub}, sub)[0]:.3f}"
        print(f"   thr>{thr:.2f} F0.5={f05:.4f} P={prec:.4f} R={rec:.4f} |{detail}")
        if f05 > best_f05:
            best_f05, best_thr = f05, thr
    with open(args.model_dir / "threshold.txt", "w") as f:
        f.write(f"{best_thr}")
    print(f"RECOMMENDED proba threshold: {best_thr} (holdout F_0.5={best_f05:.4f})")


if __name__ == "__main__":
    main()
