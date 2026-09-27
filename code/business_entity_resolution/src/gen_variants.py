"""Generate threshold-variant submission files from the proba cache (minutes).

Each variant applies a global proba cutoff plus an optional stricter
France cutoff (unseen country -> conservative matching), then groups and
writes a validated-shape matching_results file.

Example:
  python -m business_entity_resolution.src.gen_variants --variants "0.8:0.9" "0.9:0.95"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from business_entity_resolution.src.matching import group_matches, write_matching_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate threshold variant submissions.")
    p.add_argument("--proba-cache", type=Path,
                   default=Path("business_entity_resolution/src/cache/matcher/test_probas.parquet"))
    p.add_argument("--normalized-dir", type=Path,
                   default=Path("business_entity_resolution/src/cache/normalized"))
    p.add_argument("--out-dir", type=Path, default=Path("../output/variants"))
    p.add_argument("--variants", nargs="+", required=True,
                   help="Each as global[:france], e.g. 0.8:0.9")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probas = pl.read_parquet(args.proba_cache)
    print(f"Proba cache rows: {len(probas)}")
    s1c = (pl.scan_parquet(args.normalized_dir / "test_source1.parquet")
           .select("entity_id", "country").collect())
    cmap = dict(zip(s1c["entity_id"].to_list(), s1c["country"].to_list()))
    all_ids = s1c["entity_id"].to_list()
    fr = probas.with_columns(
        pl.col("source1_entity_id").map_elements(cmap.get, return_dtype=pl.String).alias("country"))

    for spec in args.variants:
        g, _, f = spec.partition(":")
        gthr, fthr = float(g), float(f) if f else float(g)
        name = f"thr{gthr}_fr{fthr}".replace(".", "p")
        accepted = fr.filter(
            ((pl.col("country") == "France") & (pl.col("proba") > fthr))
            | ((pl.col("country") != "France") & (pl.col("proba") > gthr))
        ).select("source1_entity_id", "candidate_id")
        matching_df = group_matches(accepted, all_ids)
        out = args.out_dir / f"matching_results_{name}.tsv"
        write_matching_results(matching_df, out)
        n_non = matching_df.filter(pl.col("matched_entity_ids") != "").height
        avg = len(accepted) / len(all_ids)
        print(f"{out.name}: accepted={len(accepted)} non-single={n_non} avg/S1={avg:.2f}")


if __name__ == "__main__":
    main()
