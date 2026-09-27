"""Batched character TF-IDF feature engineering (Iteration 1).

Reads the identifier-only candidate parts written by blocking
(`source1_entity_id`, `candidate_id`), joins each bounded batch to the
normalized Parquet caches, and scores every pair with sparse cosine
similarity. The TF-IDF vectorizers are fitted ONCE on a bounded sample
of normalized text and reused for every batch.

Per-pair output columns:
    source1_entity_id, candidate_id,
    name_similarity, address_similarity,
    name_is_missing, address_is_missing,
    country_match, combined_score

Missing normalized fields (empty strings) contribute 0.0 similarity;
the missingness flags are preserved for diagnostics and later classifiers.
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Iteration 1 decision weights (PRD section 8.4). Thresholding itself
# lives in the matching step so the threshold can be retuned without
# recomputing these similarities.
NAME_WEIGHT = 0.6
ADDRESS_WEIGHT = 0.4

# Single vectorizer configuration used for the whole run. Fitted once,
# never per pair and never per batch.
NGRAM_RANGE = (3, 5)
MAX_FEATURES = 100_000
MIN_DF = 2


def build_vectorizer() -> TfidfVectorizer:
    """Create the shared character TF-IDF vectorizer configuration."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )


def load_fit_corpus(normalized_dir: Path, sample_rows: int = 500_000) -> tuple[list[str], list[str]]:
    """Collect a bounded sample of normalized names/addresses for fitting.

    Sampling keeps the fit in memory while covering all three sources.
    """
    names: list[str] = []
    addresses: list[str] = []
    per_file = max(sample_rows // 3, 1)
    for fname in ("test_source1.parquet", "test_source2.parquet", "test_source3.parquet"):
        frame = (
            pl.scan_parquet(normalized_dir / fname)
            .select("normalized_name", "normalized_address")
            .head(per_file)
            .collect()
        )
        names.extend(frame["normalized_name"].to_list())
        addresses.extend(frame["normalized_address"].to_list())
    return names, addresses


def fit_vectorizers(
    names: list[str], addresses: list[str]
) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """Fit the name and address vectorizers exactly once."""
    name_vec = build_vectorizer()
    address_vec = build_vectorizer()
    name_vec.fit(names or [""])
    address_vec.fit(addresses or [""])
    return name_vec, address_vec


def save_vectorizers(name_vec: TfidfVectorizer, address_vec: TfidfVectorizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "name_vectorizer.pkl", "wb") as f:
        pickle.dump(name_vec, f)
    with open(out_dir / "address_vectorizer.pkl", "wb") as f:
        pickle.dump(address_vec, f)


def load_vectorizers(vectorizer_dir: Path) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    with open(vectorizer_dir / "name_vectorizer.pkl", "rb") as f:
        name_vec = pickle.load(f)
    with open(vectorizer_dir / "address_vectorizer.pkl", "rb") as f:
        address_vec = pickle.load(f)
    return name_vec, address_vec


def _transform_deduped(vectorizer: TfidfVectorizer, texts: list[str]):
    """Transform only unique strings, then expand rows back to pair order.

    Source 1 strings repeat heavily within a batch (many candidates per
    entity), so this skips most redundant tokenizer work. Row indexing on
    the sparse result is cheap compared to re-tokenizing.
    """
    uniq, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    matrix = vectorizer.transform(uniq.tolist())
    return matrix[inverse]


def _sparse_cosine_rows(left, right) -> np.ndarray:
    """Row-wise cosine similarity for L2-normalized TF-IDF matrices.

    Empty input strings transform to all-zero rows, whose dot product
    is 0.0, which is exactly the desired missing-field behavior.
    """
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def score_pair_batch(
    batch_df: pl.DataFrame,
    name_vec: TfidfVectorizer,
    address_vec: TfidfVectorizer,
) -> pl.DataFrame:
    """Score one joined batch and return the per-pair feature rows.

    Expects columns: source1_entity_id, candidate_id, s1_norm_name,
    s1_norm_address, s1_country, cand_norm_name, cand_norm_address,
    cand_country.
    """
    s1_names = batch_df["s1_norm_name"].fill_null("").to_list()
    cand_names = batch_df["cand_norm_name"].fill_null("").to_list()
    s1_addresses = batch_df["s1_norm_address"].fill_null("").to_list()
    cand_addresses = batch_df["cand_norm_address"].fill_null("").to_list()

    name_sim = _sparse_cosine_rows(
        _transform_deduped(name_vec, s1_names), _transform_deduped(name_vec, cand_names)
    )
    address_sim = _sparse_cosine_rows(
        _transform_deduped(address_vec, s1_addresses),
        _transform_deduped(address_vec, cand_addresses),
    )

    s1_country = batch_df["s1_country"].fill_null("").to_list()
    cand_country = batch_df["cand_country"].fill_null("").to_list()

    name_is_missing = [1 if not text else 0 for text in s1_names]
    # A pair counts as address-missing when either side lacks an address.
    address_is_missing = [
        1 if not (a and b) else 0 for a, b in zip(s1_addresses, cand_addresses)
    ]
    country_match = [1 if a == b else 0 for a, b in zip(s1_country, cand_country)]
    combined = NAME_WEIGHT * name_sim + ADDRESS_WEIGHT * address_sim

    return pl.DataFrame(
        {
            "source1_entity_id": batch_df["source1_entity_id"].to_list(),
            "candidate_id": batch_df["candidate_id"].to_list(),
            "name_similarity": np.round(name_sim, 4).tolist(),
            "address_similarity": np.round(address_sim, 4).tolist(),
            "name_is_missing": name_is_missing,
            "address_is_missing": address_is_missing,
            "country_match": country_match,
            "combined_score": np.round(combined, 4).tolist(),
        }
    )


def _join_chunk(chunk: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Attach normalized metadata to one ID chunk (fast native Polars join)."""
    return (
        chunk.join(
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


def _score_and_write_chunk(
    chunk: pl.DataFrame,
    s1: pl.DataFrame,
    pool: pl.DataFrame,
    name_vec: TfidfVectorizer,
    address_vec: TfidfVectorizer,
    out_path: Path,
) -> int:
    """Join one ID chunk to normalized metadata, score it, write one part.

    Read-only inputs are shared across threads; each task writes its own
    deterministic output file, so tasks never interfere with each other.
    """
    joined = _join_chunk(chunk, s1, pool)
    scored = score_pair_batch(joined, name_vec, address_vec)
    scored.write_parquet(out_path)
    return len(scored)


# --- Multiprocess backend: parent joins (fast), workers transform (GIL-free).
# Worker processes load the fitted vectorizers once via the initializer, so
# only small per-chunk string lists cross the process boundary. The gigabyte
# caches stay in the parent and are never duplicated per worker.

_PROC_NAME_VEC: TfidfVectorizer | None = None
_PROC_ADDR_VEC: TfidfVectorizer | None = None


def _proc_init(vectorizer_dir: str) -> None:
    global _PROC_NAME_VEC, _PROC_ADDR_VEC
    with open(Path(vectorizer_dir) / "name_vectorizer.pkl", "rb") as f:
        _PROC_NAME_VEC = pickle.load(f)
    with open(Path(vectorizer_dir) / "address_vectorizer.pkl", "rb") as f:
        _PROC_ADDR_VEC = pickle.load(f)


def _proc_score_task(payload: dict) -> dict:
    """Transform + score one chunk's string lists; returns plain arrays."""
    assert _PROC_NAME_VEC is not None and _PROC_ADDR_VEC is not None
    name_sim = _sparse_cosine_rows(
        _transform_deduped(_PROC_NAME_VEC, payload["s1_names"]),
        _transform_deduped(_PROC_NAME_VEC, payload["cand_names"]),
    )
    address_sim = _sparse_cosine_rows(
        _transform_deduped(_PROC_ADDR_VEC, payload["s1_addresses"]),
        _transform_deduped(_PROC_ADDR_VEC, payload["cand_addresses"]),
    )
    combined = NAME_WEIGHT * name_sim + ADDRESS_WEIGHT * address_sim
    return {
        "name_similarity": np.round(name_sim, 4),
        "address_similarity": np.round(address_sim, 4),
        "combined_score": np.round(combined, 4),
    }


def _proc_payload(joined: pl.DataFrame) -> dict:
    s1_names = joined["s1_norm_name"].to_list()
    cand_names = joined["cand_norm_name"].to_list()
    s1_addresses = joined["s1_norm_address"].to_list()
    cand_addresses = joined["cand_norm_address"].to_list()
    s1_country = joined["s1_country"].to_list()
    cand_country = joined["cand_country"].to_list()
    return {
        "source1_entity_id": joined["source1_entity_id"].to_list(),
        "candidate_id": joined["candidate_id"].to_list(),
        "s1_names": s1_names,
        "cand_names": cand_names,
        "s1_addresses": s1_addresses,
        "cand_addresses": cand_addresses,
        "name_is_missing": [1 if not t else 0 for t in s1_names],
        "address_is_missing": [1 if not (a and b) else 0 for a, b in zip(s1_addresses, cand_addresses)],
        "country_match": [1 if a == b else 0 for a, b in zip(s1_country, cand_country)],
    }


def run_feature_pipeline(
    candidate_cache_dir: Path,
    normalized_dir: Path,
    feature_cache_dir: Path,
    vectorizer_dir: Path,
    batch_size: int = 200_000,
    fit_sample_rows: int = 500_000,
    workers: int = 1,
    resume: bool = True,
    backend: str = "threads",
) -> Path:
    """Score every candidate part and write bounded feature Parquet parts.

    Safety design for a loaded local machine:
      - workers=1 reproduces the original single-threaded loop exactly,
        so any parallel problem can be reverted by passing workers=1.
      - backend="threads" shares the caches and vectorizers in memory;
        memory stays flat regardless of worker count, but throughput is
        capped by Python's global lock around the tokenizer.
      - backend="processes" tokenizes in separate processes (true
        parallelism, one lane per core). The parent keeps the gigabyte
        caches and performs the fast joins; only small per-chunk string
        lists cross to workers. Start with 4 workers and scale up only
        while memory stays comfortable.
      - resume=True skips feature parts already on disk, so an
        interrupted run continues where it stopped instead of redoing
        hours of work. Output filenames are deterministic.
    """
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting TF-IDF vectorizers once...")
    names, addresses = load_fit_corpus(normalized_dir, fit_sample_rows)
    name_vec, address_vec = fit_vectorizers(names, addresses)
    save_vectorizers(name_vec, address_vec, vectorizer_dir)
    print(f"  name vocab={len(name_vec.vocabulary_)} address vocab={len(address_vec.vocabulary_)}")

    print("Loading normalized caches for joins...")
    s1 = pl.scan_parquet(normalized_dir / "test_source1.parquet").select(
        "entity_id", "country", "normalized_name", "normalized_address"
    ).collect()
    pool = pl.concat(
        [
            pl.scan_parquet(normalized_dir / "test_source2.parquet").select(
                "entity_id", "country", "normalized_name", "normalized_address"
            ).collect(),
            pl.scan_parquet(normalized_dir / "test_source3.parquet").select(
                "entity_id", "country", "normalized_name", "normalized_address"
            ).collect(),
        ]
    )
    print(f"  S1={len(s1)} pool={len(pool)}")

    part_paths = sorted(candidate_cache_dir.glob("part-*.parquet"))
    print(f"Scoring {len(part_paths)} candidate parts in batches of {batch_size}...")

    # Plan every chunk upfront WITHOUT loading pair data: store part path +
    # offset + deterministic output index. This keeps memory flat no matter
    # how many parts exist, because pair data is loaded one part at a time
    # during execution. Resume works off filenames alone.
    # plan entries: (part_path, offset, n_rows, out_path or None if skipped)
    plan: list[tuple[Path, int, int, Path | None]] = []
    feature_idx = 0
    for part_path in part_paths:
        n_pairs = pl.scan_parquet(part_path).select(pl.len()).collect().item()
        for offset in range(0, n_pairs, batch_size):
            out_path = feature_cache_dir / f"features-{feature_idx:05d}.parquet"
            if resume and out_path.exists():
                plan.append((part_path, offset, 0, None))
            else:
                plan.append((part_path, offset, min(batch_size, n_pairs - offset), out_path))
            feature_idx += 1
    pending_plan = [(p, o, n, fp) for (p, o, n, fp) in plan if fp is not None]
    skipped = len(plan) - len(pending_plan)
    if skipped:
        print(f"  resume: skipping {skipped} already-written parts, {len(pending_plan)} remaining")

    def write_scored(payload: dict, scores: dict, out_path: Path) -> None:
        pl.DataFrame(
            {
                "source1_entity_id": payload["source1_entity_id"],
                "candidate_id": payload["candidate_id"],
                "name_similarity": scores["name_similarity"].tolist(),
                "address_similarity": scores["address_similarity"].tolist(),
                "name_is_missing": payload["name_is_missing"],
                "address_is_missing": payload["address_is_missing"],
                "country_match": payload["country_match"],
                "combined_score": scores["combined_score"].tolist(),
            }
        ).write_parquet(out_path)

    # Group pending chunks by part so each part is loaded exactly once.
    from collections import defaultdict
    by_part: dict[Path, list[tuple[int, int, Path]]] = defaultdict(list)
    for part_path, offset, n_rows, out_path in pending_plan:
        assert out_path is not None
        by_part[part_path].append((offset, n_rows, out_path))

    if workers <= 1:
        # Original single-threaded path, kept verbatim as the revert option.
        for part_path in part_paths:
            chunks = by_part.get(part_path, [])
            if not chunks:
                continue
            pairs = pl.read_parquet(part_path).select("source1_entity_id", "candidate_id")
            for offset, n_rows, out_path in tqdm(chunks, desc=part_path.name, unit="chunk"):
                _score_and_write_chunk(
                    pairs.slice(offset, n_rows), s1, pool, name_vec, address_vec, out_path
                )
    elif backend == "processes":
        max_workers = max(1, min(workers, (os.cpu_count() or 4) - 2))
        # Bounded in-flight payloads: the parent joins and pickles at most
        # a few chunks ahead, so queued work never accumulates in memory.
        max_pending = 2 * max_workers
        print(f"  process workers: {max_workers} (requested {workers}; caches stay in parent)")
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_proc_init,
                initargs=(str(vectorizer_dir),),
            ) as executor:
                pending: dict = {}
                progress = tqdm(total=len(pending_plan), desc="scoring chunks", unit="chunk")
                for part_path in part_paths:
                    chunks = by_part.get(part_path, [])
                    if not chunks:
                        continue
                    pairs = pl.read_parquet(part_path).select("source1_entity_id", "candidate_id")
                    chunk_iter = iter(chunks)
                    while True:
                        while len(pending) < max_pending:
                            try:
                                offset, n_rows, out_path = next(chunk_iter)
                            except StopIteration:
                                break
                            joined = _join_chunk(pairs.slice(offset, n_rows), s1, pool)
                            payload = _proc_payload(joined)
                            del joined
                            pending[executor.submit(_proc_score_task, payload)] = (payload, out_path)
                        if not pending:
                            break
                        done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                        for future in done:
                            payload, out_path = pending.pop(future)
                            write_scored(payload, future.result(), out_path)
                            del payload
                            progress.update(1)
                    del pairs
                progress.close()
        except KeyboardInterrupt:
            print("  interrupted: already-written parts are kept; rerun with resume=True to continue")
            raise
    else:
        max_workers = max(1, min(workers, (os.cpu_count() or 4) - 2))
        print(f"  thread workers: {max_workers} (requested {workers}; shared memory, no duplication)")
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool_executor:
                # Stream one part at a time so pair data never accumulates
                # across parts; each part's futures drain before the next load.
                part_chunks = [
                    (part_path, by_part[part_path])
                    for part_path in part_paths
                    if by_part.get(part_path)
                ]
                total = sum(len(chunks) for _, chunks in part_chunks)
                progress = tqdm(total=total, desc="scoring chunks", unit="chunk")
                for part_path, chunks in part_chunks:
                    pairs = pl.read_parquet(part_path).select("source1_entity_id", "candidate_id")
                    futures = {
                        pool_executor.submit(
                            _score_and_write_chunk,
                            pairs.slice(offset, n_rows),
                            s1, pool, name_vec, address_vec, out_path,
                        ): out_path
                        for offset, n_rows, out_path in chunks
                    }
                    for future in as_completed(futures):
                        future.result()
                        progress.update(1)
                    del pairs
                progress.close()
        except KeyboardInterrupt:
            print("  interrupted: already-written parts are kept; rerun with resume=True to continue")
            raise

    print(f"Feature parts complete in {feature_cache_dir} ({feature_idx} total)")
    return feature_cache_dir
