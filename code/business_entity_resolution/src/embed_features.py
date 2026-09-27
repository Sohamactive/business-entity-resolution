"""Multilingual embedding similarity (Iteration 3, Kaggle GPU).

Complements (never replaces) the TF-IDF features: embedding cosine on
normalized names rescues cross-script pairs that share zero character
n-grams. Vectors are computed ONCE per unique name and reused by index
lookup, so pair scoring stays a cheap dot product per row.

Model default: intfloat/multilingual-e5-small (MIT, ~118M params, well
under the 8B limit). Run on GPU; CPU works but is slow.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

EMB_COL = "emb_name_similarity"


def collect_unique_names(*frames: pl.DataFrame) -> list[str]:
    """Union of non-empty normalized names across frames."""
    names: set[str] = set()
    for frame in frames:
        names.update(n for n in frame["normalized_name"].to_list() if n)
    return sorted(names)


def build_name_vectors(
    names: list[str],
    model_name: str = "intfloat/multilingual-e5-small",
    batch_size: int = 1024,
) -> tuple[dict[str, int], np.ndarray]:
    """Encode unique names on GPU with normalized outputs. Saves nothing."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda")
    matrix = model.encode(
        names, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    return {name: i for i, name in enumerate(names)}, matrix


def save_vectors(index: dict[str, int], matrix: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "name_vectors.npy", matrix)
    with open(out_dir / "name_index.pkl", "wb") as f:
        pickle.dump(index, f)


def load_vectors(vec_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    with open(vec_dir / "name_index.pkl", "rb") as f:
        index = pickle.load(f)
    return index, np.load(vec_dir / "name_vectors.npy", mmap_mode="r")


def embedding_similarity(
    s1_names: list[str], cand_names: list[str],
    index: dict[str, int], matrix: np.ndarray,
) -> np.ndarray:
    """Cosine similarity by index lookup; unknown/empty names score 0.0."""
    sims = np.zeros(len(s1_names), dtype=np.float32)
    ok_s1 = np.array([index.get(t, -1) for t in s1_names])
    ok_c = np.array([index.get(t, -1) for t in cand_names])
    both = (ok_s1 >= 0) & (ok_c >= 0)
    if both.any():
        sims[both] = (matrix[ok_s1[both]] * matrix[ok_c[both]]).sum(axis=1)
    return np.round(sims, 4)


EMB_FEATURE_COLS = [
    "emb_name_similarity",
    "name_is_missing",
    "address_is_missing",
    "country_match",
    "name_len_diff",
    "name_len_ratio",
    "addr_len_diff",
]


def cheap_pair_features(
    joined: pl.DataFrame,
    index: dict[str, int],
    matrix: np.ndarray,
) -> pl.DataFrame:
    """Fast GPU-free features for one joined batch (no TF-IDF).

    Expects: source1_entity_id, candidate_id, s1_norm_name,
    s1_norm_address, s1_country, cand_norm_name, cand_norm_address,
    cand_country. Returns those IDs plus EMB_FEATURE_COLS.
    """
    s1_names = joined["s1_norm_name"].fill_null("").to_list()
    cand_names = joined["cand_norm_name"].fill_null("").to_list()
    s1_addr = joined["s1_norm_address"].fill_null("").to_list()
    cand_addr = joined["cand_norm_address"].fill_null("").to_list()
    s1_c = joined["s1_country"].fill_null("").to_list()
    cand_c = joined["cand_country"].fill_null("").to_list()

    emb = embedding_similarity(s1_names, cand_names, index, matrix)
    s1_len = np.array([len(t) for t in s1_names], dtype=np.float32)
    c_len = np.array([len(t) for t in cand_names], dtype=np.float32)
    s1_alen = np.array([len(t) for t in s1_addr], dtype=np.float32)
    c_alen = np.array([len(t) for t in cand_addr], dtype=np.float32)

    return pl.DataFrame(
        {
            "source1_entity_id": joined["source1_entity_id"].to_list(),
            "candidate_id": joined["candidate_id"].to_list(),
            "emb_name_similarity": emb.tolist(),
            "name_is_missing": [1 if not t else 0 for t in s1_names],
            "address_is_missing": [1 if not (a and b) else 0 for a, b in zip(s1_addr, cand_addr)],
            "country_match": [1 if a == b else 0 for a, b in zip(s1_c, cand_c)],
            "name_len_diff": np.abs(s1_len - c_len).tolist(),
            "name_len_ratio": (np.minimum(s1_len, c_len) / np.maximum(np.maximum(s1_len, c_len), 1)).tolist(),
            "addr_len_diff": np.abs(s1_alen - c_alen).tolist(),
        }
    )
