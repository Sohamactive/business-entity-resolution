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
    n_rows = matrix.shape[1]
    sims = np.zeros(len(s1_names), dtype=np.float32)
    ok_s1 = np.array([index.get(t, -1) for t in s1_names])
    ok_c = np.array([index.get(t, -1) for t in cand_names])
    both = (ok_s1 >= 0) & (ok_c >= 0)
    if both.any():
        sims[both] = (matrix[ok_s1[both]] * matrix[ok_c[both]]).sum(axis=1)
    return np.round(sims, 4)
