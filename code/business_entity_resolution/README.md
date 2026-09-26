# Business Entity Resolution Pipeline

This repository contains the end-to-end Entity Resolution pipeline for the Amazon ML Challenge 2026.

## Architecture

The pipeline follows a two-stage funnel:
1. **Stage 1: Normalization & Multi-Pass Blocking**
   - Script-blind Unicode NFKC normalization and legal suffix canonicalization (`normalization.py`).
   - High-recall, memory-safe blocking engine combining brand-leading name keys, sorted order-invariant name keys, and compound address keys (`blocking.py`).
   - Dynamic country partitioning supporting open-set country labels (`US`, `India`, `France`).
   - Outputs `output/candidate_pairs.tsv`.
2. **Stage 2: Feature Engineering & Matching Model**
   - Candidate pair similarity features (char n-gram Jaccard, TF-IDF cosine, digit matching).
   - LightGBM / decision threshold tuned for macro-$F_{0.5}$.
   - Outputs `output/matching_results.tsv`.

---

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set configuration options in `.env`:

```env
# Maximum candidates retained per Source 1 entity during blocking
BLOCKING_MAX_CANDIDATES=150
```

---

## Running the Pipeline

### 1. Execute Unit Tests
To verify all normalization and blocking edge cases:

```bash
PYTHONPATH=. python -m unittest src/test_blocking.py
```

### 2. Run Internal Holdout Evaluation
To measure candidate recall and pool size distributions on a held-out ground truth split:

```bash
PYTHONPATH=. python src/evaluate_blocking.py --num-s1 2000 --num-distractors 50000
```

### 3. Run Blocking on Test Set
To generate the competition candidate pairs file (`output/candidate_pairs.tsv`):

```bash
PYTHONPATH=. python src/run_blocking.py \
    --source1 ../../data/student_resource/dataset/test/test_source1.tsv \
    --source2 ../../data/student_resource/dataset/test/test_source2.tsv \
    --source3 ../../data/student_resource/dataset/test/test_source3.tsv \
    --output ../../output/candidate_pairs.tsv
```

### 4. Format Validation
Always run the submission validator before uploading:

```bash
python ../../data/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../data/student_resource/dataset/test
```

## Dual Blocking Outputs

The blocking module (`blocking.py`) supports two complementary outputs:
1. **Official Competition Output (`output/candidate_pairs.tsv`)**:
   - Aggregated per Source 1 entity with comma-separated candidate IDs (singletons represented as empty strings).
   - Saved via `save_candidate_pairs(cand_df, "output/candidate_pairs.tsv")`.
   - Verified by `validate_submission.py`.
2. **Intermediate In-Memory Pair Table**:
   - Exploded pairwise table joining Source 1 and candidate records:
     `source1_entity_id | candidate_id | s1_name | s1_address | s1_country | cand_name | cand_address | cand_country`
   - Generated via `build_candidate_pairs_table(cand_df, s1_df, pool_df)`.
   - Streamed in chunks via `iter_candidate_pair_batches(cand_df, s1_df, pool_df, batch_size=200_000)` to ensure memory safety on 16 GB RAM.

---

## Module Reference

- `src/normalization.py`: Universal text normalization, legal suffix expansions, and Polars DataFrame vectorization.
- `src/blocking.py`: Multi-key extraction (`N_FIRST`, `N_SORT`, `ANW_`, `AWP_`), inverted indexing, TSV export, and pairwise table generator/batcher.
- `src/run_blocking.py`: Command-line interface to execute blocking on full datasets.
- `src/evaluate_blocking.py`: Recall benchmarking and metric calculation against ground truth.
- `src/test_blocking.py`: Unit test suite verifying edge cases, accents, format validation, and pairwise table creation.
