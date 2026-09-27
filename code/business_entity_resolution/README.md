# Business Entity Resolution Pipeline

This repository contains the end-to-end Entity Resolution pipeline for the Amazon ML Challenge 2026.

## Architecture

The pipeline follows a two-stage funnel:
1. **Stage 1: Normalization & Multi-Pass Blocking**
   - Script-blind Unicode NFKC normalization and legal suffix canonicalization (`normalization.py`).
   - High-recall, memory-safe blocking engine combining brand-leading name keys, sorted order-invariant name keys, and compound address keys (`blocking.py`).
   - Dynamic country partitioning supporting open-set country labels (`US`, `India`, `France`).
   - Reuses normalized Parquet caches and outputs `output/candidate_pairs.tsv` plus bounded candidate-ID Parquet parts.
2. **Stage 2: Feature Engineering & Matching Model**
   - Character TF-IDF/cosine similarity features computed in sparse bounded batches.
   - Iteration 1 decision threshold tuned for macro-$F_{0.5}$; LightGBM is a later option.
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

### 3. Build Normalized Parquet Caches
Normalize each raw source once and reuse the resulting Parquet files for blocking and features.

### 4. Run Blocking on Test Set
To generate the competition candidate pairs file and reusable candidate-ID Parquet parts:

```bash
PYTHONPATH=. python -m business_entity_resolution.src.run_blocking \
    --normalized-dir src/cache/normalized \
    --output ../../output/candidate_pairs.tsv \
    --candidate-cache-dir src/cache/candidates/test_pairs \
    --batch-size 50000 \
    --force
```

The full test run produced 224,533,181 candidate pairs in approximately 5.5 minutes locally.

### 5. Format Validation
Always run the submission validator before uploading:

```bash
python ../../data/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../data/student_resource/dataset/test
```

## Blocking Outputs

The blocking module (`blocking.py`) supports two complementary outputs:
1. **Official Competition Output (`output/candidate_pairs.tsv`)**:
   - Aggregated per Source 1 entity with comma-separated candidate IDs (singletons represented as empty strings).
   - Saved via `save_candidate_pairs(cand_df, "output/candidate_pairs.tsv")`.
   - Verified by `validate_submission.py`.
2. **Candidate-ID Parquet Cache**:
   - Long-form parts under `cache/candidates/test_pairs/`.
   - Columns: `source1_entity_id`, `candidate_id`.
   - Feature engineering joins normalized metadata later in bounded batches; blocking does not repeat a full Source 2/3 metadata join.

---

## Module Reference

- `src/normalization.py`: Universal text normalization, legal suffix expansions, and Polars DataFrame vectorization.
- `src/blocking.py`: Multi-key extraction (`N_FIRST`, `N_SORT`, `ANW_`, `AWP_`), inverted indexing, TSV export, and pairwise table generator/batcher.
- `src/run_blocking.py`: Command-line interface to execute blocking on full datasets.
- `src/evaluate_blocking.py`: Recall benchmarking and metric calculation against ground truth.
- `src/test_blocking.py`: Unit test suite verifying edge cases, accents, format validation, and pairwise table creation.
