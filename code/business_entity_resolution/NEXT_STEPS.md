# AI-to-AI Handoff Specification: Feature Engineering & Matching

> **Target Audience**: AI Coding Assistants (Claude Code, Cursor, Copilot, etc.) & Engineers building the **Feature Engineering** and **Matching Model** stages.  
> **Repository Context**: Amazon ML Challenge 2026 — Business Entity Resolution  
> **Last Stage Completed**: Full-scale normalization and multi-pass blocking. Feature engineering is next.

---

## 1. Pipeline Context & The Funnel

Entity resolution follows a strict two-stage funnel:
```
Raw Data (S1, S2, S3)
       │
       ▼
1. Normalization (Unicode NFKC, lowercase, punctuation, corporate suffix expansion)
       │
       ▼
2. Blocking (Candidate Generation: multi-pass compound keys, reduction >99.999%)
       │
       ├──► output/candidate_pairs.tsv (Official compact candidate output)
       ├──► cache/candidates/test_pairs/part-*.parquet (Long-form candidate ID cache)
       │
       ▼
3. Feature Engineering (String similarity, token overlap, country match on candidate pairs)
       │  ◄── YOU ARE HERE (Stage 3 & 4)
       ▼
4. Matching Model / Threshold Decision (Predict true matches, optimize Macro F_0.5)
       │
       ▼
5. Validation (utils/validate_submission.py)
       │
       ▼
   output/matching_results.tsv (Leaderboard Scored File)
```

---

## 2. What the Blocking Step Produced (Your Inputs)

The blocking engine provides the compact submission output and a reusable long-form candidate ID cache.

### Option A: Small In-Memory Batch Streaming
At full scale, there are 1.73M Source 1 entities and 9.97M Source 2/3 entities, generating 224,533,181 candidate pairs. Blocking writes bounded Parquet parts containing only `source1_entity_id` and `candidate_id`; do not materialize all candidate text fields in memory.

To keep memory footprint $<2\text{ GB}$ on a 16 GB machine, use the provided generator `iter_candidate_pair_batches()`:

```python
from business_entity_resolution.src.blocking import (
    BlockingIndex,
    block_source1_against_index,
    iter_candidate_pair_batches,
)
from business_entity_resolution.src.normalization import normalize_dataframe
import polars as pl

# 1. Load and normalize (small tests only)
s1_df = normalize_dataframe(pl.read_csv("data/student_resource/dataset/test/test_source1.tsv", separator="\t"))
s2_df = normalize_dataframe(pl.read_csv("data/student_resource/dataset/test/test_source2.tsv", separator="\t"))
s3_df = normalize_dataframe(pl.read_csv("data/student_resource/dataset/test/test_source3.tsv", separator="\t"))
pool_df = pl.concat([s2_df, s3_df])

# 2. Build index & generate candidate pairs
index = BlockingIndex()
index.add_dataframe(pool_df)
candidate_df = block_source1_against_index(s1_df, index)

# 3. Stream candidate pairs in batches (default: 200,000 pairs/chunk)
for batch_df in iter_candidate_pair_batches(candidate_df, s1_df, pool_df, batch_size=200_000):
    # batch_df is a Polars DataFrame with 200,000 candidate pairs
    # Compute your similarity features directly on batch_df!
    ...
```

#### Batch DataFrame Schema (`batch_df`):
| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | `pl.String` | Unique ID of the Source 1 reference record (e.g. `S1-001`) |
| `candidate_id` | `pl.String` | Candidate ID from Source 2 or 3 (e.g. `S2-101` or `S3-202`) |
| `s1_name` | `pl.String` | Raw / original business name of Source 1 entity |
| `s1_address` | `pl.String` | Raw / original address of Source 1 entity |
| `s1_country` | `pl.String` | Country label of Source 1 entity (`US`, `India`, or `France`) |
| `cand_name` | `pl.String` | Raw / original business name of candidate entity |
| `cand_address` | `pl.String` | Raw / original address of candidate entity |
| `cand_country` | `pl.String` | Country label of candidate entity (always matches `s1_country`) |

> **Tip**: If you pass `include_normalized=True` to `iter_candidate_pair_batches`, it will also include `s1_norm_name`, `s1_norm_address`, `cand_norm_name`, and `cand_norm_address`, saving you from re-normalizing strings.

---

### Option B: Reading Candidate ID Parquet Parts
The full-scale feature stage should scan the generated parts:

```python
candidate_pairs = pl.scan_parquet("cache/candidates/test_pairs/part-*.parquet")
```

Join each bounded batch to the normalized source caches before calculating features. The
blocking stage intentionally does not repeat a full Source 2/3 metadata join for every batch.

### Option C: Reading Directly From Disk (`candidate_pairs.tsv`)
If you prefer running from saved disk files, the blocking step saves `output/candidate_pairs.tsv`:
- **Format**: Tab-separated TSV (`sep="\t"`).
- **Columns**: `source1_entity_id`, `candidate_entity_ids` (comma-separated list of candidate IDs).
- **Singletons**: Represented as empty strings (no quotes).

```python
import polars as pl

# Read candidate pairs
cand_df = pl.read_csv("output/candidate_pairs.tsv", separator="\t")

# Explode candidate IDs
exploded = (
    cand_df
    .filter(pl.col("candidate_entity_ids").is_not_null() & (pl.col("candidate_entity_ids").str.len_bytes() > 0))
    .with_columns(pl.col("candidate_entity_ids").str.split(","))
    .explode("candidate_entity_ids")
    .rename({"candidate_entity_ids": "candidate_id"})
)

# Join with metadata from s1_df and pool_df
pair_table = (
    exploded
    .join(s1_df.select(["entity_id", "business_name", "business_address", "country"]), left_on="source1_entity_id", right_on="entity_id")
    .rename({"business_name": "s1_name", "business_address": "s1_address", "country": "s1_country"})
    .join(pool_df.select(["entity_id", "business_name", "business_address", "country"]), left_on="candidate_id", right_on="entity_id")
    .rename({"business_name": "cand_name", "business_address": "cand_address", "country": "cand_country"})
)
```

---

## 3. What You Need to Build (Stage 3 & 4)

### 3.1 Feature Engineering
For each candidate pair `(source1_entity_id, candidate_id)`, compute similarity features:
1. **Name Similarity (Character TF-IDF)**:
   - Fit a character-level `TfidfVectorizer` once and transform names in bounded batches.
   - Calculate sparse cosine similarity between Source 1 and candidate name vectors.
   - Never fit a new vectorizer per candidate pair.
2. **Address Similarity**:
   - Calculate character TF-IDF cosine similarity between normalized addresses.
   - Preserve `address_is_missing` when the normalized address is empty.
3. **Additional features**:
   - `name_is_missing`
   - `address_is_missing`
   - `country_match`
   - Optional name-length sanity features
4. **Combined baseline metric**:
   $$\text{combined\_score} = 0.6 \times \text{name\_similarity} + 0.4 \times \text{address\_similarity}$$

### 3.2 Decision Threshold / Classifier
- Filter candidate pairs:
   $$\text{is\_match} = \text{combined\_score} > \text{threshold}$$
- Tune the threshold on an internal holdout using macro $F_{0.5}$. No final threshold is
  committed until the TF-IDF feature implementation is evaluated.
- An entity may match **zero, one, or multiple** candidates. Keep all candidates above threshold!

---

## 4. Expected Output Format (`output/matching_results.tsv`)

Your final output **must** be written to `output/matching_results.tsv`. This is the **only file scored on the leaderboard**.

### File Format Requirements
1. **Delimitation**: Strict Tab-Separated (`.tsv`), NOT comma-separated!
2. **Exact Header**: `source1_entity_id\tmatched_entity_ids`
3. **Every Source 1 Entity Must Have Exactly One Row**:
   - If test set has 1,732,544 S1 entities, the output file must have exactly 1,732,545 lines (1 header + 1,732,544 rows).
4. **Singletons (No Match)**:
   - Must be represented as an **empty string**.
   - Example line: `S1-00003\t\n`
5. **No Duplicate IDs**: No duplicate IDs within a single comma-separated list.
6. **Subset of Candidates**: Matched IDs must only come from the candidate set in `output/candidate_pairs.tsv`.
7. **Only S2- and S3- IDs**: Self-matches (`S1-`) will cause immediate rejection.

### Example `matching_results.tsv`:
```text
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### ⚠️ Critical Polars Bug to Avoid!
If writing with Polars, **never** use default `write_csv(separator="\t")`. Polars by default outputs empty strings as `""` (two quote characters), which causes the validator to fail with `matched_entity_ids contains IDs without an S2-/S3- prefix: ""`.

**Always write with `quote_style="never"` and `null_value=""`**:
```python
matching_df.write_csv(
    "output/matching_results.tsv",
    separator="\t",
    include_header=True,
    quote_style="never",
    null_value="",
)
```

---

## 5. Metric Formula: Why Macro $F_{0.5}$ Dictates Your Strategy

The challenge evaluates **macro-averaged $F_{0.5}$**:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Key Metric Implications
1. **Precision is weighted 2× over Recall**:
   - A false merge (linking two different businesses) destroys your score much faster than a missed link.
   - When in doubt on borderline candidates (e.g. score between 0.35 and 0.40), **bias towards rejecting the match**.
2. **Singletons Score 1.0**:
   - A Source 1 entity with no true matches earns a full 1.0 if you predict an empty string `""`.
   - If you falsely predict even a single match for a true singleton, its score drops to 0.0.
   - Do not predict guesses on singletons!

---

## 6. How to Validate Before Submitting

Always run the official submission validator locally before uploading to the leaderboard:

```bash
python data/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/student_resource/dataset/test
```

- **Exit code 0 (`PASS`)**: Format is 100% compliant and safe to submit.
- **Exit code 1 (`FAIL`)**: Fix the printed errors immediately before submitting.

---

## 7. Baseline Target to Beat

The values below are historical trigram-Jaccard planning references only. They are not current
TF-IDF results and must be remeasured after feature engineering is implemented.

| Configuration | Macro $F_{0.5}$ | Precision | Recall |
|---|---|---|---|
| Raw Blocking Candidates | $\approx 0.02$ | $\approx 0.02$ | $93.5\%$ |
| Trigram Jaccard Threshold $\ge 0.50$ (historical) | $0.7045$ | High | Moderate |
| Trigram Jaccard Threshold $\ge 0.40$ (historical) | $0.8143$ | Balanced | Strong |
| Target for Feature Engineering / ML Model | **$> 0.8500$** | High | High |

Your goal in Feature Engineering and Matching is to build on this baseline and push Macro $F_{0.5} > 0.85$!
