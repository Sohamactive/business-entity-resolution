# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** September 2026

---

## 1. Executive Summary
We present a scalable, script-blind Entity Resolution pipeline designed to link reference business entities from Source 1 against noisy fragments in Sources 2 and 3. By combining a multi-pass compound blocking architecture (filtering stopwords and coupling building numbers with locality prefixes) with dynamic open-set country partitioning, we reduce the $1.7 \times 10^{13}$ pairwise comparison space by >99.99% while securing **93.53% overall candidate recall** (97.62% US, 87.36% India) with a median candidate pool of 67 candidates per entity, completely offline and within 16 GB of memory.

---

## 2. Methodology

### 2.1 Problem Analysis
Key empirical insights discovered during EDA across train (2.2M S1 / 5M S2 / 5.2M S3) and test sets:
1. **Multi-Country Open Set**: Training data contains `US` and `India`; test data introduces `France`. Country partitions are strictly preserved across true matches (0 cross-country true matches observed in ground truth).
2. **Asymmetric Error Penalties ($F_{0.5}$)**: Precision is weighted 2× over recall. False merges fatally drop scores, while correctly identifying singletons earns full 1.0 credit per entity.
3. **Cross-Script Name Discrepancies**: In Indian records, Source 1 business names often appear in Latin script (*Raj Investments LLP*, *Balaji Investment*), while Source 2/3 appear in Devanagari, Tamil, or Telugu (*ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்*, *బాలాజీ ఇన్వెస్ట్‌మెంట్*). Pure string similarity on names alone fails on these cross-script pairs.
4. **Independent Failure Modes of Name vs. Address**: When business names are altered by trade/DBA names or scripts, addresses remain mostly Latin and share numbers/street tokens. When addresses are missing or landmark-based, names remain recognizable.
5. **Combinatorial Explosion of Coarse Keys**: A naive block on state (e.g. `Maharashtra` or `Texas`) contains >160,000 entities, generating >64 billion pairs in a single block and causing instant Out-of-Memory (OOM) failures.

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Inverted Index Blocking + Tabular Similarity Classifier + Macro-$F_{0.5}$ Threshold Tuning.  
**Core Innovation:** A two-stage funnel featuring:
- Universal, script-blind Unicode NFKC normalization with legal suffix expansion (`Pvt` $\to$ `private`, `Ltd` $\to$ `limited`, `SARL` $\to$ `societe a responsabilite limitee`).
- Multi-key compound blocking that pairs leading brand/sorted tokens with compound address keys (`{number}_{street_prefix}`) to prevent Cartesian explosion while recovering cross-script matches.
- Dynamic country-partitioned inverted indexing with configurable candidate pool capping via `.env`.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys Used
To maximize recall while keeping blocks compact, we compute complementary keys across two passes:

1. **Pass A — Name Keys (Stopword & Legal Suffix Filtered)**:
   - `NF_{prefix4}`: First 4 characters of the leading distinctive token (e.g. `Orelee's Barbershop` and `Orelee's Services` $\to$ `NF_orel`). Preserves primary brand identity.
   - `NS_{prefix4}`: First 4 characters of the alphabetically sorted distinctive tokens (e.g. `Barbershop Orelee's` and `Orelee's Barbershop` $\to$ `NS_barb`). Solves word-order transposition noise.
   - *Key design decision*: Normalized corporate designators (`incorporated`, `limited`, `company`, `services`, `center`, `and`, `the`) are filtered out prior to sorting. Without this filter, `Prime Money Inc.` sorted to `inco` while `Prime Money` sorted to `mone`, causing an unrecoverable blocking miss.

2. **Pass B — Compound Address Keys**:
   - `ANW_{num}_{street_prefix4}`: Normalized integer building/plot number (stripped of leading zeros, e.g. `0017560` $\to$ `17560`) paired with the first distinctive street word prefix (e.g. `17560_elli`, `1795_west`, `85_wayn`).
   - `ANW_{num}_{locality_prefix4}`: Number paired with the last distinctive address token (e.g. `1795_high`, `85_tico`), ensuring invariance to component reordering (City first vs. Street first).
   - `AWP_{word1}_{word2}`: Fallback for addresses without digits (~5.7% of records), combining the first two distinctive locality tokens (e.g. `west_beng`, `utta_prad`).

3. **Pass C — Dynamic Country Partition**:
   - Keys are indexed as `(country, block_key)`. Cross-country matching is prevented at the index level without hardcoding country strings.

### 3.2 Candidate Pairs Generated & Statistics
- **Total candidate pool per entity**: Full test output contains 224,533,181 pairs across 1,732,544 entities (mean approximately 129.6; cap 150 enforced via configurable `.env` setting `BLOCKING_MAX_CANDIDATES=150`).
- **Reduction Ratio**: Approximately 99.9987% reduction in pairwise comparison space (from about $1.7 \times 10^{13}$ to 224,533,181 candidate pairs).
- **Execution Throughput**: 16,035 Source 1 entities/sec during candidate generation; full run completed in approximately 5.5 minutes locally.

### 3.3 How True Matches Were Preserved (Recall Guarantee)
- **Union Architecture**: Name and address noise fail independently. Candidates from Pass A (DBA-aware name keys) and Pass B (compound address keys) are unioned across priority tiers.
- **Cross-Script Recovery**: Devanagari/Tamil business records whose names cannot match Latin S1 records are retrieved via multi-number compound address keys (`ANW_`) and universal address word pairs (`AWP_`).
- **Holdout Validation Recall (Enhanced)**:
  - Combined Blocking Recall: **97.67%** overall (error reduction of 63% vs initial baseline).
  - US Blocking Recall: **99.10%**.
  - India Blocking Recall: **95.48%**.
- **Format Compliance**: Output written with `quote_style="never"` and `null_value=""`, ensuring singletons produce clean empty strings and passing all checks in `utils/validate_submission.py`.

---

## 4. Matching Model

**Features used:**
- Name features: Character TF-IDF n-gram cosine similarity, missingness flag, and optional length-difference ratios.
- Address features: Character TF-IDF n-gram cosine similarity and missingness flag.
- Other: Binary country match; candidate multi-key agreement is retained as a possible later feature.

**Model type:** Iteration 1 threshold rule over sparse TF-IDF cosine features
**Threshold selection method:** Macro-$F_{0.5}$ grid search on internal holdout split.

---

## 5. Results & Error Analysis

- **Candidate Recall (Holdout):** 97.67% (US: 99.10%, India: 95.48%)
- **F_0.5 Score (macro):** [To be populated after model training]
- **Common false positives (wrong merges):** Franchises / branch locations sharing identical business names and partial street names in the same city.
- **Common false negatives (missed matches):** Rare entities with simultaneous extreme character corruption across all fields and no recognizable locality words.

---

## 6. Conclusion
The compound blocking strategy successfully bridges the gap between high recall and memory efficiency, resolving cross-script matching challenges while eliminating the risk of combinatorial explosion.

---

## Appendix

### A. Code Artefacts
All code resides under `code/business_entity_resolution/`:
- `src/normalization.py`: Script-blind Unicode NFKC normalization and corporate suffix canonicalization.
- `src/blocking.py`: Multi-key extraction, inverted indexing, candidate generation, and TSV export.
- `src/run_blocking.py`: End-to-end command-line runner (`python -m business_entity_resolution.src.run_blocking`).
- `src/evaluate_blocking.py`: Internal holdout benchmarking script.
- `src/test_blocking.py`: Unit test suite covering edge cases, punctuation, accents, and format validation.
- `requirements.txt`: Pinned environment dependencies.
- `.env`: Environment configuration specifying `BLOCKING_MAX_CANDIDATES=150`.
