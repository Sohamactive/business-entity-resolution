# AGENTS.md — Business Entity Resolution (Amazon ML Challenge 2026)

This file gives any AI coding agent (Claude Code, Cursor, etc.) working on this repo the context
needed to continue without re-deriving decisions already made. Read this before writing code.
Full rationale for every decision below lives in `PRD.md` — this file is the condensed,
agent-facing version.

---

## Project summary

Entity resolution task: match business records from **Source 1** (deduplicated reference, 2.2M
train / 1.7M test entities) against noisy records from **Source 2** and **Source 3** (~5M each).
A Source 1 entity may match zero, one, or many Source 2/3 records. Output is scored by **F_0.5**
(precision weighted 2× over recall), computed per-entity and macro-averaged.

**Team goal:** top-100 leaderboard finish. **Deadline:** challenge ends Sept 27, 11:59 PM IST.

---

## Hard constraints — never violate these

- **No external data lookups.** No APIs, no geocoding, no business registry lookups, no internet
  augmentation of any kind at runtime. Instant disqualification if found. Using an AI coding
  assistant to *write* code is fine — the pipeline itself must never call out to external
  services to resolve entities.
- **Final model:** must be MIT or Apache 2.0 licensed, **≤ 8B parameters**. No proprietary/closed
  model APIs as the final matching model.
- **Country is an open set.** Training data has US/India only; test data adds France. Never
  hardcode a country list (`["US","India","France"]`), never write `if country == "US"` branching
  logic. Any country-aware behavior must be config/lookup-driven with a generic fallback for
  unseen values.
- **Output files, exact paths:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`
  (tab-separated, not comma). Header names must match exactly:
  - `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`
  - `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`
- **One row per Source 1 test entity, no exceptions** (empty `matched_entity_ids` for
  singletons). No duplicate entity IDs within a list. No duplicate `source1_entity_id` rows.
  `matched_entity_ids` must only reference S2-/S3- IDs, never S1- (self-match = rejected).
- **Always run `utils/validate_submission.py` before treating any output as submission-ready.**
  It only checks format, never computes score — see `PRD.md §9.1` for what it does and doesn't
  catch.

---

## Architecture decisions already made (do not re-derive — see PRD.md for full rationale)

### Pipeline (the funnel)
```
raw data → normalize → block → candidate_pairs.tsv → feature engineering →
match/no-match decision → matching_results.tsv → validate_submission.py → submit
```

### Normalization (`PRD.md §6`)
- **Script-blind.** One shared normalization function for every language/script (English, Hindi,
  Punjabi, Southern Indian languages, French). No per-language branching, no transliteration, no
  translation (translation risks brushing against the external-lookup rule anyway — avoid).
- Steps: Unicode **NFKC** normalize → lowercase → strip punctuation (`&`→`and`, not deleted) →
  collapse whitespace → (optional/low-priority) digit normalization in addresses.
- Legal-suffix lookup dictionary (`Pvt`→`Private`, `Ltd`→`Limited`, `Corp`→`Corporation`, etc.)
  applied where it matches — harmless no-op on non-Latin scripts, don't build per-language suffix
  dictionaries.
- No script detection/tagging in Iteration 1 — add only as a diagnostic if validation (see below)
  shows a specific country/language underperforming.
- **Known limitation, accepted deliberately:** character n-gram similarity cannot match the same
  business recorded in two different scripts across sources (e.g. Devanagari vs. Latin
  transliteration) — this is a silent recall risk, not a crash. Detection method below.
- Output columns: `normalized_name`, `normalized_address` — everything downstream reads from
  these, never raw columns.

### Blocking (`PRD.md §7`)
- Two keys, computed from normalized fields:
  - **name_key** — sorted-token prefix (split name into words, sort alphabetically, join, take
    first N chars) — order-invariant.
  - **address_key** — coarse geography token (e.g. city/state segment) from the address.
- **Two blocking passes, unioned, not a single combined key:**
  - Pass A: group by `(country, name_key)`
  - Pass B: group by `(country, address_key)`
  - Final candidates per S1 entity = union of both passes.
- Rationale: name and address noise fail independently; union protects recall (blocking misses
  are unrecoverable downstream; extra candidates just get filtered later).
- Implementation: **polars**, not pandas, for these joins (multi-million-row scale).
- Output = `candidate_pairs.tsv`, exactly the set fed to the matching stage — not an earlier,
  unfiltered blocking pass.

### Feature engineering & matching (`PRD.md §8`)
- **String similarity: trigram (character n-gram) Jaccard** for Iteration 1 — script-agnostic,
  word-order robust, vectorizable. `difflib.SequenceMatcher` rejected (not vectorizable, order
  sensitive). TF-IDF char-n-gram + cosine is the planned Iteration 2/3 upgrade (smarter weighting,
  same script-agnostic property) — **flagged for further discussion, not finalized**.
- Features per candidate pair: `name_similarity`, `address_similarity`, `country_match` (binary),
  optional length-difference sanity features.
- **Iteration 1 — threshold rule, no trained model:**
  ```
  combined_score = 0.6 * name_similarity + 0.4 * address_similarity
  is_match = combined_score > threshold
  ```
  Threshold tuned against internal holdout F_0.5 (not accuracy).
- An S1 entity can have multiple true matches — keep **every** candidate above threshold, not
  just the top one.
- **Iteration 3+ (once blocking/features trusted):** swap threshold rule for **LightGBM/XGBoost**
  trained on the same features, labels from `train_ground_truth.tsv`. Satisfies license/size
  constraint trivially.

### Internal validation (`PRD.md §9`)
- `validate_submission.py` = format safety only. Never computes F_0.5. Run before every
  submission regardless.
- Real score estimation = our own holdout split from `train_source1.tsv` +
  `train_ground_truth.tsv`. Compute F_0.5 per-entity, macro-averaged, exactly matching the
  competition formula.
- **Always break down holdout F_0.5 by `country`, and by precision vs. recall separately** — not
  just one blended number. Low recall for a country → likely a blocking/cross-script miss (see
  Normalization limitation above). Low precision → threshold/feature problem, unrelated to
  script. Manually inspect actual missed/wrong cases before concluding the cause.

### Compute environment
- **Kaggle notebooks = primary** (30 hrs/week, stable sessions, dataset mounted at
  `/kaggle/input/<dataset-name>/dataset/...`, no re-upload needed).
- **Colab (free tier) = secondary/parallel**, for smaller tasks only — free tier disconnects on
  idle, unreliable for long blocking jobs.
- **SageMaker = deferred**, not part of Iteration 1/2. Only reconsider if a GPU-bound step (e.g.
  embedding-based similarity at scale) becomes the actual bottleneck later. $150 AWS credit kept
  in reserve for that scenario.
- **Data priority:** Kaggle dataset (primary, for compute) → local machine (source of truth for
  code, git) → Google Drive (fallback/secondary access only).

---

## Repo structure (current)
```
code/business-entity-resolution/src/   # pipeline source — build here
data/student_resource/dataset/         # local copy of train/test data
notebooks/                             # exploratory work
outputs/                               # matching_results.tsv, candidate_pairs.tsv go here
PRD.md                                 # full rationale, decision history, open questions
Documentation_template.md              # methodology write-up — fill in as you build, not at the end
main.py
```

## Explicitly open / unresolved — do not assume, flag instead
- **Documentation length conflict**: problem statement says no page limit; guidelines doc says
  1-2 pages for the initial required artefact. Unresolved — ask the team before finalizing
  `Documentation_template.md` length.
- **"Eligibility criteria"** referenced in guidelines doc for top-100 announcement, never
  defined in either source doc. Unresolved.
- **TF-IDF char n-gram upgrade (Iteration 2/3)** — flagged for further discussion, not committed.
- **Address-key exact extraction logic** — not yet finalized against real address format
  variety; needs a real-data pass before locking in.

## Working conventions
- Every pipeline change → re-run internal holdout validation (§9 above) before considering a
  real leaderboard submission — submissions are limited to 15 total (5/day × 3 days).
- Keep a version-history log of every leaderboard submission (timestamp, what changed, holdout
  F_0.5, public leaderboard F_0.5) — required for shortlisting review.
- Fill in `Documentation_template.md` incrementally as decisions are made, not as a last-minute
  writeup — it doubles as the audit-time methodology doc for top-100 teams.