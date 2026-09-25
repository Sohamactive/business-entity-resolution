# AGENTS.md — ML Challenge 2026: Business Entity Resolution

## Mission

Build a reproducible ML pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. Source 1 is the deduplicated reference set. For each Source 1 entity, find all matching records in Source 2 and Source 3. A Source 1 entity may have zero, one, or many matches.

## Authoritative materials

Use the official problem statement, challenge instructions, supplied dataset, README, validator, and methodology template as the project sources of truth. Do not invent dataset statistics, challenge requirements, experimental results, or findings. Mark unknowns as TODO until measured or confirmed.

## Challenge constraints

- Use only the supplied challenge data for resolving entities. No external databases, APIs, business lookup, geocoding, or internet data augmentation.
- The final model must satisfy the stated MIT/Apache 2.0 license requirement and be no larger than 8 billion parameters. Check model license before use.
- Training countries include US and India; test also includes France. Treat country as an open-set string value. Do not hard-code, filter, or encode only the training country set. Include every test Source 1 entity.
- Challenge runs from Sep 25, 2026 00:00 IST to Sep 27, 2026 23:59 IST. Maximum five leaderboard submissions per day across the three days. Preserve submission versions.
- Use laptop/desktop; no simultaneous logins per participant.

## Data layout and schema

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are tab-separated. Always load with an explicit separator:

```python
import pandas as pd
df = pd.read_csv(path, sep="\t")
```

Each source record has `entity_id`, `business_name`, `business_address`, and `country`. The source is indicated by its file and ID prefix (`S1-`, `S2-`, `S3-`), not a separate source column.

`train_ground_truth.tsv` has `source1_entity_id` and `matched_entity_ids`. The latter is a comma-separated list of matching S2/S3 IDs; an empty value means no matches. Test labels are not provided.

## Definitions

- **Blocking / candidate generation:** shortlist plausible Source 2 and Source 3 records for each Source 1 record before the final matching decision.
- **Candidate pair:** a Source 1-to-Source 2 or Source 1-to-Source 3 comparison passed to the matching model.
- **Final match:** a candidate the system predicts to be the same real-world business.
- **Singleton/no-match:** a Source 1 entity with no correct matches.

The `candidate_pairs.tsv` file must record the final candidate set actually passed to the inference matching model, not an earlier, broader blocking output. Every predicted final match must be included in that entity's candidate list.

## Recommended workflow — proceed one milestone at a time

1. **Read project materials and inspect data.** Confirm file paths, columns, row counts, nulls, ID uniqueness, country values, and label-list format. Show only small samples. Keep raw files unchanged.
2. **Create a reproducible validation split from training.** Hold out Source 1 entities and their labels. Ensure no held-out labels leak into fitting or threshold selection. Test has no ground truth.
3. **Implement a simple, transparent baseline.** Start with conservative text normalization, straightforward blocking, name/address similarity features, and an initial threshold. Do not begin with a large model or automated tuning.
4. **Evaluate using the official metric.** Compute F0.5 per Source 1 entity and macro-average across entities. Track precision and recall as diagnostics; explicitly evaluate correct empty predictions and false matches on true singletons.
5. **Inspect errors.** Review false positives, false negatives, missed true candidates, and erroneous non-empty predictions for true singletons.
6. **Improve blocking.** Experiment with blocking keys and string-based retrieval such as token overlap/Jaccard, edit distance, or TF-IDF cosine. Measure candidate recall and candidate-set size; remember missed candidates cap final recall.
7. **Improve matching.** Add justified name/address features and, if validation evidence supports it, train a pairwise classifier. Tune thresholds only on validation data, prioritizing the official F0.5 objective without ignoring candidate recall.
8. **Run controlled experiments.** Keep the same validation split. Record configuration, blocking strategy, features/model, threshold, score, and notes for each experiment. Change one major factor at a time where practical.
9. **Run test inference.** Predict for every test Source 1 record against test Sources 2 and 3. Preserve valid IDs and output empty lists where appropriate.
10. **Validate and package.** Create both required TSVs, run the supplied validator, then assemble the final ZIP. Do not spend leaderboard submissions on avoidable formatting errors.

Use CPU-friendly methods first. Consider GPU, distributed processing, or SageMaker automatic hyperparameter tuning only after a valid baseline exists and measurements show a benefit or bottleneck.

## Metric

The official metric is macro-averaged F-beta with beta = 0.5. It is precision-heavy and penalizes false merges. Singletons count in the macro-average: a correct empty prediction scores 1.0 for that entity; any predicted match for a true singleton scores 0.0. Optimize and report the actual challenge metric, not just pairwise accuracy.

## Required prediction files

Both files are tab-separated and belong in `output/`:

- `matching_results.tsv`: columns `source1_entity_id`, `matched_entity_ids`. Exactly one row for every test Source 1 ID. Values are comma-separated S2/S3 IDs or empty. No duplicates; IDs must exist in test S2/S3.
- `candidate_pairs.tsv`: columns `source1_entity_id`, `candidate_entity_ids`. Exactly one row per test Source 1 ID. Values are comma-separated S2/S3 candidate IDs or empty. No duplicates; IDs must exist in test S2/S3. Final matches must be a subset of candidates.

The leaderboard upload is `matching_results.tsv`. The final archive additionally includes `candidate_pairs.tsv`, runnable code, dependency pins, and the completed methodology document.

## Final archive structure

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The code folder must be self-contained and document exact end-to-end run instructions. Pin dependencies. Keep raw data separate from generated outputs and avoid committing credentials or private data to a public repository.

## Submission validator

Run from the extracted `student_resource/` directory, adjusting relative paths only if the working layout differs:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The validator checks file structure and ID constraints; it does not calculate F0.5.

## Methodology write-up template

Fill the supplied `Documentation_template.md`; do not fabricate or prematurely fill in results. Keep this structure and update bracketed prompts with measured, accurate information:

1. **Team details:** team name, team members, submission date.
2. **Executive Summary:** 2–3 sentences describing the actual approach and key innovations.
3. **Methodology**
   - **Problem Analysis:** EDA findings about noise, address/name variation, missing fields, etc.
   - **Solution Strategy:** approach type (e.g. blocking + classifier, end-to-end, graph-based, hybrid) and actual core innovation.
4. **Candidate Generation (Blocking):** blocking keys, total candidate pairs, and how candidate recall was assessed/protected.
5. **Matching Model:** name, address, and other features; model type; validation-based threshold selection method.
6. **Results & Error Analysis:** best macro F0.5 validation score and observed false-positive/false-negative patterns.
7. **Conclusion:** 2–3 sentences on actual approach, results, and lessons.
8. **Appendix**
   - **Code Artefacts:** summarize source structure and exact entry point(s) for reproducing both output files.
   - **Additional Results:** optional additional charts/tables/details.

The challenge instructions request a 1–2 page approach document; the problem statement says the provided methodology template has no page limit and prioritizes clarity and technical depth. Follow any current portal instructions and keep the write-up clear and complete.

## Engineering and collaboration

- Use small, documented modules under `src/` (e.g. loading, normalization, blocking, features/scoring, evaluation, inference, output writing).
- Make paths configurable; avoid machine-specific absolute paths.
- Fix random seeds for reproducibility.
- Avoid validation leakage.
- Keep a simple experiment log and preserve submission versions.
- Use Git branches or clearly assigned workstreams when collaborating; integrate and test changes before submission.
- Do not claim a model, score, result, or finding until it has actually been run and checked.

## Immediate next action

Inspect the actual training and test files and report their real columns, row counts, missingness, and label representation. Then proceed to a reproducible validation split. Do not assume illustrative examples in the challenge documents are actual data.
