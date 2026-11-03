# Guarded data tools

Importing these modules does not read files, create output, fit models or evaluate anything. All commands require explicit input paths, a new output directory and `--execute`. They reject held-out source filenames. No command supplies a held-out input by default.

The frozen manuscript partitions and their recorded hashes are authoritative. Do not recreate or overwrite them. `create_splits.py` documents the historical algorithm in an isolated new reproduction: NFKC/whitespace/casefold grouping, quarantine all conflicting-label groups, keep the first consistently labeled requirement per group, stable SHA-256-derived IDs, seed42 stratified 15% then 15/85% splitting. Punctuation and semantic paraphrases are not merged; zero normalized overlap is not proof of full independence. The release copy adds guards and diagnostics without changing that algorithm. The original historical source and evidence remain preserved locally.

For a separately authorized source CSV, an explicit reproduction command is:

```text
python scripts/create_splits.py --input PATH_TO_AUTHORIZED_SOURCE.csv --output-dir NEW_REPRODUCTION_DIRECTORY --encoding latin-1 --execute
```

The output directory must not exist and cannot be inside any `data/splits` location. This creates NEW train, validation, held-out-reproduction and quarantine CSVs plus an aggregate audit; it never reads an existing held-out partition and never runs model evaluation. Running this is optional and is not part of installation or standard tests.

Audit two authorized development partitions without changing their contents:

```text
python scripts/audit_datasets.py --train PATH_TO_TRAIN.csv --validation PATH_TO_VALIDATION.csv --output-dir NEW_AUDIT_DIRECTORY --execute
```

The audit records aggregate label counts, exact/normalized duplicates, cross-label groups, development overlap, and explicit-input hashes. Punctuation equivalence is diagnostic only. It does not relabel or remove anything. Stable unique `source_id` values and all six original numeric labels are required.

Run the historical TF-IDF/logistic-regression validation baseline only when explicitly desired:

```text
python scripts/baseline_tfidf.py --train PATH_TO_TRAIN.csv --validation PATH_TO_VALIDATION.csv --output-dir NEW_BASELINE_DIRECTORY --execute
```

Vocabulary, IDF and class-balanced logistic regression are fitted only on training Text/labels. Validation is prediction-only. Shared source IDs or normalized text across partitions stop fitting; inherited punctuation variants are reported. The recipe is unigrams/bigrams, `min_df=2`, 50,000 features, `max_iter=1500`, balanced weights and seed42. Raw Text is unchanged; standard TF-IDF tokenization remains part of the estimator. Six-label Macro/Weighted Precision, Recall and F1, accuracy and per-class results are written as validation metrics. Prediction exports contain derived source IDs/labels and therefore require a redistribution review before publication. These tools do not export model checkpoints or run final evaluation.

Run offline software tests from the release root:

```text
python -m pytest tests/test_data_tools.py
```

Those tests create invented CSVs in temporary directories, verify deterministic split counts and quarantine/deduplication behavior, and replace the baseline model with a spy. They do not train a classifier, load pretrained weights, read frozen records, or report manuscript metrics. The tests require pandas, scikit-learn and pytest. Reproduction also depends on the recorded Python/library versions; a seed does not establish cross-version bitwise equivalence.
