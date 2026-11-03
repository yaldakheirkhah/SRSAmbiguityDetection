# Local lineage-variant reconstruction prerequisite

This source is for the separate exploratory augmentation experiment. Synthetic
labels remain **UNVERIFIED**, including unflagged Variant A records. It does not
replace the original-only final Mac experiment, establish a controlled causal
augmentation gain, or authorize final Test evaluation.

The release deliberately excludes original/derived requirement records, generation
journals, private review tables, model weights and live approvals. Restore only
authorized local development inputs, preserving their exact bytes:

- Repository `data/splits/train.csv` and `data/splits/validation.csv`.
- The lineage-generation experiment's `runs/full/train_augmented_PROVISIONAL.csv`,
  `synthetic_accepted_PROVISIONAL.csv`, `generation_config.json` and `model_lock.json`.
- Its `reports/candidate_suspicious_flags.csv` and independently generated
  `reports/INDEPENDENT_OUTPUT_AUDIT.json`.

`python scripts/build_variants.py` verifies those inputs and writes separate A/B
data plus its actual `reports/DATASET_VERIFICATION.json`. It preserves all 4,731
original training rows and the unchanged validation. Use a clean experiment
copy and never remove original rows or relabel synthetic data automatically.
The release creates missing generated directories and, before exporting data,
checks that both recomputed weight records match any included historical expected
records. Matching files retain their exact original bytes; mismatches stop.
CSV serialization bytes and all selection/weight formulas remain unchanged.
`python scripts/verify_variants.py` requires the actual source generation evidence
and produces a new independent review; no successful review is fabricated here.

The notebook requires four development data files under `data/`:
`train_original.csv`, `validation.csv`, `train_A_Conservative_v1.csv`, and
`train_B_Full_Provisional_v1.csv`; the included training configuration and separate
A/B class-weight records; and the actual passing dataset-verification report.
After those exist, run from this experiment directory:

```text
python scripts/prepare_local_manifest.py --root . --profile lineage --authorized-local-development-data --execute
```

This source-only helper checks original development hashes and the bindings among
variant data, saved weights, configuration and gate. It writes only a new local
`package_manifest.json`; it neither copies/publishes records nor packages a ZIP.
It refuses missing inputs, unsuccessful gates, mismatched identities and manifest
overwrites. The notebook still independently checks row order, parents, flags,
duplicates, labels and training-only weights before importing the ML stack.

Run All deliberately fails closed from source alone. Its scientific code cells
remain unchanged; only a public-release preparation note is added. The historical
broad `package_gpu.py` is omitted because it copied research records and audit
reports. Use the included GPU requirements and record the actual CUDA/device and
dependency versions for any new exploratory execution. Do not infer real execution
or label correctness from this prepared source. Keep generated records and
predictions private pending permission to redistribute them.
