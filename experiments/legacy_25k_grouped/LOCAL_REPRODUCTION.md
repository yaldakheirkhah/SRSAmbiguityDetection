# Local reconstruction prerequisite

This is source for an exploratory partial recovery of historical original-content
matches. It is not a full 24,949-row lineage-verified experiment and is ineligible
for the original frozen final Test. The public release contains no requirement
records, generated grouping reports, model weights or executable author approval.

Restore only data you are permitted to use locally at repository-relative
`data/raw/dataset.csv` and `final.csv`. Do not restore or use the original frozen
validation/Test as this experiment's benchmark. The source's fixed protocol is
`config/protocol.json`; its exact historical `config/protocol.sha256` is included.
Do not change it after observing model performance.

`python scripts/build_benchmark.py` reconstructs candidate grouping, quarantine,
and the separate train/validation benchmark. `python scripts/run_tfidf_baseline.py`
and `python scripts/verify_outputs.py` operate on those newly generated outputs.
These are deliberate research commands, not installation or standard tests.
Use a clean experiment copy; preserve any existing outputs and reports. The
source release creates missing generated directories on explicit invocation.
Grouping does not establish complete parentage or semantic independence.

Before the GPU notebook can run, actual local files must exist at:

- `data/train.csv` and `data/validation.csv`, from the reconstruction.
- `config/protocol.json` and `config/protocol.sha256`, supplied source metadata.
- `reports/QUALITY_GATE.json`, produced by the reconstruction and passing all
  recorded checks; never fabricate this file or change failed checks to pass.

Then, from this experiment directory, explicitly bind only those local identities:

```text
python scripts/prepare_local_manifest.py --root . --profile legacy --authorized-local-development-data --execute
```

This writes a new `package_manifest.json` containing only names/hashes/sizes and
control metadata. It does not copy data, build a ZIP, validate redistribution
rights, prove parentage, or replace the notebook's group/row/label checks. It
refuses to overwrite an existing manifest. If input checks fail, stop before
training and investigate the evidence; do not revise a split to improve F1.

The notebook's training/input-check code cells are preserved unchanged. A release
preparation note is added. Run All requires the prepared local manifest, actual
passing gate and authorized records; it intentionally cannot run from public
source alone. The historical broad `package_gpu.py` is omitted because it copied
records and private audit reports. This metadata helper is its narrower local
identity-binding replacement, not a claim that a GPU experiment was executed.

This source subset does not include a dedicated historical GPU environment lock.
Record the actual Python/PyTorch/CUDA/Transformers versions and the notebook's
runtime checks for any new execution; do not call a new environment an exact
historical reproduction. Keep all derived records, predictions and local
manifests private unless their redistribution is separately authorized.
