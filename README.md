# AmbiFix: reproducibility release

Research code for **AmbiFix: An Interactive System for Ambiguity Detection and Resolution in Software Requirements**. This release separates the frozen six-class BERT classifier, historical development experiments, and a mock-tested Human-in-the-Loop (HITL) Streamlit Prototype.

## Final classifier evidence

The final classifier is a **new original-only Apple Silicon MPS training run**, not recovery of the historical BERT model. It used 4,731 frozen original training requirements, 1,015 validation requirements, seed 42, and training-only square-root class weights. Epoch 41 was selected by the literal maximum validation Macro F1 over 49 completed epochs. The frozen checkpoint was evaluated once on 1,015 held-out requirements on November 2, 2026, under explicit author authorization.

| Partition / checkpoint | Macro F1 | Weighted F1 | Accuracy |
|---|---:|---:|---:|
| Validation, final Mac checkpoint | 0.568594 | 0.695547 | 0.702463 |
| **Final held-out Test, same frozen checkpoint** | **0.549072** | **0.690482** | **0.699507** |
| Historical BERT validation reference, different unavailable checkpoint | 0.581140 | 0.699511 | 0.705419 |

Full-precision aggregate metrics, class order, confusion counts and artifact identities are in [results/frozen_bert_summary.json](results/frozen_bert_summary.json). Metrics were independently checked from saved prediction exports during the completed experiment. Release checks recompute arithmetic from public aggregate confusion counts; they do **not** repeat inference or constitute a clean-machine reproduction of training. Historical validation selection and ensemble experiments are [separate evidence](docs/HISTORICAL_EXPERIMENTS.md). The former 0.9259 claim is not an independent final-test result.

Residual punctuation-equivalent relationships and uncertain project/family independence remain. These results do not establish state of the art, industrial effectiveness, automatic ambiguity resolution, reinforcement learning, or a causal augmentation benefit. See [reproducibility and limitations](docs/REPRODUCIBILITY.md) and [the release audit](docs/RELEASE_AUDIT.md).

## Safe setup and checks

Python 3.12 is the documented training interpreter. From a clean checkout, create an environment and install the lightweight release/test dependencies:

```sh
python3.12 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python scripts/run_release_checks.py
python scripts/verify_published_results.py
```

These checks use synthetic fixtures, source guards and aggregate evidence. They do not need private data, download weights, train, or run final evaluation. Ordinary `python -m pytest` is restricted to public root and Prototype tests. CI follows the same offline boundary. Dependencies are pinned directly; a new clean-machine installation was not performed during this release review.

Recreate figures from actual aggregate results and epoch history, without inference:

```sh
python scripts/plot_published_results.py --output-dir generated_figures
```

The command writes PNG, PDF and SVG figures to a new directory and refuses overwrite. It does not read requirement-level data. [Training and reproduction instructions](docs/REPRODUCIBILITY.md) distinguish intentional training notebooks from setup/testing. **The final held-out evaluation is already consumed. Do not rerun it.** Its provenance script is inert without separate, bound authorization; no live approval, checkpoint or held-out dataset is included.

## Prototype

```sh
python -m streamlit run prototype/app.py
```

The default backend is explicitly **mock**. The original Prototype passed 63 mock/contract tests; this release adds safety regressions. Real BERT inference through this application and human effectiveness have **not** been verified. The historical reference contract is not authorization to integrate the final Mac model. See [Prototype scope](prototype/README.md).

## Source layout

- `scripts/`: guarded dataset audit/split construction, validation TF-IDF baseline, aggregate metric verification/plotting and safe test runner.
- `experiments/bert_mac_m3pro_20261101/`: executed final Mac recipe and source; `bert_original_retrain_20261031/`: distinct NVIDIA recipe.
- `experiments/lineage_backtranslation_mac_20261027/`, `lineage_augmentation_bert_20261027/`: provisional-label augmentation and variant preparation, excluded from the final original-only model.
- `experiments/glove_fair_baselines_20261028/`, `legacy_25k_grouped/`: separate development baselines and exploratory grouping audit.
- `prototype/`: Streamlit code, local review storage/export and mock tests.
- `results/`: aggregate classifier evidence and actual epoch history; `docs/`: provenance, rights and manuscript consistency checklist.

## Data, checkpoints and licensing

**Source/derived requirement redistribution permission has not been confirmed.** Data, record-level predictions, lineage exports, checkpoints, caches, environments and evidence ZIPs are excluded. Dataset acquisition and hash verification are described in [DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md). Checkpoints are preserved privately, referenced by hashes; no public model download is promised. Complete training reproduction requires separately authorized access to the exact frozen development files. Regenerating a split does not substitute for verifying those bytes.

This repository currently has no explicit permissive source-code license. The earlier “academic review purposes” statement does not establish dataset redistribution permission. Third-party dependencies and pretrained assets retain their own terms. A formal code-license decision remains with the author.

The known manuscript copy needs the final-result updates listed in [MANUSCRIPT_CONSISTENCY.md](docs/MANUSCRIPT_CONSISTENCY.md); this code release does not silently edit the manuscript. Original local experimental evidence remains preserved. Existing public Git history contains historical dataset/notebook content; removing it from this release tree does not erase that history.
