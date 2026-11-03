# Reproducibility protocol and boundaries

Updated November 3, 2026. This release preserves the executed Mac method, separate NVIDIA and augmentation recipes, synthetic tests and aggregate evidence. Release preparation performs no training, model inference or access to the held-out source partition. Original local evidence, model, freeze manifest, consumed lock and hashes are not altered.

## Frozen original-only experiment

Six dataset IDs 1–6 map to model indices 0–5: Lexical Ambiguity, Syntactic Ambiguity, Semantic Ambiguity, Syntax Ambiguity, Pragmatic Ambiguity, Clean. Syntactic and Syntax remain distinct original classes. Use raw `Text`; integrity normalization does not rewrite model input. The fast standard uncased BERT tokenizer dynamically pads and truncates at 192 tokens. No augmentation, resampling, relabeling or external text transformation belongs to this final classifier.

Frozen development inputs contain 4,731 training and 1,015 validation rows. Original source records numbered 7,061 before corrected splitting. Hashes/counts and training-only weights are in each original-only experiment's `data_manifest.json` and `config/`. Train hash: `c5d623c1738400170fe7b5feb23a82d4a3c15f071a77fe5c4bfba8c6c5cc70f8`; validation: `9b5c7c76dad306bff4b306753d04a0169f84dd6f27eb8312bc63f5f3322bef42`. These are recorded identities, not redistribution permissions. Do not regenerate frozen inputs merely to satisfy counts.

Train-only class counts are [324,418,415,332,568,2674]. Define `a_c = sqrt(N_train/(6*n_c))` and `w_c = a_c/mean(a)`. The fixed training loss for a complete gradient-accumulation window W is:

```text
L_W = -sum(i in W, w[y_i] * log p_i[y_i]) / sum(i in W, w[y_i])
```

All valid examples in the full window, including the final partial window, contribute to both sums. The denominator is the sum of target-class weights, not the number of examples. The implementation backpropagates summed weighted numerator losses per microbatch, divides accumulated parameter gradients by the complete window denominator, then clips and steps once. Separately averaging weighted-mean microbatch losses is a different objective. This matches the integer-class-target weighted-mean definition in the [PyTorch 2.9 CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html). No label smoothing or ignored examples are configured.

Fixed settings: seed 42; train microbatch 8, accumulation 2, nominal effective batch 16 (final window 11); evaluation batch 16; AdamW LR 2e-5, decay 0.01, betas (0.9, 0.999), epsilon 1e-8; norm clipping 1.0; eager attention, no gradient checkpointing. ReduceLROnPlateau uses max mode, patience 3, factor 0.5, min LR 2e-7, relative threshold 0.001. Cap 50/minimum 8 epochs; meaningful-improvement stopping patience 8 with delta 0.001. Saving separately uses **every strict increase in global six-class validation Macro F1**, retaining earliest exact ties. No split/threshold/recipe was revised after observing results.

Initialization is original pretrained `google-bert/bert-base-uncased`, immutable revision `86b5e0934494bd15c9632b12f734a8a67f723594`, pretrained weight SHA `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`. No previously fine-tuned checkpoint was used. Historical settings were incompletely recoverable; [hyperparameter provenance](HYPERPARAMETER_PROVENANCE.md) distinguishes logged facts from fixed new choices. Historical 0.581140 is not a target or the score of the final checkpoint.

## Mac versus NVIDIA

`experiments/bert_mac_m3pro_20261101/` contains the executed Apple Silicon MPS FP32 method. AdamW and gradient clipping use `foreach=False`; CPU operator fallback is prohibited. `experiments/bert_original_retrain_20261031/` targets RTX 5060 Ti/CUDA 12.8 with BF16 training. Both define full-window weighted normalization and FP32 neural validation with CPU FP64 probability export, but hardware/precision executions are distinct. No bitwise cross-hardware equivalence or completed NVIDIA result is asserted. Mac completed 49 epochs, selected 41, with 14,504 optimizer updates. Direct dependency pins are in `requirements_mac.txt`; transitive/hardware execution snapshots remain in private evidence.

## Intentionally reproduce training

This optional resource-intensive action is separate from setup/tests. Obtain lawfully authorized **exact frozen development files**, then place byte-exact copies at `experiments/bert_mac_m3pro_20261101/data/train.csv` and `data/validation.csv`. No held-out file is needed. Verify development inputs first using the verifier's documented CLI (see `--help`).

The notebook `notebooks/Train_Original_Weighted_BERT_Mac_M3Pro.ipynb` is an explicit training action: Run All performs resource/dependency checks, a train-only smoke and the fixed run. Create a dedicated native Apple Silicon Python 3.12 environment with the experiment's pinned requirements. MPS is required; no silent CPU fallback. The NVIDIA notebook is a separate explicit training workflow. Missing data/assets stop these workflows; neither notebook evaluates final Test.

For fresh reproduction, preserve original configuration/manifest bytes. Configuration fields `PREPARED_NOT_TRAINED` and `test_evaluation_implemented=false` are immutable **pre-execution recipe metadata**, not present outcomes; current outcome is `results/frozen_bert_summary.json`. New runs write new outputs and must be identified separately. Existing run/lock controls prohibit overwrite. Never remove a consumed final-evaluation marker or recreate authorizations.

## Public checks and figures

`python scripts/run_release_checks.py` runs curated suites in separate subprocesses to prevent identically named experiment modules contaminating each other. Private development-data tests are explicitly skipped in release copies; originals remain locally preserved. Root pytest collection is scoped to root/Prototype synthetic tests. Existing separate UI environments can be selected using `run_release_checks.py --prototype-python PATH_TO_UI_PYTHON`; this avoids combining incompatible NumPy/Arrow installations. No setup, CI or test command invokes training, actual-data splitting, model loading or final inference.

`verify_published_results.py` independently recomputes precision/recall/F1/support and accuracy from the recorded 6×6 aggregate confusion matrices, checks all averages and validates selected epoch against actual history. This is not renewed sample-level evaluation. `plot_published_results.py` uses aggregates and unsmoothed epoch values to recreate loss, validation Macro F1, raw and row-normalized validation/Test confusion matrices and per-class F1 charts in PNG/PDF/SVG. Neither accepts a dataset/checkpoint argument.

For rights-cleared private record-level exports, preserved `verify_artifacts.py` and `verify_final_evidence.py` check IDs, labels, probability order, metrics and checkpoint identities. Their required original inputs are deliberately absent publicly. Do not manufacture missing exports. `final_test_guarded.py` is executed provenance only: final evaluation was consumed November 2; no live authorization is distributed and no rerun is authorized by this release.

## Limits

Macro F1 equally averages six class F1s; weighted F1 uses evaluation supports, independent of training loss weights. Accuracy is correct/N; weighted recall equals accuracy for this single-label problem. Final Test is 710/1015 correct; its Macro F1 is 0.0195225550 below validation. No uncertainty/significance claim follows from one seed or exploratory reuse of validation.

Frozen originals retain ten punctuation-equivalent train/validation pairs (one cross-label), 19 punctuation-equivalent training duplicate keys (three cross-label), and C1 controls in 61 train/10 validation records. Completed final evidence additionally records seven punctuation-equivalent train/Test pairs (one cross-label) and one validation/Test pair. These are heuristic dependence/annotation warnings; no rows were removed, relabeled or rescored after Test. Project/document/semantic-family independence remains unknown. Avoid an unqualified “leakage-free” claim.

The Prototype originally passed 63 mock/contract tests; release boundary regressions do not establish real BERT integration, human performance, automatic resolution, deployment or online learning. The available manuscript still needs [final-result edits](MANUSCRIPT_CONSISTENCY.md). Rights and checkpoint availability remain explicit limits of public training reproduction.
