# New original-data weighted BERT: configuration provenance

> Historical NVIDIA recipe-preparation provenance (October 31), retained for the distinction between recovered facts and new choices. Its BF16 table is **not** the final Mac execution configuration. The separate [Mac FP32 recipe](../experiments/bert_mac_m3pro_20261101/config/training_config.json) and [completed aggregate results](../results/frozen_bert_summary.json) record the actual final run and once-only November 2 Test outcome. Preparation-time pending statements below are historical, not current status.

Prepared October 31, 2026, for the November 2 revision. **This is a new training experiment. The unrecoverable historical checkpoint is not reconstructed, and 0.581140 is not a promised score or a stopping target.** No model was trained during package preparation.

## What the actual historical artifacts establish

The original run summary and 18-row epoch log identify `bert_sqrtweighted_max50_20261018_214144`, selected epoch 10. Saved validation predictions previously verified Macro F1 0.5811396777932097, Weighted F1 0.6995114383470095 and accuracy 0.7054187192118226. These are historical validation measurements; no new score is assigned to this package.

The epoch log directly records LR 0.00002 for epochs 1–14, 0.00001 for epochs 15–18, and 296 successful optimizer updates per epoch. Those facts do **not** uniquely recover batch size, gradient accumulation, scheduler or early-stopping implementation. Epoch 10 of 18 is consistent with patience 8 but does not prove it. Likewise 296 updates is consistent with batch 8/accumulation 2 but is not a recovered batch-setting record.

The original executed source, exact full configuration, initialization revision, tokenizer and training environment are unavailable. The comparison notebook has eight unexecuted code cells and no outputs: it is a **reference only**, not evidence that all its settings were executed in that historical run. The different nearby three-epoch unweighted notebook is not used as a historical reconstruction either.

Historical source identities are recorded in the preserved private evidence. See `docs/HISTORICAL_EXPERIMENTS.md` for the public evidence boundary; original record-bearing archives are not redistributed here.

## Fixed new configuration

| Setting | New run choice | Evidence status / reason |
|---|---|---|
| Training data | Frozen original train, 4,731 rows | Exact existing bytes/counts checked; no augmentation or split generation |
| Model selection data | Frozen validation, 1,015 rows in original order | Exact existing bytes/counts checked; used only for selection/evaluation |
| Initialization | `google-bert/bert-base-uncased`, commit `86b5e0934494bd15c9632b12f734a8a67f723594` | New immutable original-pretrained choice; historical commit unknown; no fine-tuned initialization |
| Original pretrained safetensors SHA | `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3` | Recorded pinned-revision metadata; byte verification is a required GPU-run gate, not a claimed completed NVIDIA execution |
| Input | Original `Text`, raw, standard uncased tokenizer only | Explicit new contract; no external stemming, normalization, stopword removal or contraction expansion |
| Classes | Dataset IDs 1–6 → model indices 0–5 | Supplied trusted exported class convention; preserve Syntactic (2) and Syntax (4) separately |
| Class weights | `sqrt(N_train/(6*n_c))`, normalized to mean 1 | User-required formula; newly computed from original training labels only |
| Token maximum | 192, fast tokenizer, dynamic batch padding | New choice from unexecuted reference; not recovered actual historical token limit |
| Seed | 42 | Documented reference-based new choice; actual old seed unconfirmed |
| Physical/evaluation batch | 8 / 16 | Reference-based new choices, not proven by the old update count |
| Accumulation | 2 (nominal effective batch 16) | Reference-based new choice; corrected weighted reduction below |
| Optimizer | AdamW, LR 2e-5, decay 0.01, betas (0.9,0.999), epsilon 1e-8 | LR corroborated by actual log; other optimizer/settings explicitly new reference/default choices |
| Scheduler | ReduceLROnPlateau, max, patience 3, factor 0.5, min LR 2e-7, threshold 0.001, threshold mode `rel` | New explicit reference-based choice; old LR trace alone does not prove this scheduler |
| Epochs | Maximum 50, minimum 8 | New reference-based choices; 18 historical logged epochs is not proof of its cap |
| Checkpoint selection | Every strict increase in global six-class validation Macro F1; earliest exact tie | Explicit new correctness rule; no 0.001 threshold on saving the literal best |
| Early stopping | Patience 8, meaningful improvement >0.001, separate clock | New reference-based stopping choice; not inferred solely from historical epoch count |
| Gradient clipping | Norm 1.0 | New reference-based choice |
| Training precision | BF16 forward, FP32 parameters/loss; no automatic FP16/CPU fallback | New fixed supported-GPU contract |
| Validation | FP32 forward, TF32 disabled, FP64 softmax of FP32 logits | Explicit new full-precision evaluation choice; old actual precision unknown |
| Attention | Eager | Explicit new implementation choice; old actual attention implementation unknown |
| Gradient checkpointing | Disabled | New reference-based setting |
| Other model defaults | Unoverridden architecture/dropout defaults from the immutable original pretrained configuration; complete actual config saved | Explicit new default policy, not a recovered historical setting |
| Additional loss policy | No label smoothing or resampling; ordinary weighted cross entropy | Explicit fixed new implementation choice |
| Search/retries | One fixed seeded training run; no sweep or automatic trial restart | Protects the reused validation set from additional selection |

The new implementation deliberately corrects separately normalized microbatch-loss averaging: gradients of summed weighted cross-entropy numerators are accumulated, divided once by the **entire accumulation window's sum of target-class weights**, then clipped and stepped. Uneven last windows use their actual denominator. This is the weighted-mean loss for the effective batch; it is not necessarily identical to the missing historical implementation.

Likewise, a small literal Macro F1 improvement is saved even when it does not reset the meaningful-improvement stopping clock. Full FP32 validation differs from the unexecuted reference's AMP-forward/FP32-softmax pattern; FP64 exported probabilities do not mean FP64 neural inference.

## Computed training-only weights

| Dataset class | Train count | Normalized square-root weight |
|---|---:|---:|
| 1 Lexical Ambiguity | 324 | 1.2357818674339447 |
| 2 Syntactic Ambiguity | 418 | 1.0879932043754172 |
| 3 Semantic Ambiguity | 415 | 1.0919186285678970 |
| 4 Syntax Ambiguity | 332 | 1.2208021396880808 |
| 5 Pragmatic Ambiguity | 568 | 0.9333405954644729 |
| 6 Clean | 2674 | 0.4301635644701878 |

The machine-readable values are authoritative in [`class_weights.json`](../experiments/bert_original_retrain_20261031/config/class_weights.json). Validation supports [70,89,89,71,122,574] are audited but never used to fit weights. Loss weighting is distinct from weighted evaluation F1.

## Scientific limits

This run is a new fixed-recipe unaugmented comparator on an already reused validation partition. Differences from 0.581140 cannot establish exact historical reproduction, augmentation effects, statistical superiority, multi-seed robustness or final generalization. No threshold, split, label or recipe is changed to increase validation F1.

No exact/NFKC-whitespace-casefold training/validation overlap is present. Frozen originals retain 19 punctuation-equivalent training duplicate keys (three with differing labels) and ten punctuation-equivalent train/validation pairs (one differing-label pair). Those heuristic equivalences and known near-duplicate/source-independence/annotation limitations remain; nothing is removed or relabeled. C1 controls remain in 61 training and ten validation requirements. The grouping normalization is an integrity check, not a rewrite of model input.

The held-out partition is absent from the package and has no automatic evaluation entry point. After returning the actual new checkpoint/results, freeze their identities, independently verify the artifacts and agree a final evaluation protocol. Only a later explicit authorization may permit one final held-out evaluation.
