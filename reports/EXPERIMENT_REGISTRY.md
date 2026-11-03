# Current release experiment registry

Updated November 3, 2026. This current-facing registry supersedes historical pending statements without modifying preserved private reports. No scientific procedure was repeated during release preparation.

| Experiment / evidence | Scope | Macro F1 | Weighted F1 | Accuracy | Status and limit |
|---|---|---:|---:|---:|---|
| Final original-only Mac BERT, epoch41 | Validation1015 | 0.5685944251323821 | 0.6955474743459408 | 0.7024630541871921 | Completed and checkpoint replay verified in original execution evidence; public aggregates checked again during release. |
| Same frozen Mac model | Final held-out Test1015 | 0.5490718700859288 | 0.6904821963220387 | 0.6995073891625616 | Once-only authorized November2 evaluation; no release rerun. |
| Historical corrected weighted BERT, epoch10 | Validation1015 | 0.5811396777932097 | 0.6995114383470095 | 0.7054187192118226 | Saved predictions independently checked in prior evidence audit; checkpoint and executed full recipe unavailable. |
| Historical DeBERTa-v3-base | Validation1015 | 0.5466789591 | 0.6787051078 | 0.6679802956 | Prior saved-prediction verification; not final-test evidence or recovered inference pipeline. |
| Historical RoBERTa-base | Validation1015 | 0.5404692265 | 0.6722391144 | 0.6669950739 | Prior saved-prediction verification; thresholded checkpoint selection, not literal maximum. |
| Historical three-member probability ensemble | Validation1015 | 0.5888404445 | 0.7115885261 | 0.7133004926 | Combination reproduced from saved probabilities in prior audit; exploratory selection, no assigned final Test. |
| Historical five-member probability ensemble | Validation1015 | 0.5968259472 | 0.7173386920 | 0.7211822660 | Post-hoc exploratory combination; missing member checkpoints, no assigned final Test. |
| Historical TF-IDF balanced logistic regression | Validation1015 | 0.4809323035 | 0.5874682492 | 0.5655172414 | Prior local baseline audit; no new fit during release. |

The external unweighted original BERT comparison was report-only; no independent saved-prediction verification was established. Lineage augmentation, filtered historical augmentation, legacy25k grouping and GloVe baselines are separate development experiments. Their code inclusion does not assign them the final Mac checkpoint or its Test result. No missing metric is filled with a placeholder.

Historical0.9259 is unsupported as an independent leakage-free final Test measurement. The final Mac run uses no augmentation; all synthetic-label experiments remain distinct and do not prove automatic label correctness or augmentation benefit. See [historical scope](../docs/HISTORICAL_EXPERIMENTS.md).

The original Prototype passed63 mock/contract tests. Additional release safety tests remain mock/synthetic. Real BERT integration and human effectiveness were not verified.

Public evidence is aggregate-only [frozen_bert_summary.json](../results/frozen_bert_summary.json) and [actual training history](../results/mac_training_history.csv). Original model/records/evaluation controls/ZIPs are preserved privately. Redistribution rights, public checkpoint access and an explicit code license remain unresolved. Exact identities are recorded; absence of public records limits complete public training replay.
