# Historical experiments and result provenance

Release documentation updated November 3, 2026.

The final frozen classifier is a new original-only Mac M3 Pro BERT training run. It is distinct from the historical corrected BERT checkpoint, the augmentation experiments and exploratory ensembles. Historical scores must retain their dataset, selection and artifact limitations.

## Primary result identities

| Experiment and evaluated partition | Macro F1 | Weighted F1 | Accuracy | Evidence boundary |
|---|---:|---:|---:|---|
| Historical corrected BERT, fixed validation, selected epoch 10 | 0.5811396777932097 | 0.6995114383470095 | 0.7054187192118226 | Saved validation predictions independently checked; the original selected weights and complete executed training configuration could not be recovered. |
| New original-only Mac BERT, fixed validation, selected epoch 41 | 0.5685944251323821 | 0.6955474743459408 | 0.7024630541871921 | Complete new training run; checkpoint selected by validation Macro F1 and independently reloaded for validation verification. |
| Same frozen Mac checkpoint, one final held-out test evaluation | **0.5490718700859288** | **0.6904821963220387** | **0.6995073891625616** | Final evaluation completed November 2, 2026; metrics independently recomputed from its saved exports without repeating model inference. |

Historical run identity: `bert_sqrtweighted_max50_20261018_214144`. Final training experiment identity: `bert_mac_m3pro_20261101`. The historical 0.581140 validation result is a reference, not the final model's validation or test result and not a guaranteed retraining target.

The final Mac experiment trained on 4,731 original examples and selected its checkpoint using the unchanged 1,015-example validation partition. Its held-out test evaluation covered 1,015 examples exactly once. The final model was not retrained, retuned or replaced after those test outcomes. Test Macro F1 was lower than validation by 0.0195225550, about 1.95 percentage points.

## Legacy augmented corpus

The historically augmented corpus contained 24,949 rows and was split after augmentation, with train/evaluation overlap. The historical 0.9259 F1 claim cannot be treated as independent, leakage-free held-out performance. It must not appear as the final result of the corrected frozen benchmark. The historical notebook is an archival artifact; running it is not the standard reproducibility workflow for this release.

Later training-only augmentation experiments tracked parentage and review flags more explicitly. They remain separate experiments with provisional inherited synthetic labels. Automatically accepted or unflagged synthetic text is not expert-verified annotation. The GPU Variant A run selected epoch 11; it is not the historical original-only epoch-10 model or the final Mac epoch-41 model. Its settings and scores must not be silently substituted for either.

## Other baselines and exploratory ensembles

Archived evidence checks covered saved validation probabilities for DeBERTa-v3-base and RoBERTa-base and arithmetic combinations of saved model probabilities. The October 26 registry recorded validation Macro F1 values of 0.5466789591 for DeBERTa, 0.5404692265 for RoBERTa, 0.5888404445 for a three-model equal-probability ensemble and 0.5968259472 for a five-model exploratory ensemble. Reproducing an ensemble's arithmetic does not reproduce its member training or establish that complete deployable checkpoints are available.

The locally rerun TF-IDF/logistic-regression reference had validation Macro F1 0.4809323035. These are development-validation observations. They are not final held-out results, statistical significance estimates, multi-seed robustness evidence or evidence of test superiority. Model and ensemble comparisons made using the same development set are exploratory and must not be described as an independent final evaluation of every candidate.

The historical experiment registry dated October 26 predates the completed Mac run and contains then-correct pending statements. Such snapshots describe their preparation date. The later final completion and consumed evaluation marker establish the final status; immutable historical records should not be rewritten to erase their chronology.

## Metrics and interpretation

Macro F1 is the arithmetic mean of all six class-specific F1 values. Weighted F1 uses class supports in the evaluated partition; those weights are different from the square-root class weights used in training loss. Accuracy is the proportion of correct predictions. A high support-weighted result must not be labelled Macro F1.

The fixed class order is Lexical, Syntactic, Semantic, Syntax, Pragmatic, Clean. Syntactic and Syntax are retained as distinct dataset categories despite a documented taxonomy limitation. No category was automatically merged or relabelled.

Residual punctuation-equivalent relations remain: seven train/test pairs involving seven test rows, including one cross-label pair, and one validation/test pair. These heuristic relationships do not prove common parentage, but prevent a claim of fully established semantic, project or family independence. No rows were removed and no alternate filtered test score was calculated.

## Prototype evidence is separate

The Streamlit human-review prototype passed 63 automated tests using synthetic inputs and deterministic mock predictions. These tests establish specified software behaviors, not real BERT integration, a human-participant study or improved requirement-review effectiveness. The later existence of a complete classifier checkpoint does not verify its end-to-end use inside the prototype.

The release makes no claim of automatic ambiguity resolution, reinforcement learning, industrial deployment, human effectiveness, state-of-the-art performance or causal augmentation benefit. Final-test inference is consumed; setup, ordinary tests, plotting and evidence checks must not invoke another evaluation.
