# Manuscript consistency and remaining author edits

Release documentation updated November 3, 2026.

The completed classifier results can be reported accurately, but the known V6 manuscript copy is a pre-final draft. Preparing this repository does not establish that the manuscript has been updated. No manuscript document was modified during release preparation.

The reviewed copy is `AmbiFix_V6_Section3_2_Clarified.docx`. References below use direct body-paragraph numbers in `word/document.xml`, excluding table-cell paragraphs, from the prior read-only consistency audit. They identify that specific copy and must be rechecked if the authors use a later manuscript. They are content references, not a claim of current visual-layout verification.

## Result values to integrate

| Result | Validation | Final held-out test |
|---|---:|---:|
| Macro F1 | 0.5685944251323821 | 0.5490718700859288 |
| Weighted F1 | 0.6955474743459408 | 0.6904821963220387 |
| Accuracy | 0.7024630541871921 | 0.6995073891625616 |

These values belong to the new original-only Mac MPS experiment, selected at epoch 41 using fixed validation Macro F1. The test evaluation was executed once on the frozen checkpoint on November 2, 2026. Its metrics were independently recalculated from saved outputs. The historical corrected BERT validation Macro F1 of 0.5811396777932097 belongs to a separate epoch-10 model whose weights were unrecoverable. It must not be assigned to the new model.

## Required corrections in the reviewed copy

| Manuscript location | Current issue | Required update |
|---|---|---|
| P1 internal-review banner | GPU checkpoint and held-out test marked pending. | Identify the completed separate Mac run and final once-only test status. |
| Abstract P13, experimental setup P109, discussion P158 and conclusion P173 | Historical validation 0.5811 remains the main reference; final Mac results are absent. | Distinguish historical saved-prediction evidence, Mac validation and final held-out performance. |
| Section 3.2 P84–85; Section 4 P111–114 | Refers to pending new GPU retraining; completed GPU settings describe augmented Variant A. | Use the actual Mac FP32/MPS settings and epoch-41 checkpoint identity. Keep Variant A BF16/GPU settings attached to that experiment. |
| Tables 4 and 6 | Contain historical BERT and Variant A comparisons, without final Mac validation/test evidence. | Add clearly labelled Mac validation results and a separate final held-out table; preserve comparison limitations. |
| P114, P168 and P173 | Say the final checkpoint, retraining or test result is pending. | Replace outdated status with the completed frozen-model outcome. Do not imply another evaluation is planned. |
| P67, P107 and threats P164–168 | Existing dependence/taxonomy caveats omit final post-evaluation diagnostics. | Report residual punctuation-equivalent relations and limitations on project/semantic independence. Do not alter the scored population. |
| P91, P151–155, P170 and P173–174 | Correctly separates mock tests from real inference and human evaluation. | Preserve these restrictions after adding classifier results. |
| P179 data/code availability | Release accessibility and contents remain conditional. | Describe the actual selective release, excluded requirement records, source acquisition and unresolved licensing accurately. Do not promise public checkpoints or datasets that are not supplied. |

The original README's 0.9259 F1 and reinforcement-learning claims also conflict with the corrected evidence. Historical copies may be retained with explicit archival labels, but must not serve as the release's current result statement or recommended run-all workflow.

## Preserve the confirmed loss and labels

Section 3.2 P78–83 already describes the confirmed target-weight-normalized loss. For one effective optimizer accumulation window B,

```text
L(B) = - sum(i in B, w[y_i] * log p(y_i | x_i))
         / sum(i in B, w[y_i])
```

Both sums cover every valid labelled training example in the complete window, including its final partial window. The denominator is the sum of corresponding class weights, not the number of examples. Weighted numerators are accumulated across microbatches; accumulated gradients are divided once by the full window's weight sum before clipping and updating. Separately averaged microbatch means would generally define a different objective. The new run's nominal effective batch is 16 from two microbatches of eight; its final window has 11 examples.

Class weights are square roots of inverse training frequency, computed from training labels alone and rescaled to mean one. Preserve the [PyTorch 2.9 CrossEntropyLoss reference](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html). This explicit implementation contract describes the new run, not an independently recovered historical implementation.

The fixed class order remains Lexical, Syntactic, Semantic, Syntax, Pragmatic, Clean. Dataset class numbers 1–6 correspond to model output indices 0–5. Training used original Text; diagnostic normalization keys must not be described as silently altering model inputs. Syntactic and Syntax remain distinct labels with a construct-validity limitation.

## Required qualifications

The final test Macro F1 is lower than validation by 0.0195225550, about 1.95 percentage points. Report this without a causal explanation, significance claim or subsequent score-driven changes. Macro F1 equally averages six class F1 values; Weighted F1 uses evaluation supports rather than training-loss weights.

Seven punctuation-equivalent train/test pairs involving seven test rows, one cross-label pair, and one validation/test pair remain. These are uncertain dependence/annotation relationships, not proof of common parentage. No records were removed or relabelled and no filtered test metric was produced. Source-document, project and semantic-family independence are not established.

The prototype's 63 mock/synthetic functional tests do not verify real BERT inference in the interface, stakeholder-approved rewriting, expert label validation, human effectiveness or industrial deployment. Preserve the absence of automatic resolution, reinforcement learning and online adaptation. Single-run classification results do not establish state-of-the-art performance, statistical significance or multi-seed robustness.

## Final author check

Before submission, verify the actual submitted document against the completed final aggregate evidence, configuration and figure sources. Check every result's partition label and class order, preserve historical versus final experiment identities, update all pending statements, and make data/code availability agree with [the release restrictions](DATA_AND_LICENSING.md). This checklist does not authorize retraining, new test inference, changes to the frozen method or automatic manuscript edits.
