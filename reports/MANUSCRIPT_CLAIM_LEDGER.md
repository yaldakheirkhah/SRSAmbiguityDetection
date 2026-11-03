# Current manuscript claim ledger

Updated November 3, 2026. Current evidence supersedes historical draft status. This release does not edit manuscript files.

| Claim | Required wording / disposition |
|---|---|
| Final classification performance | Frozen Mac checkpoint: final Test Macro F1=0.549072, weighted F1=0.690482, accuracy=0.699507,1015 requirements, evaluated once under authorization. |
| Validation selection | Same Mac model selected at epoch41 by validation Macro F1=0.568594;49 epochs completed. Validation and Test are distinct. |
| Historical corrected BERT0.581140 | Different unavailable epoch10 checkpoint; saved validation evidence only. Do not assign its score to the final model or claim recovery. |
| Historical0.9259 | Do not present as independent, leakage-free held-out performance. Historical augmentation split suffered overlap. |
| Training objective | Training-only normalized square-root class weights; weighted-mean CE across the full accumulation window, denominator=sum target weights including final partial window. |
| Hardware equivalence | Mac FP32/MPS and NVIDIA BF16 recipe are distinct executions; no bitwise identity or completed NVIDIA outcome implied. |
| Independence | Retain punctuation-equivalent and unknown project/family dependence limitations. Do not claim exhaustive leakage freedom. |
| HITL integration | Mock/contract software verification only; no verified final BERT integration through UI. |
| Resolution, human effectiveness, RL, deployment | Unsupported; manual revision logging does not establish any of these effects. |
| Public reproducibility | Source/config/synthetic tests/aggregate evidence available in prepared release; records/checkpoints withheld and licenses/access unresolved. No promise of complete public artifact replay. |

See [precise manuscript edits](../docs/MANUSCRIPT_CONSISTENCY.md), [protocol](../docs/REPRODUCIBILITY.md), and [registry](EXPERIMENT_REGISTRY.md). Macro F1 weights classes equally; weighted F1 uses evaluation supports; neither is a loss weight. No statistical superiority, expert validation or state-of-the-art claim follows from these measurements.
