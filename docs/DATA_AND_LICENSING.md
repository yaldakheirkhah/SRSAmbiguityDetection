# Data access and licensing

Release documentation updated November 3, 2026.

Dataset redistribution permission has not been confirmed. This release therefore excludes original and derived requirement records. Public access to a dataset does not establish permission to republish its contents, and this repository does not grant rights to third-party material.

## Source and acquisition

The experiments used the [Fault-Prone SRS dataset](https://www.kaggle.com/datasets/corpus4panwo/fault-prone-srs-dataset), associated with the [2023 source study](https://www.mdpi.com/2076-3417/13/14/8368). Users must obtain the applicable dataset version from its source and establish permission for their intended use before running data-dependent workflows. The source study's public-availability statement is an attribution source, not a substitute for the dataset's redistribution terms or the rights in underlying SRS documents.

The release review did not recover a versioned dataset license or written permission covering the actual source texts and their derived records. The dataset landing page did not expose readable license metadata through the review tool. This is an unresolved permission gap; it does not establish that no license exists.

## Excluded records

The following are excluded pending confirmed permission:

- Original source data, `final.csv`, and frozen train/validation/test records.
- Augmented or back-translated requirement records.
- Text-bearing lineage tables, conflict reports, error examples and predictions.
- Archives and notebook outputs embedding these records.
- Local review databases or feedback exports containing requirement text or reviewer information.

Code, configurations, aggregate counts and metrics, and reviewed synthetic test fixtures can document the procedure without redistributing these records. Aggregate evidence does not replace the unavailable text-level inputs required for full independent reproduction. Recorded hashes identify the original experimental inputs; they do not grant access or redistribution rights. Do not replace frozen data with another version while claiming to reproduce the same experiment.

## Existing public history

`final.csv` was already present in the repository's public history before this selective release. Removing it from the current tree does not purge historical commits or revoke copies already obtained. This release does not represent a completed removal from public history. No history rewrite, force-push or branch deletion is authorized as part of release preparation. Any historical-data remediation requires a separate decision and careful coordination with the repository owner.

## Code and third-party licenses

The previous description that the project was released for academic review did not specify an open-source code license. A code-license choice remains an author decision. This release does not invent an MIT grant or assert that an unspecified license permits unrestricted reuse. The presence of source code on GitHub should not be read as a new license grant.

Pretrained models, tokenizers, embeddings and software dependencies retain their own licenses. Those terms do not establish rights in the requirement dataset. Large model checkpoints and caches are excluded from this source release; code that obtains third-party assets must preserve their applicable notices and terms.

## Required before a data release

Record the exact dataset version, applicable license or rights-holder permission, coverage of underlying requirement texts, permission for derived redistribution, and required attribution or restrictions. Obtain the author's code-license decision separately. Until those matters are resolved, provide source links and reproducibility instructions without publishing requirement records.
