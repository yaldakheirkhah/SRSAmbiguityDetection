# Source release audit

Prepared November 3, 2026. This release was built from the existing remote main tip in an isolated branch; the local historical audit branch and all experimental originals were preserved. It does not repeat training, inference or final evaluation.

## Included evidence and controls

- Original-only Mac/NVIDIA training implementations, immutable configurations and data-identity manifests; separate augmentation/grouping/GloVe methods; Prototype source, synthetic tests; guarded split/audit/TF-IDF tools; evaluation provenance and output-verification code; aggregate plotting and reproducibility documentation.
- Public validation/Test metrics and confusion counts transcribed from recorded aggregate exports, with source hashes. Actual 49-epoch history is retained unchanged. Aggregate arithmetic confirms validation Macro F1 0.5685944251323821 and final Test Macro F1 0.5490718700859288, weighted F1 0.6904821963220387, accuracy 0.6995073891625616. Earliest validation maximum is epoch41;14,504 optimizer updates are recorded.
- Executed final Mac training, preflight, verification, plotting and final-evaluation source/configuration are byte-identical to the canonical source. [CODE_PROVENANCE.json](CODE_PROVENANCE.json) identifies exact source/release hashes and disclosed adaptations in other copies.
- Release data-tool imports are inert, explicit execution/new output paths are required, frozen output directories and overwrite are refused. Historical data records are not used in software tests.
- Prototype reference/database checks now use explicit errors under optimized Python. Legacy assertion-based workflows explicitly prohibit optimized execution. Broad machine-specific transport/package builders are omitted. Local manifest helpers only record identities after real quality gates; they do not manufacture passing evidence or package records.
- Root default pytest discovery is limited. Full curated suites run separately to isolate identically named experiment modules and translation environment settings. No default setup, CI or software test performs model loading, downloads, training, or final Test evaluation. Actual evaluation approval receipts and consumed locks remain private and intact; no reusable authorization is published.

## Executed verification

The final curated runner passed **225 tests;2 private-development-data tests skipped**: root49, Prototype82, NVIDIA22+1skip, Mac59+1skip, lineage13. Default discovery passed131 tests. Root/data checks use invented records and model spies; Prototype tests use mock/synthetic predictions. Missing-artifact tests mock private input verification before testing incomplete exports. The real-data checks must be performed explicitly only after authorized development-file access.

Research checks used existing Python3.12.2 with NumPy1.26.4/pandas2.2.3/scikit-learn1.6.1; pytest8.4.2 was provided from an existing environment after research package paths. Prototype tests ran in their separate existing Streamlit1.50.0/NumPy2.5.4/pandas2.3.3/Arrow26 environment. Default discovery also passed with the existing UI environment plus scikit-learn1.9.1; these are software checks, not new experiment metrics. Existing training environments were not modified.

A combined local environment exposed Arrow26 requiring NumPy2 while the research pins require NumPy1.26.4. The optional new clean-install attempt was cancelled because downloads were slow; it did not complete. `requirements-test.txt` explicitly pins Arrow19.0.1, whose [official documentation](https://arrow.apache.org/docs/19.0/python/install.html#dependencies) supports NumPy1.16.6+ and Python3.12. **A fresh installation of that combined dependency set and remote CI have not been executed.** Existing compatible environments were used separately through `--prototype-python`; no clean-machine verification is claimed. NVIDIA/real Prototype inference environments were not newly tested.

Seven figure types were regenerated from aggregate counts and actual unsmoothed history in PNG/PDF/SVG, without source data or inference. Generated files remain local and excluded from Git; publication code and source aggregates are included. Metric checks passed and exact class order is enforced. Source/notebook syntax, no executed notebook outputs, selected-file size and credential/personal-path scans were checked. Scans found no credential or personal-path matches after manual review of one matrix-multiplication expression that resembles an email; scanning is not a guarantee about all historical Git objects.

## Explicit exclusions and remaining concerns

No source/frozen/augmented requirement records, per-record predictions/lineage, model/checkpoint/cache/embedding payloads, environments, feedback databases, approval controls, evidence archives, personal-path manifests, manuscript files or temporary generated figures are included. Removing tracked `final.csv` and the old unsupported notebook from the new tip preserves their local files and existing Git history.

Dataset redistribution permission remains unconfirmed, and a formal code-license choice is outstanding. Historical public Git objects already contain `final.csv`; this release does not purge that history or claim the whole repository history is rights-cleared. No new text-bearing audit commit is included from the local historical branch. Publication/access decisions require the author's review.

The known manuscript draft remains stale in the [specific locations listed](MANUSCRIPT_CONSISTENCY.md); it was not edited. Real BERT integration through the Prototype, human effectiveness, automatic resolution, RL and industrial effectiveness remain unverified. Residual punctuation-equivalent relationships and unknown project/family independence remain scientifically material limits. The public source release cannot provide full private-artifact training replay while data/model access is unresolved.

Local commit preparation is authorized; actual push requires the author's separate approval. No push, force-push, history rewrite, remote deletion or repeated final evaluation is part of this audit.
