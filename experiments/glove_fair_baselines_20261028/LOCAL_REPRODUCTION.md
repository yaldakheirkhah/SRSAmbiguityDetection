# GloVe baseline local prerequisites

These are the fixed historical source/configuration copies for the separate CNN
and LSTM validation baselines. No requirement records, embeddings or trained
checkpoints are included. Nothing is downloaded, fitted or evaluated by source
import, normal installation or the public default tests.

For deliberate local reproduction, obtain permission to use the original frozen
development partitions and restore them at repository `data/splits/train.csv`
and `data/splits/validation.csv`. Preserve their recorded configuration hashes.
Never substitute or read the original held-out Test for these commands.

With the specified training dependencies in a separate environment, the sequence
is `scripts/prepare_embeddings.py`, `scripts/prepare_data.py`, then
`scripts/train_baselines.py --smoke` and each deliberately selected
`--architecture cnn` / `--architecture lstm` run. The embedding preparation makes
an explicit network download of the pinned official GloVe archive and verifies
its publisher identity; it is optional, expensive, and not a setup command.
Release portability changes only create missing generated directories before
use. Model, fitting and checkpoint-selection definitions are unchanged.

The actual embedding preparation produces `provenance/glove_source.json`, which
data preparation requires. Vocabulary/weights/membership files and overlap reports
are generated locally and can disclose corpus content; do not publish them
without permission. Existing outputs are evidence and must not be overwritten.

`scripts/verify_results.py` also compares against historical TF-IDF predictions at
`reports/REVISION_AUDIT_TFIDF_VALIDATION_PREDICTIONS.csv` and weighted-BERT predictions
under `experiments/weighted_bert/runs/bert_sqrtweighted_max50_20261018_214144/`.
Those records are deliberately withheld. Thus its complete historical comparison
cannot pass from public source alone. `scripts/evaluate_validation.py` and
`scripts/verify_checkpoints.py` require the genuine completed local checkpoints.
Source availability is not independent reproduction of historical metrics.

Historical verification uses assertions. Do not run these historical scripts
with Python optimization (`-O`/`PYTHONOPTIMIZE`), which removes assertions.
Preserve actual environment/version reports for any new run and describe hardware
differences. No final Test evaluation is provided in this baseline workflow.
