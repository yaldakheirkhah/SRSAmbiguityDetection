# AmbiFix human review prototype

This local Streamlit prototype demonstrates recording classifier output, human
class corrections, clarification questions, manually authored rewrites, and a
versioned audit export. Its automated tests use mock predictions and synthetic
requirements. They establish software behavior only. Real BERT integration with
the final frozen Mac checkpoint and human effectiveness have not been verified.

From this directory, create a separate Python environment and install
`requirements.txt`, then run `python -m pytest` or `python -m streamlit run app.py`.
The server binds to the local machine. The default backend is explicitly marked
**DEMO / MOCK MODE**. No weights, research data, private feedback database, live
authorizations, or Test evaluation are provided or launched by these commands.

The prototype is a local research demonstration without reviewer authentication.
Reviewer pseudonyms and user-entered requirement text are stored locally in
`runtime/`; review exports preserve that content. Do not publish this directory's
runtime database or user-generated exports. Human review does not automatically
change model weights or become an approved training dataset. A class correction
does not establish that a requirement's ambiguity has been resolved.

Real mode is an optional local adapter. It fails closed if the supplied complete
safetensors checkpoint, tokenizer, explicit mapping, provenance attestation, or
matching successful validation-reference replay report is missing or invalid.
It never silently falls back to mock, downloads a replacement model, or trains.
CPU and CUDA are the supported adapter devices; this prototype adapter does not
implement MPS. Source support for a device is not verified final-model integration.
See [config/README.md](config/README.md) before supplying any model artifacts.

`requirements_model.txt` records the historical optional adapter dependency
contract; it is not needed for mock tests. `requirements_local_lock.txt` records
the historical mock UI environment and is a separate dependency snapshot.
Do not combine both snapshots as though they were a tested single environment:
their NumPy pins differ. No new real-inference dependency compatibility claim is
made by this release. Use `requirements.txt` for the mock demonstration.

The released source is a curated copy of the October 30 prototype. Release fixes
replace assertion-dependent reference-gate and database-schema checks with
explicit errors that remain active under optimized Python. Synthetic regression
tests cover these controls. Original experiment files remain preserved separately.

The helper scripts prepare owner-supplied checkpoint manifests, perform explicit
approved **validation** replay, or export local feedback. They do not authorize
the held-out Test, infer training provenance from a model file, or establish a
human study. The manuscript's final frozen-model Test result is a distinct
experiment documented by the main repository release.
