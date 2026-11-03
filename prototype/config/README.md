# Prototype adapter configuration

`model_manifest.template.json` is deliberately incomplete and cannot activate
real inference. Replace its placeholders only with genuine owner-supplied model
artifacts and an accurately stated source run; an attestation is not independent
verification of training provenance.

`approved_primary_reference.json` is preserved unchanged as a **historical
validation-reference identity contract** for
`bert_sqrtweighted_max50_20261018_214144`. That historical checkpoint was not
recoverable. The file is not a replay result, a final Mac model manifest, a grant
to use the held-out Test, or evidence that real BERT integration worked. It does
not name or include a research record. Its source prediction artifact is excluded
from this public code release pending redistribution permission.

Do not reuse this historical contract to claim that the final Mac checkpoint
matches historical predictions. Final Mac integration would require its own
genuine checkpoint manifest and explicitly approved validation reference with
the correct class order, source run, preprocessing, precision, and device.
That integration verification has not been performed for this release.

No live authorization, successful replay report, checkpoint, or validation/Test
records are shipped here. The final Test evaluation has already occurred once
and must never be triggered by prototype setup or tests.
