"""Explicit, shared contracts; dataset IDs are 1..6, model indices 0..5."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid

LABELS = {1: "Lexical Ambiguity", 2: "Syntactic Ambiguity", 3: "Semantic Ambiguity",
          4: "Syntax Ambiguity", 5: "Pragmatic Ambiguity", 6: "Clean"}
CLASS_ORDER = tuple(LABELS)
STATUSES = ("draft", "needs_stakeholder_clarification", "accepted_by_reviewer")
DECISIONS = ("accept", "correct", "needs_context")
MAX_TEXT = 20_000


class ValidationError(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def text_id(text):
    return hashlib.sha256(validate_text(text).encode("utf-8")).hexdigest()


def validate_text(value, maximum=MAX_TEXT, optional=False):
    if optional and (value is None or value == ""):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Enter nonempty text.")
    if len(value) > maximum:
        raise ValidationError(f"Text exceeds the {maximum:,}-character limit.")
    if "\x00" in value:
        raise ValidationError("Null bytes are not supported.")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValidationError("Text contains malformed Unicode.") from exc
    return value  # Preserve the exact original, including leading/trailing whitespace.


def valid_label(value, optional=False):
    if optional and value is None:
        return None
    if type(value) is not int or value not in LABELS:
        raise ValidationError("Class must be an integer dataset ID from 1 through 6.")
    return value


def valid_probabilities(values, sum_tolerance=1e-7):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValidationError("Exactly six class probabilities are required.")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValidationError("Class probabilities must be finite numbers in [0,1].")
    if not math.isclose(sum(values), 1.0, abs_tol=sum_tolerance):
        raise ValidationError("Class probabilities must sum to one.")
    return tuple(float(v) for v in values)


def uncertainty(values, threshold):
    p = valid_probabilities(values)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValidationError("Review threshold must be in [0,1].")
    entropy = -sum(v * math.log(v) for v in p if v > 0) / math.log(6)
    entropy = min(1.0, max(0.0, entropy))
    return {"measure": "normalized_predictive_entropy_v1", "value": entropy,
            "threshold": float(threshold), "prioritized": entropy >= threshold,
            "calibrated": False}


@dataclass(frozen=True)
class RawPrediction:
    probabilities: tuple
    model_id: str
    checkpoint_version: str
    is_mock: bool
    metadata: dict

    def __post_init__(self):
        valid_probabilities(self.probabilities)
        validate_text(self.model_id, 200)
        validate_text(self.checkpoint_version, 300)
        if type(self.is_mock) is not bool:
            raise ValidationError("Prediction mode must be explicit.")
        canonical(self.metadata)


def snapshot(text, raw, threshold, provenance=None):
    validate_text(text)
    p = valid_probabilities(raw.probabilities)
    selected = max(range(6), key=lambda i: p[i]) + 1
    if provenance is None:
        provenance = {"dataset": "interactive_input", "partition": "unknown", "source_id": None}
    if not isinstance(provenance, dict):
        raise ValidationError("Dataset provenance must be a JSON object when known.")
    canonical(provenance)
    return {"schema_version": 1, "prediction_id": str(uuid.uuid4()), "requirement_id": text_id(text),
            "original_text": text, "model_id": raw.model_id, "checkpoint_version": raw.checkpoint_version,
            "is_mock": raw.is_mock, "mode_label": "DEMO / MOCK MODE" if raw.is_mock else "REAL BERT / REFERENCE REPLAY VERIFIED",
            "predicted_label": selected, "predicted_name": LABELS[selected], "class_order": list(CLASS_ORDER),
            "probabilities": list(p), "confidence": p[selected-1], "uncertainty": uncertainty(p, threshold),
            "model_metadata": raw.metadata, "dataset_provenance": provenance, "timestamp_utc": utc_now()}


def validate_review(prediction, decision, selected_label, reviewer_id, review_status,
                    explanation=None, clarification_question=None, rewritten_text=None, rewrite_status=None):
    if decision not in DECISIONS or review_status not in STATUSES:
        raise ValidationError("Unsupported decision or review status.")
    reviewer_id = validate_text(reviewer_id, 80)
    if decision == "accept":
        if valid_label(selected_label) != prediction["predicted_label"]:
            raise ValidationError("Accept must retain the predicted class; use Correct to change it.")
    elif decision == "correct":
        valid_label(selected_label)
        if selected_label == prediction["predicted_label"]:
            raise ValidationError("The selected class equals the prediction; use Accept.")
    else:
        if selected_label is not None:
            raise ValidationError("Needs-context reviews must leave the classification undecided.")
        if review_status == "accepted_by_reviewer":
            raise ValidationError("An undecided classification cannot be marked accepted.")
    explanation = validate_text(explanation, 5000, optional=True)
    clarification_question = validate_text(clarification_question, 5000, optional=True)
    rewritten_text = validate_text(rewritten_text, MAX_TEXT, optional=True)
    if rewritten_text is None and rewrite_status is not None:
        raise ValidationError("A rewrite status requires manually authored rewritten text.")
    if rewritten_text is not None and rewrite_status not in STATUSES:
        raise ValidationError("Select an explicit status for the manual rewrite.")
    return {"decision_type": decision, "selected_label": selected_label,
            "selected_name": LABELS.get(selected_label), "reviewer_pseudonym": reviewer_id,
            "review_status": review_status, "reviewer_explanation": explanation,
            "clarification_question": clarification_question, "rewritten_text": rewritten_text,
            "rewrite_status": rewrite_status, "rewrite_authorship": "human" if rewritten_text is not None else None,
            "semantic_equivalence_verified": False, "automatically_resolved": False,
            "approved_for_training": False}
