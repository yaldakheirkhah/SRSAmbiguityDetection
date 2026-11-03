"""Inference and feedback collection are distinct operations; no training API."""
from .contracts import validate_text, snapshot


class ReviewService:
    def __init__(self, predictor, store):
        self.predictor, self.store = predictor, store

    def analyze(self, text, threshold=0.75, provenance=None):
        text = validate_text(text)
        raw = self.predictor.predict(text)
        prediction = snapshot(text, raw, threshold, provenance)
        duplicate_count = self.store.add_prediction(prediction)
        return prediction, duplicate_count

    def review(self, prediction_id, submission_id, **fields):
        return self.store.append_review(prediction_id, submission_id, **fields)
