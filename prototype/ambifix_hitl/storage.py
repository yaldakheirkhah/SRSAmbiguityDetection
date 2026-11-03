"""Local append-only prediction/review store; no fitting or model updates."""
from pathlib import Path
import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from .contracts import canonical, text_id, utc_now, ValidationError, validate_review

ZERO_HASH = "0" * 64


class AuditError(RuntimeError):
    pass


class Store:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix.casefold() not in {".db", ".sqlite", ".sqlite3"}:
            raise ValidationError("Use a local SQLite database filename.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO metadata VALUES('schema_version','1');
            CREATE TABLE IF NOT EXISTS requirements(
              requirement_id TEXT PRIMARY KEY, original_text TEXT NOT NULL, created_utc TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(
              prediction_id TEXT PRIMARY KEY, requirement_id TEXT NOT NULL REFERENCES requirements(requirement_id),
              payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS review_events(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
              submission_id TEXT NOT NULL UNIQUE, prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
              content TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL);
            """)
            schema=db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if schema is None or schema[0]!="1":
                raise AuditError("Unsupported audit database schema version.")
            for table in ["requirements", "predictions", "review_events", "metadata"]:
                for operation in ["UPDATE", "DELETE"]:
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} "
                               "BEGIN SELECT RAISE(ABORT,'append-only audit data'); END")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_prediction(self, prediction):
        # Serialized round trip prevents a caller mutating the persisted snapshot.
        value = json.loads(canonical(prediction))
        if value["requirement_id"] != text_id(value["original_text"]):
            raise ValidationError("Requirement identity does not match exact text.")
        payload = canonical(value)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            duplicate = db.execute("SELECT COUNT(*) FROM predictions WHERE requirement_id=?", (value["requirement_id"],)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO requirements VALUES(?,?,?)",
                       (value["requirement_id"], value["original_text"], value["timestamp_utc"]))
            existing = db.execute("SELECT original_text FROM requirements WHERE requirement_id=?", (value["requirement_id"],)).fetchone()[0]
            if existing != value["original_text"]:
                raise AuditError("Requirement text/hash identity conflict.")
            db.execute("INSERT INTO predictions VALUES(?,?,?,?)", (value["prediction_id"], value["requirement_id"], payload,
                        hashlib.sha256(payload.encode()).hexdigest()))
        return duplicate

    def prediction(self, prediction_id):
        with self.connect() as db:
            row = db.execute("SELECT payload,payload_sha256 FROM predictions WHERE prediction_id=?", (prediction_id,)).fetchone()
        if row is None:
            raise ValidationError("Unknown prediction ID.")
        if hashlib.sha256(row["payload"].encode()).hexdigest() != row["payload_sha256"]:
            raise AuditError("Prediction integrity check failed.")
        return json.loads(row["payload"])

    def append_review(self, prediction_id, submission_id, **review):
        if not isinstance(submission_id, str) or not 1 <= len(submission_id) <= 100:
            raise ValidationError("A bounded submission identifier is required.")
        prediction = self.prediction(prediction_id)
        validated = validate_review(prediction, **review)
        content = canonical({"prediction": prediction, **validated})
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT * FROM review_events WHERE submission_id=?", (submission_id,)).fetchone()
            if prior:
                if prior["content"] != content:
                    raise ValidationError("Submission already recorded with different content. Start another review.")
                return self._event(prior), True
            last = db.execute("SELECT event_hash FROM review_events ORDER BY sequence DESC LIMIT 1").fetchone()
            previous = last[0] if last else ZERO_HASH
            payload = {"schema_version": 1, "event_id": str(uuid.uuid4()), "submission_id": submission_id,
                       "prediction_id": prediction_id, "requirement_id": prediction["requirement_id"],
                       "timestamp_utc": utc_now(), **json.loads(content)}
            serialized = canonical(payload)
            event_hash = hashlib.sha256((previous + "\n" + serialized).encode()).hexdigest()
            db.execute("INSERT INTO review_events(event_id,submission_id,prediction_id,content,payload,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?)",
                       (payload["event_id"], submission_id, prediction_id, content, serialized, previous, event_hash))
            row = db.execute("SELECT * FROM review_events WHERE event_id=?", (payload["event_id"],)).fetchone()
            return self._event(row), False

    @staticmethod
    def _event(row):
        return {**json.loads(row["payload"]), "sequence": row["sequence"],
                "previous_hash": row["previous_hash"], "event_hash": row["event_hash"]}

    def events(self):
        with self.connect() as db:
            return [self._event(row) for row in db.execute("SELECT * FROM review_events ORDER BY sequence")]

    def predictions(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM predictions ORDER BY rowid")]

    def verify(self):
        with self.connect() as db:
            db.execute("BEGIN")  # One consistent snapshot while another session may append.
            preds = {}
            for row in db.execute("SELECT * FROM predictions"):
                p = json.loads(row["payload"])
                if hashlib.sha256(row["payload"].encode()).hexdigest() != row["payload_sha256"]:
                    raise AuditError("Prediction payload hash mismatch.")
                if row["prediction_id"] != p["prediction_id"] or row["requirement_id"] != text_id(p["original_text"]):
                    raise AuditError("Prediction identity mismatch.")
                req = db.execute("SELECT original_text FROM requirements WHERE requirement_id=?", (row["requirement_id"],)).fetchone()
                if req is None or req[0] != p["original_text"]:
                    raise AuditError("Original requirement differs from its prediction snapshot.")
                preds[row["prediction_id"]] = p
            previous = ZERO_HASH
            n = 0
            for row in db.execute("SELECT * FROM review_events ORDER BY sequence"):
                n += 1
                p = json.loads(row["payload"])
                expected = hashlib.sha256((previous + "\n" + row["payload"]).encode()).hexdigest()
                if row["sequence"] != n or row["previous_hash"] != previous or row["event_hash"] != expected:
                    raise AuditError("Review event chain mismatch.")
                if p["event_id"] != row["event_id"] or p["submission_id"] != row["submission_id"] or p["prediction_id"] != row["prediction_id"]:
                    raise AuditError("Event index identity mismatch.")
                if p["prediction"] != preds.get(p["prediction_id"]):
                    raise AuditError("Review changed the original model prediction.")
                body = {k:v for k,v in p.items() if k not in {"schema_version","event_id","submission_id","prediction_id","requirement_id","timestamp_utc"}}
                if canonical(body) != row["content"]:
                    raise AuditError("Review submission content mismatch.")
                previous = expected
        return {"status": "PASS", "prediction_count": len(preds), "event_count": n,
                "head_hash": previous, "scope": "local integrity, not authenticated reviewer identity"}
