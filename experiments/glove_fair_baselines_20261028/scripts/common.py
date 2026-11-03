"""Fixed preprocessing, PyTorch architectures and explicit six-class metrics."""

# Historical assertions are integrity controls; never disable them.
if not __debug__:
    raise RuntimeError("Optimized Python is prohibited for this research workflow")

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import random
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
CONFIG_PATH = BASE / "config/experiment.json"
PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
CLASSES = list(range(1, 7))


def sha(path):
    path = Path(path)
    if path.name.casefold() == "test.csv":
        raise ValueError("Held-out file is prohibited")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def config():
    return json.loads(CONFIG_PATH.read_text())


def load_frozen():
    cfg = config()
    result = []
    for role in ["train", "validation"]:
        p = ROOT / cfg["dataset"][role]
        assert p.resolve().parent == (ROOT / "data/splits").resolve()
        assert p.name == f"{role}.csv" and p.name in {"train.csv", "validation.csv"}
        assert sha(p) == cfg["dataset"][f"{role}_sha256"], f"Frozen {role} changed"
        d = pd.read_csv(p, dtype={"source_id": str}, keep_default_na=False)
        assert list(d.columns) == ["source_id", "Text", "Class"]
        assert len(d) == cfg["dataset"][f"{role}_rows"]
        assert d.source_id.is_unique and d.source_id.str.fullmatch(r"[0-9a-f]{16}").all()
        assert d.Text.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
        assert set(d.Class) == set(CLASSES)
        assert all(hashlib.sha256(norm(t).encode()).hexdigest()[:16] == sid for sid, t in zip(d.source_id, d.Text))
        result.append(d)
    assert not (set(result[0].source_id) & set(result[1].source_id))
    assert not (set(result[0].Text.map(norm)) & set(result[1].Text.map(norm)))
    return result


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


def punctkey(text):
    return re.sub(r"[\W_]+", "", norm(text))


def tokenize(text):
    # Apply no correction, contraction expansion, stemming or label-dependent rule.
    return PATTERN.findall(text.lower())


def build_vocabulary(train):
    counts = Counter(token for text in train.Text for token in tokenize(text))
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    assert not ({"<PAD>", "<UNK>"} & set(counts))
    vocabulary.update({word: i + 2 for i, word in enumerate(ordered)})
    return vocabulary, counts


def encode(texts, vocabulary, max_tokens=192):
    return [np.asarray([vocabulary.get(t, 1) for t in tokenize(text)[:max_tokens]] or [1], dtype=np.int64)
            for text in texts]


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights(labels):
    counts = np.asarray([(np.asarray(labels) == c).sum() for c in CLASSES], dtype=np.int64)
    raw = np.sqrt(len(labels) / (6.0 * counts))
    return counts, raw / raw.mean()


def metrics(true_ids, predicted_ids):
    """Zero-division=0 metrics derived from all 36 integer confusion counts.

    This implementation is independently checked against sklearn in verify_results.py.
    """
    y = np.asarray(true_ids, dtype=np.int64)
    p = np.asarray(predicted_ids, dtype=np.int64)
    assert len(y) == len(p) and len(y) > 0
    assert set(y) <= set(CLASSES) and set(p) <= set(CLASSES)
    cm = np.zeros((6, 6), dtype=np.int64)
    np.add.at(cm, (y - 1, p - 1), 1)
    tp = cm.diagonal().astype(np.float64)
    support = cm.sum(1)
    predicted_count = cm.sum(0)
    precision = np.divide(tp, predicted_count, out=np.zeros(6), where=predicted_count > 0)
    recall = np.divide(tp, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted_count, out=np.zeros(6), where=(support + predicted_count) > 0)
    result = {"accuracy": float(tp.sum() / len(y)), "class_order": CLASSES, "zero_division": 0,
              "confusion_matrix": cm.tolist()}
    for name, values in [("precision", precision), ("recall", recall), ("f1", f1)]:
        result[f"macro_{name}"] = float(values.mean())
        result[f"weighted_{name}"] = float(np.dot(values, support) / len(y))
    result["per_class"] = {str(c): {"precision": float(precision[c - 1]), "recall": float(recall[c - 1]),
                                      "f1": float(f1[c - 1]), "support": int(support[c - 1])} for c in CLASSES}
    return result


def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    ids = torch.zeros((len(batch), max(5, int(lengths.max()))), dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    return ids, lengths, torch.tensor(labels, dtype=torch.long)


def loader(seqs, labels, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(list(zip(seqs, np.asarray(labels, dtype=np.int64) - 1)),
        batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0,
        collate_fn=collate, generator=generator)


class GloveCNN(nn.Module):
    def __init__(self, matrix, settings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(matrix.copy()), freeze=False, padding_idx=0)
        self.embedding_dropout = nn.Dropout(settings["embedding_dropout"])
        self.kernels = settings["kernels"]
        self.convs = nn.ModuleList(nn.Conv1d(matrix.shape[1], settings["filters_per_kernel"], k) for k in self.kernels)
        self.dropout = nn.Dropout(settings["classifier_dropout"])
        self.classifier = nn.Linear(settings["filters_per_kernel"] * len(self.kernels), 6)

    def forward(self, ids, lengths):
        x = self.embedding_dropout(self.embedding(ids)).transpose(1, 2)
        pooled = []
        for k, conv in zip(self.kernels, self.convs):
            h = torch.relu(conv(x))
            # Do not let arbitrary additional batch padding affect global max.
            valid = (lengths - k + 1).clamp(min=1)
            positions = torch.arange(h.shape[-1], device=h.device)[None, :]
            mask = positions >= valid[:, None]
            pooled.append(h.masked_fill(mask[:, None, :], float("-inf")).amax(dim=2))
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class GloveLSTM(nn.Module):
    def __init__(self, matrix, settings):
        super().__init__()
        assert not settings["bidirectional"] and settings["layers"] == 1
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(matrix.copy()), freeze=False, padding_idx=0)
        self.embedding_dropout = nn.Dropout(settings["embedding_dropout"])
        self.lstm = nn.LSTM(matrix.shape[1], settings["hidden_size"], num_layers=1,
                            batch_first=True, bidirectional=False, dropout=0.0)
        self.dropout = nn.Dropout(settings["classifier_dropout"])
        self.classifier = nn.Linear(settings["hidden_size"], 6)

    def forward(self, ids, lengths):
        # Causal unidirectional states before padding are independent of padding.
        # Gathering at true length avoids relying on MPS packed-sequence support.
        outputs, _ = self.lstm(self.embedding_dropout(self.embedding(ids)))
        last = outputs[torch.arange(ids.shape[0], device=ids.device), lengths - 1]
        return self.classifier(self.dropout(last))


def make_model(architecture, matrix, cfg):
    return {"cnn": GloveCNN, "lstm": GloveLSTM}[architecture](matrix, cfg["architectures"][architecture])


def evaluate(model, dl, device):
    model.eval()
    all_logits = []
    with torch.inference_mode():
        for ids, lengths, _ in dl:
            all_logits.append(model(ids.to(device), lengths.to(device)).detach().cpu().numpy())
    logits = np.concatenate(all_logits).astype(np.float32, copy=False)
    assert logits.ndim == 2 and logits.shape[1] == 6 and np.isfinite(logits).all()
    p = probabilities(logits)
    return logits, p, p.argmax(1) + 1


def write_predictions(out, validation, logits, probs, predicted, cfg):
    d = validation.rename(columns={"Class": "true_label"}).copy()
    d["predicted_label"] = predicted
    names = cfg["labels"]["names"]
    d["true_name"] = [names[int(x) - 1] for x in d.true_label]
    d["predicted_name"] = [names[int(x) - 1] for x in predicted]
    for c in CLASSES:
        d[f"p_class_{c}"] = probs[:, c - 1]
    d["confidence"] = probs.max(1)
    d["correct"] = (d.true_label.to_numpy() == predicted)
    d.to_csv(out / "validation_predictions.csv", index=False, float_format="%.17g")
    np.save(out / "validation_logits_fp32.npy", logits, allow_pickle=False)
    np.save(out / "validation_probabilities_fp64.npy", probs, allow_pickle=False)
