"""Fit vocabulary on frozen original train only; audit validation without fitting."""
from collections import Counter, defaultdict
import csv
import json
import platform
import sys
import numpy as np
from common import BASE, CONFIG_PATH, ROOT, sha, write_json, config, load_frozen, tokenize, norm, punctkey, build_vocabulary, encode, class_weights


def main():
    cfg = config()
    train, val = load_frozen()
    for name in ['artifacts', 'reports']:
        directory = BASE / name
        if directory.is_symlink():
            raise ValueError('Generated directories may not be symlinks')
        directory.mkdir(exist_ok=True)
    out = BASE / "artifacts"
    assert not (out / "vocabulary.json").exists(), "Preparation is immutable; use existing artifacts or a new version"
    source = json.loads((BASE / "provenance/glove_source.json").read_text())
    glove = BASE / "embeddings" / cfg["embedding"]["member"]
    assert sha(glove) == source["member_sha256"]
    vocab, counts = build_vocabulary(train)
    write_json(out / "vocabulary.json", vocab, exclusive=True)
    write_json(out / "tokenizer.json", cfg["tokenizer"], exclusive=True)
    with (out / "training_token_counts.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "token_id", "training_occurrences"])
        writer.writerows((w, vocab[w], counts[w]) for w in sorted(counts, key=vocab.get))
    rng = np.random.default_rng(cfg["seed"])
    matrix = rng.normal(0, 0.05, (len(vocab), cfg["embedding"]["dimension"])).astype(np.float32)
    matrix[0] = 0
    found = set()
    total_lines = 0
    with glove.open("r", encoding="utf-8", errors="strict") as f:
        for line in f:
            total_lines += 1
            word, values = line.rstrip("\n").split(" ", 1)
            if word in vocab:
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                assert vector.shape == (100,) and np.isfinite(vector).all()
                assert word not in found
                matrix[vocab[word]] = vector
                found.add(word)
    assert total_lines == 400001 and len(found) > 0, "Pinned official distribution has 400K tokens + final <unk>; never substitute random matrix"
    np.save(out / "initial_embedding_matrix_fp32.npy", matrix, allow_pickle=False)
    write_json(out / "glove_matched_training_tokens.json", sorted(found), exclusive=True)
    per_class_counts, weights = class_weights(train.Class)
    write_json(out / "class_weights.json", {"fit_partition": "train only", "class_order": list(range(1, 7)),
        "counts": per_class_counts.tolist(), "formula": "sqrt(N/(6*n_c)) / mean", "weights": weights.tolist()}, exclusive=True)
    train[["source_id", "Class"]].to_csv(out / "training_membership.csv", index=False)
    val[["source_id", "Class"]].to_csv(out / "validation_membership.csv", index=False)
    duplicate_records = []
    audit = {}
    for role, d in [("train", train), ("validation", val)]:
        entry = {"rows": len(d), "class_counts": {str(c): int((d.Class == c).sum()) for c in range(1, 7)}}
        tokens = [tokenize(t) for t in d.Text]
        lengths = np.asarray([len(t) for t in tokens])
        entry.update({"token_length_percentiles": {str(p): float(np.percentile(lengths, p)) for p in [0,50,90,95,99,100]},
            "rows_truncated_at_192": int((lengths > 192).sum()), "empty_tokenizations": int((lengths == 0).sum()),
            "vocabulary_oov_token_occurrences": sum(w not in vocab for row in tokens for w in row),
            "token_occurrences": int(lengths.sum()),
            "glove_missing_training_vocabulary_types": len(counts) - len(found) if role == "train" else None,
            "rows_with_C1_controls": int(d.Text.map(lambda t: any(0x80 <= ord(c) <= 0x9f for c in t)).sum())})
        for kind, keys in [("exact", d.Text), ("normalized", d.Text.map(norm)), ("punctuation_equivalent", d.Text.map(punctkey))]:
            groups = defaultdict(list)
            for i, key in enumerate(keys):
                groups[key].append(i)
            multi = [indices for indices in groups.values() if len(indices) > 1]
            entry[kind] = {"duplicate_keys": len(multi), "extra_rows": sum(len(x)-1 for x in multi),
                           "cross_label_keys": sum(len(set(d.iloc[x].Class)) > 1 for x in multi)}
        audit[role] = entry
    for kind, transform in [("exact", lambda x: x), ("normalized", norm), ("punctuation_equivalent", punctkey),
                             ("token_sequence", lambda t: tuple(tokenize(t))),
                             ("encoded_truncated_sequence", lambda t: tuple(encode([t], vocab, 192)[0]))]:
        keys = defaultdict(list)
        for row in train.itertuples(index=False):
            keys[transform(row.Text)].append(row)
        matches = []
        for row in val.itertuples(index=False):
            for parent in keys.get(transform(row.Text), []):
                matches.append({"kind": kind, "train_source_id": parent.source_id, "validation_source_id": row.source_id,
                                "train_label": int(parent.Class), "validation_label": int(row.Class),
                                "train_Text": parent.Text, "validation_Text": row.Text})
        audit[f"cross_partition_{kind}"] = {"pairs": len(matches),
            "validation_rows": len({x["validation_source_id"] for x in matches}),
            "cross_label_pairs": sum(x["train_label"] != x["validation_label"] for x in matches)}
        duplicate_records.extend(matches)
    import pandas as pd
    pd.DataFrame(duplicate_records).to_csv(BASE / "reports/residual_overlap_pairs.csv", index=False)
    files = ["vocabulary.json", "tokenizer.json", "training_token_counts.csv", "initial_embedding_matrix_fp32.npy",
             "glove_matched_training_tokens.json", "class_weights.json", "training_membership.csv", "validation_membership.csv"]
    manifest = {"status": "TRAIN_ONLY_FITTING_COMPLETE", "configuration_sha256": sha(CONFIG_PATH),
        "train_sha256": cfg["dataset"]["train_sha256"], "validation_sha256": cfg["dataset"]["validation_sha256"],
        "training_rows": len(train), "validation_rows": len(val), "training_vocabulary_types_including_specials": len(vocab),
        "glove_lines_scanned": total_lines, "glove_matched_training_types": len(found),
        "glove_unmatched_training_types": len(counts)-len(found),
        "glove_token_coverage_train": sum(n for w,n in counts.items() if w in found) / sum(counts.values()),
        "glove_source_sha256": sha(BASE / "provenance/glove_source.json"),
        "files": {name: sha(out/name) for name in files}, "audit": audit,
        "fitting_rule": "Vocabulary/counts, OOV initialization, and class weights derived only from train. Validation transformed for diagnostics/evaluation; never used to expand vocabulary or embeddings.",
        "limitations": "Fixed inherited punctuation/near-duplicate risks; no common-parent or fully independent provenance claim. Original texts, labels, splits unchanged."}
    write_json(out / "preparation_manifest.json", manifest, exclusive=True)
    for name in files + ["preparation_manifest.json"]:
        (out/name).chmod(0o444)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
