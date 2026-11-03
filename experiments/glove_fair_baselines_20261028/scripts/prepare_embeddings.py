"""Download immutable official Stanford GloVe archive; extract only 100d vectors.

No project dataset is opened by this program. Range retries preserve partial bytes;
the complete archive must match the publisher repository's LFS SHA-256 before use.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = Path(__file__).resolve().parents[1]
REPO = "stanfordnlp/glove"
REVISION = "1db2080b2d94def6e5b0386a523102f9d8849e9d"
ARCHIVE_SIZE = 862182753
ARCHIVE_SHA = "6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837"
MEMBER = "glove.6B.100d.txt"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/glove.6B.zip"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    for name in ['embeddings', 'provenance']:
        directory = BASE / name
        if directory.is_symlink():
            raise ValueError('Generated directories may not be symlinks')
        directory.mkdir(exist_ok=True)
    archive = BASE / "embeddings/glove.6B.zip"
    partial = archive.with_suffix(".zip.partial")
    provenance = BASE / "provenance/glove_source.json"
    if not (BASE / "provenance/glove_hub_metadata.json").exists():
        r = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=60)
        r.raise_for_status()
        metadata = r.json()
        assert metadata["sha"] == REVISION
        item = next(x for x in metadata["siblings"] if x["rfilename"] == "glove.6B.zip")
        assert item["lfs"]["sha256"] == ARCHIVE_SHA and item["size"] == ARCHIVE_SIZE
        (BASE / "provenance/glove_hub_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not archive.exists():
        start_size = partial.stat().st_size if partial.exists() else 0
        assert shutil.disk_usage(BASE).free > ARCHIVE_SIZE - start_size + 900_000_000, "Insufficient free disk"
        chunk = 8 * 1024 * 1024
        last_print = 0
        def get_chunk(pos):
            end = min(ARCHIVE_SIZE - 1, pos + chunk - 1)
            for attempt in range(1, 6):
                try:
                    r = requests.get(URL, params={"download": "true", "range_start": str(pos), "attempt": str(attempt)},
                                     headers={"Range": f"bytes={pos}-{end}"}, timeout=(20, 60))
                    r.raise_for_status()
                    assert r.status_code == 206, f"Expected byte range; got {r.status_code}"
                    assert r.headers.get("Content-Range", "").startswith(f"bytes {pos}-{end}/"), r.headers.get("Content-Range")
                    assert len(r.content) == end - pos + 1
                    return pos, end, r.content
                except Exception as e:
                    print(f"Download retry {attempt} at byte {pos}: {type(e).__name__}: {e}", flush=True)
                    if attempt == 5:
                        raise
                    time.sleep(min(10, attempt*2))
        # Submit a bounded four-chunk window rather than buffering the entire archive.
        positions = iter(range(start_size, ARCHIVE_SIZE, chunk))
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = []
            for _ in range(4):
                pos = next(positions, None)
                if pos is not None:
                    pending.append(pool.submit(get_chunk, pos))
            while pending:
                pos, end, data = pending.pop(0).result()
                assert (partial.stat().st_size if partial.exists() else 0) == pos
                with partial.open("ab") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                if time.monotonic() - last_print > 15 or end == ARCHIVE_SIZE - 1:
                    print(f"Official GloVe download: {end + 1:,}/{ARCHIVE_SIZE:,} bytes", flush=True)
                    last_print = time.monotonic()
                pos = next(positions, None)
                if pos is not None:
                    pending.append(pool.submit(get_chunk, pos))
        assert partial.stat().st_size == ARCHIVE_SIZE
        assert sha(partial) == ARCHIVE_SHA, "Publisher SHA-256 mismatch; do not use embeddings"
        partial.rename(archive)
    assert archive.stat().st_size == ARCHIVE_SIZE and sha(archive) == ARCHIVE_SHA
    target = BASE / "embeddings" / MEMBER
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(MEMBER)
        assert info.file_size < 500_000_000
        if not target.exists():
            tmp = target.with_suffix(".txt.partial")
            with z.open(info) as source, tmp.open("wb") as out:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
            assert tmp.stat().st_size == info.file_size
            tmp.rename(target)
        record = {
            "status": "COMPLETE_ARCHIVE_PUBLISHER_SHA256_VERIFIED",
            "source": "Official Stanford NLP GloVe distribution, not newly trained embeddings",
            "project_page": "https://nlp.stanford.edu/projects/glove/",
            "official_distribution_documentation": "https://github.com/stanfordnlp/GloVe/blob/master/README.md",
            "repository": REPO, "revision": REVISION, "download_url": URL,
            "corpus": "Wikipedia 2014 + Gigaword 5; 6B tokens; 400,000 uncased vocabulary",
            "license_as_documented_by_publisher": "Public Domain Dedication and License v1.0",
            "embedding_dimension": 100, "archive_file": str(archive.relative_to(BASE)),
            "archive_bytes": ARCHIVE_SIZE, "archive_sha256": ARCHIVE_SHA,
            "member": MEMBER, "member_bytes": info.file_size, "member_crc32": f"{info.CRC:08x}",
            "member_sha256": sha(target),
            "other_dimensions_extracted": False,
        }
    if provenance.exists():
        earlier = json.loads(provenance.read_text())
        if earlier.get("archive_sha256_verified_at_member_recovery") is False:
            assert earlier["member_sha256"] == record["member_sha256"] and earlier["revision"] == REVISION
            # Preserve the provenance used for fitting/training. Completion is supplemental.
            completion = BASE / "provenance/glove_full_archive_verification.json"
            if completion.exists():
                assert json.loads(completion.read_text()) == record
            else:
                completion.write_text(json.dumps(record, indent=2) + "\n")
        else:
            assert earlier == record
    else:
        provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
