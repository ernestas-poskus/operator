#!/usr/bin/env python3
"""Materialize and verify the frozen document snapshot.

Every document is identified by SHA-256 in `snapshot.json`. This script clones
the two pinned corpora into `bench/doc-router/.cache/` and verifies every file,
downloading the binaries that their repositories do not store in git:

* `xberg-io/test_documents` at its pinned commit — text formats are committed,
  binaries are content-addressed in the public GCS bucket
  (`objects/<sha256>`), exactly as that repository's own fetch script uses it.
* `apache/tika` at its pinned commit — sparse checkout of the parser test
  resources.

Nothing is trusted by path: a file is accepted only when its bytes hash to the
snapshot's `sha256`. `--verify-only` never touches the network and is what CI
and the ignored experiment test call.

Usage:
  python3 bench/doc-router/fetch.py                 # materialize + verify
  python3 bench/doc-router/fetch.py --verify-only   # verify an existing cache
  python3 bench/doc-router/fetch.py --archive out.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "bench" / "doc-router"
SNAPSHOT = BENCH / "snapshot.json"
CACHE = BENCH / ".cache"
BUCKET = "https://storage.googleapis.com/xberg-test-documents/objects/"

TIKA_SPARSE = [
    "tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-microsoft-module/src/test/resources",
    "tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-miscoffice-module/src/test/resources",
    "tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-text-module/src/test/resources",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(command)}\n{result.stderr[-2000:]}")


def checkout(repo: str, commit: str, target: Path, sparse: list[str] | None) -> None:
    if target.exists():
        head = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        if head.returncode == 0 and head.stdout.strip() == commit:
            return
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if sparse:
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", repo, str(target)])
        run(["git", "-C", str(target), "sparse-checkout", "init", "--no-cone"])
        run(["git", "-C", str(target), "sparse-checkout", "set", *sparse])
        run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit])
        run(["git", "-C", str(target), "checkout", "--detach", commit])
    else:
        run(["git", "clone", "--depth", "1", "--no-checkout", repo, str(target)])
        run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit])
        run(["git", "-C", str(target), "checkout", "--detach", commit])


def download_bucket(sha: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{BUCKET}{sha}") as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def resolve(doc: dict, cache: Path) -> Path:
    """Map a snapshot document to its path inside the cache."""
    if doc["corpus"] == "xberg-corpus":
        return cache / "test_documents" / doc["rel"]
    return cache / "tika" / doc["rel"]


def materialize(docs: list[dict], cache: Path) -> list[str]:
    pins = json.loads(SNAPSHOT.read_text())["pins"]
    failures: list[str] = []
    checkout(
        "https://github.com/xberg-io/test_documents.git",
        pins["test_documents"],
        cache / "test_documents",
        None,
    )
    checkout("https://github.com/apache/tika.git", pins["tika"], cache / "tika", TIKA_SPARSE)
    for doc in docs:
        path = resolve(doc, cache)
        if not path.exists():
            try:
                download_bucket(doc["sha256"], path)
            except Exception as error:  # network or 404: report, never guess
                failures.append(f"{doc['id']}: cannot fetch {doc['sha256']}: {error}")
                continue
        if sha256(path) != doc["sha256"]:
            failures.append(f"{doc['id']}: hash mismatch at {path}")
    return failures


def verify(docs: list[dict], cache: Path) -> list[str]:
    failures = []
    for doc in docs:
        path = resolve(doc, cache)
        if not path.exists():
            failures.append(f"{doc['id']}: missing {path}")
        elif sha256(path) != doc["sha256"]:
            failures.append(f"{doc['id']}: hash mismatch at {path}")
    return failures


def archive(cache: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz") as tar:
        for name in ("test_documents", "tika"):
            tar.add(cache / name, arcname=name)
        tar.add(SNAPSHOT, arcname="snapshot.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--archive", type=Path, default=None)
    args = ap.parse_args()

    snapshot = json.loads(SNAPSHOT.read_text())
    docs = snapshot["documents"]
    cache: Path = args.cache
    cache.mkdir(parents=True, exist_ok=True)

    failures = verify(docs, cache) if args.verify_only else materialize(docs, cache)
    if failures:
        print(f"{len(failures)} snapshot failures:", file=sys.stderr)
        for failure in failures[:20]:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"verified {len(docs)} documents against {SNAPSHOT.name}")
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        archive(cache, args.archive)
        print(f"archived to {args.archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
