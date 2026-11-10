#!/usr/bin/env python3
"""Build `snapshot.json` from a benchmark checkout's corpus manifest.

The snapshot is the frozen input set for `bench/doc-router`: every document by
content hash, with its corpus provenance and the pins the verdict was measured
under. It is generated once (from the benchmark worktree that produced
`REPORT.md`) and then committed; re-running the experiment reads the snapshot,
not the live corpora.

Usage:
  python3 bench/doc-router/make_snapshot.py \
      --bench-root <benchmark-worktree>/bench/anydoc-vs-xberg \
      --bench-commit <sha of the benchmark harness state>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT = ROOT / "bench" / "doc-router" / "snapshot.json"

VENDOR_PREFIX = {
    "xberg-corpus": "vendor/test_documents/",
    "tika": "vendor/tika/",
}

# Pins of the run the routing verdict came from; the integrated versions are
# re-measured by the snapshot experiment before `golden/route.json` is frozen.
BENCH_PINS = {
    "anydoc": "0.2.4",
    "xberg": "1.3.10",
    "xberg_bench_commit": "a33f39373e9446aa109776aa4986a3c91c1efbeb",
    "test_documents": "385ddd5f11cddae0ef2b6146f800176206c0f0fd",
    "tika": "a40d54f6a6d9cf8ce485f10bb07011cd32ef5925",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cargo_lock_sha() -> str:
    lock = ROOT / "Cargo.lock"
    return sha256(lock) if lock.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", required=True)
    ap.add_argument("--bench-commit", default="")
    args = ap.parse_args()

    bench = Path(args.bench_root).resolve()
    manifest = json.loads((bench / "corpus" / "manifest.json").read_text())
    documents = manifest["documents"]

    rows = []
    missing = 0
    for doc in documents:
        path = Path(doc["path"])
        if not path.exists():
            missing += 1
            continue
        rel = ""
        for corpus, prefix in VENDOR_PREFIX.items():
            marker = f"/{prefix}"
            if corpus == doc["corpus"] and marker in doc["path"]:
                rel = doc["path"].split(marker, 1)[1]
                break
        if not rel:
            print(f"cannot derive a repository-relative path for {doc['path']}", file=sys.stderr)
            return 1
        digest = sha256(path)
        if digest != doc["sha256"]:
            print(f"hash mismatch for {doc['path']}", file=sys.stderr)
            return 1
        rows.append(
            {
                "id": doc["id"],
                "rel": rel,
                "ext": doc["ext"],
                "corpus": doc["corpus"],
                "source": doc["source"],
                "license": doc["license"],
                "size": doc["size"],
                "sha256": digest,
                "set": doc["set"],
            }
        )
    if missing:
        print(f"{missing} snapshotted documents are missing on disk", file=sys.stderr)
        return 1

    rustc = subprocess.run(["rustc", "-V"], capture_output=True, text=True, check=False).stdout.strip()
    snapshot = {
        "schema": "operator-doc-router-snapshot/1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "selection": {
            "seed": manifest.get("seed", 0),
            "max_bytes": manifest.get("max_bytes", 20_000_000),
            "bench_cap_pdf": 120,
            "bench_cap_default": 60,
            "note": "stratified per format across both corpora; oversized files are reliability-only",
        },
        "pins": {
            **BENCH_PINS,
            "bench_commit": args.bench_commit,
            "cargo_lock_sha256": cargo_lock_sha(),
            "rustc": rustc,
            "host": f"{platform.system()} {platform.machine()} {platform.release()}",
        },
        "documents": rows,
    }

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["corpus"], row["ext"])
        counts[key] = counts.get(key, 0) + 1

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot, indent=1) + "\n")
    print(f"wrote {SNAPSHOT} with {len(rows)} documents")
    for (corpus, ext), count in sorted(counts.items()):
        print(f"  {corpus:14s} {ext:5s} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
