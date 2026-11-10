#!/usr/bin/env python3
"""Run the document-engine experiment over the frozen snapshot.

Materializes the snapshot cache (via `fetch.py` if needed), then drives
`examples/docbench` over the timed documents three times — anydoc alone, xberg
alone, and the shipped router — writing raw JSONL and Markdown outputs under
`bench/doc-router/results/` (gitignored) for `score.py`.

Usage:
  python3 bench/doc-router/run.py [--reps 3] [--warmup 1] [--release]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "bench" / "doc-router"
SNAPSHOT = BENCH / "snapshot.json"
CACHE = BENCH / ".cache"
RESULTS = BENCH / "results"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_path(doc: dict) -> Path:
    root = "test_documents" if doc["corpus"] == "xberg-corpus" else "tika"
    return CACHE / root / doc["rel"]


def bench_manifest() -> Path:
    snapshot = json.loads(SNAPSHOT.read_text())
    documents = [doc for doc in snapshot["documents"] if doc["set"] == "bench"]
    missing = [doc["id"] for doc in documents if not cached_path(doc).exists()]
    if missing:
        subprocess.run([sys.executable, str(BENCH / "fetch.py")], check=True)
        missing = [doc["id"] for doc in documents if not cached_path(doc).exists()]
        if missing:
            raise SystemExit(f"{len(missing)} snapshot documents are still missing")

    drifted = [
        doc["id"]
        for doc in documents
        if sha256(cached_path(doc)) != doc["sha256"]
    ]
    if drifted:
        raise SystemExit(f"{len(drifted)} snapshot documents drifted from their hash")

    manifest = CACHE / "manifest.tsv"
    manifest.write_text(
        "".join(f"{doc['id']}\t{cached_path(doc)}\t{doc['ext']}\n" for doc in documents)
    )
    return manifest


def run_docbench(manifest: Path, engine: str, fallback: str, reps: int, warmup: int, release: bool) -> None:
    outdir = RESULTS / "outputs" / engine
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        "cargo",
        "run",
        "--example",
        "docbench",
    ]
    override = BENCH / "cargo-override.toml"
    if override.exists():
        command += ["--config", str(override)]
    if release:
        command.append("--release")
    command += [
        "--",
        "--manifest", str(manifest),
        "--engine", engine,
        "--fallback", fallback,
        "--reps", str(reps),
        "--warmup", str(warmup),
        "--jsonl", str(RESULTS / f"{engine}.jsonl"),
        "--out", str(outdir),
    ]
    print(f"running {engine}/{fallback} ...", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-release", action="store_false", dest="release", help="debug build (quality only)")
    ap.add_argument("--only", default="", help="comma-separated engines (anydoc,xberg,auto)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = bench_manifest()
    print(f"timed documents: {sum(1 for _ in manifest.open())}")

    engines = [e.strip() for e in args.only.split(",") if e.strip()] or ["anydoc", "xberg", "auto"]
    policies = {"anydoc": "strict", "xberg": "strict", "auto": "best-effort"}
    for engine in engines:
        run_docbench(manifest, engine, policies[engine], args.reps, args.warmup, args.release)
    print(f"raw results under {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
