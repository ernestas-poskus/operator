#!/usr/bin/env python3
"""Derive (or check) the routed engine table from the measured snapshot run.

The rule is the one documented in `src/document.rs` and `golden/route.json`:

    xberg becomes primary for a format when
      mean paired token-F1(xberg) - token-F1(anydoc) >= +0.03
      and xberg wins at least twice as many documents as it loses (|delta| > 0.01)
      and median latency(xberg) <= 1.5 x median latency(anydoc)

Every other format stays on anydoc. Extensions that share a parser with a
measured format inherit its verdict (`docm` -> `docx`, `pps`/`pptm`/`ppsx`/
`ppsm`/`pot` -> `pptx`, `xlsm` -> `xlsx`, `xlsb`/`dot` -> their measured base).

Usage:
  python3 bench/doc-router/derive.py            # print the derived table
  python3 bench/doc-router/derive.py --update   # write golden/
  python3 bench/doc-router/derive.py --check    # fail on drift (CI/manual)
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "bench" / "doc-router"
RESULTS = BENCH / "results"
GOLDEN = BENCH / "golden"
ROUTE = GOLDEN / "route.json"

# Must match `document::DOCUMENT_EXTENSIONS`.
EXTENSIONS = [
    "pdf", "doc", "docx", "docm", "dot", "odt", "rtf", "epub", "ppt", "pptx", "pptm",
    "pps", "ppsx", "ppsm", "pot", "xls", "xlsx", "xlsm", "xlsb", "ods", "odp",
]

# Parser families: an unmeasured extension inherits the verdict of its base.
FAMILY = {
    "docm": "docx",
    "dot": "doc",
    "pptm": "pptx",
    "pps": "ppt",
    "ppsx": "pptx",
    "ppsm": "pptx",
    "pot": "ppt",
    "xlsm": "xlsx",
    "xlsb": "xls",
}

DELTA_THRESHOLD = 0.03
WIN_RATIO = 2
LATENCY_CEILING = 1.5
WIN_TOLERANCE = 0.01


def paired(rows: list[dict], ext: str) -> list[tuple[float, float, float, float]]:
    """(anydoc_f1, xberg_f1, anydoc_ms, xberg_ms) for documents both engines converted."""
    pairs = []
    for row in rows:
        if row["ext"] != ext:
            continue
        anydoc, xberg = row.get("anydoc", {}), row.get("xberg", {})
        if anydoc.get("ok") and xberg.get("ok") and anydoc.get("ms") and xberg.get("ms"):
            pairs.append((anydoc.get("f1"), xberg.get("f1"), anydoc["ms"], xberg["ms"]))
    return pairs


def stats_for(rows: list[dict], ext: str) -> dict | None:
    pairs = paired(rows, ext)
    scored = [(a, x) for a, x, _, _ in pairs if a is not None and x is not None]
    if len(scored) < 3:
        return None
    deltas = [x - a for a, x in scored]
    ratios = [x / a for _, _, a, x in pairs]
    return {
        "paired": len(scored),
        "anydoc_f1": round(statistics.mean(a for a, _ in scored), 4),
        "xberg_f1": round(statistics.mean(x for _, x in scored), 4),
        "f1_delta": round(statistics.mean(deltas), 4),
        "wins": sum(1 for delta in deltas if delta > WIN_TOLERANCE),
        "losses": sum(1 for delta in deltas if delta < -WIN_TOLERANCE),
        "latency_ratio": round(statistics.median(ratios), 3),
    }


def verdict(stats: dict | None) -> str:
    if stats is None:
        return "anydoc"
    qualifies = (
        stats["f1_delta"] >= DELTA_THRESHOLD
        and stats["wins"] >= WIN_RATIO * max(1, stats["losses"])
        and stats["latency_ratio"] <= LATENCY_CEILING
    )
    return "xberg" if qualifies else "anydoc"


def derive() -> tuple[dict[str, str], dict[str, dict]]:
    rows = [json.loads(line) for line in (RESULTS / "per_doc.jsonl").read_text().splitlines() if line.strip()]
    stats = {ext: stats_for(rows, ext) for ext in EXTENSIONS}
    decisions = {}
    for ext in EXTENSIONS:
        base = FAMILY.get(ext, ext)
        decisions[ext] = verdict(stats.get(base)) if base != ext else verdict(stats.get(ext))
    return decisions, {ext: value for ext, value in stats.items() if value}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    decisions, stats = derive()
    summary = json.loads((RESULTS / "summary.json").read_text()) if (RESULTS / "summary.json").exists() else {}
    snapshot = json.loads((BENCH / "snapshot.json").read_text())

    print(f"{'format':6s} {'pair':>4s} {'d-F1':>7s} {'win:loss':>9s} {'lat':>5s} {'verdict':>7s}")
    for ext in EXTENSIONS:
        value = stats.get(FAMILY.get(ext, ext))
        if value is None:
            label = "no data" if FAMILY.get(ext) is None else f"via {FAMILY[ext]}"
            print(f"{ext:6s} {'-':>4s} {'-':>7s} {'-':>9s} {'-':>5s} {decisions[ext]:>7s} ({label})")
        else:
            print(
                f"{ext:6s} {value['paired']:>4d} {value['f1_delta']:>+7.3f} "
                f"{value['wins']:>4d}:{value['losses']:<4d} {value['latency_ratio']:>5.2f} {decisions[ext]:>7s}"
            )

    if args.check:
        if not ROUTE.exists():
            print("no golden/route.json to check", file=sys.stderr)
            return 1
        golden = json.loads(ROUTE.read_text())
        drift = {
            ext: (golden["formats"].get(ext), decisions[ext])
            for ext in EXTENSIONS
            if golden["formats"].get(ext) != decisions[ext]
        }
        if drift:
            print("routing drift:", json.dumps(drift, indent=1), file=sys.stderr)
            return 1
        if golden.get("router_revision") != snapshot.get("pins", {}).get("router_revision", golden.get("router_revision")):
            print("router revision pin mismatch", file=sys.stderr)
        print("golden/route.json matches the measured snapshot")
        return 0

    if not args.update:
        return 0

    GOLDEN.mkdir(parents=True, exist_ok=True)
    route = {
        "schema": "operator-doc-router-route/1",
        "router_revision": 1,
        "anydoc": snapshot["pins"]["anydoc"],
        "xberg": snapshot["pins"]["xberg"],
        "rule": (
            "xberg primary iff mean paired token-F1 delta >= +0.03 and wins >= 2x losses "
            "and xberg latency <= 1.5x anydoc; otherwise anydoc"
        ),
        "evidence": {ext: stats.get(FAMILY.get(ext, ext)) for ext in EXTENSIONS},
        "formats": decisions,
    }
    ROUTE.write_text(json.dumps(route, indent=1) + "\n")
    (GOLDEN / "summary.json").write_text(json.dumps({"pins": snapshot["pins"], "formats": stats}, indent=1) + "\n")
    shutil.copyfile(RESULTS / "per_doc.jsonl", GOLDEN / "per_doc.jsonl")
    shutil.copyfile(RESULTS / "truth_hashes.json", GOLDEN / "truth_hashes.json")
    print(f"updated {ROUTE} and golden evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
