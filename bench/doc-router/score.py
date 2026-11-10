#!/usr/bin/env python3
"""Score the snapshot run: ground-truth F1 per document and per-format speed.

Ground truth is extracted independently of both engines — container XML for
OOXML/ODF/EPUB/CSV, PyMuPDF for PDF text layers (when `uv run --with pymupdf`
is used), LibreOffice for legacy binary formats and RTF. Truth text is cached
under `.cache/truth/` and its hash recorded in `results/truth_hashes.json`, so a
re-run on a different LibreOffice/HarfBuzz stack reports truth drift instead of
silently changing the scores.

Outputs:
  results/per_doc.jsonl     one row per document with both engines' results
  results/summary.json      per-format means and ratios
  results/truth_hashes.json

Usage: python3 bench/doc-router/score.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import csv as csvmod
import hashlib
import json
import re
import statistics
import subprocess
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "bench" / "doc-router"
CACHE = BENCH / ".cache"
RESULTS = BENCH / "results"
SOFFICE = "/opt/homebrew/bin/soffice"

TOKEN_RE = re.compile(r"[^\W_]+(?:[.,][^\W_]+)?", re.UNICODE)
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*)?/?>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)
LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+\S", re.M)
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)
LINK_RE = re.compile(r"\[[^\]]*\]\(")


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def xml_text(blob) -> str:
    text = blob.decode("utf-8", "replace") if isinstance(blob, (bytes, bytearray)) else str(blob)
    text = re.sub(r"<w:tab[^>]*/>", "\t", text)
    text = re.sub(r"<w:br[^>]*/>", "\n", text)
    text = re.sub(r"<[^>]+>", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def truth_ooxml(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if path.suffix.lower() == ".docx":
            wanted = sorted(
                name
                for name in names
                if re.fullmatch(r"word/(document|header\d*|footer\d*|footnotes|endnotes)\.xml", name)
            )
            return "\n".join(xml_text(archive.read(name)) for name in wanted)
        if path.suffix.lower() == ".pptx":
            wanted = sorted(name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
            return "\n".join(xml_text(archive.read(name)) for name in wanted)
        # xlsx / xlsm
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            blob = archive.read("xl/sharedStrings.xml").decode("utf-8", "replace")
            shared = [re.sub(r"<[^>]+>", "", match) for match in re.findall(r"<si>(.*?)</si>", blob, re.S)]
        parts: list[str] = []
        for name in sorted(names):
            if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
                continue
            blob = archive.read(name).decode("utf-8", "replace")
            for attrs, cell in re.findall(r"<c\b([^>]*)>(.*?)</c>", blob, re.S):
                if "<is>" in cell:
                    parts.append(xml_text(cell))
                    continue
                value = re.search(r"<v>(.*?)</v>", cell, re.S)
                if not value:
                    continue
                if 't="s"' in attrs:
                    try:
                        parts.append(shared[int(value.group(1))])
                    except (ValueError, IndexError):
                        pass
                else:
                    parts.append(value.group(1))
            parts.append("\n")
        return "\n".join(parts)


def truth_odf(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        text = archive.read("content.xml").decode("utf-8", "replace")
    text = re.sub(r"<text:(tab|line-break)[^>]*/>", " ", text)
    text = re.sub(r"<[^>]+>", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def truth_epub(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.lower().endswith((".xhtml", ".html", ".htm")):
                collector = _TextCollector()
                collector.feed(archive.read(name).decode("utf-8", "replace"))
                parts.append("".join(collector.parts))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(parts).strip())


def truth_csv(path: Path) -> str:
    text = path.read_text("utf-8", "replace")
    try:
        return "\n".join(" ".join(row) for row in csvmod.reader(text.splitlines()))
    except Exception:
        return text


def truth_pdf(path: Path) -> str | None:
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF via `uv run --with pymupdf`
    except ImportError:
        return None
    try:
        with fitz.open(path) as document:
            return "\n\n".join(page.get_text("text") for page in document)
    except Exception:
        return None


def truth_libreoffice(path: Path, cache: Path) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (hashlib.sha256(str(path).encode()).hexdigest()[:16] + ".txt")
    if target.exists() and target.stat().st_size:
        return target.read_text("utf-8", "replace")
    subprocess.run(
        [
            SOFFICE,
            f"-env:UserInstallation=file://{BENCH / '.lo-profile'}",
            "--headless",
            "--convert-to",
            "txt:Text",
            "--outdir",
            str(cache),
            str(path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=300,
        check=False,
    )
    produced = cache / (path.stem + ".txt")
    if not produced.exists():
        return ""
    target.write_text(produced.read_text("utf-8", "replace"))
    produced.unlink(missing_ok=True)
    return target.read_text("utf-8", "replace")


def truth_text(path: Path, ext: str) -> str:
    try:
        if ext in {"docx", "pptx", "xlsx", "xlsm"}:
            return truth_ooxml(path)
        if ext in {"odt", "ods", "odp"}:
            return truth_odf(path)
        if ext == "epub":
            return truth_epub(path)
        if ext == "csv":
            return truth_csv(path)
        if ext == "pdf":
            text = truth_pdf(path)
            if text is not None:
                return text
    except Exception as error:  # corrupt container: fall back to LibreOffice
        print(f"  truth fallback {path.name}: {error}", file=sys.stderr)
    return truth_libreoffice(path, CACHE / "truth")


def tokens(text: str) -> collections.Counter:
    return collections.Counter(TOKEN_RE.findall(text.casefold()))


def md_to_text(markdown: str) -> str:
    markdown = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    markdown = re.sub(r"`([^`]*)`", r"\1", markdown)
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", markdown)
    markdown = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown)
    markdown = re.sub(r"^\s{0,3}#{1,6}\s*", "", markdown, flags=re.M)
    markdown = re.sub(r"^\s{0,3}>\s?", "", markdown, flags=re.M)
    markdown = re.sub(r"[*_~]{1,3}", "", markdown)
    markdown = re.sub(r"^\s*[-+*]\s+", "", markdown, flags=re.M)
    markdown = re.sub(r"^\s*\d+[.)]\s+", "", markdown, flags=re.M)
    markdown = markdown.replace("|", " ")
    markdown = re.sub(r"<br\s*/?>", " ", markdown)
    return HTML_TAG_RE.sub(" ", markdown)


def token_f1(truth: str, output: str) -> float:
    truth_tokens = tokens(truth)
    output_tokens = tokens(md_to_text(output))
    if not truth_tokens or not output_tokens:
        return 0.0
    overlap = sum((truth_tokens & output_tokens).values())
    precision = overlap / sum(output_tokens.values())
    recall = overlap / sum(truth_tokens.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def structure(markdown: str) -> dict:
    return {
        "headings": len(HEADING_RE.findall(markdown)),
        "list_items": len(LIST_RE.findall(markdown)),
        "table_rows": len(TABLE_RE.findall(markdown)),
        "links": len(LINK_RE.findall(markdown)),
    }


def cleanliness(markdown: str) -> dict:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", markdown) if part.strip()]
    return {
        "html_leak": len(HTML_TAG_RE.findall(markdown)),
        "control_chars": len(CONTROL_RE.findall(markdown)),
        "duplicate_paragraphs": len(paragraphs) - len(set(paragraphs)),
    }


def cached_path(doc: dict) -> Path:
    root = "test_documents" if doc["corpus"] == "xberg-corpus" else "tika"
    return CACHE / root / doc["rel"]


def load_rows(engine: str) -> dict[str, list[dict]]:
    path = RESULTS / f"{engine}.jsonl"
    rows: dict[str, list[dict]] = collections.defaultdict(list)
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]].append(row)
    return rows


def median_ns(rows: list[dict]) -> float | None:
    good = [row for row in rows if row["rep"] >= 0 and row["ok"]]
    if not good:
        return None
    return statistics.median(row["ns"] for row in good) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only the first N documents per format")
    args = ap.parse_args()

    snapshot = json.loads((BENCH / "snapshot.json").read_text())
    documents = [doc for doc in snapshot["documents"] if doc["set"] == "bench"]
    if args.limit:
        per_format: dict[str, int] = collections.Counter()
        limited = []
        for doc in documents:
            if per_format[doc["ext"]] < args.limit:
                per_format[doc["ext"]] += 1
                limited.append(doc)
        documents = limited

    anydoc_rows = load_rows("anydoc")
    xberg_rows = load_rows("xberg")
    auto_rows = load_rows("auto")

    truth_hashes: dict[str, str] = {}
    per_doc: list[dict] = []
    for index, doc in enumerate(documents, 1):
        path = cached_path(doc)
        truth = truth_text(path, doc["ext"])
        truth_hashes[doc["id"]] = hashlib.sha256(truth.encode()).hexdigest()
        record: dict = {
            "id": doc["id"],
            "ext": doc["ext"],
            "corpus": doc["corpus"],
            "size": doc["size"],
            "truth_chars": len(truth),
        }
        for engine, rows in (("anydoc", anydoc_rows), ("xberg", xberg_rows)):
            ns = median_ns(rows.get(doc["id"], []))
            output_path = RESULTS / "outputs" / engine / (doc["id"].replace("/", "__") + ".md")
            f1 = None
            padded = None
            if output_path.exists() and truth.strip():
                markdown = output_path.read_text("utf-8", "replace")
                f1 = round(token_f1(truth, markdown), 4)
                padded = {
                    **structure(markdown),
                    **cleanliness(markdown),
                    "chars": len(markdown),
                }
            record[engine] = {
                "ok": ns is not None,
                "ms": round(ns, 3) if ns is not None else None,
                "f1": f1,
                "shape": padded,
            }
        router_rows = [row for row in auto_rows.get(doc["id"], []) if row["rep"] >= 0]
        if router_rows:
            picked = router_rows[0]
            record["router"] = {
                "engine": picked.get("engine_used", ""),
                "fell_back": picked.get("fell_back", False),
                "ok": picked.get("ok", False),
            }
        per_doc.append(record)
        if index % 50 == 0:
            print(f"  scored {index}/{len(documents)}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "per_doc.jsonl").write_text("".join(json.dumps(row) + "\n" for row in per_doc))
    (RESULTS / "truth_hashes.json").write_text(json.dumps(truth_hashes, indent=1) + "\n")

    by_format: dict[str, list[dict]] = collections.defaultdict(list)
    for row in per_doc:
        if row["truth_chars"]:
            by_format[row["ext"]].append(row)
    summary: dict = {"documents": len(per_doc), "formats": {}}
    print(f"{'format':6s} {'n':>4s} {'anydoc F1':>10s} {'xberg F1':>9s} {'delta':>7s} {'anydoc ms':>10s} {'xberg ms':>9s} {'ratio':>6s}")
    for ext, rows in sorted(by_format.items()):
        pairs = [
            (row["anydoc"]["f1"], row["xberg"]["f1"])
            for row in rows
            if row["anydoc"]["f1"] is not None and row["xberg"]["f1"] is not None
        ]
        deltas = [x - a for a, x in pairs]
        speed_pairs = [
            (row["anydoc"]["ms"], row["xberg"]["ms"])
            for row in rows
            if row["anydoc"]["ms"] and row["xberg"]["ms"]
        ]
        anydoc_mean = statistics.mean(a for a, _ in pairs) if pairs else None
        xberg_mean = statistics.mean(x for _, x in pairs) if pairs else None
        ratio = (
            statistics.median(x / a for a, x in speed_pairs) if speed_pairs else None
        )
        summary["formats"][ext] = {
            "documents": len(rows),
            "paired": len(pairs),
            "anydoc_f1": round(anydoc_mean, 4) if anydoc_mean is not None else None,
            "xberg_f1": round(xberg_mean, 4) if xberg_mean is not None else None,
            "f1_delta": round(statistics.mean(deltas), 4) if deltas else None,
            "wins": sum(1 for delta in deltas if delta > 0.01),
            "losses": sum(1 for delta in deltas if delta < -0.01),
            "latency_ratio": round(ratio, 3) if ratio else None,
        }
        anydoc_ms = statistics.median([a for a, _ in speed_pairs]) if speed_pairs else float("nan")
        xberg_ms = statistics.median([x for _, x in speed_pairs]) if speed_pairs else float("nan")
        print(
            f"{ext:6s} {len(pairs):>4d} "
            f"{(anydoc_mean if anydoc_mean is not None else float('nan')):>10.3f} "
            f"{(xberg_mean if xberg_mean is not None else float('nan')):>9.3f} "
            f"{(statistics.mean(deltas) if deltas else float('nan')):>7.3f} "
            f"{anydoc_ms:>10.2f} {xberg_ms:>9.2f} "
            f"{(ratio if ratio else float('nan')):>6.2f}"
        )
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {RESULTS / 'per_doc.jsonl'} and {RESULTS / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
