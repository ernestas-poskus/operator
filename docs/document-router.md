# Document conversion router

`code-intelligence` converts office documents and text PDFs to Markdown when the opt-in
document corpus is indexed (`retrieval_documents`). Two in-process engines are
compiled in and a measured table decides which one converts each format.

## Why two engines

| | anydoc 0.2.4 | xberg 1.3.10 (extraction) |
|---|---|---|
| Scope | 14 document formats → GFM Markdown | 107 formats, plus OCR/layout/audio (not enabled here) |
| Failure policy | fail-closed (`NeedsOcr`, `Malformed`, `Encrypted`, …) | best-effort (may return partial or empty content) |
| Cost | ~8 direct deps, 24 k LOC | ~124 direct deps, ~595 k LOC in its core crate |
| Measured | faster on 15 of 17 formats and equal or better fidelity on all but two | more faithful on `pptx` and `docx`, and the only engine with a partial read for scanned PDFs |

The full experiment — 501 documents, two corpora, 17 formats — lives in
[`bench/doc-router/`](../bench/doc-router/README.md). Its frozen verdict is
`bench/doc-router/golden/route.json`, and `tests/document_router.rs` fails if
this document, that file and `src/document.rs` disagree.

## The table

| format | primary engine | why (snapshot run, xberg 1.3.10) |
|---|---|---|
| `pptx`, `pptm`, `ppsx`, `ppsm` | **xberg** | paired F1 0.974 vs 0.899 (+0.075) over 25 documents, 11 wins to 1, and 0.88× anydoc's latency |
| `docx`, `docm` | anydoc | xberg's +0.027 is below the +0.03 bar (10 wins, 5 losses) and it is 1.06× slower |
| `doc`, `dot` | anydoc | F1 0.997 vs 0.972, 3.9× faster |
| `odt` | anydoc | F1 0.962 vs 0.947, 1.14× faster |
| `ods`, `odp` | anydoc | `odp`: 0.998 vs 0.808 — xberg is 2× faster there but far less faithful |
| `rtf` | anydoc | tied on F1 (0.916 vs 0.917), 2.5× faster |
| `epub` | anydoc | 0.778 vs 0.770, 2.4× faster |
| `xls`, `xlsx`, `xlsm`, `xlsb` | anydoc | `xlsx` is a tie (0.836 vs 0.839) at 3.2× the speed |
| `pdf` (text) | anydoc | 0.918 vs 0.935 (+0.018, below the bar) at 2.5× the speed |
| `pdf` (scanned) | anydoc reports `NeedsOcr`; xberg is tried only under `best-effort` | see below |

`xls` has too few documents with independent ground truth to decide on its own,
so it (and `xlsb`) inherit anydoc from the spreadsheet family.

`csv` never reaches the converter: a delimited text file is already UTF-8 and is
read verbatim.

The rule that produces the table is mechanical and re-derivable:

> xberg becomes primary for a format when mean paired token-F1(xberg) −
> token-F1(anydoc) ≥ +0.03 **and** xberg wins at least twice as many documents
> as it loses (|Δ| > 0.01) **and** median latency(xberg) ≤ 1.5 × anydoc;
> otherwise anydoc.

## Fallback policy

When the primary engine fails, `retrieval_document_fallback` decides:

* **`best-effort`** (default): try the other engine. A non-empty conversion is
  accepted and the primary error is recorded on the result and on the chunk
  provenance (`converted document via xberg`). Encryption and I/O errors never
  fall back, and an empty conversion is a failure, never a result.
* **`strict`**: return the primary engine's error. `NeedsOcr` stays the
  escalation signal it has always been.

Scanned PDFs are the case this exists for: anydoc refuses the whole document
with `NeedsOcr`; under `best-effort` xberg may return the text-bearing pages,
which is better than dropping the document from the corpus, and the recorded
fallback keeps that visible. Under `strict` nothing is silently truncated.

## Panic isolation

xberg's `xlsb` path can panic inside its `calamine` dependency on a corrupt
package. The adapter runs each xberg conversion on a spawned task and maps a
`JoinError` to `DocError::Malformed`, so one document cannot abort a corpus
build — provided the binary unwinds. A build that sets `panic = "abort"`
(for example a global `~/.cargo/config.toml`) loses that protection; this crate
does not set `panic = "abort"` and the experiment's
`bench/doc-router/cargo-override.toml` pins `unwind` so the measured binary
matches the shipped profile.

## Cost

Measured on the snapshot's host (Apple M4 Max, 16 cores), cold `cargo build
--release --bin code-intelligence` with the same profile on both sides:

| | before (anydoc only) | after (both engines) | delta |
|---|---|---|---|
| build wall time | 340 s | 326 s | within noise |
| peak RSS during build | 10.2 GB | 10.4 GB | +0.2 GB |
| binary, unstripped | 318 MB | 357 MB | +39 MB (+12%) |
| binary, `strip -x` | 258 MB | 288 MB | +30 MB (+12%) |

The second engine costs about 12% of an already large binary (LanceDB and the
embedding stack dominate) and nothing measurable at build time. The
alternative — a separate conversion process — would remove the in-process
guarantee and the per-document panic isolation.

## Configuration

```toml
# .operator.toml
retrieval_documents = true
retrieval_document_engine = "auto"        # auto | anydoc | xberg
retrieval_document_fallback = "best-effort" # best-effort | strict
```

`auto` follows the table above. Pinning `anydoc` or `xberg` is for A/B runs,
debugging and escape hatches; it does not change the fallback semantics.

## Identity and re-embedding

The document corpus identity includes the router revision, both engine
versions, the configured mode and the fallback policy
(`document::router_identity`). Any of them changing produces a new LanceDB table
and a full re-embed, because the ingestion manifest records file hashes, not
converted text. Source-only corpora keep the incumbent anydoc version fragment,
so their table names and manifests are unchanged.

Bump `document::ROUTER_REVISION` whenever the table, an engine version or the
fallback semantics change.

## Re-running the experiment

```sh
python3 bench/doc-router/fetch.py            # ~540 MB, hash-verified
python3 bench/doc-router/run.py              # anydoc, xberg, and the router itself
uv run --with pymupdf python bench/doc-router/score.py
python3 bench/doc-router/derive.py --update  # rewrite golden/route.json
```

`derive.py --check` fails when the measured verdict differs from the committed
table. See [`bench/doc-router/README.md`](../bench/doc-router/README.md) for the
layout, tolerances and licensing of the snapshot.
