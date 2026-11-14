# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Fixed

- The TUI no longer leaves index log lines over the Sources view: while the
  dashboard owns the terminal, index diagnostics and MCP-registration notes are
  routed to the Index events panel instead of stderr, hf-hub progress bars are
  disabled, and a tab switch forces a full repaint.

## [Unreleased]

### Added

- Read-only code-intelligence MCP server exposing `repo_map`, `search_code`,
  `search_ast`, `search_semantic` and `read_source` over stdio.
- In-process indexing and hybrid retrieval built on ast-grep, LanceDB and
  fastembed/candle, with no HTTP model server.
- `code-intelligence init`, `serve`, `reindex`, `tui` and `learn` commands.
- Interactive init/dashboard TUI with language selection, include/exclude
  paths, index progress, freshness, stats, MCP tool usage and file events.
- `code_intelligence::Operator` library facade for in-process code intelligence.
- Optional `tui` feature; `--no-default-features` builds a lean library and
  headless binary.
- Newtype wrappers for the crate's domain primitives (language ids, repo
  paths, source text, counts, line numbers, scores, model/table names, paths
  and configuration flags). The wrappers are `serde(transparent)`, so the JSON
  and `.code-intelligence.toml` wire formats are unchanged.
- Document conversion now routes per format between anydoc 0.2 (primary for
  every format it wins) and xberg 1.3 (primary for `pptx`, the one format where
  it is more faithful and faster), with a typed `best-effort`/`strict` fallback
  policy, `retrieval_document_engine` and `retrieval_document_fallback`
  configuration, chunk provenance naming the engine, and a document-corpus
  identity that re-embeds when any of them changes. The routing table is
  measured: `bench/doc-router/` freezes the 507-document snapshot, both engines'
  raw results and `golden/route.json`, and `tests/document_router.rs` fails if
  the code and the golden table drift. See `docs/document-router.md`.
  Cost: +39 MB unstripped binary (+12%), no measurable build-time change.

### Changed

- The binary, MCP server entry, config file and state directory now use the
  `code-intelligence` name: hosts register `code-intelligence` (the previous
  `operator` entry is removed while reconciling), the config file is
  `.code-intelligence.toml`, and journals live under
  `~/.local/state/code-intelligence`. An existing `.operator.toml` is read and
  replaced on the next config save (`init` or the TUI), and host files are
  migrated on the next `init` or `serve`.
- Model weights are downloaded into the shared per-user cache
  (`retrieval_model_cache_dir` when set, else `HF_HOME/hub` / `HF_HUB_CACHE`
  when configured, else `XDG_CACHE_HOME/fastembed`, else
  `~/.cache/fastembed`) instead of the workspace's retrieval cache, so the
  ~2.5 GB snapshot is reused by every workspace on the machine. A complete
  cache found anywhere is still used as-is; the workspace path is only the
  fallback when no home directory is known.
- The TUI Sources tree lists `.gitignore`d sources in a dimmed, collapsed,
  unchecked section at the bottom, labelled `gitignored`; expanding and
  including one still overrides the ignore for that path.

- Interactive CLI prompts use [`demand`](https://github.com/jdx/demand); the
  `dialoguer` dependency is gone. `code-intelligence init`'s language picker is
  filterable, requires at least one selection, and returns typed
  `LanguageId` values instead of indexes.

### Fixed

- Model loading no longer creates a second `.fastembed_cache` directory next to
  the workspace `.cache`: the resolved model cache is pinned before any
  fastembed loader runs, and only `.cache` roots are probed.
