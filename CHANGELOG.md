# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Read-only code-intelligence MCP server exposing `repo_map`, `search_code`,
  `search_ast`, `search_semantic` and `read_source` over stdio.
- In-process indexing and hybrid retrieval built on ast-grep, LanceDB and
  fastembed/candle, with no HTTP model server.
- `operator init`, `serve`, `reindex`, `tui` and `learn` commands.
- Interactive init/dashboard TUI with language selection, include/exclude
  paths, index progress, freshness, stats, MCP tool usage and file events.
- `operator::Operator` library facade for in-process code intelligence.
- Optional `tui` feature; `--no-default-features` builds a lean library and
  headless binary.
- Newtype wrappers for the crate's domain primitives (language ids, repo
  paths, source text, counts, line numbers, scores, model/table names, paths
  and configuration flags). The wrappers are `serde(transparent)`, so the JSON
  and `.operator.toml` wire formats are unchanged.

### Changed

- Interactive CLI prompts use [`demand`](https://github.com/jdx/demand); the
  `dialoguer` dependency is gone. `operator init`'s language picker is
  filterable, requires at least one selection, and returns typed
  `LanguageId` values instead of indexes.
