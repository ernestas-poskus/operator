# Agent instructions

This repository builds a read-only code-intelligence MCP server. Do not add
planning, research, web, LSP, or code-editing features without a separate
design decision.

## Workflow

- Work in a dedicated git worktree; merge with `--ff-only` and remove it.
- Run `cargo check`, scoped `cargo nextest run --status-level fail -p operator`,
  `cargo clippy --all-targets`, and `cargo fmt -p operator`.
- Use `type(component): intent` commit messages. Do not add co-author trailers.
- Search with `rg`; read only the needed range. Keep tool/result records
  backward tolerant and never log source contents to the learn journal by
  default.

## Type conventions

- **Newtype pattern.** Every domain primitive gets a newtype in
  `src/types.rs`: language ids, paths, source text, counts, line numbers,
  scores, model/table names, timestamps and flags. Do not add bare `String`,
  `usize`, `u32`, `u64`, `i64`, `f32`, `bool` or `PathBuf` fields to
  configuration, tool arguments/results, index records, journal records, or
  the public API. Keep wrappers `#[serde(transparent)]` and
  `#[schemars(inline)]`, so the TOML/JSONL/MCP wire formats and host schemas
  stay unchanged; convert explicitly (`new`/`get`/`as_str`) or with
  `From`/`Into`. `argh` switches, transient TUI state and intra-function loop
  locals are the only allowed primitives; justify any other exception in the
  type's doc comment.
- **Time and dates use `time`.** Represent wall-clock instants and dates with
  `time::OffsetDateTime`/`time::Date` and durations with `time::Duration`,
  wrapped in the newtypes above — never `SystemTime`, `UNIX_EPOCH`, or bare
  epoch integers in the data model. `std::time::Instant` is only for measuring
  elapsed work. Wire records keep their established units (Unix
  seconds/millis, RFC 3339); serialize and parse them at the `src/types.rs`
  boundary, never at call sites, and keep older records readable. The existing
  integer-backed `UnixSeconds`, `DurationMs` and `UpdatedAt` are the only
  exception and migrate to `time` when their records next change.

## Architecture boundaries

- `src/api.rs` is the library facade: [`Operator`] plus the re-exported value
  types. Keep implementation modules private; the binary may use them, library
  consumers should use `api`.
- The `tui` feature is default-on for the binary and optional for library
  consumers; TUI code must not compile into a `--no-default-features` build.
- `src/languages.rs` owns ast-grep language detection and the catalog.
- `src/ast_grep.rs` owns structural search and outline.
- `src/code_search.rs`/`src/retrieval.rs` own indexing and hybrid search.
- `src/tools.rs` owns the five MCP tool implementations.
- `src/mcp_server.rs` owns stdio MCP schemas/dispatch.
- `src/learn.rs` owns the local journal and deterministic report.
- `src/tui.rs` owns the init/dashboard terminal UI; it must never write to MCP
  stdio and must restore the terminal on every exit path.
- `src/code_search.rs` publishes `IndexEvent` progress and `RetrievalIndexStatus`
  freshness; the TUI and CLI share the same discovery/exclude semantics.
- The server is read-only; reject `fix`/`rewrite`/code-action behavior.
