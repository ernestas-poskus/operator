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
