# Library API

`code-intelligence` is both a binary and a library. The library facade is
[`code_intelligence::Operator`](../src/api.rs); it owns a workspace configuration and an
optional local index and exposes the same read-only operations as the MCP
server.

```rust,no_run
use code_intelligence::{
    Config, Error, LanguageId, Limit, LineNumber, Operator, QueryText, RepoPath,
};

#[tokio::main]
async fn main() -> Result<(), Error> {
    // A configured workspace: loads `.operator.toml`, validates it and tries
    // to build the local retrieval index.
    let op = Operator::open(".").await?;

    let rust = LanguageId::try_from("rust")?;
    let outline = op.repo_map(None, Some(&rust), Limit::new(100))?;
    let hits = op
        .search_code(
            &QueryText::from("read_source"),
            "literal",
            None,
            None,
            Limit::new(20),
        )
        .await?;
    let source = op.read_source(
        &RepoPath::from("src/lib.rs"),
        Some(LineNumber::new(1)),
        Some(LineNumber::new(40)),
    )?;

    println!("{} files, {} hits, {} bytes", outline.len(), hits.len(), source.len());
    Ok(())
}
```

## Constructors

| Constructor | Index | Model cache |
|---|---|---|
| `Operator::open(path)` | built when available | prepared |
| `Operator::open_unindexed(path)` | none | untouched |
| `Operator::from_config(cfg)` | built when available | prepared |
| `Operator::from_config_unindexed(cfg)` | none | untouched |

`open` and `open_unindexed` load `.operator.toml`; `from_config` accepts a
programmatic [`Config`] for callers that do not use the CLI setup. `open`
requires a Tokio runtime; the `unindexed` constructors and the sync tools do
not.

An index build failure is not fatal to `open`: `has_index()` returns `false`,
the lexical/AST/read tools keep working, and semantic search returns
`Error::Tool` with an `index_missing` message. Use `reindex()` to retry.

## Tools

- `repo_map(path, language, limit)` — ast-grep outline.
- `search_code(query, mode, path, language, limit)` — `fts`, `literal`, or
  `regex`; `fts` falls back to the scanner when no index is loaded.
- `search_ast(pattern, rule, path, language, limit)` — pattern or YAML rule;
  `fix`/rewrite rules are rejected.
- `search_semantic(query, path, language, limit)` — dense + FTS fusion with
  optional rerank; requires an index.
- `read_source(path, start_line, end_line)` — workspace-contained line range.
- `index_status()` — inspect the on-disk index without building it.
- `reindex()` — build/refresh and replace the in-memory index.
- `journal_stats()` — read the local MCP usage journal.
- `serve_stdio()` — run the five-tool MCP server from this in-process state.

All limits are clamped to `1..=1000`. Direct library calls do not write the MCP
usage journal; only `serve_stdio` does, because journaling lives in the MCP
wrapper.

## Newtypes

The value types that cross the library boundary are newtypes over their
primitives — `LanguageId`, `RepoPath`, `SourceText`, `LineNumber`, `FileCount`,
`Score`, `Limit`, `WorkspaceRoot` and friends — instead of bare `String`,
`usize`, `f32` or `PathBuf`. They convert explicitly (`LanguageId::parse`,
`.get()`, `.as_str()`) or implicitly (`From`/`Into`), and compare against their
inner primitive (`count == 1`, `line > 0`). Every wrapper is
`serde(transparent)` and `schemars(inline)`, so the JSON, `.operator.toml` and
MCP tool-input schemas are unchanged: the typing is compile-time only.

## Errors

`code_intelligence::Error` distinguishes configuration (`Config`), I/O (`Io`), JSON
(`Json`) and tool/index failures (`Tool`). Tool methods return the same
read-only error strings the MCP tool results carry, wrapped in `Error::Tool`.

## Features

The default `tui` feature brings `ratatui`, `crossterm` and `notify` for the
init/dashboard. Disable it for a smaller library or headless binary:

```toml
code-intelligence = { version = "0.1.0", default-features = false }
```

The binary remains available either way; without `tui`, `code-intelligence tui` is not a
subcommand and `init --tui` / `reindex --tui` return a configuration error.
