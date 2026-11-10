# operator

Read-only code intelligence for coding agents, served over the Model Context
Protocol. Operator indexes a workspace locally once, then gives Codex, Claude
Code, Cursor, agy, and other MCP hosts five tools: repository outline, structural
AST search, full-text search, hybrid semantic search, and bounded source reads.

No planner, no reviewer loop, no web research, no LSP process, no code edits.

## Install

```bash
cargo install --git https://github.com/ernestas-poskus/operator --locked
# or from a local checkout:
cargo install --path . --locked
```

Requirements:

- Rust 1.95+.
- A GPU for retrieval: Metal on macOS (Apple silicon), NVIDIA CUDA on Linux and
  Windows (builds on Linux/Windows need the CUDA toolkit; macOS needs no extra
  toolchain). The semantic search tool degrades to unavailable when the GPU or
  the models are missing; `search_code`, `search_ast`, `repo_map`, and
  `read_source` still work.
- The embedded models below (downloaded once into the model cache, ~2.5 GB).

## Library

`operator` is also a library. The facade exposes the same read-only operations
as the MCP server without spawning a process:

```rust,no_run
use operator::{Error, Operator};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let op = Operator::open(".").await?;
    let map = op.repo_map(None, Some("rust"), 50)?;
    let hits = op.search_code("read_source", "literal", None, None, 10).await?;
    println!("{} files, {} hits", map.len(), hits.len());
    Ok(())
}
```

`Operator::open` loads `.operator.toml` and builds the index when the local
models and GPU are available. Use `Operator::open_unindexed` for outline, AST,
literal/regex search and source reads without loading models; semantic search
then returns an `index_missing` tool error. `Operator::from_config` accepts a
programmatic `operator::Config`, and `Operator::serve_stdio` runs the MCP server
from the same in-process state.

The default feature set includes the TUI. Library-only consumers can drop the
terminal dependencies:

```toml
operator = { path = "..", default-features = false }
```

See [docs/library.md](docs/library.md) for the full API and error semantics.

## Set up a workspace

```bash
cd my-repo
operator init
```

`init` walks the workspace with ast-grep’s language detection, intersects the
result with the supported catalog, and asks which detected languages to index:

```text
Detected:
  Rust       1,284 files
  Go           312 files
  Dart          41 files
  Markdown      17 files

? Languages to index › [x] Rust (1284 files) [x] Go (312 files) ...
```

Use `--yes` for detected defaults, `--languages rust,go,dart` for an explicit
selection, or `--all-supported` to select the full catalog. The selection is
stored in `.operator.toml`; changing it changes the index identity.

```bash
operator reindex          # build/refresh the index
operator serve            # stdio MCP server
```

`init` also writes project-local host registrations (`.mcp.json`,
`.gemini/settings.json`, `.codex/config.toml`, `.cursor/mcp.json`,
`.agents/mcp_config.json`). `serve --no-register` skips that.

## Terminal UI

On a TTY, `operator init` opens a five-tab TUI: **Sources** (languages + include/
exclude tree), **Index** (progress, freshness, file events), **Stats**,
**Usage** (MCP tool calls and learn signals), and **Review**. `operator tui`
reopens the dashboard; `operator reindex --tui` starts it on the Index tab.

See [docs/tui.md](docs/tui.md) for keys and selection semantics. Non-TTY use
`operator init --yes` (or pass `--languages`, `--include`, `--exclude`).

## MCP tools

| Tool | Use |
|---|---|
| `repo_map` | ast-grep outline: files, symbols, imports, exports; generic file entries where no outline rule exists. |
| `search_code` | Indexed full-text (`fts`), literal substring, or regex search with path/language filters. |
| `search_ast` | ast-grep pattern or YAML rule search; read-only (fix/rewrite rules are rejected). |
| `search_semantic` | LanceDB dense + full-text fusion with optional rerank; requires `operator reindex`. |
| `read_source` | Exact line range of a workspace-contained file. |

All results are bounded and read-only. Operator does not expose type-aware
references, diagnostics, rename, or edits; use the host’s compiler/LSP for those.

## Languages

All ast-grep bundled languages:

Bash, C, C++, C#, CSS, Dart, Elixir, Go, Haskell, HCL, HTML, Java, JavaScript,
JSON, Kotlin, Lua, Markdown, Nix, PHP, Python, Ruby, Rust, Scala, Solidity,
Swift, TSX, TypeScript, YAML, Zig.

“Support” means ast-grep detection, parsing and structural search; languages
with a bundled outline rule also get symbols. Languages without one get a
generic file entry. Indexing is language-agnostic, so every selected language
participates in full-text and semantic search.

## Learn from tool usage

`serve` journals its own MCP calls locally (`~/.local/state/operator`) with
hashed arguments and result shapes. It does not see host-native tools or other
MCP servers, and it never edits agent instructions.

```bash
operator learn --json        # repeated calls, repeated errors, mismatches
```

The report suggests wording or description changes; applying them is manual.

## Configuration

`.operator.toml` is written by `init` and holds only the selected languages and
retrieval settings:

```toml
schema_version = 3
languages = ["rust", "go", "dart"]
# retrieval_include = ["docs"]
# retrieval_exclude = ["tests/fixtures"]
# retrieval_embed_model = "mixedbread-ai/mxbai-embed-large-v1"
# retrieval_rerank_model = "Qwen/Qwen3-Reranker-0.6B"
# retrieval_model_cache_dir = ".cache/retrieval/models"  # default; any HF cache is discovered first
# retrieval_model_dtype = "f16"                          # or "f32"
# retrieval_embed_backend = "bert"                       # or "qwen3"
# retrieval_embed_dim = 1024                             # model width (e.g. 2560 for Qwen3-4B)
# retrieval_rerank_blend = 0.0                           # 1.0 = pure reranker order
# retrieval_rerank_context = false                       # Module: hint on reranker docs
# retrieval_query_expansion = false                      # identifier tokens in the dense query
# retrieval_fusion_gate = false                          # RRF only when lexical adds candidates
# retrieval_query_log = "/path/search-queries.jsonl"      # opt-in, query text + paths only
# retrieval_dual_embed_model = "Qwen/Qwen3-Embedding-0.6B" # opt-in secondary index
# retrieval_dual_embed_dim = 1024
# retrieval_dual_embed_backend = "qwen3"
# retrieval_dual_route = true
```

Retrieval runs in-process through fastembed's candle backend: Metal on macOS,
CUDA on Linux/Windows. An existing Hugging Face / fastembed cache is discovered
before anything downloads; see
[`docs/retrieval-backend.md`](docs/retrieval-backend.md).

The index lives under `.cache/retrieval` and is keyed by the selected language
set, chunker, embedder and backend, and include roots. Changing any of them
requires `operator reindex`. Changing the embedder or backend changes the
index identity, so the first run after such a change reindexes.

## Development

```bash
cargo check
cargo nextest run --status-level fail -p operator
cargo clippy --all-targets
cargo fmt -p operator
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
