# Tool catalog

Code-intelligence exposes five read-only MCP tools over stdio. There are no plan,
research, web, LSP, stats, replay, or schema tools.

## `repo_map`

Inputs: `path?`, `language?`, `limit?` (default 200).

Uses the in-process `ast-grep-outline` extractor. Languages with a bundled
outline rule return items, members, signatures, imports and exports. Languages
without one still appear as file entries with an empty symbol list.

## `search_code`

Inputs: `query`, `mode` (`fts|literal|regex`), `path?`, `language?`, `limit?`.

`fts` uses the LanceDB full-text index when the workspace is indexed; otherwise
it falls back to a literal scan. Literal and regex modes always scan the
selected-language corpus. Results are bounded to 200 matches.

## `search_ast`

Inputs: `pattern` xor `rule` (YAML), `language?`, `paths?`, `limit?`.

Patterns and rules run through `ast-grep-core`/`ast-grep-config` in-process.
`fix`, `rewriters`, interactive editing and update mode are rejected. Parse
errors are returned as tool errors with the source path.

## `search_semantic`

Inputs: `query`, `path?`, `language?`, `limit?`.

Hybrid dense + full-text retrieval through the existing LanceDB index and the
in-process fastembed embedder/reranker. Requires `code-intelligence reindex`; missing
index is an explicit error, never an empty result. Results carry path/line/score.

## `read_source`

Inputs: `path`, `start_line?`, `end_line?`.

Canonicalizes the path and rejects anything outside the workspace. Reads UTF-8
text only; no document conversion or OCR.

## Activation

`init` must select at least one language before `serve` starts. Language
filters are restricted to the selected set. `search_semantic` is unavailable
when no index exists or the model backend is unavailable; the other four tools
remain usable.
