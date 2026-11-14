# Terminal UI

`code-intelligence init` opens a ratatui dashboard on a TTY. The TUI is the default
`tui` feature; builds with `--no-default-features` omit it and reject `--tui`. `code-intelligence tui` reopens it for
a configured workspace; `code-intelligence reindex --tui` opens it on the Index tab.

## Tabs

| Tab | Purpose |
|---|---|
| **Sources** | Language checklist and workspace tree. Select languages; mark files/directories Include, Exclude, or Inherit. Sources matched by `.gitignore` render in a dimmed, collapsed, unchecked section at the bottom of the tree; expanding and including one overrides the ignore for that path. |
| **Index** | Index identity, freshness, progress and a live event log. While the TUI owns the terminal, index diagnostics are routed to this panel instead of stderr, so switching back to Sources never leaves log lines on the screen. |
| **Stats** | Corpus, chunk, row and freshness summary. |
| **Usage** | Code-intelligence MCP calls, latency, errors and learn signals from the local journal. |
| **Review** | Config diff and save/apply actions. |

## Keys

| Key | Action |
|---|---|
| `1`–`5` | switch tabs |
| `Tab` | focus the tree or language list |
| `↑`/`↓` | navigate |
| `←`/`→` | collapse/expand a directory |
| `Space` | toggle language, or cycle tree state |
| `i` / `e` / `u` | include / exclude / clear the focused path |
| `/` | filter the tree |
| `r` | start an incremental index |
| `b` | rebuild the index |
| `s` | save `.code-intelligence.toml` |
| `a` | save and start indexing |
| `q` / `Esc` | quit (twice if there are unsaved changes) |
| `?` | help overlay |

## Include and exclude

Tree state is tri-state: **Inherit**, **Include**, **Exclude**. The root is an
implicit include. The deepest explicit path wins; an explicit exclude beats an
include at the same path. `.gitignore` applies to the default walk; an explicit
include can override it. Paths must resolve inside the workspace.

Both lists are stored in `.code-intelligence.toml`:

```toml
schema_version = 3
languages = ["rust", "go"]
retrieval_include = ["docs"]
retrieval_exclude = ["tests/fixtures"]
```

Changing languages, includes, or excludes changes the index identity; run
`code-intelligence reindex` (or press `r` in the TUI).

## Progress, freshness and tool usage

Indexing emits file-level progress events: discover, hash, table, per-file
insert, manifest and done. The Index tab also shows fresh/stale file counts and
the manifest's last write time.

Usage is read from `~/.local/state/code-intelligence/<repo-hash>/journal` and covers
only code-intelligence's own MCP calls; it cannot see host-native tools or other MCP
servers. File events come from a local watcher and never trigger an automatic
reindex — press `r` to pay for embeddings.
