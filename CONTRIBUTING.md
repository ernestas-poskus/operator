# Contributing

## Prerequisites

- Rust 1.95 or newer (the package uses edition 2024).
- `cargo-nextest` for the test commands below:
  `cargo install cargo-nextest --locked`.
- A GPU and the fastembed model cache for live semantic-search tests. The
  offline unit suite does not require either.
- No provider API keys, retrieval tokens, or LSP binary are required.

## Build, test, and lint

Run from the repository root:

```bash
cargo check
cargo nextest run --status-level fail
cargo fmt --all
cargo clippy --all-targets
```

CI runs clippy but does not fail on warnings yet: there is an existing lint
backlog. Reducing it is welcome; do not introduce new warnings in code you
touch.

The suite is offline. Tests that need the live model backend are marked
`#[ignore]`; run them only with the GPU and model cache available.

## Workflow

Work in a branch, preferably in a dedicated git worktree, with one branch per
change:

```bash
git worktree add -b <branch> /tmp/operator-<task> main
cd /tmp/operator-<task>
# ... make the change, run the checks above ...
git checkout main
git merge --ff-only <branch>
git worktree remove /tmp/operator-<task>
git branch -d <branch>
```

Merge to `main` only after the checks pass, and run them again on the merged
tree. Do not leave stale worktrees or long-lived branches behind.

## Commit messages

Use `type(component): intent` messages. The description states the intent, not
the mechanics.

- `feat(component):` new feature
- `fix(component):` bug fix
- `enh(component):` enhancement to existing behavior
- `ref(component):` refactor with no behavior change
- `test(component):` test-only change
- `docs:` documentation
- `ci:` workflows and GitHub configuration
- `chore:` maintenance (formatting, imports, dependency bumps)

Components include `cli`, `config`, `languages`, `ast-grep`, `retrieval`,
`mcp`, `learn`, and `deps`. Do not add `Co-authored-by` trailers for tools.

## Pull requests

- Keep the diff focused. One logical change per pull request; avoid drive-by
  reformatting or unrelated refactors.
- Add tests for new behavior and a regression test for a bug fix.
- Run `cargo fmt --all` before opening the pull request; CI checks
  `cargo fmt --all -- --check`.
- Never commit secrets. `.env`, API keys, tokens, and personal config files
  stay out of git; use `.env.example` for the shape of the expected keys.
  Release binaries never embed API keys.
- Describe what changed and list the verification commands you ran.

By contributing you agree that your changes are licensed under the MIT license
(see [LICENSE](LICENSE)).

Agent contributors should also read [AGENTS.md](AGENTS.md) for the verification
workflow and review checklist.
