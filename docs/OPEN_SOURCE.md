# Open-source release notes

`code-intelligence` is released as public source under the MIT license.

## Source snapshot

This repository was created from a clean source archive, not by cloning the
private development history. It contains no private evaluation records, local
runtime state, host MCP registrations, credentials, or caches.

## GitHub release checklist

1. Confirm `git status --short` is clean.
2. Run `make verify`.
3. Confirm `cargo metadata --locked --no-deps` has no path dependency outside
   this repository.
4. Confirm the secret/path audit in the release checklist has no findings.
5. Tag the release and publish the GitHub source archive.

## Publishing to crates.io

The package is named `code-intelligence` with library crate `code_intelligence`
and binary `code-intelligence`.

1. Publish `crates/crate-map` first (`cargo publish` from `crates/crate-map`).
2. Run `cargo publish --dry-run` and review `cargo package --list`.
3. Publish `code-intelligence` (`cargo publish`).

## History

The source repository used for this release contains private development
history. That history is intentionally not part of this repository. Do not add
the old repository as a remote, and do not publish a rewritten history without a
separate review of every removed path and credential.
