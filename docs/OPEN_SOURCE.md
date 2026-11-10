# Open-source release notes

`operator` is released as public source under the MIT license. The immediate
distribution target is GitHub; crates.io is deliberately deferred.

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

## crates.io later

The crate name `operator` is already taken on crates.io. A future release needs:

1. A new package name (for example `operator-code-intel`) while keeping the
   binary target named `operator`.
2. Publish `crates/crate-map` separately, or fold it into the package and remove
   the path dependency.
3. Give the dependency a version in addition to its path.
4. Run `cargo publish --dry-run` and review `cargo package --list`.

Do not flip `publish = false` until those steps are complete.

## History

The source repository used for this release contains private development
history. That history is intentionally not part of this repository. Do not add
the old repository as a remote, and do not publish a rewritten history without a
separate review of every removed path and credential.
