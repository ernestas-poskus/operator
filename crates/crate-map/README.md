# crate-map

Cargo workspace/crate structure detection: parse a manifest, resolve the crate
boundaries around it, walk each crate's module tree, and render a compact,
deterministic **CRATE MAP**.

It exists because raw `Cargo.toml` tells a planning agent almost nothing about a
repository's shape: which crates exist, what they build, how they depend on each
other, and which files are a crate's entry points. The map answers that in a few
lines, and it is a pure function of the bytes on disk — so its output is
byte-stable and safe to hash into a cache key.

## Why parse in-process

`cargo metadata` / `cargo-modules` / `ast-grep` could answer some of it, but
shelling out is environment-dependent and cannot describe a manifest that arrived
over the web. `toml` is enough, so parsing happens here.

## Caller-supplied policy

This crate owns **no** filesystem policy. Workspace containment and the
directory-exclusion set are parameters:

```rust
use crate_map::{load_crate_record, crate_scopes, entry_first_order, manifest_prelude};

fn exclude(entry: &ignore::DirEntry) -> bool {
    entry.file_type().is_some_and(|t| t.is_dir())
        && matches!(entry.file_name().to_string_lossy().as_ref(), "target" | "node_modules")
}

let scopes = crate_scopes(workspace_root, exclude);
let record = load_crate_record(workspace_root, crate_dir, exclude)?;
let modules = entry_first_order(workspace_root, crate_dir, exclude);
let prelude = manifest_prelude(workspace_root, "Cargo.toml", &raw, exclude);
```

The four filesystem-walking entry points take the predicate; every path they
read or render is still resolved for real and refused if it lands outside the
crate (`..` components and symlink escapes included). That division mirrors
[`mentions`](../mentions), which leaves workspace containment to its caller
rather than baking one host's notion of a workspace into the parser.

## Discovery order

All three passes are required, and the third is why a repository with no
`[workspace]` table still maps correctly:

1. declared `[workspace] members` globs, minus `exclude`;
2. path-dependency crates, resolved recursively (`path = "…"` in any dependency
   table);
3. an ignore-aware walk for any remaining manifest that declares a package.

## Determinism

No timestamps, no absolute paths, no run-identifying text. Members and
dependencies are sorted; module lists keep structural order (targets first, then
the `mod` tree, then the rest). Directory walks are sorted, so a bounded walk
truncates at a stable point. Omissions are always announced and the whole block
is capped by `CRATE_MAP_MAX_CHARS`.

## API

| Item | Purpose |
|---|---|
| `parse_manifest` | manifest text → `CrateRecord` (pure) |
| `module_declarations` | `mod x;` / `#[path = "…"]` declarations (pure) |
| `render_crate_map` | `CrateRecord` → the CRATE MAP block (pure) |
| `remote_crate_map`, `annotate_manifest_text` | manifest inside fetched web text (pure) |
| `crate_scopes`, `load_crate_record`, `entry_first_order`, `manifest_prelude` | filesystem-facing, take the `exclude` predicate |
| `scope_of`, `nearest_manifest_dir`, `declares_package`, `no_manifest_message` | scope attribution and query helpers |
