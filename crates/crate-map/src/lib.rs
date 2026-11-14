//! Cargo workspace/crate structure detection — the **CRATE MAP** library.
//!
//! When the context-build path touches a `Cargo.toml` at workspace or crate
//! level (an `@Cargo.toml` mention, a `--preload-files` entry, a crate
//! directory mention, or a manifest that arrives inside a web tool result),
//! the raw TOML tells the planner almost nothing about the repository's shape.
//! This module parses the manifest, resolves the crate boundaries around it,
//! walks each crate's **module tree**, and renders a compact, deterministic
//! structural map — so the planner can name crates, targets and files without
//! spending tool calls (or context) reading manifests and directory listings.
//!
//! # Caller-supplied policy
//!
//! This crate deliberately owns **no** filesystem policy. Workspace containment
//! and the directory-exclusion set are parameters: the four filesystem-walking
//! entry points ([`crate_scopes`], [`load_crate_record`], [`entry_first_order`],
//! [`manifest_prelude`]) take an `exclude` predicate, and the caller passes its
//! own (the `code-intelligence` bin passes `code_search::is_excluded_dir`; the crate's
//! tests pass a local copy). That keeps the crate free of any host's notion of
//! what a workspace is — the same boundary `mentions` draws when it leaves
//! containment to its caller.
//!
//! # Why parse in-process
//!
//! `cargo metadata` / `cargo-modules` / `ast-grep` would answer some of this,
//! but shelling out is environment-dependent, cannot describe a manifest that
//! arrived over the web, and — decisively — the rendered map is part of the
//! round-1 preload block, which is hashed into the run cache key. The map must
//! be a pure function of the bytes on disk. `toml` is already a direct
//! dependency, so parsing costs nothing new.
//!
//! # Discovery order (all three are required)
//!
//! 1. **Declared workspace members** — `[workspace] members` globs minus
//!    `exclude`.
//! 2. **Path-dependency crates** — every `path = "…"` entry in a dependency
//!    table, resolved recursively. This is what finds `crates/mentions` and
//!    `vendor/xberg/crates/xberg` in this very repository, which declares **no**
//!    `[workspace]` table at all: its `crates/*` are path dependencies.
//! 3. **Undeclared crates** — an ignore-aware walk for remaining manifests that
//!    declare a package, marked `undeclared`. A stray or fixture `Cargo.toml`
//!    that declares nothing is not a crate boundary (see [`declares_package`]),
//!    so it cannot silently swallow its directory out of the enclosing crate.
//!
//! # Determinism
//!
//! No timestamps, no absolute paths, no run-identifying text. Everything that
//! is not structurally ordered (members, dependencies) is sorted; module lists
//! keep their structural order (entry points first). Directory walks are sorted
//! (`sort_by_file_path`) so every bounded walk truncates at a stable point and
//! the map cannot depend on the filesystem's readdir order. Omissions are always
//! announced, and the whole block is capped by [`CRATE_MAP_MAX_CHARS`].
//!
//! # Containment
//!
//! Every path this module reads or renders comes from an untrusted source — a
//! manifest can name an absolute `[lib] path`, a `#[path]` attribute can point
//! anywhere, and a directory can be a symlink. So a path is only trusted after
//! two independent checks ([`trusted_crate_file`]): it must be lexically inside
//! the crate with no `..` component, **and** it must really resolve — through
//! `..` and symlinks — to a file inside the crate. [`rel_to`] applies the same
//! `..` refusal to anything it renders, so a workspace-external path can never
//! reach the prompt or the run key.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::path::{Path, PathBuf};

/// The Cargo manifest filename this module detects.
pub const MANIFEST_FILE: &str = "Cargo.toml";

/// Module lines rendered per crate before the list is elided.
const MODULE_CAP: usize = 30;
/// Members / discovered crates rendered per workspace.
const MEMBER_CAP: usize = 40;
/// Dependency names rendered per crate before the list is elided.
const DEP_CAP: usize = 8;
/// Hard ceiling on one rendered map block.
const CRATE_MAP_MAX_CHARS: usize = 8_000;
/// Maximum nesting depth for a member that itself declares `[workspace]`.
const MAX_NESTING: usize = 2;
/// Maximum `Cargo.toml` files the undeclared-crate walk will inspect.
const MAX_WALK_MANIFESTS: usize = 512;
/// Maximum depth of a recursive path-dependency chain.
const MAX_PATH_DEP_DEPTH: usize = 8;
/// Maximum files in one crate's module tree.
const MODULE_TREE_MAX_FILES: usize = 400;
/// Source files larger than this are not scanned for `mod` declarations.
const MAX_SCANNED_SOURCE_BYTES: u64 = 1 << 20;
/// Manifest annotations appended per web tool call.
pub const WEB_MANIFEST_ANNOTATIONS: usize = 3;

// ── Data model ──────────────────────────────────────────────────────────────

/// What kind of manifest a record came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrateKind {
    /// `[workspace]` with no `[package]` — a virtual manifest.
    VirtualWorkspace,
    /// `[package]` with no `[workspace]`.
    Package,
    /// A root package that also declares `[workspace]`.
    PackageWithWorkspace,
}

impl CrateKind {
    /// The label rendered in the map.
    pub fn label(self) -> &'static str {
        match self {
            CrateKind::VirtualWorkspace => "virtual workspace",
            CrateKind::Package => "package",
            CrateKind::PackageWithWorkspace => "package + workspace",
        }
    }
}

/// Whether a target is a library or a binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    Lib,
    Bin,
}

/// One build target (`[lib]` / a `[[bin]]` / a conventional `src/lib.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateTarget {
    pub kind: TargetKind,
    pub name: String,
    /// Workspace-relative source path, when it is known to resolve to a real
    /// file inside the crate.
    pub rel_path: Option<String>,
    /// Whether the manifest declared an explicit `path` for this target — even
    /// one we refused to use (absolute, or outside the workspace). A declared
    /// path suppresses Cargo's default resolution, so it must stay
    /// distinguishable from "no path declared".
    pub has_declared_path: bool,
}

/// One crate's structural facts. `parse_manifest` fills everything except the
/// filesystem-derived fields (`targets` paths, `modules`, resolved `members`,
/// `path_deps`), which `load_crate_record` adds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateRecord {
    /// Package name; `"(workspace)"` for a virtual manifest.
    pub name: String,
    /// Workspace-relative crate directory (`""` at the workspace root).
    pub rel_dir: String,
    /// Workspace-relative manifest path.
    pub rel_manifest: String,
    pub kind: CrateKind,
    pub edition: Option<String>,
    pub resolver: Option<String>,
    /// `[package] autobins`: `None` = Cargo's default (binary auto-discovery is
    /// on unless the manifest declares a `[[bin]]`).
    pub autobins: Option<bool>,
    pub targets: Vec<CrateTarget>,
    /// Dependency names (normal + dev + build + `[workspace.dependencies]`),
    /// deduplicated and sorted.
    pub deps: Vec<String>,
    /// Dependency name → the `path` string exactly as written in the manifest.
    pub path_dep_paths: BTreeMap<String, String>,
    /// Raw `[workspace] members` patterns.
    pub member_patterns: Vec<String>,
    /// Raw `[workspace] exclude` patterns.
    pub exclude_patterns: Vec<String>,
    /// Resolved workspace members (empty unless `load_crate_record` resolved them).
    pub members: Vec<CrateRecord>,
    /// Dependency names whose `path` resolves to a crate under the workspace root.
    pub path_deps: Vec<String>,
    /// Entry-point-first module list (workspace-relative, `.rs` only).
    pub modules: Vec<String>,
    /// Module entries elided by [`MODULE_CAP`].
    pub omitted_modules: usize,
    /// Total modules found before capping.
    pub total_modules: usize,
    /// Found by the walk rather than declared as a member / path dependency.
    pub undeclared: bool,
    /// Human-readable notes (unresolved globs, paths outside the workspace).
    pub notes: Vec<String>,
}

impl CrateRecord {
    /// `""` renders as `.` so a root crate is legible.
    fn dir_display(&self) -> String {
        if self.rel_dir.is_empty() {
            ".".to_string()
        } else {
            self.rel_dir.clone()
        }
    }
}

/// A crate boundary used to attribute a source file to a crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateScope {
    pub name: String,
    /// Workspace-relative crate directory (`""` for a root package).
    pub rel_dir: String,
    /// Declared as a member or reached as a path dependency (vs. found by walk).
    pub declared: bool,
}

/// The chunk-scope label for a source file outside every crate (and the
/// fallback when a workspace has no manifest at all).
pub const NO_CRATE: &str = "(workspace root)";

// ── Manifest parsing (pure) ─────────────────────────────────────────────────

/// The workspace-relative directory a manifest path sits in: `Cargo.toml` →
/// `""`, `crates/foo/Cargo.toml` → `crates/foo`.
pub fn crate_dir_of(rel_manifest: &str) -> String {
    match rel_manifest.strip_suffix(&format!("/{MANIFEST_FILE}")) {
        Some(dir) => dir.to_string(),
        None => String::new(),
    }
}

/// Whether `path` names a Cargo manifest (any depth).
pub fn is_manifest_path(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .is_some_and(|name| name == MANIFEST_FILE)
}

/// Parse one manifest into structural facts. Pure.
///
/// Fails only when the text is not TOML, is not a table, or carries neither
/// `[package]` nor `[workspace]` — callers treat every failure as "not a
/// manifest" and fall back to their existing behavior.
pub fn parse_manifest(text: &str, rel_manifest: &str) -> Result<CrateRecord, String> {
    let doc: toml::Value =
        toml::from_str(text).map_err(|e| format!("{rel_manifest} is not valid TOML: {e}"))?;
    let table = doc
        .as_table()
        .ok_or_else(|| format!("{rel_manifest} is not a TOML table"))?;
    let package = table.get("package").and_then(|v| v.as_table());
    let workspace = table.get("workspace").and_then(|v| v.as_table());
    if package.is_none() && workspace.is_none() {
        return Err(format!(
            "{rel_manifest} has neither a [package] nor a [workspace] table"
        ));
    }

    let kind = match (package.is_some(), workspace.is_some()) {
        (false, true) => CrateKind::VirtualWorkspace,
        (true, true) => CrateKind::PackageWithWorkspace,
        (true, false) => CrateKind::Package,
        // Unreachable: the guard above rejects (false, false).
        (false, false) => CrateKind::VirtualWorkspace,
    };

    let name = package
        .and_then(|p| p.get("name"))
        .and_then(|v| v.as_str())
        .filter(|n| !n.trim().is_empty())
        .unwrap_or("(workspace)")
        .to_string();
    let edition = package
        .and_then(|p| p.get("edition"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let rel_dir = crate_dir_of(rel_manifest);
    // Manifest target paths are crate-relative; every path this module emits is
    // workspace-relative, so they are rebased here once. An absolute path is
    // dropped rather than rebased: it must never reach the model context.
    let workspace_relative = |path: &str| -> Option<String> {
        if Path::new(path).is_absolute() {
            return None;
        }
        if rel_dir.is_empty() {
            Some(path.to_string())
        } else {
            Some(format!("{rel_dir}/{path}"))
        }
    };

    let mut targets = Vec::new();
    if let Some(lib) = table.get("lib").and_then(|v| v.as_table()) {
        targets.push(CrateTarget {
            kind: TargetKind::Lib,
            name: lib
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(name.as_str())
                .replace('-', "_"),
            rel_path: lib
                .get("path")
                .and_then(|v| v.as_str())
                .and_then(&workspace_relative),
            has_declared_path: lib.get("path").is_some(),
        });
    }
    if let Some(bins) = table.get("bin").and_then(|v| v.as_array()) {
        for bin in bins {
            let Some(bin) = bin.as_table() else { continue };
            let Some(bin_name) = bin.get("name").and_then(|v| v.as_str()) else {
                continue;
            };
            targets.push(CrateTarget {
                kind: TargetKind::Bin,
                name: bin_name.to_string(),
                rel_path: bin
                    .get("path")
                    .and_then(|v| v.as_str())
                    .and_then(&workspace_relative),
                has_declared_path: bin.get("path").is_some(),
            });
        }
    }

    let mut deps: BTreeSet<String> = BTreeSet::new();
    let mut path_dep_paths: BTreeMap<String, String> = BTreeMap::new();
    let mut dep_tables: Vec<&toml::Table> = Vec::new();
    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(t) = table.get(key).and_then(|v| v.as_table()) {
            dep_tables.push(t);
        }
    }
    if let Some(t) = workspace
        .and_then(|w| w.get("dependencies"))
        .and_then(|v| v.as_table())
    {
        dep_tables.push(t);
    }
    if let Some(targets_table) = table.get("target").and_then(|v| v.as_table()) {
        for (_, target) in targets_table.iter() {
            let Some(target) = target.as_table() else {
                continue;
            };
            for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
                if let Some(t) = target.get(key).and_then(|v| v.as_table()) {
                    dep_tables.push(t);
                }
            }
        }
    }
    for t in dep_tables {
        for (dep_name, spec) in t.iter() {
            deps.insert(dep_name.clone());
            if let Some(path) = spec
                .as_table()
                .and_then(|s| s.get("path"))
                .and_then(|v| v.as_str())
            {
                path_dep_paths
                    .entry(dep_name.clone())
                    .or_insert_with(|| path.to_string());
            }
        }
    }

    let member_patterns = workspace
        .and_then(|w| w.get("members"))
        .and_then(|v| v.as_array())
        .map(|a| string_array(a))
        .unwrap_or_default();
    let exclude_patterns = workspace
        .and_then(|w| w.get("exclude"))
        .and_then(|v| v.as_array())
        .map(|a| string_array(a))
        .unwrap_or_default();
    let resolver = workspace
        .and_then(|w| w.get("resolver"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let autobins = package
        .and_then(|p| p.get("autobins"))
        .and_then(|v| v.as_bool());

    Ok(CrateRecord {
        name,
        rel_dir,
        rel_manifest: rel_manifest.to_string(),
        kind,
        edition,
        resolver,
        autobins,
        targets,
        deps: deps.into_iter().collect(),
        path_dep_paths,
        member_patterns,
        exclude_patterns,
        members: Vec::new(),
        path_deps: Vec::new(),
        modules: Vec::new(),
        omitted_modules: 0,
        total_modules: 0,
        undeclared: false,
        notes: Vec::new(),
    })
}

/// The string elements of a TOML array, dropping non-strings.
fn string_array(values: &[toml::Value]) -> Vec<String> {
    values
        .iter()
        .filter_map(|v| v.as_str())
        .map(str::to_string)
        .collect()
}

// ── Module declarations (pure) ──────────────────────────────────────────────

/// `mod` declarations in a source text, in declaration order, as
/// `(module name, #[path] override)` pairs.
///
/// Handles `mod x;`, `pub mod x;` and the `pub(crate)` / `pub(super)` /
/// `pub(self)` / `pub(in …)` visibilities; carries a preceding
/// `#[path = "…"]` onto the declaration it annotates; ignores `//` comments and
/// inline `mod x { … }` blocks (which declare no separate file). Pure.
pub fn module_declarations(decl_text: &str) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let mut pending_path: Option<String> = None;
    for raw in decl_text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if line.starts_with("#[") {
            // Attributes stack; only `#[path]` contributes state.
            if let Some(value) = path_attribute_value(line) {
                pending_path = Some(value);
            }
            continue;
        }
        match parse_mod_declaration(line) {
            Some((_name, true)) => {
                // Inline `mod x { … }` — no file, and it consumes the attribute.
                pending_path = None;
            }
            Some((name, false)) => out.push((name, pending_path.take())),
            None => pending_path = None,
        }
    }
    out
}

/// The value of an `#[path = "…"]` attribute line, if that is what it is.
fn path_attribute_value(line: &str) -> Option<String> {
    let rest = line.strip_prefix("#[path")?;
    let open = rest.find('"')?;
    let after = &rest[open + 1..];
    let close = after.find('"')?;
    Some(after[..close].to_string())
}

/// A `mod <ident>;` / `mod <ident> {` declaration line → `(name, is_inline)`.
///
/// Returns `None` for anything else, including a misspelled or non-identifier
/// module name, so a stray line never registers a phantom module.
fn parse_mod_declaration(line: &str) -> Option<(String, bool)> {
    let rest = strip_visibility(line)?;
    let rest = rest.strip_prefix("mod ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() || name.chars().next().is_some_and(|c| c.is_numeric()) {
        return None;
    }
    let tail = rest[name.len()..].trim_start();
    if tail.starts_with(';') {
        Some((name, false))
    } else if tail.starts_with('{') {
        Some((name, true))
    } else {
        // Not a declaration (e.g. `mod x = …`, or a line that merely starts
        // with the word "mod").
        None
    }
}

/// Strip a leading visibility from an item line, keeping the item text.
fn strip_visibility(line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("pub ") {
        return Some(rest);
    }
    if let Some(rest) = line.strip_prefix("pub(") {
        let close = rest.find(')')?;
        return rest[close + 1..].trim_start().into();
    }
    Some(line)
}

// ── Scope attribution (pure) ────────────────────────────────────────────────

/// The crate a workspace-relative path belongs to: the scope with the longest
/// matching directory prefix. Pure.
pub fn scope_of<'a>(scopes: &'a [CrateScope], rel_path: &str) -> Option<&'a CrateScope> {
    scopes
        .iter()
        .filter(|scope| {
            if scope.rel_dir.is_empty() {
                return true;
            }
            rel_path
                .strip_prefix(&scope.rel_dir)
                .is_some_and(|rest| rest.starts_with('/'))
        })
        .max_by_key(|scope| scope.rel_dir.len())
}

// ── Remote manifests (pure, no filesystem) ──────────────────────────────────

/// Detect a Cargo manifest inside remote text and render a manifest-only map.
/// Pure; `None` unless the text parses as TOML **and** carries `[workspace]`,
/// or `[package]` with a non-empty `name`.
pub fn remote_crate_map(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let record = parse_manifest(trimmed, MANIFEST_FILE).ok()?;
    if record.kind == CrateKind::Package && record.name == "(workspace)" {
        // A [package] table without a usable name is not evidence enough.
        return None;
    }
    Some(render_crate_map(&record, true))
}

/// `text` with manifest annotations appended for the whole text and for each
/// ```` ```toml ```` fenced block, consuming from `budget`. The original text is
/// never rewritten, reordered, or truncated — annotations are additive. Pure.
pub fn annotate_manifest_text(text: &str, budget: &mut usize) -> String {
    if *budget == 0 {
        return text.to_string();
    }
    let mut out = text.to_string();
    if let Some(map) = remote_crate_map(text) {
        *budget -= 1;
        return append_annotation(text, &map, "manifest");
    }
    for block in toml_fences(text) {
        if *budget == 0 {
            break;
        }
        let Some(map) = remote_crate_map(block) else {
            continue;
        };
        let annotation = format!("\n// crate map (toml block):\n{map}\n");
        out.push_str(&annotation);
        *budget -= 1;
    }
    out
}

/// Append the annotation to `text` — a separate helper so the "additive only"
/// rule is visible at the call site.
fn append_annotation(text: &str, map: &str, source: &str) -> String {
    let mut out = text.to_string();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!("\n// crate map ({source}):\n{map}\n"));
    out
}

/// The bodies of ```` ```toml ```` fenced blocks, in order.
fn toml_fences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("```") {
        let after_open = &rest[start + 3..];
        let Some(newline) = after_open.find('\n') else {
            break;
        };
        let info = after_open[..newline].trim().to_ascii_lowercase();
        let body_and_rest = &after_open[newline + 1..];
        let Some(end) = body_and_rest.find("```") else {
            break;
        };
        if info == "toml" {
            out.push(&body_and_rest[..end]);
        }
        rest = &body_and_rest[end + 3..];
    }
    out
}

// ── Filesystem discovery ────────────────────────────────────────────────────

/// An ignore-aware walker over `root`, applying the same exclusion set the
/// retrieval indexer uses (so the map and the index agree on what exists).
///
/// `sort_by_file_path` makes the traversal order deterministic. That matters
/// twice: every bounded walk ([`MAX_WALK_MANIFESTS`], [`MODULE_TREE_MAX_FILES`])
/// then truncates at a stable point, and the rendered map — which is hashed
/// into the run cache key — cannot depend on the filesystem's readdir order.
fn walk_builder(root: &Path, exclude: fn(&ignore::DirEntry) -> bool) -> ignore::WalkBuilder {
    let mut builder = ignore::WalkBuilder::new(root);
    builder.add_custom_ignore_filename(".rgignore");
    builder.require_git(false);
    builder.hidden(false);
    builder.sort_by_file_path(|a, b| a.cmp(b));
    builder.filter_entry(move |e| !exclude(e));
    builder
}

fn canonical_root(root: &Path) -> PathBuf {
    root.canonicalize().unwrap_or_else(|_| root.to_path_buf())
}

/// A walker over ONE crate directory: the shared exclusion set, and never a
/// descent into a nested crate — a directory that declares a package of its own
/// is a separate record, so its sources must not pollute this crate's module
/// tree.
fn crate_walk_builder(
    crate_dir: &Path,
    exclude: fn(&ignore::DirEntry) -> bool,
) -> ignore::WalkBuilder {
    let mut builder = ignore::WalkBuilder::new(crate_dir);
    builder.add_custom_ignore_filename(".rgignore");
    builder.require_git(false);
    builder.hidden(false);
    builder.sort_by_file_path(|a, b| a.cmp(b));
    let root = crate_dir.to_path_buf();
    // A single filter: `filter_entry` REPLACES any previously set filter, so
    // both conditions must live in one closure.
    builder.filter_entry(move |entry| {
        if exclude(entry) {
            return false;
        }
        if !entry.file_type().is_some_and(|t| t.is_dir()) {
            return true;
        }
        let path = entry.path();
        path == root || !declares_package(path)
    });
    builder
}

/// Whether `dir` holds a manifest that declares a real crate (a `[package]`, not
/// a virtual workspace manifest). Only such a directory is a crate *boundary*:
/// a stray or fixture `Cargo.toml` that declares no package must not silently
/// delete its subtree from the enclosing crate's module tree — and this is the
/// same rule [`crate_scopes`] applies when it decides what a crate is.
pub fn declares_package(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join(MANIFEST_FILE)) else {
        return false;
    };
    match parse_manifest(&text, MANIFEST_FILE) {
        Ok(record) => record.kind != CrateKind::VirtualWorkspace,
        Err(_) => false,
    }
}

/// Whether `path` lives inside a crate nested within `crate_dir` — a real crate
/// boundary sits between the two, so that crate owns the file.
fn inside_nested_crate(crate_dir: &Path, path: &Path) -> bool {
    let mut current = path.parent();
    while let Some(dir) = current {
        if dir == crate_dir || !dir.starts_with(crate_dir) {
            return false;
        }
        if declares_package(dir) {
            return true;
        }
        current = dir.parent();
    }
    false
}

/// Whether `path` carries a `..` component. Such a path must never be trusted
/// on a lexical prefix check alone — `Path::starts_with` compares components and
/// does NOT resolve `..`, so `/ws/src/../../outside/x.rs` "starts with" `/ws`.
fn has_parent_component(path: &Path) -> bool {
    path.components()
        .any(|c| c == std::path::Component::ParentDir)
}

/// A file this crate's module tree may read, or `None`.
///
/// The gate is **real** resolution, not a lexical prefix test: `candidate` must
/// canonicalize (following `..` and symlinks) to an existing file inside
/// `crate_dir`. Without that, a `#[path = "../../outside/evil.rs"] mod x;` line
/// makes the map read content from outside the workspace and render a
/// plausible-looking path for it into the prompt and the run key.
///
/// The returned path is the **canonical** one. Canonical form is the unique key
/// for a file, which is what keeps the target roots and the directory walk from
/// listing the same file twice when the crate dir and the workspace root reach
/// the caller in different forms (macOS `/var` vs `/private/var`). A legitimate
/// in-crate `#[path = "../gen.rs"]` therefore resolves exactly like the plain
/// path it names, instead of being refused for carrying a `..`.
fn trusted_crate_file(root: &Path, crate_dir: &Path, candidate: &Path) -> Option<PathBuf> {
    let resolved = contain(root, candidate)?;
    (resolved.is_file() && resolved.starts_with(canonical_root(crate_dir))).then_some(resolved)
}

/// Resolve `candidate` inside `root`, returning its canonical path only when it
/// stays within the workspace. Symlink escapes are rejected, like
/// [`ReadLocalFile::contain`].
fn contain(root: &Path, candidate: &Path) -> Option<PathBuf> {
    let canonical = candidate.canonicalize().ok()?;
    canonical
        .starts_with(canonical_root(root))
        .then_some(canonical)
}

/// `abs` relative to `root`, when it is inside.
///
/// Tries the path as given first (the common case: absolute paths built by
/// joining `root`), then against the canonical root — so a caller holding a
/// canonicalized directory (macOS `/var` → `/private/var`) still resolves.
/// A path carrying `..` is refused outright: a lexical prefix check cannot
/// prove such a path is inside the root, and nothing this module renders should
/// ever contain one.
fn rel_to(root: &Path, abs: &Path) -> Option<String> {
    if has_parent_component(abs) {
        return None;
    }
    let rel = abs
        .strip_prefix(root)
        .or_else(|_| abs.strip_prefix(canonical_root(root)))
        .ok()?;
    Some(rel.to_string_lossy().replace('\\', "/"))
}

/// Crate boundaries under `root`, sorted by directory.
///
/// Sources, in order: declared `[workspace] members` (minus `exclude`),
/// recursive path-dependency crates, then an ignore-aware walk for undeclared
/// crates. Every path is containment-checked; nothing outside the workspace is
/// ever read.
pub fn crate_scopes(root: &Path, exclude: fn(&ignore::DirEntry) -> bool) -> Vec<CrateScope> {
    let mut scopes: BTreeMap<String, CrateScope> = BTreeMap::new();

    let root_manifest = root.join(MANIFEST_FILE);
    let root_record = std::fs::read_to_string(&root_manifest)
        .ok()
        .and_then(|text| parse_manifest(&text, MANIFEST_FILE).ok());

    if let Some(record) = &root_record {
        // The root manifest is authoritative about itself: a root *package* is
        // a declared crate (a virtual manifest declares no crate of its own —
        // its members do).
        if record.kind != CrateKind::VirtualWorkspace {
            scopes.insert(
                String::new(),
                CrateScope {
                    name: record.name.clone(),
                    rel_dir: String::new(),
                    declared: true,
                },
            );
        }
        for member in resolve_members(root, record) {
            let rel_dir = rel_to(root, &member).unwrap_or_default();
            scopes.entry(rel_dir.clone()).or_insert(CrateScope {
                name: crate_name_of(&member).unwrap_or_else(|| rel_dir.clone()),
                rel_dir,
                declared: true,
            });
        }
        // Path-dependency crates, recursed (this is how crates/* and
        // vendor/xberg are found in a repo with no [workspace] table).
        let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::new();
        queue.push_back((root.to_path_buf(), 0));
        let mut visited: HashSet<PathBuf> = HashSet::new();
        while let Some((dir, depth)) = queue.pop_front() {
            if depth > MAX_PATH_DEP_DEPTH || !visited.insert(dir.clone()) {
                continue;
            }
            let manifest = dir.join(MANIFEST_FILE);
            let Ok(text) = std::fs::read_to_string(&manifest) else {
                continue;
            };
            let Ok(record) = parse_manifest(&text, MANIFEST_FILE) else {
                continue;
            };
            for (dep_name, dep_path) in &record.path_dep_paths {
                let Some(dep_dir) = contain(root, &dir.join(dep_path)) else {
                    continue;
                };
                if !dep_dir.is_dir() {
                    continue;
                }
                let Some(rel_dir) = rel_to(root, &dep_dir) else {
                    continue;
                };
                if scopes.contains_key(&rel_dir) {
                    continue;
                }
                let name = crate_name_of(&dep_dir).unwrap_or_else(|| dep_name.clone());
                scopes.insert(
                    rel_dir.clone(),
                    CrateScope {
                        name,
                        rel_dir,
                        declared: true,
                    },
                );
                queue.push_back((dep_dir, depth + 1));
            }
        }
    }

    // Undeclared crates: anything with a manifest the declarations missed. A
    // directory the workspace `exclude`s lands here too — it is not a member,
    // but its sources are still indexed, so it needs its own scope rather than
    // being misattributed to the root crate.
    let mut walked = 0usize;
    for entry in walk_builder(root, exclude).build().flatten() {
        if walked >= MAX_WALK_MANIFESTS {
            break;
        }
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        if entry.file_name() != std::ffi::OsStr::new(MANIFEST_FILE) {
            continue;
        }
        let Some(dir) = entry.path().parent() else {
            continue;
        };
        let Some(rel_dir) = rel_to(root, dir) else {
            continue;
        };
        walked += 1;
        if scopes.contains_key(&rel_dir) {
            continue;
        }
        // A manifest that names no package (a virtual workspace) or does not
        // parse declares no crate — it must not become a phantom scope.
        let Some(name) = crate_name_of(dir) else {
            continue;
        };
        scopes.insert(
            rel_dir.clone(),
            CrateScope {
                name,
                rel_dir,
                declared: false,
            },
        );
    }

    scopes.into_values().collect()
}

/// The package name declared in `<dir>/Cargo.toml`. `None` when there is no
/// readable manifest, it does not parse, or it declares no package (a virtual
/// workspace manifest is not a crate).
fn crate_name_of(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(MANIFEST_FILE)).ok()?;
    let record = parse_manifest(&text, MANIFEST_FILE).ok()?;
    (record.kind != CrateKind::VirtualWorkspace).then_some(record.name)
}

/// Expand `[workspace] members` globs minus `exclude`, keeping only
/// containment-checked directories (or manifest files) that exist.
fn resolve_members(root: &Path, record: &CrateRecord) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let excluded: Vec<glob::Pattern> = record
        .exclude_patterns
        .iter()
        .filter_map(|p| glob::Pattern::new(p).ok())
        .collect();
    for pattern in &record.member_patterns {
        let Ok(paths) = glob::glob(&root.join(pattern).to_string_lossy()) else {
            continue;
        };
        for path in paths.flatten() {
            let Some(dir) = contain(root, &path) else {
                continue;
            };
            let dir = if dir.is_file() {
                match dir.parent() {
                    Some(parent) => parent.to_path_buf(),
                    None => continue,
                }
            } else {
                dir
            };
            let rel = rel_to(root, &dir).unwrap_or_default();
            if excluded.iter().any(|p| p.matches(&rel)) {
                continue;
            }
            if dir.join(MANIFEST_FILE).is_file() && seen.insert(dir.clone()) {
                out.push(dir);
            }
        }
    }
    out.sort();
    out
}

/// Load the full record for a crate or workspace directory, including targets,
/// the module tree, resolved members, and member-to-member path dependencies.
/// `None` when there is no readable manifest there.
pub fn load_crate_record(
    root: &Path,
    dir: &Path,
    exclude: fn(&ignore::DirEntry) -> bool,
) -> Option<CrateRecord> {
    load_crate_record_at(root, dir, 0, exclude)
}

fn load_crate_record_at(
    root: &Path,
    dir: &Path,
    depth: usize,
    exclude: fn(&ignore::DirEntry) -> bool,
) -> Option<CrateRecord> {
    let manifest = dir.join(MANIFEST_FILE);
    let rel_manifest = rel_to(root, &manifest)?;
    let text = std::fs::read_to_string(&manifest).ok()?;
    let mut record = parse_manifest(&text, &rel_manifest).ok()?;

    record.targets = discover_targets(root, dir, &record);
    let modules = entry_first_order(root, dir, exclude);
    record.total_modules = modules.len();
    record.omitted_modules = modules.len().saturating_sub(MODULE_CAP);
    record.modules = modules;

    if record.kind != CrateKind::Package {
        let member_dirs = resolve_members(root, &record);
        if member_dirs.is_empty() && !record.member_patterns.is_empty() {
            for pattern in &record.member_patterns {
                record.notes.push(format!("unresolved member: {pattern}"));
            }
        }
        if depth < MAX_NESTING {
            for member_dir in member_dirs.iter().take(MEMBER_CAP) {
                if let Some(member) = load_crate_record_at(root, member_dir, depth + 1, exclude) {
                    record.members.push(member);
                }
            }
        }
        record.members.sort_by(|a, b| a.name.cmp(&b.name));
    }

    // Member-to-member path dependencies (the edge that matters for planning).
    let mut path_deps: BTreeSet<String> = BTreeSet::new();
    for (dep_name, dep_path) in &record.path_dep_paths {
        match contain(root, &dir.join(dep_path)) {
            Some(dep_dir) => {
                if dep_dir.join(MANIFEST_FILE).is_file() {
                    path_deps.insert(dep_name.clone());
                }
            }
            None => record
                .notes
                .push(format!("path dependency outside the workspace: {dep_name}")),
        }
    }
    record.path_deps = path_deps.into_iter().collect();
    Some(record)
}

/// The crate's build targets: its declared ones (with Cargo's own default source
/// path filled in for a target that omits `path`) plus, for a kind the manifest
/// does not pin down, the conventionally discovered ones that exist. Sorted
/// libraries first, then binaries by name.
///
/// The auto-discovery rules mirror Cargo: a target table without `path` defaults
/// to `src/lib.rs` for a lib, `src/main.rs` when a binary shares the package's
/// name, and `src/bin/<name>.rs` otherwise; an explicit `[[bin]]` section (or
/// `autobins = false`) turns binary discovery off, and an explicit `[lib]` turns
/// the conventional `src/lib.rs` off.
fn discover_targets(root: &Path, dir: &Path, record: &CrateRecord) -> Vec<CrateTarget> {
    let declares = |kind: TargetKind| record.targets.iter().any(|t| t.kind == kind);

    // Declared targets, with the default source path resolved for a pathless
    // table. A path that does not resolve to a real file inside this crate — an
    // absolute path, a `..` escape, or a path the workspace does not contain —
    // keeps the target but drops its path: an unresolvable absolute path must
    // never reach the model context, and nothing may be read through it.
    let mut targets: Vec<CrateTarget> = record
        .targets
        .iter()
        .map(|t| {
            // A target that declared a path we refused must NOT fall back to
            // Cargo's default: the manifest pinned it somewhere else.
            let candidate = match t.rel_path.clone() {
                Some(path) => Some(path),
                None if t.has_declared_path => None,
                None => default_target_path(record, t),
            };
            CrateTarget {
                kind: t.kind,
                name: t.name.clone(),
                rel_path: candidate
                    .filter(|p| trusted_crate_file(root, dir, &root.join(p)).is_some()),
                has_declared_path: t.has_declared_path,
            }
        })
        .collect();
    let mut covered: HashSet<String> = targets.iter().filter_map(|t| t.rel_path.clone()).collect();

    let push = |targets: &mut Vec<CrateTarget>, covered: &mut HashSet<String>, t: CrateTarget| {
        if let Some(path) = &t.rel_path
            && !covered.insert(path.clone())
        {
            return;
        }
        targets.push(t);
    };

    // Conventional roots, but only for a kind the manifest leaves to
    // auto-discovery: an explicit `[lib] path = "src/other.rs"` makes
    // `src/lib.rs` a plain file, not a library root — adding it anyway would
    // invent a target Cargo does not build. The same applies to a pathless
    // `[lib] crate-type = […]`, whose resolved default is `src/lib.rs` and so
    // is already covered above.
    if !declares(TargetKind::Lib) {
        let abs = dir.join("src/lib.rs");
        if abs.is_file()
            && let Some(rel) = rel_to(root, &abs)
        {
            push(
                &mut targets,
                &mut covered,
                CrateTarget {
                    kind: TargetKind::Lib,
                    name: record.name.replace('-', "_"),
                    rel_path: Some(rel),
                    has_declared_path: false,
                },
            );
        }
    }
    // An explicit `[[bin]]` section disables binary auto-discovery unless
    // `autobins` says otherwise — Cargo's rule, and the reason `src/main.rs` and
    // `src/bin/*.rs` are not added here.
    let auto_bins = record.autobins.unwrap_or(!declares(TargetKind::Bin));
    if auto_bins {
        let main = dir.join("src/main.rs");
        if main.is_file()
            && let Some(rel) = rel_to(root, &main)
        {
            push(
                &mut targets,
                &mut covered,
                CrateTarget {
                    kind: TargetKind::Bin,
                    name: record.name.clone(),
                    rel_path: Some(rel),
                    has_declared_path: false,
                },
            );
        }
        let mut extra: Vec<CrateTarget> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir.join("src/bin")) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let Some(name) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                    continue;
                };
                if let Some(rel) = rel_to(root, &path) {
                    extra.push(CrateTarget {
                        kind: TargetKind::Bin,
                        name,
                        rel_path: Some(rel),
                        has_declared_path: false,
                    });
                }
            }
        }
        extra.sort_by(|a, b| a.name.cmp(&b.name));
        for t in extra {
            push(&mut targets, &mut covered, t);
        }
    }

    targets.sort_by(|a, b| match (a.kind, b.kind) {
        (TargetKind::Lib, TargetKind::Bin) => std::cmp::Ordering::Less,
        (TargetKind::Bin, TargetKind::Lib) => std::cmp::Ordering::Greater,
        _ => a.name.cmp(&b.name),
    });
    targets
}

/// Cargo's default source path for a target table that omits `path`, as a
/// workspace-relative path. Pure.
fn default_target_path(record: &CrateRecord, target: &CrateTarget) -> Option<String> {
    let join = |rel: &str| {
        if record.rel_dir.is_empty() {
            rel.to_string()
        } else {
            format!("{}/{rel}", record.rel_dir)
        }
    };
    match target.kind {
        TargetKind::Lib => Some(join("src/lib.rs")),
        TargetKind::Bin if target.name == record.name => Some(join("src/main.rs")),
        TargetKind::Bin => Some(join(&format!("src/bin/{}.rs", target.name))),
    }
}

/// Entry-point-first module order for a crate directory: declared targets and
/// conventional roots, then a BFS over `mod` declarations, then every remaining
/// `.rs` file lexicographically. Nested crate directories are excluded — each
/// crate gets its own map entry. Paths are workspace-relative.
pub fn entry_first_order(
    root: &Path,
    crate_dir: &Path,
    exclude: fn(&ignore::DirEntry) -> bool,
) -> Vec<String> {
    // Everything below works in ONE canonical form. `crate_dir` can arrive
    // canonical (a `contain`-gated member dir) while `root` does not (macOS
    // /var vs /private/var); mixing the two forms would defeat the `seen` dedup
    // between the target roots and the walker, listing a file twice.
    let crate_name_dir = canonical_root(crate_dir);
    let manifest = crate_dir.join(MANIFEST_FILE);
    // Parse with the crate's real workspace-relative manifest path: that is what
    // makes `parse_manifest` rebase declared target paths to workspace-relative
    // form, which is how the roots below must be resolved.
    let declared = rel_to(root, &manifest)
        .and_then(|rel| {
            std::fs::read_to_string(&manifest)
                .ok()
                .map(|text| (rel, text))
        })
        .and_then(|(rel, text)| parse_manifest(&text, &rel).ok());

    let mut order: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut queue: VecDeque<PathBuf> = VecDeque::new();
    let mut root_files: HashSet<PathBuf> = HashSet::new();

    let enqueue = |path: PathBuf,
                   order: &mut Vec<PathBuf>,
                   seen: &mut HashSet<PathBuf>,
                   queue: &mut VecDeque<PathBuf>| {
        if path.is_file() && seen.insert(path.clone()) {
            order.push(path.clone());
            queue.push_back(path);
        }
    };

    // Roots: exactly the crate's discovered targets, so the module ordering and
    // the rendered `targets:` line can never disagree about which files are the
    // crate's entry points. A directory with no manifest falls back to the
    // conventional layout.
    let roots: Vec<PathBuf> = match &declared {
        Some(record) => discover_targets(root, &crate_name_dir, record)
            .into_iter()
            .filter_map(|t| t.rel_path)
            .map(|rel| root.join(rel))
            .collect(),
        None => {
            let mut v: Vec<PathBuf> = Vec::new();
            for rel in ["src/lib.rs", "src/main.rs"] {
                if crate_dir.join(rel).is_file() {
                    v.push(crate_dir.join(rel));
                }
            }
            if let Ok(entries) = std::fs::read_dir(crate_dir.join("src/bin")) {
                let mut bins: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs"))
                    .collect();
                bins.sort();
                v.extend(bins);
            }
            v
        }
    };
    for path in roots {
        // A declared `[lib]/[[bin]] path` may name anything, including a path
        // outside the workspace: gate it exactly like a `mod` candidate.
        let Some(path) = trusted_crate_file(root, &crate_name_dir, &path) else {
            continue;
        };
        root_files.insert(path.clone());
        enqueue(path, &mut order, &mut seen, &mut queue);
    }

    // BFS over `mod` declarations.
    while let Some(file) = queue.pop_front() {
        if order.len() >= MODULE_TREE_MAX_FILES {
            break;
        }
        let Some(text) = read_scannable(&file) else {
            continue;
        };
        let base = module_base_dir(&file, &root_files);
        for (name, path_override) in module_declarations(&text) {
            let candidates: Vec<PathBuf> = match path_override {
                Some(rel) => file
                    .parent()
                    .map(|parent| vec![parent.join(rel)])
                    .unwrap_or_default(),
                None => vec![
                    base.join(format!("{name}.rs")),
                    base.join(&name).join("mod.rs"),
                ],
            };
            for candidate in candidates {
                // `#[path]` may point anywhere the file system can reach; the
                // gate resolves it for real and drops anything outside the
                // crate, and a file owned by a nested crate is that crate's.
                let Some(candidate) = trusted_crate_file(root, &crate_name_dir, &candidate) else {
                    continue;
                };
                if inside_nested_crate(&crate_name_dir, &candidate) {
                    continue;
                }
                enqueue(candidate, &mut order, &mut seen, &mut queue);
            }
        }
    }

    // Everything else, lexicographic. `crate_walk_builder` never descends into
    // a nested crate, so these are this crate's own sources only.
    let mut remaining: Vec<PathBuf> = Vec::new();
    for entry in crate_walk_builder(&crate_name_dir, exclude)
        .build()
        .flatten()
    {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        // Canonical form, matching what `trusted_crate_file` seeds `seen` with —
        // a lexical comparison could list one file twice.
        let path = canonical_root(entry.path());
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if path
            .file_name()
            .is_some_and(|n| n == std::ffi::OsStr::new(MANIFEST_FILE))
        {
            continue;
        }
        if !seen.contains(&path) {
            remaining.push(path);
        }
    }
    remaining.sort();
    order.extend(remaining);

    order
        .into_iter()
        .take(MODULE_TREE_MAX_FILES)
        .filter_map(|path| rel_to(root, &path))
        .collect()
}

/// The directory child modules of `file` are resolved against.
///
/// A **crate/module root** — a declared target (`[lib] path`, a `[[bin]] path`),
/// or the conventional `lib.rs`/`main.rs`/`mod.rs` — resolves its children in
/// its own directory (`src/foo.rs` for `src/lib.rs`). Any other module file
/// resolves them inside a directory named after the file (`src/foo/bar.rs` for
/// `mod bar;` in `src/foo.rs`).
fn module_base_dir(file: &Path, root_files: &HashSet<PathBuf>) -> PathBuf {
    let parent = file.parent().unwrap_or(Path::new("."));
    if root_files.contains(file) {
        return parent.to_path_buf();
    }
    let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned());
    match stem.as_deref() {
        Some("lib") | Some("main") | Some("mod") => parent.to_path_buf(),
        Some(stem) => parent.join(stem),
        None => parent.to_path_buf(),
    }
}

/// Read a source file when it is small enough to scan for `mod` declarations.
fn read_scannable(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > MAX_SCANNED_SOURCE_BYTES {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

// ── Rendering ───────────────────────────────────────────────────────────────

/// Render the CRATE MAP block. `remote` = manifest-only (no module tree, no
/// resolved members). Bounded by [`CRATE_MAP_MAX_CHARS`]. Pure.
pub fn render_crate_map(record: &CrateRecord, remote: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("// crate map: {}\n", record.rel_manifest));
    out.push_str(&format!(
        "crate: {} ({}) — {}\n",
        record.name,
        record.dir_display(),
        record.kind.label()
    ));
    if let Some(edition) = &record.edition {
        out.push_str(&format!("edition: {edition}\n"));
    }
    if record.kind != CrateKind::Package {
        match record.resolver.as_deref() {
            Some(resolver) => out.push_str(&format!(
                "workspace: {} — {} member crate(s) (resolver \"{resolver}\")\n",
                record.dir_display(),
                record.members.len().max(record.member_patterns.len())
            )),
            None => out.push_str(&format!(
                "workspace: {} — {} member crate(s)\n",
                record.dir_display(),
                record.members.len().max(record.member_patterns.len())
            )),
        }
    }
    if !record.targets.is_empty() {
        out.push_str(&format!("targets: {}\n", render_targets(&record.targets)));
    }
    if !record.deps.is_empty() {
        out.push_str(&format!(
            "deps ({}): {}\n",
            record.deps.len(),
            render_names(&record.deps, DEP_CAP)
        ));
    }
    if remote {
        out.push_str("modules: unavailable (remote manifest)\n");
    } else {
        out.push_str(&render_modules(&record.modules, record.total_modules));
    }
    if record.kind != CrateKind::Package {
        if remote {
            out.push_str(&format!(
                "members ({} declared): {}\n",
                record.member_patterns.len(),
                render_names(&record.member_patterns, MEMBER_CAP)
            ));
        } else {
            out.push_str(&format!("members ({} resolved):\n", record.members.len()));
            for member in record.members.iter().take(MEMBER_CAP) {
                out.push_str(&format!(
                    "  - {} ({}): targets {}; deps ({}) {}\n",
                    member.name,
                    member.dir_display(),
                    render_targets(&member.targets),
                    member.deps.len(),
                    render_names(&member.deps, DEP_CAP)
                ));
                // (`remote` is impossible here — the enclosing branch already
                // handled it — so members always render their module line.)
                out.push_str("    ");
                out.push_str(&render_modules(&member.modules, member.total_modules));
            }
            let omitted = record.members.len().saturating_sub(MEMBER_CAP);
            if omitted > 0 {
                out.push_str(&format!("  … {omitted} more member crate(s) omitted\n"));
            }
        }
    }
    if !record.path_deps.is_empty() {
        out.push_str(&format!(
            "path deps: {} -> {}\n",
            record.name,
            record.path_deps.join(", ")
        ));
    }
    if !record.notes.is_empty() {
        out.push_str(&format!("notes: {}\n", record.notes.join("; ")));
    }

    if out.chars().count() > CRATE_MAP_MAX_CHARS {
        let head: String = out.chars().take(CRATE_MAP_MAX_CHARS).collect();
        return format!("{head}\n… [crate map truncated at {CRATE_MAP_MAX_CHARS} chars]\n");
    }
    out
}

/// `lib:name -> src/lib.rs; bin:name -> src/main.rs`, or `none declared`.
fn render_targets(targets: &[CrateTarget]) -> String {
    if targets.is_empty() {
        return "none declared".to_string();
    }
    targets
        .iter()
        .map(|t| {
            let kind = match t.kind {
                TargetKind::Lib => "lib",
                TargetKind::Bin => "bin",
            };
            match &t.rel_path {
                Some(path) => format!("{kind}:{} -> {path}", t.name),
                None => format!("{kind}:{}", t.name),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Up to `cap` names, with an elision marker when the list is longer.
fn render_names(names: &[String], cap: usize) -> String {
    if names.is_empty() {
        return "(none)".to_string();
    }
    if names.len() <= cap {
        return names.join(", ");
    }
    format!("{}, … +{} more", names[..cap].join(", "), names.len() - cap)
}

/// The module line: **at most [`MODULE_CAP`] paths**, with the total always
/// stated so the reader knows what was elided. Pure.
fn render_modules(modules: &[String], total: usize) -> String {
    if modules.is_empty() {
        return "modules: (none found)\n".to_string();
    }
    let shown = modules.len().min(MODULE_CAP);
    let omitted = total.saturating_sub(shown);
    let list = modules[..shown].join(", ");
    if omitted > 0 {
        format!("modules ({shown} of {total} shown): {list}, … +{omitted} more\n")
    } else {
        format!("modules ({shown} of {total}): {list}\n")
    }
}

/// The reuse point for `@`-mentions, `preload_block`, and `read_local_file`:
/// `Some(map)` when `rel` names a cargo manifest whose text parses.
pub fn manifest_prelude(
    root: &Path,
    rel: &str,
    raw: &str,
    exclude: fn(&ignore::DirEntry) -> bool,
) -> Option<String> {
    if !is_manifest_path(rel) {
        return None;
    }
    let dir = root.join(crate_dir_of(rel));
    let dir = if dir.is_dir() {
        dir
    } else {
        root.to_path_buf()
    };
    // Prefer the filesystem-derived record (it carries the module tree); fall
    // back to the manifest text alone so a synthetic/staged manifest still gets
    // a structural summary.
    if let Some(record) = load_crate_record(root, &dir, exclude) {
        return Some(render_crate_map(&record, false));
    }
    let record = parse_manifest(raw, rel).ok()?;
    Some(render_crate_map(&record, false))
}

/// The scoped "not a Cargo project here" answer (a message, not an error — the
/// workspace simply may not be a Cargo project).
pub fn no_manifest_message(root: &Path, dir: &Path) -> String {
    // Always workspace-relative: an absolute filesystem path must never reach
    // the model context (`.` when the directory cannot be expressed relatively).
    format!(
        "// crate map: no {MANIFEST_FILE} in this directory or its ancestors ({}) — \
         the workspace may not be a Cargo project\n",
        rel_to(root, dir).unwrap_or_else(|| ".".to_string())
    )
}

/// The nearest directory at or above `dir` that holds a manifest, stopping at
/// the workspace root — how a subdirectory request becomes "the crate this
/// belongs to". Paths are canonicalized first so the walk is not confused by
/// symlinked temporary roots (macOS `/var` → `/private/var`).
pub fn nearest_manifest_dir(root: &Path, dir: &Path) -> Option<PathBuf> {
    let stop = canonical_root(root);
    let mut current = Some(dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()));
    while let Some(candidate) = current {
        // Containment is checked BEFORE the manifest: a directory that
        // canonicalizes outside the workspace must never be reported as a crate
        // dir, whatever it happens to contain.
        if candidate == stop {
            return candidate.join(MANIFEST_FILE).is_file().then_some(candidate);
        }
        if !candidate.starts_with(&stop) {
            return None;
        }
        if candidate.join(MANIFEST_FILE).is_file() {
            return Some(candidate);
        }
        current = candidate.parent().map(Path::to_path_buf);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The caller supplies the exclusion policy; the crate's tests need their
    /// own copy so the suite stays self-contained (`mentions` keeps containment
    /// out of the parser the same way).
    fn test_exclude(entry: &ignore::DirEntry) -> bool {
        entry.file_type().is_some_and(|t| t.is_dir())
            && matches!(
                entry.file_name().to_string_lossy().as_ref(),
                "node_modules" | "target" | "vendor" | "dist" | ".git" | "build"
            )
    }

    const VIRTUAL: &str = r#"
[workspace]
members = ["crates/*"]
exclude = ["crates/legacy"]
resolver = "2"

[workspace.dependencies]
serde = "1"
"#;

    const PACKAGE: &str = r#"
[package]
name = "operator"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "operator"
path = "src/main.rs"

[dependencies]
serde = "1"
mentions = { path = "crates/mentions" }
xberg = { path = "vendor/xberg/crates/xberg" }
"#;

    const PACKAGE_WORKSPACE: &str = r#"
[package]
name = "root"
version = "0.1.0"

[workspace]
members = ["a", "b"]
"#;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    fn write(dir: &Path, rel: &str, text: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(path, text).expect("write");
    }

    // ── parse_manifest ──────────────────────────────────────────────────────

    #[test]
    fn parse_manifest_reads_a_virtual_workspace() {
        let record = parse_manifest(VIRTUAL, "Cargo.toml").expect("parses");
        assert_eq!(record.kind, CrateKind::VirtualWorkspace);
        assert_eq!(record.name, "(workspace)");
        assert_eq!(record.rel_dir, "");
        assert_eq!(record.member_patterns, ["crates/*"]);
        assert_eq!(record.exclude_patterns, ["crates/legacy"]);
        assert_eq!(record.resolver.as_deref(), Some("2"));
        assert_eq!(record.deps, ["serde"]);
    }

    #[test]
    fn parse_manifest_reads_a_package_with_targets_and_path_deps() {
        let record = parse_manifest(PACKAGE, "Cargo.toml").expect("parses");
        assert_eq!(record.kind, CrateKind::Package);
        assert_eq!(record.name, "operator");
        assert_eq!(record.edition.as_deref(), Some("2024"));
        assert_eq!(record.targets.len(), 1);
        assert_eq!(record.targets[0].kind, TargetKind::Bin);
        assert_eq!(record.targets[0].name, "operator");
        assert_eq!(record.targets[0].rel_path.as_deref(), Some("src/main.rs"));
        assert_eq!(
            record.path_dep_paths.get("mentions").map(String::as_str),
            Some("crates/mentions")
        );
        assert!(record.deps.contains(&"xberg".to_string()));
    }

    #[test]
    fn parse_manifest_distinguishes_package_plus_workspace() {
        let record = parse_manifest(PACKAGE_WORKSPACE, "Cargo.toml").expect("parses");
        assert_eq!(record.kind, CrateKind::PackageWithWorkspace);
        assert_eq!(record.name, "root");
        assert_eq!(record.member_patterns, ["a", "b"]);
    }

    #[test]
    fn parse_manifest_rejects_malformed_and_non_manifest_toml() {
        assert!(parse_manifest("this is not = = toml", "Cargo.toml").is_err());
        // Valid TOML, but neither a package nor a workspace: not a manifest we
        // may summarize.
        assert!(parse_manifest("key = \"value\"\n", "Cargo.toml").is_err());
    }

    #[test]
    fn parse_manifest_reads_dashed_crate_dirs() {
        let record = parse_manifest(PACKAGE, "crates/lsp-mcp-embedded/Cargo.toml").expect("parses");
        assert_eq!(record.rel_dir, "crates/lsp-mcp-embedded");
        assert_eq!(crate_dir_of("Cargo.toml"), "");
        assert_eq!(crate_dir_of("a/b/Cargo.toml"), "a/b");
    }

    // ── module_declarations ─────────────────────────────────────────────────

    #[test]
    fn module_declarations_handles_visibilities_attributes_and_inline_blocks() {
        let src = "\
// a comment mentioning mod fake;
use std::fmt;

pub mod alpha;
mod beta;
pub(crate) mod gamma;
pub(super) mod delta;
#[cfg(test)]
mod tests;
#[path = \"weird/location.rs\"]
mod renamed;
mod inline { fn inner() {} }
not_a_mod_line
";
        let decls = module_declarations(src);
        assert_eq!(
            decls,
            vec![
                ("alpha".to_string(), None),
                ("beta".to_string(), None),
                ("gamma".to_string(), None),
                ("delta".to_string(), None),
                ("tests".to_string(), None),
                ("renamed".to_string(), Some("weird/location.rs".to_string())),
            ]
        );
    }

    #[test]
    fn module_declarations_ignores_non_declarations() {
        assert!(module_declarations("mod = 3\n").is_empty());
        assert!(module_declarations("mod 9bad;\n").is_empty());
        assert!(module_declarations("modular thing;\n").is_empty());
    }

    // ── scope_of ────────────────────────────────────────────────────────────

    #[test]
    fn scope_of_picks_the_longest_matching_prefix() {
        let scopes = vec![
            CrateScope {
                name: "root".into(),
                rel_dir: "".into(),
                declared: true,
            },
            CrateScope {
                name: "mentions".into(),
                rel_dir: "crates/mentions".into(),
                declared: true,
            },
            CrateScope {
                name: "xberg".into(),
                rel_dir: "vendor/xberg/crates/xberg".into(),
                declared: true,
            },
        ];
        assert_eq!(scope_of(&scopes, "src/main.rs").unwrap().name, "root");
        assert_eq!(
            scope_of(&scopes, "crates/mentions/src/lib.rs")
                .unwrap()
                .name,
            "mentions"
        );
        assert_eq!(
            scope_of(&scopes, "vendor/xberg/crates/xberg/src/lib.rs")
                .unwrap()
                .name,
            "xberg"
        );
        // A sibling directory that merely shares a prefix is NOT the crate.
        assert_eq!(
            scope_of(&scopes, "crates/mentions-extra/x.rs")
                .unwrap()
                .name,
            "root"
        );
    }

    // ── remote / web annotations ────────────────────────────────────────────

    #[test]
    fn remote_crate_map_accepts_a_manifest_and_rejects_prose() {
        let map = remote_crate_map(PACKAGE).expect("manifest detected");
        assert!(map.contains("crate: operator"), "{map}");
        assert!(
            map.contains("modules: unavailable (remote manifest)"),
            "{map}"
        );

        assert!(remote_crate_map("This page is prose about Cargo workspaces.").is_none());
        assert!(remote_crate_map("<html><body>[package]</body></html>").is_none());
        assert!(remote_crate_map("key = \"value\"\n").is_none());
        assert!(remote_crate_map("").is_none());
    }

    #[test]
    fn annotate_manifest_text_is_additive_and_budgeted() {
        let mut budget = WEB_MANIFEST_ANNOTATIONS;
        let prose = "Notes about the project.\n";
        assert_eq!(annotate_manifest_text(prose, &mut budget), prose);

        let manifest = PACKAGE.to_string();
        let annotated = annotate_manifest_text(&manifest, &mut budget);
        assert!(
            annotated.starts_with(&manifest),
            "original must be preserved"
        );
        assert!(annotated.contains("// crate map (manifest):"));
        assert_eq!(budget, WEB_MANIFEST_ANNOTATIONS - 1);

        // A fenced manifest in a research answer is annotated too.
        let mut budget = WEB_MANIFEST_ANNOTATIONS;
        let answer = format!("Here is the manifest:\n\n```toml\n{PACKAGE}\n```\n");
        let annotated = annotate_manifest_text(&answer, &mut budget);
        assert!(
            annotated.contains("// crate map (toml block):"),
            "{annotated}"
        );

        // An exhausted budget returns the text untouched.
        let mut budget = 0;
        assert_eq!(annotate_manifest_text(&manifest, &mut budget), manifest);
    }

    // ── filesystem discovery ────────────────────────────────────────────────

    /// The shape of THIS repository: a root package with path dependencies to
    /// `crates/*` and no `[workspace]` table at all.
    fn repo_shaped_fixture(dir: &Path) {
        write(dir, "Cargo.toml", PACKAGE);
        write(
            dir,
            "src/main.rs",
            "mod cli;\nmod config;\n\nfn main() {}\n",
        );
        write(dir, "src/cli.rs", "pub fn run() {}\n");
        write(dir, "src/config.rs", "pub struct Config;\n");
        write(
            dir,
            "crates/mentions/Cargo.toml",
            "[package]\nname = \"mentions\"\nversion = \"0.1.0\"\n",
        );
        write(
            dir,
            "crates/mentions/src/lib.rs",
            "mod parse;\npub use parse::*;\n",
        );
        write(dir, "crates/mentions/src/parse.rs", "pub fn parse() {}\n");
        write(
            dir,
            "vendor/xberg/crates/xberg/Cargo.toml",
            "[package]\nname = \"xberg\"\nversion = \"0.1.0\"\n",
        );
        write(
            dir,
            "vendor/xberg/crates/xberg/src/lib.rs",
            "pub fn x() {}\n",
        );
    }

    #[test]
    fn crate_scopes_finds_path_dependency_crates_without_a_workspace_table() {
        let dir = tempdir();
        repo_shaped_fixture(dir.path());
        let scopes = crate_scopes(dir.path(), test_exclude);
        let names: Vec<&str> = scopes.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"operator"), "{names:?}");
        assert!(names.contains(&"mentions"), "{names:?}");
        // Reached through the root manifest's `path = ...`, even though the
        // walker excludes `vendor/`.
        assert!(names.contains(&"xberg"), "{names:?}");
        assert!(scopes.iter().all(|s| s.declared), "{scopes:?}");
    }

    #[test]
    fn crate_scopes_expands_members_and_scopes_excluded_dirs_separately() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", VIRTUAL);
        write(
            dir.path(),
            "crates/one/Cargo.toml",
            "[package]\nname = \"one\"\n",
        );
        write(dir.path(), "crates/one/src/lib.rs", "\n");
        write(
            dir.path(),
            "crates/legacy/Cargo.toml",
            "[package]\nname = \"legacy\"\n",
        );
        write(dir.path(), "crates/legacy/src/lib.rs", "\n");
        let scopes = crate_scopes(dir.path(), test_exclude);
        let one = scopes.iter().find(|s| s.name == "one").expect("member");
        assert!(one.declared);
        // `exclude` removes it from the workspace, but it is still a real crate
        // whose sources are indexed — so it keeps its own scope instead of
        // having its chunks misattributed to a neighbouring crate.
        let legacy = scopes.iter().find(|s| s.name == "legacy").expect("crate");
        assert!(!legacy.declared);
        // A virtual workspace root declares no crate of its own.
        assert!(scopes.iter().all(|s| !s.rel_dir.is_empty()), "{scopes:?}");
    }

    #[test]
    fn crate_scopes_marks_undeclared_crates() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"root\"\n");
        write(dir.path(), "src/lib.rs", "\n");
        write(
            dir.path(),
            "stray/Cargo.toml",
            "[package]\nname = \"stray\"\n",
        );
        write(dir.path(), "stray/src/lib.rs", "\n");
        let scopes = crate_scopes(dir.path(), test_exclude);
        let stray = scopes.iter().find(|s| s.name == "stray").expect("found");
        assert!(!stray.declared);
        assert!(scopes.iter().find(|s| s.name == "root").unwrap().declared);
    }

    #[test]
    fn crate_scopes_ignores_target_and_vendor_walks() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"root\"\n");
        write(dir.path(), "src/lib.rs", "\n");
        write(
            dir.path(),
            "target/debug/inner/Cargo.toml",
            "[package]\nname = \"built\"\n",
        );
        let scopes = crate_scopes(dir.path(), test_exclude);
        assert!(!scopes.iter().any(|s| s.name == "built"), "{scopes:?}");
    }

    #[test]
    fn entry_first_order_excludes_nested_crates_and_their_sources() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"outer\"\n");
        write(dir.path(), "src/main.rs", "mod cli;\n");
        write(dir.path(), "src/cli.rs", "");
        // A nested crate at depth, with its own Cargo.toml.
        write(
            dir.path(),
            "crates/inner/Cargo.toml",
            "[package]\nname = \"inner\"\n",
        );
        write(dir.path(), "crates/inner/src/lib.rs", "mod deep;\n");
        write(dir.path(), "crates/inner/src/deep.rs", "");
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert_eq!(order, ["src/main.rs", "src/cli.rs"], "{order:?}");
        assert!(
            order.iter().all(|p| !p.starts_with("crates/")),
            "a nested crate owns its own sources: {order:?}"
        );
        // The nested crate's own call still resolves its full tree.
        let inner = entry_first_order(dir.path(), &dir.path().join("crates/inner"), test_exclude);
        assert_eq!(
            inner,
            ["crates/inner/src/lib.rs", "crates/inner/src/deep.rs"]
        );
    }

    #[test]
    fn entry_first_order_refuses_paths_that_escape_the_crate() {
        // A `..` in `#[path]` passes a lexical `starts_with` check but resolves
        // outside the workspace: the module tree must never read it (its content
        // would land in the round-1 prompt and the run key).
        let outside = tempdir();
        write(outside.path(), "secret.rs", "pub fn leaked() {}\n");
        let dir = tempdir();
        let root = dir.path().join("repo");
        let secret = outside.path().join("secret.rs");
        write(&root, "Cargo.toml", "[package]\nname = \"app\"\n");
        write(
            &root,
            "src/lib.rs",
            &format!("#[path = \"{}\"]\nmod leaked;\n", secret.display()),
        );
        write(&root, "src/ok.rs", "pub fn ok() {}\n");

        let order = entry_first_order(&root, &root, test_exclude);
        assert_eq!(order, ["src/lib.rs", "src/ok.rs"], "{order:?}");
        assert!(
            !order
                .iter()
                .any(|p| p.contains("secret") || p.contains("..")),
            "nothing outside the workspace may be listed: {order:?}"
        );

        // The same gate guards a relative `..` escape.
        let dir2 = tempdir();
        let root2 = dir2.path().join("ws");
        let sibling = dir2.path().join("outside");
        write(&sibling, "evil.rs", "pub fn evil() {}\n");
        write(&root2, "Cargo.toml", "[package]\nname = \"app\"\n");
        write(
            &root2,
            "src/lib.rs",
            "#[path = \"../../outside/evil.rs\"]\nmod evil;\n",
        );
        let order2 = entry_first_order(&root2, &root2, test_exclude);
        assert_eq!(order2, ["src/lib.rs"], "{order2:?}");
    }

    #[test]
    fn an_in_crate_parent_path_attribute_is_honored() {
        // `#[path = "../generated/…"]` from `src/` stays inside the crate, and is
        // legal Rust: the gate resolves it for real, so it must be kept.
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"app\"\n");
        write(
            dir.path(),
            "src/lib.rs",
            "#[path = \"../generated/bindings.rs\"]\nmod bindings;\n",
        );
        write(
            dir.path(),
            "generated/bindings.rs",
            "pub fn bindings() {}\n",
        );
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert_eq!(order, ["src/lib.rs", "generated/bindings.rs"], "{order:?}");
    }

    #[test]
    fn entry_first_order_refuses_a_symlinked_escape() {
        let outside = tempdir();
        write(outside.path(), "evil.rs", "pub fn evil() {}\n");
        let dir = tempdir();
        let root = dir.path().join("ws");
        write(&root, "Cargo.toml", "[package]\nname = \"app\"\n");
        write(
            &root,
            "src/lib.rs",
            "#[path = \"link/evil.rs\"]\nmod evil;\n",
        );
        // `src/link` is a symlink out of the workspace: the lexical path looks
        // contained, so only real resolution can reject it.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.path(), root.join("src/link")).unwrap();
            let order = entry_first_order(&root, &root, test_exclude);
            assert_eq!(order, ["src/lib.rs"], "{order:?}");
        }
    }

    #[test]
    fn declared_target_paths_outside_the_workspace_are_dropped() {
        // An absolute `[lib] path` is not rebased and not rendered: the target
        // keeps its name, and no absolute path reaches the map.
        let dir = tempdir();
        write(
            dir.path(),
            "Cargo.toml",
            "[package]\nname = \"app\"\n\n[lib]\nname = \"app\"\npath = \"/etc/hosts\"\n",
        );
        write(dir.path(), "src/lib.rs", "");
        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        let rendered = render_crate_map(&record, false);
        assert!(!rendered.contains("/etc/hosts"), "{rendered}");
        assert!(rendered.contains("targets: lib:app"), "{rendered}");
        // The manifest declares the lib, so `src/lib.rs` is a plain file — the
        // conventional root must not be invented as a second target.
        assert_eq!(record.targets.len(), 1, "{record:?}");
        assert!(record.targets[0].rel_path.is_none(), "{record:?}");
        // The file is still part of the crate, just not as a target root.
        assert_eq!(record.modules, ["src/lib.rs"]);
    }

    #[test]
    fn absolute_target_paths_never_reach_the_manifest_parser() {
        let record = parse_manifest(
            "[package]\nname = \"a\"\n\n[[bin]]\nname = \"a\"\npath = \"/abs/bin.rs\"\n",
            "Cargo.toml",
        )
        .expect("parses");
        assert_eq!(record.targets.len(), 1);
        assert_eq!(record.targets[0].name, "a");
        assert!(record.targets[0].rel_path.is_none(), "{record:?}");
    }

    #[test]
    fn rel_to_refuses_paths_carrying_a_parent_component() {
        let root = Path::new("/ws");
        assert_eq!(
            rel_to(root, Path::new("/ws/src/lib.rs")).as_deref(),
            Some("src/lib.rs")
        );
        assert!(rel_to(root, Path::new("/ws/src/../../outside/x.rs")).is_none());
    }

    #[test]
    fn a_fixture_manifest_does_not_prune_its_directory() {
        // A stray `Cargo.toml` that declares no package is not a crate boundary:
        // treating it as one would silently delete its whole `.rs` subtree from
        // the enclosing crate's module tree.
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"app\"\n");
        write(dir.path(), "src/main.rs", "mod fixture;\n");
        write(dir.path(), "src/fixture/mod.rs", "pub fn f() {}\n");
        write(
            dir.path(),
            "src/fixture/Cargo.toml",
            "# a fixture, not a crate\n",
        );
        write(dir.path(), "src/fixture/helper.rs", "pub fn h() {}\n");
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert!(
            order.contains(&"src/fixture/helper.rs".to_string()),
            "{order:?}"
        );

        // A real package boundary still prunes.
        let dir2 = tempdir();
        write(dir2.path(), "Cargo.toml", "[package]\nname = \"app\"\n");
        write(dir2.path(), "src/main.rs", "");
        write(
            dir2.path(),
            "crates/inner/Cargo.toml",
            "[package]\nname = \"inner\"\n",
        );
        write(dir2.path(), "crates/inner/src/lib.rs", "");
        let order2 = entry_first_order(dir2.path(), dir2.path(), test_exclude);
        assert_eq!(order2, ["src/main.rs"], "{order2:?}");
    }

    #[test]
    fn crate_scopes_is_stable_regardless_of_walk_order() {
        // The rendered map feeds the run cache key: two runs over an unchanged
        // tree must produce identical scopes (and therefore identical maps).
        let dir = tempdir();
        repo_shaped_fixture(dir.path());
        let first = crate_scopes(dir.path(), test_exclude);
        let second = crate_scopes(dir.path(), test_exclude);
        assert_eq!(first, second);

        let root = dir.path();
        let record_a = load_crate_record(root, root, test_exclude).expect("record");
        let record_b = load_crate_record(root, root, test_exclude).expect("record");
        assert_eq!(
            render_crate_map(&record_a, false),
            render_crate_map(&record_b, false),
            "the map must be a pure function of the tree"
        );
    }

    #[test]
    fn a_pathless_declared_lib_still_resolves_and_walks_its_module_tree() {
        // `[lib] crate-type = ["rlib"]` with no `path` is the shape ~18 manifests
        // in this repository's vendor tree use. Cargo's default source is
        // `src/lib.rs`, so the target must keep its path AND stay the module
        // tree's root — otherwise the crate silently falls back to the alphabet.
        let dir = tempdir();
        write(
            dir.path(),
            "Cargo.toml",
            "[package]\nname = \"xberg\"\nversion = \"1.0.0\"\n\n[lib]\ncrate-type = [\"rlib\"]\n",
        );
        write(dir.path(), "src/lib.rs", "mod zz;\nmod aa;\n");
        write(dir.path(), "src/zz.rs", "");
        write(dir.path(), "src/aa.rs", "");
        write(dir.path(), "src/aaa_orphan.rs", "");

        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        assert_eq!(record.targets.len(), 1, "{record:?}");
        assert_eq!(
            record.targets[0].rel_path.as_deref(),
            Some("src/lib.rs"),
            "a pathless target resolves to Cargo's default"
        );
        assert!(!record.targets[0].has_declared_path);
        assert_eq!(
            entry_first_order(dir.path(), dir.path(), test_exclude),
            ["src/lib.rs", "src/zz.rs", "src/aa.rs", "src/aaa_orphan.rs"]
        );
    }

    #[test]
    fn a_declared_path_we_refuse_does_not_fall_back_to_the_default() {
        // `[lib] path = "/etc/hosts"` pins the target elsewhere: neither the
        // absolute path nor Cargo's default `src/lib.rs` may become its source.
        let dir = tempdir();
        write(
            dir.path(),
            "Cargo.toml",
            "[package]\nname = \"app\"\n\n[lib]\npath = \"/etc/hosts\"\n",
        );
        write(dir.path(), "src/lib.rs", "");
        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        assert_eq!(record.targets.len(), 1, "{record:?}");
        assert!(record.targets[0].has_declared_path);
        assert!(record.targets[0].rel_path.is_none(), "{record:?}");
        // ...but the file is still part of the crate's module tree.
        assert_eq!(
            entry_first_order(dir.path(), dir.path(), test_exclude),
            ["src/lib.rs"]
        );
    }

    #[test]
    fn autobins_false_removes_the_binary_targets() {
        let dir = tempdir();
        write(
            dir.path(),
            "Cargo.toml",
            "[package]\nname = \"app\"\nautobins = false\n",
        );
        write(dir.path(), "src/main.rs", "");
        write(dir.path(), "src/bin/tool.rs", "");
        write(dir.path(), "src/lib.rs", "");
        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        let names: Vec<&str> = record.targets.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["app"], "only the lib target survives: {record:?}");
        assert!(!record.targets.iter().any(|t| t.kind == TargetKind::Bin));
    }

    #[test]
    fn an_explicit_bin_section_disables_bin_autodiscovery() {
        let dir = tempdir();
        write(
            dir.path(),
            "Cargo.toml",
            "[package]\nname = \"app\"\n\n[[bin]]\nname = \"custom\"\npath = \"src/custom.rs\"\n",
        );
        write(dir.path(), "src/custom.rs", "");
        // Both exist, but the `[[bin]]` section turns discovery off.
        write(dir.path(), "src/main.rs", "");
        write(dir.path(), "src/bin/extra.rs", "");
        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        let bins: Vec<&str> = record
            .targets
            .iter()
            .filter(|t| t.kind == TargetKind::Bin)
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(bins, ["custom"], "{record:?}");
    }

    #[test]
    fn a_member_reached_canonically_is_not_listed_twice() {
        // A `contain`-gated member dir is canonical while the workspace root is
        // not (macOS /var vs /private/var): mixing the two forms used to list a
        // module twice. Regression pin for `entry_first_order`.
        let dir = tempdir();
        write(
            dir.path(),
            "Cargo.toml",
            "[workspace]\nmembers = [\"crates/*\"]\n",
        );
        write(
            dir.path(),
            "crates/one/Cargo.toml",
            "[package]\nname = \"one\"\nversion = \"1.0.0\"\n",
        );
        write(dir.path(), "crates/one/src/lib.rs", "pub fn one() {}\n");
        let member = dir.path().join("crates/one").canonicalize().unwrap();
        let order = entry_first_order(dir.path(), &member, test_exclude);
        assert_eq!(order, ["crates/one/src/lib.rs"], "{order:?}");
    }

    #[test]
    fn inside_nested_crate_only_fires_between_crate_boundaries() {
        let dir = tempdir();
        write(
            dir.path(),
            "stray/Cargo.toml",
            "[package]\nname = \"stray\"\n",
        );
        let root = dir.path();
        assert!(inside_nested_crate(root, &root.join("stray/src/lib.rs")));
        assert!(!inside_nested_crate(root, &root.join("src/lib.rs")));
        // Outside the crate entirely: not this crate's business either way.
        assert!(!inside_nested_crate(
            &root.join("stray"),
            &root.join("other/src/lib.rs")
        ));
    }

    /// The real-repo acceptance test: this repository has NO `[workspace]`
    /// table — its `crates/*` are path dependencies — so a members-only
    /// implementation would report an empty structure. Reaching them requires
    /// the path-dependency pass.
    #[test]
    fn entry_first_order_follows_the_module_tree_not_the_alphabet() {
        let dir = tempdir();
        // `zz_root` is the declared lib target and must come first even though
        // `aa_orphan` sorts before it.
        write(
            dir.path(),
            "Cargo.toml",
            "[package]\nname = \"crate-a\"\n\n[lib]\npath = \"src/zz_root.rs\"\n",
        );
        write(dir.path(), "src/zz_root.rs", "mod zed;\nmod alpha;\n");
        write(dir.path(), "src/zed.rs", "");
        write(dir.path(), "src/alpha.rs", "");
        write(dir.path(), "src/aa_orphan.rs", "");
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert_eq!(
            order,
            [
                "src/zz_root.rs",
                "src/zed.rs",
                "src/alpha.rs",
                "src/aa_orphan.rs"
            ]
        );
    }

    #[test]
    fn entry_first_order_resolves_mod_rs_and_nested_module_dirs() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"c\"\n");
        write(dir.path(), "src/lib.rs", "mod outer;\n");
        write(dir.path(), "src/outer/mod.rs", "mod inner;\n");
        write(dir.path(), "src/outer/inner.rs", "");
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert_eq!(
            order,
            ["src/lib.rs", "src/outer/mod.rs", "src/outer/inner.rs"]
        );
    }

    #[test]
    fn entry_first_order_honors_the_path_attribute() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"c\"\n");
        write(
            dir.path(),
            "src/lib.rs",
            "#[path = \"odd/place.rs\"]\nmod renamed;\n",
        );
        write(dir.path(), "src/odd/place.rs", "");
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert_eq!(order, ["src/lib.rs", "src/odd/place.rs"], "{order:?}");
    }

    #[test]
    fn entry_first_order_uses_a_declared_root_in_a_nested_crate() {
        // The declared path is workspace-relative once the crate lives in a
        // subdirectory: joining it onto the crate dir would silently drop the
        // root and fall back to the alphabet.
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"root\"\n");
        write(dir.path(), "src/lib.rs", "");
        write(
            dir.path(),
            "app/Cargo.toml",
            "[package]\nname = \"app\"\n\n[lib]\npath = \"src/zz_root.rs\"\n",
        );
        write(dir.path(), "app/src/zz_root.rs", "mod zed;\n");
        write(dir.path(), "app/src/zed.rs", "");
        write(dir.path(), "app/src/aa_orphan.rs", "");
        let order = entry_first_order(dir.path(), &dir.path().join("app"), test_exclude);
        assert_eq!(
            order,
            [
                "app/src/zz_root.rs",
                "app/src/zed.rs",
                "app/src/aa_orphan.rs"
            ],
            "the declared root must lead, and the conventional src/lib.rs (absent) must not appear"
        );
    }

    #[test]
    fn a_declared_target_suppresses_the_conventional_root_of_its_kind() {
        let dir = tempdir();
        write(
            dir.path(),
            "Cargo.toml",
            "[package]\nname = \"app\"\n\n[lib]\npath = \"src/other.rs\"\n",
        );
        // Both exist; only the declared one is a library root.
        write(dir.path(), "src/other.rs", "");
        write(dir.path(), "src/lib.rs", "");
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert_eq!(order[0], "src/other.rs", "{order:?}");
        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        assert_eq!(record.targets.len(), 1, "{record:?}");
        assert_eq!(
            record.targets[0].rel_path.as_deref(),
            Some("src/other.rs"),
            "src/lib.rs is a plain file under this manifest"
        );
        // ...but it is still part of the module tree.
        assert!(order.contains(&"src/lib.rs".to_string()), "{order:?}");
    }

    #[test]
    fn entry_first_order_prefers_conventional_roots() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"c\"\n");
        write(dir.path(), "src/main.rs", "");
        write(dir.path(), "src/lib.rs", "");
        write(dir.path(), "src/bin/tool.rs", "");
        write(dir.path(), "src/aaa.rs", "");
        let order = entry_first_order(dir.path(), dir.path(), test_exclude);
        assert_eq!(order[0], "src/lib.rs");
        assert_eq!(order[1], "src/main.rs");
        assert!(order.contains(&"src/bin/tool.rs".to_string()), "{order:?}");
    }

    // ── rendering ───────────────────────────────────────────────────────────

    #[test]
    fn render_is_deterministic_and_states_what_it_omitted() {
        let dir = tempdir();
        repo_shaped_fixture(dir.path());
        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        let first = render_crate_map(&record, false);
        let second = render_crate_map(&record, false);
        assert_eq!(
            first, second,
            "the map feeds the run key — it must be stable"
        );
        assert!(first.starts_with("// crate map: Cargo.toml\n"), "{first}");
        assert!(first.contains("crate: operator (.) — package"), "{first}");
        assert!(
            first.contains("targets: bin:operator -> src/main.rs"),
            "{first}"
        );
        assert!(
            first.contains("path deps: operator -> mentions, xberg"),
            "{first}"
        );
        assert!(first.contains("modules ("), "{first}");
        // No absolute paths leak into the model's context.
        assert!(!first.contains(dir.path().to_str().unwrap()), "{first}");
    }

    #[test]
    fn render_elides_long_lists_with_a_count() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"big\"\n");
        write(dir.path(), "src/lib.rs", "");
        let mut deps: Vec<String> = (0..20).map(|i| format!("dep{i:02}")).collect();
        deps.sort();
        let mut record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        record.deps = deps;
        let rendered = render_crate_map(&record, false);
        assert!(
            rendered.contains(
                "deps (20): dep00, dep01, dep02, dep03, dep04, dep05, dep06, dep07, … +12 more"
            ),
            "{rendered}"
        );
    }

    #[test]
    fn render_caps_the_module_list_not_just_its_label() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"big\"\n");
        write(dir.path(), "src/lib.rs", "");
        let mut record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        let modules: Vec<String> = (0..MODULE_CAP + 10)
            .map(|i| format!("src/module_{i:05}.rs"))
            .collect();
        record.total_modules = modules.len();
        record.modules = modules;
        let rendered = render_crate_map(&record, false);
        let line = rendered
            .lines()
            .find(|l| l.starts_with("modules ("))
            .expect("module line");
        assert_eq!(
            line.matches("src/module_").count(),
            MODULE_CAP,
            "the announced cap must bound the list, not only the label: {line}"
        );
        assert!(
            line.contains(&format!(
                "modules ({MODULE_CAP} of {} shown)",
                MODULE_CAP + 10
            )),
            "{line}"
        );
        assert!(line.ends_with("… +10 more"), "{line}");
    }

    #[test]
    fn render_states_when_no_modules_were_found() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"empty\"\n");
        let record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        let rendered = render_crate_map(&record, false);
        assert!(rendered.contains("modules: (none found)"), "{rendered}");
    }

    #[test]
    fn render_truncates_at_the_char_ceiling() {
        let dir = tempdir();
        write(dir.path(), "Cargo.toml", "[package]\nname = \"big\"\n");
        write(dir.path(), "src/lib.rs", "");
        let mut record = load_crate_record(dir.path(), dir.path(), test_exclude).expect("record");
        // Notes are free-form (unresolved members, escapes): the block ceiling
        // is the last line of defence against a pathological workspace.
        record.notes = (0..500)
            .map(|i| format!("note number {i:04} with some padding text"))
            .collect();
        let rendered = render_crate_map(&record, false);
        assert!(
            rendered.ends_with(&format!(
                "… [crate map truncated at {CRATE_MAP_MAX_CHARS} chars]\n"
            )),
            "{rendered}"
        );
        assert!(rendered.chars().count() < CRATE_MAP_MAX_CHARS + 100);
    }

    // ── prelude + tool ──────────────────────────────────────────────────────

    #[test]
    fn manifest_prelude_only_fires_for_manifests() {
        let dir = tempdir();
        repo_shaped_fixture(dir.path());
        let raw = std::fs::read_to_string(dir.path().join("Cargo.toml")).unwrap();
        assert!(manifest_prelude(dir.path(), "Cargo.toml", &raw, test_exclude).is_some());
        assert!(
            manifest_prelude(dir.path(), "src/main.rs", "fn main() {}", test_exclude).is_none()
        );
        assert!(manifest_prelude(dir.path(), "notes.txt", &raw, test_exclude).is_none());
    }
}
