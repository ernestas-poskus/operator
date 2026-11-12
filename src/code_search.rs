//! Local hybrid code search — the always-on retrieval backend.
//!
//! A fully-local two-stage retrieval pipeline (no API keys; the dense
//! embeddings and the cross-encoder reranking run in-process on the GPU through
//! fastembed's candle backend — Metal on macOS, CUDA on Linux/Windows):
//!
//! **Ingestion:** [`ignore::WalkBuilder`] discovers the workspace's source
//! files (`.gitignore`-aware), each file is chunked by the selected
//! [`ChunkerMode`] (`xberg` document-aware by default, or `symbol`
//! item-aligned — see [`crate::symbol_chunker`]) at `--retrieval-chunk-lines`
//! lines with a 10-line overlap and a `File: <path>` prefix, embedded locally
//! with mxbai-embed-large-v1 (1024-dim, in-process on the GPU), and
//! stored in an embedded LanceDB table with a Tantivy full-text index on the
//! chunk text. The table name also bakes the embedder's identity (model variant and
//! quantisation and execution device — see [`embedder_identity`]) and the FTS
//! tokenizer (E3/B), so two models, two tokenizations or two providers can never
//! share an index. A
//! sidecar manifest keyed by file-content hash makes rebuilds incremental:
//! unchanged files are never re-embedded. Every chunk also carries the Cargo
//! **crate** its file belongs to (see [`crate::crate_map`]), which is what lets
//! a search be confined to one crate.
//!
//! **Full-text tokenization (E3/B).** The FTS stage is *code-aware*: rather than
//! relying on the engine to split identifiers, every chunk is also stored as a
//! pre-tokenized searchable column ([`FTS_TOKEN_COLUMN`]) holding the
//! identifier sub-tokens of its content, path and crate
//! ([`crate::retrieval::fts_index_text`]), and the query side expands through
//! the same shared tokenizer ([`crate::retrieval::fts_query_text`]). Both sides
//! therefore agree on one vocabulary, and `readRunRecord` ≡ `read_run_record` ≡
//! `ReadRunRecord` all reach every row that carries the identifier. The
//! identifier-aware column is the only shape shipped — the `simple` baseline
//! (the default tokenizer straight over the stored source text) was the pre-E3/B
//! layer and is gone — and it joins the index identity.
//!
//! *Reachability probe (E3/B Task 1):* a caller-supplied custom tokenizer is
//! **not reachable**. `Index::FTS` takes `InvertedIndexParams`, whose
//! `base_tokenizer` names a built-in implementation by *string*; the lancedb-pinned
//! `lance-index` implements `simple`, `whitespace`, `raw` and `ngram` (plus
//! opt-in `jieba/*`/`lindera/*` language-model tokenizers) and rejects the
//! format spec's code tokenizer with `unknown base tokenizer code`
//! (`build_base_tokenizer`). The pre-tokenized column is therefore the shipped
//! shape, and it costs one full re-embed of the workspace — see [`table_name`].
//!
//! **Retrieval:** the query is embedded, the top-K dense neighbors and top-K
//! FTS matches are fetched, fused by a weighted sum of their normalized scores
//! in memory, and the top candidates are reranked by a Qwen3-Reranker-0.6B
//! cross-encoder run in-process by fastembed. The
//! returned chunks carry `path:start_line` anchors so the planner can jump
//! straight into the LSP toolset (`readSourceCode`) for the surrounding references
//! and types.
//!
//! Every model call runs in-process on the GPU through fastembed's candle backend;
//! every failure on this path degrades to "no index" (the planner just runs
//! without `search_vector_store`). Unit tests never talk to a server — the
//! pure chunking/fusion/manifest logic is tested directly and the pipeline is
//! exercised through trait doubles.

use crate::chunk_context::{ChunkEnrichment, context_prefix, enclosing_symbols, module_path_of};
use crate::config::Config;
use crate::types::{
    Blend, CacheDir, CandidateCount, ChunkCount, ChunkId, ChunkIndex, CrateName, DualRoute,
    DurationMs, Embedding, ErrorMessage, FileCount, FileHash, FtsDegraded, FusionGate, IncludePath,
    LanguageLabel, LineNumber, ManifestPath, QueryExpansion, QueryLogPath, QueryRouting, RepoPath,
    RerankContext, RerankEnabled, RowCount, Score, SourceText, TableName, UnixSeconds, Weight,
};
use lancedb::query::{ExecutableQuery, QueryBase};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::console::note;

/// Logic-version stamp for the local pipeline, hashed into the run cache key
/// and the LanceDB table name: bump whenever chunking/search/rerank behavior
/// changes so old indexes and cached rows are never replayed.
///
/// `local-search-6` (E3/B): the FTS stage gained the code-aware
/// [`FTS_TOKEN_COLUMN`], so the indexed text is not what `local-search-5` wrote.
///
/// `local-search-7` (R4): the chunk cap now uses the **measured** 3.05
/// chars/token ([`CHARS_PER_TOKEN`]) instead of the `chars/4` guess, so the
/// stored/embedded text is what the model actually sees (up to 512 real
/// tokens). The stored chunk text changes, so the identity must move.
///
/// `local-search-8`: the embedder moved to mxbai-embed-large-v1 and the char
/// cap was re-measured at 2.7 chars/token, so both the stored vectors and the
/// text change. Every workspace re-embeds once.
///
/// `local-search-9` (WS3): the cap moved from the hand-measured 2.7 to the
/// **2.49 chars/token floor** measured over the longest corpus chunks
/// (min 2.49 / p05 2.76 / median 3.22), so the
/// stored text no longer carries characters the 512-token window discards.
/// The text changes, so the identity moves and every workspace re-embeds once.
pub const LOCAL_SEARCH_LOGIC_VERSION: &str = "local-search-9";

/// The dense embedding width of the shipped embedder
/// (`mxbai-embed-large-v1`). The Arrow schema and the batch guard take their
/// width from here, and every vector returned by the model is validated
/// against it.
pub const DENSE_DIM: crate::types::EmbedDim = crate::types::EmbedDim::new(1024);
/// Overlap lines between adjacent sliding-window chunks.
pub const CHUNK_OVERLAP_LINES: crate::types::ChunkOverlap = crate::types::ChunkOverlap::new(10);
/// Candidates fetched per stage (dense and FTS) before fusion.
pub const STAGE_CANDIDATES: crate::types::CandidateCount = crate::types::CandidateCount::new(50);
/// Candidates passed to the cross-encoder after fusion, and the hard ceiling
/// on final results: the single retained cap is `min(user samples,
/// RERANK_CANDIDATES)` — a caller asking for more than the rerank pool simply
/// gets the pool back, no second independent limit.
///
/// `20` is the shipped cost/quality point. Query-time only —
/// `RERANK_CANDIDATES` is **not** in the index identity, so no workspace
/// re-embeds.
pub const RERANK_CANDIDATES: crate::types::CandidateCount = crate::types::CandidateCount::new(20);
/// Token budget per chunk, matching mxbai-embed-large-v1's 512-token context
/// window — the binding constraint on the **embedder**: the model truncates
/// to this many tokens, so a chunk over it is lossily cut before embedding.
/// Capped via [`chunk_char_budget`], which
/// uses the **measured** characters per token ([`CHARS_PER_TOKEN`]) rather
/// than the historical `chars/4` guess.
pub const EMBEDDER_MAX_TOKENS: usize = 512;
/// Characters per token the mxbai-embed-large-v1 tokenizer produces on this
/// corpus, as the **floor** over the longest chunks.
///
/// Measured through the model tokenizer over the 40 longest chunks:
/// **min 2.49, p05 2.76, median 3.22**. The floor is
/// the value the cap uses, so the stored text is inside the model's 512-token
/// window at the densest chunk as well — a mean would still let the densest
/// chunks lose their tail silently. It is the single ratio the production cap
/// and the measurement probe share.
pub const CHARS_PER_TOKEN: f64 = 2.49;
/// The character cap corresponding to `max_tokens` real tokens, from
/// [`CHARS_PER_TOKEN`]. The single derivation both the production cap
/// ([`cap_tokens`]) and the measurement probe (`crate::chunk_shape`) use, so the
/// probe can never describe a different budget than the pipeline enforces.
pub fn chunk_char_budget(max_tokens: usize) -> usize {
    ((max_tokens as f64) * CHARS_PER_TOKEN).ceil() as usize
}
/// Default token window of the shipped cross-encoder
/// (`--retrieval-rerank-max-tokens`). It is **not** tied to the embedder's 512:
/// the model's own window is 40960, and the shared-cap assumption that
/// justified 512 was measurably wrong — the measured ratio is
/// [`CHARS_PER_TOKEN`] on this corpus, so a "512-token" chunk is really ~591
/// tokens and was cut by **both** models. Query-time only, so it never joins
/// the index identity.
///
/// **512 stays (R10, R1).** The round-9 hypothesis that this cap was starving
/// the reranker was re-measured on the shipped reranker: 8192 against 512 moved
/// reranked recall@5 by +0.0068 — exactly one query — and MRR/nDCG by
/// −0.0034/−0.0009. The premise no longer holds: production chunks are 1294
/// characters and the 512-token budget is 1275 characters, so the cross-encoder
/// already sees all but ~19 characters of every chunk.
pub const DEFAULT_RERANK_MAX_TOKENS: crate::types::TokenLimit = crate::types::TokenLimit::new(512);
/// Embedding batch size.
pub const EMBED_BATCH_SIZE: usize = 32;
/// Chunks one `embed()` call may carry when several files are grouped.
///
/// `EMBED_BATCH_SIZE` bounds one embedding request; grouping consecutive
/// files up to a small multiple of it fills each HTTP call instead of handing
/// it one file's handful of chunks. A file is never split across groups (see
/// [`plan_embed_batches`]).
pub const EMBED_GROUP_CHUNKS: usize = EMBED_BATCH_SIZE * 4;
/// The incumbent chunker label baked into the index identity.
///
/// The round-5 A/B chose xberg (2/3 approvals vs 0/3), and the sliding-window
/// implementation survives only as xberg's internal failure fallback. The label
/// stays a named constant (and is the label of ChunkerMode::Xberg) so every
/// existing table, manifest and run key keeps its exact name and no workspace
/// re-embeds.
pub const CHUNKER_LABEL: &str = "xberg";

/// The selectable chunker (R8). Xberg is the shipped default; Symbol is the
/// code-aware Rust item splitter ([symbol_chunker](crate::symbol_chunker)),
/// selectable only by the measurement harness until it clears the R8 decision
/// rule. It changes what text is embedded, so it is keyed **and** joins the
/// index identity: the two modes never share a table or a manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkerMode {
    /// The xberg document-aware splitter (the incumbent).
    #[default]
    Xberg,
    /// Symbol/function-aligned Rust item chunks.
    ///
    /// Optional symbol/function-aligned chunks. Not the shipped default;
    /// `Xberg` remains the default because symbol chunking did not show a
    /// material retrieval gain.
    Symbol,
}

impl ChunkerMode {
    /// The identity label.
    pub fn label(self) -> &'static str {
        match self {
            ChunkerMode::Xberg => CHUNKER_LABEL,
            ChunkerMode::Symbol => "symbol",
        }
    }

    /// Parse a chunker value. An unknown value names the accepted set.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "xberg" => Ok(ChunkerMode::Xberg),
            "symbol" => Ok(ChunkerMode::Symbol),
            other => Err(format!(
                "invalid chunker '{other}' (expected xberg | symbol)"
            )),
        }
    }
}

impl std::str::FromStr for ChunkerMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ChunkerMode::parse(s)
    }
}

/// The FTS tokenizer label baked into the index identity (E3/B); `code` is the
/// only mode now. Same rationale as [`CHUNKER_LABEL`].
pub const FTS_TOKENIZER_LABEL: &str = "code";

/// Default dense embedder Hugging Face repo, loaded in-process by fastembed's
/// candle BERT backend (CLS + L2, 1024-d, 512-token context).
pub const DEFAULT_EMBED_MODEL: &str = "mixedbread-ai/mxbai-embed-large-v1";
/// Default cross-encoder reranker Hugging Face repo, loaded in-process by
/// fastembed's candle Qwen3 reranker. Its `score` is P("yes") — already in
/// `[0, 1]`, which is what [`RerankScoreNorm`] expects.
///
/// `Qwen3-Reranker-0.6B` is the shipped reranker; it is query-time only, so
/// changing it never changes the index identity.
pub const DEFAULT_RERANK_MODEL: &str = "Qwen/Qwen3-Reranker-0.6B";
/// Default reranker task instruction for code search. It is what the original
/// backend used, so porting the backend does not silently change the
/// query-side instruction.
pub const DEFAULT_RERANK_INSTRUCTION: &str = "Given a code search query, retrieve relevant code";

/// Stable label of the shipped dense embedding model.
///
/// It is the default the config resolves to when nothing overrides it. The
/// eval record names the **configured** model
/// (`retrieval_eval::index_identity`), so this constant is the shipped-default
/// reference the tests and docs pin rather than a value the writer stamps.
#[allow(dead_code)]
pub const DENSE_MODEL_LABEL: &str = DEFAULT_EMBED_MODEL;

/// Stable label of the shipped cross-encoder reranker.
/// Same discipline as [`DENSE_MODEL_LABEL`].
#[allow(dead_code)]
pub const RERANKER_MODEL_LABEL: &str = DEFAULT_RERANK_MODEL;

/// The local backend indexes every source file ast-grep recognizes; the
/// workspace's selected language set is applied by [`selected_source`].
///
/// The searchable full-text column (E3/B): the code-aware expansion of a
/// chunk's content, path and crate ([`crate::retrieval::fts_index_text`]).
///
/// The column stores identifier sub-tokens rather than source text, so the
/// FTS index can be built with LanceDB's *default* tokenizer and still match
/// `readRunRecord` ≡ `read_run_record` ≡ `ReadRunRecord`. The stored
/// `content`/`file_path` columns are untouched: the reranker and the in-memory
/// lexical layer keep reading the original source.
pub const FTS_TOKEN_COLUMN: &str = "fts_tokens";

/// The searchable text of one chunk — content, path and crate, expanded
/// through the shared code-aware tokenizer. Pure; never returns `None`, so the
/// column is populated in every row no matter which tokenizer mode built the
/// table.
pub fn fts_text_of(chunk: &LocalChunk) -> String {
    crate::retrieval::fts_index_text(
        chunk.content.as_str(),
        chunk.path.as_str(),
        chunk.crate_name.as_str(),
    )
}

/// The FTS query text (E3/B): the identifier expansion, so the indexed
/// `fts_tokens` column and the query share one vocabulary. Pure.
pub fn fts_query_for(query: &str) -> String {
    crate::retrieval::fts_query_text(query)
}

/// Char cap on one document's converted Markdown before chunking. A huge PDF
/// must not flood the corpus with hundreds of thousands of chunks.
pub const DOCUMENT_INDEX_MAX_CHARS: usize = 2_000_000;

/// One chunk stored in LanceDB (the spec's data model plus `start_line`).
#[derive(Clone, Debug, PartialEq)]
pub struct LocalChunk {
    pub id: ChunkId,
    pub path: RepoPath,
    pub chunk_index: ChunkIndex,
    /// 1-based line the chunk starts at in its file (LSP tool anchor).
    pub start_line: LineNumber,
    pub content: SourceText,
    pub language: LanguageLabel,
    pub file_hash: FileHash,
    /// Cargo crate this chunk's file belongs to (package name, or
    /// [`crate_map::NO_CRATE`] outside every crate). The LanceDB filter
    /// column behind `search_vector_store`'s `crate_name` argument.
    pub crate_name: CrateName,
    pub vector: Embedding,
}

// ── Pure ingestion helpers ──────────────────────────────────────────────────

/// Discover workspace files under `root` that satisfy `keep`, honoring
/// `.gitignore` rules and skipping hidden/heavy directories. Pure over the
/// filesystem (no models).
fn discover_files(root: &Path, keep: impl Fn(&Path) -> bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in workspace_walker(root).flatten() {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        if keep(path) {
            out.push(path.to_path_buf());
        }
    }
    out.sort();
    out
}

/// The one walker every corpus discovery uses: `.gitignore`-aware, hidden
/// entries visible (so their rules apply), and the always-excluded trees
/// filtered by name at any depth.
fn workspace_walker(root: &Path) -> ignore::Walk {
    ignore::WalkBuilder::new(root)
        // Don't skip hidden entries: `.gitignore` files must be visible so the
        // walker applies their rules (gitignore matching itself is on).
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true)
        // Apply .gitignore rules even when the workspace is not a git repo
        // (default `require_git` only honors them inside repositories).
        .require_git(false)
        .filter_entry(|e| !is_excluded_dir(e))
        .build()
}

/// Whether `path` is a workspace source file the corpus indexes, by
/// extension. The `serve` watcher uses it to ignore irrelevant churn. Pure.
pub(crate) fn is_indexable_source_path(path: &Path) -> bool {
    crate::languages::from_path(path).is_some()
}

/// Whether `path` belongs to one of the configured languages.
fn selected_source(cfg: &Config, path: &Path) -> bool {
    let Some(lang) = crate::languages::from_path(path) else {
        return false;
    };
    let id = crate::languages::canonical_id(lang);
    cfg.languages.iter().any(|selected| selected.as_str() == id)
}

/// Discover every ast-grep-recognized source file. Callers apply the selected
/// language set when they need the workspace's corpus.
pub fn discover_source_files(root: &Path) -> Vec<PathBuf> {
    discover_files(root, is_indexable_source_path)
}

/// Discover workspace documents the in-process converter can turn into
/// Markdown (PDF/office). Same walk/exclusions as sources.
pub fn discover_document_files(root: &Path) -> Vec<PathBuf> {
    discover_files(root, crate::document::is_document_path)
}

// ── Explicit corpus includes (`--include` / `retrieval_include`) ─────────────
//
// The default corpus is a *policy* (`.rs` sources, plus converted documents
// when the workspace has any); an include is *intent*: the user named this
// folder, so everything indexable under it joins the same table. The set is
// part of the index identity (a different include set is a different corpus,
// hence a different table and manifest), and `reindex` records it in the
// workspace `.operator.toml` so a later plan resolves the same identity.

/// The largest file an explicitly included tree contributes, in bytes.
///
/// The default corpus is filtered by extension before anything is read, so a
/// stray binary is never opened; an included tree is not, so this bounds the
/// cost of reading every file it contains. A file named *directly* as an
/// include bypasses the cap: a direct path is unambiguous intent.
pub const INCLUDE_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// The include set as one canonical, order-independent list: whitespace-trimmed,
/// empty entries dropped, a trailing `/` and a leading `./` stripped; sorted and
/// deduplicated. Pure, so the same set always yields one identity.
///
/// Path *aliases* (`a/../b` versus `b`) are deliberately not unified: resolving
/// them needs the filesystem, and the identity must be computable from config
/// alone. A workspace that spells one folder two ways builds two tables, which
/// is the same cost as any other identity change and never mixes two corpora.
pub fn normalized_includes(raw: &[impl AsRef<str>]) -> Vec<String> {
    let mut out: Vec<String> = raw
        .iter()
        .map(|entry| {
            let trimmed = entry.as_ref().trim();
            let trimmed = trimmed.strip_prefix("./").unwrap_or(trimmed);
            trimmed.trim_end_matches('/').to_string()
        })
        .filter(|entry| !entry.is_empty())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The include set resolved against the workspace root, for walking. Total:
/// a missing or escaping entry simply yields no files. `validate_includes` is
/// the strict twin `reindex` uses before it records anything.
pub fn resolve_includes(workspace_root: &Path, raw: &[impl AsRef<str>]) -> Vec<PathBuf> {
    normalized_includes(raw)
        .iter()
        .map(|entry| {
            let path = Path::new(entry);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                workspace_root.join(path)
            }
        })
        .collect()
}

/// Validate an include set for the maintenance command: it must name at least
/// one path, every path must exist, and every path must canonicalize to
/// somewhere **inside** the workspace.
///
/// Staying inside the workspace is load-bearing rather than cosmetic: manifest
/// keys stay workspace-relative, crate-scope attribution keeps working, and the
/// `serve` watcher's single recursive root still covers every indexed file.
#[allow(dead_code)]
pub fn validate_includes(
    workspace_root: &Path,
    raw: &[impl AsRef<str>],
) -> Result<Vec<PathBuf>, String> {
    let normalized = normalized_includes(raw);
    if normalized.is_empty() {
        return Err(
            "--include needs at least one non-empty path, e.g. --include docs/guides".to_string(),
        );
    }
    let root = workspace_root
        .canonicalize()
        .map_err(|e| format!("resolving the workspace {}: {e}", workspace_root.display()))?;
    let mut resolved = Vec::with_capacity(normalized.len());
    for (entry, path) in normalized.iter().zip(resolve_includes(workspace_root, raw)) {
        let canonical = path.canonicalize().map_err(|e| {
            format!(
                "--include '{entry}': {} is not readable: {e}",
                path.display()
            )
        })?;
        if !canonical.starts_with(&root) {
            return Err(format!(
                "--include '{entry}': {} is outside the workspace {}",
                canonical.display(),
                root.display()
            ));
        }
        resolved.push(canonical);
    }
    Ok(resolved)
}

/// The identity fragment an include set contributes: `""` for the empty set —
/// so a workspace that never used the knob keeps its table and manifest names
/// byte-for-byte — else `_inc<sha256[:8]>` of the canonical set. Pure.
pub fn include_identity_fragment(raw: &[impl AsRef<str>]) -> String {
    let normalized = normalized_includes(raw);
    if normalized.is_empty() {
        return String::new();
    }
    let mut h = Sha256::new();
    h.update(normalized.join("\n").as_bytes());
    format!("_inc{}", &crate::hash::hex_digest(h.finalize())[..8])
}

/// Discover every file an explicitly included path contributes.
///
/// A directly named file is taken unconditionally (explicit intent outranks the
/// size cap and the hidden-file rule). A named directory
/// is walked with the one [`workspace_walker`], so it obeys the same ignore
/// files and excluded-tree policy as the default corpus, and everything up to
/// [`INCLUDE_MAX_FILE_BYTES`] is offered; the single build loop then converts
/// documents and skips whatever does not decode as UTF-8, which is what makes
/// the policy "documents plus text" without an extension allowlist to drift.
///
/// Hidden **files** are skipped: a `.gitignore`/`.DS_Store` inside an included
/// docs tree is tree metadata, not corpus text. Hidden *directories* named as
/// the include root are still honored (the walk root is never filtered), so
/// `--include .notes` reads the notes.
pub fn discover_include_files(root: &Path) -> Vec<PathBuf> {
    if root.is_file() {
        return vec![root.to_path_buf()];
    }
    discover_files(root, |path| {
        let hidden = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with('.'));
        !hidden
            && std::fs::metadata(path)
                .map(|meta| meta.is_file() && meta.len() <= INCLUDE_MAX_FILE_BYTES)
                .unwrap_or(false)
    })
}

/// Directories that are **never** walked, whatever the ignore files say:
/// dependency, vendored-dependency, and build-output trees. They are enormous,
/// regenerated or third-party, and never useful as bulk context — and a
/// `.gitignore` that anchors one of them (`/node_modules`, which does not cover
/// `frontend/node_modules`) or is missing entirely would otherwise let a single
/// `@dir/` mention pull thousands of files into the round-1 prompt.
///
/// Matching is **by name at any depth**, so a nested occurrence
/// (`frontend/node_modules`, `crates/foo/target`, `packages/x/dist`) is skipped
/// exactly like a top-level one. The `@`-mention path additionally refuses to
/// expand such a tree even when it is named as the walk *root*, because the
/// `ignore` crate never filters depth 0 — see `ignore_walker` in `references.rs`.
/// An explicitly named **file** inside one of them is still read: a direct path
/// is unambiguous intent, unlike a directory expansion.
pub(crate) const ALWAYS_EXCLUDED_DIRS: &[&str] = &["node_modules", "target", "vendor", "dist"];

/// Whether a directory *name* is one of the always-excluded trees. Shared by
/// every walker so the policy cannot drift between them.
pub(crate) fn is_always_excluded_dir(name: &str) -> bool {
    ALWAYS_EXCLUDED_DIRS.contains(&name)
}

/// Directories (at any depth) a walker must never descend into. Shared by the
/// retrieval index, `crate_map`, and the `@`-mention expansion, so all three
/// agree on what exists in the workspace.
///
/// There are two tiers. [`ALWAYS_EXCLUDED_DIRS`] is refused everywhere, including
/// when it is the walk root the user named. This wider list (the dot-directories
/// and `build`) is refused as a *nested* entry — bulk expansion never descends
/// into it — but an explicit mention of such a directory as its own root is still
/// honored.
pub(crate) fn is_excluded_dir(entry: &ignore::DirEntry) -> bool {
    if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
        return false;
    }
    is_excluded_dir_name(&entry.file_name().to_string_lossy())
}

/// Whether a directory *name* is one of the excluded trees (either tier).
///
/// Crate-visible because every walker must agree on what exists in the
/// workspace: the retrieval index, `crate_map`, the `@`-mention expansion, and
/// the audit's grounding scan. A walker with its own list is a walker that will
/// drift — the audit had one, and it skipped neither `vendor` nor `.cargo-home`
/// (measured: 46 s per artifact to scan 745 MiB of dependency sources).
pub(crate) fn is_excluded_dir_name(name: &str) -> bool {
    is_always_excluded_dir(name)
        || matches!(
            name,
            ".git" | ".cache" | ".cargo-home" | ".spike-target" | ".lspmcp-target" | "build"
        )
}

/// Whether `path` lies inside a tree a filesystem watcher must ignore: the
/// retrieval walker's policy, plus `operator serve`'s own `.operator/` marker
/// directory (so writing the runtime marker never triggers a reindex).
///
/// Path-based because `notify` hands back raw paths, not walker entries. The
/// caller additionally skips the resolved retrieval cache dir, which may live
/// outside the workspace. Pure.
#[allow(dead_code)]
pub(crate) fn is_excluded_tree_path(path: &Path) -> bool {
    path.components().any(|c| match c {
        std::path::Component::Normal(name) => {
            let name = name.to_string_lossy();
            is_excluded_dir_name(&name) || name == ".operator"
        }
        _ => false,
    })
}

/// SHA-256 hex of `text` (content addressing for incremental ingestion).
pub fn file_hash_of(text: &str) -> FileHash {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    FileHash::new(crate::hash::hex_digest(h.finalize()))
}

/// Stable chunk id: hash of (path, chunk index, file content hash).
pub fn chunk_id(path: &str, chunk_index: u32, file_hash: &str) -> ChunkId {
    let mut h = Sha256::new();
    h.update(path.as_bytes());
    h.update([0u8]);
    h.update(chunk_index.to_le_bytes());
    h.update([0u8]);
    h.update(file_hash.as_bytes());
    ChunkId::new(crate::hash::hex_digest(h.finalize()))
}

/// The marker [`cap_tokens`] appends when it cuts a chunk at the measured
/// character budget. Public so a measurement can detect the explicit cut
/// without duplicating the literal (see `crate::chunk_shape`).
pub const CHUNK_TRUNCATION_MARKER: &str = "… [chunk truncated]";

/// Cap `text` to `max_tokens` tokens at the measured [`CHARS_PER_TOKEN`] ratio,
/// character-safe and tagging the cut. `max_tokens` is the model context
/// budget, not a raw char count.
fn cap_tokens(text: &str, max_tokens: usize) -> String {
    let max_chars = chunk_char_budget(max_tokens);
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let head: String = text.chars().take(max_chars).collect();
        format!("{head}{CHUNK_TRUNCATION_MARKER}")
    }
}

/// Cap one document's converted Markdown to DOCUMENT_INDEX_MAX_CHARS chars
/// (char-safe; a truncated tail simply produces no further chunks).
fn cap_document_text(markdown: String) -> String {
    if markdown.chars().count() <= DOCUMENT_INDEX_MAX_CHARS {
        return markdown;
    }
    markdown.chars().take(DOCUMENT_INDEX_MAX_CHARS).collect()
}

/// Language label from a file extension (the stored `language` column), or
/// `text` for an extensionless file.
fn language_of(path: &str) -> LanguageLabel {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("text")
        .into()
}

/// The derived-context prefix one chunk carries, from its own first line.
///
/// The symbol chain is resolved at the chunk's **start line**, which every
/// chunker stamps for every chunk regardless of how the window was cut — so
/// one pure derivation serves the sliding window, xberg and the ablation's cost
/// measurement identically. Pure.
pub(crate) fn chunk_prefix(
    enrichment: ChunkEnrichment,
    path: &str,
    module_path: &str,
    text: &str,
    start_line: u32,
) -> String {
    if enrichment.is_bare() {
        return String::new();
    }
    let symbols = if enrichment.symbols.is_on() {
        enclosing_symbols(text, start_line)
    } else {
        Vec::new()
    };
    context_prefix(enrichment, path, module_path, &symbols)
}

/// Discovery helper for the ablation's cost measurement: every discovered
/// source file with the crate name and crate root the module-path derivation
/// needs, from **one** crate-map walk (recomputing the scopes per file would
/// re-walk the workspace 250 times). Pure apart from the reads
/// [`discover_source_files`] already does.
///
/// Test-gated: only the `#[cfg(test)]` ablation instrument measures prefix
/// cost, so the shipped binary carries no unused path.
#[cfg(test)]
#[allow(dead_code)]
pub fn source_scopes(cfg: &Config) -> Vec<(String, String, String)> {
    let scopes = crate_map::crate_scopes(&cfg.workspace_root, is_excluded_dir);
    discover_source_files(&cfg.workspace_root)
        .into_iter()
        .map(|path| {
            let rel = path
                .strip_prefix(&cfg.workspace_root)
                .map(|rel| rel.to_string_lossy().into_owned())
                .unwrap_or_else(|_| path.to_string_lossy().into_owned());
            let (name, dir) = crate_map::scope_of(&scopes, &rel)
                .map(|scope| (scope.name.clone(), scope.rel_dir.clone()))
                .unwrap_or_else(|| (crate_map::NO_CRATE.to_string(), String::new()));
            (rel, name, dir)
        })
        .collect()
}

#[cfg(test)]
/// Sliding-window chunking: `lines_per_chunk`-line windows with a
/// [`CHUNK_OVERLAP_LINES`] overlap, each prefixed with `File: <path>` so the
/// embedding model sees the location, and capped to [`EMBEDDER_MAX_TOKENS`]
/// *estimated* tokens. Vectors start empty (filled by the embed step). Pure.
///
/// Kept as the incumbent-shape entry point; [`sliding_window_chunks_with`] is
/// the same windowing under any [`ChunkEnrichment`].
pub fn sliding_window_chunks(
    path: &str,
    text: &str,
    file_hash: &str,
    lines_per_chunk: usize,
) -> Vec<LocalChunk> {
    sliding_window_chunks_with(
        path,
        text,
        file_hash,
        lines_per_chunk,
        CHUNK_OVERLAP_LINES.get(),
        ChunkEnrichment::PATH,
        "",
    )
}

/// [`sliding_window_chunks`] with an explicit context field set and module
/// path. Pure.
pub fn sliding_window_chunks_with(
    path: &str,
    text: &str,
    file_hash: &str,
    lines_per_chunk: usize,
    overlap_lines: usize,
    enrichment: ChunkEnrichment,
    module_path: &str,
) -> Vec<LocalChunk> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let overlap = overlap_lines.min(lines_per_chunk.saturating_sub(1));
    let step = lines_per_chunk.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut index = 0u32;
    while start < lines.len() {
        let end = (start + lines_per_chunk).min(lines.len());
        let window = lines[start..end].join("\n");
        let start_line = (start + 1) as u32;
        let prefix = chunk_prefix(enrichment, path, module_path, text, start_line);
        let content = cap_tokens(&format!("{prefix}{window}"), EMBEDDER_MAX_TOKENS);
        chunks.push(LocalChunk {
            id: chunk_id(path, index, file_hash),
            path: RepoPath::from(path),
            chunk_index: ChunkIndex::new(index),
            start_line: LineNumber::new(start_line),
            content: SourceText::new(content),
            language: language_of(path),
            file_hash: FileHash::from(file_hash),
            // Chunking is crate-agnostic; `build` stamps the crate once the
            // workspace's crate boundaries are known.
            crate_name: CrateName::default(),
            vector: Embedding::default(),
        });
        index += 1;
        if end >= lines.len() {
            break;
        }
        start += step;
    }
    chunks
}

/// Chunk a source file with xberg, under the configured context enrichment
/// (`--retrieval-enrichment`).
///
/// The module path is derived here from the crate map, so every caller — the
/// ingestion loop, the in-memory lexical corpus and the ablation's cost
/// measurement — renders the same prefix for the same file.
pub fn chunks_for_file(
    cfg: &Config,
    path: &str,
    text: &str,
    file_hash: &str,
    crate_name: &str,
    crate_dir: &str,
) -> Vec<LocalChunk> {
    let enrichment = cfg.retrieval_enrichment;
    let overlap = cfg.retrieval_chunk_overlap.get();
    let module_path = module_path_of(crate_name, crate_dir, path);
    // Each chunker degrades to the sliding window on its own failure path (xberg
    // internally, symbol via an empty segmentation), so chunking never fails.
    match cfg.retrieval_chunker {
        ChunkerMode::Xberg => xberg_chunks_with(
            path,
            text,
            file_hash,
            cfg.retrieval_chunk_lines.get(),
            overlap,
            enrichment,
            &module_path,
        ),
        ChunkerMode::Symbol => symbol_chunks_with(
            path,
            text,
            file_hash,
            cfg.retrieval_chunk_lines.get(),
            overlap,
            enrichment,
            &module_path,
        ),
    }
}

/// Chunk one indexed entry. A document entry (converted Markdown) is chunked
/// by the same xberg chunker as source code, then annotated with its
/// provenance and language.
pub fn chunks_for_entry(
    cfg: &Config,
    path: &str,
    text: &str,
    file_hash: &str,
    document_engine: Option<crate::document::Engine>,
    crate_name: &str,
    crate_dir: &str,
) -> Vec<LocalChunk> {
    let mut chunks = chunks_for_file(cfg, path, text, file_hash, crate_name, crate_dir);
    if let Some(engine) = document_engine {
        annotate_document_chunks(&mut chunks, path, engine);
    }
    chunks
}

/// Stamp document chunks: the `markdown` language label and a provenance note
/// on every chunk's `File:` header, so a retrieved `path:line` anchor is never
/// mistaken for a line in the original document.
pub fn annotate_document_chunks(
    chunks: &mut [LocalChunk],
    path: &str,
    engine: crate::document::Engine,
) {
    let header = format!(
        "File: {path} (converted document via {} — line numbers refer to the converted Markdown)",
        engine.label()
    );
    for chunk in chunks.iter_mut() {
        chunk.language = LanguageLabel::from("markdown");
        if let Some(rest) = chunk.content.strip_prefix(&format!("File: {path}")) {
            chunk.content = SourceText::new(format!("{header}{rest}"));
        }
    }
}

#[cfg(test)]
/// xberg document-aware chunking (heading/boundary-aware text splitter),
/// mapped onto the operator's [`LocalChunk`] shape: same context prefix,
/// token cap, and 1-based line anchors (derived from xberg's byte offsets).
/// On any xberg failure it degrades to the legacy sliding window — chunking
/// is never a hard failure.
///
/// Kept as the incumbent-shape entry point; [`xberg_chunks_with`] is the same
/// chunking under any [`ChunkEnrichment`].
pub fn xberg_chunks(
    path: &str,
    text: &str,
    file_hash: &str,
    lines_per_chunk: usize,
) -> Vec<LocalChunk> {
    xberg_chunks_with(
        path,
        text,
        file_hash,
        lines_per_chunk,
        CHUNK_OVERLAP_LINES.get(),
        ChunkEnrichment::PATH,
        "",
    )
}

/// [`xberg_chunks`] with an explicit context field set and module path. The
/// xberg→legacy fallback carries the same enrichment and module path, so a
/// variant never silently degrades to a differently-prefixed corpus. Pure.
pub fn xberg_chunks_with(
    path: &str,
    text: &str,
    file_hash: &str,
    lines_per_chunk: usize,
    overlap_lines: usize,
    enrichment: ChunkEnrichment,
    module_path: &str,
) -> Vec<LocalChunk> {
    // xberg's overlap is in characters. The default 10-line overlap has always
    // been `lines_per_chunk * 8` chars; keep that exactly for the default and
    // scale linearly for other configured overlaps.
    let overlap_chars = lines_per_chunk
        .saturating_mul(8)
        .saturating_mul(overlap_lines)
        .saturating_div(CHUNK_OVERLAP_LINES.get());
    let config = xberg::chunking::ChunkingConfig {
        max_characters: lines_per_chunk.saturating_mul(40).max(200),
        overlap: overlap_chars.max(40),
        trim: true,
        chunker_type: xberg::chunking::ChunkerType::Text,
        ..Default::default()
    };
    let Ok(result) = xberg::chunking::chunk_text(text, &config, None) else {
        return sliding_window_chunks_with(
            path,
            text,
            file_hash,
            lines_per_chunk,
            overlap_lines,
            enrichment,
            module_path,
        );
    };
    result
        .chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let start_line = 1 + text[..c.metadata.byte_start.min(text.len())]
                .bytes()
                .filter(|b| *b == b'\n')
                .count() as u32;
            let prefix = chunk_prefix(enrichment, path, module_path, text, start_line);
            let content = cap_tokens(&format!("{prefix}{}", c.content), EMBEDDER_MAX_TOKENS);
            LocalChunk {
                id: chunk_id(path, i as u32, file_hash),
                path: RepoPath::from(path),
                chunk_index: ChunkIndex::new(i as u32),
                start_line: LineNumber::new(start_line),
                content: SourceText::new(content),
                language: language_of(path),
                file_hash: FileHash::from(file_hash),
                // Stamped by `build`, exactly like the sliding-window path.
                crate_name: CrateName::default(),
                vector: Embedding::default(),
            }
        })
        .collect()
}

#[cfg(test)]
/// Symbol/function-aligned chunking (R8): item spans are packed whole while
/// they fit both the `--retrieval-chunk-lines` window and the measured
/// character budget, and tiled into overlapping windows (each repeating the
/// item's header line) when a single item is too large. On any unrecognized
/// segmentation it degrades to xberg, exactly like xberg degrades to the
/// sliding window. Pure.
///
/// Kept as the incumbent-shape entry point; [`symbol_chunks_with`] is the same
/// chunking under any [`ChunkEnrichment`].
pub fn symbol_chunks(
    path: &str,
    text: &str,
    file_hash: &str,
    lines_per_chunk: usize,
) -> Vec<LocalChunk> {
    symbol_chunks_with(
        path,
        text,
        file_hash,
        lines_per_chunk,
        CHUNK_OVERLAP_LINES.get(),
        ChunkEnrichment::PATH,
        "",
    )
}

/// [`symbol_chunks`] with an explicit context field set and module path. Pure.
pub fn symbol_chunks_with(
    path: &str,
    text: &str,
    file_hash: &str,
    lines_per_chunk: usize,
    overlap_lines: usize,
    enrichment: ChunkEnrichment,
    module_path: &str,
) -> Vec<LocalChunk> {
    let spans = crate::symbol_chunker::rust_item_spans(text);
    if spans.is_empty() {
        return xberg_chunks_with(
            path,
            text,
            file_hash,
            lines_per_chunk,
            overlap_lines,
            enrichment,
            module_path,
        );
    }
    let all_lines: Vec<&str> = text.lines().collect();
    if all_lines.is_empty() {
        return Vec::new();
    }
    let max_chars = chunk_char_budget(EMBEDDER_MAX_TOKENS);

    // Absolute 0-based inclusive ranges of each span.
    let ranges: Vec<(usize, usize)> = spans
        .iter()
        .map(|span| {
            let a = span.start_line.saturating_sub(1).get() as usize;
            let b = a + span.text.lines().count().saturating_sub(1);
            (a, b.min(all_lines.len().saturating_sub(1)))
        })
        .collect();

    let render_len = |a: usize, b: usize| -> usize {
        let prefix = chunk_prefix(enrichment, path, module_path, text, (a + 1) as u32);
        let body: usize = all_lines[a..=b].iter().map(|l| l.chars().count()).sum();
        prefix.chars().count() + body + (b - a)
    };

    let mut chunks: Vec<LocalChunk> = Vec::new();
    let mut index = 0u32;
    let mut open: Option<(usize, usize)> = None;

    let emit = |a: usize, b: usize, index: &mut u32, chunks: &mut Vec<LocalChunk>| {
        let start_line = (a + 1) as u32;
        let prefix = chunk_prefix(enrichment, path, module_path, text, start_line);
        let body = all_lines[a..=b].join("\n");
        let content = cap_tokens(&format!("{prefix}{body}"), EMBEDDER_MAX_TOKENS);
        chunks.push(LocalChunk {
            id: chunk_id(path, *index, file_hash),
            path: RepoPath::from(path),
            chunk_index: ChunkIndex::new(*index),
            start_line: LineNumber::new(start_line),
            content: SourceText::new(content),
            language: language_of(path),
            file_hash: FileHash::from(file_hash),
            crate_name: CrateName::default(),
            vector: Embedding::default(),
        });
        *index += 1;
    };

    for (a, b) in ranges {
        if a > b {
            continue;
        }
        let span_lines = b - a + 1;
        if span_lines > lines_per_chunk || render_len(a, b) > max_chars {
            if let Some((oa, ob)) = open.take() {
                emit(oa, ob, &mut index, &mut chunks);
            }
            // Tile this over-budget item in overlapping windows, each repeating
            // the item's header line so a symbol query hits every window.
            let header = all_lines[a];
            let overlap = overlap_lines.min(lines_per_chunk.saturating_sub(1));
            let mut ws = a;
            while ws <= b {
                let start_line = (ws + 1) as u32;
                let prefix_chars = chunk_prefix(enrichment, path, module_path, text, start_line)
                    .chars()
                    .count();
                let reserve = if ws > a {
                    header.chars().count() + 1
                } else {
                    0
                };
                let mut used = prefix_chars + reserve;
                let mut we = ws;
                let mut n = 0usize;
                while we <= b {
                    let add = all_lines[we].chars().count() + if n == 0 { 0 } else { 1 };
                    if n > 0 && (n + 1 > lines_per_chunk || used + add > max_chars) {
                        break;
                    }
                    used += add;
                    n += 1;
                    we += 1;
                    if n >= lines_per_chunk {
                        break;
                    }
                }
                let we = we.saturating_sub(1).max(ws);
                let body = if ws > a {
                    let mut lines = Vec::with_capacity(we - ws + 2);
                    lines.push(header);
                    lines.extend_from_slice(&all_lines[ws..=we]);
                    lines.join("\n")
                } else {
                    all_lines[ws..=we].join("\n")
                };
                let prefix = chunk_prefix(enrichment, path, module_path, text, start_line);
                let content = cap_tokens(&format!("{prefix}{body}"), EMBEDDER_MAX_TOKENS);
                chunks.push(LocalChunk {
                    id: chunk_id(path, index, file_hash),
                    path: RepoPath::from(path),
                    chunk_index: ChunkIndex::new(index),
                    start_line: LineNumber::new(start_line),
                    content: SourceText::new(content),
                    language: language_of(path),
                    file_hash: FileHash::from(file_hash),
                    crate_name: CrateName::default(),
                    vector: Embedding::default(),
                });
                index += 1;
                if we >= b {
                    break;
                }
                let window = we - ws + 1;
                let ov = overlap.min(window.saturating_sub(1));
                ws = we + 1 - ov;
            }
            continue;
        }
        match open {
            None => open = Some((a, b)),
            Some((oa, ob)) => {
                if (b - oa + 1) <= lines_per_chunk && render_len(oa, b) <= max_chars {
                    open = Some((oa, b));
                } else {
                    emit(oa, ob, &mut index, &mut chunks);
                    open = Some((a, b));
                }
            }
        }
    }
    if let Some((oa, ob)) = open {
        emit(oa, ob, &mut index, &mut chunks);
    }
    chunks
}

/// The identity of the dense vector space written into the index: the
/// configured embed model and backend, sanitized to a path-safe token. It is the
/// fragment that keeps two embedders — including two of the same width — from
/// ever addressing one table.
///
/// Pure and model-free on purpose: [`table_name`] runs on paths that never
/// talk to the server (manifest load/save, unit tests, the "server unavailable"
/// degradation), so a failed embedding must not change the index identity.
pub fn embedder_identity(cfg: &Config) -> String {
    crate::fastembed::embedder_identity(cfg)
}

/// The identity of a workspace's local index: the **repository's** identity
/// (the git common dir, so every `git worktree` of one repo shares it — see
/// [`crate::config::repo_identity`]), the logic version, the chunk parameters,
/// the chunker, the indexed corpus (source or documents) and its converter
/// version, the **dense embedder** ([`embedder_identity`]) that fills it —
/// and the FTS tokenizer (E3/B) that decides which
/// rows a lexical query can reach.
///
/// The embedder belongs here because it *is* the vector space: a swap to
/// another 1024-dim model would otherwise reopen the same table and silently
/// mix two spaces in one index. `table_name_for` is the pure core, so a test
/// can prove two identities never collide without loading a model.
///
/// **Both** the LanceDB table name and the ingestion manifest filename derive
/// from this one string, and `manifest_path` must keep deriving from it. That
/// coupling is load-bearing: the manifest records "these file hashes are already
/// embedded", so if it outlived the table it describes, a changed knob would
/// create a fresh empty table while the manifest suppressed every re-embed —
/// leaving dense and FTS search silently empty (the lexical layer would still
/// answer, so nothing would look broken).
///
/// **One-time re-embed cost (E3/B):** the tokenizer changes what is indexed, so
/// both values must never share a table or a manifest — and the new
/// [`FTS_TOKEN_COLUMN`] does not exist in tables written before this change, so
/// the identity has to move as well. A workspace therefore builds a fresh table
/// and re-embeds its whole corpus **once** (both tokenizer modes: the column is
/// written in every row so a mode switch is always explicit, never a silently
/// empty index). The old table and manifest are left on disk, ignored.
///
/// Consequence of keying the repository rather than the worktree path: an
/// existing git workspace gets a new table name once, so its first post-change
/// run re-embeds the corpus. A non-git workspace's name is byte-identical to
/// before.
pub fn table_name(cfg: &Config) -> String {
    table_name_for(cfg, &embedder_identity(cfg))
}

/// [`table_name`] with the embedder fragment injected, so the collision test can
/// exercise two identities without loading either model.
fn table_name_for(cfg: &Config, embedder: &str) -> String {
    let mut h = Sha256::new();
    h.update(
        crate::config::repo_identity(&cfg.workspace_root)
            .to_string_lossy()
            .as_bytes(),
    );
    let name = format!(
        "code_chunks_{}_v{}_{}_{}_{}_{}_{}_{}{}{}{}{}{}",
        &crate::hash::hex_digest(h.finalize())[..16],
        LOCAL_SEARCH_LOGIC_VERSION,
        cfg.retrieval_chunk_lines,
        cfg.retrieval_chunker.label(),
        FTS_TOKENIZER_LABEL,
        // The document corpus is a different input set, and the converter
        // version changes its text — both re-embed rather than mix.
        if cfg.retrieval_documents.is_on() {
            "docs"
        } else {
            "src"
        },
        // The document corpus identity follows the converter router (engines,
        // versions, mode and fallback policy all change the converted text).
        // Source-only corpora keep the incumbent anydoc fragment, so their
        // table and manifest names are byte-identical to before.
        if cfg.retrieval_documents.is_on() {
            crate::document::router_identity(
                cfg.retrieval_document_engine,
                cfg.retrieval_document_fallback,
            )
        } else {
            crate::document::ANYDOC_VERSION.replace('.', "")
        },
        embedder,
        // E6/C: the context enrichment changes what text is embedded. The
        // incumbent (`path`) contributes the EMPTY fragment, so its table and
        // manifest are the exact names they have always had and nobody
        // re-embeds for a knob they never touched; every other field set adds
        // `_enrich-<label>`, so two enrichments can never address one corpus
        // (the manifest records file hashes, not rendered text — a shared
        // table would keep the wrong vectors).
        cfg.retrieval_enrichment.identity_fragment(),
        // Explicit includes are more indexed input, so two different include
        // sets must never share a table. The incumbent (no includes) adds the
        // EMPTY fragment, keeping every existing table and manifest name
        // byte-identical.
        include_identity_fragment(&cfg.retrieval_include),
        languages_identity_fragment(&cfg.languages),
        exclude_identity_fragment(&cfg.retrieval_exclude),
        overlap_identity_fragment(cfg.retrieval_chunk_overlap),
    );
    // LanceDB rejects table names over 128 characters. Components are bounded
    // except the embedder/model-driven fragments, so hash-truncate the tail
    // rather than fail an index build on a long model or include list; the
    // hash keeps distinct identities distinct.
    if name.len() <= 128 {
        return name;
    }
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    let digest = crate::hash::hex_digest(hasher.finalize());
    format!("{}_{}", &name[..119], &digest[..8])
}

fn exclude_identity_fragment(raw: &[impl AsRef<str>]) -> String {
    let normalized = normalized_includes(raw);
    if normalized.is_empty() {
        return String::new();
    }
    let mut hasher = Sha256::new();
    for path in &normalized {
        hasher.update(path.as_bytes());
        hasher.update([0]);
    }
    format!("_exc{}", &crate::hash::hex_digest(hasher.finalize())[..8])
}

/// A sorted selected-language set is a distinct corpus input.
fn languages_identity_fragment(languages: &[crate::types::LanguageId]) -> String {
    if languages.is_empty() {
        return String::new();
    }
    let mut langs: Vec<&str> = languages
        .iter()
        .map(crate::types::LanguageId::as_str)
        .collect();
    langs.sort_unstable();
    let joined = langs.join("-");
    let mut digest = Sha256::new();
    digest.update(joined.as_bytes());
    format!("_lg{}", &crate::hash::hex_digest(digest.finalize())[..8])
}

/// Manifest path inside the retrieval cache dir.
///
/// Namespaced by [`table_name`] — exactly the identity of the table it
/// describes, so a manifest can never be replayed against a table built from
/// different input. Older manifests (and older tables) are simply ignored.
fn manifest_path(cfg: &Config) -> PathBuf {
    manifest_path_for(cfg, &embedder_identity(cfg))
}

/// [`manifest_path`] with the embedder fragment injected, mirroring
/// [`table_name_for`] so the table/manifest coupling stays one derivation and is
/// testable without a model.
fn manifest_path_for(cfg: &Config, embedder: &str) -> PathBuf {
    cfg.retrieval_cache_dir.join(format!(
        "ingest_manifest_{}.json",
        table_name_for(cfg, embedder)
    ))
}

/// Load the previous ingestion manifest (path → file content hash).
pub fn load_manifest(cfg: &Config) -> HashMap<String, String> {
    std::fs::read_to_string(manifest_path(cfg))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Persist the ingestion manifest (best-effort; a missing manifest only costs
/// a re-embed next run).
pub fn save_manifest(cfg: &Config, manifest: &HashMap<String, String>) {
    let _ = std::fs::create_dir_all(&cfg.retrieval_cache_dir);
    if let Ok(json) = serde_json::to_string_pretty(manifest) {
        let _ = std::fs::write(manifest_path(cfg), json);
    }
}

/// How the ingestion manifest changed across a refresh: files (re)embedded and
/// files whose rows were dropped. Pure.
///
/// "Re-embedded" is a **value** comparison, not a key membership test: the
/// manifest maps path → content hash, so an edited file keeps its key and moves
/// only its hash — and it *was* re-embedded (`files_to_reindex` selects by
/// hash). Counting key-set additions alone would report every edit as `0`,
/// which is exactly the mislabelling this function exists to avoid.
pub fn manifest_delta(
    before: &HashMap<String, String>,
    after: &HashMap<String, String>,
) -> (usize, usize) {
    let added = after
        .iter()
        .filter(|(path, hash)| before.get(*path) != Some(*hash))
        .count();
    let removed = before.keys().filter(|k| !after.contains_key(*k)).count();
    (added, removed)
}

/// What the workspace's local retrieval index currently holds, read without
/// embedding anything (see [`crate::Operator::index_status`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalIndexStatus {
    /// The LanceDB table this configuration resolves to.
    pub table: TableName,
    /// Where the table and its manifest live.
    pub cache_dir: CacheDir,
    /// The ingestion manifest that decides what a rebuild would re-embed.
    pub manifest_path: ManifestPath,
    /// Files the manifest records as already embedded.
    pub manifest_files: FileCount,
    /// Rows in the table, or `None` when it does not exist yet.
    pub rows: Option<RowCount>,
    /// The explicit corpus includes this configuration resolved to, canonical
    /// and sorted (empty when the corpus is the default walk alone).
    pub includes: Vec<IncludePath>,
    /// Selected source files currently on disk.
    pub total_files: FileCount,
    /// Files whose current hash is in the manifest.
    pub fresh_files: FileCount,
    /// Files changed since the manifest was written.
    pub stale_files: FileCount,
    /// Unix milliseconds of the manifest's last write, when it exists.
    pub indexed_at: Option<UnixSeconds>,
}

/// Inspect the local retrieval index **without** building or embedding.
///
/// Never fails: an absent cache directory or table is the "not built" state
/// (`rows: None`), not an error — the same degradation the planner takes when
/// the index cannot be built.
pub async fn retrieval_index_status(cfg: &Config) -> RetrievalIndexStatus {
    let table = table_name(cfg);
    let manifest = load_manifest(cfg);
    let indexed_at = std::fs::metadata(manifest_path(cfg))
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| UnixSeconds::new(duration.as_secs()));
    let mut corpus = discover_source_files(&cfg.workspace_root);
    corpus.retain(|path| selected_source(cfg, path) && !excluded_by(cfg, path));
    let total_files = corpus.len();
    let mut fresh_files = 0usize;
    for path in &corpus {
        let rel = path
            .strip_prefix(&cfg.workspace_root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        if manifest
            .get(&rel)
            .is_some_and(|hash| hash == file_hash_of(&text).as_str())
        {
            fresh_files += 1;
        }
    }
    let rows = if cfg.retrieval_cache_dir.is_dir() {
        match lancedb::connect(cfg.retrieval_cache_dir.to_string_lossy().as_ref())
            .execute()
            .await
        {
            Ok(db) => match db.open_table(table.as_str()).execute().await {
                Ok(open) => open.count_rows(None).await.ok().map(RowCount::new),
                Err(_) => None,
            },
            Err(_) => None,
        }
    } else {
        None
    };
    RetrievalIndexStatus {
        table: TableName::from(table),
        cache_dir: cfg.retrieval_cache_dir.clone(),
        manifest_path: ManifestPath::from(manifest_path(cfg)),
        manifest_files: FileCount::new(manifest.len()),
        rows,
        includes: normalized_includes(&cfg.retrieval_include)
            .into_iter()
            .map(IncludePath::from)
            .collect(),
        total_files: FileCount::new(total_files),
        fresh_files: FileCount::new(fresh_files),
        stale_files: FileCount::new(total_files.saturating_sub(fresh_files)),
        indexed_at,
    }
}

/// Drop the retrieval index this configuration resolves to: the LanceDB table
/// **and** its ingestion manifest.
///
/// Both must go together — the manifest records "these file hashes are already
/// embedded", so a dropped table with a surviving manifest would re-create an
/// empty table and then suppress every re-embed (silently empty dense and FTS
/// search, with only the lexical layer still answering).
///
/// Returns whether anything was dropped. An absent index is `Ok(false)`, not
/// an error; a genuine removal failure is.
pub async fn drop_retrieval_index(cfg: &Config) -> Result<bool, String> {
    let mut dropped = false;
    if cfg.retrieval_cache_dir.is_dir()
        && let Ok(db) = lancedb::connect(cfg.retrieval_cache_dir.to_string_lossy().as_ref())
            .execute()
            .await
    {
        // An absent table is the expected case on a first run; the
        // concrete error is not worth distinguishing here.
        if db.drop_table(&table_name(cfg), &[]).await.is_ok() {
            dropped = true;
        }
    }
    let manifest = manifest_path(cfg);
    if manifest.exists() {
        std::fs::remove_file(&manifest)
            .map_err(|e| format!("removing {}: {e}", manifest.display()))?;
        dropped = true;
    }
    Ok(dropped)
}

/// One incremental refresh of the retrieval corpus index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RetrievalRefresh {
    /// Files whose content changed (or that are new) and were re-embedded.
    pub embedded: FileCount,
    /// Files removed from the workspace (or the corpus) and dropped.
    pub dropped: FileCount,
    /// Files the manifest now records as embedded.
    pub indexed: FileCount,
}

/// Build (or incrementally refresh) the retrieval corpus index for `cfg`.
///
/// The one path both `operator reindex` and `operator serve`'s watcher take,
/// so the two cannot drift. `Err` names the established degradation (no source
/// files, models unavailable, or LanceDB refused); the caller decides whether
/// that is fatal (`reindex`) or a logged, retried watcher pass (`serve`).
pub async fn refresh_retrieval_index(cfg: &Config) -> Result<RetrievalRefresh, String> {
    let before = load_manifest(cfg);
    if LocalCodeIndex::build(cfg).await.is_none() {
        return Err(
            "the retrieval index could not be built (no source files, models unavailable, or \
             LanceDB refused)"
                .to_string(),
        );
    }
    let after = load_manifest(cfg);
    let (embedded, dropped) = manifest_delta(&before, &after);
    Ok(RetrievalRefresh {
        embedded: FileCount::new(embedded),
        dropped: FileCount::new(dropped),
        indexed: FileCount::new(after.len()),
    })
}

/// Incremental decision: which files must be (re)embedded and which dropped.
/// `discovered` = (workspace-relative path, current content hash). Pure.
pub fn files_to_reindex(
    manifest: &HashMap<String, String>,
    discovered: &[(String, String)],
) -> (Vec<String>, Vec<String>) {
    let discovered_paths: Vec<&str> = discovered.iter().map(|(p, _)| p.as_str()).collect();
    let to_embed = discovered
        .iter()
        .filter(|(path, hash)| manifest.get(path) != Some(hash))
        .map(|(path, _)| path.clone())
        .collect();
    let to_drop = manifest
        .keys()
        .filter(|path| !discovered_paths.contains(&path.as_str()))
        .cloned()
        .collect();
    (to_embed, to_drop)
}

// ── Pure retrieval helpers ──────────────────────────────────────────────────

/// Score-aware fusion of dense and FTS results (P1#3), keeping each id's fused
/// score. Dense scores are L2 distances (lower = better) folded into similarity
/// `1/(1+d)`; FTS scores are BM25 (higher = better) folded into `s/(s+1)`. Each
/// id's fused score is the sum of its normalized contributions from each stage
/// — the dense contribution scaled by `dense_weight`, the FTS contribution by
/// `1.0` — so appearing in both stages ranks above appearing in one. Returns
/// `(id, fused score)` pairs ordered best-first (ties broken by dense order,
/// then FTS order). Degrades gracefully to single-stage ranking when either
/// list is empty. Pure.
///
/// `dense_weight` is the single tunable dense/FTS ratio: scaling both stages is
/// a no-op for ranking, so one multiplicative factor spans the whole reachable
/// ranking space. `1.0` is the historic 1:1 sum (the shipped default); `0.0`
/// ranks by BM25 alone; a larger value lets a strong dense hit outrank an
/// FTS-only one. Single-stage orderings are weight-invariant (the surviving
/// stage's own order is preserved by the stage-order tiebreak).
///
/// [`fuse_scored`] is exactly this function projected onto its ids, so the two
/// can never drift.
/// The rank-fusion constant of Reciprocal Rank Fusion (R10/R7).
///
/// `60` is the value the RRF literature and the common implementations use; it
/// damps the head of each list so that the difference between ranks 1 and 2
/// weighs less than the difference between being present and being absent.
/// Query-time only: it never joins the index identity.
pub const RRF_K: f32 = 60.0;

/// How a reranker's raw score becomes the `[0, 1]` relevance the client
/// contract promises (R10's jina finding).
///
/// The pipeline assumes `/v1/rerank` already returned a probability in
/// `[0, 1]`: the shipped rerankers do (sigmoid for the ModernBERT path,
/// `softmax([false, true])` for the CausalLM one). A reranker that returns a
/// raw similarity instead — `jina-reranker-v3` returns a late-interaction
/// cosine in `[-1, 1]`, observed `−0.1856 … 0.1920` — cannot be measured
/// through the shipped path at all, because the only client-side normalization
/// left is a clamp to `[0, 1]` that **collapses every negative score to 0.0**
/// and destroys the ranking.
///
/// `Clamp` and `Sigmoid` are per-score, so they cannot be affected by which
/// other candidates a query happened to return. `MinMax` is per query over the
/// scored candidates, which makes it the transform to use when a reranker's
/// scale is unknown but its order is meaningful; it is order-preserving by
/// construction. Query-time only: keyed, never in the index identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankScoreNorm {
    /// Pass the score through (the incumbent). Correct for a reranker that
    /// already returns a probability; the clamp downstream is then a no-op.
    #[default]
    Clamp,
    /// `1 / (1 + exp(-s))`: monotone, maps any real score into `(0, 1)`.
    Sigmoid,
    /// Per-query min-max over the scored candidates: the best candidate maps to
    /// `1.0` and the worst to `0.0`. A query whose candidates all scored alike
    /// has no scale to recover, so every candidate gets `0.5` — a tie, which is
    /// what "no information" should rank as.
    MinMax,
}

impl RerankScoreNorm {
    /// The run-key and record label.
    pub fn label(self) -> &'static str {
        match self {
            RerankScoreNorm::Clamp => "clamp",
            RerankScoreNorm::Sigmoid => "sigmoid",
            RerankScoreNorm::MinMax => "minmax",
        }
    }

    /// Parse a `--retrieval-rerank-score` value. An unknown value names the
    /// accepted set rather than silently defaulting.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "clamp" => Ok(RerankScoreNorm::Clamp),
            "sigmoid" => Ok(RerankScoreNorm::Sigmoid),
            "minmax" => Ok(RerankScoreNorm::MinMax),
            other => Err(format!(
                "invalid rerank score normalization '{other}' (expected clamp | sigmoid | minmax)"
            )),
        }
    }
}

/// Rewrite one query's rerank scores into `[0, 1]` under `mode`.
///
/// The ordering is preserved by `Sigmoid` and `MinMax` by construction, and
/// `Clamp` is the incumbent's pass-through. Both the ranking and the delivered
/// relevance therefore come from one transform, so a score can never disagree
/// with the position it was ranked at. Pure.
pub fn normalize_rerank_scores(ranked: &mut [(usize, f32)], mode: RerankScoreNorm) {
    match mode {
        RerankScoreNorm::Clamp => {}
        RerankScoreNorm::Sigmoid => {
            for (_, score) in ranked.iter_mut() {
                *score = 1.0 / (1.0 + (-*score).exp());
            }
        }
        RerankScoreNorm::MinMax => {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for (_, score) in ranked.iter() {
                lo = lo.min(*score);
                hi = hi.max(*score);
            }
            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                // One score, or all equal: there is no scale to recover, so
                // every candidate ties rather than one being called the best.
                for (_, score) in ranked.iter_mut() {
                    *score = 0.5;
                }
                return;
            }
            let span = hi - lo;
            for (_, score) in ranked.iter_mut() {
                *score = (*score - lo) / span;
            }
        }
    }
}

/// Which fusion the two stages are combined with (R7).
///
/// `Weighted` is the incumbent: a weighted sum of each stage's normalized score
/// ([`fuse_scored_pairs`]), tuned by `retrieval_dense_weight`. `Rrf` is rank
/// fusion ([`fuse_rrf_pairs`]), which ignores the scores entirely and combines
/// the two stage ranks — the community default for lists whose scores are not
/// calibrated against each other, which is exactly this pipeline's case.
///
/// The knob is query-time only: it changes the ranking but not the stored text,
/// so it is keyed and never joins the index identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fusion {
    /// Weighted sum of normalized scores (the incumbent).
    #[default]
    Weighted,
    /// Reciprocal rank fusion of the two stage orders.
    Rrf,
}

impl Fusion {
    /// The run-key and record label.
    pub fn label(self) -> &'static str {
        match self {
            Fusion::Weighted => "weighted",
            Fusion::Rrf => "rrf",
        }
    }

    /// Parse a `--retrieval-fusion` / `retrieval_fusion` value. An unknown value
    /// names the accepted set rather than silently defaulting.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "weighted" => Ok(Fusion::Weighted),
            "rrf" => Ok(Fusion::Rrf),
            other => Err(format!(
                "invalid retrieval fusion '{other}' (expected weighted | rrf)"
            )),
        }
    }
}

/// Reciprocal rank fusion: each stage contributes `1 / (RRF_K + rank)`, ranks
/// being 1-based, and the sums are ranked descending.
///
/// RRF is deliberately score-blind: it uses only where each stage *placed* an id,
/// so two stages whose scores are on unrelated scales (a cosine similarity and a
/// BM25 score here) can be combined without calibrating them. Ties break by
/// first appearance — dense before FTS, each in its own order — which makes the
/// result a pure function of the two lists.
///
/// [`fuse_rrf`] is exactly this function projected onto its ids, so the two can
/// never drift.
pub fn fuse_rrf_pairs(dense: &[(String, f32)], fts: &[(String, f32)]) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut first_seen: HashMap<String, usize> = HashMap::new();
    for (rank, (id, _)) in dense.iter().enumerate() {
        let contribution = 1.0 / (RRF_K + (rank + 1) as f32);
        scores
            .entry(id.clone())
            .and_modify(|s| *s += contribution)
            .or_insert(contribution);
        first_seen.entry(id.clone()).or_insert(rank);
    }
    for (rank, (id, _)) in fts.iter().enumerate() {
        let contribution = 1.0 / (RRF_K + (rank + 1) as f32);
        scores
            .entry(id.clone())
            .and_modify(|s| *s += contribution)
            .or_insert(contribution);
        first_seen.entry(id.clone()).or_insert(dense.len() + rank);
    }
    let mut fused: Vec<(String, f32, usize)> = scores
        .into_iter()
        .map(|(id, score)| {
            let order = *first_seen.get(&id).unwrap_or(&usize::MAX);
            (id, score, order)
        })
        .collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)));
    fused
        .into_iter()
        .map(|(id, score, _)| (id, score))
        .collect()
}

/// Whether the lexical stage contributes at least one candidate the dense
/// stage did not return — the signal the RRF error analysis found on the
/// queries where rank fusion rescued a missed first result. Pure.
pub fn lexical_stage_adds_candidates(dense: &[(String, f32)], fts: &[(String, f32)]) -> bool {
    let dense_ids: std::collections::HashSet<&str> =
        dense.iter().map(|(id, _)| id.as_str()).collect();
    fts.iter().any(|(id, _)| !dense_ids.contains(id.as_str()))
}

/// Whether a query reads as a symbol/API lookup rather than a natural-language
/// question: it names a qualified symbol (::), carries an identifier with _ or
/// camelCase, or is short enough to be a keyword query. Pure and conservative.
pub fn is_symbol_shaped_query(query: &str) -> bool {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.contains("::") {
        return true;
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() <= 4 {
        return true;
    }
    words.iter().any(|word| {
        let token = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        let underscore = token.contains('_') && token.len() > 2;
        let camel = token
            .chars()
            .zip(token.chars().skip(1))
            .any(|(a, b)| a.is_lowercase() && b.is_uppercase());
        underscore || camel
    })
}

/// The fused order for one query under the configured fusion.
///
/// One place decides which fusion runs, so the live search, the recorded trace
/// and the measurement harness can never disagree about what
/// `retrieval_fusion` means. Under `Rrf` the weight is deliberately ignored:
/// rank fusion has no weight, and honouring one would make the recorded
/// `dense_weight` look like it shaped a ranking it did not.
pub fn fuse_with_mode(
    fusion: Fusion,
    dense: &[(String, f32)],
    fts: &[(String, f32)],
    dense_weight: f32,
) -> Vec<(String, f32)> {
    match fusion {
        Fusion::Weighted => fuse_scored_pairs(dense, fts, dense_weight),
        Fusion::Rrf => fuse_rrf_pairs(dense, fts),
    }
}

pub fn fuse_scored_pairs(
    dense: &[(String, f32)],
    fts: &[(String, f32)],
    dense_weight: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut first_seen: HashMap<String, usize> = HashMap::new();
    // A zero weight means "rank by BM25 alone": the dense stage must not order
    // equal scores, or a zero-weight run would still follow dense order. Its
    // ids stay in the result (with a zero contribution) but sort after every
    // FTS contributor, in their own dense order. Above zero, today's dense-first
    // tie-break is unchanged.
    let dense_orders = dense_weight > 0.0;
    for (rank, (id, d)) in dense.iter().enumerate() {
        // `f32::NAN.max(0.0)` is 0.0, so a NaN distance degrades to a zero
        // similarity instead of poisoning every fused score with NaN.
        let sim = dense_weight * (1.0 / (1.0 + d.max(0.0)));
        scores
            .entry(id.clone())
            .and_modify(|s| *s += sim)
            .or_insert(sim);
        if dense_orders {
            first_seen.entry(id.clone()).or_insert(rank);
        }
    }
    for (rank, (id, s)) in fts.iter().enumerate() {
        let norm = s.max(0.0) / (s.max(0.0) + 1.0);
        scores
            .entry(id.clone())
            .and_modify(|v| *v += norm)
            .or_insert(norm);
        first_seen
            .entry(id.clone())
            .or_insert(usize::MAX / 2 + rank);
    }
    if !dense_orders {
        // Dense-only ids: after every FTS contributor, still in dense order.
        for (rank, (id, _)) in dense.iter().enumerate() {
            first_seen
                .entry(id.clone())
                .or_insert(usize::MAX - dense.len() + rank);
        }
    }
    let mut fused: Vec<(String, f32, usize)> = scores
        .into_iter()
        .map(|(id, s)| {
            let order = *first_seen.get(&id).unwrap_or(&usize::MAX);
            (id, s, order)
        })
        .collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)));
    fused.into_iter().map(|(id, s, _)| (id, s)).collect()
}

#[cfg(test)]
/// The ids of [`fuse_scored_pairs`], in the same order. Pure.
pub fn fuse_scored(
    dense: &[(String, f32)],
    fts: &[(String, f32)],
    dense_weight: f32,
) -> Vec<String> {
    fuse_scored_pairs(dense, fts, dense_weight)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// One stage's candidate as the retrieval eval sees it: the chunk id plus the
/// location the stage selected it at.
///
/// `score` is **stage-native** (dense = L2 distance, lower is better; FTS =
/// BM25; fused = the fused sum; reranked = the backend's relevance in `[0, 1]`)
/// and is never compared across stages — only within one stage's ordered list.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct StageHit {
    /// The chunk id the stage ranked.
    pub id: ChunkId,
    /// Workspace-relative file the chunk belongs to.
    pub path: RepoPath,
    /// 1-based line the chunk starts at.
    pub start_line: LineNumber,
    /// Stage-native score (see the struct docs).
    pub score: Score,
}

/// The per-stage candidate trace of one hybrid search: each list best-first and
/// capped exactly like the live search (`dense`/`fts` ≤ [`STAGE_CANDIDATES`],
/// `fused` ≤ [`RERANK_CANDIDATES`], `reranked` ≤ `samples.min(RERANK_CANDIDATES)`).
///
/// Purely observational: feeding it through [`LocalCodeIndex::hybrid_search`]
/// changes no result, which is why the logic version stays `local-search-4` and
/// the shared index is not orphaned.
#[derive(Debug, Clone, Default, PartialEq)]
#[allow(dead_code)]
pub struct StageTrace {
    /// Dense (vector) stage order.
    pub dense: Vec<StageHit>,
    /// FTS (Tantivy/BM25) stage order; empty when the FTS index is unavailable.
    pub fts: Vec<StageHit>,
    /// The fused, pre-rerank pool order (capped at [`RERANK_CANDIDATES`]).
    pub fused: Vec<StageHit>,
    /// The final reranked top-k the planner's tool would see.
    pub reranked: Vec<StageHit>,
}

/// The one hybrid pipeline's raw output, before it is projected onto either a
/// [`StageTrace`] (the retrieval eval's view) or [`SearchStageStats`] (the run
/// telemetry's view).
///
/// Both public entry points project this same value, so the trace and the
/// counters can never describe different searches. Private: it exposes
/// pre-truncation stage lists that only those two projections may consume.
#[allow(dead_code)]
struct HybridSearchCore {
    /// Delivered `(chunk, score)` pairs, best-first.
    chunks: Vec<(LocalChunk, Score)>,
    /// Dense stage candidates (pre-fusion, capped at [`STAGE_CANDIDATES`]).
    dense: Vec<(String, f32)>,
    /// FTS stage candidates (empty when `fts_ok` is false).
    fts: Vec<(String, f32)>,
    /// The fused, `by_id`-resolved, [`RERANK_CANDIDATES`]-capped pool the
    /// cross-encoder was handed.
    pool: Vec<(String, f32)>,
    /// The delivered `(id, rerank score)` pairs, best-first.
    reranked_pairs: Vec<(String, f32)>,
    /// The FTS stage succeeded (a false value is a dense-only degraded call).
    fts_ok: bool,
}

/// Project `(id, score)` pairs onto [`StageHit`]s through a chunk lookup.
///
/// An id the lookup does not know is **skipped**, never given a placeholder
/// location, so a trace can never name a path no chunk has. Order and scores
/// are preserved. Pure.
#[allow(dead_code)]
pub fn project_stage_hits(
    ids: &[(String, f32)],
    lookup: &impl Fn(&str) -> Option<(String, u32)>,
) -> Vec<StageHit> {
    ids.iter()
        .filter_map(|(id, score)| {
            lookup(id).map(|(path, start_line)| StageHit {
                id: ChunkId::from(id.clone()),
                path: RepoPath::from(path),
                start_line: LineNumber::new(start_line),
                score: Score::new(*score),
            })
        })
        .collect()
}

/// Keep the top `n` (score, original index) pairs, sorted by score desc.
/// Pure.
pub fn rerank_top(mut ranked: Vec<(usize, f32)>, n: usize) -> Vec<(usize, f32)> {
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(n);
    ranked
}

// ── Per-call stage accounting (E8) ──────────────────────────────────────────

/// What each stage of one hybrid search contributed, and whether reranking
/// changed the answer.
///
/// Produced by [`search_stage_stats`] from the stage outputs a search already
/// has — nothing here is recomputed from the results, so a counter can never
/// disagree with the ranking it describes. Every count is a *candidate* count:
/// what the stage returned before the delivered answer was truncated.
///
/// See [`crate::run_record::RetrievalStatsRecord`] for how a run sums these
/// and what each field means across many calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub struct SearchStageStats {
    /// Dense (vector) stage candidates.
    pub dense_candidates: CandidateCount,
    /// FTS (BM25) stage candidates.
    pub fts_candidates: CandidateCount,
    /// Distinct candidates in the dense ∪ FTS union (what fusion ranked).
    pub fused_candidates: CandidateCount,
    /// Candidates present in **both** stage lists (dense ∩ FTS) — the
    /// population the score-sum fusion can actually boost.
    pub both_stages: CandidateCount,
    /// Candidates actually handed to the cross-encoder (after the `by_id`
    /// lookup and the [`RERANK_CANDIDATES`] cap).
    pub rerank_pool: CandidateCount,
    /// Delivered chunks the reranker lifted from at or beyond the delivered
    /// count in the fused order.
    pub rerank_promotions: CandidateCount,
    /// The FTS stage errored; the search served dense-only results.
    pub fts_degraded: FtsDegraded,
}

/// The result of a hybrid search together with its per-stage accounting.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct SearchOutcome {
    /// The delivered `(chunk, score)` pairs, best-first — byte-identical to
    /// [`LocalCodeIndex::hybrid_search`].
    pub chunks: Vec<(LocalChunk, Score)>,
    /// What the stages did to produce them.
    pub stages: SearchStageStats,
}

/// The per-stage accounting of one hybrid search. Pure.
///
/// `dense`/`fts` are the stage lists the pipeline ranked; `rerank_pool` is the
/// size of the candidate pool actually handed to the cross-encoder (a
/// parameter because a fused id missing from the chunk map never becomes a
/// candidate, so it is not `min(fused, RERANK_CANDIDATES)` in general);
/// `delivered_candidate_indices` are the delivered chunks' indices **into that
/// pool**, so an index `>= delivered_candidate_indices.len()` is a chunk the
/// reranker placed inside the delivered top-k although fusion had it outside —
/// a promotion. `fts_ok` is the FTS stage's success flag, which is the only
/// way to tell a broken FTS index from a query BM25 genuinely did not match.
#[allow(dead_code)]
pub fn search_stage_stats(
    dense: &[(String, f32)],
    fts: &[(String, f32)],
    rerank_pool: usize,
    delivered_candidate_indices: &[usize],
    fts_ok: bool,
) -> SearchStageStats {
    let dense_ids: std::collections::HashSet<&str> =
        dense.iter().map(|(id, _)| id.as_str()).collect();
    let fts_ids: std::collections::HashSet<&str> = fts.iter().map(|(id, _)| id.as_str()).collect();
    let mut fused: std::collections::HashSet<&str> = dense_ids.clone();
    fused.extend(fts_ids.iter().copied());
    let delivered = delivered_candidate_indices.len();
    SearchStageStats {
        dense_candidates: CandidateCount::new(dense.len()),
        fts_candidates: CandidateCount::new(fts.len()),
        fused_candidates: CandidateCount::new(fused.len()),
        both_stages: CandidateCount::new(dense_ids.intersection(&fts_ids).count()),
        rerank_pool: CandidateCount::new(rerank_pool),
        rerank_promotions: CandidateCount::new(
            delivered_candidate_indices
                .iter()
                .filter(|idx| **idx >= delivered)
                .count(),
        ),
        fts_degraded: FtsDegraded::new(!fts_ok),
    }
}

/// The pool indices of `reranked_ids` within the fused order, for
/// [`search_stage_stats`]'s `delivered_candidate_indices`.
///
/// An id absent from `fused_ids` gets `usize::MAX` (treated as a promotion —
/// it is outside the fused candidate order by definition). Pure.
#[allow(dead_code)]
pub fn delivered_pool_indices(fused_ids: &[String], reranked_ids: &[String]) -> Vec<usize> {
    reranked_ids
        .iter()
        .map(|id| {
            fused_ids
                .iter()
                .position(|fused| fused == id)
                .unwrap_or(usize::MAX)
        })
        .collect()
}

/// L2-normalize a vector in place. Returns `false` (leaving `v` unchanged)
/// when the norm is zero or non-finite — the vector can never become NaN.
/// Returns `true` after dividing each element by the norm otherwise.
/// Pure.
pub fn l2_normalize(v: &mut [f32]) -> bool {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq == 0.0 {
        return false;
    }
    let norm = norm_sq.sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

// ── Model seam ──────────────────────────────────────────────────────────────
// The backend contract lives in [`crate::embedding::Embedded`]; the shipped
// in-process implementation is [`crate::fastembed::FastembedModels`]. Tests
// provide fakes of that trait.

/// Truncate `text` to at most `max_chars` characters, char-safe and with no
/// marker. The reranker's own context budget. Pure.
pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

// ── The index ───────────────────────────────────────────────────────────────

/// Whether a query should be served by the secondary index: one must exist and
/// the routing toggle must be on. Pure (the symbol-shape test is separate so it
/// can be unit-tested without an index).
pub fn dual_route_applies(has_secondary: bool, route_enabled: bool, query: &str) -> bool {
    has_secondary && route_enabled && is_symbol_shaped_query(query)
}

/// The planner-facing local index: LanceDB table + in-memory chunk list (for
/// the exact-identifier lexical layer) + the models.
pub struct LocalCodeIndex {
    table: lancedb::Table,
    /// Chunk id → chunk (vector left empty here; memory-only for lexical hits).
    by_id: HashMap<ChunkId, LocalChunk>,
    models: Arc<dyn crate::embedding::Embedded>,
    /// Whether the cross-encoder rerank stage runs
    /// (`--retrieval-rerank-enabled`). When off, the fused order is delivered
    /// directly and no rerank request is sent.
    rerank_enabled: RerankEnabled,
    /// Whether the cross-encoder documents get a derived `Module:` prefix.
    rerank_context: RerankContext,
    /// Whether lexical-only candidates route the fusion to RRF.
    fusion_gate: FusionGate,
    /// Optional local JSONL search-query log (off by default).
    query_log: Option<QueryLogPath>,
    /// Which vector index the dense stage searches through
    /// (`--retrieval-vector-index`). `Exact` bypasses any trained index, so an
    /// old `IVF_PQ` table is never read through its approximation.
    vector_index: crate::config::RetrievalVectorIndex,
    /// Candidates fetched per stage (dense and FTS) before fusion
    /// (`--retrieval-stage-candidates`; the shipped [`STAGE_CANDIDATES`] by
    /// default).
    stage_candidates: CandidateCount,
    /// Candidates passed to the cross-encoder after fusion
    /// (`--retrieval-rerank-candidates`; the shipped [`RERANK_CANDIDATES`] by
    /// default).
    rerank_candidates: CandidateCount,
    /// Which fusion combines the two stage lists (R7,
    /// `--retrieval-fusion`). Cached on the index for the same reason the
    /// counts are: the search methods take only the query, so reading it from
    /// here keeps one fusion decision shared by the live path and the trace.
    fusion: Fusion,
    /// How a reranker's raw score becomes `[0, 1]` (R10's jina finding),
    /// cached here for the same reason the fusion is: the search methods take
    /// only the query.
    rerank_score_norm: RerankScoreNorm,
    /// Weight of the cross-encoder score when blending it with the fused
    /// retrieval score (`0.0` = pure reranker order).
    rerank_blend: Blend,
    /// Whether the dense query carries the code-aware identifier tokens.
    query_expansion: QueryExpansion,
    /// Whether symbol-shaped queries use reciprocal-rank fusion.
    query_routing: QueryRouting,
    /// Optional secondary index (its own table + embedder) for symbol-shaped
    /// queries; built only when a dual embed model is configured.
    secondary: Option<Arc<LocalCodeIndex>>,
    /// Whether symbol-shaped queries are routed to the secondary index.
    dual_route: DualRoute,
}

/// One JSONL record for the opt-in local query log: the query text, scope,
/// sample count, latency and the returned paths. Deliberately excludes chunk
/// contents and any source text. Pure.
pub fn query_log_line(
    query: &str,
    scope: Option<&str>,
    samples: usize,
    latency_ms: u64,
    paths: &[String],
) -> String {
    let top: Vec<&String> = paths.iter().take(10).collect();
    serde_json::json!({
        "v": 1,
        "ts_unix_ms": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        "query": query,
        "scope": scope,
        "samples": samples,
        "latency_ms": latency_ms,
        "top_paths": top,
    })
    .to_string()
}

/// Append one line to the query log, creating parent directories. Best-effort:
/// a logging failure must never fail the search.
fn append_query_log(path: &std::path::Path, line: &str) {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{line}");
    }
}

/// A module-level hint for a chunk path, used only when the reranker context
/// toggle is on (the stored text already carries the `File:` prefix). Pure.
pub fn module_hint(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    // `crates/<crate>/src/<module>`: keep the crate name as the root.
    let (crate_name, after) = match normalized.find("crates/") {
        Some(index) => {
            let tail = &normalized[index + "crates/".len()..];
            match tail.split_once("/src/") {
                Some((crate_name, rest)) => (Some(crate_name), rest),
                None => (None, tail),
            }
        }
        None => (None, normalized.as_str()),
    };
    let after = match crate_name {
        Some(_) => after,
        None => match normalized.rfind("src/") {
            Some(index) => &normalized[index + 4..],
            None => normalized.as_str(),
        },
    };
    let without_ext = after.strip_suffix(".rs").unwrap_or(after);
    let mut parts: Vec<&str> = without_ext
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    if matches!(parts.last().copied(), Some("mod" | "lib" | "main")) {
        parts.pop();
    }
    let module = parts.join("::");
    match (crate_name, module.is_empty()) {
        (Some(crate_name), true) => crate_name.to_string(),
        (Some(crate_name), false) => format!("{crate_name}::{module}"),
        (None, true) => "src".to_string(),
        (None, false) => module,
    }
}

/// Identity fragment for a non-default chunk overlap (empty at the shipped
/// [`CHUNK_OVERLAP_LINES`], so existing tables/manifests keep their names).
fn overlap_identity_fragment(overlap_lines: crate::types::ChunkOverlap) -> String {
    if overlap_lines == CHUNK_OVERLAP_LINES {
        String::new()
    } else {
        format!("_ov{}", overlap_lines.get())
    }
}

/// Arrow schema for the code-chunks table (the spec's data model plus
/// `start_line` and the `crate_name` scope column).
fn code_schema(dim: usize) -> arrow_schema::Schema {
    use arrow_schema::{DataType, Field, Schema};
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("file_hash", DataType::Utf8, false),
        Field::new("crate_name", DataType::Utf8, false),
        // E3/B: the code-aware searchable text. Never null, so a tokenizer mode
        // switch can never leave the index empty without an explicit error.
        Field::new(FTS_TOKEN_COLUMN, DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ])
}

/// Build a RecordBatch from fully-embedded chunks (vectors present).
fn chunks_to_batch(chunks: &[LocalChunk]) -> Result<arrow_array::RecordBatch, String> {
    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt32Array};
    // The width comes from the vectors themselves: the embedder's guard already
    // rejected a mismatched response, so every chunk carries the configured
    // width. Empty input cannot reach here (one file always has one chunk).
    let dim = chunks.first().map_or(DENSE_DIM.get(), |c| c.vector.len());
    if chunks
        .iter()
        .any(|c| c.vector.len() != dim || c.vector.len() % dim != 0)
    {
        return Err(format!(
            "vector length {} != dense dim {dim}",
            chunks
                .iter()
                .map(|c| c.vector.len())
                .find(|l| *l != dim)
                .unwrap_or(dim)
        ));
    }
    let schema = Arc::new(code_schema(dim));
    let ids = StringArray::from(chunks.iter().map(|c| c.id.as_str()).collect::<Vec<_>>());
    let paths = StringArray::from(chunks.iter().map(|c| c.path.as_str()).collect::<Vec<_>>());
    let indexes = UInt32Array::from(
        chunks
            .iter()
            .map(|c| c.chunk_index.get())
            .collect::<Vec<_>>(),
    );
    let starts = UInt32Array::from(
        chunks
            .iter()
            .map(|c| c.start_line.get())
            .collect::<Vec<_>>(),
    );
    let contents = StringArray::from(
        chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>(),
    );
    let languages = StringArray::from(
        chunks
            .iter()
            .map(|c| c.language.as_str())
            .collect::<Vec<_>>(),
    );
    let hashes = StringArray::from(
        chunks
            .iter()
            .map(|c| c.file_hash.as_str())
            .collect::<Vec<_>>(),
    );
    let crate_names = StringArray::from(
        chunks
            .iter()
            .map(|c| c.crate_name.as_str())
            .collect::<Vec<_>>(),
    );
    let fts_tokens = StringArray::from(chunks.iter().map(fts_text_of).collect::<Vec<String>>());
    let values: Vec<f32> = chunks
        .iter()
        .flat_map(|c| c.vector.iter().copied())
        .collect();
    // The constructor wraps the inner item field: values must be a flat
    // Float32 array and the field its ITEM field (arrow 58 semantics). The
    // item's nullability must match the schema's (nullable) or the batch
    // validation rejects the column.
    let item_field = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        true,
    ));
    let vectors = FixedSizeListArray::new(
        item_field,
        dim as i32,
        Arc::new(Float32Array::from(values)),
        None,
    );
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(paths),
            Arc::new(indexes),
            Arc::new(starts),
            Arc::new(contents),
            Arc::new(languages),
            Arc::new(hashes),
            Arc::new(crate_names),
            Arc::new(fts_tokens),
            Arc::new(vectors),
        ],
    )
    .map_err(|e| format!("building record batch: {e}"))
}

/// Group consecutive files into embedding batches whose total chunk count is at
/// most `cap`. A file is never split across batches; a file larger than `cap`
/// forms a batch of its own, and `cap == 0` puts every file in its own batch.
/// Returns index ranges into `chunk_counts`. Pure.
pub fn plan_embed_batches(chunk_counts: &[usize], cap: usize) -> Vec<std::ops::Range<usize>> {
    let mut batches = Vec::new();
    let mut start = 0usize;
    let mut total = 0usize;
    for (i, count) in chunk_counts.iter().enumerate() {
        if i > start && (cap == 0 || total + count > cap) {
            batches.push(start..i);
            start = i;
            total = 0;
        }
        total += count;
    }
    if start < chunk_counts.len() {
        batches.push(start..chunk_counts.len());
    }
    batches
}

/// One `embed` call against the embedded backend.
///
/// `Err` carries the backend's message verbatim, so a caller can print exactly
/// what the pre-batching per-file path printed.
async fn embed_texts(
    models: &Arc<dyn crate::embedding::Embedded>,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, String> {
    models.embed(texts).await
}

/// Normalize, build the Arrow batch and insert one file's chunks, recording the
/// file in `new_manifest` (and printing the per-file progress line) on success.
///
/// Returns whether the file's rows were inserted. A file whose insert fails is
/// deliberately left out of the manifest so the next run retries it.
async fn insert_file_chunks(
    table: &lancedb::Table,
    new_manifest: &mut HashMap<String, String>,
    discovered_map: &HashMap<&str, &str>,
    file: &str,
    chunks: &[LocalChunk],
    vectors: &[Vec<f32>],
) -> bool {
    let chunks: Vec<LocalChunk> = chunks
        .iter()
        .cloned()
        .zip(vectors.iter().cloned())
        .map(|(mut chunk, mut vector)| {
            // Normalize every stored vector so the index's L2 metric ranks
            // identically to cosine similarity (||a-b||² = 2 - 2·cos when both
            // are unit-norm). The embedder already normalizes, so this is a
            // no-op in practice; on a degenerate vector it keeps the raw values
            // rather than storing NaN.
            let _ = l2_normalize(&mut vector);
            chunk.vector = Embedding::from(vector);
            chunk
        })
        .collect();
    let batch = match chunks_to_batch(&chunks) {
        Ok(batch) => batch,
        Err(e) => {
            note!("  (retrieval: building batch for {file} failed: {e})");
            return false;
        }
    };
    if let Err(e) = table.add(batch).execute().await {
        note!("  (retrieval: inserting {file} chunks failed: {e})");
        return false;
    }
    if let Some(hash) = discovered_map.get(file) {
        new_manifest.insert(file.to_string(), hash.to_string());
    }
    note!("  retrieval: embedded {file} ({} chunks)", chunks.len());
    true
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(dead_code)]
pub enum IndexEvent {
    DiscoverStart,
    Discovered {
        files: FileCount,
    },
    Languages {
        languages: usize,
    },
    HashProgress {
        done: FileCount,
        total: FileCount,
        changed: FileCount,
    },
    TableReady {
        table: TableName,
    },
    FileIndexed {
        path: RepoPath,
        chunks: ChunkCount,
    },
    FileSkipped {
        path: RepoPath,
        reason: ErrorMessage,
    },
    ManifestSaved {
        files: FileCount,
    },
    Done {
        embedded: FileCount,
        dropped: FileCount,
        indexed: ChunkCount,
        elapsed_ms: DurationMs,
    },
    Failed {
        reason: ErrorMessage,
    },
}

fn emit(events: Option<&tokio::sync::mpsc::UnboundedSender<IndexEvent>>, event: IndexEvent) {
    if let Some(tx) = events {
        let _ = tx.send(event);
    }
}

/// Whether a workspace-relative path is under one of the explicit exclude
/// prefixes. Excludes are prefixes, so `vendor` covers `vendor/foo`.
pub fn excluded_by(cfg: &Config, path: &Path) -> bool {
    let rel = path
        .strip_prefix(&cfg.workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let depth = |entry: &str| -> Option<usize> {
        let entry = entry.trim().trim_start_matches("./").trim_end_matches('/');
        if entry.is_empty() {
            return None;
        }
        (rel == entry || rel.starts_with(&format!("{entry}/"))).then_some(entry.len())
    };
    let exclude = cfg
        .retrieval_exclude
        .iter()
        .filter_map(|entry| depth(entry.as_str()))
        .max();
    let include = cfg
        .retrieval_include
        .iter()
        .filter_map(|entry| depth(entry.as_str()))
        .max();
    matches!((exclude, include), (Some(ex), Some(inc)) if ex >= inc)
        || matches!((exclude, include), (Some(_), None))
}

impl LocalCodeIndex {
    /// Build (or incrementally update) the local index over the workspace.
    /// Every failure logs and returns `None` — the planner runs without the
    /// retrieval tool, never fails the run.
    pub async fn build(cfg: &Config) -> Option<Arc<Self>> {
        let primary = Self::build_inner(cfg, None).await?;
        Some(Self::attach_secondary(primary, cfg).await)
    }

    /// Build the secondary index (when configured) and attach it to the
    /// primary. A failed secondary build leaves the primary working and says
    /// so; routing then falls back to the primary.
    async fn attach_secondary(mut primary: Arc<Self>, cfg: &Config) -> Arc<Self> {
        let Some(model) = cfg.retrieval_dual_embed_model.clone() else {
            return primary;
        };
        let mut dcfg = cfg.clone();
        dcfg.retrieval_dual_embed_model = None;
        dcfg.retrieval_embed_model = model;
        if let Some(dim) = cfg.retrieval_dual_embed_dim {
            dcfg.retrieval_embed_dim = dim;
        }
        if let Some(backend) = cfg.retrieval_dual_embed_backend {
            dcfg.retrieval_embed_backend = backend;
        }
        if let Some(prefix) = cfg.retrieval_dual_embed_query_prefix.clone() {
            dcfg.retrieval_embed_query_prefix = prefix;
        }
        match Self::build_inner(&dcfg, None).await {
            Some(secondary) => {
                if let Some(inner) = Arc::get_mut(&mut primary) {
                    inner.secondary = Some(secondary);
                }
            }
            None => note!(
                "  (retrieval: dual index `{}` failed to build; symbol queries fall back to the primary index)",
                dcfg.retrieval_embed_model
            ),
        }
        primary
    }

    /// Build with file-level progress events for the TUI.
    #[cfg(feature = "tui")]
    pub async fn build_with_events(
        cfg: &Config,
        events: tokio::sync::mpsc::UnboundedSender<IndexEvent>,
    ) -> Option<Arc<Self>> {
        let primary = Self::build_inner(cfg, Some(&events)).await?;
        Some(Self::attach_secondary(primary, cfg).await)
    }

    async fn build_inner(
        cfg: &Config,
        events: Option<&tokio::sync::mpsc::UnboundedSender<IndexEvent>>,
    ) -> Option<Arc<Self>> {
        emit(events, IndexEvent::DiscoverStart);
        let mut files = discover_source_files(&cfg.workspace_root);
        // Opt-in document corpus: converted to Markdown with anydoc and chunked
        // by the SAME configured chunker as sources (see chunks_for_entry).
        let document_files = if cfg.retrieval_documents.is_on() {
            discover_document_files(&cfg.workspace_root)
        } else {
            Vec::new()
        };
        // Explicit includes (`--include` / `retrieval_include`): the user named
        // these trees, so every indexable file under them joins the same corpus.
        // They are part of the index identity, so this table is the one the
        // matching configuration addresses — never a mix of two corpora.
        let include_files: Vec<PathBuf> =
            resolve_includes(&cfg.workspace_root, &cfg.retrieval_include)
                .iter()
                .flat_map(|root| discover_include_files(root))
                .collect();
        if files.is_empty() && document_files.is_empty() && include_files.is_empty() {
            note!(
                "  (retrieval: no indexable source files under {})",
                cfg.workspace_root.display()
            );
            return None;
        }
        files.extend(document_files.iter().cloned());
        files.extend(include_files);
        // A file reachable both through the default walk and an include (an
        // included subtree of the workspace) is one corpus entry, not two:
        // otherwise it would be chunked and embedded twice.
        files.sort();
        files.dedup();
        files.retain(|path| selected_source(cfg, path) && !excluded_by(cfg, path));
        emit(
            events,
            IndexEvent::Discovered {
                files: FileCount::new(files.len()),
            },
        );
        emit(
            events,
            IndexEvent::Languages {
                languages: cfg.languages.len(),
            },
        );
        // Cargo crate boundaries, computed once: every chunk is stamped with its
        // crate so a search can be confined to one (`search_vector_store`'s
        // `crate_name`), and files outside every crate fall back to the
        // workspace root label.
        let scopes = crate_map::crate_scopes(&cfg.workspace_root, is_excluded_dir);
        // The crate's NAME stamps the chunk; its ROOT DIRECTORY is what makes
        // the derived module path crate-relative (E6/C), so both come from the
        // same lookup.
        let scope_of = |rel: &str| -> (String, String) {
            crate_map::scope_of(&scopes, rel)
                .map(|scope| (scope.name.clone(), scope.rel_dir.clone()))
                .unwrap_or_else(|| (crate_map::NO_CRATE.to_string(), String::new()))
        };
        let models: Arc<dyn crate::embedding::Embedded> =
            match crate::fastembed::FastembedModels::build(cfg) {
                Ok(models) => Arc::new(models),
                Err(e) => {
                    note!("  (retrieval: fastembed backend unavailable — {e})");
                    return None;
                }
            };
        // Fail once, clearly, before paying for a corpus walk and a partial
        // embed: a missing device, cache or model id is the common
        // misconfiguration.
        if let Err(e) = models.preflight(cfg.retrieval_rerank_enabled.is_on()).await {
            note!("  (retrieval: fastembed backend unavailable — {e})");
            return None;
        }
        if models.embed_dim() != cfg.retrieval_embed_dim {
            note!(
                "  (retrieval: backend produces {}-d vectors but retrieval_embed_dim is {})",
                models.embed_dim(),
                cfg.retrieval_embed_dim
            );
            return None;
        }
        let db = match lancedb::connect(cfg.retrieval_cache_dir.to_string_lossy().as_ref())
            .execute()
            .await
        {
            Ok(db) => db,
            Err(e) => {
                note!(
                    "  (retrieval: opening LanceDB at {} failed: {e})",
                    cfg.retrieval_cache_dir.display()
                );
                return None;
            }
        };

        // Discover + hash every file once.
        let mut discovered: Vec<(String, String)> = Vec::new();
        let mut contents: HashMap<String, String> = HashMap::new();
        // Which engine converted each document, stamped onto its chunks so a
        // retrieved chunk says whether it came from anydoc or xberg.
        let mut document_engines: HashMap<String, crate::document::Engine> = HashMap::new();
        for path in &files {
            let rel = match path.strip_prefix(&cfg.workspace_root) {
                Ok(rel) => rel.to_string_lossy().into_owned(),
                Err(_) => path.to_string_lossy().into_owned(),
            };
            let is_document = crate::document::is_document_path(path);
            let text = if is_document {
                // The router's fallback turn is recorded so a partial
                // conversion (for example a scanned PDF that anydoc refuses
                // and xberg reads partially) is visible, never silent.
                match crate::document::convert_file(path).await {
                    Ok(converted) => {
                        if let Some(primary_error) = &converted.fallback {
                            note!(
                                "  (retrieval: {rel}: converted via {} after {primary_error} — partial)",
                                converted.engine.label()
                            );
                        }
                        document_engines.insert(rel.clone(), converted.engine);
                        cap_document_text(converted.markdown)
                    }
                    Err(e) => {
                        note!("  (retrieval: {rel}: {e} — skipped)");
                        continue;
                    }
                }
            } else {
                match std::fs::read_to_string(path) {
                    Ok(text) => text,
                    Err(_) => continue,
                }
            };
            let hash = file_hash_of(&text);
            contents.insert(rel.clone(), text);
            discovered.push((rel, hash.into_inner()));
        }
        if discovered.is_empty() {
            note!("  (retrieval: no readable source files)");
            return None;
        }

        let manifest = load_manifest(cfg);
        let (to_embed, to_drop) = files_to_reindex(&manifest, &discovered);
        let discovered_map: HashMap<&str, &str> = discovered
            .iter()
            .map(|(p, h)| (p.as_str(), h.as_str()))
            .collect();
        // The manifest only ever records files whose chunks were ACTUALLY
        // inserted: a failed run leaves them out so the next run retries them
        // (a poisoned manifest would silently skip the whole corpus).
        let mut new_manifest = manifest.clone();
        for path in &to_drop {
            new_manifest.remove(path);
        }

        // Open or create the table (name encodes workspace + params).
        let name = table_name(cfg);
        let table = match db.open_table(&name).execute().await {
            Ok(table) => table,
            Err(_) => match db
                .create_empty_table(&name, Arc::new(code_schema(cfg.retrieval_embed_dim.get())))
                .execute()
                .await
            {
                Ok(table) => {
                    // The FTS index is created once (works on an empty table);
                    // LanceDB keeps it updated as rows are added. The VECTOR
                    // index is created after the first insert below — an empty
                    // table cannot train it ("Creating empty vector indices
                    // with train=False is not yet implemented").
                    //
                    // E3/B: exactly ONE column is FTS-indexed, the
                    // pre-tokenized `fts_tokens` column, with the engine's
                    // default tokenizer — so the engine's own tokenizer stays
                    // the measured baseline and the code awareness lives in the
                    // text, not the engine.
                    if let Err(e) = table
                        .create_index(
                            &[FTS_TOKEN_COLUMN],
                            lancedb::index::Index::FTS(Default::default()),
                        )
                        .execute()
                        .await
                    {
                        note!("  (retrieval: FTS index creation failed, dense-only search: {e})");
                    }
                    table
                }
                Err(e) => {
                    note!("  (retrieval: creating LanceDB table {name} failed: {e})");
                    return None;
                }
            },
        };

        emit(
            events,
            IndexEvent::TableReady {
                table: TableName::from(name.clone()),
            },
        );
        // Drop rows for deleted files AND for changed files about to be
        // re-embedded. A changed file's chunk ids embed its content hash, so
        // its old rows would otherwise linger: they no longer resolve through
        // `by_id` (rebuilt from current contents) but still eat
        // STAGE_CANDIDATES/RERANK_CANDIDATES slots and grow the table on every
        // edit. Deleting by `file_path` before the insert loop makes each
        // re-embed replace, not append, that file's rows.
        for path in to_drop.iter().chain(to_embed.iter()) {
            let predicate = format!("file_path = {}", sql_string_literal(path));
            if let Err(e) = table.delete(predicate.as_str()).await {
                note!("  (retrieval: dropping stale rows for {path} failed: {e})");
            }
        }

        // Embed changed files in batches and insert.
        //
        // Chunking stays per file (it is pure and per-file), but the
        // *embedding* is grouped across consecutive files so the session gets a
        // full batch: one file is typically a handful of chunks, far below
        // `EMBED_GROUP_CHUNKS`, and a call that small underfills the HTTP
        // request.
        let mut embedded = 0usize;
        let mut embedded_chunks = 0usize;
        if !to_embed.is_empty() {
            let embed_started = std::time::Instant::now();
            // Chunk every changed file once, up front: the grouping needs the
            // per-file chunk counts, and the chunks are what gets embedded.
            let mut per_file: Vec<(String, Vec<LocalChunk>)> = Vec::with_capacity(to_embed.len());
            for file in &to_embed {
                let Some(text) = contents.get(file) else {
                    continue;
                };
                let (scope, crate_dir) = scope_of(file);
                let mut chunks = chunks_for_entry(
                    cfg,
                    file,
                    text,
                    file_hash_of(text).as_str(),
                    document_engines.get(file).copied(),
                    &scope,
                    &crate_dir,
                );
                for chunk in &mut chunks {
                    chunk.crate_name = CrateName::from(scope.clone());
                }
                per_file.push((file.clone(), chunks));
            }
            let counts: Vec<usize> = per_file.iter().map(|(_, chunks)| chunks.len()).collect();
            for range in plan_embed_batches(&counts, EMBED_GROUP_CHUNKS) {
                let group = &per_file[range];
                let texts: Vec<String> = group
                    .iter()
                    .flat_map(|(_, chunks)| chunks.iter().map(|c| c.content.as_str().to_string()))
                    .collect();
                match embed_texts(&models, texts.clone()).await {
                    Ok(vectors) if vectors.len() == texts.len() => {
                        let mut offset = 0usize;
                        for (file, chunks) in group {
                            let count = chunks.len();
                            let file_vectors = &vectors[offset..offset + count];
                            offset += count;
                            if insert_file_chunks(
                                &table,
                                &mut new_manifest,
                                &discovered_map,
                                file,
                                chunks,
                                file_vectors,
                            )
                            .await
                            {
                                embedded += 1;
                                embedded_chunks += count;
                                emit(
                                    events,
                                    IndexEvent::FileIndexed {
                                        path: RepoPath::from(file.clone()),
                                        chunks: ChunkCount::new(count),
                                    },
                                );
                            }
                        }
                    }
                    // The batch failed, or came back with the wrong cardinality:
                    // replay the group one file at a time with the pre-batching
                    // path (and its messages), so a single unembeddable file
                    // cannot cost the whole batch.
                    Ok(_) | Err(_) => {
                        for (file, chunks) in group {
                            let texts: Vec<String> = chunks
                                .iter()
                                .map(|c| c.content.as_str().to_string())
                                .collect();
                            match embed_texts(&models, texts).await {
                                Ok(vectors) if vectors.len() == chunks.len() => {
                                    if insert_file_chunks(
                                        &table,
                                        &mut new_manifest,
                                        &discovered_map,
                                        file,
                                        chunks,
                                        &vectors,
                                    )
                                    .await
                                    {
                                        embedded += 1;
                                        embedded_chunks += chunks.len();
                                        emit(
                                            events,
                                            IndexEvent::FileIndexed {
                                                path: RepoPath::from(file.clone()),
                                                chunks: ChunkCount::new(chunks.len()),
                                            },
                                        );
                                    }
                                }
                                Ok(_) => note!(
                                    "  (retrieval: embedding count mismatch for {file}; skipping)"
                                ),
                                Err(e) => {
                                    note!("  (retrieval: embedding {file} failed: {e})")
                                }
                            }
                        }
                    }
                }
            }
            note!(
                "  retrieval: embedded {embedded}/{} changed files ({embedded_chunks} chunks) in {:.1}s",
                to_embed.len(),
                embed_started.elapsed().as_secs_f64(),
            );
        } else {
            note!("  retrieval: manifest cache hit — no re-embedding");
        }
        // The vector index needs data to train: create it after the first
        // successful inserts (tolerant — on later runs it already exists, and
        // if it is ever missing, dense search degrades to flat scan).
        //
        // R2: `Index::Auto` on a vector column is LanceDB's **IVF_PQ** — an
        // approximate index whose k-means training pass is randomized, so two
        // builds of the same table rank differently and a re-created index over
        // a churned table measured 0.7222 vs 0.9630 reranked recall. Under
        // `--retrieval-vector-index exact` (the default) the dense query
        // bypasses the index entirely, so there is nothing to train and the
        // index is not built at all. `auto` restores the historic path.
        if embedded > 0 && cfg.retrieval_vector_index.uses_ann_index() {
            if let Err(e) = table
                .create_index(&["vector"], lancedb::index::Index::Auto)
                .execute()
                .await
            {
                note!("  (retrieval: vector index creation failed: {e})");
            }
        } else if embedded > 0 {
            note!(
                "  retrieval: vector index skipped (--retrieval-vector-index exact: dense search \
                 scans every stored vector)"
            );
        }
        save_manifest(cfg, &new_manifest);
        emit(
            events,
            IndexEvent::ManifestSaved {
                files: FileCount::new(new_manifest.len()),
            },
        );
        let started = std::time::Instant::now();
        // In-memory chunk map for the exact-identifier lexical layer
        // (vectors dropped: they live in LanceDB). Each chunk carries the same
        // crate stamp the stored rows got, so the lexical layer can be scoped
        // identically.
        let mut by_id = HashMap::new();
        for (path, text) in &contents {
            let hash = file_hash_of(text);
            let (crate_name, crate_dir) = scope_of(path);
            for mut chunk in chunks_for_entry(
                cfg,
                path,
                text,
                hash.as_str(),
                document_engines.get(path).copied(),
                &crate_name,
                &crate_dir,
            ) {
                chunk.crate_name = CrateName::from(crate_name.clone());
                chunk.vector = Embedding::default();
                by_id.insert(chunk.id.clone(), chunk);
            }
        }

        note!(
            "  retrieval: {} code chunks indexed across {} crate(s) under {} (provider: fastembed, local hybrid: {} + Tantivy FTS + {}, tool: search_vector_store)",
            by_id.len(),
            by_id
                .values()
                .map(|c| c.crate_name.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            cfg.workspace_root.display(),
            cfg.retrieval_embed_model,
            cfg.retrieval_rerank_model,
        );
        emit(
            events,
            IndexEvent::Done {
                embedded: FileCount::new(embedded),
                dropped: FileCount::new(to_drop.len()),
                indexed: ChunkCount::new(by_id.len()),
                elapsed_ms: DurationMs::new(started.elapsed().as_millis() as u64),
            },
        );
        Some(Arc::new(Self {
            table,
            by_id,
            models,
            rerank_enabled: cfg.retrieval_rerank_enabled,
            rerank_context: cfg.retrieval_rerank_context,
            fusion_gate: cfg.retrieval_fusion_gate,
            query_log: cfg.retrieval_query_log.clone(),
            vector_index: cfg.retrieval_vector_index,
            stage_candidates: cfg.retrieval_stage_candidates,
            rerank_candidates: cfg.retrieval_rerank_candidates,
            fusion: cfg.retrieval_fusion,
            rerank_score_norm: cfg.retrieval_rerank_score,
            rerank_blend: cfg.retrieval_rerank_blend.clamp_to(0.0, 1.0),
            query_expansion: cfg.retrieval_query_expansion,
            query_routing: cfg.retrieval_query_routing,
            secondary: None,
            dual_route: cfg.retrieval_dual_route,
        }))
    }

    /// Dense top-K `(id, L2 distance)` pairs from LanceDB (lower distance =
    /// more similar), optionally confined to one crate.
    ///
    /// With `--retrieval-vector-index exact` (the default) the query bypasses
    /// any trained vector index and scans every stored row with exact distances.
    /// That is deterministic across builds and makes the crate `only_if` filter
    /// an exact prefilter rather than an ANN postfilter that can shrink the
    /// candidate pool under the `nprobes` limit.
    /// Best-effort opt-in JSONL query log (no source content).
    fn log_query(
        &self,
        query: &str,
        scope: Option<&str>,
        samples: usize,
        started: std::time::Instant,
        chunks: &[(LocalChunk, Score)],
    ) {
        let Some(path) = self.query_log.as_deref() else {
            return;
        };
        let mut seen = std::collections::HashSet::new();
        let paths: Vec<String> = chunks
            .iter()
            .filter(|(chunk, _)| seen.insert(chunk.path.as_str()))
            .map(|(chunk, _)| chunk.path.as_str().to_string())
            .collect();
        let line = query_log_line(
            query,
            scope,
            samples,
            started.elapsed().as_millis() as u64,
            &paths,
        );
        append_query_log(path, &line);
    }

    async fn dense_top_scored(
        &self,
        query_vec: &[f32],
        limit: usize,
        scope: Option<&str>,
    ) -> Result<Vec<(String, f32)>, String> {
        let mut query = self
            .table
            .query()
            .nearest_to(query_vec.to_vec())
            .map_err(|e| e.to_string())?
            .limit(limit);
        if !self.vector_index.uses_ann_index() {
            query = query.bypass_vector_index();
        }
        if let Some(predicate) = crate_predicate(scope) {
            query = query.only_if(predicate);
        }
        let stream = query.execute().await.map_err(|e| e.to_string())?;
        collect_scored(stream, "_distance").await
    }

    /// FTS top-K `(id, BM25 score)` pairs from the Tantivy index (higher
    /// score = more relevant), optionally confined to one crate.
    async fn fts_top_scored(
        &self,
        query: &str,
        limit: usize,
        scope: Option<&str>,
    ) -> Result<Vec<(String, f32)>, String> {
        let mut builder = self
            .table
            .query()
            .full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(
                query.to_string(),
            ))
            .limit(limit);
        if let Some(predicate) = crate_predicate(scope) {
            builder = builder.only_if(predicate);
        }
        let stream = builder.execute().await.map_err(|e| e.to_string())?;
        collect_scored(stream, "_score").await
    }

    /// Two-stage hybrid search: dense + FTS → score-aware fusion →
    /// cross-encoder → top-K. Returns (chunk, score) pairs ordered best-first.
    /// `scope` confines both stages to one crate's chunks (the LanceDB
    /// `crate_name` column); `None` searches the whole workspace.
    /// `dense_weight` is the dense/FTS fusion ratio
    /// ([`fuse_scored_pairs`]); `1.0` is the historic 1:1 sum.
    ///
    /// Retained as the plain entry point the tool used before E8; the tool now
    /// goes through [`LocalCodeIndex::hybrid_search_with_stats`] so a run can
    /// account for what each stage did. Callers wanting the untraced results
    /// only (and not the telemetry) can still use this.
    // Nothing in the crate calls it any more — but no test can construct a
    // `LocalCodeIndex` (it needs the GPU backend and LanceDB), so there is no
    // in-repo caller to keep it honest. Kept because it is the documented
    // plain-result contract the stats-bearing variant must stay byte-identical
    // to, and it is the seam a future caller outside the index would use.
    #[allow(dead_code)]
    pub async fn hybrid_search(
        &self,
        query: &str,
        samples: usize,
        scope: Option<&str>,
        dense_weight: Weight,
    ) -> Result<Vec<(LocalChunk, Score)>, String> {
        let secondary = self
            .secondary
            .as_ref()
            .filter(|_| dual_route_applies(true, self.dual_route.is_on(), query));
        if let Some(secondary) = secondary {
            return Box::pin(secondary.hybrid_search(query, samples, scope, dense_weight)).await;
        }
        Ok(self
            .hybrid_search_core(query, samples, scope, dense_weight)
            .await?
            .chunks)
    }

    /// The one hybrid pipeline both the traced and the stats-bearing entry
    /// points project from.
    ///
    /// `dense_weight` is the dense/FTS fusion ratio ([`fuse_scored_pairs`]);
    /// every entry point threads it through so the trace, the counters, and the
    /// untraced result all describe the same fusion.
    ///
    /// The FTS failure is swallowed into `fts_ok = false` (with the existing
    /// best-effort announcement) so a caller can *observe* the degradation
    /// rather than infer it from an empty list that looks identical to "BM25
    /// matched nothing".
    async fn hybrid_search_core(
        &self,
        query: &str,
        samples: usize,
        scope: Option<&str>,
        dense_weight: Weight,
    ) -> Result<HybridSearchCore, String> {
        let started = std::time::Instant::now();
        let dense_query = if self.query_expansion.is_on() {
            format!("{query} {}", crate::retrieval::fts_query_text(query))
        } else {
            query.to_string()
        };
        let mut query_vec = self
            .models
            .embed_query(dense_query)
            .await
            .map_err(|e| e.to_string())?;
        // Defensive L2-normalization so a degenerate (zero/NaN) embedding can
        // never reach `nearest_to` and produce NaN distances. The embedder
        // embedder returns normalized vectors, so this is a no-op in the
        // common case; on failure we keep the raw vector and let dense search
        // run best-effort.
        let _ = l2_normalize(&mut query_vec);

        let dense = self
            .dense_top_scored(&query_vec, self.stage_candidates.get(), scope)
            .await?;
        // FTS is a best-effort layer: if the index is missing/broken we still
        // serve dense results. Both stages consume the SAME tokenization
        // (P2#8), and since E3/B that tokenization is the configured one: the
        // query text is produced by the same knob that chose the indexed
        // column, so the two sides can never drift.
        let fts_query = fts_query_for(query);
        let (fts, fts_ok) = match self
            .fts_top_scored(&fts_query, self.stage_candidates.get(), scope)
            .await
        {
            Ok(rows) => (rows, true),
            Err(e) => {
                // Best-effort layer: a missing/broken FTS index still serves
                // dense results. The failure is announced because an empty FTS
                // stage is indistinguishable from "FTS found nothing", and the
                // retrieval eval must never read a broken stage as a ranking.
                eprintln!("  (retrieval: FTS stage failed, dense-only for this query: {e})");
                (Vec::new(), false)
            }
        };

        // One shared candidate vector: the fused order is computed once and the
        // reranked list is a projection of it, so the trace's `fused` list, the
        // pool the cross-encoder actually saw, and the stats' delivered pool
        // indices cannot diverge.
        let lexical_rescue =
            self.fusion_gate.is_on() && lexical_stage_adds_candidates(&dense, &fts);
        let routed = self.query_routing.is_on() && is_symbol_shaped_query(query);
        let fusion = if lexical_rescue || routed {
            Fusion::Rrf
        } else {
            self.fusion
        };
        let fused_pairs = fuse_with_mode(fusion, &dense, &fts, dense_weight.get());
        let pool: Vec<(String, f32)> = fused_pairs
            .iter()
            .filter(|(id, _)| self.by_id.contains_key(id.as_str()))
            .take(self.rerank_candidates.get())
            .cloned()
            .collect();
        let candidates: Vec<LocalChunk> = pool
            .iter()
            .filter_map(|(id, _)| self.by_id.get(id.as_str()).cloned())
            .collect();
        if candidates.is_empty() {
            return Ok(HybridSearchCore {
                chunks: Vec::new(),
                dense,
                fts,
                pool,
                reranked_pairs: Vec::new(),
                fts_ok,
            });
        }

        // Reranking is optional: when off, deliver the fused order directly
        // and make no rerank request.
        if !self.rerank_enabled.is_on() {
            let reranked_pairs: Vec<(String, f32)> = pool
                .iter()
                .take(samples.min(self.rerank_candidates.get()))
                .cloned()
                .collect();
            let chunks: Vec<(LocalChunk, Score)> = reranked_pairs
                .iter()
                .filter_map(|(id, score)| {
                    self.by_id
                        .get(id.as_str())
                        .cloned()
                        .map(|c| (c, Score::new(*score)))
                })
                .collect();
            self.log_query(query, scope, samples, started, &chunks);
            return Ok(HybridSearchCore {
                chunks,
                dense,
                fts,
                pool,
                reranked_pairs,
                fts_ok,
            });
        }
        let docs: Vec<String> = if self.rerank_context.is_on() {
            candidates
                .iter()
                .map(|c| {
                    let module = if c.crate_name.is_empty() {
                        module_hint(c.path.as_str())
                    } else {
                        format!("{}::{}", c.crate_name, module_hint(c.path.as_str()))
                    };
                    format!("Module: {module}\n{}", c.content)
                })
                .collect()
        } else {
            candidates
                .iter()
                .map(|c| c.content.as_str().to_string())
                .collect()
        };
        let mut ranked = self.models.rerank(query.to_string(), docs.clone()).await?;
        // R10's jina finding: a reranker that returns a raw similarity is
        // unusable here without a transform, because the only normalization
        // left downstream is a clamp that flattens every negative score. The
        // configured mode is applied in one place, before the ranking is
        // decided, so a delivered score can never disagree with the position it
        // was ranked at.
        normalize_rerank_scores(&mut ranked, self.rerank_score_norm);
        let blend = self.rerank_blend.get();
        if blend > 0.0 {
            let fused: std::collections::HashMap<&str, f32> = pool
                .iter()
                .map(|(id, score)| (id.as_str(), *score))
                .collect();
            let mut fused_vals: Vec<f32> = candidates
                .iter()
                .map(|c| *fused.get(c.id.as_str()).unwrap_or(&0.0))
                .collect();
            let (lo, hi) = fused_vals
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(*v), hi.max(*v))
                });
            let span = (hi - lo).max(f32::EPSILON);
            for v in fused_vals.iter_mut() {
                *v = (*v - lo) / span;
            }
            for (index, score) in ranked.iter_mut() {
                let fused = fused_vals.get(*index).copied().unwrap_or(0.0);
                *score = blend * *score + (1.0 - blend) * fused;
            }
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        }
        // Single retained cap: final results = min(user samples, rerank pool).
        let top = rerank_top(ranked, samples.min(self.rerank_candidates.get()));
        // fastembed returns relevance scores already in [0, 1] (sigmoid applied
        // server-side), so there is no extra sigmoid fold here; the canonical
        // clamp in `retrieval.rs` is the only normalization left.
        let reranked_pairs: Vec<(String, f32)> = top
            .iter()
            .filter_map(|(idx, score)| {
                candidates
                    .get(*idx)
                    .map(|c| (c.id.as_str().to_string(), *score))
            })
            .collect();
        let chunks: Vec<(LocalChunk, Score)> = top
            .into_iter()
            .filter_map(|(idx, score)| candidates.get(idx).cloned().map(|c| (c, Score::new(score))))
            .collect();
        self.log_query(query, scope, samples, started, &chunks);
        Ok(HybridSearchCore {
            chunks,
            dense,
            fts,
            pool,
            reranked_pairs,
            fts_ok,
        })
    }
}

/// Collect the `id` + score columns of a record-batch stream in order.
/// `score_col` is `_distance` (vector, lower=better) or `_score` (FTS,
/// higher=better).
async fn collect_scored(
    stream: impl futures::Stream<Item = Result<arrow_array::RecordBatch, lancedb::Error>>,
    score_col: &str,
) -> Result<Vec<(String, f32)>, String> {
    use arrow_array::Array;
    use futures::StreamExt;
    let mut out = Vec::new();
    futures::pin_mut!(stream);
    while let Some(batch) = stream.next().await {
        let batch = batch.map_err(|e| e.to_string())?;
        let ids = batch
            .column_by_name("id")
            .ok_or_else(|| "query result missing id column".to_string())?;
        let ids = ids
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| "id column is not a string array".to_string())?;
        let scores = batch
            .column_by_name(score_col)
            .ok_or_else(|| format!("query result missing {score_col} column"))?;
        let scores = scores
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .ok_or_else(|| format!("{score_col} column is not a float32 array"))?;
        for i in 0..ids.len() {
            out.push((ids.value(i).to_string(), scores.value(i)));
        }
    }
    Ok(out)
}

/// Escape a string for a LanceDB SQL predicate literal (single quotes are
/// doubled). Pure.
pub fn sql_string_literal(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// The LanceDB predicate confining a query to one crate's chunks. `None` (no
/// scope) means no predicate — the whole-workspace search. Pure.
pub fn crate_predicate(scope: Option<&str>) -> Option<String> {
    scope.map(|name| format!("crate_name = {}", sql_string_literal(name)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RetrievalVectorIndex;
    use crate::types::LanguageId;

    fn sample_text() -> &'static str {
        "fn one() {}\nfn two() {}\nfn three() {}\nfn four() {}\nfn five() {}\nfn six() {}\nfn seven() {}"
    }

    #[test]
    fn xberg_chunks_map_onto_local_chunk_shape() {
        // R5: the xberg chunker must produce the same LocalChunk contract as
        // the legacy window: File: prefix, non-empty content, 1-based line
        // anchors, deterministic ids — and never fail (any xberg error
        // degrades to the legacy window).
        let text = "# Alpha\n\n## Beta\nfn beta() {}\n\n## Gamma\nfn gamma() {}\n\nfn delta() {}\n";
        let hash = "h";
        let chunks = xberg_chunks("src/lib.rs", text, hash, 3);
        assert!(!chunks.is_empty(), "xberg produced no chunks");
        for c in &chunks {
            assert!(
                c.content.starts_with("File: src/lib.rs"),
                "missing File: prefix"
            );
            assert!(!c.content.is_empty());
            assert!(c.start_line >= 1, "line anchors must be 1-based");
            assert_eq!(c.file_hash, hash);
            assert_eq!(c.language, language_of("src/lib.rs"));
        }
        // Deterministic: same input -> same ids/content.
        let again = xberg_chunks("src/lib.rs", text, hash, 3);
        assert_eq!(
            chunks
                .iter()
                .map(|c| (&c.id, &c.content))
                .collect::<Vec<_>>(),
            again
                .iter()
                .map(|c| (&c.id, &c.content))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn symbol_chunks_map_onto_local_chunk_shape() {
        // R8: the symbol chunker must produce the same LocalChunk contract as
        // xberg: File: prefix, non-empty content, 1-based line anchors,
        // deterministic ids, file_hash and language.
        let text = "use a::b;\n\npub fn alpha() {\n    let x = 1;\n}\n\nstruct Beta;\n";
        let hash = "h";
        let chunks = symbol_chunks("src/lib.rs", text, hash, 50);
        assert!(!chunks.is_empty(), "symbol produced no chunks");
        for c in &chunks {
            assert!(
                c.content.starts_with("File: src/lib.rs"),
                "missing File: prefix: {}",
                c.content
            );
            assert!(!c.content.is_empty());
            assert!(c.start_line >= 1, "line anchors must be 1-based");
            assert_eq!(c.file_hash, hash);
            assert_eq!(c.language, language_of("src/lib.rs"));
        }
        let again = symbol_chunks("src/lib.rs", text, hash, 50);
        assert_eq!(
            chunks
                .iter()
                .map(|c| (&c.id, &c.content))
                .collect::<Vec<_>>(),
            again
                .iter()
                .map(|c| (&c.id, &c.content))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn symbol_chunks_split_long_items_without_truncating() {
        let mut body = String::from("fn big() {\n");
        for i in 0..120 {
            body.push_str(&format!("    let value_{i} = compute({i});\n"));
        }
        body.push_str("    let marker = LAST_BODY_LINE;\n}\n");
        let chunks = symbol_chunks("src/big.rs", &body, "h", 50);
        assert!(chunks.len() >= 2, "a 122-line item must split: {chunks:?}");
        let budget = chunk_char_budget(EMBEDDER_MAX_TOKENS);
        for c in &chunks {
            assert!(
                !c.content.contains(CHUNK_TRUNCATION_MARKER),
                "the size policy, not the lossy cap, must bound this corpus: {}",
                c.content
            );
            assert!(
                c.content.chars().count() <= budget,
                "chunk over the measured budget ({} > {budget})",
                c.content.chars().count()
            );
            // Every window repeats the item header so a symbol query hits all
            // of them.
            assert!(
                c.content.contains("fn big()"),
                "window lost the header: {}",
                c.content
            );
        }
        assert!(
            chunks.last().unwrap().content.contains("LAST_BODY_LINE"),
            "the item's last body line must reach the corpus"
        );
    }

    #[test]
    fn symbol_chunks_fall_back_to_xberg_for_unsegmentable_text() {
        let text = "just prose\nwith no items\n";
        let symbol = symbol_chunks("src/x.rs", text, "h", 3);
        let xberg = xberg_chunks("src/x.rs", text, "h", 3);
        assert_eq!(
            symbol.iter().map(|c| c.content.clone()).collect::<Vec<_>>(),
            xberg.iter().map(|c| c.content.clone()).collect::<Vec<_>>(),
            "an unrecognized segmentation must fall back, not lose the file"
        );
    }

    #[test]
    fn symbol_chunks_empty_input_is_empty() {
        assert!(symbol_chunks("src/x.rs", "", "h", 50).is_empty());
        assert!(symbol_chunks("src/x.rs", "   \n\n ", "h", 50).is_empty());
    }

    #[test]
    fn sliding_window_chunks_lines_overlap_and_anchors() {
        // 7 lines, 4-line windows, 3-line overlap → step 1 → 4 windows
        // (lines 1-4, 2-5, 3-6, 4-7).
        let chunks = sliding_window_chunks("src/auth.rs", sample_text(), "hash123", 4);
        assert_eq!(chunks.len(), 4);
        let c0 = &chunks[0];
        assert_eq!(c0.start_line, 1);
        assert_eq!(c0.chunk_index, 0);
        assert!(c0.content.starts_with("File: src/auth.rs\n\nfn one()"));
        assert!(c0.content.contains("fn four()"));
        let c1 = &chunks[1];
        assert_eq!(c1.start_line, 2, "overlap: second window starts at line 2");
        let last = &chunks[3];
        assert_eq!(last.start_line, 4);
        assert!(last.content.contains("fn seven()"));
        // ids are stable and distinct.
        assert_ne!(c0.id, c1.id);
        assert_eq!(c0.id, chunk_id("src/auth.rs", 0, "hash123"));
        assert_eq!(c0.language, "rs");
        assert!(c0.vector.is_empty());
    }

    #[test]
    fn sliding_window_chunks_empty_and_short_inputs() {
        assert!(sliding_window_chunks("a.rs", "", "h", 50).is_empty());
        let one = sliding_window_chunks("a.rs", "only one line", "h", 50);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].start_line, 1);
    }

    #[test]
    fn chunk_content_is_capped_to_the_token_budget() {
        let max_chars = chunk_char_budget(EMBEDDER_MAX_TOKENS);
        let huge = "y".repeat(max_chars * 2);
        let chunks = sliding_window_chunks("big.rs", &huge, "h", 50);
        let suffix = "… [chunk truncated]";
        assert!(
            chunks
                .iter()
                .all(|c| c.content.chars().count() <= max_chars + suffix.chars().count())
        );
        assert!(chunks[0].content.ends_with(suffix));
    }

    #[test]
    fn files_to_reindex_diff_matrix() {
        let mut manifest = HashMap::new();
        manifest.insert("src/a.rs".to_string(), "h1".to_string());
        manifest.insert("src/gone.rs".to_string(), "h2".to_string());
        let discovered = vec![
            ("src/a.rs".to_string(), "h9".to_string()),
            ("src/b.rs".to_string(), "h3".to_string()),
        ];
        let (to_embed, to_drop) = files_to_reindex(&manifest, &discovered);
        assert_eq!(
            to_embed,
            vec!["src/a.rs".to_string(), "src/b.rs".to_string()]
        );
        assert_eq!(to_drop, vec!["src/gone.rs".to_string()]);
        // After the manifest catches up with the corpus: no work at all.
        let updated: HashMap<String, String> = discovered.iter().cloned().collect();
        let (embed, drop) = files_to_reindex(&updated, &discovered);
        assert!(embed.is_empty() && drop.is_empty());
    }

    /// The fixture the historic fusion test pinned: `b` in both stages, `c` an
    /// FTS-only hit with a strong BM25, `a` a weak dense-only hit.
    type Ranked = Vec<(String, f32)>;

    fn fusion_fixture() -> (Ranked, Ranked) {
        let dense = vec![
            ("a".to_string(), 10.0f32), // dense-only, far (weak)
            ("b".to_string(), 1.0f32),  // dense #2
        ];
        let fts = vec![
            ("c".to_string(), 100.0f32), // FTS-only, very high BM25
            ("b".to_string(), 5.0f32),   // also in dense
        ];
        (dense, fts)
    }

    #[test]
    fn fuse_scored_uses_scores_and_boosts_both_stages() {
        // b appears in both stages → highest. At the default weight 1.0, c's
        // strong FTS score (~0.99) outranks a's weak dense-only similarity
        // (~0.09), which a rank-only RRF would have promoted to #1 purely by
        // dense position. This is the *shipped* ordering: the knob's default
        // must reproduce it exactly.
        let (dense, fts) = fusion_fixture();
        assert_eq!(
            fuse_scored(&dense, &fts, 1.0),
            vec!["b".to_string(), "c".to_string(), "a".to_string()]
        );
    }

    /// Task E8: the per-call stage accounting must be pure, exact, and
    /// reachable without the GPU backend/LanceDB (the pipeline itself is not
    /// testable
    /// here — see `E8`'s known gap).
    fn stats_for(
        dense: &[&str],
        fts: &[&str],
        rerank_pool: usize,
        delivered: &[usize],
        fts_ok: bool,
    ) -> SearchStageStats {
        let pair = |ids: &[&str]| -> Vec<(String, f32)> {
            ids.iter()
                .enumerate()
                .map(|(i, id)| ((*id).to_string(), i as f32))
                .collect()
        };
        search_stage_stats(&pair(dense), &pair(fts), rerank_pool, delivered, fts_ok)
    }

    #[test]
    fn search_stage_stats_counts_each_stage_and_the_fused_union() {
        // a dense-only, b both, c fts-only → 3 distinct fused, 1 shared.
        let s = stats_for(&["a", "b"], &["c", "b"], 2, &[0, 1], true);
        assert_eq!(s.dense_candidates, 2);
        assert_eq!(s.fts_candidates, 2);
        assert_eq!(s.fused_candidates, 3);
        assert_eq!(s.both_stages, 1);
        assert_eq!(s.rerank_pool, 2);
        assert_eq!(s.rerank_promotions, 0);
        assert!(!s.fts_degraded.is_on());
    }

    #[test]
    fn search_stage_stats_is_all_zero_with_no_candidates() {
        let s = stats_for(&[], &[], 7, &[], true);
        assert_eq!(
            s,
            SearchStageStats {
                rerank_pool: CandidateCount::new(7),
                ..SearchStageStats::default()
            }
        );
        assert_eq!(s, stats_for(&[], &[], 7, &[], true));
    }

    #[test]
    fn search_stage_stats_reports_fts_degradation_instead_of_an_empty_list() {
        // An FTS error yields an empty stage list; `fts_degraded` is what makes
        // that observable rather than indistinguishable from "found nothing".
        let s = stats_for(&["a", "b"], &[], 2, &[0, 1], false);
        assert_eq!(s.dense_candidates, 2);
        assert_eq!(s.fts_candidates, 0);
        assert_eq!(s.fused_candidates, 2);
        assert_eq!(s.both_stages, 0);
        assert!(s.fts_degraded.is_on());
    }

    #[test]
    fn search_stage_stats_counts_duplicate_ids_once() {
        let s = stats_for(&["a", "a", "b"], &["b", "b"], 2, &[], true);
        assert_eq!(s.dense_candidates, 3);
        assert_eq!(s.fts_candidates, 2);
        assert_eq!(s.fused_candidates, 2);
        assert_eq!(s.both_stages, 1);
    }

    #[test]
    fn search_stage_stats_counts_promotions_from_outside_the_delivered_count() {
        // Delivered count k == 5: indices 0,1,2 are in place; 7 and 9 were at
        // or beyond k in the fused order and only the reranker surfaced them.
        let s = stats_for(&["a"], &[], 9, &[0, 1, 2, 7, 9], true);
        assert_eq!(s.rerank_pool, 9);
        assert_eq!(s.rerank_promotions, 2);
        // A no-op reranker (delivered == [0..k)) promotes nothing.
        assert_eq!(
            stats_for(&[], &[], 5, &[0, 1, 2], true).rerank_promotions,
            0
        );
        // No delivered chunks at all.
        assert_eq!(stats_for(&[], &[], 5, &[], true).rerank_promotions, 0);
    }

    #[test]
    fn fuse_scored_weight_moves_a_dense_hit_past_an_fts_only_hit() {
        // dense a = 1/(1+9) = 0.1 (L2 distance 9); fts c = 1/(1+1) = 0.5.
        // At weight 1.0 the FTS-only hit wins; at 6.0 the dense hit does
        // (0.6 > 0.5). Directional, and monotone in the weight.
        let dense = vec![("a".to_string(), 9.0f32)];
        let fts = vec![("c".to_string(), 1.0f32)];
        assert_eq!(
            fuse_scored(&dense, &fts, 1.0),
            vec!["c".to_string(), "a".to_string()]
        );
        assert_eq!(
            fuse_scored(&dense, &fts, 6.0),
            vec!["a".to_string(), "c".to_string()]
        );
        // The historic fixture flips `a` past `c` too, only at a much higher
        // weight: a's 0.0909·w must clear c's 0.9901.
        let (dense, fts) = fusion_fixture();
        assert_eq!(
            fuse_scored(&dense, &fts, 11.0),
            vec!["b".to_string(), "a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn fuse_scored_zero_weight_ranks_by_fts_alone_without_nan() {
        // At weight 0.0 the dense contribution is zero, so the FTS score alone
        // decides: `c` (BM25 100) then `b` (BM25 5), and the dense-only `a`
        // sinks last with a zero score. Appearing in both stages earns nothing
        // extra at a zero dense weight — the sum is weighted, not a coverage
        // bonus — which is exactly the difference from weight 1.0, where `b`
        // (both stages) outranks `c` (FTS only). No score is NaN, even for a
        // NaN distance.
        let (dense, fts) = fusion_fixture();
        let pairs = fuse_scored_pairs(&dense, &fts, 0.0);
        for (id, score) in &pairs {
            assert!(score.is_finite(), "{id} has a non-finite fused score");
        }
        let ids: Vec<String> = pairs.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids, vec!["c".to_string(), "b".to_string(), "a".to_string()]);
        // The same fixture at weight 1.0 is the opposite order: `b` is found by
        // BOTH stages, so the dense contribution lifts it above the FTS-only
        // `c`. That contrast is the whole reason the weight is configurable.
        let at_one: Vec<String> = fuse_scored_pairs(&dense, &fts, 1.0)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(at_one.first().map(String::as_str), Some("b"));
        // `a` is last (dense-only → 0.0), after both FTS contributors.
        assert_eq!(pairs.last().map(|(id, _)| id.as_str()), Some("a"));
        assert_eq!(pairs.last().map(|(_, s)| *s), Some(0.0));
        // A NaN distance degrades to similarity 0.0, never NaN.
        let nan_dense = vec![("x".to_string(), f32::NAN)];
        let nan_pairs = fuse_scored_pairs(&nan_dense, &[], 0.0);
        assert_eq!(nan_pairs.len(), 1);
        assert!(nan_pairs[0].1.is_finite());
    }

    #[test]
    fn fuse_scored_single_stage_order_is_weight_invariant() {
        // With one stage empty the surviving stage's own order must be
        // identical at every weight: at 0.0 the stage-order tiebreak carries it.
        let dense = vec![
            ("d1".to_string(), 0.5f32),
            ("d2".to_string(), 2.0f32),
            ("d3".to_string(), 7.0f32),
        ];
        let fts = vec![
            ("f1".to_string(), 9.0f32),
            ("f2".to_string(), 2.0f32),
            ("f3".to_string(), 0.1f32),
        ];
        for weight in [0.0, 1.0, 3.0] {
            assert_eq!(
                fuse_scored(&dense, &[], weight),
                vec!["d1".to_string(), "d2".to_string(), "d3".to_string()],
                "dense-only order at weight {weight}"
            );
            assert_eq!(
                fuse_scored(&[], &fts, weight),
                vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
                "fts-only order at weight {weight}"
            );
        }
    }

    #[test]
    fn fuse_scored_handles_empty_lists() {
        assert!(fuse_scored(&[], &[], 1.0).is_empty());
        assert_eq!(
            fuse_scored(&[("a".to_string(), 0.5)], &[], 1.0),
            vec!["a".to_string()]
        );
        assert_eq!(
            fuse_scored(&[], &[("a".to_string(), 3.0)], 1.0),
            vec!["a".to_string()]
        );
    }

    #[test]
    fn rerank_score_norm_parses_labels_and_rejects_an_unknown_value() {
        assert_eq!(RerankScoreNorm::default(), RerankScoreNorm::Clamp);
        assert_eq!(RerankScoreNorm::parse("clamp"), Ok(RerankScoreNorm::Clamp));
        assert_eq!(
            RerankScoreNorm::parse("sigmoid"),
            Ok(RerankScoreNorm::Sigmoid)
        );
        assert_eq!(
            RerankScoreNorm::parse("minmax"),
            Ok(RerankScoreNorm::MinMax)
        );
        assert_eq!(RerankScoreNorm::MinMax.label(), "minmax");
        let err = RerankScoreNorm::parse("softmax").unwrap_err();
        assert!(err.contains("clamp | sigmoid | minmax"), "{err}");
    }

    #[test]
    fn a_raw_similarity_reranker_keeps_its_order_under_sigmoid_or_minmax() {
        // Exactly the jina case: a late-interaction cosine in [-1, 1], observed
        // -0.1856 … 0.1920. The downstream clamp would flatten every negative
        // to 0.0 and destroy the ranking, which is why the knob exists.
        let raw = vec![(0usize, -0.1856f32), (1, -0.02), (2, 0.04), (3, 0.1920)];
        // The ranking the pipeline would deliver: descending by score, with the
        // input order breaking ties (which is what a stable sort does, and what
        // `rerank_top` does on the real path).
        let by_score = |v: &[(usize, f32)]| {
            let mut idx = v.to_vec();
            idx.sort_by(|a, b| b.1.total_cmp(&a.1));
            idx.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        };
        assert_eq!(by_score(&raw), vec![3, 2, 1, 0]);

        let mut clamped = raw.clone();
        normalize_rerank_scores(&mut clamped, RerankScoreNorm::Clamp);
        assert_eq!(clamped, raw, "clamp is the pass-through");
        let down: Vec<(usize, f32)> = clamped
            .iter()
            .map(|(i, s)| (*i, s.clamp(0.0, 1.0)))
            .collect();
        assert_eq!(
            down.iter().filter(|(_, s)| *s == 0.0).count(),
            2,
            "the motivating defect: two of four collapse to 0.0"
        );
        assert_eq!(
            by_score(&down),
            vec![3, 2, 0, 1],
            "and the clamp reorders the surviving tie — the ranking is lost"
        );

        let mut sig = raw.clone();
        normalize_rerank_scores(&mut sig, RerankScoreNorm::Sigmoid);
        assert_eq!(by_score(&sig), vec![3, 2, 1, 0], "the order survives");
        assert!(sig.iter().all(|(_, s)| (0.0..=1.0).contains(s)));
        assert!(sig[0].1 < 0.5 && sig[3].1 > 0.5);

        let mut mm = raw.clone();
        normalize_rerank_scores(&mut mm, RerankScoreNorm::MinMax);
        assert_eq!(by_score(&mm), vec![3, 2, 1, 0], "the order survives");
        assert_eq!(mm[0].1, 0.0, "the worst maps to 0");
        assert_eq!(mm[3].1, 1.0, "the best maps to 1");
    }

    #[test]
    fn minmax_with_no_span_ties_every_candidate() {
        // One candidate, or all alike: there is no scale to recover, so nothing
        // may be called the best.
        let mut one = vec![(0usize, -0.3f32)];
        normalize_rerank_scores(&mut one, RerankScoreNorm::MinMax);
        assert_eq!(one[0].1, 0.5);
        let mut same = vec![(0usize, 0.7f32), (1, 0.7), (2, 0.7)];
        normalize_rerank_scores(&mut same, RerankScoreNorm::MinMax);
        assert!(same.iter().all(|(_, s)| *s == 0.5));
        // Sigmoid maps identical inputs to identical outputs — no ordering
        // claim either way — and never leaves [0, 1].
        let mut sig = vec![(0usize, 0.7f32), (1, 0.7)];
        normalize_rerank_scores(&mut sig, RerankScoreNorm::Sigmoid);
        assert_eq!(sig[0].1, sig[1].1);
        assert!((0.0..=1.0).contains(&sig[0].1));
    }

    #[test]
    fn fusion_parses_labels_and_rejects_an_unknown_value() {
        assert_eq!(Fusion::default(), Fusion::Weighted);
        assert_eq!(Fusion::parse("weighted"), Ok(Fusion::Weighted));
        assert_eq!(Fusion::parse("rrf"), Ok(Fusion::Rrf));
        assert_eq!(Fusion::Weighted.label(), "weighted");
        assert_eq!(Fusion::Rrf.label(), "rrf");
        let err = Fusion::parse("reciprocal").unwrap_err();
        assert!(err.contains("weighted | rrf"), "{err}");
    }

    /// The ids `fuse_rrf_pairs` produces, in order — the projection the tests
    /// read, kept local so the module exports only the pair-producing function.
    fn rrf_ids(dense: &[(String, f32)], fts: &[(String, f32)]) -> Vec<String> {
        fuse_rrf_pairs(dense, fts)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    #[test]
    fn fuse_rrf_ignores_scores_and_combines_both_orders() {
        // The dense list ranks `a` first; the FTS list ranks `b` first. The
        // scores are deliberately absurd (an FTS score of 10_000 on a dense
        // distance), because rank fusion must not read them at all.
        let dense = vec![("a".to_string(), 0.001), ("b".to_string(), 0.9)];
        let fts = vec![("b".to_string(), 10_000.0), ("a".to_string(), 1.0)];
        // Both ids appear in both lists and each is first once, so the sums tie
        // and the dense order breaks the tie.
        assert_eq!(
            rrf_ids(&dense, &fts),
            vec!["a".to_string(), "b".to_string()]
        );
        // Rank 1 in one list ties with rank 1 in the other (dense order breaks
        // it), and an id found only at rank 2 loses to one found at rank 1 —
        // being in a list matters less than where in it.
        let dense = vec![("a".to_string(), 0.0), ("b".to_string(), 0.0)];
        let fts = vec![("c".to_string(), 0.0)];
        assert_eq!(
            rrf_ids(&dense, &fts),
            vec!["a".to_string(), "c".to_string(), "b".to_string()]
        );
        // Empty lists are not errors.
        assert!(rrf_ids(&[], &[]).is_empty());
    }

    #[test]
    fn fuse_rrf_is_weight_invariant_and_fuse_with_mode_dispatches() {
        let dense = vec![("a".to_string(), 0.1), ("b".to_string(), 5.0)];
        let fts = vec![("c".to_string(), 2.0)];
        // The weight changes nothing under RRF: rank fusion has no weight, and
        // honouring one would make the recorded weight look like it shaped a
        // ranking it did not.
        let at_one = fuse_with_mode(Fusion::Rrf, &dense, &fts, 1.0);
        let at_huge = fuse_with_mode(Fusion::Rrf, &dense, &fts, 1000.0);
        assert_eq!(at_one, at_huge);
        assert_eq!(
            at_one.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["a", "c", "b"]
        );
        // The weighted path still honours it, so the dispatcher is not a no-op.
        let weighted_low = fuse_with_mode(Fusion::Weighted, &dense, &fts, 0.1);
        let weighted_high = fuse_with_mode(Fusion::Weighted, &dense, &fts, 100.0);
        assert_ne!(weighted_low, weighted_high);
    }

    #[test]
    fn fuse_scored_pairs_matches_fuse_scored_order_and_scores() {
        // The staged seam must not change fusion: `fuse_scored` is exactly the
        // ids of `fuse_scored_pairs`, in order, for every existing input.
        let cases: Vec<(Ranked, Ranked)> = vec![
            (vec![], vec![]),
            (
                vec![("a".to_string(), 10.0), ("b".to_string(), 1.0)],
                vec![("c".to_string(), 100.0), ("b".to_string(), 5.0)],
            ),
            (vec![("a".to_string(), 0.5)], vec![]),
            (vec![], vec![("a".to_string(), 3.0)]),
            // A tie across stages is broken by dense order, then FTS order.
            (
                vec![("x".to_string(), 1.0), ("y".to_string(), 1.0)],
                vec![("y".to_string(), 1.0), ("x".to_string(), 1.0)],
            ),
        ];
        for (dense, fts) in cases {
            let pairs = fuse_scored_pairs(&dense, &fts, 1.0);
            let ids: Vec<String> = pairs.iter().map(|(id, _)| id.clone()).collect();
            assert_eq!(ids, fuse_scored(&dense, &fts, 1.0));
            // The fused score is the normalized sum of each stage's contribution.
            let a = pairs
                .iter()
                .find(|(id, _)| id == "a")
                .or_else(|| pairs.iter().find(|(id, _)| id == "x"));
            if let Some((_, score)) = a {
                assert!(*score > 0.0 && score.is_finite());
            }
        }
    }

    #[test]
    fn project_stage_hits_skips_unknown_ids_and_preserves_order() {
        let ids = vec![
            ("known-1".to_string(), 0.25f32),
            ("missing".to_string(), 0.5f32),
            ("known-2".to_string(), 0.75f32),
        ];
        let lookup = |id: &str| match id {
            "known-1" => Some(("src/a.rs".to_string(), 1u32)),
            "known-2" => Some(("src/b.rs".to_string(), 40u32)),
            _ => None,
        };
        let hits = project_stage_hits(&ids, &lookup);
        assert_eq!(
            hits,
            vec![
                StageHit {
                    id: "known-1".into(),
                    path: "src/a.rs".into(),
                    start_line: LineNumber::new(1),
                    score: Score::new(0.25)
                },
                StageHit {
                    id: "known-2".into(),
                    path: "src/b.rs".into(),
                    start_line: LineNumber::new(40),
                    score: Score::new(0.75)
                },
            ],
            "an unknown id is skipped, never given a placeholder location"
        );
    }

    #[test]
    fn stage_trace_default_is_empty() {
        let trace = StageTrace::default();
        assert!(trace.dense.is_empty() && trace.fts.is_empty());
        assert!(trace.fused.is_empty() && trace.reranked.is_empty());
    }

    #[test]
    fn model_labels_match_the_default_backend() {
        // The record stamps these labels; a default-model swap must break this
        // test rather than leave a stale identity in a recorded measurement.
        assert_eq!(DENSE_MODEL_LABEL, DEFAULT_EMBED_MODEL);
        assert_eq!(RERANKER_MODEL_LABEL, DEFAULT_RERANK_MODEL);
        // The stored table must be 1024-dim for the pinned embedder.
        assert_eq!(DENSE_DIM, 1024);
        assert!(DENSE_DIM > 0 && DENSE_DIM.get().is_multiple_of(8));
    }

    #[test]
    fn cap_tokens_truncates_by_estimated_tokens() {
        // 512 tokens ≈ 1275 chars at the measured 2.49 chars/token.
        let under = "x".repeat(chunk_char_budget(EMBEDDER_MAX_TOKENS));
        assert_eq!(cap_tokens(&under, EMBEDDER_MAX_TOKENS), under);
        let over = "x".repeat(chunk_char_budget(EMBEDDER_MAX_TOKENS) + 1);
        let capped = cap_tokens(&over, EMBEDDER_MAX_TOKENS);
        assert!(capped.ends_with("… [chunk truncated]"));
        assert!(
            capped.chars().count()
                <= chunk_char_budget(EMBEDDER_MAX_TOKENS) + "… [chunk truncated]".chars().count()
        );
    }

    #[test]
    fn rerank_top_sorts_and_truncates() {
        let ranked = vec![(0, 0.1), (1, 0.9), (2, 0.5), (3, 0.8)];
        let top = rerank_top(ranked, 2);
        assert_eq!(top, vec![(1, 0.9), (3, 0.8)]);
    }

    #[test]
    fn truncate_chars_is_a_char_safe_passthrough_under_the_budget() {
        // the backend already returns sigmoid-normalized rerank scores, so the only
        // score normalization left is the canonical clamp in `retrieval.rs`.
        assert_eq!(truncate_chars("hello", 10), "hello");
        assert_eq!(truncate_chars("hello", 3), "hel");
        let multibyte = "é".repeat(10);
        assert_eq!(truncate_chars(&multibyte, 4).chars().count(), 4);
        assert_eq!(0.4f32.clamp(0.0, 1.0), 0.4);
        assert_eq!(1.5f32.clamp(0.0, 1.0), 1.0);
    }

    #[test]
    fn l2_normalize_handles_zero_and_nan_without_producing_nan() {
        // Zero vector → false, unchanged, no NaN.
        let mut zero = vec![0.0f32, 0.0, 0.0];
        assert!(!l2_normalize(&mut zero));
        assert_eq!(zero, vec![0.0, 0.0, 0.0]);

        // NaN component → false, unchanged.
        let mut nan = vec![f32::NAN, 1.0];
        assert!(!l2_normalize(&mut nan));
        assert!(nan[0].is_nan());
        assert_eq!(nan[1], 1.0);

        // Unit vector → true, unchanged.
        let mut unit = vec![1.0f32, 0.0, 0.0];
        assert!(l2_normalize(&mut unit));
        assert_eq!(unit, vec![1.0, 0.0, 0.0]);

        // Non-unit → true, unit norm (sum of squares ≈ 1).
        let mut v = vec![3.0f32, 4.0];
        assert!(l2_normalize(&mut v));
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6, "norm_sq = {norm_sq}");
    }

    #[test]
    fn sql_string_literal_escapes_quotes() {
        assert_eq!(sql_string_literal("src/a.rs"), "'src/a.rs'");
        assert_eq!(sql_string_literal("it's.rs"), "'it''s.rs'");
    }

    /// The serve watcher's path predicates must agree with the retrieval
    /// walker's tree policy (plus serve's own .operator/).
    #[test]
    fn the_watcher_path_predicates_follow_the_walker_policy() {
        assert!(is_excluded_tree_path(Path::new("/ws/target/x.rs")));
        assert!(is_excluded_tree_path(Path::new("/ws/node_modules/a/b.js")));
        assert!(is_excluded_tree_path(Path::new("/ws/.git/HEAD")));
        assert!(is_excluded_tree_path(Path::new(
            "/ws/.cache/retrieval/x.rs"
        )));
        assert!(is_excluded_tree_path(Path::new("/ws/.operator/serve.json")));
        assert!(!is_excluded_tree_path(Path::new("/ws/src/lib.rs")));

        assert!(is_indexable_source_path(Path::new("/ws/src/lib.rs")));
        assert!(is_indexable_source_path(Path::new("/ws/README.md")));
        assert!(!is_indexable_source_path(Path::new("/ws/spec.pdf")));
    }

    #[test]
    fn dual_routing_requires_a_secondary_and_a_symbol_query() {
        assert!(dual_route_applies(
            true,
            true,
            "where is LocalCodeIndex defined"
        ));
        assert!(dual_route_applies(true, true, "parse_manifest"));
        assert!(!dual_route_applies(false, true, "parse_manifest"));
        assert!(!dual_route_applies(true, false, "parse_manifest"));
        assert!(!dual_route_applies(
            true,
            true,
            "how does the local index avoid re-embedding unchanged files"
        ));
    }

    #[test]
    fn the_query_log_records_no_source_content() {
        let paths = vec!["src/a.rs".to_string(), "src/b.rs".to_string()];
        let line = query_log_line("how does x work", Some("crate"), 10, 42, &paths);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["query"], "how does x work");
        assert_eq!(parsed["scope"], "crate");
        assert_eq!(parsed["samples"], 10);
        assert_eq!(parsed["latency_ms"], 42);
        assert_eq!(parsed["top_paths"][0], "src/a.rs");
        // The record must not carry chunk bodies or file text.
        assert!(!line.contains("content"));
        assert!(!line.contains("fn main"));
    }

    #[test]
    fn the_fusion_gate_fires_only_on_lexical_only_candidates() {
        let dense = vec![("a".to_string(), 0.1), ("b".to_string(), 0.2)];
        let same = vec![("a".to_string(), 3.0)];
        assert!(!lexical_stage_adds_candidates(&dense, &same));
        let adds = vec![("a".to_string(), 3.0), ("c".to_string(), 2.0)];
        assert!(lexical_stage_adds_candidates(&dense, &adds));
        assert!(!lexical_stage_adds_candidates(&dense, &[]));
    }

    #[test]
    fn module_hints_are_derived_from_chunk_paths() {
        assert_eq!(module_hint("src/code_search.rs"), "code_search");
        assert_eq!(module_hint("src/models/qwen3.rs"), "models::qwen3");
        assert_eq!(module_hint("crates/x/src/lib.rs"), "x");
        assert_eq!(module_hint("crates/x/src/a/b.rs"), "x::a::b");
        assert_eq!(module_hint("src/lib.rs"), "src");
        assert_eq!(module_hint("README.md"), "README.md");
    }

    #[test]
    fn symbol_shaped_queries_route_to_rank_fusion() {
        for query in [
            "LocalCodeIndex::build",
            "parse_manifest",
            "where is RefreshReport defined",
            "explain retrieval_embed_model",
        ] {
            assert!(
                is_symbol_shaped_query(query),
                "expected symbol-shaped: {query}"
            );
        }
        for query in [
            "how does the local index avoid re-embedding unchanged files",
            "how are large tool results compressed before the model sees them",
            "where is the plan artifact written",
            "",
        ] {
            assert!(
                !is_symbol_shaped_query(query),
                "expected natural language: {query}"
            );
        }
    }

    #[test]
    fn table_name_bakes_every_index_input() {
        let base = Config::test_default();
        let t1 = table_name(&base);

        let mut other_ws = base.clone();
        other_ws.workspace_root = PathBuf::from("/ws/other").into();
        assert_ne!(t1, table_name(&other_ws), "different workspaces");

        let mut other_lines = base.clone();
        other_lines.retrieval_chunk_lines += 10;
        assert_ne!(t1, table_name(&other_lines), "different chunk params");

        let mut other_overlap = base.clone();
        other_overlap.retrieval_chunk_overlap = CHUNK_OVERLAP_LINES - 5;
        assert_ne!(t1, table_name(&other_overlap), "different chunk overlap");

        // Opting the document corpus in changes the indexed input set.
        let mut with_docs = base.clone();
        with_docs.retrieval_documents = crate::types::Documents::on();
        assert_ne!(t1, table_name(&with_docs), "different corpus inputs");

        // Explicit includes are more indexed input: a different set is a
        // different table, while spelling/order/duplicates of one set are one
        // identity.
        let mut with_include = base.clone();
        with_include.retrieval_include = vec!["docs/guides".into()];
        assert_ne!(t1, table_name(&with_include), "different corpus inputs");
        let mut same_include = base.clone();
        same_include.retrieval_include = vec!["./docs/guides/".into(), "docs/guides".into()];
        assert_eq!(
            table_name(&with_include),
            table_name(&same_include),
            "one include set is one identity"
        );

        // The table name bakes the logic version: a bump must never open a
        // table written by older chunking/search logic. Pinned to the literal so
        // raising it is a deliberate edit — it re-embeds the workspace for every
        // user (the ingestion manifest is namespaced by the same identity).
        assert_eq!(LOCAL_SEARCH_LOGIC_VERSION, "local-search-9");
        assert!(
            t1.starts_with("code_chunks_") && t1.contains(LOCAL_SEARCH_LOGIC_VERSION),
            "{t1}"
        );
    }

    /// The identity every existing workspace's table and manifest were written
    /// under. Inlining both names means any change to an identity input is a
    /// deliberate edit that re-embeds every workspace — the frozen-name
    /// tripwire the inference refactor must keep green.
    #[test]
    fn table_name_never_exceeds_the_lancedb_limit() {
        let mut cfg = Config::test_default();
        cfg.retrieval_enrichment = crate::chunk_context::ChunkEnrichment::ALL;
        cfg.retrieval_include = vec!["crates/a-very-long-crate-name/with/deep/nested/path".into()];
        cfg.retrieval_exclude = vec!["target/generated".into()];
        cfg.languages = vec![
            LanguageId::parse("rust").expect("catalog id"),
            LanguageId::parse("python").expect("catalog id"),
            LanguageId::parse("typescript").expect("catalog id"),
        ];
        let long_embedder = format!("fastembed-{}-candle-bert-1-embed-dim-2560", "0".repeat(16));
        let name = table_name_for(&cfg, &long_embedder);
        assert!(name.len() <= 128, "{} chars: {name}", name.len());
        // Two different long identities must not collapse to one table.
        let other = table_name_for(&cfg, &format!("{long_embedder}-f32"));
        assert_ne!(name, other);
    }

    #[test]
    fn the_shipped_index_identity_is_frozen() {
        let cfg = Config::test_default();
        assert_eq!(
            table_name(&cfg),
            "code_chunks_a31f819df3a797e1_vlocal-search-9_50_xberg_code_src_024_\
             fastembed-ab98510c-candle-bert-1"
        );
        assert_eq!(
            manifest_path(&cfg)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            "ingest_manifest_code_chunks_a31f819df3a797e1_vlocal-search-9_50_xberg_\
             code_src_024_fastembed-ab98510c-candle-bert-1.json"
        );
    }

    #[test]
    fn two_different_embedders_can_never_share_a_table() {
        let cfg = Config::test_default();
        let ours = table_name(&cfg);
        assert!(
            ours.ends_with(&embedder_identity(&cfg)),
            "the table name must name the embedder that fills it: {ours}"
        );

        // A different model at the *same* width must not share the table:
        // `GTELargeENV15` is also 1024-dim, so width is not a discriminator.
        // Without the identity component these two names would be identical and
        // one table would hold two vector spaces.
        let other_model = table_name_for(&cfg, "GTELargeENV15-unquantized");
        assert_ne!(other_model, ours, "two models must never share a table");

        // Quantisation alone changes the stored vectors, so it changes the
        // table even though the model name is the same.
        let unquantized = table_name_for(&cfg, "BGELargeENV15-unquantized");
        let quantized = table_name_for(&cfg, "BGELargeENV15-quantized");
        assert_ne!(quantized, unquantized, "quantisation is part of the space");

        // The manifest is namespaced by the same identity, so a fresh table is
        // never judged by an old manifest (which would suppress every embed and
        // leave the new table empty).
        let our_manifest = manifest_path(&cfg);
        let other_manifest = manifest_path_for(&cfg, "GTELargeENV15-unquantized");
        let quantized_manifest = manifest_path_for(&cfg, "BGELargeENV15-quantized");
        assert_ne!(other_manifest, our_manifest, "one manifest per table");
        assert_ne!(quantized_manifest, our_manifest, "one manifest per table");
        assert_ne!(quantized_manifest, other_manifest);
        let our_name = our_manifest.file_name().unwrap().to_string_lossy();
        assert!(
            our_name.contains(&ours),
            "the manifest filename must derive from the table identity: {our_name}"
        );
    }

    #[test]
    fn manifest_delta_counts_both_directions() {
        let mut before = HashMap::new();
        before.insert("a.rs".to_string(), "h1".to_string());
        before.insert("gone.rs".to_string(), "h2".to_string());
        let mut after = HashMap::new();
        after.insert("a.rs".to_string(), "h1".to_string());
        after.insert("b.rs".to_string(), "h3".to_string());
        assert_eq!(manifest_delta(&before, &after), (1, 1));
        assert_eq!(manifest_delta(&after, &after), (0, 0));
        assert_eq!(manifest_delta(&HashMap::new(), &after), (2, 0));
        assert_eq!(manifest_delta(&after, &HashMap::new()), (0, 2));
    }

    #[test]
    fn manifest_delta_counts_a_changed_hash_as_re_embedded() {
        let manifest = |path: &str, hash: &str| {
            let mut m = HashMap::new();
            m.insert(path.to_string(), hash.to_string());
            m
        };
        // The reported bug: an edited file keeps its key and changes only its
        // hash, yet it IS re-embedded (`files_to_reindex` selects by hash). The
        // counter must move, or every incremental pass over an edit reads as a
        // no-op while the table is genuinely rewritten.
        assert_eq!(
            manifest_delta(&manifest("a.rs", "h1"), &manifest("a.rs", "h2")),
            (1, 0),
            "a content change is one re-embed, not zero"
        );
        // A rename is a new path plus a dropped one — unchanged by the fix.
        assert_eq!(
            manifest_delta(&manifest("a.rs", "h1"), &manifest("b.rs", "h1")),
            (1, 1)
        );
        // An edit that also drops a sibling: the two directions are independent.
        let before = manifest("a.rs", "h1");
        let mut after = manifest("a.rs", "h2");
        after.insert("b.rs".to_string(), "h3".to_string());
        assert_eq!(manifest_delta(&before, &after), (2, 0));
    }

    #[test]
    fn query_time_knobs_never_move_the_index_identity() {
        // R1/R2: the dense index mode and the reranker window change the
        // RANKING, not the stored text or vectors, so neither may move the table
        // name or the manifest — no workspace may re-embed for either.
        let base = Config::test_default();
        let table = table_name(&base);
        let manifest = manifest_path(&base);

        let mut ann = base.clone();
        ann.retrieval_vector_index = RetrievalVectorIndex::Auto;
        assert_eq!(
            table_name(&ann),
            table,
            "the dense index mode is not part of the stored text"
        );
        assert_eq!(manifest_path(&ann), manifest);

        let mut wide = base.clone();
        wide.retrieval_rerank_max_tokens = crate::types::TokenLimit::new(2048);
        assert_eq!(
            table_name(&wide),
            table,
            "the reranker window is a query-time knob"
        );
        assert_eq!(manifest_path(&wide), manifest);
    }

    #[test]
    fn plan_embed_batches_never_splits_a_file() {
        // Empty corpus: no batch at all (and no panic, no infinite loop).
        assert!(plan_embed_batches(&[], EMBED_GROUP_CHUNKS).is_empty());

        // Everything fits: one batch covering every file, in order.
        assert_eq!(plan_embed_batches(&[4, 4, 4], 32), vec![0..3]);

        // The cap closes a batch BEFORE the file that would exceed it, and no
        // file is ever split across two batches.
        assert_eq!(plan_embed_batches(&[10, 10, 1, 1], 20), vec![0..2, 2..4]);

        // A file larger than the cap is its own batch — never dropped, never
        // split.
        assert_eq!(plan_embed_batches(&[1, 50, 1], 8), vec![0..1, 1..2, 2..3]);

        // `cap == 0` degrades to one file per batch rather than looping.
        assert_eq!(plan_embed_batches(&[1, 2], 0), vec![0..1, 1..2]);

        // Deterministic: the same counts always yield the same grouping.
        assert_eq!(
            plan_embed_batches(&[3, 3, 3], 6),
            plan_embed_batches(&[3, 3, 3], 6)
        );
    }

    #[test]
    fn embedder_identity_names_the_configured_model_and_is_path_safe() {
        // The identity must name the configured embedder repo and backend: a
        // model or backend swap changes the table name — one re-embed, never a
        // shared table.
        let mut cfg = Config::test_default();
        assert_eq!(embedder_identity(&cfg), "fastembed-ab98510c-candle-bert-1");
        cfg.retrieval_embed_model = "vendor/model:v1".into();
        let other = embedder_identity(&cfg);
        assert!(other.starts_with("fastembed-") && other.ends_with("-candle-bert-1"));
        assert_ne!(other, embedder_identity(&Config::test_default()));
        assert_eq!(DENSE_MODEL_LABEL, DEFAULT_EMBED_MODEL);
    }

    #[test]
    fn the_embedder_identity_is_a_safe_path_component() {
        let base = Config::test_default();
        let identity = embedder_identity(&base);
        assert!(!identity.is_empty());
        // Never smuggle a path component (or whitespace) into the cache dir.
        for candidate in [
            table_name(&base),
            table_name_for(&base, ""),
            table_name_for(&base, "GTELargeENV15-quantized"),
        ] {
            assert!(
                !candidate.contains(['/', '\\']) && !candidate.contains(char::is_whitespace),
                "the identity must stay one path component: {candidate}"
            );
        }
        let name = manifest_path(&base)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert!(
            name.contains(&identity),
            "the manifest must name the identity it describes: {name}"
        );
    }

    #[test]
    fn document_chunks_are_annotated_and_source_chunks_are_not() {
        let cfg = Config::test_default();
        let doc = chunks_for_entry(
            &cfg,
            "spec.rtf",
            "hello world",
            "h",
            Some(crate::document::Engine::Anydoc),
            "docs",
            "",
        );
        assert_eq!(doc.len(), 1);
        assert_eq!(doc[0].language, "markdown");
        assert!(
            doc[0]
                .content
                .starts_with("File: spec.rtf (converted document"),
            "{}",
            doc[0].content
        );

        let src = chunks_for_entry(&cfg, "src/lib.rs", "fn main() {}", "h", None, "c", "");
        assert_eq!(src[0].language, "rs");
        assert!(!src[0].content.contains("converted document"));
    }

    #[test]
    fn document_discovery_matches_the_converter_extensions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.pdf"), b"%PDF-1.4").unwrap();
        std::fs::write(dir.path().join("b.docx"), b"PK").unwrap();
        std::fs::write(dir.path().join("c.rs"), b"fn main() {}").unwrap();
        std::fs::write(dir.path().join("d.txt"), b"text").unwrap();
        let names: Vec<String> = discover_document_files(dir.path())
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect();
        assert!(names.contains(&"a.pdf".to_string()), "{names:?}");
        assert!(names.contains(&"b.docx".to_string()), "{names:?}");
        assert!(!names.contains(&"c.rs".to_string()), "{names:?}");
        assert!(!names.contains(&"d.txt".to_string()), "{names:?}");
    }

    /// A minimal git layout: `<root>/main/.git/worktrees/feat` stands in for the
    /// repository's common dir, and `<root>/feat/.git` is the gitfile a linked
    /// worktree carries. Returns `(main, feat)`.
    fn worktree_fixture(root: &Path) -> (PathBuf, PathBuf) {
        let main = root.join("main");
        let gitdir = main.join(".git").join("worktrees").join("feat");
        std::fs::create_dir_all(&gitdir).unwrap();
        let feat = root.join("feat");
        std::fs::create_dir_all(&feat).unwrap();
        std::fs::write(feat.join(".git"), format!("gitdir: {}\n", gitdir.display())).unwrap();
        (main, feat)
    }

    #[test]
    fn worktrees_of_one_repository_share_the_table_and_manifest() {
        // C10: the index identity is the *repository's* git common dir, not the
        // worktree path, so a linked worktree reuses the already-embedded index
        // instead of re-embedding the whole repository.
        let dir = tempfile::tempdir().unwrap();
        let (main, feat) = worktree_fixture(dir.path());
        let at = |ws: &Path| {
            let mut c = Config::test_default();
            c.workspace_root = ws.to_path_buf().into();
            c
        };
        let main_cfg = at(&main);
        let feat_cfg = at(&feat);
        assert_eq!(
            table_name(&main_cfg),
            table_name(&feat_cfg),
            "two worktrees of one repository resolve to the same table"
        );
        // The load-bearing coupling is intact: the manifest filename still
        // derives from the very same identity.
        assert_eq!(
            manifest_path(&main_cfg).file_name(),
            manifest_path(&feat_cfg).file_name(),
            "the ingestion manifest is shared with the table it describes"
        );
        assert_eq!(
            manifest_path(&main_cfg),
            manifest_path(&feat_cfg),
            "and it lives in the same (shared) retrieval store"
        );

        // A different repository, and a directory that is not a repository at
        // all, are different indexes.
        let other = dir.path().join("other");
        std::fs::create_dir_all(other.join(".git")).unwrap();
        assert_ne!(
            table_name(&main_cfg),
            table_name(&at(&other)),
            "a different repository is a different index"
        );
        assert_ne!(
            table_name(&main_cfg),
            table_name(&at(dir.path())),
            "a non-repo directory is a different index"
        );
    }

    #[test]
    fn a_corrupt_shared_manifest_degrades_to_a_re_embed() {
        // Two worktree runs can interleave writes on the shared manifest. An
        // unparsable file must read as empty — one re-embed — never a panic and
        // never a stale skip that would leave the table empty.
        let dir = tempfile::tempdir().unwrap();
        let (main, feat) = worktree_fixture(dir.path());
        let mut feat_cfg = Config::test_default();
        feat_cfg.workspace_root = feat.into();
        feat_cfg.retrieval_cache_dir = dir.path().join("shared-retrieval").into();
        std::fs::create_dir_all(&feat_cfg.retrieval_cache_dir).unwrap();
        // A manifest that is the *same file* for both checkouts.
        let mut main_cfg = feat_cfg.clone();
        main_cfg.workspace_root = main.into();
        assert_eq!(manifest_path(&feat_cfg), manifest_path(&main_cfg));

        std::fs::write(manifest_path(&feat_cfg), "{ this is not json").unwrap();
        assert!(
            load_manifest(&feat_cfg).is_empty(),
            "a corrupt shared manifest reads as empty"
        );
        assert!(load_manifest(&main_cfg).is_empty());
    }

    #[test]
    fn manifest_path_namespaces_by_the_table_identity() {
        // A shared manifest would report every file as embedded after any knob
        // change, leaving the freshly created table empty.
        let cfg = Config::test_default();
        let path = manifest_path(&cfg);
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.contains(LOCAL_SEARCH_LOGIC_VERSION), "{name}");
        assert!(name.contains(&table_name(&cfg)), "{name}");
        assert_eq!(path.parent(), Some(cfg.retrieval_cache_dir.as_path()));

        let mut other_lines = cfg.clone();
        other_lines.retrieval_chunk_lines += 10;
        assert_ne!(
            manifest_path(&other_lines),
            path,
            "the manifest must follow the table it describes"
        );
    }

    #[test]
    fn crate_predicate_escapes_and_scopes() {
        assert_eq!(crate_predicate(None), None, "no scope means no predicate");
        assert_eq!(
            crate_predicate(Some("mentions")).as_deref(),
            Some("crate_name = 'mentions'")
        );
        assert_eq!(
            crate_predicate(Some("o'brien")).as_deref(),
            Some("crate_name = 'o''brien'")
        );
    }

    #[test]
    fn schema_carries_the_crate_scope_column() {
        let schema = code_schema(DENSE_DIM.get());
        let field = schema
            .field_with_name("crate_name")
            .expect("crate_name column");
        assert_eq!(field.data_type(), &arrow_schema::DataType::Utf8);
        assert!(!field.is_nullable());
    }

    #[test]
    fn chunks_to_batch_builds_a_valid_record_batch() {
        let mut chunk = sliding_window_chunks("src/x.rs", "fn x() {}\nfn y() {}", "h", 50)
            .into_iter()
            .next()
            .unwrap();
        chunk.vector = Embedding::from(vec![0.1; DENSE_DIM.get()]);
        chunk.crate_name = CrateName::from("mentions");
        let batch = chunks_to_batch(&[chunk]).expect("batch must build");
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 10);
        assert_eq!(
            batch.column_by_name("vector").unwrap().len(),
            1,
            "one vector row"
        );
        let crate_col = batch
            .column_by_name("crate_name")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .expect("crate_name is a string column");
        assert_eq!(crate_col.value(0), "mentions");
    }

    #[test]
    fn file_hash_is_content_addressed() {
        assert_eq!(file_hash_of("abc"), file_hash_of("abc"));
        assert_ne!(file_hash_of("abc"), file_hash_of("abd"));
        assert_eq!(file_hash_of("").len(), 64);
    }

    #[test]
    fn discovery_honors_gitignore_in_temp_workspace() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored/\n*.gen.rs\n").unwrap();
        std::fs::create_dir(dir.path().join("ignored")).unwrap();
        std::fs::write(dir.path().join("ignored/x.rs"), "fn x() {}").unwrap();
        std::fs::write(dir.path().join("keep.rs"), "fn keep() {}").unwrap();
        std::fs::write(dir.path().join("skip.gen.rs"), "fn skip() {}").unwrap();
        std::fs::create_dir(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join("target/t.rs"), "fn t() {}").unwrap();
        let files = discover_source_files(dir.path());
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["keep.rs"], "gitignore + heavy dirs filtered");
    }

    #[test]
    fn the_always_excluded_trees_are_one_shared_list() {
        // The mention walker, this indexer, and `crate_map` all read this set, so
        // a name added here is filtered everywhere — if it were duplicated, the
        // policies would drift.
        assert_eq!(
            ALWAYS_EXCLUDED_DIRS,
            ["node_modules", "target", "vendor", "dist"]
        );
        for name in ALWAYS_EXCLUDED_DIRS {
            assert!(is_always_excluded_dir(name), "{name} must be excluded");
        }
        assert!(!is_always_excluded_dir("src"));
        assert!(
            !is_always_excluded_dir("build"),
            "`build` is skipped as a nested entry, but stays mentionable on purpose"
        );
    }

    #[test]
    fn discovery_skips_every_excluded_tree_without_any_ignore_file() {
        // All four trees are skipped with no `.gitignore` in sight, and a nested
        // occurrence counts exactly like a top-level one — a root-anchored
        // `/node_modules` rule would not have covered any of these.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("keep.rs"), "fn keep() {}").unwrap();
        for (tree, file) in [
            ("frontend/node_modules/pkg/index.js", "module.exports = 1;"),
            ("target/debug/t.rs", "fn t() {}"),
            ("vendor/inner/src/lib.rs", "pub fn inner() {}"),
            ("packages/x/dist/out.rs", "fn out() {}"),
        ] {
            let path = dir.path().join(tree);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, file).unwrap();
        }
        std::fs::write(dir.path().join("frontend/app.rs"), "fn app() {}").unwrap();

        let names: Vec<String> = discover_source_files(dir.path())
            .iter()
            .map(|p| p.strip_prefix(dir.path()).unwrap().display().to_string())
            .collect();
        assert_eq!(names, vec!["frontend/app.rs", "keep.rs"], "{names:?}");
    }

    // ── Explicit corpus includes ────────────────────────────────────────────

    #[test]
    fn normalized_includes_are_canonical_and_order_independent() {
        // Whitespace, a trailing slash, a leading `./`, an empty entry and a
        // duplicate all collapse into one canonical set.
        let raw = [
            " docs/guides/ ".to_string(),
            "./docs/plans".to_string(),
            "docs/guides".to_string(),
            String::new(),
            "   ".to_string(),
        ];
        assert_eq!(
            normalized_includes(&raw),
            ["docs/guides", "docs/plans"],
            "one canonical, sorted, deduplicated set"
        );

        // The identity fragment is exactly that set's hash: absent for the empty
        // set (so the incumbent names survive untouched), order-insensitive and
        // duplicate-insensitive otherwise.
        assert_eq!(include_identity_fragment(&[] as &[&str]), "");
        let a = include_identity_fragment(&["docs/a", "docs/b"]);
        let b = include_identity_fragment(&["docs/b", "docs/a", "docs/a"]);
        assert_eq!(a, b, "the same set is one identity");
        assert_ne!(a, include_identity_fragment(&["docs/a"]));
    }

    #[test]
    fn validate_includes_refuses_missing_and_escaping_paths() {
        let root = tempfile::tempdir().unwrap();
        let ws = root.path().join("ws");
        std::fs::create_dir_all(ws.join("docs/guides")).unwrap();
        std::fs::create_dir_all(root.path().join("outside")).unwrap();

        // A workspace-relative folder resolves and passes.
        let ok = validate_includes(&ws, &["docs/guides".to_string()]).unwrap();
        assert_eq!(ok, [ws.join("docs/guides").canonicalize().unwrap()]);

        // Nothing named at all is an error naming the flag.
        let err = validate_includes(&ws, &[] as &[&str]).unwrap_err();
        assert!(err.contains("--include"), "{err}");

        // A typo fails loudly rather than building a table that lacks it.
        let err = validate_includes(&ws, &["docs/typo".to_string()]).unwrap_err();
        assert!(err.contains("docs/typo"), "{err}");

        // A path that escapes the workspace is refused: manifest keys, crate
        // scoping and the watcher's single root all assume containment.
        let err = validate_includes(&ws, &["../outside".to_string()]).unwrap_err();
        assert!(err.contains("outside the workspace"), "{err}");
    }

    #[test]
    fn include_discovery_takes_text_documents_and_honors_exclusions() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs/guides");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::create_dir_all(docs.join("target")).unwrap();
        std::fs::write(docs.join("plan.md"), "# plan").unwrap();
        std::fs::write(docs.join("spec.pdf"), "not-really-a-pdf").unwrap();
        std::fs::write(docs.join("target/skip.rs"), "fn skip() {}").unwrap();
        std::fs::write(docs.join(".gitignore"), "draft.md\n").unwrap();
        std::fs::write(docs.join("draft.md"), "draft").unwrap();
        // A sparse file just over the cap: skipped without reading 16 MiB.
        let big = docs.join("huge.txt");
        std::fs::File::create(&big)
            .unwrap()
            .set_len(INCLUDE_MAX_FILE_BYTES + 1)
            .unwrap();

        let names: Vec<String> = discover_include_files(&docs)
            .iter()
            .map(|p| p.strip_prefix(&docs).unwrap().display().to_string())
            .collect();
        assert_eq!(names, vec!["plan.md", "spec.pdf"], "{names:?}");

        // A directly named file is taken unconditionally — explicit intent
        // outranks gitignore, the exclusion list and the size cap.
        let named = discover_include_files(&docs.join("draft.md"));
        assert_eq!(named, [docs.join("draft.md")]);
    }

    // ── E3/B: the model-free FTS harness ───────────────────────────────────
    //
    // A text-only LanceDB table (no vector column, no server) so the full-text
    // layer is measured on its own. `fts_tokens` holds the code-aware
    // searchable text (`retrieval::fts_index_text`); the `content` column is
    // the untouched source text both modes keep storing.

    /// Text-only FTS table: `(id, content, fts_tokens)`, FTS index on
    /// `index_column`. `rows` are `(id, content)`; the `fts_tokens` column is
    /// derived the production way.
    async fn fts_probe_table(
        dir: &Path,
        index_column: &str,
        rows: &[(&str, &str, &str, &str)],
    ) -> Result<lancedb::Table, String> {
        use arrow_array::{RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        let db = lancedb::connect(dir.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| e.to_string())?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("fts_tokens", DataType::Utf8, false),
        ]));
        let table = db
            .create_empty_table("fts_probe", schema.clone())
            .execute()
            .await
            .map_err(|e| e.to_string())?;
        let ids = StringArray::from(rows.iter().map(|(id, _, _, _)| *id).collect::<Vec<_>>());
        let contents = StringArray::from(rows.iter().map(|(_, c, _, _)| *c).collect::<Vec<_>>());
        let tokens = StringArray::from(
            rows.iter()
                .map(|(_, c, p, k)| crate::retrieval::fts_index_text(c, p, k))
                .collect::<Vec<_>>(),
        );
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(contents),
                Arc::new(tokens) as Arc<dyn arrow_array::Array>,
            ],
        )
        .map_err(|e| e.to_string())?;
        table
            .add(batch)
            .execute()
            .await
            .map_err(|e| e.to_string())?;
        table
            .create_index(
                &[index_column],
                lancedb::index::Index::FTS(Default::default()),
            )
            .execute()
            .await
            .map_err(|e| e.to_string())?;
        Ok(table)
    }

    /// The row ids an FTS query reaches, best-first.
    async fn fts_probe_ids(table: &lancedb::Table, query: &str) -> Result<Vec<String>, String> {
        let stream = table
            .query()
            .full_text_search(lancedb::index::scalar::FullTextSearchQuery::new(
                query.to_string(),
            ))
            .limit(50)
            .execute()
            .await
            .map_err(|e| e.to_string())?;
        Ok(collect_scored(stream, "_score")
            .await?
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    /// The probe corpus: one row per naming convention, plus a path-only and a
    /// crate-only row. `(id, content, path, crate)`.
    fn fts_probe_corpus() -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
        vec![
            (
                "snake",
                "fn read_run_record(path: &Path) {}",
                "src/a.rs",
                "alpha",
            ),
            ("camel", "fn readRunRecord() {}", "src/b.rs", "alpha"),
            ("pascal", "fn ReadRunRecord() {}", "src/c.rs", "alpha"),
            (
                "acronym",
                "struct HTTPServer { port: u16 }",
                "src/d.rs",
                "alpha",
            ),
            (
                "pathonly",
                "fn unrelated_name() {}",
                "src/session/run_record.rs",
                "alpha",
            ),
            (
                "crateonly",
                "fn unrelated_name() {}",
                "src/scope/crate_name.rs",
                "operator",
            ),
        ]
    }

    #[tokio::test]
    async fn fts_default_tokenizer_misses_camel_case_identifiers() {
        // The plan's tracer: with the DEFAULT tokenizer over the raw `content`
        // column, a sub-token query reaches the snake_case row but NOT the
        // camelCase row — the asymmetry E3/B fixes. With the code-aware
        // `fts_tokens` column the SAME query reaches both, and all three
        // naming conventions retrieve both.
        let dir = tempfile::tempdir().unwrap();
        let rows = fts_probe_corpus();
        let baseline_dir = dir.path().join("baseline");
        let code_dir = dir.path().join("code");
        std::fs::create_dir_all(&baseline_dir).unwrap();
        std::fs::create_dir_all(&code_dir).unwrap();
        let baseline = fts_probe_table(&baseline_dir, "content", &rows)
            .await
            .unwrap();
        let code = fts_probe_table(&code_dir, "fts_tokens", &rows)
            .await
            .unwrap();

        let query = crate::retrieval::fts_query_text("read_run_record");
        assert_eq!(query, "read run record");

        let baseline_ids = fts_probe_ids(&baseline, &query).await.unwrap();
        assert!(
            baseline_ids.contains(&"snake".to_string()),
            "default tokenizer must reach the snake_case row: {baseline_ids:?}"
        );
        assert!(
            !baseline_ids.contains(&"camel".to_string()),
            "default tokenizer must MISS the camelCase row (the defect E3/B fixes): {baseline_ids:?}"
        );
        assert!(
            !baseline_ids.contains(&"pascal".to_string()),
            "default tokenizer must MISS the PascalCase row: {baseline_ids:?}"
        );

        let code_ids = fts_probe_ids(&code, &query).await.unwrap();
        for id in ["snake", "camel", "pascal"] {
            assert!(
                code_ids.contains(&id.to_string()),
                "code-aware tokens must reach `{id}`: {code_ids:?}"
            );
        }

        // Naming-convention parity: every spelling of the identifier retrieves
        // every row that carries it.
        for spelling in ["readRunRecord", "read_run_record", "ReadRunRecord"] {
            let ids = fts_probe_ids(&code, &crate::retrieval::fts_query_text(spelling))
                .await
                .unwrap();
            for id in ["snake", "camel", "pascal"] {
                assert!(
                    ids.contains(&id.to_string()),
                    "`{spelling}` must retrieve `{id}` from the code-aware index: {ids:?}"
                );
            }
        }

        // Path-only and crate-only terms are indexed too.
        let path_ids = fts_probe_ids(&code, &crate::retrieval::fts_query_text("run_record"))
            .await
            .unwrap();
        assert!(
            path_ids.contains(&"pathonly".to_string()),
            "path terms must be searchable: {path_ids:?}"
        );
    }

    #[test]
    fn fts_index_text_is_the_searchable_column_of_one_chunk() {
        let mut chunk = sliding_window_chunks(
            "src/session/run_record.rs",
            "fn readRunRecord() {}",
            "h",
            50,
        )
        .into_iter()
        .next()
        .unwrap();
        chunk.crate_name = CrateName::from("operator");
        let text = fts_text_of(&chunk);
        for term in ["read", "run", "record", "session"] {
            assert!(
                text.split_whitespace().any(|t| t == term),
                "{term} in {text}"
            );
        }
    }

    #[test]
    fn fts_text_of_survives_a_degenerate_chunk() {
        // No panic and no null column value: the column exists in every row, so
        // a mode switch can never produce an empty FTS index without an error.
        let mut chunk = LocalChunk {
            id: "x".into(),
            path: "()".into(),
            chunk_index: ChunkIndex::new(0),
            start_line: LineNumber::new(1),
            content: "()".into(),
            language: LanguageLabel::from("rs"),
            file_hash: "h".into(),
            crate_name: CrateName::default(),
            vector: Embedding::default(),
        };
        chunk.crate_name = CrateName::from("()");
        assert_eq!(fts_text_of(&chunk), "  ");
    }

    #[tokio::test]
    async fn fts_probe_notes_the_pinned_engine_stemming_and_stop_words() {
        // Task 2's edge case: the pinned builder DOES expose `stem` /
        // `remove_stop_words` (`InvertedIndexParams`, both defaulting to TRUE),
        // but the shipped layer uses `Index::FTS(Default::default())`, so the
        // engine applies BOTH. This test observes what that means for the two
        // named cases and prints it — the numbers are the record, not an
        // assumption.
        let dir = tempfile::tempdir().unwrap();
        let rows: Vec<(&str, &str, &str, &str)> = vec![
            ("singular", "fn record() {}", "src/a.rs", "alpha"),
            ("stopword", "fn for_in_loop() {}", "src/b.rs", "alpha"),
        ];
        let code_dir = dir.path().join("code");
        std::fs::create_dir_all(&code_dir).unwrap();
        let table = fts_probe_table(&code_dir, FTS_TOKEN_COLUMN, &rows)
            .await
            .unwrap();
        let plural = fts_probe_ids(&table, &crate::retrieval::fts_query_text("records"))
            .await
            .unwrap();
        let stop = fts_probe_ids(&table, &crate::retrieval::fts_query_text("for_in_loop"))
            .await
            .unwrap();
        eprintln!("── E3/B pinned-engine behaviour ──");
        eprintln!("  query `records` over a `record` row -> {plural:?}");
        eprintln!("  query `for_in_loop` -> {stop:?}");
        // Whatever the engine does, the sub-token vocabulary must be reachable:
        // `loop` and `in` were both emitted into the indexed text.
        let indexed = fts_text_of(&LocalChunk {
            id: "x".into(),
            path: "src/b.rs".into(),
            chunk_index: ChunkIndex::new(0),
            start_line: LineNumber::new(1),
            content: rows[1].1.into(),
            language: LanguageLabel::from("rs"),
            file_hash: "h".into(),
            crate_name: CrateName::from("alpha"),
            vector: Embedding::default(),
        });
        eprintln!("  indexed text: {indexed}");
        assert!(
            indexed.split_whitespace().any(|t| t == "loop"),
            "the sub-token vocabulary carries the identifier parts: {indexed}"
        );
    }
}

#[cfg(test)]
mod exclude_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn deepest_include_or_exclude_wins() {
        let mut cfg = Config::test_default();
        cfg.workspace_root = PathBuf::from("/ws").into();
        cfg.retrieval_exclude = vec!["vendor".into()];
        assert!(excluded_by(&cfg, &PathBuf::from("/ws/vendor/a.rs")));
        cfg.retrieval_include = vec!["vendor/keep".into()];
        assert!(!excluded_by(&cfg, &PathBuf::from("/ws/vendor/keep/a.rs")));
        assert!(excluded_by(&cfg, &PathBuf::from("/ws/vendor/other/a.rs")));
        cfg.retrieval_exclude.push("vendor/keep/no".into());
        assert!(excluded_by(&cfg, &PathBuf::from("/ws/vendor/keep/no/a.rs")));
    }
}
