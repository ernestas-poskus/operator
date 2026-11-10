//! Minimal configuration for the read-only code-intelligence server.
use std::path::{Path, PathBuf};

use crate::chunk_context::ChunkEnrichment;
use crate::types::LanguageId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetrievalVectorIndex {
    #[default]
    Exact,
    Auto,
}

impl RetrievalVectorIndex {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Auto => "auto",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "exact" => Some(Self::Exact),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    pub fn uses_ann_index(self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Weight dtype for the embedded fastembed models; `f16` is the default and
/// `f32` the numerical escape hatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelDtype {
    #[default]
    F16,
    F32,
}

impl ModelDtype {
    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }
}

/// Which fastembed candle loader backs the dense embedder. `Bert` is the
/// shipped mxbai path; `Qwen3` loads Qwen3 embedding checkpoints (last-token
/// pooling, query instruction), whose configs are not BERT-shaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedBackend {
    #[default]
    Bert,
    Qwen3,
}

impl EmbedBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Bert => "bert",
            Self::Qwen3 => "qwen3",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bert" => Some(Self::Bert),
            "qwen3" | "qwen" => Some(Self::Qwen3),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub workspace_root: PathBuf,
    /// Canonical ast-grep language ids selected at init.
    pub languages: Vec<LanguageId>,
    pub retrieval_cache_dir: PathBuf,
    pub retrieval_chunk_lines: usize,
    /// Lines repeated between consecutive sliding windows (default
    /// `CHUNK_OVERLAP_LINES`).
    pub retrieval_chunk_overlap: usize,
    pub retrieval_include: Vec<String>,
    pub retrieval_exclude: Vec<String>,
    pub retrieval_documents: bool,
    pub retrieval_dense_weight: f32,
    pub retrieval_fusion: crate::code_search::Fusion,
    pub retrieval_rerank_score: crate::code_search::RerankScoreNorm,
    pub retrieval_enrichment: ChunkEnrichment,
    pub retrieval_chunker: crate::code_search::ChunkerMode,
    pub retrieval_vector_index: RetrievalVectorIndex,
    pub retrieval_rerank_max_tokens: usize,
    pub retrieval_stage_candidates: usize,
    pub retrieval_rerank_candidates: usize,
    /// Where the HF snapshots live; `None` = discover an existing cache, else
    /// `<retrieval_cache_dir>/models`.
    pub retrieval_model_cache_dir: Option<PathBuf>,
    pub retrieval_model_dtype: ModelDtype,
    pub retrieval_embed_model: String,
    pub retrieval_embed_dim: usize,
    pub retrieval_embed_backend: EmbedBackend,
    pub retrieval_embed_query_prefix: String,
    pub retrieval_rerank_model: String,
    pub retrieval_rerank_instruction: Option<String>,
    pub retrieval_rerank_enabled: bool,
    /// Prefix the cross-encoder's documents with a derived `Module:` hint
    /// (the stored text already carries the `File:` enrichment prefix).
    pub retrieval_rerank_context: bool,
    /// Use RRF when the lexical stage contributes candidates the dense stage
    /// did not return (the error-analysis gate), weighted otherwise.
    pub retrieval_fusion_gate: bool,
    /// Optional local JSONL log of search queries for future real-traffic
    /// evaluation. Off by default; writes query text + returned paths only.
    pub retrieval_query_log: Option<PathBuf>,
    /// Optional secondary embedder; when set, symbol-shaped queries search the
    /// secondary index (its own table + model) instead of the primary one.
    pub retrieval_dual_embed_model: Option<String>,
    pub retrieval_dual_embed_dim: Option<usize>,
    pub retrieval_dual_embed_backend: Option<EmbedBackend>,
    pub retrieval_dual_embed_query_prefix: Option<String>,
    /// Whether symbol-shaped queries are routed to the secondary index.
    pub retrieval_dual_route: bool,
    /// Blend the cross-encoder score with the fused retrieval score
    /// (`blend * rerank + (1 - blend) * fused`, both min-max normalized).
    /// `0.0` is the pure-reranker order.
    pub retrieval_rerank_blend: f32,
    /// Expand the dense query with the code-aware identifier tokens before
    /// embedding (query-time only).
    pub retrieval_query_expansion: bool,
    /// Route symbol-shaped queries to reciprocal-rank fusion (query-time only).
    pub retrieval_query_routing: bool,
}

impl Config {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            retrieval_cache_dir: workspace_root.join(".cache/retrieval"),
            workspace_root,
            languages: Vec::new(),
            retrieval_chunk_lines: 50,
            retrieval_chunk_overlap: crate::code_search::CHUNK_OVERLAP_LINES,
            retrieval_include: Vec::new(),
            retrieval_exclude: Vec::new(),
            retrieval_documents: false,
            retrieval_dense_weight: 2.0,
            // RRF is the shipped fusion; it is robust across query shapes.
            retrieval_fusion: crate::code_search::Fusion::Rrf,
            retrieval_rerank_score: crate::code_search::RerankScoreNorm::Clamp,
            retrieval_enrichment: ChunkEnrichment::PATH,
            retrieval_chunker: crate::code_search::ChunkerMode::Xberg,
            retrieval_vector_index: RetrievalVectorIndex::Exact,
            retrieval_rerank_max_tokens: crate::code_search::DEFAULT_RERANK_MAX_TOKENS,
            retrieval_stage_candidates: crate::code_search::STAGE_CANDIDATES,
            retrieval_rerank_candidates: crate::code_search::RERANK_CANDIDATES,
            retrieval_model_cache_dir: None,
            retrieval_model_dtype: ModelDtype::F16,
            retrieval_embed_model: crate::code_search::DEFAULT_EMBED_MODEL.to_string(),
            retrieval_embed_dim: crate::code_search::DENSE_DIM,
            retrieval_embed_backend: EmbedBackend::Bert,
            retrieval_embed_query_prefix: String::new(),
            retrieval_rerank_model: crate::code_search::DEFAULT_RERANK_MODEL.to_string(),
            retrieval_rerank_instruction: Some(
                crate::code_search::DEFAULT_RERANK_INSTRUCTION.to_string(),
            ),
            retrieval_rerank_enabled: true,
            retrieval_rerank_context: false,
            retrieval_fusion_gate: false,
            retrieval_query_log: None,
            retrieval_dual_embed_model: None,
            retrieval_dual_embed_dim: None,
            retrieval_dual_embed_backend: None,
            retrieval_dual_embed_query_prefix: None,
            retrieval_dual_route: true,
            retrieval_rerank_blend: 0.0,
            retrieval_query_expansion: false,
            retrieval_query_routing: false,
        }
    }

    pub fn test_default() -> Self {
        Self::new(PathBuf::from("/ws/target-repo"))
    }
}

struct GitPointer {
    identity: PathBuf,
}

fn git_pointer(workspace_root: &Path) -> Option<GitPointer> {
    let dot_git = workspace_root.join(".git");
    let meta = std::fs::metadata(&dot_git).ok()?;
    if meta.is_dir() {
        return Some(GitPointer { identity: dot_git });
    }
    if !meta.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(&dot_git).ok()?;
    let target = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("gitdir:"))?
        .trim();
    if target.is_empty() {
        return None;
    }
    let target = Path::new(target);
    let git_dir = if target.is_absolute() {
        target.to_path_buf()
    } else {
        workspace_root.join(target)
    };
    if let Some(common) = git_dir
        .parent()
        .filter(|dir| dir.file_name().is_some_and(|n| n == "worktrees"))
        .and_then(|worktrees| worktrees.parent())
        .filter(|_| git_dir.file_name().is_some())
    {
        return Some(GitPointer {
            identity: common.to_path_buf(),
        });
    }
    Some(GitPointer { identity: git_dir })
}

pub fn repo_identity(workspace_root: &Path) -> PathBuf {
    match git_pointer(workspace_root) {
        Some(pointer) => std::fs::canonicalize(&pointer.identity).unwrap_or(pointer.identity),
        None => workspace_root.to_path_buf(),
    }
}

/// The shipped retrieval configuration is the program's product decision in
/// code; a change to any value is deliberate and must update this test.
#[cfg(test)]
mod shipped_retrieval_defaults {
    use super::{Config, EmbedBackend, ModelDtype};

    #[test]
    fn shipped_retrieval_defaults_are_frozen() {
        let cfg = Config::new(std::path::PathBuf::from("/ws"));

        // Index inputs — pool 20 and RRF are the shipped defaults; chunks stay
        // 50 lines / 10 overlap / xberg / path prefix.
        assert_eq!(cfg.retrieval_chunk_lines, 50);
        assert_eq!(
            cfg.retrieval_chunk_overlap,
            crate::code_search::CHUNK_OVERLAP_LINES
        );
        assert_eq!(crate::code_search::CHUNK_OVERLAP_LINES, 10);
        assert_eq!(cfg.retrieval_chunker.label(), "xberg");
        assert_eq!(cfg.retrieval_enrichment.label(), "path");
        assert_eq!(cfg.retrieval_dense_weight, 2.0);
        assert_eq!(cfg.retrieval_fusion.label(), "rrf");
        assert_eq!(cfg.retrieval_stage_candidates, 50);
        assert_eq!(cfg.retrieval_rerank_candidates, 20);
        assert_eq!(crate::code_search::RERANK_CANDIDATES, 20);
        assert_eq!(
            cfg.retrieval_vector_index,
            crate::config::RetrievalVectorIndex::Exact
        );

        // Models — mxbai 1024-d F16 through the BERT loader; Qwen3 reranker
        // with the code instruction and the configured 512-token window.
        assert_eq!(
            cfg.retrieval_embed_model,
            "mixedbread-ai/mxbai-embed-large-v1"
        );
        assert_eq!(cfg.retrieval_embed_dim, 1024);
        assert_eq!(cfg.retrieval_embed_backend, EmbedBackend::Bert);
        assert_eq!(cfg.retrieval_embed_query_prefix, "");
        assert_eq!(cfg.retrieval_model_dtype, ModelDtype::F16);
        assert_eq!(cfg.retrieval_rerank_model, "Qwen/Qwen3-Reranker-0.6B");
        assert_eq!(cfg.retrieval_rerank_max_tokens, 512);
        assert!(cfg.retrieval_rerank_enabled);
        assert_eq!(
            cfg.retrieval_rerank_instruction.as_deref(),
            Some("Given a code search query, retrieve relevant code")
        );
        assert_eq!(crate::fastembed::FASTEMBED_EMBEDDER_IMPL, "candle-bert-1");

        // Experimental knobs all default off.
        assert_eq!(cfg.retrieval_rerank_blend, 0.0);
        assert!(!cfg.retrieval_rerank_context);
        assert!(!cfg.retrieval_fusion_gate);
        assert!(!cfg.retrieval_query_expansion);
        assert!(!cfg.retrieval_query_routing);
        assert_eq!(cfg.retrieval_query_log, None);
        // Dual routing is available but requires an explicit secondary model.
        assert_eq!(cfg.retrieval_dual_embed_model, None);
        assert_eq!(cfg.retrieval_dual_embed_dim, None);
        assert_eq!(cfg.retrieval_dual_embed_backend, None);
        assert!(cfg.retrieval_dual_route);
        assert_eq!(
            cfg.retrieval_rerank_score,
            crate::code_search::RerankScoreNorm::Clamp
        );
        assert!(!cfg.retrieval_documents);
    }
}
