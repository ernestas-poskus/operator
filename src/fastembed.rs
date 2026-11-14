//! The in-process retrieval backend: fastembed's candle models on the GPU.
//!
//! macOS uses Metal; Linux and Windows use CUDA. GPU is required — there is no
//! CPU fallback, so a missing device is reported by [`FastembedModels::build`]
//! and retrieval is disabled while the rest of the server keeps serving.
//!
//! The embedder is `mixedbread-ai/mxbai-embed-large-v1` (CLS + L2, 1024-d) and
//! the reranker `Qwen/Qwen3-Reranker-0.6B` (P("yes") in [0, 1]); both are HF
//! snapshots loaded from the first complete cache [`crate::model_cache`] finds,
//! or downloaded into the configured cache directory when none exists.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device};
use fastembed::{BertTextEmbedding, Qwen3Reranker, Qwen3TextEmbedding};

use crate::config::{Config, ModelDtype};
use crate::embedding::Embedded;

/// Bumped whenever the fastembed/candle model implementation changes: it is
/// part of the index identity, so a bump forces a re-embed.
pub const FASTEMBED_EMBEDDER_IMPL: &str = "candle-bert-1";
/// Identity fragment for the Qwen3 embedding loader (same repo layout, a
/// different pooling/config family; the two can never share an index).
const QWEN3_EMBEDDER_IMPL: &str = "candle-qwen3-1";

/// Queue depth for the embedder's internal forward passes.
const EMBED_BATCH_SIZE: usize = 32;

/// The legacy HTTP model ids, kept so an existing `.code-intelligence.toml` keeps
/// working after the port.
const LEGACY_EMBED_MODEL: &str = "mxbai-embed-large-v1";
const LEGACY_RERANK_MODEL: &str = "Qwen3-Reranker-0.6B";

/// The embedder repo id for `cfg`, mapping the legacy HTTP model id.
pub fn embed_repo(cfg: &Config) -> Result<String, String> {
    map_repo(
        &cfg.retrieval_embed_model,
        LEGACY_EMBED_MODEL,
        crate::code_search::DEFAULT_EMBED_MODEL,
    )
}

/// The reranker repo id for `cfg`, mapping the legacy HTTP model id.
pub fn rerank_repo(cfg: &Config) -> Result<String, String> {
    map_repo(
        &cfg.retrieval_rerank_model,
        LEGACY_RERANK_MODEL,
        crate::code_search::DEFAULT_RERANK_MODEL,
    )
}

fn map_repo(raw: &str, legacy: &str, default: &str) -> Result<String, String> {
    let raw = raw.trim();
    if raw == legacy || raw == default {
        return Ok(default.to_string());
    }
    match raw.split_once('/') {
        Some((org, name)) if !org.is_empty() && !name.is_empty() && !name.contains('/') => {
            Ok(raw.to_string())
        }
        _ => Err(format!(
            "`{raw}` is not a Hugging Face repo id (`org/name`) and is not the legacy model id"
        )),
    }
}

/// Validate the backend knobs before any model work (used by `serve::validate`).
pub fn validate(cfg: &Config) -> Result<(), String> {
    embed_repo(cfg)?;
    rerank_repo(cfg)?;
    Ok(())
}

/// The dense-vector-space identity baked into the LanceDB table and manifest.
pub fn embedder_identity(cfg: &Config) -> String {
    let repo = embed_repo(cfg).unwrap_or_else(|_| cfg.retrieval_embed_model.clone().into_inner());
    // The repo is hashed rather than spelled out: a LanceDB table name is
    // capped at 128 chars and the full HF repo id does not fit next to the
    // logic/chunker/language fragments.
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(repo.as_bytes());
    let repo_hash = crate::hash::hex_digest(digest);
    let impl_tag = match cfg.retrieval_embed_backend {
        crate::config::EmbedBackend::Bert => FASTEMBED_EMBEDDER_IMPL,
        crate::config::EmbedBackend::Qwen3 => QWEN3_EMBEDDER_IMPL,
    };
    let mut identity = format!("fastembed-{}-{impl_tag}", &repo_hash[..8]);
    if cfg.retrieval_embed_dim != crate::code_search::DENSE_DIM {
        identity.push_str(&format!("-d{}", cfg.retrieval_embed_dim));
    }
    if cfg.retrieval_model_dtype != crate::config::ModelDtype::F16 {
        identity.push_str(&format!("-{}", cfg.retrieval_model_dtype.label()));
    }
    identity
}

/// Human-readable identity for docs and eval records (the table uses the
/// short, hashed form from [`embedder_identity`]).
#[allow(dead_code)]
pub fn embedder_label(cfg: &Config) -> String {
    let repo = embed_repo(cfg).unwrap_or_else(|_| cfg.retrieval_embed_model.clone().into_inner());
    let impl_tag = match cfg.retrieval_embed_backend {
        crate::config::EmbedBackend::Bert => FASTEMBED_EMBEDDER_IMPL,
        crate::config::EmbedBackend::Qwen3 => QWEN3_EMBEDDER_IMPL,
    };
    format!("fastembed:{repo}:{impl_tag}").replace(['/', ':'], "-")
}

/// Resolve the model cache root for this configuration and pin it for the
/// process. Called before the index build.
pub fn prepare_cache(cfg: &Config) -> PathBuf {
    let embed = embed_repo(cfg).unwrap_or_else(|_| cfg.retrieval_embed_model.clone().into_inner());
    let rerank =
        rerank_repo(cfg).unwrap_or_else(|_| cfg.retrieval_rerank_model.clone().into_inner());
    crate::model_cache::resolve_and_pin(cfg, &[embed, rerank])
}

/// One loaded embedder from either fastembed candle loader.
enum LoadedEmbedder {
    Bert(Box<BertTextEmbedding>),
    Qwen3(Box<Qwen3TextEmbedding>),
}

impl LoadedEmbedder {
    fn embed(&self, texts: &[String]) -> candle_core::Result<Vec<Vec<f32>>> {
        match self {
            Self::Bert(model) => model.embed(texts),
            Self::Qwen3(model) => model.embed(texts),
        }
    }
}

/// The loaded models behind a mutex: one GPU, one serialized call at a time.
#[derive(Default)]
struct Loaded {
    embedder: Option<LoadedEmbedder>,
    reranker: Option<Qwen3Reranker>,
    /// The embedder's own output width before any Matryoshka truncation.
    model_dim: usize,
}

struct Inner {
    device: Device,
    dtype: DType,
    embed_repo: String,
    rerank_repo: String,
    embed_max_length: usize,
    rerank_max_length: usize,
    embed_backend: crate::config::EmbedBackend,
    query_prefix: String,
    rerank_instruction: Option<String>,
    rerank_char_budget: usize,
    embed_dim: usize,
    loaded: Mutex<Loaded>,
}

/// fastembed's candle backend on the platform GPU.
#[derive(Clone)]
pub struct FastembedModels {
    inner: Arc<Inner>,
}

impl FastembedModels {
    /// Resolve configuration and open the GPU device. The models themselves load
    /// in [`Embedded::preflight`] (embedder) and on the first rerank.
    pub fn build(cfg: &Config) -> Result<Self, String> {
        let embed_repo = embed_repo(cfg)?;
        let rerank_repo = rerank_repo(cfg)?;
        let device = platform_device()?;
        let dtype = match cfg.retrieval_model_dtype {
            ModelDtype::F16 => DType::F16,
            ModelDtype::F32 => DType::F32,
        };
        Ok(Self {
            inner: Arc::new(Inner {
                device,
                dtype,
                embed_repo,
                rerank_repo,
                embed_max_length: crate::code_search::EMBEDDER_MAX_TOKENS,
                rerank_max_length: cfg.retrieval_rerank_max_tokens.max(1).get(),
                embed_backend: cfg.retrieval_embed_backend,
                query_prefix: cfg.retrieval_embed_query_prefix.clone().into_inner(),
                rerank_instruction: cfg
                    .retrieval_rerank_instruction
                    .clone()
                    .map(crate::types::Instruction::into_inner)
                    .filter(|s| !s.trim().is_empty()),
                rerank_char_budget: crate::code_search::chunk_char_budget(
                    cfg.retrieval_rerank_max_tokens.get(),
                ),
                embed_dim: cfg.retrieval_embed_dim.get(),
                loaded: Mutex::new(Loaded::default()),
            }),
        })
    }

    fn ensure_embedder(&self, loaded: &mut Loaded) -> Result<(), String> {
        if loaded.embedder.is_some() {
            return Ok(());
        }
        let embedder = match self.inner.embed_backend {
            crate::config::EmbedBackend::Bert => LoadedEmbedder::Bert(Box::new(
                BertTextEmbedding::from_hf(
                    &self.inner.embed_repo,
                    &self.inner.device,
                    self.inner.dtype,
                    self.inner.embed_max_length,
                )
                .map_err(|e| format!("loading {}: {e}", self.inner.embed_repo))?
                .with_batch_size(EMBED_BATCH_SIZE),
            )),
            crate::config::EmbedBackend::Qwen3 => {
                let model = Qwen3TextEmbedding::from_hf(
                    &self.inner.embed_repo,
                    &self.inner.device,
                    self.inner.dtype,
                    self.inner.embed_max_length,
                )
                .map_err(|e| format!("loading {}: {e}", self.inner.embed_repo))?;
                // Qwen3 checkpoints are MRL-capable; shorten explicitly when
                // the configured width differs from the hidden width.
                let model = if model.output_dim() != self.inner.embed_dim {
                    model
                        .with_output_dim(self.inner.embed_dim)
                        .map_err(|e| format!("shortening {}: {e}", self.inner.embed_repo))?
                } else {
                    model
                };
                LoadedEmbedder::Qwen3(Box::new(model))
            }
        };
        let probe = embedder
            .embed(&["code-intelligence dimension probe".to_string()])
            .map_err(|e| format!("probing {} hidden size: {e}", self.inner.embed_repo))?;
        let model_dim = probe.first().map_or(0, Vec::len);
        if model_dim < self.inner.embed_dim {
            return Err(format!(
                "{} produces {model_dim}-d vectors but retrieval_embed_dim is {} (truncation can only shorten)",
                self.inner.embed_repo, self.inner.embed_dim
            ));
        }
        loaded.embedder = Some(embedder);
        loaded.model_dim = model_dim;
        Ok(())
    }

    fn ensure_reranker(&self, loaded: &mut Loaded) -> Result<(), String> {
        if loaded.reranker.is_some() {
            return Ok(());
        }
        let reranker = Qwen3Reranker::from_hf(
            &self.inner.rerank_repo,
            &self.inner.device,
            self.inner.dtype,
            self.inner.rerank_max_length,
        )
        .map_err(|e| format!("loading {}: {e}", self.inner.rerank_repo))?;
        loaded.reranker = Some(reranker);
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Loaded> {
        self.inner
            .loaded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn embed_blocking(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let mut loaded = self.lock();
        self.ensure_embedder(&mut loaded)?;
        let embedder = loaded.embedder.as_ref().expect("checked above");
        let mut vectors = embedder.embed(&texts).map_err(|e| e.to_string())?;
        // Matryoshka: a configured width shorter than the model's own output
        // keeps the leading dimensions and renormalizes (a no-op when equal).
        if self.inner.embed_dim < loaded.model_dim {
            truncate_vectors(&mut vectors, self.inner.embed_dim);
        }
        Ok(vectors)
    }

    fn embed_query_blocking(&self, query: String) -> Result<Vec<f32>, String> {
        let text = if self.inner.query_prefix.is_empty() {
            query
        } else {
            format!("{}{query}", self.inner.query_prefix)
        };
        let mut out = self.embed_blocking(vec![text])?;
        out.pop()
            .ok_or_else(|| "empty embedding for query".to_string())
    }

    fn rerank_blocking(
        &self,
        query: String,
        docs: Vec<String>,
    ) -> Result<Vec<(usize, f32)>, String> {
        let mut loaded = self.lock();
        self.ensure_reranker(&mut loaded)?;
        let reranker = loaded.reranker.as_ref().expect("checked above");
        let docs: Vec<String> = docs
            .into_iter()
            .map(|doc| crate::code_search::truncate_chars(&doc, self.inner.rerank_char_budget))
            .collect();
        let scores = reranker
            .score(self.inner.rerank_instruction.as_deref(), &query, &docs)
            .map_err(|e| e.to_string())?;
        Ok(scores.into_iter().enumerate().collect())
    }
}

#[async_trait::async_trait]
impl Embedded for FastembedModels {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.embed_blocking(texts))
            .await
            .map_err(|e| format!("embed task: {e}"))?
    }

    async fn embed_query(&self, query: String) -> Result<Vec<f32>, String> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.embed_query_blocking(query))
            .await
            .map_err(|e| format!("embed query task: {e}"))?
    }

    async fn rerank(&self, query: String, docs: Vec<String>) -> Result<Vec<(usize, f32)>, String> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.rerank_blocking(query, docs))
            .await
            .map_err(|e| format!("rerank task: {e}"))?
    }

    async fn preflight(&self, _need_rerank: bool) -> Result<(), String> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut loaded = this.lock();
            this.ensure_embedder(&mut loaded)
        })
        .await
        .map_err(|e| format!("model load task: {e}"))?
    }

    fn embed_dim(&self) -> usize {
        self.inner.embed_dim
    }
}

/// Keep each vector's leading `dim` components and renormalize. `dim` must be
/// non-zero; a zero or wider request is a programming error guarded by the
/// loader. Pure.
fn truncate_vectors(vectors: &mut [Vec<f32>], dim: usize) {
    if dim == 0 {
        return;
    }
    for vector in vectors {
        if vector.len() > dim {
            vector.truncate(dim);
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in vector.iter_mut() {
                *value /= norm;
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn platform_device() -> Result<Device, String> {
    Device::new_metal(0).map_err(|e| format!("fastembed needs a Metal GPU on macOS: {e}"))
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn platform_device() -> Result<Device, String> {
    Device::new_cuda(0)
        .map_err(|e| format!("fastembed needs a CUDA GPU (build with the CUDA toolkit): {e}"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn platform_device() -> Result<Device, String> {
    Err("fastembed supports macOS (Metal), Linux and Windows (CUDA) only".to_string())
}

#[cfg(test)]
mod tests {
    use super::{embed_repo, embedder_identity, map_repo, rerank_repo};
    use crate::config::Config;

    #[test]
    fn truncation_keeps_a_unit_leading_prefix() {
        let mut vectors = vec![vec![3.0, 4.0, 12.0], vec![0.0, 0.0, 0.0]];
        super::truncate_vectors(&mut vectors, 2);
        assert_eq!(vectors[0].len(), 2);
        assert!((vectors[0][0] - 0.6).abs() < 1e-6);
        assert!((vectors[0][1] - 0.8).abs() < 1e-6);
        assert_eq!(vectors[1], vec![0.0, 0.0]);
    }

    #[test]
    fn the_embedder_identity_names_the_loader_family() {
        let mut cfg = Config::test_default();
        cfg.retrieval_embed_backend = crate::config::EmbedBackend::Qwen3;
        assert!(embedder_identity(&cfg).contains("candle-qwen3-1"));
        cfg.retrieval_embed_backend = crate::config::EmbedBackend::Bert;
        assert!(embedder_identity(&cfg).contains("candle-bert-1"));
    }

    #[test]
    fn legacy_ids_map_to_hf_repos_and_org_name_ids_pass_through() {
        let mut cfg = Config::test_default();
        cfg.retrieval_embed_model = "mxbai-embed-large-v1".into();
        cfg.retrieval_rerank_model = "Qwen3-Reranker-0.6B".into();
        assert_eq!(
            embed_repo(&cfg).unwrap(),
            crate::code_search::DEFAULT_EMBED_MODEL
        );
        assert_eq!(
            rerank_repo(&cfg).unwrap(),
            crate::code_search::DEFAULT_RERANK_MODEL
        );
        assert_eq!(
            map_repo("org/name", "legacy", "org/default").unwrap(),
            "org/name"
        );
        assert!(map_repo("name-only", "legacy", "org/default").is_err());
    }

    #[test]
    fn identity_names_backend_repo_and_impl_and_is_path_safe() {
        let mut cfg = Config::test_default();
        cfg.retrieval_embed_model = "mxbai-embed-large-v1".into();
        let id = embedder_identity(&cfg);
        assert_eq!(id, "fastembed-ab98510c-candle-bert-1");
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
        let label = super::embedder_label(&cfg);
        assert!(label.contains("mxbai-embed-large-v1"));
    }
}
