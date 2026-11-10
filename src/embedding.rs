//! The generic embedded-retrieval seam.
//!
//! `LocalCodeIndex` talks to an [`Embedded`] backend instead of a concrete model
//! crate: dense embedding, query embedding, cross-encoder reranking and a
//! preflight load check. [`crate::fastembed::FastembedModels`] is the shipped
//! implementation; tests provide fakes of this trait.

/// One in-process retrieval backend: a dense embedder and an optional
/// cross-encoder reranker.
///
/// Scores returned by [`Embedded::rerank`] are probabilities in `[0, 1]`, which
/// is what [`crate::code_search::RerankScoreNorm`] expects. The implementation
/// owns its device (Metal on macOS, CUDA on Linux/Windows) and serialises calls
/// onto it.
#[async_trait::async_trait]
pub trait Embedded: Send + Sync + 'static {
    /// Embed documents (chunks). The returned vectors are L2-normalized and in
    /// the order of `texts`.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String>;

    /// Embed one search query, applying the configured query-side instruction
    /// prefix (if any).
    async fn embed_query(&self, query: String) -> Result<Vec<f32>, String>;

    /// Cross-encoder scores for `docs` against `query`: `(input index, score)`
    /// pairs with scores in `[0, 1]`.
    async fn rerank(&self, query: String, docs: Vec<String>) -> Result<Vec<(usize, f32)>, String>;

    /// Verify the backend can serve (`need_rerank` says whether reranking will
    /// be used). Loading failures are reported here instead of per call.
    async fn preflight(&self, need_rerank: bool) -> Result<(), String>;

    /// The dense vector width this backend produces.
    fn embed_dim(&self) -> usize;
}
