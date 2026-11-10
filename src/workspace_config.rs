//! `.operator.toml` — the selected languages and retrieval settings.
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::types::{
    BackendLabel, Blend, CacheDir, ChunkLines, ChunkOverlap, DualRoute, EmbedDim, ExcludePath,
    FusionGate, IncludePath, LanguageId, ModelCacheDir, ModelName, QueryExpansion, QueryLogPath,
    QueryPrefix, QueryRouting, RerankContext, RerankEnabled, SchemaVersion,
};

pub const CONFIG_FILE: &str = ".operator.toml";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkspaceConfig {
    pub schema_version: Option<SchemaVersion>,
    pub languages: Option<Vec<LanguageId>>,
    pub retrieval_include: Option<Vec<IncludePath>>,
    pub retrieval_exclude: Option<Vec<ExcludePath>>,
    pub retrieval_cache_dir: Option<CacheDir>,
    pub retrieval_model_cache_dir: Option<ModelCacheDir>,
    pub retrieval_model_dtype: Option<crate::config::ModelDtype>,
    pub retrieval_embed_model: Option<ModelName>,
    /// Backend loader for the embedder repo (`bert` | `qwen3`).
    #[serde(default)]
    pub retrieval_embed_backend: Option<BackendLabel>,
    /// Embedder output width when the model is not 1024-d.
    #[serde(default)]
    pub retrieval_embed_dim: Option<EmbedDim>,
    pub retrieval_rerank_model: Option<ModelName>,
    pub retrieval_rerank_enabled: Option<RerankEnabled>,
    pub retrieval_rerank_context: Option<RerankContext>,
    pub retrieval_fusion_gate: Option<FusionGate>,
    pub retrieval_query_log: Option<QueryLogPath>,
    pub retrieval_dual_embed_model: Option<ModelName>,
    pub retrieval_dual_embed_dim: Option<EmbedDim>,
    pub retrieval_dual_embed_backend: Option<BackendLabel>,
    pub retrieval_dual_embed_query_prefix: Option<QueryPrefix>,
    pub retrieval_dual_route: Option<DualRoute>,
    pub retrieval_rerank_blend: Option<Blend>,
    pub retrieval_query_expansion: Option<QueryExpansion>,
    pub retrieval_query_routing: Option<QueryRouting>,
    pub retrieval_chunk_lines: Option<ChunkLines>,
    pub retrieval_chunk_overlap: Option<ChunkOverlap>,
    /// Document conversion engine for the opt-in document corpus
    /// (`auto` | `anydoc` | `xberg`).
    pub retrieval_document_engine: Option<crate::document::DocumentEngine>,
    /// Fallback policy when the primary engine fails
    /// (`best-effort` | `strict`).
    pub retrieval_document_fallback: Option<crate::document::FallbackPolicy>,
}

impl WorkspaceConfig {
    pub fn path_for(workspace: &Path) -> PathBuf {
        workspace.join(CONFIG_FILE)
    }

    pub fn load(workspace: &Path) -> crate::error::Result<Option<Self>> {
        let path = Self::path_for(workspace);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let cfg = toml::from_str(&text)
            .map_err(|e| crate::error::Error::Config(format!("{}: {e}", path.display())))?;
        Ok(Some(cfg))
    }

    pub fn save(&self, workspace: &Path) -> crate::error::Result<()> {
        let path = Self::path_for(workspace);
        let text = toml::to_string_pretty(self)
            .map_err(|e| crate::error::Error::Config(format!("serialize config: {e}")))?;
        std::fs::write(&path, text)?;
        Ok(())
    }

    pub fn into_config(self, workspace: &Path) -> crate::config::Config {
        let mut cfg = crate::config::Config::new(workspace.to_path_buf());
        cfg.languages = self.languages.unwrap_or_default();
        if let Some(include) = self.retrieval_include {
            cfg.retrieval_include = include;
        }
        if let Some(exclude) = self.retrieval_exclude {
            cfg.retrieval_exclude = exclude;
        }
        if let Some(cache) = self.retrieval_cache_dir {
            cfg.retrieval_cache_dir = if cache.is_absolute() {
                cache
            } else {
                CacheDir::from(workspace.join(&cache))
            };
        }
        if let Some(dir) = self.retrieval_model_cache_dir {
            cfg.retrieval_model_cache_dir = Some(if dir.is_absolute() {
                dir
            } else {
                ModelCacheDir::from(workspace.join(&dir))
            });
        }
        if let Some(dtype) = self.retrieval_model_dtype {
            cfg.retrieval_model_dtype = dtype;
        }
        if let Some(model) = self.retrieval_embed_model {
            cfg.retrieval_embed_model = model;
        }
        if let Some(parsed) = self
            .retrieval_embed_backend
            .as_deref()
            .and_then(crate::config::EmbedBackend::parse)
        {
            cfg.retrieval_embed_backend = parsed;
        }
        if let Some(dim) = self.retrieval_embed_dim {
            cfg.retrieval_embed_dim = dim;
        }
        if let Some(model) = self.retrieval_rerank_model {
            cfg.retrieval_rerank_model = model;
        }
        if let Some(enabled) = self.retrieval_rerank_enabled {
            cfg.retrieval_rerank_enabled = enabled;
        }
        if let Some(context) = self.retrieval_rerank_context {
            cfg.retrieval_rerank_context = context;
        }
        if let Some(gate) = self.retrieval_fusion_gate {
            cfg.retrieval_fusion_gate = gate;
        }
        if let Some(path) = self
            .retrieval_query_log
            .filter(|p| !p.as_path().to_string_lossy().trim().is_empty())
        {
            cfg.retrieval_query_log = Some(path);
        }
        if let Some(model) = self
            .retrieval_dual_embed_model
            .filter(|m| !m.trim().is_empty())
        {
            cfg.retrieval_dual_embed_model = Some(model);
        }
        if let Some(dim) = self.retrieval_dual_embed_dim {
            cfg.retrieval_dual_embed_dim = Some(dim);
        }
        if let Some(backend) = self
            .retrieval_dual_embed_backend
            .as_deref()
            .and_then(crate::config::EmbedBackend::parse)
        {
            cfg.retrieval_dual_embed_backend = Some(backend);
        }
        if let Some(prefix) = self.retrieval_dual_embed_query_prefix {
            cfg.retrieval_dual_embed_query_prefix = Some(prefix);
        }
        if let Some(route) = self.retrieval_dual_route {
            cfg.retrieval_dual_route = route;
        }
        if let Some(blend) = self.retrieval_rerank_blend {
            cfg.retrieval_rerank_blend = blend.clamp_to(0.0, 1.0);
        }
        if let Some(expand) = self.retrieval_query_expansion {
            cfg.retrieval_query_expansion = expand;
        }
        if let Some(routing) = self.retrieval_query_routing {
            cfg.retrieval_query_routing = routing;
        }
        if let Some(lines) = self.retrieval_chunk_lines {
            cfg.retrieval_chunk_lines = lines;
        }
        if let Some(overlap) = self.retrieval_chunk_overlap {
            cfg.retrieval_chunk_overlap = overlap;
        }
        if let Some(engine) = self.retrieval_document_engine {
            cfg.retrieval_document_engine = engine;
        }
        if let Some(fallback) = self.retrieval_document_fallback {
            cfg.retrieval_document_fallback = fallback;
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::WorkspaceConfig;
    use crate::config::ModelDtype;
    use crate::types::{ChunkLines, EmbedDim, LanguageId, RerankEnabled, SchemaVersion};
    use std::path::Path;

    #[test]
    fn the_saved_toml_is_plain_primitives() {
        let cfg = WorkspaceConfig {
            schema_version: Some(SchemaVersion::new(3)),
            languages: Some(vec![LanguageId::parse("rust").expect("catalog id")]),
            retrieval_chunk_lines: Some(ChunkLines::new(50)),
            retrieval_rerank_enabled: Some(RerankEnabled::on()),
            ..Default::default()
        };
        let text = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(text.contains("schema_version = 3"), "{text}");
        assert!(text.contains("languages = [\"rust\"]"), "{text}");
        assert!(text.contains("retrieval_chunk_lines = 50"), "{text}");
        assert!(text.contains("retrieval_rerank_enabled = true"), "{text}");
        assert!(!text.contains("LanguageId"), "{text}");
        assert!(!text.contains("ChunkLines"), "{text}");
    }

    #[test]
    fn retrieval_model_knobs_merge_into_the_config() {
        let text = r#"
languages = ["rust"]
retrieval_model_dtype = "f32"
retrieval_model_cache_dir = "cache/models"
retrieval_embed_model = "mxbai-embed-large-v1"
"#;
        let ws: WorkspaceConfig = toml::from_str(text).unwrap();
        let cfg = ws.into_config(Path::new("/ws"));
        assert_eq!(cfg.retrieval_model_dtype, ModelDtype::F32);
        assert_eq!(
            cfg.retrieval_model_cache_dir.as_deref(),
            Some(Path::new("/ws/cache/models"))
        );
        // The legacy HTTP model id stays as configured and resolves when the backend
        // builds; identity resolution is covered in `fastembed`.
        assert_eq!(cfg.retrieval_embed_model, "mxbai-embed-large-v1");
    }

    #[test]
    fn an_absolute_model_cache_dir_is_kept_as_is() {
        let text = "retrieval_model_cache_dir = \"/models\"\n";
        let ws: WorkspaceConfig = toml::from_str(text).unwrap();
        let cfg = ws.into_config(Path::new("/ws"));
        assert_eq!(
            cfg.retrieval_model_cache_dir.as_deref(),
            Some(Path::new("/models"))
        );
    }

    #[test]
    fn an_unknown_dtype_is_a_parse_error() {
        let text = "retrieval_model_dtype = \"bf16\"\n";
        assert!(toml::from_str::<WorkspaceConfig>(text).is_err());
    }

    #[test]
    fn dual_index_knobs_merge_into_the_config() {
        let text = r#"
retrieval_dual_embed_model = "Qwen/Qwen3-Embedding-0.6B"
retrieval_dual_embed_dim = 1024
retrieval_dual_embed_backend = "qwen3"
retrieval_dual_embed_query_prefix = "query: "
retrieval_dual_route = false
"#;
        let ws: WorkspaceConfig = toml::from_str(text).unwrap();
        let cfg = ws.into_config(Path::new("/ws"));
        assert_eq!(
            cfg.retrieval_dual_embed_model.as_deref(),
            Some("Qwen/Qwen3-Embedding-0.6B")
        );
        assert_eq!(cfg.retrieval_dual_embed_dim.map(EmbedDim::get), Some(1024));
        assert_eq!(
            cfg.retrieval_dual_embed_backend,
            Some(crate::config::EmbedBackend::Qwen3)
        );
        assert_eq!(
            cfg.retrieval_dual_embed_query_prefix.as_deref(),
            Some("query: ")
        );
        assert!(!cfg.retrieval_dual_route.is_on());
    }

    #[test]
    fn removed_http_keys_are_ignored_for_backward_compatibility() {
        let text = r#"
retrieval_server_url = "http://127.0.0.1:8000"
retrieval_http_timeout_ms = 30000
retrieval_api_key = "secret"
"#;
        assert!(toml::from_str::<WorkspaceConfig>(text).is_ok());
    }
}
