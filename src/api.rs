//! Library facade for in-process code intelligence.
//!
//! [`Operator`] loads a workspace configuration, owns an optional local index,
//! and exposes the same read-only operations the MCP server publishes. The
//! binary keeps the CLI/TUI; library callers do not parse process arguments.
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;

use crate::code_search::{self, LocalCodeIndex};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::languages;
use crate::learn;
use crate::tools::{self, ToolState};
use crate::workspace_config::WorkspaceConfig;

pub use crate::ast_grep::{CodeMatch, OutlineFile, OutlineSymbol};
pub use crate::code_search::{IndexEvent, RetrievalIndexStatus, RetrievalRefresh};
pub use crate::learn::JournalStats;
pub use crate::tools::{SearchHit, SemanticHit};
pub use crate::types::*;

/// One detected language and its file count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LanguageSummary {
    pub id: LanguageId,
    pub name: LanguageName,
    pub files: FileCount,
}

/// Detect the ast-grep languages present under `workspace`.
pub fn detect_languages(workspace: &Path) -> Vec<LanguageSummary> {
    languages::detect(workspace)
        .into_iter()
        .map(|detected| LanguageSummary {
            id: LanguageId::from_lang(detected.lang),
            name: LanguageName::from(languages::display_name(detected.lang)),
            files: detected.files,
        })
        .collect()
}

/// An opened workspace and its optional local retrieval index.
///
/// Construct with [`Operator::open`] for a configured workspace (index built
/// when the models and storage are available), [`Operator::open_unindexed`] for
/// lexical/AST-only use, or [`Operator::from_config`] for a programmatic
/// [`Config`]. Search methods are read-only; only [`Operator::reindex`] mutates
/// the in-memory state.
pub struct Operator {
    state: Arc<ToolState>,
}

impl std::fmt::Debug for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Operator")
            .field("workspace_root", &self.state.cfg.workspace_root)
            .field("has_index", &self.has_index())
            .finish()
    }
}

impl Operator {
    /// Open a workspace with `.operator.toml` and attempt to build the index.
    ///
    /// A missing/invalid configuration is an error. An index build failure is
    /// not: the operator remains usable for outline, literal/regex/AST and
    /// source reads, and [`Operator::has_index`] reports false.
    pub async fn open(workspace: impl AsRef<Path>) -> Result<Self> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let ws = WorkspaceConfig::load(&workspace)?
            .ok_or_else(|| Error::Config("no .operator.toml; run `operator init` first".into()))?;
        Self::from_config(ws.into_config(&workspace)).await
    }

    /// Open a configured workspace without loading models or building an index.
    pub fn open_unindexed(workspace: impl AsRef<Path>) -> Result<Self> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let ws = WorkspaceConfig::load(&workspace)?
            .ok_or_else(|| Error::Config("no .operator.toml; run `operator init` first".into()))?;
        Self::from_config_unindexed(ws.into_config(&workspace))
    }

    /// Build an operator from a programmatic configuration and attempt to build
    /// the index.
    pub async fn from_config(cfg: Config) -> Result<Self> {
        let cfg = canonical_config(cfg)?;
        crate::serve::validate(&cfg)?;
        crate::fastembed::prepare_cache(&cfg);
        let index = LocalCodeIndex::build(&cfg).await;
        Ok(Self {
            state: Arc::new(ToolState {
                journal: learn::Journal::open(&cfg.workspace_root),
                cfg,
                index,
            }),
        })
    }

    /// Build an operator from a programmatic configuration without an index.
    pub fn from_config_unindexed(cfg: Config) -> Result<Self> {
        let cfg = canonical_config(cfg)?;
        crate::serve::validate(&cfg)?;
        Ok(Self {
            state: Arc::new(ToolState {
                journal: learn::Journal::open(&cfg.workspace_root),
                cfg,
                index: None,
            }),
        })
    }

    pub fn config(&self) -> &Config {
        &self.state.cfg
    }

    pub fn workspace_root(&self) -> &Path {
        &self.state.cfg.workspace_root
    }

    pub fn has_index(&self) -> bool {
        self.state.index.is_some()
    }

    /// Read-only ast-grep outline. `limit` is clamped to `1..=1000`.
    pub fn repo_map(
        &self,
        path: Option<&PathFilter>,
        language: Option<&LanguageId>,
        limit: Limit,
    ) -> Result<Vec<OutlineFile>> {
        tool(tools::repo_map(
            &self.state,
            path,
            language,
            clamp_limit(limit),
        ))
    }

    /// Read-only full-text, literal, or regex search.
    ///
    /// `mode` is `fts`, `literal`, or `regex`. Without an index, `fts` falls
    /// back to the workspace scanner. `limit` is clamped to `1..=1000`.
    pub async fn search_code(
        &self,
        query: &QueryText,
        mode: &str,
        path: Option<&PathFilter>,
        language: Option<&LanguageId>,
        limit: Limit,
    ) -> Result<Vec<SearchHit>> {
        tool(tools::search_code(&self.state, query, mode, path, language, clamp_limit(limit)).await)
    }

    /// Read-only ast-grep pattern or YAML-rule search. `limit` is clamped to
    /// `1..=1000`.
    pub fn search_ast(
        &self,
        pattern: Option<&PatternText>,
        rule: Option<&RuleText>,
        path: Option<&PathsFilter>,
        language: Option<&LanguageId>,
        limit: Limit,
    ) -> Result<Vec<CodeMatch>> {
        tool(tools::search_ast(
            &self.state,
            pattern,
            rule,
            path,
            language,
            clamp_limit(limit),
        ))
    }

    /// Semantic search. Requires an index; returns a tool error otherwise.
    /// `limit` is clamped to `1..=1000`.
    pub async fn search_semantic(
        &self,
        query: &QueryText,
        path: Option<&PathFilter>,
        language: Option<&LanguageId>,
        limit: Limit,
    ) -> Result<Vec<SemanticHit>> {
        tool(tools::search_semantic(&self.state, query, path, language, clamp_limit(limit)).await)
    }

    /// Read a bounded line range from a workspace-contained file.
    pub fn read_source(
        &self,
        path: &RepoPath,
        start_line: Option<LineNumber>,
        end_line: Option<LineNumber>,
    ) -> Result<String> {
        tool(tools::read_source(&self.state, path, start_line, end_line))
    }

    /// Inspect the on-disk index without building it.
    pub async fn index_status(&self) -> RetrievalIndexStatus {
        code_search::retrieval_index_status(&self.state.cfg).await
    }

    /// Build or refresh the index and replace the in-memory state.
    pub async fn reindex(&mut self) -> Result<RetrievalRefresh> {
        let cfg = self.state.cfg.clone();
        let before = code_search::load_manifest(&cfg);
        let index = LocalCodeIndex::build(&cfg).await.ok_or_else(|| {
            Error::Tool(
                "the retrieval index could not be built (no source files, models unavailable, \
                 or LanceDB refused)"
                    .into(),
            )
        })?;
        let after = code_search::load_manifest(&cfg);
        let (embedded, dropped) = code_search::manifest_delta(&before, &after);
        self.state = Arc::new(ToolState {
            journal: self.state.journal.clone(),
            cfg,
            index: Some(index),
        });
        Ok(RetrievalRefresh {
            embedded: FileCount::new(embedded),
            dropped: FileCount::new(dropped),
            indexed: FileCount::new(after.len()),
        })
    }

    /// The local MCP usage journal.
    pub fn journal_stats(&self) -> Result<JournalStats> {
        learn::stats(&self.state.cfg.workspace_root)
    }

    /// Serve the five tools over stdio MCP from this in-process state.
    pub async fn serve_stdio(self) -> Result<()> {
        crate::mcp_server::OperatorMcpServer::new(self.state)
            .serve_stdio()
            .await
    }
}

fn tool<T>(result: std::result::Result<T, String>) -> Result<T> {
    result.map_err(Error::Tool)
}

fn clamp_limit(limit: Limit) -> Limit {
    limit.clamp_to(1, 1000)
}

fn canonical_workspace(workspace: &Path) -> Result<crate::types::WorkspaceRoot> {
    std::fs::canonicalize(workspace)
        .map(crate::types::WorkspaceRoot::from)
        .map_err(|error| Error::Config(format!("workspace {}: {error}", workspace.display())))
}

fn canonical_config(mut cfg: Config) -> Result<Config> {
    cfg.workspace_root = canonical_workspace(&cfg.workspace_root)?;
    Ok(cfg)
}
