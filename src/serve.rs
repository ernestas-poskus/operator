//! `operator serve`: stdio MCP only.
use std::path::PathBuf;
use std::sync::Arc;

use crate::cli::ServeArgs;
use crate::error::{Error, Result};
use crate::mcp_register;
use crate::workspace_config::WorkspaceConfig;

pub async fn run(args: ServeArgs) -> Result<()> {
    let root = args.workspace_root.unwrap_or_else(|| PathBuf::from("."));
    let workspace = std::fs::canonicalize(&root).unwrap_or(root);
    let ws = WorkspaceConfig::load(&workspace)?
        .ok_or_else(|| Error::Config("no .operator.toml; run `operator init` first".into()))?;
    let cfg = ws.into_config(&workspace);
    validate(&cfg)?;
    if !args.no_register {
        let transport = mcp_register::Transport::Stdio {
            argv: vec!["serve".into()],
        };
        let _ = mcp_register::register_operator(
            &workspace,
            &transport,
            mcp_register::Mode::Reconcile,
            "serve",
        );
    }
    crate::fastembed::prepare_cache(&cfg);
    let index = crate::code_search::LocalCodeIndex::build(&cfg).await;
    let state = Arc::new(crate::tools::ToolState {
        journal: crate::learn::Journal::open(&workspace),
        cfg,
        index,
    });
    crate::mcp_server::OperatorMcpServer::new(state)
        .serve_stdio()
        .await
}

pub fn validate(cfg: &crate::config::Config) -> Result<()> {
    if cfg.languages.is_empty() {
        return Err(Error::Config(
            "no languages selected; run `operator init`".into(),
        ));
    }
    let root = std::fs::canonicalize(&cfg.workspace_root)
        .map_err(|e| Error::Config(format!("workspace: {e}")))?;
    crate::fastembed::validate(cfg).map_err(Error::Config)?;
    for path in cfg
        .retrieval_include
        .iter()
        .map(|path| path.as_str())
        .chain(cfg.retrieval_exclude.iter().map(|path| path.as_str()))
    {
        let canonical = std::fs::canonicalize(root.join(path))
            .map_err(|e| Error::Config(format!("path {path}: {e}")))?;
        if !canonical.starts_with(&root) {
            return Err(Error::Config(format!("path {path}: outside workspace")));
        }
    }
    Ok(())
}
