//! `code-intelligence serve`: stdio MCP only.
use std::sync::Arc;

use crate::cli::ServeArgs;
use crate::error::{Error, Result};
use crate::mcp_register;
use crate::types::{Argv, CommandArg, WorkspaceRoot};
use crate::workspace_config::WorkspaceConfig;

pub async fn run(args: ServeArgs) -> Result<()> {
    let root = args
        .workspace_root
        .unwrap_or_else(|| WorkspaceRoot::from("."));
    let workspace = std::fs::canonicalize(&root)
        .map(WorkspaceRoot::from)
        .unwrap_or(root);
    let ws = WorkspaceConfig::load(&workspace)?.ok_or_else(|| {
        Error::Config("no .code-intelligence.toml; run `code-intelligence init` first".into())
    })?;
    let cfg = ws.into_config(&workspace);
    validate(&cfg)?;
    if !args.no_register {
        let transport = mcp_register::Transport::Stdio {
            argv: Argv::new(vec![CommandArg::from("serve")]),
        };
        let _ = mcp_register::register_hosts(
            &workspace,
            &transport,
            mcp_register::Mode::Reconcile,
            "serve",
        );
    }
    let index = crate::code_search::LocalCodeIndex::build(&cfg).await;
    let state = Arc::new(crate::tools::ToolState {
        journal: crate::learn::Journal::open(&workspace),
        cfg,
        index,
    });
    crate::mcp_server::CodeIntelligenceMcpServer::new(state)
        .serve_stdio()
        .await
}

pub fn validate(cfg: &crate::config::Config) -> Result<()> {
    if cfg.languages.is_empty() {
        return Err(Error::Config(
            "no languages selected; run `code-intelligence init`".into(),
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
