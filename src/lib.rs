//! `operator`: a read-only code-intelligence library and MCP server.
//!
//! The binary keeps the CLI and optional TUI; library callers use
//! [`Operator`] for in-process outline, AST, lexical/semantic search and source
//! reads. `operator::Config`/`WorkspaceConfig` describe a workspace, and
//! `Operator::serve_stdio` publishes the same five tools over MCP.
pub mod api;
mod ast_grep;
mod chunk_context;
mod cli;
mod code_search;
pub mod config;
mod document;
mod embedding;
pub mod error;
mod fastembed;
mod hash;
mod init;
mod languages;
mod learn;
mod mcp_register;
mod mcp_server;
mod model_cache;
mod retrieval;
mod serve;
mod symbol_chunker;
mod tools;
#[cfg(feature = "tui")]
mod tui;
mod types;
pub mod workspace_config;

#[cfg(feature = "tui")]
use std::io::IsTerminal;

use cli::Command;

pub use api::*;
pub use config::Config;
pub use error::{Error, Result};
pub use workspace_config::WorkspaceConfig;

pub async fn run() -> error::Result<()> {
    let cli: cli::OperatorCli = argh::from_env();
    match cli.command {
        Command::Init(args) => init::run(args).await,
        #[cfg(feature = "tui")]
        Command::Tui(args) => tui::run(args).await,
        Command::Serve(args) => serve::run(args).await,
        Command::Reindex(args) => reindex(args).await,
        Command::Learn(args) => learn::run(args),
    }
}

async fn reindex(args: cli::ReindexArgs) -> error::Result<()> {
    let root = args
        .workspace_root
        .unwrap_or_else(|| crate::types::WorkspaceRoot::from("."));
    let workspace = std::fs::canonicalize(&root)
        .map(crate::types::WorkspaceRoot::from)
        .unwrap_or(root);
    let ws = workspace_config::WorkspaceConfig::load(&workspace)?.ok_or_else(|| {
        error::Error::Config("no .operator.toml; run `operator init` first".into())
    })?;
    let cfg = ws.into_config(&workspace);
    serve::validate(&cfg)?;
    if args.status {
        let status = code_search::retrieval_index_status(&cfg).await;
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "table": status.table,
                    "manifest_files": status.manifest_files,
                    "rows": status.rows,
                    "includes": status.includes,
                }))?
            );
        } else {
            println!("table          : {}", status.table);
            println!("manifest files : {}", status.manifest_files);
            println!(
                "rows           : {}",
                status
                    .rows
                    .map_or_else(|| "not built".into(), |n| n.to_string())
            );
        }
        return Ok(());
    }
    #[cfg(feature = "tui")]
    if args.tui {
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            return Err(error::Error::Config("--tui requires a terminal".into()));
        }
        return crate::tui::run_reindex(workspace, args.rebuild).await;
    }
    #[cfg(not(feature = "tui"))]
    if args.tui {
        return Err(error::Error::Config(
            "this build was compiled without the `tui` feature".into(),
        ));
    }
    if args.rebuild {
        code_search::drop_retrieval_index(&cfg)
            .await
            .map_err(error::Error::Msg)?;
    }
    let refresh = code_search::refresh_retrieval_index(&cfg)
        .await
        .map_err(error::Error::Msg)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "embedded": refresh.embedded,
                "dropped": refresh.dropped,
                "indexed": refresh.indexed,
            }))?
        );
    } else {
        println!(
            "reindex: embedded {} file(s), dropped {}, indexed {}",
            refresh.embedded, refresh.dropped, refresh.indexed
        );
    }
    Ok(())
}
