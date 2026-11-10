//! `operator init`: detect ast-grep languages and persist the selection.
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use dialoguer::MultiSelect;

use crate::cli::InitArgs;
use crate::error::{Error, Result};
use crate::languages;
use crate::mcp_register;
use crate::workspace_config::WorkspaceConfig;

pub async fn run(args: InitArgs) -> Result<()> {
    let requested = args.workspace_root.clone();
    let root = requested.clone().unwrap_or_else(|| PathBuf::from("."));
    let workspace = std::fs::canonicalize(&root).unwrap_or(root);
    #[cfg(feature = "tui")]
    let use_tui = args.tui
        || (!args.no_tui
            && !args.yes
            && std::io::stdin().is_terminal()
            && std::io::stdout().is_terminal());
    #[cfg(feature = "tui")]
    if use_tui {
        let mut tui_args = args;
        tui_args.workspace_root = requested;
        return crate::tui::run_init(tui_args).await;
    }
    #[cfg(not(feature = "tui"))]
    if args.tui {
        return Err(Error::Config(
            "this build was compiled without the `tui` feature".into(),
        ));
    }
    let detected = languages::detect(&workspace);
    if detected.is_empty() {
        return Err(Error::Config(format!(
            "no supported languages detected under {}; supported: {}",
            workspace.display(),
            languages::supported_ids().join(", ")
        )));
    }
    let selected = if let Some(raw) = args.languages {
        parse_selection(&raw)?
    } else if args.all_supported {
        languages::supported_ids()
            .into_iter()
            .map(str::to_string)
            .collect()
    } else if args.yes || !std::io::stdin().is_terminal() {
        detected
            .iter()
            .map(|d| languages::canonical_id(d.lang).to_string())
            .collect()
    } else {
        prompt(&detected)?
    };
    if selected.is_empty() {
        return Err(Error::Config("no languages selected".into()));
    }
    for id in &selected {
        if languages::parse_id(id).is_none() {
            return Err(Error::Config(format!("unsupported language `{id}`")));
        }
    }
    let include_paths = args
        .include
        .as_ref()
        .map(|raw| parse_paths(raw))
        .transpose()?;
    let exclude_paths = args
        .exclude
        .as_ref()
        .map(|raw| parse_paths(raw))
        .transpose()?;
    validate_paths(
        &workspace,
        include_paths.iter().chain(exclude_paths.iter()).flatten(),
    )?;
    let config = WorkspaceConfig {
        schema_version: Some(3),
        languages: Some(selected.clone()),
        retrieval_include: include_paths.clone(),
        retrieval_exclude: exclude_paths.clone(),
        retrieval_cache_dir: None,
        retrieval_model_cache_dir: None,
        retrieval_model_dtype: None,
        retrieval_embed_model: None,
        retrieval_embed_backend: None,
        retrieval_embed_dim: None,
        retrieval_rerank_model: None,
        retrieval_rerank_enabled: None,
        retrieval_rerank_context: None,
        retrieval_fusion_gate: None,
        retrieval_query_log: None,
        retrieval_dual_embed_model: None,
        retrieval_dual_embed_dim: None,
        retrieval_dual_embed_backend: None,
        retrieval_dual_embed_query_prefix: None,
        retrieval_dual_route: None,
        retrieval_rerank_blend: None,
        retrieval_query_expansion: None,
        retrieval_query_routing: None,
        retrieval_chunk_lines: None,
        retrieval_chunk_overlap: None,
    };
    if WorkspaceConfig::path_for(&workspace).exists() && !args.force {
        let existing = WorkspaceConfig::load(&workspace)?.unwrap_or_default();
        let mut merged = existing;
        merged.schema_version = Some(3);
        merged.languages = Some(selected.clone());
        if include_paths.is_some() {
            merged.retrieval_include = include_paths;
        }
        if exclude_paths.is_some() {
            merged.retrieval_exclude = exclude_paths;
        }
        merged.save(&workspace)?;
    } else {
        config.save(&workspace)?;
    }
    let transport = mcp_register::Transport::Stdio {
        argv: vec!["serve".into()],
    };
    let mode = if args.force {
        mcp_register::Mode::ReplaceFile
    } else {
        mcp_register::Mode::CreateOnly
    };
    mcp_register::register_operator(&workspace, &transport, mode, "init")?;
    println!(
        "init: selected {} language(s): {}",
        selected.len(),
        selected.join(", ")
    );
    println!(
        "init: wrote {}",
        WorkspaceConfig::path_for(&workspace).display()
    );
    println!("next: operator reindex");
    Ok(())
}

fn validate_paths<'a>(workspace: &Path, paths: impl Iterator<Item = &'a String>) -> Result<()> {
    let root = std::fs::canonicalize(workspace)
        .map_err(|e| Error::Config(format!("workspace {}: {e}", workspace.display())))?;
    for path in paths {
        let canonical = std::fs::canonicalize(root.join(path))
            .map_err(|e| Error::Config(format!("path {path}: {e}")))?;
        if !canonical.starts_with(&root) {
            return Err(Error::Config(format!("path {path}: outside workspace")));
        }
    }
    Ok(())
}

fn parse_paths(raw: &str) -> Result<Vec<String>> {
    Ok(raw
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| p.trim_start_matches("./").trim_end_matches('/').to_string())
        .collect())
}

fn parse_selection(raw: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let id = part.trim();
        if id.is_empty() {
            continue;
        }
        let lang = languages::parse_id(id)
            .ok_or_else(|| Error::Config(format!("unsupported language `{id}`")))?;
        out.push(languages::canonical_id(lang).to_string());
    }
    Ok(out)
}

fn prompt(detected: &[languages::DetectedLanguage]) -> Result<Vec<String>> {
    let items: Vec<String> = detected
        .iter()
        .map(|d| format!("{} ({} files)", languages::display_name(d.lang), d.files))
        .collect();
    let defaults: Vec<bool> = vec![true; items.len()];
    let picked = MultiSelect::new()
        .with_prompt("Languages to index")
        .items(&items)
        .defaults(&defaults)
        .interact()
        .map_err(|e| Error::Config(format!("language prompt: {e}")))?;
    Ok(picked
        .into_iter()
        .map(|i| languages::canonical_id(detected[i].lang).to_string())
        .collect())
}
