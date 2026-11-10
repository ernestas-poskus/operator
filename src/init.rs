//! `operator init`: detect ast-grep languages and persist the selection.
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use demand::{DemandOption, MultiSelect};

use crate::cli::InitArgs;
use crate::error::{Error, Result};
use crate::languages;
use crate::mcp_register;
use crate::types::LanguageId;
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
    } else if args.yes || !std::io::stdin().is_terminal() {
        detected
            .iter()
            .map(|d| LanguageId::from_lang(d.lang))
            .collect()
    } else {
        prompt(&detected)?
    };
    if selected.is_empty() {
        return Err(Error::Config("no languages selected".into()));
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

fn parse_selection(raw: &str) -> Result<Vec<LanguageId>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let id = part.trim();
        if id.is_empty() {
            continue;
        }
        let lang = LanguageId::parse(id)
            .ok_or_else(|| Error::Config(format!("unsupported language `{id}`")))?;
        out.push(lang);
    }
    Ok(out)
}

/// Walk the detected languages and return the chosen ids.
///
/// The picker runs on stderr (demand draws there), so it can never corrupt a
/// host's MCP stdio stream.
fn prompt(detected: &[languages::DetectedLanguage]) -> Result<Vec<LanguageId>> {
    let mut picker = MultiSelect::new("Languages to index")
        .description("Space toggles a language, enter confirms. Detected languages start selected.")
        .min(1)
        .filterable(true);
    for entry in detected {
        let label = format!(
            "{} ({} files)",
            languages::display_name(entry.lang),
            entry.files
        );
        picker = picker.option(
            DemandOption::with_label(label, LanguageId::from_lang(entry.lang)).selected(true),
        );
    }
    picker.run().map_err(|error| match error.kind() {
        std::io::ErrorKind::Interrupted => Error::Config("language selection cancelled".into()),
        _ => Error::Config(format!("language prompt: {error}")),
    })
}
