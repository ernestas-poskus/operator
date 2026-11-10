//! The five MCP tool implementations.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignore::WalkBuilder;
use regex::Regex;
use serde::Serialize;

use crate::ast_grep::{self, CodeMatch, OutlineFile};
use crate::code_search::LocalCodeIndex;
use crate::config::Config;
use crate::languages;
use crate::types::LanguageId;

pub struct ToolState {
    pub cfg: Config,
    pub index: Option<Arc<LocalCodeIndex>>,
    pub journal: crate::learn::Journal,
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub path: String,
    pub line: usize,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct SemanticHit {
    pub path: String,
    pub line: u32,
    pub score: f32,
    pub content: String,
}

fn path_language(path: &Path) -> Option<&'static str> {
    languages::from_path(path).map(languages::canonical_id)
}

fn selected(cfg: &Config, path: &Path) -> bool {
    let Some(lang) = languages::from_path(path) else {
        return false;
    };
    let id = languages::canonical_id(lang);
    cfg.languages.iter().any(|l| l.as_str() == id)
}

fn walk(cfg: &Config, path_filter: Option<&str>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let walker = WalkBuilder::new(&cfg.workspace_root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true)
        .require_git(false)
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".git" | "node_modules" | "target" | "vendor" | ".cache"
            )
        })
        .build();
    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if !selected(cfg, path) {
            continue;
        }
        if let Some(filter) = path_filter {
            let rel = path
                .strip_prefix(&cfg.workspace_root)
                .unwrap_or(path)
                .to_string_lossy();
            if !rel.contains(filter) {
                continue;
            }
        }
        out.push(path.to_path_buf());
        if out.len() >= 20_000 {
            break;
        }
    }
    out
}

pub fn repo_map(
    state: &ToolState,
    path: Option<&str>,
    language: Option<&str>,
    limit: usize,
) -> Result<Vec<OutlineFile>, String> {
    let selected: Vec<LanguageId> = match language {
        Some(lang) => {
            vec![LanguageId::parse(lang).ok_or_else(|| format!("unsupported language `{lang}`"))?]
        }
        None => state.cfg.languages.clone(),
    };
    ast_grep::outline(&state.cfg.workspace_root, &selected, path, limit)
}

pub async fn search_code(
    state: &ToolState,
    query: &str,
    mode: &str,
    path: Option<&str>,
    language: Option<&str>,
    limit: usize,
) -> Result<Vec<SearchHit>, String> {
    match mode {
        "fts" => {
            if let Some(index) = &state.index {
                let hits = index
                    .hybrid_search(query, limit, None, state.cfg.retrieval_dense_weight)
                    .await
                    .map_err(|e| format!("index search failed: {e}"))?;
                return Ok(hits
                    .into_iter()
                    .filter(|(chunk, _)| {
                        language
                            .is_none_or(|lang| path_language(Path::new(&chunk.path)) == Some(lang))
                    })
                    .map(|(chunk, _)| SearchHit {
                        path: chunk.path,
                        line: chunk.start_line as usize,
                        text: chunk.content,
                    })
                    .collect());
            }
            scan_text(state, query, false, path, language, limit)
        }
        "literal" => scan_text(state, query, false, path, language, limit),
        "regex" => scan_text(state, query, true, path, language, limit),
        other => Err(format!(
            "unknown search mode `{other}`; use fts, literal or regex"
        )),
    }
}

fn scan_text(
    state: &ToolState,
    query: &str,
    regex: bool,
    path: Option<&str>,
    language: Option<&str>,
    limit: usize,
) -> Result<Vec<SearchHit>, String> {
    let matcher = if regex {
        Some(Regex::new(query).map_err(|e| format!("invalid regex: {e}"))?)
    } else {
        None
    };
    let mut out = Vec::new();
    for file in walk(&state.cfg, path) {
        if let Some(lang) = language
            && path_language(&file) != Some(lang)
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let rel = file
            .strip_prefix(&state.cfg.workspace_root)
            .unwrap_or(&file)
            .to_string_lossy()
            .into_owned();
        for (idx, line) in text.lines().enumerate() {
            let hit = match &matcher {
                Some(re) => re.is_match(line),
                None => line.contains(query),
            };
            if hit {
                out.push(SearchHit {
                    path: rel.clone(),
                    line: idx + 1,
                    text: line.chars().take(400).collect(),
                });
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
    }
    Ok(out)
}

pub fn search_ast(
    state: &ToolState,
    pattern: Option<&str>,
    rule: Option<&str>,
    path: Option<&str>,
    language: Option<&str>,
    limit: usize,
) -> Result<Vec<CodeMatch>, String> {
    let selected: Vec<LanguageId> = match language {
        Some(lang) => {
            vec![LanguageId::parse(lang).ok_or_else(|| format!("unsupported language `{lang}`"))?]
        }
        None => state.cfg.languages.clone(),
    };
    ast_grep::search(
        &state.cfg.workspace_root,
        &selected,
        pattern,
        rule,
        path,
        limit,
    )
}

pub async fn search_semantic(
    state: &ToolState,
    query: &str,
    path: Option<&str>,
    language: Option<&str>,
    limit: usize,
) -> Result<Vec<SemanticHit>, String> {
    if let Some(lang) = language
        && !state.cfg.languages.iter().any(|l| l.as_str() == lang)
    {
        return Err(format!("language `{lang}` is not selected"));
    }
    let index = state
        .index
        .as_ref()
        .ok_or_else(|| "index_missing: run `operator reindex` first".to_string())?;
    let hits = index
        .hybrid_search(query, limit, None, state.cfg.retrieval_dense_weight)
        .await
        .map_err(|e| format!("semantic search failed: {e}"))?;
    let mut out: Vec<SemanticHit> = hits
        .into_iter()
        .map(|(chunk, score)| SemanticHit {
            path: chunk.path,
            line: chunk.start_line,
            score,
            content: chunk.content,
        })
        .collect();
    if let Some(filter) = path {
        out.retain(|hit| hit.path.contains(filter));
    }
    if let Some(lang) = language {
        out.retain(|hit| path_language(Path::new(&hit.path)) == Some(lang));
    }
    Ok(out)
}

pub fn read_source(
    state: &ToolState,
    path: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Result<String, String> {
    let root =
        std::fs::canonicalize(&state.cfg.workspace_root).map_err(|e| format!("workspace: {e}"))?;
    let candidate = root.join(path);
    let canonical = std::fs::canonicalize(&candidate).map_err(|e| format!("{path}: {e}"))?;
    if !canonical.starts_with(&root) {
        return Err("path_outside_workspace".into());
    }
    let text = std::fs::read_to_string(&canonical).map_err(|e| format!("{path}: {e}"))?;
    let lines: Vec<&str> = text.lines().collect();
    let start = start_line.unwrap_or(1).saturating_sub(1).min(lines.len());
    let end = end_line.unwrap_or(lines.len()).min(lines.len());
    Ok(lines[start..end].join("\n"))
}
