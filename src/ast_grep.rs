//! In-process ast-grep search and outline.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ast_grep_config::{CombinedScan, GlobalRules};
use ast_grep_core::{NodeMatch, Pattern};
use ast_grep_language::{LanguageExt, SupportLang};
use ast_grep_outline::DEFAULT_OUTLINE_RULES;
use ast_grep_outline::combined_extractor::CombinedExtractors;
use ast_grep_outline::extractor::parse_outline_rules;
use ignore::WalkBuilder;
use serde::Serialize;

use crate::languages;

#[derive(Debug, Clone, Serialize)]
pub struct CodeMatch {
    pub path: String,
    pub language: String,
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutlineSymbol {
    pub path: String,
    pub language: String,
    pub name: String,
    pub kind: String,
    pub line: usize,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutlineFile {
    pub path: String,
    pub language: String,
    pub symbols: Vec<OutlineSymbol>,
}

fn selected_languages(languages: &[String]) -> Vec<SupportLang> {
    languages
        .iter()
        .filter_map(|id| languages::parse_id(id))
        .collect()
}

fn walk(
    root: &Path,
    selected: &[SupportLang],
    path_filter: Option<&str>,
) -> Vec<(PathBuf, SupportLang)> {
    let mut out = Vec::new();
    let filter = path_filter.map(str::to_string);
    let walker = WalkBuilder::new(root)
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
        if let Some(filter) = &filter {
            let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
            if !rel.contains(filter.as_str()) {
                continue;
            }
        }
        let Some(lang) = languages::from_path(path) else {
            continue;
        };
        if !selected.contains(&lang) {
            continue;
        }
        out.push((path.to_path_buf(), lang));
    }
    out
}

pub fn search(
    root: &Path,
    selected: &[String],
    pattern: Option<&str>,
    rule: Option<&str>,
    path_filter: Option<&str>,
    limit: usize,
) -> Result<Vec<CodeMatch>, String> {
    if pattern.is_some() == rule.is_some() {
        return Err("exactly one of `pattern` or `rule` is required".into());
    }
    let languages = selected_languages(selected);
    if languages.is_empty() {
        return Err("no languages selected".into());
    }
    let mut out = Vec::new();
    for (path, lang) in walk(root, &languages, path_filter) {
        if out.len() >= limit {
            break;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let ast = lang.ast_grep(&text);
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let mut matched: Vec<NodeMatch<'_, _>> = Vec::new();
        if let Some(pattern) = pattern {
            let pattern = Pattern::new(pattern, lang);
            matched.extend(ast.root().find_all(&pattern));
        } else if let Some(rule) = rule {
            let parsed =
                ast_grep_config::from_yaml_string::<SupportLang>(rule, &GlobalRules::default())
                    .map_err(|e| format!("rule parse failed: {e}"))?;
            let scan = CombinedScan::new(parsed.iter().collect());
            let result = scan.scan(&ast, false);
            for (_, matches) in result.matches {
                matched.extend(matches);
            }
        }
        for m in matched {
            if out.len() >= limit {
                break;
            }
            let start = m.start_pos();
            let end = m.end_pos();
            out.push(CodeMatch {
                path: rel.clone(),
                language: languages::canonical_id(lang).to_string(),
                line: start.line() + 1,
                column: 1,
                end_line: end.line() + 1,
                text: m.text().to_string(),
            });
        }
    }
    Ok(out)
}

fn outline_extractors() -> &'static BTreeMap<&'static str, CombinedExtractors<SupportLang>> {
    static EXTRACTORS: OnceLock<BTreeMap<&'static str, CombinedExtractors<SupportLang>>> =
        OnceLock::new();
    EXTRACTORS.get_or_init(|| {
        let mut map = BTreeMap::new();
        let Ok(rules) = parse_outline_rules::<SupportLang>(DEFAULT_OUTLINE_RULES) else {
            return map;
        };
        for lang in SupportLang::all_langs() {
            let mine: Vec<_> = rules
                .iter()
                .filter(|rule| rule.common().language == *lang)
                .cloned()
                .collect();
            if mine.is_empty() {
                continue;
            }
            if let Ok(extractor) = CombinedExtractors::try_from(mine, &Default::default()) {
                map.insert(languages::canonical_id(*lang), extractor);
            }
        }
        map
    })
}

pub fn outline(
    root: &Path,
    selected: &[String],
    path_filter: Option<&str>,
    limit: usize,
) -> Result<Vec<OutlineFile>, String> {
    let languages = selected_languages(selected);
    if languages.is_empty() {
        return Err("no languages selected".into());
    }
    let extractors = outline_extractors();
    let mut files = Vec::new();
    let mut symbols = 0usize;
    for (path, lang) in walk(root, &languages, path_filter) {
        if symbols >= limit {
            break;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let ast = lang.ast_grep(&text);
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let mut file = OutlineFile {
            path: rel,
            language: languages::canonical_id(lang).to_string(),
            symbols: Vec::new(),
        };
        if let Some(extractor) = extractors.get(languages::canonical_id(lang)) {
            for item in extractor.extract(ast.root()) {
                if symbols >= limit {
                    break;
                }
                file.symbols.push(OutlineSymbol {
                    path: file.path.clone(),
                    language: file.language.clone(),
                    name: item.entry.name.to_string(),
                    kind: format!("{:?}", item.entry.symbol_type),
                    line: item.entry.range.start.line + 1,
                    signature: item.entry.signature.to_string(),
                });
                symbols += 1;
            }
        }
        // Languages without a bundled outline rule still appear in the map,
        // so "supported" never looks like "no files found".
        files.push(file);
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn alpha() {}\n").unwrap();
        std::fs::write(dir.path().join("a.go"), "package main\nfunc Beta() {}\n").unwrap();
        std::fs::write(dir.path().join("a.dart"), "int gamma() => 1;\n").unwrap();
        dir
    }

    #[test]
    fn pattern_search_is_language_scoped_and_bounded() {
        let dir = fixture();
        let selected = vec!["rust".to_string(), "go".to_string()];
        let hits = search(
            dir.path(),
            &selected,
            Some("fn $NAME() { $$$B }"),
            None,
            None,
            10,
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].language, "rust");
    }

    #[test]
    fn rule_search_works_and_outline_falls_back_for_dart() {
        let dir = fixture();
        let selected = vec!["dart".to_string()];
        let rule = "id: any-fn\nlanguage: Dart\nrule:\n  pattern: int $N($$$A) => $$$B;";
        let hits = search(dir.path(), &selected, None, Some(rule), None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        let files = outline(dir.path(), &selected, None, 10).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].language, "dart");
    }
}
