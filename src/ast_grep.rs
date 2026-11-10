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
use crate::types::{
    ColumnNumber, LanguageId, Limit, LineNumber, PathFilter, PathsFilter, PatternText, RepoPath,
    RuleText, Signature, SourceText, SymbolKind, SymbolName,
};

#[derive(Debug, Clone, Serialize)]
pub struct CodeMatch {
    pub path: RepoPath,
    pub language: LanguageId,
    pub line: LineNumber,
    pub column: ColumnNumber,
    pub end_line: LineNumber,
    pub text: SourceText,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutlineSymbol {
    pub path: RepoPath,
    pub language: LanguageId,
    pub name: SymbolName,
    pub kind: SymbolKind,
    pub line: LineNumber,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutlineFile {
    pub path: RepoPath,
    pub language: LanguageId,
    pub symbols: Vec<OutlineSymbol>,
}

fn selected_languages(selected: &[LanguageId]) -> Vec<SupportLang> {
    selected
        .iter()
        .filter_map(|id| languages::parse_id(id.as_str()))
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
    selected: &[LanguageId],
    pattern: Option<&PatternText>,
    rule: Option<&RuleText>,
    path_filter: Option<&PathsFilter>,
    limit: Limit,
) -> Result<Vec<CodeMatch>, String> {
    if pattern.is_some() == rule.is_some() {
        return Err("exactly one of `pattern` or `rule` is required".into());
    }
    let languages = selected_languages(selected);
    if languages.is_empty() {
        return Err("no languages selected".into());
    }
    let mut out = Vec::new();
    for (path, lang) in walk(root, &languages, path_filter.map(PathsFilter::as_str)) {
        if out.len() >= limit.get() {
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
            let pattern = Pattern::new(pattern.as_str(), lang);
            matched.extend(ast.root().find_all(&pattern));
        } else if let Some(rule) = rule {
            let parsed = ast_grep_config::from_yaml_string::<SupportLang>(
                rule.as_str(),
                &GlobalRules::default(),
            )
            .map_err(|e| format!("rule parse failed: {e}"))?;
            let scan = CombinedScan::new(parsed.iter().collect());
            let result = scan.scan(&ast, false);
            for (_, matches) in result.matches {
                matched.extend(matches);
            }
        }
        for m in matched {
            if out.len() >= limit.get() {
                break;
            }
            let start = m.start_pos();
            let end = m.end_pos();
            out.push(CodeMatch {
                path: RepoPath::from(rel.clone()),
                language: LanguageId::from_lang(lang),
                line: LineNumber::new(start.line() as u32 + 1),
                column: ColumnNumber::new(1),
                end_line: LineNumber::new(end.line() as u32 + 1),
                text: SourceText::new(m.text().to_string()),
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
    selected: &[LanguageId],
    path_filter: Option<&PathFilter>,
    limit: Limit,
) -> Result<Vec<OutlineFile>, String> {
    let languages = selected_languages(selected);
    if languages.is_empty() {
        return Err("no languages selected".into());
    }
    let extractors = outline_extractors();
    let mut files = Vec::new();
    let mut symbols = 0usize;
    for (path, lang) in walk(root, &languages, path_filter.map(PathFilter::as_str)) {
        if symbols >= limit.get() {
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
            path: RepoPath::from(rel),
            language: LanguageId::from_lang(lang),
            symbols: Vec::new(),
        };
        if let Some(extractor) = extractors.get(languages::canonical_id(lang)) {
            for item in extractor.extract(ast.root()) {
                if symbols >= limit.get() {
                    break;
                }
                file.symbols.push(OutlineSymbol {
                    path: file.path.clone(),
                    language: file.language.clone(),
                    name: SymbolName::new(item.entry.name.to_string()),
                    kind: SymbolKind::new(format!("{:?}", item.entry.symbol_type)),
                    line: LineNumber::new(item.entry.range.start.line as u32 + 1),
                    signature: Signature::new(item.entry.signature.to_string()),
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
        let selected = vec![
            LanguageId::parse("rust").expect("catalog id"),
            LanguageId::parse("go").expect("catalog id"),
        ];
        let hits = search(
            dir.path(),
            &selected,
            Some(&PatternText::from("fn $NAME() { $$$B }")),
            None,
            None,
            Limit::new(10),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].language, "rust");
    }

    #[test]
    fn rule_search_works_and_outline_falls_back_for_dart() {
        let dir = fixture();
        let selected = vec![LanguageId::parse("dart").expect("catalog id")];
        let rule = "id: any-fn\nlanguage: Dart\nrule:\n  pattern: int $N($$$A) => $$$B;";
        let hits = search(
            dir.path(),
            &selected,
            None,
            Some(&RuleText::from(rule)),
            None,
            Limit::new(10),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        let files = outline(dir.path(), &selected, None, Limit::new(10)).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].language, "dart");
    }
}
