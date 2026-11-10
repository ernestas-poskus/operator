//! Language catalog and workspace detection, driven by ast-grep.
use std::collections::BTreeMap;
use std::path::Path;

use ast_grep_core::language::Language;
use ast_grep_language::SupportLang;
use ignore::WalkBuilder;

pub fn canonical_id(lang: SupportLang) -> &'static str {
    // The Debug spelling is the stable variant name; lower-case it in the
    // catalog below so no second hand-maintained table can drift.
    match lang {
        SupportLang::Bash => "bash",
        SupportLang::C => "c",
        SupportLang::Cpp => "cpp",
        SupportLang::CSharp => "csharp",
        SupportLang::Css => "css",
        SupportLang::Dart => "dart",
        SupportLang::Go => "go",
        SupportLang::Elixir => "elixir",
        SupportLang::Haskell => "haskell",
        SupportLang::Hcl => "hcl",
        SupportLang::Html => "html",
        SupportLang::Java => "java",
        SupportLang::JavaScript => "javascript",
        SupportLang::Json => "json",
        SupportLang::Kotlin => "kotlin",
        SupportLang::Lua => "lua",
        SupportLang::Markdown => "markdown",
        SupportLang::Nix => "nix",
        SupportLang::Php => "php",
        SupportLang::Python => "python",
        SupportLang::Ruby => "ruby",
        SupportLang::Rust => "rust",
        SupportLang::Scala => "scala",
        SupportLang::Solidity => "solidity",
        SupportLang::Swift => "swift",
        SupportLang::Tsx => "tsx",
        SupportLang::TypeScript => "typescript",
        SupportLang::Yaml => "yaml",
        SupportLang::Zig => "zig",
    }
}

pub fn display_name(lang: SupportLang) -> &'static str {
    match lang {
        SupportLang::Bash => "Bash",
        SupportLang::C => "C",
        SupportLang::Cpp => "C++",
        SupportLang::CSharp => "C#",
        SupportLang::Css => "CSS",
        SupportLang::Dart => "Dart",
        SupportLang::Go => "Go",
        SupportLang::Elixir => "Elixir",
        SupportLang::Haskell => "Haskell",
        SupportLang::Hcl => "HCL",
        SupportLang::Html => "HTML",
        SupportLang::Java => "Java",
        SupportLang::JavaScript => "JavaScript",
        SupportLang::Json => "JSON",
        SupportLang::Kotlin => "Kotlin",
        SupportLang::Lua => "Lua",
        SupportLang::Markdown => "Markdown",
        SupportLang::Nix => "Nix",
        SupportLang::Php => "PHP",
        SupportLang::Python => "Python",
        SupportLang::Ruby => "Ruby",
        SupportLang::Rust => "Rust",
        SupportLang::Scala => "Scala",
        SupportLang::Solidity => "Solidity",
        SupportLang::Swift => "Swift",
        SupportLang::Tsx => "TSX",
        SupportLang::TypeScript => "TypeScript",
        SupportLang::Yaml => "YAML",
        SupportLang::Zig => "Zig",
    }
}

pub fn catalog() -> &'static [SupportLang] {
    SupportLang::all_langs()
}

pub fn parse_id(value: &str) -> Option<SupportLang> {
    let value = value.trim();
    value.parse::<SupportLang>().ok().or_else(|| {
        SupportLang::all_langs()
            .iter()
            .copied()
            .find(|lang| canonical_id(*lang).eq_ignore_ascii_case(value))
    })
}

pub fn from_path(path: &Path) -> Option<SupportLang> {
    SupportLang::from_path(path)
}

#[cfg(feature = "tui")]
pub fn language_id_for_path(path: &Path) -> Option<&'static str> {
    from_path(path).map(canonical_id)
}

#[derive(Debug, Clone)]
pub struct DetectedLanguage {
    pub lang: SupportLang,
    pub files: usize,
}

/// Walk the workspace with gitignore semantics and count files by ast-grep
/// language. Symlinks are followed; failures are ignored like the index walk.
pub fn detect(root: &Path) -> Vec<DetectedLanguage> {
    let mut counts: BTreeMap<&'static str, (SupportLang, usize)> = BTreeMap::new();
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
        let path = entry.path();
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Some(lang) = from_path(path) else {
            continue;
        };
        let id = canonical_id(lang);
        let slot = counts.entry(id).or_insert_with(|| (lang, 0));
        slot.1 += 1;
    }
    counts
        .into_values()
        .map(|(lang, files)| DetectedLanguage { lang, files })
        .collect()
}

pub fn supported_ids() -> Vec<&'static str> {
    catalog().iter().copied().map(canonical_id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_every_selected_language_including_dart() {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in [
            ("a.rs", "fn main() {}"),
            ("a.go", "package main\nfunc main() {}"),
            ("a.dart", "int add(int a, int b) => a + b;"),
            ("README.md", "# hi"),
        ] {
            let path = dir.path().join(name);
            std::fs::write(&path, body).unwrap();
        }
        let found: Vec<&str> = detect(dir.path())
            .iter()
            .map(|d| canonical_id(d.lang))
            .collect();
        assert!(found.contains(&"rust"));
        assert!(found.contains(&"go"));
        assert!(found.contains(&"dart"));
        assert!(found.contains(&"markdown"));
    }

    #[test]
    fn all_catalog_ids_round_trip() {
        for lang in catalog() {
            let id = canonical_id(*lang);
            assert_eq!(parse_id(id), Some(*lang), "id {id}");
        }
    }
}
