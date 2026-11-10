//! **Structurally derived** context for a retrieval chunk (E6/C).
//!
//! The community "contextual retrieval" technique prepends a document- or
//! section-level summary to each chunk before embedding. The summary is
//! normally written by a model per chunk; here it is derived from facts the
//! workspace already knows structurally:
//!
//! - the **path** of the file the chunk came from (`File: src/a.rs`),
//! - the **crate + module** the file belongs to, from the crate map's scopes
//!   plus Cargo's module conventions (`Module: operator::code_search`),
//! - the **enclosing items** the chunk sits inside, from a brace-aware scan of
//!   the file (`Symbols: impl LocalCodeIndex > fn build`).
//!
//! Everything in this module is pure: no I/O, no models, no clock. The
//! enrichment is a *field set* ([`ChunkEnrichment`]) rather than a fixed layout
//! so the ablation can price each field on its own, and
//! [`ChunkEnrichment::PATH`] is byte-identical to the literal
//! `format!("File: {path}\n\n…")` the two chunkers have always used.
//!
//! Rendering rule: the prefix is a block of one-line fields followed by one
//! blank line, so the embedded text is
//! `<fields>\n\n<code>`. A bare chunk (or a field set whose fields all came
//! back empty, e.g. `symbols` on a chunk outside every item) renders the empty
//! string — never a stray blank line.

use crate::types::{EnrichModule, EnrichPath, EnrichSymbols};
use crate_map::NO_CRATE;

/// At most this many enclosing items are rendered (innermost wins).
pub const MAX_SYMBOLS: usize = 3;

/// Longest rendered symbol entry, in characters.
pub const MAX_SYMBOL_CHARS: usize = 60;

/// One chunk-enrichment field set: which derived context lines the chunk's
/// stored text carries. `Copy`, so a variant is a value and never a mode.
///
/// [`Default`] is [`ChunkEnrichment::PATH`] — the shipped shape — never
/// `BARE`: a struct literal that forgets the field must keep behaving like
/// today, not silently drop the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkEnrichment {
    /// The `File: <path>` line — the incumbent, and the default.
    pub path: EnrichPath,
    /// The `Module: <crate>::<module>` line derived from the crate map.
    pub module: EnrichModule,
    /// The `Symbols: <item> > <item>` line derived from a brace-aware scan.
    pub symbols: EnrichSymbols,
}

impl Default for ChunkEnrichment {
    fn default() -> Self {
        Self::PATH
    }
}

impl ChunkEnrichment {
    /// No context at all: the raw chunk text (the ablation's floor).
    pub const BARE: Self = Self {
        path: EnrichPath::off(),
        module: EnrichModule::off(),
        symbols: EnrichSymbols::off(),
    };
    /// Today's shipped shape: the path line only.
    pub const PATH: Self = Self {
        path: EnrichPath::on(),
        module: EnrichModule::off(),
        symbols: EnrichSymbols::off(),
    };
    /// Path + module.
    pub const PATH_MODULE: Self = Self {
        path: EnrichPath::on(),
        module: EnrichModule::on(),
        symbols: EnrichSymbols::off(),
    };
    /// Path + enclosing symbols.
    pub const PATH_SYMBOLS: Self = Self {
        path: EnrichPath::on(),
        module: EnrichModule::off(),
        symbols: EnrichSymbols::on(),
    };
    /// Every structural field.
    pub const ALL: Self = Self {
        path: EnrichPath::on(),
        module: EnrichModule::on(),
        symbols: EnrichSymbols::on(),
    };

    /// Whether no field is set, so [`context_prefix`] renders nothing at all.
    pub fn is_bare(self) -> bool {
        !self.path.is_on() && !self.module.is_on() && !self.symbols.is_on()
    }

    /// How many fields are set — the "fewest fields" tie-break of the
    /// ablation report's recommendation (test-gated with that instrument).
    #[cfg(test)]
    pub fn field_count(self) -> usize {
        usize::from(self.path.is_on())
            + usize::from(self.module.is_on())
            + usize::from(self.symbols.is_on())
    }

    /// Canonical label, used by the CLI flag, the run key, the index identity
    /// and the ablation report — one vocabulary, so a variant can never be
    /// named one way in a record and another in the report.
    pub fn label(self) -> &'static str {
        match (self.path.is_on(), self.module.is_on(), self.symbols.is_on()) {
            (false, false, false) => "bare",
            (true, false, false) => "path",
            (true, true, false) => "path+module",
            (true, false, true) => "path+symbols",
            (true, true, true) => "all",
            // The remaining combinations are not ablation variants; they still
            // need a stable label so a hand-built Config can be keyed.
            (false, true, false) => "module",
            (false, false, true) => "symbols",
            (false, true, true) => "module+symbols",
        }
    }

    /// Parse a [`Self::label`], tolerating `_`/`-` for `+` and the explicit
    /// field spelling `path+module+symbols`. `None` for an unknown label, so
    /// the caller can reject it by name instead of silently defaulting.
    pub fn from_label(raw: &str) -> Option<Self> {
        let normalized = raw
            .trim()
            .to_ascii_lowercase()
            .replace(['_', '-'], "+")
            .replace(" +", "+")
            .replace("+ ", "+");
        match normalized.as_str() {
            // An empty value is "absent", never "bare": a missing config key
            // must fall back to the default, not silently disable the context.
            "" => None,
            "bare" | "none" => Some(Self::BARE),
            "path" => Some(Self::PATH),
            "path+module" => Some(Self::PATH_MODULE),
            "path+symbols" => Some(Self::PATH_SYMBOLS),
            "all" | "path+module+symbols" => Some(Self::ALL),
            "module" => Some(Self {
                path: EnrichPath::off(),
                module: EnrichModule::on(),
                symbols: EnrichSymbols::off(),
            }),
            "symbols" => Some(Self {
                path: EnrichPath::off(),
                module: EnrichModule::off(),
                symbols: EnrichSymbols::on(),
            }),
            "module+symbols" => Some(Self {
                path: EnrichPath::off(),
                module: EnrichModule::on(),
                symbols: EnrichSymbols::on(),
            }),
            _ => None,
        }
    }

    /// The index-identity fragment: `""` for the incumbent, so the shipped
    /// table and ingestion manifest keep the exact name they have always had
    /// (no re-embed for anyone who never asks for more context), and a
    /// distinct `_enrich-<label>` for every other field set, so two
    /// enrichments can never share one corpus — the manifest records file
    /// hashes, so a shared table would silently keep the wrong vectors.
    pub fn identity_fragment(self) -> String {
        if self == Self::PATH {
            String::new()
        } else {
            // The label is readable in records; the table name only needs an
            // unambiguous, bounded fragment (LanceDB caps table names at 128
            // characters, and the embedder fragment is model-dependent).
            let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
            sha2::Digest::update(&mut hasher, self.label().as_bytes());
            format!(
                "_e{}",
                &crate::hash::hex_digest(sha2::Digest::finalize(hasher))[..8]
            )
        }
    }
}

/// The five ablation variants, in report order: the incumbent second, so the
/// table reads floor → incumbent → marginals → everything.
pub const VARIANTS: [ChunkEnrichment; 5] = [
    ChunkEnrichment::BARE,
    ChunkEnrichment::PATH,
    ChunkEnrichment::PATH_MODULE,
    ChunkEnrichment::PATH_SYMBOLS,
    ChunkEnrichment::ALL,
];

impl argh::FromArgValue for ChunkEnrichment {
    fn from_arg_value(value: &str) -> Result<Self, String> {
        Self::from_label(value).ok_or_else(|| {
            format!(
                "unknown retrieval enrichment '{value}' (expected one of: {})",
                VARIANTS
                    .iter()
                    .map(|v| v.label())
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
        })
    }
}

/// The crate/module path of a workspace-relative file, **including** the crate
/// prefix, or the empty string when nothing can be said.
///
/// `crate_name`/`crate_dir` come from [`crate_map::scope_of`] (`crate_dir` is
/// `""` for a root package); `crate_name == NO_CRATE` means the file is outside
/// every crate, and the plain slash-normalized path is rendered instead.
///
/// Cargo's conventions, applied to the path *below the crate root*:
/// `src/lib.rs` and `src/main.rs` are the crate root itself (so the crate name
/// alone is the module path), `src/foo/mod.rs` is `foo`, `src/foo/bar.rs` is
/// `foo::bar`, and a `src/bin/<name>.rs` target is its own binary root, so it
/// renders as `<name>`. A `-` in a crate name becomes `_`, because that is how
/// the name appears in Rust source. Pure.
pub fn module_path_of(crate_name: &str, crate_dir: &str, rel_path: &str) -> String {
    let normalized = rel_path.trim().replace('\\', "/");
    let mut rel = normalized.trim_start_matches("./").to_string();
    if !crate_dir.is_empty()
        && let Some(rest) = rel.strip_prefix(crate_dir)
    {
        rel = rest.strip_prefix('/').unwrap_or(rest).to_string();
    }
    let stem_path = rel.strip_suffix(".rs").unwrap_or(&rel);
    let mut parts: Vec<&str> = stem_path.split('/').filter(|s| !s.is_empty()).collect();
    // A leading source root is container plumbing, not a module segment.
    if matches!(
        parts.first(),
        Some(&"src" | &"tests" | &"examples" | &"benches")
    ) {
        parts.remove(0);
    }
    let file = parts.pop().unwrap_or("");
    if crate_name == NO_CRATE {
        // Outside every crate: the slash-normalized path, no crate prefix, and
        // never a bare `::`.
        let mut segs: Vec<&str> = parts.clone();
        if !matches!(file, "" | "lib" | "main" | "mod") {
            segs.push(file);
        }
        return segs.join("::");
    }
    let root = crate_name.replace('-', "_");
    if parts.first() == Some(&"bin") {
        return if file.is_empty() {
            root
        } else {
            file.to_string()
        };
    }
    let mut segs: Vec<&str> = parts.clone();
    if !matches!(file, "" | "lib" | "main" | "mod") {
        segs.push(file);
    }
    if segs.is_empty() {
        root
    } else {
        format!("{root}::{}", segs.join("::"))
    }
}

/// An item that opened a brace scope while scanning a file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenItem {
    kind: &'static str,
    name: String,
    /// Brace depth *before* the line that opened the item.
    depth: usize,
}

/// The chain of items enclosing `line` (1-based), outermost-first, capped at
/// [`MAX_SYMBOLS`] entries and rendered `<kind> <identifier>`.
///
/// The scan is brace-aware over a comment/string-stripped view of each line, so
/// braces inside a string literal, a char literal or a comment cannot miscount
/// the nesting. `line` beyond EOF clamps to the last line; a file with
/// unbalanced braces returns the headers found so far; a line inside no item
/// returns an empty vector. Pure.
pub fn enclosing_symbols(text: &str, line: u32) -> Vec<String> {
    if line == 0 {
        return Vec::new();
    }
    let target = line as usize;
    let mut lexer = Lexer::default();
    let mut stack: Vec<OpenItem> = Vec::new();
    let mut depth: usize = 0;
    let mut seen: usize = 0;
    for raw in text.lines() {
        seen += 1;
        let code = lexer.code_only(raw);
        let opens = code.matches('{').count();
        let closes = code.matches('}').count();
        // The header is registered at the depth it opens *from*, before this
        // line's own braces move the counter. A header with no brace on its own
        // line is a declaration (`mod foo;`, a trait's `fn sig();`), not a scope.
        if opens > closes
            && let Some((kind, name)) = item_header(&code)
        {
            stack.push(OpenItem { kind, name, depth });
        }
        depth = depth.saturating_sub(closes) + opens;
        stack.retain(|item| item.depth < depth);
        if seen >= target {
            break;
        }
    }
    let start = stack.len().saturating_sub(MAX_SYMBOLS);
    stack[start..]
        .iter()
        .map(|item| {
            let mut rendered = format!("{} {}", item.kind, item.name);
            if rendered.chars().count() > MAX_SYMBOL_CHARS {
                rendered = rendered.chars().take(MAX_SYMBOL_CHARS).collect();
            }
            rendered
        })
        .collect()
}

/// Render one chunk's context prefix. Pure.
///
/// [`ChunkEnrichment::PATH`] renders exactly `File: <path>\n\n` (the literal the
/// chunkers used before this module existed); a bare field set renders `""`; a
/// field set whose only requested fields came back empty renders `""` too, so
/// the chunk text never gains an empty context block.
pub fn context_prefix(
    enrichment: ChunkEnrichment,
    path: &str,
    module_path: &str,
    symbols: &[String],
) -> String {
    if enrichment.is_bare() {
        return String::new();
    }
    let mut lines: Vec<String> = Vec::new();
    if enrichment.path.is_on() {
        lines.push(format!("File: {path}"));
    }
    if enrichment.module.is_on() && !module_path.trim().is_empty() {
        lines.push(format!("Module: {module_path}"));
    }
    if enrichment.symbols.is_on() && !symbols.is_empty() {
        lines.push(format!("Symbols: {}", symbols.join(" > ")));
    }
    if lines.is_empty() {
        return String::new();
    }
    format!("{}\n\n", lines.join("\n"))
}

/// Strip a leading keyword from `text`, requiring a word boundary after it (so
/// `fn_call` is not an `fn`). Pure.
fn strip_word<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(word)?;
    match rest.chars().next() {
        None => Some(rest),
        Some(c) if c.is_alphanumeric() || c == '_' => None,
        Some(_) => Some(rest),
    }
}

/// Skip a `pub(...)` visibility group (the leading `pub` is already consumed).
fn strip_pub_group(text: &str) -> &str {
    let trimmed = text.trim_start();
    if let Some(inner) = trimmed.strip_prefix('(')
        && let Some(end) = inner.find(')')
    {
        return &inner[end + 1..];
    }
    trimmed
}

/// Skip an `extern "abi"` string after the `extern` keyword.
fn skip_abi(text: &str) -> &str {
    let trimmed = text.trim_start();
    if let Some(inner) = trimmed.strip_prefix('"')
        && let Some(end) = inner.find('"')
    {
        return &inner[end + 1..];
    }
    trimmed
}

/// The first identifier in `text`, or `None` when there is none. Pure.
fn first_ident(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    let name: String = trimmed
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

/// Drop a leading balanced `<...>` generic parameter list. Pure.
fn skip_generics(text: &str) -> &str {
    let text = text.trim_start();
    let Some(inner) = text.strip_prefix('<') else {
        return text;
    };
    let mut depth = 1usize;
    for (index, c) in inner.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return &inner[index + 1..];
                }
            }
            _ => {}
        }
    }
    text
}

/// The rendered name of an `impl` header: the implemented type, not the trait
/// (`impl Display for Foo` → `Foo`). Pure.
fn impl_name(text: &str) -> Option<String> {
    let text = skip_generics(text);
    let text = match text.find(" for ") {
        Some(index) => &text[index + 5..],
        None => text,
    };
    let text = skip_generics(text);
    first_ident(text)
}

/// The item header one code-only line declares, if any. Pure.
fn item_header(code: &str) -> Option<(&'static str, String)> {
    let mut rest = code.trim_start();
    loop {
        if let Some(r) = strip_word(rest, "pub") {
            rest = strip_pub_group(r);
            continue;
        }
        if let Some(r) = strip_word(rest, "unsafe").or_else(|| strip_word(rest, "async")) {
            rest = r;
            continue;
        }
        if let Some(r) = strip_word(rest, "default") {
            rest = r;
            continue;
        }
        if let Some(r) = strip_word(rest, "extern") {
            rest = skip_abi(r);
            continue;
        }
        break;
    }
    for (keyword, kind) in [
        ("fn", "fn"),
        ("struct", "struct"),
        ("enum", "enum"),
        ("trait", "trait"),
        ("impl", "impl"),
        ("mod", "mod"),
        ("const", "const"),
        ("static", "static"),
        ("type", "type"),
        ("union", "union"),
    ] {
        let Some(after) = strip_word(rest, keyword) else {
            continue;
        };
        if kind == "const" || kind == "static" {
            // `const fn foo` / `const unsafe fn foo` declare a function.
            let mut inner = after.trim_start();
            loop {
                if let Some(r) = strip_word(inner, "unsafe").or_else(|| strip_word(inner, "async"))
                {
                    inner = r;
                    continue;
                }
                break;
            }
            if let Some(f) = strip_word(inner, "fn") {
                return first_ident(f).map(|name| ("fn", name));
            }
        }
        let name = if kind == "impl" {
            impl_name(after)
        } else {
            first_ident(after)
        };
        return name.map(|name| (kind, name));
    }
    None
}

/// Comment/string stripping, carried across lines for block comments.
#[derive(Default)]
struct Lexer {
    in_block_comment: bool,
}

impl Lexer {
    /// `line` with string, char and comment contents removed, so braces that
    /// are not code cannot move the depth counter.
    fn code_only(&mut self, line: &str) -> String {
        let chars: Vec<char> = line.chars().collect();
        let mut out = String::new();
        let mut index = 0usize;
        while index < chars.len() {
            let c = chars[index];
            if self.in_block_comment {
                if c == '*' && chars.get(index + 1) == Some(&'/') {
                    self.in_block_comment = false;
                    index += 2;
                    continue;
                }
                index += 1;
                continue;
            }
            if c == '/' && chars.get(index + 1) == Some(&'/') {
                break;
            }
            if c == '/' && chars.get(index + 1) == Some(&'*') {
                self.in_block_comment = true;
                index += 2;
                continue;
            }
            if (c == 'r' || c == 'b')
                && let Some(end) = raw_string_end(&chars, index)
            {
                index = end;
                continue;
            }
            if c == '"' {
                index += 1;
                while index < chars.len() {
                    if chars[index] == '\\' {
                        index += 2;
                        continue;
                    }
                    if chars[index] == '"' {
                        index += 1;
                        break;
                    }
                    index += 1;
                }
                continue;
            }
            if c == '\'' {
                if chars.get(index + 1) == Some(&'\\') {
                    let mut scan = index + 2;
                    while scan < chars.len() && chars[scan] != '\'' {
                        scan += 1;
                    }
                    if scan < chars.len() {
                        index = scan + 1;
                        continue;
                    }
                } else if chars.get(index + 2) == Some(&'\'') {
                    index += 3;
                    continue;
                }
                // A lifetime (`'a`), not a char literal.
            }
            out.push(c);
            index += 1;
        }
        out
    }
}

/// The index just past a raw string starting at `start` (`r"…"`, `r#"…"#`,
/// `br#"…"#`), or `None` when no raw string starts there. Pure.
fn raw_string_end(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start;
    if chars.get(index) == Some(&'b') {
        index += 1;
    }
    if chars.get(index) != Some(&'r') {
        return None;
    }
    index += 1;
    let mut hashes = 0usize;
    while chars.get(index) == Some(&'#') {
        hashes += 1;
        index += 1;
    }
    if chars.get(index) != Some(&'"') {
        return None;
    }
    index += 1;
    while index < chars.len() {
        if chars[index] == '"' {
            let mut scan = index + 1;
            let mut found = 0usize;
            while found < hashes && chars.get(scan) == Some(&'#') {
                found += 1;
                scan += 1;
            }
            if found == hashes {
                return Some(scan);
            }
        }
        index += 1;
    }
    // An unterminated raw string swallows the rest of the line.
    Some(chars.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_renders_nothing_and_path_is_the_literal_the_chunkers_used() {
        assert_eq!(
            context_prefix(ChunkEnrichment::BARE, "src/a.rs", "crate::a", &[]),
            ""
        );
        for path in ["src/a.rs", "crates/x/src/lib.rs", ""] {
            assert_eq!(
                context_prefix(ChunkEnrichment::PATH, path, "", &[]),
                format!("File: {path}\n\n"),
                "the incumbent prefix must stay byte-identical"
            );
        }
    }

    #[test]
    fn each_field_set_renders_exactly_its_own_lines() {
        let symbols = vec!["impl Index".to_string(), "fn build".to_string()];
        assert_eq!(
            context_prefix(
                ChunkEnrichment::PATH_MODULE,
                "src/a.rs",
                "crate_a::a",
                &symbols
            ),
            "File: src/a.rs\nModule: crate_a::a\n\n"
        );
        assert_eq!(
            context_prefix(
                ChunkEnrichment::PATH_SYMBOLS,
                "src/a.rs",
                "crate_a::a",
                &symbols
            ),
            "File: src/a.rs\nSymbols: impl Index > fn build\n\n"
        );
        assert_eq!(
            context_prefix(ChunkEnrichment::ALL, "src/a.rs", "crate_a::a", &symbols),
            "File: src/a.rs\nModule: crate_a::a\nSymbols: impl Index > fn build\n\n"
        );
    }

    #[test]
    fn an_empty_field_never_renders_a_stray_line_or_block() {
        // symbols requested but nothing encloses the chunk: nothing to say.
        assert_eq!(
            context_prefix(ChunkEnrichment::PATH_SYMBOLS, "src/a.rs", "", &[]),
            "File: src/a.rs\n\n"
        );
        // module requested but the module path is empty and the path line is
        // off: never a bare `Module:` (nor a leading blank line).
        assert_eq!(
            context_prefix(
                ChunkEnrichment {
                    path: EnrichPath::off(),
                    module: EnrichModule::on(),
                    symbols: EnrichSymbols::off(),
                },
                "src/a.rs",
                "",
                &[]
            ),
            ""
        );
        // no path, no module, but symbols do exist.
        assert_eq!(
            context_prefix(
                ChunkEnrichment {
                    path: EnrichPath::off(),
                    module: EnrichModule::off(),
                    symbols: EnrichSymbols::on(),
                },
                "src/a.rs",
                "",
                &["fn only".to_string()]
            ),
            "Symbols: fn only\n\n"
        );
    }

    #[test]
    fn labels_round_trip_and_unknown_labels_are_rejected_by_name() {
        for variant in VARIANTS {
            assert_eq!(
                ChunkEnrichment::from_label(variant.label()),
                Some(variant),
                "every variant's own label must parse back"
            );
        }
        assert_eq!(
            ChunkEnrichment::from_label("Path+Module"),
            Some(ChunkEnrichment::PATH_MODULE)
        );
        assert_eq!(
            ChunkEnrichment::from_label("path_module"),
            Some(ChunkEnrichment::PATH_MODULE)
        );
        assert_eq!(
            ChunkEnrichment::from_label("path-module"),
            Some(ChunkEnrichment::PATH_MODULE)
        );
        assert_eq!(
            ChunkEnrichment::from_label("path+module+symbols"),
            Some(ChunkEnrichment::ALL)
        );
        assert_eq!(ChunkEnrichment::from_label("nonsense"), None);
        assert_eq!(
            ChunkEnrichment::from_label(""),
            None,
            "an empty value is absent, never bare"
        );
        let err = <ChunkEnrichment as argh::FromArgValue>::from_arg_value("nope").unwrap_err();
        assert!(err.contains("unknown retrieval enrichment"), "{err}");
        assert!(
            err.contains("path+symbols"),
            "the error names the choices: {err}"
        );
    }

    #[test]
    fn only_the_incumbent_has_an_empty_identity_fragment() {
        assert_eq!(ChunkEnrichment::PATH.identity_fragment(), "");
        for variant in VARIANTS.iter().filter(|v| **v != ChunkEnrichment::PATH) {
            let fragment = variant.identity_fragment();
            assert!(fragment.starts_with("_e"), "{fragment}");
            assert_eq!(fragment.len(), 10, "bounded fragment: {fragment}");
            assert!(
                !fragment.contains('+'),
                "no `+` in a table name: {fragment}"
            );
        }
        // Every variant is distinguishable, and no fragment collides.
        let mut seen: Vec<String> = VARIANTS.iter().map(|v| v.identity_fragment()).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), VARIANTS.len());
    }

    #[test]
    fn field_count_drives_the_minimal_set_tie_break() {
        assert_eq!(ChunkEnrichment::BARE.field_count(), 0);
        assert_eq!(ChunkEnrichment::PATH.field_count(), 1);
        assert_eq!(ChunkEnrichment::ALL.field_count(), 3);
        assert!(ChunkEnrichment::BARE.is_bare());
        assert!(!ChunkEnrichment::PATH.is_bare());
    }

    #[test]
    fn module_paths_follow_cargo_conventions() {
        // A root package: crate_dir is empty, the file path is crate-relative.
        let root = "operator";
        assert_eq!(module_path_of(root, "", "src/lib.rs"), "operator");
        assert_eq!(module_path_of(root, "", "src/main.rs"), "operator");
        assert_eq!(
            module_path_of(root, "", "src/code_search.rs"),
            "operator::code_search"
        );
        assert_eq!(module_path_of(root, "", "src/foo/mod.rs"), "operator::foo");
        assert_eq!(
            module_path_of(root, "", "src/foo/bar.rs"),
            "operator::foo::bar"
        );
        // A `src/bin` target is its own root, not a module of the package.
        assert_eq!(module_path_of(root, "", "src/bin/tool.rs"), "tool");
        // A path dependency: the module path is relative to the crate root.
        assert_eq!(
            module_path_of(
                "crate-map",
                "crates/crate-map",
                "crates/crate-map/src/lib.rs"
            ),
            "crate_map"
        );
        assert_eq!(
            module_path_of(
                "crate-map",
                "crates/crate-map",
                "crates/crate-map/src/scan.rs"
            ),
            "crate_map::scan"
        );
        // Outside every crate: the slash-normalized path, no crate prefix.
        assert_eq!(module_path_of(NO_CRATE, "", "docs/notes.rs"), "docs::notes");
        assert_eq!(
            module_path_of(NO_CRATE, "", "src/deep/thing.rs"),
            "deep::thing"
        );
        assert_eq!(module_path_of(NO_CRATE, "", "build.rs"), "build");
        // A `./` prefix and Windows separators normalize first.
        assert_eq!(
            module_path_of(root, "", "./src\\code_search.rs"),
            "operator::code_search"
        );
    }

    #[test]
    fn symbol_chain_is_outermost_first_and_brace_aware() {
        let text = "\
use std::fmt;\n\
\n\
// a comment with a { brace\n\
mod outer {\n\
    pub struct Index {\n\
        field: String,\n\
    }\n\
\n\
    impl Index {\n\
        /// doc with a } brace\n\
        pub fn build(&self) -> String {\n\
            let s = \"a } brace in a string\";\n\
            if s.len() > 1 {\n\
                return s;\n\
            }\n\
            String::new()\n\
        }\n\
    }\n\
}\n";
        // Line 9 (`impl Index {`) is inside `mod outer`.
        assert_eq!(
            enclosing_symbols(text, 9),
            vec!["mod outer".to_string(), "impl Index".to_string()]
        );
        // Inside `fn build`, past a string literal holding a `}`.
        assert_eq!(
            enclosing_symbols(text, 11),
            vec![
                "mod outer".to_string(),
                "impl Index".to_string(),
                "fn build".to_string()
            ]
        );
        // A brace inside a line comment or a string cannot close a scope: line
        // 16 is still inside `fn build`.
        assert_eq!(enclosing_symbols(text, 16).len(), 3);
        // Line 3 is inside `mod outer` only.
        assert_eq!(
            enclosing_symbols(text, 5),
            vec!["mod outer".to_string(), "struct Index".to_string()]
        );
        // Line 1 encloses nothing.
        assert!(enclosing_symbols(text, 1).is_empty());
        // Beyond EOF clamps to the last line rather than panicking.
        assert!(enclosing_symbols(text, 10_000).is_empty());
    }

    #[test]
    fn declarations_without_a_body_are_not_scopes() {
        let text = "\
trait Greet {\n\
    fn hello(&self);\n\
    fn bye(&self) {\n\
        println!(\"bye\");\n\
    }\n\
}\n\
mod decl;\n\
struct Unit;\n\
type Alias = u8;\n\
fn free() {\n\
    let x = 1;\n\
}\n";
        // A trait method with no body is not entered.
        assert_eq!(enclosing_symbols(text, 2), vec!["trait Greet".to_string()]);
        // A default method *is* entered.
        assert_eq!(
            enclosing_symbols(text, 4),
            vec!["trait Greet".to_string(), "fn bye".to_string()]
        );
        // `mod decl;`, `struct Unit;` and `type Alias = u8;` open nothing, so
        // line 10 is inside `fn free` only.
        assert_eq!(enclosing_symbols(text, 11), vec!["fn free".to_string()]);
        assert_eq!(enclosing_symbols(text, 9), Vec::<String>::new());
    }

    #[test]
    fn impl_headers_name_the_type_not_the_trait() {
        assert_eq!(
            enclosing_symbols(
                "impl std::fmt::Display for Foo {\n    fn f() {\n        let a = 1;\n    }\n}\n",
                2
            ),
            vec!["impl Foo".to_string(), "fn f".to_string()]
        );
        assert_eq!(
            enclosing_symbols(
                "impl<T> Foo<T> {\n    fn f() {\n        let a = 1;\n    }\n}\n",
                2
            ),
            vec!["impl Foo".to_string(), "fn f".to_string()]
        );
        assert_eq!(
            enclosing_symbols(
                "impl Foo where T: Send {\n    fn f() {\n        let a = 1;\n    }\n}\n",
                2
            ),
            vec!["impl Foo".to_string(), "fn f".to_string()]
        );
    }

    #[test]
    fn const_fn_is_a_function_and_fn_call_is_not_a_header() {
        let text = "\
pub const fn limit() -> usize {\n\
    fn_call();\n\
    fnested();\n\
}\n";
        assert_eq!(enclosing_symbols(text, 2), vec!["fn limit".to_string()]);
    }

    #[test]
    fn symbols_are_capped_in_count_and_length() {
        let text = "\
mod a {\n\
mod b {\n\
mod c {\n\
fn d() {\n\
let a_very_long_identifier_that_exceeds_the_cap_by_a_lot_indeed = 1;\n\
}\n\
}\n\
}\n\
}\n";
        let chain = enclosing_symbols(text, 5);
        assert_eq!(chain.len(), MAX_SYMBOLS, "innermost three only: {chain:?}");
        assert_eq!(chain[0], "mod b", "outermost of the kept three comes first");
        assert_eq!(chain[2], "fn d");
        let long = format!("fn {}", "x".repeat(200));
        let chain = enclosing_symbols(&format!("{long} {{\n    let a = 1;\n}}\n"), 2);
        assert_eq!(chain[0].chars().count(), MAX_SYMBOL_CHARS);
    }

    #[test]
    fn unbalanced_braces_return_what_was_found_and_raw_strings_are_skipped() {
        let text = "fn outer() {\n    let s = r#\"a } brace\"#;\n";
        assert_eq!(enclosing_symbols(text, 3), vec!["fn outer".to_string()]);
        // Unbalanced: the `}` never arrives, so the stack stays.
        assert_eq!(
            enclosing_symbols("mod x {\nfn y() {\n", 2),
            vec!["mod x".to_string(), "fn y".to_string()]
        );
    }
}
