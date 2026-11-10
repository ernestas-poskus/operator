//! Pure Rust item segmentation for the symbol chunker (R8).
//!
//! This module owns **source structure only**: one file of Rust in, one ordered
//! list of top-level item spans out. It knows nothing about tokens, chunk sizes,
//! contexts, file paths or the embedder — those live in crate::code_search.
//! The split keeps the segmenter unit-testable without a model server, exactly
//! like the sliding-window and xberg chunkers it sits beside.
//!
//! An **item span** is a contiguous run of source lines that is either one whole
//! top-level Rust item (with its attached /// docs and #[...] attributes) or a
//! preamble/leftover region between items. Nested items (a fn inside a body)
//! are part of their enclosing item: only a line at brace depth 0 can start one.

use crate::types::{LineNumber, SourceText};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemSpan {
    pub start_line: LineNumber,
    pub text: SourceText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadKind {
    SemicolonOnly,
    BraceOrSemicolon,
}

#[derive(Debug, Clone, Copy, Default)]
struct LineScan {
    depth_at_start: i32,
    depth_at_end: i32,
    semicolon: bool,
    open_brace: bool,
}

pub fn rust_item_spans(text: &str) -> Vec<ItemSpan> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let scan = scan_lines(text, lines.len());
    let attr = attr_lines(&lines);
    let mut heads: Vec<(usize, HeadKind)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if scan.get(i).map_or(0, |info| info.depth_at_start) != 0 {
            continue;
        }
        if let Some(kind) = item_kind(line) {
            heads.push((i, kind));
        }
    }
    if heads.is_empty() {
        return Vec::new();
    }
    let mut spans: Vec<ItemSpan> = Vec::new();
    let mut cursor = 0usize;
    for (head, kind) in heads {
        let end = end_of_item(head, kind, &scan).min(lines.len() - 1);
        let mut start = head;
        while start > 0 && start > cursor {
            let prev = start - 1;
            let t = lines[prev].trim_start();
            if t.starts_with("///") || attr[prev] {
                start = prev;
            } else {
                break;
            }
        }
        if start < cursor {
            start = cursor;
        }
        if start > end {
            continue;
        }
        if start > cursor {
            push_span(&mut spans, &lines, cursor, start - 1);
        }
        push_span(&mut spans, &lines, start, end);
        cursor = end + 1;
    }
    if cursor < lines.len() {
        push_span(&mut spans, &lines, cursor, lines.len() - 1);
    }
    spans
}

/// Push the span [a, b] when it holds any non-whitespace content.
fn push_span(spans: &mut Vec<ItemSpan>, lines: &[&str], a: usize, b: usize) {
    if a > b {
        return;
    }
    let slice = &lines[a..=b];
    if slice.iter().all(|l| l.trim().is_empty()) {
        return;
    }
    spans.push(ItemSpan {
        start_line: LineNumber::new((a + 1) as u32),
        text: SourceText::new(slice.join("\n")),
    });
}

fn end_of_item(head: usize, kind: HeadKind, scan: &[LineScan]) -> usize {
    let mut open = None;
    let mut semi = None;
    for (i, info) in scan.iter().enumerate().skip(head) {
        if open.is_none() && info.open_brace {
            open = Some(i);
        }
        if semi.is_none() && info.semicolon {
            semi = Some(i);
        }
        if kind == HeadKind::SemicolonOnly && semi.is_some() {
            break;
        }
        if open.is_some() && semi.is_some() {
            break;
        }
    }
    match kind {
        HeadKind::SemicolonOnly => semi.unwrap_or(scan.len().saturating_sub(1)),
        HeadKind::BraceOrSemicolon => match (open, semi) {
            (Some(o), Some(s)) if s < o => s,
            (Some(o), _) => close_of(o, scan),
            (None, Some(s)) => s,
            (None, None) => scan.len().saturating_sub(1),
        },
    }
}

fn close_of(open: usize, scan: &[LineScan]) -> usize {
    for (i, info) in scan.iter().enumerate().skip(open) {
        if info.depth_at_end == 0 {
            return i;
        }
    }
    scan.len().saturating_sub(1)
}

fn scan_lines(text: &str, line_count: usize) -> Vec<LineScan> {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(line_count);
    let mut depth = 0i32;
    let mut state = State::Normal;
    let mut cur = LineScan::default();
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'\n' {
            // A line comment and a (malformed) single-line string end at the
            // newline; a block comment or raw string may legitimately span it.
            if state == State::LineComment || state == State::Str {
                state = State::Normal;
            }
            cur.depth_at_end = depth;
            out.push(cur);
            cur = LineScan {
                depth_at_start: depth,
                ..Default::default()
            };
            i += 1;
            continue;
        }
        match state {
            State::LineComment => {}
            State::BlockComment(n) => {
                if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    if n <= 1 {
                        state = State::Normal;
                    } else {
                        state = State::BlockComment(n - 1);
                    }
                    i += 2;
                    continue;
                }
                if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                    state = State::BlockComment(n + 1);
                    i += 2;
                    continue;
                }
            }
            State::Str => {
                if b[i] == b'\\' {
                    // A backslash-newline is a string line continuation: count
                    // the line, stay in the string, skip both bytes.
                    if i + 1 < b.len() && b[i + 1] == b'\n' {
                        cur.depth_at_end = depth;
                        out.push(cur);
                        cur = LineScan {
                            depth_at_start: depth,
                            ..Default::default()
                        };
                    }
                    i += 2;
                    continue;
                }
                if b[i] == b'"' {
                    state = State::Normal;
                }
            }
            State::RawStr(hashes) => {
                if b[i] == b'"'
                    && i + 1 + hashes <= b.len()
                    && b[i + 1..i + 1 + hashes].iter().all(|c| *c == b'#')
                {
                    i += 1 + hashes + 1;
                    state = State::Normal;
                    continue;
                }
            }
            State::Normal => {
                if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                    state = State::LineComment;
                    i += 2;
                    continue;
                }
                if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                    state = State::BlockComment(1);
                    i += 2;
                    continue;
                }
                if let Some((consumed, hashes)) = raw_string_start(b, i) {
                    state = State::RawStr(hashes);
                    i += consumed;
                    continue;
                }
                if b[i] == b'"' {
                    state = State::Str;
                    i += 1;
                    continue;
                }
                if b[i] == b'b' && i + 1 < b.len() && b[i + 1] == b'"' {
                    state = State::Str;
                    i += 2;
                    continue;
                }
                if b[i] == b'\''
                    && let Some(end) = char_literal_end(b, i)
                {
                    i = end;
                    continue;
                }
                match b[i] {
                    b'{' => {
                        if depth == 0 {
                            cur.open_brace = true;
                        }
                        depth += 1;
                    }
                    b'}' => {
                        depth -= 1;
                        if depth < 0 {
                            depth = 0;
                        }
                    }
                    b';' if depth == 0 => {
                        cur.semicolon = true;
                    }
                    _ => {}
                }
            }
        }
        i += 1;
    }
    // Only a final line without a trailing newline is still pending; a text
    // ending in a newline already pushed every line.
    if !text.is_empty() && !text.ends_with('\n') {
        cur.depth_at_end = depth;
        out.push(cur);
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Normal,
    LineComment,
    BlockComment(usize),
    Str,
    RawStr(usize),
}

fn raw_string_start(b: &[u8], i: usize) -> Option<(usize, usize)> {
    let rest = &b[i..];
    let after_prefix = if rest.starts_with(b"br") || rest.starts_with(b"rb") {
        2
    } else if b[i] == b'r' {
        1
    } else {
        return None;
    };
    let mut j = i + after_prefix;
    let mut hashes = 0usize;
    while j < b.len() && b[j] == b'#' {
        hashes += 1;
        j += 1;
    }
    if j < b.len() && b[j] == b'"' {
        Some((j + 1 - i, hashes))
    } else {
        None
    }
}

fn char_literal_end(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    if j >= b.len() {
        return None;
    }
    if b[j] == b'\\' {
        j += 1;
        while j < b.len() {
            if b[j] == b'\'' {
                return Some(j + 1);
            }
            if b[j] == b'\n' {
                return None;
            }
            j += 1;
        }
        return None;
    }
    j += 1;
    while j < b.len() && (b[j] & 0xC0) == 0x80 {
        j += 1;
    }
    if j < b.len() && b[j] == b'\'' {
        Some(j + 1)
    } else {
        None
    }
}

fn item_kind(line: &str) -> Option<HeadKind> {
    let s = line.trim_start();
    if s.is_empty() || s.starts_with('#') || s.starts_with("//") {
        return None;
    }
    let rest = strip_modifiers(s);
    let t = rest.trim_start();
    if t.starts_with('{') {
        return Some(HeadKind::BraceOrSemicolon);
    }
    let token: String = t
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '!')
        .collect();
    match token.as_str() {
        "use" | "type" | "static" | "crate" => Some(HeadKind::SemicolonOnly),
        "const" => {
            let after = t.strip_prefix("const").unwrap_or("").trim_start();
            if after.starts_with("fn") {
                Some(HeadKind::BraceOrSemicolon)
            } else {
                Some(HeadKind::SemicolonOnly)
            }
        }
        "fn" | "struct" | "enum" | "union" | "trait" | "impl" | "mod" | "macro_rules!"
        | "macro" => Some(HeadKind::BraceOrSemicolon),
        _ => None,
    }
}

fn strip_modifiers(mut s: &str) -> &str {
    loop {
        let t = s.trim_start();
        if t.is_empty() {
            return t;
        }
        if let Some(rest) = t.strip_prefix("pub") {
            match rest.chars().next() {
                Some(c) if c.is_alphanumeric() || c == '_' => return t,
                Some('(') => match matching_paren(rest) {
                    Some(close) => {
                        s = &rest[close + 1..];
                        continue;
                    }
                    None => return t,
                },
                _ => {
                    s = rest;
                    continue;
                }
            }
        }
        if let Some(rest) = strip_word(t, "unsafe") {
            s = rest;
            continue;
        }
        if let Some(rest) = strip_word(t, "async") {
            s = rest;
            continue;
        }
        if let Some(rest) = strip_word(t, "default") {
            s = rest;
            continue;
        }
        if let Some(rest) = strip_word(t, "auto") {
            s = rest;
            continue;
        }
        if let Some(rest) = strip_word(t, "extern") {
            let r = rest.trim_start();
            if r.starts_with('"')
                && let Some(end) = r[1..].find('"')
            {
                s = &r[end + 2..];
            } else {
                s = r;
            }
            continue;
        }
        return t;
    }
}

fn matching_paren(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_word<'a>(t: &'a str, word: &str) -> Option<&'a str> {
    let rest = t.strip_prefix(word)?;
    match rest.chars().next() {
        None => Some(rest),
        Some(c) if c.is_whitespace() || c == '(' || c == '"' => Some(rest.trim_start()),
        _ => None,
    }
}

fn attr_lines(lines: &[&str]) -> Vec<bool> {
    let mut out = vec![false; lines.len()];
    let mut inside = false;
    let mut bracket = 0i32;
    for (i, line) in lines.iter().enumerate() {
        if inside {
            out[i] = true;
        }
        let t = line.trim_start();
        if !inside && t.starts_with("#[") {
            out[i] = true;
            inside = true;
            bracket = 0;
        }
        if inside {
            for c in line.chars() {
                match c {
                    '[' => bracket += 1,
                    ']' => bracket -= 1,
                    _ => {}
                }
            }
            if bracket <= 0 {
                inside = false;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans_of(text: &str) -> Vec<(u32, String)> {
        rust_item_spans(text)
            .into_iter()
            .map(|s| (s.start_line.get(), s.text.into_inner()))
            .collect()
    }

    #[test]
    fn a_simple_function_is_one_span_with_its_attached_docs() {
        let text = "\
/// Adds one.
#[inline]
pub fn add(a: i32) -> i32 {
    a + 1
}
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].0, 1);
        assert_eq!(spans[0].1, text.trim_end_matches('\n'));
    }

    #[test]
    fn a_brace_inside_a_string_does_not_end_the_item() {
        let text = "\
fn tricky() {
    let s = \"}\";
    let r = r#\"{ }\"#;
    let c = '}';
    /* } */
}
fn after() {}
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 2, "{spans:?}");
        assert_eq!(spans[0].1.lines().last(), Some("}"));
        assert!(spans[0].1.contains("\"}\""), "{}", spans[0].1);
        assert_eq!(spans[1].0, 7, "the second item starts after the first ends");
        assert_eq!(spans[1].1, "fn after() {}");
    }

    #[test]
    fn a_nested_fn_is_part_of_its_enclosing_item() {
        let text = "\
fn outer() {
    fn inner() {
    }
}
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert!(spans[0].1.contains("fn inner()"));
    }

    #[test]
    fn semicolon_terminated_items_end_at_the_semicolon() {
        let text = "\
use std::collections::HashMap;
pub(crate) type Alias = HashMap<String, Vec<u8>>;
const N: usize = 3;
static NAME: &str = \"x\";
struct Unit;
mod decl;
";
        let spans = spans_of(text);
        let starts: Vec<u32> = spans.iter().map(|(l, _)| *l).collect();
        assert_eq!(starts, vec![1, 2, 3, 4, 5, 6], "{spans:?}");
        assert_eq!(spans[0].1, "use std::collections::HashMap;");
        assert_eq!(spans[4].1, "struct Unit;");
        assert_eq!(spans[5].1, "mod decl;");
    }

    #[test]
    fn a_use_group_terminates_at_its_semicolon_not_its_brace() {
        let text = "\
use std::fmt::{self, Display};
fn next() {}
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 2, "{spans:?}");
        assert_eq!(spans[0].1, "use std::fmt::{self, Display};");
        assert_eq!(spans[1].1, "fn next() {}");
    }

    #[test]
    fn an_impl_block_swallows_its_methods() {
        let text = "\
impl Foo {
    pub fn a(&self) {}
    fn b(&self) {}
}

struct After;
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 2, "{spans:?}");
        assert!(spans[0].1.contains("fn b(&self)"));
        assert_eq!(spans[1].1, "struct After;");
    }

    #[test]
    fn a_file_with_no_recognized_head_is_empty() {
        assert!(rust_item_spans("").is_empty());
        assert!(rust_item_spans("   \n\n  ").is_empty());
        assert!(rust_item_spans("# just markdown\n\n- a list\n").is_empty());
        assert!(rust_item_spans("1 + 2\n3 + 4\n").is_empty());
    }

    #[test]
    fn an_unterminated_item_runs_to_the_last_line() {
        let text = "fn open() {\n    let x = 1;\n";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert_eq!(spans[0].1.lines().count(), 2, "{}", spans[0].1);
        assert!(spans[0].1.ends_with("let x = 1;"));
    }

    #[test]
    fn the_file_preamble_becomes_its_own_span() {
        let text = "\
//! Module docs.
#![allow(dead_code)]

use std::fmt;

fn f() {}
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 3, "{spans:?}");
        assert_eq!(spans[0].0, 1);
        assert!(spans[0].1.contains("//! Module docs."));
        assert!(spans[0].1.contains("#![allow(dead_code)]"));
        assert_eq!(spans[1].1, "use std::fmt;");
        assert_eq!(spans[2].1, "fn f() {}");
    }

    #[test]
    fn multi_line_attributes_attach_to_the_following_item() {
        let text = "\
#[derive(
    Debug,
    Clone,
)]
pub struct S {
    x: i32,
}
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert!(spans[0].1.starts_with("#[derive("));
        assert!(spans[0].1.contains("pub struct S"));
    }

    #[test]
    fn const_fn_is_a_braced_item() {
        let text = "\
const fn make() -> u8 {
    1
}
const LIMIT: u8 = make();
";
        let spans = spans_of(text);
        assert_eq!(spans.len(), 2, "{spans:?}");
        assert!(spans[0].1.contains("const fn make()"));
        assert_eq!(spans[1].1, "const LIMIT: u8 = make();");
    }

    #[test]
    fn spans_are_ordered_contiguous_and_cover_the_content() {
        let text = "\
use a::b;
fn f() {
    let s = \"x\";
}
struct S;
";
        let spans = rust_item_spans(text);
        for pair in spans.windows(2) {
            assert!(pair[0].start_line < pair[1].start_line);
        }
        let mut seen: Vec<&str> = Vec::new();
        for span in &spans {
            for line in span.text.lines() {
                seen.push(line);
            }
        }
        let source: Vec<&str> = text.lines().collect();
        assert_eq!(seen, source, "{spans:?}");
    }

    #[test]
    fn every_real_source_file_is_covered_by_its_spans() {
        // A smoke test over the crate's own sources: the segmenter must never
        // drop a non-blank line on real Rust, whatever the construct.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut with_spans = 0usize;
        let mut total = 0usize;
        for entry in std::fs::read_dir(&root).expect("src dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read source");
            if text.trim().is_empty() {
                continue;
            }
            total += 1;
            let spans = rust_item_spans(&text);
            if spans.is_empty() {
                continue;
            }
            with_spans += 1;
            for pair in spans.windows(2) {
                assert!(
                    pair[0].start_line < pair[1].start_line,
                    "{}: spans out of order",
                    path.display()
                );
            }
            let seen: usize = spans
                .iter()
                .flat_map(|s| s.text.lines())
                .filter(|l| !l.trim().is_empty())
                .count();
            let expected = text.lines().filter(|l| !l.trim().is_empty()).count();
            assert_eq!(
                seen,
                expected,
                "{}: {} non-blank lines not covered",
                path.display(),
                expected.saturating_sub(seen)
            );
        }
        assert!(
            total >= 20 && with_spans * 10 >= total * 9,
            "expected most of {total} sources to segment, got {with_spans}"
        );
    }
}
