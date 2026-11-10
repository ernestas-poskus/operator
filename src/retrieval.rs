//! Code-aware FTS tokenization shared by the index and query sides.

fn split_identifier_run(run: &str) -> Vec<String> {
    let chars: Vec<char> = run.chars().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let lower_to_upper = cur.is_uppercase() && (prev.is_lowercase() || prev.is_numeric());
        let acronym_then_word = i + 1 < chars.len()
            && prev.is_uppercase()
            && cur.is_uppercase()
            && chars[i + 1].is_lowercase();
        if lower_to_upper || acronym_then_word {
            out.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    out.push(chars[start..].iter().collect());
    out
}

pub fn expand_identifiers(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for run in text.split(|c: char| !c.is_alphanumeric()) {
        for token in split_identifier_run(run) {
            if token.len() >= 2 {
                out.push(token.to_ascii_lowercase());
            }
        }
    }
    out.join(" ")
}

pub fn fts_query_text(query: &str) -> String {
    let expanded = expand_identifiers(query);
    if expanded.is_empty() {
        query.to_string()
    } else {
        expanded
    }
}

pub fn fts_index_text(content: &str, path: &str, crate_name: &str) -> String {
    format!(
        "{} {} {}",
        expand_identifiers(content),
        expand_identifiers(path),
        expand_identifiers(crate_name)
    )
}
