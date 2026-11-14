//! Document-router benchmark driver used by `bench/doc-router/`.
//!
//! Converts a TSV manifest (`id<TAB>path<TAB>ext`) with the code-intelligence document
//! router, one engine/policy per invocation, and emits one JSONL row per
//! (document, repetition). This is the same code path the retrieval corpus
//! uses, so the snapshot experiment measures what ships.
//!
//! ```text
//! cargo run --release --example docbench -- \
//!   --manifest corpus.tsv --engine auto --fallback best-effort \
//!   --jsonl out.jsonl --out outputs/
//! ```
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use code_intelligence::{ConvertedDocument, DocumentEngine, FallbackPolicy, convert_document};

struct Entry {
    id: String,
    path: PathBuf,
    ext: String,
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn parse_manifest(raw: &str) -> Vec<Entry> {
    raw.lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let mut parts = line.splitn(3, '\t');
            let id = parts.next()?.to_string();
            let path = PathBuf::from(parts.next()?);
            let ext = parts.next().unwrap_or("").trim().to_string();
            Some(Entry { id, path, ext })
        })
        .collect()
}

fn row(
    id: &str,
    ext: &str,
    engine: DocumentEngine,
    fallback: FallbackPolicy,
    rep: i32,
    ns: u128,
    result: &Result<ConvertedDocument, code_intelligence::DocError>,
) -> String {
    let (ok, used, fell_back, reason, bytes, err) = match result {
        Ok(converted) => (
            true,
            converted.engine.label().to_string(),
            converted.fell_back(),
            converted
                .fallback
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
            converted.markdown.len(),
            String::new(),
        ),
        Err(error) => (
            false,
            String::new(),
            false,
            String::new(),
            0,
            error.to_string(),
        ),
    };
    format!(
        "{{\"id\":\"{}\",\"ext\":\"{}\",\"engine_requested\":\"{}\",\"fallback_policy\":\"{}\",\"engine_used\":\"{}\",\"fell_back\":{},\"fallback_reason\":\"{}\",\"rep\":{},\"ns\":{},\"ok\":{},\"err\":\"{}\",\"out_bytes\":{}}}",
        json_escape(id),
        json_escape(ext),
        engine.label(),
        fallback.label(),
        json_escape(&used),
        fell_back,
        json_escape(&reason),
        rep,
        ns,
        ok,
        json_escape(&err),
        bytes
    )
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut manifest = PathBuf::new();
    let mut jsonl = PathBuf::new();
    let mut outdir = PathBuf::new();
    let mut engine = DocumentEngine::Auto;
    let mut fallback = FallbackPolicy::BestEffort;
    let mut reps: i32 = 1;
    let mut warmup: i32 = 0;

    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--manifest" => manifest = PathBuf::from(value),
            "--jsonl" => jsonl = PathBuf::from(value),
            "--out" => outdir = PathBuf::from(value),
            "--engine" => {
                engine = DocumentEngine::parse(&value).unwrap_or_else(|| {
                    eprintln!("unknown engine '{value}' (auto | anydoc | xberg)");
                    std::process::exit(2);
                });
            }
            "--fallback" => {
                fallback = FallbackPolicy::parse(&value).unwrap_or_else(|| {
                    eprintln!("unknown fallback '{value}' (best-effort | strict)");
                    std::process::exit(2);
                });
            }
            "--reps" => reps = value.parse().unwrap_or(1),
            "--warmup" => warmup = value.parse().unwrap_or(0),
            other => eprintln!("unknown argument: {other}"),
        }
        i += 2;
    }
    if manifest.as_os_str().is_empty() || jsonl.as_os_str().is_empty() {
        eprintln!(
            "usage: docbench --manifest FILE --jsonl FILE [--engine auto|anydoc|xberg] [--fallback best-effort|strict] [--reps N] [--warmup N] [--out DIR]"
        );
        std::process::exit(2);
    }

    let entries = parse_manifest(&std::fs::read_to_string(&manifest).expect("manifest readable"));
    if !outdir.as_os_str().is_empty() {
        std::fs::create_dir_all(&outdir).expect("outdir");
    }
    let mut sink = std::io::BufWriter::new(std::fs::File::create(&jsonl).expect("jsonl"));

    for rep in -warmup..reps {
        for entry in &entries {
            let started = Instant::now();
            let result: Result<ConvertedDocument, code_intelligence::DocError> =
                convert_document(&entry.path, engine, fallback).await;
            let ns = started.elapsed().as_nanos();
            if rep == 0
                && let Ok(converted) = &result
                && !outdir.as_os_str().is_empty()
            {
                let name = entry.id.replace('/', "__") + ".md";
                let _ = std::fs::write(outdir.join(name), converted.markdown.as_bytes());
            }
            writeln!(
                sink,
                "{}",
                row(&entry.id, &entry.ext, engine, fallback, rep, ns, &result)
            )
            .ok();
        }
        sink.flush().ok();
    }
}
