//! Document -> Markdown conversion with an evidence-derived engine router.
//!
//! Binary/office documents (Word, PowerPoint, Excel, OpenDocument, RTF, EPUB,
//! PDF) are converted to GitHub-Flavored Markdown **in-process** so the planner
//! can ground a plan on a PDF spec, a DOCX design doc, or an XLSX sheet exactly
//! as it reads a `.rs` file. The converted Markdown then flows through the
//! existing pipeline unchanged: the retrieval corpus chunks it with the xberg
//! chunker, and Headroom reversibly compresses the tool result.
//!
//! # Engine router
//!
//! Two in-process converters are compiled in: [`anydoc`] and the xberg
//! extraction engine. They disagree per format, so the primary engine is a
//! *measured* table ([`XBERG_PRIMARY`]), not a guess:
//!
//! * `pptx` — xberg is both more faithful (+0.074 token F1 over 25 paired
//!   documents, 11 wins to 1) and faster (0.87× the latency), so xberg is
//!   primary.
//! * every other format — anydoc is equal or better on fidelity and covers
//!   more documents, so anydoc is primary.
//!
//! The table is reproduced — and drift-checked — by `bench/doc-router/`; the
//! `golden/route.json` there is the source of truth, and
//! `tests/document_router.rs` fails when this file and the golden disagree.
//!
//! When the primary engine fails, [`FallbackPolicy`] decides whether the
//! secondary gets a turn. `BestEffort` (the corpus default) takes a non-empty
//! conversion from the other engine and records why the primary failed;
//! `Strict` returns the primary error. Encryption never falls back (both
//! engines fail on it), and an empty conversion is a failure, never a result.
//! This is what turns a scanned PDF from "skipped" into "indexed with partial
//! text plus a recorded fallback" without ever passing off truncated output as
//! complete when the caller asked for strictness.
//!
//! Any router, engine or policy change must bump [`ROUTER_REVISION`]: the
//! retrieval table identity (see [`router_identity`]) changes with it, so the
//! document corpus re-embeds instead of mixing two output shapes.
//!
//! **Text parity.** A plain UTF-8 file (including `.csv`) never reaches this
//! module: it is read verbatim. Only a recognized document extension, or bytes
//! that are *not* valid UTF-8 and carry a detectable document signature, are
//! converted.
//!
//! [anydoc]: https://github.com/firecrawl/anydoc

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::types::{DocumentFormat, PageCount};

/// Extensions the in-process converters handle, lower-case, without a dot.
///
/// `csv` is deliberately absent: a delimited text file is already UTF-8 text
/// and must keep reading verbatim (byte-parity with the pre-converter behavior).
pub const DOCUMENT_EXTENSIONS: &[&str] = &[
    "pdf", "doc", "docx", "docm", "dot", "odt", "rtf", "epub", "ppt", "pptx", "pptm", "pps",
    "ppsx", "ppsm", "pot", "xls", "xlsx", "xlsm", "xlsb", "ods", "odp",
];

/// The pinned anydoc version, stamped into the retrieval corpus identity when
/// the document corpus is off (source-only tables are unchanged by routing).
pub const ANYDOC_VERSION: &str = "0.2.4";

/// The pinned xberg version the router was measured against.
pub const XBERG_VERSION: &str = "1.3.10";

/// Bump when the route table, an engine version or the fallback semantics
/// change: the document corpus identity includes it, so embeddings are never
/// mixed across converter output shapes.
pub const ROUTER_REVISION: u32 = 1;

/// Formats whose primary engine is xberg, from the `bench/doc-router` snapshot.
///
/// `pptx` is the only format where xberg beat anydoc on both fidelity and
/// latency; every other format keeps anydoc primary. Its PresentationML
/// siblings (`pptm`, `ppsx`, `ppsm`) share the parser and the verdict, while
/// the binary `.ppt`/`.pps`/`.pot` formats do not. Do not edit this list without
/// re-running the experiment and updating `golden/route.json`.
pub const XBERG_PRIMARY: &[&str] = &["pptx", "pptm", "ppsx", "ppsm"];

/// Which engine a caller wants. `Auto` follows [`XBERG_PRIMARY`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DocumentEngine {
    /// Route by the measured table: xberg for `pptx`, anydoc otherwise.
    #[default]
    Auto,
    /// Pin anydoc for every format (A/B runs and escape hatch).
    Anydoc,
    /// Pin xberg for every format (A/B runs and escape hatch).
    Xberg,
}

impl DocumentEngine {
    /// Stable label used by identity strings, logs and the bench harness.
    pub fn label(self) -> &'static str {
        match self {
            DocumentEngine::Auto => "auto",
            DocumentEngine::Anydoc => "anydoc",
            DocumentEngine::Xberg => "xberg",
        }
    }

    /// Parse a configured value (`auto` | `anydoc` | `xberg`).
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(DocumentEngine::Auto),
            "anydoc" => Some(DocumentEngine::Anydoc),
            "xberg" => Some(DocumentEngine::Xberg),
            _ => None,
        }
    }
}

impl std::str::FromStr for DocumentEngine {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        Self::parse(raw).ok_or_else(|| {
            format!("invalid document engine '{raw}' (expected auto | anydoc | xberg)")
        })
    }
}

/// What to do when the primary engine cannot convert a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FallbackPolicy {
    /// Try the other engine; accept a non-empty result and record the primary
    /// error on the conversion. The corpus default.
    #[default]
    BestEffort,
    /// Return the primary engine's error. The caller wants the engine it named,
    /// and "needs OCR" stays an escalation signal.
    Strict,
}

impl FallbackPolicy {
    /// Stable label used by identity strings, logs and the bench harness.
    pub fn label(self) -> &'static str {
        match self {
            FallbackPolicy::BestEffort => "best-effort",
            FallbackPolicy::Strict => "strict",
        }
    }

    /// Parse a configured value (`best-effort` | `strict`).
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "best-effort" | "best_effort" | "besteffort" => Some(FallbackPolicy::BestEffort),
            "strict" => Some(FallbackPolicy::Strict),
            _ => None,
        }
    }
}

impl std::str::FromStr for FallbackPolicy {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        Self::parse(raw).ok_or_else(|| {
            format!("invalid document fallback '{raw}' (expected best-effort | strict)")
        })
    }
}

/// A resolved engine (never `Auto`), recorded on every conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// The anydoc converter.
    Anydoc,
    /// The xberg extraction engine.
    Xberg,
}

impl Engine {
    /// Stable label used in provenance notes and identity strings.
    pub fn label(self) -> &'static str {
        match self {
            Engine::Anydoc => "anydoc",
            Engine::Xberg => "xberg",
        }
    }

    /// The other engine, for the fallback turn.
    pub fn other(self) -> Self {
        match self {
            Engine::Anydoc => Engine::Xberg,
            Engine::Xberg => Engine::Anydoc,
        }
    }
}

/// A converted document plus the provenance of the conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedDocument {
    /// GitHub-Flavored Markdown, never empty.
    pub markdown: String,
    /// The engine that produced `markdown`.
    pub engine: Engine,
    /// The detected format label (`docx`, `pptx`, …).
    pub format: DocumentFormat,
    /// The primary engine's error when the other engine produced the result.
    pub fallback: Option<DocError>,
}

impl ConvertedDocument {
    /// Whether the document was converted by the fallback engine.
    pub fn fell_back(&self) -> bool {
        self.fallback.is_some()
    }
}

/// The typed conversion failure surface (anydoc's `ConvertError` and xberg's
/// errors flattened so callers do not depend on either upstream enum).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocError {
    /// The format is unknown or cannot be converted at all.
    Unsupported(String),
    /// Some pages are scanned/image-only and need OCR.
    NeedsOcr {
        /// 1-indexed pages that need OCR.
        pages: Vec<u32>,
        /// Pages in the document.
        page_count: u32,
    },
    /// The document is encrypted or password-protected.
    Encrypted,
    /// Structurally unusable, or the converter returned no content.
    Malformed(String),
    /// A fixed safety/resource limit was exceeded.
    ResourceLimit(String),
    /// The bytes could not be read.
    Io(String),
}

impl DocError {
    /// Whether this failure is the escalation signal to an OCR-capable path.
    pub fn needs_ocr(&self) -> bool {
        matches!(self, DocError::NeedsOcr { .. })
    }

    /// Whether the other engine may be tried for this failure.
    ///
    /// Encryption fails identically in both engines and an I/O error will not
    /// improve on retry, so neither falls back. Everything else may: a
    /// malformed OOXML package, an anydoc safety limit, or a scanned page can
    /// still yield a non-empty conversion from the other engine.
    pub fn allows_fallback(&self) -> bool {
        !matches!(self, DocError::Encrypted | DocError::Io(_))
    }
}

impl fmt::Display for DocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DocError::Unsupported(what) => write!(f, "unsupported document: {what}"),
            DocError::NeedsOcr { pages, page_count } => match pages.as_slice() {
                [page] => write!(f, "page {page} of {page_count} needs OCR"),
                _ if pages.len() as u32 >= *page_count => {
                    write!(f, "all {page_count} pages need OCR")
                }
                _ => write!(f, "{} of {page_count} pages need OCR", pages.len()),
            },
            DocError::Encrypted => write!(f, "document is encrypted"),
            DocError::Malformed(detail) => write!(f, "malformed document: {detail}"),
            DocError::ResourceLimit(detail) => {
                write!(f, "document exceeds a safety limit: {detail}")
            }
            DocError::Io(detail) => write!(f, "document read failed: {detail}"),
        }
    }
}

impl std::error::Error for DocError {}

/// Whether `path` names a document this module converts (by extension).
pub fn is_document_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            DOCUMENT_EXTENSIONS
                .iter()
                .any(|known| known.eq_ignore_ascii_case(ext))
        })
}

/// The primary engine for a format label (`pptx` → xberg, else anydoc).
pub fn route_for(format: &str) -> Engine {
    if XBERG_PRIMARY
        .iter()
        .any(|known| known.eq_ignore_ascii_case(format))
    {
        Engine::Xberg
    } else {
        Engine::Anydoc
    }
}

/// The primary and secondary engine for a conversion request.
fn plan(engine: DocumentEngine, format: &str) -> (Engine, Engine) {
    let primary = match engine {
        DocumentEngine::Auto => route_for(format),
        DocumentEngine::Anydoc => Engine::Anydoc,
        DocumentEngine::Xberg => Engine::Xberg,
    };
    (primary, primary.other())
}

/// The retrieval-corpus identity fragment for the document corpus.
///
/// Includes the router revision, both engine versions and the configured mode
/// and policy, because each changes the converted text. Source-only corpora
/// keep using [`ANYDOC_VERSION`] directly, so their tables and manifests are
/// unchanged by routing.
pub fn router_identity(engine: DocumentEngine, fallback: FallbackPolicy) -> String {
    format!(
        "r{ROUTER_REVISION}anydoc{}-xberg{}-{}-{}",
        ANYDOC_VERSION.replace('.', ""),
        XBERG_VERSION.replace('.', ""),
        engine.label(),
        fallback.label()
    )
}

/// The document label written into a `// document:` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentMeta {
    /// Stable format name (`docx`, `pdf`, …).
    pub format: DocumentFormat,
    /// Page count, when the format reports one.
    pub pages: Option<PageCount>,
    /// The engine that converted the document.
    pub engine: Engine,
}

/// Flatten anydoc's `ConvertError` into [`DocError`].
fn map_convert_error(err: anydoc::ConvertError) -> DocError {
    match err {
        anydoc::ConvertError::Unsupported(what) => DocError::Unsupported(what),
        anydoc::ConvertError::NeedsOcr { pages, page_count } => {
            DocError::NeedsOcr { pages, page_count }
        }
        anydoc::ConvertError::Malformed { part, detail } => match part {
            Some(part) => DocError::Malformed(format!("{part}: {detail}")),
            None => DocError::Malformed(detail),
        },
        anydoc::ConvertError::Encrypted => DocError::Encrypted,
        anydoc::ConvertError::ResourceLimit { limit, detail } => {
            DocError::ResourceLimit(format!("{limit}: {detail}"))
        }
        anydoc::ConvertError::MissingPart { part } => {
            DocError::Malformed(format!("missing part {part}"))
        }
        anydoc::ConvertError::Io(e) => DocError::Io(e.to_string()),
        // `ConvertError` is `#[non_exhaustive]`: a future variant degrades to a
        // typed unsupported error rather than failing to compile.
        other => DocError::Unsupported(other.to_string()),
    }
}

/// Flatten xberg's error surface by classification. The upstream enum is broad
/// (OCR, layout, security, I/O), so the message is classified into the same
/// variants callers already handle.
fn map_xberg_error(err: xberg::XbergError) -> DocError {
    let text = err.to_string();
    let low = text.to_ascii_lowercase();
    if low.contains("encrypt") || low.contains("password") || low.contains("protected") {
        DocError::Encrypted
    } else if low.contains("unsupported")
        || low.contains("not supported")
        || low.contains("cannot detect")
    {
        DocError::Unsupported(text)
    } else if low.contains("limit")
        || low.contains("too large")
        || low.contains("exceeds")
        || low.contains("zip bomb")
        || low.contains("depth")
    {
        DocError::ResourceLimit(text)
    } else {
        DocError::Malformed(text)
    }
}

/// The MIME type hint xberg needs for a bytes-only conversion.
fn mime_for(format: &str) -> &'static str {
    match format {
        "doc" | "dot" => "application/msword",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docm" => "application/vnd.ms-word.document.macroenabled.12",
        "odt" => "application/vnd.oasis.opendocument.text",
        "ods" => "application/vnd.oasis.opendocument.spreadsheet",
        "odp" => "application/vnd.oasis.opendocument.presentation",
        "rtf" => "application/rtf",
        "epub" => "application/epub+zip",
        "pdf" => "application/pdf",
        "ppt" | "pps" | "pot" => "application/vnd.ms-powerpoint",
        "pptx" | "ppsx" => {
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        }
        "pptm" | "ppsm" => "application/vnd.ms-powerpoint.presentation.macroenabled.12",
        "xls" => "application/vnd.ms-excel",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsm" => "application/vnd.ms-excel.sheet.macroenabled.12",
        "xlsb" => "application/vnd.ms-excel.sheet.binary.macroenabled.12",
        _ => "application/octet-stream",
    }
}

/// Run one engine. `path` is `Some` when the file exists on disk (xberg reads
/// it itself, matching how the benchmark measured it); bytes-only callers get a
/// MIME hint instead.
async fn run_engine(
    engine: Engine,
    path: Option<&Path>,
    bytes: &[u8],
    format: Option<anydoc::Format>,
    label: &str,
) -> Result<String, DocError> {
    match engine {
        Engine::Anydoc => anydoc::to_markdown_bytes(bytes, format).map_err(map_convert_error),
        Engine::Xberg => run_xberg(path, bytes, label).await,
    }
}

/// Convert with xberg on a spawned task so a panic in one document is a typed
/// failure instead of aborting the whole corpus build.
async fn run_xberg(path: Option<&Path>, bytes: &[u8], label: &str) -> Result<String, DocError> {
    let config = xberg::ExtractionConfig {
        output_format: xberg::OutputFormat::Markdown,
        use_cache: false,
        ..Default::default()
    };
    let input = match path {
        Some(path) => xberg::ExtractInput::from_uri(path.to_string_lossy().into_owned()),
        None => xberg::ExtractInput::from_bytes(
            bytes.to_vec(),
            mime_for(label),
            Some(format!("document.{label}")),
        ),
    };
    let handle = tokio::spawn(async move { xberg::extract(input, &config).await });
    let outcome = handle
        .await
        .map_err(|join| DocError::Malformed(format!("xberg task failed: {join}")))?;
    let result = outcome.map_err(map_xberg_error)?;
    let markdown = result
        .results
        .first()
        .map(|doc| doc.content.clone())
        .unwrap_or_default();
    if markdown.trim().is_empty() {
        return Err(DocError::Malformed("xberg returned no content".into()));
    }
    Ok(markdown)
}

/// Convert in-memory document bytes to Markdown under an explicit engine and
/// fallback policy. `format` selects the parser; `None` detects it from the
/// content, and `format_hint` names the extension when detection cannot.
pub async fn convert_bytes_with(
    bytes: &[u8],
    format: Option<anydoc::Format>,
    format_hint: Option<&str>,
    engine: DocumentEngine,
    fallback: FallbackPolicy,
) -> Result<ConvertedDocument, DocError> {
    // The extension is the better label when the caller knows it (xberg needs
    // a real MIME type); the detected format is the fallback.
    let label = format_hint
        .map(|hint| hint.trim_start_matches('.').to_ascii_lowercase())
        .or_else(|| format.map(format_label).map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_string());
    convert_inner(None, bytes, format, &label, engine, fallback).await
}

/// Convert a document file to Markdown under an explicit engine and policy.
pub async fn convert_file_with(
    path: &Path,
    engine: DocumentEngine,
    fallback: FallbackPolicy,
) -> Result<ConvertedDocument, DocError> {
    let bytes = std::fs::read(path).map_err(|e| DocError::Io(e.to_string()))?;
    // Content first, extension second: anydoc's own CLI does the same, and a
    // mislabelled file (an RTF named `.doc`, a DOCX named `.doc`) then converts
    // with the parser its bytes call for instead of failing on the suffix.
    let format = anydoc::Format::from_bytes(&bytes).or_else(|| anydoc::Format::from_path(path));
    let label = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .or_else(|| format.map(format_label).map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_string());
    convert_inner(Some(path), &bytes, format, &label, engine, fallback).await
}

/// Convert a document file with the default policy (`auto` + `best-effort`).
pub async fn convert_file(path: &Path) -> Result<ConvertedDocument, DocError> {
    convert_file_with(path, DocumentEngine::Auto, FallbackPolicy::BestEffort).await
}

/// Convert in-memory bytes with the default policy (`auto` + `best-effort`).
#[allow(dead_code)]
pub async fn convert_bytes(
    bytes: &[u8],
    format: Option<anydoc::Format>,
) -> Result<ConvertedDocument, DocError> {
    convert_bytes_with(
        bytes,
        format,
        None,
        DocumentEngine::Auto,
        FallbackPolicy::BestEffort,
    )
    .await
}

async fn convert_inner(
    path: Option<&Path>,
    bytes: &[u8],
    format: Option<anydoc::Format>,
    label: &str,
    engine: DocumentEngine,
    fallback: FallbackPolicy,
) -> Result<ConvertedDocument, DocError> {
    let (primary, secondary) = plan(engine, label);
    let format_typed = DocumentFormat::from(label);
    let primary_error = match run_engine(primary, path, bytes, format, label).await {
        Ok(markdown) if !markdown.trim().is_empty() => {
            return Ok(ConvertedDocument {
                markdown,
                engine: primary,
                format: format_typed,
                fallback: None,
            });
        }
        Ok(_) => DocError::Malformed("converter returned no content".into()),
        Err(error) => error,
    };
    if fallback == FallbackPolicy::BestEffort && primary_error.allows_fallback() {
        let secondary_attempt = run_engine(secondary, path, bytes, format, label).await;
        if let Ok(markdown) = secondary_attempt
            && !markdown.trim().is_empty()
        {
            return Ok(ConvertedDocument {
                markdown,
                engine: secondary,
                format: format_typed,
                fallback: Some(primary_error),
            });
        }
    }
    Err(primary_error)
}

/// The conversion decision for one file read: `Some(format)` when the bytes
/// should be converted, `None` when they are plain text to read verbatim.
///
/// A recognized document extension always converts (RTF is ASCII, so content
/// sniffing alone would miss it). Otherwise only non-UTF-8 bytes with a
/// detectable document signature convert — a text file is never touched.
#[allow(dead_code)]
pub fn should_convert(path: &Path, bytes: &[u8]) -> Option<anydoc::Format> {
    if is_document_path(path) {
        return anydoc::Format::from_path(path).or_else(|| anydoc::Format::from_bytes(bytes));
    }
    if std::str::from_utf8(bytes).is_err() {
        return anydoc::Format::from_bytes(bytes);
    }
    None
}

/// Stable, lower-case format label for the header and routing (the upstream
/// enum has no `Display`).
pub fn format_label(format: anydoc::Format) -> &'static str {
    use anydoc::Format;
    match format {
        Format::Doc => "doc",
        Format::Docx => "docx",
        Format::Odt => "odt",
        Format::Pdf => "pdf",
        Format::Ppt => "ppt",
        Format::Pptx => "pptx",
        Format::Rtf => "rtf",
        Format::Epub => "epub",
        Format::Excel => "excel",
        Format::Ods => "ods",
        Format::Odp => "odp",
        Format::Csv => "csv",
    }
}

/// Format a converted document for the model context: a provenance header
/// (`// document: <relpath> (format=<fmt>[, pages=N][, engine=<engine>])`)
/// plus the Markdown, truncated with the same head-elision style as
/// [`crate::readfile::format_file`]. `max_chars == 0` means unlimited.
#[allow(dead_code)]
pub fn format_document(
    markdown: &str,
    rel_path: &str,
    meta: &DocumentMeta,
    max_chars: usize,
) -> String {
    let total = markdown.chars().count();
    let body: String = if max_chars > 0 && total > max_chars {
        let head: String = markdown.chars().take(max_chars).collect();
        format!("{head}\n… [truncated: {total} chars total; converted document]")
    } else {
        markdown.to_string()
    };
    let pages = meta
        .pages
        .map(|p| format!(", pages={p}"))
        .unwrap_or_default();
    format!(
        "// document: {rel_path} (format={}{pages}, engine={})\n{body}",
        meta.format,
        meta.engine.label()
    )
}

/// The permissive block returned when a document needs OCR: the planner can
/// still cite "scanned, needs OCR" as a fact, and is pointed at the deep parser
/// tool when one is attached.
#[allow(dead_code)]
pub fn needs_ocr_note(rel_path: &str, pages: &[u32], page_count: u32) -> String {
    let pages = if pages.is_empty() {
        format!("{page_count}")
    } else {
        pages
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "// document: {rel_path} (scanned/image-only; pages needing OCR: {pages} of {page_count})\n\
         // [needs-OCR: no local OCR is available for this document — if a read_document tool is listed, call it for OCR/deep parsing]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn path(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn document_extensions_cover_office_and_pdf_but_not_csv() {
        for name in [
            "a.pdf", "a.docx", "a.pptx", "a.xlsx", "a.odt", "a.rtf", "a.epub",
        ] {
            assert!(is_document_path(&path(name)), "{name} must convert");
        }
        for name in ["a.rs", "a.md", "a.csv", "a.json", "a.txt"] {
            assert!(!is_document_path(&path(name)), "{name} must stay text");
        }
        assert!(is_document_path(&path("REPORT.PDF")));
    }

    #[test]
    fn route_table_puts_the_presentationml_family_on_xberg() {
        for presentationml in ["pptx", "pptm", "ppsx", "ppsm", "PPTX"] {
            assert_eq!(route_for(presentationml), Engine::Xberg, "{presentationml}");
        }
        for other in [
            "docx", "odp", "odt", "xlsx", "rtf", "epub", "pdf", "doc", "ppt", "pps", "pot",
        ] {
            assert_eq!(route_for(other), Engine::Anydoc, "{other}");
        }
        assert_eq!(XBERG_PRIMARY, &["pptx", "pptm", "ppsx", "ppsm"]);
    }

    #[test]
    fn plan_honours_forced_engines_and_auto_routes() {
        assert_eq!(
            plan(DocumentEngine::Auto, "pptx"),
            (Engine::Xberg, Engine::Anydoc)
        );
        assert_eq!(
            plan(DocumentEngine::Auto, "docx"),
            (Engine::Anydoc, Engine::Xberg)
        );
        assert_eq!(
            plan(DocumentEngine::Anydoc, "pptx"),
            (Engine::Anydoc, Engine::Xberg)
        );
        assert_eq!(
            plan(DocumentEngine::Xberg, "pot"),
            (Engine::Xberg, Engine::Anydoc)
        );
        assert_eq!(
            plan(DocumentEngine::Xberg, "docx"),
            (Engine::Xberg, Engine::Anydoc)
        );
    }

    #[test]
    fn encryption_and_io_never_fall_back() {
        assert!(!DocError::Encrypted.allows_fallback());
        assert!(!DocError::Io("nope".into()).allows_fallback());
        for err in [
            DocError::NeedsOcr {
                pages: vec![1],
                page_count: 2,
            },
            DocError::Malformed("bad".into()),
            DocError::Unsupported("x".into()),
            DocError::ResourceLimit("depth".into()),
        ] {
            assert!(err.allows_fallback(), "{err:?}");
        }
    }

    #[test]
    fn router_identity_tracks_revision_engines_mode_and_policy() {
        let auto = router_identity(DocumentEngine::Auto, FallbackPolicy::BestEffort);
        assert!(auto.starts_with("r1anydoc024-xberg1310-auto-best-effort"));
        assert_ne!(
            auto,
            router_identity(DocumentEngine::Anydoc, FallbackPolicy::BestEffort)
        );
        assert_ne!(
            auto,
            router_identity(DocumentEngine::Auto, FallbackPolicy::Strict)
        );
        assert_eq!(
            auto,
            router_identity(DocumentEngine::Auto, FallbackPolicy::BestEffort)
        );
    }

    #[test]
    fn engine_and_policy_labels_round_trip() {
        for engine in [
            DocumentEngine::Auto,
            DocumentEngine::Anydoc,
            DocumentEngine::Xberg,
        ] {
            assert_eq!(DocumentEngine::parse(engine.label()), Some(engine));
        }
        for policy in [FallbackPolicy::BestEffort, FallbackPolicy::Strict] {
            assert_eq!(FallbackPolicy::parse(policy.label()), Some(policy));
        }
        assert!(DocumentEngine::parse("nope").is_none());
        assert!(FallbackPolicy::parse("nope").is_none());
    }

    #[test]
    fn format_document_writes_provenance_header_and_bounds_the_body() {
        let meta = DocumentMeta {
            format: "docx".into(),
            pages: None,
            engine: Engine::Anydoc,
        };
        let out = format_document("# Title\n\nbody", "docs/spec.docx", &meta, 0);
        assert_eq!(
            out,
            "// document: docs/spec.docx (format=docx, engine=anydoc)\n# Title\n\nbody"
        );

        let with_pages = DocumentMeta {
            format: "pdf".into(),
            pages: Some(PageCount::new(12)),
            engine: Engine::Xberg,
        };
        let head = format_document("body", "a.pdf", &with_pages, 0);
        assert!(
            head.starts_with("// document: a.pdf (format=pdf, pages=12, engine=xberg)"),
            "{head}"
        );

        let truncated = format_document("abcdefghij", "a.docx", &meta, 4);
        assert!(
            truncated.starts_with("// document: a.docx (format=docx, engine=anydoc)\nabcd"),
            "{truncated}"
        );
        assert!(
            truncated.contains("truncated: 10 chars total"),
            "{truncated}"
        );
    }

    #[test]
    fn convert_error_mapping_is_typed_and_displayed() {
        let cases = [
            (
                DocError::NeedsOcr {
                    pages: vec![2],
                    page_count: 5,
                },
                "page 2 of 5 needs OCR",
            ),
            (DocError::Encrypted, "document is encrypted"),
            (DocError::Unsupported("x".into()), "unsupported document: x"),
            (DocError::Malformed("bad".into()), "malformed document: bad"),
            (
                DocError::ResourceLimit("depth: too deep".into()),
                "document exceeds a safety limit: depth: too deep",
            ),
            (DocError::Io("nope".into()), "document read failed: nope"),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
        assert!(
            DocError::NeedsOcr {
                pages: vec![1],
                page_count: 1
            }
            .needs_ocr()
        );
        assert!(!DocError::Encrypted.needs_ocr());
    }

    #[test]
    fn needs_ocr_note_names_pages_and_points_at_the_deep_tool() {
        let note = needs_ocr_note("scan.pdf", &[1, 3], 4);
        assert!(note.contains("scan.pdf"), "{note}");
        assert!(note.contains("1, 3"), "{note}");
        assert!(note.contains("read_document"), "{note}");

        let all = needs_ocr_note("scan.pdf", &[], 4);
        assert!(all.contains("no local OCR"), "{all}");
        assert!(all.contains("pages needing OCR: 4 of 4"), "{all}");
    }

    #[tokio::test]
    async fn rtf_converts_end_to_end_to_markdown() {
        // A minimal, well-formed RTF document: pure ASCII so it also proves the
        // extension-based routing (content sniffing alone would call it text).
        let rtf =
            b"{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Times;}}\\f0\\fs24 Hello anydoc world.\\par}";
        let converted = convert_bytes(rtf.as_slice(), Some(anydoc::Format::Rtf))
            .await
            .expect("rtf converts");
        assert_eq!(converted.engine, Engine::Anydoc);
        assert!(converted.fallback.is_none());
        assert!(
            converted.markdown.contains("Hello anydoc world."),
            "{}",
            converted.markdown
        );
    }

    #[tokio::test]
    async fn unrecognized_bytes_are_a_typed_unsupported_error() {
        let err = convert_bytes(&[0x00, 0x01, 0x02], None)
            .await
            .expect_err("must not convert");
        assert!(matches!(err, DocError::Unsupported(_)), "{err:?}");
    }

    #[tokio::test]
    async fn strict_policy_returns_the_primary_error_instead_of_falling_back() {
        // Any bytes with no document signature fail anydoc; under Strict the
        // error is returned as-is rather than handing the bytes to xberg.
        let err = convert_bytes_with(
            &[0x00, 0x01, 0x02],
            None,
            Some("bin"),
            DocumentEngine::Auto,
            FallbackPolicy::Strict,
        )
        .await
        .expect_err("must not convert");
        assert!(matches!(err, DocError::Unsupported(_)), "{err:?}");
    }
}
