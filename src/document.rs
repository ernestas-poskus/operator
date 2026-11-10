//! Document -> Markdown conversion for the planner's file readers.
//!
//! Binary/office documents (Word, PowerPoint, Excel, OpenDocument, RTF, EPUB,
//! PDF) are converted to GitHub-Flavored Markdown **in-process** by the
//! [anydoc] crate — pure Rust, no network, no models, no `ort` — so the planner
//! can ground a plan on a PDF spec, a DOCX design doc, or an XLSX sheet exactly
//! as it reads a `.rs` file. The converted Markdown then flows through the
//! existing pipeline unchanged: [crate::readfile] bounds it with
//! `--file-read-max-chars`, the retrieval corpus chunks it with the xberg
//! chunker, and Headroom reversibly compresses the tool result.
//!
//! **Text parity.** A plain UTF-8 file (including `.csv`) never reaches this
//! module: [crate::readfile] reads it verbatim, byte-for-byte as before. Only a
//! recognized document extension, or bytes that are *not* valid UTF-8 and carry
//! a detectable document signature, are converted.
//!
//! **OCR is out of scope here.** anydoc reads text-based PDFs only; a scanned
//! page makes it fail with [`DocError::NeedsOcr`]. That is the signal to
//! escalate to the out-of-process `docling-rs serve` deep parser (see
//! [crate::docling]), never a hard read failure.
//!
//! [anydoc]: https://github.com/firecrawl/anydoc

use std::fmt;
use std::path::Path;

use crate::types::{DocumentFormat, PageCount};

/// Extensions the in-process converter handles, lower-case, without a dot.
///
/// `csv` is deliberately absent: a delimited text file is already UTF-8 text
/// and must keep reading verbatim (byte-parity with the pre-anydoc behavior).
pub const DOCUMENT_EXTENSIONS: &[&str] = &[
    "pdf", "doc", "docx", "docm", "dot", "odt", "rtf", "epub", "ppt", "pptx", "pptm", "pps",
    "ppsx", "ppsm", "pot", "xls", "xlsx", "xlsm", "xlsb", "ods", "odp",
];

/// The pinned converter version, stamped into the retrieval corpus identity
/// (a converter change must re-embed, never silently mix output shapes).
pub const ANYDOC_VERSION: &str = "0.2.4";

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

/// The typed conversion failure surface (anydoc's `ConvertError`, flattened so
/// callers do not depend on the upstream enum's `#[non_exhaustive]` shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocError {
    /// The format is unknown or cannot be converted at all.
    Unsupported(String),
    /// Some pages are scanned/image-only and need OCR, which anydoc does not do.
    NeedsOcr {
        /// 1-indexed pages that need OCR.
        pages: Vec<u32>,
        /// Pages in the document.
        page_count: u32,
    },
    /// The document is encrypted or password-protected.
    Encrypted,
    /// Structurally unusable (no meaningful content could be extracted).
    Malformed(String),
    /// A fixed safety/resource limit was exceeded.
    ResourceLimit(String),
    /// The bytes could not be read.
    Io(String),
}

impl DocError {
    /// Whether this failure is the escalation signal to the deep (docling) path.
    #[allow(dead_code)]
    pub fn needs_ocr(&self) -> bool {
        matches!(self, DocError::NeedsOcr { .. })
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

/// The document label written into the `// document:` header.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct DocumentMeta {
    /// Stable format name (`docx`, `pdf`, …).
    pub format: DocumentFormat,
    /// Page count, when the format reports one.
    pub pages: Option<PageCount>,
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

/// Convert in-memory document bytes to Markdown. `format` selects the parser;
/// `None` detects it from the content.
pub fn convert_bytes(bytes: &[u8], format: Option<anydoc::Format>) -> Result<String, DocError> {
    anydoc::to_markdown_bytes(bytes, format).map_err(map_convert_error)
}

/// Convert a document file to Markdown (format from its extension, else from
/// the content).
pub fn convert_file(path: &Path) -> Result<String, DocError> {
    let bytes = std::fs::read(path).map_err(|e| DocError::Io(e.to_string()))?;
    let format = anydoc::Format::from_path(path);
    convert_bytes(&bytes, format)
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

/// Stable, lower-case format label for the header (the upstream enum has no
/// `Display`).
#[allow(dead_code)]
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
/// (`// document: <relpath> (format=<fmt>[, pages=N])`) plus the Markdown,
/// truncated with the same head-elision style as [crate::readfile::format_file].
/// `max_chars == 0` means unlimited.
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
        "// document: {rel_path} (format={}{pages})\n{body}",
        meta.format
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
    fn should_convert_uses_extension_then_signature_only_for_binary_bytes() {
        let rtf = b"{\\rtf1\\ansi Hello}".as_slice();
        assert_eq!(
            should_convert(&path("a.rtf"), rtf),
            Some(anydoc::Format::Rtf)
        );
        assert_eq!(should_convert(&path("a.txt"), b"plain text"), None);
        // A document signature under an unknown extension is caught from the
        // bytes — but only when the bytes are not UTF-8 (a PDF header alone is
        // ASCII, so a real PDF's binary tail is what routes it).
        assert_eq!(
            should_convert(&path("a.bin"), b"%PDF-1.4\xff\xfe"),
            Some(anydoc::Format::Pdf)
        );
        assert_eq!(should_convert(&path("a.bin"), &[0xff, 0xfe, 0x00]), None);
    }

    #[test]
    fn format_document_writes_a_provenance_header_and_bounds_the_body() {
        let meta = DocumentMeta {
            format: "docx".into(),
            pages: None,
        };
        let out = format_document("# Title\n\nbody", "docs/spec.docx", &meta, 0);
        assert_eq!(
            out,
            "// document: docs/spec.docx (format=docx)\n# Title\n\nbody"
        );

        let with_pages = DocumentMeta {
            format: "pdf".into(),
            pages: Some(PageCount::new(12)),
        };
        let head = format_document("body", "a.pdf", &with_pages, 0);
        assert!(
            head.starts_with("// document: a.pdf (format=pdf, pages=12)"),
            "{head}"
        );

        let truncated = format_document("abcdefghij", "a.docx", &meta, 4);
        assert!(
            truncated.starts_with("// document: a.docx (format=docx)\nabcd"),
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

    #[test]
    fn rtf_converts_end_to_end_to_markdown() {
        // A minimal, well-formed RTF document: pure ASCII so it also proves the
        // extension-based routing (content sniffing alone would call it text).
        let rtf =
            b"{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Times;}}\\f0\\fs24 Hello anydoc world.\\par}";
        let md = convert_bytes(rtf.as_slice(), Some(anydoc::Format::Rtf)).unwrap();
        assert!(md.contains("Hello anydoc world."), "{md}");
    }

    #[test]
    fn unrecognized_bytes_are_a_typed_unsupported_error() {
        let err = convert_bytes(&[0x00, 0x01, 0x02], None).unwrap_err();
        assert!(matches!(err, DocError::Unsupported(_)), "{err:?}");
    }
}
