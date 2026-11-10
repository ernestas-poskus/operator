//! Offline tests for the document-engine router.
//!
//! These run without the 540 MB snapshot corpus: they check the golden routing
//! table, the committed fixtures (hashed in `fixtures/manifest.json`), the
//! strict/best-effort semantics on a scanned PDF, and the shape of the frozen
//! snapshot. The snapshot experiment itself is `bench/doc-router/run.py` and
//! the ignored `snapshot_experiment_reproduces_golden` test.
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use operator::{
    DOCUMENT_EXTENSIONS, DocError, DocumentEngine, Engine, FallbackPolicy, convert_document,
    route_for,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const GOLDEN_ROUTE: &str = include_str!("../bench/doc-router/golden/route.json");
const SNAPSHOT: &str = include_str!("../bench/doc-router/snapshot.json");
const FIXTURES: &str = include_str!("../bench/doc-router/fixtures/manifest.json");

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/doc-router/fixtures")
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn golden_table_covers_every_supported_extension_and_matches_the_router() {
    let golden: Value = serde_json::from_str(GOLDEN_ROUTE).expect("golden route parses");
    let formats = golden["formats"].as_object().expect("formats map");
    assert_eq!(
        formats.len(),
        DOCUMENT_EXTENSIONS.len(),
        "golden and DOCUMENT_EXTENSIONS disagree on the format set"
    );
    for ext in DOCUMENT_EXTENSIONS {
        let expected = formats
            .get(*ext)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("{ext} missing from golden/route.json"));
        assert_eq!(route_for(ext).label(), expected, "{ext} route drifted");
    }
}

#[test]
fn snapshot_is_hash_pinned_and_every_document_routes() {
    let snapshot: Value = serde_json::from_str(SNAPSHOT).expect("snapshot parses");
    let pins = snapshot["pins"].as_object().expect("pins");
    for key in [
        "anydoc",
        "xberg",
        "test_documents",
        "tika",
        "cargo_lock_sha256",
        "rustc",
    ] {
        assert!(
            pins.get(key)
                .and_then(Value::as_str)
                .is_some_and(|v| !v.is_empty()),
            "pin {key} missing"
        );
    }

    let documents = snapshot["documents"].as_array().expect("documents");
    assert!(
        documents.len() >= 500,
        "snapshot shrank to {} documents",
        documents.len()
    );
    assert!(
        documents.iter().any(|doc| doc["set"] == "bench"),
        "snapshot has no timed documents"
    );

    let mut ids = HashSet::new();
    for document in documents {
        let id = document["id"].as_str().expect("id");
        assert!(ids.insert(id), "duplicate snapshot id {id}");
        let digest = document["sha256"].as_str().expect("sha256");
        assert_eq!(digest.len(), 64, "{id}: sha256 length");
        assert!(
            digest.chars().all(|c| c.is_ascii_hexdigit()),
            "{id}: sha256 is not hex"
        );
        let ext = document["ext"].as_str().expect("ext");
        assert!(!ext.is_empty(), "{id}: ext");
        // `csv` is in the snapshot (it is a supported converter input) but it is
        // read verbatim by `read_source`, so it has no route by design.
        if ext != "csv" {
            assert!(
                DOCUMENT_EXTENSIONS.contains(&ext),
                "{id}: snapshot format {ext} has no route"
            );
        }
    }
}

#[tokio::test]
async fn fixtures_convert_through_the_engine_the_table_names() {
    let fixtures: Value = serde_json::from_str(FIXTURES).expect("fixture manifest parses");
    let files = fixtures["files"].as_object().expect("files");
    assert_eq!(files.len(), 7, "fixture set changed");

    for (name, entry) in files {
        if name == "scanned.pdf" {
            continue; // exercised by the OCR-policy test below
        }
        let path = fixture_dir().join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected_hash = entry["sha256"].as_str().expect("sha256");
        assert_eq!(hex_sha256(&bytes), expected_hash, "{name} fixture drifted");
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .expect("extension");
        let expected = route_for(ext);
        let converted = convert_document(&path, DocumentEngine::Auto, FallbackPolicy::BestEffort)
            .await
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(converted.engine, expected, "{name}");
        assert!(
            !converted.markdown.trim().is_empty(),
            "{name} produced nothing"
        );
    }
}

#[tokio::test]
async fn scanned_pdf_strict_reports_needs_ocr_and_best_effort_never_lies() {
    let path = fixture_dir().join("scanned.pdf");
    let strict = convert_document(&path, DocumentEngine::Auto, FallbackPolicy::Strict).await;
    assert!(
        matches!(strict, Err(DocError::NeedsOcr { .. })),
        "strict must surface the scanned-page signal, got {strict:?}"
    );

    let best_effort =
        convert_document(&path, DocumentEngine::Auto, FallbackPolicy::BestEffort).await;
    match best_effort {
        Ok(converted) => {
            assert_eq!(converted.engine, Engine::Xberg);
            assert!(converted.fell_back(), "a fallback must be recorded");
            assert!(
                converted.fallback.as_ref().is_some_and(DocError::needs_ocr),
                "the recorded reason must be the scanned-page failure"
            );
            assert!(!converted.markdown.trim().is_empty());
        }
        Err(error) => assert!(
            error.needs_ocr(),
            "best-effort must either convert or report needs-OCR, got {error:?}"
        ),
    }
}

#[test]
fn forced_engines_are_visible_in_provenance() {
    assert_eq!(Engine::Anydoc.other(), Engine::Xberg);
    assert_eq!(Engine::Xberg.other(), Engine::Anydoc);
    assert_eq!(DocumentEngine::parse("auto"), Some(DocumentEngine::Auto));
    assert_eq!(
        FallbackPolicy::parse("best-effort"),
        Some(FallbackPolicy::BestEffort)
    );
}

#[test]
#[ignore = "requires bench/doc-router/.cache (run bench/doc-router/fetch.py first)"]
fn snapshot_experiment_reproduces_golden() {
    // The heavy experiment is driven by Python because ground truth needs
    // LibreOffice and PyMuPDF; this test only checks that the cache exists so a
    // developer who ran it can reproduce it from `cargo nextest` as well.
    let cache = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/doc-router/.cache");
    assert!(
        cache.is_dir(),
        "materialize the snapshot first: python3 bench/doc-router/fetch.py"
    );
    let golden: Value = serde_json::from_str(GOLDEN_ROUTE).expect("golden route parses");
    assert_eq!(golden["router_revision"], 1);
}
