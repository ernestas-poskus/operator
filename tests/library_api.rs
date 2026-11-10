//! The in-process library API must work without a GPU, model download, or MCP host.
use operator::{
    Config, Error, LanguageId, Limit, LineNumber, Operator, PatternText, QueryText, RepoPath,
    SchemaVersion, WorkspaceConfig, detect_languages,
};

fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("src")).expect("src");
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\nfn main() {}\n",
    )
    .expect("source");
    dir
}

fn unindexed(dir: &tempfile::TempDir) -> Operator {
    let mut cfg = Config::new(dir.path().to_path_buf());
    cfg.languages = vec![LanguageId::parse("rust").expect("catalog id")];
    Operator::from_config_unindexed(cfg).expect("unindexed operator")
}

#[test]
fn detect_languages_sees_rust() {
    let dir = fixture();
    let languages = detect_languages(dir.path());
    assert!(
        languages
            .iter()
            .any(|lang| lang.id == "rust" && lang.files == 1)
    );
}

#[test]
fn open_unindexed_reads_the_workspace_config() {
    let dir = fixture();
    WorkspaceConfig {
        schema_version: Some(SchemaVersion::new(3)),
        languages: Some(vec![LanguageId::parse("rust").expect("catalog id")]),
        ..Default::default()
    }
    .save(dir.path())
    .expect("save config");
    let operator = Operator::open_unindexed(dir.path()).expect("open");
    assert!(!operator.has_index());
    assert_eq!(
        operator.config().languages,
        vec![LanguageId::parse("rust").expect("catalog id")]
    );
}

#[tokio::test]
async fn static_tools_work_without_an_index() {
    let dir = fixture();
    let operator = unindexed(&dir);

    let rust = LanguageId::parse("rust").expect("catalog id");
    let outline = operator
        .repo_map(None, Some(&rust), Limit::new(10))
        .expect("outline");
    assert!(outline.iter().any(|file| file.path.contains("src/lib.rs")));

    let hits = operator
        .search_code(
            &QueryText::from("pub fn add"),
            "literal",
            None,
            Some(&rust),
            Limit::new(10),
        )
        .await
        .expect("literal search");
    assert!(hits.iter().any(|hit| hit.path.contains("src/lib.rs")));

    let ast = operator
        .search_ast(
            Some(&PatternText::from("fn main() {}")),
            None,
            None,
            Some(&rust),
            Limit::new(10),
        )
        .expect("ast search");
    assert!(!ast.is_empty());

    let source = operator
        .read_source(
            &RepoPath::from("src/lib.rs"),
            Some(LineNumber::new(1)),
            Some(LineNumber::new(1)),
        )
        .expect("read");
    assert_eq!(source, "pub fn add(a: i32, b: i32) -> i32 { a + b }");
}

#[tokio::test]
async fn semantic_search_reports_the_missing_index() {
    let dir = fixture();
    let operator = unindexed(&dir);
    let error = operator
        .search_semantic(&QueryText::from("add"), None, None, Limit::new(5))
        .await
        .expect_err("semantic requires an index");
    assert!(matches!(error, Error::Tool(message) if message.contains("index_missing")));
}

#[tokio::test]
async fn status_is_available_without_an_index() {
    let dir = fixture();
    let operator = unindexed(&dir);
    let status = operator.index_status().await;
    assert!(status.rows.is_none());
}

#[test]
fn open_without_a_config_is_an_error() {
    let dir = fixture();
    let error = Operator::open_unindexed(dir.path()).expect_err("config is required");
    assert!(matches!(error, Error::Config(message) if message.contains("operator init")));
}

#[test]
fn operator_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Operator>();
}
