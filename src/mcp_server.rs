//! Stdio MCP server exposing the five code-intelligence tools.
use std::sync::Arc;
use std::time::Instant;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools;
use crate::types::{
    ErrorMessage, LanguageFilter, LanguageId, Limit, LineNumber, PathFilter, PathsFilter,
    PatternText, QueryText, RepoPath, ResultBytes, ResultCount, RuleText, Success, ToolName,
};

#[derive(Clone)]
pub struct CodeIntelligenceMcpServer {
    state: Arc<tools::ToolState>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RepoMapArgs {
    /// Optional path prefix relative to the workspace.
    #[serde(default)]
    pub path: Option<PathFilter>,
    /// Optional selected language id.
    #[serde(default)]
    pub language: Option<LanguageFilter>,
    #[serde(default)]
    pub limit: Option<Limit>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchCodeArgs {
    /// Query text.
    pub query: QueryText,
    /// `fts`, `literal` or `regex` (default `fts`).
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub path: Option<PathFilter>,
    #[serde(default)]
    pub language: Option<LanguageFilter>,
    #[serde(default)]
    pub limit: Option<Limit>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchAstArgs {
    /// ast-grep pattern (mutually exclusive with `rule`).
    #[serde(default)]
    pub pattern: Option<PatternText>,
    /// ast-grep YAML rule (mutually exclusive with `pattern`).
    #[serde(default)]
    pub rule: Option<RuleText>,
    #[serde(default)]
    pub language: Option<LanguageFilter>,
    #[serde(default)]
    pub paths: Option<PathsFilter>,
    #[serde(default)]
    pub limit: Option<Limit>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchSemanticArgs {
    pub query: QueryText,
    #[serde(default)]
    pub path: Option<PathFilter>,
    #[serde(default)]
    pub language: Option<LanguageFilter>,
    #[serde(default)]
    pub limit: Option<Limit>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadSourceArgs {
    pub path: RepoPath,
    #[serde(default)]
    pub start_line: Option<LineNumber>,
    #[serde(default)]
    pub end_line: Option<LineNumber>,
}

fn json_result<T: serde::Serialize>(value: &T) -> CallToolResult {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|e| format!("serialize: {e}"));
    CallToolResult::success(vec![ContentBlock::text(text)])
}

fn error_result(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(message.into())])
}

fn selected_language(
    state: &tools::ToolState,
    filter: Option<&LanguageFilter>,
) -> Result<Option<LanguageId>, String> {
    let Some(filter) = filter else {
        return Ok(None);
    };
    if !state
        .cfg
        .languages
        .iter()
        .any(|l| l.as_str() == filter.as_str())
    {
        return Err(format!("language `{filter}` is not selected"));
    }
    LanguageId::parse(filter.as_str())
        .map(Some)
        .ok_or_else(|| format!("language `{filter}` is not selected"))
}

#[tool_router]
impl CodeIntelligenceMcpServer {
    #[tool(
        name = "repo_map",
        description = "Read-only ast-grep outline: files, symbols, imports and exports. Use for orientation before reading source."
    )]
    async fn repo_map(
        &self,
        Parameters(args): Parameters<RepoMapArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let started = Instant::now();
        let language = match selected_language(&self.state, args.language.as_ref()) {
            Ok(language) => language,
            Err(error) => return Ok(error_result(error)),
        };
        let limit = args.limit.unwrap_or(Limit::new(200)).clamp_to(1, 1000);
        let result = tools::repo_map(&self.state, args.path.as_ref(), language.as_ref(), limit);
        let out = match result {
            Ok(files) => {
                let json = serde_json::to_value(&files).unwrap_or_default();
                self.state.journal.record(
                    &ToolName::from("repo_map"),
                    &json,
                    Success::on(),
                    None,
                    ResultBytes::new(0),
                    ResultCount::new(files.len()),
                    started.elapsed(),
                );
                json_result(&files)
            }
            Err(e) => {
                self.state.journal.record(
                    &ToolName::from("repo_map"),
                    &serde_json::json!({"path": args.path}),
                    Success::off(),
                    Some(&ErrorMessage::from(e.as_str())),
                    ResultBytes::new(0),
                    ResultCount::new(0),
                    started.elapsed(),
                );
                error_result(e)
            }
        };
        Ok(out)
    }

    #[tool(
        name = "search_code",
        description = "Search workspace text with the indexed FTS, literal substring, or regex mode. Read-only; results are bounded."
    )]
    async fn search_code(
        &self,
        Parameters(args): Parameters<SearchCodeArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let started = Instant::now();
        let language = match selected_language(&self.state, args.language.as_ref()) {
            Ok(language) => language,
            Err(error) => return Ok(error_result(error)),
        };
        let limit = args.limit.unwrap_or(Limit::new(20)).clamp_to(1, 200);
        let mode = args.mode.as_deref().unwrap_or("fts");
        match tools::search_code(
            &self.state,
            &args.query,
            mode,
            args.path.as_ref(),
            language.as_ref(),
            limit,
        )
        .await
        {
            Ok(hits) => {
                self.state.journal.record(
                    &ToolName::from("search_code"),
                    &serde_json::json!({"query": args.query, "mode": mode}),
                    Success::on(),
                    None,
                    ResultBytes::new(0),
                    ResultCount::new(hits.len()),
                    started.elapsed(),
                );
                Ok(json_result(&hits))
            }
            Err(e) => {
                self.state.journal.record(
                    &ToolName::from("search_code"),
                    &serde_json::json!({"query": args.query, "mode": mode}),
                    Success::off(),
                    Some(&ErrorMessage::from(e.as_str())),
                    ResultBytes::new(0),
                    ResultCount::new(0),
                    started.elapsed(),
                );
                Ok(error_result(e))
            }
        }
    }

    #[tool(
        name = "search_ast",
        description = "Structural ast-grep search with a pattern or a YAML rule. Read-only: fix/rewrite rules are rejected."
    )]
    async fn search_ast(
        &self,
        Parameters(args): Parameters<SearchAstArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let started = Instant::now();
        let language = match selected_language(&self.state, args.language.as_ref()) {
            Ok(language) => language,
            Err(error) => return Ok(error_result(error)),
        };
        if args.pattern.is_some() == args.rule.is_some() {
            return Ok(error_result(
                "exactly one of `pattern` or `rule` is required",
            ));
        }
        if args
            .rule
            .as_ref()
            .is_some_and(|rule| rule.contains("fix:") || rule.contains("rewriters:"))
        {
            return Ok(error_result(
                "fix/rewriter rules are not allowed in the read-only server",
            ));
        }
        let limit = args.limit.unwrap_or(Limit::new(50)).clamp_to(1, 500);
        match tools::search_ast(
            &self.state,
            args.pattern.as_ref(),
            args.rule.as_ref(),
            args.paths.as_ref(),
            language.as_ref(),
            limit,
        ) {
            Ok(matches) => {
                self.state.journal.record(
                    &ToolName::from("search_ast"),
                    &serde_json::json!({"pattern": args.pattern, "rule": args.rule.is_some()}),
                    Success::on(),
                    None,
                    ResultBytes::new(0),
                    ResultCount::new(matches.len()),
                    started.elapsed(),
                );
                Ok(json_result(&matches))
            }
            Err(e) => {
                self.state.journal.record(
                    &ToolName::from("search_ast"),
                    &serde_json::json!({"error": "parse"}),
                    Success::off(),
                    Some(&ErrorMessage::from(e.as_str())),
                    ResultBytes::new(0),
                    ResultCount::new(0),
                    started.elapsed(),
                );
                Ok(error_result(e))
            }
        }
    }

    #[tool(
        name = "search_semantic",
        description = "Hybrid LanceDB dense + full-text search over the selected-language index. Requires `code-intelligence reindex`."
    )]
    async fn search_semantic(
        &self,
        Parameters(args): Parameters<SearchSemanticArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let started = Instant::now();
        let language = match selected_language(&self.state, args.language.as_ref()) {
            Ok(language) => language,
            Err(error) => return Ok(error_result(error)),
        };
        let limit = args.limit.unwrap_or(Limit::new(20)).clamp_to(1, 100);
        match tools::search_semantic(
            &self.state,
            &args.query,
            args.path.as_ref(),
            language.as_ref(),
            limit,
        )
        .await
        {
            Ok(hits) => {
                self.state.journal.record(
                    &ToolName::from("search_semantic"),
                    &serde_json::json!({"query": args.query}),
                    Success::on(),
                    None,
                    ResultBytes::new(0),
                    ResultCount::new(hits.len()),
                    started.elapsed(),
                );
                Ok(json_result(&hits))
            }
            Err(e) => {
                self.state.journal.record(
                    &ToolName::from("search_semantic"),
                    &serde_json::json!({"query": args.query}),
                    Success::off(),
                    Some(&ErrorMessage::from(e.as_str())),
                    ResultBytes::new(0),
                    ResultCount::new(0),
                    started.elapsed(),
                );
                Ok(error_result(e))
            }
        }
    }

    #[tool(
        name = "read_source",
        description = "Read a range of a workspace source file. Paths are confined to the workspace."
    )]
    async fn read_source(
        &self,
        Parameters(args): Parameters<ReadSourceArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let started = Instant::now();
        match tools::read_source(&self.state, &args.path, args.start_line, args.end_line) {
            Ok(text) => {
                self.state.journal.record(
                    &ToolName::from("read_source"),
                    &serde_json::json!({"path": args.path}),
                    Success::on(),
                    None,
                    ResultBytes::new(text.len()),
                    ResultCount::new(1),
                    started.elapsed(),
                );
                Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
            }
            Err(e) => {
                self.state.journal.record(
                    &ToolName::from("read_source"),
                    &serde_json::json!({"path": args.path}),
                    Success::off(),
                    Some(&ErrorMessage::from(e.as_str())),
                    ResultBytes::new(0),
                    ResultCount::new(0),
                    started.elapsed(),
                );
                Ok(error_result(e))
            }
        }
    }
}

#[tool_handler(
    name = "code-intelligence",
    instructions = "Read-only code intelligence for a workspace: ast-grep structural search, outline, indexed text/semantic search, and source reads. Use the host's compiler/editor tools for type and reference semantics. No tool edits files."
)]
impl ServerHandler for CodeIntelligenceMcpServer {}

impl CodeIntelligenceMcpServer {
    pub fn new(state: Arc<tools::ToolState>) -> Self {
        Self { state }
    }

    pub async fn serve_stdio(self) -> crate::error::Result<()> {
        let (stdin, stdout) = rmcp::transport::io::stdio();
        let running = self
            .serve((stdin, stdout))
            .await
            .map_err(|e| crate::error::Error::Msg(format!("mcp server: {e}")))?;
        running
            .waiting()
            .await
            .map_err(|e| crate::error::Error::Msg(format!("mcp server: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_arguments_stay_plain_json() {
        let args: RepoMapArgs = serde_json::from_value(serde_json::json!({
            "path": "src",
            "language": "rust",
            "limit": 5
        }))
        .expect("typed args");
        assert_eq!(args.path.as_ref().map(PathFilter::as_str), Some("src"));
        assert_eq!(
            args.language.as_ref().map(LanguageFilter::as_str),
            Some("rust")
        );
        assert_eq!(args.limit.map(Limit::get), Some(5));

        // An unknown language must survive deserialization so the dispatcher can
        // return the established `language ... is not selected` tool error
        // instead of a JSON-RPC invalid-params failure.
        let args: RepoMapArgs =
            serde_json::from_value(serde_json::json!({"language": "nope"})).expect("raw filter");
        assert_eq!(
            args.language.as_ref().map(LanguageFilter::as_str),
            Some("nope")
        );
    }

    #[test]
    fn host_argument_schemas_stay_primitives() {
        let schema = serde_json::to_value(schemars::schema_for!(RepoMapArgs)).expect("schema");
        let language = schema["properties"]["language"].to_string();
        assert!(language.contains("string"), "{language}");
        assert!(!language.contains("LanguageFilter"), "{language}");
        let limit = schema["properties"]["limit"].to_string();
        assert!(limit.contains("integer"), "{limit}");
    }
}
