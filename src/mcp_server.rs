//! Stdio MCP server exposing the five code-intelligence tools.
use std::sync::Arc;
use std::time::Instant;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools;

#[derive(Clone)]
pub struct OperatorMcpServer {
    state: Arc<tools::ToolState>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RepoMapArgs {
    /// Optional path prefix relative to the workspace.
    #[serde(default)]
    pub path: Option<String>,
    /// Optional selected language id.
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchCodeArgs {
    /// Query text.
    pub query: String,
    /// `fts`, `literal` or `regex` (default `fts`).
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchAstArgs {
    /// ast-grep pattern (mutually exclusive with `rule`).
    #[serde(default)]
    pub pattern: Option<String>,
    /// ast-grep YAML rule (mutually exclusive with `pattern`).
    #[serde(default)]
    pub rule: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub paths: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchSemanticArgs {
    pub query: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadSourceArgs {
    pub path: String,
    #[serde(default)]
    pub start_line: Option<usize>,
    #[serde(default)]
    pub end_line: Option<usize>,
}

fn json_result<T: serde::Serialize>(value: &T) -> CallToolResult {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|e| format!("serialize: {e}"));
    CallToolResult::success(vec![ContentBlock::text(text)])
}

fn error_result(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(message.into())])
}

fn language_error(state: &tools::ToolState, language: &Option<String>) -> Option<String> {
    let language = language.as_deref()?;
    if state.cfg.languages.iter().any(|l| l == language) {
        None
    } else {
        Some(format!("language `{language}` is not selected"))
    }
}

#[tool_router]
impl OperatorMcpServer {
    #[tool(
        name = "repo_map",
        description = "Read-only ast-grep outline: files, symbols, imports and exports. Use for orientation before reading source."
    )]
    async fn repo_map(
        &self,
        Parameters(args): Parameters<RepoMapArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let started = Instant::now();
        if let Some(err) = language_error(&self.state, &args.language) {
            return Ok(error_result(err));
        }
        let limit = args.limit.unwrap_or(200).clamp(1, 1000);
        let result = tools::repo_map(
            &self.state,
            args.path.as_deref(),
            args.language.as_deref(),
            limit,
        );
        let out = match result {
            Ok(files) => {
                let json = serde_json::to_value(&files).unwrap_or_default();
                self.state.journal.record(
                    "repo_map",
                    &json,
                    true,
                    None,
                    0,
                    files.len(),
                    started.elapsed(),
                );
                json_result(&files)
            }
            Err(e) => {
                self.state.journal.record(
                    "repo_map",
                    &serde_json::json!({"path": args.path}),
                    false,
                    Some(&e),
                    0,
                    0,
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
        if let Some(err) = language_error(&self.state, &args.language) {
            return Ok(error_result(err));
        }
        let limit = args.limit.unwrap_or(20).clamp(1, 200);
        let mode = args.mode.as_deref().unwrap_or("fts");
        match tools::search_code(
            &self.state,
            &args.query,
            mode,
            args.path.as_deref(),
            args.language.as_deref(),
            limit,
        )
        .await
        {
            Ok(hits) => {
                self.state.journal.record(
                    "search_code",
                    &serde_json::json!({"query": args.query, "mode": mode}),
                    true,
                    None,
                    0,
                    hits.len(),
                    started.elapsed(),
                );
                Ok(json_result(&hits))
            }
            Err(e) => {
                self.state.journal.record(
                    "search_code",
                    &serde_json::json!({"query": args.query, "mode": mode}),
                    false,
                    Some(&e),
                    0,
                    0,
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
        if let Some(err) = language_error(&self.state, &args.language) {
            return Ok(error_result(err));
        }
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
        let limit = args.limit.unwrap_or(50).clamp(1, 500);
        match tools::search_ast(
            &self.state,
            args.pattern.as_deref(),
            args.rule.as_deref(),
            args.paths.as_deref(),
            args.language.as_deref(),
            limit,
        ) {
            Ok(matches) => {
                self.state.journal.record(
                    "search_ast",
                    &serde_json::json!({"pattern": args.pattern, "rule": args.rule.is_some()}),
                    true,
                    None,
                    0,
                    matches.len(),
                    started.elapsed(),
                );
                Ok(json_result(&matches))
            }
            Err(e) => {
                self.state.journal.record(
                    "search_ast",
                    &serde_json::json!({"error": "parse"}),
                    false,
                    Some(&e),
                    0,
                    0,
                    started.elapsed(),
                );
                Ok(error_result(e))
            }
        }
    }

    #[tool(
        name = "search_semantic",
        description = "Hybrid LanceDB dense + full-text search over the selected-language index. Requires `operator reindex`."
    )]
    async fn search_semantic(
        &self,
        Parameters(args): Parameters<SearchSemanticArgs>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let started = Instant::now();
        if let Some(err) = language_error(&self.state, &args.language) {
            return Ok(error_result(err));
        }
        let limit = args.limit.unwrap_or(20).clamp(1, 100);
        match tools::search_semantic(
            &self.state,
            &args.query,
            args.path.as_deref(),
            args.language.as_deref(),
            limit,
        )
        .await
        {
            Ok(hits) => {
                self.state.journal.record(
                    "search_semantic",
                    &serde_json::json!({"query": args.query}),
                    true,
                    None,
                    0,
                    hits.len(),
                    started.elapsed(),
                );
                Ok(json_result(&hits))
            }
            Err(e) => {
                self.state.journal.record(
                    "search_semantic",
                    &serde_json::json!({"query": args.query}),
                    false,
                    Some(&e),
                    0,
                    0,
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
                    "read_source",
                    &serde_json::json!({"path": args.path}),
                    true,
                    None,
                    text.len(),
                    1,
                    started.elapsed(),
                );
                Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
            }
            Err(e) => {
                self.state.journal.record(
                    "read_source",
                    &serde_json::json!({"path": args.path}),
                    false,
                    Some(&e),
                    0,
                    0,
                    started.elapsed(),
                );
                Ok(error_result(e))
            }
        }
    }
}

#[tool_handler(
    name = "operator",
    instructions = "Read-only code intelligence for a workspace: ast-grep structural search, outline, indexed text/semantic search, and source reads. Use the host's compiler/editor tools for type and reference semantics. No tool edits files."
)]
impl ServerHandler for OperatorMcpServer {}

impl OperatorMcpServer {
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
