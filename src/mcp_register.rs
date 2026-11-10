//! Project-local MCP host registrations for the `operator` server.
//!
//! Five hosts read a project-local file: Claude Code's `.mcp.json`, Gemini
//! CLI's `.gemini/settings.json`, Codex CLI's `.codex/config.toml`, Cursor's
//! `.cursor/mcp.json`, and Antigravity CLI's (Agy's) `.agents/mcp_config.json`.
//! This module writes exactly one entry in each — the `operator` server — and
//! **merges** rather than replaces: every other server, setting and TOML comment
//! in the file survives untouched.
//!
//! Two transports reach the same server:
//!
//! - [`Transport::Stdio`] launches `operator serve …`, which attaches to a live
//!   instance when `.operator/serve.json` names one (see [`crate::mcp_attach`]);
//! - [`Transport::Http`] points the host at a running instance's Streamable
//!   HTTP endpoint, so the host skips the child process entirely.
//!
//! `operator init` creates the files when absent ([`Mode::CreateOnly`]);
//! `operator serve` reconciles its entry on every start
//! ([`Mode::Reconcile`]), which is how a running instance comes to be what the
//! host files name.
//!
//! Writes are atomic (temp file plus rename), so a host reading the file while
//! `serve` reconciles it never sees a half-written document.

use crate::error::{Error, Result};
use crate::types::{Argv, HostFile, UpdatedAt};
use std::path::Path;

/// How a host should reach the operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    /// Launch `operator <argv…>` over stdio. `argv[0]` is the subcommand
    /// (`serve`), the rest are its flags.
    Stdio {
        /// The argv after the command name.
        argv: Argv,
    },
    /// Connect to a running instance over Streamable HTTP (`…/mcp`).
    #[allow(dead_code)]
    Http {
        /// The endpoint URL a host connects to.
        url: String,
    },
}

/// What to do with a file that already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Write only when the file is absent (`operator init`).
    CreateOnly,
    /// Merge and refresh our entry; leave an unreadable file untouched.
    Reconcile,
    /// Rewrite the whole file with only our entry (`--force` / `--force-register`).
    ReplaceFile,
}

/// What a registration pass did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registered {
    /// Workspace-relative files written or created, in registration order.
    pub files: Vec<HostFile>,
    /// Unix seconds when the pass finished.
    pub updated_at: UpdatedAt,
}

/// The `operator` server name in every host file.
pub const SERVER: &str = "operator";

/// The command every stdio registration launches.
pub const COMMAND: &str = "operator";

/// One host's registration file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Host {
    /// Claude Code: `.mcp.json`, `{"mcpServers": {...}}`.
    Claude,
    /// Gemini CLI: `.gemini/settings.json`, `{"mcpServers": {...}, ...}`.
    Gemini,
    /// Codex CLI: `.codex/config.toml`, `[mcp_servers.operator]`.
    Codex,
    /// Cursor: `.cursor/mcp.json`, `{"mcpServers": {...}}`.
    Cursor,
    /// Antigravity CLI (`agy`): `.agents/mcp_config.json`,
    /// `{"mcpServers": {...}}`; HTTP uses `serverUrl`.
    Agy,
}

/// The host files this deployment registers, in registration order.
const HOSTS: [(&str, Host); 5] = [
    (".mcp.json", Host::Claude),
    (".gemini/settings.json", Host::Gemini),
    (".codex/config.toml", Host::Codex),
    (".cursor/mcp.json", Host::Cursor),
    (".agents/mcp_config.json", Host::Agy),
];

fn now_unix() -> i64 {
    time::OffsetDateTime::now_utc().unix_timestamp()
}

/// Write or refresh the `operator` entry in every host file.
///
/// `log_prefix` names the caller in the stderr lines (`init`, `serve`).
/// Returns the files actually written; an untouched (or skipped) file is not
/// in the list.
pub fn register_operator(
    workspace: &Path,
    transport: &Transport,
    mode: Mode,
    log_prefix: &str,
) -> Result<Registered> {
    let mut files = Vec::new();
    for (rel, host) in HOSTS {
        let path = workspace.join(rel);
        let changed = match host {
            Host::Codex => write_codex(&path, transport, mode, log_prefix)?,
            Host::Claude | Host::Gemini | Host::Cursor | Host::Agy => {
                write_json(&path, host, transport, mode, log_prefix)?
            }
        };
        if changed {
            files.push(HostFile::from(rel));
        }
    }
    Ok(Registered {
        files,
        updated_at: UpdatedAt::new(now_unix()),
    })
}

/// The one entry a host reads, as JSON (the four JSON hosts).
fn json_entry(host: Host, transport: &Transport) -> serde_json::Value {
    match transport {
        Transport::Stdio { argv } => {
            serde_json::json!({ "command": COMMAND, "args": argv })
        }
        Transport::Http { url } => match host {
            // Claude Code reads a URL entry only when it carries `type`; an
            // entry with a url and no type is a startup error there.
            Host::Claude => serde_json::json!({ "type": "http", "url": url }),
            // Gemini CLI's consolidated remote shape (`httpUrl` is deprecated in
            // favour of `url` + `type`).
            Host::Gemini => serde_json::json!({ "url": url, "type": "http" }),
            // Cursor's remote MCP entry names the URL directly.
            Host::Cursor => serde_json::json!({ "url": url }),
            // Antigravity reads HTTP MCP entries as `serverUrl`.
            Host::Agy => serde_json::json!({ "serverUrl": url }),
            // Codex is TOML, never this shape; kept exhaustive rather than
            // panicking so the mapping stays total.
            Host::Codex => serde_json::json!({ "url": url }),
        },
    }
}

/// The whole file written when there is nothing to merge (or on ReplaceFile).
fn fresh_json(host: Host, transport: &Transport) -> Result<String> {
    let doc = serde_json::json!({ "mcpServers": { SERVER: json_entry(host, transport) } });
    serde_json::to_string_pretty(&doc)
        .map(|mut text| {
            text.push('\n');
            text
        })
        .map_err(|e| Error::msg(format!("serializing an MCP registration failed: {e}")))
}

/// Merge our entry into an existing JSON host file.
fn write_json(
    path: &Path,
    host: Host,
    transport: &Transport,
    mode: Mode,
    log_prefix: &str,
) -> Result<bool> {
    let target = json_entry(host, transport);
    if !path.exists() {
        return write_atomic(path, &fresh_json(host, transport)?, log_prefix);
    }
    if mode == Mode::CreateOnly {
        announce_kept(log_prefix, path);
        return Ok(false);
    }
    if mode == Mode::ReplaceFile {
        return write_atomic(path, &fresh_json(host, transport)?, log_prefix);
    }

    let text = read(path)?;
    let Ok(mut doc) = serde_json::from_str::<serde_json::Value>(&text) else {
        announce_unreadable(log_prefix, path);
        return Ok(false);
    };
    let Some(root) = doc.as_object_mut() else {
        announce_unreadable(log_prefix, path);
        return Ok(false);
    };
    let servers = root
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    let Some(servers) = servers.as_object_mut() else {
        eprintln!(
            "{log_prefix}: {} has a non-object `mcpServers` — leaving it",
            path.display()
        );
        return Ok(false);
    };
    if servers.get(SERVER) == Some(&target) {
        return Ok(false);
    }
    servers.insert(SERVER.to_string(), target);
    let text = serde_json::to_string_pretty(&doc)
        .map_err(|e| Error::msg(format!("serializing {} failed: {e}", path.display())))?;
    write_atomic(path, &format!("{text}\n"), log_prefix)
}

/// The whole file written when there is nothing to merge (or on ReplaceFile).
fn fresh_codex(transport: &Transport) -> String {
    let mut body = String::new();
    match transport {
        Transport::Http { url } => {
            body.push_str(&format!("url = {}\n", toml_string(url)));
        }
        Transport::Stdio { argv } => {
            body.push_str(&format!("command = {}\n", toml_string(COMMAND)));
            let args = argv
                .iter()
                .map(|a| toml_string(a.as_str()))
                .collect::<Vec<_>>()
                .join(", ");
            body.push_str(&format!("args = [{args}]\n"));
        }
    }
    format!(
        "# Project-local MCP servers for Codex CLI (user scope: ~/.codex/config.toml).\n\
         [mcp_servers.operator]\n{body}"
    )
}

/// Merge our entry into an existing `.codex/config.toml`, preserving comments
/// and every other table.
fn write_codex(path: &Path, transport: &Transport, mode: Mode, log_prefix: &str) -> Result<bool> {
    if !path.exists() {
        return write_atomic(path, &fresh_codex(transport), log_prefix);
    }
    if mode == Mode::CreateOnly {
        announce_kept(log_prefix, path);
        return Ok(false);
    }
    if mode == Mode::ReplaceFile {
        return write_atomic(path, &fresh_codex(transport), log_prefix);
    }

    let text = read(path)?;
    let Ok(mut doc) = text.parse::<toml_edit::DocumentMut>() else {
        announce_unreadable(log_prefix, path);
        return Ok(false);
    };
    let servers = doc
        .as_table_mut()
        .entry("mcp_servers")
        .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
    let Some(servers) = servers.as_table_mut() else {
        eprintln!(
            "{log_prefix}: {} has a non-table `mcp_servers` — leaving it",
            path.display()
        );
        return Ok(false);
    };
    let entry = servers
        .entry(SERVER)
        .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
    let Some(table) = entry.as_table_mut() else {
        eprintln!(
            "{log_prefix}: {} has a non-table `mcp_servers.{SERVER}` — leaving it",
            path.display()
        );
        return Ok(false);
    };
    match transport {
        Transport::Http { url } => {
            table.insert("url", toml_edit::value(url.as_str()));
            // The shape flipped: a stale stdio command must not linger next to
            // the URL, or the host has two contradictory transports.
            table.remove("command");
            table.remove("args");
        }
        Transport::Stdio { argv } => {
            table.insert("command", toml_edit::value(COMMAND));
            let mut args = toml_edit::Array::new();
            for arg in argv {
                args.push(arg.as_str());
            }
            table.insert("args", toml_edit::value(args));
            table.remove("url");
        }
    }
    let edited = doc.to_string();
    if edited == text {
        return Ok(false);
    }
    write_atomic(path, &edited, log_prefix)
}

/// A TOML basic-string literal for `s`.
fn toml_string(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .map_err(|e| Error::msg(format!("reading {} failed: {e}", path.display())))
}

/// Write `text` to `path` atomically (temp file plus rename), creating parents.
fn write_atomic(path: &Path, text: &str, log_prefix: &str) -> Result<bool> {
    let parent = path.parent().ok_or_else(|| {
        Error::msg(format!(
            "registration path has no parent: {}",
            path.display()
        ))
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| Error::msg(format!("creating {} failed: {e}", parent.display())))?;
    let tmp = path.with_extension("tmp.operator");
    std::fs::write(&tmp, text)
        .map_err(|e| Error::msg(format!("writing {} failed: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| Error::msg(format!("renaming {} failed: {e}", tmp.display())))?;
    eprintln!("{log_prefix}: wrote {}", path.display());
    Ok(true)
}

fn announce_kept(log_prefix: &str, path: &Path) {
    eprintln!(
        "{log_prefix}: {} already exists — leaving it (use --force to rewrite)",
        path.display()
    );
}

fn announce_unreadable(log_prefix: &str, path: &Path) {
    eprintln!(
        "{log_prefix}: {} is not a document the operator can edit — leaving it \
         (use --force-register to rewrite it)",
        path.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CommandArg;

    fn http() -> Transport {
        Transport::Http {
            url: "http://127.0.0.1:3921/mcp".to_string(),
        }
    }

    fn stdio() -> Transport {
        Transport::Stdio {
            argv: Argv::new(vec![
                CommandArg::from("serve"),
                CommandArg::from("--planner-model"),
                CommandArg::from("x"),
            ]),
        }
    }

    #[test]
    fn a_fresh_file_is_created_for_every_host() {
        let dir = tempfile::tempdir().unwrap();
        let done = register_operator(dir.path(), &http(), Mode::CreateOnly, "test").unwrap();
        assert_eq!(
            done.files,
            vec![
                ".mcp.json",
                ".gemini/settings.json",
                ".codex/config.toml",
                ".cursor/mcp.json",
                ".agents/mcp_config.json"
            ]
        );
        let claude: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap())
                .unwrap();
        assert_eq!(
            claude["mcpServers"]["operator"],
            serde_json::json!({"type": "http", "url": "http://127.0.0.1:3921/mcp"})
        );
        let gemini: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(".gemini/settings.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            gemini["mcpServers"]["operator"],
            serde_json::json!({"url": "http://127.0.0.1:3921/mcp", "type": "http"})
        );
        let codex = std::fs::read_to_string(dir.path().join(".codex/config.toml")).unwrap();
        assert!(codex.contains("[mcp_servers.operator]"), "{codex}");
        assert!(
            codex.contains(r#"url = "http://127.0.0.1:3921/mcp""#),
            "{codex}"
        );
        let cursor: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(".cursor/mcp.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            cursor["mcpServers"]["operator"],
            serde_json::json!({"url": "http://127.0.0.1:3921/mcp"})
        );
        let agy: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(".agents/mcp_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            agy["mcpServers"]["operator"],
            serde_json::json!({"serverUrl": "http://127.0.0.1:3921/mcp"})
        );
        // No temp file is left behind.
        assert!(!dir.path().join(".mcp.tmp.operator").exists());
    }

    #[test]
    fn reconcile_preserves_other_servers_and_settings() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(".mcp.json"),
            r#"{"mcpServers":{"other":{"command":"other"}},"extra":true}"#,
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join(".gemini")).unwrap();
        std::fs::write(
            dir.path().join(".gemini/settings.json"),
            r#"{"mcpServers":{"other":{"command":"other"}},"ui":{"theme":"dark"},"security":{"x":1}}"#,
        )
        .unwrap();

        register_operator(dir.path(), &stdio(), Mode::Reconcile, "test").unwrap();

        let claude: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap())
                .unwrap();
        assert_eq!(claude["mcpServers"]["other"]["command"], "other");
        assert_eq!(claude["extra"], true);
        assert_eq!(
            claude["mcpServers"]["operator"],
            serde_json::json!({"command": "operator", "args": ["serve", "--planner-model", "x"]})
        );
        let gemini: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(".gemini/settings.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(gemini["ui"]["theme"], "dark");
        assert_eq!(gemini["security"]["x"], 1);
        assert_eq!(gemini["mcpServers"]["other"]["command"], "other");
    }

    #[test]
    fn codex_reconcile_preserves_other_tables_and_comments_and_flips_shape() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".codex")).unwrap();
        let path = dir.path().join(".codex/config.toml");
        std::fs::write(
            &path,
            "# my codex settings\nmodel = \"gpt-5.6-sol\"\n\
             [mcp_servers.other]\ncommand = \"other\"\n\n\
             [mcp_servers.operator]\ncommand = \"operator\"\nargs = [\"serve\"]\n\n\
             [projects.\"/w\"]\ntrust_level = \"trusted\"\n",
        )
        .unwrap();

        register_operator(dir.path(), &http(), Mode::Reconcile, "test").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# my codex settings"), "{text}");
        assert!(text.contains("[mcp_servers.other]"), "{text}");
        assert!(text.contains("trust_level = \"trusted\""), "{text}");
        assert!(
            text.contains(r#"url = "http://127.0.0.1:3921/mcp""#),
            "{text}"
        );
        assert!(
            !text.contains(r#"command = "operator""#),
            "the stale stdio command must be gone: {text}"
        );
        assert!(!text.contains("args = "), "{text}");

        // And back to stdio: the URL must not linger.
        register_operator(dir.path(), &stdio(), Mode::Reconcile, "test").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(r#"command = "operator""#), "{text}");
        assert!(
            text.contains(r#"args = ["serve", "--planner-model", "x"]"#),
            "{text}"
        );
        assert!(!text.contains("url = "), "{text}");
    }

    #[test]
    fn reconcile_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        register_operator(dir.path(), &http(), Mode::Reconcile, "test").unwrap();
        let claude = std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap();
        let codex = std::fs::read_to_string(dir.path().join(".codex/config.toml")).unwrap();
        let again = register_operator(dir.path(), &http(), Mode::Reconcile, "test").unwrap();
        assert!(
            again.files.is_empty(),
            "no file should be rewritten: {:?}",
            again.files
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap(),
            claude
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join(".codex/config.toml")).unwrap(),
            codex
        );
    }

    #[test]
    fn create_only_leaves_existing_files_alone() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(".mcp.json"),
            r#"{"mcpServers":{"operator":{"command":"old"}}}"#,
        )
        .unwrap();
        let done = register_operator(dir.path(), &http(), Mode::CreateOnly, "test").unwrap();
        assert!(
            !done.files.contains(&HostFile::from(".mcp.json")),
            "{:?}",
            done.files
        );
        let claude: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap())
                .unwrap();
        assert_eq!(claude["mcpServers"]["operator"]["command"], "old");
    }

    #[test]
    fn an_unreadable_file_is_left_alone_unless_forced() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".mcp.json"), "{ not json").unwrap();
        register_operator(dir.path(), &http(), Mode::Reconcile, "test").unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap(),
            "{ not json"
        );
        register_operator(dir.path(), &http(), Mode::ReplaceFile, "test").unwrap();
        let claude: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap())
                .unwrap();
        assert_eq!(claude["mcpServers"]["operator"]["type"], "http");
    }

    #[test]
    fn replace_file_drops_other_servers() {
        // The documented `--force-register` escape hatch: a whole-file rewrite.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(".mcp.json"),
            r#"{"mcpServers":{"other":{"command":"other"}}}"#,
        )
        .unwrap();
        register_operator(dir.path(), &http(), Mode::ReplaceFile, "test").unwrap();
        let claude: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(".mcp.json")).unwrap())
                .unwrap();
        assert!(claude["mcpServers"].get("other").is_none(), "{claude}");
    }

    #[test]
    fn toml_strings_escape_quotes_and_backslashes() {
        assert_eq!(toml_string(r#"a"b\c"#), r#""a\"b\\c""#);
    }
}
