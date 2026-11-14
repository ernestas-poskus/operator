//! Local MCP tool-use journal and deterministic learning report.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cli::LearnArgs;
use crate::error::{Error, Result};
use crate::types::{
    ArgsHash, ArgsShape, CallCount, DurationMs, ErrorCount, ErrorMessage, JournalDir,
    LookbackWindow, PositionIndex, ResultBytes, ResultCount, SchemaVersion, SessionId, SignalCount,
    SignalKind, Success, Suggestion, ToolName, UnixSeconds,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEvent {
    pub version: SchemaVersion,
    /// Unix **seconds**: the historical JSONL unit, kept byte-identical.
    pub timestamp: UnixSeconds,
    pub session: SessionId,
    pub tool: ToolName,
    pub args_hash: ArgsHash,
    pub args_shape: ArgsShape,
    pub ok: Success,
    pub error: Option<ErrorMessage>,
    pub result_bytes: ResultBytes,
    pub result_count: ResultCount,
    pub duration_ms: DurationMs,
}

fn default_journal_dir(workspace: &Path) -> PathBuf {
    let root = std::env::var_os("CODE_INTELLIGENCE_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            home.join(".local/state/code-intelligence")
        });
    let identity = crate::config::repo_identity(workspace);
    let mut hasher = Sha256::new();
    hasher.update(identity.to_string_lossy().as_bytes());
    let hash = crate::hash::hex_digest(hasher.finalize());
    root.join(&hash[..16]).join("journal")
}

#[derive(Clone)]
pub struct Journal {
    path: PathBuf,
    session: SessionId,
}

impl Journal {
    pub fn open(workspace: &Path) -> Self {
        let dir = JournalDir::from(default_journal_dir(workspace));
        let _ = std::fs::create_dir_all(&dir);
        let day = time_day();
        let session = SessionId::new(format!("{:x}", std::process::id()));
        Self {
            path: dir.join(format!("{day}.jsonl")),
            session,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        tool: &ToolName,
        args: &serde_json::Value,
        ok: Success,
        error: Option<&ErrorMessage>,
        result_bytes: ResultBytes,
        result_count: ResultCount,
        duration: Duration,
    ) {
        let canonical = canonical_args(args);
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let event = JournalEvent {
            version: SchemaVersion::new(1),
            timestamp: UnixSeconds::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            ),
            session: self.session.clone(),
            tool: tool.clone(),
            args_hash: ArgsHash::new(crate::hash::hex_digest(hasher.finalize())),
            args_shape: ArgsShape::new(shape_of(args)),
            ok,
            error: error.cloned(),
            result_bytes,
            result_count,
            duration_ms: DurationMs::new(duration.as_millis() as u64),
        };
        if let Ok(line) = serde_json::to_string(&event) {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                let _ = writeln!(file, "{line}");
            }
        }
    }
}

fn canonical_args(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn shape_of(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{k}:{}", shape_name(v)))
                .collect();
            parts.sort();
            format!("{{{}}}", parts.join(","))
        }
        other => shape_name(other).to_string(),
    }
}

fn shape_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn time_day() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    format!("day-{days}")
}

fn parse_since(raw: &str) -> Duration {
    let raw = raw.trim();
    let digits: String = raw.chars().take_while(|c| c.is_ascii_digit()).collect();
    let value: u64 = digits.parse().unwrap_or(7);
    if raw.ends_with('h') {
        Duration::from_secs(value * 3600)
    } else {
        Duration::from_secs(value * 86_400)
    }
}

#[derive(Debug, Serialize)]
pub struct Signal {
    pub kind: SignalKind,
    pub tool: ToolName,
    pub count: SignalCount,
    pub first_index: PositionIndex,
    pub suggestion: Suggestion,
}

pub fn analyze(
    workspace: &Path,
    journal_dir: Option<&JournalDir>,
    since: &LookbackWindow,
) -> Result<Vec<Signal>> {
    let dir = journal_dir
        .map(|dir| dir.as_path().to_path_buf())
        .unwrap_or_else(|| default_journal_dir(workspace));
    let mut events = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            for line in std::fs::read_to_string(entry.path())?.lines() {
                if let Ok(event) = serde_json::from_str::<JournalEvent>(line) {
                    events.push(event);
                }
            }
        }
    }
    let _ = workspace;
    let _ = parse_since(since.as_str());
    let mut seen: HashMap<(ToolName, ArgsHash), usize> = HashMap::new();
    let mut signals = Vec::new();
    for (index, event) in events.iter().enumerate() {
        let key = (event.tool.clone(), event.args_hash.clone());
        if let Some(first) = seen.get(&key) {
            if index - *first <= 5 {
                signals.push(Signal {
                    kind: SignalKind::from("exact_repeat"),
                    tool: event.tool.clone(),
                    count: SignalCount::new(2),
                    first_index: PositionIndex::new(*first),
                    suggestion: Suggestion::from("The same call repeated within five calls; describe the tool as idempotent and return a cache hint."),
                });
            }
        } else {
            seen.insert(key, index);
        }
    }
    let mut errors: HashMap<String, usize> = HashMap::new();
    for event in &events {
        if !event.ok.is_on() {
            let key = format!("{}:{}", event.tool, event.error.clone().unwrap_or_default());
            *errors.entry(key).or_default() += 1;
        }
    }
    for (key, count) in errors {
        if count >= 3 {
            signals.push(Signal {
                kind: SignalKind::from("error_repeat"),
                tool: ToolName::from(key.split(':').next().unwrap_or_default()),
                count: SignalCount::new(count),
                first_index: PositionIndex::new(0),
                suggestion: Suggestion::from("A repeated error class suggests the argument shape or tool description is unclear."),
            });
        }
    }
    Ok(signals)
}

pub fn run(args: LearnArgs) -> Result<()> {
    let root = args
        .workspace_root
        .unwrap_or_else(|| crate::types::WorkspaceRoot::from("."));
    let workspace = std::fs::canonicalize(&root).unwrap_or_else(|_| root.into_path_buf());
    let signals = analyze(&workspace, args.journal_dir.as_ref(), &args.since)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&signals)?);
    } else if signals.is_empty() {
        println!("learn: no repeated or mismatched tool calls recorded");
    } else {
        for signal in signals {
            println!(
                "{}: {} (tool {}, occurrences {}, first {}) — {}",
                signal.kind,
                signal.tool,
                signal.tool,
                signal.count,
                signal.first_index,
                signal.suggestion
            );
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn _error(_: Error) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_records_keep_their_jsonl_shape() {
        let event = JournalEvent {
            version: SchemaVersion::new(1),
            timestamp: UnixSeconds::new(1_700_000_000),
            session: SessionId::from("s"),
            tool: ToolName::from("search_code"),
            args_hash: ArgsHash::from("h"),
            args_shape: ArgsShape::from("shape"),
            ok: Success::on(),
            error: None,
            result_bytes: ResultBytes::new(0),
            result_count: ResultCount::new(2),
            duration_ms: DurationMs::new(3),
        };
        let json = serde_json::to_value(&event).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "version": 1,
                "timestamp": 1_700_000_000u64,
                "session": "s",
                "tool": "search_code",
                "args_hash": "h",
                "args_shape": "shape",
                "ok": true,
                "error": null,
                "result_bytes": 0,
                "result_count": 2,
                "duration_ms": 3
            })
        );
    }

    #[test]
    fn repeated_calls_become_a_signal() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal {
            path: dir.path().join("day.jsonl"),
            session: SessionId::from("s"),
        };
        let args = serde_json::json!({"path": "a.go"});
        journal.record(
            &ToolName::from("read_source"),
            &args,
            Success::on(),
            None,
            ResultBytes::new(10),
            ResultCount::new(1),
            Duration::from_millis(1),
        );
        journal.record(
            &ToolName::from("read_source"),
            &args,
            Success::on(),
            None,
            ResultBytes::new(10),
            ResultCount::new(1),
            Duration::from_millis(1),
        );
        let signals = analyze(
            dir.path(),
            Some(&JournalDir::from(dir.path())),
            &LookbackWindow::from("7d"),
        )
        .unwrap();
        assert!(signals.iter().any(|s| s.kind == "exact_repeat"));
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolStat {
    pub tool: ToolName,
    pub calls: CallCount,
    pub errors: ErrorCount,
    pub p50_ms: DurationMs,
    pub p95_ms: DurationMs,
    pub bytes: ResultBytes,
    pub last_seen: UnixSeconds,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct JournalStats {
    pub calls: CallCount,
    pub errors: ErrorCount,
    pub bytes: ResultBytes,
    pub tools: Vec<ToolStat>,
    pub signals: SignalCount,
}

fn read_events(workspace: &Path, journal_dir: Option<&JournalDir>) -> Result<Vec<JournalEvent>> {
    let dir = journal_dir
        .map(|dir| dir.as_path().to_path_buf())
        .unwrap_or_else(|| default_journal_dir(workspace));
    let mut events = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            for line in std::fs::read_to_string(entry.path())?.lines() {
                if let Ok(event) = serde_json::from_str::<JournalEvent>(line) {
                    events.push(event);
                }
            }
        }
    }
    events.sort_by_key(|event| event.timestamp);
    Ok(events)
}

pub fn stats(workspace: &Path) -> Result<JournalStats> {
    let events = read_events(workspace, None)?;
    let signals = SignalCount::new(analyze(workspace, None, &LookbackWindow::from("30d"))?.len());
    let mut by_tool: std::collections::BTreeMap<ToolName, Vec<&JournalEvent>> = Default::default();
    for event in &events {
        by_tool.entry(event.tool.clone()).or_default().push(event);
    }
    let mut tools = Vec::new();
    for (tool, mut calls) in by_tool {
        calls.sort_by_key(|event| event.duration_ms);
        let p50 = calls
            .get(calls.len() / 2)
            .map(|e| e.duration_ms)
            .unwrap_or(DurationMs::new(0));
        let p95 = calls
            .get((calls.len() * 95 / 100).min(calls.len().saturating_sub(1)))
            .map(|e| e.duration_ms)
            .unwrap_or(DurationMs::new(0));
        tools.push(ToolStat {
            tool,
            calls: CallCount::new(calls.len()),
            errors: ErrorCount::new(calls.iter().filter(|e| !e.ok.is_on()).count()),
            p50_ms: p50,
            p95_ms: p95,
            bytes: calls.iter().map(|e| e.result_bytes).sum(),
            last_seen: calls.last().map(|e| e.timestamp).unwrap_or_default(),
        });
    }
    tools.sort_by_key(|stat| std::cmp::Reverse(stat.calls));
    Ok(JournalStats {
        calls: CallCount::new(events.len()),
        errors: ErrorCount::new(events.iter().filter(|e| !e.ok.is_on()).count()),
        bytes: events.iter().map(|e| e.result_bytes).sum(),
        tools,
        signals,
    })
}
