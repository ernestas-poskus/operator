//! CLI surface: init, serve, reindex, learn.
use argh::FromArgs;
use std::path::PathBuf;

#[derive(FromArgs, Debug)]
/// Read-only code-intelligence MCP server for coding agents.
pub struct OperatorCli {
    #[argh(subcommand)]
    pub command: Command,
}

#[derive(FromArgs, Debug)]
#[argh(subcommand)]
pub enum Command {
    Init(InitArgs),
    #[cfg(feature = "tui")]
    Tui(TuiArgs),
    Serve(ServeArgs),
    Reindex(ReindexArgs),
    Learn(LearnArgs),
}

#[derive(FromArgs, Debug)]
#[argh(subcommand, name = "init")]
/// Detect supported languages and write the workspace selection.
pub struct InitArgs {
    #[argh(positional)]
    pub workspace_root: Option<PathBuf>,
    /// overwrite an existing selection when prompted
    #[argh(switch)]
    pub force: bool,
    /// accept detected languages without prompting
    #[argh(switch)]
    pub yes: bool,
    #[argh(option)]
    /// comma-separated canonical language ids; default: detected
    pub languages: Option<String>,
    /// select every supported language
    #[argh(switch)]
    pub all_supported: bool,
    /// comma-separated workspace-relative include paths
    #[argh(option)]
    pub include: Option<String>,
    /// comma-separated workspace-relative exclude paths
    #[argh(option)]
    pub exclude: Option<String>,
    /// force the terminal UI even when stdin is redirected
    #[argh(switch)]
    pub tui: bool,
    /// never launch the terminal UI
    #[cfg_attr(not(feature = "tui"), allow(dead_code))]
    #[argh(switch)]
    pub no_tui: bool,
}

#[cfg(feature = "tui")]
#[derive(FromArgs, Debug)]
#[argh(subcommand, name = "tui")]
/// Open the setup/dashboard TUI for a configured workspace.
pub struct TuiArgs {
    #[argh(positional)]
    pub workspace_root: Option<PathBuf>,
}

#[derive(FromArgs, Debug)]
#[argh(subcommand, name = "serve")]
/// Serve the five code-intelligence tools over stdio MCP.
pub struct ServeArgs {
    #[argh(positional)]
    pub workspace_root: Option<PathBuf>,
    /// do not rewrite host MCP registration files
    #[argh(switch)]
    pub no_register: bool,
}

#[derive(FromArgs, Debug)]
#[argh(subcommand, name = "reindex")]
/// Refresh the selected-language search index.
pub struct ReindexArgs {
    #[argh(positional)]
    pub workspace_root: Option<PathBuf>,
    /// drop the current index before rebuilding
    #[argh(switch)]
    pub rebuild: bool,
    /// report index identity without building
    #[argh(switch)]
    pub status: bool,
    /// emit machine-readable JSON
    #[argh(switch)]
    pub json: bool,
    /// show terminal progress
    #[argh(switch)]
    pub tui: bool,
}

#[derive(FromArgs, Debug)]
#[argh(subcommand, name = "learn")]
/// Report repeated/mismatched MCP tool usage from the local journal.
pub struct LearnArgs {
    #[argh(positional)]
    pub workspace_root: Option<PathBuf>,
    /// emit machine-readable JSON
    #[argh(switch)]
    pub json: bool,
    /// lookback window, e.g. 7d or 24h
    #[argh(option, default = "String::from(\"7d\")")]
    pub since: String,
    /// override the journal directory
    #[argh(option)]
    pub journal_dir: Option<PathBuf>,
}
