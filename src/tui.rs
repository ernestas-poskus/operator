//! Terminal setup/dashboard for code-intelligence operator.
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Tabs};
use ratatui::{Frame, Terminal};

use crate::cli::{InitArgs, TuiArgs};
use crate::code_search::{IndexEvent, RetrievalIndexStatus};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::languages;
use crate::workspace_config::WorkspaceConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Sources,
    Index,
    Stats,
    Usage,
    Review,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selection {
    Include,
    Exclude,
}

#[derive(Debug, Clone)]
struct Node {
    path: PathBuf,
    rel: String,
    name: String,
    is_dir: bool,
    depth: usize,
    parent: Option<PathBuf>,
    language: Option<String>,
    files: usize,
}

struct App {
    workspace: PathBuf,
    mode: &'static str,
    tab: Tab,
    trees: Vec<Node>,
    expanded: HashSet<PathBuf>,
    tree_index: usize,
    tree_state: ListState,
    focus: bool,
    lang_index: usize,
    lang_state: ListState,
    selected_languages: BTreeSet<String>,
    base: WorkspaceConfig,
    detected: Vec<languages::DetectedLanguage>,
    explicit: HashMap<String, Selection>,
    search: String,
    search_mode: bool,
    help: bool,
    dirty: bool,
    quit_armed: bool,
    should_quit: bool,
    status: Option<RetrievalIndexStatus>,
    status_rx: Option<tokio::sync::mpsc::UnboundedReceiver<RetrievalIndexStatus>>,
    progress: Option<IndexEvent>,
    log: Vec<String>,
    new_events: usize,
    index_rx: Option<tokio::sync::mpsc::UnboundedReceiver<IndexEvent>>,
    _watcher: Option<RecommendedWatcher>,
    watch_rx: Option<Receiver<notify::Result<notify::Event>>>,
    stats: crate::learn::JournalStats,
    stats_at: Instant,
    started: Instant,
}

impl App {
    fn new(
        workspace: PathBuf,
        mode: &'static str,
        detected: Vec<languages::DetectedLanguage>,
    ) -> Self {
        let selected_languages = detected
            .iter()
            .map(|d| languages::canonical_id(d.lang).to_string())
            .collect();
        let trees = build_tree(&workspace);
        let mut expanded = HashSet::new();
        for node in &trees {
            if node.is_dir && node.depth == 0 {
                expanded.insert(node.path.clone());
            }
        }
        let (watch_tx, watch_rx) = std::sync::mpsc::channel();
        let watcher = notify::recommended_watcher(move |res| {
            let _ = watch_tx.send(res);
        })
        .ok()
        .and_then(|mut watcher| {
            watcher
                .watch(&workspace, RecursiveMode::Recursive)
                .ok()
                .map(|_| watcher)
        });
        let base = WorkspaceConfig::load(&workspace)
            .ok()
            .flatten()
            .unwrap_or_default();
        let mut app = Self {
            workspace,
            mode,
            tab: Tab::Sources,
            trees,
            expanded,
            tree_index: 0,
            tree_state: ListState::default(),
            focus: false,
            lang_index: 0,
            lang_state: ListState::default(),
            selected_languages,
            base,
            detected,
            explicit: HashMap::new(),
            search: String::new(),
            search_mode: false,
            help: false,
            dirty: false,
            quit_armed: false,
            should_quit: false,
            status: None,
            status_rx: None,
            progress: None,
            log: Vec::new(),
            new_events: 0,
            index_rx: None,
            _watcher: watcher,
            watch_rx: Some(watch_rx),
            stats: Default::default(),
            stats_at: Instant::now() - Duration::from_secs(10),
            started: Instant::now(),
        };
        app.prefill();
        app
    }

    fn prefill(&mut self) {
        if let Ok(Some(cfg)) = WorkspaceConfig::load(&self.workspace) {
            if let Some(langs) = cfg.languages {
                let chosen: BTreeSet<String> = langs
                    .into_iter()
                    .filter(|l| self.selected_languages.contains(l))
                    .collect();
                if !chosen.is_empty() {
                    self.selected_languages = chosen;
                }
            }
            for path in cfg.retrieval_include.unwrap_or_default() {
                self.explicit.insert(path, Selection::Include);
            }
            for path in cfg.retrieval_exclude.unwrap_or_default() {
                self.explicit.insert(path, Selection::Exclude);
            }
        }
    }

    fn config(&self) -> WorkspaceConfig {
        let mut includes = Vec::new();
        let mut excludes = Vec::new();
        for (path, state) in &self.explicit {
            match state {
                Selection::Include => includes.push(path.clone()),
                Selection::Exclude => excludes.push(path.clone()),
            }
        }
        includes.sort();
        excludes.sort();
        let mut base = self.base.clone();
        base.schema_version = Some(3);
        base.languages = Some(self.selected_languages.iter().cloned().collect());
        base.retrieval_include = Some(includes);
        base.retrieval_exclude = Some(excludes);
        base
    }

    fn config_full(&self) -> Result<Config> {
        let root = std::fs::canonicalize(&self.workspace)
            .map_err(|e| Error::Config(format!("workspace: {e}")))?;
        for path in self.includes().into_iter().chain(self.excludes()) {
            let candidate = root.join(&path);
            let canonical = std::fs::canonicalize(&candidate)
                .map_err(|e| Error::Config(format!("{path}: {e}")))?;
            if !canonical.starts_with(&root) {
                return Err(Error::Config(format!("{path}: outside workspace")));
            }
        }
        if self.selected_languages.is_empty() {
            return Err(Error::Config("select at least one language".into()));
        }
        let mut cfg = self.config().into_config(&self.workspace);
        cfg.languages = self.selected_languages.iter().cloned().collect();
        cfg.retrieval_include = self.config().retrieval_include.unwrap_or_default();
        cfg.retrieval_exclude = self.config().retrieval_exclude.unwrap_or_default();
        Ok(cfg)
    }

    fn save(&mut self) -> Result<()> {
        self.config_full()?;
        let ws = self.config();
        ws.save(&self.workspace)?;
        let transport = crate::mcp_register::Transport::Stdio {
            argv: vec!["serve".into()],
        };
        let _ = crate::mcp_register::register_operator(
            &self.workspace,
            &transport,
            crate::mcp_register::Mode::Reconcile,
            "tui",
        );
        self.dirty = false;
        self.push_log("saved .operator.toml".into());
        Ok(())
    }

    fn start_index(&mut self, rebuild: bool) {
        let cfg = match self.config_full() {
            Ok(cfg) => cfg,
            Err(error) => {
                self.push_log(format!("index: {error}"));
                return;
            }
        };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.index_rx = Some(rx);
        self.progress = None;
        self.new_events = 0;
        self.tab = Tab::Index;
        let tx2 = tx.clone();
        tokio::spawn(async move {
            if rebuild && let Err(error) = crate::code_search::drop_retrieval_index(&cfg).await {
                let _ = tx2.send(IndexEvent::Failed { reason: error });
                return;
            }
            if crate::code_search::LocalCodeIndex::build_with_events(&cfg, tx2.clone())
                .await
                .is_none()
            {
                let _ = tx2.send(IndexEvent::Failed {
                    reason: "index build failed (no files, model, or storage error)".into(),
                });
            }
        });
        self.push_log(if rebuild {
            "index: rebuild started".into()
        } else {
            "index: started".into()
        });
    }

    fn refresh_stats(&mut self) {
        if self.stats_at.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.stats_at = Instant::now();
        if let Ok(stats) = crate::learn::stats(&self.workspace) {
            self.stats = stats;
        }
    }

    fn refresh_status(&mut self) {
        let cfg = match self.config_full() {
            Ok(cfg) => cfg,
            Err(_) => return,
        };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.status_rx = Some(rx);
        tokio::spawn(async move {
            let status = crate::code_search::retrieval_index_status(&cfg).await;
            let _ = tx.send(status);
        });
    }

    fn tick(&mut self) {
        self.refresh_stats();
        let mut statuses = Vec::new();
        if let Some(rx) = self.status_rx.as_mut() {
            while let Ok(status) = rx.try_recv() {
                statuses.push(status);
            }
        }
        for status in statuses {
            self.status = Some(status);
        }
        let mut events = Vec::new();
        if let Some(rx) = self.index_rx.as_mut() {
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
        }
        for event in events {
            self.progress = Some(event.clone());
            match &event {
                IndexEvent::FileIndexed { path, chunks } => {
                    self.push_log(format!("indexed {path} ({chunks} chunks)"))
                }
                IndexEvent::FileSkipped { path, reason } => {
                    self.push_log(format!("skipped {path}: {reason}"))
                }
                IndexEvent::Failed { reason } => self.push_log(format!("failed: {reason}")),
                IndexEvent::Done {
                    embedded,
                    dropped,
                    indexed,
                    elapsed_ms,
                } => {
                    self.push_log(format!(
                        "done: embedded {embedded}, dropped {dropped}, {indexed} chunks, {} ms",
                        elapsed_ms
                    ));
                    self.progress = None;
                }
                IndexEvent::TableReady { table } => self.push_log(format!("table {table}")),
                _ => {}
            }
        }
        let mut watch_events = Vec::new();
        if let Some(rx) = &self.watch_rx {
            while let Ok(Ok(event)) = rx.try_recv() {
                watch_events.push(event);
            }
        }
        for event in watch_events {
            for path in event.paths.iter().take(5) {
                if let Ok(rel) = path.strip_prefix(&self.workspace) {
                    let rel = rel.to_string_lossy();
                    if rel.starts_with(".git")
                        || rel.contains("target/")
                        || rel.contains("node_modules/")
                    {
                        continue;
                    }
                    let kind = format!("{:?}", event.kind);
                    self.push_log(format!("file {kind}: {rel}"));
                }
            }
        }
    }

    fn push_log(&mut self, line: String) {
        self.log.push(format!(
            "{:>6.1}s  {line}",
            self.started.elapsed().as_secs_f64()
        ));
        if self.log.len() > 200 {
            self.log.remove(0);
        }
        self.new_events += 1;
    }

    fn visible(&self) -> Vec<usize> {
        self.trees
            .iter()
            .enumerate()
            .filter(|(_, node)| self.node_visible(node))
            .filter(|(_, node)| {
                self.search.is_empty()
                    || node
                        .rel
                        .to_lowercase()
                        .contains(&self.search.to_lowercase())
                    || node
                        .name
                        .to_lowercase()
                        .contains(&self.search.to_lowercase())
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn normalize_selection(&mut self) {
        let visible = self.visible().len();
        if visible == 0 {
            self.tree_index = 0;
        } else {
            self.tree_index = self.tree_index.min(visible - 1);
        }
        let langs = self.detected.len();
        if langs == 0 {
            self.lang_index = 0;
        } else {
            self.lang_index = self.lang_index.min(langs - 1);
        }
    }

    fn selected_tree_node(&self) -> Option<&Node> {
        let visible = self.visible();
        visible
            .get(self.tree_index.min(visible.len().saturating_sub(1)))
            .map(|&index| &self.trees[index])
    }

    fn node_visible(&self, node: &Node) -> bool {
        let mut parent = node.parent.clone();
        while let Some(path) = parent {
            if !self.expanded.contains(&path) {
                return false;
            }
            parent = self
                .trees
                .iter()
                .find(|n| n.path == path)
                .and_then(|n| n.parent.clone());
        }
        true
    }

    fn effective(&self, node: &Node) -> Option<Selection> {
        let mut current = Some(node.path.as_path());
        while let Some(path) = current {
            let rel = path
                .strip_prefix(&self.workspace)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            if let Some(state) = self.explicit.get(&rel) {
                return Some(*state);
            }
            current = self
                .trees
                .iter()
                .find(|n| n.path == path)
                .and_then(|n| n.parent.as_deref());
        }
        None
    }

    fn cycle(&mut self) {
        let Some(node) = self.selected_tree_node().cloned() else {
            return;
        };
        let current = self.explicit.get(&node.rel).copied();
        let next = match current {
            None => Some(Selection::Include),
            Some(Selection::Include) => Some(Selection::Exclude),
            Some(Selection::Exclude) => None,
        };
        match next {
            Some(state) => {
                self.explicit.insert(node.rel.clone(), state);
            }
            None => {
                self.explicit.remove(&node.rel);
            }
        }
        self.dirty = true;
        self.push_log(format!("{}: {:?}", node.rel, next));
    }

    fn toggle_language(&mut self) {
        let Some(lang) = self.detected.get(self.lang_index) else {
            return;
        };
        let id = languages::canonical_id(lang.lang).to_string();
        if self.selected_languages.contains(&id) {
            if self.selected_languages.len() > 1 {
                self.selected_languages.remove(&id);
            }
        } else {
            self.selected_languages.insert(id);
        }
        self.dirty = true;
    }

    fn handle_key(&mut self, key: event::KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.should_quit = true;
            return;
        }
        if self.help {
            self.help = false;
            return;
        }
        if self.search_mode {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => {
                    self.search_mode = false;
                    self.normalize_selection();
                }
                KeyCode::Backspace => {
                    self.search.pop();
                    self.normalize_selection();
                }
                KeyCode::Char(c) => {
                    self.search.push(c);
                    self.normalize_selection();
                }
                _ => {}
            }
            return;
        }
        if key.code == KeyCode::Char('?') {
            self.help = true;
            return;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.dirty && !self.quit_armed {
                    self.quit_armed = true;
                    self.push_log("unsaved changes — q again to quit".into());
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Char('1') => {
                self.tab = Tab::Sources;
                self.quit_armed = false;
            }
            KeyCode::Char('2') => {
                self.tab = Tab::Index;
                self.quit_armed = false;
            }
            KeyCode::Char('3') => {
                self.tab = Tab::Stats;
                self.quit_armed = false;
            }
            KeyCode::Char('4') => {
                self.tab = Tab::Usage;
                self.quit_armed = false;
            }
            KeyCode::Char('5') => {
                self.tab = Tab::Review;
                self.quit_armed = false;
            }
            KeyCode::Tab => {
                self.focus = !self.focus;
                self.push_log(
                    if self.focus {
                        "focus: languages"
                    } else {
                        "focus: workspace tree"
                    }
                    .into(),
                );
            }
            KeyCode::Up => {
                if self.tab == Tab::Sources && !self.focus {
                    self.tree_index = self.tree_index.saturating_sub(1);
                } else if self.tab == Tab::Sources {
                    self.lang_index = self.lang_index.saturating_sub(1);
                }
                self.normalize_selection();
            }
            KeyCode::Down => {
                if self.tab == Tab::Sources && !self.focus {
                    self.tree_index =
                        (self.tree_index + 1).min(self.visible().len().saturating_sub(1));
                } else if self.tab == Tab::Sources {
                    self.lang_index =
                        (self.lang_index + 1).min(self.detected.len().saturating_sub(1));
                }
                self.normalize_selection();
            }
            KeyCode::Right => {
                if self.tab == Tab::Sources
                    && !self.focus
                    && let Some(node) = self.selected_tree_node().cloned()
                    && node.is_dir
                {
                    self.expanded.insert(node.path);
                }
                self.normalize_selection();
            }
            KeyCode::Left => {
                if self.tab == Tab::Sources
                    && !self.focus
                    && let Some(node) = self.selected_tree_node().cloned()
                    && node.is_dir
                {
                    self.expanded.remove(&node.path);
                }
                self.normalize_selection();
            }
            KeyCode::Char(' ') => {
                if self.tab == Tab::Sources {
                    if self.focus {
                        self.toggle_language();
                    } else {
                        self.cycle();
                    }
                }
            }
            KeyCode::Char('i') => {
                if self.tab == Tab::Sources && !self.focus {
                    self.set_state(Selection::Include);
                }
            }
            KeyCode::Char('e') => {
                if self.tab == Tab::Sources && !self.focus {
                    self.set_state(Selection::Exclude);
                }
            }
            KeyCode::Char('u') => {
                if self.tab == Tab::Sources && !self.focus {
                    self.clear_state();
                }
            }
            KeyCode::Char('/') => {
                self.tab = Tab::Sources;
                self.search_mode = true;
            }
            KeyCode::Char('r') => {
                self.start_index(false);
            }
            KeyCode::Char('b') => {
                self.start_index(true);
            }
            KeyCode::Char('s') => {
                let _ = self.save();
            }
            KeyCode::Char('a') => match self.save() {
                Ok(()) => self.start_index(false),
                Err(error) => self.push_log(format!("save: {error}")),
            },
            _ => {}
        }
    }

    fn set_state(&mut self, state: Selection) {
        let Some(node) = self.selected_tree_node().cloned() else {
            return;
        };
        self.explicit.insert(node.rel, state);
        self.dirty = true;
    }

    fn clear_state(&mut self) {
        let Some(node) = self.selected_tree_node().cloned() else {
            return;
        };
        self.explicit.remove(&node.rel);
        self.dirty = true;
    }

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        if area.width < 70 || area.height < 18 {
            frame.render_widget(
                Paragraph::new("Terminal too small — resize to at least 70×18"),
                area,
            );
            return;
        }
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(4),
            ])
            .split(area);
        let titles = ["1 Sources", "2 Index", "3 Stats", "4 Usage", "5 Review"];
        let selected = match self.tab {
            Tab::Sources => 0,
            Tab::Index => 1,
            Tab::Stats => 2,
            Tab::Usage => 3,
            Tab::Review => 4,
        };
        frame.render_widget(
            Tabs::new(titles.iter().map(|t| Line::from(*t)).collect::<Vec<_>>()).select(selected),
            layout[0],
        );
        match self.tab {
            Tab::Sources => self.draw_sources(frame, layout[1]),
            Tab::Index => self.draw_index(frame, layout[1]),
            Tab::Stats => self.draw_stats(frame, layout[1]),
            Tab::Usage => self.draw_usage(frame, layout[1]),
            Tab::Review => self.draw_review(frame, layout[1]),
        }
        self.draw_footer(frame, layout[2]);
        if self.help {
            let popup = centered(area, 70, 16);
            frame.render_widget(Clear, popup);
            let help = "operator TUI\n\n1-5 tabs  Tab focus  ↑↓ navigate  ←→ collapse/expand\nSpace toggle  i include  e exclude  u clear\n/ search  r index  b rebuild  a save+index  s save\nesc/q quit  ? help";
            frame.render_widget(
                Paragraph::new(help).block(Block::default().borders(Borders::ALL).title("help")),
                popup,
            );
        }
    }

    fn draw_sources(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(area);
        let visible = self.visible();
        let selected_pos = if visible.is_empty() {
            None
        } else {
            Some(self.tree_index.min(visible.len() - 1))
        };
        let items: Vec<ListItem> = visible
            .iter()
            .map(|&index| {
                let node = &self.trees[index];
                let state = self.effective(node);
                let marker = match state {
                    Some(Selection::Include) => "[i] ",
                    Some(Selection::Exclude) => "[e] ",
                    None => "[ ] ",
                };
                let icon = if node.is_dir {
                    if self.expanded.contains(&node.path) {
                        "▾ "
                    } else {
                        "▸ "
                    }
                } else {
                    "  "
                };
                let text = format!(
                    "{}{}{}{}{}",
                    "  ".repeat(node.depth),
                    marker,
                    icon,
                    node.name,
                    if node.is_dir {
                        format!(" ({})", node.files)
                    } else {
                        String::new()
                    }
                );
                ListItem::new(text)
            })
            .collect();
        self.tree_state.select(selected_pos);
        let tree_block = Block::default()
            .borders(Borders::ALL)
            .title(if self.focus {
                "workspace"
            } else {
                "workspace [focused]"
            })
            .border_style(if self.focus {
                Style::default()
            } else {
                Style::default().fg(Color::Cyan)
            });
        frame.render_stateful_widget(
            List::new(items)
                .block(tree_block)
                .highlight_style(highlight_style())
                .highlight_symbol("▸ "),
            chunks[0],
            &mut self.tree_state,
        );

        let mut langs: Vec<ListItem> = self
            .detected
            .iter()
            .map(|d| {
                let id = languages::canonical_id(d.lang);
                let check = if self.selected_languages.contains(id) {
                    "[x]"
                } else {
                    "[ ]"
                };
                ListItem::new(format!(
                    "{check} {} ({})",
                    languages::display_name(d.lang),
                    d.files
                ))
            })
            .collect();
        if langs.is_empty() {
            langs.push(ListItem::new("no supported languages detected"));
        }
        let selected_lang = if langs.is_empty() {
            None
        } else {
            Some(self.lang_index.min(langs.len() - 1))
        };
        self.lang_state.select(selected_lang);
        let lang_block = Block::default()
            .borders(Borders::ALL)
            .title(if self.focus {
                "languages [focused]"
            } else {
                "languages"
            })
            .border_style(if self.focus {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            });
        frame.render_stateful_widget(
            List::new(langs)
                .block(lang_block)
                .highlight_style(highlight_style())
                .highlight_symbol("▸ "),
            chunks[1],
            &mut self.lang_state,
        );
    }

    fn draw_index(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(9),
                Constraint::Length(3),
                Constraint::Min(0),
            ])
            .split(area);
        let status = self
            .status
            .as_ref()
            .map(|s| {
                format!(
                    "table {} · {} files · fresh {} · stale {} · rows {}",
                    s.table,
                    s.total_files,
                    s.fresh_files,
                    s.stale_files,
                    s.rows.map_or_else(|| "not built".into(), |r| r.to_string())
                )
            })
            .unwrap_or_else(|| "index not inspected yet".into());
        let progress = self
            .progress
            .as_ref()
            .map(describe_event)
            .unwrap_or_else(|| "idle — press r to index".into());
        let text = format!(
            "{}\n\n{}\n\nlanguages: {}\nincludes: {}\nexcludes: {}",
            status,
            progress,
            self.selected_languages
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            self.includes().join(", "),
            self.excludes().join(", ")
        );
        frame.render_widget(
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("index")),
            rows[0],
        );
        let gauge = self
            .progress
            .as_ref()
            .and_then(progress_ratio)
            .unwrap_or(0.0);
        frame.render_widget(
            Gauge::default()
                .percent((gauge * 100.0) as u16)
                .label(format!("{:.0}%", gauge * 100.0)),
            rows[1],
        );
        let log: Vec<Line> = self
            .log
            .iter()
            .rev()
            .take(30)
            .rev()
            .map(|l| Line::from(l.clone()))
            .collect();
        frame.render_widget(
            Paragraph::new(log).block(Block::default().borders(Borders::ALL).title("events")),
            rows[2],
        );
    }

    fn draw_stats(&self, frame: &mut Frame, area: Rect) {
        let text = format!(
            "workspace: {}\nmode: {}\nlanguages: {}\nselected files: {}\nchunks/rows: {}\ntool calls: {}\ntool errors: {}\nsignals: {}\nindex table: {}",
            self.workspace.display(),
            self.mode,
            self.selected_languages.len(),
            self.trees
                .iter()
                .filter(|n| !n.is_dir
                    && n.language
                        .as_ref()
                        .is_some_and(|l| self.selected_languages.contains(l)))
                .count(),
            self.status
                .as_ref()
                .and_then(|s| s.rows)
                .map_or_else(|| "not built".into(), |r| r.to_string()),
            self.stats.calls,
            self.stats.errors,
            self.stats.signals,
            self.status
                .as_ref()
                .map(|s| s.table.clone())
                .unwrap_or_else(|| "not inspected".into())
        );
        frame.render_widget(
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("stats")),
            area,
        );
    }

    fn draw_usage(&self, frame: &mut Frame, area: Rect) {
        let mut lines = vec![
            Line::from(format!(
                "journal calls: {}  errors: {}  bytes: {}",
                self.stats.calls, self.stats.errors, self.stats.bytes
            )),
            Line::from(""),
        ];
        for stat in &self.stats.tools {
            lines.push(Line::from(format!(
                "{:<18} calls {:>5}  errors {:>4}  p50 {:>6}ms  p95 {:>6}ms",
                stat.tool, stat.calls, stat.errors, stat.p50_ms, stat.p95_ms
            )));
        }
        if self.stats.tools.is_empty() {
            lines.push(Line::from(
                "no journal calls yet — run `operator serve` and call a tool",
            ));
        }
        frame.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("tool usage")),
            area,
        );
    }

    fn draw_review(&self, frame: &mut Frame, area: Rect) {
        let text = format!(
            "languages: {}\n\nincludes: {}\nexcludes: {}\n\nconfig: {}\n\ns save  a save + index  q quit",
            self.selected_languages
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            self.includes().join(", "),
            self.excludes().join(", "),
            WorkspaceConfig::path_for(&self.workspace).display()
        );
        frame.render_widget(
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("review")),
            area,
        );
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let text = if self.search_mode {
            format!("search: {}_", self.search)
        } else {
            format!(
                "{}  |  Tab focus: {}  |  1-5 tabs  ↑↓ navigate  space toggle  i/e include/exclude  / search  r index  a save+index  ? help",
                self.message(),
                if self.focus { "languages" } else { "workspace" }
            )
        };
        frame.render_widget(
            Paragraph::new(text).block(Block::default().borders(Borders::ALL)),
            area,
        );
    }

    fn message(&self) -> String {
        match self.tab {
            Tab::Sources => format!(
                "{} languages · {} visible nodes{}",
                self.selected_languages.len(),
                self.visible().len(),
                if self.dirty { " · unsaved" } else { "" }
            ),
            Tab::Index => self
                .progress
                .as_ref()
                .map(describe_event)
                .unwrap_or_else(|| "index idle".into()),
            Tab::Stats => "corpus and index statistics".into(),
            Tab::Usage => "MCP tool usage from the local journal".into(),
            Tab::Review => "review and save the workspace selection".into(),
        }
    }

    fn includes(&self) -> Vec<String> {
        self.explicit
            .iter()
            .filter(|(_, s)| **s == Selection::Include)
            .map(|(p, _)| p.clone())
            .collect()
    }
    fn excludes(&self) -> Vec<String> {
        self.explicit
            .iter()
            .filter(|(_, s)| **s == Selection::Exclude)
            .map(|(p, _)| p.clone())
            .collect()
    }
}

fn highlight_style() -> Style {
    Style::default()
        .fg(Color::Black)
        .bg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}

fn describe_event(event: &IndexEvent) -> String {
    match event {
        IndexEvent::DiscoverStart => "discovering files".into(),
        IndexEvent::Discovered { files } => format!("discovered {files} files"),
        IndexEvent::Languages { languages } => format!("{languages} languages selected"),
        IndexEvent::HashProgress {
            done,
            total,
            changed,
        } => format!("hashed {done}/{total}, {changed} changed"),
        IndexEvent::TableReady { table } => format!("table ready: {table}"),
        IndexEvent::FileIndexed { path, chunks } => format!("indexed {path} ({chunks} chunks)"),
        IndexEvent::FileSkipped { path, reason } => format!("skipped {path}: {reason}"),
        IndexEvent::ManifestSaved { files } => format!("manifest saved ({files} files)"),
        IndexEvent::Done {
            embedded,
            indexed,
            elapsed_ms,
            ..
        } => format!("done: {embedded} files, {indexed} chunks, {elapsed_ms} ms"),
        IndexEvent::Failed { reason } => format!("failed: {reason}"),
    }
}

fn progress_ratio(event: &IndexEvent) -> Option<f64> {
    match event {
        IndexEvent::HashProgress { done, total, .. } if *total > 0 => {
            Some(*done as f64 / *total as f64)
        }
        IndexEvent::FileIndexed { .. } => Some(0.7),
        IndexEvent::Done { .. } => Some(1.0),
        _ => None,
    }
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

fn build_tree(root: &Path) -> Vec<Node> {
    fn walk(root: &Path, dir: &Path, depth: usize, parent: Option<&Path>, out: &mut Vec<Node>) {
        if out.len() > 20_000 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| {
            (
                e.file_type().map(|t| !t.is_dir()).unwrap_or(true),
                e.file_name(),
            )
        });
        for entry in entries {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".git" {
                continue;
            }
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir
                && matches!(
                    name.as_str(),
                    "node_modules" | "target" | "vendor" | "dist" | ".cache"
                )
            {
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            let language = if is_dir {
                None
            } else {
                languages::language_id_for_path(&path).map(str::to_string)
            };
            out.push(Node {
                path: path.clone(),
                rel,
                name,
                is_dir,
                depth,
                parent: parent.map(Path::to_path_buf),
                language,
                files: 0,
            });
            if is_dir {
                walk(root, &path, depth + 1, Some(&path), out);
            }
        }
    }
    let mut nodes = Vec::new();
    walk(root, root, 0, None, &mut nodes);
    // fill directory counts bottom-up
    for index in (0..nodes.len()).rev() {
        let path = nodes[index].path.clone();
        let count = nodes
            .iter()
            .filter(|n| !n.is_dir && n.path.starts_with(&path))
            .count();
        if nodes[index].is_dir {
            nodes[index].files = count;
        }
    }
    nodes
}

struct Guard {
    terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            crossterm::cursor::Show
        );
        let _ = self.terminal.show_cursor();
    }
}

fn terminal() -> Result<Guard> {
    enable_raw_mode().map_err(|e| Error::Msg(format!("enable raw mode: {e}")))?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)
        .map_err(|e| Error::Msg(format!("enter alternate screen: {e}")))?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend).map_err(|e| Error::Msg(format!("terminal: {e}")))?;
    Ok(Guard { terminal })
}

async fn run_app(mut app: App) -> Result<()> {
    let mut guard = terminal()?;
    guard.terminal.clear().ok();
    loop {
        app.tick();
        guard
            .terminal
            .draw(|frame| app.draw(frame))
            .map_err(|e| Error::Msg(format!("draw: {e}")))?;
        if app.should_quit {
            break;
        }
        if event::poll(Duration::from_millis(80)).map_err(|e| Error::Msg(format!("poll: {e}")))?
            && let Event::Key(key) = event::read().map_err(|e| Error::Msg(format!("read: {e}")))?
        {
            app.handle_key(key);
        }
    }
    Ok(())
}

pub async fn run_init(args: InitArgs) -> Result<()> {
    let root = args.workspace_root.unwrap_or_else(|| PathBuf::from("."));
    let workspace = std::fs::canonicalize(&root).unwrap_or(root);
    let detected = languages::detect(&workspace);
    if detected.is_empty() {
        return Err(Error::Config(format!(
            "no supported languages detected under {}; supported: {}",
            workspace.display(),
            languages::supported_ids().join(", ")
        )));
    }
    let mut app = App::new(workspace, "init", detected);
    if let Some(raw) = args.languages.as_deref() {
        let chosen: BTreeSet<String> = raw
            .split(',')
            .filter_map(|id| languages::parse_id(id.trim()))
            .map(|lang| languages::canonical_id(lang).to_string())
            .collect();
        if !chosen.is_empty() {
            app.selected_languages = chosen;
        }
    } else if args.all_supported {
        app.selected_languages = languages::supported_ids()
            .into_iter()
            .map(str::to_string)
            .collect();
    }
    if let Some(raw) = args.include.as_deref() {
        for path in raw.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            app.explicit.insert(
                path.trim_start_matches("./")
                    .trim_end_matches('/')
                    .to_string(),
                Selection::Include,
            );
        }
    }
    if let Some(raw) = args.exclude.as_deref() {
        for path in raw.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            app.explicit.insert(
                path.trim_start_matches("./")
                    .trim_end_matches('/')
                    .to_string(),
                Selection::Exclude,
            );
        }
    }
    app.refresh_status();
    run_app(app).await
}

pub async fn run_reindex(workspace: PathBuf, rebuild: bool) -> Result<()> {
    let workspace = std::fs::canonicalize(&workspace).unwrap_or(workspace);
    if WorkspaceConfig::load(&workspace)?.is_none() {
        return Err(Error::Config(
            "no .operator.toml; run `operator init` first".into(),
        ));
    }
    let detected = languages::detect(&workspace);
    let mut app = App::new(workspace, "reindex", detected);
    app.tab = Tab::Index;
    app.refresh_status();
    app.start_index(rebuild);
    run_app(app).await
}

pub async fn run(args: TuiArgs) -> Result<()> {
    let root = args.workspace_root.unwrap_or_else(|| PathBuf::from("."));
    let workspace = std::fs::canonicalize(&root).unwrap_or(root);
    if WorkspaceConfig::load(&workspace)?.is_none() {
        return Err(Error::Config(
            "no .operator.toml; run `operator init` first".into(),
        ));
    }
    let detected = languages::detect(&workspace);
    let mut app = App::new(workspace, "dashboard", detected);
    app.tab = Tab::Index;
    app.refresh_status();
    run_app(app).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn app() -> (tempfile::TempDir, App) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.path().join("a.go"), "package main\n").unwrap();
        std::fs::write(dir.path().join("a.dart"), "int x = 1;\n").unwrap();
        let detected = languages::detect(dir.path());
        let workspace = dir.path().to_path_buf();
        (dir, App::new(workspace, "test", detected))
    }

    #[test]
    fn language_toggle_and_review_config_round_trip() {
        let (_dir, mut app) = app();
        let before = app.selected_languages.len();
        app.toggle_language();
        assert_eq!(
            app.selected_languages.len(),
            before.saturating_sub(1).max(1)
        );
        let cfg = app.config();
        assert_eq!(cfg.schema_version, Some(3));
        assert!(cfg.languages.is_some());
    }

    #[test]
    fn include_and_exclude_are_persisted_as_paths() {
        let (_dir, mut app) = app();
        app.explicit.insert("src".into(), Selection::Include);
        app.explicit.insert("a.go".into(), Selection::Exclude);
        let cfg = app.config();
        assert_eq!(cfg.retrieval_include.unwrap(), vec!["src".to_string()]);
        assert_eq!(cfg.retrieval_exclude.unwrap(), vec!["a.go".to_string()]);
    }

    #[test]
    fn source_tab_renders_without_terminal() {
        let (_dir, mut app) = app();
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(text.contains("workspace"));
        assert!(text.contains("languages"));
        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .any(|cell| cell.bg == Color::Cyan),
            "the selected workspace row must have a visible highlight"
        );
        assert_eq!(app.tree_state.selected(), Some(0));
        app.focus = true;
        app.lang_index = 1.min(app.detected.len().saturating_sub(1));
        terminal.draw(|frame| app.draw(frame)).unwrap();
        assert_eq!(app.lang_state.selected(), Some(app.lang_index));
        for tab in [Tab::Index, Tab::Stats, Tab::Usage, Tab::Review] {
            app.tab = tab;
            terminal.draw(|frame| app.draw(frame)).unwrap();
        }
    }
}
