//! Whether progress chatter is withheld while a full-screen UI owns the
//! terminal.
//!
//! The TUI paints an alternate screen every frame, so a raw `eprintln!` from a
//! background index or a config write would scribble over it; ratatui's buffer
//! no longer matches the screen, so it would never repair the damage. The TUI
//! sets this flag for its lifetime and shows the same progress in its events
//! panel. CLI and `serve` runs leave it off and keep their diagnostics.

use std::sync::atomic::{AtomicBool, Ordering};

static QUIET: AtomicBool = AtomicBool::new(false);

/// Withhold or restore progress written directly to stderr.
///
/// Only the TUI (the `tui` feature) owns the terminal; a build without it never
/// calls this, so the setter would otherwise be dead code.
#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub(crate) fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

/// Whether stderr progress is currently withheld.
pub(crate) fn quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// `eprintln!` unless a full-screen UI is painting the terminal.
macro_rules! note {
    ($($arg:tt)*) => {{
        if !$crate::console::quiet() {
            eprintln!($($arg)*);
        }
    }};
}

pub(crate) use note;
