//! Stderr progress logger for the `gam` CLI and the `gamfit` Python bindings.
//!
//! Installs a global [`log`] backend that timestamps each record (elapsed since
//! process start), strips terminal control / escape sequences, and writes to
//! stderr under a write-lock so concurrent solver threads never interleave
//! partial lines. This is the sole logger bootstrap for the CLI binary and the
//! Python extension module.
//!
//! (Extracted from the former TUI `visualizer` module, which has been removed
//! along with its `crossterm`/`ratatui` dependencies; only the stderr logging
//! survives — the live chart / progress lanes were non-essential opt-in cruft.)

use log::{LevelFilter, Log, Metadata, Record};
use std::io::{self, Write};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static LOGGER: ProgressLogger = ProgressLogger;
static LOG_START: OnceLock<Instant> = OnceLock::new();
static LOG_WRITE_LOCK: Mutex<()> = Mutex::new(());

struct ProgressLogger;

impl Log for ProgressLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let lines = format_log_record(record);
        let log_lock_guard = LOG_WRITE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut stderr = io::stderr().lock();
        for line in lines {
            writeln!(stderr, "{line}").ok();
        }
        drop(log_lock_guard);
    }

    fn flush(&self) {}
}

fn format_log_record(record: &Record<'_>) -> Vec<String> {
    let elapsed = LOG_START.get_or_init(Instant::now).elapsed();
    let prefix = format!("[{}]", human_elapsed(elapsed));
    sanitize_log_message(&record.args().to_string())
        .lines()
        .map(|line| format!("{prefix} {line}"))
        .collect()
}

fn sanitize_log_message(message: &str) -> String {
    let mut sanitized = String::with_capacity(message.len());
    let mut chars = message.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\x1b' => {
                strip_escape_sequence(&mut chars);
            }
            '\r' => sanitized.push('\n'),
            '\n' | '\t' => sanitized.push(ch),
            ch if ch.is_control() => {}
            ch => sanitized.push(ch),
        }
    }
    sanitized
}

fn strip_escape_sequence<I>(chars: &mut std::iter::Peekable<I>)
where
    I: Iterator<Item = char>,
{
    match chars.next() {
        Some('[') => {
            for seq_ch in chars.by_ref() {
                if ('@'..='~').contains(&seq_ch) {
                    break;
                }
            }
        }
        Some(']') => strip_string_escape(chars),
        Some('P' | 'X' | '^' | '_') => strip_string_escape(chars),
        Some(_) | None => {}
    }
}

fn strip_string_escape<I>(chars: &mut std::iter::Peekable<I>)
where
    I: Iterator<Item = char>,
{
    while let Some(seq_ch) = chars.next() {
        if seq_ch == '\x07' {
            break;
        }
        if seq_ch == '\x1b' && chars.next_if_eq(&'\\').is_some() {
            break;
        }
    }
}

fn human_elapsed(elapsed: Duration) -> String {
    let total_secs = elapsed.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs / 60) % 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours}h {minutes:02}m {seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

/// Default verbosity when the caller installs the logger without an explicit
/// level.
///
/// A single ordinary fit (e.g. a 400-row `s(x)` P-spline) emits thousands of
/// per-iteration `[OUTER ...]` / `[GAM ALO]` `info!`/`warn!` records. Writing
/// them to stderr under a write-lock is not free — when stderr is a terminal
/// or a pipe it is *measurable* fit overhead (#1689), and for the common case
/// (a library call from Python that just wants the model back) the stream is
/// pure noise. So the out-of-the-box level is `Warn`: genuine problems still
/// surface, but the routine progress chatter is silent unless explicitly
/// requested. Callers opt back in via [`init_logging_with_level`] (the CLI's
/// `-v`/`--quiet` flags; the Python `gamfit.set_log_level(...)` shim).
///
/// This crate is deliberately env-var-free (the project bans `env::var`), so
/// verbosity is always a programmatic decision made by the host, never an
/// ambient `RUST_LOG`.
pub const DEFAULT_LOG_LEVEL: LevelFilter = LevelFilter::Warn;

/// Map a verbosity *level name* to a [`LevelFilter`]. Case-insensitive,
/// surrounding whitespace ignored. Returns `None` for anything unrecognized so
/// a caller surfacing a `--log-level` string can reject typos rather than
/// silently guessing. Accepts the standard `off|error|warn|info|debug|trace`
/// spellings plus a few friendly aliases.
pub fn parse_log_level(value: &str) -> Option<LevelFilter> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" | "silent" => Some(LevelFilter::Off),
        "error" => Some(LevelFilter::Error),
        "warn" | "warning" => Some(LevelFilter::Warn),
        "info" => Some(LevelFilter::Info),
        "debug" => Some(LevelFilter::Debug),
        "trace" | "all" => Some(LevelFilter::Trace),
        _ => None,
    }
}

/// Translate a signed verbosity delta into a level relative to the default.
///
/// `0` is the [`DEFAULT_LOG_LEVEL`] (`Warn`); each positive step is one level
/// more verbose (`info`, then `debug`, then `trace`) and each negative step is
/// one level quieter (`error`, then `off`). Saturates at the ends. This is the
/// `-v`/`-vv` / `-q`/`-qq` count semantics CLI front-ends expect, kept pure so
/// it is unit-testable without touching global logger state.
pub fn level_from_verbosity_delta(delta: i32) -> LevelFilter {
    // Ordered quietest → loudest so an index walk is a verbosity walk.
    const LADDER: [LevelFilter; 6] = [
        LevelFilter::Off,
        LevelFilter::Error,
        LevelFilter::Warn,
        LevelFilter::Info,
        LevelFilter::Debug,
        LevelFilter::Trace,
    ];
    // Index of DEFAULT_LOG_LEVEL (Warn) in LADDER.
    const DEFAULT_IDX: i32 = 2;
    let idx = (DEFAULT_IDX + delta).clamp(0, LADDER.len() as i32 - 1);
    LADDER[idx as usize]
}

/// Install the global logger at [`DEFAULT_LOG_LEVEL`]. Convenience wrapper used
/// by callers that have no verbosity control of their own (e.g. the Python
/// extension's module-init, which then lets `gamfit.set_log_level` adjust it).
pub fn init_logging() {
    init_logging_with_level(DEFAULT_LOG_LEVEL);
}

/// Install the global logger at an explicit `level`. Idempotent: the logger is
/// only registered once (first caller wins the `set_logger` race), but the
/// active max-level is always (re)applied, so a later, more deliberate call
/// (the CLI parsing `-v`) can raise or lower verbosity after an early default
/// install.
pub fn init_logging_with_level(level: LevelFilter) {
    LOG_START.get_or_init(Instant::now);
    // First caller wins the registration; a later call is a no-op `Err` we
    // intentionally ignore. The active max-level below is always (re)applied,
    // so a deliberate later call can still raise/lower verbosity.
    log::set_logger(&LOGGER).ok();
    log::set_max_level(level);
    // Log the GPU backend inventory once at startup so the "are GPUs being
    // used?" answer is visible at the top of the log, before any solver
    // dispatch site lazily checks for device support.
    gam_gpu::log_backend_inventory_once();
}

/// Set only the active verbosity, assuming the logger is already installed.
/// Used by the Python `set_log_level` shim, which runs after module-init has
/// already registered the logger.
pub fn set_log_level(level: LevelFilter) {
    log::set_max_level(level);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_level_is_warn() {
        assert_eq!(DEFAULT_LOG_LEVEL, LevelFilter::Warn);
    }

    #[test]
    fn parsing_is_case_and_whitespace_insensitive() {
        assert_eq!(parse_log_level("  INFO "), Some(LevelFilter::Info));
        assert_eq!(parse_log_level("Warn"), Some(LevelFilter::Warn));
        assert_eq!(parse_log_level("TRACE"), Some(LevelFilter::Trace));
        assert_eq!(parse_log_level("off"), Some(LevelFilter::Off));
        assert_eq!(parse_log_level("warning"), Some(LevelFilter::Warn));
        assert_eq!(parse_log_level("silent"), Some(LevelFilter::Off));
        assert_eq!(parse_log_level("error"), Some(LevelFilter::Error));
        assert_eq!(parse_log_level("debug"), Some(LevelFilter::Debug));
    }

    #[test]
    fn parsing_rejects_unrecognized_spellings() {
        // A typo must be surfaced (None) so a `--log-level` front-end can
        // reject it, never silently turn logging off or to trace.
        assert_eq!(parse_log_level(""), None);
        assert_eq!(parse_log_level("verbose"), None);
        assert_eq!(parse_log_level("loud"), None);
        assert_eq!(parse_log_level("yes"), None);
    }

    #[test]
    fn verbosity_delta_zero_is_default() {
        assert_eq!(level_from_verbosity_delta(0), DEFAULT_LOG_LEVEL);
    }

    #[test]
    fn verbosity_delta_climbs_and_saturates_loud() {
        assert_eq!(level_from_verbosity_delta(1), LevelFilter::Info);
        assert_eq!(level_from_verbosity_delta(2), LevelFilter::Debug);
        assert_eq!(level_from_verbosity_delta(3), LevelFilter::Trace);
        // Saturates at Trace, never panics on a large count.
        assert_eq!(level_from_verbosity_delta(99), LevelFilter::Trace);
    }

    #[test]
    fn verbosity_delta_descends_and_saturates_quiet() {
        assert_eq!(level_from_verbosity_delta(-1), LevelFilter::Error);
        assert_eq!(level_from_verbosity_delta(-2), LevelFilter::Off);
        // Saturates at Off.
        assert_eq!(level_from_verbosity_delta(-99), LevelFilter::Off);
    }
}
