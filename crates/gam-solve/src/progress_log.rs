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

/// Resolve the default max log level, honoring (in priority order) `GAM_LOG`
/// then `RUST_LOG`. The accepted values are the standard `log` level names
/// (`off`, `error`, `warn`, `info`, `debug`, `trace`), case-insensitive.
///
/// The default is [`LevelFilter::Warn`], NOT `Info`. The `Info` stream is a
/// firehose of per-evaluation solver diagnostics — `[OUTER …]`, `[GAM ALO]`,
/// `[KAPPA-PHASE …]`, the `[#1271-diag]` REML logdet dump, etc. A single
/// 500-row `matern(x)` fit emits tens of thousands of `Info` lines (#1688).
/// Several of those `Info`-gated blocks are not free formatting: the
/// `[#1271-diag]` probe runs *two full symmetric eigendecompositions per REML
/// evaluation* purely to populate the log line (see
/// `reml/objective.rs`). With hundreds of REML evaluations per fit that is real
/// wasted compute, not just I/O. Defaulting to `Warn` keeps genuine warnings
/// and errors visible while removing the per-iteration diagnostic cost; users
/// who want the full trace opt in with `GAM_LOG=info` / `RUST_LOG=info`.
/// The default solver log verbosity when a caller does not request one
/// explicitly: [`LevelFilter::Warn`], NOT `Info`.
///
/// The `Info` stream is a firehose of per-evaluation solver diagnostics —
/// `[OUTER …]`, `[GAM ALO]`, `[KAPPA-PHASE …]`, the `[#1271-diag]` REML logdet
/// dump, etc. A single 500-row `matern(x)` fit emits tens of thousands of
/// `Info` lines (#1688). Several of those `Info`-gated blocks are not free
/// formatting: the `[#1271-diag]` probe runs *two full symmetric
/// eigendecompositions per REML evaluation* purely to populate the log line
/// (see `reml/objective.rs`). With hundreds of REML evaluations per fit that is
/// real wasted compute, not just I/O. Defaulting to `Warn` keeps genuine
/// warnings and errors visible while removing the per-iteration diagnostic
/// cost; callers who want the full trace opt in (CLI `--log-level info`, or
/// [`init_logging_at`] from an embedding).
pub const DEFAULT_LOG_LEVEL: LevelFilter = LevelFilter::Warn;

/// Parse a single log-level name into a [`LevelFilter`]. Returns `None` when
/// the string is empty/blank. Unrecognized non-empty values map to `Info`:
/// this minimal logger has no per-module filtering, so a per-target directive
/// like `gam=info` should still turn the verbose stream on rather than be
/// silently swallowed.
pub fn parse_level_directive(raw: &str) -> Option<LevelFilter> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(match trimmed.to_ascii_lowercase().as_str() {
        "off" | "none" | "silent" | "quiet" => LevelFilter::Off,
        "error" => LevelFilter::Error,
        "warn" | "warning" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" | "all" => LevelFilter::Trace,
        _ => LevelFilter::Info,
    })
}

/// Install the stderr logger at the [`DEFAULT_LOG_LEVEL`] (`Warn`).
pub fn init_logging() {
    init_logging_at(DEFAULT_LOG_LEVEL);
}

/// Install the stderr logger at an explicit verbosity. Idempotent: the first
/// caller wins the global `log` backend registration; later calls only adjust
/// the active max level. This lets the CLI honor a `--log-level` flag and an
/// embedding (Python) opt back into the full `Info` trace.
pub fn init_logging_at(level: LevelFilter) {
    LOG_START.get_or_init(Instant::now);
    // First caller wins the global backend registration; ignore the
    // already-installed error so later calls can still adjust the level.
    if log::set_logger(&LOGGER).is_err() {
        // backend already installed by an earlier call — fall through to
        // re-apply the requested max level below.
    }
    log::set_max_level(level);
    // Log the GPU backend inventory once at startup so the "are GPUs being
    // used?" answer is visible at the top of the log, before any solver
    // dispatch site lazily checks for device support.
    gam_gpu::log_backend_inventory_once();
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_LOG_LEVEL, LevelFilter, parse_level_directive};

    #[test]
    fn default_level_is_warn_not_info() {
        // Regression guard for #1688: the default solver verbosity must stay
        // below `Info` so production fits do not pay the per-evaluation
        // diagnostic cost (eigendecompositions in the `[#1271-diag]` block).
        assert_eq!(DEFAULT_LOG_LEVEL, LevelFilter::Warn);
    }

    #[test]
    fn parse_level_directive_recognizes_standard_names() {
        assert_eq!(parse_level_directive("off"), Some(LevelFilter::Off));
        assert_eq!(parse_level_directive("error"), Some(LevelFilter::Error));
        assert_eq!(parse_level_directive("warn"), Some(LevelFilter::Warn));
        assert_eq!(parse_level_directive("info"), Some(LevelFilter::Info));
        assert_eq!(parse_level_directive("debug"), Some(LevelFilter::Debug));
        assert_eq!(parse_level_directive("trace"), Some(LevelFilter::Trace));
    }

    #[test]
    fn parse_level_directive_is_case_and_whitespace_insensitive() {
        assert_eq!(parse_level_directive("  INFO "), Some(LevelFilter::Info));
        assert_eq!(parse_level_directive("Warn"), Some(LevelFilter::Warn));
        assert_eq!(parse_level_directive("OFF"), Some(LevelFilter::Off));
    }

    #[test]
    fn parse_level_directive_blank_is_none() {
        assert_eq!(parse_level_directive(""), None);
        assert_eq!(parse_level_directive("   "), None);
    }

    #[test]
    fn parse_level_directive_unknown_falls_back_to_info() {
        // A per-target directive (`gam=info`) or any other unrecognized value
        // should still enable the verbose stream rather than be swallowed.
        assert_eq!(parse_level_directive("gam=info"), Some(LevelFilter::Info));
        assert_eq!(parse_level_directive("verbose"), Some(LevelFilter::Info));
    }
}
