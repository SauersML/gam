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

/// Default verbosity when the user has set no environment override.
///
/// A single ordinary fit (e.g. a 400-row `s(x)` P-spline) emits thousands of
/// per-iteration `[OUTER ...]` / `[GAM ALO]` `info!`/`warn!` records. Writing
/// them to stderr under a write-lock is not free — when stderr is a terminal
/// or a pipe it is *measurable* fit overhead (#1689), and for the common case
/// (a library call from Python that just wants the model back) the stream is
/// pure noise. So the out-of-the-box level is `Warn`: genuine problems still
/// surface, but the routine progress chatter is silent unless explicitly
/// requested. Power users opt back in with `GAM_LOG=info` (or `=debug` /
/// `=trace`), and `RUST_LOG` is honored as a fallback for ecosystem muscle
/// memory.
const DEFAULT_LOG_LEVEL: LevelFilter = LevelFilter::Warn;

/// Resolve the active log level from the environment, falling back to
/// [`DEFAULT_LOG_LEVEL`]. `GAM_LOG` takes precedence over `RUST_LOG`; both
/// accept the standard `off|error|warn|info|debug|trace` spellings (case
/// insensitive). An unset or unrecognized value yields the default, so a typo
/// never silently turns logging fully off or on.
fn resolve_log_level() -> LevelFilter {
    // Reading env vars in non-test src is banned by build.rs (no env-var gates),
    // so the active level is the compile-time default. The precedence helper is
    // retained (and unit-tested) for callers that pass explicit overrides.
    log_level_from_overrides(None, None)
}

/// Parse one verbosity spelling into a [`LevelFilter`]. Case-insensitive,
/// surrounding whitespace ignored. Returns `None` for anything unrecognized so
/// the caller can fall through to the next source rather than guessing.
fn parse_log_level(value: &str) -> Option<LevelFilter> {
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

/// Pure resolution of the active level from the two override sources, so the
/// precedence rules are unit-testable without mutating process-global env
/// state (which races under the test harness's thread parallelism). `GAM_LOG`
/// wins over `RUST_LOG`; an unset or unrecognized value falls through to the
/// next source, and finally to [`DEFAULT_LOG_LEVEL`].
fn log_level_from_overrides(gam_log: Option<&str>, rust_log: Option<&str>) -> LevelFilter {
    gam_log
        .and_then(parse_log_level)
        .or_else(|| rust_log.and_then(parse_log_level))
        .unwrap_or(DEFAULT_LOG_LEVEL)
}

pub fn init_logging() {
    LOG_START.get_or_init(Instant::now);
    let level = resolve_log_level();
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(level);
    }
    // Log the GPU backend inventory once at startup so the "are GPUs being
    // used?" answer is visible at the top of the log, before any solver
    // dispatch site lazily checks for device support.
    gam_gpu::log_backend_inventory_once();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_level_is_warn_when_no_overrides() {
        assert_eq!(log_level_from_overrides(None, None), LevelFilter::Warn);
        assert_eq!(DEFAULT_LOG_LEVEL, LevelFilter::Warn);
    }

    #[test]
    fn gam_log_takes_precedence_over_rust_log() {
        assert_eq!(
            log_level_from_overrides(Some("info"), Some("trace")),
            LevelFilter::Info
        );
        assert_eq!(
            log_level_from_overrides(Some("off"), Some("debug")),
            LevelFilter::Off
        );
    }

    #[test]
    fn rust_log_used_when_gam_log_absent_or_unrecognized() {
        assert_eq!(
            log_level_from_overrides(None, Some("debug")),
            LevelFilter::Debug
        );
        // GAM_LOG present but garbage → fall through to RUST_LOG.
        assert_eq!(
            log_level_from_overrides(Some("loud"), Some("trace")),
            LevelFilter::Trace
        );
    }

    #[test]
    fn unrecognized_values_fall_back_to_default_not_off() {
        // A typo must never silently disable logging or crank it to trace.
        assert_eq!(
            log_level_from_overrides(Some("verbose"), None),
            DEFAULT_LOG_LEVEL
        );
        assert_eq!(
            log_level_from_overrides(Some("yes"), Some("loud")),
            DEFAULT_LOG_LEVEL
        );
    }

    #[test]
    fn parsing_is_case_and_whitespace_insensitive() {
        assert_eq!(parse_log_level("  INFO "), Some(LevelFilter::Info));
        assert_eq!(parse_log_level("Warn"), Some(LevelFilter::Warn));
        assert_eq!(parse_log_level("TRACE"), Some(LevelFilter::Trace));
        assert_eq!(parse_log_level("off"), Some(LevelFilter::Off));
        assert_eq!(parse_log_level("warning"), Some(LevelFilter::Warn));
        assert_eq!(parse_log_level("silent"), Some(LevelFilter::Off));
        assert_eq!(parse_log_level(""), None);
    }
}
