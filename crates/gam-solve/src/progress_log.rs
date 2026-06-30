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
/// requested. Power users opt back in by calling [`set_log_level`] (e.g.
/// `set_log_level("info")`) or [`log::set_max_level`] directly — verbosity is
/// set through an explicit API, not a process-global env var (`std::env::var`
/// is banned crate-wide; see [`resolve_log_level`]).
const DEFAULT_LOG_LEVEL: LevelFilter = LevelFilter::Warn;

/// Resolve the active log level.
///
/// Reading verbosity from the environment (`GAM_LOG` / `RUST_LOG`) is NOT
/// permitted: `std::env::var` is banned crate-wide (build.rs substring scanner,
/// `feedback_no_env_vars` policy), because process-global env reads are a hidden
/// input that makes a library call's behavior depend on ambient state the caller
/// never passed. An override path landed transiently in #1696 and broke the
/// build; the supported way to change verbosity is the explicit
/// [`set_log_level`] entry point (or [`log::set_max_level`] directly), not an
/// env var. The out-of-the-box level stays [`DEFAULT_LOG_LEVEL`] (`Warn`): a
/// plain library call from Python gets quiet logs and pays no per-record stderr
/// cost (#1689), while genuine problems still surface.
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

/// Pure resolution of the active level from up to two explicitly-supplied
/// override spellings, with the primary source winning over the fallback and an
/// unset/unrecognized value falling through to [`DEFAULT_LOG_LEVEL`]. Kept pure
/// (no env, no globals) so the precedence rules are unit-testable and so the
/// public [`set_log_level`] entry point can hand it caller-provided strings.
fn log_level_from_overrides(primary: Option<&str>, fallback: Option<&str>) -> LevelFilter {
    primary
        .and_then(parse_log_level)
        .or_else(|| fallback.and_then(parse_log_level))
        .unwrap_or(DEFAULT_LOG_LEVEL)
}

/// Map a caller-supplied verbosity spelling onto a [`LevelFilter`]. Wraps the
/// internal [`parse_log_level`] for out-of-crate callers (the CLI `--log-level`
/// flag, the Python `set_log_level` shim). Returns `None` for blank/unrecognized
/// input so the caller decides the fallback rather than guessing here.
pub fn parse_level_directive(raw: &str) -> Option<LevelFilter> {
    parse_log_level(raw)
}

/// The quiet out-of-the-box verbosity, exposed so callers that want the
/// documented default after an explicit-level path can name it instead of
/// hardcoding `Warn`.
pub const fn default_log_level() -> LevelFilter {
    DEFAULT_LOG_LEVEL
}

/// Explicitly set the active log verbosity from a level spelling
/// (`off|error|warn|info|debug|trace`, case-insensitive). This is the supported
/// way to raise verbosity above the default — callers pass the level they want
/// rather than relying on a process-global env var (`std::env::var` is banned
/// crate-wide; see [`resolve_log_level`]). Returns the [`LevelFilter`] actually
/// installed (the default when `spelling` is unrecognized, so a typo never
/// silently disables logging). A no-op-safe wrapper over [`log::set_max_level`].
pub fn set_log_level(spelling: &str) -> LevelFilter {
    let level = log_level_from_overrides(Some(spelling), None);
    log::set_max_level(level);
    level
}

pub fn init_logging() {
    init_logging_at(resolve_log_level());
}

/// Install the stderr logger at an explicit verbosity. Idempotent in the sense
/// that the first caller wins the global `log` backend registration; **every**
/// call (re-)applies the requested max level, so an embedding can call
/// `init_logging()` early and later raise the level via `init_logging_at` (e.g.
/// the Python `set_log_level` shim) without losing the override. This is how a
/// caller opts back into the verbose `Info`/`debug`/`trace` solver trace that
/// the `Warn` default suppresses for performance (#1688).
pub fn init_logging_at(level: LevelFilter) {
    LOG_START.get_or_init(Instant::now);
    // First caller wins the backend registration; an already-installed logger
    // is fine — we still want to (re-)apply the requested level below, so do
    // not gate `set_max_level` on the registration result.
    if log::set_logger(&LOGGER).is_err() {
        // backend already installed by an earlier call — fall through.
    }
    log::set_max_level(level);
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
    fn primary_override_takes_precedence_over_fallback() {
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
    fn fallback_used_when_primary_absent_or_unrecognized() {
        assert_eq!(
            log_level_from_overrides(None, Some("debug")),
            LevelFilter::Debug
        );
        // Primary present but garbage → fall through to the fallback.
        assert_eq!(
            log_level_from_overrides(Some("loud"), Some("trace")),
            LevelFilter::Trace
        );
    }

    #[test]
    fn set_log_level_installs_explicit_level_and_defaults_on_typo() {
        // Explicit spelling installs that level and reports it back.
        assert_eq!(set_log_level("debug"), LevelFilter::Debug);
        assert_eq!(log::max_level(), LevelFilter::Debug);
        // A typo never silently disables logging — it falls back to the default.
        assert_eq!(set_log_level("verbose"), DEFAULT_LOG_LEVEL);
        assert_eq!(log::max_level(), DEFAULT_LOG_LEVEL);
        // Restore a quiet default so this process-global write cannot perturb
        // sibling tests that observe the level.
        log::set_max_level(DEFAULT_LOG_LEVEL);
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
    fn parse_level_directive_matches_internal_parser() {
        // The public out-of-crate entry point must behave exactly like the
        // internal parser the env-precedence helper uses.
        for spelling in ["off", "error", "warn", "info", "debug", "trace", "", "garbage"] {
            assert_eq!(parse_level_directive(spelling), parse_log_level(spelling));
        }
    }

    #[test]
    fn default_log_level_accessor_matches_const() {
        assert_eq!(default_log_level(), DEFAULT_LOG_LEVEL);
        assert_eq!(default_log_level(), LevelFilter::Warn);
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
