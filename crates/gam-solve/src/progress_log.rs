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

pub fn init_logging() {
    LOG_START.get_or_init(Instant::now);
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
    // Log the GPU backend inventory once at startup so the "are GPUs being
    // used?" answer is visible at the top of the log, before any solver
    // dispatch site lazily checks for device support.
    gam_gpu::log_backend_inventory_once();
}
