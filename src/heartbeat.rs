//! Process-wide liveness monitor with per-thread scope stacks.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const PROCESS_MONITOR_INTERVAL: Duration = Duration::from_secs(60);

/// Maximum number of per-thread phase lines emitted in one periodic dump
/// (ordered by deepest-frame age, oldest first); the remainder is summarized
/// as a count so the dump stays readable on a many-threaded process.
const PROCESS_MONITOR_MAX_PHASE_LINES: usize = 8;

/// A thread whose deepest instrumented frame has been live longer than this is
/// flagged with a loud `[process-monitor][STALL]` line so a long unlogged
/// phase is impossible to miss in the log.
const PROCESS_MONITOR_STALL_THRESHOLD: Duration = Duration::from_secs(120);

static PROCESS_MONITOR: OnceLock<Arc<ProcessMonitorState>> = OnceLock::new();

thread_local! {
    static THREAD_STACK: RefCell<ThreadStack> = RefCell::new(ThreadStack::new());
}

#[derive(Clone)]
struct FrameSnapshot {
    label: String,
    entered: Instant,
}

struct ThreadSnapshot {
    name: Option<String>,
    stack: Vec<FrameSnapshot>,
    updated: Instant,
}

struct ProcessMonitorState {
    started: Instant,
    threads: Mutex<BTreeMap<String, ThreadSnapshot>>,
}

struct ThreadStack {
    id: String,
    name: Option<String>,
    stack: Vec<FrameSnapshot>,
}

pub(crate) struct ProcessScopeGuard;

impl ThreadStack {
    fn new() -> Self {
        let thread = thread::current();
        Self {
            id: format!("{:?}", thread.id()),
            name: thread.name().map(str::to_string),
            stack: Vec::new(),
        }
    }
}

impl ProcessMonitorState {
    fn update_thread(&self, thread: &ThreadStack) {
        let mut threads = self
            .threads
            .lock()
            .expect("process monitor registry poisoned");
        if thread.stack.is_empty() {
            threads.remove(&thread.id);
        } else {
            threads.insert(
                thread.id.clone(),
                ThreadSnapshot {
                    name: thread.name.clone(),
                    stack: thread.stack.clone(),
                    updated: Instant::now(),
                },
            );
        }
    }

    fn emit(&self) {
        let threads = self
            .threads
            .lock()
            .expect("process monitor registry poisoned");
        let resource = ProcessResourceSnapshot::read();
        // `threads` only ever holds entries whose scope stack is non-empty
        // (`update_thread` removes a thread the moment its stack drains), so
        // its length is exactly the count of threads currently inside an
        // instrumented scope — the meaningful "active" metric.
        let active_threads = threads.len();

        // Build a per-thread view keyed on the DEEPEST frame (the innermost
        // phase the thread is actually executing) and how long it has been
        // there. Order oldest-first so the most-likely-stalled phases sort to
        // the top of the (capped) dump.
        struct ThreadPhase<'a> {
            thread_label: String,
            depth: usize,
            updated_ago: Duration,
            deepest_label: &'a str,
            deepest_age: Duration,
        }
        let mut phases: Vec<ThreadPhase<'_>> = Vec::with_capacity(threads.len());
        for (thread_id, thread) in threads.iter() {
            let Some(deepest) = thread.stack.last() else {
                continue;
            };
            let thread_label = match &thread.name {
                Some(name) => format!("{thread_id}/{name}"),
                None => thread_id.clone(),
            };
            phases.push(ThreadPhase {
                thread_label,
                depth: thread.stack.len(),
                updated_ago: thread.updated.elapsed(),
                deepest_label: deepest.label.as_str(),
                deepest_age: deepest.entered.elapsed(),
            });
        }
        phases.sort_by(|a, b| b.deepest_age.cmp(&a.deepest_age));

        log::info!(
            "[process-monitor] elapsed={} {} active_threads={}",
            format_duration(self.started.elapsed()),
            resource.format(),
            active_threads,
        );

        // STALL warnings first, loud and unconditional (not subject to the
        // per-dump phase-line cap) so a long unlogged phase is never silent.
        for phase in phases
            .iter()
            .filter(|p| p.deepest_age >= PROCESS_MONITOR_STALL_THRESHOLD)
        {
            log::warn!(
                "[process-monitor][STALL] thread={} phase={:?} stuck={}",
                phase.thread_label,
                phase.deepest_label,
                format_duration(phase.deepest_age),
            );
        }

        // Compact per-thread phase summary: deepest frame label + age, capped.
        for phase in phases.iter().take(PROCESS_MONITOR_MAX_PHASE_LINES) {
            log::info!(
                "[process-monitor] phase thread={} depth={} deepest={:?} in_frame={} updated_ago={}",
                phase.thread_label,
                phase.depth,
                phase.deepest_label,
                format_duration(phase.deepest_age),
                format_duration(phase.updated_ago),
            );
        }
        if phases.len() > PROCESS_MONITOR_MAX_PHASE_LINES {
            log::info!(
                "[process-monitor] phase ... and {} more active thread(s) omitted",
                phases.len() - PROCESS_MONITOR_MAX_PHASE_LINES,
            );
        }
    }
}

impl Drop for ProcessScopeGuard {
    fn drop(&mut self) {
        let state = process_monitor();
        THREAD_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.stack.pop();
            state.update_thread(&stack);
        });
    }
}

/// Start the background process monitor thread if it is not already running.
pub fn start() {
    process_monitor();
}

pub(crate) fn track_scope(label: impl Into<String>) -> ProcessScopeGuard {
    let state = process_monitor();
    THREAD_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.stack.push(FrameSnapshot {
            label: label.into(),
            entered: Instant::now(),
        });
        state.update_thread(&stack);
    });
    ProcessScopeGuard
}

fn process_monitor() -> Arc<ProcessMonitorState> {
    PROCESS_MONITOR
        .get_or_init(|| {
            let state = Arc::new(ProcessMonitorState {
                started: Instant::now(),
                threads: Mutex::new(BTreeMap::new()),
            });
            start_process_monitor_thread(Arc::clone(&state));
            state
        })
        .clone()
}

fn start_process_monitor_thread(state: Arc<ProcessMonitorState>) {
    let builder = thread::Builder::new().name("gam-process-monitor".to_string());
    match builder.spawn(move || {
        loop {
            thread::park_timeout(PROCESS_MONITOR_INTERVAL);
            state.emit();
        }
    }) {
        Ok(handle) => drop(handle),
        Err(err) => log::warn!("failed to start process monitor thread: {err}"),
    }
}

fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

#[derive(Default)]
struct ProcessResourceSnapshot {
    rss_kb: Option<u64>,
    peak_rss_kb: Option<u64>,
    threads: Option<u64>,
    read_bytes: Option<u64>,
    write_bytes: Option<u64>,
}

impl ProcessResourceSnapshot {
    fn read() -> Self {
        #[cfg(target_os = "linux")]
        {
            return Self::read_linux();
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::default()
        }
    }

    fn format(&self) -> String {
        format!(
            "rss={} peak_rss={} process_threads={} read_bytes={} write_bytes={}",
            format_kb(self.rss_kb),
            format_kb(self.peak_rss_kb),
            format_count(self.threads),
            format_bytes(self.read_bytes),
            format_bytes(self.write_bytes),
        )
    }

    #[cfg(target_os = "linux")]
    fn read_linux() -> Self {
        let mut snapshot = Self::default();
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(value) = parse_status_kb(line, "VmRSS:") {
                    snapshot.rss_kb = Some(value);
                } else if let Some(value) = parse_status_kb(line, "VmHWM:") {
                    snapshot.peak_rss_kb = Some(value);
                } else if let Some(value) = parse_status_count(line, "Threads:") {
                    snapshot.threads = Some(value);
                }
            }
        }
        if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
            for line in io.lines() {
                if let Some(value) = parse_io_bytes(line, "read_bytes:") {
                    snapshot.read_bytes = Some(value);
                } else if let Some(value) = parse_io_bytes(line, "write_bytes:") {
                    snapshot.write_bytes = Some(value);
                }
            }
        }
        snapshot
    }
}

#[cfg(target_os = "linux")]
fn parse_status_kb(line: &str, key: &str) -> Option<u64> {
    let rest = line.strip_prefix(key)?.trim();
    rest.split_whitespace().next()?.parse().ok()
}

#[cfg(target_os = "linux")]
fn parse_status_count(line: &str, key: &str) -> Option<u64> {
    let rest = line.strip_prefix(key)?.trim();
    rest.split_whitespace().next()?.parse().ok()
}

#[cfg(target_os = "linux")]
fn parse_io_bytes(line: &str, key: &str) -> Option<u64> {
    let rest = line.strip_prefix(key)?.trim();
    rest.parse().ok()
}

fn format_count(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn format_kb(value: Option<u64>) -> String {
    value
        .map(|kb| format_bytes(Some(kb.saturating_mul(1024))))
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn format_bytes(value: Option<u64>) -> String {
    let Some(bytes) = value else {
        return "<unknown>".to_string();
    };
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.1}GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.1}MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.1}KiB", bytes_f / KIB)
    } else {
        format!("{bytes}B")
    }
}
