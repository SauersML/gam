//! Process-wide liveness monitor with per-thread scope stacks.
//!
//! Each heartbeat (≈once/minute) reports three things a user watching a long
//! compute log actually needs:
//!   1. The currently-active operation — the label of the longest-running
//!      instrumented scope on any thread and how long it has been running, so a
//!      multi-minute silent window shows `active="BMS coord_corrections …" for 142s`
//!      instead of nothing.
//!   2. A TRUE busy signal — process-wide CPU utilization in cores-busy,
//!      computed from `/proc/self/stat` (utime+stime) deltas between heartbeats.
//!      A rayon fan-out saturating ~70 cores reads as `cpu=68.3 cores`, where
//!      the old `active_threads` counter only ever saw the handful of threads
//!      inside an instrumented `track_scope` (rayon workers are not) and so
//!      reported a misleading `0`.
//!   3. Progress — when a long scope registers a progress counter (via
//!      [`track_scope_with_progress`]) the heartbeat surfaces `progress=a/b (X%)`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// A shared progress counter a long-running scope can expose to the heartbeat.
///
/// A compute loop creates one via [`track_scope_with_progress`], then bumps
/// [`ScopeProgress::set`] / [`ScopeProgress::inc`] as it advances. The
/// heartbeat reads the current/total atomically and surfaces a percentage in
/// the active-scope line — no log-spam coupling between the loop and the
/// monitor cadence.
#[derive(Clone)]
pub struct ScopeProgress {
    inner: Arc<ProgressCounter>,
}

struct ProgressCounter {
    current: AtomicU64,
    total: AtomicU64,
}

impl ScopeProgress {
    /// Record progress as `current` out of `total` units (e.g. rows processed
    /// out of rows total). `total == 0` is treated as "total unknown".
    pub fn set(&self, current: u64, total: u64) {
        self.inner.current.store(current, Ordering::Relaxed);
        self.inner.total.store(total, Ordering::Relaxed);
    }

    /// Advance the current count by one, leaving the total unchanged.
    pub fn inc(&self) {
        self.inner.current.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (u64, u64) {
        (
            self.inner.current.load(Ordering::Relaxed),
            self.inner.total.load(Ordering::Relaxed),
        )
    }
}

#[derive(Clone)]
struct FrameSnapshot {
    label: String,
    entered: Instant,
    progress: Option<ScopeProgress>,
}

struct ThreadSnapshot {
    name: Option<String>,
    stack: Vec<FrameSnapshot>,
    updated: Instant,
}

struct ProcessMonitorState {
    started: Instant,
    threads: Mutex<BTreeMap<String, ThreadSnapshot>>,
    cpu: Mutex<CpuSampler>,
}

struct ThreadStack {
    id: String,
    name: Option<String>,
    stack: Vec<FrameSnapshot>,
}

pub struct ProcessScopeGuard;

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

        // TRUE busy signal: process-wide cores-busy averaged over the interval
        // since the last heartbeat, read from /proc/self/stat. Independent of
        // whether the busy threads happen to sit inside an instrumented scope,
        // so a rayon fan-out over ~70 cores reads as ~70 here. The old
        // `active_threads` counter (now `instrumented_threads`, kept as a
        // diagnostic) only ever counted threads inside a `track_scope` and so
        // reported a misleading 0 during rayon-heavy windows.
        let cpu = self
            .cpu
            .lock()
            .expect("process monitor cpu sampler poisoned")
            .sample();
        let instrumented_threads = threads.len();

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
            progress: Option<(u64, u64)>,
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
            // Surface the first progress counter found walking the stack from
            // the innermost frame outward (the innermost reporting scope is the
            // most specific "what is it doing right now").
            let progress = thread
                .stack
                .iter()
                .rev()
                .find_map(|frame| frame.progress.as_ref())
                .map(ScopeProgress::snapshot);
            phases.push(ThreadPhase {
                thread_label,
                depth: thread.stack.len(),
                updated_ago: thread.updated.elapsed(),
                deepest_label: deepest.label.as_str(),
                deepest_age: deepest.entered.elapsed(),
                progress,
            });
        }
        phases.sort_by(|a, b| b.deepest_age.cmp(&a.deepest_age));

        // Headline line: total elapsed, resource snapshot, true CPU busy
        // signal, and — front and center — the longest-running active scope so
        // a user instantly sees "what is it doing and for how long".
        let active = match phases.first() {
            Some(phase) => {
                let progress = match phase.progress {
                    Some((cur, total)) if total > 0 => {
                        let pct = (cur as f64 / total as f64) * 100.0;
                        format!(" progress={cur}/{total} ({pct:.0}%)")
                    }
                    Some((cur, _)) => format!(" progress={cur}/?"),
                    None => String::new(),
                };
                format!(
                    " active={:?} for {}{}",
                    phase.deepest_label,
                    format_duration(phase.deepest_age),
                    progress,
                )
            }
            None => " active=<idle>".to_string(),
        };

        log::info!(
            "[process-monitor] elapsed={} {} {} instrumented_threads={}{}",
            format_duration(self.started.elapsed()),
            resource.format(),
            cpu.format(),
            instrumented_threads,
            active,
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
            let progress = match phase.progress {
                Some((cur, total)) if total > 0 => {
                    let pct = (cur as f64 / total as f64) * 100.0;
                    format!(" progress={cur}/{total} ({pct:.0}%)")
                }
                Some((cur, _)) => format!(" progress={cur}/?"),
                None => String::new(),
            };
            log::info!(
                "[process-monitor] phase thread={} depth={} deepest={:?} in_frame={} updated_ago={}{}",
                phase.thread_label,
                phase.depth,
                phase.deepest_label,
                format_duration(phase.deepest_age),
                format_duration(phase.updated_ago),
                progress,
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

pub fn track_scope(label: impl Into<String>) -> ProcessScopeGuard {
    push_scope(label.into(), None)
}

/// Open a tracked scope that also exposes a live progress counter to the
/// heartbeat. The returned [`ScopeProgress`] is cheap to clone into worker
/// closures; bump it as the loop advances and the heartbeat will surface
/// `progress=a/b (X%)` on the active-scope line. The scope closes when the
/// returned guard is dropped, exactly like [`track_scope`].
pub fn track_scope_with_progress(
    label: impl Into<String>,
    total: u64,
) -> (ProcessScopeGuard, ScopeProgress) {
    let progress = ScopeProgress {
        inner: Arc::new(ProgressCounter {
            current: AtomicU64::new(0),
            total: AtomicU64::new(total),
        }),
    };
    let guard = push_scope(label.into(), Some(progress.clone()));
    (guard, progress)
}

fn push_scope(label: String, progress: Option<ScopeProgress>) -> ProcessScopeGuard {
    let state = process_monitor();
    THREAD_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.stack.push(FrameSnapshot {
            label,
            entered: Instant::now(),
            progress,
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
                cpu: Mutex::new(CpuSampler::new()),
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

/// Process-wide CPU utilization sampler.
///
/// On Linux it reads cumulative user+system CPU jiffies from `/proc/self/stat`
/// and, between consecutive heartbeats, computes the average number of cores
/// kept busy over the interval: `Δ(utime+stime)/clock_hz / Δwall`. The first
/// heartbeat has no prior sample and reports the busy figure as unknown.
struct CpuSampler {
    prev_total_ticks: Option<u64>,
    prev_wall: Option<Instant>,
    last_cores: Option<f64>,
}

impl CpuSampler {
    fn new() -> Self {
        Self {
            prev_total_ticks: None,
            prev_wall: None,
            last_cores: None,
        }
    }

    fn sample(&mut self) -> CpuSnapshot {
        let now = Instant::now();
        let ticks = read_self_cpu_ticks();
        let cores = match (ticks, self.prev_total_ticks, self.prev_wall) {
            (Some(ticks), Some(prev_ticks), Some(prev_wall)) => {
                let delta_ticks = ticks.saturating_sub(prev_ticks) as f64;
                let delta_wall = now.duration_since(prev_wall).as_secs_f64();
                let hz = clock_ticks_per_second();
                if delta_wall > 0.0 && hz > 0.0 {
                    let cores = delta_ticks / hz / delta_wall;
                    self.last_cores = Some(cores);
                    Some(cores)
                } else {
                    self.last_cores
                }
            }
            _ => None,
        };
        if let Some(ticks) = ticks {
            self.prev_total_ticks = Some(ticks);
            self.prev_wall = Some(now);
        }
        CpuSnapshot {
            cores,
            ncpu: available_parallelism(),
            window: PROCESS_MONITOR_INTERVAL,
        }
    }
}

struct CpuSnapshot {
    cores: Option<f64>,
    ncpu: Option<usize>,
    window: Duration,
}

impl CpuSnapshot {
    fn format(&self) -> String {
        match self.cores {
            Some(cores) => {
                let of = match self.ncpu {
                    Some(n) => format!("/{n}"),
                    None => String::new(),
                };
                format!(
                    "cpu={:.1}{} cores (avg over {})",
                    cores,
                    of,
                    format_duration(self.window),
                )
            }
            None => "cpu=<warming-up>".to_string(),
        }
    }
}

/// Cumulative user+system CPU jiffies for this process from `/proc/self/stat`.
///
/// `/proc/self/stat` is a single space-separated line; field 14 (`utime`) and
/// field 15 (`stime`) are the process's user/system time in clock ticks. The
/// process command name (field 2) is parenthesized and may itself contain
/// spaces, so we split after the final `)`.
#[cfg(target_os = "linux")]
fn read_self_cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after_comm = stat.rsplit_once(')')?.1;
    // After the closing ')' the remaining fields start at field 3 (state), so
    // utime is index 11 and stime is index 12 of the post-')' whitespace split.
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime.saturating_add(stime))
}

#[cfg(not(target_os = "linux"))]
fn read_self_cpu_ticks() -> Option<u64> {
    None
}

/// Clock ticks per second (`sysconf(_SC_CLK_TCK)`); on Linux this is almost
/// universally 100. We hard-pin 100 rather than linking libc just for the
/// sysconf call — the value is fixed at kernel build time and the standard
/// Linux ABI value is 100, which is what `/proc` times are reported in.
#[cfg(target_os = "linux")]
fn clock_ticks_per_second() -> f64 {
    100.0
}

#[cfg(not(target_os = "linux"))]
fn clock_ticks_per_second() -> f64 {
    0.0
}

fn available_parallelism() -> Option<usize> {
    thread::available_parallelism().ok().map(|n| n.get())
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
            Self::read_linux()
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

#[cfg(test)]
mod format_tests {
    use super::*;
    use std::time::Duration;

    // ── format_duration ───────────────────────────────────────────────────────

    #[test]
    fn format_duration_seconds_only() {
        assert_eq!(format_duration(Duration::from_secs(45)), "45s");
    }

    #[test]
    fn format_duration_minutes_and_seconds() {
        assert_eq!(format_duration(Duration::from_secs(90)), "1m30s");
    }

    #[test]
    fn format_duration_minutes_zero_seconds() {
        assert_eq!(format_duration(Duration::from_secs(120)), "2m00s");
    }

    #[test]
    fn format_duration_hours_minutes_seconds() {
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h01m01s");
    }

    #[test]
    fn format_duration_exactly_one_hour() {
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h00m00s");
    }

    #[test]
    fn format_duration_zero() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0s");
    }

    // ── format_count ─────────────────────────────────────────────────────────

    #[test]
    fn format_count_some_value() {
        assert_eq!(format_count(Some(42)), "42");
    }

    #[test]
    fn format_count_zero() {
        assert_eq!(format_count(Some(0)), "0");
    }

    #[test]
    fn format_count_none_is_unknown() {
        assert_eq!(format_count(None), "<unknown>");
    }

    // ── format_bytes ──────────────────────────────────────────────────────────

    #[test]
    fn format_bytes_none_is_unknown() {
        assert_eq!(format_bytes(None), "<unknown>");
    }

    #[test]
    fn format_bytes_small_bytes() {
        assert_eq!(format_bytes(Some(512)), "512B");
    }

    #[test]
    fn format_bytes_exactly_1_kib() {
        assert_eq!(format_bytes(Some(1024)), "1.0KiB");
    }

    #[test]
    fn format_bytes_kib_range() {
        assert_eq!(format_bytes(Some(2048)), "2.0KiB");
    }

    #[test]
    fn format_bytes_exactly_1_mib() {
        assert_eq!(format_bytes(Some(1024 * 1024)), "1.0MiB");
    }

    #[test]
    fn format_bytes_exactly_1_gib() {
        assert_eq!(format_bytes(Some(1024 * 1024 * 1024)), "1.0GiB");
    }

    #[test]
    fn format_bytes_gib_range() {
        assert_eq!(format_bytes(Some(2 * 1024 * 1024 * 1024)), "2.0GiB");
    }

    // ── format_kb ─────────────────────────────────────────────────────────────

    #[test]
    fn format_kb_none_is_unknown() {
        assert_eq!(format_kb(None), "<unknown>");
    }

    #[test]
    fn format_kb_converts_to_bytes_and_formats() {
        // 1024 kB = 1 MiB
        assert_eq!(format_kb(Some(1024)), "1.0MiB");
    }

    #[test]
    fn format_kb_small_value() {
        // 1 kB = 1024 bytes → "1.0KiB"
        assert_eq!(format_kb(Some(1)), "1.0KiB");
    }

    // ── parse_status_kb / parse_status_count / parse_io_bytes (Linux) ─────────

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_status_kb_valid_line() {
        assert_eq!(parse_status_kb("VmRSS:\t1234 kB", "VmRSS:"), Some(1234));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_status_kb_wrong_key_returns_none() {
        assert_eq!(parse_status_kb("VmRSS:\t1234 kB", "VmPeak:"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_status_count_valid_line() {
        assert_eq!(
            parse_status_count("voluntary_ctxt_switches:\t42", "voluntary_ctxt_switches:"),
            Some(42)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_io_bytes_valid_line() {
        assert_eq!(
            parse_io_bytes("read_bytes: 65536", "read_bytes:"),
            Some(65536)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_io_bytes_wrong_key_returns_none() {
        assert_eq!(parse_io_bytes("read_bytes: 65536", "write_bytes:"), None);
    }
}
