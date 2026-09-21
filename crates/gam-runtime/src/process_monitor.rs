//! Process-wide liveness monitor with per-thread scope stacks.
//!
//! Each heartbeat (≈once/minute) reports two things a user watching a long
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
//!
//! The monitor exists only once [`start`] has been called — the Python binding
//! calls it when the `gamfit` logger reaches DEBUG, the only level the monitor
//! writes at, and the CLI when its verbosity does. Importing the library starts
//! no thread. Until then [`track_scope`] records nothing.
//!
//! The monitor's state belongs to one process. A `fork()`ed child of a process
//! that started the monitor inherits the request but neither the heartbeat
//! thread nor a usable registry (a parent thread may have held its lock at the
//! fork), so the child's first use builds a fresh state and heartbeat of its
//! own and leaks the parent's.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
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

/// Whether [`start`] has been called in this process or in the process it was
/// forked from.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// The current process's monitor state. Lock-free for the same reason as the
/// worker pool's cell: a lock held at a `fork()` stays held in the child.
static PROCESS_MONITOR: AtomicPtr<ProcessMonitorState> = AtomicPtr::new(std::ptr::null_mut());

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
    pid: u32,
    started: Instant,
    threads: Mutex<BTreeMap<String, ThreadSnapshot>>,
    cpu: Mutex<CpuSampler>,
}

struct ThreadStack {
    id: String,
    name: Option<String>,
    stack: Vec<FrameSnapshot>,
}

/// Pops the scope [`track_scope`] pushed, if it pushed one.
pub struct ProcessScopeGuard {
    pushed: bool,
}

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
        // Every line carries the emitting process id. A parent and its spawned
        // fit child both run a monitor and can share one log; without the id their
        // lines interleave with no way to tell which RSS or CPU reading belongs to
        // which process (#2267: two `[process-monitor]` lines at one timestamp
        // reporting 89.1 MiB and 1.9 GiB).
        let pid = std::process::id();

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

        // Headline line: total elapsed, resource snapshot, true CPU busy
        // signal, and — front and center — the longest-running active scope so
        // a user instantly sees "what is it doing and for how long".
        let active = match phases.first() {
            Some(phase) => format!(
                " active={:?} for {}",
                phase.deepest_label,
                format_duration(phase.deepest_age),
            ),
            None => " active=<idle>".to_string(),
        };

        log::debug!(
            "[process-monitor] pid={pid} elapsed={} {} {} instrumented_threads={}{}",
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
            log::debug!(
                "[process-monitor][STALL] pid={pid} thread={} phase={:?} stuck={}",
                phase.thread_label,
                phase.deepest_label,
                format_duration(phase.deepest_age),
            );
        }

        // Compact per-thread phase summary: deepest frame label + age, capped.
        for phase in phases.iter().take(PROCESS_MONITOR_MAX_PHASE_LINES) {
            log::debug!(
                "[process-monitor] pid={pid} phase thread={} depth={} deepest={:?} in_frame={} updated_ago={}",
                phase.thread_label,
                phase.depth,
                phase.deepest_label,
                format_duration(phase.deepest_age),
                format_duration(phase.updated_ago),
            );
        }
        if phases.len() > PROCESS_MONITOR_MAX_PHASE_LINES {
            log::debug!(
                "[process-monitor] pid={pid} phase ... and {} more active thread(s) omitted",
                phases.len() - PROCESS_MONITOR_MAX_PHASE_LINES,
            );
        }
    }
}

impl Drop for ProcessScopeGuard {
    fn drop(&mut self) {
        if !self.pushed {
            return;
        }
        let state = process_monitor();
        THREAD_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.stack.pop();
            if let Some(state) = state {
                state.update_thread(&stack);
            }
        });
    }
}

/// Start the process monitor: from now on this process, and any process forked
/// from it, keeps a heartbeat thread and records [`track_scope`] frames.
pub fn start() {
    ENABLED.store(true, Ordering::Release);
    process_monitor();
}

/// Record `label` as the current thread's innermost operation until the guard
/// drops. A no-op while the monitor is not started.
pub fn track_scope(label: impl Into<String>) -> ProcessScopeGuard {
    let Some(state) = process_monitor() else {
        return ProcessScopeGuard { pushed: false };
    };
    THREAD_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.stack.push(FrameSnapshot {
            label: label.into(),
            entered: Instant::now(),
        });
        state.update_thread(&stack);
    });
    ProcessScopeGuard { pushed: true }
}

/// The current process's monitor, started on first use once [`start`] has been
/// called; `None` before that.
fn process_monitor() -> Option<&'static ProcessMonitorState> {
    if !ENABLED.load(Ordering::Acquire) {
        return None;
    }
    let pid = std::process::id();
    loop {
        let current = PROCESS_MONITOR.load(Ordering::Acquire);
        // SAFETY: a non-null pointer in `PROCESS_MONITOR` came from
        // `Box::into_raw` and is never freed, so it is valid for the rest of the
        // process.
        if let Some(state) = unsafe { current.as_ref() }
            && state.pid == pid
        {
            return Some(state);
        }
        let fresh = Box::into_raw(Box::new(ProcessMonitorState {
            pid,
            started: Instant::now(),
            threads: Mutex::new(BTreeMap::new()),
            cpu: Mutex::new(CpuSampler::new()),
        }));
        match PROCESS_MONITOR.compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire)
        {
            // A state left by the parent of a forked process is leaked: its
            // heartbeat thread is gone and its locks may be held by threads that
            // no longer exist.
            Ok(_) => {
                // SAFETY: `fresh` was just published and is never freed.
                let state: &'static ProcessMonitorState = unsafe { &*fresh };
                start_process_monitor_thread(state);
                return Some(state);
            }
            Err(_) => {
                // SAFETY: `fresh` came from `Box::into_raw` above and was not
                // published.
                drop(unsafe { Box::from_raw(fresh) });
            }
        }
    }
}

fn start_process_monitor_thread(state: &'static ProcessMonitorState) {
    let builder = thread::Builder::new().name("gam-process-monitor".to_string());
    match builder.spawn(move || {
        loop {
            thread::park_timeout(PROCESS_MONITOR_INTERVAL);
            state.emit();
        }
    }) {
        Ok(handle) => drop(handle),
        Err(err) => log::debug!("failed to start process monitor thread: {err}"),
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
/// and, between consecutive successful reads, computes the average number of
/// cores kept busy over the measured wall-clock window:
/// `Δ(utime+stime)/CLOCK_TICKS_PER_SECOND / Δwall`. The first heartbeat has no
/// prior sample and reports the busy figure as unknown.
struct CpuSampler {
    previous: Option<(u64, Instant)>,
}

impl CpuSampler {
    fn new() -> Self {
        Self { previous: None }
    }

    fn sample(&mut self) -> CpuSnapshot {
        let now = Instant::now();
        let ticks = read_self_cpu_ticks();
        let busy = match (ticks, self.previous) {
            (Some(ticks), Some((prev_ticks, prev_wall))) => busy_cores(
                ticks.saturating_sub(prev_ticks),
                now.duration_since(prev_wall),
            ),
            _ => None,
        };
        if let Some(ticks) = ticks {
            self.previous = Some((ticks, now));
        }
        CpuSnapshot {
            busy,
            ncpu: available_parallelism(),
        }
    }
}

/// Average cores kept busy by `delta_ticks` CPU jiffies over `window`; `None`
/// for an empty window, which has no average.
fn busy_cores(delta_ticks: u64, window: Duration) -> Option<(f64, Duration)> {
    let seconds = window.as_secs_f64();
    (seconds > 0.0).then(|| {
        (
            delta_ticks as f64 / CLOCK_TICKS_PER_SECOND / seconds,
            window,
        )
    })
}

struct CpuSnapshot {
    /// Average busy cores and the measured window they were averaged over.
    busy: Option<(f64, Duration)>,
    ncpu: Option<usize>,
}

impl CpuSnapshot {
    fn format(&self) -> String {
        match self.busy {
            Some((cores, window)) => {
                let of = match self.ncpu {
                    Some(n) => format!("/{n}"),
                    None => String::new(),
                };
                format!(
                    "cpu={:.1}{} cores (avg over {})",
                    cores,
                    of,
                    format_duration(window),
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

/// Clock ticks per second (`sysconf(_SC_CLK_TCK)`) that `/proc/self/stat`
/// reports `utime`/`stime` in. We hard-pin 100 rather than linking libc just
/// for the sysconf call: the value is fixed at kernel build time and the
/// standard Linux ABI value is 100. Off Linux no ticks are read, so it is
/// never used there.
const CLOCK_TICKS_PER_SECOND: f64 = 100.0;

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
                if let Some(value) = parse_proc_value(line, "VmRSS:") {
                    snapshot.rss_kb = Some(value);
                } else if let Some(value) = parse_proc_value(line, "VmHWM:") {
                    snapshot.peak_rss_kb = Some(value);
                } else if let Some(value) = parse_proc_value(line, "Threads:") {
                    snapshot.threads = Some(value);
                }
            }
        }
        if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
            for line in io.lines() {
                if let Some(value) = parse_proc_value(line, "read_bytes:") {
                    snapshot.read_bytes = Some(value);
                } else if let Some(value) = parse_proc_value(line, "write_bytes:") {
                    snapshot.write_bytes = Some(value);
                }
            }
        }
        snapshot
    }
}

/// The integer that follows `key` on a `/proc/self/{status,io}` line: the
/// first whitespace-separated token, so a trailing unit (`VmRSS:  1234 kB`)
/// is ignored and a bare counter (`read_bytes: 65536`) reads the same way.
#[cfg(target_os = "linux")]
fn parse_proc_value(line: &str, key: &str) -> Option<u64> {
    line.strip_prefix(key)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
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

    // ── parse_proc_value (Linux) ───────────────────────────────────────────────

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_proc_value_reads_the_number_before_a_unit() {
        assert_eq!(parse_proc_value("VmRSS:\t1234 kB", "VmRSS:"), Some(1234));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_proc_value_reads_a_bare_counter() {
        assert_eq!(parse_proc_value("Threads:\t42", "Threads:"), Some(42));
        assert_eq!(
            parse_proc_value("read_bytes: 65536", "read_bytes:"),
            Some(65536)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_proc_value_wrong_key_returns_none() {
        assert_eq!(parse_proc_value("VmRSS:\t1234 kB", "VmPeak:"), None);
        assert_eq!(parse_proc_value("read_bytes: 65536", "write_bytes:"), None);
        // Anchored at the line start: `cancelled_write_bytes:` is not `write_bytes:`.
        assert_eq!(
            parse_proc_value("cancelled_write_bytes: 7", "write_bytes:"),
            None
        );
    }

    // ── CPU busy window ──────────────────────────────────────────────────────

    #[test]
    fn busy_cores_averages_over_the_measured_window() {
        // 1_200 jiffies at 100 Hz is 12 CPU-seconds; over 3 s that is 4 cores.
        let window = Duration::from_secs(3);
        assert_eq!(busy_cores(1_200, window), Some((4.0, window)));
        assert_eq!(busy_cores(1_200, Duration::ZERO), None);
        let snapshot = CpuSnapshot {
            busy: busy_cores(1_200, window),
            ncpu: Some(8),
        };
        assert_eq!(snapshot.format(), "cpu=4.0/8 cores (avg over 3s)");
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    fn monitor_thread_count() -> usize {
        // Linux keeps a thread's name in a 16-byte `comm` buffer: at most 15
        // bytes of the name survive.
        let comm_name = &"gam-process-monitor"[..15];
        std::fs::read_dir("/proc/self/task")
            .map(|tasks| {
                tasks
                    .filter_map(Result::ok)
                    .filter(|task| {
                        std::fs::read_to_string(task.path().join("comm"))
                            .is_ok_and(|comm| comm.trim() == comm_name)
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    // The only test in this crate that calls `start`, so the "before" half sees
    // the process as it is at import.
    #[test]
    fn the_monitor_starts_only_when_asked() {
        assert!(!track_scope("before start").pushed);
        if cfg!(target_os = "linux") {
            assert_eq!(monitor_thread_count(), 0);
        }
        start();
        start();
        let guard = track_scope("after start");
        assert!(guard.pushed);
        drop(guard);
        if cfg!(target_os = "linux") {
            // The heartbeat names itself once it runs.
            let deadline = Instant::now() + Duration::from_secs(10);
            while monitor_thread_count() == 0 && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(monitor_thread_count(), 1, "one heartbeat per process");
        }
    }
}
