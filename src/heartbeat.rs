use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(60);

static HEARTBEAT_STATE: OnceLock<Arc<HeartbeatState>> = OnceLock::new();

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

struct HeartbeatState {
    started: Instant,
    threads: Mutex<BTreeMap<String, ThreadSnapshot>>,
}

struct ThreadStack {
    id: String,
    name: Option<String>,
    stack: Vec<FrameSnapshot>,
}

pub(crate) struct HeartbeatGuard {
    active: bool,
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

impl HeartbeatState {
    fn update_thread(&self, thread: &ThreadStack) {
        let mut threads = self.threads.lock().expect("heartbeat registry poisoned");
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
        let threads = self.threads.lock().expect("heartbeat registry poisoned");
        let resource = ProcessResourceSnapshot::read();
        log::info!(
            "[heartbeat] elapsed={} {} active_threads={}",
            format_duration(self.started.elapsed()),
            resource.format(),
            threads.len(),
        );
        for (thread_id, thread) in threads.iter() {
            let thread_label = match &thread.name {
                Some(name) => format!("{thread_id}/{name}"),
                None => thread_id.clone(),
            };
            log::info!(
                "[heartbeat] stack thread={} depth={} updated_ago={}",
                thread_label,
                thread.stack.len(),
                format_duration(thread.updated.elapsed()),
            );
            for (idx, frame) in thread.stack.iter().enumerate() {
                log::info!(
                    "[heartbeat]   #{idx}: {} [{}]",
                    frame.label,
                    format_duration(frame.entered.elapsed()),
                );
            }
        }
    }
}

impl Drop for HeartbeatGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let state = heartbeat_state();
        THREAD_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.stack.pop();
            state.update_thread(&stack);
        });
    }
}

pub(crate) fn scope(label: impl Into<String>) -> HeartbeatGuard {
    let state = heartbeat_state();
    THREAD_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.stack.push(FrameSnapshot {
            label: label.into(),
            entered: Instant::now(),
        });
        state.update_thread(&stack);
    });
    HeartbeatGuard { active: true }
}

fn heartbeat_state() -> Arc<HeartbeatState> {
    HEARTBEAT_STATE
        .get_or_init(|| {
            let state = Arc::new(HeartbeatState {
                started: Instant::now(),
                threads: Mutex::new(BTreeMap::new()),
            });
            start_heartbeat_thread(Arc::clone(&state));
            state
        })
        .clone()
}

fn start_heartbeat_thread(state: Arc<HeartbeatState>) {
    let builder = thread::Builder::new().name("gam-heartbeat".to_string());
    match builder.spawn(move || {
        loop {
            thread::park_timeout(HEARTBEAT_INTERVAL);
            state.emit();
        }
    }) {
        Ok(handle) => drop(handle),
        Err(err) => log::warn!("failed to start heartbeat thread: {err}"),
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
