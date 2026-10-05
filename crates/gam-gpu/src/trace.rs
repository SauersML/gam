//! Named wall-time spans for profiling device work.
//!
//! Every span is an NVTX range on a CUDA host whose NVTX library loads, so `nsys profile` shows it
//! over the kernels it queued. Once [`time_stages`] is on, a span is also a host accumulator: it
//! synchronizes its device at both ends, so queued GPU work is charged to the span that issued it
//! (inclusive of nested spans). Off (the default) a span costs one NVTX call and never waits.
//! Totals are process-wide and summed over threads.

use crate::tensor::Device;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static TIMING: AtomicBool = AtomicBool::new(false);
static TOTALS: Mutex<BTreeMap<&'static str, (u64, f64)>> = Mutex::new(BTreeMap::new());

/// Turn the synchronizing host accumulators on or off for every later span.
pub fn time_stages(on: bool) {
    TIMING.store(on, Ordering::Relaxed);
}

/// Whether spans currently synchronize and accumulate.
#[must_use]
pub fn timing_stages() -> bool {
    TIMING.load(Ordering::Relaxed)
}

/// One stage's accumulated inclusive wall time.
#[derive(Clone, Debug, serde::Serialize)]
pub struct StageTotal {
    pub name: &'static str,
    pub count: u64,
    pub seconds: f64,
}

/// Every stage timed so far, by name.
#[must_use]
pub fn stage_totals() -> Vec<StageTotal> {
    let totals = TOTALS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    totals.iter().map(|(name, (count, seconds))| StageTotal { name, count: *count, seconds: *seconds }).collect()
}

/// Forget every accumulated stage.
pub fn reset_stage_totals() {
    TOTALS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
}

/// A live span; it ends when dropped.
pub struct Span<'a> {
    name: &'static str,
    timed: Option<(Option<&'a Device>, Instant)>,
    #[cfg(target_os = "linux")]
    nvtx: Option<cudarc::nvtx::Range>,
}

/// Open the span `name` over work queued on `device`.
#[must_use]
pub fn span<'a>(name: &'static str, device: &'a Device) -> Span<'a> {
    open(name, Some(device))
}

/// Open the span `name` over host work only (nothing to synchronize).
#[must_use]
pub fn host_span(name: &'static str) -> Span<'static> {
    open(name, None)
}

fn open<'a>(name: &'static str, device: Option<&'a Device>) -> Span<'a> {
    #[cfg(target_os = "linux")]
    let nvtx = nvtx_present().then(|| cudarc::nvtx::Event::message(name).range());
    let timed = timing_stages().then(|| {
        if let Some(device) = device {
            settle(device, name);
        }
        (device, Instant::now())
    });
    Span {
        name,
        timed,
        #[cfg(target_os = "linux")]
        nvtx,
    }
}

/// Run `work` inside the span `name` over `device`.
pub fn within<T>(name: &'static str, device: &Device, work: impl FnOnce() -> T) -> T {
    let span = span(name, device);
    let out = work();
    drop(span);
    out
}

/// Run host-only `work` inside the span `name`.
pub fn within_host<T>(name: &'static str, work: impl FnOnce() -> T) -> T {
    let span = host_span(name);
    let out = work();
    drop(span);
    out
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        if let Some((device, started)) = self.timed.take() {
            if let Some(device) = device {
                settle(device, self.name);
            }
            let seconds = started.elapsed().as_secs_f64();
            let mut totals = TOTALS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let entry = totals.entry(self.name).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += seconds;
        }
        #[cfg(target_os = "linux")]
        drop(self.nvtx.take());
    }
}

/// Wait for `device`; a failure is the stream's, so the next real operation reports it.
fn settle(device: &Device, name: &str) {
    if let Err(error) = device.synchronize() {
        log::warn!("[trace] {name}: device synchronize failed: {error}");
    }
}

/// Whether the NVTX library loads (checked once; an absent library would panic on first call).
#[cfg(target_os = "linux")]
fn nvtx_present() -> bool {
    static PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // SAFETY: only attempts to dlopen the NVTX library by its candidate names.
    *PRESENT.get_or_init(|| unsafe { cudarc::nvtx::sys::is_culib_present() })
}
