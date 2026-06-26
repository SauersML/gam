//! Low-overhead progress ticker for long parallel loops.
//!
//! `LoopProgress::tick` advances a shared counter and lets exactly one
//! worker emit after each wall-clock interval. Callers own the log message
//! so units and totals stay local to the loop.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

pub const DEFAULT_LOOP_PROGRESS_INTERVAL_SECS: u64 = 25;

pub struct LoopProgress {
    started: Instant,
    last_emit_nanos: AtomicU64,
    interval_nanos: u64,
    progress: AtomicUsize,
}

impl LoopProgress {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            started: Instant::now(),
            last_emit_nanos: AtomicU64::new(0),
            interval_nanos: interval_secs.saturating_mul(1_000_000_000),
            progress: AtomicUsize::new(0),
        }
    }

    pub fn default_interval() -> Self {
        Self::new(DEFAULT_LOOP_PROGRESS_INTERVAL_SECS)
    }

    /// Advance the progress counter by `delta` and, if at least
    /// `interval` of wall time has passed since the last claimed print,
    /// invoke `emit(progress, elapsed_secs)` exactly once across all
    /// threads. The closure typically issues a `log::info!`.
    pub fn tick(&self, delta: usize, emit: impl FnOnce(usize, f64)) {
        let progress = self
            .progress
            .fetch_add(delta, Ordering::Relaxed)
            .saturating_add(delta);
        let elapsed = self.started.elapsed().as_nanos() as u64;
        let last = self.last_emit_nanos.load(Ordering::Relaxed);
        if elapsed < last.saturating_add(self.interval_nanos) {
            return;
        }
        if self
            .last_emit_nanos
            .compare_exchange(last, elapsed, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            emit(progress, elapsed as f64 / 1.0e9);
        }
    }
}
