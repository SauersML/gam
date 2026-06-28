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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    #[test]
    fn default_interval_constant_matches_expectation() {
        assert_eq!(DEFAULT_LOOP_PROGRESS_INTERVAL_SECS, 25);
    }

    #[test]
    fn new_with_zero_interval_emits_on_first_tick() {
        let lp = LoopProgress::new(0);
        let called = AtomicBool::new(false);
        lp.tick(1, |_progress, _elapsed| {
            called.store(true, Ordering::Relaxed);
        });
        assert!(called.load(Ordering::Relaxed), "emit should be called with zero interval");
    }

    #[test]
    fn tick_accumulates_progress_across_calls() {
        let lp = LoopProgress::new(0);
        let last_seen = AtomicUsize::new(0);
        lp.tick(5, |progress, _| {
            last_seen.store(progress, Ordering::Relaxed);
        });
        assert_eq!(last_seen.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn tick_with_large_interval_does_not_emit_on_first_call() {
        // With a 1-hour interval the first tick will have elapsed < interval,
        // so emit should NOT be called (elapsed ≥ 0 but < 3600 seconds).
        // Use an intermediate small value: 3600 seconds is definitely not elapsed
        // in a unit test.
        let lp = LoopProgress::new(3600);
        let called = AtomicBool::new(false);
        lp.tick(1, |_, _| {
            called.store(true, Ordering::Relaxed);
        });
        // The first tick starts with last=0; elapsed is a small positive number;
        // 3_600_000_000_000 ns >> any realistic elapsed, so emit is skipped.
        assert!(!called.load(Ordering::Relaxed), "emit should not fire with 1-hour interval");
    }

    #[test]
    fn tick_delta_zero_still_works() {
        let lp = LoopProgress::new(0);
        lp.tick(0, |_, _| {}); // should not panic
    }
}
