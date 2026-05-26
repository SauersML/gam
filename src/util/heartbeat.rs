//! Periodic progress heartbeats for long parallel loops.
//!
//! A `Heartbeat` lets workers in a parallel region (rayon `for_each`,
//! `par_iter`, etc.) emit a single progress line every `interval` of wall
//! time without coordination. The first worker whose tick observes that
//! `interval` has elapsed since `last_print` claims the slot via a
//! `compare_exchange` on the stored elapsed-nanoseconds counter and prints;
//! losing workers do nothing extra.
//!
//! The check is one relaxed atomic load on the hot path and a CAS only on
//! the rare claim; both are far cheaper than a `log::info!` formatting
//! cost, so this is safe to call once per item.
//!
//! Callers print the chosen message themselves via the `try_emit` closure
//! so the heartbeat line carries the units (rows, blocks, cycles, …) and
//! totals that make sense at the call site.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Default heartbeat cadence. Picked to surface ~2-3 lines per minute on
/// long parallel loops so a stalled run is visible within ~30s and a
/// healthy biobank-scale loop still emits a bounded number of lines.
pub const DEFAULT_HEARTBEAT_INTERVAL_SECS: u64 = 25;

pub struct Heartbeat {
    started: Instant,
    last_print_nanos: AtomicU64,
    interval_nanos: u64,
    progress: AtomicUsize,
}

impl Heartbeat {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            started: Instant::now(),
            last_print_nanos: AtomicU64::new(0),
            interval_nanos: interval_secs.saturating_mul(1_000_000_000),
            progress: AtomicUsize::new(0),
        }
    }

    pub fn default_interval() -> Self {
        Self::new(DEFAULT_HEARTBEAT_INTERVAL_SECS)
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
        let last = self.last_print_nanos.load(Ordering::Relaxed);
        if elapsed < last.saturating_add(self.interval_nanos) {
            return;
        }
        if self
            .last_print_nanos
            .compare_exchange(last, elapsed, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            emit(progress, elapsed as f64 / 1.0e9);
        }
    }

    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
}
