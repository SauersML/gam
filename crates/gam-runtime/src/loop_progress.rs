//! Low-overhead progress ticker for long parallel loops.
//!
//! `LoopProgress::tick` advances a shared counter and lets exactly one
//! worker emit after each wall-clock interval. Callers own the log message
//! so units and totals stay local to the loop. A loop that advances once per
//! item counts each chunk locally through `LoopProgress::chunk`, so the shared
//! counter and the clock are touched once per chunk.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

pub(crate) const DEFAULT_LOOP_PROGRESS_INTERVAL_SECS: u64 = 25;

fn elapsed_nanos(elapsed: std::time::Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

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
    ///
    /// Every call is a read-modify-write of the one counter all workers share
    /// plus a clock read, so it belongs once per chunk of work, never once per
    /// item of a parallel loop: per-row ticks over 12 threads were 13.4% of a
    /// closed-form survival fit's task-clock at n = 3·10⁵ (#2984). Per-item
    /// counting goes through [`Self::chunk`].
    pub fn tick(&self, delta: usize, emit: impl FnOnce(usize, f64)) {
        let progress = self
            .progress
            .fetch_add(delta, Ordering::Relaxed)
            .saturating_add(delta);
        let elapsed = elapsed_nanos(self.started.elapsed());
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

    /// One worker's count over one chunk of a loop. [`ChunkProgress::advance`]
    /// adds to a local count only; dropping the chunk hands that count to
    /// [`Self::tick`] once with `emit`, on every exit from the chunk's scope, an
    /// early `?` included, so the shared total never loses items.
    pub fn chunk<F: FnOnce(usize, f64)>(&self, emit: F) -> ChunkProgress<'_, F> {
        ChunkProgress {
            loop_progress: self,
            items: 0,
            emit: Some(emit),
        }
    }
}

/// A chunk's items counted locally and published to its [`LoopProgress`] when
/// the chunk is dropped; see [`LoopProgress::chunk`].
pub struct ChunkProgress<'a, F: FnOnce(usize, f64)> {
    loop_progress: &'a LoopProgress,
    items: usize,
    emit: Option<F>,
}

impl<F: FnOnce(usize, f64)> ChunkProgress<'_, F> {
    pub fn advance(&mut self, delta: usize) {
        self.items = self.items.saturating_add(delta);
    }
}

impl<F: FnOnce(usize, f64)> Drop for ChunkProgress<'_, F> {
    fn drop(&mut self) {
        if let Some(emit) = self.emit.take() {
            self.loop_progress.tick(self.items, emit);
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
        lp.tick(1, |_, _| {
            called.store(true, Ordering::Relaxed);
        });
        assert!(
            called.load(Ordering::Relaxed),
            "emit should be called with zero interval"
        );
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
        assert!(
            !called.load(Ordering::Relaxed),
            "emit should not fire with 1-hour interval"
        );
    }

    #[test]
    fn tick_delta_zero_still_works() {
        let lp = LoopProgress::new(0);
        let seen = AtomicUsize::new(usize::MAX);
        lp.tick(0, |progress, _| {
            seen.store(progress, Ordering::Relaxed);
        });
        // A zero-delta tick must not panic and leaves the counter at 0; the
        // zero interval still lets the single emit fire with progress 0.
        assert_eq!(
            seen.load(Ordering::Relaxed),
            0,
            "zero-delta tick must emit progress 0"
        );
    }

    #[test]
    fn elapsed_nanoseconds_saturate_instead_of_truncating() {
        assert_eq!(elapsed_nanos(std::time::Duration::MAX), u64::MAX);
    }

    #[test]
    fn a_chunk_publishes_its_items_once_when_it_is_dropped_2984() {
        let lp = LoopProgress::new(0);
        let emits = AtomicUsize::new(0);
        let seen = AtomicUsize::new(usize::MAX);
        {
            let mut chunk = lp.chunk(|progress, _| {
                emits.fetch_add(1, Ordering::Relaxed);
                seen.store(progress, Ordering::Relaxed);
            });
            for _ in 0..7 {
                chunk.advance(1);
            }
            assert_eq!(
                (emits.load(Ordering::Relaxed), lp.progress.load(Ordering::Relaxed)),
                (0, 0),
                "a chunk's items must stay local until the chunk is dropped"
            );
        }
        assert_eq!(
            (emits.load(Ordering::Relaxed), seen.load(Ordering::Relaxed)),
            (1, 7),
            "dropping the chunk must publish its 7 items in one emit"
        );
    }

    #[test]
    fn chunks_on_many_threads_and_early_exits_count_every_item_2984() {
        let lp = LoopProgress::new(3600);
        let emits = AtomicUsize::new(0);
        let (threads, chunks, items) = (12usize, 50usize, 100usize);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    for _ in 0..chunks {
                        let mut chunk = lp.chunk(|_, _| {
                            emits.fetch_add(1, Ordering::Relaxed);
                        });
                        for _ in 0..items {
                            chunk.advance(1);
                        }
                    }
                });
            }
        });
        let early_exit = |fail_after: usize| -> Result<(), usize> {
            let mut chunk = lp.chunk(|_, _| {
                emits.fetch_add(1, Ordering::Relaxed);
            });
            for item in 0..items {
                if item == fail_after {
                    return Err(item);
                }
                chunk.advance(1);
            }
            Ok(())
        };
        assert_eq!(early_exit(3), Err(3), "the chunk must leave through its early exit");
        assert_eq!(
            lp.progress.load(Ordering::Relaxed),
            threads * chunks * items + 3,
            "every advanced item, the 3 before an early exit included, must reach the shared total"
        );
        assert_eq!(
            emits.load(Ordering::Relaxed),
            0,
            "no chunk may emit before the one-hour interval has passed"
        );
    }
}
