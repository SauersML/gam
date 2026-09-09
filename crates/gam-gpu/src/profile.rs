use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

const MAX_STATS: usize = 1024;

#[derive(Clone, Debug, Default)]
pub struct KernelStat {
    pub name: &'static str,
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub nnz: usize,
    pub flops_est: usize,
    pub bytes_est: usize,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct KernelStatsSnapshot {
    pub stats: Vec<KernelStat>,
}

/// One thread's dispatch ring: written only by the thread that owns it and read
/// by [`snapshot`] / [`clear`], so the lock the write path takes is one no other
/// thread holds while a dispatch is in flight.
type DispatchRing = Mutex<VecDeque<KernelStat>>;

/// How many rings one arena block holds. An allocation-granularity choice, not
/// a thread bound: the chain grows a block at a time and never stops.
const RING_BLOCK: usize = 64;

/// The append-only, process-lifetime home of every thread's ring.
///
/// A ring must outlive the thread that writes it — a worker pool's history is
/// exactly what a snapshot after the pool shuts down needs — and the reference
/// a thread keeps to its own ring must have no destructor, so that a dispatch
/// issued while that thread's other locals are being torn down still has
/// somewhere to go. Both follow from owning the rings in a `static`: a filled
/// slot is never emptied and a link is never cleared, so `&'static` comes from
/// the structure rather than from a leak.
struct RingArena {
    slots: [OnceLock<DispatchRing>; RING_BLOCK],
    next: OnceLock<Box<RingArena>>,
}

impl RingArena {
    const fn new() -> Self {
        Self {
            slots: [const { OnceLock::new() }; RING_BLOCK],
            next: OnceLock::new(),
        }
    }

    /// Take the first free slot, growing the chain when a block is full.
    ///
    /// A slot lost to a thread claiming concurrently is simply skipped; the
    /// walk carries on to the next one, so two threads never share a ring.
    fn claim(&'static self) -> &'static DispatchRing {
        let mut block: &'static RingArena = self;
        loop {
            for slot in block.slots.iter() {
                if slot.set(Mutex::new(VecDeque::new())).is_ok() {
                    return slot.get().expect("the slot was filled just above");
                }
            }
            block = block.next.get_or_init(|| Box::new(RingArena::new())).as_ref();
        }
    }

    /// Every ring handed out so far, oldest block first.
    fn claimed(&'static self) -> Vec<&'static DispatchRing> {
        let mut rings = Vec::new();
        let mut block: &'static RingArena = self;
        loop {
            for slot in block.slots.iter() {
                match slot.get() {
                    Some(ring) => rings.push(ring),
                    // Slots fill in order, so the first empty one ends the
                    // occupied prefix of this block.
                    None => return rings,
                }
            }
            match block.next.get() {
                Some(next) => block = next.as_ref(),
                None => return rings,
            }
        }
    }
}

static RING_ARENA: RingArena = RingArena::new();

thread_local! {
    /// This thread's ring, claimed on its first recorded dispatch.
    ///
    /// A `Cell<Option<&'static _>>` has no destructor, so this local is
    /// readable for as long as the thread runs — including while its other
    /// locals are being destroyed — and `with` has no failure case to swallow.
    static RING: Cell<Option<&'static DispatchRing>> = const { Cell::new(None) };
}

/// Record one dispatch attempt.
///
/// A DIAGNOSTIC MUST NOT SERIALIZE THE COMPUTATION IT OBSERVES. This ring is
/// written from the dense-product dispatch seam (`try_fast_ab`), which every
/// `fast_ab` in the workspace passes through — including the small products a
/// Rayon fan-out issues thousands of times per outer evaluation, and including
/// the ones the size gate declines before any device is consulted. Behind a
/// single process-wide `Mutex` those writes were not a diagnostic but a
/// serialization point: on the #979 rigid marginal-slope arm at 16 threads a
/// frame-pointer profile put 19.5 % of the whole run inside this function and a
/// further 14.8 % in `lock_contended` beneath it, and the arm's 40-minute wall
/// carried 325 minutes of system time against 188 of user time — sixteen
/// threads taking turns in the kernel to maintain a 1024-entry ring whose
/// contents at that call rate are the last microsecond of history.
///
/// Each thread therefore keeps its own ring and takes only its own lock. The
/// recorded SET is unchanged: every attempt, device-bound or not, is still
/// kept, and no ring is discarded when its thread ends. Only the interleaving
/// of different threads' entries changes, and a ring written by racing threads
/// never defined one.
pub fn record(stat: KernelStat) {
    RING.with(|cell| {
        let ring = match cell.get() {
            Some(ring) => ring,
            None => {
                let ring = RING_ARENA.claim();
                cell.set(Some(ring));
                ring
            }
        };
        if let Ok(mut guard) = ring.lock() {
            if guard.len() == MAX_STATS {
                guard.pop_front();
            }
            guard.push_back(stat);
        }
    });
}

/// Every recorded dispatch, from every thread that has recorded one.
pub fn snapshot() -> KernelStatsSnapshot {
    let mut stats = Vec::new();
    for ring in RING_ARENA.claimed() {
        if let Ok(guard) = ring.lock() {
            stats.extend(guard.iter().cloned());
        }
    }
    KernelStatsSnapshot { stats }
}

/// Empty every ring.
pub fn clear() {
    for ring in RING_ARENA.claimed() {
        if let Ok(mut guard) = ring.lock() {
            guard.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// GPU execution telemetry (issue #1017).
//
// The original `used_device: bool` could report `true` while the device had
// silently declined the workload and the solve ran on the CPU. A boolean
// cannot expose that: it carries no count of handles created, factorizations
// run, kernels launched, or — critically — CPU fallbacks taken and why. These
// per-thread counters make the resident solver's actual device activity
// auditable, so a silent fallback shows up as `cpu_fallback_count > 0` with a
// recorded reason rather than a lie. They are observability only and never
// change any numerical result.
// ---------------------------------------------------------------------------

use std::cell::RefCell;

/// Monotonic counters describing what the GPU-resident solver actually did on
/// the current thread. Snapshot with [`telemetry_snapshot`]; reset with
/// [`telemetry_reset`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuExecutionTelemetry {
    /// Bytes uploaded host→device.
    pub h2d_bytes: usize,
    /// Bytes read back device→host.
    pub d2h_bytes: usize,
    /// Cholesky / Schur factorizations performed on the device.
    pub factorization_count: usize,
    /// cuBLAS / cuSOLVER / stream handle creations.
    pub handle_creation_count: usize,
    /// Device kernel launches (per-row + border solves).
    pub kernel_launch_count: usize,
    /// Times a path that intended to use the device fell back to the CPU.
    pub cpu_fallback_count: usize,
    /// Human-readable reasons recorded alongside each CPU fallback.
    pub cpu_fallback_reasons: Vec<String>,
    /// Opaque context identifier of the device this thread last touched
    /// (e.g. the CUDA device ordinal), `0` when no device was used.
    pub context_id: usize,
}

thread_local! {
    static EXECUTION_TELEMETRY: RefCell<GpuExecutionTelemetry> =
        RefCell::new(GpuExecutionTelemetry::default());
}

/// Mutate the calling thread's execution telemetry in place.
#[inline]
pub fn telemetry_with<R>(f: impl FnOnce(&mut GpuExecutionTelemetry) -> R) -> R {
    EXECUTION_TELEMETRY.with(|cell| f(&mut cell.borrow_mut()))
}

/// Record a host→device upload of `bytes`.
#[inline]
pub fn telemetry_record_h2d(bytes: usize) {
    telemetry_with(|t| t.h2d_bytes += bytes);
}

/// Record a device→host readback of `bytes`.
#[inline]
pub fn telemetry_record_d2h(bytes: usize) {
    telemetry_with(|t| t.d2h_bytes += bytes);
}

/// Record a device factorization (POTRF / Schur factor).
#[inline]
pub fn telemetry_record_factorization() {
    telemetry_with(|t| t.factorization_count += 1);
}

/// Record creation of a device handle/stream and the context it bound.
#[inline]
pub fn telemetry_record_handle_creation(context_id: usize) {
    telemetry_with(|t| {
        t.handle_creation_count += 1;
        t.context_id = context_id;
    });
}

/// Record a device kernel launch.
#[inline]
pub fn telemetry_record_kernel_launch() {
    telemetry_with(|t| t.kernel_launch_count += 1);
}

/// Record a CPU fallback together with the reason it happened. This is the
/// counter that would have exposed the original silent-fallback bug.
#[inline]
pub fn telemetry_record_cpu_fallback(reason: impl Into<String>) {
    telemetry_with(|t| {
        t.cpu_fallback_count += 1;
        t.cpu_fallback_reasons.push(reason.into());
    });
}

/// Snapshot the calling thread's execution telemetry.
#[must_use]
pub fn telemetry_snapshot() -> GpuExecutionTelemetry {
    telemetry_with(|t| t.clone())
}

/// Reset the calling thread's execution telemetry to zero.
pub fn telemetry_reset() {
    telemetry_with(|t| *t = GpuExecutionTelemetry::default());
}

#[cfg(test)]
mod dispatch_ring_979_tests {
    use super::*;

    /// The registry is process-wide, and `cargo test` runs these cases on
    /// concurrent threads, so each one takes this first: without it a sibling's
    /// `clear` empties the ring another case is counting
    /// ([[a test verdict must not depend on which tests share the process]]).
    static EXCLUSIVE: Mutex<()> = Mutex::new(());

    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        EXCLUSIVE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn stat(name: &'static str, n: usize) -> KernelStat {
        KernelStat {
            name,
            n,
            ..Default::default()
        }
    }

    /// The ring is per thread, so the aggregate a consumer reads must still be
    /// every thread's attempts — the property one global ring gave by
    /// construction and this design has to provide explicitly.
    ///
    /// The threads record CONCURRENTLY and then exit, so this covers both
    /// registration of a live ring and the survival of a departed thread's
    /// ring. A design that registered a ring once and then overwrote it, or
    /// that let a worker pool's history vanish at shutdown, reports a
    /// truncated run and lands here.
    #[test]
    fn every_thread_dispatch_reaches_one_snapshot() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 32;
        let exclusive = exclusive();
        clear();
        // Hold every thread open until all of them have recorded, so the rings
        // genuinely coexist instead of being visited one after another by a
        // scheduler that serialises the spawns.
        let recorded = std::sync::Barrier::new(THREADS);
        std::thread::scope(|scope| {
            for thread in 0..THREADS {
                let recorded = &recorded;
                scope.spawn(move || {
                    for index in 0..PER_THREAD {
                        record(stat("ring_test", thread * PER_THREAD + index));
                    }
                    recorded.wait();
                });
            }
        });
        let recorded = snapshot().stats;
        assert_eq!(
            recorded.len(),
            THREADS * PER_THREAD,
            "every thread's dispatches must reach the snapshot"
        );
        let mut seen: Vec<usize> = recorded.iter().map(|entry| entry.n).collect();
        seen.sort_unstable();
        let expected: Vec<usize> = (0..THREADS * PER_THREAD).collect();
        assert_eq!(
            seen, expected,
            "no thread's dispatches may be lost or duplicated"
        );
        clear();
        drop(exclusive);
    }

    /// `clear` empties every ring, not only the caller's, and a dispatch after
    /// it is visible again.
    #[test]
    fn clear_empties_every_ring_and_recording_resumes() {
        let exclusive = exclusive();
        clear();
        std::thread::scope(|scope| {
            scope.spawn(|| record(stat("before_clear", 1)));
        });
        record(stat("before_clear", 2));
        assert_eq!(
            snapshot().stats.len(),
            2,
            "a departed thread's attempt and the caller's must both be visible"
        );
        clear();
        assert!(
            snapshot().stats.is_empty(),
            "clear must empty other threads' rings too, not only the caller's"
        );
        record(stat("after_clear", 3));
        let after = snapshot().stats;
        assert_eq!(after.len(), 1, "recording must resume after a clear");
        assert_eq!(after[0].n, 3);
        clear();
        drop(exclusive);
    }

    /// A ring is bounded: the oldest attempt is dropped, never the newest, so
    /// the diagnostic is the tail of the run and not a leak.
    #[test]
    fn a_rings_capacity_drops_the_oldest_attempt() {
        let exclusive = exclusive();
        clear();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for index in 0..(MAX_STATS + 16) {
                    record(stat("bounded", index));
                }
            });
        });
        let recorded = snapshot().stats;
        assert_eq!(
            recorded.len(),
            MAX_STATS,
            "the ring is capped at its capacity"
        );
        assert_eq!(recorded[0].n, 16, "the oldest attempts are the ones dropped");
        assert_eq!(
            recorded[MAX_STATS - 1].n,
            MAX_STATS + 15,
            "the newest attempt is kept"
        );
        clear();
        drop(exclusive);
    }
}
