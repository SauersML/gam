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

static STATS: OnceLock<Mutex<VecDeque<KernelStat>>> = OnceLock::new();

fn stats() -> &'static Mutex<VecDeque<KernelStat>> {
    STATS.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_STATS)))
}

pub fn record(stat: KernelStat) {
    if let Ok(mut guard) = stats().lock() {
        if guard.len() == MAX_STATS {
            guard.pop_front();
        }
        guard.push_back(stat);
    }
}

pub fn snapshot() -> KernelStatsSnapshot {
    if let Ok(guard) = stats().lock() {
        KernelStatsSnapshot {
            stats: guard.iter().cloned().collect(),
        }
    } else {
        KernelStatsSnapshot::default()
    }
}

pub fn clear() {
    if let Ok(mut guard) = stats().lock() {
        guard.clear();
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
