//! Env-free autodetection of an installed CUDA driver via `cudarc` 0.19.
//!
//! The runtime probes the driver API exactly once at first access:
//!
//! 1. Ask `cudarc::driver::CudaContext::device_count()` for the visible
//!    device count. This implicitly initializes the driver.
//! 2. For each ordinal, retain the primary context via
//!    `CudaContext::new(ordinal)` and pull name / compute capability /
//!    total memory / SM count through the safe driver API.
//! 3. Materialize a [`GpuDeviceInfo`] per device, sort by score, and keep
//!    the full device set available for batched work partitioning.
//!
//! Probe failure is silent: callers see [`GpuRuntime::is_available`] return
//! `false` and the dispatch policy stays unused. There are no environment
//! variables or CLI flags involved in any of this.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::CudaContext;
use cudarc::driver::sys::CUdevice_attribute_enum;

use super::calibration::{DeviceCalibration, measure_device};
use super::device::GpuDeviceInfo;
use super::diagnostics;
use super::policy::DispatchPolicy;

/// Reason that GPU probing failed; never surfaced to callers, only logged.
#[derive(Debug)]
pub enum GpuProbeError {
    /// The CUDA driver could not be loaded or initialized on this host.
    DriverLibraryMissing(String),
    /// A cudarc safe-API call returned an error.
    DriverError(String),
    /// The driver reports zero usable devices.
    NoDevices,
    /// All enumerated devices failed runtime calibration (dgemm or memcpy).
    /// Callers fall through to CPU dispatch.
    CalibrationFailed,
}

impl fmt::Display for GpuProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverLibraryMissing(s) => write!(f, "CUDA driver library not found: {s}"),
            Self::DriverError(s) => write!(f, "CUDA driver call failed: {s}"),
            Self::NoDevices => f.write_str("no CUDA devices reported by the driver"),
            Self::CalibrationFailed => {
                f.write_str("all CUDA devices failed runtime calibration")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceBatchPlan {
    pub ordinal: usize,
    pub chunks: Vec<Range<usize>>,
}

/// Cached probe outcome — either a selected CUDA device set or CPU-only.
#[derive(Debug)]
pub struct GpuRuntime {
    selected_device: Option<GpuDeviceInfo>,
    devices: Vec<GpuDeviceInfo>,
    policy: DispatchPolicy,
    cpu_reason: Option<String>,
    dispatch_cursor: AtomicUsize,
}

impl GpuRuntime {
    /// Access the process-wide runtime. The probe runs at most once.
    pub fn global() -> &'static Self {
        static RUNTIME: OnceLock<GpuRuntime> = OnceLock::new();
        RUNTIME.get_or_init(Self::probe)
    }

    fn probe() -> Self {
        match probe_cuda_devices() {
            Ok(mut devices) => {
                debug_assert!(!devices.is_empty());
                devices.sort_by(|a, b| {
                    b.score()
                        .partial_cmp(&a.score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let device = devices[0].clone();
                let policy = DispatchPolicy::for_devices(&devices);
                diagnostics::log_cuda_enabled(&device, &policy);
                diagnostics::log_cuda_pool(&devices);
                Self {
                    selected_device: Some(device),
                    devices,
                    policy,
                    cpu_reason: None,
                    dispatch_cursor: AtomicUsize::new(0),
                }
            }
            Err(err) => {
                let reason = err.to_string();
                diagnostics::log_cuda_disabled(&reason);
                Self {
                    selected_device: None,
                    devices: Vec::new(),
                    policy: DispatchPolicy::for_device(None),
                    cpu_reason: Some(reason),
                    dispatch_cursor: AtomicUsize::new(0),
                }
            }
        }
    }

    /// True when a CUDA device was successfully selected.
    #[inline]
    pub fn is_available(&self) -> bool {
        self.selected_device.is_some()
    }

    /// Selected device descriptor, or `None` for CPU-only hosts.
    #[inline]
    pub fn selected_device(&self) -> Option<&GpuDeviceInfo> {
        self.selected_device.as_ref()
    }

    /// All CUDA devices visible to the process, sorted by dispatch preference.
    #[inline]
    pub fn devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    #[inline]
    pub fn device_by_ordinal(&self, ordinal: usize) -> Option<&GpuDeviceInfo> {
        self.devices.iter().find(|device| device.ordinal == ordinal)
    }

    /// Workload-size policy for the selected device.
    #[inline]
    pub fn policy(&self) -> &DispatchPolicy {
        &self.policy
    }

    /// CPU-only probe reason, or `None` when a CUDA device was selected.
    #[inline]
    pub fn cpu_reason(&self) -> Option<&str> {
        self.cpu_reason.as_deref()
    }

    pub fn plan_batched_work_for_devices(
        &self,
        devices: &[GpuDeviceInfo],
        batch_size: usize,
        fixed_bytes_per_device: usize,
        bytes_per_batch_item: usize,
    ) -> Option<Vec<DeviceBatchPlan>> {
        if batch_size == 0 {
            return Some(Vec::new());
        }
        if devices.is_empty() || bytes_per_batch_item == 0 {
            return None;
        }

        struct Candidate {
            ordinal: usize,
            score: f64,
            capacity: usize,
            chunks: Vec<Range<usize>>,
        }

        let mut candidates = devices
            .iter()
            .filter_map(|device| {
                let budget = device.dispatch_memory_budget_bytes();
                if budget <= fixed_bytes_per_device {
                    return None;
                }
                let capacity = (budget - fixed_bytes_per_device) / bytes_per_batch_item;
                if capacity == 0 {
                    return None;
                }
                Some(Candidate {
                    ordinal: device.ordinal,
                    score: device.score().max(1.0),
                    capacity,
                    chunks: Vec::new(),
                })
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next = 0usize;
        while next < batch_size {
            let total_score = candidates.iter().map(|c| c.score).sum::<f64>().max(1.0);
            let mut made_progress = false;
            for candidate in candidates.iter_mut() {
                if next >= batch_size {
                    break;
                }
                let remaining = batch_size - next;
                let proportional = ((remaining as f64) * candidate.score / total_score).ceil();
                let take = (proportional as usize)
                    .clamp(1, candidate.capacity)
                    .min(remaining);
                if take == 0 {
                    continue;
                }
                let start = next;
                next += take;
                candidate.chunks.push(start..next);
                made_progress = true;
            }
            if !made_progress {
                return None;
            }
        }

        let plans = candidates
            .into_iter()
            .filter(|candidate| !candidate.chunks.is_empty())
            .map(|candidate| DeviceBatchPlan {
                ordinal: candidate.ordinal,
                chunks: candidate.chunks,
            })
            .collect::<Vec<_>>();
        if plans.is_empty() { None } else { Some(plans) }
    }

    /// Pick a start slot for a multi-device runtime pool. Callers should try
    /// slots modulo their own pool length from this offset.
    #[inline]
    pub fn next_runtime_slot(&self, len: usize) -> usize {
        if len <= 1 {
            0
        } else {
            self.dispatch_cursor.fetch_add(1, Ordering::Relaxed) % len
        }
    }
}

/// Convenience: is a GPU available in this process?
#[inline]
pub fn gpu_available() -> bool {
    GpuRuntime::global().is_available()
}

/// Force the one-shot GPU probe to run *now* rather than lazily on first
/// dispatch. Call this once at process start so the `[GPU] CUDA acceleration
/// enabled` (or `disabled`) banner — and the calibration log lines — land at
/// the top of the run, not buried mid-fit. Idempotent.
pub fn warm() {
    let _ = GpuRuntime::global();
}

/// Convenience: the selected device, if any.
#[inline]
pub fn selected_gpu_info() -> Option<GpuDeviceInfo> {
    GpuRuntime::global().selected_device().cloned()
}

/// Convenience: every CUDA device visible to this process, sorted by dispatch
/// preference.
#[inline]
pub fn gpu_device_infos() -> Vec<GpuDeviceInfo> {
    GpuRuntime::global().devices().to_vec()
}

/// Shared `Arc<CudaContext>` for `ordinal`, or `None` if no such device.
///
/// The contexts are cached in a process-wide `OnceLock` so every cuBLAS /
/// cuSOLVER / cuSPARSE / session / calibration consumer reuses the same
/// primary context. Calling this before the probe has run will trigger
/// enumeration as a side effect, since contexts must be discovered to be
/// cached.
pub fn cuda_context_for(ordinal: usize) -> Option<Arc<CudaContext>> {
    contexts().get(ordinal).cloned()
}

/// Lazily enumerate and retain every visible primary context. Returns an
/// empty slice when no driver is present.
fn contexts() -> &'static [Arc<CudaContext>] {
    static CONTEXTS: OnceLock<Vec<Arc<CudaContext>>> = OnceLock::new();
    CONTEXTS.get_or_init(|| {
        let count = match CudaContext::device_count() {
            Ok(c) if c > 0 => c as usize,
            _ => return Vec::new(),
        };
        let mut out = Vec::with_capacity(count);
        for ordinal in 0..count {
            match CudaContext::new(ordinal) {
                Ok(ctx) => out.push(ctx),
                Err(err) => {
                    log::warn!(
                        "[GPU] CudaContext::new({}) failed: {}",
                        ordinal,
                        err
                    );
                }
            }
        }
        out
    })
}

fn probe_cuda_devices() -> Result<Vec<GpuDeviceInfo>, GpuProbeError> {
    let count = CudaContext::device_count().map_err(|err| {
        // No driver / unloadable libcuda surfaces as a DriverError here; we
        // map it to `DriverLibraryMissing` so the disabled-banner reads as
        // "no driver" rather than a noisy call error.
        GpuProbeError::DriverLibraryMissing(err.to_string())
    })?;
    if count <= 0 {
        return Err(GpuProbeError::NoDevices);
    }

    // Eagerly populate the shared-context cache so that every later caller
    // (`cuda_context_for`) sees the same `Arc<CudaContext>` we used here.
    let ctxs = contexts();
    if ctxs.is_empty() {
        return Err(GpuProbeError::DriverError(
            "no CUDA contexts could be retained".to_string(),
        ));
    }

    // ---- Phase 1: enumerate device descriptors -------------------------
    let mut descriptors: Vec<GpuDeviceDescriptor> = Vec::with_capacity(ctxs.len());
    for ctx in ctxs.iter() {
        let ordinal = ctx.ordinal();
        let name = ctx
            .name()
            .map_err(|e| GpuProbeError::DriverError(format!("cuDeviceGetName: {e}")))?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| GpuProbeError::DriverError(format!("compute_capability: {e}")))?;
        let sm_count = ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| GpuProbeError::DriverError(format!("cuDeviceGetAttribute(SM): {e}")))?;
        let total_memory_bytes = ctx
            .total_mem()
            .map_err(|e| GpuProbeError::DriverError(format!("cuDeviceTotalMem: {e}")))?;
        descriptors.push(GpuDeviceDescriptor {
            ordinal,
            name: name.trim().to_string(),
            compute_capability_major: major,
            compute_capability_minor: minor,
            sm_count,
            total_memory_bytes,
        });
    }

    // ---- Phase 2: calibrate each device --------------------------------
    // Run a real cublasDgemm + memcpy on every visible device and keep
    // only the ones that complete successfully. Calibration replaces the
    // earlier synthetic throughput proxy entirely — every dispatch
    // threshold derives from the measured numbers.
    let mut devices: Vec<GpuDeviceInfo> = Vec::with_capacity(descriptors.len());
    for desc in descriptors {
        let ctx = match cuda_context_for(desc.ordinal) {
            Some(ctx) => ctx,
            None => {
                log::warn!(
                    "[GPU] device {} '{}' skipped: context retain failed",
                    desc.ordinal,
                    desc.name
                );
                continue;
            }
        };
        let start = std::time::Instant::now();
        let calibration: DeviceCalibration = match measure_device(ctx) {
            Some(c) => c,
            None => {
                log::warn!(
                    "[GPU] device {} '{}' skipped: calibration failed",
                    desc.ordinal,
                    desc.name
                );
                continue;
            }
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "[GPU] device {} '{}' calibrated in {:.0} ms: fp64={:.0} GFLOPS h2d={:.1} GB/s d2h={:.1} GB/s",
            desc.ordinal,
            desc.name,
            elapsed_ms,
            calibration.fp64_gflops,
            calibration.h2d_gb_s,
            calibration.d2h_gb_s,
        );
        devices.push(GpuDeviceInfo {
            ordinal: desc.ordinal,
            name: desc.name,
            compute_capability_major: desc.compute_capability_major,
            compute_capability_minor: desc.compute_capability_minor,
            sm_count: desc.sm_count,
            total_memory_bytes: desc.total_memory_bytes,
            calibration,
        });
    }
    if devices.is_empty() {
        return Err(GpuProbeError::CalibrationFailed);
    }
    Ok(devices)
}

struct GpuDeviceDescriptor {
    ordinal: usize,
    name: String,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    sm_count: i32,
    total_memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_probe_is_idempotent_and_safe_without_driver() {
        // We cannot assume a driver is installed; just exercise the API.
        let first = GpuRuntime::global() as *const GpuRuntime;
        let second = GpuRuntime::global() as *const GpuRuntime;
        assert_eq!(first, second);
        // `gpu_available()` must return a stable answer.
        let a = gpu_available();
        let b = gpu_available();
        assert_eq!(a, b);
        if !a {
            assert!(selected_gpu_info().is_none());
        }
    }

    #[test]
    fn batch_planner_splits_by_capacity_and_score() {
        let runtime = GpuRuntime {
            selected_device: None,
            devices: Vec::new(),
            policy: DispatchPolicy::for_device(None),
            cpu_reason: None,
            dispatch_cursor: AtomicUsize::new(0),
        };
        let devices = vec![
            GpuDeviceInfo {
                ordinal: 0,
                name: String::new(),
                compute_capability_major: 9,
                compute_capability_minor: 0,
                sm_count: 132,
                total_memory_bytes: 80 * 1024 * 1024 * 1024,
                calibration: DeviceCalibration {
                    fp64_gflops: 30_000.0,
                    h2d_gb_s: 25.0,
                    d2h_gb_s: 25.0,
                },
            },
            GpuDeviceInfo {
                ordinal: 1,
                name: String::new(),
                compute_capability_major: 7,
                compute_capability_minor: 5,
                sm_count: 40,
                total_memory_bytes: 16 * 1024 * 1024 * 1024,
                calibration: DeviceCalibration {
                    fp64_gflops: 250.0,
                    h2d_gb_s: 12.0,
                    d2h_gb_s: 12.0,
                },
            },
        ];
        let plans = runtime
            .plan_batched_work_for_devices(&devices, 100, 0, 1024 * 1024 * 1024)
            .expect("large devices should accept 1GiB batch items");
        assert_eq!(
            plans
                .iter()
                .flat_map(|plan| plan.chunks.iter())
                .map(|range| range.end - range.start)
                .sum::<usize>(),
            100
        );
        assert!(plans.iter().any(|plan| plan.ordinal == 0));
        assert!(plans.iter().any(|plan| plan.ordinal == 1));
    }
}
