//! Env-free autodetection of an installed CUDA driver.
//!
//! The runtime probes the driver API exactly once at first access:
//!
//! 1. dlopen the platform-specific driver library (`libcuda.so.1` on Linux,
//!    `nvcuda.dll` on Windows, `libcuda.dylib` on macOS / CUDA-on-Mac).
//! 2. Call `cuInit(0)` and `cuDeviceGetCount` to enumerate visible devices.
//! 3. Materialize a [`GpuDeviceInfo`] per device and pick the best score.
//!
//! Probe failure is silent: callers see [`GpuRuntime::is_available`] return
//! `false` and the dispatch policy stays unused. There are no environment
//! variables or CLI flags involved in any of this.

use std::ffi::c_char;
use std::fmt;
use std::sync::OnceLock;

use libloading::Library;

use super::diagnostics;
use super::device::GpuDeviceInfo;
use super::driver::CudaWorkingState;
use super::policy::DispatchPolicy;

// Minimal CUDA driver ABI surface required for autodetection.
type CuResult = i32;
type CuDevice = i32;
type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, i32, CuDevice) -> CuResult;
type CuDeviceComputeCapability = unsafe extern "C" fn(*mut i32, *mut i32, CuDevice) -> CuResult;
type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, CuDevice) -> CuResult;

// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT from cuda.h. Hard-coded
// because we resolve symbols dynamically and can't include the header.
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;

/// Reason that GPU probing failed; never surfaced to callers, only logged.
#[derive(Debug)]
pub enum GpuProbeError {
    /// `libcuda` could not be dlopen'd; the host has no NVIDIA driver.
    DriverLibraryMissing(String),
    /// `libcuda` was found but a required entry point is missing.
    MissingSymbol(&'static str),
    /// A driver call returned a non-zero error code.
    DriverCall { call: &'static str, code: CuResult },
    /// The driver reports zero usable devices.
    NoDevices,
}

impl fmt::Display for GpuProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverLibraryMissing(s) => write!(f, "CUDA driver library not found: {s}"),
            Self::MissingSymbol(s) => write!(f, "CUDA driver missing symbol: {s}"),
            Self::DriverCall { call, code } => {
                write!(f, "{call} returned CUDA driver error {code}")
            }
            Self::NoDevices => f.write_str("no CUDA devices reported by the driver"),
        }
    }
}

/// Cached probe outcome — either a selected device or CPU-only.
#[derive(Debug)]
pub struct GpuRuntime {
    selected_device: Option<GpuDeviceInfo>,
    policy: DispatchPolicy,
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
                let device = devices.into_iter().next().expect("non-empty");
                let policy = DispatchPolicy::for_device(Some(&device));
                diagnostics::log_cuda_enabled(&device, &policy);
                Self {
                    selected_device: Some(device),
                    policy,
                }
            }
            Err(err) => {
                diagnostics::log_cuda_disabled(&err.to_string());
                Self {
                    selected_device: None,
                    policy: DispatchPolicy::for_device(None),
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

    /// Workload-size policy for the selected device.
    #[inline]
    pub fn policy(&self) -> &DispatchPolicy {
        &self.policy
    }

    /// The one shared `(libcuda + DriverApi + context)` used by every
    /// library runtime. Lazily created on first call so probe-only
    /// callers (`gpu_available()`, `selected_gpu_info()`) don't pay the
    /// `cuCtxCreate` cost.
    ///
    /// Returns `None` when no CUDA device is selected, or when context
    /// creation itself fails.
    pub fn cuda_working_state(&self) -> Option<&'static CudaWorkingState> {
        static STATE: OnceLock<Option<CudaWorkingState>> = OnceLock::new();
        let device = self.selected_device.as_ref()?;
        STATE
            .get_or_init(|| CudaWorkingState::init(device.ordinal))
            .as_ref()
    }
}

/// Convenience: is a GPU available in this process?
#[inline]
pub fn gpu_available() -> bool {
    GpuRuntime::global().is_available()
}

/// Convenience: the selected device, if any.
#[inline]
pub fn selected_gpu_info() -> Option<GpuDeviceInfo> {
    GpuRuntime::global().selected_device().cloned()
}

fn probe_cuda_devices() -> Result<Vec<GpuDeviceInfo>, GpuProbeError> {
    let library = load_cuda_driver()?;

    let cu_init: libloading::Symbol<'_, CuInit> =
        unsafe { library.get(b"cuInit\0") }.map_err(|_| GpuProbeError::MissingSymbol("cuInit"))?;
    let cu_device_get_count: libloading::Symbol<'_, CuDeviceGetCount> =
        unsafe { library.get(b"cuDeviceGetCount\0") }
            .map_err(|_| GpuProbeError::MissingSymbol("cuDeviceGetCount"))?;
    let cu_device_get: libloading::Symbol<'_, CuDeviceGet> =
        unsafe { library.get(b"cuDeviceGet\0") }
            .map_err(|_| GpuProbeError::MissingSymbol("cuDeviceGet"))?;
    let cu_device_get_name: libloading::Symbol<'_, CuDeviceGetName> =
        unsafe { library.get(b"cuDeviceGetName\0") }
            .map_err(|_| GpuProbeError::MissingSymbol("cuDeviceGetName"))?;
    let cu_device_compute_capability: libloading::Symbol<'_, CuDeviceComputeCapability> =
        unsafe { library.get(b"cuDeviceComputeCapability\0") }
            .map_err(|_| GpuProbeError::MissingSymbol("cuDeviceComputeCapability"))?;
    let cu_device_total_mem: libloading::Symbol<'_, CuDeviceTotalMem> = unsafe {
        library
            .get(b"cuDeviceTotalMem_v2\0")
            .or_else(|_| library.get(b"cuDeviceTotalMem\0"))
    }
    .map_err(|_| GpuProbeError::MissingSymbol("cuDeviceTotalMem"))?;
    let cu_device_get_attribute: libloading::Symbol<'_, CuDeviceGetAttribute> =
        unsafe { library.get(b"cuDeviceGetAttribute\0") }
            .map_err(|_| GpuProbeError::MissingSymbol("cuDeviceGetAttribute"))?;

    check(unsafe { cu_init(0) }, "cuInit")?;
    let mut count: i32 = 0;
    check(
        unsafe { cu_device_get_count(&mut count) },
        "cuDeviceGetCount",
    )?;
    if count <= 0 {
        return Err(GpuProbeError::NoDevices);
    }

    let mut devices = Vec::with_capacity(count as usize);
    for ordinal in 0..count {
        let mut raw_device: CuDevice = 0;
        check(
            unsafe { cu_device_get(&mut raw_device, ordinal) },
            "cuDeviceGet",
        )?;
        let mut name_bytes = [0_i8; 256];
        let name_len =
            i32::try_from(name_bytes.len()).expect("CUDA device name buffer length must fit i32");
        check(
            unsafe {
                cu_device_get_name(name_bytes.as_mut_ptr() as *mut c_char, name_len, raw_device)
            },
            "cuDeviceGetName",
        )?;
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        check(
            unsafe { cu_device_compute_capability(&mut major, &mut minor, raw_device) },
            "cuDeviceComputeCapability",
        )?;
        let mut total_memory_bytes: usize = 0;
        check(
            unsafe { cu_device_total_mem(&mut total_memory_bytes, raw_device) },
            "cuDeviceTotalMem",
        )?;
        let mut sm_count: i32 = 0;
        check(
            unsafe {
                cu_device_get_attribute(
                    &mut sm_count,
                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    raw_device,
                )
            },
            "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)",
        )?;
        devices.push(GpuDeviceInfo {
            ordinal: ordinal as usize,
            name: c_name_to_string(&name_bytes),
            compute_capability_major: major,
            compute_capability_minor: minor,
            sm_count,
            total_memory_bytes,
        });
    }
    // The `Symbol` bindings go out of scope at function exit; they don't
    // carry destructors with side effects. `library` itself is a `&'static`
    // handle owned by the `OnceLock` inside `load_cuda_driver`.
    Ok(devices)
}

fn load_cuda_driver() -> Result<&'static Library, GpuProbeError> {
    static LIBRARY: OnceLock<Option<&'static Library>> = OnceLock::new();
    let slot = LIBRARY.get_or_init(|| {
        let candidates: &[&str] = if cfg!(target_os = "windows") {
            &["nvcuda.dll"]
        } else if cfg!(target_os = "macos") {
            &["/usr/local/cuda/lib/libcuda.dylib", "libcuda.dylib"]
        } else {
            &["libcuda.so.1", "libcuda.so"]
        };
        for candidate in candidates {
            // SAFETY: We never call dtors on the loaded library; the handle is
            // intentionally leaked for the process lifetime.
            if let Ok(lib) = unsafe { Library::new(candidate) } {
                return Some(Box::leak(Box::new(lib)));
            }
        }
        None
    });
    slot.ok_or_else(|| {
        GpuProbeError::DriverLibraryMissing(if cfg!(target_os = "windows") {
            "nvcuda.dll".to_string()
        } else if cfg!(target_os = "macos") {
            "libcuda.dylib".to_string()
        } else {
            "libcuda.so.1".to_string()
        })
    })
}

#[inline]
fn check(code: CuResult, call: &'static str) -> Result<(), GpuProbeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(GpuProbeError::DriverCall { call, code })
    }
}

fn c_name_to_string(bytes: &[i8]) -> String {
    let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let raw: Vec<u8> = bytes[..nul].iter().map(|&b| b as u8).collect();
    String::from_utf8_lossy(&raw).trim().to_string()
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
}
