use super::device::{GpuDeviceInfo, GpuSelection, classify_capability};
use super::policy::{AccelPolicy, GpuDispatchPolicy, accel_policy_from_env};
use libloading::Library;
use std::ffi::c_char;
use std::sync::OnceLock;

type CuResult = i32;
type CuDevice = i32;
type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, i32, CuDevice) -> CuResult;
type CuDeviceComputeCapability = unsafe extern "C" fn(*mut i32, *mut i32, CuDevice) -> CuResult;
type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;

/// Cached runtime discovery result.
#[derive(Clone, Debug)]
pub struct GpuRuntime {
    pub selection: GpuSelection,
    pub policy: GpuDispatchPolicy,
}

impl GpuRuntime {
    #[inline]
    pub fn get() -> &'static Self {
        static RUNTIME: OnceLock<GpuRuntime> = OnceLock::new();
        RUNTIME.get_or_init(Self::probe)
    }

    pub fn probe() -> Self {
        let accel_policy = accel_policy_from_env();
        if matches!(accel_policy, AccelPolicy::CpuOnly) {
            return Self::cpu("GAM_GPU disabled acceleration".to_string(), accel_policy);
        }
        match probe_cuda_devices() {
            Ok(mut devices) if !devices.is_empty() => {
                devices.sort_by_key(|device| std::cmp::Reverse(device.score()));
                let selected = select_device(devices);
                let policy = GpuDispatchPolicy::for_device(accel_policy, Some(&selected));
                Self {
                    selection: GpuSelection::Cuda { device: selected },
                    policy,
                }
            }
            Ok(_) => Self::cpu(
                "CUDA driver loaded but reported no devices".to_string(),
                accel_policy,
            ),
            Err(err) => {
                if matches!(accel_policy, AccelPolicy::GpuOnly) {
                    log::warn!(
                        "[GAM GPU] GAM_GPU=force but CUDA probing failed: {err}; falling back to CPU"
                    );
                }
                Self::cpu(err, accel_policy)
            }
        }
    }

    #[inline]
    pub fn cuda_available(&self) -> bool {
        matches!(self.selection, GpuSelection::Cuda { .. })
    }

    #[inline]
    pub fn selected_device(&self) -> Option<&GpuDeviceInfo> {
        match &self.selection {
            GpuSelection::CpuOnly { .. } => None,
            GpuSelection::Cuda { device } => Some(device),
        }
    }

    fn cpu(reason: String, accel_policy: AccelPolicy) -> Self {
        Self {
            selection: GpuSelection::CpuOnly { reason },
            policy: GpuDispatchPolicy::for_device(accel_policy, None),
        }
    }
}

#[inline]
pub fn gpu_available() -> bool {
    GpuRuntime::get().cuda_available()
}

#[inline]
pub fn selected_gpu_info() -> Option<GpuDeviceInfo> {
    GpuRuntime::get().selected_device().cloned()
}

fn select_device(devices: Vec<GpuDeviceInfo>) -> GpuDeviceInfo {
    if let Ok(requested) = std::env::var("GAM_GPU_DEVICE") {
        if let Ok(ordinal) = requested.parse::<usize>() {
            if let Some(device) = devices.iter().find(|device| device.ordinal == ordinal) {
                return device.clone();
            }
            log::warn!(
                "[GAM GPU] requested GAM_GPU_DEVICE={ordinal} was not found; using best scored CUDA device"
            );
        }
    }
    devices
        .into_iter()
        .next()
        .expect("device list is non-empty")
}

fn probe_cuda_devices() -> Result<Vec<GpuDeviceInfo>, String> {
    let lib = load_cuda_driver()?;
    let cu_init: libloading::Symbol<'_, CuInit> =
        unsafe { lib.get(b"cuInit\0") }.map_err(|err| err.to_string())?;
    let cu_device_get_count: libloading::Symbol<'_, CuDeviceGetCount> =
        unsafe { lib.get(b"cuDeviceGetCount\0") }.map_err(|err| err.to_string())?;
    let cu_device_get: libloading::Symbol<'_, CuDeviceGet> =
        unsafe { lib.get(b"cuDeviceGet\0") }.map_err(|err| err.to_string())?;
    let cu_device_get_name: libloading::Symbol<'_, CuDeviceGetName> =
        unsafe { lib.get(b"cuDeviceGetName\0") }.map_err(|err| err.to_string())?;
    let cu_device_compute_capability: libloading::Symbol<'_, CuDeviceComputeCapability> =
        unsafe { lib.get(b"cuDeviceComputeCapability\0") }.map_err(|err| err.to_string())?;
    let cu_device_total_mem = unsafe { lib.get::<CuDeviceTotalMem>(b"cuDeviceTotalMem_v2\0") }
        .or_else(|_| unsafe { lib.get::<CuDeviceTotalMem>(b"cuDeviceTotalMem\0") })
        .map_err(|err| err.to_string())?;

    check_cuda(unsafe { cu_init(0) }, "cuInit")?;
    let mut count = 0_i32;
    check_cuda(
        unsafe { cu_device_get_count(&mut count) },
        "cuDeviceGetCount",
    )?;
    let mut devices = Vec::with_capacity(count.max(0) as usize);
    for ordinal in 0..count.max(0) {
        let mut raw_device = 0_i32;
        check_cuda(
            unsafe { cu_device_get(&mut raw_device, ordinal) },
            "cuDeviceGet",
        )?;
        let mut name_bytes = [0_i8; 256];
        check_cuda(
            unsafe {
                cu_device_get_name(name_bytes.as_mut_ptr(), name_bytes.len() as i32, raw_device)
            },
            "cuDeviceGetName",
        )?;
        let mut major = 0_i32;
        let mut minor = 0_i32;
        check_cuda(
            unsafe { cu_device_compute_capability(&mut major, &mut minor, raw_device) },
            "cuDeviceComputeCapability",
        )?;
        let mut total_memory_bytes = 0_usize;
        check_cuda(
            unsafe { cu_device_total_mem(&mut total_memory_bytes, raw_device) },
            "cuDeviceTotalMem",
        )?;
        let name = c_name_to_string(&name_bytes);
        devices.push(GpuDeviceInfo {
            ordinal: ordinal as usize,
            name,
            compute_capability_major: major,
            compute_capability_minor: minor,
            total_memory_bytes,
            free_memory_bytes: None,
            capability: classify_capability(major, minor, total_memory_bytes),
        });
    }
    Ok(devices)
}

fn load_cuda_driver() -> Result<Library, String> {
    let candidates: &[&str] = if cfg!(target_os = "windows") {
        &["nvcuda.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcuda.dylib", "libcuda.dylib"]
    } else {
        &["libcuda.so.1", "libcuda.so"]
    };
    let mut errors = Vec::new();
    for candidate in candidates {
        match unsafe { Library::new(candidate) } {
            Ok(lib) => return Ok(lib),
            Err(err) => errors.push(format!("{candidate}: {err}")),
        }
    }
    Err(format!(
        "CUDA driver library not found ({})",
        errors.join("; ")
    ))
}

#[inline]
fn check_cuda(result: CuResult, call: &str) -> Result<(), String> {
    if result == 0 {
        Ok(())
    } else {
        Err(format!("{call} failed with CUDA driver code {result}"))
    }
}

fn c_name_to_string(bytes: &[i8]) -> String {
    let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let raw: Vec<u8> = bytes[..nul].iter().map(|&b| b as u8).collect();
    String::from_utf8_lossy(&raw).trim().to_string()
}
