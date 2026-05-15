use crate::gpu::device::GpuDeviceInfo;
use crate::gpu::policy::{GpuDispatchDecision, GpuDispatchPolicy, GpuOperation};
use std::env;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct GpuContext {
    pub device: GpuDeviceInfo,
    pub policy: GpuDispatchPolicy,
}

impl GpuContext {
    #[must_use]
    pub fn target_for(&self, op: GpuOperation) -> ExecutionTarget {
        match self.policy.decide(op, true) {
            GpuDispatchDecision::Gpu => ExecutionTarget::Cuda(self.clone()),
            GpuDispatchDecision::Cpu => ExecutionTarget::Cpu,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ExecutionTarget {
    Cpu,
    Cuda(GpuContext),
}

#[derive(Clone, Debug)]
pub struct GpuRuntime {
    context: Option<GpuContext>,
    message: String,
}

impl GpuRuntime {
    #[must_use]
    pub fn probe() -> Self {
        if matches!(
            env::var("GAM_GPU").ok().as_deref(),
            Some("off" | "0" | "false" | "cpu")
        ) {
            return Self {
                context: None,
                message: "disabled by GAM_GPU".to_string(),
            };
        }
        match probe_cuda_devices() {
            Ok(mut devices) if !devices.is_empty() => {
                let selected = select_device(&mut devices);
                let policy = GpuDispatchPolicy::from_env_and_device(Some(&selected));
                Self {
                    message: format!("selected {selected}"),
                    context: Some(GpuContext {
                        device: selected,
                        policy,
                    }),
                }
            }
            Ok(_) => Self {
                context: None,
                message: "CUDA loaded but reported zero devices".to_string(),
            },
            Err(err) => Self {
                context: None,
                message: err,
            },
        }
    }

    #[must_use]
    pub fn context(&self) -> Option<&GpuContext> {
        self.context.as_ref()
    }

    #[must_use]
    pub fn is_available(&self) -> bool {
        self.context.is_some()
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

#[must_use]
pub fn gpu_available() -> bool {
    runtime().is_available()
}

#[must_use]
pub fn selected_gpu_info() -> Option<GpuDeviceInfo> {
    runtime().context().map(|ctx| ctx.device.clone())
}

#[must_use]
pub fn runtime() -> &'static GpuRuntime {
    static RUNTIME: OnceLock<GpuRuntime> = OnceLock::new();
    RUNTIME.get_or_init(GpuRuntime::probe)
}

fn select_device(devices: &mut [GpuDeviceInfo]) -> GpuDeviceInfo {
    if let Ok(value) = env::var("GAM_GPU_DEVICE") {
        if let Ok(ordinal) = value.parse::<usize>() {
            if let Some(device) = devices.iter().find(|device| device.ordinal == ordinal) {
                return device.clone();
            }
        }
    }
    devices
        .iter()
        .max_by_key(|device| device.score())
        .expect("non-empty CUDA device list")
        .clone()
}

#[cfg(all(feature = "cuda", unix))]
fn probe_cuda_devices() -> Result<Vec<GpuDeviceInfo>, String> {
    cuda_loader::probe_cuda_devices()
}

#[cfg(not(all(feature = "cuda", unix)))]
fn probe_cuda_devices() -> Result<Vec<GpuDeviceInfo>, String> {
    Err("built without the cuda feature; CPU fallback active".to_string())
}

#[cfg(all(feature = "cuda", unix))]
mod cuda_loader {
    use super::GpuDeviceInfo;
    use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
    use std::ptr;

    type CuDevice = c_int;
    type CuResult = c_int;
    type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
    type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
    type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
    type CuDeviceComputeCapability =
        unsafe extern "C" fn(*mut c_int, *mut c_int, CuDevice) -> CuResult;
    type CuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    const RTLD_NOW: c_int = 2;
    const CUDA_SUCCESS: CuResult = 0;

    pub(super) fn probe_cuda_devices() -> Result<Vec<GpuDeviceInfo>, String> {
        let lib = open_cuda()?;
        let cu_init: CuInit = symbol(lib, "cuInit")?;
        let cu_device_get_count: CuDeviceGetCount = symbol(lib, "cuDeviceGetCount")?;
        let cu_device_get: CuDeviceGet = symbol(lib, "cuDeviceGet")?;
        let cu_device_get_name: CuDeviceGetName = symbol(lib, "cuDeviceGetName")?;
        let cu_device_compute_capability: CuDeviceComputeCapability =
            symbol(lib, "cuDeviceComputeCapability")?;
        let cu_mem_get_info: Option<CuMemGetInfo> = symbol(lib, "cuMemGetInfo_v2").ok();

        check(unsafe { cu_init(0) }, "cuInit")?;
        let mut count = 0;
        check(
            unsafe { cu_device_get_count(&mut count) },
            "cuDeviceGetCount",
        )?;
        let mut devices = Vec::new();
        for ordinal in 0..count {
            let mut device = 0;
            check(
                unsafe { cu_device_get(&mut device, ordinal) },
                "cuDeviceGet",
            )?;
            let mut name_buf = [0_i8; 256];
            check(
                unsafe {
                    cu_device_get_name(name_buf.as_mut_ptr(), name_buf.len() as c_int, device)
                },
                "cuDeviceGetName",
            )?;
            let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let mut major = 0;
            let mut minor = 0;
            check(
                unsafe { cu_device_compute_capability(&mut major, &mut minor, device) },
                "cuDeviceComputeCapability",
            )?;
            let (free_memory_bytes, total_memory_bytes) =
                if let Some(cu_mem_get_info) = cu_mem_get_info {
                    let mut free = 0;
                    let mut total = 0;
                    if unsafe { cu_mem_get_info(&mut free, &mut total) } == CUDA_SUCCESS {
                        (Some(free), Some(total))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };
            devices.push(GpuDeviceInfo {
                ordinal: ordinal as usize,
                name,
                compute_capability_major: major,
                compute_capability_minor: minor,
                total_memory_bytes,
                free_memory_bytes,
            });
        }
        Ok(devices)
    }

    fn open_cuda() -> Result<*mut c_void, String> {
        for name in ["libcuda.so.1", "libcuda.so"] {
            let c_name = CString::new(name).expect("static library name contains no NUL");
            let handle = unsafe { dlopen(c_name.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                return Ok(handle);
            }
        }
        Err("could not dynamically load libcuda.so; CPU fallback active".to_string())
    }

    fn symbol<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
        let c_name = CString::new(name).expect("static symbol name contains no NUL");
        let ptr = unsafe { dlsym(handle, c_name.as_ptr()) };
        if ptr.is_null() {
            return Err(format!("missing CUDA symbol {name}"));
        }
        let mut out = std::mem::MaybeUninit::<T>::uninit();
        unsafe {
            ptr::copy_nonoverlapping(
                (&ptr as *const *mut c_void).cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                std::mem::size_of::<T>(),
            );
            Ok(out.assume_init())
        }
    }

    fn check(result: CuResult, name: &str) -> Result<(), String> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!(
                "{name} failed with CUDA driver error code {result}"
            ))
        }
    }
}
