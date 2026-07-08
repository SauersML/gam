//! Device-resident SAE basis evaluation for external-tensor (torch) interop.
//!
//! The torch manifold-SAE lane routes every forward/backward through the CPU
//! `basis_with_jet` FFI: device→host copy of the coordinates, a rayon CPU
//! evaluation, and a host→device copy of `(phi, jet)` — measured at 0.07–1.9
//! optimizer steps/s on a B200 while the GPU idles (the whole fit is the
//! bridge). This module is the GPU-resident sibling: the caller (the pyffi
//! torch bridge) passes RAW CUDA device pointers — the coordinates tensor and
//! two preallocated output tensors that torch owns — and a single NVRTC kernel
//! writes `phi` and `jet` in place. Zero host round-trips, no ownership
//! transfer across the boundary, and the basis MATH stays single-sourced in
//! this crate (the kernel mirrors [`crate::basis::PeriodicHarmonicEvaluator`]
//! exactly; the parity test pins them together).
//!
//! Synchronization contract: the caller must ensure the input tensor's
//! producing stream has completed before calling (the torch bridge issues a
//! `torch.cuda.synchronize()`), and this function synchronizes its own stream
//! before returning, so the outputs are globally visible to any stream the
//! caller reads them on afterwards.
//!
//! Safety contract: the pointers must be valid CUDA device allocations on
//! `ordinal`'s primary context (torch and cudarc both use the device primary
//! context, so allocations from either side share one address space), with
//! `t: n` doubles, `phi: n·(2H+1)` doubles and `jet: n·(2H+1)` doubles of
//! writable space (row-major; the `d = 1` jet `(n, m, 1)` is contiguous as
//! `(n, m)`).

#[cfg(target_os = "linux")]
mod device {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaFunction, CudaModule, PushKernelArg};

    /// Mirrors `PeriodicHarmonicEvaluator::evaluate` (basis.rs): column 0 is
    /// the constant, columns `2h-1`/`2h` are `sin(2πht)`/`cos(2πht)`, and the
    /// jet is the exact `t`-derivative. `--fmad=false` (the shared NVRTC
    /// options in `gam_gpu::device_cache`) keeps device rounding aligned with
    /// the CPU oracle.
    const PERIODIC_KERNEL_SRC: &str = r#"
extern "C" __global__ void sae_periodic_basis_with_jet(
    const double* __restrict__ t,
    double* __restrict__ phi,
    double* __restrict__ jet,
    long long n,
    long long num_harmonics)
{
    long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    const long long m = 2 * num_harmonics + 1;
    const double two_pi = 6.283185307179586476925286766559;
    const double ti = t[i];
    double* prow = phi + i * m;
    double* jrow = jet + i * m;
    prow[0] = 1.0;
    jrow[0] = 0.0;
    for (long long h = 1; h <= num_harmonics; ++h) {
        const double freq = two_pi * (double)h;
        const double angle = freq * ti;
        const double s = sin(angle);
        const double c = cos(angle);
        prow[2 * h - 1] = s;
        prow[2 * h] = c;
        jrow[2 * h - 1] = freq * c;
        jrow[2 * h] = -freq * s;
    }
}
"#;

    fn periodic_function(ordinal: usize) -> Result<(Arc<CudaModule>, CudaFunction), String> {
        static MODULES: OnceLock<Mutex<HashMap<usize, Arc<CudaModule>>>> = OnceLock::new();
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal).ok_or_else(|| {
            format!("sae_periodic_basis_with_jet_device: no CUDA context for ordinal {ordinal}")
        })?;
        let cache = MODULES.get_or_init(|| Mutex::new(HashMap::new()));
        let module = {
            let mut guard = cache
                .lock()
                .map_err(|err| format!("sae basis-gpu module cache poisoned: {err}"))?;
            match guard.get(&ordinal) {
                Some(module) => module.clone(),
                None => {
                    let ptx = gam_gpu::device_cache::compile_ptx_arch(PERIODIC_KERNEL_SRC)
                        .map_err(|err| {
                            format!("sae_periodic_basis_with_jet_device: NVRTC compile: {err}")
                        })?;
                    let module = ctx.load_module(ptx).map_err(|err| {
                        format!("sae_periodic_basis_with_jet_device: module load: {err}")
                    })?;
                    guard.insert(ordinal, module.clone());
                    module
                }
            }
        };
        let func = module.load_function("sae_periodic_basis_with_jet").map_err(|err| {
            format!("sae_periodic_basis_with_jet_device: load_function: {err}")
        })?;
        Ok((module, func))
    }

    /// Launch the periodic basis+jet kernel over `n` coordinates already on
    /// device `ordinal`, writing into caller-owned device buffers. See the
    /// module docs for the pointer and synchronization contracts.
    pub fn sae_periodic_basis_with_jet_device(
        ordinal: usize,
        t_dev_ptr: u64,
        n: usize,
        num_harmonics: usize,
        phi_dev_ptr: u64,
        jet_dev_ptr: u64,
    ) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        if num_harmonics == 0 {
            return Err(
                "sae_periodic_basis_with_jet_device: num_harmonics must be >= 1".to_string()
            );
        }
        if t_dev_ptr == 0 || phi_dev_ptr == 0 || jet_dev_ptr == 0 {
            return Err(
                "sae_periodic_basis_with_jet_device: null device pointer".to_string()
            );
        }
        let (_module, func) = periodic_function(ordinal)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal).ok_or_else(|| {
            format!("sae_periodic_basis_with_jet_device: no CUDA context for ordinal {ordinal}")
        })?;
        let stream = ctx.default_stream();
        let n_ll = i64::try_from(n)
            .map_err(|_| "sae_periodic_basis_with_jet_device: n overflows i64".to_string())?;
        let h_ll = i64::try_from(num_harmonics)
            .map_err(|_| "sae_periodic_basis_with_jet_device: H overflows i64".to_string())?;
        let block: u32 = 256;
        let grid: u32 = u32::try_from(n.div_ceil(block as usize))
            .map_err(|_| "sae_periodic_basis_with_jet_device: grid overflow".to_string())?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        // Raw device addresses travel as 8-byte scalars; the kernel signature
        // reinterprets them as `double*` (driver-API param buffers are untyped).
        builder.arg(&t_dev_ptr);
        builder.arg(&phi_dev_ptr);
        builder.arg(&jet_dev_ptr);
        builder.arg(&n_ll);
        builder.arg(&h_ll);
        // SAFETY: pointer validity and extents are the documented caller
        // contract; grid covers all n rows exactly once.
        unsafe { builder.launch(cfg) }
            .map_err(|err| format!("sae_periodic_basis_with_jet_device: launch: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("sae_periodic_basis_with_jet_device: sync: {err}"))?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
pub use device::sae_periodic_basis_with_jet_device;

/// Non-Linux hosts have no CUDA driver; the torch bridge falls back to the
/// CPU `basis_with_jet` path on the typed refusal, which echoes the request so
/// a misrouted call is diagnosable from the message alone.
#[cfg(not(target_os = "linux"))]
pub fn sae_periodic_basis_with_jet_device(
    ordinal: usize,
    t_dev_ptr: u64,
    n: usize,
    num_harmonics: usize,
    phi_dev_ptr: u64,
    jet_dev_ptr: u64,
) -> Result<(), String> {
    Err(format!(
        "sae_periodic_basis_with_jet_device: CUDA path is Linux-only (requested \
         ordinal {ordinal}, n {n}, H {num_harmonics}, t@0x{t_dev_ptr:x}, \
         phi@0x{phi_dev_ptr:x}, jet@0x{jet_dev_ptr:x})"
    ))
}
