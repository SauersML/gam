//! Device-resident residual-EM score and VJP kernels for torch interop.
//!
//! The CPU implementations in [`crate::criterion_atoms`] are the numerical
//! oracle.  This module mirrors those equations for CUDA tensors owned by an
//! external runtime: callers pass raw device addresses for contiguous
//! row-major buffers and the kernels write directly into caller-owned output
//! buffers.  No tensor ownership crosses the FFI boundary and no host staging
//! is involved.
//!
//! Synchronization contract: the caller must finish the stream that produced
//! every input before calling.  Each launch synchronizes its own stream before
//! returning, making its outputs visible to the caller's subsequent work.

/// Scalar type of the caller-owned CUDA tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidualEmCudaDType {
    /// IEEE-754 binary32 tensors.
    Float32,
    /// IEEE-754 binary64 tensors.
    Float64,
}

impl ResidualEmCudaDType {
    /// Parse the exact dtype names used by the Python torch bridge.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            other => Err(format!(
                "residual-EM CUDA dtype must be 'float32' or 'float64', got {other:?}"
            )),
        }
    }
}

#[cfg(target_os = "linux")]
mod device {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaModule, PushKernelArg};

    use super::ResidualEmCudaDType;

    // Mirrors criterion_atoms::{residual_em_score,residual_em_score_vjp}.
    // One thread owns one (row, atom) pair and walks its feature dimension in
    // row-major order, preserving the CPU oracle's reduction order.  The
    // shared compile helper disables fused multiply-add contraction.
    const RESIDUAL_EM_KERNEL_SRC: &str = r#"
template <typename T>
__device__ void residual_em_forward_impl(
    const T* __restrict__ x,
    const T* __restrict__ recon,
    T* __restrict__ code,
    T* __restrict__ relative_residual,
    long long pairs,
    long long atoms,
    long long dim,
    long long nonneg)
{
    const long long pair = (long long)blockIdx.x * (long long)blockDim.x
        + (long long)threadIdx.x;
    if (pair >= pairs) return;
    const long long row = pair / atoms;
    const T* xrow = x + row * dim;
    const T* rrow = recon + pair * dim;
    T row_energy = (T)0;
    T rr = (T)0;
    T rx = (T)0;
    for (long long j = 0; j < dim; ++j) {
        const T xj = xrow[j];
        const T rj = rrow[j];
        row_energy += xj * xj;
        rr += rj * rj;
        rx += rj * xj;
    }
    const T floor = (T)1.0e-12;
    const T row_scale = row_energy >= floor ? row_energy : floor;
    const T denom = rr >= floor ? rr : floor;
    const T s = rx / denom;
    const T c = nonneg != 0 ? (s >= (T)0 ? s : (T)0) : s;
    T resid = (T)0;
    for (long long j = 0; j < dim; ++j) {
        const T e = c * rrow[j] - xrow[j];
        resid += e * e;
    }
    code[pair] = c;
    relative_residual[pair] = resid / row_scale;
}

template <typename T>
__device__ void residual_em_vjp_impl(
    const T* __restrict__ x,
    const T* __restrict__ recon,
    const T* __restrict__ g_code,
    const T* __restrict__ g_relative_residual,
    T* __restrict__ grad_recon,
    long long pairs,
    long long atoms,
    long long dim,
    long long nonneg)
{
    const long long pair = (long long)blockIdx.x * (long long)blockDim.x
        + (long long)threadIdx.x;
    if (pair >= pairs) return;
    const long long row = pair / atoms;
    const T* xrow = x + row * dim;
    const T* rrow = recon + pair * dim;
    T* grow = grad_recon + pair * dim;
    T row_energy = (T)0;
    T rr = (T)0;
    T rx = (T)0;
    for (long long j = 0; j < dim; ++j) {
        const T xj = xrow[j];
        const T rj = rrow[j];
        row_energy += xj * xj;
        rr += rj * rj;
        rx += rj * xj;
    }
    const T floor = (T)1.0e-12;
    const T row_scale = row_energy >= floor ? row_energy : floor;
    const bool denom_active = rr >= floor;
    const T denom = denom_active ? rr : floor;
    const T s = rx / denom;
    const bool active = nonneg == 0 || s >= (T)0;
    const T c = nonneg != 0 ? (s >= (T)0 ? s : (T)0) : s;
    T e_dot_r = (T)0;
    for (long long j = 0; j < dim; ++j) {
        const T e = c * rrow[j] - xrow[j];
        e_dot_r += e * rrow[j];
    }
    const T two = (T)2;
    const T a = active
        ? g_code[pair] + two * g_relative_residual[pair] * e_dot_r / row_scale
        : (T)0;
    const T coeff_e = two * g_relative_residual[pair] * c / row_scale;
    for (long long j = 0; j < dim; ++j) {
        const T rj = rrow[j];
        const T ds_drj = denom_active
            ? (xrow[j] - two * s * rj) / denom
            : xrow[j] / denom;
        const T e = c * rj - xrow[j];
        grow[j] = a * ds_drj + coeff_e * e;
    }
}

extern "C" __global__ void sae_residual_em_score_f32(
    const float* x, const float* recon, float* code, float* relative_residual,
    long long pairs, long long atoms, long long dim, long long nonneg)
{
    residual_em_forward_impl<float>(x, recon, code, relative_residual,
                                    pairs, atoms, dim, nonneg);
}

extern "C" __global__ void sae_residual_em_score_f64(
    const double* x, const double* recon, double* code, double* relative_residual,
    long long pairs, long long atoms, long long dim, long long nonneg)
{
    residual_em_forward_impl<double>(x, recon, code, relative_residual,
                                     pairs, atoms, dim, nonneg);
}

extern "C" __global__ void sae_residual_em_score_vjp_f32(
    const float* x, const float* recon, const float* g_code,
    const float* g_relative_residual, float* grad_recon,
    long long pairs, long long atoms, long long dim, long long nonneg)
{
    residual_em_vjp_impl<float>(x, recon, g_code, g_relative_residual,
                                grad_recon, pairs, atoms, dim, nonneg);
}

extern "C" __global__ void sae_residual_em_score_vjp_f64(
    const double* x, const double* recon, const double* g_code,
    const double* g_relative_residual, double* grad_recon,
    long long pairs, long long atoms, long long dim, long long nonneg)
{
    residual_em_vjp_impl<double>(x, recon, g_code, g_relative_residual,
                                 grad_recon, pairs, atoms, dim, nonneg);
}
"#;

    fn module_for(ordinal: usize) -> Result<Arc<CudaModule>, String> {
        static MODULES: OnceLock<Mutex<HashMap<usize, Arc<CudaModule>>>> = OnceLock::new();
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal).ok_or_else(|| {
            format!("sae_residual_em_cuda: no CUDA context for ordinal {ordinal}")
        })?;
        let cache = MODULES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache
            .lock()
            .map_err(|err| format!("sae residual-EM CUDA module cache poisoned: {err}"))?;
        if let Some(module) = guard.get(&ordinal) {
            return Ok(module.clone());
        }
        let ptx = gam_gpu::device_cache::compile_ptx_arch(RESIDUAL_EM_KERNEL_SRC)
            .map_err(|err| format!("sae_residual_em_cuda: NVRTC compile: {err}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|err| format!("sae_residual_em_cuda: module load: {err}"))?;
        guard.insert(ordinal, module.clone());
        Ok(module)
    }

    fn validated_launch_shape(
        operation: &str,
        n: usize,
        atoms: usize,
        dim: usize,
    ) -> Result<(usize, i64, i64, i64), String> {
        if atoms == 0 || dim == 0 {
            return Err(format!(
                "{operation}: atoms and dim must be positive, got shape ({n}, {atoms}, {dim})"
            ));
        }
        let pairs = n
            .checked_mul(atoms)
            .ok_or_else(|| format!("{operation}: n*atoms overflows usize"))?;
        pairs
            .checked_mul(dim)
            .ok_or_else(|| format!("{operation}: n*atoms*dim overflows usize"))?;
        let pairs_ll =
            i64::try_from(pairs).map_err(|_| format!("{operation}: n*atoms overflows i64"))?;
        let atoms_ll =
            i64::try_from(atoms).map_err(|_| format!("{operation}: atoms overflows i64"))?;
        let dim_ll = i64::try_from(dim).map_err(|_| format!("{operation}: dim overflows i64"))?;
        Ok((pairs, pairs_ll, atoms_ll, dim_ll))
    }

    fn launch_config(
        operation: &str,
        pairs: usize,
    ) -> Result<cudarc::driver::LaunchConfig, String> {
        let block = 256_u32;
        let grid = u32::try_from(pairs.div_ceil(block as usize))
            .map_err(|_| format!("{operation}: launch grid overflows u32"))?;
        Ok(cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    /// Launch the device-resident residual-EM forward kernel.
    pub fn residual_em_score_device(
        ordinal: usize,
        dtype: ResidualEmCudaDType,
        x_dev_ptr: u64,
        recon_dev_ptr: u64,
        n: usize,
        atoms: usize,
        dim: usize,
        nonneg: bool,
        code_dev_ptr: u64,
        relative_residual_dev_ptr: u64,
    ) -> Result<(), String> {
        let operation = "sae_residual_em_score_device";
        let (pairs, pairs_ll, atoms_ll, dim_ll) = validated_launch_shape(operation, n, atoms, dim)?;
        if pairs == 0 {
            return Ok(());
        }
        if [
            x_dev_ptr,
            recon_dev_ptr,
            code_dev_ptr,
            relative_residual_dev_ptr,
        ]
        .contains(&0)
        {
            return Err(format!("{operation}: null device pointer"));
        }
        let module = module_for(ordinal)?;
        let kernel_name = match dtype {
            ResidualEmCudaDType::Float32 => "sae_residual_em_score_f32",
            ResidualEmCudaDType::Float64 => "sae_residual_em_score_f64",
        };
        let func = module
            .load_function(kernel_name)
            .map_err(|err| format!("{operation}: load_function {kernel_name}: {err}"))?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal)
            .ok_or_else(|| format!("{operation}: no CUDA context for ordinal {ordinal}"))?;
        let stream = ctx.default_stream();
        let nonneg_ll = if nonneg { 1_i64 } else { 0_i64 };
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_dev_ptr);
        builder.arg(&recon_dev_ptr);
        builder.arg(&code_dev_ptr);
        builder.arg(&relative_residual_dev_ptr);
        builder.arg(&pairs_ll);
        builder.arg(&atoms_ll);
        builder.arg(&dim_ll);
        builder.arg(&nonneg_ll);
        // SAFETY: the pointer/extents contract is checked by the torch bridge;
        // each launched thread writes one disjoint pair-sized output region.
        unsafe { builder.launch(launch_config(operation, pairs)?) }
            .map_err(|err| format!("{operation}: launch: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("{operation}: sync: {err}"))?;
        Ok(())
    }

    /// Launch the analytic device-resident residual-EM VJP kernel.
    pub fn residual_em_score_vjp_device(
        ordinal: usize,
        dtype: ResidualEmCudaDType,
        x_dev_ptr: u64,
        recon_dev_ptr: u64,
        n: usize,
        atoms: usize,
        dim: usize,
        nonneg: bool,
        g_code_dev_ptr: u64,
        g_relative_residual_dev_ptr: u64,
        grad_recon_dev_ptr: u64,
    ) -> Result<(), String> {
        let operation = "sae_residual_em_score_vjp_device";
        let (pairs, pairs_ll, atoms_ll, dim_ll) = validated_launch_shape(operation, n, atoms, dim)?;
        if pairs == 0 {
            return Ok(());
        }
        if [
            x_dev_ptr,
            recon_dev_ptr,
            g_code_dev_ptr,
            g_relative_residual_dev_ptr,
            grad_recon_dev_ptr,
        ]
        .contains(&0)
        {
            return Err(format!("{operation}: null device pointer"));
        }
        let module = module_for(ordinal)?;
        let kernel_name = match dtype {
            ResidualEmCudaDType::Float32 => "sae_residual_em_score_vjp_f32",
            ResidualEmCudaDType::Float64 => "sae_residual_em_score_vjp_f64",
        };
        let func = module
            .load_function(kernel_name)
            .map_err(|err| format!("{operation}: load_function {kernel_name}: {err}"))?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal)
            .ok_or_else(|| format!("{operation}: no CUDA context for ordinal {ordinal}"))?;
        let stream = ctx.default_stream();
        let nonneg_ll = if nonneg { 1_i64 } else { 0_i64 };
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_dev_ptr);
        builder.arg(&recon_dev_ptr);
        builder.arg(&g_code_dev_ptr);
        builder.arg(&g_relative_residual_dev_ptr);
        builder.arg(&grad_recon_dev_ptr);
        builder.arg(&pairs_ll);
        builder.arg(&atoms_ll);
        builder.arg(&dim_ll);
        builder.arg(&nonneg_ll);
        // SAFETY: see the forward launch; the VJP writes one disjoint D-vector
        // per pair into the caller-sized contiguous gradient allocation.
        unsafe { builder.launch(launch_config(operation, pairs)?) }
            .map_err(|err| format!("{operation}: launch: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("{operation}: sync: {err}"))?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
pub use device::{residual_em_score_device, residual_em_score_vjp_device};

/// Non-Linux typed refusal for the Linux-only raw CUDA forward lane.
#[cfg(not(target_os = "linux"))]
pub fn residual_em_score_device(
    ordinal: usize,
    dtype: ResidualEmCudaDType,
    x_dev_ptr: u64,
    recon_dev_ptr: u64,
    n: usize,
    atoms: usize,
    dim: usize,
    nonneg: bool,
    code_dev_ptr: u64,
    relative_residual_dev_ptr: u64,
) -> Result<(), String> {
    Err(format!(
        "sae_residual_em_score_device: CUDA path is Linux-only (ordinal {ordinal}, \
         dtype {dtype:?}, shape ({n}, {atoms}, {dim}), nonneg {nonneg}, \
         x@0x{x_dev_ptr:x}, recon@0x{recon_dev_ptr:x}, code@0x{code_dev_ptr:x}, \
         relative_residual@0x{relative_residual_dev_ptr:x})"
    ))
}

/// Non-Linux typed refusal for the Linux-only raw CUDA VJP lane.
#[cfg(not(target_os = "linux"))]
pub fn residual_em_score_vjp_device(
    ordinal: usize,
    dtype: ResidualEmCudaDType,
    x_dev_ptr: u64,
    recon_dev_ptr: u64,
    n: usize,
    atoms: usize,
    dim: usize,
    nonneg: bool,
    g_code_dev_ptr: u64,
    g_relative_residual_dev_ptr: u64,
    grad_recon_dev_ptr: u64,
) -> Result<(), String> {
    Err(format!(
        "sae_residual_em_score_vjp_device: CUDA path is Linux-only (ordinal {ordinal}, \
         dtype {dtype:?}, shape ({n}, {atoms}, {dim}), nonneg {nonneg}, \
         x@0x{x_dev_ptr:x}, recon@0x{recon_dev_ptr:x}, g_code@0x{g_code_dev_ptr:x}, \
         g_relative_residual@0x{g_relative_residual_dev_ptr:x}, \
         grad_recon@0x{grad_recon_dev_ptr:x})"
    ))
}

#[cfg(test)]
mod tests {
    use super::ResidualEmCudaDType;

    #[test]
    fn cuda_dtype_parser_is_exact() {
        assert_eq!(
            ResidualEmCudaDType::parse("float32"),
            Ok(ResidualEmCudaDType::Float32)
        );
        assert_eq!(
            ResidualEmCudaDType::parse("float64"),
            Ok(ResidualEmCudaDType::Float64)
        );
        assert!(ResidualEmCudaDType::parse("float16").is_err());
        assert!(ResidualEmCudaDType::parse("f64").is_err());
    }
}
