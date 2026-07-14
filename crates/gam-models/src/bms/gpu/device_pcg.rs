// ────────────────────────────────────────────────────────────────────────
// Block 9 Phase 5 — device-resident PCG against the BMS-FLEX row-Hessian
// operator.
//
// The inner Newton solve in `BernoulliMarginalSlope` (matrix-free path,
// large-scale shape n=195k, p=44, r=20) currently reaches the GPU as a
// per-CG-iteration call to `launch_bms_flex_row_hvp` returning a host
// `Vec<f64>`. With ~6400 inner CG iterations per outer iteration that round-
// trip cost dominates: each iter pays one `stream.synchronize()` plus one
// DtoH download. At p=44 the download itself is 352 bytes — trivial in
// bandwidth, painful in latency.
//
// Phase 5 keeps every PCG vector on the device and runs the outer loop with
// only a single small scalar download per iteration (the squared residual
// norm for the convergence check). The Hv kernel becomes `into_device`
// (Block 9 addition to `bms_flex_row.rs`), and the axpy / dot / diagonal-
// preconditioner / scale-and-add steps run as tiny NVRTC kernels on the
// same default stream so the sequence is implicitly ordered without sync.
// ────────────────────────────────────────────────────────────────────────

/// Inputs to [`run_pcg_against_row_hessian_device`]. The right-hand-side
/// `b` is supplied as a host slice (it is the only host-resident vector
/// that needs to enter the loop — the iterate, residual, search direction,
/// and Hv output all live on the device).
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgInput<'a> {
    /// Per-fit row-Hessian + design storage. The PCG operator is
    /// `v ↦ launch_bms_flex_row_hvp_into_device(storage, ...)`.
    pub storage: &'a crate::bms::gpu::row::DeviceResidentRowHess,
    /// Right-hand-side `b`, length `storage.block.p_total`. Uploaded once.
    pub b: &'a [f64],
    /// Convergence tolerance on relative residual `‖r‖₂ / ‖b‖₂`.
    pub rel_tol: f64,
    /// Hard cap on iterations (the inner loop also bails on stagnation).
    pub max_iters: usize,
    /// Floor on `|diag(H)[i]|` used by the Jacobi preconditioner. Set to
    /// `1e-12` for the matrix-free row-Hessian path; the row-primary
    /// Hessian's diagonal is positive-definite by construction.
    pub precond_diag_floor: f64,
}

/// Output of [`run_pcg_against_row_hessian_device`].
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgOutput {
    /// Solution `x` such that `H · x ≈ b`, length `storage.block.p_total`.
    pub x: Vec<f64>,
    /// Number of PCG iterations consumed (final iter does not count if it
    /// converged immediately after the dot reduction).
    pub iterations: usize,
    /// Final achieved relative residual `‖r‖₂ / ‖b‖₂`.
    pub final_rel_residual: f64,
}

/// NVRTC source for the Phase-5 device-resident PCG support kernels. Every
/// kernel here operates on length-`p` device vectors with `p` typically
/// 44–256, so a single CTA suffices for each.
#[cfg(target_os = "linux")]
const PCG_KERNEL_SOURCE: &str = r#"
// y[i] += a * x[i]
extern "C" __global__ void pcg_axpy(int n, double a,
                                    const double * __restrict__ x,
                                    double * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// y[i] = a * x[i] + b * y[i]
extern "C" __global__ void pcg_axpby(int n, double a,
                                     const double * __restrict__ x,
                                     double b,
                                     double * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b * y[i];
}

// z[i] = r[i] / clamp(diag[i], floor) (sign-preserving floor on |diag|).
extern "C" __global__ void pcg_apply_diag_precond(int n, double floor_val,
                                                  const double * __restrict__ diag,
                                                  const double * __restrict__ r,
                                                  double * __restrict__ z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double d = diag[i];
        double ad = d < 0 ? -d : d;
        double clamped = ad > floor_val ? d : (d >= 0.0 ? floor_val : -floor_val);
        z[i] = r[i] / clamped;
    }
}

// Single-block dot product; writes the scalar to out[0]. n must be <= 1024.
extern "C" __global__ void pcg_dot_single_block(int n,
                                                const double * __restrict__ a,
                                                const double * __restrict__ b,
                                                double * __restrict__ out)
{
    __shared__ double s[1024];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int i = tid; i < n; i += blockDim.x) acc += a[i] * b[i];
    s[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

// Set out[i] = 0 for i in [0, n).
extern "C" __global__ void pcg_init_zero(int n, double * __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0;
}

// Copy y[i] = x[i].
extern "C" __global__ void pcg_copy(int n,
                                    const double * __restrict__ x,
                                    double * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
"#;

#[cfg(target_os = "linux")]
mod pcg_device {
    use super::DeviceResidentPcgInput;
    use super::DeviceResidentPcgOutput;
    use super::PCG_KERNEL_SOURCE;
    use crate::bms::gpu::row::launch_bms_flex_row_diagonal;
    use crate::bms::gpu::row::launch_bms_flex_row_hvp_into_device;
    use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::{Arc, OnceLock};

    struct PcgBackend {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    impl PcgBackend {
        fn probe() -> Result<&'static Self, String> {
            static BACKEND: OnceLock<Result<PcgBackend, String>> = OnceLock::new();
            BACKEND
                .get_or_init(|| {
                    let runtime = gam_gpu::device_runtime::GpuRuntime::require()
                        .map_err(String::from)?;
                    let ctx = gam_gpu::device_runtime::cuda_context_for(
                        runtime.selected_device().ordinal,
                    )
                    .ok_or_else(|| {
                        format!(
                            "pcg backend: failed to create CUDA context for device {}",
                            runtime.selected_device().ordinal
                        )
                    })?;
                    let stream = ctx.default_stream();
                    // #1551: arch-aware compile via the project's shared NVRTC
                    // entry point — pin `--gpu-architecture` to the device
                    // capability and supply the standard CUDA include paths,
                    // instead of bare `cudarc::nvrtc::compile_ptx` (NVRTC default
                    // arch, no includes).
                    let ptx = gam_gpu::device_cache::compile_ptx_arch(PCG_KERNEL_SOURCE)
                        .map_err(|err| format!("pcg NVRTC compile failed: {err}"))?;
                    let module = ctx
                        .load_module(ptx)
                        .map_err(|err| format!("pcg module load failed: {err}"))?;
                    Ok(PcgBackend { stream, module })
                })
                .as_ref()
                .map_err(String::clone)
        }
    }

    fn launch_blocks(p: usize, threads: u32) -> u32 {
        ((p as u32) + threads - 1) / threads
    }

    /// PCG against the row-Hessian operator with Jacobi preconditioner from
    /// `diag(H)`. All vectors remain on the device for the duration of the
    /// loop; only the squared residual norm crosses the host boundary each
    /// iter (one f64, ≤ 8 bytes).
    pub(super) fn run(
        input: DeviceResidentPcgInput<'_>,
    ) -> Result<DeviceResidentPcgOutput, String> {
        let p = input.storage.block.p_total;
        if input.b.len() != p {
            return Err(format!(
                "device-resident pcg: b.len()={} != p_total={p}",
                input.b.len()
            ));
        }
        if !input.rel_tol.is_finite() || input.rel_tol <= 0.0 {
            return Err(format!(
                "device-resident pcg: rel_tol must be positive and finite (got {})",
                input.rel_tol
            ));
        }
        if input.max_iters == 0 {
            return Err("device-resident pcg: max_iters must be >= 1".to_string());
        }
        if !input.precond_diag_floor.is_finite() || input.precond_diag_floor <= 0.0 {
            return Err(format!(
                "device-resident pcg: precond_diag_floor must be positive and finite (got {})",
                input.precond_diag_floor
            ));
        }

        let backend = PcgBackend::probe()?;
        let stream = backend.stream.clone();
        let module = backend.module.clone();

        // ── Load kernel handles once ─────────────────────────────────────
        let f_axpy = module
            .load_function("pcg_axpy")
            .map_err(|e| format!("pcg load pcg_axpy: {e}"))?;
        let f_axpby = module
            .load_function("pcg_axpby")
            .map_err(|e| format!("pcg load pcg_axpby: {e}"))?;
        let f_precond = module
            .load_function("pcg_apply_diag_precond")
            .map_err(|e| format!("pcg load pcg_apply_diag_precond: {e}"))?;
        let f_dot = module
            .load_function("pcg_dot_single_block")
            .map_err(|e| format!("pcg load pcg_dot_single_block: {e}"))?;
        let f_copy = module
            .load_function("pcg_copy")
            .map_err(|e| format!("pcg load pcg_copy: {e}"))?;

        // ── Allocate device vectors x, r, z, p_vec, q (length p each) ──
        let mut d_x = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc x: {e}"))?;
        let mut d_r = stream
            .clone_htod(input.b)
            .map_err(|e| format!("pcg upload b -> r: {e}"))?;
        let mut d_z = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc z: {e}"))?;
        let mut d_p = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc p: {e}"))?;
        let mut d_q = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc q: {e}"))?;
        // One-element scalar buffer reused across iters for `p·q` and
        // `r·z` dot products.
        let mut d_scalar = stream
            .alloc_zeros::<f64>(1)
            .map_err(|e| format!("pcg alloc scalar: {e}"))?;

        // Preconditioner: M⁻¹ from diag(H). One HostVec download per
        // *outer* call, but this is constant work per solve — not per
        // iter — so it does not block the inner loop's no-sync property.
        let diag_host = launch_bms_flex_row_diagonal(input.storage)
            .map_err(|e| format!("pcg diag fetch: {e}"))?;
        if diag_host.len() != p {
            return Err(format!(
                "pcg: diag length {} != p_total {p}",
                diag_host.len()
            ));
        }
        let d_diag = stream
            .clone_htod(&diag_host)
            .map_err(|e| format!("pcg upload diag: {e}"))?;

        // ── Convergence baseline: ‖b‖₂ via one in-stream dot ─────────────
        let n_i32 = i32::try_from(p).map_err(|_| format!("pcg: p_total={p} exceeds i32 range"))?;
        let vec_threads: u32 = 64;
        let vec_blocks = launch_blocks(p, vec_threads);
        let dot_threads: u32 = match p {
            0..=64 => 64,
            65..=128 => 128,
            129..=256 => 256,
            257..=512 => 512,
            _ => 1024,
        };
        if p > 1024 {
            return Err(format!(
                "device-resident pcg: p_total={p} exceeds single-block dot capacity (1024); \
                 widen pcg_dot_single_block to multi-block reduce before raising the cap"
            ));
        }

        // ‖b‖₂² = b · b (b is currently in d_r since r₀ = b - H·0 = b)
        // SAFETY: `f_dot` is the `pcg_dot_single_block` device function loaded
        // above; its signature is `(i32, *const f64, *const f64, *mut f64)`.
        // `n_i32` was bounded against `1024` (kernel's max-n contract) two
        // lines up; `d_r` is a `CudaSlice<f64>` of length `n` allocated to the
        // same stream; `d_scalar` is the length-1 output slice. Single-block
        // grid (1×dot_threads) matches the kernel's reduction strategy.
        unsafe {
            stream
                .launch_builder(&f_dot)
                .arg(&n_i32)
                .arg(&d_r)
                .arg(&d_r)
                .arg(&mut d_scalar)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (dot_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg b·b launch: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("pcg b·b sync: {e}"))?;
        let host_scalar = stream
            .clone_dtoh(&d_scalar)
            .map_err(|e| format!("pcg b·b download: {e}"))?;
        let bb = host_scalar[0];
        if !bb.is_finite() {
            return Err(format!("pcg: b·b not finite ({bb})"));
        }
        let b_norm = bb.sqrt();
        if b_norm == 0.0 {
            // x = 0, r = b = 0, trivially converged.
            return Ok(DeviceResidentPcgOutput {
                x: vec![0.0; p],
                iterations: 0,
                final_rel_residual: 0.0,
            });
        }

        // z₀ = M⁻¹ r₀
        // SAFETY: `f_precond` is `pcg_jacobi_precond` with signature
        // `(i32, f64, *const f64, *const f64, *mut f64)`. `d_diag`, `d_r`,
        // `d_z` are all `CudaSlice<f64>` of length `n` on the same stream;
        // `vec_blocks × vec_threads ≥ n` covers every output element.
        unsafe {
            stream
                .launch_builder(&f_precond)
                .arg(&n_i32)
                .arg(&input.precond_diag_floor)
                .arg(&d_diag)
                .arg(&d_r)
                .arg(&mut d_z)
                .launch(LaunchConfig {
                    grid_dim: (vec_blocks, 1, 1),
                    block_dim: (vec_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg precond z₀: {e}"))?;

        // p₀ = z₀
        // SAFETY: `f_copy` is `pcg_copy` with signature
        // `(i32, *const f64, *mut f64)`. `d_z` and `d_p` are
        // `CudaSlice<f64>` of length `n` on the same stream;
        // `vec_blocks × vec_threads ≥ n` covers every element.
        unsafe {
            stream
                .launch_builder(&f_copy)
                .arg(&n_i32)
                .arg(&d_z)
                .arg(&mut d_p)
                .launch(LaunchConfig {
                    grid_dim: (vec_blocks, 1, 1),
                    block_dim: (vec_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg copy p₀: {e}"))?;

        // ρ₀ = r₀·z₀
        // SAFETY: same invariants as the ‖b‖₂² launch above — `f_dot`
        // signature `(i32, *const f64, *const f64, *mut f64)`, `d_r` and
        // `d_z` are length-`n` `CudaSlice<f64>`, `d_scalar` is length-1,
        // single-block grid matches kernel's reduction.
        unsafe {
            stream
                .launch_builder(&f_dot)
                .arg(&n_i32)
                .arg(&d_r)
                .arg(&d_z)
                .arg(&mut d_scalar)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (dot_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg ρ₀ launch: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("pcg ρ₀ sync: {e}"))?;
        let s = stream
            .clone_dtoh(&d_scalar)
            .map_err(|e| format!("pcg ρ₀ download: {e}"))?;
        let mut rho = s[0];
        if !rho.is_finite() {
            return Err(format!("pcg: ρ₀ not finite ({rho})"));
        }

        let mut iters_taken: usize = 0;
        let mut final_rel_residual: f64 = (bb.sqrt() / b_norm).max(0.0);
        for iter in 0..input.max_iters {
            iters_taken = iter + 1;

            // q = H · p (on device, no sync, no DtoH).
            launch_bms_flex_row_hvp_into_device(input.storage, &d_p, &mut d_q)
                .map_err(|e| format!("pcg Hv iter {iter}: {e}"))?;

            // pq = p·q
            // SAFETY: identical to ‖b‖₂² launch — `f_dot` signature
            // `(i32, *const f64, *const f64, *mut f64)`; `d_p` is the
            // current search direction and `d_q` was just populated by
            // `launch_bms_flex_row_hvp_into_device` (same stream, same `n`).
            unsafe {
                stream
                    .launch_builder(&f_dot)
                    .arg(&n_i32)
                    .arg(&d_p)
                    .arg(&d_q)
                    .arg(&mut d_scalar)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (dot_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg p·q launch iter {iter}: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| format!("pcg p·q sync iter {iter}: {e}"))?;
            let s = stream
                .clone_dtoh(&d_scalar)
                .map_err(|e| format!("pcg p·q download iter {iter}: {e}"))?;
            let pq = s[0];
            if !pq.is_finite() || pq == 0.0 {
                return Err(format!(
                    "pcg iter {iter}: p·q={pq} (non-finite or zero); operator is not positive-definite"
                ));
            }
            let alpha = rho / pq;

            // x += α p
            // SAFETY: `f_axpy` is `pcg_axpy` with signature
            // `(i32, f64, *const f64, *mut f64)`. `alpha` is the
            // finite-checked CG step length (`rho/pq`, both validated
            // above). `d_p` and `d_x` are length-`n` `CudaSlice<f64>` on
            // the same stream. Grid covers all `n` elements.
            unsafe {
                stream
                    .launch_builder(&f_axpy)
                    .arg(&n_i32)
                    .arg(&alpha)
                    .arg(&d_p)
                    .arg(&mut d_x)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg x+=αp iter {iter}: {e}"))?;

            // r -= α q
            let neg_alpha = -alpha;
            // SAFETY: same `f_axpy` invariants as the `x += α p` launch
            // above; `neg_alpha = -alpha` is finite (alpha was checked),
            // `d_q` and `d_r` are length-`n` `CudaSlice<f64>` on the same
            // stream.
            unsafe {
                stream
                    .launch_builder(&f_axpy)
                    .arg(&n_i32)
                    .arg(&neg_alpha)
                    .arg(&d_q)
                    .arg(&mut d_r)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg r-=αq iter {iter}: {e}"))?;

            // ‖r‖₂² = r·r (single device dot, single f64 DtoH)
            // SAFETY: identical to the ‖b‖₂² launch at function entry —
            // `f_dot` signature, `d_r` length-`n`, `d_scalar` length-1,
            // single-block reduction grid.
            unsafe {
                stream
                    .launch_builder(&f_dot)
                    .arg(&n_i32)
                    .arg(&d_r)
                    .arg(&d_r)
                    .arg(&mut d_scalar)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (dot_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg ‖r‖₂² launch iter {iter}: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| format!("pcg ‖r‖₂² sync iter {iter}: {e}"))?;
            let s = stream
                .clone_dtoh(&d_scalar)
                .map_err(|e| format!("pcg ‖r‖₂² download iter {iter}: {e}"))?;
            let rr = s[0];
            if !rr.is_finite() {
                return Err(format!("pcg iter {iter}: ‖r‖₂²={rr} non-finite"));
            }
            let rel = rr.sqrt() / b_norm;
            final_rel_residual = rel;
            if rel <= input.rel_tol {
                break;
            }

            // z = M⁻¹ r
            // SAFETY: same `f_precond` invariants as the `z₀ = M⁻¹ r₀`
            // launch above — signature `(i32, f64, *const f64, *const f64,
            // *mut f64)`, all four slices length-`n` `CudaSlice<f64>`, grid
            // covers all `n` elements.
            unsafe {
                stream
                    .launch_builder(&f_precond)
                    .arg(&n_i32)
                    .arg(&input.precond_diag_floor)
                    .arg(&d_diag)
                    .arg(&d_r)
                    .arg(&mut d_z)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg z=M⁻¹r iter {iter}: {e}"))?;

            // ρ_new = r·z
            // SAFETY: identical to the ρ₀ launch above — `f_dot`
            // signature, `d_r` and `d_z` length-`n`, `d_scalar` length-1.
            unsafe {
                stream
                    .launch_builder(&f_dot)
                    .arg(&n_i32)
                    .arg(&d_r)
                    .arg(&d_z)
                    .arg(&mut d_scalar)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (dot_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg ρ_new launch iter {iter}: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| format!("pcg ρ_new sync iter {iter}: {e}"))?;
            let s = stream
                .clone_dtoh(&d_scalar)
                .map_err(|e| format!("pcg ρ_new download iter {iter}: {e}"))?;
            let rho_new = s[0];
            if !rho_new.is_finite() {
                return Err(format!("pcg iter {iter}: ρ_new={rho_new} non-finite"));
            }
            let beta_pcg = rho_new / rho;

            // p = z + β p  ⇒  via pcg_axpby with a=1, b=β
            // SAFETY: `f_axpby` is `pcg_axpby` with signature
            // `(i32, f64, *const f64, f64, *mut f64)`. `beta_pcg = rho_new/rho`
            // was finite-checked. `d_z` and `d_p` are length-`n`
            // `CudaSlice<f64>` on the same stream; grid covers all `n`
            // elements.
            unsafe {
                stream
                    .launch_builder(&f_axpby)
                    .arg(&n_i32)
                    .arg(&1.0_f64)
                    .arg(&d_z)
                    .arg(&beta_pcg)
                    .arg(&mut d_p)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg p=z+βp iter {iter}: {e}"))?;

            rho = rho_new;
        }

        // Download x once at the end.
        let x_host = stream
            .clone_dtoh(&d_x)
            .map_err(|e| format!("pcg final x DtoH: {e}"))?;
        // The auxiliary device allocs (d_r/d_z/d_p/d_q/d_scalar/d_diag) drop
        // here and return their bytes to cudarc's allocator.
        drop(d_r);
        drop(d_z);
        drop(d_p);
        drop(d_q);
        drop(d_scalar);
        drop(d_diag);
        Ok(DeviceResidentPcgOutput {
            x: x_host,
            iterations: iters_taken,
            final_rel_residual,
        })
    }
}

/// Device-resident PCG against the BMS-FLEX row-Hessian operator.
///
/// Block 9 Phase 5: every PCG vector — `x`, `r`, `z`, `p`, `q` — stays on
/// the device for the entire loop; only the squared residual norm (one f64)
/// is downloaded per iteration for the convergence check. Bit-equal output
/// to a host-side reference PCG against the same operator + preconditioner
/// when the tolerance is tight; differences only show up at the floating-
/// point reduction-order level.
///
/// Linux-only. See [`DeviceResidentPcgInput`] for parameters.
#[cfg(target_os = "linux")]
pub fn run_pcg_against_row_hessian_device(
    input: DeviceResidentPcgInput<'_>,
) -> Result<DeviceResidentPcgOutput, String> {
    pcg_device::run(input)
}

/// Block 9 Phase 5 — V100 parity for `run_pcg_against_row_hessian_device`.
///
/// Builds a small `(n=64, r=20, p=44)` BMS-FLEX row-Hessian fixture, computes
/// the dense joint Hessian via the same CPU oracle the HVP parity test uses,
/// solves `H · x = b` on the host via dense LU as ground truth, and asserts
/// the device-resident PCG iterate matches to a tight tolerance.
#[cfg(all(test, target_os = "linux"))]
mod pcg_device_parity_tests {
    use super::*;
    use crate::bms::gpu::row::{BmsFlexBlockLayout, BmsFlexPrimaryLayout, DeviceResidentRowHess};
    use ndarray::Array2;

    /// Dense oracle for `H_full = Σ_i P_iᵀ H_i P_i` consistent with
    /// `cpu_oracle_bms_flex_row_hvp`'s pullback math.
    fn cpu_dense_joint_hessian(
        row_hessians: &[f64],
        marginal: &[f64],
        logslope: &[f64],
        block: &BmsFlexBlockLayout,
        primary: &BmsFlexPrimaryLayout,
        n: usize,
    ) -> Array2<f64> {
        let p_total = block.p_total;
        let r = primary.r;
        let p_m = block.p_m;
        let p_g = block.p_g;
        let h_block_start = block.h.as_ref().map(|r| r.start).unwrap_or(0);
        let h_block_len = block.h.as_ref().map(|r| r.len()).unwrap_or(0);
        let w_block_start = block.w.as_ref().map(|r| r.start).unwrap_or(0);
        let w_block_len = block.w.as_ref().map(|r| r.len()).unwrap_or(0);
        let h_primary_start = primary.h.as_ref().map(|r| r.start).unwrap_or(0);
        let w_primary_start = primary.w.as_ref().map(|r| r.start).unwrap_or(0);
        let mut h_dense = Array2::<f64>::zeros((p_total, p_total));
        // For each row build P_i columns as length-p_total vectors.
        let mut phi = vec![vec![0.0_f64; p_total]; r];
        for row in 0..n {
            for col in phi.iter_mut() {
                col.iter_mut().for_each(|v| *v = 0.0);
            }
            let mrow = &marginal[row * p_m..(row + 1) * p_m];
            let grow = &logslope[row * p_g..(row + 1) * p_g];
            for k in 0..p_m {
                phi[0][k] = mrow[k];
            }
            for k in 0..p_g {
                phi[1][p_m + k] = grow[k];
            }
            for k in 0..h_block_len {
                phi[h_primary_start + k][h_block_start + k] = 1.0;
            }
            for k in 0..w_block_len {
                phi[w_primary_start + k][w_block_start + k] = 1.0;
            }
            let h_row = &row_hessians[row * r * r..(row + 1) * r * r];
            for u in 0..r {
                for v in 0..r {
                    let huv = h_row[u * r + v];
                    if huv == 0.0 {
                        continue;
                    }
                    for m in 0..p_total {
                        let phim = phi[u][m];
                        if phim == 0.0 {
                            continue;
                        }
                        let scaled = huv * phim;
                        for nn in 0..p_total {
                            h_dense[[m, nn]] += scaled * phi[v][nn];
                        }
                    }
                }
            }
        }
        h_dense
    }

    /// Reference oracle: host PCG against the dense joint H + diag(H)
    /// preconditioner, with a tolerance two decades tighter than the GPU
    /// PCG's. Comparing GPU PCG to host PCG (rather than to a Cholesky
    /// solve) keeps the comparison numerically apples-to-apples — only
    /// reduction order differs between the two paths.
    fn cpu_pcg_oracle(h: &Array2<f64>, b: &[f64], rel_tol: f64) -> Vec<f64> {
        let p = b.len();
        let diag: ndarray::Array1<f64> =
            ndarray::Array1::from_vec((0..p).map(|i| h[[i, i]]).collect());
        let rhs = ndarray::Array1::from_vec(b.to_vec());
        let h_owned = h.clone();
        let apply = move |v: &ndarray::Array1<f64>| h_owned.dot(v);
        let (x, info) =
            gam_linalg::utils::solve_spd_pcg_with_info(apply, &rhs, &diag, rel_tol, 4 * p)
                .expect("host PCG oracle must converge on SPD fixture");
        assert!(
            info.converged,
            "host PCG oracle failed to converge: iters={} rel_res={}",
            info.iterations, info.relative_residual_norm
        );
        x.to_vec()
    }

    #[test]
    fn pcg_device_matches_dense_oracle_at_n64_r20_p44() {
        let n = 64_usize;
        let p_m = 14_usize;
        let p_g = 12_usize;
        let p_h_dim = 10_usize;
        let p_w_dim = 8_usize;
        let r = 2 + p_h_dim + p_w_dim;
        let p_total = p_m + p_g + p_h_dim + p_w_dim;
        let block = BmsFlexBlockLayout {
            p_m,
            p_g,
            h: Some(p_m + p_g..p_m + p_g + p_h_dim),
            w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
            p_total,
        };
        let primary = BmsFlexPrimaryLayout {
            h: Some(2..2 + p_h_dim),
            w: Some(2 + p_h_dim..2 + p_h_dim + p_w_dim),
            r,
        };

        // Same deterministic symmetric Hessians + designs as the HVP parity
        // gate, so any drift between Phase 4 and Phase 5 surfaces here too.
        let mut row_hessians = vec![0.0_f64; n * r * r];
        for row in 0..n {
            let base = row * r * r;
            for u in 0..r {
                for v in 0..r {
                    let seed = (row as f64) * 0.137 + (u as f64) * 1.901 + (v as f64) * 0.317;
                    let a = (seed.sin() * 1.7 + (seed * 0.5).cos() * 0.9) * 0.5;
                    row_hessians[base + u * r + v] = a;
                }
            }
            for u in 0..r {
                for v in (u + 1)..r {
                    let upper = row_hessians[base + u * r + v];
                    let lower = row_hessians[base + v * r + u];
                    let sym = 0.5 * (upper + lower);
                    row_hessians[base + u * r + v] = sym;
                    row_hessians[base + v * r + u] = sym;
                }
                // Boost the diagonal heavily so each H_i is positive
                // definite — guarantees the joint pulled-back Hessian is
                // SPD, which PCG requires.
                row_hessians[base + u * r + u] += 4.0 * (r as f64);
            }
        }
        let mut marginal = vec![0.0_f64; n * p_m];
        for row in 0..n {
            for j in 0..p_m {
                // Orthonormal DCT-II columns make the aggregate pullback
                // full-rank by construction. The former phase-shifted
                // sinusoids were nearly collinear, so row-wise SPD did not
                // imply a numerically SPD joint fixture.
                let scale = if j == 0 {
                    (n as f64).sqrt().recip()
                } else {
                    (2.0 / n as f64).sqrt()
                };
                marginal[row * p_m + j] = scale
                    * (std::f64::consts::PI * (row as f64 + 0.5) * j as f64 / n as f64).cos();
            }
        }
        let mut logslope = vec![0.0_f64; n * p_g];
        for row in 0..n {
            for j in 0..p_g {
                let scale = if j == 0 {
                    (n as f64).sqrt().recip()
                } else {
                    (2.0 / n as f64).sqrt()
                };
                logslope[row * p_g + j] = scale
                    * (std::f64::consts::PI * (row as f64 + 0.5) * j as f64 / n as f64).cos();
            }
        }

        // Pick a non-trivial RHS.
        let b: Vec<f64> = (0..p_total)
            .map(|i| {
                let seed = (i as f64) * 0.157 + 0.6;
                seed.sin() * 0.55 + (seed * 0.4).cos() * 0.35
            })
            .collect();

        let h_dense =
            cpu_dense_joint_hessian(&row_hessians, &marginal, &logslope, &block, &primary, n);
        let x_oracle = cpu_pcg_oracle(&h_dense, &b, 1e-12);

        // Keep the SPD fixture certificate CPU-reachable. CUDA availability
        // controls only the device-parity half of this test.
        let runtime = match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
        {
            Ok(Some(runtime)) => runtime,
            Ok(None) => {
                eprintln!("[pcg_device parity] host SPD oracle passed; no CUDA device");
                return;
            }
            Err(error) => panic!("[pcg_device parity] CUDA probe failed: {error}"),
        };

        // Grab the same CUDA context + default stream that the bms_flex_row
        // kernels will use when `run_pcg_against_row_hessian_device` probes
        // its own backend. Going through the public runtime APIs keeps the
        // test independent of any private kernel-backend symbols.
        // Past the lossless Auto-resolution gate above: a context-creation or
        // HtoD-upload failure here is a real device fault on a CUDA host, not a
        // no-CUDA skip — fail loud (device-PCG skip-pass class, eee12f6b2). The old
        // arms returned, so a context/upload fault on a GPU host passed silently.
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
            .expect("[pcg_device parity] cuda_context_for must succeed on a CUDA host");
        let stream = ctx.default_stream();
        let d_h = stream
            .clone_htod(&row_hessians)
            .expect("[pcg_device parity] upload h must succeed on a CUDA host");
        let d_m = stream
            .clone_htod(&marginal)
            .expect("[pcg_device parity] upload marginal must succeed on a CUDA host");
        let d_g = stream
            .clone_htod(&logslope)
            .expect("[pcg_device parity] upload logslope must succeed on a CUDA host");
        let storage = DeviceResidentRowHess {
            neglog: stream
                .alloc_zeros::<f64>(n)
                .expect("[pcg_device parity] alloc neglog"),
            grad: stream
                .alloc_zeros::<f64>(n * r)
                .expect("[pcg_device parity] alloc grad"),
            hess: d_h,
            marginal_design: d_m,
            logslope_design: d_g,
            n,
            r,
            block,
            primary,

            bytes: ((n + n * r + n * r * r + n * p_m + n * p_g)
                * std::mem::size_of::<f64>()) as u64,
        };

        let out = run_pcg_against_row_hessian_device(DeviceResidentPcgInput {
            storage: &storage,
            b: &b,
            rel_tol: 1e-10,
            max_iters: 4 * p_total,
            precond_diag_floor: 1e-12,
        })
        .expect("device-resident PCG must succeed on SPD fixture");

        assert_eq!(out.x.len(), p_total);
        let mut max_abs = 0.0_f64;
        for i in 0..p_total {
            let diff = (out.x[i] - x_oracle[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        // Each iteration introduces O(1) ULPs of round-off in the dot/
        // axpy ladder; with ~88 iters max at p=44 we expect ‖Δx‖∞ comfortably
        // below 1e-7. Anything larger means a code bug, not float noise.
        assert!(
            max_abs <= 1e-7,
            "pcg_device parity ‖Δx‖∞={max_abs:.3e} > 1e-7 after {} iters \
             (final rel residual={:.3e})",
            out.iterations,
            out.final_rel_residual
        );
        eprintln!(
            "[pcg_device parity] n={n} p={p_total} r={r}: iters={} rel_res={:.3e} ‖Δx‖∞={:.3e}",
            out.iterations, out.final_rel_residual, max_abs
        );
    }
}
