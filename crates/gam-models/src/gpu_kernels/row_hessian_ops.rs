//! Block 9 Phase 2/3 — device kernels that consume the row-primary Hessian
//! cache (the per-row `r × r` blocks materialised by
//! [`crate::bms::BernoulliMarginalSlopeFamily::build_row_primary_hessian_cache`]
//! and stored in [`crate::bms::RowPrimaryEvalCache`])
//! and emit either:
//!
//! * **Phase 2 — per-row matvec** `y_i = H_i · v_i` for every row `i ∈ [0, n)`,
//!   matching CPU
//!   [`BernoulliMarginalSlopeFamily::exact_newton_joint_hessian_matvec_from_cache`]'s
//!   `scratch.hess.dot(&row_dir)` inner contraction; or
//! * **Phase 3 — per-row diagonal** `d_i = diag(H_i)` (the `r` diagonal
//!   entries), matching the cached-diagonal fast path in
//!   [`BernoulliMarginalSlopeFamily::exact_newton_joint_hessian_diagonal_from_cache`]'s
//!   `row_hess[[u, u]]` reads.
//!
//! Both kernels assume the cached layout produced by Phase 1 (FullRowMajor:
//! `n_rows × r × r` doubles, fully symmetric per row). The design-row pullback
//! (`marginal_design.axpy_row_into` / `logslope_design.axpy_row_into` /
//! `pullback_primary_vector`) stays on the host in Phase 2/3; Phase 5 will
//! move it device-resident alongside the PCG loop.
//!
//! Numerics: f64 throughout, no `--use_fast_math`. The kernel is a plain
//! double-precision GEMV / diagonal-extraction; per-row symmetry of `H_i`
//! is preserved because the CPU oracle emits symmetric blocks and the
//! kernel reads only the row-major upper-or-lower triangle and the diagonal
//! is read once.

#[cfg(target_os = "linux")]
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};

#[cfg(target_os = "linux")]
use gam_gpu::gpu_err;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuError;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuResultExt;

/// `blockDim.x` for the per-row matvec / diagonal kernels. One CUDA block per
/// row; the 32 threads form a reusable direction tile and sweep every runtime
/// primary width in synchronized batches. Linux-only because the launcher that
/// consumes it is Linux-only.
#[cfg(target_os = "linux")]
const ROW_HV_THREADS: u32 = 32;

#[cfg(target_os = "linux")]
fn checked_row_shape_len(context: &str, dimensions: &[usize]) -> Result<usize, GpuError> {
    dimensions
        .iter()
        .copied()
        .try_fold(1_usize, |product, dimension| {
            product
                .checked_mul(dimension)
                .ok_or_else(|| GpuError::DriverCallFailed {
                    reason: format!(
                        "row_hessian_ops {context}: shape product overflow for dimensions {dimensions:?}"
                    ),
                })
        })
}

/// Per-call input bundle for [`launch_row_hessian_matvec`].
///
/// All buffers are borrowed views over host memory; the launcher uploads
/// them once per call. Future Phase 5 work will introduce a device-resident
/// twin that skips the upload.
pub(crate) struct RowHessianMatvecInputs<'a> {
    /// Number of observation rows.
    pub n_rows: usize,
    /// Primary local dimension `r` (= per-row Hessian block size).
    pub r: usize,
    /// Per-row Hessian blocks, row-major `[n_rows, r, r]`. Same layout as
    /// the CPU `BernoulliMarginalSlopeExactEvalCache.row_primary_hessians`
    /// pin and as the Phase-1 GPU FullRowMajor cache.
    pub h_rows: &'a [f64],
    /// Per-row direction, row-major `[n_rows, r]`. Produced on the CPU by
    /// `BernoulliMarginalSlopeFamily::row_primary_direction_from_flat` (one
    /// call per row), so by the time we reach the device the direction is
    /// already projected to the primary basis.
    pub v_rows: &'a [f64],
}

/// Per-row outputs from [`launch_row_hessian_matvec`].
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub(crate) struct RowHessianMatvecOutputs {
    /// Per-row product `y_i = H_i · v_i`, row-major `[n_rows, r]`.
    pub y_rows: Vec<f64>,
}

/// Per-call input bundle for [`launch_row_hessian_diag`].
pub(crate) struct RowHessianDiagInputs<'a> {
    /// Number of observation rows.
    pub n_rows: usize,
    /// Primary local dimension `r`.
    pub r: usize,
    /// Per-row Hessian blocks, row-major `[n_rows, r, r]`. Same layout as
    /// [`RowHessianMatvecInputs::h_rows`].
    pub h_rows: &'a [f64],
}

/// Per-row outputs from [`launch_row_hessian_diag`].
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub(crate) struct RowHessianDiagOutputs {
    /// Per-row diagonal `d_i[u] = H_i[u, u]`, row-major `[n_rows, r]`.
    pub d_rows: Vec<f64>,
}

#[cfg(target_os = "linux")]
impl<'a> RowHessianMatvecInputs<'a> {
    /// Validate every shape the device kernel relies on.
    pub(crate) fn validate(&self) -> Result<(), GpuError> {
        if self.n_rows == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "row_hessian_matvec inputs: n_rows must be > 0".to_string(),
            });
        }
        if self.r == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "row_hessian_matvec inputs: r must be > 0".to_string(),
            });
        }
        let nr = checked_row_shape_len("matvec [n,r]", &[self.n_rows, self.r])?;
        let nrr = checked_row_shape_len("matvec [n,r,r]", &[self.n_rows, self.r, self.r])?;
        if self.h_rows.len() != nrr {
            gam_gpu::gpu_bail!(
                "row_hessian_matvec inputs: h_rows.len()={} != n_rows({})*r({})*r = {}",
                self.h_rows.len(),
                self.n_rows,
                self.r,
                nrr
            );
        }
        if self.v_rows.len() != nr {
            gam_gpu::gpu_bail!(
                "row_hessian_matvec inputs: v_rows.len()={} != n_rows({})*r({}) = {}",
                self.v_rows.len(),
                self.n_rows,
                self.r,
                nr
            );
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl<'a> RowHessianDiagInputs<'a> {
    /// Validate every shape the device kernel relies on.
    pub(crate) fn validate(&self) -> Result<(), GpuError> {
        if self.n_rows == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "row_hessian_diag inputs: n_rows must be > 0".to_string(),
            });
        }
        if self.r == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "row_hessian_diag inputs: r must be > 0".to_string(),
            });
        }
        let nrr = checked_row_shape_len("diagonal [n,r,r]", &[self.n_rows, self.r, self.r])?;
        if self.h_rows.len() != nrr {
            gam_gpu::gpu_bail!(
                "row_hessian_diag inputs: h_rows.len()={} != n_rows({})*r({})*r = {}",
                self.h_rows.len(),
                self.n_rows,
                self.r,
                nrr
            );
        }
        Ok(())
    }
}

/// NVRTC kernel source. Two kernels share the file: per-row matvec
/// (`row_hessian_matvec_kernel`) and per-row diagonal extraction
/// (`row_hessian_diag_kernel`). One CUDA block per row; the 32 threads of
/// each block parallelise the inner `r`-loop. f64 throughout.
///
/// Parity reference on the CPU side:
///   * matvec: `scratch.hess.dot(&row_dir)` inside
///     `exact_newton_joint_hessian_matvec_from_cache` in
///     `src/families/bernoulli_marginal_slope.rs`;
///   * diag:   `row_hess[[u, u]]` reads inside
///     `exact_newton_joint_hessian_diagonal_from_cache`.
#[cfg(target_os = "linux")]
const ROW_KERNEL_SOURCE: &str = r#"
extern "C" {

// Per-row matvec: y_i[u] = sum_v H_i[u, v] * v_i[v].
// One block per row; blockDim.x = 32. Each thread accumulates a partial
// sum over the inner `v` index for its slice of `u` rows.
//
// Parity reference: `scratch.hess.dot(&row_dir)` in CPU
// exact_newton_joint_hessian_matvec_from_cache.
__global__ void row_hessian_matvec_kernel(
    const int n_rows,
    const int r,
    const double* __restrict__ h_rows, // [n_rows, r, r] row-major
    const double* __restrict__ v_rows, // [n_rows, r]    row-major
    double*       __restrict__ y_rows  // [n_rows, r]    row-major
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    // The shared array is an algorithmic tile, not a semantic width bound.
    // Every synchronized u-batch sweeps all v-tiles, so r=33 and arbitrary
    // wider checked shapes execute the same contraction without local arrays
    // proportional to r.
    extern __shared__ double v_shared[];

    const double* h_base = h_rows + (size_t)row * (size_t)r * (size_t)r;
    double*       y_base = y_rows + (size_t)row * (size_t)r;
    for (int u_base = 0; u_base < r; u_base += nthr) {
        const int u = u_base + tid;
        double acc = 0.0;
        for (int v_base = 0; v_base < r; v_base += nthr) {
            const int loaded_v = v_base + tid;
            v_shared[tid] = loaded_v < r
                ? v_rows[(size_t)row * (size_t)r + (size_t)loaded_v]
                : 0.0;
            __syncthreads();
            if (u < r) {
                const double* h_tile = h_base + (size_t)u * (size_t)r + (size_t)v_base;
                const int tile_len = r - v_base < nthr ? r - v_base : nthr;
                for (int local_v = 0; local_v < tile_len; ++local_v) {
                    acc += h_tile[local_v] * v_shared[local_v];
                }
            }
            __syncthreads();
        }
        if (u < r) {
            y_base[u] = acc;
        }
    }
}

// Per-row diagonal: d_i[u] = H_i[u, u].
// One block per row; blockDim.x = 32. Each thread extracts a strided
// subset of diagonal entries; no inner reduction is needed.
//
// Parity reference: `row_hess[[u, u]]` in CPU
// exact_newton_joint_hessian_diagonal_from_cache.
__global__ void row_hessian_diag_kernel(
    const int n_rows,
    const int r,
    const double* __restrict__ h_rows, // [n_rows, r, r] row-major
    double*       __restrict__ d_rows  // [n_rows, r]    row-major
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    const double* h_base = h_rows + (size_t)row * (size_t)r * (size_t)r;
    double*       d_base = d_rows + (size_t)row * (size_t)r;
    for (int u = tid; u < r; u += nthr) {
        d_base[u] = h_base[(size_t)u * (size_t)r + (size_t)u];
    }
}

} // extern "C"
"#;

#[cfg(target_os = "linux")]
struct RowOpsBackend {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

#[cfg(target_os = "linux")]
impl RowOpsBackend {
    fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<RowOpsBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let runtime = gam_gpu::device_runtime::GpuRuntime::require()?;
                let ctx = gam_gpu::device_runtime::cuda_context_for(
                    runtime.selected_device().ordinal,
                )
                .ok_or_else(|| {
                    gpu_err!(
                        "row_hessian_ops backend: failed to create CUDA context for device {}",
                        runtime.selected_device().ordinal
                    )
                })?;
                let stream = ctx.default_stream();
                // Shared arch+fmad options (NOT bare `compile_ptx`): #1686's
                // `--fmad=false` keeps the matvec / diag reductions
                // bit-comparable to the separately-rounded CPU oracle, and the
                // #1551 arch pin keys the kernel to the device's real compute
                // capability instead of NVRTC's pre-sm_60 default.
                let ptx = gam_gpu::device_cache::compile_ptx_arch(ROW_KERNEL_SOURCE)
                    .map_err(|err| gpu_err!("row_hessian_ops NVRTC compile failed: {err}"))?;
                let module = ctx
                    .load_module(ptx)
                    .gpu_ctx("row_hessian_ops module load failed")?;
                Ok(RowOpsBackend { stream, module })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }
}

/// Launch the per-row Hessian matvec. Linux-only; on non-Linux the entire
/// kernel cache machinery is compiled out and callers must take the CPU path.
#[cfg(target_os = "linux")]
pub(crate) fn launch_row_hessian_matvec(
    inputs: RowHessianMatvecInputs<'_>,
) -> Result<RowHessianMatvecOutputs, GpuError> {
    inputs.validate()?;
    launch_matvec_linux(inputs)
}

/// Launch the per-row Hessian diagonal extraction. Linux-only; non-Linux
/// callers compile out the call site entirely.
#[cfg(target_os = "linux")]
pub(crate) fn launch_row_hessian_diag(
    inputs: RowHessianDiagInputs<'_>,
) -> Result<RowHessianDiagOutputs, GpuError> {
    inputs.validate()?;
    launch_diag_linux(inputs)
}

#[cfg(target_os = "linux")]
fn launch_matvec_linux(
    inputs: RowHessianMatvecInputs<'_>,
) -> Result<RowHessianMatvecOutputs, GpuError> {
    let backend = RowOpsBackend::probe()?;
    let stream = &backend.stream;
    let n = inputs.n_rows;
    let r = inputs.r;
    let nr = checked_row_shape_len("matvec output [n,r]", &[n, r])?;

    let d_h = stream
        .clone_htod(inputs.h_rows)
        .gpu_ctx("row_hessian_matvec upload h_rows")?;
    let d_v = stream
        .clone_htod(inputs.v_rows)
        .gpu_ctx("row_hessian_matvec upload v_rows")?;
    let mut d_y = stream
        .alloc_zeros::<f64>(nr)
        .gpu_ctx("row_hessian_matvec alloc y_rows")?;

    let func = backend
        .module
        .load_function("row_hessian_matvec_kernel")
        .gpu_ctx("row_hessian_matvec load_function")?;

    let n_i32 = i32::try_from(n)
        .map_err(|_| gpu_err!("row_hessian_matvec: n_rows={n} exceeds i32 range"))?;
    let n_u32 = u32::try_from(n)
        .map_err(|_| gpu_err!("row_hessian_matvec: n_rows={n} exceeds u32 range"))?;
    let r_i32 =
        i32::try_from(r).map_err(|_| gpu_err!("row_hessian_matvec: r={r} exceeds i32 range"))?;
    let shared_mem_bytes = ROW_HV_THREADS
        .checked_mul(u32::try_from(std::mem::size_of::<f64>()).map_err(|_| {
            gpu_err!("row_hessian_matvec: f64 byte width exceeds CUDA shared-memory range")
        })?)
        .ok_or_else(|| gpu_err!("row_hessian_matvec: shared tile byte count overflow"))?;
    let cfg = LaunchConfig {
        grid_dim: (n_u32, 1, 1),
        block_dim: (ROW_HV_THREADS, 1, 1),
        shared_mem_bytes,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&n_i32)
        .arg(&r_i32)
        .arg(&d_h)
        .arg(&d_v)
        .arg(&mut d_y);

    // SAFETY: every kernel argument is either an `i32` (passed by value)
    // or a device pointer to a buffer whose length was validated above
    // (`validate()` matches the kernel's exact indexing pattern). Dynamic
    // shared memory is exactly one `ROW_HV_THREADS` tile sized by the Rust
    // launch authority; synchronized batches cover the complete runtime width.
    unsafe { builder.launch(cfg) }.gpu_ctx("row_hessian_matvec launch")?;
    stream
        .synchronize()
        .gpu_ctx("row_hessian_matvec synchronize")?;
    let y_rows = stream
        .clone_dtoh(&d_y)
        .gpu_ctx("row_hessian_matvec download y_rows")?;
    Ok(RowHessianMatvecOutputs { y_rows })
}

#[cfg(target_os = "linux")]
fn launch_diag_linux(inputs: RowHessianDiagInputs<'_>) -> Result<RowHessianDiagOutputs, GpuError> {
    let backend = RowOpsBackend::probe()?;
    let stream = &backend.stream;
    let n = inputs.n_rows;
    let r = inputs.r;
    let nr = checked_row_shape_len("diagonal output [n,r]", &[n, r])?;

    let d_h = stream
        .clone_htod(inputs.h_rows)
        .gpu_ctx("row_hessian_diag upload h_rows")?;
    let mut d_d = stream
        .alloc_zeros::<f64>(nr)
        .gpu_ctx("row_hessian_diag alloc d_rows")?;

    let func = backend
        .module
        .load_function("row_hessian_diag_kernel")
        .gpu_ctx("row_hessian_diag load_function")?;

    let n_i32 =
        i32::try_from(n).map_err(|_| gpu_err!("row_hessian_diag: n_rows={n} exceeds i32 range"))?;
    let n_u32 =
        u32::try_from(n).map_err(|_| gpu_err!("row_hessian_diag: n_rows={n} exceeds u32 range"))?;
    let r_i32 =
        i32::try_from(r).map_err(|_| gpu_err!("row_hessian_diag: r={r} exceeds i32 range"))?;
    let cfg = LaunchConfig {
        grid_dim: (n_u32, 1, 1),
        block_dim: (ROW_HV_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&n_i32).arg(&r_i32).arg(&d_h).arg(&mut d_d);

    // SAFETY: every kernel argument is either an `i32` (passed by value)
    // or a device pointer to a buffer whose length was validated above.
    // The kernel only reads diagonal entries `H_i[u, u]` for `u ∈ [0, r)`,
    // which is in-bounds for `h_rows.len() = n_rows*r*r`.
    unsafe { builder.launch(cfg) }.gpu_ctx("row_hessian_diag launch")?;
    stream
        .synchronize()
        .gpu_ctx("row_hessian_diag synchronize")?;
    let d_rows = stream
        .clone_dtoh(&d_d)
        .gpu_ctx("row_hessian_diag download d_rows")?;
    Ok(RowHessianDiagOutputs { d_rows })
}

/// CPU execution of the same per-row Hessian matvec. This is the live
/// non-CUDA route used by the host-pin joint-Hessian consumers.
pub(crate) fn cpu_row_hessian_matvec(inputs: &RowHessianMatvecInputs<'_>) -> Vec<f64> {
    let n = inputs.n_rows;
    let r = inputs.r;
    let mut y = vec![0.0_f64; n * r];
    for row in 0..n {
        let h_base = row * r * r;
        let v_base = row * r;
        for u in 0..r {
            let mut acc = 0.0_f64;
            for v in 0..r {
                acc += inputs.h_rows[h_base + u * r + v] * inputs.v_rows[v_base + v];
            }
            y[v_base + u] = acc;
        }
    }
    y
}

/// CPU execution of the same per-row Hessian diagonal extraction. This is the
/// live non-CUDA route used by the host-pin preconditioner consumers.
pub(crate) fn cpu_row_hessian_diag(inputs: &RowHessianDiagInputs<'_>) -> Vec<f64> {
    let n = inputs.n_rows;
    let r = inputs.r;
    let mut diagonal = vec![0.0_f64; n * r];
    for row in 0..n {
        let h_base = row * r * r;
        let diagonal_base = row * r;
        for u in 0..r {
            diagonal[diagonal_base + u] = inputs.h_rows[h_base + u * r + u];
        }
    }
    diagonal
}

#[cfg(test)]
mod tests {
    // All items below are `#[cfg(target_os = "linux")]` (GPU parity), so the
    // glob import is only live on Linux; gate it to avoid an unused-import
    // error when compiling the lib tests on other platforms.
    #[cfg(target_os = "linux")]
    use super::*;

    /// Deterministic non-trivial Hessian fixture. Generates per-row
    /// symmetric `r×r` blocks via `H_i = A_i + A_iᵀ + r·I` for a
    /// scrambled `A_i`, plus a per-row direction `v_i` with the same
    /// scrambling seed offset. Both `matvec` and `diag` parity tests
    /// share the same fixture so any regression in the cached-Hessian
    /// upload path surfaces in both. Only the GPU parity test consumes
    /// this fixture, so it tracks that test's Linux gating.
    #[cfg(target_os = "linux")]
    fn make_fixture(n_rows: usize, r: usize) -> (Vec<f64>, Vec<f64>) {
        let mut h = vec![0.0_f64; n_rows * r * r];
        let mut v = vec![0.0_f64; n_rows * r];
        for row in 0..n_rows {
            let base = row * r * r;
            for u in 0..r {
                for vv in 0..r {
                    let seed = (row as f64) * 0.137 + (u as f64) * 1.901 + (vv as f64) * 0.317;
                    let a = (seed.sin() * 1.7 + (seed * 0.5).cos() * 0.9) * 0.5;
                    h[base + u * r + vv] = a;
                }
            }
            for u in 0..r {
                for vv in (u + 1)..r {
                    let upper = h[base + u * r + vv];
                    let lower = h[base + vv * r + u];
                    let sym = 0.5 * (upper + lower);
                    h[base + u * r + vv] = sym;
                    h[base + vv * r + u] = sym;
                }
                h[base + u * r + u] += r as f64;
            }
            for u in 0..r {
                let seed = (row as f64) * 0.211 + (u as f64) * 0.733 + 1.5;
                v[row * r + u] = seed.sin() * 0.6 - (seed * 0.5).cos() * 0.4;
            }
        }
        (h, v)
    }

    // Uses the Linux-only `validate()` shape-checks; gated to match.
    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_oracle_matches_handwritten_2x2() {
        // Two rows, r = 2 — small enough to verify by hand.
        // Row 0: H = [[2, 1], [1, 3]],  v = [1, -1]  => y = [1, -2]
        // Row 1: H = [[4, 0], [0, 5]],  v = [2,  3]  => y = [8, 15]
        let h_rows = vec![2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0, 5.0];
        let v_rows = vec![1.0, -1.0, 2.0, 3.0];
        let inputs = RowHessianMatvecInputs {
            n_rows: 2,
            r: 2,
            h_rows: &h_rows,
            v_rows: &v_rows,
        };
        inputs.validate().expect("hand fixture must validate");
        let y = cpu_row_hessian_matvec(&inputs);
        assert_eq!(y, vec![1.0, -2.0, 8.0, 15.0]);

        let diag_inputs = RowHessianDiagInputs {
            n_rows: 2,
            r: 2,
            h_rows: &h_rows,
        };
        diag_inputs.validate().expect("hand fixture must validate");
        let d = cpu_row_hessian_diag(&diag_inputs);
        assert_eq!(d, vec![2.0, 3.0, 4.0, 5.0]);
    }

    // Uses Linux-only `GpuError`/`validate()`; gated to match.
    #[cfg(target_os = "linux")]
    #[test]
    fn validation_checks_shapes_without_a_primary_width_cap_932() {
        let h_rows = vec![1.0; 8];
        let v_rows = vec![1.0; 3]; // wrong: should be 4 for n=2, r=2
        let inputs = RowHessianMatvecInputs {
            n_rows: 2,
            r: 2,
            h_rows: &h_rows,
            v_rows: &v_rows,
        };
        match inputs.validate() {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(reason.contains("v_rows"), "unexpected reason: {reason}");
            }
            other => panic!("expected DriverCallFailed, got {other:?}"),
        }

        let r = 33;
        let h_rows = vec![0.0; r * r];
        let v_rows = vec![0.0; r];
        RowHessianMatvecInputs {
            n_rows: 1,
            r,
            h_rows: &h_rows,
            v_rows: &v_rows,
        }
        .validate()
        .expect("r=33 must cross the old tile width without a semantic cap");

        let overflow = RowHessianMatvecInputs {
            n_rows: usize::MAX,
            r: 2,
            h_rows: &[],
            v_rows: &[],
        }
        .validate()
        .expect_err("overflowing shapes must fail before slice comparison");
        assert!(overflow.to_string().contains("shape product overflow"));
    }

    /// CPU↔GPU parity for both kernels. The GPU launch entry points only
    /// exist on Linux (cudarc is a Linux-only dependency), so the parity
    /// test is gated to Linux as well. On Linux hosts without a CUDA
    /// runtime the test skips at runtime, mirroring the convention in
    /// `bms_flex_row_kernel_matches_cpu_oracle_when_cuda_available`.
    ///
    /// Tolerances: abs ≤ 2e-8, rel ≤ 2e-7 per the Block 9 Phase 2/3
    /// charter. Fixture has r = 33 and n_rows = 4 to cross the 32-value
    /// algorithm tile and prove that tile width is not a primary-width cap.
    #[cfg(target_os = "linux")]
    #[test]
    fn row_hessian_kernels_match_cpu_oracle_when_cuda_available() {
        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => {
                eprintln!("[row_hessian_ops parity] no CUDA device — skipping CUDA parity");
                return;
            }
            Err(error) => panic!("[row_hessian_ops parity] CUDA probe failed: {error}"),
        }
        let n_rows = 4;
        let r = 33;
        let (h_rows, v_rows) = make_fixture(n_rows, r);

        let matvec_inputs = RowHessianMatvecInputs {
            n_rows,
            r,
            h_rows: &h_rows,
            v_rows: &v_rows,
        };
        matvec_inputs
            .validate()
            .expect("matvec fixture must validate");
        let cpu_y = cpu_row_hessian_matvec(&matvec_inputs);
        let gpu_y = match launch_row_hessian_matvec(matvec_inputs) {
            Ok(out) => out.y_rows,
            Err(err) => panic!(
                "[row_hessian_ops parity] matvec kernel launch FAILED after the CUDA \
                 runtime was confirmed present ({err}) — a real device-kernel regression, \
                 not a CI infra outage. (#1017: GPU faults must fail loud, never skip-pass.)"
            ),
        };
        let tol_abs = 2e-8_f64;
        let tol_rel = 2e-7_f64;
        assert_eq!(cpu_y.len(), gpu_y.len(), "matvec output length mismatch");
        for (i, (&c, &g)) in cpu_y.iter().zip(gpu_y.iter()).enumerate() {
            let diff = (c - g).abs();
            let tol = tol_abs + tol_rel * c.abs();
            assert!(
                diff <= tol,
                "matvec[{i}]: |cpu - gpu| = {diff:.3e} > tol = {tol:.3e}; \
                 cpu={c:.17e}, gpu={g:.17e}"
            );
        }

        let diag_inputs = RowHessianDiagInputs {
            n_rows,
            r,
            h_rows: &h_rows,
        };
        diag_inputs.validate().expect("diag fixture must validate");
        let cpu_d = cpu_row_hessian_diag(&diag_inputs);
        let gpu_d = match launch_row_hessian_diag(diag_inputs) {
            Ok(out) => out.d_rows,
            Err(err) => panic!(
                "[row_hessian_ops parity] diag kernel launch FAILED after the CUDA \
                 runtime was confirmed present ({err}) — a real device-kernel regression, \
                 not a CI infra outage. (#1017: GPU faults must fail loud, never skip-pass.)"
            ),
        };
        assert_eq!(cpu_d.len(), gpu_d.len(), "diag output length mismatch");
        for (i, (&c, &g)) in cpu_d.iter().zip(gpu_d.iter()).enumerate() {
            let diff = (c - g).abs();
            let tol = tol_abs + tol_rel * c.abs();
            assert!(
                diff <= tol,
                "diag[{i}]: |cpu - gpu| = {diff:.3e} > tol = {tol:.3e}; \
                 cpu={c:.17e}, gpu={g:.17e}"
            );
        }
    }
}
