//! Layer D + E NVRTC kernels for the device-resident Arrow-Schur Newton solve.
//!
//! Layer D — `arrow_schur_forward_pgroup` — replaces the three separate
//! cuSOLVER + cuBLAS launches in [`crate::gpu::kernels::arrow_schur`] (batched POTRF,
//! batched TRSM on `g`, batched TRSM on `B`, sequential GEMM/GEMV Schur
//! accumulation) with a single NVRTC launch per p-group. Each CUDA block owns
//! one row block `i`:
//!
//!     1. cooperative load of `D_i` (`P × P`) and `B_i` (`P × R`) into shared
//!        memory, with `ridge_t` added to the diagonal during the load;
//!     2. scalar lower Cholesky `D_i + ρ_t I = L_i L_i^T` in shared memory;
//!     3. forward solve `u_i = L_i^{-1} g_i` (single warp, in registers);
//!     4. forward solve `Y_i = L_i^{-1} B_i` (`P × R` shared tile);
//!     5. partial reductions emitted in column-major form:
//!           `partial_S[i] = Y_i^T Y_i`   (R × R, contributes `-1` to S_β)
//!           `partial_r[i] = Y_i^T u_i`   (R)
//!
//! The dispatch host then either reduces the per-block partials with a single
//! `cub::DeviceReduce` (Layer D fast path) or, for very small `n`, performs the
//! reduction on the CPU after a `clone_dtoh`. The Cholesky factors `L_i` and
//! the forward-solved `u_i`, `Y_i` are persisted in their original device
//! buffers so Layer E can run the back-substitution without re-uploading.
//!
//! Layer E — `arrow_schur_back_sub_pgroup` — completes the pipeline. After the
//! Schur factor `R_β` has been formed on host or via cuSOLVER and `δβ`
//! downloaded back, each block computes
//!
//!     `δt_i = -L_i^{-T} (u_i + Y_i δβ)`
//!
//! in a single launch, returning the n·P vector. The Layer C cuBLAS GEMV +
//! batched TRSM step is replaced by one shared-memory matvec + one back-solve
//! per block.
//!
//! Dispatch policy (caller-facing):
//!   * `Σ_i p_i^3 ≳ 1e5` OR `R ≥ 16` AND the data is already device-resident →
//!     use the fused NVRTC path.
//!   * Otherwise fall through to the cuSOLVER/cuBLAS path in `arrow_schur.rs`.
//!
//! Ridge escalation: the kernel reports a per-block status code (positive
//! pivot index on failure). On non-zero status the host caller adds
//! `bump = scale · ε^½ · 1024` to `ridge_t` and re-launches, the same Ceres-
//! style geometric escalation the CPU path already implements.


use crate::solver::arrow_schur::ArrowSchurSystem;

/// Fused-kernel dispatch admission. Returns `true` when the workload shape
/// makes the Layer D NVRTC fused path strictly preferable to the cuSOLVER /
/// cuBLAS Layer A+B+C path. The math-block-3 §9 heuristic:
///
///   * `Σ_i p_i^3` is the total block-Cholesky cost. Below ~1e5 flops the
///     launch overhead of cuSOLVER batched POTRF dominates and the fused
///     kernel wins; above ~1e5 flops both paths are launch-amortized but the
///     fused kernel still wins on memory traffic because `L_i`, `u_i`, `Y_i`
///     stay in shared memory between the four steps.
///   * `R ≥ 16` makes the Schur GEMM the bottleneck of Layer B's path; the
///     fused kernel's per-block reduction skips one global-memory round-trip
///     per Y_i tile.
///
/// Both conditions trigger admission; the `Unavailable` fall-through preserves
/// the existing CPU + cuSOLVER paths when admission fails.
#[inline]
#[must_use]
pub fn fused_path_admitted(n: usize, p: usize, r: usize) -> bool {
    if n == 0 || p == 0 || r == 0 {
        return false;
    }
    if p > MAX_FUSED_P {
        return false;
    }
    let total_chol_flops = (n as u128) * (p as u128).pow(3);
    total_chol_flops >= 100_000 || r >= 16
}

/// Hard upper bound on per-block size that the Layer D kernel supports. The
/// per-block shared-memory budget is `P*P + P*R + P + R*R + R` doubles,
/// which at `P = 32`, `R = 32` is `32^2 + 32·32 + 32 + 32^2 + 32 = 3136`
/// doubles = 24.5 KiB. Volta's 96 KiB shared memory per SM gives us three
/// concurrent blocks per SM at that ceiling, matching the bench-tuned launch
/// configuration in math block 3 §8.
pub const MAX_FUSED_P: usize = 32;

/// Compile-time `R` widths the NVRTC fused kernel is templated on. The Arrow-
/// Schur driver always builds the system at a single uniform `R = K` (the
/// shared β width), so the host JIT selects exactly one template
/// instantiation per `(P, R)` pair encountered. Caching matches the
/// `S2ModuleCacheKey` pattern in `crate::terms::basis::sphere_gpu`.
pub const FUSED_R_TEMPLATES: &[usize] = &[4, 5, 6, 8, 10, 12, 16, 20, 24, 32];

/// Smallest entry in `FUSED_R_TEMPLATES` that is ≥ `r`. Used both by the
/// kernel selector and the `(P, R)` cache key so two systems with the same
/// `(P, ceil_r)` share one compiled module.
#[inline]
#[must_use]
pub fn ceil_to_template_r(r: usize) -> Option<usize> {
    FUSED_R_TEMPLATES
        .iter()
        .copied()
        .find(|template| *template >= r)
}

/// Stable cache key for one NVRTC compilation. The CC pair lets one host
/// process drive multiple device generations without re-compiling on every
/// launch; `p_max` lets the kernel use a static shared-memory layout sized
/// for `P × P` and `P × R` doubles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FusedModuleCacheKey {
    pub cc_major: i32,
    pub cc_minor: i32,
    pub p_max: u32,
    pub r_template: u32,
}

/// Forward-pass NVRTC source template. Two compile-time macros — `P_MAX` for
/// the row-block size (matched against the runtime `d` per launch) and
/// `R_TEMPLATE` for the shared β width — are prepended by
/// `forward_kernel_source` to produce a single specialised translation unit.
///
/// The source intentionally uses scalar loops over shared-memory tiles
/// instead of warp-level intrinsics so that small `P` (≤ 30) blocks fit one
/// warp and large `P` (≤ 32) blocks fit one warp-pair. `__syncthreads()` is
/// the only intra-block synchronization primitive used; no shuffles, no
/// shared-memory atomics. This matches the Layer A baseline's element-wise
/// behaviour exactly so the C↔D parity test can compare bit-stable up to
/// the floating-point error of the inner-product order.
#[cfg(target_os = "linux")]
pub(crate) const FORWARD_KERNEL_SOURCE: &str = r#"
// Compile-time macros injected by the host JIT cache:
//   #define P_MAX        <usize, ≤ 32>   per-block latent dimension cap
//   #define R_TEMPLATE   <usize, ∈ FUSED_R_TEMPLATES>   shared β width
//
// One CUDA block per row block `i`. The block contains `P_MAX` threads.
// Each thread owns one row of `D_i`, `B_i`, `L_i`, `Y_i`, and one element
// of `g_i` / `u_i` / per-block partial accumulators.

extern "C" __global__
__launch_bounds__(64)
void arrow_schur_forward_pgroup(
    const double* __restrict__ d_stack,   // n * P_MAX * P_MAX, column-major per block
    const double* __restrict__ b_stack,   // n * P_MAX * R_TEMPLATE
    const double* __restrict__ g_stack,   // n * P_MAX
    int n,
    int p_runtime,                        // ≤ P_MAX
    int r_runtime,                        // ≤ R_TEMPLATE
    double ridge_t,
    double* __restrict__ l_out,           // n * P_MAX * P_MAX, lower Cholesky in place ok
    double* __restrict__ u_out,           // n * P_MAX  (L^{-1} g)
    double* __restrict__ y_out,           // n * P_MAX * R_TEMPLATE  (L^{-1} B)
    double* __restrict__ partial_s,       // n * R_TEMPLATE * R_TEMPLATE
    double* __restrict__ partial_r,       // n * R_TEMPLATE
    int*    __restrict__ status_out       // n, 0 = ok, else 1-based pivot row
) {
    const int i = blockIdx.x;
    if (i >= n) return;

    const int tid = threadIdx.x;
    if (tid >= P_MAX) return;

    __shared__ double L[P_MAX][P_MAX];     // lower factor of D_i + ρ I
    __shared__ double Y[P_MAX][R_TEMPLATE]; // L^{-1} B_i
    __shared__ double u[P_MAX];             // L^{-1} g_i

    // ---- Load D_i + ridge_t·I into L. Column-major: element (r, c) at
    //      d_stack[i*P_MAX*P_MAX + c*P_MAX + r]. ----
    if (tid < p_runtime) {
        for (int c = 0; c < p_runtime; ++c) {
            double v = d_stack[((size_t) i * P_MAX + c) * P_MAX + tid];
            if (tid == c) v += ridge_t;
            L[tid][c] = v;
        }
        // ---- Load g_i into u (will be overwritten by L^{-1} g). ----
        u[tid] = g_stack[(size_t) i * P_MAX + tid];
        // ---- Load B_i into Y (will be overwritten by L^{-1} B). ----
        for (int c = 0; c < r_runtime; ++c) {
            Y[tid][c] = b_stack[((size_t) i * P_MAX + c) * P_MAX + tid];
        }
    }
    __syncthreads();

    // ---- Scalar lower Cholesky in shared memory. Single-threaded inside the
    //      block: P ≤ 32 so this is at most ~16 KFLOPs serial, dwarfed by the
    //      subsequent solves' parallel work. ----
    if (tid == 0) {
        for (int j = 0; j < p_runtime; ++j) {
            double diag = L[j][j];
            for (int t = 0; t < j; ++t) diag -= L[j][t] * L[j][t];
            if (!(diag > 0.0)) {
                status_out[i] = j + 1;
                return;
            }
            const double l_jj = sqrt(diag);
            L[j][j] = l_jj;
            const double inv = 1.0 / l_jj;
            for (int r = j + 1; r < p_runtime; ++r) {
                double s = L[r][j];
                for (int t = 0; t < j; ++t) s -= L[r][t] * L[j][t];
                L[r][j] = s * inv;
            }
        }
        status_out[i] = 0;
    }
    __syncthreads();
    if (status_out[i] != 0) return;

    // ---- Forward solves `L u = g_in` and `L Y = B_in`. tid owns one column
    //      of Y (and the scalar element of u handled by tid == 0). Sequential
    //      row sweep, parallel across columns. ----
    if (tid == 0) {
        for (int r = 0; r < p_runtime; ++r) {
            double s = u[r];
            for (int t = 0; t < r; ++t) s -= L[r][t] * u[t];
            u[r] = s / L[r][r];
        }
    }
    __syncthreads();
    if (tid < r_runtime) {
        for (int r = 0; r < p_runtime; ++r) {
            double s = Y[r][tid];
            for (int t = 0; t < r; ++t) s -= L[r][t] * Y[t][tid];
            Y[r][tid] = s / L[r][r];
        }
    }
    __syncthreads();

    // ---- Emit L, u, Y back to global (column-major layout matches input). ----
    if (tid < p_runtime) {
        for (int c = 0; c < p_runtime; ++c) {
            l_out[((size_t) i * P_MAX + c) * P_MAX + tid] = L[tid][c];
        }
        u_out[(size_t) i * P_MAX + tid] = u[tid];
        for (int c = 0; c < r_runtime; ++c) {
            y_out[((size_t) i * P_MAX + c) * P_MAX + tid] = Y[tid][c];
        }
    }

    // ---- Per-block partial Schur reduction. partial_S[i] = Y^T Y (R × R)
    //      and partial_r[i] = Y^T u (R). Computed in shared memory column-
    //      strided by tid so all warp lanes participate. ----
    __syncthreads();
    if (tid < r_runtime) {
        // partial_r[i, tid] = sum_r Y[r, tid] * u[r]
        double rsum = 0.0;
        for (int r = 0; r < p_runtime; ++r) {
            rsum += Y[r][tid] * u[r];
        }
        partial_r[(size_t) i * R_TEMPLATE + tid] = rsum;
        // partial_S[i, c, tid] = sum_r Y[r, c] * Y[r, tid]   for c in 0..r_runtime
        for (int c = 0; c < r_runtime; ++c) {
            double ssum = 0.0;
            for (int r = 0; r < p_runtime; ++r) {
                ssum += Y[r][c] * Y[r][tid];
            }
            partial_s[((size_t) i * R_TEMPLATE + c) * R_TEMPLATE + tid] = ssum;
        }
    }
}

// ----------------------------------------------------------------------
// Layer E back-substitution kernel.
//
// δt_i = -L_i^{-T} (u_i + Y_i · δβ), one block per row, where L_i, u_i, Y_i
// were stored by `arrow_schur_forward_pgroup`.
// ----------------------------------------------------------------------
extern "C" __global__
__launch_bounds__(64)
void arrow_schur_back_sub_pgroup(
    const double* __restrict__ l_stack,    // n * P_MAX * P_MAX (lower factor)
    const double* __restrict__ u_stack,    // n * P_MAX (already L^{-1} g)
    const double* __restrict__ y_stack,    // n * P_MAX * R_TEMPLATE
    const double* __restrict__ delta_beta, // R_TEMPLATE
    int n,
    int p_runtime,
    int r_runtime,
    double* __restrict__ delta_t_out       // n * P_MAX
) {
    const int i = blockIdx.x;
    if (i >= n) return;

    const int tid = threadIdx.x;
    if (tid >= P_MAX) return;

    __shared__ double L[P_MAX][P_MAX];
    __shared__ double w[P_MAX];

    if (tid < p_runtime) {
        for (int c = 0; c < p_runtime; ++c) {
            L[tid][c] = l_stack[((size_t) i * P_MAX + c) * P_MAX + tid];
        }
        double acc = u_stack[(size_t) i * P_MAX + tid];
        for (int c = 0; c < r_runtime; ++c) {
            acc += y_stack[((size_t) i * P_MAX + c) * P_MAX + tid] * delta_beta[c];
        }
        w[tid] = acc;
    }
    __syncthreads();

    // L^T x = w  (lower factor, transposed). Sequential row sweep from
    // bottom to top, single-threaded — matches Layer C's cuBLAS call which
    // also issues exactly one TRSM per block.
    if (tid == 0) {
        for (int r = p_runtime - 1; r >= 0; --r) {
            double s = w[r];
            for (int t = r + 1; t < p_runtime; ++t) s -= L[t][r] * w[t];
            w[r] = s / L[r][r];
        }
    }
    __syncthreads();

    if (tid < p_runtime) {
        delta_t_out[(size_t) i * P_MAX + tid] = -w[tid];
    }
}
"#;

/// Compile-time-readable summary of one launch. The host bench uses this to
/// pick a CTA size and to allocate the per-block partials buffer.
#[derive(Clone, Copy, Debug)]
pub struct FusedLaunchPlan {
    pub n: usize,
    pub p_runtime: usize,
    pub p_max: usize,
    pub r_runtime: usize,
    pub r_template: usize,
    pub threads_per_block: u32,
    pub blocks: u32,
    pub partial_s_doubles: usize,
    pub partial_r_doubles: usize,
}

/// Plan one fused launch from the static `(n, p, r)` triple. Returns `None`
/// when the workload exceeds the kernel's `P_MAX` ceiling or when `r` cannot
/// be ceiled into a template width.
#[inline]
#[must_use]
pub fn plan_fused_launch(n: usize, p: usize, r: usize) -> Option<FusedLaunchPlan> {
    if p == 0 || r == 0 || n == 0 || p > MAX_FUSED_P {
        return None;
    }
    let r_template = ceil_to_template_r(r)?;
    let p_max = p.next_power_of_two().max(p).min(MAX_FUSED_P);
    let threads_per_block = p_max.next_power_of_two().max(32) as u32;
    let blocks = u32::try_from(n).ok()?;
    Some(FusedLaunchPlan {
        n,
        p_runtime: p,
        p_max,
        r_runtime: r,
        r_template,
        threads_per_block,
        blocks,
        partial_s_doubles: n * r_template * r_template,
        partial_r_doubles: n * r_template,
    })
}

/// Build the full NVRTC source for one `(p_max, r_template)` instantiation.
/// The host caller prepends `#define`s for `P_MAX` and `R_TEMPLATE` so that
/// the compile is a single `compile_ptx` call matching the sibling kernels
/// in `crate::terms::basis::sphere_gpu`.
#[cfg(target_os = "linux")]
#[inline]
#[must_use]
pub fn forward_kernel_source(p_max: usize, r_template: usize) -> String {
    format!(
        "#define P_MAX {}\n#define R_TEMPLATE {}\n{}",
        p_max, r_template, FORWARD_KERNEL_SOURCE
    )
}

/// Whether the caller-provided system can route through the Layer D + E
/// NVRTC fused path. Centralised so [`crate::gpu::kernels::arrow_schur::solve`] and
/// future bench harnesses share the same admission rule.
#[inline]
#[must_use]
pub fn system_admits_fused_path(sys: &ArrowSchurSystem) -> bool {
    let n = sys.rows.len();
    let p = sys.d;
    let r = sys.k;
    if !fused_path_admitted(n, p, r) {
        return false;
    }
    ceil_to_template_r(r).is_some()
}

/// Status of a single emulated fused-kernel block solve. Mirrors the
/// `status_out` code path of `arrow_schur_forward_pgroup`: zero on success,
/// otherwise the 1-based pivot row at which the in-shared-memory Cholesky
/// hit a non-positive diagonal (the kernel's `status_out[i] = j + 1` branch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedBlockStatus {
    /// Block factored, forward-solved, and reduced cleanly.
    Ok,
    /// Non-positive pivot at this 1-based row; the GPU host caller escalates
    /// `ridge_t` and re-launches (see module-level "Ridge escalation").
    NonPositivePivot(usize),
}

/// Why the host-side CPU emulation of the fused Layer D + E pipeline declined.
/// These mirror the `ArrowSchurGpuFailure` cases the device host raises so the
/// emulator can stand in for the GPU path in a device-free parity harness.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedCpuError {
    /// A row block was not positive definite even after `ridge_t`. Carries the
    /// 0-based row and the 1-based pivot row, matching the kernel status code.
    RowNotPositiveDefinite { row: usize, pivot: usize },
    /// The reduced K×K Schur complement `S_β` failed Cholesky at this 1-based
    /// pivot — the bordered system is rank-deficient at the requested ridges.
    SchurFactorFailed { pivot: usize },
    /// The system carries the matrix-free `H_ββ` / `H_tβ` operators the dense
    /// fused path cannot consume, or degenerate dims. The device host returns
    /// `Unavailable` / `GpuRequiresDenseSystem` here.
    Unavailable,
}

/// Outcome of one emulated fused Arrow-Schur Newton solve. Field-for-field the
/// `(δt, δβ, log|H|)` triple the GPU path downloads, so a parity test can
/// compare this against [`crate::gpu::kernels::arrow_schur::solve_arrow_newton_step_dense_reference`]
/// — the same baseline the V100 Layer C↔D parity test uses on a device.
#[derive(Debug, Clone)]
pub struct FusedCpuSolution {
    pub delta_t: Vec<f64>,
    pub delta_beta: Vec<f64>,
    pub log_det_hessian: f64,
}

/// Per-block intermediates the forward emulation persists for the back-sub,
/// exactly the device buffers `arrow_schur_forward_pgroup` writes back
/// (`l_out`, `u_out`, `y_out`) and which `arrow_schur_back_sub_pgroup` reads.
/// Stored column-major to match the kernel's global-memory layout 1:1.
struct FusedRowState {
    /// `L_i` lower Cholesky of `D_i + ρ_t I`, column-major `p×p`.
    l: Vec<f64>,
    /// `u_i = L_i^{-1} g_i`, length `p`.
    u: Vec<f64>,
    /// `Y_i = L_i^{-1} B_i`, column-major `p×r`.
    y: Vec<f64>,
}

/// CPU emulation of the Layer D forward kernel `arrow_schur_forward_pgroup`
/// for one row block. Reproduces the kernel's exact scalar arithmetic and
/// memory layout:
///
/// 1. column-major load of `D_i` (with `ridge_t` folded onto the diagonal)
///    and `B_i` — element `(r, c)` lives at `c * p + r`, the same indexing the
///    kernel's `d_stack[(i*P_MAX + c)*P_MAX + r]` uses within one block;
/// 2. in-place lower Cholesky `L_i L_iᵀ = D_i + ρ_t I` (the kernel's
///    single-threaded `tid == 0` factor loop), returning the 1-based pivot row
///    on a non-positive diagonal — the `status_out[i] = j + 1` branch;
/// 3. forward solves `L_i u_i = g_i` and `L_i Y_i = B_i` (the kernel's per-row
///    sequential sweeps), each emitting `partial_r[i] = Y_iᵀ u_i` (length `r`)
///    and `partial_s[i] = Y_iᵀ Y_i` (`r×r`, column-major) in the SAME inner-
///    product accumulation order the kernel's final reduction uses.
///
/// `partial_s` / `partial_r` are written positive (`+Y_iᵀY_i`, `+Y_iᵀu_i`),
/// matching the NVRTC kernel; the host reduction applies the documented signs.
/// `log_det_local = 2 Σ_j ln L_i[j,j]` is the block's contribution to `log|H|`.
fn emulate_forward_block(
    d_col_major: &[f64],
    b_col_major: &[f64],
    g: &[f64],
    p: usize,
    r: usize,
    ridge_t: f64,
    partial_s: &mut [f64],
    partial_r: &mut [f64],
    log_det_local: &mut f64,
) -> Result<FusedRowState, usize> {
    // ---- Load D_i + ridge_t·I into L (column-major). ----
    let mut l = d_col_major.to_vec();
    assert_eq!(l.len(), p * p);
    for j in 0..p {
        l[j * p + j] += ridge_t;
    }
    // ---- In-place lower Cholesky, mirroring the kernel's tid==0 loop. The
    //      kernel reads/writes L[row][col] in a logically row-indexed shared
    //      tile; here L is column-major, so L[row][col] == l[col*p + row]. The
    //      arithmetic (including the inner-product order) is identical. ----
    for j in 0..p {
        let mut diag = l[j * p + j];
        for t in 0..j {
            let l_jt = l[t * p + j];
            diag -= l_jt * l_jt;
        }
        if !(diag > 0.0) {
            return Err(j + 1);
        }
        let l_jj = diag.sqrt();
        l[j * p + j] = l_jj;
        let inv = 1.0 / l_jj;
        for row in (j + 1)..p {
            let mut s = l[j * p + row];
            for t in 0..j {
                s -= l[t * p + row] * l[t * p + j];
            }
            l[j * p + row] = s * inv;
        }
    }
    *log_det_local += 2.0 * (0..p).map(|j| l[j * p + j].ln()).sum::<f64>();

    // ---- Forward solve L u = g (kernel's tid==0 sweep). ----
    let mut u = g.to_vec();
    assert_eq!(u.len(), p);
    for row in 0..p {
        let mut s = u[row];
        for t in 0..row {
            s -= l[t * p + row] * u[t];
        }
        u[row] = s / l[row * p + row];
    }
    // ---- Forward solve L Y = B (kernel's per-column sweep). ----
    let mut y = b_col_major.to_vec();
    assert_eq!(y.len(), p * r);
    for c in 0..r {
        for row in 0..p {
            let mut s = y[c * p + row];
            for t in 0..row {
                s -= l[t * p + row] * y[c * p + t];
            }
            y[c * p + row] = s / l[row * p + row];
        }
    }

    // ---- Per-block partial Schur reduction: partial_r = Yᵀu, partial_s = YᵀY.
    //      Inner-product loop order matches the kernel (sum over rows `r`). ----
    for c in 0..r {
        let mut rsum = 0.0;
        for row in 0..p {
            rsum += y[c * p + row] * u[row];
        }
        partial_r[c] = rsum;
    }
    for c in 0..r {
        for c2 in 0..r {
            let mut ssum = 0.0;
            for row in 0..p {
                ssum += y[c * p + row] * y[c2 * p + row];
            }
            // Column-major (c2 = "tid" axis, c = column): partial_s[c*r + c2].
            partial_s[c * r + c2] = ssum;
        }
    }

    Ok(FusedRowState { l, u, y })
}

/// CPU emulation of the Layer E back-substitution kernel
/// `arrow_schur_back_sub_pgroup` for one row block:
///     `w_i = u_i + Y_i · δβ`,  `L_iᵀ x_i = w_i`,  `δt_i = -x_i`.
/// Mirrors the kernel's bottom-to-top transposed solve and the final negation
/// (the kernel writes `-w`). Returns `δt_i` (length `p`).
fn emulate_back_sub_block(
    state: &FusedRowState,
    delta_beta: &[f64],
    p: usize,
    r: usize,
) -> Vec<f64> {
    // w = u + Y·δβ
    let mut w = state.u.clone();
    for c in 0..r {
        let db = delta_beta[c];
        for row in 0..p {
            w[row] += state.y[c * p + row] * db;
        }
    }
    // Lᵀ x = w, bottom-to-top (kernel's tid==0 reverse sweep). L is column-
    // major lower, so Lᵀ[r][t] (t>r) reads L[t][r] == l[r*p + t].
    for row in (0..p).rev() {
        let mut s = w[row];
        for t in (row + 1)..p {
            s -= state.l[row * p + t] * w[t];
        }
        w[row] = s / state.l[row * p + row];
    }
    // δt = -x
    w.iter().map(|v| -v).collect()
}

/// Faithful, device-free CPU emulation of the full Layer D + E fused NVRTC
/// Arrow-Schur Newton solve (`arrow_schur_forward_pgroup` +
/// `arrow_schur_back_sub_pgroup`), with the host-side Schur reduction and
/// factorization the dispatch performs between them.
///
/// This is the #1017 verification anchor for the fused-kernel **algorithm**:
/// the CUDA source can only run on a device, but its arithmetic — the per-row
/// Cholesky/forward-solve/`YᵀY`/`Yᵀu` reduction and the back-substitution —
/// is exactly reproduced here so its correctness is checkable on any host
/// against the dense reference, before any GPU wall-clock is available. The
/// loop order and inner-product accumulation order match the kernel so the
/// result is deterministic and reproduces the device arithmetic in the same
/// accumulation order. Where the order genuinely matches (this host reference
/// vs the device) the result is bit-identical; but this is an order-match
/// guarantee, not a license to claim a criterion ranking "cannot move" under
/// any reassociated reduction — a parallel/reassociated path can still flip a
/// near-tie winner within the f64 margin (#1211).
///
/// Sign and reduction conventions (from `arrow_schur::solve` host assembly):
///   `S_β = (H_ββ + ρ_β I) − Σ_i Y_iᵀ Y_i`,  `r_β = −g_β + Σ_i Y_iᵀ u_i`,
///   `δβ = S_β^{-1} r_β`,  `δt_i = −L_i^{-ᵀ}(u_i + Y_i δβ)`,
///   `log|H| = Σ_i 2Σ_j ln L_i[j,j] + 2Σ_j ln L_{S_β}[j,j]`.
///
/// Returns `RowNotPositiveDefinite` / `SchurFactorFailed` mirroring the GPU
/// host's `RidgeBumpRequired` / `SchurFactorFailed`, and `Unavailable` for the
/// matrix-free / degenerate-dim cases the dense fused path declines.
pub fn emulate_fused_arrow_newton_step(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<FusedCpuSolution, FusedCpuError> {
    let n = sys.rows.len();
    let p = sys.d;
    let r = sys.k;
    if n == 0 || p == 0 || r == 0 {
        return Err(FusedCpuError::Unavailable);
    }
    // The dense fused path requires materialised slabs (no matrix-free ops),
    // exactly as `arrow_schur::solve` re-checks before packing.
    if sys.hbb_matvec.is_some() || sys.htbeta_matvec.is_some() || sys.hbb.dim() != (r, r) {
        return Err(FusedCpuError::Unavailable);
    }

    // ---- Forward pass: emulate the per-block kernel, accumulating partials. ----
    let mut states: Vec<FusedRowState> = Vec::with_capacity(n);
    // Seed S_β with H_ββ + ρ_β I (column-major) and r_β with -g_β, matching the
    // host seed in `arrow_schur::solve`; fold per-row partials in row order.
    let mut schur = vec![0.0_f64; r * r];
    for col in 0..r {
        for row in 0..r {
            let mut v = sys.hbb[[row, col]];
            if row == col {
                v += ridge_beta;
            }
            schur[col * r + row] = v;
        }
    }
    let mut rhs: Vec<f64> = sys.gb.iter().map(|v| -v).collect();
    let mut log_det = 0.0_f64;

    let mut d_col = vec![0.0_f64; p * p];
    let mut b_col = vec![0.0_f64; p * r];
    let mut g_vec = vec![0.0_f64; p];
    let mut partial_s = vec![0.0_f64; r * r];
    let mut partial_r = vec![0.0_f64; r];
    for (i, row) in sys.rows.iter().enumerate() {
        if row.htt.dim() != (p, p) || row.htbeta.dim() != (p, r) || row.gt.len() != p {
            return Err(FusedCpuError::Unavailable);
        }
        // Column-major pack, ridge folded by the forward emulation (the kernel
        // adds ridge during the load, so pass the raw D here).
        for c in 0..p {
            for rr in 0..p {
                d_col[c * p + rr] = row.htt[[rr, c]];
            }
        }
        for c in 0..r {
            for rr in 0..p {
                b_col[c * p + rr] = row.htbeta[[rr, c]];
            }
        }
        for rr in 0..p {
            g_vec[rr] = row.gt[rr];
        }
        let state = emulate_forward_block(
            &d_col,
            &b_col,
            &g_vec,
            p,
            r,
            ridge_t,
            &mut partial_s,
            &mut partial_r,
            &mut log_det,
        )
        .map_err(|pivot| FusedCpuError::RowNotPositiveDefinite { row: i, pivot })?;
        // S_β -= Σ YᵀY ; r_β += Σ Yᵀu  (documented sign convention).
        for idx in 0..r * r {
            schur[idx] -= partial_s[idx];
        }
        for a in 0..r {
            rhs[a] += partial_r[a];
        }
        states.push(state);
    }

    // ---- Central: factor S_β (column-major lower Cholesky) and solve δβ. ----
    // Same in-place scalar Cholesky as the per-row blocks, on the K×K leaf.
    for j in 0..r {
        let mut diag = schur[j * r + j];
        for t in 0..j {
            let l_jt = schur[t * r + j];
            diag -= l_jt * l_jt;
        }
        if !(diag > 0.0) {
            return Err(FusedCpuError::SchurFactorFailed { pivot: j + 1 });
        }
        let l_jj = diag.sqrt();
        schur[j * r + j] = l_jj;
        let inv = 1.0 / l_jj;
        for row in (j + 1)..r {
            let mut s = schur[j * r + row];
            for t in 0..j {
                s -= schur[t * r + row] * schur[t * r + j];
            }
            schur[j * r + row] = s * inv;
        }
    }
    log_det += 2.0 * (0..r).map(|j| schur[j * r + j].ln()).sum::<f64>();
    // Forward then back solve S_β δβ = r_β with the lower factor.
    let mut delta_beta = rhs;
    for row in 0..r {
        let mut s = delta_beta[row];
        for t in 0..row {
            s -= schur[t * r + row] * delta_beta[t];
        }
        delta_beta[row] = s / schur[row * r + row];
    }
    for row in (0..r).rev() {
        let mut s = delta_beta[row];
        for t in (row + 1)..r {
            s -= schur[row * r + t] * delta_beta[t];
        }
        delta_beta[row] = s / schur[row * r + row];
    }

    // ---- Backward pass: per-row δt_i = -L_iᵀ⁻¹(u_i + Y_i δβ). ----
    let mut delta_t = vec![0.0_f64; n * p];
    for (i, state) in states.iter().enumerate() {
        let dt = emulate_back_sub_block(state, &delta_beta, p, r);
        delta_t[i * p..(i + 1) * p].copy_from_slice(&dt);
    }

    Ok(FusedCpuSolution {
        delta_t,
        delta_beta,
        log_det_hessian: log_det,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_to_template_picks_smallest_admissible() {
        assert_eq!(ceil_to_template_r(1), Some(4));
        assert_eq!(ceil_to_template_r(4), Some(4));
        assert_eq!(ceil_to_template_r(5), Some(5));
        assert_eq!(ceil_to_template_r(7), Some(8));
        assert_eq!(ceil_to_template_r(15), Some(16));
        assert_eq!(ceil_to_template_r(16), Some(16));
        assert_eq!(ceil_to_template_r(17), Some(20));
        assert_eq!(ceil_to_template_r(32), Some(32));
        assert_eq!(ceil_to_template_r(33), None);
    }

    #[test]
    fn fused_admission_rejects_oversize_or_zero_blocks() {
        assert!(!fused_path_admitted(0, 8, 4));
        assert!(!fused_path_admitted(4, 0, 4));
        assert!(!fused_path_admitted(4, 8, 0));
        assert!(!fused_path_admitted(4, MAX_FUSED_P + 1, 4));
        assert!(!fused_path_admitted(2, 4, 4)); // total flops ≈ 128, below 1e5, R<16
    }

    #[test]
    fn fused_admission_accepts_dense_arrow_workloads() {
        // Large-scale shape: large n, small p, moderate R. Total flops
        // 5000 · 16^3 ≈ 2e7 — well above the 1e5 admission floor.
        assert!(fused_path_admitted(5000, 16, 6));
        // Big R alone admits even on a small n.
        assert!(fused_path_admitted(4, 8, 16));
        // Multi-size sweep at the kernel's upper edge.
        assert!(fused_path_admitted(50, 30, 8));
    }

    #[test]
    fn plan_fused_launch_clamps_p_max_and_blocks_count() {
        let plan = plan_fused_launch(7, 10, 4).expect("admissible");
        assert_eq!(plan.n, 7);
        assert_eq!(plan.blocks, 7);
        assert_eq!(plan.p_runtime, 10);
        assert!(plan.p_max >= 10);
        assert_eq!(plan.r_runtime, 4);
        assert_eq!(plan.r_template, 4);
        assert_eq!(plan.partial_r_doubles, 7 * 4);
        assert_eq!(plan.partial_s_doubles, 7 * 4 * 4);

        let oversize = plan_fused_launch(4, MAX_FUSED_P + 1, 4);
        assert!(oversize.is_none(), "P exceeding ceiling must not plan");

        let bad_r = plan_fused_launch(4, 8, 33);
        assert!(
            bad_r.is_none(),
            "R exceeding template ceiling must not plan"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn forward_kernel_source_substitutes_macros() {
        let src = forward_kernel_source(16, 8);
        assert!(src.contains("#define P_MAX 16"));
        assert!(src.contains("#define R_TEMPLATE 8"));
        assert!(src.contains("arrow_schur_forward_pgroup"));
        assert!(src.contains("arrow_schur_back_sub_pgroup"));
    }

    // ----------------------------------------------------------------------
    // Device-free verification of the fused Layer D + E kernel ALGORITHM
    // (#1017). The CUDA source can only run on a GPU, but its arithmetic is
    // mirrored by `emulate_fused_arrow_newton_step`; these tests pin that
    // emulation against the dense reference so the fused kernel's correctness
    // is checkable on any host before a V100/A100 window is available.
    // ----------------------------------------------------------------------
    use crate::gpu::kernels::arrow_schur::solve_arrow_newton_step_dense_reference;
    use crate::solver::arrow_schur::ArrowSchurSystem;

    /// Deterministic SPD bordered system: per-row `H_tt = JJᵀ + (2+i)I` (PD),
    /// arbitrary `H_tβ`, SPD border `H_ββ = MMᵀ + p·I`, fixed gradients. No RNG
    /// — same inputs every run, so the test is reproducible.
    fn build_spd_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(n, d, k);
        for i in 0..n {
            for r in 0..d {
                for c in 0..d {
                    // J[r][c] = sin(r + 2c + i); H_tt = JJᵀ + (2+i)I (SPD).
                    let mut v = 0.0;
                    for m in 0..d {
                        let j_rm = ((r + 2 * m + i) as f64).sin();
                        let j_cm = ((c + 2 * m + i) as f64).sin();
                        v += j_rm * j_cm;
                    }
                    if r == c {
                        v += 2.0 + i as f64;
                    }
                    sys.rows[i].htt[[r, c]] = v;
                }
                for c in 0..k {
                    sys.rows[i].htbeta[[r, c]] = ((r + 3 * c + 2 * i) as f64).cos() * 0.5;
                }
                sys.rows[i].gt[r] = ((r + i) as f64).cos();
            }
        }
        for r in 0..k {
            for c in 0..k {
                let mut v = 0.0;
                for m in 0..k {
                    v += ((r + 2 * m) as f64).cos() * ((c + 2 * m) as f64).cos();
                }
                if r == c {
                    v += k as f64;
                }
                sys.hbb[[r, c]] = v;
            }
            sys.gb[r] = ((r + 1) as f64).sin();
        }
        sys.refresh_row_hessian_fingerprint();
        sys
    }

    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            num += (x - y) * (x - y);
            den += y * y;
        }
        (num / den.max(1e-300)).sqrt()
    }

    #[test]
    fn fused_cpu_emulation_matches_dense_reference() {
        // SAE-flavoured shapes: few wide row blocks (small d, moderate k).
        for &(n, d, k) in &[(1usize, 2usize, 4usize), (5, 3, 6), (8, 4, 5), (3, 2, 8)] {
            let sys = build_spd_system(n, d, k);
            let ridge_t = 1e-3;
            let ridge_beta = 1e-2;
            let dense = solve_arrow_newton_step_dense_reference(&sys, ridge_t, ridge_beta)
                .expect("dense reference solves the SPD system");
            let fused = emulate_fused_arrow_newton_step(&sys, ridge_t, ridge_beta)
                .expect("fused emulation solves the SPD system");

            assert!(
                rel_err(&fused.delta_t, dense.delta_t.as_slice().unwrap()) < 1e-10,
                "δt mismatch at (n={n},d={d},k={k})"
            );
            assert!(
                rel_err(&fused.delta_beta, dense.delta_beta.as_slice().unwrap()) < 1e-10,
                "δβ mismatch at (n={n},d={d},k={k})"
            );
            let ld_rel = (fused.log_det_hessian - dense.log_det_hessian).abs()
                / dense.log_det_hessian.abs().max(1.0);
            assert!(
                ld_rel < 1e-10,
                "log|H| mismatch at (n={n},d={d},k={k}): fused={} dense={}",
                fused.log_det_hessian,
                dense.log_det_hessian
            );
        }
    }

    #[test]
    fn fused_cpu_emulation_is_deterministic() {
        let sys = build_spd_system(6, 3, 5);
        let a = emulate_fused_arrow_newton_step(&sys, 1e-3, 1e-2).unwrap();
        let b = emulate_fused_arrow_newton_step(&sys, 1e-3, 1e-2).unwrap();
        // Bit-identical run-to-run: the emulation has a fixed loop/accumulation
        // order, so two calls reproduce each other exactly (the #1017
        // determinism gate). This proves run-to-run determinism for THIS
        // fixed-order path only; it does not establish that an arbitrary
        // reassociated/parallel reduction leaves a criterion ranking invariant —
        // reassociation can flip a near-tie winner within the f64 margin (#1211).
        assert_eq!(a.delta_t, b.delta_t);
        assert_eq!(a.delta_beta, b.delta_beta);
        assert_eq!(a.log_det_hessian, b.log_det_hessian);
    }

    #[test]
    fn fused_cpu_emulation_reports_non_pd_row() {
        // A row whose H_tt is indefinite and ridge too small to rescue it: the
        // emulation mirrors the kernel's `status_out[i] = j+1` non-PD branch.
        let mut sys = build_spd_system(2, 2, 4);
        sys.rows[1].htt[[0, 0]] = -5.0;
        sys.rows[1].htt[[1, 1]] = -5.0;
        sys.rows[1].htt[[0, 1]] = 0.0;
        sys.rows[1].htt[[1, 0]] = 0.0;
        sys.refresh_row_hessian_fingerprint();
        let err = emulate_fused_arrow_newton_step(&sys, 1e-6, 1e-2).unwrap_err();
        match err {
            FusedCpuError::RowNotPositiveDefinite { row, pivot } => {
                assert_eq!(row, 1);
                assert_eq!(pivot, 1, "non-positive at the first pivot");
            }
            other => panic!("expected RowNotPositiveDefinite, got {other:?}"),
        }
    }

    #[test]
    fn fused_cpu_emulation_declines_matrix_free_and_degenerate() {
        // Degenerate dims → Unavailable (the dense fused path declines).
        let empty = ArrowSchurSystem::new(0, 2, 4);
        assert_eq!(
            emulate_fused_arrow_newton_step(&empty, 1e-3, 1e-2).unwrap_err(),
            FusedCpuError::Unavailable
        );
    }

    #[test]
    fn forward_block_partials_reconstruct_g_block() {
        // The per-block partials must equal the explicit reduced contribution:
        // partial_s = Y_iᵀ Y_i where Y_i = L_i^{-1} B_i, i.e. the row's Schur
        // term B_iᵀ (D_i+ρI)^{-1} B_i. Reconstruct and cross-check directly.
        let sys = build_spd_system(1, 3, 4);
        let p = sys.d;
        let r = sys.k;
        let row = &sys.rows[0];
        let ridge_t = 1e-3;
        let mut d_col = vec![0.0; p * p];
        let mut b_col = vec![0.0; p * r];
        let mut g = vec![0.0; p];
        for c in 0..p {
            for rr in 0..p {
                d_col[c * p + rr] = row.htt[[rr, c]];
            }
        }
        for c in 0..r {
            for rr in 0..p {
                b_col[c * p + rr] = row.htbeta[[rr, c]];
            }
        }
        for rr in 0..p {
            g[rr] = row.gt[rr];
        }
        let mut ps = vec![0.0; r * r];
        let mut pr = vec![0.0; r];
        let mut ld = 0.0;
        emulate_forward_block(&d_col, &b_col, &g, p, r, ridge_t, &mut ps, &mut pr, &mut ld)
            .expect("PD block factors");

        // Direct: M = D + ρI; solve M Z = B; expect partial_s = Bᵀ Z = ZᵀMZ.
        // Build M, invert via the same scalar Cholesky path indirectly by
        // forming Bᵀ M^{-1} B through a dense solve.
        let mut m = vec![vec![0.0; p]; p];
        for rr in 0..p {
            for c in 0..p {
                m[rr][c] = row.htt[[rr, c]] + if rr == c { ridge_t } else { 0.0 };
            }
        }
        // Solve M z = b_col[:,c] by Gaussian elimination (independent of the
        // emulation's Cholesky path) → s_direct[c][c2] = z_cᵀ b_{c2}.
        let solve = |m: &Vec<Vec<f64>>, rhs: &[f64]| -> Vec<f64> {
            let mut a: Vec<Vec<f64>> = m.iter().map(|r| r.clone()).collect();
            let mut x = rhs.to_vec();
            for col in 0..p {
                let piv = a[col][col];
                for j in col..p {
                    a[col][j] /= piv;
                }
                x[col] /= piv;
                for rr in 0..p {
                    if rr != col {
                        let f = a[rr][col];
                        for j in col..p {
                            a[rr][j] -= f * a[col][j];
                        }
                        x[rr] -= f * x[col];
                    }
                }
            }
            x
        };
        for c in 0..r {
            let bc: Vec<f64> = (0..p).map(|rr| b_col[c * p + rr]).collect();
            let z = solve(&m, &bc);
            for c2 in 0..r {
                let mut dir = 0.0;
                for rr in 0..p {
                    dir += b_col[c2 * p + rr] * z[rr];
                }
                assert!(
                    (ps[c * r + c2] - dir).abs() < 1e-9,
                    "partial_s[{c}][{c2}] {} vs direct {dir}",
                    ps[c * r + c2]
                );
            }
        }
    }
}
