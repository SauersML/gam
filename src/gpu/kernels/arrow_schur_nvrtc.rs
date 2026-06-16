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

#![allow(clippy::module_name_repetitions)]

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
/// `S2ModuleCacheKey` pattern in `crate::terms::sphere_gpu`.
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
/// in `crate::terms::sphere_gpu`.
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
}
