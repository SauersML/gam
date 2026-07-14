//! Fully GPU-resident batched Arrow-Schur dense Cholesky solver.
//!
//! Implements the square-root Schur form: each local block `D_i = L_i L_i^T`
//! is factored on device, `u_i = L_i^{-1} g_i` and `Y_i = L_i^{-1} B_i` are
//! formed by triangular solves, the reduced shared system
//!     `S_β = C + ρ_β I − Σ_i Y_i^T Y_i,  r_β = -g_β + Σ_i Y_i^T u_i`
//! is assembled on device, factored once, and the back-substitution
//!     `w_i = u_i + Y_i · δβ,  L_i^T x_i = w_i,  δt_i = -x_i`
//! is run on device. Only the final `(δt, δβ, log|H|)` triple is downloaded.
//!
//! The current caller (Arrow-Schur Newton step inside PIRLS) feeds uniform
//! local block size `d` and uniform shared width `k`, so the entire pipeline
//! is dispatched as a single p-group; per-p grouping for heterogenous blocks
//! is Layer D's NVRTC fused-kernel concern and lives in this module's
//! follow-up implementation rather than in policy plumbing.
//!
//! CUDA-only probes are exported only on Linux. Platform-neutral dispatch
//! entries remain available so their callers can report a typed device decline.

use ndarray::{Array1, Array2, ArrayView2};

use crate::arrow_schur::{ArrowPcgDiagnostics, ArrowSchurSystem, DeviceSaePcgData};
use gam_linalg::triangular::{CholeskyGuard, cholesky_factor_in_place, cholesky_solve_vector};

/// Outcome of a single Arrow-Schur Newton solve.
pub struct ArrowSchurGpuSolution {
    pub delta_t: Array1<f64>,
    pub delta_beta: Array1<f64>,
    /// Natural log of the determinant of the full bordered Hessian, computed
    /// from the local Cholesky factors and the Schur factor on device.
    pub log_det_hessian: f64,
}

/// Reason a device path declined to run; lets the host caller decide between
/// CPU fallback and per-row escalation. `RidgeBumpRequired` carries the
/// estimated diagonal bump needed to clear the failed pivot.
#[derive(Debug, Clone)]
pub enum ArrowSchurGpuFailure {
    /// CUDA runtime unavailable, allocation failed, or workload below policy.
    Unavailable,
    /// A row block was not positive definite even after the requested ridge.
    /// Caller may retry with `ridge_t + bump`.
    RidgeBumpRequired { row: usize, bump: f64 },
    /// Shared Schur factor failed; bordered system is rank-deficient at the
    /// requested ridges and the CPU path should handle escalation.
    SchurFactorFailed { reason: String },
    /// The dense GPU Schur path cannot consume this system's β-block. Either
    /// the system carries matrix-free `H_ββ` / per-row `H_tβ` operators
    /// (`had_*_matvec` set), OR the dense `(K×K)` `H_ββ` block is simply absent
    /// (both flags false) — e.g. an SAE-manifold system whose β-curvature lives
    /// in a `penalty_op` / factored-frame representation with `hbb` reclaimed to
    /// a `0×0` workspace. In BOTH cases this is a capability mismatch, NOT a
    /// numerical failure: the caller should route to CPU `InexactPCG` (or supply
    /// dense buffers) rather than escalating a ridge. See `gpu/arrow_schur.rs`
    /// Part B for the planned GPU PCG path that will lift this restriction at
    /// K ≥ 5000.
    GpuRequiresDenseSystem {
        had_hbb_matvec: bool,
        had_htbeta_matvec: bool,
    },
}

/// Resolve the configured runtime without conflating probe faults with an
/// ordinary device decline. The existing GPU failure surface has a diagnostic
/// string variant, so faults flow through it; only typed Auto/Off absence
/// remains `Ok(None)` and may become `Unavailable` at a caller-specific gate.
fn resolve_runtime_for_device_path(
) -> Result<Option<&'static gam_gpu::GpuRuntime>, ArrowSchurGpuFailure> {
    gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy()).map_err(|error| {
        ArrowSchurGpuFailure::SchurFactorFailed {
            reason: format!("GPU runtime resolution failed: {error}"),
        }
    })
}

/// Relative rounding margin (multiplier on `diag_scale · √ε`) added on top of
/// the deficit-clearing shift in [`ridge_bump_to_make_pd`].
///
/// The exact shift `-(λ_min)` makes a block PD in exact arithmetic, but a
/// single retry at precisely that magnitude is routinely re-rejected by the
/// next POTRF because the rounding error of forming `D + ridge·I` and
/// re-factoring is itself O(√ε). The 1024× headroom (≈ 2¹⁰, ten extra bits
/// below the f64 mantissa's 52) clears the pivot on the first retry without
/// materially perturbing the curvature the Newton step sees. Shared by every
/// per-row / batched / fused producer so they suggest a consistent bump.
const RIDGE_BUMP_EPS_MARGIN: f64 = 1024.0;

/// Diagonal ridge bump that is GUARANTEED to make `H_tt + (ridge_t + bump)·I`
/// positive definite for a *symmetric* per-row block, sized from the block's
/// own entries rather than from the factorization's pivot index.
///
/// # Why the old `scale · |pivot| · √ε · 1024` estimate is wrong
///
/// The batched/fused device paths derive the suggested bump from the
/// factorization "pivot" — but cuSOLVER's `potrf` (and the NVRTC kernel's
/// status code) report the failing pivot as a **1-based row index**, NOT the
/// magnitude of the negative pivot. A block that is indefinite by `O(1)`
/// (e.g. `H_tt = -I`, whose smallest eigenvalue is `-1`) then yields the same
/// `bump ≈ √ε · 1024 ≈ 1.5e-5` as a block that is indefinite by `O(√ε)`. The
/// outer LM escalation, which retries at `ridge_t + bump` and grows
/// geometrically with a bounded step count, can never lift a strongly
/// indefinite block out of the negative regime, so the solve fails to recover
/// even though the block is trivially regularizable. (Surfaced by the V100
/// `ridge_bump_required_on_non_pd_row_recovers_after_bump` validation test.)
///
/// # The bound
///
/// By the Gershgorin circle theorem every eigenvalue `λ` of the symmetric
/// matrix `A = H_tt` satisfies, for some row `i`,
///   `λ ≥ A[i,i] − Σ_{j≠i} |A[i,j]|`,
/// so `λ_min(A) ≥ min_i ( A[i,i] − Σ_{j≠i} |A[i,j]| ) =: g` (the most negative
/// Gershgorin left edge). Adding `t·I` shifts every eigenvalue up by `t`, so
/// `A + t·I` is PD as soon as `t > -g`. We are already sitting at `ridge_t`, so
/// the ADDITIONAL bump needed is `-(g + ridge_t)` when that is positive. We add
/// a relative safety margin (`√ε · scale · 1024`, the same headroom the legacy
/// estimate used) so the re-factored, rounding-perturbed block clears the pivot
/// on the first retry, and a `max(1)`-scaled floor so a marginally-indefinite
/// block still gets a strictly positive, non-vanishing bump.
///
/// The returned value is the bump to ADD to the current `ridge_t`. It is always
/// strictly positive (the caller only constructs `RidgeBumpRequired` on an
/// actual non-PD failure, but the bound is defensive regardless).
#[must_use]
fn ridge_bump_to_make_pd(htt: ArrayView2<'_, f64>, ridge_t: f64) -> f64 {
    let d = htt.nrows();
    // Diagonal magnitude scale (also the legacy `scale`), and the most-negative
    // Gershgorin left edge `g = min_i (A_ii − Σ_{j≠i} |A_ij|)`.
    let mut scale = 1.0_f64;
    let mut min_gershgorin_edge = f64::INFINITY;
    for i in 0..d {
        let diag = htt[[i, i]];
        scale = scale.max(diag.abs());
        let mut off_sum = 0.0_f64;
        for j in 0..d {
            if j != i {
                off_sum += htt[[i, j]].abs();
            }
        }
        min_gershgorin_edge = min_gershgorin_edge.min(diag - off_sum);
    }
    if !min_gershgorin_edge.is_finite() {
        // d == 0 (no rows) or non-finite entries: fall back to the scale-only
        // floor so the caller still gets a strictly positive bump.
        return scale * f64::EPSILON.sqrt() * RIDGE_BUMP_EPS_MARGIN;
    }
    // Additional shift needed so `λ_min(A) + ridge_t + bump > 0`, i.e.
    // `bump > -(min_gershgorin_edge + ridge_t)`.
    let deficit = -(min_gershgorin_edge + ridge_t);
    let margin = scale * f64::EPSILON.sqrt() * RIDGE_BUMP_EPS_MARGIN;
    // Lift past the deficit (when positive) plus a rounding margin; never below
    // the scale-relative floor so a marginal block still moves.
    deficit.max(0.0) + margin
}

/// [`ridge_bump_to_make_pd`] for a `d × d` symmetric block stored column-major
/// in a flat slice with the current ridge ALREADY baked into the diagonal
/// (the device packers emit `D = H_tt + ridge_t·I` this way). Because the shift
/// is already present, the Gershgorin bound is taken at `ridge_t = 0` and the
/// returned value is still the ADDITIONAL bump to add on top of the current
/// ridge. Returns the scale-only floor when `block` is mis-sized.
// The column-major bump is a helper for the device tile packers, which only
// exist on the linux CUDA path (`mod cuda`, `#[cfg(target_os = "linux")]`). It
// therefore lives where it is used: gate it to linux. Its parity unit test is
// gated to linux to match (a `test` token in the cfg would trip the build.rs
// `#[cfg(test)]`-on-a-src-item ban; including `test` to dodge the non-linux
// dead_code lint is exactly the escape hatch that ban forbids).
#[cfg(target_os = "linux")]
#[must_use]
fn ridge_bump_to_make_pd_colmajor(block: &[f64], d: usize) -> f64 {
    if d == 0 || block.len() < d * d {
        return f64::EPSILON.sqrt() * RIDGE_BUMP_EPS_MARGIN;
    }
    // Column-major: element (row r, col c) at block[c*d + r]. The matrix is
    // symmetric, so reading by column gives the same Gershgorin edges as by row.
    let mut scale = 1.0_f64;
    let mut min_gershgorin_edge = f64::INFINITY;
    for i in 0..d {
        let diag = block[i * d + i];
        scale = scale.max(diag.abs());
        let mut off_sum = 0.0_f64;
        for j in 0..d {
            if j != i {
                off_sum += block[j * d + i].abs();
            }
        }
        min_gershgorin_edge = min_gershgorin_edge.min(diag - off_sum);
    }
    let margin = scale * f64::EPSILON.sqrt() * RIDGE_BUMP_EPS_MARGIN;
    if !min_gershgorin_edge.is_finite() {
        return margin;
    }
    (-min_gershgorin_edge).max(0.0) + margin
}

/// Entry point: attempt the fully device-resident Arrow-Schur Newton solve.
/// Returns `Err(ArrowSchurGpuFailure::Unavailable)` to indicate "device path
/// declined, fall back to CPU" — never panics.
pub fn solve_arrow_newton_step(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;

    // Detect matrix-free operators before any dim() checks so callers get a
    // clear, actionable error instead of a generic SchurFactorFailed. The GPU
    // dense-Schur path requires materialised H_ββ and per-row H_tβ slabs;
    // CPU InexactPCG is the correct fallback when either operator is abstract.
    let had_hbb_matvec = sys.hbb_matvec.is_some();
    let had_htbeta_matvec = sys.htbeta_matvec.is_some();
    if had_hbb_matvec || had_htbeta_matvec {
        return Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem {
            had_hbb_matvec,
            had_htbeta_matvec,
        });
    }

    // A `penalty_op` is the AUTHORITATIVE β-curvature source whenever it is
    // installed: assembly bypasses the dense `hbb` accumulator (and, for
    // frames-engaged SAE systems, reclaims it to a 0×0 workspace), so whatever
    // `hbb` survives is STALE relative to the operator. The dense device Schur
    // path reads ONLY `hbb`, so it must decline here — BEFORE the shape gate
    // below — even when a stale `(k, k)` `hbb` would pass that gate: proceeding
    // would silently compute the WRONG Newton step from stale curvature instead
    // of routing to the CPU matrix-free lane (which reads `penalty_op`) that
    // returns the correct one. The production caller `try_device_arrow_direct`
    // already short-circuits this shape, but enforcing it at the entry keeps
    // EVERY caller covered — the resident / reupload harnesses and any future
    // direct caller — so no path can reach the device solve with a stale dense
    // block. Both matvec flags are false here: control only reaches this point
    // when neither matrix-free operator is installed (returned above), and the
    // frames-engaged path installs `penalty_op` with no `hbb_matvec` /
    // `htbeta_matvec`.
    if sys.penalty_op.is_some() {
        return Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem {
            had_hbb_matvec: false,
            had_htbeta_matvec: false,
        });
    }

    if sys.hbb.dim() != (k, k) {
        // The dense (K×K) H_ββ block is absent (e.g. an SAE-manifold system
        // whose β-curvature is carried by a matrix-free `penalty_op` /
        // factored-frame representation, with `hbb` reclaimed to a 0×0
        // workspace at the end of assembly). This is a CAPABILITY decline, not
        // a numerical failure: the dense device Schur path simply cannot
        // consume this system, so the host must route it to the CPU lane
        // (which reads the matrix-free operators) exactly as it does for the
        // `hbb_matvec` / `htbeta_matvec` case above. Returning `SchurFactorFailed`
        // here would masquerade as a non-PD/rank-deficient factorization and be
        // escalated (and ultimately surfaced as a FATAL RemlConvergenceError)
        // by the outer LM loop instead of falling back. Decline instead. Both
        // matvec flags are false because the absence is structural (no dense
        // block was materialized), not caused by an installed matrix-free op.
        return Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem {
            had_hbb_matvec: false,
            had_htbeta_matvec: false,
        });
    }
    if n == 0 || d == 0 {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    if sys
        .rows
        .iter()
        .any(|row| row.htt.dim() != (d, d) || row.htbeta.dim() != (d, k) || row.gt.len() != d)
    {
        return Err(ArrowSchurGpuFailure::SchurFactorFailed {
            reason: "row block dimension mismatch".to_string(),
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        if ridge_t.is_nan() || ridge_beta.is_nan() {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: "ridge is NaN".to_string(),
            });
        }
        Err(ArrowSchurGpuFailure::Unavailable)
    }

    #[cfg(target_os = "linux")]
    {
        // Multi-GPU: the arrow-Schur solve is row-block separable in its forward
        // (per-row factor / whiten / partial-Schur) and backward (per-row
        // back-sub) phases — only the small shared K×K reduce+factor+δβ is
        // central. When more than one device is usable, split the WHOLE solve at
        // row-block granularity across all GPUs. The POTRF stays fused with its
        // dependent TRSM+GEMM on each tile's own stream, so no on-stream solve is
        // orphaned. On `Unavailable` (one device, shape below policy, transient)
        // fall through to the single-device fused / Layer-A paths below.
        if resolve_runtime_for_device_path()?
            .map(gam_gpu::device_runtime::GpuRuntime::device_count)
            .unwrap_or(0)
            > 1
        {
            match cuda::solve_multi_gpu(sys, ridge_t, ridge_beta) {
                Ok(sol) => return Ok(sol),
                Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump }) => {
                    return Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump });
                }
                Err(ArrowSchurGpuFailure::SchurFactorFailed { reason }) => {
                    return Err(ArrowSchurGpuFailure::SchurFactorFailed { reason });
                }
                // Unavailable / GpuRequiresDenseSystem: fall through to the
                // single-device paths (already shape-validated above).
                Err(_) => {}
            }
        }
        // Layer D admission: when the system shape passes the
        // (Σ p³ ≥ 1e5 OR R ≥ 16) heuristic and `p ≤ MAX_FUSED_P`, the fused
        // NVRTC kernel replaces the cuSOLVER/cuBLAS Layer A+B+C path with a
        // single per-row block. Layer C↔D parity (math block 3 §16 test 6)
        // requires both paths to agree to 1e-10 on identical inputs.
        if crate::gpu_kernels::arrow_schur_nvrtc::system_admits_fused_path(sys) {
            match cuda::solve_fused(sys, ridge_t, ridge_beta) {
                Ok(sol) => return Ok(sol),
                // RidgeBumpRequired must surface to the outer escalation loop —
                // the fused path's pivot diagnostic is identical in semantics
                // to the cuSOLVER batched POTRF info code.
                Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump }) => {
                    return Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump });
                }
                // Any other failure (Unavailable, SchurFactorFailed) falls
                // through to the unfused path so a flaky NVRTC compile or
                // shared-mem allocation does not abort the outer Newton step.
                Err(_) => {}
            }
        }
        cuda::solve(sys, ridge_t, ridge_beta)
    }
}

/// Build the stacked column-major D buffer (n local d×d blocks), the stacked
/// stacked B buffer (n local d×k blocks), and the stacked g buffer
/// (n local d-vectors) consumed by the device pipeline. Each block is laid
/// out column-major so a single allocation + `cuMemcpyHtoD` reaches the
/// device without per-row dispatch overhead.
#[cfg(target_os = "linux")]
fn pack_host(sys: &ArrowSchurSystem, ridge_t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    let mut d_buf = Vec::with_capacity(n * d * d);
    let mut b_buf = Vec::with_capacity(n * d * k);
    let mut g_buf = Vec::with_capacity(n * d);
    for row in &sys.rows {
        pack_block(row, ridge_t, d, k, &mut d_buf, &mut b_buf, &mut g_buf);
    }
    (d_buf, b_buf, g_buf)
}

#[cfg(target_os = "linux")]
#[inline]
fn pack_block(
    row: &crate::arrow_schur::ArrowRowBlock,
    ridge_t: f64,
    d: usize,
    k: usize,
    d_buf: &mut Vec<f64>,
    b_buf: &mut Vec<f64>,
    g_buf: &mut Vec<f64>,
) {
    for col in 0..d {
        for r in 0..d {
            let mut value = row.htt[[r, col]];
            if r == col {
                value += ridge_t;
            }
            d_buf.push(value);
        }
    }
    for col in 0..k {
        for r in 0..d {
            b_buf.push(row.htbeta[[r, col]]);
        }
    }
    for r in 0..d {
        g_buf.push(row.gt[r]);
    }
}

/// Entry that forces the Layer D + E fused NVRTC path regardless of the
/// admission heuristic. Used by the V100 Layer C↔D parity harness to drive
/// the fused kernel at small shapes the heuristic would otherwise route through
/// the cuSOLVER/cuBLAS Layer A+B+C path. The symbol only exists on the target
/// that provides the NVRTC implementation.
#[doc(hidden)]
#[cfg(target_os = "linux")]
pub fn solve_arrow_newton_step_fused_force(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
    if ridge_t.is_nan() || ridge_beta.is_nan() {
        return Err(ArrowSchurGpuFailure::SchurFactorFailed {
            reason: "ridge is NaN".to_string(),
        });
    }
    if crate::gpu_kernels::arrow_schur_nvrtc::plan_fused_launch(sys.rows.len(), sys.d, sys.k)
        .is_none()
    {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    cuda::solve_fused(sys, ridge_t, ridge_beta)
}

/// #1017 Phase 3: a device-resident Arrow-Schur frame whose constant Hessian
/// blocks (`D = H_tt`, `B = H_tβ`, border `H_ββ`) and their factors stay on the
/// device across the inner Newton loop. Construct once per frozen gate/basis
/// frame, then call [`ResidentArrowFrameHandle::solve_gradient`] once per
/// iterate with the fresh residual gradient — only the `O(n·d + p)` gradient
/// crosses to the device and only `δ` crosses back, in contrast to
/// [`solve_arrow_newton_step`] which re-uploads and re-factors the full system
/// every call. On a non-CUDA host construction returns
/// `ArrowSchurGpuFailure::Unavailable`.
#[cfg(target_os = "linux")]
pub struct ResidentArrowFrameHandle {
    inner: cuda::ResidentArrowFrame,
}

/// The resident CUDA frame has no value on hosts that cannot construct it.
/// Keeping this type uninhabited preserves the fail-loud platform contract
/// without exposing a fake non-CUDA implementation.
#[cfg(not(target_os = "linux"))]
pub enum ResidentArrowFrameHandle {}

impl ResidentArrowFrameHandle {
    /// Upload the constant Hessian blocks and perform the one-time factor work.
    pub fn new(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<Self, ArrowSchurGpuFailure> {
        // The dense device path requires materialised blocks, same admission as
        // `solve_arrow_newton_step`.
        if sys.hbb_matvec.is_some() || sys.htbeta_matvec.is_some() {
            return Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem {
                had_hbb_matvec: sys.hbb_matvec.is_some(),
                had_htbeta_matvec: sys.htbeta_matvec.is_some(),
            });
        }
        #[cfg(not(target_os = "linux"))]
        {
            if ridge_t.is_nan() || ridge_beta.is_nan() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: "ridge is NaN".to_string(),
                });
            }
            Err(ArrowSchurGpuFailure::Unavailable)
        }
        #[cfg(target_os = "linux")]
        {
            Ok(Self {
                inner: cuda::ResidentArrowFrame::new(sys, ridge_t, ridge_beta)?,
            })
        }
    }

    /// Solve `H δ = −gradient` for a fresh gradient reusing the resident factors.
    pub fn solve_gradient(
        &self,
        g_t: &[f64],
        g_beta: &[f64],
    ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
        #[cfg(not(target_os = "linux"))]
        {
            if g_t.iter().chain(g_beta).any(|v| !v.is_finite()) {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: "non-finite gradient entry".to_string(),
                });
            }
            Err(ArrowSchurGpuFailure::Unavailable)
        }
        #[cfg(target_os = "linux")]
        {
            self.inner.solve_gradient(g_t, g_beta)
        }
    }

    /// `log|H|` for the frame (constant; depends only on the factored Hessian).
    #[must_use]
    pub fn log_det_hessian(&self) -> f64 {
        #[cfg(not(target_os = "linux"))]
        {
            match *self {}
        }
        #[cfg(target_os = "linux")]
        {
            self.inner.log_det_hessian()
        }
    }
}

/// #1017: a BASE-block-resident Arrow-Schur frame for the LM ridge ladder.
///
/// Unlike [`ResidentArrowFrameHandle`] — which BAKES one ridge into its factors
/// and then serves cheap re-solves for a NEW GRADIENT at that SAME ridge — this
/// frame holds the ridge-INDEPENDENT base blocks (`D = H_tt`, `B = H_tβ`, border
/// `H_ββ`, gradient) resident and RE-FACTORS on-device at each requested
/// `(ridge_t, ridge_beta)`. That is the regime `solve_with_lm_escalation_inner`
/// actually runs: its trials re-solve the SAME system (same gradient) at
/// ESCALATING ridges, so the factor changes every trial but the base blocks do
/// not. The base blocks upload ONCE; each trial pays only a device-to-device
/// copy of the base blocks into scratch, an on-device diagonal ridge add, and the
/// factor/solve — in place of the full `O(n·d·k)` host→device re-upload that
/// [`solve_arrow_newton_step`] performs every trial. The per-trial numerics are
/// bit-identical to that re-upload path (same POTRF/TRSM/Schur/back-sub order).
#[cfg(target_os = "linux")]
pub struct ResidentBaseArrowFrameHandle {
    inner: cuda::ResidentBaseArrowFrame,
}

/// The base-resident CUDA frame is unavailable, rather than emulated, on a
/// non-CUDA host.
#[cfg(not(target_os = "linux"))]
pub enum ResidentBaseArrowFrameHandle {}

impl ResidentBaseArrowFrameHandle {
    /// Upload the ridge-independent base blocks once. No factorization runs here;
    /// each [`Self::refactor_and_solve`] performs the ridge-dependent factor+solve.
    /// The dense device path requires materialised blocks, so a matrix-free
    /// `H_ββ` / `H_tβ` operator is rejected (same admission as
    /// [`solve_arrow_newton_step`]).
    pub fn new(sys: &ArrowSchurSystem) -> Result<Self, ArrowSchurGpuFailure> {
        if sys.hbb_matvec.is_some() || sys.htbeta_matvec.is_some() {
            return Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem {
                had_hbb_matvec: sys.hbb_matvec.is_some(),
                had_htbeta_matvec: sys.htbeta_matvec.is_some(),
            });
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(ArrowSchurGpuFailure::Unavailable)
        }
        #[cfg(target_os = "linux")]
        {
            Ok(Self {
                inner: cuda::ResidentBaseArrowFrame::new(sys)?,
            })
        }
    }

    /// Factor the resident base blocks at `(ridge_t, ridge_beta)` and solve
    /// `(H + ridge)·δ = −gradient`. Only the two ridge scalars and the tiny
    /// re-diagonalised `D` cross to the device; only `δ` crosses back. A non-PD
    /// per-row block surfaces as [`ArrowSchurGpuFailure::RidgeBumpRequired`] so
    /// the LM escalation bumps and retries at the larger ridge exactly as the
    /// re-upload path does.
    pub fn refactor_and_solve(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
        #[cfg(not(target_os = "linux"))]
        {
            if ridge_t.is_nan() || ridge_beta.is_nan() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: "ridge is NaN".to_string(),
                });
            }
            Err(ArrowSchurGpuFailure::Unavailable)
        }
        #[cfg(target_os = "linux")]
        {
            self.inner.refactor_and_solve(ridge_t, ridge_beta)
        }
    }
}

/// Build a GPU-backed Schur matvec closure for CPU-driven PCG at K ≥ 5000.
///
/// Runs the fused NVRTC forward kernel once on the dense per-row `H_tβ` slabs
/// to compute `Y_i = L_i^{-1} H_tβ^(i)` for all rows, persists the `Y_i`
/// factors in a host-side buffer, and returns an `Arc<dyn Fn(...)>` closure
/// that computes the full Schur matvec
///
/// ```text
/// S·x = (H_ββ + ridge_beta·I)·x  −  Σ_i Y_i^T (Y_i·x)
/// ```
///
/// each time it is called. At K ≥ 5000 the `Σ_i Y_i^T (Y_i·x)` term
/// dominates over the host↔device transfer of the K-vector `x`, so the GPU
/// path is a clear win even with per-iteration transfer.
///
/// `H_ββ·x` is evaluated on the CPU using `sys.hbb_matvec` when present (the
/// matrix-free hook for SAE-manifold scale callers) or the dense `sys.hbb`
/// block otherwise. The `Y_i` term uses cuBLAS batched GEMV device-side; only
/// `x` (K doubles) and `out` (K doubles) cross the host↔device boundary per
/// PCG iteration.
///
/// Returns `Err(ArrowSchurGpuFailure::Unavailable)` if CUDA is unavailable or
/// the system shape is outside the fused kernel's admission range (e.g.
/// `d > MAX_FUSED_P = 32` or no CUDA context). Callers should fall back to CPU
/// `InexactPCG` on `Unavailable`.
///
/// Returns `Err(ArrowSchurGpuFailure::RidgeBumpRequired)` if a per-row Cholesky
/// factor failed at the requested `ridge_t`; the outer LM escalation should
/// bump `ridge_t` and retry.
///
/// # Composition with the matrix-free SAE Kronecker operator
///
/// When `sys.htbeta_matvec` is set (matrix-free `H_tβ` Kronecker operator),
/// the dense `H_tβ` slabs are absent — the dense forward kernel above cannot
/// run, and at `K = 100K` the dense `Y_i = L_i^{-1} H_tβ^(i)` (`d × K` per row)
/// could not be materialised anyway. Instead, `build_row_procedural_matvec`
/// returns a row-procedural Schur matvec: per row it gathers
/// `v_i = H_tβ^(i)·x` through the forward operator (sparse `O(m_i · p)`),
/// solves `(H_tt^(i) + ρ_t·I)^{-1} v_i` through a pre-computed per-row Cholesky
/// factor, and scatters `H_βt^(i)·w_i` through the sparse transpose operator
/// (`O(m_i · p)`, replacing the old `O(K)` column-probe). This is the
/// row-procedural `a_ik · Φ_k[i,m]` Kronecker apply over the active atoms only.
pub fn gpu_schur_matvec_backend(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<crate::arrow_schur::GpuSchurMatvec, ArrowSchurGpuFailure> {
    // Matrix-free H_tβ operator present: drive the row-procedural sparse
    // Kronecker apply (active atoms only) instead of the dense forward kernel.
    if sys.htbeta_matvec.is_some() {
        return build_row_procedural_matvec(sys, ridge_t, ridge_beta);
    }

    #[cfg(not(target_os = "linux"))]
    {
        // No CUDA runtime on non-Linux. NaN ridges are validated to ensure the
        // same contract as the Linux path.
        if ridge_t.is_nan() || ridge_beta.is_nan() {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        Err(ArrowSchurGpuFailure::Unavailable)
    }

    #[cfg(target_os = "linux")]
    {
        cuda::build_schur_matvec_backend(sys, ridge_t, ridge_beta)
    }
}

/// #1017 evidence lane: a device-resident, RUN-TO-RUN DETERMINISTIC framed
/// reduced-Schur `S·v` for the SLQ/surrogate `log|S|` matvec, or `None` when the
/// device/shape declines. CPU and non-Linux always return `None`, so the evidence
/// matvec is byte-identical there. Unlike `gpu_schur_matvec_backend` (whose
/// matrix-free branch returns the CPU row-procedural closure), this engages the
/// resident device apply — but only via an atomics-free reduction, so it upholds
/// `slq_reduced_schur_log_det`'s determinism contract.
pub fn build_framed_resident_evidence_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    apply_budget: usize,
) -> Option<crate::arrow_schur::GpuSchurMatvec> {
    #[cfg(not(target_os = "linux"))]
    {
        // No CUDA runtime exists off Linux. Refuse the same degenerate
        // requests the Linux path would (mirroring the stubs above), then
        // report that no resident evidence matvec is available.
        if sys.k == 0 || !ridge_t.is_finite() || !ridge_beta.is_finite() || apply_budget == 0 {
            return None;
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        cuda::build_framed_resident_evidence_matvec(sys, ridge_t, ridge_beta, apply_budget)
    }
}

/// Build a row-procedural reduced-Schur matvec for matrix-free SAE Kronecker
/// systems, eliminating the per-row latent block via cached per-row Cholesky
/// factors and applying the cross-block through the sparse forward/transpose
/// Kronecker operators (active atoms only).
///
/// The returned closure evaluates
/// `S·x = (H_ββ + ρ_β·I)·x − Σ_i H_βt^(i) (H_tt^(i) + ρ_t·I)^{-1} H_tβ^(i)·x`,
/// the same reduced Schur complement the dense path forms, but never
/// materialises the `d × K` cross-block `H_tβ^(i)`: the forward operator
/// (`out = H_tβ^(i)·x`) and transpose operator (`out += H_βt^(i)·v`) are the
/// sparse Kronecker gather/scatter from `SaeKroneckerRows`. The per-row factor
/// of `H_tt^(i) + ρ_t·I` is computed once when the closure is built and reused
/// across every CG iteration.
///
/// Returns `RidgeBumpRequired` if a per-row block is not positive definite at
/// the requested `ridge_t`; the outer LM escalation bumps `ridge_t` and retries.
fn build_row_procedural_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<crate::arrow_schur::GpuSchurMatvec, ArrowSchurGpuFailure> {
    use std::sync::Arc;
    let n = sys.rows.len();
    let k = sys.k;
    let forward = sys
        .htbeta_matvec
        .clone()
        .ok_or(ArrowSchurGpuFailure::Unavailable)?;
    let transpose = sys.htbeta_transpose_matvec.clone().ok_or_else(|| {
        // A forward operator without its sparse adjoint cannot be applied
        // row-procedurally; this is a wiring error, surfaced as a Schur failure
        // so the caller routes to the dense CPU path rather than misreporting a
        // numerical bump.
        ArrowSchurGpuFailure::SchurFactorFailed {
            reason: "row-procedural Schur matvec requires htbeta_transpose_matvec; \
                     forward operator installed without its sparse adjoint"
                .to_string(),
        }
    })?;

    // Pre-factor each per-row block H_tt^(i) + ρ_t·I = L_i L_iᵀ on the host.
    // The blocks are tiny (d_i ≲ 32) and the dense cross-block slabs are
    // absent, so there is no device forward-kernel work to amortise here; the
    // GPU win is the reduced K-system solve in `solve_reduced_beta_pcg`.
    let mut factors: Vec<Array2<f64>> = Vec::with_capacity(n);
    for (i, row) in sys.rows.iter().enumerate() {
        let di = row.htt.nrows();
        if row.htt.ncols() != di || row.gt.len() != di {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("row {i}: malformed H_tt block {:?}", row.htt.dim()),
            });
        }
        let mut block = row.htt.clone();
        for r in 0..di {
            block[[r, r]] += ridge_t;
        }
        let factor = cholesky_factor_in_place(block.view(), CholeskyGuard::NonnegativePivot)
            .ok_or_else(|| {
                // Deficit-aware bump from the block's own entries (Gershgorin),
                // so the outer LM escalation lifts a strongly-indefinite block
                // out of the negative regime in one retry.
                ArrowSchurGpuFailure::RidgeBumpRequired {
                    row: i,
                    bump: ridge_bump_to_make_pd(row.htt.view(), ridge_t),
                }
            })?;
        factors.push(factor);
    }

    // The SAE-manifold β-Hessian lives in the structured penalty operator
    // (data-fit Gauss-Newton `G ⊗ I_p` + smoothness Kronecker blocks + any
    // dense analytic-β residual), NOT in the dense `hbb` accumulator — for
    // matrix-free systems `hbb` is zero/absent. Capture the effective penalty
    // operator so `H_ββ·x` matches the CPU `schur_matvec` path exactly. The
    // operator's `matvec` adds (`y += P x`), so seed `out` from the ridge term.
    let penalty_op = sys.effective_penalty_op();
    let row_dims: Vec<usize> = sys.rows.iter().map(|row| row.htt.nrows()).collect();

    let closure: crate::arrow_schur::GpuSchurMatvec =
        Arc::new(move |x: &Array1<f64>, out: &mut Array1<f64>| {
            assert_eq!(x.len(), k, "row-procedural matvec: x.len() != k");
            assert_eq!(out.len(), k, "row-procedural matvec: out.len() != k");

            // (H_ββ + ρ_β·I)·x into out. Seed with the ridge term, then add the
            // structured penalty-side product (penalty_op.matvec is additive).
            {
                let x_slice = x.as_slice().expect("x must be contiguous");
                let out_slice = out.as_slice_mut().expect("out must be contiguous");
                for a in 0..k {
                    out_slice[a] = ridge_beta * x_slice[a];
                }
                penalty_op.matvec(x_slice, out_slice);
            }

            // out -= Σ_i H_βt^(i) (H_tt^(i) + ρ_t·I)^{-1} H_tβ^(i)·x.
            //
            // #1017: this row-procedural reduced-Schur term is the matrix-free
            // SAE path's matvec hot loop (`build_row_procedural_matvec` is the
            // host backend `gpu_schur_matvec_backend` returns when the dense
            // `H_tβ` slabs are absent — the production Qwen shape). At
            // (n≈2000 rows) it ran SERIALLY on one core and allocated a fresh
            // length-`K` `neg` plus per-row `v_i`/`w_i` on EVERY CG iteration —
            // tens of thousands of tiny heap allocations across a solve. Each
            // row contributes an independent length-`K` scatter, so the sum is
            // embarrassingly parallel; fan it across rayon over fixed row chunks
            // and fold the per-chunk length-`K` partials in chunk order so the
            // f64 reduction is deterministic (bit-identical run-to-run)
            // regardless of thread scheduling — it agrees with the serial sum up
            // to ULP-scale chunk reassociation (the #1017 verification gate).
            // Because that reassociation is a real (if tiny) departure from
            // serial, the criterion ranking across topology candidates is stable
            // except for candidates separated by less than the reassociation
            // margin, where the near-tie winner can flip — not an exact no-move
            // guarantee (#1211). Stay
            // sequential below
            // `SCHUR_MATVEC_PARALLEL_ROW_MIN` rows and when already inside a
            // rayon worker (the topology race fans candidates with
            // `run_topology_race_parallel`) — the same nested-rayon guard the
            // CPU `schur_matvec` uses. Buffers (`v_i`, `neg`) are reused across
            // rows within a chunk, so the per-row allocation churn is gone.
            let parallel = n >= crate::arrow_schur::SCHUR_MATVEC_PARALLEL_ROW_MIN
                && rayon::current_thread_index().is_none();
            if parallel {
                use rayon::prelude::*;
                const CHUNK: usize = 64;
                let partials: Vec<Array1<f64>> = (0..n)
                    .into_par_iter()
                    .chunks(CHUNK)
                    .map(|idxs| {
                        // One length-`K` scatter accumulator per chunk; the
                        // per-row latent vector `v_i` (length `d_i ≲ 32`) is the
                        // only per-row buffer, sized to the row's own `d_i`.
                        let mut neg = Array1::<f64>::zeros(k);
                        for i in idxs {
                            let di = row_dims[i];
                            // v_i = H_tβ^(i)·x (sparse Kronecker gather).
                            let mut v_i = Array1::<f64>::zeros(di);
                            forward(i, x.view(), &mut v_i);
                            // w_i = (H_tt^(i) + ρ_t·I)^{-1} v_i via L_i L_iᵀ.
                            let w_i = cholesky_solve_vector(factors[i].view(), v_i.view());
                            // neg += H_βt^(i)·w_i (sparse scatter).
                            transpose(i, w_i.view(), &mut neg);
                        }
                        neg
                    })
                    .collect();
                // #1017/#1175 floating-point parity contract: Rayon may
                // schedule chunks on any worker, but `.chunks(CHUNK).collect()`
                // returns partials in chunk-index order. Each chunk's row sum
                // is formed locally in increasing row order, then chunk
                // partials are folded left-to-right below. That makes the
                // parallel row-procedural Schur term deterministic for a fixed
                // input and chunking (no scheduling-dependent gather/scatter
                // reordering), but it is not required to be bit-identical to
                // the serial path because additions are reassociated at chunk
                // boundaries. CPU/GPU validation should therefore allow
                // ULP-scale drift while expecting stable run-to-run results.
                let mut neg = Array1::<f64>::zeros(k);
                for part in &partials {
                    for a in 0..k {
                        neg[a] += part[a];
                    }
                }
                for a in 0..k {
                    out[a] -= neg[a];
                }
            } else {
                // Serial path: reuse one `neg` and one `v_i` across rows.
                let mut neg = Array1::<f64>::zeros(k);
                for i in 0..n {
                    let di = row_dims[i];
                    // v_i = H_tβ^(i)·x (sparse Kronecker gather, length d_i).
                    let mut v_i = Array1::<f64>::zeros(di);
                    forward(i, x.view(), &mut v_i);
                    // w_i = (H_tt^(i) + ρ_t·I)^{-1} v_i via L_i L_iᵀ.
                    let w_i = cholesky_solve_vector(factors[i].view(), v_i.view());
                    // neg += H_βt^(i)·w_i (sparse scatter); subtract once at end.
                    transpose(i, w_i.view(), &mut neg);
                }
                for a in 0..k {
                    out[a] -= neg[a];
                }
            }
        });

    Ok(closure)
}

/// Solve the reduced shared β-system `S·δβ = r` fully on device with a
/// Jacobi-preconditioned conjugate-gradient (Steihaug truncated-CG) loop.
///
/// `S` is the already-reduced symmetric positive-definite `K × K` Schur
/// complement the streaming SAE joint fit accumulates across minibatches
/// (`StreamingArrowSchur::take_accumulators` summed over chunks, with the
/// global β ridge folded in). The per-row latent blocks have already been
/// eliminated into `S` on the host streaming path; the device's job is the
/// dense `K`-dimensional solve, which is the dominant cost at `K = 100K`.
///
/// The dense `S·p` matvec runs on device via cuBLAS `Dgemv`, and the PCG state
/// vectors (`x`, `r`, `z`, `p`, `S·p`) remain device-resident for the solve.
/// Jacobi preconditioning is an elementwise CUDA kernel; only convergence
/// scalars (`pᵀSp`, `rᵀz`, `‖r‖`) cross the host boundary per iteration, plus the
/// final solution vector.
///
/// Returns `Err(ArrowSchurGpuFailure::Unavailable)` when CUDA is unavailable
/// or the workload is below the dispatch policy; the caller then runs the CPU
/// reduced-β solve. Returns `Err(ArrowSchurGpuFailure::SchurFactorFailed)`
/// when `S` carries a non-positive Jacobi diagonal (caller escalates the
/// proximal ridge).
pub fn solve_reduced_beta_pcg(
    s_acc: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    max_iterations: usize,
    relative_tolerance: f64,
) -> Result<Array1<f64>, ArrowSchurGpuFailure> {
    solve_reduced_beta_pcg_with_diagnostics(s_acc, rhs_beta, max_iterations, relative_tolerance)
        .map(|(x, _)| x)
}

#[doc(hidden)]
pub fn solve_reduced_beta_pcg_with_diagnostics(
    s_acc: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    max_iterations: usize,
    relative_tolerance: f64,
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
    let k = rhs_beta.len();
    if s_acc.dim() != (k, k) {
        return Err(ArrowSchurGpuFailure::SchurFactorFailed {
            reason: format!(
                "reduced-β GPU PCG requires a square (k×k) Schur block; got {:?} for k={k}",
                s_acc.dim()
            ),
        });
    }
    if k == 0 {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }

    #[cfg(not(target_os = "linux"))]
    {
        if relative_tolerance.is_nan() || max_iterations == 0 {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: "reduced-β GPU PCG: invalid CG controls".to_string(),
            });
        }
        Err(ArrowSchurGpuFailure::Unavailable)
    }

    #[cfg(target_os = "linux")]
    {
        cuda::solve_reduced_beta_pcg_with_diagnostics(
            s_acc,
            rhs_beta,
            max_iterations,
            relative_tolerance,
        )
    }
}

pub fn solve_sae_matrix_free_pcg(
    sys: &ArrowSchurSystem,
    data: &DeviceSaePcgData,
    ridge_t: f64,
    ridge_beta: f64,
    rhs_beta: &Array1<f64>,
    max_iterations: usize,
    relative_tolerance: f64,
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
    if sys.k != data.beta_dim || rhs_beta.len() != data.beta_dim || data.p == 0 {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    #[cfg(not(target_os = "linux"))]
    {
        if ridge_t.is_nan()
            || ridge_beta.is_nan()
            || relative_tolerance.is_nan()
            || max_iterations == 0
        {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: "SAE matrix-free GPU PCG: invalid controls".to_string(),
            });
        }
        Err(ArrowSchurGpuFailure::Unavailable)
    }
    #[cfg(target_os = "linux")]
    {
        // #1017/#1026 dispatch GUARD: framed data (frame metadata present) carries
        // a factored β border `G ⊗ W_{ij}` data Hessian and dense per-row cross
        // blocks the legacy `⊗ I_p` kernel CANNOT represent — feeding it framed
        // data would silently return a WRONG Newton step (it returns Ok with no
        // fallback). Route framed systems to the dedicated framed kernel and
        // legacy full-`B` systems to the legacy kernel; the two never cross.
        if data.frame.is_some() {
            cuda::solve_sae_matrix_free_pcg_framed(
                sys,
                data,
                ridge_t,
                ridge_beta,
                rhs_beta,
                max_iterations,
                relative_tolerance,
            )
        } else {
            cuda::solve_sae_matrix_free_pcg(
                sys,
                data,
                ridge_t,
                ridge_beta,
                rhs_beta,
                max_iterations,
                relative_tolerance,
            )
        }
    }
}

/// #1017 device-resident SAE frame across the LM ridge ladder.
///
/// A single inner Newton step drives the proximal ridge ladder (up to
/// [`crate::arrow_schur::DEFAULT_PROXIMAL_MAX_ATTEMPTS`] trials) at a FIXED
/// system: only `ridge_t`/`ridge_beta` change per trial. In the per-trial
/// [`solve_sae_matrix_free_pcg`] path, `flatten_device_sae_frame_data` re-marshals
/// AND re-uploads every device operand each trial — yet the ONLY ridge-dependent
/// buffer is the per-row factored inverse `ainv = (H_tt + ridge_t·I)⁻¹` (the
/// smooth `λ S_k`, the framed `G ⊗ W`, and the dense per-row cross `H_tβ` are all
/// ridge-independent and constant across the ladder). This handle uploads the
/// ridge-independent buffers ONCE (at [`build_sae_resident_frame`]) and, per trial,
/// recomputes only `ainv` before running the identical framed PCG loop — so the
/// numbers are bit-identical to the per-trial re-flatten path while the
/// `(trials − 1) × (ridge-independent operand bytes)` re-upload is eliminated.
///
/// It is a trait object (mirroring [`crate::arrow_schur::GpuSchurMatvec`]) so the
/// concrete CUDA implementation — which owns `mod cuda`-only device buffers — can
/// be carried through the cfg-independent [`crate::arrow_schur::ArrowSolveOptions`]
/// without leaking a CUDA-only type into the shared solve options.
pub trait SaeResidentFrame {
    /// Refresh every ridge-independent numerical operand from a newly
    /// assembled nonlinear iterate while retaining the device allocations.
    /// Implementations must return `Unavailable` when any shape changes; the
    /// caller then builds a new frame. No factor or old numerical value may be
    /// reused across accepted iterates.
    fn refresh(&self, sys: &ArrowSchurSystem) -> Result<(), ArrowSchurGpuFailure>;

    /// Recompute only the ridge-dependent per-row `ainv` at this trial's ridge,
    /// then run the framed reduced-Schur PCG against the resident buffers.
    /// Returns the reduced-β step `Δβ` and the PCG diagnostics, exactly as
    /// [`solve_sae_matrix_free_pcg`] would on the framed path. `Unavailable`
    /// signals a resident-path decline the caller should retry via the per-trial
    /// flatten; `RidgeBumpRequired`/`SchurFactorFailed` are genuine numerical
    /// signals the LM escalation must respond to (propagated unchanged).
    fn resolve(
        &self,
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
        rhs_beta: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure>;
}

/// Build the device-resident SAE frame for the LM ridge ladder.
/// `Err(Unavailable)` is the decline signal — non-CUDA host, no framed device
/// data, or the offload predicate rejects the shape — exactly the contract of
/// the sibling device entry points ([`gpu_schur_matvec_backend`]): the caller
/// keeps the established per-trial re-flatten path completely unchanged.
/// `cg_iters` is the CG budget the offload gate scores (same value the
/// per-trial framed solve uses).
pub fn build_sae_resident_frame(
    sys: &ArrowSchurSystem,
    cg_iters: usize,
) -> Result<std::sync::Arc<dyn SaeResidentFrame + Send + Sync>, ArrowSchurGpuFailure> {
    // Target-independent admission: a zero-K system has no reduced-Schur block
    // to keep resident, and a zero CG budget can never consume the frame — both
    // decline on every host, keeping the per-trial flatten the single fallback
    // (on CUDA hosts this also spares a doomed device build attempt).
    if sys.k == 0 || cg_iters == 0 {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    #[cfg(target_os = "linux")]
    {
        cuda::ResidentSaeFrameHandle::build(sys, cg_iters)
            .map(|h| std::sync::Arc::new(h) as std::sync::Arc<dyn SaeResidentFrame + Send + Sync>)
            .ok_or(ArrowSchurGpuFailure::Unavailable)
    }
    // Non-CUDA host: there is no device to build a frame on.
    #[cfg(not(target_os = "linux"))]
    {
        Err(ArrowSchurGpuFailure::Unavailable)
    }
}

/// The ridge-INDEPENDENT host operands of the framed SAE reduced-Schur system,
/// marshalled into the contiguous upload layout `flatten_device_sae_frame_data`
/// consumes. Split out (with [`compute_ainv_host`], the sole ridge-DEPENDENT
/// buffer) so a single source builds both the per-trial flatten and the resident
/// frame, and so the host-marshalling cost is measurable off-device. Every field
/// here is a pure function of `(sys, data, frame)` — invariant across the ridge
/// ladder — which is exactly why the resident frame can upload them once.
#[cfg(target_os = "linux")]
pub struct FrameHostOperands {
    pub s_off: Vec<i32>,
    pub s_m: Vec<i32>,
    pub s_r: Vec<i32>,
    pub s_ptr: Vec<i32>,
    pub s_data: Vec<f64>,
    pub s_blocks: usize,
    pub g_off_i: Vec<i32>,
    pub g_off_j: Vec<i32>,
    pub g_ri: Vec<i32>,
    pub g_rj: Vec<i32>,
    pub g_mi: Vec<i32>,
    pub g_mj: Vec<i32>,
    pub g_ptr: Vec<i32>,
    pub g_data: Vec<f64>,
    pub w_ptr: Vec<i32>,
    pub w_data: Vec<f64>,
    pub g_blocks: usize,
    pub g_max_work: usize,
    pub htb_ptr: Vec<i32>,
    pub htb: Vec<f64>,
    pub q_of: Vec<i32>,
    pub n_rows: usize,
    pub k: usize,
    pub max_q: usize,
}

#[cfg(target_os = "linux")]
fn frame_checked_i32(value: usize) -> Result<i32, ArrowSchurGpuFailure> {
    i32::try_from(value).map_err(|_| ArrowSchurGpuFailure::Unavailable)
}

/// Marshal the ridge-INDEPENDENT framed operands into contiguous host buffers.
/// Bit-for-bit the same layout `flatten_device_sae_frame_data` produced inline;
/// factored out so the per-trial flatten and the resident frame share one source
/// and so the marshalling is measurable without a device.
#[cfg(target_os = "linux")]
pub fn flatten_frame_host_operands(
    sys: &ArrowSchurSystem,
    data: &DeviceSaePcgData,
    frame: &crate::arrow_schur::DeviceSaeFrameData,
) -> Result<FrameHostOperands, ArrowSchurGpuFailure> {
    let n_rows = sys.rows.len();
    let k = data.beta_dim;
    if frame.row_htbeta.len() != n_rows
        || frame.ranks.len() != frame.basis_sizes.len()
        || frame.border_offsets.len() != frame.ranks.len()
        || data.smooth_blocks.len() != frame.smooth_ranks.len()
    {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }

    // Smooth blocks.
    let mut s_off = Vec::new();
    let mut s_m = Vec::new();
    let mut s_r = Vec::new();
    let mut s_ptr = vec![0_i32];
    let mut s_data = Vec::<f64>::new();
    for (blk, &r) in data.smooth_blocks.iter().zip(frame.smooth_ranks.iter()) {
        let (m, mc) = blk.factor_a.dim();
        if m != mc {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        s_off.push(frame_checked_i32(blk.global_offset)?);
        s_m.push(frame_checked_i32(m)?);
        s_r.push(frame_checked_i32(r)?);
        for ri in 0..m {
            for ci in 0..m {
                s_data.push(blk.factor_a[[ri, ci]]);
            }
        }
        s_ptr.push(frame_checked_i32(s_data.len())?);
    }

    // Data blocks (g + w).
    let mut g_off_i = Vec::new();
    let mut g_off_j = Vec::new();
    let mut g_ri = Vec::new();
    let mut g_rj = Vec::new();
    let mut g_mi = Vec::new();
    let mut g_mj = Vec::new();
    let mut g_ptr = vec![0_i32];
    let mut g_data = Vec::<f64>::new();
    let mut w_ptr = vec![0_i32];
    let mut w_data = Vec::<f64>::new();
    let mut g_max_work = 0usize;
    for blk in &frame.frame_blocks {
        let ri = frame.ranks[blk.atom_i];
        let rj = frame.ranks[blk.atom_j];
        let (mi, mj) = blk.g.dim();
        if blk.w.dim() != (ri, rj) {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        g_off_i.push(frame_checked_i32(frame.border_offsets[blk.atom_i])?);
        g_off_j.push(frame_checked_i32(frame.border_offsets[blk.atom_j])?);
        g_ri.push(frame_checked_i32(ri)?);
        g_rj.push(frame_checked_i32(rj)?);
        g_mi.push(frame_checked_i32(mi)?);
        g_mj.push(frame_checked_i32(mj)?);
        for r in 0..mi {
            for c in 0..mj {
                g_data.push(blk.g[[r, c]]);
            }
        }
        g_ptr.push(frame_checked_i32(g_data.len())?);
        for a in 0..ri {
            for b in 0..rj {
                w_data.push(blk.w[[a, b]]);
            }
        }
        w_ptr.push(frame_checked_i32(w_data.len())?);
        g_max_work = g_max_work.max(mi * ri);
    }

    // Per-row dense cross-block + q (the factored ainv is ridge-dependent and
    // lives in `compute_ainv_host`, not here).
    let mut htb_ptr = vec![0_i32];
    let mut htb = Vec::<f64>::new();
    let mut q_of = Vec::<i32>::with_capacity(n_rows);
    let mut max_q = 0usize;
    for (i, slab) in frame.row_htbeta.iter().enumerate() {
        let qi = sys.row_dims[i];
        let q_eff = if !slab.is_empty() && slab.len() == qi * k {
            qi
        } else {
            0
        };
        q_of.push(frame_checked_i32(q_eff)?);
        max_q = max_q.max(q_eff);
        if q_eff > 0 {
            htb.extend_from_slice(slab);
        }
        htb_ptr.push(frame_checked_i32(htb.len())?);
    }
    if max_q == 0 {
        // No row contributes a reduced term — pure-penalty system. Give max_q=1
        // so the ainv buffer is non-empty.
        max_q = 1;
    }

    Ok(FrameHostOperands {
        s_off,
        s_m,
        s_r,
        s_ptr,
        s_data,
        s_blocks: data.smooth_blocks.len(),
        g_off_i,
        g_off_j,
        g_ri,
        g_rj,
        g_mi,
        g_mj,
        g_ptr,
        g_data,
        w_ptr,
        w_data,
        g_blocks: frame.frame_blocks.len(),
        g_max_work,
        htb_ptr,
        htb,
        q_of,
        n_rows,
        k,
        max_q,
    })
}

/// Recompute the ridge-DEPENDENT per-row factored inverse `ainv[i] = (H_tt^(i) +
/// ridge_t·I)⁻¹` as a row-major `n_rows × max_q × max_q` host buffer — the ONLY
/// buffer that changes across the ridge ladder. Bit-for-bit the computation
/// `flatten_device_sae_frame_data` did inline (per-row Cholesky with the
/// nonnegative-pivot guard, dense inverse via unit-column back-substitution, and
/// the Gershgorin `RidgeBumpRequired` deficit on a non-PD block).
#[cfg(target_os = "linux")]
pub fn compute_ainv_host(
    sys: &ArrowSchurSystem,
    q_of: &[i32],
    max_q: usize,
    n_rows: usize,
    ridge_t: f64,
) -> Result<Vec<f64>, ArrowSchurGpuFailure> {
    let mut ainv = vec![0.0_f64; n_rows * max_q * max_q];
    for (i, row) in sys.rows.iter().enumerate() {
        let q = q_of[i] as usize;
        if q == 0 {
            continue;
        }
        if row.htt.dim() != (q, q) {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!(
                    "framed SAE device PCG row {i}: H_tt shape {:?} != ({q}, {q})",
                    row.htt.dim()
                ),
            });
        }
        let mut block = row.htt.clone();
        for d in 0..q {
            block[[d, d]] += ridge_t;
        }
        let factor = cholesky_factor_in_place(block.view(), CholeskyGuard::NonnegativePivot)
            .ok_or_else(|| ArrowSchurGpuFailure::RidgeBumpRequired {
                row: i,
                bump: ridge_bump_to_make_pd(row.htt.view(), ridge_t),
            })?;
        for col in 0..q {
            let mut e = Array1::<f64>::zeros(q);
            e[col] = 1.0;
            let solved = cholesky_solve_vector(factor.view(), e.view());
            for r in 0..q {
                ainv[i * max_q * max_q + r * max_q + col] = solved[r];
            }
        }
    }
    Ok(ainv)
}

/// #1551 kernel-isolating parity probe: run the framed reduced-Schur matvec
/// `out = S·x` exactly once on the device and return it (no PCG, no offload-floor
/// gate). The test suite diffs this element-wise against the CPU oracle
/// [`sae_framed_schur_matvec_cpu`] to prove the GPU kernel computes the SAME
/// operator — a check that is independent of solver conditioning (unlike a
/// solved-`δβ` comparison, which can diverge purely because dense Cholesky and
/// iterative PCG resolve an ill-conditioned `S` to different accuracies).
#[doc(hidden)]
#[cfg(target_os = "linux")]
pub fn framed_schur_matvec_once_on_device(
    sys: &ArrowSchurSystem,
    data: &DeviceSaePcgData,
    ridge_t: f64,
    ridge_beta: f64,
    x: &Array1<f64>,
) -> Result<Array1<f64>, ArrowSchurGpuFailure> {
    if sys.k != data.beta_dim || x.len() != data.beta_dim || data.p == 0 {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    if data.frame.is_none() {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    cuda::framed_schur_matvec_once_on_device(sys, data, ridge_t, ridge_beta, x)
}

/// #1017 evidence-lane probe: the DETERMINISTIC framed reduced-Schur matvec
/// `out = S·x` (host penalty + atomics-free device reduced-Schur term), computed
/// once. The test harness diffs it against [`sae_framed_schur_matvec_cpu`] AND
/// runs it twice to prove run-to-run bit stability — the two gates that let this
/// operator feed the SLQ `log|S|` evidence lane without breaking its determinism
/// contract.
#[doc(hidden)]
#[cfg(target_os = "linux")]
pub fn framed_reduced_schur_det_once_on_device(
    sys: &ArrowSchurSystem,
    data: &DeviceSaePcgData,
    ridge_t: f64,
    ridge_beta: f64,
    x: &Array1<f64>,
) -> Result<Array1<f64>, ArrowSchurGpuFailure> {
    if sys.k != data.beta_dim || x.len() != data.beta_dim || data.p == 0 {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    if data.frame.is_none() {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    cuda::framed_reduced_schur_det_once_on_device(sys, data, ridge_t, ridge_beta, x)
}

/// Reference dense back-end used by tests and as the fallback when the
/// GPU declines. Kept here (not in `arrow_schur_gpu.rs`) so the validation
/// suite has one canonical baseline.
#[doc(hidden)]
pub fn solve_arrow_newton_step_dense_reference(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<ArrowSchurGpuSolution, String> {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    let total = n.checked_mul(d).ok_or("dimension overflow")? + k;
    let mut h = Array2::<f64>::zeros((total, total));
    let mut rhs = Array1::<f64>::zeros(total);
    for (i, row) in sys.rows.iter().enumerate() {
        let base = i * d;
        for c in 0..d {
            for r in 0..d {
                h[[base + r, base + c]] = row.htt[[r, c]];
            }
            h[[base + c, base + c]] += ridge_t;
        }
        for c in 0..k {
            for r in 0..d {
                let value = row.htbeta[[r, c]];
                h[[base + r, n * d + c]] = value;
                h[[n * d + c, base + r]] = value;
            }
        }
        for r in 0..d {
            rhs[base + r] = -row.gt[r];
        }
    }
    for c in 0..k {
        for r in 0..k {
            h[[n * d + r, n * d + c]] += sys.hbb[[r, c]];
        }
        h[[n * d + c, n * d + c]] += ridge_beta;
        rhs[n * d + c] = -sys.gb[c];
    }
    // #2015 — this GPU/device-reference dense factorization is INDEPENDENT of
    // the CPU dense reduced-Schur path's Jacobi/Van der Sluis diagonal
    // equilibration fix (`gam_solve::arrow_schur::reduced_solve::factor_dense_reduced_schur`).
    // Both paths are exact (a correctly-formed Cholesky of the true joint/Schur
    // matrix), so no correctness gap exists between them, but this path does
    // not yet get the CPU path's improved conditioning on an ill-scaled `H`;
    // porting the same equilibrate-then-reconstruct technique here is a
    // deliberate follow-up, not done in this change.
    let factor = cholesky_factor_in_place(h.view(), CholeskyGuard::NonnegativePivot)
        .ok_or_else(|| "dense reference Cholesky failed".to_string())?;
    let mut log_det = 0.0_f64;
    for i in 0..total {
        log_det += factor[[i, i]].ln();
    }
    log_det *= 2.0;
    let solved = cholesky_solve_vector(factor.view(), rhs.view());
    let delta_t = solved.slice(ndarray::s![..n * d]).to_owned();
    let delta_beta = solved.slice(ndarray::s![n * d..]).to_owned();
    Ok(ArrowSchurGpuSolution {
        delta_t,
        delta_beta,
        log_det_hessian: log_det,
    })
}

/// Frames-engaged reduced-Schur penalty-side matvec `out = (P_ββ + ρ_β I)·x`,
/// computed purely from the factored device data (issue #1017/#1026). This is
/// the CPU bit-parity ORACLE for the GPU `arrow_sae_*` penalty kernels on the
/// frames path: smooth `λ S_k ⊗ I_{r_k}` (each `smooth_blocks[i]` at its
/// `global_offset` with right-width `frame.smooth_ranks[i]`) plus data-fit
/// `G_{ij} ⊗ W_{ij}` (each `frame.frame_blocks` entry, with the `μ`-major /
/// frame-minor index `border_offset[atom] + basis·r + frame_coord`). The
/// accumulation order matches the device kernels exactly.
///
/// `out` is OVERWRITTEN: first set to `ρ_β·x`, then the penalty blocks add in.
#[doc(hidden)]
pub fn sae_framed_penalty_matvec_cpu(
    data: &DeviceSaePcgData,
    ridge_beta: f64,
    x: &[f64],
    out: &mut [f64],
) {
    let frame = data
        .frame
        .as_ref()
        .expect("sae_framed_penalty_matvec_cpu requires frame metadata");
    let k = data.beta_dim;
    for a in 0..k {
        out[a] = ridge_beta * x[a];
    }
    // Smooth penalty `λ S_k ⊗ I_{r_k}`: y[off + ia·r + ib] += Σ_ja S[ia,ja]·x[off + ja·r + ib].
    for (blk, &r) in data.smooth_blocks.iter().zip(frame.smooth_ranks.iter()) {
        let off = blk.global_offset;
        let m = blk.factor_a.nrows();
        for i_a in 0..m {
            for i_b in 0..r {
                let mut acc = 0.0_f64;
                for j_a in 0..m {
                    let s = blk.factor_a[[i_a, j_a]];
                    if s == 0.0 {
                        continue;
                    }
                    acc += s * x[off + j_a * r + i_b];
                }
                out[off + i_a * r + i_b] += acc;
            }
        }
    }
    // Data-fit penalty `G_{ij} ⊗ W_{ij}`.
    for blk in &frame.frame_blocks {
        let r_i = frame.ranks[blk.atom_i];
        let r_j = frame.ranks[blk.atom_j];
        let off_i = frame.border_offsets[blk.atom_i];
        let off_j = frame.border_offsets[blk.atom_j];
        let (m_i, m_j) = blk.g.dim();
        for li in 0..m_i {
            let yi_base = off_i + li * r_i;
            for lj in 0..m_j {
                let g = blk.g[[li, lj]];
                if g == 0.0 {
                    continue;
                }
                let xj_base = off_j + lj * r_j;
                for a in 0..r_i {
                    let mut acc = 0.0_f64;
                    for b in 0..r_j {
                        acc += blk.w[[a, b]] * x[xj_base + b];
                    }
                    out[yi_base + a] += g * acc;
                }
            }
        }
    }
}

/// Frames-engaged FULL reduced-Schur matvec `out = S·x` purely from the device
/// data, where `S = (P_ββ + ρ_β I) − Σ_i H_βt^(i)(H_tt^(i)+ρ_t I)⁻¹ H_tβ^(i)`
/// (issue #1017/#1026). The penalty side is [`sae_framed_penalty_matvec_cpu`];
/// the per-row reduced term reads the dense `frame.row_htbeta[i]`
/// (`q_i × border_dim`, row-major), solves against the row's
/// `H_tt^(i)+ρ_t I` Cholesky factor, and scatters the transpose back. This is
/// the size-independent bit-parity oracle the device kernel mirrors; it is also
/// the matvec the GPU PCG iterates.
#[doc(hidden)]
pub fn sae_framed_schur_matvec_cpu(
    sys: &ArrowSchurSystem,
    data: &DeviceSaePcgData,
    ridge_t: f64,
    ridge_beta: f64,
    x: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    let frame = data
        .frame
        .as_ref()
        .ok_or("sae_framed_schur_matvec_cpu requires frame metadata")?;
    let k = data.beta_dim;
    sae_framed_penalty_matvec_cpu(data, ridge_beta, x, out);
    if frame.row_htbeta.len() != sys.rows.len() {
        return Err(format!(
            "sae_framed_schur_matvec_cpu: {} row_htbeta slabs but {} rows",
            frame.row_htbeta.len(),
            sys.rows.len()
        ));
    }
    for (i, row) in sys.rows.iter().enumerate() {
        let slab = &frame.row_htbeta[i];
        if slab.is_empty() {
            continue;
        }
        let qi = sys.row_dims[i];
        if qi == 0 || slab.len() != qi * k {
            continue;
        }
        // h = H_tβ^(i) · x  (length q_i).
        let mut h = vec![0.0_f64; qi];
        for c in 0..qi {
            let base = c * k;
            let mut acc = 0.0_f64;
            for a in 0..k {
                acc += slab[base + a] * x[a];
            }
            h[c] = acc;
        }
        // solve (H_tt^(i)+ρ_t I) s = h.
        let mut block = row.htt.clone();
        for d in 0..qi {
            block[[d, d]] += ridge_t;
        }
        let factor = cholesky_factor_in_place(block.view(), CholeskyGuard::NonnegativePivot)
            .ok_or_else(|| format!("sae_framed_schur_matvec_cpu: row {i} H_tt not PD"))?;
        let s = cholesky_solve_vector(factor.view(), Array1::from_vec(h).view());
        // out -= H_βt^(i) · s = (H_tβ^(i))ᵀ · s.
        for c in 0..qi {
            let sc = s[c];
            if sc == 0.0 {
                continue;
            }
            let base = c * k;
            for a in 0..k {
                out[a] -= slab[base + a] * sc;
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod cuda {
    use super::{ArrowSchurGpuFailure, ArrowSchurGpuSolution, pack_block, pack_host};
    use crate::arrow_schur::{
        ArrowPcgDiagnostics, ArrowSchurSystem, DeviceSaeFrameData, DeviceSaePcgData, PcgStopReason,
    };
    use cudarc::cublas::sys::{
        cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{
        CudaContext, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
        PushKernelArg,
    };
    use gam_gpu::driver::to_i32;
    use gam_gpu::linalg_dispatch::{DispatchOp, route_through_gpu};
    use ndarray::Array1;
    use std::sync::{Arc, OnceLock};

    /// Per-row work slot for the row-block-granular multi-GPU solve. Inputs are
    /// the packed single-row buffers (`d×d` D block + ρ_t ridge, `d×k` B block,
    /// `d` g vector); the forward pass fills the whitened factors `l/u/y` and the
    /// per-tile reduction lands in the tile's leading slot.
    struct RowSlot {
        // Inputs (packed once on the host, column-major).
        d_block: Vec<f64>, // d*d
        b_block: Vec<f64>, // d*k
        g_vec: Vec<f64>,   // d
        // Forward outputs, kept on the host for the back-sub pass.
        l_block: Vec<f64>, // d*d lower factor, column-major
        u_vec: Vec<f64>,   // d   (= L^{-1} g)
        y_block: Vec<f64>, // d*k (= L^{-1} B), column-major
        log_det_local: f64,
        // Set on a non-PD pivot so the orchestrator can raise RidgeBumpRequired
        // for the offending global row instead of silently falling back.
        bump: Option<f64>,
        // Tile-level reduction, written into the tile's first slot only.
        tile_partial_schur: Option<Vec<f64>>, // k*k col-major, = Σ Y_iᵀY_i
        tile_partial_rhs: Option<Vec<f64>>,   // k, = Σ Y_iᵀu_i
        // Back-sub output for this row.
        delta_t_block: Vec<f64>, // d
    }

    /// Row-block-granular multi-GPU Arrow-Schur Newton solve.
    ///
    /// The solve is separable across row blocks in both phases:
    ///   * forward — each row's local Cholesky `L_i`, whitening
    ///     `u_i = L_i⁻¹g_i`, `Y_i = L_i⁻¹B_i`, and partial Schur
    ///     `(Σ Y_iᵀY_i, Σ Y_iᵀu_i)` are independent;
    ///   * backward — `δt_i = -L_iᵀ⁻¹(u_i + Y_iδβ)` is independent.
    /// Only the small shared `K×K` reduce + factor + `δβ` solve is central.
    ///
    /// `gam_gpu::pool::scatter_batched` hands each device a contiguous row
    /// tile on its own bound context/stream; the per-tile forward keeps the
    /// POTRF fused with its dependent TRSM + Schur GEMM on that one stream, so no
    /// on-stream solve is orphaned. Tile partials and per-tile `log|L|` are
    /// reduced on the host (in tile/row order), `S_β` is factored on the primary
    /// device, and the back-sub is scattered back across the same tiles.
    ///
    /// Returns `Unavailable` (caller uses a single-device path) when the system
    /// carries matrix-free operators, the shared block is not dense `K×K`, the
    /// pool is single-device, or any tile's device work declines. A non-PD tip
    /// block surfaces as `RidgeBumpRequired` for the precise global row.
    pub(super) fn solve_multi_gpu(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        if n == 0 || d == 0 || k == 0 {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        // Dense shared block + materialised per-row slabs are required; the
        // public entry already rejected matrix-free operators, but re-check so
        // this routine is safe in isolation.
        if sys.hbb_matvec.is_some() || sys.htbeta_matvec.is_some() || sys.hbb.dim() != (k, k) {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }

        let runtime = super::resolve_runtime_for_device_path()?
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        if runtime.device_count() < 2 {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }

        // Pack one slot per row (column-major), folding ρ_t into each D block.
        let mut slots: Vec<RowSlot> = Vec::with_capacity(n);
        for row in &sys.rows {
            if row.htt.dim() != (d, d) || row.htbeta.dim() != (d, k) || row.gt.len() != d {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            let mut d_block = Vec::with_capacity(d * d);
            let mut b_block = Vec::with_capacity(d * k);
            let mut g_vec = Vec::with_capacity(d);
            pack_block(row, ridge_t, d, k, &mut d_block, &mut b_block, &mut g_vec);
            slots.push(RowSlot {
                d_block,
                b_block,
                g_vec,
                l_block: Vec::new(),
                u_vec: Vec::new(),
                y_block: Vec::new(),
                log_det_local: 0.0,
                bump: None,
                tile_partial_schur: None,
                tile_partial_rhs: None,
                delta_t_block: vec![0.0; d],
            });
        }

        // ---- Forward pass: per-device row tile, fused on its own stream ----
        let forward_ok = gam_gpu::pool::scatter_batched(runtime, &mut slots, |ordinal, tile| {
            forward_tile(ordinal, d, k, tile)
        });
        if forward_ok.is_none() {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }

        // Surface a non-PD tip block as a precise per-row ridge bump.
        let row_base_of_tile = gam_gpu::pool::balanced_partition(runtime, n);
        if let Some((row, bump)) = slots
            .iter()
            .enumerate()
            .find_map(|(i, slot)| slot.bump.map(|b| (i, b)))
        {
            return Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump });
        }

        // ---- Central: reduce tile partials → S_β, r_β; factor; solve δβ ----
        // Seed S_β with H_ββ + ρ_β I (column-major) and r_β with -g_β, then fold
        // in the per-tile partials in tile order so the reduction order tracks
        // the single-device accumulation (up to inter-tile reassociation).
        let mut schur_host = vec![0.0_f64; k * k];
        for col in 0..k {
            for row in 0..k {
                let mut v = sys.hbb[[row, col]];
                if row == col {
                    v += ridge_beta;
                }
                schur_host[col * k + row] = v;
            }
        }
        let mut rhs_host: Vec<f64> = sys.gb.iter().map(|v| -v).collect();
        let mut log_det = 0.0_f64;
        for start in tile_starts(&row_base_of_tile) {
            let slot = &slots[start];
            let partial_schur = slot
                .tile_partial_schur
                .as_ref()
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let partial_rhs = slot
                .tile_partial_rhs
                .as_ref()
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            // `accumulate_schur` writes `partial_schur = -Σ_tile Y_iᵀY_i` (GEMM
            // α=-1, β=1 into a zero seed) and `partial_rhs = +Σ_tile Y_iᵀu_i`.
            // The reduced Schur is `S = (H_ββ+ρI) − Σ_all Y_iᵀY_i`, so adding the
            // (already-negated) partials reproduces the single-device sign.
            for idx in 0..k * k {
                schur_host[idx] += partial_schur[idx];
            }
            for a in 0..k {
                rhs_host[a] += partial_rhs[a];
            }
        }
        for slot in &slots {
            log_det += slot.log_det_local;
        }

        // Factor S_β and solve δβ on the primary device (small K×K leaf). The
        // stream carries the primary context (same pattern as `solve()`); no
        // thread bind is needed for the cuSOLVER/cuBLAS handles created from it.
        let primary = runtime.selected_device().ordinal;
        let stream = gam_gpu::device_runtime::cuda_context_for(primary)
            .and_then(|ctx| ctx.new_stream().ok())
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let solver =
            DnHandle::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut schur_dev = stream
            .clone_htod(&schur_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut rhs_dev = stream
            .clone_htod(&rhs_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let info = potrf_single(&solver, &stream, k, &mut schur_dev)?;
        if info != 0 {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("multi-GPU Schur Cholesky failed at pivot {info}"),
            });
        }
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, false)?;
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, true)?;
        let delta_beta_host = stream
            .clone_dtoh(&rhs_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let delta_beta = Array1::from_vec(delta_beta_host.clone());
        let l_schur_host = stream
            .clone_dtoh(&schur_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        for j in 0..k {
            log_det += l_schur_host[j * k + j].ln();
        }
        log_det *= 2.0;

        // ---- Backward pass: δt_i = -L_iᵀ⁻¹(u_i + Y_iδβ), per-device tile ----
        let delta_beta_ref = &delta_beta_host;
        let back_ok = gam_gpu::pool::scatter_batched(runtime, &mut slots, |ordinal, tile| {
            back_sub_tile(ordinal, d, k, delta_beta_ref, tile)
        });
        if back_ok.is_none() {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }

        // Stitch per-row δt into the stacked (n*d) result.
        let mut delta_t = Array1::<f64>::zeros(n * d);
        for (i, slot) in slots.iter().enumerate() {
            let base = i * d;
            for r in 0..d {
                delta_t[base + r] = slot.delta_t_block[r];
            }
        }

        Ok(ArrowSchurGpuSolution {
            delta_t,
            delta_beta,
            log_det_hessian: log_det,
        })
    }

    /// Tile starts: the leading global row index of each device tile (where the
    /// tile-level partial reduction was written by the forward pass).
    fn tile_starts(tiles: &[(usize, std::ops::Range<usize>)]) -> impl Iterator<Item = usize> + '_ {
        tiles.iter().map(|(_, range)| range.start)
    }

    /// Forward pass for one device row tile, running on `ordinal`'s bound stream.
    /// Factors each row block, whitens `u`/`Y`, accumulates the tile's partial
    /// Schur `(Σ Y_iᵀY_i, Σ Y_iᵀu_i)` into the tile's leading slot, keeps the
    /// per-row `L`/`u`/`Y` on the host for back-sub, and records the per-row
    /// `Σ_j log L_jj`. A non-PD pivot is recorded in `slot.bump` (the tile still
    /// returns `Some(())` so the orchestrator raises a precise `RidgeBumpRequired`
    /// rather than collapsing the whole batch to CPU).
    fn forward_tile(ordinal: usize, d: usize, k: usize, tile: &mut [RowSlot]) -> Option<()> {
        if tile.is_empty() {
            return Some(());
        }
        // `scatter_batched` has already bound this ordinal's context on this
        // worker thread; the stream below targets that same device.
        let stream = gam_gpu::device_runtime::cuda_context_for(ordinal)
            .and_then(|ctx| ctx.new_stream().ok())?;
        let solver = DnHandle::new(stream.clone()).ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        let m = tile.len();

        // Stack the tile's D, B, g into contiguous device buffers (same layout
        // the single-device path packs for `m` rows).
        let mut d_host = Vec::with_capacity(m * d * d);
        let mut b_host = Vec::with_capacity(m * d * k);
        let mut g_host = Vec::with_capacity(m * d);
        for slot in tile.iter() {
            d_host.extend_from_slice(&slot.d_block);
            b_host.extend_from_slice(&slot.b_block);
            g_host.extend_from_slice(&slot.g_vec);
        }
        let mut d_dev = stream.clone_htod(&d_host).ok()?;
        let mut b_dev = stream.clone_htod(&b_host).ok()?;
        let mut g_dev = stream.clone_htod(&g_host).ok()?;

        // Batched POTRF; a non-PD block records its bump and stops the tile.
        // The bump is deficit-aware (Gershgorin lower bound on λ_min of the
        // already-ridged `d_block`), NOT derived from the cuSOLVER `info` —
        // which is a 1-based pivot ROW INDEX, not a pivot magnitude — so a
        // strongly-indefinite block recovers in one outer-loop retry.
        let info_host = potrf_batched(&solver, &stream, d, m, &mut d_dev).ok()?;
        if let Some(local) = info_host.iter().position(|info| *info != 0) {
            tile[local].bump = Some(super::ridge_bump_to_make_pd_colmajor(
                &tile[local].d_block,
                d,
            ));
            return Some(());
        }

        // Whiten: u = L⁻¹ g, Y = L⁻¹ B.
        trsm_batched_lower_inplace(&blas, &stream, d, m, 1, &d_dev, &mut g_dev).ok()?;
        trsm_batched_lower_inplace(&blas, &stream, d, m, k, &d_dev, &mut b_dev).ok()?;

        // Tile partial Schur: zero-seeded so the host adds the H_ββ seed once.
        let mut schur_dev = stream.alloc_zeros::<f64>(k * k).ok()?;
        let mut rhs_dev = stream.alloc_zeros::<f64>(k).ok()?;
        accumulate_schur(&blas, d, k, m, &b_dev, &g_dev, &mut schur_dev, &mut rhs_dev).ok()?;

        // Download L, u, Y, and the tile partials.
        let l_host = stream.clone_dtoh(&d_dev).ok()?;
        let u_host = stream.clone_dtoh(&g_dev).ok()?;
        let y_host = stream.clone_dtoh(&b_dev).ok()?;
        let partial_schur = stream.clone_dtoh(&schur_dev).ok()?;
        let partial_rhs = stream.clone_dtoh(&rhs_dev).ok()?;

        for (local, slot) in tile.iter_mut().enumerate() {
            let l_base = local * d * d;
            let u_base = local * d;
            let y_base = local * d * k;
            slot.l_block = l_host[l_base..l_base + d * d].to_vec();
            slot.u_vec = u_host[u_base..u_base + d].to_vec();
            slot.y_block = y_host[y_base..y_base + d * k].to_vec();
            let mut log_det_local = 0.0_f64;
            for j in 0..d {
                log_det_local += l_host[l_base + j * d + j].ln();
            }
            slot.log_det_local = log_det_local;
        }
        tile[0].tile_partial_schur = Some(partial_schur);
        tile[0].tile_partial_rhs = Some(partial_rhs);
        Some(())
    }

    /// Back-substitution for one device row tile: `δt_i = -L_iᵀ⁻¹(u_i + Y_iδβ)`.
    /// Re-uploads the tile's kept `L`/`u`/`Y` to `ordinal`, applies the GEMV
    /// accumulate + transposed TRSM, and writes each row's `δt` into its slot.
    fn back_sub_tile(
        ordinal: usize,
        d: usize,
        k: usize,
        delta_beta: &[f64],
        tile: &mut [RowSlot],
    ) -> Option<()> {
        if tile.is_empty() {
            return Some(());
        }
        // `scatter_batched` has already bound this ordinal's context on this
        // worker thread; the stream below targets that same device.
        let stream = gam_gpu::device_runtime::cuda_context_for(ordinal)
            .and_then(|ctx| ctx.new_stream().ok())?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        let m = tile.len();

        let mut l_host = Vec::with_capacity(m * d * d);
        let mut u_host = Vec::with_capacity(m * d);
        let mut y_host = Vec::with_capacity(m * d * k);
        for slot in tile.iter() {
            l_host.extend_from_slice(&slot.l_block);
            u_host.extend_from_slice(&slot.u_vec);
            y_host.extend_from_slice(&slot.y_block);
        }
        let d_dev = stream.clone_htod(&l_host).ok()?;
        let mut g_dev = stream.clone_htod(&u_host).ok()?;
        let b_dev = stream.clone_htod(&y_host).ok()?;
        let rhs_dev = stream.clone_htod(&delta_beta.to_vec()).ok()?;

        // g ← u + Y·δβ, then x = L⁻ᵀ g; δt = -x.
        accumulate_back_sub_rhs(&blas, d, k, m, &b_dev, &rhs_dev, &mut g_dev).ok()?;
        trsm_batched_lower_inplace_transposed(&blas, &stream, d, m, 1, &d_dev, &mut g_dev).ok()?;
        let x_host = stream.clone_dtoh(&g_dev).ok()?;
        for (local, slot) in tile.iter_mut().enumerate() {
            let base = local * d;
            for r in 0..d {
                slot.delta_t_block[r] = -x_host[base + r];
            }
        }
        Some(())
    }

    pub(super) fn solve(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let runtime = route_through_gpu(DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n })
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;

        let stream = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .and_then(|ctx| ctx.new_stream().ok())
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let solver =
            DnHandle::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // ----- Pack + upload D, B, g -----
        let (d_host, b_host, g_host) = pack_host(sys, ridge_t);
        let mut d_dev = stream
            .clone_htod(&d_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut b_dev = stream
            .clone_htod(&b_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut g_dev = stream
            .clone_htod(&g_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // ----- Layer A: batched lower Cholesky of D in place -----
        // This POTRF is fused with the downstream TRSM + Schur GEMM + back-sub
        // on this one stream, so splitting only the POTRF across devices would
        // orphan the dependent on-stream solves. Multi-GPU here is the
        // whole-solve row-block split in `solve_arrow_newton_step` (see
        // `solve_multi_gpu`), not a per-layer split — this device-resident path
        // is the single-device leaf the split dispatches per tile.
        let info_host = potrf_batched(&solver, &stream, d, n, &mut d_dev)?;
        if let Some(idx) = info_host.iter().position(|info| *info != 0) {
            // `info` is cuSOLVER's 1-based pivot ROW INDEX, not a magnitude;
            // size the bump from the block's own entries (Gershgorin λ_min
            // bound) so a strongly-indefinite block recovers in one retry.
            return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                row: idx,
                bump: super::ridge_bump_to_make_pd(sys.rows[idx].htt.view(), ridge_t),
            });
        }

        // ----- Layer B (1/2): in-place triangular solves -----
        // u_i = L_i^{-1} g_i, packed as a stacked (n*d) column-vector.
        trsm_batched_lower_inplace(&blas, &stream, d, n, 1, &d_dev, &mut g_dev)?;
        // Y_i = L_i^{-1} B_i, in place over the (n*d) × k buffer (laid out as
        // n stacked column-major d×k tiles).
        trsm_batched_lower_inplace(&blas, &stream, d, n, k, &d_dev, &mut b_dev)?;

        // ----- Layer B (2/2): Schur reduction via single big GEMM / GEMV -----
        // Y_all is (n*d) × k column-major: viewing all n stacked d×k tiles as
        // one big matrix is bit-exact because each tile is column-major with
        // leading dim d and the tiles are contiguous in memory, so the
        // combined leading dim is n*d only for the *outer* matrix view. To
        // make the single-GEMM equivalence hold we must treat the stacked
        // buffer as (n*d) × k column-major with leading dim = n*d, which
        // means columns of Y_all are interleaved by row across blocks.
        // That is NOT what we packed. So we use the cuBLAS stride pattern
        // instead: stride-by-block, transpose-A, and *accumulate* into one
        // S_β buffer via beta=1 across batches. Equivalent flop count, no
        // extra reduction kernel, and correct layout.
        //
        // Concretely: schur ← C + ρ_β I; rhs ← -g_β; then for each block
        //   schur -= Y_i^T Y_i      (k×k)
        //   rhs   += Y_i^T u_i      (k)
        // We launch this as `n` sequential GEMMs/GEMVs with beta=1 on the
        // accumulator. Layer D fuses these into one NVRTC launch.
        let schur_init: Vec<f64> = {
            let mut tmp = Vec::with_capacity(k * k);
            for col in 0..k {
                for row in 0..k {
                    let mut v = sys.hbb[[row, col]];
                    if row == col {
                        v += ridge_beta;
                    }
                    tmp.push(v);
                }
            }
            tmp
        };
        let mut schur_dev = stream
            .clone_htod(&schur_init)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let rhs_init: Vec<f64> = sys.gb.iter().map(|v| -v).collect();
        let mut rhs_dev = stream
            .clone_htod(&rhs_init)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        accumulate_schur(&blas, d, k, n, &b_dev, &g_dev, &mut schur_dev, &mut rhs_dev)?;

        // ----- Layer C (1/2): factor S_β and solve for δβ -----
        let info = potrf_single(&solver, &stream, k, &mut schur_dev)?;
        if info != 0 {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("Schur Cholesky failed at pivot {info}"),
            });
        }
        // δβ ← L_S^{-T} L_S^{-1} rhs
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, false)?;
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, true)?;
        let delta_beta_host = stream
            .clone_dtoh(&rhs_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let delta_beta = Array1::from_vec(delta_beta_host.clone());

        // ----- Layer C (2/2): back-sub δt_i = -L_i^{-T} (u_i + Y_i δβ) -----
        // Already on device:
        //   g_dev holds u_i stacked (n*d).
        //   b_dev holds Y_i stacked column-major n×(d×k) tiles.
        // Compute g_dev ← g_dev + Y_block · δβ per block (cuBLAS gemv with beta=1),
        // then in-place trsm with L_i^T (CUBLAS_OP_T) to obtain x_i, and finally
        // δt_i = -x_i on host after download.
        accumulate_back_sub_rhs(&blas, d, k, n, &b_dev, &rhs_dev, &mut g_dev)?;
        trsm_batched_lower_inplace_transposed(&blas, &stream, d, n, 1, &d_dev, &mut g_dev)?;

        let x_host = stream
            .clone_dtoh(&g_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut delta_t = Array1::<f64>::zeros(n * d);
        for (i, v) in x_host.iter().enumerate() {
            delta_t[i] = -*v;
        }

        // ----- log|H| = 2 Σ log L_{i,jj} + 2 Σ log R_{β,aa} -----
        let l_local_host = stream
            .clone_dtoh(&d_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let l_schur_host = stream
            .clone_dtoh(&schur_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut log_det = 0.0_f64;
        for i in 0..n {
            let base = i * d * d;
            for j in 0..d {
                log_det += l_local_host[base + j * d + j].ln();
            }
        }
        for j in 0..k {
            log_det += l_schur_host[j * k + j].ln();
        }
        log_det *= 2.0;

        Ok(ArrowSchurGpuSolution {
            delta_t,
            delta_beta,
            log_det_hessian: log_det,
        })
    }

    fn potrf_batched(
        solver: &DnHandle,
        stream: &Arc<CudaStream>,
        p: usize,
        batch: usize,
        matrices: &mut CudaSlice<f64>,
    ) -> Result<Vec<i32>, ArrowSchurGpuFailure> {
        let p_i = to_i32(p).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let batch_i = to_i32(batch).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let matrix_len = p * p;
        let bytes_per = (matrix_len * std::mem::size_of::<f64>()) as u64;
        let (base_ptr, _record) = matrices.device_ptr_mut(stream);
        let mut ptrs = Vec::with_capacity(batch);
        for idx in 0..batch {
            ptrs.push(base_ptr + (idx as u64) * bytes_per);
        }
        let mut ptrs_dev = stream
            .clone_htod(&ptrs)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut info_dev = stream
            .alloc_zeros::<i32>(batch)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let status = {
            let (ptrs_ptr, _ptrs_record) = ptrs_dev.device_ptr_mut(stream);
            let (info_ptr, _info_record) = info_dev.device_ptr_mut(stream);
            // SAFETY: pointer array and info buffer live on the device,
            // matrices_dev holds `batch` contiguous p×p column-major blocks.
            unsafe {
                cusolver_sys::cusolverDnDpotrfBatched(
                    solver.cu(),
                    cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                    p_i,
                    ptrs_ptr as *mut *mut f64,
                    p_i,
                    info_ptr as *mut i32,
                    batch_i,
                )
            }
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        stream
            .clone_dtoh(&info_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
    }

    fn potrf_single(
        solver: &DnHandle,
        stream: &Arc<CudaStream>,
        p: usize,
        matrix: &mut CudaSlice<f64>,
    ) -> Result<i32, ArrowSchurGpuFailure> {
        let p_i = to_i32(p).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut lwork = 0_i32;
        {
            let (mat_ptr, _rec) = matrix.device_ptr_mut(stream);
            // SAFETY: buffer query against a live p-by-p column-major device matrix.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf_bufferSize(
                    solver.cu(),
                    uplo,
                    p_i,
                    mat_ptr as *mut f64,
                    p_i,
                    &mut lwork,
                )
            };
            if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
        }
        let lwork_usize = usize::try_from(lwork).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut workspace = stream
            .alloc_zeros::<f64>(lwork_usize.max(1))
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut info_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        {
            let (mat_ptr, _rec) = matrix.device_ptr_mut(stream);
            let (work_ptr, _wrec) = workspace.device_ptr_mut(stream);
            let (info_ptr, _irec) = info_dev.device_ptr_mut(stream);
            // SAFETY: all three pointers refer to live, correctly sized device buffers.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf(
                    solver.cu(),
                    uplo,
                    p_i,
                    mat_ptr as *mut f64,
                    p_i,
                    work_ptr as *mut f64,
                    lwork,
                    info_ptr as *mut i32,
                )
            };
            if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
        }
        let info_host = stream
            .clone_dtoh(&info_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        Ok(info_host[0])
    }

    /// In-place lower-triangular solves `X_i ← L_i^{-1} X_i` over the n stacked
    /// d×nrhs RHS tiles in `rhs`. Uses `cublasDtrsmBatched` so all n solves
    /// hit the device in one launch.
    fn trsm_batched_lower_inplace(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        d: usize,
        n: usize,
        nrhs: usize,
        l_stack: &CudaSlice<f64>,
        rhs_stack: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        trsm_batched_inplace_inner(blas, stream, d, n, nrhs, l_stack, rhs_stack, false)
    }

    /// As above but with `L_i^T` instead of `L_i`.
    fn trsm_batched_lower_inplace_transposed(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        d: usize,
        n: usize,
        nrhs: usize,
        l_stack: &CudaSlice<f64>,
        rhs_stack: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        trsm_batched_inplace_inner(blas, stream, d, n, nrhs, l_stack, rhs_stack, true)
    }

    fn trsm_batched_inplace_inner(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        d: usize,
        n: usize,
        nrhs: usize,
        l_stack: &CudaSlice<f64>,
        rhs_stack: &mut CudaSlice<f64>,
        transposed: bool,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let alpha = 1.0_f64;
        let d_i = to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let nrhs_i = to_i32(nrhs).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let batch_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let l_bytes_per = (d * d * std::mem::size_of::<f64>()) as u64;
        let rhs_bytes_per = (d * nrhs * std::mem::size_of::<f64>()) as u64;
        let (l_base, _l_record) = l_stack.device_ptr(stream);
        let (rhs_base, _rhs_record) = rhs_stack.device_ptr_mut(stream);
        let mut l_ptrs = Vec::with_capacity(n);
        let mut rhs_ptrs = Vec::with_capacity(n);
        for i in 0..n {
            l_ptrs.push(l_base + (i as u64) * l_bytes_per);
            rhs_ptrs.push(rhs_base + (i as u64) * rhs_bytes_per);
        }
        let mut l_ptrs_dev = stream
            .clone_htod(&l_ptrs)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut rhs_ptrs_dev = stream
            .clone_htod(&rhs_ptrs)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let (l_ptrs_ptr, _l_ptrs_rec) = l_ptrs_dev.device_ptr_mut(stream);
        let (rhs_ptrs_ptr, _rhs_ptrs_rec) = rhs_ptrs_dev.device_ptr_mut(stream);
        let op = if transposed {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let handle = *blas.handle();
        // SAFETY: pointer arrays and base buffers were just constructed from
        // live device allocations covering the entire batch.
        let status = unsafe {
            cudarc::cublas::sys::cublasDtrsmBatched(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                op,
                cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                d_i,
                nrhs_i,
                &alpha,
                l_ptrs_ptr as *const *const f64,
                d_i,
                rhs_ptrs_ptr as *const *mut f64,
                d_i,
                batch_i,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        Ok(())
    }

    /// Single-matrix lower-triangular solve: `rhs ← L^{-1} rhs` (or
    /// `L^{-T} rhs` if `transposed`). For the Schur Cholesky back-sub.
    fn trsm_single(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        n: usize,
        l: &CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
        upper: bool,
        transposed: bool,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let alpha = 1.0_f64;
        let n_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let handle = *blas.handle();
        let (l_ptr, _l_rec) = l.device_ptr(stream);
        let (rhs_ptr, _rhs_rec) = rhs.device_ptr_mut(stream);
        // SAFETY: single n×n lower factor and n-vector RHS on device.
        let status = unsafe {
            cudarc::cublas::sys::cublasDtrsm_v2(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                if upper {
                    cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                } else {
                    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                },
                if transposed {
                    cublasOperation_t::CUBLAS_OP_T
                } else {
                    cublasOperation_t::CUBLAS_OP_N
                },
                cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                n_i,
                1,
                &alpha,
                l_ptr as *const f64,
                n_i,
                rhs_ptr as *mut f64,
                n_i,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        Ok(())
    }

    /// Accumulate `schur ← schur − Σ_i Y_i^T Y_i` and `rhs ← rhs + Σ_i Y_i^T u_i`
    /// using one GEMM and one GEMV per block. Each call uses beta=1 to chain
    /// the accumulation device-side.
    fn accumulate_schur(
        blas: &CudaBlas,
        d: usize,
        k: usize,
        n: usize,
        y_stack: &CudaSlice<f64>,
        u_stack: &CudaSlice<f64>,
        schur: &mut CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let y_block_elems = d * k;
        let u_block_elems = d;
        for i in 0..n {
            let y_slice = y_stack.slice(i * y_block_elems..(i + 1) * y_block_elems);
            let u_slice = u_stack.slice(i * u_block_elems..(i + 1) * u_block_elems);
            // GEMM: schur += (-1) · Y_i^T · Y_i  (Y_i is d×k col-major; out is k×k)
            let gemm_cfg = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                k: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: -1.0,
                lda: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                ldb: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                beta: 1.0,
                ldc: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
            };
            // SAFETY: y_slice is d×k col-major, schur is k×k col-major; alpha/beta scalars set above.
            unsafe { blas.gemm(gemm_cfg, &y_slice, &y_slice, schur) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            // GEMV: rhs += 1 · Y_i^T · u_i
            let gemv_cfg = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: 1.0,
                lda: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                incx: 1,
                beta: 1.0,
                incy: 1,
            };
            // SAFETY: y_slice (d×k col-major) and u_slice (length d) are live
            // device buffers; `rhs` is the length-k accumulator.
            unsafe { blas.gemv(gemv_cfg, &y_slice, &u_slice, rhs) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    /// `#1017` resident gradient path: accumulate ONLY the Schur RHS term
    /// `rhs += Σ_i Y_iᵀ u_i`, skipping the `−Σ_i Y_iᵀ Y_i` matrix GEMM that the
    /// resident frame already folded into its persistent `L_S` factor. This is
    /// the per-iterate-cheap counterpart of [`accumulate_schur`]: the GEMV here
    /// is bit-identical to the GEMV inside `accumulate_schur` (same config, same
    /// `beta=1` accumulation order over rows), so the resident frame's `δβ`
    /// matches a full `solve()` at the same gradient.
    fn accumulate_schur_rhs_only(
        blas: &CudaBlas,
        d: usize,
        k: usize,
        n: usize,
        y_stack: &CudaSlice<f64>,
        u_stack: &CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let y_block_elems = d * k;
        let u_block_elems = d;
        for i in 0..n {
            let y_slice = y_stack.slice(i * y_block_elems..(i + 1) * y_block_elems);
            let u_slice = u_stack.slice(i * u_block_elems..(i + 1) * u_block_elems);
            let gemv_cfg = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: 1.0,
                lda: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                incx: 1,
                beta: 1.0,
                incy: 1,
            };
            // SAFETY: y_slice (d×k col-major) and u_slice (length d) are live
            // device buffers; `rhs` is the length-k accumulator.
            unsafe { blas.gemv(gemv_cfg, &y_slice, &u_slice, rhs) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    /// Accumulate `g_dev[i] ← u_i + Y_i · δβ` per block. This is the
    /// pre-trsm RHS for the back-substitution `L_i^T x_i = w_i`.
    fn accumulate_back_sub_rhs(
        blas: &CudaBlas,
        d: usize,
        k: usize,
        n: usize,
        y_stack: &CudaSlice<f64>,
        delta_beta: &CudaSlice<f64>,
        u_stack: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let y_block_elems = d * k;
        let u_block_elems = d;
        for i in 0..n {
            let y_slice = y_stack.slice(i * y_block_elems..(i + 1) * y_block_elems);
            let mut u_slice = u_stack.slice_mut(i * u_block_elems..(i + 1) * u_block_elems);
            let gemv_cfg = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: 1.0,
                lda: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                incx: 1,
                beta: 1.0,
                incy: 1,
            };
            // SAFETY: y_slice / delta_beta / u_slice are live device buffers
            // of the expected sizes (d×k, k, d).
            unsafe { blas.gemv(gemv_cfg, &y_slice, delta_beta, &mut u_slice) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────
    // Layer D + E — fused NVRTC dispatch.
    //
    // The forward kernel (`arrow_schur_forward_pgroup`) is a single launch
    // that, per row block, factors `D_i + ρI = L_i L_iᵀ` in shared memory,
    // forward-solves `u_i = L_i⁻¹ g_i` and `Y_i = L_i⁻¹ B_i`, and emits the
    // per-block Schur partials `partial_S[i] = Yᵀ Y` (R×R) and
    // `partial_r[i] = Yᵀ u` (R). The host reduces partials on the CPU after
    // dtoh (one fused sum across `n` blocks of R²+R doubles; cheap because
    // n·R² ≲ 5M doubles at large scale), assembles `S_β`, factors it via
    // cuSOLVER, and launches the back-substitution kernel
    // `arrow_schur_back_sub_pgroup` to recover `δt_i = -L_i⁻ᵀ(u_i + Y_i δβ)`
    // without re-uploading the local factors.
    // ────────────────────────────────────────────────────────────────────

    use std::collections::HashMap;
    use std::sync::Mutex;

    /// One compiled NVRTC module per `(cc_major, cc_minor, p_max, r_template)`.
    /// `cc_*` lets one process drive multiple device generations; the
    /// `(p_max, r_template)` pair selects the shared-memory layout baked into
    /// the kernel source.
    struct FusedModuleCache {
        modules: Mutex<
            HashMap<crate::gpu_kernels::arrow_schur_nvrtc::FusedModuleCacheKey, Arc<CudaModule>>,
        >,
    }

    fn fused_module_cache() -> &'static FusedModuleCache {
        static CACHE: OnceLock<FusedModuleCache> = OnceLock::new();
        CACHE.get_or_init(|| FusedModuleCache {
            modules: Mutex::new(HashMap::new()),
        })
    }

    fn fused_module_for(
        ctx: &Arc<CudaContext>,
        key: crate::gpu_kernels::arrow_schur_nvrtc::FusedModuleCacheKey,
    ) -> Result<Arc<CudaModule>, ArrowSchurGpuFailure> {
        let cache = fused_module_cache();
        if let Ok(guard) = cache.modules.lock() {
            if let Some(existing) = guard.get(&key) {
                return Ok(existing.clone());
            }
        }
        let src = crate::gpu_kernels::arrow_schur_nvrtc::forward_kernel_source(
            key.p_max as usize,
            key.r_template as usize,
        );
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&src).map_err(|err| {
            ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!(
                    "arrow-schur fused NVRTC compile (p_max={}, r={}): {err}",
                    key.p_max, key.r_template
                ),
            }
        })?;
        let module = ctx
            .load_module(ptx)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        if let Ok(mut guard) = cache.modules.lock() {
            guard.entry(key).or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    const PCG_VECTOR_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void arrow_pcg_jacobi_mul(
    const double* __restrict__ inv_diag,
    const double* __restrict__ r,
    double* __restrict__ z,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = inv_diag[idx] * r[idx];
    }
}

extern "C" __global__ void arrow_pcg_update_p(
    const double* __restrict__ z,
    double beta,
    double* __restrict__ p,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p[idx] = z[idx] + beta * p[idx];
    }
}

extern "C" __global__ void arrow_sae_init(
    double* __restrict__ out,
    const double* __restrict__ x,
    double ridge,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ridge * x[idx];
    }
}

extern "C" __global__ void arrow_sae_smooth_matvec(
    const double* __restrict__ x,
    double* __restrict__ out,
    const int* __restrict__ block_offsets,
    const int* __restrict__ block_m,
    const int* __restrict__ factor_ptr,
    const double* __restrict__ factors,
    int p,
    int n_blocks
) {
    int block_id = blockIdx.y;
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) {
        return;
    }
    int m = block_m[block_id];
    int total = m * p;
    if (linear >= total) {
        return;
    }
    int li = linear / p;
    int oc = linear - li * p;
    int off = block_offsets[block_id];
    int fbase = factor_ptr[block_id];
    double acc = 0.0;
    for (int lj = 0; lj < m; ++lj) {
        double a = factors[fbase + li * m + lj];
        acc += a * x[off + lj * p + oc];
    }
    out[off + li * p + oc] += acc;
}

extern "C" __global__ void arrow_sae_sparse_g_matvec(
    const double* __restrict__ x,
    double* __restrict__ out,
    const int* __restrict__ row_off,
    const int* __restrict__ col_off,
    const int* __restrict__ rows,
    const int* __restrict__ cols,
    const int* __restrict__ data_ptr,
    const double* __restrict__ data,
    int p,
    int n_blocks
) {
    int block_id = blockIdx.y;
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) {
        return;
    }
    int m_i = rows[block_id];
    int m_j = cols[block_id];
    int total = m_i * p;
    if (linear >= total) {
        return;
    }
    int li = linear / p;
    int oc = linear - li * p;
    int rbase = row_off[block_id];
    int cbase = col_off[block_id];
    int dbase = data_ptr[block_id];
    double acc = 0.0;
    for (int lj = 0; lj < m_j; ++lj) {
        acc += data[dbase + li * m_j + lj] * x[(cbase + lj) * p + oc];
    }
    // #1017 — a row atom co-occurs with multiple column atoms, so several
    // concurrent (atom_i, atom_j) blocks (blockIdx.y) write the SAME output
    // element `out[(rbase+li)*p+oc]`. A plain `+=` races and loses updates
    // (silently-wrong Schur matvec); accumulate atomically. `double` atomicAdd
    // needs sm_60+, guaranteed by the NVRTC arch pin (#1551).
    atomicAdd(&out[(rbase + li) * p + oc], acc);
}

extern "C" __global__ void arrow_sae_gather_u(
    const double* __restrict__ x,
    const int* __restrict__ row_ptr,
    const int* __restrict__ beta_base,
    const double* __restrict__ phi,
    double* __restrict__ u,
    int p,
    int n_rows
) {
    int row = blockIdx.y;
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || oc >= p) {
        return;
    }
    double acc = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int e = start; e < end; ++e) {
        acc += phi[e] * x[beta_base[e] + oc];
    }
    u[row * p + oc] = acc;
}

extern "C" __global__ void arrow_sae_apply_l(
    const double* __restrict__ u,
    const int* __restrict__ jac_ptr,
    const double* __restrict__ jac,
    double* __restrict__ w,
    int p,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    int jstart = jac_ptr[row];
    int q = (jac_ptr[row + 1] - jstart) / p;
    if (c >= q) {
        return;
    }
    double acc = 0.0;
    for (int oc = 0; oc < p; ++oc) {
        acc += jac[jstart + c * p + oc] * u[row * p + oc];
    }
    w[row * max_q + c] = acc;
}

extern "C" __global__ void arrow_sae_apply_ainv(
    const double* __restrict__ ainv,
    const double* __restrict__ w,
    double* __restrict__ v,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || c >= max_q) {
        return;
    }
    double acc = 0.0;
    int base = row * max_q * max_q;
    for (int j = 0; j < max_q; ++j) {
        acc += ainv[base + c * max_q + j] * w[row * max_q + j];
    }
    v[row * max_q + c] = acc;
}

extern "C" __global__ void arrow_sae_scatter_sub(
    const double* __restrict__ v,
    const int* __restrict__ jac_ptr,
    const double* __restrict__ jac,
    const int* __restrict__ row_ptr,
    const int* __restrict__ beta_base,
    const double* __restrict__ phi,
    double* __restrict__ out,
    int p,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || oc >= p) {
        return;
    }
    int jstart = jac_ptr[row];
    int q = (jac_ptr[row + 1] - jstart) / p;
    double lt_v = 0.0;
    for (int c = 0; c < q; ++c) {
        lt_v += jac[jstart + c * p + oc] * v[row * max_q + c];
    }
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int e = start; e < end; ++e) {
        atomicAdd(&out[beta_base[e] + oc], -phi[e] * lt_v);
    }
}

extern "C" __global__ void arrow_sae_diag_sub(
    double* __restrict__ diag,
    const double* __restrict__ ainv,
    const int* __restrict__ jac_ptr,
    const double* __restrict__ jac,
    const int* __restrict__ row_ptr,
    const int* __restrict__ beta_base,
    const double* __restrict__ phi,
    int p,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || oc >= p) {
        return;
    }
    int jstart = jac_ptr[row];
    int q = (jac_ptr[row + 1] - jstart) / p;
    int abase = row * max_q * max_q;
    double quad = 0.0;
    for (int c = 0; c < q; ++c) {
        double lc = jac[jstart + c * p + oc];
        for (int d = 0; d < q; ++d) {
            quad += lc * ainv[abase + c * max_q + d] * jac[jstart + d * p + oc];
        }
    }
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int e = start; e < end; ++e) {
        double pe = phi[e];
        atomicAdd(&diag[beta_base[e] + oc], -(pe * pe) * quad);
    }
}

/* ── #1017/#1026 frames-engaged device kernels ─────────────────────────────
 * The factored β border is C-space (width Σ M_k·r_k). The penalty side is the
 * smooth `λ S_k ⊗ I_{r_k}` (per-block right-width r_k) plus the data-fit
 * `G_{ij} ⊗ W_{ij}` (W = U_iᵀU_j, dense r_i×r_j). The reduced-Schur term uses
 * the per-row DENSE cross-block H_tβ^(i) (q_i × border_dim, row-major). */

extern "C" __global__ void arrow_sae_frame_smooth_matvec(
    const double* __restrict__ x,
    double* __restrict__ out,
    const int* __restrict__ block_offsets,
    const int* __restrict__ block_m,
    const int* __restrict__ block_r,
    const int* __restrict__ factor_ptr,
    const double* __restrict__ factors,
    int n_blocks
) {
    int block_id = blockIdx.y;
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) {
        return;
    }
    int m = block_m[block_id];
    int r = block_r[block_id];
    int total = m * r;
    if (linear >= total) {
        return;
    }
    int li = linear / r;
    int ib = linear - li * r;
    int off = block_offsets[block_id];
    int fbase = factor_ptr[block_id];
    double acc = 0.0;
    for (int lj = 0; lj < m; ++lj) {
        double a = factors[fbase + li * m + lj];
        acc += a * x[off + lj * r + ib];
    }
    out[off + li * r + ib] += acc;
}

extern "C" __global__ void arrow_sae_frame_g_matvec(
    const double* __restrict__ x,
    double* __restrict__ out,
    const int* __restrict__ off_i,
    const int* __restrict__ off_j,
    const int* __restrict__ r_i,
    const int* __restrict__ r_j,
    const int* __restrict__ m_i,
    const int* __restrict__ m_j,
    const int* __restrict__ g_ptr,
    const double* __restrict__ g_data,
    const int* __restrict__ w_ptr,
    const double* __restrict__ w_data,
    int n_blocks
) {
    int block_id = blockIdx.y;
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) {
        return;
    }
    int ri = r_i[block_id];
    int rj = r_j[block_id];
    int mi = m_i[block_id];
    int mj = m_j[block_id];
    int total = mi * ri;
    if (linear >= total) {
        return;
    }
    int li = linear / ri;       // basis row in atom i
    int a = linear - li * ri;   // frame coord in atom i
    int oi = off_i[block_id];
    int oj = off_j[block_id];
    int gbase = g_ptr[block_id];
    int wbase = w_ptr[block_id];
    double acc = 0.0;
    for (int lj = 0; lj < mj; ++lj) {
        double g = g_data[gbase + li * mj + lj];
        if (g == 0.0) { continue; }
        int xj_base = oj + lj * rj;
        double inner = 0.0;
        for (int b = 0; b < rj; ++b) {
            inner += w_data[wbase + a * rj + b] * x[xj_base + b];
        }
        acc += g * inner;
    }
    // #1017 — same race as `arrow_sae_sparse_g_matvec`: atom i is the row atom of
    // multiple co-occurring (i,j) frame blocks running concurrently on
    // blockIdx.y, all targeting `out[oi+li*ri+a]`. Accumulate atomically so the
    // framed G⊗W matvec is correct (the CPU oracle sums these sequentially).
    atomicAdd(&out[oi + li * ri + a], acc);
}

/* Per-row reduced-Schur subtraction with a DENSE cross-block H_tβ^(i).
 *   h_i   = H_tβ^(i) · x                (length q_i)
 *   s_i   = (H_tt^(i)+ρ_t I)⁻¹ h_i      (apply cached ainv, length q_i)
 *   out  -= (H_tβ^(i))ᵀ · s_i           (scatter into border_dim)
 * `htb` is row-major (q_i × k) flattened, `htb_ptr` gives each row's base and
 * (htb_ptr[row+1]-htb_ptr[row])/k == q_i. `q_of` carries q_i directly. */
extern "C" __global__ void arrow_sae_frame_apply_h(
    const double* __restrict__ x,
    const int* __restrict__ htb_ptr,
    const double* __restrict__ htb,
    const int* __restrict__ q_of,
    double* __restrict__ hvec,
    int k,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) { return; }
    int q = q_of[row];
    if (c >= q) { return; }
    int base = htb_ptr[row] + c * k;
    double acc = 0.0;
    for (int a = 0; a < k; ++a) {
        acc += htb[base + a] * x[a];
    }
    hvec[row * max_q + c] = acc;
}

extern "C" __global__ void arrow_sae_frame_apply_ainv(
    const double* __restrict__ ainv,
    const double* __restrict__ hvec,
    const int* __restrict__ q_of,
    double* __restrict__ svec,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || c >= max_q) { return; }
    int q = q_of[row];
    double acc = 0.0;
    int abase = row * max_q * max_q;
    for (int j = 0; j < q; ++j) {
        acc += ainv[abase + c * max_q + j] * hvec[row * max_q + j];
    }
    svec[row * max_q + c] = acc;
}

extern "C" __global__ void arrow_sae_frame_scatter_h(
    const double* __restrict__ svec,
    const int* __restrict__ htb_ptr,
    const double* __restrict__ htb,
    const int* __restrict__ q_of,
    double* __restrict__ out,
    int k,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || a >= k) { return; }
    int q = q_of[row];
    int hbase = htb_ptr[row];
    double acc = 0.0;
    for (int c = 0; c < q; ++c) {
        acc += htb[hbase + c * k + a] * svec[row * max_q + c];
    }
    atomicAdd(&out[a], -acc);
}

/* #1017 evidence-lane DETERMINISTIC reduced-Schur scatter:
   out[a] = -Σ_i Σ_c H_tβ[i][c,a]·svec[i,c]. One thread owns output coord `a` and
   sums the rows in fixed index order 0..n_rows — NO atomics, so the result is
   run-to-run bit-stable (the SLQ log|S| determinism contract). Same arithmetic as
   arrow_sae_frame_scatter_h; only the reduction order is pinned (there the sum
   over rows is an atomicAdd race). The shared atomic kernel is left untouched —
   the step-PCG relies on it. `out` is fully assigned (no init needed). */
extern "C" __global__ void arrow_sae_frame_scatter_h_det(
    const double* __restrict__ svec,
    const int* __restrict__ htb_ptr,
    const double* __restrict__ htb,
    const int* __restrict__ q_of,
    double* __restrict__ out,
    int k,
    int max_q,
    int n_rows
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= k) { return; }
    double acc = 0.0;
    for (int row = 0; row < n_rows; ++row) {
        int q = q_of[row];
        int hbase = htb_ptr[row];
        int sbase = row * max_q;
        for (int c = 0; c < q; ++c) {
            acc += htb[hbase + c * k + a] * svec[sbase + c];
        }
    }
    out[a] = -acc;
}

/* #1017 evidence-lane 2-STAGE deterministic scatter, STAGE 1 (partials):
   partial[chunk][a] = Σ_{row∈chunk} Σ_c H_tβ[row][c,a]·svec[row,c], for the
   contiguous row range [chunk·rows_per_chunk, …). grid = (⌈k/256⌉, n_chunks);
   thread owns (a, chunk). Replaces the single-strip `arrow_sae_frame_scatter_h_det`
   (⌈k/256⌉ CTAs — only 4 at k=911, one thread serial over ALL n_rows, ~94% of a
   72-SM A10 idle) with ⌈k/256⌉·n_chunks CTAs. Rows are summed in fixed order
   within the chunk and the chunks are reduced in order by stage 2, so the result
   is a FIXED reassociation of the same ordered row sum — run-to-run bit-stable
   (the SLQ log|S| determinism contract) and within the ≤1e-9 CPU-oracle gate. */
extern "C" __global__ void arrow_sae_frame_scatter_h_det_partial(
    const double* __restrict__ svec,
    const int* __restrict__ htb_ptr,
    const double* __restrict__ htb,
    const int* __restrict__ q_of,
    double* __restrict__ partial,
    int k,
    int max_q,
    int n_rows,
    int rows_per_chunk
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= k) { return; }
    int chunk = blockIdx.y;
    int row0 = chunk * rows_per_chunk;
    int row1 = row0 + rows_per_chunk;
    if (row1 > n_rows) { row1 = n_rows; }
    double acc = 0.0;
    for (int row = row0; row < row1; ++row) {
        int q = q_of[row];
        int hbase = htb_ptr[row];
        int sbase = row * max_q;
        for (int c = 0; c < q; ++c) {
            acc += htb[hbase + c * k + a] * svec[sbase + c];
        }
    }
    partial[(long long)chunk * k + a] = acc;
}

/* #1017 STAGE 2 (reduce): out[a] = -Σ_chunk partial[chunk][a], chunks summed in
   fixed order 0..n_chunks. One thread per output coord `a`; ⌈k/256⌉ CTAs. */
extern "C" __global__ void arrow_sae_frame_scatter_h_det_reduce(
    const double* __restrict__ partial,
    double* __restrict__ out,
    int k,
    int n_chunks
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= k) { return; }
    double acc = 0.0;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        acc += partial[(long long)chunk * k + a];
    }
    out[a] = -acc;
}

/* #1017 evidence-lane WARP-COOPERATIVE apply_h: hvec[i][c] = Σ_a H_tβ[i][c,a]·x[a].
   One WARP owns (row, c): lane `l` strides `a = l, l+32, …` over the contiguous
   `H_tβ[i][c,·]` slab (fully coalesced across the warp) and a fixed-order
   __shfl_down tree reduces to lane 0. Replaces the shared `arrow_sae_frame_apply_h`
   (256-thread block, only `q_i` threads active, stride-`k` uncoalesced reads) on
   the evidence path ONLY — the shared kernel is untouched (step-PCG relies on it).
   Reduction order is fixed (lane stride + tree), so the result is run-to-run
   bit-stable; it differs from the scalar kernel only by ULP reassociation, within
   the ≤1e-9 parity gate. Launch: block = max_q·32 (≤1024), grid.x = n_rows;
   warp `w = threadIdx.x/32` handles `c = w` for `w < q_i`. */
extern "C" __global__ void arrow_sae_frame_apply_h_warp(
    const double* __restrict__ x,
    const int* __restrict__ htb_ptr,
    const double* __restrict__ htb,
    const int* __restrict__ q_of,
    double* __restrict__ hvec,
    int k,
    int max_q,
    int n_rows
) {
    int row = blockIdx.x;
    if (row >= n_rows) { return; }
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int q = q_of[row];
    if (warp >= q) { return; }
    int base = htb_ptr[row] + warp * k;
    double acc = 0.0;
    for (int a = lane; a < k; a += 32) {
        acc += htb[base + a] * x[a];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        hvec[row * max_q + warp] = acc;
    }
}

/* Frame Jacobi diagonal subtraction: diag[a] -= Σ_c Σ_d H_tβ[c,a]·ainv[c,d]·H_tβ[d,a]. */
extern "C" __global__ void arrow_sae_frame_diag_sub(
    double* __restrict__ diag,
    const double* __restrict__ ainv,
    const int* __restrict__ htb_ptr,
    const double* __restrict__ htb,
    const int* __restrict__ q_of,
    int k,
    int max_q,
    int n_rows
) {
    int row = blockIdx.y;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || a >= k) { return; }
    int q = q_of[row];
    int hbase = htb_ptr[row];
    int abase = row * max_q * max_q;
    double quad = 0.0;
    for (int c = 0; c < q; ++c) {
        double hc = htb[hbase + c * k + a];
        for (int d = 0; d < q; ++d) {
            quad += hc * ainv[abase + c * max_q + d] * htb[hbase + d * k + a];
        }
    }
    atomicAdd(&diag[a], -quad);
}
"#;

    fn pcg_vector_module(
        ctx: &Arc<CudaContext>,
    ) -> Result<&'static Arc<CudaModule>, ArrowSchurGpuFailure> {
        static CACHE: gam_gpu::device_cache::PtxModuleCache =
            gam_gpu::device_cache::PtxModuleCache::new();
        CACHE
            .get_or_compile(ctx, "arrow_pcg_vector", PCG_VECTOR_KERNEL_SOURCE)
            .map_err(|err| {
                // #1551: an NVRTC compile / module-load failure of
                // PCG_VECTOR_KERNEL_SOURCE means the device SAE PCG cannot run;
                // log it (the historical silent collapse to `Unavailable` is what
                // masked the missing `--gpu-architecture` for so long) and fall
                // back to the CPU.
                log::warn!("[#1551] pcg_vector_module get_or_compile failed: {err}");
                ArrowSchurGpuFailure::Unavailable
            })
    }

    fn pcg_launch_config(n: usize) -> Result<LaunchConfig, ArrowSchurGpuFailure> {
        let threads = 256u32;
        let blocks = ((n as u32).saturating_add(threads - 1) / threads).max(1);
        Ok(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    fn launch_jacobi_mul(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        inv_diag: &CudaSlice<f64>,
        r: &CudaSlice<f64>,
        z: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let kernel = module
            .load_function("arrow_pcg_jacobi_mul")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let n_i32 = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(inv_diag).arg(r).arg(z).arg(&n_i32);
        // SAFETY: all buffers have length n and belong to `stream`; the kernel only
        // reads/writes indices `< n`.
        unsafe { builder.launch(pcg_launch_config(n)?) }
            .map(drop)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
    }

    fn launch_update_p(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        z: &CudaSlice<f64>,
        beta: f64,
        p: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let kernel = module
            .load_function("arrow_pcg_update_p")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let n_i32 = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(z).arg(&beta).arg(p).arg(&n_i32);
        // SAFETY: z/p both have length n and belong to `stream`; the kernel only
        // reads/writes indices `< n`.
        unsafe { builder.launch(pcg_launch_config(n)?) }
            .map(drop)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
    }

    struct DeviceSaePcgBuffers {
        row_ptr: CudaSlice<i32>,
        beta_base: CudaSlice<i32>,
        phi: CudaSlice<f64>,
        jac_ptr: CudaSlice<i32>,
        jac: CudaSlice<f64>,
        smooth_offsets: CudaSlice<i32>,
        smooth_m: CudaSlice<i32>,
        smooth_ptr: CudaSlice<i32>,
        smooth_data: CudaSlice<f64>,
        g_row_off: CudaSlice<i32>,
        g_col_off: CudaSlice<i32>,
        g_rows: CudaSlice<i32>,
        g_cols: CudaSlice<i32>,
        g_ptr: CudaSlice<i32>,
        g_data: CudaSlice<f64>,
        ainv: CudaSlice<f64>,
        u: CudaSlice<f64>,
        w: CudaSlice<f64>,
        v: CudaSlice<f64>,
        n_rows: usize,
        p: usize,
        k: usize,
        max_q: usize,
        smooth_blocks: usize,
        g_blocks: usize,
    }

    fn checked_i32(value: usize) -> Result<i32, ArrowSchurGpuFailure> {
        to_i32(value).ok_or(ArrowSchurGpuFailure::Unavailable)
    }

    fn sae_penalty_diag_host(
        data: &DeviceSaePcgData,
        ridge_beta: f64,
    ) -> Result<Vec<f64>, ArrowSchurGpuFailure> {
        let mut diag = vec![ridge_beta; data.beta_dim];
        for block in &data.smooth_blocks {
            let (rows, cols) = block.factor_a.dim();
            if rows != cols {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            for row in 0..rows {
                let coeff = block.factor_a[[row, row]];
                let base = block
                    .global_offset
                    .checked_add(
                        row.checked_mul(data.p)
                            .ok_or(ArrowSchurGpuFailure::Unavailable)?,
                    )
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?;
                let end = base
                    .checked_add(data.p)
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?;
                if end > diag.len() {
                    return Err(ArrowSchurGpuFailure::Unavailable);
                }
                for channel in 0..data.p {
                    diag[base + channel] += coeff;
                }
            }
        }
        for block in &data.sparse_g_blocks {
            if block.row_off != block.col_off {
                continue;
            }
            let (rows, cols) = block.data.dim();
            for row in 0..rows.min(cols) {
                let coeff = block.data[[row, row]];
                let beta_row = block
                    .row_off
                    .checked_add(row)
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?;
                let base = beta_row
                    .checked_mul(data.p)
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?;
                let end = base
                    .checked_add(data.p)
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?;
                if end > diag.len() {
                    return Err(ArrowSchurGpuFailure::Unavailable);
                }
                for channel in 0..data.p {
                    diag[base + channel] += coeff;
                }
            }
        }
        Ok(diag)
    }

    fn flatten_device_sae_data(
        sys: &ArrowSchurSystem,
        data: &DeviceSaePcgData,
        ridge_t: f64,
        stream: &Arc<CudaStream>,
    ) -> Result<DeviceSaePcgBuffers, ArrowSchurGpuFailure> {
        let n_rows = sys.rows.len();
        let p = data.p;
        let k = data.beta_dim;
        if data.a_phi.len() != n_rows || data.local_jac.len() != n_rows {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }

        let mut row_ptr_host = Vec::with_capacity(n_rows + 1);
        let mut beta_base_host = Vec::<i32>::new();
        let mut phi_host = Vec::<f64>::new();
        row_ptr_host.push(0_i32);
        for row in data.a_phi.iter() {
            for &(base, phi) in row {
                beta_base_host.push(checked_i32(base)?);
                phi_host.push(phi);
            }
            row_ptr_host.push(checked_i32(beta_base_host.len())?);
        }

        let mut jac_ptr_host = Vec::with_capacity(n_rows + 1);
        let mut jac_host = Vec::<f64>::new();
        let mut max_q = 0usize;
        jac_ptr_host.push(0_i32);
        for row_jac in data.local_jac.iter() {
            if row_jac.len() % p != 0 {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            max_q = max_q.max(row_jac.len() / p);
            jac_host.extend_from_slice(row_jac);
            jac_ptr_host.push(checked_i32(jac_host.len())?);
        }
        if max_q == 0 {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }

        let mut smooth_offsets_host = Vec::with_capacity(data.smooth_blocks.len());
        let mut smooth_m_host = Vec::with_capacity(data.smooth_blocks.len());
        let mut smooth_ptr_host = Vec::with_capacity(data.smooth_blocks.len() + 1);
        let mut smooth_data_host = Vec::<f64>::new();
        smooth_ptr_host.push(0_i32);
        for block in &data.smooth_blocks {
            let (rows, cols) = block.factor_a.dim();
            if rows != cols {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            smooth_offsets_host.push(checked_i32(block.global_offset)?);
            smooth_m_host.push(checked_i32(rows)?);
            for r in 0..rows {
                for c in 0..cols {
                    smooth_data_host.push(block.factor_a[[r, c]]);
                }
            }
            smooth_ptr_host.push(checked_i32(smooth_data_host.len())?);
        }

        let mut g_row_off_host = Vec::with_capacity(data.sparse_g_blocks.len());
        let mut g_col_off_host = Vec::with_capacity(data.sparse_g_blocks.len());
        let mut g_rows_host = Vec::with_capacity(data.sparse_g_blocks.len());
        let mut g_cols_host = Vec::with_capacity(data.sparse_g_blocks.len());
        let mut g_ptr_host = Vec::with_capacity(data.sparse_g_blocks.len() + 1);
        let mut g_data_host = Vec::<f64>::new();
        g_ptr_host.push(0_i32);
        for block in &data.sparse_g_blocks {
            let (rows, cols) = block.data.dim();
            g_row_off_host.push(checked_i32(block.row_off)?);
            g_col_off_host.push(checked_i32(block.col_off)?);
            g_rows_host.push(checked_i32(rows)?);
            g_cols_host.push(checked_i32(cols)?);
            for r in 0..rows {
                for c in 0..cols {
                    g_data_host.push(block.data[[r, c]]);
                }
            }
            g_ptr_host.push(checked_i32(g_data_host.len())?);
        }

        let mut ainv_host = vec![0.0_f64; n_rows * max_q * max_q];
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let q = data.local_jac[row_idx].len() / p;
            if row.htt.dim() != (q, q) {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!(
                        "SAE device PCG row {row_idx}: H_tt shape {:?} != ({q}, {q})",
                        row.htt.dim()
                    ),
                });
            }
            let mut block = row.htt.clone();
            for d in 0..q {
                block[[d, d]] += ridge_t;
            }
            let factor = gam_linalg::triangular::cholesky_factor_in_place(
                block.view(),
                gam_linalg::triangular::CholeskyGuard::NonnegativePivot,
            )
            .ok_or_else(|| {
                // Deficit-aware bump (Gershgorin λ_min bound) so a strongly
                // indefinite per-row block recovers in one outer-loop retry.
                ArrowSchurGpuFailure::RidgeBumpRequired {
                    row: row_idx,
                    bump: super::ridge_bump_to_make_pd(row.htt.view(), ridge_t),
                }
            })?;
            for col in 0..q {
                let mut e = Array1::<f64>::zeros(q);
                e[col] = 1.0;
                let solved = gam_linalg::triangular::cholesky_solve_vector(factor.view(), e.view());
                for r in 0..q {
                    ainv_host[row_idx * max_q * max_q + r * max_q + col] = solved[r];
                }
            }
        }

        Ok(DeviceSaePcgBuffers {
            row_ptr: stream
                .clone_htod(&row_ptr_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            beta_base: stream
                .clone_htod(&beta_base_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            phi: stream
                .clone_htod(&phi_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            jac_ptr: stream
                .clone_htod(&jac_ptr_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            jac: stream
                .clone_htod(&jac_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            smooth_offsets: stream
                .clone_htod(&smooth_offsets_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            smooth_m: stream
                .clone_htod(&smooth_m_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            smooth_ptr: stream
                .clone_htod(&smooth_ptr_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            smooth_data: stream
                .clone_htod(&smooth_data_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            g_row_off: stream
                .clone_htod(&g_row_off_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            g_col_off: stream
                .clone_htod(&g_col_off_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            g_rows: stream
                .clone_htod(&g_rows_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            g_cols: stream
                .clone_htod(&g_cols_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            g_ptr: stream
                .clone_htod(&g_ptr_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            g_data: stream
                .clone_htod(&g_data_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            ainv: stream
                .clone_htod(&ainv_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            u: stream
                .alloc_zeros::<f64>(n_rows * p)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            w: stream
                .alloc_zeros::<f64>(n_rows * max_q)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            v: stream
                .alloc_zeros::<f64>(n_rows * max_q)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            n_rows,
            p,
            k,
            max_q,
            smooth_blocks: data.smooth_blocks.len(),
            g_blocks: data.sparse_g_blocks.len(),
        })
    }

    fn launch_sae_init(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        out: &mut CudaSlice<f64>,
        x: &CudaSlice<f64>,
        ridge: f64,
        n: usize,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let kernel = module
            .load_function("arrow_sae_init")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let n_i32 = checked_i32(n)?;
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(out).arg(x).arg(&ridge).arg(&n_i32);
        // SAFETY: `out` and `x` are live device buffers with at least `n`
        // entries on `stream`; the kernel writes one in-bounds element per
        // launched index below `n`.
        unsafe { builder.launch(pcg_launch_config(n)?) }
            .map(drop)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
    }

    fn launch_sae_penalty_matvec(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        buffers: &mut DeviceSaePcgBuffers,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
        ridge_beta: f64,
    ) -> Result<(), ArrowSchurGpuFailure> {
        launch_sae_init(stream, module, out, x, ridge_beta, buffers.k)?;
        if buffers.smooth_blocks > 0 {
            let kernel = module
                .load_function("arrow_sae_smooth_matvec")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let max_m = buffers.k;
            let p_i32 = checked_i32(buffers.p)?;
            let blocks_i32 = checked_i32(buffers.smooth_blocks)?;
            let cfg = LaunchConfig {
                grid_dim: (
                    ((max_m as u32).saturating_add(255) / 256).max(1),
                    checked_i32(buffers.smooth_blocks)? as u32,
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&kernel);
            builder
                .arg(x)
                .arg(&mut *out)
                .arg(&buffers.smooth_offsets)
                .arg(&buffers.smooth_m)
                .arg(&buffers.smooth_ptr)
                .arg(&buffers.smooth_data)
                .arg(&p_i32)
                .arg(&blocks_i32);
            // SAFETY: smooth block metadata and dense smooth data were flattened
            // into live device buffers; the 2D grid covers only declared block
            // and coefficient-channel work items, and the kernel bounds-checks
            // against each block's stored size.
            unsafe { builder.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        if buffers.g_blocks > 0 {
            let kernel = module
                .load_function("arrow_sae_sparse_g_matvec")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let max_work = buffers
                .k
                .checked_div(buffers.p)
                .unwrap_or(0)
                .saturating_mul(buffers.p);
            let p_i32 = checked_i32(buffers.p)?;
            let blocks_i32 = checked_i32(buffers.g_blocks)?;
            let cfg = LaunchConfig {
                grid_dim: (
                    ((max_work as u32).saturating_add(255) / 256).max(1),
                    checked_i32(buffers.g_blocks)? as u32,
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&kernel);
            builder
                .arg(x)
                .arg(&mut *out)
                .arg(&buffers.g_row_off)
                .arg(&buffers.g_col_off)
                .arg(&buffers.g_rows)
                .arg(&buffers.g_cols)
                .arg(&buffers.g_ptr)
                .arg(&buffers.g_data)
                .arg(&p_i32)
                .arg(&blocks_i32);
            // SAFETY: sparse G block metadata/data are live device buffers built
            // from host CSR-like block descriptors; the launch dimensions cover
            // declared block work only and the kernel checks row/column bounds
            // before reading or accumulating.
            unsafe { builder.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    fn launch_sae_row_schur_sub(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        buffers: &mut DeviceSaePcgBuffers,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let p_i32 = checked_i32(buffers.p)?;
        let max_q_i32 = checked_i32(buffers.max_q)?;
        let n_rows_i32 = checked_i32(buffers.n_rows)?;
        let cfg_p_rows = LaunchConfig {
            grid_dim: (
                ((buffers.p as u32).saturating_add(255) / 256).max(1),
                checked_i32(buffers.n_rows)? as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let gather = module
            .load_function("arrow_sae_gather_u")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        {
            let mut builder = stream.launch_builder(&gather);
            builder
                .arg(x)
                .arg(&buffers.row_ptr)
                .arg(&buffers.beta_base)
                .arg(&buffers.phi)
                .arg(&mut buffers.u)
                .arg(&p_i32)
                .arg(&n_rows_i32);
            // SAFETY: `x`, row pointers, beta offsets, basis rows, and `u` are
            // live device buffers sized for `n_rows` by `p`; the kernel guards
            // row/channel indices before gathering.
            unsafe { builder.launch(cfg_p_rows) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }

        let cfg_q_rows = LaunchConfig {
            grid_dim: (
                ((buffers.max_q as u32).saturating_add(255) / 256).max(1),
                checked_i32(buffers.n_rows)? as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let apply_l = module
            .load_function("arrow_sae_apply_l")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        {
            let mut builder = stream.launch_builder(&apply_l);
            builder
                .arg(&buffers.u)
                .arg(&buffers.jac_ptr)
                .arg(&buffers.jac)
                .arg(&mut buffers.w)
                .arg(&p_i32)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: `u`, Jacobian row pointers/data, and `w` are live buffers
            // sized for the `(n_rows, p)` to `(n_rows, max_q)` multiply; the
            // kernel checks row and local-coordinate bounds.
            unsafe { builder.launch(cfg_q_rows) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }

        let apply_ainv = module
            .load_function("arrow_sae_apply_ainv")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        {
            let mut builder = stream.launch_builder(&apply_ainv);
            builder
                .arg(&buffers.ainv)
                .arg(&buffers.w)
                .arg(&mut buffers.v)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: `ainv`, `w`, and `v` are live device buffers sized for
            // `n_rows * max_q`; the kernel guards all row/local-coordinate
            // indices before reading or writing.
            unsafe { builder.launch(cfg_q_rows) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }

        let scatter = module
            .load_function("arrow_sae_scatter_sub")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        {
            let mut builder = stream.launch_builder(&scatter);
            builder
                .arg(&buffers.v)
                .arg(&buffers.jac_ptr)
                .arg(&buffers.jac)
                .arg(&buffers.row_ptr)
                .arg(&buffers.beta_base)
                .arg(&buffers.phi)
                .arg(out)
                .arg(&p_i32)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: `v`, Jacobian metadata, row pointers, beta offsets, basis
            // rows, and `out` are live buffers for `n_rows` by `p`; scatter
            // indices are checked against row and channel bounds in the kernel.
            unsafe { builder.launch(cfg_p_rows) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    fn launch_sae_diag_sub(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        buffers: &DeviceSaePcgBuffers,
        diag: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let kernel = module
            .load_function("arrow_sae_diag_sub")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let p_i32 = checked_i32(buffers.p)?;
        let max_q_i32 = checked_i32(buffers.max_q)?;
        let n_rows_i32 = checked_i32(buffers.n_rows)?;
        let cfg = LaunchConfig {
            grid_dim: (
                ((buffers.p as u32).saturating_add(255) / 256).max(1),
                checked_i32(buffers.n_rows)? as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernel);
        builder
            .arg(diag)
            .arg(&buffers.ainv)
            .arg(&buffers.jac_ptr)
            .arg(&buffers.jac)
            .arg(&buffers.row_ptr)
            .arg(&buffers.beta_base)
            .arg(&buffers.phi)
            .arg(&p_i32)
            .arg(&max_q_i32)
            .arg(&n_rows_i32);
        // SAFETY: diagonal output and all read-only SAE row metadata buffers are
        // live on `stream` with sizes matching `n_rows`, `p`, and `max_q`; the
        // kernel bounds-checks its flattened work index.
        unsafe { builder.launch(cfg) }
            .map(drop)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
    }

    fn launch_sae_matvec(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        buffers: &mut DeviceSaePcgBuffers,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
        ridge_beta: f64,
    ) -> Result<(), ArrowSchurGpuFailure> {
        launch_sae_penalty_matvec(stream, module, buffers, x, out, ridge_beta)?;
        launch_sae_row_schur_sub(stream, module, buffers, x, out)
    }

    /// Pack `D + ρ_t I`, `B`, and `g` into the strided `(n × P_MAX × P_MAX)`
    /// / `(n × P_MAX × R_TEMPLATE)` / `(n × P_MAX)` layout the fused kernel
    /// expects. Entries outside the runtime `(p, r)` window stay at zero so
    /// the kernel's per-element loops are safe to no-op there.
    fn pack_fused_host(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        p_max: usize,
        r_template: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let mut d_buf = vec![0.0_f64; n * p_max * p_max];
        let mut b_buf = vec![0.0_f64; n * p_max * r_template];
        let mut g_buf = vec![0.0_f64; n * p_max];
        for (i, row) in sys.rows.iter().enumerate() {
            // D_i + ρI, column-major in P_MAX×P_MAX strided block.
            for col in 0..d {
                let base = (i * p_max + col) * p_max;
                for r in 0..d {
                    let mut value = row.htt[[r, col]];
                    if r == col {
                        value += ridge_t;
                    }
                    d_buf[base + r] = value;
                }
            }
            // B_i in P_MAX×R_TEMPLATE strided block. The per-row (per-i) block
            // stride is `p_max · r_template` (matching the `b_buf` allocation
            // above and the kernel's `b_stack`/`y_out` layout), NOT
            // `p_max · p_max`: using the D-block multiplier here overflows the
            // buffer whenever `p_max > r_template` (e.g. d=30→p_max=32,
            // k=5→r_template=5). The within-block element offset stays
            // column-major `col·p_max + r` (P_MAX rows per column).
            for col in 0..k {
                let base = (i * r_template + col) * p_max;
                for r in 0..d {
                    b_buf[base + r] = row.htbeta[[r, col]];
                }
            }
            // g_i in P_MAX strided vector.
            let g_base = i * p_max;
            for r in 0..d {
                g_buf[g_base + r] = row.gt[r];
            }
        }
        (d_buf, b_buf, g_buf)
    }

    // -----------------------------------------------------------------------
    // #1017 Phase 3: across-iteration device residency.
    //
    // `solve()` re-packs and re-uploads `D` (`H_tt`), `B` (`H_tβ`) and `g`,
    // then re-runs the per-row POTRF and the border Schur factorization on
    // EVERY call. For the SAE joint inner Newton at a frozen gate/basis frame
    // the Hessian blocks `D`, `B`, `H_ββ` are CONSTANT across the inner loop —
    // only the gradient `g = r(z) = H z − g₀` changes per iterate. So the
    // factor work (`O(n·d³ + p³)`) and the dominant `O(n·d·p)` cross-block
    // upload are pure waste when repeated per iterate.
    //
    // `ResidentArrowFrame` performs that constant work ONCE at construction:
    // upload+ridge+POTRF of `D` (keeping `L_i` resident in `l_dev`), the
    // forward solve `Y_i = L_i^{-1} B_i` (kept resident in `y_dev`), and the
    // Schur assembly + border POTRF (keeping `L_S` resident in `schur_dev`).
    // Each subsequent `solve_gradient(g)` uploads only the `n·d` row gradient,
    // runs the cheap residual path — `u_i = L_i^{-1} g_i` (one batched TRSM),
    // Schur RHS `−g_β + Σ Y_iᵀ u_i`, `δβ = L_S^{-T} L_S^{-1} rhs` (two TRSM,
    // NO POTRF), back-sub `δt_i = −L_i^{-T}(u_i + Y_i δβ)` — and reads back only
    // `δ` and the cached log|H|. The heavy buffers never leave the device
    // across iterations; the per-iterate host transfer is `O(n·d + p)`, not
    // `O(n·d·p)`. Numerics are bit-identical to a `solve()` at the same
    // `(D, B, H_ββ, g, ridge_t, ridge_beta)` because the factor buffers and the
    // helper kernels are the same; the resident path merely SKIPS re-deriving
    // the parts that do not depend on `g`. The CPU dense reference
    // (`solve_arrow_newton_step_dense_reference`) is the parity oracle.
    pub(super) struct ResidentArrowFrame {
        n: usize,
        d: usize,
        k: usize,
        stream: Arc<CudaStream>,
        blas: CudaBlas,
        /// Per-row lower Cholesky factors `L_i` of `H_tt + ρ_t I`, stacked
        /// column-major (`n` tiles of `d×d`). Resident across iterations.
        l_dev: CudaSlice<f64>,
        /// Whitened cross blocks `Y_i = L_i^{-1} H_tβ^(i)`, stacked column-major
        /// (`n` tiles of `d×k`). Resident across iterations.
        y_dev: CudaSlice<f64>,
        /// Lower Cholesky factor `L_S` of the reduced Schur complement
        /// `S_β = H_ββ + ρ_β I − Σ_i Y_iᵀ Y_i`. Resident across iterations.
        schur_dev: CudaSlice<f64>,
        /// `log|H| = 2 Σ log L_{i,jj} + 2 Σ log L_{S,aa}`, constant for the
        /// frame (depends only on the factored Hessian, not on `g`).
        log_det_hessian: f64,
    }

    impl ResidentArrowFrame {
        /// Upload the constant Hessian blocks and perform the one-time factor
        /// work (`POTRF(D)`, `Y_i = L_i^{-1} B_i`, Schur assembly + border
        /// `POTRF`). The frame then serves cheap per-gradient solves.
        pub(super) fn new(
            sys: &ArrowSchurSystem,
            ridge_t: f64,
            ridge_beta: f64,
        ) -> Result<Self, ArrowSchurGpuFailure> {
            if ridge_t.is_nan() || ridge_beta.is_nan() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: "ridge is NaN".to_string(),
                });
            }
            let n = sys.rows.len();
            let d = sys.d;
            let k = sys.k;
            let runtime = route_through_gpu(DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n })
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let stream = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
                .and_then(|ctx| ctx.new_stream().ok())
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let solver =
                DnHandle::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let blas =
                CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            // Upload the constant blocks. `g` is uploaded per-gradient, not here.
            let (d_host, b_host, _g_host) = pack_host(sys, ridge_t);
            let mut l_dev = stream
                .clone_htod(&d_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let mut y_dev = stream
                .clone_htod(&b_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            // POTRF(D) → L_i, kept resident in l_dev.
            let info_host = potrf_batched(&solver, &stream, d, n, &mut l_dev)?;
            if let Some(idx) = info_host.iter().position(|info| *info != 0) {
                // cuSOLVER `info` is a 1-based pivot row index; size the bump
                // from the block (Gershgorin λ_min bound) so a strongly
                // indefinite block recovers in one retry.
                return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                    row: idx,
                    bump: super::ridge_bump_to_make_pd(sys.rows[idx].htt.view(), ridge_t),
                });
            }

            // Y_i = L_i^{-1} B_i, in place over y_dev. Kept resident.
            trsm_batched_lower_inplace(&blas, &stream, d, n, k, &l_dev, &mut y_dev)?;

            // Schur assembly S_β = (H_ββ + ρ_β I) − Σ Y_iᵀ Y_i, then POTRF → L_S.
            // The RHS accumulation is folded into the gradient path; here we
            // only need the (gradient-independent) Schur factor, so accumulate
            // into a throwaway rhs buffer.
            let schur_init: Vec<f64> = {
                let mut tmp = Vec::with_capacity(k * k);
                for col in 0..k {
                    for row in 0..k {
                        let mut v = sys.hbb[[row, col]];
                        if row == col {
                            v += ridge_beta;
                        }
                        tmp.push(v);
                    }
                }
                tmp
            };
            let mut schur_dev = stream
                .clone_htod(&schur_init)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            // A zero u-stack makes `Σ Y_iᵀ u_i = 0`, so only the `−Σ Y_iᵀ Y_i`
            // Schur term is accumulated (the rhs is rebuilt per gradient).
            let zero_u = stream
                .clone_htod(&vec![0.0_f64; n * d])
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let mut throwaway_rhs = stream
                .clone_htod(&vec![0.0_f64; k])
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            accumulate_schur(
                &blas,
                d,
                k,
                n,
                &y_dev,
                &zero_u,
                &mut schur_dev,
                &mut throwaway_rhs,
            )?;
            let info = potrf_single(&solver, &stream, k, &mut schur_dev)?;
            if info != 0 {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("Schur Cholesky failed at pivot {info}"),
                });
            }

            // log|H| from the resident factors (constant for the frame).
            let l_local_host = stream
                .clone_dtoh(&l_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let l_schur_host = stream
                .clone_dtoh(&schur_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let mut log_det = 0.0_f64;
            for i in 0..n {
                let base = i * d * d;
                for j in 0..d {
                    log_det += l_local_host[base + j * d + j].ln();
                }
            }
            for j in 0..k {
                log_det += l_schur_host[j * k + j].ln();
            }
            log_det *= 2.0;

            Ok(Self {
                n,
                d,
                k,
                stream,
                blas,
                l_dev,
                y_dev,
                schur_dev,
                log_det_hessian: log_det,
            })
        }

        #[inline]
        pub(super) fn log_det_hessian(&self) -> f64 {
            self.log_det_hessian
        }

        /// Solve `H δ = −gradient` for a fresh gradient `(g_t, g_β)` reusing the
        /// resident factors. Uploads only `g_t` (`n·d` scalars); reads back only
        /// `δ`. No POTRF runs here — all factorization is amortized into `new`.
        pub(super) fn solve_gradient(
            &self,
            g_t: &[f64],
            g_beta: &[f64],
        ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
            let n = self.n;
            let d = self.d;
            let k = self.k;
            if g_t.len() != n * d || g_beta.len() != k {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!(
                        "resident gradient shape mismatch: g_t={} (want {}), g_beta={} (want {})",
                        g_t.len(),
                        n * d,
                        g_beta.len(),
                        k
                    ),
                });
            }
            // Upload the per-iterate row gradient → u_i = L_i^{-1} g_i in place.
            let mut u_dev = self
                .stream
                .clone_htod(&g_t.to_vec())
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            trsm_batched_lower_inplace(&self.blas, &self.stream, d, n, 1, &self.l_dev, &mut u_dev)?;

            // Schur RHS = −g_β + Σ_i Y_iᵀ u_i. Reuse the resident Schur factor
            // (no POTRF, and skip the −Σ Y_iᵀ Y_i GEMM already baked into L_S).
            let rhs_init: Vec<f64> = g_beta.iter().map(|v| -v).collect();
            let mut rhs_dev = self
                .stream
                .clone_htod(&rhs_init)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            accumulate_schur_rhs_only(&self.blas, d, k, n, &self.y_dev, &u_dev, &mut rhs_dev)?;

            // δβ ← L_S^{-T} L_S^{-1} rhs using the resident border factor.
            trsm_single(
                &self.blas,
                &self.stream,
                k,
                &self.schur_dev,
                &mut rhs_dev,
                false,
                false,
            )?;
            trsm_single(
                &self.blas,
                &self.stream,
                k,
                &self.schur_dev,
                &mut rhs_dev,
                false,
                true,
            )?;
            let delta_beta_host = self
                .stream
                .clone_dtoh(&rhs_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let delta_beta = Array1::from_vec(delta_beta_host);

            // Back-sub δt_i = −L_i^{-T}(u_i + Y_i δβ).
            accumulate_back_sub_rhs(&self.blas, d, k, n, &self.y_dev, &rhs_dev, &mut u_dev)?;
            trsm_batched_lower_inplace_transposed(
                &self.blas,
                &self.stream,
                d,
                n,
                1,
                &self.l_dev,
                &mut u_dev,
            )?;
            let x_host = self
                .stream
                .clone_dtoh(&u_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let mut delta_t = Array1::<f64>::zeros(n * d);
            for (i, v) in x_host.iter().enumerate() {
                delta_t[i] = -*v;
            }

            Ok(ArrowSchurGpuSolution {
                delta_t,
                delta_beta,
                log_det_hessian: self.log_det_hessian,
            })
        }
    }

    /// #1017 base-resident frame for the LM ridge ladder: holds the ridge-
    /// INDEPENDENT base blocks resident and re-factors at each requested ridge.
    /// This is the residency counterpart to [`ResidentArrowFrame`], which bakes
    /// the ridge into its factors (wrong invariant for the ladder, whose trials
    /// vary the ridge while the gradient stays fixed).
    pub(super) struct ResidentBaseArrowFrame {
        n: usize,
        d: usize,
        k: usize,
        stream: Arc<CudaStream>,
        solver: DnHandle,
        blas: CudaBlas,
        /// `D = H_tt` at ridge 0, host-side (`n` stacked column-major `d×d`
        /// tiles). Per trial a copy gets `ridge_t` added to every tile diagonal
        /// and is uploaded (the `O(n·d·d)` re-diagonalised `D` is tiny relative to
        /// the resident `B`); the same copy feeds the `RidgeBumpRequired`
        /// Gershgorin bound on a non-PD pivot.
        d_base_host: Vec<f64>,
        /// `B = H_tβ` resident (`n` stacked column-major `d×k` tiles). Ridge-free;
        /// this is the bulk of the per-trial transfer the residency eliminates.
        base_b_dev: CudaSlice<f64>,
        /// Border `H_ββ` resident (column-major `k×k`). Ridge-free; `ridge_beta`
        /// is added to a per-trial device copy's diagonal.
        base_hbb_dev: CudaSlice<f64>,
        /// Row gradient `g_t` resident (`n·d`). Ridge-free.
        g_t_dev: CudaSlice<f64>,
        /// Border gradient `g_β` host-side (`k`); the tiny `−g_β` RHS is rebuilt
        /// per trial.
        gb_host: Vec<f64>,
        /// Resident all-ones vector (length `k`) whose strided daxpy adds
        /// `ridge_beta` to the `k×k` Schur base diagonal on-device.
        ones_k_dev: CudaSlice<f64>,
    }

    impl ResidentBaseArrowFrame {
        /// Upload the ridge-independent base blocks once. No POTRF runs here.
        pub(super) fn new(sys: &ArrowSchurSystem) -> Result<Self, ArrowSchurGpuFailure> {
            let n = sys.rows.len();
            let d = sys.d;
            let k = sys.k;
            let runtime = route_through_gpu(DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n })
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let stream = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
                .and_then(|ctx| ctx.new_stream().ok())
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let solver =
                DnHandle::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let blas =
                CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            // Base blocks at ridge 0 (ridge-independent); g_t stacked (n·d).
            let (d_base_host, b_host, g_host) = pack_host(sys, 0.0);
            let base_b_dev = stream
                .clone_htod(&b_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let g_t_dev = stream
                .clone_htod(&g_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            // Border H_ββ, column-major, NO ridge (ridge_beta added per trial).
            let hbb_base: Vec<f64> = {
                let mut tmp = Vec::with_capacity(k * k);
                for col in 0..k {
                    for row in 0..k {
                        tmp.push(sys.hbb[[row, col]]);
                    }
                }
                tmp
            };
            let base_hbb_dev = stream
                .clone_htod(&hbb_base)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            let gb_host: Vec<f64> = sys.gb.iter().copied().collect();
            let ones_k_dev = stream
                .clone_htod(&vec![1.0_f64; k])
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            Ok(Self {
                n,
                d,
                k,
                stream,
                solver,
                blas,
                d_base_host,
                base_b_dev,
                base_hbb_dev,
                g_t_dev,
                gb_host,
                ones_k_dev,
            })
        }

        /// Factor the resident base blocks at `(ridge_t, ridge_beta)` and solve.
        /// Mirrors the full [`solve`] sequence, but sources `D`/`B`/`H_ββ`/`g`
        /// from the resident buffers (device-to-device copy into scratch) instead
        /// of re-uploading them, so it is bit-identical to [`solve`] at the same
        /// ridge while moving only the ridge scalars + re-diagonalised `D`.
        pub(super) fn refactor_and_solve(
            &self,
            ridge_t: f64,
            ridge_beta: f64,
        ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
            if ridge_t.is_nan() || ridge_beta.is_nan() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: "ridge is NaN".to_string(),
                });
            }
            let (n, d, k) = (self.n, self.d, self.k);

            // ----- D + ridge_t·I: add on a host copy (tiny) and upload as work L.
            // Tile i is column-major d×d, so its diagonal entries are at
            // i·d·d + j·(d+1) — matching pack_block's `value += ridge_t` on r==col.
            let mut d_ridged = self.d_base_host.clone();
            for i in 0..n {
                for j in 0..d {
                    d_ridged[i * d * d + j * (d + 1)] += ridge_t;
                }
            }
            let mut l_dev = self
                .stream
                .clone_htod(&d_ridged)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            // POTRF(D) → L_i. A non-PD pivot is the LM escalation's signal.
            let info_host = potrf_batched(&self.solver, &self.stream, d, n, &mut l_dev)?;
            if let Some(idx) = info_host.iter().position(|info| *info != 0) {
                let base = idx * d * d;
                return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                    row: idx,
                    // The tile already carries ridge_t on its diagonal, so the
                    // Gershgorin bound is taken at ridge 0 (see
                    // `ridge_bump_to_make_pd_colmajor`).
                    bump: super::ridge_bump_to_make_pd_colmajor(&d_ridged[base..base + d * d], d),
                });
            }

            // ----- Y_i = L_i^{-1} B_i on a device copy of the resident base B.
            let mut y_dev = self
                .stream
                .alloc_zeros::<f64>(n * d * k)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            self.stream
                .memcpy_dtod(&self.base_b_dev, &mut y_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            trsm_batched_lower_inplace(&self.blas, &self.stream, d, n, k, &l_dev, &mut y_dev)?;

            // ----- u_i = L_i^{-1} g_i on a device copy of the resident base g_t.
            let mut u_dev = self
                .stream
                .alloc_zeros::<f64>(n * d)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            self.stream
                .memcpy_dtod(&self.g_t_dev, &mut u_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            trsm_batched_lower_inplace(&self.blas, &self.stream, d, n, 1, &l_dev, &mut u_dev)?;

            // ----- Schur S = (H_ββ + ridge_β I) − Σ Y_iᵀ Y_i.
            let mut schur_dev = self
                .stream
                .alloc_zeros::<f64>(k * k)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            self.stream
                .memcpy_dtod(&self.base_hbb_dev, &mut schur_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            // schur diag += ridge_β (column-major k×k → stride k+1 over the ones).
            device_axpy_strided(
                &self.blas,
                &self.stream,
                k,
                ridge_beta,
                &self.ones_k_dev,
                1,
                &mut schur_dev,
                k + 1,
            )?;
            let rhs_init: Vec<f64> = self.gb_host.iter().map(|v| -v).collect();
            let mut rhs_dev = self
                .stream
                .clone_htod(&rhs_init)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            accumulate_schur(
                &self.blas,
                d,
                k,
                n,
                &y_dev,
                &u_dev,
                &mut schur_dev,
                &mut rhs_dev,
            )?;

            // ----- Factor S_β, solve δβ = L_S^{-T} L_S^{-1} rhs.
            let info = potrf_single(&self.solver, &self.stream, k, &mut schur_dev)?;
            if info != 0 {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("Schur Cholesky failed at pivot {info}"),
                });
            }
            trsm_single(
                &self.blas,
                &self.stream,
                k,
                &schur_dev,
                &mut rhs_dev,
                false,
                false,
            )?;
            trsm_single(
                &self.blas,
                &self.stream,
                k,
                &schur_dev,
                &mut rhs_dev,
                false,
                true,
            )?;
            let delta_beta_host = self
                .stream
                .clone_dtoh(&rhs_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let delta_beta = Array1::from_vec(delta_beta_host);

            // ----- Back-sub δt_i = −L_i^{-T}(u_i + Y_i δβ).
            accumulate_back_sub_rhs(&self.blas, d, k, n, &y_dev, &rhs_dev, &mut u_dev)?;
            trsm_batched_lower_inplace_transposed(
                &self.blas,
                &self.stream,
                d,
                n,
                1,
                &l_dev,
                &mut u_dev,
            )?;
            let x_host = self
                .stream
                .clone_dtoh(&u_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let mut delta_t = Array1::<f64>::zeros(n * d);
            for (i, v) in x_host.iter().enumerate() {
                delta_t[i] = -*v;
            }

            // ----- log|H| = 2 Σ log L_{i,jj} + 2 Σ log L_{S,aa}.
            let l_local_host = self
                .stream
                .clone_dtoh(&l_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let l_schur_host = self
                .stream
                .clone_dtoh(&schur_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let mut log_det = 0.0_f64;
            for i in 0..n {
                let base = i * d * d;
                for j in 0..d {
                    log_det += l_local_host[base + j * d + j].ln();
                }
            }
            for j in 0..k {
                log_det += l_schur_host[j * k + j].ln();
            }
            log_det *= 2.0;

            Ok(ArrowSchurGpuSolution {
                delta_t,
                delta_beta,
                log_det_hessian: log_det,
            })
        }
    }

    pub(super) fn solve_fused(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let plan = crate::gpu_kernels::arrow_schur_nvrtc::plan_fused_launch(n, d, k)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let p_max = plan.p_max;
        let r_template = plan.r_template;

        let runtime = gam_gpu::linalg_dispatch::route_through_gpu(
            gam_gpu::linalg_dispatch::DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n },
        )
        .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let cap = &runtime.device.capability;
        let key = crate::gpu_kernels::arrow_schur_nvrtc::FusedModuleCacheKey {
            cc_major: cap.compute_major,
            cc_minor: cap.compute_minor,
            p_max: p_max as u32,
            r_template: r_template as u32,
        };
        let module = fused_module_for(&ctx, key)?;
        let forward = module
            .load_function("arrow_schur_forward_pgroup")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let back_sub = module
            .load_function("arrow_schur_back_sub_pgroup")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // ----- Upload packed D, B, g -----
        let (d_host, b_host, g_host) = pack_fused_host(sys, ridge_t, p_max, r_template);
        let d_dev = stream
            .clone_htod(&d_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let b_dev = stream
            .clone_htod(&b_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let g_dev = stream
            .clone_htod(&g_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut l_out = stream
            .alloc_zeros::<f64>(n * p_max * p_max)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut u_out = stream
            .alloc_zeros::<f64>(n * p_max)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut y_out = stream
            .alloc_zeros::<f64>(n * p_max * r_template)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut partial_s = stream
            .alloc_zeros::<f64>(plan.partial_s_doubles)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut partial_r = stream
            .alloc_zeros::<f64>(plan.partial_r_doubles)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut status_dev = stream
            .alloc_zeros::<i32>(n)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // ----- Launch forward kernel: 1 block per row, P_MAX threads -----
        let cfg = LaunchConfig {
            grid_dim: (plan.blocks, 1, 1),
            block_dim: (plan.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let p_i32 = to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let r_i32 = to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ridge_arg = ridge_t;
        {
            let mut builder = stream.launch_builder(&forward);
            builder
                .arg(&d_dev)
                .arg(&b_dev)
                .arg(&g_dev)
                .arg(&n_i32)
                .arg(&p_i32)
                .arg(&r_i32)
                .arg(&ridge_arg)
                .arg(&mut l_out)
                .arg(&mut u_out)
                .arg(&mut y_out)
                .arg(&mut partial_s)
                .arg(&mut partial_r)
                .arg(&mut status_dev);
            // SAFETY: all buffers were just allocated on `stream` with sizes
            // derived from `plan`; kernel parameter list matches the
            // FORWARD_KERNEL_SOURCE signature.
            unsafe { builder.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        stream
            .synchronize()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // ----- Check per-block pivot status -----
        let status_host = stream
            .clone_dtoh(&status_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        if let Some(row) = status_host.iter().position(|s| *s != 0) {
            // The NVRTC kernel's status code is a 1-based pivot row index, not
            // a magnitude; size the bump from the block (Gershgorin λ_min
            // bound) so a strongly indefinite block recovers in one retry.
            return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                row,
                bump: super::ridge_bump_to_make_pd(sys.rows[row].htt.view(), ridge_t),
            });
        }

        // ----- Reduce partials on host into S_β and r_β -----
        let partial_s_host = stream
            .clone_dtoh(&partial_s)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let partial_r_host = stream
            .clone_dtoh(&partial_r)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut schur_host = vec![0.0_f64; k * k];
        for col in 0..k {
            for row in 0..k {
                let mut v = sys.hbb[[row, col]];
                if row == col {
                    v += ridge_beta;
                }
                schur_host[col * k + row] = v;
            }
        }
        let mut rhs_host: Vec<f64> = sys.gb.iter().map(|v| -v).collect();
        for i in 0..n {
            // partial_S[i] stride is R_TEMPLATE × R_TEMPLATE column-major; we
            // only read the leading (k × k) sub-block.
            let s_base = i * r_template * r_template;
            for col in 0..k {
                let col_base = s_base + col * r_template;
                let dst_col_base = col * k;
                for row in 0..k {
                    schur_host[dst_col_base + row] -= partial_s_host[col_base + row];
                }
            }
            let r_base = i * r_template;
            for a in 0..k {
                rhs_host[a] += partial_r_host[r_base + a];
            }
        }

        // ----- Factor S_β on device (cuSOLVER), solve for δβ -----
        let mut schur_dev = stream
            .clone_htod(&schur_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut rhs_dev = stream
            .clone_htod(&rhs_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let solver =
            DnHandle::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let info = potrf_single(&solver, &stream, k, &mut schur_dev)?;
        if info != 0 {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("fused Schur Cholesky failed at pivot {info}"),
            });
        }
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, false)?;
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, true)?;
        let delta_beta_host = stream
            .clone_dtoh(&rhs_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let delta_beta = Array1::from_vec(delta_beta_host.clone());

        // ----- Layer E: launch back-sub kernel using persisted L, u, Y -----
        let mut delta_t_dev = stream
            .alloc_zeros::<f64>(n * p_max)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let back_cfg = LaunchConfig {
            grid_dim: (plan.blocks, 1, 1),
            block_dim: (plan.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut builder = stream.launch_builder(&back_sub);
            builder
                .arg(&l_out)
                .arg(&u_out)
                .arg(&y_out)
                .arg(&rhs_dev)
                .arg(&n_i32)
                .arg(&p_i32)
                .arg(&r_i32)
                .arg(&mut delta_t_dev);
            // SAFETY: kernel parameter list matches FORWARD_KERNEL_SOURCE
            // back-sub signature; `rhs_dev` holds δβ in the leading k entries
            // (R_TEMPLATE-strided indexing is column 0..k of the R-vector).
            unsafe { builder.launch(back_cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        stream
            .synchronize()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        let delta_t_host = stream
            .clone_dtoh(&delta_t_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut delta_t = Array1::<f64>::zeros(n * d);
        for i in 0..n {
            let src_base = i * p_max;
            let dst_base = i * d;
            for r in 0..d {
                delta_t[dst_base + r] = delta_t_host[src_base + r];
            }
        }

        // ----- log|H| = 2·Σ log L_{i,jj} + 2·Σ log R_{β,aa} -----
        let l_local_host = stream
            .clone_dtoh(&l_out)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let l_schur_host = stream
            .clone_dtoh(&schur_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut log_det = 0.0_f64;
        for i in 0..n {
            let base = i * p_max * p_max;
            for j in 0..d {
                log_det += l_local_host[base + j * p_max + j].ln();
            }
        }
        for j in 0..k {
            log_det += l_schur_host[j * k + j].ln();
        }
        log_det *= 2.0;

        Ok(ArrowSchurGpuSolution {
            delta_t,
            delta_beta,
            log_det_hessian: log_det,
        })
    }

    /// Pre-compute `Y_i = L_i^{-1} H_tβ^(i)` via the fused forward kernel and
    /// return a closure that evaluates the full Schur matvec
    /// `S·x = (H_ββ + ρ·I)·x − Σ_i Y_i^T (Y_i·x)` for each PCG iteration.
    ///
    /// The `Y_i` factors are kept in a host-side buffer after one GPU forward
    /// pass. Each matvec call runs O(N·d·K) host loops over the pre-computed
    /// buffer plus an optional `H_ββ·x` call (matrix-free or dense). This is
    /// the first landing of the GPU matvec; a future iteration can move the
    /// `Y_i·x` / `Y_i^T z_i` steps to cuBLAS batched GEMV.
    pub(super) fn build_schur_matvec_backend(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<crate::arrow_schur::GpuSchurMatvec, super::ArrowSchurGpuFailure> {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let plan = crate::gpu_kernels::arrow_schur_nvrtc::plan_fused_launch(n, d, k)
            .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let p_max = plan.p_max;
        let r_template = plan.r_template;

        let runtime = gam_gpu::linalg_dispatch::route_through_gpu(
            gam_gpu::linalg_dispatch::DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n },
        )
        .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let cap = &runtime.device.capability;
        let key = crate::gpu_kernels::arrow_schur_nvrtc::FusedModuleCacheKey {
            cc_major: cap.compute_major,
            cc_minor: cap.compute_minor,
            p_max: p_max as u32,
            r_template: r_template as u32,
        };
        let module = fused_module_for(&ctx, key)?;
        let forward = module
            .load_function("arrow_schur_forward_pgroup")
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;

        let (d_host, b_host, g_host) = pack_fused_host(sys, ridge_t, p_max, r_template);
        let d_dev = stream
            .clone_htod(&d_host)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let b_dev = stream
            .clone_htod(&b_host)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let g_dev = stream
            .clone_htod(&g_host)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let mut l_out = stream
            .alloc_zeros::<f64>(n * p_max * p_max)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let mut u_out = stream
            .alloc_zeros::<f64>(n * p_max)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let mut y_out = stream
            .alloc_zeros::<f64>(n * p_max * r_template)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let mut partial_s = stream
            .alloc_zeros::<f64>(plan.partial_s_doubles)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let mut partial_r = stream
            .alloc_zeros::<f64>(plan.partial_r_doubles)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let mut status_dev = stream
            .alloc_zeros::<i32>(n)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;

        let cfg = LaunchConfig {
            grid_dim: (plan.blocks, 1, 1),
            block_dim: (plan.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = to_i32(n).ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let p_i32 = to_i32(d).ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let r_i32 = to_i32(k).ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let ridge_arg = ridge_t;
        {
            let mut builder = stream.launch_builder(&forward);
            builder
                .arg(&d_dev)
                .arg(&b_dev)
                .arg(&g_dev)
                .arg(&n_i32)
                .arg(&p_i32)
                .arg(&r_i32)
                .arg(&ridge_arg)
                .arg(&mut l_out)
                .arg(&mut u_out)
                .arg(&mut y_out)
                .arg(&mut partial_s)
                .arg(&mut partial_r)
                .arg(&mut status_dev);
            // SAFETY: all buffers were allocated on `stream` with sizes
            // derived from `plan`; parameter list matches FORWARD_KERNEL_SOURCE.
            unsafe { builder.launch(cfg) }.map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        }
        stream
            .synchronize()
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;

        let status_host = stream
            .clone_dtoh(&status_dev)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        if let Some(row) = status_host.iter().position(|s| *s != 0) {
            // Status code is a 1-based pivot row index, not a magnitude; size
            // the bump from the block (Gershgorin λ_min bound) so a strongly
            // indefinite block recovers in one retry.
            return Err(super::ArrowSchurGpuFailure::RidgeBumpRequired {
                row,
                bump: super::ridge_bump_to_make_pd(sys.rows[row].htt.view(), ridge_t),
            });
        }

        // Download Y_i factors: n × p_max × r_template column-major per block.
        let y_host = stream
            .clone_dtoh(&y_out)
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;

        // Capture H_ββ data for the closure. Use the matrix-free hook if present
        // (SAE-manifold callers), otherwise fall back to the dense matrix rows.
        let hbb_host: Vec<f64> = sys.hbb.iter().copied().collect();
        let hbb_is_kk = sys.hbb.dim() == (k, k);
        let hbb_matvec_opt = sys.hbb_matvec.clone();

        let closure: crate::arrow_schur::GpuSchurMatvec =
            Arc::new(move |x: &Array1<f64>, out: &mut Array1<f64>| {
                assert_eq!(x.len(), k, "gpu_schur_matvec: x.len() != k");
                assert_eq!(out.len(), k, "gpu_schur_matvec: out.len() != k");

                // (H_ββ + ρ·I)·x into out.
                if let Some(ref mv) = hbb_matvec_opt {
                    mv(x.view(), out);
                    for a in 0..k {
                        out[a] += ridge_beta * x[a];
                    }
                } else if hbb_is_kk {
                    // hbb_host row-major: hbb[a, b] = hbb_host[a * k + b].
                    for a in 0..k {
                        let mut acc = ridge_beta * x[a];
                        for b in 0..k {
                            acc += hbb_host[a * k + b] * x[b];
                        }
                        out[a] = acc;
                    }
                } else {
                    for a in 0..k {
                        out[a] = ridge_beta * x[a];
                    }
                }

                // out[c] -= Σ_i (Y_i^T (Y_i·x))[c].
                // Y_i column-major at y_host[i·p_max·r_template + col·p_max + row].
                let mut z = vec![0.0_f64; d];
                for i in 0..n {
                    let y_base = i * p_max * r_template;
                    for r in 0..d {
                        let mut acc = 0.0;
                        for c in 0..k {
                            acc += y_host[y_base + c * p_max + r] * x[c];
                        }
                        z[r] = acc;
                    }
                    for c in 0..k {
                        let mut acc = 0.0;
                        for r in 0..d {
                            acc += y_host[y_base + c * p_max + r] * z[r];
                        }
                        out[c] -= acc;
                    }
                }
            });

        Ok(closure)
    }

    // ── #1017/#1026 frames-engaged device PCG ──────────────────────────────

    struct DeviceSaeFrameBuffers {
        // Smooth `λ S_k ⊗ I_{r_k}`.
        s_off: CudaSlice<i32>,
        s_m: CudaSlice<i32>,
        s_r: CudaSlice<i32>,
        s_ptr: CudaSlice<i32>,
        s_data: CudaSlice<f64>,
        s_blocks: usize,
        // Data `G_{ij} ⊗ W_{ij}`.
        g_off_i: CudaSlice<i32>,
        g_off_j: CudaSlice<i32>,
        g_ri: CudaSlice<i32>,
        g_rj: CudaSlice<i32>,
        g_mi: CudaSlice<i32>,
        g_mj: CudaSlice<i32>,
        g_ptr: CudaSlice<i32>,
        g_data: CudaSlice<f64>,
        w_ptr: CudaSlice<i32>,
        w_data: CudaSlice<f64>,
        g_blocks: usize,
        g_max_work: usize,
        // Per-row dense cross-block H_tβ^(i) + row q + factored ainv.
        htb_ptr: CudaSlice<i32>,
        htb: CudaSlice<f64>,
        q_of: CudaSlice<i32>,
        ainv: CudaSlice<f64>,
        hvec: CudaSlice<f64>,
        svec: CudaSlice<f64>,
        // #1017 2-stage deterministic scatter scratch: partial[n_chunks × k]
        // holds each row-chunk's reduced-Schur contribution before the fixed-order
        // reduce. Ridge-independent shape (derived from n_rows), so it is allocated
        // once with the resident frame and reused across every apply.
        scatter_partial: CudaSlice<f64>,
        n_chunks: usize,
        rows_per_chunk: usize,
        n_rows: usize,
        k: usize,
        max_q: usize,
    }

    /// #1017 chunking for the 2-stage deterministic reduced-Schur scatter: split
    /// the `n_rows` row reduction into contiguous chunks so stage 1 launches
    /// `⌈k/256⌉·n_chunks` CTAs (vs the single-strip `⌈k/256⌉`). ~128 chunks fills a
    /// 72-SM A10 with several CTAs even at small `k`, while keeping the partial
    /// buffer (`n_chunks·k`) small and the stage-2 reduce (`n_chunks` adds) cheap.
    fn scatter_chunking(n_rows: usize) -> (usize, usize) {
        let target_chunks = 128usize;
        let rows_per_chunk = n_rows.div_ceil(target_chunks).max(1);
        let n_chunks = n_rows.div_ceil(rows_per_chunk).max(1);
        (rows_per_chunk, n_chunks)
    }

    fn flatten_device_sae_frame_data(
        sys: &ArrowSchurSystem,
        data: &DeviceSaePcgData,
        frame: &DeviceSaeFrameData,
        ridge_t: f64,
        stream: &Arc<CudaStream>,
    ) -> Result<DeviceSaeFrameBuffers, ArrowSchurGpuFailure> {
        // #1017: single-source the marshalling. The ridge-INDEPENDENT operands
        // (smooth `λ S_k`, framed `G ⊗ W`, dense per-row cross `H_tβ`) come from
        // `flatten_frame_host_operands`; only the ridge-DEPENDENT per-row `ainv`
        // is recomputed here. `upload_frame_buffers` performs the identical
        // host→device transfer the inline body used, so the resulting buffers are
        // byte-for-byte what this function produced before the split.
        let host = super::flatten_frame_host_operands(sys, data, frame)?;
        let ainv = super::compute_ainv_host(sys, &host.q_of, host.max_q, host.n_rows, ridge_t)?;
        upload_frame_buffers(&host, &ainv, stream)
    }

    /// Upload the marshalled host operands (ridge-independent) plus the supplied
    /// per-row `ainv` (ridge-dependent) into device buffers. Shared by the
    /// per-trial [`flatten_device_sae_frame_data`] and the resident-frame build
    /// (which uploads a zero `ainv` placeholder once and overwrites only `ainv`
    /// per ladder trial), so both paths marshal through one code path.
    fn upload_frame_buffers(
        host: &super::FrameHostOperands,
        ainv: &[f64],
        stream: &Arc<CudaStream>,
    ) -> Result<DeviceSaeFrameBuffers, ArrowSchurGpuFailure> {
        let htod_i = |v: &[i32]| {
            stream
                .clone_htod(v)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)
        };
        let htod_f = |v: &[f64]| {
            stream
                .clone_htod(v)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)
        };
        let (rows_per_chunk, n_chunks) = scatter_chunking(host.n_rows);
        Ok(DeviceSaeFrameBuffers {
            s_off: htod_i(&host.s_off)?,
            s_m: htod_i(&host.s_m)?,
            s_r: htod_i(&host.s_r)?,
            s_ptr: htod_i(&host.s_ptr)?,
            s_data: htod_f(&host.s_data)?,
            s_blocks: host.s_blocks,
            g_off_i: htod_i(&host.g_off_i)?,
            g_off_j: htod_i(&host.g_off_j)?,
            g_ri: htod_i(&host.g_ri)?,
            g_rj: htod_i(&host.g_rj)?,
            g_mi: htod_i(&host.g_mi)?,
            g_mj: htod_i(&host.g_mj)?,
            g_ptr: htod_i(&host.g_ptr)?,
            g_data: htod_f(&host.g_data)?,
            w_ptr: htod_i(&host.w_ptr)?,
            w_data: htod_f(&host.w_data)?,
            g_blocks: host.g_blocks,
            g_max_work: host.g_max_work,
            htb_ptr: htod_i(&host.htb_ptr)?,
            htb: htod_f(&host.htb)?,
            q_of: htod_i(&host.q_of)?,
            ainv: htod_f(ainv)?,
            hvec: stream
                .alloc_zeros::<f64>(host.n_rows * host.max_q)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            svec: stream
                .alloc_zeros::<f64>(host.n_rows * host.max_q)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            scatter_partial: stream
                .alloc_zeros::<f64>(n_chunks * host.k)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?,
            n_chunks,
            rows_per_chunk,
            n_rows: host.n_rows,
            k: host.k,
            max_q: host.max_q,
        })
    }

    /// Overwrite every ridge-independent resident operand in place. This is the
    /// accepted-nonlinear-iterate boundary: allocations persist, numerical
    /// content does not. A shape change declines so the caller replaces the
    /// complete frame rather than partially refreshing incompatible buffers.
    fn refresh_frame_buffers(
        host: &super::FrameHostOperands,
        buffers: &mut DeviceSaeFrameBuffers,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        macro_rules! refresh {
            ($host:expr, $device:expr) => {{
                if $host.len() != $device.len() {
                    return Err(ArrowSchurGpuFailure::Unavailable);
                }
                stream
                    .memcpy_htod($host, &mut $device)
                    .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            }};
        }
        if host.s_blocks != buffers.s_blocks
            || host.g_blocks != buffers.g_blocks
            || host.g_max_work != buffers.g_max_work
            || host.n_rows != buffers.n_rows
            || host.k != buffers.k
            || host.max_q != buffers.max_q
        {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        refresh!(&host.s_off, buffers.s_off);
        refresh!(&host.s_m, buffers.s_m);
        refresh!(&host.s_r, buffers.s_r);
        refresh!(&host.s_ptr, buffers.s_ptr);
        refresh!(&host.s_data, buffers.s_data);
        refresh!(&host.g_off_i, buffers.g_off_i);
        refresh!(&host.g_off_j, buffers.g_off_j);
        refresh!(&host.g_ri, buffers.g_ri);
        refresh!(&host.g_rj, buffers.g_rj);
        refresh!(&host.g_mi, buffers.g_mi);
        refresh!(&host.g_mj, buffers.g_mj);
        refresh!(&host.g_ptr, buffers.g_ptr);
        refresh!(&host.g_data, buffers.g_data);
        refresh!(&host.w_ptr, buffers.w_ptr);
        refresh!(&host.w_data, buffers.w_data);
        refresh!(&host.htb_ptr, buffers.htb_ptr);
        refresh!(&host.htb, buffers.htb);
        refresh!(&host.q_of, buffers.q_of);
        Ok(())
    }

    fn sae_frame_penalty_diag_host(
        data: &DeviceSaePcgData,
        frame: &DeviceSaeFrameData,
        ridge_beta: f64,
    ) -> Result<Vec<f64>, ArrowSchurGpuFailure> {
        let mut diag = vec![ridge_beta; data.beta_dim];
        // Smooth: diag[off + ia·r + ib] += S[ia,ia].
        for (blk, &r) in data.smooth_blocks.iter().zip(frame.smooth_ranks.iter()) {
            let m = blk.factor_a.nrows();
            for ia in 0..m {
                let coeff = blk.factor_a[[ia, ia]];
                let base = blk.global_offset + ia * r;
                for ib in 0..r {
                    if base + ib >= diag.len() {
                        return Err(ArrowSchurGpuFailure::Unavailable);
                    }
                    diag[base + ib] += coeff;
                }
            }
        }
        // Data: on-diagonal atom blocks contribute g[li,li]·w[a,a].
        for blk in &frame.frame_blocks {
            if blk.atom_i != blk.atom_j {
                continue;
            }
            let r = frame.ranks[blk.atom_i];
            let off = frame.border_offsets[blk.atom_i];
            let (mi, mj) = blk.g.dim();
            for li in 0..mi.min(mj) {
                let gii = blk.g[[li, li]];
                let base = off + li * r;
                for a in 0..r {
                    if base + a >= diag.len() {
                        return Err(ArrowSchurGpuFailure::Unavailable);
                    }
                    diag[base + a] += gii * blk.w[[a, a]];
                }
            }
        }
        Ok(diag)
    }

    fn frame_grid(work: usize, n_rows: usize) -> Result<LaunchConfig, ArrowSchurGpuFailure> {
        Ok(LaunchConfig {
            grid_dim: (
                ((work as u32).saturating_add(255) / 256).max(1),
                checked_i32(n_rows)? as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    fn launch_sae_frame_matvec(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        buffers: &mut DeviceSaeFrameBuffers,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
        ridge_beta: f64,
    ) -> Result<(), ArrowSchurGpuFailure> {
        launch_sae_init(stream, module, out, x, ridge_beta, buffers.k)?;
        // Smooth penalty.
        if buffers.s_blocks > 0 {
            let kernel = module
                .load_function("arrow_sae_frame_smooth_matvec")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let blocks_i32 = checked_i32(buffers.s_blocks)?;
            let cfg = frame_grid(buffers.k, buffers.s_blocks)?;
            let mut b = stream.launch_builder(&kernel);
            b.arg(x)
                .arg(&mut *out)
                .arg(&buffers.s_off)
                .arg(&buffers.s_m)
                .arg(&buffers.s_r)
                .arg(&buffers.s_ptr)
                .arg(&buffers.s_data)
                .arg(&blocks_i32);
            // SAFETY: smooth block metadata/data are live device buffers; the grid
            // covers (k channels × n_blocks) and the kernel bounds-checks m·r.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        // Data penalty.
        if buffers.g_blocks > 0 {
            let kernel = module
                .load_function("arrow_sae_frame_g_matvec")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let blocks_i32 = checked_i32(buffers.g_blocks)?;
            let cfg = frame_grid(buffers.g_max_work.max(1), buffers.g_blocks)?;
            let mut b = stream.launch_builder(&kernel);
            b.arg(x)
                .arg(&mut *out)
                .arg(&buffers.g_off_i)
                .arg(&buffers.g_off_j)
                .arg(&buffers.g_ri)
                .arg(&buffers.g_rj)
                .arg(&buffers.g_mi)
                .arg(&buffers.g_mj)
                .arg(&buffers.g_ptr)
                .arg(&buffers.g_data)
                .arg(&buffers.w_ptr)
                .arg(&buffers.w_data)
                .arg(&blocks_i32);
            // SAFETY: g/w block metadata/data are live device buffers; the grid
            // covers (max m_i·r_i × n_blocks) and the kernel bounds-checks.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        // Reduced-Schur subtraction via dense per-row cross-blocks.
        let k_i32 = checked_i32(buffers.k)?;
        let max_q_i32 = checked_i32(buffers.max_q)?;
        let n_rows_i32 = checked_i32(buffers.n_rows)?;
        {
            let kernel = module
                .load_function("arrow_sae_frame_apply_h")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let cfg = frame_grid(buffers.max_q, buffers.n_rows)?;
            let mut b = stream.launch_builder(&kernel);
            b.arg(x)
                .arg(&buffers.htb_ptr)
                .arg(&buffers.htb)
                .arg(&buffers.q_of)
                .arg(&mut buffers.hvec)
                .arg(&k_i32)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: dense cross-block + pointers + hvec are live buffers sized
            // for (n_rows × max_q) / (n_rows × k); kernel guards q_i and k.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        {
            let kernel = module
                .load_function("arrow_sae_frame_apply_ainv")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let cfg = frame_grid(buffers.max_q, buffers.n_rows)?;
            let mut b = stream.launch_builder(&kernel);
            b.arg(&buffers.ainv)
                .arg(&buffers.hvec)
                .arg(&buffers.q_of)
                .arg(&mut buffers.svec)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: ainv/hvec/svec are live buffers sized for n_rows·max_q²
            // and n_rows·max_q; the kernel guards row/coord bounds.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        {
            let kernel = module
                .load_function("arrow_sae_frame_scatter_h")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let cfg = frame_grid(buffers.k, buffers.n_rows)?;
            let mut b = stream.launch_builder(&kernel);
            b.arg(&buffers.svec)
                .arg(&buffers.htb_ptr)
                .arg(&buffers.htb)
                .arg(&buffers.q_of)
                .arg(out)
                .arg(&k_i32)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: svec/cross-block/out are live buffers; the kernel atomically
            // accumulates into out[a] for a<k and reads c<q_i.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    /// #1017 evidence lane: the DETERMINISTIC device reduced-Schur term
    /// `out[a] = -Σ_i H_βt^(i)(H_tt^(i)+ρ_t I)⁻¹H_tβ^(i)x`. Reuses the per-row
    /// `apply_h`→`apply_ainv` chain (each output written by a single thread — no
    /// cross-thread race) and the atomics-free `arrow_sae_frame_scatter_h_det`
    /// (one thread per output coord, fixed row order), so the value is run-to-run
    /// bit-stable. This is the reduced-Schur half of `S·x` ONLY; the penalty side
    /// `(P_ββ + ρ_β I)x` is added by the caller on the host via the already
    /// deterministic `sae_framed_penalty_matvec_cpu`. `out` is fully assigned.
    /// `ainv` must already be primed for the target `ρ_t` in `buffers`.
    fn launch_sae_frame_reduced_schur_det(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        buffers: &mut DeviceSaeFrameBuffers,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let k_i32 = checked_i32(buffers.k)?;
        let max_q_i32 = checked_i32(buffers.max_q)?;
        let n_rows_i32 = checked_i32(buffers.n_rows)?;
        // hvec[i][c] = Σ_a H_tβ[i][c,a]·x[a] — warp-cooperative (one warp per
        // (row, c), coalesced reads). Block = max_q·32 threads (one warp per
        // possible c), grid.x = n_rows.
        {
            let kernel = module
                .load_function("arrow_sae_frame_apply_h_warp")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let block = checked_i32(buffers.max_q)?
                .checked_mul(32)
                .ok_or(ArrowSchurGpuFailure::Unavailable)? as u32;
            let cfg = LaunchConfig {
                grid_dim: (checked_i32(buffers.n_rows)? as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&kernel);
            b.arg(x)
                .arg(&buffers.htb_ptr)
                .arg(&buffers.htb)
                .arg(&buffers.q_of)
                .arg(&mut buffers.hvec)
                .arg(&k_i32)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: dense cross-block + pointers + hvec are live buffers sized
            // for (n_rows × max_q) / (n_rows × k); block = max_q·32 ≤ 1024, each
            // warp guards `warp < q_i` and strides `a < k`.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        // svec[i] = ainv_i · hvec_i.
        {
            let kernel = module
                .load_function("arrow_sae_frame_apply_ainv")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let cfg = frame_grid(buffers.max_q, buffers.n_rows)?;
            let mut b = stream.launch_builder(&kernel);
            b.arg(&buffers.ainv)
                .arg(&buffers.hvec)
                .arg(&buffers.q_of)
                .arg(&mut buffers.svec)
                .arg(&max_q_i32)
                .arg(&n_rows_i32);
            // SAFETY: ainv/hvec/svec are live buffers sized for n_rows·max_q²
            // and n_rows·max_q; the kernel guards row/coord bounds.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        // out[a] = -Σ_i Σ_c H_tβ[i][c,a]·svec[i,c] — 2-stage DETERMINISTIC scatter.
        // Stage 1: partial[chunk][a] over contiguous row chunks, launching
        // ⌈k/256⌉·n_chunks CTAs (vs the single-strip ⌈k/256⌉ = 4 at k=911 that left
        // ~94% of the SMs idle). Rows summed in fixed order within each chunk.
        let k_blocks = ((buffers.k as u32).saturating_add(255) / 256).max(1);
        {
            let kernel = module
                .load_function("arrow_sae_frame_scatter_h_det_partial")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let rows_per_chunk_i32 = checked_i32(buffers.rows_per_chunk)?;
            let cfg = LaunchConfig {
                grid_dim: (k_blocks, checked_i32(buffers.n_chunks)? as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&kernel);
            b.arg(&buffers.svec)
                .arg(&buffers.htb_ptr)
                .arg(&buffers.htb)
                .arg(&buffers.q_of)
                .arg(&mut buffers.scatter_partial)
                .arg(&k_i32)
                .arg(&max_q_i32)
                .arg(&n_rows_i32)
                .arg(&rows_per_chunk_i32);
            // SAFETY: svec/cross-block are live buffers; scatter_partial is sized
            // n_chunks·k; each thread writes one in-bounds partial[chunk·k+a], a<k.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        // Stage 2: out[a] = -Σ_chunk partial[chunk][a], chunks reduced in fixed
        // order — the fixed reassociation that keeps the scatter deterministic.
        {
            let kernel = module
                .load_function("arrow_sae_frame_scatter_h_det_reduce")
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let n_chunks_i32 = checked_i32(buffers.n_chunks)?;
            let cfg = LaunchConfig {
                grid_dim: (k_blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&kernel);
            b.arg(&buffers.scatter_partial)
                .arg(&mut *out)
                .arg(&k_i32)
                .arg(&n_chunks_i32);
            // SAFETY: scatter_partial sized n_chunks·k, out sized k; one in-bounds
            // out[a] written per a<k.
            unsafe { b.launch(cfg) }.map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    fn launch_sae_frame_diag_sub(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        buffers: &DeviceSaeFrameBuffers,
        diag: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let kernel = module
            .load_function("arrow_sae_frame_diag_sub")
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let k_i32 = checked_i32(buffers.k)?;
        let max_q_i32 = checked_i32(buffers.max_q)?;
        let n_rows_i32 = checked_i32(buffers.n_rows)?;
        let cfg = frame_grid(buffers.k, buffers.n_rows)?;
        let mut b = stream.launch_builder(&kernel);
        b.arg(diag)
            .arg(&buffers.ainv)
            .arg(&buffers.htb_ptr)
            .arg(&buffers.htb)
            .arg(&buffers.q_of)
            .arg(&k_i32)
            .arg(&max_q_i32)
            .arg(&n_rows_i32);
        // SAFETY: diag + cross-block + ainv live buffers; kernel guards a<k, c/d<q.
        unsafe { b.launch(cfg) }
            .map(drop)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
    }

    /// #1551 kernel-isolating seam: evaluate the framed reduced-Schur matvec
    /// `out = S·x` EXACTLY ONCE on the device (no PCG, no offload-floor gate) and
    /// return `out`. This is the parity probe the test harness diffs against the
    /// CPU oracle [`super::sae_framed_schur_matvec_cpu`] element-by-element, so a
    /// kernel/marshalling defect is exposed directly — independent of how the
    /// iterative solver behaves on an ill-conditioned assembled `S` (where dense
    /// Cholesky and PCG legitimately disagree at the solution level). Declines
    /// (`Unavailable`) only when CUDA is genuinely absent so the test skips
    /// cleanly off-device; it deliberately does NOT consult the offload policy so
    /// even a tiny verifiable fixture runs on the GPU.
    pub(super) fn framed_schur_matvec_once_on_device(
        sys: &ArrowSchurSystem,
        data: &DeviceSaePcgData,
        ridge_t: f64,
        ridge_beta: f64,
        x: &Array1<f64>,
    ) -> Result<Array1<f64>, ArrowSchurGpuFailure> {
        let k = x.len();
        if k == 0 || data.beta_dim != k || sys.k != k {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        let frame = data
            .frame
            .as_ref()
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        // No offload-policy filter here: the seam exists to validate the kernel on
        // ANY device, including the smallest hand-checkable fixture.
        let runtime = super::resolve_runtime_for_device_path()?
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let vector_module = pcg_vector_module(&ctx)?;
        let mut buffers = flatten_device_sae_frame_data(sys, data, frame, ridge_t, &stream)?;
        let x_dev = stream
            .clone_htod(x.as_slice().ok_or(ArrowSchurGpuFailure::Unavailable)?)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut out_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        launch_sae_frame_matvec(
            &stream,
            vector_module,
            &mut buffers,
            &x_dev,
            &mut out_dev,
            ridge_beta,
        )?;
        let out = stream
            .clone_dtoh(&out_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        Ok(Array1::from_vec(out))
    }

    /// #1017 evidence-lane probe: the DETERMINISTIC framed reduced-Schur matvec
    /// `out = S·x` computed once (host penalty via `sae_framed_penalty_matvec_cpu`
    /// + the atomics-free device reduced-Schur term
    /// [`launch_sae_frame_reduced_schur_det`]). Mirrors
    /// [`framed_schur_matvec_once_on_device`] but produces a run-to-run bit-stable
    /// result, so the test harness uses it as BOTH the CPU-parity oracle
    /// comparison and the run-twice determinism probe. No PCG, no offload gate.
    pub(super) fn framed_reduced_schur_det_once_on_device(
        sys: &ArrowSchurSystem,
        data: &DeviceSaePcgData,
        ridge_t: f64,
        ridge_beta: f64,
        x: &Array1<f64>,
    ) -> Result<Array1<f64>, ArrowSchurGpuFailure> {
        let k = x.len();
        if k == 0 || data.beta_dim != k || sys.k != k {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        let frame = data
            .frame
            .as_ref()
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let runtime = super::resolve_runtime_for_device_path()?
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let vector_module = pcg_vector_module(&ctx)?;
        let mut buffers = flatten_device_sae_frame_data(sys, data, frame, ridge_t, &stream)?;
        let x_slice = x.as_slice().ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let x_dev = stream
            .clone_htod(x_slice)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut reduced_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        launch_sae_frame_reduced_schur_det(
            &stream,
            vector_module,
            &mut buffers,
            &x_dev,
            &mut reduced_dev,
        )?;
        let reduced = stream
            .clone_dtoh(&reduced_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        // out = (P_ββ + ρ_β I)x  (deterministic host penalty)  +  reduced (= -Σ term).
        let mut out = vec![0.0_f64; k];
        super::sae_framed_penalty_matvec_cpu(data, ridge_beta, x_slice, &mut out);
        for a in 0..k {
            out[a] += reduced[a];
        }
        Ok(Array1::from_vec(out))
    }

    pub(super) fn solve_sae_matrix_free_pcg_framed(
        sys: &ArrowSchurSystem,
        data: &DeviceSaePcgData,
        ridge_t: f64,
        ridge_beta: f64,
        rhs_beta: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
        let k = rhs_beta.len();
        if k == 0 || data.beta_dim != k || sys.k != k {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        let frame = data
            .frame
            .as_ref()
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let runtime = super::resolve_runtime_for_device_path()?
            .filter(|rt| {
                rt.policy().reduced_schur_matvec_should_offload(
                    sys.rows.len(),
                    sys.k,
                    sys.d,
                    max_iterations,
                )
            })
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let vector_module = pcg_vector_module(&ctx)?;
        // #1017/#2230 residency measurement: one line per matrix-free PCG solve
        // (hence per LM ridge-ladder trial) — operand bytes by category + the
        // ridge pair, so the a100 job (RUST_LOG=info) confirms the sub-lane and
        // sizes the per-trial re-upload a base-resident frame would remove.
        log::info!(
            "#1017/#2230 {} ridge_t={ridge_t:e} ridge_beta={ridge_beta:e}",
            data.operand_byte_report()
        );
        let mut buffers = flatten_device_sae_frame_data(sys, data, frame, ridge_t, &stream)?;
        let dctx = FramedPcgCtx {
            stream: &stream,
            blas: &blas,
            module: vector_module,
            max_iterations,
            relative_tolerance,
        };
        pcg_solve_framed_body(data, frame, &mut buffers, ridge_beta, rhs_beta, &dctx)
    }

    /// Device handles + CG controls for the framed PCG loop, bundled so the
    /// shared body stays under the argument-count lint without an `#[allow]`.
    struct FramedPcgCtx<'a> {
        stream: &'a Arc<CudaStream>,
        blas: &'a CudaBlas,
        module: &'a Arc<CudaModule>,
        max_iterations: usize,
        relative_tolerance: f64,
    }

    /// The framed reduced-Schur Jacobi-PCG loop over already-built device
    /// buffers. Factored out of [`solve_sae_matrix_free_pcg_framed`] so the
    /// resident-frame path ([`ResidentSaeFrameHandle::resolve`]) — which reuses
    /// the ridge-independent buffers and re-uploads only `ainv` — runs the
    /// byte-for-byte identical solve. Everything after the buffer build is
    /// unchanged from the inline body: penalty-diagonal Jacobi preconditioner,
    /// the CG recurrence, and the `Δβ` readback.
    fn pcg_solve_framed_body(
        data: &DeviceSaePcgData,
        frame: &DeviceSaeFrameData,
        buffers: &mut DeviceSaeFrameBuffers,
        ridge_beta: f64,
        rhs_beta: &Array1<f64>,
        dctx: &FramedPcgCtx<'_>,
    ) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
        let stream = dctx.stream;
        let blas = dctx.blas;
        let vector_module = dctx.module;
        let max_iterations = dctx.max_iterations;
        let relative_tolerance = dctx.relative_tolerance;
        let k = rhs_beta.len();
        let rhs_norm = rhs_beta.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rhs_norm == 0.0 {
            return Ok((Array1::<f64>::zeros(k), ArrowPcgDiagnostics::default()));
        }
        let tol = (relative_tolerance.max(0.0) * rhs_norm).max(1e-12);
        let rhs_dev = stream
            .clone_htod(
                rhs_beta
                    .as_slice()
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?,
            )
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let diag_host = sae_frame_penalty_diag_host(data, frame, ridge_beta)?;
        let mut diag_dev = stream
            .clone_htod(&diag_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        launch_sae_frame_diag_sub(stream, vector_module, buffers, &mut diag_dev)?;
        let diag_host = stream
            .clone_dtoh(&diag_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut inv_diag = Vec::with_capacity(k);
        for (idx, &d) in diag_host.iter().enumerate() {
            if !d.is_finite() || d <= 1.0e-18 {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!(
                        "framed SAE GPU PCG: non-positive Jacobi diagonal at {idx}: {d:e}"
                    ),
                });
            }
            inv_diag.push(1.0 / d);
        }
        let inv_diag_dev = stream
            .clone_htod(&inv_diag)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        let mut x_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut r_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        device_copy(blas, stream, k, &rhs_dev, &mut r_dev)?;
        let mut z_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        launch_jacobi_mul(stream, vector_module, &inv_diag_dev, &r_dev, &mut z_dev, k)?;
        let mut p_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        device_copy(blas, stream, k, &z_dev, &mut p_dev)?;
        let mut ap_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        let mut rz = device_dot(blas, stream, k, &r_dev, &z_dev)?;
        if rz <= 0.0 || !rz.is_finite() {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("framed SAE GPU PCG: non-positive initial rᵀM⁻¹r={rz:e}"),
            });
        }
        let mut diag = ArrowPcgDiagnostics {
            precond_apply_calls: 1,
            stopping_reason: PcgStopReason::MaxIter,
            ..ArrowPcgDiagnostics::default()
        };
        for _ in 0..max_iterations.max(1) {
            launch_sae_frame_matvec(
                stream,
                vector_module,
                buffers,
                &p_dev,
                &mut ap_dev,
                ridge_beta,
            )?;
            diag.matvec_calls += 1;
            diag.iterations += 1;
            let pap = device_dot(blas, stream, k, &p_dev, &ap_dev)?;
            if pap <= 0.0 || !pap.is_finite() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("framed SAE GPU PCG: non-positive curvature pᵀAp={pap:e}"),
                });
            }
            let alpha = rz / pap;
            device_axpy(blas, stream, k, alpha, &p_dev, &mut x_dev)?;
            device_axpy(blas, stream, k, -alpha, &ap_dev, &mut r_dev)?;
            let r_norm = device_nrm2(blas, stream, k, &r_dev)?;
            if r_norm <= tol {
                diag.final_relative_residual = r_norm / rhs_norm;
                diag.stopping_reason = PcgStopReason::Converged;
                break;
            }
            launch_jacobi_mul(stream, vector_module, &inv_diag_dev, &r_dev, &mut z_dev, k)?;
            diag.precond_apply_calls += 1;
            let rz_new = device_dot(blas, stream, k, &r_dev, &z_dev)?;
            if rz_new <= 0.0 || !rz_new.is_finite() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("framed SAE GPU PCG: non-positive rᵀM⁻¹r={rz_new:e}"),
                });
            }
            let beta = rz_new / rz;
            launch_update_p(stream, vector_module, &z_dev, beta, &mut p_dev, k)?;
            rz = rz_new;
        }
        if diag.stopping_reason != PcgStopReason::Converged {
            let r_norm = device_nrm2(blas, stream, k, &r_dev)?;
            diag.final_relative_residual = r_norm / rhs_norm;
            diag.stopping_reason = PcgStopReason::MaxIter;
        }
        let x = stream
            .clone_dtoh(&x_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        Ok((Array1::from_vec(x), diag))
    }

    /// #1017 device-resident framed SAE frame across the LM ridge ladder. Holds
    /// the ridge-INDEPENDENT device operand buffers (uploaded once) plus the CUDA
    /// context/stream they live on; each [`resolve`](Self::resolve) recomputes
    /// ONLY the ridge-dependent per-row `ainv` and re-runs the identical framed
    /// PCG. Persists only `Send + Sync` cudarc handles (`Arc<CudaContext>`,
    /// `Arc<CudaStream>`, `CudaSlice`); the cheap `CudaBlas`/module are re-derived
    /// per solve, so the whole handle is safely shareable through
    /// [`crate::arrow_schur::ArrowSolveOptions`].
    pub(crate) struct ResidentSaeFrameHandle {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        buffers: std::sync::Mutex<DeviceSaeFrameBuffers>,
        q_of: Vec<i32>,
        max_q: usize,
        n_rows: usize,
        k: usize,
    }

    impl ResidentSaeFrameHandle {
        /// Build the resident frame: gate exactly as
        /// [`solve_sae_matrix_free_pcg_framed`] (framed data present, offload
        /// predicate over the CG budget, live runtime), upload the
        /// ridge-independent operands once with a zero `ainv` placeholder, and
        /// stash the metadata needed to recompute `ainv` per trial. `None` on any
        /// decline keeps the per-trial re-flatten path.
        pub(crate) fn build(sys: &ArrowSchurSystem, cg_iters: usize) -> Option<Self> {
            let data = sys.device_sae_pcg.as_ref()?;
            let frame = data.frame.as_ref()?;
            if sys.k == 0 || data.beta_dim != sys.k {
                return None;
            }
            let runtime = super::resolve_runtime_for_device_path()
                .unwrap_or_else(|failure| {
                    panic!("resident SAE frame runtime resolution failed: {failure:?}")
                })
                .filter(|rt| {
                    rt.policy().reduced_schur_matvec_should_offload(
                        sys.rows.len(),
                        sys.k,
                        sys.d,
                        cg_iters,
                    )
                })?;
            let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)?;
            let stream = ctx.new_stream().ok()?;
            let host = super::flatten_frame_host_operands(sys, data, frame).ok()?;
            // #1017/#2230 residency measurement: the ridge-independent operand
            // bytes this frame uploads ONCE for the whole LM ridge ladder. The
            // per-trial flatten path re-uploaded this same total on EVERY trial
            // (its `#1017/#2230 …` info line fires once per trial); against a
            // ladder of `T` trials the resident frame removes `(T − 1) ×` this,
            // re-uploading only the per-row `ainv` (n_rows·max_q² f64) per trial.
            log::info!(
                "#1017 SAE resident frame ENGAGED: {} uploaded ONCE for the ladder; \
                 per-trial re-upload now only ainv ({}rows × {}²·8B)",
                data.operand_byte_report(),
                host.n_rows,
                host.max_q
            );
            let zero_ainv = vec![0.0_f64; host.n_rows * host.max_q * host.max_q];
            let buffers = upload_frame_buffers(&host, &zero_ainv, &stream).ok()?;
            Some(Self {
                ctx,
                stream,
                buffers: std::sync::Mutex::new(buffers),
                q_of: host.q_of,
                max_q: host.max_q,
                n_rows: host.n_rows,
                k: host.k,
            })
        }

        /// Overwrite the resident per-row `ainv` for a fixed `ridge_t` (the single
        /// evidence ridge), WITHOUT running the PCG — so an evidence matvec closure
        /// primes the factors once and every apply reuses them. Mirrors the
        /// ridge-dependent refresh in [`SaeResidentFrame::resolve`]. A genuinely
        /// non-PD row surfaces `RidgeBumpRequired`/`SchurFactorFailed` (the caller
        /// then declines to the CPU matvec, which handles the escalation).
        pub(super) fn prime_ainv(
            &self,
            sys: &ArrowSchurSystem,
            ridge_t: f64,
        ) -> Result<(), ArrowSchurGpuFailure> {
            if sys.k != self.k || sys.rows.len() != self.n_rows {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            let ainv =
                super::compute_ainv_host(sys, &self.q_of, self.max_q, self.n_rows, ridge_t)?;
            let mut buffers = self
                .buffers
                .lock()
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            if ainv.len() != buffers.ainv.len() {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            self.stream
                .memcpy_htod(&ainv, &mut buffers.ainv)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)
        }
    }

    impl super::SaeResidentFrame for ResidentSaeFrameHandle {
        fn refresh(&self, sys: &ArrowSchurSystem) -> Result<(), ArrowSchurGpuFailure> {
            if sys.k != self.k || sys.rows.len() != self.n_rows {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            let data = sys
                .device_sae_pcg
                .as_ref()
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let frame = data
                .frame
                .as_ref()
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let host = super::flatten_frame_host_operands(sys, data, frame)?;
            if host.q_of != self.q_of || host.max_q != self.max_q {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            let mut buffers = self
                .buffers
                .lock()
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            refresh_frame_buffers(&host, &mut buffers, &self.stream)
        }

        fn resolve(
            &self,
            sys: &ArrowSchurSystem,
            ridge_t: f64,
            ridge_beta: f64,
            rhs_beta: &Array1<f64>,
            max_iterations: usize,
            relative_tolerance: f64,
        ) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
            // The ladder holds `sys` fixed across trials; if any shape drifted
            // from build time the resident buffers no longer match — decline so
            // the caller retries via the per-trial flatten.
            if sys.k != self.k || sys.rows.len() != self.n_rows || rhs_beta.len() != self.k {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            let data = sys
                .device_sae_pcg
                .as_ref()
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let frame = data
                .frame
                .as_ref()
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            // Ridge-dependent buffer only: recompute per-row ainv and overwrite
            // the resident `ainv` slice in place (a genuine non-PD block still
            // surfaces `RidgeBumpRequired` for the LM escalation, unchanged).
            let ainv = super::compute_ainv_host(sys, &self.q_of, self.max_q, self.n_rows, ridge_t)?;
            let mut buffers = self
                .buffers
                .lock()
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            if ainv.len() != buffers.ainv.len() {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
            self.stream
                .memcpy_htod(&ainv, &mut buffers.ainv)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let blas = CudaBlas::new(self.stream.clone())
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let vector_module = pcg_vector_module(&self.ctx)?;
            let dctx = FramedPcgCtx {
                stream: &self.stream,
                blas: &blas,
                module: vector_module,
                max_iterations,
                relative_tolerance,
            };
            pcg_solve_framed_body(data, frame, &mut *buffers, ridge_beta, rhs_beta, &dctx)
        }
    }

    /// #1017 evidence lane: build a device-resident, RUN-TO-RUN DETERMINISTIC
    /// framed reduced-Schur `S·v` as the SLQ/surrogate `GpuSchurMatvec`. Uploads
    /// the ridge-independent framed operands ONCE ([`ResidentSaeFrameHandle`]),
    /// primes `ainv` at the single evidence `ridge_t`, and returns a closure that
    /// per apply crosses only `x` (down) and `out` (up): the deterministic host
    /// penalty ([`super::sae_framed_penalty_matvec_cpu`]) plus the atomics-free
    /// device reduced-Schur term ([`launch_sae_frame_reduced_schur_det`]). `None`
    /// on any decline (no device / shape / offload floor / non-PD at this ridge),
    /// so the caller keeps the CPU row-procedural matvec. A per-apply device fault
    /// after a validated build is a genuine bug/OOM and panics LOUD (the #1551
    /// no-silent-CPU discipline) rather than silently degrading the evidence.
    pub(super) fn build_framed_resident_evidence_matvec(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
        apply_budget: usize,
    ) -> Option<crate::arrow_schur::GpuSchurMatvec> {
        let handle = ResidentSaeFrameHandle::build(sys, apply_budget)?;
        handle.prime_ainv(sys, ridge_t).ok()?;
        let data = sys.device_sae_pcg.as_ref()?.clone();
        let handle = Arc::new(handle);
        let k = handle.k;
        let closure: crate::arrow_schur::GpuSchurMatvec =
            Arc::new(move |x: &Array1<f64>, out: &mut Array1<f64>| {
                assert_eq!(x.len(), k, "#1017 framed evidence matvec: x.len() != k");
                assert_eq!(out.len(), k, "#1017 framed evidence matvec: out.len() != k");
                let module = pcg_vector_module(&handle.ctx)
                    .expect("#1017 framed evidence matvec: pcg_vector_module unavailable");
                let x_slice = x
                    .as_slice()
                    .expect("#1017 framed evidence matvec: x not contiguous");
                let x_dev = handle
                    .stream
                    .clone_htod(x_slice)
                    .expect("#1017 framed evidence matvec: htod(x) failed");
                let mut reduced_dev = handle
                    .stream
                    .alloc_zeros::<f64>(k)
                    .expect("#1017 framed evidence matvec: device alloc failed");
                {
                    let mut buffers = handle
                        .buffers
                        .lock()
                        .expect("#1017 framed evidence matvec: resident frame poisoned");
                    launch_sae_frame_reduced_schur_det(
                        &handle.stream,
                        module,
                        &mut buffers,
                        &x_dev,
                        &mut reduced_dev,
                    )
                    .expect("#1017 framed evidence matvec: device reduced-Schur launch failed");
                }
                let reduced = handle
                    .stream
                    .clone_dtoh(&reduced_dev)
                    .expect("#1017 framed evidence matvec: dtoh(out) failed");
                let out_slice = out
                    .as_slice_mut()
                    .expect("#1017 framed evidence matvec: out not contiguous");
                super::sae_framed_penalty_matvec_cpu(&data, ridge_beta, x_slice, out_slice);
                for a in 0..k {
                    out_slice[a] += reduced[a];
                }
            });
        Some(closure)
    }

    /// #1551 stage-isolating triage seam: run the framed reduced-Schur matvec
    /// `out = S·x` ONCE on the device (no PCG, no offload-floor gate) and return
    /// `out`, so a tiny hand-verifiable fixture can diff it against the CPU oracle
    /// `sae_framed_schur_matvec_cpu` element-by-element to localize the structural
    /// divergence to a single kernel stage. Returns `Unavailable` only when CUDA
    /// is genuinely absent (so the test skips cleanly off-device).
    pub(super) fn solve_sae_matrix_free_pcg(
        sys: &ArrowSchurSystem,
        data: &DeviceSaePcgData,
        ridge_t: f64,
        ridge_beta: f64,
        rhs_beta: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
        let k = rhs_beta.len();
        if k == 0 || data.beta_dim != k || sys.k != k {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        // #1017/#1026 GUARD: the legacy `⊗ I_p` kernel must NEVER receive framed
        // data (factored `G ⊗ W_{ij}` + dense per-row cross blocks); decline so a
        // mis-route falls back to the CPU rather than returning a wrong step.
        if data.frame.is_some() {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        // #1017 Phase-1 dispatch re-key: this is the matrix-free SAE reduced-Schur
        // PCG — the production hot path, not a single dense factorization. The
        // dense-Direct floor `dense_hessian_work_target_is_gpu(n, k)` keys on
        // `2·n·k²` and is the WRONG gate here: it ignores the per-row frame depth
        // `d` (the M dimension that multiplies the per-apply work) and the
        // `1/cg_iters` staging amortisation, so it both undercounts the SAE batched
        // work `n·k·d` and applies a cold single-launch breakeven to an apply that
        // reuses device-resident frames `max_iterations` times. Key instead on the
        // CG-amortised total batched work — the same predicate the host injection
        // gate (`maybe_inject_gpu_schur_matvec`) consults — so few-row/wide-`k`/
        // modest-`d` LLM shapes register the real `n × k × d × cg_iters` arithmetic.
        // Kernels and numerics are untouched; only where the matvec runs changes,
        // and the host falls back to the bit-identical CPU matvec when this declines.
        let runtime = super::resolve_runtime_for_device_path()?
            .filter(|rt| {
                rt.policy().reduced_schur_matvec_should_offload(
                    sys.rows.len(),
                    sys.k,
                    sys.d,
                    max_iterations,
                )
            })
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let vector_module = pcg_vector_module(&ctx)?;
        // #1017/#2230 residency measurement (legacy sparse ⊗I_p lane): see the
        // framed twin — one line per solve/ladder-trial for the a100 job to size
        // the per-trial operand re-upload.
        log::info!(
            "#1017/#2230 {} ridge_t={ridge_t:e} ridge_beta={ridge_beta:e}",
            data.operand_byte_report()
        );
        let mut buffers = flatten_device_sae_data(sys, data, ridge_t, &stream)?;

        let rhs_norm = rhs_beta.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rhs_norm == 0.0 {
            return Ok((Array1::<f64>::zeros(k), ArrowPcgDiagnostics::default()));
        }
        let tol = (relative_tolerance.max(0.0) * rhs_norm).max(1e-12);
        let rhs_dev = stream
            .clone_htod(
                rhs_beta
                    .as_slice()
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?,
            )
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let diag_host = sae_penalty_diag_host(data, ridge_beta)?;
        let mut diag_dev = stream
            .clone_htod(&diag_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        launch_sae_diag_sub(&stream, vector_module, &buffers, &mut diag_dev)?;
        let diag_host = stream
            .clone_dtoh(&diag_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut inv_diag = Vec::with_capacity(k);
        for (idx, &d) in diag_host.iter().enumerate() {
            if !d.is_finite() || d <= 1.0e-18 {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!(
                        "SAE matrix-free GPU PCG: non-positive Schur Jacobi diagonal at {idx}: {d:e}"
                    ),
                });
            }
            inv_diag.push(1.0 / d);
        }
        let inv_diag_dev = stream
            .clone_htod(&inv_diag)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        let mut x_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut r_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        device_copy(&blas, &stream, k, &rhs_dev, &mut r_dev)?;
        let mut z_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        launch_jacobi_mul(&stream, vector_module, &inv_diag_dev, &r_dev, &mut z_dev, k)?;
        let mut p_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        device_copy(&blas, &stream, k, &z_dev, &mut p_dev)?;
        let mut ap_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        let mut rz = device_dot(&blas, &stream, k, &r_dev, &z_dev)?;
        if rz <= 0.0 || !rz.is_finite() {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("SAE matrix-free GPU PCG: non-positive initial rᵀM⁻¹r={rz:e}"),
            });
        }
        let mut diag = ArrowPcgDiagnostics {
            precond_apply_calls: 1,
            stopping_reason: PcgStopReason::MaxIter,
            ..ArrowPcgDiagnostics::default()
        };

        for _ in 0..max_iterations.max(1) {
            launch_sae_matvec(
                &stream,
                vector_module,
                &mut buffers,
                &p_dev,
                &mut ap_dev,
                ridge_beta,
            )?;
            diag.matvec_calls += 1;
            diag.iterations += 1;
            let pap = device_dot(&blas, &stream, k, &p_dev, &ap_dev)?;
            if pap <= 0.0 || !pap.is_finite() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("SAE matrix-free GPU PCG: non-positive curvature pᵀAp={pap:e}"),
                });
            }
            let alpha = rz / pap;
            device_axpy(&blas, &stream, k, alpha, &p_dev, &mut x_dev)?;
            device_axpy(&blas, &stream, k, -alpha, &ap_dev, &mut r_dev)?;
            let r_norm = device_nrm2(&blas, &stream, k, &r_dev)?;
            if r_norm <= tol {
                diag.final_relative_residual = r_norm / rhs_norm;
                diag.stopping_reason = PcgStopReason::Converged;
                break;
            }
            launch_jacobi_mul(&stream, vector_module, &inv_diag_dev, &r_dev, &mut z_dev, k)?;
            diag.precond_apply_calls += 1;
            let rz_new = device_dot(&blas, &stream, k, &r_dev, &z_dev)?;
            if rz_new <= 0.0 || !rz_new.is_finite() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("SAE matrix-free GPU PCG: non-positive rᵀM⁻¹r={rz_new:e}"),
                });
            }
            let beta = rz_new / rz;
            launch_update_p(&stream, vector_module, &z_dev, beta, &mut p_dev, k)?;
            rz = rz_new;
        }
        if diag.stopping_reason != PcgStopReason::Converged {
            let r_norm = device_nrm2(&blas, &stream, k, &r_dev)?;
            diag.final_relative_residual = r_norm / rhs_norm;
            diag.stopping_reason = PcgStopReason::MaxIter;
        }
        let x = stream
            .clone_dtoh(&x_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        Ok((Array1::from_vec(x), diag))
    }

    pub(super) fn solve_reduced_beta_pcg_with_diagnostics(
        s_acc: &ndarray::Array2<f64>,
        rhs_beta: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
        let k = rhs_beta.len();
        // #1017 dispatch re-key: this is an ITERATIVE device-resident PCG, not a
        // single GEMV. `S` (k×k) is uploaded once and reused for `max_iterations`
        // `S·p` GEMVs while only convergence scalars cross PCIe, so the staging
        // cost is amortised over the whole CG solve. Gating on the flops of ONE
        // `Gemv{k,k}` (`2·k²`) understates the work by the iteration count and
        // declines shapes (e.g. k≈512) whose total iterated arithmetic
        // `2·k²·iters` clears the device floor by orders of magnitude — the same
        // single-launch-breakeven miskey #1017 fixed for the framed reduced-Schur
        // matvec. Key on the CG-amortised total work via a `Gemm{k,k,iters}` whose
        // `flops()` is exactly `2·k²·iters`; numerics and kernels are untouched,
        // and the host falls back to the bit-identical CPU PCG when this declines.
        let cg_iters = max_iterations.max(1);
        let runtime = gam_gpu::linalg_dispatch::route_through_gpu(
            gam_gpu::linalg_dispatch::DispatchOp::Gemm {
                m: k,
                n: k,
                k: cg_iters,
            },
        )
        .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .and_then(|ctx| ctx.new_stream().ok())
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let vector_module = pcg_vector_module(&ctx)?;

        // Jacobi diagonal from S; must be strictly positive for SPD.
        let mut inv_diag = vec![0.0_f64; k];
        for j in 0..k {
            let djj = s_acc[[j, j]];
            if !(djj > 0.0) {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!(
                        "reduced-β GPU PCG: Jacobi diagonal S[{j},{j}]={djj:e} not positive"
                    ),
                });
            }
            inv_diag[j] = 1.0 / djj;
        }

        // Upload S column-major (S[row,col] at col*k + row).
        let mut s_host = vec![0.0_f64; k * k];
        for col in 0..k {
            for row in 0..k {
                s_host[col * k + row] = s_acc[[row, col]];
            }
        }
        let s_dev = stream
            .clone_htod(&s_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // Steihaug truncated-CG with Jacobi preconditioner, host scalar
        // recurrences and a device `S·p` matvec. The streaming reduced solve
        // uses an unbounded trust region (pure CG to tolerance).
        let rhs_norm = rhs_beta.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rhs_norm == 0.0 {
            return Ok((Array1::<f64>::zeros(k), ArrowPcgDiagnostics::default()));
        }
        let tol = (relative_tolerance.max(0.0) * rhs_norm).max(1e-12);

        // Device-resident PCG state. Only convergence scalars cross back during
        // the loop; x/r/z/p/Sp stay on CUDA until the final solution download.
        let mut x_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut r_dev = stream
            .clone_htod(
                rhs_beta
                    .as_slice()
                    .ok_or(ArrowSchurGpuFailure::Unavailable)?,
            )
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let inv_diag_dev = stream
            .clone_htod(&inv_diag)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut z_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        launch_jacobi_mul(&stream, vector_module, &inv_diag_dev, &r_dev, &mut z_dev, k)?;
        let mut p_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        device_copy(&blas, &stream, k, &z_dev, &mut p_dev)?;
        let mut sp_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut rz = device_dot(&blas, &stream, k, &r_dev, &z_dev)?;
        let mut diag = ArrowPcgDiagnostics {
            precond_apply_calls: 1,
            stopping_reason: PcgStopReason::MaxIter,
            ..ArrowPcgDiagnostics::default()
        };
        if rz <= 0.0 || !rz.is_finite() {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("reduced-β GPU PCG: non-positive initial rᵀM⁻¹r={rz:e}"),
            });
        }

        let max_iters = max_iterations.max(1);
        for _ in 0..max_iters {
            // sp = S · p (device GEMV, S column-major k×k, op = N).
            let gemv_cfg = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: 1.0,
                lda: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                incx: 1,
                beta: 0.0,
                incy: 1,
            };
            // SAFETY: s_dev is k×k column-major, p_dev / sp_dev length k.
            unsafe { blas.gemv(gemv_cfg, &s_dev, &p_dev, &mut sp_dev) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            diag.matvec_calls += 1;
            diag.iterations += 1;

            let p_sp = device_dot(&blas, &stream, k, &p_dev, &sp_dev)?;
            if !(p_sp > 0.0) {
                // Non-positive curvature on a (proximal-ridged) SPD system means
                // numerical breakdown; surface so the caller escalates.
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("reduced-β GPU PCG: non-positive curvature pᵀSp={p_sp:e}"),
                });
            }
            let alpha = rz / p_sp;
            device_axpy(&blas, &stream, k, alpha, &p_dev, &mut x_dev)?;
            device_axpy(&blas, &stream, k, -alpha, &sp_dev, &mut r_dev)?;
            let r_norm = device_nrm2(&blas, &stream, k, &r_dev)?;
            if r_norm <= tol {
                diag.final_relative_residual = r_norm / rhs_norm;
                diag.stopping_reason = PcgStopReason::Converged;
                break;
            }
            launch_jacobi_mul(&stream, vector_module, &inv_diag_dev, &r_dev, &mut z_dev, k)?;
            diag.precond_apply_calls += 1;
            let rz_new = device_dot(&blas, &stream, k, &r_dev, &z_dev)?;
            if rz_new <= 0.0 || !rz_new.is_finite() {
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("reduced-β GPU PCG: non-positive rᵀM⁻¹r={rz_new:e}"),
                });
            }
            let beta = rz_new / rz;
            launch_update_p(&stream, vector_module, &z_dev, beta, &mut p_dev, k)?;
            rz = rz_new;
        }
        if diag.stopping_reason != PcgStopReason::Converged {
            let r_norm = device_nrm2(&blas, &stream, k, &r_dev)?;
            diag.final_relative_residual = r_norm / rhs_norm;
            diag.stopping_reason = PcgStopReason::MaxIter;
        }

        let x = stream
            .clone_dtoh(&x_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        Ok((Array1::from_vec(x), diag))
    }

    fn device_copy(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        n: usize,
        src: &CudaSlice<f64>,
        dst: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let n_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let (src_ptr, _src_rec) = src.device_ptr(stream);
        let (dst_ptr, _dst_rec) = dst.device_ptr_mut(stream);
        // SAFETY: src and dst are live device allocations on this stream with at
        // least n contiguous f64 entries and unit stride.
        let status = unsafe {
            cudarc::cublas::sys::cublasDcopy_v2(
                *blas.handle(),
                n_i,
                src_ptr as *const f64,
                1,
                dst_ptr as *mut f64,
                1,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(ArrowSchurGpuFailure::Unavailable)
        }
    }

    fn device_axpy(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        n: usize,
        alpha: f64,
        x: &CudaSlice<f64>,
        y: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let n_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let (x_ptr, _x_rec) = x.device_ptr(stream);
        let (y_ptr, _y_rec) = y.device_ptr_mut(stream);
        // SAFETY: x and y are live device allocations on this stream with at
        // least n contiguous f64 entries and unit stride; cuBLAS only reads alpha.
        let status = unsafe {
            cudarc::cublas::sys::cublasDaxpy_v2(
                *blas.handle(),
                n_i,
                &alpha,
                x_ptr as *const f64,
                1,
                y_ptr as *mut f64,
                1,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(ArrowSchurGpuFailure::Unavailable)
        }
    }

    /// As [`device_axpy`] but with explicit strides, so a unit-stride source
    /// (e.g. an all-ones vector) can target a matrix diagonal with `incy = dim+1`.
    /// Used by [`ResidentBaseArrowFrame`] to add `ridge_beta` to the resident
    /// `k×k` Schur base on-device without re-uploading the border block.
    fn device_axpy_strided(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        n: usize,
        alpha: f64,
        x: &CudaSlice<f64>,
        incx: usize,
        y: &mut CudaSlice<f64>,
        incy: usize,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let n_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let incx_i = to_i32(incx).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let incy_i = to_i32(incy).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let (x_ptr, _x_rec) = x.device_ptr(stream);
        let (y_ptr, _y_rec) = y.device_ptr_mut(stream);
        // SAFETY: x spans ≥ 1+(n−1)·incx entries and y spans ≥ 1+(n−1)·incy
        // entries, both live on this stream; cuBLAS only reads alpha by pointer.
        let status = unsafe {
            cudarc::cublas::sys::cublasDaxpy_v2(
                *blas.handle(),
                n_i,
                &alpha,
                x_ptr as *const f64,
                incx_i,
                y_ptr as *mut f64,
                incy_i,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(ArrowSchurGpuFailure::Unavailable)
        }
    }

    fn device_dot(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        n: usize,
        x: &CudaSlice<f64>,
        y: &CudaSlice<f64>,
    ) -> Result<f64, ArrowSchurGpuFailure> {
        let n_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let (x_ptr, _x_rec) = x.device_ptr(stream);
        let (y_ptr, _y_rec) = y.device_ptr(stream);
        let mut result = 0.0_f64;
        // SAFETY: x and y are live device allocations on this stream with at
        // least n contiguous f64 entries and unit stride; result is a valid host
        // out-pointer for the cuBLAS scalar.
        let status = unsafe {
            cudarc::cublas::sys::cublasDdot_v2(
                *blas.handle(),
                n_i,
                x_ptr as *const f64,
                1,
                y_ptr as *const f64,
                1,
                &mut result,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(result)
        } else {
            Err(ArrowSchurGpuFailure::Unavailable)
        }
    }

    fn device_nrm2(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        n: usize,
        x: &CudaSlice<f64>,
    ) -> Result<f64, ArrowSchurGpuFailure> {
        let n_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let (x_ptr, _x_rec) = x.device_ptr(stream);
        let mut result = 0.0_f64;
        // SAFETY: x is a live device allocation on this stream with at least n
        // contiguous f64 entries and unit stride; result is a valid host
        // out-pointer for the cuBLAS scalar.
        let status = unsafe {
            cudarc::cublas::sys::cublasDnrm2_v2(
                *blas.handle(),
                n_i,
                x_ptr as *const f64,
                1,
                &mut result,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(result)
        } else {
            Err(ArrowSchurGpuFailure::Unavailable)
        }
    }

    #[cfg(test)]
    mod tests {
        //! #1551 device-side framed-matvec triage. Lives inside `mod cuda` so it
        //! can call the private kernel launchers directly (no test-only public
        //! seam, which the ban-scanner forbids). A bare `#[cfg(test)] mod tests`
        //! is the one form the scanner permits.
        use super::*;
        use crate::arrow_schur::{
            ArrowSchurSystem, DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock,
            FactoredFrameGBlock,
        };
        use ndarray::Array2;

        /// Build the tiny hand-verifiable framed SAE fixture (2 atoms, 2 rows).
        /// Shared by the resident-frame residency test below and mirrors the
        /// `#1551` matvec-triage fixture so both exercise the same operand shapes.
        fn tiny_framed_fixture() -> (ArrowSchurSystem, DeviceSaePcgData) {
            let p = 3usize;
            let ranks = vec![2usize, 3usize];
            let basis_sizes = vec![2usize, 2usize];
            let mut border_offsets = Vec::new();
            let mut acc = 0usize;
            for k in 0..2 {
                border_offsets.push(acc);
                acc += basis_sizes[k] * ranks[k];
            }
            let border_dim = acc;
            let frame_of = |k: usize| -> Array2<f64> {
                Array2::from_shape_fn((p, ranks[k]), |(i, j)| {
                    0.1 + 0.2 * ((i + 1) as f64) * ((j + 1 + 2 * k) as f64)
                })
            };
            let frames: Vec<Array2<f64>> = (0..2).map(frame_of).collect();
            let w_of = |i: usize, j: usize| -> Array2<f64> {
                let (ui, uj) = (&frames[i], &frames[j]);
                Array2::from_shape_fn((ranks[i], ranks[j]), |(a, b)| {
                    (0..p).map(|c| ui[[c, a]] * uj[[c, b]]).sum()
                })
            };
            let mut frame_blocks = Vec::new();
            for &(i, j) in &[(0usize, 0usize), (1usize, 1usize), (0, 1), (1, 0)] {
                let (mi, mj) = (basis_sizes[i], basis_sizes[j]);
                let mut g =
                    Array2::<f64>::from_shape_fn((mi, mj), |(r, c)| 0.1 * (r + 2 * c + 1) as f64);
                if i == j {
                    for r in 0..mi.min(mj) {
                        g[[r, r]] += mi as f64 + 2.0;
                    }
                }
                frame_blocks.push(FactoredFrameGBlock {
                    atom_i: i,
                    atom_j: j,
                    g,
                    w: w_of(i, j),
                });
            }
            let mut smooth_blocks = Vec::new();
            for k in 0..2 {
                let m = basis_sizes[k];
                let mut s =
                    Array2::<f64>::from_shape_fn((m, m), |(r, c)| 0.05 * (r + c + 1) as f64);
                for r in 0..m {
                    s[[r, r]] += 1.0;
                }
                smooth_blocks.push(DeviceSaeSmoothBlock {
                    global_offset: border_offsets[k],
                    factor_a: s,
                });
            }
            let smooth_ranks = ranks.clone();
            let n = 2usize;
            let q = 2usize;
            let mut sys = ArrowSchurSystem::new(n, q, border_dim);
            let mut row_htbeta = Vec::new();
            for i in 0..n {
                let mut htt =
                    Array2::<f64>::from_shape_fn((q, q), |(r, c)| 0.3 * (r + c + 1) as f64);
                for r in 0..q {
                    htt[[r, r]] += q as f64 + 2.0;
                }
                sys.rows[i].htt = htt;
                let mut slab = vec![0.0_f64; q * border_dim];
                for c in 0..q {
                    for col in 0..border_dim {
                        let v = 0.01 * ((c + 1) * (col + 1) + i) as f64;
                        slab[c * border_dim + col] = v;
                        sys.rows[i].htbeta[[c, col]] = v;
                    }
                }
                row_htbeta.push(slab);
            }
            let data = DeviceSaePcgData {
                p,
                beta_dim: border_dim,
                a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
                local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
                smooth_blocks,
                sparse_g_blocks: Vec::new(),
                frame: Some(DeviceSaeFrameData {
                    ranks,
                    basis_sizes,
                    border_offsets,
                    frame_blocks,
                    smooth_ranks,
                    row_htbeta,
                }),
            };
            (sys, data)
        }

        /// #1017 residency invariant (no GPU needed — pure host marshalling).
        ///
        /// Verifies the property the whole resident-frame optimization rests on:
        /// across the LM ridge ladder the framed SAE operands split into a
        /// ridge-INDEPENDENT part (`flatten_frame_host_operands` — which takes no
        /// `ridge_t` at all, a compile-time proof, and is deterministic build to
        /// build) and a single ridge-DEPENDENT buffer, the per-row factored
        /// `ainv` (`compute_ainv_host`). So the resident frame can upload the
        /// ridge-independent operands once and recompute only `ainv` per trial,
        /// removing `(trials − 1) × operand_bytes` of re-upload with a
        /// bit-identical solve.
        #[test]
        fn sae_resident_frame_only_ainv_is_ridge_dependent_1017() {
            let (sys, data) = tiny_framed_fixture();
            let frame = data.frame.as_ref().expect("framed fixture");

            // Ridge-INDEPENDENT operands: identical across builds (deterministic;
            // no `ridge_t` in the signature).
            let host_a = super::super::flatten_frame_host_operands(&sys, &data, frame)
                .expect("host operands a");
            let host_b = super::super::flatten_frame_host_operands(&sys, &data, frame)
                .expect("host operands b");
            assert_eq!(
                host_a.s_data, host_b.s_data,
                "smooth λS must be ridge-independent"
            );
            assert_eq!(
                host_a.g_data, host_b.g_data,
                "frame G must be ridge-independent"
            );
            assert_eq!(
                host_a.w_data, host_b.w_data,
                "frame W must be ridge-independent"
            );
            assert_eq!(host_a.htb, host_b.htb, "row H_tβ must be ridge-independent");
            assert_eq!(host_a.q_of, host_b.q_of);

            let report = data.operand_byte_report();
            assert!(report.framed, "fixture must exercise the framed lane");
            assert!(
                report.total_bytes > 0,
                "framed operands must have nonzero bytes"
            );

            // The ONLY ridge-dependent buffer: ainv. Deterministic at a fixed
            // ridge (safe to reuse across the ladder), changing with ridge_t
            // (must be recomputed each trial).
            let ainv_lo = super::super::compute_ainv_host(
                &sys,
                &host_a.q_of,
                host_a.max_q,
                host_a.n_rows,
                1e-3,
            )
            .expect("ainv lo");
            let ainv_lo2 = super::super::compute_ainv_host(
                &sys,
                &host_a.q_of,
                host_a.max_q,
                host_a.n_rows,
                1e-3,
            )
            .expect("ainv lo repeat");
            let ainv_hi = super::super::compute_ainv_host(
                &sys,
                &host_a.q_of,
                host_a.max_q,
                host_a.n_rows,
                1e3,
            )
            .expect("ainv hi");
            assert_eq!(
                ainv_lo, ainv_lo2,
                "ainv must be deterministic at a fixed ridge"
            );
            let max_diff = ainv_lo
                .iter()
                .zip(&ainv_hi)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff > 1e-6,
                "ainv MUST change with ridge_t (else the split is wrong); max_diff={max_diff:e}"
            );

            // Ladder projection: the per-trial flatten re-uploaded `total_bytes`
            // on every trial; the resident frame uploads it once, so a ladder of
            // `trials` removes `(trials − 1) × total_bytes`.
            let trials = crate::arrow_schur::DEFAULT_PROXIMAL_MAX_ATTEMPTS + 1;
            let saved = report.total_bytes * (trials - 1);
            assert!(saved > 0);
            eprintln!(
                "#1017 resident-frame ladder saving: {trials} trials × {}B ridge-independent \
                 operand upload → resident removes {saved}B, re-uploading only ainv \
                 ({}rows × {}² × 8B) per trial",
                report.total_bytes, host_a.n_rows, host_a.max_q
            );
        }

        /// Run the framed reduced-Schur matvec `out = S·x` ONCE on the device
        /// (no PCG, no offload gate) and return `out`.
        fn device_matvec_once(
            sys: &ArrowSchurSystem,
            data: &DeviceSaePcgData,
            ridge_t: f64,
            ridge_beta: f64,
            x_host: &[f64],
        ) -> Result<Vec<f64>, ArrowSchurGpuFailure> {
            let k = x_host.len();
            let frame = data
                .frame
                .as_ref()
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let runtime = super::resolve_runtime_for_device_path()?
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
                .ok_or(ArrowSchurGpuFailure::Unavailable)?;
            let stream = ctx
                .new_stream()
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let vector_module = pcg_vector_module(&ctx)?;
            let mut buffers = flatten_device_sae_frame_data(sys, data, frame, ridge_t, &stream)?;
            let x_dev = stream
                .clone_htod(x_host)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            let mut out_dev = stream
                .alloc_zeros::<f64>(k)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            launch_sae_frame_matvec(
                &stream,
                vector_module,
                &mut buffers,
                &x_dev,
                &mut out_dev,
                ridge_beta,
            )?;
            stream
                .clone_dtoh(&out_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)
        }

        /// #1551 stage-isolating matvec triage on a TINY hand-verifiable fixture:
        /// diff the device framed matvec `S·e_col` against the CPU oracle
        /// `sae_framed_schur_matvec_cpu` for every identity column, reporting the
        /// worst-divergent border index so the structural 91% localizes to one
        /// kernel stage. Skips cleanly off-device.
        #[test]
        fn framed_sae_device_matvec_stage_diff_tiny_1551() {
            if gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
                .unwrap_or_else(|error| panic!("GPU probe fault in framed matvec test: {error}"))
                .is_none()
            {
                return;
            }
            let p = 3usize;
            let ranks = vec![2usize, 3usize];
            let basis_sizes = vec![2usize, 2usize];
            let mut border_offsets = Vec::new();
            let mut acc = 0usize;
            for k in 0..2 {
                border_offsets.push(acc);
                acc += basis_sizes[k] * ranks[k];
            }
            let border_dim = acc; // 2·2 + 2·3 = 10
            let frame_of = |k: usize| -> Array2<f64> {
                Array2::from_shape_fn((p, ranks[k]), |(i, j)| {
                    0.1 + 0.2 * ((i + 1) as f64) * ((j + 1 + 2 * k) as f64)
                })
            };
            let frames: Vec<Array2<f64>> = (0..2).map(frame_of).collect();
            let w_of = |i: usize, j: usize| -> Array2<f64> {
                let (ui, uj) = (&frames[i], &frames[j]);
                Array2::from_shape_fn((ranks[i], ranks[j]), |(a, b)| {
                    (0..p).map(|c| ui[[c, a]] * uj[[c, b]]).sum()
                })
            };
            let mut frame_blocks = Vec::new();
            for &(i, j) in &[(0usize, 0usize), (1usize, 1usize), (0, 1), (1, 0)] {
                let (mi, mj) = (basis_sizes[i], basis_sizes[j]);
                let mut g =
                    Array2::<f64>::from_shape_fn((mi, mj), |(r, c)| 0.1 * (r + 2 * c + 1) as f64);
                if i == j {
                    for r in 0..mi.min(mj) {
                        g[[r, r]] += mi as f64 + 2.0;
                    }
                }
                frame_blocks.push(FactoredFrameGBlock {
                    atom_i: i,
                    atom_j: j,
                    g,
                    w: w_of(i, j),
                });
            }
            let mut smooth_blocks = Vec::new();
            for k in 0..2 {
                let m = basis_sizes[k];
                let mut s =
                    Array2::<f64>::from_shape_fn((m, m), |(r, c)| 0.05 * (r + c + 1) as f64);
                for r in 0..m {
                    s[[r, r]] += 1.0;
                }
                smooth_blocks.push(DeviceSaeSmoothBlock {
                    global_offset: border_offsets[k],
                    factor_a: s,
                });
            }
            let smooth_ranks = ranks.clone();
            let n = 2usize;
            let q = 2usize;
            let mut sys = ArrowSchurSystem::new(n, q, border_dim);
            let mut row_htbeta = Vec::new();
            for i in 0..n {
                let mut htt =
                    Array2::<f64>::from_shape_fn((q, q), |(r, c)| 0.3 * (r + c + 1) as f64);
                for r in 0..q {
                    htt[[r, r]] += q as f64 + 2.0;
                }
                sys.rows[i].htt = htt;
                let mut slab = vec![0.0_f64; q * border_dim];
                for c in 0..q {
                    for col in 0..border_dim {
                        let v = 0.01 * ((c + 1) * (col + 1) + i) as f64;
                        slab[c * border_dim + col] = v;
                        sys.rows[i].htbeta[[c, col]] = v;
                    }
                }
                row_htbeta.push(slab);
            }
            let data = DeviceSaePcgData {
                p,
                beta_dim: border_dim,
                a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
                local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
                smooth_blocks,
                sparse_g_blocks: Vec::new(),
                frame: Some(DeviceSaeFrameData {
                    ranks,
                    basis_sizes,
                    border_offsets,
                    frame_blocks,
                    smooth_ranks,
                    row_htbeta,
                }),
            };
            let ridge_t = 1e-7;
            let ridge_beta = 1e-6;
            let mut first_bad: Option<usize> = None;
            let mut worst = 0.0_f64;
            let mut worst_at = 0usize;
            let mut worst_dev = 0.0_f64;
            let mut worst_cpu = 0.0_f64;
            for col in 0..border_dim {
                let mut x = vec![0.0_f64; border_dim];
                x[col] = 1.0;
                let dev = match device_matvec_once(&sys, &data, ridge_t, ridge_beta, &x) {
                    Ok(v) => v,
                    Err(_) => return,
                };
                let mut cpu = vec![0.0_f64; border_dim];
                super::super::sae_framed_schur_matvec_cpu(
                    &sys, &data, ridge_t, ridge_beta, &x, &mut cpu,
                )
                .expect("cpu matvec");
                for r in 0..border_dim {
                    let d = (dev[r] - cpu[r]).abs();
                    if d > 1e-9 && first_bad.is_none() {
                        first_bad = Some(r * border_dim + col);
                    }
                    if d > worst {
                        worst = d;
                        worst_at = r * border_dim + col;
                        worst_dev = dev[r];
                        worst_cpu = cpu[r];
                    }
                }
            }
            assert!(
                worst <= 1e-9,
                "[#1551 stage-diff] device framed matvec != CPU oracle: worst abs={worst:e} at \
                 (row*K+col)={worst_at} (dev={worst_dev:e} cpu={worst_cpu:e}), \
                 first_bad_idx={first_bad:?}; border layout: atom0 [0..4) rank2, atom1 [4..10) \
                 rank3 — which atom-range the bad row/col falls in pins the stage (smooth=diag, \
                 G⊗W=cross, reduced-Schur=dense per-row)",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_schur::ArrowSchurSystem;
    use ndarray::{Array2, ArrayView1};

    fn build_fixture(n: usize, d: usize, k: usize, seed: u64) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(n, d, k);
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut sample = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };
        for row in &mut sys.rows {
            let mut a = Array2::<f64>::zeros((d, d));
            for r in 0..d {
                for c in 0..d {
                    a[[r, c]] = sample();
                }
            }
            let mut htt = a.t().dot(&a);
            for r in 0..d {
                htt[[r, r]] += d as f64 + 1.0;
            }
            row.htt = htt;
            for r in 0..d {
                for c in 0..k {
                    row.htbeta[[r, c]] = 0.1 * sample();
                }
                row.gt[r] = sample();
            }
        }
        let mut hbb_a = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                hbb_a[[r, c]] = sample();
            }
        }
        let mut hbb = hbb_a.t().dot(&hbb_a);
        for r in 0..k {
            hbb[[r, r]] += k as f64 + 1.0;
        }
        sys.hbb = hbb;
        for r in 0..k {
            sys.gb[r] = sample();
        }
        sys
    }

    /// The Gershgorin ridge bump must actually make a known-indefinite block PD
    /// on the first retry — the whole point of #1711. Verified directly by
    /// re-factoring `H_tt + (ridge_t + bump)·I` with the same Cholesky guard the
    /// device readback uses, for blocks whose `λ_min` is known in closed form.
    #[test]
    fn ridge_bump_makes_known_indefinite_blocks_pd() {
        // `cholesky_factor_in_place` / `CholeskyGuard` are already in scope via
        // `super::*` (imported at the top of the module).
        // A few blocks with a CLOSED-FORM smallest eigenvalue, all at ridge_t=0.
        // (label, matrix, λ_min) — the bump must clear each one.
        let neg_identity = Array2::<f64>::from_diag(&Array1::from_elem(8, -1.0)); // λ_min = -1
        let scaled_neg = Array2::<f64>::from_diag(&Array1::from_elem(4, -250.0)); // λ_min = -250
        // Symmetric 2×2 [[1, 2], [2, 1]] has eigenvalues 3 and -1 → indefinite.
        let mut indef2 = Array2::<f64>::zeros((2, 2));
        indef2[[0, 0]] = 1.0;
        indef2[[1, 1]] = 1.0;
        indef2[[0, 1]] = 2.0;
        indef2[[1, 0]] = 2.0;
        // A genuinely PD block must get a bump that is the bare rounding margin
        // only (deficit 0), and must still factor — the helper is defensive.
        let pd = Array2::<f64>::from_diag(&Array1::from_elem(3, 5.0));

        for (label, block) in [
            ("-I (λ_min=-1)", neg_identity),
            ("-250·I (λ_min=-250)", scaled_neg),
            ("[[1,2],[2,1]] (λ_min=-1)", indef2),
            ("5·I (PD)", pd),
        ] {
            let ridge_t = 0.0;
            let bump = ridge_bump_to_make_pd(block.view(), ridge_t);
            assert!(
                bump > 0.0 && bump.is_finite(),
                "[{label}] bump must be strictly positive and finite, got {bump:e}"
            );
            let d = block.nrows();
            let mut shifted = block.clone();
            for i in 0..d {
                shifted[[i, i]] += ridge_t + bump;
            }
            assert!(
                cholesky_factor_in_place(shifted.view(), CholeskyGuard::NonnegativePivot).is_some(),
                "[{label}] H_tt + (ridge_t + bump={bump:e})·I must be PD after the \
                 Gershgorin bump, but the Cholesky still rejected it"
            );
        }
    }

    /// The column-major variant (multi-GPU tile path) must agree with the
    /// row-major helper for a symmetric block, since Gershgorin edges are
    /// invariant under reading the symmetric matrix by row vs by column. The
    /// colmajor variant takes the bound at ridge_t=0 (the ridge is already baked
    /// into the diagonal it reads), so compare against `ridge_bump_to_make_pd`
    /// with `ridge_t = 0`.
    ///
    /// Gated to linux: `ridge_bump_to_make_pd_colmajor` only exists on the
    /// linux CUDA tile path, so the parity test runs where the function does.
    #[cfg(target_os = "linux")]
    #[test]
    fn ridge_bump_colmajor_matches_rowmajor_for_symmetric_block() {
        // Symmetric 3×3 with a negative-definite-ish diagonal and off-diagonals.
        let mut a = Array2::<f64>::zeros((3, 3));
        a[[0, 0]] = -2.0;
        a[[1, 1]] = 0.5;
        a[[2, 2]] = 1.0;
        a[[0, 1]] = 0.3;
        a[[1, 0]] = 0.3;
        a[[1, 2]] = -0.4;
        a[[2, 1]] = -0.4;
        a[[0, 2]] = 0.1;
        a[[2, 0]] = 0.1;

        let row_major_bump = ridge_bump_to_make_pd(a.view(), 0.0);

        // Flatten column-major: block[c*d + r] = a[[r, c]].
        let d = 3;
        let mut col_major = vec![0.0_f64; d * d];
        for c in 0..d {
            for r in 0..d {
                col_major[c * d + r] = a[[r, c]];
            }
        }
        let col_major_bump = ridge_bump_to_make_pd_colmajor(&col_major, d);

        assert!(
            (row_major_bump - col_major_bump).abs() <= 1e-12 * row_major_bump.max(1.0),
            "colmajor bump {col_major_bump:e} must match rowmajor bump \
             {row_major_bump:e} for a symmetric block"
        );

        // And the bump must actually make it PD (sanity, same as the row-major test).
        let mut shifted = a.clone();
        for i in 0..d {
            shifted[[i, i]] += col_major_bump;
        }
        assert!(
            cholesky_factor_in_place(shifted.view(), CholeskyGuard::NonnegativePivot).is_some(),
            "colmajor Gershgorin bump must make the symmetric block PD"
        );
    }

    fn device_pcg_fixture(k: usize) -> (Array2<f64>, Array1<f64>) {
        let mut s = Array2::<f64>::zeros((k, k));
        for row in 0..k {
            s[[row, row]] = 2.5 + 0.001 * ((row % 17) as f64);
            if row + 1 < k {
                s[[row, row + 1]] = -0.05;
                s[[row + 1, row]] = -0.05;
            }
            if row + 7 < k {
                s[[row, row + 7]] = 0.01;
                s[[row + 7, row]] = 0.01;
            }
        }
        let rhs = Array1::from_shape_fn(k, |idx| ((idx as f64 + 1.0) * 0.013).sin());
        (s, rhs)
    }

    fn dense_pcg_cpu_reference(
        s: &Array2<f64>,
        rhs: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Array1<f64> {
        let k = rhs.len();
        let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rhs_norm == 0.0 {
            return Array1::<f64>::zeros(k);
        }
        let tol = (relative_tolerance.max(0.0) * rhs_norm).max(1e-12);
        let inv_diag: Vec<f64> = (0..k).map(|idx| 1.0 / s[[idx, idx]]).collect();
        let mut x = Array1::<f64>::zeros(k);
        let mut r = rhs.clone();
        let mut z = Array1::from_shape_fn(k, |idx| inv_diag[idx] * r[idx]);
        let mut p = z.clone();
        let mut sp = Array1::<f64>::zeros(k);
        let mut rz = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
        for _ in 0..max_iterations.max(1) {
            for row in 0..k {
                let mut acc = 0.0;
                for col in 0..k {
                    acc += s[[row, col]] * p[col];
                }
                sp[row] = acc;
            }
            let p_sp = p.iter().zip(sp.iter()).map(|(a, b)| a * b).sum::<f64>();
            let alpha = rz / p_sp;
            for idx in 0..k {
                x[idx] += alpha * p[idx];
                r[idx] -= alpha * sp[idx];
            }
            let r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= tol {
                break;
            }
            for idx in 0..k {
                z[idx] = inv_diag[idx] * r[idx];
            }
            let rz_next = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>();
            let beta = rz_next / rz;
            for idx in 0..k {
                p[idx] = z[idx] + beta * p[idx];
            }
            rz = rz_next;
        }
        x
    }

    #[test]
    fn device_resident_pcg_matches_cpu_reference_when_cuda_admits() {
        let (s, rhs) = device_pcg_fixture(512);
        let max_iterations = 200usize;
        let relative_tolerance = 1.0e-12;
        let cpu = dense_pcg_cpu_reference(&s, &rhs, max_iterations, relative_tolerance);
        let (device, diag) = match solve_reduced_beta_pcg_with_diagnostics(
            &s,
            &rhs,
            max_iterations,
            relative_tolerance,
        ) {
            Ok(result) => result,
            // #1017 — fail loud, never skip-pass: this fixture clears the device
            // offload floor, so a CUDA device that is PRESENT yet declines/returns
            // Err means the device PCG kernel does not run on GPU (a real fault that
            // must not masquerade as a pass via this skip). Legit skip ONLY when no
            // usable CUDA device exists (CPU CI). The exact `ArrowSchurGpuFailure`
            // variant is folded into the assert message as the diagnostic.
            Err(failure) => {
                assert!(
                    gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
                        .unwrap_or_else(|error| {
                            panic!("GPU probe fault in reduced-beta PCG test: {error}")
                        })
                        .is_none(),
                    "#1017: CUDA device present but the device reduced-beta PCG \
                     declined/faulted instead of returning a result (tag: {failure:?}) — \
                     the kernel does not run correctly on GPU"
                );
                return;
            }
        };
        let max_err = cpu
            .iter()
            .zip(device.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err <= 1.0e-10,
            "device resident PCG parity failed: max_err={max_err:e}, diag={diag:?}"
        );
        assert!(diag.matvec_calls > 0);
        assert_eq!(diag.matvec_calls, diag.iterations);
    }

    #[test]
    fn dense_reference_matches_independent_solve() {
        let sys = build_fixture(4, 5, 3, 7);
        let solution = solve_arrow_newton_step_dense_reference(&sys, 0.0, 0.0).unwrap();
        // Re-solve by an independent matrix build and a textbook
        // Gaussian-elimination Cholesky to guard against typos in the
        // reference implementation itself.
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let total = n * d + k;
        let mut h = Array2::<f64>::zeros((total, total));
        let mut g = ndarray::Array1::<f64>::zeros(total);
        for (i, row) in sys.rows.iter().enumerate() {
            let base = i * d;
            for c in 0..d {
                for r in 0..d {
                    h[[base + r, base + c]] = row.htt[[r, c]];
                }
            }
            for c in 0..k {
                for r in 0..d {
                    h[[base + r, n * d + c]] = row.htbeta[[r, c]];
                    h[[n * d + c, base + r]] = row.htbeta[[r, c]];
                }
            }
            for r in 0..d {
                g[base + r] = row.gt[r];
            }
        }
        for c in 0..k {
            for r in 0..k {
                h[[n * d + r, n * d + c]] += sys.hbb[[r, c]];
            }
            g[n * d + c] = sys.gb[c];
        }
        let l = cholesky_factor_in_place(h.view(), CholeskyGuard::NonnegativePivot).unwrap();
        let rhs = g.mapv(|v| -v);
        let expected = cholesky_solve_vector(l.view(), rhs.view());
        for i in 0..n * d {
            assert!(
                (solution.delta_t[i] - expected[i]).abs() < 1e-10 * (1.0 + expected[i].abs()),
                "delta_t[{i}] mismatch: got {} expected {}",
                solution.delta_t[i],
                expected[i]
            );
        }
        for a in 0..k {
            assert!(
                (solution.delta_beta[a] - expected[n * d + a]).abs()
                    < 1e-10 * (1.0 + expected[n * d + a].abs()),
                "delta_beta[{a}] mismatch"
            );
        }
    }

    /// #1017: the row-procedural reduced-Schur matvec (the matrix-free SAE
    /// host backend) auto-fans its per-row point-elimination sum across rayon
    /// over fixed row chunks when at the top level (`n ≥
    /// SCHUR_MATVEC_PARALLEL_ROW_MIN`), and stays serial when already inside a
    /// rayon worker. The chunk-ordered fold makes the parallel result
    /// **deterministic** (two parallel calls are bit-identical — scheduling
    /// cannot change the numbers) and it agrees with the serial accumulation up
    /// to ULP-scale chunk reassociation (the #1017 verification gate). That
    /// reassociation is a genuine f64 departure from serial, so the criterion
    /// ranking across topology candidates is stable only up to the reassociation
    /// margin: a near-tie winner inside that margin can flip. This is NOT an
    /// exact no-move guarantee (#1211); for that, the ranking path must use the
    /// fixed-order serial accumulation.
    #[test]
    fn row_procedural_matvec_parallel_deterministic_and_matches_serial() {
        use crate::arrow_schur::SCHUR_MATVEC_PARALLEL_ROW_MIN;
        let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 96; // trips the parallel path
        let d = 3usize;
        let k = 24usize;
        let mut sys = build_fixture(n, d, k, 0xA17C_0FFE);
        // Install a matrix-free forward/transpose pair that reads the dense
        // `htbeta` slabs the fixture already populated, so the procedural
        // backend has a well-defined operator to apply (and exercises exactly
        // the sparse gather/scatter the SAE Kronecker path drives).
        let slabs: Vec<Array2<f64>> = sys.rows.iter().map(|row| row.htbeta.clone()).collect();
        let forward_slabs = slabs.clone();
        let transpose_slabs = slabs;
        sys.set_row_htbeta_operator(
            move |row: usize, x: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
                let h = &forward_slabs[row];
                for r in 0..h.nrows() {
                    let mut acc = 0.0_f64;
                    for c in 0..h.ncols() {
                        acc += h[[r, c]] * x[c];
                    }
                    out[r] = acc;
                }
            },
            move |row: usize, v: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
                let h = &transpose_slabs[row];
                for r in 0..h.nrows() {
                    for c in 0..h.ncols() {
                        out[c] += h[[r, c]] * v[r];
                    }
                }
            },
        );

        let matvec = gpu_schur_matvec_backend(&sys, 0.0, 0.0)
            .expect("row-procedural matvec backend builds for matrix-free system");
        let x = Array1::from_shape_fn(k, |i| ((i as f64 + 1.0) * 0.37).sin());

        // Top-level call: auto-selects the parallel chunk-fold. Run twice and
        // assert bit-identity — the chunk-ordered reduction must not depend on
        // thread scheduling.
        let mut out_parallel_a = Array1::<f64>::zeros(k);
        matvec(&x, &mut out_parallel_a);
        let mut out_parallel_b = Array1::<f64>::zeros(k);
        matvec(&x, &mut out_parallel_b);
        for a in 0..k {
            assert_eq!(
                out_parallel_a[a].to_bits(),
                out_parallel_b[a].to_bits(),
                "row-procedural matvec parallel reduction is non-deterministic at index {a}"
            );
        }

        // Inside a rayon worker: auto-selects the serial path (nested-rayon
        // guard). `install` runs the closure on a pool thread, so
        // `current_thread_index()` is `Some`. The serial running sum and the
        // chunk-ordered parallel fold differ only by f64 reassociation.
        let mut out_serial = Array1::<f64>::zeros(k);
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build rayon pool")
            .install(|| matvec(&x, &mut out_serial));

        let max_abs = out_serial.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for a in 0..k {
            let diff = (out_parallel_a[a] - out_serial[a]).abs();
            assert!(
                diff <= 1e-12 * (1.0 + max_abs),
                "row-procedural matvec parallel vs serial diverged beyond reassociation \
                 at index {a}: {} vs {} (diff={diff:e})",
                out_parallel_a[a],
                out_serial[a]
            );
        }
    }

    /// #1017/#1026 — the frames-engaged CPU reduced-Schur matvec
    /// [`sae_framed_schur_matvec_cpu`] (the bit-parity oracle the GPU kernel
    /// mirrors) must equal the dense reduced Schur `S = (P_ββ + ρ_β I) −
    /// Σ_i H_βt^(i)(H_tt^(i)+ρ_t I)⁻¹ H_tβ^(i)` formed by the canonical dense
    /// reference, on a small framed system with mixed per-atom ranks
    /// (`r_k < p` framed + `r_k = p` un-framed). Size-independent gate.
    #[test]
    fn framed_sae_schur_matvec_matches_dense_reference() {
        use crate::arrow_schur::{
            BetaPenaltyOp, DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock,
            FactoredFrameGBlock, FactoredFrameKroneckerOp, IdentityRightKroneckerPenaltyOp,
        };

        let p = 4usize;
        // Three atoms: ranks 2 (framed), 4 (un-framed), 3 (framed).
        let ranks = vec![2usize, 4usize, 3usize];
        let basis_sizes = vec![2usize, 1usize, 2usize];
        let n_atoms = ranks.len();
        let mut border_offsets = Vec::with_capacity(n_atoms);
        let mut acc = 0usize;
        for k in 0..n_atoms {
            border_offsets.push(acc);
            acc += basis_sizes[k] * ranks[k];
        }
        let border_dim = acc; // 2*2 + 1*4 + 2*3 = 14

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut sample = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };

        // Per-atom orthonormal-ish frames U_k (p × r_k) for the W = U_iᵀU_j
        // factors; un-framed atom (r=p) uses U = I_p.
        let mut frames: Vec<Array2<f64>> = Vec::with_capacity(n_atoms);
        for k in 0..n_atoms {
            let r = ranks[k];
            let mut u = Array2::<f64>::zeros((p, r));
            for i in 0..p {
                for j in 0..r {
                    u[[i, j]] = if r == p && i == j {
                        1.0
                    } else if r == p {
                        0.0
                    } else {
                        sample()
                    };
                }
            }
            frames.push(u);
        }
        let w_of = |i: usize, j: usize| -> Array2<f64> {
            let (ui, uj) = (&frames[i], &frames[j]);
            let (ri, rj) = (ranks[i], ranks[j]);
            let mut w = Array2::<f64>::zeros((ri, rj));
            for a in 0..ri {
                for b in 0..rj {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += ui[[c, a]] * uj[[c, b]];
                    }
                    w[[a, b]] = s;
                }
            }
            w
        };

        // Co-occurring data-fit blocks: all diagonal pairs + one cross (0,2).
        let mut frame_blocks: Vec<FactoredFrameGBlock> = Vec::new();
        let mut pairs = vec![(0usize, 0usize), (1, 1), (2, 2), (0, 2), (2, 0)];
        pairs.sort();
        for &(i, j) in &pairs {
            let (mi, mj) = (basis_sizes[i], basis_sizes[j]);
            let mut g = Array2::<f64>::zeros((mi, mj));
            for r in 0..mi {
                for c in 0..mj {
                    g[[r, c]] = 0.3 * sample();
                }
            }
            // Make diagonal blocks SPD-leaning so S stays PD.
            if i == j {
                for r in 0..mi.min(mj) {
                    g[[r, r]] += mi as f64 + 2.0;
                }
            }
            frame_blocks.push(FactoredFrameGBlock {
                atom_i: i,
                atom_j: j,
                g,
                w: w_of(i, j),
            });
        }

        // Smooth blocks λ S_k (M_k × M_k), SPD.
        let mut smooth_blocks: Vec<DeviceSaeSmoothBlock> = Vec::with_capacity(n_atoms);
        let mut smooth_ranks: Vec<usize> = Vec::with_capacity(n_atoms);
        for k in 0..n_atoms {
            let m = basis_sizes[k];
            let mut a = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    a[[r, c]] = 0.2 * sample();
                }
            }
            let mut s = a.t().dot(&a);
            for r in 0..m {
                s[[r, r]] += 1.0;
            }
            smooth_blocks.push(DeviceSaeSmoothBlock {
                global_offset: border_offsets[k],
                factor_a: s,
            });
            smooth_ranks.push(ranks[k]);
        }

        // Build the system: n rows, dense htbeta slabs (q_i × border_dim).
        let n = 6usize;
        let q = 3usize;
        let mut sys = ArrowSchurSystem::new(n, q, border_dim);
        let mut row_htbeta: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            // SPD htt.
            let mut a = Array2::<f64>::zeros((q, q));
            for r in 0..q {
                for c in 0..q {
                    a[[r, c]] = sample();
                }
            }
            let mut htt = a.t().dot(&a);
            for r in 0..q {
                htt[[r, r]] += q as f64 + 1.0;
            }
            sys.rows[i].htt = htt;
            let mut slab = vec![0.0_f64; q * border_dim];
            for c in 0..q {
                for col in 0..border_dim {
                    let v = 0.15 * sample();
                    slab[c * border_dim + col] = v;
                    sys.rows[i].htbeta[[c, col]] = v;
                }
            }
            row_htbeta.push(slab);
        }

        // Dense H_ββ from the SAME penalty ops (so the dense reference's S
        // matches the device penalty side exactly).
        let data_op =
            FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), frame_blocks.clone())
                .expect("frame op");
        let mut hbb = data_op.to_dense();
        for k in 0..n_atoms {
            let op = IdentityRightKroneckerPenaltyOp {
                factor_a: smooth_blocks[k].factor_a.clone(),
                p: ranks[k],
                global_offset: border_offsets[k],
                k: border_dim,
            };
            let d = op.to_dense();
            for r in 0..border_dim {
                for c in 0..border_dim {
                    hbb[[r, c]] += d[[r, c]];
                }
            }
        }
        sys.hbb = hbb;

        let data = DeviceSaePcgData {
            p,
            beta_dim: border_dim,
            a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            smooth_blocks,
            sparse_g_blocks: Vec::new(),
            frame: Some(DeviceSaeFrameData {
                ranks: ranks.clone(),
                basis_sizes: basis_sizes.clone(),
                border_offsets: border_offsets.clone(),
                frame_blocks,
                smooth_ranks,
                row_htbeta,
            }),
        };

        let ridge_t = 1e-7;
        let ridge_beta = 1e-6;

        // Dense reference reduced Schur S (border_dim × border_dim), formed
        // exactly as solve_arrow_newton_step_dense_reference assembles the
        // bordered Hessian and eliminates the t-block.
        let mut s_dense = Array2::<f64>::zeros((border_dim, border_dim));
        for r in 0..border_dim {
            for c in 0..border_dim {
                s_dense[[r, c]] = sys.hbb[[r, c]];
            }
            s_dense[[r, r]] += ridge_beta;
        }
        for row in &sys.rows {
            let mut htt = row.htt.clone();
            for d in 0..q {
                htt[[d, d]] += ridge_t;
            }
            let factor = cholesky_factor_in_place(htt.view(), CholeskyGuard::NonnegativePivot)
                .expect("htt PD");
            // Y = (htt)⁻¹ htbeta  (q × border_dim); S -= htbetaᵀ Y.
            let mut y = Array2::<f64>::zeros((q, border_dim));
            for col in 0..border_dim {
                let mut e = Array1::<f64>::zeros(q);
                for r in 0..q {
                    e[r] = row.htbeta[[r, col]];
                }
                let solved = cholesky_solve_vector(factor.view(), e.view());
                for r in 0..q {
                    y[[r, col]] = solved[r];
                }
            }
            for r in 0..border_dim {
                for c in 0..border_dim {
                    let mut acc = 0.0;
                    for d in 0..q {
                        acc += row.htbeta[[d, r]] * y[[d, c]];
                    }
                    s_dense[[r, c]] -= acc;
                }
            }
        }

        // Probe vectors: compare S·x from the device-data CPU oracle vs dense S·x.
        let mut max_rel = 0.0_f64;
        for trial in 0..4 {
            let x: Vec<f64> = (0..border_dim)
                .map(|a| 0.3 * ((a as f64 + trial as f64) * 0.21).cos() - 0.1)
                .collect();
            let mut got = vec![0.0_f64; border_dim];
            sae_framed_schur_matvec_cpu(&sys, &data, ridge_t, ridge_beta, &x, &mut got)
                .expect("framed matvec");
            let mut want = vec![0.0_f64; border_dim];
            for r in 0..border_dim {
                let mut acc = 0.0;
                for c in 0..border_dim {
                    acc += s_dense[[r, c]] * x[c];
                }
                want[r] = acc;
            }
            let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            for a in 0..border_dim {
                let rel = (got[a] - want[a]).abs() / scale;
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel <= 1e-10,
            "framed SAE Schur matvec vs dense reference diverged: max_rel={max_rel:e}"
        );
    }

    /// #1017/#1026 — large-K, many-atom, dense-cross-pair parity for the framed
    /// SAE reduced-Schur CPU oracle. The small `framed_sae_schur_matvec_matches_dense_reference`
    /// pins `border_dim=14`/3 atoms/1 cross pair; the interactions that only
    /// appear at scale — variable per-atom `r_k` (mixed framed `r_k<p` and
    /// un-framed `r_k=p`), the prefix-sum `border_offsets`, and dense cross-atom
    /// `W_ij` coupling across MANY co-occurring `frame_blocks` and many `row_htbeta`
    /// slabs — were validated only on the A100. This pins the CPU oracle (and
    /// therefore the device kernel it mirrors) against the dense reduced Schur at
    /// 40 atoms / `border_dim≈240` / a neighbour-coupled cross-pair set, on CPU.
    #[test]
    fn framed_sae_schur_matvec_matches_dense_reference_large_k_1026() {
        use crate::arrow_schur::{
            BetaPenaltyOp, DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock,
            FactoredFrameGBlock, FactoredFrameKroneckerOp, IdentityRightKroneckerPenaltyOp,
        };

        let p = 12usize;
        let n_atoms = 40usize;
        // Variable per-atom rank: most atoms are genuinely framed (r_k<p), every
        // fifth atom is un-framed (r_k=p, U=I_p — the within-atom G⊗I_r collapse).
        let ranks: Vec<usize> = (0..n_atoms)
            .map(|k| if k % 5 == 0 { p } else { 2 + (k % 3) })
            .collect();
        let basis_sizes: Vec<usize> = (0..n_atoms).map(|k| 1 + (k % 3)).collect();
        let mut border_offsets = Vec::with_capacity(n_atoms);
        let mut acc = 0usize;
        for k in 0..n_atoms {
            border_offsets.push(acc);
            acc += basis_sizes[k] * ranks[k];
        }
        let border_dim = acc;

        let mut state = 0x0bad_c0de_dead_beefu64;
        let mut sample = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };

        // Per-atom frames U_k (p × r_k); un-framed atom (r=p) uses U = I_p.
        let mut frames: Vec<Array2<f64>> = Vec::with_capacity(n_atoms);
        for k in 0..n_atoms {
            let r = ranks[k];
            let mut u = Array2::<f64>::zeros((p, r));
            for i in 0..p {
                for j in 0..r {
                    u[[i, j]] = if r == p {
                        if i == j { 1.0 } else { 0.0 }
                    } else {
                        sample()
                    };
                }
            }
            frames.push(u);
        }
        let w_of = |i: usize, j: usize| -> Array2<f64> {
            let (ui, uj) = (&frames[i], &frames[j]);
            let (ri, rj) = (ranks[i], ranks[j]);
            let mut w = Array2::<f64>::zeros((ri, rj));
            for a in 0..ri {
                for b in 0..rj {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += ui[[c, a]] * uj[[c, b]];
                    }
                    w[[a, b]] = s;
                }
            }
            w
        };

        // Co-occurring data-fit blocks: every diagonal pair + each neighbour pair
        // (k,k+1) and its transpose — dense cross coupling across the whole border.
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for k in 0..n_atoms {
            pairs.push((k, k));
        }
        for k in 0..n_atoms - 1 {
            pairs.push((k, k + 1));
            pairs.push((k + 1, k));
        }
        pairs.sort_unstable();
        let mut frame_blocks: Vec<FactoredFrameGBlock> = Vec::new();
        for &(i, j) in &pairs {
            let (mi, mj) = (basis_sizes[i], basis_sizes[j]);
            let mut g = Array2::<f64>::zeros((mi, mj));
            for r in 0..mi {
                for c in 0..mj {
                    g[[r, c]] = 0.3 * sample();
                }
            }
            if i == j {
                for r in 0..mi.min(mj) {
                    g[[r, r]] += mi as f64 + 2.0;
                }
            }
            frame_blocks.push(FactoredFrameGBlock {
                atom_i: i,
                atom_j: j,
                g,
                w: w_of(i, j),
            });
        }

        // Smooth blocks λ S_k (M_k × M_k), SPD.
        let mut smooth_blocks: Vec<DeviceSaeSmoothBlock> = Vec::with_capacity(n_atoms);
        let mut smooth_ranks: Vec<usize> = Vec::with_capacity(n_atoms);
        for k in 0..n_atoms {
            let m = basis_sizes[k];
            let mut a = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    a[[r, c]] = 0.2 * sample();
                }
            }
            let mut s = a.t().dot(&a);
            for r in 0..m {
                s[[r, r]] += 1.0;
            }
            smooth_blocks.push(DeviceSaeSmoothBlock {
                global_offset: border_offsets[k],
                factor_a: s,
            });
            smooth_ranks.push(ranks[k]);
        }

        // n rows with SPD htt and full-width (q × border_dim) htbeta slabs.
        let n = 8usize;
        let q = 3usize;
        let mut sys = ArrowSchurSystem::new(n, q, border_dim);
        let mut row_htbeta: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut a = Array2::<f64>::zeros((q, q));
            for r in 0..q {
                for c in 0..q {
                    a[[r, c]] = sample();
                }
            }
            let mut htt = a.t().dot(&a);
            for r in 0..q {
                htt[[r, r]] += q as f64 + 1.0;
            }
            sys.rows[i].htt = htt;
            let mut slab = vec![0.0_f64; q * border_dim];
            for c in 0..q {
                for col in 0..border_dim {
                    let v = 0.15 * sample();
                    slab[c * border_dim + col] = v;
                    sys.rows[i].htbeta[[c, col]] = v;
                }
            }
            row_htbeta.push(slab);
        }

        // Dense H_ββ from the SAME penalty ops (data-fit + smooth), so the dense
        // reference S matches the device penalty side exactly.
        let data_op =
            FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), frame_blocks.clone())
                .expect("frame op");
        let mut hbb = data_op.to_dense();
        for k in 0..n_atoms {
            let op = IdentityRightKroneckerPenaltyOp {
                factor_a: smooth_blocks[k].factor_a.clone(),
                p: ranks[k],
                global_offset: border_offsets[k],
                k: border_dim,
            };
            let d = op.to_dense();
            for r in 0..border_dim {
                for c in 0..border_dim {
                    hbb[[r, c]] += d[[r, c]];
                }
            }
        }
        sys.hbb = hbb;

        let data = DeviceSaePcgData {
            p,
            beta_dim: border_dim,
            a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            smooth_blocks,
            sparse_g_blocks: Vec::new(),
            frame: Some(DeviceSaeFrameData {
                ranks: ranks.clone(),
                basis_sizes: basis_sizes.clone(),
                border_offsets: border_offsets.clone(),
                frame_blocks,
                smooth_ranks,
                row_htbeta,
            }),
        };

        let ridge_t = 1e-7;
        let ridge_beta = 1e-6;

        // Dense reference reduced Schur S = (hbb + ridge_beta I) - Σ_i htbetaᵀ (htt+ridge_t I)⁻¹ htbeta.
        let mut s_dense = sys.hbb.clone();
        for r in 0..border_dim {
            s_dense[[r, r]] += ridge_beta;
        }
        for row in &sys.rows {
            let mut htt = row.htt.clone();
            for d in 0..q {
                htt[[d, d]] += ridge_t;
            }
            let factor = cholesky_factor_in_place(htt.view(), CholeskyGuard::NonnegativePivot)
                .expect("htt PD");
            let mut y = Array2::<f64>::zeros((q, border_dim));
            for col in 0..border_dim {
                let mut e = Array1::<f64>::zeros(q);
                for r in 0..q {
                    e[r] = row.htbeta[[r, col]];
                }
                let solved = cholesky_solve_vector(factor.view(), e.view());
                for r in 0..q {
                    y[[r, col]] = solved[r];
                }
            }
            for r in 0..border_dim {
                for c in 0..border_dim {
                    let mut acc = 0.0;
                    for d in 0..q {
                        acc += row.htbeta[[d, r]] * y[[d, c]];
                    }
                    s_dense[[r, c]] -= acc;
                }
            }
        }

        let mut max_rel = 0.0_f64;
        for trial in 0..4 {
            let x: Vec<f64> = (0..border_dim)
                .map(|a| 0.3 * ((a as f64 + trial as f64) * 0.21).cos() - 0.1)
                .collect();
            let mut got = vec![0.0_f64; border_dim];
            sae_framed_schur_matvec_cpu(&sys, &data, ridge_t, ridge_beta, &x, &mut got)
                .expect("framed matvec");
            let mut want = vec![0.0_f64; border_dim];
            for r in 0..border_dim {
                let mut acc = 0.0;
                for c in 0..border_dim {
                    acc += s_dense[[r, c]] * x[c];
                }
                want[r] = acc;
            }
            let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            for a in 0..border_dim {
                let rel = (got[a] - want[a]).abs() / scale;
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel <= 1e-10,
            "large-K framed SAE Schur matvec vs dense reference diverged: \
             max_rel={max_rel:e} (n_atoms={n_atoms}, border_dim={border_dim})"
        );
    }

    /// #1017/#1026 GPU arm: when a CUDA device admits the framed SAE PCG, its
    /// solved `δβ` must match the CPU dense reduced-system solve of the SAME
    /// framed system (size-independent — a small device validates the kernel).
    /// Skips cleanly (returns) when no device is available or the policy
    /// declines (`solve_sae_matrix_free_pcg` → `Unavailable`).
    #[test]
    fn framed_sae_device_pcg_matches_cpu_when_cuda_admits() {
        use crate::arrow_schur::{
            BetaPenaltyOp, DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock,
            FactoredFrameGBlock, FactoredFrameKroneckerOp, IdentityRightKroneckerPenaltyOp,
        };

        // Large enough to clear the device-offload policy floor (k ≥ 32 and
        // n·k·d·iters ≥ MATVEC_OFFLOAD_FLOPS_MIN) so the GPU kernel actually
        // runs on a device rather than the policy declining.
        let p = 6usize;
        let n_atoms = 8usize;
        let ranks: Vec<usize> = (0..n_atoms)
            .map(|k| if k % 2 == 0 { 3usize } else { p })
            .collect();
        let basis_sizes: Vec<usize> = (0..n_atoms).map(|_| 3usize).collect();
        let mut border_offsets = Vec::with_capacity(n_atoms);
        let mut acc = 0usize;
        for k in 0..n_atoms {
            border_offsets.push(acc);
            acc += basis_sizes[k] * ranks[k];
        }
        let border_dim = acc; // Σ M_k·r_k = 4·(3·3) + 4·(3·6) = 36 + 72 = 108

        let mut state = 0xfeed_face_dead_beefu64;
        let mut sample = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };
        let mut frames: Vec<Array2<f64>> = Vec::new();
        for k in 0..n_atoms {
            let r = ranks[k];
            let mut u = Array2::<f64>::zeros((p, r));
            for i in 0..p {
                for j in 0..r {
                    u[[i, j]] = if r == p && i == j {
                        1.0
                    } else if r == p {
                        0.0
                    } else {
                        sample()
                    };
                }
            }
            frames.push(u);
        }
        let w_of = |i: usize, j: usize| {
            let (ui, uj) = (&frames[i], &frames[j]);
            let (ri, rj) = (ranks[i], ranks[j]);
            let mut w = Array2::<f64>::zeros((ri, rj));
            for a in 0..ri {
                for b in 0..rj {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += ui[[c, a]] * uj[[c, b]];
                    }
                    w[[a, b]] = s;
                }
            }
            w
        };
        let mut pairs: Vec<(usize, usize)> = (0..n_atoms).map(|k| (k, k)).collect();
        // A few off-diagonal cross blocks (symmetric pairs).
        for &(i, j) in &[(0usize, 1usize), (2, 4), (3, 6)] {
            pairs.push((i, j));
            pairs.push((j, i));
        }
        let mut frame_blocks = Vec::new();
        for &(i, j) in &pairs {
            let (mi, mj) = (basis_sizes[i], basis_sizes[j]);
            let mut g = Array2::<f64>::zeros((mi, mj));
            for r in 0..mi {
                for c in 0..mj {
                    g[[r, c]] = 0.25 * sample();
                }
            }
            if i == j {
                for r in 0..mi.min(mj) {
                    g[[r, r]] += mi as f64 + 2.0;
                }
            }
            frame_blocks.push(FactoredFrameGBlock {
                atom_i: i,
                atom_j: j,
                g,
                w: w_of(i, j),
            });
        }
        let mut smooth_blocks = Vec::new();
        let mut smooth_ranks = Vec::new();
        for k in 0..n_atoms {
            let m = basis_sizes[k];
            let mut a = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    a[[r, c]] = 0.2 * sample();
                }
            }
            let mut s = a.t().dot(&a);
            for r in 0..m {
                s[[r, r]] += 1.0;
            }
            smooth_blocks.push(DeviceSaeSmoothBlock {
                global_offset: border_offsets[k],
                factor_a: s,
            });
            smooth_ranks.push(ranks[k]);
        }
        let n = 400usize;
        let q = 4usize;
        let mut sys = ArrowSchurSystem::new(n, q, border_dim);
        let mut row_htbeta = Vec::new();
        for i in 0..n {
            let mut a = Array2::<f64>::zeros((q, q));
            for r in 0..q {
                for c in 0..q {
                    a[[r, c]] = sample();
                }
            }
            let mut htt = a.t().dot(&a);
            for r in 0..q {
                htt[[r, r]] += q as f64 + 1.0;
            }
            sys.rows[i].htt = htt;
            let mut slab = vec![0.0_f64; q * border_dim];
            for c in 0..q {
                for col in 0..border_dim {
                    // Small entries: with 400 rows the reduced-Schur subtraction
                    // Σ_i H_βtᵀ H_tt⁻¹ H_tβ must not overwhelm the PD penalty.
                    let v = 0.02 * sample();
                    slab[c * border_dim + col] = v;
                    sys.rows[i].htbeta[[c, col]] = v;
                }
            }
            row_htbeta.push(slab);
        }
        let data_op =
            FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), frame_blocks.clone())
                .expect("frame op");
        let mut hbb = data_op.to_dense();
        for k in 0..n_atoms {
            let op = IdentityRightKroneckerPenaltyOp {
                factor_a: smooth_blocks[k].factor_a.clone(),
                p: ranks[k],
                global_offset: border_offsets[k],
                k: border_dim,
            };
            let d = op.to_dense();
            for r in 0..border_dim {
                for c in 0..border_dim {
                    hbb[[r, c]] += d[[r, c]];
                }
            }
        }
        sys.hbb = hbb;
        let data = DeviceSaePcgData {
            p,
            beta_dim: border_dim,
            a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            smooth_blocks,
            sparse_g_blocks: Vec::new(),
            frame: Some(DeviceSaeFrameData {
                ranks: ranks.clone(),
                basis_sizes: basis_sizes.clone(),
                border_offsets: border_offsets.clone(),
                frame_blocks,
                smooth_ranks,
                row_htbeta,
            }),
        };
        let ridge_t = 1e-7;
        let ridge_beta = 1e-6;
        let rhs: Array1<f64> =
            Array1::from_shape_fn(border_dim, |a| ((a as f64 + 1.0) * 0.17).sin());

        let (device, diag) =
            match solve_sae_matrix_free_pcg(&sys, &data, ridge_t, ridge_beta, &rhs, 400, 1e-12) {
                Ok(result) => result,
                // #1017 — fail loud, never skip-pass: this fixture clears the device
                // offload floor, so a CUDA device that is PRESENT yet declines means the
                // framed device PCG kernel does not run on GPU (the fault must not pass
                // silently). Legit skip ONLY when no usable CUDA device exists (CPU CI).
                // The exact `ArrowSchurGpuFailure` variant is folded into the assert.
                Err(failure) => {
                    assert!(
                        gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
                            .unwrap_or_else(|error| {
                                panic!("GPU probe fault in framed PCG test: {error}")
                            })
                            .is_none(),
                        "#1017: CUDA device present but the framed device SAE PCG \
                     declined/faulted instead of returning a result (tag: {failure:?}) — \
                     the kernel does not run correctly on GPU"
                    );
                    return;
                }
            };

        // #1551 PARITY GATE — operator-residual, NOT solution-vector equality.
        //
        // The honest GPU↔CPU contract for an iterative solve of `S·δβ = rhs` is
        // that the device solution SOLVES the system DEFINED BY THE CPU ORACLE to
        // PCG tolerance — i.e. `‖S_cpu·δβ_device − rhs‖ / ‖rhs‖ ≤ tol`, where
        // `S_cpu` is applied with the bit-for-bit CPU oracle matvec the device
        // kernel mirrors (`sae_framed_schur_matvec_cpu`). This is the correct,
        // conditioning-robust gate: a near-singular assembled `S` has a large
        // condition number `κ(S)`, which amplifies an O(ε) residual difference
        // into an O(κ·ε) *solution-vector* difference, so comparing δβ vectors
        // would spuriously fail even when both operators are bit-identical and
        // both solves converged. (Historically this test compared δβ against a
        // dense-Cholesky reference and "failed" with max_rel≈0.9 because the
        // dense solve itself only reached ‖S·x−rhs‖≈0.1 on this fixture's
        // ill-conditioned S while the device PCG reached ~1e-12 — the device was
        // MORE accurate than the reference, not wrong. The kernel correctness is
        // pinned conditioning-free by `framed_sae_device_matvec_matches_cpu_oracle_*`.)
        let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let oracle_resid = |x: &Array1<f64>| -> f64 {
            let mut sx = vec![0.0_f64; border_dim];
            sae_framed_schur_matvec_cpu(
                &sys,
                &data,
                ridge_t,
                ridge_beta,
                x.as_slice().unwrap(),
                &mut sx,
            )
            .expect("cpu oracle matvec");
            let mut acc = 0.0_f64;
            for a in 0..border_dim {
                let e = sx[a] - rhs[a];
                acc += e * e;
            }
            acc.sqrt()
        };
        let s_dev_resid = oracle_resid(&device);
        let dev_rel_resid = s_dev_resid / rhs_norm.max(1e-300);

        // Independent CPU iterative solve of the SAME operator with the SAME
        // Jacobi preconditioner the device builds, via the shared `pcg_core`. If
        // the device kernel computed a different operator, the two converged
        // residuals could not BOTH be tiny.
        let precond = {
            let d = sae_frame_penalty_diag_host_for_test(&data, ridge_beta);
            // The reduced-Schur diagonal subtraction (device `arrow_sae_frame_diag_sub`)
            // mirrored on the host for the Jacobi preconditioner.
            let mut diag = d;
            for (i, row) in sys.rows.iter().enumerate() {
                let slab = &data.frame.as_ref().unwrap().row_htbeta[i];
                let qi = sys.row_dims[i];
                if slab.is_empty() || qi == 0 || slab.len() != qi * border_dim {
                    continue;
                }
                let mut block = row.htt.clone();
                for dd in 0..qi {
                    block[[dd, dd]] += ridge_t;
                }
                let factor =
                    cholesky_factor_in_place(block.view(), CholeskyGuard::NonnegativePivot)
                        .expect("row htt PD");
                // ainv = (H_tt+ρI)⁻¹ column by column.
                let mut ainv = Array2::<f64>::zeros((qi, qi));
                for col in 0..qi {
                    let mut e = Array1::<f64>::zeros(qi);
                    e[col] = 1.0;
                    let s = cholesky_solve_vector(factor.view(), e.view());
                    for r in 0..qi {
                        ainv[[r, col]] = s[r];
                    }
                }
                for a in 0..border_dim {
                    let mut quad = 0.0_f64;
                    for c in 0..qi {
                        let hc = slab[c * border_dim + a];
                        for dd in 0..qi {
                            quad += hc * ainv[[c, dd]] * slab[dd * border_dim + a];
                        }
                    }
                    diag[a] -= quad;
                }
            }
            Array1::from_vec(diag)
        };
        let mut cpu = Array1::<f64>::zeros(border_dim);
        let cpu_result = {
            let mut apply = |v: &Array1<f64>, out: &mut Array1<f64>| {
                let mut tmp = vec![0.0_f64; border_dim];
                sae_framed_schur_matvec_cpu(
                    &sys,
                    &data,
                    ridge_t,
                    ridge_beta,
                    v.as_slice().unwrap(),
                    &mut tmp,
                )
                .expect("cpu oracle matvec");
                out.assign(&Array1::from_vec(tmp));
            };
            gam_linalg::pcg::pcg_core(
                &mut apply,
                &rhs.view(),
                &precond.view(),
                1e-12,
                800,
                32,
                false,
                gam_linalg::pcg::DotReduction::Serial,
                &mut cpu.view_mut(),
            )
        };
        let s_cpu_resid = oracle_resid(&cpu);
        let cpu_rel_resid = s_cpu_resid / rhs_norm.max(1e-300);

        // GATE 1: the device solution solves the CPU-oracle system to PCG-grade
        // accuracy (proves device kernel == CPU operator AND device PCG converged).
        assert!(
            dev_rel_resid <= 1e-7,
            "[#1551] device δβ does not solve the CPU-oracle system: \
             ‖S_cpu·device−rhs‖/‖rhs‖={dev_rel_resid:e} (>1e-7) | abs={s_dev_resid:e} | \
             device PCG stop={:?} iters={} final_rel_resid={:e} — a large operator residual \
             means the device matvec is a DIFFERENT operator (kernel bug)",
            diag.stopping_reason,
            diag.iterations,
            diag.final_relative_residual,
        );
        // GATE 2: the independent CPU iterative solve of the same operator with the
        // same preconditioner also converges — both paths agree on the operator.
        assert!(
            cpu_rel_resid <= 1e-6,
            "[#1551] CPU pcg_core failed to solve the oracle system: \
             ‖S_cpu·cpu−rhs‖/‖rhs‖={cpu_rel_resid:e} (stop={:?}, iters={}) — fixture/oracle issue",
            cpu_result.stop,
            cpu_result.iterations,
        );
    }

    /// Host mirror of the device `sae_frame_penalty_diag_host` for the framed
    /// Jacobi preconditioner (penalty diagonal only; the reduced-Schur diagonal
    /// subtraction is applied by the caller). Test-only.
    fn sae_frame_penalty_diag_host_for_test(data: &DeviceSaePcgData, ridge_beta: f64) -> Vec<f64> {
        let frame = data.frame.as_ref().expect("frame");
        let mut diag = vec![ridge_beta; data.beta_dim];
        for (blk, &r) in data.smooth_blocks.iter().zip(frame.smooth_ranks.iter()) {
            let m = blk.factor_a.nrows();
            for ia in 0..m {
                let coeff = blk.factor_a[[ia, ia]];
                let base = blk.global_offset + ia * r;
                for ib in 0..r {
                    diag[base + ib] += coeff;
                }
            }
        }
        for blk in &frame.frame_blocks {
            if blk.atom_i != blk.atom_j {
                continue;
            }
            let r = frame.ranks[blk.atom_i];
            let off = frame.border_offsets[blk.atom_i];
            let (mi, mj) = blk.g.dim();
            for li in 0..mi.min(mj) {
                let gii = blk.g[[li, li]];
                let base = off + li * r;
                for a in 0..r {
                    diag[base + a] += gii * blk.w[[a, a]];
                }
            }
        }
        diag
    }

    /// #1551 DEFINITIVE kernel-correctness proof: the framed reduced-Schur matvec
    /// `out = S·x` must agree with the CPU oracle [`sae_framed_schur_matvec_cpu`]
    /// element-wise, for several independent `x`, to ≤ 1e-9. This is the parity
    /// gate that actually localizes a kernel/marshalling defect — unlike a
    /// solved-`δβ` comparison, it does NOT route through a linear solve, so it is
    /// independent of the conditioning of the assembled `S` (a near-singular `S`
    /// can make a dense-Cholesky vector and an iterative-PCG vector disagree at
    /// the *solution* level even when both operators are bit-correct; the
    /// operator itself must still match here). Fails loud if CUDA is present but
    /// the device matvec declines; skips cleanly only when no device exists.
    #[test]
    #[cfg(target_os = "linux")]
    fn framed_sae_device_matvec_matches_cpu_oracle_when_cuda_admits() {
        use crate::arrow_schur::{
            DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock, FactoredFrameGBlock,
        };

        // Hand-checkable frame fixture: a mix of framed (r<p) and identity-ride
        // (r==p) atoms, a few off-diagonal cross blocks, dense per-row H_tβ.
        let p = 6usize;
        let n_atoms = 8usize;
        let ranks: Vec<usize> = (0..n_atoms)
            .map(|k| if k % 2 == 0 { 3usize } else { p })
            .collect();
        let basis_sizes: Vec<usize> = (0..n_atoms).map(|_| 3usize).collect();
        let mut border_offsets = Vec::with_capacity(n_atoms);
        let mut acc = 0usize;
        for k in 0..n_atoms {
            border_offsets.push(acc);
            acc += basis_sizes[k] * ranks[k];
        }
        let border_dim = acc;

        let mut state = 0x1551_0017_1026_0922u64;
        let mut sample = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };
        let mut frames: Vec<Array2<f64>> = Vec::new();
        for k in 0..n_atoms {
            let r = ranks[k];
            let mut u = Array2::<f64>::zeros((p, r));
            for i in 0..p {
                for j in 0..r {
                    u[[i, j]] = if r == p && i == j {
                        1.0
                    } else if r == p {
                        0.0
                    } else {
                        sample()
                    };
                }
            }
            frames.push(u);
        }
        let w_of = |i: usize, j: usize| {
            let (ui, uj) = (&frames[i], &frames[j]);
            let (ri, rj) = (ranks[i], ranks[j]);
            let mut w = Array2::<f64>::zeros((ri, rj));
            for a in 0..ri {
                for b in 0..rj {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += ui[[c, a]] * uj[[c, b]];
                    }
                    w[[a, b]] = s;
                }
            }
            w
        };
        let mut pairs: Vec<(usize, usize)> = (0..n_atoms).map(|k| (k, k)).collect();
        for &(i, j) in &[(0usize, 1usize), (2, 4), (3, 6)] {
            pairs.push((i, j));
            pairs.push((j, i));
        }
        let mut frame_blocks = Vec::new();
        for &(i, j) in &pairs {
            let (mi, mj) = (basis_sizes[i], basis_sizes[j]);
            let mut g = Array2::<f64>::zeros((mi, mj));
            for r in 0..mi {
                for c in 0..mj {
                    g[[r, c]] = 0.25 * sample();
                }
            }
            if i == j {
                for r in 0..mi.min(mj) {
                    g[[r, r]] += mi as f64 + 2.0;
                }
            }
            frame_blocks.push(FactoredFrameGBlock {
                atom_i: i,
                atom_j: j,
                g,
                w: w_of(i, j),
            });
        }
        let mut smooth_blocks = Vec::new();
        let mut smooth_ranks = Vec::new();
        for k in 0..n_atoms {
            let m = basis_sizes[k];
            let mut a = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    a[[r, c]] = 0.2 * sample();
                }
            }
            let mut s = a.t().dot(&a);
            for r in 0..m {
                s[[r, r]] += 1.0;
            }
            smooth_blocks.push(DeviceSaeSmoothBlock {
                global_offset: border_offsets[k],
                factor_a: s,
            });
            smooth_ranks.push(ranks[k]);
        }
        // Modest row count: this seam bypasses the offload floor, so we keep the
        // fixture small and the per-row reduced-Schur term well-scaled — the
        // matvec parity does not depend on the assembled-S conditioning at all.
        let n = 32usize;
        let q = 4usize;
        let mut sys = ArrowSchurSystem::new(n, q, border_dim);
        let mut row_htbeta = Vec::new();
        for i in 0..n {
            let mut a = Array2::<f64>::zeros((q, q));
            for r in 0..q {
                for c in 0..q {
                    a[[r, c]] = sample();
                }
            }
            let mut htt = a.t().dot(&a);
            for r in 0..q {
                htt[[r, r]] += q as f64 + 1.0;
            }
            sys.rows[i].htt = htt;
            let mut slab = vec![0.0_f64; q * border_dim];
            for c in 0..q {
                for col in 0..border_dim {
                    let v = 0.3 * sample();
                    slab[c * border_dim + col] = v;
                    sys.rows[i].htbeta[[c, col]] = v;
                }
            }
            row_htbeta.push(slab);
        }
        let ridge_t = 1e-7;
        let ridge_beta = 1e-6;
        let data = DeviceSaePcgData {
            p,
            beta_dim: border_dim,
            a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
            smooth_blocks,
            sparse_g_blocks: Vec::new(),
            frame: Some(DeviceSaeFrameData {
                ranks: ranks.clone(),
                basis_sizes: basis_sizes.clone(),
                border_offsets: border_offsets.clone(),
                frame_blocks,
                smooth_ranks,
                row_htbeta,
            }),
        };

        // Several independent probe vectors x, including unit axes and dense
        // random — a marshalling stride/offset bug shows up as a per-component
        // mismatch on at least one.
        let mut probes: Vec<Array1<f64>> = Vec::new();
        probes.push(Array1::from_shape_fn(border_dim, |a| {
            ((a as f64 + 1.0) * 0.37).sin()
        }));
        probes.push(Array1::from_shape_fn(border_dim, |_| sample()));
        for axis in [0usize, border_dim / 3, border_dim - 1] {
            let mut e = Array1::<f64>::zeros(border_dim);
            e[axis] = 1.0;
            probes.push(e);
        }

        let mut any_ran = false;
        let mut worst = 0.0_f64;
        for (pi, x) in probes.iter().enumerate() {
            let device = match super::framed_schur_matvec_once_on_device(
                &sys, &data, ridge_t, ridge_beta, x,
            ) {
                Ok(out) => out,
                Err(failure) => {
                    // Fail loud: a present CUDA device that declines this seam
                    // (which deliberately ignores the offload floor) means the
                    // framed matvec kernel does not run on GPU.
                    assert!(
                        gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
                            .unwrap_or_else(|error| {
                                panic!("GPU probe fault in framed matvec parity test: {error}")
                            })
                            .is_none(),
                        "#1551: CUDA device present but the framed device matvec \
                         declined/faulted (probe {pi}, tag: {failure:?}) — the kernel \
                         does not run on GPU"
                    );
                    return;
                }
            };
            any_ran = true;
            let mut cpu = vec![0.0_f64; border_dim];
            sae_framed_schur_matvec_cpu(
                &sys,
                &data,
                ridge_t,
                ridge_beta,
                x.as_slice().unwrap(),
                &mut cpu,
            )
            .expect("cpu oracle matvec");
            let scale = cpu.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            for a in 0..border_dim {
                let rel = (device[a] - cpu[a]).abs() / scale;
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-9,
                    "[#1551 matvec-parity] probe {pi} component {a}: device={:e} cpu={:e} \
                     rel={rel:e} (>1e-9) — framed S·x kernel diverges from the CPU oracle",
                    device[a],
                    cpu[a],
                );
            }
        }
        if any_ran {
            // Positive on-device confirmation: the framed matvec ran on the GPU
            // and matched the CPU oracle across every probe. (1e-9 is far above
            // the ~1e-13 fp64 GEMV round-off; a structural marshalling bug would
            // be O(1).)
            assert!(
                gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
                    .unwrap_or_else(|error| {
                        panic!("GPU probe fault after framed matvec execution: {error}")
                    })
                    .is_some(),
                "#1551: matvec ran but no GPU runtime — unexpected"
            );
            assert!(worst <= 1e-9, "framed matvec parity worst rel = {worst:e}");
        }
    }
}
