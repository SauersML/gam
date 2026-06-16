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
//! On non-Linux builds the entire module degrades to a CPU-fallback shim.

#![allow(clippy::module_name_repetitions)]

use ndarray::{Array1, Array2};

use crate::linalg::triangular::{CholeskyGuard, cholesky_factor_in_place, cholesky_solve_vector};
use crate::solver::arrow_schur::{ArrowSchurSystem, DeviceSaePcgData, PcgDiagnostics};

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
    /// The system carries matrix-free `H_ββ` or per-row `H_tβ` operators that
    /// the dense GPU Schur path cannot consume. The caller should route to CPU
    /// `InexactPCG` (or supply dense buffers) rather than treating this as a
    /// numerical failure. See `gpu/arrow_schur.rs` Part B for the planned GPU
    /// PCG path that will lift this restriction at K ≥ 5000.
    GpuRequiresDenseSystem {
        had_hbb_matvec: bool,
        had_htbeta_matvec: bool,
    },
}

/// Safety-margin multiplier on `√(machine ε)` for the diagonal ridge bump
/// suggested when a local block fails Cholesky.
///
/// The estimated bump is `diag_scale · |pivot| · √ε · RIDGE_BUMP_EPS_MARGIN`.
/// A bare `diag_scale · √ε` ridge is the smallest perturbation that makes a
/// marginally-indefinite block PD in exact arithmetic, but a single retry at
/// that magnitude is routinely re-rejected by the next POTRF because the
/// rounding error of forming `D + ridge·I` and re-factoring is itself O(√ε).
/// The 1024× headroom (≈ 2¹⁰, i.e. ten extra bits below the f64 mantissa's
/// 52) clears the pivot on the first retry without materially perturbing the
/// curvature the Newton step sees. Shared by the per-row scalar path and the
/// batched-tile path so both suggest an identical bump.
const RIDGE_BUMP_EPS_MARGIN: f64 = 1024.0;

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

    if sys.hbb.dim() != (k, k) {
        return Err(ArrowSchurGpuFailure::SchurFactorFailed {
            reason: "CUDA arrow-Schur requires a dense shared beta block".to_string(),
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
        if crate::gpu::device_runtime::GpuRuntime::global()
            .map(crate::gpu::device_runtime::GpuRuntime::device_count)
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
        if crate::gpu::kernels::arrow_schur_nvrtc::system_admits_fused_path(sys) {
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
    row: &crate::solver::arrow_schur::ArrowRowBlock,
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

/// Test-only entry that forces the Layer D + E fused NVRTC path regardless
/// of the admission heuristic. Used by the V100 Layer C↔D parity test to
/// drive the fused kernel at small shapes the heuristic would otherwise
/// route through the cuSOLVER/cuBLAS Layer A+B+C path.
#[doc(hidden)]
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
    if crate::gpu::kernels::arrow_schur_nvrtc::plan_fused_launch(sys.rows.len(), sys.d, sys.k)
        .is_none()
    {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(ArrowSchurGpuFailure::Unavailable)
    }
    #[cfg(target_os = "linux")]
    {
        cuda::solve_fused(sys, ridge_t, ridge_beta)
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
) -> Result<crate::solver::arrow_schur::GpuSchurMatvec, ArrowSchurGpuFailure> {
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
) -> Result<crate::solver::arrow_schur::GpuSchurMatvec, ArrowSchurGpuFailure> {
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
                let scale = row
                    .htt
                    .diag()
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0);
                ArrowSchurGpuFailure::RidgeBumpRequired {
                    row: i,
                    bump: scale * f64::EPSILON.sqrt() * RIDGE_BUMP_EPS_MARGIN,
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

    let closure: crate::solver::arrow_schur::GpuSchurMatvec =
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
            // to ULP-scale chunk reassociation (the #1017 verification gate: the
            // criterion ranking across topology candidates must not move). Stay
            // sequential below
            // `SCHUR_MATVEC_PARALLEL_ROW_MIN` rows and when already inside a
            // rayon worker (the topology race fans candidates with
            // `run_topology_race_parallel`) — the same nested-rayon guard the
            // CPU `schur_matvec` uses. Buffers (`v_i`, `neg`) are reused across
            // rows within a chunk, so the per-row allocation churn is gone.
            let parallel = n >= crate::solver::arrow_schur::SCHUR_MATVEC_PARALLEL_ROW_MIN
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
                // Deterministic ordered reduction: fold chunk partials
                // left-to-right, then subtract once.
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
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurGpuFailure> {
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
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurGpuFailure> {
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

#[cfg(target_os = "linux")]
mod cuda {
    use super::{ArrowSchurGpuFailure, ArrowSchurGpuSolution, pack_block, pack_host};
    use crate::gpu::driver::to_i32;
    use crate::gpu::linalg_dispatch::{DispatchOp, route_through_gpu};
    use crate::solver::arrow_schur::{
        ArrowSchurSystem, DeviceSaePcgData, PcgDiagnostics, PcgStopReason,
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
        diag_scale: f64,   // |diag(H_tt)| scale for the ridge-bump diagnostic
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
    /// `crate::gpu::pool::scatter_batched` hands each device a contiguous row
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

        let runtime = crate::gpu::device_runtime::GpuRuntime::global()
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
            let diag_scale = row
                .htt
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            slots.push(RowSlot {
                d_block,
                b_block,
                g_vec,
                diag_scale,
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
        let forward_ok = crate::gpu::pool::scatter_batched(runtime, &mut slots, |ordinal, tile| {
            forward_tile(ordinal, d, k, tile)
        });
        if forward_ok.is_none() {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }

        // Surface a non-PD tip block as a precise per-row ridge bump.
        let row_base_of_tile = crate::gpu::pool::balanced_partition(runtime, n);
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
        let stream = crate::gpu::device_runtime::cuda_context_for(primary)
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
        let back_ok = crate::gpu::pool::scatter_batched(runtime, &mut slots, |ordinal, tile| {
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
        let stream = crate::gpu::device_runtime::cuda_context_for(ordinal)
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
        let info_host = potrf_batched(&solver, &stream, d, m, &mut d_dev).ok()?;
        if let Some(local) = info_host.iter().position(|info| *info != 0) {
            let pivot = info_host[local];
            tile[local].bump = Some(
                tile[local].diag_scale
                    * (f64::from(pivot).abs()).max(1.0)
                    * f64::EPSILON.sqrt()
                    * super::RIDGE_BUMP_EPS_MARGIN,
            );
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
        let stream = crate::gpu::device_runtime::cuda_context_for(ordinal)
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

        let stream = crate::gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
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
            let pivot = info_host[idx];
            let scale = sys.rows[idx]
                .htt
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                row: idx,
                bump: scale * (pivot.abs() as f64).max(1.0) * f64::EPSILON.sqrt() * 1024.0,
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
            HashMap<crate::gpu::kernels::arrow_schur_nvrtc::FusedModuleCacheKey, Arc<CudaModule>>,
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
        key: crate::gpu::kernels::arrow_schur_nvrtc::FusedModuleCacheKey,
    ) -> Result<Arc<CudaModule>, ArrowSchurGpuFailure> {
        let cache = fused_module_cache();
        if let Ok(guard) = cache.modules.lock() {
            if let Some(existing) = guard.get(&key) {
                return Ok(existing.clone());
            }
        }
        let src = crate::gpu::kernels::arrow_schur_nvrtc::forward_kernel_source(
            key.p_max as usize,
            key.r_template as usize,
        );
        let ptx = cudarc::nvrtc::compile_ptx(&src).map_err(|err| {
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
    out[(rbase + li) * p + oc] += acc;
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
"#;

    fn pcg_vector_module(
        ctx: &Arc<CudaContext>,
    ) -> Result<&'static Arc<CudaModule>, ArrowSchurGpuFailure> {
        static CACHE: crate::gpu::device_cache::PtxModuleCache =
            crate::gpu::device_cache::PtxModuleCache::new();
        CACHE
            .get_or_compile(ctx, "arrow_pcg_vector", PCG_VECTOR_KERNEL_SOURCE)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
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
        for row in &data.a_phi {
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
        for row_jac in &data.local_jac {
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
            let factor = crate::linalg::triangular::cholesky_factor_in_place(
                block.view(),
                crate::linalg::triangular::CholeskyGuard::NonnegativePivot,
            )
            .ok_or_else(|| {
                let scale = row
                    .htt
                    .diag()
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0);
                ArrowSchurGpuFailure::RidgeBumpRequired {
                    row: row_idx,
                    bump: scale * f64::EPSILON.sqrt() * super::RIDGE_BUMP_EPS_MARGIN,
                }
            })?;
            for col in 0..q {
                let mut e = Array1::<f64>::zeros(q);
                e[col] = 1.0;
                let solved =
                    crate::linalg::triangular::cholesky_solve_vector(factor.view(), e.view());
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
            // B_i in P_MAX×R_TEMPLATE strided block.
            for col in 0..k {
                let base = (i * p_max + col) * p_max;
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

    pub(super) fn solve_fused(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let plan = crate::gpu::kernels::arrow_schur_nvrtc::plan_fused_launch(n, d, k)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let p_max = plan.p_max;
        let r_template = plan.r_template;

        let runtime = crate::gpu::linalg_dispatch::route_through_gpu(
            crate::gpu::linalg_dispatch::DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n },
        )
        .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = crate::gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let cap = &runtime.device.capability;
        let key = crate::gpu::kernels::arrow_schur_nvrtc::FusedModuleCacheKey {
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
            let pivot = status_host[row];
            let scale = sys.rows[row]
                .htt
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                row,
                bump: scale * (pivot.abs() as f64).max(1.0) * f64::EPSILON.sqrt() * 1024.0,
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
    ) -> Result<crate::solver::arrow_schur::GpuSchurMatvec, super::ArrowSchurGpuFailure> {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let plan = crate::gpu::kernels::arrow_schur_nvrtc::plan_fused_launch(n, d, k)
            .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let p_max = plan.p_max;
        let r_template = plan.r_template;

        let runtime = crate::gpu::linalg_dispatch::route_through_gpu(
            crate::gpu::linalg_dispatch::DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n },
        )
        .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let ctx = crate::gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let cap = &runtime.device.capability;
        let key = crate::gpu::kernels::arrow_schur_nvrtc::FusedModuleCacheKey {
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
            let pivot = status_host[row];
            let scale = sys.rows[row]
                .htt
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            return Err(super::ArrowSchurGpuFailure::RidgeBumpRequired {
                row,
                bump: scale * (pivot.abs() as f64).max(1.0) * f64::EPSILON.sqrt() * 1024.0,
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

        let closure: crate::solver::arrow_schur::GpuSchurMatvec =
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

    pub(super) fn solve_sae_matrix_free_pcg(
        sys: &ArrowSchurSystem,
        data: &DeviceSaePcgData,
        ridge_t: f64,
        ridge_beta: f64,
        rhs_beta: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurGpuFailure> {
        let k = rhs_beta.len();
        if k == 0 || data.beta_dim != k || sys.k != k {
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
        let runtime = crate::gpu::device_runtime::GpuRuntime::global()
            .filter(|rt| {
                rt.policy().reduced_schur_matvec_should_offload(
                    sys.rows.len(),
                    sys.k,
                    sys.d,
                    max_iterations,
                )
            })
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = crate::gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let vector_module = pcg_vector_module(&ctx)?;
        let mut buffers = flatten_device_sae_data(sys, data, ridge_t, &stream)?;

        let rhs_norm = rhs_beta.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rhs_norm == 0.0 {
            return Ok((Array1::<f64>::zeros(k), PcgDiagnostics::default()));
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
        let mut diag = PcgDiagnostics {
            precond_apply_calls: 1,
            stopping_reason: PcgStopReason::MaxIter,
            ..PcgDiagnostics::default()
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
    ) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurGpuFailure> {
        let k = rhs_beta.len();
        let runtime = crate::gpu::linalg_dispatch::route_through_gpu(
            crate::gpu::linalg_dispatch::DispatchOp::Gemv { m: k, k },
        )
        .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = crate::gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
            .and_then(|ctx| ctx.new_stream().ok())
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let ctx = crate::gpu::device_runtime::cuda_context_for(runtime.device.ordinal)
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
            return Ok((Array1::<f64>::zeros(k), PcgDiagnostics::default()));
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
        let mut diag = PcgDiagnostics {
            precond_apply_calls: 1,
            stopping_reason: PcgStopReason::MaxIter,
            ..PcgDiagnostics::default()
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::arrow_schur::ArrowSchurSystem;
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
        let Ok((device, diag)) =
            solve_reduced_beta_pcg_with_diagnostics(&s, &rhs, max_iterations, relative_tolerance)
        else {
            return;
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
    /// to ULP-scale chunk reassociation, so the criterion ranking across
    /// topology candidates cannot move (the #1017 verification gate).
    #[test]
    fn row_procedural_matvec_parallel_deterministic_and_matches_serial() {
        use crate::solver::arrow_schur::SCHUR_MATVEC_PARALLEL_ROW_MIN;
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
}
