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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::solver::arrow_schur::ArrowSchurSystem;

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
        // Layer D admission: when the system shape passes the
        // (Σ p³ ≥ 1e5 OR R ≥ 16) heuristic and `p ≤ MAX_FUSED_P`, the fused
        // NVRTC kernel replaces the cuSOLVER/cuBLAS Layer A+B+C path with a
        // single per-row block. Layer C↔D parity (math block 3 §16 test 6)
        // requires both paths to agree to 1e-10 on identical inputs.
        if crate::gpu::arrow_schur_nvrtc::system_admits_fused_path(sys) {
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
    if crate::gpu::arrow_schur_nvrtc::plan_fused_launch(sys.rows.len(), sys.d, sys.k).is_none() {
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
        let factor = cholesky_lower_host(block.view()).ok_or_else(|| {
            let scale = row
                .htt
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            ArrowSchurGpuFailure::RidgeBumpRequired {
                row: i,
                bump: scale * f64::EPSILON.sqrt() * 1024.0,
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
            let mut neg = Array1::<f64>::zeros(k);
            for i in 0..n {
                let di = row_dims[i];
                // v_i = H_tβ^(i)·x (sparse Kronecker gather, length d_i).
                let mut v_i = Array1::<f64>::zeros(di);
                forward(i, x.view(), &mut v_i);
                // w_i = (H_tt^(i) + ρ_t·I)^{-1} v_i via L_i L_iᵀ.
                let w_i = solve_cholesky_lower_host(factors[i].view(), v_i.view());
                // neg += H_βt^(i)·w_i (sparse scatter); subtract once at the end.
                transpose(i, w_i.view(), &mut neg);
            }
            for a in 0..k {
                out[a] -= neg[a];
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
/// The dense `S·p` matvec runs on device via cuBLAS `Dgemv` (the `O(K²)` term
/// that dwarfs the `O(K)` host-side CG scalar recurrences), and the Jacobi
/// preconditioner uses `diag(S)` extracted once on the host after a single
/// `dtoh` of the diagonal. Only the `K`-vectors `p` and `S·p` cross the
/// host↔device boundary per CG iteration.
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
        cuda::solve_reduced_beta_pcg(s_acc, rhs_beta, max_iterations, relative_tolerance)
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
    let factor = cholesky_lower_host(h.view())
        .ok_or_else(|| "dense reference Cholesky failed".to_string())?;
    let mut log_det = 0.0_f64;
    for i in 0..total {
        log_det += factor[[i, i]].ln();
    }
    log_det *= 2.0;
    let solved = solve_cholesky_lower_host(factor.view(), rhs.view());
    let delta_t = solved.slice(ndarray::s![..n * d]).to_owned();
    let delta_beta = solved.slice(ndarray::s![n * d..]).to_owned();
    Ok(ArrowSchurGpuSolution {
        delta_t,
        delta_beta,
        log_det_hessian: log_det,
    })
}

#[inline]
fn cholesky_lower_host(a: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[[i, i]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

#[inline]
fn solve_cholesky_lower_host(l: ArrayView2<'_, f64>, rhs: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

#[cfg(target_os = "linux")]
mod cuda {
    use super::{ArrowSchurGpuFailure, ArrowSchurGpuSolution, pack_host};
    use crate::gpu::driver::to_i32;
    use crate::gpu::linalg::{DispatchOp, route_through_gpu};
    use crate::solver::arrow_schur::ArrowSchurSystem;
    use cudarc::cublas::sys::{
        cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use ndarray::Array1;
    use std::sync::Arc;

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

        let stream = crate::gpu::runtime::cuda_context_for(runtime.device.ordinal)
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
    // n·R² ≲ 5M doubles at biobank scale), assembles `S_β`, factors it via
    // cuSOLVER, and launches the back-substitution kernel
    // `arrow_schur_back_sub_pgroup` to recover `δt_i = -L_i⁻ᵀ(u_i + Y_i δβ)`
    // without re-uploading the local factors.
    // ────────────────────────────────────────────────────────────────────

    use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    /// One compiled NVRTC module per `(cc_major, cc_minor, p_max, r_template)`.
    /// `cc_*` lets one process drive multiple device generations; the
    /// `(p_max, r_template)` pair selects the shared-memory layout baked into
    /// the kernel source.
    struct FusedModuleCache {
        modules:
            Mutex<HashMap<crate::gpu::arrow_schur_nvrtc::FusedModuleCacheKey, Arc<CudaModule>>>,
    }

    fn fused_module_cache() -> &'static FusedModuleCache {
        static CACHE: OnceLock<FusedModuleCache> = OnceLock::new();
        CACHE.get_or_init(|| FusedModuleCache {
            modules: Mutex::new(HashMap::new()),
        })
    }

    fn fused_module_for(
        ctx: &Arc<CudaContext>,
        key: crate::gpu::arrow_schur_nvrtc::FusedModuleCacheKey,
    ) -> Result<Arc<CudaModule>, ArrowSchurGpuFailure> {
        let cache = fused_module_cache();
        if let Ok(guard) = cache.modules.lock() {
            if let Some(existing) = guard.get(&key) {
                return Ok(existing.clone());
            }
        }
        let src = crate::gpu::arrow_schur_nvrtc::forward_kernel_source(
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
        let plan = crate::gpu::arrow_schur_nvrtc::plan_fused_launch(n, d, k)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let p_max = plan.p_max;
        let r_template = plan.r_template;

        let runtime = crate::gpu::linalg::route_through_gpu(
            crate::gpu::linalg::DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n },
        )
        .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let ctx = crate::gpu::runtime::cuda_context_for(runtime.device.ordinal)
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let cap = &runtime.device.capability;
        let key = crate::gpu::arrow_schur_nvrtc::FusedModuleCacheKey {
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
        let plan = crate::gpu::arrow_schur_nvrtc::plan_fused_launch(n, d, k)
            .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let p_max = plan.p_max;
        let r_template = plan.r_template;

        let runtime = crate::gpu::linalg::route_through_gpu(
            crate::gpu::linalg::DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n },
        )
        .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let ctx = crate::gpu::runtime::cuda_context_for(runtime.device.ordinal)
            .ok_or(super::ArrowSchurGpuFailure::Unavailable)?;
        let stream = ctx
            .new_stream()
            .map_err(|_| super::ArrowSchurGpuFailure::Unavailable)?;
        let cap = &runtime.device.capability;
        let key = crate::gpu::arrow_schur_nvrtc::FusedModuleCacheKey {
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

    /// Jacobi-preconditioned conjugate-gradient solve of the dense reduced
    /// β-system `S·δβ = r` fully on device.
    ///
    /// `S` (`k × k`, symmetric positive definite) and `r` (`k`) are uploaded
    /// once. Each CG iteration evaluates `S·p` via cuBLAS `Dgemv` device-side
    /// (the `O(k²)` cost that dominates at `k = 100K`), downloads the `k`-vector
    /// result, and runs the scalar CG recurrences on the host (the `O(k)` dot
    /// products and `axpy`s are negligible beside the matvec). The Jacobi
    /// preconditioner `M^{-1} = diag(S)^{-1}` is extracted once from a single
    /// diagonal `dtoh`.
    ///
    /// Returns `Unavailable` when the workload is below the GEMV dispatch
    /// policy or no CUDA context is reachable, and `SchurFactorFailed` when the
    /// Jacobi diagonal is not strictly positive (an indefinite reduced system
    /// the caller escalates with a proximal ridge).
    pub(super) fn solve_reduced_beta_pcg(
        s_acc: &ndarray::Array2<f64>,
        rhs_beta: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<Array1<f64>, ArrowSchurGpuFailure> {
        let k = rhs_beta.len();
        let runtime = crate::gpu::linalg::route_through_gpu(
            crate::gpu::linalg::DispatchOp::Gemv { m: k, k },
        )
        .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let stream = crate::gpu::runtime::cuda_context_for(runtime.device.ordinal)
            .and_then(|ctx| ctx.new_stream().ok())
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

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
            return Ok(Array1::<f64>::zeros(k));
        }
        let tol = (relative_tolerance.max(0.0) * rhs_norm).max(1e-12);

        let mut x = vec![0.0_f64; k];
        let mut r: Vec<f64> = rhs_beta.iter().copied().collect();
        let mut z: Vec<f64> = (0..k).map(|j| inv_diag[j] * r[j]).collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let mut p_dev = stream
            .clone_htod(&p)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut sp_dev = stream
            .alloc_zeros::<f64>(k)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        let max_iters = max_iterations.max(1);
        for _ in 0..max_iters {
            // sp = S · p (device GEMV, S column-major k×k, op = N).
            stream
                .memcpy_htod(&p, &mut p_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
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
            let sp = stream
                .clone_dtoh(&sp_dev)
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

            let p_sp: f64 = p.iter().zip(&sp).map(|(a, b)| a * b).sum();
            if !(p_sp > 0.0) {
                // Non-positive curvature on a (proximal-ridged) SPD system means
                // numerical breakdown; surface so the caller escalates.
                return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                    reason: format!("reduced-β GPU PCG: non-positive curvature pᵀSp={p_sp:e}"),
                });
            }
            let alpha = rz / p_sp;
            for j in 0..k {
                x[j] += alpha * p[j];
                r[j] -= alpha * sp[j];
            }
            let r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= tol {
                break;
            }
            for j in 0..k {
                z[j] = inv_diag[j] * r[j];
            }
            let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rz_new / rz;
            for j in 0..k {
                p[j] = z[j] + beta * p[j];
            }
            rz = rz_new;
        }

        Ok(Array1::from_vec(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::arrow_schur::ArrowSchurSystem;
    use ndarray::Array2;

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
        let l = cholesky_lower_host(h.view()).unwrap();
        let rhs = g.mapv(|v| -v);
        let expected = solve_cholesky_lower_host(l.view(), rhs.view());
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
}
