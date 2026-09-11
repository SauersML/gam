//! Single preconditioned conjugate-gradient (PCG) core.
//!
//! Both the CPU SPD solver (`linalg::utils::solve_spd_pcg_with_info`, parallel,
//! residual-refresh, diagnostics) and the GPU REML trace solver
//! (`gpu::kernels::reml_trace::cg_solve`, serial, no refresh, no diagnostics)
//! historically carried their own hand-rolled CG loop. They drifted: the GPU
//! copy accepted a partial solution on lost SPD, while the CPU copy rejected
//! non-positive preconditioner diagonals and refreshed the residual every 32
//! iterations. The shared inner
//! recurrence — `alpha = rz/pᵀAp`, `x += alpha p`, `r -= alpha Ap`,
//! `beta = rz'/rz`, `p = z + beta p` — is identical.
//!
//! [`pcg_core`] is that one recurrence. The two callers are thin wrappers that
//! pick a refresh period, opt into diagnostics, and decide what a breakdown
//! means (the CPU rejects it as `None`; the GPU keeps the partial iterate).
//!
//! ## Numerics
//!
//! The inner products `rᵀz` and `pᵀAp` are accumulated **serially** (a plain
//! sequential fold). This is deliberate: it makes every iterate bit-identical
//! regardless of the host's thread count, which is what lets the GPU wrapper
//! reproduce the byte-for-byte iterates of the old serial `cg_solve`. The
//! *elementwise* O(p) vector updates (preconditioner apply and the fused
//! `p`-axpy) are reduction-free and therefore parallelized over the coefficient
//! dimension without perturbing the result, preserving the CPU solver's
//! large-`p` parallelism.

use ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};
use rayon::prelude::*;
use std::sync::Arc;

/// The smallest relative residual a `rows`-dimensional CG recurrence can resolve.
///
/// The residual is a `rows`-term accumulation, so two residuals that differ by
/// less than `γ_rows` relative to the right-hand side differ by arithmetic, not
/// by progress. A requested relative tolerance below this band is raised to it;
/// the band belongs to the arithmetic, never to the caller's problem.
fn attainable_relative_residual(rows: usize) -> f64 {
    crate::roundoff::accumulation_growth(rows)
}

/// Per-iteration trace of the PCG recurrence, sufficient to reconstruct the
/// Lanczos tridiagonal and hence Ritz-based condition estimates. Populated only
/// when the caller requests diagnostics.
#[derive(Debug, Clone)]
pub struct PcgDiagnostics {
    pub residuals: Vec<f64>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl PcgDiagnostics {
    fn new(initial_residual_norm: f64) -> Self {
        Self {
            residuals: vec![initial_residual_norm],
            alpha: Vec::new(),
            beta: Vec::new(),
        }
    }

    fn push_iteration(&mut self, alpha: f64, beta: Option<f64>, residual_norm: f64) {
        self.alpha.push(alpha);
        if let Some(beta) = beta {
            self.beta.push(beta);
        }
        self.residuals.push(residual_norm);
    }
}

/// Why the core stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcgStop {
    /// `‖r‖ ≤ tol`; the recorded iterate is the converged solution.
    Converged,
    /// Hit `max_iters` without reaching tolerance.
    MaxIters,
    /// Lost SPD or hit a non-finite scalar (e.g. `pᵀAp ≤ 0`, non-finite
    /// `alpha`/`beta`, a non-positive `rᵀz`, or a mismatched matvec length).
    /// The iterate written so far is the last numerically valid one; callers
    /// decide whether to keep it (GPU) or reject the whole solve (CPU).
    Breakdown,
    /// The preconditioner diagonal contained a non-positive or non-finite entry,
    /// violating the SPD-PCG contract (`M ≻ 0`). Detected before any iteration;
    /// the solution buffer is untouched (left at the zero initial guess).
    BadPreconditioner,
}

/// Result of a [`pcg_core`] run. The solution is written into the caller's
/// buffer; this carries the metadata about how the run terminated.
#[derive(Debug, Clone)]
pub struct PcgCoreResult {
    pub stop: PcgStop,
    pub iterations: usize,
    pub rhs_norm: f64,
    pub final_residual_norm: f64,
    pub diagnostics: Option<PcgDiagnostics>,
}

/// How the PCG inner products `rᵀz` and `pᵀAp` are accumulated.
///
/// This is the single knob that distinguishes the bit-reproducible main solve
/// from the stochastic trace probe. It is NOT a performance hint the optimizer
/// may ignore: it selects between two numerically distinct reductions, and the
/// caller is responsible for picking the one its contract allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotReduction {
    /// Strict left-to-right sequential fold. Bit-identical regardless of host
    /// thread count or SIMD width — this is what lets the GPU wrapper reproduce
    /// the byte-for-byte iterates of the old serial `cg_solve`, and what the
    /// inexact-Newton CPU solver relies on for run-to-run determinism. Latency
    /// bound: every add chains on the previous one (a single FP accumulator),
    /// so there is no add-side ILP. REQUIRED for the main solve.
    Serial,
    /// Associativity-reordered reduction (independent ILP accumulators, SIMD).
    /// The result differs from [`DotReduction::Serial`] in the low bits because
    /// floating-point addition is not associative. ONLY valid for callers that
    /// are already stochastic and loose-tolerance — the Hutchinson REML trace
    /// probes, whose per-probe CG residual (≈1e-6) sits orders of magnitude
    /// below the estimator's own sampling SE, so the reorder is dominated by
    /// Monte-Carlo noise the adaptive-K stopping rule already absorbs. MUST NOT
    /// be used where cross-thread / run-to-run bit-identity is contractual.
    Reordered,
}

/// Strict sequential inner product. A plain left-to-right fold so the result is
/// independent of thread count and SIMD width — see the module-level numerics
/// note. This is the bit-reproducible reduction used by the main solve.
#[inline]
fn serial_dot(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let mut acc = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// Associativity-reordered inner product with eight independent accumulators.
///
/// Each lane carries its own running sum so the eight partial chains pipeline
/// instead of serializing on one FP register; the optimizer is then free to
/// fold the per-lane multiply-accumulates into SIMD FMAs. The eight lanes are
/// combined pairwise at the end. The result differs from [`serial_dot`] in the
/// low mantissa bits (FP add is non-associative) — that is the whole point, and
/// it is ONLY acceptable on the stochastic trace path, never the main solve.
#[inline]
fn reordered_dot(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    // Contiguous fast path (the trace probe always hands contiguous vectors);
    // fall back to the iterator form for any strided view.
    match (a.as_slice(), b.as_slice()) {
        (Some(av), Some(bv)) => {
            const LANES: usize = 8;
            let n = av.len().min(bv.len());
            let mut acc = [0.0_f64; LANES];
            let chunks = n / LANES;
            for c in 0..chunks {
                let base = c * LANES;
                // Each lane is an independent dependency chain.
                for l in 0..LANES {
                    acc[l] += av[base + l] * bv[base + l];
                }
            }
            // Pairwise lane combine (balanced tree, not a serial sweep).
            let mut s =
                ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for i in (chunks * LANES)..n {
                s += av[i] * bv[i];
            }
            s
        }
        _ => serial_dot(a, b),
    }
}

/// Dispatch the configured inner-product reduction.
#[inline]
fn dot(a: &ArrayView1<f64>, b: &ArrayView1<f64>, reduction: DotReduction) -> f64 {
    match reduction {
        DotReduction::Serial => serial_dot(a, b),
        DotReduction::Reordered => reordered_dot(a, b),
    }
}

/// The shared preconditioned conjugate-gradient recurrence.
///
/// Solves `A x = rhs` for SPD `A`, accessed only through `apply(v, out)` which
/// must set `out <- A v`. The initial guess is `x = 0`. Convergence target is
/// `‖r‖ ≤ max(rel_tol, γ_p) · ‖rhs‖`: a textbook RELATIVE residual criterion,
/// with the relative target raised to the residual's own rounding band `γ_p`
/// (see `attainable_relative_residual`) so it never chases progress the
/// arithmetic cannot show. A zero rhs has the exact solution `x = 0`.
/// Inexact-Newton callers (e.g. Eisenstat–Walker forcing for the joint
/// PIRLS solver) rely on this relative contract: the historical
/// `max(‖rhs‖, 1)` factor silently inflated the threshold to an absolute
/// `rel_tol` whenever `‖rhs‖ < 1`, so a request like `η = 0.1` on a
/// sub-unit gradient produced an effective relative residual far above
/// `η` and trapped the outer Newton loop in a fixed-point oscillation.
///
/// * `precond_diag` — diagonal Jacobi preconditioner `M`; pass all-ones for an
///   unpreconditioned solve. A non-positive or non-finite entry, or one whose
///   reciprocal overflows, is a contract violation reported as
///   [`PcgStop::BadPreconditioner`].
/// * `refresh_period` — recompute `r ← rhs − A x` every `refresh_period`
///   iterations to shed accumulated round-off; `0` disables refresh entirely
///   (matching the GPU serial path).
/// * `record_diagnostics` — when `true`, populate [`PcgCoreResult::diagnostics`]
///   with the per-iteration `alpha`/`beta`/residual trace.
///
/// The solution iterate is written into `solution` (which must have the same
/// length as `rhs`). On [`PcgStop::Converged`]/[`PcgStop::MaxIters`]/
/// [`PcgStop::Breakdown`] it holds the last valid iterate; on
/// [`PcgStop::BadPreconditioner`] it is left as the zero initial guess.
pub fn pcg_core<F>(
    mut apply: F,
    rhs: &ArrayView1<f64>,
    precond_diag: &ArrayView1<f64>,
    rel_tol: f64,
    max_iters: usize,
    refresh_period: usize,
    record_diagnostics: bool,
    reduction: DotReduction,
    solution: &mut ArrayViewMut1<f64>,
) -> PcgCoreResult
where
    F: FnMut(&Array1<f64>, &mut Array1<f64>),
{
    let p = rhs.len();
    let rhs_norm = dot(rhs, rhs, reduction).sqrt();

    solution.fill(0.0);
    let mut diagnostics = record_diagnostics.then(|| PcgDiagnostics::new(rhs_norm));
    if precond_diag.len() != p || solution.len() != p {
        return PcgCoreResult {
            stop: PcgStop::Breakdown,
            iterations: 0,
            rhs_norm,
            final_residual_norm: rhs_norm,
            diagnostics,
        };
    }

    let mut x = Array1::<f64>::zeros(p);

    if !rhs_norm.is_finite() {
        return PcgCoreResult {
            stop: PcgStop::Breakdown,
            iterations: 0,
            rhs_norm,
            final_residual_norm: rhs_norm,
            diagnostics,
        };
    }
    if rhs_norm == 0.0 {
        return PcgCoreResult {
            stop: PcgStop::Converged,
            iterations: 0,
            rhs_norm: 0.0,
            final_residual_norm: 0.0,
            diagnostics,
        };
    }

    // Textbook PCG relative-residual criterion: ‖r‖ ≤ rel_tol · ‖rhs‖, with the
    // relative target no tighter than the residual's rounding band (the
    // early-exit above handles rhs_norm == 0).
    let tol = rel_tol.max(attainable_relative_residual(p)) * rhs_norm;

    // Precompute reciprocal preconditioner once: z = inv_m * r per iteration.
    // SPD-PCG requires M ≻ 0; a non-positive/non-finite entry is a contract
    // violation surfaced as BadPreconditioner rather than silently abs()-ed.
    let mut inv_m = Array1::<f64>::zeros(p);
    let mut bad_diag = false;
    for (slot, &m) in inv_m.iter_mut().zip(precond_diag.iter()) {
        if !m.is_finite() || m <= 0.0 {
            bad_diag = true;
            break;
        }
        *slot = 1.0 / m;
        if !slot.is_finite() {
            bad_diag = true;
            break;
        }
    }
    if bad_diag {
        return PcgCoreResult {
            stop: PcgStop::BadPreconditioner,
            iterations: 0,
            rhs_norm,
            final_residual_norm: rhs_norm,
            diagnostics,
        };
    }

    let mut r = rhs.to_owned();
    let mut z = Array1::<f64>::zeros(p);
    Zip::from(&mut z)
        .and(&r)
        .and(&inv_m)
        .par_for_each(|zi, &ri, &im| {
            *zi = ri * im;
        });
    let mut p_dir = z.clone();
    let mut rz_old = dot(&r.view(), &z.view(), reduction);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return PcgCoreResult {
            stop: PcgStop::Breakdown,
            iterations: 0,
            rhs_norm,
            final_residual_norm: rhs_norm,
            diagnostics,
        };
    }

    let mut ap = Array1::<f64>::zeros(p);
    let mut last_r_norm = rhs_norm;

    for iter in 0..max_iters {
        apply(&p_dir, &mut ap);
        if ap.len() != p {
            return PcgCoreResult {
                stop: PcgStop::Breakdown,
                iterations: iter,
                rhs_norm,
                final_residual_norm: last_r_norm,
                diagnostics,
            };
        }
        let denom = dot(&p_dir.view(), &ap.view(), reduction);
        if !denom.is_finite() || denom <= 0.0 {
            return PcgCoreResult {
                stop: PcgStop::Breakdown,
                iterations: iter,
                rhs_norm,
                final_residual_norm: last_r_norm,
                diagnostics,
            };
        }
        let alpha = rz_old / denom;
        if !alpha.is_finite() {
            return PcgCoreResult {
                stop: PcgStop::Breakdown,
                iterations: iter,
                rhs_norm,
                final_residual_norm: last_r_norm,
                diagnostics,
            };
        }
        x.scaled_add(alpha, &p_dir);
        solution.assign(&x);
        r.scaled_add(-alpha, &ap);
        if refresh_period != 0 && (iter + 1) % refresh_period == 0 {
            // Periodic residual refresh: r <- rhs - A x. Reuse `ap` as scratch
            // for A x to avoid an extra allocation.
            apply(&x, &mut ap);
            if ap.len() != p {
                return PcgCoreResult {
                    stop: PcgStop::Breakdown,
                    iterations: iter + 1,
                    rhs_norm,
                    final_residual_norm: last_r_norm,
                    diagnostics,
                };
            }
            r.assign(rhs);
            r.scaled_add(-1.0, &ap);
        }
        let r_norm = dot(&r.view(), &r.view(), reduction).sqrt();
        last_r_norm = r_norm;
        if r_norm.is_finite() && r_norm <= tol {
            if let Some(d) = diagnostics.as_mut() {
                d.push_iteration(alpha, None, r_norm);
            }
            return PcgCoreResult {
                stop: PcgStop::Converged,
                iterations: iter + 1,
                rhs_norm,
                final_residual_norm: r_norm,
                diagnostics,
            };
        }
        Zip::from(&mut z)
            .and(&r)
            .and(&inv_m)
            .par_for_each(|zi, &ri, &im| {
                *zi = ri * im;
            });
        let rz_new = dot(&r.view(), &z.view(), reduction);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            return PcgCoreResult {
                stop: PcgStop::Breakdown,
                iterations: iter + 1,
                rhs_norm,
                final_residual_norm: r_norm,
                diagnostics,
            };
        }
        let beta = rz_new / rz_old;
        if !beta.is_finite() {
            return PcgCoreResult {
                stop: PcgStop::Breakdown,
                iterations: iter + 1,
                rhs_norm,
                final_residual_norm: r_norm,
                diagnostics,
            };
        }
        if let Some(d) = diagnostics.as_mut() {
            d.push_iteration(alpha, Some(beta), r_norm);
        }
        // p <- z + beta * p (fused, SIMD-friendly via ndarray::Zip; parallel
        // over the coefficient dimension at large-scale p).
        Zip::from(&mut p_dir).and(&z).par_for_each(|pi, &zi| {
            *pi = zi + beta * *pi;
        });
        rz_old = rz_new;
    }

    PcgCoreResult {
        stop: PcgStop::MaxIters,
        iterations: max_iters,
        rhs_norm,
        final_residual_norm: last_r_norm,
        diagnostics,
    }
}

// ============================ Multi-RHS block CG ============================
//
// A single SPD operator solved against MANY right-hand sides at once — the
// shape of the sparse-dictionary decoder refresh (#1017), where one giant
// co-firing component's normal-equation operator must be solved for every one
// of `P` decoder columns. Solving the columns one at a time re-traverses the
// operator's sparse structure per column per iteration; at the measured
// production shape (K=32000, P=2048) that redundant structure traffic alone is
// petabytes and was the entire epoch wall (#1017, 69,174 s serial refresh next
// to 13.9 s of routed device compute). The block core below advances ALL
// columns together off ONE operator application per iteration, so the operator
// is streamed once per iteration regardless of the column count.
//
// ## Contract: one mathematically exact preconditioned recurrence
//
// [`pcg_multi_core`] advances independently preconditioned columns from an
// arbitrary exact initial state `X`, `R = B - A·X`, `Z = M⁻¹R`, `P = Z`.
// Every per-column inner product is a strict ascending-row fold, every vector
// update performs the same multiply-then-add sequence in the same order, and
// each column carries its own `alpha`/`beta`/convergence state. A converged (or
// broken-down) column freezes while the remaining columns continue. Backend
// parity is therefore a primitive-by-primitive contract rather than an
// accident of a zero initial guess.

/// Fixed column-tile width for the deterministic block inner products: one
/// cache line of `f64`s. Each tile's accumulators are private to one task and
/// every column's fold is strict ascending-row regardless of tiling, so this
/// constant affects performance only, never a single result bit.
const BLOCK_DOT_COLUMN_TILE: usize = 8;

/// Backend contract for [`pcg_multi_core`]: owns the block iterate state
/// `X` (solution), `R` (residual), `Z = M⁻¹R` (preconditioned residual), `P`
/// (search direction), and `AP` (operator image), each logically
/// `rows × columns`, plus the operator itself.
///
/// Numerical obligations (what makes a backend admissible):
/// * `apply_block` sets `AP ← A·P` where column `c` of `AP` is EXACTLY the
///   operator applied to column `c` of `P` — same summation order as the
///   scalar operator the backend claims to represent. Frozen columns may be
///   recomputed (their values are never read back into the recurrence).
/// * every dot primitive writes, per column, a STRICT ascending-row fold
///   `acc = fold(acc + a[i]·b[i])` — the [`DotReduction::Serial`] contract;
/// * `update_x_r` performs, for each active column `c`,
///   `X[·][c] += alpha[c]·P[·][c]` then `R[·][c] += (-alpha[c])·AP[·][c]`
///   as separate multiply-then-add per element (no FMA contraction).
/// * `refresh_preconditioned_residual` sets `Z ← M⁻¹R`.
/// * `update_p` performs, for each active column `c`,
///   `P[·][c] = Z[·][c] + beta[c]·P[·][c]` (multiply-then-add, no FMA).
///
/// The recurrence itself (scalar `alpha`/`beta` math, convergence and
/// breakdown decisions, diagnostics) lives in [`pcg_multi_core`] and is shared
/// by every backend, so a device implementation cannot drift from the CPU one.
pub trait PcgBlockBackend {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    /// Squared norm of the original right-hand side, not the initial residual.
    fn rhs_norm_squared(&mut self, out: &mut [f64]);
    /// `AP ← A·P` for all columns.
    fn apply_block(&mut self);
    /// `out[c] ← Σ_i P[i][c]·AP[i][c]`, strict ascending-`i` fold per column.
    fn dot_p_ap(&mut self, out: &mut [f64]);
    /// `out[c] ← Σ_i R[i][c]²`, strict ascending-`i` fold per column.
    fn dot_r_r(&mut self, out: &mut [f64]);
    /// `out[c] ← Σ_i R[i][c]·Z[i][c]`, strict ascending-`i` fold per column.
    fn dot_r_z(&mut self, out: &mut [f64]);
    /// Per active column: `X += alpha·P`, then `R += (-alpha)·AP`.
    fn update_x_r(&mut self, alpha: &[f64], active: &[bool]);
    /// Refresh `Z = M⁻¹R`.
    fn refresh_preconditioned_residual(&mut self);
    /// Per active column: `P = Z + beta·P`.
    fn update_p(&mut self, beta: &[f64], active: &[bool]);
}

/// Symmetric positive-definite block preconditioner with a Jacobi base and an
/// optional exact low-rank coarse correction.
///
/// The represented inverse is
///
/// ```text
/// H = D⁻¹ + C Qᵀ,
/// ```
///
/// where `D⁻¹` is the positive diagonal in `inverse_diagonal`. The only
/// positive-rank constructor accepts a scaled subspace and its operator action,
/// orthonormalizes it, proves the resulting Galerkin operator SPD by Cholesky,
/// and privately forms `Q` and `C` as
///
/// ```text
/// H = D⁻¹/² [I + U(E⁻¹ - I)Uᵀ] D⁻¹/²,
/// E = UᵀD⁻¹/² A D⁻¹/²U,
/// ```
///
/// with internally orthonormalized `U`. Consequently `H ≻ 0` is an invariant of
/// the type rather than an unchecked caller promise: `E⁻¹ ≻ 0` on `span(U)` and
/// the identity acts on its orthogonal complement. The private factors are
/// shared so every RHS tile can reuse one setup without copying `rows × rank`
/// storage.
#[derive(Clone)]
pub struct SymmetricLowRankPreconditioner {
    inverse_diagonal: Arc<[f64]>,
    projection: Arc<ndarray::Array2<f64>>,
    correction: Arc<ndarray::Array2<f64>>,
}

impl SymmetricLowRankPreconditioner {
    /// Pure Jacobi (`rank = 0`).
    pub fn jacobi(inverse_diagonal: Vec<f64>) -> Self {
        assert!(
            inverse_diagonal.iter().all(|d| d.is_finite() && *d > 0.0),
            "PCG inverse diagonal must be finite and positive"
        );
        let rows = inverse_diagonal.len();
        Self {
            inverse_diagonal: inverse_diagonal.into(),
            projection: Arc::new(ndarray::Array2::zeros((rows, 0))),
            correction: Arc::new(ndarray::Array2::zeros((rows, 0))),
        }
    }

    /// Build Jacobi plus the exact inverse of a scaled Galerkin coarse operator.
    ///
    /// `scaled_candidates` contains candidate directions in the coordinates of
    /// `S = D⁻¹/² A D⁻¹/²`. This method rank-reveals and orthonormalizes them,
    /// passes the resulting `U` to `apply_scaled` to obtain `S U`, and admits a
    /// positive-rank preconditioner only after Cholesky has proved
    /// `E = Uᵀ S U` SPD. Thus callers never supply the private `C,Q` factors and
    /// cannot construct a nonsymmetric or indefinite `H`; dependent candidates
    /// cannot smuggle an arbitrary QR completion into the represented subspace.
    pub fn from_scaled_subspace<F>(
        inverse_diagonal: Vec<f64>,
        scaled_candidates: ndarray::Array2<f64>,
        apply_scaled: F,
    ) -> Result<Self, String>
    where
        F: FnOnce(&ndarray::Array2<f64>, &mut ndarray::Array2<f64>),
    {
        use crate::faer_ndarray::{
            FaerCholesky, FaerQr, default_rrqr_rank_alpha, fast_ab_into, fast_atb_with_parallelism,
            matmul_parallelism, rrqr_with_permutation,
        };
        use faer::Side;

        let rows = inverse_diagonal.len();
        if !inverse_diagonal.iter().all(|d| d.is_finite() && *d > 0.0) {
            return Err("PCG inverse diagonal must be finite and positive".to_string());
        }
        if scaled_candidates.nrows() != rows {
            return Err(format!(
                "PCG scaled-subspace row count {} does not match diagonal length {rows}",
                scaled_candidates.nrows()
            ));
        }
        if scaled_candidates.ncols() == 0 {
            return Err("PCG scaled subspace must contain at least one candidate".to_string());
        }
        if !scaled_candidates.iter().all(|value| value.is_finite()) {
            return Err("PCG scaled-subspace candidates must be finite".to_string());
        }

        let rrqr = rrqr_with_permutation(&scaled_candidates, default_rrqr_rank_alpha())
            .map_err(|err| format!("PCG scaled-subspace RRQR failed: {err}"))?;
        let rank = rrqr.rank;
        if rank == 0 {
            return Err("PCG scaled subspace has numerical rank zero".to_string());
        }
        let mut independent = ndarray::Array2::<f64>::zeros((rows, rank));
        for q in 0..rank {
            for i in 0..rows {
                independent[[i, q]] = scaled_candidates[[i, rrqr.column_permutation[q]]];
            }
        }
        drop(scaled_candidates);
        let (mut scaled_basis, triangular) = independent
            .qr()
            .map_err(|err| format!("PCG scaled-subspace QR failed: {err}"))?;
        drop(independent);
        drop(triangular);
        if scaled_basis.dim() != (rows, rank) || !scaled_basis.iter().all(|value| value.is_finite())
        {
            return Err("PCG scaled-subspace QR did not produce a finite thin basis".to_string());
        }

        let mut image = ndarray::Array2::<f64>::zeros((rows, rank));
        apply_scaled(&scaled_basis, &mut image);
        if !image.iter().all(|value| value.is_finite()) {
            return Err("PCG scaled operator produced a non-finite coarse image".to_string());
        }

        let mut galerkin =
            fast_atb_with_parallelism(&scaled_basis, &image, matmul_parallelism(rank, rank, rows));
        drop(image);
        if !galerkin.iter().all(|value| value.is_finite()) {
            return Err("PCG Galerkin operator is non-finite".to_string());
        }
        let symmetry_factor =
            default_rrqr_rank_alpha() * f64::EPSILON * rows.max(rank).max(1) as f64;
        for a in 0..rank {
            for b in 0..a {
                let skew = (galerkin[[a, b]] - galerkin[[b, a]]).abs();
                let scale = galerkin[[a, b]].abs().max(galerkin[[b, a]].abs()).max(1.0);
                if skew > symmetry_factor * scale {
                    return Err(format!(
                        "PCG Galerkin operator is not symmetric at ({a},{b}): \
                         skew {skew:.3e} exceeds {:.3e}",
                        symmetry_factor * scale
                    ));
                }
                let value = 0.5 * (galerkin[[a, b]] + galerkin[[b, a]]);
                galerkin[[a, b]] = value;
                galerkin[[b, a]] = value;
            }
        }
        let factor = galerkin
            .cholesky(Side::Lower)
            .map_err(|err| format!("PCG Galerkin operator is not SPD at rank {rank}: {err}"))?;
        let mut inverse_correction = factor.solve_mat(&ndarray::Array2::<f64>::eye(rank));
        if !inverse_correction.iter().all(|value| value.is_finite()) {
            return Err("PCG Galerkin inverse is non-finite".to_string());
        }
        for a in 0..rank {
            for b in 0..a {
                let value = 0.5 * (inverse_correction[[a, b]] + inverse_correction[[b, a]]);
                inverse_correction[[a, b]] = value;
                inverse_correction[[b, a]] = value;
            }
            inverse_correction[[a, a]] -= 1.0;
        }

        // Convert the internally orthonormal U and U(E⁻¹-I) into physical
        // coordinates. Keeping these factors private makes every subsequent
        // hot-path application two deterministic contractions with no setup.
        for i in 0..rows {
            let inverse_sqrt = inverse_diagonal[i].sqrt();
            for q in 0..rank {
                scaled_basis[[i, q]] *= inverse_sqrt;
            }
        }
        let mut correction = ndarray::Array2::<f64>::zeros((rows, rank));
        fast_ab_into(&scaled_basis, &inverse_correction, &mut correction);
        if !correction.iter().all(|value| value.is_finite()) {
            return Err("PCG coarse correction is non-finite".to_string());
        }

        Ok(Self {
            inverse_diagonal: inverse_diagonal.into(),
            projection: Arc::new(scaled_basis),
            correction: Arc::new(correction),
        })
    }

    pub fn rows(&self) -> usize {
        self.inverse_diagonal.len()
    }

    pub fn rank(&self) -> usize {
        self.projection.ncols()
    }

    pub fn inverse_diagonal(&self) -> &[f64] {
        &self.inverse_diagonal
    }

    pub fn projection(&self) -> &ndarray::Array2<f64> {
        &self.projection
    }

    pub fn correction(&self) -> &ndarray::Array2<f64> {
        &self.correction
    }
}

/// Drive the shared CG recurrence over a block backend. Returns one
/// [`PcgCoreResult`] per column.
///
/// The backend must enter with `R = B - A·X`, `Z = M⁻¹R`, `P = Z`, and `AP`
/// arbitrary. On return, the backend's `X` holds each
/// column's final iterate: the converged solution for `Converged` columns, the
/// last numerically valid iterate for `Breakdown`/`MaxIters` columns.
pub fn pcg_multi_core<B: PcgBlockBackend>(
    backend: &mut B,
    rel_tol: f64,
    max_iters: usize,
    record_diagnostics: bool,
) -> Vec<PcgCoreResult> {
    let t = backend.columns();
    let rows = backend.rows();
    let mut rhs_squared = vec![0.0f64; t];
    let mut residual_squared = vec![0.0f64; t];
    let mut rz = vec![0.0f64; t];
    let mut scratch = vec![0.0f64; t];
    backend.rhs_norm_squared(&mut rhs_squared);
    backend.dot_r_r(&mut residual_squared);
    backend.dot_r_z(&mut rz);

    let mut done: Vec<Option<PcgCoreResult>> = vec![None; t];
    let mut active = vec![false; t];
    let mut rhs_norm = vec![0.0f64; t];
    let mut tol = vec![0.0f64; t];
    let mut rz_old = vec![0.0f64; t];
    let mut last_r_norm = vec![0.0f64; t];
    let mut diagnostics: Vec<Option<PcgDiagnostics>> = (0..t).map(|_| None).collect();

    for c in 0..t {
        let b_norm = rhs_squared[c].sqrt();
        let r_norm = residual_squared[c].sqrt();
        rhs_norm[c] = b_norm;
        last_r_norm[c] = r_norm;
        let diag = record_diagnostics.then(|| PcgDiagnostics::new(r_norm));
        if !b_norm.is_finite() || !r_norm.is_finite() {
            done[c] = Some(PcgCoreResult {
                stop: PcgStop::Breakdown,
                iterations: 0,
                rhs_norm: b_norm,
                final_residual_norm: r_norm,
                diagnostics: diag,
            });
            continue;
        }
        // Relative to the right-hand side; a zero rhs has no scale of its own,
        // so the entering residual `B − A·X` (whose exact solution is `X = 0`)
        // supplies it.
        let residual_scale = if b_norm > 0.0 { b_norm } else { r_norm };
        tol[c] = rel_tol.max(attainable_relative_residual(rows)) * residual_scale;
        if r_norm <= tol[c] {
            done[c] = Some(PcgCoreResult {
                stop: PcgStop::Converged,
                iterations: 0,
                rhs_norm: b_norm,
                final_residual_norm: r_norm,
                diagnostics: diag,
            });
            continue;
        }
        let initial_rz = rz[c];
        if !initial_rz.is_finite() || initial_rz <= 0.0 {
            done[c] = Some(PcgCoreResult {
                stop: PcgStop::Breakdown,
                iterations: 0,
                rhs_norm: b_norm,
                final_residual_norm: r_norm,
                diagnostics: diag,
            });
            continue;
        }
        rz_old[c] = initial_rz;
        active[c] = true;
        diagnostics[c] = diag;
    }

    let mut alpha = vec![0.0f64; t];
    let mut beta = vec![0.0f64; t];

    for iter in 0..max_iters {
        if !active.iter().any(|&a| a) {
            break;
        }
        backend.apply_block();
        backend.dot_p_ap(&mut scratch);
        for c in 0..t {
            if !active[c] {
                alpha[c] = 0.0;
                continue;
            }
            let denom = scratch[c];
            if !denom.is_finite() || denom <= 0.0 {
                active[c] = false;
                alpha[c] = 0.0;
                done[c] = Some(PcgCoreResult {
                    stop: PcgStop::Breakdown,
                    iterations: iter,
                    rhs_norm: rhs_norm[c],
                    final_residual_norm: last_r_norm[c],
                    diagnostics: diagnostics[c].take(),
                });
                continue;
            }
            let a = rz_old[c] / denom;
            if !a.is_finite() {
                active[c] = false;
                alpha[c] = 0.0;
                done[c] = Some(PcgCoreResult {
                    stop: PcgStop::Breakdown,
                    iterations: iter,
                    rhs_norm: rhs_norm[c],
                    final_residual_norm: last_r_norm[c],
                    diagnostics: diagnostics[c].take(),
                });
                continue;
            }
            alpha[c] = a;
        }
        backend.update_x_r(&alpha, &active);
        backend.dot_r_r(&mut residual_squared);
        backend.refresh_preconditioned_residual();
        backend.dot_r_z(&mut rz);
        for c in 0..t {
            if !active[c] {
                beta[c] = 0.0;
                continue;
            }
            let rr = residual_squared[c];
            let r_norm = rr.sqrt();
            last_r_norm[c] = r_norm;
            if r_norm.is_finite() && r_norm <= tol[c] {
                active[c] = false;
                beta[c] = 0.0;
                if let Some(d) = diagnostics[c].as_mut() {
                    d.push_iteration(alpha[c], None, r_norm);
                }
                done[c] = Some(PcgCoreResult {
                    stop: PcgStop::Converged,
                    iterations: iter + 1,
                    rhs_norm: rhs_norm[c],
                    final_residual_norm: r_norm,
                    diagnostics: diagnostics[c].take(),
                });
                continue;
            }
            let rz_new = rz[c];
            if !rz_new.is_finite() || rz_new <= 0.0 {
                active[c] = false;
                beta[c] = 0.0;
                done[c] = Some(PcgCoreResult {
                    stop: PcgStop::Breakdown,
                    iterations: iter + 1,
                    rhs_norm: rhs_norm[c],
                    final_residual_norm: r_norm,
                    diagnostics: diagnostics[c].take(),
                });
                continue;
            }
            let b = rz_new / rz_old[c];
            if !b.is_finite() {
                active[c] = false;
                beta[c] = 0.0;
                done[c] = Some(PcgCoreResult {
                    stop: PcgStop::Breakdown,
                    iterations: iter + 1,
                    rhs_norm: rhs_norm[c],
                    final_residual_norm: r_norm,
                    diagnostics: diagnostics[c].take(),
                });
                continue;
            }
            beta[c] = b;
            if let Some(d) = diagnostics[c].as_mut() {
                d.push_iteration(alpha[c], Some(b), r_norm);
            }
        }
        backend.update_p(&beta, &active);
        for c in 0..t {
            if active[c] {
                rz_old[c] = rz[c];
            }
        }
    }

    done.into_iter()
        .enumerate()
        .map(|(c, slot)| {
            slot.unwrap_or_else(|| PcgCoreResult {
                stop: PcgStop::MaxIters,
                iterations: max_iters,
                rhs_norm: rhs_norm[c],
                final_residual_norm: last_r_norm[c],
                diagnostics: diagnostics[c].take(),
            })
        })
        .collect()
}

/// CPU block backend over dense row-major `rows × columns` state, with the
/// operator supplied as a caller closure (`AP ← A·P`). The closure is
/// responsible for honoring the per-column summation-order contract of
/// [`PcgBlockBackend::apply_block`].
///
/// All block traversals parallelize only across DISJOINT outputs (row chunks
/// for the elementwise updates, `BLOCK_DOT_COLUMN_TILE`-column tiles for the
/// inner products), and every per-column fold is strict ascending-row, so the
/// results are independent of thread count — the same bit-reproducibility the
/// Serial reduction gives `pcg_core`.
pub struct CpuPcgBlockBackend<F>
where
    F: Fn(&ndarray::Array2<f64>, &mut ndarray::Array2<f64>) + Sync,
{
    x: ndarray::Array2<f64>,
    r: ndarray::Array2<f64>,
    z: ndarray::Array2<f64>,
    p: ndarray::Array2<f64>,
    ap: ndarray::Array2<f64>,
    coarse_coefficients: ndarray::Array2<f64>,
    preconditioner: SymmetricLowRankPreconditioner,
    rhs_norm_squared: Vec<f64>,
    apply: F,
}

impl<F> CpuPcgBlockBackend<F>
where
    F: Fn(&ndarray::Array2<f64>, &mut ndarray::Array2<f64>) + Sync,
{
    /// Enter the exact PCG initial state from an arbitrary solution seed.
    pub fn new(
        rhs_block: ndarray::Array2<f64>,
        initial_solution: ndarray::Array2<f64>,
        inverse_diagonal: Vec<f64>,
        apply: F,
    ) -> Self {
        Self::new_with_preconditioner(
            rhs_block,
            initial_solution,
            SymmetricLowRankPreconditioner::jacobi(inverse_diagonal),
            apply,
        )
    }

    /// Enter the exact PCG initial state with a validated symmetric
    /// low-rank-corrected preconditioner.
    pub fn new_with_preconditioner(
        rhs_block: ndarray::Array2<f64>,
        initial_solution: ndarray::Array2<f64>,
        preconditioner: SymmetricLowRankPreconditioner,
        apply: F,
    ) -> Self {
        let (m, t) = rhs_block.dim();
        assert_eq!(initial_solution.dim(), (m, t));
        assert_eq!(preconditioner.rows(), m);
        let mut rhs_norm_squared = vec![0.0; t];
        Self::column_dots(&rhs_block, &rhs_block, &mut rhs_norm_squared);
        let mut ap = ndarray::Array2::zeros((m, t));
        if initial_solution.iter().any(|&value| value != 0.0) {
            apply(&initial_solution, &mut ap);
        }
        let mut r = rhs_block;
        r.as_slice_mut()
            .expect("block backend state is standard layout")
            .par_chunks_mut(t)
            .zip(
                ap.as_slice()
                    .expect("block backend state is standard layout")
                    .par_chunks(t),
            )
            .for_each(|(rrow, aprow)| {
                for c in 0..t {
                    rrow[c] += -aprow[c];
                }
            });
        let z = ndarray::Array2::zeros((m, t));
        let p = z.clone();
        let coarse_coefficients = ndarray::Array2::zeros((preconditioner.rank(), t));
        let mut backend = Self {
            x: initial_solution,
            r,
            z,
            p,
            ap,
            coarse_coefficients,
            preconditioner,
            rhs_norm_squared,
            apply,
        };
        backend.apply_preconditioner();
        backend.p.assign(&backend.z);
        backend
    }

    /// The solution block `X` (`rows × columns`); column `c` is the final
    /// iterate reported by the matching [`PcgCoreResult`].
    pub fn solution(&self) -> &ndarray::Array2<f64> {
        &self.x
    }

    /// Consume the backend and take the solution block without copying.
    pub fn into_solution(self) -> ndarray::Array2<f64> {
        self.x
    }

    /// `Z = H R` for the symmetric low-rank preconditioner. Every coarse
    /// coefficient is a strict ascending-row fold and every expansion is an
    /// ascending-rank fold. The CUDA backend mirrors these two orders exactly.
    fn apply_preconditioner(&mut self) {
        let (m, t) = self.r.dim();
        let rank = self.preconditioner.rank();
        let rs = self
            .r
            .as_slice()
            .expect("block backend state is standard layout");
        let projection = self
            .preconditioner
            .projection()
            .as_slice()
            .expect("PCG projection is standard layout");
        let coefficients = self
            .coarse_coefficients
            .as_slice_mut()
            .expect("PCG coarse coefficients are standard layout");
        coefficients
            .par_chunks_mut(t)
            .enumerate()
            .for_each(|(q, coefficient_row)| {
                for c in 0..t {
                    let mut acc = 0.0f64;
                    for i in 0..m {
                        acc += projection[i * rank + q] * rs[i * t + c];
                    }
                    coefficient_row[c] = acc;
                }
            });

        let inverse_diagonal = self.preconditioner.inverse_diagonal();
        let correction = self
            .preconditioner
            .correction()
            .as_slice()
            .expect("PCG correction is standard layout");
        let coefficients = self
            .coarse_coefficients
            .as_slice()
            .expect("PCG coarse coefficients are standard layout");
        self.z
            .as_slice_mut()
            .expect("block backend state is standard layout")
            .par_chunks_mut(t)
            .enumerate()
            .for_each(|(i, zrow)| {
                let rrow = &rs[i * t..(i + 1) * t];
                for c in 0..t {
                    let mut acc = inverse_diagonal[i] * rrow[c];
                    for q in 0..rank {
                        acc += correction[i * rank + q] * coefficients[q * t + c];
                    }
                    zrow[c] = acc;
                }
            });
    }

    fn column_dots(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>, out: &mut [f64]) {
        let (m, t) = a.dim();
        let av = a
            .as_slice()
            .expect("block backend state is standard layout");
        let bv = b
            .as_slice()
            .expect("block backend state is standard layout");
        out.par_chunks_mut(BLOCK_DOT_COLUMN_TILE)
            .enumerate()
            .for_each(|(tile, chunk)| {
                let c0 = tile * BLOCK_DOT_COLUMN_TILE;
                let w = chunk.len();
                let mut acc = [0.0f64; BLOCK_DOT_COLUMN_TILE];
                for i in 0..m {
                    let base = i * t + c0;
                    for (l, slot) in acc.iter_mut().enumerate().take(w) {
                        *slot += av[base + l] * bv[base + l];
                    }
                }
                chunk.copy_from_slice(&acc[..w]);
            });
    }
}

impl<F> PcgBlockBackend for CpuPcgBlockBackend<F>
where
    F: Fn(&ndarray::Array2<f64>, &mut ndarray::Array2<f64>) + Sync,
{
    fn rows(&self) -> usize {
        self.x.nrows()
    }

    fn columns(&self) -> usize {
        self.x.ncols()
    }

    fn rhs_norm_squared(&mut self, out: &mut [f64]) {
        out.copy_from_slice(&self.rhs_norm_squared);
    }

    fn apply_block(&mut self) {
        (self.apply)(&self.p, &mut self.ap);
    }

    fn dot_p_ap(&mut self, out: &mut [f64]) {
        Self::column_dots(&self.p, &self.ap, out);
    }

    fn dot_r_r(&mut self, out: &mut [f64]) {
        Self::column_dots(&self.r, &self.r, out);
    }

    fn dot_r_z(&mut self, out: &mut [f64]) {
        Self::column_dots(&self.r, &self.z, out);
    }

    fn update_x_r(&mut self, alpha: &[f64], active: &[bool]) {
        let t = self.x.ncols();
        let xs = self
            .x
            .as_slice_mut()
            .expect("block backend state is standard layout");
        let rs = self
            .r
            .as_slice_mut()
            .expect("block backend state is standard layout");
        let ps = self
            .p
            .as_slice()
            .expect("block backend state is standard layout");
        let aps = self
            .ap
            .as_slice()
            .expect("block backend state is standard layout");
        xs.par_chunks_mut(t)
            .zip(rs.par_chunks_mut(t))
            .zip(ps.par_chunks(t).zip(aps.par_chunks(t)))
            .for_each(|((xrow, rrow), (prow, aprow))| {
                for c in 0..t {
                    if active[c] {
                        xrow[c] += alpha[c] * prow[c];
                        rrow[c] += -alpha[c] * aprow[c];
                    }
                }
            });
    }

    fn refresh_preconditioned_residual(&mut self) {
        self.apply_preconditioner();
    }

    fn update_p(&mut self, beta: &[f64], active: &[bool]) {
        let t = self.p.ncols();
        let ps = self
            .p
            .as_slice_mut()
            .expect("block backend state is standard layout");
        let zs = self
            .z
            .as_slice()
            .expect("block backend state is standard layout");
        ps.par_chunks_mut(t)
            .zip(zs.par_chunks(t))
            .for_each(|(prow, zrow)| {
                for c in 0..t {
                    if active[c] {
                        prow[c] = zrow[c] + beta[c] * prow[c];
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The `Serial` reduction is byte-for-byte the historical `serial_dot`
    /// (a strict left-to-right fold). This pins the main-solve contract: the
    /// dispatch must not perturb a single bit on the serial path.
    #[test]
    fn dot_serial_is_bit_identical_to_plain_left_fold() {
        // Use values whose exact fold order is observable: a Kahan-sensitive mix
        // of a large term and many small ones.
        let mut av = vec![1e16, 1.0];
        let mut bv = vec![1.0, 1.0];
        for k in 0..4096 {
            av.push(1.0);
            bv.push(((k as f64).sin()).abs() + 1e-3);
        }
        let a = Array1::from(av);
        let b = Array1::from(bv);
        // Independent strict left-to-right reference.
        let mut reference = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            reference += x * y;
        }
        let got = dot(&a.view(), &b.view(), DotReduction::Serial);
        assert_eq!(
            got.to_bits(),
            reference.to_bits(),
            "Serial reduction must be bit-identical to the plain left fold"
        );
    }

    /// The `Reordered` reduction agrees with the serial fold to a relative
    /// tolerance far tighter than the trace estimator's ~1% sampling SE (and
    /// tighter than the 1e-6 per-probe CG tolerance), so the associativity
    /// reorder is dominated by Monte-Carlo noise on the only caller that uses
    /// it. It is deliberately NOT bit-identical.
    #[test]
    fn dot_reordered_matches_serial_to_loose_tol() {
        for &n in &[7usize, 8, 9, 16, 100, 513, 1024, 4096] {
            let a: Array1<f64> = Array1::from_shape_fn(n, |i| ((i * 7 + 1) as f64).sin() * 3.0);
            let b: Array1<f64> = Array1::from_shape_fn(n, |i| ((i * 13 + 3) as f64).cos() * 2.0);
            let s = dot(&a.view(), &b.view(), DotReduction::Serial);
            let r = dot(&a.view(), &b.view(), DotReduction::Reordered);
            let rel = (s - r).abs() / s.abs().max(1e-300);
            assert!(
                rel < 1e-12,
                "n={n}: reordered rel diff {rel:.3e} should be far below trace SE"
            );
        }
    }

    /// The reordered dot must handle non-multiple-of-8 lengths (the tail loop)
    /// and produce the same value the serial path would for small inputs where
    /// the lane count exceeds the length.
    #[test]
    fn dot_reordered_handles_tail_and_short_lengths() {
        for &n in &[0usize, 1, 3, 5, 7] {
            let a: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) + 0.25);
            let b: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) * 0.5 + 1.0);
            let s = dot(&a.view(), &b.view(), DotReduction::Serial);
            let r = dot(&a.view(), &b.view(), DotReduction::Reordered);
            // Below LANES the reordered tail is itself a left fold over the same
            // order, so for these short lengths it is bit-identical.
            assert_eq!(s.to_bits(), r.to_bits(), "n={n}");
        }
    }

    #[test]
    fn pcg_core_matches_known_spd_solve() {
        // A x = b with SPD A; compare against the closed-form solution.
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        // Exact: x = A^{-1} b = (1/11)[1, 7] = [0.0909..., 0.6363...].
        let precond = array![4.0, 3.0];
        let mut x = Array1::<f64>::zeros(2);
        let result = pcg_core(
            |v: &Array1<f64>, out: &mut Array1<f64>| {
                let prod = a.dot(v);
                out.assign(&prod);
            },
            &b.view(),
            &precond.view(),
            1e-12,
            20,
            32,
            true,
            DotReduction::Serial,
            &mut x.view_mut(),
        );
        assert_eq!(result.stop, PcgStop::Converged);
        assert!((x[0] - 0.0909090909).abs() < 1e-9, "x0={}", x[0]);
        assert!((x[1] - 0.6363636363).abs() < 1e-9, "x1={}", x[1]);
        let d = result.diagnostics.expect("diagnostics recorded");
        assert!(!d.alpha.is_empty());
    }

    #[test]
    fn pcg_core_unpreconditioned_diagonal_one_iteration() {
        // Unpreconditioned (precond=1) on diagonal A converges in one step,
        // exactly as the GPU serial cg_solve did.
        let p = 8;
        let diag: Vec<f64> = (0..p).map(|i| 1.0 + i as f64).collect();
        let b: Vec<f64> = (0..p).map(|i| (i as f64) + 0.5).collect();
        let b = Array1::from_vec(b);
        let ones = Array1::<f64>::ones(p);
        let diag_clone = diag.clone();
        let mut w = Array1::<f64>::zeros(p);
        let result = pcg_core(
            |v: &Array1<f64>, out: &mut Array1<f64>| {
                for i in 0..p {
                    out[i] = diag_clone[i] * v[i];
                }
            },
            &b.view(),
            &ones.view(),
            1e-12,
            p,
            0,
            false,
            DotReduction::Serial,
            &mut w.view_mut(),
        );
        assert_eq!(result.stop, PcgStop::Converged);
        assert!(result.diagnostics.is_none());
        for i in 0..p {
            let expected = b[i] / diag[i];
            assert!((w[i] - expected).abs() < 1e-10, "w[{i}]={}", w[i]);
        }
    }

    #[test]
    fn pcg_core_rejects_bad_preconditioner() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let precond = array![-4.0, 3.0];
        let mut x = Array1::<f64>::zeros(2);
        let result = pcg_core(
            |v: &Array1<f64>, out: &mut Array1<f64>| {
                out.assign(&a.dot(v));
            },
            &b.view(),
            &precond.view(),
            1e-12,
            20,
            32,
            false,
            DotReduction::Serial,
            &mut x.view_mut(),
        );
        assert_eq!(result.stop, PcgStop::BadPreconditioner);
        assert_eq!(x, Array1::<f64>::zeros(2));
    }

    #[test]
    fn pcg_core_rejects_zero_preconditioner_entry() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let precond = array![4.0, 0.0];
        let mut x = Array1::<f64>::zeros(2);
        let result = pcg_core(
            |v: &Array1<f64>, out: &mut Array1<f64>| {
                out.assign(&a.dot(v));
            },
            &b.view(),
            &precond.view(),
            1e-12,
            20,
            32,
            false,
            DotReduction::Serial,
            &mut x.view_mut(),
        );
        assert_eq!(result.stop, PcgStop::BadPreconditioner);
        assert_eq!(x, Array1::<f64>::zeros(2));
    }

    use ndarray::Array2;

    /// Deterministic sparse SPD test operator: diagonally dominant with a few
    /// off-diagonal couplings, applied per column in a FIXED summation order
    /// (diagonal first, then ascending neighbor index) so the single-RHS and
    /// block applications are bit-identical by construction.
    struct SparseSpd {
        n: usize,
        diag: Vec<f64>,
        neigh: Vec<Vec<(usize, f64)>>,
    }

    impl SparseSpd {
        fn seeded(n: usize, seed: u64) -> Self {
            let mut state = seed.max(1);
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f64 / u64::MAX as f64) - 0.5
            };
            let mut neigh: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
            for i in 0..n {
                for j in (i + 1)..n {
                    if (i * 31 + j * 17 + (seed as usize)) % 7 == 0 {
                        let v = next();
                        neigh[i].push((j, v));
                        neigh[j].push((i, v));
                    }
                }
            }
            let diag: Vec<f64> = (0..n)
                .map(|i| {
                    let row_abs: f64 = neigh[i].iter().map(|&(_, v)| v.abs()).sum();
                    row_abs + 0.5 + next().abs()
                })
                .collect();
            Self { n, diag, neigh }
        }

        fn apply_column(&self, x: &[f64], out: &mut [f64]) {
            for i in 0..self.n {
                let mut acc = self.diag[i] * x[i];
                for &(j, v) in &self.neigh[i] {
                    acc += v * x[j];
                }
                out[i] = acc;
            }
        }

        fn apply_block(&self, p: &Array2<f64>, ap: &mut Array2<f64>) {
            let t = p.ncols();
            for i in 0..self.n {
                for c in 0..t {
                    let mut acc = self.diag[i] * p[[i, c]];
                    for &(j, v) in &self.neigh[i] {
                        acc += v * p[[j, c]];
                    }
                    ap[[i, c]] = acc;
                }
            }
        }
    }

    /// Run `pcg_core` per column (ones preconditioner, refresh 0, Serial) and
    /// `pcg_multi_core` on the same block; every column must agree BIT-FOR-BIT
    /// in stop reason, iteration count, norms, iterate, and diagnostics trace.
    fn assert_multi_matches_core(
        op: &SparseSpd,
        rhs: &Array2<f64>,
        rel_tol: f64,
        max_iters: usize,
    ) {
        let (m, t) = rhs.dim();
        let mut backend = CpuPcgBlockBackend::new(
            rhs.clone(),
            Array2::zeros((m, t)),
            vec![1.0; m],
            |p: &Array2<f64>, ap: &mut Array2<f64>| op.apply_block(p, ap),
        );
        let multi = pcg_multi_core(&mut backend, rel_tol, max_iters, true);
        assert_eq!(multi.len(), t);

        for c in 0..t {
            let b: Array1<f64> = rhs.column(c).to_owned();
            let ones = Array1::<f64>::ones(m);
            let mut x = Array1::<f64>::zeros(m);
            let single = pcg_core(
                |v: &Array1<f64>, out: &mut Array1<f64>| {
                    let mut buf = vec![0.0f64; m];
                    op.apply_column(v.as_slice().expect("contiguous"), &mut buf);
                    out.assign(&Array1::from_vec(buf));
                },
                &b.view(),
                &ones.view(),
                rel_tol,
                max_iters,
                0,
                true,
                DotReduction::Serial,
                &mut x.view_mut(),
            );
            let blocked = &multi[c];
            assert_eq!(single.stop, blocked.stop, "column {c} stop");
            assert_eq!(
                single.iterations, blocked.iterations,
                "column {c} iterations"
            );
            assert_eq!(
                single.rhs_norm.to_bits(),
                blocked.rhs_norm.to_bits(),
                "column {c} rhs_norm"
            );
            assert_eq!(
                single.final_residual_norm.to_bits(),
                blocked.final_residual_norm.to_bits(),
                "column {c} final residual"
            );
            for i in 0..m {
                assert_eq!(
                    x[i].to_bits(),
                    backend.solution()[[i, c]].to_bits(),
                    "column {c} solution row {i}"
                );
            }
            let ds = single.diagnostics.expect("single diagnostics");
            let dm = blocked.diagnostics.as_ref().expect("multi diagnostics");
            assert_eq!(ds.alpha.len(), dm.alpha.len(), "column {c} alpha trace");
            assert_eq!(ds.beta.len(), dm.beta.len(), "column {c} beta trace");
            for (k, (a, b)) in ds.alpha.iter().zip(dm.alpha.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "column {c} alpha[{k}]");
            }
            for (k, (a, b)) in ds.beta.iter().zip(dm.beta.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "column {c} beta[{k}]");
            }
            for (k, (a, b)) in ds.residuals.iter().zip(dm.residuals.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "column {c} residual[{k}]");
            }
        }
    }

    /// Heterogeneous block — well-conditioned columns, a zero column, and
    /// wildly scaled columns — must reproduce `pcg_core` per column exactly,
    /// including different per-column iteration counts (the freeze path).
    #[test]
    fn pcg_multi_matches_pcg_core_bitwise_per_column() {
        let op = SparseSpd::seeded(41, 0x1017);
        let m = op.n;
        let t = 7;
        let mut rhs = Array2::<f64>::zeros((m, t));
        for c in 0..t {
            if c == 3 {
                continue; // zero right-hand side column
            }
            let scale = 10f64.powi(c as i32 - 2);
            for i in 0..m {
                rhs[[i, c]] = scale * (((i * 13 + c * 7 + 5) as f64).sin());
            }
        }
        assert_multi_matches_core(&op, &rhs, 1e-12, 400);
    }

    /// The iteration-cap path (`MaxIters`) must freeze per column exactly as
    /// `pcg_core` reports it.
    #[test]
    fn pcg_multi_matches_pcg_core_at_iteration_cap() {
        let op = SparseSpd::seeded(29, 0xBEEF);
        let m = op.n;
        let t = 4;
        let mut rhs = Array2::<f64>::zeros((m, t));
        for c in 0..t {
            for i in 0..m {
                rhs[[i, c]] = ((i * 5 + c * 3 + 1) as f64).cos();
            }
        }
        assert_multi_matches_core(&op, &rhs, 1e-14, 3);
    }

    /// Breakdown parity: an indefinite operator must produce the same
    /// per-column `Breakdown` stop, iteration count, and last-valid iterate.
    #[test]
    fn pcg_multi_matches_pcg_core_on_breakdown() {
        let mut op = SparseSpd::seeded(17, 0xD00D);
        // Flip one diagonal negative: pᵀAp goes non-positive along the way.
        op.diag[5] = -3.0;
        let m = op.n;
        let t = 3;
        let mut rhs = Array2::<f64>::zeros((m, t));
        for c in 0..t {
            for i in 0..m {
                rhs[[i, c]] = ((i * 7 + c * 11 + 2) as f64).sin();
            }
        }
        assert_multi_matches_core(&op, &rhs, 1e-12, 200);
    }

    #[test]
    fn pcg_multi_accepts_an_exact_nonzero_seed_without_an_iteration() {
        let op = SparseSpd::seeded(37, 0x2441);
        let (m, t) = (op.n, 5);
        let mut exact = Array2::<f64>::zeros((m, t));
        for i in 0..m {
            for c in 0..t {
                exact[[i, c]] = ((i * 17 + c * 23 + 1) as f64).sin();
            }
        }
        let mut rhs = Array2::<f64>::zeros((m, t));
        op.apply_block(&exact, &mut rhs);
        let inverse_diagonal = op.diag.iter().map(|d| d.recip()).collect();
        let mut backend = CpuPcgBlockBackend::new(
            rhs,
            exact.clone(),
            inverse_diagonal,
            |p: &Array2<f64>, ap: &mut Array2<f64>| op.apply_block(p, ap),
        );

        let results = pcg_multi_core(&mut backend, 1.0e-12, m, true);

        assert!(
            results
                .iter()
                .all(|result| result.stop == PcgStop::Converged && result.iterations == 0),
            "an exact cached solution must satisfy the unchanged residual certificate before CG"
        );
        for i in 0..m {
            for c in 0..t {
                assert_eq!(
                    backend.solution()[[i, c]].to_bits(),
                    exact[[i, c]].to_bits(),
                    "a zero-iteration solve must retain the seed [{i},{c}]"
                );
            }
        }
    }

    #[test]
    fn jacobi_block_pcg_removes_diagonal_condition_growth() {
        let m = 32;
        let diagonal: Vec<f64> = (0..m)
            .map(|i| {
                let scale = (i + 1) as f64;
                scale * scale
            })
            .collect();
        let mut rhs = Array2::<f64>::zeros((m, 1));
        for i in 0..m {
            rhs[[i, 0]] = diagonal[i] * ((i * 11 + 3) as f64).sin();
        }
        let apply = |p: &Array2<f64>, ap: &mut Array2<f64>| {
            for i in 0..m {
                ap[[i, 0]] = diagonal[i] * p[[i, 0]];
            }
        };
        let mut identity =
            CpuPcgBlockBackend::new(rhs.clone(), Array2::zeros((m, 1)), vec![1.0; m], apply);
        let identity_result = pcg_multi_core(&mut identity, 1.0e-10, m, false);

        let inverse_diagonal = diagonal.iter().map(|d| d.recip()).collect();
        let mut jacobi =
            CpuPcgBlockBackend::new(rhs, Array2::zeros((m, 1)), inverse_diagonal, apply);
        let jacobi_result = pcg_multi_core(&mut jacobi, 1.0e-10, m, false);

        assert_eq!(jacobi_result[0].stop, PcgStop::Converged);
        assert_eq!(
            jacobi_result[0].iterations, 1,
            "the exact diagonal preconditioner must collapse a diagonal system to one step"
        );
        assert!(
            identity_result[0].iterations > jacobi_result[0].iterations,
            "the cold identity recurrence must expose the diagonal condition spread"
        );
    }

    #[test]
    fn symmetric_low_rank_block_preconditioner_inverts_a_non_diagonal_mode() {
        let (m, t) = (48usize, 5usize);
        let eigenvalue = 1.0e-4f64;
        let mut q = Array1::<f64>::zeros(m);
        for i in 0..m {
            q[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        q /= q.dot(&q).sqrt();

        // A = I + (λ-1)qqᵀ: diagonal scaling cannot remove this dense
        // non-diagonal outlier. The validating factory derives
        // H = I + (λ⁻¹-1)qqᵀ, its exact inverse, from A's action on q.
        let mut candidate = Array2::<f64>::zeros((m, 1));
        for i in 0..m {
            candidate[[i, 0]] = q[i];
        }
        let preconditioner = SymmetricLowRankPreconditioner::from_scaled_subspace(
            vec![1.0; m],
            candidate,
            |basis, image| {
                for c in 0..basis.ncols() {
                    let mut qt_x = 0.0f64;
                    for i in 0..m {
                        qt_x += q[i] * basis[[i, c]];
                    }
                    for i in 0..m {
                        image[[i, c]] = basis[[i, c]] + (eigenvalue - 1.0) * q[i] * qt_x;
                    }
                }
            },
        )
        .expect("the exact scaled coarse operator is SPD");
        let mut rhs = Array2::<f64>::zeros((m, t));
        for i in 0..m {
            for c in 0..t {
                rhs[[i, c]] = ((i * 17 + c * 29 + 3) as f64).sin() + (c + 1) as f64 * q[i];
            }
        }
        let apply = |x: &Array2<f64>, out: &mut Array2<f64>| {
            for c in 0..t {
                let mut qt_x = 0.0f64;
                for i in 0..m {
                    qt_x += q[i] * x[[i, c]];
                }
                for i in 0..m {
                    out[[i, c]] = x[[i, c]] + (eigenvalue - 1.0) * q[i] * qt_x;
                }
            }
        };
        let mut backend = CpuPcgBlockBackend::new_with_preconditioner(
            rhs,
            Array2::zeros((m, t)),
            preconditioner,
            apply,
        );
        let results = pcg_multi_core(&mut backend, 1.0e-10, m, false);
        assert!(
            results
                .iter()
                .all(|result| result.stop == PcgStop::Converged && result.iterations <= 2),
            "an exact non-diagonal spectral inverse must collapse the solve to rounding-scale work: \
             {:?}",
            results
                .iter()
                .map(|result| (result.stop, result.iterations))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn symmetric_low_rank_preconditioner_rejects_an_indefinite_coarse_operator() {
        let mut candidate = Array2::<f64>::zeros((8, 1));
        candidate[[3, 0]] = 1.0;
        let err = SymmetricLowRankPreconditioner::from_scaled_subspace(
            vec![1.0; 8],
            candidate,
            |basis, image| {
                image.assign(basis);
                image.mapv_inplace(|value| -value);
            },
        )
        .err()
        .expect("a negative Galerkin eigenvalue must not mint a preconditioner");
        assert!(
            err.contains("not SPD"),
            "the rejected invariant should name the failed SPD proof: {err}"
        );
    }

    /// Convergence target is RELATIVE for sub-unit rhs.
    ///
    /// With the old `tol = rel_tol · max(‖rhs‖, 1)` rule, asking for
    /// `rel_tol = 0.1` on a rhs with `‖rhs‖ ≈ 0.06` accepted a relative
    /// residual of ~0.86 (one PCG iteration), which is the looseness that
    /// can trap an inexact-Newton outer loop in a fixed-point oscillation.
    /// The textbook criterion `‖r‖ ≤ rel_tol · ‖rhs‖` must hold instead.
    #[test]
    fn pcg_core_relative_residual_holds_for_sub_unit_rhs() {
        // Mildly anisotropic SPD operator, sub-unit rhs (‖b‖ ≈ 0.062).
        let a = array![
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.25, 0.0],
            [0.0, 0.25, 6.0, 0.5],
            [0.0, 0.0, 0.5, 5.0]
        ];
        let b = array![0.03, -0.02, 0.04, 0.02];
        let precond = array![4.0, 3.0, 6.0, 5.0];
        let rel_tol = 0.1_f64;
        let rhs_norm = (b.iter().map(|x| x * x).sum::<f64>()).sqrt();
        assert!(
            rhs_norm < 1.0,
            "test premise: rhs must be sub-unit; got {rhs_norm}"
        );

        let mut x = Array1::<f64>::zeros(4);
        let result = pcg_core(
            |v: &Array1<f64>, out: &mut Array1<f64>| {
                out.assign(&a.dot(v));
            },
            &b.view(),
            &precond.view(),
            rel_tol,
            64,
            32,
            false,
            DotReduction::Serial,
            &mut x.view_mut(),
        );
        assert_eq!(result.stop, PcgStop::Converged);

        // Independently recompute ‖r‖ = ‖b − A x‖ on the returned iterate;
        // the relative criterion must hold against ‖rhs‖, NOT against
        // `max(‖rhs‖, 1)`.
        let r: Array1<f64> = &b - &a.dot(&x);
        let r_norm = (r.iter().map(|v| v * v).sum::<f64>()).sqrt();
        assert!(
            r_norm <= rel_tol * rhs_norm + 1e-12,
            "expected ‖r‖={r_norm:.3e} ≤ rel_tol·‖rhs‖={:.3e}",
            rel_tol * rhs_norm
        );
    }
}
