//! Single preconditioned conjugate-gradient (PCG) core.
//!
//! Both the CPU SPD solver (`linalg::utils::solve_spd_pcg_with_info`, parallel,
//! residual-refresh, diagnostics) and the GPU REML trace solver
//! (`gpu::kernels::reml_trace::cg_solve`, serial, no refresh, no diagnostics)
//! historically carried their own hand-rolled CG loop. They drifted: the GPU
//! copy silently floored the (absent) preconditioner and accepted a partial
//! solution on lost SPD, while the CPU copy rejected non-positive preconditioner
//! diagonals and refreshed the residual every 32 iterations. The shared inner
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

/// Floor on the requested PCG relative tolerance. Asking for convergence tighter
/// than this is below the achievable accuracy of the SPD energy minimization in
/// `f64`, so we clamp the target to avoid iterating on numerical noise.
pub(crate) const PCG_REL_TOL_FLOOR: f64 = 1e-12;

/// Floor applied to each (already non-negative) preconditioner diagonal entry
/// before reciprocation. Exactly-zero entries are treated as numerical noise and
/// floored to this value rather than producing an infinite `1/m`.
pub(crate) const PCG_PRECONDITIONER_FLOOR: f64 = 1e-12;

/// Per-iteration trace of the PCG recurrence, sufficient to reconstruct the
/// Lanczos tridiagonal and hence Ritz-based condition estimates. Populated only
/// when the caller requests diagnostics.
#[derive(Debug, Clone)]
pub(crate) struct PcgDiagnostics {
    pub(crate) residuals: Vec<f64>,
    pub(crate) alpha: Vec<f64>,
    pub(crate) beta: Vec<f64>,
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
pub(crate) enum PcgStop {
    /// `‖r‖ ≤ tol`; the recorded iterate is the converged solution.
    Converged,
    /// Hit `max_iters` without reaching tolerance.
    MaxIters,
    /// Lost SPD or hit a non-finite scalar (e.g. `pᵀAp ≤ 0`, non-finite
    /// `alpha`/`beta`, a non-positive `rᵀz`, or a mismatched matvec length).
    /// The iterate written so far is the last numerically valid one; callers
    /// decide whether to keep it (GPU) or reject the whole solve (CPU).
    Breakdown,
    /// The preconditioner diagonal contained a negative or non-finite entry,
    /// violating the SPD-PCG contract (`M ≻ 0`). Detected before any iteration;
    /// the solution buffer is untouched (left at the zero initial guess).
    BadPreconditioner,
}

/// Result of a [`pcg_core`] run. The solution is written into the caller's
/// buffer; this carries the metadata about how the run terminated.
#[derive(Debug, Clone)]
pub(crate) struct PcgCoreResult {
    pub(crate) stop: PcgStop,
    pub(crate) iterations: usize,
    pub(crate) rhs_norm: f64,
    pub(crate) final_residual_norm: f64,
    pub(crate) diagnostics: Option<PcgDiagnostics>,
}

/// Serial inner product. Sequential fold so the result is independent of thread
/// count — see the module-level numerics note.
#[inline]
fn serial_dot(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let mut acc = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// The shared preconditioned conjugate-gradient recurrence.
///
/// Solves `A x = rhs` for SPD `A`, accessed only through `apply(v, out)` which
/// must set `out <- A v`. The initial guess is `x = 0`. Convergence target is
/// `‖r‖ ≤ max(rel_tol, FLOOR) · max(‖rhs‖, 1)`.
///
/// * `precond_diag` — diagonal Jacobi preconditioner `M`; pass all-ones for an
///   unpreconditioned solve. Entries are floored to
///   [`PCG_PRECONDITIONER_FLOOR`] before reciprocation; a negative or
///   non-finite entry is a contract violation reported as
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
pub(crate) fn pcg_core<F>(
    mut apply: F,
    rhs: &ArrayView1<f64>,
    precond_diag: &ArrayView1<f64>,
    rel_tol: f64,
    max_iters: usize,
    refresh_period: usize,
    record_diagnostics: bool,
    solution: &mut ArrayViewMut1<f64>,
) -> PcgCoreResult
where
    F: FnMut(&Array1<f64>, &mut Array1<f64>),
{
    let p = rhs.len();
    let rhs_norm = serial_dot(rhs, rhs).sqrt();

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

    let tol = rel_tol.max(PCG_REL_TOL_FLOOR) * rhs_norm.max(1.0);

    // Precompute reciprocal preconditioner once: z = inv_m * r per iteration.
    // SPD-PCG requires M ≻ 0; a negative/non-finite entry is a contract
    // violation surfaced as BadPreconditioner rather than silently abs()-ed.
    let mut inv_m = Array1::<f64>::zeros(p);
    let mut bad_diag = false;
    for (slot, &m) in inv_m.iter_mut().zip(precond_diag.iter()) {
        if !m.is_finite() || m < 0.0 {
            bad_diag = true;
            break;
        }
        *slot = 1.0 / m.max(PCG_PRECONDITIONER_FLOOR);
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
    let mut rz_old = serial_dot(&r.view(), &z.view());
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
        let denom = serial_dot(&p_dir.view(), &ap.view());
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
        let r_norm = serial_dot(&r.view(), &r.view()).sqrt();
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
        let rz_new = serial_dot(&r.view(), &z.view());
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
            &mut x.view_mut(),
        );
        assert_eq!(result.stop, PcgStop::BadPreconditioner);
        assert_eq!(x, Array1::<f64>::zeros(2));
    }
}
