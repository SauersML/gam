//! Sparse Identification of Nonlinear Dynamics (SINDy) — Sequential Thresholded
//! Least Squares (STLSQ) solver, Brunton, Proctor, Kutz (PNAS 2016).
//!
//! Given a library design matrix `Θ ∈ ℝ^{n × p}` (each column is a candidate
//! nonlinearity evaluated at the `n` trajectory samples) and a target derivative
//! matrix `Ẋ ∈ ℝ^{n × d}`, STLSQ alternates:
//!   1. Solve `Ξ_active ← argmin_Ξ ‖Θ_active · Ξ − Ẋ‖² + λ ‖Ξ‖²` on the current
//!      active support (per output column independently — each column of `Ξ`
//!      is a SINDy coefficient vector for one state-variable derivative).
//!   2. Hard-threshold: drop entries with `|ξ_ij| < tol` from the support.
//! Repeat until the support stops changing or `max_rounds` is reached.
//!
//! The SCAD / MCP path (Fan-Li 2001 / Zhang 2010) uses the local quadratic
//! approximation (LQA): each round re-weights the ridge regularizer
//! coordinate-wise by `p'(|ξ_ij|) / max(|ξ_ij|, ε)`, where `p'` is the SCAD or
//! MCP derivative. This is the standard iterative re-weighted ridge surrogate
//! and reduces to plain ridge when `lam == 0`. The result remains compatible
//! with the hard-threshold step.
//!
//! The solver is fully self-contained: no GAM design / penalty / REML
//! infrastructure is involved. This file is the canonical implementation; the
//! Python `gamfit.SINDyAtoms` class wraps it via `gam-pyffi`.

use crate::faer_ndarray::FaerCholesky;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Concave-penalty family for the per-iteration re-weighting step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SindyPenaltyKind {
    /// Plain ridge (Tikhonov) on the active set.
    Ridge,
    /// SCAD (Fan & Li 2001). `a` defaults to 3.7.
    Scad,
    /// MCP (Zhang 2010). `a` (gamma) defaults to 3.0.
    Mcp,
}

/// SCAD derivative `p'_{λ,a}(|ξ|)`.
#[inline]
fn scad_grad(abs_xi: f64, lam: f64, a: f64) -> f64 {
    if abs_xi <= lam {
        lam
    } else if abs_xi <= a * lam {
        ((a * lam - abs_xi) / (a - 1.0)).max(0.0)
    } else {
        0.0
    }
}

/// MCP derivative `p'_{λ,γ}(|ξ|)`.
#[inline]
fn mcp_grad(abs_xi: f64, lam: f64, gamma: f64) -> f64 {
    if abs_xi <= gamma * lam {
        (lam - abs_xi / gamma).max(0.0)
    } else {
        0.0
    }
}

/// Per-coordinate ridge weight produced by the LQA surrogate for the given
/// concave penalty family at the current coefficient magnitude `abs_xi`.
#[inline]
fn lqa_weight(kind: SindyPenaltyKind, abs_xi: f64, lam: f64, a: f64, eps: f64) -> f64 {
    match kind {
        SindyPenaltyKind::Ridge => lam,
        SindyPenaltyKind::Scad => scad_grad(abs_xi, lam, a) / abs_xi.max(eps),
        SindyPenaltyKind::Mcp => mcp_grad(abs_xi, lam, a) / abs_xi.max(eps),
    }
}

#[derive(Debug, Clone)]
pub struct SindyStlsqResult {
    /// Coefficient matrix `Ξ ∈ ℝ^{p × d}`. Each column is the SINDy vector for
    /// one state-variable derivative; entries below `tol` are exactly zero.
    pub coefficients: Array2<f64>,
    /// Number of STLSQ rounds actually executed (≥ 1, ≤ `max_rounds`).
    pub rounds_used: usize,
    /// `true` iff the active support stabilised before `max_rounds`.
    pub converged: bool,
}

/// Run STLSQ on `(theta, dz_dt)`.
///
/// * `theta`: `(n, p)` library design, each column a library term evaluated at
///   the `n` trajectory rows.
/// * `dz_dt`: `(n, d)` target derivatives, one column per state dimension.
/// * `tol`: hard threshold; entries with `|ξ| < tol` are dropped each round.
/// * `max_rounds`: STLSQ iteration cap (must be ≥ 1).
/// * `lam`: regularization strength. For ridge, this is the L2 weight; for
///   SCAD / MCP it is the concave-penalty parameter.
/// * `kind`: penalty family.
/// * `concave_a`: SCAD's `a` (default 3.7) or MCP's `γ` (default 3.0). Ignored
///   for `Ridge`.
pub fn sindy_stlsq_solve(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    tol: f64,
    max_rounds: usize,
    lam: f64,
    kind: SindyPenaltyKind,
    concave_a: f64,
) -> Result<SindyStlsqResult, String> {
    let (n, p) = theta.dim();
    let (n_dz, d) = dz_dt.dim();
    if n == 0 || p == 0 || d == 0 {
        return Err(format!(
            "sindy_stlsq_solve requires non-empty theta and dz_dt; got theta=({n},{p}), dz_dt=({n_dz},{d})"
        ));
    }
    if n_dz != n {
        return Err(format!(
            "sindy_stlsq_solve requires theta.nrows == dz_dt.nrows; got {n} vs {n_dz}"
        ));
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(format!("sindy_stlsq_solve requires finite tol >= 0, got {tol}"));
    }
    if max_rounds == 0 {
        return Err("sindy_stlsq_solve requires max_rounds >= 1".to_string());
    }
    if !(lam.is_finite() && lam >= 0.0) {
        return Err(format!("sindy_stlsq_solve requires finite lam >= 0, got {lam}"));
    }
    if matches!(kind, SindyPenaltyKind::Scad) && !(concave_a.is_finite() && concave_a > 2.0) {
        return Err(format!(
            "sindy_stlsq_solve SCAD requires concave_a > 2, got {concave_a}"
        ));
    }
    if matches!(kind, SindyPenaltyKind::Mcp) && !(concave_a.is_finite() && concave_a > 1.0) {
        return Err(format!(
            "sindy_stlsq_solve MCP requires concave_a > 1, got {concave_a}"
        ));
    }

    // Initial seed: plain ridge solve on the full library.
    let lam_seed = if lam > 0.0 { lam } else { 1.0e-12 };
    let mut xi = ridge_full_solve(theta, dz_dt, lam_seed)?;
    let lqa_eps = (tol * 1.0e-2).max(1.0e-10);

    let mut active = vec![true; p];
    let mut prev_active = vec![false; p];
    let mut rounds_used = 0usize;
    let mut converged = false;

    for round in 0..max_rounds {
        rounds_used = round + 1;
        // Hard-threshold: an entry is kept active for column c iff
        // |xi[j,c]| >= tol AND already in the active set. The active-set per
        // output column can differ; we operate per column.
        // Active union (any state-var needs it) determines theta column slice
        // for sharing the same Gram factorization across outputs.
        for j in 0..p {
            let mut keep = false;
            for c in 0..d {
                if active[j] && xi[(j, c)].abs() >= tol {
                    keep = true;
                } else {
                    xi[(j, c)] = 0.0;
                }
            }
            active[j] = keep;
        }
        if active.iter().all(|x| !*x) {
            // Everything thresholded out: nothing to refit.
            converged = prev_active == active;
            if converged {
                break;
            }
            prev_active.copy_from_slice(&active);
            continue;
        }
        if active == prev_active {
            converged = true;
            break;
        }
        prev_active.copy_from_slice(&active);

        // Refit on the active set with LQA-weighted ridge.
        let active_idx: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(j, &on)| if on { Some(j) } else { None })
            .collect();
        let p_act = active_idx.len();
        let mut theta_act = Array2::<f64>::zeros((n, p_act));
        for (k, &j) in active_idx.iter().enumerate() {
            theta_act.column_mut(k).assign(&theta.column(j));
        }

        // Build per-coordinate ridge diagonal from current |xi| magnitudes
        // (averaged across output columns: SCAD/MCP are scalar penalties on
        // |ξ| — we use the row-max magnitude as the conservative LQA anchor
        // so a coefficient ever "large" in one column relaxes its surrogate
        // shrinkage uniformly, matching the elementwise-row interpretation
        // used in the Brunton extensions).
        let mut diag = Array1::<f64>::zeros(p_act);
        for (k, &j) in active_idx.iter().enumerate() {
            let mut mag = 0.0_f64;
            for c in 0..d {
                let v = xi[(j, c)].abs();
                if v > mag {
                    mag = v;
                }
            }
            diag[k] = lqa_weight(kind, mag, lam, concave_a, lqa_eps);
        }

        let xi_act = ridge_diag_solve(theta_act.view(), dz_dt, diag.view())?;
        // Scatter back.
        xi.fill(0.0);
        for (k, &j) in active_idx.iter().enumerate() {
            for c in 0..d {
                xi[(j, c)] = xi_act[(k, c)];
            }
        }
    }

    // Final hard-threshold pass.
    for j in 0..p {
        for c in 0..d {
            if xi[(j, c)].abs() < tol {
                xi[(j, c)] = 0.0;
            }
        }
    }

    Ok(SindyStlsqResult {
        coefficients: xi,
        rounds_used,
        converged,
    })
}

/// Plain ridge: `Ξ = (ΘᵀΘ + λ I)⁻¹ Θᵀ Ẋ`.
fn ridge_full_solve(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    lam: f64,
) -> Result<Array2<f64>, String> {
    let p = theta.ncols();
    let diag = Array1::<f64>::from_elem(p, lam.max(1.0e-12));
    ridge_diag_solve(theta, dz_dt, diag.view())
}

/// Ridge with per-coordinate diagonal: `Ξ = (ΘᵀΘ + diag(d))⁻¹ Θᵀ Ẋ`.
fn ridge_diag_solve(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    diag: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let p = theta.ncols();
    let mut gram = theta.t().dot(&theta);
    for i in 0..p {
        let d = diag[i].max(1.0e-12);
        gram[(i, i)] += d;
    }
    let rhs = theta.t().dot(&dz_dt);
    let chol = gram.cholesky(Side::Lower).map_err(|err| {
        format!("sindy_stlsq_solve ridge Cholesky failed: {err}")
    })?;
    let mut sol = rhs;
    chol.solve_mat_in_place(&mut sol);
    Ok(sol)
}

/// Automatic `λ` selection over a logarithmic grid using BIC, with STLSQ
/// applied at each grid point. Returns `(best_lam, best_result)`.
///
/// SINDy is not a GAM design block (it produces a single sparse coefficient
/// matrix with no `S_θ`-tier structure), so the REML / LAML `'auto'` machinery
/// that drives e.g. `AdaptiveTopK` does not apply. The principled standalone
/// alternative is information-criterion selection on the residual variance
/// and active-support cardinality. We use BIC (Schwarz 1978) on the per-output
/// residual sums of squares — the SINDy literature's default complexity rule.
pub fn sindy_stlsq_auto_lam(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    tol: f64,
    max_rounds: usize,
    kind: SindyPenaltyKind,
    concave_a: f64,
) -> Result<(f64, SindyStlsqResult), String> {
    let (n, _p) = theta.dim();
    if n == 0 {
        return Err("sindy_stlsq_auto_lam requires n > 0".to_string());
    }
    // Geometric grid spanning four decades around tol; matches the standard
    // SINDy hyperparameter sweep in the Brunton 2016 supplement.
    let grid: Vec<f64> = (0..9)
        .map(|i| tol.max(1.0e-6) * 10f64.powf((i as f64) - 4.0))
        .collect();
    let mut best: Option<(f64, f64, SindyStlsqResult)> = None;
    for &lam in &grid {
        let res = sindy_stlsq_solve(theta, dz_dt, tol, max_rounds, lam, kind, concave_a)?;
        let bic = bic_score(theta, dz_dt, &res.coefficients);
        let pick = match &best {
            None => true,
            Some((_, b, _)) => bic < *b,
        };
        if pick {
            best = Some((lam, bic, res));
        }
    }
    let (lam, _bic, res) = best.ok_or_else(|| {
        "sindy_stlsq_auto_lam: empty grid produced no candidates".to_string()
    })?;
    Ok((lam, res))
}

fn bic_score(theta: ArrayView2<'_, f64>, dz_dt: ArrayView2<'_, f64>, xi: &Array2<f64>) -> f64 {
    let (n, _p) = theta.dim();
    let d = dz_dt.ncols();
    let resid = &theta.dot(xi) - &dz_dt;
    let mut bic = 0.0_f64;
    let n_f = n as f64;
    for c in 0..d {
        let r = resid.index_axis(Axis(1), c);
        let rss = r.iter().map(|&x| x * x).sum::<f64>().max(1.0e-300);
        let k_active = xi.column(c).iter().filter(|&&v| v != 0.0).count() as f64;
        // Gaussian profile-likelihood BIC: n*log(rss/n) + k*log(n).
        bic += n_f * (rss / n_f).ln() + k_active * n_f.ln();
    }
    bic
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn stlsq_recovers_pure_linear_system() {
        // Library: [1, x, y], target: dx/dt = 2x - 3y, dy/dt = -x.
        // Expected Ξ ≈ [[0, 0], [2, -1], [-3, 0]].
        let n = 200;
        let mut rng_state = 0u64;
        let mut rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
        };
        let mut theta = Array2::<f64>::zeros((n, 3));
        let mut dz = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x = rand();
            let y = rand();
            theta[(i, 0)] = 1.0;
            theta[(i, 1)] = x;
            theta[(i, 2)] = y;
            dz[(i, 0)] = 2.0 * x - 3.0 * y;
            dz[(i, 1)] = -x;
        }
        let res = sindy_stlsq_solve(
            theta.view(),
            dz.view(),
            0.05,
            10,
            1.0e-3,
            SindyPenaltyKind::Ridge,
            3.7,
        )
        .expect("stlsq must succeed");
        assert!(res.converged);
        assert!((res.coefficients[(1, 0)] - 2.0).abs() < 0.05);
        assert!((res.coefficients[(2, 0)] + 3.0).abs() < 0.05);
        assert!((res.coefficients[(1, 1)] + 1.0).abs() < 0.05);
        assert!(res.coefficients[(0, 0)].abs() < 1.0e-6);
        assert!(res.coefficients[(2, 1)].abs() < 1.0e-6);
    }
}
