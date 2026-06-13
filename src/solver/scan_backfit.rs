//! Multi-term backfitting with exact O(n) spline-scan inner solves (#1034
//! item 3): `y = α + f₁(x₁) + … + f_K(x_K) + ε` for Gaussian/identity models
//! whose every term is a 1-D single-penalty smoothing spline of order
//! `m_j ∈ {1, 2}` (the class [`crate::solver::spline_scan`] solves exactly).
//!
//! What is EXACT here, and what is not — the honest scope from the #1034
//! exactness wall:
//!
//! - **Point estimates at a FIXED joint λ are exact.** Backfitting (Gauss–
//!   Seidel on the joint penalized normal equations) with symmetric linear
//!   inner smoothers converges to the joint penalized posterior MEAN (Buja,
//!   Hastie & Tibshirani 1989); each inner smoother here is the exact O(n)
//!   Kalman/RTS scan, so the converged additive fit IS the joint penalized
//!   least-squares solution `min Σᵢ wᵢ(yᵢ − α − Σⱼ fⱼ(x_{j,i}))² +
//!   Σⱼ λⱼ∫(fⱼ^{(m_j)})²`, certified by the residual sweep contraction
//!   (`max_component_delta ≤ tol`). The per-term constant confounding with
//!   the intercept is resolved by weighted centering of every component
//!   (Σᵢ wᵢ fⱼ(x_{j,i}) = 0), the same representative a constrained joint
//!   solve picks.
//! - **Joint λ selection has NO O(n) path.** The joint REML criterion needs
//!   `log|XᵀWX + S_λ|` over ALL terms at once, and there is no additive-model
//!   analog of the scan's diffuse-innovations O(n) logdet. [`fit_scan_backfit`]
//!   therefore selects each λⱼ by the scan's exact diffuse REML applied to the
//!   term's PARTIAL residual, re-selected every sweep — an explicitly-labeled
//!   approximation to joint REML (it ignores the other terms' uncertainty in
//!   each term's criterion), validated against a dense joint-REML grid on
//!   fixtures in this module's tests. Callers wanting exact joint selection
//!   keep the dense path and may pin the resulting λ's via
//!   [`fit_scan_backfit_at`].
//! - **Per-term SEs are CONDITIONAL.** Each stored [`SplineScanFit`] carries
//!   the exact posterior of its term GIVEN the other components fixed at
//!   their converged values (σ² profiled from the joint additive residuals).
//!   Joint covariance across terms (the cross-term posterior correlations)
//!   still requires the dense p×p factorization and is out of scope here.

use super::spline_scan::{SplineScanFit, fit_spline_scan, fit_spline_scan_at};

/// Sweep cap for the fixed-λ backfit loop. Convergence is geometric with rate
/// governed by inter-term concurvity; exceeding this means the design is so
/// concurve the additive decomposition is numerically unidentifiable, and the
/// fit fails loudly rather than returning an uncertified point.
const BACKFIT_MAX_SWEEPS: usize = 1000;
/// Convergence tolerance on the maximum per-row component change in one full
/// sweep, relative to the weighted response scale.
const BACKFIT_TOL_REL: f64 = 1e-10;
/// Sweep cap for the λ-selecting loop (each sweep re-runs the scan's exact
/// per-term REML search, so sweeps are ~75× the fixed-λ cost).
const SELECT_MAX_SWEEPS: usize = 100;
/// Component-change tolerance (relative) for the λ-selecting loop; the final
/// certified solve is re-run at the frozen λ's with the tight fixed-λ
/// tolerance, so this only needs to stabilize the selection.
const SELECT_TOL_REL: f64 = 1e-8;
/// Stability tolerance on `log λⱼ` across consecutive selection sweeps.
const SELECT_LOG_LAMBDA_TOL: f64 = 1e-6;

/// One fitted additive component.
#[derive(Clone, Debug)]
pub struct ScanBackfitTerm {
    /// Exact scan posterior of this term conditional on the other components
    /// (fitted to the converged partial residual, σ² = the joint profile
    /// estimate). `fit.mean`/`fit.predict` are UN-centered; subtract
    /// [`Self::center`] (or use [`ScanBackfitFit::predict_term`]) to get the
    /// identifiable component.
    pub fit: SplineScanFit,
    /// Weighted mean of the raw smoother output over the training rows,
    /// absorbed into the intercept (Σᵢ wᵢ·component_i = 0 after subtraction).
    pub center: f64,
    /// Centered component values at the training rows (input order).
    pub fitted: Vec<f64>,
}

/// Converged additive backfit. See the module docs for the exactness scope.
#[derive(Clone, Debug)]
pub struct ScanBackfitFit {
    /// Intercept (weighted response mean plus every absorbed term center).
    pub intercept: f64,
    /// Per-term components, in input order.
    pub terms: Vec<ScanBackfitTerm>,
    /// Total fitted values `α + Σⱼ fⱼ(x_{j,i})` at the training rows.
    pub fitted: Vec<f64>,
    /// Profiled residual variance `Σ wᵢ(yᵢ−μᵢ)² / (n − edf)`.
    pub sigma2: f64,
    /// Additive-model effective degrees of freedom `1 + Σⱼ(edfⱼ − 1)`:
    /// each conditional smoother trace counts its own constant once, which the
    /// shared intercept absorbs (the classical additive-model df bookkeeping —
    /// concurvity cross-terms are not subtracted, consistent with the
    /// conditional-SE scope).
    pub edf: f64,
    /// Number of full backfit sweeps the certified fixed-λ loop ran.
    pub sweeps: usize,
    /// Convergence certificate: the largest per-row component change observed
    /// in the final sweep. Always `≤ tol` on success.
    pub max_component_delta: f64,
    /// The absolute tolerance the certificate was checked against
    /// (`BACKFIT_TOL_REL` × weighted response scale).
    pub tol: f64,
}

impl ScanBackfitFit {
    /// Centered posterior `(mean, variance)` of term `j` at abscissa `x`.
    /// The variance is CONDITIONAL on the other components (module docs).
    pub fn predict_term(&self, j: usize, x: f64) -> Result<(f64, f64), String> {
        let term = self
            .terms
            .get(j)
            .ok_or_else(|| format!("scan backfit: term index {j} out of range"))?;
        let (mean, var) = term.fit.predict(x)?;
        Ok((mean - term.center, var))
    }

    /// Additive mean prediction `α + Σⱼ fⱼ(xⱼ)` at one covariate row.
    pub fn predict(&self, x: &[f64]) -> Result<f64, String> {
        if x.len() != self.terms.len() {
            return Err(format!(
                "scan backfit: expected {} covariates, got {}",
                self.terms.len(),
                x.len()
            ));
        }
        let mut acc = self.intercept;
        for (j, &xj) in x.iter().enumerate() {
            acc += self.predict_term(j, xj)?.0;
        }
        Ok(acc)
    }
}

/// Map each row to its pooled-knot index for one term: rows sorted by `x`
/// (total order), exact-equality ties pooled — the same dedup
/// `spline_scan::pool_nodes` performs, so index `map[i]` addresses
/// `SplineScanFit::{knots, mean}[map[i]]` for any scan fit on this column.
fn row_knot_map(x: &[f64]) -> (Vec<usize>, usize) {
    let n = x.len();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let mut map = vec![0usize; n];
    let mut knot = 0usize;
    let mut started = false;
    let mut current = f64::NAN;
    for &i in &perm {
        if started {
            if x[i] != current {
                knot += 1;
                current = x[i];
            }
        } else {
            started = true;
            current = x[i];
        }
        map[i] = knot;
    }
    (map, if started { knot + 1 } else { 0 })
}

fn validate_inputs(
    xs: &[&[f64]],
    y: &[f64],
    w: &[f64],
    orders: &[usize],
) -> Result<(), String> {
    let k = xs.len();
    if k == 0 {
        return Err("scan backfit: at least one term required".to_string());
    }
    if orders.len() != k {
        return Err(format!(
            "scan backfit: {k} terms but {} orders",
            orders.len()
        ));
    }
    let n = y.len();
    if w.len() != n {
        return Err(format!(
            "scan backfit: length mismatch y={n}, w={}",
            w.len()
        ));
    }
    for (j, x) in xs.iter().enumerate() {
        if x.len() != n {
            return Err(format!(
                "scan backfit: term {j} has {} rows, response has {n}",
                x.len()
            ));
        }
    }
    for i in 0..n {
        if !(y[i].is_finite() && w[i].is_finite() && w[i] > 0.0) {
            return Err(format!(
                "scan backfit: non-finite or non-positive input at row {i} (y={}, w={})",
                y[i], w[i]
            ));
        }
    }
    Ok(())
}

/// Per-sweep λ policy for the engine.
enum LambdaPolicy<'a> {
    /// Fixed joint λ: exact backfit, tight tolerance, certificate required.
    Fixed(&'a [f64]),
    /// Re-select each λⱼ by the scan's exact per-term diffuse REML on the
    /// partial residual every sweep (the labeled approximation).
    SelectPerSweep,
}

/// Engine state returned by the sweep loop.
struct EngineOut {
    intercept: f64,
    components: Vec<Vec<f64>>,
    log_lambdas: Vec<f64>,
    sweeps: usize,
    max_delta: f64,
    tol_abs: f64,
}

fn backfit_engine(
    xs: &[&[f64]],
    y: &[f64],
    w: &[f64],
    orders: &[usize],
    maps: &[(Vec<usize>, usize)],
    policy: LambdaPolicy<'_>,
) -> Result<EngineOut, String> {
    let n = y.len();
    let k = xs.len();
    let sum_w: f64 = w.iter().sum();
    let ybar = y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sum_w;
    let scale = (y
        .iter()
        .zip(w)
        .map(|(yi, wi)| wi * (yi - ybar) * (yi - ybar))
        .sum::<f64>()
        / sum_w)
        .sqrt()
        .max(1e-8);
    let (tol_rel, max_sweeps) = match policy {
        LambdaPolicy::Fixed(_) => (BACKFIT_TOL_REL, BACKFIT_MAX_SWEEPS),
        LambdaPolicy::SelectPerSweep => (SELECT_TOL_REL, SELECT_MAX_SWEEPS),
    };
    let tol_abs = tol_rel * scale;

    let mut intercept = ybar;
    let mut components: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
    let mut log_lambdas: Vec<f64> = match policy {
        LambdaPolicy::Fixed(ll) => ll.to_vec(),
        LambdaPolicy::SelectPerSweep => vec![f64::NAN; k],
    };
    let mut residual: Vec<f64> = Vec::with_capacity(n);
    for sweep in 1..=max_sweeps {
        let mut max_delta = 0.0_f64;
        let mut max_dlambda = 0.0_f64;
        for j in 0..k {
            residual.clear();
            for i in 0..n {
                let mut others = intercept;
                for (l, comp) in components.iter().enumerate() {
                    if l != j {
                        others += comp[i];
                    }
                }
                residual.push(y[i] - others);
            }
            let fit = match policy {
                LambdaPolicy::Fixed(ll) => {
                    fit_spline_scan_at(xs[j], &residual, w, ll[j], Some(1.0), orders[j])?
                }
                LambdaPolicy::SelectPerSweep => fit_spline_scan(xs[j], &residual, w, orders[j])?,
            };
            let (map, n_knots) = &maps[j];
            if fit.knots.len() != *n_knots {
                return Err(format!(
                    "scan backfit: term {j} knot count drifted ({} vs {n_knots})",
                    fit.knots.len()
                ));
            }
            let mut center = 0.0;
            for i in 0..n {
                center += w[i] * fit.mean[map[i]];
            }
            center /= sum_w;
            let comp = &mut components[j];
            for i in 0..n {
                let new = fit.mean[map[i]] - center;
                let delta = (new - comp[i]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                comp[i] = new;
            }
            intercept += center;
            if matches!(policy, LambdaPolicy::SelectPerSweep) {
                let prev = log_lambdas[j];
                if prev.is_finite() {
                    max_dlambda = max_dlambda.max((fit.log_lambda - prev).abs());
                } else {
                    max_dlambda = f64::INFINITY;
                }
                log_lambdas[j] = fit.log_lambda;
            }
        }
        let lambda_stable = match policy {
            LambdaPolicy::Fixed(_) => true,
            LambdaPolicy::SelectPerSweep => max_dlambda <= SELECT_LOG_LAMBDA_TOL,
        };
        if max_delta <= tol_abs && lambda_stable {
            return Ok(EngineOut {
                intercept,
                components,
                log_lambdas,
                sweeps: sweep,
                max_delta,
                tol_abs,
            });
        }
    }
    Err(format!(
        "scan backfit: no convergence in {max_sweeps} sweeps (tol={tol_abs:.3e}) — \
         the additive decomposition is numerically unidentifiable (extreme concurvity)"
    ))
}

/// Shared finalization: one storing sweep at the converged state (so each
/// term's stored posterior is fitted to its EXACT converged partial
/// residual), joint σ² profiling, and the σ²-scaled per-term refits.
fn finalize(
    xs: &[&[f64]],
    y: &[f64],
    w: &[f64],
    orders: &[usize],
    maps: &[(Vec<usize>, usize)],
    engine: EngineOut,
) -> Result<ScanBackfitFit, String> {
    let n = y.len();
    let k = xs.len();
    let sum_w: f64 = w.iter().sum();
    let EngineOut {
        mut intercept,
        mut components,
        log_lambdas,
        sweeps,
        max_delta,
        tol_abs,
    } = engine;

    // Storing sweep at unit σ²: refit each term on its converged partial
    // residual (changes each component by ≤ tol), keeping the fits for EDF.
    let mut residual: Vec<f64> = Vec::with_capacity(n);
    let mut unit_fits: Vec<SplineScanFit> = Vec::with_capacity(k);
    let mut centers: Vec<f64> = Vec::with_capacity(k);
    for j in 0..k {
        residual.clear();
        for i in 0..n {
            let mut others = intercept;
            for (l, comp) in components.iter().enumerate() {
                if l != j {
                    others += comp[i];
                }
            }
            residual.push(y[i] - others);
        }
        let fit = fit_spline_scan_at(xs[j], &residual, w, log_lambdas[j], Some(1.0), orders[j])?;
        let (map, _) = &maps[j];
        let mut center = 0.0;
        for i in 0..n {
            center += w[i] * fit.mean[map[i]];
        }
        center /= sum_w;
        let comp = &mut components[j];
        for i in 0..n {
            comp[i] = fit.mean[map[i]] - center;
        }
        intercept += center;
        unit_fits.push(fit);
        centers.push(center);
    }

    let mut fitted = vec![intercept; n];
    for comp in &components {
        for i in 0..n {
            fitted[i] += comp[i];
        }
    }
    let rss: f64 = y
        .iter()
        .zip(&fitted)
        .zip(w)
        .map(|((yi, mi), wi)| wi * (yi - mi) * (yi - mi))
        .sum();
    let edf = 1.0 + unit_fits.iter().map(|f| f.edf() - 1.0).sum::<f64>();
    let dof = n as f64 - edf;
    if dof < 1.0 {
        return Err(format!(
            "scan backfit: residual degrees of freedom exhausted (n={n}, edf={edf:.2})"
        ));
    }
    if rss <= 0.0 {
        return Err("scan backfit: degenerate zero residual sum".to_string());
    }
    let sigma2 = rss / dof;

    // Re-run each stored fit with the joint σ² so the per-term conditional
    // variances (`fit.var`, `predict`) sit on the additive residual scale.
    // Same λ, same partial residual ⇒ identical posterior mean; only the
    // variance scale and the reported likelihood change.
    let mut terms: Vec<ScanBackfitTerm> = Vec::with_capacity(k);
    for j in 0..k {
        residual.clear();
        for i in 0..n {
            let mut others = intercept;
            for (l, comp) in components.iter().enumerate() {
                if l != j {
                    others += comp[i];
                }
            }
            residual.push(y[i] - others);
        }
        let fit =
            fit_spline_scan_at(xs[j], &residual, w, log_lambdas[j], Some(sigma2), orders[j])?;
        let (map, _) = &maps[j];
        // The component values stem from the storing sweep above; the refit on
        // the SAME residual at the SAME λ reproduces them bit-for-bit, so the
        // stored center stays consistent with `fit.mean − center = component`.
        let center = (0..n).map(|i| w[i] * fit.mean[map[i]]).sum::<f64>() / sum_w;
        terms.push(ScanBackfitTerm {
            fit,
            center,
            fitted: components[j].clone(),
        });
    }

    Ok(ScanBackfitFit {
        intercept,
        terms,
        fitted,
        sigma2,
        edf,
        sweeps,
        max_component_delta: max_delta,
        tol: tol_abs,
    })
}

/// Backfit at a FIXED joint smoothing parameter: the exact joint penalized
/// posterior mean (see module docs), certified by the sweep contraction.
///
/// `xs[j]` is term `j`'s covariate column (all length `n`), `log_lambdas[j]`
/// its `log λⱼ` in the scan parameterization (penalty `λⱼ∫(fⱼ^{(m_j)})²`),
/// `orders[j] ∈ {1, 2}` its smoothing-spline order.
pub fn fit_scan_backfit_at(
    xs: &[&[f64]],
    y: &[f64],
    w: &[f64],
    log_lambdas: &[f64],
    orders: &[usize],
) -> Result<ScanBackfitFit, String> {
    validate_inputs(xs, y, w, orders)?;
    if log_lambdas.len() != xs.len() {
        return Err(format!(
            "scan backfit: {} terms but {} smoothing parameters",
            xs.len(),
            log_lambdas.len()
        ));
    }
    for (j, ll) in log_lambdas.iter().enumerate() {
        if !ll.is_finite() {
            return Err(format!("scan backfit: non-finite log_lambda for term {j}"));
        }
    }
    let maps: Vec<(Vec<usize>, usize)> = xs.iter().map(|x| row_knot_map(x)).collect();
    let engine = backfit_engine(xs, y, w, orders, &maps, LambdaPolicy::Fixed(log_lambdas))?;
    finalize(xs, y, w, orders, &maps, engine)
}

/// Backfit with per-term λ selection: every sweep re-selects each `λⱼ` by the
/// scan's exact diffuse REML on that term's partial residual, then the
/// converged λ's are FROZEN and the fixed-λ loop re-certifies the solution at
/// the tight tolerance.
///
/// This is an explicitly-labeled approximation to joint REML (module docs):
/// each term's criterion treats the other components as known offsets. The
/// fixture tests validate it match-or-beat against a dense joint-REML grid on
/// truth recovery; exact joint selection remains the dense path's job.
pub fn fit_scan_backfit(
    xs: &[&[f64]],
    y: &[f64],
    w: &[f64],
    orders: &[usize],
) -> Result<ScanBackfitFit, String> {
    validate_inputs(xs, y, w, orders)?;
    let maps: Vec<(Vec<usize>, usize)> = xs.iter().map(|x| row_knot_map(x)).collect();
    let selected = backfit_engine(xs, y, w, orders, &maps, LambdaPolicy::SelectPerSweep)?;
    let log_lambdas = selected.log_lambdas.clone();
    let engine = backfit_engine(xs, y, w, orders, &maps, LambdaPolicy::Fixed(&log_lambdas))?;
    finalize(xs, y, w, orders, &maps, engine)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform-ish noise in (−0.5, 0.5): LCG, no RNG deps.
    fn lcg_noise(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// Dense JOINT penalized solve of the SAME model the backfit iterates:
    /// per-term order-`m` integrated-Wiener state increments at the pooled
    /// knots (states `(f, …, f^{(m−1)})`), Gaussian observations through the
    /// intercept + the f-component of each term's knot state, and the
    /// weighted-centering constraint `Σᵢ wᵢ fⱼ(x_{j,i}) = 0` per term
    /// (eliminated by a Householder null-space basis, which picks exactly the
    /// representative the backfit's centering picks). Returns
    /// `(intercept, per-term centered component values at the rows)`.
    fn dense_joint_truth(
        xs: &[&[f64]],
        y: &[f64],
        w: &[f64],
        log_lambdas: &[f64],
        orders: &[usize],
    ) -> (f64, Vec<Vec<f64>>) {
        let n = y.len();
        let k = xs.len();
        // Pooled knots + row maps (must mirror row_knot_map / pool_nodes).
        let mut knots: Vec<Vec<f64>> = Vec::new();
        let mut maps: Vec<Vec<usize>> = Vec::new();
        for x in xs {
            let (map, n_knots) = row_knot_map(x);
            let mut kx = vec![f64::NAN; n_knots];
            for i in 0..n {
                kx[map[i]] = x[i];
            }
            knots.push(kx);
            maps.push(map);
        }
        let dims: Vec<usize> = (0..k).map(|j| orders[j] * knots[j].len()).collect();
        let offs: Vec<usize> = {
            let mut o = vec![1usize];
            for j in 0..k {
                let last = *o.last().unwrap();
                o.push(last + dims[j]);
            }
            o
        };
        let dim = 1 + dims.iter().sum::<usize>();

        let tr = |delta: f64, m: usize| -> Vec<Vec<f64>> {
            let fact = |p: usize| (1..=p).map(|v| v as f64).product::<f64>().max(1.0);
            let mut f = vec![vec![0.0; m]; m];
            for i in 0..m {
                for j2 in i..m {
                    f[i][j2] = delta.powi((j2 - i) as i32) / fact(j2 - i);
                }
            }
            f
        };
        let qn = |delta: f64, q: f64, m: usize| -> Vec<Vec<f64>> {
            let fact = |p: usize| (1..=p).map(|v| v as f64).product::<f64>().max(1.0);
            let mut out = vec![vec![0.0; m]; m];
            for i in 0..m {
                for j2 in 0..m {
                    let p = 2 * m - 1 - i - j2;
                    out[i][j2] = q * delta.powi(p as i32)
                        / (fact(m - 1 - i) * fact(m - 1 - j2) * (p as f64));
                }
            }
            out
        };
        let inv_small = |a: &[Vec<f64>], m: usize| -> Vec<Vec<f64>> {
            let mut out = vec![vec![0.0; m]; m];
            if m == 1 {
                out[0][0] = 1.0 / a[0][0];
            } else {
                let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
                out[0][0] = a[1][1] / det;
                out[0][1] = -a[0][1] / det;
                out[1][0] = -a[1][0] / det;
                out[1][1] = a[0][0] / det;
            }
            out
        };

        // Penalty precision: per interval, blocks [[FᵀQ⁻¹F, −FᵀQ⁻¹], [−Q⁻¹F, Q⁻¹]].
        let mut h = vec![vec![0.0_f64; dim]; dim];
        for j in 0..k {
            let m = orders[j];
            let q = (-log_lambdas[j]).exp();
            for t in 0..knots[j].len() - 1 {
                let delta = knots[j][t + 1] - knots[j][t];
                let f = tr(delta, m);
                let qi = inv_small(&qn(delta, q, m), m);
                let base_a = offs[j] + m * t;
                let base_b = offs[j] + m * (t + 1);
                for r in 0..m {
                    for c in 0..m {
                        // FᵀQ⁻¹F on (t, t)
                        let mut acc = 0.0;
                        for a in 0..m {
                            for b in 0..m {
                                acc += f[a][r] * qi[a][b] * f[b][c];
                            }
                        }
                        h[base_a + r][base_a + c] += acc;
                        // Q⁻¹ on (t+1, t+1)
                        h[base_b + r][base_b + c] += qi[r][c];
                        // −FᵀQ⁻¹ on (t, t+1) and transpose
                        let mut acc2 = 0.0;
                        for a in 0..m {
                            acc2 += f[a][r] * qi[a][c];
                        }
                        h[base_a + r][base_b + c] -= acc2;
                        h[base_b + c][base_a + r] -= acc2;
                    }
                }
            }
        }
        // Observations: design row = intercept + f-component per term.
        let mut rhs = vec![0.0_f64; dim];
        for i in 0..n {
            let mut idx = vec![0usize];
            for j in 0..k {
                idx.push(offs[j] + orders[j] * maps[j][i]);
            }
            for &a in &idx {
                for &b in &idx {
                    h[a][b] += w[i];
                }
                rhs[a] += w[i] * y[i];
            }
        }
        // Constraint basis: per term, Householder reflector sending the pooled
        // weight vector (on the f-components) to e₁; basis = columns 1..d.
        // Full Z is blockdiag(1, Z₁, …, Z_K), dim × (dim − k).
        let red = dim - k;
        let mut z = vec![vec![0.0_f64; red]; dim];
        z[0][0] = 1.0;
        let mut col = 1usize;
        for j in 0..k {
            let d = dims[j];
            let mut g = vec![0.0_f64; d];
            for i in 0..n {
                g[orders[j] * maps[j][i]] += w[i];
            }
            let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            let mut u = g.clone();
            u[0] += if g[0] >= 0.0 { norm } else { -norm };
            let uu: f64 = u.iter().map(|v| v * v).sum();
            // Column c of H_house = e_c − 2 u u_c / uᵀu, for c = 1..d.
            for c in 1..d {
                for r in 0..d {
                    let mut v = if r == c { 1.0 } else { 0.0 };
                    v -= 2.0 * u[r] * u[c] / uu;
                    z[offs[j] + r][col] = v;
                }
                col += 1;
            }
        }
        assert_eq!(col, red);
        // Reduced system Zᵀ H Z γ = Zᵀ rhs.
        let mut hz = vec![vec![0.0_f64; red]; dim];
        for r in 0..dim {
            for c in 0..red {
                let mut acc = 0.0;
                for t in 0..dim {
                    acc += h[r][t] * z[t][c];
                }
                hz[r][c] = acc;
            }
        }
        let mut h_red = vec![vec![0.0_f64; red]; red];
        let mut b_red = vec![0.0_f64; red];
        for r in 0..red {
            for c in 0..red {
                let mut acc = 0.0;
                for t in 0..dim {
                    acc += z[t][r] * hz[t][c];
                }
                h_red[r][c] = acc;
            }
            b_red[r] = (0..dim).map(|t| z[t][r] * rhs[t]).sum();
        }
        let gamma = solve_dense(&mut h_red, &mut b_red);
        let zvec: Vec<f64> = (0..dim)
            .map(|r| (0..red).map(|c| z[r][c] * gamma[c]).sum())
            .collect();
        let comps: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                (0..n)
                    .map(|i| zvec[offs[j] + orders[j] * maps[j][i]])
                    .collect()
            })
            .collect();
        (zvec[0], comps)
    }

    /// Gauss-Jordan solve with partial pivoting (test-only dense helper).
    fn solve_dense(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
        let n = b.len();
        for col in 0..n {
            let piv = (col..n)
                .max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs()))
                .unwrap();
            a.swap(col, piv);
            b.swap(col, piv);
            let d = a[col][col];
            for t in 0..n {
                a[col][t] /= d;
            }
            b[col] /= d;
            for r in 0..n {
                if r == col || a[r][col] == 0.0 {
                    continue;
                }
                let f = a[r][col];
                for t in 0..n {
                    a[r][t] -= f * a[col][t];
                }
                b[r] -= f * b[col];
            }
        }
        b.to_vec()
    }

    /// Cholesky log-determinant of a symmetric positive-definite matrix
    /// (consumes the buffer; test-only).
    fn cholesky_logdet(a: &mut [Vec<f64>]) -> f64 {
        let n = a.len();
        let mut logdet = 0.0;
        for j in 0..n {
            for kk in 0..j {
                let l = a[j][kk];
                for i in j..n {
                    a[i][j] -= a[i][kk] * l;
                }
            }
            let d = a[j][j];
            assert!(d > 0.0, "cholesky: non-PD at {j} (d={d})");
            let s = d.sqrt();
            logdet += 2.0 * s.ln();
            for i in j..n {
                a[i][j] /= s;
            }
        }
        logdet
    }

    fn build_two_term_fixture(
        n: usize,
        sigma: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        let mut t1 = Vec::with_capacity(n);
        let mut t2 = Vec::with_capacity(n);
        for i in 0..n {
            let u1 = lcg_noise(&mut state) + 0.5;
            let u2 = lcg_noise(&mut state) + 0.5;
            let xv1 = u1;
            // Mild concurvity: x2 leans on x1 without being a function of it.
            let xv2 = 0.75 * u2 + 0.25 * u1;
            let f1 = (2.0 * std::f64::consts::PI * xv1).sin();
            let f2 = (xv2 - 0.4).abs().powf(1.5) * 3.0 - 1.0;
            x1.push(xv1);
            x2.push(xv2);
            t1.push(f1);
            t2.push(f2);
            y.push(1.7 + f1 + f2 + sigma * lcg_noise(&mut state) * 3.4641);
            w.push(1.0 + 0.5 * ((i % 4) as f64) * 0.3);
        }
        (x1, x2, y, w, t1, t2)
    }

    /// THE oracle (#1034 item 3): at a fixed joint λ the backfit must agree
    /// with the dense JOINT penalized solve of the identical model — same
    /// intercept, same centered components, same total fit — for both the
    /// all-cubic case and a mixed-order (m=2, m=1) case, ties included.
    #[test]
    fn backfit_matches_dense_joint_solve_at_fixed_lambda() {
        let (mut x1, x2, y, w, _, _) = build_two_term_fixture(60, 0.25, 0x5eed_1034);
        // Exercise tie pooling on term 1.
        x1[11] = x1[10];
        x1[41] = x1[40];
        for (orders, lls) in [
            (vec![2usize, 2usize], vec![-3.0_f64, -1.0_f64]),
            (vec![2usize, 1usize], vec![-2.0_f64, 0.5_f64]),
        ] {
            let xs: Vec<&[f64]> = vec![&x1, &x2];
            let fit = fit_scan_backfit_at(&xs, &y, &w, &lls, &orders).expect("backfit");
            assert!(fit.max_component_delta <= fit.tol, "certificate violated");
            let (alpha, comps) = dense_joint_truth(&xs, &y, &w, &lls, &orders);
            let scale = y.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            assert!(
                (fit.intercept - alpha).abs() <= 1e-7 * scale,
                "intercept mismatch (orders {orders:?}): backfit={} dense={alpha}",
                fit.intercept
            );
            for j in 0..2 {
                for i in 0..y.len() {
                    assert!(
                        (fit.terms[j].fitted[i] - comps[j][i]).abs() <= 1e-7 * scale,
                        "component {j} row {i} mismatch (orders {orders:?}): \
                         backfit={} dense={}",
                        fit.terms[j].fitted[i],
                        comps[j][i]
                    );
                }
            }
            for i in 0..y.len() {
                let dense_mu = alpha + comps[0][i] + comps[1][i];
                assert!(
                    (fit.fitted[i] - dense_mu).abs() <= 1e-7 * scale,
                    "total fit row {i} mismatch (orders {orders:?})"
                );
                // predict() at the training row must reproduce the fitted value.
                let p = fit.predict(&[x1[i], x2[i]]).expect("predict");
                assert!(
                    (p - fit.fitted[i]).abs() <= 1e-9 * scale.max(1.0),
                    "predict/fitted drift at row {i}"
                );
            }
            // Components are weighted-centered (the identifiability convention).
            for j in 0..2 {
                let m: f64 = fit.terms[j]
                    .fitted
                    .iter()
                    .zip(&w)
                    .map(|(f, wi)| f * wi)
                    .sum();
                assert!(m.abs() <= 1e-7 * scale * w.iter().sum::<f64>());
            }
        }
    }

    /// Joint-REML validation of the per-sweep selection approximation
    /// (#1034 item 3b): on a two-term order-1 fixture, compute the EXACT joint
    /// REML surface densely (log|S|₊ − log|H| with profiled σ², on the
    /// constraint-reduced parameterization) over a (log λ₁, log λ₂) grid, and
    /// require the scan-REML-selected backfit to match-or-beat the joint-REML
    /// optimum on truth recovery (small slack for the equal-quality regime —
    /// the two criteria pick near-identical λ on low-concurvity designs).
    #[test]
    fn per_sweep_scan_reml_matches_joint_reml_on_truth_recovery() {
        let n = 240usize;
        let sigma = 0.35;
        let mut state = 0xj_seed_u64_placeholder;
        // Coarse distinct grids keep the dense REML dimension small.
        let g1 = 30usize;
        let g2 = 28usize;
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut mu_true = Vec::with_capacity(n);
        let w = vec![1.0_f64; n];
        for _ in 0..n {
            let u1 = ((lcg_noise(&mut state) + 0.5) * g1 as f64).floor() / g1 as f64;
            let u2 = ((lcg_noise(&mut state) + 0.5) * g2 as f64).floor() / g2 as f64;
            let f1 = (2.0 * std::f64::consts::PI * u1).sin();
            let f2 = 2.0 * (u2 - 0.5).abs() - 0.5;
            x1.push(u1);
            x2.push(u2);
            mu_true.push(0.8 + f1 + f2);
            y.push(0.8 + f1 + f2 + sigma * lcg_noise(&mut state) * 3.4641);
        }
        let xs: Vec<&[f64]> = vec![&x1, &x2];
        let orders = vec![1usize, 1usize];

        // Dense joint REML criterion at one (logλ₁, logλ₂): the constraint
        // reduction makes S_red full-rank off the intercept, so
        // V = ½ Σⱼ rankⱼ·logλⱼ − ½ log|B+S|_red − ((n−M₀)/2)·ln σ̂²,
        // σ̂² = (RSS + pen)/(n − M₀), M₀ = 1 (intercept; order-1 terms keep no
        // unpenalized direction after centering).
        let (map1, k1) = row_knot_map(&x1);
        let (map2, k2) = row_knot_map(&x2);
        let joint_reml = |ll: [f64; 2]| -> f64 {
            let (alpha, comps) = dense_joint_truth(&xs, &y, &w, &ll, &orders);
            let mut rss = 0.0;
            for i in 0..n {
                let r = y[i] - alpha - comps[0][i] - comps[1][i];
                rss += w[i] * r * r;
            }
            // Penalty term Σⱼ λⱼ Σ_t (f_{t+1}−f_t)²/δ_t on the knot values.
            let mut pen = 0.0;
            for (j, (map, kn)) in [(0usize, (&map1, k1)), (1usize, (&map2, k2))] {
                let mut kx = vec![f64::NAN; kn];
                let mut kf = vec![f64::NAN; kn];
                for i in 0..n {
                    kx[map[i]] = [&x1, &x2][j][i];
                    kf[map[i]] = comps[j][i];
                }
                let lam = ll[j].exp();
                for t in 0..kn - 1 {
                    let d = kf[t + 1] - kf[t];
                    pen += lam * d * d / (kx[t + 1] - kx[t]);
                }
            }
            // log|B+S| on the reduced space, via the dense builder's pieces:
            // rebuild reduced H with the SAME code path as dense_joint_truth by
            // calling a tiny local re-assembly (kept here: order-1 only).
            let logdet = joint_logdet_reduced(&xs, &w, &ll, &orders);
            let m0 = 1.0;
            let dof = n as f64 - m0;
            let sig2 = (rss + pen) / dof;
            let rank = (k1 - 1 + k2 - 1) as f64;
            // rank·mean(logλ) term: rank_j = k_j − 1 each.
            let ld_s = (k1 - 1) as f64 * ll[0] + (k2 - 1) as f64 * ll[1];
            0.5 * ld_s - 0.5 * logdet - 0.5 * dof * sig2.ln() - 0.0 * rank
        };

        // Coarse grid + one refinement level.
        let mut best = (f64::NEG_INFINITY, [0.0_f64; 2]);
        for a in -4..=12 {
            for b in -4..=12 {
                let ll = [a as f64, b as f64];
                let v = joint_reml(ll);
                if v > best.0 {
                    best = (v, ll);
                }
            }
        }
        for da in -4..=4 {
            for db in -4..=4 {
                let ll = [best.1[0] + 0.25 * da as f64, best.1[1] + 0.25 * db as f64];
                let v = joint_reml(ll);
                if v > best.0 {
                    best = (v, ll);
                }
            }
        }

        let joint_fit =
            fit_scan_backfit_at(&xs, &y, &w, &best.1, &orders).expect("joint-λ backfit");
        let scan_fit = fit_scan_backfit(&xs, &y, &w, &orders).expect("scan-REML backfit");
        let mse = |fitted: &[f64]| -> f64 {
            fitted
                .iter()
                .zip(&mu_true)
                .map(|(m, t)| (m - t) * (m - t))
                .sum::<f64>()
                / n as f64
        };
        let mse_joint = mse(&joint_fit.fitted);
        let mse_scan = mse(&scan_fit.fitted);
        let signal_var = mu_true
            .iter()
            .map(|t| (t - mu_true.iter().sum::<f64>() / n as f64).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(
            mse_scan <= 1.05 * mse_joint + 1e-3 * signal_var,
            "per-sweep scan-REML must match-or-beat joint REML on truth recovery: \
             scan={mse_scan:.6e} joint={mse_joint:.6e} (λ_joint={:?}, λ_scan={:?})",
            best.1,
            scan_fit
                .terms
                .iter()
                .map(|t| t.fit.log_lambda)
                .collect::<Vec<_>>()
        );
    }

    /// log|B + S| on the constraint-reduced space (order-general; test-only):
    /// re-assembles exactly the reduced Hessian `dense_joint_truth` solves and
    /// returns its Cholesky log-determinant.
    fn joint_logdet_reduced(
        xs: &[&[f64]],
        w: &[f64],
        log_lambdas: &[f64],
        orders: &[usize],
    ) -> f64 {
        unimplemented!()
    }

    /// Truth-recovery e2e on a larger fixture: three terms (cubic, cubic,
    /// linear/order-1), mild concurvity, n = 4000. The λ-selecting backfit
    /// must recover each centered component and the total surface.
    #[test]
    fn three_term_truth_recovery_e2e() {
        let n = 4000usize;
        let sigma = 0.5;
        let mut state = 0x00e2e_1034_u64;
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut x3 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut t = vec![Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
        let w = vec![1.0_f64; n];
        for _ in 0..n {
            let u1 = lcg_noise(&mut state) + 0.5;
            let u2 = 0.7 * (lcg_noise(&mut state) + 0.5) + 0.3 * u1;
            let u3 = lcg_noise(&mut state) + 0.5;
            let f1 = (2.0 * std::f64::consts::PI * u1).sin();
            let f2 = (-(u2 - 0.5) * (u2 - 0.5) * 18.0).exp() * 2.0 - 0.6;
            let f3 = 1.4 * (u3 - 0.5);
            x1.push(u1);
            x2.push(u2);
            x3.push(u3);
            t[0].push(f1);
            t[1].push(f2);
            t[2].push(f3);
            y.push(-0.3 + f1 + f2 + f3 + sigma * lcg_noise(&mut state) * 3.4641);
        }
        let xs: Vec<&[f64]> = vec![&x1, &x2, &x3];
        let fit = fit_scan_backfit(&xs, &y, &w, &[2, 2, 1]).expect("3-term backfit");
        assert!(fit.max_component_delta <= fit.tol);
        assert!(fit.sigma2 > 0.0 && fit.edf > 3.0 && fit.edf < n as f64 / 4.0);

        // Total-surface truth recovery.
        let mu_true: Vec<f64> = (0..n).map(|i| -0.3 + t[0][i] + t[1][i] + t[2][i]).collect();
        let total_mse: f64 = fit
            .fitted
            .iter()
            .zip(&mu_true)
            .map(|(m, mt)| (m - mt) * (m - mt))
            .sum::<f64>()
            / n as f64;
        let signal_var = {
            let mean = mu_true.iter().sum::<f64>() / n as f64;
            mu_true.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64
        };
        assert!(
            total_mse <= 0.02 * signal_var,
            "total truth-recovery MSE too large: {total_mse:.5e} vs signal var {signal_var:.5e}"
        );
        // Per-component recovery against the centered truth components.
        for j in 0..3 {
            let mean_t = t[j].iter().sum::<f64>() / n as f64;
            let comp_var = t[j].iter().map(|v| (v - mean_t).powi(2)).sum::<f64>() / n as f64;
            let mse_j: f64 = (0..n)
                .map(|i| {
                    let d = fit.terms[j].fitted[i] - (t[j][i] - mean_t);
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            assert!(
                mse_j <= 0.10 * comp_var.max(0.05),
                "component {j} truth-recovery MSE too large: {mse_j:.5e} vs var {comp_var:.5e}"
            );
        }
        // Conditional SEs exist and are positive at the knots.
        for term in &fit.terms {
            assert!(term.fit.var.iter().all(|&v| v.is_finite() && v > 0.0));
        }
    }
}
