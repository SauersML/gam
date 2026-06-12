//! Wood-style smooth-component Wald tests.
//!
//! The test follows the rank-truncated covariance inverse used by Wood (2013):
//! the penalized part of the coefficient block is tested with a pseudo-inverse
//! of rank approximately equal to the term EDF, while unpenalized null-space
//! coefficients are kept full-rank. The reference degrees of freedom use the
//! coefficient-space influence block `F_jj = (H⁻¹ X'WX)_jj`.

use crate::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView1, s};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};
use std::ops::Range;

/// Whether the residual dispersion `φ` is known or estimated from the
/// fit.  Selects the reference distribution for the Wald p-value: `Known`
/// → `χ²_{ref_df}` (e.g. binomial/Poisson), `Estimated` → `F_{ref_df,
/// residual_df}` (e.g. Gaussian where `φ̂` carries its own sampling
/// variability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothTestScale {
    Known,
    Estimated,
}

/// Inputs to `wood_smooth_test`. `beta` is the full coefficient vector;
/// the term block being tested is `beta[coeff_range]`. `covariance` is the
/// matching posterior covariance Σ̂ (full p×p; the diagonal block is sliced
/// out). **`covariance` must be the scale-included posterior covariance**
/// (mgcv `Vb`/`Vp`, i.e. `H⁻¹` already multiplied by the dispersion `φ̂`),
/// so the Wald statistic `T = β̂'·Σ̂⁻·β̂` is dimensionless — the residual
/// dispersion has already been divided out and the F-statistic is `T/ref_df`
/// with *no* further `φ̂` factor. `influence_matrix` is the optional
/// coefficient-space influence `F = H⁻¹ X'WX`; when present
/// `tr(F_jj)² / tr(F_jj²)` is used as the Wood-corrected reference d.f.
/// `edf` is the smooth's effective d.f. (rank truncation for the penalized
/// subblock); `nullspace_dim` is the fixed-effect (unpenalized) leading
/// dimension within the block. `residual_df` is the denominator d.f. for the
/// `Estimated`-scale F branch.
#[derive(Debug, Clone)]
pub struct SmoothTestInput<'a> {
    pub beta: ArrayView1<'a, f64>,
    pub covariance: &'a Array2<f64>,
    pub influence_matrix: Option<&'a Array2<f64>>,
    pub coeff_range: Range<usize>,
    pub edf: f64,
    pub nullspace_dim: usize,
    pub residual_df: f64,
    pub scale: SmoothTestScale,
    /// Lawley (1956) second-order LR mean shift `Δε = ε_k − ε_{k−q}` for the
    /// tested block (issue #939), from
    /// [`crate::inference::lawley::lawley_lr_mean_shift`] evaluated on the
    /// family cumulant jets at the fit. When present, the `Known`-scale branch
    /// reports the Bartlett-corrected p-value (`c = 1 + Δε/ref_df`, the factor
    /// completed with the same trace-corrected reference d.f. the χ² tail
    /// uses) alongside the first-order one. Ignored by the `Estimated` branch,
    /// which has its own closed-form conjugate factor. `None` ⇒ first-order
    /// only.
    pub known_scale_lr_mean_shift: Option<f64>,
}

/// Output of `wood_smooth_test`: the Wald statistic
/// `T = β̂'·Σ̂⁻ᵣ·β̂` (rank-`r` pseudo-inverse on the penalized subblock,
/// full-rank on the nullspace subblock), the reference d.f. used to compute
/// the tail probability, and the resulting `p_value` (clamped to `[0,1]`).
#[derive(Debug, Clone)]
pub struct SmoothTestResult {
    pub statistic: f64,
    pub ref_df: f64,
    pub p_value: f64,
    /// Second-order-accurate (Bartlett-corrected) p-value, reported *alongside*
    /// — never replacing — the first-order `p_value` (issue #939). Populated for
    /// the estimated-scale (Gaussian-linear / penalized-quadratic conjugate)
    /// branch, where the exact Bartlett factor is a closed form of the reference
    /// and residual degrees of freedom, and for the known-scale branch when the
    /// caller supplies the Lawley LR mean shift from the family cumulant jets
    /// (`SmoothTestInput::known_scale_lr_mean_shift`).
    pub p_value_corrected: Option<f64>,
    /// The Bartlett factor `c = E[W]/d` used for `p_value_corrected`. Its
    /// distance from `1` is the per-test diagnostic the field lacks: how far
    /// first-order inference is distorted at this `n`.
    pub bartlett_factor: Option<f64>,
}

/// Wood (2013) rank-truncated Wald smooth-component test.
///
/// Splits `beta[coeff_range]` into a leading `nullspace_dim` unpenalized
/// block (tested at full rank) and a trailing penalized block (tested at
/// rank `round(edf − nullspace_dim)` via the spectral truncated
/// pseudo-inverse of the matching Σ̂ subblock). The combined statistic
/// `T` is compared against `χ²_{ref_df}` when the scale is `Known`, or
/// `F = T/ref_df` against `F_{ref_df, residual_df}` when `Estimated`.
///
/// Because `covariance` is the scale-included posterior covariance, `T`
/// already has the dispersion `φ̂` divided out (it is a proper Wald χ²);
/// the estimated-scale F-statistic is therefore `T/ref_df` with no extra
/// `φ̂` factor. Dividing by `φ̂` a second time — the historical defect
/// fixed in issue #675 — makes the p-value scale as `1/φ̂` and so depend on
/// the units of the response. Returns `None` on degenerate inputs (empty
/// block, non-finite EDF, non-finite stat, or non-positive residual d.f.
/// in the F branch).
pub fn wood_smooth_test(input: SmoothTestInput<'_>) -> Option<SmoothTestResult> {
    let start = input.coeff_range.start;
    let end = input.coeff_range.end;
    if start >= end
        || end > input.beta.len()
        || end > input.covariance.nrows()
        || end > input.covariance.ncols()
        || !input.edf.is_finite()
        || input.edf <= 0.0
    {
        return None;
    }
    let k = end - start;
    let beta = input.beta.slice(s![start..end]).to_owned();
    let cov = block(input.covariance, start, end)?;
    let null_dim = input.nullspace_dim.min(k);
    let pen_dim = k.saturating_sub(null_dim);

    let mut statistic = 0.0;
    if null_dim > 0 {
        let beta_null = beta.slice(s![0..null_dim]).to_owned();
        let cov_null = cov.slice(s![0..null_dim, 0..null_dim]).to_owned();
        statistic += full_rank_quadratic(&beta_null, &cov_null)?;
    }
    if pen_dim > 0 {
        let beta_pen = beta.slice(s![null_dim..k]).to_owned();
        let cov_pen = cov.slice(s![null_dim..k, null_dim..k]).to_owned();
        let rank = truncated_rank(input.edf - null_dim as f64, pen_dim);
        if rank > 0 {
            statistic += truncated_quadratic(&beta_pen, &cov_pen, rank)?;
        }
    }

    let ref_df = reference_df(input.influence_matrix, start, end).unwrap_or(input.edf.max(1e-12));
    if !statistic.is_finite() || statistic < 0.0 || !ref_df.is_finite() || ref_df <= 0.0 {
        return None;
    }
    let mut p_value_corrected: Option<f64> = None;
    let mut bartlett_factor: Option<f64> = None;
    let p_value = match input.scale {
        SmoothTestScale::Known => {
            let dist = ChiSquared::new(ref_df).ok()?;
            // Second-order Bartlett correction for the known-scale branch
            // (#939). Lawley (1956): E[W] = d + Δε + O(n⁻²) with Δε =
            // ε_k − ε_{k−q} assembled by the caller from the exact family
            // cumulant jets (`crate::inference::lawley`), so the factor is
            // c = E[W]/d = 1 + Δε/d with d the same trace-corrected reference
            // d.f. the χ² tail uses. Degenerate factors (non-finite, ≤ 0) fall
            // back to first-order-only reporting inside `bartlett_correct`.
            // Reported alongside, never replacing, the first-order p-value.
            if let Some(shift) = input.known_scale_lr_mean_shift {
                if shift.is_finite() {
                    let c = 1.0 + shift / ref_df;
                    if let Some(corr) =
                        crate::inference::higher_order::bartlett_correct(statistic, ref_df, c)
                    {
                        p_value_corrected = Some(corr.corrected_p_value);
                        bartlett_factor = Some(corr.factor);
                    }
                }
            }
            1.0 - dist.cdf(statistic)
        }
        SmoothTestScale::Estimated => {
            if !input.residual_df.is_finite() || input.residual_df <= 0.0 {
                return None;
            }
            // `statistic` is already a dispersion-free Wald χ² (the covariance
            // is scale-included), so the estimated-scale F-statistic is the
            // χ² divided by its reference d.f. only — mgcv's `Tr/rank`. Dividing
            // by `φ̂` again would re-introduce a response-unit dependence (#675).
            let f_stat = statistic / ref_df;
            let dist = FisherSnedecor::new(ref_df, input.residual_df).ok()?;
            // Second-order Bartlett correction (#939). The estimated-scale branch
            // is the Gaussian-linear / penalized-quadratic conjugate case, where
            // the exact Bartlett factor `c = 1 + (q+1)/(2ν)` (q = ref_df,
            // ν = residual_df) maps the first-order reference toward the exact
            // distribution. Rescale the statistic by `c` and re-reference; this
            // recovers the nominal mean to second order at finite n. Reported
            // alongside, never replacing, the first-order p-value.
            if let Some(c) = crate::inference::higher_order::gaussian_linear_bartlett_factor(
                ref_df,
                input.residual_df,
            ) {
                let f_corrected = f_stat / c;
                let p_corr = (1.0 - dist.cdf(f_corrected)).clamp(0.0, 1.0);
                if p_corr.is_finite() {
                    p_value_corrected = Some(p_corr);
                    bartlett_factor = Some(c);
                }
            }
            1.0 - dist.cdf(f_stat)
        }
    };
    if !p_value.is_finite() {
        return None;
    }
    Some(SmoothTestResult {
        statistic,
        ref_df,
        p_value: p_value.clamp(0.0, 1.0),
        p_value_corrected,
        bartlett_factor,
    })
}

fn truncated_rank(edf_pen: f64, pen_dim: usize) -> usize {
    if pen_dim == 0 || !edf_pen.is_finite() || edf_pen <= 0.0 {
        return 0;
    }
    (edf_pen.round() as usize).clamp(1, pen_dim)
}

fn block(matrix: &Array2<f64>, start: usize, end: usize) -> Option<Array2<f64>> {
    if start >= end || end > matrix.nrows() || end > matrix.ncols() {
        return None;
    }
    Some(matrix.slice(s![start..end, start..end]).to_owned())
}

fn full_rank_quadratic(beta: &Array1<f64>, cov: &Array2<f64>) -> Option<f64> {
    truncated_quadratic(beta, cov, beta.len())
}

fn truncated_quadratic(beta: &Array1<f64>, cov: &Array2<f64>, rank: usize) -> Option<f64> {
    if beta.is_empty() || cov.nrows() != beta.len() || cov.ncols() != beta.len() || rank == 0 {
        return None;
    }
    let (evals, evecs) = cov.to_owned().eigh(faer::Side::Lower).ok()?;
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    let tol = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        * 1e-10;
    let mut q = 0.0;
    let mut used = 0usize;
    for idx in order {
        let lambda = evals[idx];
        if lambda <= tol {
            continue;
        }
        let v = evecs.column(idx);
        let proj = beta.dot(&v);
        q += proj * proj / lambda;
        used += 1;
        if used >= rank {
            break;
        }
    }
    (used > 0 && q.is_finite()).then_some(q.max(0.0))
}

fn reference_df(influence: Option<&Array2<f64>>, start: usize, end: usize) -> Option<f64> {
    let f = influence?;
    let f_block = block(f, start, end)?;
    let tr = (0..f_block.nrows()).map(|i| f_block[[i, i]]).sum::<f64>();
    let tr2 = f_block.dot(&f_block).diag().sum();
    if tr.is_finite() && tr2.is_finite() && tr > 0.0 && tr2 > 0.0 {
        Some((tr * tr / tr2).max(1e-12))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn reference_df_uses_trace_correction() {
        let beta = array![1.0, 2.0];
        let cov = array![[2.0, 0.0], [0.0, 3.0]];
        let f = array![[0.5, 0.0], [0.0, 0.25]];
        let out = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: Some(&f),
            coeff_range: 0..2,
            edf: 1.0,
            nullspace_dim: 0,
            residual_df: 20.0,
            scale: SmoothTestScale::Known,
            known_scale_lr_mean_shift: None,
        })
        .expect("smooth test");
        assert!((out.ref_df - 1.8).abs() < 1e-12);
        assert!(out.statistic > 0.0);
        assert!((0.0..=1.0).contains(&out.p_value));
    }

    /// Known-scale Bartlett wiring (#939): a positive Lawley mean shift Δε
    /// gives c = 1 + Δε/ref_df > 1, so the corrected statistic shrinks and the
    /// corrected p-value exceeds the first-order one — reported alongside it,
    /// with the exact factor surfaced. No shift ⇒ first-order only.
    #[test]
    fn known_scale_branch_applies_lawley_mean_shift() {
        let beta = array![1.0, 2.0];
        let cov = array![[2.0, 0.0], [0.0, 3.0]];
        let f = array![[0.5, 0.0], [0.0, 0.25]];
        let run = |shift: Option<f64>| {
            wood_smooth_test(SmoothTestInput {
                beta: beta.view(),
                covariance: &cov,
                influence_matrix: Some(&f),
                coeff_range: 0..2,
                edf: 1.0,
                nullspace_dim: 0,
                residual_df: 20.0,
                scale: SmoothTestScale::Known,
                known_scale_lr_mean_shift: shift,
            })
            .expect("smooth test")
        };
        let first_order = run(None);
        assert!(first_order.p_value_corrected.is_none());
        assert!(first_order.bartlett_factor.is_none());

        let shift = 0.36; // Δε ⇒ c = 1 + 0.36/1.8 = 1.2 against ref_df = 1.8.
        let corrected = run(Some(shift));
        assert!((corrected.p_value - first_order.p_value).abs() < 1e-15);
        let c = corrected.bartlett_factor.expect("factor");
        assert!((c - 1.2).abs() < 1e-12, "c = {c}");
        let p_corr = corrected.p_value_corrected.expect("corrected p");
        assert!(
            p_corr > corrected.p_value,
            "c > 1 must enlarge the p-value: {} vs {}",
            p_corr,
            corrected.p_value
        );
        // Degenerate shift (c ≤ 0) must fall back to first-order only.
        let degenerate = run(Some(-3.6));
        assert!(degenerate.p_value_corrected.is_none());
        assert!(degenerate.bartlett_factor.is_none());
    }

    /// Rescaling the response by `c` is `β → c·β`, `Σ → c²·Σ` (the covariance
    /// is scale-included). The Wald statistic `T = β'Σ⁻β` is then invariant,
    /// and — because the estimated-scale F-statistic is `T/ref_df` with no
    /// further `φ̂` factor — so is the p-value. This is the unit-level guard
    /// for issue #675: the historical `T/(ref_df·φ̂)` made the p-value scale
    /// as `1/c²` even though `T` did not move.
    #[test]
    fn estimated_scale_pvalue_is_response_unit_invariant() {
        let beta = array![2.5, -3.5, 1.8];
        let cov = array![[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 0.9]];
        let f = array![[0.7, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.4]];

        let run = |c: f64| {
            let beta_c = &beta * c;
            let cov_c = &cov * (c * c);
            wood_smooth_test(SmoothTestInput {
                beta: beta_c.view(),
                covariance: &cov_c,
                influence_matrix: Some(&f),
                coeff_range: 0..3,
                edf: 2.0,
                nullspace_dim: 0,
                residual_df: 50.0,
                scale: SmoothTestScale::Estimated,
                known_scale_lr_mean_shift: None,
            })
            .expect("smooth test")
        };

        let base = run(1.0);
        assert!(base.statistic > 0.0);
        // A non-trivial, clearly-significant p-value so the invariance check is
        // not vacuously comparing two values pinned at a boundary.
        assert!(base.p_value > 0.0 && base.p_value < 0.05);
        for c in [1e-3, 0.1, 10.0, 1e3, 1e6] {
            let scaled = run(c);
            let rel_stat = (scaled.statistic - base.statistic).abs() / base.statistic;
            assert!(
                rel_stat < 1e-9,
                "Wald statistic not scale-invariant at c={c}: {} vs {}",
                scaled.statistic,
                base.statistic
            );
            let rel_p = (scaled.p_value - base.p_value).abs() / base.p_value;
            assert!(
                rel_p < 1e-9,
                "estimated-scale p-value not scale-invariant at c={c}: {} vs {}",
                scaled.p_value,
                base.p_value
            );
        }
    }
}
