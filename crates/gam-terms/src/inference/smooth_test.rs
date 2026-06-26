//! Wood-style smooth-component Wald tests.
//!
//! The test follows the rank-truncated covariance inverse used by Wood (2013):
//! the penalized part of the coefficient block is tested with a pseudo-inverse
//! of rank approximately equal to the term EDF, while unpenalized null-space
//! coefficients are kept full-rank. The reference degrees of freedom use the
//! coefficient-space influence block `F_jj = (H⁻¹ X'WX)_jj`.
//!
//! Bartlett and Lawley mean corrections are likelihood-ratio corrections, so
//! they are not applied here. In the ordinary unpenalized Gaussian model the
//! Wald statistic satisfies `T / q ~ F(q, ν)` exactly, while under a ridge
//! penalty even the one-parameter statistic becomes `(n / (n + λ))χ²₁` rather
//! than a central χ²/F reference target.

use gam_linalg::faer_ndarray::FaerEigh;
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
    // Effective rank of the statistic: the number of covariance directions
    // actually summed across the null and penalized sub-blocks. The χ²/F
    // reference d.f. is floored at this count so a boundary-shrunk term (whose
    // Wood influence-trace d.f. collapses toward 0) is never judged against a
    // degenerate ~0-d.f. reference — the mechanism that turned a *zero* Wald
    // statistic into p≈0 for a term the fit removed (#1360).
    let mut rank_used = 0usize;
    if null_dim > 0 {
        let beta_null = beta.slice(s![0..null_dim]).to_owned();
        let cov_null = cov.slice(s![0..null_dim, 0..null_dim]).to_owned();
        let (q, used) = full_rank_quadratic(&beta_null, &cov_null)?;
        statistic += q;
        rank_used += used;
    }
    if pen_dim > 0 {
        let beta_pen = beta.slice(s![null_dim..k]).to_owned();
        let cov_pen = cov.slice(s![null_dim..k, null_dim..k]).to_owned();
        let rank = truncated_rank(input.edf - null_dim as f64, pen_dim);
        if rank > 0 {
            let (q, used) = truncated_quadratic(&beta_pen, &cov_pen, rank)?;
            statistic += q;
            rank_used += used;
        }
    }

    if rank_used == 0 {
        // No estimable direction in the block (every covariance eigenmode is
        // numerically null): the term carries no testable signal.
        return None;
    }
    // Wood (2013) influence-trace participation d.f. when available, but never
    // below `rank_used`. The historical fallback to `edf` collapsed to ~0 for a
    // shrunk term, making `χ²_{ref_df→0}` degenerate.
    let ref_df = match reference_df(input.influence_matrix, start, end) {
        Some(rd) if rd.is_finite() && rd > 0.0 => rd.max(rank_used as f64),
        _ => rank_used as f64,
    };
    if !statistic.is_finite() || statistic < 0.0 || !ref_df.is_finite() || ref_df <= 0.0 {
        return None;
    }
    let p_value = match input.scale {
        SmoothTestScale::Known => {
            let dist = ChiSquared::new(ref_df).ok()?;
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

fn full_rank_quadratic(beta: &Array1<f64>, cov: &Array2<f64>) -> Option<(f64, usize)> {
    truncated_quadratic(beta, cov, beta.len())
}

/// Returns the rank-`rank` truncated Wald quadratic together with the number of
/// covariance directions (eigenmodes above the relative tolerance) that were
/// actually summed into it. The `used` count is the *effective rank of the
/// statistic*: it can fall below `rank` when the covariance subblock is itself
/// rank-deficient. Callers fold it into the χ² reference degrees of freedom so
/// the tail probability is never evaluated against a degenerate ~0 d.f.
fn truncated_quadratic(beta: &Array1<f64>, cov: &Array2<f64>, rank: usize) -> Option<(f64, usize)> {
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
    (used > 0 && q.is_finite()).then_some((q.max(0.0), used))
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
    use statrs::distribution::{ChiSquared, ContinuousCDF};

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
        })
        .expect("smooth test");
        assert!((out.ref_df - 1.8).abs() < 1e-12);
        assert!(out.statistic > 0.0);
        assert!((0.0..=1.0).contains(&out.p_value));
    }

    #[test]
    fn known_scale_branch_reports_plain_wald_chi_square() {
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
        })
        .expect("smooth test");

        let dist = ChiSquared::new(out.ref_df).expect("chi-square");
        let expected = 1.0 - dist.cdf(out.statistic);
        assert!((out.p_value - expected).abs() < 1e-15);
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

    /// A term the fit drove to the penalty boundary (coefficients ≈ 0, EDF → 0)
    /// must read as *not* significant. The defect (#1360): the reference d.f.
    /// fell back to `edf` and collapsed toward 0, so the χ² tail of a *zero*
    /// statistic evaluated at ~0 d.f. degenerated to p ≈ 0 — an overwhelming
    /// false positive for a term that was removed. The reference d.f. is now
    /// floored at the rank actually summed (≥ 1), so a zero statistic returns
    /// p ≈ 1.
    #[test]
    fn boundary_shrunk_term_is_not_significant() {
        // Near-zero coefficients with a well-conditioned (non-degenerate)
        // covariance: the Wald statistic is ~0 regardless of how the reference
        // d.f. is formed.
        let beta = array![1e-9, -2e-9, 5e-10];
        let cov = array![[0.04, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.06]];
        // A degenerate influence block (sign-flipped near-zero leverages) so the
        // Wood trace correction is unavailable and the fallback is exercised.
        let f = array![[1e-9, 0.0, 0.0], [0.0, -1e-9, 0.0], [0.0, 0.0, 1e-12]];
        for scale in [SmoothTestScale::Known, SmoothTestScale::Estimated] {
            let out = wood_smooth_test(SmoothTestInput {
                beta: beta.view(),
                covariance: &cov,
                influence_matrix: Some(&f),
                coeff_range: 0..3,
                edf: 1e-6,
                nullspace_dim: 0,
                residual_df: 500.0,
                scale,
            })
            .expect("boundary term still produces a result");
            assert!(
                out.ref_df >= 1.0,
                "reference d.f. must not collapse below the tested rank: {}",
                out.ref_df
            );
            assert!(
                out.statistic < 1e-6,
                "boundary statistic should be ~0: {}",
                out.statistic
            );
            assert!(
                out.p_value > 0.5,
                "shrunk boundary term must not be significant (p={}, scale={:?})",
                out.p_value,
                scale
            );
        }
    }

    /// Flooring the reference d.f. at the tested rank must not weaken a genuinely
    /// significant term: a large statistic with a healthy influence block keeps
    /// its small p-value (the floor only raises a *degenerate* sub-1 d.f.).
    #[test]
    fn floor_does_not_blunt_a_real_signal() {
        let beta = array![6.0, -5.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let f = array![[0.9, 0.0], [0.0, 0.9]];
        let out = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: Some(&f),
            coeff_range: 0..2,
            edf: 2.0,
            nullspace_dim: 2,
            residual_df: 500.0,
            scale: SmoothTestScale::Known,
        })
        .expect("smooth test");
        assert!(out.statistic > 40.0, "statistic={}", out.statistic);
        assert!(
            out.p_value < 1e-6,
            "a strong term must stay significant: p={}",
            out.p_value
        );
    }
}
