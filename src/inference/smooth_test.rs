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
/// out). `influence_matrix` is the optional coefficient-space influence
/// `F = H⁻¹ X'WX`; when present `tr(F_jj)² / tr(F_jj²)` is used as the
/// Wood-corrected reference d.f. `edf` is the smooth's effective d.f.
/// (rank truncation for the penalized subblock); `nullspace_dim` is the
/// fixed-effect (unpenalized) leading dimension within the block.
/// `dispersion` is the residual variance estimate `φ̂`; `residual_df` is
/// the matching denominator d.f. for the F branch.
#[derive(Debug, Clone)]
pub struct SmoothTestInput<'a> {
    pub beta: ArrayView1<'a, f64>,
    pub covariance: &'a Array2<f64>,
    pub influence_matrix: Option<&'a Array2<f64>>,
    pub coeff_range: Range<usize>,
    pub edf: f64,
    pub nullspace_dim: usize,
    pub dispersion: f64,
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

pub fn wood_smooth_test(input: SmoothTestInput<'_>) -> Option<SmoothTestResult> {
    let start = input.coeff_range.start;
    let end = input.coeff_range.end;
    if start >= end
        || end > input.beta.len()
        || end > input.covariance.nrows()
        || end > input.covariance.ncols()
        || !input.edf.is_finite()
        || input.edf <= 0.0
        || !input.dispersion.is_finite()
        || input.dispersion <= 0.0
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
    let p_value = match input.scale {
        SmoothTestScale::Known => {
            let dist = ChiSquared::new(ref_df).ok()?;
            1.0 - dist.cdf(statistic)
        }
        SmoothTestScale::Estimated => {
            if !input.residual_df.is_finite() || input.residual_df <= 0.0 {
                return None;
            }
            let f_stat = statistic / (ref_df * input.dispersion);
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
            dispersion: 1.0,
            residual_df: 20.0,
            scale: SmoothTestScale::Known,
        })
        .expect("smooth test");
        assert!((out.ref_df - 1.8).abs() < 1e-12);
        assert!(out.statistic > 0.0);
        assert!((0.0..=1.0).contains(&out.p_value));
    }
}
