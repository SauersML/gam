//! Wood-style smooth-term Wald tests.
//!
//! The entry point here implements the covariance-rank truncation and reference
//! degrees-of-freedom pieces that should not be inlined in summary rendering.

use crate::faer_ndarray::FaerEigh;
use crate::linalg::utils::enforce_symmetry;
use crate::solver::estimate::Dispersion;
use ndarray::{Array2, ArrayView1, s};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothTestCovariance {
    /// Use smoothing-parameter-corrected Bayesian covariance (`Vp`).
    Corrected,
    /// Use conditional Bayesian covariance (`Vb`).
    Conditional,
}

#[derive(Clone, Debug)]
pub struct SmoothTestInput<'a> {
    pub beta: ArrayView1<'a, f64>,
    pub covariance: &'a Array2<f64>,
    pub coefficient_influence: Option<&'a Array2<f64>>,
    pub coeff_range: Range<usize>,
    pub edf: f64,
    pub nullspace_dim: usize,
    pub dispersion: Dispersion,
    pub residual_df: Option<f64>,
    pub constrained: bool,
}

#[derive(Clone, Debug)]
pub struct SmoothTestResult {
    pub statistic: f64,
    pub ref_df: f64,
    pub p_value: Option<f64>,
    pub valid: bool,
    pub reason: Option<String>,
}

fn covariance_block(cov: &Array2<f64>, range: Range<usize>) -> Option<Array2<f64>> {
    if range.start >= range.end || range.end > cov.nrows() || range.end > cov.ncols() {
        return None;
    }
    Some(cov.slice(s![range.clone(), range]).to_owned())
}

fn influence_ref_df(fmat: &Array2<f64>, range: Range<usize>, fallback: f64) -> f64 {
    if range.start >= range.end || range.end > fmat.nrows() || range.end > fmat.ncols() {
        return fallback.max(1e-12);
    }
    let block = fmat.slice(s![range.clone(), range]).to_owned();
    let tr = block.diag().iter().copied().sum::<f64>();
    let tr2 = block.dot(&block).diag().iter().copied().sum::<f64>();
    if tr.is_finite() && tr > 0.0 && tr2.is_finite() && tr2 > 0.0 {
        (tr * tr / tr2).max(1e-12)
    } else {
        fallback.max(1e-12)
    }
}

fn truncated_wald(beta: ArrayView1<'_, f64>, cov: &Array2<f64>, rank: usize) -> Option<f64> {
    let k = beta.len();
    if k == 0 || cov.nrows() != k || cov.ncols() != k || rank == 0 {
        return Some(0.0);
    }
    let mut sym = cov.clone();
    enforce_symmetry(&mut sym);
    let (evals, evecs) = sym.eigh(faer::Side::Lower).ok()?;
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut q = 0.0;
    let keep = rank.min(k);
    for &idx in order.iter().take(keep) {
        let lambda = evals[idx];
        if !(lambda.is_finite() && lambda > 0.0) {
            continue;
        }
        let v = evecs.column(idx);
        let proj = beta.dot(&v);
        q += proj * proj / lambda;
    }
    q.is_finite().then_some(q.max(0.0))
}

pub fn wood_smooth_test(input: SmoothTestInput<'_>) -> SmoothTestResult {
    if input.constrained {
        return SmoothTestResult {
            statistic: f64::NAN,
            ref_df: input.edf.max(1e-12),
            p_value: None,
            valid: false,
            reason: Some("shape-constrained smooth p-values are disabled because the null distribution is constrained".to_string()),
        };
    }
    let Some(cov_block) = covariance_block(input.covariance, input.coeff_range.clone()) else {
        return SmoothTestResult {
            statistic: f64::NAN,
            ref_df: input.edf.max(1e-12),
            p_value: None,
            valid: false,
            reason: Some(
                "covariance block is unavailable or has incompatible dimensions".to_string(),
            ),
        };
    };
    let k = input.beta.len();
    let ns = input.nullspace_dim.min(k);
    let penalized_k = k.saturating_sub(ns);
    let penalized_rank = input.edf.round().max(0.0) as usize;
    let penalized_rank = penalized_rank.saturating_sub(ns).min(penalized_k);

    let mut stat = 0.0;
    if ns > 0 {
        let beta_ns = input.beta.slice(s![0..ns]);
        let cov_ns = cov_block.slice(s![0..ns, 0..ns]).to_owned();
        stat += truncated_wald(beta_ns, &cov_ns, ns).unwrap_or(f64::NAN);
    }
    if penalized_k > 0 && penalized_rank > 0 {
        let beta_pen = input.beta.slice(s![ns..k]);
        let cov_pen = cov_block.slice(s![ns..k, ns..k]).to_owned();
        stat += truncated_wald(beta_pen, &cov_pen, penalized_rank).unwrap_or(f64::NAN);
    }

    let fallback_df = input.edf.round().clamp(1.0, k.max(1) as f64);
    let ref_df = input
        .coefficient_influence
        .map(|f| influence_ref_df(f, input.coeff_range.clone(), fallback_df))
        .unwrap_or(fallback_df);
    let p_value = if stat.is_finite() {
        if input.dispersion.is_estimated() {
            input.residual_df.and_then(|den_df| {
                (den_df.is_finite() && den_df > 0.0)
                    .then(|| FisherSnedecor::new(ref_df, den_df).ok())
                    .flatten()
                    .map(|dist| 1.0 - dist.cdf(stat / ref_df / input.dispersion.phi()))
            })
        } else {
            ChiSquared::new(ref_df)
                .ok()
                .map(|dist| 1.0 - dist.cdf(stat))
        }
    } else {
        None
    };

    SmoothTestResult {
        statistic: stat,
        ref_df,
        p_value,
        valid: stat.is_finite(),
        reason: (!stat.is_finite()).then(|| "smooth test statistic was non-finite".to_string()),
    }
}
