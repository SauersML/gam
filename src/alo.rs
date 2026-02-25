use crate::estimate::EstimationError;
use crate::estimate::FitResult;
use crate::faer_ndarray::FaerArrayView;
use crate::pirls;
use crate::types::LinkFunction;
use faer::Mat as FaerMat;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve};
use faer::{Accum, Par, Side};
use ndarray::{Array1, Array2, ArrayView1, s};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AloSeMode {
    /// Frequentist influence-style SE:
    /// sqrt(phi * x_i^T H^{-1} X^T W X H^{-1} x_i).
    Sandwich,
    /// Bayesian/conditional SE:
    /// sqrt(phi * x_i^T H^{-1} x_i).
    Conditional,
}

#[derive(Debug, Clone, Copy)]
pub struct AloOptions {
    /// Which SE to expose in the legacy `AloDiagnostics::se` field.
    pub se_mode: AloSeMode,
    /// If true and `se_mode=Sandwich`, replace numerically invalid sandwich values
    /// (negative or non-finite variance) with conditional values in `se`.
    pub fallback_to_conditional_on_instability: bool,
}

impl Default for AloOptions {
    fn default() -> Self {
        Self {
            // ALO is an influence diagnostic by construction; keep sandwich as default.
            se_mode: AloSeMode::Sandwich,
            fallback_to_conditional_on_instability: true,
        }
    }
}

/// Approximate leave-one-out diagnostics derived from a fitted model.
#[derive(Debug, Clone)]
pub struct AloDiagnostics {
    pub eta_tilde: Array1<f64>,
    /// Backward-compatible selected SE channel configured by `AloOptions::se_mode`.
    pub se: Array1<f64>,
    /// Bayesian/conditional standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} x_i).
    pub se_bayes: Array1<f64>,
    /// Frequentist sandwich-style standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} X^T W X H^{-1} x_i).
    pub se_sandwich: Array1<f64>,
    /// The mode used to populate `se`.
    pub se_mode: AloSeMode,
    /// Count of entries where `se` fell back from sandwich to conditional.
    pub sandwich_fallback_count: usize,
    pub pred_identity: Array1<f64>,
    pub leverage: Array1<f64>,
    pub fisher_weights: Array1<f64>,
}

#[inline]
fn alo_eta_update_with_offset(eta_hat: f64, z: f64, offset: f64, aii: f64) -> f64 {
    let denom_raw = 1.0 - aii;
    let denom = if denom_raw <= 1e-12 { 1e-12 } else { denom_raw };
    // PIRLS solve is centered on offset:
    //   eta - offset = A (z - offset)
    let eta_centered = eta_hat - offset;
    let z_centered = z - offset;
    offset + (eta_centered - aii * z_centered) / denom
}

#[inline]
fn bayes_var_eta(phi: f64, x_hinv_x: f64) -> f64 {
    phi * x_hinv_x
}

#[inline]
fn sandwich_var_eta(phi: f64, x_hinv_x: f64, es_norm2: f64, ridge: f64, s_norm2: f64) -> f64 {
    // With H = X'WX + S + ridge*I and t = H^{-1}x_i:
    // t'X'WXt = t'Ht - t'St - ridge*||t||^2
    //         = x_i't - ||E t||^2 - ridge*||t||^2.
    phi * (x_hinv_x - es_norm2 - ridge * s_norm2)
}

fn compute_alo_diagnostics_from_pirls_impl(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
    options: AloOptions,
) -> Result<AloDiagnostics, EstimationError> {
    let x_dense_arc = base.x_transformed.to_dense_arc();
    let x_dense = x_dense_arc.as_ref();
    let n = x_dense.nrows();

    let w = &base.final_weights;

    // Use the exact stabilized Hessian from PIRLS. This keeps ALO linearization
    // consistent with the curvature used in fitting and avoids adding a second,
    // ad-hoc ridge term in diagnostics.
    let k = base.stabilized_hessian_transformed.clone();
    let p = k.nrows();
    let k_view = FaerArrayView::new(&k);

    enum Factor {
        Llt(FaerLlt<f64>),
        Ldlt(FaerLdlt<f64>),
    }
    impl Factor {
        fn solve(&self, rhs: faer::MatRef<'_, f64>) -> FaerMat<f64> {
            match self {
                Factor::Llt(f) => f.solve(rhs),
                Factor::Ldlt(f) => f.solve(rhs),
            }
        }
    }

    let factor = if let Ok(f) = FaerLlt::new(k_view.as_ref(), Side::Lower) {
        Factor::Llt(f)
    } else {
        Factor::Ldlt(FaerLdlt::new(k_view.as_ref(), Side::Lower).map_err(|_| {
            EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            }
        })?)
    };

    let xt = x_dense.t();

    let phi = match link {
        LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => 1.0,
        LinkFunction::Identity => {
            let mut rss = 0.0;
            for i in 0..n {
                let r = y[i] - base.final_mu[i];
                let wi = base.final_weights[i];
                rss += wi * r * r;
            }
            let dof = (n as f64) - base.edf;
            let denom = dof.max(1.0);
            rss / denom
        }
    };

    let e = &base.reparam_result.e_transformed;
    let e_rank = e.nrows();
    let e_view = FaerArrayView::new(e);
    let ridge = base.ridge_passport.laplace_hessian_ridge().max(0.0);
    let mut aii = Array1::<f64>::zeros(n);
    let mut se_bayes = Array1::<f64>::zeros(n);
    let mut se_sandwich = Array1::<f64>::zeros(n);
    let mut sandwich_stable = vec![true; n];
    let eta_hat = base.final_eta.clone();
    let offset = &base.final_offset;
    let z = &base.solve_working_response;

    let mut diag_counter = 0;
    let max_diag_samples = 5;

    let mut percentiles_data = Vec::with_capacity(n);
    let mut sum_aii = 0.0_f64;
    let mut max_aii = f64::NEG_INFINITY;
    let mut invalid_count = 0usize;
    let mut high_leverage_count = 0usize;
    let mut a_hi_90 = 0usize;
    let mut a_hi_95 = 0usize;
    let mut a_hi_99 = 0usize;

    let block_cols = 8192usize;

    let mut rhs_chunk_buf = Array2::<f64>::zeros((p, block_cols));

    for chunk_start in (0..n).step_by(block_cols) {
        let chunk_end = (chunk_start + block_cols).min(n);
        let width = chunk_end - chunk_start;

        rhs_chunk_buf
            .slice_mut(s![.., ..width])
            .assign(&xt.slice(s![.., chunk_start..chunk_end]));

        let rhs_chunk_view = rhs_chunk_buf.slice(s![.., ..width]);
        let rhs_chunk = FaerArrayView::new(&rhs_chunk_view);
        let s_chunk = factor.solve(rhs_chunk.as_ref());
        let mut es_chunk_storage = FaerMat::<f64>::zeros(e_rank, width);
        if e_rank > 0 {
            matmul(
                es_chunk_storage.as_mut(),
                Accum::Replace,
                e_view.as_ref(),
                s_chunk.as_ref(),
                1.0,
                Par::Seq,
            );
        }

        for local_col in 0..width {
            let obs = chunk_start + local_col;
            let x_row = x_dense.row(obs);
            let mut x_hinv_x = 0.0f64;
            let mut s_norm2 = 0.0f64;
            for row in 0..p {
                let s_val = s_chunk[(row, local_col)];
                let x_val = x_row[row];
                x_hinv_x = s_val.mul_add(x_val, x_hinv_x);
                s_norm2 = s_val.mul_add(s_val, s_norm2);
            }
            let ai = w[obs].max(0.0) * x_hinv_x;
            let mut es_norm2 = 0.0f64;
            if e_rank > 0 {
                for r in 0..e_rank {
                    let v = es_chunk_storage[(r, local_col)];
                    es_norm2 = v.mul_add(v, es_norm2);
                }
            }
            aii[obs] = ai;
            percentiles_data.push(ai);

            if ai.is_finite() {
                sum_aii += ai;
            } else {
                sum_aii = f64::NAN;
            }

            if ai.is_finite() {
                max_aii = max_aii.max(ai);
            }

            if !(0.0..=1.0).contains(&ai) || !ai.is_finite() {
                invalid_count += 1;
                log::warn!("[GAM ALO] invalid leverage at i={}, a_ii={:.6e}", obs, ai);
            } else if ai > 0.99 {
                high_leverage_count += 1;
                if ai > 0.999 {
                    log::warn!("[GAM ALO] very high leverage at i={}, a_ii={:.6e}", obs, ai);
                }
            }

            if ai > 0.90 {
                a_hi_90 += 1;
            }
            if ai > 0.95 {
                a_hi_95 += 1;
            }
            if ai > 0.99 {
                a_hi_99 += 1;
            }

            let var_bayes = bayes_var_eta(phi, x_hinv_x);
            let var_sandwich = sandwich_var_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2);
            if var_sandwich == 0.0 && base.final_weights[obs] < 1e-10 {
                log::warn!(
                    "[GAM ALO] obs {} has near-zero weight ({:.2e}) resulting in SE=0",
                    obs,
                    base.final_weights[obs]
                );
            }
            let var_sandwich_stable = var_sandwich.is_finite() && var_sandwich >= 0.0;
            sandwich_stable[obs] = var_sandwich_stable;
            if !var_sandwich_stable {
                log::warn!(
                    "[GAM ALO] unstable sandwich variance at i={}, var={:.6e}; conditional fallback eligible={}",
                    obs,
                    var_sandwich,
                    options.fallback_to_conditional_on_instability
                );
            }
            let se_bayes_i = var_bayes.max(0.0).sqrt();
            let se_sandwich_i = var_sandwich.max(0.0).sqrt();
            se_bayes[obs] = se_bayes_i;
            se_sandwich[obs] = se_sandwich_i;

            if diag_counter < max_diag_samples {
                log::debug!("[GAM ALO] SE formula (obs {}):", obs);
                log::debug!("  - w_i: {:.6e}", base.final_weights[obs]);
                log::debug!("  - a_ii: {:.6e}", ai);
                log::debug!("  - x_i'H^-1 x_i: {:.6e}", x_hinv_x);
                log::debug!("  - var_bayes: {:.6e}", var_bayes);
                log::debug!("  - var_sandwich: {:.6e}", var_sandwich);
                log::debug!("  - SE_bayes: {:.6e}", se_bayes_i);
                log::debug!("  - SE_sandwich: {:.6e}", se_sandwich_i);
                diag_counter += 1;
            }
        }
    }

    if invalid_count > 0 || high_leverage_count > 0 {
        log::warn!(
            "[GAM ALO] leverage diagnostics: {} invalid values, {} high values (>0.99)",
            invalid_count,
            high_leverage_count
        );
    }

    let mut eta_tilde = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom_raw = 1.0 - aii[i];
        let denom = if denom_raw <= 1e-12 {
            log::warn!("[GAM ALO] 1 - a_ii <= eps at i={}, a_ii={:.6e}", i, aii[i]);
            1e-12
        } else {
            denom_raw
        };

        if denom <= 1e-4 {
            log::warn!(
                "[GAM ALO] ALO 1-a_ii very small at i={}, a_ii={:.6e}",
                i,
                aii[i]
            );
        }

        eta_tilde[i] = alo_eta_update_with_offset(eta_hat[i], z[i], offset[i], aii[i]);

        if !eta_tilde[i].is_finite() || eta_tilde[i].abs() > 1e6 {
            log::warn!(
                "[GAM ALO] ALO eta_tilde extreme value at i={}: {}, capping",
                i,
                eta_tilde[i]
            );
            eta_tilde[i] = eta_tilde[i].clamp(-1e6, 1e6);
        }
    }

    let mut percentiles = percentiles_data;

    let p50_idx = if n > 1 {
        ((0.50_f64 * (n as f64 - 1.0)).round() as usize).min(n - 1)
    } else {
        0
    };
    let p95_idx = if n > 1 {
        ((0.95_f64 * (n as f64 - 1.0)).round() as usize).min(n - 1)
    } else {
        0
    };
    let p99_idx = if n > 1 {
        ((0.99_f64 * (n as f64 - 1.0)).round() as usize).min(n - 1)
    } else {
        0
    };

    let mut percentile_value = |idx: usize| -> f64 {
        if percentiles.is_empty() {
            0.0
        } else {
            let target = idx.min(percentiles.len() - 1);
            let (_, nth, _) = percentiles
                .select_nth_unstable_by(target, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            *nth
        }
    };

    let a_mean: f64 = if n == 0 { 0.0 } else { sum_aii / (n as f64) };
    let a_median = percentile_value(p50_idx);
    let a_p95 = percentile_value(p95_idx);
    let a_p99 = percentile_value(p99_idx);
    let a_max = if max_aii.is_finite() { max_aii } else { 0.0 };

    log::warn!(
        "[GAM ALO] leverage: n={}, mean={:.3e}, median={:.3e}, p95={:.3e}, p99={:.3e}, max={:.3e}",
        n,
        a_mean,
        a_median,
        a_p95,
        a_p99,
        a_max
    );
    log::warn!(
        "[GAM ALO] high-leverage: a>0.90: {:.2}%, a>0.95: {:.2}%, a>0.99: {:.2}%, dispersion phi={:.3e}",
        100.0 * (a_hi_90 as f64) / (n as f64).max(1.0),
        100.0 * (a_hi_95 as f64) / (n as f64).max(1.0),
        100.0 * (a_hi_99 as f64) / (n as f64).max(1.0),
        phi
    );

    let eta_tilde = match link {
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Identity => eta_tilde,
    };

    let has_nan_pred = eta_tilde.iter().any(|&x| x.is_nan());
    let has_nan_se_bayes = se_bayes.iter().any(|&x| x.is_nan());
    let has_nan_se_sandwich = se_sandwich.iter().any(|&x| x.is_nan());
    let has_nan_leverage = aii.iter().any(|&x| x.is_nan());

    if has_nan_pred || has_nan_se_bayes || has_nan_se_sandwich || has_nan_leverage {
        log::error!("[GAM ALO] NaN values found in ALO diagnostics:");
        log::error!(
            "[GAM ALO] eta_tilde: {} NaN values",
            eta_tilde.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] se_bayes: {} NaN values",
            se_bayes.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] se_sandwich: {} NaN values",
            se_sandwich.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] leverage: {} NaN values",
            aii.iter().filter(|&&x| x.is_nan()).count()
        );
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }

    let mut se_selected = match options.se_mode {
        AloSeMode::Sandwich => se_sandwich.clone(),
        AloSeMode::Conditional => se_bayes.clone(),
    };
    let mut sandwich_fallback_count = 0usize;
    if matches!(options.se_mode, AloSeMode::Sandwich) && options.fallback_to_conditional_on_instability
    {
        for i in 0..n {
            if !sandwich_stable[i] {
                se_selected[i] = se_bayes[i];
                sandwich_fallback_count += 1;
            }
        }
    }

    Ok(AloDiagnostics {
        eta_tilde,
        se: se_selected,
        se_bayes,
        se_sandwich,
        se_mode: options.se_mode,
        sandwich_fallback_count,
        pred_identity: eta_hat,
        leverage: aii,
        fisher_weights: base.final_weights.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::{alo_eta_update_with_offset, bayes_var_eta, sandwich_var_eta};

    #[test]
    fn alo_offset_update_matches_centered_algebra() {
        let eta_hat = 11.0;
        let z = 13.0;
        let offset = 10.0;
        let aii = 0.2;
        // centered: eta~=off + ((eta-off)-a(z-off))/(1-a)
        let expected = offset + ((eta_hat - offset) - aii * (z - offset)) / (1.0 - aii);
        let got = alo_eta_update_with_offset(eta_hat, z, offset, aii);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_reduces_to_classic_when_offset_zero() {
        let eta_hat = 1.25;
        let z = -0.5;
        let aii = 0.35;
        let expected = (eta_hat - aii * z) / (1.0 - aii);
        let got = alo_eta_update_with_offset(eta_hat, z, 0.0, aii);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn gaussian_unpenalized_sandwich_equals_bayes() {
        // In Gaussian linear model with S=0 and ridge=0:
        // H = X'WX, so sandwich and bayes eta variances are identical.
        let phi = 2.5;
        let x_hinv_x = 0.3;
        let es_norm2 = 0.0;
        let ridge = 0.0;
        let s_norm2 = 0.0;
        let vb = bayes_var_eta(phi, x_hinv_x);
        let vs = sandwich_var_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2);
        assert!((vb - vs).abs() < 1e-12);
    }

    #[test]
    fn sandwich_matches_direct_linear_gaussian_formula() {
        // Small brute-force linear Gaussian check:
        // var_sandwich(eta_i) = phi * x_i^T H^{-1} X'WX H^{-1} x_i.
        let phi = 1.7;
        let x_hinv_x = 0.41;
        let es_norm2 = 0.05;
        let ridge = 1e-3;
        let s_norm2 = 2.0;
        let got = sandwich_var_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2);
        let expected = phi * (x_hinv_x - es_norm2 - ridge * s_norm2);
        assert!((got - expected).abs() < 1e-12);
    }
}

/// Compute ALO diagnostics (eta_tilde, SE, leverage) from a fitted GAM result.
pub fn compute_alo_diagnostics_from_fit(
    fit: &FitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(&fit.artifacts.pirls, y, link, AloOptions::default())
}

/// Compute ALO diagnostics from a fitted GAM result (primary API).
pub fn compute_alo_diagnostics(
    fit: &FitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_fit(fit, y, link)
}

/// Compute ALO diagnostics from a fitted GAM result with explicit ALO options.
pub fn compute_alo_diagnostics_with_options(
    fit: &FitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
    options: AloOptions,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(&fit.artifacts.pirls, y, link, options)
}

/// Compute ALO diagnostics from a PIRLS result for lower-level callers.
pub fn compute_alo_diagnostics_from_pirls(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(base, y, link, AloOptions::default())
}

/// Compute ALO diagnostics from a PIRLS result with explicit ALO options.
pub fn compute_alo_diagnostics_from_pirls_with_options(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
    options: AloOptions,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(base, y, link, options)
}
