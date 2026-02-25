use crate::estimate::EstimationError;
use crate::estimate::FitResult;
use crate::faer_ndarray::FaerArrayView;
use crate::pirls;
use crate::types::LinkFunction;
use faer::Mat as FaerMat;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve};
use faer::{Accum, Par, Side};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use std::cmp::Ordering;

/// Approximate leave-one-out diagnostics derived from a fitted model.
#[derive(Debug, Clone)]
pub struct AloDiagnostics {
    pub eta_tilde: Array1<f64>,
    pub se: Array1<f64>,
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

fn compute_alo_diagnostics_from_pirls_impl(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    let x_dense = base.x_transformed.to_dense();
    let n = x_dense.nrows();

    let w = &base.final_weights;
    let sqrt_w = w.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrt_w_col = sqrt_w.view().insert_axis(Axis(1));
    u *= &sqrt_w_col;

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

    let ut = u.t();

    let phi = match link {
        LinkFunction::Logit | LinkFunction::Probit => 1.0,
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
    let ridge = base.ridge_used.max(0.0);
    let mut aii = Array1::<f64>::zeros(n);
    let mut se_naive = Array1::<f64>::zeros(n);
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
            .assign(&ut.slice(s![.., chunk_start..chunk_end]));

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
            let u_row = u.row(obs);
            let mut ai = 0.0f64;
            let mut s_norm2 = 0.0f64;
            for row in 0..p {
                let s_val = s_chunk[(row, local_col)];
                let u_val = u_row[row];
                ai = s_val.mul_add(u_val, ai);
                s_norm2 = s_val.mul_add(s_val, s_norm2);
            }
            let mut es_norm2 = 0.0f64;
            if e_rank > 0 {
                for r in 0..e_rank {
                    let v = es_chunk_storage[(r, local_col)];
                    es_norm2 = v.mul_add(v, es_norm2);
                }
            }
            // Using H = X'WX + S + ridge I and s = H^{-1}u:
            //   s'X'WXs = s'Hs - s'Ss - ridge*||s||^2 = aii - ||E s||^2 - ridge||s||^2
            // where S = E'E (E is e_transformed).
            let quad = ai - es_norm2 - ridge * s_norm2;
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

            let wi = base.final_weights[obs].max(1e-12);
            let var_full = phi * (quad / wi);
            if var_full == 0.0 && base.final_weights[obs] < 1e-10 {
                log::warn!(
                    "[GAM ALO] obs {} has near-zero weight ({:.2e}) resulting in SE=0",
                    obs,
                    base.final_weights[obs]
                );
            }
            let se_full = var_full.max(0.0).sqrt();
            se_naive[obs] = se_full;

            if diag_counter < max_diag_samples {
                log::debug!("[GAM ALO] SE formula (obs {}):", obs);
                log::debug!("  - w_i: {:.6e}", wi);
                log::debug!("  - a_ii: {:.6e}", ai);
                log::debug!("  - var_full: {:.6e}", var_full);
                log::debug!("  - SE_naive: {:.6e}", se_full);
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
        LinkFunction::Logit | LinkFunction::Probit => eta_tilde,
        LinkFunction::Identity => eta_tilde,
    };

    let has_nan_pred = eta_tilde.iter().any(|&x| x.is_nan());
    let has_nan_se = se_naive.iter().any(|&x| x.is_nan());
    let has_nan_leverage = aii.iter().any(|&x| x.is_nan());

    if has_nan_pred || has_nan_se || has_nan_leverage {
        log::error!("[GAM ALO] NaN values found in ALO diagnostics:");
        log::error!(
            "[GAM ALO] eta_tilde: {} NaN values",
            eta_tilde.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] se: {} NaN values",
            se_naive.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] leverage: {} NaN values",
            aii.iter().filter(|&&x| x.is_nan()).count()
        );
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }

    Ok(AloDiagnostics {
        eta_tilde,
        se: se_naive,
        pred_identity: eta_hat,
        leverage: aii,
        fisher_weights: base.final_weights.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::alo_eta_update_with_offset;

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
}

/// Compute ALO diagnostics (eta_tilde, SE, leverage) from a fitted GAM result.
pub fn compute_alo_diagnostics_from_fit(
    fit: &FitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(&fit.artifacts.pirls, y, link)
}

/// Compute ALO diagnostics from a fitted GAM result (primary API).
pub fn compute_alo_diagnostics(
    fit: &FitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_fit(fit, y, link)
}

/// Compute ALO diagnostics from a PIRLS result for lower-level callers.
pub fn compute_alo_diagnostics_from_pirls(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(base, y, link)
}
