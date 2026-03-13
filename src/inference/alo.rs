use crate::estimate::EstimationError;
use crate::estimate::{FitGeometry, FitResult, UnifiedFitResult};
use crate::faer_ndarray::FaerArrayView;
use crate::linalg::utils::StableSolver;
use crate::pirls;
use crate::types::LinkFunction;
use faer::Mat as FaerMat;
use faer::linalg::matmul::matmul;
use faer::{Accum, Par};
use ndarray::{Array1, Array2, ArrayView1, s};
use std::cmp::Ordering;

/// Approximate leave-one-out diagnostics derived from a fitted model.
#[derive(Debug, Clone)]
pub struct AloDiagnostics {
    pub eta_tilde: Array1<f64>,
    /// Bayesian/conditional standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} x_i).
    pub se_bayes: Array1<f64>,
    /// Frequentist sandwich-style standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} X^T W X H^{-1} x_i).
    pub se_sandwich: Array1<f64>,
    pub pred_identity: Array1<f64>,
    pub leverage: Array1<f64>,
    pub fisherweights: Array1<f64>,
}

#[inline]
fn alo_eta_updatewith_offset(eta_hat: f64, z: f64, offset: f64, aii: f64) -> f64 {
    let denom = 1.0 - aii;
    // PIRLS solve is centered on offset:
    //   eta - offset = A (z - offset)
    let eta_centered = eta_hat - offset;
    let z_centered = z - offset;
    offset + (eta_centered - aii * z_centered) / denom
}

#[inline]
fn bayesvar_eta(phi: f64, x_hinv_x: f64) -> f64 {
    phi * x_hinv_x
}

#[inline]
fn sandwichvar_eta(phi: f64, x_hinv_x: f64, es_norm2: f64, ridge: f64, s_norm2: f64) -> f64 {
    // With H = X'WX + S + ridge*I and t = H^{-1}x_i:
    // t'X'WXt = t'Ht - t'St - ridge*||t||^2
    //         = x_i't - ||E t||^2 - ridge*||t||^2.
    phi * (x_hinv_x - es_norm2 - ridge * s_norm2)
}

#[inline]
fn variance_negative_tolerance(scale: f64) -> f64 {
    // Tight relative tolerance for cancellation from x'H^{-1}x - ||E t||^2 - ridge||t||^2.
    1e-12 * scale.abs().max(1.0)
}

fn compute_alo_diagnostics_from_pirls_impl(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    let x_dense_arc = base
        .x_transformed
        .try_to_dense_arc("ALO diagnostics require dense transformed design")
        .map_err(EstimationError::InvalidInput)?;
    let x_dense = x_dense_arc.as_ref();
    let n = x_dense.nrows();

    // Compute dispersion parameter.
    let phi = match link {
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => 1.0,
        LinkFunction::Identity => {
            let mut rss = 0.0;
            for i in 0..n {
                let r = y[i] - base.finalmu[i];
                let wi = base.finalweights[i];
                rss += wi * r * r;
            }
            let dof = (n as f64) - base.edf;
            let denom = dof.max(1.0);
            rss / denom
        }
    };

    let e = &base.reparam_result.e_transformed;
    let ridge = base.ridge_passport.laplacehessianridge().max(0.0);

    // Build model-agnostic AloInput from PIRLS geometry, then delegate.
    let input = AloInput {
        design: x_dense,
        penalized_hessian: &base.stabilizedhessian_transformed,
        working_weights: &base.finalweights,
        working_response: &base.solveworking_response,
        eta: &base.final_eta,
        offset: &base.final_offset,
        link,
        phi,
        penalty_null_space: if e.nrows() > 0 { Some(e) } else { None },
        ridge,
    };

    let result = compute_alo_from_input(&input)?;

    // PIRLS-specific post-hoc leverage diagnostics logging.
    log_leverage_diagnostics(&result.leverage, phi);

    // Final NaN guard with detailed error reporting.
    let has_nan_pred = result.eta_tilde.iter().any(|&x| x.is_nan());
    let has_nan_se_bayes = result.se_bayes.iter().any(|&x| x.is_nan());
    let has_nan_se_sandwich = result.se_sandwich.iter().any(|&x| x.is_nan());
    let has_nan_leverage = result.leverage.iter().any(|&x| x.is_nan());

    if has_nan_pred || has_nan_se_bayes || has_nan_se_sandwich || has_nan_leverage {
        log::error!("[GAM ALO] NaN values found in ALO diagnostics:");
        log::error!(
            "[GAM ALO] eta_tilde: {} NaN values",
            result.eta_tilde.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] se_bayes: {} NaN values",
            result.se_bayes.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] se_sandwich: {} NaN values",
            result.se_sandwich.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] leverage: {} NaN values",
            result.leverage.iter().filter(|&&x| x.is_nan()).count()
        );
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }

    Ok(result)
}

/// Log detailed leverage percentile diagnostics for a completed ALO computation.
fn log_leverage_diagnostics(leverage: &Array1<f64>, phi: f64) {
    let n = leverage.len();
    if n == 0 {
        return;
    }

    let mut sum_aii = 0.0_f64;
    let mut max_aii = f64::NEG_INFINITY;
    let mut invalid_count = 0usize;
    let mut high_leverage_count = 0usize;
    let mut a_hi_90 = 0usize;
    let mut a_hi_95 = 0usize;
    let mut a_hi_99 = 0usize;

    for (obs, &ai) in leverage.iter().enumerate() {
        if ai.is_finite() {
            sum_aii += ai;
            max_aii = max_aii.max(ai);
        } else {
            sum_aii = f64::NAN;
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
    }

    if invalid_count > 0 || high_leverage_count > 0 {
        log::warn!(
            "[GAM ALO] leverage diagnostics: {} invalid values, {} high values (>0.99)",
            invalid_count,
            high_leverage_count
        );
    }

    let mut percentiles_data: Vec<f64> = leverage.to_vec();

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

    let mut percentilevalue = |idx: usize| -> f64 {
        if percentiles_data.is_empty() {
            0.0
        } else {
            let target = idx.min(percentiles_data.len() - 1);
            let (_, nth, _) = percentiles_data
                .select_nth_unstable_by(target, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            *nth
        }
    };

    let a_mean: f64 = sum_aii / (n as f64);
    let a_median = percentilevalue(p50_idx);
    let a_p95 = percentilevalue(p95_idx);
    let a_p99 = percentilevalue(p99_idx);
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
}

/// Model-agnostic input for ALO diagnostics.
///
/// Any model with a design matrix, penalized Hessian, and IRLS geometry can
/// compute ALO leverages and leave-one-out predictions. This decouples ALO
/// from the single-block PIRLS solver and enables diagnostics for GAMLSS,
/// survival, and joint models.
pub struct AloInput<'a> {
    /// Dense design matrix X (n × p).
    pub design: &'a Array2<f64>,
    /// Penalized Hessian H = X'WX + S(λ) at convergence (p × p).
    pub penalized_hessian: &'a Array2<f64>,
    /// IRLS working weights at convergence (n).
    pub working_weights: &'a Array1<f64>,
    /// IRLS working response at convergence (n).
    pub working_response: &'a Array1<f64>,
    /// Fitted linear predictor η̂ (n).
    pub eta: &'a Array1<f64>,
    /// Offset vector (n). Pass zeros if no offset.
    pub offset: &'a Array1<f64>,
    /// Link function (for phi determination).
    pub link: LinkFunction,
    /// Dispersion parameter φ. For non-Gaussian families this is 1.0.
    pub phi: f64,
    /// Optional null-space projector E (rank × p) for sandwich SE.
    /// When `None`, sandwich SE is set equal to Bayesian SE.
    pub penalty_null_space: Option<&'a Array2<f64>>,
    /// Ridge added to the Hessian for logdet surface.
    pub ridge: f64,
}

impl<'a> AloInput<'a> {
    /// Build an `AloInput` from `FitGeometry` and associated vectors.
    pub fn from_geometry(
        geom: &'a FitGeometry,
        design: &'a Array2<f64>,
        eta: &'a Array1<f64>,
        offset: &'a Array1<f64>,
        link: LinkFunction,
        phi: f64,
    ) -> Self {
        Self {
            design,
            penalized_hessian: &geom.penalized_hessian,
            working_weights: &geom.working_weights,
            working_response: &geom.working_response,
            eta,
            offset,
            link,
            phi,
            penalty_null_space: None,
            ridge: 0.0,
        }
    }
}

/// Compute ALO diagnostics from model-agnostic inputs.
///
/// This is the generalized entry point that works for any model type.
/// For standard single-block GAMs, prefer `compute_alo_diagnostics_from_fit`
/// which automatically extracts the PIRLS geometry (including sandwich SE).
pub fn compute_alo_from_input(input: &AloInput) -> Result<AloDiagnostics, EstimationError> {
    let x_dense = input.design;
    let n = x_dense.nrows();
    let p = x_dense.ncols();
    let w = input.working_weights;

    let factor = StableSolver::new("alo penalized hessian")
        .factorize(input.penalized_hessian)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;

    let xt = x_dense.t();
    let phi = input.phi;
    let ridge = input.ridge;

    let e_rank = input.penalty_null_space.map(|e| e.nrows()).unwrap_or(0);

    let mut aii = Array1::<f64>::zeros(n);
    let mut se_bayes = Array1::<f64>::zeros(n);
    let mut se_sandwich = Array1::<f64>::zeros(n);

    let block_cols = 8192usize;
    let mut rhs_chunk_buf = Array2::<f64>::zeros((p, block_cols));

    for chunk_start in (0..n).step_by(block_cols) {
        let chunk_end = (chunk_start + block_cols).min(n);
        let width = chunk_end - chunk_start;

        rhs_chunk_buf
            .slice_mut(s![.., ..width])
            .assign(&xt.slice(s![.., chunk_start..chunk_end]));

        let rhs_chunkview = rhs_chunk_buf.slice(s![.., ..width]);
        let rhs_chunk = FaerArrayView::new(&rhs_chunkview);
        let s_chunk = factor.solve(rhs_chunk.as_ref());

        let mut es_chunk_storage = FaerMat::<f64>::zeros(e_rank, width);
        if e_rank > 0 {
            if let Some(e) = input.penalty_null_space {
                let eview = FaerArrayView::new(e);
                matmul(
                    es_chunk_storage.as_mut(),
                    Accum::Replace,
                    eview.as_ref(),
                    s_chunk.as_ref(),
                    1.0,
                    Par::Seq,
                );
            }
        }

        for local_col in 0..width {
            let obs = chunk_start + local_col;
            let xrow = x_dense.row(obs);
            let mut x_hinv_x = 0.0f64;
            let mut s_norm2 = 0.0f64;
            for row in 0..p {
                let sval = s_chunk[(row, local_col)];
                let xval = xrow[row];
                x_hinv_x = sval.mul_add(xval, x_hinv_x);
                s_norm2 = sval.mul_add(sval, s_norm2);
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

            let var_bayes = bayesvar_eta(phi, x_hinv_x);
            let var_sandwich = if e_rank > 0 {
                sandwichvar_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2)
            } else {
                var_bayes
            };

            if !var_bayes.is_finite() || !var_sandwich.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "ALO variance is not finite at row {obs}: bayes={var_bayes:.6e}, sandwich={var_sandwich:.6e}"
                )));
            }
            let bayes_tol = variance_negative_tolerance(phi * x_hinv_x.abs());
            if var_bayes < -bayes_tol {
                return Err(EstimationError::InvalidInput(format!(
                    "ALO Bayesian variance is materially negative at row {obs}: var={var_bayes:.6e}, tol={bayes_tol:.6e}"
                )));
            }
            if e_rank > 0 {
                let sandwich_scale =
                    phi * (x_hinv_x.abs() + es_norm2.abs() + (ridge * s_norm2).abs());
                let sandwich_tol = variance_negative_tolerance(sandwich_scale);
                if var_sandwich < -sandwich_tol {
                    return Err(EstimationError::InvalidInput(format!(
                        "ALO sandwich variance is materially negative at row {obs}: var={var_sandwich:.6e}, tol={sandwich_tol:.6e}"
                    )));
                }
            }

            se_bayes[obs] = var_bayes.max(0.0).sqrt();
            se_sandwich[obs] = var_sandwich.max(0.0).sqrt();
        }
    }

    let eta_hat = input.eta;
    let z = input.working_response;
    let offset = input.offset;

    let mut eta_tilde = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom_raw = 1.0 - aii[i];
        if denom_raw <= 0.0 || !denom_raw.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "ALO denominator is non-positive at row {i}: a_ii={:.6e}, 1-a_ii={:.6e}",
                aii[i], denom_raw
            )));
        }
        eta_tilde[i] = alo_eta_updatewith_offset(eta_hat[i], z[i], offset[i], aii[i]);
        if !eta_tilde[i].is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "ALO eta_tilde is not finite at row {i}: eta_tilde={}",
                eta_tilde[i]
            )));
        }
    }

    Ok(AloDiagnostics {
        eta_tilde,
        se_bayes,
        se_sandwich,
        pred_identity: eta_hat.clone(),
        leverage: aii,
        fisherweights: w.clone(),
    })
}

/// Compute ALO diagnostics (eta_tilde, SE, leverage) from a fitted GAM result.
pub fn compute_alo_diagnostics_from_fit(
    fit: &FitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    let pirls = fit.artifacts.pirls.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "ALO diagnostics require a PIRLS-backed fit; this fit does not expose PIRLS geometry"
                .to_string(),
        )
    })?;
    compute_alo_diagnostics_from_pirls_impl(pirls, y, link)
}

/// Compute ALO diagnostics from a fitted GAM result (primary API).
pub fn compute_alo_diagnostics(
    fit: &FitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_fit(fit, y, link)
}

/// Compute ALO diagnostics from a `UnifiedFitResult`.
///
/// Extracts `FitGeometry` from `unified.geometry`, builds an `AloInput`
/// via `from_geometry`, and delegates to `compute_alo_from_input`.
/// This avoids requiring a full `FitResult` with PIRLS artifacts.
pub fn compute_alo_diagnostics_from_unified(
    unified: &UnifiedFitResult,
    design: &Array2<f64>,
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    link: LinkFunction,
    phi: f64,
) -> Result<AloDiagnostics, EstimationError> {
    let geom = unified.geometry.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "UnifiedFitResult does not contain working-set geometry; \
             ALO diagnostics require geometry at convergence"
                .to_string(),
        )
    })?;
    let input = AloInput::from_geometry(geom, design, eta, offset, link, phi);
    compute_alo_from_input(&input)
}

/// Compute ALO diagnostics from a PIRLS result for lower-level callers.
pub fn compute_alo_diagnostics_from_pirls(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(base, y, link)
}

#[cfg(test)]
mod tests {
    use super::{alo_eta_updatewith_offset, bayesvar_eta, sandwichvar_eta};

    #[test]
    fn alo_offset_update_matches_centered_algebra() {
        let eta_hat = 11.0;
        let z = 13.0;
        let offset = 10.0;
        let aii = 0.2;
        // centered: eta~=off + ((eta-off)-a(z-off))/(1-a)
        let expected = offset + ((eta_hat - offset) - aii * (z - offset)) / (1.0 - aii);
        let got = alo_eta_updatewith_offset(eta_hat, z, offset, aii);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_reduces_to_classicwhen_offsetzero() {
        let eta_hat = 1.25;
        let z = -0.5;
        let aii = 0.35;
        let expected = (eta_hat - aii * z) / (1.0 - aii);
        let got = alo_eta_updatewith_offset(eta_hat, z, 0.0, aii);
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
        let vb = bayesvar_eta(phi, x_hinv_x);
        let vs = sandwichvar_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2);
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
        let got = sandwichvar_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2);
        let expected = phi * (x_hinv_x - es_norm2 - ridge * s_norm2);
        assert!((got - expected).abs() < 1e-12);
    }
}
