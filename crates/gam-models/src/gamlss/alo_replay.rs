use super::*;

/// Exact row-local saved-model geometry in affine likelihood coordinates.
///
/// Coordinates are `[primary predictor, log-scale predictor, wiggle
/// coefficients...]`. The first two entries use their fitted design rows;
/// every optional wiggle entry is the scalar coefficient itself and therefore
/// has a one-column constant design. This augmented coordinate system retains
/// every second derivative of `B(q0) beta_w` while remaining affine in the
/// saved coefficient vector.
#[derive(Clone, Debug, PartialEq)]
pub struct LocationScaleAloRowGeometry {
    pub nll_score: Array1<f64>,
    pub observed_hessian: Array2<f64>,
}

/// Complete saved-row state for exact Gaussian location-scale ALO replay.
pub struct GaussianLocationScaleAloRowInput<'a> {
    pub row: usize,
    pub y: f64,
    pub base_mean: f64,
    pub eta_log_sigma: f64,
    pub prior_weight: f64,
    pub response_scale: f64,
    pub wiggle_basis: &'a [f64],
    pub wiggle_basis_d1: &'a [f64],
    pub wiggle_basis_d2: &'a [f64],
    pub wiggle_beta: &'a [f64],
}

/// Complete saved-row state for exact binomial location-scale ALO replay.
pub struct BinomialLocationScaleAloRowInput<'a> {
    pub y: f64,
    pub threshold_eta: f64,
    pub eta_log_sigma: f64,
    pub prior_weight: f64,
    pub inverse_link: &'a InverseLink,
    pub wiggle_basis: &'a [f64],
    pub wiggle_basis_d1: &'a [f64],
    pub wiggle_basis_d2: &'a [f64],
    pub wiggle_beta: &'a [f64],
}

fn validate_saved_wiggle_row(
    context: &str,
    basis: &[f64],
    basis_d1: &[f64],
    basis_d2: &[f64],
    beta: &[f64],
) -> Result<(), String> {
    let dimension = beta.len();
    if basis.len() != dimension || basis_d1.len() != dimension || basis_d2.len() != dimension {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "{context} saved wiggle row mismatch: beta={dimension}, basis={}, basis_d1={}, basis_d2={}",
                basis.len(),
                basis_d1.len(),
                basis_d2.len(),
            ),
        }
        .into());
    }
    if let Some((coordinate, value)) = basis
        .iter()
        .chain(basis_d1)
        .chain(basis_d2)
        .chain(beta)
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(GamlssError::NonFinite {
            reason: format!(
                "{context} saved wiggle row has a non-finite flattened coordinate {coordinate}: {value}"
            ),
        }
        .into());
    }
    Ok(())
}

#[inline]
fn dot_slices(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

/// Replay one Gaussian location-scale row in the raw saved coefficient frame.
///
/// The production certified kernel is evaluated in the standardized fit frame
/// and then transformed analytically to raw coordinates. This preserves the
/// fitter's extreme-value semantics while pairing the result with the raw
/// precision persisted after response rescaling.
pub fn gaussian_location_scale_alo_row_geometry(
    input: GaussianLocationScaleAloRowInput<'_>,
) -> Result<LocationScaleAloRowGeometry, String> {
    let GaussianLocationScaleAloRowInput {
        row,
        y,
        base_mean,
        eta_log_sigma,
        prior_weight,
        response_scale,
        wiggle_basis,
        wiggle_basis_d1,
        wiggle_basis_d2,
        wiggle_beta,
    } = input;
    validate_saved_wiggle_row(
        "Gaussian location-scale ALO",
        wiggle_basis,
        wiggle_basis_d1,
        wiggle_basis_d2,
        wiggle_beta,
    )?;
    if !(response_scale.is_finite() && response_scale > 0.0) {
        return Err(GamlssError::InvalidInput {
            reason: format!(
                "Gaussian location-scale ALO response scale must be finite and positive, got {response_scale}"
            ),
        }
        .into());
    }

    let warped_mean = base_mean + dot_slices(wiggle_basis, wiggle_beta);
    let internal = gaussian_diagonal_row_kernel(
        row,
        y / response_scale,
        warped_mean / response_scale,
        eta_log_sigma - response_scale.ln(),
        prior_weight,
        (2.0 * std::f64::consts::PI).ln(),
    )?;
    let inverse_scale = response_scale.recip();
    let nll_q = -internal.joint_m * inverse_scale;
    let nll_s = internal.kappa * (prior_weight - internal.joint_n);
    let h_qq = internal.joint_w * inverse_scale * inverse_scale;
    let h_qs = 2.0 * internal.kappa * internal.joint_m * inverse_scale;
    let h_ss = internal.kappa_prime * (prior_weight - internal.joint_n)
        + 2.0 * internal.kappa * internal.kappa * internal.joint_n;

    let wiggle_dimension = wiggle_beta.len();
    let dimension = 2 + wiggle_dimension;
    let warp_d1 = 1.0 + dot_slices(wiggle_basis_d1, wiggle_beta);
    let warp_d2 = dot_slices(wiggle_basis_d2, wiggle_beta);
    let mut score = Array1::<f64>::zeros(dimension);
    let mut hessian = Array2::<f64>::zeros((dimension, dimension));
    score[0] = nll_q * warp_d1;
    score[1] = nll_s;
    hessian[[0, 0]] = h_qq * warp_d1 * warp_d1 + nll_q * warp_d2;
    hessian[[0, 1]] = h_qs * warp_d1;
    hessian[[1, 0]] = hessian[[0, 1]];
    hessian[[1, 1]] = h_ss;

    for j in 0..wiggle_dimension {
        let wj = 2 + j;
        score[wj] = nll_q * wiggle_basis[j];
        hessian[[0, wj]] = h_qq * warp_d1 * wiggle_basis[j] + nll_q * wiggle_basis_d1[j];
        hessian[[wj, 0]] = hessian[[0, wj]];
        hessian[[1, wj]] = h_qs * wiggle_basis[j];
        hessian[[wj, 1]] = hessian[[1, wj]];
        for k in 0..wiggle_dimension {
            hessian[[wj, 2 + k]] = h_qq * wiggle_basis[j] * wiggle_basis[k];
        }
    }

    Ok(LocationScaleAloRowGeometry {
        nll_score: score,
        observed_hessian: hessian,
    })
}

/// Replay one binomial threshold-scale row in affine saved coordinates.
///
/// The q-space NLL derivatives are the same family-aware derivatives used by
/// fitting. The surrounding chain rule retains the exact second derivatives of
/// `q0 = -eta_t exp(-eta_s)` and optional `B(q0) beta_w`.
pub fn binomial_location_scale_alo_row_geometry(
    input: BinomialLocationScaleAloRowInput<'_>,
) -> Result<LocationScaleAloRowGeometry, String> {
    let BinomialLocationScaleAloRowInput {
        y,
        threshold_eta,
        eta_log_sigma,
        prior_weight,
        inverse_link,
        wiggle_basis,
        wiggle_basis_d1,
        wiggle_basis_d2,
        wiggle_beta,
    } = input;
    validate_saved_wiggle_row(
        "binomial location-scale ALO",
        wiggle_basis,
        wiggle_basis_d1,
        wiggle_basis_d2,
        wiggle_beta,
    )?;
    if !y.is_finite() || !(0.0..=1.0).contains(&y) {
        return Err(GamlssError::InvalidInput {
            reason: format!(
                "binomial location-scale ALO response must be finite and inside [0, 1], got {y}"
            ),
        }
        .into());
    }
    if !prior_weight.is_finite() || prior_weight < 0.0 {
        return Err(GamlssError::InvalidInput {
            reason: format!(
                "binomial location-scale ALO prior weight must be finite and non-negative, got {prior_weight}"
            ),
        }
        .into());
    }

    let wiggle_value = dot_slices(wiggle_basis, wiggle_beta);
    let fitted_row = binomial_location_scalerow(
        y,
        prior_weight,
        threshold_eta,
        eta_log_sigma,
        wiggle_value,
        inverse_link,
    )?;
    let q = fitted_row.q0 + wiggle_value;
    let (nll_q, h_qq, _) = binomial_neglog_q_derivatives_dispatch(
        y,
        prior_weight,
        q,
        fitted_row.inverse_link.mu,
        fitted_row.inverse_link.d1,
        fitted_row.inverse_link.d2,
        fitted_row.inverse_link.d3,
        inverse_link,
    );

    let wiggle_dimension = wiggle_beta.len();
    let dimension = 2 + wiggle_dimension;
    let q0_derivatives = nonwiggle_q_derivs(threshold_eta, fitted_row.sigma);
    let warp_d1 = 1.0 + dot_slices(wiggle_basis_d1, wiggle_beta);
    let warp_d2 = dot_slices(wiggle_basis_d2, wiggle_beta);
    let mut dq = Array1::<f64>::zeros(dimension);
    dq[0] = warp_d1 * q0_derivatives.q_t;
    dq[1] = warp_d1 * q0_derivatives.q_ls;
    for j in 0..wiggle_dimension {
        dq[2 + j] = wiggle_basis[j];
    }

    let mut d2q = Array2::<f64>::zeros((dimension, dimension));
    d2q[[0, 0]] = warp_d2 * q0_derivatives.q_t * q0_derivatives.q_t;
    d2q[[0, 1]] =
        warp_d2 * q0_derivatives.q_t * q0_derivatives.q_ls + warp_d1 * q0_derivatives.q_tl;
    d2q[[1, 0]] = d2q[[0, 1]];
    d2q[[1, 1]] =
        warp_d2 * q0_derivatives.q_ls * q0_derivatives.q_ls + warp_d1 * q0_derivatives.q_ll;
    for j in 0..wiggle_dimension {
        let wj = 2 + j;
        d2q[[0, wj]] = wiggle_basis_d1[j] * q0_derivatives.q_t;
        d2q[[wj, 0]] = d2q[[0, wj]];
        d2q[[1, wj]] = wiggle_basis_d1[j] * q0_derivatives.q_ls;
        d2q[[wj, 1]] = d2q[[1, wj]];
    }

    let score = dq.mapv(|value| nll_q * value);
    let mut hessian = Array2::<f64>::zeros((dimension, dimension));
    for i in 0..dimension {
        for j in 0..dimension {
            hessian[[i, j]] = h_qq * dq[i] * dq[j] + nll_q * d2q[[i, j]];
        }
    }
    Ok(LocationScaleAloRowGeometry {
        nll_score: score,
        observed_hessian: hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(label: &str, actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 2e-12 * (1.0 + expected.abs()),
            "{label}: actual={actual:.16e}, expected={expected:.16e}"
        );
    }

    #[test]
    fn gaussian_saved_alo_geometry_matches_raw_scale_closed_form() {
        let y = 7.0;
        let mean = 4.0;
        let eta_sigma = 0.3;
        let weight = 1.4;
        let response_scale = 5.0;
        let geometry = gaussian_location_scale_alo_row_geometry(GaussianLocationScaleAloRowInput {
            row: 0,
            y,
            base_mean: mean,
            eta_log_sigma: eta_sigma,
            prior_weight: weight,
            response_scale,
            wiggle_basis: &[],
            wiggle_basis_d1: &[],
            wiggle_basis_d2: &[],
            wiggle_beta: &[],
        })
        .expect("Gaussian saved row must replay");

        let sigma =
            response_scale * gam_model_kernels::sigma_link::LOGB_SIGMA_FLOOR + eta_sigma.exp();
        let kappa = eta_sigma.exp() / sigma;
        let residual = y - mean;
        let residual_sq = residual * residual / (sigma * sigma);
        let expected_score = [
            -weight * residual / (sigma * sigma),
            weight * kappa * (1.0 - residual_sq),
        ];
        let expected_hessian = [
            [
                weight / (sigma * sigma),
                2.0 * weight * kappa * residual / (sigma * sigma),
            ],
            [
                2.0 * weight * kappa * residual / (sigma * sigma),
                weight
                    * (kappa * (1.0 - kappa) * (1.0 - residual_sq)
                        + 2.0 * kappa * kappa * residual_sq),
            ],
        ];
        for i in 0..2 {
            assert_close("Gaussian score", geometry.nll_score[i], expected_score[i]);
            for j in 0..2 {
                assert_close(
                    "Gaussian observed Hessian",
                    geometry.observed_hessian[[i, j]],
                    expected_hessian[i][j],
                );
            }
        }
    }

    #[test]
    fn binomial_saved_alo_wiggle_geometry_matches_logistic_chain_closed_form() {
        let y = 1.0;
        let threshold = 0.6;
        let log_sigma = -0.2;
        let weight = 1.3;
        let basis = [0.4];
        let basis_d1 = [-0.15];
        let basis_d2 = [0.07];
        let beta = [0.25];
        let geometry = binomial_location_scale_alo_row_geometry(BinomialLocationScaleAloRowInput {
            y,
            threshold_eta: threshold,
            eta_log_sigma: log_sigma,
            prior_weight: weight,
            inverse_link: &InverseLink::Standard(StandardLink::Logit),
            wiggle_basis: &basis,
            wiggle_basis_d1: &basis_d1,
            wiggle_basis_d2: &basis_d2,
            wiggle_beta: &beta,
        })
        .expect("binomial saved row must replay");

        let q0 = -threshold * (-log_sigma).exp();
        let q = q0 + basis[0] * beta[0];
        let probability = gam_linalg::utils::stable_logistic(q);
        let f1 = weight * (probability - y);
        let f2 = weight * probability * (1.0 - probability);
        let a = 1.0 + basis_d1[0] * beta[0];
        let b = basis_d2[0] * beta[0];
        let q0_t = -(-log_sigma).exp();
        let q0_s = -q0;
        let q0_ts = (-log_sigma).exp();
        let q0_ss = q0;
        let dq = [a * q0_t, a * q0_s, basis[0]];
        let d2q = [
            [
                b * q0_t * q0_t,
                b * q0_t * q0_s + a * q0_ts,
                basis_d1[0] * q0_t,
            ],
            [
                b * q0_s * q0_t + a * q0_ts,
                b * q0_s * q0_s + a * q0_ss,
                basis_d1[0] * q0_s,
            ],
            [basis_d1[0] * q0_t, basis_d1[0] * q0_s, 0.0],
        ];
        for i in 0..3 {
            assert_close("binomial score", geometry.nll_score[i], f1 * dq[i]);
            for j in 0..3 {
                assert_close(
                    "binomial observed Hessian",
                    geometry.observed_hessian[[i, j]],
                    f2 * dq[i] * dq[j] + f1 * d2q[i][j],
                );
            }
        }
    }
}
