use super::{TRANSFORMATION_MONOTONICITY_EPS, log_normal_cdf_diff_derivatives};
use ndarray::{Array1, Array2};

/// Complete local state for one saved transformation-normal likelihood row.
pub struct TransformationNormalAloRowInput<'a> {
    pub response_value_basis: &'a [f64],
    pub response_derivative_basis: &'a [f64],
    pub response_lower_basis: &'a [f64],
    pub response_upper_basis: &'a [f64],
    pub alpha: &'a [f64],
    pub additive_offset: f64,
    pub response_floor_offset: f64,
    pub response_lower_floor_offset: f64,
    pub response_upper_floor_offset: f64,
    pub prior_weight: f64,
}

/// Exact negative-log-likelihood derivatives in the affine local coordinates
/// `alpha_k(x) = covariate_row(x) beta_k`.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformationNormalAloRowGeometry {
    pub negative_log_likelihood: f64,
    pub nll_score: Array1<f64>,
    pub observed_hessian: Array2<f64>,
}

fn validate_row(input: &TransformationNormalAloRowInput<'_>) -> Result<usize, String> {
    let dimension = input.alpha.len();
    if dimension == 0
        || input.response_value_basis.len() != dimension
        || input.response_derivative_basis.len() != dimension
        || input.response_lower_basis.len() != dimension
        || input.response_upper_basis.len() != dimension
    {
        return Err(format!(
            "transformation-normal ALO row dimension mismatch: alpha={dimension}, value={}, derivative={}, lower={}, upper={}",
            input.response_value_basis.len(),
            input.response_derivative_basis.len(),
            input.response_lower_basis.len(),
            input.response_upper_basis.len(),
        ));
    }
    if !input.prior_weight.is_finite() || input.prior_weight < 0.0 {
        return Err(format!(
            "transformation-normal ALO prior weight must be finite and non-negative, got {}",
            input.prior_weight
        ));
    }
    if input
        .response_value_basis
        .iter()
        .chain(input.response_derivative_basis)
        .chain(input.response_lower_basis)
        .chain(input.response_upper_basis)
        .chain(input.alpha)
        .copied()
        .chain([
            input.additive_offset,
            input.response_floor_offset,
            input.response_lower_floor_offset,
            input.response_upper_floor_offset,
        ])
        .any(|value| !value.is_finite())
    {
        return Err("transformation-normal ALO row state must be finite".to_string());
    }
    Ok(dimension)
}

/// Replay one row of the fitted finite-support SCOP likelihood.
///
/// This is the row factorization of the same score and negative Hessian used by
/// `TransformationNormalFamily`: every component is affine in direct-alpha
/// coordinates, the monotonicity derivative floor is exact, and both
/// transformed support endpoints contribute through the normalized Gaussian
/// mass. Feasibility of the shape coordinates is owned by the fitted model's
/// Khatri-Rao cone before this row replay is called.
pub fn transformation_normal_alo_row_geometry(
    input: TransformationNormalAloRowInput<'_>,
) -> Result<TransformationNormalAloRowGeometry, String> {
    let dimension = validate_row(&input)?;
    if input.prior_weight == 0.0 {
        return Ok(TransformationNormalAloRowGeometry {
            negative_log_likelihood: 0.0,
            nll_score: Array1::zeros(dimension),
            observed_hessian: Array2::zeros((dimension, dimension)),
        });
    }

    let alpha0 = input.alpha[0];
    let mut h = input.response_value_basis[0] * alpha0
        + input.additive_offset
        + input.response_floor_offset;
    let mut h_prime = input.response_derivative_basis[0] * alpha0 + TRANSFORMATION_MONOTONICITY_EPS;
    let mut lower = input.response_lower_basis[0] * alpha0
        + input.additive_offset
        + input.response_lower_floor_offset;
    let mut upper = input.response_upper_basis[0] * alpha0
        + input.additive_offset
        + input.response_upper_floor_offset;
    for component in 1..dimension {
        let alpha = input.alpha[component];
        h += input.response_value_basis[component] * alpha;
        h_prime += input.response_derivative_basis[component] * alpha;
        lower += input.response_lower_basis[component] * alpha;
        upper += input.response_upper_basis[component] * alpha;
    }
    if !(h.is_finite() && h_prime.is_finite() && lower.is_finite() && upper.is_finite()) {
        return Err(format!(
            "transformation-normal ALO row transform is non-finite: h={h}, h_prime={h_prime}, lower={lower}, upper={upper}"
        ));
    }
    if h_prime <= 0.0 {
        return Err(format!(
            "transformation-normal ALO row derivative must be positive, got {h_prime}"
        ));
    }
    let endpoint = log_normal_cdf_diff_derivatives(upper, lower)?;
    let weight = input.prior_weight;
    let negative_log_likelihood = weight
        * (0.5 * h * h + 0.5 * (2.0 * std::f64::consts::PI).ln() - h_prime.ln() + endpoint.log_z);

    let mut dh = vec![0.0; dimension];
    let mut dh_prime = vec![0.0; dimension];
    let mut dlower = vec![0.0; dimension];
    let mut dupper = vec![0.0; dimension];
    for component in 0..dimension {
        dh[component] = input.response_value_basis[component];
        dh_prime[component] = input.response_derivative_basis[component];
        dlower[component] = input.response_lower_basis[component];
        dupper[component] = input.response_upper_basis[component];
    }

    let inverse_h_prime = 1.0 / h_prime;
    let inverse_h_prime_squared = inverse_h_prime * inverse_h_prime;
    let mut nll_score = Array1::<f64>::zeros(dimension);
    let mut observed_hessian = Array2::<f64>::zeros((dimension, dimension));
    for left in 0..dimension {
        let endpoint_first = endpoint.first[0] * dupper[left] + endpoint.first[1] * dlower[left];
        nll_score[left] =
            weight * (h * dh[left] - dh_prime[left] * inverse_h_prime + endpoint_first);
        for right in 0..dimension {
            let endpoint_second = endpoint.second[0][0] * dupper[left] * dupper[right]
                + endpoint.second[0][1] * dupper[left] * dlower[right]
                + endpoint.second[1][0] * dlower[left] * dupper[right]
                + endpoint.second[1][1] * dlower[left] * dlower[right];
            observed_hessian[[left, right]] = weight
                * (dh[left] * dh[right]
                    + dh_prime[left] * dh_prime[right] * inverse_h_prime_squared
                    + endpoint_second);
        }
    }
    if !negative_log_likelihood.is_finite()
        || nll_score.iter().any(|value| !value.is_finite())
        || observed_hessian.iter().any(|value| !value.is_finite())
    {
        return Err("transformation-normal ALO row geometry is non-finite".to_string());
    }
    Ok(TransformationNormalAloRowGeometry {
        negative_log_likelihood,
        nll_score,
        observed_hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformation_normal::log_normal_cdf_diff;

    fn scalar_nll(alpha: [f64; 2]) -> f64 {
        let value = [1.0, 0.4];
        let derivative = [0.0, 0.7];
        let lower_basis = [1.0, 0.1];
        let upper_basis = [1.0, 0.9];
        let offset = -0.15;
        let floor = 0.02;
        let lower_floor = -0.04;
        let upper_floor = 0.06;
        let weight = 1.3;
        let h = value[0] * alpha[0] + value[1] * alpha[1] + offset + floor;
        let h_prime = TRANSFORMATION_MONOTONICITY_EPS
            + derivative[0] * alpha[0]
            + derivative[1] * alpha[1];
        let lower =
            lower_basis[0] * alpha[0] + lower_basis[1] * alpha[1] + offset + lower_floor;
        let upper =
            upper_basis[0] * alpha[0] + upper_basis[1] * alpha[1] + offset + upper_floor;
        weight
            * (0.5 * h * h + 0.5 * (2.0 * std::f64::consts::PI).ln() - h_prime.ln()
                + log_normal_cdf_diff(upper, lower).expect("finite endpoint mass"))
    }

    #[test]
    fn saved_transformation_row_geometry_matches_independent_scalar_finite_difference() {
        let alpha: [f64; 2] = [0.25, 0.8];
        let geometry = transformation_normal_alo_row_geometry(TransformationNormalAloRowInput {
            response_value_basis: &[1.0, 0.4],
            response_derivative_basis: &[0.0, 0.7],
            response_lower_basis: &[1.0, 0.1],
            response_upper_basis: &[1.0, 0.9],
            alpha: &alpha,
            additive_offset: -0.15,
            response_floor_offset: 0.02,
            response_lower_floor_offset: -0.04,
            response_upper_floor_offset: 0.06,
            prior_weight: 1.3,
        })
        .expect("saved transformation-normal row must replay");
        let step = 2.0e-5;
        let base = scalar_nll(alpha);
        assert!((geometry.negative_log_likelihood - base).abs() <= 2.0e-13);
        for axis in 0..2 {
            let mut plus = alpha;
            let mut minus = alpha;
            plus[axis] += step;
            minus[axis] -= step;
            let gradient_fd = (scalar_nll(plus) - scalar_nll(minus)) / (2.0 * step);
            assert!(
                (geometry.nll_score[axis] - gradient_fd).abs() <= 2.0e-8,
                "score[{axis}] analytic={} fd={gradient_fd}",
                geometry.nll_score[axis]
            );
            for other in 0..2 {
                let mut pp = alpha;
                let mut pm = alpha;
                let mut mp = alpha;
                let mut mm = alpha;
                pp[axis] += step;
                pp[other] += step;
                pm[axis] += step;
                pm[other] -= step;
                mp[axis] -= step;
                mp[other] += step;
                mm[axis] -= step;
                mm[other] -= step;
                let hessian_fd = (scalar_nll(pp) - scalar_nll(pm) - scalar_nll(mp)
                    + scalar_nll(mm))
                    / (4.0 * step * step);
                assert!(
                    (geometry.observed_hessian[[axis, other]] - hessian_fd).abs() <= 3.0e-6,
                    "hessian[{axis},{other}] analytic={} fd={hessian_fd}",
                    geometry.observed_hessian[[axis, other]]
                );
            }
        }
        assert!(
            (geometry.observed_hessian[[0, 0]] - geometry.nll_score[0].powi(2)).abs() > 1.0e-3,
            "observed curvature must remain distinct from score covariance"
        );
    }
}
