use super::family::bernoulli_marginal_link_map;
use super::gradient_paths::rigid_standard_normal_row_kernel;
use gam_problem::InverseLink;

/// Complete saved-row state for exact rigid Bernoulli marginal-slope ALO replay.
pub struct BernoulliMarginalSlopeAloRowInput<'a> {
    pub base_link: &'a InverseLink,
    pub marginal_eta: f64,
    pub slope: f64,
    pub latent_z: f64,
    pub response: f64,
    pub prior_weight: f64,
    pub probit_frailty_scale: f64,
}

/// Negative-log-likelihood derivatives in the affine fitted coordinates
/// `[marginal eta, slope]` for one saved Bernoulli marginal-slope row.
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliMarginalSlopeAloRowGeometry {
    pub negative_log_likelihood: f64,
    pub nll_score: [f64; 2],
    pub observed_hessian: [[f64; 2]; 2],
}

/// Replay the exact rigid standard-normal row program used by fitting.
///
/// The latent score supplied here must already be in the fitted normalized and
/// calibrated coordinate system. Gaussian-shift frailty is represented by the
/// persisted probit scale, so no prediction-time approximation enters the
/// score or observed Hessian.
pub fn bernoulli_marginal_slope_alo_row_geometry(
    input: BernoulliMarginalSlopeAloRowInput<'_>,
) -> Result<BernoulliMarginalSlopeAloRowGeometry, String> {
    let marginal = bernoulli_marginal_link_map(input.base_link, input.marginal_eta)?;
    let (negative_log_likelihood, nll_score, observed_hessian) = rigid_standard_normal_row_kernel(
        marginal,
        input.slope,
        input.latent_z,
        input.response,
        input.prior_weight,
        input.probit_frailty_scale,
    )?;
    Ok(BernoulliMarginalSlopeAloRowGeometry {
        negative_log_likelihood,
        nll_score,
        observed_hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::probability::{normal_cdf, normal_pdf};
    use gam_problem::StandardLink;

    fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn rigid_saved_alo_geometry_matches_independent_probit_chain_rule() {
        let marginal_eta: f64 = 0.35;
        let slope: f64 = -0.6;
        let latent_z: f64 = 0.8;
        let response: f64 = 1.0;
        let weight: f64 = 1.7;
        let scale: f64 = 0.75;
        let geometry =
            bernoulli_marginal_slope_alo_row_geometry(BernoulliMarginalSlopeAloRowInput {
                base_link: &InverseLink::Standard(StandardLink::Probit),
                marginal_eta,
                slope,
                latent_z,
                response,
                prior_weight: weight,
                probit_frailty_scale: scale,
            })
            .expect("rigid saved marginal-slope row must replay");

        // Independent closed form for eta(q, g) = q sqrt(1 + (s g)^2) + s g z.
        // At an interior probit marginal map q == marginal_eta exactly.
        let sg = scale * slope;
        let c = (1.0 + sg * sg).sqrt();
        let eta = marginal_eta * c + sg * latent_z;
        let sign = 2.0 * response - 1.0;
        let margin = sign * eta;
        let cdf = normal_cdf(margin);
        let mills = normal_pdf(margin) / cdf;
        let nll_first_eta = -weight * sign * mills;
        let nll_second_eta = weight * mills * (margin + mills);

        let eta_q = c;
        let eta_g = marginal_eta * scale * scale * slope / c + scale * latent_z;
        let eta_qg = scale * scale * slope / c;
        let eta_gg = marginal_eta * scale * scale / c.powi(3);
        let expected_score = [nll_first_eta * eta_q, nll_first_eta * eta_g];
        let expected_hessian = [
            [
                nll_second_eta * eta_q * eta_q,
                nll_second_eta * eta_q * eta_g + nll_first_eta * eta_qg,
            ],
            [
                nll_second_eta * eta_q * eta_g + nll_first_eta * eta_qg,
                nll_second_eta * eta_g * eta_g + nll_first_eta * eta_gg,
            ],
        ];

        assert_close(
            "negative log likelihood",
            geometry.negative_log_likelihood,
            -weight * cdf.ln(),
            2e-13,
        );
        for axis in 0..2 {
            assert_close(
                &format!("score[{axis}]"),
                geometry.nll_score[axis],
                expected_score[axis],
                2e-12,
            );
            for other in 0..2 {
                assert_close(
                    &format!("hessian[{axis},{other}]"),
                    geometry.observed_hessian[axis][other],
                    expected_hessian[axis][other],
                    3e-12,
                );
            }
        }

        let score_meat = geometry.nll_score[0] * geometry.nll_score[0];
        assert!(
            (geometry.observed_hessian[0][0] - score_meat).abs() > 1e-3,
            "observed Hessian and empirical score meat must remain distinct"
        );
    }
}
