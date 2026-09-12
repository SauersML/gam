//! Cancellation-free natural-coordinate derivatives for Bernoulli links.
//!
//! A bounded inverse-link jet `(mu, mu', mu'', mu''', mu'''')` is not enough for a
//! numerically honest Bernoulli likelihood: either `mu` or `1 - mu` rounds to
//! an endpoint in the tails, precisely where the corresponding log probability
//! and score can still be finite and informative. This module carries the two
//! log-probability derivative towers directly. It is the single kernel used by
//! scalar and separable vector-response Bernoulli families.

use gam_problem::EstimationError;
use gam_solve::mixture_link::{
    inverse_link_jet_for_inverse_link, inverse_link_pdfthird_derivative_for_inverse_link,
};
use gam_spec::{InverseLink, StandardLink};

use crate::natural_observation::NaturalDiagonalObservation;

/// Natural-coordinate derivative tower for a Bernoulli inverse link.
///
/// `log_mu[j]` and `log_one_minus_mu[j]` are the `j`th derivatives with
/// respect to the linear predictor for `j = 0, 1, 2, 3, 4`. `log_fisher` is
/// `log((d mu / d eta)^2 / (mu (1 - mu)))`, evaluated without reconstructing
/// a rounded endpoint probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BernoulliNaturalJet {
    pub mu: f64,
    pub log_mu: [f64; 5],
    pub log_one_minus_mu: [f64; 5],
    pub log_fisher: f64,
}

/// One unweighted Bernoulli observation evaluated in natural coordinates.
///
/// The observed-curvature channels are derivatives of the exact log
/// likelihood. `log_fisher` is the expected-information channel used by
/// Fisher scoring. Keeping both is essential for noncanonical links: they are
/// equal only for the canonical logit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BernoulliNaturalObservation {
    pub mu: f64,
    pub log_likelihood: f64,
    pub score: f64,
    pub log_fisher: f64,
    pub negative_hessian: f64,
    pub negative_hessian_derivative: f64,
    pub negative_hessian_second_derivative: f64,
}

impl From<BernoulliNaturalObservation> for NaturalDiagonalObservation {
    fn from(observation: BernoulliNaturalObservation) -> Self {
        Self {
            log_likelihood: observation.log_likelihood,
            score: observation.score,
            negative_hessian: observation.negative_hessian,
            negative_hessian_derivative: observation.negative_hessian_derivative,
            negative_hessian_second_derivative: observation.negative_hessian_second_derivative,
        }
    }
}

#[inline]
fn response_mixture(y: f64, when_one: f64, when_zero: f64) -> f64 {
    if y == 0.0 {
        when_zero
    } else if y == 1.0 {
        when_one
    } else {
        y.mul_add(when_one, (1.0 - y) * when_zero)
    }
}


#[inline]
fn logit_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let tail = (-eta.abs()).exp();
    let (mu, one_minus_mu) = if eta >= 0.0 {
        let q = tail / (1.0 + tail);
        (1.0 - q, q)
    } else {
        let p = tail / (1.0 + tail);
        (p, 1.0 - p)
    };
    let curvature = mu * one_minus_mu;
    let third = curvature * (mu - one_minus_mu);
    let fourth = curvature * (2.0 * curvature - (mu - one_minus_mu).powi(2));
    BernoulliNaturalJet {
        mu,
        log_mu: [
            -gam_linalg::utils::stable_softplus(-eta),
            one_minus_mu,
            -curvature,
            third,
            fourth,
        ],
        log_one_minus_mu: [
            -gam_linalg::utils::stable_softplus(eta),
            -mu,
            -curvature,
            third,
            fourth,
        ],
        log_fisher: -gam_linalg::utils::stable_softplus(eta)
            - gam_linalg::utils::stable_softplus(-eta),
    }
}

#[inline]
fn probit_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let left = gam_math::probability::normal_logcdf_derivatives(eta);
    let right_at_neg_eta = gam_math::probability::normal_logcdf_derivatives(-eta);
    let log_pdf = if eta.abs() <= f64::MAX.sqrt() {
        -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln()
    } else {
        f64::NEG_INFINITY
    };
    BernoulliNaturalJet {
        mu: left[0].exp(),
        log_mu: [left[0], left[1], left[2], left[3], left[4]],
        log_one_minus_mu: [
            right_at_neg_eta[0],
            -right_at_neg_eta[1],
            right_at_neg_eta[2],
            -right_at_neg_eta[3],
            right_at_neg_eta[4],
        ],
        log_fisher: 2.0 * log_pdf - left[0] - right_at_neg_eta[0],
    }
}

#[inline]
fn cloglog_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let x = eta.exp();
    if x == f64::INFINITY {
        return BernoulliNaturalJet {
            mu: 1.0,
            log_mu: [0.0; 5],
            log_one_minus_mu: [f64::NEG_INFINITY; 5],
            log_fisher: f64::NEG_INFINITY,
        };
    }
    if x == 0.0 {
        return BernoulliNaturalJet {
            mu: 0.0,
            log_mu: [eta, 1.0, 0.0, 0.0, 0.0],
            log_one_minus_mu: [0.0; 5],
            log_fisher: eta,
        };
    }
    let mu = -(-x).exp_m1();
    let log_mu_jet = if x <= 0.01 {
        // log(1-exp(-x)) = log(x) - x/2 + x²/24 - x⁴/2880
        //                       + x⁶/181440 + O(x⁸).
        // Applying d/deta = x d/dx multiplies each x^k term by k.
        // The direct formula 1-x-x/expm1(x) loses this curvature when
        // x is small. At this cutoff the omitted fourth derivative is
        // less than 4.3e-20.
        let p1 = -0.5 * x;
        let p2 = x * x / 24.0;
        let p4 = -x.powi(4) / 2880.0;
        let p6 = x.powi(6) / 181440.0;
        [
            eta + p1 + p2 + p4 + p6,
            1.0 + p1 + 2.0 * p2 + 4.0 * p4 + 6.0 * p6,
            p1 + 4.0 * p2 + 16.0 * p4 + 36.0 * p6,
            p1 + 8.0 * p2 + 64.0 * p4 + 216.0 * p6,
            p1 + 16.0 * p2 + 256.0 * p4 + 1296.0 * p6,
        ]
    } else if x < 1.0 {
        let log_mu = eta + (mu / x).ln();
        let h = x / x.exp_m1();
        let a = 1.0 - x - h;
        let b = a * a - x - h * a;
        let b_derivative =
            -x * (2.0 * a + 1.0 - h) - 3.0 * h * a * a + h * h * a;
        [log_mu, h, h * a, h * b, h * (a * b + b_derivative)]
    } else {
        // Sum the geometric-series derivative factors with x^j exp(-x)
        // formed in log space. This preserves representable derivatives
        // after exp(-x) underflows and avoids 0 * infinity at large x.
        let q = (-x).exp();
        let inv = 1.0 / mu;
        let p1 = (eta - x).exp() * inv;
        let p2 = (2.0 * eta - x).exp() * inv.powi(2);
        let p3 = (3.0 * eta - x).exp() * (1.0 + q) * inv.powi(3);
        let p4 = (4.0 * eta - x).exp() * (1.0 + 4.0 * q + q * q) * inv.powi(4);
        [
            (-q).ln_1p(),
            p1,
            p1 - p2,
            p1 - 3.0 * p2 + p3,
            p1 - 7.0 * p2 + 6.0 * p3 - p4,
        ]
    };
    BernoulliNaturalJet {
        mu,
        log_mu: log_mu_jet,
        log_one_minus_mu: [-x, -x, -x, -x, -x],
        log_fisher: 2.0 * eta - x - log_mu_jet[0],
    }
}

#[inline]
fn loglog_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let mirrored = cloglog_natural_jet(-eta);
    BernoulliNaturalJet {
        mu: mirrored.log_one_minus_mu[0].exp(),
        log_mu: [
            mirrored.log_one_minus_mu[0],
            -mirrored.log_one_minus_mu[1],
            mirrored.log_one_minus_mu[2],
            -mirrored.log_one_minus_mu[3],
            mirrored.log_one_minus_mu[4],
        ],
        log_one_minus_mu: [
            mirrored.log_mu[0],
            -mirrored.log_mu[1],
            mirrored.log_mu[2],
            -mirrored.log_mu[3],
            mirrored.log_mu[4],
        ],
        log_fisher: mirrored.log_fisher,
    }
}

#[inline]
fn cauchit_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let (mu, one_minus_mu) = if eta > 0.0 {
        let q = (eta.recip()).atan() / std::f64::consts::PI;
        (1.0 - q, q)
    } else if eta < 0.0 {
        let p = (-eta.recip()).atan() / std::f64::consts::PI;
        (p, 1.0 - p)
    } else {
        (0.5, 0.5)
    };
    let (log_mu, log_one_minus_mu) = if eta >= 0.0 {
        ((-one_minus_mu).ln_1p(), one_minus_mu.ln())
    } else {
        (mu.ln(), (-mu).ln_1p())
    };
    let abs_eta = eta.abs();
    let log_one_plus_eta_sq = if abs_eta <= f64::MAX.sqrt() {
        (eta * eta).ln_1p()
    } else {
        2.0 * abs_eta.ln() + eta.recip().powi(2).ln_1p()
    };
    let log_d1 = -std::f64::consts::PI.ln() - log_one_plus_eta_sq;
    let ratio = if abs_eta <= 1.0 {
        eta / (1.0 + eta * eta)
    } else {
        1.0 / (eta + eta.recip())
    };
    let d2_over_d1 = -2.0 * ratio;
    let inv_one_plus_sq = if abs_eta <= 1.0 {
        1.0 / (1.0 + eta * eta)
    } else {
        let inv = eta.recip();
        inv * inv / (1.0 + inv * inv)
    };
    let d3_over_d1 = inv_one_plus_sq * (6.0 * (eta * ratio) - 2.0 * inv_one_plus_sq);
    let d4_over_d1 = 24.0 * ratio * (inv_one_plus_sq * inv_one_plus_sq - ratio * ratio);
    let d1_over_mu = (log_d1 - log_mu).exp();
    let d1_over_q = (log_d1 - log_one_minus_mu).exp();
    let left_d2_ratio = d2_over_d1 * d1_over_mu;
    let right_d2_ratio = d2_over_d1 * d1_over_q;
    let left_d3_ratio = d3_over_d1 * d1_over_mu;
    let right_d3_ratio = d3_over_d1 * d1_over_q;
    let left_d4_ratio = d4_over_d1 * d1_over_mu;
    let right_d4_ratio = d4_over_d1 * d1_over_q;
    BernoulliNaturalJet {
        mu,
        log_mu: [
            log_mu,
            d1_over_mu,
            left_d2_ratio - d1_over_mu * d1_over_mu,
            left_d3_ratio - 3.0 * d1_over_mu * left_d2_ratio + 2.0 * d1_over_mu.powi(3),
            left_d4_ratio
                - 4.0 * d1_over_mu * left_d3_ratio
                - 3.0 * left_d2_ratio * left_d2_ratio
                + 12.0 * d1_over_mu * d1_over_mu * left_d2_ratio
                - 6.0 * d1_over_mu.powi(4),
        ],
        log_one_minus_mu: [
            log_one_minus_mu,
            -d1_over_q,
            -right_d2_ratio - d1_over_q * d1_over_q,
            -right_d3_ratio - 3.0 * d1_over_q * right_d2_ratio - 2.0 * d1_over_q.powi(3),
            -right_d4_ratio
                - 4.0 * d1_over_q * right_d3_ratio
                - 3.0 * right_d2_ratio * right_d2_ratio
                - 12.0 * d1_over_q * d1_over_q * right_d2_ratio
                - 6.0 * d1_over_q.powi(4),
        ],
        log_fisher: 2.0 * log_d1 - log_mu - log_one_minus_mu,
    }
}

#[inline]
fn generic_natural_jet(
    row: usize,
    eta: f64,
    link: &InverseLink,
) -> Result<BernoulliNaturalJet, EstimationError> {
    let jet = inverse_link_jet_for_inverse_link(link, eta)?;
    let d4 = inverse_link_pdfthird_derivative_for_inverse_link(link, eta)?;
    if !(jet.mu.is_finite()
        && jet.mu > 0.0
        && jet.mu < 1.0
        && jet.d1.is_finite()
        && jet.d1 > 0.0
        && jet.d2.is_finite()
        && jet.d3.is_finite()
        && d4.is_finite())
    {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "bounded-family inverse-link jet",
            eta,
            jet.mu,
        ));
    }
    let mu = jet.mu;
    let q = 1.0 - mu;
    let r1 = jet.d1 / mu;
    let r2 = jet.d2 / mu;
    let r3 = jet.d3 / mu;
    let r4 = d4 / mu;
    let s1 = jet.d1 / q;
    let s2 = jet.d2 / q;
    let s3 = jet.d3 / q;
    let s4 = d4 / q;
    Ok(BernoulliNaturalJet {
        mu,
        log_mu: [
            mu.ln(),
            r1,
            r2 - r1 * r1,
            r3 - 3.0 * r1 * r2 + 2.0 * r1.powi(3),
            r4 - 4.0 * r1 * r3 - 3.0 * r2 * r2 + 12.0 * r1 * r1 * r2
                - 6.0 * r1.powi(4),
        ],
        log_one_minus_mu: [
            (-mu).ln_1p(),
            -s1,
            -s2 - s1 * s1,
            -s3 - 3.0 * s1 * s2 - 2.0 * s1.powi(3),
            -s4 - 4.0 * s1 * s3 - 3.0 * s2 * s2 - 12.0 * s1 * s1 * s2
                - 6.0 * s1.powi(4),
        ],
        log_fisher: 2.0 * jet.d1.ln() - mu.ln() - q.ln(),
    })
}

/// Evaluate a Bernoulli inverse link as two cancellation-free log-probability
/// derivative towers.
///
/// The standard bounded links have dedicated tail kernels. Parameterized
/// bounded links use the central inverse-link jet and are rejected if a trial
/// point cannot represent an interior probability with finite derivatives.
/// Identity and log links are not Bernoulli links and are rejected by the same
/// domain contract rather than silently clamped.
pub fn bernoulli_natural_jet(
    row: usize,
    eta: f64,
    link: &InverseLink,
) -> Result<BernoulliNaturalJet, EstimationError> {
    match link {
        InverseLink::Standard(StandardLink::Logit) => Ok(logit_natural_jet(eta)),
        InverseLink::Standard(StandardLink::Probit) => Ok(probit_natural_jet(eta)),
        InverseLink::Standard(StandardLink::CLogLog) => Ok(cloglog_natural_jet(eta)),
        InverseLink::Standard(StandardLink::LogLog) => Ok(loglog_natural_jet(eta)),
        InverseLink::Standard(StandardLink::Cauchit) => Ok(cauchit_natural_jet(eta)),
        InverseLink::Standard(link @ (StandardLink::Identity | StandardLink::Log)) => {
            Err(EstimationError::InvalidInput(format!(
                "Bernoulli likelihood requires a bounded inverse link; `{}` is not bounded to [0,1]",
                link.name()
            )))
        }
        _ => generic_natural_jet(row, eta, link),
    }
}

/// Evaluate one unweighted Bernoulli/proportion observation.
///
/// Hard `0/1` outcomes select one log-probability tower without multiplying
/// the other endpoint by zero, so a correct saturated tail never becomes
/// `0 * infinity = NaN`. Fractional responses use their literal binomial
/// proportion likelihood and therefore require both sides to be representable.
pub fn bernoulli_natural_observation(
    row: usize,
    y: f64,
    eta: f64,
    link: &InverseLink,
) -> Result<BernoulliNaturalObservation, EstimationError> {
    if !(y.is_finite() && (0.0..=1.0).contains(&y)) {
        return Err(EstimationError::InvalidInput(format!(
            "Bernoulli response at row {row} must be finite and in [0,1], got {y}"
        )));
    }
    let jet = bernoulli_natural_jet(row, eta, link)?;
    Ok(BernoulliNaturalObservation {
        mu: jet.mu,
        log_likelihood: response_mixture(y, jet.log_mu[0], jet.log_one_minus_mu[0]),
        score: response_mixture(y, jet.log_mu[1], jet.log_one_minus_mu[1]),
        log_fisher: jet.log_fisher,
        negative_hessian: -response_mixture(y, jet.log_mu[2], jet.log_one_minus_mu[2]),
        negative_hessian_derivative: -response_mixture(
            y,
            jet.log_mu[3],
            jet.log_one_minus_mu[3],
        ),
        negative_hessian_second_derivative: -response_mixture(
            y,
            jet.log_mu[4],
            jet.log_one_minus_mu[4],
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_and_cloglog_keep_informative_log_tails() {
        let logit = bernoulli_natural_jet(0, 1_000.0, &InverseLink::Standard(StandardLink::Logit))
            .expect("logit tail");
        assert_eq!(logit.mu, 1.0);
        assert_eq!(logit.log_one_minus_mu[0], -1_000.0);
        assert_eq!(logit.log_one_minus_mu[1], -1.0);

        let cloglog =
            bernoulli_natural_jet(0, -1_000.0, &InverseLink::Standard(StandardLink::CLogLog))
                .expect("cloglog tail");
        assert_eq!(cloglog.mu, 0.0);
        assert_eq!(cloglog.log_mu[0], -1_000.0);
        assert_eq!(cloglog.log_mu[1], 1.0);
    }

    #[test]
    fn loglog_is_exact_cloglog_mirror() {
        for eta in [-12.0, -1.25, 0.0, 2.5, 12.0] {
            let left = bernoulli_natural_jet(0, eta, &InverseLink::Standard(StandardLink::LogLog))
                .expect("loglog jet");
            let right =
                bernoulli_natural_jet(0, -eta, &InverseLink::Standard(StandardLink::CLogLog))
                    .expect("cloglog jet");
            assert_eq!(left.log_mu[0], right.log_one_minus_mu[0]);
            assert_eq!(left.log_one_minus_mu[0], right.log_mu[0]);
            assert_eq!(left.log_fisher, right.log_fisher);
        }
    }

    #[test]
    fn cloglog_tail_derivatives_keep_representable_curvature() {
        let negative = cloglog_natural_jet(-40.0);
        let leading = -0.5 * (-40.0_f64).exp();
        for derivative in &negative.log_mu[2..] {
            assert!((derivative / leading - 1.0).abs() < 1.0e-14);
        }

        let positive = cloglog_natural_jet(40.0_f64.ln());
        assert!(positive.log_mu[0] < 0.0);
        assert!((positive.log_mu[0] / -(-40.0_f64).exp() - 1.0).abs() < 1.0e-13);

        // exp(-750) is zero in f64, but x^4 exp(-x) is representable.
        let eta = 750.0_f64.ln();
        let x = eta.exp();
        let underflow = cloglog_natural_jet(eta);
        let expected_fourth = -(4.0 * eta - x).exp()
            * (1.0 - 6.0 / x + 7.0 / (x * x) - 1.0 / x.powi(3));
        assert!(underflow.log_mu[2] < 0.0);
        assert!(underflow.log_mu[3] > 0.0);
        assert!((underflow.log_mu[4] / expected_fourth - 1.0).abs() < 1.0e-7);

        for eta in [400.0, 709.0] {
            let saturated = cloglog_natural_jet(eta);
            assert_eq!(saturated.log_mu, [0.0; 5]);
            assert!(saturated.log_one_minus_mu.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn cauchit_keeps_log_probability_after_probability_rounds_to_one() {
        for eta in [1.0e20_f64, 1.0e100] {
            let positive = cauchit_natural_jet(eta);
            let negative = cauchit_natural_jet(-eta);
            let leading = -eta.recip() / std::f64::consts::PI;
            assert_eq!(positive.mu, 1.0);
            assert!(positive.log_mu[0] < 0.0);
            assert!((positive.log_mu[0] / leading - 1.0).abs() < 1.0e-14);
            assert_eq!(positive.log_mu[0], negative.log_one_minus_mu[0]);
        }
    }

    #[test]
    fn identity_is_not_accepted_as_a_bernoulli_link() {
        let error = bernoulli_natural_jet(7, 0.5, &InverseLink::Standard(StandardLink::Identity))
            .expect_err("identity must not enter a Bernoulli likelihood");
        assert!(matches!(error, EstimationError::InvalidInput(_)));
    }

    #[test]
    fn observation_score_and_curvature_match_log_likelihood_differences() {
        for link in [
            StandardLink::Logit,
            StandardLink::Probit,
            StandardLink::CLogLog,
            StandardLink::LogLog,
            StandardLink::Cauchit,
        ] {
            let link = InverseLink::Standard(link);
            for (row, eta) in [-2.0, -0.25, 0.7, 2.5].into_iter().enumerate() {
                let h = 2.0e-5;
                let center = bernoulli_natural_observation(row, 0.3, eta, &link)
                    .expect("center observation");
                let plus = bernoulli_natural_observation(row, 0.3, eta + h, &link)
                    .expect("plus observation");
                let minus = bernoulli_natural_observation(row, 0.3, eta - h, &link)
                    .expect("minus observation");
                let score_fd = (plus.log_likelihood - minus.log_likelihood) / (2.0 * h);
                let negative_hessian_fd = -(plus.score - minus.score) / (2.0 * h);
                let negative_hessian_derivative_fd =
                    (plus.negative_hessian - minus.negative_hessian) / (2.0 * h);
                let negative_hessian_second_derivative_fd =
                    (plus.negative_hessian_derivative - minus.negative_hessian_derivative)
                        / (2.0 * h);
                assert!(
                    (center.score - score_fd).abs() <= 2.0e-8 * (1.0 + score_fd.abs()),
                    "{} score at eta={eta}: analytic={} FD={score_fd}",
                    link.link_function().name(),
                    center.score,
                );
                assert!(
                    (center.negative_hessian - negative_hessian_fd).abs()
                        <= 3.0e-7 * (1.0 + negative_hessian_fd.abs()),
                    "{} curvature at eta={eta}: analytic={} FD={negative_hessian_fd}",
                    link.link_function().name(),
                    center.negative_hessian,
                );
                assert!(
                    (center.negative_hessian_derivative - negative_hessian_derivative_fd).abs()
                        <= 2.0e-6 * (1.0 + negative_hessian_derivative_fd.abs()),
                    "{} curvature derivative at eta={eta}: analytic={} FD={negative_hessian_derivative_fd}",
                    link.link_function().name(),
                    center.negative_hessian_derivative,
                );
                assert!(
                    (center.negative_hessian_second_derivative
                        - negative_hessian_second_derivative_fd)
                        .abs()
                        <= 2.0e-5 * (1.0 + negative_hessian_second_derivative_fd.abs()),
                    "{} curvature second derivative at eta={eta}: analytic={} FD={negative_hessian_second_derivative_fd}",
                    link.link_function().name(),
                    center.negative_hessian_second_derivative,
                );
            }
        }
    }
}
