//! #2902 row 34: the beta-logistic binomial row geometry in log space.
//!
//! The beta-logistic link is `μ = K(x)` with `K(x) = I_u(a, b)`, `u = logistic(x)`,
//! at the standardized latent argument `x = E Z + s·η`. Its deviance row went
//! through the mean, and the mean saturates while the row is an ordinary number:
//! - The #2685 fixture's certified shapes are `(a, b) = (0.01494, 0.00598)`. There
//!   the latent kernel at `x = 42.77` has `u` rounding to `1.0`, so `μ` is exactly
//!   `1.0` and the row was refused ("inverse-link value/derivative … produced 1.0"),
//!   while `1 − μ = 0.5532`.
//! - The same collapse is reachable through a link state. At `(ε, log δ) =
//!   (0.55, 0.6)` the shapes are `(1.0513, 3.1582)`, `E Z = −1.4793` and
//!   `s = 0.7603`. At `η = 20` the mean rounds to `1.0` while `ln(1 − μ) = −43.26`.
//!   At `η = 350` the complement itself leaves `f64` (`1.2e-363`) while
//!   `ln(1 − μ) = −835.62`. The mirror state `(−0.55, 0.6)` underflows the mean to
//!   exactly `0.0` at `η = −350`.
//!
//! References are mpmath at 60 digits (`mp.betainc(a, b, 0, x, regularized=True)`,
//! `mp.digamma`, `mp.polygamma` and `mp.loggamma`), not another call into this code.
//! The positive control pins that the mean route really collapses at the reference
//! points, so the reference tests cannot pass vacuously.

use super::*;
use approx::assert_relative_eq;
use gam_problem::{GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, SasLinkState};

/// One mpmath reference point.
struct Reference {
    eta: f64,
    log_mu: f64,
    log_one_minus_mu: f64,
    log_d1: f64,
}

/// The #2685 parametric fixture's certified shapes with the shape bound lifted
/// (job 1144648 arm B), and the η its assembly refused at.
const CERTIFIED_A: f64 = 0.014935714318017365;
const CERTIFIED_B: f64 = 0.0059760228950059434;
const CERTIFIED: Reference = Reference {
    eta: 42.769330449746526,
    log_mu: -0.80568690922063519,
    log_one_minus_mu: -0.59200119532110735,
    log_d1: -5.7120011953211074,
};
const CERTIFIED_ONE_MINUS_MU: f64 = 0.55321907680603913;

/// `(ε, log δ) = (0.55, 0.6)`: shapes `(1.0513, 3.1582)`, `E Z = −1.4793`,
/// `s = 0.7603`, upper tail.
const UPPER: [Reference; 2] = [
    Reference {
        eta: 20.0,
        log_mu: -1.6375112258083487e-19,
        log_one_minus_mu: -43.255939222927616,
        log_d1: -42.380005795446818,
    },
    Reference {
        eta: 350.0,
        log_mu: 0.0,
        log_one_minus_mu: -835.62493278338421,
        log_d1: -834.74899824915706,
    },
];

/// `(ε, log δ) = (−0.55, 0.6)`: shapes `(3.1582, 1.0513)`, `E Z = 1.4793`,
/// `s = 0.7603`, lower tail.
const LOWER: [Reference; 2] = [
    Reference {
        eta: -20.0,
        log_mu: -43.255939222927616,
        log_one_minus_mu: -1.6375112258083487e-19,
        log_d1: -42.380005795446818,
    },
    Reference {
        eta: -350.0,
        log_mu: -835.62493278338421,
        log_one_minus_mu: 0.0,
        log_d1: -834.74899824915706,
    },
];

/// Every value is a sum of log terms no larger than `|b·ln(1 − u)| ≈ 836`, each
/// carrying a few ulps (statrs `ln_beta`, the ascending series, `softplus`, the
/// polygamma stack behind `E Z` and `s`), so the relative error stays below
/// `836 · 64 · ε ≈ 1.2e-11` of the largest term and far below that of these
/// results; `1e-12` leaves the measured agreement room while failing any wrong
/// tail by many orders.
const REFERENCE_RELATIVE: f64 = 1.0e-12;

fn beta_logistic(epsilon: f64, log_delta: f64) -> InverseLink {
    InverseLink::BetaLogistic(SasLinkState {
        epsilon,
        log_delta,
        delta: log_delta.exp(),
    })
}

fn row(inverse_link: &InverseLink, y: f64, eta: f64) -> Result<DevianceEtaRow, EstimationError> {
    deviance_eta_row_with_log_measure_scale(
        0,
        y,
        eta,
        &GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            inverse_link.clone(),
        )),
        inverse_link,
        1.0,
        0.0,
    )
}

fn assert_reference(label: &str, got: (f64, f64, f64), reference: &Reference) {
    let (log_mu, log_one_minus_mu, log_d1) = got;
    assert_relative_eq!(log_mu, reference.log_mu, max_relative = REFERENCE_RELATIVE);
    assert_relative_eq!(
        log_one_minus_mu,
        reference.log_one_minus_mu,
        max_relative = REFERENCE_RELATIVE
    );
    assert_relative_eq!(log_d1, reference.log_d1, max_relative = REFERENCE_RELATIVE);
    assert!(
        log_mu <= 0.0 && log_one_minus_mu <= 0.0,
        "{label} at eta={}: log-probabilities must be non-positive, got ({log_mu}, \
         {log_one_minus_mu})",
        reference.eta
    );
}

/// Positive control: the mean route has nothing to express at the reference
/// points. `u` rounds to `1.0` at the certified η, and the mean itself rounds to
/// exactly `1.0` or `0.0` at the reachable ones.
#[test]
fn the_mean_route_collapses_at_the_reference_points_2902() {
    assert_eq!(
        gam_math::special::logistic(CERTIFIED.eta),
        1.0,
        "logistic(42.77) must round to 1.0, or this fixture no longer reaches the tail"
    );
    let upper = beta_logistic(0.55, 0.6);
    let (mu, _) = crate::mixture_link::inverse_link_mu_d1_for_inverse_link(&upper, UPPER[0].eta)
        .expect("finite beta-logistic eta");
    assert_eq!(mu, 1.0, "the upper-tail mean must round to exactly 1.0 at eta=20");
    let lower = beta_logistic(-0.55, 0.6);
    let (mu, _) = crate::mixture_link::inverse_link_mu_d1_for_inverse_link(&lower, LOWER[1].eta)
        .expect("finite beta-logistic eta");
    assert_eq!(mu, 0.0, "the lower-tail mean must underflow to exactly 0.0 at eta=-350");
}

#[test]
fn log_probabilities_match_mpmath_in_both_saturating_tails_2902() {
    let certified = crate::mixture_link::beta_logistic_latent_log_probabilities(
        CERTIFIED.eta,
        CERTIFIED_A,
        CERTIFIED_B,
    );
    assert_reference("certified #2685 shapes", certified, &CERTIFIED);
    assert_relative_eq!(
        certified.1.exp(),
        CERTIFIED_ONE_MINUS_MU,
        max_relative = REFERENCE_RELATIVE
    );
    for reference in &UPPER {
        let got =
            crate::mixture_link::beta_logistic_binomial_log_probabilities(reference.eta, 0.6, 0.55);
        assert_reference("upper tail (0.55, 0.6)", got, reference);
    }
    for reference in &LOWER {
        let got = crate::mixture_link::beta_logistic_binomial_log_probabilities(
            reference.eta,
            0.6,
            -0.55,
        );
        assert_reference("lower tail (-0.55, 0.6)", got, reference);
    }
}

/// A saturated row is an ordinary row. At a 0/1 response the Bernoulli
/// half-deviance is the cross-entropy, `−ln μ` or `−ln(1 − μ)`, and before the
/// log-space geometry each of these rows was refused.
#[test]
fn saturated_rows_are_accepted_at_their_cross_entropy_2902() {
    for (link, references) in [
        (beta_logistic(0.55, 0.6), &UPPER),
        (beta_logistic(-0.55, 0.6), &LOWER),
    ] {
        for reference in references.iter() {
            let one = row(&link, 1.0, reference.eta).expect("beta-logistic row, y=1");
            assert_relative_eq!(
                one.half_deviance,
                -reference.log_mu,
                max_relative = REFERENCE_RELATIVE
            );
            let zero = row(&link, 0.0, reference.eta).expect("beta-logistic row, y=0");
            assert_relative_eq!(
                zero.half_deviance,
                -reference.log_one_minus_mu,
                max_relative = REFERENCE_RELATIVE
            );
            assert_relative_eq!(
                one.eta_score,
                -(reference.log_d1 - reference.log_mu).exp(),
                max_relative = REFERENCE_RELATIVE
            );
            assert_relative_eq!(
                zero.eta_score,
                (reference.log_d1 - reference.log_one_minus_mu).exp(),
                max_relative = REFERENCE_RELATIVE
            );
        }
    }
}

/// The score channel is the derivative of the value channel the same row reports,
/// inside the saturating band as well as the interior. The central difference's
/// truncation is `O(h²·V‴/V′)` and its cancellation `ε·|V|/h`, both below `1e-9` of
/// the score at `h = 1e-6·max(1, |η|)` for these rows.
#[test]
fn saturated_eta_score_matches_a_central_difference_of_its_value_2902() {
    for (epsilon, log_delta) in [(0.55, 0.6), (-0.55, 0.6), (0.35, -0.9)] {
        let link = beta_logistic(epsilon, log_delta);
        for y in [0.0, 1.0, 0.25] {
            for eta in [-45.0_f64, -20.0, -4.0, 0.42, 3.0, 20.0, 45.0] {
                let h = 1.0e-6 * eta.abs().max(1.0);
                let centre = row(&link, y, eta).expect("beta-logistic centre row");
                let plus = row(&link, y, eta + h).expect("beta-logistic plus row").half_deviance;
                let minus = row(&link, y, eta - h).expect("beta-logistic minus row").half_deviance;
                let finite_difference = (plus - minus) / (2.0 * h);
                assert!(
                    centre.half_deviance.is_finite() && centre.eta_score.is_finite(),
                    "beta-logistic row must stay representable at eta={eta}, y={y} \
                     (eps={epsilon}, log_delta={log_delta})"
                );
                assert_relative_eq!(
                    centre.eta_score,
                    finite_difference,
                    max_relative = 1.0e-6,
                    epsilon = 1.0e-9
                );
            }
        }
    }
}
