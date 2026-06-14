//! Per-family working-state writers — the `(\mu, w, z, c, d)` and inverse-link
//! jet computations for each GLM response/link — together with the count/beta/
//! tweedie response validators and the special-function helpers they need.

use super::*;

#[inline]
pub(super) fn standard_inverse_link_jet(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<MixtureInverseLinkJet, EstimationError> {
    crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta)
}


#[inline]
pub(crate) fn bernoulli_logit_geometry_from_jet(
    eta_raw: f64,
    eta_used: f64,
    y: f64,
    priorweight: f64,
    jet: crate::mixture_link::LogitJet5,
    zero_on_nonsmooth: bool,
) -> WorkingBernoulliGeometry {
    let fisher = jet.d1;
    let nonsmooth = eta_raw != eta_used || !fisher.is_finite() || fisher < 0.0;
    let (c, d) = if nonsmooth && zero_on_nonsmooth {
        (0.0, 0.0)
    } else {
        (priorweight * jet.d2, priorweight * jet.d3)
    };
    WorkingBernoulliGeometry {
        mu: jet.mu,
        weight: priorweight * fisher,
        z: bernoulli_exact_working_response(eta_used, y, jet.mu, jet.d1),
        c,
        d,
    }
}


/// Compute working IRLS geometry for a single Bernoulli observation.
///
/// This helper returns the exact statistical working state. It does not floor
/// the Fisher mass or the working response for solver conditioning; doing so
/// would change the model rather than just the Newton system.
///
/// The weight returned is the **Fisher** (expected information) weight
/// W_F = h'(η)² / V(μ). The c and d fields are likewise the Fisher
/// derivatives c_F = dW_F/dη and d_F = d²W_F/dη².
///
/// NOTE: For non-canonical links (probit, cloglog, SAS, mixture), the
/// observed weight differs:
///   W_obs = W_F − (y−μ) · B,  B = (h''V − h'²V') / V²
/// The observed c/d include residual-dependent corrections. PIRLS keeps
/// these Fisher carriers for the score-side RHS `X'W(z-eta) - S beta`,
/// while the Newton/Laplace Hessian side may switch to the observed,
/// clamped curvature surface. The accepted Hessian-side c/d arrays are
/// stored separately in `PirlsResult::solve_c_array` / `solve_d_array`
/// and consumed directly by the REML/LAML exact-derivative code.
#[inline]
pub(crate) fn bernoulli_geometry_from_jet(
    eta_raw: f64,
    eta_used: f64,
    y: f64,
    priorweight: f64,
    jet: MixtureInverseLinkJet,
) -> WorkingBernoulliGeometry {
    let mu = jet.mu;
    let v = mu * (1.0 - mu);
    let n0 = jet.d1 * jet.d1;
    let fisher = if v.is_finite() && v > 0.0 {
        n0 / v
    } else {
        0.0
    };
    let nonsmooth =
        eta_raw != eta_used || !v.is_finite() || v <= 0.0 || !fisher.is_finite() || fisher < 0.0;
    let (c, d) = if nonsmooth {
        (0.0, 0.0)
    } else {
        let v1 = jet.d1 * (1.0 - 2.0 * mu);
        let v2 = jet.d2 * (1.0 - 2.0 * mu) - 2.0 * jet.d1 * jet.d1;
        let n1 = 2.0 * jet.d1 * jet.d2;
        let n2 = 2.0 * (jet.d2 * jet.d2 + jet.d1 * jet.d3);
        let numer1 = n1 * v - n0 * v1;
        let c = priorweight * numer1 / (v * v);
        let d = priorweight * ((n2 * v - n0 * v2) / (v * v) - 2.0 * numer1 * v1 / (v * v * v));
        (c, d)
    };
    WorkingBernoulliGeometry {
        mu,
        weight: priorweight * fisher,
        z: bernoulli_exact_working_response(eta_used, y, mu, jet.d1),
        c,
        d,
    }
}


#[inline]
pub(crate) fn bernoulli_exact_working_response(eta: f64, y: f64, mu: f64, dmu_deta: f64) -> f64 {
    // Preserve the exact IRLS score carrier W(z-eta) = y-mu whenever the link
    // jet is finite. Numerical conditioning belongs in the linear solve, not in
    // the Bernoulli likelihood geometry.
    if dmu_deta.is_finite() && dmu_deta > 0.0 {
        let delta = (y - mu) / dmu_deta;
        if delta.is_finite() {
            return eta + delta;
        }
    }
    eta
}


#[inline]
pub(crate) fn write_identityworking_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    mu.assign(eta);
    weights.assign(&priorweights);
    z.assign(&y);
    if let Some(derivs) = derivatives {
        derivs.c.fill(0.0);
        derivs.d.fill(0.0);
        derivs.dmu_deta.fill(1.0);
        derivs.d2mu_deta2.fill(0.0);
        derivs.d3mu_deta3.fill(0.0);
    }
}


/// Working state for Poisson with a log link.
///
/// `V(mu) = mu`, so the Fisher weight is `prior * mu` and the canonical-link
/// curvature buffers both equal the working weight.
#[inline]
pub(crate) fn write_poisson_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::PoissonIdentity,
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: 1.0,
                d_ratio: 1.0,
            },
            floor_weight: true,
            zero_mu_jet_on_clamp: false,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
}


/// Working state for Gamma(shape = k) with a log link.
///
/// With `mu = exp(eta)` and `V(mu) = mu^2`, the Fisher weight is the
/// prior/sample weight scaled by the fixed Gamma shape, independent of `eta`;
/// the weight is therefore written unfloored and the curvature buffers vanish.
#[inline]
pub(crate) fn write_gamma_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    shape: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::Constant { factor: shape },
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: 0.0,
                d_ratio: 0.0,
            },
            floor_weight: false,
            zero_mu_jet_on_clamp: false,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
}


pub const BETA_MU_EPS: f64 = 1.0e-12;


#[inline]
pub(crate) fn tweedie_log_weight_mu_power(mu: f64, p: f64) -> f64 {
    // Match the 1e-300 MIN_DEVIANCE floor used by the REML deviance path:
    // smaller positive mu values are below a non-degenerate f64 likelihood
    // contribution, but flooring here keeps mu^(2-p) away from underflow.
    mu.max(1.0e-300).powf(2.0 - p)
}


#[inline]
pub(crate) fn valid_negbin_theta(theta: f64) -> bool {
    theta.is_finite() && theta > 0.0
}


#[inline]
pub(crate) fn valid_count_response(y: f64) -> bool {
    y.is_finite() && y >= 0.0 && (y - y.round()).abs() <= 1e-9
}


pub(crate) fn validate_count_responses(
    y: &ArrayView1<'_, f64>,
    priorweights: &ArrayView1<'_, f64>,
    family: &str,
) -> Result<(), EstimationError> {
    for (i, (&yi, &wi)) in y.iter().zip(priorweights.iter()).enumerate() {
        if wi > 0.0 && !valid_count_response(yi) {
            crate::bail_invalid_estim!(
                "{family} response must be a finite non-negative integer at positive-weight row {i}; got {yi}"
            );
        }
    }
    Ok(())
}


#[inline]
pub(crate) fn valid_beta_phi(phi: f64) -> bool {
    phi.is_finite() && phi > 0.0
}


#[inline]
pub(crate) fn valid_beta_response(y: f64) -> bool {
    y.is_finite() && y > 0.0 && y < 1.0
}


pub(crate) fn validate_beta_responses(
    y: &ArrayView1<'_, f64>,
    priorweights: &ArrayView1<'_, f64>,
) -> Result<(), EstimationError> {
    for (i, (&yi, &wi)) in y.iter().zip(priorweights.iter()).enumerate() {
        if wi > 0.0 && !valid_beta_response(yi) {
            crate::bail_invalid_estim!(
                "beta-regression response must be finite and strictly inside (0, 1) at positive-weight row {i}; got {yi}"
            );
        }
    }
    Ok(())
}


#[inline]
pub(crate) fn valid_tweedie_response(y: f64) -> bool {
    y.is_finite() && y >= 0.0
}


pub(crate) fn validate_tweedie_responses(
    y: &ArrayView1<'_, f64>,
    priorweights: &ArrayView1<'_, f64>,
) -> Result<(), EstimationError> {
    for (i, (&yi, &wi)) in y.iter().zip(priorweights.iter()).enumerate() {
        if wi > 0.0 && !valid_tweedie_response(yi) {
            crate::bail_invalid_estim!(
                "Tweedie response must be finite and non-negative at positive-weight row {i}; got {yi}"
            );
        }
    }
    Ok(())
}


#[inline]
pub(crate) fn safe_beta_mu(mu: f64) -> f64 {
    mu.clamp(BETA_MU_EPS, 1.0 - BETA_MU_EPS)
}


#[inline]
pub(crate) fn trigamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
        + inv2 * inv2 * inv2 * inv / 42.0
        - inv2 * inv2 * inv2 * inv2 * inv / 30.0
}


#[inline]
pub(crate) fn polygamma2(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc -= 2.0 / (x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    acc - inv2 - inv3 - 0.5 * inv2 * inv2 + inv3 * inv3 / 6.0 - inv2 * inv3 * inv3 / 6.0
        + 0.3 * inv2 * inv2 * inv3 * inv3
        - 5.0 * inv2 * inv2 * inv2 * inv3 * inv3 / 6.0
}


#[inline]
pub(crate) fn polygamma3(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc += 6.0 / (x * x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv4 = inv2 * inv2;
    acc + 2.0 * inv3 + 3.0 * inv4 + 2.0 * inv4 * inv - inv4 * inv3 + 4.0 * inv4 * inv3 * inv2 / 3.0
        - 3.0 * inv4 * inv3 * inv4
        + 10.0 * inv4 * inv4 * inv4 * inv
}


#[inline]
pub(crate) fn beta_logit_working_curvature_eta_derivatives(
    prior_weight: f64,
    phi: f64,
    mu: f64,
    q: f64,
    a: f64,
    b: f64,
    trigamma_sum: f64,
) -> (f64, f64) {
    let q_prime = q * (1.0 - 2.0 * mu);
    let q_double_prime = q * (1.0 - 2.0 * mu) * (1.0 - 2.0 * mu) - 2.0 * q * q;
    let psi2_diff = polygamma2(a) - polygamma2(b);
    let psi3_sum = polygamma3(a) + polygamma3(b);
    let phi_sq = phi * phi;
    let q_sq = q * q;
    let c = prior_weight * phi_sq * (2.0 * q * q_prime * trigamma_sum + q_sq * phi * q * psi2_diff);
    let d = prior_weight
        * phi_sq
        * (2.0 * (q_prime * q_prime + q * q_double_prime) * trigamma_sum
            + 4.0 * q * q_prime * phi * q * psi2_diff
            + q_sq * (phi * q_prime * psi2_diff + phi_sq * q_sq * psi3_sum));
    (c, d)
}


/// Working state for Tweedie with a log link.
///
/// With `mu = exp(eta)`, `V(mu) = phi * mu^p`, and `g'(mu) = 1 / mu`, the Fisher
/// working weight is `mu^(2-p) / phi`, scaled by prior weight. The `mu`-jet must
/// be zeroed when `eta` is clamped because the fractional power makes the local
/// jet unreliable there. Parameter ranges and responses are validated up front.
#[inline]
pub(crate) fn write_tweedie_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    p: f64,
    phi: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    if !is_valid_tweedie_power(p) {
        crate::bail_invalid_estim!(
            "Tweedie variance power must be finite and strictly between 1 and 2; got {p}",
            p = p
        );
    }
    if !(phi.is_finite() && phi > 0.0) {
        crate::bail_invalid_estim!(
            "Tweedie dispersion phi must be finite and > 0; got {phi}",
            phi = phi
        );
    }
    validate_tweedie_responses(&y, &priorweights)?;
    let exponent = 2.0 - p;
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::TweediePower { p, phi },
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: exponent,
                d_ratio: exponent * exponent,
            },
            floor_weight: true,
            zero_mu_jet_on_clamp: true,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
    Ok(())
}


/// Working state for NB(mu, theta) with a log link and fixed theta.
///
/// The size parameter is treated as a fixed hyperparameter for this GLM stack;
/// no theta profiling or REML update is performed here. The Fisher weight is
/// `mu * theta / (theta + mu)`, written in the numerically-stable branch form
/// that avoids cancellation for very small or very large `mu / theta`.
#[inline]
pub(crate) fn write_negative_binomial_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    theta: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    if !valid_negbin_theta(theta) {
        crate::bail_invalid_estim!(
            "negative-binomial theta must be finite and > 0; got {theta}",
            theta = theta
        );
    }
    validate_count_responses(&y, &priorweights, "negative-binomial")?;
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::NegativeBinomial { theta },
            curvature: log_link_working_state::WorkingCurvature::NegativeBinomial { theta },
            floor_weight: true,
            zero_mu_jet_on_clamp: false,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
    Ok(())
}


/// Working state for Beta(mu * phi, (1 - mu) * phi) with a logit link.
#[inline]
pub(crate) fn write_beta_logit_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    phi: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    if !valid_beta_phi(phi) {
        crate::bail_invalid_estim!("beta-regression phi must be finite and > 0; got {phi}");
    }
    validate_beta_responses(&y, &priorweights)?;
    if let Some(mut derivs) = derivatives {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        let WorkingDerivSlices {
            c: c_s,
            d: d_s,
            dmu: dmu_s,
            d2: d2_s,
            d3: d3_s,
        } = working_deriv_slices(&mut derivs);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
                    let jet = logit_inverse_link_jet5(eta_i);
                    let mu_i = safe_beta_mu(jet.mu);
                    let q = (mu_i * (1.0 - mu_i)).max(BETA_MU_EPS);
                    let yi = y[i];
                    let a = (mu_i * phi).max(BETA_MU_EPS);
                    let b = ((1.0 - mu_i) * phi).max(BETA_MU_EPS);
                    let score_mu = phi * (digamma(b) - digamma(a) + yi.ln() - (1.0 - yi).ln());
                    let trigamma_sum = trigamma(a) + trigamma(b);
                    let info_mu = phi * phi * trigamma_sum;
                    let prior_weight = priorweights[i].max(0.0);
                    let raw_weight = prior_weight * q * q * info_mu;
                    let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    *mu_o = mu_i;
                    *w_o = if raw_weight > 0.0 {
                        raw_weight.max(MIN_WEIGHT)
                    } else {
                        0.0
                    };
                    *z_o = eta_i + score_mu / (q * info_mu).max(MIN_WEIGHT);
                    *dmu_o = q;
                    *d2_o = q * (1.0 - 2.0 * mu_i);
                    *d3_o = q * (1.0 - 6.0 * q);
                    if floor_active || eta_raw != eta_i {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        let (c_i, d_i) = beta_logit_working_curvature_eta_derivatives(
                            prior_weight,
                            phi,
                            mu_i,
                            q,
                            a,
                            b,
                            trigamma_sum,
                        );
                        *c_o = c_i;
                        *d_o = d_i;
                    }
                },
            );
    } else {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP);
                let jet = logit_inverse_link_jet5(eta_i);
                let mu_i = safe_beta_mu(jet.mu);
                let q = (mu_i * (1.0 - mu_i)).max(BETA_MU_EPS);
                let yi = y[i];
                let a = (mu_i * phi).max(BETA_MU_EPS);
                let b = ((1.0 - mu_i) * phi).max(BETA_MU_EPS);
                let score_mu = phi * (digamma(b) - digamma(a) + yi.ln() - (1.0 - yi).ln());
                let info_mu = phi * phi * (trigamma(a) + trigamma(b));
                let raw_weight = priorweights[i].max(0.0) * q * q * info_mu;
                *mu_o = mu_i;
                *w_o = if raw_weight > 0.0 {
                    raw_weight.max(MIN_WEIGHT)
                } else {
                    0.0
                };
                *z_o = eta_i + score_mu / (q * info_mu).max(MIN_WEIGHT);
            });
    }
    Ok(())
}
