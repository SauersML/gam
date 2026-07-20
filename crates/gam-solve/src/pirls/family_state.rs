//! Per-family working-state writers — the `(\mu, w, z, c, d)` and inverse-link
//! jet computations for each GLM response/link — together with the count/beta/
//! tweedie response validators and the special-function helpers they need.

use super::*;

#[inline]
pub(crate) fn standard_inverse_link_jet(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<MixtureInverseLinkJet, EstimationError> {
    crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta)
}

#[inline]
pub(crate) fn bernoulli_logit_geometry_from_jet(
    row: usize,
    eta: f64,
    y: f64,
    priorweight: f64,
    jet: crate::mixture_link::LogitJet5,
) -> Result<WorkingBernoulliGeometry, EstimationError> {
    if !(priorweight.is_finite() && priorweight >= 0.0) {
        return Err(EstimationError::PirlsRowGeometryUnrepresentable {
            row,
            quantity: "prior weight",
            eta,
            value: priorweight,
        });
    }
    if !eta.is_finite() {
        return Err(EstimationError::InverseLinkDomainViolation {
            link: "standard logit inverse link",
            eta,
            lower: -f64::MAX,
            upper: f64::MAX,
        });
    }
    if !(jet.mu.is_finite()
        && jet.mu >= 0.0
        && jet.mu <= 1.0
        && jet.d1.is_finite()
        && jet.d1 > 0.0
        && jet.d2.is_finite()
        && jet.d3.is_finite())
    {
        return Err(EstimationError::PirlsRowGeometryUnrepresentable {
            row,
            quantity: "canonical-logit inverse-link jet",
            eta,
            value: jet.mu,
        });
    }
    let fisher = jet.d1;
    let (weight, z, c, d) = if priorweight == 0.0 {
        (0.0, eta, 0.0, 0.0)
    } else {
        let weight = priorweight * fisher;
        let z = canonical_logit_working_response(row, eta, y, jet.d1)?;
        let c = priorweight * jet.d2;
        let d = priorweight * jet.d3;
        for (quantity, value) in [
            ("canonical-logit Fisher weight", weight),
            ("canonical-logit dW/deta", c),
            ("canonical-logit d2W/deta2", d),
        ] {
            if !value.is_finite() || (quantity.ends_with("weight") && value <= 0.0) {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row,
                    quantity,
                    eta,
                    value,
                });
            }
        }
        (weight, z, c, d)
    };
    Ok(WorkingBernoulliGeometry {
        mu: jet.mu,
        weight,
        z,
        c,
        d,
    })
}

/// Canonical-logit working response evaluated from a stable tail complement.
/// `mu` may round to exactly zero or one while `exp(-|eta|)` and the score
/// remain representable; reconstructing `y-mu` from that complement keeps the
/// exact canonical score surface alive through the declared solver tail.
#[inline]
fn canonical_logit_working_response(
    row: usize,
    eta: f64,
    y: f64,
    dmu_deta: f64,
) -> Result<f64, EstimationError> {
    if !(y.is_finite() && (0.0..=1.0).contains(&y)) {
        return Err(unrepresentable_bernoulli(row, "binomial response", eta, y));
    }
    let tail = (-eta.abs()).exp();
    let residual = if eta >= 0.0 {
        let one_minus_mu = tail / (1.0 + tail);
        if y == 1.0 {
            one_minus_mu
        } else {
            (y - 1.0) + one_minus_mu
        }
    } else {
        let mu = tail / (1.0 + tail);
        y - mu
    };
    let z = eta + residual / dmu_deta;
    if z.is_finite() {
        Ok(z)
    } else {
        Err(unrepresentable_bernoulli(
            row,
            "canonical-logit working response",
            eta,
            z,
        ))
    }
}

#[inline]
fn unrepresentable_bernoulli(
    row: usize,
    quantity: &'static str,
    eta: f64,
    value: f64,
) -> EstimationError {
    EstimationError::PirlsRowGeometryUnrepresentable {
        row,
        quantity,
        eta,
        value,
    }
}

/// Compute working IRLS geometry for a single Bernoulli observation.
///
/// The returned quantities are exact values of one smooth inverse-link surface.
/// `omm` is the stable complement `1 - mu` evaluated directly from `eta` (see
/// [`crate::mixture_link::inverse_link_complement_for_inverse_link`]): once `mu`
/// rounds to exactly `1.0` in f64 — which happens far inside the tail for the
/// noncanonical links — the naive `1.0 - mu` is a hard zero, and both the
/// Bernoulli variance `mu(1-mu)` and the working residual `y - mu` collapse. The
/// carried complement keeps them exact so a saturating cloglog/probit row can
/// proceed instead of being refused.
///
/// Once the complement itself underflows to zero the observation is predicted
/// with certainty. A *consistent* saturated row (`y == 1` at `mu == 1`, or
/// `y == 0` at `mu == 0`) is the analytic `eta -> ±inf` limit of the weight
/// formula `d1^2/(mu(1-mu)) -> 0`, so it becomes a zero-weight row — exactly the
/// output of the `priorweight == 0` branch. An *inconsistent* saturated row has
/// `-inf` log-likelihood and stays a typed refusal.
#[inline]
pub(crate) fn bernoulli_geometry_from_jet(
    row: usize,
    eta: f64,
    y: f64,
    priorweight: f64,
    jet: MixtureInverseLinkJet,
    omm: f64,
) -> Result<WorkingBernoulliGeometry, EstimationError> {
    if !(priorweight.is_finite() && priorweight >= 0.0) {
        return Err(unrepresentable_bernoulli(
            row,
            "prior weight",
            eta,
            priorweight,
        ));
    }
    // The jet may legitimately saturate (`mu` at a boundary, `d1 == 0`); reject
    // only genuinely malformed values here and decide saturation below.
    if !(jet.mu.is_finite()
        && jet.mu >= 0.0
        && jet.mu <= 1.0
        && jet.d1.is_finite()
        && jet.d1 >= 0.0
        && jet.d2.is_finite()
        && jet.d3.is_finite()
        && omm.is_finite()
        && (0.0..=1.0).contains(&omm))
    {
        return Err(unrepresentable_bernoulli(
            row,
            "inverse-link jet",
            eta,
            jet.mu,
        ));
    }
    let mu = jet.mu;
    if priorweight == 0.0 {
        return Ok(WorkingBernoulliGeometry {
            mu,
            weight: 0.0,
            z: eta,
            c: 0.0,
            d: 0.0,
        });
    }
    // Variance carried through the stable complement: `v = mu * (1 - mu)` with the
    // exact tail complement rather than `1.0 - mu`.
    let v = mu * omm;
    // Saturated row: no representable positive Fisher information. `v` has
    // underflowed to zero (the complement itself is gone) or the inverse-link
    // density `d1` vanished — the observation is predicted with certainty.
    if !(v > 0.0) || jet.d1 == 0.0 {
        // Consistent iff the response sits on the saturated boundary, so the
        // residual `y - mu` is zero in the limit.
        let consistent = if mu >= 0.5 { y == 1.0 } else { y == 0.0 };
        if consistent {
            // Analytic limit of the Fisher weight as eta -> ±inf: weight -> 0.
            return Ok(WorkingBernoulliGeometry {
                mu,
                weight: 0.0,
                z: eta,
                c: 0.0,
                d: 0.0,
            });
        }
        return Err(unrepresentable_bernoulli(
            row,
            "saturated Bernoulli row inconsistent with response",
            eta,
            y,
        ));
    }
    // Fisher weight and its eta-derivatives in a pair-then-multiply factoring that
    // stays representable through the saturating band where `d1^2` underflows
    // before the division (cloglog `eta` 5.9-6.61, probit ~24-27): each `mu'`
    // factor is paired with one variance factor before multiplying. With
    // `a = mu'/mu` and `b = mu'/(1-mu)`,
    //   W   = mu'^2 / (mu (1-mu)) = a b,
    //   a'  = mu''/mu   - a^2,          b'  = mu''/(1-mu)   + b^2,
    //   a'' = mu'''/mu   - (mu''/mu) a   - 2 a a',
    //   b'' = mu'''/(1-mu) + (mu''/(1-mu)) b + 2 b b',
    //   W' = a'b + a b',    W'' = a''b + 2 a'b' + a b''.
    // This is exactly the quotient `d/deta[mu'^2/(mu(1-mu))]` reassociated so that
    // no intermediate underflows while both `mu'` and the stable complement remain
    // representable; `v = mu * omm` above is used only for the saturation test.
    let a = jet.d1 / mu;
    let b = jet.d1 / omm;
    let fisher = a * b;
    let weight = priorweight * fisher;
    if !(fisher.is_finite() && fisher > 0.0 && weight.is_finite() && weight > 0.0) {
        return Err(unrepresentable_bernoulli(
            row,
            "Bernoulli Fisher weight",
            eta,
            weight,
        ));
    }
    let a1 = jet.d2 / mu - a * a;
    let b1 = jet.d2 / omm + b * b;
    let a2 = jet.d3 / mu - (jet.d2 / mu) * a - 2.0 * a * a1;
    let b2 = jet.d3 / omm + (jet.d2 / omm) * b + 2.0 * b * b1;
    let c = priorweight * (a1 * b + a * b1);
    let d = priorweight * (a2 * b + 2.0 * a1 * b1 + a * b2);
    if !c.is_finite() {
        return Err(unrepresentable_bernoulli(row, "dW/deta", eta, c));
    }
    if !d.is_finite() {
        return Err(unrepresentable_bernoulli(row, "d2W/deta2", eta, d));
    }
    Ok(WorkingBernoulliGeometry {
        mu,
        weight,
        z: bernoulli_exact_working_response(row, eta, y, mu, omm, jet.d1)?,
        c,
        d,
    })
}

/// Working response `z = eta + (y - mu)/mu'` with the residual reconstructed
/// from the stable complement to avoid tail cancellation. Near the upper
/// boundary (`mu -> 1`) `y - mu = (y - 1) + (1 - mu) = (y - 1) + omm`; near the
/// lower boundary (`mu -> 0`) `mu` is itself the small, exactly-evaluated value,
/// so `y - mu` is already accurate.
#[inline]
pub(crate) fn bernoulli_exact_working_response(
    row: usize,
    eta: f64,
    y: f64,
    mu: f64,
    omm: f64,
    dmu_deta: f64,
) -> Result<f64, EstimationError> {
    let residual = if mu >= 0.5 { (y - 1.0) + omm } else { y - mu };
    let z = eta + residual / dmu_deta;
    if z.is_finite() {
        Ok(z)
    } else {
        Err(unrepresentable_bernoulli(
            row,
            "Bernoulli working response",
            eta,
            z,
        ))
    }
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
) -> Result<(), EstimationError> {
    for i in 0..eta.len() {
        if !eta[i].is_finite() {
            return Err(EstimationError::InverseLinkDomainViolation {
                link: "standard identity inverse link",
                eta: eta[i],
                lower: -f64::MAX,
                upper: f64::MAX,
            });
        }
        if !(priorweights[i].is_finite() && priorweights[i] >= 0.0) {
            return Err(unrepresentable_bernoulli(
                i,
                "prior weight",
                eta[i],
                priorweights[i],
            ));
        }
        if priorweights[i] > 0.0 && !y[i].is_finite() {
            return Err(unrepresentable_bernoulli(
                i,
                "identity-link response",
                eta[i],
                y[i],
            ));
        }
    }
    ndarray::Zip::indexed(mu)
        .and(weights)
        .and(z)
        .par_for_each(|i, mu_i, weight_i, z_i| {
            *mu_i = eta[i];
            *weight_i = priorweights[i];
            *z_i = if priorweights[i] == 0.0 { eta[i] } else { y[i] };
        });
    if let Some(derivs) = derivatives {
        derivs.c.fill(0.0);
        derivs.d.fill(0.0);
        derivs.dmu_deta.fill(1.0);
        derivs.d2mu_deta2.fill(0.0);
        derivs.d3mu_deta3.fill(0.0);
    }
    Ok(())
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
) -> Result<(), EstimationError> {
    validate_count_responses(&y, &priorweights, "Poisson")?;
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::PoissonIdentity,
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: 1.0,
                d_ratio: 1.0,
            },
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    )
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
) -> Result<(), EstimationError> {
    if !(shape.is_finite() && shape > 0.0) {
        crate::bail_invalid_estim!("Gamma shape must be finite and > 0; got {shape}");
    }
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::Constant { factor: shape },
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: 0.0,
                d_ratio: 0.0,
            },
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    )
}

#[inline]
pub(crate) fn valid_negbin_theta(theta: f64) -> bool {
    theta.is_finite() && theta > 0.0
}

#[inline]
pub(crate) fn valid_count_response(y: f64) -> bool {
    y.is_finite() && y >= 0.0 && y == y.round()
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
    q: f64,
    q_prime: f64,
    q_double_prime: f64,
    a: f64,
    b: f64,
    trigamma_sum: f64,
) -> (f64, f64) {
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

#[derive(Clone, Copy)]
pub(crate) struct ExactBetaLogitRow {
    pub(crate) mu: f64,
    pub(crate) weight: f64,
    pub(crate) z: f64,
    pub(crate) c: f64,
    pub(crate) d: f64,
    pub(crate) dmu: f64,
    pub(crate) d2mu: f64,
    pub(crate) d3mu: f64,
}

#[inline]
pub(crate) fn exact_beta_logit_row(
    row: usize,
    eta: f64,
    y: Option<f64>,
    prior_weight: f64,
    phi: f64,
) -> Result<ExactBetaLogitRow, EstimationError> {
    if !(prior_weight.is_finite() && prior_weight >= 0.0) {
        return Err(unrepresentable_bernoulli(
            row,
            "prior weight",
            eta,
            prior_weight,
        ));
    }
    if !eta.is_finite() {
        return Err(EstimationError::InverseLinkDomainViolation {
            link: "standard logit inverse link",
            eta,
            lower: -f64::MAX,
            upper: f64::MAX,
        });
    }
    let jet = logit_inverse_link_jet5(eta);
    if !(jet.mu.is_finite()
        && jet.mu > 0.0
        && jet.mu < 1.0
        && jet.d1.is_finite()
        && jet.d1 > 0.0
        && jet.d2.is_finite()
        && jet.d3.is_finite())
    {
        return Err(unrepresentable_bernoulli(
            row,
            "beta-logit inverse-link jet",
            eta,
            jet.mu,
        ));
    }
    if prior_weight == 0.0 {
        return Ok(ExactBetaLogitRow {
            mu: jet.mu,
            weight: 0.0,
            z: eta,
            c: 0.0,
            d: 0.0,
            dmu: jet.d1,
            d2mu: jet.d2,
            d3mu: jet.d3,
        });
    }

    let mu = jet.mu;
    let q = jet.d1;
    let a = mu * phi;
    let b = (1.0 - mu) * phi;
    if !(a.is_finite() && a > 0.0) {
        return Err(unrepresentable_bernoulli(row, "beta shape a", eta, a));
    }
    if !(b.is_finite() && b > 0.0) {
        return Err(unrepresentable_bernoulli(row, "beta shape b", eta, b));
    }
    let trigamma_sum = trigamma(a) + trigamma(b);
    let info_mu = phi * phi * trigamma_sum;
    if !(info_mu.is_finite() && info_mu > 0.0) {
        return Err(unrepresentable_bernoulli(
            row,
            "beta mean information",
            eta,
            info_mu,
        ));
    }
    let weight = prior_weight * q * q * info_mu;
    if !(weight.is_finite() && weight > 0.0) {
        return Err(unrepresentable_bernoulli(
            row,
            "beta Fisher weight",
            eta,
            weight,
        ));
    }
    let z = if let Some(y) = y {
        let score_mu = phi * (digamma(b) - digamma(a) + y.ln() - (1.0 - y).ln());
        let score_eta_denominator = q * info_mu;
        let z = eta + score_mu / score_eta_denominator;
        if !z.is_finite() {
            return Err(unrepresentable_bernoulli(
                row,
                "beta working response",
                eta,
                z,
            ));
        }
        z
    } else {
        eta
    };
    let (c, d) = beta_logit_working_curvature_eta_derivatives(
        prior_weight,
        phi,
        q,
        jet.d2,
        jet.d3,
        a,
        b,
        trigamma_sum,
    );
    if !c.is_finite() {
        return Err(unrepresentable_bernoulli(row, "beta dW/deta", eta, c));
    }
    if !d.is_finite() {
        return Err(unrepresentable_bernoulli(row, "beta d2W/deta2", eta, d));
    }
    Ok(ExactBetaLogitRow {
        mu,
        weight,
        z,
        c,
        d,
        dmu: jet.d1,
        d2mu: jet.d2,
        d3mu: jet.d3,
    })
}

/// Working state for Tweedie with a log link.
///
/// With `mu = exp(eta)`, `V(mu) = phi * mu^p`, and `g'(mu) = 1 / mu`, the Fisher
/// working weight is `mu^(2-p) / phi`, scaled by prior weight. Parameter ranges,
/// responses, and every represented row quantity are validated up front.
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
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    )
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
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    )
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
    let certified: Vec<Result<ExactBetaLogitRow, EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|i| exact_beta_logit_row(i, eta[i], Some(y[i]), priorweights[i], phi))
        .collect();
    let certified: Vec<ExactBetaLogitRow> = certified.into_iter().collect::<Result<_, _>>()?;
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
            .zip(certified.par_iter())
            .for_each(
                |((((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o), state)| {
                    *mu_o = state.mu;
                    *w_o = state.weight;
                    *z_o = state.z;
                    *dmu_o = state.dmu;
                    *d2_o = state.d2mu;
                    *d3_o = state.d3mu;
                    *c_o = state.c;
                    *d_o = state.d;
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
            .zip(certified.par_iter())
            .for_each(|(((mu_o, w_o), z_o), state)| {
                *mu_o = state.mu;
                *w_o = state.weight;
                *z_o = state.z;
            });
    }
    Ok(())
}
