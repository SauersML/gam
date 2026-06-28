//! Shared test-only dispersion-family row-NLL oracles.
//!
//! These generic-over-[`gam_math::jet_scalar::JetScalar`] programs are the dense
//! reference NLLs that the dispersion-family tests (`dispersion_family::tests`)
//! and the gamlss behaviour tests (`super::tests`) both pin the pruned
//! production towers against. They live in this gamlss-level private
//! `#[cfg(test)]` module — reachable from both descendant test modules — rather
//! than as `#[cfg(test)]` items at production module scope.

use statrs::function::gamma::ln_gamma;

/// Test-oracle NB2 row NLL over a generic [`JetScalar<2>`], seeded on the
/// natural parameters `(μ, θ)`.
///
/// [`JetScalar<2>`]: gam_math::jet_scalar::JetScalar
#[inline]
pub(crate) fn dispersion_nb_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
    yi: f64,
    mu_value: f64,
    theta_value: f64,
    wi: f64,
) -> S {
    let mu = S::variable(mu_value, 0);
    let theta = S::variable(theta_value, 1);
    let tpm = theta.add(&mu);
    // (theta + yi).ln_gamma() - theta.ln_gamma() - ln_gamma(yi+1)
    //   + theta*theta.ln() - theta*tpm.ln() + mu.ln()*yi - tpm.ln()*yi
    let loglik = theta
        .add(&S::constant(yi))
        .ln_gamma()
        .sub(&theta.ln_gamma())
        .sub(&S::constant(ln_gamma(yi + 1.0)))
        .add(&theta.mul(&theta.ln()))
        .sub(&theta.mul(&tpm.ln()))
        .add(&mu.ln().scale(yi))
        .sub(&tpm.ln().scale(yi));
    loglik.scale(-wi)
}

/// Test-oracle Gamma row NLL over a generic [`JetScalar<2>`], seeded on `(μ, ν)`.
///
/// [`JetScalar<2>`]: gam_math::jet_scalar::JetScalar
#[inline]
pub(crate) fn dispersion_gamma_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
    yi: f64,
    y_pos: f64,
    mu_value: f64,
    nu_value: f64,
    wi: f64,
) -> S {
    let mu = S::variable(mu_value, 0);
    let nu = S::variable(nu_value, 1);
    // nu*nu.ln() - nu*mu.ln() - nu.ln_gamma() + (nu-1)*y_pos.ln() - nu*(mu.recip()*yi)
    let loglik = nu
        .mul(&nu.ln())
        .sub(&nu.mul(&mu.ln()))
        .sub(&nu.ln_gamma())
        .add(&nu.sub(&S::constant(1.0)).scale(y_pos.ln()))
        .sub(&nu.mul(&mu.recip().scale(yi)));
    loglik.scale(-wi)
}

/// Test-oracle Beta row NLL over a generic [`JetScalar<2>`], seeded on `(μ, φ)`.
///
/// [`JetScalar<2>`]: gam_math::jet_scalar::JetScalar
#[inline]
pub(crate) fn dispersion_beta_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
    yi: f64,
    mu_value: f64,
    phi_value: f64,
    wi: f64,
) -> S {
    let mu = S::variable(mu_value, 0);
    let phi = S::variable(phi_value, 1);
    let one_minus_mu = S::constant(1.0).sub(&mu);
    let yc = yi.clamp(1e-12, 1.0 - 1e-12);
    let a = mu.mul(&phi);
    let b = one_minus_mu.mul(&phi);
    // phi.ln_gamma() - a.ln_gamma() - b.ln_gamma()
    //   + (a-1)*yc.ln() + (b-1)*(1-yc).ln()
    let loglik = phi
        .ln_gamma()
        .sub(&a.ln_gamma())
        .sub(&b.ln_gamma())
        .add(&a.sub(&S::constant(1.0)).scale(yc.ln()))
        .add(&b.sub(&S::constant(1.0)).scale((1.0 - yc).ln()));
    loglik.scale(-wi)
}

/// Tweedie compound Poisson–Gamma row NLL written ONCE over a generic
/// [`JetScalar<2>`], seeded directly on the PREDICTOR primaries `(η_μ, η_d)`
/// (#932).
///
/// Unlike the NB/Gamma/Beta oracles — seeded on the natural parameters with the
/// caller applying the precision→η chain via the Fisher-orthogonal
/// `precision²·info` shortcut — this program carries `μ = exp(η_μ)` and
/// `φ = exp(−η_d)` INSIDE the tower, so `tower.g[1]` / `tower.h[1][1]` are the
/// η_d-space score and OBSERVED information directly, with the nonlinear
/// `∂²φ/∂η_d²` chain correction mechanically carried rather than re-derived.
///
/// Both density branches are smooth in `(η_μ, η_d)`:
/// * `y > 0` — the Nelder–Pregibon saddlepoint density
///   `ℓ = w·[ −dev/(2φ) − ½ln(2πφ) − ½p·ln y ]` with the unit deviance
///   `dev = 2·[ y^{2−p}/((1−p)(2−p)) − y·μ^{1−p}/(1−p) + μ^{2−p}/(2−p) ]`.
/// * `y = 0` — the exact compound-Poisson point mass
///   `ℓ = w·[ −μ^{2−p}/(φ(2−p)) ]`.
///
/// This is the single-source dense reference the production pruned single-axis
/// `dispersion_tweedie_disp_order2` is pinned against (bit-identical on the
/// consumed channels) and the cross-jet `Order2`/`Tower4` oracle.
///
/// [`JetScalar<2>`]: gam_math::jet_scalar::JetScalar
#[inline]
pub(crate) fn dispersion_tweedie_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    p: f64,
    wi: f64,
) -> S {
    let one_minus_p = 1.0 - p;
    let two_minus_p = 2.0 - p;
    // μ = exp(η_μ), φ = exp(−η_d): the natural parameters as jets in the
    // predictor primaries, so the whole derivative tower is in η-space.
    let mu = S::variable(eta_mu, 0).exp();
    let phi = S::variable(eta_d, 1).scale(-1.0).exp();
    if yi > 0.0 {
        // dev = 2·[ μ^{2−p}/(2−p) − y·μ^{1−p}/(1−p) + y^{2−p}/((1−p)(2−p)) ]
        let dev = mu
            .powf(two_minus_p)
            .scale(1.0 / two_minus_p)
            .sub(&mu.powf(one_minus_p).scale(yi / one_minus_p))
            .add(&S::constant(
                yi.powf(two_minus_p) / (one_minus_p * two_minus_p),
            ))
            .scale(2.0);
        // ℓ = dev·(−0.5/φ) − 0.5·ln(2πφ) − 0.5·p·ln y
        let loglik = dev
            .mul(&phi.recip().scale(-0.5))
            .sub(&phi.scale(2.0 * std::f64::consts::PI).ln().scale(0.5))
            .sub(&S::constant(0.5 * p * yi.ln()));
        loglik.scale(-wi)
    } else {
        // Exact point mass P(Y=0) = exp(−μ^{2−p}/(φ(2−p))).
        let c = mu.powf(two_minus_p).scale(1.0 / two_minus_p);
        let loglik = c.mul(&phi.recip()).scale(-1.0);
        loglik.scale(-wi)
    }
}
