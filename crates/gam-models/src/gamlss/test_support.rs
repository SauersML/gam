//! Shared test-only oracles for the gamlss family stack.
//!
//! Items here exist purely to pin production fast paths against a dense,
//! single-source reference. They are exercised from more than one gamlss test
//! module (the `dispersion_family` unit tests and the family-level behaviour
//! tests in `tests.rs`), so they live at the common parent rather than as
//! `#[cfg(test)]` items dangling off a production `src/` module — which is the
//! shape `dead_code` cannot see and the build-time ban-scanner rejects.

use gam_math::jet_scalar::JetScalar;

/// Tweedie compound Poisson–Gamma row NLL written ONCE over a generic
/// [`JetScalar<2>`], seeded directly on the PREDICTOR primaries `(η_μ, η_d)`
/// (#932).
///
/// Unlike the NB/Gamma/Beta oracles — which seed on the natural parameters and
/// let the caller apply the precision→η chain via the Fisher-orthogonal
/// `precision²·info` shortcut — this tower carries `μ = exp(η_μ)` and
/// `φ = exp(−η_d)` INSIDE the program, so `tower.g[1]` / `tower.h[1][1]` are
/// the η_d-space score and OBSERVED information directly, with the nonlinear
/// `∂²φ/∂η_d²` chain correction the hand path documented (the `y = 0` branch's
/// `2c/φ − c/φ = c/φ` cancellation) mechanically carried rather than re-derived.
///
/// Both density branches are smooth in `(η_μ, η_d)`:
/// * `y > 0` — the Nelder–Pregibon saddlepoint density
///   `ℓ = w·[ −dev/(2φ) − ½ln(2πφ) − ½p·ln y ]` with the unit deviance
///   `dev = 2·[ y^{2−p}/((1−p)(2−p)) − y·μ^{1−p}/(1−p) + μ^{2−p}/(2−p) ]`.
/// * `y = 0` — the exact compound-Poisson point mass
///   `ℓ = w·[ −μ^{2−p}/(φ(2−p)) ]`.
///
/// `μ` and `φ` enter the deviance only through `powf` and a `recip`, whose
/// `[f64; 5]` derivative stacks the tower owns, so no primitive is re-derived:
/// only the Leibniz/Faà-di-Bruno composition is mechanized. Production consumes
/// the pruned single-axis `dispersion_tweedie_disp_order2`; this `K=2` generic
/// is the dense oracle / cross-tool witness that pins it.
#[inline]
pub(crate) fn dispersion_tweedie_nll_generic<S: JetScalar<2>>(
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
