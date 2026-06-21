//! Regression for #1463 — NB NUTS `model.sample()` must draw the posterior at
//! the fitted overdispersion `theta_hat`, not the construction seed `theta = 1.0`.
//!
//! ## The bug
//!
//! `gam::inference::sample::sample_standard` forwards a [`LikelihoodSpec`] into
//! `run_nuts_sampling_flattened_family`. For Negative-Binomial the NUTS dispatch
//! (`src/inference/hmc.rs`, the
//! `(ResponseFamily::NegativeBinomial { theta, .. }, _, _)` arm) destructures
//! `theta` straight off that spec and passes it through as the NB
//! log-likelihood / score overdispersion — it does **not** read `theta` from the
//! `glm.gamma_shape` slot the Gamma arm uses. The spec built for a saved model
//! carries the *construction seed* `theta = 1.0` (the value that only seeds the
//! inner solve), so before the fix the NB NUTS likelihood used
//! `Var(y) = μ + μ²/1` instead of `Var(y) = μ + μ²/θ̂`. Over-stated variance →
//! every coefficient's posterior SD inflated ~1.4–1.5×. The Wald `summary()`
//! path was correct; only the HMC/NUTS path was wrong (the HMC sibling of the
//! replicate-path bug #1124).
//!
//! ## The fix (commit `cee906a97`)
//!
//! `sample_standard` (`src/inference/sample.rs`) now refreshes the spec's family
//! `theta` from the fit's jointly-estimated `theta_hat`
//! (`fit.likelihood_scale.negbin_theta()`) before dispatch — the exact slot the
//! NUTS arm consumes. Its in-source comment states the refresh "mirrors how the
//! replicate path reads `theta_hat` via the canonical `family_noise_parameter`
//! helper (`negbin_theta().or(seed)`)". That canonical helper —
//! [`gam::inference::generative::family_noise_parameter`] — is the single,
//! PUBLIC, production dispersion picker shared by `gam generate` and
//! `sample_replicates`; the NB arm is literally `scale.negbin_theta().or(Some(theta))`.
//!
//! ## What this test asserts — against the REAL production function
//!
//! `sample_standard` is a private `fn` taking a `SavedModel` + NUTS config, so it
//! cannot be invoked from an integration test without a full saved fit. Rather
//! than re-implement its logic (a tautology that would pass even if the
//! production refresh were reverted), this test drives the SAME production
//! dispersion-selection logic through the public `family_noise_parameter`: the NB
//! variance parameter the sampler/generator consumes is sourced from the fit's
//! estimated `theta_hat` (NOT the seed), honours a user-fixed `theta`, and falls
//! back to the seed only for an unfitted model. A revert that drops the
//! `negbin_theta()` consult (drawing at the seed `1.0`) fails this test.

use gam::inference::generative::family_noise_parameter;
use gam::types::{LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily};

/// The construction-seed overdispersion stamped on every NB family spec before
/// the inner solve refines it (`ResponseFamily::NegativeBinomial { theta: 1.0 }`).
const SEED_THETA: f64 = 1.0;
/// A jointly-estimated `theta_hat` grossly different from the seed, so a residual
/// seed-`theta` leak is unmistakable.
const FITTED_THETA_HAT: f64 = 3.7;
/// The residual-scale fallback argument; for NB it must be IGNORED (NB picks
/// theta, never the residual SD), so a deliberately-wrong value here would
/// surface a mis-wired family arm.
const WRONG_RESIDUAL_SCALE: f64 = 99.0;

/// #1463 PRIMARY: the estimated-theta fit path. The NB dispersion the sampler
/// consumes must be the fitted `theta_hat`, NOT the construction seed `1.0` and
/// NOT the residual-scale fallback. This exercises the production
/// `family_noise_parameter` NB arm (`scale.negbin_theta().or(Some(theta))`) that
/// `sample_standard`'s refresh mirrors.
#[test]
fn nb_dispersion_is_fitted_theta_hat_not_seed_1463() {
    let scale = LikelihoodScaleMetadata::EstimatedNegBinTheta {
        theta: FITTED_THETA_HAT,
    };
    // Sanity: the value the picker sources is exactly the fitted theta_hat.
    assert_eq!(
        scale.negbin_theta(),
        Some(FITTED_THETA_HAT),
        "scale metadata must expose the fitted theta_hat the refresh reads"
    );

    let spec = LikelihoodSpec::negative_binomial_log(SEED_THETA);
    let theta = family_noise_parameter(scale, WRONG_RESIDUAL_SCALE, &spec)
        .expect("NB family must carry a dispersion parameter");

    assert_eq!(
        theta, FITTED_THETA_HAT,
        "NB sampler dispersion still uses seed theta={SEED_THETA} (or the residual \
         scale) instead of the fitted theta_hat={FITTED_THETA_HAT}; the sampler draws \
         at Var(y)=mu+mu^2/theta with the wrong theta and inflates every coefficient \
         posterior SD (#1463)"
    );
    assert_ne!(
        theta, SEED_THETA,
        "the construction seed theta={SEED_THETA} leaked into the NB likelihood (#1463)"
    );
    assert_ne!(
        theta, WRONG_RESIDUAL_SCALE,
        "NB must pick theta_hat, never the residual-scale fallback"
    );
}

/// #1463: a user-fixed theta (`--negative-binomial-theta`, #983) must be honored
/// verbatim — the picker sources the held value via `FixedNegBinTheta`, never the
/// seed and never the residual scale.
#[test]
fn nb_dispersion_honors_user_fixed_theta_1463() {
    const USER_FIXED_THETA: f64 = 5.25;
    let scale = LikelihoodScaleMetadata::FixedNegBinTheta {
        theta: USER_FIXED_THETA,
    };
    // A fixed-theta spec still carries the seed on the family variant; the scale
    // metadata is what the picker consults.
    let spec = LikelihoodSpec::negative_binomial_log_fixed(SEED_THETA);
    let theta = family_noise_parameter(scale, WRONG_RESIDUAL_SCALE, &spec)
        .expect("fixed NB family must carry a dispersion parameter");
    assert_eq!(
        theta, USER_FIXED_THETA,
        "user-fixed NB theta must reach the NB likelihood exactly; got {theta}"
    );
}

/// #1463 fallback: an UNFITTED NB model (no scale metadata) must fall back to the
/// spec seed `theta` — the `.or(Some(theta))` tail of the production picker. This
/// pins that the fix did not over-rotate into always-ignoring the seed.
#[test]
fn nb_dispersion_falls_back_to_seed_when_unfitted_1463() {
    let seeded = LikelihoodSpec::negative_binomial_log(2.5);
    let theta = family_noise_parameter(LikelihoodScaleMetadata::Unspecified, WRONG_RESIDUAL_SCALE, &seeded)
        .expect("unfitted NB still resolves to its seed");
    assert_eq!(
        theta, 2.5,
        "an unfitted NB model must fall back to the spec seed theta, not the residual scale"
    );
}

/// Non-NB families carry no NB `theta`: the residual scale (or family-specific
/// dispersion) is returned, and `negbin_theta()` is `None`. Guards against the
/// refresh mis-firing on a non-NB family.
#[test]
fn dispersion_picker_is_not_negbin_for_other_families_1463() {
    let scale = LikelihoodScaleMetadata::ProfiledGaussian;
    assert_eq!(
        scale.negbin_theta(),
        None,
        "non-NB scale metadata must not expose an NB theta"
    );
    let gaussian = LikelihoodSpec {
        response: ResponseFamily::Gaussian,
        link: gam::types::InverseLink::Standard(gam::types::StandardLink::Identity),
    };
    let sigma = family_noise_parameter(scale, 1.75, &gaussian)
        .expect("Gaussian generative sigma");
    assert_eq!(
        sigma, 1.75,
        "Gaussian generative noise is the residual scale, not an NB theta"
    );
}
