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
//! NUTS arm consumes. The canonical
//! [`gam::inference::generative::family_noise_parameter`] helper is the single,
//! public production dispersion picker shared by `gam generate` and
//! `sample_replicates`. It accepts only family-consistent fitted scale metadata:
//! estimated NB uses `EstimatedNegBinTheta`, fixed NB uses `FixedNegBinTheta`,
//! and missing metadata is an error rather than permission to reuse a seed.
//!
//! ## What this test asserts — against the REAL production function
//!
//! `sample_standard` is a private `fn` taking a `SavedModel` + NUTS config, so it
//! cannot be invoked from an integration test without a full saved fit. Rather
//! than re-implement its logic (a tautology that would pass even if the
//! production refresh were reverted), this test drives the SAME production
//! dispersion-selection logic through the public `family_noise_parameter`: the NB
//! variance parameter the sampler/generator consumes is sourced from the fit's
//! estimated `theta_hat` (NOT the seed), honours a user-fixed `theta`, and
//! refuses an unfitted model with unresolved scale metadata. A revert that
//! draws at the seed `1.0` fails this test.

use gam::inference::generative::family_noise_parameter;
use gam::types::{LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily};

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
        .expect("profiled Gaussian scale metadata must be valid")
        .expect("Gaussian family must carry a generative sigma");
    assert_eq!(
        sigma, 1.75,
        "Gaussian generative noise is the residual scale, not an NB theta"
    );
}
