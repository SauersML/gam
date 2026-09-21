//! gam#3724: a saved transformation-normal model's posterior draws come from
//! the fit's persisted truncated law, not from a Gaussian re-centred on the
//! published coefficients.
//!
//! ## The defect
//!
//! A CTN fit's posterior is the Laplace Gaussian `N(c, Σ)` truncated to the
//! monotonicity cone `A β ≥ b`, with ambient centre `c = mode − Σ·∇`. The fit
//! persists that law's identity (the KKT mode, `c` and the exact `A β ≥ b`)
//! and publishes the truncated posterior MEAN `E[β | cone]` as `fit.beta`.
//! `sample_saved_model` rejection-sampled `N(fit.beta, Σ)` restricted to the
//! cone instead. It truncated an already-truncated law a second time, and it
//! stamped the iid pair `rhat = 1`, `ess = n` on draws it never diagnosed.
//!
//! ## What is asserted
//!
//! 1. **The draws come from the truncated-law sampler.** The result is stamped
//!    `PosteriorSampler::TruncatedLaplaceHmc`, the exact reflective HMC over
//!    the persisted identity, whose `rhat` and `ess` are measured. The old path
//!    stamped `PosteriorSampler::Laplace`.
//! 2. **The draws have the published mean.** `fit.beta` is `E[β | cone]` of
//!    the persisted law, so the draw mean must agree with it to within the
//!    draws' own Monte Carlo error `sd_j / √ess`, where `ess` is the sampler's
//!    measured minimum over coordinates. The bound is five of those standard
//!    errors per coordinate, plus the rounding of an `n`-term mean.
//! 3. **The chains mixed.** The measured split R-hat and ESS pass the
//!    sampler's own `converged` gate.
//!
//! The fixture is the log-normal law of `ctn_transform_tails_2600`: `Y = exp(Z)`
//! gives `h(y) = ln y`, the shape CTN exists for.

use gam::hmc::{NUTS_CHAINS, NutsConfig, PosteriorSampler};
use gam::inference::model::{FittedModel, PredictModelClass};
use gam::inference::model_payload_builders::fit_formula_to_payload;
use gam::sample::sample_saved_model;
use gam::test_support::synthetic::SplitMixNormalRng;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::Array2;

const N: usize = 256;

/// Monte Carlo standard errors allowed between the draw mean and the published
/// truncated posterior mean, per coordinate (two-sided `P(|Z| > 5) ≈ 5.7e-7`).
const MONTE_CARLO_STANDARD_ERRORS: f64 = 5.0;

#[test]
fn transformation_normal_draws_come_from_the_persisted_truncated_law_3724() {
    init_parallelism();
    let mut rng = SplitMixNormalRng::new(0x3724_C7Du64);
    let y: Vec<f64> = (0..N).map(|_| rng.standard_normal().exp()).collect();
    let headers = vec!["y".to_string()];
    let rows: Vec<csv::StringRecord> = y
        .iter()
        .map(|value| csv::StringRecord::from(vec![format!("{value:.17e}")]))
        .collect();
    let dataset =
        encode_recordswith_inferred_schema(headers, rows).expect("encode the lognormal column");
    let config = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ 1".to_string(), &dataset, &config)
        .expect("intercept-only transformation-normal fit");
    let model = FittedModel::from_payload(payload);
    assert_eq!(
        model.predict_model_class(),
        PredictModelClass::TransformationNormal,
        "the fixture must produce a transformation-normal model"
    );
    let published_mean = model
        .unified
        .as_ref()
        .expect("a saved CTN model carries its unified fit")
        .beta
        .clone();

    let frame = Array2::from_shape_fn((N, 1), |(i, _)| y[i]);
    let cfg = NutsConfig {
        n_samples: 1000,
        seed: 3724,
    };
    let draws = sample_saved_model(
        &model,
        frame.view(),
        &dataset.column_map(),
        Some(&dataset.headers),
        &cfg,
    )
    .expect("sample the saved transformation-normal posterior");

    assert_eq!(
        draws.sampler,
        PosteriorSampler::TruncatedLaplaceHmc,
        "CTN draws must come from the persisted truncated law's reflective HMC, whose \
         diagnostics are measured"
    );
    let p = published_mean.len();
    assert_eq!(draws.samples.nrows(), cfg.n_samples * NUTS_CHAINS);
    assert_eq!(draws.samples.ncols(), p);
    assert!(
        draws.samples.iter().all(|value| value.is_finite()),
        "every draw must be finite"
    );
    assert!(
        draws.rhat.is_finite() && draws.ess.is_finite() && draws.ess > 0.0,
        "measured diagnostics must be finite: rhat {}, ess {}",
        draws.rhat,
        draws.ess
    );
    assert!(
        draws.converged,
        "the truncated-law chains did not mix: rhat {}, ess {}",
        draws.rhat, draws.ess
    );

    let monte_carlo_scale = draws.ess.sqrt().recip();
    let n_draws = draws.samples.nrows() as f64;
    for j in 0..p {
        let gap = (draws.posterior_mean[j] - published_mean[j]).abs();
        // Plus the rounding of an `n`-term mean, so a coordinate the gauge
        // fixes (sd 0) is compared at the precision its mean is computed to.
        let bound = MONTE_CARLO_STANDARD_ERRORS * draws.posterior_std[j] * monte_carlo_scale
            + n_draws * f64::EPSILON * published_mean[j].abs();
        println!(
            "[3724] coef {j}: draw mean {:.6e}, published mean {:.6e}, gap {gap:.3e}, bound \
             {bound:.3e}",
            draws.posterior_mean[j], published_mean[j]
        );
        assert!(
            gap <= bound,
            "coefficient {j}: draw mean {} is {gap:.3e} from the published truncated posterior \
             mean {}, beyond {MONTE_CARLO_STANDARD_ERRORS} Monte Carlo standard errors \
             ({bound:.3e}; sd {}, ess {})",
            draws.posterior_mean[j],
            published_mean[j],
            draws.posterior_std[j],
            draws.ess
        );
    }
}
