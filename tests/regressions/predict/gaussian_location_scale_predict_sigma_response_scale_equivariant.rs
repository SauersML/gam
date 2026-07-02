//! Regression (#1874 / #1928): the production Gaussian location-scale σ PREDICT
//! path must reconstruct σ in raw response units with **exactly one** factor of
//! the response scale — no more, no less.
//!
//! ## Background: how the two halves of the pipeline share the response scale
//!
//! `fit_gaussian_location_scale_model` standardizes the response by
//! `s = sample_std(y)` (so the fixed log-σ soft floor `LOGB_SIGMA_FLOOR = 0.01`
//! is operationally ≈1 % of the spread, mirroring mgcv's `gaulss(b=0.01)`), fits
//! on `y/s`, then maps the coefficients back to **raw response units** in
//! `rescale_gaussian_location_scale_to_raw`:
//!
//!   * the Location/Mean coefficients are multiplied by `s`;
//!   * the log-σ (Scale) block **intercept is shifted by `+ln(s)`**, which turns
//!     the internal link `σ_int = b + exp(η_int)` into
//!     `b + exp(η_raw) = b + s·exp(η_int)` — i.e. the multiplicative `exp(η)`
//!     term is already carried into raw units by the intercept shift.
//!
//! The persisted Scale-block coefficients are therefore RAW-units coefficients
//! (`η_raw = η_int + ln s`). The correct raw σ surface — the one the #1874
//! equivariance test, the FFI, and mgcv all agree on — is
//!
//!   σ_raw(x) = s·b + exp(η_raw(x)) = s·b + s·exp(η_int(x)) = s·σ_int(x).
//!
//! The floor picks up its `s` explicitly (it sits outside the exp and cannot
//! ride the intercept shift); the exp term picks up its `s` from the intercept
//! shift. Exactly one factor of `s` on each — total one factor of `s` on σ.
//!
//! ## The defect
//!
//! `GaussianLocationScalePredictor::compute_sigma` (#1928) computes
//!
//!   σ_pred(x) = response_scale · (sigma_floor + exp(η_noise))
//!
//! reading `η_noise` from the persisted Scale block — i.e. the **raw** `η_raw`,
//! which already contains one factor of `s`. With `sigma_floor = b` and
//! `response_scale = s` that is
//!
//!   σ_pred = s·(b + exp(η_raw)) = s·b + s²·exp(η_int),
//!
//! which double-counts the response scale on the `exp` term (`s²` instead of
//! `s`). #1928's own predictor unit tests all use `response_scale = 1.0`, so the
//! double scaling is invisible there; on any real fit (`s ≠ 1`) the predicted
//! predictive σ / variance is wrong by a factor that grows without bound as the
//! response units shrink or grow. This breaks response-scale equivariance of the
//! *reported* uncertainty, which is the whole subject of #1874.
//!
//! ## What this test pins
//!
//! It fits a genuine heteroscedastic Gaussian location-scale model (so `s ≠ 1`),
//! reconstructs the reference σ the fit-side way (`s·b + exp(η_raw)`, identical
//! to the #1874 equivariance test), then drives the REAL production predictor
//! (`GaussianLocationScalePredictor` built exactly as `FittedModel::predictor`
//! builds it) over the training abscissae and requires the two to agree to
//! floating-point tolerance. On the double-scaling code the predicted σ is
//! `s×` too large/small and this fails; with a single, consistent factor of `s`
//! it passes. It is deterministic (one fit, pure reconstruction) so it cannot
//! flake on the #1809/#1811 optimizer non-reproducibility.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::gaussian_location_scale::GaussianLocationScalePredictor;
use gam_predict::{PredictInput, PredictableModel};
use ndarray::{Array1, Array2};

const LOGB_SIGMA_FLOOR: f64 = 0.01;

/// Deterministic heteroscedastic recipe (seed-42 LCG), matching the #1874
/// equivariance test so the two regressions describe the same model.
fn make_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state: u64 = 42;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let z: Vec<f64> = (0..n)
        .map(|_| {
            let u1 = next_unit().max(1e-300);
            let u2 = next_unit();
            (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos()
        })
        .collect();
    let mu_true = |t: f64| (two_pi * t).sin();
    let sigma_true = |t: f64| 0.1 + 0.2 * (two_pi * t).sin();
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x[i]) + sigma_true(x[i]) * z[i])
        .collect();
    (x, y)
}

#[test]
fn gaussian_location_scale_predict_sigma_matches_fit_reconstruction() {
    init_parallelism();

    let n = 200usize;
    let (x, y) = make_data(n);

    // ---- fit through the public location-scale path -------------------------
    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode loc-scale data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        ..
    }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };

    // A genuinely heteroscedastic response is standardized by its own spread, so
    // this fit MUST exercise `s ≠ 1` — otherwise the double-scaling defect would
    // be silently invisible (exactly the blind spot in #1928's unit tests).
    assert!(
        (response_scale - 1.0).abs() > 1e-3,
        "test precondition: response_scale must differ from 1 to exercise the \
         scale factor (got {response_scale})"
    );

    let beta_mu = fit
        .fit
        .block_by_role(BlockRole::Location)
        .or_else(|| fit.fit.block_by_role(BlockRole::Mean))
        .expect("mean block present")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    // ---- reference σ: the fit-side reconstruction (raw units) ---------------
    // σ_raw = response_scale·LOGB_SIGMA_FLOOR + exp(η_raw), where η_raw is the
    // persisted (intercept-shifted) Scale-block predictor. This is byte-for-byte
    // the reconstruction the #1874 equivariance test and the FFI use.
    let mut grid = Array2::<f64>::zeros((n, ncols));
    for (i, &t) in x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let noise_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at grid");
    let eta_raw = noise_design.design.apply(&beta_noise);
    let sigma_reference: Vec<f64> = eta_raw
        .iter()
        .map(|&e| response_scale * LOGB_SIGMA_FLOOR + e.exp())
        .collect();

    // ---- production predict path --------------------------------------------
    // Build the predictor EXACTLY as `FittedModel::predictor` does for a
    // Gaussian location-scale model (see gam-predict `lib.rs`): raw beta blocks,
    // floor = LOGB_SIGMA_FLOOR, response_scale from the payload.
    let predictor = GaussianLocationScalePredictor {
        beta_mu,
        beta_noise,
        sigma_floor: LOGB_SIGMA_FLOOR,
        response_scale,
        covariance: None,
        link_wiggle: None,
    };
    let mean_design = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid")
        .design;
    let input = PredictInput {
        design: mean_design,
        offset: Array1::zeros(n),
        design_noise: Some(
            build_term_collection_design(grid.view(), &fit.noisespec_resolved)
                .expect("rebuild noise design at grid")
                .design,
        ),
        offset_noise: Some(Array1::zeros(n)),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };
    let sigma_pred = predictor
        .predict_noise_scale(&input)
        .expect("predict_noise_scale")
        .expect("predict_noise_scale returned Some");

    // ---- assert the predict path reports the raw-unit σ ---------------------
    let mut max_rel = 0.0_f64;
    let mut worst = (0usize, 0.0_f64, 0.0_f64);
    for i in 0..n {
        let rel = (sigma_pred[i] - sigma_reference[i]).abs() / sigma_reference[i].abs().max(1e-30);
        if rel > max_rel {
            max_rel = rel;
            worst = (i, sigma_pred[i], sigma_reference[i]);
        }
    }

    assert!(
        max_rel < 1.0e-10,
        "Gaussian location-scale PREDICT σ disagrees with the fit-side raw-unit \
         reconstruction by up to {:.3e} (worst at i={}: predicted={:.6e}, \
         reference={:.6e}, ratio={:.6}). The persisted Scale-block coefficients \
         are already in RAW units (log-σ intercept shifted by +ln(response_scale) \
         in rescale_gaussian_location_scale_to_raw), so σ = response_scale·b + \
         exp(η_raw); multiplying the whole (b + exp(η_raw)) by response_scale again \
         double-counts the response scale on the exp term (s² instead of s). \
         The ratio ≈ response_scale={:.6} confirms the extra factor (#1874/#1928).",
        max_rel,
        worst.0,
        worst.1,
        worst.2,
        worst.1 / worst.2.max(1e-30),
        response_scale,
    );
}
