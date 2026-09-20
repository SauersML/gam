//! gam#3231: on a finite latent law the score's units are a coordinate choice.
//! The anchor `Σ_k w_k Φ(a + b·u_k) = Φ(q)` on `{u_k, w_k}` is unchanged under
//! `z → (z − m)/s`, `u_k → (u_k − m)/s`, `b → b·s`, `a → a + b·m`, so the same
//! frame with its score recorded as `C + D·z` is the same model. The fit solves
//! a finite law on the score's own weighted standard units, so both recordings
//! reach the same standardized score up to the rounding of `(C + D·z − m)/s`,
//! and the fits, their saved slope coefficients and their predictions agree.
//!
//! The raw recording's scale `D` is gnomon#2402's polygenic score SD. Before
//! the fix the raw score reached the kernel as given: the slope coefficient sat
//! near `b/D ≈ 1.3e3` and every outer seed on a scale set by the units.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::utils::splitmix64;
use gam_math::probability::normal_cdf;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::inference::predict_input::build_predict_input_for_model;
use gam_predict::FittedModelPredictExt;
use ndarray::Array1;

const N: usize = 600;
const TRUE_SLOPE: f64 = 0.6;
const TRUE_BETA_X: f64 = 0.5;
const TRUE_INTERCEPT: f64 = -0.4;
/// `Corr(z, x)`: the score's conditional mean moves on the marginal index, so
/// the conditional law's `m(a)` has work to do and differs from the pooled law.
const M_SHIFT: f64 = 0.6;
/// gnomon#2402's raw polygenic-score SD, and an off-centre location.
const RAW_SCALE: f64 = 4.51e-4;
const RAW_CENTER: f64 = 1.3e-2;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// One draw of `(y, x, z)`, returned twice: with the score as drawn and with
/// the score recorded as `RAW_CENTER + RAW_SCALE·z`. Every other column is the
/// same bytes in both frames. The score's mean moves with `x` (gam#3016's
/// fixture), so the pooled and the conditional laws are different laws.
fn draw(seed: u64) -> (EncodedDataset, EncodedDataset) {
    let headers = ["y", "z", "x"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let mut state = seed;
    let mut standard = Vec::with_capacity(N);
    let mut raw = Vec::with_capacity(N);
    for _ in 0..N {
        let x = next_gauss(&mut state);
        let zeta = next_gauss(&mut state);
        let z = M_SHIFT * x + residual_sd * zeta;
        let eta = (TRUE_INTERCEPT + TRUE_BETA_X * x) * c_true + TRUE_SLOPE * zeta;
        let y = u8::from(next_unit(&mut state) < normal_cdf(eta)).to_string();
        let x = format!("{x:.17e}");
        standard.push(StringRecord::from(vec![
            y.clone(),
            format!("{z:.17e}"),
            x.clone(),
        ]));
        raw.push(StringRecord::from(vec![
            y,
            format!("{:.17e}", RAW_CENTER + RAW_SCALE * z),
            x,
        ]));
    }
    (
        encode_recordswith_inferred_schema(headers.clone(), standard)
            .expect("encode the standard-units frame"),
        encode_recordswith_inferred_schema(headers, raw).expect("encode the raw-units frame"),
    )
}

fn fit(data: &EncodedDataset, latent_measure: &str, label: &str) -> FittedModel {
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        latent_measure: Some(latent_measure.to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ x".to_string(), data, &config).unwrap_or_else(|e| {
        panic!("{label} bernoulli marginal-slope fit on {latent_measure}: {e}")
    });
    FittedModel::from_payload(payload)
}

fn predict(model: &FittedModel, data: &EncodedDataset) -> Array1<f64> {
    let predictor = model
        .predictor()
        .expect("a saved marginal-slope model has a predictor");
    let zeros = Array1::zeros(data.values.nrows());
    let input = build_predict_input_for_model(
        model,
        data.values.view(),
        &data.column_map(),
        model.training_headers.as_ref(),
        &zeros,
        &zeros,
        false,
    )
    .expect("the marginal-slope prediction input");
    predictor
        .predict_plugin_response(&input)
        .expect("marginal-slope prediction")
        .mean
}

fn max_abs_difference(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `score_map_in_standard_units` is whether the saved score map is the
/// recording's standard units: a global finite law fits on them, while a
/// conditional law fits on its standardized residual ζ, whose calibration
/// carries the units and whose saved score map is the identity.
fn assert_unit_free(latent_measure: &str, score_map_in_standard_units: bool) {
    let (standard_frame, raw_frame) = draw(0x3231_0000_5EED_0001);
    let standard = fit(&standard_frame, latent_measure, "standard-units");
    let raw = fit(&raw_frame, latent_measure, "raw-units");

    let standard_map = standard
        .latent_z_normalization
        .as_ref()
        .expect("a saved score map");
    let raw_map = raw
        .latent_z_normalization
        .as_ref()
        .expect("a saved score map");
    if score_map_in_standard_units {
        // The saved score maps are the two recordings' standard units: the
        // weighted mean and sd of each, so they carry the recording's affine
        // map to the rounding of two n-term sums.
        let scale_ratio = raw_map.sd / standard_map.sd;
        let location_gap = raw_map.mean - (RAW_CENTER + RAW_SCALE * standard_map.mean);
        eprintln!(
            "[3231] {latent_measure}: score sd ratio {scale_ratio:.12e} (recorded \
             {RAW_SCALE:e}), location gap {location_gap:.3e}"
        );
        let sum_rounding = N as f64 * f64::EPSILON;
        assert!(
            (scale_ratio / RAW_SCALE - 1.0).abs() <= sum_rounding,
            "{latent_measure}: the saved score scales must stand in the recording's ratio \
             {RAW_SCALE:e}, got {scale_ratio:.17e}"
        );
        assert!(
            location_gap.abs() <= sum_rounding * (RAW_CENTER.abs() + RAW_SCALE),
            "{latent_measure}: the saved score locations must carry the recording's shift, \
             gap {location_gap:.3e}"
        );
    } else {
        assert_eq!(
            (standard_map.mean, standard_map.sd, raw_map.mean, raw_map.sd),
            (0.0, 1.0, 0.0, 1.0),
            "{latent_measure}: a conditional law's saved score map is the identity"
        );
    }

    let standard_fit = standard.fit_result.as_ref().expect("a fitted state");
    let raw_fit = raw.fit_result.as_ref().expect("a fitted state");
    let log_likelihood_gap = (standard_fit.log_likelihood - raw_fit.log_likelihood).abs();
    let coefficient_gap = standard_fit
        .blocks
        .iter()
        .zip(&raw_fit.blocks)
        .map(|(a, b)| max_abs_difference(&a.beta, &b.beta))
        .fold(0.0, f64::max);
    let coefficient_scale = standard_fit
        .blocks
        .iter()
        .flat_map(|block| block.beta.iter())
        .map(|value| value.abs())
        .fold(1.0, f64::max);

    // Each model predicts its own recording of the same rows.
    let standard_mean = predict(&standard, &standard_frame);
    let raw_mean = predict(&raw, &raw_frame);
    let prediction_gap = max_abs_difference(&standard_mean, &raw_mean);

    eprintln!(
        "[3231] {latent_measure}: |Δ log-lik| {log_likelihood_gap:.3e} (log-lik {:.6}), \
         max |Δβ| {coefficient_gap:.3e} (scale {coefficient_scale:.3}), max |Δp| \
         {prediction_gap:.3e}",
        standard_fit.log_likelihood
    );

    // Both fits solve one problem on scores that differ by the rounding of the
    // standardization. The converged fits agree to the fit's own convergence:
    // the inner solve stops at a gradient norm the outer objective sees only in
    // second order, so a coefficient gap is bounded by the square root of the
    // working precision relative to the coefficients' scale, and the
    // log-likelihood and the probabilities, smooth in the coefficients, by the
    // same band on their own scale.
    let convergence_band = f64::EPSILON.sqrt();
    assert!(
        coefficient_gap <= convergence_band * coefficient_scale,
        "{latent_measure}: the slope and marginal coefficients must not depend on the \
         score's units: max |Δβ| {coefficient_gap:.3e} beyond {:.3e}",
        convergence_band * coefficient_scale
    );
    assert!(
        log_likelihood_gap <= convergence_band * standard_fit.log_likelihood.abs().max(1.0),
        "{latent_measure}: the log-likelihood must not depend on the score's units: \
         |Δ| {log_likelihood_gap:.3e}"
    );
    assert!(
        prediction_gap <= convergence_band,
        "{latent_measure}: predictions must not depend on the score's units: max |Δp| \
         {prediction_gap:.3e}"
    );
}

#[test]
fn global_empirical_fit_does_not_depend_on_the_score_units_3231() {
    assert_unit_free("global-empirical", true);
}

#[test]
fn conditional_location_scale_fit_does_not_depend_on_the_score_units_3231() {
    assert_unit_free("conditional-location-scale", false);
}
