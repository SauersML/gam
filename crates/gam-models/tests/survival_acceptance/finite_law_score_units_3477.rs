//! gam#3477: on a finite latent law the score's units are a coordinate choice
//! for the survival marginal-slope family too. The anchor
//! `Σ_k w_k Φ(a + b·u_k) = Φ(q)` on `{u_k, w_k}` with `η = q·c(g) + g·z` is
//! unchanged under `z → (z − m)/s`, `u_k → (u_k − m)/s`, `g → g·s` (the row
//! intercept absorbs the shift), so the same frame with its score recorded as
//! `C + D·z` is the same model. The fit solves a single-score finite law in the
//! score's own weighted standard units, so both recordings reach the same
//! standardized score up to the rounding of `(C + D·z − m)/s`, and the fits,
//! their coefficients and their predicted survival agree.
//!
//! The raw recording's scale `D` is gnomon#2402's polygenic score SD. Before
//! the fix the raw score reached the kernel as given, with the slope
//! coefficient near `b/D ≈ 2e3` and every outer seed on a scale set by the
//! units.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use ndarray::Array1;
use std::collections::HashMap;

const N: usize = 500;
const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;
const BETA_X: f64 = 0.4;
const SLOPE: f64 = 0.9;
/// `E[z | x] = M_SHIFT·x`, `Var(z | x) = 1 − M_SHIFT²`: the score's mean moves
/// on the marginal index, so the pooled law is not the conditional one.
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

fn normal_cdf(x: f64) -> f64 {
    gam_math::probability::normal_cdf(x)
}

/// Root of a monotone decreasing function on `[low, high]` by bisection.
fn decreasing_root(f: impl Fn(f64) -> f64, target: f64, low: f64, high: f64) -> f64 {
    let (mut low, mut high) = (low, high);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if f(mid) > target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// The true `S(t | x, z) = Φ(−(α + b·z))`, `α` anchored on `z | x ~ N(μ, σ²)`:
/// `Φ(−(α + b·μ)/√(1 + b²σ²)) = Φ(−q)` has the closed form below.
fn survival(x: f64, z: f64, t: f64) -> f64 {
    let q = LOCATION_LEVEL + LOCATION_TREND * t.ln() + BETA_X * x;
    let variance = 1.0 - M_SHIFT * M_SHIFT;
    let alpha = q * (1.0 + SLOPE * SLOPE * variance).sqrt() - SLOPE * M_SHIFT * x;
    normal_cdf(-(alpha + SLOPE * z))
}

/// One draw of `(time, event, z, x)`, returned twice: with the score as drawn
/// and with the score recorded as `RAW_CENTER + RAW_SCALE·z`. Every other
/// column is the same bytes in both frames.
fn draw(seed: u64) -> (EncodedDataset, EncodedDataset) {
    let headers = ["time", "event", "z", "x"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let mut state = seed;
    let mut standard = Vec::with_capacity(N);
    let mut raw = Vec::with_capacity(N);
    for _ in 0..N {
        let x = next_gauss(&mut state);
        let z = M_SHIFT * x + residual_sd * next_gauss(&mut state);
        let u = next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6);
        let censor = 0.35 + 5.0 * next_unit(&mut state);
        let log_time = decreasing_root(|log_t| survival(x, z, log_t.exp()), u, -6.0, 6.0);
        let event_time = log_time.exp();
        let (time, event) = if event_time <= censor {
            (event_time, 1u8)
        } else {
            (censor, 0u8)
        };
        let time = format!("{:.17e}", time.clamp(1e-3, 1e3));
        let event = event.to_string();
        let x = format!("{x:.17e}");
        standard.push(StringRecord::from(vec![
            time.clone(),
            event.clone(),
            format!("{z:.17e}"),
            x.clone(),
        ]));
        raw.push(StringRecord::from(vec![
            time,
            event,
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

fn fit(data: &EncodedDataset, label: &str) -> FittedModel {
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        time_num_internal_knots: 3,
        latent_measure: Some("global-empirical".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("Surv(time, event) ~ x".to_string(), data, &config)
        .unwrap_or_else(|e| panic!("{label} survival marginal-slope fit: {e}"));
    FittedModel::from_payload(payload)
}

/// Predicted `S(t_i | x_i, z_i)` at every row's recorded time.
fn predict(model: &FittedModel, data: &EncodedDataset) -> Array1<f64> {
    let col_map: HashMap<String, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = Array1::<f64>::zeros(data.values.nrows());
    let prediction = predict_survival(
        SurvivalPredictRequest {
            model,
            data: data.values.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("survival marginal-slope prediction at the training rows");
    prediction.survival.column(0).to_owned()
}

fn max_abs_difference(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn global_empirical_survival_fit_does_not_depend_on_the_score_units_3477() {
    super::initialize_cpu_fitting();
    let (standard_frame, raw_frame) = draw(0x3477_0000_5EED_0001);
    let standard = fit(&standard_frame, "standard-units");
    let raw = fit(&raw_frame, "raw-units");

    // The saved score maps are the two recordings' standard units: the weighted
    // mean and sd of each, so they carry the recording's affine map to the
    // rounding of two n-term sums.
    let standard_map = standard
        .latent_z_normalization
        .as_ref()
        .expect("a saved score map");
    let raw_map = raw.latent_z_normalization.as_ref().expect("a saved score map");
    let scale_ratio = raw_map.sd / standard_map.sd;
    let location_gap = raw_map.mean - (RAW_CENTER + RAW_SCALE * standard_map.mean);
    eprintln!(
        "[3477] score sd ratio {scale_ratio:.12e} (recorded {RAW_SCALE:e}), location gap \
         {location_gap:.3e}"
    );
    let sum_rounding = N as f64 * f64::EPSILON;
    assert!(
        (scale_ratio / RAW_SCALE - 1.0).abs() <= sum_rounding,
        "the saved score scales must stand in the recording's ratio {RAW_SCALE:e}, got \
         {scale_ratio:.17e}"
    );
    assert!(
        location_gap.abs() <= sum_rounding * (RAW_CENTER.abs() + RAW_SCALE),
        "the saved score locations must carry the recording's shift, gap {location_gap:.3e}"
    );

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
    let standard_survival = predict(&standard, &standard_frame);
    let raw_survival = predict(&raw, &raw_frame);
    let prediction_gap = max_abs_difference(&standard_survival, &raw_survival);

    eprintln!(
        "[3477] |Δ log-lik| {log_likelihood_gap:.3e} (log-lik {:.6}), max |Δβ| \
         {coefficient_gap:.3e} (scale {coefficient_scale:.3}), max |ΔS| {prediction_gap:.3e}",
        standard_fit.log_likelihood
    );

    // Both fits solve one problem on scores that differ by the rounding of the
    // standardization. The converged fits agree to the fit's own convergence:
    // the inner solve stops at a gradient norm the outer objective sees only in
    // second order, so a coefficient gap is bounded by the square root of the
    // working precision relative to the coefficients' scale, and the
    // log-likelihood and the survival probabilities, smooth in the
    // coefficients, by the same band on their own scale.
    let convergence_band = f64::EPSILON.sqrt();
    assert!(
        coefficient_gap <= convergence_band * coefficient_scale,
        "the time, marginal and slope coefficients must not depend on the score's units: \
         max |Δβ| {coefficient_gap:.3e} beyond {:.3e}",
        convergence_band * coefficient_scale
    );
    assert!(
        log_likelihood_gap <= convergence_band * standard_fit.log_likelihood.abs().max(1.0),
        "the log-likelihood must not depend on the score's units: |Δ| {log_likelihood_gap:.3e}"
    );
    assert!(
        prediction_gap <= convergence_band,
        "predicted survival must not depend on the score's units: max |ΔS| {prediction_gap:.3e}"
    );
}
