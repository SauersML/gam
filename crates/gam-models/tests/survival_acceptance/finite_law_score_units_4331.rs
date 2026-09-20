//! gam#4331: with `K ≥ 2` scores the survival marginal-slope anchor reads the
//! drive `Σ_j g_j z_j` on the joint law of the score vector, and
//! `Σ_j g_j z_j = Σ_j (g_j s_j)·(z_j − m_j)/s_j + Σ_j g_j m_j` with the constant
//! absorbed by the row intercept, so each coordinate's units are a coordinate
//! choice on its own. The same frame with its second score recorded as
//! `C + D·z₁` is the same model: the fit solves every uncalibrated coordinate
//! of the joint law in its own weighted standard units and the persisted law
//! records the K maps, so both recordings reach the same standardized scores
//! up to the rounding of `(C + D·z₁ − m₁)/s₁`, and the fits, their
//! coefficients and their predicted survival agree.
//!
//! Before the fix only score 0 was mapped: score 1 reached the kernel as
//! given, with its slope near `b₁/D ≈ 2e3`, and a model saved from a
//! normalised K ≥ 2 fit was refused because the payload held one map.

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

const N: usize = 600;
const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;
const BETA_X: f64 = 0.4;
const SLOPES: [f64; 2] = [0.9, 0.7];
/// `E[z₀ | x] = M_SHIFT·x`, `Var(z₀ | x) = 1 − M_SHIFT²`; `z₁ ~ N(0, 1)`
/// independent of both.
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

/// The true `S(t | x, z) = Φ(−(α + bᵀz))`, `α` anchored on
/// `z | x ~ N(μ(x), Σ)` with `μ = (M_SHIFT·x, 0)`, `Σ = diag(1 − M_SHIFT², 1)`:
/// `Φ(−(α + bᵀμ)/√(1 + bᵀΣb)) = Φ(−q)`.
fn survival(x: f64, z: [f64; 2], t: f64) -> f64 {
    let q = LOCATION_LEVEL + LOCATION_TREND * t.ln() + BETA_X * x;
    let drive_variance =
        SLOPES[0] * SLOPES[0] * (1.0 - M_SHIFT * M_SHIFT) + SLOPES[1] * SLOPES[1];
    let alpha = q * (1.0 + drive_variance).sqrt() - SLOPES[0] * M_SHIFT * x;
    normal_cdf(-(alpha + SLOPES[0] * z[0] + SLOPES[1] * z[1]))
}

/// One draw of `(time, event, x, z₀, z₁)`, returned twice: with the scores as
/// drawn and with `z₁` recorded as `RAW_CENTER + RAW_SCALE·z₁`. Every other
/// column is the same bytes in both frames.
fn draw(seed: u64) -> (EncodedDataset, EncodedDataset) {
    let headers = ["time", "event", "x", "z0", "z1"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let mut state = seed;
    let mut standard = Vec::with_capacity(N);
    let mut raw = Vec::with_capacity(N);
    for _ in 0..N {
        let x = next_gauss(&mut state);
        let z = [
            M_SHIFT * x + residual_sd * next_gauss(&mut state),
            next_gauss(&mut state),
        ];
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
        let z0 = format!("{:.17e}", z[0]);
        standard.push(StringRecord::from(vec![
            time.clone(),
            event.clone(),
            x.clone(),
            z0.clone(),
            format!("{:.17e}", z[1]),
        ]));
        raw.push(StringRecord::from(vec![
            time,
            event,
            x,
            z0,
            format!("{:.17e}", RAW_CENTER + RAW_SCALE * z[1]),
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
        z_column: Some("z0".to_string()),
        slope_formula: Some("slope(z0, 1) + slope(z1, 1)".to_string()),
        time_num_internal_knots: 3,
        latent_measure: Some("global-empirical".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("Surv(time, event) ~ x".to_string(), data, &config)
        .unwrap_or_else(|e| panic!("{label} K=2 survival marginal-slope fit: {e}"));
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
    .expect("K=2 survival marginal-slope prediction at the training rows");
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
fn joint_law_survival_fit_does_not_depend_on_each_scores_units_4331() {
    super::initialize_cpu_fitting();
    let (standard_frame, raw_frame) = draw(0x4331_0000_5EED_0001);
    let standard = fit(&standard_frame, "standard-units");
    let raw = fit(&raw_frame, "raw-units");

    let standard_law = standard
        .survival_marginal_slope_joint_latent_law
        .as_ref()
        .expect("a K=2 global-empirical fit persists its joint latent law");
    let raw_law = raw
        .survival_marginal_slope_joint_latent_law
        .as_ref()
        .expect("a K=2 global-empirical fit persists its joint latent law");

    // Score 0 is the same bytes in both frames, so its map is the same map.
    assert_eq!(
        (standard_law.score_location[0], standard_law.score_scale[0]),
        (raw_law.score_location[0], raw_law.score_scale[0]),
        "score 0's unit map must not depend on how score 1 was recorded"
    );
    // Score 1's maps are the two recordings' weighted mean and sd, so they
    // carry the recording's affine map to the rounding of two n-term sums.
    let scale_ratio = raw_law.score_scale[1] / standard_law.score_scale[1];
    let location_gap =
        raw_law.score_location[1] - (RAW_CENTER + RAW_SCALE * standard_law.score_location[1]);
    eprintln!(
        "[4331] score-1 sd ratio {scale_ratio:.12e} (recorded {RAW_SCALE:e}), location gap \
         {location_gap:.3e}"
    );
    let sum_rounding = N as f64 * f64::EPSILON;
    assert!(
        (scale_ratio / RAW_SCALE - 1.0).abs() <= sum_rounding,
        "the saved score-1 scales must stand in the recording's ratio {RAW_SCALE:e}, got \
         {scale_ratio:.17e}"
    );
    assert!(
        location_gap.abs() <= sum_rounding * (RAW_CENTER.abs() + RAW_SCALE),
        "the saved score-1 locations must carry the recording's shift, gap {location_gap:.3e}"
    );
    // The scalar map the single-score readers see is score 0's.
    let scalar = raw
        .latent_z_normalization
        .as_ref()
        .expect("a saved score-0 map");
    assert_eq!(
        (scalar.mean, scalar.sd),
        (raw_law.score_location[0], raw_law.score_scale[0]),
        "the payload's scalar score map must be the joint law's score-0 map"
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
        "[4331] |Δ log-lik| {log_likelihood_gap:.3e} (log-lik {:.6}), max |Δβ| \
         {coefficient_gap:.3e} (scale {coefficient_scale:.3}), max |ΔS| {prediction_gap:.3e}",
        standard_fit.log_likelihood
    );

    // Both fits solve one problem on scores that differ by the rounding of the
    // standardization. A coefficient gap is bounded by the square root of the
    // working precision relative to the coefficients' scale (the inner solve
    // stops at a gradient norm the outer objective sees only in second order),
    // and the log-likelihood and the survival probabilities, smooth in the
    // coefficients, by the same band on their own scale.
    let convergence_band = f64::EPSILON.sqrt();
    assert!(
        coefficient_gap <= convergence_band * coefficient_scale,
        "the time, marginal and slope coefficients must not depend on each score's units: \
         max |Δβ| {coefficient_gap:.3e} beyond {:.3e}",
        convergence_band * coefficient_scale
    );
    assert!(
        log_likelihood_gap <= convergence_band * standard_fit.log_likelihood.abs().max(1.0),
        "the log-likelihood must not depend on each score's units: |Δ| {log_likelihood_gap:.3e}"
    );
    assert!(
        prediction_gap <= convergence_band,
        "predicted survival must not depend on each score's units: max |ΔS| {prediction_gap:.3e}"
    );
}
