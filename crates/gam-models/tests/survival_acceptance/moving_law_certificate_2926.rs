//! gam#2926 acceptance, survival route: where the conditional law of the score
//! moves on the marginal-index span, the survival marginal-slope default fits the
//! location-scale Gaussian law and certifies it against the Gaussian,
//! location-scale empirical and local laws by their cross-fitted excess anchoring
//! loss, each row's two anchors read at its partner row's times.
//!
//! The model is `S(t | x, z) = Φ(−(α(q(t, x), x) + b·z))`, `q(t, x) = γ₀ + γ₁·log t +
//! β_x·x`, with `α` defined by the anchoring equation on the TRUE law of `z | x`:
//! for a normal mixture `Σ_j p_j(x) Φ(−(α + b·μ_j(x))/√(1 + b²σ_j²)) = Φ(−q)`.
//!
//! 1. **A law whose shape moves is certified local**, and the local fit is closer
//!    to the true conditional survival than the declared location-scale law's.
//!    `z | x` is a two-normal mixture whose weight on the upper component is
//!    `Φ(1.5·x)`, which no location-scale law follows.
//! 2. **A location-scale law is certified on its Gaussian residual.** `z | x` is
//!    `N(0.6·x, 0.64)`; the location-scale Gaussian law is the true law, so no
//!    re-solve happens. A row's own exit time is an outcome of its score, and a
//!    certificate that read the anchors there chose the location-scale empirical
//!    law on a Gaussian-residual fixture (gam#2949, job 1249280); this is the pin.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_math::probability::normal_cdf;
use gam_models::bms::{LatentLawConsumed, MovingLawArm, MovingLawCertificate};
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use ndarray::Array1;
use std::collections::HashMap;

const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;
const BETA_X: f64 = 0.4;
const SLOPE: f64 = 0.9;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Root of a monotone function on `[low, high]` by bisection, `increasing` giving
/// its direction.
fn root(f: impl Fn(f64) -> f64, target: f64, increasing: bool, low: f64, high: f64) -> f64 {
    let (mut low, mut high) = (low, high);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if (f(mid) < target) == increasing {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// `(weight, mean, sd)` of each normal component of `z | x`.
type Components = Vec<(f64, f64, f64)>;

/// The true conditional survival `S(t | x, z)` of the anchored model on `law(x)`.
fn survival(components: &Components, x: f64, z: f64, t: f64) -> f64 {
    let q = LOCATION_LEVEL + LOCATION_TREND * t.ln() + BETA_X * x;
    let alpha = root(
        |alpha| {
            components
                .iter()
                .map(|&(weight, mean, sd)| {
                    weight * normal_cdf(-(alpha + SLOPE * mean) / (1.0 + SLOPE * SLOPE * sd * sd).sqrt())
                })
                .sum()
        },
        normal_cdf(-q),
        false,
        -40.0,
        40.0,
    );
    normal_cdf(-(alpha + SLOPE * z))
}

struct Fixture {
    data: gam_data::EncodedDataset,
    /// The true `S(t_i | x_i, z_i)` at every row's recorded time.
    truth: Vec<f64>,
}

fn fixture(n: usize, seed: u64, law: impl Fn(f64) -> Components) -> Fixture {
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for _ in 0..n {
        let x = next_gauss(&mut state);
        let components = law(x);
        let pick = next_unit(&mut state);
        let mut cumulative = 0.0;
        let &(_, mean, sd) = components
            .iter()
            .find(|&&(weight, _, _)| {
                cumulative += weight;
                pick < cumulative
            })
            .unwrap_or_else(|| components.last().expect("a component"));
        let z = mean + sd * next_gauss(&mut state);
        let u = next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6);
        let censor = 0.35 + 5.0 * next_unit(&mut state);
        // `S(t | x, z)` decreases in `t`: invert `S(T) = u` for `log T`.
        let log_time = root(|log_t| survival(&components, x, z, log_t.exp()), u, false, -6.0, 6.0);
        let event_time = log_time.exp();
        let (time, event) = if event_time <= censor {
            (event_time, 1u8)
        } else {
            (censor, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        truth.push(survival(&components, x, z, time));
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            z.to_string(),
            x.to_string(),
        ]));
    }
    let headers = ["time", "event", "z", "x"].iter().map(|s| s.to_string()).collect();
    Fixture {
        data: encode_recordswith_inferred_schema(headers, rows)
            .expect("encode the #2926 survival moving-law fixture"),
        truth,
    }
}

fn config(latent_measure: Option<&str>) -> FitConfig {
    FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        time_num_internal_knots: 3,
        latent_measure: latent_measure.map(str::to_string),
        ..FitConfig::default()
    }
}

struct Fitted {
    label: &'static str,
    consumed: LatentLawConsumed,
    /// `|Ŝ(t_i | x_i, z_i) − S(t_i | x_i, z_i)|` per row.
    errors: Vec<f64>,
}

fn fit(fixture: &Fixture, latent_measure: Option<&str>) -> Fitted {
    let payload = fit_formula_to_payload(
        "Surv(time, event) ~ x".to_string(),
        &fixture.data,
        &config(latent_measure),
    )
    .unwrap_or_else(|e| panic!("survival marginal-slope fit: {e}"));
    let consumed = payload
        .latent_law_consumed
        .clone()
        .expect("a saved marginal-slope model records the law it consumed");
    let model = FittedModel::from_payload(payload);
    let col_map: HashMap<String, usize> = fixture
        .data
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = Array1::<f64>::zeros(fixture.data.values.nrows());
    let prediction = predict_survival(
        SurvivalPredictRequest {
            model: &model,
            data: fixture.data.values.view(),
            col_map: &col_map,
            training_headers: Some(&fixture.data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("survival marginal-slope prediction at the training rows");
    let errors = (0..fixture.truth.len())
        .map(|row| (prediction.survival[[row, 0]] - fixture.truth[row]).abs())
        .collect();
    Fitted {
        label: consumed.label(),
        consumed,
        errors,
    }
}

fn certificate(fitted: &Fitted) -> &MovingLawCertificate {
    let LatentLawConsumed::EstimatedMovingLaw {
        certificate: Some(certificate),
        ..
    } = &fitted.consumed
    else {
        panic!("a moving conditional law must be certified; got {}", fitted.label)
    };
    eprintln!(
        "[2926 survival moving law] {}: {}",
        fitted.label,
        certificate
            .arms
            .iter()
            .map(|s| format!(
                "{:?} loss={:.4e} d={:+.3e} row_se={:.3e} fold_se={:.3e}",
                s.arm, s.loss, s.difference, s.row_se, s.fold_se
            ))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    certificate
}

/// The mean of `a − b` over the rows and its standard error.
fn paired(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len() as f64;
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let mean = diffs.iter().sum::<f64>() / n;
    let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

#[test]
fn a_survival_law_whose_shape_moves_is_certified_local_2926() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    let fixture = fixture(10_000, 0x2926_5A9E_0000_0002, |x| {
        let upper = normal_cdf(1.5 * x);
        vec![(1.0 - upper, -0.8 + 0.2 * x, 0.45), (upper, 1.3 + 0.2 * x, 0.35)]
    });
    let default = fit(&fixture, None);
    let certificate = certificate(&default);
    let location_scale = fit(&fixture, Some("conditional-location-scale"));
    let (gap, gap_se) = paired(&default.errors, &location_scale.errors);
    let mean = |errors: &[f64]| errors.iter().sum::<f64>() / errors.len() as f64;
    eprintln!(
        "[2926 survival moving law] mean |Ŝ − S|: default ({}) {:.5}, declared location-scale \
         {:.5}; paired difference {gap:+.5} (se {gap_se:.5})",
        default.label,
        mean(&default.errors),
        mean(&location_scale.errors),
    );
    assert_eq!(default.label, "estimated-local");
    // gam#2926: a shape that moves leaves the pooled residual ζ a mixture, which fails
    // the adequacy screen, so the location-scale Gaussian arm is not a candidate and
    // the fit starts on the location-scale empirical arm.
    let location_scale_gaussian = certificate
        .arms
        .iter()
        .find(|score| score.arm == MovingLawArm::LocationScaleGaussian)
        .expect("the location-scale Gaussian arm is scored");
    assert!(
        location_scale_gaussian.adequacy.is_some() && !location_scale_gaussian.admissible(),
        "a moving shape's ζ must fail the screen: {location_scale_gaussian:?}"
    );
    assert_eq!(certificate.fitted, MovingLawArm::LocationScaleEmpirical);
    assert_eq!(certificate.argmin, MovingLawArm::Local);
    assert_eq!(certificate.chosen, MovingLawArm::Local);
    for score in certificate.arms.iter().filter(|s| s.arm != MovingLawArm::Local) {
        assert!(
            score.difference > score.rule_se(),
            "the local law must beat the {:?} law by more than one paired standard error: \
             {score:?}",
            score.arm
        );
    }
    assert!(
        gap < -gap_se,
        "the local fit must be closer to the true conditional survival than the declared \
         location-scale law's: paired difference {gap:+.5} (se {gap_se:.5})"
    );
}

#[test]
fn a_survival_location_scale_law_is_certified_on_its_gaussian_residual_2926() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    let fixture = fixture(6_000, 0x2926_1055_0000_0002, |x| vec![(1.0, 0.6 * x, 0.8)]);
    let default = fit(&fixture, None);
    let certificate = certificate(&default);
    assert_eq!(default.label, "estimated-location-scale-gaussian");
    assert_eq!(certificate.fitted, MovingLawArm::LocationScaleGaussian);
    assert_eq!(
        certificate.chosen,
        MovingLawArm::LocationScaleGaussian,
        "the true law is location-scale with a Gaussian residual: {certificate:?}"
    );
    // gam#2926: the arm is a candidate because its residual passes the adequacy
    // screen, so the admissibility rule admits a Gaussian arm the data support and
    // is not a ban on Gaussian arms.
    let arm = certificate
        .arms
        .iter()
        .find(|score| score.arm == MovingLawArm::LocationScaleGaussian)
        .expect("the location-scale Gaussian arm is scored");
    assert!(
        arm.adequacy.is_some() && arm.admissible(),
        "a Gaussian residual must be screened and admitted: {arm:?}"
    );
}
