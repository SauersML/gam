//! Standing coverage gate (issue #1891): the Gaussian predictive (observation)
//! interval — the band for a NEW observation, `Var(y_new) = Var(μ̂) + σ̂²`.
//!
//! Distinct from the `se.fit` confidence band (which is audited by
//! `sbc_gaussian_smooth_band_coverage`): the confidence band covers the mean
//! function, the *predictive* interval must cover a fresh draw. It therefore
//! exercises the dispersion estimate σ̂ and its composition with the coefficient
//! covariance — precisely the surface #1875 ("recycled response SE instead of
//! joint posterior covariance") and #1878 ("delta-method vs posterior
//! simulation divergence") corrupt. A recycled or mis-scaled SE under-covers a
//! new observation even when the mean band looks fine.
//!
//! Model: `y ~ s(x1) + s(x2)`, family = gaussian (two smooths ⇒ dense standard
//! path). The predictive interval is requested via
//! `PredictUncertaintyOptions::includeobservation_interval`, giving
//! `observation_lower/upper = μ̂ ± z·√(Var(μ̂) + σ̂²)`. Coverage sweep: draw a
//! low-frequency additive truth, fit, then draw a genuinely NEW observation
//! y_new ~ N(f(x⋆), σ²) at one independent interior point and check it lies
//! inside the interval, audited by the shared Wilson verdict over the 80/90/95
//! sweep.
//!
//! Honest gate: only anti-conservative under-coverage fails; over-coverage is
//! reported but never gates.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};
use ndarray::Array1;

const N_TRAIN: usize = 160;
const NOISE_SD: f64 = 0.30;
const N_REPLICATIONS: usize = 120;
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];
const SEED: u64 = 0x1891_9E_D1_C7_10;
const DESIGN_SEED: u64 = 0x1891_DE_517_B1;

struct AdditiveTruth {
    center: f64,
    amp1: f64,
    freq1: f64,
    phase1: f64,
    amp2: f64,
    freq2: f64,
    phase2: f64,
}

impl AdditiveTruth {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            center: -1.0 + 2.0 * rng.uniform_open01(),
            amp1: 0.6 + 0.8 * rng.uniform_open01(),
            freq1: 0.6 + 0.8 * rng.uniform_open01(),
            phase1: rng.uniform_open01(),
            amp2: 0.6 + 0.8 * rng.uniform_open01(),
            freq2: 0.6 + 0.8 * rng.uniform_open01(),
            phase2: rng.uniform_open01(),
        }
    }

    fn eval(&self, x1: f64, x2: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        self.center
            + self.amp1 * (tau * (self.freq1 * x1 + self.phase1)).sin()
            + self.amp2 * (tau * (self.freq2 * x2 + self.phase2)).sin()
    }
}

fn simulate_dataset(
    x1: &[f64],
    x2: &[f64],
    truth: &AdditiveTruth,
    rng: &mut CalibrationRng,
) -> EncodedDataset {
    let rows: Vec<StringRecord> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| {
            let y = truth.eval(a, b) + NOISE_SD * rng.standard_normal();
            StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(
        vec!["x1".to_string(), "x2".to_string(), "y".to_string()],
        rows,
    )
    .expect("encode gaussian replication dataset")
}

/// The predictive (observation) interval at every training row and level.
fn predictive_interval(fit: &FitResult, level: f64) -> (Array1<f64>, Array1<f64>) {
    let FitResult::Standard(standard) = fit else {
        panic!(
            "gaussian `y ~ s(x1) + s(x2)` must fit through the dense standard path, \
             got a different FitResult variant"
        );
    };
    let design = standard.design.design.clone();
    let n = design.nrows();
    let beta = standard.fit.beta.view();
    let offset = Array1::<f64>::zeros(n);
    let family = standard
        .fit
        .likelihood_family
        .clone()
        .expect("fit records its likelihood family");
    let options = PredictUncertaintyOptions {
        confidence_level: level,
        covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
        mean_interval_method: MeanIntervalMethod::TransformEta,
        includeobservation_interval: true,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..PredictUncertaintyOptions::default()
    };
    let result =
        predict_gamwith_uncertainty(design, beta, offset.view(), family, &standard.fit, &options)
            .expect("gaussian predictive-interval prediction");
    let lower = result
        .observation_lower
        .expect("gaussian family exposes a predictive observation interval");
    let upper = result
        .observation_upper
        .expect("gaussian family exposes a predictive observation interval");
    (lower, upper)
}

#[test]
fn gaussian_predictive_interval_covers_new_observation_at_nominal() {
    let mut design_rng = CalibrationRng::new(DESIGN_SEED);
    let x1: Vec<f64> = (0..N_TRAIN).map(|_| design_rng.uniform_open01()).collect();
    let x2: Vec<f64> = (0..N_TRAIN).map(|_| design_rng.uniform_open01()).collect();
    let interior: Vec<usize> = (0..N_TRAIN)
        .filter(|&i| x1[i] > 0.1 && x1[i] < 0.9 && x2[i] > 0.1 && x2[i] < 0.9)
        .collect();
    assert!(
        interior.len() >= 20,
        "too few interior evaluation points ({}) — widen the design",
        interior.len()
    );

    let mut rng = CalibrationRng::new(SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for _ in 0..N_REPLICATIONS {
        let truth = AdditiveTruth::draw(&mut rng);
        let data = simulate_dataset(&x1, &x2, &truth, &mut rng);
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let fit =
            fit_from_formula("y ~ s(x1) + s(x2)", &data, &config).expect("gaussian additive fit");

        let pick = (rng.uniform_open01() * interior.len() as f64) as usize % interior.len();
        let j = interior[pick];
        // A genuinely NEW observation at the evaluation point, independent of the
        // training data — this is what the predictive interval must cover.
        let y_new = truth.eval(x1[j], x2[j]) + NOISE_SD * rng.standard_normal();

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let (lower, upper) = predictive_interval(&fit, level);
            assert!(
                lower[j].is_finite() && upper[j].is_finite() && upper[j] >= lower[j],
                "degenerate predictive interval at eval point {j} (level {level}): [{}, {}]",
                lower[j],
                upper[j]
            );
            if upper[j] - lower[j] > 0.0 {
                positive_width_seen = true;
            }
            if lower[j] <= y_new && y_new <= upper[j] {
                hits[level_idx] += 1;
            }
        }
    }

    assert!(
        positive_width_seen,
        "every predictive interval had zero width — the uncertainty surface is \
         not producing a real interval"
    );

    let mut failures = Vec::new();
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], N_REPLICATIONS, level);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "level {level}: empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], \
                 nominal ABOVE the CI by {:.4} — anti-conservative (recycled/mis-scaled \
                 predictive SE, the #1875/#1878 signature)",
                verdict.empirical,
                verdict.hits,
                verdict.replications,
                verdict.ci_lo,
                verdict.ci_hi,
                -verdict.slack(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "gaussian predictive interval under-covers a new observation:\n{}",
        failures.join("\n")
    );
}
