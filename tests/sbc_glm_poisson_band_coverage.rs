//! Standing coverage gate (issue #1891): the Poisson (log-link) GLM smooth
//! response-scale credible band for the mean count.
//!
//! Companion to the binomial gate (`sbc_glm_binomial_band_coverage`): the same
//! coverage-sweep mechanism against a second exponential-family band, so the
//! standing harness covers count as well as binary GLM uncertainty. Targets the
//! same #1870/#1871 genus (response-scale band under-coverage) on the log link.
//!
//! Model: `y ~ s(x)` with family = poisson. A non-Gaussian family always fits
//! the dense standard path (`FitResult::Standard`). On the log link the
//! mean-scale band is exp(η) transformed, so `mean_lower/mean_upper` is the
//! credible band for the mean count λ(x) = exp(η(x)). The gate checks whether
//! the true mean λ(x⋆) at one independent interior evaluation point lies inside
//! the band, audited by the shared Wilson verdict over the 80/90/95 sweep.
//!
//! Honest gate: only anti-conservative under-coverage fails; over-coverage is
//! reported but never gates. Low-frequency truths keep smoother bias small so a
//! calibrated band sits at or above nominal.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};
use ndarray::Array1;

const N_TRAIN: usize = 250;
const N_REPLICATIONS: usize = 150;
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];
const SEED: u64 = 0x1891_B0_15_50_A0;

/// A smooth log-mean truth η(x) drawn from the prior: a low-frequency sinusoid
/// centred so λ = exp(η) stays in a well-behaved count range (roughly 2–15).
struct SmoothLogMean {
    center: f64,
    amplitude: f64,
    frequency: f64,
    phase: f64,
}

impl SmoothLogMean {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            // exp(center) ≈ 3.0 .. 8.2 for center ∈ [1.1, 2.1].
            center: 1.1 + 1.0 * rng.uniform_open01(),
            amplitude: 0.4 + 0.6 * rng.uniform_open01(),
            frequency: 0.7 + 0.8 * rng.uniform_open01(),
            phase: rng.uniform_open01(),
        }
    }

    fn eta(&self, x: f64) -> f64 {
        self.center
            + self.amplitude * (std::f64::consts::TAU * (self.frequency * x + self.phase)).sin()
    }
}

/// Knuth's algorithm for a Poisson draw from the harness's uniform stream — the
/// harness RNG exposes only uniform/normal primitives, and Knuth is exact for
/// the small λ used here.
fn poisson_draw(lambda: f64, rng: &mut CalibrationRng) -> f64 {
    let l = (-lambda).exp();
    let mut k = 0u64;
    let mut p = 1.0f64;
    loop {
        k += 1;
        p *= rng.uniform_open01();
        if p <= l {
            break;
        }
    }
    (k - 1) as f64
}

fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn simulate_dataset(x: &[f64], truth: &SmoothLogMean, rng: &mut CalibrationRng) -> EncodedDataset {
    let rows: Vec<StringRecord> = x
        .iter()
        .map(|&xi| {
            let lambda = truth.eta(xi).exp();
            let y = poisson_draw(lambda, rng);
            StringRecord::from(vec![xi.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode poisson replication dataset")
}

fn response_band(fit: &FitResult, level: f64) -> (Array1<f64>, Array1<f64>) {
    let FitResult::Standard(standard) = fit else {
        panic!(
            "poisson `y ~ s(x)` must fit through the dense standard GLM path, \
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
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..PredictUncertaintyOptions::default()
    };
    let result =
        predict_gamwith_uncertainty(design, beta, offset.view(), family, &standard.fit, &options)
            .expect("poisson uncertainty prediction");
    (result.mean_lower, result.mean_upper)
}

#[test]
fn poisson_glm_mean_band_covers_truth_at_nominal() {
    let x = training_grid(N_TRAIN);
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;
    let span = interior_hi - interior_lo;

    let mut rng = CalibrationRng::new(SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for _ in 0..N_REPLICATIONS {
        let truth = SmoothLogMean::draw(&mut rng);
        let data = simulate_dataset(&x, &truth, &mut rng);
        let config = FitConfig {
            family: Some("poisson".to_string()),
            ..FitConfig::default()
        };
        let fit = fit_from_formula("y ~ s(x)", &data, &config).expect("poisson smooth fit");

        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        let lambda_true = truth.eta(x[j]).exp();

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let (lower, upper) = response_band(&fit, level);
            assert!(
                lower[j].is_finite() && upper[j].is_finite() && upper[j] >= lower[j],
                "degenerate band at eval point {j} (level {level}): [{}, {}]",
                lower[j],
                upper[j]
            );
            if upper[j] - lower[j] > 0.0 {
                positive_width_seen = true;
            }
            if lower[j] <= lambda_true && lambda_true <= upper[j] {
                hits[level_idx] += 1;
            }
        }
    }

    assert!(
        positive_width_seen,
        "every credible band had zero width — the uncertainty surface is not \
         producing a real band"
    );

    let mut failures = Vec::new();
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], N_REPLICATIONS, level);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "level {level}: empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], \
                 nominal ABOVE the CI by {:.4} — anti-conservative (the #1870/#1871 signature)",
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
        "poisson GLM mean credible band under-covers the truth:\n{}",
        failures.join("\n")
    );
}
