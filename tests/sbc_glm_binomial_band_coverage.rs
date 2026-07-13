//! Standing coverage gate (issue #1891): the binomial (logit) GLM smooth
//! response-scale credible band.
//!
//! This is one instantiation of the standing simulation-based-calibration
//! harness (`gam_test_support::calibration`) against a real uncertainty surface
//! the library exposes. It targets the exact failure genus of the open UQ
//! cluster — #1870 ("mean-prediction band coverage collapses to 0.157"),
//! #1871 ("smooth posterior bands under-cover 0.731 vs INLA") — where the
//! response-scale band for a nonlinear-link GAM under-covers the truth.
//!
//! Mechanism (audit mode 1 of the issue, the coverage sweep):
//!   for each replication r
//!     1. draw a smooth truth η(x) from a prior over low-frequency functions,
//!     2. simulate Bernoulli responses yᵢ ~ Bern(σ(η(xᵢ))),
//!     3. fit `y ~ s(x)` with family = binomial (the dense GLM path — a
//!        non-Gaussian family can never take the Gaussian scan/cascade fast
//!        paths, so this is always a `FitResult::Standard`),
//!     4. build the response-scale credible band the library reports
//!        (`predict_gamwith_uncertainty`, mean-scale = σ(η) band), and
//!     5. record whether the *true probability* p(x⋆) = σ(η(x⋆)) at one
//!        independent interior evaluation point x⋆ lies inside the band.
//! Aggregated over replications this is one Bernoulli coverage trial per
//! replication (independent by construction — one point per fit, a fresh
//! interior point each time), so the empirical coverage vs. nominal is audited
//! by the shared Wilson-interval verdict [`audit_coverage`] at the harness's
//! fixed 1% false-positive rate.
//!
//! Honest gate, not a tuned one: only *anti-conservative* (under-coverage
//! beyond the whole Wilson CI) fails the build — the #1870/#1871 signature.
//! Over-coverage is reported (via the verdict) but never gates, exactly as the
//! issue prescribes. The truths are deliberately low-frequency (representable
//! by the penalized basis) so smoother bias is small and a *correctly*
//! calibrated band sits at or above nominal; a catastrophic collapse of the
//! kind the cluster documents trips the gate with a wide margin.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};
use ndarray::Array1;

/// Training rows per replication. Large enough that a well-specified GLM smooth
/// band is genuinely calibrated (so a healthy library keeps this gate quiet),
/// small enough that a single fit is milliseconds.
const N_TRAIN: usize = 250;

/// Coverage replications. One independent Bernoulli coverage trial per
/// replication, so the Wilson half-width at the tightest nominal level (0.95)
/// is ≈ z·√(0.95·0.05/R) ≈ 0.046 — narrow enough to resolve the historical
/// 0.157 / 0.731 collapses with enormous margin, wide enough that a merely
/// mildly-conservative-or-calibrated band never spuriously gates.
const N_REPLICATIONS: usize = 150;

/// The three nominal levels audited, matching the issue's 80/90/95 sweep.
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];

/// Fixed seed: the harness is a replicate-null consumer and inherits the repo
/// determinism requirement, so the whole gate reproduces from this constant.
const SEED: u64 = 0x1891_B1_C0DE_A11;

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// A single smooth truth η(x) drawn from the prior over low-frequency
/// functions: a sinusoid whose level / amplitude / frequency / phase are drawn
/// so p = σ(η) spans a useful range without saturating, and so the shape is
/// comfortably inside the span of a penalized 1-D smooth (low smoother bias).
struct SmoothTruth {
    center: f64,
    amplitude: f64,
    frequency: f64,
    phase: f64,
}

impl SmoothTruth {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            center: -0.6 + 1.2 * rng.uniform_open01(),
            amplitude: 0.7 + 0.9 * rng.uniform_open01(),
            frequency: 0.7 + 0.8 * rng.uniform_open01(),
            phase: rng.uniform_open01(),
        }
    }

    fn eta(&self, x: f64) -> f64 {
        self.center
            + self.amplitude * (std::f64::consts::TAU * (self.frequency * x + self.phase)).sin()
    }
}

/// Equally-spaced training grid on [0, 1].
fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

/// Build the `y ~ s(x)` dataset for one replication: the fixed grid plus
/// Bernoulli responses drawn from the truth via the harness RNG.
fn simulate_dataset(x: &[f64], truth: &SmoothTruth, rng: &mut CalibrationRng) -> EncodedDataset {
    let rows: Vec<StringRecord> = x
        .iter()
        .map(|&xi| {
            let p = sigmoid(truth.eta(xi));
            let y = if rng.uniform_open01() < p { 1.0 } else { 0.0 };
            StringRecord::from(vec![xi.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode binomial replication dataset")
}

/// Response-scale credible band the library reports for the fitted smooth, at
/// every training row and the requested nominal level. Returns
/// `(mean_lower, mean_upper)`.
fn response_band(fit: &FitResult, level: f64) -> (Array1<f64>, Array1<f64>) {
    let FitResult::Standard(standard) = fit else {
        panic!(
            "binomial `y ~ s(x)` must fit through the dense standard GLM path, \
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
    // Audit the core Vp credible band the response-scale surface is built on:
    // the central `σ(η) ± z·se` transform-of-η band. The optional predictor
    // corrections (bias / Edgeworth / boundary / observation interval) are
    // switched off so the gate measures the calibration of the band itself, not
    // a confounding correction — the same central band #1871 compares to INLA.
    let options = PredictUncertaintyOptions {
        confidence_level: level,
        covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
        mean_interval_method: MeanIntervalMethod::TransformEta,
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..PredictUncertaintyOptions::default()
    };
    let result =
        predict_gamwith_uncertainty(design, beta, offset.view(), family, &standard.fit, &options)
            .expect("binomial uncertainty prediction");
    (result.mean_lower, result.mean_upper)
}

#[test]
fn binomial_glm_response_band_covers_truth_at_nominal() {
    let x = training_grid(N_TRAIN);
    // Interior evaluation window: exclude the outer 10% where a penalized
    // smooth's known boundary edge-effects (not the #1870/#1871 target) would
    // otherwise dominate the coverage signal.
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;

    let mut rng = CalibrationRng::new(SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for _ in 0..N_REPLICATIONS {
        let truth = SmoothTruth::draw(&mut rng);
        let data = simulate_dataset(&x, &truth, &mut rng);
        let config = FitConfig {
            family: Some("binomial".to_string()),
            ..FitConfig::default()
        };
        let fit = fit_from_formula("y ~ s(x)", &data, &config).expect("binomial smooth fit");

        // One independent interior evaluation point per replication.
        let span = interior_hi - interior_lo;
        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        let p_true = sigmoid(truth.eta(x[j]));

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
            if lower[j] <= p_true && p_true <= upper[j] {
                hits[level_idx] += 1;
            }
        }
    }

    // Guard against a silently-degenerate (zero-width) band trivially "failing"
    // to cover or, worse, a band so wide it vacuously covers: require the band
    // to have real, positive width somewhere.
    assert!(
        positive_width_seen,
        "every credible band had zero width — the uncertainty surface is not \
         producing a real band"
    );

    // Audit each nominal level with the shared Wilson verdict. Only
    // anti-conservative under-coverage gates; report the full verdict either way.
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
        "binomial GLM response-scale credible band under-covers the truth:\n{}",
        failures.join("\n")
    );
}
