//! Standing coverage gate (issue #1891): the Gaussian additive-smooth
//! confidence band (the `se.fit` credible band).
//!
//! One instantiation of the standing simulation-based-calibration harness
//! (`gam_test_support::calibration`) against the Gaussian smooth CI surface —
//! the surface at the centre of the open UQ cluster #1870/#1871, where a
//! penalized-smooth confidence band under-covers the true function.
//!
//! Model: `y ~ s(x1) + s(x2)` with family = gaussian (identity link). Two
//! separate 1-D smooths force the dense standard fit path (the exact O(n)
//! spline scan fires only for a *single* 1-D Gaussian smooth, so an additive
//! two-smooth model is always a `FitResult::Standard`), whose Bayesian
//! covariance is the object under audit. On the identity link the mean-scale
//! band equals the η-scale band, so `mean_lower/mean_upper` is precisely the
//! Wood/Nychka `se.fit` confidence band.
//!
//! Coverage sweep (audit mode 1): draw a smooth additive truth
//! f(x1, x2) = c + g1(x1) + g2(x2) from a prior over low-frequency functions,
//! simulate y = f + N(0, σ²), fit, and check whether the *true predictor*
//! f(x1⋆, x2⋆) at one independent interior evaluation point lies inside the
//! band. One Bernoulli coverage trial per replication (independent — a fresh
//! interior point each fit), audited by the shared Wilson verdict at the
//! harness's fixed 1% false-positive rate.
//!
//! The truths are low-frequency (well inside the span of a penalized smooth) so
//! smoother bias is small and a correctly-calibrated band sits at or above
//! nominal — a healthy library keeps this gate quiet. Following the issue, only
//! anti-conservative under-coverage (the #1870/#1871 collapse) gates the build;
//! over-coverage is reported but never gates. This audits *average* coverage
//! over the domain (the evaluation point is drawn fresh each replication), which
//! is the Nychka sense in which a Bayesian smooth band is calibrated.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};
use ndarray::Array1;

/// Training rows per replication.
const N_TRAIN: usize = 160;

/// Observation noise standard deviation. Moderate SNR against the ~unit-scale
/// additive signal so the smooth is comfortably recoverable (low bias).
const NOISE_SD: f64 = 0.30;

/// Coverage replications (one independent trial each). Wilson half-width at the
/// tightest level (0.95) is ≈ z·√(0.95·0.05/R) ≈ 0.051 — resolves the historical
/// 0.157 / 0.731 collapses with wide margin without spuriously gating a
/// calibrated band.
const N_REPLICATIONS: usize = 120;

/// The three nominal levels audited (issue's 80/90/95 sweep).
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];

/// Fixed seed for the whole gate (determinism requirement).
const SEED: u64 = 0x1891_6A_C0DE_B77;
/// Separate fixed seed for the (once) covariate design draw, kept distinct from
/// the replication stream so the two never alias.
const DESIGN_SEED: u64 = 0x1891_DE_517_A0;

/// A smooth additive truth f(x1, x2) = center + g1(x1) + g2(x2), each component
/// a low-frequency sinusoid drawn from the prior.
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

/// Simulate the `y ~ s(x1) + s(x2)` dataset for one replication at a fixed
/// covariate design, with fresh Gaussian noise from the harness RNG.
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

/// The `se.fit` confidence band (identity link ⇒ mean band = η band) at every
/// training row and the requested nominal level.
fn confidence_band(fit: &FitResult, level: f64) -> (Array1<f64>, Array1<f64>) {
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
    // Audit the core Vp confidence band: the central `η̂ ± z·se` band with the
    // optional predictor corrections off, so the gate measures the band's own
    // calibration — the Wood/Nychka `se.fit` band #1871 compares to INLA.
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
            .expect("gaussian uncertainty prediction");
    (result.mean_lower, result.mean_upper)
}

#[test]
fn gaussian_smooth_confidence_band_covers_truth_at_nominal() {
    // Fixed covariate design, drawn once from its own RNG stream. Independent
    // uniform coordinates so the two additive smooths are not collinear.
    let mut design_rng = CalibrationRng::new(DESIGN_SEED);
    let x1: Vec<f64> = (0..N_TRAIN).map(|_| design_rng.uniform_open01()).collect();
    let x2: Vec<f64> = (0..N_TRAIN).map(|_| design_rng.uniform_open01()).collect();
    // Interior evaluation indices: both coordinates away from the outer 10%,
    // so the coverage signal is not dominated by known smoother edge-effects.
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

        // One independent interior evaluation point per replication.
        let pick = (rng.uniform_open01() * interior.len() as f64) as usize % interior.len();
        let j = interior[pick];
        let f_true = truth.eval(x1[j], x2[j]);

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let (lower, upper) = confidence_band(&fit, level);
            assert!(
                lower[j].is_finite() && upper[j].is_finite() && upper[j] >= lower[j],
                "degenerate band at eval point {j} (level {level}): [{}, {}]",
                lower[j],
                upper[j]
            );
            if upper[j] - lower[j] > 0.0 {
                positive_width_seen = true;
            }
            if lower[j] <= f_true && f_true <= upper[j] {
                hits[level_idx] += 1;
            }
        }
    }

    assert!(
        positive_width_seen,
        "every confidence band had zero width — the uncertainty surface is not \
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
        "gaussian smooth confidence band under-covers the truth:\n{}",
        failures.join("\n")
    );
}
