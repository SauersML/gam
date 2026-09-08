//! Standing coverage gate (issue #1891): the Gaussian additive-smooth credible
//! band audited under BOTH covariance modes — the conditional `H⁻¹` band and the
//! smoothing-corrected `H⁻¹ + J·Var(ρ̂)·Jᵀ` band.
//!
//! The sibling gate `sbc_gaussian_smooth_band_coverage` audits only the default
//! required `SmoothingCorrected` mode. The open UQ cluster splits across
//! the two covariance modes:
//!   * #1870 — the mean-prediction band coverage collapses to 0.157: the
//!     CONDITIONAL band (no ρ̂ correction) under-covers because it ignores
//!     smoothing-parameter uncertainty.
//!   * #1871 — the smooth posterior bands under-cover 0.731 vs INLA: the
//!     SMOOTHING-CORRECTED band still under-covers relative to a full posterior.
//! Registering and gating BOTH modes here is the closability evidence for
//! #1870/#1871 — a red arm names exactly which covariance mode is miscalibrated.
//!
//! Model: `y ~ s(x1) + s(x2)`, family = gaussian (two smooths ⇒ dense standard
//! path). On the identity link the mean-scale band equals the η-scale band, so
//! `mean_lower/mean_upper` IS the Wood/Nychka `se.fit` band under whichever
//! covariance mode is requested. Coverage sweep: draw a low-frequency additive
//! truth from a prior, simulate y = f + N(0, σ²), fit, and check whether the
//! true predictor f(x⋆) at one independent interior evaluation point lies inside
//! each mode's band, audited by the shared Wilson verdict over the 80/90/95
//! sweep. Only anti-conservative under-coverage gates; over-coverage is reported.

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
const SEED: u64 = 0x1891_C0_7A_9E_5D;
const DESIGN_SEED: u64 = 0x1891_DE_517_C0;

/// The band configurations under audit: covariance mode × bias correction.
///
/// The bias-correction axis is load-bearing for #1870. When
/// `apply_bias_correction` is on (the DEFAULT, user-facing band), the reported
/// centre shifts to β_BC = β̂ + H⁻¹S(β̂ − μ); the matching covariance is
/// A·V·Aᵀ with A = I + H⁻¹S. The fit applies A to the SMOOTHING-CORRECTED
/// covariance (optimizer.rs:3193) but stores the CONDITIONAL covariance as raw
/// Vb — so a bias-corrected CONDITIONAL band is centred at β_BC yet reports the
/// uncertainty of the shrunken mode β̂, the over-narrow band #1870 documents.
/// This gate exercises all four cells so a red one names exactly the
/// (mode, bias) configuration that under-covers.
struct BandConfig {
    covariance_mode: InferenceCovarianceMode,
    apply_bias_correction: bool,
    label: &'static str,
}

const CONFIGS: [BandConfig; 4] = [
    BandConfig {
        covariance_mode: InferenceCovarianceMode::Conditional,
        apply_bias_correction: false,
        label: "conditional core (bias off)",
    },
    BandConfig {
        covariance_mode: InferenceCovarianceMode::Conditional,
        apply_bias_correction: true,
        label: "conditional bias-corrected (#1870 oracle)",
    },
    BandConfig {
        covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
        apply_bias_correction: false,
        label: "smoothing-corrected core (#1871)",
    },
    BandConfig {
        covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
        apply_bias_correction: true,
        label: "smoothing-corrected bias-corrected (default user band)",
    },
];

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

/// The credible band at every training row under the requested configuration.
///
/// Only the covariance mode and the bias-correction flag are varied; the
/// Edgeworth / boundary / OOD modifiers are held OFF (and no-op here anyway
/// without their inputs) so the gate isolates the covariance-mode × bias axis.
fn confidence_band(fit: &FitResult, level: f64, config: &BandConfig) -> (Array1<f64>, Array1<f64>) {
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
        covariance_mode: config.covariance_mode,
        mean_interval_method: MeanIntervalMethod::TransformEta,
        includeobservation_interval: false,
        apply_bias_correction: config.apply_bias_correction,
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
fn gaussian_mean_band_covers_truth_under_both_covariance_modes() {
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
    // hits[config][level].
    let mut hits = [[0usize; NOMINAL_LEVELS.len()]; CONFIGS.len()];
    let mut positive_width_seen = false;

    for _ in 0..N_REPLICATIONS {
        let truth = AdditiveTruth::draw(&mut rng);
        let data = simulate_dataset(&x1, &x2, &truth, &mut rng);
        let fit_config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let fit = fit_from_formula("y ~ s(x1) + s(x2)", &data, &fit_config)
            .expect("gaussian additive fit");

        let pick = (rng.uniform_open01() * interior.len() as f64) as usize % interior.len();
        let j = interior[pick];
        let f_true = truth.eval(x1[j], x2[j]);

        for (config_idx, config) in CONFIGS.iter().enumerate() {
            for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
                let (lower, upper) = confidence_band(&fit, level, config);
                assert!(
                    lower[j].is_finite() && upper[j].is_finite() && upper[j] >= lower[j],
                    "degenerate band at eval point {j} ({}, level {level}): [{}, {}]",
                    config.label,
                    lower[j],
                    upper[j]
                );
                if upper[j] - lower[j] > 0.0 {
                    positive_width_seen = true;
                }
                if lower[j] <= f_true && f_true <= upper[j] {
                    hits[config_idx][level_idx] += 1;
                }
            }
        }
    }

    assert!(
        positive_width_seen,
        "every confidence band had zero width — the uncertainty surface is not \
         producing a real band"
    );

    let mut failures = Vec::new();
    for (config_idx, config) in CONFIGS.iter().enumerate() {
        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let verdict = audit_coverage(hits[config_idx][level_idx], N_REPLICATIONS, level);
            if verdict.class == CoverageClass::AntiConservative {
                failures.push(format!(
                    "{} @ level {level}: empirical={:.4} (hits {}/{}), \
                     Wilson CI=[{:.4},{:.4}], nominal ABOVE the CI by {:.4} — anti-conservative",
                    config.label,
                    verdict.empirical,
                    verdict.hits,
                    verdict.replications,
                    verdict.ci_lo,
                    verdict.ci_hi,
                    -verdict.slack(),
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "gaussian mean band under-covers the truth (per covariance-mode × bias config):\n{}",
        failures.join("\n")
    );
}
