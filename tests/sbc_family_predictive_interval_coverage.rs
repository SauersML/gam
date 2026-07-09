//! Standing coverage gate (issue #1891): the predictive (observation) interval
//! for every non-Gaussian response family that emits one.
//!
//! The Gaussian predictive interval is gated by
//! `sbc_gaussian_predictive_interval_coverage`; the location-scale one by
//! `sbc_location_scale_predictive_coverage`. This file closes the remaining
//! per-family gap the #1891 registry enumerated as PENDING: Poisson,
//! Negative-Binomial, Gamma, Beta, Tweedie, and Binomial. Each family's
//! `family_observation_band` (gam-predict lib.rs) composes estimation
//! uncertainty `SE(μ̂)²` with the family's conditional response variance
//! `Var(Y|μ)` and reads *skew-correct* equal-tailed quantiles — the surface a
//! recycled / mis-scaled / wrong-family predictive SE (#1875/#1878) or a
//! dropped skew correction (#817/#1193/#1194) corrupts. A wrong composition
//! under-covers a genuinely NEW observation even when the mean band looks fine.
//!
//! Mechanism (shared with the Gaussian gate): draw a low-frequency smooth truth
//! from the prior, simulate a training set from the family, fit `y ~ s(x)`, then
//! draw a genuinely NEW response y_new ~ family(μ_true(x⋆)) at one independent
//! interior point and check it lands inside the requested predictive interval.
//! Empirical coverage over the 80/90/95 sweep is adjudicated by the shared
//! Wilson verdict: only anti-conservative under-coverage gates; discreteness /
//! skew over-coverage is reported but never gates.
//!
//! Determinism: a fixed per-family seed threads truth draw, simulation, fit, and
//! the new-observation draw, so each gate reproduces bit-for-bit (the harness is
//! a replicate-null consumer and inherits the repo determinism requirement).

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};
use ndarray::Array1;

const N_TRAIN: usize = 240;
const N_REPLICATIONS: usize = 120;
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];

/// A low-frequency smooth linear-predictor truth η(x) drawn from the prior. The
/// same shape family the mean-band gates use: comfortably inside the span of a
/// penalized 1-D smooth so smoother bias stays small and a calibrated interval
/// sits at or above nominal.
struct SmoothEta {
    center: f64,
    amplitude: f64,
    frequency: f64,
    phase: f64,
}

impl SmoothEta {
    fn draw(center_lo: f64, center_span: f64, amp: f64, rng: &mut CalibrationRng) -> Self {
        Self {
            center: center_lo + center_span * rng.uniform_open01(),
            amplitude: amp * (0.5 + 0.5 * rng.uniform_open01()),
            frequency: 0.7 + 0.7 * rng.uniform_open01(),
            phase: rng.uniform_open01(),
        }
    }

    fn eta(&self, x: f64) -> f64 {
        self.center
            + self.amplitude * (std::f64::consts::TAU * (self.frequency * x + self.phase)).sin()
    }
}

fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// --- Response-family samplers built on the harness's uniform/normal stream ---

/// Exact Knuth Poisson draw (small λ used here).
fn poisson_draw(lambda: f64, rng: &mut CalibrationRng) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
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

/// Marsaglia–Tsang Gamma(shape, scale) draw. Handles shape < 1 by the standard
/// boosting identity `G(a) = G(a+1)·U^{1/a}`.
fn gamma_draw(shape: f64, scale: f64, rng: &mut CalibrationRng) -> f64 {
    if shape < 1.0 {
        let g = gamma_draw(shape + 1.0, scale, rng);
        return g * rng.uniform_open01().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = rng.standard_normal();
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.uniform_open01();
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v * scale;
        }
    }
}

/// Beta(a, b) via the two-gamma ratio.
fn beta_draw(a: f64, b: f64, rng: &mut CalibrationRng) -> f64 {
    let x = gamma_draw(a, 1.0, rng);
    let y = gamma_draw(b, 1.0, rng);
    if x + y <= 0.0 { 0.5 } else { x / (x + y) }
}

/// Negative-Binomial with mean μ and dispersion θ via the Gamma–Poisson mixture.
fn negbin_draw(mu: f64, theta: f64, rng: &mut CalibrationRng) -> f64 {
    // λ ~ Gamma(shape = θ, scale = μ/θ) ⇒ E[λ] = μ, then Y ~ Poisson(λ) gives
    // Var(Y) = μ + μ²/θ, matching the family's conditional variance.
    let lambda = gamma_draw(theta, mu / theta, rng);
    poisson_draw(lambda, rng)
}

/// Compound Poisson–Gamma (Tweedie, 1 < p < 2) with mean μ and dispersion φ.
fn tweedie_draw(mu: f64, phi: f64, p: f64, rng: &mut CalibrationRng) -> f64 {
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let alpha = (2.0 - p) / (p - 1.0);
    let gamma_scale = phi * (p - 1.0) * mu.powf(p - 1.0);
    let n = poisson_draw(lambda, rng) as usize;
    (0..n).map(|_| gamma_draw(alpha, gamma_scale, rng)).sum()
}

/// A response family under a predictive-interval coverage audit.
struct FamilyCase {
    /// Family string passed to `FitConfig`.
    family: &'static str,
    /// Fixed per-family seed threading truth, simulation, fit, and new-obs draw.
    seed: u64,
    /// η-prior parameters: (center_lo, center_span, amplitude).
    eta_prior: (f64, f64, f64),
    /// Whether the mean is `exp(η)` (log link) or `σ(η)` (logit link).
    log_link: bool,
    /// Draw one response from the family at mean μ.
    sample: Box<dyn Fn(f64, &mut CalibrationRng) -> f64>,
}

impl FamilyCase {
    fn mean(&self, eta: f64) -> f64 {
        if self.log_link { eta.exp() } else { sigmoid(eta) }
    }
}

fn simulate_dataset(x: &[f64], truth: &SmoothEta, case: &FamilyCase, rng: &mut CalibrationRng) -> EncodedDataset {
    let rows: Vec<StringRecord> = x
        .iter()
        .map(|&xi| {
            let mu = case.mean(truth.eta(xi));
            let y = (case.sample)(mu, rng);
            StringRecord::from(vec![xi.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode family replication dataset")
}

/// The predictive (observation) interval at every training row and level.
fn predictive_interval(fit: &FitResult, family_name: &str, level: f64) -> (Array1<f64>, Array1<f64>) {
    let FitResult::Standard(standard) = fit else {
        panic!("{family_name} `y ~ s(x)` must fit through the dense standard path");
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
            .unwrap_or_else(|e| panic!("{family_name} predictive-interval prediction failed: {e:?}"));
    let lower = result.observation_lower.unwrap_or_else(|| {
        panic!(
            "{family_name} is registered as emitting a predictive interval but \
             `observation_lower` was None — the fit did not record the family's \
             dispersion/theta, so the registered predictive surface is unaudited"
        )
    });
    let upper = result
        .observation_upper
        .expect("observation_upper present whenever observation_lower is");
    (lower, upper)
}

/// The shared per-family coverage sweep: fit, draw a new observation, audit.
fn run_family_predictive_gate(case: &FamilyCase) {
    let x = training_grid(N_TRAIN);
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;
    let span = interior_hi - interior_lo;

    let mut rng = CalibrationRng::new(case.seed);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for _ in 0..N_REPLICATIONS {
        let truth = SmoothEta::draw(case.eta_prior.0, case.eta_prior.1, case.eta_prior.2, &mut rng);
        let data = simulate_dataset(&x, &truth, case, &mut rng);
        let config = FitConfig {
            family: Some(case.family.to_string()),
            ..FitConfig::default()
        };
        let fit = fit_from_formula("y ~ s(x)", &data, &config)
            .unwrap_or_else(|e| panic!("{} smooth fit failed: {e:?}", case.family));

        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        let mu_true = case.mean(truth.eta(x[j]));
        // A genuinely NEW observation from the family at the evaluation point,
        // independent of the training data — what the predictive interval must
        // cover.
        let y_new = (case.sample)(mu_true, &mut rng);

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let (lower, upper) = predictive_interval(&fit, case.family, level);
            assert!(
                lower[j].is_finite() && upper[j].is_finite() && upper[j] >= lower[j],
                "{} degenerate predictive interval at eval point {j} (level {level}): [{}, {}]",
                case.family,
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
        "{}: every predictive interval had zero width — not a real interval",
        case.family
    );

    let mut failures = Vec::new();
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], N_REPLICATIONS, level);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "level {level}: empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], \
                 nominal ABOVE the CI by {:.4} — anti-conservative predictive interval \
                 (the #1875/#1878 recycled/mis-scaled or #817/#1193/#1194 dropped-skew signature)",
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
        "{} predictive interval under-covers a new observation:\n{}",
        case.family,
        failures.join("\n")
    );
}

#[test]
fn poisson_predictive_interval_covers_new_observation_at_nominal() {
    // exp(center) ≈ 3 .. 8 keeps counts small enough that the zero atom and
    // discreteness matter — exactly where a symmetric band mis-covers (#817).
    run_family_predictive_gate(&FamilyCase {
        family: "poisson",
        seed: 0x1891_A0_15_50_FF,
        eta_prior: (1.1, 1.0, 0.9),
        log_link: true,
        sample: Box::new(|mu, rng| poisson_draw(mu, rng)),
    });
}

#[test]
fn negative_binomial_predictive_interval_covers_new_observation_at_nominal() {
    // True dispersion θ = 6.0: overdispersed relative to Poisson, right-skewed
    // counts where the symmetric band under-covers the upper tail (#1193).
    let theta = 6.0;
    run_family_predictive_gate(&FamilyCase {
        family: "negative-binomial",
        seed: 0x1891_DB_D1_59_A0,
        eta_prior: (1.2, 0.9, 0.8),
        log_link: true,
        sample: Box::new(move |mu, rng| negbin_draw(mu, theta, rng)),
    });
}

#[test]
fn gamma_predictive_interval_covers_new_observation_at_nominal() {
    // True dispersion φ = 0.25 (shape = 1/φ = 4): strong right skew where the
    // symmetric μ ± z·σ band mis-covers each tail (#817).
    let phi = 0.25;
    let shape = 1.0 / phi;
    run_family_predictive_gate(&FamilyCase {
        family: "gamma",
        seed: 0x1891_6A_33A_C0,
        eta_prior: (0.6, 1.0, 0.7),
        log_link: true,
        sample: Box::new(move |mu, rng| gamma_draw(shape, mu / shape, rng)),
    });
}

#[test]
fn beta_predictive_interval_covers_new_observation_at_nominal() {
    // True precision φ = 12 (a = μφ, b = (1−μ)φ): continuous on (0,1), skewed
    // toward whichever edge the mean is near, so a symmetric band mis-covers
    // both tails (#1194).
    let phi = 12.0;
    run_family_predictive_gate(&FamilyCase {
        family: "beta",
        seed: 0x1891_BE_7A_C0DE,
        eta_prior: (-0.5, 1.0, 0.9),
        log_link: false,
        sample: Box::new(move |mu, rng| {
            let m = mu.clamp(1e-4, 1.0 - 1e-4);
            beta_draw(m * phi, (1.0 - m) * phi, rng)
        }),
    });
}

#[test]
fn tweedie_predictive_interval_covers_new_observation_at_nominal() {
    // True power p = 1.5 (the bare-Tweedie default), dispersion φ = 0.6:
    // compound Poisson–Gamma with a point mass at zero plus a right-skewed
    // positive part — the #817 skew defect's compound-distribution instance.
    let phi = 0.6;
    let power = 1.5;
    run_family_predictive_gate(&FamilyCase {
        family: "tweedie",
        seed: 0x1891_7E_ED1E_00,
        eta_prior: (0.7, 0.9, 0.7),
        log_link: true,
        sample: Box::new(move |mu, rng| tweedie_draw(mu, phi, power, rng)),
    });
}

#[test]
fn binomial_predictive_interval_covers_new_observation_at_nominal() {
    // Bernoulli response (single-trial binomial): the predictive interval must
    // cover a fresh 0/1 draw. Degenerate-discrete, but the coverage sweep is
    // still meaningful — an interval that excludes the realized class under-
    // covers, and the Wilson verdict tolerates the necessary discreteness slack.
    run_family_predictive_gate(&FamilyCase {
        family: "binomial",
        seed: 0x1891_B1_00_C0DE,
        eta_prior: (-0.6, 1.2, 0.9),
        log_link: false,
        sample: Box::new(|mu, rng| if rng.uniform_open01() < mu { 1.0 } else { 0.0 }),
    });
}
