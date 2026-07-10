//! Standing coverage gate (issue #1891): the multinomial mean-probability
//! prediction interval (`MultinomialPredictionIntervals` /
//! `predict_multinomial_formula_with_intervals`).
//!
//! This surface is analogous to `mean_credible_band` for the Gaussian/GLM
//! predict path — a Wald-style interval `mean ± z·standard_error` around the
//! integrated posterior-mean class probability — but it is built by a
//! completely separate driver (`gam-models::multinomial`), so a mis-scaled or
//! recycled standard error here is invisible to every other #1891 gate. An
//! audit-completeness sweep of the library's public payload structs found this
//! target unregistered and ungated: `MultinomialPredictionIntervals` is not one
//! of the three payload types the `tests/quality/calibration` completeness
//! lint exhaustively destructures, so nothing forced it onto the registry.
//!
//! Coverage experiment: draw a smooth log-odds truth η(x) from a prior, simulate
//! a two-class categorical response `y ~ Categorical(σ(η(x)))`, fit
//! `y ~ s(x)`, and at one independent interior covariate value check whether the
//! true class-1 probability lies inside the reported `[mean_lower, mean_upper]`
//! band. Audited by the shared Wilson verdict at 80/90/95; only anti-conservative
//! under-coverage gates.

use csv::StringRecord;
use gam::families::multinomial::{
    fit_penalized_multinomial_formula, predict_multinomial_formula_with_intervals,
};
use gam::{FitConfig, encode_recordswith_inferred_schema};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};

const N_TRAIN: usize = 240;
const N_REPLICATIONS: usize = 80;
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];
const SEED: u64 = 0x1891_A17_1_C0DE;

const CLASS_LO: &str = "lo";
const CLASS_HI: &str = "hi";

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// A low-frequency smooth log-odds truth η(x) drawn from the prior — the same
/// shape family the other #1891 mean-band gates use.
struct SmoothEta {
    center: f64,
    amplitude: f64,
    frequency: f64,
    phase: f64,
}

impl SmoothEta {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            center: -0.3 + 0.6 * rng.uniform_open01(),
            amplitude: 0.5 + 0.5 * rng.uniform_open01(),
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

#[test]
fn multinomial_mean_prediction_interval_covers_true_probability_at_nominal() {
    let x = training_grid(N_TRAIN);
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;
    let span = interior_hi - interior_lo;

    let mut rng = CalibrationRng::new(SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for _ in 0..N_REPLICATIONS {
        let truth = SmoothEta::draw(&mut rng);

        let mut rows: Vec<StringRecord> = Vec::with_capacity(N_TRAIN);
        for &xi in &x {
            let p_hi = sigmoid(truth.eta(xi));
            let label = if rng.uniform_open01() < p_hi {
                CLASS_HI
            } else {
                CLASS_LO
            };
            rows.push(StringRecord::from(vec![xi.to_string(), label.to_string()]));
        }
        let headers = vec!["x".to_string(), "y".to_string()];
        let data =
            encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset");

        let model = fit_penalized_multinomial_formula(
            &data,
            "y ~ s(x, bs='tp', k=8)",
            &FitConfig::default(),
            1.0,
            60,
            1e-8,
        )
        .unwrap_or_else(|e| panic!("multinomial smooth fit failed: {e:?}"));
        let hi_col = model
            .class_levels
            .iter()
            .position(|c| c == CLASS_HI)
            .expect("hi class present among fitted class levels");

        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        let x_star = x[j];
        let p_true = sigmoid(truth.eta(x_star));

        let new_headers = vec!["x".to_string()];
        let new_rows = vec![StringRecord::from(vec![x_star.to_string()])];
        let newdata = encode_recordswith_inferred_schema(new_headers, new_rows)
            .expect("encode multinomial newdata");

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let intervals = predict_multinomial_formula_with_intervals(&model, &newdata, level)
                .unwrap_or_else(|e| panic!("multinomial predict-with-intervals failed: {e:?}"));
            let lower = intervals.mean_lower[[0, hi_col]];
            let upper = intervals.mean_upper[[0, hi_col]];
            assert!(
                lower.is_finite() && upper.is_finite() && upper >= lower,
                "degenerate multinomial mean interval at level {level}: [{lower}, {upper}]"
            );
            if upper - lower > 0.0 {
                positive_width_seen = true;
            }
            if lower <= p_true && p_true <= upper {
                hits[level_idx] += 1;
            }
        }
    }

    assert!(
        positive_width_seen,
        "every multinomial mean-probability interval had zero width — not a real interval"
    );

    let mut failures = Vec::new();
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], N_REPLICATIONS, level);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "level {level}: empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], \
                 nominal ABOVE the CI by {:.4} — anti-conservative multinomial mean interval",
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
        "multinomial mean-probability interval under-covers the true class probability:\n{}",
        failures.join("\n")
    );
}
