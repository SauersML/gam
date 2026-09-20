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
    InferenceCovarianceMode, MultinomialFitRequest, fit_penalized_multinomial_formula,
    predict_multinomial_formula_with_intervals, predict_multinomial_formula_with_intervals_in_mode,
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

        let config = FitConfig::default();
        let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
            init_lambda: 1.0,
            max_iter: 60,
            tol: 1e-8,
            ..MultinomialFitRequest::new(&data, "y ~ s(x, bs='tps', k=8)", &config)
        })
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

/// gam#2612: the two covariance definitions are DIFFERENT bands, and the
/// conditional one is the narrower.
///
/// The sweep above audits the band this driver publishes by default. It cannot
/// see WHY that band covers: an interval that is calibrated because it carries
/// the smoothing-parameter uncertainty and one that is calibrated by accident
/// look identical to a hit count. This checks the mechanism directly, on one
/// fit, with the level held fixed: the corrected band strictly contains the
/// conditional band wherever the correction reaches, and the two report
/// different `covariance_source`s.
///
/// It is the registered audit for `multinomial_mean_prediction_interval_conditional`
/// — the conditional mode is a legitimate answer to a narrower question, and a
/// fit that cannot produce the correction still publishes it, so it needs a gate
/// of its own rather than being reachable only as a fallback.
#[test]
fn multinomial_conditional_band_is_narrower_than_the_corrected_band() {
    let x = training_grid(N_TRAIN);
    let mut rng = CalibrationRng::new(SEED ^ 0x2612);
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
    let data = encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode multinomial dataset");

    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 60,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&data, "y ~ s(x, bs='tps', k=8)", &config)
    })
    .unwrap_or_else(|e| panic!("multinomial smooth fit failed: {e:?}"));

    // A fit with no retained correction cannot answer this question, and saying
    // so is better than a vacuous pass — the whole point of the gate is that the
    // correction is present and reaches the band.
    assert!(
        model.smoothing_correction_flat.is_some(),
        "this fixture must retain a ρ-uncertainty correction for the two modes to \
         differ; without one the multinomial is back to conditional-only bands, \
         which is the #2612 defect"
    );

    let grid: Vec<StringRecord> = (0..21)
        .map(|i| StringRecord::from(vec![(i as f64 / 20.0).to_string()]))
        .collect();
    let newdata = encode_recordswith_inferred_schema(vec!["x".to_string()], grid)
        .expect("encode multinomial grid");

    let level = 0.95;
    let conditional = predict_multinomial_formula_with_intervals_in_mode(
        &model,
        &newdata,
        level,
        InferenceCovarianceMode::Conditional,
    )
    .expect("conditional band");
    let corrected = predict_multinomial_formula_with_intervals_in_mode(
        &model,
        &newdata,
        level,
        InferenceCovarianceMode::SmoothingCorrected,
    )
    .expect("corrected band");
    let published = predict_multinomial_formula_with_intervals(&model, &newdata, level)
        .expect("published band");

    assert_eq!(
        conditional.covariance_source,
        InferenceCovarianceMode::Conditional,
        "the conditional band must report its own definition"
    );
    assert_eq!(
        corrected.covariance_source,
        InferenceCovarianceMode::SmoothingCorrected,
        "the corrected band must report its own definition"
    );
    assert_eq!(
        published.covariance_source,
        InferenceCovarianceMode::SmoothingCorrected,
        "a fit that retains a correction must PUBLISH the corrected band by \
         default, as every other family in the library does"
    );

    let mut strictly_wider = 0usize;
    for ((row, class), &narrow) in conditional.standard_error.indexed_iter() {
        let wide = corrected.standard_error[[row, class]];
        assert!(
            wide >= narrow - 1e-15,
            "corrected spread {wide} is narrower than conditional {narrow} at \
             ({row}, {class}): a variance component cannot subtract"
        );
        assert!(
            corrected.mean_lower[[row, class]] <= conditional.mean_lower[[row, class]] + 1e-12
                && corrected.mean_upper[[row, class]]
                    >= conditional.mean_upper[[row, class]] - 1e-12,
            "the corrected band must CONTAIN the conditional band at ({row}, {class}): \
             [{}, {}] does not contain [{}, {}]",
            corrected.mean_lower[[row, class]],
            corrected.mean_upper[[row, class]],
            conditional.mean_lower[[row, class]],
            conditional.mean_upper[[row, class]],
        );
        if wide > narrow {
            strictly_wider += 1;
        }
    }
    assert!(
        strictly_wider * 2 > conditional.standard_error.len(),
        "the correction reached only {strictly_wider} of {} published spreads; a \
         correction that is stored but never contracted through the response \
         Jacobian leaves the band conditional in everything but its label",
        conditional.standard_error.len(),
    );

    // Both bands live strictly inside the open simplex face: the log-odds
    // transform is a bijection onto (0, 1), so no endpoint is ever produced by a
    // clamp — which is what silently deleted posterior mass from the old
    // probability-scale Wald band.
    for band in [&conditional, &corrected] {
        for ((row, class), &lower) in band.mean_lower.indexed_iter() {
            let upper = band.mean_upper[[row, class]];
            assert!(
                lower > 0.0 && upper < 1.0 && lower <= upper,
                "band endpoint reached the boundary at ({row}, {class}): [{lower}, {upper}]"
            );
        }
    }
}
