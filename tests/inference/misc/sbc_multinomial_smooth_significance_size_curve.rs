//! Standing calibration gate for the multinomial smooth-term significance
//! table (`MultinomialSavedModel::smooth_significance`, issues #1891, #3569).
//!
//! Each row is the variance-component score test of one smooth term evaluated
//! at the softmax working score of the fitted multinomial model: one row per
//! active class (the term's reference-coded contrast `log(p_a / p_ref)`) and,
//! for `K ≥ 3`, one joint row testing the term in every class at once under the
//! reference-symmetric structural metric `(I − 11ᵀ/K) ⊗ S`.
//!
//! The p-value is a calibrated surface, so the gate is two-tailed: under a true
//! null the rejection count at `α ∈ {0.10, 0.05, 0.01}` and the count of
//! `p > 1 − α` must both sit within `α ± 3·MCSE`, and the p-values must pass a
//! two-sided Kolmogorov–Smirnov test against U(0, 1). A conservative test fails
//! exactly like an anti-conservative one.
//!
//! Scenarios:
//! * binary null (`K = 2`): the single class row;
//! * three-class global null: both class rows and the joint row;
//! * three-class, `x` moves only class `b` (reference `c`): `log(p_a / p_c)`
//!   does not depend on `x`, so the class-`a` row is a true null even though the
//!   shared class-centred penalty couples it to the moving class. The class-`b`
//!   and joint rows are the power controls;
//! * the joint row is a property of the fitted distribution, so relabelling the
//!   classes (which changes the reference class) must leave it unchanged.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::families::multinomial::{
    MultinomialFitRequest, MultinomialSavedModel, MultinomialSmoothContrast,
    fit_penalized_multinomial_formula,
};
use gam::terms::inference::smooth_test::SmoothTestResult;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use gam_test_support::calibration::CalibrationRng;
use rayon::prelude::*;

const N_OBS: usize = 300;
const N_REPLICATIONS: u64 = 400;
const ALPHAS: [f64; 3] = [0.10, 0.05, 0.01];
/// Band half-width in Monte-Carlo standard errors. Twelve tail counts are
/// gated per scenario; at 3·MCSE the family-wise false-alarm rate of a
/// calibrated test stays below 12 · 0.0027 ≈ 3%.
const MCSE_BAND: f64 = 3.0;
const KS_LEVEL: f64 = 0.01;
const SEED: u64 = 0x3569_0000;
const FORMULA: &str = "y ~ s(x, bs='tps', k=8)";
/// The significance table labels a smooth by its formula token, options included.
const TERM: &str = "s(x, bs='tps', k=8)";
/// Amplitude of the class-`b` log-odds curve in the signal scenario.
const SIGNAL_AMPLITUDE: f64 = 1.5;

/// Draw one dataset: `x ~ U(0, 1)`, `y ~ Categorical(softmax(η(x)))` over the
/// labels in `labels`, where `eta(x)` returns one linear predictor per label.
fn dataset(seed: u64, labels: &[&str], eta: impl Fn(f64) -> Vec<f64>) -> EncodedDataset {
    let mut rng = CalibrationRng::new(seed);
    let rows: Vec<StringRecord> = (0..N_OBS)
        .map(|_| {
            let x = rng.uniform_open01();
            let linear = eta(x);
            let top = linear.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = linear.iter().map(|e| (e - top).exp()).collect();
            let total: f64 = weights.iter().sum();
            let u = rng.uniform_open01() * total;
            let mut acc = 0.0;
            let mut chosen = labels.len() - 1;
            for (index, weight) in weights.iter().enumerate() {
                acc += weight;
                if u < acc {
                    chosen = index;
                    break;
                }
            }
            StringRecord::from(vec![x.to_string(), labels[chosen].to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode multinomial calibration dataset")
}

fn fit(data: &EncodedDataset) -> MultinomialSavedModel {
    let config = FitConfig::default();
    fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 60,
        tol: 1e-8,
        ..MultinomialFitRequest::new(data, FORMULA, &config)
    })
    .unwrap_or_else(|error| panic!("multinomial calibration fit failed: {error:?}"))
}

/// The significance row for `contrast`; a refused row under these
/// well-identified designs is itself a defect, so it panics with the reason.
fn row(model: &MultinomialSavedModel, contrast: &MultinomialSmoothContrast) -> SmoothTestResult {
    let table = model
        .smooth_significance()
        .unwrap_or_else(|error| panic!("smooth_significance failed: {error:?}"));
    let available: Vec<String> = table
        .iter()
        .map(|row| format!("{:?}/{}", row.contrast, row.term_label))
        .collect();
    let found = table
        .into_iter()
        .find(|row| &row.contrast == contrast && row.term_label == TERM)
        .unwrap_or_else(|| panic!("no {contrast:?} row for {TERM}; the table has {available:?}"));
    found.test.unwrap_or_else(|reason| {
        panic!(
            "{contrast:?} row for {TERM} was refused ({})",
            reason.label()
        )
    })
}

fn class(label: &str) -> MultinomialSmoothContrast {
    MultinomialSmoothContrast::Class(label.to_string())
}

/// Two-sided one-sample Kolmogorov–Smirnov test of `values` against U(0, 1):
/// returns `(D, asymptotic p-value)`.
fn ks_uniform(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len() as f64;
    let distance = sorted
        .iter()
        .enumerate()
        .map(|(i, &p)| ((i as f64 + 1.0) / m - p).max(p - i as f64 / m))
        .fold(0.0_f64, f64::max);
    let lambda = m.sqrt() * distance;
    let mut sum = 0.0;
    let mut k = 1.0_f64;
    loop {
        let term = (-2.0 * k * k * lambda * lambda).exp();
        let signed = if (k as u64) % 2 == 1 { term } else { -term };
        if sum + signed == sum {
            break;
        }
        sum += signed;
        k += 1.0;
    }
    (distance, (2.0 * sum).clamp(0.0, 1.0))
}

/// Two-tailed calibration verdict for one row's null p-values; returns the
/// list of violations (empty when calibrated).
fn calibration_violations(name: &str, p_values: &[f64]) -> Vec<String> {
    let m = p_values.len() as f64;
    let mut violations = Vec::new();
    let mut report = Vec::new();
    for &alpha in &ALPHAS {
        let half_width = MCSE_BAND * (alpha * (1.0 - alpha) / m).sqrt();
        let lower = p_values.iter().filter(|&&p| p <= alpha).count() as f64 / m;
        let upper = p_values.iter().filter(|&&p| p > 1.0 - alpha).count() as f64 / m;
        report.push(format!(
            "α={alpha}: P(p≤α)={lower:.4}, P(p>1−α)={upper:.4} (α ± {half_width:.4})"
        ));
        for (tail, rate) in [("p ≤ α", lower), ("p > 1 − α", upper)] {
            if (rate - alpha).abs() > half_width {
                violations.push(format!(
                    "{name}: α={alpha}: empirical P({tail}) = {rate:.4} outside \
                     α ± {MCSE_BAND}·MCSE = [{:.4}, {:.4}]",
                    alpha - half_width,
                    alpha + half_width
                ));
            }
        }
    }
    let (ks_distance, ks_p_value) = ks_uniform(p_values);
    report.push(format!("KS D {ks_distance:.4}, p {ks_p_value:.4}"));
    if ks_p_value < KS_LEVEL {
        violations.push(format!(
            "{name}: null p-values are not U(0, 1): two-sided KS D = {ks_distance:.4}, \
             p = {ks_p_value:.4} < {KS_LEVEL}"
        ));
    }
    eprintln!("#3569 {name}: {}", report.join("; "));
    violations
}

fn assert_calibrated(rows: &[(&str, Vec<f64>)]) {
    let violations: Vec<String> = rows
        .iter()
        .flat_map(|(name, p_values)| calibration_violations(name, p_values))
        .collect();
    assert!(
        violations.is_empty(),
        "multinomial smooth score test is miscalibrated under a true null:\n{}",
        violations.join("\n")
    );
}

#[test]
fn multinomial_binary_smooth_score_row_is_two_tailed_calibrated_under_the_null() {
    init_parallelism();
    let p_values: Vec<f64> = (0..N_REPLICATIONS)
        .into_par_iter()
        .map(|rep| {
            let data = dataset(SEED ^ rep, &["hi", "lo"], |_| vec![0.0, 0.0]);
            let model = fit(&data);
            assert_eq!(model.class_levels, ["hi", "lo"]);
            row(&model, &class("hi")).p_value
        })
        .collect();
    assert_calibrated(&[("K=2 class hi", p_values)]);
}

#[test]
fn multinomial_three_class_smooth_score_rows_are_calibrated_under_the_global_null() {
    init_parallelism();
    let rows: Vec<[f64; 3]> = (0..N_REPLICATIONS)
        .into_par_iter()
        .map(|rep| {
            let data = dataset(SEED ^ (1 << 32) ^ rep, &["a", "b", "c"], |_| {
                vec![0.0, 0.0, 0.0]
            });
            let model = fit(&data);
            assert_eq!(model.class_levels, ["a", "b", "c"]);
            [
                row(&model, &class("a")).p_value,
                row(&model, &class("b")).p_value,
                row(&model, &MultinomialSmoothContrast::Joint).p_value,
            ]
        })
        .collect();
    let column = |i: usize| rows.iter().map(|r| r[i]).collect::<Vec<_>>();
    assert_calibrated(&[
        ("K=3 null class a", column(0)),
        ("K=3 null class b", column(1)),
        ("K=3 null joint", column(2)),
    ]);
}

fn class_b_curve(x: f64) -> f64 {
    SIGNAL_AMPLITUDE * (2.0 * std::f64::consts::PI * x).sin()
}

/// `x` moves only class `b`; `c` is the reference, so `log(p_a / p_c) ≡ 0`
/// and the class-`a` row is a true null. The class-`b` and joint rows must
/// reject at `α = 0.05` in the large majority of replications.
#[test]
fn multinomial_class_row_is_calibrated_when_only_another_class_moves() {
    init_parallelism();
    let rows: Vec<[f64; 3]> = (0..N_REPLICATIONS)
        .into_par_iter()
        .map(|rep| {
            let data = dataset(SEED ^ (2 << 32) ^ rep, &["a", "b", "c"], |x| {
                vec![0.0, class_b_curve(x), 0.0]
            });
            let model = fit(&data);
            assert_eq!(model.class_levels, ["a", "b", "c"]);
            [
                row(&model, &class("a")).p_value,
                row(&model, &class("b")).p_value,
                row(&model, &MultinomialSmoothContrast::Joint).p_value,
            ]
        })
        .collect();
    let column = |i: usize| rows.iter().map(|r| r[i]).collect::<Vec<_>>();
    let power =
        |values: &[f64]| values.iter().filter(|&&p| p <= 0.05).count() as f64 / values.len() as f64;
    let power_b = power(&column(1));
    let power_joint = power(&column(2));
    eprintln!("#3569 power at α=0.05: class b {power_b:.4}, joint {power_joint:.4}");
    // The null band at α = 0.05 reaches 0.05 + 3·MCSE ≈ 0.083 at 400
    // replications; a working test at this amplitude must clear it by far.
    // Half the replications is the power floor.
    assert!(
        power_b >= 0.5 && power_joint >= 0.5,
        "the multinomial smooth score test has no power against a real class-b curve: \
         class b {power_b:.4}, joint {power_joint:.4}"
    );
    assert_calibrated(&[("K=3 class a with class b moving", column(0))]);
}

/// Relabelling the classes cyclically (`a → b → c → a`) makes the original
/// class `b` the reference instead of `c`. The joint row tests the term in
/// every class under the reference-symmetric metric, so it is a property of
/// the fitted distribution: its statistic and p-value must not move beyond the
/// cross-labelling drift of the fit itself. The fitted class probabilities
/// are held to 1e-3 across labellings (`multinomial_fit_invariant_to_reference_class_1587`);
/// the joint statistic is a smooth function of the same fit, so the gate holds
/// it to the same relative tolerance.
#[test]
fn multinomial_joint_row_is_invariant_to_the_reference_class() {
    init_parallelism();
    const RELATIVE_TOLERANCE: f64 = 1e-3;
    let relabelled = ["b", "c", "a"];
    let failures: Vec<String> = (0..8u64)
        .into_par_iter()
        .filter_map(|rep| {
            let eta = |x: f64| vec![0.0, class_b_curve(x) / 4.0, 0.0];
            let original = row(
                &fit(&dataset(SEED ^ (3 << 32) ^ rep, &["a", "b", "c"], eta)),
                &MultinomialSmoothContrast::Joint,
            );
            let permuted = row(
                &fit(&dataset(SEED ^ (3 << 32) ^ rep, &relabelled, eta)),
                &MultinomialSmoothContrast::Joint,
            );
            let scale = original.statistic.abs().max(1.0);
            let drift = (original.statistic - permuted.statistic).abs() / scale;
            eprintln!(
                "#3569 rep {rep}: joint statistic {:.6} vs {:.6} (relative drift {drift:.3e}), \
                 p {:.6} vs {:.6}",
                original.statistic, permuted.statistic, original.p_value, permuted.p_value
            );
            (drift > RELATIVE_TOLERANCE
                || (original.ref_df - permuted.ref_df).abs()
                    > RELATIVE_TOLERANCE * original.ref_df.max(1.0))
            .then(|| {
                format!(
                    "rep {rep}: joint row depends on the reference class: \
                         statistic {} vs {}, ref.df {} vs {}",
                    original.statistic, permuted.statistic, original.ref_df, permuted.ref_df
                )
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
