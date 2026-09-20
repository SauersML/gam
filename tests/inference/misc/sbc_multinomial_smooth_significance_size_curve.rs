//! Standing type-I size gate (issue #1891): the multinomial per-class smooth
//! significance p-value (`MultinomialSavedModel::smooth_significance` /
//! `MultinomialSmoothSignificance::p_value`).
//!
//! A completeness sweep of the library's public payload structs (the #1891
//! follow-up) found `MultinomialSmoothSignificance` unregistered and ungated.
//! It is a second, independent instance of the Wood rank-truncated Wald test
//! the registry's `wood_smooth_test_pvalue` target already gates — but through
//! entirely separate plumbing: `gam-models::multinomial`'s own block-ordered
//! coefficient assembly, per-class EDF extraction, and covariance-block slicing
//! feed the SAME shared `gam_terms::inference::smooth_test::wood_smooth_test`
//! primitive the single-response Wald test uses. A wrong block index, wrong
//! EDF, or wrong covariance slice here is invisible to every other #1891 gate.
//!
//! Audit: type-I size under a TRUE NULL (the smooth term has NO effect on the
//! class log-odds — `y` is generated independently of `x`). At
//! `α ∈ {0.01, 0.05, 0.10}` the empirical rejection rate must match `α` to
//! within Monte-Carlo error, audited as coverage of the non-rejection event at
//! nominal `1 − α`. Both tails gate (#3534): an oversized test under-covers
//! non-rejection, an undersized (conservative) test over-covers it, and each
//! is a miscalibrated p-value. A replication whose smooth block has no
//! estimable covariance direction reports no row (`wood_smooth_test` declines:
//! the term carries no testable signal); the test did not reject, so it is
//! counted as a non-rejection at every `α` and every replication enters the
//! denominator — dropping it would condition the size on the fit's outcome.
//!
//! Resolution: a never-rejecting test is detectable at nominal `1 − α` only
//! once `R > z²(1 − α)/α` (the smallest `R` whose all-hits Wilson lower bound
//! clears nominal), i.e. `R ≥ 127` at `α = 0.05` and `R ≥ 60` at `α = 0.10`;
//! `N_REPLICATIONS = 200` resolves both. At `α = 0.01` the bound is 657, so
//! that level gates only the oversized tail at this replication count.

use csv::StringRecord;
use gam::families::multinomial::{MultinomialFitRequest, fit_penalized_multinomial_formula};
use gam::{FitConfig, encode_recordswith_inferred_schema};
use gam_test_support::calibration::{CalibrationRng, audit_coverage};

const N_TRAIN: usize = 150;
const N_REPLICATIONS: usize = 200;
const ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];
const SEED: u64 = 0x1891_5A17_51A9_E;

const CLASS_LO: &str = "lo";
const CLASS_HI: &str = "hi";

fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

#[test]
fn multinomial_smooth_significance_pvalue_is_calibrated_under_the_null() {
    let x = training_grid(N_TRAIN);
    let mut rng = CalibrationRng::new(SEED);
    // Per-alpha count of NON-rejections (p_value > alpha) — the covered event
    // the shared Wilson verdict audits at nominal `1 - alpha`.
    let mut non_rejections = [0usize; ALPHAS.len()];

    for rep in 0..N_REPLICATIONS {
        // TRUE NULL: y is Categorical(0.5, 0.5), independent of x.
        let rows: Vec<StringRecord> = x
            .iter()
            .map(|&xi| {
                let label = if rng.uniform_open01() < 0.5 {
                    CLASS_HI
                } else {
                    CLASS_LO
                };
                StringRecord::from(vec![xi.to_string(), label.to_string()])
            })
            .collect();
        let headers = vec!["x".to_string(), "y".to_string()];
        let data = encode_recordswith_inferred_schema(headers, rows)
            .expect("encode null multinomial dataset");

        let config = FitConfig::default();
        let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
            init_lambda: 1.0,
            max_iter: 60,
            tol: 1e-8,
            ..MultinomialFitRequest::new(&data, "y ~ s(x, bs='tps', k=8)", &config)
        })
        .unwrap_or_else(|e| panic!("multinomial null smooth fit failed (rep {rep}): {e:?}"));

        let significance = model.smooth_significance();
        let Some(row) = significance.first() else {
            // No estimable direction in the smooth block: the test declines,
            // which is a non-rejection at every level.
            for count in &mut non_rejections {
                *count += 1;
            }
            continue;
        };
        assert!(
            row.p_value.is_finite() && (0.0..=1.0).contains(&row.p_value),
            "rep {rep}: multinomial smooth-significance p-value out of range: {}",
            row.p_value
        );

        for (alpha_idx, &alpha) in ALPHAS.iter().enumerate() {
            if row.p_value > alpha {
                non_rejections[alpha_idx] += 1;
            }
        }
    }

    let mut failures = Vec::new();
    for (alpha_idx, &alpha) in ALPHAS.iter().enumerate() {
        let nominal_non_reject = 1.0 - alpha;
        let verdict = audit_coverage(
            non_rejections[alpha_idx],
            N_REPLICATIONS,
            nominal_non_reject,
        );
        if !verdict.passed {
            failures.push(format!(
                "alpha={alpha}: non-rejection {}",
                verdict.describe()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "multinomial smooth-significance p-value is miscalibrated under the null:\n{}",
        failures.join("\n")
    );
}
