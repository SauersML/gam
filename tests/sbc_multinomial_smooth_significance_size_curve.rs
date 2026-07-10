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
//! `α ∈ {0.01, 0.05, 0.10}` the empirical rejection rate must not exceed `α`
//! beyond Monte-Carlo error, audited as coverage of the non-rejection event at
//! nominal `1 − α` (the shared `TEST_SIZE_ALPHAS` convention): an oversized test
//! under-covers non-rejection and gates; an undersized (conservative) test
//! over-covers and only reports.

use csv::StringRecord;
use gam::families::multinomial::fit_penalized_multinomial_formula;
use gam::{FitConfig, encode_recordswith_inferred_schema};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};

const N_TRAIN: usize = 150;
const N_REPLICATIONS: usize = 200;
const ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];
const SEED: u64 = 0x1891_5_M17_5_1_9E;

const CLASS_LO: &str = "lo";
const CLASS_HI: &str = "hi";

fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

#[test]
fn multinomial_smooth_significance_pvalue_is_not_oversized_under_the_null() {
    let x = training_grid(N_TRAIN);
    let mut rng = CalibrationRng::new(SEED);
    // Per-alpha count of NON-rejections (p_value > alpha) — the covered event
    // the shared Wilson verdict audits at nominal `1 - alpha`.
    let mut non_rejections = [0usize; ALPHAS.len()];
    let mut replications_used = 0usize;

    for rep in 0..N_REPLICATIONS {
        // TRUE NULL: y is Categorical(0.5, 0.5), independent of x.
        let rows: Vec<StringRecord> = x
            .iter()
            .map(|&xi| {
                let label = if rng.uniform_open01() < 0.5 { CLASS_HI } else { CLASS_LO };
                StringRecord::from(vec![xi.to_string(), label.to_string()])
            })
            .collect();
        let headers = vec!["x".to_string(), "y".to_string()];
        let data =
            encode_recordswith_inferred_schema(headers, rows).expect("encode null multinomial dataset");

        let model = fit_penalized_multinomial_formula(
            &data,
            "y ~ s(x, bs='tp', k=8)",
            &FitConfig::default(),
            1.0,
            60,
            1e-8,
        )
        .unwrap_or_else(|e| panic!("multinomial null smooth fit failed (rep {rep}): {e:?}"));

        let significance = model.smooth_significance();
        let Some(row) = significance.first() else {
            // A degenerate replication (e.g. the smooth term collapsed to the
            // nullspace) declines to report a p-value rather than fabricate
            // one; skip it rather than treat "no row" as either a hit or miss.
            continue;
        };
        assert!(
            row.p_value.is_finite() && (0.0..=1.0).contains(&row.p_value),
            "rep {rep}: multinomial smooth-significance p-value out of range: {}",
            row.p_value
        );
        replications_used += 1;

        for (alpha_idx, &alpha) in ALPHAS.iter().enumerate() {
            if row.p_value > alpha {
                non_rejections[alpha_idx] += 1;
            }
        }
    }

    assert!(
        replications_used >= N_REPLICATIONS / 2,
        "too many degenerate replications ({replications_used}/{N_REPLICATIONS} usable) — \
         the null-FPR audit needs a real sample to resolve"
    );

    let mut failures = Vec::new();
    for (alpha_idx, &alpha) in ALPHAS.iter().enumerate() {
        let nominal_non_reject = 1.0 - alpha;
        let verdict = audit_coverage(non_rejections[alpha_idx], replications_used, nominal_non_reject);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "alpha={alpha}: empirical non-reject rate={:.4} (non-rejections {}/{}), \
                 Wilson CI=[{:.4},{:.4}], nominal {nominal_non_reject} ABOVE the CI by {:.4} — \
                 the multinomial smooth-significance test rejects the true null too often \
                 (anti-conservative, the #1872/#1873 genus)",
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
        "multinomial smooth-significance p-value is oversized under the null:\n{}",
        failures.join("\n")
    );
}
