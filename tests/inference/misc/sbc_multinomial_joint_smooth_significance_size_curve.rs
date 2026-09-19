//! Standing type-I size gate (issue #1891) for the multinomial term tests on a
//! THREE-class response, where the per-class and all-classes tests differ.
//!
//! With `K = 3` a smooth term owns `M = 2` class blocks. The per-class test
//! (`MultinomialSavedModel::smooth_significance`) asks whether the term moves
//! one class's log-odds against the reference class; the all-classes test
//! (`MultinomialSavedModel::joint_smooth_significance`) asks whether the term
//! moves ANY class probability, a single Wald test on the joint block whose
//! covariance carries the cross-class correlation the per-class slices drop.
//!
//! Audit: type-I size under a TRUE NULL. `x` has no effect on any class while
//! a second covariate `z` drives the class probabilities, so the null term is
//! tested inside a genuinely multi-predictor fit. At `α ∈ {0.01, 0.05, 0.10}`
//! the rejection rate of the joint test and of each per-class test must not
//! exceed `α` beyond Monte-Carlo error, audited as coverage of the
//! non-rejection event at nominal `1 − α`: an oversized test under-covers
//! non-rejection and gates; an undersized one over-covers and only reports.

use csv::StringRecord;
use gam::families::multinomial::{MultinomialFitRequest, fit_penalized_multinomial_formula};
use gam::{FitConfig, encode_recordswith_inferred_schema};
use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};

const N_TRAIN: usize = 200;
const N_REPLICATIONS: usize = 200;
const ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];
const SEED: u64 = 0x1891_3C1A_55E5;
const CLASSES: [&str; 3] = ["a", "b", "c"];
/// The null term's label begins with this (the label may echo basis options).
const NULL_TERM: &str = "s(x";

/// Per-alpha non-rejection counts of one test across the replications.
#[derive(Default)]
struct SizeTally {
    non_rejections: [usize; ALPHAS.len()],
    used: usize,
}

impl SizeTally {
    fn record(&mut self, p_value: f64, what: &str, rep: usize) {
        assert!(
            p_value.is_finite() && (0.0..=1.0).contains(&p_value),
            "rep {rep}: {what} p-value out of range: {p_value}"
        );
        self.used += 1;
        for (alpha_idx, &alpha) in ALPHAS.iter().enumerate() {
            if p_value > alpha {
                self.non_rejections[alpha_idx] += 1;
            }
        }
    }

    fn oversized(&self, what: &str) -> Vec<String> {
        assert!(
            self.used >= N_REPLICATIONS / 2,
            "{what}: too many degenerate replications ({}/{N_REPLICATIONS} usable)",
            self.used
        );
        let mut failures = Vec::new();
        for (alpha_idx, &alpha) in ALPHAS.iter().enumerate() {
            let verdict = audit_coverage(self.non_rejections[alpha_idx], self.used, 1.0 - alpha);
            if verdict.class == CoverageClass::AntiConservative {
                failures.push(format!(
                    "{what} alpha={alpha}: empirical non-reject rate={:.4} ({}/{}), \
                     Wilson CI=[{:.4},{:.4}] excludes nominal {} — rejects the true null \
                     too often",
                    verdict.empirical,
                    verdict.hits,
                    verdict.replications,
                    verdict.ci_lo,
                    verdict.ci_hi,
                    1.0 - alpha,
                ));
            }
        }
        failures
    }
}

#[test]
fn multinomial_three_class_null_term_tests_hold_their_size() {
    let mut rng = CalibrationRng::new(SEED);
    let mut joint = SizeTally::default();
    let mut per_class = [SizeTally::default(), SizeTally::default()];

    for rep in 0..N_REPLICATIONS {
        let rows: Vec<StringRecord> = (0..N_TRAIN)
            .map(|_| {
                let x = rng.uniform_open01();
                let z = rng.uniform_open01();
                // Class probabilities depend on z only; x is null for every class.
                let eta = [0.0, (2.0 * std::f64::consts::PI * z).sin(), 1.5 * (z - 0.5)];
                let weights = eta.map(f64::exp);
                let u = rng.uniform_open01() * weights.iter().sum::<f64>();
                let class = if u < weights[0] {
                    0
                } else if u < weights[0] + weights[1] {
                    1
                } else {
                    2
                };
                StringRecord::from(vec![
                    x.to_string(),
                    z.to_string(),
                    CLASSES[class].to_string(),
                ])
            })
            .collect();
        let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
        let data = encode_recordswith_inferred_schema(headers, rows)
            .expect("encode null multinomial dataset");

        let config = FitConfig::default();
        let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
            init_lambda: 1.0,
            max_iter: 60,
            tol: 1e-8,
            ..MultinomialFitRequest::new(&data, "y ~ s(x, k=8) + s(z, k=8)", &config)
        })
        .unwrap_or_else(|e| panic!("multinomial null fit failed (rep {rep}): {e:?}"));

        let joint_row = model
            .joint_smooth_significance()
            .into_iter()
            .find(|row| row.term_label.starts_with(NULL_TERM))
            .unwrap_or_else(|| panic!("rep {rep}: no joint row for {NULL_TERM}"));
        if let Ok(test) = &joint_row.test {
            joint.record(test.p_value, "joint", rep);
        }

        let class_rows: Vec<_> = model
            .smooth_significance()
            .into_iter()
            .filter(|row| row.term_label.starts_with(NULL_TERM))
            .collect();
        assert_eq!(
            class_rows.len(),
            per_class.len(),
            "rep {rep}: one per-class row per active class"
        );
        for (tally, row) in per_class.iter_mut().zip(&class_rows) {
            if let Ok(test) = &row.test {
                tally.record(test.p_value, "per-class", rep);
            }
        }
    }

    let mut failures = joint.oversized("joint (all classes)");
    for (index, tally) in per_class.iter().enumerate() {
        failures.extend(tally.oversized(&format!("per-class {} vs c", CLASSES[index])));
    }
    assert!(
        failures.is_empty(),
        "multinomial null-term tests are oversized:\n{}",
        failures.join("\n")
    );
}
