//! `smooth(x)`, `matern(x)`, `duchon(x)`, etc., previously silently fit a
//! constant predictor `x` (only one unique value) to the mean of `y`,
//! returning identical predictions for all `x_new`. That's mathematically
//! meaningless and a quiet bug — the user sees a "successful fit" with no
//! indication that the design is rank-1 garbage.
//!
//! After the term-builder validation, any smooth with a constant feature
//! column must error at term construction with a clear actionable message
//! naming the offending variable.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

fn try_fit(formula: &str) -> Result<(), String> {
    let n = 50usize;
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec!["0.5".to_string(), format!("{:.4}", 0.1 * i as f64)]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    fit_from_formula(formula, &data, &cfg).map(|_| ())
}

#[test]
fn smooth_on_constant_column_errors_with_clear_message() {
    init_parallelism();
    let formulas = &[
        "y ~ smooth(x)",
        "y ~ s(x)",
        "y ~ matern(x)",
        "y ~ duchon(x)",
    ];
    for f in formulas {
        let err = try_fit(f).expect_err(&format!(
            "expected `{f}` to error on constant-x training data, but fit succeeded"
        ));
        let lower = err.to_lowercase();
        let names_var =
            lower.contains("'x'") || lower.contains(" x ") || lower.contains("variable 'x'");
        let names_problem = lower.contains("constant")
            || lower.contains("one unique value")
            || lower.contains("degenerate");
        assert!(
            names_var && names_problem,
            "`{f}` errored but the message is not user-friendly — missing var name or 'constant/degenerate' diagnosis: {err}",
        );
        eprintln!(
            "[smooth-constant] OK   {f} -> {}",
            err.lines().next().unwrap_or("")
        );
    }
}
