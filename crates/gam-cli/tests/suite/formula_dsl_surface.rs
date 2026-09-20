//! End-to-end CLI coverage of the formula DSL surface: intercept removal
//! (`0 + x`), backtick-quoted column names, `C()` refused in favour of `factor()`, the
//! `domain=[a, b]` spline option, and strict option parsing. Each case goes
//! through `gam fit` / `gam predict` exactly as a user would.

use std::path::Path;
use std::process::{Command, Output};

fn gam(args: &[&str]) -> Output {
    Command::new(gam_test_support::gam_binary!())
        .args(args)
        .output()
        .expect("spawn gam CLI")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("UTF-8 path")
}

fn posterior_mean(csv: &str) -> Vec<f64> {
    let mut lines = csv.lines();
    let header = lines.next().expect("prediction CSV has a header row");
    let column = header
        .split(',')
        .position(|name| name.trim() == "posterior_mean")
        .expect("prediction CSV has a `posterior_mean` column");
    lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(',')
                .nth(column)
                .expect("row has the posterior_mean field")
                .trim()
                .parse::<f64>()
                .expect("posterior_mean parses as f64")
        })
        .collect()
}

/// Deterministic design: x on (0, 2], response with a nonzero intercept so an
/// origin fit and an intercept fit genuinely disagree.
fn linear_rows() -> (Vec<f64>, Vec<f64>) {
    let n = 40;
    let x: Vec<f64> = (1..=n).map(|i| 2.0 * i as f64 / n as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| 1.5 + 0.8 * xi + 0.1 * ((i * 7 % 11) as f64 - 5.0) / 5.0)
        .collect();
    (x, y)
}

fn fit_and_predict(dir: &Path, csv_body: &str, formula: &str) -> Vec<f64> {
    let data = dir.join("train.csv");
    let model = dir.join("model.gam");
    let pred = dir.join("pred.csv");
    std::fs::write(&data, csv_body).expect("write training CSV");
    let fit = gam(&["fit", path_str(&data), formula, "--out", path_str(&model)]);
    assert!(
        fit.status.success(),
        "fit `{formula}` failed: {}",
        stderr(&fit)
    );
    let predict = gam(&[
        "predict",
        path_str(&model),
        path_str(&data),
        "--out",
        path_str(&pred),
    ]);
    assert!(
        predict.status.success(),
        "predict `{formula}` failed: {}",
        stderr(&predict)
    );
    posterior_mean(&std::fs::read_to_string(&pred).expect("read predictions"))
}

#[test]
fn zero_plus_x_is_least_squares_through_the_origin() {
    let (x, y) = linear_rows();
    let mut csv = String::from("y,x\n");
    for (xi, yi) in x.iter().zip(&y) {
        csv.push_str(&format!("{yi},{xi}\n"));
    }
    let slope =
        x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / x.iter().map(|a| a * a).sum::<f64>();

    // An unpenalized slope with the intercept removed is exactly OLS through
    // the origin; both spellings of intercept removal lower to that model.
    for formula in [
        "y ~ 0 + linear(x, double_penalty=false)",
        "y ~ linear(x, double_penalty=false) - 1",
    ] {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let fitted = fit_and_predict(scratch.path(), &csv, formula);
        assert_eq!(fitted.len(), x.len());
        for (xi, fi) in x.iter().zip(&fitted) {
            let ols = slope * xi;
            assert!(
                (fi - ols).abs() <= 1e-9 * (1.0 + ols.abs()),
                "`{formula}` at x={xi}: fitted {fi} vs origin least squares {ols}"
            );
        }
    }

    // The default penalized slope keeps its REML shrinkage ridge, but the
    // fit still has no constant: every fitted value is one slope times x.
    let scratch = tempfile::tempdir().expect("scratch directory");
    let fitted = fit_and_predict(scratch.path(), &csv, "y ~ 0 + x");
    let ratio = fitted[0] / x[0];
    for (xi, fi) in x.iter().zip(&fitted) {
        assert!(
            (fi - ratio * xi).abs() <= 1e-9 * (1.0 + fi.abs()),
            "`y ~ 0 + x` at x={xi}: fitted {fi} is not proportional to x (slope {ratio})"
        );
    }
    assert!(
        ratio > 0.0 && ratio <= slope * (1.0 + 1e-12),
        "slope {ratio} vs OLS {slope}"
    );
}

#[test]
fn backtick_column_names_fit_and_c_is_refused_through_the_cli() {
    let (x, y) = linear_rows();
    let mut csv = String::from("y,dose (mg),site-id\n");
    for (i, (xi, yi)) in x.iter().zip(&y).enumerate() {
        let site = ["north", "south", "east"][i % 3];
        let shift = [0.0, 0.5, -0.5][i % 3];
        csv.push_str(&format!("{},{xi},{site}\n", yi + shift));
    }
    let scratch = tempfile::tempdir().expect("scratch directory");
    let fitted = fit_and_predict(scratch.path(), &csv, "y ~ `dose (mg)` + factor(`site-id`)");
    assert_eq!(fitted.len(), x.len());
    assert!(fitted.iter().all(|v| v.is_finite()));

    let data = scratch.path().join("train.csv");
    let model = scratch.path().join("refused.gam");
    let out = gam(&[
        "fit",
        path_str(&data),
        "y ~ `dose (mg)` + C(`site-id`)",
        "--out",
        path_str(&model),
    ]);
    assert_eq!(
        out.status.code(),
        Some(gam::ErrorCategory::Formula.exit_code()),
        "{}",
        stderr(&out)
    );
    let error = stderr(&out);
    assert!(error.contains("`C()` is not a term function"), "{error}");
    assert!(error.contains("factor(`site-id`)"), "{error}");
}

#[test]
fn domain_that_excludes_training_data_is_rejected_by_name() {
    let (x, y) = linear_rows();
    let mut csv = String::from("y,x\n");
    for (xi, yi) in x.iter().zip(&y) {
        csv.push_str(&format!("{yi},{xi}\n"));
    }
    let scratch = tempfile::tempdir().expect("scratch directory");
    let data = scratch.path().join("train.csv");
    let model = scratch.path().join("model.gam");
    std::fs::write(&data, &csv).expect("write training CSV");

    let out = gam(&[
        "fit",
        path_str(&data),
        "y ~ s(x, domain=[0.5, 1.0])",
        "--out",
        path_str(&model),
    ]);
    assert_eq!(
        out.status.code(),
        Some(gam::ErrorCategory::Formula.exit_code()),
        "{}",
        stderr(&out)
    );
    let error = stderr(&out);
    assert!(error.contains("domain"), "{error}");
    assert!(error.contains("s(x"), "{error}");

    let widened = gam(&[
        "fit",
        path_str(&data),
        "y ~ s(x, domain=[0, 3])",
        "--out",
        path_str(&model),
    ]);
    assert!(widened.status.success(), "{}", stderr(&widened));
}

#[test]
fn malformed_option_values_fail_naming_term_and_option() {
    let (x, y) = linear_rows();
    let mut csv = String::from("y,x\n");
    for (xi, yi) in x.iter().zip(&y) {
        csv.push_str(&format!("{yi},{xi}\n"));
    }
    let scratch = tempfile::tempdir().expect("scratch directory");
    let data = scratch.path().join("train.csv");
    let model = scratch.path().join("model.gam");
    std::fs::write(&data, &csv).expect("write training CSV");

    for (formula, needles) in [
        ("y ~ s(x, k=ten)", &["s(x", "k=ten"][..]),
        (
            "y ~ s(x, degree=2, penalty_order=3)",
            &["s(x", "penalty_order=3"][..],
        ),
    ] {
        let out = gam(&["fit", path_str(&data), formula, "--out", path_str(&model)]);
        assert_eq!(
            out.status.code(),
            Some(gam::ErrorCategory::Formula.exit_code()),
            "`{formula}`: {}",
            stderr(&out)
        );
        let error = stderr(&out);
        for needle in needles {
            assert!(
                error.contains(needle),
                "`{formula}` error lacks `{needle}`: {error}"
            );
        }
    }
}
