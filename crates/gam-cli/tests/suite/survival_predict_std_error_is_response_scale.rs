//! `gam predict --uncertainty` on a survival model publishes `std_error` as
//! the posterior SD of the published `survival_prob`, the response-scale
//! quantity every prediction table's `std_error` carries, and the link-scale
//! posterior SD of `eta` under its own `eta_std_error` name. The survival
//! table used to publish the eta SD under `std_error`, beside the
//! response-scale `mean_lower` / `mean_upper`, and drop the survival SD.

use std::process::Command;

fn run_gam(args: &[&std::ffi::OsStr], what: &str) {
    let output = Command::new(gam_test_support::gam_binary!())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn {what}: {e:?}"));
    assert!(
        output.status.success(),
        "{what} failed (exit {:?}).\nstderr tail: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn survival_predict_std_error_is_the_survival_posterior_sd() {
    let td = tempfile::tempdir().expect("temp dir");
    let train_path = td.path().join("surv_se.csv");
    let model_path = td.path().join("surv_se.model.json");
    let pred_path = td.path().join("surv_se.pred.csv");
    std::fs::write(
        &train_path,
        "entry,exit,event,x\n\
         10,15,1,-0.8\n\
         20,35,0,0.4\n\
         40,60,1,-0.2\n\
         80,100,0,0.7\n\
         120,150,1,0.1\n\
         160,220,1,-0.5\n",
    )
    .expect("write survival csv");
    run_gam(
        &[
            "fit".as_ref(),
            train_path.as_os_str(),
            "Surv(entry, exit, event) ~ x".as_ref(),
            "--survival-likelihood".as_ref(),
            "weibull".as_ref(),
            "--out".as_ref(),
            model_path.as_os_str(),
        ],
        "gam fit (Weibull)",
    );
    run_gam(
        &[
            "predict".as_ref(),
            model_path.as_os_str(),
            train_path.as_os_str(),
            "--uncertainty".as_ref(),
            "--covariance-mode".as_ref(),
            "conditional".as_ref(),
            "--out".as_ref(),
            pred_path.as_os_str(),
        ],
        "gam predict --uncertainty (Weibull)",
    );

    let mut reader = csv::Reader::from_path(&pred_path).expect("open prediction csv");
    let header: Vec<String> = reader
        .headers()
        .expect("prediction csv header")
        .iter()
        .map(str::to_string)
        .collect();
    assert_eq!(
        header,
        [
            "eta",
            "survival_prob_plugin",
            "survival_prob",
            "failure_prob",
            "risk_score",
            "eta_std_error",
            "std_error",
            "mean_lower",
            "mean_upper"
        ],
    );
    let column = |name: &str| {
        header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no `{name}` column"))
    };
    let (eta_se_at, se_at) = (column("eta_std_error"), column("std_error"));
    // Every cell is printed with 12 fixed decimals.
    let print_tol = 1e-11;
    let mut rows = 0;
    for record in reader.records() {
        let record = record.expect("prediction csv row");
        let cell = |at: usize| {
            record[at]
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("parse `{}`: {e:?}", &record[at]))
        };
        let (eta_se, se) = (cell(eta_se_at), cell(se_at));
        assert!(
            eta_se > 0.0,
            "row {rows}: eta SD must be positive, got {eta_se}"
        );
        assert!(
            se > 0.0,
            "row {rows}: survival SD must be positive, got {se}"
        );
        // `S(eta) = exp(-exp(eta))` has `|dS/deta| = S e^eta <= 1/e`, and a
        // function with Lipschitz constant L has `Var f(X) <= L^2 Var X`
        // (`Var f(X) = E[(f(X) - f(X'))^2] / 2` over an independent copy).
        // The survival SD is therefore at most the eta SD over e; the eta SD
        // itself breaks this bound.
        assert!(
            se <= eta_se / std::f64::consts::E + print_tol,
            "row {rows}: std_error {se} exceeds eta_std_error / e = {}; it is not \
             the posterior SD of survival_prob",
            eta_se / std::f64::consts::E
        );
        rows += 1;
    }
    assert_eq!(rows, 6);
}
