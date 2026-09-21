//! #3560: the Royston-Parmar `gam predict --uncertainty` band is the image of
//! the eta credible interval under `S = exp(-exp(eta))`, i.e.
//! `[exp(-exp(eta + z*se)), exp(-exp(eta - z*se))]`. It used to be the
//! posterior mean `+/- z*sd(S)` clamped to `[0, 1]`, which is not a central
//! interval for a skewed `S` (one-tailed at either rail).
//!
//! The survival table's columns are the ones #4533 settled: `std_error` is the
//! posterior SD of `survival_prob`, the response-scale quantity the band
//! describes, and the link-scale SD of `eta` has its own `eta_std_error`
//! column. The band bound this test reconstructs is `z` standard errors of
//! ETA, so it reads `eta_std_error`.

use std::process::Command;

use gam::inference::model::FittedModel;
use gam_math::probability::standard_normal_quantile;

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
fn cli_royston_parmar_survival_band_is_the_transformed_eta_interval_3560() {
    let td = tempfile::tempdir().expect("temp dir");
    let train_path = td.path().join("rp_band.csv");
    let model_path = td.path().join("rp_band.model.json");
    let pred_path = td.path().join("rp_band.pred.csv");
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

    let level = 0.95_f64;
    let level_arg = level.to_string();
    run_gam(
        &[
            "predict".as_ref(),
            model_path.as_os_str(),
            train_path.as_os_str(),
            "--uncertainty".as_ref(),
            "--level".as_ref(),
            level_arg.as_ref(),
            "--covariance-mode".as_ref(),
            "conditional".as_ref(),
            "--out".as_ref(),
            pred_path.as_os_str(),
        ],
        "gam predict --uncertainty (Royston-Parmar)",
    );

    let text = std::fs::read_to_string(&pred_path).expect("read prediction csv");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
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
    // An active monotone-baseline constraint gives the eta interval of the
    // persisted truncated law instead of eta +/- z*se; the band is then still
    // that interval's image, which contains the plug-in survival.
    let saved = FittedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("load fitted model: {e:?}"));
    let constrained = saved
        .fit_result
        .as_ref()
        .and_then(|fit| fit.geometry.as_ref())
        .is_some_and(|geometry| geometry.constrained_posterior.is_some());
    let z = standard_normal_quantile(0.5 + 0.5 * level)
        .unwrap_or_else(|e| panic!("normal quantile: {e:?}"));
    let survival = |eta: f64| (-eta.exp()).exp();
    // Every column is printed with 12 fixed decimals, and |dS/deta| <= 1/e.
    let print_tol = 1e-11;
    let mut rows = 0;
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let v: Vec<f64> = line
            .split(',')
            .map(|c| {
                c.parse::<f64>()
                    .unwrap_or_else(|e| panic!("parse {c}: {e:?}"))
            })
            .collect();
        let (eta, plugin, eta_se, response_sd, lo, hi) = (v[0], v[1], v[5], v[6], v[7], v[8]);
        assert!(
            eta_se > 0.0,
            "a fitted row must carry a positive eta SE, got {eta_se}"
        );
        assert!(
            response_sd > 0.0,
            "a fitted row must carry a positive survival-scale SD, got {response_sd}"
        );
        assert!(
            0.0 < lo && lo < hi && hi < 1.0,
            "row {rows}: the band [{lo}, {hi}] must lie strictly inside (0, 1)"
        );
        assert!(
            lo - print_tol <= plugin && plugin <= hi + print_tol,
            "row {rows}: the band [{lo}, {hi}] must contain the plug-in survival {plugin}"
        );
        if !constrained {
            let (want_lo, want_hi) = (survival(eta + z * eta_se), survival(eta - z * eta_se));
            for (got, want, side) in [(lo, want_lo, "lower"), (hi, want_hi, "upper")] {
                assert!(
                    (got - want).abs() <= print_tol,
                    "row {rows}: {side} band {got} is not the transformed eta bound {want}"
                );
            }
        }
        rows += 1;
    }
    assert_eq!(rows, 6);
}
