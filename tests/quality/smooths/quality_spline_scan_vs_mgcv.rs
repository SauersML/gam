//! #1034 item 4 — external-reference match-or-beat gate for the exact O(n)
//! spline scan (#1030) against `mgcv`, the mature standard tool.
//!
//! #904 paradigm: the PRIMARY assertion is OBJECTIVE truth recovery against a
//! self-constructed smooth truth — never closeness to mgcv's own (noisy)
//! fitted output. Both estimators see the IDENTICAL data; both are scored by
//! RMSE of their fitted function against the noise-free truth on an interior
//! grid; gam's scan-routed fit must match-or-beat mgcv's
//! `gam(y ~ s(x, bs = "cr", k = large), method = "REML")` within 10%.
//!
//! Skip convention (the narrow environmental-gate escape hatch documented on
//! `r_package_available`): the gam-side ABSOLUTE truth-recovery bar is
//! asserted unconditionally; only the match-or-beat-vs-mgcv arm is skipped —
//! with an honest message — when `Rscript`/`mgcv` is genuinely absent.

use csv::StringRecord;
use gam::test_support::reference::{
    Column, QualityPair, pad_to, r_package_available, rmse, run_r,
};
use gam::{
    FitConfig, encode_recordswith_inferred_schema, fit_spline_scan_from_formula, init_parallelism,
};
use std::process::Command;

/// Known smooth truth (same family as the scan workflow gates): curvature-rich
/// but well within what a cubic smoothing spline resolves at moderate n.
fn truth_fn(x: f64) -> f64 {
    (6.0 * x).sin() + 0.5 * x * x
}

/// Deterministic truth + quasi-noise training data (no RNG): irregular
/// strictly-increasing abscissae, golden-ratio rotation noise with half-range
/// 0.15 (noise sd ≈ 0.087). Identical rows feed gam and mgcv.
fn training_xy(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let u = i as f64 / (n - 1) as f64;
        let xi = u + 0.35 * (std::f64::consts::PI * u).sin() / (n as f64);
        let noise = ((i as f64 * 0.618_033_988_749_894_9).fract() - 0.5) * 0.3;
        x.push(xi);
        y.push(truth_fn(xi) + noise);
    }
    (x, y)
}

fn encode_xy(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

/// Probe for a usable `Rscript` binary without failing the test: the spawn
/// itself errs when the interpreter is genuinely absent. This gates ONLY the
/// match-or-beat arm (`r_package_available` would panic on a missing binary
/// before it could report the package as unavailable).
fn rscript_available() -> bool {
    Command::new("Rscript").arg("--version").output().is_ok()
}

#[test]
fn spline_scan_matches_or_beats_mgcv_on_truth_recovery() {
    init_parallelism();
    let n = 600;
    let (x, y) = training_xy(n);
    let data = encode_xy(&x, &y);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // ---- gam: the exact O(n) scan path (#1030 wiring, #1034 item 4) -------
    let scan = fit_spline_scan_from_formula("y ~ s(x, double_penalty=false)", &data, &cfg)
        .expect("scan-routed fit")
        .expect("detection must fire for a single 1-D single-penalty Gaussian smooth");

    // Interior evaluation grid (truth-recovery objective metric).
    let grid: Vec<f64> = (0..200).map(|i| 0.02 + 0.96 * i as f64 / 199.0).collect();
    let truth: Vec<f64> = grid.iter().map(|&t| truth_fn(t)).collect();
    let gam_pred: Vec<f64> = grid
        .iter()
        .map(|&t| {
            let (mean, var) = scan.predict(t).expect("scan predict");
            assert!(
                mean.is_finite() && var.is_finite() && var > 0.0,
                "scan prediction must be finite with positive variance at x={t}"
            );
            mean
        })
        .collect();
    let gam_rmse = rmse(&gam_pred, &truth);

    // EDF sanity: a real smooth was fit, not the linear trend or an
    // interpolant.
    let edf = scan.edf();
    assert!(
        edf > 2.0 && edf < n as f64,
        "scan EDF {edf} outside the sane band (2, n) at n={n}"
    );

    // ABSOLUTE truth-recovery bar (unconditional, #904 PRIMARY): the injected
    // noise sd is ≈ 0.087; at n=600 the smooth must be resolved well below it.
    assert!(
        gam_rmse < 0.08,
        "scan-routed fit fails absolute truth recovery: RMSE={gam_rmse}"
    );

    // ---- mgcv match-or-beat arm (environmental gate, honest skip) ----------
    if !rscript_available() {
        eprintln!(
            "SKIP (match-or-beat arm only): Rscript is not installed; the gam-side \
             absolute truth-recovery bar above already passed. Install R + mgcv to \
             run the external-reference gate."
        );
        return;
    }
    if !r_package_available("mgcv") {
        eprintln!(
            "SKIP (match-or-beat arm only): R is present but the mgcv package is \
             not loadable; the gam-side absolute truth-recovery bar above already \
             passed. install.packages('mgcv') to run the external-reference gate."
        );
        return;
    }

    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("y", &y),
            Column::new("grid", &pad_to(&grid, n)),
            Column::new("grid_n", &vec![grid.len() as f64; n]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x, bs = "cr", k = 150), data = df, method = "REML")
        k <- df$grid_n[1]
        newd <- data.frame(x = df$grid[1:k])
        emit("pred", as.numeric(predict(m, newdata = newd)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_pred = r.vector("pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_pred.len(),
        grid.len(),
        "mgcv grid-prediction length mismatch"
    );
    let mgcv_rmse = rmse(mgcv_pred, &truth);

    eprintln!(
        "[spline-scan vs mgcv] n={n} scan_edf={edf:.3} mgcv_edf={mgcv_edf:.3} \
         rmse_to_truth(gam)={gam_rmse:.5} rmse_to_truth(mgcv)={mgcv_rmse:.5}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_spline_scan_vs_mgcv",
            "rmse_to_truth",
            gam_rmse,
            "mgcv",
            mgcv_rmse,
        )
        .line()
    );

    // Match-or-beat on the OBJECTIVE metric: gam's truth-recovery error must
    // be no worse than mgcv's by more than 10%. mgcv is an accuracy BASELINE,
    // not a target to reproduce.
    assert!(
        gam_rmse <= 1.10 * mgcv_rmse + 1e-12,
        "scan-routed fit worse than mgcv on truth recovery: \
         rmse(gam)={gam_rmse:.5} vs 1.10*rmse(mgcv)={:.5}",
        1.10 * mgcv_rmse
    );
}
