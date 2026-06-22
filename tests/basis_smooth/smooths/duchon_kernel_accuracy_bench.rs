//! Accuracy bench: the two Duchon spectral powers vs mgcv, across scenarios.
//!
//! For each scenario we fit the SAME data three ways and report truth-recovery
//! RMSE on a dense interior grid:
//!   * gam `r³`       — Duchon `s = (d−1)/2` (the magic cubic default)
//!   * gam `r²·log r` — Duchon `s = 0` (explicit `power=0`; the integer-order
//!                       Duchon kernel ≡ the thin-plate kernel)
//!   * mgcv `bs="ds", m=c(2,0)` — the mature Duchon (`r²·log r`) baseline
//!
//! This is a COMPARISON bench: it prints a table (run with `--nocapture`) and
//! asserts that every gam fit genuinely recovers the signal (beats the trivial
//! mean predictor). The low-noise 1D easy case is also an explicit regression
//! gate for issue #504: gam must match mgcv's truth-recovery sharpness instead
//! of selecting an over-smooth REML solution.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn gam_grid_fit(formula: &str, ds: &gam::data::EncodedDataset, grid: &Array2<f64>) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(formula, ds, &cfg).unwrap_or_else(|e| panic!("gam fit '{formula}': {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for '{formula}'");
    };
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("rebuild design for '{formula}': {e}"));
    design.design.apply(&fit.fit.beta).to_vec()
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|t| t * t).sum::<f64>() / v.len() as f64).sqrt()
}

/// Print one comparison row and assert both gam variants genuinely beat the
/// trivial (zero/mean) predictor, whose RMSE is `rms_truth`. A real
/// reconstruction sits clearly below it; 70% of `rms_truth` is a generous
/// "recovering, not blown up / collapsed" floor. The per-variant asserts here
/// are the substance of each bench `#[test]`: the tests delegate their checks
/// to this helper rather than duplicating the floor logic at every call site.
fn report_and_check(
    label: &str,
    rmse_r3: f64,
    rmse_r2logr: f64,
    rmse_mgcv: f64,
    rms_truth: f64,
    mgcv_match_factor: Option<f64>,
) {
    let winner = if rmse_r3 <= rmse_r2logr && rmse_r3 <= rmse_mgcv {
        "gam r³"
    } else if rmse_r2logr <= rmse_r3 && rmse_r2logr <= rmse_mgcv {
        "gam r²·log r"
    } else {
        "mgcv"
    };
    eprintln!(
        "{label:28} | gam r³={rmse_r3:.4} | gam r²·logr={rmse_r2logr:.4} | mgcv={rmse_mgcv:.4} \
         | rms_truth={rms_truth:.4} | best: {winner}"
    );
    let floor = 0.70 * rms_truth;
    assert!(
        rmse_r3 < floor,
        "{label}: gam r³ did not recover (rmse {rmse_r3:.4} ≥ {floor:.4})"
    );
    assert!(
        rmse_r2logr < floor,
        "{label}: gam r²·log r did not recover (rmse {rmse_r2logr:.4} ≥ {floor:.4})"
    );
    if let Some(factor) = mgcv_match_factor {
        let limit = factor * rmse_mgcv;
        assert!(
            rmse_r3 <= limit,
            "{label}: gam r³ should match-or-beat mgcv sharpness (rmse {rmse_r3:.4} > {limit:.4}; mgcv {rmse_mgcv:.4})"
        );
        assert!(
            rmse_r2logr <= limit,
            "{label}: gam r²·log r should match-or-beat mgcv sharpness (rmse {rmse_r2logr:.4} > {limit:.4}; mgcv {rmse_mgcv:.4})"
        );
    }
}

#[test]
fn bench_duchon_kernel_accuracy_1d() {
    init_parallelism();
    eprintln!("\n=== Duchon kernel accuracy (1D), truth recovery RMSE ===");

    // (freq cycles over [0,1], sigma, n, k)
    let scenarios = [
        ("sin 4-cycle σ.05 k20", 4.0, 0.05, 200usize, 20usize, 11u64),
        ("sin 8-cycle σ.10 k40", 8.0, 0.10, 240, 40, 12),
        ("sin 2-cycle σ.20 k15", 2.0, 0.20, 200, 15, 13),
    ];

    for (label, freq, sigma, n, k, seed) in scenarios {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, sigma).expect("normal");
        let two_pi_f = 2.0 * std::f64::consts::PI * freq;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&t| (two_pi_f * t).sin() + noise.sample(&mut rng))
            .collect();

        let headers = ["x", "y"].into_iter().map(String::from).collect();
        let rows = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
            .collect();
        let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 1d");
        let x_idx = ds.column_map()["x"];

        let m = 201usize;
        let x_test: Vec<f64> = (0..m)
            .map(|i| 0.005 + 0.99 * i as f64 / (m as f64 - 1.0))
            .collect();
        let y_truth: Vec<f64> = x_test.iter().map(|&t| (two_pi_f * t).sin()).collect();
        let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
        for (i, &t) in x_test.iter().enumerate() {
            grid[[i, x_idx]] = t;
        }

        let r3 = gam_grid_fit(&format!("y ~ duchon(x, k={k})"), &ds, &grid);
        let r2logr = gam_grid_fit(&format!("y ~ duchon(x, k={k}, power=0)"), &ds, &grid);

        let mut x_all = x.clone();
        x_all.extend_from_slice(&x_test);
        let mut y_all = y.clone();
        y_all.extend(std::iter::repeat_n(0.0, m));
        let mut is_train = vec![1.0; n];
        is_train.extend(std::iter::repeat_n(0.0, m));
        let r = run_r(
            &[
                Column::new("x", &x_all),
                Column::new("y", &y_all),
                Column::new("is_train", &is_train),
            ],
            &format!(
                r#"
                suppressPackageStartupMessages(library(mgcv))
                train <- df[df$is_train > 0.5, ]; grid <- df[df$is_train < 0.5, ]
                m <- gam(y ~ s(x, bs="ds", k={k}, m=c(2,0)), data=train, method="REML")
                emit("fitted", as.numeric(predict(m, newdata=grid)))
                "#
            ),
        );
        let mgcv = r.vector("fitted");

        let mgcv_match_factor = if label == "sin 4-cycle σ.05 k20" {
            // Issue #504: the easy, low-noise case used to over-smooth at
            // ≈0.034 RMSE while mgcv was ≈0.014. The new center geometry must
            // keep gam on mgcv's truth-recovery scale without weakening the
            // hard-case recovery checks below.
            Some(1.10)
        } else {
            None
        };
        report_and_check(
            label,
            rmse(&r3, &y_truth),
            rmse(&r2logr, &y_truth),
            rmse(mgcv, &y_truth),
            rms(&y_truth),
            mgcv_match_factor,
        );
    }
}

#[test]
fn bench_duchon_kernel_accuracy_2d() {
    init_parallelism();
    eprintln!("\n=== Duchon kernel accuracy (2D), truth recovery RMSE ===");

    // Two truths: a smooth two-bump surface and a directional ripple.
    let two_bump = |x: f64, z: f64| {
        let b = |cx: f64, cz: f64, s: f64, a: f64| {
            a * (-((x - cx).powi(2) + (z - cz).powi(2)) / (2.0 * s * s)).exp()
        };
        b(0.3, 0.3, 0.18, 1.0) + b(0.7, 0.65, 0.22, 0.8)
    };
    let ripple = |x: f64, z: f64| (2.0 * std::f64::consts::PI * (x + z)).sin();

    let scenarios: [(&str, &dyn Fn(f64, f64) -> f64, f64, usize, usize, u64); 2] = [
        ("two-bump σ.10 k49", &two_bump, 0.10, 400, 49, 21),
        ("ripple x+z σ.08 k64", &ripple, 0.08, 500, 64, 22),
    ];

    for (label, truth, sigma, n, k, seed) in scenarios {
        let mut rng = StdRng::seed_from_u64(seed);
        let u = Uniform::new(0.0_f64, 1.0).expect("u");
        let noise = Normal::new(0.0, sigma).expect("normal");
        let (mut x, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let (xi, zi) = (u.sample(&mut rng), u.sample(&mut rng));
            x.push(xi);
            z.push(zi);
            y.push(truth(xi, zi) + noise.sample(&mut rng));
        }
        let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
        let rows = (0..n)
            .map(|i| {
                csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
            })
            .collect();
        let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 2d");
        let (x_idx, z_idx) = (ds.column_map()["x"], ds.column_map()["z"]);

        let g = 25usize;
        let coord = |i: usize| 0.05 + 0.90 * i as f64 / (g as f64 - 1.0);
        let (mut gx, mut gz, mut y_truth) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..g {
            for j in 0..g {
                let (xi, zi) = (coord(i), coord(j));
                gx.push(xi);
                gz.push(zi);
                y_truth.push(truth(xi, zi));
            }
        }
        let mlen = gx.len();
        let mut grid = Array2::<f64>::zeros((mlen, ds.headers.len()));
        for i in 0..mlen {
            grid[[i, x_idx]] = gx[i];
            grid[[i, z_idx]] = gz[i];
        }

        let r3 = gam_grid_fit(&format!("y ~ duchon(x, z, k={k})"), &ds, &grid);
        let r2logr = gam_grid_fit(&format!("y ~ duchon(x, z, k={k}, power=0)"), &ds, &grid);

        let mut x_all = x.clone();
        x_all.extend_from_slice(&gx);
        let mut z_all = z.clone();
        z_all.extend_from_slice(&gz);
        let mut y_all = y.clone();
        y_all.extend(std::iter::repeat_n(0.0, mlen));
        let mut is_train = vec![1.0; n];
        is_train.extend(std::iter::repeat_n(0.0, mlen));
        let r = run_r(
            &[
                Column::new("x", &x_all),
                Column::new("z", &z_all),
                Column::new("y", &y_all),
                Column::new("is_train", &is_train),
            ],
            &format!(
                r#"
                suppressPackageStartupMessages(library(mgcv))
                train <- df[df$is_train > 0.5, ]; grid <- df[df$is_train < 0.5, ]
                m <- gam(y ~ s(x, z, bs="ds", k={k}, m=c(2,0)), data=train, method="REML")
                emit("fitted", as.numeric(predict(m, newdata=grid)))
                "#
            ),
        );
        let mgcv = r.vector("fitted");

        report_and_check(
            label,
            rmse(&r3, &y_truth),
            rmse(&r2logr, &y_truth),
            rmse(mgcv, &y_truth),
            rms(&y_truth),
            None,
        );
    }
}
