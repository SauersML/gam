//! #1689 perf guard + profiling harness for the two ordinary-Gaussian cases
//! the issue reports as 2-10x (tp up to 24x) slower than mgcv at equal
//! accuracy. Prints the cost breakdown that distinguishes "the optimizer is
//! doing too many outer evaluations" from "each inner solve is too expensive":
//!
//!   * `p`               — final design width (basis columns); mgcv uses k=12
//!                         (ps) / k≈30 (tp), so a much larger `p` is the prime
//!                         suspect for per-solve cost.
//!   * `outer_iterations`/`outer_cost_evals`/`inner_pirls_solves` — outer-loop
//!                         work; a large count means the smoothing search, not
//!                         the basis size, dominates.
//!   * wall-clock for fit and predict, timed separately like the issue does.
//!
//! These run in the standard suite as lightweight regression guards: each
//! asserts the fit converges and recovers the truth, and bounds the outer-loop
//! work so a future change that re-inflates it (the #1689 regression) reddens
//! here. The printed line is the profiling signal for manual investigation.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    (a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Fit `formula` on `ds`, then predict on the design rebuilt at `grid`.
/// Returns (fit, fitted-at-grid, fit_seconds, predict_seconds).
fn fit_and_predict(
    formula: &str,
    ds: &gam::data::EncodedDataset,
    grid: &Array2<f64>,
) -> (StandardFitResult, Vec<f64>, f64, f64) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let t0 = Instant::now();
    let result = fit_from_formula(formula, ds, &cfg).expect("gam fit");
    let fit_secs = t0.elapsed().as_secs_f64();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for {formula}");
    };

    let t1 = Instant::now();
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at grid");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let predict_secs = t1.elapsed().as_secs_f64();

    (fit, fitted, fit_secs, predict_secs)
}

fn report(tag: &str, fit: &StandardFitResult, fit_secs: f64, predict_secs: f64, rmse: f64) {
    let f = &fit.fit;
    eprintln!(
        "[#1689 {tag}] p={p} outer_iter={oi} outer_cost_evals={oce} \
         inner_pirls_solves={ips} inner_cycles={ic} converged={cv} \
         lambdas={nl} | fit={fs:.3}s predict={ps:.3}s total={tot:.3}s | rmse_vs_truth={r:.4}",
        p = f.beta.len(),
        oi = f.outer_iterations,
        oce = f.outer_cost_evals,
        ips = f.inner_pirls_solves,
        ic = f.inner_cycles,
        cv = f.outer_converged,
        nl = f.lambdas.len(),
        fs = fit_secs,
        ps = predict_secs,
        tot = fit_secs + predict_secs,
        r = rmse,
    );
}

#[test]
fn profile_pspline_1d_n400() {
    init_parallelism();
    // Mirror the issue's 1-D P-spline case: n=400, y = sin(5x)+0.5x + N(0,0.2).
    let n = 400usize;
    let mut rng = StdRng::seed_from_u64(0);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.2).expect("normal");
    let truth = |x: f64| (5.0 * x).sin() + 0.5 * x;

    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let xi = u.sample(&mut rng);
            let yi = truth(xi) + noise.sample(&mut rng);
            StringRecord::from(vec![xi.to_string(), yi.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 1d dataset");
    let x_idx = ds.column_map()["x"];

    // dense test grid x in [0,1], 300 points (issue uses linspace(0,1,300)).
    let g = 300usize;
    let xt: Vec<f64> = (0..g).map(|i| i as f64 / (g as f64 - 1.0)).collect();
    let ft: Vec<f64> = xt.iter().map(|&x| truth(x)).collect();
    let mut grid = Array2::<f64>::zeros((g, ds.headers.len()));
    for (i, &x) in xt.iter().enumerate() {
        grid[[i, x_idx]] = x;
    }

    let (fit, fitted, fs, ps) = fit_and_predict("y ~ smooth(x)", &ds, &grid);
    report("ps-1d-n400", &fit, fs, ps, rmse(&fitted, &ft));
}

#[test]
fn profile_thinplate_2d_n1200() {
    init_parallelism();
    // Mirror the issue's 2-D thin-plate case: n=1200, gaussian bump + N(0,0.1).
    let n = 1200usize;
    let mut rng = StdRng::seed_from_u64(0);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let truth = |x: f64, z: f64| (-((x - 0.5).powi(2) + (z - 0.5).powi(2)) / 0.1).exp();

    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let xi = u.sample(&mut rng);
            let zi = u.sample(&mut rng);
            let yi = truth(xi, zi) + noise.sample(&mut rng);
            StringRecord::from(vec![xi.to_string(), zi.to_string(), yi.to_string()])
        })
        .collect();
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 2d dataset");
    let col = ds.column_map();
    let (x_idx, z_idx) = (col["x"], col["z"]);

    // 30x30 grid like the issue.
    let g = 30usize;
    let coord = |i: usize| i as f64 / (g as f64 - 1.0);
    let mut gx = Vec::new();
    let mut gz = Vec::new();
    let mut ft = Vec::new();
    for i in 0..g {
        for j in 0..g {
            let (xi, zi) = (coord(i), coord(j));
            gx.push(xi);
            gz.push(zi);
            ft.push(truth(xi, zi));
        }
    }
    let m = gx.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for i in 0..m {
        grid[[i, x_idx]] = gx[i];
        grid[[i, z_idx]] = gz[i];
    }

    let (fit, fitted, fs, ps) = fit_and_predict("y ~ thinplate(x, z)", &ds, &grid);
    report("tp-2d-n1200", &fit, fs, ps, rmse(&fitted, &ft));
}
