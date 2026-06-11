//! End-to-end ARD quality test for per-axis Duchon relevance.
//!
//! With `scale_dims=true`, a Duchon smooth emits one gradient penalty per input
//! axis, each its own REML λ_a. The PROOF that this works as automatic relevance
//! determination: fit a surface whose response depends ONLY on `x1`, with `x2`
//! a pure-noise covariate, and confirm REML shrinks `x2` out — the fitted
//! surface must be ~flat along `x2` while still recovering `f(x1)`. A
//! non-working relevance mechanism would let the flexible 2-D basis chase `x2`
//! noise, inflating the `x2` variation and the truth-recovery error.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn std_dev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

#[test]
fn scale_dims_shrinks_an_irrelevant_axis() {
    init_parallelism();

    // y = sin(2π·x1) + noise; x2 is pure noise, independent of y.
    let n = 400usize;
    let mut rng = StdRng::seed_from_u64(2024);
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let two_pi = 2.0 * std::f64::consts::PI;

    let x1: Vec<f64> = (0..n).map(|_| unif.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| unif.sample(&mut rng)).collect();
    let y: Vec<f64> = x1
        .iter()
        .map(|&t| (two_pi * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![x1[i].to_string(), x2[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let x1_idx = ds.column_map()["x1"];
    let x2_idx = ds.column_map()["x2"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x1, x2, k=24, scale_dims=true)", &ds, &cfg)
        .expect("gam duchon scale_dims fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    let predict = |grid: &Array2<f64>| -> Vec<f64> {
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild design at grid");
        design.design.apply(&fit.fit.beta).to_vec()
    };

    let m = 101usize;
    // Sweep A: vary x1 (the relevant axis), hold x2 at its midpoint.
    let mut grid_a = Array2::<f64>::zeros((m, ds.headers.len()));
    let mut truth = vec![0.0; m];
    for i in 0..m {
        let t = 0.005 + 0.99 * i as f64 / (m as f64 - 1.0);
        grid_a[[i, x1_idx]] = t;
        grid_a[[i, x2_idx]] = 0.5;
        truth[i] = (two_pi * t).sin();
    }
    let fitted_a = predict(&grid_a);

    // Sweep B: vary x2 (the IRRELEVANT axis), hold x1 at its midpoint.
    let mut grid_b = Array2::<f64>::zeros((m, ds.headers.len()));
    for i in 0..m {
        grid_b[[i, x1_idx]] = 0.5;
        grid_b[[i, x2_idx]] = 0.005 + 0.99 * i as f64 / (m as f64 - 1.0);
    }
    let fitted_b = predict(&grid_b);

    let truth_rmse = rmse(&fitted_a, &truth);
    let var_along_x1 = std_dev(&fitted_a);
    let var_along_x2 = std_dev(&fitted_b);

    eprintln!(
        "duchon-ard: truth_rmse={truth_rmse:.4} var(f|x1-sweep)={var_along_x1:.4} \
         var(f|x2-sweep)={var_along_x2:.4} ratio={:.3}",
        var_along_x2 / var_along_x1.max(1e-9)
    );

    // (a) The relevant axis is recovered: a constant predictor scores
    // RMS(sin)≈0.707; recovery means well below that.
    assert!(
        truth_rmse < 0.20,
        "per-axis-relevance Duchon failed to recover sin(2π·x1): RMSE-vs-truth={truth_rmse:.4}"
    );

    // (b) ARD: the irrelevant axis is shrunk — the surface varies far less along
    // x2 than along x1. REML drove x2's relevance λ up, flattening ∂f/∂x2.
    assert!(
        var_along_x2 < 0.15 * var_along_x1,
        "scale_dims did not shrink the irrelevant axis: var(f|x2)={var_along_x2:.4} is not \
         << var(f|x1)={var_along_x1:.4} (ratio {:.3}); REML should drive x2's relevance λ up",
        var_along_x2 / var_along_x1.max(1e-9)
    );
}
