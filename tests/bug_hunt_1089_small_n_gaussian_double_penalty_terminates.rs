//! #1089 regression: a standard Gaussian-identity GAM on a small dataset
//! (n=30) with five `ps` smooths under a double (shrinkage) penalty —
//! mirroring the `wine_gamair` benchmark scenario — must terminate the outer
//! REML loop in a bounded number of iterations rather than spinning through
//! hundreds of thousands of cost-only evaluations until a wall-clock budget
//! kills it.
//!
//! The original failure: `gam fit` on the n=30 wine fold (`family=gaussian`,
//! 5 `s(., type=ps, knots=7, double_penalty=true)` smooths → `rho_dim=10`)
//! performed ~850,000 cost-only outer evaluations and never satisfied its
//! stopping criterion. The inner P-IRLS converged instantly every time; the
//! defect was purely in the outer smoothing-parameter optimizer's
//! termination on this tiny, fully-penalized design.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// A smooth Gaussian truth over five covariates, evaluated at one row.
fn truth(x: &[f64; 5]) -> f64 {
    (1.3 * x[0]).sin() + 0.5 * x[1] * x[1] - 0.7 * x[2] + (0.4 * x[3]).cos()
        + 0.2 * (x[4] - 0.5).abs()
}

fn build_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x0", "x1", "x2", "x3", "x4", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let xs: [f64; 5] = [
                ux.sample(&mut rng),
                ux.sample(&mut rng),
                ux.sample(&mut rng),
                ux.sample(&mut rng),
                ux.sample(&mut rng),
            ];
            let y = truth(&xs) + noise.sample(&mut rng);
            let mut fields: Vec<String> = xs.iter().map(|v| v.to_string()).collect();
            fields.push(y.to_string());
            StringRecord::from(fields)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn small_n_gaussian_double_penalty_outer_loop_terminates() {
    // n=30 matches the wine_gamair CV-fold size that triggered the hang.
    let data = build_data(30, 0.1, 1089);

    // Exactly the wine_gamair Rust formula shape: 5 ps smooths, knots=7,
    // double penalty on each (rho_dim = 10).
    let formula = "y ~ s(x0, type=ps, knots=7, double_penalty=true) \
                   + s(x1, type=ps, knots=7, double_penalty=true) \
                   + s(x2, type=ps, knots=7, double_penalty=true) \
                   + s(x3, type=ps, knots=7, double_penalty=true) \
                   + s(x4, type=ps, knots=7, double_penalty=true)";

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let start = std::time::Instant::now();
    let result = fit_from_formula(formula, &data, &cfg).expect("small-n double-penalty fit");
    let elapsed = start.elapsed();

    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // The outer optimizer must converge, not bail at its iteration cap.
    assert!(
        fit.fit.outer_converged,
        "outer REML loop did not converge on the n=30 double-penalty Gaussian fit \
         (outer_iterations={})",
        fit.fit.outer_iterations,
    );

    // A toy n=30 / rho_dim=10 fit must settle in a handful of outer steps. The
    // pre-fix behavior performed ~850k cost-only evals; any working stopping
    // criterion lands far below this bound.
    assert!(
        fit.fit.outer_iterations <= 200,
        "outer REML loop took {} iterations on a trivial n=30 fit; \
         expected prompt termination",
        fit.fit.outer_iterations,
    );

    // Wall-clock guard: the benchmark died at a 42-minute budget. A few
    // seconds is a generous ceiling for this toy problem on CI hardware.
    assert!(
        elapsed.as_secs() < 60,
        "n=30 double-penalty Gaussian fit took {:.1}s; outer loop is not \
         terminating promptly",
        elapsed.as_secs_f64(),
    );
}
