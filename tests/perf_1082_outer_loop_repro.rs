//! #1082 regression guard for the STANDARD exponential-family REML outer-loop
//! cycling on a Negative-Binomial 2-D tensor-product smooth.
//!
//! Root cause (verified): with the NB overdispersion `theta` ESTIMATED, the
//! inner solver re-derived `theta` from each outer iterate's warm-start `eta`,
//! so the NB working response / deviance / penalty-logdet, and hence the REML
//! criterion, drifted on every outer evaluation. The outer optimizer then
//! chased a moving target: the projected-gradient convergence test never tripped
//! and the loop ground to `max_iter` (200) without converging, the #1082
//! `gam_tensor_te_2d_negbin` wall-clock timeout. An otherwise-identical Poisson
//! fit, with no estimated dispersion, converges in a handful of outer iterations.
//!
//! Fix: freeze `theta` for the duration of the smoothing-parameter lambda search
//! (`GlmLikelihoodSpec::with_negbin_theta_frozen_for_search`, driven by the REML
//! state's `frozen_negbin_theta`), so `F(rho) = REML(rho, theta_frozen)` is a
//! stationary function of rho and the loop converges quickly. `theta` is still
//! ML-refreshed at the single final, reported accept-fit.
//!
//! This is intentionally a standalone integration test, not a module in
//! `tests/regressions.rs`: the guard is small and pure Rust, so it should be
//! runnable directly with `cargo test --test perf_1082_outer_loop_repro`.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;
use std::time::Instant;

/// Build the `(headers, rows)` for a small Poisson-count dataset on the unit
/// square drawn from the smooth log-mean truth `eta(x,z) = sin(pi x) cos(pi z)`,
/// columns `x`, `z`, `count`. Kept as records (not the encoded dataset) so the
/// helper has no dependency on the crate-internal dataset type name.
fn synthetic_count_records(n: usize, seed: u64) -> (Vec<String>, Vec<StringRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let xi = u.sample(&mut rng);
            let zi = u.sample(&mut rng);
            let eta = (PI * xi).sin() * (PI * zi).cos();
            let lambda = eta.exp().max(1e-12);
            let draw: f64 = Poisson::new(lambda).expect("rate").sample(&mut rng);
            StringRecord::from(vec![xi.to_string(), zi.to_string(), draw.to_string()])
        })
        .collect();
    let headers = ["x", "z", "count"].into_iter().map(String::from).collect();
    (headers, rows)
}

/// The #1082 guard: an estimated-theta Negative-Binomial tensor fit must
/// converge the outer REML loop in a small, bounded number of iterations and a
/// tight wall-clock budget.
#[test]
fn negbin_te_2d_outer_loop_converges_in_budget_1082() {
    init_parallelism();
    let (headers, rows) = synthetic_count_records(200, 20260530);
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode count dataset");
    let cfg = FitConfig {
        family: Some("nb".to_string()),
        ..FitConfig::default()
    };

    let t0 = Instant::now();
    let result = fit_from_formula("count ~ te(x, z, k=5)", &ds, &cfg).expect("gam nb te fit");
    let elapsed = t0.elapsed();

    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Negative-Binomial te(x, z)");
    };

    eprintln!(
        "[#1082 guard] nb te: outer_iterations={} outer_converged={} grad_norm={:?} wall={:.2}s",
        fit.fit.outer_iterations,
        fit.fit.outer_converged,
        fit.fit.outer_gradient_norm,
        elapsed.as_secs_f64(),
    );

    assert!(
        fit.fit.outer_converged,
        "#1082 regression: NB te outer REML loop did not converge \
         (outer_iterations={}, grad_norm={:?})",
        fit.fit.outer_iterations,
        fit.fit.outer_gradient_norm,
    );

    assert!(
        fit.fit.outer_iterations <= 30,
        "#1082 regression: NB te outer loop took {} iterations (expected ~5); \
         the lambda-search theta freeze is not holding the REML surface stationary",
        fit.fit.outer_iterations,
    );

    assert!(
        elapsed.as_secs_f64() < 60.0,
        "#1082 regression: NB te fit took {:.1}s (expected a few seconds)",
        elapsed.as_secs_f64(),
    );
}

/// Control: the otherwise-identical Poisson fit (no estimated dispersion) must
/// also converge quickly. Guards against a regression that slows the shared
/// standard-family outer loop while fixing the NB-specific path.
#[test]
fn poisson_te_2d_outer_loop_converges_in_budget_1082() {
    init_parallelism();
    let (headers, rows) = synthetic_count_records(200, 20260530);
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode count dataset");
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("count ~ te(x, z, k=5)", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson te(x, z)");
    };

    eprintln!(
        "[#1082 control] poisson te: outer_iterations={} outer_converged={}",
        fit.fit.outer_iterations,
        fit.fit.outer_converged,
    );

    assert!(fit.fit.outer_converged, "Poisson te outer loop did not converge");
    assert!(
        fit.fit.outer_iterations <= 30,
        "Poisson te outer loop took {} iterations (expected ~5)",
        fit.fit.outer_iterations,
    );
}
