//! #1082 regression guard for the STANDARD exponential-family REML outer-loop
//! cycling on a Negative-Binomial 2-D tensor-product smooth.
//!
//! Root cause (verified): with the NB overdispersion `theta` ESTIMATED, the
//! inner solver re-derived `theta` from each outer iterate's warm-start `eta`,
//! so the NB working response / deviance / penalty-logdet — and hence the REML
//! criterion — drifted on every outer evaluation. The outer optimizer then
//! chased a moving target: the projected-gradient convergence test never tripped
//! and the loop ground to `max_iter` (200) without converging, the #1082
//! `gam_tensor_te_2d_negbin` wall-clock timeout. (An otherwise-identical Poisson
//! fit, with no estimated dispersion, converges in ~5 outer iterations.)
//!
//! Fix: freeze `theta` for the duration of the smoothing-parameter (λ) search
//! (`GlmLikelihoodSpec::with_negbin_theta_frozen_for_search`, driven by the REML
//! state's `frozen_negbin_theta`), so `F(ρ) = REML(ρ, θ_frozen)` is a stationary
//! function of ρ and the loop converges in a handful of iterations. `theta` is
//! still ML-refreshed at the single final, reported accept-fit.
//!
//! These reproductions are pure Rust on small synthetic grids (no R / baseline
//! data), so they are FAST and run inside the normal test budget.

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

/// The #1082 guard: an estimated-θ Negative-Binomial tensor fit must converge
/// the outer REML loop in a small, bounded number of iterations and a tight
/// wall-clock budget. Before the λ-search θ-freeze fix this ran the full
/// `max_iter = 200` outer iterations with `outer_converged == false`; after the
/// fix it converges in ~5.
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

    // The outer smoothing-parameter loop must actually converge — not silently
    // exhaust its budget. This is the precise failure the bug exhibited.
    assert!(
        fit.fit.outer_converged,
        "#1082 regression: NB te outer REML loop did not converge \
         (outer_iterations={}, grad_norm={:?})",
        fit.fit.outer_iterations, fit.fit.outer_gradient_norm,
    );

    // A stationary REML surface converges in a handful of outer iterations. The
    // bug ran the full max_iter = 200; a stable θ converges in ~5. Bound well
    // below max_iter so a future re-introduction of the per-eval θ drift fails
    // here loudly rather than by wall-clock timeout. This is NOT a budget bump:
    // it is an upper guard on a loop that genuinely converges in ~5.
    assert!(
        fit.fit.outer_iterations <= 30,
        "#1082 regression: NB te outer loop took {} iterations (expected ~5); \
         the λ-search θ-freeze is not holding the REML surface stationary",
        fit.fit.outer_iterations,
    );

    // Wall-clock guard. The fit converges in well under 60 s even in a debug
    // build on the small n=200 grid; the bug spent the whole 200-iteration
    // budget. Generous enough to never flake on a slow CI box, tight enough to
    // catch a return of the cycling.
    assert!(
        elapsed.as_secs_f64() < 60.0,
        "#1082 regression: NB te fit took {:.1}s (expected a few seconds)",
        elapsed.as_secs_f64(),
    );
}

/// Control: the otherwise-identical Poisson fit (no estimated dispersion) must
/// also converge quickly. Guards against a regression that slows the shared
/// standard-family outer loop while "fixing" the NB-specific path.
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
        fit.fit.outer_iterations, fit.fit.outer_converged,
    );

    assert!(fit.fit.outer_converged, "Poisson te outer loop did not converge");
    assert!(
        fit.fit.outer_iterations <= 30,
        "Poisson te outer loop took {} iterations (expected ~5)",
        fit.fit.outer_iterations,
    );
}
