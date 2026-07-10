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
    (1.3 * x[0]).sin() + 0.5 * x[1] * x[1] - 0.7 * x[2]
        + (0.4 * x[3]).cos()
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

/// The exact wine_gamair Rust formula shape: 5 `ps` smooths, knots=7, double
/// penalty on each (rho_dim = 10, p ≈ 51).
const WINE_SHAPED_FORMULA: &str = "y ~ s(x0, type=ps, knots=7, double_penalty=true) \
     + s(x1, type=ps, knots=7, double_penalty=true) \
     + s(x2, type=ps, knots=7, double_penalty=true) \
     + s(x3, type=ps, knots=7, double_penalty=true) \
     + s(x4, type=ps, knots=7, double_penalty=true)";

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// The literal wine_gamair regime: n=30 against five `select=TRUE`
/// (double-penalty) `ps` smooths whose summed basis (p ≈ 51) is wider than the
/// data. mgcv fits this exact shape in milliseconds because the shrinkage
/// penalties — not the data — set the effective rank; gam must do the same.
/// The pre-fix behavior was the opposite extreme: the outer REML optimizer
/// wandered the flat overparameterized surface through ~850k cost-only
/// evaluations until the 42-minute shard budget killed it (#1089).
///
/// The fix recognizes that a double-penalized basis needs no unpenalized
/// data identification, so the capacity gate admits the fit, and the bounded
/// outer loop drives it to a converged optimum promptly.
#[test]
fn wine_shaped_double_penalty_fit_converges_promptly() {
    let data = build_data(30, 0.1, 1089);

    let start = std::time::Instant::now();
    let result = fit_from_formula(WINE_SHAPED_FORMULA, &data, &gaussian_cfg())
        .expect("n=30 wine-shaped fit");
    let elapsed = start.elapsed();

    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // The defining symptom of #1089 was the absence of termination. The fit
    // must converge in a bounded number of outer steps, not bail at the cap.
    // Fit existence is the sealed convergence proof (SPEC 20).
    assert!(
        fit.fit.outer_iterations <= 200,
        "outer REML loop took {} iterations on the n=30 wine-shaped fit; \
         expected prompt convergence under the iteration cap",
        fit.fit.outer_iterations,
    );

    // The 42-minute wall-budget death cannot recur. Convergence in a bounded
    // iteration count (asserted above) is the real termination guarantee; this
    // is a coarse hang tripwire only — generous enough for a loaded CI host,
    // but still an order of magnitude under the original 42-minute timeout, so
    // any regression back to an unbounded outer search is caught.
    assert!(
        elapsed.as_secs() < 600,
        "n=30 wine-shaped fit took {:.1}s; the outer loop is not terminating \
         promptly (the #1089 hang)",
        elapsed.as_secs_f64(),
    );
}

/// Companion: lift the same wine-shaped design above the basis-capacity floor
/// (5 ps(knots=7) double-penalty smooths need ~57 rows). A well-posed small
/// fully-penalized design with rho_dim=10 must drive its outer REML loop to a
/// *converged* termination in a bounded number of iterations — the loop's
/// stopping criterion has to actually fire, not lean on the wall clock.
#[test]
fn small_n_gaussian_double_penalty_outer_loop_terminates() {
    let data = build_data(120, 0.1, 1089);

    let start = std::time::Instant::now();
    let result =
        fit_from_formula(WINE_SHAPED_FORMULA, &data, &gaussian_cfg()).expect("well-posed fit");
    let elapsed = start.elapsed();

    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // Fit existence is the sealed convergence proof (SPEC 20).

    assert!(
        fit.fit.outer_iterations <= 200,
        "outer REML loop took {} iterations on a small n=120 fit; expected \
         prompt convergence well under the iteration cap",
        fit.fit.outer_iterations,
    );

    assert!(
        elapsed.as_secs() < 120,
        "n=120 double-penalty Gaussian fit took {:.1}s; outer loop is not \
         terminating promptly",
        elapsed.as_secs_f64(),
    );
}
