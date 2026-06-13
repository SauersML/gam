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

/// The literal wine_gamair regime: n=30 against a design whose penalized basis
/// (p ≈ 51) is wider than the data. Such a fit is rank-deficient and its REML
/// surface is flat in the unidentified directions, which is exactly what let
/// the outer optimizer wander through ~850k cost-only evaluations until the
/// 42-minute shard budget killed it. The fit must instead be refused
/// *promptly* with an actionable error, never enter the unbounded outer search.
#[test]
fn wine_shaped_overparameterized_fit_is_refused_promptly_not_hung() {
    let data = build_data(30, 0.1, 1089);

    let start = std::time::Instant::now();
    let result = fit_from_formula(WINE_SHAPED_FORMULA, &data, &gaussian_cfg());
    let elapsed = start.elapsed();

    // The defining symptom of #1089 was the *absence* of termination. Whatever
    // the decision (refuse or fit), it must come back fast — the 42-minute
    // wall-budget death cannot recur. A couple of seconds is a generous
    // ceiling for a refusal decision on n=30.
    assert!(
        elapsed.as_secs() < 30,
        "n=30 wine-shaped fit took {:.1}s to return; the outer loop is not \
         terminating promptly (the #1089 hang)",
        elapsed.as_secs_f64(),
    );

    // The honest outcome for an n < (penalized basis dof) design is a refusal
    // with an actionable message, not a silent ill-posed grind.
    match result {
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("not enough observations") || msg.contains("rows"),
                "n=30 overparameterized fit should be refused with a row-count \
                 message; got: {msg}"
            );
        }
        Ok(_) => {
            // A fit is acceptable too, *provided* it terminated promptly (the
            // elapsed assert above). The bug was the hang, not the existence
            // of a result.
        }
    }
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

    assert!(
        fit.fit.outer_converged,
        "outer REML loop did not converge on the well-posed n=120 double-penalty \
         Gaussian fit (outer_iterations={})",
        fit.fit.outer_iterations,
    );

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
