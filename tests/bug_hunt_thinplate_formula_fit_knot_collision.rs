//! Failing-ticket regression: a plain `thinplate(x1, x2)` smooth cannot be
//! fit through the formula entry point on ordinary 2-D data.
//!
//! `thinplate(...)` is advertised in the README ("radial smooths in arbitrary
//! dimension (`matern`, `duchon`, `thinplate`, ...)" and the `gam fit --help`
//! term list) and the sibling families `matern(...)` / `duchon(...)` fit the
//! same data without complaint. But `thinplate` aborts every fit with
//!
//!   REML smoothing optimization failed to converge: spatial kappa
//!   optimization failed: Underlying basis function generation failed:
//!   Invalid input: radial scalar evaluation failed during scalar
//!   derivative construction
//!
//! Why it happens (files/lines read):
//!
//!  * The formula builder always stamps a finite `length_scale` on a
//!    thin-plate term — `length_scale: option_f64(options, "length_scale")
//!    .unwrap_or(1.0)` (`src/terms/term_builder.rs:1472`). A `Some(length_scale)`
//!    makes the term eligible for spatial length-scale ("kappa") optimization:
//!    `spatial_term_supports_hyper_optimization` returns true whenever
//!    `get_spatial_length_scale(..).is_some()` (`src/terms/smooth.rs:2633-2635`).
//!
//!  * Kappa optimization differentiates the design w.r.t. the log length
//!    scale via `build_scalar_design_psi_derivatives_shared`
//!    (`src/terms/basis.rs:7222`), which evaluates the radial jet triplet
//!    `(phi, q, t)` for every data row against every center through
//!    `RadialScalarKind::eval_design_triplet` (`src/terms/basis.rs:7311`).
//!
//!  * For the thin-plate kernel that evaluator returns
//!    `BasisError::DegenerateAtCollision` as soon as `r < 1e-14`
//!    (`src/terms/basis.rs:3498-3517`) because the radial derivatives
//!    `q = phi'(r)/r` and `t = (phi''(r) - q)/r²` diverge at `r = 0`.
//!
//!  * But thin-plate centers are *selected from the data*
//!    (`select_thin_plate_knots`, `src/terms/basis.rs:7640`), so every fit
//!    contains at least `k` exact data/center collisions (`r = 0`). The very
//!    first such pair sets the error flag and the whole fit is aborted at
//!    `src/terms/basis.rs:7332-7336`.
//!
//! Matérn survives because its `r = 0` radial scalars are finite; Duchon
//! survives because the Duchon family is forced onto the streaming-operator
//! path that never eagerly materializes the collision triplets
//! (`operator_only = radial_kind.is_duchon_family()`, `src/terms/basis.rs`).
//! Thin-plate is neither, so it takes the eager dense path and dies on the
//! guaranteed collision.
//!
//! The truth here is a plane `f = 1 + 0.5*x1 - 0.3*x2`, which lives in the
//! thin-plate polynomial null space, so a working thin-plate smooth recovers
//! it essentially to the noise floor. The assertion is deliberately weak (the
//! fit must merely *succeed* and produce finite, non-degenerate predictions):
//! the bug is the hard abort, and when it is fixed this test passes unchanged.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(a: f64, b: f64) -> f64 {
    1.0 + 0.5 * a - 0.3 * b
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let a = ux.sample(&mut rng);
            let b = ux.sample(&mut rng);
            let y = truth(a, b) + noise.sample(&mut rng);
            StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn thinplate_formula_fit_succeeds_on_ordinary_2d_data() {
    init_parallelism();
    let data = build_dataset(400, 0.05, 91);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // This currently returns Err with the radial-collision message described
    // in the module docs. A working thin-plate fit returns Ok.
    let result = fit_from_formula("y ~ thinplate(x1, x2)", &data, &cfg).unwrap_or_else(|e| {
        panic!("thinplate(x1, x2) failed to fit ordinary 2-D data (matern/duchon fit it fine): {e}")
    });
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit for thinplate(x1, x2)");
    };

    // Predict on an interior grid and sanity-check the fit is finite and not
    // collapsed to a constant. A capable thin-plate smooth recovers a plane
    // (which lies in its polynomial null space) to ~noise level, so a 0.30
    // RMSE budget is extremely generous.
    let g: Vec<f64> = (0..20).map(|i| 0.05 + 0.90 * i as f64 / 19.0).collect();
    let m = g.len();
    let mut design_in = Array2::<f64>::zeros((m * m, 3));
    let mut truth_vals = Vec::with_capacity(m * m);
    let mut row = 0;
    for &a in &g {
        for &b in &g {
            design_in[[row, 0]] = a;
            design_in[[row, 1]] = b;
            design_in[[row, 2]] = 0.0;
            truth_vals.push(truth(a, b));
            row += 1;
        }
    }
    let design = build_term_collection_design(design_in.view(), &fit.resolvedspec)
        .expect("rebuild thinplate design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();

    assert!(
        pred.iter().all(|v| v.is_finite()),
        "thinplate predictions must all be finite"
    );

    let span = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - pred.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        span > 0.05,
        "thinplate fit collapsed to a near-constant surface (span={span:.4}); \
         the plane truth varies by ~0.8 across the grid"
    );

    let mse: f64 = pred
        .iter()
        .zip(truth_vals.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / (m * m) as f64;
    let rmse = mse.sqrt();
    assert!(
        rmse < 0.30,
        "thinplate interior RMSE {rmse:.4} exceeds the generous 0.30 budget on a \
         plane truth that lives in its own null space"
    );
}
