//! FAILING TEST — ticket: `s(x, bc=clamped)` fits cleanly but predicting at
//! new points crashes with
//!
//!     failed to build prediction design: Dimension mismatch:
//!     frozen identifiability transform mismatch: design has 22 columns
//!     but transform has 24 rows
//!
//! Repro: any non-trivial fit + predict-at-new-points cycle. The fit's frozen
//! spec stores an identifiability transform sized for the *unconstrained*
//! basis (24 cols), but the predict-time design builder applies the BC linear
//! constraints first, producing a 22-col design that doesn't match the saved
//! transform. Fix: either freeze the post-constraint transform, or apply the
//! transform before the BC reduction at predict time.
//!
//! Until fixed, this test fails at `expect(...)`. Keep it failing — it's the
//! ticket.

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

#[test]
fn bspline_bc_clamped_predict_at_new_points_succeeds() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let n = 200usize;
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bc=clamped)", &data, &cfg)
        .expect("BC clamped fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };

    // Predict at 50 new uniform points.
    let n_test = 50;
    let mut new_data = Array2::<f64>::zeros((n_test, 2));
    for i in 0..n_test {
        let xt = 0.01 + 0.98 * (i as f64) / ((n_test - 1) as f64);
        new_data[[i, 0]] = xt;
        new_data[[i, 1]] = 0.0;
    }

    // This currently panics with "frozen identifiability transform mismatch:
    // design has N columns but transform has M rows". When the fix lands,
    // predict will succeed and we'll get a finite vector of predictions.
    let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("BC clamped predict design should rebuild from frozen spec");
    let pred = design.design.apply(&fit.fit.beta);
    assert_eq!(pred.len(), n_test, "predict returned wrong length");
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "predict produced non-finite values"
    );
}
