//! Regression for GitHub issue #360: a box-constrained parametric linear term
//! — `linear(x, min=…, max=…)` or its sugar `nonnegative(x)` / `nonpositive(x)`
//! — must FIT, not panic, when the constraint actually BINDS (the unconstrained
//! least-squares coefficient lies outside the box).
//!
//! The original failure was a hard panic inside the REML inner-solution builder:
//!
//! ```text
//! InnerSolutionBuilder: penalty coordinate 0 has dimension 2 but beta length is 1
//! ```
//!
//! When the linear-inequality active set is non-empty, the inner solve and the
//! penalized Hessian are reduced to the free subspace `β = z β_f` of dimension
//! `p − active_set_size`. The Hessian operator and the penalty log-determinant
//! root were projected onto that subspace, but the penalty *coordinates* fed to
//! `InnerSolutionBuilder::build` kept their full pre-reduction width, so the
//! `coord.dim() == beta.len()` invariant fired. The fix projects each penalty
//! coordinate onto the same free basis (`R_k → R_k z`) so the dimensions move
//! in lockstep.
//!
//! Binding is the entire point of the feature, so the only case that does
//! anything was exactly the case that crashed. This test pins the binding case
//! to a fit that succeeds and respects the bound.

use gam::estimate::BlockRole;
use gam::inference::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::{FitConfig, FitResult, fit_model, materialize};
use ndarray::Array2;

/// Noiseless dataset whose unconstrained slope is strictly negative.
///
/// `y = 1 − 3x` on a symmetric grid, so the ordinary least-squares slope is
/// `−3`. A `min = 0` lower bound on the slope therefore *binds*: the feasible
/// optimum pins the slope to `0`, leaving a flat fit at the response mean.
fn negative_slope_dataset(n: usize) -> EncodedDataset {
    let mut values = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = (i as f64) / (n as f64 - 1.0) * 4.0 - 2.0; // grid on [-2, 2]
        values[[i, 0]] = 1.0 - 3.0 * x; // y
        values[[i, 1]] = x; // x
    }
    EncodedDataset {
        headers: vec!["y".into(), "x".into()],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
    }
}

/// Fit `formula` on the binding-bound dataset and assert (a) the fit succeeds
/// without panicking, and (b) the in-sample fitted values are non-decreasing in
/// `x` — i.e. the slope was clamped to `≥ 0` rather than taking the data's
/// natural `−3`.
fn assert_lower_bound_binds_without_panic(formula: &str) {
    let data = negative_slope_dataset(200);
    let config = FitConfig::default();

    let materialized = materialize(formula, &data, &config)
        .unwrap_or_else(|e| panic!("materialize('{formula}') should succeed; got {e:?}"));
    let result = fit_model(materialized.request)
        .unwrap_or_else(|e| panic!("fit_model for '{formula}' should not error; got {e:?}"));

    let fitted = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit result for '{formula}'"),
    };

    // Reconstruct the in-sample linear predictor η = Xβ (Gaussian identity link,
    // so fitted values equal η). The standard GAM has a single Mean block whose
    // β is the full coefficient vector aligned with the design columns.
    let mean_block = fitted
        .fit
        .blocks
        .iter()
        .find(|b| b.role == BlockRole::Mean)
        .expect("standard fit must have a Mean block");
    let beta = &mean_block.beta;
    let x_dense = fitted.design.design.to_dense();
    assert_eq!(
        x_dense.ncols(),
        beta.len(),
        "design column count must match coefficient length for '{formula}'"
    );

    let fitted_values = x_dense.dot(beta);
    assert!(
        fitted_values.iter().all(|v| v.is_finite()),
        "fitted values must be finite for '{formula}'"
    );

    // Non-decreasing in x: the dataset's x column is the second design input and
    // is sorted ascending, so consecutive fitted values must not decrease beyond
    // a tiny numerical slack. A clamped slope of 0 yields a flat line; any
    // residual positive curvature from the ridge penalty is still non-decreasing.
    const SLACK: f64 = 1e-6;
    for i in 1..fitted_values.len() {
        assert!(
            fitted_values[i] >= fitted_values[i - 1] - SLACK,
            "fitted values must be non-decreasing in x for '{formula}' \
             (slope clamped to ≥ 0): fitted[{i}]={} < fitted[{}]={}",
            fitted_values[i],
            i - 1,
            fitted_values[i - 1]
        );
    }

    // The clamp should kill essentially all of the −3 slope: the spread of the
    // fitted values is tiny compared with the unconstrained response spread
    // (|Δy| ≈ 3 · 4 = 12 across the grid).
    let fitted_spread = fitted_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - fitted_values.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        fitted_spread < 1.0,
        "binding min=0 bound should flatten the fit (spread ≪ 12); got spread {fitted_spread} for '{formula}'"
    );
}

#[test]
fn nonnegative_sugar_binds_lower_bound_without_inner_solution_panic() {
    assert_lower_bound_binds_without_panic("y ~ nonnegative(x)");
}

#[test]
fn linear_min_zero_binds_lower_bound_without_inner_solution_panic() {
    assert_lower_bound_binds_without_panic("y ~ linear(x, min=0)");
}
