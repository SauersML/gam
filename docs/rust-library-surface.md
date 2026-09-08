# Getting started with the Rust library

The `gam` crate is the supported in-process interface to the same formula
materializer, REML/LAML optimizer, and prediction implementation used by the
CLI and Python package. A successful call returns a fit whose inner and outer
optimizations have already supplied convergence evidence; non-convergence is a
`WorkflowError`, never a partially fitted model.

## Fit a Gaussian smooth and predict with uncertainty

This example builds an encoded data set, fits a penalized smooth with smoothing
selected by REML, predicts the training means, and propagates coefficient
uncertainty into response-scale standard errors. It is also compiled and run as
the `gam::getting_started` doctest, so changes to the public API cannot silently
make this page stale.

`scripts/public_api_census.py` enforces that review for explicit public
functions. It compares immutable Git trees by `(source path, function name)`
rather than inspecting linked binaries or matching names across the workspace.
Every removal, including a move, requires an exact entry in
`docs/public-api-census-changes.json` with the semantic reason and executable
replacement or retirement evidence. Its CI positive control replays the
`d484a091a` sweep and must both detect and refuse those removals. This gate is a
backstop against repeating that mechanism; it does not decide whether an API is
valuable and does not replace external-consumer behavior tests.

There is no compatibility obligation to recreate deleted convenience names.
Keep one current API per behavior, and use explicit model inputs where defaults
would silently choose geometry. An exported name with no internal caller can
still be useful; an abandoned configuration/report carrier with no callable
producer is not a substitute for an implemented public contract.

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let headers = vec!["x".to_owned(), "y".to_owned()];
let rows = (0..32)
    .map(|i| {
        let x = -1.0 + 2.0 * f64::from(i) / 31.0;
        // A reproducible nonlinear signal with a small, nonzero residual.
        let y = 0.4 + 1.2 * x + 0.7 * x * x + 0.02 * f64::from(i % 3);
        StringRecord::from(vec![x.to_string(), y.to_string()])
    })
    .collect();
let data = encode_recordswith_inferred_schema(headers, rows)?;
let config = FitConfig {
    family: Some("gaussian".to_owned()),
    ..FitConfig::default()
};

let FitResult::Standard(fitted) =
    fit_from_formula("y ~ smooth(x, k=8)", &data, &config)?
else {
    unreachable!("a Gaussian formula produces a standard fit")
};

let block = fitted.fit.blocks.first().expect("standard fit has one block");
let family = fitted
    .fit
    .likelihood_family
    .clone()
    .expect("built-in Gaussian fit records its likelihood");
let prediction = predict_gamwith_uncertainty(
    fitted.design.design.clone(),
    block.beta.view(),
    fitted.design.affine_offset.view(),
    family,
    &fitted.fit,
    &PredictUncertaintyOptions::default(),
)?;

assert_eq!(prediction.mean.len(), 32);
assert!(prediction.mean_standard_error.iter().all(|se| se.is_finite()));

// REML/LAML diagnostics belong to the converged fit itself.
let criterion = fitted
    .fit
    .reml_score()
    .expect("this noisy Gaussian data has a finite REML criterion");
let evidence = fitted.fit.convergence_evidence();
assert!(criterion.is_finite());
assert!(fitted.fit.outer_iterations <= config.outer_max_iter);
println!("REML={criterion:.3}; convergence={evidence:?}");
# Ok(())
# }
```

The design and affine offset above are the *training* design. For new data,
rebuild from `fitted.resolvedspec` so knots, factor levels, constraints, and
other fitted term state remain frozen; then pass the rebuilt matrix and offset
to the same prediction function.

## Reading the result

* `FitResult::Standard` is the ordinary one-linear-predictor GAM result. Other
  variants make multi-parameter, survival, and large-data representations
  explicit rather than hiding them behind a lossy common struct.
* `UnifiedFitResult::convergence_evidence()` is the construction proof carried
  by every fit. There is no separate boolean callers must remember to check.
* `UnifiedFitResult::reml_score()` returns the optimized REML/LAML criterion.
  It returns `None` only at the exact-fit, zero-dispersion Gaussian boundary,
  where no finite restricted likelihood exists; callers must not substitute a
  sentinel value when comparing models.
* `PredictUncertaintyResult::mean` is the posterior-mean response prediction,
  while `mean_standard_error` contains coefficient-uncertainty standard errors.
  Prediction errors report incompatible matrix dimensions, missing covariance,
  or invalid interval options through `EstimationError`.

## Errors and preconditions

`fit_from_formula` rejects malformed formulas, missing or incompatible columns,
invalid family/link configuration, and any optimizer run that does not converge.
Input rows must be nonempty and numeric values must be finite. Prediction
requires one design column per fitted coefficient and one offset per prediction
row. Keep `FitConfig::default()` unless changing model semantics deliberately:
smoothing selection remains REML/LAML and prediction remains posterior-mean by
default.

## Public API removal policy

```rust
use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    predict::{PredictUncertaintyOptions, predict_gamwith_uncertainty},
};
