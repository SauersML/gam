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

```rust
use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    predict::{PredictUncertaintyOptions, predict_gamwith_uncertainty},
};

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

---

## Public API surface decision record

### Rust library surface and removal decisions

The Rust library is a product, alongside the CLI and Python extension. A public
item reachable through an exported Rust module, re-export, public type, or trait
implementation is a source-graph root in every supported configuration. Its
absence from a CLI or extension symbol table does not make it unused. Generic
instantiation, inlining, and LTO do not change that rule.

The maintained high-level API is `gam::fit_from_formula` for formula/data input,
`gam::materialize` followed by `gam::fit_model` for typed requests, and
`gam::predict` for prediction. These route to the same production model owners
used by the CLI and Python wrapper. Domain crates also expose their documented
numerical and model-building contracts directly; consumers do not have to link
either executable to use them.

Rust's `dead_code` analysis operates on source definitions and visibility,
including generic bodies and re-exports. The workspace's deny-warnings production
builds retain exported library items and reject unreferenced private production
items. Test targets must also compile: test helpers stay under `#[cfg(test)]`
and are evaluated within their test source graph. A private production helper
used only by tests is a candidate to move into test scope, not a reason to delete
the tests. Compiler acceptance establishes source validity, not whether a public
contract is valuable; removing a public contract still requires a semantic
decision and review of its callers and documentation.

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

#### Decisions for the concrete #2829 cases

| Historical item | Current contract and decision |
| --- | --- |
| `fit_gam` | Maintain `gam::fit_model` / `gam::fit_from_formula` as the unified model front doors and `gam_solve::estimate::fit_gam_with_penalty_specs` as the explicit external-design API. The generic owned/borrowed design contract is exercised by `public_library_surface_2829.rs`. Recreating an alias that only supplies defaults is unnecessary. |
| `canonicalize_for_identifiability` | Maintain `gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars`. `None` explicitly requests the zero-state audit; supplied scalars audit the actual family operating point. `canonical_recovery.rs` exercises this public route. |
| `under_identified_subspace` | Maintain `gam_solve::estimate::reml::jeffreys_subspace::under_identified_subspace_in_metric`. The caller supplies the model's metric; an identity-metric convenience must not hide a coordinate convention. The public integration target tests both identity coordinates and congruent nonorthogonal coordinates. |
| `arrow_log_det_from_cache` | The public `ArrowSchurCache::arrow_log_det` reads the recorded undamped criterion value. `compute_undamped_arrow_log_det` and `undamped_arrow_log_det_with_schur` own its construction. Do not duplicate this with an evidence-module accessor. |
| `matrix_free_arrow_evidence_log_det` | Maintain `matrix_free_arrow_evidence_log_det_surrogate`, with an explicit rational-lane state for a differentiable frozen criterion and `None` for value-only SLQ. Internal documentation now names this actual public entry point. |
| `coordinate_block_log_det`, `criterion_as_atoms` | The current exact-A criterion assembles `log_det` and `log_det_tt` from the same observed information and uses `rank_adjusted_quasi_laplace_complexity`. The removed majorizer convenience is not the ranked coordinate term. Independent adjoint/criterion acceptance is tracked by #2333 and #2828. |
| `Dual2::{seed_directional,from_channels,seed_inner,seed_outer}` | Maintain the public `Dual2` value/first/second fields, `constant`, `variable`, and `Dual22::channels`. The analytic polynomial channel-order test in `gam-math` verifies the actual representation; redundant setters and getter/setter roundtrips are not separate product behavior. |
| `enable_outer_gradient_fd_capture` | Maintain `enable_outer_gradient_fd_capture_over_theta`; it arms the complete current diagnostic and owns its typed sink. Consumers now document that actual function. |
| `triangle_cocycle_defect`, `triangle_sign_product` | Consumers compose the public `ChartTransition` rotations/signs. The restored `local_chart_recovery_tests.rs` independently checks the composition on plane, sphere-band, and Swiss-roll fixtures; no duplicate arithmetic wrapper is required. |
| `LocalAtlas::co_collapse_candidates` | **Unresolved.** The three historical #2280 co-collapse contracts have no verified replacement. Public transition composition does not establish duplicate-chart detection. #2280 must resolve this behavior explicitly. |
| `cofit_block_and_curved`, `cofit_linear_via_arrow`, `cofit_composed_via_arrow` | **Unresolved.** Their modules currently retain configuration/report types but no callable producer. `tiered::fit_tiered` is the active fitting route; its fixed-point and stationarity contracts must be compared to the five historical #2023 pins before the old bridge is retired. |

This is a decision record for the issue's concrete examples, not a claim that all
1,206 historical public declarations have been audited. Remaining identities must
be resolved by crate, module/type, and signature; a same-named definition in a
different owner is not evidence of recovery. #2829 remains open until the
remaining public behavior decisions and their acceptance evidence are complete.

The external acceptance targets executed on MSI: two public-library tests, four
structural-coordinate tests, and four measured-span tests passed. The generic
fit test instantiates both owned and borrowed designs and verifies the exact
normal equations including the solver's declared stabilization ridge, with
bit-identical coefficients across ownership forms. The measured-span shear
control demonstrates that supplying the wrong metric changes the selected
dimension. These are compiled external consumers, not symbol-table probes.
Execution receipts and remaining historical acceptance gaps are recorded in
[`test-census-2818-recovery.md`](test-census-2818-recovery.md).
