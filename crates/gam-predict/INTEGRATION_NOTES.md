# gam-predict integration notes

This split did not edit the shared files owned by the integrator:
`/Cargo.toml`, `/src/lib.rs`, `/src/inference/mod.rs`, or `/Cargo.lock`.

## Shared-file wiring still needed

- Add `crates/gam-predict` to the workspace.
- Publish the engine module `src/inference/predict_io.rs` from
  `src/inference/mod.rs`.
- Repoint root public exports that still name `gam::inference::predict` or
  `gam::predict` to the downstream `gam_predict` crate.

## Deleted `gam::estimate` predict re-export block

The following symbols were removed from `src/solver/estimate/mod.rs`:

- `CoefficientUncertaintyResult`
- `InferenceCovarianceMode`
- `MeanIntervalMethod`
- `PosteriorMeanOptions`
- `PredictInput`
- `PredictPosteriorMeanResult`
- `PredictResult`
- `PredictUncertaintyOptions`
- `PredictUncertaintyResult`
- `PredictableModel`
- `coefficient_uncertainty`
- `coefficient_uncertaintywith_mode`
- `enrich_posterior_mean_bounds`
- `predict_gam`
- `predict_gam_posterior_mean`
- `predict_gam_posterior_meanwith_backend`
- `predict_gam_posterior_meanwith_fit`
- `predict_gamwith_uncertainty`

Known consumers to repoint to `gam_predict`:

- `src/main.rs`: imports `PosteriorMeanOptions`, `PredictInput`, and
  `predict_gam` from `gam::estimate`; imports `PredictableModel`,
  `predict_gam_posterior_meanwith_backend`, and `predict_gamwith_uncertainty`
  from `gam::predict`; imports `gam::inference::predict::{input,linalg}`.
- `src/main/run_predict.rs`: uses `gam::estimate::PredictUncertaintyOptions`
  and `gam::estimate::MeanIntervalMethod`.
- `src/main/model_summary.rs`: uses `gam::estimate::InferenceCovarianceMode`.
- `src/inference/generative.rs`: imports `PredictResult` from `crate::estimate`.
- `src/inference/model.rs`: constructs predictors and trait objects from
  `crate::inference::predict`.
- `src/inference/conformal.rs`: imports `PredictUncertaintyResult` and
  `interval_policy::ResponseBounds` from `crate::inference::predict`.
- `crates/gam-pyffi/src/model_ffi.rs` and
  `crates/gam-pyffi/src/geometry_ffi.rs`: use `gam::inference::predict`
  option/result types.
- Tests under `tests/` import these symbols through both `gam::estimate`,
  `gam::predict`, and `gam::inference::predict`; repoint them to
  `gam_predict`.
