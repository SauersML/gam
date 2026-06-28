use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::PredictInput;
use gam::basis::{BasisOptions, Dense, KnotSource, create_basis};
use gam::estimate::BlockRole;
use gam::families::bms::LatentMeasureKind;
use gam::families::scale_design::{build_scale_deviation_operator, scale_transform_from_payload};
use gam::families::survival::predict::SurvivalPredictError;
use gam::families::survival::predict::{
    fit_result_from_saved_model_for_prediction, resolve_termspec_for_prediction,
};
use gam::families::transformation_normal::TRANSFORMATION_MONOTONICITY_EPS;
use gam::probability::standard_normal_quantile;
use gam::inference::model::{
    FittedModel, FittedModelError, PredictModelClass, append_deployment_extension_columns,
};
use gam::linalg::utils::inf_norm;
use gam::matrix::DesignMatrix;
use gam::terms::smooth::build_term_collection_design;
use gam::term_builder::resolve_role_col;

/// Typed errors emitted while assembling a [`PredictInput`] from a saved model.
///
/// Each variant carries a pre-formatted `reason` string so `Display` is
/// byte-equivalent to the original `format!(...)` outputs the module used
/// before the typed-error migration. The category split lets callers
/// pattern-match on the failure kind without dragging the string apart.
#[derive(Debug, Clone)]
pub enum PredictInputError {
    /// Request-level input did not satisfy the predict contract: bad offset
    /// lengths, non-finite covariates, unsupported predict options for the
    /// saved model class, or unparseable model metadata at the boundary.
    InvalidInput { reason: String },
    /// Rebuilt prediction designs disagree with saved coefficient blocks or
    /// transform matrices (model/design column counts, basis shapes,
    /// reshape failures).
    DimensionMismatch { reason: String },
    /// The saved model is missing payload metadata required to drive the
    /// prediction (response knots, transform, degree, calibration block,
    /// unified fit, z column, etc.).
    MissingMetadata { reason: String },
    /// Survival-specific prediction assembly failed below this layer; the
    /// source error keeps its own semantic variant instead of being flattened
    /// into a generic predict-input bucket.
    SurvivalPrediction {
        context: &'static str,
        source: SurvivalPredictError,
    },
    /// Saved-model payload validation failed below this layer; the source
    /// error keeps its model-layer category and payload context.
    ModelPayload {
        context: &'static str,
        source: FittedModelError,
    },
}

impl std::fmt::Display for PredictInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictInputError::InvalidInput { reason }
            | PredictInputError::DimensionMismatch { reason }
            | PredictInputError::MissingMetadata { reason } => f.write_str(reason),
            PredictInputError::SurvivalPrediction { context, source } => {
                write!(f, "{context}: {source}")
            }
            PredictInputError::ModelPayload { context, source } => {
                write!(f, "{context}: {source}")
            }
        }
    }
}

impl std::error::Error for PredictInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PredictInputError::SurvivalPrediction { source, .. } => Some(source),
            PredictInputError::ModelPayload { source, .. } => Some(source),
            PredictInputError::InvalidInput { .. }
            | PredictInputError::DimensionMismatch { .. }
            | PredictInputError::MissingMetadata { .. } => None,
        }
    }
}

impl From<PredictInputError> for String {
    fn from(err: PredictInputError) -> String {
        err.to_string()
    }
}

impl From<String> for PredictInputError {
    /// Inbound conversion from the many `Result<_, String>` helpers this
    /// module still calls into (basis builders, term-collection assembly,
    /// fit deserializers). The text is preserved verbatim; we only pick a
    /// category so external messages flow through `?` without per-callsite
    /// `.map_err`.
    fn from(reason: String) -> PredictInputError {
        PredictInputError::InvalidInput { reason }
    }
}

impl From<gam_data::DataError> for PredictInputError {
    /// Inbound conversion from the typed data-layer error channel
    /// (`resolve_col` / `resolve_role_col` returning
    /// `DataError::ColumnNotFound` for formula-referenced columns missing
    /// from the prediction input). Preserves the human text byte-identical
    /// to the legacy `Display` output; the typed structural payload is
    /// flattened here because predict input has its own request-vs-model
    /// classification, but the FFI boundary path that needs the structured
    /// payload (issue #305) routes through `WorkflowError::ColumnNotFound`,
    /// not through this conversion.
    fn from(err: gam_data::DataError) -> PredictInputError {
        PredictInputError::InvalidInput {
            reason: err.to_string(),
        }
    }
}

impl From<SurvivalPredictError> for PredictInputError {
    /// Survival-prediction helpers (`resolve_termspec_for_prediction`,
    /// `fit_result_from_saved_model_for_prediction`) emit their own typed
    /// errors; keep that typed source so `?` preserves the layer that failed.
    fn from(err: SurvivalPredictError) -> PredictInputError {
        PredictInputError::SurvivalPrediction {
            context: "predict-input survival assembly",
            source: err,
        }
    }
}

impl From<FittedModelError> for PredictInputError {
    /// `FittedModel` payload helpers (deployment extension assembly,
    /// calibration validation) surface model-layer errors that remain
    /// chained here instead of being recategorized as request input.
    fn from(err: FittedModelError) -> PredictInputError {
        PredictInputError::ModelPayload {
            context: "predict-input model payload",
            source: err,
        }
    }
}

fn build_marginal_slope_local_auxiliary_matrix(
    model: &FittedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<Option<Array2<f64>>, PredictInputError> {
    let Some(LatentMeasureKind::LocalEmpirical {
        feature_cols,
        input_scales,
        ..
    }) = model.latent_measure.as_ref()
    else {
        return Ok(None);
    };
    let n = data.nrows();
    let d = feature_cols.len();
    let mut out = Array2::<f64>::zeros((n, d));
    let training_headers = model.training_headers.as_ref();
    for (local_col, &fit_col) in feature_cols.iter().enumerate() {
        let prediction_col = training_headers
            .and_then(|headers| headers.get(fit_col))
            .and_then(|name| col_map.get(name))
            .copied()
            .unwrap_or(fit_col);
        if prediction_col >= data.ncols() {
            return Err(PredictInputError::DimensionMismatch {
                reason: format!(
                    "local empirical marginal-slope prediction feature column {fit_col} is out of bounds for {} columns",
                    data.ncols()
                ),
            });
        }
        out.column_mut(local_col)
            .assign(&data.column(prediction_col));
    }
    if let Some(scales) = input_scales.as_ref() {
        if scales.len() != d {
            return Err(PredictInputError::DimensionMismatch {
                reason: format!(
                    "local empirical marginal-slope prediction input scale dimension mismatch: scales={}, features={d}",
                    scales.len()
                ),
            });
        }
        for (col, &scale) in scales.iter().enumerate() {
            if !(scale.is_finite() && scale > 0.0) {
                return Err(PredictInputError::InvalidInput {
                    reason: format!(
                        "local empirical marginal-slope prediction input scale {col} must be finite and positive, got {scale}"
                    ),
                });
            }
            out.column_mut(col).mapv_inplace(|value| value / scale);
        }
    }
    if out.iter().any(|value| !value.is_finite()) {
        return Err(PredictInputError::InvalidInput {
            reason: "local empirical marginal-slope prediction conditioning values must be finite"
                .to_string(),
        });
    }
    Ok(Some(out))
}

fn build_predict_input_for_model_inner(
    model: &FittedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
    offset_noise: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<PredictInput, PredictInputError> {
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let design_input = clipped.as_ref().map_or(data, |arr| arr.view());
    let design = build_term_collection_design(design_input, &spec).map_err(|e| {
        PredictInputError::InvalidInput {
            reason: format!("failed to build prediction design: {e}"),
        }
    })?;
    let n = data.nrows();
    if offset.len() != n || offset_noise.len() != n {
        return Err(PredictInputError::DimensionMismatch {
            reason: format!(
                "prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
                offset.len(),
                offset_noise.len()
            ),
        });
    }

    match model.predict_model_class() {
        PredictModelClass::Standard => {
            if noise_offset_supplied {
                return Err(PredictInputError::InvalidInput {
                    reason: "--noise-offset-column is not supported for standard prediction"
                        .to_string(),
                });
            }
            let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
            let beta = if model.has_link_wiggle() {
                fit_saved
                    .block_by_role(BlockRole::Mean)
                    .ok_or_else(|| PredictInputError::MissingMetadata {
                        reason: "standard link-wiggle model is missing Mean coefficient block"
                            .to_string(),
                    })?
                    .beta
                    .clone()
            } else {
                fit_saved.beta.clone()
            };
            let mean_design = if model.deployment_extensions.is_empty() {
                design.design.clone()
            } else {
                DesignMatrix::from(append_deployment_extension_columns(
                    model.payload(),
                    design_input,
                    col_map,
                    training_headers,
                    design.design.to_dense(),
                )?)
            };
            if beta.len() != mean_design.ncols() {
                return Err(PredictInputError::DimensionMismatch {
                    reason: format!(
                        "model/design mismatch: model beta has {} coefficients but new-data design has {} columns",
                        beta.len(),
                        mean_design.ncols()
                    ),
                });
            }
            Ok(PredictInput {
                design: mean_design,
                offset: offset.clone(),
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
                auxiliary_matrix: None,
            })
        }
        PredictModelClass::GaussianLocationScale
        | PredictModelClass::BinomialLocationScale
        | PredictModelClass::DispersionLocationScale => {
            // Dispersion location-scale (#913) persists no scale-deviation
            // `noise_transform`, so `scale_transform_from_payload` returns
            // `None` and the prepared noise design falls through to the raw
            // log-precision design — exactly what the predictor's precision
            // channel consumes.
            let spec_noise = resolve_termspec_for_prediction(
                &model.resolved_termspec_noise,
                training_headers,
                col_map,
                "resolved_termspec_noise",
            )?;
            let design_noise_raw = build_term_collection_design(design_input, &spec_noise)
                .map_err(|e| PredictInputError::InvalidInput {
                    reason: format!("failed to build noise prediction design: {e}"),
                })?;

            let noise_transform = scale_transform_from_payload(
                &model.noise_projection,
                &model.noise_center,
                &model.noise_scale,
                model.noise_non_intercept_start,
                model.noise_projection_ridge_alpha,
            )?;
            let prepared_noise_design = if let Some(transform) = noise_transform.as_ref() {
                build_scale_deviation_operator(
                    design.design.clone(),
                    design_noise_raw.design.clone(),
                    transform,
                )?
            } else {
                design_noise_raw.design.clone()
            };

            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: Some(prepared_noise_design),
                offset_noise: Some(offset_noise.clone()),
                auxiliary_scalar: None,
                auxiliary_matrix: None,
            })
        }
        PredictModelClass::BernoulliMarginalSlope => {
            let z_name =
                model
                    .z_column
                    .as_ref()
                    .ok_or_else(|| PredictInputError::MissingMetadata {
                        reason: "marginal-slope model is missing z_column".to_string(),
                    })?;
            let z_col = resolve_role_col(col_map, z_name, "z")?;
            let z = data.column(z_col).to_owned();
            let spec_logslope = resolve_termspec_for_prediction(
                &model.resolved_termspec_logslope.as_ref().cloned(),
                training_headers,
                col_map,
                "resolved_termspec_logslope",
            )?;
            let design_logslope = build_term_collection_design(design_input, &spec_logslope)
                .map_err(|e| PredictInputError::InvalidInput {
                    reason: format!("failed to build logslope prediction design: {e}"),
                })?;
            let auxiliary_matrix =
                build_marginal_slope_local_auxiliary_matrix(model, design_input, col_map)?;
            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: Some(design_logslope.design.clone()),
                offset_noise: Some(offset_noise.clone()),
                auxiliary_scalar: Some(z),
                auxiliary_matrix,
            })
        }
        PredictModelClass::Survival => Err(PredictInputError::InvalidInput {
            reason: "build_predict_input_for_model should not be called for survival models"
                .to_string(),
        }),
        PredictModelClass::TransformationNormal => {
            if noise_offset_supplied {
                return Err(PredictInputError::InvalidInput {
                    reason:
                        "--noise-offset-column is not supported for transformation-normal prediction"
                            .to_string(),
                });
            }
            let payload = model.payload();
            let response_knots =
                payload
                    .transformation_response_knots
                    .as_ref()
                    .ok_or_else(|| PredictInputError::MissingMetadata {
                        reason: "saved transformation-normal model missing response_knots"
                            .to_string(),
                    })?;
            let response_transform_vecs = payload
                .transformation_response_transform
                .as_ref()
                .ok_or_else(|| PredictInputError::MissingMetadata {
                    reason: "saved transformation-normal model missing response_transform"
                        .to_string(),
                })?;
            let response_degree = payload.transformation_response_degree.ok_or_else(|| {
                PredictInputError::MissingMetadata {
                    reason: "saved transformation-normal model missing response_degree".to_string(),
                }
            })?;
            let response_median = payload.transformation_response_median.ok_or_else(|| {
                PredictInputError::MissingMetadata {
                    reason: "saved transformation-normal model missing response_median".to_string(),
                }
            })?;

            let t_rows = response_transform_vecs.len();
            let t_cols = if t_rows > 0 {
                response_transform_vecs[0].len()
            } else {
                0
            };
            let mut resp_transform = ndarray::Array2::<f64>::zeros((t_rows, t_cols));
            for (i, row) in response_transform_vecs.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    resp_transform[[i, j]] = v;
                }
            }
            let resp_knots = ndarray::Array1::from_vec(response_knots.clone());

            let p_shape = resp_transform.ncols();
            let p_resp = 1 + p_shape;

            let fit_saved = model
                .unified()
                .ok_or_else(|| PredictInputError::MissingMetadata {
                    reason: "saved transformation-normal model missing unified fit".to_string(),
                })?;
            let beta = &fit_saved.blocks[0].beta;
            let p_cov = design.design.ncols();
            if beta.len() != p_resp * p_cov {
                return Err(PredictInputError::DimensionMismatch {
                    reason: format!(
                        "beta length {} != p_resp({}) * p_cov({})",
                        beta.len(),
                        p_resp,
                        p_cov
                    ),
                });
            }
            let beta_mat = beta
                .view()
                .into_shape_with_order((p_resp, p_cov))
                .map_err(|e| PredictInputError::DimensionMismatch {
                    reason: format!("beta reshape failed: {e}"),
                })?;
            let cov_mat =
                design
                    .design
                    .try_row_chunk(0..n)
                    .map_err(|e| PredictInputError::InvalidInput {
                        reason: e.to_string(),
                    })?;
            let calibration = payload
                .transformation_score_calibration
                .as_ref()
                .ok_or_else(|| PredictInputError::MissingMetadata {
                    reason: "saved transformation-normal model missing score calibration"
                        .to_string(),
                })?;
            calibration.validate("saved transformation-normal score calibration")?;

            if resp_knots.is_empty() {
                return Err(PredictInputError::MissingMetadata {
                    reason: "saved transformation-normal response knots are empty".to_string(),
                });
            }

            // ── Response-scale conditional mean E[Y|x] (issue #1612) ──────────
            //
            // The CTM latent model is `h(Y|x) ~ N(0, 1)` with `h(·|x)` strictly
            // increasing in `y`:
            //   `h(y|x) = γ₀(x) + Σ_{r≥1} I_r(y)·γ_r(x)² + offset + ε·(y − median)`
            // where `γ_r(x) = β_r · cov_row(x)` and `I_r` is the frozen I-spline
            // value basis. Because the latent variate is standard normal,
            //   `E[Y|x] = E_{Z~N(0,1)}[ h⁻¹(Z | x) ]`.
            // This is a function of the covariates alone — it does NOT depend on
            // any supplied response. The earlier predict path instead returned
            // the PIT `h(y|x)` of the supplied outcome as both linear predictor
            // and mean, which (a) made the response column mandatory and (b) made
            // the "mean" sweep with `y` at fixed `x`. We compute the genuine
            // response-scale mean here by numerically inverting the monotone
            // transform on a standard-normal quadrature and averaging.
            //
            // Quadrature: write the expectation as `E[Y|x] = ∫₀¹ Q(p|x) dp` with
            // `Q(p|x) = h⁻¹(Φ⁻¹(p) | x)` the conditional quantile function, and
            // apply the midpoint rule on `m` evenly spaced probability levels
            // `p_k = (k + ½)/m`, `z_k = Φ⁻¹(p_k)`. Working in probability space
            // keeps every evaluation point inside the finite I-spline support
            // (no normal-tail truncation) and needs no Gauss–Hermite weights.
            let monotonicity_eps = TRANSFORMATION_MONOTONICITY_EPS;
            let y_lo = resp_knots[0];
            let y_hi = resp_knots[resp_knots.len() - 1];
            if !(y_hi > y_lo) {
                return Err(PredictInputError::InvalidInput {
                    reason: format!(
                        "transformation-normal response support is degenerate: lo={y_lo}, hi={y_hi}"
                    ),
                });
            }

            // A shared fine `y`-grid spanning the response support; the I-spline
            // value basis is evaluated once here and reused for every row, so the
            // per-row inversion is a cheap monotone lookup rather than a fresh
            // basis build. `h(y|x)` is monotone in `y`, so `h` on the grid is
            // monotone too and a bracketing scan + linear interpolation inverts it.
            const GRID: usize = 257;
            let grid_y: ndarray::Array1<f64> =
                ndarray::Array1::from_shape_fn(GRID, |k| {
                    y_lo + (y_hi - y_lo) * (k as f64) / ((GRID - 1) as f64)
                });
            let (grid_val_basis, _) = create_basis::<Dense>(
                grid_y.view(),
                KnotSource::Provided(resp_knots.view()),
                response_degree,
                BasisOptions::i_spline(),
            )
            .map_err(|e| PredictInputError::InvalidInput {
                reason: e.to_string(),
            })?;
            let grid_raw_val = grid_val_basis.as_ref().clone();
            if grid_raw_val.ncols() != resp_transform.nrows() {
                return Err(PredictInputError::DimensionMismatch {
                    reason: format!(
                        "saved transformation-normal response transform shape mismatch: raw I-spline cols={} transform rows={}",
                        grid_raw_val.ncols(),
                        resp_transform.nrows()
                    ),
                });
            }
            // `grid_shape[k, r] = I_{r+1}(grid_y[k])` (shape part only, column 0
            // is the constant `1`). Linear `ε·(y − median)` floor is added per row.
            let grid_shape = grid_raw_val.dot(&resp_transform);

            // Standard-normal quadrature in probability space.
            const QUAD: usize = 48;
            let z_nodes: Vec<f64> = (0..QUAD)
                .map(|k| {
                    let p = ((k as f64) + 0.5) / (QUAD as f64);
                    standard_normal_quantile(p)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PredictInputError::InvalidInput { reason: e })?;

            let beta_mat_ref = &beta_mat;
            let cov_mat_ref = &cov_mat;
            let grid_shape_ref = &grid_shape;
            let grid_y_ref = &grid_y;
            let z_nodes_ref = &z_nodes;
            let mean_results: Vec<Result<f64, String>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let cov_row = cov_mat_ref.row(i);
                    let gamma0 = beta_mat_ref.row(0).dot(&cov_row);
                    // Squared shape factors γ_r(x)² (non-negative, r ≥ 1).
                    let mut gamma_sq = vec![0.0_f64; p_shape];
                    for r in 1..p_resp {
                        let g = beta_mat_ref.row(r).dot(&cov_row);
                        gamma_sq[r - 1] = g * g;
                    }
                    // `h(y_k | x_i)` on the shared grid: monotone increasing in k.
                    let mut h_grid = vec![0.0_f64; GRID];
                    for k in 0..GRID {
                        let mut val = gamma0;
                        let shape_row = grid_shape_ref.row(k);
                        for r in 0..p_shape {
                            val += shape_row[r] * gamma_sq[r];
                        }
                        h_grid[k] =
                            val + offset[i] + monotonicity_eps * (grid_y_ref[k] - response_median);
                        if !h_grid[k].is_finite() {
                            let max_abs_cov = inf_norm(cov_row.iter().copied());
                            return Err(format!(
                                "prediction failed: transformation-normal transform at row {i}, grid node {k} is not finite: h={:.6e}; max_abs_covariate_basis={max_abs_cov:.6e}",
                                h_grid[k]
                            ));
                        }
                    }
                    // Structural monotonicity guard: under SCOP `h' ≥ ε > 0`, so a
                    // non-increasing grid signals floating-point cancellation.
                    for k in 1..GRID {
                        if h_grid[k] <= h_grid[k - 1] {
                            return Err(format!(
                                "prediction failed: transformation-normal transform is not strictly increasing at row {i} between grid nodes {} and {k} (h={:.6e} -> {:.6e}); under SCOP h' = ε + Σ M_r γ_r² is structurally positive, so this indicates floating-point cancellation",
                                k - 1,
                                h_grid[k - 1],
                                h_grid[k]
                            ));
                        }
                    }
                    // E[Y|x] = mean over standard-normal nodes of h⁻¹(z_k | x),
                    // each obtained by bracketing z_k in the monotone grid and
                    // linearly interpolating y. z_k below/above the support map to
                    // the support endpoints (the finite-support tails carry no mass
                    // between the clamp and the endpoint).
                    let mut acc = 0.0_f64;
                    for &z in z_nodes_ref.iter() {
                        let y_inv = if z <= h_grid[0] {
                            grid_y_ref[0]
                        } else if z >= h_grid[GRID - 1] {
                            grid_y_ref[GRID - 1]
                        } else {
                            // Binary search for the bracketing interval.
                            let mut lo = 0usize;
                            let mut hi = GRID - 1;
                            while hi - lo > 1 {
                                let mid = (lo + hi) / 2;
                                if h_grid[mid] <= z {
                                    lo = mid;
                                } else {
                                    hi = mid;
                                }
                            }
                            let t = (z - h_grid[lo]) / (h_grid[hi] - h_grid[lo]);
                            grid_y_ref[lo] + t * (grid_y_ref[hi] - grid_y_ref[lo])
                        };
                        acc += y_inv;
                    }
                    Ok(acc / (QUAD as f64))
                })
                .collect();
            let conditional_mean = ndarray::Array1::<f64>::from_vec(
                mean_results.into_iter().collect::<Result<Vec<_>, _>>()?,
            );
            if conditional_mean.iter().any(|value| !value.is_finite()) {
                return Err(PredictInputError::InvalidInput {
                    reason:
                        "prediction failed: transformation-normal conditional mean E[Y|x] produced non-finite values"
                            .to_string(),
                });
            }
            // The predictor passes the offset through unchanged as `eta` and
            // `mean`, so storing E[Y|x] here yields a y-independent response-scale
            // prediction for both columns on a covariate-only frame.
            Ok(PredictInput {
                design: DesignMatrix::from(ndarray::Array2::from_shape_fn((n, 1), |_| 1.0)),
                offset: conditional_mean,
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
                auxiliary_matrix: None,
            })
        }
    }
}

/// Build a `PredictInput` for model types backed directly by `PredictableModel`.
///
/// Survival prediction has its own design assembly because it needs entry/exit
/// time geometry before it can call the same predictor/output machinery.
pub fn build_predict_input_for_model(
    model: &FittedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
    offset_noise: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<PredictInput, String> {
    build_predict_input_for_model_inner(
        model,
        data,
        col_map,
        training_headers,
        offset,
        offset_noise,
        noise_offset_supplied,
    )
    .map_err(Into::into)
}
