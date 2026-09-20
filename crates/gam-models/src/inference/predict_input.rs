use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::inference::predict_io::{FittedLatentScoreMap, LatentConditioningSpan, PredictInput};
use gam_linalg::matrix::DesignMatrix;
use gam_linalg::utils::inf_norm;
use gam_math::probability::standard_normal_quantile;
use gam_model_kernels::scale_design::{
    build_scale_deviation_operator, scale_transform_from_payload,
};
use crate::bms::LatentMeasureKind;
use crate::inference::model::{
    FittedModel, FittedModelError, PredictModelClass, SavedTransformationNormalGeometry,
    append_deployment_extension_columns,
};
use crate::survival::predict::SurvivalPredictError;
use crate::survival::predict::{
    fit_result_from_saved_model_for_prediction, resolve_termspec_for_prediction,
};
use crate::transformation_normal::{
    CTN_LOCATION_COLUMNS, CtnRowBases, CtnRowFloors, CtnTransformTable,
    TRANSFORMATION_MONOTONICITY_EPS, ctn_endpoint_bases, ctn_laplace_quantile_correction,
    ctn_response_bases_at, ctn_response_second_derivative_basis_at, ctn_row_geometry,
    transformation_normal_pit_score,
};
use gam_problem::BlockRole;
use gam_terms::smooth::build_term_collection_prediction_design;

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

/// The residual repair block's prediction features (gam#2924), read by the
/// column names the fit recorded. The columns are the caller's responsibility
/// to centre on the same reference law as at fit time; a missing or non-finite
/// column is refused rather than defaulted to zero, because a zero residual is
/// a statement ("this person's genome carries nothing beyond the score") the
/// predictor must not make on the caller's behalf.
fn build_residual_repair_feature_matrix(
    geometry: &crate::bms::ResidualRepairGeometry,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<Array2<f64>, PredictInputError> {
    let n = data.nrows();
    let width = geometry.width();
    let mut out = Array2::<f64>::zeros((n, width));
    for (local_col, name) in geometry.columns.iter().enumerate() {
        let col = *col_map.get(name).ok_or_else(|| PredictInputError::InvalidInput {
            reason: format!(
                "residual repair prediction requires column '{name}', which the prediction table \
                 does not carry"
            ),
        })?;
        if col >= data.ncols() {
            return Err(PredictInputError::DimensionMismatch {
                reason: format!(
                    "residual repair prediction column '{name}' resolves to index {col}, out of \
                     bounds for {} columns",
                    data.ncols()
                ),
            });
        }
        let column = data.column(col);
        if let Some(row) = column.iter().position(|v| !v.is_finite()) {
            return Err(PredictInputError::InvalidInput {
                reason: format!("residual repair prediction column '{name}' is non-finite at row {row}"),
            });
        }
        out.column_mut(local_col).assign(&column);
    }
    Ok(out)
}

/// The scaled context covariates a saved marginal-slope model's local latent
/// law is replayed from, one row per prediction row; `None` for every other law.
/// Shared by both marginal-slope families' predictors (gam#2926).
pub fn build_marginal_slope_local_auxiliary_matrix(
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

/// Number of nodes in the shared fine response grid used to tabulate (and then
/// invert) the CTM conditional transform `h(y|x)`.
const TRANSFORMATION_NORMAL_INVERSION_GRID: usize = 257;

/// Number of standard-normal quadrature nodes (midpoint rule in probability
/// space) used to average `h⁻¹(Z|x)` into the response-scale mean `E[Y|x]`.
const TRANSFORMATION_NORMAL_MEAN_QUADRATURE: usize = 48;

/// The chart a saved CTN model was written in, together with everything needed
/// to replay its transform: the frozen response knots / degree / coefficient
/// transform, the structural endpoint bases, and the three monotonicity floors.
///
/// This exists so the two replay paths in this module — the `E[Y|x]` inversion
/// grid and the observed-response score — cannot read the same payload
/// differently. Before gam#2680 they were two independent 60-line transcriptions
/// of the payload and both of them evaluated `Σ_k I_k(y)·γ_k(x)²`, a chart the
/// fit had left behind in `#2306`, while validating the very `parameterization`
/// marker that says so. The marker is now carried into
/// [`ctn_row_geometry`] rather than merely checked.
struct SavedCtnChart {
    chart: crate::inference::model::TransformationNormalParameterization,
    knots: Array1<f64>,
    transform: Array2<f64>,
    degree: usize,
    median: f64,
    /// `[1, 0, …, 0]` — the value basis at the lower support knot.
    lower_basis: Array1<f64>,
    /// `[1, 1ᵀT_{·1}, …]` — the value basis at the upper support knot.
    upper_basis: Array1<f64>,
    lower_floor: f64,
    upper_floor: f64,
    /// `p_resp = 1 + p_shape`.
    p_resp: usize,
    /// The PIT clip the fit calibrated its score with.
    clip_eps: f64,
}

impl SavedCtnChart {
    fn from_model(model: &FittedModel) -> Result<Self, PredictInputError> {
        let payload = model.payload();
        let geometry: &SavedTransformationNormalGeometry = payload
            .transformation_geometry
            .as_ref()
            .ok_or_else(|| PredictInputError::MissingMetadata {
                reason: "saved transformation-normal model missing the coefficient-chart geometry \
                         record; a pre-direct-α payload cannot be replayed"
                    .to_string(),
            })?;
        let knot_values = payload
            .transformation_response_knots
            .as_ref()
            .ok_or_else(|| PredictInputError::MissingMetadata {
                reason: "saved transformation-normal model missing response_knots".to_string(),
            })?;
        let transform_rows = payload
            .transformation_response_transform
            .as_ref()
            .ok_or_else(|| PredictInputError::MissingMetadata {
                reason: "saved transformation-normal model missing response_transform".to_string(),
            })?;
        let degree = payload.transformation_response_degree.ok_or_else(|| {
            PredictInputError::MissingMetadata {
                reason: "saved transformation-normal model missing response_degree".to_string(),
            }
        })?;
        let median = payload.transformation_response_median.ok_or_else(|| {
            PredictInputError::MissingMetadata {
                reason: "saved transformation-normal model missing response_median".to_string(),
            }
        })?;
        let calibration = payload
            .transformation_score_calibration
            .as_ref()
            .ok_or_else(|| PredictInputError::MissingMetadata {
                reason: "saved transformation-normal model missing score calibration".to_string(),
            })?;
        calibration.validate("saved transformation-normal score calibration")?;

        if knot_values.is_empty() {
            return Err(PredictInputError::MissingMetadata {
                reason: "saved transformation-normal response knots are empty".to_string(),
            });
        }
        let rows = transform_rows.len();
        let cols = transform_rows.first().map_or(0, Vec::len);
        if rows == 0 || cols == 0 || transform_rows.iter().any(|row| row.len() != cols) {
            return Err(PredictInputError::MissingMetadata {
                reason: "saved transformation-normal response transform is empty or ragged"
                    .to_string(),
            });
        }
        let mut transform = Array2::<f64>::zeros((rows, cols));
        for (i, row) in transform_rows.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                transform[[i, j]] = value;
            }
        }
        let knots = Array1::from_vec(knot_values.clone());
        let y_lo = knots[0];
        let y_hi = knots[knots.len() - 1];
        if !(y_hi > y_lo) {
            return Err(PredictInputError::InvalidInput {
                reason: format!(
                    "transformation-normal response support is degenerate: lo={y_lo}, hi={y_hi}"
                ),
            });
        }
        let p_resp = cols + CTN_LOCATION_COLUMNS;
        if geometry.shape_coordinate_count + CTN_LOCATION_COLUMNS != p_resp {
            return Err(PredictInputError::DimensionMismatch {
                reason: format!(
                    "saved CTN geometry declares {} shape coordinates but the persisted response \
                     transform carries {cols}",
                    geometry.shape_coordinate_count
                ),
            });
        }
        let (lower_basis, upper_basis) = ctn_endpoint_bases(&transform);
        Ok(Self {
            chart: geometry.parameterization,
            knots,
            transform,
            degree,
            median,
            lower_basis,
            upper_basis,
            lower_floor: TRANSFORMATION_MONOTONICITY_EPS * (y_lo - median),
            upper_floor: TRANSFORMATION_MONOTONICITY_EPS * (y_hi - median),
            p_resp,
            clip_eps: calibration.clip_eps,
        })
    }

    fn support(&self) -> (f64, f64) {
        (self.knots[0], self.knots[self.knots.len() - 1])
    }

    /// `([1, I_k(y)·T], [0, M_k(y)·T])` at arbitrary response values, on the
    /// frozen basis. Both are returned even where only `h` is consumed: the
    /// chart evaluator computes `h'` alongside it, and handing it the value
    /// basis in the derivative slot would make `CtnRowGeometry::h_prime` a
    /// number that means nothing.
    fn bases_at(&self, y: &Array1<f64>) -> Result<(Array2<f64>, Array2<f64>), PredictInputError> {
        let (value, derivative) = ctn_response_bases_at(
            y.view(),
            self.knots.view(),
            self.degree,
            Some(&self.transform),
        )
        .map_err(|reason| PredictInputError::InvalidInput { reason })?;
        if value.ncols() != self.p_resp {
            return Err(PredictInputError::DimensionMismatch {
                reason: format!(
                    "rebuilt transformation-normal response basis has {} columns, saved layout \
                     requires {}",
                    value.ncols(),
                    self.p_resp
                ),
            });
        }
        Ok((value, derivative))
    }

    /// `[0, M′_k(y)·T]` at arbitrary response values, on the frozen basis: the
    /// curvature basis the posterior-mean correction reads `h″` off.
    fn second_derivative_basis_at(
        &self,
        y: &Array1<f64>,
    ) -> Result<Array2<f64>, PredictInputError> {
        let second = ctn_response_second_derivative_basis_at(
            y.view(),
            self.knots.view(),
            self.degree,
            Some(&self.transform),
        )
        .map_err(|reason| PredictInputError::InvalidInput { reason })?;
        if second.ncols() != self.p_resp {
            return Err(PredictInputError::DimensionMismatch {
                reason: format!(
                    "rebuilt transformation-normal curvature basis has {} columns, saved layout \
                     requires {}",
                    second.ncols(),
                    self.p_resp
                ),
            });
        }
        Ok(second)
    }

    /// The coefficient matrix `A` (`p_resp × p_cov`) behind a saved fit.
    fn coefficient_matrix<'a>(
        &self,
        model: &'a FittedModel,
        p_cov: usize,
    ) -> Result<ndarray::ArrayView2<'a, f64>, PredictInputError> {
        let fit_saved = model
            .unified()
            .ok_or_else(|| PredictInputError::MissingMetadata {
                reason: "saved transformation-normal model missing unified fit".to_string(),
            })?;
        fit_saved.require_posterior_mean("transformation-normal prediction")
            .map_err(|error| PredictInputError::InvalidInput { reason: error.to_string() })?;
        let beta = &fit_saved.blocks[0].beta;
        if beta.len() != self.p_resp * p_cov {
            return Err(PredictInputError::DimensionMismatch {
                reason: format!(
                    "beta length {} != p_resp({}) * p_cov({p_cov})",
                    beta.len(),
                    self.p_resp
                ),
            });
        }
        beta.view()
            .into_shape_with_order((self.p_resp, p_cov))
            .map_err(|error| PredictInputError::DimensionMismatch {
                reason: format!("beta reshape failed: {error}"),
            })
    }

    /// `α_k(x_i) = ψ(x_i)ᵀ A[k, :]` for one covariate row.
    fn alpha_row(
        &self,
        coefficients: &ndarray::ArrayView2<'_, f64>,
        covariate_row: ndarray::ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        Array1::from_shape_fn(self.p_resp, |k| coefficients.row(k).dot(&covariate_row))
    }

    fn floors(&self, y: f64, additive_offset: f64) -> CtnRowFloors {
        CtnRowFloors {
            additive_offset,
            value_floor: TRANSFORMATION_MONOTONICITY_EPS * (y - self.median),
            lower_floor: self.lower_floor,
            upper_floor: self.upper_floor,
        }
    }
}

/// Materialize the per-row monotone conditional transform `h(y | x_i)` of a
/// fitted conditional transformation-normal (CTM) model on a shared fine
/// response grid, in the chart the model was written in:
///   `h(y|x) = α₀(x) + Σ_{r≥1} I_r(y)·α_r(x) + offset + ε·(y − median)`,
/// `α_r(x) = A[r,:] · cov_row(x)`, `I_r` the frozen I-spline value basis, and
/// `α_r ≥ 0` on the fitted rows by the Khatri-Rao monotonicity cone.
///
/// Returning the tabulated curve lets the response-scale conditional mean
/// `E[Y|x]` (predict, #1612) and inverse-transform response-scale sampling
/// `Y = h⁻¹(Z|x)` (generate, #1613) be built by inverting the SAME curve, so the
/// two paths can never disagree on the underlying transform.
///
/// Returns the [`CtnTransformTable`]: the length-`G` response grid, the per-row
/// latent `h(grid_y[k] | x_i)` (strictly increasing in `k`), and the two tail
/// slopes `h'(y_lo | x_i)`, `h'(y_hi | x_i)` that make the transform invertible
/// off the ends of the table. The tails are not an approximation — since
/// gam#2600 the CTN transform is affine beyond the boundary knots at exactly
/// those slopes — and carrying them in the same object is what stops a consumer
/// from silently truncating the predictive law at the training range.
fn transformation_normal_quantile_grid(
    model: &FittedModel,
    design: &gam_terms::smooth::TermCollectionPredictionDesign,
    n: usize,
    offset: &Array1<f64>,
) -> Result<CtnTransformTable, PredictInputError> {
    let offset = design
        .compose_offset(offset.view(), "transformation-normal prediction")
        .map_err(|error| PredictInputError::InvalidInput {
            reason: error.to_string(),
        })?;
    let saved = SavedCtnChart::from_model(model)?;
    let (y_lo, y_hi) = saved.support();
    let p_cov = design.design.ncols();
    let coefficients = saved.coefficient_matrix(model, p_cov)?;
    let cov_mat =
        design
            .design
            .try_row_chunk(0..n)
            .map_err(|error| PredictInputError::InvalidInput {
                reason: error.to_string(),
            })?;

    // A shared fine `y`-grid spanning the response support; the I-spline value
    // basis is evaluated once here and reused for every row, so the per-row
    // inversion is a cheap monotone lookup rather than a fresh basis build.
    // The two end nodes are written exactly, not derived: they are the anchors
    // the affine tails are measured from, and a last node a single ulp past
    // `y_hi` would be read as an exterior point by the basis and pick up the
    // tail branch instead of the boundary itself.
    const GRID: usize = TRANSFORMATION_NORMAL_INVERSION_GRID;
    let grid_y: Array1<f64> = Array1::from_shape_fn(GRID, |k| match k {
        0 => y_lo,
        k if k == GRID - 1 => y_hi,
        k => y_lo + (y_hi - y_lo) * (k as f64) / ((GRID - 1) as f64),
    });
    let (grid_value, grid_derivative) = saved.bases_at(&grid_y)?;

    // The tabulated latent is the raw transform `h`, NOT the clipped PIT score.
    // Since gam#2600 the model's CDF is `F = Φ(h)`, so the two agree wherever
    // the clip is inactive and differ only past `Φ⁻¹(clip_eps)` — and there the
    // clip is exactly the wrong operation for an inversion table: it flattens
    // the ends of a curve whose whole purpose here is to be inverted, which both
    // destroys the strict monotonicity `CtnTransformTable` requires and
    // re-imposes, one clip window further out, the very truncation this table
    // exists to remove. The clip belongs where a *score* is reported
    // (`transformation_normal_observed_scores`), which is the quantity a
    // downstream consumer has to be able to represent; a quantile is not.
    let saved_ref = &saved;
    let grid_value_ref = &grid_value;
    let grid_derivative_ref = &grid_derivative;
    let grid_y_ref = &grid_y;
    let coefficients_ref = &coefficients;
    let cov_mat_ref = &cov_mat;
    let rows: Vec<Result<(Vec<f64>, Vec<f64>), String>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let cov_row = cov_mat_ref.row(i);
            let alpha = saved_ref.alpha_row(coefficients_ref, cov_row);
            // `h` AND `h'` from the one chart evaluator, which computes both at
            // every node anyway. The derivative is what makes the tabulated
            // transform a Hermite interpolant rather than a chord, and its two
            // END values are the slopes of the transform's affine tails (the
            // first and last grid nodes ARE the fitted support endpoints).
            let mut h_row = vec![0.0_f64; GRID];
            let mut slope_row = vec![0.0_f64; GRID];
            for k in 0..GRID {
                let value_row = grid_value_ref.row(k);
                let derivative_row = grid_derivative_ref.row(k);
                let geometry = ctn_row_geometry(
                    saved_ref.chart,
                    alpha.view(),
                    CtnRowBases {
                        value: value_row,
                        derivative: derivative_row,
                        lower: saved_ref.lower_basis.view(),
                        upper: saved_ref.upper_basis.view(),
                    },
                    saved_ref.floors(grid_y_ref[k], offset[i]),
                );
                h_row[k] = geometry.h;
                slope_row[k] = geometry.h_prime;
                if !(h_row[k].is_finite() && slope_row[k].is_finite()) {
                    let max_abs_cov = inf_norm(cov_row.iter().copied());
                    return Err(format!(
                        "transformation-normal transform at row {i}, grid node {k} is not finite: h={:.6e}, h'={:.6e}; max_abs_covariate_basis={max_abs_cov:.6e}",
                        h_row[k],
                        slope_row[k]
                    ));
                }
            }
            // Structural monotonicity guard: under SCOP `h' ≥ ε > 0`, so a
            // non-increasing grid signals floating-point cancellation.
            for k in 1..GRID {
                if h_row[k] <= h_row[k - 1] {
                    return Err(format!(
                        "transformation-normal transform is not strictly increasing at row {i} between grid nodes {} and {k} (h={:.6e} -> {:.6e}); under SCOP h' = ε + Σ M_r α_r is structurally positive, so this indicates floating-point cancellation",
                        k - 1,
                        h_row[k - 1],
                        h_row[k]
                    ));
                }
            }
            Ok((h_row, slope_row))
        })
        .collect();
    let mut h_grid = Array2::<f64>::zeros((n, GRID));
    let mut slope_grid = Array2::<f64>::zeros((n, GRID));
    for (i, row) in rows.into_iter().enumerate() {
        let (h_row, slope_row) = row.map_err(|reason| PredictInputError::InvalidInput {
            reason: format!("prediction failed: {reason}"),
        })?;
        for (k, v) in h_row.into_iter().enumerate() {
            h_grid[[i, k]] = v;
        }
        for (k, v) in slope_row.into_iter().enumerate() {
            slope_grid[[i, k]] = v;
        }
    }
    CtnTransformTable::new(grid_y, h_grid, slope_grid).map_err(|reason| {
        PredictInputError::InvalidInput {
            reason: format!("prediction failed: {reason}"),
        }
    })
}

/// Evaluate the fitted CTM's calibrated latent score at one observed response
/// per row.  This is deliberately separate from ordinary prediction:
/// `predict` returns the response-scale conditional mean `E[Y|x]`, whereas an
/// observed score is the labelled-data quantity
/// `Phi^{-1}(F_hat(y_i | x_i))` consumed by a downstream marginal-slope model.
///
/// The score is evaluated in the chart the model was written in, through the
/// same `ctn_row_geometry` the fit's own `row_quantities` uses. On the training
/// rows of the model that produced it, the result therefore reproduces
/// `block_states[0].eta` to round-off — the invariant gam#2680 broke.
fn transformation_normal_observed_scores(
    model: &FittedModel,
    design: &gam_terms::smooth::TermCollectionPredictionDesign,
    response: &Array1<f64>,
    offset: &Array1<f64>,
) -> Result<Array1<f64>, PredictInputError> {
    let n = response.len();
    if design.design.nrows() != n || offset.len() != n {
        return Err(PredictInputError::DimensionMismatch {
            reason: format!(
                "transformation-normal observed-score rows disagree: response={n}, design={}, offset={}",
                design.design.nrows(),
                offset.len()
            ),
        });
    }
    if response.iter().any(|value| !value.is_finite()) {
        return Err(PredictInputError::InvalidInput {
            reason: "transformation-normal observed responses must be finite".to_string(),
        });
    }
    let offset = design
        .compose_offset(
            offset.view(),
            "transformation-normal observed-score prediction",
        )
        .map_err(|error| PredictInputError::InvalidInput {
            reason: error.to_string(),
        })?;

    let saved = SavedCtnChart::from_model(model)?;
    let p_cov = design.design.ncols();
    let coefficients = saved.coefficient_matrix(model, p_cov)?;
    let (observed_value, observed_derivative) = saved.bases_at(response)?;
    let covariate_matrix =
        design
            .design
            .try_row_chunk(0..n)
            .map_err(|error| PredictInputError::InvalidInput {
                reason: error.to_string(),
            })?;

    let saved_ref = &saved;
    let coefficients_ref = &coefficients;
    let rows: Vec<Result<f64, String>> = (0..n)
        .into_par_iter()
        .map(|row_index| {
            let covariate_row = covariate_matrix.row(row_index);
            let alpha = saved_ref.alpha_row(coefficients_ref, covariate_row);
            let value_row = observed_value.row(row_index);
            let derivative_row = observed_derivative.row(row_index);
            let geometry = ctn_row_geometry(
                saved_ref.chart,
                alpha.view(),
                CtnRowBases {
                    value: value_row,
                    derivative: derivative_row,
                    lower: saved_ref.lower_basis.view(),
                    upper: saved_ref.upper_basis.view(),
                },
                saved_ref.floors(response[row_index], offset[row_index]),
            );
            transformation_normal_pit_score(geometry.h, saved_ref.clip_eps).map_err(|error| {
                format!("transformation-normal observed score failed at row {row_index}: {error}")
            })
        })
        .collect();

    let scores = rows
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|reason| PredictInputError::InvalidInput { reason })?;
    Ok(Array1::from_vec(scores))
}

/// Number of latent-z nodes on which the CTM predict input tabulates the
/// response-scale predictive quantile ladder `h⁻¹(z_j | x_i)`. Each ladder row
/// holds the node values and then their exact latent slopes, `2·m = 66`
/// entries, the same per-row budget the 65-node value-only ladder spent. At the
/// resulting step `0.25` the exact-slope cubic Hermite carries
/// `O(Δz⁴·y⁽⁴⁾/384) ≈ 1e-5` relative error for a lognormal response.
pub const TRANSFORMATION_NORMAL_BAND_Z_NODES: usize = 33;

/// Half-width of the latent-z ladder. `Φ(4) ≈ 0.999968`, so every two-sided
/// observation level up to ≈ 0.99993 interpolates strictly inside the ladder;
/// beyond it the band continues affinely at the end node's exact slope (#2600).
pub const TRANSFORMATION_NORMAL_BAND_Z_MAX: f64 = 4.0;

/// The fixed, evenly spaced latent-z ladder shared by the CTM input builder
/// (which tabulates `h⁻¹` on it) and the transformation-normal predictor
/// (which interpolates it to build response-scale observation bands). The CTM
/// predictive is `Y | x = h⁻¹(Z | x)` with `Z ~ N(0,1)`, so the response-scale
/// `p`-quantile is exactly `h⁻¹(Φ⁻¹(p) | x)` — quantiles map through the
/// monotone inverse transform; they are NOT `E[Y|x] ± z·σ` in latent-normal
/// units.
pub(crate) fn transformation_normal_band_z_nodes() -> Array1<f64> {
    Array1::from_shape_fn(TRANSFORMATION_NORMAL_BAND_Z_NODES, |j| {
        -TRANSFORMATION_NORMAL_BAND_Z_MAX
            + 2.0 * TRANSFORMATION_NORMAL_BAND_Z_MAX * (j as f64)
                / ((TRANSFORMATION_NORMAL_BAND_Z_NODES - 1) as f64)
    })
}

/// The standard-normal midpoint quadrature nodes `z_k = Φ⁻¹((k + ½)/QUAD)` that
/// the CTM mean averages the inverse transform over. The plug-in mean and its
/// posterior-mean correction read the same nodes.
fn transformation_normal_mean_z_nodes() -> Result<Vec<f64>, PredictInputError> {
    const QUAD: usize = TRANSFORMATION_NORMAL_MEAN_QUADRATURE;
    (0..QUAD)
        .map(|k| {
            let p = ((k as f64) + 0.5) / (QUAD as f64);
            standard_normal_quantile(p)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PredictInputError::InvalidInput { reason: e })
}

/// The Laplace-order posterior-mean correction to the plug-in CTM mean `E[Y|x]`
/// (SPEC rule 3: the default prediction is the posterior mean, never the plug-in).
///
/// The CTM transform is affine in the covariate-side coordinates,
/// `h(y|x; α) = a(y)ᵀα + offset + ε·(y − median)` with `α = ψ(x)ᵀA`, so the
/// coefficient posterior `vec(A) ~ N(Â, V)` induces `α ~ N(α̂, Σ_α)` with
/// `Σ_α[k, l] = ψᵀ V[(k, ·), (l, ·)] ψ`. The response-scale posterior mean is
/// `E_Z E_α[h⁻¹(Z|x; α)]`. To the order of the Laplace approximation itself each
/// quantile moves by `½tr(Σ_α∇²y)` (see [`ctn_laplace_quantile_correction`]),
/// averaged over the same Z nodes as the plug-in mean.
///
/// The exact expectation under that Gaussian does not exist. Its predictive CDF
/// has the closed form `F(t|x) = Φ(ĥ(t)/√(1 + s²(t)))`, `s²(t) = a(t)ᵀΣ_α a(t)`,
/// but in the affine tails both `ĥ` and `s` grow linearly in `t`, so `1 − F(t)`
/// tends to a positive constant and `∫(1 − F)` diverges. The Gaussian puts mass on
/// tail slopes `γ ≤ 0` that the monotone likelihood forbids, and `E[1/γ]`
/// diverges near `γ = 0` however small that mass is. The true posterior carries
/// the `log h′` Jacobian, so its density vanishes as `γ → 0` and its mean is
/// finite. The consistent posterior mean is therefore this same-order Laplace
/// (Tierney–Kadane) correction, not a Gaussian integral taken past the region
/// where the approximation is valid.
///
/// The correction needs the fit's coefficient posterior covariance. A saved model
/// without one is refused rather than silently reporting the plug-in.
fn transformation_normal_posterior_mean_correction(
    model: &FittedModel,
    design: &gam_terms::smooth::TermCollectionPredictionDesign,
    n: usize,
    offset: &Array1<f64>,
    table: &CtnTransformTable,
) -> Result<Array1<f64>, PredictInputError> {
    if table.nrows() != n {
        return Err(PredictInputError::DimensionMismatch {
            reason: format!(
                "transformation-normal posterior mean: the transform table has {} rows, expected \
                 {n}",
                table.nrows()
            ),
        });
    }
    let offset = design
        .compose_offset(offset.view(), "transformation-normal posterior mean")
        .map_err(|error| PredictInputError::InvalidInput {
            reason: error.to_string(),
        })?;
    let saved = SavedCtnChart::from_model(model)?;
    let p_cov = design.design.ncols();
    let coefficients = saved.coefficient_matrix(model, p_cov)?;
    let p_resp = saved.p_resp;
    let covariance = ctn_coefficient_covariance(model, p_resp * p_cov, "posterior mean")?;
    let cov_mat = design
        .design
        .try_row_chunk(0..n)
        .map_err(|error| PredictInputError::InvalidInput {
            reason: error.to_string(),
        })?;
    let z_nodes = transformation_normal_mean_z_nodes()?;
    let saved_ref = &saved;
    let coefficients_ref = &coefficients;
    let cov_mat_ref = &cov_mat;
    let offset_ref = &offset;
    let z_nodes_ref = &z_nodes;
    let rows: Vec<Result<f64, String>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let cov_row = cov_mat_ref.row(i);
            let alpha = saved_ref.alpha_row(coefficients_ref, cov_row);
            let alpha_covariance = ctn_alpha_covariance(covariance, cov_row, p_resp);
            let quantiles = Array1::from_iter(z_nodes_ref.iter().map(|&z| table.invert(i, z)));
            let (value, derivative) = saved_ref.bases_at(&quantiles).map_err(String::from)?;
            let second = saved_ref
                .second_derivative_basis_at(&quantiles)
                .map_err(String::from)?;
            let mut acc = 0.0_f64;
            for (k, &y) in quantiles.iter().enumerate() {
                let geometry = ctn_row_geometry(
                    saved_ref.chart,
                    alpha.view(),
                    CtnRowBases {
                        value: value.row(k),
                        derivative: derivative.row(k),
                        lower: saved_ref.lower_basis.view(),
                        upper: saved_ref.upper_basis.view(),
                    },
                    saved_ref.floors(y, offset_ref[i]),
                );
                acc += ctn_laplace_quantile_correction(
                    value.row(k),
                    derivative.row(k),
                    second.row(k),
                    alpha.view(),
                    alpha_covariance.view(),
                    geometry.h_prime,
                );
            }
            let correction = acc / (z_nodes_ref.len() as f64);
            if correction.is_finite() {
                Ok(correction)
            } else {
                Err(format!(
                    "transformation-normal posterior-mean correction at row {i} is not finite \
                     ({correction})"
                ))
            }
        })
        .collect();
    let mut correction = Array1::<f64>::zeros(n);
    for (i, row) in rows.into_iter().enumerate() {
        correction[i] = row.map_err(|reason| PredictInputError::InvalidInput {
            reason: format!("prediction failed: {reason}"),
        })?;
    }
    Ok(correction)
}

/// The saved coefficient posterior covariance, checked against the CTN layout.
/// A model saved without one is refused rather than silently reporting the
/// plug-in.
fn ctn_coefficient_covariance<'a>(
    model: &'a FittedModel,
    p_total: usize,
    purpose: &str,
) -> Result<&'a Array2<f64>, PredictInputError> {
    let covariance = model
        .unified()
        .and_then(|fit| fit.beta_covariance())
        .ok_or_else(|| PredictInputError::MissingMetadata {
            reason: format!(
                "transformation-normal {purpose} requires the saved coefficient posterior \
                 covariance; refit the model with this runtime"
            ),
        })?;
    if covariance.dim() != (p_total, p_total) {
        return Err(PredictInputError::DimensionMismatch {
            reason: format!(
                "transformation-normal coefficient covariance is {:?}, the saved layout requires \
                 {p_total}x{p_total}",
                covariance.dim()
            ),
        });
    }
    Ok(covariance)
}

/// `Σ_α = (I ⊗ ψᵀ) V (I ⊗ ψ)`: the posterior covariance of one row's
/// covariate-side coordinates `α = ψ(x)ᵀA` under `vec(A) ~ N(Â, V)`, in the
/// row-major layout of `vec(A)`.
fn ctn_alpha_covariance(
    covariance: &Array2<f64>,
    covariate_row: ndarray::ArrayView1<'_, f64>,
    p_resp: usize,
) -> Array2<f64> {
    let p_cov = covariate_row.len();
    let mut alpha_covariance = Array2::<f64>::zeros((p_resp, p_resp));
    for k in 0..p_resp {
        for l in k..p_resp {
            let mut acc = 0.0_f64;
            for c in 0..p_cov {
                let psi_c = covariate_row[c];
                if psi_c == 0.0 {
                    continue;
                }
                let row_index = k * p_cov + c;
                for d in 0..p_cov {
                    acc += psi_c * covariance[[row_index, l * p_cov + d]] * covariate_row[d];
                }
            }
            alpha_covariance[[k, l]] = acc;
            alpha_covariance[[l, k]] = acc;
        }
    }
    alpha_covariance
}

/// One affine exterior of the posterior-predictive latent
/// `g(t) = ĥ(t)/√(1 + s²(t))`, `s²(t) = a(t)ᵀΣ_α a(t)`.
///
/// Past a boundary knot `y_b` the response basis continues affinely, so with
/// `d = t − y_b` the plug-in transform is `ĥ_b + γ·d` and the transform's
/// posterior variance is the exact quadratic `s_b² + 2c·d + v·d²`, with
/// `γ = h′(y_b)`, `c = a′ᵀΣ_α a` and `v = a′ᵀΣ_α a′` read at the knot.
#[derive(Debug)]
struct CtnPredictiveTail {
    level: f64,
    slope: f64,
    variance: f64,
    cross: f64,
    slope_variance: f64,
}

impl CtnPredictiveTail {
    fn scale(&self, d: f64) -> f64 {
        1.0 + self.variance + d * (2.0 * self.cross + d * self.slope_variance)
    }

    /// `g′(d) = N(d)/(1 + s²(d))^{3/2}`, whose numerator is linear:
    /// `N(d) = γ(1 + s_b²) − ĥ_b·c + d·(γ·c − ĥ_b·v)`.
    fn latent_slope(&self, d: f64) -> f64 {
        let scale = self.scale(d);
        self.numerator(d) / (scale * scale.sqrt())
    }

    fn numerator(&self, d: f64) -> f64 {
        self.slope * (1.0 + self.variance) - self.level * self.cross
            + d * (self.slope * self.cross - self.level * self.slope_variance)
    }

    /// Whether `g` increases strictly on the half-line `d·direction ≥ 0`. The
    /// numerator is linear, so it stays positive there exactly when it is
    /// positive at the knot and does not decrease towards the far end.
    fn increases_towards(&self, direction: f64) -> bool {
        let rate = self.slope * self.cross - self.level * self.slope_variance;
        self.numerator(0.0) > 0.0 && rate * direction >= 0.0
    }

    /// `lim g(d)` as `d → direction·∞`: `direction·γ/√v`, and unbounded when the
    /// tail slope carries no posterior variance (then `c = 0` as well, by
    /// Cauchy–Schwarz, and `g` is affine).
    fn limit(&self, direction: f64) -> f64 {
        if self.slope_variance > 0.0 {
            direction * self.slope / self.slope_variance.sqrt()
        } else {
            direction * f64::INFINITY
        }
    }

    /// The `d` on the half-line `d·direction ≥ 0` with `g(d) = z`, on a tail where
    /// `g` is strictly increasing. Squaring `ĥ_b + γd = z·√(1 + s²(d))` gives
    /// `A·d² + 2B·d + C = 0` with `A = γ² − z²v`, `B = ĥ_b·γ − z²c` and
    /// `C = ĥ_b² − z²(1 + s_b²)`; of its real roots, the crossing is the one on the
    /// half-line whose transform has the sign of `z`.
    fn root(&self, z: f64, direction: f64) -> Option<f64> {
        let z2 = z * z;
        let a = self.slope * self.slope - z2 * self.slope_variance;
        let b = self.level * self.slope - z2 * self.cross;
        let c = self.level * self.level - z2 * (1.0 + self.variance);
        let candidates = if a == 0.0 {
            if b == 0.0 {
                Vec::new()
            } else {
                vec![-c / (2.0 * b)]
            }
        } else {
            let discriminant = b * b - a * c;
            if discriminant < 0.0 {
                Vec::new()
            } else {
                // Stable pair: q = −(B + sign(B)·√D), roots q/A and C/q.
                let q = -(b + b.signum() * discriminant.sqrt());
                if q == 0.0 {
                    vec![-b / a]
                } else {
                    vec![q / a, c / q]
                }
            }
        };
        candidates
            .into_iter()
            .find(|&d| d * direction >= 0.0 && (self.level + self.slope * d) * z >= 0.0)
    }
}

/// The response-scale posterior-predictive quantile ladder (SPEC rule 3).
///
/// Under the coefficient posterior `α ~ N(α̂, Σ_α)` the transform at a fixed
/// response is Gaussian, `h(t; α) ~ N(ĥ(t), s²(t))`, so the predictive CDF of
/// `Y | x` has the closed form `F(t|x) = E_α Φ(h(t; α)) = Φ(g(t))` with
/// `g(t) = ĥ(t)/√(1 + s²(t))`. Its `p`-quantile is `g⁻¹(Φ⁻¹(p)|x)`; at `s² = 0` it
/// is the plug-in quantile, and coefficient uncertainty only widens a band.
///
/// Each row holds `g⁻¹(z_j|x)` on the fixed latent ladder followed by the exact
/// latent slopes `1/g′`. Inside the fitted support `g` is tabulated on the
/// transform table's grid and inverted on its Hermite interpolant; past each
/// boundary knot the exterior is closed form ([`CtnPredictiveTail`]).
///
/// The Gaussian puts mass on transform slopes the monotone likelihood forbids, so
/// two things can fail to exist, and each is published in the row rather than
/// papered over:
/// * `g` saturates at `±γ/√v` in each exterior, so a latent level at or past that
///   limit is never reached. Its quantile is `±∞`, stored as such.
/// * If `g` is not strictly increasing, `Φ(g)` is not a distribution function and
///   the row has no quantiles. The whole row is stored as NaN.
///
/// The predictor refuses a band that reads either, naming which.
fn transformation_normal_predictive_ladder(
    model: &FittedModel,
    design: &gam_terms::smooth::TermCollectionPredictionDesign,
    n: usize,
    offset: &Array1<f64>,
    table: &CtnTransformTable,
) -> Result<Array2<f64>, PredictInputError> {
    if table.nrows() != n {
        return Err(PredictInputError::DimensionMismatch {
            reason: format!(
                "transformation-normal predictive band: the transform table has {} rows, \
                 expected {n}",
                table.nrows()
            ),
        });
    }
    let offset = design
        .compose_offset(offset.view(), "transformation-normal predictive band")
        .map_err(|error| PredictInputError::InvalidInput {
            reason: error.to_string(),
        })?;
    let saved = SavedCtnChart::from_model(model)?;
    let p_cov = design.design.ncols();
    let coefficients = saved.coefficient_matrix(model, p_cov)?;
    let p_resp = saved.p_resp;
    let covariance = ctn_coefficient_covariance(model, p_resp * p_cov, "predictive band")?;
    let cov_mat = design
        .design
        .try_row_chunk(0..n)
        .map_err(|error| PredictInputError::InvalidInput {
            reason: error.to_string(),
        })?;
    let grid_y = table.grid_y().to_owned();
    let (grid_value, grid_derivative) = saved.bases_at(&grid_y)?;
    let z_nodes = transformation_normal_band_z_nodes();
    let m = z_nodes.len();
    let saved_ref = &saved;
    let coefficients_ref = &coefficients;
    let cov_mat_ref = &cov_mat;
    let offset_ref = &offset;
    let grid_y_ref = &grid_y;
    let grid_value_ref = &grid_value;
    let grid_derivative_ref = &grid_derivative;
    let z_nodes_ref = &z_nodes;
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let g = grid_y_ref.len();
            let cov_row = cov_mat_ref.row(i);
            let alpha = saved_ref.alpha_row(coefficients_ref, cov_row);
            let alpha_covariance = ctn_alpha_covariance(covariance, cov_row, p_resp);
            let mut latent = Array2::<f64>::zeros((1, g));
            let mut latent_slope = Array2::<f64>::zeros((1, g));
            let mut tails = Vec::with_capacity(2);
            for k in 0..g {
                let value = grid_value_ref.row(k);
                let derivative = grid_derivative_ref.row(k);
                let geometry = ctn_row_geometry(
                    saved_ref.chart,
                    alpha.view(),
                    CtnRowBases {
                        value,
                        derivative,
                        lower: saved_ref.lower_basis.view(),
                        upper: saved_ref.upper_basis.view(),
                    },
                    saved_ref.floors(grid_y_ref[k], offset_ref[i]),
                );
                let sigma_value = alpha_covariance.dot(&value);
                let variance = value.dot(&sigma_value);
                let cross = derivative.dot(&sigma_value);
                let scale = 1.0 + variance;
                latent[[0, k]] = geometry.h / scale.sqrt();
                latent_slope[[0, k]] =
                    geometry.h_prime / scale.sqrt() - geometry.h * cross / (scale * scale.sqrt());
                if k == 0 || k == g - 1 {
                    tails.push(CtnPredictiveTail {
                        level: geometry.h,
                        slope: geometry.h_prime,
                        variance,
                        cross,
                        slope_variance: derivative.dot(&alpha_covariance.dot(&derivative)),
                    });
                }
            }
            let mut ladder = vec![f64::NAN; 2 * m];
            let (lower_tail, upper_tail) = (&tails[0], &tails[1]);
            let interior = match CtnTransformTable::new(grid_y_ref.clone(), latent, latent_slope) {
                Ok(interior)
                    if lower_tail.increases_towards(-1.0) && upper_tail.increases_towards(1.0) =>
                {
                    interior
                }
                _ => return ladder,
            };
            let latent_view = interior.latent();
            let (g_lo, g_hi) = (latent_view[[0, 0]], latent_view[[0, g - 1]]);
            for (j, &z) in z_nodes_ref.iter().enumerate() {
                let (value, slope) = if z < g_lo {
                    match lower_tail.root(z, -1.0) {
                        Some(d) if z > lower_tail.limit(-1.0) => {
                            (grid_y_ref[0] + d, 1.0 / lower_tail.latent_slope(d))
                        }
                        _ => (f64::NEG_INFINITY, f64::INFINITY),
                    }
                } else if z > g_hi {
                    match upper_tail.root(z, 1.0) {
                        Some(d) if z < upper_tail.limit(1.0) => {
                            (grid_y_ref[g - 1] + d, 1.0 / upper_tail.latent_slope(d))
                        }
                        _ => (f64::INFINITY, f64::INFINITY),
                    }
                } else {
                    interior.invert_with_slope(0, z)
                };
                ladder[j] = value;
                ladder[m + j] = slope;
            }
            ladder
        })
        .collect();
    let mut quantile_ladder = Array2::<f64>::zeros((n, 2 * m));
    for (i, row) in rows.into_iter().enumerate() {
        for (j, value) in row.into_iter().enumerate() {
            quantile_ladder[[i, j]] = value;
        }
    }
    Ok(quantile_ladder)
}

/// The plug-in response-scale conditional mean `E[Y|x] = E_{Z~N(0,1)}[h⁻¹(Z|x)]`
/// at the table's coefficients, for each row of a CTM transform table, by
/// averaging the inverse over a standard-normal midpoint quadrature in probability
/// space (see the predict branch for the derivation). The generate sampler's
/// reference mean (#1613) and each posterior draw's mean read this directly;
/// `predict` adds [`transformation_normal_posterior_mean_correction`] to it.
///
/// The outermost quadrature nodes routinely fall past the tabulated latent
/// range — `Φ(h(y_lo|x))` is around `1/(n+1)` for a well-calibrated fit, and the
/// extreme midpoint node sits at `1/(2·QUAD)` — so this average is only an
/// average of the model's own quantile function because
/// [`CtnTransformTable::invert`] continues through the affine tails instead of
/// returning the support endpoint (gam#2600).
fn transformation_normal_conditional_mean(
    table: &CtnTransformTable,
) -> Result<Array1<f64>, PredictInputError> {
    let n = table.nrows();
    let z_nodes = transformation_normal_mean_z_nodes()?;
    let mean = Array1::<f64>::from_shape_fn(n, |i| {
        let mut acc = 0.0_f64;
        for &z in &z_nodes {
            acc += table.invert(i, z);
        }
        acc / (z_nodes.len() as f64)
    });
    if mean.iter().any(|value| !value.is_finite()) {
        return Err(PredictInputError::InvalidInput {
            reason: "transformation-normal conditional mean E[Y|x] produced non-finite values"
                .to_string(),
        });
    }
    Ok(mean)
}

/// The response-scale conditional quantile grid for a fitted CTM at the supplied
/// covariates — the public entry the `gam generate` path uses to build an
/// inverse-transform sampler (#1613).
pub struct TransformationNormalQuantileGrid {
    /// The fitted transform `h(·|x_i)`, tabulated on a shared response grid and
    /// carrying the slopes of its two affine tails. This is the object both the
    /// inverse-transform sampler and the `E[Y|x]` quadrature invert, so neither
    /// can truncate the predictive law at the training range (gam#2600).
    pub table: CtnTransformTable,
    /// Plug-in response-scale conditional mean `E[Y|x_i]` at the grid's
    /// coefficients: the generate spec's reference mean, and each posterior
    /// draw's mean. `predict` reports it plus the Laplace-order posterior-mean
    /// correction (#1612, SPEC rule 3).
    pub conditional_mean: Array1<f64>,
}

/// Build the CTM conditional quantile grid + response-scale mean at the supplied
/// covariates. Mirrors the design assembly of [`build_predict_input_for_model`]
/// so generation and prediction rebuild exactly the same covariate design and
/// invert the same monotone transform.
pub fn build_transformation_normal_quantile_grid(
    model: &FittedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
) -> Result<TransformationNormalQuantileGrid, String> {
    if model.predict_model_class() != PredictModelClass::TransformationNormal {
        return Err(
            "build_transformation_normal_quantile_grid called on a non-transformation-normal model"
                .to_string(),
        );
    }
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )
    .map_err(|e| String::from(PredictInputError::from(e)))?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let design_input = clipped.as_ref().map_or(data, |arr| arr.view());
    let design = build_term_collection_prediction_design(design_input, &spec)
        .map_err(|e| format!("failed to build generation design: {e}"))?;
    let n = data.nrows();
    if offset.len() != n {
        return Err(format!(
            "generation offset length mismatch: rows={n}, offset={}",
            offset.len()
        ));
    }
    let table =
        transformation_normal_quantile_grid(model, &design, n, offset).map_err(String::from)?;
    let conditional_mean = transformation_normal_conditional_mean(&table).map_err(String::from)?;
    Ok(TransformationNormalQuantileGrid {
        table,
        conditional_mean,
    })
}

/// Evaluate the calibrated CTM score `Phi^-1(F_hat(y_i | x_i))` at labelled
/// rows.  Ordinary prediction intentionally returns `E[Y|x]`; callers that
/// need the generated regressor for a marginal-slope stage must use this
/// observed-response API so a response mean can never be mistaken for a
/// latent score.
pub fn build_transformation_normal_observed_scores(
    model: &FittedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    response: &Array1<f64>,
    offset: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    if model.predict_model_class() != PredictModelClass::TransformationNormal {
        return Err(
            "build_transformation_normal_observed_scores called on a non-transformation-normal model"
                .to_string(),
        );
    }
    if response.len() != data.nrows() || offset.len() != data.nrows() {
        return Err(format!(
            "transformation-normal observed-score row mismatch: data={}, response={}, offset={}",
            data.nrows(),
            response.len(),
            offset.len()
        ));
    }
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )
    .map_err(|error| String::from(PredictInputError::from(error)))?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let design_input = clipped.as_ref().map_or(data, |array| array.view());
    let design = build_term_collection_prediction_design(design_input, &spec)
        .map_err(|error| format!("failed to build observed-score design: {error}"))?;
    transformation_normal_observed_scores(model, &design, response, offset).map_err(Into::into)
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
    let design = build_term_collection_prediction_design(design_input, &spec).map_err(|e| {
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
            // Resolve the saved runtime through its typed error path instead of
            // `has_link_wiggle`, whose boolean surface cannot distinguish no
            // wiggle from partial/corrupt metadata.  Prediction and affine
            // export must fail loudly on the latter.
            let link_wiggle = model.saved_link_wiggle()?;
            let beta = if link_wiggle.is_some() {
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
            let mean_offset = design
                .compose_offset(offset.view(), "standard prediction")
                .map_err(|error| PredictInputError::InvalidInput {
                    reason: error.to_string(),
                })?;
            Ok(PredictInput {
                design: mean_design,
                offset: mean_offset,
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
            let design_noise_raw = build_term_collection_prediction_design(design_input, &spec_noise)
                .map_err(|e| PredictInputError::InvalidInput {
                    reason: format!("failed to build noise prediction design: {e}"),
                })?;
            let mean_offset = design
                .compose_offset(offset.view(), "location-scale mean prediction")
                .map_err(|error| PredictInputError::InvalidInput {
                    reason: error.to_string(),
                })?;
            let noise_offset = design_noise_raw
                .compose_offset(offset_noise.view(), "location-scale noise prediction")
                .map_err(|error| PredictInputError::InvalidInput {
                    reason: error.to_string(),
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
                offset: mean_offset,
                design_noise: Some(prepared_noise_design),
                offset_noise: Some(noise_offset),
                auxiliary_scalar: None,
                auxiliary_matrix: None,
            })
        }
        PredictModelClass::BernoulliMarginalSlope => {
            let z = crate::inference::ctn::latent_scores(model, data, col_map)
                .map_err(|reason| PredictInputError::InvalidInput { reason })?;
            let spec_slope = resolve_termspec_for_prediction(
                &model.resolved_slopespec.as_ref().cloned(),
                training_headers,
                col_map,
                "resolved_slopespec",
            )?;
            let design_slope = build_term_collection_prediction_design(design_input, &spec_slope)
                .map_err(|e| PredictInputError::InvalidInput {
                    reason: format!("failed to build slope prediction design: {e}"),
                })?;
            let mean_offset = design
                .compose_offset(offset.view(), "marginal-slope mean prediction")
                .map_err(|error| PredictInputError::InvalidInput {
                    reason: error.to_string(),
                })?;
            // The slope offset is a slope on the score as given; the model reads the
            // score `(z − mean)/sd`, on which the same slope is `sd` times it, the
            // factor the fit applied (gam#3231).
            let score_sd = model
                .latent_z_normalization
                .as_ref()
                .ok_or_else(|| PredictInputError::MissingMetadata {
                    reason: "marginal-slope prediction requires the saved latent-z normalization"
                        .to_string(),
                })?
                .sd;
            let slope_offset = design_slope
                .compose_offset(offset_noise.view(), "marginal-slope slope prediction")
                .map_err(|error| PredictInputError::InvalidInput {
                    reason: error.to_string(),
                })?
                * score_sd;
            let local = build_marginal_slope_local_auxiliary_matrix(model, design_input, col_map)?;
            let auxiliary_matrix = match model.residual_repair.as_ref() {
                None => local,
                // Local-empirical conditioning columns first, the residual
                // features after them; the predictor reads each block by width.
                Some(geometry) => {
                    let residual =
                        build_residual_repair_feature_matrix(geometry, design_input, col_map)?;
                    Some(match local {
                        None => residual,
                        Some(local) => ndarray::concatenate![ndarray::Axis(1), local, residual],
                    })
                }
            };
            Ok(PredictInput {
                design: design.design.clone(),
                offset: mean_offset,
                design_noise: Some(design_slope.design.clone()),
                offset_noise: Some(slope_offset),
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
            // ── Response-scale conditional mean E[Y|x] (issue #1612) ──────────
            //
            // The CTM latent model is `h(Y|x) ~ N(0, 1)` with `h(·|x)` strictly
            // increasing in `y`, so the response-scale conditional mean
            //   `E[Y|x] = E_{Z~N(0,1)}[ h⁻¹(Z | x) ]`
            // is a function of the covariates alone (it does NOT depend on any
            // supplied response). We tabulate the monotone transform once via
            // `transformation_normal_quantile_grid` — the SAME curve the
            // `gam generate` inverse-transform sampler inverts (#1613) — and
            // average its inverse over a standard-normal quadrature: writing
            // `E[Y|x] = ∫₀¹ h⁻¹(Φ⁻¹(p)|x) dp`, apply the midpoint rule on `m`
            // evenly spaced probability levels `p_k = (k + ½)/m`, `z_k = Φ⁻¹(p_k)`.
            // Probability space keeps every node inside the finite I-spline
            // support (no normal-tail truncation) and needs no Gauss–Hermite
            // weights.
            let table = transformation_normal_quantile_grid(model, &design, n, offset)?;
            let conditional_mean = transformation_normal_conditional_mean(&table)?;
            // SPEC rule 3: the default prediction is the posterior mean, so the
            // plug-in mean above is carried together with its Laplace-order
            // posterior-mean correction (see the function's doc for why this, and
            // not the Gaussian predictive integral, is the posterior mean).
            let posterior_mean_correction =
                transformation_normal_posterior_mean_correction(model, &design, n, offset, &table)?;
            // Response-scale posterior-predictive quantile ladder (SPEC rule 3):
            // under the coefficient posterior the p-quantile of `Y|x` is
            // `g⁻¹(Φ⁻¹(p)|x)` with `g = ĥ/√(1 + s²)`, tabulated on the fixed z
            // ladder with its exact latent slopes, so the predictor builds genuine
            // response-scale observation bands by interpolating this matrix rather
            // than adding standard-normal quantiles to `E[Y|x]` in latent units.
            // See the builder for the closed form and for the two ways a band can
            // fail to exist.
            let quantile_ladder =
                transformation_normal_predictive_ladder(model, &design, n, offset, &table)?;
            // `offset` carries the plug-in conditional mean and `auxiliary_scalar`
            // its posterior-mean correction, so the predictor's default
            // posterior-mean pass reports their sum and the explicit plug-in pass
            // reports `offset` alone. Both are y-independent response-scale
            // predictions on a covariate-only frame.
            Ok(PredictInput {
                design: DesignMatrix::from(ndarray::Array2::from_shape_fn((n, 1), |_| 1.0)),
                offset: conditional_mean,
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: Some(posterior_mean_correction),
                auxiliary_matrix: Some(quantile_ladder),
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

impl FittedModel {
    /// The declared conditional law's standardized residual `ζ = (z − m(a))/√v(a)`
    /// of each row of `data`, through the map the fit applied to its latent score
    /// and prediction applies again (gam#3016). `None` when the fit consumed no
    /// conditional law.
    ///
    /// The conditional location-scale law is exact when ζ has the same law in every
    /// context. Its held-out adequacy check within a stratum is therefore that
    /// stratum's law of ζ against the training residual law. The raw score's law
    /// against the pooled training law also counts the location and scale shifts
    /// the conditional law absorbs.
    ///
    /// Both marginal-slope hosts read ζ from the same two pieces their predictors
    /// read: the score column (`ctn::latent_scores`) and the
    /// design of `resolved_termspec`. That design is the Bernoulli predictor's
    /// primary design and the trailing covariate block of the survival predictor's
    /// q-design, so no survival time column is needed.
    pub fn latent_conditional_residual(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        col_map: &HashMap<String, usize>,
    ) -> Result<Option<Array1<f64>>, PredictInputError> {
        let runtime = self.saved_prediction_runtime()?;
        if runtime.latent_z_conditional_calibration.is_none() {
            return Ok(None);
        }
        if self.survival_marginal_slope_joint_latent_law.is_some() {
            return Err(PredictInputError::InvalidInput {
                reason: "latent conditional residual: the model is anchored on the joint latent \
                         law of K ≥ 2 scores, which one score's residual does not describe"
                    .to_string(),
            });
        }
        let z_raw = crate::inference::ctn::latent_scores(self, data, col_map)
            .map_err(|reason| PredictInputError::InvalidInput { reason })?;
        let spec = resolve_termspec_for_prediction(
            &self.resolved_termspec,
            self.training_headers.as_ref(),
            col_map,
            "resolved_termspec",
        )?;
        let clipped = self.axis_clip_to_training_ranges(data, col_map);
        let design_input = clipped.as_ref().map_or(data, |arr| arr.view());
        let design = build_term_collection_prediction_design(design_input, &spec).map_err(|e| {
            PredictInputError::InvalidInput {
                reason: format!("failed to build the conditioning design: {e}"),
            }
        })?;
        self.fitted_latent_score(&z_raw, &design.design, "latent conditional residual")
            .map(Some)
    }

    /// Each row's fitted latent score (gam#3016): `z_raw` through the map this saved
    /// marginal-slope model applied to its latent score before any kernel read it, the
    /// saved normalisation and then the rank-INT or the conditional location-scale
    /// calibration the fit minted, the conditional one reading `a` as the whole of
    /// `conditioning`. A host that replays the fitted model on rows of its own, as the
    /// saved-model ALO does, reads the score here instead of composing the maps itself.
    pub fn fitted_latent_score(
        &self,
        z_raw: &Array1<f64>,
        conditioning: &DesignMatrix,
        context: &str,
    ) -> Result<Array1<f64>, PredictInputError> {
        let normalization =
            self.latent_z_normalization
                .as_ref()
                .ok_or_else(|| PredictInputError::MissingMetadata {
                    reason: format!("{context} requires the saved latent-z normalization"),
                })?;
        FittedLatentScoreMap {
            normalization,
            rank_int: self.latent_z_rank_int_calibration.as_ref(),
            conditional: self.latent_z_conditional_calibration.as_ref(),
            span: LatentConditioningSpan::PrimaryDesign,
        }
        .apply(z_raw, conditioning, context)
        .map_err(|error| PredictInputError::InvalidInput {
            reason: error.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `g(d) = (ĥ_b + γd)/√(1 + s²(d))`, the latent the tail's root and slope
    /// describe.
    fn latent(tail: &CtnPredictiveTail, d: f64) -> f64 {
        (tail.level + tail.slope * d) / tail.scale(d).sqrt()
    }

    /// A tail whose `g` increases towards `+∞` and saturates at `γ/√v = 5.657`,
    /// while its numerator turns negative towards `−∞`.
    fn saturating_tail() -> CtnPredictiveTail {
        CtnPredictiveTail {
            level: 1.2,
            slope: 0.8,
            variance: 0.3,
            cross: 0.05,
            slope_variance: 0.02,
        }
    }

    #[test]
    fn predictive_tail_root_solves_the_closed_form_latent() {
        let tail = saturating_tail();
        assert!(tail.increases_towards(1.0));
        assert!(!tail.increases_towards(-1.0));
        let limit = tail.limit(1.0);
        assert!((limit - 0.8 / 0.02_f64.sqrt()).abs() < 1e-12);
        for &z in &[1.5_f64, 3.0, 5.0] {
            let d = tail.root(z, 1.0).expect("a level below the limit is attained");
            assert!(d >= 0.0, "z={z}: root {d} left the half-line");
            assert!(
                (latent(&tail, d) - z).abs() <= 1e-12 * (1.0 + z),
                "z={z}: g({d}) = {} is not the level",
                latent(&tail, d)
            );
            let h = 1e-5;
            let central = (latent(&tail, d + h) - latent(&tail, d - h)) / (2.0 * h);
            assert!(
                (tail.latent_slope(d) - central).abs() <= 1e-6 * central.abs().max(1.0),
                "z={z}: g' = {} against the central difference {central}",
                tail.latent_slope(d)
            );
        }
        assert!(
            tail.root(6.0, 1.0).is_none(),
            "a level past the saturation limit has no crossing"
        );
    }

    #[test]
    fn predictive_tail_without_slope_variance_is_affine_and_unbounded() {
        let tail = CtnPredictiveTail {
            level: -0.4,
            slope: 1.5,
            variance: 0.25,
            cross: 0.0,
            slope_variance: 0.0,
        };
        assert!(tail.limit(1.0).is_infinite() && tail.limit(-1.0).is_infinite());
        for &(z, direction) in &[(2.0_f64, 1.0_f64), (-3.0, -1.0)] {
            let d = tail.root(z, direction).expect("an affine latent reaches every level");
            let expected = (z * 1.25_f64.sqrt() + 0.4) / 1.5;
            assert!(
                (d - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "z={z}: root {d} against {expected}"
            );
        }
    }
}
