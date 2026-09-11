use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::inference::predict_io::PredictInput;
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
    TRANSFORMATION_MONOTONICITY_EPS, ctn_endpoint_bases, ctn_response_bases_at, ctn_row_geometry,
    transformation_normal_pit_score,
};
use gam_problem::BlockRole;
use gam_terms::smooth::build_term_collection_design;

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
    design: &gam_terms::smooth::TermCollectionDesign,
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
    design: &gam_terms::smooth::TermCollectionDesign,
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
/// response-scale predictive quantile ladder `h⁻¹(z_j | x_i)`.
pub const TRANSFORMATION_NORMAL_BAND_Z_NODES: usize = 65;

/// Half-width of the latent-z ladder. `Φ(4) ≈ 0.999968`, so every two-sided
/// observation level up to ≈ 0.99993 interpolates strictly inside the ladder;
/// beyond it the band clamps to the outermost tabulated quantile.
pub const TRANSFORMATION_NORMAL_BAND_Z_MAX: f64 = 4.0;

/// The fixed, evenly spaced latent-z ladder shared by the CTM input builder
/// (which tabulates `h⁻¹` on it) and the transformation-normal predictor
/// (which interpolates it to build response-scale observation bands). The CTM
/// predictive is `Y | x = h⁻¹(Z | x)` with `Z ~ N(0,1)`, so the response-scale
/// `p`-quantile is exactly `h⁻¹(Φ⁻¹(p) | x)` — quantiles map through the
/// monotone inverse transform; they are NOT `E[Y|x] ± z·σ` in latent-normal
/// units.
pub fn transformation_normal_band_z_nodes() -> Array1<f64> {
    Array1::from_shape_fn(TRANSFORMATION_NORMAL_BAND_Z_NODES, |j| {
        -TRANSFORMATION_NORMAL_BAND_Z_MAX
            + 2.0 * TRANSFORMATION_NORMAL_BAND_Z_MAX * (j as f64)
                / ((TRANSFORMATION_NORMAL_BAND_Z_NODES - 1) as f64)
    })
}

/// The response-scale conditional mean `E[Y|x] = E_{Z~N(0,1)}[h⁻¹(Z|x)]` for
/// each row of a CTM transform table, by averaging the inverse over a
/// standard-normal midpoint quadrature in probability space (see the predict
/// branch for the derivation). Used by BOTH the predict mean (#1612) and the
/// generate sampler's reference mean (#1613), so they agree by construction.
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
    const QUAD: usize = TRANSFORMATION_NORMAL_MEAN_QUADRATURE;
    let z_nodes: Vec<f64> = (0..QUAD)
        .map(|k| {
            let p = ((k as f64) + 0.5) / (QUAD as f64);
            standard_normal_quantile(p)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PredictInputError::InvalidInput { reason: e })?;
    let mean = Array1::<f64>::from_shape_fn(n, |i| {
        let mut acc = 0.0_f64;
        for &z in &z_nodes {
            acc += table.invert(i, z);
        }
        acc / (QUAD as f64)
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
    /// Response-scale conditional mean `E[Y|x_i]` — the same value `predict`
    /// returns (#1612), provided so the generate spec's reference mean and the
    /// prediction mean cannot diverge.
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
    let design = build_term_collection_design(design_input, &spec)
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
    let design = build_term_collection_design(design_input, &spec)
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
            let design_noise_raw = build_term_collection_design(design_input, &spec_noise)
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
            let design_slope = build_term_collection_design(design_input, &spec_slope)
                .map_err(|e| PredictInputError::InvalidInput {
                    reason: format!("failed to build slope prediction design: {e}"),
                })?;
            let mean_offset = design
                .compose_offset(offset.view(), "marginal-slope mean prediction")
                .map_err(|error| PredictInputError::InvalidInput {
                    reason: error.to_string(),
                })?;
            let slope_offset = design_slope
                .compose_offset(offset_noise.view(), "marginal-slope slope prediction")
                .map_err(|error| PredictInputError::InvalidInput {
                    reason: error.to_string(),
                })?;
            let auxiliary_matrix =
                build_marginal_slope_local_auxiliary_matrix(model, design_input, col_map)?;
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
            // Response-scale predictive quantile ladder: `Y|x = h⁻¹(Z|x)` with
            // `Z ~ N(0,1)`, so the p-quantile of `Y|x` is `h⁻¹(Φ⁻¹(p)|x)`.
            // Tabulating `h⁻¹` on the fixed z ladder lets the predictor build
            // genuine response-scale observation bands by interpolating this
            // matrix — instead of adding standard-normal quantiles to `E[Y|x]`
            // in latent-normal units, which is wrong by exactly the (unknown to
            // the predictor) scale of `h⁻¹`.
            let z_nodes = transformation_normal_band_z_nodes();
            let quantile_ladder =
                Array2::from_shape_fn((n, z_nodes.len()), |(i, j)| table.invert(i, z_nodes[j]));
            // The predictor passes the offset through unchanged as `eta` and
            // `mean`, so storing E[Y|x] here yields a y-independent response-scale
            // prediction for both columns on a covariate-only frame.
            Ok(PredictInput {
                design: DesignMatrix::from(ndarray::Array2::from_shape_fn((n, 1), |_| 1.0)),
                offset: conditional_mean,
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
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
