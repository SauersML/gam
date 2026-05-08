use std::collections::HashMap;

use ndarray::Array1;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense,
};
use crate::estimate::{BlockRole, PredictInput};
use crate::families::scale_design::{build_scale_deviation_operator, scale_transform_from_payload};
use crate::families::survival_predict::{
    fit_result_from_saved_model_for_prediction, resolve_termspec_for_prediction,
};
use crate::families::transformation_normal::{
    TRANSFORMATION_MONOTONICITY_EPS, TRANSFORMATION_NORMAL_H_ABS_MAX,
    transformation_normal_pit_score,
};
use crate::inference::model::{FittedModel, PredictModelClass};
use crate::matrix::DesignMatrix;
use crate::smooth::build_term_collection_design;
use crate::term_builder::resolve_role_col;

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
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let design_input = clipped.as_ref().map_or(data, |arr| arr.view());
    let design = build_term_collection_design(design_input, &spec)
        .map_err(|e| format!("failed to build prediction design: {e}"))?;
    let n = data.nrows();
    if offset.len() != n || offset_noise.len() != n {
        return Err(format!(
            "prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
            offset.len(),
            offset_noise.len()
        ));
    }

    match model.predict_model_class() {
        PredictModelClass::Standard => {
            if noise_offset_supplied {
                return Err(
                    "--noise-offset-column is not supported for standard prediction".to_string(),
                );
            }
            let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
            let beta = if model.has_link_wiggle() {
                fit_saved
                    .block_by_role(BlockRole::Mean)
                    .ok_or_else(|| {
                        "standard link-wiggle model is missing Mean coefficient block".to_string()
                    })?
                    .beta
                    .clone()
            } else {
                fit_saved.beta.clone()
            };
            if beta.len() != design.design.ncols() {
                return Err(format!(
                    "model/design mismatch: model beta has {} coefficients but new-data design has {} columns",
                    beta.len(),
                    design.design.ncols()
                ));
            }
            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
                auxiliary_matrix: None,
            })
        }
        PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale => {
            let spec_noise = resolve_termspec_for_prediction(
                &model.resolved_termspec_noise,
                training_headers,
                col_map,
                "resolved_termspec_noise",
            )?;
            let design_noise_raw = build_term_collection_design(design_input, &spec_noise)
                .map_err(|e| format!("failed to build noise prediction design: {e}"))?;

            let noise_transform = scale_transform_from_payload(
                &model.noise_projection,
                &model.noise_center,
                &model.noise_scale,
                model.noise_non_intercept_start,
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
            let z_name = model
                .z_column
                .as_ref()
                .ok_or_else(|| "marginal-slope model is missing z_column".to_string())?;
            let z_col = resolve_role_col(col_map, z_name, "z")?;
            let z = data.column(z_col).to_owned();
            let spec_logslope = resolve_termspec_for_prediction(
                &model.resolved_termspec_logslope.as_ref().cloned(),
                training_headers,
                col_map,
                "resolved_termspec_logslope",
            )?;
            let design_logslope = build_term_collection_design(design_input, &spec_logslope)
                .map_err(|e| format!("failed to build logslope prediction design: {e}"))?;
            // build_marginal_slope_local_auxiliary_matrix is referenced by an
            // in-flight concurrent integration (LocalEmpirical conditioning
            // columns); until that lands, fall back to no auxiliary matrix
            // so the rest of the predict-input plumbing still compiles.
            let _ = (model, design_input, col_map);
            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: Some(design_logslope.design.clone()),
                offset_noise: Some(offset_noise.clone()),
                auxiliary_scalar: Some(z),
                auxiliary_matrix: None,
            })
        }
        PredictModelClass::Survival => Err(
            "build_predict_input_for_model should not be called for survival models".to_string(),
        ),
        PredictModelClass::TransformationNormal => {
            if noise_offset_supplied {
                return Err(
                    "--noise-offset-column is not supported for transformation-normal prediction"
                        .to_string(),
                );
            }
            let payload = model.payload();
            let response_knots = payload
                .transformation_response_knots
                .as_ref()
                .ok_or("saved transformation-normal model missing response_knots")?;
            let response_transform_vecs = payload
                .transformation_response_transform
                .as_ref()
                .ok_or("saved transformation-normal model missing response_transform")?;
            let response_degree = payload
                .transformation_response_degree
                .ok_or("saved transformation-normal model missing response_degree")?;
            let response_median = payload
                .transformation_response_median
                .ok_or("saved transformation-normal model missing response_median")?;

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

            let response_col_name = payload
                .formula
                .split('~')
                .next()
                .map(str::trim)
                .ok_or("cannot parse response column from formula")?;
            let response_col_idx = resolve_role_col(col_map, response_col_name, "response")?;
            let response_new = data.column(response_col_idx).to_owned();
            for value in response_new.iter().copied() {
                if !value.is_finite() {
                    return Err(format!(
                        "transformation-normal response value in prediction data is not finite: {value}"
                    ));
                }
            }

            let (raw_val_basis, _) = create_basis::<Dense>(
                response_new.view(),
                KnotSource::Provided(resp_knots.view()),
                response_degree,
                BasisOptions::i_spline(),
            )
            .map_err(|e| e.to_string())?;
            let raw_val = raw_val_basis.as_ref().clone();
            if raw_val.ncols() != resp_transform.nrows() {
                return Err(format!(
                    "saved transformation-normal response transform shape mismatch: raw I-spline cols={} transform rows={}",
                    raw_val.ncols(),
                    resp_transform.nrows()
                ));
            }
            let shape_val = raw_val.dot(&resp_transform);
            let p_shape = resp_transform.ncols();
            let p_resp = 1 + p_shape;
            let mut resp_val = ndarray::Array2::<f64>::zeros((n, p_resp));
            resp_val.column_mut(0).fill(1.0);
            resp_val.slice_mut(ndarray::s![.., 1..]).assign(&shape_val);

            let raw_deriv = create_ispline_derivative_dense(
                response_new.view(),
                &resp_knots,
                response_degree,
                1,
            )
            .map_err(|e| e.to_string())?;
            if raw_deriv.ncols() != resp_transform.nrows() {
                return Err(format!(
                    "saved transformation-normal derivative transform shape mismatch: raw M-spline cols={} transform rows={}",
                    raw_deriv.ncols(),
                    resp_transform.nrows()
                ));
            }
            let shape_deriv = raw_deriv.dot(&resp_transform);
            let mut resp_deriv = ndarray::Array2::<f64>::zeros((n, p_resp));
            resp_deriv
                .slice_mut(ndarray::s![.., 1..])
                .assign(&shape_deriv);

            let fit_saved = model
                .unified()
                .ok_or("saved transformation-normal model missing unified fit")?;
            let beta = &fit_saved.blocks[0].beta;
            let p_cov = design.design.ncols();
            if beta.len() != p_resp * p_cov {
                return Err(format!(
                    "beta length {} != p_resp({}) * p_cov({})",
                    beta.len(),
                    p_resp,
                    p_cov
                ));
            }
            let beta_mat = beta
                .view()
                .into_shape_with_order((p_resp, p_cov))
                .map_err(|e| format!("beta reshape failed: {e}"))?;
            let cov_mat = design
                .design
                .try_row_chunk(0..n)
                .map_err(|e| e.to_string())?;
            let calibration = payload
                .transformation_score_calibration
                .as_ref()
                .ok_or("saved transformation-normal model missing score calibration")?;
            calibration.validate("saved transformation-normal score calibration")?;

            if resp_knots.is_empty() {
                return Err("saved transformation-normal response knots are empty".to_string());
            }
            let mut response_lower_basis = vec![0.0; p_resp];
            let mut response_upper_basis = vec![0.0; p_resp];
            response_lower_basis[0] = 1.0;
            response_upper_basis[0] = 1.0;
            for col in 0..p_shape {
                response_upper_basis[col + 1] = resp_transform.column(col).sum();
            }
            let response_lower_floor_offset =
                TRANSFORMATION_MONOTONICITY_EPS * (resp_knots[0] - response_median);
            let response_upper_floor_offset = TRANSFORMATION_MONOTONICITY_EPS
                * (resp_knots[resp_knots.len() - 1] - response_median);

            // Under SCOP-CTN with I-spline shape components,
            // `h'(y, x) = ε + Σ_{r≥1} M_r(y) · γ_r(x)²`. Both M_r and γ_r²
            // are non-negative for every (β, x, y), and ε is the fixed
            // derivative floor serialized through the model definition.
            let monotonicity_eps = TRANSFORMATION_MONOTONICITY_EPS;
            let beta_mat_ref = &beta_mat;
            let cov_mat_ref = &cov_mat;
            let resp_deriv_ref = &resp_deriv;
            let min_h_prime: f64 = (0..n)
                .into_par_iter()
                .map(|i| {
                    let cov_row = cov_mat_ref.row(i);
                    let resp_row = resp_deriv_ref.row(i);
                    let mut hp = resp_row[0] * beta_mat_ref.row(0).dot(&cov_row);
                    for r in 1..p_resp {
                        let gamma = beta_mat_ref.row(r).dot(&cov_row);
                        hp += resp_row[r] * gamma * gamma;
                    }
                    hp + monotonicity_eps
                })
                .reduce(|| f64::INFINITY, f64::min);
            if min_h_prime < monotonicity_eps {
                return Err(format!(
                    "prediction failed: transformation-normal h'(y, x) numerical floor \
                     violated. Minimum evaluated h'(y, x) is {min_h_prime:.3e}, threshold \
                     {monotonicity_eps:.0e}. Under SCOP h' = ε + Σ M_r γ_r² holds \
                     structurally, so this indicates floating-point cancellation below \
                     the fixed derivative floor."
                ));
            }

            // h_i and finite-support endpoints share the same γ_r(x_i). The
            // prediction score is the fitted PIT, not a post-h location/scale
            // normalization.
            let pit_vec: Vec<Result<f64, String>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let resp_row = resp_val.row(i);
                    let cov_row = cov_mat.row(i);
                    let gamma0 = beta_mat.row(0).dot(&cov_row);
                    let mut val = resp_row[0] * gamma0;
                    let mut lower = response_lower_basis[0] * gamma0;
                    let mut upper = response_upper_basis[0] * gamma0;
                    let mut max_abs_gamma = gamma0.abs();
                    for r in 1..p_resp {
                        let gamma = beta_mat.row(r).dot(&cov_row);
                        max_abs_gamma = max_abs_gamma.max(gamma.abs());
                        val += resp_row[r] * gamma * gamma;
                        lower += response_lower_basis[r] * gamma * gamma;
                        upper += response_upper_basis[r] * gamma * gamma;
                    }
                    let h = val
                        + offset[i]
                        + monotonicity_eps * (response_new[i] - response_median);
                    let h_lower = lower + offset[i] + response_lower_floor_offset;
                    let h_upper = upper + offset[i] + response_upper_floor_offset;
                    if !h.is_finite() || !h_lower.is_finite() || !h_upper.is_finite() {
                        let max_abs_cov = cov_row.iter().copied().map(f64::abs).fold(0.0, f64::max);
                        return Err(format!(
                            "prediction failed: transformation-normal finite-support scores at row {i} are not finite: h={h:.6e}, lower={h_lower:.6e}, upper={h_upper:.6e}; max_abs_covariate_basis={max_abs_cov:.6e}, max_abs_gamma={max_abs_gamma:.6e}"
                        ));
                    }
                    transformation_normal_pit_score(h, h_lower, h_upper, calibration.clip_eps)
                        .map_err(|err| format!("prediction failed at row {i}: {err}"))
                })
                .collect();
            let calibrated = ndarray::Array1::<f64>::from_vec(
                pit_vec.into_iter().collect::<Result<Vec<_>, _>>()?,
            );
            if calibrated
                .iter()
                .any(|value| !value.is_finite() || value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX)
            {
                return Err(
                    "prediction failed: transformation-normal PIT produced non-finite or out-of-range z values"
                        .to_string(),
                );
            }
            Ok(PredictInput {
                design: DesignMatrix::from(ndarray::Array2::from_shape_fn((n, 1), |_| 1.0)),
                offset: calibrated,
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
                auxiliary_matrix: None,
            })
        }
    }
}
