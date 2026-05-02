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
            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: Some(design_logslope.design.clone()),
                offset_noise: Some(offset_noise.clone()),
                auxiliary_scalar: Some(z),
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
            let d = calibration.feature_cols.len();
            if calibration.feature_center.len() != d || calibration.feature_scale.len() != d {
                return Err(format!(
                    "saved transformation-normal calibration normalization mismatch: feature_cols={}, center={}, scale={}",
                    d,
                    calibration.feature_center.len(),
                    calibration.feature_scale.len()
                ));
            }
            for center in &calibration.rbf_centers {
                if center.len() != d {
                    return Err(format!(
                        "saved transformation-normal RBF center width {} != feature width {d}",
                        center.len()
                    ));
                }
            }
            if !(calibration.global_sd.is_finite() && calibration.global_sd > 1.0e-12)
                || !calibration.global_mean.is_finite()
                || !(calibration.rbf_bandwidth.is_finite() && calibration.rbf_bandwidth > 0.0)
            {
                return Err(format!(
                    "saved transformation-normal score calibration has invalid global mean/sd/bandwidth: mean={}, sd={}, bandwidth={}",
                    calibration.global_mean, calibration.global_sd, calibration.rbf_bandwidth
                ));
            }
            let p_poly = 1 + d + d * (d + 1) / 2;
            let p_calib = p_poly + calibration.rbf_centers.len();
            if calibration.location_beta.len() != p_calib
                || calibration.log_scale_beta.len() != p_calib
            {
                return Err(format!(
                    "saved transformation-normal calibration/design mismatch: location_beta={}, log_scale_beta={}, p_calib={p_calib}",
                    calibration.location_beta.len(),
                    calibration.log_scale_beta.len()
                ));
            }
            let mut calib_mat = ndarray::Array2::<f64>::zeros((n, p_calib));
            for i in 0..n {
                calib_mat[[i, 0]] = 1.0;
                let mut z = vec![0.0; d];
                for (j, &col) in calibration.feature_cols.iter().enumerate() {
                    if col >= data.ncols() {
                        return Err(format!(
                            "saved transformation-normal calibration feature column {col} out of range for {} columns",
                            data.ncols()
                        ));
                    }
                    z[j] = (design_input[[i, col]] - calibration.feature_center[j])
                        / calibration.feature_scale[j];
                    calib_mat[[i, 1 + j]] = z[j];
                }
                let mut col_out = 1 + d;
                for a in 0..d {
                    for b in a..d {
                        calib_mat[[i, col_out]] = z[a] * z[b];
                        col_out += 1;
                    }
                }
                for (m, center) in calibration.rbf_centers.iter().enumerate() {
                    let dist2 = (0..d)
                        .map(|j| {
                            let r = z[j] - center[j];
                            r * r
                        })
                        .sum::<f64>();
                    calib_mat[[i, p_poly + m]] = (-0.5 * dist2
                        / (calibration.rbf_bandwidth * calibration.rbf_bandwidth))
                        .exp();
                }
            }
            let location_beta = ndarray::Array1::from_vec(calibration.location_beta.clone());
            let log_scale_beta = ndarray::Array1::from_vec(calibration.log_scale_beta.clone());
            let score_location = calib_mat.dot(&location_beta);
            let score_log_scale = calib_mat.dot(&log_scale_beta);

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

            // h_i = b(x_i) + Σ_{r>=1} resp_row[i,r] · γ_r(x_i)^2.
            // The bilinear form is independent across i, so par_map_collect.
            let h_vec: Vec<Result<f64, String>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let resp_row = resp_val.row(i);
                    let cov_row = cov_mat.row(i);
                    let mut val = resp_row[0] * beta_mat.row(0).dot(&cov_row);
                    let mut max_abs_gamma = beta_mat.row(0).dot(&cov_row).abs();
                    for r in 1..p_resp {
                        let gamma = beta_mat.row(r).dot(&cov_row);
                        max_abs_gamma = max_abs_gamma.max(gamma.abs());
                        val += resp_row[r] * gamma * gamma;
                    }
                    if !val.is_finite() || val.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX {
                        let max_abs_cov = cov_row.iter().copied().map(f64::abs).fold(0.0, f64::max);
                        return Err(format!(
                            "prediction failed: transformation-normal h at row {i} is {val:.6e}, outside the standard-normal bound ±{TRANSFORMATION_NORMAL_H_ABS_MAX}; max_abs_covariate_basis={max_abs_cov:.6e}, max_abs_gamma={max_abs_gamma:.6e}"
                        ));
                    }
                    Ok(val + monotonicity_eps * (response_new[i] - response_median))
                })
                .collect();
            let h =
                ndarray::Array1::<f64>::from_vec(h_vec.into_iter().collect::<Result<Vec<_>, _>>()?);
            let h_with_offset = h + offset;
            let calibrated = ndarray::Array1::from_iter(
                h_with_offset
                    .iter()
                    .zip(score_location.iter())
                    .zip(score_log_scale.iter())
                    .map(|((&raw, &location), &log_scale)| {
                        ((raw - location) / log_scale.exp() - calibration.global_mean)
                            / calibration.global_sd
                    }),
            );
            if calibrated
                .iter()
                .any(|value| !value.is_finite() || value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX)
            {
                return Err(
                    "prediction failed: transformation-normal score calibration produced non-finite or out-of-range z values"
                        .to_string(),
                );
            }
            Ok(PredictInput {
                design: DesignMatrix::from(ndarray::Array2::from_shape_fn((n, 1), |_| 1.0)),
                offset: calibrated,
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
            })
        }
    }
}
