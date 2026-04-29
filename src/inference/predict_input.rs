use std::collections::HashMap;

use ndarray::Array1;

use crate::basis::{BasisOptions, Dense, KnotSource, create_basis};
use crate::estimate::{BlockRole, PredictInput};
use crate::families::scale_design::{build_scale_deviation_operator, scale_transform_from_payload};
use crate::families::transformation_normal::{
    TRANSFORMATION_MONOTONICITY_EPS, TRANSFORMATION_TAIL_GUARD_FRACTION,
};
use crate::families::survival_predict::{
    fit_result_from_saved_model_for_prediction, resolve_termspec_for_prediction,
};
use crate::inference::model::{FittedModel, PredictModelClass};
use crate::matrix::DesignMatrix;
use crate::smooth::build_term_collection_design;

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
    let design = build_term_collection_design(data, &spec)
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
            let design_noise_raw = build_term_collection_design(data, &spec_noise)
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
            let &z_col = col_map
                .get(z_name)
                .ok_or_else(|| format!("prediction data is missing z column '{z_name}'"))?;
            let z = data.column(z_col).to_owned();
            let spec_logslope = resolve_termspec_for_prediction(
                &model.resolved_termspec_logslope.as_ref().cloned(),
                training_headers,
                col_map,
                "resolved_termspec_logslope",
            )?;
            let design_logslope = build_term_collection_design(data, &spec_logslope)
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
            let response_col_idx = *col_map.get(response_col_name).ok_or_else(|| {
                format!(
                    "response column '{}' not found in new data",
                    response_col_name
                )
            })?;
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
                BasisOptions::value(),
            )
            .map_err(|e| e.to_string())?;
            let raw_val = raw_val_basis.as_ref().clone();
            let dev_val = raw_val.dot(&resp_transform);
            let dev_dim = resp_transform.ncols();
            let p_resp = 2 + dev_dim;
            let mut resp_val = ndarray::Array2::<f64>::zeros((n, p_resp));
            resp_val.column_mut(0).fill(1.0);
            resp_val.column_mut(1).assign(&response_new);
            resp_val.slice_mut(ndarray::s![.., 2..]).assign(&dev_val);

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

            let mut derivative_grid = resp_knots
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let mut sorted_response_new = response_new.to_vec();
            sorted_response_new
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let prediction_quantiles = sorted_response_new.len().min(129);
            if prediction_quantiles == 1 {
                derivative_grid.push(sorted_response_new[0]);
            } else if prediction_quantiles > 1 {
                for q in 0..prediction_quantiles {
                    let idx = q * (sorted_response_new.len() - 1) / (prediction_quantiles - 1);
                    derivative_grid.push(sorted_response_new[idx]);
                }
            }
            if derivative_grid.is_empty() {
                return Err(
                    "saved transformation-normal model has no finite response knots".to_string(),
                );
            }
            derivative_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let min_grid = derivative_grid[0];
            let max_grid = derivative_grid[derivative_grid.len() - 1];
            let grid_span = (max_grid - min_grid).abs().max(1.0);
            let base_grid = derivative_grid.clone();
            for window in base_grid.windows(2) {
                let left = window[0];
                let right = window[1];
                let width = right - left;
                if width <= 1.0e-12 * grid_span {
                    continue;
                }
                for sidx in 1..4 {
                    let frac = sidx as f64 / 4.0;
                    derivative_grid.push(left + frac * width);
                }
            }
            let grid_guard = TRANSFORMATION_TAIL_GUARD_FRACTION * grid_span;
            derivative_grid.push(min_grid - grid_guard);
            derivative_grid.push(max_grid + grid_guard);
            derivative_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            derivative_grid.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12 * grid_span);
            let derivative_grid = ndarray::Array1::from_vec(derivative_grid);
            let (raw_deriv_basis, _) = create_basis::<Dense>(
                derivative_grid.view(),
                KnotSource::Provided(resp_knots.view()),
                response_degree,
                BasisOptions::first_derivative(),
            )
            .map_err(|e| e.to_string())?;
            let dev_deriv = raw_deriv_basis.as_ref().dot(&resp_transform);
            let mut resp_deriv_grid =
                ndarray::Array2::<f64>::zeros((derivative_grid.len(), p_resp));
            resp_deriv_grid.column_mut(1).fill(1.0);
            resp_deriv_grid
                .slice_mut(ndarray::s![.., 2..])
                .assign(&dev_deriv);
            let (raw_obs_deriv_basis, _) = create_basis::<Dense>(
                response_new.view(),
                KnotSource::Provided(resp_knots.view()),
                response_degree,
                BasisOptions::first_derivative(),
            )
            .map_err(|e| e.to_string())?;
            let dev_obs_deriv = raw_obs_deriv_basis.as_ref().dot(&resp_transform);
            let mut resp_deriv_obs = ndarray::Array2::<f64>::zeros((n, p_resp));
            resp_deriv_obs.column_mut(1).fill(1.0);
            resp_deriv_obs
                .slice_mut(ndarray::s![.., 2..])
                .assign(&dev_obs_deriv);

            let monotonicity_eps = TRANSFORMATION_MONOTONICITY_EPS;
            let mut min_h_prime = f64::INFINITY;
            for i in 0..n {
                let cov_row = cov_mat.row(i);
                for g in 0..resp_deriv_grid.nrows() {
                    let resp_deriv_row = resp_deriv_grid.row(g);
                    let mut h_prime = 0.0;
                    for r in 0..p_resp {
                        if resp_deriv_row[r] == 0.0 {
                            continue;
                        }
                        for c in 0..p_cov {
                            h_prime += resp_deriv_row[r] * cov_row[c] * beta_mat[[r, c]];
                        }
                    }
                    min_h_prime = min_h_prime.min(h_prime);
                }
                let resp_deriv_row = resp_deriv_obs.row(i);
                let mut h_prime = 0.0;
                for r in 0..p_resp {
                    if resp_deriv_row[r] == 0.0 {
                        continue;
                    }
                    for c in 0..p_cov {
                        h_prime += resp_deriv_row[r] * cov_row[c] * beta_mat[[r, c]];
                    }
                }
                min_h_prime = min_h_prime.min(h_prime);
            }
            if min_h_prime < monotonicity_eps {
                return Err(format!(
                    "transformation-normal prediction violates monotonicity on the response grid for new data: min h'={min_h_prime:.6e}"
                ));
            }

            let mut h = ndarray::Array1::<f64>::zeros(n);
            for i in 0..n {
                let resp_row = resp_val.row(i);
                let cov_row = cov_mat.row(i);
                let mut val = 0.0;
                for r in 0..p_resp {
                    if resp_row[r] == 0.0 {
                        continue;
                    }
                    for c in 0..p_cov {
                        val += resp_row[r] * cov_row[c] * beta_mat[[r, c]];
                    }
                }
                h[i] = val;
            }
            Ok(PredictInput {
                design: DesignMatrix::from(ndarray::Array2::from_shape_fn((n, 1), |_| 1.0)),
                offset: h + offset,
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
            })
        }
    }
}
