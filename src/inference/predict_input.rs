use std::collections::HashMap;

use ndarray::Array1;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::basis::{BasisOptions, Dense, KnotSource, create_basis};
use crate::estimate::{BlockRole, PredictInput};
use crate::families::scale_design::{build_scale_deviation_operator, scale_transform_from_payload};
use crate::families::survival_predict::{
    fit_result_from_saved_model_for_prediction, resolve_termspec_for_prediction,
};
use crate::families::transformation_normal::TRANSFORMATION_MONOTONICITY_EPS;
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

            // Continuous monotonicity check (no grid sampling).
            //
            // CTN response design has columns `[1, y, dev_val(y)]` where
            // `dev_val_j(y) = Σ_k B_{k,d}(y) · resp_transform[k, j]`. So
            //   h(y, x)  = β₀(x) + y β₁(x) + Σ_k A_k(x) B_{k,d}(y)
            //   h'(y, x) = β₁(x) + Σ_k A_k(x) B'_{k,d}(y)
            // with `A_k(x) = Σ_j resp_transform[k,j] · β_{2+j}(x)`.
            //
            // Standard B-spline derivative + reindex gives
            //   h'(y, x) = β₁(x) + Σ_{k=1}^{p_basis-1} δ_k(x) N_{k,d-1}(y),
            //   δ_k(x)   = d/(t_{k+d}-t_k) · (A_k(x) - A_{k-1}(x)).
            // Because `Σ_{k=1}^{p_basis-1} N_{k,d-1}(y) = 1` on the basis
            // support [t_d, t_{p_basis}], h'(y, x) is a convex combination of
            // `{β₁(x) + δ_k(x)}`. The pointwise minimum lower bound is
            //   h'_min(x) = β₁(x) + min_k δ_k(x).
            // Outside the basis support, the B-spline first derivative
            // clamps to its boundary value, which is itself in this convex
            // combination — so the same lower bound covers extrapolation.
            // No tail-guard heuristic, no grid sampling needed.
            let kn = resp_knots
                .as_slice()
                .ok_or_else(|| "internal error: response knots are not contiguous".to_string())?;
            let p_basis = resp_transform.nrows();
            let dim_dev = resp_transform.ncols();
            let expected_knot_len = p_basis + response_degree + 1;
            if kn.len() < expected_knot_len {
                return Err(format!(
                    "saved transformation-normal knots length {} < expected {} (p_basis={}, degree={})",
                    kn.len(),
                    expected_knot_len,
                    p_basis,
                    response_degree
                ));
            }
            let monotonicity_eps = TRANSFORMATION_MONOTONICITY_EPS;
            // Parallel reduction over observation rows. The per-row state
            // (β₁(x), γ_j(x), A_k(x), running h'_lb) is fully thread-local —
            // no shared mutable state — so this is a straight rayon
            // par_iter().fold/reduce over min.
            let resp_transform_ref = &resp_transform;
            let beta_mat_ref = &beta_mat;
            let cov_mat_ref = &cov_mat;
            let kn_ref = kn;
            let min_h_prime: f64 = (0..n)
                .into_par_iter()
                .fold(
                    || {
                        (
                            f64::INFINITY,
                            vec![0.0f64; dim_dev.max(1)],
                            vec![0.0f64; p_basis.max(1)],
                        )
                    },
                    |(mut local_min, mut gamma, mut a_coef), i| {
                        let cov_row = cov_mat_ref.row(i);
                        let beta1_x = beta_mat_ref.row(1).dot(&cov_row);
                        for j in 0..dim_dev {
                            gamma[j] = beta_mat_ref.row(2 + j).dot(&cov_row);
                        }
                        for k in 0..p_basis {
                            let mut sum = 0.0;
                            for j in 0..dim_dev {
                                sum += resp_transform_ref[[k, j]] * gamma[j];
                            }
                            a_coef[k] = sum;
                        }
                        // h'_min(x) = β₁(x) + min_{k ∈ [1, p_basis-1]} δ_k(x).
                        // Knots with multiplicity (zero denominator) make
                        // N_{k,d-1} identically zero, so they contribute
                        // nothing — skip them.
                        let mut h_prime_lb = beta1_x;
                        let mut found_any = false;
                        for k in 1..p_basis {
                            let denom = kn_ref[k + response_degree] - kn_ref[k];
                            if denom <= 0.0 {
                                continue;
                            }
                            let delta_k =
                                (response_degree as f64) / denom * (a_coef[k] - a_coef[k - 1]);
                            let bound_k = beta1_x + delta_k;
                            if !found_any || bound_k < h_prime_lb {
                                h_prime_lb = bound_k;
                                found_any = true;
                            }
                        }
                        if h_prime_lb < local_min {
                            local_min = h_prime_lb;
                        }
                        (local_min, gamma, a_coef)
                    },
                )
                .map(|(local_min, _, _)| local_min)
                .reduce(|| f64::INFINITY, f64::min);
            if min_h_prime < monotonicity_eps {
                return Err(format!(
                    "prediction failed: transformation-normal fit is non-monotone in y \
                     for at least one observation. Analytic lower bound on h'(y, x) is \
                     {min_h_prime:.3e}, threshold {monotonicity_eps:.0e}. The model β \
                     admits a region where h(y, x) decreases in y, which contradicts the \
                     CTN conditional-Gaussianization contract. Refit with stronger \
                     regularization (raise the penalty seed) or with more training data; \
                     this typically arises when n is small relative to p_resp × p_cov."
                ));
            }

            // h_i = Σ_{r,c} resp_row[i,r] · cov_row[i,c] · β_mat[r,c]
            //     = resp_row[i] · (β_mat · cov_row[i])
            // The bilinear form is independent across i, so par_map_collect.
            let h_vec: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let resp_row = resp_val.row(i);
                    let cov_row = cov_mat.row(i);
                    let mut val = 0.0;
                    for r in 0..p_resp {
                        let a = resp_row[r];
                        if a == 0.0 {
                            continue;
                        }
                        for c in 0..p_cov {
                            val += a * cov_row[c] * beta_mat[[r, c]];
                        }
                    }
                    val
                })
                .collect();
            let h = ndarray::Array1::<f64>::from_vec(h_vec);
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
