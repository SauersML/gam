use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, fast_atb, fast_av};
use crate::linalg::utils::StableSolver;
use crate::matrix::{
    BlockDesignOperator, DesignBlock, DesignMatrix, LinearOperator, ReparamDesignOperator,
};
use faer::prelude::SolveLstsq;
use ndarray::{Array1, Array2, ArrayView1, s};
use std::sync::Arc;

const COLUMN_TOL: f64 = 1e-12;
const SCALE_DESIGN_TARGET_CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct ScaleDeviationTransform {
    pub projection_coef: Array2<f64>,
    pub weighted_column_mean: Array1<f64>,
    pub rescale: Array1<f64>,
    pub non_intercept_start: usize,
}

fn weighted_mean(col: ArrayView1<'_, f64>, weights: &Array1<f64>) -> Result<f64, String> {
    if col.len() != weights.len() {
        return Err("weighted mean dimension mismatch".to_string());
    }
    let w_sum: f64 = weights.iter().copied().sum();
    if !w_sum.is_finite() || w_sum <= 0.0 {
        return Err("weighted mean requires positive finite total weight".to_string());
    }
    Ok(col
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| x * w)
        .sum::<f64>()
        / w_sum)
}

fn weighted_centered_ss(col: ArrayView1<'_, f64>, weights: &Array1<f64>) -> Result<f64, String> {
    let mean = weighted_mean(col, weights)?;
    Ok(col
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| {
            let dx = x - mean;
            w * dx * dx
        })
        .sum())
}

pub fn infer_non_intercept_start(design: &Array2<f64>, weights: &Array1<f64>) -> usize {
    let mut end = 0;
    for j in 0..design.ncols() {
        let ss = weighted_centered_ss(design.column(j), weights).unwrap_or(0.0);
        if ss <= COLUMN_TOL {
            end = j + 1;
        } else {
            break;
        }
    }
    end
}

pub fn build_scale_deviation_transform(
    primary_design: &Array2<f64>,
    noise_design: &Array2<f64>,
    weights: &Array1<f64>,
    non_intercept_start: usize,
) -> Result<ScaleDeviationTransform, String> {
    if primary_design.nrows() != noise_design.nrows() || weights.len() != noise_design.nrows() {
        return Err("scale deviation transform row mismatch".to_string());
    }
    let p_primary = primary_design.ncols();
    let p_noise = noise_design.ncols();
    let mut projection_coef = Array2::<f64>::zeros((p_primary, p_noise));
    let mut weighted_column_mean = Array1::<f64>::zeros(p_noise);
    let mut rescale = Array1::<f64>::ones(p_noise);
    let first_active = non_intercept_start.min(p_noise);
    if first_active < p_noise {
        let n = primary_design.nrows();
        let active_cols = p_noise - first_active;
        let sqrtw = weights.mapv(f64::sqrt);
        let mut wx = Array2::<f64>::zeros((n, p_primary));
        for i in 0..n {
            let sw = sqrtw[i];
            for j in 0..p_primary {
                wx[[i, j]] = sw * primary_design[[i, j]];
            }
        }
        let wx_faer = FaerArrayView::new(&wx);
        let qr = wx_faer.as_ref().col_piv_qr();
        let chunk_cols = (SCALE_DESIGN_TARGET_CHUNK_BYTES
            / (n.max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(active_cols);
        let mut rhs = Array2::<f64>::zeros((n, chunk_cols));
        for chunk_start in (0..active_cols).step_by(chunk_cols) {
            let width = (active_cols - chunk_start).min(chunk_cols);
            rhs.fill(0.0);
            for i in 0..n {
                let sw = sqrtw[i];
                for j in 0..width {
                    rhs[[i, j]] = sw * noise_design[[i, first_active + chunk_start + j]];
                }
            }
            let mut rhs_mat = crate::faer_ndarray::array2_to_matmut(&mut rhs);
            qr.solve_lstsq_in_place(rhs_mat.as_mut());
            projection_coef
                .slice_mut(s![
                    ..,
                    first_active + chunk_start..first_active + chunk_start + width
                ])
                .assign(&rhs.slice(s![..p_primary, ..width]));
        }
    }
    for j in first_active..p_noise {
        let col = noise_design.column(j);
        let fitted = fast_av(primary_design, &projection_coef.column(j));
        let mut residual = &col - &fitted;
        let center = weighted_mean(residual.view(), weights)?;
        residual.mapv_inplace(|v| v - center);
        let orig_ss = weighted_centered_ss(col.view(), weights)?;
        let resid_ss = weighted_centered_ss(residual.view(), weights)?;
        let scale = if resid_ss.is_finite()
            && resid_ss > COLUMN_TOL
            && orig_ss.is_finite()
            && orig_ss > COLUMN_TOL
        {
            (orig_ss / resid_ss).sqrt()
        } else {
            1.0
        };
        weighted_column_mean[j] = center;
        rescale[j] = scale;
    }
    Ok(ScaleDeviationTransform {
        projection_coef,
        weighted_column_mean,
        rescale,
        non_intercept_start,
    })
}

pub fn apply_scale_deviation_transform(
    primary_design: &Array2<f64>,
    rawnoise_design: &Array2<f64>,
    transform: &ScaleDeviationTransform,
) -> Result<Array2<f64>, String> {
    if primary_design.nrows() != rawnoise_design.nrows() {
        return Err("scale deviation apply row mismatch".to_string());
    }
    if primary_design.ncols() != transform.projection_coef.nrows()
        || rawnoise_design.ncols() != transform.projection_coef.ncols()
    {
        return Err("scale deviation apply column mismatch".to_string());
    }
    let mut out = rawnoise_design.clone();
    for j in transform.non_intercept_start.min(out.ncols())..out.ncols() {
        let projected = fast_av(primary_design, &transform.projection_coef.column(j));
        for i in 0..out.nrows() {
            out[[i, j]] = (out[[i, j]] - projected[i] - transform.weighted_column_mean[j])
                * transform.rescale[j];
        }
    }
    Ok(out)
}

#[derive(Debug)]
struct WeightedColumnStats {
    weighted_sum: Array1<f64>,
    weighted_sum_sq: Array1<f64>,
    total_weight: f64,
}

fn validate_scale_weights(weights: &Array1<f64>) -> Result<f64, String> {
    let mut total_weight = 0.0;
    for (idx, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(format!("scale deviation weight {idx} is not finite"));
        }
        if w < 0.0 {
            return Err(format!(
                "scale deviation requires non-negative weights, got {w} at index {idx}"
            ));
        }
        total_weight += w;
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err("scale deviation requires positive finite total weight".to_string());
    }
    Ok(total_weight)
}

fn weighted_column_stats_design(
    design: &DesignMatrix,
    weights: &Array1<f64>,
) -> Result<WeightedColumnStats, String> {
    if design.nrows() != weights.len() {
        return Err(format!(
            "weighted column stats row mismatch: design has {} rows, weights have {} entries",
            design.nrows(),
            weights.len()
        ));
    }
    let total_weight = validate_scale_weights(weights)?;
    let p = design.ncols();
    let mut weighted_sum = Array1::<f64>::zeros(p);
    let mut weighted_sum_sq = Array1::<f64>::zeros(p);
    let chunk_rows = (SCALE_DESIGN_TARGET_CHUNK_BYTES / (p.max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(design.nrows().max(1));
    for start in (0..design.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(design.nrows());
        let chunk = design.row_chunk(start..end);
        for local in 0..(end - start) {
            let w = weights[start + local];
            if w == 0.0 {
                continue;
            }
            for j in 0..p {
                let x = chunk[[local, j]];
                weighted_sum[j] += w * x;
                weighted_sum_sq[j] += w * x * x;
            }
        }
    }
    Ok(WeightedColumnStats {
        weighted_sum,
        weighted_sum_sq,
        total_weight,
    })
}

fn weighted_crossprod_designs(
    left: &DesignMatrix,
    right: &DesignMatrix,
    weights: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != right.nrows() || left.nrows() != weights.len() {
        return Err(format!(
            "weighted crossprod row mismatch: left rows={}, right rows={}, weights={}",
            left.nrows(),
            right.nrows(),
            weights.len()
        ));
    }
    validate_scale_weights(weights)?;
    let p_left = left.ncols();
    let p_right = right.ncols();
    let bytes_per_row = (p_left + p_right).max(1) * std::mem::size_of::<f64>();
    let chunk_rows = (SCALE_DESIGN_TARGET_CHUNK_BYTES / bytes_per_row)
        .max(1)
        .min(left.nrows().max(1));
    let mut out = Array2::<f64>::zeros((p_left, p_right));
    for start in (0..left.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(left.nrows());
        let left_chunk = left.row_chunk(start..end);
        let right_chunk = right.row_chunk(start..end);
        let mut weighted_right = right_chunk.clone();
        for local in 0..(end - start) {
            let w = weights[start + local];
            if w == 0.0 {
                weighted_right.row_mut(local).fill(0.0);
                continue;
            }
            for j in 0..p_right {
                weighted_right[[local, j]] *= w;
            }
        }
        out += &fast_atb(&left_chunk, &weighted_right);
    }
    Ok(out)
}

fn solve_projection_system(system: &Array2<f64>, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
    if system.nrows() != system.ncols() || system.nrows() != rhs.nrows() {
        return Err(format!(
            "scale deviation projection solve mismatch: system is {}x{}, rhs is {}x{}",
            system.nrows(),
            system.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    let solver = StableSolver::new("scale deviation projection");
    if let Ok(factor) = solver.factorize(system) {
        let mut out = rhs.clone();
        let mut rhs_mat = array2_to_matmut(&mut out);
        factor.solve_in_place(rhs_mat.as_mut());
        if out.iter().all(|v| v.is_finite()) {
            return Ok(out);
        }
    }
    let inverse = solver
        .inversewith_regularization(system)
        .ok_or_else(|| "scale deviation projection solve failed".to_string())?;
    Ok(inverse.dot(rhs))
}

pub fn infer_non_intercept_start_design(
    design: &DesignMatrix,
    weights: &Array1<f64>,
) -> Result<usize, String> {
    let stats = weighted_column_stats_design(design, weights)?;
    let mut end = 0;
    for j in 0..design.ncols() {
        let centered_ss = stats.weighted_sum_sq[j]
            - stats.weighted_sum[j] * stats.weighted_sum[j] / stats.total_weight;
        if centered_ss <= COLUMN_TOL {
            end = j + 1;
        } else {
            break;
        }
    }
    Ok(end)
}

pub fn build_scale_deviation_transform_design(
    primary_design: &DesignMatrix,
    noise_design: &DesignMatrix,
    weights: &Array1<f64>,
    non_intercept_start: usize,
) -> Result<ScaleDeviationTransform, String> {
    if primary_design.nrows() != noise_design.nrows() || noise_design.nrows() != weights.len() {
        return Err(format!(
            "scale deviation transform row mismatch: primary rows={}, noise rows={}, weights={}",
            primary_design.nrows(),
            noise_design.nrows(),
            weights.len()
        ));
    }
    let noise_stats = weighted_column_stats_design(noise_design, weights)?;
    let w_sum = noise_stats.total_weight;
    let p_primary = primary_design.ncols();
    let p_noise = noise_design.ncols();
    let mut projection_coef = Array2::<f64>::zeros((p_primary, p_noise));
    let mut weighted_column_mean = Array1::<f64>::zeros(p_noise);
    let mut rescale = Array1::<f64>::ones(p_noise);
    let first_active = non_intercept_start.min(p_noise);
    if first_active >= p_noise {
        return Ok(ScaleDeviationTransform {
            projection_coef,
            weighted_column_mean,
            rescale,
            non_intercept_start,
        });
    }

    let xtwx = primary_design.compute_xtwx(weights)?;
    let xtwn = weighted_crossprod_designs(primary_design, noise_design, weights)?;
    let active_cross = xtwn.slice(s![.., first_active..]).to_owned();
    let active_coef = solve_projection_system(&xtwx, &active_cross)?;
    projection_coef
        .slice_mut(s![.., first_active..])
        .assign(&active_coef);

    let xtw1 = primary_design.apply_transpose(weights);
    for j in first_active..p_noise {
        let coef = projection_coef.column(j).to_owned();
        let fitted_weighted_sum = xtw1.dot(&coef);
        let residual_weighted_sum = noise_stats.weighted_sum[j] - fitted_weighted_sum;
        let centered_residual_mean = residual_weighted_sum / w_sum;

        let xtwx_coef = xtwx.dot(&coef);
        let fitted_weighted_ss = coef.dot(&xtwx_coef);
        let raw_cross_fitted = coef.dot(&xtwn.column(j));
        let residual_weighted_ss =
            noise_stats.weighted_sum_sq[j] - 2.0 * raw_cross_fitted + fitted_weighted_ss;

        let orig_ss = noise_stats.weighted_sum_sq[j]
            - noise_stats.weighted_sum[j] * noise_stats.weighted_sum[j] / w_sum;
        let resid_ss = residual_weighted_ss - residual_weighted_sum * residual_weighted_sum / w_sum;
        let scale = if resid_ss.is_finite()
            && resid_ss > COLUMN_TOL
            && orig_ss.is_finite()
            && orig_ss > COLUMN_TOL
        {
            (orig_ss / resid_ss).sqrt()
        } else {
            1.0
        };
        weighted_column_mean[j] = centered_residual_mean;
        rescale[j] = scale;
    }

    Ok(ScaleDeviationTransform {
        projection_coef,
        weighted_column_mean,
        rescale,
        non_intercept_start,
    })
}

fn design_block_from_matrix(design: DesignMatrix) -> DesignBlock {
    match design {
        DesignMatrix::Dense(matrix) => DesignBlock::Dense(matrix),
        other => DesignBlock::Operator(Arc::new(other)),
    }
}

fn scale_deviation_reparam_matrix(transform: &ScaleDeviationTransform) -> Arc<Array2<f64>> {
    let p_noise = transform.projection_coef.ncols();
    let p_primary = transform.projection_coef.nrows();
    let mut q = Array2::<f64>::zeros((p_noise + p_primary + 1, p_noise));
    for j in 0..p_noise {
        let scale = transform.rescale[j];
        q[[j, j]] = scale;
        q[[p_noise + p_primary, j]] = -transform.weighted_column_mean[j] * scale;
        for i in 0..p_primary {
            q[[p_noise + i, j]] = -transform.projection_coef[[i, j]] * scale;
        }
    }
    Arc::new(q)
}

pub fn build_scale_deviation_operator(
    primary_design: DesignMatrix,
    rawnoise_design: DesignMatrix,
    transform: &ScaleDeviationTransform,
) -> Result<DesignMatrix, String> {
    if primary_design.nrows() != rawnoise_design.nrows() {
        return Err(format!(
            "scale deviation operator row mismatch: primary rows={}, noise rows={}",
            primary_design.nrows(),
            rawnoise_design.nrows()
        ));
    }
    if primary_design.ncols() != transform.projection_coef.nrows()
        || rawnoise_design.ncols() != transform.projection_coef.ncols()
    {
        return Err(format!(
            "scale deviation operator column mismatch: primary cols={}, noise cols={}, transform is {}x{}",
            primary_design.ncols(),
            rawnoise_design.ncols(),
            transform.projection_coef.nrows(),
            transform.projection_coef.ncols()
        ));
    }
    let n = rawnoise_design.nrows();
    let base = BlockDesignOperator::new(vec![
        design_block_from_matrix(rawnoise_design),
        design_block_from_matrix(primary_design),
        DesignBlock::Intercept(n),
    ])?;
    let operator = ReparamDesignOperator::new(
        Arc::new(base),
        scale_deviation_reparam_matrix(transform),
    )?;
    Ok(DesignMatrix::Operator(Arc::new(operator)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DesignMatrix;
    use std::sync::Arc;

    /// Verify the real scale-deviation pipeline for an overdetermined system.
    #[test]
    fn scale_deviation_transform_overdetermined() {
        let n = 1000;
        let p_primary = 10;
        let p_noise = 5;

        let mut primary = Array2::<f64>::zeros((n, p_primary));
        let mut noise = Array2::<f64>::zeros((n, p_noise));
        for i in 0..n {
            for j in 0..p_primary {
                primary[[i, j]] = ((i * 3 + j * 11) as f64 * 0.1).sin();
            }
            for j in 0..p_noise {
                noise[[i, j]] = ((i * 5 + j * 13) as f64 * 0.1).cos();
            }
        }
        // Intercept-like first column
        noise.column_mut(0).fill(1.0);
        let weights = Array1::<f64>::ones(n);

        let transform = build_scale_deviation_transform(&primary, &noise, &weights, 1)
            .expect("transform should succeed for overdetermined inputs");
        let transformed = apply_scale_deviation_transform(&primary, &noise, &transform)
            .expect("apply should succeed for overdetermined inputs");

        assert_eq!(transform.projection_coef.dim(), (p_primary, p_noise));
        assert_eq!(transformed.dim(), (n, p_noise));
        assert!(transformed.iter().all(|v| v.is_finite()));
        assert!(transformed.column(0).iter().all(|&v| v == 1.0));

        let primary_design = DesignMatrix::Dense(Arc::new(primary.clone()));
        let noise_design = DesignMatrix::Dense(Arc::new(noise.clone()));
        let non_intercept_start = infer_non_intercept_start_design(&noise_design, &weights)
            .expect("design-native non-intercept detection should succeed");
        assert_eq!(non_intercept_start, 1);
        let design_transform = build_scale_deviation_transform_design(
            &primary_design,
            &noise_design,
            &weights,
            non_intercept_start,
        )
        .expect("design-native transform should succeed");
        let transformed_design = build_scale_deviation_operator(
            primary_design,
            noise_design,
            &design_transform,
        )
        .expect("design-native operator should build")
        .to_dense();

        assert_eq!(design_transform.projection_coef.dim(), (p_primary, p_noise));
        assert_eq!(transformed_design.dim(), transformed.dim());
        for i in 0..n {
            for j in 0..p_noise {
                assert!(
                    (transformed_design[[i, j]] - transformed[[i, j]]).abs() <= 1e-8,
                    "design-native transform mismatch at ({i}, {j}): {} vs {}",
                    transformed_design[[i, j]],
                    transformed[[i, j]]
                );
            }
        }
    }
}
