use crate::faer_ndarray::{FaerArrayView, FaerSvd, fast_av};
use crate::matrix::{
    BlockDesignOperator, DesignBlock, DesignMatrix, ReparamDesignOperator,
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

/// Solve the projection `argmin_B ||sqrt(W)*(X*B - N)||_F` given the cross-products
/// `xtwx = X'WX` (p_primary x p_primary) and `xtwn = X'WN` (p_primary x p_noise_active).
///
/// Instead of forming `(X'WX)^{-1} X'WN` via Cholesky (which squares the condition
/// number of X), we use a truncated SVD on `X'WX` to build a numerically stable
/// pseudoinverse.  Singular values below `max(sigma) * tol` are zeroed, so the
/// result is well-behaved even when `X` is rank-deficient or ill-conditioned.
fn solve_projection_system(
    xtwx: &Array2<f64>,
    xtwn: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let p = xtwx.nrows();
    debug_assert_eq!(xtwx.ncols(), p);
    debug_assert_eq!(xtwn.nrows(), p);
    let k = xtwn.ncols();

    if p == 0 || k == 0 {
        return Ok(Array2::<f64>::zeros((p, k)));
    }

    // SVD of the symmetric PSD matrix X'WX = U * diag(sigma) * V'.
    // For a symmetric matrix U == V, but we request both for generality.
    let (u_opt, sigma, vt_opt) = xtwx
        .svd(true, true)
        .map_err(|e| format!("SVD failed in solve_projection_system: {e:?}"))?;
    let u = u_opt.ok_or("SVD did not produce left singular vectors")?;
    let vt = vt_opt.ok_or("SVD did not produce right singular vectors")?;

    // Truncation threshold: same convention as numpy/scipy lstsq.
    let sigma_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let tol = sigma_max * (p as f64) * f64::EPSILON;

    // Compute pseudoinverse application: B = V * diag(1/sigma_trunc) * U' * xtwn.
    // Step 1: tmp = U' * xtwn  (p x k)
    let ut_rhs = u.t().dot(xtwn);
    // Step 2: scale each row i of tmp by 1/sigma[i] (or 0 if below tolerance)
    let mut scaled = ut_rhs;
    for i in 0..p {
        let s = sigma[i];
        let inv_s = if s > tol { 1.0 / s } else { 0.0 };
        for j in 0..k {
            scaled[[i, j]] *= inv_s;
        }
    }
    // Step 3: result = V * scaled = Vt' * scaled
    Ok(vt.t().dot(&scaled))
}

pub fn build_scale_deviation_transform_design(
    primary_design: &DesignMatrix,
    noise_design: &DesignMatrix,
    weights: &Array1<f64>,
    non_intercept_start: usize,
) -> Result<ScaleDeviationTransform, String> {
    if primary_design.nrows() != noise_design.nrows() || weights.len() != noise_design.nrows() {
        return Err("scale deviation transform design row mismatch".to_string());
    }
    let n = primary_design.nrows();
    let p_primary = primary_design.ncols();
    let p_noise = noise_design.ncols();
    let mut projection_coef = Array2::<f64>::zeros((p_primary, p_noise));
    let mut weighted_column_mean = Array1::<f64>::zeros(p_noise);
    let mut rescale = Array1::<f64>::ones(p_noise);
    let first_active = non_intercept_start.min(p_noise);

    if first_active < p_noise {
        let active_cols = p_noise - first_active;

        // Accumulate cross-products X'WX and X'WN via row chunks, avoiding
        // full materialisation of the n x p design matrices.
        let mut xtwx = Array2::<f64>::zeros((p_primary, p_primary));
        let mut xtwn = Array2::<f64>::zeros((p_primary, active_cols));

        let chunk_rows = (SCALE_DESIGN_TARGET_CHUNK_BYTES
            / (p_primary.max(p_noise).max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(n.max(1));

        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let x_chunk = primary_design.row_chunk(start..end);
            let n_chunk = noise_design.row_chunk(start..end);
            let rows = end - start;

            for i in 0..rows {
                let w = weights[start + i];
                if w == 0.0 {
                    continue;
                }
                // Accumulate X'WX
                for a in 0..p_primary {
                    let wxa = w * x_chunk[[i, a]];
                    for b in a..p_primary {
                        let val = wxa * x_chunk[[i, b]];
                        xtwx[[a, b]] += val;
                        if a != b {
                            xtwx[[b, a]] += val;
                        }
                    }
                }
                // Accumulate X'WN (active columns only)
                for a in 0..p_primary {
                    let wxa = w * x_chunk[[i, a]];
                    for j in 0..active_cols {
                        xtwn[[a, j]] += wxa * n_chunk[[i, first_active + j]];
                    }
                }
            }
        }

        // Solve the projection via SVD-based pseudoinverse (numerically stable).
        let coef = solve_projection_system(&xtwx, &xtwn)?;
        projection_coef
            .slice_mut(s![.., first_active..])
            .assign(&coef);
    }

    // Compute residual statistics: center and rescale.
    // We need column-level access to the noise design and the fitted values
    // X * projection_coef[.., j].  We reuse row_chunk for streaming access.
    let chunk_rows = (SCALE_DESIGN_TARGET_CHUNK_BYTES
        / (p_primary.max(p_noise).max(1) * std::mem::size_of::<f64>()))
    .max(1)
    .min(n.max(1));

    for j in first_active..p_noise {
        // First pass: compute weighted mean of residual = noise_col - X * proj_col.
        let proj_col = projection_coef.column(j);
        let mut w_sum = 0.0;
        let mut w_resid_sum = 0.0;

        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let x_chunk = primary_design.row_chunk(start..end);
            let n_chunk = noise_design.row_chunk(start..end);
            let rows = end - start;
            for i in 0..rows {
                let w = weights[start + i];
                let mut fitted = 0.0;
                for k in 0..p_primary {
                    fitted += x_chunk[[i, k]] * proj_col[k];
                }
                let resid = n_chunk[[i, j]] - fitted;
                w_sum += w;
                w_resid_sum += w * resid;
            }
        }

        if !w_sum.is_finite() || w_sum <= 0.0 {
            return Err("scale deviation requires positive finite total weight".to_string());
        }
        let center = w_resid_sum / w_sum;

        // Second pass: weighted centered sum-of-squares for original and residual.
        let mut orig_css = 0.0;
        let mut resid_css = 0.0;
        // We also need the weighted mean of the original noise column.
        let mut w_noise_sum = 0.0;

        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let n_chunk = noise_design.row_chunk(start..end);
            let rows = end - start;
            for i in 0..rows {
                let w = weights[start + i];
                w_noise_sum += w * n_chunk[[i, j]];
            }
        }
        let noise_mean = w_noise_sum / w_sum;

        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let x_chunk = primary_design.row_chunk(start..end);
            let n_chunk = noise_design.row_chunk(start..end);
            let rows = end - start;
            for i in 0..rows {
                let w = weights[start + i];
                let nij = n_chunk[[i, j]];
                let d_orig = nij - noise_mean;
                orig_css += w * d_orig * d_orig;
                let mut fitted = 0.0;
                for k in 0..p_primary {
                    fitted += x_chunk[[i, k]] * proj_col[k];
                }
                let d_resid = nij - fitted - center;
                resid_css += w * d_resid * d_resid;
            }
        }

        let scale = if resid_css.is_finite()
            && resid_css > COLUMN_TOL
            && orig_css.is_finite()
            && orig_css > COLUMN_TOL
        {
            (orig_css / resid_css).sqrt()
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

fn design_block_from_matrix(design: DesignMatrix) -> DesignBlock {
    match design {
        DesignMatrix::Dense(matrix) => DesignBlock::Dense(matrix),
        other => DesignBlock::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(other))),
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
    let operator =
        ReparamDesignOperator::new(Arc::new(base), scale_deviation_reparam_matrix(transform))?;
    Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        Arc::new(operator),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DesignMatrix;

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

        let primary_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(primary.clone()));
        let noise_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(noise.clone()));
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
        let transformed_design =
            build_scale_deviation_operator(primary_design, noise_design, &design_transform)
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
