use crate::faer_ndarray::FaerArrayView;
use faer::prelude::SolveLstsq;
use ndarray::{Array1, Array2, ArrayView1, s};

const COLUMN_TOL: f64 = 1e-12;

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
        let mut wx = Array2::<f64>::zeros((n, p_primary));
        let mut wy = Array2::<f64>::zeros((n, active_cols));
        for i in 0..n {
            let sw = weights[i].sqrt();
            for j in 0..p_primary {
                wx[[i, j]] = sw * primary_design[[i, j]];
            }
            for j in 0..active_cols {
                wy[[i, j]] = sw * noise_design[[i, first_active + j]];
            }
        }
        let wx_faer = FaerArrayView::new(&wx);
        let qr = wx_faer.as_ref().col_piv_qr();
        let mut rhs = wy;
        let mut rhs_mat = crate::faer_ndarray::array2_to_matmut(&mut rhs);
        qr.solve_lstsq_in_place(rhs_mat.as_mut());
        projection_coef
            .slice_mut(s![.., first_active..])
            .assign(&rhs.slice(s![..p_primary, ..]));
    }
    for j in first_active..p_noise {
        let col = noise_design.column(j).to_owned();
        let proj = projection_coef.column(j).to_owned();
        let fitted = crate::faer_ndarray::fast_av(primary_design, &proj);
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
        projection_coef.column_mut(j).assign(&proj);
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
    let projected = crate::faer_ndarray::fast_ab(primary_design, &transform.projection_coef);
    for j in transform.non_intercept_start.min(out.ncols())..out.ncols() {
        for i in 0..out.nrows() {
            out[[i, j]] = (out[[i, j]] - projected[[i, j]] - transform.weighted_column_mean[j])
                * transform.rescale[j];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
