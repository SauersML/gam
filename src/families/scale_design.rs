use crate::faer_ndarray::FaerArrayView;
use faer::prelude::SolveLstsq;
use ndarray::{Array1, Array2, ArrayView1};

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

/// Solve the weighted least-squares projection `argmin_c || W^{1/2}(y - X c) ||^2`
/// using column-pivoted QR on `W^{1/2} X`.
///
/// Column-pivoted QR is a direct method (no iterative convergence issues)
/// that handles rank-deficiency naturally through pivoting, gracefully
/// dealing with collinear columns (e.g. saturated I-spline columns that
/// duplicate the intercept) without failing.
fn solveweighted_projection_dense(
    design: &Array2<f64>,
    target: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let (n, p) = design.dim();
    if target.len() != n || weights.len() != n {
        return Err("weighted projection dimension mismatch".to_string());
    }
    // Form W^{1/2} X and W^{1/2} y, then solve via column-pivoted QR.
    // This is a direct method (no convergence issues) and handles
    // rank-deficiency naturally through the pivoting.
    let mut wx = Array2::<f64>::zeros((n, p));
    let mut wy = Array1::<f64>::zeros(n);
    for i in 0..n {
        let sw = weights[i].sqrt();
        wy[i] = sw * target[i];
        for j in 0..p {
            wx[[i, j]] = sw * design[[i, j]];
        }
    }
    let wx_faer = FaerArrayView::new(&wx);
    let qr = wx_faer.as_ref().col_piv_qr();
    // solve_lstsq_in_place solves the least-squares problem for the n×p
    // overdetermined system; the p-length solution is written into the first
    // p entries of the n-length rhs vector.
    let mut rhs = wy;
    let mut rhs_mat = crate::faer_ndarray::array1_to_col_matmut(&mut rhs);
    qr.solve_lstsq_in_place(rhs_mat.as_mut());
    Ok(rhs.slice(ndarray::s![..p]).to_owned())
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
    for j in non_intercept_start.min(p_noise)..p_noise {
        let col = noise_design.column(j).to_owned();
        let proj = solveweighted_projection_dense(primary_design, &col, weights)?;
        let fitted = primary_design.dot(&proj);
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
    let projected = primary_design.dot(&transform.projection_coef);
    for j in transform.non_intercept_start.min(out.ncols())..out.ncols() {
        for i in 0..out.nrows() {
            out[[i, j]] = (out[[i, j]] - projected[[i, j]] - transform.weighted_column_mean[j])
                * transform.rescale[j];
        }
    }
    Ok(out)
}
