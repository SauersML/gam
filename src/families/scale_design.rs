use crate::faer_ndarray::{FaerArrayView, default_rrqr_rank_alpha, fast_ab};
use crate::matrix::DesignMatrix;
use dyn_stack::{MemBuffer, MemStack};
use faer::Unbind;
use faer::prelude::ReborrowMut;
use faer::{Conj, get_global_parallelism};
use ndarray::{Array1, Array2, s};
use std::ops::Range;

const COLUMN_TOL: f64 = 1e-12;
const SCALE_DESIGN_TARGET_CHUNK_BYTES: usize = 8 * 1024 * 1024;
// Explicit replay of the sigma orthogonalization on new rows is much more
// sensitive than fitting on the training rows: near-null QR directions can
// yield enormous projection coefficients that reconstruct the training design
// but catastrophically amplify modest prediction rows. Use a stronger cutoff
// than the generic RRQR epsilon rule and add a last-resort coefficient cap.
const SCALE_PROJECTION_REPLAY_RCOND_FLOOR: f64 = 1e-8;
const SCALE_PROJECTION_MAX_ABS_COEF: f64 = 1e6;

#[derive(Clone, Debug)]
pub struct ScaleDeviationTransform {
    pub projection_coef: Array2<f64>,
    pub weighted_column_mean: Array1<f64>,
    pub rescale: Array1<f64>,
    pub non_intercept_start: usize,
}

#[derive(Clone, Copy)]
enum ScaleDesignMatrixRef<'a> {
    Dense(&'a Array2<f64>),
    Design(&'a DesignMatrix),
}

impl ScaleDesignMatrixRef<'_> {
    fn nrows(self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.nrows(),
            Self::Design(matrix) => matrix.nrows(),
        }
    }

    fn ncols(self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.ncols(),
            Self::Design(matrix) => matrix.ncols(),
        }
    }

    fn row_chunk(self, rows: Range<usize>) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => matrix.slice(s![rows, ..]).to_owned(),
            Self::Design(matrix) => matrix.row_chunk(rows),
        }
    }
}

pub fn infer_non_intercept_start(design: &Array2<f64>, weights: &Array1<f64>) -> usize {
    infer_non_intercept_start_impl(
        ScaleDesignMatrixRef::Dense(design),
        weights,
        "weighted column stats row mismatch".to_string(),
    )
    .unwrap_or(0)
}

pub fn build_scale_deviation_transform(
    primary_design: &Array2<f64>,
    noise_design: &Array2<f64>,
    weights: &Array1<f64>,
    non_intercept_start: usize,
) -> Result<ScaleDeviationTransform, String> {
    build_scale_deviation_transform_impl(
        ScaleDesignMatrixRef::Dense(primary_design),
        ScaleDesignMatrixRef::Dense(noise_design),
        weights,
        non_intercept_start,
        "scale deviation transform row mismatch",
    )
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
    let n = rawnoise_design.nrows();
    let p_primary = primary_design.ncols();
    let p_noise = rawnoise_design.ncols();
    let chunk_rows = scale_design_row_chunk_size(n, p_primary.max(p_noise));
    let mut out = Array2::<f64>::zeros((n, p_noise));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let primary_chunk = primary_design.slice(s![start..end, ..]).to_owned();
        let noise_chunk = rawnoise_design.slice(s![start..end, ..]).to_owned();
        let chunk = apply_scale_deviation_reparam_chunk(&primary_chunk, &noise_chunk, transform);
        out.slice_mut(s![start..end, ..]).assign(&chunk);
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

fn scale_design_row_chunk_size(nrows: usize, max_cols: usize) -> usize {
    (SCALE_DESIGN_TARGET_CHUNK_BYTES / (max_cols.max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(nrows.max(1))
}

fn weighted_column_stats(
    design: ScaleDesignMatrixRef<'_>,
    weights: &Array1<f64>,
    row_mismatch_error: String,
) -> Result<WeightedColumnStats, String> {
    if design.nrows() != weights.len() {
        return Err(row_mismatch_error);
    }
    let total_weight = validate_scale_weights(weights)?;
    let p = design.ncols();
    let mut weighted_sum = Array1::<f64>::zeros(p);
    let mut weighted_sum_sq = Array1::<f64>::zeros(p);
    let chunk_rows = scale_design_row_chunk_size(design.nrows(), p);
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

fn infer_non_intercept_start_impl(
    design: ScaleDesignMatrixRef<'_>,
    weights: &Array1<f64>,
    row_mismatch_error: String,
) -> Result<usize, String> {
    let stats = weighted_column_stats(design, weights, row_mismatch_error)?;
    let mut end = 0;
    for j in 0..stats.weighted_sum.len() {
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

fn build_weighted_primary_design(
    primary_design: ScaleDesignMatrixRef<'_>,
    sqrtw: &Array1<f64>,
    chunk_rows: usize,
) -> Array2<f64> {
    let n = primary_design.nrows();
    let p_primary = primary_design.ncols();
    let mut wx = Array2::<f64>::zeros((n, p_primary));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let x_chunk = primary_design.row_chunk(start..end);
        for local in 0..(end - start) {
            let sw = sqrtw[start + local];
            for col in 0..p_primary {
                wx[[start + local, col]] = sw * x_chunk[[local, col]];
            }
        }
    }
    wx
}

fn scale_projection_rank_tolerance(leading_diag: f64, major_dim: usize) -> f64 {
    let leading_diag = leading_diag.max(1.0);
    let generic =
        default_rrqr_rank_alpha() * f64::EPSILON * (major_dim.max(1) as f64) * leading_diag;
    let replay_safe = SCALE_PROJECTION_REPLAY_RCOND_FLOOR * leading_diag;
    generic.max(replay_safe)
}

fn scale_projection_effective_rank(r_diag: &[f64], leading_diag: f64, major_dim: usize) -> usize {
    let tol = scale_projection_rank_tolerance(leading_diag, major_dim);
    r_diag.iter().take_while(|&&d| d.abs() > tol).count()
}

fn solve_scale_projection_for_rank(
    primary_design: ScaleDesignMatrixRef<'_>,
    noise_design: ScaleDesignMatrixRef<'_>,
    sqrtw: &Array1<f64>,
    qr: &faer::linalg::solvers::ColPivQr<f64>,
    r_diag: &[f64],
    perm_fwd: &[usize],
    rank: usize,
    first_active: usize,
    chunk_rows: usize,
) -> Array2<f64> {
    let n = primary_design.nrows();
    let p_primary = primary_design.ncols();
    let p_noise = noise_design.ncols();
    let mut projection_coef = Array2::<f64>::zeros((p_primary, p_noise));
    let active_cols = p_noise.saturating_sub(first_active);
    if active_cols == 0 || p_primary == 0 || rank == 0 {
        return projection_coef;
    }
    let r = qr.thin_R();
    let chunk_cols = (SCALE_DESIGN_TARGET_CHUNK_BYTES / (n.max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(active_cols);

    for chunk_start in (0..active_cols).step_by(chunk_cols) {
        let width = (active_cols - chunk_start).min(chunk_cols);
        let mut rhs = Array2::<f64>::zeros((n, width));
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let noise_chunk = noise_design.row_chunk(start..end);
            for local in 0..(end - start) {
                let sw = sqrtw[start + local];
                for col in 0..width {
                    rhs[[start + local, col]] =
                        sw * noise_chunk[[local, first_active + chunk_start + col]];
                }
            }
        }

        let mut rhs_mat = crate::faer_ndarray::array2_to_matmut(&mut rhs);
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.Q_basis(),
            qr.Q_coeff(),
            Conj::Yes,
            rhs_mat.rb_mut(),
            get_global_parallelism(),
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
                    n,
                    qr.Q_coeff().nrows(),
                    width,
                ),
            )),
        );

        let mut pivoted_solution = Array2::<f64>::zeros((rank, width));
        for col in 0..width {
            for i in (0..rank).rev() {
                let mut accum = rhs[[i, col]];
                for k in (i + 1)..rank {
                    accum -= r[(i, k)] * pivoted_solution[[k, col]];
                }
                let rii = r_diag[i];
                if rii != 0.0 {
                    pivoted_solution[[i, col]] = accum / rii;
                }
            }
        }
        for pivoted_col in 0..rank {
            let orig_col = perm_fwd[pivoted_col];
            for rhs_col in 0..width {
                projection_coef[[orig_col, first_active + chunk_start + rhs_col]] =
                    pivoted_solution[[pivoted_col, rhs_col]];
            }
        }
    }

    projection_coef
}

fn solve_scale_projection(
    primary_design: ScaleDesignMatrixRef<'_>,
    noise_design: ScaleDesignMatrixRef<'_>,
    weights: &Array1<f64>,
    first_active: usize,
    chunk_rows: usize,
) -> Result<Array2<f64>, String> {
    let n = primary_design.nrows();
    let p_primary = primary_design.ncols();
    let p_noise = noise_design.ncols();
    let mut projection_coef = Array2::<f64>::zeros((p_primary, p_noise));
    let active_cols = p_noise.saturating_sub(first_active);

    if active_cols == 0 || p_primary == 0 {
        return Ok(projection_coef);
    }

    let sqrtw = weights.mapv(f64::sqrt);
    let wx = build_weighted_primary_design(primary_design, &sqrtw, chunk_rows);
    let wx_faer = FaerArrayView::new(&wx);
    // Keep one explicit RRQR factorization and one explicit rank policy for all
    // projection coefficients so dense and operator-backed paths reuse the same solve.
    let qr = wx_faer.as_ref().col_piv_qr();
    let r = qr.thin_R();
    let diag_len = r.nrows().min(r.ncols());
    let r_diag = (0..diag_len).map(|i| r[(i, i)]).collect::<Vec<_>>();
    let leading_diag = r_diag.first().copied().map(f64::abs).unwrap_or(0.0);
    let mut rank = scale_projection_effective_rank(&r_diag, leading_diag, n.max(p_primary));
    let (perm_fwd, _) = qr.P().arrays();
    let perm_fwd: Vec<usize> = perm_fwd.iter().map(|idx| idx.unbound()).collect();
    loop {
        projection_coef = solve_scale_projection_for_rank(
            primary_design,
            noise_design,
            &sqrtw,
            &qr,
            &r_diag,
            &perm_fwd,
            rank,
            first_active,
            chunk_rows,
        );
        let max_abs_coef = projection_coef
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        if !max_abs_coef.is_finite() || (max_abs_coef > SCALE_PROJECTION_MAX_ABS_COEF && rank > 0) {
            rank -= 1;
            if rank == 0 {
                return Ok(Array2::<f64>::zeros((p_primary, p_noise)));
            }
            continue;
        }
        break;
    }

    Ok(projection_coef)
}

fn apply_projection_chunk(
    primary_chunk: &Array2<f64>,
    projection_coef: &Array2<f64>,
    first_active: usize,
) -> Array2<f64> {
    if first_active >= projection_coef.ncols() {
        Array2::<f64>::zeros((primary_chunk.nrows(), 0))
    } else {
        fast_ab(
            primary_chunk,
            &projection_coef.slice(s![.., first_active..]).to_owned(),
        )
    }
}

fn build_scale_deviation_transform_impl(
    primary_design: ScaleDesignMatrixRef<'_>,
    noise_design: ScaleDesignMatrixRef<'_>,
    weights: &Array1<f64>,
    non_intercept_start: usize,
    row_mismatch_error: &str,
) -> Result<ScaleDeviationTransform, String> {
    if primary_design.nrows() != noise_design.nrows() || weights.len() != noise_design.nrows() {
        return Err(row_mismatch_error.to_string());
    }
    validate_scale_weights(weights)?;

    let n = primary_design.nrows();
    let p_primary = primary_design.ncols();
    let p_noise = noise_design.ncols();
    let first_active = non_intercept_start.min(p_noise);
    let chunk_rows = scale_design_row_chunk_size(n, p_primary.max(p_noise));
    let projection_coef = solve_scale_projection(
        primary_design,
        noise_design,
        weights,
        first_active,
        chunk_rows,
    )?;
    let mut weighted_column_mean = Array1::<f64>::zeros(p_noise);
    let mut rescale = Array1::<f64>::ones(p_noise);
    let active_cols = p_noise - first_active;

    if active_cols > 0 {
        let projection_only_transform = ScaleDeviationTransform {
            projection_coef: projection_coef.clone(),
            weighted_column_mean: Array1::<f64>::zeros(p_noise),
            rescale: Array1::<f64>::ones(p_noise),
            non_intercept_start,
        };
        let mut w_sum = 0.0;
        let mut w_resid_sum = Array1::<f64>::zeros(active_cols);
        let mut w_noise_sum = Array1::<f64>::zeros(active_cols);

        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let x_chunk = primary_design.row_chunk(start..end);
            let noise_chunk = noise_design.row_chunk(start..end);
            let resid_chunk = apply_scale_deviation_reparam_chunk(
                &x_chunk,
                &noise_chunk,
                &projection_only_transform,
            );
            for local in 0..(end - start) {
                let w = weights[start + local];
                if w == 0.0 {
                    continue;
                }
                w_sum += w;
                for jj in 0..active_cols {
                    let nij = noise_chunk[[local, first_active + jj]];
                    w_noise_sum[jj] += w * nij;
                    w_resid_sum[jj] += w * resid_chunk[[local, first_active + jj]];
                }
            }
        }

        if !w_sum.is_finite() || w_sum <= 0.0 {
            return Err("scale deviation requires positive finite total weight".to_string());
        }

        let resid_center = w_resid_sum.mapv(|sum| sum / w_sum);
        let noise_mean = w_noise_sum.mapv(|sum| sum / w_sum);
        let mut orig_css = Array1::<f64>::zeros(active_cols);
        let mut resid_css = Array1::<f64>::zeros(active_cols);

        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let x_chunk = primary_design.row_chunk(start..end);
            let noise_chunk = noise_design.row_chunk(start..end);
            let resid_chunk = apply_scale_deviation_reparam_chunk(
                &x_chunk,
                &noise_chunk,
                &projection_only_transform,
            );
            for local in 0..(end - start) {
                let w = weights[start + local];
                if w == 0.0 {
                    continue;
                }
                for jj in 0..active_cols {
                    let nij = noise_chunk[[local, first_active + jj]];
                    let d_orig = nij - noise_mean[jj];
                    orig_css[jj] += w * d_orig * d_orig;
                    let d_resid = resid_chunk[[local, first_active + jj]] - resid_center[jj];
                    resid_css[jj] += w * d_resid * d_resid;
                }
            }
        }

        for jj in 0..active_cols {
            let j = first_active + jj;
            let scale = if resid_css[jj].is_finite()
                && resid_css[jj] > COLUMN_TOL
                && orig_css[jj].is_finite()
                && orig_css[jj] > COLUMN_TOL
            {
                (orig_css[jj] / resid_css[jj]).sqrt()
            } else {
                1.0
            };
            weighted_column_mean[j] = resid_center[jj];
            rescale[j] = scale;
        }
    }

    Ok(ScaleDeviationTransform {
        projection_coef,
        weighted_column_mean,
        rescale,
        non_intercept_start,
    })
}

pub fn infer_non_intercept_start_design(
    design: &DesignMatrix,
    weights: &Array1<f64>,
) -> Result<usize, String> {
    infer_non_intercept_start_impl(
        ScaleDesignMatrixRef::Design(design),
        weights,
        format!(
            "weighted column stats row mismatch: design has {} rows, weights have {} entries",
            design.nrows(),
            weights.len()
        ),
    )
}

pub fn build_scale_deviation_transform_design(
    primary_design: &DesignMatrix,
    noise_design: &DesignMatrix,
    weights: &Array1<f64>,
    non_intercept_start: usize,
) -> Result<ScaleDeviationTransform, String> {
    build_scale_deviation_transform_impl(
        ScaleDesignMatrixRef::Design(primary_design),
        ScaleDesignMatrixRef::Design(noise_design),
        weights,
        non_intercept_start,
        "scale deviation transform design row mismatch",
    )
}

/// Apply the scale-deviation reparameterisation to a chunk of rows.
///
/// Instead of embedding the projection coefficients into a large augmented
/// matrix (which changes FP operation order relative to the canonical
/// `apply_projection_chunk`), we compute the projection via the shared
/// helper and then fold in rescaling and centering explicitly.  This
/// guarantees bit-identical projection arithmetic on both paths.
fn apply_scale_deviation_reparam_chunk(
    primary_chunk: &Array2<f64>,
    noise_chunk: &Array2<f64>,
    transform: &ScaleDeviationTransform,
) -> Array2<f64> {
    let rows = noise_chunk.nrows();
    let p_noise = noise_chunk.ncols();
    let first_active = transform.non_intercept_start.min(p_noise);
    let mut out = Array2::<f64>::zeros((rows, p_noise));

    // Pass-through columns (intercept-like) are copied verbatim.
    for j in 0..first_active {
        for i in 0..rows {
            out[[i, j]] = noise_chunk[[i, j]];
        }
    }

    // Active columns: residual = noise - projection, then center & rescale.
    if first_active < p_noise {
        let fitted =
            apply_projection_chunk(primary_chunk, &transform.projection_coef, first_active);
        for j in first_active..p_noise {
            let jj = j - first_active;
            let scale = transform.rescale[j];
            let center = transform.weighted_column_mean[j];
            for i in 0..rows {
                out[[i, j]] = (noise_chunk[[i, j]] - fitted[[i, jj]] - center) * scale;
            }
        }
    }

    out
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
    let p_primary = primary_design.ncols();
    let p_noise = rawnoise_design.ncols();
    let chunk_rows = scale_design_row_chunk_size(n, p_primary.max(p_noise));
    let mut out = Array2::<f64>::zeros((n, p_noise));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let primary_chunk = primary_design.row_chunk(start..end);
        let noise_chunk = rawnoise_design.row_chunk(start..end);
        let chunk = apply_scale_deviation_reparam_chunk(&primary_chunk, &noise_chunk, transform);
        out.slice_mut(s![start..end, ..]).assign(&chunk);
    }
    Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        out,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DesignMatrix;

    fn assert_matrix_close(lhs: &Array2<f64>, rhs: &Array2<f64>, tol: f64, label: &str) {
        assert_eq!(
            lhs.dim(),
            rhs.dim(),
            "{label} shape mismatch: left {:?}, right {:?}",
            lhs.dim(),
            rhs.dim()
        );
        for i in 0..lhs.nrows() {
            for j in 0..lhs.ncols() {
                assert!(
                    (lhs[[i, j]] - rhs[[i, j]]).abs() <= tol,
                    "{label} mismatch at ({i}, {j}): {} vs {}",
                    lhs[[i, j]],
                    rhs[[i, j]]
                );
            }
        }
    }

    fn assert_transform_close(
        lhs: &ScaleDeviationTransform,
        rhs: &ScaleDeviationTransform,
        tol: f64,
    ) {
        assert_eq!(lhs.non_intercept_start, rhs.non_intercept_start);
        assert_matrix_close(
            &lhs.projection_coef,
            &rhs.projection_coef,
            tol,
            "projection coefficients",
        );
        assert_eq!(
            lhs.weighted_column_mean.len(),
            rhs.weighted_column_mean.len()
        );
        assert_eq!(lhs.rescale.len(), rhs.rescale.len());
        for j in 0..lhs.weighted_column_mean.len() {
            assert!(
                (lhs.weighted_column_mean[j] - rhs.weighted_column_mean[j]).abs() <= tol,
                "weighted column mean mismatch at {j}: {} vs {}",
                lhs.weighted_column_mean[j],
                rhs.weighted_column_mean[j]
            );
            assert!(
                (lhs.rescale[j] - rhs.rescale[j]).abs() <= tol,
                "rescale mismatch at {j}: {} vs {}",
                lhs.rescale[j],
                rhs.rescale[j]
            );
        }
    }

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
        assert_transform_close(&transform, &design_transform, 1e-10);
        assert_matrix_close(
            &transformed_design,
            &transformed,
            1e-8,
            "transformed design",
        );
    }

    #[test]
    fn scale_deviation_transform_rank_deficient_primary_matches_design_path() {
        let n = 384;
        let p_primary = 4;
        let p_noise = 4;
        let mut primary = Array2::<f64>::zeros((n, p_primary));
        let mut noise = Array2::<f64>::zeros((n, p_noise));
        let mut weights = Array1::<f64>::zeros(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            let wobble = (17.0 * t).sin();
            primary[[i, 0]] = 1.0;
            primary[[i, 1]] = t;
            primary[[i, 2]] = t + 1e-12 * wobble;
            primary[[i, 3]] = 2.0 * t - 1e-12 * wobble;

            noise[[i, 0]] = 1.0;
            noise[[i, 1]] = 0.7 * t + 0.2 * (9.0 * t).cos();
            noise[[i, 2]] = primary[[i, 1]] - primary[[i, 2]] + 0.1 * (13.0 * t).sin();
            noise[[i, 3]] = 0.5 * primary[[i, 3]] + 0.3 * (5.0 * t).cos();

            weights[i] = if i % 17 == 0 {
                0.0
            } else {
                0.5 + (11.0 * t).sin().abs()
            };
        }

        let transform = build_scale_deviation_transform(&primary, &noise, &weights, 1)
            .expect("dense transform should succeed for ill-conditioned primary");
        let transformed = apply_scale_deviation_transform(&primary, &noise, &transform)
            .expect("dense apply should succeed for ill-conditioned primary");

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
        .expect("design-native transform should succeed for ill-conditioned primary");
        let transformed_design =
            build_scale_deviation_operator(primary_design, noise_design, &design_transform)
                .expect("design-native operator should build for ill-conditioned primary")
                .to_dense();

        assert_transform_close(&transform, &design_transform, 1e-10);
        assert_matrix_close(
            &transformed_design,
            &transformed,
            1e-8,
            "ill-conditioned transformed design",
        );
    }

    #[test]
    fn scale_projection_rank_uses_replay_safe_cutoff() {
        let leading = 4.0;
        let diag = vec![
            leading,
            1e-4 * leading,
            2.0 * SCALE_PROJECTION_REPLAY_RCOND_FLOOR * leading,
            0.5 * SCALE_PROJECTION_REPLAY_RCOND_FLOOR * leading,
        ];
        let rank = scale_projection_effective_rank(&diag, leading, 18);
        assert_eq!(
            rank, 3,
            "replay-safe scale projection rank should drop diagonals below the replay cutoff"
        );

        let dropped = scale_projection_effective_rank(
            &[
                leading,
                1e-4 * leading,
                0.5 * SCALE_PROJECTION_REPLAY_RCOND_FLOOR * leading,
            ],
            leading,
            18,
        );
        assert_eq!(dropped, 2);
    }

    #[test]
    fn scale_projection_drops_high_gain_replay_direction() {
        let n = 32;
        let eps = 2e-8;
        let mut primary = Array2::<f64>::zeros((n, 2));
        let mut noise = Array2::<f64>::zeros((n, 2));
        let weights = Array1::<f64>::ones(n);
        for i in 0..n {
            let t = i as f64;
            primary[[i, 0]] = 1.0;
            primary[[i, 1]] = eps * t;
            noise[[i, 0]] = 1.0;
            noise[[i, 1]] = (0.37 * t).sin();
        }

        let chunk_rows = scale_design_row_chunk_size(n, 2);
        let sqrtw = weights.mapv(f64::sqrt);
        let wx = build_weighted_primary_design(
            ScaleDesignMatrixRef::Dense(&primary),
            &sqrtw,
            chunk_rows,
        );
        let qr = FaerArrayView::new(&wx).as_ref().col_piv_qr();
        let r = qr.thin_R();
        let diag_len = r.nrows().min(r.ncols());
        let r_diag = (0..diag_len).map(|i| r[(i, i)]).collect::<Vec<_>>();
        let leading_diag = r_diag.first().copied().map(f64::abs).unwrap_or(0.0);
        let rank = scale_projection_effective_rank(&r_diag, leading_diag, n.max(primary.ncols()));
        assert_eq!(
            rank, 2,
            "replay-safe rank floor should still keep the high-gain direction before coefficient capping"
        );

        let (perm_fwd, _) = qr.P().arrays();
        let perm_fwd: Vec<usize> = perm_fwd.iter().map(|idx| idx.unbound()).collect();
        let uncapped_projection = solve_scale_projection_for_rank(
            ScaleDesignMatrixRef::Dense(&primary),
            ScaleDesignMatrixRef::Dense(&noise),
            &sqrtw,
            &qr,
            &r_diag,
            &perm_fwd,
            rank,
            1,
            chunk_rows,
        );
        let uncapped_max_abs = uncapped_projection
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            uncapped_max_abs > SCALE_PROJECTION_MAX_ABS_COEF,
            "constructed replay solve should exceed the coefficient cap before fallback, got {uncapped_max_abs}"
        );

        let capped_projection = solve_scale_projection(
            ScaleDesignMatrixRef::Dense(&primary),
            ScaleDesignMatrixRef::Dense(&noise),
            &weights,
            1,
            chunk_rows,
        )
        .expect("projection solve should succeed");
        let capped_max_abs = capped_projection
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            capped_max_abs <= SCALE_PROJECTION_MAX_ABS_COEF,
            "fallback should remove high-gain replay directions, got max coefficient {capped_max_abs}"
        );
    }
}
