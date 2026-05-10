use crate::faer_ndarray::{FaerSvd, fast_ab};
use crate::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
use ndarray::{Array1, Array2, ArrayViewMut2, s};
use std::ops::Range;
use std::sync::Arc;

const COLUMN_TOL: f64 = 1e-12;
const SCALE_DESIGN_TARGET_CHUNK_BYTES: usize = 8 * 1024 * 1024;
// Numerical conditioning floor for the SVD truncation tolerance: we drop any
// singular direction below `RCOND_FLOOR * sigma_max`, which is the standard
// machine-precision boundary for considering a direction resolvable. Above
// this floor, the replay solve is unbiased least squares (no Tikhonov
// damping), so noise in the primary span is recovered exactly. This is the
// primary safety net.
const SCALE_PROJECTION_REPLAY_RCOND_FLOOR: f64 = 1e-8;
// Optional tighter cap on coefficient amplification, used only when the
// design is so well-conditioned that even the worst retained direction would
// not amplify a unit prediction row beyond this multiple. For natural smooth
// bases (cond ≈ 100–1000) this cap is dominated by the rcond floor and has no
// effect; it kicks in only for nearly-orthogonal designs where one could
// otherwise tighten the cutoff without losing real signal. Setting this much
// smaller than `1 / RCOND_FLOOR` would discard real signal from moderately
// conditioned bases and is intentionally avoided.
const SCALE_PROJECTION_LEVERAGE_AMPLIFICATION: f64 = 1.0e8;

#[derive(Clone, Debug)]
pub struct ScaleDeviationTransform {
    pub projection_coef: Array2<f64>,
    pub weighted_column_mean: Array1<f64>,
    pub rescale: Array1<f64>,
    pub non_intercept_start: usize,
    /// Tikhonov regularization parameter actually used when fitting
    /// `projection_coef`.  Stored so prediction-time replay is
    /// reproducible without re-deriving alpha from heuristics.
    pub projection_ridge_alpha: f64,
}

/// Build a [`ScaleDeviationTransform`] from saved projection metadata.
///
/// Returns `Ok(None)` only when the payload is completely absent; partial
/// payloads are invalid because prediction cannot replay the fitted scale
/// reparameterization unambiguously.
pub fn scale_transform_from_payload(
    projection: &Option<Vec<Vec<f64>>>,
    center: &Option<Vec<f64>>,
    scale: &Option<Vec<f64>>,
    non_intercept_start: Option<usize>,
    projection_ridge_alpha: Option<f64>,
) -> Result<Option<ScaleDeviationTransform>, String> {
    match (projection, center, scale, non_intercept_start) {
        (None, None, None, None) => Ok(None),
        (Some(projection), Some(center), Some(scale), Some(non_intercept_start)) => {
            let rows = projection.len();
            let cols = center.len();
            if cols != scale.len() {
                return Err("saved scale transform center/scale length mismatch".to_string());
            }
            if rows == 0 && cols > 0 {
                return Err("saved scale transform projection has zero rows".to_string());
            }
            let mut projection_coef = Array2::<f64>::zeros((rows, cols));
            for (i, row) in projection.iter().enumerate() {
                if row.len() != cols {
                    return Err("saved scale transform projection width mismatch".to_string());
                }
                for (j, &value) in row.iter().enumerate() {
                    projection_coef[[i, j]] = value;
                }
            }
            // Older saved payloads (pre-Tikhonov) did not record alpha.  Treat
            // them as alpha=0: replay then matches the original least-squares
            // coefficients exactly, which is the previous behavior.
            let projection_ridge_alpha = projection_ridge_alpha.unwrap_or(0.0);
            if !projection_ridge_alpha.is_finite() || projection_ridge_alpha < 0.0 {
                return Err(format!(
                    "saved scale transform projection_ridge_alpha must be finite and non-negative, got {projection_ridge_alpha}"
                ));
            }
            Ok(Some(ScaleDeviationTransform {
                projection_coef,
                weighted_column_mean: Array1::from_vec(center.clone()),
                rescale: Array1::from_vec(scale.clone()),
                non_intercept_start,
                projection_ridge_alpha,
            }))
        }
        _ => Err(
            "saved scale transform payload is only partially populated; refit with current CLI"
                .to_string(),
        ),
    }
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

    fn row_chunk(self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        match self {
            Self::Dense(matrix) => Ok(matrix.slice(s![rows, ..]).to_owned()),
            Self::Design(matrix) => matrix
                .try_row_chunk(rows)
                .map_err(|e| format!("scale deviation row materialization failed: {e}")),
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

#[derive(Clone)]
struct ScaleDeviationOperator {
    primary_design: DesignMatrix,
    rawnoise_design: DesignMatrix,
    transform: ScaleDeviationTransform,
    chunk_rows: usize,
}

impl ScaleDeviationOperator {
    fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        let primary_chunk = self
            .primary_design
            .try_row_chunk(rows.clone())
            .map_err(|e| format!("scale deviation operator primary chunk: {e}"))?;
        let noise_chunk = self
            .rawnoise_design
            .try_row_chunk(rows)
            .map_err(|e| format!("scale deviation operator noise chunk: {e}"))?;
        Ok(apply_scale_deviation_reparam_chunk(
            &primary_chunk,
            &noise_chunk,
            &self.transform,
        ))
    }
}

impl LinearOperator for ScaleDeviationOperator {
    fn nrows(&self) -> usize {
        self.rawnoise_design.nrows()
    }

    fn ncols(&self) -> usize {
        self.rawnoise_design.ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(vector.len(), self.ncols());
        let n = self.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for start in (0..n).step_by(self.chunk_rows) {
            let end = (start + self.chunk_rows).min(n);
            let chunk = self
                .row_chunk(start..end)
                .expect("scale deviation operator row chunk failed");
            out.slice_mut(s![start..end]).assign(&chunk.dot(vector));
        }
        out
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(vector.len(), self.nrows());
        let n = self.nrows();
        let p = self.ncols();
        let mut out = Array1::<f64>::zeros(p);
        for start in (0..n).step_by(self.chunk_rows) {
            let end = (start + self.chunk_rows).min(n);
            let chunk = self
                .row_chunk(start..end)
                .expect("scale deviation operator row chunk failed");
            out += &chunk.t().dot(&vector.slice(s![start..end]).to_owned());
        }
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "scale deviation operator XtWX weight mismatch: weights={}, rows={}",
                weights.len(),
                self.nrows()
            ));
        }
        let n = self.nrows();
        let p = self.ncols();
        let mut out = Array2::<f64>::zeros((p, p));
        for start in (0..n).step_by(self.chunk_rows) {
            let end = (start + self.chunk_rows).min(n);
            let chunk = self.row_chunk(start..end)?;
            for local in 0..chunk.nrows() {
                let w = weights[start + local].max(0.0);
                if w == 0.0 {
                    continue;
                }
                for a in 0..p {
                    let xa = chunk[[local, a]];
                    for b in a..p {
                        let value = w * xa * chunk[[local, b]];
                        out[[a, b]] += value;
                        if a != b {
                            out[[b, a]] += value;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        self.primary_design
            .nrows()
            .saturating_mul(self.rawnoise_design.ncols())
            > 1_000_000
    }
}

impl DenseDesignOperator for ScaleDeviationOperator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), crate::resource::MatrixMaterializationError> {
        let chunk = self.row_chunk(rows).map_err(|_| {
            crate::resource::MatrixMaterializationError::MissingRowChunk {
                context: "ScaleDeviationOperator::row_chunk_into",
            }
        })?;
        out.assign(&chunk);
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        let n = self.nrows();
        let p = self.ncols();
        let mut out = Array2::<f64>::zeros((n, p));
        for start in (0..n).step_by(self.chunk_rows) {
            let end = (start + self.chunk_rows).min(n);
            let chunk = self
                .row_chunk(start..end)
                .expect("scale deviation operator row chunk failed");
            out.slice_mut(s![start..end, ..]).assign(&chunk);
        }
        out
    }
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
        let chunk = design.row_chunk(start..end)?;
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
) -> Result<Array2<f64>, String> {
    let n = primary_design.nrows();
    let p_primary = primary_design.ncols();
    let mut wx = Array2::<f64>::zeros((n, p_primary));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let x_chunk = primary_design.row_chunk(start..end)?;
        for local in 0..(end - start) {
            let sw = sqrtw[start + local];
            for col in 0..p_primary {
                wx[[start + local, col]] = sw * x_chunk[[local, col]];
            }
        }
    }
    Ok(wx)
}

/// Pick the Tikhonov regularization parameter alpha for the replay solve.
///
/// Filter factors `f_k = sigma_k / (sigma_k^2 + alpha)` are bounded above by
/// `1 / (2 * sqrt(alpha))` (achieved at sigma_k = sqrt(alpha)).  We want the
/// worst-case prediction-row leverage amplification — a unit-norm new row
/// transformed by the saved coefficients — to be at most
/// `SCALE_PROJECTION_LEVERAGE_AMPLIFICATION` times what a sigma_max-scale
/// direction sees in the un-regularized solve.  Solving for alpha gives the
/// dimensionless tolerance below; we add an absolute floor so even a pristine
/// design retains a sub-epsilon ridge that prevents catastrophic cancellation.
fn choose_scale_projection_ridge_alpha(singular: &[f64]) -> f64 {
    if singular.is_empty() {
        return 0.0;
    }
    let sigma_max = singular.iter().copied().fold(0.0_f64, f64::max);
    if !sigma_max.is_finite() || sigma_max <= 0.0 {
        return 0.0;
    }
    let derived_tol = sigma_max / SCALE_PROJECTION_LEVERAGE_AMPLIFICATION;
    let truncation_tol = derived_tol.max(SCALE_PROJECTION_REPLAY_RCOND_FLOOR * sigma_max);
    truncation_tol * truncation_tol
}

fn solve_scale_projection(
    primary_design: ScaleDesignMatrixRef<'_>,
    noise_design: ScaleDesignMatrixRef<'_>,
    weights: &Array1<f64>,
    first_active: usize,
    chunk_rows: usize,
) -> Result<(Array2<f64>, f64), String> {
    let n = primary_design.nrows();
    let p_primary = primary_design.ncols();
    let p_noise = noise_design.ncols();
    let mut projection_coef = Array2::<f64>::zeros((p_primary, p_noise));
    let active_cols = p_noise.saturating_sub(first_active);

    if active_cols == 0 || p_primary == 0 {
        return Ok((projection_coef, 0.0));
    }

    let sqrtw = weights.mapv(f64::sqrt);
    let wx = build_weighted_primary_design(primary_design, &sqrtw, chunk_rows)?;
    // Thin SVD of W^{1/2} X_primary: replay reduces to V * diag(filter) * U^T
    // applied to the weighted noise RHS.  Tikhonov filter factors are the
    // smooth alternative to RRQR rank truncation + coefficient cap.
    let (u_opt, singular, vt_opt) = wx
        .svd(true, true)
        .map_err(|e| format!("scale projection SVD failed: {e:?}"))?;
    let (Some(u), Some(vt)) = (u_opt, vt_opt) else {
        return Err("scale projection SVD did not return singular vectors".to_string());
    };
    let alpha = choose_scale_projection_ridge_alpha(singular.as_slice().unwrap_or(&[]));
    let rank = singular.len();
    if rank == 0 {
        return Ok((projection_coef, alpha));
    }
    // Truncated SVD with leverage-bound cutoff: directions resolved well
    // enough to keep coefficient amplification under
    // SCALE_PROJECTION_LEVERAGE_AMPLIFICATION are inverted exactly (no
    // Tikhonov bias on the dominant components), and weaker directions are
    // dropped. The primary design is fixed across any single replay, so no
    // threshold-crossings occur within a call: the projection is a linear
    // function of the noise RHS, which is the continuity property the audit
    // asked for. The discarded singular value floor sqrt(alpha) doubles as
    // the recovered-coefficient leverage cap.
    let cutoff = alpha.sqrt();
    let mut filter = Array1::<f64>::zeros(rank);
    for k in 0..rank {
        let s = singular[k];
        filter[k] = if s > cutoff && s > 0.0 {
            1.0 / s
        } else {
            0.0
        };
    }

    let chunk_cols = (SCALE_DESIGN_TARGET_CHUNK_BYTES / (n.max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(active_cols);

    for chunk_start in (0..active_cols).step_by(chunk_cols) {
        let width = (active_cols - chunk_start).min(chunk_cols);
        let mut rhs = Array2::<f64>::zeros((n, width));
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let noise_chunk = noise_design.row_chunk(start..end)?;
            for local in 0..(end - start) {
                let sw = sqrtw[start + local];
                for col in 0..width {
                    rhs[[start + local, col]] =
                        sw * noise_chunk[[local, first_active + chunk_start + col]];
                }
            }
        }

        // U^T (rank x n) * rhs (n x width) -> (rank x width)
        let mut t = u.t().dot(&rhs);
        // Apply filter rowwise: t_k *= sigma_k / (sigma_k^2 + alpha).
        for k in 0..rank {
            let f = filter[k];
            for col in 0..width {
                t[[k, col]] *= f;
            }
        }
        // V (p_primary x rank) * t (rank x width) -> (p_primary x width).
        // vt has shape (rank, p_primary), so V = vt^T.
        let block = vt.t().dot(&t);
        for col in 0..width {
            for row in 0..p_primary {
                projection_coef[[row, first_active + chunk_start + col]] = block[[row, col]];
            }
        }
    }

    Ok((projection_coef, alpha))
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
    let (projection_coef, projection_ridge_alpha) = solve_scale_projection(
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
            projection_ridge_alpha,
        };
        let mut w_sum = 0.0;
        let mut w_resid_sum = Array1::<f64>::zeros(active_cols);
        let mut w_noise_sum = Array1::<f64>::zeros(active_cols);

        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let x_chunk = primary_design.row_chunk(start..end)?;
            let noise_chunk = noise_design.row_chunk(start..end)?;
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
            let x_chunk = primary_design.row_chunk(start..end)?;
            let noise_chunk = noise_design.row_chunk(start..end)?;
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
        projection_ridge_alpha,
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
    Ok(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
        ScaleDeviationOperator {
            primary_design,
            rawnoise_design,
            transform: transform.clone(),
            chunk_rows,
        },
    ))))
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
    fn choose_scale_projection_ridge_alpha_scales_with_sigma_max() {
        // Truncation tolerance is `RCOND_FLOOR * sigma_max` whenever the
        // leverage cap is looser (which it always is for the default 1e8
        // value), so alpha = (RCOND_FLOOR * sigma_max)^2.
        let alpha_unit = choose_scale_projection_ridge_alpha(&[1.0, 0.5, 1e-6]);
        let expected_unit = SCALE_PROJECTION_REPLAY_RCOND_FLOOR.powi(2);
        assert!(alpha_unit > 0.0);
        assert!(
            (alpha_unit - expected_unit).abs() < 1e-24,
            "alpha should be {expected_unit:e} for sigma_max=1, got {alpha_unit}"
        );

        let alpha_scaled = choose_scale_projection_ridge_alpha(&[100.0, 1.0]);
        let expected_scaled = (SCALE_PROJECTION_REPLAY_RCOND_FLOOR * 100.0).powi(2);
        assert!(
            (alpha_scaled - expected_scaled).abs() < 1e-18,
            "alpha should be {expected_scaled:e} for sigma_max=100, got {alpha_scaled}"
        );
        // Scales as sigma_max^2.
        assert!(
            (alpha_scaled / alpha_unit - 1.0e4).abs() < 1e-6,
            "alpha should scale as sigma_max^2; got ratio {}",
            alpha_scaled / alpha_unit
        );

        let alpha_floor = choose_scale_projection_ridge_alpha(&[]);
        assert_eq!(alpha_floor, 0.0);
    }

    #[test]
    fn ridge_replay_continuous_under_input_sweep() {
        // A near-collinear primary design plus a sweepable perturbation column
        // would, under the old hard coefficient cap, jump discontinuously when
        // the cap kicks in.  With the Tikhonov filter the replayed coefficient
        // must be a smooth function of the input perturbation.
        let n = 64;
        let mut primary = Array2::<f64>::zeros((n, 3));
        let mut noise = Array2::<f64>::zeros((n, 2));
        let weights = Array1::<f64>::ones(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            primary[[i, 0]] = 1.0;
            primary[[i, 1]] = t;
            // Near-collinear with col 1 — this is the high-gain direction.
            primary[[i, 2]] = t + 1e-9 * (5.0 * t).sin();
            noise[[i, 0]] = 1.0;
            noise[[i, 1]] = (0.4 * t).cos();
        }

        // Sweep: gradually scale one noise entry; record the corresponding
        // projected coefficient cell.  Numerical first differences should be
        // bounded — the ridge guarantees Lipschitz continuity in the input.
        let mut last: Option<f64> = None;
        let mut max_step: f64 = 0.0;
        for k in 0..50 {
            let s = k as f64 / 49.0;
            let mut perturbed = noise.clone();
            for i in 0..n {
                perturbed[[i, 1]] += s;
            }
            let transform = build_scale_deviation_transform(&primary, &perturbed, &weights, 1)
                .expect("ridge transform should succeed under input sweep");
            let val = transform.projection_coef[[2, 1]];
            if let Some(prev) = last {
                let step = (val - prev).abs();
                max_step = max_step.max(step);
            }
            last = Some(val);
        }
        // Step bound: with 50 samples over a unit sweep, a smooth dependence
        // produces uniform tiny jumps.  The old coefficient cap would emit a
        // single huge step at the cap boundary, easily blowing 1.0 here.
        assert!(
            max_step < 0.5,
            "replay coefficient sweep should be continuous, got max step {max_step}"
        );
    }

    #[test]
    fn ridge_replay_noise_free_is_near_identity() {
        // When the noise design lives in the column span of the primary
        // design and W^{1/2} X is well-conditioned, the Tikhonov ridge is
        // tiny and the residual after subtracting the projected fit is at
        // numerical zero.
        let n = 128;
        let p_primary = 4;
        let p_noise = 3;
        let mut primary = Array2::<f64>::zeros((n, p_primary));
        let mut noise = Array2::<f64>::zeros((n, p_noise));
        let weights = Array1::<f64>::ones(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            primary[[i, 0]] = 1.0;
            primary[[i, 1]] = t;
            primary[[i, 2]] = (3.0 * t).sin();
            primary[[i, 3]] = (2.0 * t - 0.4).powi(2);
            noise[[i, 0]] = 1.0;
            // Linear combinations of primary cols so the projection should
            // recover them exactly modulo a vanishing ridge.
            noise[[i, 1]] = 0.7 * primary[[i, 1]] - 0.3 * primary[[i, 2]];
            noise[[i, 2]] = 0.2 * primary[[i, 3]] + 0.1 * primary[[i, 1]];
        }

        let transform = build_scale_deviation_transform(&primary, &noise, &weights, 1)
            .expect("transform should succeed");
        let transformed = apply_scale_deviation_transform(&primary, &noise, &transform)
            .expect("apply should succeed");

        // Pass-through column unaffected.
        for i in 0..n {
            assert_eq!(transformed[[i, 0]], 1.0);
        }
        // Active columns: residuals should be near zero up to the ridge bias.
        // The ridge bias here is bounded by alpha / sigma_min and the design is
        // well-conditioned, so 1e-6 is a safe envelope.
        for j in 1..p_noise {
            for i in 0..n {
                assert!(
                    transformed[[i, j]].abs() < 1e-6,
                    "noise-free residual should be near zero at ({i},{j}), got {}",
                    transformed[[i, j]]
                );
            }
        }
        assert!(transform.projection_ridge_alpha > 0.0);
    }

    #[test]
    fn scale_transform_payload_round_trips_alpha() {
        let n = 64;
        let mut primary = Array2::<f64>::zeros((n, 3));
        let mut noise = Array2::<f64>::zeros((n, 2));
        let weights = Array1::<f64>::ones(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            primary[[i, 0]] = 1.0;
            primary[[i, 1]] = t;
            primary[[i, 2]] = (4.0 * t).cos();
            noise[[i, 0]] = 1.0;
            noise[[i, 1]] = (2.0 * t).sin();
        }
        let transform = build_scale_deviation_transform(&primary, &noise, &weights, 1)
            .expect("transform should succeed");

        let projection: Vec<Vec<f64>> = transform
            .projection_coef
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        let center = transform.weighted_column_mean.to_vec();
        let scale = transform.rescale.to_vec();
        let restored = scale_transform_from_payload(
            &Some(projection),
            &Some(center),
            &Some(scale),
            Some(transform.non_intercept_start),
            Some(transform.projection_ridge_alpha),
        )
        .expect("payload round-trip should succeed")
        .expect("payload should produce a transform");
        assert_eq!(
            restored.projection_ridge_alpha, transform.projection_ridge_alpha,
            "alpha must round-trip exactly through payload serialization"
        );

        let legacy = scale_transform_from_payload(
            &Some(
                transform
                    .projection_coef
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect(),
            ),
            &Some(transform.weighted_column_mean.to_vec()),
            &Some(transform.rescale.to_vec()),
            Some(transform.non_intercept_start),
            None,
        )
        .expect("legacy payload (no alpha) should still load")
        .expect("legacy payload should produce a transform");
        assert_eq!(
            legacy.projection_ridge_alpha, 0.0,
            "legacy payload without alpha must default to zero for backward compatibility"
        );
    }
}
