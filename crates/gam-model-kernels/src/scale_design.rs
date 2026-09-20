use gam_linalg::faer_ndarray::fast_ab;
use gam_linalg::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, FiniteSignedWeightsView, LinearOperator,
};
use ndarray::{Array1, Array2, ArrayViewMut2, s};
use std::ops::Range;
use std::sync::Arc;

/// Typed error variants for the scale-deviation design module.
///
/// External-facing helpers continue to return `Result<_, String>`; this enum
/// is materialized internally and converted at the boundary so that error
/// text remains byte-identical to the previous `format!` output.
#[derive(Debug, Clone)]
pub enum ScaleDesignError {
    /// Weight vector contains an invalid entry (NaN/inf, negative, or sums
    /// to a non-positive / non-finite total).
    InvalidWeights { reason: String },
    /// Dimensions of the supplied matrices/vectors are inconsistent.
    IncompatibleDimensions { reason: String },
    /// Input value is not finite where finiteness is required (e.g. saved
    /// projection cutoff alpha).
    NonFiniteInput { reason: String },
    /// Saved payload is partially populated or the projection is degenerate
    /// (e.g. zero rows with non-empty columns).
    DegenerateDesign { reason: String },
    /// Row materialization from an underlying `DesignMatrix` failed.
    RowMaterializationFailed { reason: String },
}

impl_reason_error_boilerplate! {
    ScaleDesignError {
        InvalidWeights,
        IncompatibleDimensions,
        NonFiniteInput,
        DegenerateDesign,
        RowMaterializationFailed,
    }
}

// Imported, not transcribed (#2704): the same streamed working-set budget,
// used for a row chunk in `scale_design_row_chunk_size`.
const SCALE_DESIGN_TARGET_CHUNK_BYTES: usize =
    gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES;

#[derive(Clone, Debug)]
pub struct ScaleDeviationTransform {
    pub projection_coef: Array2<f64>,
    pub weighted_column_mean: Array1<f64>,
    pub rescale: Array1<f64>,
    pub non_intercept_start: usize,
    /// Squared SVD truncation cutoff a saved model's `projection_coef` was fitted
    /// with. Carried so that model's prediction-time replay reads exactly what it
    /// saved; no current fit derives a projection (#3015).
    pub projection_ridge_alpha: f64,
}

impl ScaleDeviationTransform {
    /// Identity (no-op) reparameterization: zero projection, zero centering,
    /// unit rescale. [`build_scale_deviation_operator`] with this transform
    /// returns the raw scale design verbatim, and the saved-payload round-trip
    /// replays the same identity at prediction time.
    ///
    /// A location and a scale predictor remain SEPARATELY identifiable even when
    /// they share a covariate basis: they enter the likelihood through different
    /// sufficient statistics (the standardized residual versus its square / the
    /// log-scale), so residualizing the scale design against the location design
    /// — replacing `X_σ` with `(I − P_{X_μ}) X_σ` — imposes a spurious
    /// constraint and erases real heteroscedastic signal whenever the two blocks
    /// overlap. The Gaussian and binomial location-scale paths keep their log-σ
    /// designs un-residualized (`location_scale_log_sigma_design`, #3015); this
    /// constructor lets the survival location-scale path do the same while
    /// preserving the transform plumbing (payload serialization, prediction-time
    /// replay).
    pub fn identity(p_primary: usize, p_noise: usize, non_intercept_start: usize) -> Self {
        ScaleDeviationTransform {
            projection_coef: Array2::<f64>::zeros((p_primary, p_noise)),
            weighted_column_mean: Array1::<f64>::zeros(p_noise),
            rescale: Array1::<f64>::ones(p_noise),
            non_intercept_start,
            projection_ridge_alpha: 0.0,
        }
    }
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
    scale_transform_from_payload_typed(
        projection,
        center,
        scale,
        non_intercept_start,
        projection_ridge_alpha,
    )
    .map_err(|e| e.to_string())
}

fn scale_transform_from_payload_typed(
    projection: &Option<Vec<Vec<f64>>>,
    center: &Option<Vec<f64>>,
    scale: &Option<Vec<f64>>,
    non_intercept_start: Option<usize>,
    projection_ridge_alpha: Option<f64>,
) -> Result<Option<ScaleDeviationTransform>, ScaleDesignError> {
    match (projection, center, scale, non_intercept_start) {
        (None, None, None, None) => Ok(None),
        (Some(projection), Some(center), Some(scale), Some(non_intercept_start)) => {
            let rows = projection.len();
            let cols = center.len();
            if cols != scale.len() {
                return Err(ScaleDesignError::IncompatibleDimensions {
                    reason: "saved scale transform center/scale length mismatch".to_string(),
                });
            }
            if rows == 0 && cols > 0 {
                return Err(ScaleDesignError::DegenerateDesign {
                    reason: "saved scale transform projection has zero rows".to_string(),
                });
            }
            let mut projection_coef = Array2::<f64>::zeros((rows, cols));
            for (i, row) in projection.iter().enumerate() {
                if row.len() != cols {
                    return Err(ScaleDesignError::IncompatibleDimensions {
                        reason: "saved scale transform projection width mismatch".to_string(),
                    });
                }
                for (j, &value) in row.iter().enumerate() {
                    projection_coef[[i, j]] = value;
                }
            }
            let Some(projection_ridge_alpha) = projection_ridge_alpha else {
                return Err(ScaleDesignError::DegenerateDesign {
                    reason:
                        "saved scale transform payload is missing projection_ridge_alpha; refit"
                            .to_string(),
                });
            };
            if !projection_ridge_alpha.is_finite() || projection_ridge_alpha < 0.0 {
                return Err(ScaleDesignError::NonFiniteInput {
                    reason: format!(
                        "saved scale transform projection_ridge_alpha must be finite and non-negative, got {projection_ridge_alpha}"
                    ),
                });
            }
            Ok(Some(ScaleDeviationTransform {
                projection_coef,
                weighted_column_mean: Array1::from_vec(center.clone()),
                rescale: Array1::from_vec(scale.clone()),
                non_intercept_start,
                projection_ridge_alpha,
            }))
        }
        _ => Err(ScaleDesignError::DegenerateDesign {
            reason: "saved scale transform payload is only partially populated; refit".to_string(),
        }),
    }
}

#[derive(Clone, Copy)]
enum ScaleDesignMatrixRef<'a> {
    Design(&'a DesignMatrix),
}

impl ScaleDesignMatrixRef<'_> {
    #[inline]
    fn nrows(self) -> usize {
        match self {
            Self::Design(matrix) => matrix.nrows(),
        }
    }

    #[inline]
    fn ncols(self) -> usize {
        match self {
            Self::Design(matrix) => matrix.ncols(),
        }
    }

    fn row_chunk(self, rows: Range<usize>) -> Result<Array2<f64>, ScaleDesignError> {
        match self {
            Self::Design(matrix) => {
                matrix
                    .try_row_chunk(rows)
                    .map_err(|e| ScaleDesignError::RowMaterializationFailed {
                        reason: format!("scale deviation row materialization failed: {e}"),
                    })
            }
        }
    }
}

fn dim_err(reason: impl Into<String>) -> ScaleDesignError {
    ScaleDesignError::IncompatibleDimensions {
        reason: reason.into(),
    }
}

struct ScaleDeviationOperator {
    primary_design: DesignMatrix,
    rawnoise_design: DesignMatrix,
    transform: ScaleDeviationTransform,
    chunk_rows: usize,
}

impl ScaleDeviationOperator {
    fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, ScaleDesignError> {
        let primary_chunk = self
            .primary_design
            .try_row_chunk(rows.clone())
            .map_err(|e| ScaleDesignError::RowMaterializationFailed {
                reason: format!("scale deviation operator primary chunk: {e}"),
            })?;
        let noise_chunk = self.rawnoise_design.try_row_chunk(rows).map_err(|e| {
            ScaleDesignError::RowMaterializationFailed {
                reason: format!("scale deviation operator noise chunk: {e}"),
            }
        })?;
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
            return Err(dim_err(format!(
                "scale deviation operator XtWX weight mismatch: weights={}, rows={}",
                weights.len(),
                self.nrows()
            ))
            .to_string());
        }
        FiniteSignedWeightsView::try_from_array(weights)
            .map_err(|reason| format!("scale deviation operator XtWX: {reason}"))?;
        let n = self.nrows();
        let p = self.ncols();
        let mut out = Array2::<f64>::zeros((p, p));
        for start in (0..n).step_by(self.chunk_rows) {
            let end = (start + self.chunk_rows).min(n);
            let chunk = self.row_chunk(start..end).map_err(|e| e.to_string())?;
            for local in 0..chunk.nrows() {
                let w = weights[start + local];
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
}

impl DenseDesignOperator for ScaleDeviationOperator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), gam_runtime::resource::MatrixMaterializationError> {
        let chunk = self.row_chunk(rows).map_err(|err| {
            gam_runtime::resource::MatrixMaterializationError::RowMaterializationFailed {
                context: "ScaleDeviationOperator::row_chunk_into",
                reason: err.to_string(),
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
    /// Rows that actually contributed, i.e. those with a nonzero weight. This
    /// is the accumulation depth of `weighted_sum` and `weighted_sum_sq` — the
    /// zero-weight rows are `continue`d and round nothing — and so it is what
    /// the roundoff band of a quantity built from them is a function of.
    contributing_rows: usize,
}

fn validate_scale_weights(weights: &Array1<f64>) -> Result<f64, ScaleDesignError> {
    let mut total_weight = 0.0;
    for (idx, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(ScaleDesignError::NonFiniteInput {
                reason: format!("scale deviation weight {idx} is not finite"),
            });
        }
        if w < 0.0 {
            return Err(ScaleDesignError::InvalidWeights {
                reason: format!(
                    "scale deviation requires non-negative weights, got {w} at index {idx}"
                ),
            });
        }
        total_weight += w;
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(ScaleDesignError::InvalidWeights {
            reason: "scale deviation requires positive finite total weight".to_string(),
        });
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
) -> Result<WeightedColumnStats, ScaleDesignError> {
    if design.nrows() != weights.len() {
        return Err(dim_err(row_mismatch_error));
    }
    let total_weight = validate_scale_weights(weights)?;
    let p = design.ncols();
    let mut weighted_sum = Array1::<f64>::zeros(p);
    let mut weighted_sum_sq = Array1::<f64>::zeros(p);
    let chunk_rows = scale_design_row_chunk_size(design.nrows(), p);
    let mut contributing_rows = 0usize;
    for start in (0..design.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(design.nrows());
        let chunk = design.row_chunk(start..end)?;
        for local in 0..(end - start) {
            let w = weights[start + local];
            if w == 0.0 {
                continue;
            }
            contributing_rows += 1;
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
        contributing_rows,
    })
}

fn infer_non_intercept_start_impl(
    design: ScaleDesignMatrixRef<'_>,
    weights: &Array1<f64>,
    row_mismatch_error: String,
) -> Result<usize, ScaleDesignError> {
    let stats = weighted_column_stats(design, weights, row_mismatch_error)?;
    let mut end = 0;
    for j in 0..stats.weighted_sum.len() {
        // Textbook one-pass centered sum of squares, and therefore a
        // DIFFERENCE OF TWO LARGE NON-NEGATIVE QUANTITIES. For the constant
        // column this test exists to find, the two are equal in exact
        // arithmetic and the computed difference is nothing but the roundoff of
        // the accumulations that produced them — so the threshold it is
        // compared against has to be that roundoff, and nothing else.
        //
        // Both operands are sums of non-negative terms (weights are validated
        // non-negative above), so `Σ|terms| = raw_ss + mean_ss` exactly, and
        // Wilkinson's band applies with no cancellation inside either operand.
        // Depth: each contributing row commits two multiplies and one add into
        // `weighted_sum_sq` and one multiply and one add into `weighted_sum`,
        // then the correction costs a square, a divide and the subtraction.
        let raw_ss = stats.weighted_sum_sq[j];
        let mean_ss = stats.weighted_sum[j] * stats.weighted_sum[j] / stats.total_weight;
        let centered_ss = raw_ss - mean_ss;
        let roundoff_floor = gam_linalg::roundoff::accumulation_band(
            5 * stats.contributing_rows + 3,
            raw_ss + mean_ss,
        );
        // The value replaced here was an ABSOLUTE `1e-12`, which cannot be
        // right for a quantity in the units of `Σ w x²`: a constant column of
        // magnitude `1e6` over `1e6` unit-weight rows leaves a cancellation
        // residue near `1e8`, so the column it was written to find was not
        // found, while a genuinely varying column whose values are `~1e-7`
        // scores `~1e-14` and was misread as constant.
        if centered_ss <= roundoff_floor {
            end = j + 1;
        } else {
            break;
        }
    }
    Ok(end)
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
    .map_err(|e| e.to_string())
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
    build_scale_deviation_operator_typed(primary_design, rawnoise_design, transform)
        .map_err(|e| e.to_string())
}

fn build_scale_deviation_operator_typed(
    primary_design: DesignMatrix,
    rawnoise_design: DesignMatrix,
    transform: &ScaleDeviationTransform,
) -> Result<DesignMatrix, ScaleDesignError> {
    if primary_design.nrows() != rawnoise_design.nrows() {
        return Err(dim_err(format!(
            "scale deviation operator row mismatch: primary rows={}, noise rows={}",
            primary_design.nrows(),
            rawnoise_design.nrows()
        )));
    }
    if primary_design.ncols() != transform.projection_coef.nrows()
        || rawnoise_design.ncols() != transform.projection_coef.ncols()
    {
        return Err(dim_err(format!(
            "scale deviation operator column mismatch: primary cols={}, noise cols={}, transform is {}x{}",
            primary_design.ncols(),
            rawnoise_design.ncols(),
            transform.projection_coef.nrows(),
            transform.projection_coef.ncols()
        )));
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
    use gam_linalg::matrix::DesignMatrix;
    use ndarray::array;

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

    #[test]
    fn scale_deviation_operator_gram_preserves_signed_weights() {
        let primary = array![[1.0], [2.0], [-1.0], [0.5]];
        let noise = array![[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 1.5]];
        let transform = ScaleDeviationTransform::identity(1, 2, 0);
        let design = build_scale_deviation_operator(
            DesignMatrix::Dense(DenseDesignMatrix::from(primary)),
            DesignMatrix::Dense(DenseDesignMatrix::from(noise.clone())),
            &transform,
        )
        .unwrap();
        let weights = array![2.0, -3.0, 0.25, -1.5];
        let weighted_noise = noise.clone() * weights.view().insert_axis(ndarray::Axis(1));
        let expected = noise.t().dot(&weighted_noise);
        let got = design.diag_xtw_x(&weights).unwrap();
        assert_matrix_close(&got, &expected, 1e-12, "signed scale-deviation Gram");

        let bad = array![1.0, f64::NAN, f64::INFINITY, 1.0];
        let err = design.diag_xtw_x(&bad).unwrap_err();
        assert!(err.contains("row 1"), "unexpected diagnostic: {err}");
    }

    /// A saved model's residualized noise design replays bit for bit after the
    /// transform-fitting route is gone (#3015): read from its payload fields, the
    /// transform maps each active column to `(noise − primary·P − center)·rescale` and
    /// passes the intercept through. Every input is a short dyadic rational, so each
    /// product and sum is exact in f64 and the expected design has one bit pattern,
    /// whatever order the operator accumulates in.
    #[test]
    fn a_saved_scale_transform_replays_its_noise_design_bit_for_bit() {
        let primary = array![[1.0, 0.5], [1.0, -0.25], [1.0, 1.5], [1.0, -2.0]];
        let noise = array![
            [1.0, 0.75, 1.0],
            [1.0, -1.25, 0.5],
            [1.0, 2.5, -0.75],
            [1.0, 0.125, 3.0]
        ];
        let projection = vec![vec![0.0, 0.25, -0.5], vec![0.0, 0.125, 1.0]];
        let center = vec![0.0, 0.0625, -0.375];
        let rescale = vec![1.0, 2.0, 0.5];
        let transform = scale_transform_from_payload(
            &Some(projection.clone()),
            &Some(center.clone()),
            &Some(rescale.clone()),
            Some(1),
            Some(0.0),
        )
        .expect("saved transform")
        .expect("a populated payload carries a transform");
        let replayed = build_scale_deviation_operator(
            DesignMatrix::Dense(DenseDesignMatrix::from(primary.clone())),
            DesignMatrix::Dense(DenseDesignMatrix::from(noise.clone())),
            &transform,
        )
        .expect("replay operator")
        .to_dense();
        let expected = Array2::from_shape_fn(noise.dim(), |(i, j)| {
            if j == 0 {
                noise[[i, 0]]
            } else {
                let fitted =
                    primary[[i, 0]] * projection[0][j] + primary[[i, 1]] * projection[1][j];
                (noise[[i, j]] - fitted - center[j]) * rescale[j]
            }
        });
        for ((i, j), value) in replayed.indexed_iter() {
            assert_eq!(
                value.to_bits(),
                expected[[i, j]].to_bits(),
                "replayed noise design entry ({i}, {j}): {value} against {}",
                expected[[i, j]]
            );
        }
    }

    /// A saved model's transform — the only way a residualized noise design still
    /// reaches prediction since no fit residualizes one (#3015) — replays exactly as
    /// it was written.
    #[test]
    fn scale_transform_payload_round_trips_every_field() {
        let transform = ScaleDeviationTransform {
            projection_coef: array![[0.0, 0.25], [0.0, -1.5], [0.0, 0.125]],
            weighted_column_mean: array![0.0, 0.375],
            rescale: array![1.0, 2.5],
            non_intercept_start: 1,
            projection_ridge_alpha: 1.0e-16,
        };
        let projection: Vec<Vec<f64>> = transform
            .projection_coef
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        let restored = scale_transform_from_payload(
            &Some(projection),
            &Some(transform.weighted_column_mean.to_vec()),
            &Some(transform.rescale.to_vec()),
            Some(transform.non_intercept_start),
            Some(transform.projection_ridge_alpha),
        )
        .expect("payload round-trip should succeed")
        .expect("payload should produce a transform");
        assert_eq!(restored.projection_coef, transform.projection_coef);
        assert_eq!(restored.weighted_column_mean, transform.weighted_column_mean);
        assert_eq!(restored.rescale, transform.rescale);
        assert_eq!(restored.non_intercept_start, transform.non_intercept_start);
        assert_eq!(
            restored.projection_ridge_alpha.to_bits(),
            transform.projection_ridge_alpha.to_bits(),
            "alpha must round-trip exactly through payload serialization"
        );
    }
}
