use gam_linalg::faer_ndarray::{FaerSvd, fast_ab};
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
    /// Thin SVD of the weighted primary design failed or produced no
    /// singular vectors.
    SvdFailed { reason: String },
}

impl_reason_error_boilerplate! {
    ScaleDesignError {
        InvalidWeights,
        IncompatibleDimensions,
        NonFiniteInput,
        DegenerateDesign,
        RowMaterializationFailed,
        SvdFailed,
    }
}

// Floor on a centred weighted sum of squares below which the noise-replay
// rescale factor `sqrt(orig_css / resid_css)` is not formed and the column is
// left at unit scale.
//
// This is a DECLARED, UNMEASURED policy and not a derived bound, stated here
// rather than left implicit. Unlike the intercept-detection test in
// `infer_non_intercept_start_impl` (which compares a cancelling difference and
// so has a roundoff band to compare against), both operands here are
// already-centred sums of non-negative terms: they cancel nothing, so their
// roundoff band is a fixed fraction of themselves and a band-based test
// degenerates to `> 0.0`. What this floor actually guards is the AMPLIFICATION
// of the ratio as `resid_css` shrinks, and the amplification a replay solve can
// absorb has not been measured. Retuning it without that measurement would move
// published column scales, so the value is left where it was found.
const RESCALE_CENTERED_SS_FLOOR: f64 = 1e-12;
// Imported, not transcribed (#2704): the same streamed working-set budget,
// used for a row chunk in `scale_design_row_chunk_size` and a column chunk in
// the replay solve. `SCALE_OPERATOR_MATRIX_FREE_PCG_THRESHOLD` below is NOT a
// derivation of this constant despite its prose: it is an independent literal
// rounded to 10^6 doubles, 48_576 elements short of the 1_048_576 f64 elements
// this budget actually holds. Deriving it exactly would move a routing
// threshold by ~5%, so it is left as the independent literal it actually is
// rather than dressed as a derivation from a value that is itself unmeasured.
const SCALE_DESIGN_TARGET_CHUNK_BYTES: usize =
    gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES;
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
// Above this many materialized entries (rows × noise columns) the scale-deviation
// operator routes its normal-equation solve through matrix-free PCG instead of
// forming a dense `XᵀWX`. The dense path costs `O(n · p²)` time and `O(p²)`
// memory; once the explicit operator footprint reaches ~10⁶ doubles (~8 MB,
// the same order as `SCALE_DESIGN_TARGET_CHUNK_BYTES` but NOT derived from it —
// see the note there) the chunked matrix-free path is the cheaper, more
// cache-friendly route.
const SCALE_OPERATOR_MATRIX_FREE_PCG_THRESHOLD: usize = 1_000_000;

#[derive(Clone, Debug)]
pub struct ScaleDeviationTransform {
    pub projection_coef: Array2<f64>,
    pub weighted_column_mean: Array1<f64>,
    pub rescale: Array1<f64>,
    pub non_intercept_start: usize,
    /// Squared SVD truncation cutoff used when fitting `projection_coef`.
    /// Stored so prediction-time replay is reproducible without re-deriving
    /// the cutoff from heuristics.
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
    /// overlap. The Gaussian location-scale path already keeps its log-σ design
    /// un-residualized (`identified_gaussian_log_sigma_design`); this constructor
    /// lets the survival location-scale path do the same while preserving the
    /// transform plumbing (payload serialization, prediction-time replay).
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
    Dense(&'a Array2<f64>),
    Design(&'a DesignMatrix),
}

impl ScaleDesignMatrixRef<'_> {
    #[inline]
    fn nrows(self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.nrows(),
            Self::Design(matrix) => matrix.nrows(),
        }
    }

    #[inline]
    fn ncols(self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.ncols(),
            Self::Design(matrix) => matrix.ncols(),
        }
    }

    fn row_chunk(self, rows: Range<usize>) -> Result<Array2<f64>, ScaleDesignError> {
        match self {
            Self::Dense(matrix) => Ok(matrix.slice(s![rows, ..]).to_owned()),
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
    .map_err(|e| e.to_string())
}

#[derive(Clone)]
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

    fn uses_matrix_free_pcg(&self) -> bool {
        self.primary_design
            .nrows()
            .saturating_mul(self.rawnoise_design.ncols())
            > SCALE_OPERATOR_MATRIX_FREE_PCG_THRESHOLD
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

fn build_weighted_primary_design(
    primary_design: ScaleDesignMatrixRef<'_>,
    sqrtw: &Array1<f64>,
    chunk_rows: usize,
) -> Result<Array2<f64>, ScaleDesignError> {
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

/// Pick the squared singular-value cutoff for the replay solve.
///
/// Retained directions use the exact inverse `1 / sigma_k`; directions at or
/// below `sqrt(alpha)` are dropped. We want the worst-case prediction-row
/// leverage amplification — a unit-norm new row transformed by the saved
/// coefficients — to be at most `SCALE_PROJECTION_LEVERAGE_AMPLIFICATION`
/// times what a sigma_max-scale direction sees in the un-regularized solve.
/// The rcond floor supplies the minimum cutoff for numerical conditioning.
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
) -> Result<(Array2<f64>, f64), ScaleDesignError> {
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
    // applied to the weighted noise RHS. Retained singular directions use the
    // exact inverse; unresolved directions are dropped by the cutoff below.
    let (u_opt, singular, vt_opt) =
        wx.svd(true, true)
            .map_err(|e| ScaleDesignError::SvdFailed {
                reason: format!("scale projection SVD failed: {e:?}"),
            })?;
    let (Some(u), Some(vt)) = (u_opt, vt_opt) else {
        return Err(ScaleDesignError::SvdFailed {
            reason: "scale projection SVD did not return singular vectors".to_string(),
        });
    };
    let alpha = choose_scale_projection_ridge_alpha(singular.as_slice().unwrap_or(&[]));
    let rank = singular.len();
    if rank == 0 {
        return Ok((projection_coef, alpha));
    }
    // Truncated SVD with leverage-bound cutoff: directions resolved well
    // enough to keep coefficient amplification under
    // SCALE_PROJECTION_LEVERAGE_AMPLIFICATION are inverted exactly (no
    // damping on the dominant components), and weaker directions are
    // dropped. The primary design is fixed across any single replay, so no
    // threshold-crossings occur within a call: the projection is a linear
    // function of the noise RHS, which is the continuity property the audit
    // asked for. The discarded singular value floor sqrt(alpha) doubles as
    // the recovered-coefficient leverage cap.
    let cutoff = alpha.sqrt();
    let mut filter = Array1::<f64>::zeros(rank);
    for k in 0..rank {
        let s = singular[k];
        filter[k] = if s > cutoff && s > 0.0 { 1.0 / s } else { 0.0 };
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
        // Apply filter rowwise: t_k *= 1 / sigma_k for retained directions.
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
) -> Result<ScaleDeviationTransform, ScaleDesignError> {
    if primary_design.nrows() != noise_design.nrows() || weights.len() != noise_design.nrows() {
        return Err(dim_err(row_mismatch_error.to_string()));
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
            return Err(ScaleDesignError::InvalidWeights {
                reason: "scale deviation requires positive finite total weight".to_string(),
            });
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
                && resid_css[jj] > RESCALE_CENTERED_SS_FLOOR
                && orig_css[jj].is_finite()
                && orig_css[jj] > RESCALE_CENTERED_SS_FLOOR
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
    .map_err(|e| e.to_string())
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
        // the cap kicks in. With a fixed SVD cutoff, the replayed coefficient
        // is a linear function of the input perturbation.
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
        // projected coefficient cell. Numerical first differences should be
        // bounded because the fixed projection operator is linear in the input.
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
    }
}
