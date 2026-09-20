use gam_linalg::matrix::{DesignMatrix, FactorizedSystem, SymmetricMatrix};
use gam_runtime::resource::prediction_chunk_rows;
use gam_solve::constrained_posterior::ConstrainedPosteriorCorrection;
use ndarray::{Array1, Array2, ArrayView2, s};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::borrow::Cow;
use std::ops::Range;

pub enum PredictionCovarianceBackend<'a> {
    Dense(ArrayView2<'a, f64>),
    Factorized {
        factor: Box<dyn FactorizedSystem>,
        dim: usize,
        /// Multiplicative dispersion `φ` applied to the result of the solve.
        ///
        /// The fallback backend factorizes the *unscaled* penalized Hessian
        /// `H = X'WX + S`, but the module invariant is `Vb = φ · H^{-1}`.
        /// Without this scale the variance computed via
        /// `H^{-1} rhs` would silently drop `φ` and disagree with the stored
        /// `beta_covariance()` (which is already φ-scaled).
        phi_scale: f64,
        /// Factored variance removed by inequality truncation. The lift already
        /// lives on the same φ-scaled covariance metric returned by this
        /// backend, so it is subtracted after the ambient precision solve. It
        /// is the truncation of the ambient covariance this backend applies:
        /// the conditional one's, or `Vp`'s own when `smoothing_factor` is set.
        constrained_correction: Option<Cow<'a, ConstrainedPosteriorCorrection>>,
        /// The smoothing correction's square-root factor `B` on the active
        /// coordinates, when the backend applies the smoothing-corrected
        /// `Vp = Vb + B·Bᵀ` of a fit whose inference stayed factorized (#3283).
        smoothing_factor: Option<Array2<f64>>,
        /// The coefficient gauge's section `T` when the factorized precision
        /// lives on active coordinates `θ` of `β = T·θ + a` rather than on the
        /// saved coefficients. The backend then applies `T·Cov(θ)·Tᵀ`, the
        /// covariance of the saved coefficients; the affine shift moves no
        /// covariance.
        gauge_lift: Option<ArrayView2<'a, f64>>,
    },
}

impl<'a> PredictionCovarianceBackend<'a> {
    pub fn from_dense(covariance: ArrayView2<'a, f64>) -> Self {
        Self::Dense(covariance)
    }

    /// Factorize the penalized Hessian and multiply the resulting `H^{-1} rhs`
    /// by the supplied dispersion `φ`, so the backend returns `φ · H^{-1} rhs`
    /// (i.e. the `Vb = φ · H^{-1}` covariance application).
    pub(crate) fn from_factorized_hessian_scaled(
        hessian: SymmetricMatrix,
        phi: f64,
    ) -> Result<Self, String> {
        Self::from_factorized_hessian_scaled_with_correction(hessian, phi, None)
    }

    pub fn from_factorized_hessian_scaled_with_correction(
        hessian: SymmetricMatrix,
        phi: f64,
        constrained_correction: Option<&'a ConstrainedPosteriorCorrection>,
    ) -> Result<Self, String> {
        if hessian.nrows() != hessian.ncols() {
            return Err(format!(
                "prediction precision backend requires a square Hessian, got {}x{}",
                hessian.nrows(),
                hessian.ncols()
            ));
        }
        let has_nonzero = match &hessian {
            SymmetricMatrix::Dense(m) => m.iter().any(|v| v.abs() > 0.0),
            SymmetricMatrix::Sparse(m) => {
                let (_, vals) = m.parts();
                vals.iter().any(|v| v.abs() > 0.0)
            }
        };
        if !has_nonzero {
            return Err("prediction precision backend requires a non-zero Hessian".to_string());
        }
        // `phi` is the coefficient-covariance scale of `Vb = phi * H^{-1}`.
        // Zero is a legitimate value (the exact-fit profiled Gaussian, whose
        // stored dense `Vb` is `0 * H^{-1}`), so it is applied as-is; a
        // non-finite or negative scale is not a covariance scale at all and is
        // refused rather than replaced by 1, which would silently publish the
        // unscaled `H^{-1}` as `Vb`.
        if !(phi.is_finite() && phi >= 0.0) {
            return Err(format!(
                "prediction precision backend requires a finite non-negative coefficient-covariance \
                 scale, got {phi}"
            ));
        }
        let dim = hessian.nrows();
        let factor = hessian.factorize()?;
        let phi_scale = phi;
        Ok(Self::Factorized {
            factor,
            dim,
            phi_scale,
            constrained_correction: constrained_correction.map(Cow::Borrowed),
            smoothing_factor: None,
            gauge_lift: None,
        })
    }

    /// Carry a conditional factorized backend (built with no truncation) to
    /// the smoothing-corrected law `Vp = Vb + B·Bᵀ` of a fit whose inference
    /// stayed factorized (#3283). `smoothing_factor` is `B` on the backend's
    /// active coordinates. `truncation` builds a constrained fit's lift at
    /// `Vp` from `Vp·Aᵀ`, which this closure receives through
    /// [`Self::apply_ambient_active`]: the corrected law is the truncation at
    /// `Vp`'s own lift, as the fit publishes it, never `Vb`'s lift plus `B·Bᵀ`.
    pub fn with_smoothing_correction(
        self,
        smoothing_factor: Array2<f64>,
        truncation: impl FnOnce(&Self) -> Result<Option<ConstrainedPosteriorCorrection>, String>,
    ) -> Result<Self, String> {
        let Self::Factorized {
            factor,
            dim,
            phi_scale,
            constrained_correction,
            smoothing_factor: previous,
            gauge_lift,
        } = self
        else {
            return Err(
                "a dense prediction covariance already carries its smoothing correction; only a \
                 factorized precision takes the correction's factor"
                    .to_string(),
            );
        };
        if constrained_correction.is_some() || previous.is_some() || gauge_lift.is_some() {
            return Err(
                "the smoothing correction's factor joins a conditional factorized precision \
                 before any truncation or gauge lift"
                    .to_string(),
            );
        }
        if smoothing_factor.nrows() != dim {
            return Err(format!(
                "the smoothing correction factor has {} rows but the factorized precision is \
                 {dim}x{dim}",
                smoothing_factor.nrows()
            ));
        }
        let mut corrected = Self::Factorized {
            factor,
            dim,
            phi_scale,
            constrained_correction: None,
            smoothing_factor: Some(smoothing_factor),
            gauge_lift: None,
        };
        let marginal_correction = truncation(&corrected)?;
        if let Self::Factorized {
            constrained_correction,
            ..
        } = &mut corrected
        {
            *constrained_correction = marginal_correction.map(Cow::Owned);
        }
        Ok(corrected)
    }

    /// The untruncated ambient covariance this backend's law is built from,
    /// applied on its ACTIVE coordinates: `φ·H⁻¹·rhs`, plus `B·(Bᵀ·rhs)` on a
    /// smoothing-corrected backend. No truncation and no gauge lift: this is
    /// the block `Σ·Aᵀ` a truncation's lift is derived from.
    pub fn apply_ambient_active(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        match self {
            Self::Dense(covariance) => Err(format!(
                "a dense {}x{} prediction covariance has no separate ambient law",
                covariance.nrows(),
                covariance.ncols()
            )),
            Self::Factorized {
                factor,
                dim,
                phi_scale,
                smoothing_factor,
                ..
            } => {
                if rhs.nrows() != *dim {
                    return Err(format!(
                        "ambient covariance application has {} rows, expected {dim}",
                        rhs.nrows()
                    ));
                }
                let mut solved = factor.solvemulti(rhs)?;
                if (*phi_scale - 1.0).abs() > 0.0 {
                    solved.mapv_inplace(|v| v * *phi_scale);
                }
                if let Some(smoothing) = smoothing_factor {
                    solved += &smoothing.dot(&smoothing.t().dot(rhs));
                }
                Ok(solved)
            }
        }
    }

    /// Carry a factorized active-coordinate precision to the saved coefficient
    /// frame through the coefficient gauge section `T` (#1561).
    pub fn with_gauge_lift(self, lift: ArrayView2<'a, f64>) -> Result<Self, String> {
        match self {
            Self::Dense(covariance) => Err(format!(
                "a dense {}x{} prediction covariance is already on the saved coefficients; \
                 only a factorized active-coordinate precision takes a gauge lift",
                covariance.nrows(),
                covariance.ncols()
            )),
            Self::Factorized {
                factor,
                dim,
                phi_scale,
                constrained_correction,
                smoothing_factor,
                ..
            } => {
                if lift.ncols() != dim {
                    return Err(format!(
                        "the coefficient gauge lifts {} active coordinates but the factorized \
                         precision is {dim}x{dim}",
                        lift.ncols()
                    ));
                }
                Ok(Self::Factorized {
                    factor,
                    dim,
                    phi_scale,
                    constrained_correction,
                    smoothing_factor,
                    gauge_lift: Some(lift),
                })
            }
        }
    }

    pub fn parameter_dim(&self) -> usize {
        match self {
            Self::Dense(covariance) => covariance.nrows(),
            Self::Factorized {
                dim, gauge_lift, ..
            } => gauge_lift.as_ref().map_or(*dim, |lift| lift.nrows()),
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(covariance) => covariance.nrows(),
            Self::Factorized { .. } => self.parameter_dim(),
        }
    }

    pub fn apply_columns(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        if rhs.nrows() != self.nrows() {
            return Err(format!(
                "prediction covariance backend column mismatch: rhs has {} rows, expected {}",
                rhs.nrows(),
                self.nrows()
            ));
        }
        match self {
            Self::Dense(covariance) => Ok(covariance.dot(rhs)),
            Self::Factorized {
                factor,
                phi_scale,
                constrained_correction,
                smoothing_factor,
                gauge_lift,
                ..
            } => {
                // Through a gauge the right-hand side is carried to the active
                // coordinates, solved and corrected there, and lifted back.
                let active_rhs = gauge_lift.as_ref().map(|lift| lift.t().dot(rhs));
                let rhs_active = active_rhs.as_ref().unwrap_or(rhs);
                let mut solved = factor.solvemulti(rhs_active)?;
                if (*phi_scale - 1.0).abs() > 0.0 {
                    solved.mapv_inplace(|v| v * *phi_scale);
                }
                if let Some(smoothing) = smoothing_factor {
                    solved += &smoothing.dot(&smoothing.t().dot(rhs_active));
                }
                if let Some(correction) = constrained_correction {
                    let normal_rhs = correction.lift.t().dot(rhs_active);
                    let removed = correction
                        .lift
                        .dot(&correction.removed_normal_variance.dot(&normal_rhs));
                    solved -= &removed;
                }
                Ok(match gauge_lift {
                    Some(lift) => lift.dot(&solved),
                    None => solved,
                })
            }
        }
    }
}

pub(crate) fn design_row_chunk(
    design: &DesignMatrix,
    rows: Range<usize>,
) -> Result<Array2<f64>, String> {
    if rows.end > design.nrows() || rows.start > rows.end {
        return Err(format!(
            "design_row_chunk row range {}..{} is out of bounds for {} rows",
            rows.start,
            rows.end,
            design.nrows()
        ));
    }
    design.try_row_chunk(rows).map_err(|e| e.to_string())
}

struct LocalCovarianceChunk {
    start: usize,
    end: usize,
    values: Vec<Vec<Array1<f64>>>,
}

fn compute_local_covariance_chunk(
    backend: &PredictionCovarianceBackend<'_>,
    rows: Range<usize>,
    local_dim: usize,
    gradients: Vec<Array2<f64>>,
) -> Result<LocalCovarianceChunk, String> {
    if gradients.len() != local_dim {
        return Err(format!(
            "rowwise_local_covariances chunk builder returned {} local components, expected {}",
            gradients.len(),
            local_dim
        ));
    }
    let parameter_dim = backend.nrows();
    let start = rows.start;
    let end = rows.end;
    let rows_in_chunk = end - start;

    // Pack all local-gradient blocks as a single multi-RHS solve:
    // rhs[:, component*R .. (component+1)*R] = gradients[component].t().
    // This amortizes dense matrix multiplies / factorized solves across the
    // whole chunk and avoids per-row RHS allocations.
    let mut rhs = Array2::<f64>::zeros((parameter_dim, rows_in_chunk * local_dim));
    for (component, grad) in gradients.iter().enumerate() {
        if grad.nrows() != rows_in_chunk || grad.ncols() != parameter_dim {
            return Err(format!(
                "rowwise_local_covariances component {component} has shape {}x{}, expected {}x{}",
                grad.nrows(),
                grad.ncols(),
                rows_in_chunk,
                parameter_dim
            ));
        }
        let col_start = component * rows_in_chunk;
        let col_end = col_start + rows_in_chunk;
        rhs.slice_mut(s![.., col_start..col_end]).assign(&grad.t());
    }

    let solved = backend.apply_columns(&rhs)?;
    if solved.nrows() != parameter_dim || solved.ncols() != rows_in_chunk * local_dim {
        return Err(format!(
            "rowwise_local_covariances backend returned {}x{}, expected {}x{}",
            solved.nrows(),
            solved.ncols(),
            parameter_dim,
            rows_in_chunk * local_dim
        ));
    }

    let mut values: Vec<Vec<Array1<f64>>> = (0..local_dim)
        .map(|_| {
            (0..local_dim)
                .map(|_| Array1::<f64>::zeros(rows_in_chunk))
                .collect::<Vec<_>>()
        })
        .collect();

    // Build the per-row local covariance entries. For each (a, b), row r is
    // gradients[a][r, :] · solved[:, b*R + r]. Off-diagonal entries are
    // symmetrized to preserve the previous factorized-backend round-off behavior.
    for a in 0..local_dim {
        let g_a = gradients[a].view();
        for b in a..local_dim {
            let s_b = solved.slice(s![.., b * rows_in_chunk..(b + 1) * rows_in_chunk]);
            if a == b {
                for local_row in 0..rows_in_chunk {
                    values[a][b][local_row] = g_a.row(local_row).dot(&s_b.column(local_row));
                }
            } else {
                let g_b = gradients[b].view();
                let s_a = solved.slice(s![.., a * rows_in_chunk..(a + 1) * rows_in_chunk]);
                for local_row in 0..rows_in_chunk {
                    let v_ab = g_a.row(local_row).dot(&s_b.column(local_row));
                    let v_ba = g_b.row(local_row).dot(&s_a.column(local_row));
                    let value = 0.5 * (v_ab + v_ba);
                    values[a][b][local_row] = value;
                    values[b][a][local_row] = value;
                }
            }
        }
    }

    Ok(LocalCovarianceChunk { start, end, values })
}

fn empty_local_covariance_output(n_rows: usize, local_dim: usize) -> Vec<Vec<Array1<f64>>> {
    (0..local_dim)
        .map(|_| {
            (0..local_dim)
                .map(|_| Array1::<f64>::zeros(n_rows))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn assemble_local_covariance_chunks(
    n_rows: usize,
    local_dim: usize,
    chunks: Vec<LocalCovarianceChunk>,
) -> Vec<Vec<Array1<f64>>> {
    let mut out = empty_local_covariance_output(n_rows, local_dim);
    for chunk in chunks {
        for a in 0..local_dim {
            for b in 0..local_dim {
                out[a][b]
                    .slice_mut(s![chunk.start..chunk.end])
                    .assign(&chunk.values[a][b]);
            }
        }
    }
    out
}

pub fn rowwise_local_covariances<F>(
    backend: &PredictionCovarianceBackend<'_>,
    n_rows: usize,
    local_dim: usize,
    mut build_chunk: F,
) -> Result<Vec<Vec<Array1<f64>>>, String>
where
    F: FnMut(Range<usize>) -> Result<Vec<Array2<f64>>, String>,
{
    if local_dim == 0 {
        return Err("rowwise_local_covariances requires local_dim > 0".to_string());
    }
    let parameter_dim = backend.nrows();
    let chunk_rows = prediction_chunk_rows(parameter_dim, local_dim, n_rows);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < n_rows {
        let end = (start + chunk_rows).min(n_rows);
        let rows = start..end;
        let gradients = build_chunk(rows.clone())?;
        chunks.push(compute_local_covariance_chunk(
            backend, rows, local_dim, gradients,
        )?);
        start = end;
    }
    Ok(assemble_local_covariance_chunks(n_rows, local_dim, chunks))
}

pub(crate) fn rowwise_local_covariances_parallel<F>(
    backend: &PredictionCovarianceBackend<'_>,
    n_rows: usize,
    local_dim: usize,
    build_chunk: F,
) -> Result<Vec<Vec<Array1<f64>>>, String>
where
    F: Fn(Range<usize>) -> Result<Vec<Array2<f64>>, String> + Sync,
{
    if local_dim == 0 {
        return Err("rowwise_local_covariances requires local_dim > 0".to_string());
    }
    let parameter_dim = backend.nrows();
    let chunk_rows = prediction_chunk_rows(parameter_dim, local_dim, n_rows);
    let n_chunks = n_rows.div_ceil(chunk_rows);
    let mut chunks = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_rows;
            let end = (start + chunk_rows).min(n_rows);
            let rows = start..end;
            let gradients = build_chunk(rows.clone())?;
            compute_local_covariance_chunk(backend, rows, local_dim, gradients)
        })
        .collect::<Result<Vec<_>, String>>()?;
    chunks.sort_by_key(|chunk| chunk.start);
    Ok(assemble_local_covariance_chunks(n_rows, local_dim, chunks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::array;

    fn sparse_design_from_dense(dense: &Array2<f64>) -> DesignMatrix {
        let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                let value = dense[[i, j]];
                if value != 0.0 {
                    triplets.push(Triplet::new(i, j, value));
                }
            }
        }
        let sparse = SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
            .expect("assemble sparse design");
        DesignMatrix::Sparse(gam_linalg::matrix::SparseDesignMatrix::new(sparse))
    }

    #[test]
    fn rowwise_local_covariances_match_dense_direct_formula() {
        let covariance = array![[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.1]];
        let backend = PredictionCovarianceBackend::from_dense(covariance.view());
        let grads0 = array![[1.0, 0.0, 2.0], [0.5, -1.0, 0.0], [0.0, 1.0, 1.0]];
        let grads1 = array![[0.0, 1.0, 1.0], [1.0, 0.5, -0.5], [2.0, 0.0, 0.5]];
        let out = rowwise_local_covariances(&backend, 3, 2, |rows| {
            Ok(vec![
                grads0.slice(s![rows.clone(), ..]).to_owned(),
                grads1.slice(s![rows, ..]).to_owned(),
            ])
        })
        .expect("chunked local covariances");

        for i in 0..3 {
            let g0 = grads0.row(i).to_owned();
            let g1 = grads1.row(i).to_owned();
            let expected00 = g0.dot(&covariance.dot(&g0));
            let expected01 = g0.dot(&covariance.dot(&g1));
            let expected11 = g1.dot(&covariance.dot(&g1));
            assert!((out[0][0][i] - expected00).abs() <= 1e-12);
            assert!((out[0][1][i] - expected01).abs() <= 1e-12);
            assert!((out[1][1][i] - expected11).abs() <= 1e-12);
        }
    }

    #[test]
    fn factorized_backend_applies_a_zero_scale_and_refuses_an_invalid_one() {
        let precision = array![[4.0, 0.6], [0.6, 3.0]];
        let rhs = array![[1.0, -0.5], [0.2, 2.0]];
        // The exact-fit profiled Gaussian publishes `Vb = 0 * H^{-1}` on the
        // dense route; the factorized route must agree instead of returning
        // the unscaled `H^{-1}`.
        let zero = PredictionCovarianceBackend::from_factorized_hessian_scaled(
            SymmetricMatrix::Dense(precision.clone()),
            0.0,
        )
        .expect("a zero coefficient-covariance scale is valid");
        let columns = zero.apply_columns(&rhs).expect("covariance columns");
        assert!(columns.iter().all(|&v| v == 0.0), "{columns:?}");
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let refusal = PredictionCovarianceBackend::from_factorized_hessian_scaled(
                SymmetricMatrix::Dense(precision.clone()),
                bad,
            );
            assert!(
                refusal.is_err(),
                "an invalid coefficient-covariance scale {bad} must be refused"
            );
        }
    }

    #[test]
    fn constrained_factorized_backend_matches_the_persisted_dense_moment() {
        let precision = array![[4.0, 0.6, 0.1], [0.6, 3.0, -0.2], [0.1, -0.2, 2.5]];
        let phi = 1.7;
        let factor = SymmetricMatrix::Dense(precision.clone())
            .factorize()
            .expect("factorize ambient precision");
        let mut dense = factor
            .solvemulti(&Array2::eye(3))
            .expect("ambient covariance");
        dense *= phi;
        let correction = ConstrainedPosteriorCorrection {
            lift: array![[0.3, -0.1], [0.2, 0.25], [-0.05, 0.4]],
            removed_normal_variance: array![[0.8, 0.1], [0.1, 0.5]],
            normal_mean_shift: array![0.2, 0.1],
            rows: vec![0, 1],
            normal_upper_limits: vec![f64::INFINITY, f64::INFINITY],
        };
        correction.apply_to_covariance_in_place(&mut dense);
        let backend = PredictionCovarianceBackend::from_factorized_hessian_scaled_with_correction(
            SymmetricMatrix::Dense(precision),
            phi,
            Some(&correction),
        )
        .expect("factorized constrained covariance");
        let rhs = array![[1.0, -0.5], [0.2, 2.0], [-0.7, 0.3]];
        let actual = backend.apply_columns(&rhs).expect("covariance columns");
        let expected = dense.dot(&rhs);
        for (&left, &right) in actual.iter().zip(expected.iter()) {
            assert!((left - right).abs() <= 2e-12, "{left} != {right}");
        }
    }

    #[test]
    fn gauge_lifted_factorized_backend_applies_the_saved_coefficient_covariance_1561() {
        // A precision on 3 active coordinates behind a section that lifts to 5 saved
        // coefficients: saved coefficient 1 was dropped (a zero row), and saved
        // coefficient 4 mixes all three active coordinates.
        let precision = array![[4.0, 0.6, 0.1], [0.6, 3.0, -0.2], [0.1, -0.2, 2.5]];
        let phi = 1.7;
        let lift = array![
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, -0.25, 0.75]
        ];
        let correction = ConstrainedPosteriorCorrection {
            lift: array![[0.3, -0.1], [0.2, 0.25], [-0.05, 0.4]],
            removed_normal_variance: array![[0.8, 0.1], [0.1, 0.5]],
            normal_mean_shift: array![0.2, 0.1],
            rows: vec![0, 1],
            normal_upper_limits: vec![f64::INFINITY, f64::INFINITY],
        };
        let factor = SymmetricMatrix::Dense(precision.clone())
            .factorize()
            .expect("factorize active precision");
        let mut active_covariance = factor
            .solvemulti(&Array2::eye(3))
            .expect("active covariance");
        active_covariance *= phi;
        correction.apply_to_covariance_in_place(&mut active_covariance);
        let saved_covariance = lift.dot(&active_covariance).dot(&lift.t());

        let backend = PredictionCovarianceBackend::from_factorized_hessian_scaled_with_correction(
            SymmetricMatrix::Dense(precision.clone()),
            phi,
            Some(&correction),
        )
        .expect("factorized constrained covariance")
        .with_gauge_lift(lift.view())
        .expect("a 5x3 section lifts a 3x3 precision");
        assert_eq!(backend.parameter_dim(), 5);
        assert_eq!(backend.nrows(), 5);
        let rhs = array![[1.0, -0.5], [0.2, 2.0], [-0.7, 0.3], [0.4, 0.0], [-1.1, 0.6]];
        let actual = backend.apply_columns(&rhs).expect("saved covariance columns");
        let expected = saved_covariance.dot(&rhs);
        for (&left, &right) in actual.iter().zip(expected.iter()) {
            assert!((left - right).abs() <= 2e-12, "{left} != {right}");
        }
        assert!(
            actual.row(1).iter().all(|value| value.abs() <= 2e-12),
            "the dropped saved coefficient carries no covariance"
        );

        let narrow = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]];
        let refusal = PredictionCovarianceBackend::from_factorized_hessian_scaled(
            SymmetricMatrix::Dense(precision),
            phi,
        )
        .expect("factorized covariance")
        .with_gauge_lift(narrow.view());
        assert!(
            refusal.is_err(),
            "a 5x2 section must not lift a 3x3 precision"
        );
        let dense = PredictionCovarianceBackend::from_dense(saved_covariance.view());
        assert!(
            dense.with_gauge_lift(lift.view()).is_err(),
            "a dense saved covariance takes no gauge lift"
        );
    }

    #[test]
    fn parallel_rowwise_local_covariances_match_serial_for_many_chunks() {
        let p = 96usize;
        let n = 1537usize;
        let covariance = Array2::from_shape_fn((p, p), |(i, j)| {
            if i == j {
                1.0 + (i as f64) * 0.001
            } else {
                0.0005 / (1.0 + i.abs_diff(j) as f64)
            }
        });
        let backend = PredictionCovarianceBackend::from_dense(covariance.view());
        let grads0 = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i % 17) as f64 - 8.0) * 0.01 + ((j % 11) as f64) * 0.002
        });
        let grads1 = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i % 13) as f64) * 0.003 - ((j % 7) as f64) * 0.004
        });

        let serial = rowwise_local_covariances(&backend, n, 2, |rows| {
            Ok(vec![
                grads0.slice(s![rows.clone(), ..]).to_owned(),
                grads1.slice(s![rows, ..]).to_owned(),
            ])
        })
        .expect("serial local covariances");
        let parallel = rowwise_local_covariances_parallel(&backend, n, 2, |rows| {
            Ok(vec![
                grads0.slice(s![rows.clone(), ..]).to_owned(),
                grads1.slice(s![rows, ..]).to_owned(),
            ])
        })
        .expect("parallel local covariances");

        for a in 0..2 {
            for b in 0..2 {
                for i in 0..n {
                    assert!(
                        (serial[a][b][i] - parallel[a][b][i]).abs() <= 1e-12,
                        "entry ({a},{b}) row {i}: serial={} parallel={}",
                        serial[a][b][i],
                        parallel[a][b][i]
                    );
                }
            }
        }
    }

    #[test]
    fn design_row_chunk_preserves_sparse_rows() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = sparse_design_from_dense(&dense);
        let chunk = design_row_chunk(&sparse, 1..3).expect("sparse row chunk");
        assert_eq!(chunk, dense.slice(s![1..3, ..]).to_owned());
    }
}
