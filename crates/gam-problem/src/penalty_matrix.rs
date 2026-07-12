//! The `PenaltyMatrix` carrier (dense / Kronecker / scaled) used by every
//! custom-family block, plus its constructors and the `Array2` conversion.

use ndarray::{Array1, Array2, Axis};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// A penalty matrix that may be stored in Kronecker-factored form.
///
/// For tensor-product terms (e.g. time-varying survival covariates), the penalty
/// has the structure `S = left ⊗ right` (Kronecker product). Keeping this
/// factored avoids materializing (p_left × p_right)² dense entries and enables
/// exact log-determinant computation via `log|A ⊗ B| = n_B log|A| + n_A log|B|`.
///
/// Dense penalties are stored as-is.  Callers that need a raw `Array2<f64>` can
/// call `as_dense()` (zero-cost for Dense, lazy-materialized for KroneckerFactored).
#[derive(Clone, Debug)]
pub enum PenaltyMatrix {
    Dense(Array2<f64>),
    KroneckerFactored {
        left: Array2<f64>,
        right: Array2<f64>,
    },
    /// Block-local penalty: `local` is `block_dim × block_dim`, embedded at
    /// `col_range` in the full parameter space of dimension `total_dim`.
    /// Avoids materializing the full `total_dim × total_dim` matrix.
    Blockwise {
        local: Array2<f64>,
        col_range: std::ops::Range<usize>,
        total_dim: usize,
    },
    /// Wrapper assigning this penalty component to a user-visible precision
    /// label. Components with the same label share one smoothing parameter.
    Labeled {
        label: String,
        inner: Box<PenaltyMatrix>,
    },
    /// Wrapper fixing this penalty component at a physical log-precision.
    /// Fixed components remain in the block-local physical penalty layout but
    /// are removed from the REML outer coordinate vector.
    Fixed {
        log_lambda: f64,
        inner: Box<PenaltyMatrix>,
    },
}

impl PenaltyMatrix {
    /// Number of rows (= number of columns, since penalties are square).
    pub fn dim(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::KroneckerFactored { left, right } => left.nrows() * right.nrows(),
            Self::Blockwise { total_dim, .. } => *total_dim,
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.dim(),
        }
    }

    /// Returns (nrows, ncols) like Array2::dim().
    ///
    /// Reports the ACTUAL storage shape, not `(dim(), dim())`: a malformed
    /// non-square carrier must be visible to validation instead of being
    /// laundered into a fabricated square shape.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Dense(m) => m.dim(),
            Self::KroneckerFactored { left, right } => {
                (left.nrows() * right.nrows(), left.ncols() * right.ncols())
            }
            Self::Blockwise { total_dim, .. } => (*total_dim, *total_dim),
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.shape(),
        }
    }

    /// Validate this penalty as the carrier of a quadratic form `½·λ·βᵀSβ`
    /// on a coefficient block of width `expected_dim`.
    ///
    /// Establishes, at the model boundary, everything downstream code assumes
    /// without re-checking: square storage of the right size, finite entries,
    /// symmetry, and positive semidefiniteness (for `Blockwise`, also that the
    /// embedded range is consistent). A nonsymmetric `S` would make the
    /// implemented gradient `λSβ` disagree with the true gradient of the
    /// quadratic, `λ·sym(S)β`; an indefinite `S` makes the penalized objective
    /// unbounded below along its negative mode while positive-eigenspace
    /// filtering silently drops that mode from ranks and log-determinants.
    /// Neither is a fittable model, so both are rejected rather than coerced.
    pub fn validate(&self, expected_dim: usize) -> Result<(), String> {
        let (nrows, ncols) = self.shape();
        if nrows != ncols || nrows != expected_dim {
            return Err(format!(
                "penalty must be {expected_dim}x{expected_dim}, got {nrows}x{ncols}"
            ));
        }
        match self {
            Self::Dense(m) => validate_symmetric_psd_core(m, "dense penalty"),
            Self::KroneckerFactored { left, right } => {
                // A ⊗ B is symmetric PSD when both factors are (the canonical
                // tensor-product construction). Validating the factors avoids
                // materializing the product and rejects the NSD⊗NSD encoding,
                // which downstream Kronecker logdet identities do not support.
                validate_symmetric_psd_core(left, "Kronecker left factor")?;
                validate_symmetric_psd_core(right, "Kronecker right factor")
            }
            Self::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                if col_range.end > *total_dim || col_range.len() != local.nrows() {
                    return Err(format!(
                        "blockwise penalty embedding is inconsistent: local {}x{} at columns \
                         {}..{} of total_dim {}",
                        local.nrows(),
                        local.ncols(),
                        col_range.start,
                        col_range.end,
                        total_dim
                    ));
                }
                validate_symmetric_psd_core(local, "blockwise local penalty")
            }
            Self::Labeled { inner, .. } => inner.validate(expected_dim),
            Self::Fixed { log_lambda, inner } => {
                crate::validate_log_strength(*log_lambda)
                    .map_err(|error| format!("fixed penalty log-precision: {error}"))?;
                inner.validate(expected_dim)
            }
        }
    }

    /// Materialize the full dense matrix.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(m) => m.clone(),
            Self::KroneckerFactored { left, right } => kronecker_product(left, right),
            Self::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                let mut g = Array2::zeros((*total_dim, *total_dim));
                g.slice_mut(ndarray::s![
                    col_range.start..col_range.end,
                    col_range.start..col_range.end
                ])
                .assign(local);
                g
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.to_dense(),
        }
    }

    /// Borrow the inner dense matrix if Dense, otherwise materialize.
    pub fn as_dense_cow(&self) -> std::borrow::Cow<'_, Array2<f64>> {
        match self {
            Self::Dense(m) => std::borrow::Cow::Borrowed(m),
            Self::KroneckerFactored { .. }
            | Self::Blockwise { .. }
            | Self::Labeled { .. }
            | Self::Fixed { .. } => std::borrow::Cow::Owned(self.to_dense()),
        }
    }

    /// Returns a reference to the inner matrix if this is a Dense variant.
    pub fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(m) => Some(m),
            Self::Fixed { inner, .. } => inner.as_dense_ref(),
            Self::KroneckerFactored { .. } | Self::Blockwise { .. } | Self::Labeled { .. } => None,
        }
    }

    pub fn with_precision_label(self, label: impl Into<String>) -> Self {
        Self::Labeled {
            label: label.into(),
            inner: Box::new(self),
        }
    }

    pub fn precision_label(&self) -> Option<&str> {
        match self {
            Self::Labeled { label, .. } => Some(label.as_str()),
            Self::Fixed { .. } => None,
            _ => None,
        }
    }

    pub fn with_fixed_log_lambda(self, log_lambda: f64) -> Self {
        Self::Fixed {
            log_lambda,
            inner: Box::new(self),
        }
    }

    pub fn fixed_log_lambda(&self) -> Option<f64> {
        match self {
            Self::Fixed { log_lambda, .. } => Some(*log_lambda),
            Self::Labeled { inner, .. } => inner.fixed_log_lambda(),
            _ => None,
        }
    }

    /// Compute S * v using the row-major Kronecker vec trick when factored:
    ///   (A ⊗ B) vec_rm(V) = vec_rm(A V Bᵀ)
    /// where V = reshape(v, (p_left, p_right)).
    pub fn dot(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(m) => m.dot(v),
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                // v is ordered by i_left * p_right + i_right.
                let v_mat =
                    ndarray::ArrayView2::from_shape((p_left, p_right), v.as_slice().unwrap())
                        .unwrap();
                let avbt = left.dot(&v_mat).dot(&right.t());
                let standard = avbt.as_standard_layout();
                Array1::from_iter(standard.iter().copied())
            }
            Self::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                let mut out = Array1::zeros(*total_dim);
                let v_block = v.slice(ndarray::s![col_range.clone()]);
                let result_block = local.dot(&v_block);
                out.slice_mut(ndarray::s![col_range.clone()])
                    .assign(&result_block);
                out
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.dot(v),
        }
    }

    /// Add λ * self to a mutable dense accumulator.
    pub fn add_scaled_to(&self, lambda: f64, target: &mut Array2<f64>) {
        match self {
            Self::Dense(m) => {
                target.scaled_add(lambda, m);
            }
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                for i1 in 0..p_left {
                    for j1 in 0..p_left {
                        let a_ij = left[[i1, j1]];
                        if a_ij == 0.0 {
                            continue;
                        }
                        let scaled_a = lambda * a_ij;
                        for i2 in 0..p_right {
                            let row = i1 * p_right + i2;
                            for j2 in 0..p_right {
                                let col = j1 * p_right + j2;
                                target[[row, col]] += scaled_a * right[[i2, j2]];
                            }
                        }
                    }
                }
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                target
                    .slice_mut(ndarray::s![col_range.clone(), col_range.clone()])
                    .scaled_add(lambda, local);
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => {
                inner.add_scaled_to(lambda, target)
            }
        }
    }

    /// Add λ * diag(self) to a mutable diagonal accumulator.
    pub fn add_scaled_diag_to(&self, lambda: f64, target: &mut Array1<f64>) {
        match self {
            Self::Dense(m) => {
                let p = m.nrows().min(target.len());
                for j in 0..p {
                    target[j] += lambda * m[[j, j]];
                }
            }
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                assert_eq!(target.len(), p_left * p_right);
                for i_left in 0..p_left {
                    let left_diag = left[[i_left, i_left]];
                    if left_diag == 0.0 {
                        continue;
                    }
                    let scaled_left = lambda * left_diag;
                    for i_right in 0..p_right {
                        target[i_left * p_right + i_right] +=
                            scaled_left * right[[i_right, i_right]];
                    }
                }
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                let width = local.nrows().min(col_range.len());
                for local_idx in 0..width {
                    target[col_range.start + local_idx] += lambda * local[[local_idx, local_idx]];
                }
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => {
                inner.add_scaled_diag_to(lambda, target)
            }
        }
    }

    /// Compute the quadratic form β' S β.
    pub fn quadratic_form(&self, beta: &Array1<f64>) -> f64 {
        match self {
            Self::Dense(m) => beta.dot(&m.dot(beta)),
            Self::KroneckerFactored { .. } => {
                let sv = self.dot(beta);
                beta.dot(&sv)
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                let beta_block = beta.slice(ndarray::s![col_range.clone()]);
                let sv = local.dot(&beta_block);
                beta_block.dot(&sv)
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.quadratic_form(beta),
        }
    }

    /// Access dimensions like an Array2.
    pub fn nrows(&self) -> usize {
        self.dim()
    }

    pub fn ncols(&self) -> usize {
        self.dim()
    }
}

impl From<Array2<f64>> for PenaltyMatrix {
    fn from(m: Array2<f64>) -> Self {
        Self::Dense(m)
    }
}

/// Core quadratic-form validity: square, finite, symmetric (up to a
/// scale-relative round-off band), and positive semidefinite (eigenvalues
/// above the relative eigensolver noise floor `p·ε·‖S‖`, the same relative
/// classification the REML pseudo-logdet kernel uses — never an absolute
/// floor, so validity is invariant under `S → c·S`).
fn validate_symmetric_psd_core(matrix: &Array2<f64>, what: &str) -> Result<(), String> {
    use gam_linalg::faer_ndarray::FaerEigh;

    let (nrows, ncols) = matrix.dim();
    if nrows != ncols {
        return Err(format!("{what} is not square: {nrows}x{ncols}"));
    }
    let mut max_abs = 0.0_f64;
    for ((row, col), &value) in matrix.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "{what} has non-finite entry at ({row},{col}): {value}"
            ));
        }
        max_abs = max_abs.max(value.abs());
    }
    // Symmetry: relative to the matrix scale so a legitimately large penalty
    // is not rejected for round-off and a small one cannot hide genuine skew.
    let sym_tol = 1e-10 * max_abs.max(1.0);
    for row in 0..nrows {
        for col in (row + 1)..ncols {
            let asymmetry = (matrix[[row, col]] - matrix[[col, row]]).abs();
            if asymmetry > sym_tol {
                return Err(format!(
                    "{what} is not symmetric at ({row},{col}): |S - Sᵀ| = {asymmetry:.3e}; \
                     the gradient of βᵀSβ/2 is sym(S)β, so a skew component would make the \
                     implemented objective and gradient describe different functions"
                ));
            }
        }
    }
    if nrows == 0 || max_abs == 0.0 {
        return Ok(()); // the zero penalty is trivially PSD
    }
    let (eigenvalues, _) = matrix
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("{what} eigendecomposition failed during validation: {e}"))?;
    let max_abs_eval = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let psd_tol = 100.0 * (nrows as f64) * f64::EPSILON * max_abs_eval;
    if let Some(&min_eval) = eigenvalues
        .iter()
        .filter(|&&ev| ev < -psd_tol)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
        return Err(format!(
            "{what} is not positive semidefinite: min eigenvalue {min_eval:.6e} \
             (max |eigenvalue| {max_abs_eval:.6e}); the penalized objective is unbounded \
             below along the negative mode while rank/logdet filtering would silently \
             drop it"
        ));
    }
    Ok(())
}

/// Computes the Kronecker product A ⊗ B for penalty matrix construction.
/// This is used to create tensor product penalties that enforce smoothness
/// in multiple dimensions for interaction terms.
fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (arows, a_cols) = a.dim();
    let (brows, b_cols) = b.dim();
    if arows == 0 || a_cols == 0 || brows == 0 || b_cols == 0 {
        return Array2::zeros((arows * brows, a_cols * b_cols));
    }
    let mut result = Array2::zeros((arows * brows, a_cols * b_cols));

    result
        .axis_chunks_iter_mut(Axis(0), brows)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row_block)| {
            let arow = a.row(i);
            let col_chunks = row_block.axis_chunks_iter_mut(Axis(1), b_cols);
            for (j, mut block) in col_chunks.into_iter().enumerate() {
                let aval = arow[j];
                if aval == 0.0 {
                    continue;
                }
                for (dest, &src) in block.iter_mut().zip(b.iter()) {
                    *dest = aval * src;
                }
            }
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── Dense variant ─────────────────────────────────────────────────────────

    #[test]
    fn dense_dim_and_shape() {
        let m = array![[1.0, 0.0], [0.0, 2.0]];
        let p = PenaltyMatrix::Dense(m);
        assert_eq!(p.dim(), 2);
        assert_eq!(p.shape(), (2, 2));
        assert_eq!(p.nrows(), 2);
        assert_eq!(p.ncols(), 2);
    }

    #[test]
    fn dense_to_dense_is_clone() {
        let m = array![[3.0, 1.0], [1.0, 4.0]];
        let p = PenaltyMatrix::Dense(m.clone());
        assert_eq!(p.to_dense(), m);
    }

    #[test]
    fn dense_dot_product() {
        // [[1, 0], [0, 2]] · [3, 5] = [3, 10]
        let m = array![[1.0, 0.0], [0.0, 2.0]];
        let p = PenaltyMatrix::Dense(m);
        let v = ndarray::array![3.0, 5.0];
        let result = p.dot(&v);
        assert_eq!(result.as_slice().unwrap(), &[3.0, 10.0]);
    }

    #[test]
    fn dense_quadratic_form() {
        // beta' S beta with S=diag(1,2), beta=[3,2] → 9 + 8 = 17
        let m = array![[1.0, 0.0], [0.0, 2.0]];
        let p = PenaltyMatrix::Dense(m);
        let beta = ndarray::array![3.0, 2.0];
        assert!((p.quadratic_form(&beta) - 17.0).abs() < 1e-14);
    }

    #[test]
    fn dense_add_scaled_to() {
        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let p = PenaltyMatrix::Dense(s);
        let mut acc = ndarray::Array2::<f64>::zeros((2, 2));
        p.add_scaled_to(3.0, &mut acc);
        assert_eq!(acc, array![[3.0, 0.0], [0.0, 3.0]]);
    }

    #[test]
    fn dense_add_scaled_diag_to() {
        let s = array![[2.0, 5.0], [5.0, 7.0]];
        let p = PenaltyMatrix::Dense(s);
        let mut diag = ndarray::array![0.0, 0.0];
        p.add_scaled_diag_to(1.0, &mut diag);
        // diagonal entries are 2.0 and 7.0
        assert_eq!(diag.as_slice().unwrap(), &[2.0, 7.0]);
    }

    // ── KroneckerFactored variant ─────────────────────────────────────────────

    #[test]
    fn kronecker_dim_is_product() {
        let left = array![[1.0, 0.0], [0.0, 1.0]]; // 2×2
        let right = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // 3×3
        let p = PenaltyMatrix::KroneckerFactored { left, right };
        assert_eq!(p.dim(), 6);
    }

    #[test]
    fn kronecker_to_dense_identity_x_identity() {
        // I_2 ⊗ I_2 = I_4
        let eye2 = ndarray::Array2::<f64>::eye(2);
        let p = PenaltyMatrix::KroneckerFactored {
            left: eye2.clone(),
            right: eye2,
        };
        let dense = p.to_dense();
        assert_eq!(dense, ndarray::Array2::<f64>::eye(4));
    }

    #[test]
    fn kronecker_dot_matches_dense_dot() {
        let left = array![[2.0, 0.0], [0.0, 3.0]];
        let right = array![[1.0, 1.0], [0.0, 1.0]];
        let p = PenaltyMatrix::KroneckerFactored {
            left: left.clone(),
            right: right.clone(),
        };
        // Compare to materialised version
        let dense = p.to_dense();
        let v = ndarray::array![1.0, 2.0, 3.0, 4.0];
        let got = p.dot(&v);
        let expected = dense.dot(&v);
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-14, "got={a} expected={b}");
        }
    }

    // ── Blockwise variant ─────────────────────────────────────────────────────

    #[test]
    fn blockwise_dim_is_total() {
        let local = array![[1.0, 0.0], [0.0, 1.0]];
        let p = PenaltyMatrix::Blockwise {
            local,
            col_range: 1..3,
            total_dim: 5,
        };
        assert_eq!(p.dim(), 5);
    }

    #[test]
    fn blockwise_to_dense_embeds_local_block() {
        // 3×3 total with local 2×2 at cols 1..3
        let local = array![[2.0, 1.0], [1.0, 3.0]];
        let p = PenaltyMatrix::Blockwise {
            local,
            col_range: 1..3,
            total_dim: 3,
        };
        let dense = p.to_dense();
        assert_eq!(dense[[0, 0]], 0.0);
        assert_eq!(dense[[1, 1]], 2.0);
        assert_eq!(dense[[1, 2]], 1.0);
        assert_eq!(dense[[2, 1]], 1.0);
        assert_eq!(dense[[2, 2]], 3.0);
    }

    #[test]
    fn blockwise_dot_only_touches_block() {
        let local = array![[2.0, 0.0], [0.0, 3.0]];
        let p = PenaltyMatrix::Blockwise {
            local,
            col_range: 1..3,
            total_dim: 4,
        };
        let v = ndarray::array![7.0, 1.0, 2.0, 9.0];
        let out = p.dot(&v);
        // v[1..3] = [1,2]; local * [1,2] = [2,6]; embedded at positions 1..3
        assert_eq!(out.as_slice().unwrap(), &[0.0, 2.0, 6.0, 0.0]);
    }

    // ── Labeled / Fixed wrappers ──────────────────────────────────────────────

    #[test]
    fn labeled_inherits_dim_and_delegates_dot() {
        let m = array![[1.0, 0.0], [0.0, 2.0]];
        let p = PenaltyMatrix::Dense(m).with_precision_label("smooth");
        assert_eq!(p.dim(), 2);
        assert_eq!(p.precision_label(), Some("smooth"));
        let v = ndarray::array![3.0, 4.0];
        let out = p.dot(&v);
        assert_eq!(out.as_slice().unwrap(), &[3.0, 8.0]);
    }

    #[test]
    fn fixed_inherits_dim_and_exposes_log_lambda() {
        let m = array![[5.0, 0.0], [0.0, 5.0]];
        let p = PenaltyMatrix::Dense(m).with_fixed_log_lambda(2.5);
        assert_eq!(p.dim(), 2);
        assert_eq!(p.fixed_log_lambda(), Some(2.5));
    }

    // ── Boundary validation ───────────────────────────────────────────────────

    #[test]
    fn shape_reports_actual_storage_not_fabricated_square() {
        // A malformed 2x3 dense carrier must be visible as 2x3, not laundered
        // into 2x2 through dim().
        let p = PenaltyMatrix::Dense(Array2::<f64>::zeros((2, 3)));
        assert_eq!(p.shape(), (2, 3));
        assert!(p.validate(2).is_err());
        assert!(p.validate(3).is_err());
    }

    #[test]
    fn validate_accepts_canonical_carriers() {
        let dense = PenaltyMatrix::Dense(array![[2.0, -1.0], [-1.0, 2.0]]);
        assert_eq!(dense.validate(2), Ok(()));

        let kron = PenaltyMatrix::KroneckerFactored {
            left: array![[1.0, -1.0], [-1.0, 1.0]],
            right: ndarray::Array2::<f64>::eye(3),
        };
        assert_eq!(kron.validate(6), Ok(()));

        let blockwise = PenaltyMatrix::Blockwise {
            local: array![[1.0, 0.0], [0.0, 1.0]],
            col_range: 1..3,
            total_dim: 4,
        };
        assert_eq!(blockwise.validate(4), Ok(()));
    }

    #[test]
    fn validate_rejects_asymmetric_indefinite_and_nonfinite() {
        // Nonsymmetric: gradient of βᵀSβ/2 is sym(S)β, not Sβ.
        let skew = PenaltyMatrix::Dense(array![[1.0, 1.0], [0.0, 1.0]]);
        assert!(skew.validate(2).unwrap_err().contains("not symmetric"));

        // Indefinite: objective unbounded below along the negative mode.
        let indefinite = PenaltyMatrix::Dense(array![[1.0, 0.0], [0.0, -1.0]]);
        assert!(
            indefinite
                .validate(2)
                .unwrap_err()
                .contains("not positive semidefinite")
        );

        let nan = PenaltyMatrix::Dense(array![[f64::NAN, 0.0], [0.0, 1.0]]);
        assert!(nan.validate(2).unwrap_err().contains("non-finite"));

        // Fixed wrapper must carry a supported physical log-precision.
        let bad_fixed = PenaltyMatrix::Dense(ndarray::Array2::<f64>::eye(2))
            .with_fixed_log_lambda(f64::INFINITY);
        assert!(bad_fixed.validate(2).unwrap_err().contains("must be finite"));

        let finite_but_out_of_domain = PenaltyMatrix::Dense(ndarray::Array2::<f64>::eye(2))
            .with_fixed_log_lambda(crate::LOG_STRENGTH_MAX + 1.0);
        assert!(
            finite_but_out_of_domain
                .validate(2)
                .unwrap_err()
                .contains("must be finite and in")
        );
    }

    #[test]
    fn validate_rejects_inconsistent_blockwise_embedding() {
        // local width disagrees with the embedded column range.
        let p = PenaltyMatrix::Blockwise {
            local: ndarray::Array2::<f64>::eye(3),
            col_range: 1..3,
            total_dim: 4,
        };
        assert!(p.validate(4).is_err());
        // range runs past total_dim.
        let q = PenaltyMatrix::Blockwise {
            local: ndarray::Array2::<f64>::eye(2),
            col_range: 3..5,
            total_dim: 4,
        };
        assert!(q.validate(4).is_err());
    }
}
