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
    pub fn shape(&self) -> (usize, usize) {
        let d = self.dim();
        (d, d)
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
}
