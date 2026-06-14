//! The `PenaltyMatrix` carrier (dense / Kronecker / scaled) used by every
//! custom-family block, plus its constructors and the `Array2` conversion.

use super::*;

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
            Self::KroneckerFactored { left, right } => {
                crate::terms::construction::kronecker_product(left, right)
            }
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
