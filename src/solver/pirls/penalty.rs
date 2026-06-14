use crate::construction::KroneckerReparamResult;
use crate::faer_ndarray::{fast_ab, fast_atb, fast_atv, fast_av};
use crate::matrix::DesignMatrix;
use faer::sparse::SparseRowMat;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Coordinate frame for PIRLS inner iteration.
pub(super) enum WorkingCoordinateDesign {
    OriginalSparseNative,
    TransformedExplicit {
        x_transformed: DesignMatrix,
        x_csr: Option<SparseRowMat<usize, f64>>,
    },
    TransformedImplicit {
        transform: WorkingReparamTransform,
    },
}

#[derive(Clone)]
pub(super) enum WorkingReparamTransform {
    Dense(Arc<Array2<f64>>),
    Kronecker(Arc<KroneckerQsTransform>),
}

impl WorkingReparamTransform {
    pub(super) fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(qs) => fast_av(qs.as_ref(), vector),
            Self::Kronecker(transform) => transform.apply(vector),
        }
    }

    pub(super) fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(qs) => fast_atv(qs, vector),
            Self::Kronecker(transform) => transform.apply_transpose(vector),
        }
    }

    pub(super) fn materialize_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(qs) => qs.as_ref().clone(),
            Self::Kronecker(transform) => transform.materialize(),
        }
    }

    pub(super) fn conjugate_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Dense(qs) => {
                let tmp = fast_atb(qs, matrix);
                symmetrize_dense_matrix(&fast_ab(&tmp, qs))
            }
            Self::Kronecker(transform) => transform.conjugate_matrix(matrix),
        }
    }
}

#[derive(Clone)]
pub(super) enum PirlsPenalty {
    Dense {
        s_transformed: Array2<f64>,
        e_transformed: Array2<f64>,
        linear_shift: Array1<f64>,
        constant_shift: f64,
        /// Aggregated prior-mean target `μ` in *transformed* coordinates,
        /// summed over the canonical penalties' `full_width_prior_mean()`.
        /// Used to keep the fixed stabilization ridge `δI` (and other PSD
        /// rescue ridges) from biasing the recovered β away from the prior
        /// mean: any site that adds `δI` to the penalized Hessian must also
        /// add `δ · prior_mean_target` to the RHS so the augmented system
        /// `(H + δI) β = r + δμ` keeps `β = μ` exact when the data has no
        /// pull (X'WX = 0, X'Wz = 0). When all blocks have zero prior, this
        /// vector is all zero and the RHS shift is a no-op.
        prior_mean_target: Array1<f64>,
    },
    Diagonal {
        diag: Array1<f64>,
        positive_indices: Vec<usize>,
        linear_shift: Array1<f64>,
        constant_shift: f64,
        /// See `Dense::prior_mean_target`.
        prior_mean_target: Array1<f64>,
    },
}

impl PirlsPenalty {
    pub(super) fn dim(&self) -> usize {
        match self {
            Self::Dense { s_transformed, .. } => s_transformed.ncols(),
            Self::Diagonal { diag, .. } => diag.len(),
        }
    }

    pub(super) fn rank(&self) -> usize {
        match self {
            Self::Dense { e_transformed, .. } => e_transformed.nrows(),
            Self::Diagonal {
                positive_indices, ..
            } => positive_indices.len(),
        }
    }

    pub(super) fn add_to_hessian(&self, hessian: &mut Array2<f64>) {
        match self {
            Self::Dense { s_transformed, .. } => {
                *hessian += s_transformed;
            }
            Self::Diagonal { diag, .. } => {
                for i in 0..diag.len() {
                    hessian[[i, i]] += diag[i];
                }
            }
        }
    }

    pub(super) fn apply(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense { s_transformed, .. } => crate::faer_ndarray::fast_av(s_transformed, beta),
            Self::Diagonal { diag, .. } => diag * beta,
        }
    }

    pub(super) fn linear_shift(&self) -> &Array1<f64> {
        match self {
            Self::Dense { linear_shift, .. } | Self::Diagonal { linear_shift, .. } => linear_shift,
        }
    }

    /// Prior-mean target `μ` in transformed coordinates (see field docs on
    /// the [`PirlsPenalty::Dense::prior_mean_target`] variant). The returned
    /// slice has length `dim()`.
    pub(super) fn prior_mean_target(&self) -> &Array1<f64> {
        match self {
            Self::Dense {
                prior_mean_target, ..
            }
            | Self::Diagonal {
                prior_mean_target, ..
            } => prior_mean_target,
        }
    }

    pub(super) fn constant_shift(&self) -> f64 {
        match self {
            Self::Dense { constant_shift, .. } | Self::Diagonal { constant_shift, .. } => {
                *constant_shift
            }
        }
    }

    pub(super) fn shifted_gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut value = self.apply(beta);
        value -= self.linear_shift();
        value
    }

    pub(super) fn shifted_quadratic(&self, beta: &Array1<f64>) -> f64 {
        let s_beta = self.apply(beta);
        beta.dot(&s_beta) - 2.0 * beta.dot(self.linear_shift()) + self.constant_shift()
    }
}

#[derive(Clone)]
pub(super) struct KroneckerQsTransform {
    pub(super) marginal_qs: Vec<Array2<f64>>,
    pub(super) dims: Vec<usize>,
    pub(super) p: usize,
}

impl KroneckerQsTransform {
    pub(super) fn new(result: &KroneckerReparamResult) -> Self {
        let dims = result.marginal_dims.clone();
        let p = dims.iter().product();
        Self {
            marginal_qs: result.marginal_qs.clone(),
            dims,
            p,
        }
    }

    pub(super) fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_internal(vector, false)
    }

    pub(super) fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_internal(vector, true)
    }

    pub(crate) fn apply_internal(&self, vector: &Array1<f64>, transpose: bool) -> Array1<f64> {
        assert_eq!(vector.len(), self.p);
        // Ping-pong two thread-local scratch buffers across axes so we
        // allocate at most twice per thread for the whole solver lifetime
        // instead of once per `apply` call per axis.
        kron_apply_scratch::with(|scratch| {
            let (front, back) = scratch.pair_with_capacity(self.p);
            front.clear();
            front.extend_from_slice(vector.as_slice().expect("Array1 must be contiguous"));
            for (axis, q) in self.marginal_qs.iter().enumerate() {
                back.clear();
                back.resize(front.len(), 0.0);
                apply_kron_mode_into(front, &self.dims, axis, q, transpose, back);
                std::mem::swap(front, back);
            }
            // Clone out the final result (one allocation per `apply`, vs. the
            // previous N+1 allocations across N axes); the scratch retains
            // its capacity for the next call on this thread.
            Array1::from(front.clone())
        })
    }

    pub(super) fn materialize(&self) -> Array2<f64> {
        let mut qs = Array2::<f64>::zeros((self.p, self.p));
        for j in 0..self.p {
            let mut e = Array1::<f64>::zeros(self.p);
            e[j] = 1.0;
            let col = self.apply(&e);
            qs.column_mut(j).assign(&col);
        }
        qs
    }

    pub(super) fn conjugate_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let p = self.p;
        let mut right = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let col = fast_av(matrix, &self.column(j));
            right.column_mut(j).assign(&col);
        }
        let mut out = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let transformed_col = self.apply_transpose(&right.column(j).to_owned());
            out.column_mut(j).assign(&transformed_col);
        }
        symmetrize_dense_matrix(&out)
    }

    pub(crate) fn column(&self, j: usize) -> Array1<f64> {
        let mut e = Array1::<f64>::zeros(self.p);
        e[j] = 1.0;
        self.apply(&e)
    }
}

#[inline]
pub(super) fn symmetrize_dense_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t().to_owned()) * 0.5
}

pub(super) fn apply_kron_mode_into(
    data: &[f64],
    dims: &[usize],
    axis: usize,
    q: &Array2<f64>,
    transpose: bool,
    out: &mut [f64],
) {
    let before: usize = dims[..axis].iter().product();
    let dim = dims[axis];
    let after: usize = dims[axis + 1..].iter().product();
    assert_eq!(out.len(), data.len());
    for b in 0..before {
        for s in 0..after {
            for i in 0..dim {
                let mut acc = 0.0;
                for a in 0..dim {
                    let coeff = if transpose { q[[a, i]] } else { q[[i, a]] };
                    acc += coeff * data[(b * dim + a) * after + s];
                }
                out[(b * dim + i) * after + s] = acc;
            }
        }
    }
}

/// Attach a penalty shift (prior-mean correction) to an existing PirlsPenalty.
pub(super) fn attach_penalty_shift(
    penalty: &mut PirlsPenalty,
    linear_shift: Array1<f64>,
    constant_shift: f64,
    prior_mean_target: Array1<f64>,
) {
    match penalty {
        PirlsPenalty::Dense {
            linear_shift: target,
            constant_shift: constant,
            prior_mean_target: mean_target,
            ..
        }
        | PirlsPenalty::Diagonal {
            linear_shift: target,
            constant_shift: constant,
            prior_mean_target: mean_target,
            ..
        } => {
            *target = linear_shift;
            *constant = constant_shift;
            *mean_target = prior_mean_target;
        }
    }
}

/// Thread-local ping-pong scratch buffers for Kronecker mode application.
/// Sized lazily to the largest p ever seen on this thread.
pub(super) mod kron_apply_scratch {
    use std::cell::RefCell;

    thread_local! {
        pub(crate) static SCRATCH: RefCell<Pair> = const { RefCell::new(Pair::new()) };
    }

    pub(super) struct Pair {
        a: Vec<f64>,
        b: Vec<f64>,
    }

    impl Pair {
        pub(super) const fn new() -> Self {
            Self {
                a: Vec::new(),
                b: Vec::new(),
            }
        }

        pub(super) fn pair_with_capacity(
            &mut self,
            capacity: usize,
        ) -> (&mut Vec<f64>, &mut Vec<f64>) {
            if self.a.capacity() < capacity {
                self.a.reserve(capacity - self.a.capacity());
            }
            if self.b.capacity() < capacity {
                self.b.reserve(capacity - self.b.capacity());
            }
            (&mut self.a, &mut self.b)
        }
    }

    pub(super) fn with<R>(f: impl FnOnce(&mut Pair) -> R) -> R {
        SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
    }
}
