use ahash::AHasher;
use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagMut, DiagRef};
use faer::linalg::cholesky::lblt::factor::{self, LbltParams, PivotingStrategy};
use faer::linalg::solvers::{self, Solve};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::{Auto, Mat, MatMut, MatRef, Side, Spec, get_global_parallelism};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use std::hash::Hasher;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FaerLinalgError {
    #[error("SVD failed to converge")]
    SvdNoConvergence,
    #[error("Self-adjoint eigendecomposition failed: {0:?}")]
    SelfAdjointEigen(solvers::EvdError),
    #[error("Cholesky factorization failed: {0:?}")]
    Cholesky(solvers::LltError),
    #[error("LDLT factorization failed: {0:?}")]
    Ldlt(solvers::LdltError),
}

#[inline]
pub fn array2_to_mat_mut(array: &mut Array2<f64>) -> MatMut<'_, f64> {
    assert!(
        array.is_standard_layout(),
        "array2_to_mat_mut expects standard layout storage"
    );
    let (nrows, ncols) = array.dim();
    let slice = array
        .as_slice_memory_order_mut()
        .expect("standard-layout array must expose a contiguous slice");
    MatMut::from_row_major_slice_mut(slice, nrows, ncols)
}

#[inline]
pub fn array1_to_col_mat_mut(array: &mut Array1<f64>) -> MatMut<'_, f64> {
    let len = array.len();
    let slice = array
        .as_slice_memory_order_mut()
        .expect("vector must expose a contiguous slice");
    MatMut::from_row_major_slice_mut(slice, len, 1)
}

#[inline]
pub fn hash_array2(matrix: &Array2<f64>) -> u64 {
    let mut hasher = AHasher::default();
    hasher.write_usize(matrix.nrows());
    hasher.write_usize(matrix.ncols());
    for value in matrix.iter() {
        hasher.write(&value.to_ne_bytes());
    }
    hasher.finish()
}

fn mat_to_array(mat: MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(i, j)| mat[(i, j)])
}

fn diag_to_array(diag: DiagRef<'_, f64>) -> Array1<f64> {
    let mat = diag.column_vector().as_mat();
    Array1::from_shape_fn(mat.nrows(), |i| mat[(i, 0)])
}

fn compute_bunch_kaufman_inertia(
    diag: &Array1<f64>,
    subdiag: &Array1<f64>,
) -> (usize, usize, usize) {
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut zero = 0usize;
    let n = diag.len();
    let mut idx = 0usize;
    while idx < n {
        if idx + 1 < n && subdiag[idx].abs() > 1e-12 {
            let a = diag[idx];
            let b = subdiag[idx];
            let c = diag[idx + 1];
            let trace = a + c;
            let det = a * c - b * b;
            let discr = (trace * trace / 4.0 - det).max(0.0);
            let root = discr.sqrt();
            let eigenvalues = [trace / 2.0 + root, trace / 2.0 - root];
            for value in eigenvalues.iter() {
                if *value > 1e-12 {
                    positive += 1;
                } else if *value < -1e-12 {
                    negative += 1;
                } else {
                    zero += 1;
                }
            }
            idx += 2;
        } else {
            let value = diag[idx];
            if value > 1e-12 {
                positive += 1;
            } else if value < -1e-12 {
                negative += 1;
            } else {
                zero += 1;
            }
            idx += 1;
        }
    }
    (positive, negative, zero)
}

pub fn ldlt_rook(
    matrix: &Array2<f64>,
) -> Result<
    (
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Vec<usize>,
        Vec<usize>,
        (usize, usize, usize),
    ),
    FaerLinalgError,
> {
    let (nrows, ncols) = matrix.dim();
    if nrows != ncols {
        return Err(FaerLinalgError::Cholesky(
            solvers::LltError::NonPositivePivot { index: 0 },
        ));
    }
    let n = nrows;
    let mut factor = matrix.to_owned();
    let mut subdiag = Array1::<f64>::zeros(n);
    let mut perm_fwd = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];

    let mut faer_mat = array2_to_mat_mut(&mut factor);
    let subdiag_slice = subdiag
        .as_slice_memory_order_mut()
        .expect("1-D array should expose contiguous slice");
    let mut diag_mut = DiagMut::from_slice_mut(subdiag_slice);
    let par = get_global_parallelism();
    let mut params = <LbltParams as Auto<f64>>::auto();
    params.pivoting = PivotingStrategy::Rook;
    let params_spec = Spec::new(params);
    let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, f64>(
        n,
        par,
        params_spec,
    ));
    let mut stack = MemStack::new(&mut mem);

    factor::cholesky_in_place(
        faer_mat.as_mut(),
        diag_mut.as_mut(),
        &mut perm_fwd,
        &mut perm_inv,
        par,
        &mut stack,
        params_spec,
    );

    let mut diag = Array1::<f64>::zeros(n);
    for i in 0..n {
        diag[i] = factor[(i, i)];
        factor[(i, i)] = 1.0;
        for j in i + 1..n {
            factor[(i, j)] = 0.0;
        }
    }

    let inertia = compute_bunch_kaufman_inertia(&diag, &subdiag);

    Ok((factor, diag, subdiag, perm_fwd, perm_inv, inertia))
}

pub struct FaerArrayView<'a> {
    view: MatRef<'a, f64>,
}

impl<'a> FaerArrayView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix2>) -> Self {
        let (rows, cols) = array.dim();
        let strides = array.strides();
        // SAFETY: `ArrayBase` guarantees that the pointer returned by `as_ptr` is valid for the
        // lifetime of the array view, and the stride metadata from `strides()` accurately describes
        // how to traverse the 2-D view in memory. We forward this information to faer so that it can
        // operate on the ndarray-backed storage without performing any intermediate copies.
        let view =
            unsafe { MatRef::from_raw_parts(array.as_ptr(), rows, cols, strides[0], strides[1]) };
        Self { view }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        self.view
    }
}

pub struct FaerColView<'a> {
    view: MatRef<'a, f64>,
}

impl<'a> FaerColView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix1>) -> Self {
        let len = array.len();
        let stride = array.strides()[0];
        // SAFETY: identical reasoning as `FaerArrayView::new`; here we reinterpret the 1-D ndarray
        // storage as an n×1 matrix so that faer can consume it directly.
        let view = unsafe { MatRef::from_raw_parts(array.as_ptr(), len, 1, stride, 0) };
        Self { view }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        self.view
    }
}

pub trait FaerSvd {
    fn svd(
        &self,
        compute_u: bool,
        compute_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerSvd for ArrayBase<S, Ix2> {
    fn svd(
        &self,
        compute_u: bool,
        compute_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let faer_mat = faer_view.as_ref();
        if !compute_u && !compute_vt {
            let (rows, cols) = faer_mat.shape();
            let mut singular = Diag::<f64>::zeros(rows.min(cols));
            let par = get_global_parallelism();
            let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
                rows,
                cols,
                ComputeSvdVectors::No,
                ComputeSvdVectors::No,
                par,
                Default::default(),
            ));
            let mut stack = MemStack::new(&mut mem);
            svd::svd(
                faer_mat,
                singular.as_mut(),
                None,
                None,
                par,
                &mut stack,
                Default::default(),
            )
            .map_err(|_| FaerLinalgError::SvdNoConvergence)?;
            let singular_values = diag_to_array(singular.as_ref());
            return Ok((None, singular_values, None));
        }

        let (rows, cols) = faer_mat.shape();
        let compute_u_flag = if compute_u {
            ComputeSvdVectors::Full
        } else {
            ComputeSvdVectors::No
        };
        let compute_v_flag = if compute_vt {
            ComputeSvdVectors::Full
        } else {
            ComputeSvdVectors::No
        };

        let mut singular = Diag::<f64>::zeros(rows.min(cols));
        let mut u_storage = compute_u.then(|| Mat::<f64>::zeros(rows, rows));
        let mut v_storage = compute_vt.then(|| Mat::<f64>::zeros(cols, cols));

        let par = get_global_parallelism();
        let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
            rows,
            cols,
            compute_u_flag,
            compute_v_flag,
            par,
            Default::default(),
        ));
        let mut stack = MemStack::new(&mut mem);

        svd::svd(
            faer_mat.as_ref(),
            singular.as_mut(),
            u_storage.as_mut().map(|mat| mat.as_mut()),
            v_storage.as_mut().map(|mat| mat.as_mut()),
            par,
            &mut stack,
            Default::default(),
        )
        .map_err(|_| FaerLinalgError::SvdNoConvergence)?;

        let singular_values = diag_to_array(singular.as_ref());
        let u_opt = u_storage.map(|mat| mat_to_array(mat.as_ref()));
        let vt_opt = v_storage.map(|mat| {
            let mat_ref = mat.as_ref();
            Array2::from_shape_fn((mat_ref.ncols(), mat_ref.nrows()), |(i, j)| mat_ref[(j, i)])
        });

        Ok((u_opt, singular_values, vt_opt))
    }
}

pub trait FaerEigh {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerEigh for ArrayBase<S, Ix2> {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let eigen = faer_view
            .as_ref()
            .self_adjoint_eigen(side)
            .map_err(FaerLinalgError::SelfAdjointEigen)?;
        let values = diag_to_array(eigen.S());
        let vectors = mat_to_array(eigen.U());
        Ok((values, vectors))
    }
}

pub struct FaerCholeskyFactor {
    factor: solvers::Llt<f64>,
}

impl FaerCholeskyFactor {
    pub fn solve_vec(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let rhs_view = FaerColView::new(rhs);
        let sol = self.factor.solve(rhs_view.as_ref());
        Array1::from_shape_fn(rhs.len(), |i| sol[(i, 0)])
    }

    pub fn solve_mat(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let rhs_view = FaerArrayView::new(rhs);
        let sol = self.factor.solve(rhs_view.as_ref());
        mat_to_array(sol.as_ref())
    }

    pub fn diag(&self) -> Array1<f64> {
        diag_to_array(self.factor.L().diagonal())
    }

    pub fn lower_triangular(&self) -> Array2<f64> {
        mat_to_array(self.factor.L())
    }
}

pub trait FaerCholesky {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerCholesky for ArrayBase<S, Ix2> {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let factor = faer_view
            .as_ref()
            .llt(side)
            .map_err(FaerLinalgError::Cholesky)?;
        Ok(FaerCholeskyFactor { factor })
    }
}

pub trait FaerQr {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerQr for ArrayBase<S, Ix2> {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let qr = faer_view.as_ref().qr();
        let q = qr.compute_Q();
        let r = qr.R();
        Ok((mat_to_array(q.as_ref()), mat_to_array(r)))
    }
}
