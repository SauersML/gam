use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagMut, DiagRef};
use faer::linalg::cholesky::lblt::factor::{self, LbltParams, PivotingStrategy};
use faer::linalg::solvers::{self, Solve};
pub use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::{Auto, Conj, Mat, MatMut, MatRef, Par, Side, Spec, get_global_parallelism};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use thiserror::Error;

const BK_BLOCK_TOL: f64 = 1e-12;
const SYMMETRY_REL_TOL: f64 = 1e-12;
const SYMMETRY_ABS_TOL: f64 = 1e-12;
const RECONSTRUCTION_REL_TOL: f64 = 1e-8;
const RECONSTRUCTION_ABS_TOL: f64 = 1e-12;
const RRQR_RANK_ALPHA: f64 = 100.0;

#[derive(Debug, Error)]
pub enum FaerLinalgError {
    #[error("Factorization failed")]
    FactorizationFailed,
    #[error("SVD failed to converge")]
    SvdNoConvergence,
    #[error("Self-adjoint eigendecomposition failed: {0:?}")]
    SelfAdjointEigen(solvers::EvdError),
    #[error("Cholesky factorization failed: {0:?}")]
    Cholesky(solvers::LltError),
    #[error("LDLT factorization failed: {0:?}")]
    Ldlt(solvers::LdltError),
}

pub enum FaerSymmetricFactor {
    Llt(FaerLlt<f64>),
    Ldlt(FaerLdlt<f64>),
    Lblt(FaerLblt<f64>),
}

impl FaerSymmetricFactor {
    #[inline]
    pub fn solve(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        match self {
            FaerSymmetricFactor::Llt(f) => f.solve(rhs),
            FaerSymmetricFactor::Ldlt(f) => f.solve(rhs),
            FaerSymmetricFactor::Lblt(f) => f.solve(rhs),
        }
    }

    #[inline]
    pub fn solve_in_place(&self, rhs: MatMut<'_, f64>) {
        match self {
            FaerSymmetricFactor::Llt(f) => f.solve_in_place(rhs),
            FaerSymmetricFactor::Ldlt(f) => f.solve_in_place(rhs),
            FaerSymmetricFactor::Lblt(f) => f.solve_in_place(rhs),
        }
    }
}

impl crate::matrix::FactorizedSystem for FaerSymmetricFactor {
    fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
        let mut out = rhs.clone();
        let mut out_mat = array1_to_col_matmut(&mut out);
        self.solve_in_place(out_mat.as_mut());
        if !out.iter().all(|v| v.is_finite()) {
            return Err("symmetric factor solve produced non-finite values".to_string());
        }
        Ok(out)
    }

    fn solvemulti(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros(rhs.raw_dim());
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                out[[i, j]] = rhs[[i, j]];
            }
        }
        let mut out_mat = array2_to_matmut(&mut out);
        self.solve_in_place(out_mat.as_mut());
        if !out.iter().all(|v| v.is_finite()) {
            return Err("symmetric factor multi-solve produced non-finite values".to_string());
        }
        Ok(out)
    }

    fn logdet(&self) -> f64 {
        match self {
            FaerSymmetricFactor::Llt(f) => {
                2.0 * f
                    .L()
                    .diagonal()
                    .column_vector()
                    .iter()
                    .map(|&x| x.ln())
                    .sum::<f64>()
            }
            FaerSymmetricFactor::Ldlt(f) => {
                f.D().column_vector().iter().map(|&x| x.ln()).sum::<f64>()
            }
            FaerSymmetricFactor::Lblt(..) => {
                // lblt doesn't easily expose diagonal determinant. Fallback to sparse or other representations if needed, but typically Lblt is indefinite!
                // Actually faer doesn't easily expose lblt logdet since it has 2x2 blocks.
                // For our ML systems, if we dropped to LBLT, the matrix was indefinite and logdet is ill-defined (or complex).
                f64::NAN
            }
        }
    }
}

/// Factorize a symmetric system with LLT -> LDLT -> LBLT fallback.
#[inline]
pub fn factorize_symmetricwith_fallback(
    matrix: MatRef<'_, f64>,
    side: Side,
) -> Result<FaerSymmetricFactor, FaerLinalgError> {
    if let Ok(llt) = FaerLlt::new(matrix, side) {
        return Ok(FaerSymmetricFactor::Llt(llt));
    }
    let ldlt_err = match FaerLdlt::new(matrix, side) {
        Ok(ldlt) => return Ok(FaerSymmetricFactor::Ldlt(ldlt)),
        Err(err) => err,
    };
    let lblt = catch_unwind(AssertUnwindSafe(|| FaerLblt::new(matrix, side)))
        .map_err(|_| FaerLinalgError::Ldlt(ldlt_err))?;
    Ok(FaerSymmetricFactor::Lblt(lblt))
}

#[inline]
fn should_use_faer_matmul(m: usize, n: usize, k: usize) -> bool {
    // Small, centralized dispatch policy:
    // - stay on ndarray for tiny products to avoid setup overhead,
    // - switch to faer GEMM/GEMV for moderate+ sizes.
    const MIN_DIM: usize = 32;
    const MIN_FLOP_SCALE: usize = 64 * 64;
    (m >= MIN_DIM || n >= MIN_DIM || k >= MIN_DIM)
        && m.saturating_mul(n).saturating_mul(k) >= MIN_FLOP_SCALE
}

#[inline]
fn matmul_parallelism(m: usize, n: usize, k: usize) -> Par {
    // Prefer a work-based policy over per-dimension thresholds.
    // Tall/skinny products (e.g. N x p with large N, modest p) should still
    // parallelize when total work is high.
    const PAR_MIN_FLOP_SCALE: usize = 2_000_000;
    const PAR_MIN_LONG_DIM: usize = 256;
    let flop_scale = m.saturating_mul(n).saturating_mul(k);
    let long_dim = m.max(n).max(k);
    if flop_scale >= PAR_MIN_FLOP_SCALE && long_dim >= PAR_MIN_LONG_DIM {
        get_global_parallelism()
    } else {
        Par::Seq
    }
}

#[inline]
pub fn array2_to_matmut(array: &mut Array2<f64>) -> MatMut<'_, f64> {
    let (rows, cols) = array.dim();
    let strides = array.strides();

    // Check if we can get a pointer.
    // If the array is contiguous (either C or F order), or simply sliced with strides,
    // faer can handle it as long as we pass the pointer and strides.
    // However, as_mut_ptr() requires a mutable reference.
    // ndarray's as_ptr/as_mut_ptr works for both layouts.

    let s0 = strides[0];
    let s1 = strides[1];

    // SAFETY: We are creating a MatMut from the raw parts of the Array2.
    // We strictly follow the dimensions and strides provided by ndarray.
    unsafe { MatMut::from_raw_parts_mut(array.as_mut_ptr(), rows, cols, s0, s1) }
}

#[inline]
pub fn array1_to_col_matmut(array: &mut Array1<f64>) -> MatMut<'_, f64> {
    let len = array.len();
    let stride = array.strides()[0];
    unsafe {
        MatMut::from_raw_parts_mut(
            array.as_mut_ptr(),
            len,
            1,
            stride,
            0, // col stride irrelevant for 1 column
        )
    }
}

/// Compute A^T * A using faer's SIMD-optimized GEMM.
/// This is MUCH faster than ndarray's .t().dot() for matrices where n > ~100.
///
/// For a matrix A of shape (n, p), this computes the (p, p) result.
/// Uses zero-copy view when possible, falls back to copy for non-contiguous arrays.
#[inline]
pub fn fast_ata<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>) -> Array2<f64> {
    let p = a.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    fast_ata_into(a, &mut out);
    out
}

/// Compute A^T * A into a pre-allocated output buffer.
/// `out` must be shaped (p, p) where A is (n, p).
#[inline]
pub fn fast_ata_into<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>, out: &mut Array2<f64>) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    debug_assert_eq!(out.nrows(), p, "output rows must match p");
    debug_assert_eq!(out.ncols(), p, "output cols must match p");

    if !should_use_faer_matmul(p, p, n) {
        out.assign(&a.t().dot(a));
        return;
    }

    let mut outview = array2_to_matmut(out);

    let aview = FaerArrayView::new(a);
    let a_ref = aview.as_ref();
    let a_t = a_ref.transpose();
    let par = matmul_parallelism(p, p, n);
    matmul(outview.as_mut(), Accum::Replace, a_t, a_ref, 1.0, par);
}

/// Compute A^T * B using faer's SIMD-optimized GEMM.
/// For A of shape (n, p) and B of shape (n, q), this computes the (p, q) result.
/// Uses zero-copy views when possible.
#[inline]
pub fn fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (n_a, p) = a.dim();
    let (n_b, q) = b.dim();
    debug_assert_eq!(n_a, n_b, "A and B must have same number of rows");

    // For very small matrices, ndarray might be faster due to less overhead
    if !should_use_faer_matmul(p, q, n_a) {
        return a.t().dot(b);
    }

    let mut result = Mat::<f64>::zeros(p, q);

    let aview = FaerArrayView::new(a);
    let bview = FaerArrayView::new(b);
    let a_ref = aview.as_ref();
    let b_ref = bview.as_ref();

    // dst = A^T * B
    let par = matmul_parallelism(p, q, n_a);
    matmul(
        result.as_mut(),
        Accum::Replace,
        a_ref.transpose(),
        b_ref,
        1.0,
        par,
    );

    mat_to_array(result.as_ref())
}

/// Compute A * B using faer's SIMD-optimized GEMM.
/// For A of shape (n, p) and B of shape (p, q), this computes the (n, q) result.
/// Uses zero-copy views when possible.
#[inline]
pub fn fast_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    let (n, _) = a.dim();
    let (_, q) = b.dim();
    let mut out = Array2::<f64>::zeros((n, q));
    fast_ab_into(a, b, &mut out);
    out
}

/// Compute A * v using faer's SIMD-optimized GEMV.
/// For A of shape (n, p) and v of shape (p,), this computes the (n,) result.
#[inline]
pub fn fast_av<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (n, p) = a.dim();
    debug_assert_eq!(p, v.len(), "A cols must match v length");

    if !should_use_faer_matmul(n, 1, p) {
        return a.dot(v);
    }

    let mut result = Mat::<f64>::zeros(n, 1);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();

    let par = matmul_parallelism(n, 1, p);
    matmul(result.as_mut(), Accum::Replace, a_ref, v_ref, 1.0, par);

    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        out[i] = result[(i, 0)];
    }
    out
}

/// Compute A * v into a pre-allocated output buffer.
/// `out` must be length n where A is (n, p) and v is length p.
#[inline]
pub fn fast_av_into<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    v: &Array1<f64>,
    out: &mut Array1<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    debug_assert_eq!(v.len(), p, "vector length must match A cols");
    debug_assert_eq!(out.len(), n, "output length must match A rows");

    if !should_use_faer_matmul(n, 1, p) {
        out.assign(&a.dot(v));
        return;
    }

    let mut outview = array1_to_col_matmut(out);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();
    let par = matmul_parallelism(n, 1, p);
    matmul(outview.as_mut(), Accum::Replace, a_ref, v_ref, 1.0, par);
}

/// Compute A^T * v into a pre-allocated output buffer.
/// `out` must be length p where A is (n, p) and v is length n.
#[inline]
pub fn fast_atv_into<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    v: &Array1<f64>,
    out: &mut Array1<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    debug_assert_eq!(v.len(), n, "vector length must match A rows");
    debug_assert_eq!(out.len(), p, "output length must match A cols");

    if !should_use_faer_matmul(p, 1, n) {
        out.assign(&a.t().dot(v));
        return;
    }

    let mut outview = array1_to_col_matmut(out);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();
    let par = matmul_parallelism(p, 1, n);
    matmul(
        outview.as_mut(),
        Accum::Replace,
        a_ref.transpose(),
        v_ref,
        1.0,
        par,
    );
}

/// Compute A^T * v using faer's SIMD-optimized GEMV.
/// For A of shape (n, p) and v of shape (n,), this computes the (p,) result.
#[inline]
pub fn fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (n, p) = a.dim();
    debug_assert_eq!(n, v.len(), "A rows must match v length");

    // For very small arrays, ndarray might be faster
    if !should_use_faer_matmul(p, 1, n) {
        return a.t().dot(v);
    }

    let mut result = Mat::<f64>::zeros(p, 1);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();

    // dst = A^T * v (treating v as n×1 matrix)
    let par = matmul_parallelism(p, 1, n);
    matmul(
        result.as_mut(),
        Accum::Replace,
        a_ref.transpose(),
        v_ref,
        1.0,
        par,
    );

    let mut out = Array1::<f64>::zeros(p);
    for i in 0..p {
        out[i] = result[(i, 0)];
    }
    out
}

/// Compute A^T * diag(W) * A using streaming chunks to avoid O(n*p) allocation.
#[inline]
pub fn fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
) -> Array2<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use ndarray::{ShapeBuilder, s};

    let (n, p) = x.dim();
    debug_assert_eq!(n, w.len(), "X rows must match W length");
    if n == 0 || p == 0 {
        return Array2::<f64>::zeros((p, p));
    }
    if !should_use_faer_matmul(p, p, n) {
        let w_x = Array2::from_shape_fn((n, p), |(i, j)| w[i] * x[[i, j]]);
        return x.t().dot(&w_x);
    }

    // Streaming chunked: peak allocation is chunk_rows × p instead of n × p.
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    let chunk_rows = (TARGET_BYTES / (p.max(1) * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n);

    let mut result = Array2::<f64>::zeros((p, p).f());
    let mut weighted_chunk = Array2::<f64>::zeros((chunk_rows, p).f());
    let mut out_view = array2_to_matmut(&mut result);

    for start in (0..n).step_by(chunk_rows) {
        let rows = (n - start).min(chunk_rows);
        {
            let x_slice = x.slice(s![start..start + rows, ..]);
            let mut chunk = weighted_chunk.slice_mut(s![0..rows, ..]);
            for local in 0..rows {
                let sqrtw = w[start + local].max(0.0).sqrt();
                for col in 0..p {
                    chunk[[local, col]] = x_slice[[local, col]] * sqrtw;
                }
            }
        }
        let chunk_slice = weighted_chunk.slice(s![0..rows, ..]);
        let chunk_view = FaerArrayView::new(&chunk_slice);
        let par = matmul_parallelism(p, p, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            chunk_view.as_ref().transpose(),
            chunk_view.as_ref(),
            1.0,
            par,
        );
    }

    result
}

/// Compute A^T * diag(W) * B using streaming chunks.
#[inline]
pub fn fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Array2<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use ndarray::{ShapeBuilder, s};

    let (n, q) = y.dim();
    let px = x.ncols();
    debug_assert_eq!(n, w.len(), "Y rows must match W length");
    debug_assert_eq!(n, x.nrows(), "X rows must match Y rows");
    if n == 0 || px == 0 || q == 0 {
        return Array2::<f64>::zeros((px, q));
    }
    if !should_use_faer_matmul(px, q, n) {
        let w_y = Array2::from_shape_fn((n, q), |(i, j)| w[i] * y[[i, j]]);
        return x.t().dot(&w_y);
    }

    // Streaming: only allocate chunk_rows × q for the weighted Y slice.
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    let total_cols = px + q;
    let chunk_rows = (TARGET_BYTES / (total_cols.max(1) * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n);

    let mut result = Array2::<f64>::zeros((px, q).f());
    let mut wy_chunk = Array2::<f64>::zeros((chunk_rows, q).f());
    let mut out_view = array2_to_matmut(&mut result);

    for start in (0..n).step_by(chunk_rows) {
        let rows = (n - start).min(chunk_rows);
        {
            let y_slice = y.slice(s![start..start + rows, ..]);
            let mut chunk = wy_chunk.slice_mut(s![0..rows, ..]);
            for local in 0..rows {
                let wi = w[start + local];
                for col in 0..q {
                    chunk[[local, col]] = y_slice[[local, col]] * wi;
                }
            }
        }
        let x_slice = x.slice(s![start..start + rows, ..]);
        let wy_slice = wy_chunk.slice(s![0..rows, ..]);
        let x_view = FaerArrayView::new(&x_slice);
        let wy_view = FaerArrayView::new(&wy_slice);
        let par = matmul_parallelism(px, q, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            x_view.as_ref().transpose(),
            wy_view.as_ref(),
            1.0,
            par,
        );
    }

    result
}

fn mat_to_array(mat: MatRef<'_, f64>) -> Array2<f64> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    if nrows == 0 || ncols == 0 {
        return out;
    }
    // ndarray is row-major by default. Write row-by-row for best cache behavior
    // on the output side.
    if let Some(out_slice) = out.as_slice_memory_order_mut() {
        // Row-major: out_slice[i * ncols + j] = mat[(i, j)]
        for i in 0..nrows {
            let row_start = i * ncols;
            for j in 0..ncols {
                out_slice[row_start + j] = mat[(i, j)];
            }
        }
    } else {
        for j in 0..ncols {
            for i in 0..nrows {
                out[[i, j]] = mat[(i, j)];
            }
        }
    }
    out
}

/// Write faer matmul result A*B directly into a pre-allocated ndarray Array2.
/// Avoids the intermediate faer::Mat allocation and mat_to_array copy.
#[inline]
pub fn fast_ab_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    out: &mut Array2<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    let (p_b, q) = b.dim();
    debug_assert_eq!(p, p_b, "A and B must have compatible inner dimensions");
    debug_assert_eq!(out.dim(), (n, q), "output dimensions must match A*B result");

    if !should_use_faer_matmul(n, q, p) {
        out.assign(&a.dot(b));
        return;
    }

    let aview = FaerArrayView::new(a);
    let bview = FaerArrayView::new(b);
    let a_ref = aview.as_ref();
    let b_ref = bview.as_ref();

    let par = matmul_parallelism(n, q, p);
    let mut outview = array2_to_matmut(out);
    matmul(outview.as_mut(), Accum::Replace, a_ref, b_ref, 1.0, par);
}

fn diag_to_array(diag: DiagRef<'_, f64>) -> Array1<f64> {
    let mat = diag.column_vector().as_mat();
    let mut out = Array1::<f64>::zeros(mat.nrows());
    for i in 0..mat.nrows() {
        out[i] = mat[(i, 0)];
    }
    out
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
        if idx + 1 < n && subdiag[idx].abs() > BK_BLOCK_TOL {
            let a = diag[idx];
            let b = subdiag[idx];
            let c = diag[idx + 1];
            let trace = a + c;
            let det = a * c - b * b;
            let discr = (trace * trace / 4.0 - det).max(0.0);
            let root = discr.sqrt();
            let eigenvalues = [trace / 2.0 + root, trace / 2.0 - root];
            for value in eigenvalues.iter() {
                if *value > BK_BLOCK_TOL {
                    positive += 1;
                } else if *value < -BK_BLOCK_TOL {
                    negative += 1;
                } else {
                    zero += 1;
                }
            }
            idx += 2;
        } else {
            let value = diag[idx];
            if value > BK_BLOCK_TOL {
                positive += 1;
            } else if value < -BK_BLOCK_TOL {
                negative += 1;
            } else {
                zero += 1;
            }
            idx += 1;
        }
    }
    (positive, negative, zero)
}

fn is_symmetricwith_tolerance(matrix: &Array2<f64>, rel_tol: f64, abs_tol: f64) -> bool {
    let (nrows, ncols) = matrix.dim();
    if nrows != ncols {
        return false;
    }
    let mut scale = 0.0f64;
    for i in 0..nrows {
        for j in 0..ncols {
            scale = scale.max(matrix[[i, j]].abs());
        }
    }
    let tol = abs_tol + rel_tol * scale.max(1.0);
    for i in 0..nrows {
        for j in i + 1..ncols {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tol {
                return false;
            }
        }
    }
    true
}

fn reconstruct_from_bunch_kaufman(
    l_unit_lower: &Array2<f64>,
    d_diag: &Array1<f64>,
    d_subdiag: &Array1<f64>,
    perm_inv: &[usize],
) -> Array2<f64> {
    let n = d_diag.len();
    let mut b = Array2::<f64>::zeros((n, n));
    let mut i = 0usize;
    while i < n {
        if i + 1 < n && d_subdiag[i].abs() > BK_BLOCK_TOL {
            b[[i, i]] = d_diag[i];
            b[[i, i + 1]] = d_subdiag[i];
            b[[i + 1, i]] = d_subdiag[i];
            b[[i + 1, i + 1]] = d_diag[i + 1];
            i += 2;
        } else {
            b[[i, i]] = d_diag[i];
            i += 1;
        }
    }

    let tmp = l_unit_lower.dot(&b).dot(&l_unit_lower.t());
    let mut out = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for col in 0..n {
            out[[row, col]] = tmp[[perm_inv[row], perm_inv[col]]];
        }
    }
    out
}

fn isvalid_inverse_permutation(perm_fwd: &[usize], perm_inv: &[usize], n: usize) -> bool {
    if perm_fwd.len() != n || perm_inv.len() != n {
        return false;
    }

    let mut seen_fwd = vec![false; n];
    for &p in perm_fwd {
        if p >= n || seen_fwd[p] {
            return false;
        }
        seen_fwd[p] = true;
    }

    let mut seen_inv = vec![false; n];
    for &p in perm_inv {
        if p >= n || seen_inv[p] {
            return false;
        }
        seen_inv[p] = true;
    }

    for i in 0..n {
        if perm_inv[perm_fwd[i]] != i || perm_fwd[perm_inv[i]] != i {
            return false;
        }
    }

    true
}

fn validate_ldlt_rook_outputs(
    matrix: &Array2<f64>,
    l_unit_lower: &Array2<f64>,
    d_diag: &Array1<f64>,
    d_subdiag: &Array1<f64>,
    perm_fwd: &[usize],
    perm_inv: &[usize],
) -> bool {
    let n = matrix.nrows();
    if !isvalid_inverse_permutation(perm_fwd, perm_inv, n) {
        return false;
    }

    if !l_unit_lower.iter().all(|v| v.is_finite())
        || !d_diag.iter().all(|v| v.is_finite())
        || !d_subdiag.iter().all(|v| v.is_finite())
    {
        return false;
    }

    let reconstructed = reconstruct_from_bunch_kaufman(l_unit_lower, d_diag, d_subdiag, perm_inv);
    if !reconstructed.iter().all(|v| v.is_finite()) {
        return false;
    }
    let max_abs_err = (&reconstructed - matrix)
        .iter()
        .fold(0.0f64, |acc, &x| acc.max(x.abs()));
    let input_scale = matrix.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    let tol = RECONSTRUCTION_ABS_TOL + RECONSTRUCTION_REL_TOL * input_scale.max(1.0);
    max_abs_err <= tol
}

/// Computes a symmetric-indefinite rook-pivoted `LBL^T` factorization.
///
/// Returns `(l_unit_lower, d_diag, d_subdiag, perm_fwd, perm_inv, inertia)` where:
/// - `l_unit_lower`: unit-lower-triangular `L`.
/// - `d_diag`: diagonal entries of block-diagonal `B`.
/// - `d_subdiag`: off-diagonal entries for `2x2` blocks in `B` (zeros for `1x1` pivots).
/// - `perm_fwd`, `perm_inv`: permutation arrays from faer.
/// - `inertia`: `(n_pos, n_neg, nzero)` computed from `B` blocks.
///
/// This mirrors faer's bunch-kaufman storage contract for `cholesky_in_place`.
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
        return Err(FaerLinalgError::FactorizationFailed);
    }
    if !matrix.iter().all(|v| v.is_finite()) {
        return Err(FaerLinalgError::FactorizationFailed);
    }
    if !is_symmetricwith_tolerance(matrix, SYMMETRY_REL_TOL, SYMMETRY_ABS_TOL) {
        return Err(FaerLinalgError::FactorizationFailed);
    }
    let n = nrows;
    // faer LBLT contract (bunch-kaufman::factor::cholesky_in_place):
    // - `matrix` (in-place output) stores:
    //   - strict lower triangle: L (unit-lower multipliers),
    //   - diagonal: diagonal of block-diagonal B factor (1x1 and 2x2 blocks).
    // - `DiagMut` argument stores only the off-diagonal terms of 2x2 blocks in B.
    //
    // We keep these as explicit outputs:
    // - `l_unit_lower`: L with diagonal normalized to 1, upper triangle zeroed.
    // - `d_diag`: diagonal of B.
    // - `d_subdiag`: subdiagonal entries for 2x2 B blocks.
    let mut l_unit_lower = matrix.to_owned();
    let mut d_subdiag = Array1::<f64>::zeros(n);
    let mut perm_fwd = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];

    let mut faer_mat = array2_to_matmut(&mut l_unit_lower);
    let subdiag_slice = d_subdiag
        .as_slice_memory_order_mut()
        .expect("1-D array should expose contiguous slice");
    let mut b_subdiagmut = DiagMut::from_slice_mut(subdiag_slice);
    let par = get_global_parallelism();
    let mut params = <LbltParams as Auto<f64>>::auto();
    params.pivoting = PivotingStrategy::Rook;
    let paramsspec = Spec::new(params);
    let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, f64>(
        n, par, paramsspec,
    ));
    let stack = MemStack::new(&mut mem);

    factor::cholesky_in_place(
        faer_mat.as_mut(),
        b_subdiagmut.as_mut(),
        &mut perm_fwd,
        &mut perm_inv,
        par,
        stack,
        paramsspec,
    );

    // Extract B diagonal from faer in-place diagonal, then normalize L diagonal to 1.
    let mut d_diag = Array1::<f64>::zeros(n);
    for i in 0..n {
        d_diag[i] = l_unit_lower[(i, i)];
        l_unit_lower[(i, i)] = 1.0;
        for j in i + 1..n {
            l_unit_lower[(i, j)] = 0.0;
        }
    }

    if !validate_ldlt_rook_outputs(
        matrix,
        &l_unit_lower,
        &d_diag,
        &d_subdiag,
        &perm_fwd,
        &perm_inv,
    ) {
        return Err(FaerLinalgError::FactorizationFailed);
    }

    let inertia = compute_bunch_kaufman_inertia(&d_diag, &d_subdiag);

    Ok((l_unit_lower, d_diag, d_subdiag, perm_fwd, perm_inv, inertia))
}

pub struct FaerArrayView<'a> {
    ptr: *const f64,
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
    owned: Option<Array2<f64>>,
    marker: PhantomData<&'a f64>,
}

impl<'a> FaerArrayView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix2>) -> Self {
        let (rows, cols) = array.dim();
        let strides = array.strides();
        // Guard against layouts that can alias or reverse memory traversal (e.g.
        // negative/zero strides). These can violate assumptions in faer kernels.
        // For such layouts we materialize a compact owned copy.
        if strides[0] <= 0 || strides[1] <= 0 {
            let owned = array.to_owned();
            let owned_strides = owned.strides();
            return Self {
                ptr: owned.as_ptr(),
                rows,
                cols,
                row_stride: owned_strides[0],
                col_stride: owned_strides[1],
                owned: Some(owned),
                marker: PhantomData,
            };
        }

        Self {
            ptr: array.as_ptr(),
            rows,
            cols,
            row_stride: strides[0],
            col_stride: strides[1],
            owned: None,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        let (ptr, rows, cols, row_stride, col_stride) = if let Some(owned) = &self.owned {
            let strides = owned.strides();
            (
                owned.as_ptr(),
                owned.nrows(),
                owned.ncols(),
                strides[0],
                strides[1],
            )
        } else {
            (
                self.ptr,
                self.rows,
                self.cols,
                self.row_stride,
                self.col_stride,
            )
        };
        // SAFETY: pointer/shape/strides either come directly from a live ndarray
        // view with positive strides, or from an owned compact copy stored inside
        // this wrapper, which guarantees validity for the returned view lifetime.
        unsafe { MatRef::from_raw_parts(ptr, rows, cols, row_stride, col_stride) }
    }
}

pub struct FaerColView<'a> {
    ptr: *const f64,
    len: usize,
    stride: isize,
    owned: Option<Array1<f64>>,
    marker: PhantomData<&'a f64>,
}

impl<'a> FaerColView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix1>) -> Self {
        let len = array.len();
        let stride = array.strides()[0];
        if stride <= 0 {
            let owned = array.to_owned();
            return Self {
                ptr: owned.as_ptr(),
                len,
                stride: 1,
                owned: Some(owned),
                marker: PhantomData,
            };
        }
        Self {
            ptr: array.as_ptr(),
            len,
            stride,
            owned: None,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        let (ptr, len, stride) = if let Some(owned) = &self.owned {
            (owned.as_ptr(), owned.len(), 1)
        } else {
            (self.ptr, self.len, self.stride)
        };
        // SAFETY: analogous to FaerArrayView::as_ref.
        unsafe { MatRef::from_raw_parts(ptr, len, 1, stride, 0) }
    }
}

pub trait FaerSvd {
    fn svd(
        &self,
        compute_u: bool,
        computevt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerSvd for ArrayBase<S, Ix2> {
    fn svd(
        &self,
        compute_u: bool,
        computevt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError> {
        let faerview = FaerArrayView::new(self);
        let faer_mat = faerview.as_ref();
        if !compute_u && !computevt {
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
            let stack = MemStack::new(&mut mem);
            svd::svd(
                faer_mat,
                singular.as_mut(),
                None,
                None,
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| FaerLinalgError::SvdNoConvergence)?;
            let singularvalues = diag_to_array(singular.as_ref());
            return Ok((None, singularvalues, None));
        }

        let (rows, cols) = faer_mat.shape();
        let rank = rows.min(cols);
        let compute_u_flag = if compute_u {
            ComputeSvdVectors::Thin
        } else {
            ComputeSvdVectors::No
        };
        let computev_flag = if computevt {
            ComputeSvdVectors::Thin
        } else {
            ComputeSvdVectors::No
        };

        let mut singular = Diag::<f64>::zeros(rows.min(cols));
        let mut u_storage = compute_u.then(|| Mat::<f64>::zeros(rows, rank));
        let mut v_storage = computevt.then(|| Mat::<f64>::zeros(cols, rank));

        let par = get_global_parallelism();
        let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
            rows,
            cols,
            compute_u_flag,
            computev_flag,
            par,
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);

        svd::svd(
            faer_mat.as_ref(),
            singular.as_mut(),
            u_storage.as_mut().map(|mat| mat.as_mut()),
            v_storage.as_mut().map(|mat| mat.as_mut()),
            par,
            stack,
            Default::default(),
        )
        .map_err(|_| FaerLinalgError::SvdNoConvergence)?;

        let singularvalues = diag_to_array(singular.as_ref());
        let u_opt = u_storage.map(|mat| mat_to_array(mat.as_ref()));
        let vt_opt = v_storage.map(|mat| {
            let mat_ref = mat.as_ref();
            let mut out = Array2::<f64>::zeros((mat_ref.ncols(), mat_ref.nrows()));
            for j in 0..mat_ref.nrows() {
                for i in 0..mat_ref.ncols() {
                    out[[i, j]] = mat_ref[(j, i)];
                }
            }
            out
        });

        Ok((u_opt, singularvalues, vt_opt))
    }
}

pub trait FaerEigh {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerEigh for ArrayBase<S, Ix2> {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
        let faerview = FaerArrayView::new(self);
        let eigen = faerview
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
    pub fn solvevec(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let mut rhs = rhs.to_owned();
        let mut rhsview = array1_to_col_matmut(&mut rhs);
        self.factor.solve_in_place(rhsview.as_mut());
        rhs
    }

    pub fn solve_mat_into<S: Data<Elem = f64>>(
        &self,
        rhs: &ArrayBase<S, Ix2>,
        out: &mut Array2<f64>,
    ) {
        if out.dim() != rhs.dim() {
            *out = Array2::<f64>::zeros(rhs.dim());
        }
        out.assign(rhs);
        let mut rhsview = array2_to_matmut(out);
        self.factor.solve_in_place(rhsview.as_mut());
    }

    pub fn solve_mat(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(rhs.dim());
        self.solve_mat_into(rhs, &mut out);
        out
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
        let faerview = FaerArrayView::new(self);
        let factor = faerview
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
        let faerview = FaerArrayView::new(self);
        let qr = faerview.as_ref().qr();
        let q = qr.compute_thin_Q();
        let r = qr.thin_R();
        Ok((mat_to_array(q.as_ref()), mat_to_array(r)))
    }
}

/// Compute an orthonormal basis for `null(a^T)` using column-pivoted QR on `a`.
///
/// This is intended for tall/skinny matrices where `a ∈ R^{m×n}` with `m >= n`.
/// If `A P^T = Q R`, then the trailing `m-rank(A)` columns of `Q` span
/// `null(A^T)`.
pub fn rrqr_nullspace_basis<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    rank_alpha: f64,
) -> Result<(Array2<f64>, usize), FaerLinalgError> {
    let faerview = FaerArrayView::new(a);
    let qr = faerview.as_ref().col_piv_qr();
    let r = qr.thin_R();
    let diag_len = r.nrows().min(r.ncols());
    let leading_diag = if diag_len > 0 { r[(0, 0)].abs() } else { 0.0 };
    let tol = rank_alpha
        * f64::EPSILON
        * (a.nrows().max(a.ncols()).max(1) as f64)
        * leading_diag.max(1.0);
    let rank = (0..diag_len).filter(|&i| r[(i, i)].abs() > tol).count();
    let z = if rank >= a.nrows() {
        Array2::<f64>::zeros((a.nrows(), 0))
    } else {
        let nullity = a.nrows() - rank;
        let mut selector = Mat::<f64>::zeros(a.nrows(), nullity);
        for j in 0..nullity {
            selector[(rank + j, j)] = 1.0;
        }
        let par = get_global_parallelism();
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.Q_basis(),
            qr.Q_coeff(),
            Conj::No,
            selector.as_mut(),
            par,
            MemStack::new(&mut MemBuffer::new(
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
                    a.nrows(),
                    qr.Q_coeff().nrows(),
                    nullity,
                ),
            )),
        );
        mat_to_array(selector.as_ref())
    };
    Ok((z, rank))
}

#[inline]
pub fn default_rrqr_rank_alpha() -> f64 {
    RRQR_RANK_ALPHA
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    fn inertia_from_eigs(a: &Array2<f64>, tol: f64) -> (usize, usize, usize) {
        let (evals, _) = a
            .eigh(Side::Lower)
            .expect("eigen decomposition should succeed");
        let mut pos = 0usize;
        let mut neg = 0usize;
        let mut zero = 0usize;
        for &v in &evals {
            if v > tol {
                pos += 1;
            } else if v < -tol {
                neg += 1;
            } else {
                zero += 1;
            }
        }
        (pos, neg, zero)
    }

    #[test]
    fn ldlt_rook_reconstructs_input_and_matches_inertia() {
        // Symmetric indefinite matrix that exercises rook pivoting / 2x2 blocks.
        let a = array![
            [0.0, 2.0, 0.5, 0.0],
            [2.0, 0.0, -1.0, 0.3],
            [0.5, -1.0, 1.5, 0.4],
            [0.0, 0.3, 0.4, -0.2]
        ];

        let (l, d_diag, d_subdiag, perm_fwd, perm_inv, inertia) =
            ldlt_rook(&a).expect("ldlt_rook should succeed");
        let _ = perm_fwd;

        // L should be unit-lower and upper triangle should be zeroed by construction.
        for i in 0..l.nrows() {
            assert!((l[[i, i]] - 1.0).abs() < 1e-12);
            for j in i + 1..l.ncols() {
                assert!(l[[i, j]].abs() < 1e-12);
            }
        }

        let a_rec = reconstruct_from_bunch_kaufman(&l, &d_diag, &d_subdiag, &perm_inv);
        let max_abs_err = (&a_rec - &a)
            .iter()
            .fold(0.0f64, |acc, &x| acc.max(x.abs()));
        assert!(
            max_abs_err < 1e-8,
            "reconstruction error too large: {max_abs_err:e}"
        );

        let eig_inertia = inertia_from_eigs(&a, 1e-9);
        assert_eq!(inertia, eig_inertia);
    }

    #[test]
    fn ldlt_rook_rejects_non_finite_input() {
        let a = array![[1.0, f64::NAN], [f64::NAN, 2.0]];
        assert!(matches!(
            ldlt_rook(&a),
            Err(FaerLinalgError::FactorizationFailed)
        ));
    }

    #[test]
    fn ldlt_rook_rejects_non_symmetric_input() {
        let a = array![[1.0, 10.0], [0.0, 1.0]];
        assert!(matches!(
            ldlt_rook(&a),
            Err(FaerLinalgError::FactorizationFailed)
        ));
    }

    #[test]
    fn ldlt_rook_accepts_tiny_symmetry_roundoff() {
        let eps = 1e-14;
        let a = array![[2.0, 1.0 + eps], [1.0, 3.0]];
        assert!(ldlt_rook(&a).is_ok());
    }

    #[test]
    fn rrqr_nullspace_basis_is_orthonormal_and_annihilates_transpose() {
        let a = array![[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0],];
        let (z, rank) =
            rrqr_nullspace_basis(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(rank, 2);
        assert_eq!(z.nrows(), 4);
        assert_eq!(z.ncols(), 2);

        let gram = z.t().dot(&z);
        let ident = Array2::<f64>::eye(z.ncols());
        let gram_err = (&gram - &ident)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(gram_err < 1e-10, "Z is not orthonormal: {gram_err:e}");

        let residual = a.t().dot(&z);
        let resid_max = residual.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(resid_max < 1e-10, "A^T Z residual too large: {resid_max:e}");
    }

    #[test]
    fn rrqr_nullspace_basis_detectszero_rank_matrix() {
        let a = Array2::<f64>::zeros((5, 2));
        let (z, rank) =
            rrqr_nullspace_basis(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(rank, 0);
        assert_eq!(z.dim(), (5, 5));
        let ident = Array2::<f64>::eye(5);
        let max_err = (&z.slice(s![.., ..5]).to_owned() - &ident)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_err < 1e-10, "zero matrix should yield identity basis");
    }
}
