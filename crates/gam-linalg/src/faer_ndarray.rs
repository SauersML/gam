use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagRef};
use faer::linalg::solvers::{self, Solve};
pub use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::prelude::ReborrowMut;
use faer::{Conj, Mat, MatMut, MatRef, Par, Side, Unbind, get_global_parallelism};
use ndarray::{Array1, Array2, ArrayBase, ArrayViewMut1, Data, Ix1, Ix2};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use thiserror::Error;

const RRQR_RANK_ALPHA: f64 = 100.0;

thread_local! {
    static NESTED_PARALLEL_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct NestedParallelGuard;

impl NestedParallelGuard {
    #[inline]
    fn enter() -> Self {
        NESTED_PARALLEL_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self
    }
}

impl Drop for NestedParallelGuard {
    #[inline]
    fn drop(&mut self) {
        NESTED_PARALLEL_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Run `body` with the current thread marked as inside a data-parallel row
/// region, so any faer GEMM it issues (directly or transitively) pins to
/// `Par::Seq` via [`effective_global_parallelism`] instead of re-fanning the
/// global Rayon pool. The guard is held for exactly the duration of `body` and
/// dropped on return — including early `?` returns from inside `body`, since the
/// guard lives in this function's frame.
///
/// Call this from the per-chunk/per-row closure of an `into_par_iter` whose body
/// performs GEMM, to prevent the Rayon-pool × faer-pool oversubscription.
#[inline]
pub fn with_nested_parallel<T>(body: impl FnOnce() -> T) -> T {
    let guard = NestedParallelGuard::enter();
    let out = body();
    drop(guard);
    out
}

/// `true` when the current thread is inside at least one [`NestedParallelGuard`]
/// scope, i.e. a parallel row reduction is already in flight on this thread.
#[inline]
pub fn in_nested_parallel_region() -> bool {
    NESTED_PARALLEL_DEPTH.with(|depth| depth.get() > 0)
}

/// faer parallelism policy that respects nested data-parallel regions: returns
/// faer's global policy at the top level, but `Par::Seq` once a
/// [`NestedParallelGuard`] is active so a GEMM issued from inside a parallel row
/// fan-out does not multiply the live thread count against the outer pool.
///
/// Use this in place of `faer::get_global_parallelism()` for any matmul that can
/// be reached from inside a row-parallel closure.
#[inline]
pub fn effective_global_parallelism() -> Par {
    if in_nested_parallel_region() {
        Par::Seq
    } else {
        get_global_parallelism()
    }
}

#[derive(Debug, Error)]
pub enum FaerLinalgError {
    #[error("Factorization failed in {context}")]
    FactorizationFailed { context: &'static str },
    #[error("SVD failed to converge in {context}")]
    SvdNoConvergence { context: &'static str },
    #[error("Self-adjoint eigendecomposition input contains non-finite values in {context}")]
    SelfAdjointEigenNonFiniteInput { context: &'static str },
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

#[inline]
pub fn cholesky_factor_logdet(factor: MatRef<'_, f64>) -> f64 {
    2.0 * diagonal_log_sum(factor.diagonal())
}

#[inline]
fn diagonal_log_sum(diagonal: DiagRef<'_, f64>) -> f64 {
    diagonal
        .column_vector()
        .iter()
        .map(|&x| x.ln())
        .sum::<f64>()
}

impl FaerSymmetricFactor {
    /// Returns the dimension of the factorized square matrix.
    #[inline]
    pub fn n(&self) -> usize {
        use faer::linalg::solvers::ShapeCore;
        match self {
            FaerSymmetricFactor::Llt(f) => f.nrows(),
            FaerSymmetricFactor::Ldlt(f) => f.nrows(),
            FaerSymmetricFactor::Lblt(f) => f.nrows(),
        }
    }

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
            FaerSymmetricFactor::Llt(f) => cholesky_factor_logdet(f.L()),
            FaerSymmetricFactor::Ldlt(f) => diagonal_log_sum(f.D()),
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
const fn should_use_faer_matmul(m: usize, n: usize, k: usize) -> bool {
    // Small, centralized dispatch policy:
    // - stay on ndarray for tiny products to avoid setup overhead,
    // - switch to faer GEMM/GEMV for moderate+ sizes.
    const MIN_DIM: usize = 32;
    const MIN_FLOP_SCALE: usize = 64 * 64;
    (m >= MIN_DIM || n >= MIN_DIM || k >= MIN_DIM)
        && m.saturating_mul(n).saturating_mul(k) >= MIN_FLOP_SCALE
}

#[inline]
pub fn matmul_parallelism(m: usize, n: usize, k: usize) -> Par {
    // Prefer a work-based policy over per-dimension thresholds.
    // Tall/skinny products (e.g. N x p with large N, modest p) should still
    // parallelize when total work is high.
    const PAR_MIN_FLOP_SCALE: usize = 2_000_000;
    const PAR_MIN_LONG_DIM: usize = 256;
    let flop_scale = m.saturating_mul(n).saturating_mul(k);
    let long_dim = m.max(n).max(k);
    if flop_scale >= PAR_MIN_FLOP_SCALE && long_dim >= PAR_MIN_LONG_DIM {
        // `effective_global_parallelism` collapses to `Par::Seq` when this GEMM
        // is reached from inside a `NestedParallelGuard` row region, preventing
        // the Rayon-pool × faer-pool multiplicative oversubscription.
        effective_global_parallelism()
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

    // SAFETY: array.as_mut_ptr() is ndarray's logical (0, 0) pointer, and
    // ndarray's dimensions plus signed element strides describe every initialized
    // element of this uniquely borrowed Array2 for the returned MatMut lifetime.
    unsafe { MatMut::from_raw_parts_mut(array.as_mut_ptr(), rows, cols, s0, s1) }
}

#[inline]
pub fn array1_to_col_matmut(array: &mut Array1<f64>) -> MatMut<'_, f64> {
    let len = array.len();
    let stride = array.strides()[0];
    // SAFETY: array.as_mut_ptr() is ndarray's logical first-element pointer, and
    // len plus the signed element stride describe every initialized element of
    // this uniquely borrowed Array1 for the returned len×1 MatMut lifetime.
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
/// Uses a zero-copy view for positive-stride layouts and copies only layouts
/// with non-positive strides.
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
    use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri_matmul};

    let (n, p) = a.dim();
    assert_eq!(out.nrows(), p, "output rows must match p");
    assert_eq!(out.ncols(), p, "output cols must match p");

    if !should_use_faer_matmul(p, p, n) {
        out.assign(&a.t().dot(a));
        return;
    }

    let mut outview = array2_to_matmut(out);

    let aview = FaerArrayView::new(a);
    let a_ref = aview.as_ref();
    let a_t = a_ref.transpose();
    let par = matmul_parallelism(p, p, n);
    tri_matmul(
        outview.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        a_t,
        BlockStructure::Rectangular,
        a_ref,
        BlockStructure::Rectangular,
        1.0,
        par,
    );
    // Mirror lower triangle to upper to populate the full symmetric output.
    for i in 0..p {
        for j in (i + 1)..p {
            out[[i, j]] = out[[j, i]];
        }
    }
}

/// Compute A^T * B using faer's SIMD-optimized GEMM.
/// For A of shape (n, p) and B of shape (n, q), this computes the (p, q) result.
/// Uses zero-copy views when possible.
#[inline]
pub fn fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_atb(a.view(), b.view()))
    {
        return out;
    }
    let (n_a, p) = a.dim();
    let q = b.ncols();
    fast_atb_with_parallelism(a, b, matmul_parallelism(p, q, n_a))
}

/// Compute A^T * B with an explicit faer parallelism policy for callers that
/// are already running independent products in an outer Rayon task.
#[inline]
pub fn fast_atb_with_parallelism<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    par: Par,
) -> Array2<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (n_a, p) = a.dim();
    let (n_b, q) = b.dim();
    assert_eq!(n_a, n_b, "A and B must have same number of rows");

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

/// Compute A * B^T using faer's SIMD-optimized GEMM.
/// For A of shape (m, k) and B of shape (n, k), this computes the (m, n) result.
#[inline]
pub fn fast_abt<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (m, k_a) = a.dim();
    let (n, k_b) = b.dim();
    assert_eq!(
        k_a, k_b,
        "A and B must have same number of columns for A·Bᵀ"
    );

    if !should_use_faer_matmul(m, n, k_a) {
        return a.dot(&b.t());
    }

    let mut result = Mat::<f64>::zeros(m, n);
    let aview = FaerArrayView::new(a);
    let bview = FaerArrayView::new(b);
    let par = matmul_parallelism(m, n, k_a);
    matmul(
        result.as_mut(),
        Accum::Replace,
        aview.as_ref(),
        bview.as_ref().transpose(),
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
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_ab(a.view(), b.view()))
    {
        return out;
    }
    let n = a.nrows();
    let q = b.ncols();
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
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_av(a.view(), v.view()))
    {
        return out;
    }
    fast_av_impl(a, v)
}

#[inline]
fn fast_av_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat};

    let (n, p) = a.dim();
    assert_eq!(p, v.len(), "A cols must match v length");

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
pub fn fast_av_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    fast_av_into_impl(a, v, out);
}

#[inline]
fn fast_av_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(v.len(), p, "vector length must match A cols");
    assert_eq!(out.len(), n, "output length must match A rows");

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

/// Compute A * v into a pre-allocated `ArrayViewMut1` slice. Like
/// [`fast_av_into`] but accepts a writable slice rather than `&mut Array1`,
/// so callers can write directly into a sub-range of a larger buffer
/// without intermediate allocation.
///
/// `out` must have length n where A is (n, p) and v is length p.
#[inline]
pub fn fast_av_view_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: ArrayViewMut1<'_, f64>,
) {
    fast_av_view_into_impl(a, v, out);
}

#[inline]
fn fast_av_view_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(v.len(), p, "vector length must match A cols");
    assert_eq!(out.len(), n, "output length must match A rows");

    if !should_use_faer_matmul(n, 1, p) {
        let prod = a.dot(v);
        out.assign(&prod);
        return;
    }

    let len = out.len();
    let stride = out.strides()[0];
    // SAFETY: out.as_mut_ptr() is ndarray's logical first-element pointer, and
    // len plus the signed element stride describe every initialized element of
    // this uniquely borrowed view for the returned len×1 MatMut lifetime.
    let outview = unsafe {
        MatMut::from_raw_parts_mut(
            out.as_mut_ptr(),
            len,
            1,
            stride,
            0, // col stride irrelevant for 1 column
        )
    };

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();
    let par = matmul_parallelism(n, 1, p);
    matmul(outview, Accum::Replace, a_ref, v_ref, 1.0, par);
}

/// Compute A^T * v using faer's SIMD-optimized GEMV.
/// For A of shape (n, p) and v of shape (n,), this computes the (p,) result.
#[inline]
pub fn fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_atv(a.view(), v.view()))
    {
        return out;
    }
    fast_atv_impl(a, v)
}

#[inline]
fn fast_atv_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Array1<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(n, v.len(), "A rows must match v length");

    // For very small arrays, ndarray might be faster
    if !should_use_faer_matmul(p, 1, n) {
        return a.t().dot(v);
    }

    let mut out = Array1::<f64>::zeros(p);
    let mut outview = array1_to_col_matmut(&mut out);

    let aview = FaerArrayView::new(a);
    let vview = FaerColView::new(v);
    let a_ref = aview.as_ref();
    let v_ref = vview.as_ref();

    // dst = A^T * v (treating v as n×1 matrix)
    let par = matmul_parallelism(p, 1, n);
    matmul(
        outview.as_mut(),
        Accum::Replace,
        a_ref.transpose(),
        v_ref,
        1.0,
        par,
    );

    out
}

/// Compute A^T * v into a pre-allocated output buffer.
/// `out` must be length p where A is (n, p) and v is length n.
#[inline]
pub fn fast_atv_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    fast_atv_into_impl(a, v, out);
}

#[inline]
fn fast_atv_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    out: &mut Array1<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    assert_eq!(v.len(), n, "vector length must match A rows");
    assert_eq!(out.len(), p, "output length must match A cols");

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

/// Compute A^T * diag(W) * A using streaming chunks to avoid O(n*p) allocation.
#[inline]
pub fn fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
) -> Array2<f64> {
    assert_eq!(
        x.nrows(),
        w.len(),
        "fast_xt_diag_x row/weight length mismatch"
    );
    if let Some(out) =
        crate::gpu_hook::gpu_dispatch().and_then(|d| d.try_fast_xt_diag_x(x.view(), w.view()))
    {
        return out;
    }
    let p = x.ncols();
    fast_xt_diag_x_with_parallelism(x, w, matmul_parallelism(p, p, x.nrows()))
}

/// Compute A^T * diag(W) * A with an explicit faer parallelism policy for
/// callers that parallelize multiple independent Hessian blocks externally.
#[inline]
pub fn fast_xt_diag_x_with_parallelism<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    par: Par,
) -> Array2<f64> {
    assert_eq!(
        x.nrows(),
        w.len(),
        "fast_xt_diag_x_with_parallelism row/weight length mismatch"
    );
    fast_xt_diag_x_with_parallelism_impl(x, w, par)
}

#[inline]
fn fast_xt_diag_x_with_parallelism_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    par: Par,
) -> Array2<f64> {
    use ndarray::ShapeBuilder;

    let p = x.ncols();
    // F-order result so the symmetric lower-triangle accumulation writes
    // column-contiguously; the kernel mirrors to a full symmetric matrix.
    let mut result = Array2::<f64>::zeros((p, p).f());
    stream_weighted_crossprod_into(
        x,
        w,
        &mut result,
        CrossprodStructure::SymmetricLower,
        CrossprodAccum::Replace,
        par,
    );
    result
}

/// Output packaging for [`stream_weighted_crossprod_into`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CrossprodStructure {
    /// Compute every entry of the (symmetric) Gram via full GEMM.
    Full,
    /// Accumulate only the lower triangle via triangular matmul (~50% fewer
    /// FLOPs), then mirror once into the upper triangle for a full symmetric
    /// result. Mathematically identical output to [`Full`](Self::Full).
    SymmetricLower,
}

/// Accumulation policy for [`stream_weighted_crossprod_into`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CrossprodAccum {
    /// Overwrite `out` with `Xᵀ·diag(W)·X`, ignoring prior contents.
    Replace,
    /// Add `Xᵀ·diag(W)·X` into the existing contents of `out`.
    Add,
}

/// Shared dense weighted-Gram kernel: accumulate `Xᵀ·diag(W)·X` into `out`.
///
/// This is the single tuned implementation of the chunked row-scaling +
/// matmul strategy; the matrix-returning (`fast_xt_diag_x*`) entry points and
/// stream-in callers share it so that performance tuning, negative-weight
/// handling, chunk sizing, and layout fixes land in exactly one place.
///
/// Computes the product as `Xᵀ·(W·X)` to preserve the sign of `W`: the prior
/// `sqrt(max(0, w))`-then-Gram form clipped negative weights to zero, which
/// corrupted observed-Hessian assembly when any block carried heavy residuals
/// (e.g. under the logb σ link).
///
/// Peak working-set allocation is `chunk_rows × p × 8` bytes (~8 MB) rather
/// than `n × p × 8` bytes for a materialized `W·X`.
///
/// `out` must be `p × p`. With [`CrossprodStructure::SymmetricLower`] the
/// lower triangle is accumulated and then mirrored, so on return `out` holds
/// the full symmetric matrix regardless of `structure`.
pub fn stream_weighted_crossprod_into<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    out: &mut Array2<f64>,
    structure: CrossprodStructure,
    accum: CrossprodAccum,
    par: Par,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri_matmul};
    use ndarray::s;

    let (n, p) = x.dim();
    assert_eq!(n, w.len(), "X rows must match W length");
    assert_eq!(out.nrows(), p, "output rows must match X cols");
    assert_eq!(out.ncols(), p, "output cols must match X cols");
    if p == 0 {
        return;
    }
    if n == 0 {
        if accum == CrossprodAccum::Replace {
            out.fill(0.0);
        }
        return;
    }

    if !should_use_faer_matmul(p, p, n) {
        // Tiny products: ndarray's own GEMM avoids faer setup overhead.
        let w_x = Array2::from_shape_fn((n, p), |(i, j)| w[i] * x[[i, j]]);
        let gram = x.t().dot(&w_x);
        match accum {
            CrossprodAccum::Replace => out.assign(&gram),
            CrossprodAccum::Add => *out += &gram,
        }
        return;
    }

    // Streaming chunked: peak allocation is chunk_rows × p instead of n × p.
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    let chunk_rows = (TARGET_BYTES / (p.max(1) * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n);

    // Triangular accumulation requires a zero baseline in the lower triangle
    // because each chunk's `Accum::Add` lands there; for a Replace request we
    // zero up front and add every chunk, for an Add request the caller's
    // contents are preserved and every chunk adds on top.
    if accum == CrossprodAccum::Replace {
        out.fill(0.0);
    }

    // Row-major wx_chunk so the per-row scaling loop has stride-1 writes
    // alongside stride-1 reads from a row-major X. An F-order wx_chunk would
    // force strided writes by `chunk_rows`, breaking vectorization and cache
    // locality on the per-PIRLS-iter Hessian assembly. faer's matmul handles
    // either layout via FaerArrayView.
    let mut wx_chunk = Array2::<f64>::zeros((chunk_rows, p));

    let x_is_row_major = x.is_standard_layout();
    let w_slice_opt = w.as_slice();

    // Scope the faer mutable view so its borrow on `out` ends before the
    // symmetric mirror step.
    {
        let mut out_view = array2_to_matmut(out);
        for start in (0..n).step_by(chunk_rows) {
            let rows = (n - start).min(chunk_rows);
            {
                let chunk_slice = wx_chunk
                    .as_slice_mut()
                    .expect("row-major chunk is contiguous");
                if x_is_row_major && let (Some(x_all), Some(w_all)) = (x.as_slice(), w_slice_opt) {
                    for local in 0..rows {
                        let src = start + local;
                        let wi = w_all[src];
                        let src_off = src * p;
                        let dst_off = local * p;
                        let src_row = &x_all[src_off..src_off + p];
                        let dst_row = &mut chunk_slice[dst_off..dst_off + p];
                        for col in 0..p {
                            dst_row[col] = src_row[col] * wi;
                        }
                    }
                } else {
                    let x_slice = x.slice(s![start..start + rows, ..]);
                    for local in 0..rows {
                        let wi = w[start + local];
                        let xrow = x_slice.row(local);
                        let dst_off = local * p;
                        let dst_row = &mut chunk_slice[dst_off..dst_off + p];
                        for (col, xij) in xrow.iter().enumerate() {
                            dst_row[col] = xij * wi;
                        }
                    }
                }
            }
            let x_slice = x.slice(s![start..start + rows, ..]);
            let wx_slice = wx_chunk.slice(s![0..rows, ..]);
            let x_view = FaerArrayView::new(&x_slice);
            let wx_view = FaerArrayView::new(&wx_slice);
            match structure {
                CrossprodStructure::SymmetricLower => {
                    // X^T diag(W) X is symmetric; accumulate the lower triangle
                    // only, then mirror once after the chunk loop. ~50% fewer
                    // FLOPs vs. full GEMM.
                    tri_matmul(
                        out_view.as_mut(),
                        BlockStructure::TriangularLower,
                        Accum::Add,
                        x_view.as_ref().transpose(),
                        BlockStructure::Rectangular,
                        wx_view.as_ref(),
                        BlockStructure::Rectangular,
                        1.0,
                        par,
                    );
                }
                CrossprodStructure::Full => {
                    matmul(
                        out_view.as_mut(),
                        Accum::Add,
                        x_view.as_ref().transpose(),
                        wx_view.as_ref(),
                        1.0,
                        par,
                    );
                }
            }
        }
    }

    if structure == CrossprodStructure::SymmetricLower {
        // Mirror lower triangle to upper for a full symmetric output.
        for i in 0..p {
            for j in (i + 1)..p {
                out[[i, j]] = out[[j, i]];
            }
        }
    }
}

/// Compute A^T * diag(W) * B using streaming chunks.
#[inline]
pub fn fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Array2<f64> {
    assert_eq!(x.nrows(), y.nrows(), "fast_xt_diag_y X/Y row mismatch");
    assert_eq!(
        y.nrows(),
        w.len(),
        "fast_xt_diag_y row/weight length mismatch"
    );
    if let Some(out) = crate::gpu_hook::gpu_dispatch()
        .and_then(|d| d.try_fast_xt_diag_y(x.view(), w.view(), y.view()))
    {
        return out;
    }
    fast_xt_diag_y_impl(x, w, y)
}

#[inline]
fn fast_xt_diag_y_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Array2<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use ndarray::{ShapeBuilder, s};

    let (n, q) = y.dim();
    let px = x.ncols();
    assert_eq!(n, w.len(), "Y rows must match W length");
    assert_eq!(n, x.nrows(), "X rows must match Y rows");
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
    // Row-major wy_chunk — same rationale as fast_xt_diag_x: stride-1
    // writes alongside stride-1 reads from a row-major Y.
    let mut wy_chunk = Array2::<f64>::zeros((chunk_rows, q));

    let y_is_row_major = y.is_standard_layout();
    let w_slice_opt = w.as_slice();

    {
        let mut out_view = array2_to_matmut(&mut result);

        for start in (0..n).step_by(chunk_rows) {
            let rows = (n - start).min(chunk_rows);
            {
                let chunk_slice = wy_chunk
                    .as_slice_mut()
                    .expect("row-major chunk is contiguous");
                if y_is_row_major && let (Some(y_all), Some(w_all)) = (y.as_slice(), w_slice_opt) {
                    for local in 0..rows {
                        let src = start + local;
                        let wi = w_all[src];
                        let src_off = src * q;
                        let dst_off = local * q;
                        let src_row = &y_all[src_off..src_off + q];
                        let dst_row = &mut chunk_slice[dst_off..dst_off + q];
                        for col in 0..q {
                            dst_row[col] = src_row[col] * wi;
                        }
                    }
                } else {
                    let y_slice = y.slice(s![start..start + rows, ..]);
                    for local in 0..rows {
                        let wi = w[start + local];
                        let yrow = y_slice.row(local);
                        let dst_off = local * q;
                        let dst_row = &mut chunk_slice[dst_off..dst_off + q];
                        for (col, yij) in yrow.iter().enumerate() {
                            dst_row[col] = yij * wi;
                        }
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
    }

    result
}

/// Compute the 2×2 block joint Hessian in a single streaming pass:
///   [X_a^T diag(w_aa) X_a,   X_a^T diag(w_ab) X_b]
///   [X_b^T diag(w_ab) X_a,   X_b^T diag(w_bb) X_b]
///
/// This reads X_a and X_b once per chunk instead of twice (saving 50% bandwidth).
pub fn fast_joint_hessian_2x2<
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
    S5: Data<Elem = f64>,
>(
    x_a: &ArrayBase<S1, Ix2>,
    x_b: &ArrayBase<S2, Ix2>,
    w_aa: &ArrayBase<S3, Ix1>,
    w_ab: &ArrayBase<S4, Ix1>,
    w_bb: &ArrayBase<S5, Ix1>,
) -> Array2<f64> {
    if let Some(out) = crate::gpu_hook::gpu_dispatch().and_then(|d| {
        d.try_fast_joint_hessian_2x2(
            x_a.view(),
            x_b.view(),
            w_aa.view(),
            w_ab.view(),
            w_bb.view(),
        )
    }) {
        return out;
    }
    fast_joint_hessian_2x2_impl(x_a, x_b, w_aa, w_ab, w_bb)
}

#[inline]
fn fast_joint_hessian_2x2_impl<
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
    S5: Data<Elem = f64>,
>(
    x_a: &ArrayBase<S1, Ix2>,
    x_b: &ArrayBase<S2, Ix2>,
    w_aa: &ArrayBase<S3, Ix1>,
    w_ab: &ArrayBase<S4, Ix1>,
    w_bb: &ArrayBase<S5, Ix1>,
) -> Array2<f64> {
    use faer::Accum;
    use faer::linalg::matmul::matmul;
    use ndarray::{ShapeBuilder, s};

    let n = x_a.nrows();
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    let total = pa + pb;
    assert_eq!(n, x_b.nrows());
    assert_eq!(n, w_aa.len());
    assert_eq!(n, w_ab.len());
    assert_eq!(n, w_bb.len());

    if n == 0 || total == 0 {
        return Array2::<f64>::zeros((total, total));
    }

    // For small problems, fall back to separate computations
    if !should_use_faer_matmul(pa.max(pb), pa.max(pb), n) {
        let waa_xa = Array2::from_shape_fn((n, pa), |(i, j)| w_aa[i] * x_a[[i, j]]);
        let wab_xb = Array2::from_shape_fn((n, pb), |(i, j)| w_ab[i] * x_b[[i, j]]);
        let wbb_xb = Array2::from_shape_fn((n, pb), |(i, j)| w_bb[i] * x_b[[i, j]]);
        let mut out = Array2::<f64>::zeros((total, total));
        out.slice_mut(s![..pa, ..pa]).assign(&x_a.t().dot(&waa_xa));
        out.slice_mut(s![..pa, pa..]).assign(&x_a.t().dot(&wab_xb));
        out.slice_mut(s![pa.., pa..]).assign(&x_b.t().dot(&wbb_xb));
        // Mirror upper to lower
        for i in 0..total {
            for j in 0..i {
                out[[i, j]] = out[[j, i]];
            }
        }
        return out;
    }

    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    // Need buffers for: waa_xa(chunk×pa) + wab_xb(chunk×pb) + wbb_xb(chunk×pb)
    let cols_needed = pa + 2 * pb;
    let chunk_rows = (TARGET_BYTES / (cols_needed.max(1) * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n);

    let mut out = Array2::<f64>::zeros((total, total).f());
    // Row-major weighted buffers so the per-row scale loops have stride-1
    // writes (the previous F-order layout strided writes by chunk_rows
    // across `pa` / `pb`, gutting vectorization on the per-PIRLS-iter
    // joint Hessian assembly). faer's matmul handles either layout.
    let mut waa_xa_buf = Array2::<f64>::zeros((chunk_rows, pa));
    let mut wab_xb_buf = Array2::<f64>::zeros((chunk_rows, pb));
    let mut wbb_xb_buf = Array2::<f64>::zeros((chunk_rows, pb));

    let xa_is_row_major = x_a.is_standard_layout();
    let xb_is_row_major = x_b.is_standard_layout();
    let waa_slice_opt = w_aa.as_slice();
    let wab_slice_opt = w_ab.as_slice();
    let wbb_slice_opt = w_bb.as_slice();

    {
        let mut out_mat = array2_to_matmut(&mut out);

        for start in (0..n).step_by(chunk_rows) {
            let rows = (n - start).min(chunk_rows);
            let xa_slice = x_a.slice(s![start..start + rows, ..]);
            let xb_slice = x_b.slice(s![start..start + rows, ..]);

            // Weight X_a and X_b in a single pass through this chunk.
            {
                let waa_chunk = waa_xa_buf
                    .as_slice_mut()
                    .expect("row-major waa chunk is contiguous");
                let wab_chunk = wab_xb_buf
                    .as_slice_mut()
                    .expect("row-major wab chunk is contiguous");
                let wbb_chunk = wbb_xb_buf
                    .as_slice_mut()
                    .expect("row-major wbb chunk is contiguous");

                if xa_is_row_major
                    && xb_is_row_major
                    && let (Some(xa_all), Some(xb_all)) = (x_a.as_slice(), x_b.as_slice())
                    && let (Some(waa_all), Some(wab_all), Some(wbb_all)) =
                        (waa_slice_opt, wab_slice_opt, wbb_slice_opt)
                {
                    for local in 0..rows {
                        let i = start + local;
                        let waa_i = waa_all[i];
                        let wab_i = wab_all[i];
                        let wbb_i = wbb_all[i];
                        let xa_off = i * pa;
                        let xa_row = &xa_all[xa_off..xa_off + pa];
                        let xb_off = i * pb;
                        let xb_row = &xb_all[xb_off..xb_off + pb];
                        let waa_off = local * pa;
                        let wab_off = local * pb;
                        let wbb_off = local * pb;
                        let waa_row = &mut waa_chunk[waa_off..waa_off + pa];
                        for col in 0..pa {
                            waa_row[col] = xa_row[col] * waa_i;
                        }
                        let wab_row = &mut wab_chunk[wab_off..wab_off + pb];
                        let wbb_row = &mut wbb_chunk[wbb_off..wbb_off + pb];
                        for col in 0..pb {
                            let xij = xb_row[col];
                            wab_row[col] = xij * wab_i;
                            wbb_row[col] = xij * wbb_i;
                        }
                    }
                } else {
                    for local in 0..rows {
                        let i = start + local;
                        let waa_i = w_aa[i];
                        let wab_i = w_ab[i];
                        let wbb_i = w_bb[i];
                        let waa_off = local * pa;
                        let wab_off = local * pb;
                        let wbb_off = local * pb;
                        let waa_row = &mut waa_chunk[waa_off..waa_off + pa];
                        let xa_row = xa_slice.row(local);
                        for (col, xij) in xa_row.iter().enumerate() {
                            waa_row[col] = xij * waa_i;
                        }
                        let wab_row = &mut wab_chunk[wab_off..wab_off + pb];
                        let wbb_row = &mut wbb_chunk[wbb_off..wbb_off + pb];
                        let xb_row = xb_slice.row(local);
                        for (col, xij) in xb_row.iter().enumerate() {
                            wab_row[col] = xij * wab_i;
                            wbb_row[col] = xij * wbb_i;
                        }
                    }
                }
            }

            let xa_view = FaerArrayView::new(&xa_slice);
            let xb_view = FaerArrayView::new(&xb_slice);
            let waa_xa_slice = waa_xa_buf.slice(s![0..rows, ..]);
            let wab_xb_slice = wab_xb_buf.slice(s![0..rows, ..]);
            let wbb_xb_slice = wbb_xb_buf.slice(s![0..rows, ..]);
            let waa_xa_view = FaerArrayView::new(&waa_xa_slice);
            let wab_xb_view = FaerArrayView::new(&wab_xb_slice);
            let wbb_xb_view = FaerArrayView::new(&wbb_xb_slice);

            // Block [0..pa, 0..pa]: X_a^T diag(w_aa) X_a
            matmul(
                out_mat.rb_mut().submatrix_mut(0, 0, pa, pa),
                Accum::Add,
                xa_view.as_ref().transpose(),
                waa_xa_view.as_ref(),
                1.0,
                matmul_parallelism(pa, pa, rows),
            );
            // Block [0..pa, pa..total]: X_a^T diag(w_ab) X_b
            matmul(
                out_mat.rb_mut().submatrix_mut(0, pa, pa, pb),
                Accum::Add,
                xa_view.as_ref().transpose(),
                wab_xb_view.as_ref(),
                1.0,
                matmul_parallelism(pa, pb, rows),
            );
            // Block [pa..total, pa..total]: X_b^T diag(w_bb) X_b
            matmul(
                out_mat.rb_mut().submatrix_mut(pa, pa, pb, pb),
                Accum::Add,
                xb_view.as_ref().transpose(),
                wbb_xb_view.as_ref(),
                1.0,
                matmul_parallelism(pb, pb, rows),
            );
        }
    } // out_mat dropped
    // Mirror upper triangle to lower
    for i in 0..total {
        for j in 0..i {
            out[[i, j]] = out[[j, i]];
        }
    }
    out
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
    fast_ab_into_impl(a, b, out);
}

#[inline]
fn fast_ab_into_impl<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    out: &mut Array2<f64>,
) {
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let (n, p) = a.dim();
    let (p_b, q) = b.dim();
    assert_eq!(p, p_b, "A and B must have compatible inner dimensions");
    assert_eq!(out.dim(), (n, q), "output dimensions must match A*B result");

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
    #[inline]
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
        // SAFETY: ptr/shape/strides come from either a live ndarray view
        // (positive strides, validated bounds/alignment) or the owned
        // compact copy held inside this wrapper — no mutable aliasing.
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
    #[inline]
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
        // SAFETY: ptr/len/stride come from either a live ndarray column
        // (positive stride, validated bounds/alignment) or the owned
        // compact copy; ncols=1 so the 0 col-stride is unused.
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
            .map_err(|_| FaerLinalgError::SvdNoConvergence {
                context: "faer SVD singular values only",
            })?;
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
        .map_err(|_| FaerLinalgError::SvdNoConvergence {
            context: "faer SVD with vectors",
        })?;

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
        fn try_eigh(
            matrix: &Array2<f64>,
            side: Side,
        ) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
            let faerview = FaerArrayView::new(matrix);
            let eigen = catch_unwind(AssertUnwindSafe(|| {
                faerview.as_ref().self_adjoint_eigen(side)
            }))
            .map_err(|_| FaerLinalgError::FactorizationFailed {
                context: "self-adjoint eigendecomposition panic boundary",
            })?
            .map_err(FaerLinalgError::SelfAdjointEigen)?;
            let values = diag_to_array(eigen.S());
            let vectors = mat_to_array(eigen.U());
            Ok((values, vectors))
        }

        let owned = self.to_owned();
        if owned.nrows() != owned.ncols() {
            return Err(FaerLinalgError::FactorizationFailed {
                context: "self-adjoint eigendecomposition non-square input",
            });
        }
        if owned.nrows() == 0 {
            return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
        }
        if owned.iter().any(|value| !value.is_finite()) {
            return Err(FaerLinalgError::SelfAdjointEigenNonFiniteInput {
                context: "self-adjoint eigendecomposition input validation",
            });
        }
        if let Ok((evals, evecs)) = try_eigh(&owned, side)
            && evals.iter().all(|value| value.is_finite())
            && evecs.iter().all(|value| value.is_finite())
        {
            return Ok((evals, evecs));
        }

        let mut repaired = owned.clone();
        crate::matrix::symmetrize_in_place(&mut repaired);

        let scale = repaired
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
            .max(1.0);
        let scaled = repaired.mapv(|value| value / scale);
        // Relative diagonal-jitter ladder for the eigendecomposition repair: the
        // matrix is pre-scaled to unit max-abs, so these are fractions of its
        // scale. We try the unperturbed matrix first, then escalate the ridge by
        // two decades per attempt until the factorization yields all-finite
        // eigenpairs, accepting the smallest jitter that succeeds.
        const JITTER_SCHEDULE: [f64; 6] = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4];
        let jitter_schedule = JITTER_SCHEDULE;
        let mut last_error = FaerLinalgError::FactorizationFailed {
            context: "self-adjoint eigendecomposition repair attempts",
        };

        for &jitter in &jitter_schedule {
            let mut candidate = scaled.clone();
            if jitter > 0.0 {
                let n = candidate.nrows();
                for i in 0..n {
                    candidate[[i, i]] += jitter;
                }
            }

            match try_eigh(&candidate, side) {
                Ok((mut evals, evecs))
                    if evals.iter().all(|value| value.is_finite())
                        && evecs.iter().all(|value| value.is_finite()) =>
                {
                    for value in &mut evals {
                        *value = (*value - jitter) * scale;
                    }
                    return Ok((evals, evecs));
                }
                Ok((_, _)) => {
                    last_error = FaerLinalgError::SelfAdjointEigenNonFiniteInput {
                        context: "self-adjoint eigendecomposition repaired output validation",
                    };
                }
                Err(err) => {
                    last_error = err;
                }
            }
        }

        Err(last_error)
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

    pub fn solve_mat_in_place(&self, rhs: &mut Array2<f64>) {
        let mut rhsview = array2_to_matmut(rhs);
        self.factor.solve_in_place(rhsview.as_mut());
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
        self.solve_mat_in_place(out);
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
///
/// The trailing columns of `Q` are reconstructed by applying the stored
/// Householder reflector sequence to canonical basis vectors. When `A` is
/// numerically rank zero (e.g. an entirely unpenalized block penalty in a
/// parametric-only GLM), *every* reflector is degenerate — the Householder
/// vector of a zero column has zero norm, so faer's coefficients become
/// non-finite and the reconstructed basis is filled with `NaN`. Mathematically
/// a rank-zero `m×n` matrix has `null(A^T) = R^m`, whose canonical orthonormal
/// basis is the identity, so we return `I_m` directly instead of routing through
/// the (undefined) reflectors. This keeps every downstream consumer — REML
/// null-space log-determinants, identifiability audits — finite and exact for
/// the fully-unpenalized case. For `rank >= 1` at least one well-defined
/// reflector seeds the block, and the reconstruction stays finite.
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
    } else if rank == 0 {
        // Numerically rank-zero input: the whole space is the null space.
        // Return the canonical orthonormal basis directly; the Householder
        // reflectors of a zero matrix are degenerate and would yield NaN.
        Array2::<f64>::eye(a.nrows())
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
pub const fn default_rrqr_rank_alpha() -> f64 {
    RRQR_RANK_ALPHA
}

/// Result of a column-pivoted QR with rank detection and column permutation.
///
/// `A · P = Q · R` where the permutation `P` is exposed as the forward index
/// array: column `j` of `A · P` corresponds to original column
/// `column_permutation[j]` of `A`. With rank `r < min(m, n)`, the trailing
/// `min(m, n) - r` entries of `column_permutation` name the columns that the
/// pivoted QR demoted past the rank threshold — i.e., the columns identified
/// as redundant. Identifiability auditors (`identifiability::audit`)
/// use that suffix to attribute `DroppedColumn` entries to specific original
/// columns.
pub struct RrqrWithPermutation {
    pub rank: usize,
    pub column_permutation: Vec<usize>,
    pub leading_diag_abs: f64,
    pub rank_tol: f64,
}

/// Column-pivoted rank-revealing QR returning the rank, the column permutation,
/// and the rank-detection tolerance. Use this when callers need to name which
/// columns the pivoted QR demoted past the rank threshold.
///
/// The rank cutoff matches [`rrqr_nullspace_basis`]: a column-pivoted QR is
/// computed on `a`; columns with `|R[i, i]| > tol` count toward the rank,
/// where `tol = rank_alpha · eps · max(m, n, 1) · max(|R[0, 0]|, 1)`. Returns
/// `Err` when `a` has zero rows.
pub fn rrqr_with_permutation<S: Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
    rank_alpha: f64,
) -> Result<RrqrWithPermutation, FaerLinalgError> {
    if a.nrows() == 0 {
        return Err(FaerLinalgError::FactorizationFailed {
            context: "rrqr_with_permutation: input has zero rows",
        });
    }
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
    let (forward, _inverse) = qr.P().arrays();
    let column_permutation: Vec<usize> = forward.iter().copied().map(|idx| idx.unbound()).collect();
    Ok(RrqrWithPermutation {
        rank,
        column_permutation,
        leading_diag_abs: leading_diag,
        rank_tol: tol,
    })
}

/// Result of a Gram-driven column-pivoted RRQR (see
/// [`rrqr_from_gram_with_permutation`]). Carries the same rank / permutation /
/// tolerance as [`RrqrWithPermutation`], plus a `verdict_margin` that measures
/// how unambiguous the rank cut is — the ratio between the smallest *kept*
/// pivot and the rank tolerance. A large margin means squaring the design into
/// a Gram could not have flipped any rank decision; a small margin means the
/// verdict sits near the cliff and the caller should re-confirm on the full
/// (un-squared) design to stay bit-exact.
pub struct RrqrFromGram {
    pub rank: usize,
    pub column_permutation: Vec<usize>,
    pub rank_tol: f64,
    /// Leading pivot magnitude `|R[0,0]|` of the square-root factor — equal to
    /// the largest column norm of the original tall design (col-piv QR pivots the
    /// largest-norm column first), so it matches the tall path's
    /// `RrqrWithPermutation::leading_diag_abs`.
    pub leading_diag_abs: f64,
    /// `min_kept_pivot / rank_tol` (∞ when full rank with no kept pivot below
    /// tol, i.e. every pivot is comfortably above; `0` when rank is 0).
    pub verdict_margin: f64,
}

/// Column-pivoted rank-revealing QR computed from the design's `p × p` Gram
/// `G = AᵀA` (or penalty-augmented `AᵀA + SᵀS`) instead of from the tall
/// `m × p` design itself.
///
/// # Why this is exact (in exact arithmetic)
///
/// Column-pivoted QR selects, at each step, the not-yet-pivoted column with the
/// largest residual norm, where the residual is the part orthogonal to the
/// already-chosen columns. Those residual norms — and the resulting pivot
/// sequence, the diagonal magnitudes `|R[i,i]|`, and hence the rank cut — are a
/// function of the column *inner products* only, i.e. of the Gram `G`. Running
/// col-piv QR on the Cholesky factor `R₀` of `G` (`R₀ᵀR₀ = G`, `R₀` is `p × p`)
/// reproduces the identical pivot order and identical `|R[i,i]|` as col-piv QR
/// on the original `m × p` matrix, because both see the same column geometry.
/// This is the standard "pivoted QR depends only on the Gram" identity and lets
/// the joint identifiability rank verdict run in `O(p³)` instead of streaming
/// all `m ≈ 2·10⁵` rows again.
///
/// # Tolerance
///
/// The rank cutoff must match what the tall-matrix [`rrqr_with_permutation`]
/// would have used, so the caller passes `m_rows` (the row count of the
/// original tall design, including any appended penalty rows). The tolerance is
/// `rank_alpha · eps · max(m_rows, p) · max(|R[0,0]|, 1)` — bit-identical to the
/// tall path, since `|R[0,0]|` (the leading pivot magnitude = largest column
/// norm) is the same in both factorizations.
///
/// # Finite-precision guard
///
/// Forming `G = AᵀA` squares the condition number, so a rank decision that sits
/// right at the tolerance cliff could in principle flip. The returned
/// `verdict_margin` lets the caller detect that case and fall back to the exact
/// tall RRQR; in the overwhelmingly common well-separated case (full column
/// rank, smallest pivot orders of magnitude above tol) the margin is huge and
/// no fallback is needed.
pub fn rrqr_from_gram_with_permutation<S: Data<Elem = f64>>(
    gram: &ArrayBase<S, Ix2>,
    m_rows: usize,
    rank_alpha: f64,
) -> Result<RrqrFromGram, FaerLinalgError> {
    let p = gram.ncols();
    if p == 0 {
        return Ok(RrqrFromGram {
            rank: 0,
            column_permutation: Vec::new(),
            rank_tol: 0.0,
            leading_diag_abs: 0.0,
            verdict_margin: 0.0,
        });
    }
    if gram.nrows() != p {
        return Err(FaerLinalgError::FactorizationFailed {
            context: "rrqr_from_gram_with_permutation: Gram is not square",
        });
    }
    // Symmetric square-root factor F (p×p) with FᵀF = G. The Gram is PSD by
    // construction (AᵀA), so its eigendecomposition G = V·diag(λ)·Vᵀ gives the
    // factor F = diag(√λ₊)·Vᵀ (rows indexed by eigenpair, columns by original
    // design column). Any factor with FᵀF = G reproduces the same column
    // geometry, which is all col-piv QR consumes — we use the eigen square root
    // rather than a bare Cholesky because Cholesky fails on the numerically
    // semidefinite Gram that is exactly the rank-deficient case we must classify.
    // Tiny-negative eigenvalues from finite precision are clamped to zero.
    let (evals, evecs) = gram.eigh(Side::Lower)?;
    let mut f = Array2::<f64>::zeros((p, p));
    for k in 0..p {
        let scale = evals[k].max(0.0).sqrt();
        if scale == 0.0 {
            continue;
        }
        for i in 0..p {
            f[[k, i]] = scale * evecs[[i, k]];
        }
    }
    // Single col-piv QR on F. Its pivot order, per-pivot |R[i,i]| magnitudes,
    // and leading pivot equal those of col-piv QR on the original tall design
    // (FᵀF = G), so this reproduces the exact tall-path geometry.
    let faer_f = FaerArrayView::new(&f);
    let qr = faer_f.as_ref().col_piv_qr();
    let r = qr.thin_R();
    let diag_len = r.nrows().min(r.ncols());
    let pivots: Vec<f64> = (0..diag_len).map(|i| r[(i, i)].abs()).collect();
    let leading_diag = pivots.first().copied().unwrap_or(0.0);
    let (forward, _inverse) = qr.P().arrays();
    let column_permutation: Vec<usize> = forward.iter().copied().map(|idx| idx.unbound()).collect();
    // Re-scale the tolerance from F's `max(p, p)=p` row dimension to the
    // original tall design's `max(m_rows, p)`, keeping the rank cut bit-
    // identical to what the tall [`rrqr_with_permutation`] would have produced.
    let tol = rank_alpha * f64::EPSILON * (m_rows.max(p).max(1) as f64) * leading_diag.max(1.0);
    let rank = pivots.iter().filter(|&&v| v > tol).count();
    let min_kept = pivots[..rank].iter().copied().fold(f64::INFINITY, f64::min);
    let max_dropped = pivots[rank..].iter().copied().fold(0.0f64, f64::max);
    // Margin: how far the verdict is from the cliff. Use the smaller of
    // (min_kept / tol) and (tol / max_dropped) so a near-tol dropped pivot also
    // shrinks the margin. A margin ≫ 1 means no rank decision could flip.
    let kept_margin = if rank == 0 {
        f64::INFINITY
    } else {
        min_kept / tol
    };
    let dropped_margin = if rank == diag_len {
        f64::INFINITY
    } else {
        tol / max_dropped.max(f64::MIN_POSITIVE)
    };
    // Gram-squaring precision floor. Forming `G = XᵀX` collapses the bottom half
    // of the spectrum: a true singular value below `√ε · σ_max` is lost in the
    // rounding of `G` (its squared value `σ² < ε·σ_max²` underflows the Gram's
    // representable range), and the eigen-square-root then RESURRECTS it as a
    // SPURIOUS pivot of magnitude `≈ √(ε·σ_max²) = √ε · σ_max` — orders of
    // magnitude ABOVE the true σ and above `tol`. That artefact makes col-piv QR
    // on `F` KEEP a column the tall (un-squared) QR would demote: an EXACTLY
    // collinear alias (true σ = 0, so `σ² = 0` floored at `≈ ε·σ_max²`) shows up
    // as a kept pivot near `√ε · leading`, over-ranking the design and dropping
    // nothing (gam#933: a callback-owned column aliased with a higher-priority
    // anchor was never demoted, so the reduction never ran and the MAP-uniqueness
    // check then fired on the raw collinear joint design). `min_kept / tol` does
    // NOT catch this — the spurious pivot sits comfortably above `tol`, so the
    // existing margin reports a falsely-confident verdict. The honest test is
    // whether the smallest KEPT pivot is itself near the Gram precision floor
    // `√ε · leading`: if so, the Gram path cannot distinguish it from a true zero
    // and the verdict MUST be re-confirmed on the full-precision tall design.
    // Encode that as a third margin term `min_kept / (√ε · leading)` so a kept
    // pivot in the floor regime shrinks `verdict_margin` below the caller's
    // fallback threshold; for a genuinely full-rank design every kept pivot is
    // `≫ √ε · leading` and this term is large, leaving the fast path intact.
    let gram_precision_floor = f64::EPSILON.sqrt() * leading_diag.max(1.0);
    let kept_floor_margin = if rank == 0 {
        f64::INFINITY
    } else {
        min_kept / gram_precision_floor.max(f64::MIN_POSITIVE)
    };
    let verdict_margin = kept_margin.min(dropped_margin).min(kept_floor_margin);
    Ok(RrqrFromGram {
        rank,
        column_permutation,
        rank_tol: tol,
        leading_diag_abs: leading_diag,
        verdict_margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    /// Local mirror of the audit's `JOINT_GRAM_RRQR_MIN_VERDICT_MARGIN` fallback
    /// threshold, used only by the regression tests below to assert the verdict
    /// margin lands on the correct side of the cliff. Kept in sync by value (1e3).
    const JOINT_GRAM_RRQR_TRUST_MARGIN_FOR_TEST: f64 = 1.0e3;

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
    fn rrqr_with_permutation_attributes_redundant_column() {
        // 3 columns, column 2 is a duplicate of column 0 → rank 2, column 2
        // is the redundant one that the pivoted QR should demote past the
        // rank threshold. (Column 1 contributes a different direction.)
        let a = array![
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let result =
            rrqr_with_permutation(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(result.rank, 2);
        assert_eq!(result.column_permutation.len(), 3);
        let demoted = result.column_permutation[result.rank..].to_vec();
        assert!(
            demoted.contains(&2) || demoted.contains(&0),
            "demoted suffix should include one of the aliased columns (0 or 2), got {demoted:?}"
        );
        let mut sorted = result.column_permutation.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![0, 1, 2],
            "permutation must be a valid bijection on 0..n"
        );
    }

    #[test]
    fn rrqr_with_permutation_full_rank_returns_identity_like_order() {
        let a = array![[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]];
        let result =
            rrqr_with_permutation(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(result.rank, 2);
        let mut sorted = result.column_permutation.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1]);
    }

    #[test]
    fn rrqr_with_permutation_rejects_zero_rows() {
        let a = Array2::<f64>::zeros((0, 3));
        assert!(rrqr_with_permutation(&a, default_rrqr_rank_alpha()).is_err());
    }

    #[test]
    fn rrqr_nullspace_basis_square_zero_matrix_is_finite_identity() {
        // Square zero matrix (the parametric-only penalty case): null(A^T) is
        // the whole space, so the basis must be a finite orthonormal 3x3 set.
        let a = Array2::<f64>::zeros((3, 3));
        let (z, rank) =
            rrqr_nullspace_basis(&a, default_rrqr_rank_alpha()).expect("RRQR should succeed");
        assert_eq!(rank, 0);
        assert_eq!(z.dim(), (3, 3));
        assert!(
            z.iter().all(|v| v.is_finite()),
            "square zero matrix produced a non-finite null basis: {z:?}"
        );
        let gram = z.t().dot(&z);
        let ident = Array2::<f64>::eye(3);
        let gram_err = (&gram - &ident)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(gram_err < 1e-10, "Z is not orthonormal: {gram_err:e}");
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

    //
    // Eigendecomposition NoConvergence on pathological matrices
    //
    // These tests lock down the hardened contract for FaerEigh::eigh:
    // non-finite input must be rejected explicitly, while finite symmetric
    // matrices still produce finite spectra.
    //

    #[test]
    fn eigh_on_nan_matrix_rejects_non_finite_input() {
        let mat = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, f64::NAN],
            [0.0, 0.0, f64::NAN, 4.0]
        ];
        let err = mat
            .eigh(Side::Lower)
            .expect_err("non-finite symmetric input must be rejected");
        assert!(matches!(
            err,
            FaerLinalgError::SelfAdjointEigenNonFiniteInput { .. }
        ));
    }

    #[test]
    fn fast_ata_matches_full_gemm_above_threshold() {
        // Pick (n, p) large enough to trigger the faer triangular path
        // (should_use_faer_matmul threshold is MIN_DIM=32, MIN_FLOP_SCALE=64*64).
        let n = 200;
        let p = 40;
        let a: Array2<f64> = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i * 7 + j * 3) as f64).sin() + 0.1 * j as f64
        });
        let expected = a.t().dot(&a);
        let got = fast_ata(&a);
        let max_err = (&got - &expected)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_err < 1e-10, "fast_ata mismatch: {max_err:e}");
        // Output must be fully populated and symmetric.
        for i in 0..p {
            for j in 0..p {
                assert!((got[[i, j]] - got[[j, i]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fast_xt_diag_x_matches_naive_above_threshold() {
        let n = 400;
        let p = 36;
        let x: Array2<f64> =
            Array2::from_shape_fn((n, p), |(i, j)| (i as f64 * 0.1).cos() + j as f64 * 0.05);
        let w: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64 * 0.03).sin());
        // Naive reference: X^T diag(w) X.
        let wx = Array2::from_shape_fn((n, p), |(i, j)| w[i] * x[[i, j]]);
        let expected = x.t().dot(&wx);
        let got = fast_xt_diag_x(&x, &w);
        let max_err = (&got - &expected)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_err < 1e-9, "fast_xt_diag_x mismatch: {max_err:e}");
        for i in 0..p {
            for j in 0..p {
                assert!((got[[i, j]] - got[[j, i]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn stream_weighted_crossprod_full_and_triangular_parity_with_negative_weights() {
        // The stream-in and matrix-returning `fast_xt_diag_x*` packaging modes
        // share one kernel. Both packaging modes — and both accumulation
        // modes — must reproduce the naive `Xᵀ·diag(w)·X` reference, including signed
        // (negative) weights, which the pre-unification sqrt-clip form
        // silently corrupted.
        //
        // Exercise both the streaming faer path (n large enough to clear
        // `should_use_faer_matmul`) and the tiny ndarray fallback (small n,p).
        for &(n, p) in &[(900usize, 40usize), (8usize, 3usize)] {
            let x: Array2<f64> =
                Array2::from_shape_fn((n, p), |(i, j)| (i as f64 * 0.07).cos() + j as f64 * 0.013);
            // Weights span both signs and zero so negative-weight handling and
            // sign preservation are genuinely tested.
            let w: Array1<f64> =
                Array1::from_shape_fn(n, |i| (i as f64 * 0.11).sin() - 0.25 * (i % 3) as f64);
            assert!(
                w.iter().any(|&v| v < 0.0),
                "weight vector must contain negatives to test sign preservation"
            );

            // Naive reference: Xᵀ diag(w) X with signed weights.
            let wx = Array2::from_shape_fn((n, p), |(i, j)| w[i] * x[[i, j]]);
            let expected = x.t().dot(&wx);

            let par = matmul_parallelism(p, p, n);

            // Full output, Replace.
            let mut full = Array2::<f64>::ones((p, p));
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut full,
                CrossprodStructure::Full,
                CrossprodAccum::Replace,
                par,
            );

            // Triangular+mirror output, Replace. Seed with garbage to prove
            // Replace clears prior contents (incl. the upper triangle, which
            // the triangular path only reaches via the mirror).
            let mut tri = Array2::<f64>::from_elem((p, p), -7.0);
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut tri,
                CrossprodStructure::SymmetricLower,
                CrossprodAccum::Replace,
                par,
            );

            let full_err = (&full - &expected)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            let tri_err = (&tri - &expected)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            assert!(
                full_err < 1e-9,
                "full kernel mismatch (n={n}, p={p}): {full_err:e}"
            );
            assert!(
                tri_err < 1e-9,
                "triangular kernel mismatch (n={n}, p={p}): {tri_err:e}"
            );

            // Full and triangular packaging must agree elementwise, and both
            // must be exactly symmetric.
            for i in 0..p {
                for j in 0..p {
                    assert!(
                        (full[[i, j]] - tri[[i, j]]).abs() < 1e-12,
                        "full vs triangular disagree at ({i},{j})"
                    );
                    assert!(
                        (tri[[i, j]] - tri[[j, i]]).abs() < 1e-12,
                        "triangular output not symmetric at ({i},{j})"
                    );
                }
            }

            // Accumulation parity: Add into a pre-filled buffer must equal the
            // prior contents plus the Gram, for both structures.
            let base = Array2::<f64>::from_elem((p, p), 1.5);
            let mut add_full = base.clone();
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut add_full,
                CrossprodStructure::Full,
                CrossprodAccum::Add,
                par,
            );
            let mut add_tri = base.clone();
            stream_weighted_crossprod_into(
                &x,
                &w,
                &mut add_tri,
                CrossprodStructure::SymmetricLower,
                CrossprodAccum::Add,
                par,
            );
            let expected_add = &base + &expected;
            let add_full_err = (&add_full - &expected_add)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            let add_tri_err = (&add_tri - &expected_add)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            assert!(
                add_full_err < 1e-9,
                "full Add mismatch (n={n}, p={p}): {add_full_err:e}"
            );
            assert!(
                add_tri_err < 1e-9,
                "triangular Add mismatch (n={n}, p={p}): {add_tri_err:e}"
            );

            // The matrix.rs adapter (Full + Replace into a zeroed buffer) must
            // match the faer_ndarray return-style adapter bit-for-functionally.
            let returned = fast_xt_diag_x(&x, &w);
            let returned_err = (&returned - &full)
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            assert!(
                returned_err < 1e-12,
                "return adapter vs stream-into adapter disagree (n={n}, p={p}): {returned_err:e}"
            );
        }
    }

    #[test]
    fn eigh_succeeds_on_same_structure_without_nan() {
        // Control: the same matrix with finite values produces finite eigenvalues.
        let mat = array![[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 1.5]];
        let (evals, _) = mat
            .eigh(Side::Lower)
            .expect("eigh should succeed on a well-conditioned finite matrix");
        assert!(
            evals.iter().all(|&v| v.is_finite()),
            "all eigenvalues should be finite"
        );
    }

    /// gam#933 regression: the Gram-squared RRQR must NOT silently over-rank an
    /// EXACTLY collinear design. Forming `G = XᵀX` squares the spectrum, so the
    /// zero singular value of an exact alias underflows to `≈ ε·σ_max²` in `G` and
    /// the eigen-square-root resurrects it as a SPURIOUS pivot `≈ √ε·σ_max` that
    /// sits above `tol` — col-piv QR on the Gram factor would KEEP it and report
    /// full rank. The precision-floor margin term must catch this: the smallest
    /// kept pivot is near `√ε·leading`, so `verdict_margin` collapses below the
    /// caller's fallback threshold, forcing the full-precision tall path (which
    /// sees the true zero singular value and demotes the column).
    #[test]
    fn gram_rrqr_flags_low_margin_on_exact_collinearity_so_caller_falls_back() {
        // Joint design [1, x | x, x²] with x ∈ [-1, 1]: columns 1 and 2 are an
        // EXACT duplicate (the #933 callback-owned alias), so the true rank is 3.
        let n = 48usize;
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let mut a = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            a[[i, 0]] = 1.0;
            a[[i, 1]] = x[i];
            a[[i, 2]] = x[i];
            a[[i, 3]] = x[i] * x[i];
        }
        let alpha = default_rrqr_rank_alpha();

        // The tall (un-squared) RRQR is the full-precision reference: it must see
        // rank 3 and demote one of the duplicate x columns.
        let tall = rrqr_with_permutation(&a, alpha).expect("tall RRQR should succeed");
        assert_eq!(tall.rank, 3, "tall RRQR must demote the exact alias");

        // The Gram-squared RRQR must report a SMALL verdict_margin here so the
        // caller re-confirms on the tall design instead of trusting a possibly
        // over-ranked Gram verdict. (We do not assert the Gram rank itself —
        // squaring may report 3 or 4 — only that the margin signals the cliff.)
        let unit = Array1::<f64>::ones(n);
        let gram = fast_xt_diag_x_with_parallelism(&a, &unit, faer::get_global_parallelism());
        let gram_rrqr =
            rrqr_from_gram_with_permutation(&gram, n, alpha).expect("Gram RRQR should succeed");
        assert!(
            gram_rrqr.verdict_margin < JOINT_GRAM_RRQR_TRUST_MARGIN_FOR_TEST,
            "exact-collinearity Gram verdict must report low margin to force tall \
             fallback; got margin={:.3e} (rank={})",
            gram_rrqr.verdict_margin,
            gram_rrqr.rank,
        );
    }

    /// Companion to the regression above: a genuinely full-rank, moderately
    /// conditioned design must keep a LARGE Gram verdict margin so the fast Gram
    /// path is retained (the precision-floor term must not trip on real, small-
    /// but-nonzero singular values).
    #[test]
    fn gram_rrqr_keeps_high_margin_on_full_rank_design() {
        let n = 200usize;
        let p = 5usize;
        let mut a = Array2::<f64>::zeros((n, p));
        // Deterministic, well-separated columns (distinct low-order polynomials).
        for i in 0..n {
            let t = (i as f64) / (n as f64 - 1.0);
            a[[i, 0]] = 1.0;
            a[[i, 1]] = t;
            a[[i, 2]] = t * t;
            a[[i, 3]] = t * t * t;
            a[[i, 4]] = (t * 6.0).sin();
        }
        let alpha = default_rrqr_rank_alpha();
        let unit = Array1::<f64>::ones(n);
        let gram = fast_xt_diag_x_with_parallelism(&a, &unit, faer::get_global_parallelism());
        let gram_rrqr =
            rrqr_from_gram_with_permutation(&gram, n, alpha).expect("Gram RRQR should succeed");
        assert_eq!(gram_rrqr.rank, p, "full-rank design must keep all columns");
        assert!(
            gram_rrqr.verdict_margin >= JOINT_GRAM_RRQR_TRUST_MARGIN_FOR_TEST,
            "full-rank design must keep a high margin (fast Gram path); got {:.3e}",
            gram_rrqr.verdict_margin,
        );
    }
}
