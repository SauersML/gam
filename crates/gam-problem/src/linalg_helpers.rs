use faer::{MatRef, Par, get_global_parallelism};
use ndarray::{Array2, ArrayBase, ArrayView1, ArrayViewMut1, Data, Ix2};

#[inline]
pub(crate) fn effective_global_parallelism() -> Par {
    if gam_linalg::faer_ndarray::in_nested_parallel_region() {
        Par::Seq
    } else {
        get_global_parallelism()
    }
}

#[inline]
pub(crate) fn dense_matvec_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.ncols(), x.len());
    assert_eq!(matrix.nrows(), out.len());
    for (row, out_value) in matrix.rows().into_iter().zip(out.iter_mut()) {
        *out_value = row.dot(&x);
    }
}

#[inline]
pub(crate) fn dense_matvec_scaled_add_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    scale: f64,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.ncols(), x.len());
    assert_eq!(matrix.nrows(), out.len());
    if scale == 0.0 {
        return;
    }
    for (row, out_value) in matrix.rows().into_iter().zip(out.iter_mut()) {
        *out_value += scale * row.dot(&x);
    }
}

pub(crate) fn dense_transpose_matvec_scaled_add_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    scale: f64,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.nrows(), x.len());
    assert_eq!(matrix.ncols(), out.len());
    if scale == 0.0 {
        return;
    }
    for (row, x_value) in matrix.rows().into_iter().zip(x.iter().copied()) {
        let row_scale = scale * x_value;
        if row_scale == 0.0 {
            continue;
        }
        for (out_value, entry) in out.iter_mut().zip(row.iter().copied()) {
            *out_value += row_scale * entry;
        }
    }
}

#[inline]
pub(crate) fn dense_bilinear(
    matrix: &Array2<f64>,
    v: ArrayView1<'_, f64>,
    u: ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(matrix.ncols(), v.len());
    assert_eq!(matrix.nrows(), u.len());
    let mut total = 0.0;
    for (row, u_value) in matrix.rows().into_iter().zip(u.iter().copied()) {
        total += u_value * row.dot(&v);
    }
    total
}

#[inline]
pub(crate) const fn should_use_faer_matmul(m: usize, n: usize, k: usize) -> bool {
    // Small, centralized dispatch policy:
    // - stay on ndarray for tiny products to avoid setup overhead,
    // - switch to faer GEMM/GEMV for moderate+ sizes.
    const MIN_DIM: usize = 32;
    const MIN_FLOP_SCALE: usize = 64 * 64;
    (m >= MIN_DIM || n >= MIN_DIM || k >= MIN_DIM)
        && m.saturating_mul(n).saturating_mul(k) >= MIN_FLOP_SCALE
}

#[inline]
pub(crate) fn matmul_parallelism(m: usize, n: usize, k: usize) -> Par {
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

/// Compute A^T * B using faer's SIMD-optimized GEMM.
/// For A of shape (n, p) and B of shape (n, q), this computes the (p, q) result.
/// Uses zero-copy views when possible.
#[inline]
pub fn fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Array2<f64> {
    if let Some(out) = crate::gpu::linalg_dispatch::try_fast_atb(a.view(), b.view()) {
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

// `FaerArrayView` is single-sourced in `gam-linalg` (`faer_ndarray`); the carve
// left a byte-identical copy here. Re-use the canonical one (this crate already
// depends on gam-linalg and calls into `gam_linalg::faer_ndarray`).
use gam_linalg::faer_ndarray::FaerArrayView;
