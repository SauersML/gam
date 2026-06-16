//! Mixed-precision fused kernels for the streaming SAE corpus driver (#973).
//!
//! The on-disk activation shards store rows as `f32` (the
//! mixed-precision-storage contract — see [`super::shard_reader`]): half the
//! bytes of `f64`, so a corpus that would not fit in RAM as `f64` streams
//! comfortably as `f32`. But the SAE inner solve and the REML accumulators
//! demand `f64` numerics — silently summing millions of `f32` products in
//! `f32` loses the low bits and makes the accumulation **order-dependent**,
//! which would break the deterministic any-order accumulation primitive the
//! rest of #973 is built on.
//!
//! The contract these kernels enforce is therefore: **read `f32`, accumulate
//! `f64`**. Every product is promoted to `f64` *before* it enters the running
//! sum, and the running sum is always `f64`. Two callers that visit the same
//! rows in different orders still differ only by `f64` round-off (which the
//! deterministic-accumulation layer compensates), never by `f32` truncation.
//!
//! All functions here are pure (no I/O, no allocation beyond the returned
//! aggregate, no global state) over `&[f32]` / [`ndarray::ArrayView2`], so they
//! are trivially testable and trivially parallelizable by the caller.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// `f64`-accumulated dot product of two `f32` rows.
///
/// Each lane product `a[i] * b[i]` is formed in `f64` (both operands promoted
/// first) and added to an `f64` accumulator. Panics if the slices differ in
/// length — a length mismatch is a structural bug in the caller (mismatched
/// shard `p`), not a recoverable runtime condition.
#[inline]
pub fn dot_f32_f64(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot_f32_f64: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut acc = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += f64::from(x) * f64::from(y);
    }
    acc
}

/// `f64`-accumulated squared L2 norm of an `f32` row.
#[inline]
pub fn norm_sq_f32_f64(a: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    for &x in a.iter() {
        let xd = f64::from(x);
        acc += xd * xd;
    }
    acc
}

/// `f64`-accumulated `y += alpha * x` for an `f32` source row into an `f64`
/// destination. The destination stays `f64` so repeated accumulation over many
/// rows never loses precision to `f32` round-off.
#[inline]
pub fn axpy_f32_into_f64(alpha: f64, x: &[f32], y: &mut [f64]) {
    assert_eq!(
        x.len(),
        y.len(),
        "axpy_f32_into_f64: length mismatch {} vs {}",
        x.len(),
        y.len()
    );
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * f64::from(xi);
    }
}

/// `f64`-accumulated matrix-vector product `A · v` where the rows of `A` are
/// supplied as a borrowed `f32` view and `v` is `f64`.
///
/// Returns one `f64` entry per row. The inner contraction over columns runs in
/// `f64` (each `a[r, c]` promoted before multiplying `v[c]`), so the result is
/// the `f64`-exact `A v` of the `f32`-rounded matrix — never an `f32`-rounded
/// product.
pub fn gemv_f32_rows_f64(a: ArrayView2<'_, f32>, v: ArrayView1<'_, f64>) -> Array1<f64> {
    assert_eq!(
        a.ncols(),
        v.len(),
        "gemv_f32_rows_f64: A has {} cols but v has {}",
        a.ncols(),
        v.len()
    );
    let mut out = Array1::<f64>::zeros(a.nrows());
    for (r, row) in a.outer_iter().enumerate() {
        let mut acc = 0.0_f64;
        for (c, &aij) in row.iter().enumerate() {
            acc += f64::from(aij) * v[c];
        }
        out[r] = acc;
    }
    out
}

/// `f64`-accumulated transpose matrix-vector product `Aᵀ · u`.
///
/// `A` rows are `f32`; `u` has one `f64` entry per row of `A`. The result has
/// one `f64` entry per column of `A`. This is the form the SAE encoder uses to
/// fold a batch of residuals back onto the dictionary atoms. Accumulation is
/// column-keyed in `f64`, so visiting the batch's rows in any order yields the
/// same `f64`-round-off result.
pub fn gemv_t_f32_rows_f64(a: ArrayView2<'_, f32>, u: ArrayView1<'_, f64>) -> Array1<f64> {
    assert_eq!(
        a.nrows(),
        u.len(),
        "gemv_t_f32_rows_f64: A has {} rows but u has {}",
        a.nrows(),
        u.len()
    );
    let mut out = Array1::<f64>::zeros(a.ncols());
    for (r, row) in a.outer_iter().enumerate() {
        let ur = u[r];
        for (c, &aij) in row.iter().enumerate() {
            out[c] += f64::from(aij) * ur;
        }
    }
    out
}

/// `f64`-accumulated Gram matrix `Aᵀ · A` of an `f32`-row batch.
///
/// Returns a symmetric `p × p` `f64` matrix where `p = A.ncols()`. Only the
/// upper triangle is computed in the hot loop and mirrored, halving the work;
/// every product is `f64`-promoted before summation, so the Gram is the
/// `f64`-exact Gram of the `f32`-rounded rows. This is the per-batch
/// contribution the deterministic any-order accumulator folds together across
/// shards.
pub fn gram_f32_rows_f64(a: ArrayView2<'_, f32>) -> Array2<f64> {
    let p = a.ncols();
    let mut g = Array2::<f64>::zeros((p, p));
    for row in a.outer_iter() {
        // Promote the whole row once so the O(p^2) inner loop reads f64.
        let rd: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
        for i in 0..p {
            let ri = rd[i];
            if ri == 0.0 {
                continue;
            }
            for j in i..p {
                g[(i, j)] += ri * rd[j];
            }
        }
    }
    // Mirror the upper triangle into the lower triangle.
    for i in 0..p {
        for j in (i + 1)..p {
            g[(j, i)] = g[(i, j)];
        }
    }
    g
}

/// `f64`-accumulated cross-product `Aᵀ · B` of two `f32`-row batches that share
/// a row index (same number of rows, possibly different column counts).
///
/// Returns a `(A.ncols × B.ncols)` `f64` matrix. Used to form the design ↔
/// residual cross-moments the REML outer step consumes, again with the
/// read-`f32` / accumulate-`f64` contract so the result is order-stable up to
/// `f64` round-off.
pub fn cross_f32_rows_f64(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> Array2<f64> {
    assert_eq!(
        a.nrows(),
        b.nrows(),
        "cross_f32_rows_f64: row mismatch {} vs {}",
        a.nrows(),
        b.nrows()
    );
    let pa = a.ncols();
    let pb = b.ncols();
    let mut c = Array2::<f64>::zeros((pa, pb));
    for (arow, brow) in a.outer_iter().zip(b.outer_iter()) {
        let ad: Vec<f64> = arow.iter().map(|&x| f64::from(x)).collect();
        let bd: Vec<f64> = brow.iter().map(|&x| f64::from(x)).collect();
        for i in 0..pa {
            let ai = ad[i];
            if ai == 0.0 {
                continue;
            }
            for j in 0..pb {
                c[(i, j)] += ai * bd[j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dot_accumulates_in_f64() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        assert_eq!(dot_f32_f64(&a, &b), 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn norm_sq_matches_dot_with_self() {
        let a = [0.5_f32, -1.5, 2.25];
        assert_eq!(norm_sq_f32_f64(&a), dot_f32_f64(&a, &a));
    }

    #[test]
    fn axpy_folds_into_f64_destination() {
        let x = [1.0_f32, -2.0, 4.0];
        let mut y = vec![10.0_f64, 10.0, 10.0];
        axpy_f32_into_f64(0.5, &x, &mut y);
        assert_eq!(y, vec![10.5, 9.0, 12.0]);
    }

    #[test]
    fn gemv_matches_manual() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let v = array![10.0_f64, 100.0];
        let out = gemv_f32_rows_f64(a.view(), v.view());
        assert_eq!(out, array![210.0, 430.0, 650.0]);
    }

    #[test]
    fn gemv_t_is_adjoint_of_gemv() {
        // <A v, u> == <v, Aᵀ u> exactly in f64.
        let a = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v = array![1.0_f64, 0.5, -2.0];
        let u = array![3.0_f64, -1.0];
        let av = gemv_f32_rows_f64(a.view(), v.view());
        let atu = gemv_t_f32_rows_f64(a.view(), u.view());
        let lhs: f64 = av.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = v.iter().zip(atu.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn gram_is_symmetric_and_correct() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let g = gram_f32_rows_f64(a.view());
        // Aᵀ A = [[1+9, 2+12],[2+12, 4+16]] = [[10,14],[14,20]]
        assert_eq!(g, array![[10.0, 14.0], [14.0, 20.0]]);
        assert_eq!(g[(0, 1)], g[(1, 0)]);
    }

    #[test]
    fn gram_order_independent_in_f64() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let a_rev = array![[5.0_f32, 6.0], [3.0, 4.0], [1.0, 2.0]];
        assert_eq!(gram_f32_rows_f64(a.view()), gram_f32_rows_f64(a_rev.view()));
    }

    #[test]
    fn cross_matches_manual() {
        let a = array![[1.0_f32, 0.0], [0.0, 1.0]];
        let b = array![[2.0_f32, 3.0, 4.0], [5.0, 6.0, 7.0]];
        let c = cross_f32_rows_f64(a.view(), b.view());
        assert_eq!(c, array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]);
    }
}
