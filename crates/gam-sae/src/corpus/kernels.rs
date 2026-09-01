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
