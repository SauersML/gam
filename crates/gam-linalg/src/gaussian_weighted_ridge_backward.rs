//! Closed-form analytic VJP (reverse-mode adjoint) of the Gaussian row-weighted
//! ridge solve [`crate::utils::gaussian_weighted_ridge`].
//!
//! The forward solves, per problem, `β = A⁻¹ b` with `A = XᵀWX + λS`,
//! `b = XᵀWY`, `W = diag(weights)`, and returns `(β, fitted = Xβ)`. This module
//! is the single Rust source of truth for the backward: given the upstream
//! cotangents `β̄` (w.r.t. `coef`) and `f̄` (w.r.t. `fitted`), it returns the
//! gradients w.r.t. `X`, `Y`, `penalty` and `weights`. The math previously lived
//! transcribed in `gamfit/torch/_basis.py::_gwr_vjp`; it now lives here and the
//! torch `autograd.Function` keeps only tape plumbing.
//!
//! Closed form (for `A` SPD symmetric, `β̄` is `M×D`, `f̄` is `N×D`):
//!
//! ```text
//! β̄_tot = β̄ + Xᵀf̄
//! b̄      = A⁻¹ β̄_tot                      (M×D)
//! Ā      = −b̄ βᵀ                           (M×M)
//! grad_X = W·X·(Ā + Āᵀ) + W·Y·b̄ᵀ + f̄·βᵀ   (N×M)
//! grad_Y = W·X·b̄                           (N×D)
//! grad_S = sym(λ·Ā)                        (M×M)
//! grad_w[i] = Σ Ā[m,m']·X[i,m]·X[i,m'] + Σ b̄[m,d]·X[i,m]·Y[i,d]
//! ```

use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, factorize_symmetricwith_fallback};
use faer::Side;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, s};

/// Exact analytic VJP of one Gaussian row-weighted ridge solve.
///
/// Inputs mirror the forward: `x` is `(N, M)`, `y` is `(N, D)`, `penalty` is
/// `(M, M)`, `weights` is `(N,)`, `coef` (the forward `β`) is `(M, D)`. The
/// upstream cotangents are `grad_coef` (`β̄`, `(M, D)`) and `grad_fitted` (`f̄`,
/// `(N, D)`). Returns `(grad_x, grad_y, grad_penalty, grad_weights)` with shapes
/// `((N, M), (N, D), (M, M), (N,))`.
pub fn gaussian_weighted_ridge_backward(
    grad_coef: ArrayView2<'_, f64>,
    grad_fitted: ArrayView2<'_, f64>,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    coef: ArrayView2<'_, f64>,
    ridge_lambda: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>), String> {
    let n = x.nrows();
    let m = x.ncols();
    let d = y.ncols();
    if n == 0 || m == 0 || d == 0 {
        return Err("weighted ridge backward inputs cannot be empty".to_string());
    }
    if y.nrows() != n {
        return Err(format!(
            "X/Y row mismatch: X has {n} rows but Y has {} rows",
            y.nrows()
        ));
    }
    if weights.len() != n {
        return Err(format!(
            "weights length mismatch: expected {n}, got {}",
            weights.len()
        ));
    }
    if penalty.nrows() != m || penalty.ncols() != m {
        return Err(format!(
            "penalty shape mismatch: expected {m}x{m}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if coef.nrows() != m || coef.ncols() != d {
        return Err(format!(
            "coef shape mismatch: expected {m}x{d}, got {}x{}",
            coef.nrows(),
            coef.ncols()
        ));
    }
    if grad_coef.nrows() != m || grad_coef.ncols() != d {
        return Err(format!(
            "grad_coef shape mismatch: expected {m}x{d}, got {}x{}",
            grad_coef.nrows(),
            grad_coef.ncols()
        ));
    }
    if grad_fitted.nrows() != n || grad_fitted.ncols() != d {
        return Err(format!(
            "grad_fitted shape mismatch: expected {n}x{d}, got {}x{}",
            grad_fitted.nrows(),
            grad_fitted.ncols()
        ));
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(format!(
            "ridge_lambda must be finite and non-negative; got {ridge_lambda}"
        ));
    }

    // Row-weighted design/response, `WX` and `WY` (`W = diag(weights)`).
    let mut wx = x.to_owned();
    let mut wy = y.to_owned();
    for i in 0..n {
        let wi = weights[i];
        wx.row_mut(i).iter_mut().for_each(|value| *value *= wi);
        wy.row_mut(i).iter_mut().for_each(|value| *value *= wi);
    }

    // `A = XᵀWX + λS` (the same SPD system the forward factorizes). Adding
    // `λ·penalty` unconditionally matches the Python closed form; for `λ = 0` the
    // added block is exactly zero.
    let mut a = x.t().dot(&wx);
    a += &(penalty.to_owned() * ridge_lambda);

    // `b̄ = A⁻¹ (β̄ + Xᵀf̄)`, solved in place through the symmetric factorization.
    let mut b_bar = grad_coef.to_owned() + x.t().dot(&grad_fitted);
    let factor = factorize_symmetricwith_fallback(FaerArrayView::new(&a).as_ref(), Side::Lower)
        .map_err(|err| format!("weighted ridge backward factorization failed: {err}"))?;
    {
        let mut view = array2_to_matmut(&mut b_bar);
        factor.solve_in_place(view.as_mut());
    }
    if b_bar.iter().any(|value| !value.is_finite()) {
        return Err("weighted ridge backward solve produced non-finite values".to_string());
    }

    // `Ā = −b̄ βᵀ`.
    let a_bar = -(b_bar.dot(&coef.t()));
    let a_bar_t = a_bar.t().to_owned();
    let a_bar_sym = &a_bar + &a_bar_t; // Ā + Āᵀ

    // `grad_X = WX·(Ā + Āᵀ) + WY·b̄ᵀ + f̄·βᵀ`.
    let b_bar_t = b_bar.t();
    let coef_t = coef.t();
    let grad_x = wx.dot(&a_bar_sym) + wy.dot(&b_bar_t) + grad_fitted.dot(&coef_t);

    // `grad_Y = WX·b̄`.
    let grad_y = wx.dot(&b_bar);

    // `grad_S = sym(λ·Ā)`.
    let gp = &a_bar * ridge_lambda;
    let gp_t = gp.t().to_owned();
    let grad_penalty = (&gp + &gp_t) * 0.5;

    // `grad_w[i] = Σ_{m,m'} Ā[m,m'] X[i,m] X[i,m'] + Σ_{m,d} b̄[m,d] X[i,m] Y[i,d]`.
    let xa = x.dot(&a_bar); // (N, M)
    let xb = x.dot(&b_bar); // (N, D)
    let mut grad_w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut term = 0.0;
        for col in 0..m {
            term += xa[[i, col]] * x[[i, col]];
        }
        for col in 0..d {
            term += xb[[i, col]] * y[[i, col]];
        }
        grad_w[i] = term;
    }

    Ok((grad_x, grad_y, grad_penalty, grad_w))
}

/// Batched analytic VJP of [`crate::utils::gaussian_weighted_ridge_batch`].
///
/// One problem per leading-axis slice of the padded `(K, N_max, ·)` tensors,
/// honoring optional per-problem active `row_counts`. Padded rows (index
/// `≥ row_counts[k]`) are masked exactly as the forward ignores them: their
/// weights and upstream fitted-cotangent are zeroed before the per-problem VJP,
/// and their `grad_X`/`grad_Y`/`grad_weights` rows are forced to zero after
/// (the forward output is independent of every padded row). `grad_penalty` is
/// the sum across problems (accumulated in problem order to match the reference
/// torch loop bit-for-bit on well-conditioned inputs).
///
/// Returns `(grad_x, grad_y, grad_penalty, grad_weights)` with shapes
/// `((K, N_max, M), (K, N_max, D), (M, M), (K, N_max))`.
pub fn gaussian_weighted_ridge_batch_backward(
    grad_coef: ArrayView3<'_, f64>,
    grad_fitted: ArrayView3<'_, f64>,
    x: ArrayView3<'_, f64>,
    y: ArrayView3<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView2<'_, f64>,
    coef: ArrayView3<'_, f64>,
    ridge_lambda: f64,
    row_counts: Option<ArrayView1<'_, usize>>,
) -> Result<(Array3<f64>, Array3<f64>, Array2<f64>, Array2<f64>), String> {
    let (batch, n_max, m) = x.dim();
    let (_, _, d) = y.dim();
    if batch == 0 || n_max == 0 || m == 0 || d == 0 {
        return Err("batched weighted ridge backward inputs cannot be empty".to_string());
    }
    if y.dim() != (batch, n_max, d)
        || grad_fitted.dim() != (batch, n_max, d)
        || weights.dim() != (batch, n_max)
        || grad_coef.dim() != (batch, m, d)
        || coef.dim() != (batch, m, d)
    {
        return Err("batched weighted ridge backward shape mismatch".to_string());
    }
    let active_rows: Vec<usize> = match row_counts {
        Some(counts) => {
            if counts.len() != batch {
                return Err(format!(
                    "row_counts length mismatch: expected {batch}, got {}",
                    counts.len()
                ));
            }
            for (b, &n_rows) in counts.iter().enumerate() {
                if n_rows > n_max {
                    return Err(format!(
                        "row_counts[{b}]={n_rows} exceeds padded row count {n_max}"
                    ));
                }
            }
            counts.to_vec()
        }
        None => vec![n_max; batch],
    };

    let mut grad_x = Array3::<f64>::zeros((batch, n_max, m));
    let mut grad_y = Array3::<f64>::zeros((batch, n_max, d));
    let mut grad_penalty = Array2::<f64>::zeros((m, m));
    let mut grad_w = Array2::<f64>::zeros((batch, n_max));

    for b in 0..batch {
        let active = active_rows[b];
        if active == 0 {
            continue;
        }

        // Mask padded rows by zeroing their weights (they drop out of every VJP
        // term, matching the forward, which sees only the active prefix) and
        // their upstream fitted-cotangent (padded fitted rows are not real
        // outputs, so their cotangent must not leak into grad_X / grad_coef).
        let mut wk = weights.slice(s![b, ..]).to_owned();
        let mut gfit_k = grad_fitted.slice(s![b, .., ..]).to_owned();
        if active < n_max {
            for i in active..n_max {
                wk[i] = 0.0;
                gfit_k.row_mut(i).iter_mut().for_each(|value| *value = 0.0);
            }
        }

        let (gxk, gyk, gpk, gwk) = gaussian_weighted_ridge_backward(
            grad_coef.slice(s![b, .., ..]),
            gfit_k.view(),
            x.slice(s![b, .., ..]),
            y.slice(s![b, .., ..]),
            penalty,
            wk.view(),
            coef.slice(s![b, .., ..]),
            ridge_lambda,
        )?;

        // Scatter, forcing padded rows to exactly zero. grad_X/grad_Y rows
        // already vanish (zero weight + zeroed cotangent), but grad_w[i] is
        // built from the un-zeroed X/Y rows, so it must be zeroed explicitly.
        for i in 0..active {
            grad_x.slice_mut(s![b, i, ..]).assign(&gxk.row(i));
            grad_y.slice_mut(s![b, i, ..]).assign(&gyk.row(i));
            grad_w[[b, i]] = gwk[i];
        }
        grad_penalty += &gpk;
    }

    Ok((grad_x, grad_y, grad_penalty, grad_w))
}

#[cfg(test)]
mod gwr_backward_tests {
    use super::*;
    use crate::utils::gaussian_weighted_ridge;
    use ndarray::{Array1, Array2, Array3};

    // Deterministic small fixture; values chosen so `A = XᵀWX + λS` is SPD.
    fn fixture() -> (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Array1<f64>,
        f64,
    ) {
        let x = ndarray::arr2(&[
            [0.5, -1.2, 0.3],
            [1.1, 0.4, -0.7],
            [-0.6, 0.9, 1.4],
            [0.2, -0.3, 0.8],
            [1.5, 0.1, -0.2],
            [-0.9, 1.1, 0.6],
        ]);
        let y = ndarray::arr2(&[
            [1.0, -0.5],
            [0.3, 0.8],
            [-1.1, 0.2],
            [0.6, 0.4],
            [0.9, -0.7],
            [-0.2, 1.3],
        ]);
        // SPD penalty `PPᵀ + m·I`.
        let p = ndarray::arr2(&[
            [0.7, -0.2, 0.1],
            [0.4, 0.9, -0.3],
            [-0.5, 0.2, 0.8],
        ]);
        let penalty = p.dot(&p.t()) + Array2::<f64>::eye(3) * 3.0;
        let weights = ndarray::arr1(&[0.8, 1.2, 0.6, 1.5, 0.9, 1.1]);
        (x, y, penalty, weights, 0.7)
    }

    // Independent element-wise reference of the closed form (guards against
    // transposition / axis bugs in the vectorized kernel). Shares only the SPD
    // solve for `b̄`.
    fn reference(
        grad_coef: &Array2<f64>,
        grad_fitted: &Array2<f64>,
        x: &Array2<f64>,
        y: &Array2<f64>,
        penalty: &Array2<f64>,
        weights: &Array1<f64>,
        coef: &Array2<f64>,
        lam: f64,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = x.nrows();
        let m = x.ncols();
        let d = y.ncols();
        // A elementwise.
        let mut a = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut s = lam * penalty[[i, j]];
                for row in 0..n {
                    s += weights[row] * x[[row, i]] * x[[row, j]];
                }
                a[[i, j]] = s;
            }
        }
        // beta_bar_tot elementwise.
        let mut bbt = Array2::<f64>::zeros((m, d));
        for i in 0..m {
            for dd in 0..d {
                let mut s = grad_coef[[i, dd]];
                for row in 0..n {
                    s += x[[row, i]] * grad_fitted[[row, dd]];
                }
                bbt[[i, dd]] = s;
            }
        }
        // b̄ = A⁻¹ bbt via the crate solve (shared numeric).
        let factor =
            factorize_symmetricwith_fallback(FaerArrayView::new(&a).as_ref(), Side::Lower).unwrap();
        let mut b_bar = bbt.clone();
        {
            let mut view = array2_to_matmut(&mut b_bar);
            factor.solve_in_place(view.as_mut());
        }
        // Ā[i,j] = -Σ_dd b̄[i,dd] coef[j,dd].
        let mut a_bar = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for dd in 0..d {
                    s += b_bar[[i, dd]] * coef[[j, dd]];
                }
                a_bar[[i, j]] = -s;
            }
        }
        // grad_x.
        let mut grad_x = Array2::<f64>::zeros((n, m));
        for row in 0..n {
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..m {
                    s += weights[row] * x[[row, j]] * (a_bar[[j, i]] + a_bar[[i, j]]);
                }
                for dd in 0..d {
                    s += weights[row] * y[[row, dd]] * b_bar[[i, dd]];
                    s += grad_fitted[[row, dd]] * coef[[i, dd]];
                }
                grad_x[[row, i]] = s;
            }
        }
        // grad_y.
        let mut grad_y = Array2::<f64>::zeros((n, d));
        for row in 0..n {
            for dd in 0..d {
                let mut s = 0.0;
                for i in 0..m {
                    s += weights[row] * x[[row, i]] * b_bar[[i, dd]];
                }
                grad_y[[row, dd]] = s;
            }
        }
        // grad_penalty.
        let mut grad_penalty = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                grad_penalty[[i, j]] = 0.5 * (lam * a_bar[[i, j]] + lam * a_bar[[j, i]]);
            }
        }
        // grad_w.
        let mut grad_w = Array1::<f64>::zeros(n);
        for row in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                for j in 0..m {
                    s += a_bar[[i, j]] * x[[row, i]] * x[[row, j]];
                }
                for dd in 0..d {
                    s += b_bar[[i, dd]] * x[[row, i]] * y[[row, dd]];
                }
            }
            grad_w[row] = s;
        }
        (grad_x, grad_y, grad_penalty, grad_w)
    }

    fn max_abs2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |acc, (x, y)| acc.max((x - y).abs()))
    }

    fn max_abs1(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |acc, (x, y)| acc.max((x - y).abs()))
    }

    // Transcription pin: the vectorized kernel reproduces the element-wise
    // closed form (the exact math the Python `_gwr_vjp` transcribed) to 1e-12.
    #[test]
    fn kernel_matches_elementwise_reference() {
        let (x, y, penalty, weights, lam) = fixture();
        let (coef, _fitted) =
            gaussian_weighted_ridge(x.view(), y.view(), penalty.view(), weights.view(), lam)
                .unwrap();
        // Arbitrary upstream cotangents.
        let grad_coef = ndarray::arr2(&[[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]]);
        let grad_fitted = ndarray::arr2(&[
            [0.2, 0.1],
            [-0.5, 0.6],
            [0.8, -0.3],
            [0.4, 0.7],
            [-0.9, 0.2],
            [0.1, -0.6],
        ]);

        let (gx, gy, gp, gw) = gaussian_weighted_ridge_backward(
            grad_coef.view(),
            grad_fitted.view(),
            x.view(),
            y.view(),
            penalty.view(),
            weights.view(),
            coef.view(),
            lam,
        )
        .unwrap();
        let (rx, ry, rp, rw) =
            reference(&grad_coef, &grad_fitted, &x, &y, &penalty, &weights, &coef, lam);

        assert!(max_abs2(&gx, &rx) < 1e-12, "grad_x: {}", max_abs2(&gx, &rx));
        assert!(max_abs2(&gy, &ry) < 1e-12, "grad_y: {}", max_abs2(&gy, &ry));
        assert!(max_abs2(&gp, &rp) < 1e-12, "grad_penalty: {}", max_abs2(&gp, &rp));
        assert!(max_abs1(&gw, &rw) < 1e-12, "grad_w: {}", max_abs1(&gw, &rw));
    }

    // Objective correctness: the analytic VJP equals the central finite
    // difference of the forward `(coef, fitted)` contracted with the cotangents.
    #[test]
    fn kernel_matches_finite_differences() {
        let (x, y, penalty, weights, lam) = fixture();
        let (coef, _fitted) =
            gaussian_weighted_ridge(x.view(), y.view(), penalty.view(), weights.view(), lam)
                .unwrap();
        let grad_coef = ndarray::arr2(&[[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]]);
        let grad_fitted = ndarray::arr2(&[
            [0.2, 0.1],
            [-0.5, 0.6],
            [0.8, -0.3],
            [0.4, 0.7],
            [-0.9, 0.2],
            [0.1, -0.6],
        ]);

        let (gx, gy, gp, gw) = gaussian_weighted_ridge_backward(
            grad_coef.view(),
            grad_fitted.view(),
            x.view(),
            y.view(),
            penalty.view(),
            weights.view(),
            coef.view(),
            lam,
        )
        .unwrap();

        // Scalar loss L = <grad_coef, coef> + <grad_fitted, fitted>; the VJP is
        // dL/d(input). Central differences of the forward validate it.
        let loss = |xx: &Array2<f64>,
                    yy: &Array2<f64>,
                    pp: &Array2<f64>,
                    ww: &Array1<f64>|
         -> f64 {
            let (c, f) =
                gaussian_weighted_ridge(xx.view(), yy.view(), pp.view(), ww.view(), lam).unwrap();
            let mut s = 0.0;
            for (a, b) in c.iter().zip(grad_coef.iter()) {
                s += a * b;
            }
            for (a, b) in f.iter().zip(grad_fitted.iter()) {
                s += a * b;
            }
            s
        };

        let h = 1e-6;
        let fd2 = |base: &Array2<f64>,
                   idx: (usize, usize),
                   eval: &dyn Fn(&Array2<f64>) -> f64|
         -> f64 {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[idx] += h;
            minus[idx] -= h;
            (eval(&plus) - eval(&minus)) / (2.0 * h)
        };

        // grad_x
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let g = fd2(&x, (i, j), &|xx| loss(xx, &y, &penalty, &weights));
                assert!((g - gx[[i, j]]).abs() < 1e-6, "grad_x[{i},{j}]");
            }
        }
        // grad_y
        for i in 0..y.nrows() {
            for j in 0..y.ncols() {
                let g = fd2(&y, (i, j), &|yy| loss(&x, yy, &penalty, &weights));
                assert!((g - gy[[i, j]]).abs() < 1e-6, "grad_y[{i},{j}]");
            }
        }
        // grad_penalty: symmetric part only (the forward uses `penalty` in the
        // symmetric quadratic `λS`, so only sym(grad) is identifiable). Perturb
        // symmetrically to match the analytic symmetrization.
        for i in 0..penalty.nrows() {
            for j in i..penalty.ncols() {
                let mut plus = penalty.clone();
                let mut minus = penalty.clone();
                plus[[i, j]] += h;
                minus[[i, j]] -= h;
                if i != j {
                    plus[[j, i]] += h;
                    minus[[j, i]] -= h;
                }
                let g = (loss(&x, &y, &plus, &weights) - loss(&x, &y, &minus, &weights)) / (2.0 * h);
                // The symmetric-perturbation FD equals gp[i,j]+gp[j,i] for i!=j
                // and gp[i,i] on the diagonal.
                let analytic = if i == j {
                    gp[[i, j]]
                } else {
                    gp[[i, j]] + gp[[j, i]]
                };
                assert!((g - analytic).abs() < 1e-6, "grad_penalty[{i},{j}]");
            }
        }
        // grad_w
        for i in 0..weights.len() {
            let mut plus = weights.clone();
            let mut minus = weights.clone();
            plus[i] += h;
            minus[i] -= h;
            let g = (loss(&x, &y, &penalty, &plus) - loss(&x, &y, &penalty, &minus)) / (2.0 * h);
            assert!((g - gw[i]).abs() < 1e-6, "grad_w[{i}]");
        }
    }

    // Batch-of-1 equals the single-problem path; and padded rows produce exactly
    // zero gradient (mirrors `gaussian_weighted_ridge_batch`'s active-prefix
    // forward).
    #[test]
    fn batch_matches_single_and_zeros_padding() {
        let (x, y, penalty, weights, lam) = fixture();
        let (coef, _f) =
            gaussian_weighted_ridge(x.view(), y.view(), penalty.view(), weights.view(), lam)
                .unwrap();
        let grad_coef = ndarray::arr2(&[[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9]]);
        let grad_fitted = ndarray::arr2(&[
            [0.2, 0.1],
            [-0.5, 0.6],
            [0.8, -0.3],
            [0.4, 0.7],
            [-0.9, 0.2],
            [0.1, -0.6],
        ]);
        let (gx, gy, gp, gw) = gaussian_weighted_ridge_backward(
            grad_coef.view(),
            grad_fitted.view(),
            x.view(),
            y.view(),
            penalty.view(),
            weights.view(),
            coef.view(),
            lam,
        )
        .unwrap();

        // Batch of 1 (no padding).
        let x3 = x.clone().insert_axis(ndarray::Axis(0));
        let y3 = y.clone().insert_axis(ndarray::Axis(0));
        let w2 = weights.clone().insert_axis(ndarray::Axis(0));
        let coef3 = coef.clone().insert_axis(ndarray::Axis(0));
        let gc3 = grad_coef.clone().insert_axis(ndarray::Axis(0));
        let gf3 = grad_fitted.clone().insert_axis(ndarray::Axis(0));
        let (bx, by, bp, bw) = gaussian_weighted_ridge_batch_backward(
            gc3.view(),
            gf3.view(),
            x3.view(),
            y3.view(),
            penalty.view(),
            w2.view(),
            coef3.view(),
            lam,
            None,
        )
        .unwrap();
        assert!(max_abs2(&bx.index_axis(ndarray::Axis(0), 0).to_owned(), &gx) < 1e-12);
        assert!(max_abs2(&by.index_axis(ndarray::Axis(0), 0).to_owned(), &gy) < 1e-12);
        assert!(max_abs2(&bp, &gp) < 1e-12);
        assert!(max_abs1(&bw.index_axis(ndarray::Axis(0), 0).to_owned(), &gw) < 1e-12);

        // Padded batch: pad the single problem to N_max = 8 with garbage rows and
        // row_counts = [6]; the active-prefix result must equal the single path
        // and padded rows must be exactly zero.
        let n_max = 8usize;
        let mut xp = Array3::<f64>::from_elem((1, n_max, x.ncols()), 9.9);
        let mut yp = Array3::<f64>::from_elem((1, n_max, y.ncols()), -4.4);
        let mut wp = Array2::<f64>::from_elem((1, n_max), 7.7);
        let mut gfp = Array3::<f64>::from_elem((1, n_max, y.ncols()), 3.3);
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                xp[[0, i, j]] = x[[i, j]];
            }
            for j in 0..y.ncols() {
                yp[[0, i, j]] = y[[i, j]];
                gfp[[0, i, j]] = grad_fitted[[i, j]];
            }
            wp[[0, i]] = weights[i];
        }
        let counts = Array1::<usize>::from(vec![x.nrows()]);
        let (px, py, pp, pw) = gaussian_weighted_ridge_batch_backward(
            gc3.view(),
            gfp.view(),
            xp.view(),
            yp.view(),
            penalty.view(),
            wp.view(),
            coef3.view(),
            lam,
            Some(counts.view()),
        )
        .unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((px[[0, i, j]] - gx[[i, j]]).abs() < 1e-12);
            }
            for j in 0..y.ncols() {
                assert!((py[[0, i, j]] - gy[[i, j]]).abs() < 1e-12);
            }
            assert!((pw[[0, i]] - gw[i]).abs() < 1e-12);
        }
        for i in x.nrows()..n_max {
            for j in 0..x.ncols() {
                assert_eq!(px[[0, i, j]], 0.0);
            }
            for j in 0..y.ncols() {
                assert_eq!(py[[0, i, j]], 0.0);
            }
            assert_eq!(pw[[0, i]], 0.0);
        }
        assert!(max_abs2(&pp, &gp) < 1e-12);
    }
}
