use faer::Side;
use gam::alo::compute_alo_diagnostics_from_pirls;
use gam::faer_ndarray::{FaerArrayView, FaerColView, factorize_symmetric_with_fallback, fast_ata};
use gam::pirls::{self, PirlsConfig};
use gam::types::{LinkFunction, LogSmoothingParamsView};
use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Bernoulli, Distribution, Normal};

fn generate_synthetic_binary_data(
    n: usize,
    p: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = normal.sample(&mut rng);
        }
    }
    let mut beta = Array1::zeros(p);
    for j in 0..p {
        beta[j] = normal.sample(&mut rng) / (j as f64 + 1.0).sqrt();
    }
    let eta = x.dot(&beta);
    let probs = eta.mapv(|v| 1.0 / (1.0 + (-v).exp()));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let dist = Bernoulli::new(probs[i]).unwrap();
        y[i] = if dist.sample(&mut rng) { 1.0 } else { 0.0 };
    }
    (x, y, probs)
}

fn fit_unpenalized(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w_prior: &Array1<f64>,
    link: LinkFunction,
) -> pirls::PirlsResult {
    let rho = Array1::<f64>::zeros(0);
    let offset = Array1::<f64>::zeros(x.nrows());
    let rs_original: Vec<Array2<f64>> = Vec::new();
    let cfg = PirlsConfig {
        link_function: link,
        max_iterations: 100,
        convergence_tolerance: 1e-10,
        firth_bias_reduction: matches!(link, LinkFunction::Logit),
    };
    let (res, _) = pirls::fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        x.view(),
        offset.view(),
        y.view(),
        w_prior.view(),
        &rs_original,
        None,
        None,
        x.ncols(),
        &cfg,
        None,
        None,
        None,
        None,
    )
    .expect("unpenalized PIRLS fit");
    res
}

fn beta_in_original_basis(fit: &pirls::PirlsResult) -> Array1<f64> {
    fit.reparam_result.qs.dot(fit.beta_transformed.as_ref())
}

fn loo_compare(
    alo_pred: &Array1<f64>,
    alo_se: &Array1<f64>,
    true_loo_pred: &Array1<f64>,
    true_loo_se: &Array1<f64>,
) -> (f64, f64, f64, f64) {
    let n = alo_pred.len();
    let mut sum_sq_pred = 0.0;
    let mut max_abs_pred: f64 = 0.0;
    let mut sum_sq_se = 0.0;
    let mut max_abs_se: f64 = 0.0;
    for i in 0..n {
        let d_pred = alo_pred[i] - true_loo_pred[i];
        sum_sq_pred += d_pred * d_pred;
        max_abs_pred = max_abs_pred.max(d_pred.abs());
        let d_se = alo_se[i] - true_loo_se[i];
        sum_sq_se += d_se * d_se;
        max_abs_se = max_abs_se.max(d_se.abs());
    }
    (
        (sum_sq_pred / n as f64).sqrt(),
        max_abs_pred,
        (sum_sq_se / n as f64).sqrt(),
        max_abs_se,
    )
}

#[test]
fn alo_se_calculation_correct() {
    let n = 100;
    let p = 5;
    let (x, y, _) = generate_synthetic_binary_data(n, p, 42);
    let mut w = Array1::<f64>::ones(n);
    for i in 0..n / 4 {
        w[i] = 5.0;
    }
    for i in n / 4..(n / 2) {
        w[i] = 0.2;
    }
    let fit = fit_unpenalized(&x, &y, &w, LinkFunction::Logit);
    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Logit).unwrap();

    let x_dense = fit.x_transformed.to_dense();
    let sqrt_w = fit.final_weights.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrt_w_col = sqrt_w.view().insert_axis(Axis(1));
    u *= &sqrt_w_col;
    let k = fit.penalized_hessian_transformed.clone();
    let p_dim = k.nrows();
    let k_view = FaerArrayView::new(&k);
    let factor = factorize_symmetric_with_fallback(k_view.as_ref(), Side::Lower).unwrap();
    let xtwx = fast_ata(&u);

    for irow in 0..10 {
        let ui = u.row(irow).to_owned();
        let rhs = FaerColView::new(&ui);
        let si = factor.solve(rhs.as_ref());
        let si_arr = Array1::from_shape_fn(p_dim, |j| si[(j, 0)]);
        let t_i = xtwx.dot(&si_arr);
        let quad: f64 = si_arr.iter().zip(t_i.iter()).map(|(a, b)| a * b).sum();
        let wi = fit.final_weights[irow].max(1e-300);
        let expected_se = (quad / wi).max(0.0).sqrt();
        assert!(
            (alo.se_sandwich[irow] - expected_se).abs() < 1e-10,
            "SE mismatch at row {irow}: got {}, expected {}",
            alo.se_sandwich[irow],
            expected_se
        );
    }
}

#[test]
fn alo_hat_diag_sane_and_bounded() {
    let n = 200;
    let p = 12;
    let (x, y, _) = generate_synthetic_binary_data(n, p, 42);
    let w = Array1::<f64>::ones(n);
    let fit = fit_unpenalized(&x, &y, &w, LinkFunction::Logit);
    let x_dense = fit.x_transformed.to_dense();
    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Logit).unwrap();
    let leverage = alo.leverage;

    for &a in &leverage {
        assert!(a >= 0.0);
        assert!(a < 1.0);
    }

    let mean = leverage.sum() / n as f64;
    let expected = p as f64 / n as f64;
    assert!((mean - expected).abs() < 0.05);

    let x_leverage: Vec<f64> = (0..n).map(|i| x.row(i).dot(&x.row(i))).collect();
    let x_mean = x_leverage.iter().sum::<f64>() / n as f64;
    let a_mean = mean;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_a = 0.0;
    for i in 0..n {
        let dx = x_leverage[i] - x_mean;
        let da = leverage[i] - a_mean;
        cov += dx * da;
        var_x += dx * dx;
        var_a += da * da;
    }
    let corr = cov / (var_x.sqrt() * var_a.sqrt());
    assert!(corr > 0.3);

    // Zero prior weight should force zero leverage in ALO geometry.
    let mut w_zero = w.clone();
    w_zero[10] = 0.0;
    let fit_zero = fit_unpenalized(&x, &y, &w_zero, LinkFunction::Logit);
    let alo_zero =
        compute_alo_diagnostics_from_pirls(&fit_zero, y.view(), LinkFunction::Logit).unwrap();
    assert!(alo_zero.leverage[10].abs() < 1e-12);
    let _ = x_dense; // keep parity with original helper context
}

#[test]
fn alo_matches_exact_linearized_loo_small_n_binomial() {
    let n = 150;
    let p = 10;
    let (x, y, _) = generate_synthetic_binary_data(n, p, 42);
    let w = Array1::<f64>::ones(n);
    let fit = fit_unpenalized(&x, &y, &w, LinkFunction::Logit);
    let x_dense = fit.x_transformed.to_dense();
    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Logit).unwrap();

    let w_full = fit.final_weights.clone();
    let sqrt_w = w_full.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrt_w_col = sqrt_w.view().insert_axis(Axis(1));
    u *= &sqrt_w_col;

    let mut h = fit.penalized_hessian_transformed.clone();
    for d in 0..h.nrows() {
        h[[d, d]] += 1e-12;
    }
    let p_dim = h.nrows();
    let h_view = FaerArrayView::new(&h);
    let factor = factorize_symmetric_with_fallback(h_view.as_ref(), Side::Lower).unwrap();
    let ut = u.t();
    let xtwx = ut.dot(&u);
    let rhs = ut.to_owned();
    let rhs_view = FaerArrayView::new(&rhs);
    let s_all = factor.solve(rhs_view.as_ref());
    let s_all_nd = Array2::from_shape_fn((p_dim, n), |(i, j)| s_all[(i, j)]);

    let eta_hat = x_dense.dot(fit.beta_transformed.as_ref());
    let z = &fit.solve_working_response;
    let mut loo_pred = Array1::<f64>::zeros(n);
    let mut naive_se = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut aii = 0.0;
        for r in 0..p_dim {
            aii += u[[i, r]] * s_all_nd[(r, i)];
        }
        let denom = (1.0 - aii).max(1e-12);
        loo_pred[i] = (eta_hat[i] - aii * z[i]) / denom;
        let mut quad = 0.0;
        for r in 0..p_dim {
            let mut tmp = 0.0;
            for c in 0..p_dim {
                tmp += xtwx[[r, c]] * s_all_nd[(c, i)];
            }
            quad += s_all_nd[(r, i)] * tmp;
        }
        let wi = w_full[i].max(1e-12);
        naive_se[i] = (quad / wi).max(0.0).sqrt();
    }

    let (rmse_pred, max_abs_pred, rmse_se, max_abs_se) =
        loo_compare(&alo.eta_tilde, &alo.se_sandwich, &loo_pred, &naive_se);
    assert!(rmse_pred <= 1e-9);
    assert!(max_abs_pred <= 1e-8);
    assert!(rmse_se <= 1e-9);
    assert!(max_abs_se <= 1e-8);
}

#[test]
fn alo_matches_true_loo_small_n_binomial_refit() {
    let n = 150;
    let p = 10;
    let (x, y, _) = generate_synthetic_binary_data(n, p, 42);
    let w = Array1::<f64>::ones(n);
    let fit = fit_unpenalized(&x, &y, &w, LinkFunction::Logit);
    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Logit).unwrap();

    let x_dense = fit.x_transformed.to_dense();
    let sqrt_w = fit.final_weights.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrt_w_col = sqrt_w.view().insert_axis(Axis(1));
    u *= &sqrt_w_col;
    let k = fit.penalized_hessian_transformed.clone();
    let xtwx = u.t().dot(&u);
    let k_view = FaerArrayView::new(&k);
    let factor = factorize_symmetric_with_fallback(k_view.as_ref(), Side::Lower).unwrap();
    let mut naive_se = Array1::<f64>::zeros(n);
    for i in 0..n {
        let ui = u.row(i).to_owned();
        let rhs = FaerColView::new(&ui);
        let s = factor.solve(rhs.as_ref());
        let s_arr = Array1::from_shape_fn(p, |j| s[(j, 0)]);
        let t = xtwx.dot(&s_arr);
        let quad: f64 = s_arr.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
        let wi = fit.final_weights[i].max(1e-12);
        naive_se[i] = (quad / wi).max(0.0).sqrt();
    }

    let mut loo_pred = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut x_loo = Array2::zeros((n - 1, p));
        let mut y_loo = Array1::zeros(n - 1);
        let mut w_loo = Array1::zeros(n - 1);
        let mut idx = 0usize;
        for j in 0..n {
            if j == i {
                continue;
            }
            for k in 0..p {
                x_loo[[idx, k]] = x[[j, k]];
            }
            y_loo[idx] = y[j];
            w_loo[idx] = w[j];
            idx += 1;
        }
        let fit_loo = fit_unpenalized(&x_loo, &y_loo, &w_loo, LinkFunction::Logit);
        let beta_loo = beta_in_original_basis(&fit_loo);
        let x_i = x.row(i).to_owned();
        loo_pred[i] = x_i.dot(&beta_loo);
    }

    let (rmse_pred, max_abs_pred, rmse_se, max_abs_se) =
        loo_compare(&alo.eta_tilde, &alo.se_sandwich, &loo_pred, &naive_se);
    assert!(rmse_pred <= 1e-2);
    assert!(max_abs_pred <= 8e-2);
    assert!(rmse_se <= 1e-10);
    assert!(max_abs_se <= 1e-9);
}

#[test]
fn alo_error_is_driven_by_saturated_points() {
    let large = 12.0;
    let mut rows = Vec::new();
    rows.extend(std::iter::repeat_n((-large, 0.0), 40));
    rows.extend(std::iter::repeat_n((large, 1.0), 20));
    rows.push((-large, 1.0));
    rows.push((large, 0.0));

    let n = rows.len();
    let p = 2;
    let mut x = Array2::<f64>::zeros((n, p));
    x.column_mut(0).fill(1.0);
    let mut y = Array1::<f64>::zeros(n);
    for (i, (feature, label)) in rows.into_iter().enumerate() {
        x[[i, 1]] = feature;
        y[i] = label;
    }

    let w = Array1::<f64>::ones(n);
    let fit = fit_unpenalized(&x, &y, &w, LinkFunction::Logit);
    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Logit).unwrap();

    let mut loo_pred = Array1::<f64>::zeros(n);
    let mut loo_se = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut x_loo = Array2::<f64>::zeros((n - 1, p));
        let mut y_loo = Array1::<f64>::zeros(n - 1);
        let mut idx = 0usize;
        for j in 0..n {
            if j == i {
                continue;
            }
            x_loo.row_mut(idx).assign(&x.row(j));
            y_loo[idx] = y[j];
            idx += 1;
        }
        let w_loo = Array1::<f64>::ones(n - 1);
        let fit_loo = fit_unpenalized(&x_loo, &y_loo, &w_loo, LinkFunction::Logit);
        let beta_loo = beta_in_original_basis(&fit_loo);
        let x_i = x.row(i);
        loo_pred[i] = x_i.dot(&beta_loo);

        let mut xtwx = Array2::<f64>::zeros((p, p));
        for r in 0..(n - 1) {
            let wi = fit_loo.final_weights[r];
            if wi == 0.0 {
                continue;
            }
            let xi = x_loo.row(r);
            for a in 0..p {
                for b in 0..p {
                    xtwx[[a, b]] += wi * xi[a] * xi[b];
                }
            }
        }
        for d in 0..p {
            xtwx[[d, d]] += 1e-10;
        }
        let k_view = FaerArrayView::new(&xtwx);
        let llt = factorize_symmetric_with_fallback(k_view.as_ref(), Side::Lower).unwrap();
        let ui = x_i.to_owned();
        let rhs = FaerColView::new(&ui);
        let sol = llt.solve(rhs.as_ref());
        let mut quad = 0.0;
        for r in 0..p {
            quad += x_i[r] * sol[(r, 0)];
        }
        loo_se[i] = quad.sqrt();
    }

    let (rmse_pred, max_abs_pred, _, _) =
        loo_compare(&alo.eta_tilde, &alo.se_sandwich, &loo_pred, &loo_se);
    let beta_full = beta_in_original_basis(&fit);
    let eta_full = x.dot(&beta_full);
    let z_full = &fit.solve_working_response;
    let max_working_jump = z_full
        .iter()
        .zip(eta_full.iter())
        .map(|(&zv, &ev)| (zv - ev).abs())
        .fold(0.0_f64, f64::max);

    assert!(rmse_pred > 1e-2);
    assert!(max_abs_pred > 1e-1);
    assert!(max_working_jump > 25.0);
}
