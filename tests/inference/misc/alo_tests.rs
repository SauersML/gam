use faer::Side;
use gam::alo::{AloInput, compute_alo_diagnostics_from_pirls, compute_alo_from_input};
use gam::construction::CanonicalPenalty;
use gam::faer_ndarray::{FaerArrayView, FaerColView, factorize_symmetricwith_fallback, fast_ata};
use gam::matrix::{PsdWeightsView, SignedWeightsView};
use gam::pirls::{self, PenaltyConfig, PirlsConfig, PirlsProblem};
use gam::types::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkFunction, LogSmoothingParamsView,
    ResponseFamily, StandardLink,
};
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
    let canonical: Vec<gam::construction::CanonicalPenalty> = Vec::new();
    let standard_link = StandardLink::try_from(link)
        .expect("ALO PIRLS tests only pass stateless standard links into these helpers");
    let cfg = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        link_kind: InverseLink::Standard(standard_link),
        max_iterations: 100,
        convergence_tolerance: 1e-10,
        firth_bias_reduction: matches!(link, LinkFunction::Logit),
        initial_lm_lambda: None,
        arrow_schur: None,
    };
    let (res, _) = pirls::fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view())
            .expect("test fixture smoothing parameters satisfy the closed domain"),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: w_prior.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p: x.ncols(),
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &cfg,
        None,
    )
    .expect("unpenalized PIRLS fit");
    res
}

fn fit_identity_penalized(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w_prior: &Array1<f64>,
    link: LinkFunction,
    lambda: f64,
) -> pirls::PirlsResult {
    let rho = Array1::from_vec(vec![lambda.ln()]);
    let offset = Array1::<f64>::zeros(x.nrows());
    let standard_link = StandardLink::try_from(link)
        .expect("ALO PIRLS tests only pass stateless standard links into these helpers");
    let root = Array2::<f64>::eye(x.ncols());
    let local = root.t().dot(&root);
    let canonical = vec![CanonicalPenalty {
        root,
        col_range: 0..x.ncols(),
        total_dim: x.ncols(),
        nullity: 0,
        local,
        prior_mean: Array1::zeros(x.ncols()),
        positive_eigenvalues: vec![1.0; x.ncols()],
        op: None,
    }];
    let cfg = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        link_kind: InverseLink::Standard(standard_link),
        max_iterations: 100,
        convergence_tolerance: 1e-10,
        firth_bias_reduction: false,
        initial_lm_lambda: None,
        arrow_schur: None,
    };
    let (res, _) = pirls::fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view())
            .expect("test fixture smoothing parameters satisfy the closed domain"),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: w_prior.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p: x.ncols(),
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &cfg,
        None,
    )
    .expect("identity-penalized PIRLS fit");
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
fn alo_uses_exact_dense_stabilized_hessian_export_from_penalized_pirls() {
    let n = 90;
    let p = 6;
    let (x, y, _) = generate_synthetic_binary_data(n, p, 2026);
    let w = Array1::<f64>::ones(n);
    let fit = fit_identity_penalized(&x, &y, &w, LinkFunction::Probit, 0.35);

    let exported = fit
        .dense_stabilizedhessian_transformed("ALO test exact Hessian export")
        .expect("exact dense Hessian export");
    assert_eq!(exported.nrows(), p);
    assert_eq!(exported.ncols(), p);
    assert!(exported.iter().all(|v| v.is_finite()));

    let x_transformed = fit.x_transformed.to_dense();
    let sqrtw = fit.finalweights.mapv(f64::sqrt);
    let mut weighted_x = x_transformed.clone();
    weighted_x *= &sqrtw.view().insert_axis(Axis(1));
    let mut expected = weighted_x.t().dot(&weighted_x);
    expected += &fit.reparam_result.s_transformed;
    let ridge = fit.ridge_passport.laplacehessianridge().max(0.0);
    for d in 0..p {
        expected[[d, d]] += ridge;
    }

    let max_abs_diff = exported
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_diff <= 1e-8,
        "exported ALO Hessian must be the exact penalized PIRLS Hessian; max_abs_diff={max_abs_diff:.3e}"
    );

    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Probit)
        .expect("ALO should accept exact dense Hessian exported from penalized PIRLS");
    assert_eq!(alo.leverage.len(), n);
    assert!(alo.leverage.iter().all(|v| v.is_finite()));
}

/// #935: the one `FitSensitivity` operator's case-deletion channel, exposed
/// from a converged PIRLS fit via `compute_case_deletion_from_pirls`, must
/// agree with the ALO dialect on the quantity they BOTH compute — the
/// per-observation leverage `h_ii = w_i x_iᵀ H⁻¹ x_i`. Both read the same
/// converged penalized Hessian and working weights, so a disagreement would
/// be exactly the "two sites, two inverses" bug class #935 dismantles. We
/// also confirm the influence diagnostic (dfbeta / Cook's distance) is
/// finite and correctly shaped — capability the operator had no production
/// entry point for until now.
#[test]
fn case_deletion_from_pirls_leverage_matches_alo_dialect() {
    let n = 90;
    let p = 6;
    let (x, y, _) = generate_synthetic_binary_data(n, p, 2026);
    let w = Array1::<f64>::ones(n);
    let fit = fit_identity_penalized(&x, &y, &w, LinkFunction::Probit, 0.35);

    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Probit)
        .expect("ALO diagnostics");
    let influence =
        gam::alo::compute_case_deletion_from_pirls(&fit, y.view(), LinkFunction::Probit)
            .expect("case-deletion diagnostics must not error on a converged fit")
            .expect("no leverage-one row in this well-conditioned fit");

    assert_eq!(influence.leverage.len(), n);
    assert_eq!(influence.dfbeta.nrows(), n);
    assert_eq!(influence.dfbeta.ncols(), p);
    assert_eq!(influence.cooks_distance.len(), n);
    assert!(influence.dfbeta.iter().all(|v| v.is_finite()));
    assert!(
        influence
            .cooks_distance
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );

    // The leverage channel is the same hat value ALO computes; the two
    // dialects share one factored inverse, so they must agree to machine
    // precision (not merely "close").
    let max_lev_diff = influence
        .leverage
        .iter()
        .zip(alo.leverage.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_lev_diff <= 1e-9,
        "case-deletion leverage must equal the ALO dialect's hat value on the \
         same fit (one shared H⁻¹); max_abs_diff={max_lev_diff:.3e}"
    );
}

#[test]
fn alo_solve_setup_rejects_non_square_dense_hessian_instead_of_workaround() {
    let design = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
    let bad_hessian = Array2::<f64>::ones((2, 1));
    let hessian_weights = Array1::from_vec(vec![0.25, 0.25, 0.25]);
    let score_weights = hessian_weights.clone();
    let working_response = Array1::from_vec(vec![0.0, 0.5, 1.0]);
    let eta = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let offset = Array1::<f64>::zeros(3);
    let input = AloInput {
        design: &design,
        penalized_hessian: &bad_hessian,
        hessian_weights: SignedWeightsView::from_array(&hessian_weights),
        score_weights: PsdWeightsView::try_from_array(&score_weights).expect("psd weights"),
        working_response: &working_response,
        eta: &eta,
        offset: &offset,
        link: LinkFunction::Logit,
        phi: 1.0,
        penalty_root: None,
        ridge: 0.0,
        score_curvature: None,
    };

    let err = compute_alo_from_input(&input).expect_err("bad Hessian shape must fail setup");
    let msg = err.to_string();
    assert!(
        msg.contains("dense exact penalized Hessian with shape 2x2"),
        "unexpected ALO setup error: {msg}"
    );
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
    let sqrtw = fit.finalweights.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrtw_col = sqrtw.view().insert_axis(Axis(1));
    u *= &sqrtw_col;
    let k = fit.penalized_hessian_transformed.to_dense();
    let p_dim = k.nrows();
    let kview = FaerArrayView::new(&k);
    let factor = factorize_symmetricwith_fallback(kview.as_ref(), Side::Lower).unwrap();
    let xtwx = fast_ata(&u);

    for irow in 0..10 {
        let ui = u.row(irow).to_owned();
        let rhs = FaerColView::new(&ui);
        let si = factor.solve(rhs.as_ref());
        let si_arr = Array1::from_shape_fn(p_dim, |j| si[(j, 0)]);
        let t_i = xtwx.dot(&si_arr);
        let quad: f64 = si_arr.iter().zip(t_i.iter()).map(|(a, b)| a * b).sum();
        let wi = fit.finalweights[irow].max(1e-300);
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
    let mut wzero = w.clone();
    wzero[10] = 0.0;
    let fitzero = fit_unpenalized(&x, &y, &wzero, LinkFunction::Logit);
    let alozero =
        compute_alo_diagnostics_from_pirls(&fitzero, y.view(), LinkFunction::Logit).unwrap();
    assert!(alozero.leverage[10].abs() < 1e-12);
}

#[test]
fn alo_matches_exact_frozen_curvature_loo_small_n_binomial() {
    // `eta_tilde` for a canonical-link GLM is the EXACT frozen-curvature
    // leave-`i`-out predictor: the scalar fixed point
    //   η = η̂_i + h_i · ℓ_i'(η),   h_i = x_iᵀH⁻¹x_i (unweighted influence),
    //   ℓ_i'(η) = c_i (μ(η) − y_i),  c_i = W_H[i] / μ'(η̂_i),
    // iterated to convergence (see `alo_eta_exact_frozen_curvature`). This is a
    // genuine second-order REFINEMENT of the classical single-Newton-step
    // ("linearized") ALO, not equal to it — the linearized predictor is only the
    // FIRST Newton iterate of this same fixed point. We therefore reconstruct the
    // production fixed point independently here (a cross-implementation oracle of
    // the SAME equation) and require `eta_tilde` to match it to solver tolerance,
    // and we additionally pin that the linearized one-step is the first iterate
    // (so the refinement is doing real work beyond it). The sandwich SE is
    // unaffected by the refinement and is still checked to the exact value.
    let n = 150;
    let p = 10;
    let (x, y, _) = generate_synthetic_binary_data(n, p, 42);
    let w = Array1::<f64>::ones(n);
    let fit = fit_unpenalized(&x, &y, &w, LinkFunction::Logit);
    let x_dense = fit.x_transformed.to_dense();
    let alo = compute_alo_diagnostics_from_pirls(&fit, y.view(), LinkFunction::Logit).unwrap();

    let w_full = fit.finalweights.clone();
    let sqrtw = w_full.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrtw_col = sqrtw.view().insert_axis(Axis(1));
    u *= &sqrtw_col;

    let mut h = fit.penalized_hessian_transformed.to_dense();
    for d in 0..h.nrows() {
        h[[d, d]] += 1e-12;
    }
    let p_dim = h.nrows();
    let hview = FaerArrayView::new(&h);
    let factor = factorize_symmetricwith_fallback(hview.as_ref(), Side::Lower).unwrap();
    let ut = u.t();
    let xtwx = ut.dot(&u);
    let rhs = ut.to_owned();
    let rhsview = FaerArrayView::new(&rhs);
    let s_all = factor.solve(rhsview.as_ref());
    let s_all_nd = Array2::from_shape_fn((p_dim, n), |(i, j)| s_all[(i, j)]);

    let eta_hat = x_dense.dot(fit.beta_transformed.as_ref());
    let z = &fit.solveworking_response;
    let dmu_hat = &fit.solve_dmu_deta;
    let mut loo_pred = Array1::<f64>::zeros(n);
    let mut one_step = Array1::<f64>::zeros(n);
    let mut naive_se = Array1::<f64>::zeros(n);
    for i in 0..n {
        // Weighted leverage a_ii = w_i x_iᵀH⁻¹x_i (drives the one-step denom);
        // the unweighted influence h_i = x_iᵀH⁻¹x_i drives the frozen-curvature
        // fixed point exactly as the production path does.
        let mut aii_weighted = 0.0;
        for r in 0..p_dim {
            aii_weighted += u[[i, r]] * s_all_nd[(r, i)];
        }
        let wi = w_full[i].max(1e-12);
        let h_i = aii_weighted / wi;

        // Single-Newton-step ("linearized") ALO: the first iterate.
        let denom = (1.0 - aii_weighted).max(1e-12);
        one_step[i] = (eta_hat[i] - aii_weighted * z[i]) / denom;

        // Exact frozen-curvature fixed point, reconstructed independently.
        let c_i = wi / dmu_hat[i].abs().max(1e-12);
        let mut eta = one_step[i];
        for _ in 0..64 {
            let mu = 1.0 / (1.0 + (-eta).exp());
            let dmu = mu * (1.0 - mu);
            let ell_prime = c_i * (mu - y[i]);
            let residual = eta - eta_hat[i] - h_i * ell_prime;
            if residual.abs() <= 1e-12 {
                break;
            }
            let jac = 1.0 - h_i * c_i * dmu;
            eta -= residual / jac;
        }
        loo_pred[i] = eta;

        let mut quad = 0.0;
        for r in 0..p_dim {
            let mut tmp = 0.0;
            for c in 0..p_dim {
                tmp += xtwx[[r, c]] * s_all_nd[(c, i)];
            }
            quad += s_all_nd[(r, i)] * tmp;
        }
        naive_se[i] = (quad / wi).max(0.0).sqrt();
    }

    // eta_tilde must equal the independently-reconstructed exact frozen-curvature
    // LOO to solver tolerance.
    let (rmse_pred, max_abs_pred, rmse_se, max_abs_se) =
        loo_compare(&alo.eta_tilde, &alo.se_sandwich, &loo_pred, &naive_se);
    assert!(
        rmse_pred <= 1e-9,
        "eta_tilde must match the exact frozen-curvature LOO: rmse_pred={rmse_pred:.6e}"
    );
    assert!(max_abs_pred <= 1e-8, "max_abs_pred={max_abs_pred:.6e}");
    assert!(rmse_se <= 1e-9);
    assert!(max_abs_se <= 1e-8);

    // The refinement must genuinely differ from the linearized first iterate
    // (otherwise it is not the second-order predictor it claims to be).
    let (rmse_vs_one_step, _, _, _) =
        loo_compare(&alo.eta_tilde, &alo.se_sandwich, &one_step, &naive_se);
    assert!(
        rmse_vs_one_step > 1e-7,
        "exact frozen-curvature eta_tilde must refine beyond the linearized one-step, \
         but they coincide (rmse={rmse_vs_one_step:.6e})"
    );
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
    let sqrtw = fit.finalweights.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrtw_col = sqrtw.view().insert_axis(Axis(1));
    u *= &sqrtw_col;
    let k = fit.penalized_hessian_transformed.to_dense();
    let kview = FaerArrayView::new(&k);
    let factor = factorize_symmetricwith_fallback(kview.as_ref(), Side::Lower).unwrap();
    let mut naive_se = Array1::<f64>::zeros(n);
    for i in 0..n {
        let ui = u.row(i).to_owned();
        let rhs = FaerColView::new(&ui);
        let s = factor.solve(rhs.as_ref());
        let s_arr = Array1::from_shape_fn(p, |j| s[(j, 0)]);
        let quad: f64 = ui.iter().zip(s_arr.iter()).map(|(a, b)| a * b).sum();
        let wi = fit.finalweights[i].max(1e-12);
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

    // Reference single-Newton-step ALO computed inline from the converged
    // geometry: η̃¹ = η̂_i + h_i ℓ_i'(η̂_i) / (1 − w_i h_i), with
    // h_i = x_iᵀH⁻¹x_i, ℓ_i'(η̂_i) = μ_i − y_i, w_i = μ_i(1−μ_i). This is the
    // classical first-order ALO that the exact frozen-curvature scalar solve
    // replaces. We report it alongside the exact predictor as the second
    // approximation of the SAME full-refit ground truth (see the estimand note
    // at the assertions below).
    let eta_hat = &fit.final_eta;
    let mut one_step = Array1::<f64>::zeros(n);
    for i in 0..n {
        let ui = u.row(i).to_owned();
        let rhs = FaerColView::new(&ui);
        let s = factor.solve(rhs.as_ref());
        let quad: f64 = ui.iter().enumerate().map(|(j, a)| a * s[(j, 0)]).sum();
        let wi = fit.finalweights[i];
        let h_i = quad / wi.max(1e-12);
        let mu_i = fit.finalmu[i];
        let ell_prime = mu_i - y[i];
        let denom = (1.0 - wi * h_i).max(1e-12);
        one_step[i] = eta_hat[i] + h_i * ell_prime / denom;
    }
    let (rmse_one_step, _, _, _) = loo_compare(&one_step, &alo.se_sandwich, &loo_pred, &naive_se);

    let (rmse_pred, max_abs_pred, rmse_se, max_abs_se) =
        loo_compare(&alo.eta_tilde, &alo.se_sandwich, &loo_pred, &naive_se);

    // ESTIMAND NOTE (the correctness this test legitimately asserts).
    // `loo_pred` here is the TRUE FULL NONLINEAR LOO REFIT: each fold re-runs
    // `fit_unpenalized` on the n−1 retained rows, so the converged Hessian
    // H₋ᵢ = Σ_{j≠i} w_j(β₋ᵢ) x_j x_jᵀ is rebuilt at the dropped-row optimum.
    // The production ALO `eta_tilde` is the EXACT FROZEN-CURVATURE LOO: it
    // solves the dropped-row stationarity condition with H held at the full-fit
    // value. These are DIFFERENT estimands — they differ at first order in the
    // O(1/n) change of H when row i is dropped, which neither the frozen-
    // curvature solve nor the linearized one-step captures. There is therefore
    //   • NO theorem that the frozen-curvature predictor beats the one-step
    //     against the full refit (on this seed the one-step happens to land
    //     marginally closer), and
    //   • no reason the gap should shrink below the O(1/n) Hessian-change scale.
    // The genuine round-off correctness claim (eta_tilde == the exact
    // frozen-curvature fixed point to 1e-9) is asserted separately in
    // `alo_matches_exact_frozen_curvature_loo_small_n_binomial`. Here we assert
    // only what is true: the frozen-curvature ALO APPROXIMATES the full refit to
    // within the O(1/n) Hessian-change scale, just as the one-step does, and
    // both stay well inside the spread of the LOO predictions they estimate.
    let loo_mean = loo_pred.sum() / n as f64;
    let loo_spread =
        (loo_pred.iter().map(|v| (v - loo_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    // O(1/n) Hessian-change budget on the LOO scale: with the predictions
    // spanning `loo_spread` and a per-fold Hessian relative change of order p/n,
    // the frozen-vs-refit RMSE is bounded by a small multiple of
    // loo_spread * p / n. We allow a 4× safety factor (the constant absorbs the
    // canonical-link leverage prefactor) — this is the principled scale, not a
    // hand-tuned pass threshold.
    let hessian_change_budget = 4.0 * loo_spread * (p as f64) / (n as f64);
    eprintln!(
        "binomial refit LOO n={n} p={p}: loo_spread={loo_spread:.4e} \
         budget={hessian_change_budget:.4e} | exact rmse={rmse_pred:.6e} \
         max_abs={max_abs_pred:.6e} one-step rmse={rmse_one_step:.6e}"
    );
    assert!(
        rmse_pred <= hessian_change_budget,
        "exact frozen-curvature ALO must approximate the full nonlinear LOO refit \
         to within the O(1/n) Hessian-change scale: rmse_pred={rmse_pred:.6e} > \
         budget={hessian_change_budget:.6e} (loo_spread={loo_spread:.6e})"
    );
    // The linearized one-step is the same order-of-approximation to the same
    // ground truth, so it must also sit inside the budget — this pins that the
    // budget is calibrated to the estimand gap, not silently inflated.
    assert!(
        rmse_one_step <= hessian_change_budget,
        "one-step ALO must also approximate the full refit within the same \
         Hessian-change budget (calibration check): one-step rmse={rmse_one_step:.6e} \
         > budget={hessian_change_budget:.6e}"
    );
    // Worst-case single-fold deviation: bounded by the same scale but with a
    // looser per-point factor (the max picks up the highest-leverage row).
    assert!(
        max_abs_pred <= 8.0 * hessian_change_budget,
        "exact ALO max-abs LOO deviation exceeds the Hessian-change scale: \
         max_abs_pred={max_abs_pred:.6e} (budget={hessian_change_budget:.6e})"
    );
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
            let wi = fit_loo.finalweights[r];
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
        let kview = FaerArrayView::new(&xtwx);
        let llt = factorize_symmetricwith_fallback(kview.as_ref(), Side::Lower).unwrap();
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
    let z_full = &fit.solveworking_response;
    let maxworking_jump = z_full
        .iter()
        .zip(eta_full.iter())
        .map(|(&zv, &ev)| (zv - ev).abs())
        .fold(0.0_f64, f64::max);

    assert!(rmse_pred > 1e-2);
    assert!(max_abs_pred > 1e-1);
    assert!(maxworking_jump > 25.0);
}
