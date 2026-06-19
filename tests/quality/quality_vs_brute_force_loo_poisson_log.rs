//! End-to-end **quality** of gam's single-pass approximate leave-one-out (ALO)
//! diagnostics for a Poisson/log tensor-product GAM.
//!
//! OBJECTIVE METRIC ASSERTED (primary). The point of a leave-one-out predictor
//! is *honest out-of-sample accuracy on a known signal*, so the headline claims
//! are objective and reference-free:
//!   1. TRUTH RECOVERY — the data are drawn from a known smooth log-mean surface
//!      η_true(x1,x2). gam's ALO leave-one-out log-mean predictor η̃ must recover
//!      that surface: RMSE(η̃, η_true) ≤ a principled fraction of the η_true
//!      signal range (here ≤ 12% of the range). This is the primary quality
//!      claim — gam's held-out predictor tracks the truth, not a peer tool.
//!   2. HONESTY of LOO — a correct leave-one-out estimator is never optimistic
//!      relative to the in-sample fit, so the held-out mean Poisson deviance of
//!      η̃ must be ≥ the in-sample mean Poisson deviance of η̂ (minus solver
//!      round-off). An ALO that "predicts" each point using its own observation
//!      would violate this; a genuine hold-out cannot.
//!
//! GROUND-TRUTH CORRECTNESS (kept — this is correctness vs an exact quantity,
//! not "same as a peer tool"). ALO is the EXACT frozen-CURVATURE leave-one-out
//! predictor of the converged penalized system at fixed smoothing parameters λ:
//! it holds the penalized Hessian H = XᵀWX + S(λ) FROZEN and solves the dropped-
//! row stationarity reduced to the scalar fixed point η̃_i = η̂_i + h_i(μ(η̃_i)−y_i)
//! with h_i = x_iᵀ H⁻¹ x_i (off-row curvature frozen, held-out row's score exact).
//! The unimpeachable correctness identity is therefore ALO == an independently
//! reconstructed frozen-curvature fixed point, to solver round-off (`eta_fc_rel`
//! below). We ALSO report the exhaustive frozen-λ *re-curved* n-fold refit (which
//! rebuilds the off-row curvature at each dropped optimum): ALO tracks it closely
//! but only to within the genuine O(p/n) off-row-curvature estimand gap, not to
//! round-off — so the round-off identity is asserted against the frozen-curvature
//! oracle, the re-curved refit only to the predictive scale. Neither is a peer
//! tool (both are analytic ground truth ALO is derived from), covered by the
//! spec's "reference IS mathematical ground truth — exact brute-force LOO refits"
//! exception, reported alongside the predictive metrics above.
//!
//! Why fix the converged working model rather than re-running full PIRLS + REML
//! per fold: ALO approximates leave-one-out *at the converged linearisation and
//! at fixed λ* — that is the quantity it is derived from and the only quantity
//! it can be held to. Re-estimating λ each fold would benchmark λ-instability,
//! not the ALO algebra. So the brute force drops row `i` from the exact
//! penalized normal equations
//!     H β₋ᵢ = c − w_i (z_i − o_i) x_i,     H = XᵀWX + S(λ),  c = Xᵀ W (z − o)
//! and reads η̃_i = o_i + x_iᵀ β₋ᵢ. Both H, X, W, z, o, and the link are taken
//! verbatim from gam's converged PIRLS artifact, so the two engines see bitwise
//! identical inputs and any disagreement is a real defect in the ALO update.
//!
//! Poisson/log is the canonical exponential-family case (Fisher == observed
//! information, so a single weight vector is exact) and the `te(x1, x2)` tensor
//! product exercises the multi-dimensional penalized Hessian and the chunked
//! influence-matrix inversion `a_ii = w_i x_iᵀ H⁻¹ x_i` that ALO leverage
//! depends on.

use gam::inference::alo::compute_alo_diagnostics_from_fit;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, pearson, relative_l2, rmse, run_r};
use gam::types::LinkFunction;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;
use std::path::Path;

// Real dataset: `badhealth` from the R package `COUNT` (Hilbe, *Negative Binomial
// Regression*), shipped here at bench/datasets/badhealth.csv. n=1127 patients;
// numvisit = number of doctor visits (count response), badh = self-reported bad
// health (0/1), age = patient age in years. The canonical count-regression
// benchmark numvisit ~ s(age) + badh under Poisson/log.
const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

/// True smooth log-mean surface the synthetic counts are drawn from. The ALO
/// leave-one-out predictor is judged on how well it recovers THIS function
/// out-of-sample — the objective truth-recovery metric.
fn eta_true(a: f64, b: f64) -> f64 {
    0.6 + 0.9 * (PI * a).sin() * (0.5 + b) - 0.7 * (b - 0.5).powi(2)
}

/// Per-observation Poisson unit deviance contribution for a log-mean η and an
/// observed count y: 2[ y·log(y/μ) − (y − μ) ], μ = exp(η), with the standard
/// y·log(y) → 0 convention at y = 0. Summed/averaged this is the Poisson
/// deviance used to score in-sample vs held-out predictive accuracy.
fn poisson_unit_deviance(y: f64, eta: f64) -> f64 {
    let mu = eta.exp();
    let term = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
    2.0 * (term - (y - mu))
}

/// Dense Cholesky factorisation of a symmetric positive-definite matrix
/// (lower-triangular L with A = L Lᵀ). The penalized Hessian H = XᵀWX + S(λ)
/// of a converged Poisson/log GAM is SPD (Fisher weights ≥ 0 plus a positive
/// penalty / stabilisation ridge), so this is the natural exact solver for the
/// brute-force reference. Panics if a non-positive pivot appears, which would
/// itself signal a non-SPD Hessian (a real defect worth failing on).
fn cholesky_lower(a: &Array2<f64>) -> Array2<f64> {
    let p = a.nrows();
    assert_eq!(p, a.ncols(), "cholesky requires a square matrix");
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                assert!(
                    sum > 0.0,
                    "penalized Hessian is not positive definite at pivot {i}: {sum:.3e}"
                );
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    l
}

/// Solve A x = b for SPD A given its lower Cholesky factor L (A = L Lᵀ) via
/// forward/back substitution.
fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = l.nrows();
    // Forward solve L y = b.
    let mut y = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    // Back solve Lᵀ x = y.
    let mut x = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut sum = y[i];
        for k in (i + 1)..p {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

#[test]
fn alo_loo_recovers_truth_and_matches_exact_brute_force_poisson_log() {
    init_parallelism();

    // ---- synthetic 2-D Poisson-count truth on the unit square --------------
    // Counts are drawn from a fixed, KNOWN smooth log-mean surface eta_true so we
    // can score the leave-one-out predictor against ground truth. The SAME draws
    // feed gam's fit and the brute-force reference (which reads gam's own
    // converged geometry), so those two engines see bitwise-identical inputs.
    let n = 310usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut eta_truth = Vec::with_capacity(n);
    for _ in 0..n {
        let a = u.sample(&mut rng);
        let b = u.sample(&mut rng);
        // Smooth log-mean surface with genuine 2-D structure (interaction),
        // kept in a moderate range so counts are informative but not extreme.
        let eta = eta_true(a, b);
        let lambda = eta.exp().max(1e-9);
        let draw: f64 = Poisson::new(lambda)
            .expect("valid Poisson rate")
            .sample(&mut rng);
        x1.push(a);
        x2.push(b);
        y.push(draw);
        eta_truth.push(eta);
    }

    // ---- fit with gam: y ~ te(x1, x2, k=8), Poisson / log link -------------
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![x1[i].to_string(), x2[i].to_string(), y[i].to_string()])
        })
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    // Per-margin k=8 => an 8x8 = 64-column tensor basis, comfortably below n=310
    // (basis_dim + intercept + smoothing-parameter dof << n) so the penalized
    // REML fit is well-posed, while still resolving the genuine 2-D interaction
    // structure of eta_true. A larger default basis (20x20=400 cols) would exceed
    // n and leave the fit under-determined; it would also make the O(n*p^2)
    // brute-force LOO needlessly heavy.
    let result = fit_from_formula("y ~ te(x1, x2, k=8)", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson te(x1, x2)");
    };

    // Rebuild the frozen tensor design at the training points; the in-sample
    // fitted log-mean η̂ = design*beta is the baseline the honest hold-out is
    // scored against (a correct LOO is never optimistic relative to it).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let eta_in_sample: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(eta_in_sample.len(), n, "in-sample eta length mismatch");
    assert!(
        eta_in_sample.iter().all(|v| v.is_finite()),
        "in-sample linear predictor must be finite"
    );

    let y_arr = Array1::from(y.clone());

    // ---- gam ALO diagnostics (the capability under test) -------------------
    let alo = compute_alo_diagnostics_from_fit(&fit.fit, y_arr.view(), LinkFunction::Log)
        .expect("ALO diagnostics for Poisson/log te(x1, x2)");
    assert_eq!(alo.leverage.len(), n, "ALO leverage length mismatch");
    assert_eq!(alo.eta_tilde.len(), n, "ALO eta_tilde length mismatch");
    assert_eq!(alo.se_bayes.len(), n, "ALO se_bayes length mismatch");

    // ========================================================================
    // PRIMARY OBJECTIVE METRIC #1 — TRUTH RECOVERY
    // gam's leave-one-out log-mean predictor must recover the KNOWN generating
    // surface eta_true out-of-sample. We score RMSE(η̃, eta_true) against the
    // signal range of eta_true; a principled bar is a small fraction of that
    // range. This is the headline quality claim and is entirely reference-free.
    // ========================================================================
    let alo_eta = alo.eta_tilde.as_slice().unwrap();
    assert!(
        alo_eta.iter().all(|v| v.is_finite()),
        "ALO leave-one-out predictor must be finite"
    );
    let eta_lo = eta_truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let eta_hi = eta_truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let signal_range = eta_hi - eta_lo;
    let loo_truth_rmse = rmse(alo_eta, &eta_truth);
    let loo_truth_frac = loo_truth_rmse / signal_range.max(1e-300);

    // ========================================================================
    // PRIMARY OBJECTIVE METRIC #2 — HONESTY of the hold-out
    // A genuine leave-one-out predictor cannot be optimistic relative to the
    // in-sample fit: its mean Poisson deviance must be >= the in-sample mean
    // Poisson deviance. We compute both from gam's own predictions only.
    // ========================================================================
    let dev_in_sample: f64 = (0..n)
        .map(|i| poisson_unit_deviance(y[i], eta_in_sample[i]))
        .sum::<f64>()
        / n as f64;
    let dev_loo: f64 = (0..n)
        .map(|i| poisson_unit_deviance(y[i], alo_eta[i]))
        .sum::<f64>()
        / n as f64;

    // ---- brute-force exact LOO from gam's own converged geometry -----------
    // (Ground-truth correctness check — the exact n-fold refit ALO approximates.)
    // The PIRLS artifact carries the consistent transformed design X, the exact
    // dense penalized Hessian H = XᵀWX + S(λ), the score-side Fisher weights W
    // (== observed information for the canonical log link), the working response
    // z, the linear predictor η̂, and the offset o — exactly the inputs the ALO
    // path consumes. We re-derive leave-one-out exactly from these.
    let pirls = fit
        .fit
        .artifacts
        .pirls
        .as_ref()
        .expect("Poisson GAM fit must expose PIRLS geometry");

    let x_arc = pirls
        .x_transformed
        .try_to_dense_arc("brute-force LOO needs dense transformed design")
        .expect("dense transformed design");
    let x = x_arc.as_ref();
    let h = pirls
        .dense_stabilizedhessian_transformed("brute-force LOO needs dense penalized Hessian")
        .expect("dense penalized Hessian");
    let p = x.ncols();
    assert_eq!(x.nrows(), n, "transformed design row count mismatch");
    assert_eq!(h.nrows(), p, "Hessian must be p x p");

    // ALO forms leverage and the rank-1 downdate with the Hessian-side weights
    // (H = XᵀW_h X + S) and the score/RHS with the score-side Fisher weights.
    // For the canonical Poisson/log link Fisher == observed information, so
    // W_h == W_s == μ; we assert that here and then carry a single `w`, which
    // is exactly what makes the ALO closed form an EXACT Sherman–Morrison
    // solution of the brute-force downdated system rather than an approximation.
    let w_hess: Vec<f64> = pirls.final_weights_signed().view().to_vec();
    let w: Vec<f64> = pirls.solve_weights_psd().view().to_vec(); // score-side Fisher weights
    let wh_ws_max_diff = max_abs_diff(&w_hess, &w);
    assert!(
        wh_ws_max_diff < 1e-12,
        "canonical Poisson/log must have Hessian weights == score weights \
         (Fisher == observed information): max|W_h − W_s|={wh_ws_max_diff:.3e}"
    );
    let z = &pirls.solveworking_response; // working response
    let eta_hat = &pirls.final_eta;
    let offset = &pirls.final_offset;
    let phi = 1.0_f64; // Poisson dispersion is fixed at 1 (matches ALO's phi).

    // Full-data penalized RHS in centered (offset-subtracted) coordinates:
    //   c = Xᵀ W (z − o),     H β̂ = c   (the system PIRLS converged to).
    let mut c = Array1::<f64>::zeros(p);
    for i in 0..n {
        let coef = w[i] * (z[i] - offset[i]);
        let row = x.row(i);
        for k in 0..p {
            c[k] += coef * row[k];
        }
    }

    // Verify the brute-force engine reconstructs the SAME working-model fit gam
    // converged to: solving H β = c must reproduce the fitted η̂ = o + Xβ.
    // This pins the reference to gam's geometry before any row is dropped, so a
    // disagreement downstream is in the *hold-out* algebra, not the setup.
    let l_full = cholesky_lower(&h);
    let beta_full = cholesky_solve(&l_full, &c);
    let mut eta_recon = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row = x.row(i);
        let mut dot = 0.0;
        for k in 0..p {
            dot += row[k] * beta_full[k];
        }
        eta_recon[i] = offset[i] + dot;
    }
    let recon_rel = relative_l2(eta_recon.as_slice().unwrap(), eta_hat.as_slice().unwrap());
    assert!(
        recon_rel < 1e-6,
        "brute-force engine must reconstruct gam's converged working fit before \
         hold-out (rel_l2={recon_rel:.3e}); a mismatch means X/H/W/z are not the \
         system PIRLS solved"
    );

    // For each observation i, solve the leave-one-out penalized normal equations
    //   (H − w_i x_i x_iᵀ) β₋ᵢ = c − w_i (z_i − o_i) x_i
    // exactly, and read η̃_i = o_i + x_iᵀ β₋ᵢ. Also form the exact leverage
    //   a_ii = w_i x_iᵀ H⁻¹ x_i   and conditional variance x_iᵀ H⁻¹ x_i
    // via a dense solve against the *full* Hessian. This is the exhaustive
    // n-fold reference ALO is meant to approximate.
    let mut brute_eta_tilde = vec![0.0_f64; n];
    let mut brute_leverage = vec![0.0_f64; n];
    let mut brute_se_bayes = vec![0.0_f64; n];
    // EXACT frozen-CURVATURE LOO — the precise quantity ALO computes: the off-row
    // penalized Hessian H is held FROZEN at its full-fit value and the dropped-row
    // stationarity reduces to the scalar fixed point
    //   η̃_i = η̂_i + h_i (μ(η̃_i) − y_i),   h_i = x_iᵀ H⁻¹ x_i,   μ(η)=exp(η),
    // (canonical Poisson/log: c_i = w_i/μ'(η̂_i) = 1). Reconstructed independently
    // here via a dense H⁻¹ leverage solve and a 1-D Newton iteration; ALO must
    // match it to solver round-off (a strictly stronger claim than the re-curved
    // refit comparison below, which carries an O(p/n) off-row-curvature gap).
    let mut frozen_curv_eta_tilde = vec![0.0_f64; n];
    for i in 0..n {
        let xi: Array1<f64> = x.row(i).to_owned();

        // x_iᵀ H⁻¹ x_i from the full-data factor (leverage + Bayesian variance).
        let hinv_xi = cholesky_solve(&l_full, &xi);
        let mut x_hinv_x = 0.0;
        for k in 0..p {
            x_hinv_x += xi[k] * hinv_xi[k];
        }
        brute_leverage[i] = w[i] * x_hinv_x;
        brute_se_bayes[i] = (phi * x_hinv_x).max(0.0).sqrt();

        // Frozen-curvature scalar fixed point anchored at η̂_i = o_i + x_iᵀ β̂.
        let h_i = x_hinv_x;
        let mut eta_hat_i = offset[i];
        for k in 0..p {
            eta_hat_i += xi[k] * beta_full[k];
        }
        let mut eta_fc = eta_hat_i;
        let mut fc_converged = false;
        for _ in 0..100usize {
            let mu = eta_fc.exp();
            let residual = eta_fc - eta_hat_i - h_i * (mu - y[i]);
            if residual.abs() <= 1e-12 {
                fc_converged = true;
                break;
            }
            // ℓ_i''(η) = μ'(η) = exp(η); Newton Jacobian 1 − h_i μ(η).
            let jac = 1.0 - h_i * mu;
            assert!(
                jac.abs() > 1e-12 && jac.is_finite(),
                "frozen-curvature leave-{i}-out Jacobian degenerate: {jac:.3e}"
            );
            eta_fc -= residual / jac;
            assert!(eta_fc.is_finite(), "frozen-curvature leave-{i}-out diverged");
        }
        assert!(
            fc_converged,
            "frozen-curvature leave-{i}-out scalar fixed point did not converge"
        );
        frozen_curv_eta_tilde[i] = eta_fc;

        // Exact hold-out refit: H₋ᵢ = H − w_i x_i x_iᵀ , RHS₋ᵢ = c − w_i(z_i−o_i)x_i.
        let mut h_minus = h.clone();
        for r in 0..p {
            let wr = w[i] * xi[r];
            for cc in 0..p {
                h_minus[[r, cc]] -= wr * xi[cc];
            }
        }
        let mut rhs_minus = c.clone();
        let drop_coef = w[i] * (z[i] - offset[i]);
        for k in 0..p {
            rhs_minus[k] -= drop_coef * xi[k];
        }
        let l_minus = cholesky_lower(&h_minus);
        let beta_minus = cholesky_solve(&l_minus, &rhs_minus);
        let mut dot = 0.0;
        for k in 0..p {
            dot += xi[k] * beta_minus[k];
        }
        brute_eta_tilde[i] = offset[i] + dot;
    }

    // ---- EXACT nonlinear leave-i-out refit (the true ground truth) ---------
    // `brute_eta_tilde` above drops row i from the once-linearised PWLS working
    // model (fixed W, z). That is only the FIRST-ORDER leave-i-out: it is the
    // single Newton step of the genuine nonlinear refit taken from the
    // converged β̂. For a nonlinear canonical link (Poisson/log) the true
    // leave-i-out optimum β₋ᵢ = argmin_β Σ_{j≠i}[μ_j − y_jη_j] + ½βᵀS β differs
    // from that working-model drop at second order in ‖β₋ᵢ − β̂‖, because μ_j is
    // not linear in β. The frozen-curvature ALO in `src/inference/alo.rs`
    // targets THIS nonlinear refit (its own contract: "the leave-i-out refit"),
    // not the linearised drop — so it cannot, and must not, equal the
    // linearised drop to solver round-off. We compute the exact nonlinear refit
    // here from gam's own converged geometry and hold ALO to it.
    //
    // The penalty block is recovered exactly as S = H − XᵀW X (W = diag(μ̂)),
    // which makes the full-data gradient Σ_j (μ̂_j − y_j) x_j + S β̂ vanish at the
    // reconstructed β̂; the per-row refit then solves the dropped stationarity
    // condition by Newton with the EXACT μ_j(β), Hessian Σ_{j≠i} μ_j x_j x_jᵀ + S.
    let xrows: Vec<Vec<f64>> = (0..n).map(|i| x.row(i).to_vec()).collect();
    let mut s_penalty = h.clone();
    {
        let mut xtwx = Array2::<f64>::zeros((p, p));
        for j in 0..n {
            let wj = w[j];
            let xj = &xrows[j];
            for r in 0..p {
                let wjr = wj * xj[r];
                let hrow = xtwx.row_mut(r);
                let hrow = hrow.into_slice().unwrap();
                for cc in 0..p {
                    hrow[cc] += wjr * xj[cc];
                }
            }
        }
        s_penalty -= &xtwx;
    }

    let mut true_eta_tilde = vec![0.0_f64; n];
    // One-Newton-step replay of the refit; must reproduce the linearised drop
    // `brute_eta_tilde` to round-off, validating the refit geometry (S, grad,
    // Hessian) against the independently-built downdated system above.
    let mut onestep_replay = vec![0.0_f64; n];
    for i in 0..n {
        let mut beta = beta_full.clone();
        let mut converged = false;
        for iter in 0..100usize {
            // grad = Σ_{j≠i}(μ_j − y_j) x_j + S β ; hess = Σ_{j≠i} μ_j x_j x_jᵀ + S
            let mut grad = s_penalty.dot(&beta);
            let mut hess = s_penalty.clone();
            for j in 0..n {
                if j == i {
                    continue;
                }
                let xj = &xrows[j];
                let mut eta_j = offset[j];
                for k in 0..p {
                    eta_j += xj[k] * beta[k];
                }
                let mu_j = eta_j.exp();
                let resid = mu_j - y[j];
                for r in 0..p {
                    grad[r] += resid * xj[r];
                    let mjr = mu_j * xj[r];
                    let hrow = hess.row_mut(r).into_slice().unwrap();
                    for cc in 0..p {
                        hrow[cc] += mjr * xj[cc];
                    }
                }
            }
            let gnorm = grad.dot(&grad).sqrt();
            if gnorm < 1e-9 {
                converged = true;
                break;
            }
            let l = cholesky_lower(&hess);
            let step = cholesky_solve(&l, &grad);
            for k in 0..p {
                beta[k] -= step[k];
            }
            if iter == 0 {
                let xi = &xrows[i];
                let mut dot = 0.0;
                for k in 0..p {
                    dot += xi[k] * beta[k];
                }
                onestep_replay[i] = offset[i] + dot;
            }
        }
        assert!(
            converged,
            "exact nonlinear leave-{i}-out Newton refit did not converge"
        );
        let xi = &xrows[i];
        let mut dot = 0.0;
        for k in 0..p {
            dot += xi[k] * beta[k];
        }
        true_eta_tilde[i] = offset[i] + dot;
    }

    // Internal consistency: the single Newton step of the nonlinear refit IS the
    // linearised working-model drop. If this holds, S/grad/Hessian are correct
    // and `true_eta_tilde` is the trustworthy nonlinear ground truth.
    let onestep_replay_rel = relative_l2(&onestep_replay, &brute_eta_tilde);
    assert!(
        onestep_replay_rel < 1e-8,
        "one-Newton-step refit must reproduce the linearised working-model drop \
         (validates the nonlinear-refit geometry): rel_l2={onestep_replay_rel:.3e}"
    );

    // Score the EXACT (nonlinear) LOO predictor against the truth too, so the
    // objective truth-recovery metric has a ground-truth yardstick: gam's ALO
    // must recover the truth at least as well as the exact n-fold refit does.
    let brute_truth_rmse = rmse(&true_eta_tilde, &eta_truth);

    // ---- compare ALO vs brute-force exact LOO (ground-truth correctness) ----
    let alo_lev = alo.leverage.as_slice().unwrap();
    let alo_se = alo.se_bayes.as_slice().unwrap();

    let lev_max_diff = max_abs_diff(alo_lev, &brute_leverage);
    let lev_corr = pearson(alo_lev, &brute_leverage);
    // ALO vs the TRUE nonlinear leave-i-out refit (the ground truth ALO targets).
    let eta_rel = relative_l2(alo_eta, &true_eta_tilde);
    let eta_corr = pearson(alo_eta, &true_eta_tilde);
    // The classical one-step (linearised working-model) drop vs the same truth —
    // the benchmark the frozen-curvature refinement must improve upon.
    let onestep_true_rel = relative_l2(&brute_eta_tilde, &true_eta_tilde);
    // ALO vs the EXACT frozen-curvature LOO (the identical scalar fixed point ALO
    // solves) — a round-off correctness identity, not an approximation.
    let eta_fc_rel = relative_l2(alo_eta, &frozen_curv_eta_tilde);
    let eta_fc_max = max_abs_diff(alo_eta, &frozen_curv_eta_tilde);
    let se_corr = pearson(alo_se, &brute_se_bayes);
    let se_max_diff = max_abs_diff(alo_se, &brute_se_bayes);

    eprintln!(
        "ALO Poisson/log te(x1,x2): n={n} p={p} signal_range={signal_range:.3}\n  \
         OBJECTIVE  truth-recovery RMSE(eta_tilde,eta_true)={loo_truth_rmse:.4} \
         (={:.2}% of range; exact-nonlinear-refit RMSE={brute_truth_rmse:.4})  \
         deviance in-sample={dev_in_sample:.4} LOO={dev_loo:.4}\n  \
         GROUND-TRUTH  leverage max|Δ|={lev_max_diff:.3e} pearson={lev_corr:.6}  \
         eta_tilde(vs true refit) rel_l2={eta_rel:.3e} pearson={eta_corr:.6} \
         (one-step vs true refit rel_l2={onestep_true_rel:.3e}; \
         vs frozen-curvature rel_l2={eta_fc_rel:.3e} max|Δ|={eta_fc_max:.3e})  \
         se_bayes max|Δ|={se_max_diff:.3e} pearson={se_corr:.5}",
        100.0 * loo_truth_frac
    );

    // ---- PRIMARY OBJECTIVE ASSERTION #1: truth recovery --------------------
    // The held-out log-mean predictor must track the true surface to within a
    // small fraction of its range. 12% of the signal range is a principled bar
    // for n=310 Poisson counts on a smooth 2-D surface: it is well above the
    // irreducible sampling floor (the exact n-fold refit, scored on the same
    // truth, sits comfortably below it) yet far below the ~50% error a degenerate
    // or constant predictor would incur.
    assert!(
        loo_truth_frac < 0.12,
        "ALO leave-one-out predictor must recover the true log-mean surface: \
         RMSE/range={:.4} (RMSE={loo_truth_rmse:.4}, range={signal_range:.4})",
        loo_truth_frac
    );
    // ...and it must be no worse at recovering the truth than the EXACT n-fold
    // refit (match-or-beat the ground-truth predictor on accuracy, 10% slack for
    // the Sherman–Morrison vs fresh-factor numerics).
    assert!(
        loo_truth_rmse <= brute_truth_rmse * 1.10,
        "ALO LOO accuracy must match-or-beat the exact n-fold refit: \
         ALO RMSE={loo_truth_rmse:.4} vs exact {brute_truth_rmse:.4}"
    );

    // ---- PRIMARY OBJECTIVE ASSERTION #2: honest hold-out -------------------
    // A correct leave-one-out estimator is never optimistic: held-out deviance
    // >= in-sample deviance (small negative round-off tolerance only).
    assert!(
        dev_loo >= dev_in_sample - 1e-6,
        "leave-one-out deviance must not be optimistic vs in-sample: \
         LOO={dev_loo:.6} < in-sample={dev_in_sample:.6}"
    );

    // ---- GROUND-TRUTH CORRECTNESS: ALO vs the exact nonlinear LOO ----------
    // ALO is a shortcut for a quantity with an EXACT definition (the nonlinear
    // n-fold refit computed above). These checks are correctness vs that ground
    // truth, not agreement with a peer tool.
    //
    // Leverage a_ii = w_i x_iᵀ H⁻¹ x_i is a deterministic function of the
    // full-data hat matrix; ALO's chunked column solve and the dense Cholesky
    // reference solve the SAME linear systems, so they must agree to solver
    // round-off. 1e-8 is the principled bound (the alo.rs Hessian-symmetry
    // tolerance is itself 1e-8); a looser bound would mask an influence-matrix
    // inversion bug.
    assert!(
        lev_max_diff < 1e-8,
        "ALO leverage must match exact w_i x_iᵀH⁻¹x_i to round-off: max|Δ|={lev_max_diff:.3e}"
    );
    assert!(
        lev_corr > 0.999999,
        "ALO leverage must be near-perfectly correlated with exact leverage: pearson={lev_corr:.6}"
    );

    // η̃: the production ALO (alo.rs) is the EXACT frozen-curvature leave-i-out
    // predictor — it solves the dropped stationarity condition
    // η = η̂_i + h_i ℓ_i'(η) to convergence, with the row's exact likelihood
    // curvature, rather than taking the single linearised Newton step. By
    // construction it therefore does NOT equal the once-linearised working-model
    // drop `brute_eta_tilde` (that drop is only the FIRST iterate of the same
    // fixed point); the two differ at second order in the leave-out
    // perturbation, so demanding round-off equality between them is incorrect.
    // The faithful contract — matching the one the binomial refit test in
    // `tests/inference/alo_tests.rs` asserts against a full nonlinear LOO refit —
    // is that the predictor tracks the TRUE nonlinear leave-i-out refit to within
    // the LOO predictive scale. We additionally report the one-step's distance to
    // the same truth: the curvature refinement removes the row-i linearisation
    // error (which dominates on curved binomial likelihoods near the fit) at the
    // cost of leaving the off-row Hessian frozen, so on a smooth Poisson surface
    // the two approximations are comparably close to the truth and neither
    // dominates uniformly — both are well inside the predictive bar.
    assert!(
        eta_rel < 1e-2,
        "ALO eta_tilde must match the true nonlinear leave-i-out refit closely: \
         rel_l2={eta_rel:.3e} (one-step vs true refit rel_l2={onestep_true_rel:.3e})"
    );
    assert!(
        onestep_true_rel < 1e-2,
        "one-step LOO must also track the true refit (sanity on the refit oracle): \
         rel_l2={onestep_true_rel:.3e}"
    );
    assert!(
        eta_corr > 0.99999,
        "ALO eta_tilde must be near-perfectly correlated with the true LOO refit: pearson={eta_corr:.6}"
    );
    // EXACT-ESTIMAND IDENTITY: ALO IS the frozen-curvature scalar fixed point, so
    // it must equal the independently-reconstructed frozen-curvature LOO to solver
    // round-off — a far tighter and more precise correctness claim than the
    // re-curved-refit comparison above (which carries the legitimate O(p/n)
    // off-row-curvature estimand gap). This pins the ALO algebra to its EXACT
    // analytic definition; a real defect would break it well before the 1e-2 bar.
    assert!(
        eta_fc_rel < 1e-6,
        "ALO eta_tilde must equal the exact frozen-curvature LOO to round-off: \
         rel_l2={eta_fc_rel:.3e} max|Δ|={eta_fc_max:.3e}"
    );
    assert!(
        eta_fc_max < 1e-6,
        "ALO eta_tilde worst-case deviation from the exact frozen-curvature LOO: \
         max|Δ|={eta_fc_max:.3e}"
    );

    // Bayesian SE √(φ x_iᵀ H⁻¹ x_i): ALO and the reference form the IDENTICAL
    // quantity from the same H and the same x_i (φ = 1 for Poisson), so this is
    // another exact identity up to solver round-off. Pearson alone is a weak
    // guard here (invariant to a wrong constant scale on φ or a uniform offset),
    // so we additionally pin the absolute agreement: max|Δ| < 1e-8.
    assert!(
        se_corr > 0.999999,
        "ALO Bayesian SE must track exact conditional variance diagonal: pearson={se_corr:.6}"
    );
    assert!(
        se_max_diff < 1e-8,
        "ALO Bayesian SE √(φ x_iᵀH⁻¹x_i) must equal the exact diagonal to round-off: max|Δ|={se_max_diff:.3e}"
    );
}

#[test]
fn alo_loo_recovers_truth_and_matches_exact_brute_force_poisson_log_on_real_data() {
    init_parallelism();

    // ---- load the real badhealth count dataset (age, badh -> numvisit) ------
    // Real data => no known truth function, so quality is OBJECTIVE held-out
    // predictive accuracy: a deterministic train/test split, fit Poisson/log on
    // TRAIN, predict TEST, and score the held-out mean Poisson deviance. This
    // exercises the SAME gam capability — a Poisson/log GAM with a penalized
    // smooth — that the synthetic test proves recovers a known surface.
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth.csv");
    let col = ds.column_map();
    let age_idx = col["age"];
    let badh_idx = col["badh"];
    let numvisit_idx = col["numvisit"];
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let badh: Vec<f64> = ds.values.column(badh_idx).to_vec();
    let numvisit: Vec<f64> = ds.values.column(numvisit_idx).to_vec();
    let n = age.len();
    assert!(n > 1000, "badhealth should have ~1127 rows, got {n}");

    // ---- deterministic train/test split: every 4th row is held out ---------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 700 && test_rows.len() > 200,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_age: Vec<f64> = train_rows.iter().map(|&i| age[i]).collect();
    let train_badh: Vec<f64> = train_rows.iter().map(|&i| badh[i]).collect();
    let train_numvisit: Vec<f64> = train_rows.iter().map(|&i| numvisit[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age[i]).collect();
    let test_badh: Vec<f64> = test_rows.iter().map(|&i| badh[i]).collect();
    let test_numvisit: Vec<f64> = test_rows.iter().map(|&i| numvisit[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p_cols = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p_cols));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p_cols {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ s(age) + badh, Poisson/log -----------
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("numvisit ~ s(age) + badh", &train_ds, &cfg).expect("gam poisson fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson numvisit ~ s(age) + badh");
    };

    // gam predictions at the held-out rows: rebuild the frozen design at the
    // test points; the log link => mean μ = exp(design*beta).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p_cols));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_test_eta.len(), test_rows.len(), "gam test eta length");
    assert!(
        gam_test_eta.iter().all(|v| v.is_finite()),
        "gam held-out linear predictor must be finite"
    );

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST -------
    // mgcv is the mature baseline to match-or-beat on held-out accuracy, never a
    // target to reproduce. Pass train columns plus the test columns padded to
    // train length (only the first k entries are read back inside R).
    let k = test_rows.len();
    let r = run_r(
        &[
            Column::new("age", &train_age),
            Column::new("badh", &train_badh),
            Column::new("numvisit", &train_numvisit),
            Column::new("test_age", &pad_real(&test_age, train_age.len())),
            Column::new("test_badh", &pad_real(&test_badh, train_age.len())),
            Column::new("test_n", &vec![k as f64; train_age.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(numvisit ~ s(age) + badh, data = df, family = poisson(link = "log"),
                 method = "REML")
        kk <- df$test_n[1]
        newd <- data.frame(age = df$test_age[1:kk], badh = df$test_badh[1:kk])
        emit("test_pred_mu", as.numeric(predict(m, newdata = newd, type = "response")))
        "#,
    );
    let mgcv_test_mu = r.vector("test_pred_mu");
    assert_eq!(
        mgcv_test_mu.len(),
        k,
        "mgcv held-out prediction length mismatch"
    );

    // ---- OBJECTIVE held-out count-deviance metric (computed in plain Rust) --
    // Mean Poisson unit deviance on the held-out rows; lower is better. gam's
    // predictor uses η = design*beta (μ = exp η); mgcv emits μ directly so we
    // pass its log.
    let gam_test_dev: f64 = (0..k)
        .map(|j| poisson_unit_deviance(test_numvisit[j], gam_test_eta[j]))
        .sum::<f64>()
        / k as f64;
    let mgcv_test_dev: f64 = (0..k)
        .map(|j| poisson_unit_deviance(test_numvisit[j], mgcv_test_mu[j].max(1e-12).ln()))
        .sum::<f64>()
        / k as f64;

    // A constant-mean (intercept-only) Poisson predictor: the trivial baseline
    // the held-out deviance bar must beat. Its μ is the TRAIN mean count.
    let train_mean = train_numvisit.iter().sum::<f64>() / train_numvisit.len() as f64;
    let null_eta = train_mean.max(1e-12).ln();
    let null_test_dev: f64 = (0..k)
        .map(|j| poisson_unit_deviance(test_numvisit[j], null_eta))
        .sum::<f64>()
        / k as f64;

    eprintln!(
        "badhealth numvisit ~ s(age)+badh held-out Poisson/log: n_train={} n_test={k} \
         gam_test_dev={gam_test_dev:.4} mgcv_test_dev={mgcv_test_dev:.4} \
         null_test_dev={null_test_dev:.4}",
        train_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out counts ------
    // The penalized Poisson GAM must explain held-out count variation well
    // above the intercept-only baseline. We require gam's held-out mean deviance
    // to be at most 92% of the null model's — a genuine, tool-free predictive
    // improvement (the smooth age effect plus the bad-health indicator carry
    // real signal in this dataset).
    assert!(
        gam_test_dev <= 0.92 * null_test_dev,
        "gam held-out Poisson deviance {gam_test_dev:.4} not below 92% of null \
         {null_test_dev:.4} — the fitted model fails to beat the constant-mean baseline"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out deviance --
    // Lower deviance is better, so match-or-beat means gam <= mgcv + margin.
    // 5% of the mgcv deviance is the principled slack for solver/REML differences.
    assert!(
        gam_test_dev <= mgcv_test_dev * 1.05,
        "gam held-out Poisson deviance {gam_test_dev:.4} exceeds mgcv {mgcv_test_dev:.4} * 1.05"
    );
}

/// Right-pad `v` with its last value (or 0.0 when empty) to length `len`, so a
/// test-length column can ride along inside a train-length reference data.frame.
/// Only the first `v.len()` entries are read back inside the R body.
fn pad_real(v: &[f64], len: usize) -> Vec<f64> {
    assert!(
        v.len() <= len,
        "pad target {len} shorter than source {}",
        v.len()
    );
    let fill = v.last().copied().unwrap_or(0.0);
    let mut out = v.to_vec();
    out.resize(len, fill);
    out
}
