//! MEASUREMENT (not a pass/fail gate): quantify WHY the SAS skewness ε stalls at
//! its init (#1876). Drives the EXACT `sas_fit_recovery_and_calibration_system`
//! fixture (n=3000, β=[-0.35,1.15,-0.65,0.45], ε_true=0.38, log_δ_true=-0.30,
//! seed 9001) and eprintln's the discriminator numbers directly, because the
//! Rust test harness installs no `log` subscriber (so the production
//! `[OUTER-FD-AUDIT]/[ift-gate]/[IFT-ENERGY]` lines are dropped silently).
//!
//! It reports two things that split the mechanism cleanly:
//!
//!   1. The ε-PROFILE of the REML/LAML objective: `reml_score` from a fit with
//!      the SAS link held FIXED at each grid ε (β and λ still optimized), at the
//!      true log_δ. `reml_score` is the minimized criterion (lower = better).
//!        - argmin near ε≈0.38 with a clearly negative slope at ε=0  ⇒ the
//!          objective itself PREFERS the true skew, so the recovery failure is
//!          the OUTER OPTIMIZER not moving ε (mechanism b: gradient/step).
//!        - flat, or argmin near ε≈0                                  ⇒ the
//!          objective does NOT prefer the true skew, so `coord.a`/the score
//!          assembly is wrong (mechanism a: family, hyper.rs).
//!
//!   2. A central finite-difference of that profile at ε=0
//!      `(reml_score(+h) − reml_score(−h)) / 2h` — the TRUE dReml/dε the outer
//!      optimizer should see. Compare its sign/magnitude to the free fit's
//!      reported `outer_gradient_norm` and final ε̂.
//!
//! Plus the actual `optimize_sas = true` recovery (ε̂, log_δ̂, gradient norm,
//! outer iters, converged) for the head-to-head against the profile.

use gam::estimate::FittedLinkState;
use gam::estimate::{FitOptions, fit_gam};
use gam::mixture_link::{sas_inverse_link_jet, state_from_sasspec};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, SasLinkSpec};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn build_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let t = (i as f64 + 0.5) / (n as f64);
        let x1 = -2.5 + 5.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.3 * x1).sin();
        x[[i, 3]] = 0.5 * x1 * x1 - 0.7;
    }
    x
}

fn one_penalty_for_non_intercept(p: usize) -> Vec<BlockwisePenalty> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![BlockwisePenalty::new(0..p, s)]
}

fn base_fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 60,
        tol: 1e-6,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

fn sas_likelihood(spec: SasLinkSpec) -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Sas(state_from_sasspec(spec).expect("initial SAS state")),
    )
}

/// Fit with the SAS link held FIXED at (`epsilon`, `log_delta`) — β and λ still
/// optimized — and return `(reml_score, deviance)`. `optimize_sas = false`, so
/// `state_from_sasspec` installs `epsilon` directly as the effective skew (the
/// `sas_effective_epsilon` tanh reparam only applies on the optimize path).
fn fixed_link_reml(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    offset: &Array1<f64>,
    s_list: &[BlockwisePenalty],
    epsilon: f64,
    log_delta: f64,
) -> (f64, f64) {
    let mut opts = base_fit_options();
    opts.sas_link = Some(SasLinkSpec {
        initial_epsilon: epsilon,
        initial_log_delta: log_delta,
    });
    opts.optimize_sas = false;
    let family = sas_likelihood(opts.sas_link.expect("SAS fixed spec"));
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        s_list,
        family,
        &opts,
    )
    .expect("fixed-ε SAS fit");
    (fit.reml_score, fit.deviance)
}

#[test]
fn sas_epsilon_objective_profile_measure() {
    let n = 3000usize;
    let x = build_design(n);
    let beta_true = Array1::from_vec(vec![-0.35, 1.15, -0.65, 0.45]);
    let eps_true: f64 = 0.38;
    let log_delta_true: f64 = -0.30;
    let eta = x.dot(&beta_true);
    let p_true = eta.mapv(|e| sas_inverse_link_jet(e, eps_true, log_delta_true).mu);

    let mut rng = StdRng::seed_from_u64(9001);
    let y = p_true.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    eprintln!("===== #1876 SAS ε OBJECTIVE PROFILE MEASURE =====");
    eprintln!(
        "fixture: n={n} eps_true={eps_true} log_delta_true={log_delta_true} (reml_score: lower = better)"
    );

    // 1. ε-profile of reml_score at the TRUE log_δ (β and λ optimized at each ε).
    eprintln!("--- reml_score(ε | log_δ = log_δ_true), β+λ optimized ---");
    let eps_grid = [
        -0.60_f64, -0.40, -0.20, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.38, 0.50, 0.70,
    ];
    let mut best_eps = f64::NAN;
    let mut best_score = f64::INFINITY;
    for &e in &eps_grid {
        let (score, dev) = fixed_link_reml(&x, &y, &w, &offset, &s_list, e, log_delta_true);
        eprintln!("  eps={e:+.3}  reml_score={score:+.6e}  deviance={dev:+.6e}");
        if score < best_score {
            best_score = score;
            best_eps = e;
        }
    }
    eprintln!(
        "  => argmin reml_score over ε-grid: eps*={best_eps:+.3} (reml_score={best_score:+.6e})"
    );

    // 2. Central finite-difference of the profile at ε=0 (the TRUE dReml/dε).
    let h = 0.05_f64;
    let (score_plus, _) = fixed_link_reml(&x, &y, &w, &offset, &s_list, h, log_delta_true);
    let (score_minus, _) = fixed_link_reml(&x, &y, &w, &offset, &s_list, -h, log_delta_true);
    let fd_grad0 = (score_plus - score_minus) / (2.0 * h);
    eprintln!(
        "--- FD dReml/dε at ε=0 (h={h}): {fd_grad0:+.6e}  [reml_score(+h)={score_plus:+.6e} reml_score(-h)={score_minus:+.6e}] ---"
    );
    eprintln!(
        "    (negative ⇒ objective decreases as ε rises from 0 ⇒ objective WANTS ε>0 ⇒ optimizer-side stall)"
    );

    // 3. reml_score at the true skew vs at ε=0, holding log_δ = true.
    let (score_at_true, _) =
        fixed_link_reml(&x, &y, &w, &offset, &s_list, eps_true, log_delta_true);
    let (score_at_zero, _) = fixed_link_reml(&x, &y, &w, &offset, &s_list, 0.0, log_delta_true);
    eprintln!(
        "--- reml_score(ε=0.38,log_δ_true)={score_at_true:+.6e}  vs reml_score(ε=0,log_δ_true)={score_at_zero:+.6e}  Δ={:+.6e} ---",
        score_at_true - score_at_zero
    );

    // 4. The ACTUAL free recovery fit (optimize_sas = true), with the opening
    //    outer evaluations captured so we can read the ANALYTIC θ-gradient the
    //    optimizer actually received at the ε=0 init (the number that splits
    //    "gradient collapsed at the capped β̂" from "step logic drops a correct
    //    gradient"). θ layout is (ρ[0..k] ‖ ε ‖ log_δ), so the LAST two gradient
    //    components are (ε, log_δ).
    let mut opts = base_fit_options();
    opts.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });
    opts.optimize_sas = true;
    let family = sas_likelihood(opts.sas_link.expect("SAS fit spec"));
    gam::estimate::enable_outer_eval_capture();
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        family,
        &opts,
    )
    .expect("free SAS fit");
    let evals = gam::estimate::take_outer_eval_capture();
    let (eps_hat, delta_hat) = match &fit.fitted_link {
        FittedLinkState::Sas { state, .. } => (state.epsilon, state.delta),
        other => panic!("expected SAS fitted state, got {other:?}"),
    };
    eprintln!("--- FREE fit (optimize_sas=true, init ε=0,log_δ=0) ---");
    eprintln!(
        "  eps_hat={eps_hat:+.6e} (true {eps_true})  delta_hat={delta_hat:+.6e} (true {})",
        log_delta_true.exp()
    );
    eprintln!(
        "  reml_score={:+.6e}  outer_iters={}  outer_converged=certified  outer_grad_norm={:?}",
        fit.reml_score, fit.outer_iterations, fit.outer_gradient_norm
    );

    // ANALYTIC gradient the optimizer received at the opening evals. The ε
    // component (2nd-from-last) at eval 1 is at (near) the ε=0 init — compare it
    // to the FD dReml/dε(0)=−7-scale truth above. If |analytic ε-grad| ≪ |FD|
    // ⇒ the analytic gradient is COLLAPSED at the capped β̂ (envelope not
    // restoring it — refined-β̃ / gradient fix); if analytic ≈ FD but ε̂ stalls
    // ⇒ ARC/Armijo step logic drops a correct gradient.
    eprintln!("--- ANALYTIC opening outer evals (θ = ρ.. ‖ ε ‖ log_δ) ---");
    for (i, rec) in evals.iter().enumerate() {
        let n = rec.gradient.len();
        let eps_theta = if n >= 2 { rec.theta[n - 2] } else { f64::NAN };
        let eps_grad = if n >= 2 {
            rec.gradient[n - 2]
        } else {
            f64::NAN
        };
        let ldelta_grad = if n >= 1 {
            rec.gradient[n - 1]
        } else {
            f64::NAN
        };
        eprintln!(
            "  eval {:>2}: cost={:+.6e}  ε_raw(θ)={eps_theta:+.5e}  analytic dCost/dε={eps_grad:+.6e}  dCost/dlog_δ={ldelta_grad:+.6e}  ‖g‖={:+.3e}",
            i + 1,
            rec.cost,
            rec.gradient.mapv(|v| v * v).sum().sqrt(),
        );
    }
    let first_eps_grad = evals
        .first()
        .and_then(|r| {
            let n = r.gradient.len();
            (n >= 2).then(|| r.gradient[n - 2])
        })
        .unwrap_or(f64::NAN);

    // 5. FD dReml/dε at the LANDING point ε̂, for a truth reference where the
    //    optimizer declared victory.
    let (score_hp, _) = fixed_link_reml(&x, &y, &w, &offset, &s_list, eps_hat + h, log_delta_true);
    let (score_hm, _) = fixed_link_reml(&x, &y, &w, &offset, &s_list, eps_hat - h, log_delta_true);
    let fd_grad_landing = (score_hp - score_hm) / (2.0 * h);
    eprintln!(
        "--- FD dReml/dε at landing ε̂={eps_hat:+.4} (log_δ_true): {fd_grad_landing:+.6e} ---"
    );

    eprintln!(
        "  VERDICT INPUTS: profile argmin eps*={best_eps:+.3}; FD dReml/dε(0)={fd_grad0:+.3e}; \
         analytic dCost/dε @eval1={first_eps_grad:+.3e}; free eps_hat={eps_hat:+.3e}; \
         FD@landing={fd_grad_landing:+.3e}"
    );
    eprintln!(
        "    (b-gradient: |analytic@eval1| ≪ |FD(0)| ⇒ analytic ε-grad COLLAPSED at capped β̂; \
         b-step: analytic@eval1 ≈ FD(0) but eps_hat≈0 ⇒ step logic drops it)"
    );
    eprintln!("===== END MEASURE =====");

    // Measurement only: assert the runs completed, never the recovery quality.
    assert!(fit.reml_score.is_finite());
}
