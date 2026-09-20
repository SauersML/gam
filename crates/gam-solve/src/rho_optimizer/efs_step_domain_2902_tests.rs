//! #2902 — the EFS log-λ step is taken whole and bounded only by the outer
//! domain.
//!
//! `efs_log_step_from_grad` used to box every coordinate's step to
//! `±EFS_MAX_STEP = 5`, a width no property of the problem sets. On a
//! well-determined penalty whose REML root sits far from the seed, the box
//! turned one multiplicative step into several and spent an inner solve per
//! extra step. The fixture is the orthogonal Gaussian REML criterion with unit
//! noise, a single smoothing parameter and penalty eigenvalues `s = (1, 2, 4,
//! 8)`:
//!
//! ```text
//!   β̂_j = y_j / (1 + λ s_j),   pen = λ Σ s_j β̂_j²,   t = Σ λ s_j / (1 + λ s_j)
//!   V(ρ) = ½ [Σ (y_j − β̂_j)² + pen + Σ log(1 + λ s_j) − m ρ]
//!   ∂V/∂ρ = ½ (pen + t − m),       q_eff = pen.
//! ```
//!
//! With the strong signal `y = (3000, 2000, 1000, 500)` the first
//! multiplicative step from `ρ = 0` is `log(1 − 2g/q) ≈ −14.88` and REML's
//! root is at `ρ* ≈ −15.56`. Measured: with the box the walk evaluated EFS at
//! `0, −5, −10, −15` before reaching the root (7 EFS evaluations in all); the
//! whole step evaluates `0, −14.88` and then the root (5 in all).

use super::*;
use crate::estimate::reml::reml_outer_engine::efs_log_step_from_grad;

const PENALTY_EIGENVALUES: [f64; 4] = [1.0, 2.0, 4.0, 8.0];
const STRONG_SIGNAL: [f64; 4] = [3000.0, 2000.0, 1000.0, 500.0];

/// `(V, ∂V/∂ρ, q_eff)` of the orthogonal Gaussian REML criterion at `rho`.
fn orthogonal_reml(rho: f64, y: &[f64; 4]) -> (f64, f64, f64) {
    let lambda = rho.exp();
    let m = PENALTY_EIGENVALUES.len() as f64;
    let (mut rss, mut pen, mut trace, mut logdet) = (0.0, 0.0, 0.0, 0.0);
    for (&s, &yj) in PENALTY_EIGENVALUES.iter().zip(y.iter()) {
        let shrink = 1.0 + lambda * s;
        let beta = yj / shrink;
        rss += (yj - beta) * (yj - beta);
        pen += lambda * s * beta * beta;
        trace += lambda * s / shrink;
        logdet += shrink.ln();
    }
    let value = 0.5 * (rss + pen + logdet - m * rho);
    let gradient = 0.5 * (pen + trace - m);
    (value, gradient, pen)
}

/// The EFS walk on the strong-signal fixture takes the first multiplicative
/// step whole: the second EFS evaluation sits at the model root
/// `log(1 − 2g/q)` from the seed, not at the old box edge `−5`, and the walk
/// certifies REML's root in fewer evaluations than the boxed walk needed.
#[test]
fn efs_takes_the_whole_multiplicative_step_toward_a_far_root_2902() {
    let efs_points: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(Array1::from_elem(1, 0.0))
        .with_max_iter(50);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(orthogonal_reml(theta[0], &STRONG_SIGNAL).0),
        |_: &mut (), theta: &Array1<f64>| {
            let (value, gradient, _) = orthogonal_reml(theta[0], &STRONG_SIGNAL);
            Ok(OuterEval {
                cost: value,
                gradient: Array1::from_elem(1, gradient),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        {
            let efs_points = Arc::clone(&efs_points);
            Some(move |_: &mut (), theta: &Array1<f64>| {
                efs_points.lock().expect("efs point log").push(theta[0]);
                let (value, gradient, q_eff) = orthogonal_reml(theta[0], &STRONG_SIGNAL);
                Ok(EfsEval {
                    cost: value,
                    steps: vec![efs_log_step_from_grad(q_eff, gradient).unwrap_or(0.0)],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    consecutive_restored_incumbents: None,
                })
            })
        },
    );

    let result = problem
        .run(&mut obj, "EFS far-root walk")
        .expect("the EFS walk on the orthogonal REML fixture must succeed");
    let points = efs_points.lock().expect("efs point log").clone();
    assert_eq!(
        result.plan_used.solver,
        Solver::Efs,
        "EFS points: {points:?}"
    );
    assert!(result.converged(), "EFS points: {points:?}");

    let (_, g0, q0) = orthogonal_reml(0.0, &STRONG_SIGNAL);
    let root_step = (1.0 - 2.0 * g0 / q0).ln();
    assert!(
        root_step < -5.0,
        "fixture must put the model root beyond the old box"
    );
    assert!(points.len() >= 2, "EFS points: {points:?}");
    assert!(
        (points[1] - root_step).abs() <= 1e-9 * root_step.abs(),
        "the second EFS evaluation must sit at the whole model step {root_step} \
         from the seed, not a truncated one (EFS points: {points:?})"
    );

    let (_, g_final, _) = orthogonal_reml(result.rho[0], &STRONG_SIGNAL);
    assert!(
        g_final.abs() <= 1e-6,
        "the walk must end at REML's root: ρ={} ∂V/∂ρ={g_final:e} (EFS points: {points:?})",
        result.rho[0]
    );
    let away_from_root = points
        .iter()
        .take_while(|&&rho| orthogonal_reml(rho, &STRONG_SIGNAL).1.abs() > 1e-6)
        .count();
    assert_eq!(
        away_from_root, 2,
        "the walk must reach the root after the seed and one whole step; the boxed \
         walk spent four evaluations getting there (EFS points: {points:?})"
    );
}
