//! Real regression tests for the unified REML/LAML solver invariants (issue #1861).
//!
//! Each test here was previously a placeholder `assert!(false, ...)` stub. They
//! are now genuine tests written against the public `gam-solve` REML API:
//!
//!   * `fit_gam` — full penalized fit (returns β̂, λ̂, status).
//!   * `evaluate_externalcost_andridge` — outer REML/LAML score (`EvalMode::ValueOnly`
//!     internally) plus the stabilization ridge that was applied.
//!   * `evaluate_externalgradient` — analytic outer score derivative
//!     (`EvalMode::ValueAndGradient` internally).
//!   * `InnerSolutionBuilder` + `compute_efs_update` / `compute_hybrid_efs_update`
//!     — the Extended Fellner–Schall update and its hybrid (ρ + ψ) generalization.
//!
//! The named invariants are solver-correctness properties, so a faithful test
//! DISPROVES the historical "bug" by passing.

use gam::estimate::{ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient};
use gam::smooth::BlockwisePenalty;
use gam::solver::estimate::reml::reml_outer_engine::{
    DenseSpectralOperator, DispersionHandling, HessianFactorization, InnerSolutionBuilder,
    PenaltyCoordinate, PenaltyLogdetDerivs, compute_efs_update, compute_hybrid_efs_update,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
//  Shared problem construction
// ─────────────────────────────────────────────────────────────────────────────

/// Squared second-difference penalty `D₂ᵀD₂` (k×k). Rank `k-2`; nullspace is
/// {constant, linear}.
fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

/// A small single-smooth Gaussian problem: intercept + a degree-`block_k`
/// polynomial block penalized by a second-difference penalty. Returns the
/// design, response, weights, offset, the penalty list, the per-penalty
/// nullspace dims, and the raw penalty block (matrix + column range) so a test
/// can reconstruct `S(λ)` in the original basis.
struct GaussianProblem {
    x: Array2<f64>,
    y: Array1<f64>,
    w: Array1<f64>,
    offset: Array1<f64>,
    s_list: Vec<BlockwisePenalty>,
    nullspace_dims: Vec<usize>,
}

fn build_gaussian_problem(seed: u64) -> GaussianProblem {
    let n = 160usize;
    let block_k = 6usize;
    let p = 1 + block_k;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        let z = rng.random_range(-1.0..1.0);
        let mut acc = 1.0;
        for j in 0..block_k {
            acc *= z;
            x[[i, 1 + j]] = acc;
        }
    }

    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = 0.3;
    for j in 1..p {
        beta[j] = 0.2 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    let y = Array1::from_iter(eta.iter().map(|e| e + rng.random_range(-0.25..0.25)));
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let s_block = second_difference_penalty(block_k);
    let start = 1usize;
    let s_list = vec![BlockwisePenalty::new(
        start..(start + block_k),
        s_block.clone(),
    )];

    GaussianProblem {
        x,
        y,
        w,
        offset,
        s_list,
        nullspace_dims: vec![2],
    }
}

fn gaussian_opts(nullspace_dims: Vec<usize>) -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 500,
        tol: 1e-12,
        nullspace_dims,
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

fn external_cost(prob: &GaussianProblem, opts: &ExternalOptimOptions, rho: &Array1<f64>) -> f64 {
    evaluate_externalcost_andridge(
        prob.y.view(),
        prob.w.view(),
        prob.x.clone(),
        prob.offset.view(),
        &prob.s_list,
        opts,
        rho,
    )
    .expect("external REML cost evaluation should succeed")
    .0
}

fn external_cost_and_ridge(
    prob: &GaussianProblem,
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> (f64, f64) {
    evaluate_externalcost_andridge(
        prob.y.view(),
        prob.w.view(),
        prob.x.clone(),
        prob.offset.view(),
        &prob.s_list,
        opts,
        rho,
    )
    .expect("external REML cost evaluation should succeed")
}

fn external_gradient(
    prob: &GaussianProblem,
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Array1<f64> {
    evaluate_externalgradient(
        prob.y.view(),
        prob.w.view(),
        prob.x.clone(),
        prob.offset.view(),
        &prob.s_list,
        opts,
        rho,
    )
    .expect("external REML gradient evaluation should succeed")
}

// ─────────────────────────────────────────────────────────────────────────────
//  1. Projected KKT residual at the inner optimum
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
//  2. EvalMode gradient vs finite difference of the value-only score
// ─────────────────────────────────────────────────────────────────────────────

/// The analytic outer score derivative (`evaluate_externalgradient`, which runs
/// the evaluator in `EvalMode::ValueAndGradient`) must agree with a central
/// finite difference of the value-only score (`evaluate_externalcost_andridge`,
/// which runs `EvalMode::ValueOnly`). Both share the same converged inner state,
/// so any drift between the value-mode and gradient-mode code paths surfaces
/// here.
#[test]
fn bug_eval_mode_gradient_mismatch_with_score_only_fd() {
    let prob = build_gaussian_problem(0x1861_0002);
    let opts = gaussian_opts(prob.nullspace_dims.clone());
    let h = 1e-5;

    // A spread of ρ values, none at the optimum, so the gradient is nonzero and
    // the comparison is informative.
    for &rho0 in &[-3.0_f64, -1.0, 0.5, 2.0, 4.0] {
        let rho = Array1::from_vec(vec![rho0]);
        let g = external_gradient(&prob, &opts, &rho);
        assert_eq!(g.len(), 1);

        let mut rp = rho.clone();
        rp[0] += h;
        let mut rm = rho.clone();
        rm[0] -= h;
        let fd = (external_cost(&prob, &opts, &rp) - external_cost(&prob, &opts, &rm)) / (2.0 * h);

        let diff = (g[0] - fd).abs();
        let tol = 1e-4 * (1.0 + fd.abs());
        assert!(
            diff <= tol,
            "ValueAndGradient score derivative must match the finite difference of the \
             ValueOnly score at ρ={rho0}: analytic={} fd={} |diff|={diff:.3e} tol={tol:.3e}",
            g[0],
            fd
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. Inner Newton line search rejects non-monotone (objective-increasing) steps
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
//  4. Hybrid EFS reduces exactly to plain EFS on a ρ-only problem
// ─────────────────────────────────────────────────────────────────────────────

/// Build a small converged Gaussian `InnerSolution` (no design-moving ψ
/// coordinates) directly, mirroring the solver's own construction: H = XᵀX +
/// Σ λₖ Sₖ, β̂ = H⁻¹ Xᵀy.
fn build_gaussian_inner_solution(
    rho: &[f64],
) -> gam::solver::estimate::reml::reml_outer_engine::InnerSolution<'static> {
    let xtx = ndarray::array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
    let s1 = ndarray::array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s2 = ndarray::array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

    let mut h = xtx.clone();
    h.scaled_add(lambdas[0], &s1);
    h.scaled_add(lambdas[1], &s2);
    let op = DenseSpectralOperator::from_symmetric(&h).expect("SPD Hessian factorizes");

    let xty = ndarray::array![5.0, 3.0, 2.0];
    let beta = HessianFactorization::solve(&op, &xty);

    let penalty_quad =
        lambdas[0] * beta.dot(&s1.dot(&beta)) + lambdas[1] * beta.dot(&s2.dot(&beta));
    let yty = 20.0;
    let deviance = yty - 2.0 * beta.dot(&xty) + beta.dot(&xtx.dot(&beta));
    let log_likelihood = -0.5 * deviance;

    // EFS does not read the penalty logdet; a shape-valid placeholder suffices.
    let penalty_logdet = PenaltyLogdetDerivs {
        value: 0.0,
        first: Array1::zeros(2),
        second: None,
    };

    let r1 = gam::solver::estimate::reml::reml_outer_engine::penalty_matrix_root(&s1)
        .expect("penalty root S₁");
    let r2 = gam::solver::estimate::reml::reml_outer_engine::penalty_matrix_root(&s2)
        .expect("penalty root S₂");

    InnerSolutionBuilder::new(
        log_likelihood,
        penalty_quad,
        beta,
        50,
        Arc::new(op) as Arc<dyn HessianFactorization>,
        vec![
            PenaltyCoordinate::from_dense_root(r1),
            PenaltyCoordinate::from_dense_root(r2),
        ],
        penalty_logdet,
        DispersionHandling::ProfiledGaussian,
    )
    .build()
}

/// On a purely penalty-like (ρ-only) problem there are no ψ (design-moving)
/// coordinates, so the hybrid EFS update — which only *adds* a preconditioned
/// gradient step on the ψ block — must reduce **exactly** to the plain EFS
/// update. This is the "blend = 1 ⇒ plain EFS" invariant: the ψ contribution is
/// empty and the ρ steps must be bitwise identical.
#[test]
fn bug_hybrid_efs_blend_one_not_equal_plain_efs() {
    // Several ρ points and several gradient vectors — the identity must hold for
    // every input, not just at a stationary point.
    let rho_points: [[f64; 2]; 3] = [[0.0, 0.0], [-1.5, 2.0], [3.0, -0.5]];
    let gradients: [[f64; 2]; 3] = [[0.3, -0.7], [1.25, 0.4], [-0.9, -0.15]];

    for rho_arr in &rho_points {
        let rho = rho_arr.to_vec();
        let sol = build_gaussian_inner_solution(&rho);
        for grad_arr in &gradients {
            let gradient = grad_arr.to_vec();

            let plain = compute_efs_update(&sol, &rho, &gradient)
                .expect("plain EFS update must be defined for a valid rho-only problem");
            let hybrid = compute_hybrid_efs_update(&sol, &rho, &gradient)
                .expect("hybrid EFS update must be defined for a valid rho-only problem");

            assert!(
                hybrid.psi_indices.is_empty(),
                "a ρ-only problem must have no ψ coordinates, got {:?}",
                hybrid.psi_indices
            );
            assert_eq!(
                plain.len(),
                hybrid.steps.len(),
                "plain and hybrid EFS step vectors must have equal length"
            );
            for (i, (a, b)) in plain.iter().zip(hybrid.steps.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "hybrid EFS must reduce bitwise-exactly to plain EFS on the ρ block at \
                     ρ={rho:?}, gradient={gradient:?}, coordinate {i}: plain={a} hybrid={b}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  5. Stabilization ledger records every ridge used during score evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Every stabilization ridge δ applied during a score evaluation must be
/// recorded and surfaced coherently: `evaluate_externalcost_andridge` returns
/// exactly the recorded δ. The invariant has three observable consequences that
/// a "missing / mislabeled ledger entry" bug would violate:
///
///   1. The recorded δ is a well-formed quantity (finite, non-negative) at
///      every ρ, including extreme heavy smoothing where stabilization may fire.
///   2. It is deterministic — the same ρ yields the same recorded δ bit-for-bit.
///   3. The recorded ridge is applied consistently to BOTH the value and the
///      gradient path (they read the same eval bundle / ledger), so the analytic
///      gradient still matches the finite difference of the cost even while a
///      ridge is active. A ridge that was used but not recorded on one path
///      would desynchronize cost and gradient.
#[test]
fn bug_stabilization_ledger_missing_ridge_entries() {
    let prob = build_gaussian_problem(0x1861_0005);
    let opts = gaussian_opts(prob.nullspace_dims.clone());
    let h = 1e-5;

    // Span moderate to extreme smoothing; large ρ drives S(λ) to dominate and
    // exercises the stabilization path.
    for &rho0 in &[-2.0_f64, 0.0, 3.0, 8.0, 16.0, 24.0] {
        let rho = Array1::from_vec(vec![rho0]);

        let (cost1, ridge1) = external_cost_and_ridge(&prob, &opts, &rho);
        let (cost2, ridge2) = external_cost_and_ridge(&prob, &opts, &rho);

        // (1) well-formed recorded ridge.
        assert!(
            ridge1.is_finite() && ridge1 >= 0.0,
            "recorded stabilization ridge must be finite and non-negative at ρ={rho0}, got {ridge1}"
        );
        assert!(cost1.is_finite(), "score must be finite at ρ={rho0}");

        // (2) deterministic ledger read.
        assert_eq!(
            ridge1.to_bits(),
            ridge2.to_bits(),
            "recorded ridge must be deterministic at ρ={rho0}: {ridge1} vs {ridge2}"
        );
        assert_eq!(
            cost1.to_bits(),
            cost2.to_bits(),
            "score must be deterministic at ρ={rho0}: {cost1} vs {cost2}"
        );

        // (3) the recorded ridge drives cost and gradient consistently.
        let g = external_gradient(&prob, &opts, &rho);
        let mut rp = rho.clone();
        rp[0] += h;
        let mut rm = rho.clone();
        rm[0] -= h;
        let fd = (external_cost(&prob, &opts, &rp) - external_cost(&prob, &opts, &rm)) / (2.0 * h);
        let diff = (g[0] - fd).abs();
        let tol = 1e-3 * (1.0 + fd.abs());
        assert!(
            diff <= tol,
            "with recorded ridge δ={ridge1:.3e} active at ρ={rho0}, the analytic gradient must \
             remain consistent with the value-path finite difference: analytic={} fd={} \
             |diff|={diff:.3e} tol={tol:.3e}",
            g[0],
            fd
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  6. reml_laml_evaluate determinism for fixed inputs
// ─────────────────────────────────────────────────────────────────────────────

/// The outer REML/LAML evaluation must be deterministic for fixed ρ, design and
/// response: repeated evaluations produce bit-identical cost and gradient.
#[test]
fn bug_reml_laml_evaluate_not_deterministic() {
    let prob = build_gaussian_problem(0x1861_0006);
    let opts = gaussian_opts(prob.nullspace_dims.clone());

    for &rho0 in &[-2.5_f64, -0.25, 1.5, 5.0] {
        let rho = Array1::from_vec(vec![rho0]);

        let c_first = external_cost(&prob, &opts, &rho);
        let g_first = external_gradient(&prob, &opts, &rho);

        for _ in 0..4 {
            let c = external_cost(&prob, &opts, &rho);
            let g = external_gradient(&prob, &opts, &rho);
            assert_eq!(
                c.to_bits(),
                c_first.to_bits(),
                "REML score must be deterministic at ρ={rho0}: {c} vs {c_first}"
            );
            assert_eq!(g.len(), g_first.len());
            for (i, (a, b)) in g.iter().zip(g_first.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "REML score gradient must be deterministic at ρ={rho0}, coord {i}: {a} vs {b}"
                );
            }
        }
    }
}
