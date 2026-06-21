//! Regression for #1417 — learnable IBP alpha must include the explicit
//! data-Hessian log-determinant derivative in the outer-ρ gradient.
//!
//! ## The bug
//!
//! With a LEARNABLE IBP alpha the forward assignment is
//!   `a_ik = σ(ℓ_ik/τ) · π_k(α)`,   `π_k(α) = (α/(α+1))^k`,
//! so the reconstruction Gauss–Newton matrix depends EXPLICITLY on `α`: every
//! data-Jacobian column for atom `k` carries one `π_k`/`a_k` factor, hence
//!   `∂H_data[a,b]/∂logα = ((k_a + k_b)/(α+1)) · H_data[a,b] ≠ 0`  for `k>0`.
//! The outer `½log|H|` evidence term therefore has a non-zero `α`-derivative
//!   `½ tr(H⁻¹ ∂H_data/∂logα)`
//! through the data blocks `H_ββ`, `H_tβ`, `H_tt`. The original analytic
//! gradient populated `logdet_trace[0]` from the assignment-PRIOR Hessian trace
//! only; the data-block term was missing, so the learnable-α outer gradient was
//! incomplete at coordinate 0 (log-α).
//!
//! ## The certificate
//!
//! For a learnable-α IBP model `resolved_α = base_α · exp(rho.log_lambda_sparse)`
//! (`resolve_learnable_weight`), so `∂logα/∂(coord 0) = 1`: coordinate 0 of the
//! outer-ρ gradient IS the log-α derivative. The FULL analytic gradient
//! (explicit + direct logdet traces + Occam + #1006 implicit correction) must
//! match a CENTERED finite difference of the actual REML criterion with the
//! inner problem re-solved at each ρ. The FD therefore carries the true
//! `½ tr(H⁻¹ ∂H/∂logα)` total — prior AND data — at coord 0. Dropping the
//! data-block term makes the analytic coord-0 gradient disagree with the FD;
//! with the #1417 fix (`learnable_ibp_data_logdet_alpha_trace`) it agrees.
//!
//! A FIXED-α IBP control arm asserts the data term is identically zero there:
//! coord 0 of a fixed-α model is `log λ_sparse` (prior weight), not log-α, and
//! its gradient is unchanged.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::solver::arrow_schur::ArrowFactorCache;
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldLoss, SaeManifoldRho, SaeManifoldTerm,
};

struct Fixture {
    term: SaeManifoldTerm,
    target: Array2<f64>,
    rho: SaeManifoldRho,
}

/// K=2 periodic-harmonic SAE fixture under a given assignment `mode`.
///
/// `log_lambda_sparse` is coordinate 0 of the outer-ρ vector. For a LEARNABLE
/// IBP `mode` it is the log-α lever (`resolved_α = base_α·exp(coord0)`), so the
/// coord-0 FD stresses exactly the #1417 data-Hessian-logdet-α term.
fn fixture(mode: AssignmentMode, log_lambda_sparse: f64) -> Fixture {
    let n = 80usize;
    let p = 6usize;
    let k_atoms = 2usize;
    let m = 5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator");

    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));
    let weights0 = [
        [0.20, -0.10, 0.06, 0.03, -0.04, 0.08],
        [0.70, -0.25, 0.40, 0.12, -0.35, 0.18],
        [0.15, 0.55, -0.25, 0.28, 0.08, -0.22],
        [0.08, -0.04, 0.03, -0.02, 0.01, 0.06],
        [-0.06, 0.02, 0.05, 0.04, -0.03, 0.01],
    ];
    let weights1 = [
        [-0.10, 0.05, 0.08, -0.02, 0.05, -0.03],
        [-0.30, 0.42, 0.12, -0.20, 0.16, 0.30],
        [0.48, 0.10, -0.32, 0.18, 0.26, -0.14],
        [0.04, 0.07, -0.02, 0.03, -0.05, 0.02],
        [0.03, -0.05, 0.04, 0.01, 0.02, -0.04],
    ];

    for row in 0..n {
        let phase = (row as f64 + 0.25) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.18).fract();
        let route = if row < n / 2 { 1.7 } else { -1.7 };
        logits[[row, 0]] = route;
        // A genuine second active gate so the per-atom IBP logits all sit inside
        // their optimization band (the k>0 prior factor `π_k(α)` is then live).
        logits[[row, 1]] = if row % 3 == 0 { 0.9 } else { 0.3 };
        let theta0 = std::f64::consts::TAU * coords[0][[row, 0]];
        let theta1 = std::f64::consts::TAU * coords[1][[row, 0]];
        let basis0 = [
            1.0,
            theta0.sin(),
            theta0.cos(),
            (2.0 * theta0).sin(),
            (2.0 * theta0).cos(),
        ];
        let basis1 = [
            1.0,
            theta1.sin(),
            theta1.cos(),
            (2.0 * theta1).sin(),
            (2.0 * theta1).cos(),
        ];
        let mix0 = 1.0 / (1.0 + (-route / 0.7).exp());
        let mix1 = 1.0 - mix0;
        for col in 0..p {
            let mut v0 = 0.0;
            let mut v1 = 0.0;
            for b in 0..m {
                v0 += basis0[b] * weights0[b][col];
                v1 += basis1[b] * weights1[b][col];
            }
            target[[row, col]] = mix0 * v0 + mix1 * v1;
        }
    }

    let mut atoms = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom_idx].view())
            .expect("periodic basis evaluation");
        let decoder = if atom_idx == 0 {
            Array2::from_shape_fn((m, p), |(row, col)| weights0[row][col])
        } else {
            Array2::from_shape_fn((m, p), |(row, col)| weights1[row][col])
        };
        let mut smooth = Array2::<f64>::eye(m);
        smooth[[0, 0]] = 0.0;
        let atom = SaeManifoldAtom::new(
            format!("circle_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            smooth,
        )
        .expect("circle atom")
        .with_basis_evaluator(Arc::new(
            PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator clone"),
        ) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -8.0,
        vec![Array1::from_vec(vec![-8.0]), Array1::from_vec(vec![-8.0])],
    );
    Fixture { term, target, rho }
}

fn evaluate(
    start: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    inner_max_iter: usize,
) -> (SaeManifoldTerm, f64, SaeManifoldLoss, ArrowFactorCache) {
    let mut term = start.clone();
    let (value, loss, cache) = term
        .reml_criterion_with_cache(
            target.view(),
            rho,
            None,
            inner_max_iter,
            0.45,
            1.0e-6,
            1.0e-6,
        )
        .unwrap_or_else(|err| panic!("REML criterion failed: {err}"));
    (term, value, loss, cache)
}

fn centered_fd(
    start: &SaeManifoldTerm,
    target: &Array2<f64>,
    template: &SaeManifoldRho,
    coord: usize,
    inner_max_iter: usize,
) -> f64 {
    let h = 2.0e-4;
    let mut plus = template.to_flat();
    let mut minus = template.to_flat();
    plus[coord] += h;
    minus[coord] -= h;
    let rho_plus = template.from_flat(plus.view());
    let rho_minus = template.from_flat(minus.view());
    let (_, vp, _, _) = evaluate(start, target, &rho_plus, inner_max_iter);
    let (_, vm, _, _) = evaluate(start, target, &rho_minus, inner_max_iter);
    (vp - vm) / (2.0 * h)
}

/// The full analytic outer-ρ gradient must match a centered finite difference of
/// the re-solved REML criterion at EVERY coordinate (the FD carries the
/// envelope/IFT terms). For learnable α this asserts coord 0 (= log-α) includes
/// the #1417 data-Hessian-logdet-α contribution.
fn assert_full_gradient_matches_fd(label: &str, f: &Fixture) {
    let (converged, _value, loss, cache) = evaluate(&f.term, &f.target, &f.rho, 8);
    let components = converged
        .analytic_outer_rho_gradient_at_converged(f.target.view(), &f.rho, &loss, &cache)
        .expect("analytic components");
    let analytic = components.gradient();
    let n_params = f.rho.to_flat().len();

    for coord in 0..n_params {
        let fd = centered_fd(&converged, &f.target, &f.rho, coord, 8);
        let diff = (fd - analytic[coord]).abs();
        let tol = 2.5e-3 * (1.0 + fd.abs().max(analytic[coord].abs()));
        assert!(
            diff <= tol,
            "[{label}] full rho gradient coord {coord}: fd={fd:.8e}, analytic={:.8e}, diff={diff:.3e}, tol={tol:.3e}",
            analytic[coord]
        );
    }
}

/// #1417 PRIMARY: a LEARNABLE-α IBP model. Coordinate 0 is log-α, so the coord-0
/// centered FD of the re-solved criterion carries the full
/// `½ tr(H⁻¹ ∂H/∂logα)` — assignment-prior AND data-Hessian blocks. Without the
/// `learnable_ibp_data_logdet_alpha_trace` term the analytic coord-0 gradient
/// would omit the data-block contribution and disagree with this FD.
#[test]
fn learnable_alpha_outer_gradient_includes_data_hessian_logdet_alpha_1417() {
    let f = fixture(AssignmentMode::ibp_map(0.7, 0.9, true), -1.5);
    assert_full_gradient_matches_fd("ibp_map_learnable_alpha", &f);
}

/// #1417 CONTROL: a FIXED-α IBP model. Here coordinate 0 is `log λ_sparse` (the
/// prior weight), NOT log-α, so the data-Hessian-α term is identically zero and
/// the fixed-α gradient is unchanged. The same all-coordinate FD certificate
/// must still hold — i.e. the fix did not perturb the fixed-α path.
#[test]
fn fixed_alpha_outer_gradient_unchanged_control_1417() {
    let f = fixture(AssignmentMode::ibp_map(0.7, 0.9, false), -1.5);
    assert_full_gradient_matches_fd("ibp_map_fixed_alpha", &f);
}
