//! Owed-work regression gate for GitHub issue #1418.
//!
//! ISSUE: the implicit-function (IFT) correction in the SAE outer-ρ REML
//! gradient is `−½·Γᵀ·θ̂_ρ` with the sensitivity `θ̂_ρ = −A⁻¹ ∂g/∂ρ`, where
//! `A = ∇²_θθ L` is the EXACT stationarity Jacobian of the inner fit. The matrix
//! the inner solve actually factors is a stability-conditioned surrogate `B`
//! (Gauss-Newton data curvature with the residual-curvature term `⟨r, ∂²f⟩`
//! dropped; the softmax entropy Hessian replaced by its Gershgorin/Fisher PSD
//! majorizer; the periodic ARD curvature `V''` replaced by `max(V'',0)`). The
//! `½log|B|` Laplace value term is consistent with `Γ = ½tr(B⁻¹ ∂B/∂θ)`, but the
//! implicit STEP must use the exact `A`, not `B`. Using `B` for the IFT solve
//! biases the correction by `(B⁻¹ − A⁻¹)`, which is nonzero exactly when the
//! dropped curvature `ΔC = A − B` is nonzero (large residual, indefinite entropy,
//! periodic ARD past a quarter period).
//!
//! FIX (landed in `analytic_outer_rho_gradient_components`,
//! `solve_exact_stationarity`, `apply_exact_hessian_minus_b` in
//! `src/terms/sae/manifold/construction.rs`): the IFT correction now applies the
//! TRUE `A⁻¹` via a `B⁻¹`-preconditioned Neumann fixed point
//! (`A = B + ΔC`, `x = B⁻¹ rhs − B⁻¹(ΔC x)`), with `ΔC` assembled matrix-free
//! over all three dropped channels. `B` survives only as the preconditioner.
//!
//! CERTIFICATE (public API only, survives refactors of the private solver): the
//! full analytic outer-ρ gradient — explicit + direct log-det traces + Occam +
//! the implicit-state correction — must match a centered finite difference of the
//! actual custom quasi-Laplace criterion (the inner problem is re-solved at each perturbed ρ, so
//! the FD carries the true envelope/IFT terms governed by `A`). The fixture is
//! deliberately built with a LARGE, unmodellable residual at the inner optimum on
//! a curved (periodic-harmonic) basis, so the dropped residual curvature
//! `⟨r, ∂²f⟩` is genuinely large: were the implicit step still using `B`, the
//! analytic gradient would deviate from the FD by `O(‖B⁻¹ − A⁻¹‖)` and the bound
//! below would fail.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::solver::arrow_schur::ArrowFactorCache;
use gam::terms::{
    latent::LatentManifold, sae::manifold::AssignmentMode,
    sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment,
    sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldLoss, sae::manifold::SaeManifoldRho,
    sae::manifold::SaeManifoldTerm,
};

struct Fixture {
    term: SaeManifoldTerm,
    target: Array2<f64>,
    rho: SaeManifoldRho,
}

/// K=2 periodic-harmonic SAE fixture with a LARGE, deliberately unmodellable
/// residual at the inner optimum.
///
/// The decoders are set to a low-rank, smooth shape while the target carries a
/// strong high-frequency / cross-channel component the K=2 periodic basis CANNOT
/// represent, so the inner fit stops at a stationary point with a large residual
/// `r`. Because the periodic-harmonic basis has nonzero second jets `∂²f/∂t²`,
/// the dropped residual-curvature term `ΔC_tt = ⟨r, ∂²f⟩` is then large — this is
/// precisely the regime in which `θ̂_ρ` computed with `B` instead of the exact `A`
/// is visibly wrong. Softmax sparsity is active (so the indefinite entropy
/// curvature delta is live) and the latent coordinates sweep a full period (so
/// the periodic ARD curvature `V''` is negative on part of the support, making
/// the `min(V'',0)` channel of `ΔC` nonzero too).
fn high_residual_curvature_fixture(mode: AssignmentMode, log_lambda_sparse: f64) -> Fixture {
    let n = 96usize;
    let p = 6usize;
    let k_atoms = 2usize;
    let m = 5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator");

    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));

    // Smooth, low-rank decoders the basis CAN represent (these set the modelled
    // part); the target adds an unmodellable high-frequency term on top.
    let decoder_weight = |b: usize, col: usize, atom: usize| -> f64 {
        0.12 + 0.04 * (b as f64) - 0.03 * (col as f64) + 0.05 * (atom as f64)
    };

    for row in 0..n {
        let phase = (row as f64 + 0.5) / n as f64; // sweeps a full period
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.31).fract();
        // Both gates genuinely active so softmax sparsity / entropy curvature is
        // live and neither atom is degenerate.
        let route = if row % 2 == 0 { 1.1 } else { -1.1 };
        logits[[row, 0]] = route;
        logits[[row, 1]] = if row % 3 == 0 { 0.8 } else { 0.4 };

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
                v0 += basis0[b] * decoder_weight(b, col, 0);
                v1 += basis1[b] * decoder_weight(b, col, 1);
            }
            let modelled = mix0 * v0 + mix1 * v1;
            // Large unmodellable component: a high harmonic (period m beyond the
            // basis order) plus a deterministic per-(row,col) jitter. This forces
            // a big residual at the inner optimum, hence a large ⟨r, ∂²f⟩.
            let high = 0.9 * (5.0 * theta0 + 0.7 * (col as f64)).sin()
                + 0.6 * (4.0 * theta1).cos()
                + 0.25 * (((row * 7 + col * 13) % 11) as f64 - 5.0);
            target[[row, col]] = modelled + high;
        }
    }

    let mut atoms = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom_idx].view())
            .expect("periodic basis evaluation");
        let decoder = Array2::from_shape_fn((m, p), |(b, col)| decoder_weight(b, col, atom_idx));
        // Penalize the harmonic coefficients (leave the constant column free) so a
        // genuine λ_smooth gradient channel exists.
        let mut smooth = Array2::<f64>::eye(m);
        smooth[[0, 0]] = 0.0;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    // log_lambda_smooth and log_ard set to genuinely active (small) levels so the
    // periodic ARD curvature is nonzero and the `min(V'',0)` ΔC channel is live.
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -2.0,
        vec![Array1::from_vec(vec![-1.0]), Array1::from_vec(vec![-1.0])],
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
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            inner_max_iter,
            0.45,
            1.0e-6,
            1.0e-6,
        )
        .unwrap_or_else(|err| panic!("quasi-Laplace criterion failed: {err}"));
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
    let rho_plus = template.from_flat(plus.view()).unwrap();
    let rho_minus = template.from_flat(minus.view()).unwrap();
    let (_, vp, _, _) = evaluate(start, target, &rho_plus, inner_max_iter);
    let (_, vm, _, _) = evaluate(start, target, &rho_minus, inner_max_iter);
    (vp - vm) / (2.0 * h)
}

/// The full analytic outer-ρ gradient must match a centered finite difference of
/// the actual re-solved quasi-Laplace criterion. The FD is the ground truth for the IFT
/// step: it differentiates the value through the inner re-solve, so it carries
/// the exact stationarity Jacobian `A`. The analytic path matches it ONLY if its
/// implicit correction also uses `A` (the #1418 fix) and not the surrogate `B`.
fn assert_full_gradient_matches_fd(label: &str, f: &Fixture) {
    let inner_iters = 12usize;
    let (converged, _value, loss, cache) = evaluate(&f.term, &f.target, &f.rho, inner_iters);
    let components = converged
        .analytic_outer_rho_gradient_at_converged(f.target.view(), &f.rho, &loss, &cache)
        .expect("analytic components");
    let analytic = components.gradient();
    let n_params = f.rho.to_flat().len();

    // Sanity: the fixture must actually carry large residual curvature, otherwise
    // it would not distinguish A from B and the certificate would be vacuous. The
    // data-fit loss at the inner optimum is far from zero by construction.
    assert!(
        loss.data_fit > 1.0,
        "[{label}] fixture is too well-fit (data_fit {:.3e}); residual curvature \
         ΔC=⟨r,∂²f⟩ would be negligible and the A-vs-B test vacuous",
        loss.data_fit
    );

    for coord in 0..n_params {
        let fd = centered_fd(&converged, &f.target, &f.rho, coord, inner_iters);
        let diff = (fd - analytic[coord]).abs();
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic[coord].abs()));
        assert!(
            diff <= tol,
            "[{label}] full rho gradient coord {coord}: fd={fd:.8e}, analytic={:.8e}, \
             diff={diff:.3e}, tol={tol:.3e} — implicit step likely using surrogate B \
             instead of exact stationarity Jacobian A (#1418)",
            analytic[coord]
        );
    }
}

/// #1418 (softmax): the entropy curvature delta (`H_entropy − D`) and the
/// residual-curvature delta (`⟨r, ∂²f⟩`) are both large and nonzero here. The
/// analytic outer gradient must still track the re-solved FD, proving the IFT
/// step inverts the exact `A` rather than the assembled majorizer `B`.
#[test]
fn sae_ift_uses_exact_stationarity_jacobian_softmax_high_residual_1418() {
    let f = high_residual_curvature_fixture(AssignmentMode::softmax(0.7), -1.0);
    assert_full_gradient_matches_fd("softmax_high_residual", &f);
}

/// #1418 (ThresholdGate): isolates the residual-curvature channel of `ΔC` (no entropy
/// majorizer in gate modes), so this arm is a clean test that the dropped
/// Gauss-Newton residual curvature `⟨r, ∂²f⟩` is restored in the implicit step.
#[test]
fn sae_ift_uses_exact_stationarity_jacobian_threshold_gate_high_residual_1418() {
    let f = high_residual_curvature_fixture(AssignmentMode::threshold_gate(0.7, 0.0), -1.0);
    assert_full_gradient_matches_fd("threshold_gate_high_residual", &f);
}
