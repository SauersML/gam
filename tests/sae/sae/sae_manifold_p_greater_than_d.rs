//! Regression test for issue #393: `sae_manifold_fit` panics with `index out
//! of bounds` whenever the decoder output dimension `p` exceeds the per-row
//! tangent block dimension `d = K + Σ_k d_atom_k`.
//!
//! Pre-fix, the matrix-free `H_tβ^(row)` operator installed by
//! `SaeManifoldTerm::assemble_arrow_schur_scaled` wired `apply_jbeta` (which
//! writes `p` entries into its output) directly as the `forward` closure of
//! `ArrowSchurSystem::set_row_htbeta_operator` (whose contract requires
//! `out.len() == d`). When `p > d` the closure walked past the end of the
//! length-`d` buffer and the process aborted. The transpose closure had the
//! mirror defect — it expected its input `v` to be a length-`p` vector but
//! the solver passes a length-`d` tangent residual.
//!
//! The fix factors `H_tβ = L · J_β` correctly: forward applies `apply_jbeta`
//! into a p-vector scratch and then `apply_l` into the d-length output;
//! transpose applies `apply_l_t` into the p-vector scratch and then
//! `scatter_jbeta_t` to the K-length β accumulator.
//!
//! This test exercises the exact configuration from the issue repro:
//! K=1 circle atom (d_atom=1), so the per-row tangent dim is
//! `1 (assign) + 1 (coord) = 2`, with `p = 5 > d = 2`. It asserts that the
//! joint Newton solve runs to completion and produces a finite fit.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::terms::{
    latent::LatentManifold, sae::manifold::AssignmentMode,
    sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment,
    sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};

/// Issue #393: `p > d` must not panic. Single circle atom (K=1, d_atom=1),
/// `p = 5` decoder columns; pre-fix this triggered an index-out-of-bounds
/// abort in `SaeKroneckerRows::apply_jbeta` writing index 2 into a length-2
/// buffer at the first Schur matvec.
#[test]
fn sae_manifold_p_greater_than_d_does_not_panic() {
    let n = 200usize;
    let p = 5usize;
    let m_basis = 7usize;

    // Synthesise a circle-supported signal: F = [cos(θ), sin(θ)], Z = F · W + ε.
    let mut z = Array2::<f64>::zeros((n, p));
    let mut true_phase = Array2::<f64>::zeros((n, 1));
    // Deterministic Wᵀ ∈ ℝ^{2×p}; arbitrary, but fixed.
    let w_loadings: [[f64; 5]; 2] = [[0.7, -0.3, 0.4, 0.1, -0.5], [0.2, 0.6, -0.4, 0.3, 0.1]];
    for i in 0..n {
        let t = (i as f64 + 0.5) / (n as f64); // ∈ (0, 1)
        true_phase[[i, 0]] = t;
        let a = std::f64::consts::TAU * t;
        let f0 = a.cos();
        let f1 = a.sin();
        for j in 0..p {
            z[[i, j]] = f0 * w_loadings[0][j] + f1 * w_loadings[1][j];
        }
    }

    // Build a single Periodic atom (Circle on [0, 1)) with p = 5 output columns.
    let evaluator = PeriodicHarmonicEvaluator::new(m_basis).expect("periodic evaluator");
    let (phi0, jet0) = evaluator
        .evaluate(true_phase.view())
        .expect("periodic atom evaluation");
    let mut penalty = Array2::<f64>::eye(m_basis);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new(
        "circle_atom",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        Array2::<f64>::zeros((m_basis, p)),
        penalty,
    )
    .expect("circle atom")
    .with_basis_evaluator(Arc::new(
        PeriodicHarmonicEvaluator::new(m_basis).expect("periodic evaluator clone"),
    ) as Arc<dyn SaeBasisEvaluator>);

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![true_phase],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.5),
    )
    .expect("assignment construction");

    let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("term construction");
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(1)]);
    let ridge = 1.0e-6;

    // 20 Newton iterations is more than enough; pre-fix the first step aborts.
    let mut last_total = f64::INFINITY;
    for _ in 0..20 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .expect("Newton step");
        last_total = loss.total();
        if !last_total.is_finite() {
            break;
        }
    }

    assert!(
        last_total.is_finite(),
        "issue #393: SAE Newton loss diverged at p>d (n={n}, p={p}): loss={last_total}"
    );
    let fitted = term.fitted();
    assert_eq!(fitted.dim(), (n, p));
    for ((row, col), v) in fitted.indexed_iter() {
        assert!(
            v.is_finite(),
            "issue #393: fitted[{row},{col}] = {v} is non-finite"
        );
    }
}
