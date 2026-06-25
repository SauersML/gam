//! Owed-work regression for #1416 — the IBP log-determinant ρ-trace must
//! contract the FULL cross-row off-diagonal of the rank-one Woodbury source,
//! not only the diagonal.
//!
//! ## The defect (now fixed)
//!
//! The IBP prior Hessian for one column `k` is
//!
//! ```text
//!   H_p = d·J Jᵀ + diag(s, c),
//! ```
//!
//! where `J_i = ∂z_i/∂ℓ_i` (`z_jac`), `d = ∂s/∂M` (`cross_row_d`) is the scalar
//! coefficient of the per-column rank-one block, and the empirical mass
//! `M_k = Σ_i z_ik` couples EVERY row pair `(i, j)`. The solver installs the
//! full rank-one `d·J Jᵀ` via Woodbury, so for fixed alpha the entire IBP prior
//! scales with `λ_sparse = eᵖ` and the correct direct log-det contribution to
//! the outer-ρ gradient is the FULL trace `½ tr(H⁻¹ ∂H_p/∂ρ_sparse)` — diagonal
//! AND the off-diagonal rank-one cross terms
//! `½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j`. The pre-fix non-softmax ρ-trace branch
//! contracted only the diagonal, dropping those cross terms.
//!
//! ## What this test pins — against the REAL production trace
//!
//! The production cross-row contraction lives on
//! `SaeManifoldTerm::assignment_log_strength_hessian_trace` (a `pub(crate)` fn,
//! not reachable from an external integration test). Rather than re-derive a
//! 2×2 trace identity in the test — which would be a TAUTOLOGY that passes even
//! if the production trace dropped the off-diagonal — this test drives the FULL
//! production outer-ρ gradient via the PUBLIC
//! `analytic_outer_rho_gradient_at_converged` on a fixed-α IBP-MAP SAE term and
//! pins it against a CENTERED FINITE DIFFERENCE of the actual re-solved REML
//! criterion. Coordinate 0 (`log_lambda_sparse`) drives the IBP prior strength,
//! so its analytic gradient IS the `½ ∂log|H|/∂ρ_sparse` cross-row trace; the FD
//! re-solves the inner problem at each perturbed ρ, so it carries the TRUE full
//! trace (diagonal + cross-row off-diagonal). A diagonal-only contraction (the
//! #1416 bug) makes coord 0 disagree with this FD and fails.
//!
//! The fixture deliberately activates a genuine cross-row Woodbury source: a
//! fixed-α IBP-MAP gate with both atoms live so the empirical mass `M_k` couples
//! rows and `cross_row_d ≠ 0`. A vacuity guard asserts the fixture is not
//! degenerately well-fit (so the trace term is non-negligible).
//!
//! This complements the in-crate dense-FD guard
//! `terms::sae::manifold::tests::ibp_rho_sparse_logdet_trace_matches_dense_fd_1416`
//! (which exercises `assignment_log_strength_hessian_trace` directly through the
//! crate-internal fixture) with a standalone gate depending only on the public
//! crate API.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

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

/// K=2 periodic-harmonic SAE fixture under a fixed-α IBP-MAP gate. Both atoms
/// are genuinely active so the empirical mass `M_k = Σ_i z_ik` couples the rows
/// and the per-column cross-row Woodbury coefficient `cross_row_d` is nonzero —
/// the exact source whose off-diagonal the #1416 ρ-trace must contract.
fn ibp_cross_row_fixture(log_lambda_sparse: f64) -> Fixture {
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
        // Both gates active on a meaningful fraction of rows so the IBP mass
        // M_k couples rows on BOTH columns (a live cross-row Woodbury source on
        // each column, not just one).
        logits[[row, 0]] = if row % 2 == 0 { 1.3 } else { 0.5 };
        logits[[row, 1]] = if row % 3 == 0 { 1.1 } else { 0.4 };
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
        // A genuine residual the K=2 basis cannot fully represent, so the inner
        // fit stops at a stationary point with nonzero data-fit (the cross-row
        // trace is then non-negligible — see the vacuity guard).
        let high = 0.6 * (4.0 * theta0).sin() + 0.4 * (3.0 * theta1).cos();
        for col in 0..p {
            let mut v0 = 0.0;
            let mut v1 = 0.0;
            for b in 0..m {
                v0 += basis0[b] * weights0[b][col];
                v1 += basis1[b] * weights1[b][col];
            }
            target[[row, col]] = 0.5 * v0 + 0.5 * v1 + high;
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

    // FIXED-α IBP-MAP (`learnable_alpha = false`): coordinate 0 of the outer-ρ
    // vector is `log_lambda_sparse`, the IBP prior STRENGTH whose Hessian is the
    // cross-row Woodbury source. (A learnable-α gate would route coord 0 to
    // log-α instead — that is the #1417 path; here we want the prior-strength
    // ρ_sparse trace, the #1416 site.)
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        AssignmentMode::ibp_map(0.7, 0.9, false),
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

/// #1416: the full production outer-ρ gradient — whose coordinate 0
/// (`log_lambda_sparse`) is the IBP `½ ∂log|H|/∂ρ_sparse` cross-row trace — must
/// match a centered finite difference of the re-solved REML criterion at EVERY
/// coordinate. The FD differentiates the value through the inner re-solve, so it
/// carries the TRUE `½ tr(H⁻¹ ∂H_p/∂ρ_sparse)` (diagonal AND the rank-one
/// off-diagonal `½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j`). The diagonal-only contraction
/// (the #1416 bug) drops the off-diagonal and makes coord 0 disagree with the FD.
#[test]
fn ibp_rho_sparse_logdet_trace_includes_cross_row_offdiagonal_1416() {
    let f = ibp_cross_row_fixture(-1.0);
    let inner_iters = 8usize;
    let (converged, _value, loss, cache) = evaluate(&f.term, &f.target, &f.rho, inner_iters);

    // Vacuity guard: the fixture must carry a genuine (non-degenerate) data fit,
    // otherwise the cross-row Woodbury coefficient and the trace it feeds would
    // be negligible and the off-diagonal could not be distinguished from zero.
    assert!(
        loss.data_fit > 1.0e-2,
        "fixture is too well-fit (data_fit {:.3e}); the IBP cross-row trace would be \
         negligible and the diagonal-vs-full distinction vacuous",
        loss.data_fit
    );

    let components = converged
        .analytic_outer_rho_gradient_at_converged(f.target.view(), &f.rho, &loss, &cache)
        .expect("analytic outer-rho gradient components");
    let analytic = components.gradient();
    let n_params = f.rho.to_flat().len();
    assert!(
        n_params >= 1,
        "outer-rho vector must carry log_lambda_sparse at coord 0"
    );

    for coord in 0..n_params {
        let fd = centered_fd(&converged, &f.target, &f.rho, coord, inner_iters);
        let diff = (fd - analytic[coord]).abs();
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic[coord].abs()));
        assert!(
            diff <= tol,
            "[#1416] outer-rho gradient coord {coord}: fd={fd:.8e}, analytic={:.8e}, \
             diff={diff:.3e}, tol={tol:.3e}. Coord 0 is the IBP ½∂log|H|/∂ρ_sparse \
             cross-row trace — a mismatch there means the rank-one off-diagonal was \
             dropped (the diagonal-only #1416 bug).",
            analytic[coord]
        );
    }
}
