//! #1782 — non-PD reduced-Schur seed-refusal classification tests, split out of
//! `tests.rs` to keep that file under the 10k-line ban gate. Uses the crate-level
//! manifold items via `super::*` and the shared test builders (e.g.
//! `planted_circle_embedded`) via `super::tests::*`.

use super::tests::*;
use super::*;
use gam_solve::arrow_schur::ArrowSchurError;
use ndarray::array;

/// #1782 — the exact production string emitted when a seed ρ's off-optimum
/// inner state leaves the reduced joint-Hessian Schur complement indefinite:
/// `run_joint_fit_arrow_schur` → `converge_inner_for_undamped_logdet` bubbles up
/// `ArrowSchurError::SchurFactorFailed { reason }` whose `Display` is
/// "arrow-Schur: Schur complement Cholesky failed: <reason>", wrapped by the
/// criterion's own `format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}")`.
/// The `reason` for a non-PD pivot is
/// "non-PD pivot <v> at index <i> (matrix is not positive definite)".
fn schur_non_pd_seed_refusal_message() -> String {
    format!(
        "SaeManifoldTerm::run_joint_fit_arrow_schur: {}",
        ArrowSchurError::SchurFactorFailed {
            reason: "non-PD pivot -2.5e-09 at index 2 (matrix is not positive definite)"
                .to_string(),
        }
    )
}

/// #1782 regression — at a seed ρ, a K>1 `jumprelu`/`softmax` (or a
/// rank-deficient `euclidean`/`linear`) fit's OFF-OPTIMUM inner state can leave
/// the reduced joint-Hessian Schur complement indefinite, so the undamped
/// Laplace factorization refuses with an `ArrowSchurError::SchurFactorFailed`
/// ("Schur complement Cholesky failed: … not positive definite"). Before the fix
/// `is_recoverable_value_probe_refusal` did NOT classify that message as a
/// recoverable infeasible-ρ probe, so it propagated as a hard
/// `RemlOptimizationFailed`; with a single PCA seed the fixed-point seed-startup
/// validation then rejected every candidate → "no candidate seeds passed outer
/// startup validation (SAE manifold)". `ibp_map`+`circle`'s seed lands in the PD
/// region and never trips it — exactly why it converged on identical data while
/// the other assignments/topologies did not.
///
/// The fix classifies the non-PD Schur-complement refusal as recoverable
/// (requiring BOTH the "Schur complement Cholesky failed" and "not positive
/// definite" markers, so genuine shape/dimension/non-finite Schur defects stay
/// fatal). This test pins the classifier against the EXACT production message —
/// RED before the fix (the assertion below fails), GREEN after.
#[test]
pub(crate) fn non_pd_schur_seed_refusal_is_recoverable_1782() {
    let msg = schur_non_pd_seed_refusal_message();
    // Sanity: the constructed message carries both required markers.
    assert!(
        msg.contains("Schur complement Cholesky failed") && msg.contains("not positive definite"),
        "fixture message must carry both non-PD Schur markers: {msg}"
    );
    // The load-bearing #1782 assertion: the non-PD Schur seed refusal is a
    // recoverable infeasible-ρ probe (the outer lanes return +∞ / a finite
    // collapse wall and steer back into the PD region), NOT a fatal error that
    // empties the seed cascade.
    assert!(
        SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(&msg),
        "the non-PD reduced-Schur seed refusal must be recoverable (#1782); got fatal for: {msg}"
    );

    // Guard against over-broadening: a `SchurFactorFailed` whose reason is NOT a
    // non-PD pivot (a genuine shape / non-finite Schur defect, lacking the "not
    // positive definite" marker) must STILL be fatal, so the classifier does not
    // silently mask real factorization bugs as recoverable probes.
    let genuine_defect = format!(
        "SaeManifoldTerm::run_joint_fit_arrow_schur: {}",
        ArrowSchurError::SchurFactorFailed {
            reason: "non-finite entry at linear index 3".to_string(),
        }
    );
    assert!(
        genuine_defect.contains("Schur complement Cholesky failed"),
        "fixture must exercise the Schur-failure surface: {genuine_defect}"
    );
    assert!(
        !SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(&genuine_defect),
        "a non-PD-marker-free Schur factorization defect must stay FATAL (#1782 must not \
         weaken validation into accepting genuine defects): {genuine_defect}"
    );
}

/// #1782 — the fixed-point seed-startup VALIDATION lane (`eval_efs` → `efs_step`)
/// is the lane SAE seeds are validated on (`fixed_point_available == true` →
/// `Solver::Efs`; `run_fixed_point_outer_solver` rejects the seed when
/// `eval_step` errors). Drive a K>1 `jumprelu` euclidean term through the full
/// outer cascade on planted-circle data and assert it does NOT die with the
/// "no candidate seeds passed outer startup validation" abort — i.e. at least one
/// seed passes startup validation and the fit runs to a result. Uses the same
/// single-PCA-seed budget the production `sae_manifold_fit` FFI entry uses, so an
/// infeasible-ρ seed refusal would empty the candidate set exactly as in the
/// issue.
#[test]
pub(crate) fn planted_circle_multi_atom_jumprelu_clears_startup_validation_1782() {
    use gam_solve::rho_optimizer::OuterProblem;
    use gam_solve::seeding::SeedConfig;

    let n = 40usize;
    let k_atoms = 3usize;
    let z = planted_circle_embedded(n, 6, 0.03);
    let p = z.ncols();
    // K>1 euclidean (constant+linear) atoms sharing one 1-D coordinate, with a
    // jumprelu (threshold-gate) assignment — the non-ibp assignment / non-circle
    // topology combination the issue reports as failing.
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (row as f64 / n as f64) - 0.5);
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 1).unwrap());
    let (phi0, jet0) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi0.ncols();
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let mut decoder = Array2::<f64>::zeros((m, p));
        // Distinct-but-overlapping decoder directions across atoms so the joint
        // decoder design is near-rank-deficient (the K>1 co-collapse regime that
        // drives the reduced Schur toward indefiniteness at the off-optimum seed).
        for col in 0..p {
            decoder[[m - 1, col]] = 0.1 + 0.02 * ((atom_idx + col) % 3) as f64;
        }
        atoms.push(
            SaeManifoldAtom::new(
                format!("euclid{atom_idx}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi0.clone(),
                jet0.clone(),
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_evaluator(evaluator.clone()),
        );
        coord_blocks.push(coords.clone());
        manifolds.push(LatentManifold::Euclidean);
    }
    let assignment_mode = AssignmentMode::jumprelu(1.0, 0.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n, k_atoms), 0.5),
        coord_blocks,
        manifolds,
        assignment_mode,
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();

    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k_atoms])
        .seed_scaled_by_dispersion_for_assignment(1.0, assignment_mode)
        .unwrap();
    let init_rho_flat = init_rho.to_flat();
    let n_params = init_rho_flat.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_initial_rho(init_rho_flat)
        .with_seed_config(SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold");
    if let Err(err) = &result {
        let msg = err.to_string();
        assert!(
            !msg.contains("no candidate seeds passed outer startup validation"),
            "#1782: K>1 jumprelu euclidean fit must not abort with an emptied seed cascade; got: {msg}"
        );
    }
    // The fit produced a result (converged or best-so-far); the startup
    // validation accepted the seed, which is the #1782 contract.
    result.expect("#1782: multi-atom jumprelu fit must run to a result, not a seed-cascade abort");
}
