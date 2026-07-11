//! Deflation/PD-region log-det-trace regression tests (#1026/#1590),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet` are sourced from the sibling
//! `tests` module.

use super::tests::{fixed_state_logdet, gamma_fd_tiny_fixture};
use super::*;

/// Deflation-derivative regression for a NON-α ρ-component. The deflation that
/// the ordered Beta--Bernoulli-prior negative curvature triggers stiffens the WHOLE per-row `H_tt`
/// block (logit AND coordinate slots), so it corrupts EVERY outer ρ-component's
/// `½ tr(H⁻¹ ∂H/∂ρ)` trace — not only the ordered Beta--Bernoulli α one. This pins the ARD
/// log-precision trace (`ard_log_precision_hessian_trace`, routed through the
/// kept-subspace `latent_inverse_diagonal_kept`) against the fixed-state central
/// difference of `log|H|` w.r.t. `log_ard[atom][axis]`, with deflation active.
#[test]
pub(crate) fn ard_log_precision_trace_matches_dense_fd_pd_region_deflation() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    rho.log_lambda_sparse = 0.5;
    // The ARD log-precision stays at the fixture default; lifting it off the floor
    // pushes the inner solve into a non-PD basin at this ρ. The ARD curvature block is
    // small but live, and its log-α derivative is exactly what the trace and the
    // FD oracle both probe — with deflation active (5 directions).
    let (_value, _loss, cache) = term
        .penalized_laml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    assert!(
        cache.gauge_deflated_directions > 0,
        "ARD deflation regression requires a deflated direction; got {}",
        cache.gauge_deflated_directions
    );
    let solver = DeflatedArrowSolver::plain(&cache);
    let analytic = term
        .ard_log_precision_hessian_trace(&rho, &cache, &solver)
        .expect("ARD log-precision trace");

    let h = 1.0e-5;
    let mut checked = 0usize;
    for atom in 0..rho.log_ard.len() {
        for axis in 0..rho.log_ard[atom].len() {
            let mut rho_plus = rho.clone();
            let mut rho_minus = rho.clone();
            rho_plus.log_ard[atom][axis] += h;
            rho_minus.log_ard[atom][axis] -= h;
            let fd_half = 0.5
                * (fixed_state_logdet(term.clone(), &target, &rho_plus)
                    - fixed_state_logdet(term.clone(), &target, &rho_minus))
                / (2.0 * h);
            let a = analytic[atom][axis];
            let tol = 5.0e-3 * (1.0 + fd_half.abs().max(a.abs()));
            assert!(
                (fd_half - a).abs() <= tol,
                "ARD trace atom={atom} axis={axis}: fd={fd_half:.8e} analytic={a:.8e} \
                 gap={:.6e} tol={tol:.6e}",
                (fd_half - a).abs()
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no ARD axes were checked");
}

/// #Bug4 — the assignment log-strength ρ-trace must carry NO contribution from a
/// FIXED (ungated) logit. `assignment_prior_grad_hdiag` zeroes the assembled
/// `htt` diagonal entry of every fixed logit, so its ρ-derivative — which the
/// `assignment_log_strength_hessian_trace` contracts against the selected-inverse
/// diagonals — must also be zero. Equivalently, the trace must be INVARIANT to the
/// fixed atom's logit VALUES (its curvature contribution is identically masked),
/// while remaining sensitive to a FREE atom's logits (proving the fixture is not
/// vacuous). We hold the converged cache/solver fixed and only re-evaluate the
/// analytic trace with perturbed logits, isolating the masked source term.
#[test]
pub(crate) fn assignment_log_strength_trace_ignores_fixed_logit_bug4() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    // Atom 1 is the #1026 ungated background tier: a FIXED (inert) logit.
    term.assignment = term
        .assignment
        .clone()
        .with_ungated(vec![false, true])
        .unwrap();
    assert!(
        term.assignment.logit_is_fixed(1) && !term.assignment.logit_is_fixed(0),
        "atom 1 must be the fixed (ungated) logit, atom 0 free"
    );
    // Find a PD-region ρ (the ungated atom shifts the feasibility boundary), then
    // fit the term there so the selected inverse is well-posed.
    {
        let mut found = None;
        for &r in &[1.0_f64, 1.5, 2.0, 2.5, 3.0, 0.5, 0.0, -0.5] {
            let mut probe = term.clone();
            let mut rr = rho.clone();
            rr.log_lambda_sparse = r;
            if probe
                .penalized_laml_criterion_with_cache(
                    target.view(),
                    &rr,
                    None,
                    5,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                )
                .is_ok()
            {
                found = Some(r);
                break;
            }
        }
        rho.log_lambda_sparse =
            found.expect("no PD-region ρ found for the ungated fixed-logit fixture");
    }
    let (_value, _loss, cache) = term
        .penalized_laml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache at the PD ρ");
    let solver = DeflatedArrowSolver::plain(&cache);

    // The selected-inverse diagonals the trace contracts against must be nonzero
    // (else "no contribution" is vacuously true).
    let inv_diag = solver
        .latent_inverse_diagonal()
        .expect("selected-inverse diagonal");
    assert!(
        inv_diag.iter().any(|&d| d.abs() > 1e-12),
        "the fixture must have a nonzero selected-inverse diagonal"
    );

    let base_trace = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("baseline prior-Hessian ρ trace");

    // Perturb ONLY the FIXED atom's (atom 1) logits, on the SAME cache. Because
    // its curvature source (`hdiag`/third channels) is masked to zero, the trace
    // must be BIT-IDENTICAL — the fixed logit contributes nothing.
    let mut fixed_perturbed = term.clone();
    for row in 0..fixed_perturbed.n_obs() {
        fixed_perturbed.assignment.logits[[row, 1]] += 1.7;
    }
    let fixed_trace = fixed_perturbed
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("fixed-logit-perturbed trace");
    assert_eq!(
        fixed_trace.to_bits(),
        base_trace.to_bits(),
        "perturbing a FIXED (ungated) logit must not move the ρ-trace \
         (base={base_trace:.12e}, perturbed={fixed_trace:.12e})"
    );

    // Perturbing the FREE atom's (atom 0) logits DOES move the trace — the fixture
    // genuinely exercises the contracted curvature source, so the invariance above
    // is a real mask, not a dead path.
    let mut free_perturbed = term.clone();
    for row in 0..free_perturbed.n_obs() {
        free_perturbed.assignment.logits[[row, 0]] += 1.7;
    }
    let free_trace = free_perturbed
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("free-logit-perturbed trace");
    assert!(
        (free_trace - base_trace).abs() > 1e-9,
        "perturbing a FREE logit must move the ρ-trace (non-vacuity): \
         base={base_trace:.12e}, perturbed={free_trace:.12e}"
    );
}
