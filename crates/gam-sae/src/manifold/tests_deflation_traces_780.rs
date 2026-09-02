//! Deflation/PD-region log-det-trace regression tests (#1026/#1590),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet_sample` are sourced from the sibling
//! `tests` module.

use super::tests::gamma_fd_tiny_fixture;
use super::*;

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
                .penalized_quasi_laplace_criterion_with_cache(
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
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            5,
            0.4,
            1.0e-6,
            1.0e-6,
        )
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
