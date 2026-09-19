//! Deflation/PD-region log-det-trace regression tests (#1026/#1590),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet_sample` are sourced from the sibling
//! `tests` module.

use super::tests_recovery_split_780::{
    FiniteDifferenceStratumCertificate, certified_central_logdet_difference,
    fixed_state_logdet_sample, gamma_fd_tiny_fixture,
};
use super::*;

/// Deflation-derivative regression for a NON-α ρ-component. A row deflation
/// stiffens the WHOLE per-row `H_tt` block (logit AND coordinate slots), so it
/// corrupts EVERY outer ρ-component's `½ tr(H⁻¹ ∂H/∂ρ)` trace — not only the
/// ordered Beta--Bernoulli α one. This pins the ARD log-precision trace
/// (`ard_log_precision_hessian_trace`, through its per-slot Daleckii–Krein
/// correction) against the fixed-state central difference of `log|H|` w.r.t.
/// `log_ard[atom][axis]`, with deflation active. The anchor is the shared ordered
/// Beta--Bernoulli deflated anchor,
/// [`super::tests_deflated_from_probes_2712::obb_deflated_anchor`], where every
/// row's logit slots deflate by construction: at the historical temperature 0.7
/// the gate-logit Jacobian gives the gates an interior mode and no row deflates
/// (#2080).
#[test]
pub(crate) fn ard_log_precision_trace_matches_dense_fd_pd_region_deflation() {
    // The `gauge_deflated_directions > 0` assertion below is all-or-nothing: the
    // gate is `|g'Hg| <= 1e-8 * max_diag * |g|^2`, so the count collapses to 0 the
    // moment every orbit direction clears the bar and a bare `got 0` cannot say
    // whether the bar is marginally too tight, something now stiffens H_tt along
    // the orbit, or this fixture no longer has a near-null orbit at all — three
    // causes with three different fixes (#2228/#2500).
    //
    // `factor_gauge_deflated_evidence_row` reports the closest disqualified
    // direction, in units of the bar, on exactly that branch — but at `debug`,
    // and NO gam-sae test installs a logger, so without this line the diagnostic
    // is present and silent. Install it at `Debug` so the number reaches the
    // failure output.
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Debug);
    // The ARD log-precision stays at the fixture default; lifting it off the floor
    // pushes the inner solve into a non-PD basin. The ARD curvature block is small
    // but live, and its log-α derivative is exactly what the trace and the FD
    // oracle both probe, with deflation active.
    let (term, rho, target, cache) =
        super::tests_deflated_from_probes_2712::obb_deflated_anchor(
            "ARD log-precision trace on the deflated ordered Beta--Bernoulli anchor",
        );
    assert!(
        cache.gauge_deflated_directions > 0,
        "ARD deflation regression requires a deflated direction; got {}",
        cache.gauge_deflated_directions
    );
    let solver = DeflatedArrowSolver::plain(&cache);
    let analytic = term
        .ard_log_precision_hessian_trace(&rho, &cache, &solver, EvidenceOperator::Majorizer)
        .expect("ARD log-precision trace");

    let h = 1.0e-5;
    let fd_stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
    let mut checked = 0usize;
    for atom in 0..rho.log_ard.len() {
        for axis in 0..rho.log_ard[atom].len() {
            let mut rho_plus = rho.clone();
            let mut rho_minus = rho.clone();
            rho_plus.log_ard[atom][axis] += h;
            rho_minus.log_ard[atom][axis] -= h;
            let fd_half = 0.5
                * certified_central_logdet_difference(
                    &format!("ARD trace atom={atom} axis={axis}"),
                    &fd_stratum,
                    fixed_state_logdet_sample(term.clone(), &target, &rho_plus),
                    fixed_state_logdet_sample(term.clone(), &target, &rho_minus),
                    h,
                );
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

/// #Bug4 — the assignment log-strength ρ-trace carries NO contribution from a FIXED logit.
/// Under frozen routing (#1033) every logit is fixed: the assembled assignment prior zeroes
/// every logit `htt` diagonal entry, so its ρ-derivative — which
/// `assignment_log_strength_hessian_trace` contracts against the selected-inverse diagonals —
/// is zero too, and the trace is INVARIANT to the logit VALUES. The same perturbation on the
/// thawed assignment moves the trace, so the fixture is not vacuous. The converged cache and
/// solver are held fixed and only the analytic trace is re-evaluated, isolating the masked
/// source term.
#[test]
pub(crate) fn assignment_log_strength_trace_ignores_fixed_logit_bug4() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    // Find a PD-region ρ, then fit the term there so the selected inverse is well-posed.
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
            found.expect("no PD-region ρ found for the ordered Beta--Bernoulli fixture");
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

    // Perturbing a FREE logit on the thawed assignment DOES move the trace — the fixture
    // genuinely exercises the contracted curvature source, so the invariance below is a real
    // mask, not a dead path.
    let base_trace = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("baseline prior-Hessian ρ trace");
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

    // Frozen routing holds every logit, on the SAME cache. Its curvature source is masked to
    // zero, so perturbing every free logit must leave the trace BIT-IDENTICAL.
    let mut frozen = term.clone();
    frozen.assignment.frozen_logits = Some(frozen.assignment.logits.clone());
    assert!(
        frozen.assignment.logits_are_fixed(),
        "frozen routing must fix every logit"
    );
    let frozen_trace = frozen
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("frozen-routing trace");
    let mut frozen_perturbed = frozen.clone();
    for row in 0..frozen_perturbed.n_obs() {
        for atom in 0..frozen_perturbed.k_atoms() {
            frozen_perturbed.assignment.logits[[row, atom]] += 1.7;
        }
    }
    let frozen_perturbed_trace = frozen_perturbed
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("frozen-routing perturbed trace");
    assert_eq!(
        frozen_perturbed_trace.to_bits(),
        frozen_trace.to_bits(),
        "perturbing a FIXED (frozen-routing) logit must not move the ρ-trace \
         (base={frozen_trace:.12e}, perturbed={frozen_perturbed_trace:.12e})"
    );
}
