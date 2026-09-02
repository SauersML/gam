//! Deflation/PD-region log-det-trace regression tests (#1026/#1590),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet_sample` are sourced from the sibling
//! `tests` module.

use super::tests::{
    FiniteDifferenceStratumCertificate, certified_central_logdet_difference,
    fixed_state_logdet_sample, gamma_fd_tiny_fixture,
};
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
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    rho.log_lambda_sparse = 0.5;
    // The ARD log-precision stays at the fixture default; lifting it off the floor
    // pushes the inner solve into a non-PD basin at this ρ. The ARD curvature block is
    // small but live, and its log-α derivative is exactly what the trace and the
    // FD oracle both probe — with deflation active (5 directions).
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
        .expect("converged cache");
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

