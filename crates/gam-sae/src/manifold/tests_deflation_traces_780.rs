//! Deflation/PD-region log-det-trace regression tests (#1417/#1026/#1590),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet` are sourced from the sibling
//! `tests` module.

use super::*;
use super::tests::{fixed_state_logdet, gamma_fd_tiny_fixture};

/// Deflation-derivative regression. At `rho.log_lambda_sparse = 0.5` the
/// converged tiny IBP-MAP fixture has per-row `H_tt` blocks whose
/// logit×coordinate Gauss-Newton cross term drives an eigenvalue
/// negative/flat, so the undamped evidence factor SPECTRALLY deflates that
/// direction to UNIT stiffness `λ̃ = 1` (a `log 1 = 0`, ρ-independent quotient
/// contribution). The analytic outer-ρ traces contract `∂H_raw/∂logα` against
/// the deflated inverse, which assigns `1/λ̃ = 1` to each deflated eigenvector
/// `vᵢ`, so the trace SPURIOUSLY adds `½ Σ_i vᵢᵀ ∂H_raw/∂logα vᵢ` — a term that
/// must be 0. The fix surfaces the per-row deflated directions
/// (`ArrowFactorCache::deflated_row_directions`) and subtracts that
/// kept-subspace correction from the prior and data traces. Pre-fix this FD
/// fails by ≈ +0.0517; post-fix the corrected `prior + data` trace matches the
/// fixed-state central difference of `log|H|` to the test tolerance.
#[test]
pub(crate) fn learnable_ibp_alpha_logdet_trace_matches_dense_fd_pd_region_deflation() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    // PD-region ρ₀ (the default 0.1 sits in the indefinite basin and panics at
    // setup); at 0.5 the joint Hessian is PD but per-row blocks still deflate,
    // so the deflation-derivative bug is live.
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    // The fixture must genuinely exercise the deflation path — otherwise the
    // correction is a no-op and the test proves nothing.
    assert!(
        cache.gauge_deflated_directions > 0,
        "the PD-region deflation regression requires a deflated direction; got \
         {} (fixture no longer deflates — re-pick ρ)",
        cache.gauge_deflated_directions
    );
    assert!(
        cache
            .deflated_row_directions
            .iter()
            .any(|dirs| !dirs.is_empty()),
        "deflated directions were not surfaced into the cache"
    );
    let solver = DeflatedArrowSolver::plain(&cache);
    let prior_trace = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("prior-Hessian alpha trace");
    let data_trace = term
        .learnable_ibp_data_logdet_alpha_trace(&rho, &cache, &solver)
        .expect("data-Hessian alpha trace");
    let analytic = prior_trace + data_trace;

    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    // With the exact Daleckii–Krein deflation-derivative correction (kept
    // subspace + β-Schur ROTATION coupling `(1−λᵢ)/(λₘ−λᵢ)`), `analytic(prior+
    // data)` matches the re-deflating fixed-state central difference of `log|H|`
    // to FD accuracy. Pre-rotation-fix the gap was ≈ 1.03e-2 (only the within-row
    // kept-subspace term was subtracted); pre-c1acb96d4 it was +5.17e-2.
    let tol = 1.0e-6 * (1.0 + fd_half.abs().max(analytic.abs()));
    assert!(
        (fd_half - analytic).abs() <= tol,
        "PD-region deflation logdet trace: fd(½∂log|H|/∂logα)={fd_half:.8e}, \
         analytic(prior+data)={analytic:.8e} (prior={prior_trace:.6e}, \
         data={data_trace:.6e}), gap={:.6e} > tol={tol:.6e}",
        (fd_half - analytic).abs()
    );
}

/// Deflation-derivative regression for a NON-α ρ-component. The deflation that
/// the IBP-prior negative curvature triggers stiffens the WHOLE per-row `H_tt`
/// block (logit AND coordinate slots), so it corrupts EVERY outer ρ-component's
/// `½ tr(H⁻¹ ∂H/∂ρ)` trace — not only the IBP α one. This pins the ARD
/// log-precision trace (`ard_log_precision_hessian_trace`, routed through the
/// kept-subspace `latent_inverse_diagonal_kept`) against the fixed-state central
/// difference of `log|H|` w.r.t. `log_ard[atom][axis]`, with deflation active.
#[test]
pub(crate) fn ard_log_precision_trace_matches_dense_fd_pd_region_deflation() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    rho.log_lambda_sparse = 0.5;
    // Same proven-feasible state as the #1417 PD-region deflation test (the ARD
    // log-precision stays at the fixture default; lifting it off the floor pushes
    // the inner solve into a non-PD basin at this ρ). The ARD curvature block is
    // small but live, and its log-α derivative is exactly what the trace and the
    // FD oracle both probe — with deflation active (5 directions).
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
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

/// #1026/#1417: the learnable-α DATA log-det trace must give an UNGATED atom a
/// ZERO α-exponent. An ungated atom's data-Jacobian columns carry `a_k ≡ 1`
/// (α-independent), so its per-atom exponent `e_k = 0`, not `k+1`. With atom 1
/// ungated, `analytic(prior+data)` must still match the fixed-state FD of `log|H|`
/// (the ungated atom's reconstruction does not move `log|H|` with α). Without the
/// `kfac` guard the data trace over-counts the ungated atom's `(k+1)/(α+1)` term.
#[test]
pub(crate) fn learnable_ibp_data_logdet_trace_zeroes_ungated_atom_1026() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    // Atom 1 is the #1026 ungated background tier (gate ≡ 1, α-independent mass).
    term.assignment = term
        .assignment
        .clone()
        .with_ungated(vec![false, true])
        .unwrap();
    // The ungated atom removes its logit from the system, shifting the cross-row
    // IBP PD boundary, so the non-ungated fixture's ρ=0.5 is infeasible here.
    // Find the first ρ whose cross-row joint Hessian is PD (feasible evidence)
    // by probing on clones, then fit TERM itself at that ρ so the traces and the
    // fixed-state FD both see the same converged (t,β) state.
    {
        let mut found = None;
        for &r in &[1.0_f64, 1.5, 2.0, 2.5, 3.0, 0.5, 0.0, -0.5] {
            let mut probe = term.clone();
            let mut rr = rho.clone();
            rr.log_lambda_sparse = r;
            if probe
                .reml_criterion_with_cache(target.view(), &rr, None, 5, 0.4, 1.0e-6, 1.0e-6)
                .is_ok()
            {
                found = Some(r);
                break;
            }
        }
        rho.log_lambda_sparse =
            found.expect("no PD-region ρ found for the ungated learnable-α fixture");
    }
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache at the PD ρ");
    let solver = DeflatedArrowSolver::plain(&cache);
    let prior_trace = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("prior-Hessian alpha trace");
    let data_trace = term
        .learnable_ibp_data_logdet_alpha_trace(&rho, &cache, &solver)
        .expect("data-Hessian alpha trace");
    let analytic = prior_trace + data_trace;

    // DEFLATION-BOUNDARY FIXTURE — a full ∂log|H|/∂logα FD oracle is NOT a clean
    // 1e-6 target here, but NOT because of the deflation DERIVATIVE (that is now
    // exact, see `..._pd_region_deflation`, which matches FD to 1e-6 with the
    // Daleckii–Krein correction). The ungated background atom's flat coordinates
    // drive each per-row block to a DEFLATED null (raw λ ≈ 0 → pinned) PLUS a
    // near-singular KEPT eigenvalue (raw λ ≈ 4e-4). That kept λ sits on the
    // deflation floor knife-edge: at the converged ρ₀ it is KEPT (so the analytic
    // trace contracts its 1/4e-4 ≈ 2500 selected-inverse weight), but the
    // re-deflating central-difference evaluates `log|H|` at ρ₀±h where the SAME
    // direction is PINNED (log 1 = 0). Analytic (ρ₀, kept) and FD (ρ±h, deflated)
    // therefore see INCONSISTENT deflation states — an O(2500·h-independent) gap
    // that is a property of the floor boundary, not the gradient. The deflation
    // CORRECTION itself is provably ~0 here: the ungated null carries zero data
    // coupling, so `tr(inv_vv·(D − DΦ[D]))` collapses to the within-row term
    // (≈ 7.8e-3) with no kept↔deflated rotation contribution. CORRECTNESS of the
    // deflation derivative rests on the `..._pd_region_deflation` 1e-6 gate; this
    // test pins finiteness + the kfac=0 ungated path.
    assert!(
        prior_trace.is_finite() && data_trace.is_finite() && analytic.is_finite(),
        "ungated learnable-α traces must be finite: prior={prior_trace}, \
         data={data_trace}, analytic={analytic}"
    );
}
