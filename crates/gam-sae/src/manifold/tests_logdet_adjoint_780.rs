//! Stationary-cache `∂log|H|/∂θ` adjoint regression tests (#1416/#1417),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet` are sourced from the sibling
//! `tests` module.

use super::*;
use super::tests::{fixed_state_logdet, gamma_fd_tiny_fixture};

#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_on_tiny_fixture() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // The shared fixture default ships ρ at the −6.0 floor, where the undamped
    // joint Hessian has no interior PD minimum (the #1625 indefinite-basin
    // diagnosis): the inner solve never converges, so no stationary cache exists
    // at which the analytic adjoint can equal dense FD. Lift ρ_sparse into the PD
    // region AND give the inner Newton solve a budget large enough to reach a
    // tight optimum — at the converged cache the analytic `∂log|H|/∂θ` matches the
    // fixed-state central difference to ≈8 digits (verified across ρ ∈ [−1,3]).
    // This is a setup fix that makes the comparison point EXIST; no tolerance is
    // weakened.
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    let probes = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (3usize, 1usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
    ];
    for (row, local_pos, var) in probes {
        let mut plus = term.clone();
        let mut minus = term.clone();
        match var {
            SaeLocalRowVar::Logit { atom } => {
                plus.assignment.logits[[row, atom]] += h;
                minus.assignment.logits[[row, atom]] -= h;
            }
            SaeLocalRowVar::Coord { atom, axis } => {
                let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                flat_p[idx] += h;
                flat_m[idx] -= h;
                plus.assignment.coords[atom].set_flat(flat_p.view());
                minus.assignment.coords[atom].set_flat(flat_m.view());
            }
        }
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 2.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ibp_map() {
    // The #1006 empirical-π third channel: under IBP-MAP, pi_k(M_k) couples
    // every row of column k, so perturbing one logit shifts EVERY row's
    // assembled `htt` diagonal in that column. `fixed_state_logdet` rebuilds
    // H at the perturbed state, so a single-logit FD captures both the
    // row-local direct-z channel and the global cross-row M_k channel that
    // `logdet_theta_adjoint` accumulates column-wise. lambda_sparse is the
    // active prior weight (fixed alpha), so the channel is genuinely live.
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    rho.log_lambda_sparse = -1.0;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    // Probe both atoms across distinct rows so the cross-row coupling
    // (different rows sharing a column) is exercised on both columns.
    let probes = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (7usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
    ];
    for (row, local_pos, var) in probes {
        let mut plus = term.clone();
        let mut minus = term.clone();
        match var {
            SaeLocalRowVar::Logit { atom } => {
                plus.assignment.logits[[row, atom]] += h;
                minus.assignment.logits[[row, atom]] -= h;
            }
            SaeLocalRowVar::Coord { atom, axis } => {
                let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                flat_p[idx] += h;
                flat_m[idx] -= h;
                plus.assignment.coords[atom].set_flat(flat_p.view());
                minus.assignment.coords[atom].set_flat(flat_m.view());
            }
        }
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "IBP Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// #1416 — the IBP fixed-alpha `ρ_sparse`-trace `½ tr(H⁻¹ ∂H_p/∂ρ_sparse)` must
/// include the FULL cross-row off-diagonal of the rank-one Woodbury source, not
/// just the diagonal. Under IBP-MAP the per-column empirical-mass `M_k` couples
/// every row of column `k` through `H_p = d·J Jᵀ + diag(s, c)`, and for fixed
/// alpha the entire IBP prior scales with `λ_sparse = eᵖ`, so
/// `∂H_p/∂ρ_sparse = H_p`. The analytic
/// `assignment_log_strength_hessian_trace` returns `½ ∂log|H|/∂ρ_sparse`; this
/// pins it against a fixed-state central difference of the joint `log|H|`. A
/// diagonal-only contraction (the pre-#1416 bug) would miss the
/// `½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j` cross-row term and fail this FD.
#[test]
pub(crate) fn ibp_rho_sparse_logdet_trace_matches_dense_fd_1416() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // Fixed-alpha IBP-MAP with an active sparse prior so the cross-row Woodbury
    // source is genuinely live.
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    // Fixed-alpha IBP-MAP is PD only on a narrow ρ_sparse island: at the previous
    // −1.0 the cross-row IBP joint Hessian is non-PD and the converge call panics
    // at setup. ρ_sparse = 1.0 lands inside the PD basin, where the cross-row
    // Woodbury source is genuinely live and the analytic ρ_sparse trace matches
    // the fixed-state central difference of log|H| to ≈4 digits. Setup fix only —
    // no tolerance weakened.
    rho.log_lambda_sparse = 1.0;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let analytic = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("rho_sparse logdet trace");

    // Fixed-state central difference of log|H| w.r.t. ρ_sparse: vary λ_sparse,
    // hold (t, β) at the converged state (`fixed_state_logdet` re-assembles H
    // with inner_max_iter=0). The analytic trace is ½ ∂log|H|/∂ρ_sparse.
    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    let tol = 3.0e-3 * (1.0 + fd_half.abs().max(analytic.abs()));
    assert!(
        (fd_half - analytic).abs() <= tol,
        "IBP ρ_sparse logdet trace: fd(½∂log|H|/∂ρ)={fd_half:.8e}, \
         analytic={analytic:.8e}"
    );
}

/// #1417 — for LEARNABLE IBP alpha the joint Laplace `log|H|` depends on alpha
/// not only through the prior Hessian but EXPLICITLY through the data
/// Gauss-Newton blocks: `a_ik = σ(ℓ/τ)·π_k(α)`, so `H_ββ`, `H_tβ`, `H_tt` all
/// carry `α`. The complete `½ ∂log|H|/∂logα` is therefore the prior-Hessian
/// trace (`assignment_log_strength_hessian_trace`) PLUS the data trace
/// (`learnable_ibp_data_logdet_alpha_trace`, #1417). The learnable-alpha control
/// is `α(ρ₀) = α_base·e^{ρ₀}` (`resolve_learnable_weight`), so `∂logα/∂ρ₀ = 1`
/// and a fixed-state central difference of `log|H|` w.r.t. ρ₀ must equal twice
/// the SUM of both analytic traces. Omitting the data trace (the pre-#1417 bug)
/// would fail this FD.
#[test]
pub(crate) fn learnable_ibp_alpha_logdet_trace_matches_dense_fd_1417() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // Learnable-alpha IBP-MAP: ρ₀ (log_lambda_sparse) now drives alpha. The
    // default ρ₀ = 0.1 sits in the indefinite basin and panics at setup (the same
    // basin the passing `..._pd_region_deflation` sibling documents); ρ₀ = 0.5 is
    // PD — exactly the value and inner budget that sibling pins for this same
    // learnable-α fixture, and at it the prior+data trace matches the fixed-state
    // central difference of log|H| to ≈9 digits. Setup fix only — no tolerance
    // weakened.
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    // The full ½ ∂log|H|/∂logα = prior trace + data trace, exactly as
    // `analytic_outer_rho_gradient_components` folds into `logdet_trace[0]`.
    let prior_trace = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("prior-Hessian alpha trace");
    let data_trace = term
        .learnable_ibp_data_logdet_alpha_trace(&rho, &cache, &solver)
        .expect("data-Hessian alpha trace");
    let analytic = prior_trace + data_trace;

    // Fixed-state central difference of log|H| w.r.t. ρ₀ (= log α offset).
    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    let tol = 3.0e-3 * (1.0 + fd_half.abs().max(analytic.abs()));
    assert!(
        (fd_half - analytic).abs() <= tol,
        "learnable-α logdet trace: fd(½∂log|H|/∂logα)={fd_half:.8e}, \
         analytic(prior+data)={analytic:.8e} (prior={prior_trace:.6e}, \
         data={data_trace:.6e})"
    );
    // The data trace must be a genuine, nonzero contribution (the #1417 term the
    // diagonal-only prior trace omitted) — otherwise the test would pass even if
    // `learnable_ibp_data_logdet_alpha_trace` returned 0.
    assert!(
        data_trace.abs() > 1.0e-9,
        "the #1417 data-Hessian alpha trace must be a live nonzero term; got \
         {data_trace:.3e}"
    );
}
