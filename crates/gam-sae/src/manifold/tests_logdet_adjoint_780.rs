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
    // Probe both atoms across distinct rows so the cross-row coupling (different
    // rows sharing a column) is exercised on both columns, AND probe the COORD
    // channel — the #1641 defect made BOTH the logit and the coord channel of the
    // IBP θ-adjoint disagree with dense FD (logit ~4× over tol; coord ~10× off),
    // because the cross-row Woodbury pass double-counted the rank-one self term and
    // carried the ρ-trace ½ instead of the full trace. The coord slots do not pass
    // through the Woodbury pass, but they contract the SAME assembled `htt`
    // (whose IBP diagonal carries the cross-row self curvature), so they guard the
    // one-operator consistency of the whole θ-adjoint, not just the logit lane.
    //
    // Dense IBP layout (K = 2, `last_row_layout = None`): per row block, local
    // positions `0..K` are the logit slots (atom = local_pos) and `K..2K` are the
    // coordinate slots (atom = local_pos − K, axis 0), so local_pos 2 ↔ atom 0
    // coord and local_pos 3 ↔ atom 1 coord.
    let probes = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (7usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (1usize, 2usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
        (6usize, 3usize, SaeLocalRowVar::Coord { atom: 1, axis: 0 }),
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
    // Fixed-alpha IBP-MAP is PD only on a JAGGED ρ_sparse landscape on this n=10
    // periodic-bilinear fixture: most values (including the previously-pinned 1.0,
    // which a `log_lambda_sparse`-sweep shows is non-PD — the converge call panics
    // with "Schur complement Cholesky failed: non-PD pivot") leave the undamped
    // joint Hessian indefinite at setup. The contiguous island ρ_sparse ∈
    // [−1.0, −0.4] is solidly PD, and −0.8 sits in its interior: it converges to
    // the SAME PD cache for every inner budget (iter ∈ {5…40}), the cross-row
    // Woodbury source is genuinely live there (max|d_k| ≈ 0.21), and the analytic
    // ρ_sparse trace matches the fixed-state central difference of log|H| to ≈10
    // digits. Setup fix only — no tolerance weakened.
    rho.log_lambda_sparse = -0.8;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
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



/// #1625 (scope expansion) — the LEARNABLE-α IBP-MAP logit θ-adjoint. This is
/// the cross-row Woodbury logit channel of `Γ = tr(H⁻¹ ∂H/∂ℓ)` under
/// `learnable_alpha = true`, a path the fixed-alpha `..._ibp_map` sibling never
/// exercises. Under learnable α the resolved weight convention flips (`weight`
/// stays 1.0 and `log_lambda_sparse` drives `α` via `resolve_learnable_weight`
/// instead of scaling the prior), so the per-column Woodbury coefficient
/// `d_k = w·s'_k` and its mass-derivative `dd_k = w·s''_k` take DIFFERENT numeric
/// values than the fixed-alpha path — yet a single logit perturbation holds α
/// fixed (it only moves `M_k` and the local `z`), so the same off-diagonal
/// cross-row contraction must hold.
///
/// The comparison point must EXIST and be STATIONARY: like the indefinite-basin
/// diagnosis driving the whole #1625 fix, the analytic
/// `Γ = tr(H⁻¹ ∂H/∂θ)` equals the fixed-state central difference of `log|H|`
/// only at a CONVERGED inner cache. A short inner budget (e.g. `iter = 5`) leaves
/// (t, β) non-stationary, and `fixed_state_logdet` (which re-solves with
/// `iter = 0`) then differences `log|H|` about a different state, manufacturing a
/// spurious O(several-%) mismatch that does NOT shrink with the FD step — the
/// tell that it is a state desync, not truncation. Converging the inner solve
/// (`iter = 200`, tol `1e-8`) makes Γ and the FD share one stationary state, and
/// the learnable-α logit adjoint then matches to ≈6 digits.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ibp_map_learnable_alpha_1625() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    // ρ₀ = 0.6 drives a PD learnable-α cache on this fixture (a sweep shows the
    // default 0.1 and ρ₀ ≤ −0.8 are non-PD for learnable α); the cross-row
    // Woodbury source is genuinely live there.
    rho.log_lambda_sparse = 0.6;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-8, 1.0e-8)
        .expect("converged learnable-α cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    // Probe both atoms across distinct rows so the cross-row coupling (different
    // rows sharing a column) is exercised on both columns under learnable α.
    let probes = [
        (0usize, 0usize, 0usize),
        (4usize, 1usize, 1usize),
        (7usize, 0usize, 0usize),
    ];
    for (row, local_pos, atom) in probes {
        let mut plus = term.clone();
        let mut minus = term.clone();
        plus.assignment.logits[[row, atom]] += h;
        minus.assignment.logits[[row, atom]] -= h;
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "learnable-α IBP Gamma row={row} local_pos={local_pos}: \
             fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// #1416 (compact-layout completion) — the IBP cross-row ρ-trace
/// (`assignment_log_strength_hessian_trace`) must add the
/// `½ d_k Σ_{i≠j}(H⁻¹)_{ij} J_i J_j` off-diagonal term under the COMPACT
/// (#1420 top-`k`) row layout, not only the dense layout. The cross-row Woodbury
/// source is installed for both layouts and the θ-adjoint already differentiates
/// both, but the ρ-trace cross-row pass (and the deflation self-curvature
/// downdate) were gated `if last_row_layout.is_none()` — so whenever the budget /
/// `top_k` engaged the compact layout the ρ-gradient of `log|H|` silently dropped
/// the cross-row term.
///
/// A FULL-SUPPORT compact layout (every row active for both atoms) is
/// geometrically IDENTICAL to dense — same logit slots, same assembled `H` — so
/// its `½ ∂log|H|/∂ρ_sparse` must equal both the dense analytic trace and the
/// dense fixed-state central difference. Before the fix the compact trace skipped
/// the cross-row pass and diverged from both; the sibling
/// `ibp_rho_sparse_logdet_trace_matches_dense_fd_1416` confirms the dropped
/// off-diagonal term is genuinely nonzero at this ρ (max|d_k| ≈ 0.21), so this
/// equality is non-vacuous.
#[test]
pub(crate) fn ibp_rho_sparse_logdet_trace_compact_layout_matches_dense_1416() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    // Same solidly-PD island the dense sibling pins (ρ_sparse ∈ [−1.0, −0.4]).
    rho.log_lambda_sparse = -0.8;

    // Converge the inner fit. The tiny fixture's dense Gram is far under the host
    // budget, so production keeps the dense layout (`last_row_layout = None`);
    // this also mutates `term` to the converged (t, β, logit) state.
    let (_value, _loss, dense_cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
        .expect("dense converged cache");
    let dense_solver = DeflatedArrowSolver::plain(&dense_cache);
    let analytic_dense = term
        .assignment_log_strength_hessian_trace(&rho, &dense_cache, &dense_solver)
        .expect("dense rho_sparse trace");

    // Re-assemble the SAME converged state under a forced full-support compact
    // layout, factor it, and recompute the ρ-trace. `assemble_arrow_schur_inner`
    // sets `last_row_layout = Some(layout)`, so the trace takes the compact path.
    let n = target.nrows();
    let coord_dims = vec![1usize, 1usize];
    let coord_offsets = term.assignment.coord_offsets();
    let full_active: Vec<Vec<usize>> = (0..n).map(|_| vec![0usize, 1usize]).collect();
    let layout = SaeRowLayout::from_active_atoms(full_active, coord_dims, coord_offsets);
    let probe = SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM;
    let sys = term
        .assemble_arrow_schur_inner(target.view(), &rho, None, 1.0, probe, Some(Some(layout)))
        .expect("full-support compact assembly");
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_dt, _db, compact_cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).expect("compact factor");
    let compact_solver = DeflatedArrowSolver::plain(&compact_cache);
    let analytic_compact = term
        .assignment_log_strength_hessian_trace(&rho, &compact_cache, &compact_solver)
        .expect("compact rho_sparse trace");

    // Full-support compact must reproduce the dense trace to roundoff.
    let struct_tol = 1.0e-7 * (1.0 + analytic_dense.abs());
    assert!(
        (analytic_dense - analytic_compact).abs() <= struct_tol,
        "compact-layout IBP ρ_sparse logdet trace must equal the dense trace on \
         full support: dense={analytic_dense:.10e}, compact={analytic_compact:.10e}"
    );

    // And the compact trace must independently match the dense fixed-state central
    // difference of log|H| (the full ½ ∂log|H|/∂ρ_sparse including the cross-row
    // off-diagonal) — FD-validating the compact path itself, not just the
    // dense/compact equality.
    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    let fd_tol = 3.0e-3 * (1.0 + fd_half.abs().max(analytic_compact.abs()));
    assert!(
        (fd_half - analytic_compact).abs() <= fd_tol,
        "compact-layout IBP ρ_sparse logdet trace vs dense FD: \
         fd(½∂log|H|/∂ρ)={fd_half:.8e}, compact analytic={analytic_compact:.8e}"
    );
}
