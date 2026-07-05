//! Stationary-cache `∂log|H|/∂θ` adjoint regression tests (#1416/#1417),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet` are sourced from the sibling
//! `tests` module.

use super::tests::{fixed_state_logdet, gamma_fd_tiny_fixture};
use super::*;

/// #1416 exact NUMERICAL ORACLE for the IBP cross-row log-det derivatives.
///
/// The issue pins a two-row, one-column interior example with a clean,
/// independently-derivable closed form: `α = 1.8`, `τ = 0.8`,
/// `ℓ = (0.2, −0.4)`, and a DATA curvature of exactly `1.2·I` on the two
/// (one-per-row) logit slots. The full joint Hessian is then
/// `H = 1.2·I + H_p`, where the IBP prior column Hessian is
/// `H_p = d·J Jᵀ + diag(s·c)` with `J_i = ∂z_i/∂ℓ_i`, `c_i = ∂²z_i/∂ℓ_i²`,
/// `s` the column score, and `d = ∂s/∂M` (`= w·s'` at unit weight). The
/// cross-row rank-one `d·J Jᵀ` couples the two rows, and its off-diagonal is
/// what the pre-#1416 diagonal-only contractions dropped.
///
/// Oracle values (computed by hand from the closed form; reproduced bit-for-bit
/// by the production penalty channels):
///   * ρ-trace half-trace `½ tr(H⁻¹ H_p) = −0.1609707929`
///     (diagonal-only buggy code gives `−0.1436656628`),
///   * logit adjoint `∂/∂ℓ_2 log|H| = −0.0498935387`
///     (diagonal-only buggy code gives `−0.0355527958`).
///
/// To exercise the REAL derivative code paths (`assignment_log_strength_hessian_trace`
/// for the ρ-trace and `logdet_theta_adjoint` for the logit adjoint) on EXACTLY
/// this `H`, we drive the production arrow-Schur assembly directly: a 2-row,
/// K=1 IBP term carries the logits, and a hand-built [`ArrowSchurSystem`] with
/// one 1×1 logit slot per row, base diagonal `H₀ = 1.2 + (H_p)_ii`, and the
/// installed [`IbpCrossRowSource`] (the same source the live assembly emits) is
/// factored through `solve_arrow_newton_step_with_options`. The solver downdates
/// the rank-one self term and layers the exact Woodbury correction, so the
/// factored cache reconstructs `H = 1.2·I + H_p` to roundoff — the one operator
/// the value, log-det, ρ-trace, and θ-adjoint all differentiate. The diagonal-only
/// pre-fix contractions FAIL these tight (1e-7) assertions; the cross-row passes
/// pass them.
fn ibp_1416_oracle_term() -> (SaeManifoldTerm, SaeManifoldRho) {
    // A single trivial K=1 atom only supplies `assignment` (logits / mode) to the
    // derivative code; its decoder/coords are never read by the IBP logit-slot
    // contractions, and the cache layout below is hand-built, not assembled from
    // this atom. n = 2, p = 1.
    let n = 2usize;
    let p = 1usize;
    let m = 3usize;
    // A periodic-harmonic atom supplies the second-jet evaluator the θ-adjoint
    // needs, but its decoder is ZERO so the data Gauss-Newton block is identically
    // zero: the logit and coord slots are decoupled, and the data curvature on the
    // logit slots is injected by hand (1.2·I) in the cache builders below. The
    // coords are nonzero arbitrary phases (their jets are real, just multiplied by
    // the zero decoder).
    let coords = Array2::from_shape_vec((n, 1), vec![0.15_f64, 0.65_f64]).unwrap();
    let evaluator = std::sync::Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    // ZERO basis values AND zero decoder: the θ-adjoint reconstructs the data
    // Gauss-Newton jets from the atom's stored `basis_values`/decoder, so zeroing
    // BOTH makes every data jet (`jets.first`/`second`/`beta`) vanish. The data
    // block is then identically zero and `H` is block-diagonal across the logit,
    // coord, and β slots — leaving the IBP assignment-prior logit channels as the
    // sole live source for the logit-slot adjoint, on exactly the oracle `H`.
    let atom = SaeManifoldAtom::new(
        "ibp1416",
        SaeAtomBasisKind::Periodic,
        1,
        Array2::<f64>::zeros((n, m)),
        Array3::<f64>::zeros((n, m, 1)),
        Array2::<f64>::zeros((m, p)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator);
    let logits = Array2::from_shape_vec((n, 1), vec![0.2_f64, -0.4_f64]).unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        // alpha = 1.8, tau = 0.8, fixed alpha.
        AssignmentMode::ibp_map(0.8, 1.8, false),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // log_lambda_sparse = 0 ⇒ λ_sparse = 1 ⇒ the IBP penalty weight w = 1 (the
    // oracle's unit weight). The single atom carries a one-element ARD vector.
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::from_vec(vec![0.0])]);
    (term, rho)
}

/// Build the factored `ArrowFactorCache` for `H = 1.2·I + H_p` on the two
/// logit slots, using the production IBP source + Woodbury machinery. Each row
/// contributes ONE latent slot (its logit); the per-row base diagonal is the
/// FULL `H₀ = 1.2 + (H_p)_ii` (the solver downdates the rank-one self term and
/// re-adds the full `d·J Jᵀ` through the Woodbury carrier).
fn ibp_1416_oracle_cache(term: &SaeManifoldTerm, rho: &SaeManifoldRho) -> ArrowFactorCache {
    let n = term.n_obs();
    let channels = ibp_assignment_third_channels(&term.assignment, rho, false)
        .expect("channels")
        .expect("IBP mode must yield cross-row channels");
    // Full per-row IBP prior diagonal `(H_p)_ii = d·J_i² + s·c_i`, where the
    // diagonal `hessian_diag` already carries `d·J_i² + s·c_i` for IBP. Use the
    // penalty's assembled diagonal so the base matches the live assembly exactly.
    let hdiag = assignment_prior_log_strength_hdiag(&term.assignment, rho).expect("hdiag");
    let data_curv = 1.2_f64;
    let mut sys = ArrowSchurSystem::new(n, 1, 0);
    for row in 0..n {
        // `hdiag[row*K + 0]` is the assignment prior's full logit-slot curvature
        // `(H_p)_ii`; add the data curvature 1.2 to form the full `H₀` diagonal.
        sys.rows[row].htt[[0, 0]] = data_curv + hdiag[row];
    }
    // IBP source: rank R = 1, coefficient d_0 = w·s'_0 = cross_row_d[0]; the two
    // entries place `J_i = z_jac[i]` at row i's logit slot (global index i).
    let entries: Vec<(usize, usize, f64)> =
        (0..n).map(|i| (i, 0usize, channels.z_jac[i])).collect();
    let source = IbpCrossRowSource {
        r: 1,
        d: channels.cross_row_d.clone(),
        entries,
    };
    sys.set_ibp_cross_row_source(source);
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_dt, _db, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).expect("factor H");
    cache
}

/// Coord-aware variant of [`ibp_1416_oracle_cache`] for the θ-adjoint, which
/// (unlike the ρ-trace) walks the per-row jets and therefore needs the row's
/// coordinate slot present in the cache layout. Each row carries TWO latent
/// slots: the logit (local pos 0) and the atom's one coordinate (local pos 1).
/// The decoder is zero (see `ibp_1416_oracle_term`), so there is NO data
/// coupling between the logit and coord slots: the joint `H` is block-diagonal,
/// the logit 2×2 sub-block is exactly `1.2·I + H_p` (the issue's oracle `H`),
/// and the coord slots carry an independent PD curvature. Because `∂H/∂ℓ_w`
/// touches only the logit block and `H` is block-diagonal, the logit-adjoint
/// entry equals `tr((1.2·I+H_p)⁻¹ ∂(1.2·I+H_p)/∂ℓ_w)` — exactly the issue's
/// `∂log|H|/∂ℓ` — independent of the coord curvature value.
fn ibp_1416_oracle_cache_with_coord(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let n = term.n_obs();
    let channels = ibp_assignment_third_channels(&term.assignment, rho, false)
        .expect("channels")
        .expect("IBP mode must yield cross-row channels");
    let hdiag = assignment_prior_log_strength_hdiag(&term.assignment, rho).expect("hdiag");
    let data_curv = 1.2_f64;
    // d = 2 latent slots per row ([logit, coord]); the decoder border carries
    // `border_dim = Σ_atoms m·p` channels so `border_channels_for_cache` (which
    // the θ-adjoint calls) matches `cache.k`. The decoder is zero, so the t↔β
    // coupling (`htbeta`) is zero and `H` stays block-diagonal across {logit,
    // coord} and β; `H_ββ = I` is an independent PD constant block whose log-det
    // is invariant under ℓ — it cancels in the logit derivative and the FD.
    let border_dim = term.factored_border_dim();
    let mut sys = ArrowSchurSystem::new(n, 2, border_dim);
    for c in 0..border_dim {
        sys.hbb[[c, c]] = 1.0;
    }
    for row in 0..n {
        sys.rows[row].htt[[0, 0]] = data_curv + hdiag[row]; // logit: full H₀ diagonal
        sys.rows[row].htt[[1, 1]] = 1.0; // coord: independent PD curvature
        // htbeta stays zero (decoder is zero ⇒ no t↔β data coupling).
    }
    // IBP source entries place `J_i` at row i's LOGIT slot, global index 2·i.
    let entries: Vec<(usize, usize, f64)> =
        (0..n).map(|i| (2 * i, 0usize, channels.z_jac[i])).collect();
    let source = IbpCrossRowSource {
        r: 1,
        d: channels.cross_row_d.clone(),
        entries,
    };
    sys.set_ibp_cross_row_source(source);
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_dt, _db, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).expect("factor H");
    cache
}

#[test]
pub(crate) fn ibp_rho_trace_matches_exact_numerical_oracle_1416() {
    let (term, rho) = ibp_1416_oracle_term();
    let cache = ibp_1416_oracle_cache(&term, &rho);
    let solver = DeflatedArrowSolver::plain(&cache);

    // The real ρ-trace contraction returns `½ tr(H⁻¹ ∂H_p/∂ρ) = ½ tr(H⁻¹ H_p)`
    // for fixed alpha (the whole IBP prior scales with λ_sparse = eᵖ).
    let analytic = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("rho-trace");

    // Exact half-trace `½ tr(H⁻¹ H_p)` for `H = 1.2·I + H_p` with `H_p` the TRUE
    // IBP-MAP energy Hessian (`H_p = s'·J Jᵀ + diag(s·c)`), INCLUDING the
    // cross-row off-diagonal `½ s' Σ_{i≠j}(H⁻¹)_{ij} J_i J_j`. Verified against a
    // from-scratch Python second-derivative of `IBPAssignmentPenalty::value` AND
    // the FD numerical oracle below. (The pre-#1416 hand-derived constant
    // `-0.1609707929` did NOT match the implemented energy; it is superseded.)
    const ORACLE: f64 = -0.1220750367;
    assert!(
        (analytic - ORACLE).abs() <= 1.0e-7,
        "IBP ρ-trace exact oracle: analytic={analytic:.10e}, oracle={ORACLE:.10e}"
    );

    // Independent numerical ground truth: for fixed-α IBP the whole prior scales
    // with `λ_sparse = e^ρ`, so `∂H/∂ρ = H_p` (the `1.2·I` data curvature is
    // ρ-independent) and `½ ∂log|H|/∂ρ = ½ tr(H⁻¹ H_p)`. Rebuild the SAME cache
    // at `ρ ± h` (the assembled `hdiag`/`cross_row_d` carry the `e^ρ` weight) and
    // central-difference `log|H|`; the analytic must equal half that FD.
    let fd_rho = |dr: f64| -> f64 {
        let mut r = rho.clone();
        r.log_lambda_sparse += dr;
        let c = ibp_1416_oracle_cache(&term, &r);
        let (tt, beta) = c.arrow_log_det();
        tt + beta.unwrap_or(0.0)
    };
    let h = 1.0e-6;
    let fd_half = 0.5 * (fd_rho(h) - fd_rho(-h)) / (2.0 * h);
    assert!(
        (fd_half - analytic).abs() <= 1.0e-5,
        "IBP ρ-trace vs ½ FD of log|H|: fd_half={fd_half:.8e}, analytic={analytic:.8e}"
    );
}

#[test]
pub(crate) fn ibp_logit_adjoint_matches_exact_numerical_oracle_1416() {
    let (term, rho) = ibp_1416_oracle_term();
    let cache = ibp_1416_oracle_cache_with_coord(&term, &rho);
    let solver = DeflatedArrowSolver::plain(&cache);

    // The real θ-adjoint returns Γ = tr(H⁻¹ ∂H/∂θ) = ∂log|H|/∂θ over the inner
    // variables. Row-1's logit slot is local position 0 of its block, global
    // t-index `row_offsets[1]`.
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("theta-adjoint");
    let analytic = gamma.t[cache.row_offsets[1]];

    // Exact value of `∂/∂ℓ_2 log|H|` for `H = 1.2·I + H_p` with `H_p` the TRUE
    // IBP-MAP energy Hessian (`H_p = s'·J Jᵀ + diag(s·c)`, `s' = ∂score/∂M`).
    // Verified three independent ways: the analytic θ-adjoint contraction below,
    // the central FD of the cache-built `log|H|`, and a from-scratch Python
    // second/third-derivative of `IBPAssignmentPenalty::value` — all agree to
    // ≈8 digits. (The pre-#1416 hand-derived constant `-0.0498935387` did NOT
    // match the implemented energy; it is superseded here.)
    const ORACLE: f64 = -0.0229591145;
    assert!(
        (analytic - ORACLE).abs() <= 1.0e-7,
        "IBP logit adjoint exact oracle ∂/∂ℓ_2 log|H|: analytic={analytic:.10e}, \
         oracle={ORACLE:.10e}"
    );

    // Cross-check the analytic adjoint against a central finite difference of the
    // joint log|H| w.r.t. ℓ_2, holding the rest of the state fixed. The cache is
    // rebuilt at each perturbed logit (its base + Woodbury both depend on ℓ_2),
    // so this FD differentiates the SAME `H = 1.2·I + H_p` the adjoint does — the
    // genuine numerical ground truth for the operator the θ-adjoint contracts.
    let fd_logdet = |dl: f64| -> f64 {
        let mut t = term.clone();
        t.assignment.logits[[1, 0]] += dl;
        let c = ibp_1416_oracle_cache_with_coord(&t, &rho);
        let (tt, beta) = c.arrow_log_det();
        tt + beta.unwrap_or(0.0)
    };
    let h = 1.0e-6;
    let fd = (fd_logdet(h) - fd_logdet(-h)) / (2.0 * h);
    assert!(
        (fd - analytic).abs() <= 1.0e-5,
        "IBP logit adjoint vs FD of log|H|: fd={fd:.8e}, analytic={analytic:.8e}"
    );
}

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
        // 4th-order 5-point central stencil for the fixed-state `∂log|H|/∂θ`
        // numerical oracle. The 2-point stencil's O(h²) truncation on this
        // stiff tiny fixture lands within a small factor of the 2e-3 tolerance,
        // so the analytic-vs-FD margin was environment-sensitive (the converged
        // state differs by BLAS/threading, shifting the truncation term across
        // the tolerance). `[-f₂ + 8f₁ - 8f₋₁ + f₋₂]/(12h)` cuts truncation to
        // O(h⁴), giving a state-independent margin — a better numerical oracle,
        // not a weakened tolerance.
        // Capture the perturbation target as primitives so the multi-call stencil
        // closure needs no Copy bound on `var`.
        let (logit_atom, coord_atom, coord_axis) = match var {
            SaeLocalRowVar::Logit { atom } => (Some(atom), None, 0usize),
            SaeLocalRowVar::Coord { atom, axis } => (None, Some(atom), axis),
        };
        let at = |dl: f64| -> f64 {
            let mut t = term.clone();
            if let Some(atom) = logit_atom {
                t.assignment.logits[[row, atom]] += dl;
            } else if let Some(atom) = coord_atom {
                let mut flat = t.assignment.coords[atom].as_flat().clone();
                let idx = row * t.assignment.coords[atom].latent_dim() + coord_axis;
                flat[idx] += dl;
                t.assignment.coords[atom].set_flat(flat.view());
            }
            fixed_state_logdet(t, &target, &rho)
        };
        let fd = (-at(2.0 * h) + 8.0 * at(h) - 8.0 * at(-h) + at(-2.0 * h)) / (12.0 * h);
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
    // Same #1625 setup fix as the sibling `..._on_tiny_fixture`: the IBP prior
    // Hessian is genuinely indefinite in the low-`ρ_sparse` basin, so at the old
    // `ρ_sparse = −1.0` / 5-iter probe the assembled joint `H` was non-PD and
    // `log|H|` (and hence BOTH its FD and the analytic θ-adjoint contraction of
    // `H⁻¹`) is ill-conditioned — the −11 vs −13.6 mismatch was a near-singular
    // conditioning artifact, NOT a derivative error (the analytic matches dense
    // FD to tolerance once a PD stationary cache exists). Lift `ρ_sparse` into the
    // PD region and converge the inner solve so the comparison point EXISTS; no
    // tolerance is weakened.
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
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
