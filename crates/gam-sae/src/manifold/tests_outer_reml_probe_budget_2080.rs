//! #2080 — the OUTER REML ρ-search must terminate in a BOUNDED number of
//! criterion evaluations at wide output dimension (`p ≈ 96`), where the outer
//! line search overshoots into the adjacent indefinite (non-PD Laplace) basin on
//! nearly every probe.
//!
//! Before the fix each such infeasible PROBE ground the inner refinement budget
//! (the FD-safeguard value probes routed through the ACCEPTED `16×/64×
//! inner_max_iter` budget, and the non-PD arm of
//! `converge_inner_for_undamped_logdet` refined the probe up to that budget before
//! refusing) — so a single wide-`p` gradient point issued ~2·d_ρ full-width inner
//! solves, each grinding thousands of inner iterations: the wide-`p` hang. The fix
//! makes an infeasible-ρ PROBE return the typed refusal after one diagnostic
//! factor pass (`refine_progress_extension == false` fast-fails the non-PD arm),
//! runs the FD safeguard's value probes on the PROBE budget over a THROWAWAY clone
//! (so they never mutate the accepted basin), and gates the full 2·d_ρ FD
//! escalation on the inner-criterion width.
//!
//! This exercises the FULL outer `OuterProblem::run` ("SAE manifold") path — the
//! existing #2027 width test explicitly bypasses the outer ρ-search — and asserts
//! a PROBE-COUNT budget (per SPEC's ban on wall-clock budgets), zero mutating
//! value probes, and a materially positive reconstruction EV.

use super::tests::{deterministic_circle_noise, global_ev};
use super::*;
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_solve::rho_optimizer::OuterProblem;
use ndarray::{Array2, ArrayView2, array, s};
use std::sync::Arc;

/// Two planted circles on DISJOINT ambient column parities (circle A on the even
/// output channels, circle B on the odd), driven by two incommensurate phases and
/// per-column standardized. Together they span a rank-4 subspace of the whitened
/// `p`-dim cloud, so an honest K=2 dictionary explains a materially positive
/// fraction of the variance. `p` is the wide-`p` knob that drives the outer hang.
fn two_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        if j % 2 == 0 {
            fa[[0, j]] = deterministic_circle_noise(j, 0);
            fa[[1, j]] = deterministic_circle_noise(j, 1);
        } else {
            fb[[0, j]] = deterministic_circle_noise(j, 2);
            fb[[1, j]] = deterministic_circle_noise(j, 3);
        }
    }
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let ta = std::f64::consts::TAU * (row as f64) / (n as f64);
        let tb = std::f64::consts::TAU * (2.0 * row as f64 + 0.37) / (n as f64);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for j in 0..p {
            z[[row, j]] = ca * fa[[0, j]]
                + sa * fa[[1, j]]
                + cb * fb[[0, j]]
                + sb * fb[[1, j]]
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    for j in 0..p {
        let mut mean = 0.0_f64;
        for row in 0..n {
            mean += z[[row, j]];
        }
        mean /= n as f64;
        let mut var = 0.0_f64;
        for row in 0..n {
            let d = z[[row, j]] - mean;
            var += d * d;
        }
        let sd = (var / n as f64).sqrt().max(1.0e-12);
        for row in 0..n {
            z[[row, j]] = (z[[row, j]] - mean) / sd;
        }
    }
    z
}

/// Build a K-atom, d=1 periodic SAE term seeded the way the production cold path
/// does (PCA-seed the per-atom coordinates, ridge-LSQ each per-atom decoder), with
/// IBP-MAP assignment. Returns the term and the seed reconstruction dispersion the
/// outer cascade scales its ρ seed by. `harmonics` sets the basis size `m = 1 +
/// 2·harmonics`.
fn two_circle_periodic_term(
    z: ArrayView2<'_, f64>,
    k: usize,
    harmonics: usize,
) -> (SaeManifoldTerm, f64) {
    let n = z.nrows();
    let p = z.ncols();
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(num_basis).unwrap());
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let atom_dims = vec![dim; k];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims).unwrap();
    let mut atoms = Vec::with_capacity(k);
    let mut coords_blocks = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    let mut rss = 0.0_f64;
    for atom_idx in 0..k {
        let coords = seed_coords.slice(s![atom_idx, .., 0..dim]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mm = phi.ncols();
        let mut xtx = fast_atb(&phi, &phi);
        for i in 0..mm {
            xtx[[i, i]] += 1.0e-8;
        }
        let xtz = fast_atb(&phi, &z.to_owned());
        let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
        let fitted = phi.dot(&decoder);
        for row in 0..n {
            for col in 0..p {
                let r = z[[row, col]] - fitted[[row, col]];
                rss += r * r;
            }
        }
        let atom = SaeManifoldAtom::new(
            "circle",
            SaeAtomBasisKind::Periodic,
            dim,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(mm),
        )
        .unwrap()
        .with_basis_evaluator(evaluator.clone());
        atoms.push(atom);
        coords_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let seed_dispersion = (rss / (k * n * p) as f64).max(1.0e-12);
    let logits = Array2::<f64>::from_elem((n, k), 6.0);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let assignment =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coords_blocks, manifolds, mode)
            .unwrap();
    (
        SaeManifoldTerm::new(atoms, assignment).unwrap(),
        seed_dispersion,
    )
}

/// Drive the full outer `OuterProblem::run` path on a wide two-circle fixture and
/// return `(reconstruction EV, probe telemetry)`.
fn run_wide_outer_fit(
    n: usize,
    p: usize,
    k: usize,
    harmonics: usize,
) -> (f64, OuterProbeTelemetry) {
    let z = two_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), k, harmonics);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let seed = init_rho.to_flat();
    let n_params = seed.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold")
        .expect("#2080 wide-p outer REML fit must terminate, not hang / abort");
    let telemetry = objective.probe_telemetry();
    let fitted = objective.into_fitted();
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    (ev, telemetry)
}

/// #2080 — the wide-`p` (p=96) K=2 outer REML fit must terminate in a bounded
/// number of criterion evaluations, run every value probe on a throwaway clone
/// (zero mutating value probes), and recover a materially positive EV — even
/// though the outer line search overshoots into the non-PD basin on many probes.
#[test]
fn wide_p_outer_reml_terminates_within_probe_budget_2080() {
    let n = 96usize;
    let p = 96usize;
    let k = 2usize;
    let harmonics = 2usize; // m = 5: [1, sin2πt, cos2πt, sin4πt, cos4πt]
    let (ev, telemetry) = run_wide_outer_fit(n, p, k, harmonics);
    eprintln!(
        "[#2080] wide-p outer fit: ev={ev:.4}, criterion_calls={}, fd_probe_calls={}, \
         infeasible(non_pd_per_row={},cross_row={},schur={},inner_nc={}), \
         wall_cost_value_probes={}, mutating_value_probes={}",
        telemetry.criterion_calls,
        telemetry.fd_probe_calls,
        telemetry.infeasible_non_pd_per_row,
        telemetry.infeasible_cross_row,
        telemetry.infeasible_schur,
        telemetry.infeasible_inner_not_converged,
        telemetry.wall_cost_value_probes,
        telemetry.mutating_value_probes,
    );
    // Bounded criterion (eval / eval_cost / efs) budget — a PROBE COUNT, not a
    // wall-clock limit (SPEC bans time budgets). With `with_max_iter(4)` and a
    // single seed the outer loop cannot issue an unbounded number of full
    // criterion evals; the pre-fix hang was UNBOUNDED inner work PER probe, not an
    // unbounded probe count, so this asserts the complementary invariant.
    assert!(
        telemetry.criterion_calls <= 64,
        "outer REML issued {} criterion calls; expected a bounded (<= 64) probe budget",
        telemetry.criterion_calls
    );
    // Every FD / line-search value probe runs on a throwaway clone: the accepted
    // warm-start basin is never corrupted by a rejected probe (#2080 defect 3).
    assert_eq!(
        telemetry.mutating_value_probes, 0,
        "value probes must not mutate the accepted term basin (found {})",
        telemetry.mutating_value_probes
    );
    // The FD-safeguard probe count is bounded by the outer iteration / seed budget
    // times the per-gradient probe count (2 directional + up to 2·d_ρ escalation),
    // so it stays small — the escalation is gated on criterion width, never
    // unbounded. The bound is generous (per-gradient-point cost × a safe multiple
    // of the outer budget); the exact count is logged above.
    let per_gradient_probe_bound = 2 + 2 * n_params_for(k);
    assert!(
        telemetry.fd_probe_calls <= 16 * per_gradient_probe_bound,
        "FD probe count {} exceeded the bounded per-iteration budget ({})",
        telemetry.fd_probe_calls,
        16 * per_gradient_probe_bound,
    );
    assert!(
        ev.is_finite() && ev > 0.20,
        "wide-p K=2 two-circle outer fit must recover a materially positive EV \
         (got {ev:.4}); two disjoint circles span a rank-4 subspace an honest K=2 \
         dictionary recovers"
    );
}

/// d_ρ for a K-atom, per-atom d=1 ARD periodic fit: 1 (sparse) + K (smooth) + K
/// (ARD) = 1 + 2K.
fn n_params_for(k: usize) -> usize {
    1 + 2 * k
}

/// #2080 — heavier K=3 wide-`p` variant (the issue's headline shape). Same
/// bounded-probe-budget contract.
#[test]
fn wide_p_outer_reml_terminates_k3_heavy_2080() {
    let (ev, telemetry) = run_wide_outer_fit(96, 96, 3, 2);
    eprintln!(
        "[#2080 heavy] K=3 wide-p outer fit: ev={ev:.4}, criterion_calls={}, fd_probe_calls={}, \
         infeasible_total={}, mutating_value_probes={}",
        telemetry.criterion_calls,
        telemetry.fd_probe_calls,
        telemetry.infeasible_total(),
        telemetry.mutating_value_probes,
    );
    assert!(telemetry.criterion_calls <= 96);
    assert_eq!(telemetry.mutating_value_probes, 0);
    assert!(ev.is_finite() && ev > 0.15);
}
