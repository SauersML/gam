//! #2230/#2087 — the outer REML ρ-search must descend the basin LOWER ENVELOPE
//! `V*(ρ) = min_b V_b(ρ)` over a small bundle of saved inner basins, not the
//! hysteretic single warm-start trajectory `V_{b(warm,ρ)}(ρ)` whose value JUMPS
//! at basin-boundary crossings (the measured pathology: hours of `[#1026]
//! restoring inner-fit reconstruction incumbent` churn = the outer line search
//! oscillating across a boundary the trajectory-dependent criterion cannot
//! represent).
//!
//! These tests drive PRODUCTION entry points — the full `OuterProblem::run`
//! ("SAE manifold") path and the `OuterObjective` value lane `eval_cost` the
//! generic planner calls — and assert:
//!  (A) a two-basin fit engages the envelope, keeps a BOUNDED bundle (the state
//!      that replaces unbounded oscillation), and still fits;
//!  (B) the `inner_max_iter == 0` FREEZE contract bypasses the bundle entirely
//!      (verbatim reuse, no exploration);
//!  (C) at a FIXED ρ the envelope value NEVER increases as more basins are
//!      admitted mid-fit (admission can only lower the envelope).

use super::tests::{deterministic_circle_noise, global_ev};
use super::*;
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_solve::rho_optimizer::{OuterObjective, OuterProblem};
use ndarray::{Array1, Array2, ArrayView2, array, s};
use std::sync::Arc;

/// Two planted circles on DISJOINT ambient column parities (circle A on the even
/// output channels, circle B on the odd), per-column standardized — a rank-4
/// whitened cloud an honest K=2 dictionary explains. With two atoms competing for
/// the same rows this has genuinely distinct inner ROUTINGS (which atom claims
/// which circle), i.e. the coexisting basins the envelope exists to track.
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
/// IBP-MAP assignment. Returns the term and the seed reconstruction dispersion.
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

/// Construct a production `SaeManifoldOuterObjective` for the K=2 two-circle
/// fixture at the given inner budget.
fn two_circle_objective(
    n: usize,
    p: usize,
    k: usize,
    harmonics: usize,
    inner_max_iter: usize,
) -> (SaeManifoldOuterObjective, Array2<f64>, Array1<f64>) {
    let z = two_circle_wide_target(n, p, 0.03);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), k, harmonics);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let seed = init_rho.to_flat();
    let objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho,
        inner_max_iter,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    (objective, z, seed)
}

/// (A) A two-basin fit engages the envelope, keeps the bundle BOUNDED by the
/// derived `max_members`, and still fits. The bounded bundle size is the finite
/// state that replaces the unbounded restore-churn the single trajectory
/// produced (task 8a: an equivalent counter — the max bundle size — stays ≤ a
/// small derived bound).
#[test]
fn two_basin_outer_fit_engages_bounded_envelope() {
    let n = 96;
    let p = 48;
    let k = 2;
    let (mut objective, z, seed) = two_circle_objective(n, p, k, 2, 8);
    let n_params = seed.len();
    OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold basin envelope")
        .expect("two-basin outer REML fit must terminate");
    let telemetry = objective.probe_telemetry();

    // The dense-admitted value lanes ran the envelope (not the streaming/freeze
    // bypass).
    assert!(
        telemetry.basin_envelope_evals > 0,
        "the basin envelope must engage on a dense-admitted two-circle fit"
    );
    // The bundle stayed BOUNDED by the derived cap `max_members = clamp(K,2,4)`.
    let cap = k.clamp(2, 4);
    assert!(
        telemetry.basin_max_members >= 1 && telemetry.basin_max_members <= cap,
        "bundle size {} must stay in 1..={cap} (bounded state, not unbounded churn)",
        telemetry.basin_max_members
    );
    // The fit still recovers materially positive reconstruction variance.
    let fitted = objective.into_fitted();
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    assert!(ev > 0.3, "two-circle K=2 envelope fit EV {ev} too low");
}

/// (B) The `inner_max_iter == 0` FREEZE contract MUST bypass the bundle: a freeze
/// evaluation is verbatim reuse, no exploration, so the envelope machinery never
/// engages and the bundle is never even seeded.
#[test]
fn freeze_contract_bypasses_the_bundle() {
    let n = 64;
    let p = 32;
    let k = 2;
    // inner_max_iter == 0 ⇒ freeze lane.
    let (mut objective, _z, seed) = two_circle_objective(n, p, k, 2, 0);
    // Drive the production value lane directly a few times.
    for _ in 0..4 {
        let value = objective
            .eval_cost(&seed)
            .expect("freeze-lane value evaluation should complete");
        assert!(value.is_finite(), "freeze-lane value must be finite");
    }
    let telemetry = objective.probe_telemetry();
    assert_eq!(
        telemetry.basin_envelope_evals, 0,
        "the freeze lane must never run the basin envelope"
    );
    assert_eq!(
        telemetry.basin_max_members,
        0,
        "the freeze lane must never seed the basin bundle"
    );
}

/// (C) Anti-hysteresis: the envelope is a well-defined FUNCTION of ρ, so the
/// production `eval_cost` value at a FIXED ρ is STABLE across re-evaluation — the
/// exact property the hysteretic single-trajectory criterion lacked (its value
/// depended on which basin the warm start last landed in, so re-evaluating the
/// same ρ churned). The value lane never commits `self.term`, so the discovery
/// trajectory is identical across visits and every saved member re-converges from
/// its own already-at-ρ state, giving a reproducible envelope value.
///
/// (The complementary invariant — admitting a basin can only LOWER the envelope —
/// is the bundle's own algebra, unit-tested in `basin_bundle.rs`
/// [`admitting_a_better_basin_lowers_the_envelope_and_switches_argmin`] and
/// [`envelope_is_continuous_across_the_basin_crossing`].)
#[test]
fn fixed_rho_envelope_value_is_stable_across_re_evaluation() {
    let n = 96;
    let p = 48;
    let k = 2;
    let (mut objective, _z, seed) = two_circle_objective(n, p, k, 2, 8);

    let c1 = objective
        .eval_cost(&seed)
        .expect("first seed-ρ envelope eval must succeed");
    let c2 = objective
        .eval_cost(&seed)
        .expect("second seed-ρ envelope eval must succeed");
    let c3 = objective
        .eval_cost(&seed)
        .expect("third seed-ρ envelope eval must succeed");
    let telemetry = objective.probe_telemetry();

    // The envelope value at a fixed ρ is reproducible to within the inner
    // objective stall band (the level below which the criterion is converged).
    let tol = SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * c1.abs().max(1.0) * 16.0;
    assert!(
        (c2 - c1).abs() <= tol && (c3 - c1).abs() <= tol,
        "fixed-ρ envelope not stable: c1={c1} c2={c2} c3={c3} (tol {tol})"
    );
    // The envelope engaged on every visit and the bundle stayed bounded.
    assert_eq!(
        telemetry.basin_envelope_evals, 3,
        "three eval_cost calls must run exactly three envelope evals"
    );
    let cap = k.clamp(2, 4);
    assert!(
        telemetry.basin_max_members >= 1 && telemetry.basin_max_members <= cap,
        "bundle size {} outside 1..={cap}",
        telemetry.basin_max_members
    );
}
