//! Eisenstat–Walker inexact line-search probes for the outer REML ρ-search.
//!
//! The outer BFGS/ARC line search evaluates its Armijo/Wolfe comparison probes
//! through `eval_with_order(Value)`; those criterion values only need to RANK
//! candidate ρ steps, so their inner `(t, β)` solve may stop at the forcing
//! gate `max(η_j·r_entry, τ_full)` with `η_j` tied to the outer optimality
//! measure (EW96 Choice 2 — see the block comment above `EW_FORCING_GAMMA` in
//! `outer_objective.rs` for the forcing sequence and its citations). Gradient
//! evaluations at accepted iterates, the cross-seed ranking lane, the FD
//! safeguard/certificate probes, and the terminal acceptance all stay at FULL
//! tolerance, so the fit the user receives is unchanged.
//!
//! This family asserts the acceptance contract:
//!   1. the terminal (full-tolerance) criterion is identical between the
//!      forcing-on and always-tight arms on a fixed problem driven through an
//!      identical outer-walk call sequence;
//!   2. `loosened_probe_calls > 0` on a multi-iteration outer walk under
//!      forcing (and exactly 0 on the always-tight arm);
//!   3. the probe lane's inner Newton iteration grants are STRICTLY lower in
//!      total under forcing than on the always-tight arm.

use super::tests::{deterministic_circle_noise, global_ev};
use super::*;
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_solve::rho_optimizer::{OuterObjective, OuterProblem};
use ndarray::{Array1, Array2, ArrayView2, array, s};
use std::sync::Arc;

/// A single centered circle embedded in `p` standardized ambient channels —
/// the same well-specified K=1 target shape the #2080/#2153 probe-budget tests
/// use, sized small so the deterministic outer-walk simulation stays fast.
fn one_circle_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut frame = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        frame[[0, j]] = deterministic_circle_noise(j, 0);
        frame[[1, j]] = deterministic_circle_noise(j, 1);
    }
    for r in 0..2 {
        let nrm = (0..p)
            .map(|j| frame[[r, j]] * frame[[r, j]])
            .sum::<f64>()
            .sqrt();
        for j in 0..p {
            frame[[r, j]] /= nrm.max(1.0e-300);
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let t = std::f64::consts::TAU * (row as f64) / (n as f64);
        let (c, sn) = (t.cos(), t.sin());
        for j in 0..p {
            z[[row, j]] = c * frame[[0, j]]
                + sn * frame[[1, j]]
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

/// K=1, d=1 periodic term seeded the production cold way (PCA coordinates,
/// ridge-LSQ decoder), IBP-MAP assignment. Returns the term and the seed
/// reconstruction dispersion the ρ seed is scaled by.
fn one_circle_periodic_term(z: ArrayView2<'_, f64>, harmonics: usize) -> (SaeManifoldTerm, f64) {
    let n = z.nrows();
    let p = z.ncols();
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(num_basis).unwrap());
    let basis_kinds = vec![SaeAtomBasisKind::Periodic];
    let atom_dims = vec![dim];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims).unwrap();
    let coords = seed_coords.slice(s![0, .., 0..dim]).to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mm = phi.ncols();
    let mut xtx = fast_atb(&phi, &phi);
    for i in 0..mm {
        xtx[[i, i]] += 1.0e-8;
    }
    let xtz = fast_atb(&phi, &z.to_owned());
    let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
    let fitted = phi.dot(&decoder);
    let mut rss = 0.0_f64;
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
    .with_basis_evaluator(evaluator);
    let seed_dispersion = (rss / (n * p) as f64).max(1.0e-12);
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        mode,
    )
    .unwrap();
    (
        SaeManifoldTerm::new(vec![atom], assignment).unwrap(),
        seed_dispersion,
    )
}

fn forcing_test_objective(
    z: &Array2<f64>,
    inner_max_iter: usize,
) -> (SaeManifoldOuterObjective, Array1<f64>) {
    let (term, seed_dispersion) = one_circle_periodic_term(z.view(), 2);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
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
    (objective, seed)
}

/// The acceptance contract, driven through a DETERMINISTIC outer-walk call
/// sequence (the exact pattern the BFGS driver produces: gradient evaluation
/// at the seed, gradient evaluation at the next accepted iterate — which
/// establishes the EW ratio — then a train of line-search value probes, then
/// the terminal full-tolerance gradient evaluation). Driving the objective
/// directly makes the two arms take an IDENTICAL ρ sequence, so every
/// comparison below is apples-to-apples regardless of line-search internals.
#[test]
fn eisenstat_walker_forcing_loosens_probes_and_preserves_terminal_criterion() {
    let z = one_circle_target(64, 12, 0.05);
    // A tiny inner chunk (`inner_max_iter = 1`) makes warm-started probes
    // multi-chunk, so the forcing gate has room to stop strictly before the
    // full-tolerance gate — the regime the lever exists for.
    let (mut forced, seed) = forcing_test_objective(&z, 1);
    let (mut tight, seed_b) = forcing_test_objective(&z, 1);
    assert_eq!(seed, seed_b, "fixture must be deterministic across arms");
    tight.set_probe_forcing_enabled(false);

    // Outer iterate 0 (seed) and iterate 1: two accepted-gradient evaluations
    // establish the EW Choice-2 ratio on the forced arm. The small 0.05 step
    // keeps ‖g_1‖/‖g_0‖ near 1 so the forcing term stays materially loose.
    let rho_0 = seed.clone();
    let rho_1 = rho_0.mapv(|v| v + 0.05);
    for objective in [&mut forced, &mut tight] {
        let eval_0 = objective.eval(&rho_0).expect("seed gradient eval");
        assert!(eval_0.cost.is_finite());
        let eval_1 = objective.eval(&rho_1).expect("iterate-1 gradient eval");
        assert!(eval_1.cost.is_finite());
    }

    // A line-search probe train around the accepted iterate — the Value-order
    // lane, exactly what the Wolfe search issues. Identical points on both
    // arms; each probe restores the accepted basin, so the arms stay in
    // lockstep no matter how far each probe's inner solve ran.
    let probe_steps = [0.35, -0.30, 0.25, -0.20, 0.15];
    let mut forced_probe_costs = Vec::new();
    let mut tight_probe_costs = Vec::new();
    for &step in &probe_steps {
        let rho_probe = rho_1.mapv(|v| v + step);
        let vf = forced
            .eval_with_order(&rho_probe, OuterEvalOrder::Value)
            .expect("forced value probe")
            .cost;
        let vt = tight
            .eval_with_order(&rho_probe, OuterEvalOrder::Value)
            .expect("tight value probe")
            .cost;
        forced_probe_costs.push(vf);
        tight_probe_costs.push(vt);
    }

    let forced_telemetry = forced.probe_telemetry();
    let tight_telemetry = tight.probe_telemetry();
    // (2) the forcing lane genuinely engaged: at least one probe stopped at
    // the loosened gate strictly before full tolerance; the always-tight arm
    // never loosens by construction.
    assert!(
        forced_telemetry.loosened_probe_calls > 0,
        "EW forcing must loosen at least one multi-chunk line-search probe; telemetry: \
         {forced_telemetry:?}"
    );
    assert_eq!(
        tight_telemetry.loosened_probe_calls, 0,
        "the always-tight arm must never count a loosened probe; telemetry: {tight_telemetry:?}"
    );
    // (3) strictly fewer inner Newton iteration grants in the probe lane. Both
    // arms count grants through the SAME lane, so the totals are directly
    // comparable; loosening must convert to genuinely skipped inner work.
    assert!(
        forced_telemetry.probe_inner_iterations < tight_telemetry.probe_inner_iterations,
        "forced probe lane must spend strictly fewer inner iteration grants \
         (forced {} vs tight {})",
        forced_telemetry.probe_inner_iterations,
        tight_telemetry.probe_inner_iterations
    );
    // Loosened probe values still present to the line search on the same
    // finite scale as the tight arm's (wall costs included — the wall is the
    // documented finite infeasibility barrier, never `∞`/NaN).
    for (vf, vt) in forced_probe_costs.iter().zip(tight_probe_costs.iter()) {
        assert!(vf.is_finite(), "forced probe cost must be finite, got {vf}");
        assert!(vt.is_finite(), "tight probe cost must be finite, got {vt}");
    }

    // (1) terminal acceptance is FULL tolerance and unchanged: a fresh
    // gradient-lane evaluation (the accepted-iterate lane, which never
    // loosens) at the same terminal ρ̂. Each probe restored the accepted
    // basin, and the pending probe handoff is dropped on a ρ mismatch, so
    // both arms enter this evaluation from identical committed states; the
    // criterion both return is priced at the identical inner KKT optimum. The
    // bound is round-off headroom on top of that shared full-tolerance
    // stationarity — NOT a loosened comparison.
    let rho_hat = rho_1.mapv(|v| v + 0.02);
    let v_forced = forced.eval(&rho_hat).expect("forced terminal eval").cost;
    let v_tight = tight.eval(&rho_hat).expect("tight terminal eval").cost;
    assert!(v_forced.is_finite() && v_tight.is_finite());
    assert!(
        (v_forced - v_tight).abs() <= 1.0e-8 * (1.0 + v_tight.abs()),
        "terminal full-tolerance criterion must be identical across arms: forced {v_forced} vs \
         tight {v_tight}"
    );
}

/// End-to-end: the forcing lane engages inside the REAL outer driver
/// (`OuterProblem::run`) on a multi-iteration walk, the fit terminates, and
/// the returned reconstruction is materially positive — the loosened probes
/// changed the probe spend, not the delivered fit quality.
#[test]
fn ew_forcing_engages_on_full_outer_walk() {
    let z = one_circle_target(64, 12, 0.05);
    let (mut objective, seed) = forcing_test_objective(&z, 2);
    let n_params = seed.len();
    let result = OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(8)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold EW forcing")
        .expect("forced outer walk must terminate");
    let telemetry = objective.probe_telemetry();
    assert!(
        telemetry.loosened_probe_calls > 0,
        "a multi-iteration outer walk must issue at least one loosened line-search probe; \
         telemetry: {telemetry:?}"
    );
    assert!(
        telemetry.probe_inner_iterations > 0,
        "the counted probe lane must have driven the line-search value probes; telemetry: \
         {telemetry:?}"
    );
    objective
        .certify_outer_result(&result)
        .expect("forced outer walk must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    assert!(
        ev > 0.5,
        "the delivered (full-tolerance) fit must still recover the planted circle; EV = {ev}"
    );
}
