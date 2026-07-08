//! Outer-criterion row subsampling (#977-ride HT subsample of the ρ search).
//!
//! The manifold SAE selects every penalty weight by an outer REML/EFS search
//! whose every criterion probe runs a bounded inner joint fit. Above a row-count
//! threshold the search runs on a deterministic Horvitz–Thompson row subsample —
//! the `n_sub`-row restriction with inverse-inclusion weights `w_i = N/n_sub`
//! installed through the #977 `row_loss_weights` seam — so each probe is the HT
//! estimate of the full-`N` REML criterion, while the final fit and every
//! reported quantity are restored to full `N` at the selected ρ.
//!
//! These tests assert the three contract points:
//!   (a) `n_sub >= N` (and the below-threshold path) reproduces the unsubsampled
//!       criterion exactly — subsampling never engages and the identity-mask
//!       weight-1 subsample equals the full-`N` criterion;
//!   (b) on a planted circle whose smoothing ρ the full search selects, the
//!       forced-subsampled search selects a ρ whose full-`N` refit reconstructs
//!       the signal within a principled tolerance of the full search;
//!   (c) engagement is recorded in `OuterProbeTelemetry`.

use super::tests::{deterministic_circle_noise, global_ev};
use super::*;
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_solve::rho_optimizer::{OuterObjective, OuterProblem};
use ndarray::{Array2, ArrayView2, array, s};
use std::sync::Arc;

/// A single centered circle embedded in `p` standardized ambient channels — a
/// correctly-specified K=1 target, so the fitted EV is high and any ρ-selection
/// difference between the full and subsampled searches shows up as an EV gap.
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
        let (c, s) = (t.cos(), t.sin());
        for j in 0..p {
            z[[row, j]] = c * frame[[0, j]]
                + s * frame[[1, j]]
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

/// A K=1, d=1 periodic SAE term seeded the way the production cold path does
/// (PCA-seed the coordinates, ridge-LSQ the decoder), IBP-MAP assignment, with a
/// live basis evaluator so `materialize_chunk` can re-evaluate `Φ(t)` at the
/// gathered subsample coordinates. Returns the term and the seed dispersion.
fn planted_circle_term(z: ArrayView2<'_, f64>, harmonics: usize) -> (SaeManifoldTerm, f64) {
    let n = z.nrows();
    let p = z.ncols();
    let k = 1usize;
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(num_basis).unwrap());
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let atom_dims = vec![dim; k];
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
    let seed_dispersion = (rss / (n * p) as f64).max(1.0e-12);
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
    let logits = Array2::<f64>::from_elem((n, k), 6.0);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        mode,
    )
    .unwrap();
    (SaeManifoldTerm::new(vec![atom], assignment).unwrap(), seed_dispersion)
}

fn seed_rho(seed_dispersion: f64) -> SaeManifoldRho {
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap()
}

// ── (a) exact reproduction of the unsubsampled criterion ────────────────────

#[test]
fn below_threshold_never_engages_and_is_byte_identical() {
    // The production threshold only engages at large N; a modest fit stays on the
    // full-N path (no subsample state, telemetry blank), byte-identical to the
    // pre-subsampling behavior.
    assert!(plan_outer_criterion_subsample_rows(1000).is_none());
    assert_eq!(
        plan_outer_criterion_subsample_rows(
            OUTER_CRITERION_SUBSAMPLE_ROWS * OUTER_CRITERION_SUBSAMPLE_MIN_CUT
        ),
        Some(OUTER_CRITERION_SUBSAMPLE_ROWS)
    );

    let z = one_circle_target(400, 10, 0.05);
    let (term, disp) = planted_circle_term(z.view(), 3);
    let objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, seed_rho(disp), 8, 0.04, 1.0e-6, 1.0e-6);
    assert!(!objective.outer_row_subsample_engaged());
    assert_eq!(objective.probe_telemetry().subsample_rows, 0);
    assert_eq!(objective.probe_telemetry().subsample_full_rows, 0);
}

#[test]
fn n_sub_ge_n_is_a_noop_and_reproduces_the_full_criterion() {
    let z = one_circle_target(400, 10, 0.05);
    let (term, disp) = planted_circle_term(z.view(), 3);
    let n = z.nrows();
    let rho = seed_rho(disp);

    // Full-N criterion on the pristine term.
    let mut full_term = term.clone();
    let (cost_full, _) = full_term
        .reml_criterion(z.view(), &rho, None, 8, 0.04, 1.0e-6, 1.0e-6)
        .unwrap();

    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, rho.clone(), 8, 0.04, 1.0e-6, 1.0e-6);
    // Forcing engagement at n_sub == N is a no-op (the full-N path is exactly the
    // criterion), so the objective is unchanged and stays at full N.
    assert!(!objective.engage_outer_row_subsample(n));
    assert!(!objective.outer_row_subsample_engaged());

    // The identity-mask, weight-1 subsample reproduces the full-N criterion: the
    // uniform inclusion weight collapses to the unweighted path and the gathered
    // rows are the whole term in order.
    let identity: Vec<usize> = (0..n).collect();
    let (mut sub_term, sub_target) = objective.build_subsampled_term(&identity).unwrap();
    assert_eq!(sub_target.nrows(), n);
    assert!(sub_term.row_loss_weights().is_none());
    let (cost_sub, _) = sub_term
        .reml_criterion(sub_target.view(), &rho, None, 8, 0.04, 1.0e-6, 1.0e-6)
        .unwrap();
    assert!(
        (cost_sub - cost_full).abs() <= 1.0e-6 * (1.0 + cost_full.abs()),
        "identity-mask subsample criterion {cost_sub} must reproduce the full-N criterion {cost_full}"
    );
}

#[test]
fn uniform_mask_is_exact_size_and_deterministic() {
    let seed = outer_subsample_seed(4096, 32, 2, 128);
    let a = deterministic_uniform_row_mask(4096, 512, seed);
    let b = deterministic_uniform_row_mask(4096, 512, seed);
    assert_eq!(a.len(), 512);
    assert_eq!(a, b, "same (n, n_sub, seed) must draw the identical mask");
    assert!(a.windows(2).all(|w| w[0] < w[1]), "mask must be sorted, unique");
    assert!(*a.last().unwrap() < 4096);
    // n_sub >= n_full degrades to the identity mask.
    assert_eq!(deterministic_uniform_row_mask(300, 300, seed), (0..300).collect::<Vec<_>>());
    assert_eq!(deterministic_uniform_row_mask(300, 999, seed), (0..300).collect::<Vec<_>>());
}

// ── (c) telemetry records engagement ────────────────────────────────────────

#[test]
fn telemetry_records_subsample_engagement_and_probe_count() {
    let z = one_circle_target(768, 12, 0.05);
    let (term, disp) = planted_circle_term(z.view(), 3);
    let rho = seed_rho(disp);
    let n_full = z.nrows();
    let n_sub = 192usize;
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, rho.clone(), 8, 0.04, 1.0e-6, 1.0e-6);

    // Force engagement below the production threshold (the test seam).
    assert!(objective.engage_outer_row_subsample(n_sub));
    assert!(objective.outer_row_subsample_engaged());
    assert_eq!(objective.probe_telemetry().subsample_rows, n_sub);
    assert_eq!(objective.probe_telemetry().subsample_full_rows, n_full);

    // One value probe runs on the subsample (n_sub rows) and tallies a criterion call.
    let rho_flat = rho.to_flat();
    let subsampled_cost = objective.eval_cost(&rho_flat).unwrap();
    assert!(
        subsampled_cost.is_finite(),
        "subsampled outer criterion must be finite, got {subsampled_cost}"
    );
    let probes = objective.probe_telemetry().criterion_calls;
    assert!(probes >= 1);

    // Restoring for the final fit swaps back to full N and records the
    // probes-on-subsample count.
    objective.restore_full_rows_for_final_fit();
    assert!(!objective.outer_row_subsample_engaged());
    assert_eq!(objective.probe_telemetry().subsample_probe_calls, probes);
}

// ── (b) subsampled search selects ρ within a principled tolerance ────────────

#[test]
fn subsampled_search_selects_rho_close_to_full_search() {
    let z = one_circle_target(768, 12, 0.04);

    // Full-N ρ search (below threshold, so it never auto-engages).
    let (term_full, disp) = planted_circle_term(z.view(), 3);
    let rho = seed_rho(disp);
    let seed_flat = rho.to_flat();
    let n_params = seed_flat.len();
    let mut obj_full =
        SaeManifoldOuterObjective::new(term_full, z.clone(), None, rho.clone(), 8, 0.04, 1.0e-6, 1.0e-6);
    assert!(!obj_full.outer_row_subsample_engaged());
    OuterProblem::new(n_params)
        .with_initial_rho(seed_flat.clone())
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut obj_full, "SAE manifold full-N")
        .expect("full-N outer REML fit must terminate");
    let fitted_full = obj_full.into_fitted();
    let ev_full = global_ev(z.view(), fitted_full.term.fitted().view());
    let lam_full = fitted_full.rho.log_lambda_smooth[0];

    // Same fit with the outer ρ search forced onto a HT row subsample.
    let (term_sub, disp_sub) = planted_circle_term(z.view(), 3);
    let rho_sub = seed_rho(disp_sub);
    let mut obj_sub =
        SaeManifoldOuterObjective::new(term_sub, z.clone(), None, rho_sub.clone(), 8, 0.04, 1.0e-6, 1.0e-6);
    assert!(obj_sub.engage_outer_row_subsample(192));
    assert!(obj_sub.outer_row_subsample_engaged());
    OuterProblem::new(n_params)
        .with_initial_rho(seed_flat)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut obj_sub, "SAE manifold subsampled")
        .expect("subsampled outer REML fit must terminate");
    // The final fit is at full N (the search's subsample is restored inside
    // into_fitted).
    let fitted_sub = obj_sub.into_fitted();
    let ev_sub = global_ev(z.view(), fitted_sub.term.fitted().view());
    let lam_sub = fitted_sub.rho.log_lambda_smooth[0];

    // The subsampled search recovers the planted circle essentially as well as the
    // full search, and selects a smoothing strength within a principled tolerance.
    assert!(
        ev_sub > 0.5,
        "subsampled fit must recover a materially positive EV, got {ev_sub}"
    );
    assert!(
        ev_sub >= ev_full - 0.08,
        "subsampled EV {ev_sub} must be within tolerance of the full-N EV {ev_full}"
    );
    assert!(
        (lam_sub - lam_full).abs() < 1.5,
        "subsampled smoothing ρ (logλ={lam_sub}) must land within tolerance of the \
         full-N choice (logλ={lam_full})"
    );
}
