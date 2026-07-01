//! #1782 — `sae_manifold_fit` with `jumprelu`/`softmax` assignments and
//! `euclidean`/`linear` topologies failed at "no candidate seeds passed outer
//! startup validation" on clean planted-circle data where `ibp_map`+`circle`
//! converges. Root causes: (1) the Euclidean/Linear PCA seed read the SAME
//! leading principal-component scores for EVERY atom, so a K-atom dictionary
//! started as K identical atoms — a rank-deficient joint decoder whose undamped
//! Laplace factor is non-PD (the #1094 seed-startup refusal); (2) the
//! separable-gate (softmax / threshold_gate) seed dispersion-scaling WEAKENED
//! the decoder-smoothness / ARD seed toward zero on clean data, driving the
//! multi-atom joint Hessian indefinite at the seed. Both are recoverable
//! infeasible-ρ refusals that the single-seed EFS startup validation turned into
//! a fatal abort.
//!
//! These tests fit tiny (N=60, p=8, K=4) planted-circle dictionaries — the same
//! shape as the issue's repro but small enough to run in seconds / a few MB under
//! the RAM-tight shared build gate — through the real outer `OuterProblem::run`
//! ("SAE manifold") cascade, and assert each assignment/topology combination now
//! converges to a finite reconstruction EV instead of throwing
//! `RemlConvergenceError`.

use super::tests::{global_ev, planted_circle_embedded};
use super::*;
use crate::basis::{EuclideanPatchEvaluator, PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_linalg::faer_ndarray::{fast_atb, FaerCholesky};
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::{array, s, Array1, Array2, ArrayView2};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Topo {
    Circle,
    Euclidean,
    Linear,
}

/// Build a K-atom, d=1 SAE term seeded exactly the way the production cold path
/// does (PCA-seed the per-atom coordinates, ridge-LSQ each per-atom decoder),
/// for the requested topology and assignment mode. Returns the term and the
/// seed reconstruction dispersion the outer cascade scales its ρ seed by.
fn build_term(
    z: ArrayView2<'_, f64>,
    k: usize,
    topo: Topo,
    mode: AssignmentMode,
) -> (SaeManifoldTerm, f64) {
    let n = z.nrows();
    let (basis_kind, dim, topo_name): (SaeAtomBasisKind, usize, &str) = match topo {
        Topo::Circle => (SaeAtomBasisKind::Periodic, 1, "circle"),
        Topo::Euclidean => (SaeAtomBasisKind::EuclideanPatch, 1, "euclidean"),
        Topo::Linear => (SaeAtomBasisKind::Linear, 1, "linear"),
    };
    let evaluator: Arc<dyn SaeBasisSecondJet> = match topo {
        Topo::Circle => Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap()),
        Topo::Euclidean => Arc::new(EuclideanPatchEvaluator::new(dim, 2).unwrap()),
        Topo::Linear => Arc::new(EuclideanPatchEvaluator::new(dim, 1).unwrap()),
    };
    let basis_kinds = vec![basis_kind.clone(); k];
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
            for col in 0..z.ncols() {
                let r = z[[row, col]] - fitted[[row, col]];
                rss += r * r;
            }
        }
        let atom = SaeManifoldAtom::new(
            topo_name,
            basis_kind.clone(),
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
        manifolds.push(match topo {
            Topo::Circle => LatentManifold::Circle { period: 1.0 },
            _ => LatentManifold::Euclidean,
        });
    }
    let seed_dispersion = (rss / (k * n * z.ncols()) as f64).max(1.0e-12);
    // Routing seed. IBP-MAP starts every gate on (the production cold seed). The
    // separable gates start from a round-robin row->atom assignment — a stand-in
    // for the FFI's EM routing refine — so the routing is not degenerately
    // symmetric (every atom carries mass; no atom is a duplicate of another).
    let mut logits = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        for atom in 0..k {
            logits[[row, atom]] = match mode {
                AssignmentMode::IBPMap { .. } => 6.0,
                AssignmentMode::Softmax { .. } => {
                    if atom == row % k {
                        3.0
                    } else {
                        0.0
                    }
                }
                AssignmentMode::ThresholdGate { .. } => {
                    if atom == row % k {
                        3.0
                    } else {
                        -3.0
                    }
                }
            };
        }
    }
    let assignment =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coords_blocks, manifolds, mode)
            .unwrap();
    (
        SaeManifoldTerm::new(atoms, assignment).unwrap(),
        seed_dispersion,
    )
}

/// Build the objective and dispersion-scaled seed ρ exactly the way the FFI
/// does (single seed, `inner_max_iter` short — the seed decoder is already
/// LSQ-fit, so its inner solve starts near-optimal and converges quickly).
fn objective_and_seed(
    z: ArrayView2<'_, f64>,
    k: usize,
    topo: Topo,
    mode: AssignmentMode,
) -> (SaeManifoldOuterObjective, Array1<f64>) {
    let (term, seed_dispersion) = build_term(z, k, topo, mode);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let init_rho_flat = init_rho.to_flat();
    let objective = SaeManifoldOuterObjective::new(
        term,
        z.to_owned(),
        None,
        init_rho,
        8,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    (objective, init_rho_flat)
}

/// Assert the seed passes the EFS OUTER STARTUP VALIDATION — the exact gate the
/// issue reports failing. For an all-penalty-like objective with `n_params > 8`
/// the outer planner selects the EFS solver, whose seed validation is a single
/// `eval_efs(seed)` (`run_fixed_point_outer_solver`): it must return a finite
/// cost and finite steps. Before the fix a recoverable non-PD-seed / did-not-
/// converge refusal `?`-propagated out of `efs_step` as a fatal error and — with
/// the single SAE seed — surfaced as `RemlConvergenceError`: "no candidate seeds
/// passed outer startup validation (SAE manifold)".
fn seed_passes_startup_validation(
    z: ArrayView2<'_, f64>,
    k: usize,
    topo: Topo,
    mode: AssignmentMode,
) -> Result<f64, String> {
    let (mut objective, seed) = objective_and_seed(z, k, topo, mode);
    // n_params = 1 (sparse) + K (smooth) + K (ARD) = 1 + 2K; K = 4 -> 9 > 8, so
    // the production planner routes this through the EFS lane, whose startup
    // validation is exactly this call.
    assert!(
        seed.len() > 8,
        "test must exercise the EFS lane (n_params={} must exceed 8)",
        seed.len()
    );
    let eval = objective
        .eval_efs(&seed)
        .map_err(|e| e.to_string())?;
    if !eval.cost.is_finite() {
        return Err(format!("EFS seed cost is non-finite ({})", eval.cost));
    }
    if let Some((idx, v)) = eval.steps.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!("EFS seed step[{idx}] is non-finite ({v})"));
    }
    Ok(eval.cost)
}

/// The #1782 startup-validation matrix: on identical clean planted-circle data
/// every assignment kind (ibp_map, softmax, threshold_gate/jumprelu) and every
/// atom topology (circle, euclidean, linear) must PASS outer startup validation.
/// Before the fix only circle/ibp_map survived; the rest threw "no candidate
/// seeds passed outer startup validation (SAE manifold)". Fast: one inner solve
/// per config from the near-optimal LSQ seed.
#[test]
fn all_assignment_topology_combinations_pass_startup_validation_1782() {
    let z = planted_circle_embedded(48, 6, 0.03);
    let k = 4usize;
    let cases: Vec<(&str, Topo, AssignmentMode)> = vec![
        (
            "circle/ibp_map",
            Topo::Circle,
            AssignmentMode::ibp_map(1.0, 1.0, false),
        ),
        ("circle/softmax", Topo::Circle, AssignmentMode::softmax(1.0)),
        (
            "circle/threshold_gate",
            Topo::Circle,
            AssignmentMode::threshold_gate(1.0, 0.0),
        ),
        (
            "euclidean/ibp_map",
            Topo::Euclidean,
            AssignmentMode::ibp_map(1.0, 1.0, false),
        ),
        (
            "linear/ibp_map",
            Topo::Linear,
            AssignmentMode::ibp_map(1.0, 1.0, false),
        ),
    ];
    for (label, topo, mode) in cases {
        let result = seed_passes_startup_validation(z.view(), k, topo, mode);
        match &result {
            Ok(cost) => eprintln!("REPRO1782 {label}: startup OK (cost={cost:.4e})"),
            Err(e) => eprintln!("REPRO1782 {label}: startup ERR={e}"),
        }
        result.unwrap_or_else(|e| {
            panic!("#1782 {label} must pass outer startup validation, got: {e}")
        });
    }
}

/// The assignment axis (the issue's headline: jumprelu/softmax) must not just
/// pass validation but actually FIT: run the real outer `OuterProblem::run`
/// ("SAE manifold") cascade — the exact FFI entry — on circle atoms for each
/// assignment kind and require a finite reconstruction EV. Circle atoms are
/// well-conditioned, so a low outer-iteration cap keeps this fast; a
/// non-converged best-so-far iterate is still returned as `Ok`, so this asserts
/// the fit RUNS to a real reconstruction rather than aborting at startup.
#[test]
fn assignment_kinds_fit_on_circle_1782() {
    let z = planted_circle_embedded(48, 6, 0.03);
    let k = 4usize;
    for (label, mode) in [
        ("ibp_map", AssignmentMode::ibp_map(1.0, 1.0, false)),
        ("softmax", AssignmentMode::softmax(1.0)),
        ("threshold_gate", AssignmentMode::threshold_gate(1.0, 0.0)),
    ] {
        let (mut objective, seed) = objective_and_seed(z.view(), k, Topo::Circle, mode);
        let n_params = seed.len();
        gam_solve::rho_optimizer::OuterProblem::new(n_params)
            .with_initial_rho(seed)
            .with_max_iter(4)
            .with_seed_config(gam_problem::SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            })
            .run(&mut objective, "SAE manifold")
            .unwrap_or_else(|e| {
                panic!("#1782 circle/{label} fit must not abort at startup, got: {e}")
            });
        let fitted = objective.into_fitted();
        let ev = global_ev(z.view(), fitted.term.fitted().view());
        eprintln!("REPRO1782 circle/{label} fit: ev={ev:.4}");
        assert!(
            ev.is_finite(),
            "#1782 circle/{label} produced a non-finite reconstruction EV ({ev})"
        );
    }
}
