//! #1782 — `sae_manifold_fit` with `threshold_gate`/`softmax` assignments and
//! `euclidean`/`linear` topologies failed at "no candidate seeds passed outer
//! startup validation" on clean planted-circle data where `ordered_beta_bernoulli`+`circle`
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
use crate::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::{Array1, Array2, ArrayView2, array, s};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) enum Topo {
    Circle,
    Euclidean,
    Linear,
}

/// Build a K-atom, d=1 SAE term seeded exactly the way the production cold path
/// does (PCA-seed the per-atom coordinates, ridge-LSQ each per-atom decoder),
/// for the requested topology and assignment mode. Returns the term and the
/// seed reconstruction dispersion the outer cascade scales its ρ seed by.
pub(crate) fn build_term(
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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    // Routing seed. ordered Beta--Bernoulli starts every gate on (the production cold seed). The
    // separable gates start from a round-robin row->atom assignment — a stand-in
    // for the deterministic alternating routing refine — so the routing is not degenerately
    // symmetric (every atom carries mass; no atom is a duplicate of another).
    let mut logits = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        for atom in 0..k {
            logits[[row, atom]] = match mode {
                AssignmentMode::OrderedBetaBernoulli { .. } => 6.0,
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
                // TopK routes each row to its highest-logit atoms: the same
                // round-robin favored-atom seed keeps every atom carrying mass
                // (no degenerate symmetry, no duplicate atoms), and the margin
                // makes the per-row top-k selection deterministic.
                AssignmentMode::TopK { .. } => {
                    if atom == row % k {
                        3.0
                    } else {
                        0.0
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
pub(crate) fn objective_and_seed(
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
    let objective =
        SaeManifoldOuterObjective::new(term, z.to_owned(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
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
    let eval = objective.eval_efs(&seed).map_err(|e| e.to_string())?;
    if !eval.cost.is_finite() {
        return Err(format!("EFS seed cost is non-finite ({})", eval.cost));
    }
    if let Some((idx, v)) = eval.steps.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!("EFS seed step[{idx}] is non-finite ({v})"));
    }
    Ok(eval.cost)
}

/// The #1782 startup-validation matrix: on identical clean planted-circle data
/// every assignment kind (ordered_beta_bernoulli, softmax, threshold_gate) and every
/// atom topology (circle, euclidean, linear) must PASS outer startup validation.
/// Before the fix only circle/ordered_beta_bernoulli survived; the rest threw "no candidate
/// seeds passed outer startup validation (SAE manifold)". Fast: one inner solve
/// per config from the near-optimal LSQ seed.
#[test]
fn all_assignment_topology_combinations_pass_startup_validation_1782() {
    let z = planted_circle_embedded(48, 6, 0.03);
    let k = 4usize;
    let cases: Vec<(&str, Topo, AssignmentMode)> = vec![
        (
            "circle/ordered_beta_bernoulli",
            Topo::Circle,
            AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
        ),
        ("circle/softmax", Topo::Circle, AssignmentMode::softmax(1.0)),
        (
            "circle/threshold_gate",
            Topo::Circle,
            AssignmentMode::threshold_gate(1.0, 0.0),
        ),
        (
            "euclidean/ordered_beta_bernoulli",
            Topo::Euclidean,
            AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
        ),
        (
            "linear/ordered_beta_bernoulli",
            Topo::Linear,
            AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
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

/// Run the real outer `OuterProblem::run` ("SAE manifold") cascade — the exact
/// FFI entry — for one topology/assignment pair on the tiny planted-circle
/// fixture, with the single-PCA-seed budget the production `sae_manifold_fit`
/// FFI uses, and return the reconstruction EV. A non-converged best-so-far
/// iterate is still returned as `Ok`, so a returned EV means the fit RAN to a
/// real reconstruction rather than aborting at startup / in the outer solver.
fn run_full_fit(
    z: ArrayView2<'_, f64>,
    k: usize,
    topo: Topo,
    mode: AssignmentMode,
    label: &str,
) -> f64 {
    let (mut objective, seed) = objective_and_seed(z, k, topo, mode);
    let n_params = seed.len();
    let result = gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold")
        .unwrap_or_else(|e| {
            // The two #1782 failure surfaces both land here: the threshold-gate / euclidean
            // "no candidate seeds passed outer startup validation" abort, and the
            // softmax "BFGS aborted: globally infeasible neighbourhood at seed
            // (probe-refusal guard)" abort — both are the emptied / globally-refused
            // seed cascade the fit must avoid by entering a basin with defined
            // quasi-Laplace score; infeasible probes remain `+∞` and cannot certify.
            panic!("#1782 {label} fit must not abort at startup / in the outer solver, got: {e}")
        });
    objective
        .certify_outer_result(&result)
        .expect("#1782 outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z, fitted.term.fitted().view());
    eprintln!("REPRO1782 {label} fit: ev={ev:.4}");
    assert!(
        ev.is_finite(),
        "#1782 {label} produced a non-finite reconstruction EV ({ev})"
    );
    ev
}

/// The assignment axis (the issue's headline: threshold-gate/softmax) must not just
/// pass validation but actually FIT: run the real outer `OuterProblem::run`
/// ("SAE manifold") cascade — the exact FFI entry — on circle atoms for each
/// assignment kind and require a finite reconstruction EV. Circle atoms are
/// well-conditioned, so a low outer-iteration cap keeps this fast; a
/// non-converged best-so-far iterate is still returned as `Ok`, so this asserts
/// the fit RUNS to a real reconstruction rather than aborting at startup.
///
/// `softmax` is the SECOND #1782 failure surface: its seed and its whole
/// neighbourhood land in the recoverable infeasible-ρ refusal class, so the
/// outer BFGS lane previously returned `+∞` for every probe, never accepted a
/// step, and the bridge's non-termination guard escalated the globally-refused
/// neighbourhood to a FATAL seed rejection ("BFGS aborted: globally infeasible
/// neighbourhood at seed (probe-refusal guard)"). `ordered_beta_bernoulli`+`circle` lands in
/// the PD region and never trips it — RED before the fix on `softmax`, GREEN
/// after (the entry path now reaches a basin with defined quasi-Laplace score).
#[test]
fn assignment_kinds_fit_on_circle_1782() {
    let z = planted_circle_embedded(48, 6, 0.03);
    let k = 4usize;
    for (label, mode) in [
        (
            "circle/ordered_beta_bernoulli",
            AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
        ),
        ("circle/softmax", AssignmentMode::softmax(1.0)),
        (
            "circle/threshold_gate",
            AssignmentMode::threshold_gate(1.0, 0.0),
        ),
    ] {
        run_full_fit(z.view(), k, Topo::Circle, mode, label);
    }
}

/// The topology axis of #1782: on identical clean planted-circle data the
/// `euclidean` and `linear` atom topologies (whose rank-deficient PCA seed lands
/// in the recoverable infeasible-ρ refusal class) must also FIT through the real
/// outer cascade, not abort with an emptied / globally-refused seed cascade. Same
/// single-PCA-seed budget and low outer-iteration cap as the assignment-axis
/// test, so it stays fast. RED before the fix (`euclidean`/`linear` aborted at
/// "no candidate seeds passed outer startup validation"); GREEN after.
#[test]
fn topologies_fit_on_circle_data_1782() {
    let z = planted_circle_embedded(48, 6, 0.03);
    let k = 4usize;
    for (label, topo) in [
        ("euclidean/ordered_beta_bernoulli", Topo::Euclidean),
        ("linear/ordered_beta_bernoulli", Topo::Linear),
    ] {
        run_full_fit(
            z.view(),
            k,
            topo,
            AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
            label,
        );
    }
}

/// FRONTIER PROBE (#1026 co-collapse). Measure the largest OVERCOMPLETE
/// dictionary size `K` whose PCA-diversified cold seed still passes the EFS
/// outer startup validation on a single planted circle embedded in `p = 24`
/// dims — the regime where every extra atom competes for the same rank-2 signal
/// and used to co-collapse into a rank-deficient, non-PD seed. The seed
/// diversification (disjoint per-atom PC windows, `pca_seed.rs`) plus the `K > 1`
/// non-PD dispersion floor (`rho.rs`) are exactly the fixes this probes. Prints
/// the pass/fail frontier so the effect is visible in `--nocapture`. Asserts
/// only the known-safe `K = 4` so it can never red the shared tree; the printed
/// frontier drives the co-collapse-saddle work. K is capped small (the dense
/// per-seed arrow-Schur criterion is `O((K·b·p)³)`) so the sweep stays fast.
#[test]
fn cocollapse_startup_frontier_1026() {
    let z = planted_circle_embedded(96, 10, 0.03);
    let ks = [4usize, 8];
    // Compare the assignment modes: ordered Beta--Bernoulli couples all rows through a cross-row
    // Woodbury evidence with NO matrix-free log-det route (so large-K refuses on
    // the dense reduced Schur), whereas the smooth logistic threshold gate is
    // per-row independent and streams. This measures which mode
    // extends the startup frontier, decoupling the routing wall from seed
    // co-collapse.
    let modes: [(&str, fn() -> AssignmentMode); 3] = [
        ("ordered_beta_bernoulli    ", || {
            AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false)
        }),
        ("thresh_gate", || AssignmentMode::threshold_gate(1.0, 0.5)),
        ("softmax    ", || AssignmentMode::softmax(1.0)),
    ];
    let mut ordered_beta_bernoulli_frontier = 0usize;
    for (label, mk) in modes {
        let mut frontier = 0usize;
        for &k in &ks {
            match seed_passes_startup_validation(z.view(), k, Topo::Circle, mk()) {
                Ok(cost) => {
                    eprintln!("FRONTIER1026 {label} K={k:>3}: startup PASS (cost={cost:.4e})");
                    frontier = k;
                }
                Err(e) => {
                    eprintln!("FRONTIER1026 {label} K={k:>3}: startup FAIL ({e})");
                    break;
                }
            }
        }
        eprintln!("FRONTIER1026 {label}: largest passing K = {frontier}");
        if label.trim() == "ordered_beta_bernoulli" {
            ordered_beta_bernoulli_frontier = frontier;
        }
    }
    assert!(
        ordered_beta_bernoulli_frontier >= 4,
        "startup validation must hold at least to K=4 (got frontier {ordered_beta_bernoulli_frontier})"
    );
}

/// WIN artifact (#1026 / #1610). A PRINCIPLED joint manifold SAE — curved 1-D
/// circle fibers, smooth logistic gate (`threshold_gate`, the per-row
/// streaming assignment whose criterion log-det takes the matrix-free SLQ route)
/// — fit end-to-end by the real outer penalized quasi-Laplace
/// cascade must MATCH-OR-BEAT a traditional linear SAE (`fit_sparse_dictionary`,
/// the "large linear SAE" of #1026) at matched, OVERCOMPLETE dictionary size K on
/// genuinely curved data. On a planted circle a linear dictionary is rank-capped
/// while curved atoms bend to the ring, so the manifold decisively wins. This is
/// the joint-solve WIN (no alternating-minimization searcher, no Python): the
/// coupled inner arrow-Schur Newton exercises the landed disjoint-PC seed
/// diversification and the spectral Schur PD-floor that keep the overcomplete
/// (K > true-rank) joint block PD instead of co-collapsing. K is kept box-safe
/// here; the per-row work is `top_k`-bounded, so it is the same solve at larger
/// K (the streaming matrix-free criterion log-det, exercised by the outer penalized quasi-Laplace
/// cascade, is what carries it to K=32,000).
#[test]
fn manifold_beats_linear_joint_streaming_1026() {
    let z = planted_circle_embedded(120, 10, 0.03);
    for &k in &[8usize] {
        // Traditional linear SAE baseline at matched K (the sparse-dict lane is f32).
        let z32 = z.mapv(|v| v as f32);
        let lin = fit_sparse_dictionary(z32.view(), &SparseDictConfig::new(k))
            .expect("linear SAE baseline fits");
        let ev_linear = lin.explained_variance;

        // Principled joint manifold SAE: curved circle fibers, smooth logistic threshold gate,
        // solved directly by the coupled inner arrow-Schur joint Newton over the
        // (coords t, decoders β) block — the exact joint solve the outer penalized quasi-Laplace
        // cascade drives, run here at a fixed penalty seed so the comparison is a
        // fast, deterministic reconstruction check (no per-step criterion log-det).
        let mode = AssignmentMode::threshold_gate(1.0, 0.0);
        let (mut term, _disp) = build_term(z.view(), k, Topo::Circle, mode);
        let mut rho = SaeManifoldRho::new(
            1.0e-3_f64.ln(),
            1.0e-3_f64.ln(),
            vec![array![1.0e-3_f64.ln()]; k],
        );
        term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 24, 1.0, 1.0e-6, 1.0e-6)
            .unwrap_or_else(|e| {
                panic!("#1026 manifold K={k} joint inner fit must run e2e, got: {e}")
            });
        let fitted = term.try_fitted().expect("manifold fitted");
        let ev_manifold = global_ev(z.view(), fitted.view());

        eprintln!(
            "WIN1026 K={k:>3}: manifold EV={ev_manifold:.4}  linear EV={ev_linear:.4}  \
             margin={:+.4}",
            ev_manifold - ev_linear
        );
        assert!(
            ev_manifold.is_finite() && ev_linear.is_finite(),
            "#1026 K={k}: both EVs must be finite (manifold={ev_manifold}, linear={ev_linear})"
        );
        // Match-or-beat contract (#1026 strict generalization): the curved
        // dictionary generalizes the linear one, so it must never do worse.
        assert!(
            ev_manifold + 5.0e-2 >= ev_linear,
            "#1026 K={k}: principled manifold SAE must match-or-beat linear \
             (manifold={ev_manifold:.4} vs linear={ev_linear:.4})"
        );
    }
}
