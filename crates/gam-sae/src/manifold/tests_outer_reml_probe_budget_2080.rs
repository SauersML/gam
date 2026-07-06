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
use gam_solve::rho_optimizer::{OuterObjective, OuterProblem};
use ndarray::{Array1, Array2, ArrayView2, array, s};
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

/// A single centered circle embedded in `p` standardized ambient channels. This
/// is the cheap K=1 #2153 regression target; unlike the two-circle fixture, the
/// model is correctly specified, so any long Strong-Wolfe probe train is solver
/// pathology rather than target mismatch.
fn one_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
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
            z[[row, j]] =
                c * frame[[0, j]] + s * frame[[1, j]] + sigma * deterministic_circle_noise(row, j + 7);
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
    let z = two_circle_wide_target(n, p, 0.03);
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

/// Same full outer path as `run_wide_outer_fit`, but intentionally starts from
/// the generated seed rather than pinning `initial_rho`. This is the K=1 cold
/// path that #2153 exposed: with the optimizer's raw identity iter-0 metric, the
/// first `-g` step can be orders too large, so Strong-Wolfe spends full
/// value-probe solves backtracking instead of making progress.
fn run_k1_generated_seed_outer_fit(
    n: usize,
    p: usize,
    harmonics: usize,
) -> (f64, OuterProbeTelemetry) {
    let z = one_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let n_params = init_rho.to_flat().len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    OuterProblem::new(n_params)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold K=1 generated seed")
        .expect("#2153 K=1 generated-seed circle fit must terminate");
    let telemetry = objective.probe_telemetry();
    let fitted = objective.into_fitted();
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    (ev, telemetry)
}

#[derive(Clone, Copy, Debug)]
struct CeilingPathologyConfig {
    n: usize,
    p: usize,
    harmonics: usize,
    sigma: f64,
    inner_max_iter: usize,
    outer_max_iter: usize,
    initial_step_norm: f64,
    materialization_ratio_floor: f64,
    step_collapse_radius: f64,
    huge_final_gradient_floor: f64,
    pin_initial_rho: bool,
}

impl Default for CeilingPathologyConfig {
    fn default() -> Self {
        Self {
            n: 96,
            p: 96,
            harmonics: 2,
            sigma: 0.05,
            inner_max_iter: 8,
            outer_max_iter: 8,
            initial_step_norm: 0.25,
            materialization_ratio_floor: 0.05,
            step_collapse_radius: 1.0e-3,
            huge_final_gradient_floor: 10.0,
            pin_initial_rho: true,
        }
    }
}

#[derive(Clone, Debug)]
struct CeilingPathologyReport {
    initial_cost: f64,
    initial_grad_norm: f64,
    predicted_decrease: f64,
    actual_decrease: f64,
    materialization_ratio: f64,
    outer_converged: bool,
    outer_iterations: usize,
    final_value: f64,
    final_grad_norm: f64,
    rho_displacement: f64,
    ev: f64,
    telemetry: OuterProbeTelemetry,
    outer_error: Option<String>,
    predicted_decrease_not_materializing: bool,
    step_collapsed: bool,
    huge_final_gradient: bool,
    live_lock_present: bool,
}

fn l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn seeded_k1_circle_objective(
    cfg: CeilingPathologyConfig,
) -> (Array2<f64>, SaeManifoldRho, Array1<f64>, SaeManifoldOuterObjective) {
    let z = one_circle_wide_target(cfg.n, cfg.p, cfg.sigma);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, cfg.harmonics);
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let seed = init_rho.to_flat();
    let objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho.clone(),
        cfg.inner_max_iter,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    (z, init_rho, seed, objective)
}

/// CEILING-VS-PATHOLOGY decisive instrument (#2156): run the mid-scale curved
/// OUTER-REML fit and classify the line-search live-lock signature.
///
/// The first-step model check is intentionally independent of the optimizer's
/// private line-search trace: it measures whether the initial analytic descent
/// direction predicts a material REML decrease that the value path does not
/// realize. The full outer run then reports the operational symptoms that make
/// this a solver pathology rather than an information ceiling: collapsed
/// accepted ρ displacement and a large final gradient.
fn run_ceiling_vs_pathology_instrument(
    cfg: CeilingPathologyConfig,
) -> CeilingPathologyReport {
    let probe_seeded = seeded_k1_circle_objective(cfg);
    let seed_probe = probe_seeded.2;
    let mut probe_objective = probe_seeded.3;
    let initial = OuterObjective::eval(&mut probe_objective, &seed_probe)
        .expect("#2156 instrument initial REML gradient eval must complete");
    let initial_grad_norm = l2_norm(&initial.gradient);
    let mut trial = seed_probe.clone();
    if initial_grad_norm.is_finite() && initial_grad_norm > 0.0 {
        let scale = cfg.initial_step_norm / initial_grad_norm;
        for idx in 0..trial.len() {
            trial[idx] -= scale * initial.gradient[idx];
        }
    }
    let predicted_decrease = if initial_grad_norm.is_finite() {
        cfg.initial_step_norm * initial_grad_norm
    } else {
        f64::NAN
    };
    let trial_cost = OuterObjective::eval_cost(&mut probe_objective, &trial)
        .expect("#2156 instrument initial REML value probe must complete");
    let actual_decrease = initial.cost - trial_cost;
    let materialization_ratio = if predicted_decrease.is_finite()
        && predicted_decrease > f64::MIN_POSITIVE
        && actual_decrease.is_finite()
    {
        actual_decrease / predicted_decrease
    } else {
        f64::NAN
    };
    let predicted_decrease_not_materializing =
        materialization_ratio.is_finite()
            && materialization_ratio < cfg.materialization_ratio_floor;

    let fit_seeded = seeded_k1_circle_objective(cfg);
    let z = fit_seeded.0;
    let seed = fit_seeded.2;
    let mut objective = fit_seeded.3;
    let n_params = seed.len();
    let mut problem = OuterProblem::new(n_params)
        .with_max_iter(cfg.outer_max_iter)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    if cfg.pin_initial_rho {
        problem = problem.with_initial_rho(seed.clone());
    }
    let run = problem.run(&mut objective, "SAE manifold ceiling-vs-pathology #2156");
    let telemetry = objective.probe_telemetry();
    match run {
        Ok(result) => {
            let final_grad_norm = result.final_grad_norm.unwrap_or(f64::NAN);
            let rho_displacement = l2_norm(&(&result.rho - &seed));
            let step_collapsed =
                rho_displacement.is_finite() && rho_displacement <= cfg.step_collapse_radius;
            let huge_final_gradient = final_grad_norm.is_finite()
                && final_grad_norm >= cfg.huge_final_gradient_floor;
            let fitted = objective.into_fitted();
            let ev = global_ev(z.view(), fitted.term.fitted().view());
            let live_lock_present =
                predicted_decrease_not_materializing && step_collapsed && huge_final_gradient;
            CeilingPathologyReport {
                initial_cost: initial.cost,
                initial_grad_norm,
                predicted_decrease,
                actual_decrease,
                materialization_ratio,
                outer_converged: result.converged,
                outer_iterations: result.iterations,
                final_value: result.final_value,
                final_grad_norm,
                rho_displacement,
                ev,
                telemetry,
                outer_error: None,
                predicted_decrease_not_materializing,
                step_collapsed,
                huge_final_gradient,
                live_lock_present,
            }
        }
        Err(err) => CeilingPathologyReport {
            initial_cost: initial.cost,
            initial_grad_norm,
            predicted_decrease,
            actual_decrease,
            materialization_ratio,
            outer_converged: false,
            outer_iterations: 0,
            final_value: f64::NAN,
            final_grad_norm: f64::NAN,
            rho_displacement: f64::NAN,
            ev: f64::NAN,
            telemetry,
            outer_error: Some(err.to_string()),
            predicted_decrease_not_materializing,
            step_collapsed: false,
            huge_final_gradient: false,
            live_lock_present: true,
        },
    }
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

/// #2153 — K=1 manifold fits must not live-lock in Strong-Wolfe line search from
/// a cold generated seed. The regression is a probe-count assertion, not a
/// wall-clock deadline: the first BFGS step is normalized by the seed gradient
/// norm, so the line search should accept a bounded step instead of spending
/// repeated full inner solves on rejected value probes.
#[test]
fn k1_generated_seed_circle_outer_reml_does_not_livelock_2153() {
    let (ev, telemetry) = run_k1_generated_seed_outer_fit(32, 24, 1);
    eprintln!(
        "[#2153] K=1 generated-seed outer fit: ev={ev:.4}, criterion_calls={}, \
         fd_probe_calls={}, wall_cost_value_probes={}, infeasible_total={}, \
         mutating_value_probes={}",
        telemetry.criterion_calls,
        telemetry.fd_probe_calls,
        telemetry.wall_cost_value_probes,
        telemetry.infeasible_total(),
        telemetry.mutating_value_probes,
    );
    assert!(
        telemetry.criterion_calls <= 32,
        "#2153 K=1 generated-seed fit issued {} criterion calls; expected a \
         bounded first-line-search probe budget",
        telemetry.criterion_calls
    );
    assert_eq!(
        telemetry.mutating_value_probes, 0,
        "#2153 line-search value probes must not mutate the accepted term basin"
    );
    assert!(
        ev.is_finite() && ev > 0.30,
        "#2153 K=1 generated-seed circle fit must converge to a real positive-EV \
         basin (got {ev:.4})"
    );
}

/// Expert Test 1+2 / #2156 — decide "solver pathology" versus "real ceiling".
///
/// Before the adjoint-gradient fix, this fixture reports the combined pathology:
/// the analytic descent model predicts a material decrease, the value path does
/// not realize it, the accepted outer iterate barely moves, and the final
/// gradient remains large. After the fix the same source-level instrument should
/// report `live_lock_present=false`; the top-M envelope report then tells the
/// reader whether any remaining low curved EV is an information ceiling.
#[test]
fn ceiling_vs_pathology_outer_reml_instrument_2156() {
    let report = run_ceiling_vs_pathology_instrument(CeilingPathologyConfig::default());
    eprintln!(
        "[#2156 ceiling-vs-pathology] initial_cost={:.6e}, initial_grad_norm={:.6e}, \
         predicted_decrease={:.6e}, actual_decrease={:.6e}, materialization_ratio={:.6e}, \
         outer_converged={}, outer_iterations={}, final_value={:.6e}, final_grad_norm={:.6e}, \
         rho_displacement={:.6e}, ev={:.4}, criterion_calls={}, fd_probe_calls={}, \
         wall_cost_value_probes={}, infeasible_total={}, outer_error={:?}, \
         predicted_not_materializing={}, step_collapsed={}, huge_final_gradient={}, \
         live_lock_present={}",
        report.initial_cost,
        report.initial_grad_norm,
        report.predicted_decrease,
        report.actual_decrease,
        report.materialization_ratio,
        report.outer_converged,
        report.outer_iterations,
        report.final_value,
        report.final_grad_norm,
        report.rho_displacement,
        report.ev,
        report.telemetry.criterion_calls,
        report.telemetry.fd_probe_calls,
        report.telemetry.wall_cost_value_probes,
        report.telemetry.infeasible_total(),
        report.outer_error,
        report.predicted_decrease_not_materializing,
        report.step_collapsed,
        report.huge_final_gradient,
        report.live_lock_present,
    );
    assert!(
        !report.live_lock_present,
        "#2156 CEILING-vs-PATHOLOGY instrument detected the live-lock signature: \
         predicted decrease did not materialize, accepted ρ step collapsed, and \
         final gradient stayed huge; report={report:?}"
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

/// gamfit#2138 (fit-robustness half) — the curved (periodic) atom's inner
/// Newton/arrow-Schur solve must CONVERGE on a small (`n = 35`) fold at high
/// working rank (`m = 9`, all harmonics genuinely excited, so no rank reduction
/// collapses the design), across the whole smoothing sweep — including the
/// over-smoothed tail where the undamped Laplace log-det system is worst
/// conditioned. Before the robustness work an ill-conditioned small-`n`
/// high-rank probe `ρ` could grind the inner refinement budget and surface a
/// `RemlConvergenceError` (the theory experiments had to work around it with a
/// lower rank + fixed smoothing); the inner joint fit's Armijo line search +
/// proximal-correction LM ridge escalation keeps every step a descent step, so
/// the fit reaches a finite, materially-positive-EV basin at every ρ instead of
/// diverging. Each fixed-ρ evaluation must return `Ok` with a finite REML cost
/// and (for the feasible low-to-mid smoothing range) a materially positive EV.
#[test]
fn small_fold_high_rank_circle_inner_solve_converges_2138() {
    let n = 35usize;
    let p = 12usize;
    let harmonics = 4usize;
    let m = 1 + 2 * harmonics;
    let mut frames = Array2::<f64>::zeros((2 * harmonics, p));
    for h in 0..2 * harmonics {
        for j in 0..p {
            frames[[h, j]] = deterministic_circle_noise(h, j);
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let t = row as f64 / n as f64;
        for hh in 0..harmonics {
            let ang = std::f64::consts::TAU * (hh as f64 + 1.0) * t;
            let (c, s) = (ang.cos(), ang.sin());
            for j in 0..p {
                z[[row, j]] += c * frames[[2 * hh, j]] + s * frames[[2 * hh + 1, j]];
            }
        }
    }
    for j in 0..p {
        let mean: f64 = (0..n).map(|r| z[[r, j]]).sum::<f64>() / n as f64;
        let var: f64 = (0..n).map(|r| (z[[r, j]] - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt().max(1.0e-12);
        for r in 0..n {
            z[[r, j]] = (z[[r, j]] - mean) / sd;
        }
    }
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let seed_coords =
        sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1]).unwrap();
    let coords = seed_coords.slice(s![0, .., 0..1]).to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut xtx = fast_atb(&phi, &phi);
    for i in 0..m {
        xtx[[i, i]] += 1.0e-8;
    }
    let xtz = fast_atb(&phi, &z);
    let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n, 1), 6.0),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(1.0, 1.0, false),
    )
    .unwrap();
    let base = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // The whole smoothing sweep, from flexible (-8) through the over-smoothed tail
    // (+8) where the undamped Laplace log-det is worst conditioned.
    for &smooth in &[-8.0_f64, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0] {
        let mut t = base.clone();
        let r = SaeManifoldRho::new(0.02_f64.ln(), smooth, vec![array![0.0]]);
        let evaluated = t
            .reml_criterion_with_cache(z.view(), &r, None, 60, 0.04, 1.0e-6, 1.0e-6)
            .unwrap_or_else(|err| {
                panic!(
                    "#2138: high-working-rank (m={m}) circle inner solve must converge at a \
                     small (n={n}) fold, smoothing={smooth}, not diverge into a \
                     RemlConvergenceError; got: {err}"
                )
            });
        let cost = evaluated.0;
        let ev = global_ev(z.view(), t.fitted().view());
        // The design is full working rank (all m columns excited): no rank
        // reduction should collapse it, so the ill-conditioned regime is real.
        assert_eq!(
            t.atoms[0].basis_size(),
            m,
            "#2138: the multi-harmonic target must keep the atom at full working rank m={m}",
        );
        assert!(
            cost.is_finite(),
            "#2138: inner solve returned a non-finite REML cost at smoothing={smooth}",
        );
        assert!(
            ev.is_finite() && ev > 0.30,
            "#2138: high-working-rank small-fold circle fit must recover a materially positive \
             EV at smoothing={smooth} (got {ev:.4}), proving the inner solve reached a real \
             basin rather than a diverged / collapsed state",
        );
    }
}
