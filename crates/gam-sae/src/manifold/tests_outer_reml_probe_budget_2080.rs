//! #2080 — the OUTER PENALIZED-LAML ρ-search must terminate in a BOUNDED number of
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
use gam_solve::rho_optimizer::{OuterEvalOrder, OuterObjective, OuterProblem};
use ndarray::{Array1, Array2, ArrayView2, array, s};
use std::sync::Arc;

/// #2080 — two atoms may each have a full-rank decoder design while their
/// CONCATENATED design is rank-deficient. With identical weighted constant
/// columns, `δB₀ = c, δB₁ = −c` leaves the reconstruction exactly unchanged;
/// an atom-local `G_k + λS_k` audit sees two positive scalar Grams and misses
/// this coupled redistribution gauge entirely.
#[test]
fn joint_decoder_gauge_quotients_full_rank_atom_redistribution_2080() -> Result<(), String> {
    let n = 4usize;
    let phi = Array2::<f64>::ones((n, 1));
    let jet = ndarray::Array3::<f64>::zeros((n, 1, 1));
    let make_atom = |name: &str, decoder: f64| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Linear,
            1,
            phi.clone(),
            jet.clone(),
            array![[decoder]],
            Array2::<f64>::zeros((1, 1)),
        )
    };
    let coords = Array2::<f64>::zeros((n, 1));
    let assignment = SaeAssignment::from_blocks_with_mode(
        Array2::<f64>::zeros((n, 2)),
        vec![coords.clone(), coords],
        AssignmentMode::softmax(1.0),
    )?;
    let term = SaeManifoldTerm::new(
        vec![make_atom("shared-a", 1.0)?, make_atom("shared-b", -0.5)?],
        assignment,
    )?;

    // Both atom-local weighted designs are individually rank one (their full
    // possible rank), so the superseded per-atom eigensolves have no null.
    let weights = term.assignment.assignments();
    for atom_idx in 0..2 {
        let gram = (0..n)
            .map(|row| {
                let value = weights[[row, atom_idx]] * phi[[row, 0]];
                value * value
            })
            .sum::<f64>();
        assert!(
            gram > 0.0,
            "atom {atom_idx} must have a full-rank scalar Gram"
        );
    }

    let gauges = term.joint_decoder_beta_null_directions(&[0.0, 0.0])?;
    assert_eq!(
        gauges.len(),
        1,
        "the two identical scalar designs have exactly one coupled decoder gauge"
    );
    let coord_dim = n * term.assignment.row_block_dim();
    let delta_t = Array1::<f64>::zeros(coord_dim);
    let delta_beta = array![1.0_f64, -1.0];
    let raw = delta_beta.dot(&delta_beta);
    let quotient =
        term.quotient_newton_step_norm_sq(delta_t.view(), delta_beta.view(), raw, &[0.0, 0.0])?;
    assert!(
        quotient <= f64::EPSILON * (1.0 + raw),
        "joint redistribution must vanish on the identified quotient; raw={raw:.3e}, quotient={quotient:.3e}"
    );
    Ok(())
}

/// Two planted circles on DISJOINT ambient column parities (circle A on the even
/// output channels, circle B on the odd), driven by two independent phases on an
/// exact Cartesian product grid and per-column standardized. Together they span a
/// rank-4 subspace of the whitened `p`-dim cloud, so an honest K=2 dictionary
/// explains a materially positive fraction of the variance. `p` is the wide-`p`
/// knob that drives the outer hang.
pub(super) fn independent_two_circle_phases(n: usize, row: usize) -> (f64, f64) {
    let mut n1 = 1usize;
    let root = (n as f64).sqrt() as usize;
    for d in 1..=root.max(1) {
        if n % d == 0 {
            n1 = d;
        }
    }
    let n2 = n / n1.max(1);
    assert!(
        n1 > 1 && n2 > 1,
        "two-circle fixture needs a nontrivial Cartesian phase grid, got {n1}x{n2}"
    );
    let i = row % n1;
    let j = (row / n1) % n2;
    (
        std::f64::consts::TAU * (i as f64) / (n1 as f64),
        std::f64::consts::TAU * (j as f64) / (n2 as f64),
    )
}

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
        let (ta, tb) = independent_two_circle_phases(n, row);
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

/// Build a K-atom, d=1 periodic SAE term seeded the way the production cold path
/// does (PCA-seed the per-atom coordinates, ridge-LSQ each per-atom decoder), with
/// ordered Beta--Bernoulli-MAP assignment. Returns the term and the seed reconstruction dispersion the
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
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let assignment =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coords_blocks, manifolds, mode)
            .unwrap();
    (
        SaeManifoldTerm::new(atoms, assignment).unwrap(),
        seed_dispersion,
    )
}

/// #2080 — a reactive legal entry must replace a nonzero but invalid cold
/// dictionary with a separated data-derived basin before asking for evidence.
/// The old zero-decoder gate skipped this production-shaped seed because both
/// independently fitted decoders had material norm; the fixed-ρ entry then spent
/// its whole refinement budget unwinding their double fit and never reached KKT.
#[test]
fn reactive_entry_reseeds_nonzero_k2_seed_to_strict_separated_root_2080() {
    let z = two_circle_wide_target(48, 24, 0.03);
    let k = 2usize;
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), k, 2);
    let seed_norms: Vec<f64> = term
        .atoms
        .iter()
        .map(|atom| {
            atom.decoder_coefficients
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    assert!(
        seed_norms.iter().all(|norm| norm.is_finite() && *norm > 0.0),
        "regression requires the nonzero decoder seed that bypassed the old cold-entry placement; norms={seed_norms:?}"
    );

    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let contract = OuterObjective::reactive_domain_scalar_contract(&objective)
        .expect("reactive scalar contract query")
        .expect("dense K=2 objective must own a reactive scalar contract");
    let entry_rho = OuterObjective::outer_domain_upper_bound(&objective)
        .expect("reactive rho entry query")
        .expect("dense K=2 objective must own a reactive rho entry");

    OuterObjective::begin_reactive_domain_waypoint(&mut objective)
        .expect("entry transaction must begin");
    OuterObjective::install_reactive_domain_scalar_state(&mut objective, contract.entry())
        .expect("separated legal entry must install");

    // The target's two factors occupy disjoint output parities. The objective-
    // owned entry placement may swap atom labels, but the two decoders must
    // specialize to opposite parities instead of both carrying the full target.
    let mut even_dominant = Vec::with_capacity(k);
    for atom in &objective.term.atoms {
        let mut even_energy = 0.0_f64;
        let mut odd_energy = 0.0_f64;
        for ((_, output), value) in atom.decoder_coefficients.indexed_iter() {
            if output % 2 == 0 {
                even_energy += value * value;
            } else {
                odd_energy += value * value;
            }
        }
        assert!(
            even_energy.is_finite()
                && odd_energy.is_finite()
                && (even_energy > 0.0 || odd_energy > 0.0)
                && even_energy != odd_energy,
            "entry decoder must carry a finite, parity-identifiable factor; even={even_energy:.6e}, odd={odd_energy:.6e}"
        );
        even_dominant.push(even_energy > odd_energy);
    }
    assert_ne!(
        even_dominant[0], even_dominant[1],
        "entry placement must put the two planted factors on distinct atoms; parity dominance={even_dominant:?}"
    );

    let entry_eval_result = OuterObjective::eval_with_order(
        &mut objective,
        &entry_rho,
        OuterEvalOrder::Value,
    );
    if let Err(error) = &entry_eval_result {
        let entry_rho_state = objective.baseline_rho.from_flat(entry_rho.view());
        let system = objective
            .term
            .assemble_arrow_schur(z.view(), &entry_rho_state, None)
            .expect("failed-entry KKT diagnostic assembly");
        let assignment_dim = objective.term.assignment.assignment_coord_dim();
        let mut assignment_grad_sq = 0.0_f64;
        let mut chart_grad_sq = 0.0_f64;
        for row in &system.rows {
            for (index, value) in row.gt.iter().enumerate() {
                if index < assignment_dim {
                    assignment_grad_sq += value * value;
                } else {
                    chart_grad_sq += value * value;
                }
            }
        }
        let decoder_grad = system.gb.iter().map(|value| value * value).sum::<f64>().sqrt();
        let assignments = objective
            .term
            .assignment
            .try_assignments()
            .expect("failed-entry assignment diagnostic");
        let (assignment_min, assignment_max) = assignments.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(*value), maximum.max(*value)),
        );
        let decoder_norms: Vec<f64> = objective
            .term
            .atoms
            .iter()
            .map(|atom| {
                atom.decoder_coefficients
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        eprintln!(
            "[#2080 reactive-entry KKT] error={error}; assignment_grad={:.6e}, \
             chart_grad={:.6e}, decoder_grad={decoder_grad:.6e}, \
             assignment_range=[{assignment_min:.6e},{assignment_max:.6e}], \
             decoder_norms={decoder_norms:?}, lambda_smooth={:?}",
            assignment_grad_sq.sqrt(),
            chart_grad_sq.sqrt(),
            entry_rho_state.lambda_smooth_vec(),
        );
    }
    let entry_eval = entry_eval_result.expect("separated legal entry must solve to finite evidence");
    assert!(
        entry_eval.cost.is_finite(),
        "separated legal entry returned non-finite evidence {}",
        entry_eval.cost
    );
    OuterObjective::commit_reactive_domain_waypoint(&mut objective, &entry_rho)
        .expect("finite entry must commit its full converged state");

    // Reassemble the exact committed entry and independently recheck the same
    // strict raw-or-quotient KKT predicate that authorizes penalized LAML evidence. A
    // finite value alone cannot satisfy this regression.
    let committed_rho = objective.current_rho.clone();
    let system = objective
        .term
        .assemble_arrow_schur(z.view(), &committed_rho, None)
        .expect("committed entry KKT assembly");
    let raw_kkt_sq = SaeManifoldTerm::system_grad_norm_sq(&system);
    let raw_kkt = raw_kkt_sq.sqrt();
    let quotient_kkt = objective.term.quotient_gradient_norm_from_system(
        &system,
        raw_kkt_sq,
        &committed_rho.lambda_smooth_vec(),
    );
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * objective.term.inner_iterate_scale();
    assert!(
        SaeManifoldTerm::evidence_kkt_stationary(raw_kkt, quotient_kkt, tolerance),
        "committed legal entry is not a strict envelope root: raw KKT={raw_kkt:.6e}, quotient KKT={quotient_kkt:.6e}, tolerance={tolerance:.6e}"
    );
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
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let seed = init_rho.to_flat();
    let n_params = seed.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold")
        .expect("#2080 wide-p outer penalized-LAML fit must terminate, not hang / abort");
    assert!(
        result.converged,
        "#2080 wide-p acceptance requires a CONVERGED outer penalized-LAML optimum, not a \
         finite max-iteration/line-search incumbent: iterations={}, final_value={:.6e}, \
         final_grad_norm={:?}",
        result.iterations, result.final_value, result.final_grad_norm,
    );
    let telemetry = objective.probe_telemetry();
    objective
        .certify_outer_result(&result)
        .expect("#2080 wide-p outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
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
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let n_params = init_rho.to_flat().len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold K=1 generated seed")
        .expect("#2153 K=1 generated-seed circle fit must terminate");
    assert!(
        result.converged,
        "#2153 K=1 acceptance requires a converged outer optimum: iterations={}, \
         final_value={:.6e}, final_grad_norm={:?}",
        result.iterations, result.final_value, result.final_grad_norm,
    );
    let telemetry = objective.probe_telemetry();
    objective
        .certify_outer_result(&result)
        .expect("#2153 outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
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
) -> (
    Array2<f64>,
    SaeManifoldRho,
    Array1<f64>,
    SaeManifoldOuterObjective,
) {
    let z = one_circle_wide_target(cfg.n, cfg.p, cfg.sigma);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, cfg.harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
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
fn run_ceiling_vs_pathology_instrument(cfg: CeilingPathologyConfig) -> CeilingPathologyReport {
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
        .expect("#2156 instrument initial penalized LAML value probe must complete");
    let actual_decrease = initial.cost - trial_cost;
    let materialization_ratio = if predicted_decrease.is_finite()
        && predicted_decrease > f64::MIN_POSITIVE
        && actual_decrease.is_finite()
    {
        actual_decrease / predicted_decrease
    } else {
        f64::NAN
    };
    let predicted_decrease_not_materializing = materialization_ratio.is_finite()
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
            let huge_final_gradient =
                final_grad_norm.is_finite() && final_grad_norm >= cfg.huge_final_gradient_floor;
            objective
                .certify_outer_result(&result)
                .expect("ceiling-pathology outer result must certify the installed state");
            let fitted = objective.into_fitted().expect("outer fit was evaluated");
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

/// #2080 — the wide-`p` (p=96) K=2 outer penalized-LAML fit must terminate in a bounded
/// number of criterion evaluations and recover a materially positive EV — even
/// though the outer line search overshoots into the non-PD basin on many probes.
#[test]
fn wide_p_outer_reml_terminates_within_probe_budget_2080() {
    let n = 96usize;
    let p = 96usize;
    let k = 2usize;
    let harmonics = 2usize; // m = 5: [1, sin2πt, cos2πt, sin4πt, cos4πt]
    let (ev, telemetry) = run_wide_outer_fit(n, p, k, harmonics);
    eprintln!(
        "[#2080] wide-p outer fit: ev={ev:.4}, criterion_calls={}, \
         infeasible(non_pd_per_row={},cross_row={},schur={},inner_nc={}), \
         infeasible_criterion_evals={}, reactive_scalar_installs={}, \
         reactive_target_restores={}",
        telemetry.criterion_calls,
        telemetry.infeasible_non_pd_per_row,
        telemetry.infeasible_cross_row,
        telemetry.infeasible_schur,
        telemetry.infeasible_inner_not_converged,
        telemetry.infeasible_criterion_evals,
        telemetry.reactive_scalar_installs,
        telemetry.reactive_target_restores,
    );
    assert!(
        telemetry.reactive_scalar_installs > 0,
        "the initially undefined wide-K=2 seed must traverse genuine objective-installed scalar waypoints"
    );
    assert!(
        telemetry.reactive_target_restores > 0,
        "the wide-K=2 continuation must restore the objective's literal scalar target before certification"
    );
    // Bounded criterion (eval / eval_cost / efs) budget — a PROBE COUNT, not a
    // wall-clock limit (SPEC bans time budgets). With `with_max_iter(4)` and a
    // single seed the outer loop cannot issue an unbounded number of full
    // criterion evals; the pre-fix hang was UNBOUNDED inner work PER probe, not an
    // unbounded probe count, so this asserts the complementary invariant.
    assert!(
        telemetry.criterion_calls <= 64,
        "outer penalized-LAML issued {} criterion calls; expected a bounded (<= 64) probe budget",
        telemetry.criterion_calls
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
         infeasible_criterion_evals={}, infeasible_total={}",
        telemetry.criterion_calls,
        telemetry.infeasible_criterion_evals,
        telemetry.infeasible_total(),
    );
    assert!(
        telemetry.criterion_calls <= 32,
        "#2153 K=1 generated-seed fit issued {} criterion calls; expected a \
         bounded first-line-search probe budget",
        telemetry.criterion_calls
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
         rho_displacement={:.6e}, ev={:.4}, criterion_calls={}, \
         infeasible_criterion_evals={}, infeasible_total={}, outer_error={:?}, \
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
        report.telemetry.infeasible_criterion_evals,
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

/// #2080 — heavier K=3 wide-`p` variant (the issue's headline shape). Same
/// bounded-probe-budget contract.
#[test]
fn wide_p_outer_reml_terminates_k3_heavy_2080() {
    let (ev, telemetry) = run_wide_outer_fit(96, 96, 3, 2);
    eprintln!(
        "[#2080 heavy] K=3 wide-p outer fit: ev={ev:.4}, criterion_calls={}, \
         infeasible_total={}",
        telemetry.criterion_calls,
        telemetry.infeasible_total(),
    );
    assert!(telemetry.criterion_calls <= 96);
    assert!(ev.is_finite() && ev > 0.15);
}

/// #2080 ENTANGLED two-circle target — two equal-variance circles on OVERLAPPING
/// (dense, all-column) 2-frames, unlike `two_circle_wide_target`'s even/odd
/// DISJOINT split. Equal variance in a shared 4-D signal subspace makes the
/// pairing rotation-AMBIGUOUS to PCA (any orthonormal rotation of the 4 leading
/// PCs has the same variance), so the PCA-residual chart seed hands both atoms a
/// MIXTURE of the two circles → they fit the same mixed subspace → `μ̂ = 1.0`
/// co-collapse. Fourth-order (kurtosis) independence — the joint-Jacobi ISA seed —
/// is what resolves the rotation. This is the minimal faithful repro of the
/// issue's entangled product-of-circles co-collapse.
fn entangled_two_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        fa[[0, j]] = deterministic_circle_noise(j, 0);
        fa[[1, j]] = deterministic_circle_noise(j, 1);
        fb[[0, j]] = deterministic_circle_noise(j, 2);
        fb[[1, j]] = deterministic_circle_noise(j, 3);
    }
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    // Tile the 2-TORUS on an independent grid: θ_a and θ_b must be STATISTICALLY
    // INDEPENDENT (a genuine product of two circles). A dependent parameterization
    // (θ_b = 2θ_a, the previous `2*row`) is a single 1-D Lissajous/(1,2)-knot curve
    // with only ONE true latent factor — a K=2 fit then CORRECTLY leaves one atom
    // redundant, which no seed can split and which is not the co-collapse we are
    // testing. ISA separates independent subspaces, so the fixture must contain two.
    // `independent_two_circle_phases` chooses the largest divisor of `n` at or
    // below √n, so `(row mod n1, row / n1)` is a bijection onto the n1×n2 grid:
    // `(θ_a, θ_b)` is jointly uniform and therefore independent.
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let (ta, tb) = independent_two_circle_phases(n, row);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for jj in 0..p {
            z[[row, jj]] = ca * fa[[0, jj]]
                + sa * fa[[1, jj]]
                + cb * fb[[0, jj]]
                + sb * fb[[1, jj]]
                + sigma * deterministic_circle_noise(row, jj + 7);
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

/// #2080/#2023 — the ENTANGLED co-collapse regression (fails-before / passes-after
/// the joint-Jacobi ISA chart seed). Matched K=2 on the overlapping-frame two-circle
/// target: with the PCA-residual seed both atoms co-collapse onto the same mixed
/// subspace (the outer solver then thrashes infeasible probes to a `Fatal` abort);
/// with the independence-separating ISA seed the two circles land on DISTINCT atoms.
/// Collapse is EV-INVISIBLE, so a positive EV alone is not enough — the load-bearing
/// assertion is that BOTH atoms carry material decoder norm (a weakest/strongest
/// ratio well above the ~0.13 collapse regime and inside the ~0.42 healthy regime
/// measured on this shape).
#[test]
fn entangled_two_circle_outer_reml_separates_2080() {
    let n = 240usize;
    let p = 96usize;
    let k = 2usize;
    let harmonics = 2usize;
    let z = entangled_two_circle_wide_target(n, p, 0.03);
    // Diagnostic: how many independent circle planes does the joint-Jacobi ISA
    // split κ-CERTIFY on this target? If < k, the seed falls back to the PCA peel
    // and cannot separate — distinguishing "certificate failed" from "engaged but
    // under-separated" (the two contingencies for a co-collapse red).
    let isa_certified = match super::isa_seed::capture_signal_span(z.view(), k) {
        Ok(Some(parts)) => super::isa_seed::isa_extract_certified_planes(
            z.view(),
            &parts,
            k,
            &super::isa_seed::IsaSeedConfig::default(),
        )
        .len(),
        _ => 0,
    };
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), k, harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    let seed = init_rho.to_flat();
    let n_params = seed.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold entangled two-circle")
        .expect("#2080 entangled two-circle outer penalized-LAML fit must terminate, not abort");
    assert!(
        result.converged,
        "#2080 entangled acceptance requires a converged outer penalized-LAML optimum: \
         iterations={}, final_value={:.6e}, final_grad_norm={:?}",
        result.iterations, result.final_value, result.final_grad_norm,
    );
    objective
        .certify_outer_result(&result)
        .expect("entangled two-circle outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    let mut norms = vec![0.0_f64; k];
    for (i, atom) in fitted.term.atoms.iter().enumerate() {
        norms[i] = atom
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
    }
    let hi = norms.iter().copied().fold(0.0_f64, f64::max);
    let lo = norms.iter().copied().fold(f64::INFINITY, f64::min);
    let ratio = lo / hi.max(1.0e-300);
    eprintln!(
        "[#2080 entangled] isa_certified_planes={isa_certified}/{k}, ev={ev:.4}, \
         decoder_norms={norms:?}, ratio={ratio:.3}"
    );
    assert!(
        ev.is_finite() && ev > 0.20,
        "entangled K=2 fit must recover a materially positive EV (got {ev:.4})"
    );
    assert!(hi > 0.0, "at least one atom must carry decoder norm");
    assert!(
        ratio > 0.30,
        "both entangled circles must be recovered on DISTINCT atoms (no co-collapse); \
         norms={norms:?} ratio={ratio:.3} — a ratio near the ~0.13 collapse regime is the \
         μ̂ = 1.0 shared-subspace collapse the joint-Jacobi ISA seed must prevent"
    );
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
/// diverging. Each fixed-ρ evaluation must return `Ok` with a finite penalized LAML cost
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
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
    )
    .unwrap();
    let base = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // The whole smoothing sweep, from flexible (-8) through the over-smoothed tail
    // (+8) where the undamped Laplace log-det is worst conditioned.
    for &smooth in &[-8.0_f64, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0] {
        let mut t = base.clone();
        let r = SaeManifoldRho::new(0.02_f64.ln(), smooth, vec![array![0.0]]);
        let evaluated = t
            .penalized_laml_criterion_with_cache(z.view(), &r, None, 60, 0.04, 1.0e-6, 1.0e-6)
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
            "#2138: inner solve returned a non-finite penalized LAML cost at smoothing={smooth}",
        );
        assert!(
            ev.is_finite() && ev > 0.30,
            "#2138: high-working-rank small-fold circle fit must recover a materially positive \
             EV at smoothing={smooth} (got {ev:.4}), proving the inner solve reached a real \
             basin rather than a diverged / collapsed state",
        );
    }
}

/// #2080 COST-LANE PROFILER + criterion-finiteness gate.
/// Measures how a SINGLE outer penalized-LAML criterion evaluation scales in ambient width
/// `p` for the correctly-specified K=1 circle (no co-collapse). Splits the wall
/// time into: (A) the damped inner (t,β) Newton solve `run_joint_fit_arrow_schur`
/// and (B) the residual = the undamped-logdet re-converge + dense β-Schur factor
/// that `penalized_laml_criterion_with_cache_refine_policy` adds on top. This localizes the
/// cubic-in-p term the issue tracks. Asserted invariant: the inner solve
/// converges finitely and the full criterion is FINITE (rankable) at every
/// width — a non-finite criterion on this correctly-specified probe is the
/// #1094-class outer refusal. Widths kept small enough for the standard shard;
/// the wide tail (64/96/128) is profiling territory for the #2080 owner's
/// dedicated runs.
#[test]
fn profile_wide_p_criterion_cost_2080() {
    let harmonics = 2usize; // m = 1 + 2*2 = 5 basis columns per atom
    for &p in &[16usize, 32, 48] {
        let n = 96usize;
        let z = one_circle_wide_target(n, p, 0.05);
        let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
        let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
        let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
            .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
            .unwrap();
        let beta_dim = term.beta_dim();

        // Phase A: damped inner solve alone.
        let mut ta = term.clone();
        let mut rho_a = rho.clone();
        let a0 = std::time::Instant::now();
        ta.run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 8, 0.04, 1.0e-6, 1.0e-6)
            .expect("inner solve");
        let dt_a = a0.elapsed().as_secs_f64();

        // Phase A+B: full criterion (inner solve + undamped logdet + dense Schur).
        let mut tb = term.clone();
        let b0 = std::time::Instant::now();
        let evaluated = tb
            .penalized_laml_criterion_with_cache_refine_policy(
                z.view(),
                &rho,
                None,
                8,
                0.04,
                1.0e-6,
                1.0e-6,
                true,
            )
            .expect("full criterion");
        let dt_full = b0.elapsed().as_secs_f64();
        let dt_b = (dt_full - dt_a).max(0.0);
        assert!(
            evaluated.0.is_finite(),
            "#2080/#1094: the outer penalized-LAML criterion must be RANKABLE (finite) on a \
             correctly-specified K=1 wide-p circle at p={p}; a non-finite value here \
             is the probe-refusal failure class (got {})",
            evaluated.0
        );
        eprintln!(
            "[#2080 profile] p={p:>3} beta_dim={beta_dim:>4} | inner_solve={dt_a:8.3}s | logdet_phase={dt_b:8.3}s | full={dt_full:8.3}s | cost={:.4}",
            evaluated.0
        );
    }
}
