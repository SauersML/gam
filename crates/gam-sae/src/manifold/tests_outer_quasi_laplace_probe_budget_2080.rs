//! #2080 — the outer penalized quasi-Laplace `ρ` search must terminate in a bounded number of
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
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_atb};
use gam_solve::rho_optimizer::{OuterEval, OuterEvalOrder, OuterObjective, OuterProblem};
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
        SaeManifoldAtom::new_with_provided_function_gram(
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

pub(super) fn two_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
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
pub(super) fn one_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
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
/// ordered Beta--Bernoulli assignment. Returns the term and the seed reconstruction dispersion the
/// outer cascade scales its ρ seed by. `harmonics` sets the basis size `m = 1 +
/// 2·harmonics`.
pub(super) fn two_circle_periodic_term(
    z: ArrayView2<'_, f64>,
    k: usize,
    harmonics: usize,
) -> (SaeManifoldTerm, f64) {
    let n = z.nrows();
    let p = z.ncols();
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(
        PeriodicHarmonicEvaluator::new(num_basis)
            .expect("num_basis = 1 + 2*harmonics is a valid odd periodic basis width"),
    );
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let atom_dims = vec![dim; k];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims)
        .expect("one basis kind and one dim per atom, matching the fixture target");
    let mut atoms = Vec::with_capacity(k);
    let mut coords_blocks = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    let mut rss = 0.0_f64;
    for atom_idx in 0..k {
        let coords = seed_coords.slice(s![atom_idx, .., 0..dim]).to_owned();
        let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the seeded coords lie in the periodic chart domain");
        let mm = phi.ncols();
        let mut xtx = fast_atb(&phi, &phi);
        for i in 0..mm {
            xtx[[i, i]] += 1.0e-8;
        }
        let xtz = fast_atb(&phi, &z.to_owned());
        let decoder = xtx
        .cholesky(Side::Lower)
        .expect("phi^T phi + 1e-8 I is positive definite")
        .solve_mat(&xtz);
        let fitted = phi.dot(&decoder);
        for row in 0..n {
            for col in 0..p {
                let r = z[[row, col]] - fitted[[row, col]];
                rss += r * r;
            }
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "circle",
            SaeAtomBasisKind::Periodic,
            dim,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(mm),
        )
        .expect("phi, jet, decoder and gram were built with matching shapes")
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
            .expect("one logit column, coord block and manifold per atom");
    (
        SaeManifoldTerm::new(atoms, assignment)
            .expect("atoms and assignment were built over the same k atoms"),
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
            atom.decoder_coefficients()
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    assert!(
        seed_norms
            .iter()
            .all(|norm| norm.is_finite() && *norm > 0.0),
        "regression requires the nonzero decoder seed that bypassed the old cold-entry placement; norms={seed_norms:?}"
    );

    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
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
        for ((_, output), value) in atom.decoder_coefficients().indexed_iter() {
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

    let entry_eval_result =
        OuterObjective::eval_with_order(&mut objective, &entry_rho, OuterEvalOrder::Value);
    if let Err(error) = &entry_eval_result {
        let entry_rho_state = objective
            .baseline_rho
            .from_flat(entry_rho.view())
            .expect("entry_rho came from `to_flat` on this same rho layout");
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
        let decoder_grad = system
            .gb
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
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
                atom.decoder_coefficients()
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
            entry_rho_state
                .lambda_smooth_vec()
                .expect("the fixture rho carries one smoothing block per atom"),
        );
    }
    let entry_eval =
        entry_eval_result.expect("separated legal entry must solve to finite evidence");
    assert!(
        entry_eval.cost.is_finite(),
        "separated legal entry returned non-finite evidence {}",
        entry_eval.cost
    );
    OuterObjective::commit_reactive_domain_waypoint(&mut objective, &entry_rho)
        .expect("finite entry must commit its full converged state");

    // Reassemble the exact committed entry and independently recheck the same
    // strict raw-or-quotient KKT predicate that authorizes penalized quasi-Laplace score. A
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
        &committed_rho
            .lambda_smooth_vec()
            .expect("the fixture rho carries one smoothing block per atom"),
    );
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * objective.term.inner_iterate_scale();
    assert!(
        SaeManifoldTerm::quasi_laplace_kkt_stationary(raw_kkt, quotient_kkt, tolerance),
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
        .expect("seed dispersion is finite and strictly positive");
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
        .expect("#2080 wide-p outer penalized quasi-Laplace fit must terminate, not hang / abort");
    assert!(
        result.converged(),
        "#2080 wide-p acceptance requires a CONVERGED outer penalized quasi-Laplace optimum, not a \
         finite max-iteration/line-search incumbent: iterations={}, final_value={:.6e}, \
         final_grad_norm={:?}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
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
        .expect("seed dispersion is finite and strictly positive");
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
        result.converged(),
        "#2153 K=1 acceptance requires a converged outer optimum: iterations={}, \
         final_value={:.6e}, final_grad_norm={:?}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
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
        .expect("seed dispersion is finite and strictly positive");
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
        .expect("#2156 instrument initial penalized quasi-Laplace value probe must complete");
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
                outer_converged: result.converged(),
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

/// #2080 — the wide-`p` (p=96) K=2 outer penalized quasi-Laplace fit must terminate in a bounded
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
         infeasible(non_pd_per_row={},schur={},inner_nc={}), \
         infeasible_criterion_evals={}, reactive_scalar_installs={}, \
         reactive_target_restores={}",
        telemetry.criterion_calls,
        telemetry.infeasible_non_pd_per_row,
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
        "outer penalized quasi-Laplace issued {} criterion calls; expected a bounded (<= 64) probe budget",
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
        .expect("seed dispersion is finite and strictly positive");
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
        .expect("#2080 entangled two-circle outer penalized quasi-Laplace fit must terminate, not abort");
    assert!(
        result.converged(),
        "#2080 entangled acceptance requires a converged outer penalized quasi-Laplace optimum: \
         iterations={}, final_value={:.6e}, final_grad_norm={:?}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
    );
    objective
        .certify_outer_result(&result)
        .expect("entangled two-circle outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    let mut norms = vec![0.0_f64; k];
    for (i, atom) in fitted.term.atoms.iter().enumerate() {
        norms[i] = atom
            .decoder_coefficients()
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
/// diverging. Each fixed-ρ evaluation must return `Ok` with a finite penalized quasi-Laplace cost
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
        Arc::new(
        PeriodicHarmonicEvaluator::new(m).expect("m is a valid odd periodic basis width"),
    );
    let seed_coords =
        sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1])
            .expect("one periodic basis kind and one dim for the single atom");
    let coords = seed_coords.slice(s![0, .., 0..1]).to_owned();
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the seeded coords lie in the periodic chart domain");
    let mut xtx = fast_atb(&phi, &phi);
    for i in 0..m {
        xtx[[i, i]] += 1.0e-8;
    }
    let xtz = fast_atb(&phi, &z);
    let decoder = xtx
        .cholesky(Side::Lower)
        .expect("phi^T phi + 1e-8 I is positive definite")
        .solve_mat(&xtz);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("phi, jet, decoder and gram were built with matching shapes")
    .with_basis_evaluator(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n, 1), 6.0),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
    )
    .expect("one logit column, coord block and manifold for the single atom");
    let base = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("the single atom and its assignment agree on atom count");
    // The whole smoothing sweep, from flexible (-8) through the over-smoothed tail
    // (+8) where the undamped Laplace log-det is worst conditioned.
    for &smooth in &[-8.0_f64, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0] {
        let mut t = base.clone();
        let r = SaeManifoldRho::new(0.02_f64.ln(), smooth, vec![array![0.0]]);
        let evaluated = t
            .penalized_quasi_laplace_criterion_with_cache(
                z.view(),
                &r,
                None,
                60,
                0.04,
                1.0e-6,
                1.0e-6,
            )
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
            "#2138: inner solve returned a non-finite penalized quasi-Laplace cost at smoothing={smooth}",
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
/// Measures how a SINGLE outer penalized quasi-Laplace criterion evaluation scales in ambient width
/// `p` for the correctly-specified K=1 circle (no co-collapse). Splits the wall
/// time into: (A) the damped inner (t,β) Newton solve `run_joint_fit_arrow_schur`
/// and (B) the residual = the undamped-logdet re-converge + dense β-Schur factor
/// that `penalized_quasi_laplace_criterion_with_cache_refine_policy` adds on top. This localizes the
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
            .expect("seed dispersion is finite and strictly positive");
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
            .penalized_quasi_laplace_criterion_with_cache_refine_policy(
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
            "#2080/#1094: the outer penalized quasi-Laplace criterion must be RANKABLE (finite) on a \
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

/// #2080 wide-p per-eval localizer (zz_measure diagnostics). Splits a single
/// K=1 criterion evaluation into: (A) the damped inner (t,β) Newton solve,
/// (M) the dense exact-A materialization column-by-column, (E) the full exact
/// observed-information log-dets (M + two dense `eigh`), and the residual
/// refine-loop remainder = full − A − E, so the cubic-in-p term is localized
/// in ONE run. The in-suite sweep stops at p=32 to stay affordable (`#[ignore]`
/// is banned; widen the sweep locally when hunting the tail).
#[test]
fn zz_measure_wide_p_criterion_cost_localizer_2080() {
    let harmonics = 2usize; // m = 5 basis columns per atom
    for &p in &[16usize, 32] {
        let n = 96usize;
        let z = one_circle_wide_target(n, p, 0.05);
        let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
        let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
        let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
            .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
            .expect("seed dispersion is finite and strictly positive");
        let beta_dim = term.beta_dim();

        // Phase A: damped inner solve alone.
        let mut ta = term.clone();
        let mut rho_a = rho.clone();
        let a0 = std::time::Instant::now();
        ta.run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 8, 0.04, 1.0e-6, 1.0e-6)
            .expect("inner solve");
        let dt_a = a0.elapsed().as_secs_f64();

        // Full criterion (returns the converged undamped cache).
        let mut tb = term.clone();
        let f0 = std::time::Instant::now();
        let (_cost, _loss, cache) = tb
            .penalized_quasi_laplace_criterion_with_cache(
                z.view(),
                &rho,
                None,
                8,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .expect("full criterion");
        let dt_full = f0.elapsed().as_secs_f64();
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;

        // Phase M: dense exact-A materialization (dim column matvecs).
        let m0 = std::time::Instant::now();
        let a_dense = tb
            .materialize_exact_hessian_dense(&rho, z.view(), &cache)
            .expect("materialize A");
        let dt_m = m0.elapsed().as_secs_f64();

        // Phase E: full exact observed-information log-dets (materialize + 2 eigh).
        let e0 = std::time::Instant::now();
        let log_dets = tb
            .exact_observed_information_log_dets(&rho, z.view(), &cache)
            .expect("observed-information log-dets");
        let dt_e = e0.elapsed().as_secs_f64();

        let dt_eigh = (dt_e - dt_m).max(0.0);
        let dt_refine = (dt_full - dt_a - dt_e).max(0.0);
        eprintln!(
            "[#2080 localize] p={p:>3} beta_dim={beta_dim:>4} dim={dim:>4} a_dim={} \
             log_dets={log_dets:?} | inner_A={dt_a:8.3}s | refine_rest={dt_refine:8.3}s | \
             materialize_M={dt_m:8.3}s | eigh_E-M={dt_eigh:8.3}s | full={dt_full:8.3}s",
            a_dense.nrows(),
        );
        // The cost attribution above is only about the right work if each phase
        // actually produced its object: a square exact A covering at least the
        // t-block, and finite log-dets. A refusal laundered into a non-finite
        // `log_dets` would otherwise be timed and reported as a successful phase.
        assert!(
            a_dense.nrows() == a_dense.ncols() && a_dense.nrows() >= total_t,
            "p={p}: the materialized exact A must be a square operator covering the {total_t} \
             t-coordinates, got {}x{}",
            a_dense.nrows(),
            a_dense.ncols()
        );
        assert!(
            log_dets.0.is_finite() && log_dets.1.is_finite(),
            "p={p}: the exact observed-information log-dets this phase is timed on must be \
             finite, got {log_dets:?}"
        );
        assert!(
            dim > 0 && beta_dim > 0,
            "p={p}: the localizer must be run on a nonempty system (dim={dim}, \
             beta_dim={beta_dim})"
        );
    }
}

/// #2439 MEASUREMENT (zz_measure) — WHY does asking for the gradient change the value?
///
/// `value_lane_prices_at_shared_fixed_point_2228` measures that the same objective
/// at the same ρ reports `1.1359321989218647e3` under `OuterEvalOrder::Value` and
/// `1.1363163948443218e3` under `ValueAndGradient` — 22,700× the certification
/// bound, with the `Value` lane matching the bare criterion to all 17 digits.
///
/// Two candidate causes, and this probe decides between them in one run:
///
/// 1. **the assembly differs** — both lanes converge to the SAME inner mode and
///    still price it differently, which would put the defect in what is summed;
/// 2. **the lanes evaluate at DIFFERENT inner modes** — `eval()` re-solves and
///    lands elsewhere, so the value is computed at one mode while the gradient
///    describes another. That is a violation of the envelope identity
///    `dV/dρ = ∂ℓ_p/∂ρ|_θ̂` by construction, not merely a discrepancy.
///
/// Both lanes publish `inner_beta_hint`, so comparing them bitwise separates the
/// two without instrumenting either lane.
///
/// The probe also varies something the failing test fixes: that test builds a
/// FRESH objective per lane, so the `#2080 (a)` hand-off — install the value
/// probe's converged inner state before the gradient lane's criterion loop, whose
/// stated purpose is "same converged optimum ⇒ identical criterion value" — never
/// happens. Production evaluates both on ONE objective. Arm B runs that ordering.
/// If B agrees where A disagrees, the hand-off is load-bearing and the value is a
/// functional of the starting state rather than of ρ; if B disagrees too, the
/// defect is independent of the hand-off.
#[test]
fn zz_measure_2439_value_vs_gradient_inner_mode() {
    let n = 96usize;
    let p = 48usize;
    let z = one_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let imi = 16usize;
    let (lr, re, rb) = (0.04_f64, 1.0e-6_f64, 1.0e-6_f64);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 4.0_f64, vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let rho_flat = rho.to_flat();

    let report = |tag: &str, a: &OuterEval, b: &OuterEval| {
        let diff = (a.cost - b.cost).abs();
        let bound = f64::EPSILON.sqrt() * a.cost.abs().max(b.cost.abs()).max(1.0);
        eprintln!(
            "[#2439 {tag}] value={:.16e} grad_lane={:.16e} diff={diff:.6e} bound={bound:.6e}",
            a.cost, b.cost
        );
        // Both lanes ACCEPTED (the `.expect(...)`s above), so both must price a
        // finite cost — `diff` and the certification `bound` it is compared against
        // are otherwise undefined and the two candidate causes cannot be separated.
        assert!(
            a.cost.is_finite() && b.cost.is_finite(),
            "[#2439 {tag}] both accepted lanes must price a finite cost \
             (value={}, grad_lane={})",
            a.cost,
            b.cost
        );
        assert!(
            diff.is_finite() && bound > 0.0,
            "[#2439 {tag}] the lane gap and its certification bound must be well-defined \
             (diff={diff}, bound={bound})"
        );
        match (&a.inner_beta_hint, &b.inner_beta_hint) {
            (Some(bv), Some(bg)) if bv.len() == bg.len() => {
                let identical = bv
                    .iter()
                    .zip(bg.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
                let max_abs = bv
                    .iter()
                    .zip(bg.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max);
                eprintln!(
                    "[#2439 {tag}] beta len={} identical={identical} \
                     max|dbeta|={max_abs:.6e} => {}",
                    bv.len(),
                    if identical {
                        "SAME MODE: the assembly differs (cause 1)"
                    } else {
                        "DIFFERENT MODES: envelope identity broken by construction (cause 2)"
                    }
                );
            }
            (bv, bg) => eprintln!(
                "[#2439 {tag}] beta hints not comparable: value={:?} grad={:?}",
                bv.as_ref().map(|v| v.len()),
                bg.as_ref().map(|v| v.len())
            ),
        }
    };

    // Arm A — the failing test's construction: a FRESH objective per lane, so no
    // converged inner state is handed from the value probe to the gradient lane.
    let mut obj_v =
        SaeManifoldOuterObjective::new(term.clone(), z.clone(), None, rho.clone(), imi, lr, re, rb);
    let a_value = OuterObjective::eval_with_order(&mut obj_v, &rho_flat, OuterEvalOrder::Value)
        .expect("arm A value lane");
    let mut obj_g =
        SaeManifoldOuterObjective::new(term.clone(), z.clone(), None, rho.clone(), imi, lr, re, rb);
    let a_grad =
        OuterObjective::eval_with_order(&mut obj_g, &rho_flat, OuterEvalOrder::ValueAndGradient)
            .expect("arm A gradient lane");
    report("A fresh-objective-per-lane", &a_value, &a_grad);

    // Arm B — production's ordering: ONE objective, value probe first, so the
    // `#2080 (a)` hand-off can install the probe's converged inner state.
    let mut obj =
        SaeManifoldOuterObjective::new(term.clone(), z.clone(), None, rho.clone(), imi, lr, re, rb);
    let b_value = OuterObjective::eval_with_order(&mut obj, &rho_flat, OuterEvalOrder::Value)
        .expect("arm B value lane");
    let b_grad =
        OuterObjective::eval_with_order(&mut obj, &rho_flat, OuterEvalOrder::ValueAndGradient)
            .expect("arm B gradient lane");
    report("B shared-objective ", &b_value, &b_grad);
}

/// #2228 MEASUREMENT (zz_measure) — is the value-lane fixture's inner solve at a
/// FLOOR, or is it still converging and merely out of budget?
///
/// This is the question the #2267 ratchet left open. That work took one criterion
/// evaluation on this fixture from 808 inner iterations / 52 s to 272 / 4.36 s,
/// and ‖g‖ moved only 1.522323e-1 → 1.100073e-1 against an unchanged tolerance of
/// 1.435797e-3 — still 77× above. Iteration budget is therefore no longer what
/// stands between this fixture and its tolerance, but "no longer the BINDING
/// constraint" is not the same claim as "more budget cannot help", and the two
/// have opposite owners:
///
/// * still converging (some budget succeeds) ⇒ a RATE problem, and rate belongs to
///   the step rule and its accept rule (#2267), not to the refusal contract;
/// * floored (no budget succeeds) ⇒ the inner solve cannot reach `1e-5 ·
///   inner_iterate_scale()` on this problem at all, and a contract that demands it
///   is asking for something unattainable — which is #2228's own question.
///
/// The sweep answers it without parsing anything out of an error message: sweep the
/// refine budget and record only whether the criterion returns. A binary outcome per
/// budget is all the question needs, and it is exactly the acceptance bar this issue
/// is written against ("the fit must converge, not refuse").
#[test]
fn zz_measure_2228_value_lane_budget_sweep() {
    let n = 96usize;
    let p = 48usize;
    let z = one_circle_wide_target(n, p, 0.05);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let (lr, re, rb) = (0.04_f64, 1.0e-6_f64, 1.0e-6_f64);
    // The sweep's binary CONVERGED/REFUSED outcome only answers the rate-vs-floor
    // question if every budget is run against the same well-posed wide-p target.
    assert!(
        z.nrows() == n && z.ncols() == p && z.iter().all(|v| v.is_finite()),
        "[#2228 sweep] the wide-p target must be a finite {n}x{p} matrix, got {}x{}",
        z.nrows(),
        z.ncols()
    );

    for imi in [8usize, 16, 32, 64, 128] {
        // Rebuild the term and rho per budget: the criterion mutates inner state,
        // so a shared term would make each budget start where the previous one
        // stopped and the sweep would measure a warm continuation instead of the
        // budget it names.
        let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, 2);
        let rho = SaeManifoldRho::new(0.02_f64.ln(), 4.0_f64, vec![array![0.0]])
            .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
            .expect("seed dispersion is finite and strictly positive");
        // FULL budget (refine_progress_extension = true): the arm that must reach
        // the root the test prices against.
        let mut t = term.clone();
        let started = std::time::Instant::now();
        let full =
            t.penalized_quasi_laplace_criterion_with_cache(z.view(), &rho, None, imi, lr, re, rb);
        let full_secs = started.elapsed().as_secs_f64();
        // COARSE budget (refine_progress_extension = false): the raw diagnostic
        // policy, which the fixture requires to be inadequate at the same budget.
        // If raising `imi` ever made this one adequate too, the test's
        // coarse-vs-full premise would go vacuous, so sweep both together.
        let mut t_coarse = term.clone();
        let started = std::time::Instant::now();
        let coarse = t_coarse.penalized_quasi_laplace_criterion_with_cache_refine_policy(
            z.view(),
            &rho,
            None,
            imi,
            lr,
            re,
            rb,
            false,
        );
        let coarse_secs = started.elapsed().as_secs_f64();
        let full_report = match &full {
            Ok((value, _, _)) => format!("CONVERGED value={value:.9e} ({full_secs:.2}s)"),
            Err(err) => format!("REFUSED ({full_secs:.2}s): {err:?}"),
        };
        let coarse_report = match &coarse {
            Ok(evaluated) => format!("CONVERGED value={:.9e} ({coarse_secs:.2}s)", evaluated.0),
            Err(err) => format!("REFUSED ({coarse_secs:.2}s): {err:?}"),
        };
        eprintln!("[#2228 sweep] imi={imi:>4} full={full_report}");
        eprintln!("[#2228 sweep] imi={imi:>4} coarse={coarse_report}");
        // "CONVERGED" is the answer this sweep records, so an `Ok` carrying a
        // non-finite value would flip the rate-vs-floor verdict while printing as
        // a converged budget.
        if let Ok((value, _, _)) = &full {
            assert!(
                value.is_finite(),
                "[#2228 sweep] imi={imi}: a CONVERGED full-budget criterion must carry a finite \
                 value, got {value}"
            );
        }
        if let Ok(evaluated) = &coarse {
            assert!(
                evaluated.0.is_finite(),
                "[#2228 sweep] imi={imi}: a CONVERGED coarse-budget criterion must carry a \
                 finite value, got {}",
                evaluated.0
            );
        }
    }
}

/// PROBE (#2080 Class-B discriminator). Is the residual the inner gate measures
/// the gradient of the objective the inner line search descends?
///
/// [`zz_measure_k2_wide_p_inner_trajectory_2080`] shows a trajectory where the
/// two disagree in SIGN: over the tail of the K=2 wide-`p` solve the penalised
/// objective falls monotonically while `‖g‖` rises monotonically, iterate after
/// iterate. Only two readings of that are possible.
///
///   * The gate's `g` IS `∇(penalized_objective_total)`, and the solve is simply
///     unable to move the stiff directions that carry it. Then a steepest-descent
///     FINITE DIFFERENCE along `−g` must realise a decrease of `h·‖g‖²` to
///     leading order, because that is what a gradient means.
///   * The gate's `g` is NOT that gradient — a different objective, a different
///     chart, or a term present on one side only. Then the finite difference
///     along `−g` will not match `h·‖g‖²`, and no amount of iteration or budget
///     can make the gate clear, because the solve is minimising one function and
///     the gate is certifying the stationarity of another.
///
/// The measurement is direct: assemble the system at the iterate the solve
/// actually reaches, read `g` off it (`row.gt` per row plus `sys.gb`, the exact
/// blocks [`sae_manifold_newton_directional_decrease`] contracts), step along
/// `−g` with [`SaeManifoldTerm::apply_newton_step`] at several `h`, and compare
/// the realised decrease against the predicted `h·‖g‖²`. The ratio is reported
/// per `h`; a consistent ratio near 1 as `h → 0` says gradient, anything else
/// says desync. Diagnostic only — it asserts that the readings are finite.
#[test]
fn zz_measure_k2_wide_p_gradient_is_the_objectives_2080() {
    let z = two_circle_wide_target(96, 96, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 2, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");

    // Read the question at BOTH ends of the trajectory: the cold seed, where the
    // solve is descending happily, and the tail plateau, where `‖g‖` and the
    // objective move in opposite directions. A desync that is present at both is
    // structural; one that appears only at the plateau is state-dependent.
    for &warmup in &[0usize, 128] {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        if warmup > 0 {
            term.run_joint_fit_arrow_schur_for_quasi_laplace(
                z.view(),
                &mut rho_fixed,
                None,
                warmup,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .expect("inner evidence fit must not hard-error on the K=2 wide-p rung");
        }
        let sys = term
            .assemble_arrow_schur(z.view(), &rho_fixed, None)
            .expect("reassemble at the warmed iterate");

        // `d = −g`, in the variable-stride `(row_offsets, row_dims)` layout the
        // solver's own step vectors use.
        let total_t = sys.row_offsets[sys.rows.len()];
        let mut dir_t = Array1::<f64>::zeros(total_t);
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let row_base = sys.row_offsets[row_idx];
            for axis in 0..sys.row_dims[row_idx] {
                dir_t[row_base + axis] = -row.gt[axis];
            }
        }
        let mut dir_b = Array1::<f64>::zeros(sys.k);
        for idx in 0..sys.k {
            dir_b[idx] = -sys.gb[idx];
        }
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        // Cross-check the direction against the solver's own contraction, so a
        // layout mistake here cannot be read as a desync: for `d = −g` the
        // directional decrease `−gᵀd` must be exactly `‖g‖²`.
        let contracted =
            sae_manifold_newton_directional_decrease(&sys, dir_t.view(), dir_b.view());
        let base_objective = term
            .penalized_objective_total(z.view(), &rho_fixed, None, 1.0)
            .expect("objective at the warmed iterate");
        eprintln!(
            "[2080-FD] warmup={warmup} ‖g‖²={grad_norm_sq:.9e} (−gᵀd via solver contraction \
             ={contracted:.9e}, rel_gap={:.3e}) obj={base_objective:.12e}",
            (contracted - grad_norm_sq).abs() / grad_norm_sq.max(f64::MIN_POSITIVE)
        );

        let snapshot = term.snapshot_mutable_state();
        let mut finite_readings = 0usize;
        for &h in &[1.0e-4_f64, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8] {
            term.restore_mutable_state(&snapshot)
                .expect("restore before each finite-difference trial");
            let stepped = term
                .apply_newton_step(dir_t.view(), dir_b.view(), h)
                .and_then(|()| term.penalized_objective_total(z.view(), &rho_fixed, None, 1.0));
            match stepped {
                Ok(objective) => {
                    let realised = base_objective - objective;
                    let predicted = h * grad_norm_sq;
                    eprintln!(
                        "[2080-FD] warmup={warmup} h={h:.1e} | predicted_decrease={predicted:.9e} \
                         | realised_decrease={realised:.9e} | ratio={:.6}",
                        realised / predicted
                    );
                    if realised.is_finite() {
                        finite_readings += 1;
                    }
                }
                Err(err) => eprintln!("[2080-FD] warmup={warmup} h={h:.1e} | step failed: {err}"),
            }
        }
        term.restore_mutable_state(&snapshot)
            .expect("restore after the finite-difference sweep");
        assert!(
            finite_readings > 0,
            "[2080-FD] warmup={warmup}: no finite finite-difference reading; the probe \
             measured nothing"
        );
    }
}

/// PROBE (#2080 Class-B, PRE-REGISTERED A/B). Would a gradient-related direction
/// clear the gate that the arrow-Schur direction cannot?
///
/// [`zz_measure_k2_wide_p_gradient_is_the_objectives_2080`] establishes that the
/// gate's `g` IS `∇(penalized_objective_total)` — a steepest-descent finite
/// difference realises `h·‖g‖²` to five digits at the plateau — so descent along
/// `−g` is genuinely available there. [`zz_measure_k2_wide_p_inner_trajectory_2080`]
/// establishes that the solver's own direction `Δ` is nearly ORTHOGONAL to `g`
/// (`gᵀΔ/(‖g‖‖Δ‖) ≈ 2e-3`), which is precisely the condition under which Armijo
/// backtracking stops guaranteeing `‖g‖ → 0`.
///
/// This probe runs the two arms from ONE shared state so the comparison is not
/// across fixtures: arm A continues the production inner solve, arm B takes
/// plain Armijo steepest-descent steps. Same iterate, same iteration count, same
/// objective, same sufficient-decrease constant.
///
/// **Registered in advance:** if arm B drives `‖g‖` materially down while arm A
/// holds it flat, the defect is the DIRECTION and a gradient-related fallback is
/// the fix. If arm B's `‖g‖` also stalls, the diagnosis is incomplete and no
/// angle condition should be built on it. Diagnostic only; it asserts that both
/// arms produced readings.
#[test]
fn zz_measure_k2_wide_p_gradient_arm_vs_solver_arm_2080() {
    let z = two_circle_wide_target(96, 96, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 2, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");

    // The shared starting state: the plateau the production solve reaches and
    // then cannot leave.
    let warmup = 128usize;
    let mut shared = base.clone();
    let mut rho_fixed = rho.clone();
    shared
        .run_joint_fit_arrow_schur_for_quasi_laplace(
            z.view(),
            &mut rho_fixed,
            None,
            warmup,
            0.04,
            1.0e-6,
            1.0e-6,
        )
        .expect("inner evidence fit must not hard-error on the K=2 wide-p rung");
    let entry = shared.snapshot_mutable_state();

    let report = |tag: &str, term: &mut SaeManifoldTerm, iter: usize| -> Option<(f64, f64, f64)> {
        let sys = term.assemble_arrow_schur(z.view(), &rho_fixed, None).ok()?;
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        let quotient = term.quotient_gradient_norm_from_system(
            &sys,
            grad_norm_sq,
            &rho_fixed.lambda_smooth_vec().ok()?,
        );
        let objective = term
            .penalized_objective_total(z.view(), &rho_fixed, None, 1.0)
            .ok()?;
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        eprintln!(
            "[2080-AB] {tag} iter={iter:>4} ‖g‖={:.6e} ‖Π⊥g‖={quotient:.6e} tol={tolerance:.6e} \
             obj={objective:.12e}",
            grad_norm_sq.sqrt()
        );
        Some((grad_norm_sq.sqrt(), quotient, objective))
    };

    // ── Arm A: the production inner solve, continued from the shared state.
    let mut arm_a = shared.clone();
    let mut a_readings = 0usize;
    report("A/solver ", &mut arm_a, 0);
    for round in 1..=8usize {
        let mut rho_a = rho_fixed.clone();
        if arm_a
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                z.view(),
                &mut rho_a,
                None,
                25,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .is_err()
        {
            eprintln!("[2080-AB] A/solver  round {round} hard-errored");
            break;
        }
        if report("A/solver ", &mut arm_a, round * 25).is_some() {
            a_readings += 1;
        }
    }

    // ── Arm B: plain Armijo steepest descent from the same shared state.
    let mut arm_b = shared;
    arm_b
        .restore_mutable_state(&entry)
        .expect("arm B starts from the shared entry state");
    report("B/gradient", &mut arm_b, 0);
    let mut b_readings = 0usize;
    let mut trial_step = 1.0_f64;
    for iter in 1..=200usize {
        let Ok(sys) = arm_b.assemble_arrow_schur(z.view(), &rho_fixed, None) else {
            break;
        };
        let total_t = sys.row_offsets[sys.rows.len()];
        let mut dir_t = Array1::<f64>::zeros(total_t);
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let row_base = sys.row_offsets[row_idx];
            for axis in 0..sys.row_dims[row_idx] {
                dir_t[row_base + axis] = -row.gt[axis];
            }
        }
        let mut dir_b = Array1::<f64>::zeros(sys.k);
        for idx in 0..sys.k {
            dir_b[idx] = -sys.gb[idx];
        }
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        // Normalise, so the trial length is a distance and not a gradient scale.
        let grad_norm = grad_norm_sq.sqrt();
        if !(grad_norm.is_finite() && grad_norm > 0.0) {
            break;
        }
        dir_t.mapv_inplace(|v| v / grad_norm);
        dir_b.mapv_inplace(|v| v / grad_norm);
        // Directional decrease along the UNIT gradient direction is exactly ‖g‖.
        let Ok(pre) = arm_b.penalized_objective_total(z.view(), &rho_fixed, None, 1.0) else {
            break;
        };
        let snapshot = arm_b.snapshot_mutable_state();
        let mut accepted = false;
        let mut alpha = trial_step;
        for _ in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
            if arm_b.restore_mutable_state(&snapshot).is_err() {
                break;
            }
            let post = arm_b
                .apply_newton_step(dir_t.view(), dir_b.view(), alpha)
                .and_then(|()| arm_b.penalized_objective_total(z.view(), &rho_fixed, None, 1.0));
            if let Ok(post) = post
                && post.is_finite()
                && post <= pre - SAE_MANIFOLD_ARMIJO_C1 * alpha * grad_norm
            {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            // A DISCARDED restore result means the arm silently continues from a
            // half-rolled-back state and every later reading is still attributed
            // to the B arm. `restore_mutable_state` is documented as a TOTAL
            // structural inverse that commits nothing on failure, so an error
            // here is an invariant failure of this harness's own snapshot, not a
            // recoverable condition -- and `let _ok = ...` is banned in this
            // workspace for exactly that reason.
            arm_b.restore_mutable_state(&snapshot).unwrap_or_else(|error| {
                panic!("B-arm snapshot restore after a failed line search: {error:?}")
            });
            eprintln!("[2080-AB] B/gradient line search found no acceptable step at iter={iter}");
            break;
        }
        // Ratchet the trial length the same way the production loop does.
        trial_step = (alpha * 2.0).min(1.0e2);
        if iter % 25 == 0 && report("B/gradient", &mut arm_b, iter).is_some() {
            b_readings += 1;
        }
    }

    assert!(
        a_readings > 0 && b_readings > 0,
        "[2080-AB] the A/B needs a reading from BOTH arms to be a comparison; \
         got A={a_readings}, B={b_readings}"
    );
}

/// PROBE (#2080 Class-B localizer, block split). WHICH block carries the
/// residual the inner gate cannot clear?
///
/// The companion probes establish that `g` is the objective's true gradient and
/// that neither the arrow-Schur direction nor steepest descent moves it, with
/// the gradient arm's own accepted step pinning the curvature along `−g` at
/// `λ ≈ 7e4` on a `96×96, K=2` fixture. A conditioning number that large on a
/// fixture that small is a property of the inner parametrisation, so the next
/// question is structural rather than numerical: `‖g‖² = Σ_rows ‖g_t,row‖² +
/// ‖g_β‖²`, and within each row block the leading `assignment_coord_dim` axes are
/// GATE LOGITS and the remainder are CHART COORDINATES. Splitting the norm three
/// ways names the block, and reporting the same split for the solver's step `Δ`
/// says whether the step even lives where the residual does.
///
/// Diagnostic only: it asserts the split is a partition of the norm it claims to
/// decompose, so a layout error cannot be read as a finding.
#[test]
fn zz_measure_k2_wide_p_residual_block_split_2080() {
    let z = two_circle_wide_target(96, 96, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 2, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");

    for &warmup in &[0usize, 32, 128] {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        if warmup > 0 {
            term.run_joint_fit_arrow_schur_for_quasi_laplace(
                z.view(),
                &mut rho_fixed,
                None,
                warmup,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .expect("inner evidence fit must not hard-error on the K=2 wide-p rung");
        }
        let sys = term
            .assemble_arrow_schur(z.view(), &rho_fixed, None)
            .expect("reassemble at the warmed iterate");
        let assignment_dim = term.assignment.assignment_coord_dim();

        let mut logit_sq = 0.0_f64;
        let mut coord_sq = 0.0_f64;
        for row in sys.rows.iter() {
            for (axis, &value) in row.gt.iter().enumerate() {
                if axis < assignment_dim {
                    logit_sq += value * value;
                } else {
                    coord_sq += value * value;
                }
            }
        }
        let beta_sq = sys.gb.iter().map(|&v| v * v).sum::<f64>();
        let total = SaeManifoldTerm::system_grad_norm_sq(&sys);
        // The split must BE the norm, or the shares below describe nothing.
        let partition_gap = (logit_sq + coord_sq + beta_sq - total).abs()
            / total.max(f64::MIN_POSITIVE);
        assert!(
            partition_gap <= 1.0e-12,
            "[2080-BLK] warmup={warmup}: the three-way split is not a partition of ‖g‖² \
             (logit={logit_sq:.6e}, coord={coord_sq:.6e}, beta={beta_sq:.6e}, total={total:.6e}, \
             rel_gap={partition_gap:.3e})"
        );
        let share = |part: f64| 100.0 * part / total.max(f64::MIN_POSITIVE);
        eprintln!(
            "[2080-BLK] warmup={warmup:>4} ‖g‖={:.6e} | logit {:.6e} ({:.2}%) | coord {:.6e} \
             ({:.2}%) | beta {:.6e} ({:.2}%) | assignment_dim={assignment_dim} k_border={}",
            total.sqrt(),
            logit_sq.sqrt(),
            share(logit_sq),
            coord_sq.sqrt(),
            share(coord_sq),
            beta_sq.sqrt(),
            share(beta_sq),
            sys.k,
        );
    }
}

/// PROBE (#2080 Class-B localizer, β-step annihilation). Does the arrow/Schur
/// REDUCTION use the β curvature it is handed?
///
/// The block-split probe above localizes the plateau residual: at the 128-iterate
/// plateau on the `96×96, K=2` wide-`p` rung, 99.89% of `‖g‖` sits in the β
/// (decoder border) block, bit-stable across a 4× iteration increase, while the
/// objective still descends and the solver's step is nearly orthogonal to `g`
/// (`cos ≈ 2.4e-3`). At FROZEN `t` the decoder enters the objective
/// QUADRATICALLY — least-squares reconstruction plus a quadratic smoothness
/// penalty — so the exact β-block Newton step must zero `g_β` outright, and
/// `H_ββ` (`effective_penalty_op`) has been verified to carry BOTH the data-fit
/// Gauss-Newton term and the smoothness penalty. So the curvature is present;
/// the open question is whether the step USES it.
///
/// This compares the β component of the SOLVER's actual Newton step `Δ_β`
/// against `δ_block = −H_ββ⁻¹ g_β`, the step that solves the β subproblem alone
/// at frozen `t`. The decisive number is `‖Δ_β‖ / ‖δ_block‖`, pre-registered:
///
/// * ratio `≈ 1` — the reduction transmits the β step at roughly full length, so
///   the β block IS being solved and the live mechanism is (i): `H_ββ` is not the
///   true β curvature (an analytic penalty writing gradient but no curvature),
///   and the plateau is a curvature-content defect, not a reduction defect.
/// * ratio `≈ 0` — the reduction ANNIHILATES the β step even though `H_ββ` is
///   right: mechanism (ii), the arrow/Schur back-substitution or its β
///   deflation/pinning is discarding the border increment. That would make the
///   plateau structural in the reduction, not in the assembled curvature.
/// * anything else (ratio `O(1)` but not `≈ 1`, or a ratio near 1 with `cos ≈ 0`)
///   — a THIRD outcome: the reduction produces a β step of comparable size in a
///   DIFFERENT direction, i.e. the coupling through `H_tβ` is redirecting the
///   border increment rather than dropping it. The reported cosine separates
///   this from the first two: only the first outcome should show both ratio and
///   cosine near one.
///
/// One structural fact sharpens the first outcome, and it is worth stating
/// because it costs nothing to have: this probe passes `analytic_penalties =
/// None` to BOTH the warmup fit and the reassembly, exactly as the block-split
/// probe above did when it measured the 99.89% share. So the system under
/// measurement carries only the data-fit Gauss-Newton term and the quadratic
/// smoothness/ARD priors — and for a LINEAR decoder at frozen `t` every one of
/// those is exactly quadratic in β, which makes Gauss-Newton the exact Hessian
/// rather than a majorizer. `H_ββ` here is therefore the true β curvature by
/// construction, not by assumption. That EXCLUDES the whole
/// gradient-written-but-curvature-skipped family (the `dense_beta_curvature`
/// early returns, of which the isometry β pullback is the sharpest) as the
/// mechanism for THIS fixture: none of those penalties is present to write a
/// gradient. If the plateau still reproduces here — and the block split says it
/// does — then outcome (i) is already refuted for this fixture and the reduction
/// is where the remaining explanation has to live.
///
/// Diagnostic only. It asserts nothing about the ratio — only that the block
/// solve is its own solution (`‖H_ββ·δ_block + g_β‖ / ‖g_β‖ ≈ 0`), so a bad
/// linear solve cannot be misread as a finding. A non-PD `H_ββ` is REPORTED
/// rather than panicked on and rather than silently ridged: a non-PD β block at
/// the plateau would itself be the finding, and a ridge would fabricate the
/// answer the probe exists to measure.
#[test]
fn zz_measure_k2_wide_p_beta_step_is_annihilated_2080() {
    let z = two_circle_wide_target(96, 96, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 2, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");

    for &warmup in &[32usize, 128] {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        term.run_joint_fit_arrow_schur_for_quasi_laplace(
            z.view(),
            &mut rho_fixed,
            None,
            warmup,
            0.04,
            1.0e-6,
            1.0e-6,
        )
        .expect("inner evidence fit must not hard-error on the K=2 wide-p rung");
        let sys = term
            .assemble_arrow_schur(z.view(), &rho_fixed, None)
            .expect("reassemble at the warmed iterate");
        if sys.k == 0 {
            eprintln!("[2080-BSTEP] warmup={warmup:>4}: k=0, there is no β border to measure");
            continue;
        }

        // The β curvature the SYSTEM carries, exactly as the reduction sees it:
        // the installed penalty operator when there is one, the assembled `hbb`
        // otherwise. `k` is 40 at the plateau, so densifying is free here.
        let hbb = sys.effective_penalty_op().to_dense();
        let g_beta_norm = sys.gb.iter().map(|&v| v * v).sum::<f64>().sqrt();

        // `δ_block = −H_ββ⁻¹ g_β`. The Newton RHS convention is the system's own
        // (`rhs_beta = −sys.gb`, arrow_schur/newton_step.rs), so this is the same
        // sign the solver's β increment carries and the two are comparable
        // without a flip. A refused factorization is a READING, not a panic.
        let factor = match hbb.cholesky(faer::Side::Lower) {
            Ok(factor) => factor,
            Err(error) => {
                eprintln!(
                    "[2080-BSTEP] warmup={warmup:>4}: hbb is not positive definite at pivot i \
                     (dense Cholesky refused: {error:?}); k={} ‖g_β‖={g_beta_norm:.6e} — a \
                     non-PD β block at the plateau is itself the finding, so no ridge is applied \
                     and no block step is reported",
                    sys.k,
                );
                continue;
            }
        };
        let delta_block = factor.solvevec(&sys.gb.mapv(|value| -value));

        // The residual the block step leaves. This is the probe's own self-check:
        // it must be ~0 or nothing below describes the β subproblem.
        let block_residual = hbb.dot(&delta_block) + &sys.gb;
        let block_residual_rel = block_residual.iter().map(|&v| v * v).sum::<f64>().sqrt()
            / g_beta_norm.max(f64::MIN_POSITIVE);

        // The SOLVER's step, through the SAME entry point and the SAME options
        // the criterion's undamped evidence lane uses
        // (construction_quasi_laplace.rs:339 / :974). The β increment is returned
        // as its own array, so there is no layout slicing to get wrong.
        let options = ArrowSolveOptions::direct()
            .with_gpu_policy(term.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let (_delta_t, delta_beta, _cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .expect("the production inner step must solve at the warmed K=2 wide-p iterate");

        let block_norm = delta_block.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let solver_norm = delta_beta.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let ratio = solver_norm / block_norm.max(f64::MIN_POSITIVE);
        let cosine =
            delta_beta.dot(&delta_block) / (solver_norm * block_norm).max(f64::MIN_POSITIVE);
        // `−½ g_β·δ_block = ½ g_β' H_ββ⁻¹ g_β`: the decrease the β subproblem alone
        // would buy at frozen `t`, which is what the plateau is failing to collect.
        let predicted_decrease = -0.5 * sys.gb.dot(&delta_block);

        assert!(
            block_residual_rel <= 1.0e-6,
            "[2080-BSTEP] warmup={warmup}: the β block solve does not solve its own system \
             (‖H_ββ·δ_block + g_β‖/‖g_β‖ = {block_residual_rel:.3e}), so the ratio below would \
             be measuring a bad solve rather than the reduction. The bar is 1e-6 rather than \
             machine precision because a backward-stable Cholesky bounds the residual by \
             `‖H‖‖δ‖·eps`, and `‖H‖‖δ‖ ≈ κ·‖g‖` — so this ratio scales as `κ·eps`, and the \
             plateau's inner conditioning is already measured at `κ ≈ 7e4` (which predicts \
             ~1e-11). Anything at 1e-6 is a broken solve, not conditioning; the value is \
             printed on every reading either way so the scaling stays visible"
        );
        eprintln!(
            "[2080-BSTEP] warmup={warmup:>4} k={} ‖g_β‖={g_beta_norm:.6e} | block \
             ‖δ_block‖={block_norm:.6e} predicted_decrease={predicted_decrease:.6e} | solver \
             ‖Δ_β‖={solver_norm:.6e} | ratio={ratio:.6e} cos={cosine:+.6e} | \
             block_solve_rel_residual={block_residual_rel:.3e}",
            sys.k,
        );
    }
}

/// PROBE (#2080 Class-B localizer, reduced-Schur PD floor). Is the plateau
/// residual sitting in directions the solver has CLAMPED?
///
/// `solve_dense_reduced_system` (gam-solve `arrow_schur/reduced_solve.rs`) has an
/// ill-conditioned-but-PD branch: when the reduced Schur's κ estimate exceeds
/// `safe_spd_kappa_max`, it calls `spectral_pd_floored_schur(schur, 1e-8)`, which
/// eigendecomposes `S` and clamps every eigenvalue UP to `floor = 1e-8·λ_max`
/// before solving. Its stated justification is a claim about the DATA, made at a
/// site that only sees the spectrum:
///
/// > on the dead subspace the correct Δβ IS ≈0 (those atoms have no signal), so
/// > the only "inaccuracy" is in directions whose true step is zero
///
/// That holds for a genuinely dead over-complete atom (`β_k → 0`, no gradient
/// either), which is the #1026 case it was written for. It does NOT hold for a
/// direction with SMALL curvature and a REAL gradient — and this thread has
/// already identified exactly such a direction: decoder redistribution leaves
/// the reconstruction exactly invariant (so the data-fit Gauss-Newton term
/// contributes no curvature) while the smoothness penalty is not invariant along
/// it (so the objective genuinely descends there, slowly and forever). For such
/// a direction the true Newton step is `g_i/λ_i`, which is LARGE, and clamping
/// `λ_i` up to `floor` shortens it by `λ_i/floor` — after which the residual can
/// never clear and is bit-stable, because the same clamp fires identically on
/// every iteration.
///
/// This probe rebuilds `S = H_ββ − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)` from the
/// system's own public blocks (`ridge_β = 0`, matching the criterion lane's
/// `solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, …)`), eigendecomposes
/// it, and reports where `g_β` lives relative to the floor. Pre-registered:
///
/// * most of `‖g_β‖²` in sub-floor directions, with a large shortening factor —
///   the PD floor IS the plateau. The solver declares those directions dead
///   while the acceptance gate goes on measuring the residual in them, and that
///   disagreement is the defect: a direction cannot be both too dead to solve
///   and live enough to refuse on.
/// * energy concentrated in healthy (`λ ≫ floor`) directions — the clamp is
///   innocent, the reduction transmits the step it should, and the explanation
///   has to move back to the assembled curvature or the `H_tβ` coupling.
/// * NO eigenvalue below the floor at all — the floored branch never fires here,
///   the hypothesis is refuted outright, and `κ(S)` is not the story. This third
///   outcome is the one worth naming explicitly: the branch is κ-gated, so it is
///   entirely possible this fixture never enters it, in which case every number
///   below describes a code path the fixture does not take.
///
/// Diagnostic only. The assertions are self-checks — that the rebuilt `S` is
/// symmetric and that its eigendecomposition reconstructs it — so a bad rebuild
/// cannot be read as a finding. A non-PD `H_tt` row block or a zero-width
/// `htbeta` slab (which would mean the dense cross-block is not populated on
/// this route, and the rebuild is therefore not the solver's `S`) is REPORTED
/// and the reading abandoned, never patched around.
#[test]
fn zz_measure_k2_wide_p_schur_floor_clamps_the_residual_2080() {
    let z = two_circle_wide_target(96, 96, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 2, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");

    for &warmup in &[32usize, 128] {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        term.run_joint_fit_arrow_schur_for_quasi_laplace(
            z.view(),
            &mut rho_fixed,
            None,
            warmup,
            0.04,
            1.0e-6,
            1.0e-6,
        )
        .expect("inner evidence fit must not hard-error on the K=2 wide-p rung");
        let sys = term
            .assemble_arrow_schur(z.view(), &rho_fixed, None)
            .expect("reassemble at the warmed iterate");
        let k = sys.k;
        if k == 0 {
            eprintln!("[2080-FLOOR] warmup={warmup:>4}: k=0, there is no β border to measure");
            continue;
        }

        // `S = H_ββ − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`, rebuilt from the same
        // blocks the reduction consumes. Anything that would make this NOT the
        // solver's `S` is reported and abandons the reading.
        let mut schur = sys.effective_penalty_op().to_dense();
        let mut rebuild_refused: Option<String> = None;
        for (row_idx, row) in sys.rows.iter().enumerate() {
            if row.htbeta.ncols() != k {
                rebuild_refused = Some(format!(
                    "row {row_idx} carries a {}-column htbeta slab against a border of {k}; the \
                     dense cross-block is not populated on this route, so a rebuild here would \
                     not be the solver's S",
                    row.htbeta.ncols(),
                ));
                break;
            }
            if row.htbeta.nrows() == 0 {
                continue;
            }
            let factor = match row.htt.cholesky(faer::Side::Lower) {
                Ok(factor) => factor,
                Err(error) => {
                    rebuild_refused = Some(format!(
                        "row {row_idx} H_tt is not positive definite ({error:?}); a non-PD row \
                         block at the plateau is itself the finding"
                    ));
                    break;
                }
            };
            // `X = H_tt⁻¹ H_tβ`, then `S -= H_tβᵀ X`.
            let mut solved = row.htbeta.clone();
            factor.solve_mat_in_place(&mut solved);
            let correction = fast_atb(&row.htbeta, &solved);
            schur = schur - correction;
        }
        if let Some(reason) = rebuild_refused {
            eprintln!("[2080-FLOOR] warmup={warmup:>4}: reduced-Schur rebuild abandoned: {reason}");
            continue;
        }

        // Self-check 1: the rebuilt S must be symmetric, or the eigendecomposition
        // below is describing a different operator than the reduction factors.
        let mut asymmetry = 0.0_f64;
        let mut magnitude = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                asymmetry = asymmetry.max((schur[[i, j]] - schur[[j, i]]).abs());
                magnitude = magnitude.max(schur[[i, j]].abs());
            }
        }
        let asymmetry_rel = asymmetry / magnitude.max(f64::MIN_POSITIVE);
        assert!(
            asymmetry_rel <= 1.0e-9,
            "[2080-FLOOR] warmup={warmup}: the rebuilt reduced Schur is not symmetric \
             (max|S−Sᵀ|/max|S| = {asymmetry_rel:.3e}), so it is not the operator the reduction \
             eigendecomposes and nothing below describes the solver"
        );

        let (evals, evecs) = schur
            .eigh(faer::Side::Lower)
            .expect("a symmetric finite reduced Schur must eigendecompose");
        // Self-check 2: `V diag(λ) Vᵀ` must reconstruct `S`, so a mis-oriented
        // eigenvector matrix cannot be read as a residual living in the wrong
        // subspace — the entire finding is a projection onto these columns.
        let mut scaled = evecs.clone();
        for col in 0..k {
            let lambda = evals[col];
            for entry in scaled.column_mut(col) {
                *entry *= lambda;
            }
        }
        // `fast_atb(a, b) = aᵀb`, so passing `Vᵀ` and `(VΛ)ᵀ` yields `V (VΛ)ᵀ = V Λ Vᵀ`.
        let evecs_t = evecs.t().to_owned();
        let scaled_t = scaled.t().to_owned();
        let reconstructed = fast_atb(&evecs_t, &scaled_t);
        let mut reconstruction_gap = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                let gap = (reconstructed[[i, j]] - schur[[i, j]]).abs();
                reconstruction_gap = reconstruction_gap.max(gap);
            }
        }
        let reconstruction_rel = reconstruction_gap / magnitude.max(f64::MIN_POSITIVE);
        assert!(
            reconstruction_rel <= 1.0e-9,
            "[2080-FLOOR] warmup={warmup}: the eigendecomposition does not reconstruct the \
             rebuilt Schur (max|VΛVᵀ − S|/max|S| = {reconstruction_rel:.3e}); the projection of \
             g_β onto these columns would be meaningless"
        );

        // The floor the solver would apply, and where `g_β` actually lives.
        let lambda_max = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let lambda_min = evals.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
        let floor = gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR * lambda_max;
        let projected = evecs.t().dot(&sys.gb);
        let total_energy = projected.iter().map(|&v| v * v).sum::<f64>();
        let mut floored_count = 0usize;
        let mut floored_energy = 0.0_f64;
        // The worst shortening the clamp imposes on a direction that actually
        // carries gradient: `|g_i|/λ_i` (true) against `|g_i|/floor` (clamped),
        // i.e. `floor/λ_i`, weighted by whether `g_i` is there at all.
        let mut worst_shortening = 1.0_f64;
        let mut worst_shortening_energy_share = 0.0_f64;
        for idx in 0..k {
            let lambda = evals[idx];
            if lambda.is_finite() && lambda >= floor {
                continue;
            }
            floored_count += 1;
            let energy = projected[idx] * projected[idx];
            floored_energy += energy;
            if lambda.is_finite() && lambda > 0.0 {
                let shortening = floor / lambda;
                if shortening > worst_shortening {
                    worst_shortening = shortening;
                    worst_shortening_energy_share = energy / total_energy.max(f64::MIN_POSITIVE);
                }
            }
        }
        let floored_share = 100.0 * floored_energy / total_energy.max(f64::MIN_POSITIVE);
        eprintln!(
            "[2080-FLOOR] warmup={warmup:>4} k={k} kappa={:.6e} | lambda_max={lambda_max:.6e} \
             lambda_min={lambda_min:.6e} floor={floor:.6e} | floored={floored_count}/{k} \
             carrying {floored_share:.2}% of ‖g_β‖² | worst_shortening={worst_shortening:.6e} \
             (that direction holds {:.2}% of ‖g_β‖²) | sym_rel={asymmetry_rel:.3e} \
             recon_rel={reconstruction_rel:.3e}",
            lambda_max / lambda_min.abs().max(f64::MIN_POSITIVE),
            100.0 * worst_shortening_energy_share,
        );
    }
}

/// Ordinary least-squares slope of `ln y` on `ln x`, plus the largest absolute
/// residual in `ln y`. A cost exponent is only readable when the fit is tight,
/// so the residual travels with the slope and callers report both.
fn log_slope(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert_eq!(xs.len(), ys.len(), "log-log fit needs paired samples");
    assert!(xs.len() >= 2, "a slope needs at least two samples");
    let lx: Vec<f64> = xs.iter().map(|v| v.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|v| v.ln()).collect();
    let count = lx.len() as f64;
    let mx = lx.iter().sum::<f64>() / count;
    let my = ly.iter().sum::<f64>() / count;
    let mut sxy = 0.0_f64;
    let mut sxx = 0.0_f64;
    for idx in 0..lx.len() {
        sxy += (lx[idx] - mx) * (ly[idx] - my);
        sxx += (lx[idx] - mx) * (lx[idx] - mx);
    }
    if sxx == 0.0 {
        // A constant abscissa carries no exponent. Say so rather than returning
        // the 0/0 NaN a reader would have to interpret.
        return (f64::NAN, 0.0);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let mut worst = 0.0_f64;
    for idx in 0..lx.len() {
        worst = worst.max((ly[idx] - (intercept + slope * lx[idx])).abs());
    }
    (slope, worst)
}

/// Median of a small sample.
fn median_of(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    sorted[sorted.len() / 2]
}

/// #2080 COST-EXPONENT MEASUREMENT (zz_measure) — is the wide-`p` criterion cost
/// really CUBIC in ambient width, and if so, in WHAT dimension?
///
/// The thread's headline mechanism ("the REML criterion's dense `p x p`
/// logdet/Hessian work scales ~O(p^3)") is a COST MODEL asserted from two
/// wall-clock points: p=16 converges in 62 s, p=96 does not return in 600 s. A
/// censored upper bound at one width cannot separate `p^2` from `p^3`; those two
/// points only lower-bound the exponent at `ln(600/62)/ln 6 = 1.27`. This probe
/// sweeps the width and fits the LOG SLOPE of each phase against the candidate
/// exponents instead of accepting the stated cubic.
///
/// **Every phase is FIXED WORK.** The neighbouring cost probes time the whole
/// criterion, whose wall clock is (arithmetic per inner iteration) x (a count of
/// inner iterations that is itself a function of `p`) — an exponent read off
/// that is a product of two unseparated factors. Here the inner fit runs on a
/// FIXED warmup budget and the timed phases downstream (assembly, one Newton
/// arrow/Schur solve, the dense exact-A materialization, the two symmetric
/// eigendecompositions) each do a fixed amount of arithmetic per call, so their
/// slopes are arithmetic exponents.
///
/// It also measures the operand the model is asserted about. There is no dense
/// `p x p` matrix on this route: `materialize_exact_hessian_dense` builds a
/// `dim x dim` operator with `dim = total_t + k`, `total_t = rows x latent dim`
/// (INDEPENDENT of `p`) and `k = sum_atoms basis_size x p` (linear in `p`). So a
/// cubic term, if present, is cubic in the BETA BORDER, and the sweep is chosen
/// to cross the `total_t`-dominated / `k`-dominated crossover.
///
/// Third: a generic dense cost argument is false for a STRUCTURED operator. Beta
/// enters the reconstruction linearly and channel-wise, and the penalty assembly
/// writes the smooth block as exactly `G (x) I_p` (`c[[total_t + off + nu*r + oc,
/// total_t + off + mu*r + oc]]`), with the solve path installing an
/// `IdentityRightKroneckerPenaltyOp` rather than a dense `hbb`. If the
/// beta-beta block is numerically Kronecker then `log|A_bb| = p * log|G|` and
/// nothing about the border is intrinsically cubic — the measured cubic would be
/// an artifact of materializing and eigendecomposing a Kronecker factor this
/// route never exploits. The probe reports the two numbers that decide it: the
/// largest cross-channel entry and the largest across-channel spread of the
/// same-channel entries, both relative to the block's own scale.
///
/// Arms are INTERLEAVED (the width order rotates every repetition) because a
/// wall clock measured under contention is not a runtime; the per-repetition
/// table and the min-vs-median slope pair are what say whether the fit is
/// resolved.
///
/// MEASURED 2026-08-01, acn112, `finished in 8.38 s`, n=96, K=1, m=5, warmup=16,
/// 3 interleaved repetitions per width (minimum over reps, seconds):
///
/// ```text
/// p    total_t   k   dim | warm    assemble newton  materialize eigh
/// 12   192      20   212 | 0.1078  0.0014   0.0013  0.0889      0.0154
/// 24   192      20   212 | 0.1721  0.0020   0.0016  0.1335      0.0156
/// 48   192      20   212 | 0.2506  0.0030   0.0016  0.1897      0.0145
/// 96   192      20   212 | 0.3540  0.0061   0.0013  0.2477      0.0131
/// log slope vs p (min / median over reps):
///   warm +0.569/+0.480   assemble +0.714/+0.715   newton -0.004/-0.006
///   materialize +0.494/+0.473   eigh -0.081/-0.038
/// cross_channel_rel ~1e-16, channel_spread_rel ~2e-15 at every width
/// ```
///
/// REPLICATED: an earlier independent run of the same probe on the same box gave
/// `warm +0.574/+0.525, assemble +0.817/+0.754, newton +0.039/+0.039,
/// materialize +0.560/+0.605, eigh +0.049/+0.098`. Every phase agrees to within
/// ~0.1 between runs, and the `eigh` slope straddles zero in both — the sign of a
/// quantity with no `p` dependence to find, not of a measurement that missed one.
/// The Kronecker numbers are bit-identical across runs.
///
/// **The cubic is not there, and neither is the operand.** `dim` is 212 at every
/// width: `total_t = 192 = rows x latent dim` and `k = 20 = 5 basis x frame rank
/// 4`, both independent of `p`, because the decoder is profiled onto a Grassmann
/// frame (#972) whose rank is set by the atom rather than by the ambient width.
/// The `eigh` phase — the one the thread calls cubic — has exponent **-0.08**,
/// i.e. constant. Nothing in either run exceeds **+0.82**; the whole measured
/// wide-`p` cost is SUB-LINEAR over 12 -> 96, and it lives in the phases that
/// stream the `n x p` target (the inner fit and the assembly), not in any dense
/// factorization. The beta-beta block is `G (x) I_r` to round-off, as predicted.
///
/// The criterion itself is NOT on the timed path here, deliberately: at the time
/// of writing `penalized_quasi_laplace_criterion_with_cache` refuses on this
/// same K=1 fixture at p=16 ("objective stalled for 3 consecutive refine
/// rounds", |g|=1.563201e-2 against tolerance 1.012736e-3) in 9.86 s — a refusal, not a cost. The
/// probe records that outcome per width as a READING rather than gating on it,
/// so the cost question stays answerable while the refusal is someone's separate
/// defect.
#[test]
fn zz_measure_wide_p_cost_exponent_2080() {
    let harmonics = 2usize; // m = 1 + 2*2 = 5 basis columns per atom
    let widths = [12usize, 24, 48, 96];
    let reps = 3usize;
    let n = 96usize;
    let warmup = 16usize; // FIXED inner budget: the same work at every width.
    let nw = widths.len();
    let mut warm = vec![vec![0.0_f64; reps]; nw];
    let mut assemble = vec![vec![0.0_f64; reps]; nw];
    let mut newton = vec![vec![0.0_f64; reps]; nw];
    let mut materialize = vec![vec![0.0_f64; reps]; nw];
    let mut eigh = vec![vec![0.0_f64; reps]; nw];
    let mut shape = vec![(0_usize, 0_usize, 0_usize); nw];
    let mut kron = vec![(0.0_f64, 0.0_f64, 0.0_f64); nw];
    // Per-width beta-layout channel stride (0 = the border is not a whole number
    // of basis blocks, i.e. a framed decoder the Kronecker read cannot index).
    let mut chan_of = vec![0_usize; nw];
    let mut criterion_note = vec![String::new(); nw];

    for rep in 0..reps {
        for slot in 0..nw {
            // Rotate the visit order so no width is always measured first.
            let wi = (slot + rep) % nw;
            let p = widths[wi];
            let z = one_circle_wide_target(n, p, 0.05);
            let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
            let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
            let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
                .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
                .expect("seed dispersion is finite and strictly positive");

            // Phase W: the inner evidence fit on a FIXED iteration budget.
            let mut term = base.clone();
            let mut rho_fixed = rho.clone();
            let w0 = std::time::Instant::now();
            term.run_joint_fit_arrow_schur_for_quasi_laplace(
                z.view(),
                &mut rho_fixed,
                None,
                warmup,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .expect("the inner evidence fit must not hard-error on the K=1 wide-p rung");
            warm[wi][rep] = w0.elapsed().as_secs_f64();

            // Phase S: reassembly of the arrow/Schur system at the warmed iterate.
            let s0 = std::time::Instant::now();
            let sys = term
                .assemble_arrow_schur(z.view(), &rho_fixed, None)
                .expect("reassemble at the warmed iterate");
            assemble[wi][rep] = s0.elapsed().as_secs_f64();
            assert!(
                sys.k > 0,
                "p={p}: there must be a beta border to measure, got k=0"
            );

            // Phase N: ONE production inner Newton step through the same entry
            // point and options the criterion's undamped evidence lane uses. Its
            // by-product is the `ArrowFactorCache` the log-det phases consume,
            // which is how this probe reaches them without going through the
            // criterion's convergence gate.
            let options = ArrowSolveOptions::direct()
                .with_gpu_policy(term.gpu_policy)
                .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
                .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
            let n0 = std::time::Instant::now();
            let (_delta_t, _delta_beta, cache) =
                solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                    .expect("the production inner step must solve at the warmed iterate");
            newton[wi][rep] = n0.elapsed().as_secs_f64();

            let total_t = cache.delta_t_len();
            let k = cache.k;

            // Phase M: dense exact-A materialization — `dim` Hessian-vector
            // applies plus the symmetrization sweep.
            let m0 = std::time::Instant::now();
            let a_dense = term
                .materialize_exact_hessian_dense(&rho_fixed, z.view(), &cache)
                .expect("materialize the exact stationarity Hessian");
            materialize[wi][rep] = m0.elapsed().as_secs_f64();
            assert_eq!(
                (a_dense.nrows(), a_dense.ncols()),
                (total_t + k, total_t + k),
                "p={p}: the materialized exact A must be the square joint operator"
            );

            // Phase E: the two dense symmetric eigendecompositions the exact
            // observed-information log-dets perform (joint `A`, then the
            // coordinate block `A_tt`). Run here directly rather than through
            // `exact_observed_information_log_dets` so an indefinite-A REFUSAL
            // reports as a reading below instead of destroying the cost sample.
            let a_tt = a_dense.slice(s![..total_t, ..total_t]).to_owned();
            let e0 = std::time::Instant::now();
            let (joint_eigs, _joint_vecs) = a_dense
                .eigh(faer::Side::Lower)
                .expect("the joint exact-A eigendecomposition must succeed");
            let (tt_eigs, _tt_vecs) = a_tt
                .eigh(faer::Side::Lower)
                .expect("the coordinate-block eigendecomposition must succeed");
            eigh[wi][rep] = e0.elapsed().as_secs_f64();
            assert!(
                joint_eigs.len() == total_t + k && tt_eigs.len() == total_t,
                "p={p}: both timed eigendecompositions must have produced a full spectrum"
            );

            shape[wi] = (total_t, k, total_t + k);

            if rep == 0 {
                // What the production route does with those spectra, as a
                // READING: `Ok` or the typed refusal, never a gate.
                criterion_note[wi] = match term.exact_observed_information_log_dets(
                    &rho_fixed,
                    z.view(),
                    &cache,
                ) {
                    Ok((joint, tt)) => format!("log_dets ok joint={joint:.6e} tt={tt:.6e}"),
                    Err(error) => format!("log_dets refused: {error}"),
                };

                // The `off + basis*chan + channel` layout is `chan == p` only
                // when no factored frame is engaged; a framed decoder narrows the
                // per-basis stride to the frame rank. Derive the stride from the
                // border itself rather than assuming it, so a framed run reports
                // its own layout instead of indexing a structure that is not there.
                let beta_dim = term.beta_dim();
                let basis_total = if p > 0 && beta_dim % p == 0 {
                    beta_dim / p
                } else {
                    0
                };
                let chan = if basis_total > 0 && k % basis_total == 0 {
                    k / basis_total
                } else {
                    0
                };
                chan_of[wi] = chan;
                if chan > 0 {
                    let stride = chan;
                    let mut block_max = 0.0_f64;
                    let mut cross_channel = 0.0_f64;
                    let mut channel_spread = 0.0_f64;
                    for b1 in 0..basis_total {
                        for b2 in 0..basis_total {
                            let mut lo = f64::INFINITY;
                            let mut hi = f64::NEG_INFINITY;
                            for c1 in 0..stride {
                                let same = a_dense[[
                                    total_t + b1 * stride + c1,
                                    total_t + b2 * stride + c1,
                                ]];
                                block_max = block_max.max(same.abs());
                                lo = lo.min(same);
                                hi = hi.max(same);
                                for c2 in 0..stride {
                                    if c1 == c2 {
                                        continue;
                                    }
                                    let off = a_dense[[
                                        total_t + b1 * stride + c1,
                                        total_t + b2 * stride + c2,
                                    ]];
                                    block_max = block_max.max(off.abs());
                                    cross_channel = cross_channel.max(off.abs());
                                }
                            }
                            channel_spread = channel_spread.max(hi - lo);
                        }
                    }
                    kron[wi] = (cross_channel, channel_spread, block_max);
                }
            }
        }
    }

    let xs: Vec<f64> = widths.iter().map(|&p| p as f64).collect();
    let dims: Vec<f64> = shape.iter().map(|s| s.2 as f64).collect();
    eprintln!(
        "[2080-EXP] n={n} K=1 harmonics={harmonics} warmup={warmup} reps={reps} interleaved \
         (the width order rotates per rep)"
    );
    for wi in 0..nw {
        let (total_t, k, dim) = shape[wi];
        let (cross_channel, channel_spread, block_max) = kron[wi];
        let scale = block_max.max(f64::MIN_POSITIVE);
        eprintln!(
            "[2080-EXP] p={:>3} total_t={total_t:>4} k={k:>5} dim={dim:>5} | \
             warm min={:8.4}s | assemble min={:8.4}s | newton min={:8.4}s | \
             materialize min={:8.4}s | eigh min={:8.4}s",
            widths[wi],
            warm[wi].iter().copied().fold(f64::INFINITY, f64::min),
            assemble[wi].iter().copied().fold(f64::INFINITY, f64::min),
            newton[wi].iter().copied().fold(f64::INFINITY, f64::min),
            materialize[wi].iter().copied().fold(f64::INFINITY, f64::min),
            eigh[wi].iter().copied().fold(f64::INFINITY, f64::min),
        );
        eprintln!(
            "[2080-KRON] p={:>3} border_k={k} chan_stride={} block_max={block_max:.6e} \
             cross_channel_rel={:.3e} channel_spread_rel={:.3e}",
            widths[wi],
            chan_of[wi],
            cross_channel / scale,
            channel_spread / scale,
        );
        eprintln!("[2080-LOGDET] p={:>3} {}", widths[wi], criterion_note[wi]);
        for rep in 0..reps {
            eprintln!(
                "[2080-REPS] p={:>3} rep={rep} warm={:.4}s assemble={:.4}s newton={:.4}s \
                 materialize={:.4}s eigh={:.4}s",
                widths[wi],
                warm[wi][rep],
                assemble[wi][rep],
                newton[wi][rep],
                materialize[wi][rep],
                eigh[wi][rep],
            );
        }
    }

    for (label, series) in [
        ("warm", &warm),
        ("assemble", &assemble),
        ("newton", &newton),
        ("materialize", &materialize),
        ("eigh", &eigh),
    ] {
        let mins: Vec<f64> = series
            .iter()
            .map(|r| r.iter().copied().fold(f64::INFINITY, f64::min))
            .collect();
        let meds: Vec<f64> = series.iter().map(|r| median_of(r)).collect();
        let (slope_min, resid_min) = log_slope(&xs, &mins);
        let (slope_med, resid_med) = log_slope(&xs, &meds);
        let dim_varies = dims.iter().any(|d| *d != dims[0]);
        if dim_varies {
            let (slope_dim, resid_dim) = log_slope(&dims, &mins);
            eprintln!(
                "[2080-SLOPE] {label:>12} vs p: min={slope_min:+.3} (max ln-resid \
                 {resid_min:.3}) med={slope_med:+.3} (max ln-resid {resid_med:.3}) | vs dim: \
                 min={slope_dim:+.3} (max ln-resid {resid_dim:.3})"
            );
        } else {
            eprintln!(
                "[2080-SLOPE] {label:>12} vs p: min={slope_min:+.3} (max ln-resid \
                 {resid_min:.3}) med={slope_med:+.3} (max ln-resid {resid_med:.3}) | vs dim: \
                 NOT FITTABLE, dim is CONSTANT at {} across the whole sweep",
                dims[0]
            );
        }
    }

    for wi in 0..nw {
        assert!(
            warm[wi].iter().all(|v| v.is_finite() && *v > 0.0)
                && materialize[wi].iter().all(|v| v.is_finite() && *v > 0.0)
                && eigh[wi].iter().all(|v| v.is_finite() && *v > 0.0),
            "p={}: every timed phase must have produced a positive finite duration",
            widths[wi]
        );
    }

    // THE LOAD-BEARING GATE, and the refutation of the thread's cost model.
    // The exact-A operator this route materializes and eigendecomposes has
    // dimension `total_t + k`, and NEITHER term grows with `p`: `total_t` is
    // `rows x latent dim`, and the decoder is profiled onto a Grassmann frame
    // (#972) whose rank `r` is set by the atom, not by the ambient width, so the
    // border is `sum_atoms basis_size x r`. A shape that is identical at p=12 and
    // p=96 cannot carry a `p^3` term, whatever the wall clock says. This is a
    // structural equality, not a timing bound, so it is a deterministic gate: a
    // future change that un-profiles the border (`border_frame_rank == p`) goes
    // red HERE, where the cost consequence is stated, rather than surfacing as a
    // wide-`p` hang someone has to re-diagnose.
    for wi in 1..nw {
        assert_eq!(
            shape[wi], shape[0],
            "the exact-A route's operator dimension must not depend on the ambient width: \
             p={} gives (total_t, k, dim) = ({}, {}, {}) but p={} gives ({}, {}, {})",
            widths[0],
            shape[0].0,
            shape[0].1,
            shape[0].2,
            widths[wi],
            shape[wi].0,
            shape[wi].1,
            shape[wi].2,
        );
    }

    // The beta-beta block is exactly `G (x) I_r` in the `basis*r + channel`
    // layout: measured at 1e-16 (cross-channel) and 2e-15 (across-channel
    // spread) relative to the block's own scale, i.e. round-off. Gate it well
    // clear of that so the structure is a fact the tree keeps, not one this
    // probe happened to observe once.
    for wi in 0..nw {
        if chan_of[wi] == 0 {
            continue;
        }
        let (cross_channel, channel_spread, block_max) = kron[wi];
        let scale = block_max.max(f64::MIN_POSITIVE);
        assert!(
            block_max > 0.0,
            "p={}: an indexable beta-beta block (channel stride {}) must be nonzero for its \
             Kronecker structure to mean anything",
            widths[wi],
            chan_of[wi]
        );
        assert!(
            cross_channel / scale <= 1.0e-12 && channel_spread / scale <= 1.0e-12,
            "p={}: the beta-beta block must be `G (x) I_{}` to round-off -- beta enters the \
             reconstruction linearly and channel-wise -- but cross_channel_rel={:.3e} and \
             channel_spread_rel={:.3e}",
            widths[wi],
            chan_of[wi],
            cross_channel / scale,
            channel_spread / scale,
        );
    }
}
