//! #2015 SAE convergence keystone — DECISIVE inner-solve measurement.
//!
//! The behavior-anchored two-block SAE fit's inner (t, β) solve does not certify
//! convergence on real 600-row data: the latest REAL verdict (GHA validate-one
//! 29144079962) froze at `‖g‖=1.410137` / `‖Π⊥gauge g‖=1.387610` against a tight
//! `2.76e-3` tolerance after 1050 inner iterations, with raw ≈ gauge (only
//! ~0.023 gauge mass) so the residual is PHYSICAL, not a gauge-nullspace artifact.
//!
//! Two probes here route the whole keystone fix. Both run on a FRAME-ACTIVE
//! wide-p K=4 in-regime config (decoder dof ≪ data: K·3·8 = 96 decoder params
//! vs 200·8 = 1600 data cells) with a deliberately OFF-manifold high-residual
//! target, so the inner optimum has a genuine physical residual like the real
//! 600-row case — NOT the near-exact 48-row H2-degenerate regime that misleads.
//!
//! 1. `inner_gnorm_vs_budget_trajectory_2015` — run the inner solve at increasing
//!    iteration budgets and log ‖g‖ / ‖Π⊥gauge g‖ at each. If ‖g‖ keeps dropping
//!    → under-budgeted (raise the cap). If ‖g‖ FLOORS regardless of budget →
//!    genuine wall (solver-side 2nd-order/trust-region fix, or envelope-accept).
//!
//! 2. `inner_gradient_matches_penalized_objective_fd_2015` — a DIRECT central-FD
//!    of the inner penalized objective (`penalized_objective_total`) wrt θ and β
//!    vs the assembled analytic inner gradient (`gt`/`gb`). This does NOT route
//!    through any resolve, so — unlike an outer-FD-through-a-resolve — it CANNOT
//!    be fooled by an under-converged inner budget. analytic == FD ⇒ the inner
//!    gradient is CORRECT and the floor is a real non-stationarity the SOLVER
//!    can't clear; analytic ≠ FD ⇒ a real inner-term desync to fix in the term.
//!
//! Diagnostic `eprintln!` tables (log::warn is dropped by the test harness); grep
//! the CI log for the `[2015-TRAJ]` / `[2015-INNERFD]` channel tags.

use super::*;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Frame-active wide-p K=4 in-regime SAE term with an off-manifold high-residual
/// target and a rho carrying live smoothing/ARD (so the inner optimum is a real,
/// non-degenerate physical residual — the 600-row regime, not the 48-row one).
fn wide_p_k4_in_regime() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n_obs = 200usize;
    let p_out = 8usize;
    let k = 4usize;

    let mut atoms = Vec::with_capacity(k);
    let mut coord_blocks = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let f = (atom_idx as f64) + 1.0;
        // Distinct, well-spread d=1 coordinates per atom so the euclidean patch
        // basis (degree-2, 3 columns) is genuinely non-degenerate across rows.
        let mut coords = Array2::<f64>::zeros((n_obs, 1));
        for row in 0..n_obs {
            let x = row as f64;
            coords[[row, 0]] = -0.6 + 0.011 * f * x + 0.05 * (0.03 * f * x + 0.4 * f).sin();
        }
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let n_basis = phi.ncols();
        // Distinct nonzero decoders per atom, wide p_out.
        let decoder = Array2::<f64>::from_shape_fn((n_basis, p_out), |(m, c)| {
            0.12 * f * ((m + 1) as f64) - 0.04 * (c as f64)
                + 0.03 * (0.7 * f + 0.2 * m as f64).cos()
        });
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("euclid_d1_{atom_idx}"),
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(n_basis),
        )
        .unwrap()
        .with_basis_second_jet(evaluator);
        atoms.push(atom);
        coord_blocks.push(coords);
    }

    let manifolds = vec![LatentManifold::Euclidean; k];
    let logits =
        Array2::<f64>::from_shape_fn((n_obs, k), |(r, c)| 0.25 * (c as f64) - 0.002 * (r as f64));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();

    // OFF-manifold high-residual target: a rich multi-harmonic field a rank-3
    // degree-2 euclidean K=4 model cannot reconstruct, so the inner optimum has a
    // genuine physical residual (the 600-row real regime), not a near-exact fit.
    let target = Array2::<f64>::from_shape_fn((n_obs, p_out), |(r, c)| {
        let x = (r as f64) / n_obs as f64;
        let y = (c as f64) + 1.0;
        0.8 * (6.3 * x + 0.5 * y).sin() + 0.5 * (12.1 * x - 0.3 * y).cos()
            - 0.35 * (19.7 * x * y * 0.1).sin()
            + 0.2 * ((r * p_out + c) as f64 % 7.0)
    });

    // Live smoothing + ARD so the inner optimum is a real penalized fixed point.
    let log_ard = vec![Array1::from_elem(1, 0.0_f64); k];
    let rho = SaeManifoldRho::new(-1.0, (1.0e-2_f64).ln(), log_ard);
    (term, target, rho)
}

/// PROBE (mechanism ID): ‖g‖ vs inner-iteration budget, with per-budget
/// discriminators. The round-1 trajectory established NO FLOOR + a catastrophic
/// non-monotone spike (67.9→17.1→22.1→2.887e8@80→0.115→…→2.25e-3@1280). The
/// Armijo line search already gates the accepted step on OBJECTIVE descent
/// (`run_joint_fit_arrow_schur` line-search + proximal-correction both require a
/// strict decrease vs `pre_step_total`), so the 2.887e8 is a GRADIENT spike, NOT
/// an accepted objective increase. This probe distinguishes the two remaining
/// mechanisms by logging, at each returned iterate:
///   - `pen_obj` — is the objective monotone across budgets (⇒ the spike is a
///     non-stationary perturbation at an objective-good state) or does it also
///     spike (⇒ a guard was bypassed)?
///   - `reseeds` (`dictionary_cocollapse_reseeds`) — did a NON-objective-guarded
///     collapse reseed (`enforce_active_mass_guard` / `enforce_decoder_norm_guard`
///     / `enforce_structural_coherence_guard`, applied post-acceptance OUTSIDE the
///     line search) fire at the spike? (mechanism A)
///   - `max|coord|` — did an Armijo-accepted step drive a coordinate to a huge
///     value where the degree-2 basis makes ‖g‖ explode? (mechanism B)
/// A distinguishes {reseeds↑, coords modest}; B distinguishes {reseeds=0, coords
/// huge}. The fix differs: A ⇒ make the reseeds gradient-aware / quiescent near
/// convergence; B ⇒ a coordinate trust-region / reconditioner.
///
/// Asserts the round-1 "no floor" fact as a real regression guard (so this is not
/// a print-only probe): at the largest budget the inner solve DOES reach a small
/// ‖g‖ — the solver CAN converge, the wall is trajectory instability, not a
/// genuine stationary floor. If a future change reintroduces a floor this fails.
#[test]
fn inner_gnorm_vs_budget_trajectory_2015() {
    let (base, target, rho) = wide_p_k4_in_regime();
    let lr = 0.4;
    let ridge = 1.0e-6;

    eprintln!(
        "[2015-TRAJ] wide-p K=4 in-regime: n_obs={} p_out={} k={} beta_dim={} coord_dim={}",
        base.n_obs(),
        base.output_dim(),
        base.k_atoms(),
        base.beta_dim(),
        base.n_obs() * base.k_atoms(),
    );
    eprintln!(
        "[2015-TRAJ] budget | ‖g‖(raw) | ‖Π⊥gauge g‖ | pen_obj | max|coord| | reseeds | \
         decnorm[min,max] | fp"
    );

    let budgets = [10usize, 20, 40, 80, 160, 320, 640, 1280];
    let mut prev: Option<f64> = None;
    let mut final_g = f64::INFINITY;
    for &budget in &budgets {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        let outcome = term
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                target.view(),
                &mut rho_fixed,
                None,
                budget,
                lr,
                ridge,
                ridge,
            )
            .expect("inner evidence fit must not hard-error");
        let sys = term
            .assemble_arrow_schur(target.view(), &rho_fixed, None)
            .expect("reassemble at fitted iterate");
        let gsq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        let g = gsq.sqrt();
        let qg = term.quotient_gradient_norm_from_system(&sys, gsq, &rho_fixed.lambda_smooth_vec());
        let pen_obj = term
            .penalized_objective_total(target.view(), &rho_fixed, None, 1.0)
            .unwrap_or(f64::NAN);
        let mut max_abs_coord = 0.0_f64;
        for atom_idx in 0..term.k_atoms() {
            for &v in term.assignment.coords[atom_idx].as_flat().iter() {
                max_abs_coord = max_abs_coord.max(v.abs());
            }
        }
        let mut min_dec = f64::INFINITY;
        let mut max_dec = 0.0_f64;
        for atom in &term.atoms {
            let nrm = atom
                .decoder_coefficients
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            min_dec = min_dec.min(nrm);
            max_dec = max_dec.max(nrm);
        }
        let delta = match prev {
            Some(p) => format!("Δ‖g‖={:+.3e}", g - p),
            None => "Δ‖g‖=—".to_string(),
        };
        eprintln!(
            "[2015-TRAJ] {budget:>5} | {g:.6e} | {qg:.6e} | {pen_obj:.6e} | {max_abs_coord:.4e} | \
             {} | [{min_dec:.3e},{max_dec:.3e}] | fp={} | {delta}",
            term.dictionary_cocollapse_reseeds, outcome.fixed_point,
        );
        prev = Some(g);
        final_g = g;
    }
    eprintln!(
        "[2015-TRAJ] VERDICT: mechanism A (reseeds↑, coords modest) vs B (reseeds=0, coords huge)."
    );

    // Round-1 "no floor" regression guard: the inner solve reaches a small ‖g‖ at
    // the largest budget — convergence is achievable; the wall is trajectory
    // instability, not a stationary floor. (Round-1 measured 2.25e-3 at 1280.)
    assert!(
        final_g < 1.0e-1,
        "inner solve reintroduced a ‖g‖ floor: final ‖g‖={final_g:.6e} at budget 1280 \
         (round-1 reached 2.25e-3 — convergence must remain achievable)"
    );
}

/// Trap-immune gate: the assembled analytic inner gradient (`gt`/`gb`) vs a
/// DIRECT central-FD of `penalized_objective_total` at a well-converged iterate.
/// No resolve is on the path, so an under-converged budget cannot fool this: it
/// isolates a genuine inner-term desync from a solver-side non-stationarity.
#[test]
fn inner_gradient_matches_penalized_objective_fd_2015() {
    let (base, target, rho) = wide_p_k4_in_regime();

    // Drive well into the floor first so the FD is taken at the same physical
    // iterate the trajectory probe stalls on.
    let mut term = base;
    let mut rho_fixed = rho.clone();
    term.run_joint_fit_arrow_schur_for_quasi_laplace(
        target.view(),
        &mut rho_fixed,
        None,
        640,
        0.4,
        1.0e-6,
        1.0e-6,
    )
    .expect("inner evidence fit must not hard-error");
    let rho = rho_fixed;

    // The reference-function Gram is fixed by construction, so every coordinate
    // perturbation below differentiates the same objective as the assembled
    // analytic coordinate gradient.
    let mut base = term;
    // Freeze the transient collinearity gates on the base and re-install them on
    // every perturbation clone (the custom `Clone` resets them to `None`), exactly
    // as the optimizer's snapshot/restore does — otherwise the FD silently omits
    // the frozen-gate repulsion/barrier gradients `gb` legitimately carries.
    base.refresh_decoder_repulsion_gate();
    base.refresh_barrier_coactivation_gate();
    let base = base;
    let reinstall = |t: &mut SaeManifoldTerm| {
        t.decoder_repulsion_gate = base.decoder_repulsion_gate.clone();
        t.barrier_coactivation_gate = base.barrier_coactivation_gate.clone();
    };

    let obj = |t: &SaeManifoldTerm| -> f64 {
        t.penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("inner penalized objective")
    };

    let mut assembled = base.clone();
    let sys = assembled
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("assemble at converged iterate");

    let eps = 1.0e-6;

    // Coordinate block: gt.
    let coord_offsets = base.assignment.coord_offsets();
    let mut worst_coord_rel = 0.0_f64;
    let mut worst_coord_abs = 0.0_f64;
    let mut worst_coord: (f64, f64, usize, usize) = (0.0, 0.0, 0, 0);
    for atom_idx in 0..base.k_atoms() {
        let off = coord_offsets[atom_idx];
        let d = base.assignment.coords[atom_idx].latent_dim();
        let base_flat = base.assignment.coords[atom_idx].as_flat().clone();
        let n_atom = base_flat.len() / d;
        for row in 0..n_atom {
            for axis in 0..d {
                let lin = row * d + axis;

                let mut plus = base.clone();
                reinstall(&mut plus);
                let mut fp = base_flat.clone();
                fp[lin] += eps;
                plus.assignment.coords[atom_idx].set_flat(fp.view());
                let cp = plus.assignment.coords[atom_idx].as_matrix();
                plus.atoms[atom_idx].refresh_basis(cp.view()).unwrap();
                let op = obj(&plus);

                let mut minus = base.clone();
                reinstall(&mut minus);
                let mut fm = base_flat.clone();
                fm[lin] -= eps;
                minus.assignment.coords[atom_idx].set_flat(fm.view());
                let cm = minus.assignment.coords[atom_idx].as_matrix();
                minus.atoms[atom_idx].refresh_basis(cm.view()).unwrap();
                let om = obj(&minus);

                let fd = (op - om) / (2.0 * eps);
                let analytic = sys.rows[row].gt[off + axis];
                let abs = (analytic - fd).abs();
                let rel = abs / analytic.abs().max(fd.abs()).max(1.0e-9);
                if rel > worst_coord_rel {
                    worst_coord_rel = rel;
                    worst_coord = (analytic, fd, atom_idx, row * d + axis);
                }
                worst_coord_abs = worst_coord_abs.max(abs);
            }
        }
    }

    // Decoder block: gb.
    let beta = base.flatten_beta();
    let mut worst_beta_rel = 0.0_f64;
    let mut worst_beta_abs = 0.0_f64;
    let mut worst_beta: (f64, f64, usize) = (0.0, 0.0, 0);
    for beta_idx in 0..beta.len() {
        let mut bp = beta.clone();
        bp[beta_idx] += eps;
        let mut plus = base.clone();
        reinstall(&mut plus);
        plus.set_flat_beta(bp.view()).unwrap();
        let op = obj(&plus);

        let mut bm = beta.clone();
        bm[beta_idx] -= eps;
        let mut minus = base.clone();
        reinstall(&mut minus);
        minus.set_flat_beta(bm.view()).unwrap();
        let om = obj(&minus);

        let fd = (op - om) / (2.0 * eps);
        let analytic = sys.gb[beta_idx];
        let abs = (analytic - fd).abs();
        let rel = abs / analytic.abs().max(fd.abs()).max(1.0e-9);
        if rel > worst_beta_rel {
            worst_beta_rel = rel;
            worst_beta = (analytic, fd, beta_idx);
        }
        worst_beta_abs = worst_beta_abs.max(abs);
    }

    let gsq = SaeManifoldTerm::system_grad_norm_sq(&sys);
    eprintln!(
        "[2015-INNERFD] iterate ‖g‖={:.6e}  base_obj={:.6e}",
        gsq.sqrt(),
        obj(&base),
    );
    eprintln!(
        "[2015-INNERFD] COORD gt: worst rel_err={worst_coord_rel:.3e} abs_err={worst_coord_abs:.3e} \
         at (atom={},lin={}) analytic={:.6e} fd={:.6e}",
        worst_coord.2, worst_coord.3, worst_coord.0, worst_coord.1,
    );
    eprintln!(
        "[2015-INNERFD] DECODER gb: worst rel_err={worst_beta_rel:.3e} abs_err={worst_beta_abs:.3e} \
         at beta_idx={} analytic={:.6e} fd={:.6e}",
        worst_beta.2, worst_beta.0, worst_beta.1,
    );
    eprintln!(
        "[2015-INNERFD] VERDICT: analytic==FD ⇒ inner gradient CORRECT (floor is solver-side, \
         not a term bug); analytic≠FD ⇒ real inner-term desync."
    );

    // The gradient the solver descends MUST be the gradient of the objective it
    // prices. A gross mismatch here (rel_err ≫ 1e-3 at both blocks) would be a
    // term bug; a match certifies the ‖g‖ floor as a real non-stationarity.
    assert!(
        worst_coord_rel <= 1.0e-3 || worst_coord_abs <= 1.0e-5,
        "inner coord gradient gt desyncs from penalized-objective central FD: \
         rel={worst_coord_rel:.3e} abs={worst_coord_abs:.3e}"
    );
    assert!(
        worst_beta_rel <= 1.0e-3 || worst_beta_abs <= 1.0e-5,
        "inner decoder gradient gb desyncs from penalized-objective central FD: \
         rel={worst_beta_rel:.3e} abs={worst_beta_abs:.3e}"
    );
}
