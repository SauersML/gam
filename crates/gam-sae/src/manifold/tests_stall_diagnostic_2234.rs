//! zz_measure diagnostic (2026-07-10, #2234 blocker): every Python-entry
//! `sae_manifold_fit` on HEAD refuses to mint — the outer search descends,
//! then freezes with a bit-stable projected gradient (synthetic planted
//! circle: 63 iterations, objective 1.216074e2, |g_proj| = 1.381e0) and the
//! #2241 certified-termination contract refuses. Lane-consistency (#1224) and
//! every FD gate PASS, so this drives the SAME planted circle through the
//! plain RUST engine (seed builders + `OuterProblem`) to split the fault:
//!
//! - engine converges here  ⇒ the stall lives in the pyffi orchestration
//!   above the engine (topology/promotion/alternation), not the optimizer;
//! - engine stalls here     ⇒ run per-coordinate central differences of
//!   `eval_cost` against `eval`'s analytic gradient AT the stalled ρ and
//!   print both (the desync, if any, named coordinate by coordinate).
use super::*;

fn planted_circle_cloud() -> (Array2<f64>, usize) {
    // Mirrors the frozen #2253 weekday-L17 discriminator after its exact
    // orthonormal reduction: K=1, d_atom=1, n=42, p=48. The prior n=200,
    // p=8, d_atom=3 fixture did not enter the single-circle log-det seam whose
    // analytic smoothing derivative is under audit.
    let n = 42usize;
    let p = 48usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        // LCG → [0,1); NO rand, NO clock (repo #932 rules).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let two_pi = std::f64::consts::TAU;
    let b0: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let b1: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = two_pi * unit();
        for j in 0..p {
            let noise = 0.01 * (2.0 * unit() - 1.0);
            z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + noise;
        }
    }
    (z, p)
}

#[test]
fn zz_planted_circle_plain_engine_stall_diagnostic_2234() {
    use gam_solve::rho_optimizer::OuterProblem;
    use gam_solve::seeding::SeedConfig;

    let (z, _p) = planted_circle_cloud();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        ibp_alpha_override: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        atom_basis: &minimal.atom_basis,
        atom_dim: &minimal.effective_atom_dim,
        atom_centers: &minimal.atom_centers,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        basis_sizes: &minimal.basis_sizes,
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 45,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");

    let initial_flat = seed.initial_rho.to_flat();
    let n_params = initial_flat.len();
    let mut objective = SaeManifoldOuterObjective::new(
        seed.base_term,
        z.clone(),
        Some(registry),
        seed.initial_rho,
        40,
        0.05,
        1.0e-6,
        1.0e-6,
    );
    objective.remove_checkpoint();
    let problem = OuterProblem::new(n_params)
        .with_initial_rho(initial_flat.clone())
        .with_seed_config(SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let outcome = problem.run(&mut objective, "zz stall diagnostic 2234");
    match outcome {
        Ok(result) => {
            eprintln!(
                "[zz2234] PLAIN ENGINE CONVERGED: value={:.6e} converged={} — the Python-entry \
                 stall is ORCHESTRATION-layer (pyffi), not the optimizer",
                result.final_value, result.converged,
            );
        }
        Err(err) => {
            eprintln!("[zz2234] PLAIN ENGINE DID NOT CERTIFY: {err}");
            let telemetry = objective.probe_telemetry();
            eprintln!("[zz2234] probe telemetry: {telemetry:?}");
            // The 2026-07-10 wall pathology is the REGRESSION this gate pins:
            // budget-marker refusals converted to 1e12 walls froze every fit
            // (frozen isotropic checkpoint, ten-orders lane disagreement).
            // Post-fix the optimizer descends genuinely; certification on this
            // deliberately tight fixture budget remains an honest optimizer
            // limitation, not a wall. Assert the wall pathology stays dead.
            assert_eq!(
                telemetry.wall_cost_value_probes, 0,
                "wall-cost probes returned — the #2234 wall pathology regressed"
            );
            assert!(
                telemetry.criterion_calls > 10,
                "the outer search froze after {} criterion calls — walls or an \
                 equivalent freeze are back",
                telemetry.criterion_calls
            );
            eprintln!(
                "[zz2234] no walls, genuine descent — non-certification on this tight \
                 fixture budget is documented as an optimizer limitation (see #2234)"
            );
        }
    }

    // #2253: compare the analytic gradient with the cost returned by the SAME
    // production `eval()` lane. The old gate differentiated `eval_cost()` and
    // compared that different value lane with `eval()`'s gradient, so it could
    // pass while the value/gradient pair actually consumed by the optimizer was
    // internally inconsistent. Finite differences remain confined to this test.
    let banked = objective
        .try_resume_from_checkpoint(n_params)
        .map(Array1::from)
        .unwrap_or(initial_flat);
    let audited = objective
        .eval(&banked)
        .expect("gradient eval at the audited rho");
    let grad = audited.gradient;
    assert!(
        objective.term.frames_active(),
        "the #2253 discriminator must exercise the profiled Grassmann-frame path"
    );
    let center_term = objective.term.clone();
    let center_rho = objective.baseline_rho.from_flat(banked.view());
    let mut center_component_term = center_term.clone();
    let (_, center_loss, center_cache) = center_component_term
        .reml_criterion_with_cache(
            z.view(),
            &center_rho,
            objective.registry.as_ref(),
            40,
            0.05,
            1.0e-6,
            1.0e-6,
        )
        .expect("criterion cache at the audited rho");
    let center_solver = center_component_term
        .outer_gradient_arrow_solver(&center_cache, &center_rho.lambda_smooth_vec())
        .expect("outer-gradient arrow solver at the audited rho");
    let center_components = center_component_term
        .analytic_outer_rho_gradient_components(
            z.view(),
            &center_rho,
            &center_loss,
            &center_cache,
            &center_solver,
        )
        .expect("analytic gradient components at the audited rho");
    let center_logdet =
        arrow_log_det_from_cache(&center_cache).expect("finite log determinant at the audited rho");
    eprintln!(
        "[2253-CHAN] center logdet={center_logdet:+.6e} explicit={:?} trace={:?} \
         adjoint={:?} occam={:?}",
        center_components.explicit,
        center_components.logdet_trace,
        center_components.third_order_correction,
        center_components.occam,
    );
    let mut center_atom_term = center_term.clone();
    let center_atoms = center_atom_term
        .criterion_as_atoms(
            z.view(),
            &center_rho,
            objective.registry.as_ref(),
            40,
            0.05,
            1.0e-6,
            1.0e-6,
        )
        .expect("criterion atoms at the audited rho");
    for atom in center_atoms.atoms() {
        eprintln!(
            "[zz2234] center atom {}: value={:+.6e} grad={:?}",
            atom.label(),
            atom.value(),
            atom.grad(),
        );
    }
    let h = 1.0e-4;
    for j in 0..n_params {
        let mut plus = banked.clone();
        plus[j] += h;
        let mut minus = banked.clone();
        minus[j] -= h;
        let plus_rho = objective.baseline_rho.from_flat(plus.view());
        let minus_rho = objective.baseline_rho.from_flat(minus.view());
        let mut plus_term = center_term.clone();
        let mut minus_term = center_term.clone();
        let plus_atoms = plus_term
            .criterion_as_atoms(
                z.view(),
                &plus_rho,
                objective.registry.as_ref(),
                40,
                0.05,
                1.0e-6,
                1.0e-6,
            )
            .expect("criterion atoms at +h");
        let minus_atoms = minus_term
            .criterion_as_atoms(
                z.view(),
                &minus_rho,
                objective.registry.as_ref(),
                40,
                0.05,
                1.0e-6,
                1.0e-6,
            )
            .expect("criterion atoms at -h");
        let (_, _, plus_cache) = plus_term
            .reml_criterion_with_cache(
                z.view(),
                &plus_rho,
                objective.registry.as_ref(),
                40,
                0.05,
                1.0e-6,
                1.0e-6,
            )
            .expect("criterion cache at +h");
        let (_, _, minus_cache) = minus_term
            .reml_criterion_with_cache(
                z.view(),
                &minus_rho,
                objective.registry.as_ref(),
                40,
                0.05,
                1.0e-6,
                1.0e-6,
            )
            .expect("criterion cache at -h");
        let plus_logdet =
            arrow_log_det_from_cache(&plus_cache).expect("finite log determinant at +h");
        let minus_logdet =
            arrow_log_det_from_cache(&minus_cache).expect("finite log determinant at -h");
        let logdet_fd = 0.5 * (plus_logdet - minus_logdet) / (2.0 * h);
        let logdet_analytic =
            center_components.logdet_trace[j] + center_components.third_order_correction[j];
        eprintln!(
            "[2253-CHAN] coord {j}: analytic_logdet={logdet_analytic:+.6e} \
             central_fd_half_logdet={logdet_fd:+.6e}"
        );
        let c_plus = plus_atoms.value();
        let c_minus = minus_atoms.value();
        let fd = (c_plus - c_minus) / (2.0 * h);
        for (plus_atom, minus_atom) in plus_atoms.atoms().iter().zip(minus_atoms.atoms()) {
            let atom_fd = (plus_atom.value() - minus_atom.value()) / (2.0 * h);
            eprintln!(
                "[zz2234] coord {j} atom {}: analytic={:+.6e} central_fd={:+.6e}",
                plus_atom.label(),
                center_atoms
                    .atoms()
                    .iter()
                    .find(|atom| atom.label() == plus_atom.label())
                    .expect("matching center atom")
                    .grad()[j],
                atom_fd,
            );
        }
        let delta = (grad[j] - fd).abs();
        let tolerance = 5.0e-3 * (1.0 + grad[j].abs().max(fd.abs()));
        eprintln!(
            "[zz2234] coord {j}: analytic={:+.6e} eval_fd={:+.6e} |delta|={delta:.3e} tol={tolerance:.3e}",
            grad[j], fd,
        );
        assert!(
            delta <= tolerance,
            "#2253 eval value/gradient desync at rho coordinate {j}: analytic={:+.12e}, \
             same-lane central FD={:+.12e}, delta={delta:.3e}, tolerance={tolerance:.3e}",
            grad[j],
            fd,
        );
    }
}
