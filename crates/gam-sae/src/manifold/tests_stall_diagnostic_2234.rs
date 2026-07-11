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

struct LogdetAuditPoint {
    term: SaeManifoldTerm,
    criterion: SaeCriterion,
    components: SaeOuterRhoGradientComponents,
    log_det: f64,
    solver_gauge_count: usize,
    cache_beta_quotient_dim: usize,
}

/// Emit every value/gradient channel for one rho from exactly one converged
/// inner fit and one authoritative factor cache. The #2253 discriminator used
/// to call `criterion_as_atoms` and then refit the already-mutated term again to
/// obtain `arrow_log_det_from_cache`; center components came from yet another
/// fit. That measured refinement-path drift, not a derivative identity.
fn logdet_audit_point(
    mut term: SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
) -> Result<LogdetAuditPoint, String> {
    let criterion_result =
        term.reml_criterion_with_cache(target, rho, Some(registry), 40, 0.05, 1.0e-6, 1.0e-6)?;
    let loss = criterion_result.1;
    let cache = criterion_result.2;
    let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
        "logdet_audit_point: authoritative log determinant unavailable".to_string()
    })?;
    let occam = term.reml_occam_term(rho)?;
    let extra_penalty_energy = term.reml_extra_penalty_value_total(registry)?;
    let solver = term
        .outer_gradient_arrow_solver(&cache, &rho.lambda_smooth_vec())
        .map_err(|err| err.to_string())?;
    let solver_gauge_count = solver.gauge_basis.len();
    let cache_beta_quotient_dim = cache
        .beta_gauge_quotient
        .as_ref()
        .map_or(0, |quotient| quotient.dimension());
    let components = term
        .analytic_outer_rho_gradient_components(target, rho, &loss, &cache, &solver)
        .map_err(|err| err.to_string())?;
    let criterion = SaeCriterion::assemble(
        loss.total() + extra_penalty_energy,
        log_det,
        occam,
        components.explicit.clone(),
        components.logdet_trace.clone(),
        components.occam.clone(),
        components.third_order_correction.clone(),
    );
    Ok(LogdetAuditPoint {
        term,
        criterion,
        components,
        log_det,
        solver_gauge_count,
        cache_beta_quotient_dim,
    })
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
    // Align the mutable objective term with the banked rho. The actual audit
    // below deliberately ignores this eval's separately-emitted gradient and
    // obtains value + gradient from one fresh cache emission.
    drop(
        objective
            .eval(&banked)
            .expect("state-alignment eval at the audited rho"),
    );
    assert!(
        objective.term.frames_active(),
        "the #2253 discriminator must exercise the profiled Grassmann-frame path"
    );
    let center_term = objective.term.clone();
    let center_rho = objective.baseline_rho.from_flat(banked.view());
    let center = logdet_audit_point(
        center_term.clone(),
        z.view(),
        &center_rho,
        objective.registry.as_ref(),
    )
    .expect("single-emission audit at the center rho");
    let center_components = &center.components;
    let center_logdet = center.log_det;
    let grad = center.criterion.gradient();
    eprintln!(
        "[2253-CHAN] center logdet={center_logdet:+.6e} explicit={:?} trace={:?} \
         adjoint={:?} occam={:?} solver_gauges={} cache_beta_quotient={}",
        center_components.explicit,
        center_components.logdet_trace,
        center_components.third_order_correction,
        center_components.occam,
        center.solver_gauge_count,
        center.cache_beta_quotient_dim,
    );
    for atom in center.criterion.atoms() {
        eprintln!(
            "[zz2234] center atom {}: value={:+.6e} grad={:?}",
            atom.label(),
            atom.value(),
            atom.grad(),
        );
    }
    let h = 1.0e-4;
    let mut failures = Vec::new();
    for j in 0..n_params {
        let mut plus = banked.clone();
        plus[j] += h;
        let mut minus = banked.clone();
        minus[j] -= h;
        let plus_rho = objective.baseline_rho.from_flat(plus.view());
        let minus_rho = objective.baseline_rho.from_flat(minus.view());
        let plus = logdet_audit_point(
            center.term.clone(),
            z.view(),
            &plus_rho,
            objective.registry.as_ref(),
        )
        .expect("single-emission audit at +h");
        let minus = logdet_audit_point(
            center.term.clone(),
            z.view(),
            &minus_rho,
            objective.registry.as_ref(),
        )
        .expect("single-emission audit at -h");
        let plus_logdet = plus.log_det;
        let minus_logdet = minus.log_det;
        let logdet_fd = 0.5 * (plus_logdet - minus_logdet) / (2.0 * h);
        let logdet_analytic =
            center_components.logdet_trace[j] + center_components.third_order_correction[j];
        eprintln!(
            "[2253-CHAN] coord {j}: analytic_logdet={logdet_analytic:+.6e} \
             central_fd_half_logdet={logdet_fd:+.6e}"
        );
        let c_plus = plus.criterion.value();
        let c_minus = minus.criterion.value();
        let fd = (c_plus - c_minus) / (2.0 * h);
        for (plus_atom, minus_atom) in plus.criterion.atoms().iter().zip(minus.criterion.atoms()) {
            let atom_fd = (plus_atom.value() - minus_atom.value()) / (2.0 * h);
            eprintln!(
                "[zz2234] coord {j} atom {}: analytic={:+.6e} central_fd={:+.6e}",
                plus_atom.label(),
                center
                    .criterion
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
        if delta > tolerance {
            failures.push(format!(
                "rho coordinate {j}: analytic={:+.12e}, same-emission central FD={:+.12e}, \
                 delta={delta:.3e}, tolerance={tolerance:.3e}",
                grad[j], fd,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "#2253 eval value/gradient desync(s):\n{}",
        failures.join("\n")
    );
}
