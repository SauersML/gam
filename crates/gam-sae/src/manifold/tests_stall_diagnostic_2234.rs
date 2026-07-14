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
    raw_cache_components: Result<SaeOuterRhoGradientComponents, String>,
    log_det: f64,
    kkt_grad_norm: f64,
    quotient_kkt_grad_norm: f64,
    kkt_tolerance: f64,
    branch_certificate: BranchCertificate,
    exact_chart_gauge_count: usize,
    solver_gauge_count: usize,
    cache_beta_quotient_dim: usize,
    loss_smoothness: f64,
    raw_smoothness_sum: f64,
    smooth_renorm: f64,
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
    registry: Option<&AnalyticPenaltyRegistry>,
    inner_max_iter: usize,
) -> Result<LogdetAuditPoint, String> {
    let (criterion_value, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target,
            rho,
            registry,
            inner_max_iter,
            0.05,
            1.0e-6,
            1.0e-6,
        )
        .map_err(|error| error.to_string())?;
    let raw_smoothness_sum: f64 = term
        .decoder_smoothness_value_per_atom(&rho.lambda_smooth_vec().unwrap())
        .expect("smoothness evaluation must preserve CUDA failures")
        .iter()
        .sum();
    let smooth_renorm = if raw_smoothness_sum.abs() > 0.0 {
        loss.smoothness / raw_smoothness_sum
    } else {
        1.0
    };
    let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
        "logdet_audit_point: authoritative log determinant unavailable".to_string()
    })?;
    let residual = term.reconstruction_residual(target, rho)?;
    let dispersion = term.reconstruction_dispersion(&loss, &cache, rho, Some(residual.view()))?;
    let d_eff = term.per_atom_realised_rank_dof(rho, dispersion)?;
    let n_eff = term.per_atom_effective_sample_size();
    let log_det_tt = super::construction::coordinate_block_log_det(&cache)?;
    let quasi_laplace_complexity = super::construction::rank_adjusted_quasi_laplace_complexity(
        log_det, log_det_tt, &d_eff, &n_eff,
    )
    .map_err(|error| error.to_string())?;
    let occam = term.reml_occam_term(rho)?;
    let extra_penalty_energy = term
        .reml_extra_penalty_value_total(registry)
        .map_err(|error| error.to_string())?;
    let solver = term
        .outer_gradient_arrow_solver(&cache, &rho.lambda_smooth_vec().unwrap())
        .map_err(|err| err.to_string())?;
    let exact_chart_gauge_count = term.dense_step_gauge_vectors()?.len();
    let solver_gauge_count = solver.gauge_basis.len();
    let cache_beta_quotient_dim = cache
        .beta_gauge_quotient
        .as_ref()
        .map_or(0, |quotient| quotient.dimension());
    let components = term
        .analytic_outer_rho_gradient_components(target, rho, &loss, &cache, &solver)
        .map_err(|err| err.to_string())?;
    let raw_cache_solver = DeflatedArrowSolver::plain(&cache);
    let raw_cache_components = term
        .analytic_outer_rho_gradient_components(target, rho, &loss, &cache, &raw_cache_solver)
        .map_err(|err| err.to_string());
    let criterion = SaeCriterion::assemble(
        loss.total() + extra_penalty_energy,
        quasi_laplace_complexity,
        occam,
        components.explicit.clone(),
        components.logdet_trace.clone(),
        components.occam.clone(),
        components.third_order_correction.clone(),
    );
    let criterion_roundoff =
        64.0 * f64::EPSILON * (1.0 + criterion_value.abs().max(criterion.value().abs()));
    if (criterion.value() - criterion_value).abs() > criterion_roundoff {
        return Err(format!(
            "logdet_audit_point: atomized criterion {} != authoritative value {} \
             (roundoff {})",
            criterion.value(),
            criterion_value,
            criterion_roundoff,
        ));
    }
    let mut kkt_term = term.clone();
    let kkt_system = kkt_term.assemble_arrow_schur(target, rho, registry)?;
    let kkt_grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&kkt_system);
    let kkt_grad_norm = kkt_grad_norm_sq.sqrt();
    let quotient_kkt_grad_norm = kkt_term.quotient_gradient_norm_from_system(
        &kkt_system,
        kkt_grad_norm_sq,
        &rho.lambda_smooth_vec().unwrap(),
    );
    let kkt_tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * kkt_term.inner_iterate_scale();
    if !SaeManifoldTerm::quasi_laplace_kkt_stationary(
        kkt_grad_norm,
        quotient_kkt_grad_norm,
        kkt_tolerance,
    ) {
        return Err(format!(
            "logdet_audit_point: an off-KKT state cannot emit/certify an analytic envelope \
             gradient (raw={kkt_grad_norm:.6e}, quotient={quotient_kkt_grad_norm:.6e}, \
             tolerance={kkt_tolerance:.6e})"
        ));
    }
    let branch_certificate =
        BranchCertificate::from_arrow_cache(&cache, MajorizerAnchorMode::FrozenAnchor);
    // #2253 mechanism gate: a model whose evidence cache was returned must be
    // an exact recurrence of the evidence-only inner map. Re-enter from the
    // returned state with the same bounded chunk; any accepted strict move
    // proves the cache was formed at the coarse KKT admission band rather than
    // at the differentiable no-descent root.
    let mut recurrence_term = term.clone();
    let recurrence_entry_frames: Vec<_> = recurrence_term
        .atoms
        .iter()
        .map(|atom| atom.decoder_frame.clone())
        .collect();
    let mut recurrence_rho = rho.clone();
    let recurrence = recurrence_term.run_joint_fit_arrow_schur_for_quasi_laplace(
        target,
        &mut recurrence_rho,
        registry,
        inner_max_iter,
        0.05,
        1.0e-6,
        1.0e-6,
    )?;
    let decoder_frames_recurred =
        decoder_frames_match_exactly(&recurrence_term, &recurrence_entry_frames);
    if !recurrence.fixed_point || !decoder_frames_recurred {
        return Err(format!(
            "logdet_audit_point: evidence cache returned at a non-idempotent inner state \
             (budget={inner_max_iter}, KKT={kkt_grad_norm:.6e}, \
             quotient KKT={quotient_kkt_grad_norm:.6e}, tolerance={kkt_tolerance:.6e}, \
             reported_fixed_point={}, decoder_frames_recurred={decoder_frames_recurred})",
            recurrence.fixed_point,
        ));
    }
    Ok(LogdetAuditPoint {
        term,
        criterion,
        components,
        raw_cache_components,
        log_det,
        kkt_grad_norm,
        quotient_kkt_grad_norm,
        kkt_tolerance,
        branch_certificate,
        exact_chart_gauge_count,
        solver_gauge_count,
        cache_beta_quotient_dim,
        loss_smoothness: loss.smoothness,
        raw_smoothness_sum,
        smooth_renorm,
    })
}

fn frozen_raw_logdet(
    mut term: SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: Option<&AnalyticPenaltyRegistry>,
) -> Result<f64, String> {
    let criterion_result = term
        .penalized_quasi_laplace_criterion_with_cache(
            target, rho, registry, 0, 0.05, 1.0e-6, 1.0e-6,
        )
        .map_err(|error| error.to_string())?;
    arrow_log_det_from_cache(&criterion_result.2)
        .ok_or_else(|| "frozen_raw_logdet: authoritative log determinant unavailable".to_string())
}

fn decoder_frames_match_exactly(
    current: &SaeManifoldTerm,
    expected: &[Option<GrassmannFrame>],
) -> bool {
    current.atoms.len() == expected.len()
        && current.atoms.iter().zip(expected).all(|(atom, saved)| {
            match (&atom.decoder_frame, saved) {
                (Some(current), Some(expected)) => {
                    current.frame() == expected.frame()
                        && current.gauge_singular_values() == expected.gauge_singular_values()
                }
                (None, None) => true,
                _ => false,
            }
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
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
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
            // The 2026-07-10 infeasibility pathology is the REGRESSION this gate
            // pins: provisional budget-marker refusals froze every fit (frozen
            // isotropic checkpoint, large lane disagreement).
            // Post-fix the optimizer descends genuinely; certification on this
            // deliberately tight fixture budget remains an honest optimizer
            // limitation, not an infeasible probe. Assert that pathology stays dead.
            assert_eq!(
                telemetry.infeasible_criterion_evals, 0,
                "infeasible probes returned — the #2234 pathology regressed"
            );
            assert!(
                telemetry.criterion_calls > 10,
                "the outer search froze after {} criterion calls — infeasibility or an \
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
        .expect("checkpoint rho must satisfy the objective domain")
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
    let center_rho = objective.baseline_rho.from_flat(banked.view()).unwrap();
    let h = 1.0e-4;
    let mut failures = Vec::new();
    for inner_max_iter in [40usize, 200usize] {
        let center = logdet_audit_point(
            center_term.clone(),
            z.view(),
            &center_rho,
            objective.registry.as_ref(),
            inner_max_iter,
        )
        .expect("single-emission audit at the center rho");
        let center_components = &center.components;
        let center_logdet = center.log_det;
        let grad = center.criterion.gradient();
        eprintln!(
            "[2253-CHAN] budget={inner_max_iter} center logdet={center_logdet:+.6e} \
             explicit={:?} trace={:?} adjoint={:?} occam={:?} \
             kkt={:+.6e} quotient_kkt={:+.6e} kkt_tol={:+.6e} \
             exact_chart_gauges={} solver_gauges={} cache_beta_quotient={}",
            center_components.explicit,
            center_components.logdet_trace,
            center_components.third_order_correction,
            center_components.occam,
            center.kkt_grad_norm,
            center.quotient_kkt_grad_norm,
            center.kkt_tolerance,
            center.exact_chart_gauge_count,
            center.solver_gauge_count,
            center.cache_beta_quotient_dim,
        );
        eprintln!(
            "[2253-CERT] budget={inner_max_iter} {:?}",
            center.branch_certificate
        );
        eprintln!(
            "[2253-SMOOTH] budget={inner_max_iter} loss_smoothness={:+.17e} \
             raw_smoothness_sum={:+.17e} renorm={:+.17e}",
            center.loss_smoothness, center.raw_smoothness_sum, center.smooth_renorm,
        );
        let smooth_roundoff = 64.0
            * f64::EPSILON
            * (1.0
                + center
                    .loss_smoothness
                    .abs()
                    .max(center.raw_smoothness_sum.abs()));
        assert!(
            (center.loss_smoothness - center.raw_smoothness_sum).abs() <= smooth_roundoff,
            "#2253 dense full-batch smoothness scale diverged between the returned loss and \
             current term: loss={:+.17e}, raw_sum={:+.17e}, renorm={:+.17e}, \
             roundoff={smooth_roundoff:.3e}",
            center.loss_smoothness,
            center.raw_smoothness_sum,
            center.smooth_renorm,
        );
        for atom in center.criterion.atoms() {
            eprintln!(
                "[zz2234] budget={inner_max_iter} center atom {}: value={:+.6e} grad={:?}",
                atom.label(),
                atom.value(),
                atom.grad(),
            );
        }
        for j in 0..n_params {
            let mut plus = banked.clone();
            plus[j] += h;
            let mut minus = banked.clone();
            minus[j] -= h;
            let plus_rho = objective.baseline_rho.from_flat(plus.view()).unwrap();
            let minus_rho = objective.baseline_rho.from_flat(minus.view()).unwrap();
            let plus = logdet_audit_point(
                center.term.clone(),
                z.view(),
                &plus_rho,
                objective.registry.as_ref(),
                inner_max_iter,
            )
            .expect("single-emission audit at +h");
            let minus = logdet_audit_point(
                center.term.clone(),
                z.view(),
                &minus_rho,
                objective.registry.as_ref(),
                inner_max_iter,
            )
            .expect("single-emission audit at -h");
            let plus_logdet = plus.log_det;
            let minus_logdet = minus.log_det;
            let logdet_fd = 0.5 * (plus_logdet - minus_logdet) / (2.0 * h);
            let logdet_analytic =
                center_components.logdet_trace[j] + center_components.third_order_correction[j];
            let raw_cache_analytic = center.raw_cache_components.as_ref().map(|components| {
                components.logdet_trace[j] + components.third_order_correction[j]
            });
            eprintln!(
                "[2253-CHAN] budget={inner_max_iter} coord {j}: \
                 analytic_logdet={logdet_analytic:+.6e} \
                 raw_cache_analytic={raw_cache_analytic:?} \
                 central_fd_half_logdet={logdet_fd:+.6e}"
            );
            let frozen_plus = frozen_raw_logdet(
                center.term.clone(),
                z.view(),
                &plus_rho,
                objective.registry.as_ref(),
            )
            .expect("frozen raw logdet at +h");
            let frozen_minus = frozen_raw_logdet(
                center.term.clone(),
                z.view(),
                &minus_rho,
                objective.registry.as_ref(),
            )
            .expect("frozen raw logdet at -h");
            let frozen_raw_logdet_fd = 0.5 * (frozen_plus - frozen_minus) / (2.0 * h);
            let raw_cache_direct = center
                .raw_cache_components
                .as_ref()
                .map(|components| components.logdet_trace[j]);
            eprintln!(
                "[2253-FIXED] budget={inner_max_iter} coord {j}: \
                 deflated_direct={:+.6e} raw_cache_direct={raw_cache_direct:?} \
                 frozen_raw_half_logdet_fd={frozen_raw_logdet_fd:+.6e}",
                center_components.logdet_trace[j],
            );
            let c_plus = plus.criterion.value();
            let c_minus = minus.criterion.value();
            let fd = (c_plus - c_minus) / (2.0 * h);
            for (plus_atom, minus_atom) in
                plus.criterion.atoms().iter().zip(minus.criterion.atoms())
            {
                let atom_fd = (plus_atom.value() - minus_atom.value()) / (2.0 * h);
                eprintln!(
                    "[zz2234] budget={inner_max_iter} coord {j} atom {}: \
                     analytic={:+.6e} central_fd={:+.6e}",
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
                "[zz2234] budget={inner_max_iter} coord {j}: analytic={:+.6e} \
                 eval_fd={:+.6e} |delta|={delta:.3e} tol={tolerance:.3e}",
                grad[j], fd,
            );
            if delta > tolerance {
                failures.push(format!(
                    "budget {inner_max_iter}, rho coordinate {j}: analytic={:+.12e}, \
                     same-emission central FD={:+.12e}, delta={delta:.3e}, \
                     tolerance={tolerance:.3e}",
                    grad[j], fd,
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "#2253 eval value/gradient desync(s):\n{}",
        failures.join("\n")
    );
}
