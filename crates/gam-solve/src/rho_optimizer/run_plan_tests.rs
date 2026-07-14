use super::*;
use ::opt::FixedPointObjective;
use ndarray::array;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ─── #934 first-order optimality certificate ──────────────────────

/// Quadratic ½‖ρ − c‖² with value and gradient from the SAME center:
/// the certificate must attest consistency at the optimum.
#[test]
fn certificate_attests_consistent_quadratic() {
    let center = array![0.3, -0.7];
    let cost_center = center.clone();
    let grad_center = center.clone();
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![2.0, 2.0])
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            let d = rho - &cost_center;
            Ok(0.5 * d.dot(&d))
        },
        move |_: &mut (), rho: &Array1<f64>| {
            let d = rho - &grad_center;
            Ok(OuterEval {
                cost: 0.5 * d.dot(&d),
                gradient: d,
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "certificate consistent quadratic")
        .expect("consistent quadratic must optimize");
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("gradient-based solve must ship a certificate");
    assert!(
        cert.certifies(),
        "consistent quadratic optimum failed certification: {}",
        cert.summary(),
    );
    assert!(
        cert.lambdas_railed.is_empty(),
        "interior optimum reported railed λ: {}",
        cert.summary(),
    );
    assert!(cert.stationarity.bound() > 0.0 && cert.stationarity.projected_norm().is_finite());
}

/// #979: a solver may finish before a family's nominal sampled-derivative
/// budget. The sampled optimum is useful as a checkpoint, but the runner must
/// then optimize the exact objective rather than merely re-evaluating and
/// rejecting that checkpoint during certification.
#[test]
fn sampled_outer_pilot_is_followed_by_exact_polish_before_certification_979() {
    #[derive(Default)]
    struct PilotState {
        exact: bool,
        pilot_derivative_evals: usize,
        exact_derivative_evals: usize,
        transitions: usize,
    }

    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0])
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem
        .build_objective(
            PilotState::default(),
            |state: &mut PilotState, rho: &Array1<f64>| {
                let center = if state.exact { 1.0 } else { 2.0 };
                let delta = rho[0] - center;
                Ok(0.5 * delta * delta)
            },
            |state: &mut PilotState, rho: &Array1<f64>| {
                if state.exact {
                    state.exact_derivative_evals += 1;
                } else {
                    state.pilot_derivative_evals += 1;
                }
                let center = if state.exact { 1.0 } else { 2.0 };
                let delta = rho[0] - center;
                Ok(OuterEval {
                    cost: 0.5 * delta * delta,
                    gradient: array![delta],
                    hessian: HessianValue::Dense(array![[1.0]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut PilotState)>,
            None::<fn(&mut PilotState, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
        .with_exact_polish(|state: &mut PilotState| {
            if state.exact || state.pilot_derivative_evals == 0 {
                return false;
            }
            state.exact = true;
            state.transitions += 1;
            true
        });

    let result = problem
        .run(&mut obj, "sampled-pilot exact-polish regression #979")
        .expect("the exact polish must converge and certify");
    assert!(
        (result.rho[0] - 1.0).abs() < 1.0e-7,
        "returned sampled optimum instead of exact optimum: rho={:?}",
        result.rho,
    );
    assert_eq!(
        obj.state.transitions, 1,
        "exact transition must be single-shot"
    );
    assert!(obj.state.pilot_derivative_evals > 0);
    assert!(obj.state.exact_derivative_evals > 0);
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(OuterCriterionCertificate::certifies),
        "exact-polished result must carry the mandatory certificate",
    );
}

#[test]
fn rho_uncertainty_diagnostic_does_not_change_outer_solution() {
    let center = array![0.25];
    let seed_config = gam_problem::SeedConfig {
        max_seeds: 1,
        seed_budget: 1,
        ..Default::default()
    };
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_initial_rho(array![1.5])
        .with_seed_config(seed_config)
        .with_problem_size(8, 3);
    let config = problem.config();

    let mut without_diagnostic = problem.build_objective(
        (),
        {
            let center = center.clone();
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &center;
                Ok(0.5 * d.dot(&d))
            }
        },
        {
            let center = center.clone();
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &center;
                Ok(OuterEval {
                    cost: 0.5 * d.dot(&d),
                    gradient: d,
                    hessian: HessianValue::Dense(array![[1.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut with_diagnostic = problem.build_objective(
        (),
        {
            let center = center.clone();
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &center;
                Ok(0.5 * d.dot(&d))
            }
        },
        {
            let center = center.clone();
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &center;
                Ok(OuterEval {
                    cost: 0.5 * d.dot(&d),
                    gradient: d,
                    hessian: HessianValue::Dense(array![[1.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );

    let baseline =
        run_outer_uncertified(&mut without_diagnostic, &config, "rho-diagnostic-baseline")
            .expect("baseline outer run");
    let diagnosed = run_outer(&mut with_diagnostic, &config, "rho-diagnostic-run")
        .expect("diagnostic outer run");

    assert_eq!(baseline.rho, diagnosed.rho);
    assert_eq!(
        baseline.final_value.to_bits(),
        diagnosed.final_value.to_bits()
    );
    assert_eq!(baseline.iterations, diagnosed.iterations);
    assert_eq!(baseline.final_grad_norm, diagnosed.final_grad_norm);
    assert!(diagnosed.rho_uncertainty_diagnostic.is_some());
}

/// The desync bug genus (#748/#752/#901): the gradient path optimizes a
/// criterion whose center is silently shifted from the value path's.
/// The optimizer happily converges where the WRONG gradient vanishes; the
/// analytic certificate must reject the two objective values measured at the
/// identical returned point when they disagree beyond roundoff.
#[test]
fn certificate_flags_value_gradient_desync() {
    let value_center = array![0.0, 0.0];
    let wrong_center = array![3.0, -2.0];
    let wrong_center_for_eval = wrong_center.clone();
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![1.0, 1.0])
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    // eval(): a self-consistent but WRONG world (shifted center) so the
    // line search accepts steps and BFGS converges to wrong_center.
    // eval_cost(): the TRUE criterion value — the path the audit probes.
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            let d = rho - &value_center;
            Ok(0.5 * d.dot(&d))
        },
        move |_: &mut (), rho: &Array1<f64>| {
            let d = rho - &wrong_center_for_eval;
            Ok(OuterEval {
                cost: 0.5 * d.dot(&d),
                gradient: d,
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let error = problem
        .run(&mut obj, "certificate desynced quadratic")
        .expect_err("desynchronised value/gradient paths must not mint a result");
    assert!(
        matches!(error, EstimationError::RemlDidNotConverge { .. }),
        "desynchronisation must be typed outer non-convergence, got {error}"
    );
}

#[test]
fn certificate_hessian_psd_probe_classifies_definiteness() {
    assert_eq!(
        certificate_hessian_is_psd(&Array2::<f64>::eye(3)),
        Some(true)
    );
    let indefinite = array![[1.0, 2.0], [2.0, 1.0]];
    assert_eq!(certificate_hessian_is_psd(&indefinite), Some(false));
    assert_eq!(
        certificate_hessian_is_psd(&Array2::<f64>::zeros((0, 0))),
        None
    );
    let non_finite = array![[f64::NAN]];
    assert_eq!(certificate_hessian_is_psd(&non_finite), None);
}

#[test]
fn newton_predicted_decrease_is_curvature_scaled() {
    // Diagonal PD Hessian: predicted decrease = ½ Σ gᵢ²/Hᵢᵢ (Newton decrement/2).
    let hessian = array![[4.0, 0.0], [0.0, 100.0]];
    let grad = array![2.0, 20.0];
    let got = newton_predicted_decrease(&hessian, &grad).expect("PD Hessian decrement");
    // ½·(2²/4 + 20²/100) = ½·(1 + 4) = 2.5 (shift ~√ε·100 ≈ 1.5e-6 is negligible).
    assert!((got - 2.5).abs() < 1.0e-4, "got {got}");

    // A residual concentrated on a STIFF (high-curvature) direction maps to a
    // NEGLIGIBLE predicted decrease — the curvature-scaled certificate certifies
    // it even though its raw magnitude is not tiny. This is the #2253 flat-valley
    // case: |g| above the crude score-relative band, ½gᵀH⁻¹g below tolerance.
    let stiff_h = array![[1.0e6, 0.0], [0.0, 1.0e6]];
    let stiff_g = array![0.0717, 0.0];
    let stiff = newton_predicted_decrease(&stiff_h, &stiff_g).expect("stiff decrement");
    assert!(
        stiff < 1.0e-8,
        "stiff residual should predict tiny decrease, got {stiff}"
    );

    // A residual along a NEAR-FLAT direction (a linear ramp with real descent)
    // inflates gᵀH⁻¹g and is NOT certified as negligible — the routine must never
    // waive a genuine descent direction.
    let flat_h = array![[1.0e-9, 0.0], [0.0, 1.0]];
    let flat_g = array![0.05, 0.0];
    let flat = newton_predicted_decrease(&flat_h, &flat_g).expect("flat decrement");
    assert!(
        flat > 1.0,
        "flat-direction residual should predict large decrease, got {flat}"
    );

    // Malformed / mismatched shapes yield `None`.
    assert!(newton_predicted_decrease(&array![[1.0, 0.0], [0.0, 1.0]], &array![1.0]).is_none());
    assert!(newton_predicted_decrease(&array![[f64::NAN]], &array![1.0]).is_none());
}

#[test]
fn certificate_rail_detection_uses_outer_box() {
    let config = OuterConfig::default(); // rho_bound = 30
    let rho = array![29.8, 0.0, -29.6];
    assert_eq!(certificate_railed_lambdas(&rho, 3, &config), vec![0, 2]);
    // Only the leading rho_dim coordinates are λ axes.
    assert_eq!(certificate_railed_lambdas(&rho, 1, &config), vec![0]);
    let bounded = OuterConfig {
        bounds: Some((array![-5.0, -5.0, -5.0], array![5.0, 5.0, 5.0])),
        ..OuterConfig::default()
    };
    let pinned = array![4.9, -4.7, 0.0];
    assert_eq!(certificate_railed_lambdas(&pinned, 3, &bounded), vec![0, 1]);
}

/// Helper: build an objective `0.5·ρ₀² + slope·ρ₁` (analytic gradient
/// `[ρ₀, slope]`) and run the post-FD-purge certificate audit
/// (`certify_outer_optimality`, which re-evaluates the analytic gradient at
/// the point and returns `Err` on non-certification) at a constructed point
/// where ρ₁ sits on the upper box bound.
fn audit_at_railed_optimum(
    config: &OuterConfig,
    theta_hat: Array1<f64>,
    value_slope_railed: f64,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let slope = value_slope_railed;
    let mut obj = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                // True criterion VALUE: free coord 0 is a consistent
                // quadratic; railed coord 1 has slope `slope`.
                Ok(0.5 * rho[0] * rho[0] + slope * rho[1])
            },
            move |_: &mut (), rho: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * rho[0] * rho[0] + slope * rho[1],
                    gradient: array![rho[0], slope],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let mut result = OuterResult::new(
        theta_hat,
        0.0,
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    certify_outer_optimality(&mut obj, config, "railed-audit-unit", &mut result)
}

/// KKT projection: at an optimum whose ONLY nonzero gradient component pulls
/// INTO an active box bound, the bound multiplier balances it, so the
/// projected gradient drops it and the certificate certifies with the
/// coordinate reported railed — a legitimately bound-pinned optimum is not a
/// false alarm.
#[test]
fn certificate_certifies_kkt_stationary_railed_optimum() {
    let bounded = OuterConfig {
        bounds: Some((array![-5.0, -5.0], array![5.0, 5.0])),
        ..OuterConfig::default()
    };
    // ρ₁ railed at the upper bound (5.0); ρ₀ interior at its quadratic min.
    // The objective slope on ρ₁ is −7: descent pushes ρ₁ UP into the active
    // upper bound, so the component is KKT-balanced and must be projected out.
    let cert = audit_at_railed_optimum(&bounded, array![0.0, 5.0], -7.0)
        .expect("KKT-stationary railed optimum must certify");
    assert_eq!(
        cert.lambdas_railed,
        vec![1],
        "coord 1 must be detected as railed: {}",
        cert.summary(),
    );
    assert!(
        cert.certifies(),
        "bound-pinned KKT-stationary optimum was rejected: {}",
        cert.summary(),
    );
    assert!(
        cert.stationarity.projected_norm() <= cert.stationarity.bound(),
        "projected gradient must drop the railed component: {}",
        cert.summary(),
    );
    assert!(
        cert.stationarity.raw_norm() > cert.stationarity.bound(),
        "raw gradient norm must still see the railed slope (the projection, \
         not the raw norm, carries the KKT verdict): {}",
        cert.summary(),
    );
}

/// Build an interior 1-coordinate objective `½·ρ₀²` (analytic gradient `[ρ₀]`,
/// analytic Dense Hessian `[[1]]`) and certify at `theta_hat` with NO
/// `operator_stop_reason` set — i.e. the non-flat-valley exit path a fit takes
/// when it is already stationary at iteration 0. `objective_scale = 80` makes
/// the arithmetic gradient floor `80·√ε`, mirroring the Gaussian-linear
/// standard-REML fit's matrix-factorization resolution.
fn audit_interior_with_dense_curvature(
    theta_hat: Array1<f64>,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let config = OuterConfig {
        tolerance: 1.0e-12,
        objective_scale: Some(80.0),
        ..OuterConfig::default()
    };
    let mut obj = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(0.5 * rho[0] * rho[0]),
            move |_: &mut (), rho: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * rho[0] * rho[0],
                    gradient: array![rho[0]],
                    hessian: HessianValue::Dense(array![[1.0]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let mut result = OuterResult::new(
        theta_hat,
        0.0,
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    // `operator_stop_reason` is left None: this exercises the NON-flat-valley
    // exit, proving the curvature-scaled widening is not gated to a specific
    // stop reason (#2091/#2011).
    certify_outer_optimality(&mut obj, &config, "curvature-widen-unit", &mut result)
}

/// The curvature-scaled widening is NOT gated to a `CostStallFlatValley` exit:
/// a fit already stationary at iteration 0 (a 2-parameter Gaussian-linear REML
/// with λ→0) reaches certification with `operator_stop_reason = None`, a
/// projected gradient above the arithmetic score·√ε floor, and a
/// NEGLIGIBLE Newton decrement. The Newton decrement — not the exit reason — is
/// the stationarity certificate, so the point must certify.
#[test]
fn curvature_widening_certifies_stationary_point_on_any_exit_reason() {
    let arithmetic_floor = 80.0 * f64::EPSILON.sqrt();
    // |Pg| = 2e-6 > 80·√ε, but ½·gᵀH⁻¹g = ½·(2e-6)² = 2e-12,
    // orders of magnitude below any outer objective tolerance: stationary to
    // second order, must certify DESPITE operator_stop_reason = None.
    let cert = audit_interior_with_dense_curvature(array![2.0e-6])
        .expect("second-order-stationary point must certify via the curvature bound");
    assert!(
        cert.certifies(),
        "curvature-stationary interior point was rejected: {}",
        cert.summary(),
    );
    assert!(
        cert.stationarity.projected_norm() > arithmetic_floor,
        "the test must exercise the ABOVE-solver-bound regime (else it proves \
         nothing about the widening): {}",
        cert.summary(),
    );
}

/// The widening is direction/curvature-aware, not a blanket loosening: a
/// genuinely non-stationary interior point (large Newton decrement) must still
/// be rejected even though the same broadened gate is reached.
#[test]
fn curvature_widening_still_rejects_genuine_nonstationarity() {
    // |Pg| = 1.0 with a unit Hessian → ½·gᵀH⁻¹g = 0.5 ≫ objective_tol, so the
    // curvature bound ‖Pg‖·√(tol/Δpred) stays FAR below ‖Pg‖ and cannot rescue
    // the point.
    let outcome = audit_interior_with_dense_curvature(array![1.0]);
    assert!(
        outcome.is_err(),
        "a genuinely non-stationary point must not be certified by the curvature bound",
    );
}

fn terminal_fidelity_feedback(cap_value: usize) -> InnerProgressFeedback {
    InnerProgressFeedback {
        cap: Arc::new(AtomicUsize::new(cap_value)),
        accepted_iter: Arc::new(AtomicUsize::new(0)),
        last_iters: Arc::new(AtomicUsize::new(cap_value)),
        last_converged: Arc::new(AtomicBool::new(false)),
        ift_residual: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
        accept_rho: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
    }
}

/// Standard REML regression for #2309.  The search cache contains a
/// cap-produced sample whose moderate gradient is paired with artificial stiff
/// curvature; the old certificate reused it and widened its bound enough to
/// certify.  Terminal certification must clear that cache, evaluate at cap=0,
/// and reject the genuinely non-stationary full-fidelity gradient.
#[test]
fn standard_reml_certificate_uses_fresh_uncapped_inner_state_2309() {
    struct StandardState {
        feedback: InnerProgressFeedback,
        coarse_cache_present: bool,
        reset_count: usize,
        evaluated_caps: Vec<usize>,
    }

    let feedback = terminal_fidelity_feedback(3);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_tolerance(1.0e-6)
        .with_outer_inner_cap(feedback.clone());
    let config = problem.config();
    let state = StandardState {
        feedback,
        coarse_cache_present: true,
        reset_count: 0,
        evaluated_caps: Vec::new(),
    };
    let mut obj = problem.build_objective(
        state,
        |_: &mut StandardState, _: &Array1<f64>| Ok(10.0),
        |state: &mut StandardState, _: &Array1<f64>| {
            let cap = state.feedback.cap.load(Ordering::Relaxed);
            state.evaluated_caps.push(cap);
            if state.coarse_cache_present {
                return Ok(OuterEval {
                    cost: 10.0,
                    gradient: array![0.5],
                    hessian: HessianValue::Dense(array![[1.25e8]]),
                    inner_beta_hint: None,
                });
            }
            state.feedback.last_iters.store(12, Ordering::Relaxed);
            state
                .feedback
                .last_converged
                .store(cap == 0, Ordering::Relaxed);
            Ok(OuterEval {
                cost: 10.0,
                gradient: array![37.0],
                hessian: HessianValue::Dense(array![[1.0]]),
                inner_beta_hint: None,
            })
        },
        Some(|state: &mut StandardState| {
            state.reset_count += 1;
            state.coarse_cache_present = false;
            state.feedback.last_iters.store(0, Ordering::Relaxed);
            state
                .feedback
                .last_converged
                .store(false, Ordering::Relaxed);
        }),
        None::<fn(&mut StandardState, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut result = OuterResult::new(
        array![0.0],
        10.0,
        1,
        true,
        OuterPlan {
            solver: Solver::Arc,
            hessian_source: HessianSource::Analytic,
        },
    );

    assert!(
        certify_outer_optimality(&mut obj, &config, "standard REML #2309", &mut result).is_err(),
        "the full-fidelity gradient has real descent and must not inherit the coarse widened bound",
    );
    assert_eq!(obj.state.reset_count, 1);
    assert_eq!(obj.state.evaluated_caps, vec![0]);
    assert_eq!(obj.state.feedback.cap.load(Ordering::Relaxed), 3);
    assert_eq!(result.final_gradient.as_ref(), Some(&array![37.0]));
}

/// Mixture/SAS regression for the augmented `[rho | link]` layout.  It proves
/// that terminal reset happens before the final link state is evaluated and
/// that a coarse rho-only artifact cannot donate its curvature-scaled bound to
/// the fresh link-coordinate gradient.
#[test]
fn mixture_reml_certificate_recomputes_augmented_theta_at_full_fidelity_2309() {
    struct MixtureState {
        feedback: InnerProgressFeedback,
        coarse_rho_cache_present: bool,
        reset_count: usize,
        evaluated_caps: Vec<usize>,
        last_theta: Option<Array1<f64>>,
    }

    let feedback = terminal_fidelity_feedback(3);
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_psi_dim(1)
        .with_tolerance(1.0e-6)
        .with_outer_inner_cap(feedback.clone());
    let config = problem.config();
    let state = MixtureState {
        feedback,
        coarse_rho_cache_present: true,
        reset_count: 0,
        evaluated_caps: Vec::new(),
        last_theta: None,
    };
    let mut obj = problem.build_objective(
        state,
        |_: &mut MixtureState, _: &Array1<f64>| Ok(10.0),
        |state: &mut MixtureState, theta: &Array1<f64>| {
            let cap = state.feedback.cap.load(Ordering::Relaxed);
            state.evaluated_caps.push(cap);
            state.last_theta = Some(theta.clone());
            if state.coarse_rho_cache_present {
                return Ok(OuterEval {
                    cost: 10.0,
                    gradient: array![0.0, 0.5],
                    hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, 1.25e8]]),
                    inner_beta_hint: None,
                });
            }
            state.feedback.last_iters.store(14, Ordering::Relaxed);
            state
                .feedback
                .last_converged
                .store(cap == 0, Ordering::Relaxed);
            Ok(OuterEval {
                cost: 10.0,
                gradient: array![0.0, 37.0],
                hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, 1.0]]),
                inner_beta_hint: None,
            })
        },
        Some(|state: &mut MixtureState| {
            state.reset_count += 1;
            state.coarse_rho_cache_present = false;
            state.last_theta = None;
            state.feedback.last_iters.store(0, Ordering::Relaxed);
            state
                .feedback
                .last_converged
                .store(false, Ordering::Relaxed);
        }),
        None::<fn(&mut MixtureState, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let theta_hat = array![0.2, -0.8];
    let mut result = OuterResult::new(
        theta_hat.clone(),
        10.0,
        1,
        true,
        OuterPlan {
            solver: Solver::Arc,
            hessian_source: HessianSource::Analytic,
        },
    );

    assert!(
        certify_outer_optimality(&mut obj, &config, "mixture REML #2309", &mut result).is_err(),
        "the full-fidelity link gradient has real descent and must not inherit the coarse widened bound",
    );
    assert_eq!(obj.state.reset_count, 1);
    assert_eq!(obj.state.evaluated_caps, vec![0]);
    assert_eq!(obj.state.last_theta.as_ref(), Some(&theta_hat));
    assert_eq!(obj.state.feedback.cap.load(Ordering::Relaxed), 3);
    assert_eq!(result.final_gradient.as_ref(), Some(&array![0.0, 37.0]));
}

fn audit_gradient_only_roundoff_residual_2269(
    residual: f64,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let objective_scale = 80.0;
    let config = OuterConfig {
        tolerance: 1.0e-12,
        objective_scale: Some(objective_scale),
        ..OuterConfig::default()
    };
    // The value oracle is exactly flat. `residual` represents the forward-error
    // remainder of its analytic matrix-factorization score: this is the
    // gradient-only case, so no Hessian/decrement or trajectory-noise rescue is
    // available to the final certificate.
    let mut obj = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .build_objective(
            (),
            move |_: &mut (), _: &Array1<f64>| Ok(objective_scale),
            move |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: objective_scale,
                    gradient: array![residual],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let mut result = OuterResult::new(
        array![0.0],
        objective_scale,
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    certify_outer_optimality(
        &mut obj,
        &config,
        "gradient-only-roundoff-audit-2269",
        &mut result,
    )
}

#[test]
fn gradient_only_certificate_uses_objective_roundoff_resolution_2269() {
    let scale = 80.0;
    let arithmetic_floor = scale * f64::EPSILON.sqrt();
    let residual = 0.5 * arithmetic_floor;

    let certificate = audit_gradient_only_roundoff_residual_2269(residual)
        .expect("a flat score's sub-roundoff residual must certify without curvature or probes");
    assert!(certificate.certifies());
    assert!(certificate.stationarity.bound() >= arithmetic_floor);
    assert!(certificate.stationarity.projected_norm() <= certificate.stationarity.bound());
}

#[test]
fn gradient_only_certificate_rejects_residual_above_roundoff_2269() {
    let arithmetic_floor = 80.0 * f64::EPSILON.sqrt();
    assert!(audit_gradient_only_roundoff_residual_2269(2.0 * arithmetic_floor).is_err());
}

/// The projection must NOT blunt the certificate's real job: genuine
/// non-stationarity on a FREE (interior) coordinate must still reject the
/// point even when a different coordinate is railed.
#[test]
fn certificate_rejects_genuine_interior_nonstationarity() {
    let bounded = OuterConfig {
        bounds: Some((array![-5.0, -5.0], array![5.0, 5.0])),
        ..OuterConfig::default()
    };
    // ρ₁ railed at the upper bound (KKT-balanced slope −7), but ρ₀ = 2.5 is
    // interior with analytic slope 2.5 — real feasible descent remains, so
    // certification must fail with typed non-convergence.
    let err = audit_at_railed_optimum(&bounded, array![2.5, 5.0], -7.0)
        .expect_err("interior non-stationarity must reject certification");
    let msg = format!("{err}");
    assert!(
        msg.contains("stationary") || msg.contains("converge"),
        "rejection must be a typed stationarity failure, got: {msg}"
    );
}

/// With nothing railed, the projection is the identity: an interior
/// stationary point certifies with an empty railed list.
#[test]
fn certificate_full_space_unchanged_when_nothing_railed() {
    let bounded = OuterConfig {
        bounds: Some((array![-30.0, -30.0], array![30.0, 30.0])),
        ..OuterConfig::default()
    };
    // Interior optimum far from both bounds; slope 0 on ρ₁ and ρ₀ at its
    // quadratic minimum, so the analytic gradient vanishes identically.
    let cert = audit_at_railed_optimum(&bounded, array![0.0, 1.0], 0.0)
        .expect("interior stationary point must certify");
    assert!(
        cert.lambdas_railed.is_empty(),
        "no coordinate is near a bound: {}",
        cert.summary(),
    );
    assert!(
        cert.certifies(),
        "consistent interior optimum flagged: {}",
        cert.summary(),
    );
}

// The two `outer_scaled_tolerance_*` tests that lived here have
// been removed: the helper is gone in favor of opt 0.5.0's
// `GradientTolerance::relative_to_cost(τ)`. Equivalent threshold
// coverage now lives upstream as
// `opt::tests::gradient_tolerance_relative_to_cost_matches_textbook_form`.

struct FailingSeedMaterializationOperator {
    dim: usize,
}

impl HessianOperator for FailingSeedMaterializationOperator {
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), ObjectiveEvalError> {
        out.assign(v);
        Ok(())
    }

    fn materialization(&self) -> HessianMaterialization {
        HessianMaterialization::RepeatedHvp
    }

    fn materialize_dense(&self) -> Result<Array2<f64>, ObjectiveEvalError> {
        Err(ObjectiveEvalError::fatal("seed materialization failed"))
    }
}

#[test]
fn materialize_dense_uses_single_batched_apply_mat() {
    struct BatchedOnlyHessian {
        matrix: Array2<f64>,
        apply_calls: Arc<AtomicUsize>,
        apply_mat_calls: Arc<AtomicUsize>,
        rhs_columns: Arc<AtomicUsize>,
    }

    impl HessianOperator for BatchedOnlyHessian {
        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn apply_into(
            &self,
            v: &Array1<f64>,
            out: &mut Array1<f64>,
        ) -> Result<(), ObjectiveEvalError> {
            self.apply_calls.fetch_add(1, Ordering::Relaxed);
            out.assign(&self.matrix.dot(v));
            Ok(())
        }

        fn apply_mat(
            &self,
            factor: ArrayView2<'_, f64>,
        ) -> Result<Array2<f64>, ObjectiveEvalError> {
            self.apply_mat_calls.fetch_add(1, Ordering::Relaxed);
            self.rhs_columns
                .fetch_add(factor.ncols(), Ordering::Relaxed);
            Ok(self.matrix.dot(&factor))
        }

        fn materialization(&self) -> HessianMaterialization {
            HessianMaterialization::BatchedHvp
        }
    }

    let apply_calls = Arc::new(AtomicUsize::new(0));
    let apply_mat_calls = Arc::new(AtomicUsize::new(0));
    let rhs_columns = Arc::new(AtomicUsize::new(0));
    let op = BatchedOnlyHessian {
        matrix: array![[2.0, 0.25, -0.5], [0.5, 3.0, 1.0], [-0.25, 2.0, 4.0]],
        apply_calls: Arc::clone(&apply_calls),
        apply_mat_calls: Arc::clone(&apply_mat_calls),
        rhs_columns: Arc::clone(&rhs_columns),
    };

    let dense = op
        .materialize_dense()
        .expect("batched dense materialization");
    let expected = array![[2.0, 0.375, -0.375], [0.375, 3.0, 1.5], [-0.375, 1.5, 4.0]];
    assert_eq!(dense, expected);
    assert_eq!(
        apply_mat_calls.load(Ordering::Relaxed),
        1,
        "dense materialization must batch all identity columns into one apply_mat call"
    );
    assert_eq!(
        rhs_columns.load(Ordering::Relaxed),
        3,
        "the single batched materialization call must include every identity RHS"
    );
    assert_eq!(
        apply_calls.load(Ordering::Relaxed),
        0,
        "operators with batched apply_mat must not be probed column-by-column"
    );
}

#[test]
fn plan_analytic_hessian_selects_arc() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 3,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn plan_prefer_gradient_only_does_not_hide_analytic_hessian() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 3,
        psi_dim: 1,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: true,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn plan_survival_baseline_exact_hessian_selects_arc() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 3,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn plan_no_hessian_few_params_selects_bfgs() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 3,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn plan_no_hessian_many_params_selects_bfgs() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 12,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn plan_cost_only_few_params_fails_loudly_with_bfgs() {
    // No analytic gradient, no analytic Hessian, few params, no
    // fixed-point lane: a genuinely cost-only objective, which is a
    // programming error since every outer objective now supplies an
    // analytic gradient. The planner emits Bfgs, which the runner rejects
    // loudly for needing a gradient the objective cannot supply — by
    // design, a cost-only objective has no working primary.
    let cap = OuterCapability {
        gradient: Derivative::Unavailable,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 5,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
}

#[test]
fn plan_cost_only_many_params_with_fixed_point_still_efs() {
    // With the fixed-point lane eligible (many params,
    // fixed_point_available), a no-gradient/no-Hessian objective still
    // gets Efs.
    let cap = OuterCapability {
        gradient: Derivative::Unavailable,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 20,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Efs);
    assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn plan_cost_only_few_params_with_fixed_point_uses_only_valid_solver() {
    let cap = OuterCapability {
        gradient: Derivative::Unavailable,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 2,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let selected = plan(&cap);
    assert_eq!(selected.solver, Solver::Efs);
    assert_eq!(selected.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn no_gradient_efs_requires_and_accepts_explicit_full_coverage_certificate() {
    // Two coordinates is below the analytic-gradient BFGS crossover. With no
    // gradient, the explicit fixed-point lane is nevertheless the only valid
    // solver and must be selected rather than rejected on problem size.
    let n = 2;
    let center = Array1::from_elem(n, 0.25);
    let problem = OuterProblem::new(n)
        .with_gradient(Derivative::Unavailable)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(Array1::zeros(n))
        .with_max_iter(8)
        .with_tolerance(1.0e-8);
    let step_center = center.clone();
    let proof_center = center.clone();
    let mut objective = problem
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                let d = rho - &center;
                Ok(0.5 * d.dot(&d))
            },
            |_: &mut (), rho: &Array1<f64>| Ok(OuterEval::value_only(0.0, rho.len(), None)),
            None::<fn(&mut ())>,
            Some(move |_: &mut (), rho: &Array1<f64>| {
                let update = &step_center - rho;
                Ok(EfsEval {
                    cost: 0.5 * update.dot(&update),
                    steps: update.to_vec(),
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                    consecutive_restored_incumbents: None,
                })
            }),
        )
        .with_fixed_point_certificate(move |_: &mut (), rho: &Array1<f64>| {
            let update = &proof_center - rho;
            Ok(FixedPointCertificateEval {
                cost: 0.5 * update.dot(&update),
                coordinates: update
                    .iter()
                    .map(|&value| FixedPointCoordinateCertificate::covered(value, 1.0))
                    .collect(),
            })
        });
    let result = problem
        .run(&mut objective, "explicit fixed-point certificate")
        .expect("fully covered analytic fixed point must certify");
    assert_eq!(result.plan_used.solver, Solver::Efs);
    assert!(matches!(
        result.converged_via,
        Some(OuterConvergedVia::FixedPointStationary { .. })
    ));
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|certificate| certificate.stationarity.is_fixed_point())
    );
}

#[test]
fn no_gradient_efs_refuses_guarded_zero_as_uncovered_certificate() {
    let n = 9;
    let problem = OuterProblem::new(n)
        .with_gradient(Derivative::Unavailable)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(Array1::zeros(n))
        .with_max_iter(2);
    let mut objective = problem
        .build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), rho: &Array1<f64>| Ok(OuterEval::value_only(0.0, rho.len(), None)),
            None::<fn(&mut ())>,
            Some(|_: &mut (), rho: &Array1<f64>| {
                Ok(EfsEval {
                    cost: 0.0,
                    steps: vec![0.0; rho.len()],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                    consecutive_restored_incumbents: None,
                })
            }),
        )
        .with_fixed_point_certificate(|_: &mut (), rho: &Array1<f64>| {
            let mut coordinates =
                vec![FixedPointCoordinateCertificate::covered(0.0, 1.0); rho.len()];
            coordinates[3] = FixedPointCoordinateCertificate::uncovered(
                "guard held; no root-equivalent update exists",
            );
            Ok(FixedPointCertificateEval {
                cost: 0.0,
                coordinates,
            })
        });
    let error = problem
        .run(&mut objective, "uncovered fixed-point coordinate")
        .expect_err("an uncovered guarded zero must never certify");
    assert!(
        error
            .to_string()
            .contains("lacks root-equivalent analytic coverage"),
        "unexpected error: {error}"
    );
}

#[test]
fn plan_no_gradient_with_declared_hessian_stays_bfgs() {
    // Contradictory capability (Hessian declared but no gradient) keeps the
    // Bfgs reject-with-context path.
    let cap = OuterCapability {
        gradient: Derivative::Unavailable,
        hessian: DeclaredHessianForm::Either,
        n_params: 4,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn plan_boundary_8_params_uses_bfgs() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: SMALL_OUTER_BFGS_MAX_PARAMS,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn plan_boundary_9_params_uses_bfgs() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: SMALL_OUTER_BFGS_MAX_PARAMS + 1,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn plan_efs_selected_for_penalty_like_many_params() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Efs);
    assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn plan_penalty_like_without_fixed_point_stays_bfgs() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn plan_efs_selected_few_params_when_penalty_like() {
    // The former ≤8-coordinate BFGS crossover routed exactly the failing
    // small fits (2–7 ρ coords) into the fragile Wolfe/probe lane while large
    // fits got the robust trace-based fixed point. A fixed-point-capable,
    // all-penalty-like objective now routes to EFS at every dimension (see
    // `SMALL_OUTER_BFGS_MAX_PARAMS`).
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 5,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Efs);
    assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn plan_efs_not_selected_with_analytic_hessian() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 20,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    // Arc is always preferred when analytic Hessian is available.
    assert_eq!(p.solver, Solver::Arc);
}

#[test]
fn plan_efs_with_no_gradient_penalty_like_many_params() {
    // Even without analytic gradient, EFS works because it doesn't
    // need the gradient at all.
    let cap = OuterCapability {
        gradient: Derivative::Unavailable,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 20,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Efs);
    assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn plan_efs_allowed_with_barrier_config() {
    // When barrier_config is present (monotonicity constraints), EFS is
    // still selected at plan time. The runtime barrier-curvature guard
    // in the EFS loop handles safety.
    let barrier = BarrierConfig {
        tau: 1e-6,
        constrained_indices: vec![0, 1],
        lower_bounds: vec![0.0, 0.0],
        bound_signs: vec![1.0, 1.0],
    };
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: Some(barrier),
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Efs);
    assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn plan_efs_allowed_with_barrier_config_no_gradient() {
    // Even without analytic gradient, EFS is selected when all coords
    // are penalty-like and the problem is above the small-problem
    // BFGS cutoff, regardless of barrier presence.
    let barrier = BarrierConfig {
        tau: 1e-6,
        constrained_indices: vec![0],
        lower_bounds: vec![0.0],
        bound_signs: vec![1.0],
    };
    let cap = OuterCapability {
        gradient: Derivative::Unavailable,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 20,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: Some(barrier),
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Efs);
    assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn barrier_curvature_significant_blocks_efs_at_runtime() {
    // Verify that barrier_curvature_is_significant correctly detects
    // when coefficients are near their bounds.
    let barrier = BarrierConfig {
        tau: 1e-6,
        constrained_indices: vec![0],
        lower_bounds: vec![0.0],
        bound_signs: vec![1.0],
    };
    // β very close to bound → curvature is large
    let beta_near = Array1::from_vec(vec![0.001]);
    assert!(barrier.barrier_curvature_is_significant(&beta_near, 1.0, 0.01));

    // β far from bound → curvature is negligible
    let beta_far = Array1::from_vec(vec![10.0]);
    assert!(!barrier.barrier_curvature_is_significant(&beta_far, 1.0, 0.01));
}

#[test]
fn barrier_curvature_locally_concentrated_covers_both_failure_modes() {
    // τ = 1e-6 (BarrierConfig default).
    // For the dimensional check τ/Δ² ≥ saturation_threshold:
    //   • Δ = 1e-3 ⇒ τ/Δ² = 1.0 (right at saturation = 1.0)
    //   • Δ = 1e-2 ⇒ τ/Δ² = 1e-2 (well below)
    //   • Δ = 1e-4 ⇒ τ/Δ² = 100 (well above)
    let barrier = BarrierConfig {
        tau: 1e-6,
        constrained_indices: vec![0, 1],
        lower_bounds: vec![0.0, 0.0],
        bound_signs: vec![1.0, 1.0],
    };

    // Mode (b) symmetric near-boundary: slacks uniform & both small.
    // With saturation = 1.0, Δ = 1e-2 stays under the saturation
    // wall and ratio is healthy → not concentrated. Δ = 1e-4
    // saturates absolutely → concentrated.
    let mild_uniform = Array1::from_vec(vec![1.0e-2, 1.0e-2]);
    assert!(!barrier.barrier_curvature_locally_concentrated(&mild_uniform, 0.1, 1.0));
    let tight_uniform = Array1::from_vec(vec![1.0e-4, 1.0e-4]);
    assert!(barrier.barrier_curvature_locally_concentrated(&tight_uniform, 0.1, 1.0));

    // Mode (b) is gated by saturation_threshold: with a very large
    // threshold (effectively disabling (b)), tight uniform stops
    // tripping until you also relax (a) — the asymmetric ratio
    // check — which on uniform slacks is necessarily false.
    assert!(!barrier.barrier_curvature_locally_concentrated(&tight_uniform, 0.1, 1.0e9));

    // Large uniform slacks: neither mode trips.
    let large_uniform = Array1::from_vec(vec![10.0, 10.0]);
    assert!(!barrier.barrier_curvature_locally_concentrated(&large_uniform, 0.1, 1.0));

    // Mode (a) asymmetric concentration: one slack 100× tighter
    // than the other, all in a regime where mode (b) DOESN'T fire.
    // Δ_min = 1e-2 ⇒ τ/Δ² = 1e-2 ≪ 1.0 saturation. So only the
    // ratio check is doing work here.
    let imbalanced = Array1::from_vec(vec![1.0e-2, 1.0]);
    assert!(barrier.barrier_curvature_locally_concentrated(&imbalanced, 0.1, 1.0));
    // With a permissive ratio (1e-3) and mode (b) effectively off
    // (huge threshold), neither check trips.
    assert!(!barrier.barrier_curvature_locally_concentrated(&imbalanced, 1.0e-3, 1.0e9));

    // Infeasible (β ≤ l) → conservatively concentrated.
    let infeasible = Array1::from_vec(vec![-0.5, 1.0]);
    assert!(barrier.barrier_curvature_locally_concentrated(&infeasible, 0.1, 1.0));
}

#[test]
fn hessian_result_reports_analytic_variant() {
    let h = Array2::<f64>::eye(3);
    let result = HessianValue::Dense(h.clone());
    assert!(result.is_analytic());
    match result {
        HessianValue::Dense(extracted) => assert_eq!(extracted, h),
        HessianValue::Operator(_) | HessianValue::Unavailable => {
            panic!("expected dense analytic Hessian")
        }
    }
}

#[test]
fn zero_params_selects_arc() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 0,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn closure_objective_delegates() {
    let mut obj = ClosureObjective {
        state: 42_i32,
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        },
        cost_fn: |_: &mut i32, _: &Array1<f64>| Ok(1.0),
        eval_fn: |_: &mut i32, _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 1.0,
                gradient: Array1::zeros(1),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        eval_order_fn: None::<
            fn(&mut i32, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        >,
        reset_fn: Some(|st: &mut i32| {
            *st = 42;
        }),
        efs_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        fixed_point_certificate_fn: None,
        exact_polish_fn: None,
        screening_proxy_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<f64, EstimationError>>,
        seed_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
        terminal_eval_order: None,
    };
    assert_eq!(obj.capability().n_params, 1);
    assert_eq!(obj.eval_cost(&Array1::zeros(1)).unwrap(), 1.0);
}

#[test]
fn closure_terminal_order_overrides_efs_finalization() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable);
    let mut obj = problem
        .build_objective_with_eval_order(
            (Vec::<OuterEvalOrder>::new(), 0usize),
            |_, _: &Array1<f64>| Ok(0.0),
            |_, _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            |state: &mut (Vec<OuterEvalOrder>, usize),
             _: &Array1<f64>,
             order: OuterEvalOrder| {
                state.0.push(order);
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut (Vec<OuterEvalOrder>, usize))>,
            Some(
                |state: &mut (Vec<OuterEvalOrder>, usize), _: &Array1<f64>| {
                    state.1 += 1;
                    Ok(EfsEval {
                        cost: 0.0,
                        steps: vec![0.0],
                        beta: None,
                        psi_gradient: None,
                        psi_indices: None,
                        inner_hessian_scale: None,
                        logdet_enclosure_gap: None,
                        consecutive_restored_incumbents: None,
                    })
                },
            ),
        )
        .with_terminal_eval_order(OuterEvalOrder::ValueAndGradient);
    let efs_plan = OuterPlan {
        solver: Solver::Efs,
        hessian_source: HessianSource::EfsFixedPoint,
    };

    obj.finalize_outer_result(&array![0.0], &efs_plan)
        .expect("terminal analytic installation");

    assert_eq!(obj.state.0, vec![OuterEvalOrder::ValueAndGradient]);
    assert_eq!(
        obj.state.1, 0,
        "the EFS search evaluator must not overwrite analytic terminal ownership",
    );
}

#[test]
fn closure_objective_seed_inner_state_delegates_when_hook_present() {
    let mut obj = ClosureObjective {
        state: Vec::<f64>::new(),
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        },
        cost_fn: |_: &mut Vec<f64>, _: &Array1<f64>| Ok(0.0),
        eval_fn: |_: &mut Vec<f64>, _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        eval_order_fn: None::<
            fn(&mut Vec<f64>, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        >,
        reset_fn: None::<fn(&mut Vec<f64>)>,
        efs_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        fixed_point_certificate_fn: None,
        exact_polish_fn: None,
        screening_proxy_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<f64, EstimationError>>,
        seed_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
        terminal_eval_order: None,
    }
    .with_seed_inner_state(|state: &mut Vec<f64>, beta: &Array1<f64>| {
        state.extend(beta.iter().copied());
        Ok(SeedOutcome::Installed)
    });

    let outcome = obj.seed_inner_state(&array![1.5, -2.0]).unwrap();
    assert_eq!(outcome, SeedOutcome::Installed);
    assert_eq!(obj.state, vec![1.5, -2.0]);
}

#[test]
fn writable_inner_seed_hook_does_not_authorize_off_target_evaluations() {
    #[derive(Default)]
    struct State {
        seen: Vec<Array1<f64>>,
        seed_calls: usize,
    }

    let literal_seed = array![0.375];
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(literal_seed.clone())
        .with_bounds(array![-8.0], array![8.0])
        .with_max_iter(1)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let literal_for_cost = literal_seed.clone();
    let literal_for_eval = literal_seed.clone();
    let mut obj = problem
        .build_objective(
            State::default(),
            move |state: &mut State, theta: &Array1<f64>| {
                state.seen.push(theta.clone());
                let delta = theta[0] - literal_for_cost[0];
                Ok(0.5 * delta * delta)
            },
            move |state: &mut State, theta: &Array1<f64>| {
                state.seen.push(theta.clone());
                let delta = theta[0] - literal_for_eval[0];
                Ok(OuterEval {
                    cost: 0.5 * delta * delta,
                    gradient: array![delta],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![11.0]),
                })
            },
            Some(|_: &mut State| {}),
            None::<fn(&mut State, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
        .with_seed_inner_state(|state: &mut State, _: &Array1<f64>| {
            state.seed_calls += 1;
            Ok(SeedOutcome::Installed)
        });
    let config = problem.config();
    let cap = obj.capability();
    let the_plan = plan(&cap);
    let outcome = run_outer_with_plan(
        &mut obj,
        &config,
        "writable seed hook phase authority",
        &cap,
        &the_plan,
    )
    .expect("literal stationary seed must be accepted");
    assert!(matches!(outcome, PlanRunOutcome::Converged(_)));
    assert!(!obj.state.seen.is_empty());
    assert!(obj.state.seen.iter().all(|theta| {
        theta.len() == literal_seed.len()
            && theta
                .iter()
                .zip(literal_seed.iter())
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    }));
    assert_eq!(
        obj.state.seed_calls, 0,
        "a writable hook is cache replay capability, not permission to invent a continuation phase",
    );
}

#[test]
fn auxiliary_psi_is_never_synthetically_oversmoothed() {
    #[derive(Default)]
    struct State {
        seen: Vec<Array1<f64>>,
    }

    let literal_seed = array![0.25, -1.75];
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_psi_dim(1)
        .with_initial_rho(literal_seed.clone())
        .with_bounds(array![-8.0, -8.0], array![8.0, 8.0])
        .with_max_iter(1)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            num_auxiliary_trailing: 1,
            ..Default::default()
        });
    let literal_for_cost = literal_seed.clone();
    let literal_for_eval = literal_seed.clone();
    let mut obj = problem
        .build_objective(
            State::default(),
            move |state: &mut State, theta: &Array1<f64>| {
                state.seen.push(theta.clone());
                let delta = theta - &literal_for_cost;
                Ok(0.5 * delta.dot(&delta))
            },
            move |state: &mut State, theta: &Array1<f64>| {
                state.seen.push(theta.clone());
                let delta = theta - &literal_for_eval;
                Ok(OuterEval {
                    cost: 0.5 * delta.dot(&delta),
                    gradient: delta,
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![7.0]),
                })
            },
            Some(|_: &mut State| {}),
            None::<fn(&mut State, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
        .with_seed_inner_state(|_: &mut State, _: &Array1<f64>| Ok(SeedOutcome::Installed));
    let config = problem.config();
    let cap = obj.capability();
    let the_plan = plan(&cap);
    let outcome = run_outer_with_plan(
        &mut obj,
        &config,
        "psi phase authority",
        &cap,
        &the_plan,
    )
    .expect("literal stationary joint seed must be accepted");
    assert!(matches!(outcome, PlanRunOutcome::Converged(_)));
    assert!(!obj.state.seen.is_empty());
    assert!(obj
        .state
        .seen
        .iter()
        .all(|theta| theta[1].to_bits() == literal_seed[1].to_bits()));
}

#[test]
fn hybrid_efs_backtracking_uses_half_step_after_first_rejection() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 12,
        psi_dim: 1,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let mut obj = ClosureObjective {
        state: (),
        cap: cap.clone(),
        cost_fn: |_: &mut (), theta: &Array1<f64>| {
            let psi = theta[11];
            let cost = if (psi - 0.0).abs() < 1e-12 {
                1.0
            } else if (psi - 0.5).abs() < 1e-12 {
                0.5
            } else {
                2.0
            };
            Ok(cost)
        },
        eval_fn: |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[11].abs(),
                gradient: Array1::zeros(theta.len()),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        eval_order_fn: None::<
            fn(&mut (), &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        >,
        reset_fn: None::<fn(&mut ())>,
        efs_fn: Some(|_: &mut (), theta: &Array1<f64>| {
            let mut steps = vec![0.0; theta.len()];
            steps[11] = 1.0;
            Ok(EfsEval {
                cost: 1.0,
                steps,
                beta: None,
                psi_gradient: Some(array![1.0]),
                psi_indices: Some(vec![11]),
                inner_hessian_scale: None,
                logdet_enclosure_gap: None,
                consecutive_restored_incumbents: None,
            })
        }),
        fixed_point_certificate_fn: None,
        exact_polish_fn: None,
        screening_proxy_fn: None::<fn(&mut (), &Array1<f64>) -> Result<f64, EstimationError>>,
        seed_fn: None::<fn(&mut (), &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
        terminal_eval_order: None,
    };
    let mut bridge = OuterFixedPointBridge {
        obj: &mut obj,
        layout: cap.theta_layout(),
        barrier_config: None,
        fixed_point_tolerance: 1e-8,
        consecutive_psi_zero_iters: 0,
        last_restored_incumbent_streak: None,
        recurrent_incumbent_exit: Arc::new(Mutex::new(None)),
    };

    let sample = bridge
        .eval_step(&Array1::zeros(cap.n_params))
        .expect("hybrid EFS step should backtrack cleanly");

    assert_eq!(sample.status, FixedPointStatus::Continue);
    assert_eq!(sample.step.len(), cap.n_params);
    assert_eq!(sample.step[11], 0.5);
    assert!(
        sample
            .step
            .iter()
            .enumerate()
            .all(|(idx, &value)| idx == 11 || value == 0.0)
    );
}

#[test]
fn fixed_point_stops_on_second_consecutive_restored_incumbent_2241() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 1,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let mut obj = ClosureObjective {
        // Start at one so the first outer evaluation reports two inner restore
        // events. They happened inside one rho evaluation and must not, by
        // themselves, count as an outer recurrence.
        state: 1_usize,
        cap: cap.clone(),
        cost_fn: |_: &mut usize, rho: &Array1<f64>| Ok(10.0 - 1.0e-4 * rho[0]),
        eval_fn: |_: &mut usize, rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: 10.0 - 1.0e-4 * rho[0],
                gradient: array![-1.0],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        eval_order_fn: None::<
            fn(&mut usize, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        >,
        reset_fn: None::<fn(&mut usize)>,
        efs_fn: Some(|restores: &mut usize, rho: &Array1<f64>| {
            *restores += 1;
            Ok(EfsEval {
                // The criterion is still improving and the fixed-point step is
                // deliberately far above tolerance: only the model-state
                // certificate may stop this walk.
                cost: 10.0 - 1.0e-4 * rho[0],
                steps: vec![0.25],
                beta: None,
                psi_gradient: None,
                psi_indices: None,
                inner_hessian_scale: None,
                logdet_enclosure_gap: None,
                consecutive_restored_incumbents: Some(*restores),
            })
        }),
        fixed_point_certificate_fn: None,
        exact_polish_fn: None,
        screening_proxy_fn: None::<fn(&mut usize, &Array1<f64>) -> Result<f64, EstimationError>>,
        seed_fn: None::<fn(&mut usize, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
        terminal_eval_order: None,
    };
    let mut bridge = OuterFixedPointBridge {
        obj: &mut obj,
        layout: cap.theta_layout(),
        barrier_config: None,
        fixed_point_tolerance: 1.0e-8,
        consecutive_psi_zero_iters: 0,
        last_restored_incumbent_streak: None,
        recurrent_incumbent_exit: Arc::new(Mutex::new(None)),
    };

    let first = bridge
        .eval_step(&array![0.0])
        .expect("same-rho inner refinements are not an outer recurrence");
    assert_eq!(first.status, FixedPointStatus::Continue);

    let second = bridge
        .eval_step(&array![0.25])
        .expect("recurrent restored incumbent is a valid stop certificate");
    assert_eq!(second.status, FixedPointStatus::Stop);
    assert_eq!(
        second.value,
        10.0 - 1.0e-4 * 0.25,
        "retain the lower-cost restored point"
    );
}

#[test]
fn run_bfgs_mode_aware_eval_skips_hessian_work() {
    let seen_orders = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![1.0])
        .with_max_iter(1);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run on BFGS".to_string(),
            ))
        },
        {
            let seen_orders = Arc::clone(&seen_orders);
            move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                seen_orders.lock().unwrap().push(order);
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "mode-aware bfgs first order")
        .expect("BFGS should use the order-aware first-order bridge");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    let seen_orders = seen_orders.lock().unwrap();
    assert!(
        !seen_orders.is_empty(),
        "mode-aware eval hook should have been used"
    );
    assert!(
        seen_orders
            .iter()
            .all(|order| *order != OuterEvalOrder::ValueGradientHessian),
        "BFGS must not request Hessian work, saw {seen_orders:?}"
    );
    assert!(
        seen_orders.contains(&OuterEvalOrder::ValueAndGradient),
        "BFGS should request value+gradient at accepted points, saw {seen_orders:?}"
    );
}

// The historical bridge-side `rejects_oversized_bfgs_cost_probe_before_objective`
// test exercised a mechanism (returning `BFGS_LINE_SEARCH_REJECT_COST`
// from `eval_cost` on overreach) that has been retired in favor of
// `opt::Bfgs::with_axis_step_caps` — the line-search direction is now
// shortened up front by opt itself, so the bridge never sees an
// oversized probe in the first place. The equivalent invariant now
// lives in opt's `with_axis_step_caps` test surface.

#[test]
fn first_order_bridge_keeps_true_gradient_on_repeated_flat_cost() {
    let eval_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1000.0),
        {
            let eval_calls = Arc::clone(&eval_calls);
            move |_: &mut (), _: &Array1<f64>| {
                let call = eval_calls.fetch_add(1, Ordering::Relaxed);
                let cost = match call {
                    0 => 999.9995,
                    1 => 999.9990,
                    _ => 999.9987,
                };
                Ok(OuterEval {
                    cost,
                    gradient: array![4.0],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut bridge = OuterFirstOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        outer_inner_cap: None,
        iter_count: 0,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        value_probe_cache: Vec::new(),
        cost_stall: None,
        cost_stall_bounds: None,
        consecutive_probe_refusals: 0,
    };

    let first = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
        .expect("first flat-cost eval should expose the true gradient");
    let second = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
        .expect("second flat-cost eval should expose the true gradient");
    let third = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
        .expect("third flat-cost eval should expose the true gradient");
    let fourth = FirstOrderObjective::eval_grad(&mut bridge, &array![0.0])
        .expect("fourth flat-cost eval should expose the true gradient");

    assert_eq!(first.gradient[0], 4.0);
    assert_eq!(second.gradient[0], 4.0);
    assert_eq!(third.gradient[0], 4.0);
    assert_eq!(fourth.gradient[0], 4.0);
    assert_eq!(bridge.last_g_norm, Some(4.0));
    assert_eq!(eval_calls.load(Ordering::Relaxed), 4);
}

#[test]
fn outer_second_order_bridge_separates_first_and_second_order_requests() {
    let seen_orders = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        {
            let seen_orders = Arc::clone(&seen_orders);
            move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                seen_orders.lock().unwrap().push(order);
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: match order {
                        OuterEvalOrder::Value => HessianValue::Unavailable,
                        OuterEvalOrder::ValueAndGradient => HessianValue::Unavailable,
                        OuterEvalOrder::ValueGradientHessian => HessianValue::Dense(array![[2.0]]),
                    },
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: None,
        cost_stall_bounds: None,
    };
    let grad_sample = FirstOrderObjective::eval_grad(&mut bridge, &array![1.0]).expect("grad eval");
    assert_eq!(grad_sample.value, 1.0);
    assert_eq!(grad_sample.gradient, array![2.0]);
    let hess_sample =
        SecondOrderObjective::eval_hessian(&mut bridge, &array![1.0]).expect("hessian eval");
    assert_eq!(hess_sample.value, 1.0);
    assert_eq!(hess_sample.gradient, array![2.0]);
    assert_eq!(hess_sample.hessian, Some(array![[2.0]]));
    let seen_orders = seen_orders.lock().unwrap();
    assert!(
        *seen_orders
            == vec![
                OuterEvalOrder::ValueAndGradient,
                OuterEvalOrder::ValueGradientHessian
            ],
        "second-order bridge should split first-order and second-order requests, saw {seen_orders:?}"
    );
}

/// Phase 1.1 — On `HessianSource::Analytic` the bridge MUST surface a
/// fatal error rather than producing `SecondOrderSample { hessian: None }`
/// when the runtime returns `HessianValue::Unavailable`. A `None` here
/// would let `opt::SecondOrderCache::finite_difference_hessian` silently
/// estimate the Hessian by finite-differencing the gradient — at large-scale
/// scale, hours of work per silently-mis-routed step. The seed loop
/// should retry, demote, or fail loudly instead.
#[test]
fn analytic_route_unavailable_hessian_is_fatal() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), theta: &Array1<f64>, _order: OuterEvalOrder| {
            Ok(OuterEval {
                cost: theta[0] * theta[0],
                gradient: array![2.0 * theta[0]],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: None,
        cost_stall_bounds: None,
    };
    let err = SecondOrderObjective::eval_hessian(&mut bridge, &array![1.0])
        .expect_err("Analytic route must reject Unavailable Hessian, not pass None to opt");
    match err {
        ObjectiveEvalError::Fatal { message } => {
            assert!(
                message.contains("HessianSource::Analytic") && message.contains("Unavailable"),
                "fatal message should explain the analytic-route mismatch, saw: {message}"
            );
        }
        ObjectiveEvalError::Recoverable { message } => panic!(
            "Analytic-route Hessian violations must be Fatal (FD estimation is forbidden); \
                 got Recoverable: {message}"
        ),
    }
}

/// #1237 — On a near-separable multinomial fit the outer REML criterion
/// decreases monotonically as λ→0, so several log-λ directions slam to the
/// lower box bound and the ARC outer loop cycles to `max_iter` without ever
/// certifying a stationary point (the #1082 multinomial timeout). The
/// `OuterSecondOrderBridge` now carries the same cost-stall guard the BFGS
/// bridge does: once the REML score has stopped improving over
/// `COST_STALL_WINDOW` evals, the bridge returns the `Fatal` cost-stall
/// sentinel so the runner halts ARC at the published best iterate. The guard
/// fires from `eval_hessian` — `opt::Arc`'s per-iterate oracle evaluates the
/// (value, grad, Hessian) triple there and NEVER calls `eval_grad` on the ARC
/// route, so the guard must live where ARC actually steps. The converged
/// verdict rides on the BOUND-PROJECTED gradient: a direction pinned at the
/// bound with a persistent out-of-bounds ∂V/∂ρ is KKT-stationary even though
/// its raw gradient never vanishes. This drives the ARC bridge directly (no
/// 380s end-to-end fit) and asserts the stall is reached and certified.
#[test]
fn arc_bridge_finite_cost_stall_defers_at_bound_separation() {
    // A flat objective at the lower bound `rho = -10` whose raw gradient is a
    // constant `g = +1`: its descent step `-g = -1` points further DOWN, out of
    // the feasible box, so under the corrected KKT projection (#1074, a14b71220)
    // it is the infeasible bound-multiplier pull and projects to 0. The projected
    // KKT residual there is 0, so a stall is a CONVERGED optimum — exactly the
    // separation signature (the REML score has bottomed out but the unprojected
    // gradient keeps pushing λ→0 forever). NOTE: a NEGATIVE gradient at a lower
    // bound is FEASIBLE interior descent (step `-g = +1` points back into the
    // box) and is retained by the projection — it would (correctly) report
    // NON-converged, which is why the fixture uses `+1`.
    let lo = array![-10.0];
    let hi = array![10.0];
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), _: &Array1<f64>, order: OuterEvalOrder| {
            Ok(OuterEval {
                // Constant cost: the score has flat-lined (separation valley).
                cost: 1.0,
                // Gradient's descent step points out of the lower bound; raw norm
                // = 1 forever, but the bound-projected residual at rho=-10 is 0.
                gradient: array![1.0],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => HessianValue::Dense(array![[1.0]]),
                    _ => HessianValue::Unavailable,
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    // Threshold the projected residual (0 here) must clear; any positive value
    // certifies the at-bound stall as converged.
    let guard = CostStallGuard::new(1.0e-6, COST_STALL_WINDOW, 1.0e-3, exit.clone());
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((lo.clone(), hi.clone())),
    };
    // Hammer eval_hessian at the lower bound — the ARC per-iterate oracle path.
    // Every finite sample, including the one that fills the stall window, must
    // retain its Hessian so ARC owns the convergence verdict.
    for _ in 0..(COST_STALL_WINDOW + 2) {
        let sample = SecondOrderObjective::eval_hessian(&mut bridge, &lo)
            .expect("finite ARC stall sample must reach the second-order solver");
        assert_eq!(sample.hessian, Some(array![[1.0]]));
    }
    let published = exit.lock().unwrap().take().expect("best iterate published");
    assert!(
        !published.converged,
        "the bridge may retain a recovery checkpoint but cannot certify finite \
         second-order convergence before ARC checks reduced curvature"
    );
    assert_eq!(published.rho, lo, "best iterate is the bound-pinned ρ");
    assert_eq!(published.value, 1.0);
}

/// #979: a finite cost/gradient stall at an interior strict saddle must never
/// intercept the analytic Hessian. ARC needs the negative eigenvalue to take its
/// cubic hard-case step; a first-order bridge sentinel would strand the search at
/// the same indefinite point later rejected by final certification.
#[test]
fn arc_bridge_finite_stall_delivers_interior_negative_curvature() {
    let point = array![0.0];
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        |_: &mut (), _: &Array1<f64>, order: OuterEvalOrder| {
            Ok(OuterEval {
                cost: 1.0,
                gradient: array![0.0],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => HessianValue::Dense(array![[-1.0]]),
                    _ => HessianValue::Unavailable,
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((array![-10.0], array![10.0])),
    };

    for _ in 0..5 {
        let sample = SecondOrderObjective::eval_hessian(&mut bridge, &point)
            .expect("strict-saddle Hessian must reach ARC after the stall window fills");
        assert_eq!(sample.hessian, Some(array![[-1.0]]));
    }
    let published = exit.lock().unwrap().take().expect("best checkpoint published");
    assert!(!published.converged);
}

/// Near-separable multinomial timeout (#1082/#1237), FEASIBLE bound-pinned arm.
/// Here every trial is feasible and the raw cost keeps DECREASING (λ→0 lowers
/// the REML criterion monotonically), but the bound-PROJECTED gradient is
/// already zero — the iterate is KKT-stationary at the box bound and the
/// remaining descent is pure bound-pinned drift. A naive cost-improvement test
/// resets the no-improvement streak on every step (the cost really is dropping)
/// and the loop never certifies; `opt::Arc`'s own gradient-tolerance check never
/// trips either, because it tests the RAW gradient, which points out of the box
/// forever. The guard now counts a KKT-stationary-at-bound trial as
/// no-improvement, so the window fills and ARC halts at the best feasible
/// iterate. This drives the ARC bridge directly and asserts the halt.
#[test]
fn arc_bridge_finite_stall_defers_kkt_stationary_bound_descent() {
    let lo = array![-10.0];
    let hi = array![10.0];
    // Strictly-decreasing cost so the cost-improvement test alone would NEVER
    // fire; the gradient's descent step points out of the lower bound (`g = +1`,
    // step `-g = -1` exits the box), so under the corrected KKT projection
    // (#1074, a14b71220) it projects to 0 (raw |g|=1, projected KKT residual at
    // rho=-10 is 0) and the halt rides on stationarity.
    let step = std::cell::Cell::new(0u32);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), _: &Array1<f64>, order: OuterEvalOrder| {
            let k = step.get();
            step.set(k + 1);
            Ok(OuterEval {
                // Monotonically decreasing by far more than the rel-tol floor:
                // a pure cost-stall test could never fill its window here.
                cost: 1.0 - (k as f64),
                // Gradient whose descent step exits the lower bound: raw norm = 1
                // forever, bound-projected residual = 0 (KKT-stationary).
                gradient: array![1.0],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => HessianValue::Dense(array![[1.0]]),
                    _ => HessianValue::Unavailable,
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(1.0e-6, COST_STALL_WINDOW, 1.0e-3, exit.clone());
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((lo.clone(), hi.clone())),
    };
    for _ in 0..(COST_STALL_WINDOW + 2) {
        let sample = SecondOrderObjective::eval_hessian(&mut bridge, &lo)
            .expect("finite bound sample must reach ARC with curvature");
        assert_eq!(sample.hessian, Some(array![[1.0]]));
    }
    let published = exit.lock().unwrap().take().expect("best iterate published");
    assert!(
        !published.converged,
        "ARC, not the finite bridge stall, must certify a PSD bound optimum"
    );
    assert_eq!(published.rho, lo, "best iterate is the bound-pinned ρ");
}

/// Near-separable multinomial timeout (#1082/#1237), infeasible-trial arm. ARC
/// finds one feasible iterate, then keeps probing the unbounded λ→0 separating
/// region where the inner softmax solve does not converge and every trial comes
/// back INFEASIBLE (cost = +∞). Those infeasible trials are rejected by the
/// finite-eval validator BEFORE the finite cost-stall guard sees them, so the
/// no-improvement window can never fill — and the outer loop used to grind to
/// `max_iter` (the timeout). The bridge now feeds infeasible trials to the
/// guard's dedicated infeasible-streak path: a run of `COST_STALL_WINDOW`
/// consecutive infeasible trials after a feasible best halts ARC at that best
/// feasible iterate. This drives the ARC bridge directly and asserts the halt.
#[test]
fn arc_bridge_cost_stall_halts_on_infeasible_separation_run() {
    let lo = array![-10.0];
    let hi = array![10.0];
    // The single feasible point: a small projected gradient so the eventual
    // halt certifies CONVERGED (a clean stationary optimum on the feasible side).
    let feasible_rho = array![0.0];
    let eval_idx = std::cell::Cell::new(0usize);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let feasible_for_obj = feasible_rho.clone();
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), x: &Array1<f64>, order: OuterEvalOrder| {
            let n = eval_idx.get();
            eval_idx.set(n + 1);
            // First call: the lone feasible iterate (finite cost, tiny gradient).
            // Every later call: an infeasible λ→0 probe (cost = +∞).
            if n == 0 && x == &feasible_for_obj {
                Ok(OuterEval {
                    cost: 1.0,
                    gradient: array![1.0e-9],
                    hessian: match order {
                        OuterEvalOrder::ValueGradientHessian => HessianValue::Dense(array![[1.0]]),
                        _ => HessianValue::Unavailable,
                    },
                    inner_beta_hint: None,
                })
            } else {
                Ok(OuterEval::infeasible(1))
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(1.0e-6, COST_STALL_WINDOW, 1.0e-3, exit.clone());
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((lo.clone(), hi.clone())),
    };
    // One feasible eval records the best; the next `COST_STALL_WINDOW` infeasible
    // evals fill the infeasible-streak window and trip the sentinel.
    let mut sentinel_fired = false;
    // First: the feasible iterate.
    SecondOrderObjective::eval_hessian(&mut bridge, &feasible_rho)
        .expect("feasible iterate must evaluate cleanly");
    let separating = array![-10.0];
    for _ in 0..(COST_STALL_WINDOW + 2) {
        match SecondOrderObjective::eval_hessian(&mut bridge, &separating) {
            Ok(_) => panic!("an infeasible (cost=∞) trial must not return a finite sample"),
            Err(ObjectiveEvalError::Fatal { message }) => {
                assert_eq!(
                    message, ARC_INFEASIBLE_STALL_SENTINEL,
                    "infeasible-run halt must use the non-converged ARC checkpoint sentinel"
                );
                sentinel_fired = true;
                break;
            }
            // Before the window fills, infeasible trials surface as the normal
            // recoverable non-finite-cost error (the optimizer shrinks + retries).
            Err(ObjectiveEvalError::Recoverable { .. }) => {}
        }
    }
    assert!(
        sentinel_fired,
        "ARC bridge must halt after {} consecutive infeasible separating trials",
        COST_STALL_WINDOW
    );
    let published = exit.lock().unwrap().take().expect("best iterate published");
    assert!(
        !published.converged,
        "an infeasible current probe has no synchronized Hessian, so its stored \
         best can only be a non-converged checkpoint"
    );
    assert_eq!(
        published.rho, feasible_rho,
        "the published best must be the lone FEASIBLE iterate, not a separating λ→0 probe"
    );
    assert_eq!(published.value, 1.0, "published cost is the feasible cost");
}

/// Regression for the `with_initial_sample` ARC route: opt serves the seed
/// sample from its internal cache, so the bridge never sees an `eval_hessian`
/// call at that feasible point. The cost-stall guard must still know about the
/// seed before infeasible λ→0 trial probes arrive; otherwise
/// `observe_infeasible` has no best finite iterate to publish and ARC can grind
/// to `max_iter`.
#[test]
fn arc_cost_stall_guard_uses_cached_initial_sample_as_feasible_best() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let seed = array![0.0, 0.0];
    guard.observe_seed(&seed, 10.0, 5.0e-4);

    let separating_probe = array![-10.0, -10.0];
    assert!(
        matches!(
            guard.observe_infeasible(&separating_probe),
            CostStallVerdict::Continue
        ),
        "first infeasible probe should only start the streak"
    );
    assert!(
        matches!(
            guard.observe_infeasible(&separating_probe),
            CostStallVerdict::Continue
        ),
        "second infeasible probe should still be below the window"
    );
    assert!(
        matches!(
            guard.observe_infeasible(&separating_probe),
            CostStallVerdict::Converged
        ),
        "third infeasible probe should halt back to the cached seed"
    );

    let published = exit.lock().unwrap().take().expect("seed best published");
    assert_eq!(published.rho, seed);
    assert_eq!(published.value, 10.0);
    assert_eq!(published.grad_norm, 5.0e-4);
    assert!(published.converged);
}

/// A run of infeasible cubic trials cannot justify halting at an incumbent
/// whose synchronized reduced Hessian proves negative curvature. That pattern
/// is a domain-wall encounter during saddle escape, not a separating optimum;
/// ARC must retain control so increasing sigma can shrink the escape step.
#[test]
fn arc_infeasible_stall_refuses_cached_strict_saddle_2316() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit);
    let seed = array![0.0, 0.0];
    guard.observe_second_order_seed(&seed, 10.0, 5.0e-2, Some(false));

    let infeasible = array![-8.0, -8.0];
    for probe in 0..9 {
        assert!(
            matches!(
                guard.observe_infeasible(&infeasible),
                CostStallVerdict::Continue
            ),
            "strict-saddle infeasible probe {probe} must return control to ARC"
        );
    }
}

#[test]
fn reduced_hessian_psd_keeps_weak_bound_direction_in_critical_cone_2316() {
    let lower = array![0.0, -2.0];
    let upper = array![2.0, 2.0];
    let point = array![0.0, 0.0];
    let hessian = array![[-1.0, 0.0], [0.0, 2.0]];

    assert_eq!(
        reduced_hessian_psd_at_point(&point, &array![0.0, 0.0], &hessian, Some((&lower, &upper)),),
        Some(false),
        "a zero-multiplier lower-bound axis remains in the critical cone"
    );
    assert_eq!(
        reduced_hessian_psd_at_point(&point, &array![1.0, 0.0], &hessian, Some((&lower, &upper))),
        Some(true),
        "strict complementarity removes the bound-normal direction"
    );
}

#[test]
fn bfgs_bridge_halts_infeasible_probe_run_back_to_cached_seed() {
    let seed = array![0.0];
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        |_: &mut (), _: &Array1<f64>, _: OuterEvalOrder| Ok(OuterEval::infeasible(1)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, COST_STALL_WINDOW, 1.0e-3, exit.clone());
    guard.observe_seed(&seed, 10.0, 5.0e-4);
    let lo = array![-10.0];
    let hi = array![10.0];
    let mut bridge = OuterFirstOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        outer_inner_cap: None,
        iter_count: 0,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        value_probe_cache: Vec::new(),
        cost_stall: Some(guard),
        cost_stall_bounds: Some((lo, hi)),
        consecutive_probe_refusals: 0,
    };

    // A real BFGS line search probes a *sequence of distinct* trial ρ along its
    // search direction, not one point repeated. The bridge caches each probed ρ
    // bit-exactly (`value_probe_cache`/`same_outer_point`) and short-circuits an
    // exact repeat WITHOUT re-entering the cost-stall guard — so re-probing a
    // single fixed ρ would only ever register ONE infeasible observation and the
    // streak could never fill the window. Walk distinct points (1.0, 2.0, …) so
    // each probe is a fresh guard observation, exactly as the line search does.
    let mut sentinel_fired = false;
    for i in 0..(COST_STALL_WINDOW + 2) {
        let trial = array![1.0 + i as f64];
        match ZerothOrderObjective::eval_cost(&mut bridge, &trial) {
            Ok(cost) => panic!("infeasible probe unexpectedly returned finite cost {cost}"),
            Err(ObjectiveEvalError::Fatal { message }) => {
                assert_eq!(
                    message, COST_STALL_CONVERGED_SENTINEL,
                    "BFGS infeasible-probe halt must use the shared cost-stall sentinel"
                );
                sentinel_fired = true;
                break;
            }
            Err(ObjectiveEvalError::Recoverable { .. }) => {}
        }
    }
    assert!(
        sentinel_fired,
        "BFGS bridge must halt after {} consecutive infeasible probes when a finite seed is cached",
        COST_STALL_WINDOW
    );
    let published = exit.lock().unwrap().take().expect("seed best published");
    assert_eq!(published.rho, seed);
    assert_eq!(published.value, 10.0);
    assert_eq!(published.grad_norm, 5.0e-4);
    assert!(published.converged);
}

#[test]
fn constrained_stationary_probe_replaces_stale_nonstationary_best() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let stale_seed = array![0.0, 0.0];
    guard.observe_seed(&stale_seed, 1.0, 2.0);

    // Genuine near-separable case: the REML criterion decreases monotonically
    // toward the λ→0 lower bound, so the bound probe carries a cost AT OR BELOW
    // the seed. It is the certificate-bearing optimum and must be published with
    // a converged verdict — superseding the (non-stationary, higher raw-cost)
    // seed.
    let boundary_probe = array![-10.0, -10.0];
    let verdict = guard.observe_constrained_stationary(&boundary_probe, 0.5, 0.0, true, None);
    assert!(
        matches!(verdict, CostStallVerdict::Converged),
        "a finite constrained-stationary separation probe should halt immediately"
    );

    let published = exit
        .lock()
        .unwrap()
        .take()
        .expect("stationary probe published");
    assert_eq!(
        published.rho, boundary_probe,
        "publish the KKT-certified boundary probe, not the older raw-cost best"
    );
    assert_eq!(published.value, 0.5);
    assert_eq!(published.grad_norm, 0.0);
    assert!(published.converged);
}

/// #1355 regression: a constrained-stationary (lower-bound separation) probe
/// that REGRESSES materially on the best feasible iterate must NOT be adopted.
///
/// For a multi-penalty RKHS smooth (duchon/matern) an over-smoothing collapse
/// corner — some operator penalties railed at the λ→0 lower bound, OTHERS railed
/// at the λ→∞ upper bound shrinking the fit to a bare constant — passes the
/// `lower_bound_outward_active_count` separation test yet has a REML cost far
/// worse than the interior optimum the optimizer already found (typically the
/// grid-prepass seed). Adopting it published a degenerate EDF≈1 constant fit.
/// The guard must keep the strictly-better incumbent instead.
#[test]
fn constrained_stationary_probe_keeps_better_incumbent() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    // A good interior fit (the prepass seed): low cost on a genuinely flat
    // valley floor — a residual outer gradient modestly above tolerance but
    // below FLAT_VALLEY_STALL_GRAD_CEILING (not yet certified stationary, yet a
    // legitimate flat-valley floor rather than a #1426 stuck stall, so the guard
    // halts-and-publishes it rather than escaping to keep descending).
    let good_seed = array![3.0, 30.0, 3.0, 3.0];
    let good_seed_grad = FLAT_VALLEY_STALL_GRAD_CEILING * 0.5;
    guard.observe_seed(&good_seed, -231.86, good_seed_grad);

    // The collapse corner: two axes pinned at the λ→0 lower bound (looks like a
    // separation probe) while two more rail at λ→∞; its cost is hundreds of
    // units WORSE than the incumbent.
    let collapse_corner = array![30.0, 29.95, -30.0, -30.0];
    let verdict =
        guard.observe_constrained_stationary(&collapse_corner, 587.84, 0.0, true, None);

    // The probe regresses, so the guard must NOT halt-and-publish it as the
    // optimum on this single observation; it folds in as an ordinary
    // non-improving step (the window has not yet filled).
    assert!(
        matches!(verdict, CostStallVerdict::Continue),
        "a constrained-stationary probe that regresses the incumbent must not be \
         adopted as the optimum"
    );
    // `observe_seed` seeds the shared exit cell with the feasible best (#1371),
    // so the cell tracks the GOOD incumbent — never the regressing corner —
    // while the no-improvement window is still filling.
    {
        let cell = exit.lock().unwrap();
        let best = cell.as_ref().expect("seed best tracked in exit cell");
        assert_eq!(
            best.rho, good_seed,
            "the exit cell must track the good incumbent, not the regressing corner"
        );
    }

    // Driving the no-improvement window to its limit halts on the GOOD
    // incumbent, never on the collapse corner.
    guard.observe_constrained_stationary(&collapse_corner, 587.84, 0.0, true, None);
    let final_verdict =
        guard.observe_constrained_stationary(&collapse_corner, 587.84, 0.0, true, None);
    assert!(
        !matches!(final_verdict, CostStallVerdict::Continue),
        "the stall window should eventually fill and halt"
    );
    let published = exit
        .lock()
        .unwrap()
        .take()
        .expect("incumbent best published on stall");
    assert_eq!(
        published.rho, good_seed,
        "halt back to the good interior incumbent, not the collapse corner"
    );
    assert_eq!(published.value, -231.86);
}

/// #1426 regression: a cost stall whose best-iterate projected gradient is FAR
/// above the outer tolerance is NOT a flat-valley floor and must NOT be halted
/// as one. On ~7% of gamma/log datasets at default k the inner PIRLS hit its
/// iteration cap, leaving the outer objective and analytic gradient
/// inconsistent: the cost flatlined while the projected gradient stayed at
/// |g|≈11. The old guard classified that as a `FlatValleyStall`, halted, and
/// shipped a silent near-unpenalized full-basis overfit. The fixed guard must
/// instead return `StuckKeepDescending` (no halt) so the optimizer keeps
/// descending toward the well-penalized optimum, for a bounded number of escapes
/// before it falls back to a halt so a genuinely pathological surface still
/// terminates.
#[test]
fn cost_stall_far_above_tolerance_keeps_descending_not_flat_valley() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let seed = array![0.0, 0.0];
    // Best iterate has a HUGE residual gradient (the #1426 |g|≈11 signature),
    // orders of magnitude above the ceiling — the inner solve did not converge.
    let stuck_grad = 10.9;
    assert!(
        stuck_grad > FLAT_VALLEY_STALL_GRAD_CEILING,
        "test premise: the stuck residual must exceed the flat-valley ceiling"
    );
    guard.observe_seed(&seed, 10.0, stuck_grad);

    // Drive the no-improvement window to the stall via infeasible λ→0 probes,
    // exactly as the outer loop does when ARC/BFGS keep probing the unpenalized
    // ridge. The third probe fills the window and reaches the stall verdict.
    let probe = array![-10.0, -10.0];
    assert!(matches!(
        guard.observe_infeasible(&probe),
        CostStallVerdict::Continue
    ));
    assert!(matches!(
        guard.observe_infeasible(&probe),
        CostStallVerdict::Continue
    ));
    let verdict = guard.observe_infeasible(&probe);
    assert!(
        matches!(verdict, CostStallVerdict::StuckKeepDescending { .. }),
        "a cost stall with |g|≈11 ≫ ceiling must NOT be classified as a \
         flat-valley floor and halted (#1426); it must keep descending. Got {:?}",
        std::mem::discriminant(&verdict)
    );
    // Crucially: no halt was shipped — the optimizer is allowed to continue.
    // (The exit cell still tracks the running best via `publish_best_so_far`,
    // but the verdict did not request a halt.)

    // The escape budget is finite: after STUCK_STALL_MAX_ESCAPES escapes the
    // guard falls back to a FlatValleyStall halt so the loop still terminates.
    let mut last = verdict;
    for _ in 0..(STUCK_STALL_MAX_ESCAPES + 3) {
        // Re-fill the window each round (each escape reset it).
        guard.observe_infeasible(&probe);
        guard.observe_infeasible(&probe);
        last = guard.observe_infeasible(&probe);
        if matches!(last, CostStallVerdict::FlatValleyStall { .. }) {
            break;
        }
    }
    assert!(
        matches!(last, CostStallVerdict::FlatValleyStall { .. }),
        "after exhausting the bounded escape budget the guard must eventually \
         halt (reported non-converged) so the loop terminates"
    );
    let published = exit
        .lock()
        .unwrap()
        .take()
        .expect("best published on eventual halt");
    assert!(
        !published.converged,
        "a stuck halt above tolerance must report converged=false (never claim \
         a converged result with |g| far above tolerance)"
    );
}

/// #2253 regression: the stuck-stall cap bounds CONSECUTIVE fruitless escapes,
/// not the lifetime number of productive shelf crossings. The Qwen K=1 circle
/// replay consumed its historical budget even though a late escape restored a
/// large objective decrease; retaining the lifetime count killed the seed at
/// the next shelf even though the trajectory was descending.
#[test]
fn cost_stall_productive_descent_replenishes_escape_budget_2253() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let seed = array![0.0, 0.0];
    let stuck_grad = 10.9;
    guard.observe_seed(&seed, 10.0, stuck_grad);

    // Consume the entire escape budget without any real improvement. Each
    // three-probe infeasible window must be granted exactly one escape.
    let infeasible_probe = array![-10.0, -10.0];
    for escape_idx in 0..STUCK_STALL_MAX_ESCAPES {
        assert!(matches!(
            guard.observe_infeasible(&infeasible_probe),
            CostStallVerdict::Continue
        ));
        assert!(matches!(
            guard.observe_infeasible(&infeasible_probe),
            CostStallVerdict::Continue
        ));
        let verdict = guard.observe_infeasible(&infeasible_probe);
        assert!(
            matches!(verdict, CostStallVerdict::StuckKeepDescending { .. }),
            "escape {} of {} must still keep descending",
            escape_idx + 1,
            STUCK_STALL_MAX_ESCAPES,
        );
    }

    // A trusted, super-floor accepted improvement proves the escapes were
    // productive. This must replenish the consecutive-fruitless budget.
    let descended = array![1.0, -1.0];
    let descent = guard.observe(&descended, -20.0, stuck_grad, true);
    assert!(
        matches!(descent, CostStallVerdict::Continue),
        "genuine descent must resume the trajectory rather than halt"
    );

    // At the next shelf the guard must grant a fresh escape. Without c28e6f9f7
    // the lifetime counter remains exhausted and this is FlatValleyStall.
    assert!(matches!(
        guard.observe_infeasible(&infeasible_probe),
        CostStallVerdict::Continue
    ));
    assert!(matches!(
        guard.observe_infeasible(&infeasible_probe),
        CostStallVerdict::Continue
    ));
    let verdict = guard.observe_infeasible(&infeasible_probe);
    assert!(
        matches!(verdict, CostStallVerdict::StuckKeepDescending { .. }),
        "a productive descent must replenish the stuck-stall escape budget (#2253)"
    );

    let published = exit.lock().unwrap();
    let best = published.as_ref().expect("running best remains published");
    assert_eq!(best.rho, descended);
    assert_eq!(best.value, -20.0);
    assert!(!best.converged);
}

/// #1426 companion (revised for the #509 score-relative escape gate): a GENUINE
/// flat-valley floor — cost flatlined AND the projected gradient has floored at
/// its irreducible band, i.e. only modestly above the SCORE-RELATIVE
/// certified-stationary bound — must still halt as a `FlatValleyStall` (the
/// legitimately-flat REML surface of #1082/#1237). The keep-descending escape
/// must not weaken that path.
///
/// The discriminator is now score-relative, not the legacy fixed absolute
/// `FLAT_VALLEY_STALL_GRAD_CEILING`: a stall whose residual is within
/// `FLAT_VALLEY_STALL_ESCAPE_MARGIN` of `score_relative_grad_bound` is "essentially
/// at the band" and halts directly. With a realistic REML score (`|value| ≈ 1e3`)
/// the band caps at `FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP = 1.0`, so a residual of
/// `1.2` sits just above the certify band (not converged) yet within `1.5×` of it
/// (a true floor) and must halt. This is the legitimately-flat case; the #509
/// monotone seed-park instead floors WELL clear of the band (|g| ≈ 2 on a band ≈
/// 0.6) and is granted escapes — covered by
/// `cost_stall_above_score_relative_band_keeps_descending`.
#[test]
fn cost_stall_modestly_above_tolerance_still_halts_as_flat_valley() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let seed = array![0.0, 0.0];
    // Realistic REML score scale so the score-relative band caps at 1.0.
    let score = -1.0e3;
    // Residual just above the certified band (1.0) but within the 1.5× escape
    // margin (1.5): a real flat-valley floor — the surface has genuinely
    // flattened and no escape will drive the residual lower.
    let valley_grad = 1.2;
    assert!(
        valley_grad > FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP
            && valley_grad < FLAT_VALLEY_STALL_ESCAPE_MARGIN * FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP,
        "test premise: a flat-valley floor sits just above the certify band, within the escape margin"
    );
    guard.observe_seed(&seed, score, valley_grad);

    let probe = array![-10.0, -10.0];
    guard.observe_infeasible(&probe);
    guard.observe_infeasible(&probe);
    let verdict = guard.observe_infeasible(&probe);
    assert!(
        matches!(verdict, CostStallVerdict::FlatValleyStall { .. }),
        "a residual at the score-relative band is a genuine flat-valley floor and \
         must halt as before (#1082/#1237 unaffected). Got {:?}",
        std::mem::discriminant(&verdict)
    );
    let published = exit.lock().unwrap().take().expect("best published");
    assert!(!published.converged);
}

/// #509 regression: a cost stall at an INTERIOR ρ whose projected gradient is
/// well clear of the score-relative certified-stationary band still has a genuine
/// feasible descent direction and must NOT be halted as a flat valley — it must
/// keep descending. A shape-constrained (box-reparam β=Tγ) smooth whose inequality
/// is non-binding stalls this way near the integer seed: the cumulative-sum
/// coordinate change makes per-step cost progress fall below the relative floor for
/// a window even though the projected gradient (|g| ≈ 2 on a score ≈ 600, band ≈
/// 0.6) still descends strongly toward the well-penalized REML optimum. The legacy
/// fixed `FLAT_VALLEY_STALL_GRAD_CEILING = 5.0` halted it (2 < 5) and parked the
/// fit at its seed; the score-relative escape gate keeps it descending.
#[test]
fn cost_stall_above_score_relative_band_keeps_descending() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let seed = array![3.0, -3.0];
    let score = -6.0e2;
    // Well above the certified band (≈ 0.6 = 1e-3·600) and above the 1.5× escape
    // margin (≈ 0.9), but BELOW the legacy fixed ceiling (5.0) — exactly the band
    // the old gate falsely halted.
    let descending_grad = 2.0;
    guard.observe_seed(&seed, score, descending_grad);

    let probe = array![-10.0, -10.0];
    guard.observe_infeasible(&probe);
    guard.observe_infeasible(&probe);
    let verdict = guard.observe_infeasible(&probe);
    assert!(
        matches!(verdict, CostStallVerdict::StuckKeepDescending { .. }),
        "an interior stall well clear of the score-relative band has feasible \
         descent left and must keep descending, not halt (#509). Got {:?}",
        std::mem::discriminant(&verdict)
    );
}

/// #2241 — the probe-noise-floor certificate: a stalled walk whose residual
/// projected gradient is BELOW σ̂/Δ (the criterion's measured evaluation-noise
/// floor over the stall window, divided by the radius the accepted steps
/// actually probed) is flat relative to its own noise scale and must halt
/// CONVERGED, even when the residual sits above both the absolute tolerance
/// and the score-relative band.
#[test]
fn cost_stall_certifies_converged_below_probe_noise_floor_2241() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    // Tight absolute threshold and a small score, so neither the absolute nor
    // the score-relative band (1e-3·(1+10) = 0.011) can certify |g| = 0.5:
    // only the noise-floor certificate can.
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-9, exit.clone());
    let residual_grad = 0.5;
    guard.observe_seed(&array![0.0, 0.0], 10.0, residual_grad);
    // Three accepted, trusted iterates with O(1e-3) probe steps whose values
    // scatter by O(1e-3) around the incumbent: no improvement beyond the
    // relative floor, so the window fills, and the measured noise floor is
    // σ̂ = median{8e-4, 4e-4, 6e-4} = 6e-4 over a probed radius Δ = 1e-3
    // ⇒ certified gradient band σ̂/Δ = 0.6 > 0.5 = |g|.
    guard.observe(&array![1.0e-3, 0.0], 10.0 + 8.0e-4, residual_grad, true);
    guard.observe(&array![2.0e-3, 0.0], 10.0 + 4.0e-4, residual_grad, true);
    let verdict = guard.observe(&array![3.0e-3, 0.0], 10.0 + 1.0e-3, residual_grad, true);
    assert!(
        matches!(verdict, CostStallVerdict::Converged),
        "a stall whose residual gradient cannot move the criterion beyond its \
         own evaluation-noise floor over the probed radius is flat relative to \
         its noise scale and must certify (#2241). Got {:?}",
        std::mem::discriminant(&verdict)
    );
    let published = exit.lock().unwrap().take().expect("halt published");
    assert!(published.converged);
    let noise_bound = published
        .noise_grad_bound
        .expect("the halt must carry the measured noise-floor bound");
    assert!(
        residual_grad <= noise_bound && noise_bound <= FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP,
        "the certifying bound must cover the residual and respect the absolute \
         cap; got {noise_bound}"
    );
}

/// #1689 — a criterion-flat ARC halt that the guard certified through the
/// score-relative band must retain that halt provenance through the mandatory
/// analytic final-point certificate. The old runner marked only NON-converged
/// stalls as `CostStallFlatValley`; a converged stall lost the marker and the
/// final certificate incorrectly reverted to the raw absolute bound.
#[test]
fn criterion_flat_provenance_preserves_score_relative_certificate_1689() {
    let score = -982.0_f64;
    let residual = 0.042_f64;
    assert!(
        residual > 1.0e-6 && residual < flat_valley_converged_grad_bound(score),
        "fixture must require the criterion-flat band rather than raw tolerance"
    );
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_tolerance(1.0e-10)
        .with_objective_scale(Some(1_200.0));
    let config = problem.config();
    let mut obj = problem.build_objective_with_eval_order(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(score + residual * rho[0]),
        move |_: &mut (), rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: score + residual * rho[0],
                gradient: array![residual],
                hessian: HessianValue::Dense(array![[0.0]]),
                inner_beta_hint: None,
            })
        },
        move |_: &mut (), rho: &Array1<f64>, _: OuterEvalOrder| {
            Ok(OuterEval {
                cost: score + residual * rho[0],
                gradient: array![residual],
                hessian: HessianValue::Dense(array![[0.0]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut result = outer_result_with_gradient_norm(
        array![0.0],
        score,
        10,
        Some(residual),
        true,
        OuterPlan {
            solver: Solver::Arc,
            hessian_source: HessianSource::Analytic,
        },
    );
    result.operator_stop_reason = Some(OperatorTrustRegionStopReason::CostStallFlatValley);

    let certificate = certify_outer_optimality(
        &mut obj,
        &config,
        "#1689 criterion-flat provenance regression",
        &mut result,
    )
    .expect("score-relative flat certificate must survive final remeasurement");
    assert!(certificate.certifies());
    assert!(certificate.stationarity.bound() >= flat_valley_converged_grad_bound(score));
    assert!(matches!(
        result.converged_via,
        Some(OuterConvergedVia::CriterionFlat { .. })
    ));
}

/// #2241 companion — the noise certificate must be un-gameable by collapsed
/// step sizes: Δ → 0 inflates the raw σ̂/Δ arbitrarily, but the bound is capped
/// at `FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP`, so a genuinely steep point
/// (|g| = 2 > cap) can never be certified through the noise route; it keeps
/// descending exactly as the score-relative escape gate (#509) demands.
#[test]
fn probe_noise_floor_capped_never_certifies_steep_point_2241() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-9, exit.clone());
    let steep_grad = 2.0;
    guard.observe_seed(&array![0.0, 0.0], 10.0, steep_grad);
    // Degenerate 1e-9 probe steps with O(1e-3) value scatter: raw σ̂/Δ ≈ 1e6,
    // which without the cap would "certify" a steep stuck point.
    guard.observe(&array![1.0e-9, 0.0], 10.0 + 8.0e-4, steep_grad, true);
    guard.observe(&array![2.0e-9, 0.0], 10.0 + 4.0e-4, steep_grad, true);
    let verdict = guard.observe(&array![3.0e-9, 0.0], 10.0 + 1.0e-3, steep_grad, true);
    assert!(
        !matches!(verdict, CostStallVerdict::Converged),
        "collapsed probe steps must never manufacture a noise-floor convergence \
         at a genuinely steep point (#2241 anti-gaming cap)"
    );
}

#[test]
fn lower_bound_outward_axes_mark_separation_stationarity() {
    let lower = array![-10.0, -10.0, -10.0, -10.0];
    let upper = array![10.0, 10.0, 10.0, 10.0];
    let rho = array![-10.0, -10.0, 0.25, 1.0];
    // #1074/#1082 sign fix (mirrors a14b712): "outward" at an active LOWER bound
    // means the minimization descent step -g_i exits BELOW the box, i.e. g_i > 0.
    // Idx 0,1 are pinned at the lower bound with strong POSITIVE gradients (the
    // genuine outward/separation pull) and must be counted; idx 2,3 are interior
    // and must not. (The earlier fixture used negative gradients here, which is
    // feasible interior descent under the corrected convention, not outward.)
    let gradient = array![2.0e-2, 4.0e-2, 3.0, -1.0];

    assert_eq!(
        lower_bound_outward_active_count(&rho, &gradient, Some(&(lower, upper)), 1.0e-3),
        LOWER_BOUND_SEPARATION_ACTIVE_MIN,
        "two lower-bound axes with outward (positive) gradients are enough to \
         identify a separation-bound stationary probe"
    );
}

// Phase 5 (Cargo dep at opt 0.3) replaces the gam-side bridge
// seed cache with `opt::{Bfgs, Arc, NewtonTrustRegion}::with_initial_sample`.
// The two cache tests that lived here have been removed;
// equivalent integration coverage now lives upstream as
// `opt::tests::with_initial_sample_serves_first_call_from_cache`
// and `opt::tests::bfgs_with_initial_sample_serves_first_call_from_cache`.
// The fatal-on-Analytic-route contract (Phase 1.1) is still tested
// here since it lives in gam's `build_bridge_hessian_for_source`.

#[test]
fn outer_config_default() {
    let cfg = OuterConfig::default();
    assert_eq!(cfg.tolerance, 1e-5);
    assert_eq!(cfg.max_iter, 200);
    assert_eq!(cfg.rho_bound, 30.0);
}

#[test]
fn plan_hybrid_efs_selected_for_psi_coords_many_params() {
    // When ψ (design-moving) coords are present and the problem is above
    // the small-problem BFGS cutoff, the planner should select HybridEfs
    // instead of falling back to BFGS.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 1,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::HybridEfs);
    assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
}

#[test]
fn plan_psi_without_fixed_point_stays_bfgs() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 1,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn plan_hybrid_efs_no_gradient_selected_for_psi_coords() {
    // Even without analytic gradient, hybrid EFS works because the
    // gradient is computed internally by the unified evaluator.
    let cap = OuterCapability {
        gradient: Derivative::Unavailable,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 1,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::HybridEfs);
    assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
}

// ----------------------------------------------------------------------
// Routing regression tests (spec section 12).
//
// Post-#1 (compute-budget failure paths removed) and #2 (Hessian
// cost-gating in custom_family.rs removed), the planner no longer
// downgrades `(Analytic, Analytic)` to BFGS at any problem size. The
// contract is:
//
//   high dense work + analytic+analytic     → ARC + Analytic
//                                             (runtime then chooses
//                                              operator HVP per family)
//   high dense work + analytic + Unavailable → BFGS + BfgsApprox
//                                             (matrix-free not advertised
//                                              by the family — BFGS is
//                                              still the right choice)
//
// `routing_log_line()` exposes a stable token that large-scale log
// regressions in tests/bench_large_scale_runner_test.py pin against.
// ----------------------------------------------------------------------

fn cap_for_routing(
    gradient: Derivative,
    hessian: DeclaredHessianForm,
    n_params: usize,
) -> OuterCapability {
    OuterCapability {
        gradient,
        hessian,
        n_params,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    }
}

#[test]
fn routing_analytic_analytic_stays_arc_at_large_scale() {
    // Large-scale standard GAM (n=320K, p=65, k=6) used to trigger the
    // aggregate `k·n·p²` cost-driven downgrade. Post-#1 the planner has
    // no scale-driven downgrade, so `(Analytic, Analytic)` must stay on
    // ARC + Analytic regardless of the problem dimensions.
    let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 6);
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn routing_analytic_analytic_stays_arc_at_dense_work_scale() {
    // n=3·10⁵, p=300 used to trigger the per-inner-solve `n·p²` downgrade
    // (`2.7·10¹⁰ ≫ 5·10⁹`). Post-#1, no work-hint API exists; ARC stays.
    let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 3);
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn routing_unavailable_hessian_routes_to_bfgs() {
    // Spec section 12: when the family cannot provide a second derivative
    // (matrix-free or otherwise), BFGS is the correct route.
    let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Unavailable, 8);
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn routing_explicit_prefer_gradient_only_does_not_override_exact_hessian() {
    // The primary REML outer must never hide an analytic Hessian behind a
    // quasi-Newton route. Auxiliary gradient-only optimizers are separate
    // solver classes; this flag is ignored for Analytic+Analytic primary
    // capabilities.
    let mut cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 6);
    cap.prefer_gradient_only = true;
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn routing_log_line_arc_analytic_does_not_advertise_matrix_free() {
    // Token pinned by tests/bench_large_scale_runner_test.py. Renaming
    // any of these substrings is a log-regression and breaks downstream
    // grep patterns.
    let p = OuterPlan {
        solver: Solver::Arc,
        hessian_source: HessianSource::Analytic,
    };
    let line = p.routing_log_line();
    assert!(line.contains("solver=Arc"), "got {line}");
    assert!(line.contains("hessian=Analytic"), "got {line}");
    assert!(line.contains("matrix-free=false"), "got {line}");
}

#[test]
fn routing_log_line_bfgs_reports_no_matrix_free() {
    let p = OuterPlan {
        solver: Solver::Bfgs,
        hessian_source: HessianSource::BfgsApprox,
    };
    let line = p.routing_log_line();
    assert!(line.contains("solver=Bfgs"), "got {line}");
    assert!(line.contains("hessian=BfgsApprox"), "got {line}");
    assert!(line.contains("matrix-free=false"), "got {line}");
}

#[test]
fn routing_log_line_efs_reports_no_matrix_free() {
    // EFS variants don't expose a Hessian operator either, so the
    // matrix-free token is `false`.
    for source in [
        HessianSource::EfsFixedPoint,
        HessianSource::HybridEfsFixedPoint,
    ] {
        let p = OuterPlan {
            solver: Solver::Efs,
            hessian_source: source,
        };
        assert!(
            p.routing_log_line().contains("matrix-free=false"),
            "{:?} should not advertise matrix-free",
            source
        );
    }
}

// ----------------------------------------------------------------------
// Per-family routing regression tests.
//
// Each family that gains matrix-free Hessian operators must, at the
// OuterProblem build site, declare both derivatives `Analytic` so the
// planner stays on ARC + Analytic. These tests pin that contract from
// the planner side. The runtime's choice between dense-Hessian-assembly
// and operator-HVPs is independent of the planner; a separate per-family
// test (in the family's own module) should pin that.
//
// ----------------------------------------------------------------------

#[test]
fn routing_custom_family_gamlss_stays_on_arc_when_both_derivs_analytic() {
    // Post-#5/#12, GAMLSS advertises matrix-free directional operators
    // for the joint Hessian; the OuterProblem build site must declare
    // both derivatives Analytic so ARC + Analytic stays in effect.
    let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 4);
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn routing_matern_iso_kappa_stays_on_arc_when_both_derivs_analytic() {
    // Post-#7, Matern/TPS spatial κ/τ derivative drifts ship as
    // HyperOperators; planner contract: (Analytic, Analytic) → ARC.
    let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 5);
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn routing_matern_iso_large_kappa_dim_stays_on_arc_with_analytic_hessian() {
    // Spatial isotropic κ no longer declares Hessian unavailable when
    // kappa_dim > 30.  Large κ blocks are represented by exact HVP
    // operators at evaluation time, so the planner must keep second-order
    // ARC instead of selecting HybridEFS.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 37,
        psi_dim: 31,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn routing_marginal_slope_stays_on_arc_when_both_derivs_analytic() {
    // Bernoulli/survival marginal-slope: the planner contract is the
    // same — (Analytic, Analytic) → ARC + Analytic. Runtime selects
    // operator HVPs via `use_joint_matrix_free_path`.
    let cap = cap_for_routing(Derivative::Analytic, DeclaredHessianForm::Either, 3);
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn plan_hybrid_efs_selected_few_params() {
    // ψ-carrying fixed-point objectives route to HybridEfs at every
    // dimension: the former ≤8-coordinate BFGS crossover sent exactly the
    // failing small fits into the fragile Wolfe/probe lane (see
    // `SMALL_OUTER_BFGS_MAX_PARAMS`).
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 5,
        psi_dim: 1,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::HybridEfs);
    assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
}

#[test]
fn plan_exact_hvp_capability_selects_arc_even_when_fixed_point_is_available() {
    // Large spatial/custom-family problems may also expose EFS/HybridEFS
    // fixed-point traces, but an explicit dense Hessian or exact HVP
    // operator is stronger geometry. The planner must therefore select
    // ARC + Analytic rather than cost-demoting to BFGS/EFS when the
    // evaluator advertises second-order capability.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 64,
        psi_dim: 16,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: true,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
    assert_eq!(p.hessian_source, HessianSource::Analytic);
}

#[test]
fn plan_hybrid_efs_not_selected_with_analytic_hessian() {
    // Arc is always preferred when analytic Hessian is available,
    // even with ψ coordinates.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 20,
        psi_dim: 1,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Arc);
}

#[test]
fn plan_pure_efs_not_hybrid_when_all_penalty_like() {
    // When all coords are penalty-like (no ψ), pure EFS is selected
    // even if has_psi_coords is false.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Efs);
    assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
}

#[test]
fn automatic_fallbacks_preserve_analytic_hessian_for_arc_primary() {
    // For an (Analytic, Analytic) capability the planner emits ARC. The
    // cascade MUST NOT add a BFGS+BfgsApprox demotion: doing so discards
    // the analytic outer Hessian ARC was using, replaces it with a
    // strictly weaker rank-2 approximation, and silently masks ARC's
    // actual failure mode (budget exhaustion, indefinite curvature)
    // under a BFGS Strong-Wolfe plateau. ARC budget exhaustion is
    // handled by the per-attempt retry ladder in
    // `run_outer_with_strategy`; once that is exhausted, the caller
    // sees the genuine analytic-Hessian non-convergence verbatim.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 12,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    assert_eq!(plan(&cap).solver, Solver::Arc);
    let attempts = automatic_fallback_attempts(&cap);
    assert!(
        attempts.is_empty(),
        "ARC primary must not lateral-demote to BFGS+BfgsApprox; \
             ARC budget retries live in the runner",
    );
}

#[test]
fn automatic_fallbacks_from_efs_prefer_analytic_bfgs_over_fd() {
    // When the primary plan is EFS, the first fallback must keep the
    // analytic gradient and just disable the fixed-point path so the
    // planner picks gradient-based BFGS. Silently downgrading to finite
    // differences here was the long-standing production bug we are
    // guarding against.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    assert_eq!(plan(&cap).solver, Solver::Efs);

    let attempts = automatic_fallback_attempts(&cap);
    assert!(!attempts.is_empty(), "EFS failure must have a fallback");
    assert_eq!(attempts[0].gradient, Derivative::Analytic);
    assert_eq!(attempts[0].hessian, DeclaredHessianForm::Unavailable);
    assert!(attempts[0].disable_fixed_point);
    assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);

    assert!(
        attempts.iter().all(|c| c.gradient == Derivative::Analytic),
        "fallback cascade must stay on analytic-gradient attempts",
    );
}

#[test]
fn automatic_fallbacks_from_hybrid_efs_prefer_analytic_bfgs_over_fd() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 2,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    assert_eq!(plan(&cap).solver, Solver::HybridEfs);

    let attempts = automatic_fallback_attempts(&cap);
    assert!(!attempts.is_empty());
    assert_eq!(attempts[0].gradient, Derivative::Analytic);
    assert!(attempts[0].disable_fixed_point);
    assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);
}

#[test]
fn disabled_fallback_hybrid_efs_capability_routes_to_bfgs_primary() {
    // Production Matérn60 exact adaptive regularization at large scale:
    // rho_dim=3 retained quadratic penalties, psi_dim=6 adaptive λ/ε
    // coordinates, n_params=9, analytic gradient, and exact outer Hessian
    // cost-gated unavailable. Structurally this is HybridEFS-shaped, but
    // HybridEFS with ψ coordinates is not a standalone primary solver: its
    // ψ backtracking path can legitimately request the first-order escape
    // ladder. If that ladder is disabled, the runner must route the primary
    // attempt directly to BFGS instead of relying on call sites to remember
    // `.with_disable_fixed_point(true)`.
    let trapped_cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 9,
        psi_dim: 6,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    assert_eq!(plan(&trapped_cap).solver, Solver::HybridEfs);

    let disabled_config = OuterConfig {
        fallback_policy: FallbackPolicy::Disabled,
        ..OuterConfig::default()
    };
    let primary_cap = primary_capability_for_config(
        trapped_cap.clone(),
        &disabled_config,
        "large-scale exact adaptive",
    );
    assert!(primary_cap.disable_fixed_point);
    assert_eq!(plan(&primary_cap).solver, Solver::Bfgs);

    let pure_efs_cap = OuterCapability {
        psi_dim: 0,
        ..trapped_cap.clone()
    };
    assert_eq!(plan(&pure_efs_cap).solver, Solver::Efs);
    let pure_primary_cap =
        primary_capability_for_config(pure_efs_cap.clone(), &disabled_config, "pure EFS");
    assert!(!pure_primary_cap.disable_fixed_point);
    assert_eq!(plan(&pure_primary_cap).solver, Solver::Efs);

    let no_gradient_cap = OuterCapability {
        gradient: Derivative::Unavailable,
        ..trapped_cap.clone()
    };
    assert_eq!(plan(&no_gradient_cap).solver, Solver::HybridEfs);
    let no_gradient_primary_cap = primary_capability_for_config(
        no_gradient_cap.clone(),
        &disabled_config,
        "gradient-unavailable hybrid EFS",
    );
    assert!(!no_gradient_primary_cap.disable_fixed_point);
    assert_eq!(plan(&no_gradient_primary_cap).solver, Solver::HybridEfs);

    let automatic_config = OuterConfig::default();
    let automatic_cap = primary_capability_for_config(
        trapped_cap.clone(),
        &automatic_config,
        "large-scale exact adaptive",
    );
    assert!(!automatic_cap.disable_fixed_point);
    assert_eq!(plan(&automatic_cap).solver, Solver::HybridEfs);

    let automatic_attempts = automatic_fallback_attempts(&trapped_cap);
    assert!(!automatic_attempts.is_empty());
    assert!(automatic_attempts[0].disable_fixed_point);
    assert_eq!(plan(&automatic_attempts[0]).solver, Solver::Bfgs);
}

#[test]
fn disabled_fallback_hybrid_efs_problem_uses_bfgs_without_calling_efs() {
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(9)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_psi_dim(6)
        .with_fallback_policy(FallbackPolicy::Disabled)
        .with_initial_rho(Array1::zeros(9))
        .with_max_iter(5);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            Some(move |_: &mut (), _: &Array1<f64>| {
                efs_calls.fetch_add(1, Ordering::Relaxed);
                Err(EstimationError::RemlOptimizationFailed(format!(
                    "{} synthetic large-scale adaptive HybridEFS escape",
                    EFS_FIRST_ORDER_FALLBACK_MARKER,
                )))
            })
        },
    );

    let result = problem
        .run(&mut obj, "disabled fallback marker")
        .expect("disabled-fallback HybridEFS-shaped problem should route directly to BFGS");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(
        efs_calls.load(Ordering::Relaxed),
        0,
        "central primary-capability canonicalization should avoid the EFS hook entirely"
    );
}

#[test]
fn automatic_fallbacks_without_gradient_stop_at_fixed_point_status() {
    for (psi_dim, expected_solver) in [(0, Solver::Efs), (2, Solver::HybridEfs)] {
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 15,
            psi_dim,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, expected_solver);
        assert!(
            automatic_fallback_attempts(&cap).is_empty(),
            "gradient-unavailable fixed-point capabilities must not fabricate a BFGS fallback",
        );
    }
}

#[test]
fn automatic_fallbacks_do_not_repeat_arc_when_fixed_point_is_irrelevant() {
    // The contract here is that the cascade does not lateral-hop ARC
    // through the EFS planner arm when `fixed_point_available=true` is
    // incidentally set on an (Analytic, Analytic) capability that the
    // planner already chose ARC for. Combined with the
    // analytic-Hessian-preservation contract enforced by
    // `automatic_fallbacks_preserve_analytic_hessian_for_arc_primary`,
    // the ARC primary now has zero degraded fallbacks — the runner's
    // ARC budget-bump retry ladder owns recovery.
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Either,
        n_params: 15,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    assert_eq!(plan(&cap).solver, Solver::Arc);

    let attempts = automatic_fallback_attempts(&cap);
    assert!(
        attempts.is_empty(),
        "ARC primary with incidental fixed_point_available must not \
             cascade through the EFS arm or lateral-demote to BFGS",
    );
}

#[test]
fn plan_disable_fixed_point_forces_bfgs_even_when_efs_eligible() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: 15,
        psi_dim: 0,
        fixed_point_available: true,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: true,
    };
    let p = plan(&cap);
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn run_malformed_gradient_seed_surfaces_as_error() {
    // A capability that declares Analytic gradient but returns a malformed
    // one must fail loudly. The previous numerical-gradient fallback masked
    // the underlying bug by silently spinning a cost-only BFGS; that path is
    // disabled in production.
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(Array1::zeros(2))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        |_: &mut (), _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let err = problem
        .run(&mut obj, "test gradient mismatch")
        .expect_err("malformed analytic gradient must surface as error");
    assert!(
        matches!(err, EstimationError::RemlOptimizationFailed(_)),
        "unexpected error variant: {err:?}",
    );
}

#[test]
fn run_bfgs_ignores_malformed_hessian_payload() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![0.0])
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[0] * theta[0],
                gradient: array![2.0 * theta[0]],
                // First-order paths must ignore Hessian payload quality.
                hessian: HessianValue::Dense(array![[f64::NAN, 0.0], [0.0, 1.0]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "bfgs should ignore malformed hessian payload")
        .expect("valid first-order data should be enough for BFGS");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(result.plan_used.hessian_source, HessianSource::BfgsApprox);
}

#[test]
fn finite_outer_eval_reports_gradient_length_mismatch() {
    let err = finite_outer_eval_or_error(
        "test gradient mismatch",
        OuterThetaLayout::new(2, 0),
        OuterEval {
            cost: 0.0,
            gradient: Array1::zeros(1),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        },
    )
    .expect_err("gradient mismatch should be rejected");
    let message = match err {
        ObjectiveEvalError::Recoverable { message } | ObjectiveEvalError::Fatal { message } => {
            message
        }
    };
    assert!(
        message.contains("outer gradient length mismatch"),
        "unexpected error: {message}"
    );
}

#[test]
fn run_with_initial_seed_still_considers_generated_candidates() {
    let generated =
        crate::seeding::generate_rho_candidates(1, None, &gam_problem::SeedConfig::default());
    let valid_seed = generated
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let expected_seed = valid_seed.clone();
    let initial_seed = array![9.0];
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_initial_rho(initial_seed)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let valid_seed = valid_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(0.0)
                } else {
                    Ok(f64::INFINITY)
                }
            }
        },
        move |_: &mut (), theta: &Array1<f64>| {
            if theta == valid_seed {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            } else {
                Ok(OuterEval::infeasible(theta.len()))
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "generated seed should remain reachable")
        .expect("generated seed should still be eligible when an initial seed is provided");
    assert_eq!(result.rho, expected_seed);
}

#[derive(Clone, Copy)]
enum ReactiveDomainMode {
    FiniteAtColdSeed,
    OpensFromHeavySide,
    NeverOpens,
}

/// Analytic one-dimensional objective used to pin the reactive domain-entry
/// contract. Its real optimum is the literal seed. In the repair fixture the
/// exact seed starts outside the domain, while installation of the objective's
/// smoother scalar entry opens a connected finite branch that remains defined
/// when the coupled path returns to its literal scalar target.
struct ReactiveDomainObjective {
    seed: f64,
    mode: ReactiveDomainMode,
    domain_open: bool,
    exact_seed_value_evals: usize,
    off_seed_value_evals: usize,
    rho_value_evals: Vec<f64>,
    derivative_evals: usize,
    installed_scalar_states: Vec<crate::continuation_path::ContinuationScalarState>,
    checkpoint_domain_open: Option<bool>,
}

impl ReactiveDomainObjective {
    fn new(seed: f64, mode: ReactiveDomainMode) -> Self {
        Self {
            seed,
            mode,
            domain_open: matches!(mode, ReactiveDomainMode::FiniteAtColdSeed),
            exact_seed_value_evals: 0,
            off_seed_value_evals: 0,
            rho_value_evals: Vec::new(),
            derivative_evals: 0,
            installed_scalar_states: Vec::new(),
            checkpoint_domain_open: None,
        }
    }

    fn is_exact_seed(&self, rho: &Array1<f64>) -> bool {
        rho.len() == 1 && rho[0].to_bits() == self.seed.to_bits()
    }

    fn finite_cost(&self, rho: &Array1<f64>) -> f64 {
        let delta = rho[0] - self.seed;
        0.5 * delta * delta
    }

    fn scalar_contract() -> crate::continuation_path::ContinuationScalarContract {
        crate::continuation_path::ContinuationScalarContract::new(
            crate::continuation_path::ContinuationScalarState::new(4.0, vec![0.0])
                .expect("valid reactive entry"),
            crate::continuation_path::ContinuationScalarState::new(0.75, vec![2.0])
                .expect("valid literal target"),
        )
        .expect("matching scalar dimensions")
    }
}

impl OuterObjective for ReactiveDomainObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.rho_value_evals.push(rho[0]);
        if self.is_exact_seed(rho) {
            self.exact_seed_value_evals += 1;
        } else {
            self.off_seed_value_evals += 1;
        }
        if self.domain_open {
            Ok(self.finite_cost(rho))
        } else {
            Ok(f64::INFINITY)
        }
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        self.derivative_evals += 1;
        if !self.domain_open {
            return Ok(OuterEval::infeasible(rho.len()));
        }
        let delta = rho[0] - self.seed;
        Ok(OuterEval {
            cost: 0.5 * delta * delta,
            gradient: array![delta],
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        })
    }

    fn reset(&mut self) {
        self.domain_open = matches!(self.mode, ReactiveDomainMode::FiniteAtColdSeed);
    }

    fn seed_inner_state(&mut self, _: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        Ok(SeedOutcome::NoSlot)
    }

    fn outer_domain_upper_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(Some(array![2.5]))
    }

    fn outer_domain_lower_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(Some(array![-2.5]))
    }

    fn reactive_domain_scalar_contract(
        &self,
    ) -> Result<Option<crate::continuation_path::ContinuationScalarContract>, EstimationError> {
        Ok(Some(Self::scalar_contract()))
    }

    fn install_reactive_domain_scalar_state(
        &mut self,
        state: &crate::continuation_path::ContinuationScalarState,
    ) -> Result<(), EstimationError> {
        self.installed_scalar_states.push(state.clone());
        if matches!(self.mode, ReactiveDomainMode::OpensFromHeavySide)
            && !state.bitwise_eq(Self::scalar_contract().target())
        {
            self.domain_open = true;
        }
        Ok(())
    }

    fn begin_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        if self.checkpoint_domain_open.is_some() {
            return Err(EstimationError::RemlOptimizationFailed(
                "nested reactive test waypoint".to_string(),
            ));
        }
        self.checkpoint_domain_open = Some(self.domain_open);
        Ok(())
    }

    fn commit_reactive_domain_waypoint(&mut self, _: &Array1<f64>) -> Result<(), EstimationError> {
        self.checkpoint_domain_open.take().ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(
                "reactive test commit without checkpoint".to_string(),
            )
        })?;
        Ok(())
    }

    fn rollback_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        self.domain_open = self.checkpoint_domain_open.take().ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(
                "reactive test rollback without checkpoint".to_string(),
            )
        })?;
        Ok(())
    }
}

fn run_reactive_domain_fixture(
    mode: ReactiveDomainMode,
) -> (
    Result<OuterResult, EstimationError>,
    ReactiveDomainObjective,
) {
    const SEED: f64 = 0.125;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![SEED])
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .with_max_iter(4);
    let mut objective = ReactiveDomainObjective::new(SEED, mode);
    let result = problem.run(&mut objective, "reactive-domain-entry fixture");
    (result, objective)
}

fn reactive_arrival_state(
    rho: Array1<f64>,
    cost: f64,
) -> crate::estimate::reml::continuation::ContinuationState {
    crate::estimate::reml::continuation::ContinuationState {
        last_rho: rho,
        last_eval: OuterEval {
            cost,
            gradient: Array1::zeros(0),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        },
        last_beta: Array1::zeros(0),
        steps_accepted: 1,
    }
}

#[test]
fn reactive_domain_arrival_accepts_exact_finite_literal_seed() {
    let seed = array![0.125, -0.75];
    let state = reactive_arrival_state(seed.clone(), 3.5);
    assert!(
        reactive_arrival_postcondition(&state, &seed).is_ok(),
        "an exact finite literal-seed state must authorize arrival"
    );
}

#[test]
fn active_outer_domain_refuses_singleton_search_interval() {
    let mut config = OuterConfig::default();
    config.bounds = Some((array![-1_000.0], array![1_000.0]));
    let error = install_objective_domain(&mut config, 1, Some(array![700.0]), Some(array![700.0]))
        .expect_err("an active optimizer coordinate needs a nonzero-width interval");
    let message = error.to_string();
    assert!(
        message.contains("no finite searchable interval")
            && message.contains("fixed-rho")
            && message.contains("lower < upper"),
        "unexpected singleton-domain refusal: {message}"
    );
}

#[test]
fn custom_box_and_seed_are_intersected_with_both_objective_faces() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-1_000.0], array![1_000.0])
        .with_initial_rho(array![-900.0])
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .with_max_iter(1);
    let mut objective = ReactiveDomainObjective::new(0.125, ReactiveDomainMode::FiniteAtColdSeed);
    drop(problem.run(&mut objective, "two-sided objective-domain intersection"));
    assert!(!objective.rho_value_evals.is_empty());
    assert!(
        objective
            .rho_value_evals
            .iter()
            .all(|&rho| (-2.5..=2.5).contains(&rho)),
        "configured box or seed leaked past objective domain: {:?}",
        objective.rho_value_evals
    );
    assert!(
        objective
            .rho_value_evals
            .iter()
            .any(|rho| rho.to_bits() == (-2.5_f64).to_bits()),
        "the out-of-domain initial seed must project onto the exact lower face"
    );
}

#[test]
fn reactive_domain_arrival_rejects_an_earlier_waypoint() {
    let seed = array![0.125, -0.75];
    let state = reactive_arrival_state(array![0.25, -0.75], 3.5);
    let error = reactive_arrival_postcondition(&state, &seed)
        .expect_err("an earlier continuation waypoint must not authorize arrival");
    assert!(
        error.contains("not the literal seed"),
        "unexpected error: {error}"
    );
}

#[test]
fn reactive_domain_arrival_rejects_nonfinite_literal_seed_evidence() {
    let seed = array![0.125, -0.75];
    let state = reactive_arrival_state(seed.clone(), f64::INFINITY);
    let error = reactive_arrival_postcondition(&state, &seed)
        .expect_err("undefined exact-seed evidence must not authorize arrival");
    assert!(
        error.contains("retained non-finite evidence"),
        "unexpected error: {error}"
    );
}

#[test]
fn reactive_domain_entry_leaves_finite_seed_on_zero_heavy_work_path() {
    let (result, objective) = run_reactive_domain_fixture(ReactiveDomainMode::FiniteAtColdSeed);
    let result = result.expect("a finite exact seed must optimize and certify directly");
    assert_eq!(result.rho, array![0.125]);
    assert!(
        objective.installed_scalar_states.is_empty(),
        "a finite literal seed must install no continuation scalar state"
    );
    assert_eq!(
        objective.off_seed_value_evals, 0,
        "a finite literal seed must not evaluate a heavy continuation waypoint"
    );
    assert!(objective.exact_seed_value_evals > 0);
    assert!(objective.derivative_evals > 0);
    assert!(
        objective.checkpoint_domain_open.is_none(),
        "successful path must close every waypoint transaction"
    );
}

#[test]
fn reactive_domain_entry_repairs_nonfinite_seed_from_heavy_side() {
    let (result, objective) = run_reactive_domain_fixture(ReactiveDomainMode::OpensFromHeavySide);
    let result = result.expect("the connected heavy-side branch must reach a finite exact seed");
    assert_eq!(result.rho, array![0.125]);
    assert!(
        objective
            .installed_scalar_states
            .first()
            .expect("scalar entry installation")
            .bitwise_eq(ReactiveDomainObjective::scalar_contract().entry()),
        "the first installed waypoint must be the objective-owned literal entry"
    );
    assert!(
        objective
            .installed_scalar_states
            .last()
            .expect("scalar target installation")
            .bitwise_eq(ReactiveDomainObjective::scalar_contract().target()),
        "successful arrival must install the literal scalar target"
    );
    assert!(
        objective.off_seed_value_evals > 0,
        "the initially undefined seed must activate heavy continuation work"
    );
    assert!(
        objective
            .rho_value_evals
            .iter()
            .any(|rho| rho.to_bits() == 2.5_f64.to_bits()),
        "reactive entry must evaluate the objective-owned legal upper face"
    );
    assert!(
        objective
            .rho_value_evals
            .iter()
            .all(|rho| rho.to_bits() != 30.0_f64.to_bits()),
        "the generic +30 box must not leak past an objective-domain contract"
    );
    assert!(
        objective.exact_seed_value_evals >= 2,
        "the exact seed must be probed before entry and verified after arrival"
    );
    assert!(objective.derivative_evals > 0);
    assert!(
        objective.checkpoint_domain_open.is_none(),
        "successful repaired path must close every waypoint transaction"
    );
}

#[test]
fn reactive_domain_entry_keeps_unrepairable_seed_as_typed_refusal() {
    let (result, objective) = run_reactive_domain_fixture(ReactiveDomainMode::NeverOpens);
    let error = result.expect_err("a path that never establishes finite evidence must refuse");
    assert!(
        error.to_string().contains("reactive domain entry refused"),
        "unexpected refusal: {error}"
    );
    assert!(
        !objective.installed_scalar_states.is_empty(),
        "an undefined capable seed must install the certified scalar entry"
    );
    assert_eq!(
        objective.derivative_evals, 0,
        "the outer solver must not start without finite exact-seed evidence"
    );
    assert!(
        !objective.domain_open,
        "failed entry must roll back full state"
    );
    assert!(
        objective.checkpoint_domain_open.is_none(),
        "typed refusal must not leak an active waypoint transaction"
    );
}

#[test]
fn run_indefinite_analytic_seed_stays_on_arc() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_initial_rho(array![0.0])
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.0,
                gradient: array![0.0],
                hessian: HessianValue::Dense(array![[-1.0]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "indefinite seed geometry")
        .expect("indefinite analytic seed geometry should stay on the second-order plan");
    assert_eq!(result.plan_used.solver, Solver::Arc);
    assert_eq!(result.plan_used.hessian_source, HessianSource::Analytic);
}

#[test]
fn run_seed_materialization_failure_surfaces_arc_error_verbatim() {
    // Under the budget-bump retry ladder (commit c96c4233), an ARC
    // primary with `(Analytic, Analytic)` capability has zero degraded
    // fallbacks. A seed-materialization failure surfaces as `Err`
    // verbatim — there is no lateral demote to BFGS+BfgsApprox that
    // would silently discard the analytic outer Hessian. Materialization
    // failures are deterministic w.r.t. rho, so the budget-bump retry
    // ladder cannot rescue them; the operator returns the same Err on
    // every retry. Hence the runner returns the original Err.
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_initial_rho(array![0.0])
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.0,
                gradient: array![0.0],
                hessian: HessianValue::Operator(Arc::new(FailingSeedMaterializationOperator {
                    dim: 1,
                })),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let err = problem
        .run(&mut obj, "seed materialization failure")
        .expect_err(
            "ARC primary must surface the materialization failure verbatim — \
                 no lateral demote to BFGS+BfgsApprox",
        );
    let msg = err.to_string();
    assert!(
        msg.contains("seed materialization failed"),
        "error must propagate the underlying materialization message; got: {msg}"
    );
}

#[test]
fn run_nonconverged_arc_returns_typed_checkpoint_after_budget_retry_ladder() {
    // When an ARC primary exhausts its iteration budget, the runner
    // reseeds a fresh ARC attempt from the previous attempt's last
    // ρ and trust radius (up to two retries) and uncaps the inner
    // PIRLS cap for the resumed run via the InnerProgressFeedback
    // handle. Retries are gated on attempt-over-attempt `‖g‖`
    // halving so a deterministic-replay trajectory falls through.
    // The objective's analytic outer Hessian is preserved across
    // every attempt — no lateral demote to BFGS+BfgsApprox. After
    // the retries are exhausted (or the gate fires), the runner
    // returns typed non-convergence carrying the last rho checkpoint rather
    // than an `OuterResult` that could reach fitted-model assembly.
    //
    // We use `cost = x^4`, `grad = 4 x^3`, `hess = 12 x^2` from
    // `initial_rho = [5.0]` with `max_iter = 1`. Newton-style ARC
    // steps on x^4 contract the gradient by ~3× per attempt, so
    // the halving gate passes and both retries proceed; ARC still
    // cannot reach the optimum in three single-iter attempts.
    //
    // Arc + Gaussian has `effective_seed_budget == 1`, so `initial_rho = [5.0]`
    // is the sole authoritative start: no second budgeted seed lets the always-
    // injected neutral baseline `[0.0]` (the EXACT global minimum of x⁴, cost 0)
    // screen in and win — which would make ARC start already-optimal, report
    // converged in 0 iters, and never exercise the retry ladder this test drives.
    // (Any effective-budget-1 profile works; Arc Gaussian and GLM are both floored
    // to 1 now — see `effective_seed_budget`.)
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let (_d, session) = tmp_cache_session("nonconverged-arc-cache");
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_initial_rho(array![5.0])
        .with_max_iter(1)
        .with_cache_session(Arc::clone(&session));
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0].powi(4)),
        |_: &mut (), theta: &Array1<f64>| {
            let x = theta[0];
            Ok(OuterEval {
                cost: x.powi(4),
                gradient: array![4.0 * x.powi(3)],
                hessian: HessianValue::Dense(array![[12.0 * x.powi(2)]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let error = problem
        .run(&mut obj, "nonconverged arc should stay on arc")
        .expect_err("an exhausted ARC ladder must return typed non-convergence");
    let EstimationError::RemlDidNotConverge { rho_checkpoint, .. } = error else {
        panic!("expected typed REML non-convergence, got {error}");
    };
    // The ladder must have genuinely stepped away from neither the optimum
    // (rho=0, where x⁴ is stationary) nor stalled at the seed: ARC contracts
    // toward 0 but cannot reach it in the single-iter budget, so the reported
    // ρ is strictly between the optimum and the [5.0] start.
    assert!(
        rho_checkpoint[0].abs() > 1.0e-6 && rho_checkpoint[0] < 5.0,
        "the budget ladder must have made partial progress from the [5.0] seed \
         toward the x⁴ optimum without reaching it; got rho={:?}",
        rho_checkpoint
    );
}

#[test]
fn parsimonious_keep_best_breaks_laml_tie_toward_more_smoothing() {
    let plan = OuterPlan {
        solver: Solver::Bfgs,
        hessian_source: HessianSource::BfgsApprox,
    };
    let rho_dim = 2usize;

    // Two CONVERGED optima whose LAML values are a statistical tie (within the
    // relative band): a flexible (low-Σρ) basin scoring epsilon BETTER, and a
    // parsimonious (high-Σρ) basin. The parsimonious one must win the tie.
    let flexible = OuterResult::new(array![-3.0, -3.0], 100.0, 1, true, plan);
    let mut parsimonious = OuterResult::new(array![3.0, 3.0], 100.05, 1, true, plan);
    parsimonious.final_grad_norm = Some(0.0);

    // gap 0.05 <= 1e-3 * 100.05 (=0.10005) → tie band → prefer larger Σρ.
    assert!(candidate_improves_best_parsimonious(
        &parsimonious,
        Some(&flexible),
        rho_dim,
    ));
    // The flexible (lower-LAML) candidate must NOT displace the parsimonious
    // incumbent on a tie — the tie-break is asymmetric toward more smoothing.
    assert!(!candidate_improves_best_parsimonious(
        &flexible,
        Some(&parsimonious),
        rho_dim,
    ));

    // A DECISIVE LAML advantage for the flexible basin (gap far outside the
    // band) must still win: a fit that genuinely needs the flexibility is not
    // sacrificed to parsimony.
    let decisive_flexible = OuterResult::new(array![-3.0, -3.0], 90.0, 1, true, plan);
    assert!(candidate_improves_best_parsimonious(
        &decisive_flexible,
        Some(&parsimonious),
        rho_dim,
    ));
}

/// #1575: the parsimony-await second-seed waiver fires ONLY for a slot-0 result
/// that is curvature-pinned (score-relative |g| well inside the tie band) AND
/// well-penalized (every leading smoothing λ ≥ 1). Only analytically certified
/// candidates can reach this predicate.
#[test]
fn parsimony_second_seed_waived_only_for_sharp_well_penalized_optimum() {
    let plan = OuterPlan {
        solver: Solver::Arc,
        hessian_source: HessianSource::Analytic,
    };
    let rho_dim = 2usize;
    // `at_band(frac, score)` is the residual gradient sitting `frac`× the
    // score-relative sharpness band: frac<1 is inside (sharp), frac>1 outside.
    let at_band =
        |frac: f64, score: f64| PARSIMONY_SHARP_GRAD_REL_BAND * (1.0 + score.abs()) * frac;

    // The redundant case: converged, every smoothing ρ ≥ 0, residual gradient
    // two orders inside the tie band. Slot 1 would only re-derive this — waive.
    let mut redundant = OuterResult::new(array![3.0, 0.0], 1082.972, 6, true, plan);
    redundant.final_grad_norm = Some(at_band(0.01, 1082.972)); // 0.01× the band → sharp
    assert!(
        parsimony_second_seed_is_redundant(&redundant, rho_dim),
        "a converged, sharp, well-penalized slot-0 optimum makes the heavy seed redundant"
    );
    // Exactly on the band boundary still counts as sharp (≤, not <).
    let mut on_band = redundant.clone();
    on_band.final_grad_norm = Some(at_band(1.0, 1082.972));
    assert!(
        parsimony_second_seed_is_redundant(&on_band, rho_dim),
        "the score-relative band is inclusive at its edge"
    );

    // #1373 under-penalized basin: a smoothing λ < 1 (ρ < 0) is exactly the
    // overshoot the heavy seed guards against — never waive, even when sharp.
    let mut under_penalized = redundant.clone();
    under_penalized.rho = array![-0.5, 3.0];
    assert!(
        !parsimony_second_seed_is_redundant(&under_penalized, rho_dim),
        "a single under-penalized (ρ<0) coordinate keeps the parsimony seed (#1373)"
    );

    // Flat-valley non-sharp optimum: converged at a residual ABOVE the band, so
    // the parsimony tie-break could still slide ρ toward the heavier basin.
    let mut flat_valley = redundant.clone();
    flat_valley.final_grad_norm = Some(at_band(10.0, 1082.972)); // 10× the band → not sharp
    assert!(
        !parsimony_second_seed_is_redundant(&flat_valley, rho_dim),
        "a converged-but-flat optimum above the tie band keeps the parsimony seed"
    );

    // No measured gradient cannot certify sharpness.
    let mut no_grad = redundant.clone();
    no_grad.final_grad_norm = None;
    assert!(
        !parsimony_second_seed_is_redundant(&no_grad, rho_dim),
        "an unmeasured gradient cannot prove a curvature-pinned optimum"
    );

    // The score-relative band scales with |score|: a residual that is absolutely
    // large is still sharp when the LAML magnitude is large enough.
    let mut large_score = OuterResult::new(array![2.0, 2.0], 5.0e6, 6, true, plan);
    large_score.final_grad_norm = Some(40.0); // ≤ 1e-5·(1+5e6) = 50.00001
    assert!(
        parsimony_second_seed_is_redundant(&large_score, rho_dim),
        "sharpness is score-relative, not an absolute gradient threshold"
    );

    // Trailing auxiliary coordinates (e.g. a GAMLSS log-scale predictor) are not
    // smoothing parameters and must not block the waiver: only the leading
    // rho_dim coordinates are tested for λ ≥ 1.
    let mut with_aux = OuterResult::new(array![3.0, -7.0], 100.0, 6, true, plan);
    with_aux.final_grad_norm = Some(at_band(0.1, 100.0));
    assert!(
        parsimony_second_seed_is_redundant(&with_aux, 1),
        "a negative trailing auxiliary coordinate (ρ_dim=1) must not block the waiver"
    );

    // With no smoothing dimension the parsimony tie-break is a no-op.
    assert!(
        !parsimony_second_seed_is_redundant(&redundant, 0),
        "rho_dim=0 has no smoothing parameter for the parsimony seed to decide"
    );
}

#[test]
fn gaussian_multistart_compares_converged_seed_costs() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 2;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let started = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_max_iter(4);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(if theta[0] < -1.0 { 0.0 } else { 10.0 }),
        {
            let started = Arc::clone(&started);
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: if theta[0] < -1.0 { 0.0 } else { 10.0 },
                    gradient: array![0.0],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "Gaussian quality multistart")
        .expect("Gaussian multistart should compare both converged seeds");
    let starts = started.lock().unwrap();
    assert!(
        starts.len() >= 2,
        "Gaussian quality mode should not stop at the first converged seed"
    );
    assert!(
        result.rho[0] < -1.0,
        "lower-cost converged Gaussian seed should win"
    );
    assert_eq!(result.final_value, 0.0);
}

/// #1575 end-to-end wiring: drive the real multi-start loop with the
/// parsimonious (GeneralizedLinear) risk profile and `seed_budget = 2`. A slot-0
/// seed that CONVERGES to a sharp, well-penalized optimum (every smoothing
/// λ ≥ 1) must BREAK the multi-start after a single seed — the heavy slot-1 seed
/// is provably redundant. A slot-0 seed that converges to an UNDER-penalized
/// (ρ < 0) optimum is the #1373 overshoot regime, so the heavy seed must STILL
/// run. Counts genuine seed solves by intersecting the recorded solver evals
/// with the generated seed candidates (a seed-startup eval lands exactly on a
/// candidate; interior trial steps and the converged optimum do not).
#[test]
fn parsimony_multistart_breaks_after_sharp_well_penalized_first_seed() {
    fn seeds_run(center: f64) -> (usize, OuterResult) {
        let mut seed_config = gam_problem::SeedConfig::default();
        seed_config.seed_budget = 2;
        seed_config.risk_profile = gam_problem::SeedRiskProfile::GeneralizedLinear;
        let candidates: Vec<Array1<f64>> =
            crate::seeding::generate_rho_candidates(1, None, &seed_config);
        // The optimum must not coincide with any generated seed, so only true
        // seed-startup evals (which land exactly on a candidate) are counted.
        assert!(
            candidates.iter().all(|c| (c[0] - center).abs() > 1e-9),
            "test premise: the optimum {center} must not equal a generated seed"
        );
        let started = Arc::new(Mutex::new(Vec::<Array1<f64>>::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_max_iter(16);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), theta: &Array1<f64>| {
                let d = theta[0] - center;
                Ok(0.5 * d * d)
            },
            {
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    let d = theta[0] - center;
                    Ok(OuterEval {
                        cost: 0.5 * d * d,
                        gradient: array![d],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "parsimony multistart wiring")
            .expect("a strictly-convex quadratic outer objective converges");
        let starts = started.lock().unwrap();
        let mut origins: Vec<f64> = starts
            .iter()
            .filter(|t| candidates.iter().any(|c| (c[0] - t[0]).abs() < 1e-9))
            .map(|t| t[0])
            .collect();
        origins.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        origins.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        (origins.len(), result)
    }

    // Well-penalized minimum (ρ = 2.7 ≥ 0): slot 0 is sharp and every λ ≥ 1, so
    // the heavy seed is redundant — exactly ONE seed solves.
    let (well_penalized_seeds, well_result) = seeds_run(2.7);
    assert!(well_result.converged, "well-penalized fit converges");
    assert!(
        (well_result.rho[0] - 2.7).abs() < 1e-4,
        "publishes the slot-0 optimum, got {}",
        well_result.rho[0]
    );
    assert_eq!(
        well_penalized_seeds, 1,
        "a sharp, well-penalized slot-0 optimum must break the multi-start after one seed (#1575)"
    );

    // Under-penalized minimum (ρ = -2.7 < 0): the #1373 overshoot regime — the
    // heavy parsimony seed must still run.
    let (under_penalized_seeds, under_result) = seeds_run(-2.7);
    assert!(under_result.converged, "under-penalized fit converges");
    assert!(
        (under_result.rho[0] + 2.7).abs() < 1e-4,
        "publishes the slot-0 optimum, got {}",
        under_result.rho[0]
    );
    assert_eq!(
        under_penalized_seeds, 2,
        "an under-penalized (ρ<0) slot-0 optimum must keep the parsimony second seed (#1373)"
    );
}

#[test]
fn run_starts_solver_with_direct_startup_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), theta: &Array1<f64>| {
                calls.lock().unwrap().push("cost");
                Ok(theta[0] * theta[0])
            }
        },
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), theta: &Array1<f64>| {
                calls.lock().unwrap().push("eval");
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianValue::Dense(array![[2.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem
        .run(&mut obj, "solver should start from a direct startup eval")
        .expect("analytic plans should start with a direct full evaluation");
    let calls = calls.lock().unwrap();
    let first_eval_idx = calls
        .iter()
        .position(|call| *call == "eval")
        .expect("solver should eventually request a full eval");
    assert!(
        first_eval_idx == 0,
        "startup should not perform a separate cost-screening pass first: {calls:?}"
    );
}

#[test]
fn run_screening_reorders_expensive_generated_seeds_before_full_startup_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 2;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::GeneralizedLinear;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
        .last()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let started = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let valid_seed = valid_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(0.0)
                } else {
                    Ok(1000.0)
                }
            }
        },
        {
            let valid_seed = valid_seed.clone();
            let started = Arc::clone(&started);
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "screening should reorder expensive seeds")
        .expect("screened startup should reach the best generated seed");
    assert_eq!(result.rho, valid_seed);
    let started_snapshot: Vec<Array1<f64>> = started.lock().unwrap().clone();
    // The interior-extreme promotion (#1074/#1373/#1426) reserves slot 0 for the
    // most-flexible interior seed and slot 1 for the heaviest, so screening's
    // cost rank resumes at slot 2. (This promotion runs INSIDE
    // `rank_seeds_with_screening`, so its footprint at slots 0/1 — here the
    // generator's `[0.0]` and `[12.0]`, NOT the raw generator-first `[1.0]` — is
    // itself proof that screening ran.) The lowest-cost generated seed must lead
    // that reorderable tail: screening moved it ahead of the other equal-or-
    // higher-cost seeds it is allowed to reorder, exactly as the original "front"
    // assertion intended before the promotion reserved the first two slots.
    assert_eq!(
        started_snapshot.get(2).cloned(),
        Some(valid_seed),
        "screening should rank the lowest-cost seed at the head of the reorderable \
         tail (slots 0/1 are reserved for the promoted flexible/heaviest seeds); \
         started order was {started_snapshot:?}",
    );
    assert_eq!(screening_cap.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn thrown_screening_error_is_fatal_across_multistart_and_solver_plans() {
    const SENTINEL: &str = "fatal outer evaluation sentinel";

    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 2;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::GeneralizedLinear;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), _theta: &Array1<f64>| {
                calls.fetch_add(1, Ordering::Relaxed);
                Err(EstimationError::InvalidInput(SENTINEL.to_string()))
            }
        },
        |_: &mut (), _theta: &Array1<f64>| -> Result<OuterEval, EstimationError> {
            panic!("a fatal screening error must prevent full outer evaluation")
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );

    let error = match problem.run(&mut obj, "fatal screening error") {
        Err(error) => error,
        Ok(_) => panic!("a fatal screening error unexpectedly minted an outer result"),
    };
    assert!(error.is_fatal_outer_evaluation(), "{error}");
    assert!(error.to_string().contains(SENTINEL), "{error}");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "a thrown evaluator error must not be replayed across seeds, cap stages, or solver plans"
    );
    assert_eq!(
        screening_cap.load(Ordering::Relaxed),
        0,
        "fatal screening exit must restore the caller's inner-iteration cap"
    );
}

#[test]
fn initial_rho_with_single_seed_budget_skips_expensive_screening() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 1;
    // This test asserts the `initial_rho + seed_budget==1` screening-skip
    // (`explicit_initial_rho_owns_single_seed_budget`) fires. That skip keys off
    // the EFFECTIVE budget, not the requested one. Pin the fixture to Gaussian,
    // whose `effective_seed_budget` is 1, so the `seed_budget == 1` skip guard is
    // true and the skip is genuinely exercised — the behaviour this test guards.
    // (A profile whose effective budget were > 1 would make the guard false and
    // let screening run instead; Arc Gaussian and GLM are both floored to 1.)
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let screening_calls = Arc::new(AtomicUsize::new(0));
    let initial_seed = array![9.0];
    let started = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(initial_seed.clone())
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let screening_calls = Arc::clone(&screening_calls);
            move |_: &mut (), _theta: &Array1<f64>| {
                screening_calls.fetch_add(1, Ordering::Relaxed);
                Ok(0.0)
            }
        },
        {
            let started = Arc::clone(&started);
            let initial_seed = initial_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                if theta == initial_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "initial rho should be authoritative")
        .expect("initial-rho startup should not spend seed-screening solves");
    assert_eq!(result.rho, initial_seed);
    assert_eq!(
        screening_calls.load(Ordering::Relaxed),
        0,
        "explicit initial rho plus seed_budget=1 should skip screening"
    );
    assert_eq!(
        started.lock().unwrap().first().cloned(),
        Some(initial_seed),
        "solver should start from the explicit initial rho"
    );
    assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
}

#[test]
fn run_screening_reorders_bfgs_seeds_before_full_startup_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let initial_seed = array![9.0];
    let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let started = Arc::new(Mutex::new(Vec::new()));
    let screening_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(initial_seed)
        .with_screen_initial_rho(true)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let valid_seed = valid_seed.clone();
            let screening_calls = Arc::clone(&screening_calls);
            move |_: &mut (), theta: &Array1<f64>| {
                screening_calls.fetch_add(1, Ordering::Relaxed);
                if theta == valid_seed {
                    Ok(0.0)
                } else {
                    Ok(1000.0)
                }
            }
        },
        {
            let valid_seed = valid_seed.clone();
            let started = Arc::clone(&started);
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Unavailable,
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "BFGS screening should reorder expensive seeds")
        .expect("screened BFGS startup should reach the best generated seed");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(result.rho, valid_seed);
    let started_snapshot: Vec<Array1<f64>> = started.lock().unwrap().clone();
    // As in the analytic-gradient sibling test: the interior-extreme promotion
    // (#1074/#1373/#1426) reserves slot 0 (most-flexible interior seed) and slot 1
    // (heaviest interior seed — here the screened-in initial ρ=9.0), so screening's
    // cost rank resumes at slot 2. The lowest-cost generated seed must lead that
    // reorderable tail — screening moved it ahead of every other equal-or-higher-
    // cost seed it is allowed to reorder.
    assert_eq!(
        started_snapshot.get(2).cloned(),
        Some(valid_seed),
        "BFGS screening should rank the lowest-cost seed at the head of the \
         reorderable tail (slots 0/1 are reserved for the promoted flexible/heaviest \
         seeds); started order was {started_snapshot:?}",
    );
    assert!(
        screening_calls.load(Ordering::Relaxed) > 1,
        "BFGS seed screening should rank candidates with cost-only probes first",
    );
    assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
}

#[test]
fn screening_cap_survives_per_seed_reset_before_proxy_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 3;
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let proxy_saw_cap = Arc::new(AtomicBool::new(false));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_max_iter(1);
    let mut obj = problem.build_objective_with_screening_proxy(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[0].abs(),
                gradient: array![0.0],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
            Ok(OuterEval {
                cost: theta[0].abs(),
                gradient: array![0.0],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        {
            let screening_cap = Arc::clone(&screening_cap);
            Some(move |_: &mut ()| {
                screening_cap.store(0, Ordering::Relaxed);
            })
        },
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        {
            let screening_cap = Arc::clone(&screening_cap);
            let proxy_saw_cap = Arc::clone(&proxy_saw_cap);
            move |_: &mut (), theta: &Array1<f64>| {
                let cap = screening_cap.load(Ordering::Relaxed);
                if cap > 0 {
                    proxy_saw_cap.store(true, Ordering::Relaxed);
                    Ok(theta[0].abs())
                } else {
                    Err(EstimationError::RemlOptimizationFailed(
                        "screening proxy ran without an active cap".to_string(),
                    ))
                }
            }
        },
    );
    problem
        .run(&mut obj, "screening cap reset regression")
        .expect("screening cap should be restored after each per-seed reset");
    assert!(
        proxy_saw_cap.load(Ordering::Relaxed),
        "screening proxy should observe a nonzero cap"
    );
    assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
}

#[test]
fn rank_seeds_cascade_escalates_when_initial_cap_collapses_all() {
    // When every seed's cost is non-finite at the initial screening cap
    // we must NOT jump straight to a fully uncapped re-evaluation on
    // every seed (the original two-stage protocol). Instead the cap
    // should escalate geometrically (initial → 4× → 16× → uncapped),
    // exiting the moment any cap stage produces a finite cost. This
    // test forces a cost function that returns non-finite for cap < 12
    // and finite for cap ≥ 12, then asserts the cascade exits at the
    // 4× stage with a meaningful ranking — never reaching the uncapped
    // pass.
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.screen_max_inner_iterations = 3;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let initial_seed = array![5.0];
    let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let max_cap_seen = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(initial_seed.clone())
        .with_screen_initial_rho(true)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let screening_cap = Arc::clone(&screening_cap);
            let max_cap_seen = Arc::clone(&max_cap_seen);
            let valid_seed = valid_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                let cap = screening_cap.load(Ordering::Relaxed);
                max_cap_seen.fetch_max(cap, Ordering::Relaxed);
                // Mimic an inner solver that needs ≥ 12 iterations of
                // budget to certify a finite cost; below that it returns
                // a non-finite "could not converge" signal.
                if cap > 0 && cap < 12 {
                    return Ok(f64::NAN);
                }
                if theta == valid_seed {
                    Ok(0.0)
                } else {
                    Ok(1000.0)
                }
            }
        },
        {
            let valid_seed = valid_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem
        .run(&mut obj, "cascade should escalate")
        .expect("cascade should reach a finite cost at the 4× cap stage");
    // The cascade is [3, 12, 48, 0]; the 4× stage (cap=12) is the first
    // stage that produces a finite cost, so the cascade must exit there
    // and never escalate to 48 or to the uncapped (0) stage.
    let max_cap = max_cap_seen.load(Ordering::Relaxed);
    assert_eq!(
        max_cap, 12,
        "cascade should stop at the 4× cap stage; observed max cap = {max_cap}"
    );
    assert_eq!(
        screening_cap.load(Ordering::Relaxed),
        0,
        "screening cap must be restored to its previous value after cascade"
    );
}

#[test]
fn run_efs_skips_global_cost_screening() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 6;
    seed_config.seed_budget = 1;
    let screening_calls = Arc::new(AtomicUsize::new(0));
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(15)
        .with_gradient(Derivative::Unavailable)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let screening_calls = Arc::clone(&screening_calls);
            move |_: &mut (), _: &Array1<f64>| {
                screening_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(0.0)
            }
        },
        |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            Some(move |_: &mut (), theta: &Array1<f64>| {
                efs_calls.fetch_add(1, Ordering::Relaxed);
                Ok(EfsEval {
                    cost: 0.0,
                    steps: vec![0.0; theta.len()],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                    consecutive_restored_incumbents: None,
                })
            })
        },
    );
    problem
        .run(
            &mut obj,
            "EFS should not use a separate global cost-screening pass",
        )
        .expect("first generated EFS seed should be sufficient");
    assert_eq!(
        screening_calls.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "EFS startup should not call eval_cost just to screen seeds"
    );
    assert_eq!(
        efs_calls.load(Ordering::Relaxed),
        1,
        "the validated seed sample must be reused instead of paying the EFS inner solve twice"
    );
}

#[test]
fn run_efs_skips_invalid_leading_seed_without_spending_budget() {
    let generated =
        crate::seeding::generate_rho_candidates(15, None, &gam_problem::SeedConfig::default());
    let valid_seed = generated
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let invalid_seed = Array1::from_elem(15, 9.0);
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(15)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_initial_rho(invalid_seed)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
        None::<fn(&mut ())>,
        {
            let valid_seed = valid_seed.clone();
            Some(move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(EfsEval {
                        cost: 0.0,
                        steps: vec![0.0; theta.len()],
                        beta: None,
                        psi_gradient: None,
                        psi_indices: None,
                        inner_hessian_scale: None,
                        logdet_enclosure_gap: None,
                        consecutive_restored_incumbents: None,
                    })
                } else {
                    Err(EstimationError::RemlOptimizationFailed(
                        "invalid EFS seed".to_string(),
                    ))
                }
            })
        },
    );
    let result = problem
        .run(&mut obj, "efs generated seed should remain reachable")
        .expect("invalid startup seeds should not consume the only EFS seed slot");
    assert_eq!(result.rho, valid_seed);
    assert_eq!(result.plan_used.solver, Solver::Efs);
}

#[test]
fn run_efs_runtime_fallback_marker_degrades_to_bfgs_immediately() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 2;
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(12)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_initial_rho(Array1::zeros(12))
        .with_max_iter(5);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            Some(move |_: &mut (), _: &Array1<f64>| {
                efs_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Err(EstimationError::RemlOptimizationFailed(format!(
                    "{} synthetic runtime escape hatch",
                    EFS_FIRST_ORDER_FALLBACK_MARKER,
                )))
            })
        },
    );
    let result = problem
        .run(&mut obj, "efs runtime fallback marker")
        .expect("runtime EFS escape hatch should degrade to BFGS");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(
        efs_calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "runtime fallback marker should abort the EFS attempt immediately"
    );
}

#[test]
fn run_rejects_invalid_theta_layout() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_psi_dim(2)
        .with_initial_rho(Array1::zeros(1))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        |_: &mut (), _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let err = problem
        .run(&mut obj, "test invalid layout")
        .expect_err("invalid theta layout should fail cleanly");
    assert!(
        err.to_string().contains("invalid outer theta layout"),
        "unexpected error: {err}"
    );
}

#[test]
fn effective_seed_budget_caps_expensive_solver_retries() {
    assert_eq!(
        effective_seed_budget(
            4,
            Solver::Efs,
            gam_problem::SeedRiskProfile::GeneralizedLinear,
        ),
        1
    );
    assert_eq!(
        effective_seed_budget(4, Solver::HybridEfs, gam_problem::SeedRiskProfile::Survival,),
        1
    );
    // #1575/#1074/#1426: Arc + GeneralizedLinear is floored to a single seed too
    // (the initial.sp seed reaches the heavily-penalized GLM basin), regardless of
    // the requested budget.
    assert_eq!(
        effective_seed_budget(
            3,
            Solver::Arc,
            gam_problem::SeedRiskProfile::GeneralizedLinear,
        ),
        1
    );
    assert_eq!(
        effective_seed_budget(
            1,
            Solver::Arc,
            gam_problem::SeedRiskProfile::GeneralizedLinear,
        ),
        1
    );
    assert_eq!(
        effective_seed_budget(3, Solver::Arc, gam_problem::SeedRiskProfile::Survival,),
        1
    );
    // #1689/#1757: Arc + Gaussian is floored to a single seed (the analytic
    // initial.sp seed lands the correct basin, so the second full outer solve is
    // redundant), regardless of the requested budget.
    assert_eq!(
        effective_seed_budget(3, Solver::Arc, gam_problem::SeedRiskProfile::Gaussian),
        1
    );
    assert_eq!(
        effective_seed_budget(3, Solver::Bfgs, gam_problem::SeedRiskProfile::Survival,),
        3
    );
}

#[test]
fn run_arc_projects_seed_before_seed_validation_eval() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 1;
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_bounds(array![0.0], array![1.0])
        .with_initial_rho(array![2.0])
        .with_seed_config(seed_config)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
        {
            let seen = Arc::clone(&seen);
            move |_: &mut (), theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 0.25).powi(2),
                    gradient: array![2.0 * (theta[0] - 0.25)],
                    hessian: HessianValue::Dense(array![[2.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem
        .run(&mut obj, "arc seed projection")
        .expect("arc should evaluate the projected seed");
    assert_eq!(
        seen.lock().unwrap().first().cloned(),
        Some(array![1.0]),
        "Arc must project the seed before validating the initial sample",
    );
}

#[test]
fn run_bfgs_projects_seed_before_seed_validation_eval() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 1;
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![0.0], array![1.0])
        .with_initial_rho(array![2.0])
        .with_seed_config(seed_config)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
        {
            let seen = Arc::clone(&seen);
            move |_: &mut (), theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 0.25).powi(2),
                    gradient: array![2.0 * (theta[0] - 0.25)],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem
        .run(&mut obj, "bfgs seed projection")
        .expect("BFGS should evaluate the projected seed");
    assert_eq!(
        seen.lock().unwrap().first().cloned(),
        Some(array![1.0]),
        "BFGS must project the seed before validating the initial sample",
    );
}

fn tmp_cache_session(label: &str) -> (tempfile::TempDir, Arc<CacheSession>) {
    let dir = tempfile::tempdir().unwrap();
    let store = gam_runtime::warm_start::WarmStartStore::open(
        dir.path().to_path_buf(),
        gam_runtime::warm_start::StoreOptions {
            size_budget_bytes: 1024 * 1024,
            ttl: std::time::Duration::from_secs(60),
        },
    )
    .unwrap();
    let mut fp = gam_runtime::warm_start::Fingerprinter::new();
    fp.absorb_str(b"outer-test", label);
    let key = fp.finalize();
    (dir, Arc::new(CacheSession::open(store, key)))
}

#[test]
fn checkpointing_objective_persists_finite_evals() {
    let (_d, session) = tmp_cache_session("ckpt-persist");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| Err(EstimationError::InvalidInput("eval not used".into())),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
    // Initial: nothing on disk.
    assert!(session.try_load().is_none());
    // First eval persists.
    let v0 = wrapped.eval_cost(&array![3.0]).unwrap();
    assert!((v0 - 9.0).abs() < 1e-12);
    let on_disk = session.try_load().expect("first eval should checkpoint");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert!((payload.cost - 9.0).abs() < 1e-12);
    assert_eq!(payload.rho, vec![3.0]);
    // Strictly improving eval must bypass the 2-second rate limit.
    let v1 = wrapped.eval_cost(&array![0.5]).unwrap();
    assert!((v1 - 0.25).abs() < 1e-12);
    let on_disk = session
        .try_load()
        .expect("improving eval should checkpoint");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert!((payload.cost - 0.25).abs() < 1e-12);
    assert_eq!(payload.rho, vec![0.5]);
    // Non-finite values must not corrupt the on-disk best-known iterate.
    let v_inf = wrapped.eval_cost(&array![f64::NAN]);
    match v_inf {
        Ok(value) => assert!(!value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }
    let on_disk = session.try_load().expect("prior best preserved");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert!((payload.cost - 0.25).abs() < 1e-12);
}

#[test]
fn checkpointing_objective_never_persists_reactive_initialization_waypoints() {
    let (_d, session) = tmp_cache_session("ckpt-reactive-phase");
    let mut inner =
        ReactiveDomainObjective::new(0.125, ReactiveDomainMode::FiniteAtColdSeed);
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());

    wrapped
        .begin_reactive_domain_waypoint()
        .expect("begin typed reactive waypoint");
    let entry_cost = wrapped
        .eval_cost(&array![2.5])
        .expect("finite initialization waypoint");
    assert!(entry_cost.is_finite());
    assert!(
        session.try_load().is_none(),
        "a reactive initialization waypoint is not an outer candidate and must not become a restart seed",
    );
    wrapped
        .commit_reactive_domain_waypoint(&array![2.5])
        .expect("commit typed reactive waypoint");

    wrapped
        .eval_cost(&array![0.125])
        .expect("literal target evaluation");
    let payload = decode_iterate(
        &session
            .try_load()
            .expect("literal target should checkpoint")
            .payload,
        1,
    )
    .expect("literal checkpoint decodes");
    assert_eq!(payload.rho, vec![0.125]);
}

#[test]
fn checkpointing_objective_rejects_wrong_dim_on_decode() {
    // A payload from a 3-dim fit is invalid input for a 5-dim resume.
    let bytes = encode_iterate(&array![1.0, 2.0, 3.0], None, None, 0.5, 0).expect("encode");
    assert!(decode_iterate(&bytes, 3).is_some());
    assert!(decode_iterate(&bytes, 5).is_none());
}

#[test]
fn schema_two_iterate_is_rejected_after_hessian_provenance_break_2253() {
    let obsolete = serde_json::json!({
        "schema": 2,
        "rho": [0.5],
        "beta": [1.0],
        "hessian": [4.0],
        "hessian_dim": 1,
        "cost": 2.0,
        "eval_id": 7
    });
    let bytes = serde_json::to_vec(&obsolete).expect("serialize obsolete payload");
    assert!(decode_iterate(&bytes, 1).is_none());
}

#[test]
fn transferred_hessian_requires_current_analytic_capability_2253() {
    let hessian = array![[2.0_f64, 0.0], [0.0, 3.0]];
    assert!(
        eligible_transferred_outer_hessian(
            Some(&hessian),
            DeclaredHessianForm::Unavailable,
            2,
        )
        .is_none()
    );
    assert!(
        eligible_transferred_outer_hessian(Some(&hessian), DeclaredHessianForm::Dense, 2)
            .is_some()
    );
}

#[test]
fn iterate_payload_round_trips_beta() {
    // Every persisted entry that comes with an inner-β hint round-trips
    // (ρ, β) together — that pair lets a resume open inner PIRLS in the
    // basin of quadratic attraction regardless of where ρ sits.
    let rho = array![10.0, -10.0, 5.0];
    let beta = array![0.12, -0.34, 0.56, 7.89];
    let bytes = encode_iterate(&rho, Some(&beta), None, 1.0, 7).expect("encode");
    let decoded = decode_iterate(&bytes, rho.len()).expect("decode");
    assert_eq!(decoded.rho, rho.to_vec());
    assert_eq!(decoded.beta, beta.to_vec());
    // ρ-only writes (β = None) still encode but with an empty beta slot.
    let ro_bytes = encode_iterate(&rho, None, None, 1.0, 7).expect("encode-rho-only");
    let ro = decode_iterate(&ro_bytes, rho.len()).expect("decode-rho-only");
    assert!(ro.beta.is_empty());
}

#[test]
fn iterate_payload_round_trips_converged_outer_hessian() {
    // The converged outer curvature persists alongside (ρ, β) so the next
    // structurally-matching fit can seed BFGS with H⁻¹ for a quasi-Newton
    // first step instead of restarting from an unscaled identity metric.
    let rho = array![0.5, -1.5];
    let h = array![[4.0, 1.0], [1.0, 3.0]];
    let bytes = encode_iterate(&rho, None, Some(&h), 1.0, 0).expect("encode");
    let decoded = decode_iterate(&bytes, rho.len()).expect("decode");
    assert_eq!(decoded.hessian_dim, 2);
    assert_eq!(decoded.hessian, vec![4.0, 1.0, 1.0, 3.0]);

    // The classifier surfaces the square Hessian as a (dim, flat) pair on the
    // Seed decision so the resume path can reconstruct and invert it.
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload: bytes,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    let CacheSeedDecision::Seed {
        hessian: decoded_h, ..
    } = classify_cache_entry_for_outer(&loaded, 2)
    else {
        panic!("expected Seed decision");
    };
    let (dim, flat) = decoded_h.expect("Seed must carry the persisted Hessian");
    assert_eq!(dim, 2);
    assert_eq!(flat, vec![4.0, 1.0, 1.0, 3.0]);
}

#[test]
fn iterate_payload_scrubs_non_finite_or_non_square_hessian() {
    // A malformed curvature must never reach the warm-start metric: a
    // non-square or non-finite Hessian is scrubbed to "no Hessian" while the
    // ρ/β seed is preserved, so the resume degrades to the scalar metric
    // rather than corrupting the first BFGS step.
    let rho = array![0.0];
    let nan_h = array![[f64::NAN]];
    let bytes = encode_iterate(&rho, None, Some(&nan_h), 1.0, 0).expect("encode");
    // encode_iterate itself drops a non-finite Hessian before serialization.
    let decoded = decode_iterate(&bytes, 1).expect("decode");
    assert_eq!(decoded.hessian_dim, 0);
    assert!(decoded.hessian.is_empty());
}

#[test]
fn note_persists_inner_beta_hint_from_eval() {
    // Write-side proof of the principled fix: when the inner solver
    // surfaces β via OuterEval::inner_beta_hint, CheckpointingObjective
    // captures it on every accepted eval AND exposes it for finalize.
    let (_d, session) = tmp_cache_session("note-persists-beta");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[0] * theta[0],
                gradient: array![2.0 * theta[0]],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![1.5, 2.5, 3.5]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
    let eval = wrapped.eval(&array![0.5]).expect("eval ok");
    assert!((eval.cost - 0.25).abs() < 1e-12);
    let on_disk = session
        .try_load()
        .expect("eval with finite β must persist a (ρ,β) checkpoint");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert_eq!(payload.beta, vec![1.5, 2.5, 3.5]);
    let captured = wrapped.last_inner_beta().expect("β was captured");
    assert_eq!(captured.to_vec(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn note_rejects_nonfinite_inner_beta() {
    // A divergent inner state must NOT poison the cache: persisting a
    // non-finite β would re-create the inner-PIRLS budget-exhaustion
    // failure mode at boundary ρ where the cached β is supposed to
    // place the resume inside Newton's quadratic basin.
    let (_d, session) = tmp_cache_session("note-rejects-bad-beta");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[0] * theta[0],
                gradient: array![2.0 * theta[0]],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![f64::NAN, 0.5]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
    let eval = wrapped.eval(&array![0.5]).expect("eval ok");
    assert!((eval.cost - 0.25).abs() < 1e-12);
    assert!(
        session.try_load().is_none(),
        "non-finite β must abort the checkpoint write, not poison the cache",
    );
    assert!(
        wrapped.last_inner_beta().is_none(),
        "non-finite β must not be exposed via last_inner_beta()",
    );
}

#[test]
fn classify_extracts_beta_from_v2_payload() {
    // The classifier propagates `beta` from the v2 payload onto its
    // Seed/ExactFinal decisions so the dispatcher can hand it to
    // OuterObjective::seed_inner_state. Without this, the (ρ, β) payload
    // would write β but never resurface it on resume.
    let rho = array![1.0, 2.0];
    let beta = array![10.0, 20.0, 30.0];
    let payload = encode_iterate(&rho, Some(&beta), None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    let CacheSeedDecision::Seed {
        beta: decoded_beta, ..
    } = classify_cache_entry_for_outer(&loaded, 2)
    else {
        panic!("expected Seed decision");
    };
    assert_eq!(decoded_beta, beta.to_vec());

    // ρ-only payload (legacy or family-without-β) decodes to empty beta.
    let payload = encode_iterate(&rho, None, None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    let CacheSeedDecision::Seed {
        beta: decoded_beta, ..
    } = classify_cache_entry_for_outer(&loaded, 2)
    else {
        panic!("expected Seed decision");
    };
    assert!(
        decoded_beta.is_empty(),
        "ρ-only payload must produce an empty beta so the dispatcher skips seed_inner_state"
    );
}

#[test]
fn cached_beta_binds_only_to_its_bitwise_matching_generated_seed() {
    struct ReplayObj {
        installed: Option<Array1<f64>>,
        seed_calls: usize,
    }
    impl OuterObjective for ReplayObj {
        fn capability(&self) -> OuterCapability {
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: 2,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }

        fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
            Ok(theta.dot(theta))
        }

        fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: 2.0 * theta,
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        }

        fn reset(&mut self) {
            self.installed = None;
        }

        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            self.seed_calls += 1;
            self.installed = Some(beta.clone());
            Ok(SeedOutcome::Installed)
        }
    }

    let owner = array![1.0, 2.0];
    let beta = array![7.0, 8.0, 9.0];
    let config = OuterConfig {
        initial_inner_seed: Some(BoundInnerSeed {
            theta: owner.clone(),
            beta: beta.clone(),
        }),
        ..OuterConfig::default()
    };
    let one_ulp_away = array![f64::from_bits(1.0_f64.to_bits() + 1), 2.0];
    let candidates = [one_ulp_away, owner, array![-1.0, 2.0]];
    let mut objective = ReplayObj {
        installed: None,
        seed_calls: 0,
    };

    for (index, candidate) in candidates.iter().enumerate() {
        objective.reset();
        install_matching_initial_inner_seed(
            &mut objective,
            &config,
            candidate,
            "bitwise seed ownership",
        )
        .expect("cache replay decision");
        if index == 1 {
            assert_eq!(objective.installed, Some(beta.clone()));
        } else {
            assert!(
                objective.installed.is_none(),
                "cached beta leaked into non-owning generated seed {index}",
            );
        }
    }
    assert_eq!(objective.seed_calls, 1);
}

#[test]
fn run_calls_seed_inner_state_with_cached_beta() {
    // End-to-end read-side wiring: a cache hit carrying β must call
    // OuterObjective::seed_inner_state(&beta) *before* the first BFGS
    // eval. We verify this by routing through a custom OuterObjective
    // that records the β it was seeded with.
    struct RecordingObj {
        seeded: Arc<Mutex<Option<Array1<f64>>>>,
        first_eval_seeded: Arc<Mutex<Option<Array1<f64>>>>,
        eval_count: Arc<Mutex<usize>>,
    }
    impl OuterObjective for RecordingObj {
        fn capability(&self) -> OuterCapability {
            // Analytic gradient AND analytic Hessian so the planner picks
            // the same Hessian-bearing path a real fit takes; using
            // Unavailable here would test a degenerate plan.
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Dense,
                n_params: 2,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }
        fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
            Ok(theta.dot(theta))
        }
        fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            let mut eval_count = self.eval_count.lock().unwrap();
            if *eval_count == 0 {
                *self.first_eval_seeded.lock().unwrap() = self.seeded.lock().unwrap().clone();
            }
            *eval_count += 1;
            // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: 2.0 * theta,
                hessian: HessianValue::Dense(2.0 * Array2::<f64>::eye(theta.len())),
                inner_beta_hint: None,
            })
        }
        fn reset(&mut self) {
            *self.seeded.lock().unwrap() = None;
        }
        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            *self.seeded.lock().unwrap() = Some(beta.clone());
            Ok(SeedOutcome::Installed)
        }
    }

    let (_d, session) = tmp_cache_session("seed-inner-state-call");
    let bytes = encode_iterate(
        &array![1.0, 2.0],
        Some(&array![7.5, 8.5, 9.5]),
        None,
        5.0,
        3,
    )
    .expect("encode");
    session.checkpoint(&bytes, Some(5.0), Some(3));

    let seeded: Arc<Mutex<Option<Array1<f64>>>> = Arc::new(Mutex::new(None));
    let first_eval_seeded: Arc<Mutex<Option<Array1<f64>>>> = Arc::new(Mutex::new(None));
    let eval_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let mut obj = RecordingObj {
        seeded: Arc::clone(&seeded),
        first_eval_seeded: Arc::clone(&first_eval_seeded),
        eval_count: Arc::clone(&eval_count),
    };

    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_max_iter(1)
        .with_cache_session(Arc::clone(&session));
    match problem.run(&mut obj, "seed-inner-state-call") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }

    let observed = first_eval_seeded.lock().unwrap().clone();
    assert_eq!(
        observed,
        Some(array![7.5, 8.5, 9.5]),
        "the first exact evaluation after the per-seed reset must observe the cached β",
    );
}

#[test]
fn run_skips_seed_inner_state_when_payload_has_no_beta() {
    // Symmetric guard: a ρ-only warm-start entry must NOT invoke
    // seed_inner_state — calling it with an empty / zero / garbage β
    // would silently degrade a family that has a non-trivial inner
    // default into one started at zeros.
    struct CountingObj {
        seed_calls: Arc<Mutex<usize>>,
    }
    impl OuterObjective for CountingObj {
        fn capability(&self) -> OuterCapability {
            // Analytic gradient AND analytic Hessian so the planner picks
            // the same Hessian-bearing path a real fit takes; using
            // Unavailable here would test a degenerate plan.
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Dense,
                n_params: 2,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }
        fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
            Ok(theta.dot(theta))
        }
        fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: 2.0 * theta,
                hessian: HessianValue::Dense(2.0 * Array2::<f64>::eye(theta.len())),
                inner_beta_hint: None,
            })
        }
        fn reset(&mut self) {}
        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            *self.seed_calls.lock().unwrap() += beta.len().max(1);
            Ok(SeedOutcome::Installed)
        }
    }

    let (_d, session) = tmp_cache_session("seed-inner-state-skip");
    // ρ-only payload — no β.
    let bytes = encode_iterate(&array![1.0, 2.0], None, None, 5.0, 3).expect("encode");
    session.checkpoint(&bytes, Some(5.0), Some(3));

    let seed_calls: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let mut obj = CountingObj {
        seed_calls: Arc::clone(&seed_calls),
    };

    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_max_iter(1)
        .with_cache_session(Arc::clone(&session));
    match problem.run(&mut obj, "seed-inner-state-skip") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }

    assert_eq!(
        *seed_calls.lock().unwrap(),
        0,
        "seed_inner_state must not fire when the cached payload carries no β",
    );
}

#[test]
fn cache_entry_classifier_honors_finite_seeds_regardless_of_saturation() {
    // The classifier no longer reshapes ρ based on shape. Any finite,
    // correctly-dimensioned payload is honored as the next run's seed.
    // Boundary-saturated entries written under the v2 (ρ, β) invariant
    // are a *legitimate* finding — the smoothness wants to be near-null
    // — and the persisted β puts the next inner solve at zero-gradient,
    // making the cold-β failure mode impossible to re-create from cache.
    for rho_seed in [array![9.0, 0.0], array![10.0, -10.0], array![-10.0, 10.0]] {
        let payload = encode_iterate(&rho_seed, None, None, 1.0, 0).expect("encode");
        let loaded = gam_runtime::warm_start::LoadedEntry {
            entry: gam_runtime::warm_start::WarmStartEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: gam_runtime::warm_start::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: gam_runtime::warm_start::LoadSource::Preloaded,
        };

        assert!(cache_entry_would_help_outer(&loaded, 2));
        let CacheSeedDecision::Seed { rho, .. } = classify_cache_entry_for_outer(&loaded, 2) else {
            panic!(
                "finite seed {:?} must be honored unchanged; the read-side clamp / \
                     all-saturated-discard branches were band-aids over the missing β cache",
                rho_seed
            );
        };
        assert_eq!(rho, rho_seed, "ρ must round-trip without reshaping");
    }
}

#[test]
fn cache_entry_classifier_rejects_only_structural_failures() {
    // Only structural failures discard: payload shape (wrong rho_dim,
    // non-finite payload internals → decode None → "payload-shape-mismatch")
    // and non-finite warm-start metadata → "non-finite-payload". Saturation
    // and β presence are NOT discards here: saturation is honored, and
    // ρ-only payloads decode cleanly with an empty β slot.

    // Non-finite metadata objective: decode succeeds (finite payload
    // cost), but the entry-level objective is NaN — discard as
    // non-finite-payload.
    let payload = encode_iterate(&array![0.5, 0.5], None, None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(f64::NAN),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    assert!(matches!(
        classify_cache_entry_for_outer(&loaded, 2),
        CacheSeedDecision::Discard {
            reason: "non-finite-payload",
            ..
        }
    ));

    // Dimension mismatch: 2-D payload viewed as a 3-D problem → decode
    // rejects shape → "payload-shape-mismatch".
    let payload = encode_iterate(&array![0.5, 0.5], None, None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    assert!(matches!(
        classify_cache_entry_for_outer(&loaded, 3),
        CacheSeedDecision::Discard {
            reason: "payload-shape-mismatch",
            ..
        }
    ));
}

#[test]
fn exact_final_warm_start_hit_is_helpful_even_at_boundary() {
    let payload = encode_iterate(&array![10.0, -10.0], None, None, 1.0, 3).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(3),
            kind: gam_runtime::warm_start::EntryKind::Final,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Exact,
    };

    assert!(cache_entry_would_help_outer(&loaded, 2));
    assert!(matches!(
        classify_cache_entry_for_outer(&loaded, 2),
        CacheSeedDecision::ExactFinal { iterations: 3, .. }
    ));
}

#[test]
fn checkpointing_objective_mirrors_checkpoints() {
    let (_primary_dir, primary) = tmp_cache_session("ckpt-primary");
    let (_mirror_dir, mirror) = tmp_cache_session("ckpt-mirror");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| Err(EstimationError::InvalidInput("eval not used".into())),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped =
        CheckpointingObjective::new(&mut inner, Arc::clone(&primary), vec![Arc::clone(&mirror)]);

    let value = wrapped.eval_cost(&array![4.0]).unwrap();
    assert_eq!(value, 16.0);

    let primary_payload =
        decode_iterate(&primary.try_load().expect("primary checkpoint").payload, 1)
            .expect("primary decode");
    let mirror_payload = decode_iterate(&mirror.try_load().expect("mirror checkpoint").payload, 1)
        .expect("mirror decode");
    assert_eq!(primary_payload.rho, vec![4.0]);
    assert_eq!(mirror_payload.rho, vec![4.0]);
    assert_eq!(primary_payload.cost, mirror_payload.cost);
}

#[test]
fn cached_rho_is_prepended_as_first_seed() {
    // Whitebox: pre-seed the session with a known iterate, then run
    // an OuterProblem with a deliberately-different `initial_rho`.
    // The runner must visit the cached rho before the configured
    // `initial_rho` because `try_load` overrode it.
    let (_d, session) = tmp_cache_session("seed-prepend");
    // Hand-write the cached checkpoint: rho = [2.5], cost = 0.25.
    // Final exact hits return immediately; checkpoints still exercise the
    // regular seed-prepend path.
    let payload = encode_iterate(&array![2.5], None, None, 0.25, 0).expect("encode");
    session.checkpoint(&payload, Some(0.25), Some(0));
    assert!(
        session.try_load().is_some(),
        "precondition: cache populated"
    );

    let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
    // A gradient-bearing BFGS problem. Bounds must contain the cached rho
    // so the projector doesn't snap it away.
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-5.0], array![5.0])
        .with_initial_rho(array![-3.0]) // deliberately not 2.5
        .with_max_iter(8)
        .with_cache_session(Arc::clone(&session));
    let mut obj = problem.build_objective(
        seen.clone(),
        |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            seen.lock().unwrap().push(theta.clone());
            Ok((theta[0] - 2.5).powi(2))
        },
        {
            let seen = seen.clone();
            move |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                // Record full-eval points too: startup now performs a DIRECT full
                // evaluation at the seed (the `run_starts_solver_with_direct_startup_eval`
                // contract) routed through THIS eval path, not a separate cost-screening
                // pass. The cached rho is consumed by that startup eval, so a recorder
                // that watched only the cost closure never saw the exact cached ρ — it
                // only caught the later line-search cost probes clustered near it.
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 2.5).powi(2),
                    gradient: array![2.0 * (theta[0] - 2.5)],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
        None::<
            fn(&mut Arc<Mutex<Vec<Array1<f64>>>>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    );
    match problem.run(&mut obj, "seed-prepend") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }
    // The cached rho (2.5) must appear in the eval trace, and it must
    // appear no later than the configured initial_rho (−3.0). Both
    // are inside the bounds so the projector cannot rewrite them.
    let evals = seen.lock().unwrap();
    let pos_cached = evals.iter().position(|r| (r[0] - 2.5).abs() < 1e-9);
    let pos_initial = evals.iter().position(|r| (r[0] + 3.0).abs() < 1e-9);
    assert!(
        pos_cached.is_some(),
        "cached rho must be evaluated; saw {:?}",
        *evals
    );
    if let (Some(c), Some(i)) = (pos_cached, pos_initial) {
        assert!(
            c <= i,
            "cached rho (idx {c}) must precede initial_rho (idx {i})",
        );
    }
}

#[test]
fn all_saturated_cached_rho_is_honored_as_seed() {
    // Inverse of the prior `all_saturated_cached_rho_is_discarded_before_seed_validation`
    // test. Under v1 the cache stored ρ-only, so resuming at boundary ρ
    // forced PIRLS to cold-start β against a Hessian with condition
    // number `≈ e^{2·rho_bound}` — Newton degraded to O(1/k) descent
    // that exhausted the cycle budget. The "discard if all-saturated"
    // branch was a read-side band-aid; it suppressed a legitimate
    // resume signal in exchange for tolerating the broken contract.
    //
    // Under v2 the iterate payload carries (ρ, β). When β is persisted
    // alongside boundary ρ the next inner solve opens at zero gradient,
    // and the conditioning is no longer a barrier. Therefore the
    // classifier no longer reshapes ρ based on saturation: every
    // finite, correctly-dimensioned entry is used as the seed. This
    // test pins that contract.
    let (_d, session) = tmp_cache_session("all-saturated-honored");
    let payload = encode_iterate(&array![10.0, -10.0], None, None, 1.0, 0).expect("encode");
    session.checkpoint(&payload, Some(1.0), Some(0));
    assert!(
        session.try_load().is_some(),
        "precondition: cache populated"
    );

    let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_initial_rho(array![0.0, 0.0])
        .with_rho_bound(10.0)
        .with_max_iter(1)
        .with_cache_session(Arc::clone(&session));

    let mut obj = problem.build_objective(
        seen.clone(),
        |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| Ok(theta.dot(theta)),
        |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            seen.lock().unwrap().push(theta.clone());
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
        None::<
            fn(&mut Arc<Mutex<Vec<Array1<f64>>>>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    );

    match problem.run(&mut obj, "all-saturated-honored") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }
    let evals = seen.lock().unwrap();
    assert!(
        evals.iter().any(|rho| rho == array![10.0, -10.0]),
        "cached saturated ρ must be evaluated unchanged under v2 (ρ, β) invariant; saw {:?}",
        *evals
    );
}

#[test]
fn exact_final_cache_hit_skips_outer_validation() {
    let (_d, session) = tmp_cache_session("final-skip");
    let payload = encode_iterate(&array![2.5], None, None, 0.25, 7).expect("encode");
    session.finalize(&payload, Some(0.25), Some(7));

    let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
    // The exact final cache hit short-circuits before any solver runs, so
    // the declared derivatives only need to make a well-formed plan.
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-5.0], array![5.0])
        .with_initial_rho(array![-3.0])
        .with_max_iter(8)
        .with_cache_session(Arc::clone(&session));
    let mut obj = problem.build_objective(
        seen.clone(),
        |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            seen.lock().unwrap().push(theta.clone());
            Ok((theta[0] - 2.5).powi(2))
        },
        |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: (theta[0] - 2.5).powi(2),
                gradient: array![2.0 * (theta[0] - 2.5)],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
        None::<
            fn(&mut Arc<Mutex<Vec<Array1<f64>>>>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    );

    let result = problem
        .run(&mut obj, "final-skip")
        .expect("final exact hit should return cached outer result");
    assert_eq!(result.rho, array![2.5]);
    assert_eq!(result.final_value, 0.25);
    assert_eq!(result.iterations, 7);
    assert!(result.converged);
    assert!(
        seen.lock().unwrap().is_empty(),
        "exact final hit should not evaluate the outer objective"
    );
}
