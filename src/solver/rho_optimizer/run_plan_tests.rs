use super::*;
use crate::model_types::result_types::CERTIFICATE_Z_GATE;
use ::opt::FixedPointObjective;
use ndarray::array;
use std::sync::atomic::{AtomicUsize, Ordering};
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
        .with_seed_config(crate::seeding::SeedConfig {
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
                hessian: HessianResult::Unavailable,
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
        cert.first_order_consistent(),
        "consistent value/gradient paths flagged as desynced: {}",
        cert.summary(),
    );
    assert!(
        cert.lambdas_railed.is_empty(),
        "interior optimum reported railed λ: {}",
        cert.summary(),
    );
    assert!(cert.fd_step > 0.0 && cert.fd_error > 0.0);
}

#[test]
fn rho_uncertainty_diagnostic_does_not_change_outer_solution() {
    let center = array![0.25];
    let seed_config = crate::seeding::SeedConfig {
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
                    hessian: HessianResult::Analytic(array![[1.0]]),
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
                    hessian: HessianResult::Analytic(array![[1.0]]),
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
/// The optimizer happily converges where the WRONG gradient vanishes;
/// the certificate's FD of the actual value path must expose it.
#[test]
fn certificate_flags_value_gradient_desync() {
    let value_center = array![0.0, 0.0];
    let wrong_center = array![3.0, -2.0];
    let wrong_center_for_eval = wrong_center.clone();
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![1.0, 1.0])
        .with_seed_config(crate::seeding::SeedConfig {
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
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "certificate desynced quadratic")
        .expect("desynced quadratic still returns a result");
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("gradient-based solve must ship a certificate");
    // At wrong_center the analytic slope is ~0 but the true value path
    // slopes by v·(wrong_center − value_center) along the audit
    // direction. Guard the assertion on that projection being visible
    // (the deterministic direction is not axis-aligned, so it is).
    assert!(
        cert.fd_directional.abs() > 1e-3,
        "audit direction nearly orthogonal to the desync displacement: {}",
        cert.summary(),
    );
    assert!(
        !cert.first_order_consistent(),
        "value↔gradient desync NOT flagged: {}",
        cert.summary(),
    );
    assert!(cert.agreement_z > CERTIFICATE_Z_GATE);
}

#[test]
fn certificate_audit_direction_is_deterministic_and_context_sensitive() {
    let theta = array![1.5, -0.25, 7.0];
    let a = certificate_audit_direction(&theta, "ctx-one");
    let b = certificate_audit_direction(&theta, "ctx-one");
    assert_eq!(a, b, "same fingerprint must give the same direction");
    let c = certificate_audit_direction(&theta, "ctx-two");
    assert!(
        (&a - &c).iter().any(|d| d.abs() > 1e-12),
        "different context must give a different direction",
    );
    assert!((a.dot(&a).sqrt() - 1.0).abs() < 1e-12, "unit norm");
}

#[test]
fn certificate_hessian_pd_probe_classifies_definiteness() {
    assert_eq!(
        certificate_hessian_is_pd(&Array2::<f64>::eye(3)),
        Some(true)
    );
    let indefinite = array![[1.0, 2.0], [2.0, 1.0]];
    assert_eq!(certificate_hessian_is_pd(&indefinite), Some(false));
    assert_eq!(
        certificate_hessian_is_pd(&Array2::<f64>::zeros((0, 0))),
        None
    );
    let non_finite = array![[f64::NAN]];
    assert_eq!(certificate_hessian_is_pd(&non_finite), None);
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

/// Helper: build an objective whose VALUE path (the one the certificate
/// probes via `eval_cost`) is `0.5·ρ₀² + slope_railed·ρ₁`, and run the audit
/// directly at a constructed optimum where ρ₁ is railed at the upper box
/// bound. The analytic gradient is supplied separately so we control the
/// railed-vs-free desync structure exactly.
fn audit_at_railed_optimum(
    config: &OuterConfig,
    theta_hat: Array1<f64>,
    analytic_gradient: Array1<f64>,
    value_slope_railed: f64,
) -> Option<CriterionCertificate> {
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
                    hessian: HessianResult::Unavailable,
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
    result.final_gradient = Some(analytic_gradient);
    audit_first_order_optimality(&mut obj, config, "railed-audit-unit", &result)
}

/// TASK 1+2+3: at an optimum where the ONLY disagreement lives on a railed
/// box-bound coordinate, the certificate must NOT report a gradient-objective
/// desync. The full-direction FD-vs-analytic check would flag it (the value
/// slope along the railed coord disagrees with the analytic gradient there,
/// as it legitimately can under KKT), but the audit restricts the comparison
/// to the free (box-interior) subspace, where the two paths agree.
#[test]
fn certificate_does_not_false_flag_when_only_railed_coordinate_disagrees() {
    let bounded = OuterConfig {
        bounds: Some((array![-5.0, -5.0], array![5.0, 5.0])),
        ..OuterConfig::default()
    };
    // ρ₁ railed at the upper bound (5.0); ρ₀ interior at its quadratic min.
    let theta_hat = array![0.0, 5.0];
    // Analytic gradient: free coord 0 consistent (∂=ρ₀=0); railed coord 1
    // reports a small KKT-balanced slope (−0.5) that DISAGREES with the
    // value path's slope of +7.0 along ρ₁.
    let analytic_gradient = array![0.0, -0.5];
    let value_slope_railed = 7.0;

    // First, confirm the FULL-direction comparison WOULD flag this, so the
    // test actually exercises the artifact (not a no-op). Reconstruct the
    // full audit direction and the full-space directional derivatives.
    let full_dir = certificate_audit_direction(&theta_hat, "railed-audit-unit");
    assert!(
        full_dir[1].abs() > 1e-3,
        "audit direction must have a non-trivial railed component to make \
         the full-space check meaningful: {full_dir:?}",
    );
    let analytic_full = analytic_gradient.dot(&full_dir);
    // Value-path directional slope along the full direction:
    //   ∂/∂s [0.5(s·d₀)² + 7·(5 + s·d₁)] at s=0 = 7·d₁  (the ρ₀ term is O(s)).
    let fd_full_slope = value_slope_railed * full_dir[1];
    assert!(
        (analytic_full - fd_full_slope).abs() > 1e-2,
        "full-direction analytic and value-path slopes should disagree \
         (artifact precondition): analytic={analytic_full} fd≈{fd_full_slope}",
    );

    let cert = audit_at_railed_optimum(&bounded, theta_hat, analytic_gradient, value_slope_railed)
        .expect("railed audit must still produce a certificate");

    assert_eq!(
        cert.lambdas_railed,
        vec![1],
        "coord 1 must be detected as railed: {}",
        cert.summary(),
    );
    // The free subspace is coord 0 only, where value (½ρ₀²) and gradient (ρ₀)
    // agree exactly at ρ₀=0 → directional derivatives ≈ 0 and consistent.
    assert!(
        cert.first_order_consistent(),
        "railed-coordinate disagreement was FALSE-FLAGGED as a desync; the \
         free subspace agrees, so this must be reported consistent: {}",
        cert.summary(),
    );
    assert!(
        cert.agreement_z < CERTIFICATE_Z_GATE,
        "projected-onto-free z must be small: {}",
        cert.summary(),
    );
}

/// TASK 3 guard: the projection must NOT blunt the certificate's real job.
/// A genuine desync on a FREE (interior) coordinate must still fire even when
/// a different coordinate is railed.
#[test]
fn certificate_still_fires_on_genuine_interior_gradient_desync() {
    let bounded = OuterConfig {
        bounds: Some((array![-5.0, -5.0], array![5.0, 5.0])),
        ..OuterConfig::default()
    };
    // ρ₁ railed at the upper bound; ρ₀ interior but the analytic gradient on
    // the FREE coord 0 is wrong: it claims ∂=0 while the value path slopes by
    // ρ₀ (here ρ₀=2.5 → true slope 2.5). This is the #748/#752/#808 genus on
    // a free coordinate and MUST be caught.
    let theta_hat = array![2.5, 5.0];
    let analytic_gradient = array![0.0, -0.5]; // coord 0 wrong (should be 2.5)
    let value_slope_railed = 7.0;

    let cert = audit_at_railed_optimum(&bounded, theta_hat, analytic_gradient, value_slope_railed)
        .expect("railed audit must still produce a certificate");

    assert_eq!(
        cert.lambdas_railed,
        vec![1],
        "coord 1 railed: {}",
        cert.summary()
    );
    assert!(
        !cert.first_order_consistent(),
        "genuine interior (free-coordinate) desync was masked by the railed \
         projection — the certificate failed its core job: {}",
        cert.summary(),
    );
    assert!(
        cert.agreement_z > CERTIFICATE_Z_GATE,
        "interior desync must exceed the z gate: {}",
        cert.summary(),
    );
}

/// TASK 3 invariance: with nothing railed, the projection is identity and the
/// audit is byte-identical to the full-space path. A consistent interior
/// optimum stays clean; the railed list is empty.
#[test]
fn certificate_full_space_unchanged_when_nothing_railed() {
    let bounded = OuterConfig {
        bounds: Some((array![-30.0, -30.0], array![30.0, 30.0])),
        ..OuterConfig::default()
    };
    // Interior optimum, far from both bounds; gradient matches the value
    // path's slope on BOTH coords (value 0.5ρ₀² + 7ρ₁ ⇒ ∂₁=7).
    let theta_hat = array![0.0, 1.0];
    let analytic_gradient = array![0.0, 7.0];
    let cert = audit_at_railed_optimum(&bounded, theta_hat, analytic_gradient, 7.0)
        .expect("interior audit must produce a certificate");
    assert!(
        cert.lambdas_railed.is_empty(),
        "no coordinate is near a bound: {}",
        cert.summary(),
    );
    assert!(
        cert.first_order_consistent(),
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

impl OuterHessianOperator for FailingSeedMaterializationOperator {
    fn dim(&self) -> usize {
        self.dim
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        Ok(v.clone())
    }

    fn is_cheap_to_materialize(&self) -> bool {
        true
    }

    fn materialize_dense(&self) -> Result<Array2<f64>, String> {
        Err("seed materialization failed".to_string())
    }
}

#[test]
fn materialize_dense_uses_single_batched_mul_mat() {
    struct BatchedOnlyHessian {
        matrix: Array2<f64>,
        matvec_calls: Arc<AtomicUsize>,
        mul_mat_calls: Arc<AtomicUsize>,
        rhs_columns: Arc<AtomicUsize>,
    }

    impl OuterHessianOperator for BatchedOnlyHessian {
        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            self.matvec_calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.matrix.dot(v))
        }

        fn mul_mat(&self, factor: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
            self.mul_mat_calls.fetch_add(1, Ordering::Relaxed);
            self.rhs_columns
                .fetch_add(factor.ncols(), Ordering::Relaxed);
            Ok(self.matrix.dot(&factor))
        }
    }

    let matvec_calls = Arc::new(AtomicUsize::new(0));
    let mul_mat_calls = Arc::new(AtomicUsize::new(0));
    let rhs_columns = Arc::new(AtomicUsize::new(0));
    let op = BatchedOnlyHessian {
        matrix: array![[2.0, 0.25, -0.5], [0.5, 3.0, 1.0], [-0.25, 2.0, 4.0]],
        matvec_calls: Arc::clone(&matvec_calls),
        mul_mat_calls: Arc::clone(&mul_mat_calls),
        rhs_columns: Arc::clone(&rhs_columns),
    };

    let dense = op
        .materialize_dense()
        .expect("batched dense materialization");
    let expected = array![[2.0, 0.375, -0.375], [0.375, 3.0, 1.5], [-0.375, 1.5, 4.0]];
    assert_eq!(dense, expected);
    assert_eq!(
        mul_mat_calls.load(Ordering::Relaxed),
        1,
        "dense materialization must batch all identity columns into one mul_mat call"
    );
    assert_eq!(
        rhs_columns.load(Ordering::Relaxed),
        3,
        "the single batched materialization call must include every identity RHS"
    );
    assert_eq!(
        matvec_calls.load(Ordering::Relaxed),
        0,
        "operators with batched mul_mat must not be probed column-by-column"
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
fn plan_efs_not_selected_few_params_even_if_penalty_like() {
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
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
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
    let result = HessianResult::Analytic(h.clone());
    assert!(result.is_analytic());
    match result {
        HessianResult::Analytic(extracted) => assert_eq!(extracted, h),
        HessianResult::Operator(_) | HessianResult::Unavailable => {
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
                hessian: HessianResult::Unavailable,
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
        screening_proxy_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<f64, EstimationError>>,
        seed_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
        continuation_prewarm: true,
    };
    assert_eq!(obj.capability().n_params, 1);
    assert_eq!(obj.eval_cost(&Array1::zeros(1)).unwrap(), 1.0);
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
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            })
        },
        eval_order_fn: None::<
            fn(&mut Vec<f64>, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        >,
        reset_fn: None::<fn(&mut Vec<f64>)>,
        efs_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        screening_proxy_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<f64, EstimationError>>,
        seed_fn: None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
        continuation_prewarm: true,
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
                hessian: HessianResult::Unavailable,
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
            })
        }),
        screening_proxy_fn: None::<fn(&mut (), &Array1<f64>) -> Result<f64, EstimationError>>,
        seed_fn: None::<fn(&mut (), &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
        continuation_prewarm: true,
    };
    let mut bridge = OuterFixedPointBridge {
        obj: &mut obj,
        layout: cap.theta_layout(),
        barrier_config: None,
        fixed_point_tolerance: 1e-8,
        consecutive_psi_zero_iters: 0,
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
                    hessian: HessianResult::Unavailable,
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
                    hessian: HessianResult::Unavailable,
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
                        OuterEvalOrder::Value => HessianResult::Unavailable,
                        OuterEvalOrder::ValueAndGradient => HessianResult::Unavailable,
                        OuterEvalOrder::ValueGradientHessian => {
                            HessianResult::Analytic(array![[2.0]])
                        }
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
/// when the runtime returns `HessianResult::Unavailable`. A `None` here
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
                hessian: HessianResult::Unavailable,
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
fn arc_bridge_cost_stall_certifies_at_bound_separation() {
    // A flat objective at the lower bound `rho = -10` whose raw gradient is a
    // constant `g = -1` (points further DOWN, out of the feasible box): the
    // projected KKT residual there is 0, so a stall is a CONVERGED optimum —
    // exactly the separation signature (the REML score has bottomed out but the
    // unprojected gradient keeps pushing λ→0 forever).
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
                // Gradient points out of the lower bound; raw norm = 1 forever,
                // but the bound-projected residual at rho=-10 is 0.
                gradient: array![-1.0],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => HessianResult::Analytic(array![[1.0]]),
                    _ => HessianResult::Unavailable,
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
    // The guard tolerates the first `COST_STALL_WINDOW` no-improve steps, then
    // halts with the sentinel.
    let mut sentinel_fired = false;
    for _ in 0..(COST_STALL_WINDOW + 2) {
        match SecondOrderObjective::eval_hessian(&mut bridge, &lo) {
            Ok(_) => {}
            Err(ObjectiveEvalError::Fatal { message }) => {
                assert_eq!(
                    message, COST_STALL_CONVERGED_SENTINEL,
                    "ARC cost-stall must halt via the shared convergence sentinel"
                );
                sentinel_fired = true;
                break;
            }
            Err(other) => panic!("unexpected ARC bridge error: {other:?}"),
        }
    }
    assert!(
        sentinel_fired,
        "ARC bridge must halt the cost-stall valley within {} evals (separation never settles otherwise)",
        COST_STALL_WINDOW + 2
    );
    let published = exit.lock().unwrap().take().expect("best iterate published");
    assert!(
        published.converged,
        "an at-bound stall with a ZERO projected KKT residual must certify CONVERGED \
         (raw |g|=1 is the out-of-bounds separation gradient, not non-stationarity)"
    );
    assert_eq!(published.rho, lo, "best iterate is the bound-pinned ρ");
    assert_eq!(published.value, 1.0);
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
fn arc_bridge_cost_stall_halts_on_kkt_stationary_bound_descent() {
    let lo = array![-10.0];
    let hi = array![10.0];
    // Strictly-decreasing cost so the cost-improvement test alone would NEVER
    // fire; the gradient points out of the lower bound (raw |g|=1, projected
    // KKT residual at rho=-10 is 0), so the halt rides on stationarity.
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
                // Out-of-bounds gradient at the lower bound: raw norm = 1 forever,
                // bound-projected residual = 0 (KKT-stationary).
                gradient: array![-1.0],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => HessianResult::Analytic(array![[1.0]]),
                    _ => HessianResult::Unavailable,
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
    let mut sentinel_fired = false;
    for _ in 0..(COST_STALL_WINDOW + 2) {
        match SecondOrderObjective::eval_hessian(&mut bridge, &lo) {
            Ok(_) => {}
            Err(ObjectiveEvalError::Fatal { message }) => {
                assert_eq!(
                    message, COST_STALL_CONVERGED_SENTINEL,
                    "KKT-stationary-at-bound halt must use the shared convergence sentinel"
                );
                sentinel_fired = true;
                break;
            }
            Err(other) => panic!("unexpected ARC bridge error: {other:?}"),
        }
    }
    assert!(
        sentinel_fired,
        "ARC bridge must halt the bound-pinned descent within {} evals even though the raw \
         cost is still strictly decreasing (the projected KKT residual is zero)",
        COST_STALL_WINDOW + 2
    );
    let published = exit.lock().unwrap().take().expect("best iterate published");
    assert!(
        published.converged,
        "a KKT-stationary at-bound stall (projected |g|=0 ≤ 1e-3) must certify CONVERGED"
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
                        OuterEvalOrder::ValueGradientHessian => {
                            HessianResult::Analytic(array![[1.0]])
                        }
                        _ => HessianResult::Unavailable,
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
                    message, COST_STALL_CONVERGED_SENTINEL,
                    "infeasible-run halt must use the shared convergence sentinel"
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
        published.converged,
        "halt back to the feasible iterate (projected |g|≈1e-9 ≤ 1e-3) must certify CONVERGED"
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

#[test]
fn bfgs_bridge_halts_infeasible_probe_run_back_to_cached_seed() {
    let seed = array![0.0];
    let trial = array![1.0];
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

    let mut sentinel_fired = false;
    for _ in 0..(COST_STALL_WINDOW + 2) {
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
    let verdict = guard.observe_constrained_stationary(&boundary_probe, 0.5, 0.0, true);
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
    let verdict = guard.observe_constrained_stationary(&collapse_corner, 587.84, 0.0, true);

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
    let _ = guard.observe_constrained_stationary(&collapse_corner, 587.84, 0.0, true);
    let final_verdict = guard.observe_constrained_stationary(&collapse_corner, 587.84, 0.0, true);
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
        let _ = guard.observe_infeasible(&probe);
        let _ = guard.observe_infeasible(&probe);
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

/// #1426 companion: a GENUINE flat-valley floor — cost flatlined AND the
/// projected gradient is only modestly above tolerance (well below the ceiling)
/// — must still halt as a `FlatValleyStall` (the legitimately-flat REML surface
/// of #1082/#1237). The #1426 escape must not weaken that path.
#[test]
fn cost_stall_modestly_above_tolerance_still_halts_as_flat_valley() {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-6, 3, 1.0e-3, exit.clone());
    let seed = array![0.0, 0.0];
    // Residual modestly above tolerance but BELOW the ceiling: a real flat
    // valley floor (the surface has genuinely flattened).
    let valley_grad = FLAT_VALLEY_STALL_GRAD_CEILING * 0.5;
    guard.observe_seed(&seed, 10.0, valley_grad);

    let probe = array![-10.0, -10.0];
    let _ = guard.observe_infeasible(&probe);
    let _ = guard.observe_infeasible(&probe);
    let verdict = guard.observe_infeasible(&probe);
    assert!(
        matches!(verdict, CostStallVerdict::FlatValleyStall { .. }),
        "a modest residual below the ceiling is a genuine flat-valley floor and \
         must halt as before (#1082/#1237 unaffected by the #1426 fix)"
    );
    let published = exit.lock().unwrap().take().expect("best published");
    assert!(!published.converged);
}

#[test]
fn lower_bound_outward_axes_mark_separation_stationarity() {
    let lower = array![-10.0, -10.0, -10.0, -10.0];
    let upper = array![10.0, 10.0, 10.0, 10.0];
    let rho = array![-10.0, -10.0, 0.25, 1.0];
    let gradient = array![-2.0e-2, -4.0e-2, 3.0, -1.0];

    assert_eq!(
        lower_bound_outward_active_count(&rho, &gradient, Some(&(lower, upper)), 1.0e-3),
        LOWER_BOUND_SEPARATION_ACTIVE_MIN,
        "two lower-bound axes with outward gradients are enough to identify \
         a separation-bound stationary probe"
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
fn plan_hybrid_efs_not_selected_few_params() {
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
    assert_eq!(p.solver, Solver::Bfgs);
    assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
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
                hessian: HessianResult::Unavailable,
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
                hessian: HessianResult::Unavailable,
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
                hessian: HessianResult::Analytic(array![[f64::NAN, 0.0], [0.0, 1.0]]),
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
            hessian: HessianResult::Unavailable,
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
        crate::seeding::generate_rho_candidates(1, None, &crate::seeding::SeedConfig::default());
    let valid_seed = generated
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let expected_seed = valid_seed.clone();
    let initial_seed = array![9.0];
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                    hessian: HessianResult::Unavailable,
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

#[test]
fn run_indefinite_analytic_seed_stays_on_arc() {
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                hessian: HessianResult::Analytic(array![[-1.0]]),
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
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                hessian: HessianResult::Operator(Arc::new(FailingSeedMaterializationOperator {
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
fn run_nonconverged_arc_stays_on_arc_after_budget_retry_ladder() {
    // When an ARC primary exhausts its iteration budget, the runner
    // reseeds a fresh ARC attempt from the previous attempt's last
    // ρ and trust radius (up to two retries) and uncaps the inner
    // PIRLS cap for the resumed run via the InnerProgressFeedback
    // handle. Retries are gated on attempt-over-attempt `‖g‖`
    // halving so a deterministic-replay trajectory falls through.
    // The objective's analytic outer Hessian is preserved across
    // every attempt — no lateral demote to BFGS+BfgsApprox. After
    // the retries are exhausted (or the gate fires), the runner
    // returns the final `Ok(OuterResult{converged:false})` from
    // the last ARC attempt; the plan stays ARC + Analytic Hessian.
    //
    // We use `cost = x^4`, `grad = 4 x^3`, `hess = 12 x^2` from
    // `initial_rho = [5.0]` with `max_iter = 1`. Newton-style ARC
    // steps on x^4 contract the gradient by ~3× per attempt, so
    // the halving gate passes and both retries proceed; ARC still
    // cannot reach the optimum in three single-iter attempts.
    let mut seed_config = crate::seeding::SeedConfig::default();
    seed_config.seed_budget = 1;
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
                hessian: HessianResult::Analytic(array![[12.0 * x.powi(2)]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "nonconverged arc should stay on arc")
        .expect(
            "ARC ladder must surface the last non-converged ARC result rather than \
                 demoting to BFGS+BfgsApprox",
        );
    assert_eq!(
        result.plan_used.solver,
        Solver::Arc,
        "ARC primary must not lateral-demote after budget exhaustion"
    );
    assert_eq!(
        result.plan_used.hessian_source,
        HessianSource::Analytic,
        "analytic outer Hessian must be preserved across the budget-bump retry ladder"
    );
    assert!(
        !result.converged,
        "test fixture is engineered so the ladder cannot converge; \
             converged=true would mean the fixture stopped exercising the ladder"
    );
}

#[test]
fn candidate_selection_prefers_lower_cost_within_same_convergence_class() {
    let plan = OuterPlan {
        solver: Solver::Bfgs,
        hessian_source: HessianSource::BfgsApprox,
    };
    let mut nonconverged_hi = OuterResult::new(array![0.0], 9.0, 1, false, plan);
    nonconverged_hi.final_grad_norm = Some(1.0);
    let mut nonconverged_lo = OuterResult::new(
        array![1.0],
        1.0,
        1,
        false,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    nonconverged_lo.final_grad_norm = Some(1.0);
    let mut converged = OuterResult::new(
        array![2.0],
        5.0,
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    converged.final_grad_norm = Some(0.0);

    assert!(candidate_improves_best(&nonconverged_hi, None));
    assert!(candidate_improves_best(
        &nonconverged_lo,
        Some(&nonconverged_hi)
    ));
    assert!(!candidate_improves_best(
        &nonconverged_hi,
        Some(&nonconverged_lo)
    ));
    assert!(candidate_improves_best(&converged, Some(&nonconverged_lo)));
    assert!(!candidate_improves_best(&nonconverged_lo, Some(&converged)));
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

    // The convergence-class rule is unchanged: a converged candidate always
    // beats a non-converged incumbent regardless of LAML/parsimony.
    let nonconverged = OuterResult::new(array![5.0, 5.0], 50.0, 1, false, plan);
    assert!(candidate_improves_best_parsimonious(
        &parsimonious,
        Some(&nonconverged),
        rho_dim,
    ));
    assert!(!candidate_improves_best_parsimonious(
        &nonconverged,
        Some(&parsimonious),
        rho_dim,
    ));
}

#[test]
fn gaussian_multistart_compares_converged_seed_costs() {
    let mut seed_config = crate::seeding::SeedConfig::default();
    seed_config.seed_budget = 2;
    seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
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
                    hessian: HessianResult::Unavailable,
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

/// #1082 separable-multinomial guard: on an expensive-solver risk profile
/// (ARC + GeneralizedLinear, `seed_budget = 2`), a first seed that lands a
/// FEASIBLE cost-stall flat-valley result (the near-separable λ→0 ridge: finite
/// cost, gradient never clears tolerance) must STOP the multi-start instead of
/// paying a second expensive seed that cannot reach a stationary point.
/// Regression for the penguin-species timeout where the wasted second seed
/// crawled ~70s/eval.
#[test]
fn expensive_multistart_stops_after_feasible_nonstationary_seed() {
    let plan = OuterPlan {
        solver: Solver::Arc,
        hessian_source: HessianSource::Analytic,
    };
    let mut result = OuterResult::new(array![0.0], 12.0, 9, false, plan);
    result.final_grad_norm = Some(5.0);
    result.operator_stop_reason = Some(OperatorTrustRegionStopReason::CostStallFlatValley);

    assert!(
        should_stop_expensive_multistart_after_best(Some(&result), Some(2), false),
        "a finite cost-stall flat-valley result is the separable-fit signature; \
         trying another expensive seed only repeats the λ→0 crawl (#1082)"
    );

    let mut plain_nonconverged = result.clone();
    plain_nonconverged.operator_stop_reason = Some(OperatorTrustRegionStopReason::IterationBudget);
    assert!(
        !should_stop_expensive_multistart_after_best(Some(&plain_nonconverged), Some(2), false),
        "ordinary finite nonconvergence may still be seed-sensitive and must keep the \
         second expensive seed"
    );
    assert!(
        !should_stop_expensive_multistart_after_best(Some(&result), Some(2), true),
        "Gaussian quality-compare mode intentionally evaluates remaining seeds"
    );
}

#[test]
fn run_starts_solver_with_direct_startup_eval() {
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                    hessian: HessianResult::Analytic(array![[2.0]]),
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
    let mut seed_config = crate::seeding::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 2;
    seed_config.risk_profile = crate::seeding::SeedRiskProfile::GeneralizedLinear;
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
                        hessian: HessianResult::Analytic(array![[1.0]]),
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
    assert_eq!(
        started.lock().unwrap().first().cloned(),
        Some(valid_seed),
        "screening should move the lowest-cost seed to the front before full startup eval",
    );
    assert_eq!(screening_cap.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn initial_rho_with_single_seed_budget_skips_expensive_screening() {
    let mut seed_config = crate::seeding::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 1;
    seed_config.risk_profile = crate::seeding::SeedRiskProfile::GeneralizedLinear;
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
                        hessian: HessianResult::Analytic(array![[1.0]]),
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
    let mut seed_config = crate::seeding::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
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
                        hessian: HessianResult::Unavailable,
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
    assert_eq!(
        started.lock().unwrap().first().cloned(),
        Some(valid_seed),
        "BFGS screening should move the lowest-cost seed to the front before full startup eval",
    );
    assert!(
        screening_calls.load(Ordering::Relaxed) > 1,
        "BFGS seed screening should rank candidates with cost-only probes first",
    );
    assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
}

#[test]
fn screening_cap_survives_per_seed_reset_before_proxy_eval() {
    let mut seed_config = crate::seeding::SeedConfig::default();
    seed_config.max_seeds = 3;
    seed_config.seed_budget = 1;
    seed_config.risk_profile = crate::seeding::SeedRiskProfile::Gaussian;
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
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            })
        },
        |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
            Ok(OuterEval {
                cost: theta[0].abs(),
                gradient: array![0.0],
                hessian: HessianResult::Unavailable,
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
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                        hessian: HessianResult::Analytic(array![[1.0]]),
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
    let mut seed_config = crate::seeding::SeedConfig::default();
    seed_config.max_seeds = 6;
    seed_config.seed_budget = 1;
    let screening_calls = Arc::new(AtomicUsize::new(0));
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
        Some(|_: &mut (), theta: &Array1<f64>| {
            Ok(EfsEval {
                cost: 0.0,
                steps: vec![0.0; theta.len()],
                beta: None,
                psi_gradient: None,
                psi_indices: None,
                inner_hessian_scale: None,
                logdet_enclosure_gap: None,
            })
        }),
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
}

#[test]
fn run_efs_skips_invalid_leading_seed_without_spending_budget() {
    let generated =
        crate::seeding::generate_rho_candidates(15, None, &crate::seeding::SeedConfig::default());
    let valid_seed = generated
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let invalid_seed = Array1::from_elem(15, 9.0);
    let mut seed_config = crate::seeding::SeedConfig::default();
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
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                hessian: HessianResult::Unavailable,
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
                hessian: HessianResult::Unavailable,
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
            crate::seeding::SeedRiskProfile::GeneralizedLinear,
            false,
        ),
        1
    );
    assert_eq!(
        effective_seed_budget(
            4,
            Solver::HybridEfs,
            crate::seeding::SeedRiskProfile::Survival,
            false,
        ),
        1
    );
    assert_eq!(
        effective_seed_budget(
            3,
            Solver::Arc,
            crate::seeding::SeedRiskProfile::GeneralizedLinear,
            true,
        ),
        2
    );
    assert_eq!(
        effective_seed_budget(
            1,
            Solver::Arc,
            crate::seeding::SeedRiskProfile::GeneralizedLinear,
            true,
        ),
        2
    );
    assert_eq!(
        effective_seed_budget(
            3,
            Solver::Arc,
            crate::seeding::SeedRiskProfile::Survival,
            false,
        ),
        1
    );
    assert_eq!(
        effective_seed_budget(
            3,
            Solver::Bfgs,
            crate::seeding::SeedRiskProfile::Survival,
            false,
        ),
        3
    );
}

#[test]
fn run_arc_projects_seed_before_seed_validation_eval() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                    hessian: HessianResult::Analytic(array![[2.0]]),
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
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                    hessian: HessianResult::Unavailable,
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
fn checkpointing_objective_rejects_wrong_dim_on_decode() {
    // A payload from a 3-dim fit is invalid input for a 5-dim resume.
    let bytes = encode_iterate(&array![1.0, 2.0, 3.0], None, None, 0.5, 0).expect("encode");
    assert!(decode_iterate(&bytes, 3).is_some());
    assert!(decode_iterate(&bytes, 5).is_none());
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
                hessian: HessianResult::Unavailable,
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
                hessian: HessianResult::Unavailable,
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
fn run_calls_seed_inner_state_with_cached_beta() {
    // End-to-end read-side wiring: a cache hit carrying β must call
    // OuterObjective::seed_inner_state(&beta) *before* the first BFGS
    // eval. We verify this by routing through a custom OuterObjective
    // that records the β it was seeded with.
    struct RecordingObj {
        seeded: Arc<Mutex<Option<Array1<f64>>>>,
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
            *self.eval_count.lock().unwrap() += 1;
            // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: 2.0 * theta,
                hessian: HessianResult::Analytic(2.0 * Array2::<f64>::eye(theta.len())),
                inner_beta_hint: None,
            })
        }
        fn reset(&mut self) {}
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
    let eval_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let mut obj = RecordingObj {
        seeded: Arc::clone(&seeded),
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

    let observed = seeded.lock().unwrap().clone();
    assert_eq!(
        observed,
        Some(array![7.5, 8.5, 9.5]),
        "dispatcher must call seed_inner_state with the cached β before run_outer",
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
                hessian: HessianResult::Analytic(2.0 * Array2::<f64>::eye(theta.len())),
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
        |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: (theta[0] - 2.5).powi(2),
                gradient: array![2.0 * (theta[0] - 2.5)],
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            })
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
    let mut seed_config = crate::seeding::SeedConfig::default();
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
                hessian: HessianResult::Unavailable,
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
                hessian: HessianResult::Unavailable,
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

// ─── continuation pre-warm budget on a warm-start store hit ──────────

/// An expensive-shape outer problem with no cache hit keeps its
/// shape-derived continuation pre-warm budget. This pins the cold-start
/// contract so the hit-path skip below is provably the only behavior change.
#[test]
fn prewarm_budget_cold_start_keeps_shape_budget() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        // n_params >= EXPENSIVE_PREWARM_RHO_DIM makes this an "expensive" shape.
        n_params: EXPENSIVE_PREWARM_RHO_DIM,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let config = OuterConfig {
        warm_start_cache_hit: false,
        ..OuterConfig::default()
    };
    // Single-seed expensive shape => SINGLE_EXPENSIVE_PREWARM_BUDGET (capped at
    // PATH_BUDGET). The exact value is the existing cold-start contract; the
    // load-bearing assertion is that it is strictly positive (pre-warm runs).
    let budget = continuation_prewarm_step_budget(&config, &cap, 1, 1);
    assert_eq!(
        budget,
        SINGLE_EXPENSIVE_PREWARM_BUDGET
            .min(crate::solver::estimate::reml::continuation::PATH_BUDGET),
        "cold-start expensive single-seed shape must keep its shape-derived budget"
    );
    assert!(
        budget > 0,
        "cold start must still run the continuation pre-warm"
    );
}

/// On a warm-start store hit the seed is already near-optimal, so the
/// continuation pre-warm budget collapses to zero regardless of problem
/// shape — the only difference vs the cold-start case above is the flag.
#[test]
fn prewarm_budget_warm_start_cache_hit_is_zero() {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        n_params: EXPENSIVE_PREWARM_RHO_DIM,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let config = OuterConfig {
        warm_start_cache_hit: true,
        ..OuterConfig::default()
    };
    let budget = continuation_prewarm_step_budget(&config, &cap, 1, 1);
    assert_eq!(
        budget, 0,
        "a warm-start store hit must skip the redundant continuation pre-warm"
    );
}

// ─── #979 cost-cliff pre-warm budget scaling ─────────────────────────

/// Build a single-seed cold `OuterConfig` reporting `p_coefficients`, with a
/// cheap rho dimension so `expensive_shape` is driven only by the coefficient
/// dim (mirrors the binary marginal-slope outer: a couple of ρ, a basis that
/// grows with center count).
fn prewarm_config_for_p(p_coefficients: usize) -> (OuterConfig, OuterCapability) {
    let cap = OuterCapability {
        gradient: Derivative::Analytic,
        hessian: DeclaredHessianForm::Unavailable,
        // Below EXPENSIVE_PREWARM_RHO_DIM so the rho dimension never declares
        // the shape expensive on its own — the coefficient dim is the lever.
        n_params: 2,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    };
    let config = OuterConfig {
        warm_start_cache_hit: false,
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(2500),
            p_coefficients: Some(p_coefficients),
        },
        ..OuterConfig::default()
    };
    (config, cap)
}

/// The continuation pre-warm step budget must SCALE DOWN as the coefficient
/// dimension (center count) grows past the #979 cost cliff, instead of paying
/// the full `PATH_BUDGET` of multi-second inner solves per seed. This pins the
/// binary marginal-slope acceptance regime: two `matern(centers=K)` formulas
/// give `p ≈ 2K`, so centers ∈ {4, 12, 20} land at p ≈ {8, 24, 40}.
#[test]
fn prewarm_budget_scales_down_past_cost_cliff() {
    let path_budget = crate::solver::estimate::reml::continuation::PATH_BUDGET;

    // centers ≈ 4 (p ≈ 8): below the cliff, the cheap fit keeps the full
    // budget so the seed-continuation accuracy is untouched.
    let (cfg4, cap4) = prewarm_config_for_p(8);
    let b4 = continuation_prewarm_step_budget(&cfg4, &cap4, 1, 1);
    assert_eq!(
        b4, path_budget,
        "a cheap (small-p) cold fit must keep the full pre-warm budget"
    );

    // centers ≈ 8 (p ≈ 16): just past the cliff, the budget must already have
    // collapsed far below PATH_BUDGET (the empirical centers≈8→10 cliff).
    let (cfg8, cap8) = prewarm_config_for_p(16);
    let b8 = continuation_prewarm_step_budget(&cfg8, &cap8, 1, 1);
    assert!(
        b8 < path_budget && b8 >= PREWARM_MIN_SCALED_BUDGET,
        "just past the cost cliff the pre-warm budget must collapse below \
         PATH_BUDGET ({path_budget}) yet stay >= {PREWARM_MIN_SCALED_BUDGET}; got {b8}"
    );

    // centers ≈ 12 (p ≈ 24) and centers ≈ 20 (p ≈ 40): the budget is
    // non-increasing in p, and the per-seed pre-warm WORK proxy `budget · p`
    // stays bounded (so the centers=20 fit does not pay 64 inner solves).
    let (cfg12, cap12) = prewarm_config_for_p(24);
    let b12 = continuation_prewarm_step_budget(&cfg12, &cap12, 1, 1);
    let (cfg20, cap20) = prewarm_config_for_p(40);
    let b20 = continuation_prewarm_step_budget(&cfg20, &cap20, 1, 1);

    assert!(
        b8 >= b12 && b12 >= b20,
        "pre-warm budget must be non-increasing in center count: \
         p=16->{b8}, p=24->{b12}, p=40->{b20}"
    );
    assert!(
        b20 >= PREWARM_MIN_SCALED_BUDGET,
        "even the largest fit must still anneal >= {PREWARM_MIN_SCALED_BUDGET} \
         legs so the warm β stays near-optimal; got {b20}"
    );
    // Bounded total work: budget · p must not exceed the target product (plus
    // one p of slack for the integer-division floor), for every above-cliff p.
    for (b, p) in [(b8, 16usize), (b12, 24), (b20, 40)] {
        assert!(
            b * p <= PREWARM_COST_BUDGET_COEFF_PRODUCT + p,
            "above-cliff pre-warm work budget·p={} must stay bounded by ~{} (p={p})",
            b * p,
            PREWARM_COST_BUDGET_COEFF_PRODUCT
        );
    }
}

/// The cost-scaling helper is the identity below the cliff and never returns
/// zero (the pre-warm must always run at least its floor of legs on a cold
/// fit, so capping cannot regress the seed-continuation accuracy).
#[test]
fn cost_scaled_prewarm_budget_is_bounded_and_never_zero() {
    let path_budget = crate::solver::estimate::reml::continuation::PATH_BUDGET;
    // Identity below the cliff.
    for p in [0usize, 1, 8, PREWARM_COST_CLIFF_COEFF_DIM] {
        assert_eq!(
            cost_scaled_prewarm_budget(path_budget, p),
            path_budget,
            "below/at the cliff the budget is unscaled (p={p})"
        );
    }
    // Past the cliff: in [floor, base], non-increasing, never zero.
    let mut prev = path_budget + 1;
    for p in (PREWARM_COST_CLIFF_COEFF_DIM + 1)..=256 {
        let b = cost_scaled_prewarm_budget(path_budget, p);
        assert!(
            b >= PREWARM_MIN_SCALED_BUDGET,
            "p={p} budget {b} below floor"
        );
        assert!(b <= path_budget, "p={p} budget {b} above base");
        assert!(
            b <= prev,
            "p={p} budget {b} not non-increasing (prev {prev})"
        );
        prev = b;
    }
}

// ─── #979 outer wall-clock deadline (survival marginal-slope hang) ────

/// gam#979 — second, last-resort termination guarantee for the survival
/// marginal-slope hang. The slow-geometric-rate stall guard
/// (`loop_guard::slow_geometric_rate_exceeds_projection_cap`) ends the inner
/// joint-Newton when the residual is descending too slowly to finish, but the
/// outer search can still cascade across MANY ρ seeds/plans (every one
/// rejecting on the monotonicity-pinned baseline). The process-global
/// `OUTER_WALL_CLOCK_DEADLINE`, armed once around the whole survival fit, is the
/// catch-all that the inner joint-Newton cycle loop checks
/// (`inner_blockwise_fit.rs`: `if cycle > 0 && outer_wall_clock_deadline_exceeded()`)
/// so the public API returns its best-effort iterate in bounded time instead of
/// hanging to timeout.
///
/// This pins the three load-bearing properties of that mechanism with NO
/// wall-clock assertion (deterministic on the armed `Instant` only):
///   1. nothing armed  => `exceeded()` is `false` (every non-survival fit is
///      byte-for-byte unchanged — the guard is opt-in);
///   2. a deadline already in the PAST => `exceeded()` is `true` (the inner loop
///      breaks and the fit terminates);
///   3. a deadline in the FUTURE => `exceeded()` is `false` (a fast survival fit
///      is never cut short before it converges);
///   4. `clear()` restores the unbounded default so a stale past deadline can
///      never leak forward and bound a later, unrelated fit.
///
/// Serialised inside one `#[test]` (the deadline is a process-global) and always
/// cleared on exit so it cannot perturb any sibling test.
#[test]
fn outer_wall_clock_deadline_bounds_then_clears_979() {
    use std::time::{Duration, Instant};

    // Start from a known-clear state regardless of any prior test ordering.
    clear_outer_wall_clock_deadline();
    assert!(
        !outer_wall_clock_deadline_exceeded(),
        "with nothing armed the deadline guard must be inert (non-survival fits unchanged)"
    );

    // (2) A deadline in the past => exceeded immediately. The inner loop's
    //     `cycle > 0 && outer_wall_clock_deadline_exceeded()` check then breaks,
    //     returning the best-effort iterate instead of grinding to timeout.
    arm_outer_wall_clock_deadline(Instant::now() - Duration::from_secs(1));
    assert!(
        outer_wall_clock_deadline_exceeded(),
        "a past deadline must report exceeded so the inner joint-Newton terminates"
    );

    // (4) Clearing restores the unbounded default — a stale deadline must never
    //     leak to a later, unrelated fit.
    clear_outer_wall_clock_deadline();
    assert!(
        !outer_wall_clock_deadline_exceeded(),
        "clearing must restore the unbounded default (no stale-deadline leak)"
    );

    // (3) A generous future deadline must NOT fire, so a fast survival fit is
    //     never cut short before its KKT/REML certificate.
    arm_outer_wall_clock_deadline(Instant::now() + Duration::from_secs(3600));
    assert!(
        !outer_wall_clock_deadline_exceeded(),
        "a future deadline must not fire — a fast fit converges normally"
    );

    // Always leave the global clear so no sibling test sees an armed deadline.
    clear_outer_wall_clock_deadline();
    assert!(!outer_wall_clock_deadline_exceeded());
}
