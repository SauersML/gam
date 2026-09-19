#![cfg(test)]

use super::*;
use crate::custom_family::BlockWorkingSet;
use faer::sparse::{SparseColMat, Triplet};
use gam_problem::SasLinkSpec;
use gam_solve::gauge::Gauge;
use gam_solve::mixture_link::{state_from_beta_logisticspec, state_from_sasspec};
use ndarray::{Array1, array};

fn survival_ls_log_survival_stack(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<[f64; 5], String> {
    let (log_s, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
            inverse_link,
            eta,
            0.0,
        )?;
    Ok([log_s, -r, -dr, -ddr, -dddr])
}

fn survival_ls_log_pdf_stack(
    inverse_link: &InverseLink,
    eta: f64,
    deriv_log_scale: f64,
) -> Result<[f64; 5], String> {
    let (log_pdf, d1, d2, d3, d4) =
        SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
            inverse_link,
            eta,
            deriv_log_scale,
        )?;
    Ok([log_pdf, d1, d2, d3, d4])
}

fn survival_ls_positive_log_stack(value: f64) -> [f64; 5] {
    let (log_v, d1, d2, d3, d4) = SurvivalLocationScaleFamily::logwith_derivatives_positive(value);
    [log_v, d1, d2, d3, d4]
}

impl SurvivalExactRowKernel {
}

fn sparse_design_from_dense(dense: &Array2<f64>) -> DesignMatrix {
    let mut triplets = Vec::new();
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            let value = dense[[i, j]];
            if value != 0.0 {
                triplets.push(Triplet::new(i, j, value));
            }
        }
    }
    DesignMatrix::from(
        SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
            .expect("build sparse design"),
    )
}

/// Parity test for issue #410: the survival covariate spatial-ψ derivative
/// blocks (`Static` template) are produced by the *shared* exact-derivative
/// engine, not a survival-local re-implementation. A custom/built-in family
/// and the survival family with identical anisotropic-Matérn specs must
/// therefore yield bit-identical ψ-derivative blocks — design embedding,
/// penalty components, anisotropic cross-rows, and implicit-operator action.
#[test]
fn survival_static_spatial_psi_blocks_match_shared_engine() {
    use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use gam_terms::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};

    let n = 12usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (0.41 * i as f64).sin() + 0.15 * x0;
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
    }

    let spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "spatial".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::EqualMass { num_centers: 6 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(0.7),
                    nu: MaternNu::ThreeHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: Some(vec![0.0, 0.0]),
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    };

    let base_design =
        build_term_collection_design(data.view(), &spec).expect("build base spatial design");
    let resolvedspec =
        freeze_term_collection_from_design(&spec, &base_design).expect("freeze spatial term spec");
    let resolved_design = build_term_collection_design(data.view(), &resolvedspec)
        .expect("rebuild frozen spatial design");

    // Built-in / canonical path: the shared exact-derivative engine.
    let shared = crate::spatial_psi_bridge::build_block_spatial_psi_derivatives(
        data.view(),
        &resolvedspec,
        &resolved_design,
    )
    .expect("shared engine spatial psi derivatives")
    .expect("anisotropic spatial derivative rows from shared engine");

    // Survival consumer path: the Static adapter must delegate to the same engine.
    let survival = build_survival_covariate_block_psi_derivatives(
        data.view(),
        &resolvedspec,
        &resolved_design,
        &SurvivalCovariateTermBlockTemplate::Static,
    )
    .expect("survival static spatial psi derivatives")
    .expect("anisotropic spatial derivative rows from survival adapter");

    assert_eq!(
        shared.len(),
        survival.len(),
        "shared engine and survival adapter must emit the same number of ψ blocks"
    );

    let psi_dim = shared.len();
    for (axis, (a, b)) in shared.iter().zip(survival.iter()).enumerate() {
        assert_eq!(
            a.penalty_index, b.penalty_index,
            "penalty_index axis {axis}"
        );
        assert_eq!(
            a.implicit_axis, b.implicit_axis,
            "implicit_axis axis {axis}"
        );
        assert_eq!(
            a.implicit_group_id, b.implicit_group_id,
            "implicit_group_id axis {axis}"
        );
        assert_eq!(a.x_psi, b.x_psi, "x_psi axis {axis}");
        assert_eq!(
            a.s_psi_components, b.s_psi_components,
            "s_psi_components axis {axis}"
        );
        assert_eq!(a.x_psi_psi, b.x_psi_psi, "x_psi_psi axis {axis}");
        assert_eq!(
            a.s_psi_psi_components, b.s_psi_psi_components,
            "s_psi_psi_components axis {axis}"
        );

        // Implicit-operator action parity: identical embedding and identical
        // forward/transpose maps on deterministic probe vectors.
        match (a.implicit_operator.as_ref(), b.implicit_operator.as_ref()) {
            (Some(op_a), Some(op_b)) => {
                assert_eq!(op_a.n_data(), op_b.n_data(), "operator n_data axis {axis}");
                assert_eq!(op_a.p_out(), op_b.p_out(), "operator p_out axis {axis}");
                let p = op_a.p_out();
                let u: Array1<f64> = (0..p)
                    .map(|j| 0.3 + 0.11 * (j as f64) - 0.07 * ((axis + j) as f64).cos())
                    .collect();
                for probe_axis in 0..psi_dim {
                    let fwd_a = op_a
                        .forward_mul(probe_axis, &u.view())
                        .expect("shared forward_mul");
                    let fwd_b = op_b
                        .forward_mul(probe_axis, &u.view())
                        .expect("survival forward_mul");
                    assert_eq!(
                        fwd_a, fwd_b,
                        "forward_mul mismatch block {axis} probe-axis {probe_axis}"
                    );
                    let nd = op_a.n_data();
                    let v: Array1<f64> = (0..nd)
                        .map(|r| 0.2 - 0.05 * (r as f64) + 0.13 * ((r + probe_axis) as f64).sin())
                        .collect();
                    let tr_a = op_a
                        .transpose_mul(probe_axis, &v.view())
                        .expect("shared transpose_mul");
                    let tr_b = op_b
                        .transpose_mul(probe_axis, &v.view())
                        .expect("survival transpose_mul");
                    assert_eq!(
                        tr_a, tr_b,
                        "transpose_mul mismatch block {axis} probe-axis {probe_axis}"
                    );
                }
            }
            (None, None) => {}
            _ => panic!("implicit_operator presence diverged at axis {axis}"),
        }
    }
}

fn test_link_wiggle_metadata(beta_link_wiggle: &Array1<f64>) -> (Array1<f64>, usize) {
    let seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    for degree in [2usize, 3, 1] {
        for num_internal_knots in 0..=8 {
            if let Ok(knots) = crate::wiggle::monotone_warp_knots_from_seed(
                seed.view(),
                degree,
                num_internal_knots,
            ) && let Ok(block) = crate::wiggle::buildwiggle_block_input_from_orders(
                seed.view(),
                &knots,
                degree,
                &[2],
                false,
            ) && block.design.ncols() == beta_link_wiggle.len()
            {
                return (knots, degree);
            }
        }
    }
    panic!(
        "could not synthesize valid link wiggle metadata for {} coefficients",
        beta_link_wiggle.len()
    );
}

fn test_survival_fit(
    beta_time: Array1<f64>,
    beta_threshold: Array1<f64>,
    beta_log_sigma: Array1<f64>,
    beta_link_wiggle: Option<Array1<f64>>,
) -> UnifiedFitResult {
    let lambdas_linkwiggle = beta_link_wiggle.as_ref().map(|_| Array1::zeros(0));
    let (link_wiggle_knots, link_wiggle_degree) = beta_link_wiggle
        .as_ref()
        .map(|beta| {
            let (knots, degree) = test_link_wiggle_metadata(beta);
            (Some(knots), Some(degree))
        })
        .unwrap_or((None, None));
    survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
        training_sample_size: 32,
        beta_time,
        beta_threshold,
        beta_log_sigma,
        beta_link_wiggle,
        link_wiggle_knots,
        link_wiggle_degree,
        lambdas_time: Array1::zeros(0),
        lambdas_threshold: Array1::zeros(0),
        lambdas_log_sigma: Array1::zeros(0),
        lambdas_linkwiggle,
        log_likelihood: 0.0,
        reml_score: Some(0.0),
        stable_penalty_term: 0.0,
        penalized_objective: Some(0.0),
        used_device: false,
        outer_iterations: 0,
        outer_gradient_norm: None,
        criterion_certificate: None,
        outer_converged: true,
        covariance_conditional: None,
        covariance_corrected: None,
        smoothing_correction: None,
        smoothing_correction_absence: None,
        geometry: None,
        penalty_block_trace: Vec::new(),
        edf_by_block: Vec::new(),
        edf_rank_bound: Vec::new(),
        coefficient_mode_selection: Default::default(),
    })
    .expect("valid survival test fit")
}

fn survival_fit_parts_with_outer_evidence(
    outer_iterations: usize,
    criterion_certificate: Option<gam_solve::rho_optimizer::OuterCriterionCertificate>,
) -> SurvivalLocationScaleFitResultParts {
    SurvivalLocationScaleFitResultParts {
        training_sample_size: 32,
        beta_time: array![0.1],
        beta_threshold: array![0.2],
        beta_log_sigma: array![0.0],
        beta_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        // The time block carries one real smoothing parameter. Both gates these
        // fixtures exist to exercise — "outer iterations ran without an analytic
        // stationarity certificate" and the preservation of a carried
        // certificate — live behind `!log_lambdas.is_empty()` in
        // `FitConvergenceEvidence::try_from_parts`, because a zero-dimensional
        // outer certificate (|g|=|Pg|=bound=0) proves no equation and
        // dimensionality, not the driver's iteration counter, is the semantic
        // authority there. With all three lambda blocks empty these parts took
        // the `Fixed` branch instead: `outer_iterations` was canonicalized to 0
        // and `criterion_certificate` erased, so neither gate was reachable and
        // both fixtures measured the lambda-free normalization rather than the
        // outer-evidence contract they name.
        lambdas_time: array![1.0],
        lambdas_threshold: Array1::zeros(0),
        lambdas_log_sigma: Array1::zeros(0),
        lambdas_linkwiggle: None,
        log_likelihood: -1.0,
        reml_score: Some(1.0),
        stable_penalty_term: 0.0,
        penalized_objective: Some(1.0),
        used_device: false,
        outer_iterations,
        outer_gradient_norm: Some(0.0),
        criterion_certificate,
        outer_converged: true,
        covariance_conditional: None,
        covariance_corrected: None,
        smoothing_correction: None,
        smoothing_correction_absence: None,
        geometry: None,
        penalty_block_trace: Vec::new(),
        edf_by_block: Vec::new(),
        edf_rank_bound: Vec::new(),
        coefficient_mode_selection: Default::default(),
    }
}

fn certified_survival_fit_quadratic() -> gam_solve::rho_optimizer::CertifiedOuterResult {
    use gam_problem::{DeclaredHessianForm, Derivative, HessianValue, OuterEval};
    use gam_solve::rho_optimizer::OuterProblem;

    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_tolerance(1.0e-8)
        .with_max_iter(40)
        .with_initial_rho(array![0.5]);
    let mut objective = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * (theta[0] - 0.25).powi(2)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * (theta[0] - 0.25).powi(2),
                gradient: array![theta[0] - 0.25],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<
            fn(
                &mut (),
                &Array1<f64>,
            ) -> Result<gam_problem::EfsEval, gam_solve::estimate::EstimationError>,
        >,
    );
    problem
        .run_certified(&mut objective, "survival fit finalization certificate fixture")
        .expect("a real convex outer solve must issue the preservation proof")
}

#[test]
fn survival_fit_finalization_rejects_dropped_outer_certificate() {
    let error = survival_fit_from_parts(survival_fit_parts_with_outer_evidence(2, None))
        .expect_err("positive outer iterations without their certificate must not mint a fit");
    assert!(
        error.contains("analytic stationarity certificate"),
        "unexpected missing-certificate error: {error}"
    );
}

#[test]
fn survival_fit_finalization_preserves_outer_certificate() {
    let certified_outer = certified_survival_fit_quadratic();
    let outer_iterations = certified_outer.iterations();
    assert!(
        outer_iterations > 0,
        "certificate fixture must exercise nontrivial outer-work preservation"
    );
    let certificate = certified_outer.criterion_certificate().clone();
    let expected = certificate.summary();
    let fit = survival_fit_from_parts(survival_fit_parts_with_outer_evidence(
        outer_iterations,
        Some(certificate),
    ))
    .expect("a carried certifying outer proof must survive finalization");

    assert_eq!(fit.outer_iterations, outer_iterations);
    assert_eq!(
        fit.convergence_evidence()
            .outer_certificate()
            .expect("finalized fit must retain analytic outer evidence")
            .summary(),
        expected
    );
    assert_eq!(
        fit.artifacts
            .criterion_certificate
            .as_ref()
            .expect("fit artifacts must retain the same outer evidence")
            .summary(),
        expected
    );
}

/// gam#2661: finalization carries the inner fit's mode-selection record rather than
/// defaulting it to `NotRecorded`.
#[test]
fn survival_fit_finalization_preserves_the_mode_selection_record_2661() {
    let selection = gam_solve::model_types::CoefficientModeSelection::AnchoredContinuation {
        steps: 4,
        endpoint_discrepancy: 1.0e-9,
    };
    let mut parts = survival_fit_parts_with_outer_evidence(0, None);
    parts.coefficient_mode_selection = selection.clone();
    let fit = survival_fit_from_parts(parts).expect("a fixed-outer survival fit finalizes");
    assert_eq!(fit.artifacts.coefficient_mode_selection, selection);
}

fn survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
    SurvivalLocationScaleFamily {
        n: 3,
        y: array![1.0, 0.0, 1.0],
        w: array![1.0, 0.8, 1.2],
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: 1e-8,
        x_time_entry: Arc::new(array![[1.0], [1.0], [1.0]]),
        x_time_exit: Arc::new(array![[1.2], [0.9], [1.4]]),
        x_time_deriv: Arc::new(array![[1.0], [1.0], [1.0]]),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [0.4],
            [-0.6]
        ])),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [-0.3],
            [0.5]
        ])),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        entry_active: Arc::from(vec![true; 3]),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: true,
    }
}

fn survival_exact_newton_test_states(
    family: &SurvivalLocationScaleFamily,
    beta_t: f64,
    beta_thr: f64,
    beta_ls: f64,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    // Stacked time eta layout is `[entry; exit; deriv]` (gam#1396): the entry
    // channel occupies `0..n`, the exit channel `n..2n`, matching the solver
    // design's `MultiChannelOperator` stacking and `validate_joint_states`.
    let mut eta_time = Array1::<f64>::zeros(3 * n);
    for i in 0..n {
        eta_time[i] = family.x_time_entry[[i, 0]] * beta_t;
        eta_time[n + i] = family.x_time_exit[[i, 0]] * beta_t;
        eta_time[2 * n + i] = family.x_time_deriv[[i, 0]] * beta_t;
    }
    let eta_thr =
        Array1::from_iter((0..n).map(|i| family.x_threshold.dot_row(i, &array![beta_thr])));
    let eta_ls = Array1::from_iter((0..n).map(|i| family.x_log_sigma.dot_row(i, &array![beta_ls])));
    vec![
        ParameterBlockState {
            beta: array![beta_t],
            eta: eta_time,
        },
        ParameterBlockState {
            beta: array![beta_thr],
            eta: eta_thr,
        },
        ParameterBlockState {
            beta: array![beta_ls],
            eta: eta_ls,
        },
    ]
}

/// Total data-fit log-likelihood `ℓ = Σ_i w_i·log L_i` of the survival
/// location-scale family at the given block states, evaluated with an
/// arbitrary inverse link (the rest of the family fixed). Mirrors the
/// `offset_channel_geometry` row loop: the dynamic geometry (u0 = h0+q0,
/// u1 = h1+q1) depends only on the block states, so swapping the link
/// re-evaluates only the kernel coefficients. Used to finite-difference the
/// inverse-link data-fit θ-gradient.
fn survival_ls_total_log_likelihood_with_link(
    family: &SurvivalLocationScaleFamily,
    block_states: &[ParameterBlockState],
    link: &InverseLink,
) -> f64 {
    let mut probe = family.clone();
    probe.inverse_link = link.clone();
    let dynamic = probe
        .build_dynamic_geometry(block_states)
        .expect("dynamic geometry");
    let mut ll = 0.0;
    for i in 0..probe.n {
        if probe.w[i] <= 0.0 {
            continue;
        }
        let state = probe.row_predictor_state_at(&dynamic, i);
        if let Some(kernel) = probe.exact_row_kernel(i, state).expect("row kernel") {
            ll += kernel.log_likelihood_at(&state);
        }
    }
    ll
}

/// FD check for `SurvivalLocationScaleFamily::link_param_data_fit_gradient`:
/// the analytic `∂(−ℓ)/∂θ_link` for the SAS link `(ε, log δ)` must match a
/// central difference of the data-fit `−ℓ` at fixed β. This is the exact
/// data-fit term of the inverse-link profile-NLL gradient.
#[test]
fn link_param_data_fit_gradient_matches_finite_difference_sas() {
    let mut family = survival_exact_newton_test_family();
    let epsilon0 = 0.15;
    let log_delta0 = -0.25;
    family.inverse_link = InverseLink::Sas(
        state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon0,
            initial_log_delta: log_delta0,
        })
        .expect("sas state"),
    );
    let states = survival_exact_newton_test_states(&family, 0.35, 0.3, -0.1);

    let analytic = family
        .link_param_data_fit_gradient(&states)
        .expect("link param data-fit gradient")
        .expect("SAS link has free parameters");
    assert_eq!(analytic.len(), 2, "SAS link has two parameters (ε, log δ)");

    let neg_ll = |epsilon: f64, log_delta: f64| -> f64 {
        let link = InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .expect("sas state"),
        );
        -survival_ls_total_log_likelihood_with_link(&family, &states, &link)
    };

    let h = 1e-6;
    let fd_epsilon =
        (neg_ll(epsilon0 + h, log_delta0) - neg_ll(epsilon0 - h, log_delta0)) / (2.0 * h);
    let fd_log_delta =
        (neg_ll(epsilon0, log_delta0 + h) - neg_ll(epsilon0, log_delta0 - h)) / (2.0 * h);

    assert!(
        (analytic[0] - fd_epsilon).abs() <= 1e-5 * fd_epsilon.abs().max(1.0),
        "∂(−ℓ)/∂ε mismatch: analytic={}, fd={}",
        analytic[0],
        fd_epsilon
    );
    assert!(
        (analytic[1] - fd_log_delta).abs() <= 1e-5 * fd_log_delta.abs().max(1.0),
        "∂(−ℓ)/∂log δ mismatch: analytic={}, fd={}",
        analytic[1],
        fd_log_delta
    );
}

/// FD check for `SurvivalLocationScaleFamily::link_param_joint_psi_terms`
/// (#2904): each SAS shape axis's `score_psi` and `hessian_psi` must match a
/// central difference over that parameter of the row-kernel NLL gradient and
/// Hessian at fixed β, and its `objective_psi` must be the data-fit θ-gradient.
#[test]
fn link_param_joint_psi_terms_match_finite_difference_sas_2904() {
    let mut family = survival_exact_newton_test_family();
    let (epsilon0, log_delta0) = (0.15, -0.25);
    let sas = |epsilon: f64, log_delta: f64| {
        InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .expect("sas state"),
        )
    };
    family.inverse_link = sas(epsilon0, log_delta0);
    let states = survival_exact_newton_test_states(&family, 0.35, 0.3, -0.1);
    let data_fit = family
        .link_param_data_fit_gradient(&states)
        .expect("link param data-fit gradient")
        .expect("SAS link has free parameters");
    let nll_gradient_and_hessian = |link: InverseLink| {
        let mut probe = family.clone();
        probe.inverse_link = link;
        let dynamic = probe
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = probe.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        let p = *kernel.offsets.last().expect("joint block offsets");
        let mut gradient = Array1::<f64>::zeros(p);
        let mut hessian = Array2::<f64>::zeros((p, p));
        for row in 0..probe.n {
            let (_, row_gradient, row_hessian) =
                crate::row_kernel::RowKernel::<SLS_ROW_K>::row_kernel(&kernel, row)
                    .expect("row kernel");
            crate::row_kernel::RowKernel::<SLS_ROW_K>::jacobian_transpose_action(
                &kernel,
                row,
                &row_gradient,
                gradient.as_slice_mut().expect("contiguous gradient"),
            );
            crate::row_kernel::RowKernel::<SLS_ROW_K>::add_pullback_hessian(
                &kernel,
                row,
                &row_hessian,
                &mut hessian,
            );
        }
        (gradient, hessian)
    };
    let h = 1e-6;
    let probes = [
        (sas(epsilon0 + h, log_delta0), sas(epsilon0 - h, log_delta0)),
        (sas(epsilon0, log_delta0 + h), sas(epsilon0, log_delta0 - h)),
    ];
    for (axis, (plus, minus)) in probes.into_iter().enumerate() {
        let terms = family
            .link_param_joint_psi_terms(&states, axis)
            .expect("link-shape psi terms")
            .expect("SAS link has free parameters");
        assert_eq!(terms.objective_psi, data_fit[axis]);
        let (gradient_plus, hessian_plus) = nll_gradient_and_hessian(plus);
        let (gradient_minus, hessian_minus) = nll_gradient_and_hessian(minus);
        let fd_score = (&gradient_plus - &gradient_minus) / (2.0 * h);
        let fd_hessian = (&hessian_plus - &hessian_minus) / (2.0 * h);
        for (analytic, fd) in terms.score_psi.iter().zip(fd_score.iter()) {
            assert!(
                (analytic - fd).abs() <= 1e-5 * fd.abs().max(1.0),
                "axis {axis} score_psi mismatch: analytic={analytic}, fd={fd}"
            );
        }
        for (analytic, fd) in terms.hessian_psi.iter().zip(fd_hessian.iter()) {
            assert!(
                (analytic - fd).abs() <= 1e-5 * fd.abs().max(1.0),
                "axis {axis} hessian_psi mismatch: analytic={analytic}, fd={fd}"
            );
        }
    }
}

/// FD check for
/// `SurvivalLocationScaleFamily::link_param_joint_psihessian_directional_derivative`
/// (#2904): along a coefficient direction `u`, each SAS shape axis's mixed drift
/// `D_β H_θ[u]` must match a central difference over that parameter of the
/// observed-information drift `D_β H[u]` at fixed β.
#[test]
fn link_param_joint_psihessian_directional_derivative_matches_finite_difference_sas_2904() {
    let mut family = survival_exact_newton_test_family();
    let (epsilon0, log_delta0) = (0.15, -0.25);
    let sas = |epsilon: f64, log_delta: f64| {
        InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .expect("sas state"),
        )
    };
    family.inverse_link = sas(epsilon0, log_delta0);
    let states = survival_exact_newton_test_states(&family, 0.35, 0.3, -0.1);
    let direction = array![0.4, -0.7, 0.25];
    let drift_at = |link: InverseLink| {
        let mut probe = family.clone();
        probe.inverse_link = link;
        probe
            .exact_newton_joint_hessian_directional_derivative_rescaled(&states, &direction, 0.0)
            .expect("observed-information drift")
            .expect("survival location-scale serves the observed-information drift")
    };
    let h = 1e-6;
    let probes = [
        (sas(epsilon0 + h, log_delta0), sas(epsilon0 - h, log_delta0)),
        (sas(epsilon0, log_delta0 + h), sas(epsilon0, log_delta0 - h)),
    ];
    for (axis, (plus, minus)) in probes.into_iter().enumerate() {
        let analytic = family
            .link_param_joint_psihessian_directional_derivative(
                &states,
                axis,
                direction.as_slice().expect("contiguous direction"),
                &crate::row_kernel::RowSet::All,
            )
            .expect("link-shape mixed drift")
            .expect("SAS link has free parameters");
        let finite_difference = (&drift_at(plus) - &drift_at(minus)) / (2.0 * h);
        let scale = finite_difference
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            scale > 1e-8,
            "axis {axis}: the mixed drift is not exercised (max {scale:e})"
        );
        for (analytic, fd) in analytic.iter().zip(finite_difference.iter()) {
            assert!(
                (analytic - fd).abs() <= 1e-5 * fd.abs().max(1.0),
                "axis {axis} mixed drift mismatch: analytic={analytic}, fd={fd}"
            );
        }
    }
}

/// #2904: the exact-joint LAML gradient along the SAS shape axes is the
/// derivative of the LAML value. The family serves the shape as family-owned
/// hyper coordinates evaluated gradient-only, and a central difference of the
/// profiled objective over each parameter, at the same ρ and warm-started from
/// the base mode, reproduces the analytic component.
#[test]
fn survival_location_scale_outer_link_shape_gradient_matches_finite_difference_sas_2904() {
    let (epsilon0, log_delta0) = (0.15, -0.25);
    let family_at = |epsilon: f64, log_delta: f64| {
        survival_exact_newton_test_familywith_inverse_link(InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .expect("sas state"),
        ))
    };
    let layout_at = |epsilon: f64, log_delta: f64| {
        crate::custom_family::CustomFamilyHyperLayout::new(
            vec![Vec::new(), Vec::new(), Vec::new()],
            vec![0, 1],
            array![epsilon, log_delta],
        )
        .expect("shape-axis hyper layout")
    };
    // Every block carries a ridge. With the threshold and log-σ blocks unpenalized,
    // job 582344 measured the base inner solve stalling after 20 cycles at residual
    // 19.6 against a tolerance of 5e-6, so no gradient was ever compared.
    let mut specs = survival_outergradient_testspecs();
    for spec in specs.iter_mut().skip(1) {
        spec.penalties = vec![PenaltyMatrix::Dense(Array2::eye(1))];
        spec.initial_log_lambdas = array![0.0];
    }
    let rho = array![0.0, 0.0, 0.0];
    let options = crate::custom_family::BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ..crate::custom_family::BlockwiseFitOptions::default()
    };
    let base = crate::custom_family::evaluate_custom_family_joint_hyper_owned(
        &family_at(epsilon0, log_delta0),
        &specs,
        &options,
        &rho,
        &layout_at(epsilon0, log_delta0),
        None,
        gam_problem::EvalMode::ValueAndGradient,
    )
    .expect("exact-joint LAML value and gradient over the shape axes")
    .result;
    assert!(base.inner_converged, "the base inner solve did not converge");
    assert_eq!(base.gradient.len(), rho.len() + 2);
    let value_at = |epsilon: f64, log_delta: f64, rho: &Array1<f64>| {
        let probe = crate::custom_family::evaluate_custom_family_joint_hyper_owned(
            &family_at(epsilon, log_delta),
            &specs,
            &options,
            rho,
            &layout_at(epsilon, log_delta),
            Some(&base.warm_start),
            gam_problem::EvalMode::ValueOnly,
        )
        .expect("exact-joint LAML value over the shape axes")
        .result;
        assert!(probe.inner_converged, "a probe inner solve did not converge");
        probe.objective
    };
    // The base point again, warm-started from the base mode, value-only and with
    // the gradient. An objective or gradient that moves under a warm re-solve came
    // from a state the base solve had not settled; a value-only objective that
    // differs from the value-and-gradient objective at the same mode is a value
    // path that disagrees with the gradient path.
    // The fixture bounds its one time coefficient below at 0. A base mode on that
    // bound evaluates the Laplace criterion on the active face's tangent, so the
    // face this pin differentiates across is printed before anything is asserted.
    let base_time_coefficient = base
        .warm_start
        .block_beta_view(0)
        .expect("the time block's coefficients")[0];
    eprintln!(
        "[2695] base time coefficient={base_time_coefficient:.12e} (lower bound 0; on the bound: {})",
        base_time_coefficient <= f64::EPSILON
    );
    let warm_value = value_at(epsilon0, log_delta0, &rho);
    let warm = crate::custom_family::evaluate_custom_family_joint_hyper_owned(
        &family_at(epsilon0, log_delta0),
        &specs,
        &options,
        &rho,
        &layout_at(epsilon0, log_delta0),
        Some(&base.warm_start),
        gam_problem::EvalMode::ValueAndGradient,
    )
    .expect("warm exact-joint LAML value and gradient at the base point")
    .result;
    eprintln!(
        "[2695] base objective={:.12e}, warm value-only={:.12e}, warm value+gradient={:.12e} \
         (base converged {}, warm converged {})",
        base.objective, warm_value, warm.objective, base.inner_converged, warm.inner_converged
    );
    for k in 0..base.gradient.len() {
        eprintln!(
            "[2695] component {k}: base gradient={:.9e}, warm re-solve gradient={:.9e}",
            base.gradient[k], warm.gradient[k]
        );
    }
    // Component by component. The base point is evaluated again with the rho outer
    // audit armed, which records every rho coordinate's gradient split into the
    // criterion's parts; each ± value probe is armed for the criterion's four value
    // components. Each part is printed beside the central difference of its own
    // component before anything is asserted, so the misdifferentiated component is
    // named. The part count is printed first: an audit that records nothing shows.
    {
        use gam_solve::estimate::outer_eval_capture::{enable_rho_outer_audit, take_rho_outer_audit};
        enable_rho_outer_audit();
        crate::custom_family::evaluate_custom_family_joint_hyper_owned(
            &family_at(epsilon0, log_delta0),
            &specs,
            &options,
            &rho,
            &layout_at(epsilon0, log_delta0),
            Some(&base.warm_start),
            gam_problem::EvalMode::ValueAndGradient,
        )
        .expect("audited exact-joint LAML value and gradient at the base point");
        let audit = take_rho_outer_audit().expect("rho outer audit armed");
        eprintln!(
            "[2695] audit: {} rho gradient parts recorded, criterion recorded: {}",
            audit.parts.len(),
            audit.criterion.is_some()
        );
        let components_at = |rho_probe: &Array1<f64>| {
            enable_rho_outer_audit();
            value_at(epsilon0, log_delta0, rho_probe);
            take_rho_outer_audit()
                .and_then(|probe_audit| probe_audit.criterion)
                .map(|(cost, components)| (cost, components))
        };
        // Raw values, not only their differences: whether each probe's inner solve
        // moved the coefficients off the base mode, and what the kkt and logdet_h
        // components are at the base and at each probe.
        let beta_at = |rho_probe: &Array1<f64>| -> [f64; 3] {
            let probe = crate::custom_family::evaluate_custom_family_joint_hyper_owned(
                &family_at(epsilon0, log_delta0),
                &specs,
                &options,
                rho_probe,
                &layout_at(epsilon0, log_delta0),
                Some(&base.warm_start),
                gam_problem::EvalMode::ValueOnly,
            )
            .expect("coefficient probe")
            .result;
            let coefficient = |block: usize| {
                probe.warm_start.block_beta_view(block).expect("block coefficients")[0]
            };
            [coefficient(0), coefficient(1), coefficient(2)]
        };
        let base_beta = [0, 1, 2].map(|block| {
            base.warm_start.block_beta_view(block).expect("block coefficients")[0]
        });
        let base_components = audit.criterion.map_or([f64::NAN; 4], |(_, components)| components);
        eprintln!(
            "[2695] base components fixed_beta={:.12e} logdet_h={:.12e} logdet_s={:.12e} \
             kkt={:.12e}; base beta={:.12e},{:.12e},{:.12e}",
            base_components[0],
            base_components[1],
            base_components[2],
            base_components[3],
            base_beta[0],
            base_beta[1],
            base_beta[2]
        );
        let step = 1e-4;
        for part in &audit.parts {
            let k = part.index;
            if k >= rho.len() {
                continue;
            }
            let mut plus = rho.clone();
            plus[k] += step;
            let mut minus = rho.clone();
            minus[k] -= step;
            match (components_at(&plus), components_at(&minus)) {
                (Some((cost_plus, plus_components)), Some((cost_minus, minus_components))) => {
                    let plus_beta = beta_at(&plus);
                    let minus_beta = beta_at(&minus);
                    eprintln!(
                        "[2695] rho {k} values: kkt base={:.12e} plus={:.12e} minus={:.12e}; \
                         logdet_h plus={:.12e} minus={:.12e}; beta plus-base={:.3e},{:.3e},{:.3e} \
                         minus-base={:.3e},{:.3e},{:.3e}",
                        base_components[3],
                        plus_components[3],
                        minus_components[3],
                        plus_components[1],
                        minus_components[1],
                        plus_beta[0] - base_beta[0],
                        plus_beta[1] - base_beta[1],
                        plus_beta[2] - base_beta[2],
                        minus_beta[0] - base_beta[0],
                        minus_beta[1] - base_beta[1],
                        minus_beta[2] - base_beta[2]
                    );
                    let fd = |idx: usize| (plus_components[idx] - minus_components[idx]) / (2.0 * step);
                    let analytic_kkt = part.total - (part.fixed_beta + part.logdet_h + part.logdet_s);
                    eprintln!(
                        "[2695] rho {k}: total analytic={:.9e} fd={:.9e}; fixed_beta analytic={:.9e} \
                         fd={:.9e}; logdet_h analytic={:.9e} (frozen {:.9e}, mode response {:.9e}) \
                         fd={:.9e}; logdet_s analytic={:.9e} fd={:.9e}; kkt analytic={:.9e} fd={:.9e}",
                        part.total,
                        (cost_plus - cost_minus) / (2.0 * step),
                        part.fixed_beta,
                        fd(0),
                        part.logdet_h,
                        part.frozen_logdet_h,
                        part.mode_response_logdet_h,
                        fd(1),
                        part.logdet_s,
                        fd(2),
                        analytic_kkt,
                        fd(3)
                    );
                }
                _ => eprintln!(
                    "[2695] rho {k}: a probe recorded no criterion components; analytic \
                     total={:.9e} fixed_beta={:.9e} logdet_h={:.9e} (frozen {:.9e}, mode \
                     response {:.9e}) logdet_s={:.9e}",
                    part.total,
                    part.fixed_beta,
                    part.logdet_h,
                    part.frozen_logdet_h,
                    part.mode_response_logdet_h,
                    part.logdet_s
                ),
            }
        }
    }
    // Discriminator: the rho 0 central difference with probes solved to a
    // tolerance tight enough that no one-step KKT correction remains to price.
    // With r → 0 the difference needs neither IFT correction term, so it is the
    // profiled derivative the analytic gradient must equal.
    {
        use gam_solve::estimate::outer_eval_capture::{enable_rho_outer_audit, take_rho_outer_audit};
        let tight_options = crate::custom_family::BlockwiseFitOptions {
            inner_tol: 1e-12,
            outer_tol: 1e-12,
            ..options.clone()
        };
        let tight_at = |rho_probe: &Array1<f64>, mode: gam_problem::EvalMode| {
            enable_rho_outer_audit();
            let probe = crate::custom_family::evaluate_custom_family_joint_hyper_owned(
                &family_at(epsilon0, log_delta0),
                &specs,
                &tight_options,
                rho_probe,
                &layout_at(epsilon0, log_delta0),
                Some(&base.warm_start),
                mode,
            )
            .expect("tight exact-joint LAML evaluation")
            .result;
            let kkt = take_rho_outer_audit()
                .and_then(|probe_audit| probe_audit.criterion)
                .map_or(f64::NAN, |(_, components)| components[3]);
            (probe.objective, kkt, probe.inner_converged, probe.gradient)
        };
        let (tight_base_value, tight_base_kkt, tight_base_converged, tight_base_gradient) =
            tight_at(&rho, gam_problem::EvalMode::ValueAndGradient);
        for step in [1e-4, 1e-3] {
            let mut plus = rho.clone();
            plus[0] += step;
            let mut minus = rho.clone();
            minus[0] -= step;
            let (plus_value, plus_kkt, plus_converged, plus_gradient) =
                tight_at(&plus, gam_problem::EvalMode::ValueOnly);
            let (minus_value, minus_kkt, minus_converged, minus_gradient) =
                tight_at(&minus, gam_problem::EvalMode::ValueOnly);
            eprintln!(
                "[2695] tight rho 0 h={step:e}: fd={:.9e} (kkt plus={:.3e} minus={:.3e}, \
                 converged {} {}, gradient lengths {} {}); tight base value={:.12e} kkt={:.3e} \
                 converged {}, analytic rho 0={:.9e}",
                (plus_value - minus_value) / (2.0 * step),
                plus_kkt,
                minus_kkt,
                plus_converged,
                minus_converged,
                plus_gradient.len(),
                minus_gradient.len(),
                tight_base_value,
                tight_base_kkt,
                tight_base_converged,
                tight_base_gradient.get(0).copied().unwrap_or(f64::NAN)
            );
        }
    }
    let shape_difference = |axis: usize, h: f64| {
        let (plus, minus) = if axis == 0 {
            ((epsilon0 + h, log_delta0), (epsilon0 - h, log_delta0))
        } else {
            ((epsilon0, log_delta0 + h), (epsilon0, log_delta0 - h))
        };
        (value_at(plus.0, plus.1, &rho) - value_at(minus.0, minus.1, &rho)) / (2.0 * h)
    };
    let rho_difference = |k: usize, h: f64| {
        let mut plus = rho.clone();
        plus[k] += h;
        let mut minus = rho.clone();
        minus[k] -= h;
        (value_at(epsilon0, log_delta0, &plus) - value_at(epsilon0, log_delta0, &minus)) / (2.0 * h)
    };
    // Every component is printed before anything is asserted. The ρ components go
    // through the same evaluation path as the shape components, so they separate a
    // shape-axis defect from an evaluation-path defect. Each difference is taken at
    // two steps, so a step-size artefact shows as disagreement between them.
    let h = 1e-4;
    let mut mismatches = Vec::new();
    for k in 0..rho.len() + 2 {
        let (label, finite_difference, coarse) = if k < rho.len() {
            ("rho", rho_difference(k, h), rho_difference(k, 10.0 * h))
        } else {
            let axis = k - rho.len();
            ("shape", shape_difference(axis, h), shape_difference(axis, 10.0 * h))
        };
        let analytic = base.gradient[k];
        eprintln!(
            "[2904] {label} component {k}: outer gradient={analytic:.9e}, finite difference \
             h={h:e} {finite_difference:.9e}, h={:e} {coarse:.9e}",
            10.0 * h
        );
        if (analytic - finite_difference).abs() > 1e-4 * finite_difference.abs().max(1.0) {
            mismatches.push(format!(
                "{label} component {k}: outer gradient={analytic}, finite difference={finite_difference}"
            ));
        }
    }
    assert!(mismatches.is_empty(), "outer gradient mismatches: {mismatches:?}");
}

/// Build a single-row survival LS family with the production default
/// derivative guard (1e-6) for monotonicity-floor probes.
fn survival_ls_default_guard_unit_family() -> SurvivalLocationScaleFamily {
    SurvivalLocationScaleFamily {
        n: 1,
        y: array![1.0],
        w: array![1.0],
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        x_time_entry: Arc::new(array![[1.0]]),
        x_time_exit: Arc::new(array![[1.0]]),
        x_time_deriv: Arc::new(array![[1.0]]),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        entry_active: Arc::from(vec![true; 1]),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: true,
    }
}

/// #1396 regression: the event Jacobian `g = d_raw + qdot` is formed as a
/// compensated subtraction of two near-equal-magnitude, opposite-sign operands
/// (the constrained `d_raw` and the unconstrained threshold/log-σ `qdot`). At a
/// feasible monotone boundary that cancellation can tip the reconstructed `g` a
/// hair below zero — strictly smaller in magnitude than the derivative guard —
/// which the monotonicity check must FLOOR to the guard rather than rejecting as
/// a non-monotone state (the `heart_failure_structural_time_small` abort). A
/// genuinely non-monotone state (g negative by far more than the guard) must
/// still hard-error.
#[test]
fn survival_ls_monotonicity_floors_near_cancellation_negative_velocity() {
    let family = survival_ls_default_guard_unit_family();
    let guard = family.derivative_guard;

    // A near-cancellation that lands g just barely negative: d_raw and qdot are
    // O(1) and opposite-signed, differing only at the ~1e-7 level — exactly the
    // boundary-cancellation regime. `survival_predictor_state` forms
    // g = compensated_difference(d_raw, -qdot1) = d_raw + qdot1.
    let d_raw = 1.0_f64;
    let qdot1 = -(1.0_f64 + 2.0e-7); // g = d_raw + qdot1 = -2.0e-7, within the guard band
    let state = survival_predictor_state(0.1, 0.2, d_raw, -0.3, -0.3, qdot1, 0.0, true);
    assert!(
        state.g < 0.0 && state.g.abs() < guard,
        "fixture must produce a tiny-negative velocity inside the guard band: g={}, guard={guard}",
        state.g,
    );
    let kernel = family
        .exact_row_kernel(0, state)
        .expect("near-cancellation negative velocity must be floored, not rejected")
        .expect("positive-weight row");
    // The #1396 property is that the row is ACCEPTED with a finite `log g`.
    // Since gam#2695 the guarded branch is the continued logarithm rather than
    // a substitution of `guard` for `g`, so the value is finite and BELOW
    // `log(guard)` — the whole point of the continuation is that leaving the
    // feasible region costs something instead of being free — and it is the
    // guarded channel's own value at this `g`, not at a different one.
    let expected =
        SurvivalLocationScaleFamily::log_with_derivatives_guarded(state.g, guard).0;
    assert!(
        kernel.log_g.is_finite() && (kernel.log_g - expected).abs() <= 1e-12,
        "a near-cancellation negative velocity must be admitted with the guarded \
         channel's own value: log_g={}, expected {expected}",
        kernel.log_g,
    );
    assert!(
        kernel.log_g < guard.ln(),
        "the continued branch must charge for g={} being below guard={guard}: \
         log_g={} is not below log(guard)={}",
        state.g,
        kernel.log_g,
        guard.ln(),
    );

    // A genuinely non-monotone state (g negative by far more than the guard)
    // must still be rejected — the floor does not mask real violations.
    let bad_state = survival_predictor_state(0.1, 0.2, 1.0, -0.3, -0.3, -1.5, 0.0, true);
    assert!(
        bad_state.g < -guard,
        "fixture must produce a large-negative velocity below -guard: g={}",
        bad_state.g,
    );
    let err = family
        .exact_row_kernel(0, bad_state)
        .expect_err("a genuinely non-monotone velocity must hard-error");
    assert!(
        err.contains("monotonicity violated"),
        "unexpected error for non-monotone velocity: {err}",
    );
}

/// Build a fully time-varying, non-wiggle survival LS family whose three
/// blocks are single-column designs carrying the fixture primaries
/// verbatim (every block coefficient is 1), so all nine kernel channels —
/// including the entry and derivative threshold/log-sigma channels — are
/// live and mutually distinct.
fn survival_ls_joint_oracle_family(
    inverse_link: &InverseLink,
    primaries: &[[f64; SLS_ROW_K]],
    event: &[f64],
    weight: &[f64],
) -> SurvivalLocationScaleFamily {
    let n = primaries.len();
    let col = |ch: usize| Array2::from_shape_fn((n, 1), |(r, _)| primaries[r][ch]);
    let dense =
        |ch: usize| DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(col(ch)));
    SurvivalLocationScaleFamily {
        n,
        y: Array1::from(event.to_vec()),
        w: Array1::from(weight.to_vec()),
        inverse_link: inverse_link.clone(),
        derivative_guard: 1e-8,
        x_time_entry: Arc::new(col(0)),
        x_time_exit: Arc::new(col(1)),
        x_time_deriv: Arc::new(col(2)),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: dense(3),
        x_threshold_entry: Some(dense(4)),
        x_threshold_deriv: Some(dense(5)),
        x_log_sigma: dense(6),
        x_log_sigma_entry: Some(dense(7)),
        x_log_sigma_deriv: Some(dense(8)),
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        entry_active: Arc::from(vec![true; n]),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: true,
    }
}

/// Block states matching [`survival_ls_joint_oracle_family`]: every block
/// coefficient is 1, and the eta vectors carry the stacked layout
/// `validate_joint_states` expects for time-varying blocks. The time block
/// stacks `[entry; exit; derivative]` (matching the solver design's
/// `MultiChannelOperator` order), while the threshold / log-sigma blocks stack
/// `[exit; entry; derivative]` — exactly the conventions the production
/// `prepare.rs` stacking and `validate_joint_states` slicing use (gam#1396).
fn survival_ls_joint_oracle_states(primaries: &[[f64; SLS_ROW_K]]) -> Vec<ParameterBlockState> {
    let n = primaries.len();
    let stacked = |first: usize, second: usize, deriv: usize| {
        let mut eta = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            eta[i] = primaries[i][first];
            eta[n + i] = primaries[i][second];
            eta[2 * n + i] = primaries[i][deriv];
        }
        eta
    };
    vec![
        // Time block: `[entry(ch0); exit(ch1); deriv(ch2)]`.
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(0, 1, 2),
        },
        // Threshold block: `[exit(ch3); entry(ch4); deriv(ch5)]`.
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(3, 4, 5),
        },
        // Log-sigma block: `[exit(ch6); entry(ch7); deriv(ch8)]`.
        ParameterBlockState {
            beta: array![1.0],
            eta: stacked(6, 7, 8),
        },
    ]
}

/// gam#3035: the all-axes dense overrides agree with the generic per-row
/// reductions on the full data and on a Horvitz–Thompson-weighted subsample.
#[test]
fn survival_ls_dense_overrides_match_generic_on_every_row_set_3035() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let primaries: Vec<[f64; SLS_ROW_K]> = vec![
                [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
                [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
                [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
                [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
            ];
            let event = [1.0, 1.0, 1.0, 0.35];
            let weight = [1.0, 1.2, 1.1, 1.3];
            for distribution in [
                ResidualDistribution::Gaussian,
                ResidualDistribution::Gumbel,
                ResidualDistribution::Logistic,
            ] {
                let inverse_link = residual_distribution_inverse_link(distribution);
                let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
                let states = survival_ls_joint_oracle_states(&primaries);
                let dynamic = family.build_dynamic_geometry(&states).expect("dynamic geometry");
                let kernel = SurvivalLsRowKernel {
                    family: &family,
                    dynamic: &dynamic,
                    deriv_log_scale: 0.0,
                    offsets: family.joint_block_offsets(),
                };
                crate::row_kernel::row_set_override_tests::assert_dense_overrides_match_generic(
                    &format!("survival location-scale {distribution:?}"),
                    &kernel,
                    &[0.7, -0.5, 0.9],
                    &[-1.1, 0.8, 0.3],
                    1e-13,
                );
            }
        })
        .expect("spawn wide-stack override thread")
        .join();
    assert!(join_result.is_ok(), "survival LS override pinning thread must complete");
}

/// The hand-derived analytic joint-Hessian directional derivative
/// (`exact_newton_joint_hessian_directional_derivative_from_parts`) must agree
/// with the jet-tower-certified generic row-kernel directional derivative on a
/// FULLY TIME-VARYING family — i.e. with the derivative threshold/log-sigma
/// channels (the velocity / `qdot` coordinate) live. The pre-existing FD
/// coverage only exercises non-time-varying fixtures, where the velocity
/// coordinate is inert, so it cannot witness a dropped `qdot` third-order
/// contribution.
///
/// #932: `row_kernel_directional_supported()` is now enabled for non-wiggle
/// rows, so the row-kernel reference below is exactly the path a production fit
/// takes — and the generic row-kernel directional derivative now consumes the
/// PACKED `OneSeed<9>` scalar (no dense `Tower4<9>`). This test pins that packed
/// path against the hand path to 1e-7, so enabling the gate is behaviour-
/// preserving; both call sites are invoked explicitly here, independent of the
/// gate.
#[test]
fn survival_ls_joint_directional_derivative_matches_tower_time_varying() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_joint_directional_derivative_time_varying_body)
        .expect("spawn wide-stack directional-derivative thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS joint directional-derivative time-varying oracle thread must complete"
    );
}

fn survival_ls_joint_directional_derivative_time_varying_body() {
    use crate::row_kernel::{RowSet, row_kernel_directional_derivative_generic};

    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 0.35];
    let weight = [1.0, 1.2, 1.1, 1.3];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        let states = survival_ls_joint_oracle_states(&primaries);
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = SurvivalLsRowKernel {
            family: &family,
            dynamic: &dynamic,
            deriv_log_scale: 0.0,
            offsets: family.joint_block_offsets(),
        };
        for direction in [
            array![0.7, -0.5, 0.9],
            array![-1.1, 0.8, 0.3],
            array![0.4, 1.2, -0.6],
            array![1.0, 0.0, 0.0],
            array![0.0, 1.0, 0.0],
            array![0.0, 0.0, 1.0],
        ] {
            let dir_slice = direction.as_slice().expect("contiguous direction");
            let reference =
                row_kernel_directional_derivative_generic(&kernel, &RowSet::All, dir_slice)
                    .expect("tower-certified directional derivative");
            let hand = family
                .exact_newton_joint_hessian_directional_derivative_rescaled_from_parts(
                    &direction, &dynamic, 0.0,
                )
                .expect("hand directional derivative")
                .expect("hand directional derivative present");
            assert_eq!(reference.dim(), hand.dim(), "directional dH shape");
            for ((a, b), &want) in reference.indexed_iter() {
                let got = hand[[a, b]];
                assert!(
                    (got - want).abs() <= 1e-7 * (1.0 + want.abs()),
                    "{distribution:?} dir={direction} joint directional dH[{a}][{b}] mismatch: \
                     hand={got} tower-reference={want}"
                );
            }
        }
    }
}

fn packed_sls_dense(
    family: &SurvivalLocationScaleFamily,
    states: &[ParameterBlockState],
    deriv_log_scale: f64,
) -> Array2<f64> {
    let dynamic = family
        .build_dynamic_geometry(states)
        .expect("packed SLS dynamic geometry");
    match family
        .survival_ls_coefficient_hessian(
            &dynamic,
            deriv_log_scale,
            None,
            SlsCoefficientHessianTarget::DenseFull,
        )
        .expect("packed SLS dense Hessian")
    {
        SlsCoefficientHessian::DenseFull(dense) => dense,
        _ => panic!("dense target returned another packed SLS shape"),
    }
}

#[test]
fn survival_ls_packed_targets_apply_ht_mask_once_932() {
    use crate::row_kernel::{build_row_kernel_cache, row_kernel_hessian_dense};

    let family = survival_exact_newton_test_family();
    let states = survival_exact_newton_test_states(&family, 0.3, -0.4, 0.2);
    let dynamic = family
        .build_dynamic_geometry(&states)
        .expect("HT target dynamic geometry");
    let mask = array![2.0, 0.0, 0.5];
    let target = |shape| {
        family
            .survival_ls_coefficient_hessian(&dynamic, 0.0, Some(&mask), shape)
            .expect("masked packed SLS target")
    };
    let dense = match target(SlsCoefficientHessianTarget::DenseFull) {
        SlsCoefficientHessian::DenseFull(dense) => dense,
        _ => panic!("dense target returned another packed SLS shape"),
    };

    // Independent generic pullback: RowSet owns the one HT multiplication while
    // the ordinary RowKernel dense reducer visits exactly the selected rows.
    let kernel = family.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
    let rows = row_set_from_survival_mask(Some(&mask), family.n);
    let cache = build_row_kernel_cache(&kernel, &rows).expect("masked generic row cache");
    let generic = row_kernel_hessian_dense(&kernel, &cache, &rows)
        .expect("generic masked row-kernel Hessian");
    assert_eq!(dense.dim(), generic.dim());
    for ((row, column), &expected) in generic.indexed_iter() {
        let got = dense[[row, column]];
        assert!(
            (got - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "masked dense Hessian [{row},{column}] packed={got:.12e} generic={expected:.12e}"
        );
    }

    let blocks = match target(SlsCoefficientHessianTarget::BlockDiagonal) {
        SlsCoefficientHessian::BlockDiagonal(blocks) => blocks,
        _ => panic!("block target returned another packed SLS shape"),
    };
    let diagonal = match target(SlsCoefficientHessianTarget::DiagonalOnly) {
        SlsCoefficientHessian::DiagonalOnly(diagonal) => diagonal,
        _ => panic!("diagonal target returned another packed SLS shape"),
    };
    let offsets = family.joint_block_offsets();
    for block in 0..3 {
        let (start, end) = (offsets[block], offsets[block + 1]);
        let expected = dense.slice(s![start..end, start..end]);
        for ((row, column), &value) in blocks[block].indexed_iter() {
            assert!(
                (value - expected[[row, column]]).abs()
                    <= 1e-9 * expected[[row, column]].abs().max(1.0),
                "masked block {block} [{row},{column}] disagrees with dense target"
            );
        }
    }
    for coefficient in 0..diagonal.len() {
        assert!(
            (diagonal[coefficient] - dense[[coefficient, coefficient]]).abs()
                <= 1e-9 * dense[[coefficient, coefficient]].abs().max(1.0),
            "masked diagonal [{coefficient}] disagrees with dense target"
        );
    }
}

/// #921/#932: the packed 27-pair coefficient lowering must reproduce the
/// generic `RowKernel<9>` dense pullback. The two paths share only the canonical
/// row program: one lowers its 27 structural pairs through grouped X'WX calls,
/// while the oracle materializes the generic per-row 9×9 pullback.
#[test]
fn survival_ls_row_kernel_matches_packed_coefficient_lowering() {
    // Row-kernel assembly plus the directional-Hessian FD oracle keep several
    // dense joint Hessians live on the stack; run on a wide-stack thread like
    // the other survival-LS jet-tower oracles.
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_row_kernel_matches_packed_coefficient_lowering_body)
        .expect("spawn wide-stack row-kernel oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS row-kernel oracle thread must complete"
    );
}

fn survival_ls_row_kernel_matches_packed_coefficient_lowering_body() {
    use crate::row_kernel::{
        RowSet, build_row_kernel_cache, row_kernel_directional_derivative, row_kernel_gradient,
        row_kernel_hessian_dense, row_kernel_log_likelihood,
    };

    let family = survival_exact_newton_test_family();
    let n = family.n;
    let beta_t = 0.3_f64;
    let beta_thr = -0.4_f64;
    let beta_ls = 0.2_f64;
    let states = survival_exact_newton_test_states(&family, beta_t, beta_thr, beta_ls);

    let dynamic = family
        .build_dynamic_geometry(&states)
        .expect("dynamic geometry");
    let kernel = SurvivalLsRowKernel {
        family: &family,
        dynamic: &dynamic,
        deriv_log_scale: 0.0,
        offsets: family.joint_block_offsets(),
    };

    let cache = build_row_kernel_cache(&kernel, &RowSet::All).expect("row kernel cache");
    let h_new = row_kernel_hessian_dense(&kernel, &cache, &RowSet::All)
        .expect("generic row-kernel Hessian");
    let h_old = packed_sls_dense(&family, &states, 0.0);
    assert_eq!(h_new.dim(), h_old.dim(), "joint hessian shape");
    for ((a, b), &old) in h_old.indexed_iter() {
        let new = h_new[[a, b]];
        assert!(
            (new - old).abs() <= 1e-9 * (1.0 + old.abs()),
            "joint Hessian [{a}][{b}] mismatch: new={new}, old={old}"
        );
    }

    // Log-likelihood: the generic engine returns ℓ = -Σ nll_i; the bespoke
    // per-row log-likelihood sums `exact_row_kernel(row).log_likelihood()`.
    let ll_new = row_kernel_log_likelihood(&cache, &RowSet::All);
    let mut ll_old = 0.0;
    for i in 0..n {
        let state = family.row_predictor_state_at(&dynamic, i);
        if let Some(k) = family.exact_row_kernel(i, state).expect("row kernel") {
            ll_old += k.log_likelihood_at(&state);
        }
    }
    assert!(
        (ll_new - ll_old).abs() <= 1e-9 * (1.0 + ll_old.abs()),
        "log-likelihood mismatch: new={ll_new}, old={ll_old}"
    );

    // Gradient assembles at the right coefficient dimension.
    let g_new = row_kernel_gradient(&kernel, &cache, &RowSet::All);
    assert_eq!(g_new.len(), *kernel.offsets.last().unwrap());

    let direction = array![0.17, -0.11, 0.07];
    let d_new = row_kernel_directional_derivative(
        &kernel,
        &RowSet::All,
        direction
            .as_slice()
            .expect("literal direction is contiguous"),
    )
    .expect("row-kernel directional derivative");
    let eps = 1e-5;
    let plus = survival_exact_newton_test_states(
        &family,
        beta_t + eps * direction[0],
        beta_thr + eps * direction[1],
        beta_ls + eps * direction[2],
    );
    let minus = survival_exact_newton_test_states(
        &family,
        beta_t - eps * direction[0],
        beta_thr - eps * direction[1],
        beta_ls - eps * direction[2],
    );
    let h_plus = packed_sls_dense(&family, &plus, 0.0);
    let h_minus = packed_sls_dense(&family, &minus, 0.0);
    let d_fd = (&h_plus - &h_minus) / (2.0 * eps);
    for ((a, b), &fd) in d_fd.indexed_iter() {
        let new = d_new[[a, b]];
        assert!(
            (new - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
            "directional Hessian [{a}][{b}] mismatch: new={new}, fd={fd}"
        );
    }
}

/// #932: packed coefficient-level guard for the time-varying joint Hessian
/// across every residual distribution.
///
/// `survival_ls_row_kernel_matches_packed_coefficient_lowering` (#921) pins the
/// generic row-kernel joint Hessian (`row_kernel_hessian_dense`, sourced from
/// the once-written `sls_row_nll` through `Order2<9>`) to the packed coefficient
/// lowering. But that fixture is Gaussian-only and uses the simple block shape
/// (`x_{threshold,log_sigma}_{entry,deriv} = None`), so it only exercises the
/// smallest alias-resolved plan. Fully time-varying designs exercise all 24
/// structural pairs, including derivative-design cross terms — exactly the #736
/// dropped/sign-flipped genus — which previously had only a per-row oracle
/// (`survival_ls_joint_row_kernel_agrees_with_jet_tower_program_all_channels`),
/// never assembled into the joint matrix and compared.
///
/// This fills that gap. `survival_ls_joint_oracle_family` populates every
/// entry/deriv design, so the packed plan retains all 24 groups; the generic
/// engine builds the same joint Hessian from the single-sourced row
/// NLL. They must agree to ~1e-9 (no FD: both are analytic), for
/// Gaussian / Gumbel (Weibull AFT) / Logistic (log-logistic AFT). A dropped
/// structural pair shifts a joint entry well outside 1e-9 and fails loudly.
#[test]
fn survival_ls_time_varying_joint_hessian_matches_single_sourced_tower_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_time_varying_joint_hessian_tower_body)
        .expect("spawn wide-stack assembler-tower oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS time-varying assembler-vs-tower oracle thread must complete"
    );
}

fn survival_ls_time_varying_joint_hessian_tower_body() {
    use crate::row_kernel::{
        RowSet, build_row_kernel_cache, row_kernel_gradient, row_kernel_hessian_dense,
        row_kernel_log_likelihood,
    };

    // Same nine-channel fixture the all-channels per-row oracle uses: exact
    // deaths, right-censored rows, deep / effectively-absent left truncation,
    // extreme exit tails, and a fractional event weight — every channel and
    // cross block populated, all clear of the monotonicity guard.
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [-6.5, 5.6, 1.1, -0.7, -0.3, -0.15, 0.2, 0.4, 0.1],
        [-1.0, -5.2, 0.7, 0.5, 0.6, 0.3, -0.1, -0.3, 0.25],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 0.0, 1.0, 0.35];
    let weight = [1.0, 0.8, 1.2, 0.9, 1.1, 1.3];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        // Sanity: this fixture must drive the time-varying assembler branches.
        assert!(
            family.x_threshold_entry.is_some()
                && family.x_threshold_deriv.is_some()
                && family.x_log_sigma_entry.is_some()
                && family.x_log_sigma_deriv.is_some()
                && family.x_link_wiggle.is_none(),
            "fixture must populate every entry/deriv design and no link-wiggle so the \
             assembler takes its time-varying branches"
        );
        let states = survival_ls_joint_oracle_states(&primaries);

        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = SurvivalLsRowKernel {
            family: &family,
            dynamic: &dynamic,
            deriv_log_scale: 0.0,
            offsets: family.joint_block_offsets(),
        };

        // Single-sourced tower joint Hessian: row kernel (Order2<9> over
        // sls_row_nll) → dense block assembly.
        let cache = build_row_kernel_cache(&kernel, &RowSet::All).expect("row kernel cache");
        let h_tower = row_kernel_hessian_dense(&kernel, &cache, &RowSet::All)
            .expect("single-sourced tower row-kernel Hessian");

        let h_packed = packed_sls_dense(&family, &states, 0.0);

        assert_eq!(
            h_tower.dim(),
            h_packed.dim(),
            "{distribution:?}: joint Hessian shape mismatch"
        );
        for ((a, b), &packed) in h_packed.indexed_iter() {
            let tower = h_tower[[a, b]];
            assert!(
                (tower - packed).abs() <= 1e-9 * (1.0 + packed.abs()),
                "{distribution:?}: joint Hessian [{a}][{b}] packed {packed} != \
                 generic single-sourced tower {tower}"
            );
        }

        // Gradient: the single-sourced engine's ∇(nll) must assemble at the
        // joint coefficient dimension and stay finite (the gradient and Hessian
        // share the one cache, so a consistent triple).
        let g_tower = row_kernel_gradient(&kernel, &cache, &RowSet::All);
        assert_eq!(
            g_tower.len(),
            *kernel.offsets.last().unwrap(),
            "{distribution:?}: gradient dimension"
        );
        assert!(
            g_tower.iter().all(|v| v.is_finite()),
            "{distribution:?}: single-sourced gradient must be finite"
        );

        // Log-likelihood consistency: the engine's ℓ = −Σ nll_i must match the
        // bespoke per-row `exact_row_kernel(row).log_likelihood()` sum.
        let ll_tower = row_kernel_log_likelihood(&cache, &RowSet::All);
        let mut ll_bespoke = 0.0;
        for i in 0..family.n {
            let state = family.row_predictor_state_at(&dynamic, i);
            if let Some(k) = family.exact_row_kernel(i, state).expect("row kernel") {
                ll_bespoke += k.log_likelihood_at(&state);
            }
        }
        assert!(
            (ll_tower - ll_bespoke).abs() <= 1e-9 * (1.0 + ll_bespoke.abs()),
            "{distribution:?}: log-likelihood single-sourced {ll_tower} != bespoke {ll_bespoke}"
        );
    }
}

/// #932: the production survival-LS log-likelihood block GRADIENT
/// (`evaluate_log_likelihood_and_block_gradients` — the LIVE outer-Newton
/// gradient path; the sparse hand assembler is the live joint-Hessian path as a
/// measured perf exception) must equal the single-sourced row-kernel gradient.
///
/// The joint Hessian is now pinned to the tower (the time-varying assembler
/// oracle above + the #921 simple-shape oracle), and the gradient-vs-FD SAS
/// test covers one link, but no exact oracle pinned the bespoke block gradient
/// to `row_kernel_gradient` (built from the same `sls_row_nll` the Hessian uses)
/// across distributions and the time-varying shape. `survival_joint_gradient
/// _evaluation_matches_evaluate_block_gradients` only checks the bespoke path
/// against itself.
///
/// `row_kernel_gradient` returns ∇(nll) = −∇ℓ (the cached per-row jets are of
/// the negative log-likelihood, pulled back), while
/// `evaluate_log_likelihood_and_block_gradients` returns the log-likelihood
/// gradient ∇ℓ; both block orders are `[time, threshold, log_sigma]`
/// (`block_gradients = vec![grad_time, grad_t, grad_ls]` and
/// `joint_block_offsets`), so the flattened bespoke ∇ℓ must equal `−g_tower`
/// to ~1e-9, for Gaussian / Gumbel / Logistic on the every-channel fixture. A
/// dropped term in the hand block gradient now fails loudly.
#[test]
fn survival_ls_block_gradient_matches_single_sourced_tower_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(survival_ls_block_gradient_tower_body)
        .expect("spawn wide-stack gradient oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS block-gradient-vs-tower oracle thread must complete"
    );
}

fn survival_ls_block_gradient_tower_body() {
    use crate::row_kernel::{RowSet, build_row_kernel_cache, row_kernel_gradient};

    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [-6.5, 5.6, 1.1, -0.7, -0.3, -0.15, 0.2, 0.4, 0.1],
        [-1.0, -5.2, 0.7, 0.5, 0.6, 0.3, -0.1, -0.3, 0.25],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 0.0, 1.0, 0.35];
    let weight = [1.0, 0.8, 1.2, 0.9, 1.1, 1.3];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        let states = survival_ls_joint_oracle_states(&primaries);

        // Single-sourced tower gradient (∇nll = −∇ℓ).
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");
        let kernel = SurvivalLsRowKernel {
            family: &family,
            dynamic: &dynamic,
            deriv_log_scale: 0.0,
            offsets: family.joint_block_offsets(),
        };
        let cache = build_row_kernel_cache(&kernel, &RowSet::All).expect("row kernel cache");
        let g_tower_nll = row_kernel_gradient(&kernel, &cache, &RowSet::All);

        // Bespoke production block gradients (∇ℓ), flattened in the joint
        // [time, threshold, log_sigma] layout.
        let (_ll, block_gradients) = family
            .evaluate_log_likelihood_and_block_gradients(&states)
            .expect("bespoke block gradients");
        let offsets = family.joint_block_offsets();
        let total = *offsets.last().unwrap();
        let mut g_bespoke_ll = vec![0.0_f64; total];
        let mut pos = 0usize;
        for block in &block_gradients {
            for &v in block.iter() {
                g_bespoke_ll[pos] = v;
                pos += 1;
            }
        }
        assert_eq!(
            pos, total,
            "{distribution:?}: flattened bespoke gradient width {pos} != joint total {total}"
        );
        assert_eq!(
            g_tower_nll.len(),
            total,
            "{distribution:?}: tower gradient width"
        );

        // ∇ℓ_bespoke == −∇nll_tower.
        for i in 0..total {
            let bespoke = g_bespoke_ll[i];
            let tower = -g_tower_nll[i];
            assert!(
                (bespoke - tower).abs() <= 1e-9 * (1.0 + tower.abs()),
                "{distribution:?}: block gradient[{i}] bespoke ∇ℓ {bespoke:.9e} != \
                 single-sourced −∇nll {tower:.9e}"
            );
        }
    }
}

/// Build a location-scale family whose three coefficient blocks are each
/// `p`-columns wide (and `n`-rows) so `joint_block_dims()` == `[p, p, p]`.
/// The advertisement guards (`validate_joint_specs`) compare the spec
/// widths against `joint_block_dims()`, so the family's design widths must
/// equal the spec widths for the HVP-availability path to be exercised
/// (gam#848); the previous fixture left the family at 1-column designs
/// while building width-200 specs, so the guard correctly rejected them.
fn survival_large_scale_block_test_family(p: usize) -> SurvivalLocationScaleFamily {
    let n = 3usize;
    let mut family = survival_exact_newton_test_family();
    family.x_threshold = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::<f64>::zeros((n, p)),
    ));
    family.x_log_sigma = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::<f64>::zeros((n, p)),
    ));
    family.x_time_entry = Arc::new(Array2::<f64>::zeros((n, p)));
    family.x_time_exit = Arc::new(Array2::<f64>::zeros((n, p)));
    family.x_time_deriv = Arc::new(Array2::<f64>::zeros((n, p)));
    family
}

#[test]
fn survival_location_scale_advertises_outer_hvp_at_large_scale_dimensions() {
    let family = survival_large_scale_block_test_family(200);
    let mk_spec = |name: &str, p: usize| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((family.n, p)))),
        offset: Array1::zeros(family.n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![
        mk_spec("time", 200),
        mk_spec("threshold", 200),
        mk_spec("log_sigma", 200),
    ];

    assert!(family.outer_hyper_hessian_hvp_available(&specs));
}

#[test]
fn survival_location_scale_planner_keeps_analytic_hessian_at_large_scale_dimensions() {
    let family = survival_large_scale_block_test_family(200);
    let mk_spec = |name: &str, p: usize| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((family.n, p)))),
        offset: Array1::zeros(family.n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![
        mk_spec("time", 200),
        mk_spec("threshold", 200),
        mk_spec("log_sigma", 200),
    ];
    let options = crate::custom_family::BlockwiseFitOptions::default();

    let (gradient, hessian) =
        crate::custom_family::custom_family_outer_derivatives(&family, &specs, &options);
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(
        hessian,
        gam_problem::DeclaredHessianForm::Either,
        "large survival location-scale fits must not be demoted to BFGS when the explicit HVP operator covers the dimensions"
    );
}

mod derivative_identities;
mod time_block_identification;

fn survival_exact_newton_test_familywith_inverse_link(
    inverse_link: InverseLink,
) -> SurvivalLocationScaleFamily {
    SurvivalLocationScaleFamily {
        inverse_link,
        ..survival_exact_newton_test_family()
    }
}

fn sparse_survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
    let mut family = survival_exact_newton_test_family();
    family.x_threshold = sparse_design_from_dense(&array![[1.0], [0.4], [-0.6]]);
    family.x_log_sigma = sparse_design_from_dense(&array![[1.0], [-0.3], [0.5]]);
    family
}

fn survival_exact_newton_threshold_states(beta_threshold: f64) -> Vec<ParameterBlockState> {
    vec![
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.1, 0.35, -0.2, 0.25, 0.6, 0.15, 0.5, 0.7, 0.6],
        },
        ParameterBlockState {
            beta: array![beta_threshold],
            eta: array![beta_threshold, 0.4 * beta_threshold, -0.6 * beta_threshold],
        },
        ParameterBlockState {
            beta: array![-0.15],
            eta: array![-0.15, 0.045, -0.075],
        },
    ]
}

fn survival_exact_newton_rebuild_states(
    beta_time: &Array1<f64>,
    beta_threshold: &Array1<f64>,
    beta_log_sigma: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    vec![
        ParameterBlockState {
            beta: beta_time.clone(),
            eta: array![
                beta_time[0],
                beta_time[0],
                beta_time[0],
                1.2 * beta_time[0],
                0.9 * beta_time[0],
                1.4 * beta_time[0],
                beta_time[0] + 0.5,
                beta_time[0] + 0.7,
                beta_time[0] + 0.6
            ],
        },
        ParameterBlockState {
            beta: beta_threshold.clone(),
            eta: array![
                beta_threshold[0],
                0.4 * beta_threshold[0],
                -0.6 * beta_threshold[0]
            ],
        },
        ParameterBlockState {
            beta: beta_log_sigma.clone(),
            eta: array![
                beta_log_sigma[0],
                -0.3 * beta_log_sigma[0],
                0.5 * beta_log_sigma[0]
            ],
        },
    ]
}

fn survival_outergradient_testspecs() -> Vec<ParameterBlockSpec> {
    vec![
        ParameterBlockSpec {
            name: "time_transform".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [1.0],
                [1.0],
                [1.2],
                [0.9],
                [1.4],
                [1.0],
                [1.0],
                [1.0]
            ])),
            offset: Array1::zeros(9),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [0.4],
                [-0.6]
            ])),
            offset: Array1::zeros(3),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.35]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [-0.3],
                [0.5]
            ])),
            offset: Array1::zeros(3),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![-0.15]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ]
}

fn survival_non_probit_test_links() -> Vec<(&'static str, InverseLink)> {
    vec![
        (
            "logistic",
            residual_distribution_inverse_link(ResidualDistribution::Logistic),
        ),
        (
            "cloglog",
            residual_distribution_inverse_link(ResidualDistribution::Gumbel),
        ),
        (
            "sas",
            InverseLink::Sas(
                state_from_sasspec(SasLinkSpec {
                    initial_epsilon: 0.1,
                    initial_log_delta: -0.2,
                })
                .expect("sas state"),
            ),
        ),
        (
            "beta-logistic",
            InverseLink::BetaLogistic(
                state_from_beta_logisticspec(SasLinkSpec {
                    initial_epsilon: 0.05,
                    initial_log_delta: 0.1,
                })
                .expect("beta-logistic state"),
            ),
        ),
    ]
}

fn spec_from_dense_for_test(
    name: &str,
    design: DesignMatrix,
    gauge_priority: u8,
) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design,
        offset: Array1::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

#[test]
fn survival_log_likelihood_only_matches_sum_of_exact_row_kernels() {
    let family = survival_exact_newton_test_family();
    let states = survival_exact_newton_rebuild_states(&array![0.1], &array![0.2], &array![-0.15]);
    let (h0, h1, d_raw, ..) = family.validate_joint_states(&states).expect("joint states");
    let dynamic = family
        .build_dynamic_geometry(&states)
        .expect("dynamic geometry");

    let mut row_sum = 0.0;
    for i in 0..family.n {
        // The scale divides the time transform too (#2695): `u = h·s + q` and
        // `du1/dt = s1·g` with `g = (d_raw − h1·eta_ls') + qdot`, `s = e^{−eta_ls}`.
        let s0 = dynamic.inv_sigma_entry[i];
        let s1 = dynamic.inv_sigma_exit[i];
        let state = survival_predictor_state(
            h0[i] * s0,
            h1[i] * s1,
            d_raw[i] - h1[i] * dynamic.eta_ls_deriv_exit[i],
            dynamic.q_entry[i],
            dynamic.q_exit[i],
            dynamic.qdot_exit[i],
            -dynamic.eta_ls_exit[i],
            family.entry_active[i],
        );
        if let Some(kernel) = family.exact_row_kernel(i, state).expect("exact row kernel") {
            row_sum += kernel.log_likelihood_at(&state);
        }
    }

    let scalar = family
        .log_likelihood_only(&states)
        .expect("scalar log-likelihood");
    assert!(
        (scalar - row_sum).abs() < 1e-12,
        "scalar survival log-likelihood should equal the sum of exact row kernels; scalar={} row_sum={}",
        scalar,
        row_sum
    );
}

#[test]
fn survival_joint_gradient_evaluation_matches_evaluate_block_gradients() {
    let family = survival_exact_newton_test_family();
    let states = survival_exact_newton_rebuild_states(&array![0.2], &array![0.35], &array![-0.15]);
    let specs = survival_outergradient_testspecs();
    let joint = family
        .exact_newton_joint_gradient_evaluation(&states, &specs)
        .expect("joint gradient evaluation")
        .expect("survival location-scale should provide joint gradient");
    let eval = family.evaluate(&states).expect("full evaluate");

    assert!((joint.log_likelihood - eval.log_likelihood).abs() <= 1e-12);

    let mut expected = Array1::<f64>::zeros(joint.gradient.len());
    let mut offset = 0usize;
    for (spec, work) in specs.iter().zip(eval.blockworking_sets.iter()) {
        let width = spec.design.ncols();
        let BlockWorkingSet::ExactNewton { gradient, .. } = work else {
            panic!("survival location-scale blocks should use exact Newton");
        };
        expected
            .slice_mut(s![offset..offset + width])
            .assign(gradient);
        offset += width;
    }

    for (actual, expected) in joint.gradient.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

#[test]
fn survival_exact_row_kernel_rejects_invalid_event_target_instead_of_clamping() {
    let mut family = survival_exact_newton_test_family();
    family.y[0] = 1.5;
    let states = survival_exact_newton_rebuild_states(&array![0.1], &array![0.2], &array![-0.15]);
    let err = match family.log_likelihood_only(&states) {
        Ok(_) => panic!("invalid event target should error"),
        Err(err) => err,
    };
    assert!(
        err.contains("event target must lie in [0,1]"),
        "expected explicit event-target validation error, got: {err}"
    );
}

#[test]
fn logwith_derivatives_positive_matches_exact_log() {
    let x = 0.25;
    let (log_x, d1, d2, d3, d4) = SurvivalLocationScaleFamily::logwith_derivatives_positive(x);
    assert!((log_x - x.ln()).abs() <= 1e-15);
    assert!((d1 - 1.0 / x).abs() <= 1e-15);
    assert!((d2 + 1.0 / (x * x)).abs() <= 1e-15);
    assert!((d3 - 2.0 / (x * x * x)).abs() <= 1e-15);
    assert!((d4 + 6.0 / (x * x * x * x)).abs() <= 1e-12);
}

#[test]
fn inverse_link_survival_prob_complements_failure_prob() {
    let eta = 0.37;
    let failure = inverse_link_failure_prob_checked(
        &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        eta,
    )
    .expect("failure probability");
    let survival = inverse_link_survival_prob_checked(
        &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        eta,
    )
    .expect("survival probability");
    assert!((survival - (1.0 - failure)).abs() <= 1e-14);
}

#[test]
fn lift_conditional_covariance_rejects_time_map_wider_than_raw() {
    let z = array![[1.0, 0.0]];
    let time_gauge = Gauge::from_block_transforms(&[z]);
    let err = survival_location_scale_finalization_gauge(&time_gauge, 0, 0, 0, 0, 0, 0, None)
        .expect_err(
            "a reduced time block wider than the raw time map must fail before ndarray assignment",
        );
    assert!(
        err.contains("time map is wider than tall"),
        "unexpected covariance-lift error: {err}"
    );
}

#[test]
fn lift_conditional_covariance_preserveswiggle_block() {
    let z = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]];
    let time_gauge = Gauge::from_block_transforms(&[z]);
    let finalization_gauge =
        survival_location_scale_finalization_gauge(&time_gauge, 1, 1, 0, 1, 1, 0, Some(1))
            .expect("location-scale finalization gauge");
    let cov_reduced = array![
        [2.0, 0.1, 0.2, 0.3, 0.4],
        [0.1, 3.0, 0.5, 0.6, 0.7],
        [0.2, 0.5, 4.0, 0.8, 0.9],
        [0.3, 0.6, 0.8, 5.0, 1.1],
        [0.4, 0.7, 0.9, 1.1, 6.0],
    ];
    let lifted =
        lift_conditional_covariance(&cov_reduced, &finalization_gauge).expect("covariance lift");
    assert_eq!(lifted.dim(), (6, 6));
    assert!((lifted[[5, 5]] - 6.0).abs() <= 1e-12);
    assert!((lifted[[0, 5]] - 0.4).abs() <= 1e-12);
    assert!((lifted[[3, 5]] - 0.9).abs() <= 1e-12);
    assert!((lifted[[4, 5]] - 1.1).abs() <= 1e-12);
}

#[test]
fn finalization_gauge_preserves_absent_optional_block_topology() {
    let time_gauge = Gauge::identity(&[2]);
    let finalization_gauge =
        survival_location_scale_finalization_gauge(&time_gauge, 1, 1, 0, 1, 1, 0, None)
            .expect("three-block location-scale finalization gauge");

    assert_eq!(finalization_gauge.n_blocks(), 3);
    assert_eq!(finalization_gauge.block_starts_raw, vec![0, 2, 3, 4]);
    assert_eq!(finalization_gauge.block_starts_reduced, vec![0, 2, 3, 4]);
}

#[test]
fn finalization_gauge_composes_non_square_active_frames_and_affine_shift() {
    let time_gauge = Gauge::from_block_transform_with_shift(
        array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]],
        array![0.25, -0.5, 0.75],
    );
    let finalization_gauge =
        survival_location_scale_finalization_gauge(&time_gauge, 1, 2, 1, 1, 1, 0, Some(1))
            .expect("location-scale finalization gauge");
    assert_eq!(finalization_gauge.t_full.dim(), (7, 5));
    assert_eq!(
        finalization_gauge.affine_shift,
        array![0.25, -0.5, 0.75, 0.0, 0.0, 0.0, 0.0]
    );

    // Model the inner canonical audit removing one additional time direction.
    // Its raw partition is exactly the finalizer's active partition.
    let inner_gauge = Gauge::from_block_transforms_with_shift(
        &[
            array![[1.0], [0.25]],
            Array2::<f64>::eye(1),
            Array2::<f64>::eye(1),
            Array2::<f64>::eye(1),
        ],
        array![0.1, -0.2, 0.0, 0.0, 0.0],
    );
    let composed = inner_gauge
        .left_compose(&finalization_gauge)
        .expect("compatible non-square coefficient gauges compose");
    assert_eq!(composed.t_full.dim(), (7, 4));

    let raw_row = array![[0.2, -0.3, 0.7, 1.1, -1.3, 0.4, 2.0]];
    let through_both = inner_gauge.restrict_design(&finalization_gauge.restrict_design(&raw_row));
    let through_composed = composed.restrict_design(&raw_row);
    for (actual, expected) in through_composed.iter().zip(through_both.iter()) {
        assert!((actual - expected).abs() <= 1e-14);
    }

    let expected_shift =
        finalization_gauge.t_full.dot(&inner_gauge.affine_shift) + &finalization_gauge.affine_shift;
    assert_eq!(composed.affine_shift, expected_shift);
}

#[test]
fn weighted_crossprod_dense_falls_back_when_row_scaled_product_would_overflow() {
    let left = array![[1.0e-200]];
    let right = array![[1.0e200]];
    let weights = array![1.0e200];

    let cross = weighted_crossprod_dense_with_parallelism(
        &left,
        &weights,
        &right,
        gam_linalg::faer_ndarray::pool_parallelism(),
    )
        .expect("stable weighted cross-product should avoid overflow");
    let expected = 1.0e200;
    let rel_err = ((cross[[0, 0]] - expected) / expected).abs();

    assert!(cross[[0, 0]].is_finite());
    assert!(
        rel_err <= 1e-12,
        "unexpected weighted cross-product: {}",
        cross[[0, 0]]
    );
}

#[test]
fn scale_dense_rows_saturates_without_nan_when_coefficients_are_huge() {
    let mat = array![[1.0e200], [2.0e-200]];
    let coeffs = array![1.0e200, 1.0e200];

    let scaled = scale_dense_rows(&mat, &coeffs)
        .expect("row scaling should saturate overflow instead of producing NaN");

    assert!(scaled.iter().all(|value| value.is_finite()));
    assert!(scaled[[0, 0]] > 1.0e300);
    assert!((scaled[[1, 0]] - 2.0).abs() <= 1e-12);
}

#[test]
fn threshold_exact_newton_hessian_matches_negative_gradient_jacobian() {
    let family = survival_exact_newton_test_family();
    let beta_t = 0.35;
    let states = survival_exact_newton_threshold_states(beta_t);
    let eval = family.evaluate(&states).expect("evaluate at center");
    let BlockWorkingSet::ExactNewton { gradient, hessian } =
        &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
    else {
        panic!("threshold block should use exact newton");
    };
    let hessian = hessian.to_dense();

    let eps = 1e-6;
    let eval_plus = family
        .evaluate(&survival_exact_newton_threshold_states(beta_t + eps))
        .expect("evaluate at beta + eps");
    let eval_minus = family
        .evaluate(&survival_exact_newton_threshold_states(beta_t - eps))
        .expect("evaluate at beta - eps");
    let grad_plus = match &eval_plus.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
    {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
        _ => panic!("threshold block should use exact newton"),
    };
    let grad_minus =
        match &eval_minus.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("threshold block should use exact newton"),
        };
    let fd_neggrad_jac = -(grad_plus - grad_minus) / (2.0 * eps);

    assert!(
        (gradient[0]).is_finite() && hessian[[0, 0]].is_finite(),
        "non-finite threshold exact-newton quantities: grad={} hess={}",
        gradient[0],
        hessian[[0, 0]]
    );
    assert_eq!(
        hessian[[0, 0]].signum(),
        fd_neggrad_jac.signum(),
        "threshold Hessian sign mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
    assert!(
        (hessian[[0, 0]] - fd_neggrad_jac).abs() <= 1e-5,
        "threshold Hessian mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
}

#[test]
fn log_sigma_exact_newton_hessian_matches_negative_gradient_jacobian() {
    let family = survival_exact_newton_test_familywith_inverse_link(
        residual_distribution_inverse_link(ResidualDistribution::Logistic),
    );
    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
    let eval = family.evaluate(&states).expect("evaluate at center");
    let BlockWorkingSet::ExactNewton { hessian, .. } =
        &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA]
    else {
        panic!("log-sigma block should use exact newton");
    };
    let hessian = hessian.to_dense();

    let eps = 1e-6;
    let grad_at = |beta_ls: f64| -> f64 {
        let eval = family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                &array![beta_ls],
            ))
            .expect("evaluate shifted log-sigma");
        match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("log-sigma block should use exact newton"),
        }
    };
    let fd_neggrad_jac =
        -(grad_at(beta_log_sigma[0] + eps) - grad_at(beta_log_sigma[0] - eps)) / (2.0 * eps);

    assert_eq!(
        hessian[[0, 0]].signum(),
        fd_neggrad_jac.signum(),
        "log-sigma Hessian sign mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
    assert!(
        (hessian[[0, 0]] - fd_neggrad_jac).abs() <= 1e-5,
        "log-sigma Hessian mismatch: analytic={} fd={}",
        hessian[[0, 0]],
        fd_neggrad_jac
    );
}

#[test]
fn exact_newton_block_directional_derivatives_matchfd_for_non_probit_links() {
    let extracthessian = |eval: FamilyEvaluation, block_idx: usize| -> Array2<f64> {
        match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
                panic!("expected exact newton block")
            }
        }
    };

    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let eps = 1e-6;

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let base_eval = family.evaluate(&states).expect("base eval");

        for (block_idx, direction) in [
            (SurvivalLocationScaleFamily::BLOCK_TIME, array![1.0]),
            (SurvivalLocationScaleFamily::BLOCK_THRESHOLD, array![1.0]),
            (SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA, array![1.0]),
        ] {
            let analytic = family
                .exact_newton_hessian_directional_derivative(&states, block_idx, &direction)
                .expect("analytic dH")
                .expect("expected exact dH");

            let mut beta_time_plus = beta_time.clone();
            let mut beta_threshold_plus = beta_threshold.clone();
            let mut beta_log_sigma_plus = beta_log_sigma.clone();
            match block_idx {
                SurvivalLocationScaleFamily::BLOCK_TIME => {
                    beta_time_plus += &(eps * &direction);
                }
                SurvivalLocationScaleFamily::BLOCK_THRESHOLD => {
                    beta_threshold_plus += &(eps * &direction);
                }
                SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA => {
                    beta_log_sigma_plus += &(eps * &direction);
                }
                _ => panic!("unexpected block"),
            }

            let plus_states = survival_exact_newton_rebuild_states(
                &beta_time_plus,
                &beta_threshold_plus,
                &beta_log_sigma_plus,
            );
            let h_plus =
                extracthessian(family.evaluate(&plus_states).expect("plus eval"), block_idx);
            let h_base = extracthessian(base_eval.clone(), block_idx);
            let fd = (h_plus - h_base) / eps;
            gam_test_support::assert_matrix_derivativefd(
                &fd,
                &analytic,
                5e-4,
                &format!("survival {label} block {} dH", block_idx),
            );
        }
    }
}

#[test]
fn joint_exact_newton_hessian_matches_negative_gradient_jacobian_for_non_probit_links() {
    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let eps = 1e-6;

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let analytic = family
            .exact_newton_joint_hessian(&states)
            .expect("joint exact hessian")
            .expect("expected exact joint hessian");

        let flattengrad = |eval: FamilyEvaluation| -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(3);
            for (block_idx, slot) in out.iter_mut().enumerate() {
                *slot = match &eval.blockworking_sets[block_idx] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    BlockWorkingSet::Diagonal { .. }
                    | BlockWorkingSet::NaturalDiagonal { .. } => {
                        panic!("expected exact newton block")
                    }
                };
            }
            out
        };

        let mut fd = Array2::<f64>::zeros((3, 3));
        for j in 0..3 {
            let mut beta_time_plus = beta_time.clone();
            let mut beta_threshold_plus = beta_threshold.clone();
            let mut beta_log_sigma_plus = beta_log_sigma.clone();
            let mut beta_time_minus = beta_time.clone();
            let mut beta_threshold_minus = beta_threshold.clone();
            let mut beta_log_sigma_minus = beta_log_sigma.clone();
            match j {
                0 => {
                    beta_time_plus[0] += eps;
                    beta_time_minus[0] -= eps;
                }
                1 => {
                    beta_threshold_plus[0] += eps;
                    beta_threshold_minus[0] -= eps;
                }
                2 => {
                    beta_log_sigma_plus[0] += eps;
                    beta_log_sigma_minus[0] -= eps;
                }
                other => panic!("FD probe block {other} out of range (expected 0..3)"),
            }
            let grad_plus = flattengrad(
                family
                    .evaluate(&survival_exact_newton_rebuild_states(
                        &beta_time_plus,
                        &beta_threshold_plus,
                        &beta_log_sigma_plus,
                    ))
                    .expect("eval plus"),
            );
            let grad_minus = flattengrad(
                family
                    .evaluate(&survival_exact_newton_rebuild_states(
                        &beta_time_minus,
                        &beta_threshold_minus,
                        &beta_log_sigma_minus,
                    ))
                    .expect("eval minus"),
            );
            let col = -(grad_plus - grad_minus) / (2.0 * eps);
            fd.column_mut(j).assign(&col);
        }

        gam_test_support::assert_matrix_derivativefd(
            &fd,
            &analytic,
            2e-4,
            &format!("survival {label} joint H"),
        );
    }
}

#[test]
fn joint_exact_newton_score_matches_loglikelihoodfd_for_non_probit_links() {
    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let eps = 1e-6;

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let eval = family.evaluate(&states).expect("evaluate");
        let analytic = Array1::from_vec(vec![
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_TIME] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
        ]);

        let objective = |bt: &Array1<f64>, bth: &Array1<f64>, bls: &Array1<f64>| -> f64 {
            family
                .evaluate(&survival_exact_newton_rebuild_states(bt, bth, bls))
                .expect("eval objective")
                .log_likelihood
        };

        let mut fd = Array1::<f64>::zeros(3);
        fd[0] = (objective(
            &array![beta_time[0] + eps],
            &beta_threshold,
            &beta_log_sigma,
        ) - objective(
            &array![beta_time[0] - eps],
            &beta_threshold,
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[1] = (objective(
            &beta_time,
            &array![beta_threshold[0] + eps],
            &beta_log_sigma,
        ) - objective(
            &beta_time,
            &array![beta_threshold[0] - eps],
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[2] = (objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] + eps],
        ) - objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] - eps],
        )) / (2.0 * eps);

        for j in 0..3 {
            let abs = (analytic[j] - fd[j]).abs();
            if analytic[j].abs().max(fd[j].abs()) >= 1e-8 {
                assert_eq!(
                    analytic[j].signum(),
                    fd[j].signum(),
                    "survival {label} joint score sign mismatch at {j}: analytic={} fd={}",
                    analytic[j],
                    fd[j]
                );
            }
            assert!(
                abs <= 1e-5,
                "survival {label} joint score mismatch at {j}: analytic={} fd={} abs={}",
                analytic[j],
                fd[j],
                abs
            );
        }
    }
}

#[test]
fn joint_exact_newton_log_sigma_block_matches_fd_in_far_exp_tail() {
    let family = survival_exact_newton_test_family();
    let beta_time = array![0.2];
    let beta_threshold = array![0.1 * crate::sigma_link::safe_exp(700.0)];
    let beta_log_sigma0 = 701.0_f64;
    let beta_log_sigma = array![beta_log_sigma0];

    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
    let eval = family.evaluate(&states).expect("evaluate");
    let (analytic_score, analytic_info) =
        match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                (gradient[0], hessian.to_dense()[[0, 0]])
            }
            _ => panic!("expected exact newton log-sigma block"),
        };

    let objective = |beta_ls: &Array1<f64>| -> f64 {
        family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                beta_ls,
            ))
            .expect("eval objective")
            .log_likelihood
    };
    let score_at = |beta_ls: f64| -> f64 {
        let eval = family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                &array![beta_ls],
            ))
            .expect("eval score");
        match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact newton log-sigma block"),
        }
    };
    let h = 1e-4;
    let ll_plus = objective(&array![beta_log_sigma0 + h]);
    let ll_minus = objective(&array![beta_log_sigma0 - h]);
    let score_fd = (ll_plus - ll_minus) / (2.0 * h);
    let info_fd = -(score_at(beta_log_sigma0 + h) - score_at(beta_log_sigma0 - h)) / (2.0 * h);

    // The scale divides the time transform too (#2695), so at this fixture the
    // far-tail rows are moderate again: row 2 has u0 ≈ u1 ≈ 3.7e150 but its
    // entry and exit indices differ only by the scaled time gap
    // `(h1 − h0)·e^{−η_σ} ≈ 5e−154`, and the log-sigma score is O(1)
    // (≈ −2.168: −0.970 from row 0, −1.198 from row 2, of which the
    // `u·δu ≈ 1.8e−3` share is what the #2342 regroup must carry). `|ℓ| ≈ 7e2`
    // comes from the event log-densities' `−η_σ` terms.
    //
    // Central-difference FD error at h=1e-4: the score's truncation is
    // `h²/6·|ℓ'''| = O(1e-9)` and its rounding `ε·|ℓ|/h ≈ 1.6e-9`. A second
    // difference of `ℓ` would lose `ε·|ℓ|/h² ≈ 1.6e-5` to rounding against an
    // O(3e-2) information, so the information is checked against the central
    // difference of the analytic score instead (rounding `ε·|score|/h ≈ 5e-12`),
    // which the first assertion pins to the objective. A score that drops the
    // regroup's scaled time gap is off by `1.2·u·δu/2 ≈ 1.1e-3` and fails the
    // first bound by five orders; the sign checks are kept absolute.
    const SCORE_REL_TOL: f64 = 1e-8;
    const INFO_REL_TOL: f64 = 1e-6;
    assert_eq!(
        analytic_score.signum(),
        score_fd.signum(),
        "survival log-sigma score sign mismatch: analytic={analytic_score} fd={score_fd}"
    );
    assert!(
        (analytic_score - score_fd).abs() <= SCORE_REL_TOL * score_fd.abs().max(1.0),
        "the exact-newton survival log-sigma score should match the far-tail finite difference at beta_log_sigma={beta_log_sigma0}; got {} vs {} (rel budget {})",
        analytic_score,
        score_fd,
        SCORE_REL_TOL * score_fd.abs().max(1.0)
    );
    assert_eq!(
        analytic_info.signum(),
        info_fd.signum(),
        "survival log-sigma information sign mismatch: analytic={analytic_info} fd={info_fd}"
    );
    assert!(
        (analytic_info - info_fd).abs() <= INFO_REL_TOL * info_fd.abs().max(1.0),
        "the exact-newton survival log-sigma information should match the far-tail finite difference at beta_log_sigma={beta_log_sigma0}; got {} vs {} (rel budget {})",
        analytic_info,
        info_fd,
        INFO_REL_TOL * info_fd.abs().max(1.0)
    );
}

#[test]
fn survival_q_chain_derivatives_match_exact_exp_link_in_far_tails() {
    let eta_t = 2.0;
    for &eta_ls in &[701.0_f64, -30.0_f64] {
        let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
        let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
        assert!((q_t + inv_sigma).abs() <= 1e-15);
        assert!((q_ls - eta_t * inv_sigma).abs() <= 1e-15);
        assert!((q_tl - inv_sigma).abs() <= 1e-15);
        assert!((q_ll + eta_t * inv_sigma).abs() <= 1e-15);
        assert!((q_tl_ls + inv_sigma).abs() <= 1e-15);
        assert!((q_ll_ls - eta_t * inv_sigma).abs() <= 1e-15);
        let h = 1e-6;
        let q = |ls: f64| -eta_t * exp_sigma_inverse_from_eta_scalar(ls);
        let q_fd = (q(eta_ls + h) - q(eta_ls - h)) / (2.0 * h);
        assert!(
            (q_ls - q_fd).abs() <= (1e-8 * q_fd.abs()).max(1e-8),
            "q_s finite difference mismatch at eta_ls={eta_ls}: analytic={q_ls} fd={q_fd}"
        );
    }
}

#[test]
fn survival_exact_log_sigma_dh_matches_far_tail_third_derivative() {
    // Representable far tail: q spans ~{4e-2, 1e21, 2e14} across the three
    // rows, so the u>1e3 stable paired path engages while the log-likelihood
    // (|ℓ| ~ 5e41) and its third derivative stay inside f64 — the old
    // beta_log_sigma=701 fixture pushed |ℓ| beyond 1e300 into clamp-land,
    // where an ABSOLUTE 1e-3 gate on an ~1e148-scale quantity demands 1e-151
    // relative agreement no implementation can deliver (#2342; the clamp-land
    // fixture keeps a finiteness gate below). The FD reference at h=1e-3
    // resolves this magnitude to ~1e-6 relative, so the RELATIVE 1e-3 gate is
    // honest.
    let family = survival_exact_newton_test_family();
    let beta_time = array![0.2];
    let beta_threshold = array![0.1 * crate::sigma_link::safe_exp(20.0)];
    let beta_log_sigma0 = 21.0_f64;
    let beta_log_sigma = array![beta_log_sigma0];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

    let analytic = family
        .exact_newton_hessian_directional_derivative(
            &states,
            SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA,
            &array![1.0],
        )
        .expect("analytic dH")
        .expect("expected exact dH");

    let objective = |beta_ls: f64| -> f64 {
        family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                &array![beta_ls],
            ))
            .expect("eval objective")
            .log_likelihood
    };
    let h = 1e-3_f64;
    let fd3 = (objective(beta_log_sigma0 + 2.0 * h) - 2.0 * objective(beta_log_sigma0 + h)
        + 2.0 * objective(beta_log_sigma0 - h)
        - objective(beta_log_sigma0 - 2.0 * h))
        / (2.0 * h.powi(3));
    assert!(
        (analytic[[0, 0]] + fd3).abs() < 1e-3 * fd3.abs().max(1.0),
        "the exact-newton survival log-sigma dH entry should equal the negative third derivative in the far tail at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
        analytic[[0, 0]],
        -fd3
    );
}

/// #2342 clamp-land finiteness gate: at the extreme `beta_log_sigma = 701`
/// fixture the likelihood itself exceeds f64 range (`q ~ e150`-scale rows,
/// `|ℓ| > 1e300`), so no DERIVATIVE agreement can be asserted there — but the
/// analytic dH must still be FINITE. The zero-stack far-tail rows (censored,
/// `S ≈ 1`, all outer derivatives exactly zero) used to compose against
/// clamped index channels and manufacture `0·∞ = NaN`.
#[test]
fn survival_exact_log_sigma_dh_stays_finite_at_clamped_extreme_tail() {
    let family = survival_exact_newton_test_family();
    let beta_time = array![0.2];
    let beta_threshold = array![0.1 * crate::sigma_link::safe_exp(700.0)];
    let beta_log_sigma = array![701.0_f64];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

    let analytic = family
        .exact_newton_hessian_directional_derivative(
            &states,
            SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA,
            &array![1.0],
        )
        .expect("analytic dH")
        .expect("expected exact dH");
    assert!(
        analytic[[0, 0]].is_finite(),
        "clamped extreme-tail dH must stay finite (a zero-stack row must \
         contribute exactly zero, never 0·∞ = NaN); got {}",
        analytic[[0, 0]]
    );

    let joint = family
        .exact_newton_joint_hessian_directional_derivative(&states, &array![0.0, 0.0, 1.0])
        .expect("analytic joint dH")
        .expect("expected exact joint dH");
    assert!(
        joint.iter().all(|v| v.is_finite()),
        "clamped extreme-tail joint dH must stay finite everywhere; got {joint:?}"
    );
}

#[test]
fn survival_joint_exact_log_sigma_dh_matches_far_tail_third_derivative() {
    // Same representable far-tail fixture and RELATIVE gate as
    // `survival_exact_log_sigma_dh_matches_far_tail_third_derivative` (see the
    // #2342 rationale there); this variant pins the joint-path entry.
    let family = survival_exact_newton_test_family();
    let beta_time = array![0.2];
    let beta_threshold = array![0.1 * crate::sigma_link::safe_exp(20.0)];
    let beta_log_sigma0 = 21.0_f64;
    let beta_log_sigma = array![beta_log_sigma0];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &array![0.0, 0.0, 1.0])
        .expect("analytic joint dH")
        .expect("expected exact joint dH");

    let objective = |beta_ls: f64| -> f64 {
        family
            .evaluate(&survival_exact_newton_rebuild_states(
                &beta_time,
                &beta_threshold,
                &array![beta_ls],
            ))
            .expect("eval objective")
            .log_likelihood
    };
    let h = 1e-3_f64;
    let fd3 = (objective(beta_log_sigma0 + 2.0 * h) - 2.0 * objective(beta_log_sigma0 + h)
        + 2.0 * objective(beta_log_sigma0 - h)
        - objective(beta_log_sigma0 - 2.0 * h))
        / (2.0 * h.powi(3));
    assert!(
        (analytic[[2, 2]] + fd3).abs() < 1e-3 * fd3.abs().max(1.0),
        "the exact joint survival dH log-sigma/log-sigma entry should equal the negative third derivative in the far tail at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
        analytic[[2, 2]],
        -fd3
    );
}

#[test]
fn joint_exact_newton_score_matches_loglikelihoodfd_near_fitted_non_probit_points() {
    let eps = 1e-6;
    let cases = vec![
        (
            "logistic-near-fit",
            residual_distribution_inverse_link(ResidualDistribution::Logistic),
            array![0.7746886451475979],
            array![-0.6407086184606554],
            array![-0.15],
        ),
        (
            "cloglog-near-fit",
            residual_distribution_inverse_link(ResidualDistribution::Gumbel),
            array![0.8153913537182474],
            array![14.123707996892579],
            array![1.4355329717917449],
        ),
    ];

    for (label, inverse_link, beta_time, beta_threshold, beta_log_sigma) in cases {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let eval = family.evaluate(&states).expect("evaluate");
        let analytic = Array1::from_vec(vec![
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_TIME] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("expected exact newton block"),
            },
        ]);

        let objective = |bt: &Array1<f64>, bth: &Array1<f64>, bls: &Array1<f64>| -> f64 {
            family
                .evaluate(&survival_exact_newton_rebuild_states(bt, bth, bls))
                .expect("eval objective")
                .log_likelihood
        };

        let mut fd = Array1::<f64>::zeros(3);
        fd[0] = (objective(
            &array![beta_time[0] + eps],
            &beta_threshold,
            &beta_log_sigma,
        ) - objective(
            &array![beta_time[0] - eps],
            &beta_threshold,
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[1] = (objective(
            &beta_time,
            &array![beta_threshold[0] + eps],
            &beta_log_sigma,
        ) - objective(
            &beta_time,
            &array![beta_threshold[0] - eps],
            &beta_log_sigma,
        )) / (2.0 * eps);
        fd[2] = (objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] + eps],
        ) - objective(
            &beta_time,
            &beta_threshold,
            &array![beta_log_sigma[0] - eps],
        )) / (2.0 * eps);

        for j in 0..3 {
            let abs = (analytic[j] - fd[j]).abs();
            if analytic[j].abs().max(fd[j].abs()) >= 1e-8 {
                assert_eq!(
                    analytic[j].signum(),
                    fd[j].signum(),
                    "survival {label} joint score sign mismatch at {j}: analytic={} fd={}",
                    analytic[j],
                    fd[j]
                );
            }
            assert!(
                abs <= 5e-4,
                "survival {label} joint score mismatch at {j}: analytic={} fd={} abs={}",
                analytic[j],
                fd[j],
                abs
            );
        }
    }
}

/// #1389 regression: the joint-Hessian directional-derivative velocity (event
/// Jacobian `g`) pass is skipped before any `p²` allocation when no weighted row
/// carries live qdot-derivative mass. Censored rows carry event_weight 0, so an
/// all-censored fixture has `d1_qdot1 = d2_qdot1 = d_h_d = 0` on every row and
/// the velocity term is identically zero — the skip path must therefore produce
/// a directional derivative that still matches the finite difference of the
/// joint gradient (i.e. the skip omits only a zero contribution).
#[test]
fn joint_dh_velocity_skip_is_exact_on_all_censored_rows() {
    let mut family = survival_exact_newton_test_family();
    // All rows censored ⇒ event_weight 0 on every row ⇒ no velocity mass, so the
    // #1389 `any_live_qdot` guard short-circuits the velocity pass.
    family.y = array![0.0, 0.0, 0.0];

    let beta_time = array![0.2];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![-0.15];
    let states = survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

    // Direction perturbs every block so any dropped cross-velocity term would
    // show up in the comparison.
    let direction = array![1.0, 1.0, 1.0];
    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("analytic joint dH")
        .expect("expected exact joint dH");

    // The directional derivative of the joint NLL Hessian must equal the central
    // finite difference of that Hessian along `direction`. Because the velocity
    // pass is skipped (all rows censored), `analytic` carries no velocity term;
    // the FD-of-Hessian is the independent ground truth, so a match certifies the
    // skip dropped only a zero contribution.
    let eps = 1e-6;
    let hessian_at = |scale: f64| -> Array2<f64> {
        let bt = &beta_time + scale * direction[0];
        let bth = &beta_threshold + scale * direction[1];
        let bls = &beta_log_sigma + scale * direction[2];
        family
            .exact_newton_joint_hessian(&survival_exact_newton_rebuild_states(&bt, &bth, &bls))
            .expect("joint hessian")
            .expect("expected exact joint hessian")
    };
    let fd = (&hessian_at(eps) - &hessian_at(-eps)) / (2.0 * eps);
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (analytic[[r, c]] - fd[[r, c]]).abs() <= 5e-4,
                "all-censored velocity-skip dH[{r}][{c}] mismatch: analytic={} fd={}",
                analytic[[r, c]],
                fd[[r, c]],
            );
        }
    }
}

#[test]
fn row_derivative_identities_hold_for_non_probit_links() {
    let beta_time = array![0.8153913537182474];
    let beta_threshold = array![0.35];
    let beta_log_sigma = array![0.4];

    for (label, inverse_link) in survival_non_probit_test_links() {
        let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let (h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, .., etaw) =
            family.validate_joint_states(&states).expect("joint states");
        // For time-invariant blocks, eta_ls_entry == eta_ls_exit.
        let inv_sigma = eta_ls_exit.mapv(exp_sigma_inverse_from_eta_scalar);
        let inv_sigma_entry = eta_ls_entry.mapv(exp_sigma_inverse_from_eta_scalar);

        for i in 0..family.n {
            let state = survival_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                -eta_t_entry[i] * inv_sigma_entry[i] + etaw.map_or(0.0, |w| w[i]),
                -eta_t_exit[i] * inv_sigma[i] + etaw.map_or(0.0, |w| w[i]),
                0.0,
                -eta_ls_exit[i],
                family.entry_active[i],
            );
            let row = family
                .row_derivatives(i, state)
                .expect("row derivatives")
                .expect("active row");

            let ell_h0 = row.grad_time_eta_h0;
            let ell_h1 = row.grad_time_eta_h1;
            let ell_q = row.d1_q0 + row.d1_q1;
            assert!(
                (ell_q - ell_h0 - ell_h1).abs() <= 1e-10,
                "survival {label} row {i} violated ell_q = ell_h0 + ell_h1: q={} h0={} h1={}",
                ell_q,
                ell_h0,
                ell_h1
            );
        }
    }
}

#[test]
fn sparse_exact_newton_matches_denseworking_sets() {
    let dense_family = survival_exact_newton_test_family();
    let sparse_family = sparse_survival_exact_newton_test_family();
    let states = survival_exact_newton_threshold_states(0.35);

    let dense_eval = dense_family.evaluate(&states).expect("dense evaluate");
    let sparse_eval = sparse_family.evaluate(&states).expect("sparse evaluate");
    assert!((dense_eval.log_likelihood - sparse_eval.log_likelihood).abs() <= 1e-12);
    assert_eq!(
        dense_eval.blockworking_sets.len(),
        sparse_eval.blockworking_sets.len()
    );
    for (dense_block, sparse_block) in dense_eval
        .blockworking_sets
        .iter()
        .zip(sparse_eval.blockworking_sets.iter())
    {
        match (dense_block, sparse_block) {
            (
                BlockWorkingSet::ExactNewton {
                    gradient: dense_g,
                    hessian: dense_h,
                },
                BlockWorkingSet::ExactNewton {
                    gradient: sparse_g,
                    hessian: sparse_h,
                },
            ) => {
                let dense_h = dense_h.to_dense();
                let sparse_h = sparse_h.to_dense();
                assert_eq!(dense_g.len(), sparse_g.len());
                assert_eq!(dense_h.dim(), sparse_h.dim());
                for i in 0..dense_g.len() {
                    assert!((dense_g[i] - sparse_g[i]).abs() <= 1e-12);
                }
                for i in 0..dense_h.nrows() {
                    for j in 0..dense_h.ncols() {
                        assert!((dense_h[[i, j]] - sparse_h[[i, j]]).abs() <= 1e-12);
                    }
                }
            }
            _ => panic!("expected exact-newton blocks"),
        }
    }

    let direction = array![0.2];
    let dense_dh = dense_family
        .exact_newton_hessian_directional_derivative(&states, 1, &direction)
        .expect("dense directional derivative")
        .expect("dense threshold directional derivative");
    let sparse_dh = sparse_family
        .exact_newton_hessian_directional_derivative(&states, 1, &direction)
        .expect("sparse directional derivative")
        .expect("sparse threshold directional derivative");
    assert_eq!(dense_dh.dim(), sparse_dh.dim());
    for i in 0..dense_dh.nrows() {
        for j in 0..dense_dh.ncols() {
            assert!((dense_dh[[i, j]] - sparse_dh[[i, j]]).abs() <= 1e-12);
        }
    }
}

/// Full-path structural monotonicity regression for the
/// heart_failure_survival workflow setup.
#[test]
fn heart_failure_full_fit_structural_time_coefficients() {
    // 20 rows with realistic-ish I-spline-like structure.
    let n = 20;
    let p_time = 8; // 8 time basis columns

    // Entry times all near zero (left-truncation at 0) — like __entry=0.
    let age_entry = Array1::from_elem(n, 1e-9_f64);
    // Exit times spread out like real survival data.
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        age_exit[i] = 4.0 + (i as f64) * 14.0; // 4 to 270
    }

    // Events: ~1/3 event rate.
    let mut event_target = Array1::<f64>::zeros(n);
    for i in [0, 3, 5, 8, 12, 17] {
        event_target[i] = 1.0;
    }
    let weights = Array1::ones(n);

    // Build I-spline-like time designs.
    // Entry design is all zeros (I-spline = 0 below knot range).
    let design_entry = Array2::<f64>::zeros((n, p_time));

    // Exit design: monotonically increasing I-spline-like columns.
    let mut design_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64); // 0 to 1
        for j in 0..p_time {
            let center = (j as f64 + 0.5) / (p_time as f64);
            // Smooth sigmoid-like I-spline approximation.
            let x = 8.0 * (t - center);
            design_exit[[i, j]] = 1.0 / (1.0 + (-x).exp());
        }
    }

    // Derivative design: derivative of I-spline columns.
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        for j in 0..p_time {
            let center = (j as f64 + 0.5) / (p_time as f64);
            let x = 8.0 * (t - center);
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            // Derivative of sigmoid * chain_rule (1/t).
            let deriv = 8.0 * sigmoid * (1.0 - sigmoid);
            let chain = 1.0 / age_exit[i];
            design_derivative_exit[[i, j]] = deriv * chain;
        }
    }

    // The workflow carries the derivative floor in the offsets, so the
    // structural time coefficients only need to stay non-negative.
    let derivative_offset_exit =
        Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
    let offset_entry = Array1::<f64>::zeros(n);
    let offset_exit = Array1::<f64>::zeros(n);

    // Simple difference penalty.
    let mut penalty = Array2::<f64>::zeros((p_time, p_time));
    for i in 0..(p_time - 1) {
        penalty[[i, i]] += 1.0;
        penalty[[i, i + 1]] -= 1.0;
        penalty[[i + 1, i]] -= 1.0;
        penalty[[i + 1, i + 1]] += 1.0;
    }

    let spec = SurvivalLocationScaleSpec {
        age_entry,
        age_exit,
        event_target,
        weights,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry,
            offset_exit,
            derivative_offset_exit: derivative_offset_exit.clone(),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![penalty.clone()],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(array![0.0]),
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        persistent_warm_start_store: None,
        cache_mirror_sessions: Vec::new(),
    };

    match fit_survival_location_scale_with_geometry(spec).map(|(fit, _)| fit) {
        Ok(result) => {
            // Structural-monotonicity invariant implied by the test's
            // name: the I-spline-like time block carries structural
            // lower bounds of zero (see
            // `structural_time_coefficient_lower_bounds`), and
            // the constrained solve/max-step limiter represents that
            // cone directly. Every accepted coefficient must therefore
            // satisfy β ≥ 0 — the precondition for the monotone
            // I-spline reconstruction the workflow consumes downstream.
            assert!(
                result.beta_time().iter().all(|&b| b.is_finite()),
                "structural time coefficients must be finite: {:?}",
                result.beta_time(),
            );
            assert!(
                result.beta_time().iter().all(|&b| b >= 0.0),
                "structural time coefficients must be non-negative after constrained solve: {:?}",
                result.beta_time(),
            );
            // Parallel invariant for BLOCK_LINK_WIGGLE: monotone-link
            // wiggle coefficients are structurally non-negative. This
            // test configures `linkwiggle_block: None`, so the block is
            // absent — but if it is ever enabled here the represented
            // block constraint must enforce the same invariant.
            if let Some(beta_link_wiggle) = result.beta_link_wiggle() {
                assert!(
                    beta_link_wiggle.iter().all(|&b| b.is_finite()),
                    "link-wiggle coefficients must be finite: {beta_link_wiggle:?}",
                );
                assert!(
                    beta_link_wiggle.iter().all(|&b| b >= 0.0),
                    "link-wiggle coefficients must be non-negative after constrained solve: {beta_link_wiggle:?}",
                );
            }
        }
        Err(e) => {
            panic!("fit_survival_location_scale failed: {e}");
        }
    }
}

/// Small structural-monotonicity regression for the
/// heart_failure_survival workflow setup.
#[test]
fn heart_failure_structural_time_small() {
    // 6 rows: 3 events, 3 non-events.  Single time column for simplicity.
    let n = 6;
    // I-spline-like designs: entry is all zero (left truncation at t=0),
    // exit has non-trivial values, derivative is the B-spline derivative.
    let x_entry = Array2::<f64>::zeros((n, 2));
    let x_exit = array![
        [0.1, 0.05],
        [0.3, 0.15],
        [0.5, 0.35],
        [0.7, 0.55],
        [0.9, 0.80],
        [1.0, 0.95],
    ];
    let x_deriv = array![
        [0.2, 0.1],
        [0.3, 0.2],
        [0.3, 0.3],
        [0.3, 0.3],
        [0.2, 0.3],
        [0.1, 0.2],
    ];
    let offset_deriv = Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);

    let family = SurvivalLocationScaleFamily {
        n,
        y: array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        w: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        x_time_entry: Arc::new(x_entry),
        x_time_exit: Arc::new(x_exit.clone()),
        x_time_deriv: Arc::new(x_deriv.clone()),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0, 0.0]),
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        entry_active: Arc::from(vec![true; n]),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: true,
    };

    // Build initial states with beta=0 and a feasible positive derivative offset.
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: {
                let mut eta = Array1::<f64>::zeros(3 * n);
                eta.slice_mut(ndarray::s![2 * n..3 * n])
                    .fill(DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
                eta
            },
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
    ];

    // Step 1: Verify initial evaluate succeeds on the feasible domain.
    let eval = family
        .evaluate(&states)
        .expect("initial evaluate with positive d_eta/dt should succeed");

    // Step 2: Extract time block gradient and Hessian.
    let (grad, hess) = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, hessian } => {
            (gradient.clone(), hessian.to_dense())
        }
        _ => panic!("expected exact-newton for time block"),
    };

    // Step 3: Simulate Newton step (H + ridge*I) * delta = grad - S*beta.
    // With beta=0 and no penalty: (H + ridge*I) * delta = grad.
    let ridge = 1e-6_f64;
    let p = 2;
    let mut lhs = hess.clone();
    for i in 0..p {
        lhs[[i, i]] += ridge;
    }
    // Solve via direct inversion (2x2).
    let det = lhs[[0, 0]] * lhs[[1, 1]] - lhs[[0, 1]] * lhs[[1, 0]];
    assert!(
        det.abs() > 1e-30,
        "heart-failure Newton fixture must have an invertible ridged Hessian; det={det}"
    );
    let inv00 = lhs[[1, 1]] / det;
    let inv01 = -lhs[[0, 1]] / det;
    let inv10 = -lhs[[1, 0]] / det;
    let inv11 = lhs[[0, 0]] / det;
    let delta = array![
        inv00 * grad[0] + inv01 * grad[1],
        inv10 * grad[0] + inv11 * grad[1]
    ];
    assert!(
        delta.iter().all(|v| v.is_finite()),
        "Newton delta has non-finite entries: {:?}",
        delta
    );

    // Step 4: Compute new d_raw after the step.
    let new_d_raw = x_deriv.dot(&delta) + &offset_deriv;
    for (i, &v) in new_d_raw.iter().enumerate() {
        assert!(
            v.is_finite(),
            "d_raw[{i}] is non-finite ({v}) after Newton step with delta={:?}",
            delta
        );
    }

    // Step 5: Verify evaluate succeeds with the new state.
    let new_eta_time = {
        let mut eta = Array1::<f64>::zeros(3 * n);
        // h0 = x_entry * delta (all zero since x_entry is zero)
        // h1 = x_exit * delta
        let h1 = x_exit.dot(&delta);
        eta.slice_mut(ndarray::s![n..2 * n]).assign(&h1);
        // d_raw = x_deriv * delta + offset_deriv
        eta.slice_mut(ndarray::s![2 * n..3 * n]).assign(&new_d_raw);
        eta
    };
    let new_states = vec![
        ParameterBlockState {
            beta: delta.clone(),
            eta: new_eta_time,
        },
        states[1].clone(),
        states[2].clone(),
    ];
    family
        .evaluate(&new_states)
        .unwrap_or_else(|e| panic!("evaluate failed after Newton step: {e}"));
}

#[test]
fn evaluate_survival_location_scale_rejects_non_finite_d_eta_dt() {
    let n = 2;
    let family = SurvivalLocationScaleFamily {
        n,
        y: array![1.0, 0.0],
        w: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        x_time_entry: Arc::new(Array2::zeros((n, 1))),
        x_time_exit: Arc::new(Array2::ones((n, 1))),
        x_time_deriv: Arc::new(Array2::ones((n, 1))),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: lower_bound_constraints(&array![0.0]),
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::ones((n, 1)),
        )),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        entry_active: Arc::from(vec![true; n]),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: true,
    };

    let mut eta_time = Array1::<f64>::zeros(3 * n);
    eta_time[2 * n] = f64::NAN;
    eta_time[2 * n + 1] = 0.25;
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_time,
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(n),
        },
    ];

    let eval = match family.evaluate(&states) {
        Ok(_) => panic!("non-finite d_eta/dt must be rejected"),
        Err(err) => err,
    };
    assert!(eval.contains("non-finite"));
}

#[test]
fn q_chain_derivatives_match_exact_exp_link_in_lower_tail() {
    let eta_t = 2.0;
    let eta_ls = -30.0;
    let q = |ls: f64| -eta_t * exp_sigma_inverse_from_eta_scalar(ls);
    let h = 1e-6;
    let q_left = q(eta_ls - h);
    let q_mid = q(eta_ls);
    let q_right = q(eta_ls + h);
    assert!(
        q_left != q_mid && q_right != q_mid,
        "exact exp-link q should remain eta_ls-sensitive in the lower tail"
    );

    let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    assert!((q_t + inv_sigma).abs() <= 1e-15);
    assert!((q_ls - eta_t * inv_sigma).abs() <= 1e-15);
    assert!((q_tl - inv_sigma).abs() <= 1e-15);
    assert!((q_ll + eta_t * inv_sigma).abs() <= 1e-15);
    assert!((q_tl_ls + inv_sigma).abs() <= 1e-15);
    assert!((q_ll_ls - eta_t * inv_sigma).abs() <= 1e-15);
}

#[test]
fn survival_q0dot_from_base_preserves_far_tail_cancellation() {
    let eta_t = 1e-10;
    let eta_ls = -700.0;
    let eta_t_deriv = 1.0 - 1e-12;
    let eta_ls_deriv = 1e10;
    let base = survival_base_q_scalars(eta_t, eta_ls);

    // The location channel's rate carries no scale (#2695): it is the local
    // cancellation `eta_t·eta_ls' − eta_t'` in one fused multiply-add, and the
    // scaled `dq0/dt = e^{−eta_ls}·r` is formed from it.
    let rate = survival_q0dot_from_base(base, eta_t_deriv, eta_ls_deriv);
    assert_eq!(
        rate.to_bits(),
        eta_t.mul_add(eta_ls_deriv, -eta_t_deriv).to_bits()
    );
    let factorized = safe_product(exp_sigma_inverse_from_eta_scalar(eta_ls), rate);
    let expected = safe_product(
        exp_sigma_inverse_from_eta_scalar(eta_ls),
        eta_t.mul_add(eta_ls_deriv, -eta_t_deriv),
    );
    let expanded = safe_sum2(
        safe_product(base.q_t, eta_t_deriv),
        safe_product(base.q_ls, eta_ls_deriv),
    );

    assert!(factorized.is_finite());
    assert!(expected.is_finite());
    assert!(
        (factorized - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "factorized qdot mismatch: got {factorized}, expected {expected}"
    );
    // The expanded (distributed) form sums two ~inv_sigma-magnitude terms
    // whose difference is 12 orders smaller, so it is in the huge-magnitude
    // regime with only ~4 surviving digits, while the factorized form does the
    // local cancellation `eta_t·eta_ls_deriv − eta_t_deriv` BEFORE the
    // inv_sigma product and keeps full relative precision. (A former
    // `factorized.abs() <= 1e206` ceiling here was a relic of the removed
    // +500 σ-inverse cap: under the current f64-representability saturation of
    // `exp_sigma_inverse_from_eta_scalar` the exact value is ≈1.01e292 —
    // magnitude is not the property this test pins; preserved precision is.)
    assert!(expanded.abs() >= 1e200);
    let factorized_rel = ((factorized - expected) / expected).abs().max(1e-16);
    let expanded_rel = ((expanded - expected) / expected).abs();
    assert!(
        expanded_rel >= 1e3 * factorized_rel,
        "far-tail cancellation not demonstrated: expanded rel err {expanded_rel:e} \
         should dwarf factorized rel err {factorized_rel:e}"
    );
}

#[test]
fn compensated_difference_carries_explicit_roundoff_bound() {
    let lhs = 1.0e217 + 1.0e201;
    let rhs = 1.0e217;
    let diff = compensated_difference(lhs, rhs);

    assert!(diff.value.is_finite());
    assert!(diff.roundoff_slack.is_finite());
    assert!(diff.roundoff_slack >= 0.0);
    assert!(diff.operand_scale >= rhs.abs());
}

#[test]
fn logistic_residual_tail_derivatives_should_match_stable_closed_forms() {
    let z = 50.0_f64;
    let e = (-z).exp();
    let denom = 1.0_f64 + e;
    let stable_pdf = e / denom.powi(2);
    let stable_d1 = e * (e - 1.0) / denom.powi(3);
    let stable_d2 = e * (e * e - 4.0 * e + 1.0) / denom.powi(4);
    let stable_d3 = e * (e * e * e - 11.0 * e * e + 11.0 * e - 1.0) / denom.powi(5);

    let dist = ResidualDistribution::Logistic;
    assert!(
        (dist.pdf(z) - stable_pdf).abs() < 1e-30,
        "logistic residual pdf should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdf(z),
        stable_pdf
    );
    assert!(
        (dist.pdf_derivative(z) - stable_d1).abs() < 1e-30,
        "logistic residual pdf' should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdf_derivative(z),
        stable_d1
    );
    assert!(
        (dist.pdfsecond_derivative(z) - stable_d2).abs() < 1e-30,
        "logistic residual pdf'' should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdfsecond_derivative(z),
        stable_d2
    );
    assert!(
        (dist.pdfthird_derivative(z) - stable_d3).abs() < 1e-30,
        "logistic residual pdf''' should equal the stable tail formula at z={z}; got {} vs {}",
        dist.pdfthird_derivative(z),
        stable_d3
    );
}

#[test]
fn gumbel_cdf_negative_tail_should_match_expm1_form() {
    let z = -50.0_f64;
    let ez = z.exp();
    let stable_cdf = -(-ez).exp_m1();
    let dist = ResidualDistribution::Gumbel;
    assert!(stable_cdf > 0.0);
    assert!(
        (dist.cdf(z) - stable_cdf).abs() < 1e-30,
        "gumbel cdf should equal -expm1(-exp(z)) in the negative tail at z={z}; got {} vs {}",
        dist.cdf(z),
        stable_cdf
    );
}

#[test]
fn probit_survival_helper_matches_upper_tail_probability() {
    let eta = 10.0_f64;
    let stable_survival = 0.5 * libm::erfc(eta / std::f64::consts::SQRT_2);
    assert!(stable_survival > 0.0);
    let helper = inverse_link_survival_probvalue(&InverseLink::Standard(StandardLink::Probit), eta);
    assert!(
        (helper - stable_survival).abs() < 1e-30,
        "probit survival helper should use the upper-tail erfc form at eta={eta}; got {} vs {}",
        helper,
        stable_survival
    );
}

#[test]
fn cloglog_survival_helper_matches_negative_tail_function() {
    let eta = -100.0_f64;
    let stable_survival = (-(eta.exp())).exp();
    let helper =
        inverse_link_survival_probvalue(&InverseLink::Standard(StandardLink::CLogLog), eta);
    assert_eq!(stable_survival, 1.0);
    assert!(
        (helper - stable_survival).abs() < 1e-30,
        "cloglog survival helper should evaluate exp(-exp(eta)) itself, not a clamped surrogate, at eta={eta}; got {} vs {}",
        helper,
        stable_survival
    );
}

#[test]
fn positive_log_cumulative_hazard_maps_to_baseline_cloglog_survival() {
    let cumulative_hazard = 4.0_f64;
    let eta = cumulative_hazard.ln();
    let survival =
        inverse_link_survival_probvalue(&InverseLink::Standard(StandardLink::CLogLog), eta);
    let expected = (-cumulative_hazard).exp();
    assert!(
        (survival - expected).abs() < 1e-15,
        "baseline cloglog survival should be exp(-H0) when eta = log(H0); got {} vs {}",
        survival,
        expected
    );
}

/// #932 (survival link-wiggle — the issue's named next step): the survival
/// location-scale JOINT row NLL written ONCE over [`JetScalar`] is extended with
/// the link-wiggle warp `q = q0 + Σ_j βw_j·B_j(q0)` (and the qdot coupling
/// `g = hdot + m1·qdot0`, `m1 = 1 + Σ_j βw_j·B'_j(q0_exit)`), with the βw amplitudes as
/// extra jet primaries. The mechanically-derived joint Hessian — including the
/// `(η, βw)` and `(βw, βw)` cross blocks a fixed `JᵀHJ` pullback would drop —
/// is pinned against central finite differences of the SAME program's value,
/// across the three residual distributions. This validates the nonlinear
/// link-wiggle pullback (the issue's §5/§13 map-inside-the-program) in the
/// survival row program — the foundation of the single-sourced link-wiggle
/// joint Hessian (`survival_ls_wiggle_joint_hessian_dense`) shipped in
/// production.
#[test]
fn survival_ls_wiggle_jet_program_joint_hessian_matches_fd_932() {
    use gam_math::jet_scalar::JetScalar;
    use gam_math::jet_tower::{RowProgram, program_row_kernel};

    const PW: usize = 2;
    const KW: usize = SLS_ROW_K + PW; // 9 base channels + 2 wiggle amplitudes

    // Smooth wiggle basis B_j and its first three derivatives at q (any C^3
    // basis exercises the warp; the production spline supplies the same stack).
    fn basis(j: usize, q: f64) -> [f64; 4] {
        match j {
            0 => [0.5 * q * q, q, 1.0, 0.0],
            _ => [q * q * q / 6.0, 0.5 * q * q, q, 1.0],
        }
    }

    struct WiggleProg {
        link: InverseLink,
        w: f64,
        d: f64,
        p: [f64; KW],
    }
    impl RowProgram<KW> for WiggleProg {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; KW], String> {
            if row != 0 {
                return Err(format!("wiggle program: row {row} out of range"));
            }
            Ok(self.p)
        }
        fn eval<S: JetScalar<KW>>(&self, row: usize, p: &[S; KW]) -> Result<S, String> {
            if row != 0 {
                return Err(format!("wiggle program: row {row} out of range"));
            }
            // Base nine-channel survival indices (exactly `sls_row_nll`).
            let inv_sigma_entry = p[7].neg().exp();
            let q0 = p[4].mul(&inv_sigma_entry).neg();
            let inv_sigma_exit = p[6].neg().exp();
            let q1 = p[3].mul(&inv_sigma_exit).neg();
            let qdot0 = inv_sigma_exit.mul(&p[3].mul(&p[8]).sub(&p[5]));

            // Link-wiggle warp: amplitudes are primaries 9..9+PW; each basis is
            // composed onto the BASE index jet (so it carries the η-dependence).
            let q0v = gam_math::nested_dual::JetField::value(&q0);
            let q1v = gam_math::nested_dual::JetField::value(&q1);
            let mut q0w = q0;
            let mut q1w = q1;
            let mut m1 = S::constant(1.0);
            for j in 0..PW {
                let bw = p[SLS_ROW_K + j];
                let b0 = basis(j, q0v);
                q0w = q0w.add(&bw.mul(&q0.compose_unary([b0[0], b0[1], b0[2], 0.0, 0.0])));
                let b1 = basis(j, q1v);
                q1w = q1w.add(&bw.mul(&q1.compose_unary([b1[0], b1[1], b1[2], b1[3], 0.0])));
                m1 = m1.add(&bw.mul(&q1.compose_unary([b1[1], b1[2], b1[3], 0.0, 0.0])));
            }
            // The scale divides the time transform too (#2695).
            let u0w = p[0].mul(&inv_sigma_entry).add(&q0w);
            let u1w = p[1].mul(&inv_sigma_exit).add(&q1w);
            let g = inv_sigma_exit
                .mul(&p[2].sub(&p[1].mul(&p[8])))
                .add(&m1.mul(&qdot0));

            let mut nll = u0w
                .compose_unary(survival_ls_log_survival_stack(
                    &self.link,
                    gam_math::nested_dual::JetField::value(&u0w),
                )?)
                .scale(self.w);
            let censored_weight = self.w * (1.0 - self.d);
            if censored_weight != 0.0 {
                nll = nll.add(
                    &u1w.compose_unary(survival_ls_log_survival_stack(
                        &self.link,
                        gam_math::nested_dual::JetField::value(&u1w),
                    )?)
                    .scale(-censored_weight),
                );
            }
            let event_weight = self.w * self.d;
            if event_weight != 0.0 {
                nll = nll
                    .add(
                        &u1w.compose_unary(survival_ls_log_pdf_stack(
                            &self.link,
                            gam_math::nested_dual::JetField::value(&u1w),
                            0.0,
                        )?)
                        .scale(-event_weight),
                    )
                    .add(
                        &g.compose_unary(survival_ls_positive_log_stack(
                            gam_math::nested_dual::JetField::value(&g),
                        ))
                        .scale(-event_weight),
                    );
            }
            Ok(nll)
        }
    }

    // η-rich, moderate-tail base primaries; βw amplitudes nonzero so the warp
    // and every wiggle cross block are exercised (event row d=1 → entry logS +
    // exit logφ + qdot log_g all live).
    let p0: [f64; KW] = [
        0.25, 0.9, 1.3, 0.6, -0.1, 0.1, -0.2, -0.05, 0.3, // 9 base channels
        0.3, -0.2, // βw_0, βw_1
    ];

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let link = residual_distribution_inverse_link(distribution);
        let value = |p: [f64; KW]| -> f64 {
            program_row_kernel(
                &WiggleProg {
                    link: link.clone(),
                    w: 1.0,
                    d: 1.0,
                    p,
                },
                0,
            )
            .expect("wiggle program value")
            .0
        };
        let h_jet = program_row_kernel(
            &WiggleProg {
                link: link.clone(),
                w: 1.0,
                d: 1.0,
                p: p0,
            },
            0,
        )
        .expect("wiggle program jet")
        .2;

        let hs = 1e-4;
        for a in 0..KW {
            for b in 0..KW {
                let mut pp = p0;
                pp[a] += hs;
                pp[b] += hs;
                let mut pm = p0;
                pm[a] += hs;
                pm[b] -= hs;
                let mut mp = p0;
                mp[a] -= hs;
                mp[b] += hs;
                let mut mm = p0;
                mm[a] -= hs;
                mm[b] -= hs;
                let fd = (value(pp) - value(pm) - value(mp) + value(mm)) / (4.0 * hs * hs);
                let scale = h_jet[a][b].abs().max(fd.abs()).max(1.0);
                assert!(
                    (h_jet[a][b] - fd).abs() <= 2e-3 * scale,
                    "{distribution:?}: wiggle joint Hessian [{a}][{b}] jet {} vs FD {}",
                    h_jet[a][b],
                    fd
                );
            }
        }
    }
}

/// #932 (survival link-wiggle single-source verification): the production
/// wiggle joint Hessian — `survival_ls_wiggle_joint_hessian_dense`, the §13
/// warp row program (`sls_row_nll` extended with `q = q0 + Σ βw·B(q0)` and the
/// qdot coupling `g = hdot + m1·qdot0`) that every production consumer now routes through
/// — must equal an INDEPENDENT tower assembled here from `wiggle_nll` with a
/// hand-rolled `JᵀHJ` pullback. This cross-validates the §13 path AND its
/// row-kernel pullback against independent code; combined with the FD oracle
/// `survival_ls_wiggle_jet_program_joint_hessian_matches_fd_932` (which pins the
/// §13 primary algebra to finite differences), the wiggle joint Hessian is
/// fully verified. The legacy bespoke `assemble_h_wiggle` is RETIRED for wiggle:
/// it disagreed with the §13 source by ~15% at `[0][0]` (a dropped warp coupling
/// — the #736 duplicate-engine genus), so the last production consumer of it (the
/// trust-region metric floor) was repointed to the §13 source.
#[test]
fn survival_ls_wiggle_joint_hessian_matches_assembler_932() {
    use gam_math::jet_scalar::{JetScalar, Order2};

    // event rows (d=1) so entry-logS + exit-logphi + qdot-log_g are all live;
    // moderate-tail primaries clear of the monotonicity guard.
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();

    // Seed indices for the wiggle DESIGN matrix `x_link_wiggle` only (its column
    // count `pw` and `etaw = X·betaw`). The warp basis derivative stacks are
    // evaluated later at the model's unwarped q0 indices, exactly where
    // `sls_row_nll_wiggle` composes them.
    let q0_exit = Array1::from_shape_fn(n, |i| {
        primaries[i][1] - primaries[i][3] * (-primaries[i][6]).exp()
    });

    // A small monotone wiggle basis; degree/knots chosen for a few columns.
    // Degree-3 clamped knot vector: the I-spline derivative path integrates a
    // degree-3 B-spline, whose `validate_knots_for_degree` floor is `2*(3+1)=8`
    // knots, so the previous 5-knot vector aborted this oracle in fixture setup
    // (the basis builder returned `Insufficient knots for degree 3 spline`). The
    // clamp endpoints (-2.5, 3.2) bracket every `q0_exit`/`q0_entry` index above
    // so the warp basis is evaluated strictly inside its support. (Mirrors the
    // clamped 8-knot pattern the other wiggle gate tests in this file use.)
    let knots = array![-2.5, -2.5, -2.5, -2.5, 3.2, 3.2, 3.2, 3.2];
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("wiggle design B(q0_exit)");
    let pw = xwiggle.ncols();
    let betaw = Array1::from_shape_fn(pw, |b| 0.25 - 0.08 * b as f64);

    // The single-source §13 warp evaluated on a generic jet scalar, KW = 9 + pw.
    fn wiggle_nll<const KW: usize, S: JetScalar<KW>>(
        vars: &[S; KW],
        kernel: &SurvivalExactRowKernel,
        pw: usize,
        b0e: &[f64],
        b1e: &[f64],
        b2e: &[f64],
        b0x: &[f64],
        b1x: &[f64],
        b2x: &[f64],
        b3x: &[f64],
    ) -> S {
        let inv_sigma_entry = vars[7].neg().exp();
        let q0 = vars[4].mul(&inv_sigma_entry).neg();
        let inv_sigma_exit = vars[6].neg().exp();
        let q1 = vars[3].mul(&inv_sigma_exit).neg();
        let qdot0 = vars[3].mul(&vars[8]).sub(&vars[5]);
        let mut q0w = q0;
        let mut q1w = q1;
        let mut m1 = S::constant(1.0);
        for j in 0..pw {
            let bw = vars[9 + j];
            q0w = q0w.add(&bw.mul(&q0.compose_unary([b0e[j], b1e[j], b2e[j], 0.0, 0.0])));
            q1w = q1w.add(&bw.mul(&q1.compose_unary([b0x[j], b1x[j], b2x[j], b3x[j], 0.0])));
            m1 = m1.add(&bw.mul(&q1.compose_unary([b1x[j], b2x[j], b3x[j], 0.0, 0.0])));
        }
        // The scale divides the time transform too (#2695): `du1/dt =
        // e^{−η_ls}·g`, and the kernel's rate stack is at `g`.
        let u0w = vars[0].mul(&inv_sigma_entry).add(&q0w);
        let u1w = vars[1].mul(&inv_sigma_exit).add(&q1w);
        let g = vars[2].sub(&vars[1].mul(&vars[8])).add(&m1.mul(&qdot0));
        let mut nll = u0w
            .compose_unary([
                kernel.log_s0,
                -kernel.r0,
                -kernel.dr0,
                -kernel.ddr0,
                -kernel.dddr0,
            ])
            .scale(kernel.w);
        let cw = kernel.w * (1.0 - kernel.d);
        if cw != 0.0 {
            nll = nll.add(
                &u1w.compose_unary([
                    kernel.log_s1,
                    -kernel.r1,
                    -kernel.dr1,
                    -kernel.ddr1,
                    -kernel.dddr1,
                ])
                .scale(-cw),
            );
        }
        let ew = kernel.w * kernel.d;
        if ew != 0.0 {
            nll = nll
                .add(
                    &u1w.compose_unary([
                        kernel.logphi1,
                        kernel.dlogphi1,
                        kernel.d2logphi1,
                        kernel.d3logphi1,
                        kernel.d4logphi1,
                    ])
                    .scale(-ew),
                )
                .add(
                    &g.compose_unary([
                        kernel.log_g,
                        kernel.d_log_g,
                        kernel.d2_log_g,
                        kernel.d3_log_g,
                        kernel.d4_log_g,
                    ])
                    .scale(-ew),
                )
                .add(&vars[6].scale(ew));
        }
        nll
    }

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let mut family =
            survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        family.x_link_wiggle = Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
        ));
        family.wiggle_knots = Some(knots.clone());
        family.wiggle_degree = Some(degree);

        let mut states = survival_ls_joint_oracle_states(&primaries);
        let etaw = xwiggle.dot(&betaw);
        states.push(ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw,
        });

        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");

        // Warp basis derivative stacks at the unwarped model indices q0. The
        // criterion defines q=q0+βB(q0), then u=h+q; `dynamic.q_*` is already q
        // and therefore cannot be the composition center without double-warping.
        let u1_index = &dynamic.q_base_exit;
        let u0_index = &dynamic.q_base_entry;
        let bx0 = survival_wiggle_basis_with_options(
            u1_index.view(),
            &knots,
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let bx1 = survival_wiggle_basis_with_options(
            u1_index.view(),
            &knots,
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();
        let bx2 = survival_wiggle_basis_with_options(
            u1_index.view(),
            &knots,
            degree,
            BasisOptions::second_derivative(),
        )
        .unwrap();
        let bx3 = survival_wiggle_third_basis(u1_index.view(), &knots, degree).unwrap();
        let be0 = survival_wiggle_basis_with_options(
            u0_index.view(),
            &knots,
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let be1 = survival_wiggle_basis_with_options(
            u0_index.view(),
            &knots,
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();
        let be2 = survival_wiggle_basis_with_options(
            u0_index.view(),
            &knots,
            degree,
            BasisOptions::second_derivative(),
        )
        .unwrap();

        // Coefficient offsets = cumulative block beta widths (time,thr,ls,wiggle).
        let widths: Vec<usize> = states.iter().map(|s| s.beta.len()).collect();
        let mut offsets = vec![0usize];
        for w in &widths {
            offsets.push(offsets.last().unwrap() + w);
        }
        let ncoef = *offsets.last().unwrap();
        let wiggle_off = offsets[3];

        // Base-channel design rows via the production SurvivalLsRowKernel.
        let base_kernel = SurvivalLsRowKernel {
            family: &family,
            dynamic: &dynamic,
            deriv_log_scale: 0.0,
            offsets: offsets[0..4].to_vec(),
        };

        let mut h_tower = Array2::<f64>::zeros((ncoef, ncoef));
        for row in 0..n {
            // Per-row primary Hessian from the §13 warp at Order2<9+pw>.
            let pvals = base_kernel.row_primary_values(row);
            let state = family.row_predictor_state_at(&dynamic, row);
            let kernel = family
                .exact_row_kernel_rescaled(row, state, 0.0)
                .expect("exact row kernel")
                .expect("exact row kernel present");
            macro_rules! run_kw {
                ($kw:literal) => {{
                    let mut vars = [<Order2<$kw> as JetScalar<$kw>>::constant(0.0); $kw];
                    for a in 0..9 {
                        vars[a] = <Order2<$kw> as JetScalar<$kw>>::variable(pvals[a], a);
                    }
                    for b in 0..pw {
                        vars[9 + b] = <Order2<$kw> as JetScalar<$kw>>::variable(betaw[b], 9 + b);
                    }
                    let out = wiggle_nll::<$kw, Order2<$kw>>(
                        &vars,
                        &kernel,
                        pw,
                        &be0.row(row).to_vec(),
                        &be1.row(row).to_vec(),
                        &be2.row(row).to_vec(),
                        &bx0.row(row).to_vec(),
                        &bx1.row(row).to_vec(),
                        &bx2.row(row).to_vec(),
                        &bx3.row(row).to_vec(),
                    );
                    let h = out.h();
                    // Channel design rows: base 0..8 via channel_row, βw -> e_b.
                    let mut jrows: Vec<(usize, Vec<f64>)> = Vec::with_capacity(9 + pw);
                    for ch in 0..9usize {
                        match (
                            base_kernel.channel_block(ch),
                            base_kernel.channel_row(ch, row),
                        ) {
                            (Some(bk), Some(r)) => jrows.push((offsets[bk], r.to_vec())),
                            _ => jrows.push((usize::MAX, vec![])),
                        }
                    }
                    for b in 0..pw {
                        let mut e = vec![0.0; pw];
                        e[b] = 1.0;
                        jrows.push((wiggle_off, e));
                    }
                    for a in 0..(9 + pw) {
                        let (oa, ra) = &jrows[a];
                        if *oa == usize::MAX {
                            continue;
                        }
                        for bcol in 0..(9 + pw) {
                            let hab = h[a][bcol];
                            if hab == 0.0 {
                                continue;
                            }
                            let (ob, rb) = &jrows[bcol];
                            if *ob == usize::MAX {
                                continue;
                            }
                            for (ia, &va) in ra.iter().enumerate() {
                                if va == 0.0 {
                                    continue;
                                }
                                let wv = hab * va;
                                for (ib, &vb) in rb.iter().enumerate() {
                                    h_tower[[oa + ia, ob + ib]] += wv * vb;
                                }
                            }
                        }
                    }
                }};
            }
            assert_eq!(9 + pw, 12, "fixture's clamped cubic basis width changed");
            run_kw!(12);
        }

        // #932: the production single-source §13 wiggle joint Hessian
        // (`survival_ls_wiggle_joint_hessian_dense` — the path the Newton step
        // and, after the trust-floor fix, every production consumer now uses)
        // must equal the INDEPENDENT tower assembled above from `wiggle_nll`
        // with a hand-rolled JᵀHJ pullback. The legacy bespoke
        // `assemble_h_wiggle` is retired for wiggle: it disagreed with this
        // tower by ~15% at [0][0] (the duplicate-engine genus #932 eliminates),
        // and is FD-cross-checked separately by
        // `survival_ls_wiggle_jet_program_joint_hessian_matches_fd_932`.
        let dense =
            super::row_kernel::survival_ls_wiggle_joint_hessian_dense(&family, &dynamic, 0.0)
                .expect("§13 dense wiggle Hessian");
        for ((a, b), &dj) in dense.indexed_iter() {
            let tj = h_tower[[a, b]];
            assert!(
                (tj - dj).abs() <= 1e-9 * (1.0 + dj.abs()),
                "{distribution:?}: §13 wiggle joint Hessian [{a}][{b}] dense {dj} != independent tower {tj}"
            );
        }
    }
}

/// #932: the live `evaluate()` block-diagonal target must carry the same
/// single-sourced §13 curvature for the link-wiggle block as the joint Hessian.
///
/// The retired bespoke `assemble_h_wiggle` summed three INDEPENDENT
/// single-channel Gauss-Newton terms
/// (`Xw_exitᵀ(-d2_q1)Xw_exit + Xw_entryᵀ(-d2_q0)Xw_entry +
/// Xw_qdotᵀ(-d2_qdot1)Xw_qdot`) and DROPPED the exit-index ↔ qdot warp
/// cross-coupling: one wiggle coefficient `βw_j` warps BOTH the exit index
/// `q1w = q1 + Σβw·B(q1)` (through `B(q1)`) AND the qdot multiplier
/// `m1 = 1 + Σβw·B'(q1)` (through `B'(q1)`), so the wiggle-wiggle Hessian
/// carries a mixed `∂²ℓ/∂q1∂qdot·(∂q1/∂βw)(∂qdot/∂βw)` term the three-way sum
/// omitted (compare the log-sigma block, which DOES carry its analogous
/// exit↔deriv cross term). That drop mis-scaled the inner LINK_WIGGLE Newton
/// step by ~15% at `[0][0]` — the #736 duplicate-derivative genus. The
/// block-diagonal wiggle block must now equal the `[wiggle, wiggle]` sub-block
/// of the §13 single source (`survival_ls_wiggle_joint_hessian_dense`) exactly.
///
/// The coupling is only live at NONZERO wiggle coefficients, so the fixture
/// seeds monotone-feasible (nonnegative) `βw` rather than the zero warp the
/// other wiggle gates use.
#[test]
fn survival_ls_block_diagonal_wiggle_block_matches_single_source_932() {
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();
    let q0_exit = Array1::from_shape_fn(n, |i| {
        primaries[i][1] - primaries[i][3] * (-primaries[i][6]).exp()
    });
    let knots = array![-2.5, -2.5, -2.5, -2.5, 3.2, 3.2, 3.2, 3.2];
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("wiggle design B(q0_exit)");
    let pw = xwiggle.ncols();
    assert!(
        pw >= 2,
        "fixture must have a multi-column wiggle basis, got {pw}"
    );

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let mut family =
            survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        family.x_link_wiggle = Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
        ));
        family.wiggle_knots = Some(knots.clone());
        family.wiggle_degree = Some(degree);

        // Nonzero, monotone-feasible (nonnegative) wiggle coefficients so the
        // warp coupling term the bespoke assembler dropped is actually live.
        let betaw = Array1::from_shape_fn(pw, |b| 0.02 + 0.005 * b as f64);
        let mut states = survival_ls_joint_oracle_states(&primaries);
        states.push(ParameterBlockState {
            eta: xwiggle.dot(&betaw),
            beta: betaw,
        });

        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("wiggle dynamic geometry");
        let dense =
            super::row_kernel::survival_ls_wiggle_joint_hessian_dense(&family, &dynamic, 0.0)
                .expect("§13 dense joint Hessian");
        let offsets = family.joint_block_offsets();
        let (lo, hi) = (
            offsets[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE],
            offsets[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE + 1],
        );
        let evaluation = family
            .evaluate(&states)
            .expect("live wiggle block evaluation");
        let wiggle_block =
            match &evaluation.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE] {
                BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
                _ => panic!("wiggle block must use exact Newton curvature"),
            };
        assert_eq!(
            wiggle_block.dim(),
            (hi - lo, hi - lo),
            "block-diagonal wiggle block shape"
        );

        for i in 0..(hi - lo) {
            for j in 0..(hi - lo) {
                let bd = wiggle_block[[i, j]];
                let ss = dense[[lo + i, lo + j]];
                assert!(
                    (bd - ss).abs() <= 1e-9 * (1.0 + ss.abs()),
                    "{distribution:?}: block-diagonal wiggle Hessian [{i}][{j}] = {bd:.9e} \
                     disagrees with the §13 single source {ss:.9e} \
                     (drift {:.3e}) — the dropped exit↔qdot warp coupling",
                    (bd - ss).abs()
                );
            }
        }
    }
}

/// Runtime-sized #932 regression: a valid cubic link-wiggle basis wider than
/// the retired 11-column const-generic ceiling must run every production packed
/// derivative channel. This exercises the actual survival family/kernel rather
/// than a validator in isolation.
#[test]
fn survival_ls_wiggle_runtime_backend_runs_above_old_width_ceiling_932() {
    use super::row_kernel::{
        survival_ls_wiggle_directional_derivative_dense, survival_ls_wiggle_joint_hessian_dense,
        survival_ls_wiggle_second_directional_derivative_dense,
    };
    use crate::row_kernel::RowSet;

    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let q0_exit = Array1::from_shape_fn(primaries.len(), |i| {
        primaries[i][1] - primaries[i][3] * (-primaries[i][6]).exp()
    });
    // Cubic clamping (four repeated endpoints) plus eleven interior knots gives
    // fifteen cubic B-splines; the wiggle/I-spline construction drops TWO columns
    // (an earlier count assumed one), leaving thirteen wiggle columns — safely
    // above the retired pw<=11 const-generic production ceiling the runtime arena
    // backend must now clear. (Asserted below so a basis-width change re-trips.)
    let knots = Array1::from_vec(vec![
        -2.5, -2.5, -2.5, -2.5, -2.3, -2.0, -1.4, -0.8, -0.2, 0.4, 1.0, 1.6, 2.2, 2.7, 3.0, 3.2,
        3.2, 3.2, 3.2,
    ]);
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("wide wiggle design");
    let pw = xwiggle.ncols();
    assert!(
        pw > 11,
        "fixture must exceed the retired width ceiling, got {pw}"
    );

    let inverse_link = residual_distribution_inverse_link(ResidualDistribution::Gaussian);
    let mut family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
    family.x_link_wiggle = Some(DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
    ));
    family.wiggle_knots = Some(knots);
    family.wiggle_degree = Some(degree);

    let mut states = survival_ls_joint_oracle_states(&primaries);
    let betaw = Array1::<f64>::zeros(pw);
    states.push(ParameterBlockState {
        beta: betaw.clone(),
        eta: xwiggle.dot(&betaw),
    });
    let dynamic = family
        .build_dynamic_geometry(&states)
        .expect("wide dynamic geometry");
    let n_coefficients: usize = states.iter().map(|state| state.beta.len()).sum();
    let direction_u: Vec<f64> = (0..n_coefficients)
        .map(|axis| 0.01 * (axis as f64 + 1.0))
        .collect();
    let direction_v: Vec<f64> = (0..n_coefficients)
        .map(|axis| -0.007 * (axis as f64 + 0.5))
        .collect();

    let hessian = survival_ls_wiggle_joint_hessian_dense(&family, &dynamic, 0.0)
        .expect("wide runtime Hessian");
    let third = survival_ls_wiggle_directional_derivative_dense(
        &family,
        &dynamic,
        0.0,
        &RowSet::All,
        &direction_u,
    )
    .expect("wide runtime contracted third");
    let fourth = survival_ls_wiggle_second_directional_derivative_dense(
        &family,
        &dynamic,
        0.0,
        &RowSet::All,
        &direction_u,
        &direction_v,
    )
    .expect("wide runtime contracted fourth");
    for matrix in [&hessian, &third, &fourth] {
        assert_eq!(matrix.dim(), (n_coefficients, n_coefficients));
        assert!(matrix.iter().all(|value| value.is_finite()));
    }
}

/// #932 gap (c): a DIRECT third- AND fourth-order oracle on the PRODUCTION
/// survival-LS link-wiggle path. The existing direct wiggle tests pin only the
/// value/gradient/Hessian; the higher-order channels the log-det adjoint
/// consumes (`Γ_a = tr(H⁻¹ ∂H/∂θ_a)` and its second directional) had no
/// independent production witness. This drives the two live higher-order
/// entry points —
///   * third order: [`survival_ls_wiggle_directional_derivative_dense`],
///     the contracted `D_dir H = Σ_c ℓ_abc dir_c`,
///   * fourth order: [`survival_ls_wiggle_second_directional_derivative_dense`],
///     the contracted `D_u D_v H = Σ_cd ℓ_abcd u_c v_d`,
/// — and pins each against an INDEPENDENT Richardson-extrapolated 5-point
/// central-difference witness built from OTHER production entry points, exactly as
/// `flex_verify_932_tests` differences the hand path's own returned value:
///   * `D_dir H` is cross-checked against a Richardson derivative of the
///     production joint Hessian [`survival_ls_wiggle_joint_hessian_dense`]
///     along the coefficient direction `dir` (the coefficient→KW-primary map is
///     linear, so `d/ds H(β + s·dir)|₀` IS `D_dir H`, with no dropped `dJ/dβ`
///     term), and
///   * `D_u D_v H` against a Richardson derivative of the production directional
///     `D_u H(β + s·v)` along `v`.
/// FD stencils use independent arithmetic (analytic packed `OneSeed`/`TwoSeed`
/// jets vs finite differences of the `Order2` Hessian), so agreement is a true
/// correctness proof of the higher-order jets; a dropped warp-coupling term
/// would show O(1) relative error, far above the bounds asserted here.
#[test]
fn survival_ls_wiggle_third_and_fourth_directional_match_fd_932() {
    use super::row_kernel::{
        survival_ls_wiggle_directional_derivative_dense, survival_ls_wiggle_joint_hessian_dense,
        survival_ls_wiggle_second_directional_derivative_dense,
    };
    use crate::row_kernel::RowSet;

    // Event and mixed censoring rows; moderate-tail primaries clear of the monotonicity guard,
    // matching the joint-Hessian oracle's regime so the ±h·dir stencils stay in
    // the smooth interior of the warp basis and the residual link.
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 1.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();

    let q0_exit = Array1::from_shape_fn(n, |i| {
        primaries[i][1] - primaries[i][3] * (-primaries[i][6]).exp()
    });
    let knots = array![-2.5, -2.5, -2.5, -2.5, 3.2, 3.2, 3.2, 3.2];
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("wiggle design B(q0_exit)");
    let pw = xwiggle.ncols();
    // Wiggle amplitude must keep the survival monotonicity contract
    // (d_eta/dt > 0 at every row) satisfied at the base point AND across the
    // FD stencil's ±s sweeps; 0.25-scale coefficients drove row 1 to
    // d_eta/dt = -2.6e-3 and production (correctly) refused the fixture.
    let betaw = Array1::from_shape_fn(pw, |b| 0.06 - 0.02 * b as f64);
    // Coefficient layout: [time(1), threshold(1), log_sigma(1), wiggle(pw)].
    let ncoef = 3 + pw;

    for (distribution, event) in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ]
    .into_iter()
    .flat_map(|distribution| [event, [0.0, 1.0, 0.0, 1.0]].map(|event| (distribution, event)))
    {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let mut family =
            survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        family.x_link_wiggle = Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
        ));
        family.wiggle_knots = Some(knots.clone());
        family.wiggle_degree = Some(degree);

        // Build dynamic geometry at the perturbed coefficient vector `β + δ`. The
        // oracle-family eta vectors are the raw primary channels at β = 1, so a
        // base-block coefficient `1 + δ_k` scales that block's eta linearly; the
        // wiggle block re-forms `etaw = X·(βw + δw)`.
        let build = |bt: f64, bthr: f64, bls: f64, bw: &Array1<f64>| {
            let mut states = survival_ls_joint_oracle_states(&primaries);
            states[0].eta.mapv_inplace(|e| e * bt);
            states[0].beta = array![bt];
            states[1].eta.mapv_inplace(|e| e * bthr);
            states[1].beta = array![bthr];
            states[2].eta.mapv_inplace(|e| e * bls);
            states[2].beta = array![bls];
            let etaw = xwiggle.dot(bw);
            states.push(ParameterBlockState {
                beta: bw.clone(),
                eta: etaw,
            });
            let dynamic = family
                .build_dynamic_geometry(&states)
                .expect("dynamic geometry");
            dynamic
        };

        // Coefficient vector `β + s·dir` split back into per-block pieces.
        let perturbed = |s: f64, dir: &[f64], bw: &mut Array1<f64>| -> (f64, f64, f64) {
            for b in 0..pw {
                bw[b] = betaw[b] + s * dir[3 + b];
            }
            (1.0 + s * dir[0], 1.0 + s * dir[1], 1.0 + s * dir[2])
        };

        let hessian_at = |s: f64, dir: &[f64]| {
            let mut bw = betaw.clone();
            let (bt, bthr, bls) = perturbed(s, dir, &mut bw);
            let dynamic = build(bt, bthr, bls, &bw);
            survival_ls_wiggle_joint_hessian_dense(&family, &dynamic, 0.0)
                .expect("§13 dense wiggle Hessian")
        };
        let directional_at = |s: f64, dir_v: &[f64], dir_u: &[f64]| {
            let mut bw = betaw.clone();
            let (bt, bthr, bls) = perturbed(s, dir_v, &mut bw);
            let dynamic = build(bt, bthr, bls, &bw);
            survival_ls_wiggle_directional_derivative_dense(
                &family,
                &dynamic,
                0.0,
                &RowSet::All,
                dir_u,
            )
            .expect("§13 dense wiggle directional")
        };
        // Fourth-order 5-point first derivative of a matrix-valued map at s = 0:
        // f'(0) ≈ (−f(2h) + 8 f(h) − 8 f(−h) + f(−2h)) / (12 h).
        let five_point = |fph: &Array2<f64>,
                          fp2h: &Array2<f64>,
                          fmh: &Array2<f64>,
                          fm2h: &Array2<f64>,
                          h: f64| {
            for matrix in [fph, fp2h, fmh, fm2h] {
                assert_eq!(matrix.dim(), (ncoef, ncoef));
                assert!(
                    matrix.iter().all(|value| value.is_finite()),
                    "{distribution:?} event={event:?}: non-finite stencil input"
                );
            }
            (fp2h.mapv(|x| -x) + fph.mapv(|x| 8.0 * x) - fmh.mapv(|x| 8.0 * x) + fm2h) / (12.0 * h)
        };

        let dynamic0 = build(1.0, 1.0, 1.0, &betaw);

        // DIAGNOSTIC (#932): report max relative error per direction family to
        // localize the coeff-space FD vs analytic pullback convention gap.
        let mk = |du: &[f64], dv: &[f64], label: &str| {
            let d_dir_analytic = survival_ls_wiggle_directional_derivative_dense(
                &family,
                &dynamic0,
                0.0,
                &RowSet::All,
                du,
            )
            .expect("analytic first directional");
            let d2_analytic = survival_ls_wiggle_second_directional_derivative_dense(
                &family,
                &dynamic0,
                0.0,
                &RowSet::All,
                du,
                dv,
            )
            .expect("analytic second directional");
            let h3 = 1e-2;
            let coarse_third = five_point(
                &hessian_at(h3, du),
                &hessian_at(2.0 * h3, du),
                &hessian_at(-h3, du),
                &hessian_at(-2.0 * h3, du),
                h3,
            );
            let fine_h3 = 0.5 * h3;
            let fine_third = five_point(
                &hessian_at(fine_h3, du),
                &hessian_at(2.0 * fine_h3, du),
                &hessian_at(-fine_h3, du),
                &hessian_at(-2.0 * fine_h3, du),
                fine_h3,
            );
            // The 5-point stencil has leading error C h^4. Extrapolating the
            // h and h/2 estimates cancels that term exactly, leaving O(h^6).
            let coarse_extrap_third = (fine_third.mapv(|x| 16.0 * x) - &coarse_third) / 15.0;
            let finer_h3 = 0.5 * fine_h3;
            let finer_third = five_point(
                &hessian_at(finer_h3, du),
                &hessian_at(2.0 * finer_h3, du),
                &hessian_at(-finer_h3, du),
                &hessian_at(-2.0 * finer_h3, du),
                finer_h3,
            );
            let middle_extrap_third = (finer_third.mapv(|x| 16.0 * x) - &fine_third) / 15.0;
            let fd_third = middle_extrap_third;
            let mut third_max = 0.0_f64;
            let mut third_at = (0, 0);
            let mut third_remainder = 0.0_f64;
            let mut third_coarse_change = 0.0_f64;
            let mut third_fine_change = 0.0_f64;
            for ((a, b), &analytic) in d_dir_analytic.indexed_iter() {
                assert!(
                    [analytic, fd_third[[a, b]], coarse_extrap_third[[a, b]]]
                        .iter()
                        .all(|value| value.is_finite()),
                    "{distribution:?} event={event:?} {label}: non-finite third[{a},{b}]"
                );
                let e = (analytic - fd_third[[a, b]]).abs() / (1.0 + analytic.abs());
                if e > third_max {
                    third_max = e;
                    third_at = (a, b);
                }
                third_remainder = third_remainder.max(
                    (fd_third[[a, b]] - coarse_extrap_third[[a, b]]).abs()
                        / (63.0 * (1.0 + fd_third[[a, b]].abs())),
                );
                let scale = 1.0 + finer_third[[a, b]].abs();
                third_coarse_change = third_coarse_change
                    .max((fine_third[[a, b]] - coarse_third[[a, b]]).abs() / scale);
                third_fine_change =
                    third_fine_change.max((finer_third[[a, b]] - fine_third[[a, b]]).abs() / scale);
            }
            let third_ratio = third_coarse_change / third_fine_change;
            // An order-four stencil should contract by 2^4=16 under step
            // halving. Once both changes are already below the correctness
            // gate, roundoff dominates and no order estimate is needed;
            // otherwise require observed order in [3,5].
            let third_order_observed =
                third_fine_change < 1.0e-5 || (8.0..=32.0).contains(&third_ratio);
            let h4 = 2e-2;
            let coarse_fourth = five_point(
                &directional_at(h4, dv, du),
                &directional_at(2.0 * h4, dv, du),
                &directional_at(-h4, dv, du),
                &directional_at(-2.0 * h4, dv, du),
                h4,
            );
            let fine_h4 = 0.5 * h4;
            let fine_fourth = five_point(
                &directional_at(fine_h4, dv, du),
                &directional_at(2.0 * fine_h4, dv, du),
                &directional_at(-fine_h4, dv, du),
                &directional_at(-2.0 * fine_h4, dv, du),
                fine_h4,
            );
            let coarse_extrap_fourth = (fine_fourth.mapv(|x| 16.0 * x) - &coarse_fourth) / 15.0;
            let finer_h4 = 0.5 * fine_h4;
            let finer_fourth = five_point(
                &directional_at(finer_h4, dv, du),
                &directional_at(2.0 * finer_h4, dv, du),
                &directional_at(-finer_h4, dv, du),
                &directional_at(-2.0 * finer_h4, dv, du),
                finer_h4,
            );
            let middle_extrap_fourth = (finer_fourth.mapv(|x| 16.0 * x) - &fine_fourth) / 15.0;
            let fd_fourth = middle_extrap_fourth;
            let mut fourth_max = 0.0_f64;
            let mut fourth_at = (0, 0);
            let mut fourth_remainder = 0.0_f64;
            let mut fourth_coarse_change = 0.0_f64;
            let mut fourth_fine_change = 0.0_f64;
            for ((a, b), &analytic) in d2_analytic.indexed_iter() {
                assert!(
                    [analytic, fd_fourth[[a, b]], coarse_extrap_fourth[[a, b]]]
                        .iter()
                        .all(|value| value.is_finite()),
                    "{distribution:?} event={event:?} {label}: non-finite fourth[{a},{b}]"
                );
                let e = (analytic - fd_fourth[[a, b]]).abs() / (1.0 + analytic.abs());
                if e > fourth_max {
                    fourth_max = e;
                    fourth_at = (a, b);
                }
                fourth_remainder = fourth_remainder.max(
                    (fd_fourth[[a, b]] - coarse_extrap_fourth[[a, b]]).abs()
                        / (63.0 * (1.0 + fd_fourth[[a, b]].abs())),
                );
                let scale = 1.0 + finer_fourth[[a, b]].abs();
                fourth_coarse_change = fourth_coarse_change
                    .max((fine_fourth[[a, b]] - coarse_fourth[[a, b]]).abs() / scale);
                fourth_fine_change = fourth_fine_change
                    .max((finer_fourth[[a, b]] - fine_fourth[[a, b]]).abs() / scale);
            }
            let fourth_ratio = fourth_coarse_change / fourth_fine_change;
            let fourth_order_observed =
                fourth_fine_change < 1.0e-4 || (8.0..=32.0).contains(&fourth_ratio);
            eprintln!(
                "ZZ932 {distribution:?} event={event:?} {label}: third_max={third_max:.3e} at {third_at:?} \
                 (analytic={:+.9e}, fd={:+.9e}, remainder={third_remainder:.3e}, \
                 raw_ratio={third_ratio:.3}), \
                 fourth_max={fourth_max:.3e} at {fourth_at:?} \
                 (analytic={:+.9e}, fd={:+.9e}, remainder={fourth_remainder:.3e}, \
                 raw_ratio={fourth_ratio:.3})",
                d_dir_analytic[third_at],
                fd_third[third_at],
                d2_analytic[fourth_at],
                fd_fourth[fourth_at],
            );
            (
                third_max,
                third_at,
                third_remainder,
                third_order_observed,
                fourth_max,
                fourth_at,
                fourth_remainder,
                fourth_order_observed,
            )
        };

        let full_u: Vec<f64> = (0..ncoef)
            .map(|c| match c {
                0 => 0.7,
                1 => -0.5,
                2 => 0.4,
                _ => 0.3 - 0.11 * (c - 3) as f64,
            })
            .collect();
        let full_v: Vec<f64> = (0..ncoef)
            .map(|c| match c {
                0 => -0.35,
                1 => 0.6,
                2 => -0.45,
                _ => -0.12 + 0.09 * (c - 3) as f64,
            })
            .collect();
        let wig = |base: &[f64]| -> Vec<f64> {
            (0..ncoef)
                .map(|c| if c < 3 { 0.0 } else { base[c] })
                .collect()
        };
        let baseonly = |base: &[f64]| -> Vec<f64> {
            (0..ncoef)
                .map(|c| if c < 3 { base[c] } else { 0.0 })
                .collect()
        };
        let wiggle_u = wig(&full_u);
        let wiggle_v = wig(&full_v);
        let base_u = baseonly(&full_u);
        let base_v = baseonly(&full_v);
        let cases = [
            ("FULL", mk(&full_u, &full_v, "FULL")),
            ("WIGGLE_ONLY", mk(&wiggle_u, &wiggle_v, "WIGGLE_ONLY")),
            ("BASE_ONLY", mk(&base_u, &base_v, "BASE_ONLY")),
        ];
        let mut failures = Vec::new();
        for (
            label,
            (
                third_max,
                third_at,
                third_remainder,
                third_order_observed,
                fourth_max,
                fourth_at,
                fourth_remainder,
                fourth_order_observed,
            ),
        ) in cases
        {
            // The header's contract: a dropped warp-coupling term shows O(1)
            // relative error, so these bounds gate correctness while leaving
            // generous room for the extrapolated stencils' own truncation/
            // cancellation noise (coarse h=1e-2 / 2e-2).
            if third_max >= 1.0e-5 {
                failures.push(format!("{label} third={third_max:.3e} at {third_at:?}"));
            }
            if third_remainder >= 1.0e-5 {
                failures.push(format!(
                    "{label} third Richardson remainder={third_remainder:.3e}"
                ));
            }
            if !third_order_observed {
                failures.push(format!(
                    "{label} third stencil did not show order-four convergence"
                ));
            }
            if fourth_max >= 1.0e-4 {
                failures.push(format!("{label} fourth={fourth_max:.3e} at {fourth_at:?}"));
            }
            if fourth_remainder >= 1.0e-4 {
                failures.push(format!(
                    "{label} fourth Richardson remainder={fourth_remainder:.3e}"
                ));
            }
            if !fourth_order_observed {
                failures.push(format!(
                    "{label} fourth stencil did not show order-four convergence"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{distribution:?} analytic higher directional mismatch(es): {}",
            failures.join(", ")
        );
    }
}

/// #932 link-wiggle DOUBLE-WARP guard (the decisive production oracle for the
/// single-warp fix). The link warp is a SINGLE warp of the residual index:
/// `build_dynamic_geometry` composes `q = q0 + Σ βw·B(q0)` ONCE and the family's
/// direct log-likelihood (`log_likelihood_only`, replicated per-row below via the
/// production `row_predictor_state_at` + `exact_row_kernel` at the once-warped
/// `dynamic.q_exit`) is the correct single-warp objective. The single-source
/// wiggle jet kernel reconstructs that SAME warp from the UNWARPED base
/// predictors (`q_base_exit`/`q_base_entry`) with βw a live differentiable
/// variable. So the kernel's row NLL VALUE must equal `-log_likelihood` per row.
///
/// This is the invariant a re-warp violates: the previous kernel read the
/// already-warped primaries AND composed the warp a second time, so this oracle
/// would have failed by O(1). It calls PRODUCTION on both sides (the wiggle
/// kernel value vs the direct-objective log-likelihood) — not a replay — so it is
/// the true cross-check that the double-warp is gone and the reconstructed warp
/// matches the geometry's single warp exactly.
#[test]
fn survival_ls_wiggle_kernel_value_matches_direct_loglik_932() {
    use super::row_kernel::SurvivalLsWiggleRowKernel;
    use gam_math::jet_scalar::{DynamicJetArena, RuntimeJetScalar};

    // Reuse the directional oracle's regime: event rows, moderate-tail primaries
    // clear of the monotonicity guard, cubic wiggle with nonzero amplitudes so
    // the warp (and the h≠0 baseline that distinguishes residual-index warping
    // from full-index warping) is genuinely exercised.
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 0.0]; // mix event + censored so both loglik arms are pinned
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();

    let q0_exit = Array1::from_shape_fn(n, |i| {
        primaries[i][1] - primaries[i][3] * (-primaries[i][6]).exp()
    });
    let knots = array![-2.5, -2.5, -2.5, -2.5, 3.2, 3.2, 3.2, 3.2];
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("wiggle design B(q0_exit)");
    let pw = xwiggle.ncols();
    let betaw = Array1::from_shape_fn(pw, |b| 0.06 - 0.02 * b as f64);

    for distribution in [
        ResidualDistribution::Gaussian,
        ResidualDistribution::Gumbel,
        ResidualDistribution::Logistic,
    ] {
        let inverse_link = residual_distribution_inverse_link(distribution);
        let mut family =
            survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
        family.x_link_wiggle = Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
        ));
        family.wiggle_knots = Some(knots.clone());
        family.wiggle_degree = Some(degree);

        let mut states = survival_ls_joint_oracle_states(&primaries);
        // Base blocks at the identity coefficient (β=1: the oracle-family eta
        // vectors ARE the raw primary channels), mirroring the directional oracle.
        states[0].beta = array![1.0];
        states[1].beta = array![1.0];
        states[2].beta = array![1.0];
        states.push(ParameterBlockState {
            beta: betaw.clone(),
            eta: xwiggle.dot(&betaw),
        });
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");

        // Wiggle jet kernel: single warp reconstructed from the UNWARPED base
        // predictors, βw live.
        let kernel =
            SurvivalLsWiggleRowKernel::new(&family, &dynamic, 0.0).expect("wiggle row kernel");

        for row in 0..n {
            let arena = DynamicJetArena::new();
            let kernel_nll = kernel
                .row_order2(row, &arena)
                .expect("wiggle kernel value")
                .value();
            // Direct single-warp objective: exactly `log_likelihood_only`'s per-row
            // body, reading the geometry's ONCE-warped index `dynamic.q_exit`.
            let state = family.row_predictor_state_at(&dynamic, row);
            let direct_ll = family
                .exact_row_kernel(row, state)
                .expect("exact row kernel")
                .map_or(0.0, |kernel| kernel.log_likelihood_at(&state));
            // kernel value is the row NLL = -log_likelihood; equality proves the
            // reconstructed single warp == the geometry's single warp.
            assert!(
                (kernel_nll + direct_ll).abs() <= 1e-9,
                "{distribution:?} row {row}: wiggle kernel NLL {kernel_nll} != -direct log-lik {}; \
                 a double-warp would make these differ by O(1)",
                -direct_ll
            );
        }
    }
}

/// #1569: the post-update monotone-cone feasibility check
/// ([`validate_linear_constraints`]) must accept any β the DOWNSTREAM gates
/// (`check_linear_feasibility` / `project_onto_linear_constraints`) already
/// certify as feasible — both certify to the absolute
/// [`MONOTONE_CONE_FEASIBILITY_GATE_TOL`] (`1e-8`, gam#797/#1108). A binding
/// guard row left at slack ~-5e-9 by accumulated inner-solve round-off is
/// numerically AT the boundary, not a violation; the previous `1e-10·scale`
/// threshold hard-errored it, failing an otherwise-feasible survival-LS fit on a
/// pure numerical-precision mismatch. The floor at the gate tolerance fixes that
/// while still rejecting a genuine violation an order of magnitude past the gate.
#[test]
fn validate_linear_constraints_accepts_roundoff_feasible_iterate_1569() {
    // One guard row `β_0 ≥ 0`: A = [[1]], b = [0], so the row scale is 1 and the
    // effective tolerance is `max(1e-10·1, 1e-8) = 1e-8`.
    let constraints = LinearInequalityConstraints {
        a: array![[1.0]],
        b: array![0.0],
    };

    // Round-off-feasible: slack = -5e-9, INSIDE the 1e-8 downstream gate. The
    // rest of the pipeline treats this iterate as feasible, so the post-update
    // sanity check must NOT reject it.
    let roundoff = Array1::from_vec(vec![-5e-9]);
    assert!(
        validate_linear_constraints("test", &roundoff, &constraints).is_ok(),
        "a β at slack -5e-9 (feasible to the 1e-8 gate) must not be rejected"
    );

    // Strictly interior: trivially accepted.
    let interior = Array1::from_vec(vec![0.5]);
    assert!(validate_linear_constraints("test", &interior, &constraints).is_ok());

    // Round-off-feasible exactly at the previous (too-strict) 1e-10 boundary —
    // also accepted now (it is well inside the gate).
    let near_old_floor = Array1::from_vec(vec![-9e-10]);
    assert!(validate_linear_constraints("test", &near_old_floor, &constraints).is_ok());

    // Genuine violation an order of magnitude PAST the gate: slack = -1e-7. Must
    // still be REJECTED — the floor only relaxes round-off, not real violations.
    let violation = Array1::from_vec(vec![-1e-7]);
    let err = validate_linear_constraints("test", &violation, &constraints)
        .expect_err("a β at slack -1e-7 (10x past the 1e-8 gate) must be rejected");
    assert!(
        err.contains("violates represented linear constraint"),
        "unexpected error message: {err}"
    );
}

/// Build a strongly-heteroscedastic survival LS family whose LOCATION (threshold)
/// block has two coefficients with disjoint row support: coefficient 0 loads only
/// on the small-σ rows (where the log-scale predictor `η_σ` is very negative ⇒ a
/// LARGE `inv_sigma = exp(−η_σ)`), and coefficient 1 loads only on the large-σ
/// rows (small `inv_sigma`). The location channel enters the standardized index
/// as `u = inv_sigma·(h − η_t)`, so `∂u/∂η_t = −inv_sigma` and the
/// location-location likelihood-Hessian diagonal — part of the joint trust metric
/// `D` the joint-Newton globalization whitens by — scales as
/// `Σ_r exp(−2 η_σ,r) X_{rj}²`. Coefficient 0's metric entry is therefore many
/// orders of magnitude ABOVE coefficient 1's: the #1569 metric-starvation regime.
/// The scale divides the time transform too (#2695), so the time baseline is
/// small enough that `u = inv_sigma·h` stays moderate on the small-σ rows: a
/// deep-tail `u` there would flatten the curvature the ratio is built from.
fn survival_ls_heteroscedastic_two_col_location_family()
-> (SurvivalLocationScaleFamily, Vec<ParameterBlockState>) {
    // Six rows: the first three sit at very small σ (η_σ ≈ −5, inv_sigma ≈ 148),
    // the last three at large σ (η_σ ≈ +3, inv_sigma ≈ 0.05). The 2-column
    // LOCATION design is block-disjoint so each location coefficient loads on
    // exactly one σ regime. The wide Δη_σ = 8 split (`exp(2·8) ≈ 9e6`) pushes the
    // location-block metric ratio PAST the floor's metric-condition cap (1e6), so
    // the floor genuinely binds — this is the harder-than-the-gate regime #1569
    // targets.
    let n = 6usize;
    // Benign single-column time baseline.
    let x_time = array![[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]];
    // log-σ design: a single column; with β_ls = 1 the small-σ rows get η_σ = −5
    // and the large-σ rows get η_σ = +3.
    let x_log_sigma = array![[-5.0], [-5.0], [-5.0], [3.0], [3.0], [3.0]];
    // Location (threshold) design: two disjoint columns, one per σ regime.
    let x_threshold = array![
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ];
    let family = SurvivalLocationScaleFamily {
        n,
        y: array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        w: array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: 1e-8,
        x_time_entry: Arc::new(x_time.clone()),
        x_time_exit: Arc::new(x_time.clone()),
        x_time_deriv: Arc::new(x_time.clone()),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        time_linear_constraints: None,
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_threshold.clone(),
        )),
        x_threshold_entry: None,
        x_threshold_deriv: None,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_log_sigma.clone(),
        )),
        x_log_sigma_entry: None,
        x_log_sigma_deriv: None,
        x_link_wiggle: None,
        wiggle_knots: None,
        wiggle_degree: None,
        location_log_time: None,
        entry_active: Arc::from(vec![true; n]),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: true,
    };
    // Block betas: a small time β (so `u = inv_sigma·h` is O(1) where inv_sigma ≈ 148);
    // zero location β; β_ls = 1 so η_σ realizes the
    // −3 / +1 split above.
    let beta_t = array![0.005];
    let beta_thr = array![0.0, 0.0];
    let beta_ls = array![1.0];
    let mut eta_time = Array1::<f64>::zeros(3 * n);
    for i in 0..n {
        eta_time[i] = x_time[[i, 0]] * beta_t[0];
        eta_time[n + i] = x_time[[i, 0]] * beta_t[0];
        eta_time[2 * n + i] = x_time[[i, 0]] * beta_t[0];
    }
    let eta_thr = Array1::from_iter(
        (0..n).map(|i| x_threshold[[i, 0]] * beta_thr[0] + x_threshold[[i, 1]] * beta_thr[1]),
    );
    let eta_ls = Array1::from_iter((0..n).map(|i| x_log_sigma[[i, 0]] * beta_ls[0]));
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_time,
        },
        ParameterBlockState {
            beta: beta_thr,
            eta: eta_thr,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
    ];
    (family, states)
}

/// #1569: the scale-aware time-block trust-metric floor must (a) engage on a
/// strongly heteroscedastic coupled fit, and (b) cap the dynamic range that
/// `exp(−η_σ)` injects into the TIME block's trust metric, so the
/// affine-covariant joint-Newton step cannot over-reach on a metric-starved time
/// coordinate. This is the mechanism-level regression guard for the globalization
/// fix: it asserts the BEFORE-state (a pathological metric ratio, far worse than
/// the cap) and the AFTER-state (the floor brings the ratio to the cap).
#[test]
fn survival_ls_scale_aware_location_block_trust_metric_floor_caps_starvation_1569() {
    let (family, states) = survival_ls_heteroscedastic_two_col_location_family();
    // `joint_trust_metric_block_floor` validates specs against
    // `joint_block_dims()` — one spec per block, each as wide as its design.
    // An empty vector is an inconsistent partition, not "no specs needed".
    let specs = vec![
        spec_from_dense_for_test(
            "time",
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                family.x_time_entry.as_ref().clone(),
            )),
            200,
        ),
        spec_from_dense_for_test("threshold", family.x_threshold.clone(), 150),
        spec_from_dense_for_test("log_sigma", family.x_log_sigma.clone(), 120),
    ];

    let offsets = family.joint_block_offsets();
    let (loc_start, loc_end) = (
        offsets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD],
        offsets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD + 1],
    );
    assert_eq!(loc_end - loc_start, 2, "two-column location block expected");

    // ---- raw (pre-floor) joint trust metric = joint Hessian diagonal. This is
    // exactly the diagonal the generic joint-Newton driver whitens by before the
    // family floor is applied.
    let log_scale = family.hessian_deriv_log_rescale(&states);
    let dynamic = family
        .build_dynamic_geometry(&states)
        .expect("trust-metric dynamic geometry");
    let h_diagonal = match family
        .survival_ls_coefficient_hessian(
            &dynamic,
            log_scale,
            None,
            SlsCoefficientHessianTarget::DiagonalOnly,
        )
        .expect("packed diagonal-only joint Hessian")
    {
        SlsCoefficientHessian::DiagonalOnly(diagonal) => diagonal,
        _ => panic!("diagonal target returned another packed SLS shape"),
    };
    let raw_diag: Vec<f64> = (loc_start..loc_end).map(|j| h_diagonal[j].abs()).collect();
    let raw_max = raw_diag.iter().copied().fold(0.0_f64, f64::max);
    let raw_min = raw_diag.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        raw_max.is_finite() && raw_min.is_finite() && raw_min > 0.0,
        "raw location-block metric diagonal must be finite and positive: {raw_diag:?}"
    );
    let raw_ratio = raw_max / raw_min;

    // BEFORE-state: the exp(−η_σ) split (η_σ ∈ {−5, +3}) drives a HUGE dynamic
    // range into the location-block metric — coefficient 0 (small-σ rows) sees
    // ~exp(−2·−5)=exp(10) curvature via `∂u/∂η_t = −inv_sigma`, coefficient 1
    // (large-σ rows) ~exp(−2·3)=exp(−6), a ratio ~exp(16) ≈ 9e6 from the scale
    // alone. It must EXCEED the metric-condition cap (1e6) the floor enforces, so
    // the floor genuinely binds and does real work on this regime.
    let cap = 1.0 / SCALE_COUPLED_TRUST_METRIC_FLOOR_REL; // 1e6
    assert!(
        raw_ratio > cap,
        "expected a pathological pre-floor location-metric ratio above the cap \
         {cap:.0e} (the #1569 exp(−η_σ) starvation), got {raw_ratio:.3e} \
         (diag={raw_diag:?})"
    );

    // ---- AFTER-state: the family floor caps the location-block metric ratio.
    let floor = family
        .joint_trust_metric_block_floor(&states, &specs)
        .expect("floor computation")
        .expect("strongly heteroscedastic coupled fit must produce a floor");
    assert_eq!(floor.len(), offsets[offsets.len() - 1], "full-width floor");
    // The floor is zero on the TIME block, which is left unfloored; positive on the
    // scale-coupled location block.
    let (time_start, time_end) = (
        offsets[SurvivalLocationScaleFamily::BLOCK_TIME],
        offsets[SurvivalLocationScaleFamily::BLOCK_TIME + 1],
    );
    for j in time_start..time_end {
        assert_eq!(
            floor[j], 0.0,
            "floor must be zero on the unfloored time block at {j}"
        );
    }
    for j in loc_start..loc_end {
        assert!(
            floor[j] > 0.0,
            "floor must be positive on the location block at {j}"
        );
    }
    // Apply the floor exactly as the driver does: D_i ← max(D_i, floor_i).
    let floored: Vec<f64> = (0..2)
        .map(|j| raw_diag[j].max(floor[loc_start + j]))
        .collect();
    let floored_max = floored.iter().copied().fold(0.0_f64, f64::max);
    let floored_min = floored.iter().copied().fold(f64::INFINITY, f64::min);
    let floored_ratio = floored_max / floored_min;
    // The floor caps the ratio at the metric-condition cap (1e6); it strictly
    // tightens the starved coordinate's metric and never loosens the dominant one.
    assert!(
        floored_ratio <= cap * (1.0 + 1e-9),
        "floor must cap the location-block metric ratio at {cap:.0e}, got {floored_ratio:.3e}"
    );
    assert!(
        floored_ratio < raw_ratio,
        "floor must REDUCE the location-block metric ratio: before={raw_ratio:.3e} \
         after={floored_ratio:.3e}"
    );
    // The floor only raised the STARVED coordinate (the dominant one is unchanged).
    assert_eq!(
        floored_max, raw_max,
        "floor must not loosen the dominant location-coordinate metric"
    );
}

/// Regression for gam#2112: the reduced constant-scale parametric-AFT MLE must
/// CONVERGE on benign fully-observed lognormal data and recover the closed-form
/// lognormal MLE, `μ̂ = mean(log t)` and `σ̂ = sd(log t)` (population/MLE `1/n`
/// variance).
///
/// Before the fix, `fit_parametric_aft_direct_mle` certified stationarity with
/// an ABSOLUTE tolerance on the sup-norm of the SUMMED log-likelihood gradient
/// `g = ∇ℓ`, floored at `REDUCED_AFT_*_TOL_FLOOR = 1e-8`. Because `g` is a sum
/// over the `n` rows, its attainable round-off floor at the true MLE grows like
/// `n·ε`, so for `n ≳ 1000` that floor exceeds the fixed tolerance and the loop
/// runs all `max_iter` iterations and hard-errors "failed to converge" on data
/// whose MLE is closed-form. Empirically, on this `n = 2000` sample the summed
/// gradient plateaus at a sup-norm of `≈ 2.3e-7` at the numerical optimum (where
/// the half-Newton-decrement `½·gᵀH⁻¹g ≈ 1e-17`, i.e. machine-zero suboptimality
/// and `μ̂/σ̂` recovered to `~1e-6`). With `tol = 1e-8` the OLD gate demanded
/// `|g|_∞ ≤ 1e-8`, which that `2.3e-7` floor can NEVER reach → a spurious
/// 200-iteration hard error. The fix stops on the affine-invariant, sample-size-
/// invariant Newton decrement instead, so this converges. The survival location-scale
/// fit solves to `SURVIVAL_LAML_STATIONARITY_RELATIVE_TOL = 1e-8`, which is therefore
/// exactly the pre-fix failing regime and gives the test teeth.
#[test]
fn reduced_parametric_aft_converges_and_recovers_lognormal_mle_2112() {
    // Deterministic lognormal sample: log t ~ N(mu0, sigma0), all fully observed.
    let n = 2000usize;
    let mu0 = 1.5_f64;
    let sigma0 = 0.7_f64;
    // Seeded LCG + Box-Muller: reproducible standard normals, no RNG dependency.
    let mut state: u64 = 0x2112_2112_dead_beef;
    let mut next_u = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut z = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_u().max(1e-300);
        let u2 = next_u();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (std::f64::consts::TAU * u2).cos());
        z.push(r * (std::f64::consts::TAU * u2).sin());
    }
    z.truncate(n);
    let logt: Vec<f64> = z.iter().map(|zi| mu0 + sigma0 * zi).collect();
    let age_exit = Array1::from_iter(logt.iter().map(|l| l.exp()));
    // Left-truncation entry ~1e-4·t is far below the mass (~9σ), so S(entry) ≈ 1
    // and the fit is the standard uncensored lognormal MLE.
    let age_entry = Array1::from_iter(age_exit.iter().map(|t| t * 1e-4));
    let event_target = Array1::ones(n);
    let weights = Array1::ones(n);

    // Closed-form lognormal MLE (all events, negligible truncation).
    let mean_logt = logt.iter().sum::<f64>() / n as f64;
    let var_logt = logt.iter().map(|l| (l - mean_logt).powi(2)).sum::<f64>() / n as f64;
    let sd_logt = var_logt.sqrt();

    // Time block: 2 columns with a diag(0, 1) penalty whose 1-D null space is the
    // leading (log-t) column. Under constant scale the block reduces to the pinned
    // unit-log-t warp with zero free columns, so the fit routes through the reduced
    // parametric-AFT direct MLE (the code path fixed for gam#2112).
    let mut design_entry = Array2::<f64>::zeros((n, 2));
    let mut design_exit = Array2::<f64>::zeros((n, 2));
    let mut design_deriv = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        design_entry[[i, 0]] = age_entry[i].ln();
        design_exit[[i, 0]] = age_exit[i].ln();
        design_deriv[[i, 0]] = 1.0 / age_exit[i];
    }
    let penalty = array![[0.0, 0.0], [0.0, 1.0]];

    let spec = SurvivalLocationScaleSpec {
        age_entry,
        age_exit,
        event_target,
        weights,
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_deriv),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::from_elem(
                n,
                DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            ),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![penalty],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        persistent_warm_start_store: None,
        cache_mirror_sessions: Vec::new(),
    };

    // The fit must take the reduced parametric-AFT route (the fixed code path);
    // otherwise this would not exercise gam#2112 at all.
    let prepared = prepare_survival_location_scale_model(&spec).expect("prepare succeeds");
    assert!(
        prepared.is_reduced_parametric_aft(),
        "test must exercise the reduced parametric-AFT direct MLE (the gam#2112 code path)"
    );

    // The crux of gam#2112: on benign fully-observed lognormal data at n=2000 the
    // fit must CONVERGE (pre-fix it hard-errored after 200 Newton iterations).
    let (fit, _geo) = fit_survival_location_scale_with_geometry(spec)
        .expect("reduced parametric-AFT MLE must converge on benign lognormal data (gam#2112)");
    // The fit existing at all is the convergence proof: the sealed
    // `FitConvergenceEvidence` constructor refuses non-converged assembly.

    // Closed-form MLE recovery: μ̂ = mean(log t), σ̂ = sd(log t).
    let mu_hat = fit.beta_threshold()[0];
    let sigma_hat = fit.beta_log_sigma()[0].exp();
    assert!(
        (mu_hat - mean_logt).abs() < 1e-4,
        "location MLE μ̂={mu_hat} must match closed-form mean(log t)={mean_logt}"
    );
    assert!(
        (sigma_hat - sd_logt).abs() < 1e-4,
        "scale MLE σ̂={sigma_hat} must match closed-form sd(log t)={sd_logt}"
    );
}

// ---------------------------------------------------------------------------
// gam#2112: the reduced parametric-AFT (constant-scale location-scale survival)
// direct Newton MLE must certify stationarity with a SCALE-INVARIANT criterion
// (the Newton decrement ½·gᵀH⁻¹g), not an absolute tolerance on the SUMMED
// log-likelihood gradient — whose attainable floor grows like n·ε, so an
// absolute tolerance spuriously fails to converge on benign data as n (or the
// total weight) grows. These tests drive the real reduced-AFT path
// (`prepare_survival_location_scale_model` → `is_reduced_parametric_aft` →
// `fit_reduced_parametric_aft` → `fit_parametric_aft_direct_mle`).
// ---------------------------------------------------------------------------

/// Deterministic lognormal AFT sample: `log t ~ Normal(mu, sigma)`, fully
/// observed. Returns `(age_exit, event, log_t)`.
fn reduced_aft_lognormal_sample(
    n: usize,
    mu: f64,
    sigma: f64,
    seed: u64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut state = seed;
    let next_u01 = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut log_t = Array1::<f64>::zeros(n);
    for i in 0..n {
        let u1 = next_u01(&mut state).max(1e-12);
        let u2 = next_u01(&mut state);
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        log_t[i] = mu + sigma * z;
    }
    let age_exit = log_t.mapv(f64::exp);
    let event = Array1::<f64>::ones(n);
    (age_exit, event, log_t)
}

/// Build a constant-scale lognormal-AFT `SurvivalLocationScaleSpec` from event
/// times: a monotone I-spline-like time basis over `log t` (rank-1 penalty null
/// space → the reduced log-t warp), an intercept location and constant log-σ
/// (both unpenalized). `weights` scales every row's likelihood contribution.
fn reduced_aft_lognormal_spec(
    age_exit: &Array1<f64>,
    event: &Array1<f64>,
    weight: f64,
) -> SurvivalLocationScaleSpec {
    let n = age_exit.len();
    let p_time = 6usize;
    let age_entry = Array1::from_elem(n, 1e-9_f64);
    let log_t: Vec<f64> = age_exit.iter().map(|&t| t.max(1e-12).ln()).collect();
    let lo = log_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = log_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-6);
    // Monotone I-spline-like value / derivative rows over log t.
    let mut design_exit = Array2::<f64>::zeros((n, p_time));
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let lt = log_t[i];
        for j in 0..p_time {
            let center = lo + span * (j as f64 + 0.5) / (p_time as f64);
            let x = 6.0 / span * (lt - center);
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            design_exit[[i, j]] = sigmoid;
            // d/dt = d/d(log t) * d(log t)/dt = sigmoid'(x)*(6/span) * (1/t).
            let dsig = sigmoid * (1.0 - sigmoid) * (6.0 / span);
            design_derivative_exit[[i, j]] = dsig / age_exit[i].max(1e-12);
        }
    }
    // I-spline is 0 below the knot range: entry near t=0 contributes nothing.
    let design_entry = Array2::<f64>::zeros((n, p_time));
    // 1st-difference penalty: null space = the constant vector (rank 1), the
    // affine log-t baseline the reduce collapses onto.
    let mut penalty = Array2::<f64>::zeros((p_time, p_time));
    for j in 0..(p_time - 1) {
        penalty[[j, j]] += 1.0;
        penalty[[j, j + 1]] -= 1.0;
        penalty[[j + 1, j]] -= 1.0;
        penalty[[j + 1, j + 1]] += 1.0;
    }
    let derivative_offset_exit =
        Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
    SurvivalLocationScaleSpec {
        age_entry,
        age_exit: age_exit.clone(),
        event_target: event.clone(),
        weights: Array1::from_elem(n, weight),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit,
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![penalty],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(array![0.0]),
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        persistent_warm_start_store: None,
        cache_mirror_sessions: Vec::new(),
    }
}

/// Closed-form uncensored lognormal MLE: `mu_hat = mean(log t)`,
/// `sigma_hat = population sd(log t)`.
fn lognormal_closed_form_mle(log_t: &Array1<f64>) -> (f64, f64) {
    let mu = log_t.mean().unwrap();
    let sigma = (log_t.mapv(|v| (v - mu).powi(2)).sum() / (log_t.len() as f64)).sqrt();
    (mu, sigma)
}

/// gam#2112: the reduced parametric-AFT direct MLE must CONVERGE and recover the
/// closed-form lognormal MLE at sample sizes where the removed absolute
/// summed-gradient tolerance failed (n ≳ 1000). This drives the real reduced
/// path and checks the fitted location intercept / constant log-σ against the
/// closed form `mu = mean(log t)`, `log σ = log sd(log t)`.
#[test]
fn reduced_parametric_aft_converges_and_recovers_mle_at_scale() {
    for &n in &[2000usize, 5000, 10000] {
        let (age_exit, event, log_t) = reduced_aft_lognormal_sample(n, 1.4, 0.5, 0);
        let spec = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);

        // Confirm we are exercising the fixed code path (the direct parametric
        // AFT MLE), not the coupled REML fallback.
        let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
        assert!(
            prepared.is_reduced_parametric_aft(),
            "n={n}: expected the reduced parametric-AFT regime (the fit_parametric_aft_direct_mle path)"
        );
        assert!(
            prepared.family.location_log_time.is_some(),
            "n={n}: the log-t AFT baseline must be encoded for a lognormal AFT"
        );

        // The core regression: this used to hard-error with
        // "direct parametric-AFT MLE: failed to converge after 200 Newton
        // iterations" for n ≳ 1000. It must now converge.
        let (fit, _) = fit_survival_location_scale_with_geometry(spec)
            .unwrap_or_else(|e| panic!("n={n}: reduced parametric-AFT MLE must converge: {e}"));

        let (mu_hat, sigma_hat) = lognormal_closed_form_mle(&log_t);
        let loc = fit.beta_threshold()[0];
        let log_sigma = fit.beta_log_sigma()[0];
        // A CONVERGED Newton MLE lands on the closed-form optimum. The recovered
        // values match to ~1e-6 in practice; 1e-3 leaves ample slack for the
        // Newton stop while still catching a mis-converged / non-stationary fit.
        assert!(
            (loc - mu_hat).abs() < 1e-3,
            "n={n}: location {loc:.6} != closed-form mu {mu_hat:.6}"
        );
        assert!(
            (log_sigma - sigma_hat.ln()).abs() < 1e-3,
            "n={n}: log-sigma {log_sigma:.6} != closed-form {:.6}",
            sigma_hat.ln()
        );
    }
}

/// gam#2112 (the mechanism, from a second angle): the stopping criterion must be
/// invariant to the TOTAL WEIGHT, exactly as it must be invariant to `n`. The
/// per-row weights multiply every likelihood contribution, so a uniform weight
/// `W` scales the summed log-likelihood — and hence the summed gradient `g = ∇ℓ`
/// and Hessian `H = −∇²ℓ` — by `W`, while leaving the MLE `θ̂ = argmax ℓ`
/// unchanged. The removed absolute test on `‖g‖∞` therefore fails to converge
/// for large `W` (the summed gradient's floor scales with `W`), whereas the
/// Newton-decrement test `gᵀH⁻¹g` cancels the `W` and certifies stationarity at
/// the SAME `θ̂`. Fitting the identical sample at `W = 1` and a large `W` must
/// converge to the same coefficients.
#[test]
fn reduced_parametric_aft_stopping_criterion_is_weight_scale_invariant() {
    let (age_exit, event, _log_t) = reduced_aft_lognormal_sample(1500, 1.2, 0.6, 7);

    let fit_at_weight = |w: f64| -> (f64, f64) {
        let spec = reduced_aft_lognormal_spec(&age_exit, &event, w);
        let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
        assert!(
            prepared.is_reduced_parametric_aft(),
            "expected reduced parametric-AFT regime"
        );
        let (fit, _) = fit_survival_location_scale_with_geometry(spec).unwrap_or_else(|e| {
            panic!("reduced parametric-AFT MLE must converge at total-weight scale w={w}: {e}")
        });
        (fit.beta_threshold()[0], fit.beta_log_sigma()[0])
    };

    let (loc1, ls1) = fit_at_weight(1.0);
    // W = 500 makes the summed gradient 500× larger — well past the regime where
    // the old absolute 1e-7 gradient tolerance could ever be met — yet the MLE
    // is identical, so a scale-invariant criterion converges to the same point.
    let (loc500, ls500) = fit_at_weight(500.0);

    assert!(
        (loc1 - loc500).abs() < 1e-6,
        "location must be weight-scale invariant: w=1 -> {loc1:.9}, w=500 -> {loc500:.9}"
    );
    assert!(
        (ls1 - ls500).abs() < 1e-6,
        "log-sigma must be weight-scale invariant: w=1 -> {ls1:.9}, w=500 -> {ls500:.9}"
    );
}

// ---------------------------------------------------------------------------
// #2446: the exact response-moment rule integrates the joint Gaussian
// `N(E_π[β], Σ_π)` and nothing else. The `β_w ≥ 0` cone is corrected for ONCE,
// upstream; `cone_clipped_coordinate_displacement` was a third application of
// it and is gone, so its three unit tests went with it (the code under test
// left the call graph — retaining them would require retaining dead code
// `-D warnings` forbids).
// ---------------------------------------------------------------------------

/// gam#2695 — FD ORACLE on the LINK-WIGGLE joint block gradient.
///
/// `survival_ls_block_gradient_matches_single_sourced_tower_932` pins the block
/// gradient against the §13 tower for the THREE-block family
/// (`x_link_wiggle: None`). The wiggle path has no such witness: the wiggle row
/// kernel exposes `hessian_dense`, `directional_derivative_dense` and
/// `second_directional_derivative_dense`, but NO gradient, so nothing compares
/// `evaluate_log_likelihood_and_block_gradients` against a second source once a
/// link warp is installed — and that is the one arm where the design is
/// `beta`-dependent (`q = q0 + Σ_j βw_j·B_j(q0)`, `m1 = 1 + Σ_j βw_j·B'_j(q0)`),
/// i.e. the one arm where a dropped chain-rule term is possible.
///
/// This differentiates the routine against ITSELF: the same call returns `ℓ` and
/// `∇ℓ`, so a central difference of the returned value must reproduce the
/// returned gradient at every coefficient of every block. The step is
/// `ε^(1/3)·(1+|x|)` — the central-difference optimum, derived from `f64::EPSILON`
/// and the coefficient scale, not chosen — and the bound is
/// `64·ε^(2/3)·(1+|analytic|)`, the matching truncation+roundoff floor.
#[test]
fn survival_ls_link_wiggle_block_gradient_matches_finite_difference_2695() {
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();
    let q0_exit = Array1::from_shape_fn(n, |i| {
        -primaries[i][3] * (-primaries[i][6]).exp()
    });
    let knots = Array1::from_vec(vec![
        -3.0, -3.0, -3.0, -3.0, -1.5, 0.0, 1.5, 3.0, 3.0, 3.0, 3.0,
    ]);
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("link wiggle design");
    let pw = xwiggle.ncols();
    assert!(pw >= 2, "the fixture must install a real wiggle block, got pw={pw}");

    let inverse_link = residual_distribution_inverse_link(ResidualDistribution::Gaussian);
    // BOTH channel layouts. `evaluate_log_likelihood_and_block_gradients` takes a
    // different branch for the threshold and log-sigma blocks depending on
    // whether the family carries separate entry/derivative designs: the `Some`
    // arm contracts the three channels independently, the `None` arm uses the
    // #2342 regrouped `S1·dq_exit + d1_q0·(dq_entry − dq_exit)` sum. The witness
    // fit (an intercept threshold and an intercept log-sigma) takes the `None`
    // arm, so a fixture that only exercises `Some` would leave the live branch
    // unwitnessed.
    for time_varying_channels in [true, false] {
    let mut family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
    if !time_varying_channels {
        family.x_threshold_entry = None;
        family.x_threshold_deriv = None;
        family.x_log_sigma_entry = None;
        family.x_log_sigma_deriv = None;
    }
    family.x_link_wiggle = Some(DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
    ));
    family.wiggle_knots = Some(knots.clone());
    family.wiggle_degree = Some(degree);

    // The oracle family's three additive designs are single columns holding the
    // primary channels, so `eta_channel = channel · beta` exactly and the states
    // can be rebuilt from the coefficients alone.
    // Small positive amplitudes. The warp multiplies the event-time Jacobian by
    // `m1 = 1 + sum_j betaw_j * B'_j(q0)`, and the family REFUSES any state whose
    // `d_eta/dt` falls to zero (structural monotonicity, floor 1e-8); at
    // `0.05 + 0.03*j` this fixture's row 1 lands at `d_eta/dt = -1.243e-2` and the
    // gradient call errs before any derivative is compared. These amplitudes keep
    // `m1` within a few percent of 1 while leaving every wiggle column live.
    let beta_w0 = Array1::from_shape_fn(pw, |j| 0.002 + 0.001 * (j as f64));
    let build = |betas: [f64; 3], beta_w: &Array1<f64>| -> Vec<ParameterBlockState> {
        let stacked = |first: usize, second: usize, deriv: usize, scale: f64| {
            let mut eta = Array1::<f64>::zeros(3 * n);
            for i in 0..n {
                eta[i] = primaries[i][first] * scale;
                eta[n + i] = primaries[i][second] * scale;
                eta[2 * n + i] = primaries[i][deriv] * scale;
            }
            eta
        };
        let flat = |channel: usize, scale: f64| {
            Array1::from_shape_fn(n, |i| primaries[i][channel] * scale)
        };
        vec![
            ParameterBlockState {
                beta: array![betas[0]],
                eta: stacked(0, 1, 2, betas[0]),
            },
            ParameterBlockState {
                beta: array![betas[1]],
                eta: if time_varying_channels {
                    stacked(3, 4, 5, betas[1])
                } else {
                    flat(3, betas[1])
                },
            },
            ParameterBlockState {
                beta: array![betas[2]],
                eta: if time_varying_channels {
                    stacked(6, 7, 8, betas[2])
                } else {
                    flat(6, betas[2])
                },
            },
            ParameterBlockState {
                beta: beta_w.clone(),
                eta: xwiggle.dot(beta_w),
            },
        ]
    };

    let betas0 = [1.0_f64, 1.0, 1.0];
    let states = build(betas0, &beta_w0);
    let (ll0, block_gradients) = family
        .evaluate_log_likelihood_and_block_gradients(&states)
        .expect("wiggle block gradients");
    assert_eq!(
        block_gradients.len(),
        4,
        "the wiggle family must report four blocks"
    );
    assert!(ll0.is_finite(), "fixture log-likelihood must be finite");
    let analytic: Vec<f64> = block_gradients
        .iter()
        .flat_map(|block| block.iter().copied())
        .collect();
    assert_eq!(analytic.len(), 3 + pw);

    // DISCRIMINATOR (gam#2695). The trust-region ratio's numerator and
    // denominator do not come from the same evaluator: `actual_reduction` is
    // built from `log_likelihood_only` (the backtracking fast path) while the
    // Newton RHS is built from `evaluate`'s block gradients. If those two
    // evaluate different functions of beta, no gradient fix repairs the ratio.
    // Pin them at the same state before differentiating either.
    {
        use crate::custom_family::CustomFamily;
        let fast = family
            .log_likelihood_only(&states)
            .expect("fast-path log-likelihood");
        assert!(
            (fast - ll0).abs() <= 1e-9 * (1.0 + ll0.abs()),
            "the backtracking fast path `log_likelihood_only` reports {fast:.9e} where \
             `evaluate_log_likelihood_and_block_gradients` reports {ll0:.9e} at the SAME \
             state (arm time_varying_channels={time_varying_channels}); the trust-region ratio \
             compares two different objectives"
        );
    }

    let ll_at = |betas: [f64; 3], beta_w: &Array1<f64>| -> f64 {
        family
            .evaluate_log_likelihood_and_block_gradients(&build(betas, beta_w))
            .expect("perturbed log-likelihood")
            .0
    };

    let cbrt_eps = f64::EPSILON.cbrt();
    let bound_scale = 64.0 * cbrt_eps * cbrt_eps;
    let mut worst = 0.0_f64;
    let mut worst_index = 0usize;
    for index in 0..analytic.len() {
        let base = if index < 3 { betas0[index] } else { beta_w0[index - 3] };
        let h = cbrt_eps * (1.0 + base.abs());
        let shift = |delta: f64| -> f64 {
            let mut betas = betas0;
            let mut beta_w = beta_w0.clone();
            if index < 3 {
                betas[index] = base + delta;
            } else {
                beta_w[index - 3] = base + delta;
            }
            ll_at(betas, &beta_w)
        };
        let fd = (shift(h) - shift(-h)) / (2.0 * h);
        let tol = bound_scale * (1.0 + analytic[index].abs());
        let drift = (fd - analytic[index]).abs();
        if drift > worst {
            worst = drift;
            worst_index = index;
        }
        assert!(
            drift <= tol,
            "arm time_varying_channels={time_varying_channels}, coefficient {index} \
             (block {}): analytic ∂ℓ/∂β = {:.9e} but the central \
             difference of the SAME call's ℓ is {:.9e} (drift {:.3e} > {:.3e}); the joint \
             gradient does not differentiate the log-likelihood it returns",
            if index < 3 { index } else { 3 },
            analytic[index],
            fd,
            drift,
            tol
        );
    }
    // Non-vacuity: the gradient must not be the zero vector at this fixture, or
    // the loop above would pass on an empty claim.
    assert!(
        analytic.iter().any(|value| value.abs() > 1e-6),
        "arm time_varying_channels={time_varying_channels}: fixture must produce a non-trivial \
         gradient (worst drift {worst:.3e} at {worst_index})"
    );
    }
}

/// gam#2695 — the same oracle at a warp that is actually ON.
///
/// `survival_ls_link_wiggle_block_gradient_matches_finite_difference_2695` runs
/// at `betaw_j = 0.002 + 0.001·j`, i.e. `m1 = 1 + Σ_j βw_j·B'_j(q0)` within a
/// few percent of 1, because at larger amplitudes THAT fixture's row 1 drives
/// `dη/dt` negative and the family refuses before any derivative is compared.
/// A warp that is off is exactly the state in which a dropped warp chain-rule
/// term is invisible, and that is why the oracle is green while the production
/// witness is not: measured on
/// `survival_location_scale_saved_fit_preserves_linkwiggle_metadata`, the
/// coordinates whose analytic `∂ℓ/∂β` disagrees with a central difference of
/// the solver's own objective are exactly the ones evaluated at `βw = O(1)`,
/// and the ones at `βw ≈ 1e-6` agree to nine digits.
///
/// The refusal is avoidable rather than intrinsic. `qdot = m1·r` with
/// `r = inv_sigma·(η_t·η_ls' − η_t')`, so on the layout the production witness
/// actually uses — a time-INVARIANT threshold and log-sigma, i.e.
/// `x_threshold_deriv = x_log_sigma_deriv = None` — `r ≡ 0`, the warp cannot
/// touch the event Jacobian at all, and `βw` is free to be O(1). This test
/// therefore reproduces the witness's own channel layout AND its warp
/// magnitude, which is the pair the existing oracle cannot hold at once.
#[test]
fn survival_ls_link_wiggle_block_gradient_matches_finite_difference_at_a_real_warp_2695() {
    survival_ls_link_wiggle_real_warp_oracle_2695(3.0);
}

/// gam#2695 — the same real warp, with the knot domain DELIBERATELY narrower
/// than the range of `q0` the rows sit at.
///
/// The wiggle knots are frozen at fit setup from the seed `q0`, and `q0`
/// depends on the threshold and log-sigma coefficients
/// (`q0 = −η_t·e^{−η_ls}`), which move by orders of magnitude during the outer
/// search. So a production fit spends most of its iterates with rows OUTSIDE
/// the warp's own knot domain, and that is where `create_ispline_dense`'s
/// saturating convention and `create_ispline_derivative_dense`'s
/// linear-extension convention disagree. Every arm with the rows inside the
/// span — including the wide-span arm above — is blind to it.
#[test]
fn survival_ls_link_wiggle_block_gradient_matches_finite_difference_outside_the_knot_domain_2695() {
    survival_ls_link_wiggle_real_warp_oracle_2695(0.5);
}

fn survival_ls_link_wiggle_real_warp_oracle_2695(knot_half_span: f64) {
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();
    let q0_exit = Array1::from_shape_fn(n, |i| -primaries[i][3] * (-primaries[i][6]).exp());
    let half = knot_half_span;
    let knots = Array1::from_vec(vec![
        -half,
        -half,
        -half,
        -half,
        -0.5 * half,
        0.0,
        0.5 * half,
        half,
        half,
        half,
        half,
    ]);
    // State which regime this arm is in, so neither can silently become the
    // other: `[-half, half]` is the modelling interval the I-spline is
    // saturating outside of.
    let outside = q0_exit.iter().filter(|q| q.abs() > half).count();
    if half < 1.0 {
        assert!(
            outside > 0 && outside < n,
            "the narrow-span arm must STRADDLE the knot domain [{:.3}, {:.3}] — rows outside \
             it exercise the saturating branch and rows inside it keep the warp multiplier \
             materially away from 1; got {outside} of {n} outside, q0 = {q0_exit:?}",
            -half,
            half
        );
    } else {
        assert_eq!(
            outside, 0,
            "the wide-span arm must keep every row inside the knot domain"
        );
    }
    let degree = 3usize;
    let xwiggle =
        survival_wiggle_basis_with_options(q0_exit.view(), &knots, degree, BasisOptions::value())
            .expect("link wiggle design");
    let pw = xwiggle.ncols();
    assert!(pw >= 2, "the fixture must install a real wiggle block, got pw={pw}");

    let inverse_link = residual_distribution_inverse_link(ResidualDistribution::Gaussian);
    let mut family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
    // The witness's layout: time-invariant threshold and log-sigma. This is the
    // `None` arm of `evaluate_log_likelihood_and_block_gradients`, and it is
    // what makes `r ≡ 0` so a real warp is admissible.
    family.x_threshold_entry = None;
    family.x_threshold_deriv = None;
    family.x_log_sigma_entry = None;
    family.x_log_sigma_deriv = None;
    family.x_link_wiggle = Some(DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
    ));
    family.wiggle_knots = Some(knots.clone());
    family.wiggle_degree = Some(degree);

    let beta_w0 = Array1::from_shape_fn(pw, |j| 0.30 + 0.10 * (j as f64));
    let build = |betas: [f64; 3], beta_w: &Array1<f64>| -> Vec<ParameterBlockState> {
        let stacked = |first: usize, second: usize, deriv: usize, scale: f64| {
            let mut eta = Array1::<f64>::zeros(3 * n);
            for i in 0..n {
                eta[i] = primaries[i][first] * scale;
                eta[n + i] = primaries[i][second] * scale;
                eta[2 * n + i] = primaries[i][deriv] * scale;
            }
            eta
        };
        let flat =
            |channel: usize, scale: f64| Array1::from_shape_fn(n, |i| primaries[i][channel] * scale);
        vec![
            ParameterBlockState {
                beta: array![betas[0]],
                eta: stacked(0, 1, 2, betas[0]),
            },
            ParameterBlockState {
                beta: array![betas[1]],
                eta: flat(3, betas[1]),
            },
            ParameterBlockState {
                beta: array![betas[2]],
                eta: flat(6, betas[2]),
            },
            ParameterBlockState {
                beta: beta_w.clone(),
                eta: xwiggle.dot(beta_w),
            },
        ]
    };

    let betas0 = [1.0_f64, 1.0, 1.0];
    let states = build(betas0, &beta_w0);
    let (ll0, block_gradients) = family
        .evaluate_log_likelihood_and_block_gradients(&states)
        .expect("wiggle block gradients at a real warp");
    assert!(ll0.is_finite(), "fixture log-likelihood must be finite");

    // Non-vacuity: the warp must actually be ON. `m1 = 1 + Σ_j βw_j·B'_j(q0)`
    // is the factor every threshold / log-sigma channel is scaled by, so if it
    // is 1 to a few percent this test measures the same thing the existing
    // oracle already does.
    let warp = family
        .wiggle_geometry(q0_exit.view(), beta_w0.view())
        .expect("wiggle geometry")
        .expect("the fixture installs knots and a degree");
    let max_warp = warp
        .dq_dq0
        .iter()
        .map(|value| (value - 1.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_warp > 0.2,
        "the warp must be materially on for this test to differ from the small-amplitude \
         oracle; max |m1 - 1| = {max_warp:.3e}"
    );

    let analytic: Vec<f64> = block_gradients
        .iter()
        .flat_map(|block| block.iter().copied())
        .collect();
    assert_eq!(analytic.len(), 3 + pw);

    let ll_at = |betas: [f64; 3], beta_w: &Array1<f64>| -> f64 {
        family
            .evaluate_log_likelihood_and_block_gradients(&build(betas, beta_w))
            .expect("perturbed log-likelihood")
            .0
    };

    let cbrt_eps = f64::EPSILON.cbrt();
    let bound_scale = 64.0 * cbrt_eps * cbrt_eps;
    for index in 0..analytic.len() {
        let base = if index < 3 {
            betas0[index]
        } else {
            beta_w0[index - 3]
        };
        let h = cbrt_eps * (1.0 + base.abs());
        let shift = |delta: f64| -> f64 {
            let mut betas = betas0;
            let mut beta_w = beta_w0.clone();
            if index < 3 {
                betas[index] = base + delta;
            } else {
                beta_w[index - 3] = base + delta;
            }
            ll_at(betas, &beta_w)
        };
        let fd = (shift(h) - shift(-h)) / (2.0 * h);
        let tol = bound_scale * (1.0 + analytic[index].abs());
        assert!(
            (fd - analytic[index]).abs() <= tol,
            "coefficient {index} (block {}) at a REAL warp (max |m1-1| = {max_warp:.3e}): \
             analytic ∂ℓ/∂β = {:.9e} but the central difference of the SAME call's ℓ is \
             {fd:.9e} (drift {:.3e} > {tol:.3e})",
            if index < 3 { index } else { 3 },
            analytic[index],
            (fd - analytic[index]).abs(),
        );
    }
}

/// gam#2695 — the OBSERVED INFORMATION must be a continuous function of β as a
/// row's `q₀` crosses a knot of the link warp.
///
/// This is the contract taken from the side the inner solve actually refuses
/// on. `Φ = ½ Σ g(λ(Z_JᵀHZ_J))` is part of the objective the trust region
/// accepts on, so a step in `H` is a step in the OBJECTIVE and
/// `actual/predicted` cannot approach `1` at any step size — the `[0,0,2,0]`
/// signature this issue is named for.
///
/// Three things the fixture has to get right, each of which silently defeats
/// the measurement when it does not:
///
/// * **The crossing row must be an EVENT row.** `m1 = 1 + Σ βw_j I′_j(q1)`
///   reaches the likelihood only through `log g`, which is added when
///   `w·d ≠ 0`. A censored crossing row measures the warp's VALUE channel alone
///   and reports "continuous" for the wrong reason.
/// * **The warp must be ON.** The `I″` channel that survives at `βw = 0` is
///   real, but the `I‴` one is `βw`-weighted, and only the second is what the
///   knot vector fixes at degree 3.
/// * **The knots must be the ones production builds.** The negative control
///   below is the same fixture on a CLAMPED vector, where the measurement fails.
///
/// The assertion is scale-free: the gap across the knot must FALL with the
/// step. A step leaves it flat.
#[test]
fn joint_hessian_is_continuous_as_q0_crosses_a_link_warp_knot_2695() {
    let (warp_coarse, warp_fine) = link_warp_hessian_gap_across_a_knot_2695(3, true, 3.0e-2);
    assert!(
        warp_coarse > 0.0,
        "the Hessian must respond to β at all for this measurement to mean anything"
    );
    // A step leaves the gap flat in `h`; a continuous `H` divides it by the same
    // factor the step was divided by. Allow half an order against the exact
    // 100× so the pin reads the ORDER, not a rate.
    assert!(
        warp_fine <= warp_coarse / 50.0,
        "the joint Hessian does not close across a link-warp knot: the gap is \
         {warp_coarse:.6e} at h=1e-3 and {warp_fine:.6e} at h=1e-5, a ratio of {:.3e} \
         against the 100x a continuous H must give",
        warp_coarse / warp_fine.max(f64::MIN_POSITIVE),
    );
}

/// Non-vacuity for the pin above, and the measurement that says the KNOT VECTOR
/// is what closed it: the same fixture, same degree, same warp amplitude, on a
/// clamped knot vector — where the boundary knot's multiplicity `degree + 1`
/// puts a step into `I′` and, through the tail-free evaluator, into `H`.
#[test]
fn the_clamped_knot_vector_does_not_close_the_hessian_2695() {
    let (coarse, fine) = link_warp_hessian_gap_across_a_knot_2695(3, false, 3.0e-2);
    assert!(
        fine > coarse / 10.0,
        "the clamped arm must STEP for the warp-vector pin to be measuring the knot \
         vector rather than the fixture: gap {coarse:.6e} at h=1e-3 and {fine:.6e} at \
         h=1e-5, a ratio of {:.3e}",
        coarse / fine.max(f64::MIN_POSITIVE),
    );
}

/// A degree-2 composed warp is not admissible at all, and no knot vector fixes
/// it: `H` reads `I″` with NO `βw` factor, and a degree-2 I-spline is `C¹`, so
/// `I″` is piecewise constant and steps at every knot. Pinned on the WARP
/// vector, so it cannot be read as a statement about the clamped one.
#[test]
fn a_degree_two_composed_warp_steps_even_on_the_warp_knot_vector_2695() {
    for amplitude in [1.0e-6_f64, 3.0e-2] {
        let (coarse, fine) = link_warp_hessian_gap_across_a_knot_2695(2, true, amplitude);
        assert!(
            fine > coarse / 10.0,
            "degree 2 at βw={amplitude:.1e} must STEP — if it closes, the `I″` channel \
             with no `βw` factor has been removed and the degree floor can go: gap \
             {coarse:.6e} at h=1e-3 and {fine:.6e} at h=1e-5",
        );
    }
}

/// gam#2695 — the term of the OBJECTIVE, not the matrix behind it, must close
/// across a knot at the degree a composed warp is actually built at.
///
/// `joint_hessian_is_continuous_as_q0_crosses_a_link_warp_knot_2695` pins `H`.
/// This pins `Φ = ½ Σ g(λ(Z_JᵀHZ_J))`, which is what `trial_penalty` subtracts
/// and therefore what the trust region compares two points on. The two are not
/// the same assertion: `Φ` is a log-determinant of a FLOORED spectrum, so a step
/// in `H` could in principle land where `g` is flat, and a `Φ` that closes is
/// the statement the accept test needs.
///
/// The degree is not written here. It is
/// [`crate::wiggle::composed_warp_minimum_degree`], so if that floor is ever
/// lowered this pin measures the lowered value and goes red rather than
/// continuing to certify a degree production no longer builds.
#[test]
fn the_objective_jeffreys_term_closes_across_a_link_warp_knot_at_the_built_degree_2695() {
    let degree = crate::wiggle::composed_warp_minimum_degree();
    for amplitude in [1.0e-6_f64, 3.0e-2] {
        let (coarse, fine) = link_warp_knot_crossing_gap_2695(
            degree,
            true,
            amplitude,
            LinkWarpKnotReading::ObjectiveJeffreysTerm,
        );
        assert!(
            coarse > 0.0,
            "Φ must respond to β at all for this measurement to mean anything              (degree {degree}, βw={amplitude:.1e})"
        );
        assert!(
            fine <= coarse / 50.0,
            "the inner objective's Jeffreys term does not close across a link-warp knot at              degree {degree}, βw={amplitude:.1e}: gap {coarse:.6e} at h=1e-3 and {fine:.6e}              at h=1e-5, a ratio of {:.3e} against the 100x a continuous Φ must give",
            coarse / fine.max(f64::MIN_POSITIVE),
        );
    }
}

/// gam#2695 — the optimizer consumes `∇Φ`, not only `Φ`, so the realized
/// composed-warp degree must make that analytic gradient continuous when an
/// EVENT row's model-dependent index crosses an interior knot.
///
/// This pin reads the production Jeffreys gradient across an event-row
/// crossing. Measured on this fixture (`knot_ladder_2695`), that gradient steps
/// at degree 3 and is continuous from degree 4 on, which is what sets the floor.
#[test]
fn the_objective_jeffreys_gradient_closes_across_a_link_warp_knot_at_the_built_degree_2695() {
    let degree = crate::wiggle::composed_warp_minimum_degree();
    for amplitude in [1.0e-6_f64, 3.0e-2] {
        let (coarse, fine) = link_warp_knot_crossing_gap_2695(
            degree,
            true,
            amplitude,
            LinkWarpKnotReading::ObjectiveJeffreysGradient,
        );
        assert!(
            coarse > 0.0,
            "∇Φ must respond to the event-row knot crossing for this measurement to \
             mean anything (degree {degree}, βw={amplitude:.1e})"
        );
        assert!(
            fine <= coarse / 50.0,
            "the inner objective's Jeffreys gradient does not close across a link-warp \
             knot at degree {degree}, βw={amplitude:.1e}: gap {coarse:.6e} at h=1e-3 \
             and {fine:.6e} at h=1e-5, a ratio of {:.3e} against the 100x a continuous \
             gradient must give",
            coarse / fine.max(f64::MIN_POSITIVE),
        );
    }
}

/// Non-vacuity for the C¹ floor: one degree below the realized production
/// degree, the objective value is continuous but its analytic gradient steps.
#[test]
fn a_degree_three_composed_warp_makes_the_objective_gradient_jump_2695() {
    let degree = crate::wiggle::composed_warp_minimum_degree() - 1;
    assert_eq!(degree, 3, "this arm must measure the C⁰-but-not-C¹ degree");
    for amplitude in [1.0e-6_f64, 3.0e-2] {
        let (coarse, fine) = link_warp_knot_crossing_gap_2695(
            degree,
            true,
            amplitude,
            LinkWarpKnotReading::ObjectiveJeffreysGradient,
        );
        assert!(
            coarse > 0.0 && fine > coarse / 10.0,
            "degree 3 must make ∇Φ jump across the event-row knot: gap \
             {coarse:.6e} at h=1e-3 and {fine:.6e} at h=1e-5"
        );
    }
}

/// Non-vacuity, and the measurement the floor exists for: at degree 2 the SAME
/// reading is a jump, and it is a jump of the size the shipped witness reported
/// (`2.976461e-1` there, `O(1e-1)` here — same mechanism, different state).
///
/// This is the test that would have caught the regression from the other side:
/// if the floor is removed, production builds degree 2 again and this arm is
/// what production is then minimising.
///
/// It does NOT bound the floor from below on its own, and that is worth stating
/// where the reader is: this fixture does not excite the `βw`-weighted `I‴`
/// channel above its own resolution, so it reports degree 3 as closing when the
/// shipped witness's `Φ` still jumps by `2.6e-7` there. The order is fixed on
/// the FIT — see `COMPOSED_WARP_REQUIRED_CONTINUOUS_BASIS_DERIVATIVE_ORDER` — and this arm
/// is the degree-2 lower bound only.
#[test]
fn a_degree_two_composed_warp_makes_the_objective_term_jump_2695() {
    for amplitude in [1.0e-6_f64, 3.0e-2] {
        let (coarse, fine) = link_warp_knot_crossing_gap_2695(
            2,
            true,
            amplitude,
            LinkWarpKnotReading::ObjectiveJeffreysTerm,
        );
        assert!(
            fine > coarse / 10.0,
            "degree 2 at βw={amplitude:.1e} must make Φ JUMP — if it closes, the `I″`              channel with no `βw` factor is gone from `H` and the composed-warp degree              floor can go with it: gap {coarse:.6e} at h=1e-3 and {fine:.6e} at h=1e-5",
        );
        assert!(
            fine > 1.0e-3,
            "the degree-2 jump must be a real one, not a rounding plateau: {fine:.6e}"
        );
    }
}

/// `(gap at h=1e-3, gap at h=1e-5)` of the joint Hessian across a knot that one
/// EVENT row's exit `q₀` crosses as `β_thr` moves through `1`.
///
/// `q₀ = −η_t·e^{−η_ls}` and both predictors are linear in their single
/// coefficient here, so at `β_ls = 1` every row's exit `q₀` is exactly linear in
/// `β_thr`: the knot is placed ON row 2's `q₀` at `β_thr = 1` and the crossing
/// is known rather than searched for.
fn link_warp_hessian_gap_across_a_knot_2695(
    degree: usize,
    warp_knots: bool,
    warp_amplitude: f64,
) -> (f64, f64) {
    link_warp_knot_crossing_gap_2695(
        degree,
        warp_knots,
        warp_amplitude,
        LinkWarpKnotReading::ObservedInformation,
    )
}

/// `(gap at h=1e-3, gap at h=1e-5)` of `read` across a knot that one EVENT row's
/// exit `q₀` crosses as `β_thr` moves through `1`.
fn link_warp_knot_crossing_gap_2695(
    degree: usize,
    warp_knots: bool,
    warp_amplitude: f64,
    read: LinkWarpKnotReading,
) -> (f64, f64) {
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();
    let q0_slope = |row: usize| -> f64 { -primaries[row][3] * (-primaries[row][6]).exp() };
    const B_STAR: f64 = 1.0;
    // event = 1.0, so the `log g` channel that carries `m1` is live on the row
    // that crosses.
    const CROSSING_ROW: usize = 2;
    assert_eq!(event[CROSSING_ROW], 1.0, "the crossing row must be an event row");

    let centre = q0_slope(CROSSING_ROW) * B_STAR;
    let knots = if warp_knots {
        // A simple-knot grid of unit spans whose interior lands on `centre`.
        let spans = 4usize;
        Array1::from_shape_fn(spans + 1 + 2 * degree, |i| {
            centre + (i as f64 - (degree + 2) as f64)
        })
    } else {
        let left = centre - 2.0;
        let right = centre + 2.0;
        let mut values = vec![left; degree + 1];
        values.push(centre - 1.0);
        values.push(centre);
        values.push(centre + 1.0);
        values.extend(std::iter::repeat_n(right, degree + 1));
        Array1::from_vec(values)
    };

    let seed_q0 = Array1::from_shape_fn(n, |i| q0_slope(i) * B_STAR);
    let xwiggle =
        survival_wiggle_basis_with_options(seed_q0.view(), &knots, degree, BasisOptions::value())
            .expect("link wiggle design");
    let pw = xwiggle.ncols();
    assert!(pw >= 2, "the fixture must install a real wiggle block, got pw={pw}");

    let inverse_link = residual_distribution_inverse_link(ResidualDistribution::Gaussian);
    let mut family = survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
    family.x_link_wiggle = Some(DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
    ));
    family.wiggle_knots = Some(knots.clone());
    family.wiggle_degree = Some(degree);
    let beta_w = Array1::from_shape_fn(pw, |j| warp_amplitude * (1.0 + 0.3 * (j as f64)));

    let states_at = |beta_thr: f64| -> Vec<ParameterBlockState> {
        let stacked = |first: usize, second: usize, deriv: usize, scale: f64| {
            let mut eta = Array1::<f64>::zeros(3 * n);
            for i in 0..n {
                eta[i] = primaries[i][first] * scale;
                eta[n + i] = primaries[i][second] * scale;
                eta[2 * n + i] = primaries[i][deriv] * scale;
            }
            eta
        };
        vec![
            ParameterBlockState { beta: array![1.0], eta: stacked(0, 1, 2, 1.0) },
            ParameterBlockState { beta: array![beta_thr], eta: stacked(3, 4, 5, beta_thr) },
            ParameterBlockState { beta: array![1.0], eta: stacked(6, 7, 8, 1.0) },
            ParameterBlockState { beta: beta_w.clone(), eta: xwiggle.dot(&beta_w) },
        ]
    };
    let hessian_at = |beta_thr: f64| -> Array2<f64> {
        let states = states_at(beta_thr);
        family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian across the knot")
            .expect("the family exposes an exact joint hessian")
    };
    let jeffreys_gradient_at =
        |beta_thr: f64, h_joint: &Array2<f64>| -> Array1<f64> {
            let states = states_at(beta_thr);
            let p = h_joint.nrows();
            let z = Array2::<f64>::eye(p);
            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
                h_joint.view(),
                z.view(),
                |direction: &Array1<f64>| {
                    family.exact_newton_joint_hessian_directional_derivative_rescaled(
                        &states, direction, 0.0,
                    )
                },
            )
            .expect("joint Jeffreys gradient across the knot")
            .1
        };

    let straddles = |h: f64| {
        // Confirm the step really straddles the knot before reading the gap.
        let q0_plus = q0_slope(CROSSING_ROW) * (B_STAR + h);
        let q0_minus = q0_slope(CROSSING_ROW) * (B_STAR - h);
        assert!(
            (q0_plus - centre).signum() != (q0_minus - centre).signum(),
            "step h={h:.1e} must straddle the knot {centre:.9e}; got q0 {q0_minus:.9e} \
             and {q0_plus:.9e}"
        );
    };
    let gap = |h: f64| -> f64 {
        straddles(h);
        let plus = hessian_at(B_STAR + h);
        let minus = hessian_at(B_STAR - h);
        match read {
            LinkWarpKnotReading::ObservedInformation => plus
                .iter()
                .zip(minus.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max),
            // The OBJECTIVE's own term, not a matrix norm of its input. `Φ` is
            // what `trial_penalty` subtracts and what the accept test therefore
            // compares, so this is the quantity whose continuity the trust
            // region needs — read on the FULL identifiable span (`Z_J = I`),
            // which is what the inner solve installs for this family.
            LinkWarpKnotReading::ObjectiveJeffreysTerm => {
                (jeffreys_value_2695(&plus) - jeffreys_value_2695(&minus)).abs()
            }
            LinkWarpKnotReading::ObjectiveJeffreysGradient => {
                let plus_gradient = jeffreys_gradient_at(B_STAR + h, &plus);
                let minus_gradient = jeffreys_gradient_at(B_STAR - h, &minus);
                plus_gradient
                    .iter()
                    .zip(minus_gradient.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            }
        }
    };
    (gap(1.0e-3), gap(1.0e-5))
}

/// Which quantity a knot-crossing measurement reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinkWarpKnotReading {
    /// The joint observed information `H` the Jeffreys term is built from.
    ObservedInformation,
    /// `Φ = ½ Σ g(λ(Z_JᵀHZ_J))`, the term of the inner objective itself.
    ObjectiveJeffreysTerm,
    /// `∇Φ`, the analytic gradient used by the Newton right-hand side and KKT gate.
    ObjectiveJeffreysGradient,
}

/// `Φ` at a joint Hessian, on the full identifiable span, through the SAME entry
/// point the inner solve's accept test uses.
fn jeffreys_value_2695(h_joint: &Array2<f64>) -> f64 {
    let p = h_joint.nrows();
    let z = Array2::<f64>::eye(p);
    gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
        h_joint.view(),
        z.view(),
        |_: &Array1<f64>| Ok(None),
    )
    .expect("joint Jeffreys term at the crossing fixture")
    .0
}

/// TEMPORARY gam#2695 probe — does `H` close across a knot the crossing row's
/// `q₀` walks over, by degree, by warp amplitude, and by knot convention?
///
/// The crossing row must be an EVENT row: the `log g` channel that carries
/// `m1 = 1 + Σ βw_j I′_j(q1)` — and with it the `I″` term that has no `βw`
/// factor — is added only when `w·d ≠ 0`, so a censored crossing row measures
/// the warp's VALUE channel alone and reports "continuous" for the wrong
/// reason. (It did: the first run of this probe crossed row 1, `event = 0`.)
#[test]
fn probe_2695_joint_hessian_across_an_interior_knot() {
    let primaries: Vec<[f64; SLS_ROW_K]> = vec![
        [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
        [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
        [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
        [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
    ];
    let event = [1.0, 0.0, 1.0, 1.0];
    let weight = [1.0, 0.8, 1.2, 1.1];
    let n = primaries.len();
    let q0_slope = |row: usize| -> f64 { -primaries[row][3] * (-primaries[row][6]).exp() };
    const B_STAR: f64 = 1.0;
    // event = 1.0, so the `log g` channel is live on the row that crosses.
    const CROSSING_ROW: usize = 2;

    for degree in 2..=5usize {
        for warp_amplitude in [1.0e-6_f64, 3.0e-2] {
            for clamped in [true, false] {
                let centre = q0_slope(CROSSING_ROW) * B_STAR;
                let knots = if clamped {
                    let left = centre - 2.0;
                    let right = centre + 2.0;
                    let mut values = vec![left; degree + 1];
                    values.push(centre - 1.0);
                    values.push(centre);
                    values.push(centre + 1.0);
                    values.extend(std::iter::repeat_n(right, degree + 1));
                    Array1::from_vec(values)
                } else {
                    // A simple-knot warp vector whose grid lands ON `centre`.
                    let spans = 4usize;
                    let width = 1.0_f64;
                    Array1::from_shape_fn(spans + 1 + 2 * degree, |i| {
                        centre + width * (i as f64 - (degree + 2) as f64)
                    })
                };

                let seed_q0 = Array1::from_shape_fn(n, |i| q0_slope(i) * B_STAR);
                let xwiggle = survival_wiggle_basis_with_options(
                    seed_q0.view(),
                    &knots,
                    degree,
                    BasisOptions::value(),
                )
                .expect("link wiggle design");
                let pw = xwiggle.ncols();
                let inverse_link =
                    residual_distribution_inverse_link(ResidualDistribution::Gaussian);
                let mut family =
                    survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
                family.x_link_wiggle = Some(DesignMatrix::Dense(
                    gam_linalg::matrix::DenseDesignMatrix::from(xwiggle.clone()),
                ));
                family.wiggle_knots = Some(knots.clone());
                family.wiggle_degree = Some(degree);
                let beta_w =
                    Array1::from_shape_fn(pw, |j| warp_amplitude * (1.0 + 0.3 * (j as f64)));

                let hessian_at = |beta_thr: f64| -> Result<Array2<f64>, String> {
                    let stacked = |first: usize, second: usize, deriv: usize, scale: f64| {
                        let mut eta = Array1::<f64>::zeros(3 * n);
                        for i in 0..n {
                            eta[i] = primaries[i][first] * scale;
                            eta[n + i] = primaries[i][second] * scale;
                            eta[2 * n + i] = primaries[i][deriv] * scale;
                        }
                        eta
                    };
                    let states = vec![
                        ParameterBlockState { beta: array![1.0], eta: stacked(0, 1, 2, 1.0) },
                        ParameterBlockState {
                            beta: array![beta_thr],
                            eta: stacked(3, 4, 5, beta_thr),
                        },
                        ParameterBlockState { beta: array![1.0], eta: stacked(6, 7, 8, 1.0) },
                        ParameterBlockState {
                            beta: beta_w.clone(),
                            eta: xwiggle.dot(&beta_w),
                        },
                    ];
                    family
                        .exact_newton_joint_hessian(&states)
                        .and_then(|h| h.ok_or_else(|| "no joint hessian".to_string()))
                };

                let gap = |h: f64| -> Result<f64, String> {
                    let plus = hessian_at(B_STAR + h)?;
                    let minus = hessian_at(B_STAR - h)?;
                    Ok(plus
                        .iter()
                        .zip(minus.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max))
                };
                let label = if clamped { "clamped" } else { "warp" };
                match (gap(1.0e-3), gap(1.0e-5)) {
                    (Ok(coarse), Ok(fine)) => println!(
                        "[2695-INT] degree={degree} knots={label} pw={pw} \
                         betaw={warp_amplitude:.1e} gapH(1e-3)={coarse:.6e} \
                         gapH(1e-5)={fine:.6e} ratio={:.3e} (100x = continuous, 1x = a jump)",
                        coarse / fine.max(f64::MIN_POSITIVE)
                    ),
                    (Err(error), _) | (_, Err(error)) => println!(
                        "[2695-INT] degree={degree} knots={label} pw={pw} \
                         betaw={warp_amplitude:.1e} REFUSED: {error}"
                    ),
                }
            }
        }
    }
}


// The INDEPENDENT dense oracle `SurvivalLsJointNllProgram` is restored here
// too: `d484a091a` deleted it (a test-only `RowProgram<9>` impl, so vacuously
// unreachable from any binary) and only a doc comment elsewhere in this file
// still named it. It re-enters verbatim, because its whole value is that it is
// a SECOND, generic expression of the row NLL that the production hand path is
// graded against — rewriting it would forfeit that independence. A trait impl
// needs a named type, so this one item cannot be a closure; the two gates that
// consume it own nothing else a sweep can name.

/// #932 (survival follow-up, the issue's named next step): the survival
/// location-scale JOINT row NLL written ONCE over generic jet scalars in the
/// production kernel's nine linear-predictor primaries
/// `(h0, h1, d_raw, eta_t_exit, eta_t_entry, eta_t_deriv, eta_ls_exit,
/// eta_ls_entry, eta_ls_deriv)` — the exact `SLS_ROW_K` channel layout of
/// [`SurvivalLsRowKernel`]. The whole nonlinear composition that the
/// production path hand-writes is expressed here as generic `JetScalar`
/// arithmetic:
///
/// ```text
///   u0 = (h0 − eta_t_entry) · exp(−eta_ls_entry)          (entry index)
///   u1 = (h1 − eta_t_exit)  · exp(−eta_ls_exit)           (exit index)
///   g  = exp(−eta_ls_exit)·(d_raw − eta_t_deriv − (h1 − eta_t_exit)·eta_ls_deriv)
///   nll = w·[ log S(u0) − (1−d)·log S(u1) − d·(log f(u1) + log g) ]
/// ```
///
/// so the tower mechanizes EXACTLY the calculus the hand path splits
/// across `q_chain_derivs_scalar` + `compose_survival_dynamic_q` (the
/// per-row `D/D2/D3` map tensors of `SurvivalLsRowKernel::row_maps`) and
/// the `row_kernel` / `row_third_contracted` Faà di Bruno accumulation
/// loops — the entry/exit/qdot cross blocks where the #736 bug genus
/// lives. Tail-critical primitives enter through the family's OWN
/// hand-certified stacks (`survival_ls_log_survival_stack` /
/// `_log_pdf_stack` / `_positive_log_stack`), so no probit/CLogLog/logit
/// primitive is re-derived: only the composition is mechanized.
struct SurvivalLsJointNllProgram<'a> {
    inverse_link: &'a InverseLink,
    primaries: Vec<[f64; SLS_ROW_K]>,
    event: Vec<f64>,
    weight: Vec<f64>,
}

impl gam_math::jet_tower::RowProgram<SLS_ROW_K> for SurvivalLsJointNllProgram<'_> {
    fn n_rows(&self) -> usize {
        self.primaries.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; SLS_ROW_K], String> {
        self.primaries
            .get(row)
            .copied()
            .ok_or_else(|| format!("survival LS joint program: row {row} out of range"))
    }

    fn eval<S: gam_math::jet_scalar::JetScalar<SLS_ROW_K>>(
        &self,
        row: usize,
        p: &[S; SLS_ROW_K],
    ) -> Result<S, String> {
        let w = *self
            .weight
            .get(row)
            .ok_or_else(|| format!("survival LS joint program: weight row {row} missing"))?;
        let d = self.event[row];
        if w <= 0.0 {
            return Ok(S::constant(0.0));
        }

        // Entry index: u0 = (h0 − eta_t_entry) · exp(−eta_ls_entry): the scale
        // divides the whole residual, time transform included (#2695).
        let inv_sigma_entry = p[7].neg().exp();
        let u0 = p[0].sub(&p[4]).mul(&inv_sigma_entry);
        // Exit index: u1 = (h1 − eta_t_exit) · exp(−eta_ls_exit).
        let inv_sigma_exit = p[6].neg().exp();
        let residual_exit = p[1].sub(&p[3]);
        let u1 = residual_exit.mul(&inv_sigma_exit);
        // Event Jacobian g = du1/dt
        //   = exp(−eta_ls_exit)·(d_raw − eta_t_deriv − (h1 − eta_t_exit)·eta_ls_deriv).
        let g = inv_sigma_exit.mul(&p[2].sub(&p[5]).sub(&residual_exit.mul(&p[8])));

        // NLL = w·log S(u0) − w(1−d)·log S(u1) − w·d·(log f(u1) + log g),
        // term-for-term the sign layout of `SurvivalExactRowKernel::
        // log_likelihood` / `nll_index_tower` (left truncation divides the
        // likelihood by S(u0), so its log ADDS to the NLL).
        let mut nll = u0
            .compose_unary(survival_ls_log_survival_stack(
                self.inverse_link,
                u0.value(),
            )?)
            .scale(w);

        let censored_weight = w * (1.0 - d);
        if censored_weight != 0.0 {
            nll = nll.add(
                &u1.compose_unary(survival_ls_log_survival_stack(
                    self.inverse_link,
                    u1.value(),
                )?)
                .scale(-censored_weight),
            );
        }

        let event_weight = w * d;
        if event_weight != 0.0 {
            nll = nll
                .add(
                    &u1.compose_unary(survival_ls_log_pdf_stack(
                        self.inverse_link,
                        u1.value(),
                        0.0,
                    )?)
                    .scale(-event_weight),
                )
                .add(
                    &g.compose_unary(survival_ls_positive_log_stack(g.value()))
                        .scale(-event_weight),
                );
        }

        Ok(nll)
    }
}

// ─── #932 packed-directional oracle, restored (#2818) ───────────────────────
//
// `c0a21b554` deleted both gates below because they no longer compiled, and they
// no longer compiled because `d484a091a` had deleted `Tower4::third_contracted`
// and `Tower4::fourth_contracted` from `gam-math` — two `pub` Rust-library
// methods whose only callers were tests, so neither emits a symbol in the CLI or
// pyffi binary and the sweep's criterion ("no production artifact links this
// function") held vacuously. The oracle they served —
// `SurvivalLsJointNllProgram` as an INDEPENDENT `RowProgram<9>`, and the
// production `RowKernel::row_third_contracted` / `row_fourth_contracted` under
// test — survived untouched.
//
// The rebuild contracts the dense tower from its `pub` `t3` / `t4` fields in the
// definitional full-nest form, inside the test. That is strictly stronger than
// what was lost: the oracle no longer routes through a shared `gam-math` helper
// the code under test could have been co-wrong with, and neither gate owns a
// named test-only item for a future sweep to prune — the fixture and the whole
// body are closures.

/// #932 single-source / packed-scalar contract: the production
/// `row_third_contracted` / `row_fourth_contracted` (evaluated through the
/// PACKED directional scalars `OneSeed<9>` / `TwoSeed<9>` — 1.46 / 2.8 KiB,
/// never the ~50 KiB dense `Tower4<9>`) must equal the contraction of the
/// INDEPENDENT dense `SurvivalLsJointNllProgram` `Tower4<9>` (a separate row-NLL
/// implementation, not the production `sls_row_nll`) to ≤ 1e-9. The packed
/// scalars fold the contraction direction INTO the differentiation via the
/// nilpotent ε/δ, never materialising `t3`/`t4`; the independent tower
/// materialises the full tensor then contracts. A regression that desyncs the
/// packed path from the dense answer — or reintroduces a separate hand
/// directional tower — fails here. This is the oracle that lets
/// `row_kernel_directional_supported()` return true: the memory-bounded packed
/// path is provably the dense-tower answer.
#[test]
fn survival_ls_packed_directional_matches_dense_tower_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            use crate::row_kernel::RowKernel;
            use gam_math::jet_tower::{Tower4, program_full_tower};

            let dense_third = |tower: &Tower4<SLS_ROW_K>, dir: &[f64; SLS_ROW_K]| {
                let mut out = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
                for a in 0..SLS_ROW_K {
                    for b in 0..SLS_ROW_K {
                        let mut acc = 0.0;
                        for c in 0..SLS_ROW_K {
                            acc += tower.t3[a][b][c] * dir[c];
                        }
                        out[a][b] = acc;
                    }
                }
                out
            };
            let dense_fourth = |tower: &Tower4<SLS_ROW_K>,
                                u: &[f64; SLS_ROW_K],
                                v: &[f64; SLS_ROW_K]| {
                let mut out = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
                for a in 0..SLS_ROW_K {
                    for b in 0..SLS_ROW_K {
                        let mut acc = 0.0;
                        for c in 0..SLS_ROW_K {
                            for d in 0..SLS_ROW_K {
                                acc += tower.t4[a][b][c][d] * u[c] * v[d];
                            }
                        }
                        out[a][b] = acc;
                    }
                }
                out
            };

            let primaries: Vec<[f64; SLS_ROW_K]> = vec![
                [0.2, 0.9, 1.3, 0.6, 0.4, 0.25, 0.3, 0.1, -0.2],
                [-0.4, 0.5, 0.9, -0.8, -0.5, 0.4, -0.25, 0.35, 0.3],
                [1.4, 2.1, 0.8, -1.1, -0.9, 0.2, 0.45, 0.55, -0.35],
                [0.1, 0.6, 1.0, 0.3, 0.2, -0.3, -0.2, 0.15, 0.25],
            ];
            let event = [1.0, 0.0, 1.0, 0.35];
            let weight = [1.0, 0.8, 1.2, 1.3];
            let n = primaries.len();

            // Dense deterministic directions so every one of the nine channels
            // participates in every contraction (no dropped/flipped cross block
            // can hide).
            let dirs: [[f64; SLS_ROW_K]; 3] = [
                [0.7, -1.3, 0.5, 0.9, -0.6, 0.3, -1.1, 0.4, 0.8],
                [-0.4, 0.6, -1.1, 0.3, 1.2, -0.7, 0.5, -0.9, 0.2],
                [1.2, 0.2, -0.7, -0.5, 0.4, 1.0, -0.3, 0.6, -1.2],
            ];

            // Non-vacuity: agreement is only evidence where the quantities are
            // free to disagree, so track the largest dense third/fourth entry the
            // gate actually judged and refuse a numerically dead fixture.
            let mut max_dense_third = 0.0_f64;
            let mut max_dense_fourth = 0.0_f64;

            for distribution in [
                ResidualDistribution::Gaussian,
                ResidualDistribution::Gumbel,
                ResidualDistribution::Logistic,
            ] {
                let inverse_link = residual_distribution_inverse_link(distribution);
                let family =
                    survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
                let states = survival_ls_joint_oracle_states(&primaries);
                let dynamic = family
                    .build_dynamic_geometry(&states)
                    .expect("dynamic geometry");
                let kernel = SurvivalLsRowKernel {
                    family: &family,
                    dynamic: &dynamic,
                    deriv_log_scale: 0.0,
                    offsets: family.joint_block_offsets(),
                };
                // INDEPENDENT dense ground truth: the `SurvivalLsJointNllProgram`
                // `RowProgram<9>` (the same one the all-channels oracle uses) — an
                // independent generic implementation of the row NLL, NOT the
                // production `sls_row_nll`. Comparing the packed production
                // contractions against THIS independent tower (rather than
                // `sls_row_nll` at `Tower4`) keeps the oracle's truth genuinely
                // independent of the code under test.
                let program = SurvivalLsJointNllProgram {
                    inverse_link: &inverse_link,
                    primaries: primaries.clone(),
                    event: event.to_vec(),
                    weight: weight.to_vec(),
                };

                for row in 0..n {
                    // Dense ground truth: build the full Tower4<9> once and
                    // contract its t3 / t4 channels. The production methods must
                    // reproduce these exactly through the packed OneSeed /
                    // TwoSeed scalars.
                    let tower = program_full_tower(&program, row).expect("dense row tower");
                    for u in &dirs {
                        let dense_third_row = dense_third(&*tower, u);
                        let packed_third =
                            RowKernel::row_third_contracted(&kernel, row, u).expect("packed third");
                        for a in 0..SLS_ROW_K {
                            for b in 0..SLS_ROW_K {
                                let want = dense_third_row[a][b];
                                let got = packed_third[a][b];
                                max_dense_third = max_dense_third.max(want.abs());
                                assert!(
                                    (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                                    "{distribution:?} row {row} third[{a}][{b}]: packed OneSeed \
                                     {got} vs dense Tower4 {want}"
                                );
                            }
                        }
                        for v in &dirs {
                            let dense_fourth_row = dense_fourth(&*tower, u, v);
                            let packed_fourth =
                                RowKernel::row_fourth_contracted(&kernel, row, u, v)
                                    .expect("packed fourth");
                            for a in 0..SLS_ROW_K {
                                for b in 0..SLS_ROW_K {
                                    let want = dense_fourth_row[a][b];
                                    let got = packed_fourth[a][b];
                                    max_dense_fourth = max_dense_fourth.max(want.abs());
                                    assert!(
                                        (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                                        "{distribution:?} row {row} fourth[{a}][{b}]: packed \
                                         TwoSeed {got} vs dense Tower4 {want}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
            assert!(
                max_dense_third > 1e-3 && max_dense_fourth > 1e-3,
                "the fixture's dense third/fourth contractions are numerically dead \
                 (max|third|={max_dense_third:.3e}, max|fourth|={max_dense_fourth:.3e}); the \
                 parity assertions above would then agree with anything"
            );
        })
        .expect("spawn wide-stack packed-directional oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS packed-directional #932 oracle thread must complete"
    );
}

/// #932 STRESS hardening of [`survival_ls_packed_directional_matches_dense_tower_932`].
///
/// The benign-fixture oracle above proves the packed `OneSeed`/`TwoSeed`
/// contractions equal the independent dense `Tower4<9>` on moderate primaries.
/// This arm hardens that gate in two ways a benign equality check cannot:
///
///   1. HIGH-CURVATURE / NEAR-DEGENERATE fixture. The primaries are pushed into
///      the regime where the per-row NLL stacks saturate and their high-order
///      jets blow up — exactly where a dropped/mis-scaled 3rd-or-4th-order term
///      hides on a benign point:
///        * deep-tail exit/entry indices `u0,u1` (large negative → `log S`
///          curvature large; the `compose_unary` survival stack is evaluated far
///          from 0 where its 3rd/4th derivatives dominate);
///        * extreme `log-σ` channels (`exp(−η_lσ)` spans ~e^−2..e^2), so the
///          threshold contributions to `u0,u1,g` are strongly amplified;
///        * a deliberately SMALL-but-positive event Jacobian `g` (near the
///          `log g` singularity, where `∂ⁿ log g = (−1)ⁿ⁻¹(n−1)!/gⁿ` is huge),
///          stressing the `survival_ls_positive_log_stack` chain at 3rd/4th order.
///      A vacuity guard asserts the fixture actually reaches this regime (small
///      `g`, large `|u|`) so the stress is real, not nominal.
///
///   2. PLANTED SIGN-FLIP tripwire. Equality `packed == dense` alone does not
///      prove the oracle could SEE a wrong packed value. After the exact match we
///      negate a representative 4th-order cross entry and assert the packed value
///      does NOT match the flip — i.e. the oracle has genuine resolving power
///      against a sign/term error on the very block it guards.
#[test]
fn survival_ls_packed_directional_matches_dense_tower_high_curvature_932() {
    let join_result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            use crate::row_kernel::RowKernel;
            use gam_math::jet_tower::{Tower4, program_full_tower};

            let dense_third = |tower: &Tower4<SLS_ROW_K>, dir: &[f64; SLS_ROW_K]| {
                let mut out = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
                for a in 0..SLS_ROW_K {
                    for b in 0..SLS_ROW_K {
                        let mut acc = 0.0;
                        for c in 0..SLS_ROW_K {
                            acc += tower.t3[a][b][c] * dir[c];
                        }
                        out[a][b] = acc;
                    }
                }
                out
            };
            let dense_fourth = |tower: &Tower4<SLS_ROW_K>,
                                u: &[f64; SLS_ROW_K],
                                v: &[f64; SLS_ROW_K]| {
                let mut out = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
                for a in 0..SLS_ROW_K {
                    for b in 0..SLS_ROW_K {
                        let mut acc = 0.0;
                        for c in 0..SLS_ROW_K {
                            for d in 0..SLS_ROW_K {
                                acc += tower.t4[a][b][c][d] * u[c] * v[d];
                            }
                        }
                        out[a][b] = acc;
                    }
                }
                out
            };

            // Channel layout (matches `SurvivalLsJointNllProgram::eval`):
            //   [0]=t_entry [1]=t_exit [2]=t_deriv [3]=thr_exit [4]=thr_entry
            //   [5]=thr_deriv [6]=lσ_exit [7]=lσ_entry [8]=lσ_deriv.
            // These rows drive `u0,u1` deep into the tail and `g` small-positive:
            //   inv_σ_exit = e^{−p6}, u1 = (p1 − p3)·inv_σ_exit,
            //   du1/dt = inv_σ_exit·g, g = p2 − p5 − (p1 − p3)·p8 (must stay > 0
            //   for log g). The scale's `−p6` enters `log(du1/dt)` linearly, so
            //   `g` alone sets the size of the log g jets (#2695).
            let primaries: Vec<[f64; SLS_ROW_K]> = vec![
                // Large log-σ swing (p6=1.8 ⇒ inv_σ_exit≈0.165; p7=−1.6 ⇒
                // inv_σ_entry≈4.95), big thresholds ⇒ |u0|,|u1| large.
                [2.4, -3.1, 0.9, 2.2, 3.5, 0.7, 1.8, -1.6, 0.5],
                // Deep-tail censored row with a SMALL event-Jacobian-style g build
                // and a strongly negative exit index.
                [-2.8, -4.2, 0.35, -2.6, -3.4, 1.3, -1.7, 1.5, 0.9],
                // Near-degenerate g: p2=0.12, g=0.12−0.18−2.4·(−0.6)=1.38 → still
                // positive but with large threshold curvature feeding u1.
                [0.6, 3.8, 0.12, 1.4, 2.1, 0.18, 0.4, -0.5, -0.6],
                // Tiny g with big tail: p2=0.05, g=0.05−0.05−3.7·(−0.04)=0.148
                // (small ⇒ huge log g jets).
                [-1.2, 4.6, 0.05, 0.9, -2.3, 0.05, 1.1, -1.3, -0.04],
            ];
            let event = [1.0, 0.0, 1.0, 1.0];
            let weight = [1.0, 0.9, 1.2, 0.8];
            let n = primaries.len();

            let dirs: [[f64; SLS_ROW_K]; 3] = [
                [0.7, -1.3, 0.5, 0.9, -0.6, 0.3, -1.1, 0.4, 0.8],
                [-0.4, 0.6, -1.1, 0.3, 1.2, -0.7, 0.5, -0.9, 0.2],
                [1.2, 0.2, -0.7, -0.5, 0.4, 1.0, -0.3, 0.6, -1.2],
            ];

            // Vacuity guard: confirm at least one event row actually reaches the
            // high-curvature regime — a small-positive `g` and a large-magnitude
            // exit index `u1` — so the stress is genuine, not a nominal
            // relabelling of a benign point.
            let mut min_event_g = f64::INFINITY;
            let mut max_abs_u1 = 0.0_f64;
            for (row, p) in primaries.iter().enumerate() {
                let inv_sigma_exit = (-p[6]).exp();
                let u1 = (p[1] - p[3]) * inv_sigma_exit;
                let g = p[2] - p[5] - (p[1] - p[3]) * p[8];
                assert!(
                    g > 0.0,
                    "fixture row {row} has non-positive event Jacobian g={g:.4e}; log g undefined"
                );
                if event[row] != 0.0 {
                    min_event_g = min_event_g.min(g);
                }
                max_abs_u1 = max_abs_u1.max(u1.abs());
            }
            assert!(
                min_event_g < 0.2,
                "high-curvature fixture vacuous: smallest event-row g={min_event_g:.4e} is not \
                 near the log g singularity (want < 0.2); the small-g 3rd/4th-order stress is \
                 absent"
            );
            assert!(
                max_abs_u1 > 3.0,
                "high-curvature fixture vacuous: largest |u1|={max_abs_u1:.4e} is not deep in \
                 the survival tail (want > 3.0); the saturated log-survival curvature stress is \
                 absent"
            );

            for distribution in [
                ResidualDistribution::Gaussian,
                ResidualDistribution::Gumbel,
                ResidualDistribution::Logistic,
            ] {
                let inverse_link = residual_distribution_inverse_link(distribution);
                let family =
                    survival_ls_joint_oracle_family(&inverse_link, &primaries, &event, &weight);
                let states = survival_ls_joint_oracle_states(&primaries);
                let dynamic = family
                    .build_dynamic_geometry(&states)
                    .expect("dynamic geometry");
                let kernel = SurvivalLsRowKernel {
                    family: &family,
                    dynamic: &dynamic,
                    deriv_log_scale: 0.0,
                    offsets: family.joint_block_offsets(),
                };
                let program = SurvivalLsJointNllProgram {
                    inverse_link: &inverse_link,
                    primaries: primaries.clone(),
                    event: event.to_vec(),
                    weight: weight.to_vec(),
                };

                // A slightly looser relative tolerance than the benign oracle's
                // 1e-9: the deep-tail/small-g jets have magnitudes up to ~1e3, so
                // the `(1+|want|)` relative band already scales with that; the
                // absolute floor stays tight. A genuine dropped term is
                // O(magnitude), far outside this.
                let rel_tol = 1e-8_f64;

                for row in 0..n {
                    let tower = program_full_tower(&program, row)
                        .expect("dense row tower (high curvature)");
                    for u in &dirs {
                        let dense_third_row = dense_third(&*tower, u);
                        let packed_third =
                            RowKernel::row_third_contracted(&kernel, row, u).expect("packed third");
                        for a in 0..SLS_ROW_K {
                            for b in 0..SLS_ROW_K {
                                let want = dense_third_row[a][b];
                                let got = packed_third[a][b];
                                assert!(
                                    (got - want).abs() <= rel_tol * (1.0 + want.abs()),
                                    "{distribution:?} HC row {row} third[{a}][{b}]: packed \
                                     OneSeed {got} vs dense Tower4 {want}"
                                );
                            }
                        }
                        for v in &dirs {
                            let dense_fourth_row = dense_fourth(&*tower, u, v);
                            let packed_fourth =
                                RowKernel::row_fourth_contracted(&kernel, row, u, v)
                                    .expect("packed fourth");
                            for a in 0..SLS_ROW_K {
                                for b in 0..SLS_ROW_K {
                                    let want = dense_fourth_row[a][b];
                                    let got = packed_fourth[a][b];
                                    assert!(
                                        (got - want).abs() <= rel_tol * (1.0 + want.abs()),
                                        "{distribution:?} HC row {row} fourth[{a}][{b}]: packed \
                                         TwoSeed {got} vs dense Tower4 {want}"
                                    );
                                }
                            }
                        }
                    }
                }

                // ── Planted sign-flip tripwire ──────────────────────────────
                // Pick the event row with the smallest g (max log-g curvature)
                // and a 4th-order cross entry that is genuinely nonzero, then
                // assert that negating the dense truth leaves the packed band:
                // the oracle can SEE a sign/term error on the block it guards
                // (not just confirm equality).
                let trip_row = 3usize; // tiny-g, deep-tail event row
                let du = &dirs[0];
                let dv = &dirs[1];
                let trip_tower = program_full_tower(&program, trip_row).expect("trip tower");
                let dense_fourth_trip = dense_fourth(&*trip_tower, du, dv);
                let packed_fourth = RowKernel::row_fourth_contracted(&kernel, trip_row, du, dv)
                    .expect("trip packed fourth");
                // (t_deriv, lσ_deriv) = [2][8]: a cross block that genuinely
                // couples the event Jacobian's time and scale rates through g.
                // (`[2][6]` is structurally zero: the scale enters `log(du1/dt)`
                // as the linear `−lσ_exit`, #2695.)
                let (ca, cb) = (2usize, 8usize);
                let want = dense_fourth_trip[ca][cb];
                assert!(
                    want.abs() > 1e-6,
                    "{distribution:?} the tripwire entry fourth[{ca}][{cb}] = {want:+.9e} is \
                     numerically zero, so the sign-flip control never ran"
                );
                let flipped = -packed_fourth[ca][cb];
                assert!(
                    (flipped - want).abs() > 1e-8 * (1.0 + want.abs()),
                    "{distribution:?} oracle failed to reject a planted fourth[{ca}][{cb}] sign \
                     flip: flipped {flipped:+.9e} vs dense truth {want:+.9e} — the \
                     high-curvature gate has no resolving power against a cross-block sign error"
                );
            }
        })
        .expect("spawn wide-stack high-curvature packed-directional oracle thread")
        .join();
    assert!(
        join_result.is_ok(),
        "survival LS high-curvature #932 oracle thread must complete"
    );
}

/// #2390 (#2385 instance 1), production path: one NEAR-wall link-wiggle
/// coordinate must not erase the block's cross-covariance with
/// `(h, threshold, log σ)` from the exact response moments.
///
/// The terminal covariance is already computed on the ACTIVE FACE, so a
/// genuinely PINNED coordinate arrives with an exactly-zero covariance row and
/// contributes no displacement at all. The coordinates that actually bind the
/// feasibility clip are the near-wall but still-SLACK ones just outside the
/// `1e-10` tightness band — routinely occupied, since the inner constrained
/// solve's own KKT band is `1e-6·scale + 1e-10`. A single global
/// fraction-to-boundary factor multiplies EVERY coordinate's displacement by
/// that coordinate's ~`1e-8` ratio, freezing the conditional mean at `β̂_w` for
/// every latent node.
///
/// Observable signature, with no reference to any hand-computed moment: negate
/// the whole link-wiggle ↔ `(h, threshold, log σ)` cross block. That is the
/// congruence `Σ' = D Σ D` with `D = diag(I, −I)`, so `Σ'` is PSD, its
/// `(h, threshold, log σ)` projection is unchanged, and the conditional
/// covariance `cov_ww − R Rᵀ` is unchanged (`R → −R`). The ONLY thing that
/// changes is the SIGN of the conditional-mean displacement — i.e. the sign of
/// the correlation between the realized warp and the realized predictor. If the
/// displacement has been frozen, both covariances give the same moments.
#[test]
pub(crate) fn near_wall_wiggle_coordinate_keeps_cross_covariance_in_moments_2390() {
    // Coordinate 1 sits just OUTSIDE the 1e-10 active-face tightness band:
    // slack, so it keeps a full-width covariance row, but max(β̂,0)/|d| ~ 1e-8.
    let fit = test_survival_fit(
        array![0.4, -0.1],
        array![0.2, 0.3],
        array![-0.5, 0.1],
        Some(array![0.30, 1.0e-9]),
    );
    let x_threshold_dense = array![[1.0, -0.2]];
    let x_log_sigma_dense = array![[1.0, 0.3]];
    let eta_threshold_offset = array![0.7];
    let eta_log_sigma_offset = array![0.4];
    let eta_t = x_threshold_dense.dot(&fit.beta_threshold()) + &eta_threshold_offset;
    let eta_ls = x_log_sigma_dense.dot(&fit.beta_log_sigma()) + &eta_log_sigma_offset;
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(eta_ls.iter())
            .map(|(&t, &ls)| -t * exp_sigma_inverse_from_eta_scalar(ls)),
    );
    let degree = fit
        .artifacts
        .survival_link_wiggle_degree
        .expect("fit wiggle degree");
    let base_knots = fit
        .artifacts
        .survival_link_wiggle_knots
        .clone()
        .expect("fit wiggle knots");
    // Re-center the wiggle knots on the realized q0 so both I-spline columns
    // carry weight there. A basis row of zeros would make the whole comparison
    // vacuous, so the centering is asserted below rather than assumed.
    let lo = base_knots.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = base_knots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shift = q0[0] - 0.5 * (lo + hi);
    let knots = base_knots.mapv(|k| k + shift);
    let basis = survival_wiggle_basis_with_options(q0.view(), &knots, degree, BasisOptions::value())
        .expect("link wiggle basis");
    assert!(
        basis[[0, 0]] > 1.0e-3,
        "the interior wiggle coordinate must carry basis weight at q0, got {}",
        basis[[0, 0]]
    );

    let input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5]],
        eta_time_offset_exit: array![0.2],
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_threshold_dense.clone(),
        )),
        eta_threshold_offset,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_log_sigma_dense.clone(),
        )),
        eta_log_sigma_offset,
        x_link_wiggle: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(basis.clone()),
        )),
        link_wiggle_knots: Some(knots.clone()),
        link_wiggle_degree: Some(degree),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };

    // Σ = G Gᵀ is PSD by construction, and both link-wiggle rows load on the
    // same latent factors as the time / threshold / log-sigma rows, so the
    // wiggle block carries a substantial cross-covariance with them.
    let g = array![
        [0.18, 0.00, 0.00, 0.00],
        [0.05, 0.16, 0.00, 0.00],
        [0.00, 0.07, 0.15, 0.00],
        [0.04, 0.00, 0.13, 0.00],
        [0.00, 0.06, 0.00, 0.14],
        [0.03, 0.00, 0.05, 0.12],
        [0.12, 0.10, 0.09, 0.08],
        [0.09, 0.11, 0.07, 0.10],
    ];
    let covariance = g.dot(&g.t());
    let mut flipped = covariance.clone();
    for j in 0..6 {
        for w in 6..8 {
            flipped[[j, w]] = -covariance[[j, w]];
            flipped[[w, j]] = -covariance[[w, j]];
        }
    }

    let (mean, second) =
        exact_survival_response_moments(&input, &fit, &covariance).expect("response moments");
    let (mean_flipped, second_flipped) =
        exact_survival_response_moments(&input, &fit, &flipped).expect("flipped response moments");

    let mean_gap = (mean[0] - mean_flipped[0]).abs();
    let second_gap = (second[0] - second_flipped[0]).abs();
    assert!(
        mean_gap > 1.0e-8,
        "the near-wall wiggle coordinate erased the cross-covariance from E[S]: \
         {} vs {} (gap {mean_gap:.3e})",
        mean[0],
        mean_flipped[0]
    );
    assert!(
        second_gap > 1.0e-8,
        "the near-wall wiggle coordinate erased the cross-covariance from E[S^2]: \
         {} vs {} (gap {second_gap:.3e})",
        second[0],
        second_flipped[0]
    );
    // The moments are still probabilities, and the second moment still respects
    // Jensen against the first.
    for (m1, m2) in [(mean[0], second[0]), (mean_flipped[0], second_flipped[0])] {
        assert!((0.0..=1.0).contains(&m1) && (0.0..=1.0).contains(&m2));
        assert!(
            m2 + 1.0e-9 >= m1 * m1,
            "E[S^2]={m2} must dominate E[S]^2={}",
            m1 * m1
        );
    }
}

/// #2446: with a DETERMINISTIC wiggle basis row the whole linear predictor is a
/// single scalar Gaussian, so the nested outer×inner rule has a closed-form
/// answer and this is an IDENTITY, not an A/B.
///
/// Setting the threshold and log-sigma covariance blocks to exactly zero makes
/// `q0` — and therefore the I-spline row `b` — a constant, and leaves
///
/// ```text
///   η = s·h + q0 + bᵀβ_w,     E[η] = s·μ_h + q0 + bᵀE_π[β_w],
///   Var[η] = s²·aᵀΣ_hh a + 2s·aᵀΣ_hw b + bᵀΣ_ww b,
///
/// with `s = e^{−μ_ls}` deterministic too: the scale divides the time transform
/// (#2695).
/// ```
///
/// Production must return `E[S(η)]` and `E[S(η)²]` for THAT scalar Gaussian.
/// The reference integrates it by direct density quadrature — deliberately not
/// production's Gauss-Hermite rule, so the construction is independent of the
/// rule under test.
///
/// The identity holds because the outer×inner factorization of a joint Gaussian
/// is exact when the inner mean is the AFFINE conditional mean. It fails as soon
/// as anything nonlinear is applied to that mean, which is what the
/// fraction-to-boundary cone clip did: it removed essentially the whole
/// `2·aᵀΣ_hw b` cross term. The two assertions below the fixture pin that this
/// gate is not vacuous — the clip bound at every latent node (the per-node
/// displacement scale is orders above each wall) and the term it destroyed is a
/// large share of `Var[η]`.
#[test]
fn nested_response_moment_rule_reproduces_the_scalar_gaussian_law_2446() {
    // Near-wall link-wiggle coefficients: `E_π[β_w]` is interior but tiny, which
    // is the regime the cone clip was written for and the regime it broke.
    let beta_w = array![0.02, 0.02];
    let fit = test_survival_fit(
        array![0.4, -0.1],
        array![0.2, 0.3],
        array![-0.5, 0.1],
        Some(beta_w.clone()),
    );
    let a_h = array![1.0, 0.5];
    let eta_time_offset_exit = array![0.2];
    let x_threshold_dense = array![[1.0, -0.2]];
    let x_log_sigma_dense = array![[1.0, 0.3]];
    let eta_threshold_offset = array![0.7];
    let eta_log_sigma_offset = array![0.4];

    let mu_h = a_h.dot(&fit.beta_time()) + eta_time_offset_exit[0];
    let mu_t = x_threshold_dense.row(0).dot(&fit.beta_threshold()) + eta_threshold_offset[0];
    let mu_ls = x_log_sigma_dense.row(0).dot(&fit.beta_log_sigma()) + eta_log_sigma_offset[0];
    // Deterministic because the threshold and log-sigma covariance blocks are
    // zero below, so production evaluates `q0` at exactly this point.
    let q0 = survival_q0_from_eta(mu_t, mu_ls);

    let degree = fit
        .artifacts
        .survival_link_wiggle_degree
        .expect("fit wiggle degree");
    let base_knots = fit
        .artifacts
        .survival_link_wiggle_knots
        .clone()
        .expect("fit wiggle knots");
    // Re-center the knots on the realized q0 so BOTH I-spline columns carry
    // weight there; a zero basis row would make the comparison vacuous.
    let lo = base_knots.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = base_knots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let knots = base_knots.mapv(|k| k + (q0 - 0.5 * (lo + hi)));
    let basis = survival_wiggle_basis_with_options(
        Array1::from_vec(vec![q0]).view(),
        &knots,
        degree,
        BasisOptions::value(),
    )
    .expect("link wiggle basis");
    let b = basis.row(0).to_owned();
    assert!(
        b[0] > 1.0e-3 && b[1] > 1.0e-3,
        "both wiggle coordinates must carry basis weight at q0, got {b:?}"
    );

    let input = SurvivalLocationScalePredictInput {
        x_time_exit: array![[1.0, 0.5]],
        eta_time_offset_exit,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_threshold_dense.clone(),
        )),
        eta_threshold_offset,
        x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            x_log_sigma_dense.clone(),
        )),
        eta_log_sigma_offset,
        x_link_wiggle: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(basis.clone()),
        )),
        link_wiggle_knots: Some(knots.clone()),
        link_wiggle_degree: Some(degree),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
    };

    // Blocks are [time(0..2), threshold(2..4), log_sigma(4..6), wiggle(6..8)].
    // The `(time, wiggle)` sub-block is `F Fᵀ`, hence PSD, and the threshold and
    // log-sigma rows are exactly zero — that is what makes `q0` and `b`
    // deterministic and the predictor one scalar Gaussian.
    let f = array![
        [0.30, 0.00, 0.00, 0.00],
        [0.10, 0.25, 0.00, 0.00],
        [0.20, 0.10, 0.15, 0.00],
        [0.18, 0.12, 0.00, 0.14],
    ];
    let block = f.dot(&f.t());
    let mut covariance = Array2::<f64>::zeros((8, 8));
    for i in 0..2 {
        for j in 0..2 {
            covariance[[i, j]] = block[[i, j]];
            covariance[[i, 6 + j]] = block[[i, 2 + j]];
            covariance[[6 + j, i]] = block[[2 + j, i]];
            covariance[[6 + i, 6 + j]] = block[[2 + i, 2 + j]];
        }
    }

    let s_hh = covariance.slice(s![0..2, 0..2]).to_owned();
    let s_hw = covariance.slice(s![0..2, 6..8]).to_owned();
    let s_ww = covariance.slice(s![6..8, 6..8]).to_owned();
    let scale = exp_sigma_inverse_from_eta_scalar(mu_ls);
    let var_h = a_h.dot(&s_hh.dot(&a_h));
    let cross = a_h.dot(&s_hw.dot(&b));
    let mean_eta = scale * mu_h + q0 + b.dot(&beta_w);
    let var_eta = scale * scale * var_h + 2.0 * scale * cross + b.dot(&s_ww.dot(&b));
    assert!(var_h > 0.0 && var_eta > 0.0, "degenerate fixture");

    // Non-vacuity 1: the removed clip bound at essentially every latent node.
    // The per-node conditional-mean displacement is `Σ_wh a / sd_h · z`, so this
    // is its scale at `|z| = 1` against each coordinate's wall.
    let displacement_scale = s_hw.t().dot(&a_h).mapv(f64::abs) / var_h.sqrt();
    for j in 0..2 {
        assert!(
            displacement_scale[j] > 5.0 * beta_w[j],
            "coordinate {j} is not near-wall: displacement scale {} vs wall {}",
            displacement_scale[j],
            beta_w[j]
        );
    }
    // Non-vacuity 2: the cross term the clip destroyed is a large share of the
    // variance being asserted, so passing this cannot be a coincidence.
    assert!(
        2.0 * scale * cross > 0.3 * var_eta,
        "the cross term {} is too small a share of Var[eta] {var_eta} to gate anything",
        2.0 * scale * cross
    );

    // Reference: E[S] and E[S^2] for the scalar Gaussian above, by DIRECT
    // density quadrature. Not production's Gauss-Hermite rule — the point is an
    // independent construction. Beyond ±10 sd the density is below 1e-22.
    let sd_eta = var_eta.sqrt();
    let (nodes, weights) = gam_math::special::gauss_legendre(96);
    let half = 10.0 * sd_eta;
    let mut reference_first = 0.0;
    let mut reference_second = 0.0;
    let mut mass = 0.0;
    for (t, wgt) in nodes.iter().zip(weights.iter()) {
        let eta = mean_eta + half * t;
        let standardized = half * t / sd_eta;
        let weight = half * wgt * (-0.5 * standardized * standardized).exp();
        let p = inverse_link_survival_prob_checked(&input.inverse_link, eta).expect("inverse link");
        reference_first += weight * p;
        reference_second += weight * p * p;
        mass += weight;
    }
    reference_first /= mass;
    reference_second /= mass;

    let (mean, second) =
        exact_survival_response_moments(&input, &fit, &covariance).expect("response moments");
    assert!(
        (mean[0] - reference_first).abs() <= 5.0e-5,
        "E[S] must equal the scalar-Gaussian law: production {} vs closed-form reference {} \
         (mean_eta={mean_eta:.6}, var_eta={var_eta:.6})",
        mean[0],
        reference_first
    );
    assert!(
        (second[0] - reference_second).abs() <= 5.0e-5,
        "E[S^2] must equal the scalar-Gaussian law: production {} vs closed-form reference {} \
         (mean_eta={mean_eta:.6}, var_eta={var_eta:.6})",
        second[0],
        reference_second
    );
}

/// #2695: the scale divides the whole standardized residual, time transform
/// included, so `log g` carries `−η_σ` and a covariate-dependent σ is identified.
///
/// The family's log-likelihood must equal the heteroscedastic Gaussian-residual
/// likelihood written out here from the textbook form
/// `u = (h − η_t)/σ`, `g = ḣ/σ`, `ℓ = w·[d(log φ(u1) + log g) + (1−d)·log S(u1) − log S(u0)]`.
/// The former residual `u = h − η_t·e^{−η_σ}` with `g = ḣ` depended on
/// `(η_t, η_σ)` only through `η_t·e^{−η_σ}`, so the per-row reparameterization
/// `(η_t, η_σ) → (c·η_t, η_σ + ln c)` left it exactly unchanged: σ(x) had no
/// likelihood of its own. The negative control computes that former closed form
/// and shows it is blind to the reparameterization while the family is not.
#[test]
fn the_scale_divides_the_time_transform_so_a_covariate_scale_is_identified_2695() {
    let family = survival_exact_newton_test_family();
    let n = family.n;
    let beta_time = 0.6;
    let base = survival_exact_newton_test_states(&family, beta_time, 0.0, 0.0);
    let h_entry: Vec<f64> = (0..n).map(|i| base[0].eta[i]).collect();
    let h_exit: Vec<f64> = (0..n).map(|i| base[0].eta[n + i]).collect();
    let hdot: Vec<f64> = (0..n).map(|i| base[0].eta[2 * n + i]).collect();
    let states_at = |eta_t: &Array1<f64>, eta_ls: &Array1<f64>| {
        let mut states = base.clone();
        states[1].eta = eta_t.clone();
        states[2].eta = eta_ls.clone();
        states
    };
    let log_pdf = |u: f64| -0.5 * u * u - 0.5 * (2.0 * std::f64::consts::PI).ln();
    let log_survival = |u: f64| gam_math::probability::normal_logsf(u);
    // `scaled == true` is the model; `false` is the former ratio-only residual.
    let closed_form = |eta_t: &Array1<f64>, eta_ls: &Array1<f64>, scaled: bool| -> f64 {
        (0..n)
            .map(|i| {
                let s = (-eta_ls[i]).exp();
                let (u0, u1, g) = if scaled {
                    ((h_entry[i] - eta_t[i]) * s, (h_exit[i] - eta_t[i]) * s, hdot[i] * s)
                } else {
                    (h_entry[i] - eta_t[i] * s, h_exit[i] - eta_t[i] * s, hdot[i])
                };
                let d = family.y[i];
                family.w[i]
                    * (d * (log_pdf(u1) + g.ln()) + (1.0 - d) * log_survival(u1)
                        - log_survival(u0))
            })
            .sum()
    };

    let eta_t = array![0.3, -0.2, 0.5];
    let eta_ls = array![0.4, -0.7, 0.9];
    let c = array![1.7, 0.6, 2.3];
    let eta_t_moved = &eta_t * &c;
    let eta_ls_moved = &eta_ls + &c.mapv(f64::ln);

    let family_base = family
        .log_likelihood_only(&states_at(&eta_t, &eta_ls))
        .expect("family log-likelihood at the base point");
    let family_moved = family
        .log_likelihood_only(&states_at(&eta_t_moved, &eta_ls_moved))
        .expect("family log-likelihood at the reparameterized point");
    let model_base = closed_form(&eta_t, &eta_ls, true);
    let model_moved = closed_form(&eta_t_moved, &eta_ls_moved, true);
    let former_base = closed_form(&eta_t, &eta_ls, false);
    let former_moved = closed_form(&eta_t_moved, &eta_ls_moved, false);

    // Agreement to rounding: every term is O(1), so the band is a few ulps of
    // the magnitudes summed.
    let band = |a: f64, b: f64| 64.0 * f64::EPSILON * (1.0 + a.abs() + b.abs());
    assert!(
        (family_base - model_base).abs() <= band(family_base, model_base),
        "family {family_base:.17e} vs heteroscedastic closed form {model_base:.17e}"
    );
    assert!(
        (family_moved - model_moved).abs() <= band(family_moved, model_moved),
        "family {family_moved:.17e} vs heteroscedastic closed form {model_moved:.17e} \
         at the reparameterized point"
    );
    // Negative control: the former residual cannot see the move ...
    assert!(
        (former_base - former_moved).abs() <= band(former_base, former_moved),
        "the ratio-only residual must be invariant under (c·η_t, η_σ + ln c): \
         {former_base:.17e} vs {former_moved:.17e}"
    );
    // ... while the model resolves it far beyond rounding, so σ(x) is identified.
    assert!(
        (family_base - family_moved).abs() > 1.0e6 * band(family_base, family_moved),
        "σ(x) is not identified: the family is blind to (c·η_t, η_σ + ln c): \
         {family_base:.17e} vs {family_moved:.17e}"
    );
}

/// #2695: a row entering at the origin is not left-truncated, so its likelihood
/// carries no `S(entry)` factor (`survival/base.rs` drops it by the same
/// `age_entry > ENTRY_AT_ORIGIN_THRESHOLD` predicate). The family's
/// log-likelihood and the kernel's entry stack must both see the entry term on
/// the left-truncated row alone. The control conditions every row on its entry
/// and must differ by far more than rounding, so the fixture's entry term is
/// live.
#[test]
fn a_row_entering_at_the_origin_carries_no_entry_factor_2695() {
    let active = [false, true, false];
    let family = SurvivalLocationScaleFamily {
        entry_active: Arc::from(active.to_vec()),
        ..survival_exact_newton_test_family()
    };
    let control = survival_exact_newton_test_family();
    let n = family.n;
    let states = survival_exact_newton_test_states(&family, 0.6, 0.2, -0.3);
    let h_entry: Vec<f64> = (0..n).map(|i| states[0].eta[i]).collect();
    let h_exit: Vec<f64> = (0..n).map(|i| states[0].eta[n + i]).collect();
    let hdot: Vec<f64> = (0..n).map(|i| states[0].eta[2 * n + i]).collect();
    let log_pdf = |u: f64| -0.5 * u * u - 0.5 * (2.0 * std::f64::consts::PI).ln();
    let log_survival = |u: f64| gam_math::probability::normal_logsf(u);
    let closed_form = |entry_on: &[bool]| -> f64 {
        (0..n)
            .map(|i| {
                let (eta_t, s) = (states[1].eta[i], (-states[2].eta[i]).exp());
                let (u0, u1) = ((h_entry[i] - eta_t) * s, (h_exit[i] - eta_t) * s);
                let d = family.y[i];
                let entry = if entry_on[i] { log_survival(u0) } else { 0.0 };
                family.w[i]
                    * (d * (log_pdf(u1) + (hdot[i] * s).ln()) + (1.0 - d) * log_survival(u1)
                        - entry)
            })
            .sum()
    };
    let band = |a: f64, b: f64| 64.0 * f64::EPSILON * (1.0 + a.abs() + b.abs());

    let masked = family.log_likelihood_only(&states).expect("masked log-likelihood");
    let expected = closed_form(&active);
    assert!(
        (masked - expected).abs() <= band(masked, expected),
        "an origin row must carry no S(entry): family {masked:.17e} vs closed form {expected:.17e}"
    );
    let conditioned = control.log_likelihood_only(&states).expect("control log-likelihood");
    let expected_conditioned = closed_form(&[true, true, true]);
    assert!(
        (conditioned - expected_conditioned).abs() <= band(conditioned, expected_conditioned),
        "control: family {conditioned:.17e} vs closed form {expected_conditioned:.17e}"
    );
    assert!(
        (masked - conditioned).abs() > 1.0e6 * band(masked, conditioned),
        "the fixture's entry term must be live: {masked:.17e} vs {conditioned:.17e}"
    );

    // The entry stack of an origin row is exactly zero.
    let dynamic = family.build_dynamic_geometry(&states).expect("dynamic geometry");
    for (row, &on) in active.iter().enumerate() {
        let state = family.row_predictor_state_at(&dynamic, row);
        let kernel = family
            .exact_row_kernel(row, state)
            .expect("row kernel")
            .expect("positive-weight row");
        let entry_stack = [kernel.log_s0, kernel.r0, kernel.dr0, kernel.ddr0, kernel.dddr0];
        assert_eq!(
            entry_stack.iter().all(|value| *value == 0.0),
            !on,
            "row {row}: entry stack {entry_stack:?} with entry_active={on}"
        );
    }
}

/// #2695: the explicit ψ terms `(V_ψ, g_ψ, H_ψ)` served from the row program are
/// the ψ-derivatives of the NLL value, coefficient gradient and coefficient
/// Hessian when ψ moves the threshold and log-σ designs `X → X + ψ·X_ψ`, at a
/// heteroscedastic point. Central differences of the family's own NLL, gradient
/// and Hessian at `ψ = ±h` are the second source; the step `ε^(1/3)` and the bound
/// `64·ε^(2/3)·(1 + |analytic|)` are the central-difference truncation and
/// roundoff floor.
#[test]
fn the_explicit_psi_terms_are_the_psi_derivatives_of_the_nll_2695() {
    use crate::custom_family::CustomFamily;

    let (beta_t, beta_thr, beta_ls) = (0.6, 0.35, -0.4);
    let x_thr_psi = array![[0.7], [-0.2], [0.9]];
    let x_ls_psi = array![[-0.5], [0.8], [0.3]];
    let family_at = |psi: f64| {
        let base = survival_exact_newton_test_family();
        let thr = base.x_threshold.to_dense() + psi * &x_thr_psi;
        let ls = base.x_log_sigma.to_dense() + psi * &x_ls_psi;
        SurvivalLocationScaleFamily {
            x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(thr)),
            x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(ls)),
            ..base
        }
    };
    let nll_terms = |psi: f64| {
        let family = family_at(psi);
        let states = survival_exact_newton_test_states(&family, beta_t, beta_thr, beta_ls);
        let (ll, blocks) = family
            .evaluate_log_likelihood_and_block_gradients(&states)
            .expect("value and gradient");
        let gradient =
            Array1::from_iter(blocks.iter().flat_map(|block| block.iter().map(|&g| -g)));
        let hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("joint Hessian")
            .expect("joint Hessian present");
        (-ll, gradient, hessian)
    };

    let family = family_at(0.0);
    let states = survival_exact_newton_test_states(&family, beta_t, beta_thr, beta_ls);
    let dynamic = family.build_dynamic_geometry(&states).expect("dynamic geometry");
    let z_thr = x_thr_psi.column(0).mapv(|x| x * beta_thr);
    let z_ls = x_ls_psi.column(0).mapv(|x| x * beta_ls);
    let direction = SurvivalJointPsiDirection {
        x_t_exit_psi: Some(x_thr_psi.clone()),
        x_t_entry_psi: Some(x_thr_psi.clone()),
        x_t_deriv_psi: None,
        x_ls_exit_psi: Some(x_ls_psi.clone()),
        x_ls_entry_psi: Some(x_ls_psi.clone()),
        x_ls_deriv_psi: None,
        z_t_exit_psi: z_thr.clone(),
        z_t_entry_psi: z_thr,
        z_t_deriv_psi: Array1::zeros(family.n),
        z_ls_exit_psi: z_ls.clone(),
        z_ls_entry_psi: z_ls,
        z_ls_deriv_psi: Array1::zeros(family.n),
        x_t_exit_action: None,
        x_t_entry_action: None,
        x_t_deriv_action: None,
        x_ls_exit_action: None,
        x_ls_entry_action: None,
        x_ls_deriv_action: None,
    };
    let (objective_psi, score_psi, hessian_psi) = survival_ls_joint_psi_first_order_terms(
        &family, &dynamic, &direction, None, true,
    )
    .expect("explicit psi terms");
    let hessian_psi = hessian_psi.expect("dense psi Hessian");

    let h = f64::EPSILON.cbrt();
    let (value_plus, gradient_plus, hessian_plus) = nll_terms(h);
    let (value_minus, gradient_minus, hessian_minus) = nll_terms(-h);
    let bound = |analytic: f64| 64.0 * f64::EPSILON.powf(2.0 / 3.0) * (1.0 + analytic.abs());
    let fd_value = (value_plus - value_minus) / (2.0 * h);
    assert!(
        (objective_psi - fd_value).abs() <= bound(objective_psi),
        "V_psi {objective_psi:.12e} vs central difference {fd_value:.12e}"
    );
    for a in 0..score_psi.len() {
        let fd = (gradient_plus[a] - gradient_minus[a]) / (2.0 * h);
        assert!(
            (score_psi[a] - fd).abs() <= bound(score_psi[a]),
            "g_psi[{a}] {:.12e} vs central difference {fd:.12e}",
            score_psi[a]
        );
        for b in 0..score_psi.len() {
            let fd = (hessian_plus[[a, b]] - hessian_minus[[a, b]]) / (2.0 * h);
            assert!(
                (hessian_psi[[a, b]] - fd).abs() <= bound(hessian_psi[[a, b]]),
                "H_psi[{a}][{b}] {:.12e} vs central difference {fd:.12e}",
                hessian_psi[[a, b]]
            );
        }
    }
    // Non-vacuity: ψ moves the scale, which the time coefficient (index 0) now
    // sees through the scaled time channel, so its ψ score is not zero.
    assert!(
        score_psi[0].abs() > 1.0e6 * bound(0.0),
        "the time coefficient's ψ score must be exercised: {score_psi:?}"
    );
}

/// gam#2695 degree ladder (child module so this file stays under the line gate).
mod knot_ladder_2695;

/// #3090: the direct parametric-AFT step on an indefinite Hessian.
mod absolute_newton_3090;

/// #3185: the direct parametric-AFT backtracking floor and stall.
mod line_search_3185;
