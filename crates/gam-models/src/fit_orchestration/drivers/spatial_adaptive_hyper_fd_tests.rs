// End-to-end finite-difference oracles for the custom-family **adaptive-ψ**
// projected-logdet REML hypergradient AND outer-Hessian on a real spatial
// `SpatialAdaptiveExactFamily` — the second half of the #901 gate.
//
// `exact_spatial_adaptive_joint_hypergradient_matches_finite_difference`
// differentiates the production custom-family joint REML cost
// (`evaluate_custom_family_joint_hyper`) by central differences on all six ψ
// axes (log λ mass/tension/curvature + log ε mass/grad/curvature) and compares
// against the analytic envelope hypergradient dF/dρ = J_a + ½tr(H⁻¹ Ḣ_a) and
// its outer-Hessian. This is the exact reproduction the #901 engine fix targets:
// for a penalty-ψ family (`joint_jeffreys_information_depends_on_psi() == false`)
// the explicit-ψ Firth perturbation/curvature drift terms are spurious because
// ∂_ψ(data info) ≡ 0 (the design X is fixed), and the pre-fix engine folded
// θ-dependent Charbonnier penalty curvature into the Firth term Φ, biasing the
// gradient by the Charbonnier group dimension (mass 1 / tension 2 / curv 4).
//
// `adaptive_hyper_derivative_dispatch_matches_reference` (#426) pins the
// unified `adaptive_block_eval` / `adaptive_block_drift_eval` / dispatch surface
// bit-for-bit against an independent hand-rolled reference.
//
// These fixtures were authored in the pre-#1521 monolith under
// `tests/src_modules/smooths/smooth_adaptive_bounded_duchon_tests.rs`. The
// #1521 crate carve moved their private dependencies (`SpatialAdaptiveExactFamily`,
// `build_spatial_adaptive_joint_hyper_scaffold`'s deps, the scalar/grouped
// operator helpers, `fit_term_collection_forspec`,
// `extract_spatial_operator_runtime_caches`, `compute_initial_epsilons`,
// `build_spatial_adaptive_hyperspecs`) DOWN into the gam-models
// `fit_orchestration::drivers` module, but #1601 commented their `include!` out
// of `gam_terms::smooth::tests` "for relocation" and the relocation never
// happened — so these two #901 gates compiled into NO binary. They are
// re-homed here alongside their already-relocated iso-κ / length-scale siblings,
// where every private driver symbol is in scope via `use super::*` (the
// `design_construction.rs` + `spatial_optimization.rs` files are `include!`d
// flat into one module namespace). The only cross-crate paths the orphaned
// monolith reached through `crate::` (`crate::custom_family::*`,
// `crate::solver::estimate::reml::reml_outer_engine::*`) are rewritten to their
// carved homes `gam_custom_family::*` / `gam_solve::estimate::reml::*`.
#[cfg(test)]
mod spatial_adaptive_hyper_fd_tests {
    use super::*;
    use gam_solve::model_types::AdaptiveRegularizationOptions;
    // `CenterStrategy` and `MaternIdentifiability` already arrive through
    // `super::*` (the parent drivers module imports them from
    // `gam_terms::basis`), so re-importing them would collide (E0252); only the
    // Duchon/Matern spec types absent from the parent surface are pulled in.
    use gam_terms::basis::{
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec,
        MaternNu, OneDimensionalBoundary, SpatialIdentifiability,
    };
    use ndarray::array;

/// Builds the shared spatial-adaptive joint-hyper evaluation scaffolding
/// (hyperspecs, the zero-ψ derivative blocks, the `base_family` with empty
/// adaptive params, and the single `eta` block spec) from a Gaussian
/// baseline fit. Shared verbatim across the FD-gradient and
/// gradient-lambda-profile pins; the per-test outer-loop cycle counts and
/// θ probing stay at each call site.
fn build_spatial_adaptive_joint_hyper_scaffold(
    baseline: &FittedTermCollection,
    runtime_caches: &[SpatialOperatorRuntimeCache],
    y: &Array1<f64>,
    n: usize,
) -> (
    SpatialAdaptiveExactFamily,
    ParameterBlockSpec,
    Vec<Vec<CustomFamilyBlockPsiDerivative>>,
) {
    let hyperspecs = build_spatial_adaptive_hyperspecs(runtime_caches.len());
    let zero_psi_op: std::sync::Arc<dyn gam_custom_family::CustomFamilyPsiDerivativeOperator> =
        std::sync::Arc::new(gam_custom_family::ZeroPsiDerivativeOperator::new(
            baseline.design.design.nrows(),
            baseline.design.design.ncols(),
        ));
    let derivative_blocks = vec![
        hyperspecs
            .iter()
            .map(|_| CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: Array2::<f64>::zeros((0, 0)),
                s_psi: Array2::<f64>::zeros((0, 0)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: None,
                s_psi_psi: None,
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: Some(std::sync::Arc::clone(&zero_psi_op)),
                implicit_axis: 0,
                implicit_group_id: None,
            })
            .collect::<Vec<_>>(),
    ];
    let base_family = SpatialAdaptiveExactFamily {
        family: LikelihoodSpec::gaussian_identity(),
        latent_cloglog_state: None,
        mixture_link_state: None,
        sas_link_state: None,
        y: Arc::new(y.clone()),
        weights: Arc::new(Array1::ones(n)),
        design: baseline.design.design.to_dense_arc(),
        offset: Arc::new(Array1::zeros(n)),
        linear_constraints: baseline.design.linear_constraints.clone(),
        runtime_caches: Arc::new(runtime_caches.to_vec()),
        adaptive_params: Vec::new(),
        fixed_quadratichessian: Arc::new(Array2::<f64>::zeros((
            baseline.design.design.ncols(),
            baseline.design.design.ncols(),
        ))),
        hyperspecs: Arc::new(hyperspecs),
        exact_eval_cache: Arc::new(Mutex::new(None)),
    };
    let blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: baseline.design.design.clone(),
        offset: Array1::zeros(n),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(baseline.fit.beta.clone()),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    (base_family, blockspec, derivative_blocks)
}

#[test]
fn exact_spatial_adaptive_joint_hypergradient_matches_finite_difference() {
    let n = 36usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (0.31 * i as f64).sin();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        y[i] = (4.0 * x0).sin() + 0.35 * x1 + 0.2 * ((x0 - 0.55) * 18.0).tanh();
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: 0.6,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                    nullspace_shrinkage_survived: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let baseline = fit_term_collection_forspec(
        data.view(),
        y.view(),
        Array1::ones(n).view(),
        Array1::zeros(n).view(),
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &FitOptions {
            max_iter: 30,
            penalty_shrinkage_floor: None,
            ..FitOptions::default()
        },
    )
    .expect("baseline fit");
    let runtime_caches =
        extract_spatial_operator_runtime_caches(&spec, &baseline.design).expect("runtime caches");
    assert_eq!(runtime_caches.len(), 1);

    let adaptive_opts = AdaptiveRegularizationOptions::default();
    let (eps_0_init, eps_g_init, eps_c_init) = compute_initial_epsilons(
        &baseline.fit.beta,
        &runtime_caches,
        adaptive_opts.min_epsilon,
    )
    .expect("initial epsilons");
    let (base_family, blockspec, derivative_blocks) =
        build_spatial_adaptive_joint_hyper_scaffold(&baseline, &runtime_caches, &y, n);
    // The analytic hypergradient is the *envelope* derivative dF/dρ = J_a +
    // ½tr(H⁻¹ Ḣ_a), valid only at a converged inner mode (∂F/∂β = 0). The
    // central-difference reference re-solves the inner mode at each θ±h, so a
    // loosely-converged inner solve makes the FD probe and the analytic
    // envelope evaluate two different functions and disagree on the
    // tension-dominated direction (the stiffest β-mode of the 2-D Matérn,
    // hence the largest residual ∂F/∂β at a partial solve). Drive the inner
    // solve to f64-grade stationarity so the FD reference is exact.
    let outer_opts = BlockwiseFitOptions {
        inner_max_cycles: 200,
        inner_tol: 1e-10,
        outer_max_iter: 30,
        outer_tol: 1e-10,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };

    let evaluate_theta = |theta: &Array1<f64>, need_hessian: bool| {
        let family = base_family.with_adaptive_params(
            vec![SpatialAdaptiveTermHyperParams {
                lambda: [theta[0].exp(), theta[1].exp(), theta[2].exp()],
                epsilon: [theta[3].exp(), theta[4].exp(), theta[5].exp()],
            }],
            Arc::new(Array2::<f64>::zeros((
                baseline.design.design.ncols(),
                baseline.design.design.ncols(),
            ))),
        );
        evaluate_custom_family_joint_hyper(
            &family,
            std::slice::from_ref(&blockspec),
            &outer_opts,
            &Array1::zeros(0),
            &derivative_blocks,
            None,
            if need_hessian {
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian
            } else {
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient
            },
        )
        .expect("joint hyper eval")
    };

    let theta = array![
        baseline.fit.lambdas[runtime_caches[0].mass_penalty_global_idx]
            .max(1e-6)
            .ln(),
        baseline.fit.lambdas[runtime_caches[0].tension_penalty_global_idx]
            .max(1e-6)
            .ln(),
        baseline.fit.lambdas[runtime_caches[0].stiffness_penalty_global_idx]
            .max(1e-6)
            .ln(),
        eps_0_init.max(1e-6).ln(),
        eps_g_init.max(1e-6).ln(),
        eps_c_init.max(1e-6).ln(),
    ];
    let analytic = evaluate_theta(&theta, true);
    assert_eq!(analytic.gradient.len(), theta.len());
    assert!(
        analytic.outer_hessian.is_analytic(),
        "adaptive joint hyper evaluation must expose exact Hessian curvature"
    );
    assert_eq!(
        analytic.outer_hessian.dim(),
        Some(theta.len()),
        "adaptive joint hyper Hessian must span all lambda/epsilon coordinates"
    );
    let analytic_hessian = analytic
        .outer_hessian
        .clone()
        .materialize_dense()
        .expect("adaptive joint hyper Hessian should materialize")
        .expect("adaptive joint hyper Hessian should be present");
    let h = 1e-5;
    for j in 0..theta.len() {
        let mut plus = theta.clone();
        plus[j] += h;
        let mut minus = theta.clone();
        minus[j] -= h;
        let fd = (evaluate_theta(&plus, false).objective - evaluate_theta(&minus, false).objective)
            / (2.0 * h);
        assert!(
            (analytic.gradient[j] - fd).abs() < 5e-3 * (1.0 + fd.abs()),
            "adaptive joint hypergradient mismatch at {j}: analytic={}, fd={fd}",
            analytic.gradient[j]
        );
        let grad_fd = (evaluate_theta(&plus, false).gradient
            - evaluate_theta(&minus, false).gradient)
            / (2.0 * h);
        for i in 0..theta.len() {
            assert!(
                (analytic_hessian[[i, j]] - grad_fd[i]).abs() < 5e-2 * (1.0 + grad_fd[i].abs()),
                "adaptive joint hyper-Hessian mismatch at ({i},{j}): analytic={}, fd={}",
                analytic_hessian[[i, j]],
                grad_fd[i]
            );
        }
    }
}

/// Parity test for the unified adaptive hyper-derivative dispatch (issue
/// #426). The unified `adaptive_block_eval` / `adaptive_block_drift_eval`
/// engines must reproduce, bit-for-bit, the per-(component, derivative-kind)
/// pieces that the previous hand-rolled `adaptive_block_*` /
/// `adaptive_block_*_drift` functions assembled. The right-hand reference
/// below is the same closed-form composition those duplicated functions
/// performed: pick the matching state accessor, run it through the matching
/// scalar / grouped operator with the component weight, and embed into the
/// global coordinate range. Equality is exact (both sides call the identical
/// accessors and operators), so the bound is a true `==`.
#[test]
fn adaptive_hyper_derivative_dispatch_matches_reference() {
    let n = 40usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = x;
        y[i] =
            0.3 * (2.0 * std::f64::consts::PI * x).sin() + 1.1 / (1.0 + (-(x - 0.5) / 0.02).exp());
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 15 },
                    length_scale: Some(0.9),
                    power: 2.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    // Same reason as
                    // `exact_spatial_adaptive_1dobjective_profile_has_finite_gradient_lambda_surface`:
                    // the adaptive overlay requires an EXPLICIT Stiffness
                    // penalty for `extract_spatial_operator_runtime_caches`
                    // to surface a cache. The default Duchon spec disables
                    // Stiffness, so pin `all_active()` here too.
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let baseline = fit_term_collection_forspec(
        data.view(),
        y.view(),
        Array1::ones(n).view(),
        Array1::zeros(n).view(),
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &FitOptions {
            max_iter: 20,
            penalty_shrinkage_floor: None,
            ..FitOptions::default()
        },
    )
    .expect("baseline fit");
    let runtime_caches =
        extract_spatial_operator_runtime_caches(&spec, &baseline.design).expect("runtime caches");
    assert_eq!(runtime_caches.len(), 1);
    let (eps_0, eps_g, eps_c) = compute_initial_epsilons(&baseline.fit.beta, &runtime_caches, 1e-8)
        .expect("initial epsilons");
    let hyperspecs = build_spatial_adaptive_hyperspecs(runtime_caches.len());
    let p = baseline.design.design.ncols();

    let family = SpatialAdaptiveExactFamily {
        family: LikelihoodSpec::gaussian_identity(),
        latent_cloglog_state: None,
        mixture_link_state: None,
        sas_link_state: None,
        y: Arc::new(y.clone()),
        weights: Arc::new(Array1::ones(n)),
        design: baseline.design.design.to_dense_arc(),
        offset: Arc::new(Array1::zeros(n)),
        linear_constraints: baseline.design.linear_constraints.clone(),
        runtime_caches: Arc::new(runtime_caches.clone()),
        adaptive_params: vec![SpatialAdaptiveTermHyperParams {
            lambda: [0.7, 1.3, 0.4],
            epsilon: [eps_0, eps_g, eps_c],
        }],
        fixed_quadratichessian: Arc::new(Array2::<f64>::zeros((p, p))),
        hyperspecs: Arc::new(hyperspecs),
        exact_eval_cache: Arc::new(Mutex::new(None)),
    };

    let beta = baseline.fit.beta.clone();
    let eval = family.exact_evaluation(&beta).expect("exact evaluation");
    let cache = &family.runtime_caches[0];
    let params = &family.adaptive_params[0];
    let state = &eval.adaptive_states[0];
    let range = cache.coeff_global_range.clone();

    // Independent reference assembly for one (component, kind) parts triple.
    let reference_parts = |component: AdaptiveComponent,
                           kind: HyperDerivativeKind|
     -> (f64, Array1<f64>, Array2<f64>) {
        let (objective, grad_local, hess_local) = match component {
            AdaptiveComponent::Magnitude => {
                let lambda = params.lambda[0];
                let mag = &state.magnitude;
                let (obj, gc, hd) = match kind {
                    HyperDerivativeKind::Rho => (
                        mag.penalty_value(),
                        mag.betagradient_coeff(),
                        mag.betahessian_diag(),
                    ),
                    HyperDerivativeKind::LogEpsilonFirst => (
                        mag.log_epsilon_gradient_terms().sum(),
                        mag.log_epsilon_betagradient_coeff(),
                        mag.log_epsilon_betahessian_diag(),
                    ),
                    HyperDerivativeKind::LogEpsilonSecond => (
                        mag.log_epsilon_hessian_terms().sum(),
                        mag.log_epsilon_beta_mixed_second_coeff(),
                        mag.log_epsilon_betahessian_second_diag(),
                    ),
                };
                (
                    lambda * obj,
                    lambda * scalar_operatorgradient(&cache.d0, &gc),
                    lambda * scalar_operatorhessian(&cache.d0, &hd),
                )
            }
            AdaptiveComponent::Gradient => {
                let lambda = params.lambda[1];
                let grad = &state.gradient;
                let (obj, gb, hb) = match kind {
                    HyperDerivativeKind::Rho => (
                        grad.penalty_value(),
                        grad.betagradient_blocks(),
                        grad.betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonFirst => (
                        grad.log_epsilon_gradient_terms().sum(),
                        grad.log_epsilon_betagradient_blocks(),
                        grad.log_epsilon_betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonSecond => (
                        grad.log_epsilon_hessian_terms().sum(),
                        grad.log_epsilon_beta_mixed_second_blocks(),
                        grad.log_epsilon_betahessian_second_blocks(),
                    ),
                };
                (
                    lambda * obj,
                    lambda
                        * grouped_operatorgradient(&cache.d1, cache.dimension, &gb)
                            .expect("grouped gradient"),
                    lambda
                        * grouped_operatorhessian(&cache.d1, cache.dimension, &hb)
                            .expect("grouped hessian"),
                )
            }
            AdaptiveComponent::Curvature => {
                let lambda = params.lambda[2];
                let group = cache.dimension * cache.dimension;
                let curv = &state.curvature;
                let (obj, gb, hb) = match kind {
                    HyperDerivativeKind::Rho => (
                        curv.penalty_value(),
                        curv.betagradient_blocks(),
                        curv.betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonFirst => (
                        curv.log_epsilon_gradient_terms().sum(),
                        curv.log_epsilon_betagradient_blocks(),
                        curv.log_epsilon_betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonSecond => (
                        curv.log_epsilon_hessian_terms().sum(),
                        curv.log_epsilon_beta_mixed_second_blocks(),
                        curv.log_epsilon_betahessian_second_blocks(),
                    ),
                };
                (
                    lambda * obj,
                    lambda
                        * grouped_operatorgradient(&cache.d2, group, &gb)
                            .expect("grouped gradient"),
                    lambda
                        * grouped_operatorhessian(&cache.d2, group, &hb).expect("grouped hessian"),
                )
            }
        };
        let mut grad = Array1::<f64>::zeros(p);
        grad.slice_mut(s![range.clone()]).assign(&grad_local);
        let mut hess = Array2::<f64>::zeros((p, p));
        hess.slice_mut(s![range.clone(), range.clone()])
            .assign(&hess_local);
        (objective, grad, hess)
    };

    let components = [
        AdaptiveComponent::Magnitude,
        AdaptiveComponent::Gradient,
        AdaptiveComponent::Curvature,
    ];
    let kinds = [
        HyperDerivativeKind::Rho,
        HyperDerivativeKind::LogEpsilonFirst,
        HyperDerivativeKind::LogEpsilonSecond,
    ];

    for &component in &components {
        for &kind in &kinds {
            let (obj_new, grad_new, hess_new) = family
                .adaptive_block_eval(&eval, 0, component, kind)
                .expect("unified block eval");
            let (obj_ref, grad_ref, hess_ref) = reference_parts(component, kind);
            assert_eq!(
                obj_new, obj_ref,
                "objective mismatch for {component:?}/{kind:?}"
            );
            assert_eq!(
                grad_new, grad_ref,
                "gradient mismatch for {component:?}/{kind:?}"
            );
            assert_eq!(
                hess_new, hess_ref,
                "hessian mismatch for {component:?}/{kind:?}"
            );
        }
    }

    // Directional-drift parity: independent reference per (component, drift).
    let direction = {
        let mut d = Array1::<f64>::zeros(p);
        for (i, v) in d.iter_mut().enumerate() {
            *v = 0.05 + 0.01 * (i as f64);
        }
        d
    };
    let reference_drift = |component: AdaptiveComponent, drift: HyperDriftKind| -> Array2<f64> {
        let direction_local = direction.slice(s![range.clone()]);
        let local = match component {
            AdaptiveComponent::Magnitude => {
                let d0_u = cache.d0.dot(&direction_local);
                let mag = &state.magnitude;
                let diag = match drift {
                    HyperDriftKind::Rho => mag.directionalhessian_diag(&d0_u),
                    HyperDriftKind::LogEpsilon => {
                        mag.log_epsilon_betahessian_directional_diag(&d0_u)
                    }
                };
                params.lambda[0] * scalar_operatorhessian(&cache.d0, &diag)
            }
            AdaptiveComponent::Gradient => {
                let d1_u = cache.d1.dot(&direction_local);
                let blocks_in = collocationgradient_blocks(&d1_u, cache.dimension)
                    .expect("collocation gradient blocks");
                let grad = &state.gradient;
                let blocks = match drift {
                    HyperDriftKind::Rho => grad.directionalhessian_blocks(&blocks_in),
                    HyperDriftKind::LogEpsilon => {
                        grad.log_epsilon_betahessian_directional_blocks(&blocks_in)
                    }
                };
                params.lambda[1]
                    * grouped_operatorhessian(&cache.d1, cache.dimension, &blocks)
                        .expect("grouped hessian")
            }
            AdaptiveComponent::Curvature => {
                let group = cache.dimension * cache.dimension;
                let d2_u = cache.d2.dot(&direction_local);
                let blocks_in = collocationhessian_blocks(&d2_u, cache.dimension)
                    .expect("collocation hessian blocks");
                let curv = &state.curvature;
                let blocks = match drift {
                    HyperDriftKind::Rho => curv.directionalhessian_blocks(&blocks_in),
                    HyperDriftKind::LogEpsilon => {
                        curv.log_epsilon_betahessian_directional_blocks(&blocks_in)
                    }
                };
                params.lambda[2]
                    * grouped_operatorhessian(&cache.d2, group, &blocks).expect("grouped hessian")
            }
        };
        let mut out = Array2::<f64>::zeros((p, p));
        out.slice_mut(s![range.clone(), range.clone()])
            .assign(&local);
        out
    };

    for &component in &components {
        for &drift in &[HyperDriftKind::Rho, HyperDriftKind::LogEpsilon] {
            let drift_new = family
                .adaptive_block_drift_eval(&eval, 0, component, drift, &direction)
                .expect("unified block drift eval");
            let drift_ref = reference_drift(component, drift);
            assert_eq!(
                drift_new, drift_ref,
                "drift mismatch for {component:?}/{drift:?}"
            );
        }
    }

    // Dispatch-surface parity: the public entry points must route to the
    // same engine output. `adaptive_hyper_parts` on a per-term `log lambda`
    // spec equals the `Rho` block eval; on a shared `log epsilon` spec it
    // equals the summed `LogEpsilonFirst` blocks (single cache ⇒ the shared
    // sum is the single block's first-order `log epsilon` eval).
    let check_dispatch_parity = |specs: [(SpatialAdaptiveHyperKind, AdaptiveComponent); 3],
                                 deriv_kind: HyperDerivativeKind| {
        for (kind, component) in specs {
            let hyper = SpatialAdaptiveHyperSpec {
                cache_index: 0,
                kind,
            };
            let (obj_disp, grad_disp, hess_disp) = family
                .adaptive_hyper_parts(&eval, hyper)
                .expect("hyper parts");
            let (obj_ref, grad_ref, hess_ref) = reference_parts(component, deriv_kind);
            assert_eq!(obj_disp, obj_ref);
            assert_eq!(grad_disp, grad_ref);
            assert_eq!(hess_disp, hess_ref);
        }
    };

    check_dispatch_parity(
        [
            (
                SpatialAdaptiveHyperKind::LogLambdaMagnitude,
                AdaptiveComponent::Magnitude,
            ),
            (
                SpatialAdaptiveHyperKind::LogLambdaGradient,
                AdaptiveComponent::Gradient,
            ),
            (
                SpatialAdaptiveHyperKind::LogLambdaCurvature,
                AdaptiveComponent::Curvature,
            ),
        ],
        HyperDerivativeKind::Rho,
    );

    check_dispatch_parity(
        [
            (
                SpatialAdaptiveHyperKind::LogEpsilonMagnitude,
                AdaptiveComponent::Magnitude,
            ),
            (
                SpatialAdaptiveHyperKind::LogEpsilonGradient,
                AdaptiveComponent::Gradient,
            ),
            (
                SpatialAdaptiveHyperKind::LogEpsilonCurvature,
                AdaptiveComponent::Curvature,
            ),
        ],
        HyperDerivativeKind::LogEpsilonFirst,
    );
}

}
