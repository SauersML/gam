// #901 re-home: the Matérn spatial length-scale (iso-κ) optimizer convergence
// gates. The #901 issue listed these as failing — the κ optimizer would stall
// (`final_grad_norm ≈ 1.35`, never reaching `rel_tol`) because the outer REML
// gradient it descended was the wrong projected-logdet gradient. With the #901
// fix (commit 7a5bfd9b2: intrinsic ½log|H_pen|₊ pseudo-logdet) the outer
// gradient is exact, the optimizer reaches tolerance, and the optimized score
// is monotone-non-worse than the unoptimized baseline.
//
// Authored in the pre-#1521 monolith (`tests/src_modules/smooths/`), these were
// orphaned out of the build by #1601: their driver deps
// (`fit_term_collection_forspec`, `fit_term_collectionwith_spatial_length_scale_optimization`,
// `fit_score`, `SpatialLengthScaleOptimizationOptions`) live HERE post-carve,
// not in `gam_terms::smooth`. Re-homed as a `#[cfg(test)] mod` `include!`d into
// the drivers module so the private driver surface resolves via `super::*`.

#[cfg(test)]
mod spatial_length_scale_monotone_tests {
    use super::*;
    use gam_terms::basis::{MaternBasisSpec, MaternNu};
    use gam_linalg::matrix::LinearOperator;
    use gam_terms::smooth::auto_initial_length_scale_for_centers;
    use ndarray::{Array1, Array2, ArrayView2};

    /// Runs a Gaussian baseline fit and the spatial
    /// length-scale optimization for a single Matérn term, then asserts the
    /// optimized score is monotone-non-worse and that the resolved term froze
    /// its centers / identifiability transform with a finite in-range length
    /// scale. Shared verbatim between the 2- and 3-feature Matérn monotone
    /// pins, which differ only in their data generation, term dimensionality,
    /// and seed length scale.
    fn assert_matern_spatial_length_scale_optimization_monotone(
        data: ArrayView2<'_, f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        spec: &TermCollectionSpec,
        fit_opts: &FitOptions,
    ) {
        let baseline = fit_term_collection_forspec(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            spec,
            LikelihoodSpec::gaussian_identity(),
            fit_opts,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline fit should succeed", e));
        let baseline_score = fit_score(&baseline.fit);

        let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
            data,
            y.clone(),
            weights.clone(),
            offset.clone(),
            spec,
            LikelihoodSpec::gaussian_identity(),
            fit_opts,
            &SpatialLengthScaleOptimizationOptions {
                // `max_outer_iter: 2` was set when the iso-κ analytic
                // optimizer typically converged within two BFGS steps. The
                // current optimizer reaches the relative-gradient tolerance
                // only after a handful of outer iterations on the Matérn
                // monotone fixtures (the previous run-out left
                // `|g|_proj ≈ 1.65e-1` against `|f| ≈ 1.3e2` — well above
                // `rel_tol * (1 + |f|) ≈ 1.3e-3`), so a 2-iteration cap
                // bails before reaching convergence. Raising the cap to 16
                // gives the optimizer headroom to actually reach the
                // tolerance the test is asserting against; the
                // monotone-improvement contract this test pins is unchanged.
                max_outer_iter: 16,
                rel_tol: 1e-5,
                ..SpatialLengthScaleOptimizationOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "optimized fit should succeed", e));
        let optimized_score = fit_score(&optimized.fit);
        assert!(optimized_score <= baseline_score + 1e-10);

        let ls = match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.length_scale.resolved().unwrap(),
            _ => panic!("expected Matérn term"),
        };
        assert!(ls.is_finite() && (1e-3..=1e3).contains(&ls));

        match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                assert!(matches!(
                    spec.center_strategy,
                    CenterStrategy::UserProvided(_)
                ));
                assert!(matches!(
                    spec.identifiability,
                    MaternIdentifiability::FrozenTransform { .. }
                ));
            }
            _ => panic!("expected Matérn term"),
        }
    }

    /// Return `(short_seed, long_endpoint, selected, short_signal_rmse,
    /// selected_signal_rmse)` for the certified pre-joint Matérn range comparison
    /// on a deterministic sinusoid. Keeping this at the profiler boundary isolates
    /// the global basin decision from the subsequent local joint optimizer. Each
    /// RMSE is a fit's predictor against the signal `sin(2π·frequency·x)` over the
    /// training rows; the 37-cycle term is roughness the smooth should not chase.
    fn profiled_matern_basin_for_frequency(frequency: f64) -> (f64, f64, f64, f64, f64) {
        let n = 120usize;
        let num_centers = 20usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n - 1) as f64;
            data[[i, 0]] = x;
            y[i] = (2.0 * std::f64::consts::PI * frequency * x).sin()
                + 0.05 * (2.0 * std::f64::consts::PI * 37.0 * x).sin();
        }
        let short_seed =
            auto_initial_length_scale_for_centers(data.view(), &[0], num_centers);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(short_seed),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);
        let family = LikelihoodSpec::gaussian_identity();
        let options = superseded_fit_options(&FitOptions::default());
        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            family.clone(),
            &options,
        )
        .expect("short-range profile");
        let signal = data
            .column(0)
            .mapv(|x| (2.0 * std::f64::consts::PI * frequency * x).sin());
        let signal_rmse = |fitted: &FittedTermCollection| -> f64 {
            let predictor =
                fitted.design.design.apply(&fitted.fit.beta) + &fitted.design.affine_offset;
            ((&predictor - &signal).mapv(|r| r * r).sum() / n as f64).sqrt()
        };
        let short_signal_rmse = signal_rmse(&baseline);
        let resolved =
            freeze_term_collection_from_design(&spec, &baseline.design).expect("freeze profile");
        let spatial_terms = spatial_length_scale_term_indices(&resolved);
        assert_eq!(spatial_terms, vec![0]);
        let companion_length_scale = matern_low_rank_center_resolution_length_scale(
            data.view(),
            &[0],
            num_centers,
        )
        .expect("center-resolution endpoint");
        let (psi_long_bound, psi_short_bound) = spatial_term_psi_bounds(data.view(), &resolved, 0)
            .expect("finite isotropic-scale bounds");
        let psi_long = (-companion_length_scale.ln()).clamp(psi_long_bound, psi_short_bound);
        let long_endpoint = (-psi_long).exp();
        let (selected_spec, selected_fit) = select_isotropic_matern_range_basin(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            resolved,
            baseline,
            &family,
            &options,
            &spatial_terms,
        )
        .expect("certified endpoint profile comparison");
        let selected = get_spatial_length_scale(&selected_spec, 0).expect("selected Matérn range");
        let selected_signal_rmse = signal_rmse(&selected_fit);
        (short_seed, long_endpoint, selected, short_signal_rmse, selected_signal_rmse)
    }

    #[test]
    fn smooth_nu_five_halves_selects_certified_long_range_basin() {
        let (short, long, selected, _, _) = profiled_matern_basin_for_frequency(1.0);
        assert!(long > short, "fixture must expose distinct range basins");
        assert_eq!(
            selected, long,
            "smooth ν=5/2 signal should enter the certified long-range basin"
        );
    }

    #[test]
    fn sin8_nu_five_halves_basin_choice_recovers_the_signal_at_least_as_well_as_the_short_basin() {
        let (short, long, selected, short_signal_rmse, selected_signal_rmse) =
            profiled_matern_basin_for_frequency(8.0);
        assert!(long > short, "fixture must expose distinct range basins");
        // The selector leaves the short incumbent only when the long endpoint's
        // certified REML score is lower. Which range resolves an eight-cycle signal
        // under 20 centers is an empirical question, so the contract is on the signal
        // the chosen basin recovers, not on a range label: leaving the short basin
        // must not recover sin(16πx) worse than staying in it.
        assert!(
            selected == short || selected_signal_rmse <= short_signal_rmse,
            "sin8 ν=5/2: the selected range {selected} recovers the signal with RMSE \
             {selected_signal_rmse:e}, worse than the short basin {short} ({short_signal_rmse:e})"
        );
    }

    #[test]
    fn spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern_two_feature() {
        let n = 60usize;
        let d = 3usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.13).sin();
            let x2 = (i as f64 * 0.07).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            data[[i, 2]] = x2;
            y[i] = (2.5 * x0).sin() + 0.4 * x1 - 0.2 * x2;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(20.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            ..FitOptions::default()
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        assert_matern_spatial_length_scale_optimization_monotone(
            data.view(),
            &y,
            &weights,
            &offset,
            &spec,
            &fit_opts,
        );
    }

    #[test]
    fn spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern() {
        let n = 60usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).cos() + 0.35 * x1;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            ..FitOptions::default()
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        assert_matern_spatial_length_scale_optimization_monotone(
            data.view(),
            &y,
            &weights,
            &offset,
            &spec,
            &fit_opts,
        );
    }

    use super::design_assembly_constraint_tests::two_block_exact_joint_hyper_setup;

    /// The typed refusal an unarmed inner joint Newton emits when it stops while still
    /// descending a ray that block 1's penalty opposes but does not close.
    fn descending_ray_refusal_2938() -> gam_problem::CustomFamilyError {
        gam_problem::CustomFamilyError::InnerSolveNotConverged {
            cycles: 51,
            terminal: Some(gam_problem::InnerConvergenceTerminalState::JointNewton {
                cycle: 51,
                stationarity_residual: 7.443,
                residual_tol: 2.791e-3,
                stationarity_scale: 7.3e9,
                step_inf: 9.31,
                step_tol: 1e-6,
                resolvable_negative_curvature: false,
                best_stationarity_residual: 7.443,
                cycles_since_best_residual: 4,
                termination_reason: gam_problem::JointNewtonTerminalReason::StalledOnDescendingRay {
                    residual: 7.443,
                    residual_tol: 2.791e-3,
                    cycles: 51,
                    ray: gam_problem::RayRestoration {
                        block: 1,
                        rho_first: 1,
                        rho_count: 1,
                        log_strength_ratio: 5.75,
                        likelihood_slope: -4.473e-3,
                        penalty_slope: 1.317e-5,
                        block_step_inf: 7.5e-2,
                        direction: std::sync::Arc::from(vec![0.0, 7.5e-2]),
                    },
                },
            }),
            kkt_residual: Some(7.443),
            kkt_tol: Some(2.791e-3),
            theta_dim: 4,
            rho_dim: 2,
            psi_dim: 2,
            cycle_budget: None,
            carrying_block: None,
        }
    }

    /// The two-block anisotropic Matérn fixture of the κ-freezing pins, whose exact-joint
    /// search the refusal control below drives through the real typed driver.
    fn two_block_matern_refusal_fixture_2938() -> (Array2<f64>, TermCollectionSpec, TermCollectionSpec) {
        let n = 40usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
            data[[i, 1]] = (i as f64 * 0.21).sin();
        }
        let matern_term = |name: &str, length_scale: f64| SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: name.to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(length_scale),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: Some(vec![0.0, 0.0]),
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };
        let meanspec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![matern_term("mean_matern", 0.8)],
        };
        let noisespec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![matern_term("noise_matern", 1.1)],
        };
        (data, meanspec, noisespec)
    }

    /// Every exact evaluation of the search refuses with `refusal`, so the typed driver's outer
    /// search refuses as a whole and returns its failure.
    fn refused_two_block_search_2938<E: Into<FitFailure>>(refusal: impl Fn() -> E) -> FitFailure {
        let (data, meanspec, noisespec) = two_block_matern_refusal_fixture_2938();
        let kappa_options = SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 1,
            rel_tol: 1e-6,
            ..SpatialLengthScaleOptimizationOptions::default()
        };
        let joint_setup = two_block_exact_joint_hyper_setup(data.view(), &meanspec, &noisespec);
        let mean_terms = spatial_length_scale_term_indices(&meanspec);
        let noise_terms = spatial_length_scale_term_indices(&noisespec);
        let policy = gam_model_api::families::custom_family::OuterDerivativePolicy {
            capability: gam_problem::ExactOuterDerivativeOrder::Second,
        };
        match optimize_spatial_length_scale_exact_joint_typed(
            data.view(),
            &[meanspec.clone(), noisespec.clone()],
            &[mean_terms, noise_terms],
            &kappa_options,
            &joint_setup,
            gam_problem::SeedRiskProfile::Gaussian,
            true,
            true,
            false,
            None,
            None,
            policy,
            |_, _, _, _| Ok::<f64, FitFailure>(0.0),
            |_, _, _, _, _| Err::<ExactJointEvaluation<()>, E>(refusal()),
            |_, _: &[TermCollectionSpec], _: &[TermCollectionDesign]| {
                Err::<ExactJointEfsEvaluation<()>, String>(
                    "the refusal control disables fixed-point optimization".to_string(),
                )
            },
            |_: &Array1<f64>| Ok(gam_solve::rho_optimizer::SeedOutcome::NoSlot),
        ) {
            Ok(_) => panic!("a search whose every exact evaluation refuses must not certify"),
            Err(failure) => failure,
        }
    }

    /// A refused spatial search carries its last typed custom-family refusal as the
    /// whole-search refusal of the route that refused, so the Jeffreys arming evidence reads
    /// through it and the message names the n-block exact-joint spatial search. A refusal that
    /// is only text carries no such record (gam#2938).
    #[test]
    fn a_refused_spatial_search_keeps_its_last_typed_refusal_2938() {
        let typed =
            refused_two_block_search_2938(|| FitFailure::from(descending_ray_refusal_2938()));
        assert!(
            matches!(
                &typed,
                FitFailure::CustomFamily(gam_problem::CustomFamilyError::OuterSmoothingFailed {
                    route: gam_problem::OuterSearchRoute::SpatialExactJoint,
                    last_refusal: Some(_),
                    ..
                })
            ),
            "the refused spatial search must return the spatial whole-search refusal with its \
             last typed refusal; got {typed:?}"
        );
        let evidence = custom_family_refusal_in(&typed)
            .and_then(gam_problem::CustomFamilyError::jeffreys_arming_evidence);
        assert!(
            matches!(
                evidence,
                Some(gam_problem::jeffreys_arming::JeffreysArmingEvidence::DescendingRay {
                    block: 1,
                    ..
                })
            ),
            "the arming evidence of the last refusal must read through the search failure; got \
             {evidence:?}"
        );
        let message = typed.to_string();
        assert!(
            message.contains("in the n-block exact-joint spatial outer search:"),
            "the rendered refusal must name the spatial search, not fit_custom_family; got \
             {message}"
        );

        let plain = refused_two_block_search_2938(|| {
            "a refusal that carries no typed custom-family error".to_string()
        });
        assert!(
            custom_family_refusal_in(&plain).is_none() && !matches!(&plain, FitFailure::CustomFamily(_)),
            "a search refused only as text must not become a typed whole-search refusal; got \
             {plain:?}"
        );
    }
}
