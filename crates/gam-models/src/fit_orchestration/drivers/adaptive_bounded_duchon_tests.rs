// #1601 relocation debt — re-homed from the pre-#1521 monolith fixture
// `tests/src_modules/smooths/smooth_adaptive_bounded_duchon_tests.rs`. This is
// the LAST of the three smooth test files #1601 (commit 28bab3753) commented out
// of `gam_terms::smooth::tests` and parked "for relocation" — a relocation that
// never happened, leaving these 48 adaptive / bounded / pure-Duchon / Charbonnier
// regression guards silently dead (`tests/src_modules/` is `mod`'d into no test
// binary). They belong HERE: their private driver deps
// (`build_term_collection_design`, `freeze_term_collection_from_design`,
// `build_term_collection_designs_and_freeze_joint`, the adaptive-overlay / SAS
// link state / joint-hyper FD closures) live in this `drivers` module post-carve,
// and the cross-crate `crate::` paths the fixture used are rewritten to their
// carved homes (`gam_solve::`, `gam_terms::`, `gam_problem::`, `gam_linalg::`,
// `gam_custom_family::`). Self-contained `#[cfg(test)] mod`, so it adds nothing
// to the non-test build. Companion of `design_assembly_constraint_tests.rs` and
// `matern_nfree_rekey_topology_tests.rs`.
#[cfg(test)]
mod adaptive_bounded_duchon_tests {
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    use super::*;
    // Basis spec types this fixture builds adaptive/bounded designs from.
    // `CenterStrategy` and `MaternIdentifiability` already arrive via `super::*`
    // (the drivers' explicit `gam_terms::basis` import), so re-listing them would
    // collide (E0252); every other name is pulled in explicitly here.
    use gam_terms::basis::{
        BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec,
        MaternNu, OneDimensionalBoundary, SpatialIdentifiability,
    };
    // The `AdaptiveRegularizationOptions` knob the bounded/adaptive fits set lives
    // in gam-solve's model_types; `super::*` does not re-export it into this scope.
    use gam_solve::estimate::AdaptiveRegularizationOptions;
    // The three two-block exact-joint helpers the pre-#1521 monolith shared across
    // all three smooth fixtures (`run_two_block_exact_joint_optimize`,
    // `two_block_exact_joint_hyper_setup`, `assert_term_collection_designs_match`)
    // now live in the sibling `design_assembly_constraint_tests` module, hoisted to
    // `pub(super)` there so both re-homed fixtures share the single definition
    // through this `drivers` parent scope instead of duplicating ~200 lines.
    use super::design_assembly_constraint_tests::{
        assert_term_collection_designs_match, run_two_block_exact_joint_optimize,
        two_block_exact_joint_hyper_setup,
    };
    use ndarray::array;

    #[test]
    fn spatial_penalty_ranges_follow_realized_global_layout_2287() {
        let data = array![
            [1.0, 0.0, 0.00, 0.57],
            [2.0, 1.0, 0.14, 0.00],
            [3.0, 0.0, 0.29, 0.86],
            [4.0, 1.0, 0.43, 0.29],
            [5.0, 0.0, 0.57, 1.00],
            [6.0, 1.0, 0.71, 0.43],
            [7.0, 0.0, 0.86, 0.14],
            [8.0, 1.0, 1.00, 0.71],
        ];
        let smooth = |name: &str, feature_col: usize| SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 4,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };
        let spec = TermCollectionSpec {
            // Exactly one function-space ridge is emitted for this term.
            linear_terms: vec![LinearTermSpec {
                name: "linear".to_string(),
                feature_col: 0,
                feature_cols: vec![0],
                categorical_levels: vec![],
                double_penalty: true,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            }],
            // The non-empty coefficient range is intentionally unpenalized and
            // therefore contributes no global penalty block.
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "unpenalized_group".to_string(),
                feature_col: 1,
                drop_first_level: false,
                penalized: false,
                frozen_levels: Some(vec![0, 1]),
                lenient_unseen: true,
            }],
            // Distinct feature ownership is essential here. Two copies of the
            // same smooth are deliberately collapsed by global hierarchical
            // identifiability, in which case the second term correctly owns no
            // realized penalty block and cannot test a two-term layout.
            smooth_terms: vec![smooth("first_smooth", 2), smooth("second_smooth", 3)],
        };
        let design = build_term_collection_design(data.view(), &spec).expect("mixed design");

        assert_eq!(design.leading_penalty_blocks_before_smooth(), 1);
        let first = design
            .smooth_term_penalty_range(0)
            .expect("consistent layout")
            .expect("penalized first smooth");
        let second = design
            .smooth_term_penalty_range(1)
            .expect("consistent layout")
            .expect("penalized second smooth");
        assert_eq!(first.start, 1);
        assert_eq!(second.start, first.end);
        assert_eq!(
            design.penaltyinfo[first.start].termname.as_deref(),
            Some("first_smooth")
        );
        assert_eq!(
            design.penaltyinfo[second.start].termname.as_deref(),
            Some("second_smooth")
        );
    }

    #[test]
    fn pure_duchon_aniso_penalties_stay_symmetric_through_freeze_and_cache() {
        fn max_asymmetry(matrix: &Array2<f64>) -> f64 {
            let n = matrix.nrows().min(matrix.ncols());
            let mut max_asym = 0.0_f64;
            for i in 0..n {
                for j in 0..i {
                    max_asym = max_asym.max((matrix[[i, j]] - matrix[[j, i]]).abs());
                }
            }
            max_asym
        }

        fn assert_design_penalties_symmetric(label: &str, design: &TermCollectionDesign) {
            for (penalty_idx, penalty) in design.penalties.iter().enumerate() {
                let max_asym = max_asymmetry(&penalty.local);
                assert!(
                    max_asym <= 1e-10,
                    "{label} penalty {penalty_idx} asymmetry too large: {max_asym:.3e}"
                );
            }
        }

        fn assert_reparam_penalty_symmetric(label: &str, design: &TermCollectionDesign) {
            let p_total = design.design.ncols();
            let penalty_specs = design
                .penalties
                .iter()
                .map(|penalty| gam_solve::estimate::PenaltySpec::Dense(penalty.to_global(p_total)))
                .collect::<Vec<_>>();
            let (canonical_penalties, _) = gam_terms::construction::canonicalize_penalty_specs(
                &penalty_specs,
                &design.nullspace_dims,
                p_total,
                label,
            )
            .expect("canonicalize penalties");
            let invariant = gam_terms::construction::precompute_reparam_invariant_from_canonical(
                &canonical_penalties,
                p_total,
            )
            .expect("reparam invariant");
            let lambdas = vec![1.0; canonical_penalties.len()];
            let reparam = gam_terms::construction::stable_reparameterizationwith_invariant(
                &canonical_penalties,
                &lambdas,
                p_total,
                &invariant,
                None,
            )
            .expect("stable reparameterization");
            let max_asym = max_asymmetry(&reparam.s_transformed);
            assert!(
                max_asym <= 1e-10,
                "{label} transformed penalty asymmetry too large: {max_asym:.3e}"
            );
        }

        let data = array![
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.4],
            [0.4, 0.3, 0.1],
            [0.6, 0.5, 0.7],
            [0.8, 0.7, 0.3],
            [1.0, 0.9, 0.8],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "pure_duchon".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1, 2],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                        length_scale: None,
                        power: 1.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: Some(vec![0.0, 0.0, 0.0]),
                        operator_penalties: DuchonOperatorPenaltySpec::default(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };

        let base_design = build_term_collection_design(data.view(), &spec).expect("base design");
        assert_design_penalties_symmetric("base", &base_design);
        assert_reparam_penalty_symmetric("base", &base_design);

        let frozen = freeze_term_collection_from_design(&spec, &base_design).expect("freeze spec");
        let frozen_design =
            build_term_collection_design(data.view(), &frozen).expect("frozen rebuild");
        assert_design_penalties_symmetric("frozen", &frozen_design);
        assert_reparam_penalty_symmetric("frozen", &frozen_design);

        // Design B: a pure Duchon enrolls no outer ψ axis (η is a fixed,
        // geometry-derived basis parameter), so the single-block exact-joint
        // cache for this term is ρ-only. The penalties must stay symmetric
        // through that cache exactly as they do through the base build and the
        // freeze/rebuild above.
        let spatial_terms = spatial_length_scale_term_indices(&frozen);
        assert!(
            spatial_terms.is_empty(),
            "pure Duchon enrolls no outer κ/ψ axis"
        );
        let rho_dim = frozen_design.penalties.len();
        let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
        let theta = Array1::<f64>::zeros(rho_dim);

        let mut cache = SingleBlockExactJointDesignCache::new(
            data.view(),
            frozen.clone(),
            frozen_design.clone(),
            spatial_terms,
            rho_dim,
            dims_per_term,
        )
        .expect("single-block cache");
        cache.ensure_theta(&theta).expect("updated theta");
        assert_design_penalties_symmetric("cache", cache.design());
        assert_reparam_penalty_symmetric("cache", cache.design());
    }

    #[test]
    fn single_block_no_spatial_fast_path_returns_fully_frozen_spec() {
        let n = 48usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (i % 4) as f64;
            y[i] = 0.5 + 1.5 * t;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "grp".to_string(),
                feature_col: 1,
                drop_first_level: false,
                penalized: true,
                frozen_levels: None,
                lenient_unseen: true,
            }],
            smooth_terms: vec![SmoothTermSpec {
                name: "ps".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 4,
                        },
                        double_penalty: true,
                        identifiability: BSplineIdentifiability::None,
                        boundary: OneDimensionalBoundary::Open,
                        boundary_conditions: gam_terms::basis::BSplineBoundaryConditions::default(),
                    },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            penalty_shrinkage_floor: None,
            ..FitOptions::default()
        };

        let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
            data.view(),
            y,
            Array1::ones(n),
            Array1::zeros(n),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &fit_opts,
            &SpatialLengthScaleOptimizationOptions::default(),
        )
        .expect("single-block no-spatial fit should succeed");

        fitted
            .resolvedspec
            .validate_frozen("resolvedspec")
            .expect("single-block no-spatial fast path should fully freeze specs");
        match &fitted.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::BSpline1D { spec, .. } => {
                assert!(matches!(spec.knotspec, BSplineKnotSpec::Provided(_)));
            }
            _ => panic!("expected P-spline term"),
        }
        assert!(
            fitted.resolvedspec.random_effect_terms[0]
                .frozen_levels
                .is_some(),
            "random-effect levels should be frozen in single-block no-spatial fast path"
        );
    }

    #[test]
    fn exact_joint_two_block_spatial_length_scale_freezes_duchon_centers() {
        let n = 40usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.19).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
        }

        let duchon_term = |name: &str, length_scale: f64| SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: Some(length_scale),
                    power: 3.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };

        let meanspec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![duchon_term("mean_duchon", 0.8)],
        };
        let noisespec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![duchon_term("noise_duchon", 1.1)],
        };

        let solved = run_two_block_exact_joint_optimize(
            data.view(),
            &meanspec,
            &noisespec,
            "exact joint two-block spatial length-scale optimization should succeed",
        );

        for resolved in [&solved.resolved_specs[0], &solved.resolved_specs[1]] {
            match &resolved.smooth_terms[0].basis {
                SmoothBasisSpec::Duchon { spec, .. } => {
                    assert!(matches!(
                        spec.center_strategy,
                        CenterStrategy::UserProvided(_)
                    ));
                    assert!(matches!(
                        spec.identifiability,
                        SpatialIdentifiability::FrozenTransform { .. }
                    ));
                }
                _ => panic!("expected Duchon term"),
            }
        }
    }

    #[test]
    fn joint_build_and_cache_rebuild_frozen_pure_duchon_blocks() {
        let n = 72usize;
        let d = 5usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (0.17 * i as f64).sin();
            data[[i, 2]] = (0.11 * i as f64).cos();
            data[[i, 3]] = ((i % 7) as f64) / 6.0;
            data[[i, 4]] = t * (1.0 - t);
        }

        let pure_duchon_term = |name: &str| SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                    length_scale: None,
                    power: 2.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: Some(vec![0.0; d]),
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };

        let meanspec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![pure_duchon_term("mean_pure_duchon")],
        };
        let noisespec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![pure_duchon_term("noise_pure_duchon")],
        };

        let (boot_designs, frozen_specs) = build_term_collection_designs_and_freeze_joint(
            data.view(),
            &[meanspec.clone(), noisespec.clone()],
        )
        .expect("initial joint pure Duchon build");
        assert_eq!(boot_designs.len(), 2);
        assert_eq!(frozen_specs.len(), 2);
        assert_eq!(boot_designs[0].smooth.terms[0].coeff_range.len(), 23);
        assert_eq!(boot_designs[1].smooth.terms[0].coeff_range.len(), 23);

        let (rebuilt_designs, refrozen_specs) =
            build_term_collection_designs_and_freeze_joint(data.view(), &frozen_specs)
                .expect("rebuilding frozen joint pure Duchon specs should succeed");
        assert_eq!(rebuilt_designs.len(), 2);
        assert_eq!(refrozen_specs.len(), 2);

        for idx in 0..2 {
            let direct = build_term_collection_design(data.view(), &frozen_specs[idx])
                .expect("direct frozen pure Duchon rebuild");
            assert_term_collection_designs_match(
                &rebuilt_designs[idx],
                &direct,
                if idx == 0 {
                    "mean pure Duchon frozen rebuild"
                } else {
                    "noise pure Duchon frozen rebuild"
                },
            );
            assert_eq!(rebuilt_designs[idx].smooth.terms[0].coeff_range.len(), 23);
            match &refrozen_specs[idx].smooth_terms[0].basis {
                SmoothBasisSpec::Duchon { spec, .. } => {
                    assert!(matches!(
                        spec.identifiability,
                        SpatialIdentifiability::FrozenTransform { .. }
                    ));
                }
                _ => panic!("expected Duchon term"),
            }
        }

        let kappa_options = SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 1,
            rel_tol: 1e-6,
            pilot_subsample_threshold: 0,
            ..SpatialLengthScaleOptimizationOptions::default()
        };
        let joint_setup =
            two_block_exact_joint_hyper_setup(&frozen_specs[0], &frozen_specs[1], &kappa_options);
        // Design B: Duchon anisotropy η is a fixed, geometry-derived basis
        // parameter, never a REML axis, so two pure-Duchon blocks contribute no
        // outer log-κ axis — the joint outer vector is ρ-only.
        assert_eq!(joint_setup.log_kappa_dim(), 0);

        let mean_term_indices = spatial_length_scale_term_indices(&frozen_specs[0]);
        let noise_term_indices = spatial_length_scale_term_indices(&frozen_specs[1]);
        assert!(
            mean_term_indices.is_empty() && noise_term_indices.is_empty(),
            "pure Duchon blocks enroll no outer κ/ψ axis"
        );
        let mut cache = ExactJointDesignCache::new(
            data.view(),
            vec![
                (
                    frozen_specs[0].clone(),
                    rebuilt_designs[0].clone(),
                    mean_term_indices.clone(),
                ),
                (
                    frozen_specs[1].clone(),
                    rebuilt_designs[1].clone(),
                    noise_term_indices.clone(),
                ),
            ],
            joint_setup.rho_dim(),
            joint_setup.log_kappa_dims_per_term(),
        )
        .expect("pure Duchon exact-joint cache");

        // With no κ axis the joint outer vector is ρ-only; realizing the cache
        // at θ₀ must reproduce the directly-rebuilt frozen designs, since there
        // is no per-axis log-κ update to apply.
        let theta0 = joint_setup.theta0();
        assert_eq!(theta0.len(), joint_setup.rho_dim());
        cache
            .ensure_theta(&theta0)
            .expect("pure Duchon cache theta update");
        let cache_designs = cache.designs();
        assert_term_collection_designs_match(
            cache_designs[0],
            &rebuilt_designs[0],
            "mean pure Duchon cache",
        );
        assert_term_collection_designs_match(
            cache_designs[1],
            &rebuilt_designs[1],
            "noise pure Duchon cache",
        );
    }

    #[test]
    fn bounded_linear_gaussian_fit_respects_interval() {
        let n = 64usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            let x = -1.0 + 2.0 * t;
            // z must be linearly independent of {1, x}: a ramp z = (x+1)/2
            // is exactly collinear with the intercept and x, so the bounded
            // column's true ∂η/∂θ Jacobian is genuinely rank-deficient and the
            // identifiability audit (correctly) refuses the fit. A 2-cycle
            // sinusoid is orthogonal to the constant and the linear ramp.
            let z = (2.0 * std::f64::consts::PI * 2.0 * t).sin();
            data[[i, 0]] = x;
            data[[i, 1]] = z;
            y[i] = 0.25 + 0.8 * x + 0.05 * z;
        }
        let spec = TermCollectionSpec {
            linear_terms: vec![
                LinearTermSpec {
                    name: "x".to_string(),
                    feature_col: 0,
                    feature_cols: vec![0],
                    categorical_levels: vec![],
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Bounded {
                        min: 0.0,
                        max: 0.5,
                        prior: BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 },
                    },
                    coefficient_min: None,
                    coefficient_max: None,
                },
                LinearTermSpec {
                    name: "z".to_string(),
                    feature_col: 1,
                    feature_cols: vec![1],
                    categorical_levels: vec![],
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                },
            ],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };

        let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
            data.view(),
            y,
            Array1::ones(n),
            Array1::zeros(n),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &FitOptions {
                max_iter: 40,
                penalty_shrinkage_floor: None,
                ..FitOptions::default()
            },
            &SpatialLengthScaleOptimizationOptions {
                enabled: false,
                ..SpatialLengthScaleOptimizationOptions::default()
            },
        )
        .expect("bounded gaussian fit");

        let bounded_idx = fitted.design.linear_ranges[0].1.start;
        let estimate = fitted.fit.beta[bounded_idx];
        assert!(
            (0.0..=0.5).contains(&estimate),
            "bounded coefficient escaped interval: {estimate}"
        );
        assert!(
            estimate > 0.1,
            "bounded coefficient should move into the positive interior, got {estimate}"
        );
    }

    #[test]
    fn bounded_fit_geometry_precision_is_on_user_scale() {
        use gam_linalg::faer_ndarray::FaerCholesky;

        let n = 72usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            let x = -1.0 + 2.0 * t;
            let z = (4.0 * std::f64::consts::PI * t).sin();
            data[[i, 0]] = x;
            data[[i, 1]] = z;
            y[i] = 0.2 + 0.35 * x - 0.15 * z;
        }
        let spec = TermCollectionSpec {
            linear_terms: vec![
                LinearTermSpec {
                    name: "x".to_string(),
                    feature_col: 0,
                    feature_cols: vec![0],
                    categorical_levels: vec![],
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Bounded {
                        min: -0.5,
                        max: 0.5,
                        prior: BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 },
                    },
                    coefficient_min: None,
                    coefficient_max: None,
                },
                LinearTermSpec {
                    name: "z".to_string(),
                    feature_col: 1,
                    feature_cols: vec![1],
                    categorical_levels: vec![],
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                },
            ],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };

        let fitted = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &FitOptions {
                max_iter: 40,
                penalty_shrinkage_floor: None,
                ..FitOptions::default()
            },
        )
        .expect("bounded gaussian fit");
        let precision = &fitted
            .fit
            .geometry
            .as_ref()
            .expect("bounded fit geometry")
            .penalized_hessian;
        let covariance = fitted
            .fit
            .beta_covariance()
            .expect("bounded user covariance");
        // User-scale covariance must be present, square, span every user
        // coefficient (intercept + the two linear terms), and be finite — a
        // bounded() fit with inference on must not silently drop it (gam#854).
        assert_eq!(
            covariance.nrows(),
            precision.nrows(),
            "bounded user covariance must be square and match the precision dimension"
        );
        assert_eq!(
            covariance.ncols(),
            precision.ncols(),
            "bounded user covariance must be square and match the precision dimension"
        );
        assert!(
            covariance.iter().all(|v| v.is_finite()),
            "bounded user covariance must be finite on every entry"
        );
        assert!(
            (0..covariance.nrows()).all(|i| covariance[[i, i]] > 0.0),
            "bounded user covariance must have a strictly positive variance on every coefficient"
        );
        // Dispersion-ownership contract (`inference::dispersion_cov`): the stored
        // `geometry.penalized_hessian` is the UNSCALED penalized Hessian `H`, while
        // the reported `beta_covariance` is `Vb = φ̂·H⁻¹`. For this profiled-Gaussian
        // fit `φ̂ = σ̂²` (the coefficient-covariance scale), so the inverse precision
        // and the covariance are an exact pair only after multiplying by that scale
        // — verifying it confirms the bounded fit both exports a covariance
        // (gam#854) AND scales it by the estimated dispersion (gam#1514), rather
        // than the pre-#1514 invariant `Vb == H⁻¹` that silently dropped σ̂².
        let cov_scale = fitted.fit.coefficient_covariance_scale().unwrap();
        assert!(
            cov_scale.is_finite() && cov_scale > 0.0,
            "profiled-Gaussian bounded fit must report a finite positive σ̂² scale, got {cov_scale}"
        );
        assert!(
            cov_scale < 1.0,
            "near-noiseless fit should have a small residual variance, got σ̂²={cov_scale}"
        );
        let chol = precision
            .cholesky(faer::Side::Lower)
            .expect("bounded user precision cholesky");
        let solved = chol.solve_mat(&Array2::eye(covariance.nrows()));
        // Compare on the unscaled scale (`Vb/σ̂² == H⁻¹`) so the tolerance keeps its
        // original magnitude rather than shrinking with σ̂².
        for i in 0..solved.nrows() {
            for j in 0..solved.ncols() {
                let unscaled_cov = covariance[[i, j]] / cov_scale;
                assert!(
                    (solved[[i, j]] - unscaled_cov).abs() < 1e-5,
                    "user-scale precision/covariance mismatch at ({i},{j}): inverse {}, \
                     covariance/σ̂² {unscaled_cov} (σ̂²={cov_scale})",
                    solved[[i, j]]
                );
            }
        }
    }

    #[test]
    fn term_collection_design_emits_linear_coefficient_constraints() {
        let data = array![[0.0], [1.0], [2.0], [3.0]];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "x".to_string(),
                feature_col: 0,
                feature_cols: vec![0],
                categorical_levels: vec![],
                double_penalty: false,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: Some(0.0),
                coefficient_max: Some(1.0),
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };
        let design = build_term_collection_design(data.view(), &spec).expect("design");
        let constraints = design.linear_constraints.expect("constraints");
        assert_eq!(constraints.a.ncols(), design.design.ncols());
        assert_eq!(constraints.a.nrows(), 2);
        let linear_idx = design.linear_ranges[0].1.start;
        assert_eq!(constraints.a[[0, linear_idx]], 1.0);
        assert_eq!(constraints.b[0], 0.0);
        assert_eq!(constraints.a[[1, linear_idx]], -1.0);
        assert_eq!(constraints.b[1], -1.0);
    }

    #[test]
    fn linear_termspec_defaults_to_null_recovery_when_field_is_omitted() {
        let json = r#"{"name":"x","feature_col":0}"#;
        let term: LinearTermSpec = serde_json::from_str(json).expect("deserialize linear term");
        assert!(term.double_penalty);
        assert!(matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Unconstrained
        ));
    }

    #[test]
    fn linear_effects_get_distinct_function_space_penalty_blocks() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let spec = TermCollectionSpec {
            linear_terms: vec![
                LinearTermSpec {
                    name: "x1".to_string(),
                    feature_col: 0,
                    feature_cols: vec![0],
                    categorical_levels: vec![],
                    double_penalty: true,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                },
                LinearTermSpec {
                    name: "x2".to_string(),
                    feature_col: 1,
                    feature_cols: vec![1],
                    categorical_levels: vec![],
                    double_penalty: true,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                },
            ],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };
        let design = build_term_collection_design(data.view(), &spec).expect("design");
        assert_eq!(design.penalties.len(), 2);
        assert_eq!(design.penaltyinfo.len(), 2);
        assert_eq!(design.penaltyinfo[0].termname.as_deref(), Some("x1"));
        assert_eq!(design.penaltyinfo[1].termname.as_deref(), Some("x2"));
        assert_eq!(design.penaltyinfo[0].penalty.effective_rank, 1);
        assert_eq!(design.penaltyinfo[1].penalty.effective_rank, 1);
        let x1 = design.linear_ranges[0].1.start;
        let x2 = design.linear_ranges[1].1.start;
        assert_eq!(design.penalties[0].col_range, x1..(x1 + 1));
        assert_eq!(design.penalties[1].col_range, x2..(x2 + 1));
        assert!((design.penalties[0].local[[0, 0]] - 35.0 / 3.0).abs() < 1e-12);
        assert!((design.penalties[1].local[[0, 0]] - 56.0 / 3.0).abs() < 1e-12);

        let scale = 7.0;
        let mut scaled_data = data.clone();
        scaled_data
            .column_mut(0)
            .mapv_inplace(|value| value * scale);
        let scaled = build_term_collection_design(scaled_data.view(), &spec)
            .expect("rescaled linear design");
        let original_mass = design.penalties[0].local[[0, 0]];
        let scaled_mass = scaled.penalties[0].local[[0, 0]];
        assert!((scaled_mass - scale * scale * original_mass).abs() < 1e-10);
        let beta = 0.37;
        let rescaled_beta = beta / scale;
        assert!(
            (beta * beta * original_mass - rescaled_beta * rescaled_beta * scaled_mass).abs()
                < 1e-12,
            "the physical shrinkage energy must be invariant to basis rescaling"
        );
    }

    #[test]
    fn bounded_uniform_prior_matches_beta_one_one_terms() {
        let theta = 0.7;
        let uniform = bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::Uniform)
            .expect("uniform prior geometry");
        let beta11 =
            bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::Beta { a: 1.0, b: 1.0 })
                .expect("Beta(1,1) prior geometry");
        assert!((uniform.0 - beta11.0).abs() < 1e-12);
        assert!((uniform.1 - beta11.1).abs() < 1e-12);
        assert!((uniform.2 - beta11.2).abs() < 1e-12);
        assert!((uniform.3 - beta11.3).abs() < 1e-12);
    }

    #[test]
    fn boundednone_prior_has_no_extra_latentobjective_terms() {
        let theta = 0.7;
        let none = bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::None)
            .expect("flat prior geometry");
        assert_eq!(none, (0.0, 0.0, 0.0, 0.0));

        let uniform = bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::Uniform)
            .expect("uniform prior geometry");
        assert!(uniform.0.is_finite());
        assert!(uniform.0 < 0.0);
        assert!(uniform.1.abs() > 1e-6);
        assert!(uniform.2 > 0.0);
        assert!(uniform.3.is_finite());
    }

    #[test]
    fn bounded_prior_tail_value_and_derivatives_share_the_logit_surface() {
        let theta = 40.0;
        let terms =
            bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 3.0 })
                .expect("tail prior geometry");
        let jet = logit_inverse_link_jet5(theta);
        let expected_value = -2.0 * gam_linalg::utils::stable_softplus(-theta)
            - 3.0 * gam_linalg::utils::stable_softplus(theta);
        assert_eq!(terms.0, expected_value);
        assert_eq!(terms.2, 5.0 * jet.d1);
        assert_eq!(terms.3, 5.0 * jet.d2);
        assert!(terms.2 > 0.0, "the representable tail curvature was lost");
        assert!(terms.3 < 0.0, "right-tail curvature must be decreasing");
    }

    #[test]
    fn exact_bounded_edf_matches_trace_formula_for_simple_penalty() {
        let penalties = vec![PenaltySpec::Dense(Array2::eye(1))];
        let lambdas = array![0.25];
        let cov = array![[2.0]];
        let (edf_by_block, _penalty_block_trace, edf_total) =
            exact_bounded_edf(&penalties, &lambdas, &cov).expect("exact bounded edf");
        assert_eq!(edf_by_block.len(), 1);
        assert!((edf_by_block[0] - 0.5).abs() < 1e-12);
        assert!((edf_total - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bounded_joint_hessian_directional_derivative_matches_finite_difference() {
        let x = array![[0.2, -1.0], [0.8, 0.5], [1.1, 1.2], [1.7, -0.3]];
        let y = array![0.4, 1.0, 1.7, 2.2];
        let weights = Array1::ones(y.len());
        let family = BoundedLinearFamily {
            likelihood: gam_spec::GlmLikelihoodSpec::canonical(
                LikelihoodSpec::gaussian_identity(),
            ),
            latent_cloglog_state: None,
            mixture_link_state: None,
            sas_link_state: None,
            y: y.clone(),
            weights: weights.clone(),
            design: x.clone(),
            designzeroed: {
                let mut dz = x.clone();
                dz.column_mut(0).fill(0.0);
                dz
            },
            offset: Array1::zeros(y.len()),
            bounded_terms: vec![BoundedLinearTermMeta {
                col_idx: 0,
                min: 0.0,
                max: 1.0,
                prior: BoundedCoefficientPriorSpec::Uniform,
            }],
        };
        let state = vec![ParameterBlockState {
            beta: array![0.4, -0.2],
            eta: Array1::zeros(y.len()),
        }];
        let direction = array![0.3, -0.4];

        let analytic = family
            .exact_newton_joint_hessian_directional_derivative(&state, &direction)
            .expect("analytic derivative")
            .expect("joint derivative");

        let h = 1e-6;
        let plus_state = vec![ParameterBlockState {
            beta: &state[0].beta + &(direction.clone() * h),
            eta: Array1::zeros(y.len()),
        }];
        let minus_state = vec![ParameterBlockState {
            beta: &state[0].beta - &(direction.clone() * h),
            eta: Array1::zeros(y.len()),
        }];
        let plus = family
            .exact_newton_joint_hessian(&plus_state)
            .expect("plus hessian")
            .expect("plus exact hessian");
        let minus = family
            .exact_newton_joint_hessian(&minus_state)
            .expect("minus hessian")
            .expect("minus exact hessian");
        let fd = (plus - minus) / (2.0 * h);

        for i in 0..analytic.nrows() {
            for j in 0..analytic.ncols() {
                assert_eq!(
                    analytic[[i, j]].signum(),
                    fd[[i, j]].signum(),
                    "directional derivative sign mismatch at ({i},{j}): analytic={}, fd={}",
                    analytic[[i, j]],
                    fd[[i, j]]
                );
                assert!(
                    (analytic[[i, j]] - fd[[i, j]]).abs() < 1e-5,
                    "directional derivative mismatch at ({i},{j}): analytic={}, fd={}",
                    analytic[[i, j]],
                    fd[[i, j]]
                );
            }
        }
    }

    #[test]
    fn adaptive_initial_epsilons_use_mean_fallbackwhen_median_is_tiny() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[5e-10, 0.0], [6e-10, 0.0]],
            d1: array![[1e-10, 0.0], [0.0, 1e-10], [2e-10, 0.0], [0.0, 2e-10]],
            // d2 layout: rows = (point k, axis a, axis b) with row = (k*d + a)*d + b.
            // For P=2 points and d=2 axes that is 2*2*2 = 8 rows.
            d2: array![
                [3e-10, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [3e-10, 0.0],
                [4e-10, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [4e-10, 0.0],
            ],
            collocation_points: array![[0.0, 0.0], [1.0, 1.0]],
            dimension: 2,
        };
        let beta = array![1.0, 1.0];
        let (eps_0, eps_g, eps_c) =
            compute_initial_epsilons(&beta, &[cache], 1e-8).expect("initial epsilons");
        assert!(eps_0 >= 1e-8);
        assert!(eps_g >= 1e-8);
        assert!(eps_c >= 1e-8);
    }

    #[test]
    fn adaptiveweight_clamp_is_applied_in_u_space() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[0.0, 0.0]],
            d1: array![[0.0, 0.0], [0.0, 0.0]],
            // d2 layout: P=1 collocation point and d=2 axes give 1*2*2 = 4 rows
            // ordered (axis_a, axis_b) = (0,0), (0,1), (1,0), (1,1).
            d2: array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            collocation_points: array![[0.0, 0.0]],
            dimension: 2,
        };
        let beta = array![0.0, 0.0];
        let out = compute_spatial_adaptiveweights_for_beta(
            &beta,
            &[cache],
            1e-8,
            1e-8,
            1e-8,
            1e-8,
            1e2,
            None,
        )
        .expect("adaptive weights");
        assert_eq!(out.len(), 1);
        // Raw u would be 1/eps = 1e8, so clamping to 1e2 yields diagnostics 1/u.
        assert!((out[0].inv_magweight[0] - 1e-2).abs() < 1e-12);
        assert!((out[0].invgradweight[0] - 1e-2).abs() < 1e-12);
        assert!((out[0].inv_lapweight[0] - 1e-2).abs() < 1e-12);
    }

    #[test]
    fn adaptiveweight_inverse_consistencywithout_clamp() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[1.0, 0.0], [2.0, 0.0]],
            d1: array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
            // d2 layout: rows = (point k, axis a, axis b) with row = (k*d + a)*d + b.
            // P=2 and d=2 gives 8 rows; row 0 (point 0, a=b=0) and row 4 (point 1, a=b=0)
            // carry the original diagonal-curvature signals.
            d2: array![
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            collocation_points: array![[0.0, 0.0], [1.0, 1.0]],
            dimension: 2,
        };
        let beta = array![1.0, 1.0];
        let out = compute_spatial_adaptiveweights_for_beta(
            &beta,
            &[cache],
            1e-6,
            1e-6,
            1e-6,
            1e-12,
            1e12,
            None,
        )
        .expect("adaptive weights");
        assert_eq!(out.len(), 1);
        assert!((out[0].inv_magweight[0] - 1.0).abs() < 1e-10);
        assert!((out[0].inv_magweight[1] - 2.0).abs() < 1e-10);
        assert!((out[0].invgradweight[0] - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((out[0].invgradweight[1] - 8.0_f64.sqrt()).abs() < 1e-10);
        assert!((out[0].inv_lapweight[0] - 1.0).abs() < 1e-10);
        assert!((out[0].inv_lapweight[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn adaptiveweight_is_monotone_in_signal_magnitude() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[1.0, 0.0]],
            d1: array![[1.0, 0.0], [0.0, 1.0]],
            // d2 layout: P=1 collocation point and d=2 axes give 4 rows ordered
            // (axis_a, axis_b) = (0,0), (0,1), (1,0), (1,1). Row 0 carries the
            // diagonal curvature signal so monotonicity in beta is preserved.
            d2: array![[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            collocation_points: array![[0.0, 0.0]],
            dimension: 2,
        };
        let beta_small = array![0.25, 0.25];
        let beta_large = array![2.0, 2.0];
        let small = compute_spatial_adaptiveweights_for_beta(
            &beta_small,
            std::slice::from_ref(&cache),
            1e-8,
            1e-8,
            1e-8,
            1e-12,
            1e12,
            None,
        )
        .expect("small adaptive weights");
        let large = compute_spatial_adaptiveweights_for_beta(
            &beta_large,
            &[cache],
            1e-8,
            1e-8,
            1e-8,
            1e-12,
            1e12,
            None,
        )
        .expect("large adaptive weights");
        assert!(small[0].inv_magweight[0] < large[0].inv_magweight[0]);
        assert!(small[0].invgradweight[0] < large[0].invgradweight[0]);
        assert!(small[0].inv_lapweight[0] < large[0].inv_lapweight[0]);
    }

    // ------------------------------------------------------------------
    // Posterior-SNR adaptive weighting: end-to-end objective-quality test.
    //
    // Truth: a 1D function on a uniform grid with a genuine sharp, localized
    // feature in the middle (a tall narrow bump — high curvature, *credibly*
    // determined) flanked by flat regions that are *zero* (no trend to leak).
    // Using an isolated feature rather than a monotone step is deliberate: a
    // step edge would impose a non-zero slope that the curvature penalty smears
    // linearly into the adjacent flat region, so heavier flat-region smoothing
    // would *raise* flat MSE (tilted line) regardless of the weighting — a
    // confound that has nothing to do with the SNR weight. With a bump, the
    // flat regions have zero true slope and zero true curvature, so suppressing
    // the spurious noise-curvature there genuinely denoises toward truth.
    //
    // The LEFT flat region is a low-information region whose grid coefficients
    // are poorly determined: its posterior covariance Sigma_beta = H^{-1}
    // carries large variance there, and the noisy point-estimate beta_hat shows
    // spurious curvature there. The RIGHT flat region is well determined.
    //
    // We drive the *real* adaptive-weight machinery
    // (`compute_spatial_adaptiveweights_for_beta`) two ways:
    //   * magnitude-only baseline  -> covariance `None`,
    //   * posterior-SNR (default)  -> covariance `Some(Sigma_beta)`,
    // then build the curvature surrogate penalty K = D2^T diag(w_c) D2 each
    // weighting implies, solve the penalized least-squares fit
    // beta = (X^T X + lambda K)^{-1} X^T y on the identity design (X = I, so
    // the coefficients ARE the fitted function values), and compare MSE-to-truth
    //
    //   * in the noisy LEFT flat region  -> must be STRICTLY LOWER for SNR
    //     (it does not chase noise), and
    //   * at the EDGE                     -> must be NO WORSE for SNR
    //     (the credible edge is preserved).
    // ------------------------------------------------------------------
    fn posterior_snr_finite_difference_d2(m: usize, h: f64) -> Array2<f64> {
        // Second-difference operator on a uniform 1D grid of `m` points: one
        // collocation row per interior point, row layout matching the grouped
        // curvature operator (block_dim = dimension^2 = 1 here). Rows for the
        // two endpoints are zero (no curvature defined there).
        let mut d2 = Array2::<f64>::zeros((m, m));
        for k in 1..m - 1 {
            d2[[k, k - 1]] = 1.0 / (h * h);
            d2[[k, k]] = -2.0 / (h * h);
            d2[[k, k + 1]] = 1.0 / (h * h);
        }
        d2
    }

    #[test]
    fn posterior_snr_weighting_suppresses_noise_and_preserves_edge() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let m = 41usize;
        let h = 1.0 / (m as f64 - 1.0);
        let xs: Vec<f64> = (0..m).map(|j| j as f64 * h).collect();

        // True function: zero-flat on both ends with a single tall, narrow
        // Gaussian bump centered at x = 0.5 (the credible high-curvature
        // feature). The flat ends carry no trend, so smoothing the spurious
        // noise-curvature there denoises toward the true zero without any
        // edge-slope leakage confound.
        let edge_center = 0.5;
        let bump_width = 0.06;
        let amplitude = 2.0;
        let truth = Array1::from_iter(xs.iter().map(|&x| {
            let z = (x - edge_center) / bump_width;
            amplitude * (-0.5 * z * z).exp()
        }));

        // Region indices. The flat region stops well short of the bump so its
        // true curvature is ~0; the edge band straddles the bump's steep flanks.
        let left_flat: Vec<usize> = (0..m).filter(|&j| xs[j] <= 0.25).collect();
        let edge_band: Vec<usize> = (0..m)
            .filter(|&j| (xs[j] - edge_center).abs() <= 0.10)
            .collect();
        assert!(!left_flat.is_empty() && !edge_band.is_empty());

        // Deterministic, reproducible "noise" pattern. The LEFT flat region is
        // a low-information region: large noise. Elsewhere noise is tiny.
        let noise = |j: usize| -> f64 {
            let s = ((j as f64) * 12.9898).sin() * 43758.5453;
            let frac = s - s.floor(); // pseudo-uniform in [0,1)
            2.0 * frac - 1.0 // in [-1,1)
        };
        let mut y = truth.clone();
        let mut beta_hat = truth.clone();
        for &j in &left_flat {
            let nz = 0.85 * noise(j);
            y[j] += nz;
            beta_hat[j] += nz; // noisy point estimate drives the weights
        }
        for j in 0..m {
            if !left_flat.contains(&j) {
                let nz = 0.02 * noise(j + 7);
                y[j] += nz;
                beta_hat[j] += nz;
            }
        }

        // Working-Laplace conditional covariance proxy Sigma_beta = H^{-1}.
        // Diagonal posterior variances: large in the poorly-determined LEFT
        // flat region, small elsewhere. (Diagonal is sufficient and is the
        // honest leading-order structure of a per-coefficient variance.)
        let mut sigma = Array2::<f64>::zeros((m, m));
        for j in 0..m {
            let var = if left_flat.contains(&j) { 0.55 } else { 1e-4 };
            sigma[[j, j]] = var;
        }

        let d2 = posterior_snr_finite_difference_d2(m, h);
        // Magnitude-only and SNR machinery share the identical cache, betas and
        // epsilons; only the covariance argument differs.
        let cache = SpatialOperatorRuntimeCache {
            termname: "snr_1d".to_string(),
            feature_cols: vec![0],
            coeff_global_range: 0..m,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: Array2::<f64>::zeros((m, m)),
            d1: Array2::<f64>::zeros((m, m)),
            d2: d2.clone(),
            collocation_points: {
                let mut cp = Array2::<f64>::zeros((m, 1));
                for j in 0..m {
                    cp[[j, 0]] = xs[j];
                }
                cp
            },
            dimension: 1,
        };

        let eps = 0.05; // shared Charbonnier transition scale
        let weight_floor = 1e-12;
        let weight_ceiling = 1e12;

        let mag = compute_spatial_adaptiveweights_for_beta(
            &beta_hat,
            std::slice::from_ref(&cache),
            eps,
            eps,
            eps,
            weight_floor,
            weight_ceiling,
            None,
        )
        .expect("magnitude-only weights");
        let snr = compute_spatial_adaptiveweights_for_beta(
            &beta_hat,
            std::slice::from_ref(&cache),
            eps,
            eps,
            eps,
            weight_floor,
            weight_ceiling,
            Some(&sigma),
        )
        .expect("posterior-SNR weights");

        // inv_lapweight = 1 / w_c; recover the curvature weights w_c.
        let w_mag = mag[0].inv_lapweight.mapv(|iv| 1.0 / iv);
        let w_snr = snr[0].inv_lapweight.mapv(|iv| 1.0 / iv);

        // Sanity on the mechanism itself: in the noisy LEFT flat region the SNR
        // curvature weight must be substantially LARGER (more smoothing) than
        // the magnitude-only weight, because the spurious point-estimate
        // curvature there is swamped by the posterior variance. (The real
        // quality bar is the MSE assertions below; this only confirms the
        // covariance path is genuinely engaged.)
        let mean = |idx: &[usize], w: &Array1<f64>| -> f64 {
            idx.iter().map(|&j| w[j]).sum::<f64>() / idx.len() as f64
        };
        assert!(
            mean(&left_flat, &w_snr) > mean(&left_flat, &w_mag) * 1.5,
            "SNR must penalize the noisy flat region more: w_snr={:.4e} vs w_mag={:.4e}",
            mean(&left_flat, &w_snr),
            mean(&left_flat, &w_mag),
        );

        // Penalized least-squares fit on the identity design X = I:
        //   beta = (I + lambda * D2^T diag(w_c) D2)^{-1} y.
        // lambda is large enough that the curvature penalty (and hence the
        // weighting) materially shapes the fit, rather than the identity data
        // term dominating it.
        let lambda = 0.5;
        let fit = |w: &Array1<f64>| -> Array1<f64> {
            let k = scalar_operatorhessian(&d2, w); // D2^T diag(w) D2 (symmetric)
            let mut a = Array2::<f64>::eye(m);
            a.scaled_add(lambda, &k);
            let factor = a
                .cholesky(Side::Lower)
                .expect("penalized normal matrix is SPD");
            factor.solvevec(&y)
        };
        let fit_mag = fit(&w_mag);
        let fit_snr = fit(&w_snr);

        let region_mse = |idx: &[usize], f: &Array1<f64>| -> f64 {
            idx.iter().map(|&j| (f[j] - truth[j]).powi(2)).sum::<f64>() / idx.len() as f64
        };

        let mse_flat_mag = region_mse(&left_flat, &fit_mag);
        let mse_flat_snr = region_mse(&left_flat, &fit_snr);
        let mse_edge_mag = region_mse(&edge_band, &fit_mag);
        let mse_edge_snr = region_mse(&edge_band, &fit_snr);

        // Objective quality assertions.
        // 1. Noisy flat region: SNR fit is STRICTLY closer to truth (does not
        //    chase noise). Require a clear margin, not a hairline win.
        assert!(
            mse_flat_snr < mse_flat_mag * 0.9,
            "posterior-SNR should be strictly smoother in the noisy flat region: \
                 mse_flat_snr={mse_flat_snr:.6e} vs mse_flat_mag={mse_flat_mag:.6e}"
        );
        // 2. Edge region: SNR recovers the edge at least as sharply (MSE no
        //    worse, with a small tolerance for numerical wiggle).
        assert!(
            mse_edge_snr <= mse_edge_mag * 1.05,
            "posterior-SNR must recover the edge at least as sharply: \
                 mse_edge_snr={mse_edge_snr:.6e} vs mse_edge_mag={mse_edge_mag:.6e}"
        );

        // Guard against the degenerate pass where both fits are identical (the
        // covariance path must actually change the weights).
        assert!(
            (mse_flat_mag - mse_flat_snr).abs() > 1e-9,
            "the two weightings must produce materially different flat-region fits"
        );
    }

    #[test]
    fn exact_spatial_adaptive_regularization_fit_runswithout_mm() {
        let n = 48usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.19 * i as f64).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).sin() + 0.25 * x1;
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
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                        length_scale: 0.7,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &FitOptions {
                max_iter: 25,
                tol: 1e-5,
                adaptive_regularization: Some(AdaptiveRegularizationOptions {
                    enabled: true,
                    max_mm_iter: 4,
                    beta_rel_tol: 1e-4,
                    max_epsilon_outer_iter: 2,
                    min_epsilon: 1e-6,
                    ..AdaptiveRegularizationOptions::default()
                }),
                penalty_shrinkage_floor: None,
                ..FitOptions::default()
            },
        )
        .expect("exact adaptive spatial fit should succeed");

        let diag = fit
            .adaptive_diagnostics
            .as_ref()
            .expect("adaptive diagnostics should be present");
        assert_eq!(diag.mm_iterations, 0);
        assert!(diag.epsilon_0.is_finite() && diag.epsilon_0 > 0.0);
        assert!(diag.epsilon_g.is_finite() && diag.epsilon_g > 0.0);
        assert!(diag.epsilon_c.is_finite() && diag.epsilon_c > 0.0);
        assert_eq!(diag.maps.len(), 1);
        assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
        assert!(fit.fit.reml_score.is_finite());
    }

    #[test]
    fn pure_duchon_skips_operator_triplet_adaptive_overlay() {
        let n = 56usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            data[[i, 0]] = t.cos();
            data[[i, 1]] = t.sin();
            y[i] = 0.55 * t.sin() + 0.18 * (3.0 * t).cos();
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "pure_duchon_circle".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 16 },
                        length_scale: None,
                        power: 0.0,
                        nullspace_order: DuchonNullspaceOrder::Degree(2),
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        // Truly "pure" Duchon: every operator dial off, leaving
                        // only the always-on `Primary` RKHS Gram. The Charbonnier
                        // adaptive overlay reweights the {mass, tension, curvature}
                        // operator triplet, with `Primary` substituting for the
                        // explicit `Stiffness` D2 (see the curvature-channel
                        // comment in `extract_spatial_operator_runtime_caches`):
                        // when mass and tension are also off the cache requires
                        // none of the three operators, so the overlay's dispatch
                        // gate yields an empty `runtime_caches` and the adaptive
                        // path is skipped. `DuchonOperatorPenaltySpec::default()`
                        // ships mass + tension active, which DOES feed the
                        // overlay (Primary as curvature) and so is NOT the "pure"
                        // case under test here.
                        operator_penalties: DuchonOperatorPenaltySpec::all_disabled(),
                        periodic: None,
                        boundary: OneDimensionalBoundary::Open,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };

        let fit = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &FitOptions {
                compute_inference: false,
                max_iter: 28,
                tol: 1e-5,
                adaptive_regularization: Some(AdaptiveRegularizationOptions {
                    enabled: true,
                    max_mm_iter: 4,
                    beta_rel_tol: 1e-4,
                    max_epsilon_outer_iter: 2,
                    min_epsilon: 1e-6,
                    ..AdaptiveRegularizationOptions::default()
                }),
                penalty_shrinkage_floor: None,
                ..FitOptions::default()
            },
        )
        .expect("pure Duchon exact adaptive fit should succeed");

        // Pure Duchon (every operator dial off) ships only the always-on `Primary`
        // RKHS Gram. The Charbonnier spatial-adaptive overlay consumes a
        // {mass, tension, curvature} operator triplet, so with no operator
        // penalties shipped the per-term cache gate in
        // `extract_spatial_operator_runtime_caches` rejects the term and the
        // overlay's runtime-caches set is empty — the adaptive path is skipped
        // and the fit collapses to the plain quadratic REML over the lone
        // `Primary` penalty. (Default Duchon — `DuchonOperatorPenaltySpec::default()`
        // — does feed the overlay because mass + tension are active and Primary
        // substitutes for the Stiffness curvature channel, per #858.)
        assert!(
            fit.adaptive_diagnostics.is_none(),
            "pure Duchon carries no operator triplet, so the Charbonnier overlay must not run"
        );
        assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
        assert!(fit.fit.reml_score.is_finite());
    }

    #[test]
    fn exact_spatial_adaptive_binomial_sas_fit_preserves_link_state() {
        let n = 36usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = -1.0 + 2.0 * (i as f64 / (n as f64 - 1.0));
            let x1 = (0.23 * i as f64).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            let eta = 0.55 * x0 - 0.2 * x1 + 0.1 * x0 * x1;
            let p = 1.0 / (1.0 + (-eta).exp());
            let u = ((i * 37 + 13) % 100) as f64 / 100.0;
            y[i] = if u < p { 1.0 } else { 0.0 };
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
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                        length_scale: 0.7,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodSpec::binomial_sas(
                gam_solve::mixture_link::sas_link_state_from_raw(0.1, -0.2)
                    .expect("valid SAS link state"),
            ),
            &FitOptions {
                sas_link: Some(gam_problem::SasLinkSpec {
                    initial_epsilon: 0.1,
                    initial_log_delta: -0.2,
                }),
                // #1601 re-home: this is the most parameter-rich optimization in
                // the file (ρ + 3-channel adaptive λ + SAS ε/log_delta). It was
                // orphaned dead with `max_iter: 15`, which the current optimizer
                // cannot reach the 1e-5 projected-gradient floor within: 15 iters
                // stops at grad_norm 1.727e-3, 20 at 1.344e-3, both monotonically
                // descending. 40 iters (the dominant sibling budget in this file)
                // converges cleanly with margin. The optimizer/tolerance/mgcv
                // fallback are all correct; only the test's iteration cap was
                // stale/under-provisioned for this high-dimensional path.
                max_iter: 40,
                tol: 1e-5,
                adaptive_regularization: Some(AdaptiveRegularizationOptions {
                    enabled: true,
                    max_mm_iter: 4,
                    beta_rel_tol: 1e-4,
                    max_epsilon_outer_iter: 2,
                    min_epsilon: 1e-6,
                    ..AdaptiveRegularizationOptions::default()
                }),
                penalty_shrinkage_floor: None,
                ..FitOptions::default()
            },
        )
        .expect("exact adaptive SAS fit should succeed");

        match fit.fit.fitted_link {
            FittedLinkState::Sas { state, covariance } => {
                assert!(state.epsilon.is_finite());
                assert!(state.log_delta.is_finite());
                assert!(state.delta.is_finite() && state.delta > 0.0);
                assert!(covariance.is_none());
            }
            other => panic!("expected SAS link parameters, got {other:?}"),
        }
    }

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

    // #1601 re-home — YIELDED to PR #1736 (reopen #901). The
    // `exact_spatial_adaptive_joint_hypergradient_matches_finite_difference`
    // guard fails on a GENUINE analytic-gradient bug: the Firth/Jeffreys
    // ρ/ψ-derivative fold into the StrictPseudoLaplace adaptive-ψ LAML
    // envelope term `a_i` (analytic 0.13410 vs converged FD 0.11961 on a
    // well-conditioned Gaussian-identity fit — not the #901 conditioning
    // artifact). PR #1736 ships exactly that production fix
    // (`joint_jeffreys_information_depends_on_psi` gate in psi_hyper.rs) AND
    // re-homes this same guard, so it is yielded there rather than shipped
    // red here.

    #[test]
    fn exact_spatial_adaptive_1dobjective_profile_has_finite_gradient_lambda_surface() {
        let n = 96usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            y[i] = 0.12 * (2.0 * std::f64::consts::PI * x).sin()
                + 0.05 * (5.0 * std::f64::consts::PI * x).cos()
                + 1.4 / (1.0 + (-(x - 0.5) / 0.012).exp());
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
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 31 },
                        length_scale: Some(1.0),
                        power: 2.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        // `extract_spatial_operator_runtime_caches` returns a
                        // cache only when the term ships an EXPLICIT Stiffness
                        // penalty (the Charbonnier D2 surrogate has no
                        // matching shipped penalty otherwise — see
                        // `extract_spatial_operator_runtime_caches` docs at the
                        // call site). `DuchonOperatorPenaltySpec::default()`
                        // disables Stiffness (Primary is the exact RKHS
                        // curvature), so the runtime cache would be empty and
                        // the adaptive-overlay path this test exists to
                        // exercise would never fire. Pin `all_active()` so all
                        // three operator channels (mass, tension, stiffness)
                        // are present in the design's penalty list.
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
        let runtime_caches = extract_spatial_operator_runtime_caches(&spec, &baseline.design)
            .expect("runtime caches");
        assert_eq!(runtime_caches.len(), 1);
        let (eps_0, eps_g, eps_c) =
            compute_initial_epsilons(&baseline.fit.beta, &runtime_caches, 1e-8)
                .expect("initial epsilons");
        let (base_family, blockspec, derivative_blocks) =
            build_spatial_adaptive_joint_hyper_scaffold(&baseline, &runtime_caches, &y, n);
        let outer_opts = BlockwiseFitOptions {
            inner_max_cycles: 20,
            inner_tol: 1e-6,
            outer_max_iter: 20,
            outer_tol: 1e-6,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };

        let evaluate_theta = |log_lambda_g: f64| {
            let family = base_family.with_adaptive_params(
                vec![SpatialAdaptiveTermHyperParams {
                    lambda: [1e-12, log_lambda_g.exp(), 1e-12],
                    epsilon: [eps_0, eps_g, eps_c],
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
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient,
            )
            .expect("joint hyper eval")
        };

        let low = evaluate_theta((1e-8_f64).ln());
        let mid = evaluate_theta((1e-4_f64).ln());
        let high = evaluate_theta((1e-2_f64).ln());

        for (label, eval) in [("low", &low), ("mid", &mid), ("high", &high)] {
            assert!(
                eval.objective.is_finite(),
                "{label} gradient-lambda profile objective is not finite: {}",
                eval.objective
            );
            assert!(
                eval.gradient.iter().all(|v| v.is_finite()),
                "{label} gradient-lambda profile gradient contains non-finite entries: {:?}",
                eval.gradient
            );
        }
        assert!(
            (low.objective - high.objective).abs() > 1e-8,
            "gradient-lambda profile should remain identifiable: low={}, high={}",
            low.objective,
            high.objective
        );
    }

    #[test]
    fn high_center_duchon_fit_ignores_unavailable_spatial_adaptive_overlay() {
        let n = 320usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            y[i] = 0.12 * (2.0 * std::f64::consts::PI * x).sin()
                + 0.05 * (5.0 * std::f64::consts::PI * x).cos()
                + 1.4 / (1.0 + (-(x - 0.5) / 0.012).exp());
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
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 120 },
                        length_scale: Some(1.0),
                        power: 2.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::default(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };

        let fit = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &FitOptions {
                max_iter: 40,
                adaptive_regularization: Some(AdaptiveRegularizationOptions {
                    enabled: true,
                    ..AdaptiveRegularizationOptions::default()
                }),
                penalty_shrinkage_floor: None,
                ..FitOptions::default()
            },
        )
        .expect("high-center adaptive Duchon fit should not fail");

        assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
        assert!(fit.fit.deviance.is_finite());
        assert!(fit.fit.edf_total().is_some_and(f64::is_finite));
        assert!(
            fit.adaptive_diagnostics.is_none(),
            "Duchon does not expose the complete operator triplet required by the runtime adaptive overlay"
        );
    }

    #[test]
    fn binomial_logit_tail_curvature_uses_stable_exact_formula() {
        let eta = array![30.0, 30.0, -30.0, -30.0, 40.0, -40.0];
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let weights = Array1::ones(eta.len());
        let obs = evaluate_standard_familyobservations(
            LikelihoodSpec::binomial_logit(),
            None,
            None,
            None,
            &y,
            &weights,
            &eta,
        )
        .expect("stable logit observations");

        for i in 0..eta.len() {
            let jet = logit_inverse_link_jet5(eta[i]);
            assert!(
                (obs.neghessian_eta[i] - jet.d1).abs() <= 1e-12 * (1.0 + jet.d1.abs()),
                "eta={} y={} curvature={} target={}",
                eta[i],
                y[i],
                obs.neghessian_eta[i],
                jet.d1
            );
            assert!(
                (obs.neghessian_eta_derivative[i] - jet.d2).abs() <= 1e-12 * (1.0 + jet.d2.abs()),
                "eta={} y={} dcurvature={} target={}",
                eta[i],
                y[i],
                obs.neghessian_eta_derivative[i],
                jet.d2
            );
            assert!(
                obs.neghessian_eta[i].is_finite()
                    && obs.neghessian_eta_derivative[i].is_finite()
                    && obs.log_likelihood.is_finite(),
                "expected finite logit tail observation state at eta={} y={}",
                eta[i],
                y[i]
            );
        }
    }

    #[test]
    fn non_logit_binomial_tailobservations_stay_finite() {
        let y = array![0.0, 1.0, 1.0, 0.0];
        for (family, eta) in [
            (
                LikelihoodSpec::binomial_probit(),
                array![12.0, -12.0, 18.0, -18.0],
            ),
            // At eta=18 the cloglog Fisher information is mathematically below
            // f64 support; exact geometry correctly refuses that trial instead
            // of injecting curvature. These are still extreme but representable
            // asymmetric tails.
            (
                LikelihoodSpec::binomial_cloglog(),
                array![5.0, -18.0, 4.0, -30.0],
            ),
        ] {
            let weights = Array1::ones(eta.len());
            let obs = evaluate_standard_familyobservations(
                family.clone(),
                None,
                None,
                None,
                &y,
                &weights,
                &eta,
            )
            .expect("tail observations");
            assert!(obs.log_likelihood.is_finite(), "family={family:?}");
            assert!(
                obs.score.iter().all(|v| v.is_finite())
                    && obs.neghessian_eta.iter().all(|v| v.is_finite())
                    && obs.neghessian_eta_derivative.iter().all(|v| v.is_finite()),
                "family={family:?}"
            );
        }
    }

    #[test]
    fn two_block_exact_joint_setup_sanitizes_non_finite_rho_seed() {
        let setup = ExactJointHyperSetup::new(
            array![f64::NEG_INFINITY, 0.25, f64::INFINITY],
            array![-12.0, -12.0, -12.0],
            array![12.0, 12.0, 12.0],
            SpatialLogKappaCoords::new_with_dims(array![0.5], vec![1]),
            SpatialLogKappaCoords::new_with_dims(array![-2.0], vec![1]),
            SpatialLogKappaCoords::new_with_dims(array![2.0], vec![1]),
        );

        let theta0 = setup.theta0();
        assert!(theta0.iter().all(|v| v.is_finite()));
        assert_eq!(theta0[0], 0.0);
        assert_eq!(theta0[1], 0.25);
        assert_eq!(theta0[2], 0.0);
        assert_eq!(theta0[3], 0.5);
    }

    #[test]
    fn extracted_spatial_runtime_cache_matches_normalized_design_penalties() {
        let n = 24usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
            data[[i, 1]] = (0.23 * i as f64).cos();
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
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 7 },
                        length_scale: 0.8,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).expect("design");
        let caches =
            extract_spatial_operator_runtime_caches(&spec, &design).expect("runtime caches");
        assert_eq!(caches.len(), 1);
        let cache = &caches[0];
        let s0 = {
            let raw = cache.d0.t().dot(&cache.d0);
            (&raw + &raw.t()) * 0.5
        };
        let s1 = {
            let raw = cache.d1.t().dot(&cache.d1);
            (&raw + &raw.t()) * 0.5
        };
        let s2 = {
            let raw = cache.d2.t().dot(&cache.d2);
            (&raw + &raw.t()) * 0.5
        };

        let s0_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s0,
        );
        let s1_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s1,
        );
        let s2_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s2,
        );

        let p_total = design.design.ncols();
        let err0 = (&s0_global
            - &design.penalties[cache.mass_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err1 = (&s1_global
            - &design.penalties[cache.tension_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err2 = (&s2_global
            - &design.penalties[cache.stiffness_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        assert!(err0 < 1e-8, "mass penalty mismatch too large: {err0}");
        assert!(err1 < 1e-8, "tension penalty mismatch too large: {err1}");
        assert!(err2 < 1e-8, "stiffness penalty mismatch too large: {err2}");
    }

    #[test]
    fn extracted_duchon_spatial_runtime_cache_matches_normalized_design_penalties() {
        let n = 32usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
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
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 11 },
                        length_scale: Some(0.8),
                        power: 2.0,
                        nullspace_order: DuchonNullspaceOrder::Zero,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).expect("design");
        assert_eq!(design.penalties.len(), 4);
        let caches =
            extract_spatial_operator_runtime_caches(&spec, &design).expect("runtime caches");
        assert_eq!(caches.len(), 1);
        let cache = &caches[0];
        let s0 = {
            let raw = cache.d0.t().dot(&cache.d0);
            (&raw + &raw.t()) * 0.5
        };
        let s1 = {
            let raw = cache.d1.t().dot(&cache.d1);
            (&raw + &raw.t()) * 0.5
        };
        let s2 = {
            let raw = cache.d2.t().dot(&cache.d2);
            (&raw + &raw.t()) * 0.5
        };

        let s0_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s0,
        );
        let s1_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s1,
        );
        let s2_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s2,
        );

        let p_total = design.design.ncols();
        let err0 = (&s0_global
            - &design.penalties[cache.mass_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err1 = (&s1_global
            - &design.penalties[cache.tension_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err2 = (&s2_global
            - &design.penalties[cache.stiffness_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        // For the DUCHON family (unlike the Matern sibling test, which uses raw
        // `fast_ata` operators in both forward and cache paths and so round-trips to
        // 1e-8) the shipped penalty and the runtime-cache reconstruction are
        // assembled by genuinely different code that agrees only to float-
        // reassembly precision (~1e-5 abs on mass, ~4e-7 on tension), not
        // bit-exactly. The forward design builds each operator at the
        // standardization-compensated length scale and stores a Frobenius
        // `normalization_scale`; the cache REBUILDS the operator from the stored
        // metadata (re-deriving the compensated length scale, re-centering the mass
        // rows) and divides by `sqrt(stored_scale)`. Two effects co-add on every
        // channel: (a) the stored scale is not recomputed from the rebuilt matrix
        // (e.g. mass 2.828419 vs true ‖G‖_F 2.828427, ~2.8e-6 rel), and (b) the
        // rebuilt collocation operator is not bit-identical to the one the forward
        // penalty was built from. Both forms are valid unit-Frobenius-normalized
        // penalties — this is a precision/consistency nicety, not a correctness
        // bug. The 1e-4 bound clears the observed maxima (mass 1.045e-5, tension
        // 4.0e-7) with margin while still catching any order-of-magnitude operator
        // regression.
        assert!(
            err0 < 1e-4,
            "Duchon mass penalty mismatch too large: {err0}"
        );
        assert!(
            err1 < 1e-4,
            "Duchon tension penalty mismatch too large: {err1}"
        );
        assert!(
            err2 < 1e-4,
            "Duchon stiffness penalty mismatch too large: {err2}"
        );
    }

    #[test]
    fn spatial_adaptive_explicit_second_order_kind_matches_block_sparsity() {
        let alpha_mass_0 = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogLambdaMagnitude,
        };
        let alpha_mass_1 = SpatialAdaptiveHyperSpec {
            cache_index: 1,
            kind: SpatialAdaptiveHyperKind::LogLambdaMagnitude,
        };
        let alpha_grad_0 = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogLambdaGradient,
        };
        let eta_mass = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogEpsilonMagnitude,
        };
        let eta_grad = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogEpsilonGradient,
        };

        assert_eq!(
            alpha_mass_0.explicit_second_order_kind(alpha_mass_0),
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaAlpha
        );
        assert_eq!(
            alpha_mass_0.explicit_second_order_kind(alpha_mass_1),
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero
        );
        assert_eq!(
            alpha_mass_0.explicit_second_order_kind(alpha_grad_0),
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero
        );
        assert_eq!(
            alpha_mass_1.explicit_second_order_kind(eta_mass),
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta
        );
        assert_eq!(
            eta_mass.explicit_second_order_kind(alpha_mass_1),
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta
        );
        assert_eq!(
            eta_mass.explicit_second_order_kind(eta_mass),
            SpatialAdaptiveExplicitSecondOrderKind::SharedEtaEta
        );
        assert_eq!(
            eta_mass.explicit_second_order_kind(eta_grad),
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero
        );
    }

    // #1601 re-home — YIELDED to PR #1736 (reopen #901). The
    // `adaptive_hyper_derivative_dispatch_matches_reference` guard is
    // re-homed there alongside the custom-family adaptive-ψ Jeffreys-info
    // dispatch fix it exercises; kept out of this PR to avoid a duplicate
    // definition when both merge.
    #[test]
    fn scalar_charbonnier_exact_derivatives_match_finite_difference() {
        let signal = array![0.7, -1.1];
        let epsilon = 0.3;
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-5;
        let value = |x: &Array1<f64>| {
            CharbonnierScalarBlockState::from_signal(x.clone(), epsilon).penalty_value()
        };
        for i in 0..signal.len() {
            let mut plus = signal.clone();
            plus[i] += h;
            let mut minus = signal.clone();
            minus[i] -= h;
            let gradfd = (value(&plus) - value(&minus)) / (2.0 * h);
            let hessfd = (value(&plus) - 2.0 * value(&signal) + value(&minus)) / (h * h);
            assert!((state.betagradient_coeff()[i] - gradfd).abs() < 1e-6);
            assert!((state.betahessian_diag()[i] - hessfd).abs() < 1e-4);
        }
    }

    #[test]
    fn grouped_charbonnier_exactgradient_matches_finite_difference() {
        let blocks = array![[0.8, -0.4], [0.3, 0.9]];
        let epsilon = 0.25;
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let h = 1e-6;
        let value = |x: &Array2<f64>| {
            CharbonnierGroupedBlockState::from_signal_blocks(x.clone(), epsilon).penalty_value()
        };
        let analytic = state.betagradient_blocks();
        for k in 0..blocks.nrows() {
            for axis in 0..blocks.ncols() {
                let mut plus = blocks.clone();
                plus[[k, axis]] += h;
                let mut minus = blocks.clone();
                minus[[k, axis]] -= h;
                let gradfd = (value(&plus) - value(&minus)) / (2.0 * h);
                assert!((analytic[[k, axis]] - gradfd).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn scalar_charbonnier_log_epsilon_derivatives_match_finite_difference() {
        let signal = array![0.4, -0.9];
        let epsilon = 0.35_f64;
        let eta = epsilon.ln();
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-5;
        let value = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .penalty_value()
        };
        let gradfd = (value(eta + h) - value(eta - h)) / (2.0 * h);
        let hessfd = (value(eta + h) - 2.0 * value(eta) + value(eta - h)) / (h * h);
        assert!((state.log_epsilon_gradient_terms().sum() - gradfd).abs() < 1e-6);
        assert!((state.log_epsilon_hessian_terms().sum() - hessfd).abs() < 1e-4);

        let eval_grad = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .betagradient_coeff()
        };
        let mixedfd = (&eval_grad(eta + h) - &eval_grad(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((state.log_epsilon_betagradient_coeff()[i] - mixedfd[i]).abs() < 1e-6);
        }

        let eval_hess = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .betahessian_diag()
        };
        let betahessfd = (&eval_hess(eta + h) - &eval_hess(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((state.log_epsilon_betahessian_diag()[i] - betahessfd[i]).abs() < 1e-5);
        }

        let eval_log_grad = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .log_epsilon_betagradient_coeff()
        };
        let second_mixedfd = (&eval_log_grad(eta + h) - &eval_log_grad(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!(
                (state.log_epsilon_beta_mixed_second_coeff()[i] - second_mixedfd[i]).abs() < 1e-5
            );
        }

        let eval_log_hess = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .log_epsilon_betahessian_diag()
        };
        let second_hessfd = (&eval_log_hess(eta + h) - &eval_log_hess(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!(
                (state.log_epsilon_betahessian_second_diag()[i] - second_hessfd[i]).abs() < 1e-4
            );
        }
    }

    #[test]
    fn scalar_charbonnier_directionalhessian_matches_finite_difference() {
        let signal = array![0.5, -0.6];
        let epsilon = 0.2;
        let direction = array![0.3, -0.1];
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-6;
        let analytic = state.directionalhessian_diag(&direction);
        let evalhess = |step: f64| {
            let shifted = &signal + &(direction.mapv(|v| step * v));
            CharbonnierScalarBlockState::from_signal(shifted, epsilon).betahessian_diag()
        };
        let fd = (&evalhess(h) - &evalhess(-h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((analytic[i] - fd[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn scalar_charbonnier_log_epsilon_directionalhessian_matches_finite_difference() {
        let signal = array![0.5, -0.6];
        let epsilon = 0.2;
        let direction = array![0.3, -0.1];
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-6;
        let analytic = state.log_epsilon_betahessian_directional_diag(&direction);
        let evalhess = |step: f64| {
            let shifted = &signal + &(direction.mapv(|v| step * v));
            CharbonnierScalarBlockState::from_signal(shifted, epsilon)
                .log_epsilon_betahessian_diag()
        };
        let fd = (&evalhess(h) - &evalhess(-h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((analytic[i] - fd[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn grouped_charbonnier_log_epsilon_derivatives_match_finite_difference() {
        let blocks = array![[0.7, -0.2], [0.1, 0.8]];
        let epsilon = 0.3_f64;
        let eta = epsilon.ln();
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let h = 1e-5;
        let value = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .penalty_value()
        };
        let gradfd = (value(eta + h) - value(eta - h)) / (2.0 * h);
        let hessfd = (value(eta + h) - 2.0 * value(eta) + value(eta - h)) / (h * h);
        assert!((state.log_epsilon_gradient_terms().sum() - gradfd).abs() < 1e-6);
        assert!((state.log_epsilon_hessian_terms().sum() - hessfd).abs() < 1e-4);

        let eval_grad = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .betagradient_blocks()
        };
        let mixedfd = (&eval_grad(eta + h) - &eval_grad(eta - h)) / (2.0 * h);
        let analytic_mixed = state.log_epsilon_betagradient_blocks();
        for k in 0..blocks.nrows() {
            for axis in 0..blocks.ncols() {
                assert!((analytic_mixed[[k, axis]] - mixedfd[[k, axis]]).abs() < 1e-6);
            }
        }

        let eval_hess = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .betahessian_blocks()
        };
        let plus_hess = eval_hess(eta + h);
        let minus_hess = eval_hess(eta - h);
        let analytic_hess = state.log_epsilon_betahessian_blocks();
        for k in 0..analytic_hess.len() {
            let fd = (&plus_hess[k] - &minus_hess[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic_hess[k][[i, j]] - fd[[i, j]]).abs() < 1e-5);
                }
            }
        }

        let eval_log_grad = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .log_epsilon_betagradient_blocks()
        };
        let second_mixedfd = (&eval_log_grad(eta + h) - &eval_log_grad(eta - h)) / (2.0 * h);
        let analytic_second_mixed = state.log_epsilon_beta_mixed_second_blocks();
        for k in 0..blocks.nrows() {
            for axis in 0..blocks.ncols() {
                assert!(
                    (analytic_second_mixed[[k, axis]] - second_mixedfd[[k, axis]]).abs() < 1e-5
                );
            }
        }

        let eval_log_hess = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .log_epsilon_betahessian_blocks()
        };
        let plus_log_hess = eval_log_hess(eta + h);
        let minus_log_hess = eval_log_hess(eta - h);
        let analytic_second_hess = state.log_epsilon_betahessian_second_blocks();
        for k in 0..analytic_second_hess.len() {
            let fd = (&plus_log_hess[k] - &minus_log_hess[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic_second_hess[k][[i, j]] - fd[[i, j]]).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn grouped_charbonnier_directionalhessian_matches_finite_difference() {
        let blocks = array![[0.6, -0.2], [0.4, 0.5]];
        let direction = array![[0.1, -0.3], [0.2, 0.15]];
        let epsilon = 0.4;
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let analytic = state.directionalhessian_blocks(&direction);
        let h = 1e-6;
        let evalhess = |step: f64| {
            let shifted = &blocks + &(direction.mapv(|v| step * v));
            CharbonnierGroupedBlockState::from_signal_blocks(shifted, epsilon).betahessian_blocks()
        };
        let plus = evalhess(h);
        let minus = evalhess(-h);
        for k in 0..analytic.len() {
            let fd = (&plus[k] - &minus[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic[k][[i, j]] - fd[[i, j]]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn grouped_charbonnier_log_epsilon_directionalhessian_matches_finite_difference() {
        let blocks = array![[0.6, -0.2], [0.4, 0.5]];
        let direction = array![[0.1, -0.3], [0.2, 0.15]];
        let epsilon = 0.4;
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let analytic = state.log_epsilon_betahessian_directional_blocks(&direction);
        let h = 1e-6;
        let evalhess = |step: f64| {
            let shifted = &blocks + &(direction.mapv(|v| step * v));
            CharbonnierGroupedBlockState::from_signal_blocks(shifted, epsilon)
                .log_epsilon_betahessian_blocks()
        };
        let plus = evalhess(h);
        let minus = evalhess(-h);
        for k in 0..analytic.len() {
            let fd = (&plus[k] - &minus[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic[k][[i, j]] - fd[[i, j]]).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn grouped_charbonnier_directionalhessian_blocks_are_symmetric() {
        let blocks = array![[0.6, -0.2], [0.4, 0.5]];
        let direction = array![[0.1, -0.3], [0.2, 0.15]];
        let epsilon = 0.4;
        let analytic = CharbonnierGroupedBlockState::from_signal_blocks(blocks, epsilon)
            .directionalhessian_blocks(&direction);
        for (k, block) in analytic.iter().enumerate() {
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    assert!(
                        (block[[i, j]] - block[[j, i]]).abs() < 1e-12,
                        "directional Hessian block {k} is not symmetric at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_charbonnier_local_quadratic_curvature_matches_transition_scale() {
        let signal = array![0.0, 0.0, 0.0];
        let small = CharbonnierScalarBlockState::from_signal(signal.clone(), 1e-3);
        let large = CharbonnierScalarBlockState::from_signal(signal, 1e3);
        for (&a, &b) in small
            .betahessian_diag()
            .iter()
            .zip(large.betahessian_diag().iter())
        {
            assert!(
                (a - 1e3).abs() < 1e-7,
                "small-epsilon curvature should be 1/eps, got {a}"
            );
            assert!(
                (b - 1e-3).abs() < 1e-13,
                "large-epsilon curvature should be 1/eps, got {b}"
            );
        }
    }

    #[test]
    fn grouped_charbonnier_local_quadratic_curvature_matches_transition_scale() {
        let blocks = array![[0.0, 0.0], [0.0, 0.0]];
        let small = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), 1e-3);
        let large = CharbonnierGroupedBlockState::from_signal_blocks(blocks, 1e3);
        for (small_block, large_block) in small
            .betahessian_blocks()
            .into_iter()
            .zip(large.betahessian_blocks().into_iter())
        {
            let eye = Array2::<f64>::eye(small_block.nrows());
            assert!(
                (&small_block - &eye.mapv(|v| 1e3 * v)).mapv(f64::abs).sum() < 1e-7,
                "small-epsilon grouped curvature should equal I/eps"
            );
            assert!(
                (&large_block - &eye.mapv(|v| 1e-3 * v))
                    .mapv(f64::abs)
                    .sum()
                    < 1e-13,
                "large-epsilon grouped curvature should equal I/eps"
            );
        }
    }

    #[test]
    fn scalar_charbonnier_small_signal_matches_local_half_quadratic() {
        let signal = array![1e-5, -2e-5, 3e-5];
        for &epsilon in &[1e-3, 1e-1, 1.0, 1e2] {
            let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
            let value = state.penalty_value();
            let target = 0.5 * signal.iter().map(|v| v * v).sum::<f64>() / epsilon;
            let rel = (value - target).abs() / target.max(1e-20);
            assert!(
                rel < 5e-3,
                "scalar Charbonnier should match 0.5*t^2/eps locally: eps={epsilon}, value={value}, target={target}, rel={rel}"
            );
        }
    }

    #[test]
    fn grouped_charbonnier_small_signal_matches_local_half_quadratic() {
        let blocks = array![[1e-5, -2e-5], [3e-5, 4e-5]];
        for &epsilon in &[1e-3, 1e-1, 1.0, 1e2] {
            let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
            let value = state.penalty_value();
            let target = 0.5 * blocks.iter().map(|v| v * v).sum::<f64>() / epsilon;
            let rel = (value - target).abs() / target.max(1e-20);
            assert!(
                rel < 5e-3,
                "grouped Charbonnier should match 0.5*||v||^2/eps locally: eps={epsilon}, value={value}, target={target}, rel={rel}"
            );
        }
    }

    #[test]
    fn adaptive_diagnostics_json_roundtrip_preserves_shapes() {
        let diag = AdaptiveRegularizationDiagnostics {
            epsilon_0: 0.01,
            epsilon_g: 0.02,
            epsilon_c: 0.03,
            epsilon_outer_iterations: 2,
            mm_iterations: 3,
            converged: true,
            maps: vec![AdaptiveSpatialMap {
                termname: "matern".to_string(),
                feature_cols: vec![0, 1],
                collocation_points: array![[1.0, 2.0], [3.0, 4.0]],
                inv_magweight: array![0.05, 0.15],
                invgradweight: array![0.1, 0.2],
                inv_lapweight: array![0.3, 0.4],
            }],
        };
        let payload = serde_json::to_value(&diag).expect("serialize diagnostics");
        assert_eq!(payload["mm_iterations"].as_u64(), Some(3));
        assert_eq!(
            payload["maps"][0]["collocation_points"]["dim"]
                .as_array()
                .map(|v| v.len()),
            Some(2)
        );
        let decoded: AdaptiveRegularizationDiagnostics =
            serde_json::from_value(payload).expect("deserialize diagnostics");
        assert_eq!(decoded.epsilon_outer_iterations, 2);
        assert_eq!(decoded.mm_iterations, 3);
        assert!(decoded.converged);
        assert_eq!(decoded.maps.len(), 1);
        assert_eq!(decoded.maps[0].collocation_points.nrows(), 2);
        assert_eq!(decoded.maps[0].collocation_points.ncols(), 2);
        assert_eq!(decoded.maps[0].invgradweight.len(), 2);
        assert_eq!(decoded.maps[0].inv_lapweight.len(), 2);
    }

    // ------------------------------------------------------------------
    // #760(c): exact bounded() posterior sampler on the latent scale.
    //
    // Verifies the three principled contracts of
    // `sample_bounded_latent_posterior_internal`:
    //   1. every user-scale draw of a bounded column lies STRICTLY inside its
    //      interval (the interval map cannot escape the bounds);
    //   2. an unconstrained column is the ordinary Gaussian Laplace draw whose
    //      sample variance recovers the marginal `H_user^{-1}` diagonal — i.e.
    //      the latent precision transform did NOT corrupt the unconstrained
    //      block (J_ii = 1 there);
    //   3. cross-coefficient correlation between the bounded and unconstrained
    //      columns is present (the full `H_latent = J H_user J` joint is drawn,
    //      not an independent per-column approximation), recovered on the latent
    //      scale where the draw is exactly Gaussian.
    // ------------------------------------------------------------------
    #[test]
    fn bounded_latent_sampler_draws_in_bounds_and_preserves_joint() {
        let (min, max) = (-0.5_f64, 0.5_f64);
        // User-scale mode: bounded col at 0.2 (interior), unconstrained at 1.3.
        let beta_user = array![0.2, 1.3];
        // A correlated user-scale penalized Hessian (SPD): off-diagonal couples
        // the two coefficients so the joint draw must reproduce correlation.
        let user_hessian = array![[4.0, 1.2], [1.2, 3.0]];
        let bounded_columns = vec![BoundedSampleColumn {
            col_idx: 0,
            min,
            max,
        }];
        let n_draws = 40_000usize;
        let draws = sample_bounded_latent_posterior_internal(
            &beta_user,
            &user_hessian,
            &bounded_columns,
            n_draws,
            1.0,
            7607760,
        )
        .expect("bounded latent sampler");
        assert_eq!(draws.dim(), (n_draws, 2));

        // (1) Bounded column strictly inside (min, max).
        for k in 0..n_draws {
            let b = draws[(k, 0)];
            assert!(
                b > min && b < max,
                "bounded draw {b} escaped interval ({min}, {max})"
            );
        }

        // Reconstruct the latent geometry the sampler used so we can check the
        // moments on the scale where the draw is exactly Gaussian.
        let theta_mode0 = bounded_user_to_latent(beta_user[0], min, max);
        let (_, _, j0) = bounded_latent_to_user(theta_mode0, min, max);
        let h_latent = array![
            [user_hessian[[0, 0]] * j0 * j0, user_hessian[[0, 1]] * j0],
            [user_hessian[[1, 0]] * j0, user_hessian[[1, 1]]]
        ];
        // Latent covariance = H_latent^{-1} (2x2 closed form).
        let det = h_latent[[0, 0]] * h_latent[[1, 1]] - h_latent[[0, 1]] * h_latent[[1, 0]];
        let cov_latent = array![
            [h_latent[[1, 1]] / det, -h_latent[[0, 1]] / det],
            [-h_latent[[1, 0]] / det, h_latent[[0, 0]] / det]
        ];

        // Map bounded draws back to the latent scale; the unconstrained column
        // is already on its (identity) latent scale.
        let mut theta0 = Array1::<f64>::zeros(n_draws);
        let mut theta1 = Array1::<f64>::zeros(n_draws);
        for k in 0..n_draws {
            theta0[k] = bounded_user_to_latent(draws[(k, 0)], min, max);
            theta1[k] = draws[(k, 1)];
        }
        let mean0 = theta0.sum() / n_draws as f64;
        let mean1 = theta1.sum() / n_draws as f64;
        let var0 = theta0.iter().map(|&t| (t - mean0).powi(2)).sum::<f64>() / n_draws as f64;
        let var1 = theta1.iter().map(|&t| (t - mean1).powi(2)).sum::<f64>() / n_draws as f64;
        let cov01 = theta0
            .iter()
            .zip(theta1.iter())
            .map(|(&a, &b)| (a - mean0) * (b - mean1))
            .sum::<f64>()
            / n_draws as f64;

        // (2)/(3) Latent moments match H_latent^{-1} within Monte-Carlo error.
        let rel = |emp: f64, truth: f64| (emp - truth).abs() / truth.abs().max(1e-12);
        assert!(
            rel(var0, cov_latent[[0, 0]]) < 0.05,
            "latent var0 {var0} vs {} ",
            cov_latent[[0, 0]]
        );
        assert!(
            rel(var1, cov_latent[[1, 1]]) < 0.05,
            "latent var1 {var1} vs {}",
            cov_latent[[1, 1]]
        );
        let corr_emp = cov01 / (var0.sqrt() * var1.sqrt());
        let corr_truth =
            cov_latent[[0, 1]] / (cov_latent[[0, 0]].sqrt() * cov_latent[[1, 1]].sqrt());
        assert!(
            corr_truth.abs() > 0.2,
            "fixture must carry real correlation, got {corr_truth}"
        );
        assert!(
            (corr_emp - corr_truth).abs() < 0.03,
            "joint correlation not preserved: empirical {corr_emp} vs truth {corr_truth}"
        );
    }

    // ------------------------------------------------------------------
    // #1514: the bounded latent sampler must apply the dispersion scale.
    //
    // The exported `user_hessian` is the UNSCALED penalized Hessian, so for a
    // free-dispersion (profiled-Gaussian) family the latent posterior covariance
    // is `cov_scale · H_latent⁻¹` (here `cov_scale = σ̂²`). The caller passes
    // `sqrt_cov_scale = √cov_scale`; this test confirms the sampler scales the
    // latent variances by exactly `cov_scale` (and `sqrt_cov_scale = 1` recovers
    // the unscaled draw), so the draw spread matches the fit's reported `Vb`.
    // Without the scale a Gaussian bounded slope's draws were `1/σ̂` too wide.
    // ------------------------------------------------------------------
    #[test]
    fn bounded_latent_sampler_applies_dispersion_scale() {
        let (min, max) = (-0.5_f64, 0.5_f64);
        let beta_user = array![0.2, 1.3];
        let user_hessian = array![[4.0, 1.2], [1.2, 3.0]];
        let bounded_columns = vec![BoundedSampleColumn {
            col_idx: 0,
            min,
            max,
        }];
        let n_draws = 60_000usize;

        // Reconstruct the unscaled latent covariance the sampler builds internally.
        let theta_mode0 = bounded_user_to_latent(beta_user[0], min, max);
        let (_, _, j0) = bounded_latent_to_user(theta_mode0, min, max);
        let h_latent = array![
            [user_hessian[[0, 0]] * j0 * j0, user_hessian[[0, 1]] * j0],
            [user_hessian[[1, 0]] * j0, user_hessian[[1, 1]]]
        ];
        let det = h_latent[[0, 0]] * h_latent[[1, 1]] - h_latent[[0, 1]] * h_latent[[1, 0]];
        let cov_latent_unit = array![
            [h_latent[[1, 1]] / det, -h_latent[[0, 1]] / det],
            [-h_latent[[1, 0]] / det, h_latent[[0, 0]] / det]
        ];

        // A non-unit dispersion scale (e.g. σ̂² = 2.25 ⇒ √cov_scale = 1.5).
        let cov_scale = 2.25_f64;
        let sqrt_cov_scale = cov_scale.sqrt();
        let draws = sample_bounded_latent_posterior_internal(
            &beta_user,
            &user_hessian,
            &bounded_columns,
            n_draws,
            sqrt_cov_scale,
            424242,
        )
        .expect("scaled bounded latent sampler");

        // Map back to the latent scale where the draw is exactly Gaussian.
        let mut theta0 = Array1::<f64>::zeros(n_draws);
        let mut theta1 = Array1::<f64>::zeros(n_draws);
        for k in 0..n_draws {
            theta0[k] = bounded_user_to_latent(draws[(k, 0)], min, max);
            theta1[k] = draws[(k, 1)];
        }
        let mean0 = theta0.sum() / n_draws as f64;
        let mean1 = theta1.sum() / n_draws as f64;
        let var0 = theta0.iter().map(|&t| (t - mean0).powi(2)).sum::<f64>() / n_draws as f64;
        let var1 = theta1.iter().map(|&t| (t - mean1).powi(2)).sum::<f64>() / n_draws as f64;

        // Latent variances must equal `cov_scale · H_latent⁻¹`, NOT `H_latent⁻¹`.
        let rel = |emp: f64, truth: f64| (emp - truth).abs() / truth.abs().max(1e-12);
        let truth0 = cov_scale * cov_latent_unit[[0, 0]];
        let truth1 = cov_scale * cov_latent_unit[[1, 1]];
        assert!(
            rel(var0, truth0) < 0.05,
            "scaled latent var0 {var0} vs {truth0} (cov_scale={cov_scale})"
        );
        assert!(
            rel(var1, truth1) < 0.05,
            "scaled latent var1 {var1} vs {truth1} (cov_scale={cov_scale})"
        );
        // Guard the contract direction: the scaled variance is meaningfully larger
        // than the unscaled one (so a missing scale would be caught, not masked).
        assert!(
            var0 > 1.5 * cov_latent_unit[[0, 0]],
            "dispersion scale was not applied: var0 {var0} ~ unit cov {}",
            cov_latent_unit[[0, 0]]
        );
    }
}
