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
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, OneDimensionalBoundary,
        PenaltySource, SpatialIdentifiability,
    };
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
            [1.0, -0.8, 0.0, 0.0, 0.00, 0.57],
            [2.0, -0.4, 1.0, 1.0, 0.14, 0.00],
            [3.0, -0.1, 0.0, 2.0, 0.29, 0.86],
            [4.0, 0.2, 1.0, 3.0, 0.43, 0.29],
            [5.0, 0.5, 0.0, 0.0, 0.57, 1.00],
            [6.0, 0.7, 1.0, 1.0, 0.71, 0.43],
            [7.0, 0.9, 0.0, 2.0, 0.86, 0.14],
            [8.0, 1.1, 1.0, 3.0, 1.00, 0.71],
        ];
        let smooth = |name: &str, feature_col: usize| SmoothTermSpec {
            frozen_parametric_residualization: None,
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
                    double_penalty: true,
                    identifiability: BSplineIdentifiability::None,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };
        let spec = TermCollectionSpec {
            // Two linear columns but only the first emits a function-space
            // ridge: coefficient width is deliberately not penalty count.
            linear_terms: vec![
                LinearTermSpec {
                    name: "penalized_linear".to_string(),
                    feature_col: 0,
                    feature_cols: vec![0],
                    categorical_levels: vec![],
                    double_penalty: true,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                    frozen_function_mass: None,
                },
                LinearTermSpec {
                    name: "unpenalized_linear".to_string(),
                    feature_col: 1,
                    feature_cols: vec![1],
                    categorical_levels: vec![],
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                    frozen_function_mass: None,
                },
            ],
            // Likewise, both random effects own non-empty coefficient ranges
            // but only the first emits a ridge.
            random_effect_terms: vec![
                RandomEffectTermSpec {
                    name: "penalized_group".to_string(),
                    feature_col: 2,
                    drop_first_level: false,
                    penalized: true,
                    frozen_levels: Some(vec![0, 1]),
                    lenient_unseen: true,
                },
                RandomEffectTermSpec {
                    name: "unpenalized_group".to_string(),
                    feature_col: 3,
                    drop_first_level: false,
                    penalized: false,
                    frozen_levels: Some(vec![0, 1, 2, 3]),
                    lenient_unseen: true,
                },
            ],
            // Distinct feature ownership is essential here. Two copies of the
            // same smooth are deliberately collapsed by global hierarchical
            // identifiability, in which case the second term correctly owns no
            // realized penalty block and cannot test a two-term layout.
            // Each surviving smooth emits BOTH its primary roughness and its
            // function-space null ridge, so every repeated axis has width two.
            smooth_terms: vec![smooth("first_smooth", 4), smooth("second_smooth", 5)],
        };
        let design = build_term_collection_design(data.view(), &spec).expect("mixed design");

        assert_eq!(design.leading_penalty_blocks_before_smooth(), 2);
        assert_eq!(design.penalties.len(), 6);
        assert_eq!(design.penaltyinfo.len(), design.penalties.len());
        assert_eq!(design.penaltyinfo[0].termname.as_deref(), Some("penalized_linear"));
        assert!(matches!(
            &design.penaltyinfo[0].penalty.source,
            PenaltySource::Other(source) if source == "LinearTermRidge"
        ));
        assert_eq!(design.penaltyinfo[1].termname.as_deref(), Some("penalized_group"));
        assert!(matches!(
            &design.penaltyinfo[1].penalty.source,
            PenaltySource::Other(source) if source == "RandomEffectRidge(penalized_group)"
        ));
        let first = design
            .smooth_term_penalty_range(0)
            .expect("consistent layout")
            .expect("penalized first smooth");
        let second = design
            .smooth_term_penalty_range(1)
            .expect("consistent layout")
            .expect("penalized second smooth");
        assert_eq!(first, 2..4);
        assert_eq!(second, 4..6);
        assert_eq!(second.start, first.end);
        for (range, name) in [(first.clone(), "first_smooth"), (second.clone(), "second_smooth")]
        {
            let infos = &design.penaltyinfo[range.clone()];
            assert_eq!(infos.len(), 2);
            assert!(infos.iter().all(|info| info.termname.as_deref() == Some(name)));
            assert!(infos.iter().any(|info| {
                matches!(info.penalty.source, PenaltySource::Primary)
            }));
            assert!(infos.iter().any(|info| {
                matches!(info.penalty.source, PenaltySource::DoublePenaltyNullspace)
            }));
            for (global_index, info) in range.zip(infos.iter()) {
                assert_eq!(info.global_index, global_index);
                assert!(info.penalty.effective_rank > 0);
            }
        }

        // Independent consumer cross-check: the incremental κ realizer's
        // production range resolver translates the actual emitter layout into
        // smooth-local coordinates without constructing a second spec cursor.
        let (smooth_ranges, full_ranges) = emitted_smooth_penalty_ranges(&design)
            .expect("incremental realizer accepts composed emitted layout");
        assert_eq!(smooth_ranges, vec![0..2, 2..4]);
        assert_eq!(full_ranges, vec![first, second]);
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
            frozen_parametric_residualization: None,
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
                    input_scale: None,
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
            frozen_parametric_residualization: None,
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
            frozen_parametric_residualization: None,
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
                input_scale: None,
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
            frozen_parametric_residualization: None,
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
                input_scale: None,
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
                    frozen_function_mass: None,
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
                    frozen_function_mass: None,
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
                    frozen_function_mass: None,
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
                    frozen_function_mass: None,
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
                frozen_function_mass: None,
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
    fn linear_termspec_defaults_to_unpenalized_mle_when_field_is_omitted() {
        // Parametric effects are UNPENALIZED/MLE by default -- the field's own
        // doc says so and `default_linear_term_double_penalty()` returns false;
        // `linear(x, double_penalty=true)` is the opt-in. This copy encoded the
        // retired "null recovery" semantics, and the owning crate carries the
        // OPPOSITE assertion on byte-identical JSON
        // (`missing_linear_double_penalty_deserializes_to_unpenalized_mle`), so
        // the two were asserting contradictory defaults for one deserialization.
        let json = r#"{"name":"x","feature_col":0}"#;
        let term: LinearTermSpec = serde_json::from_str(json).expect("deserialize linear term");
        assert!(!term.double_penalty);
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
                    frozen_function_mass: None,
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
                    frozen_function_mass: None,
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
