// Unit tests of `term_specs.rs`, inlined into that module by
// `include!("term_specs_tests.rs")` at the end of the parent file. Every
// module here is `#[cfg(test)]` and reaches the parent's private items through
// `super`, exactly as it did when the text lived in `term_specs.rs`; only the
// file boundary moved (#780 line-count gate).

#[cfg(test)]
mod joint_unpenalized_dim_tests {
    use super::{ActivePenalty, ActivePenaltyInfo, PenaltySource, joint_unpenalized_dim};
    use ndarray::{Array2, array};

    fn active_penalty(
        matrix: Array2<f64>,
        effective_rank: usize,
        nullity: usize,
        original_index: usize,
        source: PenaltySource,
    ) -> ActivePenalty {
        ActivePenalty {
            matrix,
            nullity,
            null_eigenvectors: None,
            op: None,
            info: ActivePenaltyInfo {
                source,
                original_index,
                effective_rank,
                normalization_scale: 1.0,
                kronecker_factors: None,
                structural_null_frame: None,
            },
        }
    }

    #[test]
    fn no_penalty_is_fully_unpenalized() {
        assert_eq!(joint_unpenalized_dim(4, &[]), 4);
    }

    #[test]
    fn single_penalty_returns_its_own_null_space() {
        // A 3×3 penalty that penalizes only the last coordinate ⇒ 2-dim null
        // space (the first two coordinates are unpenalized).
        let s = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]];
        let penalties = [active_penalty(s, 1, 2, 0, PenaltySource::Primary)];
        assert_eq!(joint_unpenalized_dim(3, &penalties), 2);
    }

    #[test]
    fn complementary_double_penalty_has_empty_joint_null_space() {
        // The #1360 case in miniature: a "bending" penalty that leaves the
        // first coordinate (its 2-dim... here 1-dim) null, plus a
        // complementary "null-space ridge" that penalizes exactly that
        // coordinate. Per-penalty null dims are {1, 2} and sum to 3 (≈ p),
        // but the INTERSECTION is empty: every coordinate is penalized by
        // someone, so the joint unpenalized dim is 0.
        let bending = array![[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        let ridge = array![[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let penalties = [
            active_penalty(bending, 2, 1, 0, PenaltySource::Primary),
            active_penalty(ridge, 1, 2, 1, PenaltySource::DoublePenaltyNullspace),
        ];
        assert_eq!(joint_unpenalized_dim(3, &penalties), 0);
    }

    #[test]
    fn partial_overlap_keeps_shared_null_direction() {
        // Two penalties that BOTH leave coordinate 0 unpenalized ⇒ the shared
        // null direction survives the intersection (joint unpenalized dim 1),
        // even though naively summing the per-penalty dims would give 4.
        let a = array![[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]];
        let b = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]];
        let penalties = [
            active_penalty(a, 1, 2, 0, PenaltySource::Primary),
            active_penalty(b, 1, 2, 1, PenaltySource::OperatorStiffness),
        ];
        assert_eq!(joint_unpenalized_dim(3, &penalties), 1);
    }

    #[test]
    fn tensor_product_penalties_read_the_materialized_kronecker_blocks() {
        // te(x, z) with 3×2 margins: S_x ⊗ I and I ⊗ S_z. Their joint null
        // space is null(S_x) ⊗ null(S_z), 1 × 1 = 1-dimensional, which the
        // retired fallback reported as 0 for any ≥2-penalty term it did not
        // materialize.
        let s_x = array![[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]];
        let s_z = array![[1.0, -1.0], [-1.0, 1.0]];
        let kron = |a: &Array2<f64>, b: &Array2<f64>| {
            let (ra, ca) = a.dim();
            let (rb, cb) = b.dim();
            Array2::from_shape_fn((ra * rb, ca * cb), |(i, j)| {
                a[[i / rb, j / cb]] * b[[i % rb, j % cb]]
            })
        };
        let penalties = [
            active_penalty(
                kron(&s_x, &Array2::eye(2)),
                4,
                2,
                0,
                PenaltySource::TensorMarginal { dim: 0 },
            ),
            active_penalty(
                kron(&Array2::eye(3), &s_z),
                3,
                3,
                1,
                PenaltySource::TensorMarginal { dim: 1 },
            ),
        ];
        assert_eq!(joint_unpenalized_dim(6, &penalties), 1);
    }

    #[test]
    #[should_panic(expected = "on a 2-coefficient term")]
    fn a_penalty_block_of_the_wrong_shape_is_a_construction_defect() {
        let full: Array2<f64> = array![[0.0, 0.0], [0.0, 1.0]];
        let wrong: Array2<f64> = array![[1.0]];
        let penalties = [
            active_penalty(full, 1, 1, 0, PenaltySource::Primary),
            active_penalty(wrong, 1, 0, 1, PenaltySource::TensorMarginal { dim: 0 }),
        ];
        joint_unpenalized_dim(2, &penalties);
    }
}

#[cfg(test)]
mod spatial_psi_bound_coordinate_tests {
    use super::*;
    use crate::basis::{MaternIdentifiability, MaternNu};
    use ndarray::array;

    fn frozen_matern_bounds(theta: f64, dilation: f64) -> (f64, f64) {
        let source = array![
            [-1.7, -0.4],
            [-1.1, 0.8],
            [-0.2, -1.3],
            [0.5, 1.6],
            [1.4, -0.7],
            [2.1, 0.5],
        ];
        let (cos_theta, sin_theta) = (theta.cos(), theta.sin());
        let mut data = Array2::<f64>::zeros(source.raw_dim());
        for row in 0..source.nrows() {
            let x = source[[row, 0]];
            let y = source[[row, 1]];
            data[[row, 0]] = dilation * (cos_theta * x - sin_theta * y);
            data[[row, 1]] = dilation * (sin_theta * x + cos_theta * y);
        }
        let input_scale = estimate_isotropic_scale(data.view()).expect("isotropic input scale");
        let mut centers = data.clone();
        input_scale.standardize(&mut centers);
        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::UserProvided(centers),
                        length_scale: crate::basis::MaternLengthScale::fixed(1.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: Some(input_scale),
                },
                shape: ShapeConstraint::None.into(),
                joint_null_rotation: None,
            }],
            level: Default::default(),
        };
        spatial_term_psi_bounds(data.view(), &spec, 0).expect("finite spatial ψ bounds")
    }

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() <= 1e-12,
            "coordinate-equivalent bounds differ: left={left:.16e}, right={right:.16e}"
        );
    }

    /// The search box a κ optimizer is handed must contain the length scale it
    /// is seeded at and graded against (#2454).
    ///
    /// Stated as containment rather than as a numeric window, because the point
    /// is not where the edge lands — it is that `clamp_to_bounds` has nothing to
    /// do. A window that excludes the incumbent makes `min` over the box free to
    /// return something strictly worse than the incumbent, which is exactly the
    /// "optimizing κ made the fit worse" refusal the monotone fixtures reported
    /// as a solver failure.
    ///
    /// Both directions are pinned: an incumbent far OUTSIDE the geometry window
    /// must be inside the search box, and an incumbent inside it must not move
    /// the box at all (widened, never narrowed, and never gratuitously).
    #[test]
    fn psi_search_box_contains_the_incumbent_length_scale_2454() {
        let source = array![
            [-1.7, -0.4],
            [-1.1, 0.8],
            [-0.2, -1.3],
            [0.5, 1.6],
            [1.4, -0.7],
            [2.1, 0.5],
        ];
        let box_for = |length_scale: f64| -> ((f64, f64), (f64, f64)) {
            let input_scale =
                estimate_isotropic_scale(source.view()).expect("isotropic input scale");
            let mut centers = source.clone();
            input_scale.standardize(&mut centers);
            let spec = TermCollectionSpec {
                linear_terms: Vec::new(),
                random_effect_terms: Vec::new(),
                smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                    name: "matern".to_string(),
                    basis: SmoothBasisSpec::Matern {
                        feature_cols: vec![0, 1],
                        spec: MaternBasisSpec {
                            periodic: None,
                            center_strategy: CenterStrategy::UserProvided(centers),
                            length_scale: crate::basis::MaternLengthScale::fixed(length_scale),
                            nu: MaternNu::FiveHalves,
                            include_intercept: false,
                            double_penalty: true,
                            identifiability: MaternIdentifiability::CenterSumToZero,
                            aniso_log_scales: None,
                        },
                        input_scale: Some(input_scale),
                    },
                    shape: ShapeConstraint::None.into(),
                    joint_null_rotation: None,
                }],
                level: Default::default(),
            };
            let geometry =
                spatial_term_psi_bounds(source.view(), &spec, 0).expect("finite geometry window");
            let search =
                spatial_term_psi_search_box(source.view(), &spec, 0).expect("finite search box");
            (geometry, search)
        };

        // An incumbent far past the long-range edge of the geometry window —
        // #2454's fixture shape, where `length_scale = 12` sat about six data
        // diameters out. The geometry window does not depend on the incumbent,
        // so its own edge places the fixture one nat outside it.
        let (unit_geometry, _) = box_for(1.0);
        let far = (1.0 - unit_geometry.0).exp();
        let (geometry, search) = box_for(far);
        let psi_far = -far.ln();
        assert!(
            psi_far < geometry.0,
            "fixture must place the incumbent OUTSIDE the geometry window, got \
             psi={psi_far} against [{}, {}]",
            geometry.0,
            geometry.1
        );
        assert!(
            search.0 <= psi_far && psi_far <= search.1,
            "the search box [{}, {}] must contain the incumbent psi={psi_far}; a seed \
             the box excludes is projected onto its edge and the optimum is then taken \
             over a set that does not contain the point it is graded against (#2454)",
            search.0,
            search.1
        );
        assert!(
            search.1 == geometry.1 && search.0 <= geometry.0,
            "the search box must be the geometry window WIDENED, never narrowed: \
             geometry=[{}, {}] search=[{}, {}]",
            geometry.0,
            geometry.1,
            search.0,
            search.1
        );

        // An incumbent already inside the window must leave the box untouched.
        let (geometry_mid, search_mid) = box_for((-0.5 * (geometry.0 + geometry.1)).exp());
        assert!(
            search_mid == geometry_mid,
            "an incumbent inside the geometry window must not move the search box: \
             geometry=[{}, {}] search=[{}, {}]",
            geometry_mid.0,
            geometry_mid.1,
            search_mid.0,
            search_mid.1
        );
    }

    #[test]
    fn standardized_center_bounds_return_to_original_units_under_rotation_and_scaling() {
        let base = frozen_matern_bounds(0.0, 1.0);
        let rotated = frozen_matern_bounds(0.61, 1.0);
        assert_close(rotated.0, base.0);
        assert_close(rotated.1, base.1);

        let dilation = 4.0_f64;
        let rotated_scaled = frozen_matern_bounds(0.61, dilation);
        let expected_shift = dilation.ln();
        assert_close(rotated_scaled.0, base.0 - expected_shift);
        assert_close(rotated_scaled.1, base.1 - expected_shift);
    }
}

#[cfg(test)]
mod tensor_function_space_runtime_tests {
    use super::*;
    use crate::basis::{
        BSplineBoundaryConditions, BSplineEndpointBoundaryCondition, OneDimensionalBoundary,
    };
    use ndarray::array;

    fn marginal() -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 2,
            penalty_order: 1,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 2,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: BSplineBoundaryConditions::default(),
        }
    }

    #[test]
    fn function_space_tensor_ridge_uses_exact_canonical_runtime() {
        let data = array![
            [0.00, 0.13],
            [0.15, 0.82],
            [0.29, 0.37],
            [0.43, 0.95],
            [0.58, 0.21],
            [0.71, 0.66],
            [0.86, 0.48],
            [1.00, 0.04]
        ];
        let mut spec = TensorBSplineSpec {
            marginalspecs: vec![marginal(), marginal()],
            periods: Vec::new(),
            double_penalty: true,
            identifiability: TensorBSplineIdentifiability::None,
            penalty_decomposition: TensorBSplinePenaltyDecomposition::MarginalKroneckerSum,
        };
        let built = build_tensor_bspline_basis(data.view(), &[0, 1], &spec, true)
            .expect("double-penalty tensor basis");
        assert!(
            built
                .active_penalties
                .iter()
                .any(|penalty| { matches!(penalty.info.source, PenaltySource::TensorGlobalRidge) })
        );

        spec.double_penalty = false;
        let singly_penalized = build_tensor_bspline_basis(data.view(), &[0, 1], &spec, true)
            .expect("single-penalty tensor basis");
        // Each margin block is `S_dim ⊗ G_other / 1ᵀ G_other 1` (#1561, SPEC rule 5).
        let mut margin_blocks = 0usize;
        for penalty in &singly_penalized.active_penalties {
            let PenaltySource::TensorMarginal { dim } = &penalty.info.source else {
                continue;
            };
            margin_blocks += 1;
            let factors = penalty
                .info
                .kronecker_factors
                .as_ref()
                .expect("a tensor margin block keeps its Kronecker factors");
            assert_eq!(factors.len(), 2);
            let gram = &factors[1 - *dim];
            let measure = gram.sum();
            assert!(
                (measure - 1.0).abs() <= gam_linalg::roundoff::accumulation_growth(2 * gram.len()),
                "margin {dim}'s other-margin Gram is not averaged over its domain: 1ᵀG1 = {measure}"
            );
        }
        assert_eq!(margin_blocks, 2, "one penalty block per margin");
    }

    fn cubic_marginal() -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 2,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: BSplineBoundaryConditions::default(),
        }
    }

    fn physical_null_ridges(built: &BasisBuildResult) -> Vec<Array2<f64>> {
        built
            .active_penalties
            .iter()
            .filter(|penalty| matches!(penalty.info.source, PenaltySource::TensorGlobalRidge))
            .map(|penalty| penalty.matrix.mapv(|v| v * penalty.info.normalization_scale))
            .collect()
    }

    fn least_squares(design: &Array2<f64>, target: &Array1<f64>) -> Array1<f64> {
        use gam_linalg::faer_ndarray::FaerEigh;
        let normal = design.t().dot(design);
        let (evals, evecs) = FaerEigh::eigh(&normal, faer::Side::Lower).expect("normal eigh");
        let projected = evecs.t().dot(&design.t().dot(target));
        evecs.dot(&(&projected / &evals))
    }

    /// #1561: the tensor double penalty used to be ONE ridge over the whole
    /// joint polynomial null, so REML shrank a supported x trend together with
    /// absent z and x·z trends. Each functional-ANOVA block of that null (under
    /// the domain measure) now carries its own REML coordinate, and each block
    /// ridge measures exactly the integrated square of its own block function and
    /// nothing of the others, in the raw chart and in every identifiability chart.
    #[test]
    fn tensor_null_ridge_gives_each_retained_anova_block_its_own_coordinate() {
        let side = 12;
        let data = Array2::from_shape_fn((side * side, 2), |(row, col)| {
            let index = if col == 0 { row / side } else { row % side };
            index as f64 / (side - 1) as f64
        });
        let block_functions: [(fn(f64, f64) -> f64, f64); 4] = [
            (|_, _| 1.0, 1.0),
            (|x, _| x - 0.5, 1.0 / 12.0),
            (|_, z| z - 0.5, 1.0 / 12.0),
            (|x, z| (x - 0.5) * (z - 0.5), 1.0 / 144.0),
        ];
        let charts = [
            (TensorBSplineIdentifiability::None, vec![0usize, 1, 2, 3]),
            (TensorBSplineIdentifiability::SumToZero, vec![1, 2, 3]),
            (TensorBSplineIdentifiability::MarginalSumToZero, vec![3]),
        ];
        for (identifiability, retained) in charts {
            let label = format!("{identifiability:?}");
            let spec = TensorBSplineSpec {
                marginalspecs: vec![cubic_marginal(), cubic_marginal()],
                periods: Vec::new(),
                double_penalty: true,
                identifiability,
                penalty_decomposition: TensorBSplinePenaltyDecomposition::MarginalKroneckerSum,
            };
            let built = build_tensor_bspline_basis(data.view(), &[0, 1], &spec, true)
                .expect("double-penalty tensor basis");
            let ridges = physical_null_ridges(&built);
            assert_eq!(
                ridges.len(),
                retained.len(),
                "{label}: one null ridge per retained ANOVA block"
            );
            let design = built.design.to_dense();
            for (&block, ridge_owner) in retained.iter().zip(0..) {
                let (function, energy) = block_functions[block];
                let target = Array1::from_iter(data.rows().into_iter().map(|p| function(p[0], p[1])));
                let beta = least_squares(&design, &target);
                let residual = &design.dot(&beta) - &target;
                assert!(
                    residual.iter().all(|r| r.abs() < 1e-9),
                    "{label}: block {block} is representable in the chart"
                );
                for (ridge_index, ridge) in ridges.iter().enumerate() {
                    let measured = beta.dot(&ridge.dot(&beta));
                    let expected = if ridge_index == ridge_owner { energy } else { 0.0 };
                    assert!(
                        (measured - expected).abs() <= 1e-8 * energy,
                        "{label}: ridge {ridge_index} on block {block} measured {measured:.6e}, \
                         expected {expected:.6e}"
                    );
                }
            }
        }
    }

    /// A frozen tensor chart is the basis sum-to-zero chart composed with every
    /// later collection transform. A term residualized against an owner smooth
    /// (`s(x) + te(x, z)`) and saved before the collection gauge became
    /// orthonormal carries a whitener, whose columns span several decades of
    /// scale. Rebuilding the null-block ridges in that chart once read the
    /// chart's null space off its badly scaled primary penalty, found spurious
    /// null directions, and failed every prediction with "tensor null blocks span
    /// 4 of the chart's 12 null directions". The null space is a property of the
    /// chart's column span, so each ridge must still measure exactly its own
    /// block function.
    #[test]
    fn tensor_null_ridges_survive_a_badly_scaled_frozen_chart() {
        let side = 12;
        let data = Array2::from_shape_fn((side * side, 2), |(row, col)| {
            let index = if col == 0 { row / side } else { row % side };
            index as f64 / (side - 1) as f64
        });
        let block_functions: [(fn(f64, f64) -> f64, f64); 3] = [
            (|x, _| x - 0.5, 1.0 / 12.0),
            (|_, z| z - 0.5, 1.0 / 12.0),
            (|x, z| (x - 0.5) * (z - 0.5), 1.0 / 144.0),
        ];
        let mut spec = TensorBSplineSpec {
            marginalspecs: vec![cubic_marginal(), cubic_marginal()],
            periods: Vec::new(),
            double_penalty: true,
            identifiability: TensorBSplineIdentifiability::SumToZero,
            penalty_decomposition: TensorBSplinePenaltyDecomposition::MarginalKroneckerSum,
        };
        let centered = build_tensor_bspline_basis(data.view(), &[0, 1], &spec, true)
            .expect("sum-to-zero tensor basis");
        let BasisMetadata::TensorBSpline {
            identifiability_transform: Some(sum_to_zero),
            ..
        } = &centered.metadata
        else {
            panic!("a sum-to-zero tensor records its chart");
        };
        // A whitener-shaped chart change `H D`: an orthogonal reflection times
        // column scales spanning six decades.
        let q = sum_to_zero.ncols();
        let v = Array1::from_iter((0..q).map(|i| (i + 1) as f64));
        let reflection = Array2::<f64>::eye(q)
            - &(v.view().insert_axis(Axis(1)).dot(&v.view().insert_axis(Axis(0)))
                * (2.0 / v.dot(&v)));
        let scales = Array1::from_iter(
            (0..q).map(|j| 10f64.powf(3.0 * (2.0 * j as f64 / (q - 1) as f64 - 1.0))),
        );
        let whitener = &reflection * &scales.view().insert_axis(Axis(0));
        spec.identifiability = TensorBSplineIdentifiability::FrozenTransform {
            transform: sum_to_zero.dot(&whitener),
        };
        let built = build_tensor_bspline_basis(data.view(), &[0, 1], &spec, true)
            .expect("a frozen chart with badly scaled columns rebuilds");
        let ridges = physical_null_ridges(&built);
        assert_eq!(ridges.len(), block_functions.len());
        // Coefficients are solved in the well-conditioned sum-to-zero chart and
        // carried into the frozen one exactly: `(H D)⁻¹ = D⁻¹ H`.
        let design = centered.design.to_dense();
        for (owner, (function, energy)) in block_functions.iter().enumerate() {
            let target = Array1::from_iter(data.rows().into_iter().map(|p| function(p[0], p[1])));
            let beta = &reflection.dot(&least_squares(&design, &target)) / &scales;
            for (ridge_index, ridge) in ridges.iter().enumerate() {
                let measured = beta.dot(&ridge.dot(&beta));
                let expected = if ridge_index == owner { *energy } else { 0.0 };
                assert!(
                    (measured - expected).abs() <= 1e-8 * energy,
                    "ridge {ridge_index} on block {owner} measured {measured:.6e}, \
                     expected {expected:.6e}"
                );
            }
        }
    }

    #[test]
    fn tensor_nonzero_anchor_is_rejected_before_its_affine_lift_can_be_dropped() {
        let data = array![[0.0, 0.0], [0.25, 0.75], [0.75, 0.25], [1.0, 1.0]];
        let mut anchored = marginal();
        anchored.boundary_conditions.left =
            BSplineEndpointBoundaryCondition::Anchored { value: 1.25 };
        let spec = TensorBSplineSpec {
            marginalspecs: vec![anchored, marginal()],
            periods: Vec::new(),
            double_penalty: false,
            identifiability: TensorBSplineIdentifiability::None,
            penalty_decomposition: TensorBSplinePenaltyDecomposition::MarginalKroneckerSum,
        };

        let error = build_tensor_bspline_basis(data.view(), &[0, 1], &spec, true)
            .expect_err("a tensor margin cannot silently discard an inhomogeneous lift");
        let message = error.to_string();
        assert!(message.contains("TensorBSpline margin 0"));
        assert!(message.contains("non-zero endpoint anchor"));
        assert!(message.contains("explicit model offset"));
    }
}

#[cfg(test)]
mod random_effect_signed_zero_tests {
    use super::{RandomEffectTermSpec, build_random_effect_block};
    use ndarray::array;

    fn spec() -> RandomEffectTermSpec {
        RandomEffectTermSpec {
            name: "g".to_string(),
            feature_col: 0,
            frozen_levels: None,
            lenient_unseen: true,
        }
    }

    #[test]
    fn signed_zero_rows_share_one_group() {
        // A column mixing +0.0 and -0.0 for the physically same group must
        // intern as ONE level, and every row (either spelling) must resolve to
        // that single group column — the #2145 fit-side regression.
        let data = array![[-0.0_f64], [0.0], [1.0], [-0.0], [1.0]];
        let block = build_random_effect_block(data.view(), &spec()).unwrap();
        assert_eq!(
            block.num_groups, 2,
            "0.0/-0.0 must not split into two groups"
        );
        // Rows 0,1,3 are the same group; rows 2,4 the other.
        assert_eq!(block.group_ids[0], block.group_ids[1]);
        assert_eq!(block.group_ids[0], block.group_ids[3]);
        assert_eq!(block.group_ids[2], block.group_ids[4]);
        assert_ne!(block.group_ids[0], block.group_ids[2]);
    }

    #[test]
    fn frozen_positive_zero_matches_negative_zero_row() {
        // A model frozen on +0.0 must resolve a -0.0 prediction row to the same
        // column — the #2145 predict-side regression that dropped the effect.
        let mut s = spec();
        s.frozen_levels = Some(vec![0.0_f64.to_bits(), 1.0_f64.to_bits()]);
        let data = array![[-0.0_f64], [1.0]];
        let block = build_random_effect_block(data.view(), &s).unwrap();
        assert_eq!(
            block.group_ids[0],
            Some(0),
            "-0.0 must match the +0.0 column"
        );
        assert_eq!(block.group_ids[1], Some(1));
    }

    #[test]
    fn frozen_negative_zero_matches_positive_zero_row() {
        // The symmetric direction: a legacy model interned on -0.0 (pre-fix)
        // must still resolve a +0.0 prediction row after canonicalization.
        let mut s = spec();
        s.frozen_levels = Some(vec![(-0.0_f64).to_bits(), 1.0_f64.to_bits()]);
        let data = array![[0.0_f64], [1.0]];
        let block = build_random_effect_block(data.view(), &s).unwrap();
        assert_eq!(
            block.group_ids[0],
            Some(0),
            "+0.0 must match the -0.0 column"
        );
    }

    // ---- #2137: fixed factor (`factor(g)`) strict-unseen enforcement --------

    fn fixed_factor_spec() -> RandomEffectTermSpec {
        // A numeric-coded `factor(year)`: full one-hot,
        // FIXED (`lenient_unseen=false`), vocabulary pinned at fit.
        let mut s = spec();
        s.name = "year".to_string();
        s.lenient_unseen = false;
        s
    }

    #[test]
    fn fixed_factor_rejects_unseen_numeric_level_at_predict() {
        // The numeric-coded `factor(year)` gap (#2137): the column reaches the
        // operator as plain numbers (no categorical schema to pre-filter it), so
        // the operator that owns the frozen vocabulary must reject an unseen
        // code rather than encode an all-zero (centering-point) row.
        let mut s = fixed_factor_spec();
        s.frozen_levels = Some(vec![2000.0_f64.to_bits(), 2001.0_f64.to_bits()]);
        let data = array![[2000.0_f64], [1999.0]];
        let err = build_random_effect_block(data.view(), &s)
            .expect_err("an unseen fixed-factor level must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("unseen level"),
            "message must name the defect: {msg}"
        );
        assert!(
            msg.contains("1999"),
            "message must name the integer level (not 1999.0): {msg}"
        );
        assert!(msg.contains("year"), "message must name the column: {msg}");
    }

    #[test]
    fn fixed_factor_accepts_seen_numeric_levels_at_predict() {
        // Control: every seen level still resolves; strictness rejects only the
        // genuinely out-of-vocabulary code.
        let mut s = fixed_factor_spec();
        s.frozen_levels = Some(vec![2000.0_f64.to_bits(), 2001.0_f64.to_bits()]);
        let data = array![[2001.0_f64], [2000.0]];
        let block = build_random_effect_block(data.view(), &s).unwrap();
        assert_eq!(block.group_ids[0], Some(1));
        assert_eq!(block.group_ids[1], Some(0));
    }

    #[test]
    fn fixed_factor_at_fit_time_derives_vocabulary_and_never_false_rejects() {
        // At FIT (`frozen_levels=None`) the vocabulary is derived from this very
        // data, so no row is unseen — the strict guard must not fire even though
        // the factor is strict.
        let mut s = fixed_factor_spec();
        s.frozen_levels = None;
        let data = array![[2000.0_f64], [2001.0], [2002.0], [2000.0]];
        let block = build_random_effect_block(data.view(), &s)
            .expect("fit-time build must not reject its own levels");
        assert_eq!(block.num_groups, 3);
    }

    #[test]
    fn random_effect_still_tolerates_unseen_numeric_level() {
        // Non-regression: a lenient random effect (`group`/`re`/`s(bs="re")`)
        // encodes an unseen level as an all-zero (population-mean) row, NOT a
        // rejection — the held-out-group contract (#2102) is unchanged.
        let mut s = spec(); // lenient_unseen = true
        s.frozen_levels = Some(vec![2000.0_f64.to_bits(), 2001.0_f64.to_bits()]);
        let data = array![[2000.0_f64], [1999.0]];
        let block = build_random_effect_block(data.view(), &s)
            .expect("a random effect tolerates unseen levels");
        assert_eq!(block.group_ids[0], Some(0));
        assert_eq!(
            block.group_ids[1], None,
            "unseen level → population mean, not a reject"
        );
    }
}

#[cfg(test)]
mod pca_function_mass_tests {
    use super::{PenaltySource, build_pca_smooth_basis, parse_f64_2d_npy_header};
    use ndarray::{Array1, Array2, array};
    use std::io::Write;
    use std::path::PathBuf;

    fn quadratic_form(matrix: &Array2<f64>, coefficients: &Array1<f64>) -> f64 {
        coefficients.dot(&matrix.dot(coefficients))
    }

    fn assert_close(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= 1e-11 * scale,
            "values differ: left={left:.16e}, right={right:.16e}"
        );
    }

    fn write_f64_npy(scores: &Array2<f64>) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "gam_terms_pca_function_mass_{}.npy",
            std::process::id()
        ));
        let mut header = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}), }}",
            scores.nrows(),
            scores.ncols()
        );
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let header_len = u16::try_from(header.len()).expect("test .npy header fits u16");

        let mut file = std::fs::File::create(&path).expect("create test .npy");
        file.write_all(b"\x93NUMPY").expect("write .npy magic");
        file.write_all(&[1, 0]).expect("write .npy version");
        file.write_all(&header_len.to_le_bytes())
            .expect("write .npy header length");
        file.write_all(header.as_bytes())
            .expect("write .npy header");
        for &value in scores {
            file.write_all(&value.to_le_bytes())
                .expect("write .npy score");
        }
        path
    }

    fn npy_v1_bytes(mut header: String) -> Vec<u8> {
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let header_len = u16::try_from(header.len()).expect("test header fits v1");
        let mut bytes = b"\x93NUMPY".to_vec();
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes
    }

    #[test]
    fn npy_header_parser_uses_exact_ast_fields_2293() {
        let path = PathBuf::from("scores.npy");
        let bytes = npy_v1_bytes(
            "{'shape':(3, 2), 'note':'True', 'descr':'<f8', 'fortran_order':False,}".to_string(),
        );
        let (offset, rows, cols) =
            parse_f64_2d_npy_header(&bytes, &path).expect("valid reordered header");
        assert_eq!((rows, cols), (3, 2));
        assert_eq!(offset, bytes.len());

        for header in [
            "{'descr':'<f8','fortran_order':True,'shape':(3,2),}",
            "{'descr':'>f8','fortran_order':False,'shape':(3,2),}",
            "{'descr':'<f8','shape':(3,2),}",
            "{'descr':'<f8','fortran_order':'False','shape':(3,2),}",
            "{'descr':'<f8','fortran_order':False,'shape':(6,),}",
        ] {
            let invalid = npy_v1_bytes(header.to_string());
            assert!(
                parse_f64_2d_npy_header(&invalid, &path).is_err(),
                "{header}"
            );
        }
    }

    #[test]
    fn pca_penalty_quadratic_equals_empirical_fitted_function_norm() {
        let data = array![[1.0, 2.0], [-1.0, 0.5], [2.0, -0.5], [0.25, -1.5]];
        let basis = array![[1.0, 0.5], [-0.25, 2.0]];
        let built = build_pca_smooth_basis(data.view(), &[0, 1], &basis, false, None, None, 2)
            .expect("full-rank PCA basis");
        let coefficients = array![0.7, -1.2];
        let design = built.design.to_dense();
        let fitted = design.dot(&coefficients);
        let expected = fitted.dot(&fitted) / fitted.len() as f64;
        let actual = quadratic_form(&built.active_penalties[0].matrix, &coefficients);

        assert_close(actual, expected);
        assert_eq!(built.active_penalties[0].nullity, 0);
        assert_eq!(
            built.active_penalties[0].info.source,
            PenaltySource::OperatorMass
        );
    }

    #[test]
    fn pca_function_mass_is_invariant_to_nonorthogonal_score_reparameterization() {
        let scores = array![[1.0, 2.0], [-1.0, 0.5], [2.0, -0.5], [0.25, -1.5]];
        let identity = Array2::<f64>::eye(2);
        // An invertible scale-plus-shear, deliberately not orthogonal.
        let transform = array![[2.0, 0.5], [0.0, 0.25]];
        let base_coefficients = array![0.8, -1.1];
        // transform * transformed_coefficients == base_coefficients.
        let transformed_coefficients = array![1.5, -4.4];

        let base = build_pca_smooth_basis(scores.view(), &[0, 1], &identity, false, None, None, 2)
            .expect("base PCA chart");
        let transformed =
            build_pca_smooth_basis(scores.view(), &[0, 1], &transform, false, None, None, 2)
                .expect("reparameterized PCA chart");

        let fitted_base = base.design.to_dense().dot(&base_coefficients);
        let fitted_transformed = transformed.design.to_dense().dot(&transformed_coefficients);
        for (&left, &right) in fitted_base.iter().zip(fitted_transformed.iter()) {
            assert_close(left, right);
        }
        assert_close(
            quadratic_form(&base.active_penalties[0].matrix, &base_coefficients),
            quadratic_form(
                &transformed.active_penalties[0].matrix,
                &transformed_coefficients,
            ),
        );
    }

    #[test]
    fn rank_deficient_pca_score_design_is_rejected() {
        let scores = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
        let result = build_pca_smooth_basis(
            scores.view(),
            &[0, 1],
            &Array2::<f64>::eye(2),
            false,
            None,
            None,
            2,
        );
        let err = result.err().expect("zero score column must be rejected");
        let message = err.to_string();
        assert!(
            message.contains("rank deficient"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("rank 1 < 2"),
            "missing RRQR evidence: {message}"
        );
    }

    #[test]
    fn lazy_and_dense_pca_function_mass_penalties_match() {
        let scores = array![[1.0, 2.0], [-1.0, 0.5], [2.0, -0.5], [0.25, -1.5]];
        let path = write_f64_npy(&scores);
        let dense = build_pca_smooth_basis(
            scores.view(),
            &[0, 1],
            &Array2::<f64>::eye(2),
            false,
            None,
            None,
            2,
        )
        .expect("dense PCA basis");
        let lazy_data = Array2::<f64>::zeros((scores.nrows(), 0));
        let lazy = build_pca_smooth_basis(
            lazy_data.view(),
            &[],
            &Array2::<f64>::zeros((0, scores.ncols())),
            false,
            None,
            Some(&path),
            2,
        )
        .expect("lazy PCA basis");
        std::fs::remove_file(&path).expect("remove test .npy");

        for (&left, &right) in dense.active_penalties[0]
            .matrix
            .iter()
            .zip(lazy.active_penalties[0].matrix.iter())
        {
            assert_close(left, right);
        }
        for (&left, &right) in dense
            .design
            .to_dense()
            .iter()
            .zip(lazy.design.to_dense().iter())
        {
            assert_close(left, right);
        }
    }
}

#[cfg(test)]
mod factor_smooth_null_component_tests {
    use super::*;
    use crate::basis::{BasisOptions, BasisWorkspace, Dense, KnotSource};
    use gam_linalg::faer_ndarray::FaerCholesky;
    use ndarray::{Array1, Array2};

    const DEGREE: usize = 3;

    /// Cubic knots with deliberately asymmetric interior breaks, so no symmetry
    /// of the coefficient chart lines up with the function-space split.
    fn knots() -> Array1<f64> {
        Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.2, 0.45, 0.6, 1.0, 1.0, 1.0, 1.0])
    }

    /// `double_penalty` as the DSL defaults it: on for `fs` (it gates the
    /// per-component null penalties), off for `sz` (whose pooled null-function
    /// penalties are emitted unconditionally).
    fn marginal(double_penalty: bool) -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: DEGREE,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Provided(knots()),
            double_penalty,
            identifiability: BSplineIdentifiability::None,
            boundary: crate::basis::OneDimensionalBoundary::Open,
            boundary_conditions: crate::basis::BSplineBoundaryConditions::default(),
        }
    }

    fn grouped_data(n_levels: usize) -> Array2<f64> {
        let n = 60;
        Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                i as f64 / (n - 1) as f64
            } else {
                (i % n_levels) as f64
            }
        })
    }

    fn build(flavour: FactorSmoothFlavour, n_levels: usize) -> LocalSmoothTermBuild {
        let spec = FactorSmoothSpec {
            continuous_cols: vec![0],
            group_col: 1,
            marginal: marginal(matches!(flavour, FactorSmoothFlavour::Fs { .. })),
            flavour,
            group_frozen_levels: None,
            frozen_global_orthogonality: None,
            adaptive: false,
        };
        build_factor_smooth(
            grouped_data(n_levels).view(),
            &spec,
            "null_components",
            &mut BasisWorkspace::new(),
        )
        .expect("build factor smooth")
    }

    /// Coefficients of the constant function and of the linear function
    /// centred at the modeling interval's midpoint. B-splines reproduce `x`
    /// with their Greville abscissae as coefficients.
    fn constant_and_centred_linear() -> (Array1<f64>, Array1<f64>) {
        let knots = knots();
        let p = knots.len() - DEGREE - 1;
        let midpoint = 0.5 * (knots[DEGREE] + knots[p]);
        let linear = Array1::from_shape_fn(p, |i| {
            (1..=DEGREE).map(|r| knots[i + r]).sum::<f64>() / DEGREE as f64 - midpoint
        });
        (Array1::ones(p), linear)
    }

    fn charge(penalty: &Array2<f64>, coefficients: &Array1<f64>) -> f64 {
        coefficients.dot(&penalty.dot(coefficients))
    }

    /// The two null components must be the constant and the centred linear
    /// function, each blind to the other: one charges the constant and nothing
    /// for the centred line, the other the reverse.
    fn assert_constant_and_centred_linear(level_blocks: &[Array2<f64>]) {
        let (constant, linear) = constant_and_centred_linear();
        assert_eq!(level_blocks.len(), 2, "one null component per null dimension");
        let charges: Vec<(f64, f64)> = level_blocks
            .iter()
            .map(|block| (charge(block, &constant), charge(block, &linear)))
            .collect();
        let intercept = charges
            .iter()
            .position(|&(on_constant, on_linear)| on_constant > on_linear)
            .expect("a component charging the constant");
        let (on_constant, leak_to_linear) = charges[intercept];
        let (leak_to_constant, on_linear) = charges[1 - intercept];
        assert!(
            leak_to_linear <= 1e-10 * on_constant,
            "the intercept component charges the centred line {leak_to_linear:e} (constant {on_constant:e})"
        );
        assert!(
            leak_to_constant <= 1e-10 * on_linear,
            "the slope component charges the constant {leak_to_constant:e} (line {on_linear:e})"
        );
    }

    #[test]
    fn fs_null_penalties_charge_the_constant_and_the_centred_line_separately() {
        let built = build(FactorSmoothFlavour::Fs {}, 2);
        let p = knots().len() - DEGREE - 1;
        let nulls: Vec<Array2<f64>> = built.active_penalties[1..]
            .iter()
            .map(|penalty| penalty.matrix.slice(s![0..p, 0..p]).to_owned())
            .collect();
        assert_constant_and_centred_linear(&nulls);
    }

    #[test]
    fn sz_null_penalties_charge_the_constant_and_the_centred_line_separately() {
        let n_levels = 3;
        let built = build(FactorSmoothFlavour::Sz, n_levels);
        let p = knots().len() - DEGREE - 1;
        let nulls: Vec<Array2<f64>> = built
            .active_penalties
            .iter()
            .filter(|penalty| matches!(penalty.info.source, PenaltySource::DoublePenaltyNullspace))
            .map(|penalty| penalty.matrix.slice(s![0..p, 0..p]).to_owned())
            .collect();
        assert_constant_and_centred_linear(&nulls);
    }

    fn marginal_metrics() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let knots = knots();
        (
            crate::basis::bspline_derivative_penalty_matrix(knots.view(), DEGREE, 2).expect("S"),
            crate::basis::bspline_function_gram(&knots, DEGREE).expect("G"),
            crate::basis::bspline_derivative_penalty_matrix(knots.view(), DEGREE, 1).expect("D1"),
        )
    }

    /// A deterministic, well-conditioned, far-from-orthogonal chart change.
    fn chart_change(p: usize) -> Array2<f64> {
        let lower = Array2::from_shape_fn((p, p), |(i, j)| match i.cmp(&j) {
            std::cmp::Ordering::Equal => 1.0 + 0.1 * i as f64,
            std::cmp::Ordering::Greater => 0.35 * ((3 * i + 5 * j) as f64).sin(),
            std::cmp::Ordering::Less => 0.0,
        });
        let upper = Array2::from_shape_fn((p, p), |(i, j)| match i.cmp(&j) {
            std::cmp::Ordering::Equal => 1.0,
            std::cmp::Ordering::Less => 0.4 * ((2 * i + 7 * j) as f64).cos(),
            std::cmp::Ordering::Greater => 0.0,
        });
        lower.dot(&upper)
    }

    fn congruence(t: &Array2<f64>, m: &Array2<f64>) -> Array2<f64> {
        t.t().dot(m).dot(t)
    }

    fn components(s: &Array2<f64>, g: &Array2<f64>, d: &Array2<f64>) -> Vec<Array2<f64>> {
        crate::basis::null_function_mass_components(s, g, || Ok(d.clone()), "test")
            .expect("null components")
            .iter()
            .map(|factor| factor.t().dot(factor))
            .collect()
    }

    fn max_abs(m: &Array2<f64>) -> f64 {
        m.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    #[test]
    fn null_component_penalties_transform_by_congruence_with_the_chart() {
        let (s, g, d) = marginal_metrics();
        let t = chart_change(s.nrows());
        let reference = components(&s, &g, &d);
        let moved = components(
            &congruence(&t, &s),
            &congruence(&t, &g),
            &congruence(&t, &d),
        );
        assert_eq!(reference.len(), 2);
        assert_eq!(moved.len(), reference.len());
        for (k, (r, r_moved)) in reference.iter().zip(&moved).enumerate() {
            let expected = congruence(&t, r);
            let error = max_abs(&(&expected - r_moved));
            assert!(
                error <= 1e-9 * max_abs(&expected),
                "component {k} is not covariant: error {error:e}"
            );
        }
    }

    fn penalized_fit(x: &Array2<f64>, y: &Array1<f64>, penalty: &Array2<f64>) -> Array1<f64> {
        let system = x.t().dot(x) + penalty;
        let factor = FaerCholesky::cholesky(&system, faer::Side::Lower).expect("SPD system");
        x.dot(&factor.solvevec(&x.t().dot(y)))
    }

    /// Reparameterizing only the null-space coordinates (`β = Tβ'` with `T`
    /// moving the null directions among themselves and into the range) leaves
    /// the wiggliness penalty unchanged. The fitted function at fixed smoothing
    /// parameters must not move: the null components are functions, so their
    /// penalties follow the chart.
    #[test]
    fn penalized_fit_is_invariant_to_null_space_reparameterization() {
        let (s, g, d) = marginal_metrics();
        let p = s.nrows();
        let n = 80;
        let xs = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y = xs.mapv(|x| 1.3 - 2.1 * x + 0.4 * (7.0 * x).sin());
        let (basis, _) = crate::basis::create_basis::<Dense>(
            xs.view(),
            KnotSource::Provided(knots().view()),
            DEGREE,
            BasisOptions::value(),
        )
        .expect("design");
        let x = (*basis).clone();

        // Null frame of S (constant and linear coefficient vectors), then a
        // non-orthogonal map that mixes the null coordinates and shears range
        // directions into them. `S T = S` because `S N = 0`.
        let (constant, linear) = constant_and_centred_linear();
        let mut null = Array2::<f64>::zeros((p, 2));
        null.column_mut(0).assign(&constant);
        null.column_mut(1).assign(&(&linear + 0.8 * &constant));
        let mix = ndarray::array![[0.7, 2.5], [-1.9, 0.4]];
        let shear = Array2::from_shape_fn((2, p), |(a, j)| 0.3 * ((a + 2 * j) as f64).sin());
        let t = Array2::<f64>::eye(p) + null.dot(&mix).dot(&shear);
        assert!(max_abs(&(congruence(&t, &s) - &s)) <= 1e-9 * max_abs(&s));

        let (lambda_s, lambdas) = (0.02, [3.0, 0.05]);
        let total = |s: &Array2<f64>, parts: &[Array2<f64>]| {
            parts
                .iter()
                .zip(lambdas)
                .fold(s.mapv(|v| lambda_s * v), |acc, (part, lambda)| acc + part.mapv(|v| lambda * v))
        };
        let reference = penalized_fit(&x, &y, &total(&s, &components(&s, &g, &d)));
        let (s_t, g_t, d_t) = (congruence(&t, &s), congruence(&t, &g), congruence(&t, &d));
        let moved = penalized_fit(&x.dot(&t), &y, &total(&s_t, &components(&s_t, &g_t, &d_t)));
        let error = (&reference - &moved).iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(error <= 1e-9, "fitted values moved by {error:e} under a null-space reparameterization");
    }

    /// Eigenvectors of one repeated slope-energy eigenvalue are one component;
    /// splitting them would be an eigensolver gauge.
    #[test]
    fn degenerate_null_functions_form_one_component() {
        let p = 4;
        let mut s = Array2::<f64>::zeros((p, p));
        s[[3, 3]] = 1.0;
        let g = Array2::<f64>::eye(p);
        let d = Array2::from_diag(&Array1::from(vec![0.0, 2.0, 2.0, 5.0]));
        let parts = components(&s, &g, &d);
        assert_eq!(parts.len(), 2);
        let ranks: Vec<usize> = parts
            .iter()
            .map(|part| (0..p).filter(|&i| part[[i, i]] > 0.5).count())
            .collect();
        assert_eq!(ranks, vec![1, 2]);
        assert!((parts[0][[0, 0]] - 1.0).abs() <= 1e-12);
        assert!((parts[1][[1, 1]] - 1.0).abs() <= 1e-12 && (parts[1][[2, 2]] - 1.0).abs() <= 1e-12);
    }

    /// The cubic-regression slope energy is exact on the functions it can
    /// represent: zero on the constant and `b − a` on `f(x) = x`, whose value
    /// coefficients are the knots themselves.
    #[test]
    fn cubic_regression_slope_energy_is_exact_on_linear_functions() {
        let cr_knots = Array1::from(vec![-0.4, 0.1, 0.35, 1.2, 2.0]);
        let d = crate::basis::cubic_regression_slope_energy(&cr_knots).expect("D1");
        let ones = Array1::<f64>::ones(cr_knots.len());
        assert!(charge(&d, &ones).abs() <= 1e-12);
        assert!((charge(&d, &cr_knots) - 2.4).abs() <= 1e-12);
    }
}

#[cfg(test)]
mod factor_smooth_heldout_group_tests {
    use super::*;
    use crate::basis::BasisWorkspace;
    use ndarray::{Array1, array};

    fn pinned_marginal() -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Provided(Array1::from(vec![
                0.0, 0.0, 0.0, 0.0, 0.25, 0.6, 1.0, 1.0, 1.0, 1.0,
            ])),
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: crate::basis::OneDimensionalBoundary::Open,
            boundary_conditions: crate::basis::BSplineBoundaryConditions::default(),
        }
    }

    fn factor_smooth_term(
        flavour: FactorSmoothFlavour,
        frozen: Option<Vec<u64>>,
    ) -> SmoothTermSpec {
        SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "fs_heldout".to_string(),
            basis: SmoothBasisSpec::FactorSmooth {
                spec: FactorSmoothSpec {
                    continuous_cols: vec![0],
                    group_col: 1,
                    marginal: pinned_marginal(),
                    flavour,
                    group_frozen_levels: frozen,
                    frozen_global_orthogonality: None,
                    adaptive: false,
                },
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }
    }

    const FROZEN_01: [f64; 2] = [0.0, 1.0];

    fn frozen_bits() -> Vec<u64> {
        FROZEN_01.iter().map(|v| v.to_bits()).collect()
    }

    /// #2365: in the frozen (predict/replay) context, a `bs="re"` row whose
    /// group is outside the training vocabulary must build with an all-zero
    /// row — zero fitted deviation, population prediction — instead of
    /// erroring before the random-effect operator can apply its held-out-group
    /// contract.
    #[test]
    fn re_heldout_group_row_is_zero_deviation() {
        let data = array![[0.1, 0.0], [0.5, 1.0], [0.9, 7.0]];
        let term = factor_smooth_term(FactorSmoothFlavour::Re, Some(frozen_bits()));
        let mut workspace = BasisWorkspace::default();
        let build = build_single_local_smooth_term(data.view(), &term, &mut workspace)
            .expect("a held-out group must not fail the bs=\"re\" design build");
        let dense = build
            .design
            .try_to_dense_by_chunks("heldout test")
            .expect("dense");
        assert!(
            dense.row(2).iter().all(|&v| v == 0.0),
            "unseen-group row must carry zero deviation across every group block, got {:?}",
            dense.row(2)
        );
        assert!(
            dense.row(0).iter().any(|&v| v != 0.0) && dense.row(1).iter().any(|&v| v != 0.0),
            "in-vocabulary rows must still populate their group blocks"
        );
    }

    /// The `fs` flavour estimates a per-level deviation FUNCTION — an unseen
    /// level has no zero-deviation population fallback — so the frozen-context
    /// build must stay strict (#2102/#2137 must not regress through #2365).
    #[test]
    fn fs_heldout_group_stays_strict() {
        let data = array![[0.1, 0.0], [0.5, 1.0], [0.9, 7.0]];
        let term = factor_smooth_term(
            FactorSmoothFlavour::Fs {},
            Some(frozen_bits()),
        );
        let mut workspace = BasisWorkspace::default();
        let err = match build_single_local_smooth_term(data.view(), &term, &mut workspace) {
            Ok(_) => panic!("fs must reject an unseen grouping level"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("unseen grouping level"),
            "fs unseen-level refusal must name the defect, got: {err}"
        );
    }
}

#[cfg(test)]
mod linear_term_contract_tests {
    use super::LinearTermSpec;

    #[test]
    fn missing_linear_double_penalty_deserializes_to_unpenalized_mle() {
        let term: LinearTermSpec = serde_json::from_str(r#"{"name":"x","feature_col":0}"#)
            .expect("minimal saved linear term");
        assert!(
            !term.double_penalty,
            "descriptor and formula defaults must both preserve parametric MLE semantics"
        );
    }
}

#[cfg(test)]
mod frozen_factor_level_collection_tests {
    use super::*;

    fn marginal() -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Provided(Array1::from(vec![
                0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0,
            ])),
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: crate::basis::OneDimensionalBoundary::Open,
            boundary_conditions: crate::basis::BSplineBoundaryConditions::default(),
        }
    }

    fn linear_gate(name: &str, col: usize, value: f64) -> LinearTermSpec {
        LinearTermSpec {
            name: name.to_string(),
            feature_col: col,
            feature_cols: Vec::new(),
            categorical_levels: vec![(col, value.to_bits())],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }
    }

    fn smooth(name: &str, basis: SmoothBasisSpec) -> SmoothTermSpec {
        SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: name.to_string(),
            basis,
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }
    }

    fn canonical_levels(values: &[f64]) -> Vec<u64> {
        values
            .iter()
            .map(|&value| gam_data::canonical_level_bits(value))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// gam#2787: a fixed categorical main effect is represented by a frozen
    /// random-effect block.  It must be visible even when the co-fitted smooth
    /// is a wholly unrelated numeric leaf; scanning `smooth_terms` alone made
    /// representative summary rows invent a non-level midpoint and erased the
    /// entire smooth significance table.
    #[test]
    fn categorical_main_effect_is_collected_outside_the_smooth_tree() {
        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "g".to_string(),
                feature_col: 1,
                frozen_levels: Some(vec![2.0_f64.to_bits(), 1.0_f64.to_bits()]),
                lenient_unseen: false,
            }],
            smooth_terms: vec![smooth(
                "s(x)",
                SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: marginal(),
                },
            )],
            level: Default::default(),
        };

        let levels = spec.frozen_factor_levels_by_col();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels.get(&1), Some(&canonical_levels(&[1.0, 2.0])));
        assert!(
            !levels.contains_key(&0),
            "numeric smooth axis is not a factor"
        );
    }

    /// Every place in `TermCollectionSpec` that can gate rows categorically
    /// contributes to the same canonical union, including factors nested under
    /// multiple smooth wrappers.  Numeric wrappers/coordinates remain absent.
    #[test]
    fn collector_unifies_linear_random_effect_and_nested_smooth_factor_carriers() {
        let nested = SmoothBasisSpec::ByVariable {
            by_col: 2,
            kind: BySmoothKind::Level {
                level_bits: 3.0_f64.to_bits(),
            },
            by: ByVariableSpec::Level {
                value_bits: 3.0_f64.to_bits(),
                label: "three".to_string(),
            },
            inner: Box::new(SmoothBasisSpec::FactorSumToZero {
                by_col: 3,
                levels: vec![5.0_f64.to_bits(), 4.0_f64.to_bits()],
                frozen_global_orthogonality: None,
                inner: Box::new(SmoothBasisSpec::BySmooth {
                    by_kind: ByVarKind::Factor {
                        feature_col: 4,
                        ordered: false,
                        frozen_levels: Some(vec![7.0_f64.to_bits(), 6.0_f64.to_bits()]),
                    },
                    smooth: Box::new(SmoothBasisSpec::FactorSmooth {
                        spec: FactorSmoothSpec {
                            continuous_cols: vec![8],
                            group_col: 5,
                            marginal: marginal(),
                            flavour: FactorSmoothFlavour::Fs {},
                            group_frozen_levels: Some(vec![9.0_f64.to_bits(), 8.0_f64.to_bits()]),
                            frozen_global_orthogonality: None,
                            adaptive: false,
                        },
                    }),
                }),
            }),
        };
        let numeric_wrappers = SmoothBasisSpec::ByVariable {
            by_col: 10,
            kind: BySmoothKind::Numeric,
            by: ByVariableSpec::Numeric,
            inner: Box::new(SmoothBasisSpec::BySmooth {
                by_kind: ByVarKind::Numeric { feature_col: 11 },
                smooth: Box::new(SmoothBasisSpec::BSpline1D {
                    feature_col: 12,
                    spec: marginal(),
                }),
            }),
        };
        let spec = TermCollectionSpec {
            // The two signed-zero spellings are the same gate under the design
            // contract and must collapse to one representative level.
            linear_terms: vec![linear_gate("zero+", 0, 0.0), linear_gate("zero-", 0, -0.0)],
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "main".to_string(),
                feature_col: 1,
                frozen_levels: Some(vec![2.0_f64.to_bits(), 1.0_f64.to_bits()]),
                lenient_unseen: false,
            }],
            smooth_terms: vec![
                smooth("nested", nested),
                smooth("numeric", numeric_wrappers),
            ],
            level: Default::default(),
        };

        let levels = spec.frozen_factor_levels_by_col();
        assert_eq!(levels.get(&0), Some(&canonical_levels(&[0.0])));
        assert_eq!(levels.get(&1), Some(&canonical_levels(&[1.0, 2.0])));
        assert_eq!(levels.get(&2), Some(&canonical_levels(&[3.0])));
        assert_eq!(levels.get(&3), Some(&canonical_levels(&[4.0, 5.0])));
        assert_eq!(levels.get(&4), Some(&canonical_levels(&[6.0, 7.0])));
        assert_eq!(levels.get(&5), Some(&canonical_levels(&[8.0, 9.0])));
        assert_eq!(
            levels.len(),
            6,
            "only categorical carriers belong in the map"
        );
        for numeric_col in [8, 10, 11, 12] {
            assert!(
                !levels.contains_key(&numeric_col),
                "numeric feature column {numeric_col} was misclassified as categorical"
            );
        }
    }
}
