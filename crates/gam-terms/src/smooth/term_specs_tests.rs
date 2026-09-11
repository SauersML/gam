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
    fn non_materialized_penalty_falls_back_conservatively() {
        // A penalty whose stored block is not p_local × p_local (e.g. a
        // Kronecker tensor factor). With ≥2 penalties the conservative joint
        // dim is 0 (never over-rejecting).
        let full: Array2<f64> = array![[0.0, 0.0], [0.0, 1.0]];
        let factor: Array2<f64> = array![[1.0]]; // wrong shape for p_local=2
        let mixed_penalties = [
            active_penalty(full, 1, 1, 0, PenaltySource::Primary),
            active_penalty(
                factor.clone(),
                2,
                0,
                1,
                PenaltySource::TensorMarginal { dim: 0 },
            ),
        ];
        assert_eq!(joint_unpenalized_dim(2, &mixed_penalties), 0);
        // With a single non-materialized penalty, fall back to its own null dim.
        let factor_penalties = [active_penalty(
            factor,
            2,
            2,
            0,
            PenaltySource::TensorMarginal { dim: 0 },
        )];
        assert_eq!(joint_unpenalized_dim(4, &factor_penalties), 2);
    }
}

#[cfg(test)]
mod kronecker_penalty_system_tests {
    use super::KroneckerPenaltySystem;
    use ndarray::array;

    #[test]
    fn double_penalty_rank_derivatives_use_only_joint_null_space() {
        let penalties = vec![
            array![[0.0, 0.0], [0.0, 2.0]],
            array![[0.0, 0.0], [0.0, 3.0]],
        ];
        let system = KroneckerPenaltySystem::new(penalties, vec![2usize, 2usize], true).unwrap();
        let lambdas = vec![5.0, 7.0, 11.0];

        let (logdet, rank, grad, hess) = system.logdet_rank_and_derivatives(&lambdas, 0.0);

        let expected_diag = [11.0_f64, 21.0, 10.0, 31.0];
        let expected_logdet: f64 = expected_diag.iter().map(|v| v.ln()).sum();
        assert_eq!(rank, 4);
        assert!((logdet - expected_logdet).abs() <= 1e-12);
        assert!(
            (grad[2] - 1.0).abs() <= 1e-12,
            "double-penalty rank derivative must count only the joint null mode, got {}",
            grad[2]
        );
        assert!(hess[[2, 2]].abs() <= 1e-12);
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
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        spatial_term_psi_bounds(
            data.view(),
            &spec,
            0,
            &SpatialLengthScaleOptimizationOptions::default(),
        )
        .expect("finite spatial ψ bounds")
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
        let options = SpatialLengthScaleOptimizationOptions::default();
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
                    shape: ShapeConstraint::None,
                    joint_null_rotation: None,
                }],
            };
            let geometry = spatial_term_psi_bounds(source.view(), &spec, 0, &options)
                .expect("finite geometry window");
            let search = spatial_term_psi_search_box(source.view(), &spec, 0, &options)
                .expect("finite search box");
            (geometry, search)
        };

        // An incumbent far past the long-range edge of the geometry window —
        // #2454's fixture shape, where `length_scale = 12` sits about six data
        // diameters out.
        let far = 1.0e3_f64;
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
        let built = build_tensor_bspline_basis(data.view(), &[0, 1], &spec)
            .expect("double-penalty tensor basis");
        assert!(
            built
                .active_penalties
                .iter()
                .any(|penalty| { matches!(penalty.info.source, PenaltySource::TensorGlobalRidge) })
        );
        assert!(
            built.kronecker_factored.is_none(),
            "the legacy factored runtime cannot represent a function-metric global ridge"
        );

        spec.double_penalty = false;
        let singly_penalized = build_tensor_bspline_basis(data.view(), &[0, 1], &spec)
            .expect("single-penalty tensor basis");
        assert!(
            singly_penalized.kronecker_factored.is_some(),
            "the exact marginal-only fast path must remain available"
        );
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
            let built = build_tensor_bspline_basis(data.view(), &[0, 1], &spec)
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

        let error = build_tensor_bspline_basis(data.view(), &[0, 1], &spec)
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
            drop_first_level: false,
            penalized: true,
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
        // A numeric-coded `factor(year)`: full one-hot (`drop_first_level=false`),
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
        let smooth_penalty = 2.5;
        let built = build_pca_smooth_basis(
            data.view(),
            &[0, 1],
            &basis,
            false,
            smooth_penalty,
            None,
            None,
            2,
        )
        .expect("full-rank PCA basis");
        let coefficients = array![0.7, -1.2];
        let design = built.design.to_dense();
        let fitted = design.dot(&coefficients);
        let expected = smooth_penalty * fitted.dot(&fitted) / fitted.len() as f64;
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
        let smooth_penalty = 1.7;

        let base = build_pca_smooth_basis(
            scores.view(),
            &[0, 1],
            &identity,
            false,
            smooth_penalty,
            None,
            None,
            2,
        )
        .expect("base PCA chart");
        let transformed = build_pca_smooth_basis(
            scores.view(),
            &[0, 1],
            &transform,
            false,
            smooth_penalty,
            None,
            None,
            2,
        )
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
            1.0,
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
        let smooth_penalty = 2.25;
        let path = write_f64_npy(&scores);
        let dense = build_pca_smooth_basis(
            scores.view(),
            &[0, 1],
            &Array2::<f64>::eye(2),
            false,
            smooth_penalty,
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
            smooth_penalty,
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
mod canonical_nullspace_direction_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn per_axis_null_penalties_are_invariant_to_eigensolver_gauge_2315() {
        let inv_sqrt_two = 0.5_f64.sqrt();
        let z = array![
            [inv_sqrt_two, 0.0],
            [inv_sqrt_two, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ];
        let rotation = array![[0.6, -0.8], [0.8, 0.6]];
        let rotated = z.dot(&rotation);
        let reference = canonical_nullspace_directions(&z).expect("canonical null basis");
        let actual =
            canonical_nullspace_directions(&rotated).expect("rotated canonical null basis");
        for axis in 0..reference.ncols() {
            let reference_penalty = reference
                .column(axis)
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&reference.column(axis).insert_axis(Axis(0)));
            let actual_penalty = actual
                .column(axis)
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&actual.column(axis).insert_axis(Axis(0)));
            let max_error = reference_penalty
                .iter()
                .zip(actual_penalty.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_error <= 256.0 * f64::EPSILON,
                "axis {axis} changed by {max_error:e}"
            );
        }
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
                },
            },
            shape: ShapeConstraint::None,
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
            shape: ShapeConstraint::None,
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
                drop_first_level: false,
                penalized: true,
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
                drop_first_level: false,
                penalized: true,
                frozen_levels: Some(vec![2.0_f64.to_bits(), 1.0_f64.to_bits()]),
                lenient_unseen: false,
            }],
            smooth_terms: vec![
                smooth("nested", nested),
                smooth("numeric", numeric_wrappers),
            ],
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
