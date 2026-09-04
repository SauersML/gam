// gam#2760: a spatial hyperparameter move is represented as the collection's
// FIXED chart on a moving term-local build, `X_g(ψ) = P_C · X_local(ψ) · Q · T0`.
// The one place that can be checked without any derivative is ψ = ψ̂, the fit's
// own length scale: there the replay must reproduce the collection's realized
// term exactly — the same columns, the same penalty blocks with the same
// nullities, the same REML criterion at any ρ — because the joint [ρ, ψ] search
// evaluates its criterion on the replay and grades the result against the
// scalar-ρ route's, which was fit on the collection.
//
// Before the gauge carried the joint-null rotation `Q`, the replay of the
// `kappa_loop_n_scaling` Duchon spec at the fit's own ℓ produced a block whose
// every column and every penalty differed from the collection's (the unrotated
// local build put through a chart derived on the rotated one) and a criterion
// 679 above the collection's at n = 1000. The scalar route's ρ̂ was then a point
// with `|g| ≈ 10²` on the joint route's function, and the ladder's n ≥ 4000
// rungs died in that search.
#[cfg(test)]
mod spatial_realizer_chart_2760_tests {
    use super::*;
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    use gam_terms::basis::{
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternNu,
        OneDimensionalBoundary, SpatialIdentifiability,
    };
    use ndarray::{Array1, Array2, s};

    fn grid_1d(n: usize) -> (Array2<f64>, Array1<f64>) {
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
            x[[i, 0]] = t;
            y[i] = t.sin();
        }
        (x, y)
    }

    /// The `kappa_loop_n_scaling` spec verbatim: a Linear null space with every
    /// operator penalty active gives the term a non-trivial joint-null rotation.
    fn duchon_spec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "duchon_1d".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        periodic: None,
                        length_scale: Some(1.0),
                        power: 1.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        }
    }

    fn matern_spec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "matern_1d".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                        periodic: None,
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(1.0),
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
        }
    }

    fn frobenius(a: &Array2<f64>) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    /// The realizer's rebuild of `design` at the fit's own ψ, after one rebuild
    /// AWAY from it — the sequence the joint route runs (the ψ-Gram tensor build
    /// realizes the design at many other ψ before the seed is evaluated), so the
    /// term is genuinely re-realized rather than served from the initial design.
    fn replay_at_own_psi<'d>(
        x: ArrayView2<'d, f64>,
        frozen: &TermCollectionSpec,
        design: &TermCollectionDesign,
    ) -> TermCollectionDesign {
        let spatial_terms = spatial_length_scale_term_indices(frozen);
        let dims_per_term = spatial_dims_per_term(frozen, &spatial_terms);
        let rho_dim = design.penalties.len();
        let mut cache = SingleBlockExactJointDesignCache::new(
            x,
            frozen.clone(),
            design.clone(),
            spatial_terms,
            rho_dim,
            dims_per_term,
        )
        .expect("single-block cache");
        let mut theta_away = Array1::<f64>::zeros(rho_dim + 1);
        theta_away[rho_dim] = 0.3;
        cache.ensure_theta(&theta_away).expect("realize away from the fit's psi");
        let theta_own = Array1::<f64>::zeros(rho_dim + 1);
        cache.ensure_theta(&theta_own).expect("realize back at the fit's psi");
        cache.design().clone()
    }

    fn scalar_criterion(
        d: &TermCollectionDesign,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        opts: &ExternalOptimOptions,
        rho: &Array1<f64>,
    ) -> f64 {
        gam_solve::estimate::evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            d.design.clone(),
            offset.view(),
            &d.penalties,
            opts,
            rho,
        )
        .expect("scalar criterion")
        .0
    }

    /// Design, penalties, nullities and criterion of the replay against the
    /// collection's, at the fit's own length scale.
    fn assert_replay_is_the_collection(label: &str, spec: TermCollectionSpec, n: usize) {
        let (x, y) = grid_1d(n);
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let design = build_term_collection_design(x.view(), &spec).expect("design");
        let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");
        let rho_dim = design.penalties.len();
        let term = &design.smooth.terms[0];
        let gauge = term
            .collection_gauge
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: the collection decided no gauge, so this pin exercises nothing"));
        let replay = replay_at_own_psi(x.view(), &frozen, &design);

        let base = design.design.to_dense();
        let rebuilt = replay.design.to_dense();
        assert_eq!(base.dim(), rebuilt.dim(), "{label}: replay width");
        let design_scale = max_abs(&base).max(1.0);
        let design_gap = max_abs(&(&base - &rebuilt));
        assert!(
            design_gap <= 1e-9 * design_scale,
            "{label}: the replay at the fit's own psi is not the collection's design: max|Δ|={design_gap:.3e} against max|X|={design_scale:.3e}"
        );
        assert_eq!(
            design.nullspace_dims, replay.nullspace_dims,
            "{label}: the replay changed a penalty nullity"
        );
        assert_eq!(design.penalties.len(), replay.penalties.len(), "{label}: penalty count");
        for (k, (a, b)) in design.penalties.iter().zip(replay.penalties.iter()).enumerate() {
            assert_eq!(a.col_range, b.col_range, "{label}: penalty {k} range");
            let gap = frobenius(&(&a.local - &b.local));
            assert!(
                gap <= 1e-9 * frobenius(&a.local).max(1.0),
                "{label}: penalty {k} ({:?}) differs between the collection and its replay: ‖ΔS‖_F={gap:.3e}",
                design.penaltyinfo[k].penalty.source
            );
        }

        // The criterion the joint route evaluates on the replay is the criterion
        // the scalar route certified on the collection, at any ρ.
        let opts = external_opts_for_design(
            &LikelihoodSpec::gaussian_identity(),
            &design,
            &FitOptions {
                compute_inference: false,
                max_iter: 200,
                tol: 1e-12,
                ..FitOptions::default()
            },
        );
        for probe in [0.0_f64, -6.0, 3.0] {
            let rho = Array1::from_iter((0..rho_dim).map(|j| probe + 0.2 - 0.1 * j as f64));
            let on_collection = scalar_criterion(&design, &y, &weights, &offset, &opts, &rho);
            let on_replay = scalar_criterion(&replay, &y, &weights, &offset, &opts, &rho);
            let mut theta = Array1::<f64>::zeros(rho_dim + 1);
            theta.slice_mut(s![..rho_dim]).assign(&rho);
            let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
                y.view(),
                weights.view(),
                &design.design,
                offset.view(),
                &design.penalties,
                &opts,
                "gam#2760 replay pin",
            )
            .expect("joint evaluator");
            let joint_on_replay = evaluator
                .evaluate_cost_only(
                    &replay.design,
                    &replay.penalties,
                    &replay.nullspace_dims,
                    replay.linear_constraints.clone(),
                    &theta,
                    rho_dim,
                    None,
                    "gam#2760 replay pin",
                    None,
                )
                .expect("joint criterion on the replay");
            let band = 1e-9 * on_collection.abs().max(1.0);
            assert!(
                (on_replay - on_collection).abs() <= band,
                "{label}: at rho offset {probe} the scalar criterion on the replay ({on_replay:.12e}) is not the collection's ({on_collection:.12e}); gap {:.3e}",
                on_replay - on_collection
            );
            assert!(
                (joint_on_replay - on_collection).abs() <= band,
                "{label}: at rho offset {probe} the joint route's criterion on the replay ({joint_on_replay:.12e}) is not the scalar route's on the collection ({on_collection:.12e}); gap {:.3e}",
                joint_on_replay - on_collection
            );
        }
        eprintln!(
            "[gam#2760 {label}] replay == collection: max|ΔX|={design_gap:.2e}, penalties {}, gauge arm={:?}, rotation carried={}",
            design.penalties.len(),
            gauge.arm,
            gauge.joint_null_rotation.is_some()
        );
    }

    /// The Duchon spec of the #2760 ladder: its joint-null rotation is
    /// non-trivial, and the pin below shows it is load-bearing.
    #[test]
    fn the_realizer_replays_the_collections_chart_at_the_fits_own_length_scale_2760() {
        assert_replay_is_the_collection("duchon linear all-operators", duchon_spec(), 600);
    }

    /// A Matérn control: the same contract on the other radial family.
    #[test]
    fn the_realizer_replays_a_matern_collection_at_its_own_length_scale_2760() {
        assert_replay_is_the_collection("matern nu=5/2 double-penalty", matern_spec(), 400);
    }

    /// Positive control for the pin above: the rotation the gauge now carries is
    /// the piece that was missing. Putting the UNROTATED local build through the
    /// gauge's fixed chart — what the replay did before — is not the collection's
    /// block, and applying the carried `Q` first makes it so.
    #[test]
    fn the_gauges_joint_null_rotation_is_load_bearing_2760() {
        let (x, _y) = grid_1d(600);
        let spec = duchon_spec();
        let design = build_term_collection_design(x.view(), &spec).expect("design");
        let term = &design.smooth.terms[0];
        let gauge = term.collection_gauge.as_ref().expect("collection gauge");
        let rotation = gauge
            .joint_null_rotation
            .as_ref()
            .expect("the ladder's Duchon term has a joint-null rotation, or this control tests nothing");
        // The term's global columns are the ones its penalties cover;
        // `coeff_range` is smooth-local and the intercept precedes it.
        let block_range = design.penalties[0].col_range.clone();
        let base = design
            .design
            .to_dense()
            .slice(s![.., block_range])
            .to_owned();
        let mut workspace = gam_terms::basis::BasisWorkspace::new();
        let local = gam_terms::smooth::build_single_local_smooth_term(
            x.view(),
            &spec.smooth_terms[0],
            &mut workspace,
        )
        .expect("term-local build");
        let unrotated = local.design.to_dense();
        let placed = |block: Array2<f64>| -> Array2<f64> {
            gam_terms::smooth::place_term_in_collection_gauge(
                gauge,
                gam_terms::smooth::LocalTermRealization {
                    design: gam_linalg::matrix::DesignMatrix::Dense(
                        gam_linalg::matrix::DenseDesignMatrix::from(block),
                    ),
                    metadata: &local.metadata,
                    active_penalties: &local.active_penalties,
                    dropped_penalties: local.dropped_penalties.clone(),
                    linear_constraints_local: None,
                    joint_null_rotation: None,
                    termname: "duchon_1d",
                },
            )
            .expect("gauge placement")
            .design
            .to_dense()
        };
        let without_rotation = max_abs(&(&placed(unrotated.clone()) - &base));
        let with_rotation = max_abs(&(&placed(unrotated.dot(&rotation.rotation)) - &base));
        let scale = max_abs(&base).max(1.0);
        assert!(
            with_rotation <= 1e-9 * scale,
            "rotating by the carried Q before the fixed chart does not reproduce the collection: max|Δ|={with_rotation:.3e}"
        );
        assert!(
            without_rotation > 1e-3 * scale,
            "the unrotated block through the fixed chart already reproduces the collection (max|Δ|={without_rotation:.3e}); the rotation is not load-bearing here and this control is vacuous"
        );
    }
}
