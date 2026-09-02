    use super::tests_multidir_channels::{
        latent_survival_row_primary_fourth_contracted_multidir_reference,
        latent_survival_row_primary_gradient_hessian_multidir_reference,
        latent_survival_row_primary_third_contracted_multidir_reference,
    };
    use super::tests_multidir_row::latent_survival_row_primary_log_jet_multidir_reference;
    use super::*;
    use crate::custom_family::BlockWorkingSet;
    use gam_linalg::matrix::DenseDesignMatrix;
    use ndarray::array;

    fn learnable_sigma_test_family() -> LatentSurvivalFamily {
        LatentSurvivalFamily {
            event_target: array![1u8, 0u8],
            weights: array![1.0, 0.7],
            latent_sd_fixed: None,
            hazard_loading: HazardLoading::LoadedVsUnloaded,
            unloaded_mass_entry: array![0.02, 0.03],
            unloaded_mass_exit: array![0.05, 0.08],
            unloaded_hazard_exit: array![0.04, 0.0],
            x_time_entry: array![[1.0, -0.2], [0.4, 0.7]],
            x_time_exit: array![[1.3, 0.1], [0.9, 1.0]],
            x_time_derivative_exit: array![[0.8, 0.4], [0.6, 0.5]],
            x_time_right: array![[1.3, 0.1], [0.9, 1.0]],
            time_offset_right: Array1::zeros(2),
            unloaded_mass_right: Array1::zeros(2),
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0, -0.3], [0.2, 0.9]])),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        }
    }

    fn learnable_sigma_test_joint_beta() -> Array1<f64> {
        array![0.15, 0.25, 0.1, -0.15, 0.35_f64.ln()]
    }

    #[test]
    fn saved_latent_survival_alo_matches_deterministic_frailty_oracle() {
        const K: usize = 5;
        let weight = 1.7;
        let quadrature = QuadratureContext::new();
        let geometry = latent_survival_alo_row_geometry(LatentSurvivalAloRowInput {
            quadrature: &quadrature,
            hazard_loading: HazardLoading::Full,
            event_code: 0,
            prior_weight: weight,
            q_entry: -0.4,
            q_exit: 0.2,
            qdot_exit: 1.1,
            q_right: 0.2,
            mu: -0.1,
            sigma: LatentSurvivalAloSigma::Fixed(0.0),
            unloaded_mass_entry: 0.0,
            unloaded_mass_exit: 0.0,
            unloaded_mass_right: 0.0,
            unloaded_hazard_exit: 0.0,
        })
        .expect("exact saved latent-survival row");

        let values: [f64; K] = geometry
            .coordinate_values
            .as_slice()
            .expect("owned coordinates are contiguous")
            .try_into()
            .expect("fixed-scale latent survival has five primaries");
        let variables: [Order2<K>; K] =
            std::array::from_fn(|axis| Order2::variable(values[axis], axis));
        // At sigma=0 and full loading, right-censoring has
        // NLL = w[exp(q_exit + mu) - exp(q_entry + mu)].
        let oracle = variables[1]
            .add(&variables[4])
            .exp()
            .sub(&variables[0].add(&variables[4]).exp())
            .scale(weight);
        for left in 0..K {
            assert!((geometry.nll_score[left] - oracle.g()[left]).abs() <= 3e-12);
            for right in 0..K {
                assert!(
                    (geometry.observed_hessian[[left, right]] - oracle.h()[left][right]).abs()
                        <= 4e-12
                );
            }
        }
    }

    #[test]
    fn saved_latent_binary_alo_matches_deterministic_frailty_oracle() {
        const K: usize = 3;
        let weight = 0.8;
        let quadrature = QuadratureContext::new();
        let geometry = latent_binary_alo_row_geometry(LatentBinaryAloRowInput {
            quadrature: &quadrature,
            hazard_loading: HazardLoading::Full,
            event: 1,
            prior_weight: weight,
            q_entry: -0.6,
            q_exit: 0.3,
            mu: -0.2,
            sigma: 0.0,
            unloaded_mass_entry: 0.0,
            unloaded_mass_exit: 0.0,
        })
        .expect("exact saved latent-binary row");

        let values: [f64; K] = geometry
            .coordinate_values
            .as_slice()
            .expect("owned coordinates are contiguous")
            .try_into()
            .expect("latent binary has three live primaries");
        let variables: [Order2<K>; K] =
            std::array::from_fn(|axis| Order2::variable(values[axis], axis));
        let log_survival = variables[0]
            .add(&variables[2])
            .exp()
            .sub(&variables[1].add(&variables[2]).exp());
        let one = variables[0].compose_unary([1.0, 0.0, 0.0, 0.0, 0.0]);
        // Event NLL = -w log(1 - exp(log_survival)).
        let oracle = one.sub(&log_survival.exp()).ln().neg().scale(weight);
        for left in 0..K {
            assert!((geometry.nll_score[left] - oracle.g()[left]).abs() <= 4e-12);
            for right in 0..K {
                assert!(
                    (geometry.observed_hessian[[left, right]] - oracle.h()[left][right]).abs()
                        <= 6e-12
                );
            }
        }
    }

    /// Regression (frailty scale block deletion): a learnable-σ latent-survival
    /// fit routes the pre-fit identifiability audit CHANNEL-AWARE, so the
    /// `log_sigma` scale block — realised as a single constant-of-ones column —
    /// is never aliased against the `mean` intercept and dropped. Before the
    /// `output_channel_assignment` override the family used the trait default
    /// (every block → channel 0); the flat single-channel RRQR then saw the two
    /// constant columns (mean intercept, log-σ constant) as a cross-block alias,
    /// attributed the drop to the lowest-priority `log_sigma` block, and deleted
    /// the ONLY handle on the frailty scale — which both froze σ and left the
    /// reduced spec width one short of the family's raw joint Hessian, aborting
    /// every outer LAML eval with `joint exact-newton Hessian validation … got
    /// 15x15, expected 14x14`. The contract this guards: the scale channel must
    /// be distinct from the location (mean) channel.
    #[test]
    fn latent_survival_learnable_sigma_block_lives_on_a_distinct_output_channel() {
        let family = learnable_sigma_test_family();
        assert!(
            family.latent_sd_fixed.is_none(),
            "test fixture must be the learnable-σ family"
        );

        // The three blocks the learnable-σ builder emits, in order. Only the
        // block NAME drives `output_channel_assignment`, so borrow the real
        // `build_log_sigma_blockspec` shape and relabel for time/mean.
        let mut time_spec = build_log_sigma_blockspec(0.5, family.event_target.len());
        time_spec.name = "time_transform".to_string();
        let mut mean_spec = build_log_sigma_blockspec(0.5, family.event_target.len());
        mean_spec.name = "mean".to_string();
        let log_sigma_spec = build_log_sigma_blockspec(0.5, family.event_target.len());
        assert_eq!(log_sigma_spec.name, "log_sigma");
        let specs = vec![time_spec, mean_spec, log_sigma_spec];

        let channels = family
            .output_channel_assignment(&specs)
            .expect("latent survival must declare an explicit channel assignment");
        assert_eq!(channels.len(), specs.len());

        let (time_ch, mean_ch, log_sigma_ch) = (channels[0], channels[1], channels[2]);
        // The load-bearing assertion: the scale channel is NOT the location
        // channel, so the channel-aware audit never aliases σ's constant column
        // against the mean intercept.
        assert_ne!(
            log_sigma_ch, mean_ch,
            "log_sigma (frailty scale) must not share the mean's output channel, or the \
             identifiability audit will alias its constant column against the mean intercept \
             and delete the scale parameter"
        );
        // Each latent-survival block drives a structurally distinct output, so
        // all three channels are distinct (block-diagonal true Jacobian).
        assert_ne!(time_ch, mean_ch);
        assert_ne!(time_ch, log_sigma_ch);
        let n_outputs = channels.iter().copied().max().unwrap() + 1;
        assert!(
            n_outputs >= 3,
            "learnable-σ latent survival must expose ≥3 output channels (time, mean, scale), \
             got {n_outputs}"
        );
    }

    fn survival_stress_test_family(n: usize) -> LatentSurvivalFamily {
        LatentSurvivalFamily {
            event_target: Array1::from_iter((0..n).map(|i| if i % 3 == 0 { 1u8 } else { 0u8 })),
            weights: Array1::from_iter((0..n).map(|i| 0.55 + 0.03 * ((i % 7) as f64))),
            latent_sd_fixed: None,
            hazard_loading: HazardLoading::LoadedVsUnloaded,
            unloaded_mass_entry: Array1::from_iter(
                (0..n).map(|i| 0.015 + 0.0015 * ((i % 11) as f64)),
            ),
            unloaded_mass_exit: Array1::from_iter((0..n).map(|i| 0.06 + 0.002 * ((i % 13) as f64))),
            unloaded_hazard_exit: Array1::from_iter((0..n).map(|i| {
                if i % 4 == 0 {
                    0.018 + 0.001 * ((i % 5) as f64)
                } else {
                    0.0
                }
            })),
            x_time_entry: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.2 + 0.03 * ((i + 2 * j) % 9) as f64 - if j == 1 { 0.12 } else { 0.0 }
            }),
            x_time_exit: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.35 + 0.025 * ((2 * i + j) % 10) as f64 - if j == 2 { 0.08 } else { 0.0 }
            }),
            x_time_derivative_exit: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.45 + 0.015 * ((i + 3 * j) % 8) as f64
            }),
            x_time_right: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.35 + 0.025 * ((2 * i + j) % 10) as f64 - if j == 2 { 0.08 } else { 0.0 }
            }),
            time_offset_right: Array1::zeros(n),
            unloaded_mass_right: Array1::zeros(n),
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_shape_fn(
                (n, 3),
                |(i, j)| 0.1 + 0.04 * ((3 * i + j) % 7) as f64 - if j == 0 { 0.18 } else { 0.0 },
            ))),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        }
    }

    fn survival_stress_test_joint_beta() -> Array1<f64> {
        array![0.18, 0.11, 0.07, 0.13, -0.09, 0.05, 0.12, 0.42_f64.ln()]
    }

    fn latent_survival_states_from_joint_beta(
        family: &LatentSurvivalFamily,
        joint_beta: &Array1<f64>,
    ) -> Vec<ParameterBlockState> {
        let slices = family.joint_slices();
        let n = family.event_target.len();
        let beta_time = joint_beta.slice(s![slices.time.clone()]).to_owned();
        let beta_mean = joint_beta.slice(s![slices.mean.clone()]).to_owned();

        let mut eta_time = Array1::<f64>::zeros(3 * n);
        eta_time
            .slice_mut(s![0..n])
            .assign(&gam_linalg::faer_ndarray::fast_av(
                &family.x_time_entry,
                &beta_time,
            ));
        eta_time
            .slice_mut(s![n..2 * n])
            .assign(&gam_linalg::faer_ndarray::fast_av(
                &family.x_time_exit,
                &beta_time,
            ));
        eta_time
            .slice_mut(s![2 * n..3 * n])
            .assign(&gam_linalg::faer_ndarray::fast_av(
                &family.x_time_derivative_exit,
                &beta_time,
            ));

        let mut states = vec![
            ParameterBlockState {
                beta: beta_time,
                eta: eta_time,
            },
            ParameterBlockState {
                beta: beta_mean.clone(),
                eta: family.x_mean.dot(&beta_mean),
            },
        ];
        if let Some(log_sigma) = slices.log_sigma {
            let beta_log_sigma = array![joint_beta[log_sigma.start]];
            states.push(ParameterBlockState {
                beta: beta_log_sigma.clone(),
                eta: beta_log_sigma,
            });
        }
        states
    }

    fn max_relative_array1(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| (l - r).abs() / l.abs().max(r.abs()).max(1e-12))
            .fold(0.0_f64, f64::max)
    }

    fn max_relative_array2(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| (l - r).abs() / l.abs().max(r.abs()).max(1e-12))
            .fold(0.0_f64, f64::max)
    }

    fn frobenius_relative_array2(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        let mut diff2 = 0.0_f64;
        let mut scale2 = 0.0_f64;
        for (l, r) in left.iter().zip(right.iter()) {
            let d = l - r;
            diff2 += d * d;
            scale2 += l * l + r * r;
        }
        diff2.sqrt() / scale2.sqrt().max(1e-12)
    }

    fn latent_survival_row_loglik_from_primary(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        primary: &Array1<f64>,
    ) -> f64 {
        let q_entry = primary[LATENT_SURVIVAL_PRIMARY_Q_ENTRY];
        let q_exit = primary[LATENT_SURVIVAL_PRIMARY_Q_EXIT];
        let qdot_exit = primary[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT];
        let q_right = primary[LATENT_SURVIVAL_PRIMARY_Q_RIGHT];
        let mu = primary[LATENT_SURVIVAL_PRIMARY_MU];
        let sigma = primary[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA].exp();
        latent_survival_row_primary_gradient_hessian(
            quadctx,
            row,
            LatentSurvivalPrimaryPoint {
                q_entry,
                q_exit,
                qdot_exit,
                q_right,
                mu,
                sigma,
            },
            true,
        )
        .expect("row primary evaluation")
        .0
    }

    #[derive(Clone, Copy, Debug)]
    struct RichardsonDerivative {
        value: f64,
        uncertainty: f64,
    }

    fn floating_point_gamma(operations: usize) -> f64 {
        let accumulated = operations as f64 * f64::EPSILON;
        accumulated / (1.0 - accumulated)
    }

    /// Fourth-order Richardson derivative and its local, measured error budget.
    ///
    /// The first term is the standard central-difference truncation estimate
    /// from step halving. The second propagates the ordinary floating-point
    /// `γ_n` bounds through the two central differences and their Richardson
    /// combination. No production derivative path is called by this authority.
    fn richardson_central_difference(
        mut value_at: impl FnMut(f64) -> f64,
        coordinate: f64,
        step: f64,
    ) -> RichardsonDerivative {
        let coarse_plus = value_at(coordinate + step);
        let coarse_minus = value_at(coordinate - step);
        let fine_step = 0.5 * step;
        let fine_plus = value_at(coordinate + fine_step);
        let fine_minus = value_at(coordinate - fine_step);
        let coarse = (coarse_plus - coarse_minus) / (2.0 * step);
        let fine = (fine_plus - fine_minus) / (2.0 * fine_step);
        let value = (4.0 * fine - coarse) / 3.0;

        let gamma = floating_point_gamma(3);
        let coarse_roundoff =
            gamma * (coarse_plus.abs() + coarse_minus.abs()) / (2.0 * step);
        let fine_roundoff =
            gamma * (fine_plus.abs() + fine_minus.abs()) / (2.0 * fine_step);
        let combine_roundoff = gamma * (4.0 * fine.abs() + coarse.abs()) / 3.0;
        RichardsonDerivative {
            value,
            uncertainty: (fine - coarse).abs() / 3.0
                + (4.0 * fine_roundoff + coarse_roundoff) / 3.0
                + combine_roundoff,
        }
    }

    fn latent_survival_value_fd_authority(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        point: LatentSurvivalPrimaryPoint,
    ) -> f64 {
        latent_survival_row_primary_log_jet_multidir_reference(quadctx, row, point, &[])
            .expect("value-only finite-difference authority")
            .coeff(0)
    }

    fn latent_test_specs(n: usize, block_dims: &[(&str, usize)]) -> Vec<ParameterBlockSpec> {
        block_dims
            .iter()
            .map(|(name, p)| ParameterBlockSpec {
                name: (*name).to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((n, *p)))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            })
            .collect()
    }

    fn fixed_sigma_binary_test_family() -> LatentBinaryFamily {
        LatentBinaryFamily {
            event_target: array![1u8, 0u8],
            weights: array![1.0, 0.7],
            latent_sd: 0.35,
            hazard_loading: HazardLoading::LoadedVsUnloaded,
            unloaded_mass_entry: array![0.02, 0.03],
            unloaded_mass_exit: array![0.05, 0.08],
            x_time_entry: array![[1.0, -0.2], [0.4, 0.7]],
            x_time_exit: array![[1.3, 0.1], [0.9, 1.0]],
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0, -0.3], [0.2, 0.9]])),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        }
    }

    #[test]
    fn latent_survival_offset_residuals_reject_missing_block_state() {
        let error = learnable_sigma_test_family()
            .offset_channel_residuals(&[])
            .expect_err("missing fitted blocks must not become zero residuals");
        match error {
            LatentSurvivalError::BlockMismatch { reason } => {
                assert!(reason.contains("got 0"), "unexpected mismatch: {reason}");
            }
            other => panic!("missing fitted blocks must be a block mismatch, got {other:?}"),
        }
    }

    #[test]
    fn latent_binary_offset_residuals_reject_missing_block_state() {
        let error = fixed_sigma_binary_test_family()
            .offset_channel_residuals(&[])
            .expect_err("missing fitted blocks must not become zero residuals");
        match error {
            LatentSurvivalError::BlockMismatch { reason } => {
                assert!(reason.contains("got 0"), "unexpected mismatch: {reason}");
            }
            other => panic!("missing fitted blocks must be a block mismatch, got {other:?}"),
        }
    }

    fn latent_binary_states_from_joint_beta(
        family: &LatentBinaryFamily,
        joint_beta: &Array1<f64>,
    ) -> Vec<ParameterBlockState> {
        let slices = family.joint_slices();
        let n = family.event_target.len();
        let beta_time = joint_beta.slice(s![slices.time.clone()]).to_owned();
        let beta_mean = joint_beta.slice(s![slices.mean.clone()]).to_owned();

        let mut eta_time = Array1::<f64>::zeros(3 * n);
        eta_time
            .slice_mut(s![0..n])
            .assign(&gam_linalg::faer_ndarray::fast_av(
                &family.x_time_entry,
                &beta_time,
            ));
        eta_time
            .slice_mut(s![n..2 * n])
            .assign(&gam_linalg::faer_ndarray::fast_av(
                &family.x_time_exit,
                &beta_time,
            ));

        vec![
            ParameterBlockState {
                beta: beta_time,
                eta: eta_time,
            },
            ParameterBlockState {
                beta: beta_mean.clone(),
                eta: family.x_mean.dot(&beta_mean),
            },
        ]
    }

    fn assert_scalar_is_scaled(got: f64, unweighted: f64, scale: f64, quantity: &str) {
        let expected = unweighted * scale;
        let tolerance = 128.0 * f64::EPSILON * expected.abs().max(f64::MIN_POSITIVE);
        assert!(
            (got - expected).abs() <= tolerance,
            "{quantity} did not scale with its positive row weight: got={got:?}, expected={expected:?}, unweighted={unweighted:?}, scale={scale:?}, tolerance={tolerance:?}"
        );
    }

    fn assert_vector_is_scaled(
        got: &Array1<f64>,
        unweighted: &Array1<f64>,
        scale: f64,
        quantity: &str,
    ) {
        assert_eq!(got.len(), unweighted.len());
        for (index, (&actual, &base)) in got.iter().zip(unweighted.iter()).enumerate() {
            assert_scalar_is_scaled(actual, base, scale, &format!("{quantity}[{index}]"));
        }
    }

    fn assert_matrix_is_scaled(
        got: &Array2<f64>,
        unweighted: &Array2<f64>,
        scale: f64,
        quantity: &str,
    ) {
        assert_eq!(got.dim(), unweighted.dim());
        for ((row, col), &actual) in got.indexed_iter() {
            assert_scalar_is_scaled(
                actual,
                unweighted[[row, col]],
                scale,
                &format!("{quantity}[{row},{col}]"),
            );
        }
    }

    #[test]
    fn binary_log_survival_math_is_cancellation_free_and_derivative_order_aware() {
        let near_one_survival: f64 = -1.0e-16;
        let expected = (-near_one_survival.exp_m1()).ln();
        let got = binary_log_likelihood_from_log_survival(near_one_survival, 1)
            .expect("near-boundary binary event likelihood");
        assert_eq!(got.to_bits(), expected.to_bits());

        // At s=-1e-100 the value, score, Hessian, and third derivative are
        // representable, while the fourth derivative is mathematically beyond
        // f64 range. Value/order-2 callers must succeed; only the order-4 path
        // is allowed to refuse the unrepresentable result.
        let extreme = -1.0e-100;
        assert!(
            binary_log_likelihood_from_log_survival(extreme, 1)
                .expect("value remains representable")
                .is_finite()
        );
        let (_, first) = binary_from_log_survival_through_first(extreme, 1)
            .expect("first derivative remains representable");
        assert!(first.is_finite());
        let second =
            binary_from_log_survival(extreme, 1).expect("second derivative remains representable");
        assert!(second.outer_scale.is_finite());
        let (_, third) = binary_from_log_survival_through_third(extreme, 1)
            .expect("third derivative remains representable");
        assert!(third.is_finite());
        let fourth = binary_from_log_survival_through_fourth(extreme, 1)
            .expect_err("unrepresentable fourth derivative must be explicit");
        assert!(
            fourth.to_string().contains("derivative order 4"),
            "unexpected fourth-derivative error: {fourth}"
        );
    }

    #[test]
    fn latent_likelihood_preserves_every_positive_weight_and_scales_all_derivatives() {
        // This power-of-two scale is far below the deleted 1e-12 omission
        // threshold. Multiplication by it changes only the exponent, making
        // value/gradient/Hessian scaling a sharp regression oracle rather than
        // a loose floating-point approximation.
        let tiny_normal = 2.0_f64.powi(-48);

        let mut survival_unit = learnable_sigma_test_family();
        survival_unit.weights = array![1.0, 0.0];
        let survival_states = latent_survival_states_from_joint_beta(
            &survival_unit,
            &learnable_sigma_test_joint_beta(),
        );
        let (survival_ll, survival_gradient, survival_hessian) = survival_unit
            .evaluate_exact_newton_joint_dense(&survival_states)
            .expect("unit-weight latent-survival evaluation");
        let mut survival_tiny = survival_unit.clone();
        survival_tiny.weights[0] = tiny_normal;
        let (tiny_survival_ll, tiny_survival_gradient, tiny_survival_hessian) = survival_tiny
            .evaluate_exact_newton_joint_dense(&survival_states)
            .expect("tiny-positive latent-survival evaluation");
        assert_scalar_is_scaled(
            tiny_survival_ll,
            survival_ll,
            tiny_normal,
            "survival log likelihood",
        );
        assert_vector_is_scaled(
            &tiny_survival_gradient,
            &survival_gradient,
            tiny_normal,
            "survival gradient",
        );
        assert_matrix_is_scaled(
            &tiny_survival_hessian,
            &survival_hessian,
            tiny_normal,
            "survival Hessian",
        );

        let mut binary_unit = fixed_sigma_binary_test_family();
        binary_unit.weights = array![1.0, 0.0];
        let binary_beta = array![0.15, 0.25, 0.1, -0.15];
        let binary_states = latent_binary_states_from_joint_beta(&binary_unit, &binary_beta);
        let (binary_ll, binary_gradient, binary_hessian) = binary_unit
            .evaluate_exact_newton_joint_dense(&binary_states)
            .expect("unit-weight latent-binary evaluation");
        let mut binary_tiny = binary_unit.clone();
        binary_tiny.weights[0] = tiny_normal;
        let (tiny_binary_ll, tiny_binary_gradient, tiny_binary_hessian) = binary_tiny
            .evaluate_exact_newton_joint_dense(&binary_states)
            .expect("tiny-positive latent-binary evaluation");
        assert_scalar_is_scaled(
            tiny_binary_ll,
            binary_ll,
            tiny_normal,
            "binary log likelihood",
        );
        assert_vector_is_scaled(
            &tiny_binary_gradient,
            &binary_gradient,
            tiny_normal,
            "binary gradient",
        );
        assert_matrix_is_scaled(
            &tiny_binary_hessian,
            &binary_hessian,
            tiny_normal,
            "binary Hessian",
        );

        // A largest-subnormal sample weight remains a real likelihood row.
        // Its single-row log-likelihood product is representable and must not
        // collapse to the all-zero dormant-row result.
        let largest_subnormal = f64::from_bits((1_u64 << 52) - 1);
        survival_tiny.weights[0] = largest_subnormal;
        let subnormal_survival_ll = survival_tiny
            .log_likelihood_only(&survival_states)
            .expect("subnormal latent-survival likelihood");
        assert_ne!(subnormal_survival_ll, 0.0);
        assert_scalar_is_scaled(
            subnormal_survival_ll,
            survival_ll,
            largest_subnormal,
            "subnormal survival log likelihood",
        );
        binary_tiny.weights[0] = largest_subnormal;
        let subnormal_binary_ll = binary_tiny
            .log_likelihood_only(&binary_states)
            .expect("subnormal latent-binary likelihood");
        assert_ne!(subnormal_binary_ll, 0.0);
        assert_scalar_is_scaled(
            subnormal_binary_ll,
            binary_ll,
            largest_subnormal,
            "subnormal binary log likelihood",
        );

        // Even the smallest positive subnormal survives when the mathematical
        // product is representable. If it is not representable, the checked
        // scaling primitive refuses it explicitly instead of converting the
        // row into a semantic zero.
        let smallest_subnormal = f64::from_bits(1);
        assert_eq!(
            checked_weighted_row_value(smallest_subnormal, 1.0, 0, "test")
                .expect("representable smallest-subnormal product")
                .to_bits(),
            smallest_subnormal.to_bits()
        );
        let underflow = checked_weighted_row_value(smallest_subnormal, 0.25, 0, "test")
            .expect_err("non-zero underflow must be explicit");
        assert!(
            underflow.contains("underflowed"),
            "unexpected error: {underflow}"
        );
    }

    #[test]
    fn zero_weight_likelihood_rows_are_dormant_before_response_and_predictor_access() {
        let mut survival = learnable_sigma_test_family();
        survival.weights = array![1.0, 0.0];
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&survival, &beta);
        let expected = survival
            .evaluate_exact_newton_joint_dense(&states)
            .expect("clean zero-weight survival reference");

        survival.event_target[1] = 17;
        survival.unloaded_mass_entry[1] = f64::NAN;
        survival.unloaded_mass_exit[1] = f64::NEG_INFINITY;
        survival.unloaded_mass_right[1] = -1.0;
        survival.unloaded_hazard_exit[1] = f64::NAN;
        survival.time_offset_right[1] = f64::NAN;
        survival.x_time_right.row_mut(1).fill(f64::NAN);
        let mut dormant_states = states.clone();
        let n = survival.event_target.len();
        dormant_states[LatentSurvivalFamily::BLOCK_TIME].eta[1] = f64::NAN;
        dormant_states[LatentSurvivalFamily::BLOCK_TIME].eta[n + 1] = f64::INFINITY;
        dormant_states[LatentSurvivalFamily::BLOCK_TIME].eta[2 * n + 1] = f64::NEG_INFINITY;
        dormant_states[LatentSurvivalFamily::BLOCK_MEAN].eta[1] = f64::NAN;
        let got = survival
            .evaluate_exact_newton_joint_dense(&dormant_states)
            .expect("zero-weight survival row must not inspect dormant response/predictors");
        assert_eq!(got, expected);

        let mut binary = fixed_sigma_binary_test_family();
        binary.weights = array![1.0, 0.0];
        let binary_beta = array![0.15, 0.25, 0.1, -0.15];
        let binary_states = latent_binary_states_from_joint_beta(&binary, &binary_beta);
        let expected = binary
            .evaluate_exact_newton_joint_dense(&binary_states)
            .expect("clean zero-weight binary reference");
        binary.event_target[1] = 17;
        binary.unloaded_mass_entry[1] = f64::NAN;
        binary.unloaded_mass_exit[1] = f64::NEG_INFINITY;
        let mut dormant_binary_states = binary_states.clone();
        let n = binary.event_target.len();
        dormant_binary_states[LatentBinaryFamily::BLOCK_TIME].eta[1] = f64::NAN;
        dormant_binary_states[LatentBinaryFamily::BLOCK_TIME].eta[n + 1] = f64::INFINITY;
        dormant_binary_states[LatentBinaryFamily::BLOCK_MEAN].eta[1] = f64::NAN;
        let got = binary
            .evaluate_exact_newton_joint_dense(&dormant_binary_states)
            .expect("zero-weight binary row must not inspect dormant response/predictors");
        assert_eq!(got, expected);
    }

    #[test]
    fn invalid_likelihood_weight_preflight_is_atomic_and_precedes_row_evaluation() {
        let mut family = fixed_sigma_binary_test_family();
        let beta = array![0.15, 0.25, 0.1, -0.15];
        let mut states = latent_binary_states_from_joint_beta(&family, &beta);
        // If row evaluation ran first, this active row would fail on its NaN
        // predictor before the invalid second weight was discovered.
        states[LatentBinaryFamily::BLOCK_MEAN].eta[0] = f64::NAN;
        for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            family.weights = array![1.0, invalid];
            let error = family
                .evaluate_exact_newton_joint_dense(&states)
                .expect_err("invalid likelihood weight must refuse the whole call");
            assert!(
                error.contains("latent-binary row 2 has invalid likelihood weight"),
                "weight preflight did not win atomically for {invalid:?}: {error}"
            );
            assert!(
                !error.contains("predictor"),
                "row evaluation ran before weight preflight for {invalid:?}: {error}"
            );
        }
    }

    // --- shared latent-interval validation engine: parity / contract tests ---

    use crate::survival::location_scale::{TimeBlockInput, TimeBlockMonotonicity};

    /// Minimal, structurally valid `TimeBlockInput` for `n` rows and `p_time`
    /// columns, used to exercise the shared validation driver without standing
    /// up a full term-collection design.
    fn validation_time_block(n: usize, p_time: usize) -> TimeBlockInput {
        let design = |fill: f64| {
            DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem(
                (n, p_time),
                fill,
            )))
        };
        TimeBlockInput {
            design_entry: design(0.1),
            design_exit: design(0.2),
            design_derivative_exit: design(0.3),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::zeros(n),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }
    }

    fn empty_meanspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
        }
    }

    /// Quadratic form `nᵀ M n`.
    fn quad_form(m: &Array2<f64>, n_vec: &Array1<f64>) -> f64 {
        n_vec.dot(&m.dot(n_vec))
    }

    /// The latent-survival time-block ridge must COVER the monotone baseline's
    /// unpenalized affine null direction — the exact direction whose flat MAP
    /// makes the interval-censored warm-start refuse
    /// (`check_map_uniqueness`: `nᵀ S n < tol`, dominant block `time_transform`).
    ///
    /// The I-spline value-space penalty leaves a 1-D null space spanned by the
    /// constant coefficient direction (`constant γ ↦ affine log Λ`). We stand it
    /// in here with the first-difference penalty `D₁ᵀD₁`, whose null space is
    /// exactly that constant/level direction, on a full-rank endpoint design so
    /// the function Gram is SPD. Before installation the direction carries no
    /// penalty curvature; after installation the appended shrinkage penalty
    /// gives it the block's function-norm curvature, so the assembled penalty is
    /// no longer flat there.
    #[test]
    fn latent_time_nullspace_shrinkage_covers_affine_null_direction() {
        let n = 12usize;
        let p = 4usize;

        // Full column-rank endpoint designs (a Vandermonde over a spread grid),
        // so the endpoint-averaged function Gram is SPD.
        let design_at = |shift: f64| {
            Array2::<f64>::from_shape_fn((n, p), |(i, j)| {
                let t = shift + (i as f64) / (n as f64 - 1.0);
                t.powi(j as i32)
            })
        };
        let x_entry = design_at(0.0);
        let x_exit = design_at(0.3);

        // First-difference penalty D₁ᵀD₁ (order-1 P-spline): PSD with a 1-D null
        // space = the constant (level) direction, mirroring the I-spline
        // value-space penalty's affine null.
        let mut penalty = Array2::<f64>::zeros((p, p));
        for k in 0..p - 1 {
            penalty[[k, k]] += 1.0;
            penalty[[k + 1, k + 1]] += 1.0;
            penalty[[k, k + 1]] -= 1.0;
            penalty[[k + 1, k]] -= 1.0;
        }

        let mut time_block = TimeBlockInput {
            design_entry: DesignMatrix::Dense(DenseDesignMatrix::from(x_entry.clone())),
            design_exit: DesignMatrix::Dense(DenseDesignMatrix::from(x_exit.clone())),
            design_derivative_exit: DesignMatrix::Dense(DenseDesignMatrix::from(x_exit.clone())),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::zeros(n),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![penalty.clone()],
            nullspace_dims: vec![1],
            initial_log_lambdas: Some(Array1::from_elem(1, 0.5)),
            initial_beta: None,
        };

        // The constant (level) direction is the penalty's null direction.
        let null_dir = Array1::from_elem(p, 1.0 / (p as f64).sqrt());
        let primary_before = quad_form(&penalty, &null_dir);
        let penalty_scale = penalty.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(
            primary_before < 1e-9 * penalty_scale,
            "affine direction must start unpenalized: nᵀ S n = {primary_before:.3e}"
        );

        let installed = install_latent_time_nullspace_shrinkage_penalty(&mut time_block)
            .expect("shrinkage installation must succeed on a full-rank endpoint design");
        assert!(installed, "a penalty with a null space must gain a ridge");

        // Bookkeeping stays self-consistent: one extra penalty, its structural
        // nullspace dim recorded as 0 (a full-rank-on-support ridge), and the
        // seed ρ vector widened to match.
        assert_eq!(time_block.penalties.len(), 2);
        assert_eq!(time_block.nullspace_dims, vec![1, 0]);
        assert_eq!(
            time_block.initial_log_lambdas.as_ref().map(|s| s.len()),
            Some(2)
        );

        // The appended ridge gives the previously-null direction the block's
        // function-norm curvature: `nᵀ R n ≈ nᵀ G n`, the endpoint-averaged
        // function Gram of the null direction (the Marra & Wood identity for
        // `n` spanning the 1-D null space).
        let shrinkage = &time_block.penalties[1];
        let function_gram =
            (x_entry.t().dot(&x_entry) + x_exit.t().dot(&x_exit)).mapv(|v| v / (2 * n) as f64);
        let ridge_curvature = quad_form(shrinkage, &null_dir);
        let function_norm = quad_form(&function_gram, &null_dir);
        assert!(
            function_norm > 0.0,
            "the null direction has positive function norm on a full-rank design"
        );
        assert!(
            ridge_curvature > 0.5 * function_norm,
            "ridge must cover the affine null direction: nᵀ R n = {ridge_curvature:.3e} vs \
             nᵀ G n = {function_norm:.3e}"
        );
        assert!(
            (ridge_curvature - function_norm).abs() < 1e-6 * function_norm.max(1.0),
            "ridge curvature on the 1-D null must equal the function norm: \
             nᵀ R n = {ridge_curvature:.3e}, nᵀ G n = {function_norm:.3e}"
        );

        // The assembled time-block penalty is no longer flat along the direction
        // the MAP-uniqueness audit flagged.
        let total: Array2<f64> = &penalty + shrinkage;
        assert!(
            quad_form(&total, &null_dir) > 0.5 * function_norm,
            "assembled penalty must cover the previously-null direction"
        );
    }

    /// A valid two-row latent-survival term spec (one exact event under loaded
    /// hazard, one right-censored row).
    fn valid_survival_spec(n: usize, p_time: usize) -> LatentSurvivalTermSpec {
        LatentSurvivalTermSpec {
            age_entry: Array1::zeros(n),
            age_exit: Array1::from_elem(n, 1.0),
            event_target: Array1::from_shape_fn(n, |i| (i % 2) as u8),
            weights: Array1::from_elem(n, 1.0),
            derivative_guard: 0.0,
            time_block: validation_time_block(n, p_time),
            time_design_right: None,
            time_offset_right: None,
            unloaded_mass_entry: Array1::from_elem(n, 0.01),
            unloaded_mass_exit: Array1::from_elem(n, 0.05),
            unloaded_mass_right: Array1::zeros(0),
            unloaded_hazard_exit: Array1::from_elem(n, 0.02),
            meanspec: empty_meanspec(),
            mean_offset: Array1::zeros(n),
        }
    }

    /// A valid latent-binary term spec mirroring `valid_survival_spec` but
    /// without the per-row unloaded hazard.
    fn valid_binary_spec(n: usize, p_time: usize) -> LatentBinaryTermSpec {
        LatentBinaryTermSpec {
            age_entry: Array1::zeros(n),
            age_exit: Array1::from_elem(n, 1.0),
            event_target: Array1::from_shape_fn(n, |i| (i % 2) as u8),
            weights: Array1::from_elem(n, 1.0),
            derivative_guard: 0.0,
            time_block: validation_time_block(n, p_time),
            unloaded_mass_entry: Array1::from_elem(n, 0.01),
            unloaded_mass_exit: Array1::from_elem(n, 0.05),
            meanspec: empty_meanspec(),
            mean_offset: Array1::zeros(n),
        }
    }

    fn loaded_frailty() -> FrailtySpec {
        FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed { sigma: 0.3 },
            loading: HazardLoading::LoadedVsUnloaded,
        }
    }

    /// Both adapters route through the shared `validate_latent_interval_inputs`
    /// engine, but each must still emit its own context prefix and (for the
    /// size-mismatch / unloaded-decomposition diagnostics) the hazard-aware vs
    /// mass-only message variant. This pins the byte-for-byte contract the
    /// unification had to preserve, the property the issue's "old vs new
    /// validation errors" parity test guards.
    #[test]
    fn latent_interval_validation_parity_across_models() {
        let n = 2;
        let p_time = 2;
        let data = Array2::<f64>::zeros((n, 3));

        // 1. A clean spec validates and round-trips the typed scale.
        //    Binary then extracts the required fixed scalar.
        let surv_sigma = validate_latent_survival_inputs(
            data.view(),
            &valid_survival_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect("valid survival spec must validate");
        assert_eq!(surv_sigma, FrailtyScale::Fixed { sigma: 0.3 });
        let bin_sigma = validate_latent_binary_inputs(
            data.view(),
            &valid_binary_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect("valid binary spec must validate");
        assert_eq!(bin_sigma, 0.3);

        // 2. Empty data: shared driver, per-model context prefix.
        let empty = Array2::<f64>::zeros((0, 3));
        let surv_empty = validate_latent_survival_inputs(
            empty.view(),
            &valid_survival_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect_err("empty data must be rejected");
        assert_eq!(
            surv_empty.to_string(),
            "latent-survival requires a non-empty dataset"
        );
        let bin_empty = validate_latent_binary_inputs(
            empty.view(),
            &valid_binary_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect_err("empty data must be rejected");
        assert_eq!(
            bin_empty.to_string(),
            "latent-binary requires a non-empty dataset"
        );

        // 3. Size mismatch: survival's message carries `unloaded_hazard=`,
        //    binary's does not. This is the one shape that distinguishes the
        //    two row views feeding the shared driver.
        let mut surv_bad = valid_survival_spec(n, p_time);
        surv_bad.weights = Array1::from_elem(n + 1, 1.0);
        let surv_size = validate_latent_survival_inputs(data.view(), &surv_bad, &loaded_frailty())
            .expect_err("size mismatch must be rejected");
        let surv_msg = surv_size.to_string();
        assert!(
            surv_msg.starts_with("latent-survival size mismatch")
                && surv_msg.contains("unloaded_hazard="),
            "survival size-mismatch message must include unloaded_hazard: {surv_msg}"
        );
        let mut bin_bad = valid_binary_spec(n, p_time);
        bin_bad.weights = Array1::from_elem(n + 1, 1.0);
        let bin_size = validate_latent_binary_inputs(data.view(), &bin_bad, &loaded_frailty())
            .expect_err("size mismatch must be rejected");
        let bin_msg = bin_size.to_string();
        assert!(
            bin_msg.starts_with("latent-binary size mismatch")
                && !bin_msg.contains("unloaded_hazard"),
            "binary size-mismatch message must omit unloaded_hazard: {bin_msg}"
        );

        // 4. Invalid unloaded decomposition: survival reports `exit_hazard=`,
        //    binary reports only the two masses.
        let mut surv_neg_hazard = valid_survival_spec(n, p_time);
        surv_neg_hazard.unloaded_hazard_exit[0] = -1.0;
        let surv_decomp =
            validate_latent_survival_inputs(data.view(), &surv_neg_hazard, &loaded_frailty())
                .expect_err("negative unloaded hazard must be rejected");
        assert_eq!(
            surv_decomp.to_string(),
            "latent-survival row 1 has invalid unloaded hazard decomposition: entry_mass=0.01, exit_mass=0.05, exit_hazard=-1"
        );
        let mut bin_bad_mass = valid_binary_spec(n, p_time);
        bin_bad_mass.unloaded_mass_exit[0] = 0.0; // exit < entry
        let bin_decomp =
            validate_latent_binary_inputs(data.view(), &bin_bad_mass, &loaded_frailty())
                .expect_err("non-monotone unloaded mass must be rejected");
        assert_eq!(
            bin_decomp.to_string(),
            "latent-binary row 1 has invalid unloaded mass decomposition: entry_mass=0.01, exit_mass=0"
        );

        // 5. Per-row interval/event/weight diagnostics share one engine, so an
        //    identical invalid input yields identical (modulo prefix) text.
        let mut surv_event = valid_survival_spec(n, p_time);
        surv_event.event_target[1] = 7;
        let surv_event_err =
            validate_latent_survival_inputs(data.view(), &surv_event, &loaded_frailty())
                .expect_err("invalid event target must be rejected");
        assert_eq!(
            surv_event_err.to_string(),
            "latent-survival row 2 has invalid event target 7; expected 0 or 1"
        );
        let mut bin_event = valid_binary_spec(n, p_time);
        bin_event.event_target[1] = 7;
        let bin_event_err =
            validate_latent_binary_inputs(data.view(), &bin_event, &loaded_frailty())
                .expect_err("invalid event target must be rejected");
        assert_eq!(
            bin_event_err.to_string(),
            "latent-binary row 2 has invalid event target 7; expected 0 or 1"
        );

        // 6. Frailty policy divergence: survival accepts a learned scale,
        //    binary rejects it.
        let learnable = FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
            loading: HazardLoading::LoadedVsUnloaded,
        };
        let surv_learnable = validate_latent_survival_inputs(
            data.view(),
            &valid_survival_spec(n, p_time),
            &learnable,
        )
        .expect("survival accepts a learnable latent scale");
        assert_eq!(
            surv_learnable,
            FrailtyScale::Learned { initial_sigma: 0.5 }
        );
        let bin_learnable =
            validate_latent_binary_inputs(data.view(), &valid_binary_spec(n, p_time), &learnable)
                .expect_err("binary requires a fixed latent scale");
        assert_eq!(
            bin_learnable.to_string(),
            // `LatentBinaryModel::frailty_policy` routes through
            // `fixed_latent_hazard_frailty_typed(frailty, "latent-binary")`,
            // whose reason is `"{context} requires a fixed hazard-multiplier
            // sigma"`. The stale "currently" predates that shared driver.
            "latent-binary requires a fixed hazard-multiplier sigma"
        );

        // 7. The time-block shape check is owned by the shared driver: a
        //    column-count mismatch is reported with the per-model prefix.
        let mut surv_time_bad = valid_survival_spec(n, p_time);
        surv_time_bad.time_block.design_entry = DesignMatrix::Dense(DenseDesignMatrix::from(
            Array2::from_elem((n, p_time + 1), 0.1),
        ));
        let surv_time_err =
            validate_latent_survival_inputs(data.view(), &surv_time_bad, &loaded_frailty())
                .expect_err("time block column mismatch must be rejected");
        assert!(
            surv_time_err
                .to_string()
                .starts_with("latent-survival time block column mismatch"),
            "unexpected survival time-block message: {surv_time_err}"
        );
    }

    #[test]
    fn latent_interval_validation_treats_exact_zero_rows_as_response_dormant() {
        let n = 2;
        let data = Array2::<f64>::zeros((n, 1));

        let mut survival = valid_survival_spec(n, 1);
        survival.weights[0] = 0.0;
        survival.age_entry[0] = f64::NAN;
        survival.age_exit[0] = f64::NEG_INFINITY;
        survival.event_target[0] = 19;
        survival.unloaded_mass_entry[0] = f64::NAN;
        survival.unloaded_mass_exit[0] = -1.0;
        survival.unloaded_hazard_exit[0] = f64::INFINITY;
        validate_latent_survival_inputs(data.view(), &survival, &loaded_frailty())
            .expect("zero-weight survival response row must be dormant");

        let mut binary = valid_binary_spec(n, 1);
        binary.weights[0] = 0.0;
        binary.age_entry[0] = f64::NAN;
        binary.age_exit[0] = f64::NEG_INFINITY;
        binary.event_target[0] = 19;
        binary.unloaded_mass_entry[0] = f64::NAN;
        binary.unloaded_mass_exit[0] = -1.0;
        validate_latent_binary_inputs(data.view(), &binary, &loaded_frailty())
            .expect("zero-weight binary response row must be dormant");

        // The weight scan is whole-vector and precedes response geometry. A
        // later invalid weight therefore wins deterministically over the bad
        // active response in row 1.
        survival.weights = array![1.0, f64::NAN];
        let error = validate_latent_survival_inputs(data.view(), &survival, &loaded_frailty())
            .expect_err("non-finite weight must atomically refuse validation");
        assert!(
            error
                .to_string()
                .contains("latent-survival row 2 has invalid weight"),
            "unexpected atomic preflight error: {error}"
        );
    }

    #[test]
    fn latent_survival_coefficient_cost_uses_joint_coupled_formula() {
        // `evaluate_exact_newton_joint_dense` builds a fully dense joint
        // Hessian over (Σ p_b)² across the time, mean, and log-σ blocks via
        // per-row pullback of the latent-survival primary kernel. The override
        // must reflect that joint coupling rather than the block-diagonal
        // default.
        let family = learnable_sigma_test_family();
        let n = family.event_target.len() as u64;
        let p_time = 2u64;
        let p_mean = 2u64;
        let p_log_sigma = 1u64;
        let specs = vec![
            ParameterBlockSpec {
                name: "time".to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((
                    n as usize,
                    p_time as usize,
                )))),
                offset: Array1::zeros(n as usize),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
            ParameterBlockSpec {
                name: "mean".to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((
                    n as usize,
                    p_mean as usize,
                )))),
                offset: Array1::zeros(n as usize),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((
                    n as usize,
                    p_log_sigma as usize,
                )))),
                offset: Array1::zeros(n as usize),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
        ];
        let p_total = p_time + p_mean + p_log_sigma;
        let expected_joint = n * p_total * p_total;
        let expected_block_diag =
            n * (p_time * p_time + p_mean * p_mean + p_log_sigma * p_log_sigma);
        assert_eq!(family.coefficient_hessian_cost(&specs), expected_joint);
        // Cross-block fill (time–mean, time–log_sigma, mean–log_sigma) makes
        // the joint cost strictly larger than the block-diagonal default.
        assert!(expected_joint > expected_block_diag);
    }

    #[test]
    fn latent_family_planner_keeps_outer_hessian_at_large_n() {
        use crate::custom_family::custom_family_outer_derivatives;
        use gam_problem::{DeclaredHessianForm, Derivative};

        let options = BlockwiseFitOptions::default();
        let large_n = 50_001;

        let survival = learnable_sigma_test_family();
        let survival_specs =
            latent_test_specs(large_n, &[("time", 2), ("mean", 2), ("log_sigma", 1)]);
        let (surv_grad, surv_hess) =
            custom_family_outer_derivatives(&survival, &survival_specs, &options);
        assert_eq!(surv_grad, Derivative::Analytic);
        assert_eq!(surv_hess, DeclaredHessianForm::Either);

        let binary = fixed_sigma_binary_test_family();
        let binary_specs = latent_test_specs(large_n, &[("time", 2), ("mean", 2)]);
        let (bin_grad, bin_hess) =
            custom_family_outer_derivatives(&binary, &binary_specs, &options);
        assert_eq!(bin_grad, Derivative::Analytic);
        assert_eq!(bin_hess, DeclaredHessianForm::Either);
    }

    #[test]
    fn latent_families_arm_self_vanishing_levenberg_on_ill_conditioning() {
        // Regression guard for #1108. The interval-censored row contribution
        // `ℓ = log[S(L) − S(R)]` is the log of a DIFFERENCE of survival kernels and
        // is legitimately NON-concave (indefinite per-row Hessian) away from the
        // optimum; on the constrained (monotone-cone) coupled time block this can
        // make the penalized joint Hessian full-rank yet indefinite / severely
        // ill-conditioned at the cold-start seed. The coupled exact-joint inner
        // solver only adds the self-vanishing Levenberg–Marquardt diagonal floor
        // (the cure for a full-rank ill-conditioned reflected QP that otherwise
        // oscillates the trust region into a snapshot-less stall) when the family
        // opts in via `levenberg_on_ill_conditioning()`. Both latent families MUST
        // keep this armed (the default is `false`, which leaves the interval inner
        // solve diverging with "exited the joint Newton path before convergence").
        assert!(
            learnable_sigma_test_family().levenberg_on_ill_conditioning(),
            "LatentSurvivalFamily must arm the self-vanishing Levenberg floor so the \
             indefinite interval-censored joint Hessian converges (see #1108)"
        );
        assert!(
            fixed_sigma_binary_test_family().levenberg_on_ill_conditioning(),
            "LatentBinaryFamily must arm the self-vanishing Levenberg floor on its \
             constrained coupled time block (see #1108)"
        );
    }

    #[test]
    fn latent_binary_exact_joint_hessian_and_workspace_matvec_match_fd() {
        let family = fixed_sigma_binary_test_family();
        let beta = array![0.15, 0.25, 0.1, -0.15];
        let states = latent_binary_states_from_joint_beta(&family, &beta);
        // `exact_newton_joint_gradient_evaluation` takes states and specs as
        // parallel per-block arrays; an empty spec slice is an inconsistent
        // parameter partition, not "no opinion".
        let specs = latent_test_specs(family.event_target.len(), &[("time", 2), ("mean", 2)]);
        let h = 1e-6;

        let analytic_hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("analytic latent binary joint hessian evaluation")
            .expect("latent binary should expose exact joint hessian");

        for j in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[j] += h;
            let gradient_plus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_binary_states_from_joint_beta(&family, &beta_plus),
                    &specs,
                )
                .expect("joint gradient plus")
                .expect("joint gradient should exist")
                .gradient;

            let mut beta_minus = beta.clone();
            beta_minus[j] -= h;
            let gradient_minus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_binary_states_from_joint_beta(&family, &beta_minus),
                    &specs,
                )
                .expect("joint gradient minus")
                .expect("joint gradient should exist")
                .gradient;

            let fd_column = -((&gradient_plus - &gradient_minus) / (2.0 * h));
            let analytic_column = analytic_hessian.column(j).to_owned();
            let rel = max_relative_array1(&analytic_column, &fd_column);
            assert!(
                rel < 5e-4,
                "latent binary joint Hessian column {j} mismatch: rel={rel}, analytic={analytic_column:?}, fd={fd_column:?}"
            );
        }

        let workspace = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("latent binary hessian workspace")
            .expect("workspace should exist");
        let direction = array![0.4, -0.2, 0.3, 0.1];
        let hv = workspace
            .hessian_matvec(&direction)
            .expect("workspace matvec")
            .expect("workspace should support matvec");
        let dense_hv = analytic_hessian.dot(&direction);
        assert!(
            max_relative_array1(&hv, &dense_hv) < 1e-12,
            "latent binary workspace HVP mismatch: hv={hv:?}, dense={dense_hv:?}"
        );

        let dh = workspace
            .directional_derivative(&direction)
            .expect("workspace dH")
            .expect("workspace should support dH");
        let fd_step = 1e-5;
        let h_plus = family
            .exact_newton_joint_hessian(&latent_binary_states_from_joint_beta(
                &family,
                &(beta.clone() + &(fd_step * &direction)),
            ))
            .expect("hessian plus")
            .expect("hessian plus should exist");
        let h_minus = family
            .exact_newton_joint_hessian(&latent_binary_states_from_joint_beta(
                &family,
                &(beta - &(fd_step * &direction)),
            ))
            .expect("hessian minus")
            .expect("hessian minus should exist");
        let fd_dh = (&h_plus - &h_minus) / (2.0 * fd_step);
        assert!(
            max_relative_array2(&dh, &fd_dh) < 2e-4,
            "latent binary workspace dH mismatch: dh={dh:?}, fd={fd_dh:?}"
        );

        let direction_v = array![-0.15, 0.25, 0.08, -0.12];
        let d2h = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &direction,
                &direction_v,
            )
            .expect("latent binary d2H")
            .expect("latent binary should expose d2H");
        let d2_step = 4e-4;
        let dh_plus = family
            .exact_newton_joint_hessian_directional_derivative(
                &latent_binary_states_from_joint_beta(
                    &family,
                    &(array![0.15, 0.25, 0.1, -0.15] + d2_step * &direction_v),
                ),
                &direction,
            )
            .expect("latent binary dH plus")
            .expect("latent binary should expose dH plus");
        let dh_minus = family
            .exact_newton_joint_hessian_directional_derivative(
                &latent_binary_states_from_joint_beta(
                    &family,
                    &(array![0.15, 0.25, 0.1, -0.15] - d2_step * &direction_v),
                ),
                &direction,
            )
            .expect("latent binary dH minus")
            .expect("latent binary should expose dH minus");
        let fd_d2h = (dh_plus - dh_minus) / (2.0 * d2_step);
        let d2h_rel = frobenius_relative_array2(&d2h, &fd_d2h);
        assert!(
            d2h_rel < 2e-2,
            "latent binary combined TwoSeed d2H mismatch: rel={d2h_rel}, analytic={d2h:?}, fd={fd_d2h:?}"
        );
    }

    #[test]
    fn latent_survival_learnable_sigma_block_matches_family_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        // States and specs are parallel per-block arrays for this hook.
        let specs = latent_test_specs(
            family.event_target.len(),
            &[("time", 2), ("mean", 2), ("log_sigma", 1)],
        );
        let slices = family.joint_slices();
        let sigma_idx = slices
            .log_sigma
            .as_ref()
            .expect("learnable sigma test family should expose log_sigma")
            .start;
        let h = 2e-4;

        let eval = family
            .evaluate(&states)
            .expect("learnable latent survival evaluation");
        let joint_gradient = family
            .exact_newton_joint_gradient_evaluation(&states, &specs)
            .expect("joint gradient evaluation")
            .expect("joint gradient should exist")
            .gradient;
        let joint_hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian evaluation")
            .expect("joint hessian should exist");
        assert_eq!(eval.blockworking_sets.len(), 3);

        let (block_grad, block_neg_hess) =
            match &eval.blockworking_sets[LatentSurvivalFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, hessian } => {
                    let neg_hess = match hessian {
                        SymmetricMatrix::Dense(mat) => mat[[0, 0]],
                        _ => panic!("log_sigma block should use a dense exact-Newton Hessian"),
                    };
                    (gradient[0], neg_hess)
                }
                _ => panic!("log_sigma block should use ExactNewton"),
            };

        assert!((block_grad - joint_gradient[sigma_idx]).abs() < 1e-12);
        assert!((block_neg_hess - joint_hessian[[sigma_idx, sigma_idx]]).abs() < 1e-12);

        let mut beta_plus = beta.clone();
        beta_plus[sigma_idx] += h;
        let ll_plus = family
            .log_likelihood_only(&latent_survival_states_from_joint_beta(&family, &beta_plus))
            .expect("ll plus");
        let ll_0 = family.log_likelihood_only(&states).expect("ll base");
        let mut beta_minus = beta.clone();
        beta_minus[sigma_idx] -= h;
        let ll_minus = family
            .log_likelihood_only(&latent_survival_states_from_joint_beta(
                &family,
                &beta_minus,
            ))
            .expect("ll minus");

        let fd_grad = (ll_plus - ll_minus) / (2.0 * h);
        let fd_neg_hess = -(ll_plus - 2.0 * ll_0 + ll_minus) / (h * h);
        assert!(
            (joint_gradient[sigma_idx] - fd_grad).abs()
                / joint_gradient[sigma_idx]
                    .abs()
                    .max(fd_grad.abs())
                    .max(1e-12)
                < 2e-3,
            "family log_sigma grad={}, fd={fd_grad}",
            joint_gradient[sigma_idx]
        );
        assert!(
            (joint_hessian[[sigma_idx, sigma_idx]] - fd_neg_hess).abs()
                / joint_hessian[[sigma_idx, sigma_idx]]
                    .abs()
                    .max(fd_neg_hess.abs())
                    .max(1e-10)
                < 2e-2,
            "family log_sigma neg_hess={}, fd={fd_neg_hess}",
            joint_hessian[[sigma_idx, sigma_idx]]
        );
    }

    #[test]
    fn latent_survival_exact_joint_hessian_matches_gradient_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        // States and specs are parallel per-block arrays for this hook.
        let specs = latent_test_specs(
            family.event_target.len(),
            &[("time", 2), ("mean", 2), ("log_sigma", 1)],
        );
        let h = 1e-6;

        let analytic_hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("analytic joint hessian evaluation")
            .expect("latent survival should expose exact joint hessian");

        for j in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[j] += h;
            let gradient_plus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_survival_states_from_joint_beta(&family, &beta_plus),
                    &specs,
                )
                .expect("joint gradient plus")
                .expect("joint gradient should exist")
                .gradient;

            let mut beta_minus = beta.clone();
            beta_minus[j] -= h;
            let gradient_minus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_survival_states_from_joint_beta(&family, &beta_minus),
                    &specs,
                )
                .expect("joint gradient minus")
                .expect("joint gradient should exist")
                .gradient;

            let fd_column = (&gradient_plus - &gradient_minus) / (2.0 * h);
            let analytic_column = analytic_hessian.column(j).to_owned();
            let rel = max_relative_array1(&analytic_column, &(-fd_column));
            assert!(
                rel < 5e-4,
                "joint Hessian column {j} mismatch: rel={rel}, analytic={analytic_column:?}, fd={:?}",
                -((&gradient_plus - &gradient_minus) / (2.0 * h))
            );
        }
    }

    /// FD check for `LatentSurvivalFamily::offset_channel_residuals`: each
    /// channel residual sums to `∂(−ℓ)/∂o_ch` for a uniform additive offset on
    /// that time channel (the baseline-θ enters only through these offsets).
    /// `o_ch` shifts `eta_time[ch-slice]` uniformly, so `Σ_i r^ch_i` is exactly
    /// the directional derivative of `−ℓ` along a constant offset on channel ch.
    /// This validates the envelope-theorem latent baseline-θ gradient primitive.
    #[test]
    fn latent_survival_offset_channel_residuals_match_finite_difference() {
        let family = survival_stress_test_family(24);
        let beta = survival_stress_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let n = family.event_target.len();

        let residuals = family
            .offset_channel_residuals(&states)
            .expect("offset channel residuals");
        let sum_entry: f64 = residuals.entry.sum();
        let sum_exit: f64 = residuals.exit.sum();
        let sum_deriv: f64 = residuals.derivative.sum();

        #[derive(Clone, Copy)]
        enum TimeOffsetChannel {
            Entry,
            Exit,
            Derivative,
        }

        // `−ℓ` after shifting one time channel's eta by a constant δ.
        let neg_ll_with_offset = |channel: TimeOffsetChannel, delta: f64| -> f64 {
            let mut shifted = states.clone();
            let slice = match channel {
                TimeOffsetChannel::Entry => s![0..n],
                TimeOffsetChannel::Exit => s![n..2 * n],
                TimeOffsetChannel::Derivative => s![2 * n..3 * n],
            };
            shifted[LatentSurvivalFamily::BLOCK_TIME]
                .eta
                .slice_mut(slice)
                .mapv_inplace(|v| v + delta);
            let (ll, _) = family
                .evaluate_exact_newton_joint_gradient_dense(&shifted)
                .expect("shifted joint gradient evaluation");
            -ll
        };

        let h = 1e-6;
        let fd_entry = (neg_ll_with_offset(TimeOffsetChannel::Entry, h)
            - neg_ll_with_offset(TimeOffsetChannel::Entry, -h))
            / (2.0 * h);
        let fd_exit = (neg_ll_with_offset(TimeOffsetChannel::Exit, h)
            - neg_ll_with_offset(TimeOffsetChannel::Exit, -h))
            / (2.0 * h);
        let fd_deriv = (neg_ll_with_offset(TimeOffsetChannel::Derivative, h)
            - neg_ll_with_offset(TimeOffsetChannel::Derivative, -h))
            / (2.0 * h);

        assert!(
            (sum_entry - fd_entry).abs() <= 1e-5 * fd_entry.abs().max(1.0),
            "entry-channel residual sum mismatch: analytic={sum_entry}, fd={fd_entry}"
        );
        assert!(
            (sum_exit - fd_exit).abs() <= 1e-5 * fd_exit.abs().max(1.0),
            "exit-channel residual sum mismatch: analytic={sum_exit}, fd={fd_exit}"
        );
        assert!(
            (sum_deriv - fd_deriv).abs() <= 1e-5 * fd_deriv.abs().max(1.0),
            "derivative-channel residual sum mismatch: analytic={sum_deriv}, fd={fd_deriv}"
        );
    }

    #[test]
    fn latent_survival_exact_joint_parallel_stress_is_repeatable() {
        let family = survival_stress_test_family(96);
        let beta = survival_stress_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let direction_u = array![0.03, -0.02, 0.01, 0.04, -0.015, 0.025, -0.005, 0.02];
        let direction_v = array![-0.01, 0.035, -0.025, 0.015, 0.02, -0.01, 0.03, -0.015];

        let (ll_a, grad_a) = family
            .evaluate_exact_newton_joint_gradient_dense(&states)
            .expect("stress joint gradient evaluation");
        let (ll_b, grad_b) = family
            .evaluate_exact_newton_joint_gradient_dense(&states)
            .expect("repeat stress joint gradient evaluation");
        assert_eq!(ll_a.to_bits(), ll_b.to_bits());
        assert_eq!(grad_a, grad_b);

        let (joint_ll_a, joint_grad_a, hess_a) = family
            .evaluate_exact_newton_joint_dense(&states)
            .expect("stress joint dense evaluation");
        let (joint_ll_b, joint_grad_b, hess_b) = family
            .evaluate_exact_newton_joint_dense(&states)
            .expect("repeat stress joint dense evaluation");
        assert_eq!(joint_ll_a.to_bits(), joint_ll_b.to_bits());
        assert_eq!(joint_grad_a, joint_grad_b);
        assert_eq!(hess_a, hess_b);
        assert!(hess_a.iter().all(|value| value.is_finite()));
        assert!(max_relative_array2(&hess_a, &hess_a.t().to_owned()) < 1e-12);

        let dh_a = family
            .exact_newton_joint_hessian_directional_derivative_dense(&states, &direction_u)
            .expect("stress joint dH evaluation");
        let dh_b = family
            .exact_newton_joint_hessian_directional_derivative_dense(&states, &direction_u)
            .expect("repeat stress joint dH evaluation");
        assert_eq!(dh_a, dh_b);
        assert!(dh_a.iter().all(|value| value.is_finite()));
        assert!(max_relative_array2(&dh_a, &dh_a.t().to_owned()) < 1e-12);

        let d2h_a = family
            .exact_newton_joint_hessian_second_directional_derivative_dense(
                &states,
                &direction_u,
                &direction_v,
            )
            .expect("stress joint d2H evaluation");
        let d2h_b = family
            .exact_newton_joint_hessian_second_directional_derivative_dense(
                &states,
                &direction_u,
                &direction_v,
            )
            .expect("repeat stress joint d2H evaluation");
        assert_eq!(d2h_a, d2h_b);
        assert!(d2h_a.iter().all(|value| value.is_finite()));
        assert!(max_relative_array2(&d2h_a, &d2h_a.t().to_owned()) < 1e-12);
    }

    #[test]
    fn latent_survival_exact_joint_dh_matches_hessian_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let h = 2e-4;
        let direction = array![0.07, -0.03, 0.05, 0.02, -0.04];

        let analytic = family
            .exact_newton_joint_hessian_directional_derivative(&states, &direction)
            .expect("analytic joint dH evaluation")
            .expect("latent survival should expose exact joint dH");

        let hessian_plus = family
            .exact_newton_joint_hessian(&latent_survival_states_from_joint_beta(
                &family,
                &(beta.clone() + h * &direction),
            ))
            .expect("joint hessian plus")
            .expect("joint hessian should exist");
        let hessian_minus = family
            .exact_newton_joint_hessian(&latent_survival_states_from_joint_beta(
                &family,
                &(beta.clone() - h * &direction),
            ))
            .expect("joint hessian minus")
            .expect("joint hessian should exist");

        let fd = (&hessian_plus - &hessian_minus) / (2.0 * h);
        let rel = frobenius_relative_array2(&analytic, &fd);
        assert!(rel < 2e-3, "joint dH mismatch: rel={rel}");
    }

    #[test]
    fn latent_survival_exact_joint_d2h_matches_directional_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let h = 5e-4;
        let direction_u = array![0.07, -0.03, 0.05, 0.02, -0.04];
        let direction_v = array![-0.02, 0.06, -0.01, 0.03, 0.05];

        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &direction_u,
                &direction_v,
            )
            .expect("analytic joint d2H evaluation")
            .expect("latent survival should expose exact joint d2H");
        let swapped = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &direction_v,
                &direction_u,
            )
            .expect("swapped analytic joint d2H evaluation")
            .expect("latent survival should expose exact joint d2H");
        let symmetry_rel = max_relative_array2(&analytic, &swapped);
        assert!(
            symmetry_rel < 1e-10,
            "joint d2H should be symmetric in directions, got rel={symmetry_rel}"
        );

        let dh_plus = family
            .exact_newton_joint_hessian_directional_derivative(
                &latent_survival_states_from_joint_beta(
                    &family,
                    &(beta.clone() + h * &direction_v),
                ),
                &direction_u,
            )
            .expect("joint dH plus")
            .expect("joint dH should exist");
        let dh_minus = family
            .exact_newton_joint_hessian_directional_derivative(
                &latent_survival_states_from_joint_beta(
                    &family,
                    &(beta.clone() - h * &direction_v),
                ),
                &direction_u,
            )
            .expect("joint dH minus")
            .expect("joint dH should exist");

        let fd = (&dh_plus - &dh_minus) / (2.0 * h);
        let rel = frobenius_relative_array2(&analytic, &fd);
        assert!(rel < 2.5e-2, "joint d2H mismatch: rel={rel}");
    }

    #[test]
    fn latent_survival_row_primary_derivatives_match_fd() {
        let quadctx = QuadratureContext::new();
        let row = LatentSurvivalRow::exact_event(0.35, 1.4, 0.1, 0.45, 0.8, 0.12);
        // [q_entry, q_exit, qdot_exit, q_right, mu, log_sigma]. This is an
        // exact-event row, so the `q_right` channel is inert (the likelihood
        // does not depend on it); the FD loop below confirms its gradient/Hessian
        // entries are zero.
        let primary = array![
            0.35f64.ln(),
            1.4f64.ln(),
            0.8,
            1.6f64.ln(),
            -0.2,
            0.4f64.ln()
        ];
        let sigma = primary[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA].exp();
        let h_grad = 1e-6;
        let h_hess = 2e-4;

        let (_, gradient, neg_hessian) = latent_survival_row_primary_gradient_hessian(
            &quadctx,
            &row,
            LatentSurvivalPrimaryPoint {
                q_entry: primary[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
                q_exit: primary[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
                qdot_exit: primary[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
                q_right: primary[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
                mu: primary[LATENT_SURVIVAL_PRIMARY_MU],
                sigma,
            },
            true,
        )
        .expect("analytic row primary gradient/hessian");

        for j in 0..LATENT_SURVIVAL_PRIMARY_DIM {
            let mut plus = primary.clone();
            plus[j] += h_grad;
            let mut minus = primary.clone();
            minus[j] -= h_grad;
            let fd_grad = (latent_survival_row_loglik_from_primary(&quadctx, &row, &plus)
                - latent_survival_row_loglik_from_primary(&quadctx, &row, &minus))
                / (2.0 * h_grad);
            let rel_grad =
                (gradient[j] - fd_grad).abs() / gradient[j].abs().max(fd_grad.abs()).max(1e-12);
            assert!(
                rel_grad < 2e-4,
                "row primary grad[{j}] mismatch: analytic={}, fd={fd_grad}, rel={rel_grad}",
                gradient[j]
            );

            for k in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                let mut pp = primary.clone();
                pp[j] += h_hess;
                pp[k] += h_hess;
                let mut pm = primary.clone();
                pm[j] += h_hess;
                pm[k] -= h_hess;
                let mut mp = primary.clone();
                mp[j] -= h_hess;
                mp[k] += h_hess;
                let mut mm = primary.clone();
                mm[j] -= h_hess;
                mm[k] -= h_hess;
                let fd_neg_hess = -(latent_survival_row_loglik_from_primary(&quadctx, &row, &pp)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &pm)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &mp)
                    + latent_survival_row_loglik_from_primary(&quadctx, &row, &mm))
                    / (4.0 * h_hess * h_hess);
                let analytic = neg_hessian[[j, k]];
                let abs_err = (analytic - fd_neg_hess).abs();
                let rel = abs_err / analytic.abs().max(fd_neg_hess.abs()).max(1e-10);
                assert!(
                    abs_err < 2e-5 || rel < 2e-3,
                    "row primary neg_hess[{j},{k}] mismatch: analytic={analytic}, fd={fd_neg_hess}, abs_err={abs_err}, rel={rel}"
                );
            }
        }
    }

    #[test]
    fn latent_survival_interval_row_primary_derivatives_match_fd() {
        // Interval-censored row jet `ℓ = log[S(L) − S(R)] − log S(entry)`. The
        // dynamic two-state numerator differentiates BOTH boundary masses
        // `M_L = exp(q_exit)` (left, `q_exit`) and `M_R = exp(q_right)` (right,
        // `q_right`) independently — channels that the static
        // `LatentSurvivalRowJet::interval_censored` (μ-only) never exercises. This
        // FD-verifies the gradient AND neg-Hessian of the interval contribution
        // w.r.t. ALL six primary coordinates (q_entry, q_exit/L, qdot_exit,
        // q_right/R, mu, log_sigma) on a WELL-POSED bracket where `S(L) − S(R)` is
        // comfortably positive (M_L = e^{−0.4} ≈ 0.67 well below M_R = e^{0.5} ≈
        // 1.65, so the survival-mass difference is large and the log-of-a-
        // difference curvature is well-conditioned).
        let quadctx = QuadratureContext::new();
        // Bracket masses: entry < L < R with comfortable gaps.
        let q_entry = -1.2_f64; // M_entry = e^{−1.2} ≈ 0.30
        let q_exit = -0.4_f64; // L: M_L = e^{−0.4} ≈ 0.67
        let q_right = 0.5_f64; // R: M_R = e^{0.5} ≈ 1.65 (> M_L)
        let mu = -0.15_f64;
        let log_sigma = 0.3_f64; // σ ≈ 1.35
        // Small, monotone unloaded masses (entry ≤ left ≤ right); qdot is inert
        // for the interval contribution.
        let row = LatentSurvivalRow::interval_censored(
            q_entry.exp(), // mass_entry (consistency only; jet reads q's)
            q_exit.exp(),  // mass_left
            q_right.exp(), // mass_right
            0.01,          // mass_unloaded_entry
            0.02,          // mass_unloaded_left
            0.05,          // mass_unloaded_right
        );
        assert!(matches!(
            row.event_type,
            LatentSurvivalEventType::IntervalCensored
        ));

        // [q_entry, q_exit/L, qdot_exit, q_right/R, mu, log_sigma]. qdot_exit is
        // inert for interval rows (no hazard-derivative channel); the FD loop
        // confirms its gradient/Hessian entries are 0.
        let primary = array![q_entry, q_exit, 0.7, q_right, mu, log_sigma];
        let sigma = primary[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA].exp();
        let h_grad = 1e-6;
        let h_hess = 2e-4;

        let (_, gradient, neg_hessian) = latent_survival_row_primary_gradient_hessian(
            &quadctx,
            &row,
            LatentSurvivalPrimaryPoint {
                q_entry: primary[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
                q_exit: primary[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
                qdot_exit: primary[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
                q_right: primary[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
                mu: primary[LATENT_SURVIVAL_PRIMARY_MU],
                sigma,
            },
            true,
        )
        .expect("analytic interval row primary gradient/hessian");

        // The interval contribution must be a positive survival-mass difference
        // at this bracket, so the value channel is finite.
        let value = latent_survival_row_loglik_from_primary(&quadctx, &row, &primary);
        assert!(
            value.is_finite(),
            "interval row log-likelihood must be finite on a well-posed bracket, got {value}"
        );

        for j in 0..LATENT_SURVIVAL_PRIMARY_DIM {
            let mut plus = primary.clone();
            plus[j] += h_grad;
            let mut minus = primary.clone();
            minus[j] -= h_grad;
            let fd_grad = (latent_survival_row_loglik_from_primary(&quadctx, &row, &plus)
                - latent_survival_row_loglik_from_primary(&quadctx, &row, &minus))
                / (2.0 * h_grad);
            let rel_grad =
                (gradient[j] - fd_grad).abs() / gradient[j].abs().max(fd_grad.abs()).max(1e-12);
            assert!(
                rel_grad < 2e-4,
                "interval row primary grad[{j}] mismatch: analytic={}, fd={fd_grad}, rel={rel_grad}",
                gradient[j]
            );

            for k in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                let mut pp = primary.clone();
                pp[j] += h_hess;
                pp[k] += h_hess;
                let mut pm = primary.clone();
                pm[j] += h_hess;
                pm[k] -= h_hess;
                let mut mp = primary.clone();
                mp[j] -= h_hess;
                mp[k] += h_hess;
                let mut mm = primary.clone();
                mm[j] -= h_hess;
                mm[k] -= h_hess;
                let fd_neg_hess = -(latent_survival_row_loglik_from_primary(&quadctx, &row, &pp)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &pm)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &mp)
                    + latent_survival_row_loglik_from_primary(&quadctx, &row, &mm))
                    / (4.0 * h_hess * h_hess);
                let analytic = neg_hessian[[j, k]];
                let abs_err = (analytic - fd_neg_hess).abs();
                let rel = abs_err / analytic.abs().max(fd_neg_hess.abs()).max(1e-10);
                assert!(
                    abs_err < 5e-5 || rel < 3e-3,
                    "interval row primary neg_hess[{j},{k}] mismatch: analytic={analytic}, fd={fd_neg_hess}, abs_err={abs_err}, rel={rel}"
                );
            }
        }
    }

    type LatentFullPrimaryChannels = (f64, Array1<f64>, Array2<f64>, Array2<f64>, Array2<f64>);

    fn latent_full_primary_channels_at(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        include_log_sigma: bool,
        use_multidir_reference: bool,
        point: LatentSurvivalPrimaryPoint,
    ) -> LatentFullPrimaryChannels {
        let direction_u = array![
            0.17,
            -0.11,
            0.09,
            0.13,
            -0.07,
            if include_log_sigma { 0.05 } else { 0.0 }
        ];
        let direction_v = array![
            -0.08,
            0.14,
            -0.06,
            0.04,
            0.12,
            if include_log_sigma { -0.09 } else { 0.0 }
        ];
        if use_multidir_reference {
            let (value, gradient, hessian) =
                latent_survival_row_primary_gradient_hessian_multidir_reference(
                    quadctx,
                    row,
                    point,
                    include_log_sigma,
                )
                .expect("pre-cutover MultiDirJet VGH reference");
            let third = latent_survival_row_primary_third_contracted_multidir_reference(
                quadctx,
                row,
                point,
                &direction_u,
                include_log_sigma,
            )
            .expect("pre-cutover MultiDirJet third reference");
            let fourth = latent_survival_row_primary_fourth_contracted_multidir_reference(
                quadctx,
                row,
                point,
                &direction_u,
                &direction_v,
                include_log_sigma,
            )
            .expect("pre-cutover MultiDirJet fourth reference");
            (value, gradient, hessian, third, fourth)
        } else {
            let (value, gradient, hessian) = latent_survival_row_primary_gradient_hessian(
                quadctx,
                row,
                point,
                include_log_sigma,
            )
            .expect("one-pass Order2 VGH");
            let third = latent_survival_row_primary_third_contracted(
                quadctx,
                row,
                point,
                &direction_u,
                include_log_sigma,
            )
            .expect("one-pass OneSeed third");
            let fourth = latent_survival_row_primary_fourth_contracted(
                quadctx,
                row,
                point,
                &direction_u,
                &direction_v,
                include_log_sigma,
            )
            .expect("one-pass TwoSeed fourth");
            (value, gradient, hessian, third, fourth)
        }
    }

    fn latent_full_primary_channels(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        include_log_sigma: bool,
        use_multidir_reference: bool,
    ) -> LatentFullPrimaryChannels {
        latent_full_primary_channels_at(
            quadctx,
            row,
            include_log_sigma,
            use_multidir_reference,
            LatentSurvivalPrimaryPoint {
                q_entry: -1.2,
                q_exit: -0.4,
                qdot_exit: 0.73,
                q_right: 0.5,
                mu: -0.15,
                sigma: 0.3_f64.exp(),
            },
        )
    }

    fn assert_latent_full_channels_close(
        label: &str,
        got: &LatentFullPrimaryChannels,
        reference: &LatentFullPrimaryChannels,
    ) {
        const ABSOLUTE_TOLERANCE: f64 = 5e-11;
        const RELATIVE_TOLERANCE: f64 = 5e-10;

        // Absolute tolerance governs channels near zero; relative tolerance
        // governs channels with live scale.  Reducing their maxima separately
        // and OR-ing those two maxima is not equivalent to applying that rule
        // channel by channel: two different, individually valid channels can
        // supply the two maxima and make the aggregate fail.
        let mut worst_scaled_error = 0.0_f64;
        let mut worst_channel = String::new();
        let mut worst_values = (0.0_f64, 0.0_f64);
        let mut worst_absolute = 0.0_f64;
        let mut worst_relative = 0.0_f64;
        let mut record = |channel: String, left: f64, right: f64| {
            let absolute = (left - right).abs();
            let scale = left.abs().max(right.abs());
            let relative = if !left.is_finite() || !right.is_finite() {
                f64::INFINITY
            } else if scale == 0.0 {
                0.0
            } else {
                absolute / scale
            };
            let scaled_error = if !left.is_finite() || !right.is_finite() {
                f64::INFINITY
            } else {
                (absolute / ABSOLUTE_TOLERANCE).min(relative / RELATIVE_TOLERANCE)
            };
            if scaled_error > worst_scaled_error {
                worst_scaled_error = scaled_error;
                worst_channel = channel;
                worst_values = (left, right);
                worst_absolute = absolute;
                worst_relative = relative;
            }
        };
        record("value".to_string(), got.0, reference.0);
        for (a, (&left, &right)) in got.1.iter().zip(reference.1.iter()).enumerate() {
            record(format!("gradient[{a}]"), left, right);
        }
        for ((a, b), &left) in got.2.indexed_iter() {
            record(format!("hessian[{a},{b}]"), left, reference.2[[a, b]]);
        }
        for ((a, b), &left) in got.3.indexed_iter() {
            record(format!("third[{a},{b}]"), left, reference.3[[a, b]]);
        }
        for ((a, b), &left) in got.4.indexed_iter() {
            record(format!("fourth[{a},{b}]"), left, reference.4[[a, b]]);
        }
        assert!(
            worst_scaled_error <= 1.0,
            "{label}: one-pass channels differ from the pre-cutover MultiDirJet oracle: \
             worst mixed error={worst_scaled_error:e} at {worst_channel} \
             (one-pass={}, oracle={}, abs={worst_absolute:e}, rel={worst_relative:e}, \
             abs_tol={ABSOLUTE_TOLERANCE:e}, rel_tol={RELATIVE_TOLERANCE:e})",
            worst_values.0,
            worst_values.1,
        );
    }

    /// Pins the one-pass scalar layouts to the complete pre-cutover output on
    /// every event branch and at both live primary dimensions.  This is stronger
    /// than an FD-only oracle: it covers value, gradient, negative Hessian,
    /// contracted third, and contracted fourth simultaneously.
    #[test]
    fn latent_survival_one_pass_matches_multidir_all_events_all_channels_932() {
        let quadctx = QuadratureContext::new();
        let rows = [
            (
                "right",
                LatentSurvivalRow::right_censored(0.3, 0.67, 0.01, 0.02),
            ),
            (
                "exact",
                LatentSurvivalRow::exact_event(0.3, 0.67, 0.01, 0.02, 0.73, 0.08),
            ),
            (
                "interval",
                LatentSurvivalRow::interval_censored(0.3, 0.67, 1.65, 0.01, 0.02, 0.05),
            ),
        ];
        for (event, row) in &rows {
            for include_log_sigma in [false, true] {
                let reference =
                    latent_full_primary_channels(&quadctx, row, include_log_sigma, true);
                let got = latent_full_primary_channels(&quadctx, row, include_log_sigma, false);
                let dimension = if include_log_sigma { 6 } else { 5 };
                assert_latent_full_channels_close(
                    &format!("event={event}, K={dimension}"),
                    &got,
                    &reference,
                );
            }
        }
    }

    /// Exact-event tail audit: small/large loaded masses, displaced frailty
    /// locations, narrow/wide scales, and the learnable-scale axis remain
    /// representable in both implementations.
    ///
    /// This deliberately does not assert floating output against floating
    /// output. Every production cumulant is the uniquely rounded exact
    /// recurrence over its rounded moments; the test-only MultiDir primitive
    /// independently proves its output lies in a `γ_n·Σ|term|` band derived
    /// from its 457-operation K=4 schedule. In a large-mass K=6 channel the two
    /// certified bands legitimately differ by 1.9e-9, so equality would make
    /// the less accurate implementation the specification. The ordinary-scale
    /// all-event test above still cross-checks the shared row algebra.
    #[test]
    fn latent_survival_exact_tails_are_independently_certified_all_channels_932() {
        let quadctx = QuadratureContext::new();
        let regimes: [(&str, f64, f64, f64, f64, f64, f64); 3] = [
            (
                "tiny-mass-left",
                -14.0_f64,
                -10.0_f64,
                0.31,
                0.0,
                -5.0,
                0.08,
            ),
            ("large-mass-right", 2.0, 6.0, 1.7, 0.0, 3.5, 0.45),
            ("wide-frailty", -3.0, 1.5, 0.62, 0.0, -1.8, 4.0),
        ];
        for (name, q_entry, q_exit, qdot, q_right, mu, sigma) in regimes {
            let row =
                LatentSurvivalRow::exact_event(q_entry.exp(), q_exit.exp(), 0.01, 0.04, qdot, 0.07);
            let point = LatentSurvivalPrimaryPoint {
                q_entry,
                q_exit,
                qdot_exit: qdot,
                q_right,
                mu,
                sigma,
            };
            for include_log_sigma in [false, true] {
                let reference =
                    latent_full_primary_channels_at(&quadctx, &row, include_log_sigma, true, point);
                let got = latent_full_primary_channels_at(
                    &quadctx,
                    &row,
                    include_log_sigma,
                    false,
                    point,
                );
                let all_finite = |channels: &LatentFullPrimaryChannels| {
                    channels.0.is_finite()
                        && channels.1.iter().all(|value| value.is_finite())
                        && channels.2.iter().all(|value| value.is_finite())
                        && channels.3.iter().all(|value| value.is_finite())
                        && channels.4.iter().all(|value| value.is_finite())
                };
                let dimension = if include_log_sigma { 6 } else { 5 };
                assert!(
                    all_finite(&got),
                    "exact-tail={name}, K={dimension}: certified one-pass output is non-finite"
                );
                assert!(
                    all_finite(&reference),
                    "exact-tail={name}, K={dimension}: forward-error-graded MultiDir output is non-finite"
                );
            }
        }
    }

    #[test]
    fn latent_survival_derivative_support_stays_inline_932() {
        let base_terms = [
            LatentKernelPrimaryTerm {
                coeff: 0.08,
                q_exp: 0,
                qdot_power: 0,
                tau_exp: 0,
                k: 0,
            },
            LatentKernelPrimaryTerm {
                coeff: 1.0,
                q_exp: 1,
                qdot_power: 1,
                tau_exp: 0,
                k: 1,
            },
        ];
        let primary: [LatentKernelPrimaryDirection; LATENT_SURVIVAL_PRIMARY_DIM] =
            std::array::from_fn(|a| {
                latent_survival_map_exit_direction(
                    latent_survival_basis_direction(a),
                    LatentSurvivalEventType::ExactEvent,
                )
            });
        let u_coeff = [0.17, -0.11, 0.09, 0.13, -0.07, 0.05];
        let v_coeff = [-0.08, 0.14, -0.06, 0.04, 0.12, -0.09];
        let u = latent_kernel_direction_linear_combination(&primary, &u_coeff);
        let v = latent_kernel_direction_linear_combination(&primary, &v_coeff);
        let suffix_u = [u];
        let suffix_v = [v];
        let suffix_uv = [u, v];
        let suffixes: [&[LatentKernelPrimaryDirection]; 4] =
            [&[], &suffix_u, &suffix_v, &suffix_uv];
        let mut maximum_support = 0usize;
        for suffix in suffixes {
            for a in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                for b in a..LATENT_SURVIVAL_PRIMARY_DIM {
                    let terms = latent_kernel_term_sequence_inline(
                        &base_terms,
                        &[primary[a], primary[b]],
                        suffix,
                    );
                    assert!(!terms.spilled(), "derivative term support spilled to heap");
                    maximum_support = maximum_support.max(terms.len());
                }
            }
        }
        eprintln!(
            "LATENT-ONE-PASS-932 derivative-support max_terms={maximum_support} inline_capacity={LATENT_TERM_INLINE_CAPACITY} heap_allocations=0"
        );
        assert!(maximum_support <= LATENT_TERM_INLINE_CAPACITY);
    }

    fn best_elapsed_seconds(mut run: impl FnMut(), iterations: usize, samples: usize) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..samples {
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                run();
            }
            best = best.min(started.elapsed().as_secs_f64());
        }
        best
    }

    fn measured_channel_ratio(
        mut reference: impl FnMut(),
        mut one_pass: impl FnMut(),
        iterations: usize,
        samples: usize,
    ) -> (f64, f64, f64) {
        reference();
        one_pass();
        let reference_seconds = best_elapsed_seconds(&mut reference, iterations, samples);
        let one_pass_seconds = best_elapsed_seconds(&mut one_pass, iterations, samples);
        (
            reference_seconds * 1e6 / iterations as f64,
            one_pass_seconds * 1e6 / iterations as f64,
            one_pass_seconds / reference_seconds,
        )
    }

    /// Full-output pre-cutover benchmark: the baseline includes all 21/28 VGH
    /// row sweeps and all 15/21 third/fourth pair sweeps; the candidate returns
    /// the identical five-channel payload via one Order2, one OneSeed, and one
    /// TwoSeed row evaluation.  Run with `--release -- --nocapture` for the
    /// reported ratios; the debug configuration uses one iteration to keep the
    /// ordinary suite bounded while still enforcing the direction of the win.
    #[test]
    fn measure_latent_survival_one_pass_full_output_k5_k6_932() {
        let quadctx = QuadratureContext::new();
        let row = LatentSurvivalRow::exact_event(0.3, 0.67, 0.01, 0.02, 0.73, 0.08);
        let q_entry = -1.2;
        let q_exit = -0.4;
        let qdot_exit = 0.73;
        let q_right = 0.5;
        let mu = -0.15;
        let sigma = 0.3_f64.exp();
        let point = LatentSurvivalPrimaryPoint {
            q_entry,
            q_exit,
            qdot_exit,
            q_right,
            mu,
            sigma,
        };
        let iterations = if cfg!(debug_assertions) { 1 } else { 5 };
        let samples = if cfg!(debug_assertions) { 1 } else { 3 };

        for include_log_sigma in [false, true] {
            let direction_u = array![
                0.17,
                -0.11,
                0.09,
                0.13,
                -0.07,
                if include_log_sigma { 0.05 } else { 0.0 }
            ];
            let direction_v = array![
                -0.08,
                0.14,
                -0.06,
                0.04,
                0.12,
                if include_log_sigma { -0.09 } else { 0.0 }
            ];
            let dimension = if include_log_sigma { 6 } else { 5 };
            let (vgh_reference_us, vgh_one_pass_us, vgh_ratio) = measured_channel_ratio(
                || {
                    std::hint::black_box(
                        latent_survival_row_primary_gradient_hessian_multidir_reference(
                            std::hint::black_box(&quadctx),
                            std::hint::black_box(&row),
                            point,
                            include_log_sigma,
                        )
                        .expect("prechange VGH benchmark"),
                    );
                },
                || {
                    std::hint::black_box(
                        latent_survival_row_primary_gradient_hessian(
                            std::hint::black_box(&quadctx),
                            std::hint::black_box(&row),
                            point,
                            include_log_sigma,
                        )
                        .expect("one-pass VGH benchmark"),
                    );
                },
                iterations,
                samples,
            );
            let (third_reference_us, third_one_pass_us, third_ratio) = measured_channel_ratio(
                || {
                    std::hint::black_box(
                        latent_survival_row_primary_third_contracted_multidir_reference(
                            std::hint::black_box(&quadctx),
                            std::hint::black_box(&row),
                            point,
                            std::hint::black_box(&direction_u),
                            include_log_sigma,
                        )
                        .expect("prechange third benchmark"),
                    );
                },
                || {
                    std::hint::black_box(
                        latent_survival_row_primary_third_contracted(
                            std::hint::black_box(&quadctx),
                            std::hint::black_box(&row),
                            point,
                            std::hint::black_box(&direction_u),
                            include_log_sigma,
                        )
                        .expect("one-pass third benchmark"),
                    );
                },
                iterations,
                samples,
            );
            let (fourth_reference_us, fourth_one_pass_us, fourth_ratio) = measured_channel_ratio(
                || {
                    std::hint::black_box(
                        latent_survival_row_primary_fourth_contracted_multidir_reference(
                            std::hint::black_box(&quadctx),
                            std::hint::black_box(&row),
                            point,
                            std::hint::black_box(&direction_u),
                            std::hint::black_box(&direction_v),
                            include_log_sigma,
                        )
                        .expect("prechange fourth benchmark"),
                    );
                },
                || {
                    std::hint::black_box(
                        latent_survival_row_primary_fourth_contracted(
                            std::hint::black_box(&quadctx),
                            std::hint::black_box(&row),
                            point,
                            std::hint::black_box(&direction_u),
                            std::hint::black_box(&direction_v),
                            include_log_sigma,
                        )
                        .expect("one-pass fourth benchmark"),
                    );
                },
                iterations,
                samples,
            );
            let (full_reference_us, full_one_pass_us, full_ratio) = measured_channel_ratio(
                || {
                    std::hint::black_box(latent_full_primary_channels(
                        std::hint::black_box(&quadctx),
                        std::hint::black_box(&row),
                        include_log_sigma,
                        true,
                    ));
                },
                || {
                    std::hint::black_box(latent_full_primary_channels(
                        std::hint::black_box(&quadctx),
                        std::hint::black_box(&row),
                        include_log_sigma,
                        false,
                    ));
                },
                iterations,
                samples,
            );
            let (combined_reference_us, combined_one_pass_us, combined_ratio) =
                measured_channel_ratio(
                    || {
                        std::hint::black_box(
                            latent_survival_row_primary_gradient_hessian_multidir_reference(
                                &quadctx,
                                &row,
                                point,
                                include_log_sigma,
                            )
                            .expect("prechange combined VGH"),
                        );
                        for direction in [&direction_u, &direction_v] {
                            std::hint::black_box(
                                latent_survival_row_primary_third_contracted_multidir_reference(
                                    &quadctx,
                                    &row,
                                    point,
                                    direction,
                                    include_log_sigma,
                                )
                                .expect("prechange combined third"),
                            );
                        }
                        std::hint::black_box(
                            latent_survival_row_primary_fourth_contracted_multidir_reference(
                                &quadctx,
                                &row,
                                point,
                                &direction_u,
                                &direction_v,
                                include_log_sigma,
                            )
                            .expect("prechange combined fourth"),
                        );
                    },
                    || {
                        if include_log_sigma {
                            let backend = LatentTwoSeedBackend {
                                direction_u: std::array::from_fn(|a| direction_u[a]),
                                direction_v: std::array::from_fn(|a| direction_v[a]),
                            };
                            std::hint::black_box(
                                latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
                                    &backend, &quadctx, &row, point,
                                )
                                .expect("combined K6 TwoSeed"),
                            );
                        } else {
                            std::hint::black_box(
                                latent_survival_row_primary_two_seed_fixed_sigma(
                                    &quadctx,
                                    &row,
                                    point,
                                    &direction_u,
                                    &direction_v,
                                )
                                .expect("combined K5 TwoSeed"),
                            );
                        }
                    },
                    iterations,
                    samples,
                );
            let pair_count = dimension * (dimension + 1) / 2;
            let order2_width = 1 + dimension + pair_count;
            let vgh_reference_bundle_allocs = 2 * (1 + dimension + pair_count);
            let contracted_reference_bundle_allocs = 2 * pair_count;
            eprintln!(
                "LATENT-ONE-PASS-OPS-932 K={dimension} signed-term-reductions/state VGH {}->{order2_width} T3 {}->{} T4 {}->{}",
                1 + 2 * dimension + 4 * pair_count,
                8 * pair_count,
                2 * order2_width,
                16 * pair_count,
                4 * order2_width,
            );
            eprintln!(
                "LATENT-ONE-PASS-932 K={dimension} VGH prechange={vgh_reference_us:.3}us one-pass={vgh_one_pass_us:.3}us ratio={vgh_ratio:.4} speedup={:.2}x bundle-Vec-allocs={vgh_reference_bundle_allocs}->2; T3 prechange={third_reference_us:.3}us one-pass={third_one_pass_us:.3}us ratio={third_ratio:.4} speedup={:.2}x bundle-Vec-allocs={contracted_reference_bundle_allocs}->2; T4 prechange={fourth_reference_us:.3}us one-pass={fourth_one_pass_us:.3}us ratio={fourth_ratio:.4} speedup={:.2}x bundle-Vec-allocs={contracted_reference_bundle_allocs}->2; FULL-3PASS prechange={full_reference_us:.3}us one-pass={full_one_pass_us:.3}us ratio={full_ratio:.4} speedup={:.2}x bundle-Vec-allocs={}->6; FULL-COMBINED prechange={combined_reference_us:.3}us one-pass={combined_one_pass_us:.3}us ratio={combined_ratio:.4} speedup={:.2}x bundle-Vec-allocs={}->2; derivative-plan-heap-allocs=0 three-pass-output-ndarray-allocs=4->4 combined-output-ndarray-allocs=5->0",
                1.0 / vgh_ratio,
                1.0 / third_ratio,
                1.0 / fourth_ratio,
                1.0 / full_ratio,
                vgh_reference_bundle_allocs + 2 * contracted_reference_bundle_allocs,
                1.0 / combined_ratio,
                vgh_reference_bundle_allocs + 3 * contracted_reference_bundle_allocs,
            );
            for (channel, ratio) in [
                ("VGH", vgh_ratio),
                ("T3", third_ratio),
                ("T4", fourth_ratio),
                ("full-3pass", full_ratio),
                ("full-combined", combined_ratio),
            ] {
                // #932: release-only. Every entry here is a wall-clock ratio,
                // and without optimization the one-pass schedule's advantage is
                // not generated -- measured in the default test lane, K=5 VGH
                // reports ratio=3.2125, i.e. three times SLOWER, which is the
                // missing optimizer rather than a pessimization. The channel
                // values themselves are checked for exactness by
                // `latent_survival_one_pass_exact_tails_match_multidir_all_channels_932`,
                // which is build-independent and unaffected by this gate.
                assert!(
                    cfg!(debug_assertions) || ratio < 1.0,
                    "K={dimension} one-pass {channel} must beat the exact pre-cutover path: ratio={ratio}"
                );
            }
        }
    }

    #[test]
    fn latent_signed_log_cumulant_preserves_cancellation_2566() {
        // m_ab and m_a*m_b are about 7.9e13, while their cumulant is about
        // 7.4e4. The deliberately tiny separation is represented in log space;
        // the production recurrence must not exponentiate either large moment
        // before subtracting them.
        let separation = 2.0_f64.powi(-30);
        let mut moments = [LatentSignedLog::ZERO; 16];
        moments[0] = LatentSignedLog::ONE;
        moments[0b0001] = LatentSignedLog {
            log_abs: 16.0,
            sign: 1.0,
        };
        moments[0b0010] = moments[0b0001];
        moments[0b0011] = LatentSignedLog {
            log_abs: 32.0 + separation,
            sign: 1.0,
        };

        let got = latent_certified_cumulants(moments, 0b0011, "2566 cancellation oracle")
            .expect("certified finite cumulant")[0b0011]
            .value;
        // 100-decimal exact recurrence over the rounded binary64 moments. A
        // direct binary64 subtraction rounds each ~7.9e13 monomial before their
        // cancellation and returns 73540.0, so it is not an exact oracle.
        let expected = 73_540.001_795_945_1_f64;
        assert_eq!(
            got, expected,
            "certified cumulant changed its unique rounded value"
        );

        assert!(
            latent_signed_log_normalized(f64::INFINITY, 1.0, 0.0, "2566 nonfinite oracle")
                .is_err(),
            "a non-finite derivative magnitude must be refused, never rewritten as zero"
        );
    }

    #[test]
    fn latent_cumulant_expansion_certifies_ill_conditioned_k6_channel_2597() {
        // Exact rounded moment inputs dumped from the large-mass K=6
        // fourth[1,5] channel. The pre-cutover MultiDir composition returned
        // -1.1184771812986583 for this raw cumulant, while 100-decimal exact
        // recurrence over these binary64 moments rounds to the value below.
        // This fixture pins the production contract to the inputs rather than
        // to either floating implementation.
        let log_abs = [
            0.0,
            3.401906354177129,
            5.202067255568281,
            8.594448894699298,
            2.6768393081078585,
            6.071040407753003,
            7.870357096670745,
            11.25488513810204,
            3.185899672908363,
            6.579844920691869,
            8.379473650793713,
            11.763744180619511,
            5.854469159372471,
            9.24057751754387,
            11.039349558929501,
            14.415629263653415,
        ];
        let signs = [
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
            -1.0, 1.0, -1.0, 1.0,
        ];
        let moments = std::array::from_fn(|index| LatentSignedLog {
            log_abs: log_abs[index],
            sign: signs[index],
        });
        let cumulants =
            latent_certified_cumulants(moments, 0b1111, "2597 K6 exact-expansion oracle")
                .expect("the K6 rounded moments have a uniquely rounded cumulant");
        let certified = cumulants[0b1111];
        assert_eq!(certified.value, -1.118477179328992);

        // Grade the pre-cutover implementation against its own structural
        // forward-error entitlement. It is inside that band, but the band spans
        // many rounding cells, so its 1.97e-9 disagreement is explicitly
        // cancellation-unresolved rather than accepted as exact.
        const MULTIDIR_MAX_OPERATIONS: usize = 34 * 10 + 17 * 5 + 32;
        const _: () = assert!(MULTIDIR_MAX_OPERATIONS == 457);
        let accumulated = MULTIDIR_MAX_OPERATIONS as f64 * f64::EPSILON;
        let gamma =
            LatentExactExpansion::next_up(accumulated / (1.0 - accumulated));
        let relative_roundoff =
            LatentExactExpansion::next_up(gamma * certified.absolute_term_mass);
        let gradual_underflow =
            MULTIDIR_MAX_OPERATIONS as f64 * f64::from_bits(1);
        let forward_error_bound =
            LatentExactExpansion::next_up(relative_roundoff + gradual_underflow);
        let multidir = -1.1184771812986583_f64;
        let observed_error =
            LatentExactExpansion::next_up((multidir - certified.value).abs());
        assert!(
            observed_error <= forward_error_bound,
            "MultiDir escaped its derived K=4 composition band: error={observed_error:e}, \
             bound={forward_error_bound:e}"
        );
        let rounding_cell_radius = 0.5
            * (certified.value - LatentExactExpansion::next_down(certified.value))
                .min(LatentExactExpansion::next_up(certified.value) - certified.value);
        assert!(
            forward_error_bound >= rounding_cell_radius,
            "the ill-conditioned MultiDir channel must be typed cancellation-unresolved: \
             bound={forward_error_bound:e}, rounding_cell_radius={rounding_cell_radius:e}"
        );
    }

    #[test]
    fn latent_cumulant_expansion_types_unresolved_moment_underflow_2597() {
        let mut moments = [LatentSignedLog::ZERO; 16];
        moments[0] = LatentSignedLog::ONE;
        moments[0b0001] = LatentSignedLog {
            log_abs: -800.0,
            sign: 1.0,
        };
        let error = latent_certified_cumulants(
            moments,
            0b0001,
            "2597 underflowed rounded moment",
        )
        .expect_err("a moment rounded to zero must not masquerade as an exact derivative");
        assert!(
            matches!(
                error,
                LatentSurvivalError::DerivativeAccuracyUnresolved { .. }
            ),
            "accuracy loss must use the typed refusal, got {error:?}"
        );
    }

    /// gam#2714: a monomial whose exact residual binary64 cannot represent is
    /// not a derivative binary64 cannot represent.
    ///
    /// `two_product` is error-free only while the product stays at or above
    /// `2^-970`; below it the FMA residual can need bits under `2^-1074` and
    /// calling it exact would be a lie. That premise is right. Refusing on it
    /// is not — it throws away the whole derivative over a monomial that cannot
    /// move the answer by an ulp.
    ///
    /// The witness is the smallest possible instance of the recurrence. Two
    /// perfectly ordinary normal moments, `e^-350 ≈ 1e-152` each, against a
    /// leading moment of `1`:
    ///
    /// ```text
    /// κ(0b0011) = m(0b0011) − κ(0b0001)·m(0b0010) = 1 − 1e-304
    /// ```
    ///
    /// `1e-304` is 304 orders below `1` and 252 orders below `1`'s own half-ulp
    /// `2^-53`, so the correctly rounded cumulant is `1.0` and nothing about
    /// the arithmetic is in doubt. The product is nonetheless under `2^-970`,
    /// which is what refused it — the same shape as the tie-to-even refusal
    /// this file already fixed for gam#2538, where the certification refused
    /// precisely the inputs it can decide.
    ///
    /// Measured on the #2714 witness: five of the seven outer seeds of the
    /// veteran latent-frailty fit die with `derivative mask 0b1111 has no
    /// certified binary64 value: an exact-expansion product entered the
    /// unprovable underflow range`.
    #[test]
    fn latent_cumulant_expansion_certifies_a_monomial_binary64_cannot_carry_2714() {
        let tiny = LatentSignedLog {
            log_abs: -350.0,
            sign: 1.0,
        };
        // The two factors of the refused monomial are ordinary normals, and
        // their product is not: that separation is the whole fixture.
        let factor = tiny.log_abs.exp();
        assert!(
            factor >= f64::MIN_POSITIVE && factor.is_finite(),
            "each factor must be an ordinary normal binary64, got {factor:e}"
        );
        assert!(
            factor * factor < f64::MIN_POSITIVE / f64::EPSILON,
            "the fixture is only a witness if the PRODUCT lands under 2^-970, got {:e}",
            factor * factor
        );

        let mut moments = [LatentSignedLog::ZERO; 16];
        moments[0] = LatentSignedLog::ONE;
        moments[0b0001] = tiny;
        moments[0b0010] = tiny;
        moments[0b0011] = LatentSignedLog::ONE;

        let certified = latent_certified_cumulants(moments, 0b0011, "2714 subnormal monomial")
            .expect(
                "a monomial 304 orders below the leading moment cannot decide the rounding of \
                 the cumulant, so it cannot refuse it either",
            );
        assert_eq!(
            certified[0b0011].value, 1.0,
            "κ(0b0011) = 1 − 1e-304 rounds to 1.0 in binary64; the subtracted monomial is 252 \
             orders below 1.0's own half-ulp"
        );
        assert_eq!(
            certified[0b0001].value, factor,
            "the pointed singleton cumulant is its own moment"
        );
    }

    /// The other side of gam#2714, so the repair above cannot become "drop the
    /// small terms": where the mass binary64 genuinely cannot carry REACHES the
    /// answer's rounding cell, the derivative must still be refused.
    ///
    /// The bound is one subnormal ulp, `2^-1074`, per underflowing monomial, so
    /// it competes with a rounding cell only at the very bottom of the normal
    /// range. This fixture puts the cumulant in the binade `[2^-1021, 2^-1020)`,
    /// where the half-ulp is exactly `2^-1074` — the same size as the mass — and
    /// nothing above that binade can be affected by one such monomial. The
    /// fixture asserts both facts rather than trusting them, so a change to the
    /// bound cannot leave the test silently exercising a different regime.
    #[test]
    fn latent_cumulant_expansion_refuses_when_the_unrepresentable_mass_reaches_the_cell_2714() {
        // `e^-707.4 ≈ 5.5e-308`: an ordinary normal, one binade above the
        // smallest, whose rounding cell radius is one subnormal ulp.
        let leading = LatentSignedLog {
            log_abs: -707.4,
            sign: 1.0,
        };
        let magnitude = leading.log_abs.exp();
        assert!(
            magnitude >= f64::MIN_POSITIVE,
            "the leading moment must be a normal binary64, got {magnitude:e}"
        );
        let cell_radius = 0.5 * (LatentExactExpansion::next_up(magnitude) - magnitude);
        assert_eq!(
            cell_radius,
            2.0_f64.powi(-1074),
            "the fixture is only in the contested regime if the rounding cell radius IS the \
             per-monomial bound"
        );

        // A monomial whose two factors are ordinary normals and whose product
        // is not: the same construction as the certifying witness above, at a
        // scale where what it loses is no longer negligible.
        let factor = LatentSignedLog {
            log_abs: -370.0,
            sign: 1.0,
        };
        let factor_magnitude = factor.log_abs.exp();
        assert!(
            factor_magnitude >= f64::MIN_POSITIVE
                && factor_magnitude * factor_magnitude < f64::MIN_POSITIVE / f64::EPSILON,
            "the monomial must underflow the product bar while its factors do not"
        );

        let mut moments = [LatentSignedLog::ZERO; 16];
        moments[0] = LatentSignedLog::ONE;
        moments[0b0001] = factor;
        moments[0b0010] = factor;
        moments[0b0011] = leading;

        let error = latent_certified_cumulants(moments, 0b0011, "2714 cell-scale residual")
            .expect_err(
                "an unrepresentable residual the size of the rounding cell must stay refused",
            );
        assert!(
            matches!(
                error,
                LatentSurvivalError::DerivativeAccuracyUnresolved { .. }
            ),
            "accuracy loss must use the typed refusal, got {error:?}"
        );
    }

    /// An exact midpoint has no unique NEAREST binary64, but it has a unique
    /// CORRECTLY ROUNDED one: ties-to-even, the mode `certified_round`'s own
    /// doc comment names and the mode every binary64 operation downstream of it
    /// uses. This pins the rounded values, not a refusal -- refusing the tie
    /// killed all seven outer seeds of the gam#2538 frailty fit in 0.49 s.
    #[test]
    fn latent_cumulant_expansion_rounds_an_exact_tie_to_even_2597() {
        // `2^-53` is exactly half an ulp at 1.0, so this is the midpoint
        // between 1.0 and `1 + 2^-52`. 1.0 has the even significand.
        let midpoint = LatentExactExpansion::scalar(1.0)
            .add(LatentExactExpansion::scalar(2.0_f64.powi(-53)))
            .expect("the two-term midpoint has exact expansion support");
        assert_eq!(
            midpoint
                .certified_round()
                .expect("an exact tie rounds to even, it is not unroundable"),
            1.0,
        );

        // One ulp up, the midpoint between `1 + 2^-52` (odd significand) and
        // `1 + 2^-51` (even) must round the OTHER way, so the assertion above
        // cannot be satisfied by a rule that simply returns the lower value.
        let odd = f64::from_bits(1.0_f64.to_bits() + 1);
        let upper_midpoint = LatentExactExpansion::scalar(odd)
            .add(LatentExactExpansion::scalar(2.0_f64.powi(-53)))
            .expect("the two-term midpoint has exact expansion support");
        assert_eq!(
            upper_midpoint
                .certified_round()
                .expect("an exact tie rounds to even, it is not unroundable"),
            f64::from_bits(1.0_f64.to_bits() + 2),
        );
    }

    /// The accumulated sum of the components is only a NEIGHBOURHOOD of the
    /// exact value. When it misses the nearest binary64, the exact boundary
    /// comparison inside `certified_round` names the neighbour that is nearer,
    /// and the honest response is to step onto it -- not to refuse the value as
    /// unroundable. Refusing it is what stood between gam#2538 and a fit once
    /// the tie arm was fixed: every seed died on `the exact cumulant lies
    /// outside the candidate binary64 rounding cell`.
    #[test]
    fn latent_cumulant_expansion_walks_onto_the_nearest_binary64_2538() {
        // `1.0 + 2^-53` is an exact midpoint and rounds to even, i.e. back to
        // `1.0`, twice -- so the accumulation loses BOTH small components and
        // lands a full ulp below the exact sum `1 + 2^-52`, which is itself
        // exactly representable. This is the accumulation being wrong, not the
        // value being unroundable.
        let half_ulp_at_one = 2.0_f64.powi(-53);
        let mut witness = LatentExactExpansion::ZERO;
        witness.components[0] = 1.0;
        witness.components[1] = half_ulp_at_one;
        witness.components[2] = half_ulp_at_one;
        witness.len = 3;

        let mut accumulated = 0.0_f64;
        for index in 0..witness.len {
            accumulated += witness.component(index);
        }
        assert_eq!(
            accumulated, 1.0,
            "the fixture is only a witness if the accumulation really misses",
        );

        let exact = 1.0 + 2.0_f64.powi(-52);
        assert!(
            exact - 1.0 == 2.0_f64.powi(-52),
            "the exact sum must be representable, so the correct answer is not itself a tie",
        );
        assert_eq!(
            witness
                .certified_round()
                .expect("a value one ulp from the accumulated sum is roundable, not unroundable"),
            exact,
        );
    }

    /// #2566 diagnostic (zz_measure): a FINE log-σ sweep of the curvature, to
    /// find the discontinuity the coarse ladder can only bracket.
    ///
    /// What is established: the analytic negative Hessian is healthy at
    /// `log σ = 4` (ratio 0.85) and catastrophic at `log σ = 6` (ratio 97, sign
    /// inverted), it does NOT respond to a node-count doubling that moves the
    /// gradient channel by 34%, `max_k` does not flip the bundle mode anywhere in
    /// `0..8`, and the reported mode is `ControlledAsymptotic` across the whole
    /// range so it does not separate the healthy rows from the broken ones.
    ///
    /// Every one of those excludes a mechanism without locating one. A branch
    /// switch is a DISCONTINUITY in σ, and the ladder samples σ at whole-integer
    /// `log σ` — far too coarse to see one. This walks `log σ` in steps of 0.05
    /// from 4.0 to 7.0 and prints the analytic curvature beside a value-only FD
    /// authority at every step. A smooth curve that merely degrades says the loss
    /// is gradual cancellation; a jump between adjacent samples localizes a
    /// routing switch to a 0.05-wide interval and names the σ to go look at.
    ///
    /// The two channels are printed together on purpose: they come from the SAME
    /// jet call, so a step in one and not the other is the sharpest evidence
    /// available about where they part company.
    ///
    /// RESULT (measured): the answer was "gradual cancellation", not a routing
    /// switch — the curve degraded into noise rather than stepping at one σ. That
    /// is what #2610 then removed, by evaluating the derivative term lists in the
    /// `∂_a^j K_0` basis where the cancelling rung sum is performed on integer
    /// coefficients. This sweep is now monotone across the whole range with a
    /// step-to-step relative jump of `0.0487` at every sample, so it has changed
    /// role: it is no longer hunting a discontinuity, it is the regression
    /// witness that the jump stays at the step scale.
    ///
    /// Prints only; never asserts a bound.
    #[test]
    fn zz_measure_2566_curvature_fine_sweep() {
        let quadctx = QuadratureContext::new();
        let row = LatentSurvivalRow::right_censored(0.3, 0.67, 0.01, 0.02);
        let point_at = |log_sigma: f64| LatentSurvivalPrimaryPoint {
            q_entry: -1.2,
            q_exit: -0.4,
            qdot_exit: 0.73,
            q_right: 0.5,
            mu: -0.15,
            sigma: log_sigma.exp(),
        };
        eprintln!(
            "#2566 sweep: log_sigma  value           gradient        neg_hessian     \
             d(neg_hess) vs prev"
        );
        let mut previous: Option<f64> = None;
        for step in 0..=60usize {
            let log_sigma = 4.0 + 0.05 * (step as f64);
            let Ok((value, gradient, negative_hessian)) =
                latent_survival_row_primary_gradient_hessian(
                    &quadctx,
                    &row,
                    point_at(log_sigma),
                    true,
                )
            else {
                eprintln!("#2566 sweep: {log_sigma:>9.2}  REFUSED");
                previous = None;
                continue;
            };
            let curvature = negative_hessian[[
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            ]];
            // Relative jump against the previous sample. On a smooth branch this
            // is O(step); at a routing switch it is O(1) or worse, and that is the
            // whole signal this sweep exists to produce.
            let jump = match previous {
                Some(prev) if prev != 0.0 => {
                    format!("{:>12.4}", (curvature - prev).abs() / prev.abs())
                }
                _ => "           -".to_string(),
            };
            eprintln!(
                "#2566 sweep: {log_sigma:>9.2}  {value:>14.6e}  {:>14.6e}  \
                 {curvature:>14.6e}  {jump}",
                gradient[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA],
            );
            previous = Some(curvature);
        }
    }

    /// #2566 diagnostic (zz_measure): which BRANCH the kernel bundle takes, as a
    /// function of `log σ` and of the requested `max_k`.
    ///
    /// This is the instrument for the question the 513-vs-1025 A/B left open. At
    /// `log σ = 6` the analytic negative Hessian did not move to any printed
    /// figure across a doubling of the quadrature node count, while the same
    /// doubling shifted the FD authority by 34% — so the two channels are not
    /// routing through the same branch there, and no amount of quadrature
    /// resolution reaches the row where the sign inverts.
    ///
    /// RESULT (measured, run 30364896222): the `max_k` hypothesis this was built
    /// to test is **REFUTED**, and so is a second idea it was going to support.
    ///
    /// The mode is CONSTANT in `k` across `0..8` at every `log σ ≥ 3`, so the
    /// second-order path's larger `k` does not flip the branch:
    ///
    /// ```text
    ///   log σ = 2  ->  S S Q Q Q Q Q Q Q     (the only QuadratureFallback rows)
    ///   log σ = 3  ->  A A A A A A A A A
    ///   log σ = 6  ->  A A A A A A A A A
    ///   log σ = 7  ->  A A A A A A A A A
    /// ```
    ///
    /// Two things follow, both negative and both worth keeping:
    ///
    /// * **`mode` cannot be the fidelity signal a consumer refuses on.** It is
    ///   `ControlledAsymptotic` at `log σ = 4` (ratio 0.85, healthy) and equally
    ///   `ControlledAsymptotic` at `log σ = 7` (ratio 402, sign inverted). It does
    ///   not separate the good rows from the catastrophic ones, so surfacing it
    ///   out of `latent_survival_row_primary_gradient_hessian` — which currently
    ///   drops it, while `logk_q_derivatives` keeps it — would buy nothing.
    /// * **`mode` is a fidelity CLASS, not a record of which kernel ran.** The
    ///   513-vs-1025 A/B moved `log σ = 5` by 335×, and this probe reports that
    ///   row as `A`, not `Q`. So a row can be node-count-sensitive while
    ///   reporting a non-quadrature mode; do not read the tags as routing.
    ///
    /// Prints only; never asserts a bound.
    #[test]
    fn zz_measure_2566_kernel_bundle_routing_by_k() {
        let quadctx = QuadratureContext::new();
        let mu = -0.15_f64;
        eprintln!(
            "#2566 routing: mode per (log_sigma, q, max_k); mu={mu}. NOTE: the tag \
             is a fidelity CLASS, not a record of which kernel ran — log_sigma=5 \
             reports A yet moved 335% under a node-count change."
        );
        for log_sigma in [-3.0_f64, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] {
            let sigma = log_sigma.exp();
            // The ladder fixture's three time coordinates; the latent path calls
            // the bundle with `state.q.exp()` and one of these is that `q`.
            for q in [-1.2_f64, -0.4, 0.5] {
                let mut row = String::new();
                for max_k in 0..=8usize {
                    match log_kernel_bundle(&quadctx, q.exp(), mu, sigma, max_k) {
                        Ok(bundle) => {
                            let tag = match format!("{:?}", bundle.mode).as_str() {
                                "ExactClosedForm" => "C",
                                "ExactSpecialFunction" => "S",
                                "ControlledAsymptotic" => "A",
                                "QuadratureFallback" => "Q",
                                other => {
                                    eprintln!("#2566 routing: UNKNOWN mode {other}");
                                    "?"
                                }
                            };
                            row.push_str(tag);
                        }
                        Err(_) => row.push('E'),
                    }
                }
                eprintln!(
                    "#2566 routing: log_sigma={log_sigma:>5.1} q={q:>5.1}  \
                     max_k=0..8 -> {row}"
                );
            }
        }
        eprintln!("#2566 routing: legend C=ExactClosedForm S=ExactSpecialFunction \
                   A=ControlledAsymptotic Q=QuadratureFallback E=refused");
    }

    /// #2566 diagnostic (zz_measure): the WHOLE ladder that
    /// `latent_log_sigma_curvature_tracks_gradient_fd_scale_ladder_2566`
    /// asserts, printed without asserting.
    ///
    /// The gate stops at its first failure, so it reports one row and hides the
    /// shape of the residual across the rest of the range. This probe runs the
    /// identical construction and prints every row, which is what an A/B over
    /// the quadrature node count has to compare.
    ///
    /// Pre-registered reading for that A/B: `e187d267f` raised
    /// `CLOGLOG_GUMBEL_QUAD_MIN_NODES` 97 → 513 and left the gate red at
    /// `log σ = 5` with `disagreement/uncertainty ≈ 3.75`. If the remaining
    /// residual is still quadrature INPUT error, raising the node cap must move
    /// `disagreement` materially; if it is structural in the derivative program,
    /// `disagreement` will be unchanged and node count is not the lever.
    /// Prints only; never asserts a bound.
    #[test]
    fn zz_measure_2566_curvature_fd_ladder_full() {
        let quadctx = QuadratureContext::new();
        let row = LatentSurvivalRow::right_censored(0.3, 0.67, 0.01, 0.02);
        let point_at = |log_sigma: f64| LatentSurvivalPrimaryPoint {
            q_entry: -1.2,
            q_exit: -0.4,
            qdot_exit: 0.73,
            q_right: 0.5,
            mu: -0.15,
            sigma: log_sigma.exp(),
        };
        let analytic_at = |log_sigma: f64| {
            latent_survival_row_primary_gradient_hessian(
                &quadctx,
                &row,
                point_at(log_sigma),
                true,
            )
            .expect("analytic latent-survival row")
        };
        let value_gradient_authority = |log_sigma: f64| {
            let step = f64::EPSILON.cbrt() * (1.0 + log_sigma.abs());
            richardson_central_difference(
                |at| latent_survival_value_fd_authority(&quadctx, &row, point_at(at)),
                log_sigma,
                step,
            )
        };
        let gradient_sample = |log_sigma: f64| {
            let (_, gradient, _) = analytic_at(log_sigma);
            let analytic = gradient[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA];
            let authority = value_gradient_authority(log_sigma);
            (
                analytic,
                (analytic - authority.value).abs() + authority.uncertainty,
            )
        };

        eprintln!(
            "#2566 ladder: log_sigma  analytic        fd_authority    disagreement    \
             uncertainty     ratio"
        );
        for log_sigma in [-3.0_f64, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] {
            let step = f64::EPSILON.cbrt() * (1.0 + log_sigma.abs());
            let (_, _, negative_hessian) = analytic_at(log_sigma);
            let analytic = negative_hessian[[
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            ]];

            let (gradient_coarse_plus, error_coarse_plus) = gradient_sample(log_sigma + step);
            let (gradient_coarse_minus, error_coarse_minus) = gradient_sample(log_sigma - step);
            let half_step = 0.5 * step;
            let (gradient_fine_plus, error_fine_plus) = gradient_sample(log_sigma + half_step);
            let (gradient_fine_minus, error_fine_minus) = gradient_sample(log_sigma - half_step);

            let coarse = -(gradient_coarse_plus - gradient_coarse_minus) / (2.0 * step);
            let fine = -(gradient_fine_plus - gradient_fine_minus) / (2.0 * half_step);
            let authority = (4.0 * fine - coarse) / 3.0;

            let coarse_input_uncertainty =
                (error_coarse_plus + error_coarse_minus) / (2.0 * step);
            let fine_input_uncertainty =
                (error_fine_plus + error_fine_minus) / (2.0 * half_step);
            let gamma = floating_point_gamma(3);
            let coarse_roundoff = gamma
                * (gradient_coarse_plus.abs() + gradient_coarse_minus.abs())
                / (2.0 * step);
            let fine_roundoff = gamma
                * (gradient_fine_plus.abs() + gradient_fine_minus.abs())
                / (2.0 * half_step);
            let combine_roundoff = gamma * (4.0 * fine.abs() + coarse.abs()) / 3.0;
            let uncertainty = (fine - coarse).abs() / 3.0
                + (4.0 * (fine_input_uncertainty + fine_roundoff)
                    + coarse_input_uncertainty
                    + coarse_roundoff)
                    / 3.0
                + combine_roundoff;
            let disagreement = (analytic - authority).abs();

            eprintln!(
                "#2566 ladder: {log_sigma:>9.1}  {analytic:>14.6e}  {authority:>14.6e}  \
                 {disagreement:>14.6e}  {uncertainty:>14.6e}  {:>8.3}",
                disagreement / uncertainty
            );
        }
    }

    /// The primary-jet producer must REFUSE a non-finite operating coordinate
    /// rather than carry it into the channels it returns.
    ///
    /// #2541 asked which producer level loses finiteness in the log-sigma
    /// gradient. Part of that answer is negative and worth pinning: the boundary
    /// this routine presents to its callers is sound, so a non-finite channel
    /// downstream is GENERATED inside the row expression rather than inherited
    /// from a diverged scale or location handed in by the caller. Without this
    /// gate that distinction is re-derived by hand every time a row-named
    /// refusal appears.
    ///
    /// `sigma == 0.0` is the one accepted degenerate input: it is the fixed
    /// zero-frailty case `LatentSurvivalPrimaryPoint::log_sigma_factor`
    /// documents, and it must return finite channels rather than refuse.
    #[test]
    fn latent_primary_jet_refuses_non_finite_operating_coordinates_2541() {
        let quadctx = QuadratureContext::new();
        let rows: [(&str, LatentSurvivalRow); 3] = [
            (
                "right",
                LatentSurvivalRow::right_censored(0.3, 0.67, 0.01, 0.02),
            ),
            (
                "exact",
                LatentSurvivalRow::exact_event(0.3, 0.67, 0.01, 0.02, 0.73, 0.08),
            ),
            (
                "interval",
                LatentSurvivalRow::interval_censored(0.3, 0.67, 1.65, 0.01, 0.02, 0.05),
            ),
        ];
        let healthy = LatentSurvivalPrimaryPoint {
            q_entry: -1.2,
            q_exit: -0.4,
            qdot_exit: 0.73,
            q_right: 0.5,
            mu: -0.15,
            sigma: 0.3_f64.exp(),
        };
        let refused: [(&str, LatentSurvivalPrimaryPoint); 8] = [
            (
                "sigma=NaN",
                LatentSurvivalPrimaryPoint { sigma: f64::NAN, ..healthy },
            ),
            (
                "sigma=+inf",
                LatentSurvivalPrimaryPoint { sigma: f64::INFINITY, ..healthy },
            ),
            (
                "sigma=-1",
                LatentSurvivalPrimaryPoint { sigma: -1.0, ..healthy },
            ),
            (
                "mu=NaN",
                LatentSurvivalPrimaryPoint { mu: f64::NAN, ..healthy },
            ),
            (
                "mu=+inf",
                LatentSurvivalPrimaryPoint { mu: f64::INFINITY, ..healthy },
            ),
            (
                "q_exit=NaN",
                LatentSurvivalPrimaryPoint { q_exit: f64::NAN, ..healthy },
            ),
            (
                "q_exit=+inf",
                LatentSurvivalPrimaryPoint { q_exit: f64::INFINITY, ..healthy },
            ),
            (
                "q_exit=-inf",
                LatentSurvivalPrimaryPoint { q_exit: f64::NEG_INFINITY, ..healthy },
            ),
        ];
        for (event, row) in &rows {
            for (name, point) in refused {
                let outcome =
                    latent_survival_row_primary_gradient_hessian(&quadctx, row, point, true);
                match outcome {
                    Err(_) => {}
                    Ok((value, gradient, hessian)) => panic!(
                        "event={event}: the primary jet accepted {name} instead of refusing it; \
                         value={value:?}, log-sigma gradient={:?}, log-sigma curvature={:?}",
                        gradient[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA],
                        hessian[[
                            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
                            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                        ]],
                    ),
                }
            }

            // The documented degenerate case: fixed zero frailty is admissible
            // and must produce finite channels, not a refusal.
            let zero_scale = LatentSurvivalPrimaryPoint { sigma: 0.0, ..healthy };
            let (value, gradient, hessian) =
                latent_survival_row_primary_gradient_hessian(&quadctx, row, zero_scale, true)
                    .unwrap_or_else(|error| {
                        panic!("event={event}: fixed zero frailty must be admissible: {error}")
                    });
            assert!(
                value.is_finite()
                    && gradient.iter().all(|entry| entry.is_finite())
                    && hessian.iter().all(|entry| entry.is_finite()),
                "event={event}: fixed zero frailty produced a non-finite channel: \
                 value={value:?}, gradient={gradient:?}"
            );
        }
    }

    #[test]
    fn latent_interval_log_difference_stays_finite_at_small_absolute_mass_2565() {
        let quadctx = QuadratureContext::new();
        let row =
            LatentSurvivalRow::interval_censored(0.3, 0.67, 1.65, 0.01, 0.02, 0.05);
        let cases = [
            ("small Hessian tail", -6.0_f64, 20.0_f64, -1.0_f64),
            ("small gradient tail", 12.0, 5.0, -1.0),
            ("deep location tail", 20.0, 20.0, 0.0),
            ("deep mass tail", 40.0, -0.15, 0.0),
        ];
        let point_at = |log_mass: f64, mu: f64, log_sigma: f64| {
            LatentSurvivalPrimaryPoint {
                q_entry: log_mass,
                q_exit: log_mass + 0.8,
                qdot_exit: 0.73,
                q_right: log_mass + 1.6,
                mu,
                sigma: log_sigma.exp(),
            }
        };

        for (label, log_mass, mu, log_sigma) in cases {
            let point = point_at(log_mass, mu, log_sigma);
            let (value, gradient, hessian) =
                latent_survival_row_primary_gradient_hessian(&quadctx, &row, point, true)
                    .unwrap_or_else(|error| panic!("{label} must remain representable: {error}"));
            assert!(
                value.is_finite()
                    && gradient.iter().all(|entry| entry.is_finite())
                    && hessian.iter().all(|entry| entry.is_finite()),
                "{label} returned Ok with a non-finite channel: \
                 value={value:?}, gradient={gradient:?}, hessian={hessian:?}"
            );
        }

        // The same log-difference primitive serves the contracted third and
        // fourth channels; exercise them at the first old Hessian-overflow point.
        let point = point_at(-6.0, 20.0, -1.0);
        let mut direction = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        direction[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA] = 1.0;
        let third =
            latent_survival_row_primary_third_contracted(&quadctx, &row, point, &direction, true)
                .expect("small absolute interval mass must have finite contracted third order");
        let fourth = latent_survival_row_primary_fourth_contracted(
            &quadctx,
            &row,
            point,
            &direction,
            &direction,
            true,
        )
        .expect("small absolute interval mass must have finite contracted fourth order");
        assert!(third.iter().all(|entry| entry.is_finite()));
        assert!(fourth.iter().all(|entry| entry.is_finite()));
    }

    #[test]
    fn latent_interval_log_difference_matches_closed_form_curvature_2565() {
        // With A = exp(a), B = 1/2, and a = 0,
        //
        //   log(A - B) = -log(2),
        //   d/da log(A - B) = 2,
        //   d²/da² log(A - B) = -2.
        //
        // This is an algebraic oracle independent of the multi-direction
        // reference, which deliberately shares the stable production primitive.
        let log_left = Order2::<1>::variable(0.0, 0);
        let log_right = Order2::<1>::constant(-std::f64::consts::LN_2);
        let out = latent_survival_positive_log_difference_jet(
            &log_left,
            0.0,
            &log_right,
            0.0,
            2,
            "interval closed-form discriminator",
            latent_order2_all_finite,
        )
        .expect("the closed-form interval mass and derivatives are representable");
        assert_eq!(out.value(), -std::f64::consts::LN_2);
        assert_eq!(out.g()[0], 2.0);
        assert_eq!(out.h()[0][0], -2.0);
    }

    #[test]
    fn latent_interval_log_difference_refuses_domain_and_derivative_overflow_2565() {
        let left = Order2::<1>::constant(0.0);
        let equal = Order2::<1>::constant(0.0);
        let domain_error = latent_survival_positive_log_difference_jet(
            &left,
            0.0,
            &equal,
            0.0,
            2,
            "interval boundary discriminator",
            latent_order2_all_finite,
        )
        .expect_err("equal boundaries have zero interval mass");
        assert!(matches!(
            &domain_error,
            LatentSurvivalError::NumericalFailure { .. }
        ));
        assert!(
            domain_error.to_string().contains("positive survival-mass difference"),
            "domain refusal must identify the estimand: {domain_error}"
        );

        // The value log(1-exp(delta)) is finite here, as is its first
        // derivative, but the true second derivative exceeds f64. The producer
        // must refuse that derivative domain instead of returning Ok with inf.
        let near_boundary = Order2::<1>::variable(-f64::MIN_POSITIVE, 0);
        let overflow_error = latent_survival_positive_log_difference_jet(
            &left,
            0.0,
            &near_boundary,
            0.0,
            2,
            "interval derivative discriminator",
            latent_order2_all_finite,
        )
        .expect_err("unrepresentable log-gap derivatives require typed refusal");
        assert!(matches!(
            &overflow_error,
            LatentSurvivalError::NumericalFailure { .. }
        ));
        assert!(
            overflow_error.to_string().contains("derivative order 2"),
            "refusal must name the first unrepresentable derivative: {overflow_error}"
        );
    }

    /// The #2714 witness's own family shape: a frailty multiplying the WHOLE
    /// hazard (`HazardLoading::Full`, so every unloaded component is zero), a
    /// LEARNED scale, and both event types present.
    ///
    /// This is deliberately not `learnable_sigma_test_family`, which is
    /// `LoadedVsUnloaded` with non-zero unloaded masses. The two shapes reach
    /// different kernel term lists — `Full` leaves the exact-event numerator as
    /// the single `q·q̇·K₁` rung with no `K₀` companion — and the gates below are
    /// about which basis that rung is evaluated in.
    fn full_loading_learned_sigma_family() -> LatentSurvivalFamily {
        let n = 4;
        LatentSurvivalFamily {
            event_target: array![1u8, 0u8, 1u8, 1u8],
            weights: array![1.0, 0.8, 1.0, 1.3],
            latent_sd_fixed: None,
            hazard_loading: HazardLoading::Full,
            unloaded_mass_entry: Array1::zeros(n),
            unloaded_mass_exit: Array1::zeros(n),
            unloaded_hazard_exit: Array1::zeros(n),
            // `q_entry = β₀` and `q_exit = β₀ + dᵢ·β₁` with `dᵢ > 0`, so
            // `q_exit ≥ q_entry` holds at EVERY β the sweeps below visit as long
            // as `β₁ > 0` — which they hold fixed. `build_latent_survival_row`
            // refuses a non-monotone cumulative mass, so a fixture that only
            // happened to be monotone at the one β I picked would be measuring
            // that refusal instead of the likelihood.
            x_time_entry: array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            x_time_exit: array![[1.0, 0.35], [1.0, 0.90], [1.0, 1.70], [1.0, 2.60]],
            // `q̇_exit = β₁ > 0`, which the exact-event rows also require — the
            // same choice buys both.
            x_time_derivative_exit: array![
                [0.0, 1.00],
                [0.0, 1.00],
                [0.0, 1.00],
                [0.0, 1.00]
            ],
            x_time_right: array![[1.0, 0.35], [1.0, 0.90], [1.0, 1.70], [1.0, 2.60]],
            time_offset_right: Array1::zeros(n),
            unloaded_mass_right: Array1::zeros(n),
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(array![
                [1.0, -0.40],
                [1.0, 0.15],
                [1.0, 0.60],
                [1.0, -0.90]
            ])),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        }
    }

    /// The three log-likelihoods a latent-survival family exposes, at one β.
    ///
    /// `accept_test` is [`CustomFamily::log_likelihood_only`] — what the
    /// joint-Newton trust region evaluates at the TRIAL point. `gradient_source`
    /// is `exact_newton_joint_gradient_evaluation`'s scalar — what becomes
    /// `current_log_likelihood`, hence `old_objective`, hence the OTHER end of
    /// `actual_reduction = old_objective − trial_objective`. `working_sets` is
    /// `evaluate`'s, which the same solver reads on the non-joint path.
    fn latent_survival_three_log_likelihoods(
        family: &LatentSurvivalFamily,
        specs: &[ParameterBlockSpec],
        states: &[ParameterBlockState],
    ) -> (f64, f64, f64) {
        let accept_test = family
            .log_likelihood_only(states)
            .expect("log_likelihood_only");
        let gradient_source = family
            .exact_newton_joint_gradient_evaluation(states, specs)
            .expect("joint gradient evaluation")
            .expect("latent survival exposes a joint gradient evaluation")
            .log_likelihood;
        let working_sets = family.evaluate(states).expect("evaluate").log_likelihood;
        (accept_test, gradient_source, working_sets)
    }

    /// #2714: a family must have ONE log-likelihood, not one per consumer.
    ///
    /// The joint-Newton trust ratio is
    ///
    /// ```text
    /// ρ = (old_objective − trial_objective) / (rhs·δ − ½δᵀHδ)
    /// ```
    ///
    /// and the two halves of that numerator are produced by DIFFERENT family
    /// hooks. `old_objective` is built from `current_log_likelihood`, which
    /// `load_joint_gradient_evaluation` reads off
    /// `exact_newton_joint_gradient_evaluation`; `trial_objective` is built from
    /// `log_likelihood_only`, which the line search calls at `β + δ`. Writing
    /// `b(β) = ℓ_accept(β) − ℓ_gradient(β)` for the gap between them,
    ///
    /// ```text
    /// actual_reduction = −[ℓ_gradient(β+δ) − ℓ_gradient(β)] − b(β+δ) + (penalty terms),
    /// ```
    ///
    /// so ANY non-zero `b` enters the numerator as a term that does not vanish
    /// with `‖δ‖`. The trust region then shrinks the step without shrinking the
    /// disagreement, `ρ` stops approaching 1, and every attempt is refused on the
    /// objective while the model and the likelihood accept it — the
    /// `rejects[model,likelihood,objective,feasibility] = [0,0,2,0]` signature at
    /// radius `1e-12` this issue is filed on.
    ///
    /// The two hooks reach the kernel by different routes, which is why this is
    /// not automatic: `log_likelihood_only` sums
    /// `LatentSurvivalRowJet::evaluate(..).log_lik`, assembled by
    /// [`LogKernelSumJet`] over the RUNG basis (`kernel_ratio_jet` on
    /// `log_values`), while the gradient hook sums
    /// `latent_survival_row_primary_gradient_hessian`'s value channel, which
    /// prefers the `∂_a^j K₀` basis (#2610). Those are an exact change of basis
    /// in real arithmetic and two different quadratures in f64.
    ///
    /// The bar is `1e-13` relative, which is a statement about ONE function
    /// evaluated twice, not about two approximations of one integral: `b` is
    /// what the trust region must not see, and the objective's own resolution
    /// witness works at `|F|·ε`.
    #[test]
    fn latent_survival_family_has_one_log_likelihood_2714() {
        let family = full_loading_learned_sigma_family();
        let specs = latent_test_specs(
            family.event_target.len(),
            &[("time", 2), ("mean", 2), ("log_sigma", 1)],
        );
        let slices = family.joint_slices();
        let sigma_idx = slices
            .log_sigma
            .as_ref()
            .expect("learned scale exposes a log_sigma block")
            .start;

        // A β sweep rather than one point: the disagreement this gate exists for
        // is state-dependent by construction (the `∂_a` tower is admitted per
        // bundle by its own measured cancellation), so a single β can sit on
        // either side of that admission and say nothing about the other.
        let base = array![-0.60, 0.85, -0.25, 0.40, 0.0_f64];
        let mut worst_accept_vs_gradient = 0.0_f64;
        let mut worst_accept_vs_working_sets = 0.0_f64;
        let mut worst_at = (0.0_f64, 0.0_f64);
        let mut evaluated = 0usize;
        for log_sigma in [-2.5_f64, -1.5, -0.7, 0.0, 0.7, 1.5, 2.5] {
            for time_shift in [-0.8_f64, -0.3, 0.0, 0.35, 0.9] {
                let mut beta = base.clone();
                beta[0] += time_shift;
                beta[sigma_idx] = log_sigma;
                let states = latent_survival_states_from_joint_beta(&family, &beta);
                let (accept_test, gradient_source, working_sets) =
                    latent_survival_three_log_likelihoods(&family, &specs, &states);
                let scale = accept_test
                    .abs()
                    .max(gradient_source.abs())
                    .max(working_sets.abs())
                    .max(1.0);
                let gap_gradient = (accept_test - gradient_source).abs() / scale;
                let gap_working = (accept_test - working_sets).abs() / scale;
                if gap_gradient > worst_accept_vs_gradient {
                    worst_accept_vs_gradient = gap_gradient;
                    worst_at = (log_sigma, time_shift);
                }
                worst_accept_vs_working_sets = worst_accept_vs_working_sets.max(gap_working);
                evaluated += 1;
            }
        }
        eprintln!(
            "[2714] one-log-likelihood: {evaluated} states; worst |accept − gradient|/scale = \
             {worst_accept_vs_gradient:.6e} at (log_sigma, time_shift) = {worst_at:?}; \
             worst |accept − evaluate|/scale = {worst_accept_vs_working_sets:.6e}"
        );
        assert!(
            worst_accept_vs_gradient <= 1e-13,
            "#2714: the log-likelihood the accept test evaluates at the trial β and the one the \
             base-point objective is built from disagree by {worst_accept_vs_gradient:.6e} \
             relative at (log_sigma, time_shift) = {worst_at:?}. That gap enters \
             `actual_reduction` as a term independent of ‖δ‖, so no trust radius can make ρ \
             approach 1 and every step is refused on the objective."
        );
        assert!(
            worst_accept_vs_working_sets <= 1e-13,
            "#2714: `log_likelihood_only` and `evaluate` disagree by \
             {worst_accept_vs_working_sets:.6e} relative"
        );
    }

    /// #2714, the structural half: the value-only lift and the order-two lift
    /// return the SAME BITS, not merely the same number to a tolerance.
    ///
    /// `log_likelihood_only` is cheap because it evaluates the row expression
    /// through `LatentValueBackend`, which skips the `K + K(K+1)/2` normalised
    /// moments the gradient/Hessian lift needs. That is only sound if the value
    /// channel it publishes is the one the gradient lift would have published —
    /// which requires the kernel bundle, the `∂_a` tower and the basis choice to
    /// be identical, and every one of those is derived from `K` and `max_k`.
    ///
    /// Asserting bit equality rather than a tolerance is the point. A tolerance
    /// would pass on two lifts that build DIFFERENT bundles and happen to agree
    /// to `1e-15`, which is exactly the state this issue is about: the trust
    /// ratio divides by quantities of that size near its fixed point, so
    /// "agrees to round-off" is not a substitute for "is the same object".
    #[test]
    fn latent_survival_value_lift_is_bit_identical_to_the_gradient_lift_2714() {
        let quadctx = QuadratureContext::new();
        let mut compared = 0usize;
        for (event_code, qdot_exit) in [(1u8, 0.85_f64), (0u8, 1.0)] {
            let row = build_latent_survival_row(
                0,
                HazardLoading::Full,
                latent_survival_event_type_for(event_code),
                -0.60,
                0.10,
                qdot_exit,
                0.10,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            .expect("witness-shaped latent survival row");
            for log_sigma in [-2.5_f64, -1.0, 0.0, 1.0, 2.5] {
                for mu in [-1.3_f64, -0.2, 0.0, 0.4, 1.1] {
                    for include_log_sigma in [true, false] {
                        let point = LatentSurvivalPrimaryPoint {
                            q_entry: -0.60,
                            q_exit: 0.10,
                            qdot_exit,
                            q_right: 0.10,
                            mu,
                            sigma: log_sigma.exp(),
                        };
                        let value = latent_survival_row_primary_value(
                            &quadctx,
                            &row,
                            point,
                            include_log_sigma,
                        )
                        .expect("value-only lift");
                        let (gradient_value, _, _) = latent_survival_row_primary_gradient_hessian(
                            &quadctx,
                            &row,
                            point,
                            include_log_sigma,
                        )
                        .expect("order-two lift");
                        assert_eq!(
                            value.to_bits(),
                            gradient_value.to_bits(),
                            "#2714: the value-only lift returns {value:.17e} and the \
                             order-two lift {gradient_value:.17e} at (event={event_code}, \
                             log_sigma={log_sigma}, mu={mu}, \
                             include_log_sigma={include_log_sigma}). The accept test uses \
                             the first and the trust ratio's other end uses the second, so \
                             any gap is a constant of the backtracking ladder."
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert_eq!(
            compared, 100,
            "#2714: the bit-identity sweep must actually run its 100 states"
        );
    }

    /// #2714: the joint gradient must differentiate the objective the accept
    /// test evaluates — in EVERY coordinate, not the one this family already
    /// pins.
    ///
    /// `latent_survival_learnable_sigma_block_matches_family_fd` central-
    /// differences `log_likelihood_only` against the joint gradient at ONE
    /// coordinate (log-σ), at ONE β, with a `2e-3` relative bar. That is a gate
    /// on the log-σ channel's chain rule, and it is silent about the time and
    /// mean channels, which carry the exact-event `K₁` rung the two evaluators
    /// disagree on.
    ///
    /// The step is `ε^{1/3}(1+|β_j|)` — the central-difference optimum for a
    /// value known to `ε` — and the bar `64·ε^{2/3}(1+|analytic|)` is that
    /// optimum's own error with slack for the `f'''` factor. A gradient that
    /// belongs to a DIFFERENT function than the one differenced fails this by
    /// orders, not by the slack.
    #[test]
    fn latent_survival_joint_gradient_differentiates_the_accept_objective_2714() {
        let family = full_loading_learned_sigma_family();
        let specs = latent_test_specs(
            family.event_target.len(),
            &[("time", 2), ("mean", 2), ("log_sigma", 1)],
        );
        let slices = family.joint_slices();
        let sigma_idx = slices
            .log_sigma
            .as_ref()
            .expect("learned scale exposes a log_sigma block")
            .start;
        let base = array![-0.60, 0.85, -0.25, 0.40, 0.0_f64];

        let mut worst_ratio = 0.0_f64;
        let mut worst_report = String::new();
        for log_sigma in [-2.0_f64, -0.7, 0.0, 0.9, 2.0] {
            let mut beta = base.clone();
            beta[sigma_idx] = log_sigma;
            let states = latent_survival_states_from_joint_beta(&family, &beta);
            let analytic = family
                .exact_newton_joint_gradient_evaluation(&states, &specs)
                .expect("joint gradient evaluation")
                .expect("latent survival exposes a joint gradient evaluation")
                .gradient;
            for j in 0..beta.len() {
                let h = f64::EPSILON.cbrt() * (1.0 + beta[j].abs());
                let mut beta_plus = beta.clone();
                beta_plus[j] += h;
                let mut beta_minus = beta.clone();
                beta_minus[j] -= h;
                let ll_plus = family
                    .log_likelihood_only(&latent_survival_states_from_joint_beta(
                        &family, &beta_plus,
                    ))
                    .expect("accept-test log-likelihood at +h");
                let ll_minus = family
                    .log_likelihood_only(&latent_survival_states_from_joint_beta(
                        &family,
                        &beta_minus,
                    ))
                    .expect("accept-test log-likelihood at −h");
                let fd = (ll_plus - ll_minus) / (2.0 * h);
                let bound =
                    64.0 * f64::EPSILON.powf(2.0 / 3.0) * (1.0 + analytic[j].abs()).max(1.0);
                let ratio = (analytic[j] - fd).abs() / bound;
                if ratio > worst_ratio {
                    worst_ratio = ratio;
                    worst_report = format!(
                        "coordinate {j} at log_sigma={log_sigma}: analytic={:.12e}, \
                         central difference of log_likelihood_only={fd:.12e}, \
                         |gap|={:.6e}, bound={bound:.6e}",
                        analytic[j],
                        (analytic[j] - fd).abs()
                    );
                }
            }
        }
        eprintln!("[2714] gradient-vs-accept-objective: worst gap/bound = {worst_ratio:.6e}");
        assert!(
            worst_ratio <= 1.0,
            "#2714: the joint gradient is not the derivative of the objective the accept test \
             evaluates — {worst_report} (gap/bound = {worst_ratio:.6e}). rhs·δ and \
             actual_reduction then measure two different functions and ρ cannot approach 1."
        );
    }
