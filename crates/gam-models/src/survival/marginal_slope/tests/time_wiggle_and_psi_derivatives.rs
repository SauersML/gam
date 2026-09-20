//! Time-wiggle and flex ψ-derivative tests for the survival marginal-slope family,
//! split out of `tests.rs` along that seam to keep it under the 10k-line gate.

use super::*;

#[test]
fn timewiggle_scorewarp_family_supports_second_order_exact_outer_path() {
    let score_runtime = test_deviation_runtime();
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 5))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 5))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 5))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::ones(1)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 0)))).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let specs = vec![
        dummy_blockspec(5),
        dummy_blockspec(0),
        dummy_blockspec(score_runtime.basis_dim()),
    ];
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
}

#[test]
fn timewiggle_time_jacobian_nonzero_at_zero_beta_linearization() {
    // Regression: when `timewiggle(...)` disables the base time basis the
    // time block's coefficient slots are zero placeholder columns. The
    // identifiability canonicaliser linearises every block at β=0 by
    // calling `effective_jacobian_at` with `beta = &[]`. Previously the
    // timewiggle callback inferred block existence from the (empty)
    // coefficient slice, drove `beta_tw` empty, and returned an all-zero
    // time Jacobian — so the compiler reported "block 0 fully aliased:
    // structural residual Gram has no positive eigenspace". The true
    // derivative ∂q/∂β_tw[j] = B_j(h) at β=0 is the wiggle basis value and
    // is nonzero.
    let (knots, degree, p_tw) = standard_test_time_wiggle();
    assert!(p_tw > 0);
    let n = 3usize;
    // Base time basis disabled: p_base = 0, every column is a wiggle slot,
    // densified as zeros (the placeholder tail the workflow appends).
    let zeros = Arc::new(Array2::<f64>::zeros((n, p_tw)));
    // Pilot coordinates in the interior of the knot span [0, 1].
    let offset_entry = Arc::new(array![0.2, 0.4, 0.6]);
    let offset_exit = Arc::new(array![0.5, 0.7, 0.9]);
    let offset_deriv = Arc::new(array![1.0, 1.0, 1.0]);
    let jac_cb = SmsTimewiggleTimeJacobian::new(
        Arc::clone(&zeros),
        Arc::clone(&zeros),
        Arc::clone(&zeros),
        Arc::new(Array2::<f64>::zeros((n, 0))), // p_m = 0
        Arc::clone(&offset_entry),
        Arc::clone(&offset_exit),
        Arc::clone(&offset_deriv),
        Arc::new(Array1::<f64>::zeros(n)), // marginal_offset = 0
        knots.clone(),
        degree,
        p_tw,
        0,
    );
    let empty: Vec<f64> = Vec::new();
    let state = crate::custom_family::FamilyLinearizationState {
        beta: &empty,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = crate::custom_family::BlockEffectiveJacobian::effective_jacobian_at(&jac_cb, &state)
        .expect("timewiggle time jacobian at beta=0");
    assert_eq!(jac.dim(), (3 * n, p_tw));

    // q0 rows (0..n) must equal the wiggle basis at the entry pilot
    // coordinate (c_i = 1 at β_g = 0), not the broken all-zero placeholder.
    let basis_entry =
        monotone_wiggle_basis_with_derivative_order(offset_entry.view(), &knots, degree, 0)
            .expect("entry basis");
    for i in 0..n {
        for j in 0..p_tw {
            assert_close(
                jac[[i, j]],
                basis_entry[[i, j]],
                1e-12,
                &format!("q0 wiggle col ({i},{j})"),
            );
        }
    }
    // The time block must carry a positive structural eigenspace: its Gram
    // Jᵀ J is not the zero matrix.
    let gram = jac.t().dot(&jac);
    let trace: f64 = (0..p_tw).map(|j| gram[[j, j]]).sum();
    assert!(
        trace > 1e-6,
        "time block Gram is structurally zero (fully-aliased artifact): trace={trace}"
    );
}

#[test]
fn survival_marginal_slope_advertises_outer_hvp_at_large_psi_dim() {
    // `dummy_penalized_blockspec` materializes single-row designs, so the
    // family row count must be 1 to satisfy the HVP availability row guard
    // (`parameter_block_specs_match_rows`, added in 28a1c035f). The
    // "large psi dim" under test is the 32-column block below, not `n`.
    let n = 1usize;
    let family = make_block_psi_test_family(n);
    let specs = vec![
        dummy_penalized_blockspec(0, 0),
        dummy_penalized_blockspec(1, 31),
        dummy_penalized_blockspec(1, 1),
    ];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        ..BlockwiseFitOptions::default()
    };

    let (gradient, hessian) = custom_family_outer_derivatives(&family, &specs, &options);

    assert!(family.inner_coefficient_hessian_hvp_available(&specs));
    assert!(family.outer_hyper_hessian_hvp_available(&specs));
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &options),
        ExactOuterDerivativeOrder::Second
    );
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Either);
}

#[test]
fn exact_outer_row_work_gate_keeps_large_timewiggle_link_models_under_linear_flex_budget() {
    let link_runtime = test_deviation_runtime();
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 80,
        entry_at_origin: Arc::new(Array1::from_elem(80, false)),
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 12))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 12))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 12))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::ones(1)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 20))),
        slope_layout: (DesignMatrix::from(Array2::zeros((1, 20)))).into(),
        score_warp: None,
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let specs = vec![
        dummy_penalized_blockspec(12, 2),
        dummy_penalized_blockspec(20, 2),
        dummy_penalized_blockspec(link_runtime.basis_dim(), 2),
    ];
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
}

#[test]
fn timewiggle_scorewarp_beta_hessian_directional_derivative_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_flat = Array1::zeros(slices.total);
    d_beta_flat[slices.time.start] = 0.07;
    d_beta_flat[slices.time.start + 1] = -0.03;
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_flat[h_range.start] = 0.02;
    }

    let directional = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &d_beta_flat)
        .expect("timewiggle flex beta-Hessian directional derivative should evaluate")
        .expect("directional derivative should exist");
    assert_eq!(directional.nrows(), slices.total);
    assert_eq!(directional.ncols(), slices.total);
    assert!(directional.iter().all(|value| value.is_finite()));
}

#[test]
fn timewiggle_scorewarp_beta_hessian_second_directional_derivative_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_u = Array1::zeros(slices.total);
    let mut d_beta_v = Array1::zeros(slices.total);
    d_beta_u[slices.time.start] = 0.07;
    d_beta_u[slices.time.start + 1] = -0.03;
    d_beta_u[slices.marginal.start] = 0.05;
    d_beta_u[slices.slope.start] = -0.04;
    d_beta_v[slices.time.start + 2] = 0.06;
    d_beta_v[slices.marginal.start + 1] = -0.02;
    d_beta_v[slices.slope.start] = 0.03;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_u[h_range.start] = 0.02;
        d_beta_v[h_range.start] = -0.01;
    }

    let second = family
        .exact_newton_joint_hessiansecond_directional_derivative(
            &block_states,
            &d_beta_u,
            &d_beta_v,
        )
        .expect("timewiggle flex beta-Hessian second directional derivative should evaluate")
        .expect("second directional derivative should exist");
    assert_eq!(second.nrows(), slices.total);
    assert_eq!(second.ncols(), slices.total);
    assert!(second.iter().all(|value| value.is_finite()));
}

/// The single-row time-wiggle fixture of the #2893 gates, with an optional score warp.
fn timewiggle_marginal_slope_family(score_warp: Option<DeviationRuntime>) -> SurvivalMarginalSlopeFamily {
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.4, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.7, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(array![[0.7, -0.2]]),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    }
}

/// Block states of `timewiggle_marginal_slope_family` at a flat β: time (5), marginal (2),
/// slope (1), then the score warp and the influence absorber when present, with every `η` rebuilt
/// from β.
fn timewiggle_marginal_slope_states(
    family: &SurvivalMarginalSlopeFamily,
    beta: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    let marginal_beta = beta.slice(s![5..7]).to_owned();
    let marginal_design = family.marginal_design.to_dense().to_owned();
    let mut states = vec![
        ParameterBlockState {
            beta: beta.slice(s![..5]).to_owned(),
            eta: array![0.0],
        },
        ParameterBlockState {
            eta: marginal_design.dot(&marginal_beta),
            beta: marginal_beta,
        },
        ParameterBlockState {
            beta: beta.slice(s![7..8]).to_owned(),
            eta: array![beta[7]],
        },
    ];
    let score_width = family
        .score_warp
        .as_ref()
        .map_or(0, |runtime| runtime.basis_dim());
    if family.score_warp.is_some() {
        states.push(ParameterBlockState {
            beta: beta.slice(s![8..8 + score_width]).to_owned(),
            eta: Array1::zeros(1),
        });
    }
    if family.influence_absorber.is_some() {
        states.push(ParameterBlockState {
            beta: beta.slice(s![8 + score_width..]).to_owned(),
            eta: Array1::zeros(1),
        });
    }
    states
}

/// A flat β for `timewiggle_marginal_slope_family`, with small alternating score-warp
/// coefficients when the warp is present and small influence coefficients when an absorber is.
fn timewiggle_marginal_slope_beta(family: &SurvivalMarginalSlopeFamily) -> Array1<f64> {
    let base = [0.0, 0.08, -0.03, 0.02, -0.01, 0.35, -0.1, 0.2];
    let score_width = family
        .score_warp
        .as_ref()
        .map_or(0, |runtime| runtime.basis_dim());
    let influence_width = family
        .influence_absorber
        .as_ref()
        .map_or(0, |z_tilde| z_tilde.ncols());
    Array1::from_shape_fn(base.len() + score_width + influence_width, |i| {
        if i < base.len() {
            base[i]
        } else if i >= base.len() + score_width {
            0.05 * (i + 1 - base.len() - score_width) as f64
        } else if i % 2 == 0 {
            0.02
        } else {
            -0.02
        }
    })
}

/// gam#2893: `D²_β H[u, v]` for a time wiggle, alone and with a score warp, against a
/// Ridders-certified central difference of the family's own `D_β H[v]` along `u`, after
/// `D_β H[v]` is itself differenced against the joint Hessian. A score-warp coordinate moves
/// the flex primaries, so a rigid primary evaluator misses its curvature; `dJ` has q rows only,
/// so `dJᵀ H dJ` never reaches the slope or flex columns. Both directions move the wiggle
/// coefficients and the entry and exit design rows are nonzero, so every `m_k` moves through
/// `γ` as well as through `h` in the base and marginal columns.
#[test]
fn timewiggle_beta_hessian_second_directional_derivative_matches_finite_difference_2893() {
    for score_warp in [None, Some(test_deviation_runtime())] {
        let label = if score_warp.is_some() {
            "timewiggle + score warp"
        } else {
            "timewiggle"
        };
        let family = timewiggle_marginal_slope_family(score_warp);
        let beta = timewiggle_marginal_slope_beta(&family);
        let states_at = |beta: &Array1<f64>| timewiggle_marginal_slope_states(&family, beta);
        let u = Array1::from_shape_fn(beta.len(), |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
        let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
        let h = 1e-3;
        let gate = |what: &str, analytic: &Array2<f64>, at: &dyn Fn(f64) -> Array2<f64>| {
            let coarse = (at(h) - at(-h)) / (2.0 * h);
            let fine = (at(0.5 * h) - at(-0.5 * h)) / h;
            let scale = analytic
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()))
                .max(1e-12);
            for ((index, &want), (&c, &f)) in analytic
                .indexed_iter()
                .zip(coarse.iter().zip(fine.iter()))
            {
                let value = (4.0 * f - c) / 3.0;
                let uncertainty = (f - c).abs() / 3.0;
                let denominator = scale.max(want.abs()).max(value.abs());
                assert!(
                    uncertainty <= 0.05 * denominator,
                    "{label}: {what}{index:?}: the difference oracle did not resolve \
                     (value={value:.6e}, uncertainty={uncertainty:.3e})"
                );
                assert!(
                    (want - value).abs() <= 1e-5 * denominator + 4.0 * uncertainty,
                    "{label}: {what}{index:?}: analytic={want:.9e} fd={value:.9e} \
                     uncertainty={uncertainty:.3e} scale={scale:.3e}"
                );
            }
        };
        let states = states_at(&beta);
        let first = family
            .exact_newton_joint_hessian_directional_derivative(&states, &v)
            .expect("D_beta H[v]")
            .expect("a time wiggle publishes D_beta H");
        gate("D_beta H[v]", &first, &|t| {
            family
                .exact_newton_joint_hessian(&states_at(&(&beta + &(&v * t))))
                .expect("joint Hessian")
                .expect("survival marginal-slope publishes an explicit joint Hessian")
        });
        let second = family
            .exact_newton_joint_hessiansecond_directional_derivative(&states, &u, &v)
            .expect("D2_beta H[u, v]")
            .expect("a time wiggle publishes D2_beta H");
        gate("D2_beta H[u, v]", &second, &|t| {
            family
                .exact_newton_joint_hessian_directional_derivative(
                    &states_at(&(&beta + &(&u * t))),
                    &v,
                )
                .expect("displaced D_beta H[v]")
                .expect("a time wiggle publishes D_beta H")
        });
    }
}

/// `timewiggle_marginal_slope_family` with its slope varying along follow-up: the exit
/// channel keeps the family's own slope design, and the entry and exit-rate channels get
/// their own rows (gam#2767).
fn timewiggle_follow_up_slope_family() -> SurvivalMarginalSlopeFamily {
    let mut family = timewiggle_marginal_slope_family(None);
    let layout: SlopeLayout = DesignMatrix::from(array![[1.0]]).into();
    family.slope_layout = layout
        .with_follow_up(
            DesignMatrix::from(array![[0.6]]),
            DesignMatrix::from(array![[0.45]]),
        )
        .expect("a shared slope layout accepts a follow-up margin");
    family
}

/// gam#2767: a time-wiggle baseline beside a follow-up-varying slope. The wiggle deforms the
/// three location primaries only, so every coefficient-space object it assembles must pull the
/// three slope channels back through their own design rows. The joint gradient is graded
/// against the joint log-likelihood, the joint Hessian against the joint gradient, `D_β H[v]`
/// against the joint Hessian, and `D²_β H[u, v]` against `D_β H[v]`, each with a
/// Ridders-certified central difference.
#[test]
fn timewiggle_follow_up_slope_beta_calculus_matches_finite_difference_2767() {
    let family = timewiggle_follow_up_slope_family();
    assert!(family.flex_timewiggle_active(), "the fixture must engage the time wiggle");
    assert!(
        family.slope_is_follow_up_varying(),
        "the fixture must run the six-primary frame"
    );
    let beta = timewiggle_marginal_slope_beta(&family);
    let states_at = |beta: &Array1<f64>| timewiggle_marginal_slope_states(&family, beta);
    let u = Array1::from_shape_fn(beta.len(), |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
    let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
    let h = 1e-3;
    let gate = |what: &str, analytic: &Array2<f64>, at: &dyn Fn(f64) -> Array2<f64>| {
        let coarse = (at(h) - at(-h)) / (2.0 * h);
        let fine = (at(0.5 * h) - at(-0.5 * h)) / h;
        let scale = analytic
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1e-12);
        for ((index, &want), (&c, &f)) in analytic
            .indexed_iter()
            .zip(coarse.iter().zip(fine.iter()))
        {
            let value = (4.0 * f - c) / 3.0;
            let uncertainty = (f - c).abs() / 3.0;
            let denominator = scale.max(want.abs()).max(value.abs());
            assert!(
                uncertainty <= 0.05 * denominator,
                "{what}{index:?}: the difference oracle did not resolve \
                 (value={value:.6e}, uncertainty={uncertainty:.3e})"
            );
            assert!(
                (want - value).abs() <= 1e-5 * denominator + 4.0 * uncertainty,
                "{what}{index:?}: analytic={want:.9e} fd={value:.9e} \
                 uncertainty={uncertainty:.3e} scale={scale:.3e}"
            );
        }
    };
    let states = states_at(&beta);

    // The score pullback is graded on its own, not only as the reference `H v` is
    // differenced against.
    let (log_likelihood, gradient) = family
        .evaluate_exact_newton_joint_gradient_dynamic_q(&states)
        .expect("joint gradient");
    assert!(log_likelihood.is_finite(), "the fixture must sit inside the follow-up domain");
    gate("g . v", &array![[gradient.dot(&v)]], &|t| {
        let displaced = family
            .evaluate_exact_newton_joint_gradient_dynamic_q(&states_at(&(&beta + &(&v * t))))
            .expect("displaced joint log-likelihood")
            .0;
        array![[displaced]]
    });

    let hessian = family
        .exact_newton_joint_hessian(&states)
        .expect("joint Hessian")
        .expect("survival marginal-slope publishes an explicit joint Hessian");

    // The entry and rate channels must reach the Hessian, or this gate cannot tell the
    // six-primary pullback from the time-constant one it replaced.
    let static_family = timewiggle_marginal_slope_family(None);
    let static_hessian = static_family
        .exact_newton_joint_hessian(&timewiggle_marginal_slope_states(&static_family, &beta))
        .expect("static joint Hessian")
        .expect("survival marginal-slope publishes an explicit joint Hessian");
    assert!(
        (&hessian - &static_hessian)
            .iter()
            .any(|difference| difference.abs() > 1e-6),
        "the follow-up channels must move the joint Hessian"
    );

    let hv = hessian.dot(&v).insert_axis(Axis(1));
    gate("H v", &hv, &|t| {
        let displaced_gradient = family
            .evaluate_exact_newton_joint_gradient_dynamic_q(&states_at(&(&beta + &(&v * t))))
            .expect("displaced joint gradient")
            .1;
        (-displaced_gradient).insert_axis(Axis(1))
    });

    let first = family
        .exact_newton_joint_hessian_directional_derivative(&states, &v)
        .expect("D_beta H[v]")
        .expect("a time wiggle publishes D_beta H");
    assert!(
        first.row(7).iter().any(|value| value.abs() > 1e-8),
        "D_beta H[v] must carry a slope row for this gate to grade the slope crosses"
    );
    gate("D_beta H[v]", &first, &|t| {
        family
            .exact_newton_joint_hessian(&states_at(&(&beta + &(&v * t))))
            .expect("joint Hessian")
            .expect("survival marginal-slope publishes an explicit joint Hessian")
    });

    let second = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &u, &v)
        .expect("D2_beta H[u, v]")
        .expect("a time wiggle publishes D2_beta H");
    assert!(
        second.row(7).iter().any(|value| value.abs() > 1e-8),
        "D2_beta H[u, v] must carry a slope row for this gate to grade the slope crosses"
    );
    gate("D2_beta H[u, v]", &second, &|t| {
        family
            .exact_newton_joint_hessian_directional_derivative(
                &states_at(&(&beta + &(&u * t))),
                &v,
            )
            .expect("displaced D_beta H[v]")
            .expect("a time wiggle publishes D_beta H")
    });
}

/// gam#2893: the build-once flex + time-wiggle sweep reproduces the single-axis `D_β H[e_a]` on
/// every coefficient axis.
#[test]
fn timewiggle_flex_all_axes_directional_derivative_matches_single_axis_2893() {
    let family = timewiggle_marginal_slope_family(Some(test_deviation_runtime()));
    let beta = timewiggle_marginal_slope_beta(&family);
    let states = timewiggle_marginal_slope_states(&family, &beta);
    let axes = family
        .exact_newton_joint_hessian_directional_derivative_timewiggle_flex_all_axes(&states)
        .expect("build-once all-axes sweep");
    assert_eq!(axes.len(), beta.len());
    for (index, swept) in axes.iter().enumerate() {
        let mut axis = Array1::<f64>::zeros(beta.len());
        axis[index] = 1.0;
        let single = family
            .exact_newton_joint_hessian_directional_derivative(&states, &axis)
            .expect("single-axis D_beta H")
            .expect("a time wiggle publishes D_beta H");
        let scale = single
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1e-12);
        let gap = (swept - &single)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            gap <= 1e-12 * scale,
            "axis {index}: build-once sweep vs single axis: gap {gap:e}, scale {scale:e}"
        );
    }
}
/// gam#2893: the time-wiggle joint third information derivative `D³H[u, v, e_a]`, served by the
/// Jeffreys hook through the ζ composition, matches a Ridders-certified central difference of the ζ
/// sweep `{D²H[v, e_a]}` along `u` on every coefficient axis, and it is symmetric under swapping its
/// third axis with a free axis. Every ζ frame is graded: the rigid program's closed-form fifth derivatives beside a
/// time-constant and a follow-up-varying slope, and the FLEX base with a score warp, alone and beside
/// an influence absorber.
#[test]
fn timewiggle_joint_third_information_matches_differenced_second_directional_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let specs: Vec<_> = states
            .iter()
            .map(|state| dummy_blockspec(state.beta.len()))
            .collect();
        let u = Array1::from_shape_fn(beta.len(), |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
        let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
        let axes = family
            .jeffreys_third_information_derivative()
            .expect("every ζ frame exposes its third information derivative")
            .third_directional_all_axes(&states, &specs, &u, &v)
            .expect("third information derivative")
            .expect("a time wiggle publishes the third information derivative");
        assert_eq!(axes.len(), beta.len());
        let scale = axes
            .iter()
            .flat_map(|matrix| matrix.iter())
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            scale > 1e-8,
            "{frame:?}: the joint third information derivative must be nonzero on this fixture"
        );
        for c in 0..beta.len() {
            for a in 0..beta.len() {
                for b in 0..beta.len() {
                    let gap = (axes[c][[a, b]] - axes[a][[c, b]]).abs();
                    assert!(
                        gap <= 1e-9 * scale,
                        "{frame:?}: D3H[u, v, e_{c}][{a}, {b}] vs D3H[u, v, e_{a}][{c}, {b}]: gap \
                         {gap:e}, scale {scale:e}"
                    );
                }
            }
        }
        // Mixed partials commute: `D³H[u, v, e_a] = D_u D²H[v, e_a]`. One Ridders ladder along u of
        // the ζ `{D²H[v, e_a]}` sweep grades every axis in four displaced passes instead of four per
        // axis; timewiggle_all_axes_second_directional_derivative_matches_single_axis_2893 grades
        // that sweep against the single-direction routine.
        assert_all_match_ridders_2893(&format!("{frame:?} D3H[u, v, e_a]"), &axes, &|t| {
            family
                .exact_newton_joint_hessian_second_directional_derivative_timewiggle_all_axes(
                    &timewiggle_marginal_slope_states(&family, &(&beta + &(&u * t))),
                    &v,
                )
                .expect("displaced D2_beta H[v, e_a] sweep")
        });
    }
}

/// gam#2893: the time-wiggle sweep of `D²_β H[u, e_a]` through the ζ composition reproduces the
/// single-direction second directional derivative on every coefficient axis, on every ζ frame.
#[test]
fn timewiggle_all_axes_second_directional_derivative_matches_single_axis_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let u = Array1::from_shape_fn(beta.len(), |i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.45);
        let swept = family
            .exact_newton_joint_hessian_second_directional_derivative_timewiggle_all_axes(
                &states, &u,
            )
            .expect("build-once all-axes second sweep");
        assert_eq!(swept.len(), beta.len());
        let single: Vec<Array2<f64>> = (0..beta.len())
            .map(|index| {
                let mut axis = Array1::<f64>::zeros(beta.len());
                axis[index] = 1.0;
                family
                    .exact_newton_joint_hessiansecond_directional_derivative(&states, &u, &axis)
                    .expect("single-axis D2_beta H")
                    .expect("a time wiggle publishes D2_beta H")
            })
            .collect();
        let scale = single
            .iter()
            .flat_map(|matrix| matrix.iter())
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            scale > 1e-8,
            "{frame:?}: D2_beta H[u, e_a] must be nonzero on this fixture"
        );
        for (index, (swept_axis, single_axis)) in swept.iter().zip(single.iter()).enumerate() {
            let gap = (swept_axis - single_axis)
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            assert!(
                gap <= 1e-10 * scale,
                "{frame:?} axis {index}: ζ sweep vs single axis: gap {gap:e}, scale {scale:e}"
            );
        }
    }
}

/// The ζ frames of the #2893 design-difference gates: the rigid row program beside a time-constant
/// and a follow-up-varying slope, and the FLEX program with a score warp, alone and beside an
/// influence absorber.
#[derive(Clone, Copy, Debug)]
enum TimewiggleDesignPsiFrame {
    Rigid,
    RigidFollowUpSlope,
    ScoreWarp,
    ScoreWarpInfluence,
}

impl TimewiggleDesignPsiFrame {
    const ALL: [Self; 4] = [
        Self::Rigid,
        Self::RigidFollowUpSlope,
        Self::ScoreWarp,
        Self::ScoreWarpInfluence,
    ];

    /// The frame's family.
    fn family(self) -> SurvivalMarginalSlopeFamily {
        match self {
            Self::Rigid => timewiggle_marginal_slope_family(None),
            Self::RigidFollowUpSlope => timewiggle_follow_up_slope_family(),
            Self::ScoreWarp => timewiggle_marginal_slope_family(Some(test_deviation_runtime())),
            Self::ScoreWarpInfluence => {
                let mut family = timewiggle_marginal_slope_family(Some(test_deviation_runtime()));
                family.influence_absorber = Some(array![[0.6, -0.3]]);
                family
            }
        }
    }

    /// The design ψ axes of `timewiggle_design_psi_blocks` this frame serves. The follow-up-varying
    /// slope records no time margin, so its slope ψ is refused and only the marginal ψ is graded.
    fn psi_axes(self) -> std::ops::Range<usize> {
        match self {
            Self::RigidFollowUpSlope => 0..1,
            Self::Rigid | Self::ScoreWarp | Self::ScoreWarpInfluence => 0..2,
        }
    }

    /// The design ψ pairs this frame serves.
    fn psi_pairs(self) -> &'static [(usize, usize)] {
        match self {
            Self::RigidFollowUpSlope => &[(0, 0)],
            Self::Rigid | Self::ScoreWarp | Self::ScoreWarpInfluence => &[(0, 0), (0, 1), (1, 1)],
        }
    }
}

/// Two design ψ axes for `timewiggle_marginal_slope_family`: a marginal length scale, then a
/// slope length scale, each with its diagonal second design derivative.
fn timewiggle_design_psi_blocks() -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>
{
    timewiggle_design_psi_blocks_at([0.0, 0.0])
}

/// The first and second design derivatives `(X_ψ, X_ψψ)` of the marginal and the slope design ψ
/// axis of the #2893 gates.
fn timewiggle_design_psi_rows() -> [(Array2<f64>, Array2<f64>); 2] {
    [
        (array![[0.3, -0.25]], array![[0.12, 0.07]]),
        (array![[0.4]], array![[-0.15]]),
    ]
}

/// `timewiggle_design_psi_blocks` at the design ψ `t = [marginal, slope]`: each axis's first
/// design derivative moves to `X_ψ + ψ·X_ψψ`.
fn timewiggle_design_psi_blocks_at(
    t: [f64; 2],
) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
    let axis = |(x_psi, x_psi_psi): (Array2<f64>, Array2<f64>), t: f64| {
        let width = x_psi.ncols();
        crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            &x_psi + &(&x_psi_psi * t),
            Array2::zeros((width, width)),
            None,
            Some(vec![x_psi_psi]),
            None,
            None,
        )
    };
    let [marginal, slope] = timewiggle_design_psi_rows();
    vec![
        Vec::new(),
        vec![axis(marginal, t[0])],
        vec![axis(slope, t[1])],
        Vec::new(),
    ]
}

/// The family of `frame` with its marginal and slope designs moved to the design ψ
/// `t = [marginal, slope]` of `timewiggle_design_psi_blocks_at`, `X(ψ) = X + ψ·X_ψ + ½ψ²·X_ψψ`, and
/// its block states at `beta` with every `η` rebuilt from the moved designs. Only a nonzero slope ψ
/// replaces the slope layout, so a follow-up-varying layout keeps its channels.
fn timewiggle_design_psi_displaced(
    frame: TimewiggleDesignPsiFrame,
    t: [f64; 2],
    beta: &Array1<f64>,
) -> (SurvivalMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let moved = |design: Array2<f64>, (x_psi, x_psi_psi): (Array2<f64>, Array2<f64>), t: f64| {
        design + &(&x_psi * t) + &(&x_psi_psi * (0.5 * t * t))
    };
    let [marginal_rows, slope_rows] = timewiggle_design_psi_rows();
    let mut family = frame.family();
    let marginal = moved(family.marginal_design.to_dense(), marginal_rows, t[0]);
    family.marginal_design = DesignMatrix::from(marginal);
    if t[1] != 0.0 {
        let slope = moved(
            family.slope_layout.coefficient_design().to_dense(),
            slope_rows,
            t[1],
        );
        family.slope_layout = DesignMatrix::from(slope).into();
    }
    let mut states = timewiggle_marginal_slope_states(&family, beta);
    states[2].eta = family
        .slope_layout
        .coefficient_design()
        .to_dense()
        .dot(&states[2].beta);
    (family, states)
}

/// gam#2893: on every ζ frame, the time-wiggle `{D_β_a D_β ∂_ψ H[v]}` served through the ζ composition
/// matches a Ridders-certified central difference of the ψ Hessian drift `D_β ∂_ψ H[v]` along
/// every coefficient axis, for a marginal and a slope design ψ.
#[test]
fn timewiggle_design_psi_by_beta_third_information_matches_finite_difference_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let blocks = timewiggle_design_psi_blocks();
        let options = BlockwiseFitOptions::default();
        assert!(family.timewiggle_zeta_available());
        let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
        for psi in frame.psi_axes() {
            let analytic = family
                .design_psi_hessian_second_directional_derivative_all_beta_axes_with_options(
                    &states, &blocks, psi, &v, &options,
                )
                .expect("design-by-coefficient third information derivative")
                .expect("a design ψ axis publishes its third information derivative");
            assert_eq!(analytic.len(), beta.len());
            for (axis_idx, matrix) in analytic.iter().enumerate() {
                let mut axis = Array1::<f64>::zeros(beta.len());
                axis[axis_idx] = 1.0;
                assert_matches_ridders_2893(&format!("{frame:?} ψ {psi} axis {axis_idx}"), matrix, &|t| {
                    family
                        .psi_hessian_directional_derivative_with_options(
                            &timewiggle_marginal_slope_states(&family, &(&beta + &(&axis * t))),
                            &blocks,
                            psi,
                            &v,
                            &options,
                        )
                        .expect("design ψ Hessian drift")
                        .expect("a design ψ axis publishes its Hessian drift")
                });
            }
        }
    }
}

/// gam#2893: on every ζ frame, the time-wiggle `{D_β_a ∂²_ψiψj H}` served through the ζ composition
/// matches a Ridders-certified central difference of the ψψ Hessian along every coefficient
/// axis, for the marginal diagonal, the cross-block and the slope diagonal pairs.
#[test]
fn timewiggle_design_psi_pair_third_information_matches_finite_difference_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let blocks = timewiggle_design_psi_blocks();
        let options = BlockwiseFitOptions::default();
        let total = beta.len();
        for &(psi_i, psi_j) in frame.psi_pairs() {
            let analytic = family
                .design_psi_pair_hessian_directional_derivative_all_beta_axes_with_options(
                    &states, &blocks, psi_i, psi_j, &options,
                )
                .expect("design-pair third information derivative")
                .expect("a design pair publishes its third information derivative");
            assert_eq!(analytic.len(), total);
            for (axis_idx, matrix) in analytic.iter().enumerate() {
                let mut axis = Array1::<f64>::zeros(total);
                axis[axis_idx] = 1.0;
                assert_matches_ridders_2893(
                    &format!("{frame:?} ψ pair ({psi_i},{psi_j}) axis {axis_idx}"),
                    matrix,
                    &|t| {
                        let terms = family
                            .psi_second_order_terms_inner_with_options(
                                &timewiggle_marginal_slope_states(&family, &(&beta + &(&axis * t))),
                                &blocks,
                                psi_i,
                                psi_j,
                                None,
                                &options,
                            )
                            .expect("design pair terms")
                            .expect("a design pair publishes its terms");
                        match terms.hessian_psi_psi_operator.as_ref() {
                            Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
                            None => terms.hessian_psi_psi.clone(),
                        }
                    },
                );
            }
        }
    }
}

/// gam#2893: on every ζ frame, the time-wiggle `{D_β_a ∂_ψ H}` sweep through the ζ composition reproduces
/// the single-direction ψ Hessian drift on every coefficient axis, for a marginal and a slope
/// design ψ.
#[test]
fn timewiggle_design_psi_hessian_all_beta_axes_matches_single_axis_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let blocks = timewiggle_design_psi_blocks();
        let options = BlockwiseFitOptions::default();
        for psi in frame.psi_axes() {
            let swept = family
                .psi_hessian_directional_derivatives_all_beta_axes_with_options(
                    &states, &blocks, psi, &options,
                )
                .expect("design ψ Hessian sweep")
                .expect("a time wiggle publishes the ψ Hessian sweep");
            assert_eq!(swept.len(), beta.len());
            let single: Vec<Array2<f64>> = (0..beta.len())
                .map(|index| {
                    let mut axis = Array1::<f64>::zeros(beta.len());
                    axis[index] = 1.0;
                    family
                        .psi_hessian_directional_derivative_with_options(
                            &states, &blocks, psi, &axis, &options,
                        )
                        .expect("single-axis design ψ Hessian drift")
                        .expect("a design ψ axis publishes its Hessian drift")
                })
                .collect();
            let scale = single
                .iter()
                .flat_map(|matrix| matrix.iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            assert!(
                scale > 1e-8,
                "{frame:?} ψ {psi}: the ψ Hessian drift must be nonzero on this fixture"
            );
            for (index, (swept_axis, single_axis)) in swept.iter().zip(single.iter()).enumerate() {
                let gap = (swept_axis - single_axis)
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                assert!(
                    gap <= 1e-10 * scale,
                    "{frame:?} ψ {psi} axis {index}: ζ sweep vs single axis: gap {gap:e}, scale {scale:e}"
                );
            }
        }
    }
}

/// gam#3061: on every ζ frame, the time-wiggle `{D_β_a ∂_θ H}` sweep of a baseline-chart
/// coordinate through the ζ composition reproduces the single-direction baseline Hessian drift
/// on every coefficient axis, for every coordinate of a Gompertz–Makeham chart.
#[test]
fn timewiggle_baseline_psi_hessian_all_beta_axes_matches_single_axis_3061() {
    let config = crate::survival::construction::SurvivalBaselineConfig {
        target: crate::survival::construction::SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.08),
        rate: Some(0.22),
        makeham: Some(0.04),
    };
    let geometry = Arc::new(
        crate::survival::construction::build_survival_marginal_slope_baseline_geometry(
            &array![0.25],
            &array![1.0],
            &config,
        )
        .expect("build baseline geometry")
        .expect("Gompertz-Makeham has a nonlinear baseline chart"),
    );
    for frame in TimewiggleDesignPsiFrame::ALL {
        let mut family = frame.family();
        family.family_hyper =
            SurvivalMarginalSlopeFamilyHyperState::new(Some(Arc::clone(&geometry)), None)
                .expect("install baseline family coordinates");
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let options = BlockwiseFitOptions::default();
        for axis in 0..geometry.theta.len() {
            let swept = family
                .baseline_psi_hessian_directional_derivatives_all_beta_axes_with_options(
                    &states, axis, &options,
                )
                .expect("baseline Hessian sweep")
                .expect("a time wiggle publishes the baseline Hessian sweep");
            assert_eq!(swept.len(), beta.len());
            let single: Vec<Array2<f64>> = (0..beta.len())
                .map(|index| {
                    let mut direction = Array1::<f64>::zeros(beta.len());
                    direction[index] = 1.0;
                    family
                        .baseline_exact_joint_psihessian_directional_derivative_with_options(
                            &states, axis, &direction, &options,
                        )
                        .expect("single-axis baseline Hessian drift")
                        .expect("a baseline axis publishes its Hessian drift")
                })
                .collect();
            let scale = single
                .iter()
                .flat_map(|matrix| matrix.iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            assert!(
                scale > 1e-8,
                "{frame:?} θ {axis}: the baseline Hessian drift must be nonzero on this fixture"
            );
            for (index, (swept_axis, single_axis)) in swept.iter().zip(single.iter()).enumerate() {
                let gap = (swept_axis - single_axis)
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                assert!(
                    gap <= 1e-10 * scale,
                    "{frame:?} θ {axis} axis {index}: ζ sweep vs single axis: gap {gap:e}, scale {scale:e}"
                );
            }
        }
    }
}

/// The family of `frame` on the Gompertz–Makeham chart of the #3061 gates with baseline
/// coordinate `axis` displaced by `t`: the chart is re-derived from the displaced coordinate and
/// the entry, exit and exit-derivative offsets are the displaced chart's own, so `t` moves exactly
/// what the coordinate owns.
fn timewiggle_baseline_family_at(
    frame: TimewiggleDesignPsiFrame,
    axis: usize,
    t: f64,
) -> SurvivalMarginalSlopeFamily {
    let config = crate::survival::construction::SurvivalBaselineConfig {
        target: crate::survival::construction::SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.08),
        rate: Some(0.22),
        makeham: Some(0.04),
    };
    let mut theta = crate::survival::construction::survival_baseline_theta_from_config(&config)
        .expect("baseline theta")
        .expect("Gompertz-Makeham carries a baseline chart");
    theta[axis] += t;
    let config =
        crate::survival::construction::survival_baseline_config_from_theta(config.target, &theta)
            .expect("the displaced baseline coordinate stays in the chart's domain");
    let geometry = Arc::new(
        crate::survival::construction::build_survival_marginal_slope_baseline_geometry(
            &array![0.25],
            &array![1.0],
            &config,
        )
        .expect("build baseline geometry")
        .expect("Gompertz-Makeham has a nonlinear baseline chart"),
    );
    let mut family = frame.family();
    family.offset_entry = Arc::new(geometry.offset_entry.clone());
    family.offset_exit = Arc::new(geometry.offset_exit.clone());
    family.derivative_offset_exit = Arc::new(geometry.derivative_offset_exit.clone());
    family.family_hyper = SurvivalMarginalSlopeFamilyHyperState::new(Some(geometry), None)
        .expect("install baseline family coordinates");
    family
}

/// gam#3061: on every ζ frame, the time-wiggle baseline-chart terms `∂_θ ℓ̄`, `∂_θ ∇_β ℓ̄` and
/// `∂_θ H`, served through the ζ composition, and the `{D_β_a ∂_θ H}` sweep match Ridders
/// differences of the joint objective, gradient, Hessian and `D_β H[e_a]` along every coordinate
/// of a Gompertz–Makeham chart. The chart moves `h₀`, `h₁` and `∂h₁` under the wiggle, so the
/// warp's own response to the moved offsets is part of every graded object.
#[test]
fn timewiggle_baseline_psi_terms_match_finite_difference_3061() {
    let options = BlockwiseFitOptions::default();
    for frame in TimewiggleDesignPsiFrame::ALL {
        let base = timewiggle_baseline_family_at(frame, 0, 0.0);
        assert!(base.timewiggle_zeta_available());
        let beta = timewiggle_marginal_slope_beta(&base);
        let states = timewiggle_marginal_slope_states(&base, &beta);
        let total = beta.len();
        let specs: Vec<_> = states
            .iter()
            .map(|state| dummy_blockspec(state.beta.len()))
            .collect();
        let theta_len = base
            .rigid_baseline_geometry()
            .expect("the fixture installs a baseline chart")
            .theta
            .len();
        for axis in 0..theta_len {
            let label = format!("{frame:?} θ {axis}");
            let joint_at = |t: f64| {
                let family = timewiggle_baseline_family_at(frame, axis, t);
                let displaced = timewiggle_marginal_slope_states(&family, &beta);
                let evaluation = family
                    .exact_newton_joint_gradient_evaluation(&displaced, &specs)
                    .expect("joint gradient evaluation")
                    .expect("survival marginal-slope publishes a joint gradient evaluation");
                let hessian = family
                    .exact_newton_joint_hessian(&displaced)
                    .expect("joint hessian")
                    .expect("survival marginal-slope publishes an explicit joint hessian");
                (-evaluation.log_likelihood, -evaluation.gradient, hessian)
            };
            let terms = base
                .baseline_exact_joint_psi_terms_with_options(&states, axis, &options)
                .expect("baseline θ terms")
                .expect("a nonlinear baseline chart publishes θ terms");
            assert!(
                terms.hessian_psi_operator.is_none(),
                "{label}: the ζ composition publishes a dense θ Hessian"
            );
            assert_matches_ridders_2893(
                &format!("{label} objective"),
                &Array2::from_elem((1, 1), terms.objective_psi),
                &|t| Array2::from_elem((1, 1), joint_at(t).0),
            );
            assert_matches_ridders_2893(
                &format!("{label} score"),
                &terms.score_psi.clone().insert_axis(Axis(1)),
                &|t| joint_at(t).1.insert_axis(Axis(1)),
            );
            assert_matches_ridders_2893(&format!("{label} Hessian"), &terms.hessian_psi, &|t| {
                joint_at(t).2
            });
            let swept = base
                .baseline_psi_hessian_directional_derivatives_all_beta_axes_with_options(
                    &states, axis, &options,
                )
                .expect("baseline Hessian sweep")
                .expect("a time wiggle publishes the baseline Hessian sweep");
            assert_all_match_ridders_2893(&format!("{label} sweep"), &swept, &|t| {
                let family = timewiggle_baseline_family_at(frame, axis, t);
                let displaced = timewiggle_marginal_slope_states(&family, &beta);
                (0..total)
                    .map(|index| {
                        let mut direction = Array1::<f64>::zeros(total);
                        direction[index] = 1.0;
                        family
                            .exact_newton_joint_hessian_directional_derivative(
                                &displaced, &direction,
                            )
                            .expect("displaced D_beta H[e_a]")
                            .expect("a time wiggle publishes D_beta H")
                    })
                    .collect()
            });
        }
    }
}

/// gam#3061: on every ζ frame, the time-wiggle baseline-chart third information derivatives an
/// armed Jeffreys outer Hessian reads, `{D_β_a D_β ∂_θ H[v]}` and `{D_β_a ∂²_θθ' H}`, served
/// through the ζ composition, match Ridders differences of the ζ `D²_β H[v, e_a]` sweep and the
/// `{D_β_a ∂_θ H}` sweep along every coordinate of a Gompertz–Makeham chart.
#[test]
fn timewiggle_baseline_psi_third_information_matches_finite_difference_3061() {
    let options = BlockwiseFitOptions::default();
    for frame in TimewiggleDesignPsiFrame::ALL {
        let base = timewiggle_baseline_family_at(frame, 0, 0.0);
        assert!(base.timewiggle_zeta_available());
        let beta = timewiggle_marginal_slope_beta(&base);
        let states = timewiggle_marginal_slope_states(&base, &beta);
        let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
        let theta_len = base
            .rigid_baseline_geometry()
            .expect("the fixture installs a baseline chart")
            .theta
            .len();
        for axis in 0..theta_len {
            let by_beta = base
                .baseline_exact_joint_psihessian_second_directional_derivative_all_beta_axes_with_options(
                    &states, axis, &v, &options,
                )
                .expect("baseline-by-coefficient third information derivative");
            assert_eq!(by_beta.len(), beta.len());
            assert_all_match_ridders_2893(&format!("{frame:?} θ {axis} by β"), &by_beta, &|t| {
                let family = timewiggle_baseline_family_at(frame, axis, t);
                family
                    .exact_newton_joint_hessian_second_directional_derivative_timewiggle_all_axes(
                        &timewiggle_marginal_slope_states(&family, &beta),
                        &v,
                    )
                    .expect("displaced D²_β H[v, e_a] sweep")
            });
            for other_axis in 0..theta_len {
                let pair = base
                    .baseline_exact_joint_psisecond_order_hessian_directional_derivative_all_beta_axes_with_options(
                        &states, axis, other_axis, &options,
                    )
                    .expect("baseline-pair third information derivative");
                assert_eq!(pair.len(), beta.len());
                assert_all_match_ridders_2893(
                    &format!("{frame:?} θ pair ({axis},{other_axis})"),
                    &pair,
                    &|t| {
                        let family = timewiggle_baseline_family_at(frame, other_axis, t);
                        family
                            .baseline_psi_hessian_directional_derivatives_all_beta_axes_with_options(
                                &timewiggle_marginal_slope_states(&family, &beta),
                                axis,
                                &options,
                            )
                            .expect("displaced baseline Hessian sweep")
                            .expect("a time wiggle publishes the baseline Hessian sweep")
                    },
                );
            }
        }
    }
}

/// gam#3304: on every ζ frame, the time-wiggle baseline-chart objects the outer Hessian reads
/// beside `∂_θ H`, served through the ζ composition, match Ridders differences along the chart:
/// the drift `D_β ∂_θ H[v]` against the displaced `D_β H[v]`, the chart pair terms
/// `∂²_θθ' {ℓ̄, ∇_β ℓ̄, H}` against the displaced `∂_θ` terms, and the chart-by-design pair
/// terms `∂²_θψ {ℓ̄, ∇_β ℓ̄, H}` against the displaced design ψ terms, for every coordinate of a
/// Gompertz–Makeham chart and every design ψ the frame serves.
#[test]
fn timewiggle_baseline_psi_drift_and_pair_terms_match_finite_difference_3304() {
    let options = BlockwiseFitOptions::default();
    let blocks = timewiggle_design_psi_blocks();
    let dense_hessian = |dense: &Array2<f64>, operator: Option<&Arc<dyn gam_problem::HyperOperator>>, total: usize| {
        match operator {
            Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
            None => dense.clone(),
        }
    };
    for frame in TimewiggleDesignPsiFrame::ALL {
        let base = timewiggle_baseline_family_at(frame, 0, 0.0);
        assert!(base.timewiggle_zeta_available());
        let beta = timewiggle_marginal_slope_beta(&base);
        let states = timewiggle_marginal_slope_states(&base, &beta);
        let total = beta.len();
        let v = Array1::from_shape_fn(total, |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
        let theta_len = base
            .rigid_baseline_geometry()
            .expect("the fixture installs a baseline chart")
            .theta
            .len();
        for axis in 0..theta_len {
            let label = format!("{frame:?} θ {axis}");
            let drift = base
                .baseline_exact_joint_psihessian_directional_derivative_with_options(
                    &states, axis, &v, &options,
                )
                .expect("baseline Hessian drift")
                .expect("a baseline axis publishes its Hessian drift");
            assert_matches_ridders_2893(&format!("{label} drift"), &drift, &|t| {
                let family = timewiggle_baseline_family_at(frame, axis, t);
                family
                    .exact_newton_joint_hessian_directional_derivative(
                        &timewiggle_marginal_slope_states(&family, &beta),
                        &v,
                    )
                    .expect("displaced D_beta H[v]")
                    .expect("a time wiggle publishes D_beta H")
            });
            for other_axis in 0..theta_len {
                let pair_label = format!("{frame:?} θ pair ({axis},{other_axis})");
                let pair = base
                    .baseline_exact_joint_psisecond_order_terms_with_options(
                        &states, axis, other_axis, &options,
                    )
                    .expect("baseline pair terms")
                    .expect("a baseline pair publishes its terms");
                let pair_hessian =
                    dense_hessian(&pair.hessian_psi_psi, pair.hessian_psi_psi_operator.as_ref(), total);
                let first_at = |t: f64| {
                    let family = timewiggle_baseline_family_at(frame, other_axis, t);
                    family
                        .baseline_exact_joint_psi_terms_with_options(
                            &timewiggle_marginal_slope_states(&family, &beta),
                            axis,
                            &options,
                        )
                        .expect("displaced baseline θ terms")
                        .expect("a nonlinear baseline chart publishes θ terms")
                };
                assert_matches_ridders_2893(
                    &format!("{pair_label} objective"),
                    &Array2::from_elem((1, 1), pair.objective_psi_psi),
                    &|t| Array2::from_elem((1, 1), first_at(t).objective_psi),
                );
                assert_matches_ridders_2893(
                    &format!("{pair_label} score"),
                    &pair.score_psi_psi.clone().insert_axis(Axis(1)),
                    &|t| first_at(t).score_psi.insert_axis(Axis(1)),
                );
                assert_matches_ridders_2893(&format!("{pair_label} Hessian"), &pair_hessian, &|t| {
                    let first = first_at(t);
                    dense_hessian(&first.hessian_psi, first.hessian_psi_operator.as_ref(), total)
                });
            }
            for psi in frame.psi_axes() {
                let mixed_label = format!("{frame:?} θ {axis} × ψ {psi}");
                let mixed = base
                    .baseline_design_exact_joint_psisecond_order_terms_with_options(
                        &states, &blocks, axis, psi, &options,
                    )
                    .expect("baseline-by-design pair terms")
                    .expect("a baseline-by-design pair publishes its terms");
                let mixed_hessian =
                    dense_hessian(&mixed.hessian_psi_psi, mixed.hessian_psi_psi_operator.as_ref(), total);
                let design_at = |t: f64| {
                    let family = timewiggle_baseline_family_at(frame, axis, t);
                    family
                        .psi_terms(&timewiggle_marginal_slope_states(&family, &beta), &blocks, psi)
                        .expect("displaced design ψ terms")
                        .expect("a design ψ publishes its terms")
                };
                assert_matches_ridders_2893(
                    &format!("{mixed_label} objective"),
                    &Array2::from_elem((1, 1), mixed.objective_psi_psi),
                    &|t| Array2::from_elem((1, 1), design_at(t).objective_psi),
                );
                assert_matches_ridders_2893(
                    &format!("{mixed_label} score"),
                    &mixed.score_psi_psi.clone().insert_axis(Axis(1)),
                    &|t| design_at(t).score_psi.insert_axis(Axis(1)),
                );
                assert_matches_ridders_2893(&format!("{mixed_label} Hessian"), &mixed_hessian, &|t| {
                    let first = design_at(t);
                    dense_hessian(&first.hessian_psi, first.hessian_psi_operator.as_ref(), total)
                });
            }
        }
    }
}

#[test]
fn link_flex_blockwise_exact_newton_matches_joint_principal_blocks() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let slope_design = array![[1.0], [0.5]];
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 2,
        entry_at_origin: Arc::new(Array1::from_elem(2, false)),
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(slope_design.clone())).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let marginal_beta = array![0.35, -0.1];
    let slope_beta = array![0.2];
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_design.dot(&slope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    assert_blockwise_matches_joint_principal_blocks(&family, &block_states);
}

#[test]
fn link_flex_marginal_psi_terms_return_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let slope_beta = array![0.2];
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_beta.clone(),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(score_runtime.basis_dim()),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(link_runtime.basis_dim()),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_terms(&block_states, &derivative_blocks, 0)
        .expect("link flex psi terms should evaluate")
        .expect("psi terms should exist");
    assert!(terms.objective_psi.is_finite());
    assert_eq!(terms.score_psi.len(), slices.total);
    assert!(terms.score_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_operator.is_some());
}

#[test]
fn link_flex_marginal_psi_second_order_returns_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let slope_design = array![[1.2, -0.3]];
    let slope_beta = array![0.2, -0.05];
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(slope_design.clone())).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_design.dot(&slope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[0.3, 0.8]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_second_order_terms(&block_states, &derivative_blocks, 0, 1)
        .expect("link flex psi second-order path should evaluate")
        .expect("psi second-order terms should exist");
    assert!(terms.objective_psi_psi.is_finite());
    assert_eq!(terms.score_psi_psi.len(), slices.total);
    assert!(terms.score_psi_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_psi_operator.is_some());
}

#[test]
fn link_flex_marginal_psi_hessian_directional_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_flat = Array1::zeros(slices.total);
    d_beta_flat[slices.time.start] = 0.07;
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_flat[h_range.start] = 0.02;
    }
    if let Some(w_range) = slices.link_dev.as_ref() {
        d_beta_flat[w_range.start] = -0.03;
    }

    let hess_dir = family
        .psi_hessian_directional_derivative(&block_states, &derivative_blocks, 0, &d_beta_flat)
        .expect("link flex psi-Hessian directional path should evaluate")
        .expect("psi-Hessian directional derivative should exist");
    assert_eq!(hess_dir.nrows(), slices.total);
    assert_eq!(hess_dir.ncols(), slices.total);
    assert!(hess_dir.iter().all(|value| value.is_finite()));
}

#[test]
fn timewiggle_marginal_psi_terms_return_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_terms(&block_states, &derivative_blocks, 0)
        .expect("timewiggle psi terms should evaluate")
        .expect("psi terms should exist");
    assert!(terms.objective_psi.is_finite());
    assert_eq!(terms.score_psi.len(), slices.total);
    assert!(terms.score_psi.iter().all(|value| value.is_finite()));
    // A time wiggle takes the ζ composition, which publishes a dense Hessian-ψ (gam#2893).
    let hessian_psi = match terms.hessian_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(slices.total)),
        None => terms.hessian_psi.clone(),
    };
    assert_eq!(hessian_psi.dim(), (slices.total, slices.total));
    assert!(hessian_psi.iter().all(|value| value.is_finite()));
}

#[test]
fn timewiggle_blockwise_exact_newton_matches_joint_principal_blocks() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let marginal_beta = array![0.35, -0.1];
    let slope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 2,
        entry_at_origin: Arc::new(Array1::from_elem(2, false)),
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0], [0.5]])).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: array![0.2, 0.1],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    assert_blockwise_matches_joint_principal_blocks(&family, &block_states);
}

#[test]
fn flex_timewiggle_fast_gradient_matches_dense_joint_gradient() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let slope_design = array![[1.0]];
    let slope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(slope_design.clone())).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_design.dot(&slope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];

    let (fast_ll, fast_grad) = family
        .evaluate_exact_newton_joint_gradient_dynamic_q(&block_states)
        .expect("fast gradient should evaluate");
    let (dense_ll, dense_grad, _) = family
        .evaluate_exact_newton_joint_dynamic_q_dense(&block_states)
        .expect("dense joint derivatives should evaluate");

    assert_close(fast_ll, dense_ll, 1e-10, "log-likelihood");
    assert_eq!(fast_grad.len(), dense_grad.len());
    for idx in 0..fast_grad.len() {
        assert_close(
            fast_grad[idx],
            dense_grad[idx],
            1e-8,
            &format!("gradient[{idx}]"),
        );
    }
}

/// #932 genus / sibling-of-flex-Hessian witness: the survival marginal-slope
/// dynamic-q JOINT Hessian is a HAND-assembled chain-rule pullback of the
/// per-row primary (4×4) Hessian through the `dq`/`d2q` time-wiggle geometry
/// (`accumulate_dynamic_q_core_hessian` + `row_dynamic_q_geometry_into`). The
/// flex primary-Hessian bug (a dropped/mis-signed 2nd-order coupling that
/// shipped because its FD oracle was committed UNRUN) proved that an
/// unvalidated bespoke pullback can be silently wrong. The ONLY pre-existing
/// joint-Hessian test here checks `blockwise == joint-dense` — both sides share
/// THIS pullback, so it cannot catch a wrong pullback. This oracle closes that
/// gap directly: it central-differences the full joint GRADIENT and compares to
/// the analytic joint HESSIAN on the TIME-WIGGLE-active path (the only path
/// where `q` is nonlinear in the coefficients, hence the only path with a
/// nontrivial `d2q` that a hand-derivation can get wrong). A real wrong
/// derivative is h-independent and orders above the FD floor; FD truncation
/// shrinks with h.
#[test]
fn timewiggle_joint_hessian_matches_central_fd_of_joint_gradient() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let slope_design = array![[1.0]];
    let slope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.4, -0.1, 0.2, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.6, 0.3, -0.15, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.2, -0.1, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(slope_design.clone())).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    assert!(
        family.flex_timewiggle_active(),
        "fixture must engage the nonlinear time-wiggle q geometry"
    );

    // Base block states. Per-block coefficient consumption (verified against the
    // geometry / dense entry): the TIME block (0) feeds `q` through its design
    // directly (its `eta` is unused); the MARGINAL (1) and SLOPE (2) blocks
    // feed through their `eta = design·beta`; the SCORE-WARP (3) and LINK-DEV
    // (4) deviation blocks feed through their `beta` directly. A faithful
    // perturbation therefore rebuilds `eta = design·beta` for blocks 1 and 2.
    let base_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_design.dot(&slope_beta),
        },
        ParameterBlockState {
            // score-warp block width is `basis_dim * score_dim`; with a
            // single score column here that is `basis_dim`. Seed it with a
            // small smoothly-varying nonzero so the deviation block carries
            // genuine curvature into the joint Hessian.
            beta: Array1::from_iter(
                (0..score_runtime.basis_dim() * family.score_dim())
                    .map(|j| 0.03 * (-0.4_f64).powi(j as i32)),
            ),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::from_iter(
                (0..link_runtime.basis_dim()).map(|j| -0.02 * (0.5_f64).powi(j as i32)),
            ),
            eta: Array1::zeros(1),
        },
    ];
    assert_eq!(
        base_states[3].beta.len(),
        score_runtime.basis_dim() * family.score_dim()
    );
    assert_eq!(base_states[4].beta.len(), link_runtime.basis_dim());

    let slices = block_slices(&family, &base_states);
    let p_total = slices.total;

    // Rebuild block states from a flat coefficient vector, keeping each block's
    // `eta` consistent with its `beta` exactly as the production solver does.
    let states_from_flat = |flat: &Array1<f64>| -> Vec<ParameterBlockState> {
        let mut states = base_states.clone();
        states[0].beta = flat.slice(s![slices.time.clone()]).to_owned();
        states[1].beta = flat.slice(s![slices.marginal.clone()]).to_owned();
        states[1].eta = marginal_design.dot(&states[1].beta);
        states[2].beta = flat.slice(s![slices.slope.clone()]).to_owned();
        states[2].eta = slope_design.dot(&states[2].beta);
        if let Some(range) = slices.score_warp.clone() {
            states[3].beta = flat.slice(s![range]).to_owned();
        }
        if let Some(range) = slices.link_dev.clone() {
            let block_index = 3 + usize::from(family.score_warp.is_some());
            states[block_index].beta = flat.slice(s![range]).to_owned();
        }
        states
    };

    // Flatten the base states into the joint coefficient layout.
    let mut base_flat = Array1::<f64>::zeros(p_total);
    base_flat
        .slice_mut(s![slices.time.clone()])
        .assign(&base_states[0].beta);
    base_flat
        .slice_mut(s![slices.marginal.clone()])
        .assign(&base_states[1].beta);
    base_flat
        .slice_mut(s![slices.slope.clone()])
        .assign(&base_states[2].beta);
    if let Some(range) = slices.score_warp.clone() {
        base_flat.slice_mut(s![range]).assign(&base_states[3].beta);
    }
    if let Some(range) = slices.link_dev.clone() {
        let block_index = 3 + usize::from(family.score_warp.is_some());
        base_flat
            .slice_mut(s![range])
            .assign(&base_states[block_index].beta);
    }

    let joint_gradient_at = |flat: &Array1<f64>| -> Array1<f64> {
        let states = states_from_flat(flat);
        let (_ll, grad) = family
            .evaluate_exact_newton_joint_gradient_dynamic_q(&states)
            .expect("perturbed joint gradient");
        grad
    };

    let (_ll, _grad, analytic_hessian) = family
        .evaluate_exact_newton_joint_dynamic_q_dense(&base_states)
        .expect("analytic joint dense gradient + hessian");
    assert_eq!(analytic_hessian.shape(), &[p_total, p_total]);

    // Central-difference step. The joint gradient is smooth in every joint
    // coefficient; 1e-5 balances O(h^2) truncation against the FP cancellation
    // floor of the per-perturbation primary intercept re-solve.
    let h = 1e-5;
    let mut max_rel = 0.0_f64;
    let mut worst = (0usize, 0usize, 0.0_f64, 0.0_f64);
    for u in 0..p_total {
        let mut plus = base_flat.clone();
        plus[u] += h;
        let mut minus = base_flat.clone();
        minus[u] -= h;
        let grad_plus = joint_gradient_at(&plus);
        let grad_minus = joint_gradient_at(&minus);
        for v in 0..p_total {
            // The joint "gradient" returned by the exact-Newton path is the
            // score (∇ of the log-likelihood = −∇nll), while the joint dense
            // Hessian is the observed information (+∇²nll = −∇(score)). The
            // analytic Hessian therefore equals the NEGATED Jacobian of the
            // joint gradient, so the central FD must carry the leading minus —
            // exactly as every sibling FD-vs-Hessian oracle in the codebase
            // does (cf. survival/location_scale `fd_neggrad_jac`). Omitting it
            // produced an exact sign flip on every entry.
            let fd = -(grad_plus[v] - grad_minus[v]) / (2.0 * h);
            let analytic = analytic_hessian[[v, u]];
            let denom = 1.0 + analytic.abs().max(fd.abs());
            let rel = (analytic - fd).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
                worst = (v, u, analytic, fd);
            }
        }
    }

    assert!(
        max_rel <= 1e-6,
        "timewiggle joint Hessian disagrees with central FD of the joint gradient: \
         worst entry H[{}][{}] = {:.6e} vs FD {:.6e} (rel {max_rel:.3e}); a chain-rule \
         pullback term (dq/d2q) is dropped or mis-signed",
        worst.0,
        worst.1,
        worst.2,
        worst.3,
    );
}

#[test]
fn row_dynamic_q_geometry_into_pooled_matches_fresh_allocation_bitwise() {
    // Regression: `row_dynamic_q_geometry_into` reuses a caller-owned
    // `SurvivalMarginalSlopeDynamicRow` workspace (resized + zero-filled
    // in place) instead of allocating nine fresh Array2/Array1 buffers
    // per row. Both code paths must return bit-for-bit identical
    // contents, with the workspace path additionally verified to leave
    // the same answer when re-entered against an already-populated
    // buffer (so the in-place `reset` correctly wipes stale state).
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let marginal_beta = array![0.35, -0.1];
    let slope_design = array![[1.0], [0.5]];
    let slope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 2,
        entry_at_origin: Arc::new(Array1::from_elem(2, false)),
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(slope_design.clone())).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_design.dot(&slope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    // Compare each row, twice — first into a virgin workspace, then
    // into the same workspace re-used after a different row has
    // already populated it. Both must equal the fresh-allocation path
    // bit-for-bit.
    let mut workspace = SurvivalMarginalSlopeDynamicRow::empty_workspace();
    // Pre-load the workspace with row 1 so the row 0 comparison
    // exercises the buffer-reuse zeroing logic on a non-trivial state.
    family
        .row_dynamic_q_geometry_into(1, &block_states, &mut workspace)
        .expect("preload workspace with row 1");
    for row in [0usize, 1usize, 0usize] {
        let fresh = family
            .row_dynamic_q_geometry(row, &block_states)
            .expect("fresh-allocation row geometry");
        family
            .row_dynamic_q_geometry_into(row, &block_states, &mut workspace)
            .expect("pooled-workspace row geometry");
        assert_eq!(workspace.q0.to_bits(), fresh.q0.to_bits(), "row {row} q0");
        assert_eq!(workspace.q1.to_bits(), fresh.q1.to_bits(), "row {row} q1");
        assert_eq!(
            workspace.qd1.to_bits(),
            fresh.qd1.to_bits(),
            "row {row} qd1"
        );
        let array1_pairs: [(&Array1<f64>, &Array1<f64>, &str); 6] = [
            (&workspace.dq0_time, &fresh.dq0_time, "dq0_time"),
            (&workspace.dq1_time, &fresh.dq1_time, "dq1_time"),
            (&workspace.dqd1_time, &fresh.dqd1_time, "dqd1_time"),
            (&workspace.dq0_marginal, &fresh.dq0_marginal, "dq0_marginal"),
            (&workspace.dq1_marginal, &fresh.dq1_marginal, "dq1_marginal"),
            (
                &workspace.dqd1_marginal,
                &fresh.dqd1_marginal,
                "dqd1_marginal",
            ),
        ];
        for (lhs, rhs, label) in array1_pairs {
            assert_eq!(lhs.shape(), rhs.shape(), "row {row} {label} shape");
            for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert_eq!(
                    l.to_bits(),
                    r.to_bits(),
                    "row {row} {label}[{i}] lhs={l:.17e} rhs={r:.17e}",
                );
            }
        }
        let array2_pairs: [(&Array2<f64>, &Array2<f64>, &str); 9] = [
            (
                &workspace.d2q0_time_time,
                &fresh.d2q0_time_time,
                "d2q0_time_time",
            ),
            (
                &workspace.d2q1_time_time,
                &fresh.d2q1_time_time,
                "d2q1_time_time",
            ),
            (
                &workspace.d2qd1_time_time,
                &fresh.d2qd1_time_time,
                "d2qd1_time_time",
            ),
            (
                &workspace.d2q0_time_marginal,
                &fresh.d2q0_time_marginal,
                "d2q0_time_marginal",
            ),
            (
                &workspace.d2q1_time_marginal,
                &fresh.d2q1_time_marginal,
                "d2q1_time_marginal",
            ),
            (
                &workspace.d2qd1_time_marginal,
                &fresh.d2qd1_time_marginal,
                "d2qd1_time_marginal",
            ),
            (
                &workspace.d2q0_marginal_marginal,
                &fresh.d2q0_marginal_marginal,
                "d2q0_marginal_marginal",
            ),
            (
                &workspace.d2q1_marginal_marginal,
                &fresh.d2q1_marginal_marginal,
                "d2q1_marginal_marginal",
            ),
            (
                &workspace.d2qd1_marginal_marginal,
                &fresh.d2qd1_marginal_marginal,
                "d2qd1_marginal_marginal",
            ),
        ];
        for (lhs, rhs, label) in array2_pairs {
            assert_eq!(lhs.shape(), rhs.shape(), "row {row} {label} shape");
            for ((idx, l), r) in lhs.indexed_iter().zip(rhs.iter()) {
                assert_eq!(
                    l.to_bits(),
                    r.to_bits(),
                    "row {row} {label}[{idx:?}] lhs={l:.17e} rhs={r:.17e}",
                );
            }
        }
    }
}

#[test]
fn flex_timewiggle_operator_to_dense_matches_evaluate_dense_joint_hessian() {
    // Regression: SurvivalMarginalSlopeExactNewtonJointHessianWorkspace::
    // hessian_dense() now returns the already-built operator's
    // to_dense() instead of re-running
    // evaluate_exact_newton_joint_dynamic_q_dense (a second full n-row
    // sweep). Both code paths must agree on the joint p×p Hessian.
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let marginal_beta = array![0.35, -0.1];
    let slope_design = array![[1.0], [0.5]];
    let slope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 2,
        entry_at_origin: Arc::new(Array1::from_elem(2, false)),
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(slope_design.clone())).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_design.dot(&slope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    let (_, _, dense) = family
        .evaluate_exact_newton_joint_dynamic_q_dense(&block_states)
        .expect("dense joint Hessian");
    let (operator, _, _, _) = family
        .exact_newton_joint_hessian_operator(&block_states, &BlockwiseFitOptions::default())
        .expect("joint Hessian operator");
    let op_dense = operator.to_dense();

    assert_eq!(op_dense.shape(), dense.shape());
    let diff = max_abs_diff_mat(&op_dense, &dense);
    assert!(
        diff <= 1e-10,
        "operator.to_dense() differs from evaluate_dense by {diff:.3e}",
    );
}

#[test]
fn timewiggle_marginal_slope_psi_second_order_returns_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let slope_design = array![[1.2, -0.3]];
    let slope_beta = array![0.2, -0.05];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(slope_design.clone())).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: slope_beta.clone(),
            eta: slope_design.dot(&slope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[0.3, 0.8]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_second_order_terms(&block_states, &derivative_blocks, 0, 1)
        .expect("timewiggle scorewarp psi second-order path should evaluate")
        .expect("psi second-order terms should exist");
    assert!(terms.objective_psi_psi.is_finite());
    assert_eq!(terms.score_psi_psi.len(), slices.total);
    assert!(terms.score_psi_psi.iter().all(|value| value.is_finite()));
    // A time wiggle takes the ζ composition, which publishes a dense Hessian-ψψ (gam#2893).
    let hessian_psi_psi = match terms.hessian_psi_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(slices.total)),
        None => terms.hessian_psi_psi.clone(),
    };
    assert_eq!(hessian_psi_psi.dim(), (slices.total, slices.total));
    assert!(hessian_psi_psi.iter().all(|value| value.is_finite()));
}

#[test]
fn timewiggle_marginal_psi_hessian_directional_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: None,
        n: 1,
        entry_at_origin: Arc::new(Array1::from_elem(1, false)),
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        slope_layout: (DesignMatrix::from(array![[1.0]])).into(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        flex_jet_arenas: new_flex_jet_arena_pool(),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_flat = Array1::zeros(slices.total);
    d_beta_flat[slices.time.start] = 0.07;
    d_beta_flat[slices.time.start + 1] = -0.03;
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.slope.start] = -0.04;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_flat[h_range.start] = 0.02;
    }

    let slices = block_slices(&family, &block_states);
    let hess_dir = family
        .psi_hessian_directional_derivative(&block_states, &derivative_blocks, 0, &d_beta_flat)
        .expect("timewiggle scorewarp psi-Hessian directional path should evaluate")
        .expect("psi-Hessian directional derivative should exist");
    assert_eq!(hess_dir.nrows(), slices.total);
    assert_eq!(hess_dir.ncols(), slices.total);
    assert!(hess_dir.iter().all(|value| value.is_finite()));
}

/// gam#2893: on every ζ frame, the time-wiggle ψ Hessian sweep `{D_β_a ∂_ψ H}` served through the ζ
/// composition matches a Ridders difference of the joint `D_β H[e_a]` along the design motion of
/// a marginal and a slope design ψ. Mixed partials commute, so this grades the ζ sweep from the
/// joint Hessian calculus alone, without the ψ calculus of `psi_terms`.
#[test]
fn timewiggle_design_psi_hessian_sweep_matches_design_difference_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let blocks = timewiggle_design_psi_blocks();
        let options = BlockwiseFitOptions::default();
        for psi in frame.psi_axes() {
            let swept = family
                .psi_hessian_directional_derivatives_all_beta_axes_with_options(
                    &states, &blocks, psi, &options,
                )
                .expect("design ψ Hessian sweep")
                .expect("a time wiggle publishes the ψ Hessian sweep");
            assert_eq!(swept.len(), beta.len());
            for (axis_idx, matrix) in swept.iter().enumerate() {
                let mut axis = Array1::<f64>::zeros(beta.len());
                axis[axis_idx] = 1.0;
                assert_matches_ridders_2893(&format!("{frame:?} ψ {psi} axis {axis_idx}"), matrix, &|t| {
                    let mut moved_t = [0.0; 2];
                    moved_t[psi] = t;
                    let (moved, moved_states) = timewiggle_design_psi_displaced(frame, moved_t, &beta);
                    moved
                        .exact_newton_joint_hessian_directional_derivative(&moved_states, &axis)
                        .expect("displaced D_β H[e_a]")
                        .expect("a time wiggle publishes D_β H")
                });
            }
        }
    }
}

/// gam#2893: on every ζ frame, the time-wiggle `{D_β_a D_β ∂_ψ H[v]}` served through the ζ composition
/// matches a Ridders difference of the ζ `D²_β H[v, e_a]` sweep along the design motion of a
/// marginal and a slope design ψ, without the ψ calculus of `psi_terms`.
#[test]
fn timewiggle_design_psi_by_beta_third_matches_design_difference_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let blocks = timewiggle_design_psi_blocks();
        let options = BlockwiseFitOptions::default();
        let v = Array1::from_shape_fn(beta.len(), |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
        for psi in frame.psi_axes() {
            let analytic = family
                .design_psi_hessian_second_directional_derivative_all_beta_axes_with_options(
                    &states, &blocks, psi, &v, &options,
                )
                .expect("design-by-coefficient third information derivative")
                .expect("a design ψ axis publishes its third information derivative");
            assert_eq!(analytic.len(), beta.len());
            for (axis_idx, matrix) in analytic.iter().enumerate() {
                assert_matches_ridders_2893(&format!("{frame:?} ψ {psi} axis {axis_idx}"), matrix, &|t| {
                    let mut moved_t = [0.0; 2];
                    moved_t[psi] = t;
                    let (moved, moved_states) = timewiggle_design_psi_displaced(frame, moved_t, &beta);
                    moved
                        .exact_newton_joint_hessian_second_directional_derivative_timewiggle_all_axes(
                            &moved_states,
                            &v,
                        )
                        .expect("displaced D²_β H[v, e_a] sweep")
                        .swap_remove(axis_idx)
                });
            }
        }
    }
}

/// gam#2893: on every ζ frame, the time-wiggle `{D_β_a ∂²_ψiψj H}` served through the ζ composition matches
/// a Ridders difference of the ζ ψ Hessian sweep `{D_β_a ∂_ψi H}` along the design motion of ψ_j,
/// for the marginal diagonal, the cross-block and the slope diagonal pairs. On a diagonal pair the
/// ψ_i design derivative itself moves to `X_ψ + ψ·X_ψψ`.
#[test]
fn timewiggle_design_psi_pair_third_matches_design_difference_2893() {
    for frame in TimewiggleDesignPsiFrame::ALL {
        let family = frame.family();
        let beta = timewiggle_marginal_slope_beta(&family);
        let states = timewiggle_marginal_slope_states(&family, &beta);
        let blocks = timewiggle_design_psi_blocks();
        let options = BlockwiseFitOptions::default();
        for &(psi_i, psi_j) in frame.psi_pairs() {
            let analytic = family
                .design_psi_pair_hessian_directional_derivative_all_beta_axes_with_options(
                    &states, &blocks, psi_i, psi_j, &options,
                )
                .expect("design-pair third information derivative")
                .expect("a design pair publishes its third information derivative");
            assert_eq!(analytic.len(), beta.len());
            for (axis_idx, matrix) in analytic.iter().enumerate() {
                assert_matches_ridders_2893(
                    &format!("{frame:?} ψ pair ({psi_i},{psi_j}) axis {axis_idx}"),
                    matrix,
                    &|t| {
                        let mut moved_t = [0.0; 2];
                        moved_t[psi_j] = t;
                        let (moved, moved_states) = timewiggle_design_psi_displaced(frame, moved_t, &beta);
                        moved
                            .psi_hessian_directional_derivatives_all_beta_axes_with_options(
                                &moved_states,
                                &timewiggle_design_psi_blocks_at(moved_t),
                                psi_i,
                                &options,
                            )
                            .expect("displaced design ψ Hessian sweep")
                            .expect("a time wiggle publishes the ψ Hessian sweep")
                            .swap_remove(axis_idx)
                    },
                );
            }
        }
    }
}

/// gam#2893: the time-wiggle design ψ terms `∂_ψ ℓ̄`, `∂_ψ ∇_β ℓ̄` and `∂_ψ H` and the ψ Hessian
/// drift `D_β ∂_ψ H[v]`, served through the ζ composition, match Ridders differences of the joint
/// objective, gradient, Hessian and `D_β H[v]` along the design motion of a marginal and a slope
/// design ψ a frame serves. They are the outer ψ gradient and Hessian inputs. Every ζ frame is
/// graded: the rigid program beside a time-constant and a follow-up-varying slope, and the FLEX
/// program with a score warp, alone and beside an influence absorber.
#[test]
fn timewiggle_design_psi_terms_and_drift_match_design_difference_2893() {
    let blocks = timewiggle_design_psi_blocks();
    let options = BlockwiseFitOptions::default();
    for frame in TimewiggleDesignPsiFrame::ALL {
        let base = frame.family();
        assert!(base.timewiggle_zeta_available());
        let beta = timewiggle_marginal_slope_beta(&base);
        let states = timewiggle_marginal_slope_states(&base, &beta);
        let total = beta.len();
        let specs: Vec<_> = states
            .iter()
            .map(|state| dummy_blockspec(state.beta.len()))
            .collect();
        let v = Array1::from_shape_fn(total, |i| ((i * 5 + 1) % 13) as f64 / 13.0 - 0.5);
        for psi in frame.psi_axes() {
            let label = format!("{frame:?} ψ {psi}");
            let displaced_at = |t: f64| {
                let mut motion = [0.0; 2];
                motion[psi] = t;
                timewiggle_design_psi_displaced(frame, motion, &beta)
            };
            let joint_at = |t: f64| {
                let (family, displaced) = displaced_at(t);
                let evaluation = family
                    .exact_newton_joint_gradient_evaluation(&displaced, &specs)
                    .expect("joint gradient evaluation")
                    .expect("survival marginal-slope publishes a joint gradient evaluation");
                let hessian = family
                    .exact_newton_joint_hessian(&displaced)
                    .expect("joint hessian")
                    .expect("survival marginal-slope publishes an explicit joint hessian");
                (-evaluation.log_likelihood, -evaluation.gradient, hessian)
            };
            let terms = base
                .psi_terms(&states, &blocks, psi)
                .expect("design ψ terms")
                .expect("a design ψ publishes its terms");
            let hessian_psi = match terms.hessian_psi_operator.as_ref() {
                Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
                None => terms.hessian_psi.clone(),
            };
            assert_matches_ridders_2893(
                &format!("{label} objective"),
                &Array2::from_elem((1, 1), terms.objective_psi),
                &|t| Array2::from_elem((1, 1), joint_at(t).0),
            );
            assert_matches_ridders_2893(
                &format!("{label} score"),
                &terms.score_psi.clone().insert_axis(Axis(1)),
                &|t| joint_at(t).1.insert_axis(Axis(1)),
            );
            assert_matches_ridders_2893(&format!("{label} Hessian"), &hessian_psi, &|t| {
                joint_at(t).2
            });
            let drift = base
                .psi_hessian_directional_derivative_with_options(&states, &blocks, psi, &v, &options)
                .expect("design ψ Hessian drift")
                .expect("a design ψ publishes its Hessian drift");
            assert_matches_ridders_2893(&format!("{label} drift"), &drift, &|t| {
                let (family, displaced) = displaced_at(t);
                family
                    .exact_newton_joint_hessian_directional_derivative(&displaced, &v)
                    .expect("displaced D_beta H[v]")
                    .expect("a time wiggle publishes D_beta H")
            });
        }
    }
}

/// gam#2893: the time-wiggle design ψ pair terms `∂²_ψiψj ℓ̄`, `∂²_ψiψj ∇_β ℓ̄` and `∂²_ψiψj H`,
/// served through the ζ composition, match Ridders differences of the ψ_i terms along the design
/// motion of ψ_j, for the marginal diagonal, the cross-block pair and the slope diagonal on every ζ
/// frame that serves them. On a diagonal pair the ψ_i design derivative itself moves to
/// `X_ψ + ψ·X_ψψ`.
#[test]
fn timewiggle_design_psi_pair_terms_match_design_difference_2893() {
    let blocks = timewiggle_design_psi_blocks();
    let options = BlockwiseFitOptions::default();
    for frame in TimewiggleDesignPsiFrame::ALL {
        let base = frame.family();
        let beta = timewiggle_marginal_slope_beta(&base);
        let states = timewiggle_marginal_slope_states(&base, &beta);
        let total = beta.len();
        for &(psi_i, psi_j) in frame.psi_pairs() {
            let label = format!("{frame:?} ψ pair ({psi_i},{psi_j})");
            let terms = base
                .psi_second_order_terms_inner_with_options(
                    &states, &blocks, psi_i, psi_j, None, &options,
                )
                .expect("design pair terms")
                .expect("a design pair publishes its terms");
            let hessian_psi_psi = match terms.hessian_psi_psi_operator.as_ref() {
                Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
                None => terms.hessian_psi_psi.clone(),
            };
            let first_at = |t: f64| {
                let mut motion = [0.0; 2];
                motion[psi_j] = t;
                let (family, displaced) = timewiggle_design_psi_displaced(frame, motion, &beta);
                let first = family
                    .psi_terms(&displaced, &timewiggle_design_psi_blocks_at(motion), psi_i)
                    .expect("displaced design ψ terms")
                    .expect("a design ψ publishes its terms");
                let hessian = match first.hessian_psi_operator.as_ref() {
                    Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
                    None => first.hessian_psi.clone(),
                };
                (first.objective_psi, first.score_psi, hessian)
            };
            assert_matches_ridders_2893(
                &format!("{label} objective"),
                &Array2::from_elem((1, 1), terms.objective_psi_psi),
                &|t| Array2::from_elem((1, 1), first_at(t).0),
            );
            assert_matches_ridders_2893(
                &format!("{label} score"),
                &terms.score_psi_psi.clone().insert_axis(Axis(1)),
                &|t| first_at(t).1.insert_axis(Axis(1)),
            );
            assert_matches_ridders_2893(&format!("{label} Hessian"), &hessian_psi_psi, &|t| {
                first_at(t).2
            });
        }
    }
}
