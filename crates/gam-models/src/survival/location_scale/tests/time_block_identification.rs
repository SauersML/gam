//! Survival location-scale time-block and link-wiggle construction: feasible steps inside
//! the derivative guard, identified time blocks and their null spaces, the pinned time-warp lift,
//! structural coefficient bounds, and the time initializers and gauge priorities model
//! preparation assigns.
#![cfg(test)]

use super::*;

#[test]
fn time_block_post_update_leaves_beta_unchanged() {
    // The QP owns feasibility. The post-update hook may validate the
    // accepted beta, but it must not silently repair a missing constraint
    // row after the solver has produced a step.
    let family = survival_exact_newton_test_family();
    let spec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, 1)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let feasible = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            }],
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &spec,
            array![0.5],
        )
        .expect("return time beta");
    assert_eq!(feasible, array![0.5]);

    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            }],
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &spec,
            array![-2.0],
        )
        .expect_err("post-update must reject, not repair, infeasible time beta");
    assert!(
        err.contains("violates represented linear constraint"),
        "unexpected error: {err}"
    );
}

#[test]
fn time_block_feasible_step_stays_inside_derivative_guard() {
    let family = survival_exact_newton_test_family();
    let states = vec![
        ParameterBlockState {
            beta: array![0.1],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(
            &states,
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &array![-2.0],
        )
        .expect("time step ceiling")
        .expect("time step should be bounded");
    // The guard row is a unit row here, so scaled slack is `0.1` and the scaled
    // drift of `-2.0` is `-2.0`: the exact fraction to the boundary is `0.05`.
    // The clipped step stops one primal-feasibility tolerance short of the face
    // in that metric — an ABSOLUTE retreat, `tol/|scaled drift|`, not a fraction
    // of the step (gam#2695).
    assert!(
        (alpha - 0.05).abs() <= 1e-12,
        "a clipped step lands on the blocking face: alpha={alpha:.12e}"
    );
    let feasible = states[0].beta[0] + alpha * -2.0;
    // The clipped step lands ON the blocking face, so the row is tight there and
    // can enter the active-set solver's working face (gam#2695, gam#2714).
    assert!(
        feasible.abs() <= 1.0e-12,
        "the clipped endpoint must sit on the face, got {feasible:.6e}"
    );
}

#[test]
fn latent_time_constraints_use_exact_derivative_guard_rows() {
    let constraints = structural_time_coefficient_constraints(
        &DesignMatrix::from(array![[1.0, 1.0], [2.0, -1.0]]),
        &array![0.25, 0.75],
        1.0,
    )
    .expect("exact derivative guard constraints")
    .expect("nonzero derivative rows");

    let scale0 = 2.0_f64.sqrt();
    let scale1 = 5.0_f64.sqrt();
    let expected_a = array![[1.0 / scale0, 1.0 / scale0], [2.0 / scale1, -1.0 / scale1]];
    let expected_b = array![0.75 / scale0, 0.25 / scale1];
    assert!(
        (&constraints.a - &expected_a)
            .iter()
            .all(|v| v.abs() <= 1e-12),
        "scaled A mismatch: got {:?}, expected {:?}",
        constraints.a,
        expected_a
    );
    assert!(
        (&constraints.b - &expected_b)
            .iter()
            .all(|v| v.abs() <= 1e-12),
        "scaled b mismatch: got {:?}, expected {:?}",
        constraints.b,
        expected_b
    );
}

#[test]
fn time_block_feasible_step_accepts_zero_beta_when_offset_encodes_guard() {
    let family = survival_exact_newton_test_family();
    let states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 1e-8],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(
            &states,
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &array![0.0],
        )
        .expect("zero-step structural state should be valid")
        .expect("time step should be bounded");
    assert_eq!(alpha, 1.0);
}

#[test]
fn max_feasible_link_wiggle_step_refuses_a_non_finite_direction_2721() {
    let mut family = survival_exact_newton_test_family();
    family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])));
    let beta = array![1.0, 1.0];
    // Positive control: a finite BINDING direction is evaluated and clipped.
    let bounded = family
        .max_feasible_link_wiggle_step(&beta, &array![-2.0, 0.0])
        .expect("a finite direction must be evaluated")
        .expect("the linkwiggle step fraction is always reported");
    assert!(
        bounded > 0.0 && bounded < 1.0,
        "a binding finite direction should clip the step, got {bounded}"
    );
    // The defect (gam#2721): NaN fails `drift < 0.0`, so this returned Ok(1.0)
    // -- a step that is not a number, certified as fully feasible.
    let message = family
        .max_feasible_link_wiggle_step(&beta, &array![f64::NAN, 0.0])
        .expect_err("a non-finite direction component must be refused");
    assert!(
        message.contains("non-finite"),
        "the refusal must name the non-finite component, got: {message}"
    );
}

/// gam#2719, the witness geometry. The link-wiggle seed sits at `beta == 0`,
/// exactly on every face of its own cone, and the joint-Newton direction has a
/// tiny negative component there. The measured drifts on
/// `survival_location_scale_saved_fit_preserves_linkwiggle_metadata` run down
/// to `-3.291437e-18`; the old coordinate loop answered `alpha = 0` for every
/// one of them, and 314 of the 379 refusals it produced were of steps whose
/// endpoint the solver's own `1e-8` contract calls feasible.
#[test]
fn linkwiggle_step_admits_a_sub_tolerance_drift_off_an_active_coefficient() {
    let mut family = survival_exact_newton_test_family();
    family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])));
    let on_the_face = array![0.0, 0.0];
    let measured_drift = array![-3.291_437e-18, -5.808_407e-18];
    let alpha = family
        .max_feasible_link_wiggle_step(&on_the_face, &measured_drift)
        .expect("an in-band drift keeps a feasible origin")
        .expect("the linkwiggle step fraction is always reported");
    assert_eq!(
        alpha, 1.0,
        "a drift ten orders below the feasibility contract must not limit the step"
    );
    // And the claim that relief rests on: the endpoint really is feasible.
    let endpoint = &on_the_face + &measured_drift;
    let cone = crate::wiggle::monotone_wiggle_nonnegative_constraints(endpoint.len())
        .expect("the block declares its cone");
    let (violation, _) = cone
        .max_scaled_violation(endpoint.view())
        .expect("violation sweep");
    assert!(violation <= gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL);

    // Positive control on the same face: a drift ABOVE the contract still
    // blocks, and reports `0.0` as an ANSWER rather than as an error — the
    // caller must project onto the face, and no smaller step can help.
    let real_drift = array![-3.961_401e-6, 0.0];
    let blocked = family
        .max_feasible_link_wiggle_step(&on_the_face, &real_drift)
        .expect("a blocked face is an answer, not an error")
        .expect("the linkwiggle step fraction is always reported");
    assert_eq!(blocked, 0.0);
}

/// The barrier hook and the constraint set the blockwise QP enforces must be
/// the same cone. They are built from one constructor now; this pins that a
/// step the hook admits is a step the QP's own feasibility metric accepts, for
/// a spread of directions including the pathological one.
#[test]
fn linkwiggle_barrier_hook_agrees_with_the_declared_cone() {
    let mut family = survival_exact_newton_test_family();
    family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])));
    let cone = crate::wiggle::monotone_wiggle_nonnegative_constraints(2).expect("cone");
    let cases = [
        (array![0.0_f64, 0.0], array![-1.0e-18_f64, -1.0e-18]),
        (array![0.0_f64, 0.5], array![-1.0e-6_f64, -1.0]),
        (array![0.25_f64, 0.5], array![-1.0_f64, 0.25]),
        (array![2.0_f64, 3.0], array![0.5_f64, 0.5]),
        (array![0.0_f64, 1.0], array![1.0_f64, -1.0e-12]),
    ];
    for (beta, direction) in cases {
        let alpha = family
            .max_feasible_link_wiggle_step(&beta, &direction)
            .unwrap_or_else(|e| panic!("hook refused {beta:?} along {direction:?}: {e}"))
            .expect("the linkwiggle step fraction is always reported");
        let endpoint = &beta + &(&direction * alpha);
        let (violation, row) = cone
            .max_scaled_violation(endpoint.view())
            .expect("violation sweep");
        assert!(
            violation <= gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
            "hook admitted alpha={alpha:.6e} from {beta:?} along {direction:?}, \
             leaving scaled violation {violation:.3e} at row {row:?}"
        );
    }
}

#[test]
fn linkwiggle_block_post_update_leaves_beta_unchanged() {
    let mut family = survival_exact_newton_test_family();
    family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])));
    family.wiggle_knots = Some(array![-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0]);
    family.wiggle_degree = Some(3);
    let spec = ParameterBlockSpec {
        name: "linkwiggle".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, 2)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let returned = family
        .post_update_block_beta(
            &[
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.1, 0.2],
                    eta: array![0.0, 0.0, 0.0],
                },
            ],
            SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
            &spec,
            array![0.3, 0.0],
        )
        .expect("return linkwiggle beta");
    assert_eq!(returned, array![0.3, 0.0]);

    let err = family
        .post_update_block_beta(
            &[
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                },
                ParameterBlockState {
                    beta: array![0.1, 0.2],
                    eta: array![0.0, 0.0, 0.0],
                },
            ],
            SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
            &spec,
            array![0.3, -0.1],
        )
        .expect_err("infeasible link-wiggle beta must be rejected");
    assert!(
        err.contains("violates represented nonnegativity"),
        "unexpected error: {err}"
    );
}

#[test]
fn linkwiggle_block_feasible_step_stays_nonnegative() {
    let mut family = survival_exact_newton_test_family();
    family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])));
    family.wiggle_knots = Some(array![-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0]);
    family.wiggle_degree = Some(3);
    let states = vec![
        ParameterBlockState {
            beta: array![0.1],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.2, 0.4],
            eta: array![0.0, 0.0, 0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(
            &states,
            SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
            &array![-1.0, -0.1],
        )
        .expect("linkwiggle step ceiling")
        .expect("linkwiggle step should be bounded");
    assert!(alpha > 0.0 && alpha < 1.0);
    let feasible = &states[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE].beta
        + &(array![-1.0, -0.1] * alpha);
    assert!(feasible.iter().all(|&value| value >= 0.0));
}

#[test]
fn wip_outergradient_testspecs_shape() {
    let specs = survival_outergradient_testspecs();
    assert_eq!(specs.len(), 3);
    assert_eq!(specs[0].name, "time_transform");
    assert_eq!(specs[1].name, "threshold");
    assert_eq!(specs[2].name, "log_sigma");
}

#[test]
fn identified_time_block_preserves_input_designs() {
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(prepared.design_entry, design_entry);
    assert_eq!(prepared.design_exit, design_exit);
    assert_eq!(prepared.design_derivative_exit, design_derivative_exit);
}

#[test]
fn identified_time_block_preserves_expected_nullspace_dimension() {
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry),
        design_exit: DesignMatrix::from(design_exit),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };

    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    let p = time_block.design_entry.ncols();

    assert_eq!(
        prepared.transform.gauge.raw_total(),
        p,
        "identifiability transform must stay in the original coefficient space"
    );
    assert_eq!(
        prepared.transform.gauge.reduced_total(),
        p,
        "anchored time basis should keep the full coefficient dimension"
    );
    assert_eq!(
        prepared.design_entry.ncols(),
        p,
        "prepared entry design should keep the full anchored basis width"
    );
    assert_eq!(
        prepared.design_exit.ncols(),
        p,
        "prepared exit design should keep the full anchored basis width"
    );
    assert_eq!(
        prepared.transform.gauge.block_transform(0),
        Array2::<f64>::eye(p)
    );
    assert_eq!(
        prepared.transform.gauge.affine_shift,
        Array1::<f64>::zeros(p)
    );
}

#[test]
fn identified_time_block_can_reduce_to_parametric_penalty_nullspace() {
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: Some(array![0.5, 0.2, 9.0]),
    };

    // log(t_exit) for the unit-log-t warp-slope pin (issue #892).
    let log_time_entry = array![-1.0_f64, -0.5, 0.0];
    let log_time_exit = array![0.0_f64, 0.405_465_108, 0.916_290_731];
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        true,
        log_time_entry.view(),
        log_time_exit.view(),
    )
    .expect("prepare time block");
    // Canonical gauge pin (#892): the warp slope is folded into the offset,
    // so the FREE time block collapses to the single row-constant direction.
    // The Gauge map is now p×1 (was p×2), with the pinned unit-log-t
    // warp carried by `Gauge::affine_shift` rather than a free column.
    assert_eq!(prepared.transform.gauge.raw_total(), 3);
    assert_eq!(prepared.transform.gauge.reduced_total(), 1);
    assert_eq!(prepared.transform.gauge.affine_shift.len(), 3);
    assert!(
        prepared
            .transform
            .gauge
            .affine_shift
            .iter()
            .any(|&v| v.abs() > 1e-9),
        "pinned warp must contribute a non-zero Gauge affine_shift"
    );
    assert_eq!(prepared.design_entry.ncols(), 1);
    assert_eq!(prepared.design_exit.ncols(), 1);
    assert_eq!(prepared.design_derivative_exit.ncols(), 1);
    assert!(prepared.coefficient_lower_bounds.is_none());
    // The reduced block lives on the penalty null space, so `zᵀ S z` is
    // exactly zero: there is no curvature left to penalize. An unpenalized
    // parametric block has no smoothing parameter, so the projected-to-zero
    // penalties are dropped entirely — the block carries ZERO penalties and
    // therefore contributes no ρ coordinate to the outer REML search
    // (issue #736/#735/#721).
    assert!(
        prepared.penalties.is_empty(),
        "reduced parametric time block must be unpenalized (no smoothing parameter), got {} penalties",
        prepared.penalties.len()
    );
    assert!(
        prepared.nullspace_dims.is_empty(),
        "reduced parametric time block carries no penalty null-space bookkeeping"
    );
}

#[test]
fn pinned_time_warp_affine_lift_round_trips() {
    // Golden round-trip (issue #892): on a rank-clean pinned reduced fit the
    // raw time coefficients must be reconstructed EXACTLY through the
    // Gauge-owned affine section `β_raw = T · θ + a`. A wrong lift silently
    // corrupts every reported survival time-coefficient, so this guards the
    // finalize math directly. Choose a known reduced free coefficient `θ` and
    // verify the lifted raw coefficient reproduces both the free constant
    // direction (`θ · z_c`) and the pinned unit-log-t warp (`a`),
    // and that the design image `X · β_raw` equals
    // `(X · z_c) θ + X · a` (the free design plus the folded
    // offset), which is what the geometry actually consumes.
    let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
    let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let log_time_entry = array![-1.0_f64, -0.5, 0.0];
    let log_time_exit = array![0.0_f64, 0.405_465_108, 0.916_290_731];
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        true,
        log_time_entry.view(),
        log_time_exit.view(),
    )
    .expect("prepare time block");
    // Pin fired: single free column + non-zero pinned warp.
    assert_eq!(prepared.transform.gauge.reduced_total(), 1);
    let theta = array![0.731_f64];
    let beta_raw = prepared
        .transform
        .gauge
        .lift_block_betas(&[theta.clone()])
        .remove(0);
    // β_raw equals the free contribution plus the pinned warp, exactly.
    let z_c = prepared.transform.gauge.block_transform(0);
    let expected_raw =
        &(&z_c.column(0).to_owned() * theta[0]) + &prepared.transform.gauge.affine_shift;
    for (got, want) in beta_raw.iter().zip(expected_raw.iter()) {
        assert!(
            (got - want).abs() <= 1e-12,
            "affine lift must reconstruct raw coefficients exactly: got {got}, want {want}"
        );
    }
    // The raw design image matches free-design·θ + augmented offset delta,
    // i.e. what the solver geometry sees: X·β_raw = (X·z_c)·θ + X·z_t.
    let raw_image = design_exit.dot(&beta_raw);
    let folded = &prepared.design_exit.column(0).to_owned() * theta[0]
        + &(&prepared.offset_exit - &time_block.offset_exit);
    for (got, want) in raw_image.iter().zip(folded.iter()) {
        assert!(
            (got - want).abs() <= 1e-9,
            "raw design image must equal free image plus folded offset: got {got}, want {want}"
        );
    }
    // The folded exit offset has unit slope vs log t (the canonical gauge).
    let delta = &prepared.offset_exit - &time_block.offset_exit;
    let log_mean = log_time_exit.sum() / 3.0;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..3 {
        let xc = log_time_exit[i] - log_mean;
        sxx += xc * xc;
        sxy += xc * (delta[i] - delta.sum() / 3.0);
    }
    assert!(
        (sxy / sxx - 1.0).abs() <= 1e-9,
        "pinned warp must have unit data-scale slope vs log t, got {}",
        sxy / sxx
    );
}

#[test]
fn rank1_reduced_time_warp_removes_warp_and_flags_location_log_time() {
    // The real survival regime (issue #892): a 1st-difference time penalty
    // gives a DIMENSION-1 null space — a single monotone log-t column. The
    // reduce must REMOVE the time warp entirely (zero free columns, empty
    // designs + p×0 transform, zero value/derivative offsets so `h ≡ 0`, no
    // constraint, no penalties) and instead FLAG `location_log_time_offset`,
    // so the caller carries the σ-scaled `log t` baseline on the location `q`
    // channel (u = inv_sigma·(log t − η_t)). The threshold keeps its intercept
    // (`pinned_free_row_constant == false`). A penalty `diag(0,1,1)` has the
    // 1-D null space {e0}; design column 0 is monotone in log t.
    let design_entry = array![
        [0.0, 1.0, 0.2],
        [0.405_465_108, 1.0, 0.5],
        [0.916_290_731, 1.0, 1.0]
    ];
    let design_exit = array![
        [0.0, 0.5, 0.3],
        [0.405_465_108, 1.5, 0.8],
        [0.916_290_731, 2.5, 1.4]
    ];
    let design_derivative_exit = array![[1.0, 1.0, 0.2], [0.5, 1.0, 0.3], [0.3, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let log_time_entry = array![-1.0_f64, -0.5, 0.0];
    let log_time_exit = array![0.0_f64, 0.405_465_108, 0.916_290_731];
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        true,
        log_time_entry.view(),
        log_time_exit.view(),
    )
    .expect("prepare time block");

    // Warp removed: zero free columns, empty designs + p×0 transform.
    assert_eq!(prepared.transform.gauge.reduced_total(), 0);
    assert_eq!(prepared.transform.gauge.raw_total(), 3);
    assert_eq!(prepared.design_exit.ncols(), 0);
    assert_eq!(prepared.design_entry.ncols(), 0);
    assert_eq!(prepared.design_derivative_exit.ncols(), 0);
    assert_eq!(prepared.design_exit.nrows(), 3);
    assert_eq!(prepared.initial_beta, Some(Array1::<f64>::zeros(0)));
    // No free coefficients → no derivative-guard constraint, no penalties.
    assert!(prepared.linear_constraints.is_none());
    assert!(prepared.penalties.is_empty());
    // `h ≡ 0`: zero value offsets and zero derivative offset (the warp is gone;
    // the log-t baseline lives on the location channel, not here).
    assert_eq!(prepared.offset_exit, Array1::<f64>::zeros(3));
    assert_eq!(prepared.offset_entry, Array1::<f64>::zeros(3));
    assert_eq!(prepared.derivative_offset_exit, Array1::<f64>::zeros(3));
    // No affine shift; the location-log-time flag is set.
    assert!(
        prepared
            .transform
            .gauge
            .affine_shift
            .iter()
            .all(|&v| v.abs() <= 1e-12)
    );
    assert!(
        prepared.location_log_time_offset,
        "rank-1 reduce must flag the σ-scaled log-t location baseline"
    );
    // No free time column → threshold keeps its intercept.
    assert!(!prepared.pinned_free_row_constant);
}

#[test]
fn identified_time_block_uses_structural_coefficient_constraints() {
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]]),
        design_exit: DesignMatrix::from(array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]]),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: Some(array![-0.5, 0.2, -1.5]),
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(
        prepared.coefficient_lower_bounds,
        Some(array![f64::NEG_INFINITY, 0.0, 0.0])
    );
    let constraints = lower_bound_constraints(
        prepared
            .coefficient_lower_bounds
            .as_ref()
            .expect("time coefficient lower bounds"),
    )
    .expect("time coefficient constraints");
    assert_eq!(constraints.a, array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert_eq!(constraints.b, Array1::<f64>::zeros(2));
    assert_eq!(prepared.initial_beta, Some(array![-0.5, 0.2, 0.0]));
}

#[test]
fn identified_time_block_constrains_monotone_timewiggle_tail_coefficients() {
    let design_derivative_exit = array![
        [0.0, 1.0, 0.2, 0.0],
        [0.0, 1.0, 0.3, 0.0],
        [0.0, 1.0, 0.4, 0.0]
    ];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(array![
            [1.0, 0.0, 0.2, 0.0],
            [1.0, 1.0, 0.5, 0.0],
            [1.0, 2.0, 1.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [1.0, 0.5, 0.3, 0.0],
            [1.0, 1.5, 0.8, 0.0],
            [1.0, 2.5, 1.4, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![Array2::eye(4)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: Some(array![-0.5, 0.2, -1.5, -2.0]),
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        1,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(
        prepared.coefficient_lower_bounds,
        Some(array![f64::NEG_INFINITY, 0.0, 0.0, 0.0])
    );
    assert_eq!(prepared.initial_beta, Some(array![-0.5, 0.2, 0.0, 0.0]));
}

/// #2332 regression: a genuine monotone I-spline SHAPE column whose M-spline
/// derivative support is inactive at every training row (a tail column beyond
/// the largest training exit time) must still be bound `β ≥ 0` — the exact
/// domain-wide monotonicity certificate — because it VARIES IN VALUE across the
/// observed entry∪exit domain (which is exactly why `keep_cols` retained it).
/// The old builder decided the sign cone from the training-row DERIVATIVE design
/// alone, so this column read as all-zero (like the free constant column) and
/// was left `NEG_INFINITY` (unconstrained); the penalized fit then drove it
/// negative and produced a non-monotone warp at prediction horizons in its
/// support. Column 2 below is exactly that tail column: value rises 0 → 0.6
/// between the last two rows (so it survives `keep_cols`) while its exit-time
/// derivative is 0 at every training row.
#[test]
fn structural_bounds_constrain_derivative_inactive_tail_shape_column() {
    // col 0: free level/intercept (value-constant [1,1,1], derivative ≡ 0).
    // col 1: ordinary active shape column (value varies, derivative active).
    // col 2: TAIL shape column — value varies (0 → 0.6) but derivative ≈ 0 at
    //        every training exit row (support lands past the largest exit time).
    let design_entry =
        DesignMatrix::from(array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]);
    let design_exit =
        DesignMatrix::from(array![[1.0, 0.5, 0.0], [1.0, 1.5, 0.0], [1.0, 2.5, 0.6]]);
    let design_derivative_exit =
        DesignMatrix::from(array![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
    let derivative_offset_exit = Array1::from_elem(3, 1e-6);

    let bounds = structural_time_coefficient_lower_bounds(
        &design_entry,
        &design_exit,
        &design_derivative_exit,
        &derivative_offset_exit,
        1e-6,
    )
    .expect("structural bounds")
    .expect("some bounds");
    assert_eq!(bounds, array![f64::NEG_INFINITY, 0.0, 0.0]);
    // The distinguishing signal is value-variation, NOT derivative activity:
    // the tail column (col 2) is derivative-inactive at every training row yet
    // is still bound because its value varies over entry∪exit.
    assert!(bounds[2] == 0.0, "derivative-inactive tail column must be bound");
}

/// #2332 corollary: a genuinely value-CONSTANT column (the free level/intercept)
/// must stay unconstrained even when other columns are shape columns. This pins
/// the classifier to the `keep_cols` value-variation criterion: only the
/// value-constant baseline level stays free.
#[test]
fn structural_bounds_keep_constant_level_column_free() {
    // Two value-constant columns (a constant 1 level and a constant 0 pad) plus
    // one value-varying shape column. Only the shape column is bound.
    let design_entry =
        DesignMatrix::from(array![[1.0, 0.0, 0.2], [1.0, 0.0, 0.5], [1.0, 0.0, 1.0]]);
    let design_exit =
        DesignMatrix::from(array![[1.0, 0.0, 0.3], [1.0, 0.0, 0.8], [1.0, 0.0, 1.4]]);
    let design_derivative_exit =
        DesignMatrix::from(array![[0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4]]);
    let derivative_offset_exit = Array1::from_elem(3, 1e-6);

    let bounds = structural_time_coefficient_lower_bounds(
        &design_entry,
        &design_exit,
        &design_derivative_exit,
        &derivative_offset_exit,
        1e-6,
    )
    .expect("structural bounds")
    .expect("some bounds");
    assert_eq!(bounds, array![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0]);
}

/// #2332: an all-value-constant time design (no shape column at all — e.g. the
/// empty-basis `learn_timewiggle` regime with only zero tail placeholders) still
/// returns `Ok(None)` so the caller's downstream regime handling is preserved.
#[test]
fn structural_bounds_no_shape_column_returns_none() {
    let design_entry = DesignMatrix::from(array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]);
    let design_exit = DesignMatrix::from(array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]);
    let design_derivative_exit = DesignMatrix::from(array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    let derivative_offset_exit = Array1::from_elem(3, 1e-6);

    let bounds = structural_time_coefficient_lower_bounds(
        &design_entry,
        &design_exit,
        &design_derivative_exit,
        &derivative_offset_exit,
        1e-6,
    )
    .expect("structural bounds");
    assert!(bounds.is_none(), "no shape column must return Ok(None)");
}

#[test]
fn identified_time_block_rejects_offsets_below_derivative_guard() {
    let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]]),
        design_exit: DesignMatrix::from(array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]]),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::zeros(3),
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let err = match prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    ) {
        Ok(_) => panic!("offsets below the guard must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.contains("require derivative offsets to encode the derivative guard"),
        "unexpected error: {err}"
    );
}

#[test]
fn prepare_model_accepts_time_initializer_when_offset_completes_guard() {
    let n = 3usize;
    let derivative_guard = 5e-10;
    let derivative_offset_exit = Array1::from_elem(n, 6e-10);
    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1.0),
        age_exit: Array1::from_elem(n, 5e9),
        event_target: array![1.0, 0.0, 1.0],
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: derivative_offset_exit.clone(),
            penalties: vec![Array2::zeros((1, 1))],
            nullspace_dims: vec![1],
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

    let prepared = prepare_survival_location_scale_model(&spec)
        .expect("offset-supported time initializer should be accepted");
    let beta_init = prepared.blockspecs[0]
        .initial_beta
        .as_ref()
        .expect("time initializer should be present");
    let d_raw_init = Array2::ones((n, 1)).dot(beta_init) + &derivative_offset_exit;
    assert!(
        d_raw_init.iter().all(|v| *v >= derivative_guard),
        "initializer must satisfy derivative guard once offsets are included: {d_raw_init:?}"
    );
}

#[test]
fn prepare_model_seeds_structural_time_initializer_when_offset_equals_guard() {
    let n = 20usize;
    let p_time = 8usize;
    let derivative_guard = DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD;
    let derivative_offset_exit = Array1::from_elem(n, derivative_guard);
    let age_exit = Array1::from_iter((0..n).map(|i| 4.0 + (i as f64) * 14.0));
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        for j in 0..p_time {
            let center = (j as f64 + 0.5) / (p_time as f64);
            let x = 8.0 * (t - center);
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            design_derivative_exit[[i, j]] = 8.0 * sigmoid * (1.0 - sigmoid) / age_exit[i];
        }
    }

    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1e-9),
        age_exit: age_exit.clone(),
        event_target: Array1::zeros(n),
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, p_time))),
            design_exit: DesignMatrix::from(Array2::zeros((n, p_time))),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: derivative_offset_exit.clone(),
            penalties: vec![Array2::eye(p_time)],
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

    let prepared = prepare_survival_location_scale_model(&spec)
        .expect("guard-sized derivative offset should still seed time initializer");
    let beta_init = prepared.blockspecs[0]
        .initial_beta
        .as_ref()
        .expect("time initializer should be present");
    let d_raw_init = design_derivative_exit.dot(beta_init) + &derivative_offset_exit;

    assert!(beta_init.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(beta_init.iter().any(|v| *v > 0.0));
    assert!(
        d_raw_init
            .iter()
            .all(|v| v.is_finite() && *v >= derivative_guard),
        "initializer must satisfy derivative guard once offsets are included: {d_raw_init:?}"
    );
}

#[test]
fn prepare_model_assigns_distinct_descending_gauge_priorities() {
    // Regression for #366: every location-scale block previously carried
    // the uniform `gauge_priority: 100`, which made the redundant
    // intercept direction in the flat joint design un-attributable and
    // forced the identifiability audit to refuse (`fatal = true`).  The
    // four blocks must now own strictly descending priorities so the
    // surplus constant is attributed to the lower-priority block.
    let n = 4usize;
    let derivative_guard = 1e-6;
    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1.0),
        age_exit: Array1::from_iter((0..n).map(|i| 5.0 + i as f64)),
        event_target: array![1.0, 0.0, 1.0, 1.0],
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::from_elem(n, 2e-6),
            penalties: vec![Array2::zeros((1, 1))],
            nullspace_dims: vec![1],
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

    let prepared =
        prepare_survival_location_scale_model(&spec).expect("location-scale model prepares");

    let priority = |name: &str| {
        prepared
            .blockspecs
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("missing block '{name}'"))
            .gauge_priority
    };
    let time = priority("time_transform");
    let threshold = priority("threshold");
    let log_sigma = priority("log_sigma");
    assert_eq!(
        time, 200,
        "time_transform must own the highest gauge priority"
    );
    assert!(
        time > threshold && threshold > log_sigma,
        "gauge priorities must be strictly descending so the redundant \
             intercept is attributable: time={time}, threshold={threshold}, \
             log_sigma={log_sigma}"
    );
    // The whole point of the fix: no two structural blocks may share a
    // gauge priority (equal priority is what produced the fatal audit).
    let mut seen = std::collections::HashSet::new();
    for block in &prepared.blockspecs {
        assert!(
            seen.insert(block.gauge_priority),
            "blocks must carry distinct gauge priorities; '{}' duplicates {}",
            block.name,
            block.gauge_priority,
        );
    }
}

#[test]
fn prepare_model_fixes_the_constant_log_sigma_the_threshold_scale_aliases() {
    let n = 4usize;
    let derivative_guard = 1e-6;
    let spec = SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1.0),
        age_exit: Array1::from_iter((0..n).map(|i| 5.0 + i as f64)),
        event_target: array![1.0, 0.0, 1.0, 1.0],
        weights: Array1::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Logistic),
        derivative_guard,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::from_elem(n, 2e-6),
            penalties: vec![Array2::zeros((1, 1))],
            nullspace_dims: vec![1],
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

    // Outside the σ-scaled log-t baseline every row depends on `η_t·e^{−η_σ}`
    // only, so the constant log-σ is exactly the `(η_t, η_σ) → (c·η_t, η_σ + log c)`
    // ray: it is fixed and the log-σ block keeps no column.
    let prepared =
        prepare_survival_location_scale_model(&spec).expect("location-scale model prepares");
    assert!(
        prepared.family.location_log_time.is_none(),
        "the fixture must not be the σ-scaled log-t regime, where −log σ identifies σ"
    );
    assert_eq!(
        prepared.log_sigma_fixed_cols, 1,
        "the constant log-sigma is aliased with the threshold's scale and must be fixed"
    );
    assert_eq!(prepared.log_sigma_full_ncols, 1);
    let log_sigma = prepared
        .blockspecs
        .iter()
        .find(|block| block.name == "log_sigma")
        .expect("prepared model should contain log_sigma block");
    assert_eq!(log_sigma.design.ncols(), 0);

    // A nonzero threshold offset cannot be rescaled by any c ≠ 1, so σ is
    // identified and the constant log-σ stays free.
    let mut offset_spec = spec.clone();
    if let CovariateBlockKind::Static(block) = &mut offset_spec.threshold_block {
        block.offset = Array1::from_elem(n, 0.5);
    }
    let prepared = prepare_survival_location_scale_model(&offset_spec)
        .expect("location-scale model with a threshold offset prepares");
    assert_eq!(prepared.log_sigma_fixed_cols, 0);
    let log_sigma = prepared
        .blockspecs
        .iter()
        .find(|block| block.name == "log_sigma")
        .expect("prepared model should contain log_sigma block");
    assert_eq!(log_sigma.design.ncols(), 1);
}

#[test]
fn identified_time_block_degenerate_entry_preserves_full_dimension() {
    let design_entry = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let design_exit = array![[0.1, 0.5, 0.9], [0.2, 0.6, 1.0], [0.3, 0.7, 1.0]];
    let design_derivative_exit = array![[0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [0.1, 0.1, 0.0]];
    let time_block = TimeBlockInput {
        design_entry: DesignMatrix::from(design_entry.clone()),
        design_exit: DesignMatrix::from(design_exit.clone()),
        design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
        offset_entry: Array1::zeros(3),
        offset_exit: Array1::zeros(3),
        derivative_offset_exit: Array1::from_elem(3, 1e-6),
        penalties: vec![Array2::eye(3)],
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        1e-6,
        0,
        false,
        array![-1.0_f64, -0.5, 0.0].view(),
        array![0.0_f64, 0.5, 1.0].view(),
    )
    .expect("prepare time block");
    assert_eq!(prepared.design_entry, design_entry);
    assert_eq!(prepared.design_exit, design_exit);
    assert_eq!(prepared.design_derivative_exit, design_derivative_exit);
}

#[test]
fn resolve_survival_time_anchor_defaults_to_earliest_entry() {
    let age_entry = array![5.0, 1.0, 3.0];
    let anchor =
        crate::survival::construction::survival_earliest_entry_time_anchor(&age_entry)
            .expect("resolve default anchor");
    assert!((anchor - 1.0).abs() <= 1e-12);
}

/// The #892 log-t collapse needs only a constant scale and no time wiggle, so a
/// penalized threshold beside a constant scale keeps it: the warp is removed and
/// `−log t` rides the location channel although the fit is not the fully reduced
/// parametric AFT. Saved replay dispatches on `time_parameterization`, and
/// recording such a fit as `MonotoneWarp` dropped `−log t` from its predicted
/// survival (S = 1 at every age on the age scale, a time-flat curve on the
/// follow-up scale). A smooth scale keeps the warp and must say so.
#[test]
fn time_parameterization_follows_the_log_time_collapse_not_the_smoothing_layout() {
    let (age_exit, event, _log_t) = reduced_aft_lognormal_sample(400, 1.4, 0.5, 3);
    let n = age_exit.len();
    let reduced_aft = SurvivalLocationScaleTimeParameterization::ReducedParametricAft;

    let parametric =
        prepare_survival_location_scale_model(&reduced_aft_lognormal_spec(&age_exit, &event, 1.0))
            .expect("prepare the parametric AFT");
    assert!(parametric.is_reduced_parametric_aft());
    assert_eq!(parametric.time_parameterization(), reduced_aft);

    // Intercept, x and x²; the curvature penalty leaves {1, x} unpenalized, so it
    // is a smoothing penalty (null space 2), never a parametric ridge.
    let x = Array1::from_shape_fn(n, |i| (i as f64 + 0.5) / n as f64 - 0.5);
    let mut smooth_design = Array2::<f64>::ones((n, 3));
    smooth_design.column_mut(1).assign(&x);
    smooth_design.column_mut(2).assign(&x.mapv(|v| v * v));
    let mut curvature = Array2::<f64>::zeros((3, 3));
    curvature[[2, 2]] = 1.0;
    let smooth_block = || {
        CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(smooth_design.clone()),
            offset: Array1::zeros(n),
            penalties: vec![gam_terms::penalty_spec::PenaltySpec::Dense(curvature.clone())],
            nullspace_dims: vec![2],
            initial_log_lambdas: None,
            initial_beta: None,
        })
    };

    let mut smooth_threshold = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);
    smooth_threshold.threshold_block = smooth_block();
    let smooth_threshold = prepare_survival_location_scale_model(&smooth_threshold)
        .expect("prepare a penalized threshold beside a constant scale");
    assert!(!smooth_threshold.is_reduced_parametric_aft());
    assert!(smooth_threshold.family.location_log_time.is_some());
    assert_eq!(smooth_threshold.time_parameterization(), reduced_aft);

    let mut smooth_scale = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);
    smooth_scale.log_sigma_block = smooth_block();
    let smooth_scale = prepare_survival_location_scale_model(&smooth_scale)
        .expect("prepare a smooth scale");
    assert!(smooth_scale.family.location_log_time.is_none());
    assert_eq!(
        smooth_scale.time_parameterization(),
        SurvivalLocationScaleTimeParameterization::MonotoneWarp
    );
}

/// A collapsed warp reads none of the time block's offsets (#892), so the offsets
/// a Weibull baseline target defines cannot enter a constant-scale fit: the
/// prepared time block carries only zero offsets, and the fitted likelihood and
/// coefficients are bitwise the same for every `(scale, shape)`. The target has
/// no parameter in this likelihood, which is why materialize refuses one here
/// instead of searching it.
#[test]
fn collapsed_warp_likelihood_is_invariant_in_the_weibull_target() {
    use crate::survival::construction::{
        SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode,
        build_survival_time_offsets_for_likelihood,
    };

    let (age_exit, event, _log_t) = reduced_aft_lognormal_sample(400, 1.4, 0.5, 5);
    let inverse_link = residual_distribution_inverse_link(ResidualDistribution::Gaussian);
    let fit_with_target = |scale: f64, shape: f64| {
        let mut spec = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);
        let target = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(scale),
            shape: Some(shape),
            rate: None,
            makeham: None,
        };
        let (entry, exit, derivative) = build_survival_time_offsets_for_likelihood(
            &spec.age_entry,
            &spec.age_exit,
            &target,
            SurvivalLikelihoodMode::LocationScale,
            Some(&inverse_link),
        )
        .expect("Weibull target offsets");
        assert!(
            exit.iter().any(|&value| value != 0.0),
            "the Weibull target must define nonzero time offsets"
        );
        spec.time_block.offset_entry = entry;
        spec.time_block.offset_exit = exit;
        spec.time_block.derivative_offset_exit =
            derivative + DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD;

        let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
        assert!(prepared.family.location_log_time.is_some());
        let time = &prepared.blockspecs[SurvivalLocationScaleFamily::BLOCK_TIME];
        assert!(
            time.offset.iter().all(|&value| value == 0.0)
                && time
                    .stacked_offset
                    .as_ref()
                    .is_none_or(|offset| offset.iter().all(|&value| value == 0.0)),
            "the collapsed time block must read none of the target's offsets"
        );
        let (fit, _) = fit_survival_location_scale_with_geometry(spec).expect("collapsed fit");
        (
            fit.log_likelihood_at_mode(),
            fit.beta_threshold(),
            fit.beta_log_sigma(),
        )
    };

    let (ll_a, threshold_a, log_sigma_a) = fit_with_target(3.0, 1.0);
    let (ll_b, threshold_b, log_sigma_b) = fit_with_target(9.0, 2.5);
    assert_eq!(ll_a.to_bits(), ll_b.to_bits(), "log-likelihood {ll_a} vs {ll_b}");
    assert_eq!(threshold_a, threshold_b);
    assert_eq!(log_sigma_a, log_sigma_b);
}

/// gam#3037: the flexible I-spline time block declares its coordinate cone and
/// nothing else, because the cone already implies every training row's
/// derivative guard. The premise is checked on the real survival I-spline
/// construction: the derivative design is non-negative entrywise and every
/// column carrying it is bounded, so the least value of `D_i β + o_i` over the
/// cone is `o_i`, which is the guard under the default linear baseline. Before
/// the fix, the block declared `p + n` rows for this `p`-row set. That gave a
/// degenerate vertex wherever a guard row bound, and a constrained Laplace
/// normalizer priced over every row.
#[test]
fn flexible_time_block_declares_its_coordinate_cone_once_3037() {
    let n = 80;
    let age_entry = Array1::from_shape_fn(n, |i| 18.0 + 0.5 * i as f64);
    let age_exit = Array1::from_shape_fn(n, |i| age_entry[i] + 1.0 + 2.5 * (i % 7) as f64);
    let build = crate::survival::build_survival_time_basis(
        &age_entry,
        &age_exit,
        crate::survival::SurvivalTimeBasisConfig::ISpline {
            degree: 3,
            knots: Array1::zeros(0),
            keep_cols: Vec::new(),
        },
        Some(4),
    )
    .expect("build the survival I-spline time basis");
    let guard = crate::survival::survival_derivative_guard_for_likelihood(
        crate::survival::SurvivalLikelihoodMode::LocationScale,
    );
    let time_block = TimeBlockInput {
        design_entry: build.x_entry_time.clone(),
        design_exit: build.x_exit_time.clone(),
        design_derivative_exit: build.x_derivative_time.clone(),
        offset_entry: Array1::zeros(n),
        offset_exit: Array1::zeros(n),
        // The linear baseline has no derivative of its own: each offset is the guard.
        derivative_offset_exit: Array1::from_elem(n, guard),
        penalties: build.penalties.clone(),
        nullspace_dims: build.nullspace_dims.clone(),
        initial_log_lambdas: None,
        initial_beta: None,
    };
    let prepared = prepare_identified_time_block(
        &time_block,
        guard,
        0,
        false,
        age_entry.mapv(f64::ln).view(),
        age_exit.mapv(f64::ln).view(),
    )
    .expect("prepare the flexible time block");

    let bounds = prepared
        .coefficient_lower_bounds
        .as_ref()
        .expect("the flexible block carries its cone bounds");
    let declared = prepared
        .linear_constraints
        .as_ref()
        .expect("the flexible block declares constraints");
    let cone = lower_bound_constraints(bounds).expect("the cone has rows");
    assert_eq!(declared.a, cone.a, "declared rows must be the cone rows");
    assert_eq!(declared.b, cone.b, "declared bounds must be the cone bounds");
    let bounded = bounds.iter().filter(|lower| lower.is_finite()).count();
    assert!(bounded > 0);
    assert_eq!(
        declared.a.nrows(),
        bounded,
        "one row per bounded coefficient, none per training row"
    );

    for ((row, col), &value) in prepared.design_derivative_exit.indexed_iter() {
        assert!(
            value >= 0.0,
            "derivative design entry ({row}, {col}) = {value:e} is negative"
        );
        if value != 0.0 {
            assert_eq!(
                bounds[col], 0.0,
                "column {col} carries derivative {value:e} at row {row} but is not bounded"
            );
        }
    }
}
