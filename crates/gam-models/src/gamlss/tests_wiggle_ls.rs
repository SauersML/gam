// Child module of `gamlss::tests` (see the `#[path]` declaration there):
// wiggle-family FD gates, binomial location-scale expected-info and release
// cells, NB dispersion convergence, and the zz2155 mode-geography probes.
// Split out of tests.rs for the source-file length budget only.
#![cfg(test)]

use super::*;
use gam_terms::basis::initializewiggle_knots_from_seed;

#[test]
pub(crate) fn nonwiggle_family_evaluate_returns_exact_newton_blockswhen_designs_are_present() {
    let n = 6usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).cos(),
                _ => unreachable!(),
            }
        }),
    ));
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };

    let beta_t = array![0.2, -0.15];
    let beta_ls = array![-0.1, 0.05];
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: threshold_design.matrixvectormultiply(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: log_sigma_design.matrixvectormultiply(&beta_ls),
        },
    ];

    let eval = family.evaluate(&states).expect("evaluate nonwiggle family");
    assert_eq!(eval.blockworking_sets.len(), 2);
    let joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let pt = beta_t.len();
    let pls = beta_ls.len();

    for (block_idx, (start, end)) in [(0usize, pt), (pt, pt + pls)].into_iter().enumerate() {
        let blockhessian = match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
                panic!("expected exact newton block")
            }
        };
        let joint_block = joint.slice(s![start..end, start..end]).to_owned();
        gam_test_support::assert_matrix_derivativefd(
            &joint_block,
            &blockhessian,
            1e-10,
            &format!("nonwiggle block {block_idx} principal block"),
        );
    }
}

#[test]
pub(crate) fn nonwiggle_family_joint_exacthessian_directional_derivative_matches_finite_difference()
{
    let n = 8usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                _ => unreachable!(),
            }
        }),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }),
    ));
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };

    let rebuild_states = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: threshold_design.matrixvectormultiply(beta_t),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: log_sigma_design.matrixvectormultiply(beta_ls),
            },
        ]
    };

    let beta_t = array![0.2, -0.1];
    let beta_ls = array![-0.15, 0.08];
    let states = rebuild_states(&beta_t, &beta_ls);
    let base_h = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let direction = array![0.2, 0.3, -0.15, 0.1];
    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("joint dH")
        .expect("expected joint exact dH");

    let eps = 1e-6;
    let dir_t = direction.slice(s![0..beta_t.len()]).to_owned();
    let dir_ls = direction.slice(s![beta_t.len()..]).to_owned();
    let states_plus = rebuild_states(&(&beta_t + &(eps * &dir_t)), &(&beta_ls + &(eps * &dir_ls)));
    let h_plus = family
        .exact_newton_joint_hessian(&states_plus)
        .expect("plus joint hessian")
        .expect("expected plus joint hessian");
    let fd = (h_plus - base_h) / eps;
    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 2e-3, "nonwiggle joint dH");
}

#[test]
pub(crate) fn nonwiggle_family_joint_exacthessiansecond_directional_derivative_matches_finite_difference()
 {
    let n = 8usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                _ => unreachable!(),
            }
        }),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }),
    ));
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };

    let rebuild_states = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: threshold_design.matrixvectormultiply(beta_t),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: log_sigma_design.matrixvectormultiply(beta_ls),
            },
        ]
    };

    let beta_t = array![0.2, -0.1];
    let beta_ls = array![-0.15, 0.08];
    let states = rebuild_states(&beta_t, &beta_ls);
    let direction_u = array![0.2, 0.3, -0.15, 0.1];
    let directionv = array![-0.05, 0.12, 0.08, -0.09];
    let analytic = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &direction_u, &directionv)
        .expect("joint d2H")
        .expect("expected joint exact d2H");

    let eps = 1e-6;
    let step_t = directionv.slice(s![0..beta_t.len()]).to_owned();
    let step_ls = directionv.slice(s![beta_t.len()..]).to_owned();
    let states_plus = rebuild_states(
        &(&beta_t + &(eps * &step_t)),
        &(&beta_ls + &(eps * &step_ls)),
    );
    let states_minus = rebuild_states(
        &(&beta_t - &(eps * &step_t)),
        &(&beta_ls - &(eps * &step_ls)),
    );
    let d_h_plus = family
        .exact_newton_joint_hessian_directional_derivative(&states_plus, &direction_u)
        .expect("joint dH plus")
        .expect("expected joint exact dH plus");
    let d_h_minus = family
        .exact_newton_joint_hessian_directional_derivative(&states_minus, &direction_u)
        .expect("joint dH minus")
        .expect("expected joint exact dH minus");
    let fd = (d_h_plus - d_h_minus) / (2.0 * eps);
    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 4e-3, "nonwiggle joint d2H");
}

#[test]
pub(crate) fn degeneratewiggle_seed_uses_broad_fallback_domain() {
    let q_seed = Array1::zeros(9);
    let degree = 3usize;
    let knots = initializewiggle_knots_from_seed(q_seed.view(), degree, 5)
        .expect("initialize degenerate wiggle knots");
    let bs_degree = monotone_wiggle_internal_degree(degree).expect("cubic wiggle degree") + 1;
    let domain_min = knots[bs_degree];
    let domain_max = knots[knots.len() - bs_degree - 1];
    assert!(
        domain_min <= -2.9,
        "unexpected left fallback boundary: {domain_min}"
    );
    assert!(
        domain_max >= 2.9,
        "unexpected right fallback boundary: {domain_max}"
    );
}

#[test]
pub(crate) fn split_wiggle_penalty_orders_uses_requested_order_one_as_primary() {
    let (primary, extras) =
        split_wiggle_penalty_orders(2, &[1, 2, 3, 3]).expect("valid derivative orders");
    assert_eq!(primary, 1);
    assert_eq!(extras, vec![2, 3]);
}

#[test]
pub(crate) fn selected_wiggle_function_penalties_keep_order_one() {
    let q_seed = Array1::linspace(-1.0, 1.0, 11);
    let degree = 3usize;
    let num_internal_knots = 5usize;
    let cfg = WiggleBlockConfig {
        degree,
        num_internal_knots,
        penalty_order: 1,
        double_penalty: false,
    };
    let selected =
        select_wiggle_basis_from_seed(q_seed.view(), &cfg, &[1, 3]).expect("selected wiggle basis");

    assert_eq!(selected.block.penalties.len(), 2);
    assert_eq!(selected.block.nullspace_dims, vec![0, 2]);
}

#[test]
pub(crate) fn binomial_location_scale_generative_matches_coremu() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let eta_t = Array1::from_vec(vec![0.8, -0.4, 0.2, -1.1, 0.0, 0.5, -0.7]);
    let eta_ls = Array1::from_vec(vec![-3.0, -1.2, -0.1, 0.3, 1.1, 2.0, 4.0]);

    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_t.clone(),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_ls.clone(),
        },
    ];
    let spec = family.generativespec(&states).expect("generative spec");
    let core = binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
        .expect("core");
    for i in 0..n {
        assert!(
            (spec.mean[i] - core.mu[i]).abs() < 1e-7,
            "mean mismatch at {i}: got {}, expected {}",
            spec.mean[i],
            core.mu[i]
        );
    }
}

/// The batched outer-gradient override on `BinomialLocationScaleFamily`
/// must produce a gradient that agrees with the central finite
/// difference of the same family's outer cost. This is the strongest
/// available correctness property: it does not depend on whether the
/// generic per-coordinate path is reachable in this build, only on the
/// scale-invariant identity `g_k = (V(ρ + h e_k) − V(ρ − h e_k)) / (2h)`
/// at converged β̂. Because the unified evaluator already routes
/// `ValueAndGradient` calls through the batched override (custom_family.rs
/// at the `batched_outer_gradient_terms` call site), this also pins the
/// wiring: any future regression that detaches the override from the
/// dispatcher will trip the FD check via stale (zero) gradients.
#[test]
pub(crate) fn binomial_location_scale_batched_gradient_matches_finite_difference() {
    use crate::custom_family::BlockwiseFitOptions;

    // 7-row, two-block intercept-only problem with a unit-Identity
    // penalty per block. Larger n risks PIRLS taking many iterations and
    // amplifying FD round-off; small p keeps the leverage-block sizes
    // (p_t = 1, p_ls = 1) tiny so the manual reference is trivial to
    // sanity-check.
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design),
        log_sigma_design: Some(base.log_sigma_design),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };

    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let rho = array![0.05, -0.15];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: 1e-10,
        outer_max_iter: 1,
        ..BlockwiseFitOptions::default()
    };

    let eval_outer = |rho: &Array1<f64>| {
        let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
        let result = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &options,
            rho,
            &test_design_hyper_layout(&derivative_blocks),
            None,
            gam_problem::EvalMode::ValueAndGradient,
        )
        .expect("objective+gradient at rho");
        (result.objective, result.gradient)
    };

    let (f0, g0) = eval_outer(&rho);
    assert!(f0.is_finite(), "outer cost must be finite at rho");
    assert_eq!(g0.len(), rho.len());

    let h = 1e-5;
    // Same noise-floor convention as the existing wiggle-family FD test
    // (custom_family.rs `outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active`):
    // below floor `EPS·|cost|/h`, the FD estimator can't resolve the
    // true gradient.
    let cost_magnitude = f0.abs().max(1.0);
    let noise_floor = (10.0 * f64::EPSILON * cost_magnitude / h).max(1e-9);

    for k in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[k] += h;
        rho_m[k] -= h;
        let (fp, _) = eval_outer(&rho_p);
        let (fm, _) = eval_outer(&rho_m);
        let gfd = (fp - fm) / (2.0 * h);
        let both_in_noise = g0[k].abs() < noise_floor && gfd.abs() < noise_floor;
        if !both_in_noise {
            let abs_err = (g0[k] - gfd).abs();
            let rel_err = abs_err / gfd.abs().max(g0[k].abs()).max(1e-12);
            assert!(
                rel_err < 1e-3 || abs_err < 1e-6,
                "batched gradient mismatch at coord {k}: \
                     batched={:.6e}, fd={:.6e}, abs_err={:.3e}, rel_err={:.3e}",
                g0[k],
                gfd,
                abs_err,
                rel_err,
            );
        }
    }
}

pub(crate) fn binomial_mean_wiggle_operator_fixture() -> (
    BinomialMeanWiggleFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
    Array2<f64>,
) {
    let x_eta = array![
        [1.0, -0.9],
        [1.0, -0.45],
        [1.0, -0.1],
        [1.0, 0.2],
        [1.0, 0.55],
        [1.0, 0.9],
    ];
    let beta_eta = array![-0.15, 0.7];
    let eta = x_eta.dot(&beta_eta);
    let degree = 3usize;
    let knots = initializewiggle_knots_from_seed(eta.view(), degree, 4).expect("mean-wiggle knots");
    let family = BinomialMeanWiggleFamily {
        y: array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        weights: array![1.0, 0.8, 1.2, 1.0, 0.7, 1.1],
        link_kind: InverseLink::Standard(StandardLink::Logit),
        wiggle_knots: knots,
        wiggle_degree: degree,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        frozen_warp_design: None,
        continuation: false,
    };
    let basis = family.wiggle_design(eta.view()).expect("wiggle basis");
    let beta_w = Array1::from_iter((0..basis.ncols()).map(|j| 0.015 * (j as f64 + 1.0)));
    let etaw = basis.dot(&beta_w);
    let states = vec![
        ParameterBlockState {
            beta: beta_eta,
            eta: eta.clone(),
        },
        ParameterBlockState {
            beta: beta_w,
            eta: etaw,
        },
    ];
    let specs = vec![
        ParameterBlockSpec {
            name: "eta".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_eta.clone())),
            offset: Array1::zeros(eta.len()),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(basis)),
            offset: Array1::zeros(eta.len()),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    (family, states, specs, x_eta)
}


#[test]
pub(crate) fn binomial_location_scale_expected_info_derivatives_match_finite_difference() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    // gam#1020: the expected-information override must disarm the
    // observed-Hessian "Jeffreys skippable" matvec pre-checks.
    assert!(!family.joint_jeffreys_information_matches_observed_hessian());
    let beta_t = Array1::from_iter((0..x_t.ncols()).map(|j| 0.12 - 0.03 * j as f64));
    let beta_ls = Array1::from_iter((0..x_ls.ncols()).map(|j| -0.08 + 0.02 * j as f64));
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: x_t.dot(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: x_ls.dot(&beta_ls),
        },
    ];
    let total = beta_t.len() + beta_ls.len();
    let u = Array1::from_iter((0..total).map(|j| 0.03 * (j as f64 + 0.4).sin()));
    let v = Array1::from_iter((0..total).map(|j| -0.02 * (j as f64 + 0.7).cos()));

    let info = |direction: &Array1<f64>, scale: f64| {
        let mut next = states.clone();
        let pt = beta_t.len();
        next[BinomialLocationScaleFamily::BLOCK_T]
            .beta
            .scaled_add(scale, &direction.slice(s![0..pt]));
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
            .beta
            .scaled_add(scale, &direction.slice(s![pt..total]));
        next[BinomialLocationScaleFamily::BLOCK_T].eta =
            x_t.dot(&next[BinomialLocationScaleFamily::BLOCK_T].beta);
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
            x_ls.dot(&next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
        family
            .joint_jeffreys_information_with_specs(&next, &specs)
            .expect("expected information")
            .expect("expected information available")
    };

    let h0 = family
        .joint_jeffreys_information_with_specs(&states, &specs)
        .expect("expected information")
        .expect("expected information available");
    assert_close_matrix(&info(&u, 0.0), &h0, 1e-12, "expected information value");

    let eps = 1e-5;
    let hp = info(&u, eps);
    let hm = info(&u, -eps);
    let fd_first = (&hp - &hm) / (2.0 * eps);
    let analytic_first = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &u)
        .expect("expected dI")
        .expect("expected dI available");
    assert_close_matrix(&analytic_first, &fd_first, 1e-7, "expected dI");

    let mut states_plus = states.clone();
    let pt = beta_t.len();
    states_plus[BinomialLocationScaleFamily::BLOCK_T]
        .beta
        .scaled_add(eps, &v.slice(s![0..pt]));
    states_plus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .scaled_add(eps, &v.slice(s![pt..total]));
    states_plus[BinomialLocationScaleFamily::BLOCK_T].eta =
        x_t.dot(&states_plus[BinomialLocationScaleFamily::BLOCK_T].beta);
    states_plus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
        x_ls.dot(&states_plus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
    let d_plus = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states_plus, &specs, &u)
        .expect("expected dI plus")
        .expect("expected dI plus available");

    let mut states_minus = states.clone();
    states_minus[BinomialLocationScaleFamily::BLOCK_T]
        .beta
        .scaled_add(-eps, &v.slice(s![0..pt]));
    states_minus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .scaled_add(-eps, &v.slice(s![pt..total]));
    states_minus[BinomialLocationScaleFamily::BLOCK_T].eta =
        x_t.dot(&states_minus[BinomialLocationScaleFamily::BLOCK_T].beta);
    states_minus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
        x_ls.dot(&states_minus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
    let d_minus = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states_minus, &specs, &u)
        .expect("expected dI minus")
        .expect("expected dI minus available");
    let fd_second = (&d_plus - &d_minus) / (2.0 * eps);
    let analytic_second = family
        .joint_jeffreys_information_second_directional_derivative_with_specs(
            &states, &specs, &u, &v,
        )
        .expect("expected d2I")
        .expect("expected d2I available");
    assert_close_matrix(&analytic_second, &fd_second, 1e-7, "expected d2I");
}

/// gam#2922: the explicit-ψ Jeffreys terms used to take the observed `hessian_psi` as the motion
/// of the Jeffreys information, and this family's information is the expected one. On a design
/// hyperparameter moving the threshold design the observed motion fails a Richardson difference of
/// the expected information along that motion, on the bar the Bernoulli marginal-slope family's
/// expected-information pins hold, and the family supplies no motion of its own, so the explicit-ψ
/// Jeffreys terms refuse it instead of substituting the observed one.
#[test]
pub(crate) fn binomial_location_scale_observed_psi_motion_is_not_the_expected_information_motion_2922()
{
    let base = binomial_location_scale_base_fixture();
    let n = base.n;
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design")
        .to_owned();
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design")
        .to_owned();
    let beta_t = Array1::from_iter((0..x_t.ncols()).map(|j| 0.35 - 0.03 * j as f64));
    let beta_ls = Array1::from_iter((0..x_ls.ncols()).map(|j| -0.15 + 0.02 * j as f64));
    let x_psi = Array2::from_shape_fn((n, x_t.ncols()), |(i, j)| 0.4 * ((i + 3 * j) as f64 * 0.53).sin());
    let derivative_blocks = vec![
        vec![CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi: x_psi.clone(),
            s_psi: Array2::zeros((x_t.ncols(), x_t.ncols())),
            s_psi_components: None,
            s_psi_penalty_components: None,
            x_psi_psi: None,
            s_psi_psi: None,
            s_psi_psi_components: None,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        }],
        Vec::new(),
    ];
    let states_at = |design_t: &Array2<f64>| {
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: design_t.dot(&beta_t),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: x_ls.dot(&beta_ls),
            },
        ]
    };
    let states = states_at(&x_t);
    let hyper_layout = crate::custom_family::CustomFamilyHyperLayout::new(
        derivative_blocks.clone(),
        Vec::new(),
        Array1::zeros(1),
    )
    .expect("design hyper layout");
    assert!(!family.joint_jeffreys_information_matches_observed_hessian());
    assert!(
        matches!(
            family
                .joint_jeffreys_information_psi_derivative(&states, &specs, &hyper_layout, 0)
                .expect("psi derivative hook"),
            crate::custom_family::JeffreysInformationMotion::Unpublished { .. }
        ),
        "the family supplies no expected-information psi motion, so the explicit-psi Jeffreys terms refuse it"
    );
    let observed = family
        .exact_newton_joint_psi_terms_for_specs(&states, &specs, &derivative_blocks, 0)
        .expect("observed psi terms")
        .expect("the design axis publishes psi terms");
    let observed = match observed.hessian_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(x_t.ncols() + x_ls.ncols())),
        None => observed.hessian_psi,
    };
    let information_at = |t: f64| {
        let moved = &x_t + &(&x_psi * t);
        let mut moved_specs = specs.clone();
        moved_specs[BinomialLocationScaleFamily::BLOCK_T].design =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(moved.clone()));
        family
            .joint_jeffreys_information_with_specs(&states_at(&moved), &moved_specs)
            .expect("expected information")
            .expect("expected information available")
    };
    let step = 1e-3;
    let coarse = (information_at(step) - information_at(-step)) / (2.0 * step);
    let fine = (information_at(0.5 * step) - information_at(-0.5 * step)) / step;
    let scale = observed
        .iter()
        .chain(fine.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        scale > 1e-6,
        "the psi motion carries no curvature on this fixture ({scale:.3e})"
    );
    let misses = observed
        .indexed_iter()
        .filter(|&((row, column), &want)| {
            let value = (4.0 * fine[[row, column]] - coarse[[row, column]]) / 3.0;
            let uncertainty = (fine[[row, column]] - coarse[[row, column]]).abs() / 3.0;
            let denominator = scale.max(want.abs()).max(value.abs());
            !((want - value).abs() <= 1e-5 * denominator + 4.0 * uncertainty)
        })
        .count();
    assert!(
        misses > 0,
        "the observed psi motion matched the expected information's difference, so this fixture shows no gap"
    );
}

/// Layer-5 deliverable (gam#979 / gam#1020): the Tier-B Jeffreys term built
/// on the EXPECTED Fisher information must NOT reward probit saturation,
/// whereas the OBSERVED-information Jeffreys term DOES — which is the long
/// quasi-flat descent valley that made the constrained-wiggle inner solve
/// walk `|β|→∞`.
///
/// Mechanism. For probit `q ↦ Φ(q)`, drive the threshold predictor `η_t`
/// (hence `q`) into saturation. The OBSERVED per-row curvature
/// `−∂²ℓ/∂q² = w·(z·q′ + …)` carries the misclassification term that GROWS
/// like `q²` on rows the saturated mean gets wrong, so `½log det H_obs`
/// climbs without bound — Φ_obs rewards walking toward saturation. The
/// EXPECTED Fisher weight `w^F = φ(q)²/(p(1−p))` DECAYS as `q→±∞` (the
/// Gaussian pdf `φ` kills the numerator faster than `p(1−p)→0` shrinks the
/// denominator), so `½log det H_exp` is bounded above — Φ_exp has no valley.
///
/// The assertion: across a saturation sweep, Φ on the expected information
/// stays bounded (and ultimately decreases), while Φ on the observed
/// information grows past it — the exact sign that the expected-information
/// hook removes the gam#979 saturation reward. Both Φ are evaluated through
/// the SAME `joint_jeffreys_term` value path on the FULL identifiable span
/// (`Z_J = I`), differing only in the information matrix consumed.
#[test]
pub(crate) fn expected_info_jeffreys_does_not_reward_probit_saturation() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    let total = x_t.ncols() + x_ls.ncols();
    let z = Array2::<f64>::eye(total);

    // Φ on a supplied information matrix at threshold β_t (log-σ fixed at 0,
    // so σ = 1 and q = -β_t scans the probit argument across saturation).
    let phi_on = |info: &Array2<f64>| -> f64 {
        let (phi, _grad, _hphi) =
            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
                info.view(),
                z.view(),
                |_: &Array1<f64>| Ok(None),
            )
            .expect("jeffreys term value");
        phi
    };
    let states_at = |beta_t: f64| -> Vec<ParameterBlockState> {
        let bt = Array1::from_elem(x_t.ncols(), beta_t);
        let bls = Array1::zeros(x_ls.ncols());
        vec![
            ParameterBlockState {
                eta: x_t.dot(&bt),
                beta: bt,
            },
            ParameterBlockState {
                eta: x_ls.dot(&bls),
                beta: bls,
            },
        ]
    };

    // Sweep the threshold into deep probit saturation.
    let betas = [1.0_f64, 2.0, 3.0, 4.0, 6.0, 8.0];
    let mut phi_obs = Vec::with_capacity(betas.len());
    let mut phi_exp = Vec::with_capacity(betas.len());
    for &b in betas.iter() {
        let states = states_at(b);
        let obs = family
            .exact_newton_joint_hessian_with_specs(&states, &specs)
            .expect("observed hessian")
            .expect("observed hessian available");
        let exp = family
            .joint_jeffreys_information_with_specs(&states, &specs)
            .expect("expected information")
            .expect("expected information available");
        phi_obs.push(phi_on(&obs));
        phi_exp.push(phi_on(&exp));
    }

    // (1) The expected-information Jeffreys term is BOUNDED across the sweep
    // (no runaway reward); concretely it does not increase from its
    // mild-saturation value to its deepest-saturation value — the valley is
    // gone (decaying expected information).
    let exp_first = phi_exp[0];
    let exp_last = *phi_exp.last().expect("nonempty");
    assert!(
        exp_last <= exp_first + 1e-9,
        "expected-info Jeffreys Φ rewarded saturation: Φ_exp went {exp_first:.6} → {exp_last:.6} \
             across β_t {:?} (full sweep {phi_exp:?})",
        betas
    );

    // (2) The observed-information Jeffreys term, in contrast, GROWS into
    // saturation and overtakes the expected one — the genuine valley the
    // layer-5 hook exists to remove. This makes the test a real
    // discriminator: it fails if the family silently reverts to observed
    // information.
    let obs_last = *phi_obs.last().expect("nonempty");
    assert!(
        obs_last > exp_last + 0.5,
        "observed-info Jeffreys Φ did not exhibit the saturation valley the \
             expected-info hook removes: Φ_obs_last={obs_last:.6} vs Φ_exp_last={exp_last:.6} \
             (Φ_obs sweep {phi_obs:?}, Φ_exp sweep {phi_exp:?})"
    );
}

#[test]
pub(crate) fn binomial_location_scale_expected_info_contracted_trace_matches_second_directional() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    let beta_t = Array1::from_iter((0..x_t.ncols()).map(|j| 0.11 - 0.02 * j as f64));
    let beta_ls = Array1::from_iter((0..x_ls.ncols()).map(|j| -0.07 + 0.03 * j as f64));
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: x_t.dot(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: x_ls.dot(&beta_ls),
        },
    ];
    let total = beta_t.len() + beta_ls.len();
    let weight = Array2::from_shape_fn((total, total), |(i, j)| {
        0.03 * ((i + 2 * j + 1) as f64).sin()
    });
    let contracted = family
        .joint_jeffreys_information_contracted_trace_hessian_with_specs(&states, &specs, &weight)
        .expect("contracted trace")
        .expect("contracted trace present");
    let mut expected = Array2::<f64>::zeros((total, total));
    for a in 0..total {
        let mut axis_a = Array1::<f64>::zeros(total);
        axis_a[a] = 1.0;
        for b in a..total {
            let mut axis_b = Array1::<f64>::zeros(total);
            axis_b[b] = 1.0;
            let second = family
                .joint_jeffreys_information_second_directional_derivative_with_specs(
                    &states, &specs, &axis_a, &axis_b,
                )
                .expect("expected d2I")
                .expect("expected d2I present");
            let mut trace = 0.0;
            for row in 0..total {
                for col in 0..total {
                    trace += weight[[row, col]] * second[[col, row]];
                }
            }
            expected[[a, b]] = trace;
            expected[[b, a]] = trace;
        }
    }
    assert_close_matrix(&contracted, &expected, 1e-9, "expected contracted trace");
}

#[test]
pub(crate) fn binomial_location_scale_expected_hphi_drift_matches_finite_difference() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    let beta_t = Array1::from_iter((0..x_t.ncols()).map(|j| 0.09 - 0.02 * j as f64));
    let beta_ls = Array1::from_iter((0..x_ls.ncols()).map(|j| -0.06 + 0.04 * j as f64));
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: x_t.dot(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: x_ls.dot(&beta_ls),
        },
    ];
    let total = beta_t.len() + beta_ls.len();
    // IDENTIFIABLE-SPAN Jeffreys subspace `Z_J`. The binomial location-scale map
    // `q = −η_t/σ` carries an EXACT threshold↔scale gauge degeneracy: the
    // direction `(δη_t = η_t, δη_ls = 1)` gives `q̇ = q_t·η_t + q_ls = −η_t/σ +
    // η_t/σ = 0`, so the per-row q-gradient — hence the whole expected Fisher
    // information `I(β)` — is rank-deficient by exactly one along this gauge
    // axis. On the constant-design fixture every row is proportional, so `I` is
    // rank 1 and its smallest eigenvalue is structurally ZERO. Differencing the
    // floored-pseudo-inverse `H_Φ` over the FULL span (`Z = I`) therefore
    // central-differences a quantity whose near-zero-eigenvalue eigenvector is
    // arbitrary up to numerical noise: the FD is meaningless (it swings by
    // O(1/floor) with the eps choice) even though the analytic drift is exact.
    // Production never runs the Jeffreys term on the raw gauge-degenerate span;
    // it reduces to the identifiable coordinates first. We mirror that here by
    // taking `Z_J` to be the eigenvectors of the base information with
    // non-negligible eigenvalue, so the reduced `H_Φ` is well-conditioned and
    // its central difference converges to the analytic directional derivative at
    // the 1e-7 bar. (The dropped gauge axis carries no identifiable curvature, so
    // restricting to it loses nothing the objective ever uses.)
    let z = {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;
        let base_info = family
            .joint_jeffreys_information_with_specs(&states, &specs)
            .expect("base expected info")
            .expect("base expected info present");
        let mut sym = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                sym[[i, j]] = 0.5 * (base_info[[i, j]] + base_info[[j, i]]);
            }
        }
        let (evals, evecs) = sym.eigh(Side::Lower).expect("base info eigendecomposition");
        let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
        // Keep the identifiable directions (curvature ≥ a tiny fraction of the
        // dominant eigenvalue); drop the structural gauge null space.
        let keep: Vec<usize> = (0..total)
            .filter(|&i| evals[i] > lambda_max * 1e-8)
            .collect();
        assert!(
            !keep.is_empty(),
            "base information must have an identifiable direction"
        );
        let mut z = Array2::<f64>::zeros((total, keep.len()));
        for (col, &i) in keep.iter().enumerate() {
            z.column_mut(col).assign(&evecs.column(i));
        }
        z
    };
    let direction = Array1::from_shape_fn(total, |i| 0.03 * ((i + 1) as f64).sin());
    let perturb = |scale: f64| {
        let mut next = states.clone();
        let pt = beta_t.len();
        next[BinomialLocationScaleFamily::BLOCK_T]
            .beta
            .scaled_add(scale, &direction.slice(s![0..pt]));
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
            .beta
            .scaled_add(scale, &direction.slice(s![pt..total]));
        next[BinomialLocationScaleFamily::BLOCK_T].eta =
            x_t.dot(&next[BinomialLocationScaleFamily::BLOCK_T].beta);
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
            x_ls.dot(&next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
        next
    };
    let hphi_at = |block_states: &[ParameterBlockState]| {
        let info = family
            .joint_jeffreys_information_with_specs(block_states, &specs)
            .expect("expected info")
            .expect("expected info present");
        let (_phi, _grad, hphi) =
            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
                info.view(),
                z.view(),
                |axis: &Array1<f64>| {
                    family.joint_jeffreys_information_directional_derivative_with_specs(
                        block_states,
                        &specs,
                        axis,
                    )
                },
            )
            .expect("hphi term");
        hphi
    };
    let eps = 1e-5;
    let h_plus = hphi_at(&perturb(eps));
    let h_minus = hphi_at(&perturb(-eps));
    let fd = (&h_plus - &h_minus) / (2.0 * eps);
    let info = family
        .joint_jeffreys_information_with_specs(&states, &specs)
        .expect("expected info")
        .expect("expected info present");
    // Mode-response drift `D_β H_Φ[δ]` via the production-level perturbation core
    // (the `joint_jeffreys_hphi_directional_derivative` oracle is a thin wrapper
    // over this: `Hdot[δ]` once, then the perturbation derivative). Calling the
    // core directly keeps the oracle private to its own `#[cfg(test)]` module.
    let pert_h = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &direction)
        .expect("Hdot[delta]")
        .expect("Hdot[delta] present");
    let analytic =
        gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_hphi_perturbation_derivative(
            info.view(),
            z.view(),
            |axis: &Array1<f64>| {
                family.joint_jeffreys_information_directional_derivative_with_specs(
                    &states, &specs, axis,
                )
            },
            &pert_h,
            |axis: &Array1<f64>| {
                family.joint_jeffreys_information_second_directional_derivative_with_specs(
                    &states, &specs, &direction, axis,
                )
            },
        )
        .expect("hphi drift");
    assert_close_matrix(&analytic, &fd, 1e-7, "expected H_phi drift");
}

#[test]
pub(crate) fn binomial_mean_wiggle_hessian_operators_match_dense_derivatives() {
    let (family, states, specs, x_eta) = binomial_mean_wiggle_operator_fixture();
    let p_eta = x_eta.ncols();
    let pw = states[BinomialMeanWiggleFamily::BLOCK_WIGGLE].beta.len();
    let total = p_eta + pw;
    let dir_u = Array1::from_iter((0..total).map(|j| 0.03 * (j as f64 + 1.0).sin()));
    let dir_v = Array1::from_iter((0..total).map(|j| -0.02 * (j as f64 + 0.5).cos()));

    let dense_h = family
        .exact_newton_joint_hessian_with_specs(&states, &specs)
        .expect("dense H")
        .expect("dense H available");
    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace")
        .expect("workspace available");
    let h_columns = Array2::from_shape_fn((total, total), |(i, j)| if i == j { 1.0 } else { 0.0 });
    let op_h = gam_problem::HyperOperator::mul_mat(
        family
            .bmw_static_hessian_operator(&states, Arc::new(x_eta.clone()))
            .expect("static op")
            .as_ref(),
        &h_columns,
    );
    assert_close_matrix(&op_h, &dense_h, 1e-10, "static H operator");
    let hv = workspace
        .hessian_matvec(&dir_u)
        .expect("workspace HVP")
        .expect("workspace HVP available");
    let hv_dense = dense_h.dot(&dir_u);
    let hv_err = (&hv - &hv_dense).mapv(f64::abs).sum();
    assert!(hv_err < 1e-10, "workspace HVP mismatch {hv_err:.3e}");

    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative_with_specs(&states, &specs, &dir_u)
        .expect("dense dH")
        .expect("dense dH available");
    let op_dh = workspace
        .directional_derivative_operator(&dir_u)
        .expect("dH operator")
        .expect("dH operator available")
        .to_dense();
    assert_close_matrix(&op_dh, &dense_dh, 1e-10, "directional dH operator");

    let dense_d2h = family
        .exact_newton_joint_hessian_second_directional_derivative_with_specs(
            &states, &specs, &dir_u, &dir_v,
        )
        .expect("dense d2H")
        .expect("dense d2H available");
    let op_d2h = workspace
        .second_directional_derivative_operator(&dir_u, &dir_v)
        .expect("d2H operator")
        .expect("d2H operator available")
        .to_dense();
    assert_close_matrix(
        &op_d2h,
        &dense_d2h,
        1e-10,
        "second directional d2H operator",
    );
}

#[test]
pub(crate) fn binomial_mean_wiggle_planner_keeps_second_order_at_large_n() {
    let n = 50_001usize;
    let family = BinomialMeanWiggleFamily {
        y: Array1::zeros(n),
        weights: Array1::ones(n),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        wiggle_knots: initializewiggle_knots_from_seed(Array1::linspace(-1.0, 1.0, 9).view(), 3, 4)
            .expect("large-n knots"),
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        frozen_warp_design: None,
        continuation: false,
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "eta".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, 2)),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, 34)),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    assert!(family.inner_coefficient_hessian_hvp_available(&specs));
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        crate::custom_family::ExactOuterDerivativeOrder::Second
    );
}

// ── #1606: NB location-scale (GAMLSS-style joint mean/dispersion) inner solve ──
//
// Regression for gam#1606: a negative-binomial location-scale fit
// (`family="nb"` + a dispersion smooth) ABORTED at fit time with an
// `IntegrationError` on well-posed heteroscedastic count data, while every
// sibling path (plain NB, Gaussian-LS, Gamma-LS) fit the same design. Root
// cause: the NB dispersion (log-θ) block assembled its IRLS curvature from the
// per-row OBSERVED Hessian channel `−∂²ℓ/∂θ²`, which carries the row-specific
// `ψ′(θ+y)` term and goes NEGATIVE for every row whose count sits below its
// current fitted precision. Replacing each negative row by an arbitrary
// epsilon then divides the exact score by ~0 in the
// working response, producing O(1e10) IRLS targets that explode the dispersion
// step and stall the inner block-cyclic solve, whose non-convergence is then
// escalated to a hard error. The fix switches the dispersion curvature to the
// EXPECTED (Fisher) information `ψ′(θ)−ψ′(θ+μ)−1/θ+1/(θ+μ) > 0` (Fisher
// scoring; the working RESPONSE still carries the exact score, so the penalized
// optimum is unchanged — only the inner conditioning improves), matching the
// mean block, which always used its closed-form expected info.

// Direct root-cause regression (the fail-before / pass-after gate): at a
// heteroscedastic iterate whose fitted precision sits ABOVE the data's true
// overdispersion (the regime the inner solve traverses), the NB dispersion
// working set must stay well-conditioned. With the pre-fix OBSERVED curvature
// the per-row information is negative for these rows, gets epsilon-clamped,
// and the working response `disp_response` blows up
// to O(1e9)+ (the exact score divided by ~0). With the EXPECTED (Fisher)
// curvature the response stays O(1) and the per-row IRLS weight reflects
// genuine positive curvature. This asserts the bounded, well-conditioned
// behaviour — it FAILS on the observed-curvature code (huge |disp_response|)
// and PASSES on the Fisher-curvature fix.
#[test]
fn nb_dispersion_working_set_stays_bounded_above_optimum_1606() {
    use super::super::dispersion_family::dispersion_row_kernel;

    // Overdispersed rows (true θ small, large counts) evaluated at a high fitted
    // precision η_d = ln(8): there μ²/θ_true ≫ μ, so y ≫ μ for many rows while
    // the model currently believes the precision is large — exactly where
    // `−∂²ℓ/∂θ²` goes negative.
    let mu = 20.0_f64;
    let eta_mu = mu.ln();
    let eta_d = 8.0_f64.ln(); // fitted θ = 8, well above the true overdispersion
    // A spread of counts straddling μ, including the small/zero counts that
    // drive the observed information negative.
    let counts = [0.0_f64, 2.0, 4.0, 6.0, 8.0, 22.0, 27.0, 40.0, 63.0, 95.0];
    let mut saw_overdispersed_row = false;
    for &yi in &counts {
        let row = dispersion_row_kernel(
            DispersionFamilyKind::NegativeBinomial,
            yi,
            eta_mu,
            eta_d,
            1.0,
        );
        // The working response is `η_d + score/(θ·info)`. With the Fisher
        // information it is O(1); with the floored observed information it is
        // O(1e9)+. Pin a generous-but-decisive bound: anything below 1e6 is the
        // well-conditioned Fisher path, anything above is the broken floored
        // observed path (the real failures are ~1e10).
        assert!(
            row.disp_response.is_finite() && row.disp_response.abs() < 1.0e6,
            "NB dispersion working response must stay bounded at an above-optimum \
             iterate (gam#1606): y={yi}, disp_response={:.6e} (an O(1e9)+ value is the \
             pre-fix floored-observed-curvature blow-up)",
            row.disp_response,
        );
        // The per-row IRLS weight must be a genuine positive curvature, not the
        // ~0 floor that the negative observed information collapses to.
        assert!(
            row.disp_weight.is_finite() && row.disp_weight >= 0.0,
            "NB dispersion working weight must be a finite non-negative curvature: \
             y={yi}, disp_weight={:.6e}",
            row.disp_weight,
        );
        if yi < mu {
            saw_overdispersed_row = true;
            // These are precisely the rows whose OBSERVED information is negative
            // (count below fitted precision); the Fisher weight keeps them at a
            // strictly positive, finite curvature.
            assert!(
                row.disp_weight > 0.0,
                "below-fitted-precision rows must still carry positive Fisher \
                 curvature: y={yi}, disp_weight={:.6e}",
                row.disp_weight,
            );
        }
    }
    assert!(
        saw_overdispersed_row,
        "fixture must include rows below the fitted precision (the negative-observed-info regime)"
    );
}

// End-to-end contract check: the documented NB location-scale fit drives the
// two-block custom-family inner solve through the public fixed-log-λ entry
// point (`fit_custom_family_fixed_log_lambdas`, which runs `inner_blockwise_fit`
// and returns `Err` when the inner solve fails to converge — the same
// non-convergence the profile-objective evaluator escalates), and must converge
// and predict finite, strictly positive per-row means. The Python repro
// (`bug_hunt_nb_location_scale_inner_solve_abort_test`) cannot run under the
// build.rs author-guard deadlock, so this Rust-level fit stands in for it.
#[test]
fn nb_location_scale_inner_solve_converges_on_heteroscedastic_counts() {
    use super::super::dispersion_family::{DispersionFamilyKind, DispersionGlmLocationScaleFamily};
    use crate::custom_family::fit_custom_family_fixed_log_lambdas;

    // Deterministic LCG so the synthetic data (and thread-independent inner
    // path) is byte-reproducible — the issue noted order/thread-state-dependent
    // flips, so the fixture must not depend on any global RNG state.
    struct Lcg(u64);
    impl Lcg {
        fn next_u01(&mut self) -> f64 {
            // Numerical Recipes LCG constants.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // top 53 bits → [0,1)
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn next_gamma_shape_ge1(&mut self, shape: f64) -> f64 {
            // Marsaglia–Tsang for shape ≥ 1 (we only call with shape ≥ 1).
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                // crude standard normal via sum of 12 uniforms − 6
                let mut z = -6.0;
                for _ in 0..12 {
                    z += self.next_u01();
                }
                let v = (1.0 + c * z).powi(3);
                if v <= 0.0 {
                    continue;
                }
                let u = self.next_u01();
                if u.ln() < 0.5 * z * z + d - d * v + d * (v).ln() {
                    return d * v;
                }
            }
        }
        fn next_poisson(&mut self, lambda: f64) -> f64 {
            // Knuth, fine for the moderate λ here.
            let l = (-lambda).exp();
            let mut k = 0.0;
            let mut p = 1.0;
            loop {
                k += 1.0;
                p *= self.next_u01();
                if p <= l {
                    return k - 1.0;
                }
            }
        }
        fn next_nb(&mut self, mu: f64, theta: f64) -> f64 {
            // Gamma–Poisson mixture: λ ~ Gamma(theta, mu/theta), Y ~ Pois(λ).
            let lam = if theta >= 1.0 {
                self.next_gamma_shape_ge1(theta) * (mu / theta)
            } else {
                // boost shape by 1 then scale down (Stuart's method)
                let g = self.next_gamma_shape_ge1(theta + 1.0);
                let u = self.next_u01().max(1e-300);
                g * u.powf(1.0 / theta) * (mu / theta)
            };
            self.next_poisson(lam.max(1e-9))
        }
    }

    let n = 600usize;
    let p = 6usize;
    // The mean and dispersion smooths ride on TWO DISTINCT covariates (x for the
    // mean, z for the dispersion). Each channel's design is a sum-to-zero,
    // column-orthonormal polynomial basis in its OWN covariate built from the
    // monomials t¹..tᵖ (NO constant column): the per-channel level is carried by
    // a constant `offset`, exactly as a centered production `s(x)` smooth plus a
    // gauge-fixed intercept. Dropping the constant column is what keeps the flat
    // pre-fit identifiability audit happy — a single-channel custom family sees
    // both channels' designs as ordinary columns, and two identical all-ones
    // intercept columns across blocks would alias (overlap 1.0) and fail the
    // audit. With no constant column and two different covariates, the
    // concatenated [mean | log_precision] joint design is full-rank and
    // alias-free, while the constant offsets still let each η reach its level.
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let zs: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.6180339887) % 1.0) // golden-ratio low-discrepancy spread
        .collect();
    // Modified Gram–Schmidt over the monomials t¹, t², … (skip t⁰), each column
    // first centered to mean-zero so it is orthogonal to the constant direction
    // too. Every resulting column is a distinct, mutually-orthonormal,
    // sum-to-zero direction; none is the constant, and across two different
    // covariates none coincides cross-block.
    let build_design = |t: &[f64]| -> Array2<f64> {
        let mut cols: Vec<Array1<f64>> = Vec::with_capacity(p);
        for j in 0..p {
            // monomial t^(j+1), centered to mean zero.
            let mut v = Array1::from_shape_fn(n, |i| t[i].powi((j + 1) as i32));
            let mean = v.sum() / (n as f64);
            v.mapv_inplace(|e| e - mean);
            for c in &cols {
                let proj = v.dot(c);
                v.scaled_add(-proj, c);
            }
            let nrm = v.dot(&v).sqrt().max(1e-12);
            v.mapv_inplace(|e| e / nrm);
            cols.push(v);
        }
        let mut d = Array2::<f64>::zeros((n, p));
        for (j, c) in cols.iter().enumerate() {
            d.column_mut(j).assign(c);
        }
        d
    };
    let mean_x = build_design(&xs);
    let disp_x = build_design(&zs);
    // Per-channel constant level carried by the offset (centered smooth + level).
    let mean_offset = Array1::from_elem(n, 1.4_f64);
    let disp_offset = Array1::from_elem(n, 0.5_f64);

    // True surfaces: mean μ(x) = exp(η_μ) sweeps a moderate count range, and the
    // dispersion log θ(z) sweeps from high overdispersion (small θ) to near-
    // Poisson (large θ) — the heteroscedastic regime that drives the dispersion
    // block's η_d across the negative-observed-info zone.
    let eta_mu_true: Vec<f64> = xs.iter().map(|&x| 1.4 + 1.1 * (2.2 * x).sin()).collect();
    let log_theta_true: Vec<f64> = zs.iter().map(|&z| -1.2 + 3.4 * z).collect();

    let mut rng = Lcg(0x1606_2024_dead_beef);
    let y = Array1::from_shape_fn(n, |i| {
        let mu = eta_mu_true[i].exp();
        let theta = log_theta_true[i].exp();
        rng.next_nb(mu, theta)
    });
    // Sanity: the response must be a non-degenerate count vector.
    assert!(
        y.iter().any(|&v| v > 0.0) && y.iter().all(|&v| v >= 0.0 && v.fract() == 0.0),
        "synthetic NB response must be non-negative integer counts with positive mass"
    );

    let weights = Array1::from_elem(n, 1.0);
    let family = DispersionGlmLocationScaleFamily {
        kind: DispersionFamilyKind::NegativeBinomial,
        y: y.clone(),
        weights,
        jeffreys_armed: true,
    };

    // Each block: a wiggliness penalty that shrinks the higher-order
    // (orthonormal) polynomial columns, leaving the two lowest-order columns
    // (the 2-dim penalty nullspace) free. This gives the smooth genuine
    // shrinkage at a moderate fixed smoothing parameter, mirroring the `s(x)`
    // production path.
    let make_penalty = || {
        let mut pmat = Array2::<f64>::zeros((p, p));
        for j in 2..p {
            // Increasing penalty weight on higher-order columns.
            pmat[[j, j]] = (j as f64 - 1.0).powi(2);
        }
        PenaltyMatrix::Dense(pmat)
    };

    let mk_spec = |name: &str, design: Array2<f64>, offset: Array1<f64>| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
        offset,
        penalties: vec![make_penalty()],
        // Penalty nullspace = the two lowest-order columns = 2 unpenalized dirs.
        nullspace_dims: vec![2],
        initial_log_lambdas: Array1::from_elem(1, (0.5_f64).ln()),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![
        mk_spec("mean", mean_x, mean_offset),
        mk_spec("log_precision", disp_x, disp_offset),
    ];

    let options = BlockwiseFitOptions::default();

    // The fixed-log-λ fit runs `inner_blockwise_fit` and returns `Err` exactly
    // when the inner solve fails to converge — the non-convergence the profile
    // objective escalates to the fatal abort. Before the fix this returns
    // `Err(Optimization{ "...inner solve did not converge..." })`.
    let result = fit_custom_family_fixed_log_lambdas(&family, &specs, &options, None);
    let fit = result.unwrap_or_else(|e| {
        panic!(
            "NB location-scale inner solve must converge on heteroscedastic count data \
             (gam#1606); instead the inner blockwise solve aborted: {e:?}"
        )
    });

    // Predicted per-row means must be finite and strictly positive (the contract
    // the issue requires: the NB LS fit predicts finite positive per-row means).
    // The mean-channel predictor η_μ is block 0's converged `eta`.
    let eta_mu = &fit.block_states[DispersionGlmLocationScaleFamily::BLOCK_MEAN].eta;
    assert_eq!(eta_mu.len(), n, "mean predictor must cover every row");
    assert!(
        eta_mu.iter().all(|&e| e.is_finite()),
        "fitted mean predictor must be finite on every row"
    );
    assert!(
        eta_mu.iter().all(|&e| e.exp().is_finite() && e.exp() > 0.0),
        "fitted per-row means must be finite and strictly positive"
    );
}

// =====================================================================
// #2155 inner-mode geography measurement harness (zz_measure diagnostics).
//
// The outer-optimizer half of #2155 mode (b) landed (6966b2a31 / 8739251b6 /
// ece539920); the residual blocker is the measured warm/cold inner-solve
// bimodality: a warm-started binomial mean-wiggle solve reaches a strictly
// lower mode than a cold solve at the same ρ, so the cold-reproducible
// terminal state is not the mode the search descended. These zz tests map the
// mode geography of the REAL #2155 fixture (600 rows, seed 2155, y ~ x with a
// flexible link) at FIXED wiggle log-λ, comparing:
//   (a) the production cold seed (pilot β, β_w = 0) jumped straight to the
//       target λ, against
//   (b) a deterministic warp-penalty continuation: the same solve reached
//       through a descending λ ladder anchored at the exact large-λ limit
//       (the pilot fit — see fit_orchestration/fit.rs on the wiggle model
//       containing the baseline as its large-λ limiting case).
// If (b) reaches a strictly lower penalized objective than (a) in a λ region,
// the graduated continuation is the correct canonical cold solve for this
// family and becomes the production fix.
// =====================================================================

/// A binomial location-scale wiggle family with one converged-looking state
/// per block (threshold, log-sigma, wiggle), for the row-lowering gates.
///
/// Built from the surviving constructors: the knots from the seed index, the
/// wiggle block from those knots, the family from its fields.
pub(crate) fn bls_wiggle_workspace_fixture() -> (
    BinomialLocationScaleWiggleFamily,
    Vec<ParameterBlockState>,
) {
    let link_kind = InverseLink::Standard(StandardLink::Probit);
    let n = 48usize;
    let pt = 3usize;
    let pls = 2usize;
    let xt = Array2::from_shape_fn((n, pt), |(i, j)| {
        ((i as f64) * 0.17 + (j as f64) * 0.29).sin() * 0.4
    });
    let xls = Array2::from_shape_fn((n, pls), |(i, j)| {
        ((i as f64) * 0.23 + (j as f64) * 0.41).cos() * 0.3
    });
    let beta_t = array![0.20, -0.10, 0.05];
    let beta_ls = array![0.30, -0.15];
    let eta_t = xt.dot(&beta_t);
    let eta_ls = xls.dot(&beta_ls);
    let q_seed = Array1::linspace(-1.0, 1.0, n);
    let knots = initializewiggle_knots_from_seed(q_seed.view(), 2, 3).expect("wiggle knots");
    let y = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let threshold_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xt.clone()));
    let log_sigma_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
    let family = BinomialLocationScaleWiggleFamily {
        y,
        weights,
        link_kind,
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        jeffreys_armed: false,
    };
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(eta_ls.iter())
            .map(|(&eta_t_i, &eta_ls_i)| {
                binomial_location_scale_q0(eta_t_i, gam_model_kernels::sigma_link::exp_sigma_from_eta_scalar(eta_ls_i))
            }),
    );
    let wiggle_design_current = family
        .wiggle_design(q0.view())
        .expect("current wiggle basis");
    let pw = wiggle_design_current.ncols();
    let beta_w = Array1::from_shape_fn(pw, |j| 0.05 * ((j + 1) as f64).cos());
    let eta_w = wiggle_design_current.dot(&beta_w);
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: beta_w,
            eta: eta_w,
        },
    ];
    (family, states)
}

/// #932 exact-tower oracle for the canonical binomial location-scale WIGGLE
/// order-two row program.
///
/// `BinomialLocationScaleWiggleFamily::wiggle_order2_rows` lowers the shared
/// row expression into per-row joint-Hessian coefficients for the composed
/// index
/// `q = q0(η_t, η_ls) + Σ_j βw_j·B_j(q0)` via the chain factors `m = B'·βw + 1`,
/// `g2 = B''·βw`. The cross-block coefficients (`coeff_tw_*`, `coeff_lw_*`,
/// `coeff_ww`: threshold/log-sigma × wiggle and wiggle × wiggle) are exactly the
/// #736 dropped/sign-flipped cross-term genus, and until now no exact oracle
/// pinned them to an independent tower — only an operator-vs-dense check and
/// an FD approximation covered them.
///
/// This is the #932 single-source guard. For each row `i` and basis column `j`
/// we build an INDEPENDENT order-2 jet `Tower2<3>` over `(η_t, η_ls, βw_j)`
/// (the other `βw_k` held at their fixed values), compose the wiggle basis onto
/// the non-wiggle index tower
/// (`q = q0_tower + Σ_k coef_k · q0_tower.compose_unary([B_k, B'_k, B''_k])`),
/// then compose the binomial neglog objective onto `q`
/// (`nll = q.compose_unary([·, m1, m2])`). The resulting `3×3` Hessian block IS
/// every `coeff_*` mechanically:
///   `h[0][0]=coeff_tt`, `h[0][1]=coeff_tl`, `h[1][1]=coeff_ll`,
///   `h[0][2]=coeff_tw_b·B_j + coeff_tw_d·B'_j`,
///   `h[1][2]=coeff_lw_b·B_j + coeff_lw_d·B'_j`,
///   `h[2][2]=coeff_ww·B_j²`.
/// A dropped or sign-flipped generated coefficient shifts a block well outside
/// 1e-9
/// and fails loudly, for probit / logit / cloglog. The value channel is
/// irrelevant to the Hessian (`compose_unary`'s `h` reads only `f'`/`f''`), so a
/// placeholder `0.0` is passed for the objective value.
#[test]
pub(crate) fn binomial_location_scale_wiggle_order2_rows_match_jet_tower_932() {
    use super::super::binomial_q_derivs::binomial_neglog_q_derivatives_dispatch;
    use gam_math::jet_tower::Tower2;

    let (probit_family, states) = bls_wiggle_workspace_fixture();
    let n = probit_family.y.len();

    for link in [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
    ] {
        // Designs, knots and the block etas are link-independent (`q0` and the
        // wiggle basis do not depend on the binomial link), so reuse them and
        // swap only `link_kind` for each arm.
        let family = BinomialLocationScaleWiggleFamily {
            y: probit_family.y.clone(),
            weights: probit_family.weights.clone(),
            link_kind: link.clone(),
            threshold_design: probit_family.threshold_design.clone(),
            log_sigma_design: probit_family.log_sigma_design.clone(),
            wiggle_knots: probit_family.wiggle_knots.clone(),
            wiggle_degree: probit_family.wiggle_degree,
            policy: probit_family.policy.clone(),
            jeffreys_armed: probit_family.jeffreys_armed,
        };

        let pieces = family
            .wiggle_order2_rows(&states)
            .expect("canonical wiggle order-two rows");

        let eta_t = &states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta;
        let eta_ls = &states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta;
        let etaw = &states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta;
        let betaw = &states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta;

        let core0 = binomial_location_scale_core(
            &family.y,
            &family.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &family.link_kind,
        )
        .expect("binomial location-scale core");

        // Same basis tensors the canonical row program consumes:
        // pieces.{b0,d0} are exactly B and B' it used; recompute B'' for the
        // order-2 composition.
        let b0 = &pieces.b0;
        let d0 = &pieces.d0;
        let dd0 = family
            .wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())
            .expect("wiggle second-derivative basis");
        let pw = b0.ncols();

        for i in 0..n {
            let qi = core0.q0[i] + etaw[i];
            let (m1, m2, _m3) = binomial_neglog_q_derivatives_dispatch(
                family.y[i],
                family.weights[i],
                qi,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &family.link_kind,
            );

            // Non-wiggle index q0 = -η_t · exp(-η_ls) over axes (η_t, η_ls);
            // axis 2 is reserved for the per-column wiggle amplitude.
            let eta_t_t = Tower2::<3>::variable(eta_t[i], 0);
            let eta_ls_t = Tower2::<3>::variable(eta_ls[i], 1);

            let q0_tower = (eta_t_t * -1.0) * (eta_ls_t * -1.0).exp();

            for j in 0..pw {
                let mut q = q0_tower;
                for k in 0..pw {
                    let coef = if k == j {
                        Tower2::<3>::variable(betaw[j], 2)
                    } else {
                        Tower2::<3>::constant(betaw[k])
                    };
                    let basis_k = q0_tower.compose_unary([b0[[i, k]], d0[[i, k]], dd0[[i, k]]]);
                    q = q + coef * basis_k;
                }
                let nll = q.compose_unary([0.0, m1, m2]);
                let h = nll.h;

                let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);

                assert!(
                    close(h[0][0], pieces.coeff_tt[i]),
                    "{link:?} coeff_tt[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[0][0],
                    pieces.coeff_tt[i]
                );
                assert!(
                    close(h[0][1], pieces.coeff_tl[i]),
                    "{link:?} coeff_tl[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[0][1],
                    pieces.coeff_tl[i]
                );
                assert!(
                    close(h[1][1], pieces.coeff_ll[i]),
                    "{link:?} coeff_ll[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[1][1],
                    pieces.coeff_ll[i]
                );

                let tw = pieces.coeff_tw_b[i] * b0[[i, j]] + pieces.coeff_tw_d[i] * d0[[i, j]];
                let lw = pieces.coeff_lw_b[i] * b0[[i, j]] + pieces.coeff_lw_d[i] * d0[[i, j]];
                let ww = pieces.coeff_ww[i] * b0[[i, j]] * b0[[i, j]];
                assert!(
                    close(h[0][2], tw),
                    "{link:?} (η_t,βw) cross[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[0][2],
                    tw
                );
                assert!(
                    close(h[1][2], lw),
                    "{link:?} (η_ls,βw) cross[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[1][2],
                    lw
                );
                assert!(
                    close(h[2][2], ww),
                    "{link:?} (βw,βw)[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[2][2],
                    ww
                );
            }
        }
    }
}

/// #932 release speed gate for the binomial location-scale WIGGLE order-two
/// row program: the production typed-probe lowering (`order2_row`, ONE
/// `Order2<4>` evaluation per row with 8 coefficient channels, cost linear in
/// the basis width `pw`) must beat the naive per-(row, column) dense-tower
/// assembly (`pw` independent `Tower2<3>` compositions per row — the generic
/// shape the oracle above uses as its witness, and the only alternative
/// representation since the pre-cutover hand ladder was deleted by the #932
/// single-source migration).
///
/// The timed unit is one row. Everything a batch shares — the location-scale
/// core, the wiggle bases through second order, the block etas — is built
/// once outside the timed region for both arms; the production arm reads it
/// through the batch's row program, the tower arm through the same core and
/// bases the oracle uses. The first version of this gate timed
/// `wiggle_order2_rows` whole-batch against a whole-batch tower assembly on
/// a 10-row fixture: 11 µs per call of which the row loop was a tenth, so
/// the ratio (0.97 on an EPYC 9V74 runner) was a race of two setups and the
/// allocations in them, at a 14% resolution. Each arm is outlined so its
/// fixture-invariant work cannot be hoisted out of the batch, and the
/// predictors are nudged per call so no two calls share an input.
#[test]
pub(crate) fn release_measure_bls_wiggle_order2_rows_vs_per_column_tower_932() {
    use super::super::binomial_q_derivs::binomial_neglog_q_derivatives_dispatch;
    use gam_math::jet_tower::Tower2;
    use gam_math::paired_timing::{SpeedGate, batched, paired_interleaved};

    let (family, states) = bls_wiggle_workspace_fixture();
    let n = family.y.len();
    let eta_t = &states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta;
    let eta_ls = &states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta;
    let etaw = &states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta;
    let betaw = &states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta;

    // Production: the batch's row program, built once.
    let program = family
        .wiggle_row_program(&states)
        .expect("production wiggle row program");

    #[inline(never)]
    fn production_row(
        program: &BinomialLocationScaleWiggleRowProgram<'_>,
        row: usize,
        eta_t: f64,
        eta_ls: f64,
    ) -> f64 {
        let h = program
            .order2_row(row, eta_t, eta_ls, BinomialWiggleRowOuter::Observed)
            .expect("production wiggle order-two row");
        h[0][0] + h[0][1] + h[2][2] + h[0][2]
    }

    // The naive generic assembly: per (row, column) an independent Tower2<3>
    // over (eta_t, eta_ls, betaw_j), composed exactly as the oracle above,
    // over the same core and bases built once.
    let core = binomial_location_scale_core(
        &family.y,
        &family.weights,
        eta_t,
        eta_ls,
        Some(etaw),
        &family.link_kind,
    )
    .expect("binomial location-scale core");
    let b0 = family
        .wiggle_basiswith_options(core.q0.view(), BasisOptions::value())
        .expect("wiggle value basis");
    let d0 = family
        .wiggle_basiswith_options(core.q0.view(), BasisOptions::first_derivative())
        .expect("wiggle first-derivative basis");
    let dd0 = family
        .wiggle_basiswith_options(core.q0.view(), BasisOptions::second_derivative())
        .expect("wiggle second-derivative basis");

    struct TowerBatch<'a> {
        family: &'a BinomialLocationScaleWiggleFamily,
        core: &'a BinomialLocationScaleCore,
        b0: &'a Array2<f64>,
        d0: &'a Array2<f64>,
        dd0: &'a Array2<f64>,
        etaw: &'a Array1<f64>,
        betaw: &'a Array1<f64>,
    }

    #[inline(never)]
    fn tower_row(batch: &TowerBatch<'_>, row: usize, eta_t: f64, eta_ls: f64) -> f64 {
        let family = batch.family;
        let core = batch.core;
        let qi = core.q0[row] + batch.etaw[row];
        let (m1, m2, _m3) = binomial_neglog_q_derivatives_dispatch(
            family.y[row],
            family.weights[row],
            qi,
            core.mu[row],
            core.dmu_dq[row],
            core.d2mu_dq2[row],
            core.d3mu_dq3[row],
            &family.link_kind,
        );
        let eta_t_t = Tower2::<3>::variable(eta_t, 0);
        let eta_ls_t = Tower2::<3>::variable(eta_ls, 1);
        let q0_tower = (eta_t_t * -1.0) * (eta_ls_t * -1.0).exp();
        let pw = batch.b0.ncols();
        let mut folded = 0.0;
        for j in 0..pw {
            let mut q = q0_tower;
            for k in 0..pw {
                let coef = if k == j {
                    Tower2::<3>::variable(batch.betaw[j], 2)
                } else {
                    Tower2::<3>::constant(batch.betaw[k])
                };
                let basis_k =
                    q0_tower.compose_unary([batch.b0[[row, k]], batch.d0[[row, k]], batch.dd0[[row, k]]]);
                q = q + coef * basis_k;
            }
            let nll = q.compose_unary([0.0, m1, m2]);
            folded += nll.h[0][0] + nll.h[0][2] + nll.h[2][2];
        }
        folded
    }

    let tower_batch = TowerBatch {
        family: &family,
        core: &core,
        b0: &b0,
        d0: &d0,
        dd0: &dd0,
        etaw,
        betaw,
    };

    // Sanity: both arms produce finite folds on every row of the fixture.
    for row in 0..n {
        assert!(production_row(&program, row, eta_t[row], eta_ls[row]).is_finite());
        assert!(tower_row(&tower_batch, row, eta_t[row], eta_ls[row]).is_finite());
    }

    // Speed contract, release profile only (`SpeedGate::open` documents why):
    // the typed-probe lowering must beat the naive per-(row, column) tower
    // assembly. One arm call is one row; a batch walks the fixture's rows in
    // turn, each at its own nudged predictors.
    if cfg!(debug_assertions) {
        return;
    }
    let mut gate = SpeedGate::open("BLS-WIGGLE-ORDER2-932");
    let mut production_cursor = 0usize;
    let mut tower_cursor = 0usize;
    let timing = paired_interleaved(
        15,
        10,
        0x9320_B15B,
        batched(n, |nudge| {
            let row = production_cursor % n;
            production_cursor += 1;
            production_row(&program, row, eta_t[row] + nudge, eta_ls[row] + nudge)
        }),
        batched(n, |nudge| {
            let row = tower_cursor % n;
            tower_cursor += 1;
            tower_row(&tower_batch, row, eta_t[row] + nudge, eta_ls[row] + nudge)
        }),
    );
    gate.faster("order2", &timing, "production", "per_column_tower");
    gate.finish();
}

#[test]
pub(crate) fn binomial_location_scalewiggle_termswith_matern_spatial_blocks_fit_finitely() {
    let n = 30usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.5 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 9 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let q_seed = Array1::linspace(-1.5, 1.5, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");
    let spec = BinomialLocationScaleWiggleTermSpec {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        thresholdspec: simple_matern_term_collection(&[0, 1], 0.45),
        log_sigmaspec: empty_term_collection(),
        threshold_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
        wiggle_knots: knots,
        wiggle_degree: 2,
        wiggle_block,
    };
    let fit = fit_binomial_location_scalewiggle_terms(
        data.view(),
        spec,
        &spatial_fit_smoke_options(),
        &spatial_kappa_options(),
    )
    .expect("binomial location-scale wiggle spatial fit");
    assert!(fit.fit.penalized_objective().is_some_and(f64::is_finite));
    assert_eq!(fit.fit.block_states.len(), 3);
}

/// gam#2647, as the property rather than as a symptom: **the fitted optimum must
/// not depend on the inner cycle budget.**
///
/// The failure this pins is not "the solve ran out of cycles". It is that the
/// penalized criterion had no minimiser: the monotone warp's linear element is a
/// rescale of the index it is composed onto, and with an order-2 roughness and
/// `double_penalty = false` that element was unpenalized, so the objective
/// decreased monotonically along the orbit toward an infimum at `‖β‖ = ∞`. A
/// solve on such a criterion returns wherever its budget happened to stop, and
/// the measured signature was exactly that: `‖β‖∞` climbing 230× while
/// `½βᵀSβ` fell like `‖β‖⁻²` and `−loglik` stayed flat to `8e-4`.
///
/// So the discriminating question is not "does it converge" — a guard can make
/// anything stop — but "does it converge to the SAME point from two different
/// budgets". Before the gauge closure the 48- and 200-cycle arms disagreed and
/// neither was stationary; after it they agree bit-for-bit. This test cannot be
/// satisfied by loosening a tolerance or lengthening a budget, which is what
/// makes it worth its ~0.2 s.
#[test]
pub(crate) fn binomial_location_scalewiggle_optimum_is_budget_independent_2647() {
    let n = 30usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.5 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 9 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let q_seed = Array1::linspace(-1.5, 1.5, n);

    let fit_at = |inner_max_cycles: usize| -> f64 {
        let (wiggle_block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
            q_seed.view(),
            2,
            4,
            2,
            false,
        )
        .expect("wiggle block");
        let spec = BinomialLocationScaleWiggleTermSpec {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: InverseLink::Standard(StandardLink::Probit),
            thresholdspec: simple_matern_term_collection(&[0, 1], 0.45),
            log_sigmaspec: empty_term_collection(),
            threshold_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
            wiggle_knots: knots,
            wiggle_degree: 2,
            wiggle_block,
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles,
            ..spatial_fit_smoke_options()
        };
        let fit = fit_binomial_location_scalewiggle_terms(
            data.view(),
            spec,
            &options,
            &spatial_kappa_options(),
        )
        .unwrap_or_else(|error| {
            panic!(
                "binomial location-scale wiggle spatial fit refused at \
                 inner_max_cycles={inner_max_cycles}: {error}"
            )
        });
        fit.fit
            .penalized_objective()
            .expect("a converged fit must carry a penalized objective")
    };

    let short = fit_at(48);
    let long = fit_at(200);
    assert!(
        short.is_finite() && long.is_finite(),
        "penalized objective must be finite at both budgets: 48 -> {short}, 200 -> {long}"
    );
    // Same criterion, same minimiser: the two solves differ only in a budget
    // neither of them needs, so any difference at all is evidence the objective
    // is still being descended rather than located.
    let spread = (short - long).abs() / (1.0 + short.abs());
    assert!(
        spread <= 1e-9,
        "the fitted optimum moved with the inner cycle budget (gam#2647): \
         48 cycles -> {short:.12e}, 200 cycles -> {long:.12e}, relative spread {spread:.3e}. \
         A criterion whose argmin depends on how long you look for it does not have one."
    );
}

/// #932: at βw = 0 the WIGGLE joint-Hessian assembler must reduce EXACTLY to
/// the (already tower-pinned) non-wiggle assembler on the (η_t, η_ls) block.
///
/// The canonical wiggle row program differentiates the composed index `q = q0 +
/// Σ_j βw_j·B_j(q0)` through `m = B'·βw + 1` and `g2 = B''·βw`; its
/// `coeff_tw_*` / `coeff_lw_*` / `coeff_ww` cross blocks are the #736 genus. A
/// full wiggle tower is
/// a larger unit (#932 comment), but one structurally-certain invariant is
/// cheap and independent: at `βw = 0` we have `m = 1`, `g2 = 0`, `etaw = 0`, so
/// `q = q0` and the wiggle base coefficients collapse to the non-wiggle ones
/// (`coeff_tt = hessian_coeff(m1, m2, q0_t, q0_t, 0)`, etc.). Therefore the
/// wiggle joint Hessian's top-left `(pt+pls)` block must equal the non-wiggle
/// `exact_newton_joint_hessian_from_designs` joint matrix built from the SAME
/// data — two INDEPENDENT hand assemblers (the non-wiggle one is itself pinned
/// to the single-source tower by
/// `binomial_location_scale_joint_hessian_matches_single_sourced_tower_932`),
/// so agreement transitively pins the wiggle base block to the tower and would
/// catch a typo in the wiggle base chain. Across probit / logit / cloglog.
#[test]
pub(crate) fn binomial_wiggle_joint_hessian_reduces_to_nonwiggle_at_zero_betaw_932() {
    let n = 10usize;
    let pt = 3usize;
    let pls = 2usize;
    let xt = Array2::from_shape_fn((n, pt), |(i, j)| {
        ((i as f64) * 0.17 + (j as f64) * 0.29).sin() * 0.4 + 0.2
    });
    let xls = Array2::from_shape_fn((n, pls), |(i, j)| {
        ((i as f64) * 0.23 + (j as f64) * 0.41).cos() * 0.3
    });
    let beta_t = array![0.20, -0.10, 0.05];
    let beta_ls = array![0.30, -0.15];
    let eta_t = xt.dot(&beta_t);
    let eta_ls = xls.dot(&beta_ls);
    let y = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_iter((0..n).map(|i| 0.5 + 0.2 * i as f64));
    let q_seed = Array1::linspace(-1.0, 1.0, n);
    let (_wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");

    for link in [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
    ] {
        let threshold_design =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xt.clone()));
        let log_sigma_design =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
        let wiggle_family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: link.clone(),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
            wiggle_knots: knots.clone(),
            wiggle_degree: 2,
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
            jeffreys_armed: false,
        };
        // βw = 0 ⇒ etaw = 0, m = 1, g2 = 0, q = q0.
        let q0 = Array1::from_iter(
            eta_t
                .iter()
                .zip(eta_ls.iter())
                .map(|(&t, &l)| binomial_location_scale_q0(t, exp_sigma_from_eta_scalar(l))),
        );
        let wiggle_design_current = wiggle_family
            .wiggle_design(q0.view())
            .expect("current wiggle basis");
        let pw = wiggle_design_current.ncols();
        let beta_w = Array1::<f64>::zeros(pw);
        let eta_w = Array1::<f64>::zeros(n);
        let wiggle_states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls.clone(),
            },
            ParameterBlockState {
                beta: beta_w,
                eta: eta_w,
            },
        ];
        let h_wiggle = wiggle_family
            .exact_newton_joint_hessian(&wiggle_states)
            .expect("wiggle joint Hessian")
            .expect("wiggle joint Hessian present");
        assert_eq!(h_wiggle.dim(), (pt + pls + pw, pt + pls + pw));

        // Non-wiggle reference on identical data / states.
        let nonwiggle_family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: link.clone(),
            threshold_design: None,
            log_sigma_design: None,
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
            jeffreys_armed: false,
        };
        let nonwiggle_states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls.clone(),
            },
        ];
        let h_nonwiggle = nonwiggle_family
            .exact_newton_joint_hessian_from_designs(&nonwiggle_states, &xt, &xls)
            .expect("non-wiggle joint Hessian")
            .expect("non-wiggle joint Hessian present");
        assert_eq!(h_nonwiggle.dim(), (pt + pls, pt + pls));

        // The wiggle (η_t, η_ls) top-left block must equal the non-wiggle joint
        // Hessian exactly (both are analytic; βw = 0 makes them the same model).
        for a in 0..(pt + pls) {
            for b in 0..(pt + pls) {
                let w = h_wiggle[[a, b]];
                let nw = h_nonwiggle[[a, b]];
                assert!(
                    (w - nw).abs() <= 1e-9 * (1.0 + nw.abs()),
                    "{link:?}: wiggle (β_w=0) joint Hessian [{a}][{b}] {w:.9e} != \
                     non-wiggle {nw:.9e}"
                );
            }
        }
    }
}

/// #2387 contract for the observed joint Hessian and its ψ tower (supersedes
/// the #684 Fisher-cross-zero pin). Production deliberately builds the
/// OBSERVED Wood–Pya–Säfken LAML curvature (`gaussian_locscale_observed_joint_
/// row_coeffs`: `mm = w`, `ml = 2κm`, `ll = κ′(a−n) + 2κ²n`; see #1561 — the
/// old block-Fisher object zeroed `ml`, dropped the cross-block Schur deficit
/// and biased λ̂_σ upward). The fixtures carry NONZERO residuals, so the
/// observed `2κm` cross is genuinely nonzero here and a Fisher-zero pin is
/// wrong by design. What IS invariant, and what this test pins:
///
///  * the dense joint Hessian equals the independent row-sum assembly
///    `Σ_i x_iᵀ · [[mm, ml], [ml, ll]]_i · x_i` of those observed
///    coefficients (content pin, tight tolerance);
///  * every ψ-layer builder (`exact_newton_joint_psi*`: 1st-order, 2nd-order,
///    mixed β·ψ) is the exact matrix derivative of that SAME observed object,
///    pinned against full-matrix central finite differences in which the
///    ψ-perturbed family is REBUILT at every probe point (fresh row-scalar
///    cache — the FD gate never freezes the cache), with the design model
///    `X(ψ) = X + ψ·X_ψ + ½ψ²·X_ψψ` matching the derivative payloads.
///
/// A regression to the Fisher object (or any dropped observed cross term)
/// fails the content pin; a ψ-builder that differentiates a different object
/// than the dense path fails the FD gates. Both families (plain and wiggle)
/// are covered; the wiggle FD rebuilds the q0-dependent basis at each probe
/// exactly as `refresh_all_block_etas` does at fit time.
#[test]
pub(crate) fn gaussian_location_scale_joint_hessian_is_observed_and_psi_layers_match_matrix_fd_2387()
{
    use gam_problem::HyperOperator;

    // Materialize an `ExactNewtonJointPsiTerms` joint Hessian regardless of
    // whether the family returns it dense or operator-backed.
    fn materialize(
        dense: &Array2<f64>,
        operator: Option<&dyn HyperOperator>,
        total: usize,
    ) -> Array2<f64> {
        match operator {
            Some(op) => op.to_dense(),
            None => {
                assert_eq!(dense.dim(), (total, total));
                dense.clone()
            }
        }
    }

    // Max |entry| over the rectangular block H[r0..r1, c0..c1].
    fn block_max_abs(h: &Array2<f64>, r0: usize, r1: usize, c0: usize, c1: usize) -> f64 {
        let mut m = 0.0_f64;
        for r in r0..r1 {
            for c in c0..c1 {
                m = m.max(h[[r, c]].abs());
            }
        }
        m
    }

    // Full-matrix closeness against a reference, scaled by the reference's
    // magnitude so O(1) and O(10⁻³) entries are judged uniformly.
    fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64, label: &str) {
        assert_eq!(actual.dim(), expected.dim(), "{label}: dimension mismatch");
        let scale = 1.0 + expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for r in 0..expected.nrows() {
            for c in 0..expected.ncols() {
                let a = actual[[r, c]];
                let e = expected[[r, c]];
                assert!(
                    (a - e).abs() <= tol * scale,
                    "{label}: entry ({r},{c}) diverged: got {a:.12e}, expected {e:.12e} \
                     (tol {tol:.1e} × scale {scale:.3e})"
                );
            }
        }
    }

    // ---- Non-wiggle GaussianLocationScaleFamily ----------------------
    {
        let (family, states, specs) = gls_workspace_fixture();
        let p_mu = states[GaussianLocationScaleFamily::BLOCK_MU].beta.len();
        let p_ls = states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA]
            .beta
            .len();
        let total = p_mu + p_ls;
        let n = family.y.len();
        let xmu = specs[0]
            .design
            .as_dense_ref()
            .expect("dense xmu")
            .to_owned();
        let xls = specs[1]
            .design
            .as_dense_ref()
            .expect("dense xls")
            .to_owned();

        // ψ design-Jacobian payloads on the MEAN (μ) block, with a 2nd-order
        // payload so the 2nd-order builder is exercised too.
        let x_mu_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.2 + 0.11 * ((i as f64) * 0.37 + (j as f64) * 0.53).sin()
        });
        let x_mu_psi_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.07 * ((i as f64) * 0.19 + (j as f64) * 0.23).cos()
        });
        let derivative_blocks = vec![
            vec![CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: x_mu_psi.clone(),
                s_psi: Array2::zeros((p_mu, p_mu)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: Some(vec![x_mu_psi_psi.clone()]),
                s_psi_psi: Some(vec![Array2::zeros((p_mu, p_mu))]),
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: None,
                implicit_axis: 0,
                implicit_group_id: None,
            }],
            Vec::new(),
        ];

        // (A) Content pin: the dense joint Hessian IS the observed object —
        // the independent row-sum assembly of the single-source coefficients.
        let dense_h = family
            .exact_newton_joint_hessian(&states)
            .expect("dense joint Hessian build")
            .expect("dense joint Hessian present");
        let rows = family
            .get_or_compute_row_scalars(&states[0].eta, &states[1].eta)
            .expect("row scalars");
        let (mm, ml, ll) = gaussian_locscale_observed_joint_row_coeffs(&rows);
        let mut expected = Array2::<f64>::zeros((total, total));
        for i in 0..n {
            for a in 0..p_mu {
                for b in 0..p_mu {
                    expected[[a, b]] += xmu[[i, a]] * mm[i] * xmu[[i, b]];
                }
                for b in 0..p_ls {
                    let v = xmu[[i, a]] * ml[i] * xls[[i, b]];
                    expected[[a, p_mu + b]] += v;
                    expected[[p_mu + b, a]] += v;
                }
            }
            for a in 0..p_ls {
                for b in 0..p_ls {
                    expected[[p_mu + a, p_mu + b]] += xls[[i, a]] * ll[i] * xls[[i, b]];
                }
            }
        }
        assert_matrix_close(
            &dense_h,
            &expected,
            1e-10,
            "#2387: dense joint Hessian must equal the observed row-sum assembly",
        );
        // Non-vacuity: the observed 2κm cross is genuinely nonzero on this
        // fixture (a Fisher regression would zero it and fail the content pin
        // loudly, not vacuously).
        let cross_mag = block_max_abs(&dense_h, 0, p_mu, p_mu, total);
        assert!(
            cross_mag > 1e-1,
            "#2387: fixture must exercise a nonzero observed 2κm cross, got {cross_mag:.3e}"
        );

        // Dense observed H with the μ design perturbed along the ψ payloads:
        // X(ψ) = X + ψ·X_ψ + ½ψ²·X_ψψ, family REBUILT per probe (fresh cache).
        let dense_h_at = |t: f64| -> Array2<f64> {
            let xmu_t = &xmu + &(t * &x_mu_psi) + &((0.5 * t * t) * &x_mu_psi_psi);
            let fam_t = GaussianLocationScaleFamily {
                y: family.y.clone(),
                weights: family.weights.clone(),
                mu_design: Some(DesignMatrix::Dense(
                    gam_linalg::matrix::DenseDesignMatrix::from(xmu_t.clone()),
                )),
                log_sigma_design: family.log_sigma_design.clone(),
                policy: gam_runtime::resource::ResourcePolicy::default_library(),
                cached_row_scalars: std::sync::RwLock::new(None),
            };
            let states_t = vec![
                ParameterBlockState {
                    beta: states[0].beta.clone(),
                    eta: xmu_t.dot(&states[0].beta),
                },
                states[1].clone(),
            ];
            fam_t
                .exact_newton_joint_hessian(&states_t)
                .expect("perturbed dense joint Hessian build")
                .expect("perturbed dense joint Hessian present")
        };

        // (B) 1st-order ψ builder == central FD of the dense observed H.
        let layout = test_design_hyper_layout(&derivative_blocks);
        let psi = family
            .exact_newton_joint_psi_terms(&states, &specs, &layout, 0)
            .expect("psi terms call")
            .expect("gaussian psi terms present");
        let h_psi = materialize(&psi.hessian_psi, psi.hessian_psi_operator.as_deref(), total);
        let h1 = 1e-5;
        let fd1 = (&dense_h_at(h1) - &dense_h_at(-h1)) / (2.0 * h1);
        assert_matrix_close(
            &h_psi,
            &fd1,
            1e-6,
            "#2387: 1st-order ψ joint Hessian must be the matrix FD of the dense observed H",
        );

        // (C) 2nd-order ψ builder == second central FD.
        let psi2 = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &layout, 0, 0)
            .expect("psi 2nd-order call")
            .expect("gaussian psi 2nd-order present");
        let h_psi2 = materialize(
            &psi2.hessian_psi_psi,
            psi2.hessian_psi_psi_operator.as_deref(),
            total,
        );
        let h2 = 1e-3;
        let fd2 = (&(&dense_h_at(h2) + &dense_h_at(-h2)) - &(2.0 * &dense_h)) / (h2 * h2);
        assert_matrix_close(
            &h_psi2,
            &fd2,
            5e-4,
            "#2387: 2nd-order ψ joint Hessian must be the second matrix FD of the dense observed H",
        );

        // (D) Mixed β·ψ builder == central FD in β (along d_beta) of the
        // 1st-order ψ-Hessian, states rebuilt (η from designs) per probe.
        let d_beta = Array1::from_shape_fn(total, |i| 0.05 + 0.13 * ((i + 1) as f64).sin());
        let psi_h_at_beta = |t: f64| -> Array2<f64> {
            let mut st = states.clone();
            for j in 0..p_mu {
                st[0].beta[j] += t * d_beta[j];
            }
            for j in 0..p_ls {
                st[1].beta[j] += t * d_beta[p_mu + j];
            }
            st[0].eta = xmu.dot(&st[0].beta);
            st[1].eta = xls.dot(&st[1].beta);
            let psi_t = family
                .exact_newton_joint_psi_terms(&st, &specs, &layout, 0)
                .expect("perturbed psi terms call")
                .expect("perturbed psi terms present");
            materialize(&psi_t.hessian_psi, psi_t.hessian_psi_operator.as_deref(), total)
        };
        let mixed = family
            .exact_newton_joint_psihessian_directional_derivative(&states, &specs, &layout, 0, &d_beta)
            .expect("psi mixed-drift call")
            .expect("gaussian psi mixed-drift present");
        assert_eq!(mixed.dim(), (total, total));
        let hb = 1e-5;
        let fdm = (&psi_h_at_beta(hb) - &psi_h_at_beta(-hb)) / (2.0 * hb);
        assert_matrix_close(
            &mixed,
            &fdm,
            1e-5,
            "#2387: mixed β·ψ ψ-Hessian must be the β-directional FD of the 1st-order ψ-Hessian",
        );
    }

    // ---- Wiggle GaussianLocationScaleWiggleFamily --------------------
    {
        let (family, states, specs, xmu, xls, _xw_seed) = gls_wiggle_workspace_fixture();
        let p_mu = states[GaussianLocationScaleWiggleFamily::BLOCK_MU]
            .beta
            .len();
        let p_ls = states[GaussianLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
            .beta
            .len();
        let p_w = states[GaussianLocationScaleWiggleFamily::BLOCK_WIGGLE]
            .beta
            .len();
        let total = p_mu + p_ls + p_w;
        let n = family.y.len();
        let ls0 = p_mu;
        let ls1 = p_mu + p_ls;
        let w0 = p_mu + p_ls;
        let w1 = total;

        // ψ design-Jacobian on the MEAN (μ) block (psi_index 0), exercising
        // both mean⊥scale crosses and their derivatives.
        let x_mu_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.18 + 0.09 * ((i as f64) * 0.41 + (j as f64) * 0.29).sin()
        });
        let x_mu_psi_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.06 * ((i as f64) * 0.17 + (j as f64) * 0.31).cos()
        });
        let derivative_blocks = vec![
            vec![CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: x_mu_psi.clone(),
                s_psi: Array2::zeros((p_mu, p_mu)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: Some(vec![x_mu_psi_psi.clone()]),
                s_psi_psi: Some(vec![Array2::zeros((p_mu, p_mu))]),
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: None,
                implicit_axis: 0,
                implicit_group_id: None,
            }],
            Vec::new(),
            Vec::new(),
        ];

        // Dense observed H with the μ design perturbed along the ψ payloads.
        // The wiggle block has dynamic geometry (q0-dependent basis), so the
        // rebuild recomputes η_w from the perturbed q0 exactly as
        // `refresh_all_block_etas` does at fit time — a frozen wiggle basis
        // would silently drop the basis-drift channel from the FD reference.
        let dense_h_at = |t: f64| -> Array2<f64> {
            let xmu_t = &xmu + &(t * &x_mu_psi) + &((0.5 * t * t) * &x_mu_psi_psi);
            let fam_t = GaussianLocationScaleWiggleFamily {
                y: family.y.clone(),
                weights: family.weights.clone(),
                mu_design: Some(DesignMatrix::Dense(
                    gam_linalg::matrix::DenseDesignMatrix::from(xmu_t.clone()),
                )),
                log_sigma_design: family.log_sigma_design.clone(),
                wiggle_knots: family.wiggle_knots.clone(),
                wiggle_degree: family.wiggle_degree,
                policy: gam_runtime::resource::ResourcePolicy::default_library(),
                jeffreys_armed: true,
            };
            let eta_mu_t = xmu_t.dot(&states[0].beta);
            let eta_w_t = fam_t
                .wiggle_design(eta_mu_t.view())
                .expect("perturbed wiggle basis at q0")
                .dot(&states[2].beta);
            let states_t = vec![
                ParameterBlockState {
                    beta: states[0].beta.clone(),
                    eta: eta_mu_t,
                },
                states[1].clone(),
                ParameterBlockState {
                    beta: states[2].beta.clone(),
                    eta: eta_w_t,
                },
            ];
            fam_t
                .exact_newton_joint_hessian(&states_t)
                .expect("perturbed wiggle dense joint Hessian build")
                .expect("perturbed wiggle dense joint Hessian present")
        };

        let dense_h = dense_h_at(0.0);
        assert_eq!(dense_h.dim(), (total, total));
        // Non-vacuity: both mean⊥scale observed crosses are present (the old
        // Fisher pin asserted these were zero; observed they are not).
        let c_ml = block_max_abs(&dense_h, 0, p_mu, ls0, ls1);
        let c_wl = block_max_abs(&dense_h, w0, w1, ls0, ls1);
        assert!(
            c_ml > 1e-3,
            "#2387 (wiggle): μ↔logσ observed cross must be nonzero on this fixture, got {c_ml:.3e}"
        );
        assert!(
            c_wl > 1e-4,
            "#2387 (wiggle): wiggle↔logσ observed cross must be nonzero on this fixture, got {c_wl:.3e}"
        );

        // 1st-order ψ builder == central FD.
        let layout = test_design_hyper_layout(&derivative_blocks);
        let psi = family
            .exact_newton_joint_psi_terms(&states, &specs, &layout, 0)
            .expect("wiggle psi terms call")
            .expect("wiggle psi terms present");
        let h_psi = materialize(&psi.hessian_psi, psi.hessian_psi_operator.as_deref(), total);
        let h1 = 1e-5;
        let fd1 = (&dense_h_at(h1) - &dense_h_at(-h1)) / (2.0 * h1);
        assert_matrix_close(
            &h_psi,
            &fd1,
            1e-6,
            "#2387 (wiggle): 1st-order ψ joint Hessian must be the matrix FD of the dense observed H",
        );

        // 2nd-order ψ builder == second central FD.
        let psi2 = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &layout, 0, 0)
            .expect("wiggle psi 2nd-order call")
            .expect("wiggle psi 2nd-order present");
        let h_psi2 = materialize(
            &psi2.hessian_psi_psi,
            psi2.hessian_psi_psi_operator.as_deref(),
            total,
        );
        let h2 = 1e-3;
        let fd2 = (&(&dense_h_at(h2) + &dense_h_at(-h2)) - &(2.0 * &dense_h)) / (h2 * h2);
        assert_matrix_close(
            &h_psi2,
            &fd2,
            5e-4,
            "#2387 (wiggle): 2nd-order ψ joint Hessian must be the second matrix FD of the dense observed H",
        );

        // Mixed β·ψ builder == central FD in β of the 1st-order ψ-Hessian.
        // The β perturbation moves ALL THREE blocks; η_w is rebuilt from the
        // perturbed q0 AND the perturbed wiggle coefficients.
        let d_beta = Array1::from_shape_fn(total, |i| 0.04 + 0.1 * ((i + 1) as f64).cos());
        let psi_h_at_beta = |t: f64| -> Array2<f64> {
            let mut st = states.clone();
            for j in 0..p_mu {
                st[0].beta[j] += t * d_beta[j];
            }
            for j in 0..p_ls {
                st[1].beta[j] += t * d_beta[p_mu + j];
            }
            for j in 0..p_w {
                st[2].beta[j] += t * d_beta[w0 + j];
            }
            st[0].eta = xmu.dot(&st[0].beta);
            st[1].eta = xls.dot(&st[1].beta);
            st[2].eta = family
                .wiggle_design(st[0].eta.view())
                .expect("perturbed-β wiggle basis at q0")
                .dot(&st[2].beta);
            let psi_t = family
                .exact_newton_joint_psi_terms(&st, &specs, &layout, 0)
                .expect("perturbed-β wiggle psi terms call")
                .expect("perturbed-β wiggle psi terms present");
            materialize(&psi_t.hessian_psi, psi_t.hessian_psi_operator.as_deref(), total)
        };
        let mixed = family
            .exact_newton_joint_psihessian_directional_derivative(&states, &specs, &layout, 0, &d_beta)
            .expect("wiggle psi mixed-drift call")
            .expect("wiggle psi mixed-drift present");
        assert_eq!(mixed.dim(), (total, total));
        let hb = 1e-5;
        let fdm = (&psi_h_at_beta(hb) - &psi_h_at_beta(-hb)) / (2.0 * hb);
        assert_matrix_close(
            &mixed,
            &fdm,
            1e-5,
            "#2387 (wiggle): mixed β·ψ ψ-Hessian must be the β-directional FD of the 1st-order ψ-Hessian",
        );
    }
}

#[test]
pub(crate) fn gls_wiggle_joint_loglik_gradient_matches_finite_difference_and_legacy_does_not_2621() {
    let (family, states, specs, xmu, xls, xw_seed) = gls_wiggle_workspace_fixture();
    let n = family.y.len();
    let p_mu = states[0].beta.len();
    let p_ls = states[1].beta.len();
    let pw = states[2].beta.len();
    let total = p_mu + p_ls + pw;

    // Rebuild the three-block state from coefficients, re-evaluating the
    // q0-dependent wiggle basis. A frozen basis would drop the basis-drift
    // channel from the FD reference and the FD would agree with a gradient
    // that is wrong for the model the fit actually solves.
    let rebuild = |beta_mu: &Array1<f64>,
                   beta_ls: &Array1<f64>,
                   beta_w: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        let eta_mu = xmu.dot(beta_mu);
        let eta_ls = xls.dot(beta_ls);
        let eta_w = family
            .wiggle_design(eta_mu.view())
            .expect("wiggle basis at q0")
            .dot(beta_w);
        vec![
            ParameterBlockState {
                beta: beta_mu.clone(),
                eta: eta_mu,
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: beta_w.clone(),
                eta: eta_w,
            },
        ]
    };

    let loglik = |beta_mu: &Array1<f64>, beta_ls: &Array1<f64>, beta_w: &Array1<f64>| -> f64 {
        family
            .log_likelihood_only(&rebuild(beta_mu, beta_ls, beta_w))
            .expect("wiggle log likelihood")
    };

    // One measurement point: the analytic hook, the legacy working-set
    // assembly, and a central finite difference of the family's own log
    // likelihood, all at the same coefficients. Returns the worst per-coordinate
    // absolute error of each candidate against the FD, scaled by the FD's own
    // magnitude, plus the same quantity restricted to the scale block and to the
    // wiggle block.
    struct GradientAudit {
        exact: f64,
        legacy: f64,
        legacy_scale_block: f64,
        legacy_wiggle_block: f64,
    }

    let audit = |beta_mu: &Array1<f64>, beta_ls: &Array1<f64>, beta_w: &Array1<f64>| -> GradientAudit {
        let point = rebuild(beta_mu, beta_ls, beta_w);
        let evaluation = family
            .exact_newton_joint_gradient_evaluation(&point, &specs)
            .expect("joint gradient evaluation")
            .expect("this family must now serve its own joint score");
        assert_eq!(
            evaluation.gradient.len(),
            total,
            "joint score must span all three blocks"
        );
        let value = loglik(beta_mu, beta_ls, beta_w);
        assert!(
            (evaluation.log_likelihood - value).abs() <= 1e-12 * (1.0 + value.abs()),
            "the hook's log likelihood must be the family's own: got {:.12e}, expected {:.12e}",
            evaluation.log_likelihood,
            value,
        );

        let eps = 1e-6;
        let mut fd = Array1::<f64>::zeros(total);
        for j in 0..total {
            let bump = |sign: f64| -> f64 {
                let mut mu = beta_mu.clone();
                let mut ls = beta_ls.clone();
                let mut w = beta_w.clone();
                if j < p_mu {
                    mu[j] += sign * eps;
                } else if j < p_mu + p_ls {
                    ls[j - p_mu] += sign * eps;
                } else {
                    w[j - p_mu - p_ls] += sign * eps;
                }
                loglik(&mu, &ls, &w)
            };
            fd[j] = (bump(1.0) - bump(-1.0)) / (2.0 * eps);
        }

        // The legacy working-set assembly, reconstructed here from the family's
        // own `evaluate` exactly as `exact_newton_joint_gradient_from_eval` does.
        let eval = family.evaluate(&point).expect("wiggle evaluate");
        let designs = [&xmu, &xls, &xw_seed];
        let mut legacy = Array1::<f64>::zeros(total);
        let mut offset = 0usize;
        for block in 0..3 {
            let width = point[block].beta.len();
            match &eval.blockworking_sets[block] {
                BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                } => {
                    let weighted = Array1::from_shape_fn(n, |i| {
                        working_weights[i] * (working_response[i] - point[block].eta[i])
                    });
                    legacy
                        .slice_mut(s![offset..offset + width])
                        .assign(&designs[block].t().dot(&weighted));
                }
                BlockWorkingSet::ExactNewton { gradient, .. } => {
                    legacy.slice_mut(s![offset..offset + width]).assign(gradient);
                }
                BlockWorkingSet::NaturalDiagonal { score, .. } => {
                    legacy
                        .slice_mut(s![offset..offset + width])
                        .assign(&designs[block].t().dot(score));
                }
            }
            offset += width;
        }

        let scale = 1.0 + fd.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = |candidate: &Array1<f64>, lo: usize, hi: usize| -> (usize, f64) {
            (lo..hi).fold((lo, 0.0_f64), |(arg, worst), j| {
                let err = (candidate[j] - fd[j]).abs() / scale;
                if err > worst { (j, err) } else { (arg, worst) }
            })
        };

        let (exact_arg, exact_err) = worst(&evaluation.gradient, 0, total);
        assert!(
            exact_err <= 5e-6,
            "the single-source joint score must be the derivative of the family's own log \
             likelihood: worst coordinate {exact_arg} of {total} is off by {exact_err:.3e} \
             (relative to FD scale {scale:.6e}); analytic={:.12e} fd={:.12e}",
            evaluation.gradient[exact_arg],
            fd[exact_arg],
        );

        GradientAudit {
            exact: exact_err,
            legacy: worst(&legacy, 0, total).1,
            legacy_scale_block: worst(&legacy, p_mu, p_mu + p_ls).1,
            legacy_wiggle_block: worst(&legacy, p_mu + p_ls, total).1,
        }
    };

    // (1) The shared fixture's own point. `beta_w` there is O(0.05), so
    // `dq_dq0` is within 1e-2 of 1 and the two Gauss-Seidel offsets nearly
    // cancel: the legacy gradient is only ~5e-4 wrong here. That is why this
    // point alone cannot be the non-vacuity gate.
    let mild = audit(&states[0].beta, &states[1].beta, &states[2].beta);

    // (2) A stiff point in the regime the production refusals sit in: a wiggle
    // amplitude an order up and a mean an order up, so `dq_dq0` departs from 1
    // and the two halves of the shared linear predictor are both large. This is
    // where the legacy assembly's missing `dq_dq0` factor and its static seed
    // basis both bite.
    let stiff_mu = states[0].beta.mapv(|v| v * 6.0);
    let stiff_ls = states[1].beta.clone();
    let stiff_w = states[2].beta.mapv(|v| v * 12.0);
    let stiff = audit(&stiff_mu, &stiff_ls, &stiff_w);

    // Non-vacuity: the legacy assembly this hook replaces must be measurably
    // wrong, or the gate proves nothing. Reported alongside the analytic error
    // at the same point so the separation is legible rather than asserted.
    assert!(
        stiff.legacy > 1e-2,
        "the legacy working-set assembly is expected to DISAGREE with the finite \
         difference at the stiff point — if it now agrees, the Gauss-Seidel working \
         responses have changed and this gate no longer measures #2621: legacy is off \
         by {:.3e} where the single-source score is off by {:.3e} (mild point: legacy \
         {:.3e}, exact {:.3e})",
        stiff.legacy,
        stiff.exact,
        mild.legacy,
        mild.exact,
    );

    // And the error is concentrated on the wiggle block — the block that shares
    // the mean's linear predictor — not on the independently-channelled scale
    // block, which is what localized the fault in the first place.
    assert!(
        stiff.legacy_scale_block < stiff.legacy_wiggle_block,
        "the legacy error must be concentrated on the wiggle block, not on the \
         independently-channelled scale block: scale block {:.3e} vs wiggle block {:.3e}",
        stiff.legacy_scale_block,
        stiff.legacy_wiggle_block,
    );
}

/// #932. `GaussianLocationScaleWiggleFamily` reads its predictor-space row tower from
/// the generated `gaussian_normalized_row` atom, then pulls it back through the warp
/// `q = q₀ + Σ_j βw_j B_j(q₀)` by hand: `gls_wiggle_first_directional_coeffs`,
/// `gls_wiggle_second_directional_coeffs` and the dense `_from_designs` blocks. Nothing
/// compared that pullback with the likelihood it differentiates. This test takes exact
/// nested num-dual derivatives in β of
/// `f(β) = Σ_i w_i (½ (y_i − q_i)² / σ_i² + log σ_i)`, `σ = LOGB_SIGMA_FLOOR + e^{η_ls}`,
/// and requires the dense observed Hessian and its first and second directional
/// derivatives to match to rounding.
///
/// The warp enters as its Taylor polynomial about the base index `q₀*`:
/// `Σ_{k≤3} B_j^(k)(q₀*) δ^k / k!` per column, plus the base-coefficient term
/// `(Σ_j βw*_j B_j''''(q₀*)) δ⁴ / 24 = d⁴q/dq₀⁴ · δ⁴ / 24`. Every compared channel is at
/// most a fourth β-derivative and `βw` enters linearly, so `(βw − βw*)·δ⁴` is fifth
/// order and the polynomial is exact for every channel under test. The basis tables are
/// the production primitives; what this pins is the algebra built on them.
#[test]
pub(crate) fn gaussian_wiggle_joint_hessian_and_directional_derivatives_match_exact_derivatives_932()
{
    use num_dual::{Dual2, DualNum, HyperDual, second_derivative, second_partial_derivative};

    struct WarpTables {
        q0: Array1<f64>,
        basis: Array2<f64>,
        d1: Array2<f64>,
        d2: Array2<f64>,
        d3: Array2<f64>,
        d4q: Array1<f64>,
    }

    fn warped_negative_log_likelihood<D: DualNum<f64> + Copy>(
        family: &GaussianLocationScaleWiggleFamily,
        xmu: &Array2<f64>,
        xls: &Array2<f64>,
        warp: &WarpTables,
        coefficients: &[D],
    ) -> D {
        let p_mu = xmu.ncols();
        let p_ls = xls.ncols();
        let p_w = warp.basis.ncols();
        let half = D::from(0.5);
        let mut total = D::zero();
        for i in 0..family.y.len() {
            let mut q0 = D::zero();
            for j in 0..p_mu {
                q0 += D::from(xmu[[i, j]]) * coefficients[j];
            }
            let delta = q0 - D::from(warp.q0[i]);
            let delta2 = delta * delta;
            let delta3 = delta2 * delta;
            let mut q = q0 + D::from(warp.d4q[i] / 24.0) * delta2 * delta2;
            for j in 0..p_w {
                let column = D::from(warp.basis[[i, j]])
                    + D::from(warp.d1[[i, j]]) * delta
                    + D::from(0.5 * warp.d2[[i, j]]) * delta2
                    + D::from(warp.d3[[i, j]] / 6.0) * delta3;
                q += coefficients[p_mu + p_ls + j] * column;
            }
            let mut eta_ls = D::zero();
            for j in 0..p_ls {
                eta_ls += D::from(xls[[i, j]]) * coefficients[p_mu + j];
            }
            let sigma = D::from(crate::sigma_link::LOGB_SIGMA_FLOOR) + eta_ls.exp();
            let residual = D::from(family.y[i]) - q;
            total += D::from(family.weights[i])
                * (half * residual * residual / (sigma * sigma) + sigma.ln());
        }
        total
    }

    let (family, states, _specs, xmu, xls, _xw_seed) = gls_wiggle_workspace_fixture();
    let beta: Vec<f64> = states
        .iter()
        .flat_map(|state| state.beta.iter().copied())
        .collect();
    let total = beta.len();
    let p_mu = states[GaussianLocationScaleWiggleFamily::BLOCK_MU].beta.len();
    let p_ls = states[GaussianLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
        .beta
        .len();
    let q0 = states[GaussianLocationScaleWiggleFamily::BLOCK_MU].eta.clone();
    let geometry = family
        .wiggle_geometry(
            q0.view(),
            states[GaussianLocationScaleWiggleFamily::BLOCK_WIGGLE]
                .beta
                .view(),
        )
        .expect("wiggle geometry at the base index");
    let warp = WarpTables {
        q0,
        basis: geometry.basis.clone(),
        d1: geometry.basis_d1.clone(),
        d2: geometry.basis_d2.clone(),
        d3: geometry.basis_d3.clone(),
        d4q: geometry.d4q_dq04.clone(),
    };
    let u = Array1::from_shape_fn(total, |k| 0.3 * ((k as f64) * 0.71 + 0.2).sin());
    let v = Array1::from_shape_fn(total, |k| -0.25 * ((k as f64) * 0.43 + 0.9).cos());

    // One outer mixed partial in `(s, t)` evaluates each Hessian entry at
    // `β + s·u + t·v`, returning `(H_ab, D H[u]_ab, D H[v]_ab, D² H[u, v]_ab)`.
    let lift = |value: f64| HyperDual::<f64, f64>::from(value);
    let mut hessian = Array2::<f64>::zeros((total, total));
    let mut hessian_u = Array2::<f64>::zeros((total, total));
    let mut hessian_uv = Array2::<f64>::zeros((total, total));
    for a in 0..total {
        for b in a..total {
            let (entry, entry_u, _, entry_uv) = second_partial_derivative(
                |(s, t): (HyperDual<f64, f64>, HyperDual<f64, f64>)| {
                    let point: Vec<HyperDual<f64, f64>> = (0..total)
                        .map(|k| lift(beta[k]) + s * lift(u[k]) + t * lift(v[k]))
                        .collect();
                    if a == b {
                        second_derivative(
                            |x: Dual2<HyperDual<f64, f64>, f64>| {
                                let mut coefficients: Vec<Dual2<HyperDual<f64, f64>, f64>> =
                                    point.iter().map(|&value| Dual2::from_re(value)).collect();
                                coefficients[a] = x;
                                warped_negative_log_likelihood(
                                    &family,
                                    &xmu,
                                    &xls,
                                    &warp,
                                    &coefficients,
                                )
                            },
                            point[a],
                        )
                        .2
                    } else {
                        second_partial_derivative(
                            |(x, other): (
                                HyperDual<HyperDual<f64, f64>, f64>,
                                HyperDual<HyperDual<f64, f64>, f64>,
                            )| {
                                let mut coefficients: Vec<HyperDual<HyperDual<f64, f64>, f64>> =
                                    point.iter().map(|&value| HyperDual::from_re(value)).collect();
                                coefficients[a] = x;
                                coefficients[b] = other;
                                warped_negative_log_likelihood(
                                    &family,
                                    &xmu,
                                    &xls,
                                    &warp,
                                    &coefficients,
                                )
                            },
                            (point[a], point[b]),
                        )
                        .3
                    }
                },
                (0.0, 0.0),
            );
            for (matrix, value) in [
                (&mut hessian, entry),
                (&mut hessian_u, entry_u),
                (&mut hessian_uv, entry_uv),
            ] {
                matrix[[a, b]] = value;
                matrix[[b, a]] = value;
            }
        }
    }

    let produced = family
        .exact_newton_joint_hessian(&states)
        .expect("dense wiggle joint Hessian")
        .expect("dense wiggle joint Hessian present");
    let produced_u = family
        .exact_newton_joint_hessian_directional_derivative(&states, &u)
        .expect("wiggle dH[u]")
        .expect("wiggle dH[u] present");
    let produced_uv = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &u, &v)
        .expect("wiggle d2H[u, v]")
        .expect("wiggle d2H[u, v] present");
    assert_eq!(produced.dim(), (total, total));
    for a in 0..total {
        for b in 0..total {
            for (label, got, want) in [
                ("H", produced[[a, b]], hessian[[a, b]]),
                ("dH[u]", produced_u[[a, b]], hessian_u[[a, b]]),
                ("d2H[u,v]", produced_uv[[a, b]], hessian_uv[[a, b]]),
            ] {
                let tolerance = 1.0e-10 * got.abs().max(want.abs()).max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "{label}[{a},{b}]: production {got:+.17e} against the exact derivative {want:+.17e}"
                );
            }
        }
    }

    // The pullback under test lives in the wiggle blocks; they must carry curvature
    // or the comparison above passes on zeros.
    let block_peak = |matrix: &Array2<f64>, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>| {
        let mut peak = 0.0_f64;
        for r in rows {
            for c in cols.clone() {
                peak = peak.max(matrix[[r, c]].abs());
            }
        }
        peak
    };
    for (label, matrix) in [("dH[u]", &hessian_u), ("d2H[u,v]", &hessian_uv)] {
        let mean_wiggle = block_peak(matrix, 0..p_mu, p_mu + p_ls..total);
        let scale_wiggle = block_peak(matrix, p_mu..p_mu + p_ls, p_mu + p_ls..total);
        assert!(
            mean_wiggle > 1.0e-6 && scale_wiggle > 1.0e-6,
            "the exact {label} must carry mean×wiggle ({mean_wiggle:.3e}) and \
             scale×wiggle ({scale_wiggle:.3e}) curvature, or the pullback is compared on zeros"
        );
    }
}

