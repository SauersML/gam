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

#[test]
pub(crate) fn poisson_extreme_eta_uses_exact_exp_and_refuses_only_unrepresentable_geometry() {
    use crate::custom_family::{CustomFamily, ParameterBlockState};
    let poisson = PoissonLogFamily {
        y: Array1::from_vec(vec![1.0, 2.0, 3.0]),
        weights: Array1::from_vec(vec![1.0, 1.0, 1.0]),
    };
    let extreme_eta = Array1::from_vec(vec![0.5, 709.0, -0.3]);
    let eval_result = poisson.evaluate(&[ParameterBlockState {
        beta: Array1::zeros(0),
        eta: extreme_eta,
    }]);
    let eval =
        eval_result.expect("Poisson evaluate must succeed while exact geometry is representable");
    match &eval.blockworking_sets[0] {
        crate::custom_family::BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            let all_finite = working_response.iter().all(|v| v.is_finite())
                && working_weights.iter().all(|v| v.is_finite())
                && eval.log_likelihood.is_finite();
            assert!(
                all_finite,
                "Poisson evaluate should produce finite outputs for all eta, \
                     but got non-finite values: ll={}, z={:?}, w={:?}",
                eval.log_likelihood, working_response, working_weights
            );
        }
        _ => panic!("expected Diagonal block"),
    }

    let refused = match poisson.evaluate(&[ParameterBlockState {
        beta: Array1::zeros(0),
        eta: Array1::from_vec(vec![0.5, 710.0, -0.3]),
    }]) {
        Ok(_) => panic!("overflowing exact exp geometry must be refused"),
        Err(err) => err,
    };
    assert!(refused.contains("row 1"), "unexpected refusal: {refused}");
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

// =====================================================================
// #2621 — the Gaussian location-scale-WIGGLE joint log-likelihood gradient
//
// This family emits `BlockWorkingSet::Diagonal` working sets and supplies an
// exact joint HESSIAN. Before this gate it supplied no joint GRADIENT, so
// `load_joint_gradient_evaluation` fell through to the legacy assembly, which
// recovers a per-block score from the IRLS working set as
// `X_bᵀ(w ⊙ (z_b − η_b))`. For this family that is wrong twice over:
//
//   * `z_mu = response − η_w` and `z_wiggle = response − η_mu` are GAUSS–SEIDEL
//     working responses — each of the two location-channel blocks absorbs the
//     other's η as an offset. That is correct for a blockwise solve and is not
//     the gradient of the joint likelihood;
//   * the family's predictor is `q(q0, βw) = q0 + B(q0)·βw`, so the μ block
//     reaches `q` through `dq_dq0 = 1 + B'(q0)·βw`, and the wiggle block's
//     design is `B(q0)` at the CURRENT `q0`, not the spec's static seed grid.
//     The legacy assembly has neither factor.
//
// A per-block directional audit of the legacy gradient on the failing fixtures
// measured realized-`Δℓ`-over-predicted-`grad·δ` ratios of `+0.2585` on μ and
// `−6.5539` (wrong SIGN) on the wiggle block, with the independently-channelled
// scale block exact at `+1.000000`. So the two blocks that are two halves of
// one linear predictor were the two that disagreed.
//
// The gate asserts BOTH directions, because a fix that is not gated on
// re-detecting its own symptom silently passes once the symptom moves:
//   (a) the hook's gradient matches a central finite difference of the family's
//       OWN `log_likelihood_only`, per coordinate, with the wiggle basis
//       refreshed at the perturbed `q0` exactly as `refresh_all_block_etas`
//       does at fit time;
//   (b) the legacy working-set assembly does NOT — otherwise this test would
//       be green on a tree where nothing had been fixed.
