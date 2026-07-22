// Child module of `gamlss::tests` (see the `#[path]` declaration there):
// wiggle-family FD gates, binomial location-scale expected-info and release
// cells, NB dispersion convergence, and the zz2155 mode-geography probes.
// Split out of tests.rs for the source-file length budget only.
use super::*;

/// Build the nontrivial-design BLS Wiggle family + designs + wiggle block
/// shared by the FD-gradient and FD-joint-Hessian tests below.
pub(crate) fn wiggle_nontrivial_fixture() -> (
    BinomialLocationScaleWiggleFamily,
    DesignMatrix,
    DesignMatrix,
    ParameterBlockInput,
    Array1<f64>,
    Array1<f64>,
) {
    let n = 9usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let t_grid = Array1::linspace(0.0, 1.0, n);
    let threshold_x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => 1.0,
        1 => t_grid[i] - 0.5,
        2 => (2.0 * std::f64::consts::PI * t_grid[i]).sin(),
        _ => unreachable!(),
    });
    let log_sigma_x = Array2::from_shape_fn((n, 2), |(i, j)| match j {
        0 => 1.0,
        1 => (3.0 * std::f64::consts::PI * t_grid[i]).cos(),
        _ => unreachable!(),
    });
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        threshold_x.clone(),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        log_sigma_x.clone(),
    ));
    let q_seed = Array1::linspace(-1.3, 1.1, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    (
        family,
        threshold_design,
        log_sigma_design,
        wiggle_block,
        y,
        weights,
    )
}

/// Rebuild the three-block state for the nontrivial-design wiggle fixture.
pub(crate) fn rebuild_wiggle_nontrivial_states(
    family: &BinomialLocationScaleWiggleFamily,
    threshold_design: &DesignMatrix,
    log_sigma_design: &DesignMatrix,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    beta_t: &Array1<f64>,
    beta_ls: &Array1<f64>,
    betaw: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    let eta_t = threshold_design.matrixvectormultiply(beta_t);
    let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
    let core_q0 =
        binomial_location_scale_core(y, weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let etaw = family
        .wiggle_design(core_q0.q0.view())
        .expect("wiggle design")
        .dot(betaw);
    vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw,
        },
    ]
}

/// Extract the exact-Newton gradient for one block of a wiggle evaluation.
pub(crate) fn extract_wiggle_gradient(eval: &FamilyEvaluation, block_idx: usize) -> Array1<f64> {
    match &eval.blockworking_sets[block_idx] {
        BlockWorkingSet::ExactNewton {
            gradient,
            hessian: _,
        } => gradient.clone(),
        BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
    }
}

#[test]
pub(crate) fn wiggle_familygradients_match_finite_differencewith_nontrivial_designs() {
    let (family, threshold_design, log_sigma_design, wiggle_block, y, weights) =
        wiggle_nontrivial_fixture();

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        rebuild_wiggle_nontrivial_states(
            &family,
            &threshold_design,
            &log_sigma_design,
            &y,
            &weights,
            beta_t,
            beta_ls,
            betaw,
        )
    };

    let objective = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>, betaw: &Array1<f64>| {
        let states = rebuild_states(beta_t, beta_ls, betaw);
        -family.evaluate(&states).expect("evaluate").log_likelihood
    };

    let extractgradient = extract_wiggle_gradient;

    let beta_t = Array1::from_vec(vec![0.15, -0.3, 0.2]);
    let beta_ls = Array1::from_vec(vec![-0.2, 0.1]);
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);
    let eval = family.evaluate(&states).expect("evaluate");
    let eps = 1e-6;

    for block_idx in 0..3 {
        let analytic = extractgradient(&eval, block_idx);
        let mut fd = Array1::<f64>::zeros(analytic.len());
        for j in 0..analytic.len() {
            let mut beta_t_plus = beta_t.clone();
            let mut beta_ls_plus = beta_ls.clone();
            let mut betaw_plus = betaw.clone();
            let mut beta_t_minus = beta_t.clone();
            let mut beta_ls_minus = beta_ls.clone();
            let mut betaw_minus = betaw.clone();
            match block_idx {
                BinomialLocationScaleWiggleFamily::BLOCK_T => {
                    beta_t_plus[j] += eps;
                    beta_t_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                    beta_ls_plus[j] += eps;
                    beta_ls_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                    betaw_plus[j] += eps;
                    betaw_minus[j] -= eps;
                }
                _ => unreachable!(),
            }
            let f_plus = objective(&beta_t_plus, &beta_ls_plus, &betaw_plus);
            let f_minus = objective(&beta_t_minus, &beta_ls_minus, &betaw_minus);
            fd[j] = (f_plus - f_minus) / (2.0 * eps);
        }
        gam_test_support::assert_matrix_derivativefd(
            &fd.insert_axis(Axis(1)),
            &(-&analytic).insert_axis(Axis(1)),
            2e-4,
            &format!("wiggle block {block_idx} score"),
        );
    }
}

#[test]
pub(crate) fn wiggle_family_joint_hessian_matches_fd_gradients_with_nontrivial_designs() {
    let (family, threshold_design, log_sigma_design, wiggle_block, y, weights) =
        wiggle_nontrivial_fixture();

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        rebuild_wiggle_nontrivial_states(
            &family,
            &threshold_design,
            &log_sigma_design,
            &y,
            &weights,
            beta_t,
            beta_ls,
            betaw,
        )
    };

    let extractgradient = extract_wiggle_gradient;

    let beta_t = Array1::from_vec(vec![0.15, -0.3, 0.2]);
    let beta_ls = Array1::from_vec(vec![-0.2, 0.1]);
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);
    let h_joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let pt = beta_t.len();
    let pls = beta_ls.len();
    let eps = 1e-6;
    let total = pt + pls + betaw.len();
    let mut fd = Array2::<f64>::zeros((total, total));
    let source_offsets = [0usize, pt, pt + pls];

    for source_block in 0..3 {
        let source_len = states[source_block].beta.len();
        for j in 0..source_len {
            let mut beta_t_plus = beta_t.clone();
            let mut beta_ls_plus = beta_ls.clone();
            let mut betaw_plus = betaw.clone();
            let mut beta_t_minus = beta_t.clone();
            let mut beta_ls_minus = beta_ls.clone();
            let mut betaw_minus = betaw.clone();
            match source_block {
                BinomialLocationScaleWiggleFamily::BLOCK_T => {
                    beta_t_plus[j] += eps;
                    beta_t_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                    beta_ls_plus[j] += eps;
                    beta_ls_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                    betaw_plus[j] += eps;
                    betaw_minus[j] -= eps;
                }
                _ => unreachable!(),
            }
            let eval_plus = family
                .evaluate(&rebuild_states(&beta_t_plus, &beta_ls_plus, &betaw_plus))
                .expect("eval plus");
            let eval_minus = family
                .evaluate(&rebuild_states(&beta_t_minus, &beta_ls_minus, &betaw_minus))
                .expect("eval minus");

            let mut row_offset = 0usize;
            for target_block in 0..3 {
                let grad_plus = extractgradient(&eval_plus, target_block);
                let grad_minus = extractgradient(&eval_minus, target_block);
                let col = (&grad_plus - &grad_minus).mapv(|v| -v / (2.0 * eps));
                let col_idx = source_offsets[source_block] + j;
                fd.slice_mut(s![
                    row_offset..row_offset + grad_plus.len(),
                    col_idx..col_idx + 1
                ])
                .assign(&col.insert_axis(Axis(1)));
                row_offset += grad_plus.len();
            }
        }
    }

    gam_test_support::assert_matrix_derivativefd(&fd, &h_joint, 4e-4, "wiggle joint hessian");
}

#[test]
pub(crate) fn wiggle_family_joint_exacthessian_directional_derivative_matches_finite_difference() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let eta_t = threshold_design.matrixvectormultiply(&beta_t);
    let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);
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
            beta: betaw.clone(),
            eta: etaw,
        },
    ];

    let base_h = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let direction = Array1::ones(base_h.nrows());
    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("joint dH")
        .expect("expected joint exact dH");

    let eps = 1e-6;
    let mut plus_states = states.clone();
    let beta_layout = GamlssBetaLayout::withwiggle(
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T]
            .beta
            .len(),
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
            .beta
            .len(),
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE]
            .beta
            .len(),
    );
    let (dir_t, dir_ls, dirw) = beta_layout
        .split_three(&direction, "wiggle test direction split")
        .expect("split wiggle test direction");
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta =
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta + &(eps * dir_t);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta =
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta + &(eps * dir_ls);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta =
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta + &(eps * dirw);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta = threshold_design
        .matrixvectormultiply(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta = log_sigma_design
        .matrixvectormultiply(
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta,
        );
    let plus_core_q0 = binomial_location_scale_core(
        &y,
        &weights,
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta,
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta,
        None,
        &family.link_kind,
    )
    .expect("plus core q0");
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta = family
        .wiggle_design(plus_core_q0.q0.view())
        .expect("plus wiggle design")
        .dot(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta);

    let h_plus = family
        .exact_newton_joint_hessian(&plus_states)
        .expect("plus joint hessian")
        .expect("expected plus joint hessian");
    let fd = (h_plus - base_h) / eps;
    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 2e-3, "joint dH");
}

#[test]
pub(crate) fn wiggle_family_joint_exacthessiansecond_directional_derivative_matches_finite_difference()
 {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 4, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 4,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        let eta_t = threshold_design.matrixvectormultiply(beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
        let core_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let etaw = family
            .wiggle_design(core_q0.q0.view())
            .expect("wiggle design")
            .dot(betaw);
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t,
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ]
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let betaw = Array1::from_vec(vec![0.03; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);

    let pt = beta_t.len();
    let pls = beta_ls.len();
    let pw = betaw.len();
    let total = pt + pls + pw;
    let direction_u = Array1::from_shape_fn(total, |k| 0.2 + 0.1 * (k as f64));
    let directionv = Array1::from_shape_fn(total, |k| -0.15 + 0.07 * (k as f64));

    let analytic = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &direction_u, &directionv)
        .expect("joint d2H")
        .expect("expected joint exact d2H");

    let eps = 1e-6;
    let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
    let (step_t, step_ls, stepw) = beta_layout
        .split_three(&directionv, "wiggle d2H test directionv")
        .expect("split wiggle test direction");

    let states_plus = rebuild_states(
        &(&beta_t + &(eps * &step_t)),
        &(&beta_ls + &(eps * &step_ls)),
        &(&betaw + &(eps * &stepw)),
    );
    let states_minus = rebuild_states(
        &(&beta_t - &(eps * &step_t)),
        &(&beta_ls - &(eps * &step_ls)),
        &(&betaw - &(eps * &stepw)),
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

    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 4e-3, "joint d2H");
}

#[test]
pub(crate) fn wiggle_family_joint_hessian_cross_blocks_match_finite_difference_of_gradients() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        let eta_t = threshold_design.matrixvectormultiply(beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
        let core_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let etaw = family
            .wiggle_design(core_q0.q0.view())
            .expect("wiggle design")
            .dot(betaw);
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t,
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ]
    };

    let extractgradient = |eval: &FamilyEvaluation, block_idx: usize| -> Array1<f64> {
        match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton {
                gradient,
                hessian: _,
            } => gradient.clone(),
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
        }
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);

    let h_joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");

    let pt = beta_t.len();
    let pls = beta_ls.len();
    let pw = betaw.len();
    let eps = 1e-6;

    let fd_cross_block = |target_block: usize, source_block: usize| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((
            states[target_block].beta.len(),
            states[source_block].beta.len(),
        ));
        for j in 0..states[source_block].beta.len() {
            let mut beta_t_plus = beta_t.clone();
            let mut beta_ls_plus = beta_ls.clone();
            let mut betaw_plus = betaw.clone();
            let mut beta_t_minus = beta_t.clone();
            let mut beta_ls_minus = beta_ls.clone();
            let mut betaw_minus = betaw.clone();
            match source_block {
                BinomialLocationScaleWiggleFamily::BLOCK_T => {
                    beta_t_plus[j] += eps;
                    beta_t_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                    beta_ls_plus[j] += eps;
                    beta_ls_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                    betaw_plus[j] += eps;
                    betaw_minus[j] -= eps;
                }
                _ => panic!("unexpected block"),
            }

            let eval_plus = family
                .evaluate(&rebuild_states(&beta_t_plus, &beta_ls_plus, &betaw_plus))
                .expect("eval plus");
            let eval_minus = family
                .evaluate(&rebuild_states(&beta_t_minus, &beta_ls_minus, &betaw_minus))
                .expect("eval minus");
            let grad_plus = extractgradient(&eval_plus, target_block);
            let grad_minus = extractgradient(&eval_minus, target_block);
            let col = (&grad_plus - &grad_minus).mapv(|v| -v / (2.0 * eps));
            out.slice_mut(ndarray::s![.., j]).assign(&col);
        }
        out
    };

    let fd_t_ls = fd_cross_block(
        BinomialLocationScaleWiggleFamily::BLOCK_T,
        BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
    );
    let fd_tw = fd_cross_block(
        BinomialLocationScaleWiggleFamily::BLOCK_T,
        BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
    );
    let fd_lsw = fd_cross_block(
        BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
        BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
    );

    let h_t_ls = h_joint.slice(ndarray::s![0..pt, pt..pt + pls]).to_owned();
    let h_tw = h_joint
        .slice(ndarray::s![0..pt, pt + pls..pt + pls + pw])
        .to_owned();
    let h_lsw = h_joint
        .slice(ndarray::s![pt..pt + pls, pt + pls..pt + pls + pw])
        .to_owned();

    gam_test_support::assert_matrix_derivativefd(&fd_t_ls, &h_t_ls, 2e-4, "H_t_ls");
    gam_test_support::assert_matrix_derivativefd(&fd_tw, &h_tw, 4e-4, "H_tw");
    gam_test_support::assert_matrix_derivativefd(&fd_lsw, &h_lsw, 6e-4, "H_lsw");
}

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
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
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
pub(crate) fn wiggle_basis_is_structurally_monotone_for_nonnegative_coefficients() {
    let q_seed = Array1::linspace(-2.0, 2.0, 17);
    let degree = 3usize;
    let num_internal_knots = 6usize;
    let penalty_order = 2usize;

    let (block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
        q_seed.view(),
        degree,
        num_internal_knots,
        penalty_order,
        false,
    )
    .expect("wiggle block");
    let design = match &block.design {
        DesignMatrix::Dense(x) => x.to_dense_arc(),
        DesignMatrix::Sparse(_) => panic!("expected dense wiggle design"),
    };
    let beta = Array1::from_elem(design.ncols(), 0.2);
    let derivative = monotone_wiggle_basis_with_derivative_order(q_seed.view(), &knots, degree, 1)
        .expect("wiggle derivative basis")
        .dot(&beta);
    assert!(
        derivative.iter().all(|&value| value >= -1e-12),
        "I-spline wiggle derivative must stay non-negative for non-negative coefficients: min={}",
        derivative.iter().fold(f64::INFINITY, |acc, &v| acc.min(v))
    );
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
pub(crate) fn wiggle_block_design_matches_ispline_basis() {
    let q_seed = Array1::linspace(-1.0, 1.0, 11);
    let degree = 2usize;
    let num_internal_knots = 4usize;
    let penalty_order = 2usize;

    let (block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
        q_seed.view(),
        degree,
        num_internal_knots,
        penalty_order,
        false,
    )
    .expect("wiggle block");
    let (basis, _) = create_basis::<Dense>(
        q_seed.view(),
        KnotSource::Provided(knots.view()),
        monotone_wiggle_internal_degree(degree).expect("wiggle degree"),
        BasisOptions::i_spline(),
    )
    .expect("I-spline basis");
    let expected = (*basis).clone();

    let got = match &block.design {
        DesignMatrix::Dense(x) => x.to_dense_arc(),
        DesignMatrix::Sparse(_) => panic!("expected dense wiggle design"),
    };
    assert_eq!(got.dim(), expected.dim());
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            assert!(
                (got[[i, j]] - expected[[i, j]]).abs() < 1e-10,
                "wiggle design mismatch at ({}, {}): got {}, expected {}",
                i,
                j,
                got[[i, j]],
                expected[[i, j]]
            );
        }
    }
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
pub(crate) fn wiggle_geometry_and_generative_use_same_sigma_link_as_core() {
    let n = 8usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let eta_t = Array1::from_vec(vec![0.5, -0.6, 0.1, -0.3, 0.9, -0.2, 0.4, -0.8]);
    let eta_ls = Array1::from_vec(vec![-2.5, -1.5, -0.5, 0.0, 0.7, 1.4, 2.2, 3.0]);

    let q_seed = Array1::linspace(-1.5, 1.5, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");

    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: None,
        log_sigma_design: None,
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.15; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);

    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_t.clone(),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_ls.clone(),
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw.clone(),
        },
    ];

    let wigglespec = wiggle_block
        .clone()
        .intospec("wiggle")
        .expect("wiggle spec");
    let (geom_x, _) = family
        .block_geometry(&states, &wigglespec)
        .expect("block geometry");
    let geom = match geom_x {
        DesignMatrix::Dense(x) => x.to_dense(),
        DesignMatrix::Sparse(_) => panic!("expected dense wiggle geometry design"),
    };
    let expected_geom = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("expected wiggle geometry");
    assert_eq!(geom.dim(), expected_geom.dim());
    for i in 0..geom.nrows() {
        for j in 0..geom.ncols() {
            assert!(
                (geom[[i, j]] - expected_geom[[i, j]]).abs() < 1e-12,
                "geometry mismatch at ({i}, {j}): got {}, expected {}",
                geom[[i, j]],
                expected_geom[[i, j]]
            );
        }
    }

    let generated = family.generativespec(&states).expect("generative spec");
    let core = binomial_location_scale_core(
        &y,
        &weights,
        &eta_t,
        &eta_ls,
        Some(&etaw),
        &family.link_kind,
    )
    .expect("core with wiggle");
    for i in 0..n {
        assert!(
            (generated.mean[i] - core.mu[i]).abs() < 1e-7,
            "wiggle mean mismatch at {i}: got {}, expected {}",
            generated.mean[i],
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
                |_axis: &Array1<f64>| Ok(None),
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
                cached_row_scalars: std::sync::RwLock::new(None),
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

/// #932 release speed gate for the binomial location-scale WIGGLE order-two
/// row program: the production typed-probe lowering
/// (`wiggle_order2_rows`, ONE `Order2<4>` evaluation per row with 8
/// coefficient channels, cost linear in the basis width `pw`) must beat the
/// naive per-(row, column) dense-tower assembly (`pw` independent `Tower2<3>`
/// compositions per row — the generic shape the oracle below uses as its
/// witness, and the only alternative representation since the pre-cutover
/// hand ladder was deleted by the #932 single-source migration). Both sides
/// consume the same block states and build their own bases, so this is a
/// system-level race of everything the joint-Hessian consumer needs. Emits
/// the harness-parsed `hand_over_production` token
/// (`per_column_tower_ns / production_ns`); the MSI release harness fails
/// closed on any cell `<= 1`.
#[test]
pub(crate) fn release_measure_bls_wiggle_order2_rows_vs_per_column_tower_932() {
    use super::super::binomial_q_derivs::binomial_neglog_q_derivatives_dispatch;
    use gam_math::jet_tower::Tower2;
    use std::time::Instant;

    let (family, states, _specs, _xt, _xls, _wd) = bls_wiggle_workspace_fixture();

    let production_batch = |states: &[ParameterBlockState]| -> f64 {
        let pieces = family
            .wiggle_order2_rows(states)
            .expect("production wiggle order-two rows");
        let n = pieces.coeff_tt.len();
        pieces.coeff_tt[0] + pieces.coeff_tl[0] + pieces.coeff_ww[n - 1] + pieces.coeff_tw_b[0]
    };

    // The naive generic assembly: per (row, column) an independent Tower2<3>
    // over (eta_t, eta_ls, betaw_j), composed exactly as the oracle below.
    let generic_batch = |states: &[ParameterBlockState]| -> f64 {
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
        let b0 = family
            .wiggle_basiswith_options(core0.q0.view(), BasisOptions::value())
            .expect("wiggle value basis");
        let d0 = family
            .wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())
            .expect("wiggle first-derivative basis");
        let dd0 = family
            .wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())
            .expect("wiggle second-derivative basis");
        let n = family.y.len();
        let pw = b0.ncols();
        let mut folded = 0.0;
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
                folded += nll.h[0][0] + nll.h[0][2] + nll.h[2][2];
            }
        }
        folded
    };

    // Sanity: both batches produce finite folds on the fixture.
    assert!(production_batch(&states).is_finite());
    assert!(generic_batch(&states).is_finite());

    // Feedback-coupled timing barrier (no `std::hint::black_box`): the first
    // threshold eta is nudged by a negligible multiple of the running
    // checksum, so neither whole-batch call can be hoisted or dropped.
    let base_eta = states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta[0];
    let best_ns = |use_generic: bool| -> f64 {
        let iterations = 60usize;
        let mut work = states.clone();
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut checksum = 0.0_f64;
            let started = Instant::now();
            for _ in 0..iterations {
                work[BinomialLocationScaleWiggleFamily::BLOCK_T].eta[0] =
                    base_eta + checksum * 1e-18;
                checksum += if use_generic {
                    generic_batch(&work)
                } else {
                    production_batch(&work)
                };
            }
            assert!(
                checksum.is_finite(),
                "wiggle order-two release-measure checksum must stay finite"
            );
            best = best.min(started.elapsed().as_secs_f64());
        }
        best * 1e9 / iterations as f64
    };

    let production_ns = best_ns(false);
    let generic_ns = best_ns(true);
    eprintln!(
        "BLS-WIGGLE-ORDER2-932 production={production_ns:.2} ns/batch \
         per_column_tower={generic_ns:.2} ns/batch hand_over_production={:.6}",
        generic_ns / production_ns,
    );
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

    let (probit_family, states, _specs, _xt, _xls, _wd) = bls_wiggle_workspace_fixture();
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
