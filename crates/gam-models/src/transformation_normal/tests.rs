#![cfg(test)]
use super::*;
use crate::custom_family::custom_family_outer_derivatives;
use gam_test_support::{assert_matrix_derivativefd, assert_matrix_derivativefd_rel};
use ndarray::array;

fn test_design_hyper_layout(
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
) -> CustomFamilyHyperLayout {
    let axis_count = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    CustomFamilyHyperLayout::new(
        derivative_blocks.to_vec(),
        Vec::new(),
        Array1::zeros(axis_count),
    )
    .expect("test CTN hyper layout")
}

#[test]
pub(crate) fn exact_ctn_mode_branch_anchors_on_the_accepted_iterate_2765() {
    let warm = |value: f64| {
        CustomFamilyWarmStart::from_cached_beta(&[1], &array![value])
            .expect("one-coefficient CTN mode seed")
    };
    let anchor_beta = |candidates: &[Option<CustomFamilyWarmStart>]| {
        candidates
            .first()
            .and_then(Option::as_ref)
            .expect("a compatible anchor mode")
            .block_beta_view(0)
            .expect("one CTN coefficient")[0]
    };
    let rho = Array1::zeros(0);
    let value_only = gam_problem::EvalMode::ValueOnly;
    let with_gradient = gam_problem::EvalMode::ValueAndGradient;

    // No mode exists yet: the only candidate is a cold solve.
    let mut state = ExactCoefficientModeBranch::default();
    let (first_iterate, candidates) = state.candidates(value_only, &rho);
    assert!(!first_iterate);
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].is_none());

    // Before any iterate is accepted, value-only seed probes carry their
    // converged mode forward; a probe that did not converge leaves no trace.
    state.record_value(value_only, warm(1.0), true);
    state.record_value(value_only, warm(9.0), false);
    let (_, candidates) = state.candidates(value_only, &rho);
    assert_eq!(candidates.len(), 1, "one start per evaluation, never a cold solve beside it");
    assert_eq!(anchor_beta(&candidates), 1.0);
    state.record_value(value_only, warm(2.0), true);

    // The first derivative-bearing evaluation is an accepted iterate: it is
    // solved from the carried mode and announces itself once.
    let (first_iterate, candidates) = state.candidates(with_gradient, &rho);
    assert!(first_iterate);
    assert_eq!(anchor_beta(&candidates), 2.0);
    state.record_value(with_gradient, warm(3.0), true);
    let (first_iterate, _) = state.candidates(with_gradient, &rho);
    assert!(!first_iterate, "the announcement is made exactly once");

    // Line-search probes start from the accepted iterate's mode and cannot
    // replace it, whether they converge or not, so the value at a trial θ is
    // a function of θ and the iterate — not of the probe order.
    state.record_value(value_only, warm(4.0), true);
    state.record_value(value_only, warm(5.0), false);
    let (_, candidates) = state.candidates(value_only, &rho);
    assert_eq!(
        anchor_beta(&candidates),
        3.0,
        "a value-only probe must not replace the accepted iterate's mode"
    );
    assert!(
        !state.install_seed(warm(6.0)),
        "an outer-cache seed must not displace the mode this walk certified"
    );

    // The next accepted iterate moves the anchor with the walk; a
    // derivative-bearing evaluation that did not converge does not.
    state.record_value(with_gradient, warm(7.0), false);
    let (_, candidates) = state.candidates(value_only, &rho);
    assert_eq!(anchor_beta(&candidates), 3.0);
    state.record_value(with_gradient, warm(8.0), true);
    let (_, candidates) = state.candidates(value_only, &rho);
    assert_eq!(anchor_beta(&candidates), 8.0);

    // A branch that has never seen a mode solves cold at its first iterate.
    let mut cold = ExactCoefficientModeBranch::default();
    let (first_iterate, candidates) = cold.candidates(with_gradient, &rho);
    assert!(first_iterate);
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].is_none());
}

pub(crate) fn dense_first_order_psi_hessian(terms: &ExactNewtonJointPsiTerms) -> Array2<f64> {
    if terms.hessian_psi.nrows() > 0 {
        terms.hessian_psi.clone()
    } else {
        terms
            .hessian_psi_operator
            .as_ref()
            .expect("CTN psi first-order terms must expose either dense Hessian or operator")
            .to_dense()
    }
}

#[test]
pub(crate) fn ctn_penalty_scale_seed_uses_likelihood_to_penalty_ratio() {
    let penalties = vec![
        PenaltyMatrix::Dense(array![[2.0, 0.0], [0.0, 2.0]]),
        PenaltyMatrix::Dense(array![[4.0, 0.0], [0.0, 4.0]]),
    ];
    let rho = ctn_penalty_scale_log_lambdas(&penalties, 8.0);
    assert!((rho[0] - 4.0_f64.ln()).abs() < 1.0e-12);
    assert!((rho[1] - 2.0_f64.ln()).abs() < 1.0e-12);
}

#[test]
pub(crate) fn prebuilt_ctn_family_uses_explicit_rho_without_reseeding() {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let config = toy_scop_ctn_config();
    let (val_basis, deriv_basis, response_penalties, knots, transform) =
        build_response_basis(&response, &config).expect("toy response basis builds");
    let weights = Array1::ones(response.len());
    let offset = Array1::zeros(response.len());
    let covariate = array![[1.0], [1.0], [1.0], [1.0]];
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        response_penalties,
        knots,
        config.response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(covariate)),
        vec![],
        &config,
        None,
    )
    .expect("prebuilt CTN family");

    let derived = family
        .penalty_scale_log_lambdas()
        .expect("data-scaled smoothing seed");
    let rho_dim = derived.len();
    assert!(rho_dim > 0, "fixture must carry a smoothing coordinate");
    let explicit = Array1::from_iter((0..rho_dim).map(|index| -0.75 + 0.125 * index as f64));
    let supplied = family
        .block_spec(&explicit)
        .expect("explicit-rho coefficient block");
    assert!(beta_bits_match(
        &supplied.initial_log_lambdas,
        &explicit,
    ));
    assert!(beta_bits_match(
        supplied.initial_beta.as_ref().expect("initial beta"),
        &family.initial_beta,
    ));

    let wrong_length = match family.block_spec(&Array1::zeros(rho_dim + 1)) {
        Ok(_) => panic!("wrong-length explicit rho must fail"),
        Err(error) => error,
    };
    assert!(wrong_length.contains("smoothing vector has length"));

    let mut non_finite = explicit;
    non_finite[0] = f64::NAN;
    let non_finite = match family.block_spec(&non_finite) {
        Ok(_) => panic!("non-finite explicit rho must fail"),
        Err(error) => error,
    };
    assert!(non_finite.contains("invalid transformation smoothing strength"));
}

#[test]
pub(crate) fn tensor_psi_penalty_derivatives_carry_response_mass_gram_layout() {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());
    // The covariate-direction penalty is `G_y ⊗ S_{x,j}(κ)`, so every ψ/κ
    // derivative component lifts through the κ-independent response value-basis
    // mass Gram `G_y = Vᵀ W V` (gam#2306). Compute it from the same basis and
    // weights the family sees.
    let expected_g_resp =
        weighted_function_gram(val_basis.view(), weights.view(), p_resp, "response")
            .expect("response mass gram");
    let cov_design = array![[1.0, 0.2], [1.0, -0.1], [1.0, 0.4], [1.0, -0.3]];
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        vec![],
        knots,
        toy_scop_ctn_config().response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(cov_design.clone())),
        vec![],
        &toy_scop_ctn_config(),
        None,
    )
    .expect("toy transformation family");

    let ds0 = array![[1.0, 0.25], [0.25, 2.0]];
    let ds1 = array![[3.0, -0.5], [-0.5, 4.0]];
    let ds1_second = array![[5.0, 0.75], [0.75, 6.0]];
    let mut cov_deriv = CustomFamilyBlockPsiDerivative::new(
        None,
        Array2::zeros((response.len(), cov_design.ncols())),
        Array2::zeros((0, 0)),
        Some(vec![(0, ds0.clone()), (1, ds1.clone())]),
        None,
        None,
        Some(vec![vec![(1, ds1_second.clone())]]),
    );
    cov_deriv.s_psi_penalty_components = Some(vec![
        (0, PenaltyMatrix::Dense(ds0.clone())),
        (1, PenaltyMatrix::Dense(ds1.clone())),
    ]);
    cov_deriv.s_psi_psi_penalty_components =
        Some(vec![vec![(1, PenaltyMatrix::Dense(ds1_second.clone()))]]);

    let tensor_derivs =
        build_tensor_psi_derivatives(&family, &[cov_deriv]).expect("tensor derivatives");
    let first = tensor_derivs[0]
        .s_psi_penalty_components
        .as_ref()
        .expect("first derivatives");
    let got_indices: Vec<usize> = first.iter().map(|(idx, _)| *idx).collect();
    assert_eq!(got_indices, vec![0, 1]);
    assert_covariate_penalty_component(&first[0].1, &expected_g_resp, &ds0);
    assert_covariate_penalty_component(&first[1].1, &expected_g_resp, &ds1);

    let second = tensor_derivs[0]
        .s_psi_psi_penalty_components
        .as_ref()
        .expect("second derivatives");
    assert_eq!(second.len(), 1);
    let got_second_indices: Vec<usize> = second[0].iter().map(|(idx, _)| *idx).collect();
    assert_eq!(got_second_indices, vec![1]);
    assert_covariate_penalty_component(&second[0][0].1, &expected_g_resp, &ds1_second);
}

/// The CTN tensor penalty list is assembled as `[covariate.., response.., double]`
/// and the layout records that order load-bearingly (the psi-derivative channel
/// addresses the G_x-bearing response/double penalties by `n_cov+i`). This pins
/// the order so an innocent future reorder cannot silently desync the
/// κ-derivatives (gam#2306).
#[test]
pub(crate) fn ctn_tensor_penalty_layout_orders_covariate_response_double() {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let config = TransformationNormalConfig {
        double_penalty: true,
        response_degree: 1,
        response_num_internal_knots: 2,
        ..TransformationNormalConfig::default()
    };
    let (val_basis, deriv_basis, response_penalties, knots, transform) =
        build_response_basis(&response, &config).expect("response basis builds");
    let n_response = response_penalties.len();
    assert!(n_response >= 1, "toy config must carry a response roughness penalty");
    let p_shape = val_basis.ncols() - 1;
    let affine = affine_shape_direction(knots.view(), config.response_degree, p_shape)
        .expect("affine shape direction");
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());
    let cov_design = array![[1.0, 0.2], [1.0, -0.1], [1.0, 0.4], [1.0, -0.3]];
    let p_cov = cov_design.ncols();
    let s_cov = array![[2.0, -1.0], [-1.0, 2.0]];
    let g_cov = weighted_function_gram(cov_design.view(), weights.view(), p_cov, "covariate")
        .expect("covariate mass gram");

    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        response_penalties,
        knots,
        config.response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
        vec![PenaltyMatrix::Dense(s_cov.clone())],
        &config,
        None,
    )
    .expect("toy transformation family");

    let layout = family.tensor_penalty_layout;
    assert_eq!(layout.n_covariate, 1);
    assert_eq!(layout.n_response, n_response);
    assert!(layout.has_double);
    assert_eq!(layout.total(), family.tensor_penalties.len());
    assert_eq!(layout.response_indices(), 1..1 + n_response);
    let double_index = 1 + n_response;

    // Covariate penalty (index 0) carries S_x on the right.
    let PenaltyMatrix::KroneckerFactored { right, .. } = &family.tensor_penalties[0] else {
        panic!("covariate penalty must be Kronecker-factored");
    };
    assert_eq!(right, &s_cov);

    // Every response penalty carries the covariate mass Gram G_x on the right.
    for r in layout.response_indices() {
        let PenaltyMatrix::KroneckerFactored { right, .. } = &family.tensor_penalties[r] else {
            panic!("response penalty must be Kronecker-factored");
        };
        for ((i, j), &value) in right.indexed_iter() {
            assert!(
                (value - g_cov[[i, j]]).abs() <= 1e-12 * (1.0 + g_cov[[i, j]].abs()),
                "response penalty {r} right factor must be G_x"
            );
        }
    }

    // The double penalty is the full-rank shape-row ridge shape_resp ⊗ I_cov:
    // its covariate factor MUST be the identity (not the rank-deficient G_x), so
    // it pins weakly-identified shape×covariate directions (no rank_deficient_H_pen).
    let PenaltyMatrix::KroneckerFactored { left, right } =
        &family.tensor_penalties[double_index]
    else {
        panic!("double penalty must be Kronecker-factored");
    };
    for ((i, j), &value) in right.indexed_iter() {
        let want: f64 = if i == j { 1.0 } else { 0.0 };
        assert_eq!(value, want, "double penalty covariate factor must be identity");
    }
    // The shape-row ridge is the ORTHOGONAL PROJECTOR onto the shape rows minus
    // the affine direction, not the bare identity (gam#2600): a ridge that
    // reaches the affine direction cancels the order-2 roughness's own null and
    // makes `α = 0` — the constant transformation, where the likelihood is
    // undefined — the penalty's unique minimiser.
    let affine_norm_sq = affine.dot(&affine);
    for ((i, j), &value) in left.indexed_iter() {
        let identity: f64 = if i == j && i > 0 { 1.0 } else { 0.0 };
        let want = if i > 0 && j > 0 {
            identity - affine[i - 1] * affine[j - 1] / affine_norm_sq
        } else {
            identity
        };
        assert!(
            (value - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "double penalty left factor [{i},{j}] = {value} != shape-row projector {want}"
        );
    }
    let mut beta_affine = Array1::<f64>::zeros(left.nrows());
    beta_affine.slice_mut(s![1..]).assign(&affine);
    let ridge_at_affine = beta_affine.dot(&left.dot(&beta_affine));
    assert!(
        ridge_at_affine.abs() <= 1e-12 * affine_norm_sq,
        "the shape-row ridge must annihilate the affine transformation, got {ridge_at_affine:.6e}"
    );
}

/// First-order κ-derivative of the response-roughness penalty `S_y ⊗ G_x(κ)`
/// (assembled from the emitted `s_psi_penalty_components`) matches a central
/// finite difference of the assembled penalty, rebuilding Ψ(κ±h) per leg so the
/// covariate mass Gram `G_x` is re-evaluated each side (no frozen cache). This
/// is the desync guard for the moving-Ψ penalty channel (gam#2306).
#[test]
pub(crate) fn ctn_response_penalty_gx_first_order_kappa_derivative_matches_fd() {
    let psi = array![0.15, -0.10];
    let h = 1e-6;
    let (family, blocks, _state, _spec) =
        toy_family_and_derivatives_with_penalty_mode(&psi, true);
    let p_total = family.p_total();
    let n_pen = family.tensor_penalties.len();
    assert!(
        family.tensor_penalty_layout.n_response >= 1,
        "penalty-mode family must carry the G_x-bearing response penalty"
    );
    // Unit smoothing strengths: the penalty-derivative assembly scales each
    // component by lambda[idx]; matching lambdas on both sides isolates dG_x/dκ.
    let lambdas = Array1::<f64>::ones(n_pen);

    let analytic_first = |derivs: &[CustomFamilyBlockPsiDerivative], a: usize| -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        if let Some(components) = derivs[a].s_psi_penalty_components.as_ref() {
            for (idx, component) in components {
                s.scaled_add(lambdas[*idx], &component.to_dense());
            }
        }
        s
    };
    let assembled = |psi_eval: &Array1<f64>| -> Array2<f64> {
        let (f, _, _, _) = toy_family_and_derivatives_with_penalty_mode(psi_eval, true);
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        for (k, penalty) in f.tensor_penalties.iter().enumerate() {
            s.scaled_add(lambdas[k], &penalty.to_dense());
        }
        s
    };

    for a in 0..psi.len() {
        let mut psi_plus = psi.clone();
        psi_plus[a] += h;
        let mut psi_minus = psi.clone();
        psi_minus[a] -= h;
        let fd = (assembled(&psi_plus) - assembled(&psi_minus)) / (2.0 * h);
        let analytic = analytic_first(&blocks[0], a);
        assert_matrix_derivativefd(
            &fd,
            &analytic,
            2e-4,
            &format!("CTN response penalty dG_x/dkappa axis {a}"),
        );
    }
}

/// Second-order κ-derivative of `S_y ⊗ G_x(κ)` (assembled from
/// `s_psi_psi_penalty_components`) matches a central finite difference of the
/// first-order derivative, per-leg Ψ rebuild. Validates the `d²G_x/dκ²` channel
/// — including the `(∂Ψ/∂κ_a)ᵀW(∂Ψ/∂κ_j)` cross term — used by the outer
/// exact-Newton Hessian (gam#2306).
#[test]
pub(crate) fn ctn_response_penalty_gx_second_order_kappa_derivative_matches_fd() {
    let psi = array![0.15, -0.10];
    let h = 1e-5;
    let (family, _blocks, _state, _spec) =
        toy_family_and_derivatives_with_penalty_mode(&psi, true);
    let p_total = family.p_total();
    let n_pen = family.tensor_penalties.len();
    let lambdas = Array1::<f64>::ones(n_pen);

    let analytic_first_at = |psi_eval: &Array1<f64>, a: usize| -> Array2<f64> {
        let (_, blocks, _, _) = toy_family_and_derivatives_with_penalty_mode(psi_eval, true);
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        if let Some(components) = blocks[0][a].s_psi_penalty_components.as_ref() {
            for (idx, component) in components {
                s.scaled_add(lambdas[*idx], &component.to_dense());
            }
        }
        s
    };
    let analytic_second = |a: usize, j: usize| -> Array2<f64> {
        let (_, blocks, _, _) = toy_family_and_derivatives_with_penalty_mode(&psi, true);
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        if let Some(rows) = blocks[0][a].s_psi_psi_penalty_components.as_ref() {
            if let Some(pairs) = rows.get(j) {
                for (idx, component) in pairs {
                    s.scaled_add(lambdas[*idx], &component.to_dense());
                }
            }
        }
        s
    };

    for a in 0..psi.len() {
        for j in 0..psi.len() {
            let mut psi_plus = psi.clone();
            psi_plus[j] += h;
            let mut psi_minus = psi.clone();
            psi_minus[j] -= h;
            let fd =
                (analytic_first_at(&psi_plus, a) - analytic_first_at(&psi_minus, a)) / (2.0 * h);
            let analytic = analytic_second(a, j);
            assert_matrix_derivativefd(
                &fd,
                &analytic,
                3e-4,
                &format!("CTN response penalty d2G_x/dkappa axes ({a},{j})"),
            );
        }
    }
}

#[test]
pub(crate) fn tensor_psi_row_chunks_are_window_consistent() {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let (val_basis, deriv_basis, knots, transform, _) = toy_response_basis(&response);
    let psi = array![0.15, -0.10];
    let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(&psi);
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        vec![],
        knots,
        toy_scop_ctn_config().response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
        vec![],
        &toy_scop_ctn_config(),
        None,
    )
    .expect("toy transformation family");

    let tensor_derivs =
        build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor derivatives");
    let op = tensor_derivs[0]
        .implicit_operator
        .as_ref()
        .expect("tensor psi operator should be implicit");
    let mat_op = op
        .as_materializable()
        .expect("toy tensor psi operator should remain materializable for reference");
    let rows = 1..3;

    let first_dense = mat_op
        .materialize_first(0)
        .expect("dense first derivative reference");
    let first_chunk = op
        .row_chunk_first(0, rows.clone())
        .expect("chunked first derivative");
    assert_eq!(
        first_chunk,
        first_dense.slice(s![rows.clone(), ..]).to_owned()
    );

    let second_diag_full = op
        .row_chunk_second_diag(0, 0..op.n_data())
        .expect("full row-chunk second diagonal reference");
    let second_diag_chunk = op
        .row_chunk_second_diag(0, rows.clone())
        .expect("chunked second diagonal derivative");
    assert_eq!(
        second_diag_chunk,
        second_diag_full.slice(s![rows.clone(), ..]).to_owned()
    );

    let second_cross_full = op
        .row_chunk_second_cross(0, 1, 0..op.n_data())
        .expect("full row-chunk second cross reference");
    let second_cross_chunk = op
        .row_chunk_second_cross(0, 1, rows.clone())
        .expect("chunked second cross derivative");
    assert_eq!(
        second_cross_chunk,
        second_cross_full.slice(s![rows, ..]).to_owned()
    );
}

pub(crate) fn assert_covariate_penalty_component(
    penalty: &PenaltyMatrix,
    expected_left: &Array2<f64>,
    expected_right: &Array2<f64>,
) {
    let PenaltyMatrix::KroneckerFactored { left, right } = penalty else {
        panic!("expected KroneckerFactored penalty component");
    };
    assert_eq!(right, expected_right);
    assert_eq!(left.dim(), expected_left.dim());
    for ((r, c), &value) in left.indexed_iter() {
        let want = expected_left[[r, c]];
        assert!(
            (value - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "covariate penalty left factor [{r},{c}] = {value} != expected G_y {want}"
        );
    }
}

pub(crate) fn toy_covariate_design_and_derivs(
    psi: &Array1<f64>,
) -> (Array2<f64>, Vec<CustomFamilyBlockPsiDerivative>) {
    let x0 = array![[1.00, 0.40], [1.10, 0.35], [1.20, 0.45], [0.95, 0.50],];
    let x_a = array![[0.10, -0.02], [0.08, 0.01], [0.12, -0.01], [0.09, 0.03],];
    let x_b = array![[-0.04, 0.06], [-0.02, 0.05], [-0.03, 0.04], [-0.01, 0.07],];
    let x_aa = array![[0.02, 0.00], [0.01, 0.01], [0.02, -0.01], [0.01, 0.02],];
    let x_ab = array![[0.01, -0.01], [0.00, 0.02], [0.01, 0.01], [0.00, -0.01],];
    let x_bb = array![[-0.01, 0.02], [-0.02, 0.01], [-0.01, 0.00], [-0.02, 0.02],];
    let design = &x0
        + &(x_a.clone() * psi[0])
        + &(x_b.clone() * psi[1])
        + &(x_aa.clone() * (0.5 * psi[0] * psi[0]))
        + &(x_ab.clone() * (psi[0] * psi[1]))
        + &(x_bb.clone() * (0.5 * psi[1] * psi[1]));
    let d_a = &x_a + &(x_aa.clone() * psi[0]) + &(x_ab.clone() * psi[1]);
    let d_b = &x_b + &(x_ab.clone() * psi[0]) + &(x_bb.clone() * psi[1]);
    let deriv_a = CustomFamilyBlockPsiDerivative::new(
        None,
        d_a,
        Array2::zeros((0, 0)),
        None,
        Some(vec![x_aa.clone(), x_ab.clone()]),
        None,
        None,
    );
    let deriv_b = CustomFamilyBlockPsiDerivative::new(
        None,
        d_b,
        Array2::zeros((0, 0)),
        None,
        Some(vec![x_ab, x_bb]),
        None,
        None,
    );
    (design, vec![deriv_a, deriv_b])
}

/// Minimal SCOP-CTN config used by every toy fixture in this test module:
/// degree-1 I-splines on 2 internal knots produce the smallest valid
/// SCOP-CTN configuration (p_resp = 4 monotone basis columns).
pub(crate) fn toy_scop_ctn_config() -> TransformationNormalConfig {
    TransformationNormalConfig {
        double_penalty: false,
        response_degree: 1,
        response_num_internal_knots: 2,
        ..TransformationNormalConfig::default()
    }
}

/// Build (val, deriv, knots, transform, p_resp) from a real
/// `build_response_basis` call so test fixtures match the production
/// I-spline contract exactly.
pub(crate) fn toy_response_basis(
    response: &Array1<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array2<f64>, usize) {
    let config = toy_scop_ctn_config();
    let (val, deriv, _penalties, knots, transform) =
        build_response_basis(response, &config).expect("toy response basis builds");
    let p_resp = val.ncols();
    (val, deriv, knots, transform, p_resp)
}

/// Deterministic probe vector of length `p_total` used by tests that
/// previously hand-rolled p_total=4 arrays. Generated from a tiny PRNG so
/// each call with a different seed yields linearly-independent probes.
pub(crate) fn toy_probe_vector(p_total: usize, seed: u64) -> Array1<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    Array1::from_iter((0..p_total).map(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
        (bits - 0.5) * 0.8
    }))
}

fn toy_family_and_derivatives_with_penalty_mode(
    psi: &Array1<f64>,
    include_response_penalties: bool,
) -> (
    TransformationNormalFamily,
    Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ParameterBlockState,
    ParameterBlockSpec,
) {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let config = toy_scop_ctn_config();
    let (val_basis, deriv_basis, response_penalties, knots, transform) =
        build_response_basis(&response, &config).expect("toy response basis builds");
    let p_resp = val_basis.ncols();
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());
    let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(psi);
    let p_cov = cov_design.ncols();
    let p_total = p_resp * p_cov;
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        if include_response_penalties {
            response_penalties
        } else {
            vec![]
        },
        knots,
        toy_scop_ctn_config().response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
        vec![],
        &toy_scop_ctn_config(),
        None,
    )
    .expect("toy transformation family");
    let derivative_blocks =
        vec![build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor psi derivs")];
    // Positive alpha across the response axis with mild covariate variation so
    // h' = (M ⊗_row B_cov)·β stays strictly positive on every row (M-splines
    // are non-negative; the toy covariate design is positive-valued).
    let mut beta_vec = Vec::with_capacity(p_total);
    for k in 0..p_resp {
        let base = 0.6 + 0.05 * k as f64;
        for j in 0..p_cov {
            if j == 0 {
                beta_vec.push(base);
            } else {
                beta_vec.push(0.05 + 0.02 * k as f64 * (j as f64));
            }
        }
    }
    let beta = Array1::from(beta_vec);
    assert_eq!(beta.len(), p_total);
    let h_prime = family.x_deriv_kron.forward_mul(&beta);
    assert!(
        h_prime.iter().all(|v| *v > 0.25),
        "toy beta must keep h' positive, got {h_prime:?}"
    );
    let state = ParameterBlockState {
        beta,
        eta: Array1::zeros(h_prime.len()),
    };
    let rho0 = family
        .penalty_scale_log_lambdas()
        .expect("toy smoothing seed");
    let spec = family.block_spec(&rho0).expect("toy coefficient block");
    (family, derivative_blocks, state, spec)
}

pub(crate) fn toy_family_and_derivatives(
    psi: &Array1<f64>,
) -> (
    TransformationNormalFamily,
    Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ParameterBlockState,
    ParameterBlockSpec,
) {
    toy_family_and_derivatives_with_penalty_mode(psi, false)
}

#[test]
pub(crate) fn direct_alpha_ctn_exposes_exact_factored_monotonicity_cone() {
    let (family, _, state, spec) = toy_family_and_derivatives(&array![0.15, -0.10]);
    let constraints = family
        .block_linear_constraints(std::slice::from_ref(&state), 0, &spec)
        .expect("CTN monotonicity constraints build")
        .expect("direct-alpha CTN must constrain its shape rows");
    let gam_problem::ConstraintSet::KhatriRaoCone(cone) = constraints else {
        panic!("direct-alpha CTN must keep the large cone factored");
    };

    let p_resp = family.response_val_basis.ncols();
    let p_cov = family.covariate_design.ncols();
    assert_eq!(cone.p_left(), p_resp);
    assert_eq!(cone.coupled_rows(), &(1..p_resp).collect::<Vec<_>>());
    assert_eq!(cone.ncols(), spec.design.ncols());
    assert_eq!(cone.nrows(), family.n_obs() * (p_resp - 1));
    assert_eq!(cone.factor().nrows(), family.n_obs());
    assert_eq!(cone.factor().ncols(), p_cov);
    assert!(
        cone.values(state.beta.view())
            .expect("feasible toy alpha values")
            .iter()
            .all(|value| *value > 0.0),
        "the production warm-start shape field must be strictly feasible"
    );

    let mut location_only = Array1::<f64>::zeros(spec.design.ncols());
    location_only[0] = -100.0;
    assert!(
        cone.values(location_only.view())
            .expect("unconstrained location values")
            .iter()
            .all(|value| *value == 0.0),
        "response row zero is the unconstrained location field"
    );

    let mut invalid_shape = Array1::<f64>::zeros(spec.design.ncols());
    invalid_shape[p_cov] = -1.0;
    assert!(
        cone.values(invalid_shape.view())
            .expect("invalid shape values")
            .iter()
            .any(|value| *value < 0.0),
        "a negative realized shape field must violate the cone"
    );
}

#[test]
pub(crate) fn ctn_row_quantity_cache_matches_direct_formulas() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let row = family
        .row_quantities(&state.beta)
        .expect("toy row quantities");
    // Direct-alpha SCOP-CTN is affine in the coefficient block.
    let direct_h = family.x_val_kron.forward_mul(&state.beta)
        + family.offset.as_ref()
        + family.response_floor_offset.as_ref();
    let direct_h_prime = family
        .x_deriv_kron
        .forward_mul(&state.beta)
        .mapv(|hp| hp + TRANSFORMATION_MONOTONICITY_EPS);
    let weights = family.weights.as_ref();

    for i in 0..direct_h.len() {
        assert!(
            (row.h[i] - direct_h[i]).abs() <= 1.0e-14,
            "h[{i}] mismatch: cached={} direct={}",
            row.h[i],
            direct_h[i]
        );
        assert!(
            (row.h_prime[i] - direct_h_prime[i]).abs() <= 1.0e-14,
            "h_prime[{i}] mismatch: cached={} direct={}",
            row.h_prime[i],
            direct_h_prime[i]
        );
    }

    let mut expected_ll = 0.0;
    for i in 0..direct_h.len() {
        let hp = direct_h_prime[i];
        // gam#2600: the row density is the most-likely-transformation density
        // log φ(h) + log h', with NO renormalization by the endpoint mass. log φ
        // carries the −½ln(2π) constant, which `build_transformation_row_derived`
        // includes deliberately so the reported absolute log-likelihood (and AIC)
        // is comparable to mlt/tram; it is coefficient-independent.
        expected_ll += weights[i]
            * (-0.5 * direct_h[i] * direct_h[i] - 0.5 * (2.0 * std::f64::consts::PI).ln()
                + hp.ln());
    }

    assert!(
        (row.log_likelihood - expected_ll).abs() <= 1.0e-14,
        "cached log-likelihood={} expected={expected_ll}",
        row.log_likelihood
    );
}

#[test]
pub(crate) fn transformation_normal_pit_score_is_the_model_cdf_2600() {
    // The fitted CDF is `F = Φ(h)`, so the PIT score is `Φ⁻¹(F) = h` — no
    // endpoints anywhere in it, and in particular no dependence on where the
    // fitted knot range happened to land.
    let center = transformation_normal_pit_score(0.0, 1.0e-12).expect("symmetric PIT score");
    assert_eq!(center, 0.0);
    for h in [-3.25, -0.5, 0.75, 2.5] {
        assert_eq!(transformation_normal_pit_score(h, 1.0e-12).unwrap(), h);
    }

    // A far-tail transform value used to be a typed `OutsideCertifiedDomain`
    // refusal whenever it left `[lower, upper]`, because the CONDITIONAL PIT
    // saturates to exactly 0/1 there and a clamped answer would have been
    // fabricated. Under `F = Φ(h)` it is an ordinary extreme probability, and
    // the clip window is the only thing that bounds the reported score.
    let clip = 1.0e-12;
    let lower = standard_normal_quantile(clip).expect("lower clip quantile");
    let upper = standard_normal_quantile(1.0 - clip).expect("upper clip quantile");
    for h in [37.5, -37.5, 1.0e5] {
        let score = transformation_normal_pit_score(h, clip)
            .expect("an out-of-range response is a probability, not a refusal");
        assert_eq!(score, h.clamp(lower, upper));
    }

    // Genuinely-malformed input (NaN h) is still rejected by the early
    // `is_finite()` guard.
    let nan_err =
        transformation_normal_pit_score(f64::NAN, 1.0e-12).expect_err("NaN h must be rejected");
    assert!(nan_err.contains("finite"));
}

#[test]
pub(crate) fn ctn_row_quantity_cache_is_exact_beta_keyed() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let row_a = family
        .row_quantities(&state.beta)
        .expect("first row quantity build");
    let row_a_again = family
        .row_quantities(&state.beta)
        .expect("same beta row quantity lookup");
    assert!(Arc::ptr_eq(&row_a.h, &row_a_again.h));
    assert!(Arc::ptr_eq(&row_a.h_prime, &row_a_again.h_prime));

    let mut beta_b = state.beta.clone();
    beta_b[0] += 0.125;
    let row_b = family
        .row_quantities(&beta_b)
        .expect("updated beta row quantity build");
    assert!(!Arc::ptr_eq(&row_a.h, &row_b.h));
    assert!(row_b.matches_beta(&beta_b));
    assert!(!row_b.matches_beta(&state.beta));
    assert!(
        row_a
            .h
            .iter()
            .zip(row_b.h.iter())
            .any(|(&left, &right)| left.to_bits() != right.to_bits())
    );

    let row_b_again = family
        .row_quantities(&beta_b)
        .expect("updated beta row quantity lookup");
    assert!(Arc::ptr_eq(&row_b.h, &row_b_again.h));
}

#[test]
pub(crate) fn ctn_row_quantities_reject_nonrepresentable_exact_derivatives() {
    let h = array![0.0];
    let h_prime = array![1.0e-100];
    let weights = array![1.0];
    let err = build_transformation_row_derived(&h, &h_prime, &weights)
        .expect_err("1/h'^4 overflows f64 and must not be clamped");
    assert!(
        err.contains("1/h'^4") && err.contains("outside the finite exact-derivative range"),
        "unexpected error: {err}"
    );
}

#[test]
pub(crate) fn transformation_normal_uses_compact_gaussian_outer_seeding() {
    let psi = array![0.15, -0.10];
    let (family, _, _, _) = toy_family_and_derivatives(&psi);
    let seed_config = family.outer_seed_config(6);
    assert_eq!(seed_config.bounds, (-12.0, 12.0));
    assert_eq!(seed_config.max_seeds, 1);
    assert_eq!(seed_config.seed_budget, 1);
    assert_eq!(seed_config.screen_max_inner_iterations, 2);
    assert_eq!(
        seed_config.risk_profile,
        gam_solve::seeding::SeedRiskProfile::Gaussian
    );
    assert_eq!(seed_config.num_auxiliary_trailing, 0);
}

#[test]
pub(crate) fn max_feasible_step_size_delegates_to_the_factored_cone() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let p_total = state.beta.len();
    let mut delta = toy_probe_vector(p_total, 0xDE17A);
    delta[0] = -0.30;

    let block_states = vec![state.clone()];
    let alpha_prod = family
        .max_feasible_step_size(&block_states, 0, &delta)
        .expect("toy max_feasible_step_size returns Ok");
    assert_eq!(alpha_prod, None);

    let bad_delta = Array1::<f64>::zeros(p_total + 1);
    assert!(
        family
            .max_feasible_step_size(&block_states, 0, &bad_delta)
            .is_err(),
        "dimension mismatches should still be rejected before line search"
    );
}

#[test]
pub(crate) fn warm_start_absorbs_offset_into_affine_seed() {
    // The direct-alpha warm start is built directly in coefficient space: choose a
    // positive constant shape seed for h', subtract its induced value
    // contribution, then solve the unconstrained location row. The fixed
    // monotonicity floor is part of h, so the value target includes
    // ε(y-median) and the derivative target includes ε.
    let response = array![2.0, 3.0, 4.0, 5.0];
    let (val_basis, deriv_basis, knots, transform, _p_resp) = toy_response_basis(&response);
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::from_elem(response.len(), 0.7);
    let cov_rows = response.len();
    let covariate_design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem(
        (cov_rows, 1),
        1.0,
    )));
    let warm_start = TransformationWarmStart {
        location: Array1::from_elem(response.len(), 1.0),
        scale: Array1::from_elem(response.len(), 2.0),
    };
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        vec![],
        knots,
        toy_scop_ctn_config().response_degree,
        transform,
        &weights,
        &offset,
        covariate_design,
        vec![],
        &toy_scop_ctn_config(),
        Some(&warm_start),
    )
    .expect("transformation family");

    let row = family
        .row_quantities(&family.initial_beta)
        .expect("row quantities at initial beta");
    let h = row.h.as_ref();
    let h_prime = row.h_prime.as_ref();
    // expected_h[i] = (response[i] - location)/scale = (y - 1)/2.
    let expected_h: Array1<f64> = response.mapv(|y| {
        (y - 1.0) / 2.0 + TRANSFORMATION_MONOTONICITY_EPS * (y - family.response_median())
    });
    let expected_h_prime = Array1::from_elem(response.len(), 0.5 + TRANSFORMATION_MONOTONICITY_EPS);

    for i in 0..expected_h.len() {
        assert!(
            (h[i] - expected_h[i]).abs() < 1e-9,
            "h[{i}] mismatch: got {}, expected {}",
            h[i],
            expected_h[i]
        );
        assert!(
            (h_prime[i] - expected_h_prime[i]).abs() < 1e-9,
            "h_prime[{i}] mismatch: got {}, expected {}",
            h_prime[i],
            expected_h_prime[i]
        );
    }

    assert_eq!(response.len(), family.n_obs());
}

#[test]
pub(crate) fn kronecker_dense_fast_paths_match_dense_materialization() {
    let left = array![[1.0, -0.4], [0.5, 0.3], [-0.2, 0.9], [1.1, -0.7],];
    let right = array![
        [0.2, 1.0, -0.3],
        [0.4, -0.5, 0.8],
        [0.7, 0.1, 0.6],
        [-0.2, 0.9, 0.5],
    ];
    let weights = array![0.7, 0.0, 0.9, 1.2];
    let v = array![0.6, -0.3, 0.5, 0.8];
    let kron = KroneckerDesign::new_khatri_rao(
        &left,
        DesignMatrix::Dense(DenseDesignMatrix::from(right.clone())),
    )
    .expect("kronecker design");

    let dense = dense_rowwise_kronecker(left.view(), right.view());
    let expected_transpose = dense.t().dot(&v);
    let expected_gram = fast_atb(&weight_rows(&dense, &weights), &dense);

    let got_transpose = kron.transpose_mul(&v);
    let got_gram = kron
        .weighted_gram(&weights, &ResourcePolicy::default_library())
        .unwrap();
    let got_diagonal_mean = kron
        .weighted_gram_diagonal_mean(&weights, &ResourcePolicy::default_library())
        .unwrap();
    let expected_diagonal_mean = matrix_diag_mean_abs(&expected_gram);

    let transpose_err = (&got_transpose - &expected_transpose)
        .iter()
        .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    let gram_err = (&got_gram - &expected_gram)
        .iter()
        .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(
        transpose_err < 1e-10,
        "Kronecker transpose fast path mismatch: max_abs={transpose_err}"
    );
    assert!(
        gram_err < 1e-10,
        "Kronecker weighted Gram fast path mismatch: max_abs={gram_err}"
    );
    assert!(
        (got_diagonal_mean - expected_diagonal_mean).abs() < 1e-12,
        "Kronecker weighted Gram diagonal mean mismatch: got={got_diagonal_mean}, expected={expected_diagonal_mean}"
    );
}

/// Strongly non-Gaussian (heavy right-skew, exponential-shaped) response so
/// the data-driven complexity cap in `effective_response_num_internal_knots`
/// is non-binding and the structural sample/tensor caps remain the gate.
pub(crate) fn skewed_response(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| {
        let u = (i as f64 + 0.5) / n as f64;
        // Inverse-CDF of a unit exponential: skewness 2, excess kurtosis 6,
        // so the complexity budget saturates well above the structural caps.
        -(1.0 - u).ln()
    }))
}

#[test]
pub(crate) fn large_samples_allow_richer_response_basis_than_small_samples() {
    let config = TransformationNormalConfig::default();
    let small_resp = skewed_response(40);
    let large_resp = skewed_response(4000);
    let small = effective_response_num_internal_knots(&config, 40, 20, small_resp.view());
    let large = effective_response_num_internal_knots(&config, 4000, 20, large_resp.view());
    assert!(large >= small);
    assert!(
        large > small,
        "large-sample tensor cap should relax the small-sample response bottleneck"
    );
}

#[test]
pub(crate) fn near_gaussian_response_trims_response_basis_below_skewed_response() {
    // A clean location-scale Gaussian transformation cannot identify a heavy
    // shape block, so the data-driven complexity cap must collapse its knot
    // budget far below a strongly non-Gaussian response at the same n / p_cov.
    let config = TransformationNormalConfig::default();
    let n = 2000usize;
    // Gaussian-ish (symmetric, mesokurtic) response via a fine standard-normal
    // quantile grid: skewness ≈ 0, excess kurtosis ≈ 0 ⇒ minimal shape budget.
    let gaussian: Array1<f64> = Array1::from_iter((0..n).map(|i| {
        let u = (i as f64 + 0.5) / n as f64;
        standard_normal_quantile(u).expect("strictly interior normal quantile")
    }));
    let gaussian_knots = effective_response_num_internal_knots(&config, n, 8, gaussian.view());
    let skewed_knots =
        effective_response_num_internal_knots(&config, n, 8, skewed_response(n).view());
    assert!(
        gaussian_knots < skewed_knots,
        "near-Gaussian transformation should use a smaller response basis \
             than a strongly skewed one (gaussian={gaussian_knots}, skewed={skewed_knots})"
    );
    assert!(
        gaussian_knots <= 4,
        "near-Gaussian transformation knot budget should collapse to a handful \
             of internal knots, got {gaussian_knots}"
    );
}

#[test]
pub(crate) fn transformation_normal_joint_psi_second_order_terms_match_fd() {
    let psi = array![0.15, -0.10];
    let h = 1e-6;
    let row_offset = Arc::new(array![0.70, -0.20, 0.40, -0.50]);
    let (mut family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    family.offset = Arc::clone(&row_offset);
    let states = vec![state.clone()];
    let specs = vec![spec];

    let analytic = family
        .exact_newton_joint_psisecond_order_terms(
            &states,
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
            0,
            1,
        )
        .expect("analytic psi second-order terms")
        .expect("psi second-order terms should be present");

    let eval_first = |psi_eval: &Array1<f64>| {
        let (mut f_eval, deriv_eval, state_eval, spec_eval) = toy_family_and_derivatives(psi_eval);
        f_eval.offset = Arc::clone(&row_offset);
        let states_eval = vec![state_eval];
        let specs_eval = vec![spec_eval];
        f_eval
            .exact_newton_joint_psi_terms(
                &states_eval,
                &specs_eval,
                &test_design_hyper_layout(&deriv_eval),
                0,
            )
            .expect("first-order psi terms")
            .expect("first-order terms should be present")
    };

    let mut psi_plus = psi.clone();
    psi_plus[1] += h;
    let plus = eval_first(&psi_plus);
    let mut psi_minus = psi.clone();
    psi_minus[1] -= h;
    let minus = eval_first(&psi_minus);

    let objective_fd = (plus.objective_psi - minus.objective_psi) / (2.0 * h);
    assert!(
        (analytic.objective_psi_psi - objective_fd).abs() < 1e-5,
        "objective psi second-order mismatch: analytic={}, fd={objective_fd}",
        analytic.objective_psi_psi
    );

    let score_fd = (&plus.score_psi - &minus.score_psi) / (2.0 * h);
    for idx in 0..score_fd.len() {
        assert!(
            (analytic.score_psi_psi[idx] - score_fd[idx]).abs() < 1e-5,
            "score psi second-order mismatch at {idx}: analytic={}, fd={}",
            analytic.score_psi_psi[idx],
            score_fd[idx]
        );
    }

    let hess_fd =
        (dense_first_order_psi_hessian(&plus) - dense_first_order_psi_hessian(&minus)) / (2.0 * h);
    // The CTN psi-psi second-order kernel exposes its dense p_total×p_total
    // block through `hessian_psi_psi` when the family materializes it
    // eagerly, or through an operator-backed `hessian_psi_psi_operator`
    // when the family stages the Hessian as HVPs. The FD comparison needs
    // the dense matrix either way, so materialize the operator on demand.
    let analytic_hessian = if analytic.hessian_psi_psi.nrows() > 0 {
        analytic.hessian_psi_psi.clone()
    } else {
        analytic
            .hessian_psi_psi_operator
            .as_ref()
            .expect("CTN psi-psi must expose either dense Hessian or operator")
            .to_dense()
    };
    assert_matrix_derivativefd(
        &hess_fd,
        &analytic_hessian,
        2e-4,
        "transformation normal psi second-order Hessian",
    );
}

#[test]
pub(crate) fn transformation_normal_joint_psi_first_order_matches_normalized_loglik_fd() {
    let psi = array![0.15, -0.10];
    let h = 1e-6;
    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let beta = state.beta.clone();
    let states = vec![state.clone()];
    let specs = vec![spec];

    let analytic = family
        .exact_newton_joint_psi_terms(
            &states,
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
            0,
        )
        .expect("analytic psi first-order terms")
        .expect("first-order terms should be present");

    let eval_negative_loglik = |psi_eval: &Array1<f64>| {
        let (f_eval, _, mut state_eval, _) = toy_family_and_derivatives(psi_eval);
        state_eval.beta = beta.clone();
        -f_eval
            .log_likelihood_only(std::slice::from_ref(&state_eval))
            .expect("log-likelihood at perturbed psi")
    };

    let mut psi_plus = psi.clone();
    psi_plus[0] += h;
    let mut psi_minus = psi.clone();
    psi_minus[0] -= h;
    let fd = (eval_negative_loglik(&psi_plus) - eval_negative_loglik(&psi_minus)) / (2.0 * h);

    assert!(
        (analytic.objective_psi - fd).abs() < 1e-6,
        "normalized CTN psi objective mismatch: analytic={}, fd={fd}",
        analytic.objective_psi
    );

    assert_eq!(analytic.hessian_psi.nrows(), 0);
    assert_eq!(analytic.hessian_psi.ncols(), 0);
    let op = analytic
        .hessian_psi_operator
        .as_ref()
        .expect("CTN psi first-order Hessian must be operator-backed");
    assert_eq!(op.dim(), beta.len());

    let direction = toy_probe_vector(beta.len(), 407);
    let h_beta = 1e-6;
    let eval_score = |beta_eval: &Array1<f64>| {
        let mut state_eval = state.clone();
        state_eval.beta = beta_eval.clone();
        family
            .exact_newton_joint_psi_terms(
                std::slice::from_ref(&state_eval),
                &specs,
                &test_design_hyper_layout(&derivative_blocks),
                0,
            )
            .expect("first-order psi terms at shifted beta")
            .expect("shifted first-order terms should be present")
            .score_psi
    };
    let beta_plus = &beta + &(direction.clone() * h_beta);
    let beta_minus = &beta - &(direction.clone() * h_beta);
    let score_fd = (eval_score(&beta_plus) - eval_score(&beta_minus)) / (2.0 * h_beta);
    let hvp = op.mul_vec(&direction);
    for idx in 0..hvp.len() {
        let tol = 2e-5 * score_fd[idx].abs().max(1.0);
        assert!(
            (hvp[idx] - score_fd[idx]).abs() <= tol,
            "first-order psi Hessian operator mismatch at {idx}: analytic={:.6e}, fd={:.6e}",
            hvp[idx],
            score_fd[idx]
        );
    }

    let mut factor = Array2::<f64>::zeros((beta.len(), 4));
    for (col, seed) in [408_u64, 409, 410, 411].into_iter().enumerate() {
        factor
            .column_mut(col)
            .assign(&toy_probe_vector(beta.len(), seed));
    }
    let got_mat = op.mul_mat(&factor);
    for col in 0..factor.ncols() {
        let want_col = op.mul_vec(&factor.column(col).to_owned());
        for row in 0..beta.len() {
            let tol = 1.0e-11 * want_col[row].abs().max(1.0) + 1.0e-11;
            assert!(
                (got_mat[[row, col]] - want_col[row]).abs() <= tol,
                "first-order psi Hessian batched mul_mat mismatch at ({row}, {col}): got={:.6e}, want={:.6e}",
                got_mat[[row, col]],
                want_col[row],
            );
        }
    }
    let got_trace = op.trace_projected_factor(&factor);
    let want_trace = factor
        .iter()
        .zip(got_mat.iter())
        .map(|(&f, &bf)| f * bf)
        .sum::<f64>();
    let tol = 1.0e-11 * want_trace.abs().max(1.0) + 1.0e-11;
    assert!(
        (got_trace - want_trace).abs() <= tol,
        "first-order psi Hessian projected trace mismatch: got={:.6e}, want={:.6e}",
        got_trace,
        want_trace,
    );
}

#[test]
pub(crate) fn ctn_psi_workspace_first_order_matches_per_axis_path_bit_equivalent() {
    // Bit-equivalence guard for `TransformationNormalPsiWorkspace`. The
    // workspace's single-pass kernel must produce the same per-axis
    // `objective_psi` and `score_psi` as the per-axis `scop_psi_terms`
    // path that the previous CTN code path used. We compare across every
    // ψ axis at once — there is no axis whose accumulated state can
    // mask a bug in another axis.
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let states = vec![state.clone()];
    let specs = vec![spec];
    let n_psi = derivative_blocks[0].len();
    assert!(
        n_psi >= 2,
        "toy CTN fixture must expose at least 2 ψ axes for the workspace check, got {n_psi}"
    );

    // Per-axis ground truth via the existing direct hook.
    let mut per_axis: Vec<ExactNewtonJointPsiTerms> = Vec::with_capacity(n_psi);
    for psi_index in 0..n_psi {
        per_axis.push(
            family
                .exact_newton_joint_psi_terms(
                    &states,
                    &specs,
                    &test_design_hyper_layout(&derivative_blocks),
                    psi_index,
                )
                .expect("per-axis ψ terms")
                .expect("per-axis ψ terms must be present"),
        );
    }

    // All-axes pass via the workspace.
    let workspace = family
        .exact_newton_joint_psi_workspace(
            &states,
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
        )
        .expect("CTN ψ workspace constructor")
        .expect("CTN ψ workspace must be present");
    let mut shared_factor = Array2::<f64>::zeros((state.beta.len(), 3));
    for (col, seed) in [70_001_u64, 80_001_u64, 90_001_u64].into_iter().enumerate() {
        shared_factor
            .column_mut(col)
            .assign(&toy_probe_vector(state.beta.len(), seed));
    }
    let projected_cache = ProjectedFactorCache::default();

    for psi_index in 0..n_psi {
        let cached = workspace
            .first_order_terms(psi_index)
            .expect("workspace first-order terms")
            .expect("workspace first-order terms must be present");
        let expected = &per_axis[psi_index];

        // Objective: the workspace fold is order-permutation-equivalent
        // to the per-axis fold; allow a tiny floating-point slack on top
        // of bit equality so reductions over different chunk shapes
        // (rayon's deterministic-order fold groups rows differently than
        // the serial loop) do not flake the test.
        let obj_diff = (cached.objective_psi - expected.objective_psi).abs();
        let obj_scale = expected.objective_psi.abs().max(1.0);
        assert!(
            obj_diff <= 1.0e-12 * obj_scale,
            "ψ workspace objective_psi[axis={psi_index}] mismatch: cached={}, per-axis={}, |diff|={obj_diff}",
            cached.objective_psi,
            expected.objective_psi,
        );

        assert_eq!(
            cached.score_psi.len(),
            expected.score_psi.len(),
            "ψ workspace score_psi length mismatch at axis {psi_index}"
        );
        for idx in 0..expected.score_psi.len() {
            let diff = (cached.score_psi[idx] - expected.score_psi[idx]).abs();
            let scale = expected.score_psi[idx].abs().max(1.0);
            assert!(
                diff <= 1.0e-12 * scale,
                "ψ workspace score_psi[axis={psi_index}, idx={idx}] mismatch: cached={}, per-axis={}, |diff|={diff}",
                cached.score_psi[idx],
                expected.score_psi[idx],
            );
        }

        // The per-axis matrix-free Hessian operator must remain present
        // and dimension-matching; we do not compare its action here
        // because the operator is constructed directly from the same
        // `row_quantities` cache the per-axis path uses.
        let cached_op = cached
            .hessian_psi_operator
            .as_ref()
            .expect("workspace ψ Hessian operator must be present");
        assert_eq!(cached_op.dim(), state.beta.len());
        let cached_trace =
            cached_op.trace_projected_factor_cached(&shared_factor, &projected_cache);
        let direct_trace = cached_op.trace_projected_factor(&shared_factor);
        let trace_tol = 1.0e-10 * direct_trace.abs().max(1.0) + 1.0e-10;
        assert!(
            (cached_trace - direct_trace).abs() <= trace_tol,
            "workspace ψ cached projected trace mismatch at axis {psi_index}: cached={cached_trace:.6e}, direct={direct_trace:.6e}",
        );
    }
}

/// gam#979. `∂H/∂ψ_axis` is a weighted Gram over the response blocks and is
/// assembled once per axis per evaluation
/// (`scop_psi_hessian_dense_from_cov`); every action the outer engine asks
/// this operator for is a read of that assembly. Nothing in the file derives
/// the same matrix a second way any more, so the gate is against the object
/// the matrix IS: `∂H/∂ψ` is the β-Jacobian of the ψ score, and `score_psi`
/// is computed by a completely separate row loop in `scop_psi_terms`.
///
/// This differences the WHOLE Jacobian, not one direction: a wrong block
/// factor, a wrong power of `h'`, a `(k, l)` block placed at `(l, k)` when the
/// two differ, or a missing symmetrisation of the `cov`/`cψ` cross term all
/// move at least one column off its difference.
#[test]
pub(crate) fn ctn_psi_hessian_assembly_is_the_beta_jacobian_of_the_psi_score_979() {
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let specs = vec![spec];
    let beta = state.beta.clone();
    let p_total = beta.len();
    let step = 1.0e-6;

    let terms = family
        .exact_newton_joint_psi_terms(
            std::slice::from_ref(&state),
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
            0,
        )
        .expect("analytic psi first-order terms")
        .expect("first-order terms should be present");
    let operator = terms
        .hessian_psi_operator
        .as_ref()
        .expect("CTN psi first-order Hessian must be operator-backed");
    let analytic = operator.to_dense();
    assert_eq!(analytic.nrows(), p_total);
    assert_eq!(analytic.ncols(), p_total);

    let score_at = |beta_eval: &Array1<f64>| {
        let mut shifted = state.clone();
        shifted.beta = beta_eval.clone();
        family
            .exact_newton_joint_psi_terms(
                std::slice::from_ref(&shifted),
                &specs,
                &test_design_hyper_layout(&derivative_blocks),
                0,
            )
            .expect("first-order psi terms at shifted beta")
            .expect("shifted first-order terms should be present")
            .score_psi
    };

    let mut peak = 0.0_f64;
    for column in 0..p_total {
        let mut plus = beta.clone();
        let mut minus = beta.clone();
        plus[column] += step;
        minus[column] -= step;
        let fd = (score_at(&plus) - score_at(&minus)) / (2.0 * step);
        for row in 0..p_total {
            let want = fd[row];
            let got = analytic[[row, column]];
            peak = peak.max(want.abs());
            let tol = 2.0e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "psi Hessian assembly [{row}, {column}] = {got:.9e} against the score's own \
                 central difference {want:.9e} (tol {tol:.3e})"
            );
        }
    }

    // The matrix is a Hessian: symmetric to accumulation order.
    for row in 0..p_total {
        for column in row + 1..p_total {
            let (upper, lower) = (analytic[[row, column]], analytic[[column, row]]);
            let tol = 1.0e-11 * upper.abs().max(1.0) + 1.0e-11;
            assert!(
                (upper - lower).abs() <= tol,
                "psi Hessian assembly is not symmetric at ({row}, {column}): \
                 {upper:.9e} vs {lower:.9e}"
            );
        }
    }

    // The fixture has to move the Hessian at all, or every comparison above
    // passes on a pair of zero matrices.
    assert!(
        peak > 1.0e-4,
        "the fixture's psi axis produces no Hessian drift (peak |fd| = {peak:.3e}); \
         the comparison above would then be vacuous"
    );
}

#[test]
pub(crate) fn ctn_psi_workspace_second_order_matches_per_pair_path() {
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let states = vec![state.clone()];
    let specs = vec![spec];
    let n_psi = derivative_blocks[0].len();

    let workspace = family
        .exact_newton_joint_psi_workspace(
            &states,
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
        )
        .expect("CTN ψ workspace constructor")
        .expect("CTN ψ workspace must be present");
    let mut shared_factor = Array2::<f64>::zeros((state.beta.len(), 3));
    for (col, seed) in [10_001_u64, 20_001_u64, 30_001_u64].into_iter().enumerate() {
        shared_factor
            .column_mut(col)
            .assign(&toy_probe_vector(state.beta.len(), seed));
    }
    let projected_cache = ProjectedFactorCache::default();

    for psi_i in 0..n_psi {
        for psi_j in psi_i..n_psi {
            let direct = family
                .exact_newton_joint_psisecond_order_terms(
                    &states,
                    &specs,
                    &test_design_hyper_layout(&derivative_blocks),
                    psi_i,
                    psi_j,
                )
                .expect("direct ψ-ψ terms")
                .expect("direct ψ-ψ terms must be present");
            let cached = workspace
                .second_order_terms(psi_i, psi_j)
                .expect("workspace ψ-ψ terms")
                .expect("workspace ψ-ψ terms must be present");

            let obj_diff = (cached.objective_psi_psi - direct.objective_psi_psi).abs();
            let obj_scale = direct.objective_psi_psi.abs().max(1.0);
            assert!(
                obj_diff <= 1.0e-12 * obj_scale,
                "ψ workspace objective_psi_psi[{psi_i},{psi_j}] mismatch: cached={}, direct={}, |diff|={obj_diff}",
                cached.objective_psi_psi,
                direct.objective_psi_psi,
            );

            assert_eq!(
                cached.score_psi_psi.len(),
                direct.score_psi_psi.len(),
                "ψ workspace score_psi_psi length mismatch at pair ({psi_i},{psi_j})"
            );
            for idx in 0..direct.score_psi_psi.len() {
                let diff = (cached.score_psi_psi[idx] - direct.score_psi_psi[idx]).abs();
                let scale = direct.score_psi_psi[idx].abs().max(1.0);
                assert!(
                    diff <= 1.0e-12 * scale,
                    "ψ workspace score_psi_psi[pair=({psi_i},{psi_j}), idx={idx}] mismatch: cached={}, direct={}, |diff|={diff}",
                    cached.score_psi_psi[idx],
                    direct.score_psi_psi[idx],
                );
            }

            let cached_op = cached
                .hessian_psi_psi_operator
                .as_ref()
                .expect("workspace ψ-ψ Hessian operator must be present");
            let direct_op = direct
                .hessian_psi_psi_operator
                .as_ref()
                .expect("direct ψ-ψ Hessian operator must be present");
            assert_eq!(cached_op.dim(), direct_op.dim());
            assert_eq!(cached_op.dim(), state.beta.len());

            let cached_trace =
                cached_op.trace_projected_factor_cached(&shared_factor, &projected_cache);
            let direct_trace = cached_op.trace_projected_factor(&shared_factor);
            let trace_tol = 1.0e-10 * direct_trace.abs().max(1.0) + 1.0e-10;
            assert!(
                (cached_trace - direct_trace).abs() <= trace_tol,
                "workspace ψ-ψ cached projected trace mismatch at pair ({psi_i},{psi_j}): cached={cached_trace:.6e}, direct={direct_trace:.6e}",
            );
        }
    }
}

#[test]
pub(crate) fn transformation_normal_joint_psi_second_order_terms_are_operator_backed() {
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let states = vec![state.clone()];
    let specs = vec![spec];

    let terms = family
        .exact_newton_joint_psisecond_order_terms(
            &states,
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
            0,
            1,
        )
        .expect("analytic psi second-order terms")
        .expect("psi second-order terms should be present");

    assert_eq!(terms.hessian_psi_psi.nrows(), 0);
    assert_eq!(terms.hessian_psi_psi.ncols(), 0);
    let op = terms
        .hessian_psi_psi_operator
        .as_ref()
        .expect("CTN psi-psi Hessian must be operator-backed");
    assert!(op.is_implicit());
    let p = state.beta.len();
    assert_eq!(op.dim(), p);
    assert!(op.has_fast_bilinear_view());

    let dense = op.to_dense();
    assert_eq!(dense.nrows(), p);
    assert_eq!(dense.ncols(), p);

    let v = toy_probe_vector(p, 901);
    let got_vec = op.mul_vec(&v);
    let want_vec = dense.dot(&v);
    for i in 0..p {
        let tol = 1e-10 * want_vec[i].abs().max(1.0) + 1e-10;
        assert!(
            (got_vec[i] - want_vec[i]).abs() <= tol,
            "psi-psi operator matvec mismatch at {i}: got={:.6e}, want={:.6e}",
            got_vec[i],
            want_vec[i]
        );
    }

    let mut factor = Array2::<f64>::zeros((p, 3));
    for (col, seed) in [902_u64, 903, 904].into_iter().enumerate() {
        factor.column_mut(col).assign(&toy_probe_vector(p, seed));
    }
    let got_mat = op.mul_mat(&factor);
    let want_mat = dense.dot(&factor);
    for row in 0..p {
        for col in 0..factor.ncols() {
            let tol = 1e-10 * want_mat[[row, col]].abs().max(1.0) + 1e-10;
            assert!(
                (got_mat[[row, col]] - want_mat[[row, col]]).abs() <= tol,
                "psi-psi operator mul_mat mismatch at ({row}, {col}): got={:.6e}, want={:.6e}",
                got_mat[[row, col]],
                want_mat[[row, col]]
            );
        }
    }

    let left = toy_probe_vector(p, 905);
    let right = toy_probe_vector(p, 906);
    let got_bilinear = op.bilinear_view(left.view(), right.view());
    let want_bilinear = right.dot(&dense.dot(&left));
    let tol = 1e-10 * want_bilinear.abs().max(1.0) + 1e-10;
    assert!(
        (got_bilinear - want_bilinear).abs() <= tol,
        "psi-psi operator bilinear mismatch: got={:.6e}, want={:.6e}",
        got_bilinear,
        want_bilinear
    );

    let got_trace = op.trace_projected_factor(&factor);
    let want_trace = factor
        .iter()
        .zip(want_mat.iter())
        .map(|(&f, &bf)| f * bf)
        .sum::<f64>();
    let tol = 1e-10 * want_trace.abs().max(1.0) + 1e-10;
    assert!(
        (got_trace - want_trace).abs() <= tol,
        "psi-psi operator projected trace mismatch: got={:.6e}, want={:.6e}",
        got_trace,
        want_trace
    );
}

#[test]
pub(crate) fn transformation_normal_joint_psihessian_directional_derivative_matches_fd() {
    let psi = array![0.15, -0.10];
    let h = 1e-6;
    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let direction = toy_probe_vector(spec.design.ncols(), 701);
    let specs = vec![spec];

    let analytic = family
        .exact_newton_joint_psihessian_directional_derivative(
            std::slice::from_ref(&state),
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
            0,
            &direction,
        )
        .expect("analytic psi hessian directional derivative")
        .expect("psi hessian directional derivative should be present");

    let eval_hess = |beta: &Array1<f64>| {
        let mut shifted_state = state.clone();
        shifted_state.beta = beta.clone();
        let terms = family
            .exact_newton_joint_psi_terms(
                std::slice::from_ref(&shifted_state),
                &specs,
                &test_design_hyper_layout(&derivative_blocks),
                0,
            )
            .expect("first-order psi terms at shifted beta")
            .expect("shifted first-order terms should be present");
        dense_first_order_psi_hessian(&terms)
    };

    let beta_plus = &state.beta + &(direction.clone() * h);
    let beta_minus = &state.beta - &(direction.clone() * h);
    let fd = (eval_hess(&beta_plus) - eval_hess(&beta_minus)) / (2.0 * h);
    assert_matrix_derivativefd(
        &fd,
        &analytic,
        2e-4,
        "transformation normal psi hessian directional derivative",
    );

    let workspace = family
        .exact_newton_joint_psi_workspace(
            &[state.clone()],
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
        )
        .expect("CTN psi workspace constructor")
        .expect("CTN psi workspace must be present");
    let drift_op = workspace
        .hessian_directional_derivative(0, &direction)
        .expect("workspace psi dH operator")
        .expect("workspace psi dH operator must be present");
    let DriftDerivResult::Operator(drift_op) = drift_op else {
        panic!("CTN workspace psi dH must be operator-backed");
    };
    let probe = toy_probe_vector(state.beta.len(), 90_001_u64);
    let got_vec = drift_op.mul_vec(&probe);
    let want_vec = analytic.dot(&probe);
    for i in 0..state.beta.len() {
        let vec_tol = 1.0e-10 * want_vec[i].abs().max(1.0) + 1.0e-10;
        assert!(
            (got_vec[i] - want_vec[i]).abs() <= vec_tol,
            "workspace psi dH matvec mismatch at {i}: got={:.6e}, want={:.6e}",
            got_vec[i],
            want_vec[i],
        );
    }
    let mut factor = Array2::<f64>::zeros((state.beta.len(), 3));
    for (col, seed) in [91_001_u64, 92_001_u64, 93_001_u64].into_iter().enumerate() {
        factor
            .column_mut(col)
            .assign(&toy_probe_vector(state.beta.len(), seed));
    }
    let got_mat = drift_op.mul_mat(&factor);
    let want_mat = analytic.dot(&factor);
    for row in 0..state.beta.len() {
        for col in 0..factor.ncols() {
            let mat_tol = 1.0e-10 * want_mat[[row, col]].abs().max(1.0) + 1.0e-10;
            assert!(
                (got_mat[[row, col]] - want_mat[[row, col]]).abs() <= mat_tol,
                "workspace psi dH matmat mismatch at ({row}, {col}): got={:.6e}, want={:.6e}",
                got_mat[[row, col]],
                want_mat[[row, col]],
            );
        }
    }
    let got_trace = drift_op.trace_projected_factor(&factor);
    let want_trace = factor
        .iter()
        .zip(want_mat.iter())
        .map(|(&f, &bf)| f * bf)
        .sum::<f64>();
    let trace_tol = 1.0e-10 * want_trace.abs().max(1.0) + 1.0e-10;
    assert!(
        (got_trace - want_trace).abs() <= trace_tol,
        "workspace psi dH projected trace mismatch: got={got_trace:.6e}, want={want_trace:.6e}",
    );
}

#[test]
pub(crate) fn transformation_normal_joint_hessian_second_directional_derivative_matches_fd() {
    let psi = array![0.15, -0.10];
    let h = 1e-6;
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let p = state.beta.len();
    let dir_u = toy_probe_vector(p, 801);
    let dir_v = toy_probe_vector(p, 802);

    let analytic = family
        .exact_newton_joint_hessiansecond_directional_derivative(
            std::slice::from_ref(&state),
            &dir_u,
            &dir_v,
        )
        .expect("analytic second directional derivative")
        .expect("second directional derivative should be present");

    let eval_dh = |beta: &Array1<f64>| {
        let shifted_state = ParameterBlockState {
            beta: beta.clone(),
            eta: state.eta.clone(),
        };
        family
            .exact_newton_joint_hessian_directional_derivative(
                std::slice::from_ref(&shifted_state),
                &dir_u,
            )
            .expect("first directional derivative at shifted beta")
            .expect("shifted first directional derivative should be present")
    };

    let beta_plus = &state.beta + &(dir_v.clone() * h);
    let beta_minus = &state.beta - &(dir_v * h);
    let fd = (eval_dh(&beta_plus) - eval_dh(&beta_minus)) / (2.0 * h);
    assert_matrix_derivativefd(&fd, &analytic, 2e-4, "transformation normal joint d2H");
}

#[test]
pub(crate) fn ctn_joint_hessian_workspace_matvec_matches_dense() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();

    let dense = family
        .exact_newton_joint_hessian(std::slice::from_ref(&state))
        .expect("dense joint Hessian build")
        .expect("dense joint Hessian present");
    assert_eq!(dense.nrows(), p);
    assert_eq!(dense.ncols(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
        .expect("workspace build")
        .expect("workspace present");

    // `hessian_dense` is amortization-gated; the toy spec carries no
    // penalties, so `expected_reuse=1` against `p/SAFETY≥2` correctly
    // routes through matrix-free. We're testing dense/HVP agreement,
    // not the gate, so force the dense build via `hessian_dense_forced`.
    // The amortization-gate behavior is exercised separately in
    // `ctn_dense_hessian_amortization_gate_picks_matrix_free_when_p_dominates_reuse`.
    let dense_from_workspace = workspace
        .hessian_dense_forced()
        .expect("workspace forced dense Hessian call")
        .expect("workspace forced dense Hessian present");
    assert_eq!(dense_from_workspace.nrows(), p);
    assert_eq!(dense_from_workspace.ncols(), p);
    for i in 0..p {
        for j in 0..p {
            let want = dense[[i, j]];
            let got = dense_from_workspace[[i, j]];
            assert!(
                (want - got).abs() <= 1e-12 * want.abs().max(1.0) + 1e-12,
                "workspace dense mismatch at ({i}, {j}): dense={want:.6e}, workspace={got:.6e}"
            );
        }
    }

    // Diagonal must agree element-wise (matrix-free pre-square path vs. dense gram).
    let diag_op = workspace
        .hessian_diagonal()
        .expect("diagonal call")
        .expect("diagonal present");
    assert_eq!(diag_op.len(), p);
    for i in 0..p {
        let want = dense[[i, i]];
        let got = diag_op[i];
        assert!(
            (want - got).abs() <= 1e-12 * want.abs().max(1.0) + 1e-12,
            "diagonal mismatch at {i}: dense={want:.6e}, workspace={got:.6e}"
        );
    }

    // Hessian-vector product must agree with dense H · v across a few
    // randomly chosen directions (deterministic seed for stability).
    let directions = [
        toy_probe_vector(p, 101),
        toy_probe_vector(p, 102),
        toy_probe_vector(p, 103),
    ];
    for (k, v) in directions.iter().enumerate() {
        assert_eq!(v.len(), p);
        let want = dense.dot(v);
        let got = workspace
            .hessian_matvec(v)
            .expect("matvec call")
            .expect("matvec present");
        assert_eq!(got.len(), p);
        for i in 0..p {
            let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "matvec[{k}, {i}] mismatch: dense={:.6e}, workspace={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn ctn_direct_hessian_matvec_honors_outer_subsample_weights() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let masked = family
        .with_outer_subsample(&array![0.0, 2.5, 0.0, 1.5])
        .expect("non-binary outer subsample weights");
    let row_quantities = masked
        .row_quantities(&state.beta)
        .expect("masked row quantities");
    let (_, dense) = masked
        .scop_gradient_and_negative_hessian(&state.beta, &row_quantities)
        .expect("masked dense Hessian");

    let probe = toy_probe_vector(state.beta.len(), 607);
    let want = dense.dot(&probe);
    let mut got = Array1::<f64>::zeros(probe.len());
    masked
        .scop_hessian_matvec_into(&state.beta, &row_quantities, &probe, &mut got)
        .expect("masked direct Hessian matvec");

    for i in 0..want.len() {
        let tolerance = 1e-11 * want[i].abs().max(1.0) + 1e-12;
        assert!(
            (want[i] - got[i]).abs() <= tolerance,
            "masked direct Hessian matvec mismatch at {i}: dense={:.6e}, direct={:.6e}, tolerance={tolerance:.6e}",
            want[i],
            got[i]
        );
    }
}

#[test]
pub(crate) fn ctn_joint_hessian_workspace_matvec_into_primes_dense_cache() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let p = state.beta.len();
    let row_quantities = family.row_quantities(&state.beta).expect("row quantities");
    let workspace = TransformationNormalJointHessianWorkspace::new(
        Arc::new(family.clone()),
        state.beta.clone(),
        row_quantities,
    )
    .expect("workspace build");
    assert!(workspace.dense_hessian_cache_enabled());
    assert_eq!(
        workspace.hessian_source_preference_for_intent(MaterializationIntent::InnerSolve),
        JointHessianSourcePreference::Dense,
        "a CTN Hessian that will be cached densely must expose that matrix to the inner spectral mode solver"
    );
    assert!(workspace.dense_hessian_cache.get().is_none());

    let dense = family
        .exact_newton_joint_hessian(std::slice::from_ref(&state))
        .expect("dense joint Hessian build")
        .expect("dense joint Hessian present");
    let v = toy_probe_vector(p, 12_345);
    let want = dense.dot(&v);
    let mut got = Array1::<f64>::zeros(p);
    workspace
        .apply_hessian_into(&v, &mut got)
        .expect("workspace matvec_into");
    assert!(workspace.dense_hessian_cache.get().is_some());
    for i in 0..p {
        let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
        assert!(
            (want[i] - got[i]).abs() <= tol,
            "cached matvec_into mismatch at {i}: dense={:.6e}, workspace={:.6e}",
            want[i],
            got[i]
        );
    }

    let v2 = toy_probe_vector(p, 12_346);
    let want2 = dense.dot(&v2);
    workspace
        .apply_hessian_into(&v2, &mut got)
        .expect("second workspace matvec_into");
    for i in 0..p {
        let tol = 1e-12 * want2[i].abs().max(1.0) + 1e-12;
        assert!(
            (want2[i] - got[i]).abs() <= tol,
            "second cached matvec_into mismatch at {i}: dense={:.6e}, workspace={:.6e}",
            want2[i],
            got[i]
        );
    }
}

#[test]
pub(crate) fn ctn_coefficient_hessian_cost_uses_dense_for_small_problems() {
    // Toy family: n=4, p_resp=2, p_cov=2 → p_total=4. The matrix-free
    // gate `use_joint_matrix_free_path(4, 4)` returns false (well below
    // every threshold), so the override must report the dense Khatri–Rao
    // gram cost n·(p_resp·p_cov)² = 4·16 = 64.
    let psi = array![0.15, -0.10];
    let (family, _, _, _) = toy_family_and_derivatives(&psi);
    let n = family.response_val_basis.nrows() as u64;
    let p_resp = family.response_val_basis.ncols() as u64;
    let p_cov = family.covariate_design.ncols() as u64;
    assert!(!crate::custom_family::use_joint_matrix_free_path(
        (p_resp * p_cov) as usize,
        n as usize,
    ));
    let p_total = p_resp * p_cov;
    let expected_dense = n * p_total * p_total;
    assert_eq!(family.coefficient_hessian_cost(&[]), expected_dense);
}

#[test]
pub(crate) fn ctn_coefficient_hessian_cost_switches_to_matvec_when_matrix_free_active() {
    // p_cov=256 keeps p_total = p_resp · p_cov ≥ JOINT_MATRIX_FREE_MIN_DIM
    // so matrix-free is ALWAYS active for any n. The override must report
    // the per-Hv matvec cost n·(p_resp + p_cov), not the dense p² gram.
    // n=8 keeps the test allocation small (~16 KB for covariate_design).
    let n = 8usize;
    let p_cov = 256usize;
    let response = Array1::from_iter((0..n).map(|i| (i as f64) / (n - 1) as f64));
    let (val_basis, deriv_basis, knots, transform, _p_resp) = toy_response_basis(&response);
    let weights = Array1::from_elem(n, 1.0);
    let offset = Array1::zeros(n);
    // Non-degenerate covariate design: small column-wise variation makes
    // the joint warm-start solve well-posed without changing the
    // matrix-free gating behavior tested below.
    let mut cov_design = Array2::<f64>::zeros((n, p_cov));
    for i in 0..n {
        for j in 0..p_cov {
            cov_design[[i, j]] = 0.1 + 0.01 * (i as f64) + 0.001 * (j as f64);
        }
    }
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        vec![],
        knots,
        toy_scop_ctn_config().response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
        vec![],
        &toy_scop_ctn_config(),
        None,
    )
    .expect("matrix-free-eligible CTN family");
    let p_resp = family.response_val_basis.ncols() as u64;
    let actual_p_cov = family.covariate_design.ncols() as u64;
    let p_total = p_resp * actual_p_cov;
    assert!(crate::custom_family::use_joint_matrix_free_path(
        p_total as usize,
        n,
    ));
    let expected_matvec = (n as u64) * (p_resp + actual_p_cov);
    assert_eq!(family.coefficient_hessian_cost(&[]), expected_matvec);
    // Sanity: the matrix-free cost is dramatically smaller than the dense
    // would have been (the whole point of branching).
    let dense_cost = (n as u64) * p_total * p_total;
    assert!(expected_matvec < dense_cost / 100);
}

#[test]
pub(crate) fn ctn_inner_and_outer_hvp_capabilities_are_advertised() {
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, _, spec) = toy_family_and_derivatives(&psi);
    let specs = std::slice::from_ref(&spec);

    assert!(family.inner_coefficient_hessian_hvp_available(specs));
    assert!(family.outer_hyper_hessian_hvp_available(specs));
    assert!(family.outer_hyper_hessian_dense_available(specs));
    assert_eq!(
        family.exact_outer_derivative_order(specs, &BlockwiseFitOptions::default()),
        crate::custom_family::ExactOuterDerivativeOrder::Second
    );

    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        ..BlockwiseFitOptions::default()
    };
    let (gradient, hessian) = custom_family_outer_derivatives(&family, specs, &options);
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Either);

    let rho_dim = spec.initial_log_lambdas.len();
    let psi_dim = derivative_blocks[0].len();
    let outer_plan = gam_solve::rho_optimizer::plan(&gam_solve::rho_optimizer::OuterCapability {
        gradient,
        hessian,
        n_params: rho_dim + psi_dim,
        psi_dim,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: true,
    });
    assert_eq!(outer_plan.solver, gam_solve::rho_optimizer::Solver::Arc);
    assert_eq!(
        outer_plan.hessian_source,
        gam_solve::rho_optimizer::HessianSource::Analytic
    );
}

#[test]
pub(crate) fn ctn_large_n_outer_hvp_capability_selects_operator_path() {
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, _, spec) = toy_family_and_derivatives(&psi);
    let specs = std::slice::from_ref(&spec);
    assert!(family.outer_hyper_hessian_hvp_available(specs));

    let rho_dim = spec.initial_log_lambdas.len();
    let psi_dim = derivative_blocks[0].len();
    let k_outer = rho_dim + psi_dim;
    // `outer_hessian_route_plan` is purely a cost-based crossover
    // over `(n_obs, p_dim, k_outer)`; commit 7f7705c removed the
    // callback-kernel short-circuit that previously let CTN trip the
    // operator path on its analytic HVP alone.  Per the current
    // function docstring, family-supplied directional θθ operators
    // route via `HessianDerivativeProvider::family_outer_hessian_operator`
    // and short-circuit this predicate at the call site.  The
    // meaningful invariant for this test is therefore the dispatcher
    // verdict below — `custom_family_outer_derivatives` must still
    // return `Analytic / Analytic` for both gradient and Hessian.
    // We retain the threshold-tuple sanity check on the predicate so
    // a future regression that broke the cost crossover (e.g. flipped
    // a `>=` to `>`) would still be caught here.
    assert!(
        gam_solve::estimate::reml::reml_outer_engine::outer_hessian_route_plan(
            gam_solve::estimate::reml::reml_outer_engine::MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD,
            gam_solve::estimate::reml::reml_outer_engine::MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N,
            k_outer,
            true,
            false,
            false,
        )
        .use_operator
    );

    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        ..BlockwiseFitOptions::default()
    };
    let (gradient, hessian) = custom_family_outer_derivatives(&family, specs, &options);
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Either);
}

#[test]
pub(crate) fn ctn_joint_hessian_workspace_dh_operator_matches_dense() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();
    let d_beta = toy_probe_vector(p, 201);
    assert_eq!(d_beta.len(), p);

    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative(std::slice::from_ref(&state), &d_beta)
        .expect("dense dH build")
        .expect("dense dH present");

    let workspace = family
        .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&d_beta)
        .expect("dH operator call")
        .expect("dH operator present");

    let probes = [
        toy_probe_vector(p, 202),
        toy_probe_vector(p, 203),
        toy_probe_vector(p, 204),
    ];
    let mut probe_mat = Array2::<f64>::zeros((p, probes.len()));
    for (j, w) in probes.iter().enumerate() {
        probe_mat.column_mut(j).assign(w);
    }
    let want_mat = dense_dh.dot(&probe_mat);
    let got_mat = dh_op.mul_mat(&probe_mat);
    for i in 0..p {
        for j in 0..probes.len() {
            let tol = 1e-12 * want_mat[[i, j]].abs().max(1.0) + 1e-12;
            assert!(
                (want_mat[[i, j]] - got_mat[[i, j]]).abs() <= tol,
                "dH op matmat[{}, {}] mismatch: dense={:.6e}, op={:.6e}",
                i,
                j,
                want_mat[[i, j]],
                got_mat[[i, j]]
            );
        }
    }
    let want_trace = probe_mat
        .iter()
        .zip(want_mat.iter())
        .map(|(&f, &bf)| f * bf)
        .sum::<f64>();
    let got_trace = dh_op.trace_projected_factor(&probe_mat);
    let trace_tol = 1e-12 * want_trace.abs().max(1.0) + 1e-12;
    assert!(
        (want_trace - got_trace).abs() <= trace_tol,
        "dH op projected trace mismatch: dense={want_trace:.6e}, op={got_trace:.6e}"
    );
    let cache = ProjectedFactorCache::default();
    let cached_trace = dh_op.trace_projected_factor_cached(&probe_mat, &cache);
    assert!(
        (want_trace - cached_trace).abs() <= trace_tol,
        "dH op cached projected trace mismatch: dense={want_trace:.6e}, op={cached_trace:.6e}"
    );
    let d_beta_2 = toy_probe_vector(p, 205);
    let dense_dh_2 = family
        .exact_newton_joint_hessian_directional_derivative(std::slice::from_ref(&state), &d_beta_2)
        .expect("second dense dH build")
        .expect("second dense dH present");
    let dh_op_2 = workspace
        .directional_derivative_operator(&d_beta_2)
        .expect("second dH operator call")
        .expect("second dH operator present");
    let want_mat_2 = dense_dh_2.dot(&probe_mat);
    let want_trace_2 = probe_mat
        .iter()
        .zip(want_mat_2.iter())
        .map(|(&f, &bf)| f * bf)
        .sum::<f64>();
    let cached_trace_2 = dh_op_2.trace_projected_factor_cached(&probe_mat, &cache);
    let trace_tol_2 = 1e-12 * want_trace_2.abs().max(1.0) + 1e-12;
    assert!(
        (want_trace_2 - cached_trace_2).abs() <= trace_tol_2,
        "second dH op cached projected trace mismatch: dense={want_trace_2:.6e}, op={cached_trace_2:.6e}"
    );
    for (k, w) in probes.iter().enumerate() {
        assert_eq!(w.len(), p);
        let want = dense_dh.dot(w);
        let got = dh_op.mul_vec(w);
        assert_eq!(got.len(), p);
        for i in 0..p {
            let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "dH op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn ctn_joint_hessian_workspace_d2h_operator_matches_dense() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();
    let dir_u = toy_probe_vector(p, 301);
    let dir_v = toy_probe_vector(p, 302);

    let dense_d2h = family
        .exact_newton_joint_hessiansecond_directional_derivative(
            std::slice::from_ref(&state),
            &dir_u,
            &dir_v,
        )
        .expect("dense d2H build")
        .expect("dense d2H present");

    let workspace = family
        .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
        .expect("workspace build")
        .expect("workspace present");
    let d2h_op = workspace
        .second_directional_derivative_operator(&dir_u, &dir_v)
        .expect("d2H operator call")
        .expect("d2H operator present");

    let probes = [
        toy_probe_vector(p, 303),
        toy_probe_vector(p, 304),
        toy_probe_vector(p, 305),
    ];
    let mut probe_mat = Array2::<f64>::zeros((p, probes.len()));
    for (j, w) in probes.iter().enumerate() {
        probe_mat.column_mut(j).assign(w);
    }
    let want_mat = dense_d2h.dot(&probe_mat);
    let got_mat = d2h_op.mul_mat(&probe_mat);
    for i in 0..p {
        for j in 0..probes.len() {
            let tol = 1e-12 * want_mat[[i, j]].abs().max(1.0) + 1e-12;
            assert!(
                (want_mat[[i, j]] - got_mat[[i, j]]).abs() <= tol,
                "d2H op matmat[{}, {}] mismatch: dense={:.6e}, op={:.6e}",
                i,
                j,
                want_mat[[i, j]],
                got_mat[[i, j]]
            );
        }
    }
    for (k, w) in probes.iter().enumerate() {
        assert_eq!(w.len(), p);
        let want = dense_d2h.dot(w);
        let got = d2h_op.mul_vec(w);
        assert_eq!(got.len(), p);
        for i in 0..p {
            let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "d2H op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

/// Cached CTN barrier dH operator check (third-derivative formula
/// `D(∇²B)[u]v = -2 μ Dᵀ((Du)(Dv)/c³)`).
///
/// At fixed direction `d_beta`, builds `H(β ± ε d_beta) v` matrix-free via
/// `apply_hessian` and checks that the centered perturbation quotient
/// converges to the operator's `mul_vec(v)`. This locks in both the analytic formula and the
/// `inv_hp_cu` cache (a stale cache would only show up under ε perturbation,
/// not in the dense-equivalence test that probes a single iterate).
#[test]
pub(crate) fn ctn_dh_operator_matches_fd_under_beta_perturbation() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();
    let d_beta = toy_probe_vector(p, 401);
    let v = toy_probe_vector(p, 402);
    assert_eq!(d_beta.len(), p);
    assert_eq!(v.len(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
        .expect("workspace build")
        .expect("workspace present");
    let want = workspace
        .directional_derivative_operator(&d_beta)
        .expect("dH op call")
        .expect("dH op present")
        .mul_vec(&v);

    let eps = 1e-5;
    let make_state = |scale: f64| ParameterBlockState {
        beta: &state.beta + &(d_beta.mapv(|b| scale * b)),
        eta: state.eta.clone(),
    };
    let plus = family
        .exact_newton_joint_hessian_workspace(
            std::slice::from_ref(&make_state(eps)),
            &[spec.clone()],
        )
        .expect("plus workspace")
        .expect("plus workspace present");
    let minus = family
        .exact_newton_joint_hessian_workspace(
            std::slice::from_ref(&make_state(-eps)),
            &[spec.clone()],
        )
        .expect("minus workspace")
        .expect("minus workspace present");
    let hv_plus = plus
        .hessian_matvec(&v)
        .expect("plus matvec")
        .expect("plus matvec");
    let hv_minus = minus
        .hessian_matvec(&v)
        .expect("minus matvec")
        .expect("minus matvec");
    let fd: Array1<f64> = (&hv_plus - &hv_minus).mapv(|x| x / (2.0 * eps));

    for i in 0..p {
        let scale = want[i].abs().max(1.0);
        // O(ε²) centered FD on a smooth Hessian gives ~1e-7 relative error
        // at ε=1e-5; loose 5e-5 tolerance covers the dominant truncation
        // term plus the inflation by `||v||·||d_beta||`.
        let tol = 5e-5 * scale + 5e-7;
        assert!(
            (want[i] - fd[i]).abs() <= tol,
            "dH FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
            want[i],
            fd[i],
            tol,
        );
    }
}

/// Cached CTN barrier d²H operator check (fourth-derivative
/// formula `D²(∇²B)[u,w]v = 6 μ Dᵀ((Du)(Dw)(Dv)/c⁴)`).
///
/// A centered perturbation of the dH operator along `dir_w` recovers d²H[u, w] · v;
/// this exercises both the cached `inv_hp_qu` and the chained Khatri–Rao
/// apply on the perturbed iterate.
#[test]
pub(crate) fn ctn_d2h_operator_matches_fd_under_beta_perturbation() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();
    let dir_u = toy_probe_vector(p, 501);
    let dir_w = toy_probe_vector(p, 502);
    let v = toy_probe_vector(p, 503);

    let workspace = family
        .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
        .expect("workspace build")
        .expect("workspace present");
    let want = workspace
        .second_directional_derivative_operator(&dir_u, &dir_w)
        .expect("d2H op call")
        .expect("d2H op present")
        .mul_vec(&v);

    let eps = 1e-5;
    let make_state = |scale: f64| ParameterBlockState {
        beta: &state.beta + &(dir_w.mapv(|b| scale * b)),
        eta: state.eta.clone(),
    };
    let plus_ws = family
        .exact_newton_joint_hessian_workspace(
            std::slice::from_ref(&make_state(eps)),
            &[spec.clone()],
        )
        .expect("plus ws")
        .expect("plus ws present");
    let minus_ws = family
        .exact_newton_joint_hessian_workspace(
            std::slice::from_ref(&make_state(-eps)),
            &[spec.clone()],
        )
        .expect("minus ws")
        .expect("minus ws present");
    let dh_plus = plus_ws
        .directional_derivative_operator(&dir_u)
        .expect("plus dH op call")
        .expect("plus dH op present")
        .mul_vec(&v);
    let dh_minus = minus_ws
        .directional_derivative_operator(&dir_u)
        .expect("minus dH op call")
        .expect("minus dH op present")
        .mul_vec(&v);
    let fd: Array1<f64> = (&dh_plus - &dh_minus).mapv(|x| x / (2.0 * eps));

    for i in 0..p {
        let scale = want[i].abs().max(1.0);
        let tol = 5e-5 * scale + 5e-7;
        assert!(
            (want[i] - fd[i]).abs() <= tol,
            "d2H FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
            want[i],
            fd[i],
            tol,
        );
    }
}

/// FD check for the CTN barrier `∇²B v` operator itself: centered FD on the
/// log-likelihood gradient w.r.t. β reproduces `H(β) v` (to within FD
/// truncation). This is the `μ Dᵀ((Dv)/c²)` formula plus the
/// β-independent `X_val^T W X_val` term.
#[test]
pub(crate) fn ctn_hessian_matvec_matches_grad_fd() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();
    let v = toy_probe_vector(p, 601);

    let workspace = family
        .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
        .expect("workspace build")
        .expect("workspace present");
    let hv = workspace
        .hessian_matvec(&v)
        .expect("matvec call")
        .expect("matvec present");

    let eps = 1e-6;
    // CTN's `evaluate()` returns the score (gradient of log-likelihood)
    // through the working-set; the joint Hessian is `-d²ℓ/dβ²`, so
    // `H · v ≈ -[grad(β + εv) - grad(β - εv)] / (2ε)`.
    let make_state = |scale: f64| ParameterBlockState {
        beta: &state.beta + &(v.mapv(|b| scale * b)),
        eta: state.eta.clone(),
    };
    let grad_at = |st: &ParameterBlockState| -> Array1<f64> {
        let eval = family
            .evaluate(std::slice::from_ref(st))
            .expect("evaluate must succeed");
        match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
            _ => panic!("CTN must report ExactNewton working set"),
        }
    };
    let grad_plus = grad_at(&make_state(eps));
    let grad_minus = grad_at(&make_state(-eps));
    // The score is +∂ℓ/∂β, and H = -∂²ℓ/∂β². Centered FD on the score gives
    // dscore/dβ · v = -H · v, so we negate to compare against `hv`.
    let fd: Array1<f64> = (&grad_plus - &grad_minus).mapv(|x| -x / (2.0 * eps));

    for i in 0..p {
        let scale = hv[i].abs().max(1.0);
        let tol = 1e-4 * scale + 1e-6;
        assert!(
            (hv[i] - fd[i]).abs() <= tol,
            "Hv FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
            hv[i],
            fd[i],
            tol,
        );
    }
}

#[test]
pub(crate) fn ctn_scop_gradient_matches_loglikelihood_fd() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();

    let analytic = family
        .exact_newton_joint_gradient_evaluation(std::slice::from_ref(&state), &[spec])
        .expect("SCOP analytic gradient evaluation")
        .expect("SCOP analytic gradient must be present");
    assert_eq!(analytic.gradient.len(), p);

    let eps = 1e-6;
    for coord in 0..p {
        let mut beta_plus = state.beta.clone();
        beta_plus[coord] += eps;
        let plus_state = ParameterBlockState {
            beta: beta_plus,
            eta: state.eta.clone(),
        };
        let ll_plus = family
            .log_likelihood_only(std::slice::from_ref(&plus_state))
            .expect("positive perturbation remains feasible");

        let mut beta_minus = state.beta.clone();
        beta_minus[coord] -= eps;
        let minus_state = ParameterBlockState {
            beta: beta_minus,
            eta: state.eta.clone(),
        };
        let ll_minus = family
            .log_likelihood_only(std::slice::from_ref(&minus_state))
            .expect("negative perturbation remains feasible");

        let fd = (ll_plus - ll_minus) / (2.0 * eps);
        let scale = fd.abs().max(analytic.gradient[coord].abs()).max(1.0);
        let tol = 5e-6 * scale + 5e-8;
        assert!(
            (analytic.gradient[coord] - fd).abs() <= tol,
            "SCOP gradient FD mismatch at {coord}: analytic={:.6e}, fd={:.6e}, tol={:.6e}",
            analytic.gradient[coord],
            fd,
            tol,
        );
    }
}

#[test]
pub(crate) fn ctn_exact_newton_joint_gradient_evaluation_matches_evaluate() {
    // The joint-Newton inner solver prefers
    // `exact_newton_joint_gradient_evaluation` over `evaluate()` to refresh
    // the gradient between cycles. Lock in that the override returns
    // exactly the same log-likelihood and flat gradient that the dense
    // path produces (up to floating-point summation order).
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let p = spec.design.ncols();

    let eval = family
        .evaluate(std::slice::from_ref(&state))
        .expect("evaluate must succeed on the toy fixture");
    let want_ll = eval.log_likelihood;
    let want_grad = match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
        _ => panic!("CTN must report an ExactNewton block working set"),
    };
    assert_eq!(want_grad.len(), p);

    let gradient_eval = family
        .exact_newton_joint_gradient_evaluation(std::slice::from_ref(&state), &[spec.clone()])
        .expect("gradient-only call")
        .expect("gradient-only result must be present");
    assert!(
        (want_ll - gradient_eval.log_likelihood).abs() <= 1e-12 * want_ll.abs().max(1.0) + 1e-12,
        "log-likelihood mismatch: evaluate={:.6e}, gradient-only={:.6e}",
        want_ll,
        gradient_eval.log_likelihood,
    );
    assert_eq!(gradient_eval.gradient.len(), p);
    for i in 0..p {
        let tol = 1e-12 * want_grad[i].abs().max(1.0) + 1e-12;
        assert!(
            (want_grad[i] - gradient_eval.gradient[i]).abs() <= tol,
            "gradient mismatch at {i}: evaluate={:.6e}, gradient-only={:.6e}",
            want_grad[i],
            gradient_eval.gradient[i],
        );
    }
}

// ---------------------------------------------------------------------------
// SPEC-5 response-direction function-space penalty (#2306)
// ---------------------------------------------------------------------------

/// Deterministic well-spread response sample for the response-basis penalty
/// tests. A monotone spread over [-2, 2] gives well-separated I-spline knots.
fn spec5_penalty_response() -> Array1<f64> {
    Array1::from_iter((0..200).map(|i| (i as f64) / 199.0 * 4.0 - 2.0))
}

/// The realized response-direction penalty is the EXACT function-space
/// I-spline roughness Gram embedded with an unpenalized location row/column —
/// never a coefficient-difference operator. This is the concrete #2306
/// cutover: `build_response_basis` must emit `ispline_function_penalties`, not
/// `create_difference_penalty_matrix`.
#[test]
pub(crate) fn ctn_response_penalty_is_exact_ispline_function_roughness() {
    let config = TransformationNormalConfig::default();
    let response = spec5_penalty_response();
    let (val_basis, _deriv, penalties, knots, _transform) =
        build_response_basis(&response, &config).expect("response basis builds");
    assert!(
        !penalties.is_empty(),
        "response basis must carry at least the primary roughness penalty"
    );
    let p_resp = val_basis.ncols();
    let p_shape = p_resp - 1;
    let order = config.response_penalty_order;

    let expected = ispline_function_penalties(knots.view(), config.response_degree, order, false)
        .expect("exact I-spline function roughness")
        .roughness;
    assert_eq!(expected.dim(), (p_shape, p_shape));

    let primary = &penalties[0];
    assert_eq!(primary.dim(), (p_resp, p_resp));
    // Location row/column is unpenalized.
    for j in 0..p_resp {
        assert!(
            primary[[0, j]].abs() < 1e-15 && primary[[j, 0]].abs() < 1e-15,
            "location row/column must be unpenalized at index {j}"
        );
    }
    // Shape block equals the exact function-space roughness bitwise-close.
    let block = primary.slice(s![1.., 1..]);
    for r in 0..p_shape {
        for c in 0..p_shape {
            assert!(
                (block[[r, c]] - expected[[r, c]]).abs()
                    <= 1e-12 * expected[[r, c]].abs().max(1.0),
                "shape block ({r},{c}) = {:.6e} but exact function roughness = {:.6e}",
                block[[r, c]],
                expected[[r, c]],
            );
        }
    }

    // Discriminator: the retired coefficient-difference operator is a DIFFERENT
    // matrix, so the cutover genuinely changed the penalized metric.
    let difference =
        gam_terms::basis::create_difference_penalty_matrix(p_shape, order, None).unwrap();
    let mut max_rel = 0.0_f64;
    for r in 0..p_shape {
        for c in 0..p_shape {
            let scale = expected[[r, c]].abs().max(difference[[r, c]].abs()).max(1e-9);
            max_rel = max_rel.max((expected[[r, c]] - difference[[r, c]]).abs() / scale);
        }
    }
    assert!(
        max_rel > 0.1,
        "function-space roughness must differ materially from the difference operator (max rel {max_rel:.3e})"
    );
}

/// The response-direction penalty is the roughness of the represented
/// I-spline value function: `βᵀ S β = ∫ (dᵐ/dyᵐ Σ_k β_k I_k(y))² dy`.
/// Matching this quadrature (and NOT the scale-free difference operator)
/// proves the penalty is a scale/knot-width-aware function-space metric.
#[test]
pub(crate) fn ctn_response_penalty_matches_direct_function_roughness_quadrature() {
    let config = TransformationNormalConfig::default();
    let response = spec5_penalty_response();
    let (val_basis, _deriv, penalties, knots, _transform) =
        build_response_basis(&response, &config).expect("response basis builds");
    let p_resp = val_basis.ncols();
    let p_shape = p_resp - 1;
    let order = config.response_penalty_order;

    // Deterministic shape coefficients, location coefficient left at zero.
    let beta_shape = toy_probe_vector(p_shape, 0x5C0F_u64.wrapping_add(order as u64));
    let mut beta = Array1::<f64>::zeros(p_resp);
    beta.slice_mut(s![1..]).assign(&beta_shape);

    let quad_form = beta.dot(&penalties[0].dot(&beta));
    assert!(quad_form > 0.0, "roughness of a nontrivial shape must be positive");

    // Direct Simpson quadrature of the m-th derivative squared over the full
    // knot support.
    let lower = *knots.first().unwrap();
    let upper = *knots.last().unwrap();
    let panels = 40_000usize; // even -> composite Simpson
    let step = (upper - lower) / panels as f64;
    let grid = Array1::from_iter((0..=panels).map(|i| lower + step * i as f64));
    let deriv_basis = gam_terms::basis::create_ispline_derivative_dense(
        grid.view(),
        &knots,
        config.response_degree,
        order,
    )
    .expect("m-th derivative basis on quadrature grid");
    assert_eq!(deriv_basis.dim(), (panels + 1, p_shape));
    let fm = deriv_basis.dot(&beta_shape);
    let mut integral = fm[0] * fm[0] + fm[panels] * fm[panels];
    for i in 1..panels {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        integral += weight * fm[i] * fm[i];
    }
    integral *= step / 3.0;

    let rel = (quad_form - integral).abs() / quad_form.abs();
    assert!(
        rel < 2e-4,
        "penalty quadratic form {quad_form:.8e} must match direct function roughness {integral:.8e} (rel {rel:.3e})"
    );

    // The scale-free difference operator does NOT reproduce the function-space
    // roughness — this is exactly why the difference operator was wrong.
    let difference =
        gam_terms::basis::create_difference_penalty_matrix(p_shape, order, None).unwrap();
    let difference_form = beta_shape.dot(&difference.dot(&beta_shape));
    let diff_rel = (difference_form - integral).abs() / integral.abs();
    assert!(
        diff_rel > 0.1,
        "difference operator quadratic form {difference_form:.8e} must NOT match the function roughness {integral:.8e} (rel {diff_rel:.3e})"
    );
}

/// The assembled covariate-direction tensor penalty is the exact SPEC-5
/// function-measure roughness `½ βᵀ (G_y ⊗ S_x) β` with `G_y = Vᵀ W V` the
/// response value-basis mass Gram — NOT the retired identity shape-row factor
/// (`diag(0,1,…,1) ⊗ S_x`), which is coefficient geometry (gam#2306). This
/// pins three discriminating properties: the left factor equals `G_y`, a
/// centering field in `null(S_x)` stays exactly unpenalized (free intercept),
/// and a genuine covariate main-effect of the location field IS smoothed — the
/// behavior the identity shape-row factor dropped.
#[test]
pub(crate) fn ctn_covariate_penalty_is_response_mass_gram_function_roughness() {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());
    let cov_design = array![[1.0, 0.2], [1.0, -0.1], [1.0, 0.4], [1.0, -0.3]];
    let p_cov = cov_design.ncols();

    // Rank-1 covariate roughness with a known null vector [1, 1]: `S_x v = 0`,
    // so any covariate coefficient row proportional to [1, 1] carries no
    // roughness while [1, 0] does.
    let s_cov = array![[1.0, -1.0], [-1.0, 1.0]];

    let expected_g_resp =
        weighted_function_gram(val_basis.view(), weights.view(), p_resp, "response")
            .expect("response mass gram");

    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        val_basis,
        deriv_basis,
        vec![],
        knots,
        toy_scop_ctn_config().response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
        vec![PenaltyMatrix::Dense(s_cov.clone())],
        &toy_scop_ctn_config(),
        None,
    )
    .expect("toy transformation family");

    // Covariate penalties are assembled first; index 0 is `G_y ⊗ S_x`.
    let PenaltyMatrix::KroneckerFactored { left, right } = &family.tensor_penalties[0] else {
        panic!("covariate-direction penalty must be Kronecker-factored");
    };
    assert_eq!(right, &s_cov, "right factor must be the covariate roughness Gram");
    assert_eq!(left.dim(), (p_resp, p_resp));
    for ((r, c), &value) in left.indexed_iter() {
        let want = expected_g_resp[[r, c]];
        assert!(
            (value - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "left factor [{r},{c}] = {value} must equal G_y {want}"
        );
    }

    // Discriminator: G_y is materially different from the retired identity
    // shape-row factor diag(0, 1, …, 1) — the cutover genuinely changed the
    // penalized metric.
    let mut max_gap = 0.0_f64;
    for r in 0..p_resp {
        for c in 0..p_resp {
            let old: f64 = if r == c && r > 0 { 1.0 } else { 0.0 };
            let scale = expected_g_resp[[r, c]].abs().max(old.abs()).max(1e-9);
            max_gap = max_gap.max((expected_g_resp[[r, c]] - old).abs() / scale);
        }
    }
    assert!(
        max_gap > 0.1,
        "G_y must differ materially from the identity shape-row factor (max rel {max_gap:.3e})"
    );

    let penalty_dense = family.tensor_penalties[0].to_dense();
    let quad_form = |a_rows: &[[f64; 2]]| -> f64 {
        let mut beta = Array1::<f64>::zeros(p_resp * p_cov);
        for (k, row) in a_rows.iter().enumerate() {
            for (a, &value) in row.iter().enumerate() {
                beta[k * p_cov + a] = value;
            }
        }
        beta.dot(&penalty_dense.dot(&beta))
    };

    // A location field whose covariate coefficients lie in null(S_x) (∝ [1,1])
    // is a null-roughness direction: the free intercept stays unpenalized.
    let mut null_location = vec![[0.0, 0.0]; p_resp];
    null_location[0] = [1.0, 1.0];
    assert!(
        quad_form(&null_location).abs() < 1e-12,
        "a centering field in null(S_x) must carry zero covariate roughness"
    );

    // A genuine covariate main-effect of the location field (∉ null(S_x)) IS
    // penalized now — exactly the term the identity shape-row factor dropped.
    let mut main_effect = vec![[0.0, 0.0]; p_resp];
    main_effect[0] = [1.0, 0.0];
    let expected_main = expected_g_resp[[0, 0]] * 1.0; // [1,0] S_x [1,0]ᵀ = 1
    let got_main = quad_form(&main_effect);
    assert!(
        (got_main - expected_main).abs() <= 1e-12 * (1.0 + expected_main.abs()),
        "location main-effect roughness {got_main} must equal G_y[0,0]·(vᵀS_x v) {expected_main}"
    );
    assert!(
        got_main > 1e-6,
        "location main-effect must be smoothed (was dropped by the identity shape-row factor)"
    );
}

/// SPEC-5 basis-change invariance of the covariate-direction penalty: the
/// penalized roughness of a fixed transformation `h` is invariant to the choice
/// of covariate design basis. Under a reparameterization `Ψ' = Ψ T` (T
/// invertible), the same function has covariate coefficients `A' = A T⁻ᵀ` and
/// the roughness Gram transforms covariantly to `S_x' = Tᵀ S_x T`, so the
/// assembled `βᵀ (G_y ⊗ S_x) β` is unchanged. An identity coefficient factor
/// would NOT be basis-covariant — this pins the penalty as a genuine
/// function-space quantity (gam#2306).
#[test]
pub(crate) fn ctn_covariate_penalty_is_basis_change_invariant() {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());

    // Base covariate design Ψ and an invertible reparameterization T (det 1).
    let psi_design = array![[1.0, 0.2], [1.0, -0.1], [1.0, 0.4], [1.0, -0.3]];
    let p_cov = psi_design.ncols();
    let t = array![[1.0, 0.5], [0.0, 1.0]];
    // T⁻ᵀ for T = [[1, 0.5], [0, 1]] is [[1, 0], [-0.5, 1]].
    let t_inv_t = array![[1.0, 0.0], [-0.5, 1.0]];
    let psi_reparam = psi_design.dot(&t);
    let s_cov = array![[2.0, -1.0], [-1.0, 2.0]];
    let s_cov_reparam = t.t().dot(&s_cov).dot(&t);

    let build = |cov: Array2<f64>, pen: Array2<f64>| -> (PenaltyMatrix, usize) {
        let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov)),
            vec![PenaltyMatrix::Dense(pen)],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("toy transformation family");
        (family.tensor_penalties[0].clone(), p_resp)
    };

    let (penalty_base, p_resp) = build(psi_design.clone(), s_cov.clone());
    let (penalty_reparam, p_resp_b) = build(psi_reparam, s_cov_reparam);
    assert_eq!(p_resp, p_resp_b);

    // A fixed transformation as a coefficient matrix A (p_resp × p_cov) in the
    // base basis, and its reparameterized coefficients A' = A T⁻ᵀ.
    let a_flat = toy_probe_vector(p_resp * p_cov, 0xB6C1_u64);
    let a = a_flat
        .clone()
        .into_shape_with_order((p_resp, p_cov))
        .expect("reshape A");
    let a_reparam = a.dot(&t_inv_t);

    let beta_base = a_flat;
    let beta_reparam = a_reparam
        .into_shape_with_order(p_resp * p_cov)
        .expect("flatten A'");

    let base_dense = penalty_base.to_dense();
    let reparam_dense = penalty_reparam.to_dense();
    let roughness_base = beta_base.dot(&base_dense.dot(&beta_base));
    let roughness_reparam = beta_reparam.dot(&reparam_dense.dot(&beta_reparam));

    assert!(
        roughness_base > 1e-6,
        "probe transformation must carry nontrivial covariate roughness"
    );
    let rel = (roughness_base - roughness_reparam).abs() / roughness_base.abs();
    assert!(
        rel <= 1e-10,
        "covariate roughness must be basis-change invariant: base {roughness_base:.8e} vs reparam {roughness_reparam:.8e} (rel {rel:.3e})"
    );
}

/// Unsupported response-direction derivative orders are hard errors, not
/// silently skipped no-ops (the retired `if order==0 || order>=p {return Ok}`
/// path). Order 0 is the value function; an order above the I-spline value
/// degree has an identically-zero derivative and carries no roughness.
#[test]
pub(crate) fn ctn_response_penalty_rejects_unsupported_derivative_order() {
    let response = spec5_penalty_response();

    // response_degree = 1 -> value_degree = 2; order 3 is unsupported.
    let too_high = TransformationNormalConfig {
        response_degree: 1,
        response_penalty_order: 3,
        response_extra_penalty_orders: vec![],
        ..TransformationNormalConfig::default()
    };
    let err = build_response_basis(&response, &too_high)
        .expect_err("derivative order above the value degree must be rejected");
    assert!(
        err.contains("exceeds the I-spline value degree"),
        "unexpected too-high-order error: {err}"
    );

    // Order 0 is not a roughness penalty.
    let zero_order = TransformationNormalConfig {
        response_penalty_order: 0,
        response_extra_penalty_orders: vec![],
        ..TransformationNormalConfig::default()
    };
    let err0 = build_response_basis(&response, &zero_order)
        .expect_err("derivative order 0 must be rejected");
    assert!(
        err0.contains("derivative order must be >= 1"),
        "unexpected zero-order error: {err0}"
    );

    // An unsupported order supplied only through the EXTRA orders list is also
    // rejected (no silent skip on the secondary path).
    let extra_bad = TransformationNormalConfig {
        response_degree: 1,
        response_penalty_order: 1,
        response_extra_penalty_orders: vec![3],
        ..TransformationNormalConfig::default()
    };
    let err_extra = build_response_basis(&response, &extra_bad)
        .expect_err("unsupported extra derivative order must be rejected");
    assert!(
        err_extra.contains("exceeds the I-spline value degree"),
        "unexpected extra-order error: {err_extra}"
    );
}

/// gam#2600: every CTN shape-block penalty must vanish on the AFFINE
/// transformation.
///
/// The response-shape block exists to bend `h` away from the affine
/// `(y − μ)/σ` map, so `h' ≡ const > 0` is the null this penalty family is
/// supposed to recover as `λ → ∞`. Before the fix two of the three assembled
/// penalties reached that direction — the extra order-1 roughness (anchored
/// I-splines give order-`m` structural nullity `m − 1`, so order 1 is positive
/// definite) and the double-penalty shape ridge — which put the penalty's
/// unique minimiser at `α = 0`, i.e. `h' ≡ 0`: a CONSTANT transformation that
/// maps every response to one score and at which the likelihood is undefined.
/// At `λ ≈ 1370` on the wine arm that collapse was measured to be objectively
/// better by `Δobj = +9.987e5`, buying ≈1e6 of penalty for ≈50 of likelihood at
/// `ρ = 1.000`, and every downstream symptom followed from it.
#[test]
pub(crate) fn ctn_shape_penalties_annihilate_the_affine_transformation_2600() {
    let response = skewed_response(64);
    let config = TransformationNormalConfig::default();
    assert!(
        config.response_extra_penalty_orders.contains(&1) && config.double_penalty,
        "this pin is only meaningful while the default carries the two penalties that \
         reached the affine direction: extra orders {:?}, double_penalty {}",
        config.response_extra_penalty_orders,
        config.double_penalty
    );
    let (resp_val, resp_deriv, resp_penalties, knots, transform) =
        build_response_basis(&response, &config).expect("response basis builds");
    let p_resp = resp_val.ncols();
    let p_shape = p_resp - 1;
    let affine = affine_shape_direction(knots.view(), config.response_degree, p_shape)
        .expect("affine shape direction");

    // (1) The direction is the affine transformation itself: unit slope at every
    // observation, to floating point. This is what makes the null-space claim
    // below a statement about the MODEL and not about an arbitrary vector.
    for i in 0..response.len() {
        let slope: f64 = (0..p_shape)
            .map(|k| affine[k] * resp_deriv[[i, k + 1]])
            .sum();
        assert!(
            (slope - 1.0).abs() < 1.0e-10,
            "affine direction must give h' = 1 at row {i}, got {slope:.17e}"
        );
    }

    let mut beta_affine = Array1::<f64>::zeros(p_resp);
    beta_affine.slice_mut(s![1..]).assign(&affine);
    let affine_norm_sq = beta_affine.dot(&beta_affine);

    // (2) Every response-direction penalty is zero there — and still bites on a
    // bent shape, so the assertion cannot be satisfied by a zero matrix.
    for (index, penalty) in resp_penalties.iter().enumerate() {
        let quad = beta_affine.dot(&penalty.dot(&beta_affine));
        let scale = affine_norm_sq * penalty.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            quad.abs() <= 1.0e-12 * scale,
            "response penalty {index} does not annihilate the affine transformation: \
             quadratic form {quad:.6e} against scale {scale:.6e}"
        );
        let bent = toy_probe_vector(p_resp, 101 + index as u64);
        let bent_quad = bent.dot(&penalty.dot(&bent));
        assert!(
            bent_quad > 0.0,
            "response penalty {index} is inert on a bent shape ({bent_quad:.6e}); the \
             null-space assertion above would then be vacuous"
        );
    }

    // (3) The same for the ASSEMBLED tensor penalties, which is what the
    // optimizer actually sees. This is where the double-penalty ridge lives, and
    // it is the sum over these that decides whether collapse is preferred.
    let n = response.len();
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        resp_val,
        resp_deriv,
        resp_penalties,
        knots,
        config.response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::ones((n, 1)))),
        vec![],
        &config,
        None,
    )
    .expect("intercept-only CTN family");
    assert_eq!(
        family.tensor_penalties.len(),
        3,
        "intercept-only default config assembles order-2 roughness, order-1 roughness, \
         and the double-penalty ridge"
    );
    for (index, penalty) in family.tensor_penalties.iter().enumerate() {
        let dense = penalty.to_dense();
        let quad = beta_affine.dot(&dense.dot(&beta_affine));
        let scale = affine_norm_sq * dense.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            quad.abs() <= 1.0e-12 * scale,
            "tensor penalty {index} does not annihilate the affine transformation: \
             quadratic form {quad:.6e} against scale {scale:.6e}"
        );
        let bent = toy_probe_vector(p_resp, 211 + index as u64);
        let bent_quad = bent.dot(&dense.dot(&bent));
        assert!(
            bent_quad > 0.0,
            "tensor penalty {index} is inert on a bent shape ({bent_quad:.6e})"
        );
    }
}

/// gam#2600: the two exact-curvature entry points that carry no dense oracle of
/// their own — the Hessian DIAGONAL and the projected directional TRACE — must
/// agree with the dense assembly they are shortcuts for.
///
/// Both are row loops over the same per-row factors as
/// `scop_gradient_and_negative_hessian` and `scop_hessian_directional_derivative`,
/// and the endpoint-normalizer deletion edited all four. The operator tests
/// cover `mul_vec`/`mul_mat` against dense; nothing covered these two, so a term
/// dropped from a shortcut and not from the dense path (or the reverse) had no
/// gate. This is that gate, and it is an identity, so it carries no tolerance
/// beyond accumulation order.
#[test]
pub(crate) fn ctn_hessian_diagonal_and_projected_trace_match_dense_assembly_2600() {
    let psi = array![0.15, -0.10];
    let (family, _, state, spec) = toy_family_and_derivatives(&psi);
    let quantities = family
        .row_quantities(&state.beta)
        .expect("toy row quantities");
    let p_total = state.beta.len();

    // (1) The exact diagonal is the diagonal of the exact dense Hessian.
    let (_, dense) = family
        .scop_gradient_and_negative_hessian(&state.beta, &quantities)
        .expect("dense SCOP information");
    let diagonal = family
        .scop_hessian_diagonal(&state.beta, &quantities)
        .expect("SCOP information diagonal");
    assert_eq!(diagonal.len(), p_total);
    for index in 0..p_total {
        let (got, want) = (diagonal[index], dense[[index, index]]);
        assert!(
            (got - want).abs() <= 1.0e-12 * want.abs().max(1.0),
            "diagonal[{index}] = {got:.17e} against dense[{index},{index}] = {want:.17e}"
        );
    }

    // (2) The projected directional trace is `tr(Fᵀ · dH[u] · F)`, assembled here
    // from the dense directional derivative so the two routes cannot drift.
    let direction = toy_probe_vector(p_total, 7_001);
    let factor = {
        let mut columns = Array2::<f64>::zeros((p_total, 3));
        for column in 0..3 {
            let probe = toy_probe_vector(p_total, 7_100 + column as u64);
            for row in 0..p_total {
                columns[[row, column]] = probe[row];
            }
        }
        columns
    };
    let dense_dh = family
        .scop_hessian_directional_derivative(&state.beta, &direction, &quantities)
        .expect("dense SCOP dH");
    let mut expected = 0.0;
    for column in 0..factor.ncols() {
        let f = factor.column(column);
        for i in 0..p_total {
            for j in 0..p_total {
                expected += f[i] * dense_dh[[i, j]] * f[j];
            }
        }
    }
    let workspace = family
        .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec])
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&direction)
        .expect("dH operator call")
        .expect("dH operator present");
    let got = dh_op.trace_projected_factor(&factor);
    assert!(
        (got - expected).abs() <= 1.0e-10 * expected.abs().max(1.0),
        "projected trace {got:.17e} against dense assembly {expected:.17e}"
    );
    // The fixture has to have curvature in this direction, or both sides are 0.
    assert!(
        expected.abs() > 1.0e-6,
        "the probe direction produces no directional curvature ({expected:.3e}); \
         the comparison above would then be vacuous"
    );
}

/// gam#979. The CTN Hessian's β-derivatives `D H[u]` and `D² H[u, v]` are each
/// ONE weighted-Gram assembly (the `scop_curvature` module header), and the
/// outer engine's operators now do nothing but read that assembly — which is
/// what took the large-scale preprocessor off a 17 s-per-projection row sweep
/// that never left one core.
///
/// Removing the second, row-streaming implementation removed the only thing
/// the assembly was ever compared against, and comparing an operator with the
/// builder it reads is not a comparison. So the gate is against the object
/// these matrices are derivatives OF: a central difference of the value
/// Hessian the inner solve actually minimises. A wrong block factor, a wrong
/// power of `hp`, a transposed `(k, l)` placement, or a direction contracted
/// on the wrong axis all move the analytic side off the difference.
#[test]
pub(crate) fn ctn_hessian_beta_derivatives_are_derivatives_of_the_value_hessian_979() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let p_total = state.beta.len();
    let u = toy_probe_vector(p_total, 9_701);
    let v = toy_probe_vector(p_total, 9_702);
    let step = 1.0e-6;

    let value_hessian_at = |beta: &Array1<f64>| {
        let quantities = family.row_quantities(beta).expect("shifted row quantities");
        family
            .scop_gradient_and_negative_hessian(beta, &quantities)
            .expect("dense SCOP information")
            .1
    };
    let directional_at = |beta: &Array1<f64>, direction: &Array1<f64>| {
        let quantities = family.row_quantities(beta).expect("shifted row quantities");
        family
            .scop_hessian_directional_derivative(beta, direction, &quantities)
            .expect("dense SCOP dH")
    };
    let shifted = |direction: &Array1<f64>, scale: f64| &state.beta + &(direction * scale);

    let quantities = family
        .row_quantities(&state.beta)
        .expect("base row quantities");

    // D H[u] against the central difference of H along u.
    let analytic_dh = family
        .scop_hessian_directional_derivative(&state.beta, &u, &quantities)
        .expect("dense SCOP dH");
    let fd_dh = (value_hessian_at(&shifted(&u, step)) - value_hessian_at(&shifted(&u, -step)))
        / (2.0 * step);
    assert_matrix_derivativefd_rel(&fd_dh, &analytic_dh, 1.0e-5, "CTN dH[u] against d/dt H");

    // D² H[u, v] against the central difference of D H[u] along v.
    let analytic_d2h = family
        .scop_hessian_second_directional_derivative(&state.beta, &u, &v, &quantities)
        .expect("dense SCOP d2H");
    let fd_d2h = (directional_at(&shifted(&v, step), &u)
        - directional_at(&shifted(&v, -step), &u))
        / (2.0 * step);
    assert_matrix_derivativefd_rel(
        &fd_d2h,
        &analytic_d2h,
        1.0e-5,
        "CTN d2H[u, v] against d/dt dH[u]",
    );

    // Both differences have to be reading real curvature, or every assertion
    // above passes on a pair of zero matrices.
    let peak = |matrix: &Array2<f64>| matrix.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        peak(&analytic_dh) > 1.0e-4,
        "the probe direction produces no first-order curvature drift ({:.3e})",
        peak(&analytic_dh)
    );
    assert!(
        peak(&analytic_d2h) > 1.0e-4,
        "the probe pair produces no second-order curvature drift ({:.3e})",
        peak(&analytic_d2h)
    );
}

/// gam#2600, the defect this issue turned out to be: the CTN inner objective
/// must be COERCIVE — it must go to `+∞` in every direction of `β`, so that a
/// minimizer exists at all.
///
/// The direction that failed is the escape ray: raise the unpenalized location
/// column to `κ·c` and contract the shape to `α/κ`. Under the old
/// endpoint-renormalized density `φ(h)h' / [Φ(h_hi) − Φ(h_lo)]` the three
/// transformed quantities `h`, `h_lo`, `h_hi` move together, the conditional law
/// converges to a truncated exponential in the normalized shape coordinate, and
/// the objective converges to a FINITE limit from above — measured on the wine
/// fixture as `141.0858 → 141.0604164` over `c ∈ [1, ∞)`, monotone and never
/// stationary. Every solver-side hypothesis on that issue was a symptom of an
/// inner problem whose infimum was simply not attained.
///
/// Under the most-likely-transformation density `φ(h)h'` the same ray costs
/// `½Σh² ~ ½nκ²c²` from the Gaussian kernel AND `n log κ` from the `−log h'`
/// barrier, so it diverges quadratically. The assertion carries no tuned
/// constant: it is monotonicity plus divergence past an arbitrary large bound.
#[test]
pub(crate) fn ctn_penalized_objective_is_coercive_in_the_location_column_2600() {
    let response = skewed_response(64);
    let n = response.len();
    let config = TransformationNormalConfig::default();
    let (resp_val, resp_deriv, resp_penalties, knots, transform) =
        build_response_basis(&response, &config).expect("response basis builds");
    let p_resp = resp_val.ncols();
    let p_shape = p_resp - 1;
    let affine = affine_shape_direction(knots.view(), config.response_degree, p_shape)
        .expect("affine shape direction");
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        resp_val,
        resp_deriv,
        resp_penalties,
        knots,
        config.response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::ones((n, 1)))),
        vec![],
        &config,
        None,
    )
    .expect("intercept-only CTN family");
    let rho = family
        .penalty_scale_log_lambdas()
        .expect("data-scaled smoothing seed");
    let dense: Vec<Array2<f64>> = family
        .tensor_penalties
        .iter()
        .map(|penalty| penalty.to_dense())
        .collect();

    // The ray. `c` and the shape scale are read off the fixture rather than
    // chosen: `c` is one response standard deviation on the latent scale and the
    // shape is the affine transformation that standardizes the response, which
    // is where an honest fit sits.
    let mean = response.sum() / n as f64;
    let variance = response.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / n as f64;
    let base_slope = 1.0 / variance.sqrt();
    let penalized_objective = |kappa: f64| -> f64 {
        let mut beta = Array1::<f64>::zeros(p_resp);
        for k in 0..p_shape {
            beta[k + 1] = (base_slope / kappa) * affine[k];
        }
        beta[0] = kappa;
        let quantities = family
            .row_quantities(&beta)
            .expect("row quantities on the escape ray");
        let penalty: f64 = dense
            .iter()
            .enumerate()
            .map(|(index, matrix)| 0.5 * rho[index].exp() * beta.dot(&matrix.dot(&beta)))
            .sum();
        -quantities.log_likelihood + penalty
    };

    let mut previous = penalized_objective(1.0);
    let base = previous;
    let mut kappa = 2.0;
    // 2^1 … 2^20: `h` reaches ~1e6, which is the family's own
    // `TRANSFORMATION_NORMAL_H_ABS_MAX` domain bound, so this walks the ray as
    // far as the model admits it.
    for _ in 0..19 {
        let objective = penalized_objective(kappa);
        assert!(
            objective > previous,
            "the objective must rise along the escape ray, but at κ={kappa} it fell \
             {previous:.9} → {objective:.9}; under the endpoint-renormalized density it \
             FELL monotonically to a finite limit, which is why the inner solve had no mode"
        );
        previous = objective;
        kappa *= 2.0;
    }
    // Divergence, not merely monotonicity: a monotone sequence can still be
    // bounded, and a bounded one is exactly the defect. `1e6` is an arbitrary
    // large bound, not a threshold — the true growth here is ~½n κ² ≈ 3e13.
    assert!(
        previous > base + 1.0e6,
        "the objective is bounded along the escape ray: {base:.6e} → {previous:.6e}. \
         A bounded ray means the infimum is not attained and no inner mode exists."
    );
}

/// gam#2600: the CTN negative log-likelihood is CONVEX in the coefficients, so
/// its observed information is positive semidefinite everywhere on the feasible
/// set — the property that makes a most-likely-transformation model well posed
/// (Hothorn–Möst–Bühlmann 2018). `−log φ(h) = ½h²` is a convex quadratic in `β`
/// because `h` is linear in `β`, and `−log h'` is convex because `h'` is linear
/// in `β`; the exact Hessian is `Σ w (∇h ∇hᵀ + ∇h' ∇h'ᵀ / h'²)`, a sum of two
/// Gram matrices.
///
/// The endpoint renormalizer broke exactly this: `log Z = log[Φ(u) − Φ(l)]` is
/// CONCAVE in `(l, u)` by Prékopa, so `−log Z` contributed a concave term and
/// the assembled information was indefinite. `resolvable_negative_curvature=true`
/// on every terminal cycle of every refusal recorded on that issue was that
/// indefiniteness.
#[test]
pub(crate) fn ctn_observed_information_is_positive_semidefinite_2600() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let p_total = state.beta.len();
    // The base point, plus perturbations along the escape ray and along random
    // feasible directions — the ray first, because that is the direction whose
    // curvature the renormalizer flipped.
    let mut points = vec![state.beta.clone()];
    for kappa in [4.0_f64, 64.0] {
        let mut beta = state.beta.clone();
        for (index, value) in beta.iter_mut().enumerate() {
            if index < family.covariate_design.ncols() {
                *value *= kappa;
            } else {
                *value /= kappa;
            }
        }
        points.push(beta);
    }
    for seed in 0..6_u64 {
        let mut beta = state.beta.clone();
        let probe = toy_probe_vector(p_total, 4_000 + seed);
        for (value, step) in beta.iter_mut().zip(probe.iter()) {
            *value += 0.15 * step;
        }
        points.push(beta);
    }

    let mut checked = 0usize;
    for beta in points {
        let Ok(quantities) = family.row_quantities(&beta) else {
            // A perturbation that leaves the monotonicity cone is not a
            // counter-example to convexity ON the feasible set; skip it.
            continue;
        };
        let (_, hessian) = family
            .scop_gradient_and_negative_hessian(&beta, &quantities)
            .expect("exact SCOP information at a feasible point");
        let (eigenvalues, _) =
            gam_linalg::faer_ndarray::strict_symmetric_eigh(&hessian, faer::Side::Lower)
                .expect("symmetric eigendecomposition of the observed information");
        let largest = eigenvalues.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        // The only admissible negative eigenvalue is backward error of the
        // eigensolver itself, which is `O(ε·‖H‖)`.
        assert!(
            smallest >= -1.0e-9 * largest.max(1.0),
            "the observed information is indefinite: λ_min={smallest:.6e} against \
             λ_max={largest:.6e}. The CTN negative log-likelihood is a sum of two Gram \
             matrices and cannot have negative curvature; a concave term has been \
             reintroduced into it."
        );
        checked += 1;
    }
    assert!(
        checked >= 5,
        "only {checked} of the probe points were feasible; the assertion above is then \
         close to vacuous"
    );
}

/// gam#2600 null recovery: at ANY smoothing strength the CTN penalized objective
/// must prefer a non-degenerate transformation to the constant map.
///
/// This is the behavioural half of
/// [`ctn_shape_penalties_annihilate_the_affine_transformation_2600`], and it is
/// the property that actually failed. Because every shape penalty now vanishes
/// on the affine direction, the penalized objective restricted to the affine ray
/// `α = t · v` is the pure likelihood, and the likelihood diverges as `t → 0`
/// (`log h' → −∞`), so no smoothing strength can buy the collapse. Before the
/// fix the order-1 roughness and the shape ridge both grew like `t²` on that
/// ray, so at `λ ≈ 1370` on the wine arm collapsing was better by
/// `Δobj = +9.987e5` and the fit walked to a constant transformation.
///
/// The assertion is comparative and carries no tuned threshold: the collapsed
/// point must simply score worse than the standardizing affine transformation
/// under a smoothing strength `e^10 ≈ 2.2e4` times the data-scaled seed.
#[test]
pub(crate) fn ctn_penalized_objective_never_prefers_the_constant_transformation_2600() {
    let response = skewed_response(64);
    let n = response.len();
    let config = TransformationNormalConfig::default();
    let (resp_val, resp_deriv, resp_penalties, knots, transform) =
        build_response_basis(&response, &config).expect("response basis builds");
    let p_resp = resp_val.ncols();
    let p_shape = p_resp - 1;
    let affine = affine_shape_direction(knots.view(), config.response_degree, p_shape)
        .expect("affine shape direction");
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let resp_val_kept = resp_val.clone();
    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        &response,
        resp_val,
        resp_deriv,
        resp_penalties,
        knots,
        config.response_degree,
        transform,
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::ones((n, 1)))),
        vec![],
        &config,
        None,
    )
    .expect("intercept-only CTN family");
    // Ten e-folds above the data-scaled seed: far past the strength at which the
    // wine arm collapsed.
    let rho = family
        .penalty_scale_log_lambdas()
        .expect("data-scaled smoothing seed")
        .mapv(|value| value + 10.0);
    let dense: Vec<Array2<f64>> = family
        .tensor_penalties
        .iter()
        .map(|penalty| penalty.to_dense())
        .collect();

    // The standardizing affine transformation `h ≈ (y − ȳ)/sd(y)`, and the same
    // transformation scaled down by a millionfold, which is the collapse.
    let mean = response.sum() / n as f64;
    let variance = response.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / n as f64;
    let reference_slope = 1.0 / variance.sqrt();
    let penalized_objective = |slope: f64| -> (f64, f64) {
        let mut beta = Array1::<f64>::zeros(p_resp);
        for k in 0..p_shape {
            beta[k + 1] = slope * affine[k];
        }
        let mut shape_mean = 0.0;
        for i in 0..n {
            for k in 0..p_shape {
                shape_mean += beta[k + 1] * resp_val_kept[[i, k + 1]];
            }
        }
        beta[0] = -shape_mean / n as f64;
        let quantities = family
            .row_quantities(&beta)
            .expect("row quantities on the affine ray");
        let penalty: f64 = dense
            .iter()
            .enumerate()
            .map(|(index, matrix)| 0.5 * rho[index].exp() * beta.dot(&matrix.dot(&beta)))
            .sum();
        (-quantities.log_likelihood + penalty, penalty)
    };

    let (reference_objective, reference_penalty) = penalized_objective(reference_slope);
    let (collapsed_objective, collapsed_penalty) = penalized_objective(reference_slope * 1.0e-6);
    assert!(
        collapsed_objective > reference_objective,
        "the penalized objective prefers the collapsed transformation: \
         collapsed {collapsed_objective:.6e} <= affine {reference_objective:.6e} \
         (penalties {collapsed_penalty:.6e} vs {reference_penalty:.6e})"
    );
    // Both points sit on the affine ray, so the penalty must not be what decides
    // the comparison above. It is zero there in exact arithmetic, but it is
    // EVALUATED as a cancelling quadratic form at coefficient scale
    // `λ‖β‖²` — with `λ ≈ 3e7` here that carries an `ε/|Δ|` floor of order
    // `1e-7`, so requiring it below machine epsilon of the objective would be
    // asserting against the arithmetic rather than against the model. Assert the
    // property that actually matters: the residual cannot flip the verdict.
    let margin = collapsed_objective - reference_objective;
    assert!(
        reference_penalty.abs() + collapsed_penalty.abs() < 0.5 * margin,
        "residual penalty on the affine ray ({reference_penalty:.6e} and \
         {collapsed_penalty:.6e}) is not negligible against the margin it must not \
         overturn ({margin:.6e})"
    );
}
