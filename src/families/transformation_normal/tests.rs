use super::*;
use crate::custom_family::custom_family_outer_derivatives;
use crate::test_support::assert_matrix_derivativefd;
use ndarray::array;

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
    let likelihood_gram = array![[8.0, 0.0], [0.0, 8.0]];
    let penalties = vec![
        PenaltyMatrix::Dense(array![[2.0, 0.0], [0.0, 2.0]]),
        PenaltyMatrix::Dense(array![[4.0, 0.0], [0.0, 4.0]]),
    ];
    let rho = ctn_penalty_scale_log_lambdas(&penalties, &likelihood_gram);
    assert!((rho[0] - 4.0_f64.ln()).abs() < 1.0e-12);
    assert!((rho[1] - 2.0_f64.ln()).abs() < 1.0e-12);
}

#[test]
pub(crate) fn tensor_psi_penalty_derivatives_follow_shape_only_scop_layout() {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());
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
    assert_shape_penalty_component(&first[0].1, p_resp, &ds0);
    assert_shape_penalty_component(&first[1].1, p_resp, &ds1);

    let second = tensor_derivs[0]
        .s_psi_psi_penalty_components
        .as_ref()
        .expect("second derivatives");
    assert_eq!(second.len(), 1);
    let got_second_indices: Vec<usize> = second[0].iter().map(|(idx, _)| *idx).collect();
    assert_eq!(got_second_indices, vec![1]);
    assert_shape_penalty_component(&second[0][0].1, p_resp, &ds1_second);
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

pub(crate) fn assert_shape_penalty_component(
    penalty: &PenaltyMatrix,
    p_resp: usize,
    expected_right: &Array2<f64>,
) {
    let PenaltyMatrix::KroneckerFactored { left, right } = penalty else {
        panic!("expected KroneckerFactored penalty component");
    };
    assert_eq!(right, expected_right);
    assert_eq!(left.nrows(), p_resp);
    assert_eq!(left.ncols(), p_resp);
    for r in 0..p_resp {
        for c in 0..p_resp {
            let expected = if r == c && r > 0 { 1.0 } else { 0.0 };
            assert_eq!(left[[r, c]], expected);
        }
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

pub(crate) fn toy_family_and_derivatives(
    psi: &Array1<f64>,
) -> (
    TransformationNormalFamily,
    Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ParameterBlockState,
    ParameterBlockSpec,
) {
    let response = array![-1.0, -0.2, 0.6, 1.3];
    let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
    let weights = Array1::from_elem(response.len(), 1.0);
    let offset = Array1::zeros(response.len());
    let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(psi);
    let p_cov = cov_design.ncols();
    let p_total = p_resp * p_cov;
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
    let derivative_blocks =
        vec![build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor psi derivs")];
    // Positive γ across the response axis with mild covariate variation so
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
    let spec = family.block_spec();
    (family, derivative_blocks, state, spec)
}

#[test]
pub(crate) fn ctn_row_quantity_cache_matches_direct_formulas() {
    let psi = array![0.15, -0.10];
    let (family, _, state, _) = toy_family_and_derivatives(&psi);
    let row = family
        .row_quantities(&state.beta)
        .expect("toy row quantities");
    // SCOP-CTN forward: h = X_val · γ²-affine + offset + ε(y−median),
    // h' = X_deriv · γ²-affine + ε.
    let direct_h = family.x_val_kron.scop_affine_squared_forward(&state.beta)
        + family.offset.as_ref()
        + family.response_floor_offset.as_ref();
    let direct_h_prime = family
        .x_deriv_kron
        .scop_affine_squared_forward(&state.beta)
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

    let p_resp = family.response_val_basis.ncols();
    let p_cov = family.covariate_design.ncols();
    let beta_mat = state
        .beta
        .view()
        .into_shape_with_order((p_resp, p_cov))
        .expect("toy beta reshape");
    let cov = family
        .covariate_design
        .try_row_chunk(0..family.n_obs())
        .expect("toy covariate rows");
    let mut h_lower = Array1::<f64>::zeros(cov.nrows());
    let mut h_upper = Array1::<f64>::zeros(cov.nrows());
    let mut gamma = vec![0.0; p_resp];
    for i in 0..cov.nrows() {
        let cov_row = cov.row(i);
        for k in 0..p_resp {
            gamma[k] = beta_mat.row(k).dot(&cov_row);
        }
        let mut lower = family.response_lower_basis[0] * gamma[0]
            + family.offset[i]
            + family.response_lower_floor_offset;
        let mut upper = family.response_upper_basis[0] * gamma[0]
            + family.offset[i]
            + family.response_upper_floor_offset;
        for k in 1..p_resp {
            lower += family.response_lower_basis[k] * gamma[k] * gamma[k];
            upper += family.response_upper_basis[k] * gamma[k] * gamma[k];
        }
        h_lower[i] = lower;
        h_upper[i] = upper;
    }

    let mut expected_ll = 0.0;
    for i in 0..direct_h.len() {
        assert!(
            (row.h_lower[i] - h_lower[i]).abs() <= 1.0e-14,
            "h_lower[{i}] mismatch: cached={} direct={}",
            row.h_lower[i],
            h_lower[i]
        );
        assert!(
            (row.h_upper[i] - h_upper[i]).abs() <= 1.0e-14,
            "h_upper[{i}] mismatch: cached={} direct={}",
            row.h_upper[i],
            h_upper[i]
        );
        let hp = direct_h_prime[i];
        let log_z = log_normal_cdf_diff(h_upper[i], h_lower[i]).expect("endpoint mass");
        expected_ll += weights[i] * (-0.5 * direct_h[i] * direct_h[i] + hp.ln() - log_z);
    }

    assert!(
        (row.log_likelihood - expected_ll).abs() <= 1.0e-14,
        "cached log-likelihood={} expected={expected_ll}",
        row.log_likelihood
    );
}

#[test]
pub(crate) fn ctn_endpoint_normalizer_derivatives_are_finite_in_positive_tail() {
    let q = log_normal_cdf_diff_derivatives(38.0, 37.0).expect("positive-tail endpoint normalizer");
    assert!(q.first[0].is_finite());
    assert!(q.first[1].is_finite());
    assert!(q.second[0][0].is_finite());
    assert!(q.third[0][0][0].is_finite());
    assert!(q.fourth[0][0][0][0].is_finite());
    assert!(q.first[0] > 0.0);
    assert!(q.first[1] < 0.0);
}

#[test]
pub(crate) fn transformation_normal_pit_score_uses_finite_support_normalizer() {
    let center =
        transformation_normal_pit_score(0.0, -2.0, 2.0, 1.0e-12).expect("symmetric PIT score");
    assert!(center.abs() <= 1.0e-12);

    let positive_tail = transformation_normal_pit_score(37.5, 37.0, 38.0, 1.0e-12)
        .expect("positive-tail PIT score");
    assert!(positive_tail.is_finite());

    // Extrapolation past the upper endpoint is *not* an error: the PIT
    // mapping clamps `h` to `[lower, upper]` so `u → 1`, and the
    // `clip_eps` clamp on the standard-normal quantile call yields the
    // upper-tail extreme finite value (`≈ Φ⁻¹(1 - clip_eps)`). At
    // large-scale shape, an honest test sample at-or-just-beyond the
    // training response support routinely lands here from boundary
    // roundoff alone, so failing closed would ship a hard prediction
    // error on every CTN bootstrap pass.
    let above_upper = transformation_normal_pit_score(2.1, -2.0, 2.0, 1.0e-12)
        .expect("extrapolation above upper endpoint should clamp, not error");
    assert!(above_upper.is_finite());
    assert!(above_upper > 0.0, "h>upper must produce upper-tail PIT");
    let below_lower = transformation_normal_pit_score(-2.1, -2.0, 2.0, 1.0e-12)
        .expect("extrapolation below lower endpoint should clamp, not error");
    assert!(below_lower.is_finite());
    assert!(below_lower < 0.0, "h<lower must produce lower-tail PIT");

    // Genuinely-malformed input (NaN h) must still be rejected by the
    // early `is_finite()` guard — the soft-clamp is for legitimate
    // numerical extrapolation, not for non-finite values.
    let nan_err = transformation_normal_pit_score(f64::NAN, -2.0, 2.0, 1.0e-12)
        .expect_err("NaN h must still be rejected");
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
    let h_lower = array![-8.0];
    let h_upper = array![8.0];
    let weights = array![1.0];
    let err = build_transformation_row_derived(&h, &h_prime, &h_lower, &h_upper, &weights)
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
        crate::seeding::SeedRiskProfile::Gaussian
    );
    assert_eq!(seed_config.num_auxiliary_trailing, 0);
}

#[test]
pub(crate) fn max_feasible_step_size_is_unconstrained_for_scop_derivative() {
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
    // The SCOP squared-γ warm start is built directly in β-space: choose a
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
    let weights = array![0.7, 1.4, 0.9, 1.2];
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
    let got_gram = kron.weighted_gram(&weights, &ResourcePolicy::default_library());

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
        .exact_newton_joint_psisecond_order_terms(&states, &specs, &derivative_blocks, 0, 1)
        .expect("analytic psi second-order terms")
        .expect("psi second-order terms should be present");

    let eval_first = |psi_eval: &Array1<f64>| {
        let (mut f_eval, deriv_eval, state_eval, spec_eval) = toy_family_and_derivatives(psi_eval);
        f_eval.offset = Arc::clone(&row_offset);
        let states_eval = vec![state_eval];
        let specs_eval = vec![spec_eval];
        f_eval
            .exact_newton_joint_psi_terms(&states_eval, &specs_eval, &deriv_eval, 0)
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
        .exact_newton_joint_psi_terms(&states, &specs, &derivative_blocks, 0)
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
                &derivative_blocks,
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
                .exact_newton_joint_psi_terms(&states, &specs, &derivative_blocks, psi_index)
                .expect("per-axis ψ terms")
                .expect("per-axis ψ terms must be present"),
        );
    }

    // All-axes pass via the workspace.
    let workspace = family
        .exact_newton_joint_psi_workspace(&states, &specs, &derivative_blocks)
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

/// Direct kernel-level bit-equivalence guard for the fused all-axes
/// projected-trace path (Fix #2). Compares
/// [`scop_psi_hessian_trace_factor_all_axes_chunk_from_cov`] called once
/// with every ψ axis's `cov_psi` against
/// [`scop_psi_hessian_trace_factor_from_cov`] called once per axis. Both
/// kernels accumulate over the same rows in the same parallel rayon
/// reduction tree, so the fused output must equal the per-axis output to
/// well within any reasonable floating-point reduction tolerance.
#[test]
pub(crate) fn ctn_psi_hessian_trace_all_axes_matches_per_axis_path_bit_equivalent() {
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, state, _spec) = toy_family_and_derivatives(&psi);
    let n_psi = derivative_blocks[0].len();
    assert!(
        n_psi >= 2,
        "toy CTN fixture must expose at least 2 ψ axes for the fused trace check, got {n_psi}"
    );

    let row_quantities = family
        .row_quantities(&state.beta)
        .expect("toy CTN row quantities");

    // Build a non-trivial dense factor so every block of the ψ-Hessian
    // contributes to the projected trace. Three columns exercise both the
    // diagonal and off-diagonal Kronecker structure.
    let p_total = state.beta.len();
    let rank = 3;
    let mut factor = Array2::<f64>::zeros((p_total, rank));
    for col in 0..rank {
        let seed = 17_001_u64.wrapping_add(col as u64 * 13_337);
        factor
            .column_mut(col)
            .assign(&toy_probe_vector(p_total, seed));
    }

    // Materialise covariate and per-axis cov_psi over the full row range.
    let cov_arc = family
        .covariate_dense_arc()
        .expect("toy CTN covariate dense");
    let cov: &Array2<f64> = cov_arc.as_ref();
    let block_derivs = &derivative_blocks[0];
    let op_arc = block_derivs[0]
        .implicit_operator
        .as_ref()
        .expect("toy CTN ψ operator")
        .clone();
    let op = op_arc
        .as_any()
        .downcast_ref::<TensorKroneckerPsiOperator>()
        .expect("toy CTN ψ operator must be tensor-backed");
    let mut cov_psi_arrays: Vec<Array2<f64>> = Vec::with_capacity(n_psi);
    for deriv in block_derivs.iter() {
        cov_psi_arrays.push(
            op.materialize_cov_first_axis(deriv.implicit_axis)
                .expect("toy CTN ψ cov derivative materialise"),
        );
    }
    let cov_psi_views: Vec<ArrayView2<'_, f64>> = cov_psi_arrays.iter().map(|m| m.view()).collect();

    // Per-axis ground truth: invoke the legacy single-axis kernel n_psi
    // times.
    let per_axis_traces: Vec<f64> = (0..n_psi)
        .map(|axis_idx| {
            family
                .scop_psi_hessian_trace_factor_from_cov(
                    &state.beta,
                    &row_quantities,
                    block_derivs[axis_idx].implicit_axis,
                    cov,
                    &cov_psi_arrays[axis_idx],
                    factor.view(),
                )
                .expect("per-axis ψ projected trace")
        })
        .collect();

    // Fused all-axes pass: a single row-streaming traversal across every
    // axis. Calling the chunked kernel with `row_start=0` and the full-n
    // covariate views is equivalent to streaming a single chunk that
    // covers the entire dataset.
    let fused_traces = family
        .scop_psi_hessian_trace_factor_all_axes_chunk_from_cov(
            &state.beta,
            &row_quantities,
            0,
            cov.view(),
            &cov_psi_views,
            factor.view(),
        )
        .expect("fused all-axes ψ projected trace");

    assert_eq!(
        per_axis_traces.len(),
        fused_traces.len(),
        "per-axis vs fused trace length mismatch"
    );
    for (axis_idx, (&per_axis, &fused)) in
        per_axis_traces.iter().zip(fused_traces.iter()).enumerate()
    {
        let scale = per_axis.abs().max(fused.abs()).max(1.0);
        let abs_diff = (per_axis - fused).abs();
        let rel_diff = abs_diff / scale;
        assert!(
            rel_diff < 1.0e-12,
            "axis {axis_idx}: per-axis kernel = {per_axis:.6e}, fused kernel = {fused:.6e}, |Δ| = {abs_diff:.3e}, rel = {rel_diff:.3e}"
        );
    }
}

#[test]
pub(crate) fn ctn_psi_workspace_second_order_matches_per_pair_path() {
    let psi = array![0.15, -0.10];
    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let states = vec![state.clone()];
    let specs = vec![spec];
    let n_psi = derivative_blocks[0].len();

    let workspace = family
        .exact_newton_joint_psi_workspace(&states, &specs, &derivative_blocks)
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
                    &derivative_blocks,
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
        .exact_newton_joint_psisecond_order_terms(&states, &specs, &derivative_blocks, 0, 1)
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
            &derivative_blocks,
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
                &derivative_blocks,
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
        .exact_newton_joint_psi_workspace(&[state.clone()], &specs, &derivative_blocks)
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
    assert!(file!().ends_with(".rs"));
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
    assert_eq!(
        gradient,
        crate::solver::rho_optimizer::Derivative::Analytic
    );
    assert_eq!(
        hessian,
        crate::solver::rho_optimizer::DeclaredHessianForm::Either
    );

    let rho_dim = spec.initial_log_lambdas.len();
    let psi_dim = derivative_blocks[0].len();
    let outer_plan =
        crate::solver::rho_optimizer::plan(&crate::solver::rho_optimizer::OuterCapability {
            gradient,
            hessian,
            n_params: rho_dim + psi_dim,
            psi_dim,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        });
    assert_eq!(
        outer_plan.solver,
        crate::solver::rho_optimizer::Solver::Arc
    );
    assert_eq!(
        outer_plan.hessian_source,
        crate::solver::rho_optimizer::HessianSource::Analytic
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
        crate::solver::estimate::reml::unified::outer_hessian_route_plan(
            crate::solver::estimate::reml::unified::MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD,
            crate::solver::estimate::reml::unified::MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N,
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
    assert_eq!(
        gradient,
        crate::solver::rho_optimizer::Derivative::Analytic
    );
    assert_eq!(
        hessian,
        crate::solver::rho_optimizer::DeclaredHessianForm::Either
    );
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

/// Pairwise oracle for Phase-2 outer-HVP cross-checking.
///
/// Builds the toy CTN fixture (n=4, p_resp=2, p_cov=2, ψ_dim=2), calls the
/// existing pairwise body `exact_newton_joint_psisecond_order_terms(i, j)`
/// for every (i, j) pair, computes the directional contraction
/// `Σ_j v_j · pair(i, j)` for a fixed direction `v`, and writes the full
/// likelihood-only result to `/tmp/ctn_pairwise_oracle.json`.
///
/// Independent verification path for the SCOP CTN HVP work. The old Python
/// scripts used the pre-SCOP linear tensor likelihood and were removed so
/// they cannot be mistaken for ground truth.
///
/// Run via:
///     cargo test --release ctn_pairwise_oracle_dumps_json -- --nocapture
#[test]
pub(crate) fn ctn_pairwise_oracle_dumps_json() {
    let psi = array![0.15, -0.10];
    let v = array![0.4, -0.7];

    let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
    let block_states = vec![state.clone()];
    let specs = vec![spec.clone()];
    let beta = state.beta.clone();

    let psi_dim = psi.len();
    let p_total = beta.len();
    let n_obs = family.weights.as_ref().len();

    eprintln!("[oracle] toy CTN: n={n_obs}, p_resp=2, p_cov=2, p_total={p_total}, ψ_dim={psi_dim}");
    eprintln!("[oracle] β = {:?}", beta.as_slice().unwrap());
    eprintln!("[oracle] ψ = {:?}", psi.as_slice().unwrap());
    eprintln!("[oracle] v = {:?}", v.as_slice().unwrap());

    // The CTN psi-psi second-order kernel can return the dense block
    // either eagerly (`hessian_psi_psi`, p_total×p_total) or lazily
    // (`hessian_psi_psi_operator`, materialized via `to_dense`). The
    // oracle dump records the dense numbers, so materialize on demand.
    let dense_pair_hessian = |terms: &ExactNewtonJointPsiSecondOrderTerms| -> Array2<f64> {
        if terms.hessian_psi_psi.nrows() > 0 {
            terms.hessian_psi_psi.clone()
        } else {
            terms
                .hessian_psi_psi_operator
                .as_ref()
                .expect("CTN psi-psi must expose either dense Hessian or operator")
                .to_dense()
        }
    };

    // Per-pair pairwise body — likelihood pieces only (no penalty/logdet).
    let mut pair_records = Vec::new();
    for i in 0..psi_dim {
        for j in 0..psi_dim {
            let terms = family
                .exact_newton_joint_psisecond_order_terms(
                    &block_states,
                    &specs,
                    &derivative_blocks,
                    i,
                    j,
                )
                .expect("pairwise call ok")
                .expect("pairwise returns Some for valid i,j");
            let g_inf = terms
                .score_psi_psi
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            let b_dense = dense_pair_hessian(&terms);
            let b_inf = b_dense.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            eprintln!(
                "[oracle] pair (i={i}, j={j}): a={:.10}, ‖g‖∞={:.6e}, ‖b_mat‖∞={:.6e}",
                terms.objective_psi_psi, g_inf, b_inf,
            );
            pair_records.push(serde_json::json!({
                "i": i,
                "j": j,
                "a": terms.objective_psi_psi,
                "g": terms.score_psi_psi.to_vec(),
                "b_mat": b_dense.iter().copied().collect::<Vec<f64>>(),
                "b_mat_shape": [b_dense.nrows(), b_dense.ncols()],
            }));
        }
    }

    // Directional contraction: Σ_j v_j · pair(i, j) for each free axis i.
    let mut a_dir = Array1::<f64>::zeros(psi_dim);
    let mut g_dir = Array2::<f64>::zeros((psi_dim, p_total));
    let mut b_dir = vec![Array2::<f64>::zeros((p_total, p_total)); psi_dim];
    for i in 0..psi_dim {
        for j in 0..psi_dim {
            let terms = family
                .exact_newton_joint_psisecond_order_terms(
                    &block_states,
                    &specs,
                    &derivative_blocks,
                    i,
                    j,
                )
                .expect("pairwise call ok")
                .expect("pairwise returns Some for valid i,j");
            a_dir[i] += v[j] * terms.objective_psi_psi;
            let mut g_row = g_dir.slice_mut(s![i, ..]);
            g_row.scaled_add(v[j], &terms.score_psi_psi);
            b_dir[i].scaled_add(v[j], &dense_pair_hessian(&terms));
        }
    }

    eprintln!("[oracle] directional contraction Σ_j v_j · pair(i, j):");
    for i in 0..psi_dim {
        eprintln!(
            "[oracle]   i={i}: a_dir={:.10}, ‖g_dir‖∞={:.6e}, ‖b_dir‖∞={:.6e}",
            a_dir[i],
            g_dir.row(i).iter().fold(0.0f64, |m, x| m.max(x.abs())),
            b_dir[i].iter().fold(0.0f64, |m, x| m.max(x.abs())),
        );
    }

    let directional_records: Vec<_> = (0..psi_dim)
        .map(|i| {
            serde_json::json!({
                "i": i,
                "a_dir": a_dir[i],
                "g_dir": g_dir.row(i).to_vec(),
                "b_dir": b_dir[i].iter().copied().collect::<Vec<f64>>(),
                "b_dir_shape": [p_total, p_total],
            })
        })
        .collect();

    let blob = serde_json::json!({
        "config": {
            "n": n_obs,
            "p_resp": 2,
            "p_cov": 2,
            "p_total": p_total,
            "psi_dim": psi_dim,
            "beta": beta.to_vec(),
            "psi": psi.to_vec(),
            "v": v.to_vec(),
        },
        "pairwise": pair_records,
        "directional_contraction": directional_records,
        "note": "Likelihood-only pieces from exact_newton_joint_psisecond_order_terms. \
                 Penalty/logdet contributions are added by the unified evaluator's \
                 outer_hessian_entry. Cross-check this against sympy-shadow's symbolic \
                 derivation of the same likelihood quantities at the same toy config.",
    });

    let path = "/tmp/ctn_pairwise_oracle.json";
    std::fs::write(
        path,
        serde_json::to_string_pretty(&blob).expect("serialize ok"),
    )
    .expect("write ok");
    eprintln!("[oracle] wrote {path}");

    // Sanity assertions: nothing NaN, directional contraction is consistent
    // with element-wise summation.
    assert!(a_dir.iter().all(|x| x.is_finite()));
    assert!(g_dir.iter().all(|x| x.is_finite()));
    assert!(b_dir.iter().all(|m| m.iter().all(|x| x.is_finite())));
}
