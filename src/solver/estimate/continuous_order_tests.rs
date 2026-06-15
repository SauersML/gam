use super::*;

fn try_compute_continuous_smoothness_order(
    lambda_tilde: &[f64],
    normalization_scale: &[f64],
    eps: f64,
) -> Option<ContinuousSmoothnessOrder> {
    if lambda_tilde.len() != 3 || normalization_scale.len() != 3 {
        return None;
    }
    Some(compute_continuous_smoothness_order(
        [lambda_tilde[0], lambda_tilde[1], lambda_tilde[2]],
        [
            normalization_scale[0],
            normalization_scale[1],
            normalization_scale[2],
        ],
        eps,
    ))
}

#[test]
fn continuous_order_formula_matches_closed_form() {
    let out = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
    assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
    let r = out.r_ratio.expect("R");
    let nu = out.nu.expect("nu");
    let kappa2 = out.kappa2.expect("kappa2");
    assert!((r - (100.0 / 6.0)).abs() < 1e-12);
    assert!((nu - (r / (r - 2.0))).abs() < 1e-12);
    assert!((kappa2 - (10.0 / ((r - 2.0) * 3.0))).abs() < 1e-12);
}

#[test]
fn continuous_order_unscales_lambdas_exactly_by_ck() {
    let out = compute_continuous_smoothness_order([6.0, 15.0, 9.0], [3.0, 5.0, 9.0], 1e-12);
    // Physical lambdas must satisfy lambda_k = lambda_tilde_k / c_k.
    assert!((out.lambda0 - 2.0).abs() < 1e-12);
    assert!((out.lambda1 - 3.0).abs() < 1e-12);
    assert!((out.lambda2 - 1.0).abs() < 1e-12);
}

#[test]
fn continuous_order_invalid_ck_is_guarded() {
    let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 0.0, 1.0], 1e-12);
    assert_eq!(
        out.status,
        ContinuousSmoothnessOrderStatus::UndefinedZeroLambda
    );
    assert!(out.r_ratio.is_none());
}

#[test]
fn continuous_order_is_invariant_to_penalty_normalization_reversal() {
    let base = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
    let scaled = compute_continuous_smoothness_order(
        [2.0 * 4.0, 10.0 * 0.5, 3.0 * 8.0],
        [4.0, 0.5, 8.0],
        1e-12,
    );
    assert_eq!(base.status, ContinuousSmoothnessOrderStatus::Ok);
    assert_eq!(scaled.status, ContinuousSmoothnessOrderStatus::Ok);
    assert!((base.r_ratio.unwrap() - scaled.r_ratio.unwrap()).abs() < 1e-12);
    assert!((base.nu.unwrap() - scaled.nu.unwrap()).abs() < 1e-12);
    assert!((base.kappa2.unwrap() - scaled.kappa2.unwrap()).abs() < 1e-12);
}

#[test]
fn continuous_order_flags_non_matern_regimewhen_r_le_4() {
    let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
    assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
    assert!(out.nu.is_none());
    assert!(out.kappa2.is_none());
}

#[test]
fn continuous_order_reports_effective_nu_kappa_in_non_matern_bandwhen_r_gt_2() {
    let out = compute_continuous_smoothness_order([1.0, 3.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
    assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
    let r = out.r_ratio.expect("R");
    assert!(r > 2.0 && r < 4.0);
    assert!(out.nu.is_some());
    assert!(out.kappa2.is_some());
}

#[test]
fn continuous_order_boundary_r_equals_four_is_matern_square_case() {
    let out = compute_continuous_smoothness_order([1.0, 2.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
    assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
    let nu = out.nu.expect("nu");
    assert!((nu - 2.0).abs() < 1e-12);
}

#[test]
fn continuous_order_guardszero_or_nearzero_lambda() {
    let out = compute_continuous_smoothness_order([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
    assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
    assert!(out.r_ratio.is_none());
}

#[test]
fn continuous_order_first_order_limitwhen_lambda2_collapses() {
    let out = compute_continuous_smoothness_order([2.0, 4.0, 1e-20], [1.0, 1.0, 1.0], 1e-12);
    assert_eq!(out.status, ContinuousSmoothnessOrderStatus::FirstOrderLimit);
    assert_eq!(out.nu, Some(1.0));
    let k2 = out.kappa2.expect("kappa2");
    assert!((k2 - 0.5).abs() < 1e-12);
}

#[test]
fn continuous_order_intrinsic_limitwhen_lambda0_collapses() {
    let out = compute_continuous_smoothness_order([1e-20, 4.0, 2.0], [1.0, 1.0, 1.0], 1e-12);
    assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
    assert_eq!(out.nu, Some(1.0));
    assert_eq!(out.kappa2, Some(0.0));
}

#[test]
fn continuous_order_is_only_defined_for_three_penalties_per_term() {
    let ok = try_compute_continuous_smoothness_order(&[2.0, 10.0, 3.0], &[1.0, 1.0, 1.0], 1e-12);
    let two = try_compute_continuous_smoothness_order(&[2.0, 10.0], &[1.0, 1.0], 1e-12);
    let four = try_compute_continuous_smoothness_order(
        &[2.0, 10.0, 3.0, 7.0],
        &[1.0, 1.0, 1.0, 1.0],
        1e-12,
    );
    assert!(ok.is_some());
    assert!(two.is_none());
    assert!(four.is_none());
}
