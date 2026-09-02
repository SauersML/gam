use gam::families::bms::{MarginalSlopeCovarianceShape, marginal_slope_covariance_from_scores};
use ndarray::{Array1, array};

// The K=1 reduction is an identity over the reals, not over IEEE-754.
// `marginal_slope_preserving_scale` squares the probit scale once and applies
// it to the quadratic form of the *raw* slopes (the diagonal quadratic form
// accumulates `coefficient * slope * slope`), so production evaluates
// `fl(fl(p*p) * fl(s*s))` while the scalar identity below folds the scale into
// the slope first and evaluates `fl(fl(p*s) * fl(p*s))`. Same real number, up
// to an ulp apart, so `to_bits()` equality would pin the association order
// rather than the reduction. It is pinned to a few-ulp bound instead.
//
// `magnitude` is the size of the largest intermediates the reference sums, so
// a cancelling total is still held to the accuracy its inputs allow.

#[test]
fn multi_z_covariance_shape_auto_derives_from_score_geometry() {
    let weights = Array1::ones(6);

    let diagonal_scores = array![
        [-1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -3.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
    ];
    let diagonal = marginal_slope_covariance_from_scores(diagonal_scores.view(), &weights)
        .expect("diagonal covariance");
    assert_eq!(diagonal.shape(), MarginalSlopeCovarianceShape::Diagonal);

    // Every nonzero off-diagonal is exact geometry and therefore remains Full.
    const N_FULL: usize = 64;
    let mut full_scores = ndarray::Array2::<f64>::zeros((N_FULL, 2));
    for i in 0..N_FULL {
        let t = (i as f64) / (N_FULL as f64);
        let h3 = (2.0 * std::f64::consts::PI * 3.0 * t).cos();
        let h5 = (2.0 * std::f64::consts::PI * 5.0 * t).cos();
        full_scores[[i, 0]] = h3;
        full_scores[[i, 1]] = 0.7 * h3 + 0.4 * h5;
    }
    let weights_full = Array1::ones(N_FULL);
    let full =
        marginal_slope_covariance_from_scores(full_scores.view(), &weights_full).expect("full cov");
    assert_eq!(full.shape(), MarginalSlopeCovarianceShape::Full);

    let low_rank_scores = array![
        [-2.0, -4.0, 1.0, 3.0],
        [-1.0, -2.0, 0.5, 1.5],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, -0.5, -1.5],
        [2.0, 4.0, -1.0, -3.0],
        [3.0, 6.0, -1.5, -4.5],
    ];
    let collinear = marginal_slope_covariance_from_scores(low_rank_scores.view(), &weights)
        .expect("collinear covariance");
    assert_eq!(collinear.shape(), MarginalSlopeCovarianceShape::Full);
}
