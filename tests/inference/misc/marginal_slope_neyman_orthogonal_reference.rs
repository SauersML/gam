//! The optional influence kernel remains independently testable. Native CTN
//! prediction does not install it or claim Neyman orthogonality.
use gam::families::marginal_slope_orthogonal::{ScoreInfluenceJacobian, influence_block_design};
use ndarray::{Array1, Array2};

#[test]
fn influence_block_design_is_diag_scaled_jacobian() {
    let columns = Array2::from_shape_fn((6, 3), |(i, k)| (i as f64 + 1.0) * 0.1 - k as f64 * 0.07);
    let beta = Array1::from_shape_fn(6, |i| 0.3 + 0.2 * i as f64);
    let jac = ScoreInfluenceJacobian { columns: columns.clone(), z: Array1::zeros(6) };
    let block = influence_block_design(&jac, &beta, 1.7);
    assert_eq!(block.dim(), columns.dim());
    for i in 0..6 {
        for k in 0..3 {
            assert!((block[[i, k]] - 1.7 * beta[i] * columns[[i, k]]).abs() < 1e-12);
        }
    }
}
