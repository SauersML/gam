use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, SpatialIdentifiability,
    build_duchon_basis_log_kappa_derivative,
};
use ndarray::Array2;

#[test]
fn log_kappa_derivative_matrix_is_symmetric() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.3, 0.1, 0.6, 0.7, 1.1, 0.9, 1.4, 1.2],
    )
    .unwrap();
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: Some(1.2),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: Default::default(),
    };
    let m = build_duchon_basis_log_kappa_derivative(data.view(), &spec)
        .unwrap()
        .design_derivative;
    let skew = (&m - &m.t())
        .mapv(f64::abs)
        .iter()
        .fold(0.0_f64, |a: f64, &b| a.max(b));
    assert!(
        skew < 1e-10,
        "log-kappa derivative matrix should be symmetric because the underlying kernel is symmetric; got max skew {skew}"
    );
}
