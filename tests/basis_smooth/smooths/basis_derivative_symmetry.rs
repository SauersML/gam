use gam::terms::basis::{BasisMetadata, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, SpatialIdentifiability, build_duchon_basis};
use ndarray::Array2;

#[test]
fn log_kappa_derivative_matrix_is_symmetric() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.3, 0.1, 0.6, 0.7, 1.1, 0.9, 1.4, 1.2],
    )
    .unwrap();
    let mut spec = DuchonBasisSpec {
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
    // #2638: the log-kappa derivative is a FROZEN-chart derivative. The forward
    // `build_duchon_basis` ADOPTS a data-metric radial reparam `V` for a spec that
    // carries none (#1355), so replay the cold build's `V` onto the spec before
    // asking for the derivative -- exactly what the kappa-optimizer does. Without
    // it the derivative would be taken in a chart the forward build never ships.
    let cold = build_duchon_basis(data.view(), &spec).expect("cold Duchon build");
    let BasisMetadata::Duchon { radial_reparam, .. } = &cold.metadata else {
        panic!("expected Duchon metadata from the cold build");
    };
    spec.radial_reparam = radial_reparam.clone();
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
