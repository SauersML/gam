use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, build_duchon_basis,
};
use ndarray::Array2;

#[test]
fn anisotropic_build_handles_dominant_axis_without_instability() {
    let data = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.0, 0.0, 0.0, 0.4, 0.2, 0.1, 0.8, 0.5, 0.2, 1.2, 0.7, 0.3, 1.6, 1.0, 0.5,
        ],
    )
    .unwrap();
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: Default::default(),
        aniso_log_scales: Some(vec![6.0, -3.0, -3.0]),
        operator_penalties: Default::default(),
        boundary: Default::default(),
    };
    let b = build_duchon_basis(data.view(), &spec)
        .unwrap()
        .design
        .to_dense();
    assert!(
        b.iter().all(|v| v.is_finite()),
        "anisotropic build should remain finite when one axis dominates strongly"
    );
}
