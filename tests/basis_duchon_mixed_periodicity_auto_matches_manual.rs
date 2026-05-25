use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, build_duchon_basis,
    build_duchon_basis_mixed_periodicity_auto,
};
use ndarray::Array2;

#[test]
fn mixed_periodicity_auto_dispatch_matches_manual_path() {
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.2, 0.4, 0.5, 0.8, 0.7, 1.1, 1.0, 1.4, 1.3, 1.8],
    )
    .unwrap();
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: Some(vec![Some(1.0), None]),
        length_scale: Some(1.1),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: Default::default(),
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: Default::default(),
    };
    let auto = build_duchon_basis_mixed_periodicity_auto(
        data.view(),
        &spec,
        &[true, false],
        Some(&[1.0, 1.0]),
    )
    .unwrap()
    .design
    .to_dense();
    let manual = build_duchon_basis(data.view(), &spec)
        .unwrap()
        .design
        .to_dense();
    let err = (&auto - &manual)
        .mapv(f64::abs)
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));
    assert!(
        err < 1e-10,
        "automatic mixed-periodicity dispatch should match the manual dispatch result exactly on overlapping inputs; got max abs error {err}"
    );
}
