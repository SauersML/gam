use gam::terms::basis::{
    BasisWorkspace, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, build_duchon_basis,
    build_duchon_basiswithworkspace,
};
use ndarray::Array2;

#[test]
fn workspace_variant_matches_nonworkspace_result() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    )
    .unwrap();
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: Some(1.3),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: Default::default(),
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: Default::default(),
    };
    let mut ws = BasisWorkspace::default();
    let a = build_duchon_basis(data.view(), &spec)
        .unwrap()
        .design
        .to_dense();
    let b = build_duchon_basiswithworkspace(data.view(), &spec, &mut ws)
        .unwrap()
        .design
        .to_dense();
    let err = (&a - &b)
        .mapv(f64::abs)
        .iter()
        .fold(0.0_f64, |x, &y| x.max(y));
    assert!(
        err < 1e-12,
        "workspace and non-workspace Duchon builders should produce identical design matrices; got max abs error {err}"
    );
}
