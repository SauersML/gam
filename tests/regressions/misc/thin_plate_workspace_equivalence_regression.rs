use gam::terms::basis::{
    BasisWorkspace, CenterStrategy, SpatialIdentifiability, ThinPlateBasisSpec,
    build_thin_plate_basis, build_thin_plate_basiswithworkspace,
};
use ndarray::Array2;

#[test]
fn bug_thin_plate_workspace_variant_mismatch() {
    let data = Array2::from_shape_vec(
        (8, 2),
        vec![
            -1.0, -1.0, -0.5, 0.25, 0.0, 0.0, 0.2, -0.3, 0.75, 0.5, 1.0, -0.25, 1.5, 1.0, -1.2, 0.9,
        ],
    )
    .expect("shape");

    let spec = ThinPlateBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
        periodic: None,
        length_scale: 1.0,
        identifiability: SpatialIdentifiability::OrthogonalToParametric,
        double_penalty: true,
        radial_reparam: None,
    };

    let no_ws = build_thin_plate_basis(data.view(), &spec).expect("thin-plate build without ws");
    let mut ws = BasisWorkspace::default();
    let with_ws = build_thin_plate_basiswithworkspace(data.view(), &spec, &mut ws)
        .expect("thin-plate build with ws");

    let a = no_ws.design.to_dense();
    let b = with_ws.design.to_dense();
    assert_eq!(a.dim(), b.dim(), "design shape mismatch");
    assert_eq!(
        no_ws.active_penalties.len(),
        with_ws.active_penalties.len(),
        "penalty count mismatch"
    );

    let max_design_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_design_diff <= 1e-12,
        "bug thin-plate workspace mismatch: max |Δdesign|={max_design_diff:e}"
    );
}
