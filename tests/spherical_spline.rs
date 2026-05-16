use gam::basis::{
    CenterStrategy, SphericalSplineBasisSpec, build_spherical_spline_basis,
    spherical_wahba_kernel_matrix,
};
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::{ParsedTerm, parse_formula};
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::terms::basis::BasisMetadata;
use gam::terms::smooth::SmoothBasisSpec;
use gam::terms::term_builder::build_termspec;
use ndarray::array;

#[test]
fn wahba_kernel_is_longitude_periodic_and_symmetric() {
    let points = array![[0.0, 0.0], [25.0, 179.0], [-60.0, -45.0]];
    let shifted = array![[0.0, 360.0], [25.0, -181.0], [-60.0, 315.0]];

    let k = spherical_wahba_kernel_matrix(points.view(), points.view(), 2, false).expect("kernel");
    let k_shifted =
        spherical_wahba_kernel_matrix(points.view(), shifted.view(), 2, false).expect("kernel");

    for i in 0..k.nrows() {
        for j in 0..k.ncols() {
            assert!(
                (k[(i, j)] - k[(j, i)]).abs() < 1e-12,
                "kernel not symmetric"
            );
            assert!(
                (k[(i, j)] - k_shifted[(i, j)]).abs() < 1e-12,
                "kernel should be invariant to 360-degree longitude shifts"
            );
        }
    }
}

#[test]
fn spherical_basis_builds_constrained_design_and_penalties() {
    let data = array![
        [-80.0, -170.0],
        [-40.0, -60.0],
        [0.0, 0.0],
        [35.0, 80.0],
        [70.0, 160.0]
    ];
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        penalty_order: 2,
        double_penalty: true,

        radians: false,
    
        method: gam::basis::SphereMethod::Wahba,
        max_degree: None,};

    let built = build_spherical_spline_basis(data.view(), &spec).expect("sphere basis");
    assert_eq!(built.design.nrows(), data.nrows());
    assert_eq!(built.design.ncols(), data.nrows() - 1);
    assert_eq!(built.penalties.len(), 2);
    assert_eq!(built.penalties[0].nrows(), data.nrows() - 1);

    match built.metadata {
        BasisMetadata::Sphere {
            centers,
            penalty_order,
            constraint_transform,
        } => {
            assert_eq!(centers, data);
            assert_eq!(penalty_order, 2);
            let z = constraint_transform.expect("coefficient constraint transform");
            for col in 0..z.ncols() {
                let sum = z.column(col).sum();
                assert!(sum.abs() < 1e-12, "constraint column {col} sum {sum}");
            }
        }
        other => panic!("unexpected metadata: {other:?}"),
    }
}

#[test]
fn sphere_formula_and_mgcv_sos_alias_resolve_to_sphere_basis() {
    let parsed = parse_formula("y ~ sphere(lat, lon, k=4) + s(lat, lon, bs=\"sos\", k=4)")
        .expect("formula parses");
    assert_eq!(parsed.terms.len(), 2);
    assert!(
        matches!(parsed.terms[0], ParsedTerm::Smooth { ref vars, .. } if vars == &vec!["lat".to_string(), "lon".to_string()])
    );

    let values = array![
        [1.0, -80.0, -170.0],
        [2.0, -30.0, -60.0],
        [3.0, 0.0, 0.0],
        [4.0, 30.0, 60.0],
        [5.0, 80.0, 170.0]
    ];
    let ds = EncodedDataset {
        headers: vec!["y".into(), "lat".into(), "lon".into()],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "lat".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "lon".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous; 3],
    };
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam::ResourcePolicy::default_library(),
    )
    .expect("term spec");
    assert_eq!(spec.smooth_terms.len(), 2);
    assert!(matches!(
        spec.smooth_terms[0].basis,
        SmoothBasisSpec::Sphere { .. }
    ));
    assert!(matches!(
        spec.smooth_terms[1].basis,
        SmoothBasisSpec::Sphere { .. }
    ));
}

#[test]
fn wahba_kernel_radians_matches_degrees() {
    let deg = array![[10.0, 25.0], [-30.0, -60.0], [45.0, 170.0]];
    let to_rad = std::f64::consts::PI / 180.0;
    let mut rad = deg.clone();
    rad.mapv_inplace(|v| v * to_rad);
    let k_deg =
        spherical_wahba_kernel_matrix(deg.view(), deg.view(), 2, false).expect("kernel deg");
    let k_rad = spherical_wahba_kernel_matrix(rad.view(), rad.view(), 2, true).expect("kernel rad");
    for i in 0..k_deg.nrows() {
        for j in 0..k_deg.ncols() {
            assert!(
                (k_deg[(i, j)] - k_rad[(i, j)]).abs() < 1e-12,
                "kernel(deg) != kernel(rad) at ({i},{j}): {} vs {}",
                k_deg[(i, j)],
                k_rad[(i, j)]
            );
        }
    }
}

#[test]
fn wahba_kernel_invariant_under_rigid_rotation_of_sphere() {
    // rotate all points by 30 deg longitude; intrinsic kernel must be unchanged
    let pts = array![[0.0, 10.0], [25.0, 60.0], [-45.0, -120.0]];
    let mut rot = pts.clone();
    for r in 0..rot.nrows() {
        rot[(r, 1)] = ((rot[(r, 1)] + 30.0_f64 + 180.0_f64).rem_euclid(360.0_f64)) - 180.0_f64;
    }
    let k = spherical_wahba_kernel_matrix(pts.view(), pts.view(), 2, false).expect("k");
    let k_rot = spherical_wahba_kernel_matrix(rot.view(), rot.view(), 2, false).expect("k_rot");
    for i in 0..k.nrows() {
        for j in 0..k.ncols() {
            assert!(
                (k[(i, j)] - k_rot[(i, j)]).abs() < 1e-12,
                "kernel not rotation-invariant at ({i},{j}): {} vs {}",
                k[(i, j)],
                k_rot[(i, j)]
            );
        }
    }
}

#[test]
fn spherical_harmonic_basis_builds_with_correct_width_and_diagonal_penalty() {
    use gam::basis::SphereMethod;
    let data = array![
        [-80.0, -170.0],
        [-40.0, -60.0],
        [0.0, 0.0],
        [35.0, 80.0],
        [70.0, 160.0]
    ];
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 2,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(3),
    };
    let built = build_spherical_spline_basis(data.view(), &spec).expect("sphere harmonic basis");
    // dim = L(L+2) = 3*5 = 15
    assert_eq!(built.design.ncols(), 15);
    assert_eq!(built.penalties.len(), 1);
    let p = built.penalties[0].clone();
    // diagonal
    for i in 0..p.nrows() {
        for j in 0..p.ncols() {
            if i != j {
                assert!(p[(i, j)].abs() < 1e-12, "off-diag penalty entry {i},{j}");
            }
        }
    }
}

#[test]
fn spherical_harmonic_basis_rotation_invariant_gram_under_longitude_shift() {
    use gam::basis::SphereMethod;
    let data = array![
        [10.0, 20.0],
        [-30.0, -40.0],
        [50.0, 110.0],
        [-60.0, -100.0],
        [25.0_f64, 80.0_f64],
    ];
    let mut rotated = data.clone();
    for r in 0..rotated.nrows() {
        let lon = rotated[(r, 1)] + 47.0_f64;
        rotated[(r, 1)] = ((lon + 180.0_f64).rem_euclid(360.0_f64)) - 180.0_f64;
    }
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 2,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(3),
    };
    let a = build_spherical_spline_basis(data.view(), &spec).expect("base");
    let b = build_spherical_spline_basis(rotated.view(), &spec).expect("rotated");
    // XᵀX is invariant under the block-orthogonal (sin(mφ),cos(mφ)) rotation
    // induced by a pure longitude shift.
    let da = a.design.to_dense();
    let db = b.design.to_dense();
    let ga = da.t().dot(&da);
    let gb = db.t().dot(&db);
    for i in 0..ga.nrows() {
        for j in 0..ga.ncols() {
            assert!(
                (ga[(i, j)] - gb[(i, j)]).abs() < 1e-10,
                "Gram entry ({i},{j}) not rotation-invariant: {} vs {}",
                ga[(i, j)],
                gb[(i, j)]
            );
        }
    }
}

#[test]
fn spherical_basis_rejects_bad_latitudes_and_wrong_dimension() {
    let bad_lat = array![[91.0, 0.0], [0.0, 10.0]];
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::UserProvided(bad_lat.clone()),
        penalty_order: 2,
        double_penalty: false,

        radians: false,
    
        method: gam::basis::SphereMethod::Wahba,
        max_degree: None,};
    let err = build_spherical_spline_basis(bad_lat.view(), &spec).expect_err("invalid latitude");
    assert!(err.to_string().contains("latitude must be in [-90, 90]"));

    let wrong_dim = array![[0.0, 0.0, 1.0], [10.0, 20.0, 1.0]];
    let err = spherical_wahba_kernel_matrix(wrong_dim.view(), wrong_dim.view(), 2, false)
        .expect_err("wrong dimension");
    assert!(err.to_string().contains("exactly two columns"));
}
