use gam::basis::{ActivePenalty, BasisBuildResult, CenterStrategy, PenaltySource, SphericalSplineBasisSpec, build_spherical_spline_basis};
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::{ParsedTerm, parse_formula};
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::terms::basis::BasisMetadata;
use gam::terms::smooth::SmoothBasisSpec;
use gam::terms::term_builder::build_termspec;
use ndarray::{array, s};

fn primary_penalty(built: &BasisBuildResult) -> &ActivePenalty {
    built
        .active_penalties
        .iter()
        .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
        .expect("basis must retain its primary penalty")
}

#[test]
fn spherical_basis_builds_raw_wahba_design_and_penalties() {
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
        max_degree: None,
        wahba_kernel: Default::default(),
        identifiability: Default::default(),
    };

    let built = build_spherical_spline_basis(data.view(), &spec).expect("sphere basis");
    assert_eq!(built.design.nrows(), data.nrows());
    assert_eq!(built.design.ncols(), data.nrows());
    assert_eq!(built.active_penalties.len(), 2);
    assert_eq!(primary_penalty(&built).matrix.nrows(), data.nrows());
    let nullspace_shrinkage = built
        .active_penalties
        .iter()
        .find(|penalty| matches!(penalty.info.source, PenaltySource::DoublePenaltyNullspace))
        .expect("double_penalty=true must retain its null-space shrinkage penalty");
    assert_eq!(nullspace_shrinkage.matrix.nrows(), data.nrows());

    match built.metadata {
        BasisMetadata::Sphere {
            centers,
            penalty_order,
            method,
            max_degree,
            wahba_kernel,
            constraint_transform,
        } => {
            assert_eq!(centers, data);
            assert_eq!(penalty_order, 2);
            assert_eq!(method, gam::basis::SphereMethod::Wahba);
            assert_eq!(max_degree, None);
            assert_eq!(wahba_kernel, Default::default());
            let z = constraint_transform.expect("raw coefficient chart transform");
            assert_eq!(z.nrows(), centers.nrows());
            assert_eq!(z.ncols(), centers.nrows());
            for r in 0..z.nrows() {
                for c in 0..z.ncols() {
                    let expected = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (z[(r, c)] - expected).abs() < 1e-12,
                        "raw Wahba chart should keep identity transform; z[{r},{c}]={}",
                        z[(r, c)]
                    );
                }
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
fn sphere_m4_wahba_formula_enforces_stable_center_floor_only_for_m4() {
    let parsed = parse_formula(
        "y ~ sphere(lat, lon, k=25, m=4, kernel=pseudo) + sphere(lat, lon, k=25, m=2, kernel=pseudo)",
    )
    .expect("formula parses");
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
    let SmoothBasisSpec::Sphere { spec: m4_spec, .. } = &spec.smooth_terms[0].basis else {
        panic!("expected m=4 sphere basis");
    };
    let CenterStrategy::FarthestPoint {
        num_centers: m4_centers,
    } = &m4_spec.center_strategy
    else {
        panic!("expected m=4 farthest-point centers");
    };
    assert_eq!(
        *m4_centers, 30,
        "m=4 Wahba needs the stable center floor that fixed the k=25 seed regression"
    );

    let SmoothBasisSpec::Sphere { spec: m2_spec, .. } = &spec.smooth_terms[1].basis else {
        panic!("expected m=2 sphere basis");
    };
    let CenterStrategy::FarthestPoint {
        num_centers: m2_centers,
    } = &m2_spec.center_strategy
    else {
        panic!("expected m=2 farthest-point centers");
    };
    assert_eq!(
        *m2_centers, 25,
        "the m=4 stability floor must not change ordinary Wahba k semantics"
    );
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
        wahba_kernel: Default::default(),
        identifiability: Default::default(),
    };
    let built = build_spherical_spline_basis(data.view(), &spec).expect("sphere harmonic basis");
    // dim = L(L+2) = 3*5 = 15
    assert_eq!(built.design.ncols(), 15);
    assert_eq!(built.active_penalties.len(), 1);
    let p = &primary_penalty(&built).matrix;
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
fn spherical_harmonic_penalty_order_changes_penalty_shape() {
    use gam::basis::SphereMethod;
    let data = array![
        [-80.0, -170.0],
        [-40.0, -60.0],
        [0.0, 0.0],
        [35.0, 80.0],
        [70.0, 160.0]
    ];
    let mut spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 1,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(3),
        wahba_kernel: Default::default(),
        identifiability: Default::default(),
    };
    let built_m1 = build_spherical_spline_basis(data.view(), &spec).expect("m=1 harmonic basis");
    let p1 = &primary_penalty(&built_m1).matrix;
    spec.penalty_order = 4;
    let built_m4 = build_spherical_spline_basis(data.view(), &spec).expect("m=4 harmonic basis");
    let p4 = &primary_penalty(&built_m4).matrix;

    let low_degree_ratio_m1 = p1[(3, 3)] / p1[(0, 0)];
    let low_degree_ratio_m4 = p4[(3, 3)] / p4[(0, 0)];
    assert!(
        low_degree_ratio_m4 > 10.0 * low_degree_ratio_m1,
        "harmonic penalty_order should steepen high-degree shrinkage: m1 ratio={low_degree_ratio_m1}, m4 ratio={low_degree_ratio_m4}"
    );
}

#[test]
fn spherical_harmonic_rejects_invalid_penalty_order() {
    use gam::basis::SphereMethod;
    let data = array![[0.0, 0.0], [10.0, 20.0]];
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 0,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(2),
        wahba_kernel: Default::default(),
        identifiability: Default::default(),
    };
    let err = build_spherical_spline_basis(data.view(), &spec)
        .expect_err("invalid harmonic penalty order");
    assert!(err.to_string().contains("penalty_order"));
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
        wahba_kernel: Default::default(),
        identifiability: Default::default(),
    };
    let a = build_spherical_spline_basis(data.view(), &spec).expect("base");
    let b = build_spherical_spline_basis(rotated.view(), &spec).expect("rotated");
    // Under a pure longitude shift the harmonic design columns rotate inside
    // each (l, m) block by a 2x2 orthogonal block, so D_rotated = D · R for
    // some block-orthogonal R. The COLUMN SPAN is invariant; equivalently
    // the eigenvalue spectrum of D D^T is invariant (the row Gram).
    let da = a.design.to_dense();
    let db = b.design.to_dense();
    let row_gram_a = da.dot(&da.t());
    let row_gram_b = db.dot(&db.t());
    for i in 0..row_gram_a.nrows() {
        for j in 0..row_gram_a.ncols() {
            assert!(
                (row_gram_a[(i, j)] - row_gram_b[(i, j)]).abs() < 1e-10,
                "row Gram entry ({i},{j}) not rotation-invariant: {} vs {}",
                row_gram_a[(i, j)],
                row_gram_b[(i, j)]
            );
        }
    }
}

#[test]
fn spherical_harmonic_basis_accepts_non_contiguous_views() {
    use gam::basis::SphereMethod;
    let padded = array![
        [-80.0, -170.0, 1.0],
        [-40.0, -60.0, 2.0],
        [0.0, 0.0, 3.0],
        [35.0, 80.0, 4.0],
        [70.0, 160.0, 5.0],
        [15.0, -20.0, 6.0],
    ];
    let data = padded.slice(s![..;2, 0..2]);
    assert!(!data.is_standard_layout());
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order: 2,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(2),
        wahba_kernel: Default::default(),
        identifiability: Default::default(),
    };
    let built = build_spherical_spline_basis(data, &spec)
        .expect("harmonic basis should not require contiguous lat/lon rows");
    assert_eq!(built.design.nrows(), 3);
    assert_eq!(built.design.ncols(), 8);
}

