//! Compare both spatial axes to rebuilt values in a fixed coefficient chart.
use gam_terms::basis::{
    BasisMetadata, BasisWorkspace, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, FixedRowSpaceProjector, OneDimensionalBoundary,
    SpatialIdentifiability, build_duchon_basis,
    build_duchon_basis_log_kappa_aniso_derivativeswith_collocationwithworkspace,
};
use ndarray::{Array1, Array2};

fn relative_gap(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    let scale = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(scale > 0.0, "the numerical derivative must be nonzero");
    (a - b).iter().map(|v| v * v).sum::<f64>().sqrt() / scale
}

#[test]
fn raw_axis_design_derivatives_match_the_frozen_forward_basis() {
    assert_raw_axis_design_derivatives(1.0, 0.25);
}

#[test]
fn amplified_raw_axis_design_derivatives_match_the_frozen_forward_basis() {
    assert_raw_axis_design_derivatives(1e-6, 0.25);
}

#[test]
fn isotropic_raw_axis_design_derivatives_match_the_frozen_forward_basis() {
    assert_raw_axis_design_derivatives(1.0, 0.0);
}

fn assert_raw_axis_design_derivatives(length_scale: f64, contrast: f64) {
    let data = Array2::from_shape_fn((80, 2), |(i, j)| {
        if j == 0 {
            i as f64 / 79.0
        } else {
            (i as f64 * 0.618_033_988_749_894_9).fract()
        }
    });
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(length_scale),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: Some(vec![contrast, -contrast]),
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let base = build_duchon_basis(data.view(), &spec).expect("base basis");
    let BasisMetadata::Duchon {
        centers,
        identifiability_transform,
        radial_reparam,
        aniso_log_scales,
        operator_collocation_points,
        ..
    } = &base.metadata
    else {
        panic!("Duchon metadata");
    };
    assert_eq!(aniso_log_scales.as_deref(), Some([contrast, -contrast].as_slice()),
        "explicit anisotropy, including zero, must be honored literally");
    spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
    spec.radial_reparam = radial_reparam.clone();
    spec.aniso_log_scales = aniso_log_scales.clone();
    spec.identifiability = match identifiability_transform {
        Some(transform) => SpatialIdentifiability::FrozenTransform {
            transform: transform.clone(),
        },
        None => SpatialIdentifiability::None,
    };
    let derivatives = build_duchon_basis_log_kappa_aniso_derivativeswith_collocationwithworkspace(
        data.view(),
        &spec,
        centers.view(),
        identifiability_transform.as_ref(),
        operator_collocation_points
            .as_ref()
            .map(|points| points.view()),
        &mut BasisWorkspace::default(),
    )
    .expect("per-axis derivatives");
    let operator = derivatives
        .implicit_operator
        .expect("Duchon derivative operator");
    let realize = |steps: [f64; 2]| {
        let mut shifted = spec.clone();
        let mean = steps.iter().sum::<f64>() / 2.0;
        shifted.length_scale =
            Some(spec.length_scale.expect("hybrid length scale") * (-mean).exp());
        for (j, value) in shifted
            .aniso_log_scales
            .as_mut()
            .expect("anisotropy")
            .iter_mut()
            .enumerate()
        {
            *value += steps[j] - mean;
        }
        build_duchon_basis(data.view(), &shifted)
            .expect("shifted frozen basis")
            .design
            .to_dense()
    };
    let mut worst = 0.0_f64;
    let mut analytic_axes = Vec::new();
    let mut numerical_axes = Vec::new();
    for axis in 0..2 {
        let analytic = operator
            .materialize_first(axis)
            .expect("analytic design derivative");
        let shift = |h| {
            let mut steps = [0.0; 2];
            steps[axis] = h;
            realize(steps)
        };
        let coarse = (shift(2e-3) - shift(-2e-3)) / 4e-3;
        let fine = (shift(1e-3) - shift(-1e-3)) / 2e-3;
        let numerical = (4.0 * fine - coarse) / 3.0;
        let gap = relative_gap(&analytic, &numerical);
        let mut centered_analytic = analytic.clone();
        let mut centered_numerical = numerical.clone();
        for matrix in [&mut centered_analytic, &mut centered_numerical] {
            for mut column in matrix.columns_mut() {
                let mean = column.sum() / column.len() as f64;
                column -= mean;
            }
        }
        let centered_gap = relative_gap(&centered_analytic, &centered_numerical);
        eprintln!(
            "axis={axis} design_derivative_relative_gap={gap:.8e} centered_gap={centered_gap:.8e}"
        );
        worst = worst.max(gap);
        analytic_axes.push(analytic);
        numerical_axes.push(numerical);

        let u = Array1::from_shape_fn(analytic_axes[axis].ncols(), |j| (j as f64 + 0.3).sin());
        let v = Array1::from_shape_fn(data.nrows(), |j| (j as f64 + 0.7).cos());
        let forward = operator
            .forward_mul(axis, &u.view())
            .expect("first forward action");
        let transpose = operator
            .transpose_mul(axis, &v.view())
            .expect("first transpose action");
        assert!(
            (&forward - &analytic_axes[axis].dot(&u))
                .iter()
                .all(|value| value.abs() < 1e-10)
        );
        assert!(
            (&transpose - &analytic_axes[axis].t().dot(&v))
                .iter()
                .all(|value| value.abs() < 1e-10)
        );

        let numerical_second = |h| (shift(h) + shift(-h) - 2.0 * realize([0.0; 2])) / (h * h);
        let second = operator
            .materialize_second_diag(axis)
            .expect("second design derivative");
        let numerical = (4.0 * numerical_second(1e-3) - numerical_second(2e-3)) / 3.0;
        let gap = relative_gap(&second, &numerical);
        eprintln!("axis={axis} second_derivative_relative_gap={gap:.8e}");
        worst = worst.max(gap);
        let forward = operator
            .forward_mul_second_diag(axis, &u.view())
            .expect("second forward action");
        let transpose = operator
            .transpose_mul_second_diag(axis, &v.view())
            .expect("second transpose action");
        assert!(
            (&forward - &second.dot(&u))
                .iter()
                .all(|value| value.abs() < 1e-10)
        );
        assert!(
            (&transpose - &second.t().dot(&v))
                .iter()
                .all(|value| value.abs() < 1e-10)
        );
    }
    let numerical_cross = |h| {
        (realize([h, h]) - realize([h, -h]) - realize([-h, h]) + realize([-h, -h])) / (4.0 * h * h)
    };
    let cross = operator
        .materialize_second_cross(0, 1)
        .expect("mixed design derivative");
    let numerical = (4.0 * numerical_cross(1e-3) - numerical_cross(2e-3)) / 3.0;
    let gap = relative_gap(&cross, &numerical);
    eprintln!("cross_derivative_relative_gap={gap:.8e}");
    worst = worst.max(gap);
    let constraint =
        Array2::from_shape_fn(
            (data.nrows(), 2),
            |(i, j)| {
                if j == 0 { 1.0 } else { data[[i, 0]] }
            },
        );
    let projector =
        FixedRowSpaceProjector::from_constraint_block(constraint.view()).expect("row projector");
    let projected = operator
        .clone()
        .with_fixed_row_space_projection(projector.clone())
        .expect("projected operator");
    let u = Array1::from_shape_fn(cross.ncols(), |j| (j as f64 + 0.3).sin());
    let v = Array1::from_shape_fn(data.nrows(), |j| (j as f64 + 0.7).cos());
    for axis in 0..2 {
        let mut expected = analytic_axes[axis].clone();
        projector
            .project_matrix_in_place(&mut expected)
            .expect("project first");
        let actual = projected.materialize_first(axis).expect("projected first");
        assert!(relative_gap(&actual, &expected) < 1e-10);
        let mut row = Array1::zeros(actual.ncols());
        projected
            .row_vector_first_into(axis, 3, row.view_mut())
            .expect("projected single row");
        assert!(
            (&row - &actual.row(3))
                .iter()
                .all(|value| value.abs() < 1e-10)
        );
        let forward = projected
            .forward_mul(axis, &u.view())
            .expect("projected forward");
        let transpose = projected
            .transpose_mul(axis, &v.view())
            .expect("projected transpose");
        assert!(
            (&forward - &actual.dot(&u))
                .iter()
                .all(|value| value.abs() < 1e-10)
        );
        assert!(
            (&transpose - &actual.t().dot(&v))
                .iter()
                .all(|value| value.abs() < 1e-10)
        );
        let mut second = operator.materialize_second_diag(axis).expect("second");
        projector
            .project_matrix_in_place(&mut second)
            .expect("project second");
        assert!(
            relative_gap(
                &projected
                    .materialize_second_diag(axis)
                    .expect("projected second"),
                &second
            ) < 1e-10
        );
    }
    let mut expected_cross = cross.clone();
    projector
        .project_matrix_in_place(&mut expected_cross)
        .expect("project cross");
    let actual = projected
        .materialize_second_cross(0, 1)
        .expect("projected cross");
    assert!(relative_gap(&actual, &expected_cross) < 1e-10);
    let forward = projected
        .forward_mul_second_cross(0, 1, &u.view())
        .expect("cross forward");
    let transpose = projected
        .transpose_mul_second_cross(0, 1, &v.view())
        .expect("cross transpose");
    assert!(
        (&forward - &actual.dot(&u))
            .iter()
            .all(|value| value.abs() < 1e-10)
    );
    assert!(
        (&transpose - &actual.t().dot(&v))
            .iter()
            .all(|value| value.abs() < 1e-10)
    );
    eprintln!(
        "contrast_gap={:.8e} global_gap={:.8e}",
        relative_gap(
            &(&analytic_axes[0] - &analytic_axes[1]),
            &(&numerical_axes[0] - &numerical_axes[1])
        ),
        relative_gap(
            &(&analytic_axes[0] + &analytic_axes[1]),
            &(&numerical_axes[0] + &numerical_axes[1])
        )
    );
    assert!(
        worst < 1e-4,
        "analytic per-axis design derivative mismatch: {worst:.8e}"
    );
}
