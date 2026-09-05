//! #2818 recovery of the non-Euclidean basis gauges specified by #2315.

use gam_terms::basis::{
    BasisBuildResult, CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
    build_constant_curvature_basis, build_spherical_spline_basis,
    constant_curvature_kernel_psi_jets,
};
use gam_terms::smooth::build_pca_smooth_basis;
use ndarray::{Array2, array};

fn assert_matrix_scaled(actual: &Array2<f64>, expected: &Array2<f64>, scale: f64, tolerance: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, column), &want) in expected.indexed_iter() {
        let got = actual[[row, column]] / scale;
        assert!(
            (got - want).abs() <= tolerance * (1.0 + want.abs()),
            "entry({row},{column}) actual/scale={got:.16e}, expected={want:.16e}, scale={scale:.6e}"
        );
    }
}

fn assert_geometry_scaled(
    actual: &BasisBuildResult,
    expected: &BasisBuildResult,
    scale: f64,
    tolerance: f64,
) {
    assert_matrix_scaled(
        &actual.design.to_dense(),
        &expected.design.to_dense(),
        scale,
        tolerance,
    );
    assert_eq!(
        actual.active_penalties.len(),
        expected.active_penalties.len()
    );
    assert!(!expected.active_penalties.is_empty());
    for (observed, reference) in actual
        .active_penalties
        .iter()
        .zip(expected.active_penalties.iter())
    {
        assert_eq!(observed.info.source, reference.info.source);
        assert_eq!(observed.info.effective_rank, reference.info.effective_rank);
        assert_eq!(observed.nullity, reference.nullity);
        assert_matrix_scaled(&observed.matrix, &reference.matrix, scale, tolerance);
    }
}

fn spherical_spec(method: SphereMethod, radians: bool) -> SphericalSplineBasisSpec {
    SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
        penalty_order: 2,
        double_penalty: false,
        radians,
        method,
        max_degree: Some(3),
        wahba_kernel: SphereWahbaKernel::Sobolev,
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
    }
}

#[test]
fn sphere_constant_curvature_and_pca_obey_their_non_euclidean_gauges_2315() {
    let degrees = array![
        [-62.0_f64, -150.0],
        [-41.0, -77.0],
        [-18.0, -12.0],
        [4.0, 39.0],
        [23.0, 101.0],
        [47.0, 166.0],
        [66.0, -115.0],
        [11.0, -171.0]
    ];
    let radians = degrees.mapv(f64::to_radians);
    for method in [SphereMethod::Wahba, SphereMethod::Harmonic] {
        let reference =
            build_spherical_spline_basis(degrees.view(), &spherical_spec(method, false)).unwrap();
        let actual =
            build_spherical_spline_basis(radians.view(), &spherical_spec(method, true)).unwrap();
        assert_geometry_scaled(&actual, &reference, 1.0, 2e-9);
    }

    let chart_data = array![
        [-0.42, -0.18],
        [-0.31, 0.22],
        [-0.08, -0.34],
        [0.13, 0.29],
        [0.27, -0.11],
        [0.38, 0.17]
    ];
    let centers = array![[-0.36, -0.04], [-0.12, 0.25], [0.16, -0.21], [0.34, 0.13]];
    let kappa = -0.7_f64;
    let length_scale = 0.55_f64;
    let reference_spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        kappa,
        kappa_fixed: true,
        length_scale,
        length_scale_fixed: true,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let reference = build_constant_curvature_basis(chart_data.view(), &reference_spec).unwrap();
    let reference_jets =
        constant_curvature_kernel_psi_jets(chart_data.view(), centers.view(), kappa, length_scale)
            .unwrap();
    // An all-zero derivative implementation would satisfy scale equivariance
    // vacuously, so both curvature derivative orders need resolved witnesses.
    let first_curvature_witness = reference_jets
        .d_kappa
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let second_curvature_witness = reference_jets
        .d_kappa2
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    assert!(first_curvature_witness > 1e-8);
    assert!(second_curvature_witness > 1e-8);
    for factor in [1e-9_f64, 1.0, 1e9] {
        let scaled_data = chart_data.mapv(|value| factor * value);
        let scaled_centers = centers.mapv(|value| factor * value);
        let scaled_kappa = kappa / factor.powi(2);
        let scaled_length = length_scale * factor;
        let actual = build_constant_curvature_basis(
            scaled_data.view(),
            &ConstantCurvatureBasisSpec {
                center_strategy: CenterStrategy::UserProvided(scaled_centers.clone()),
                kappa: scaled_kappa,
                length_scale: scaled_length,
                ..reference_spec.clone()
            },
        )
        .unwrap();
        // K=l*(exp(-d/l)-1) carries one length power; each kappa derivative
        // adds two because kappa rescales with inverse squared chart length.
        assert_geometry_scaled(&actual, &reference, factor, 2e-8);
        let jets = constant_curvature_kernel_psi_jets(
            scaled_data.view(),
            scaled_centers.view(),
            scaled_kappa,
            scaled_length,
        )
        .unwrap();
        assert_matrix_scaled(&jets.value, &reference_jets.value, factor, 2e-8);
        assert_matrix_scaled(&jets.d_kappa, &reference_jets.d_kappa, factor.powi(3), 2e-8);
        assert_matrix_scaled(
            &jets.d_kappa2,
            &reference_jets.d_kappa2,
            factor.powi(5),
            3e-7,
        );
    }

    let data = array![
        [-1.0, 0.3],
        [-0.4, 1.1],
        [0.2, -0.7],
        [0.8, 0.5],
        [1.3, -0.2],
        [1.7, 0.9]
    ];
    let mean = array![0.35, 0.15];
    let loadings = array![[0.8, -0.3], [0.6, 0.9]];
    let reference = build_pca_smooth_basis(
        data.view(),
        &[0, 1],
        &loadings,
        true,
        1.7,
        Some(&mean),
        None,
        32,
    )
    .unwrap();
    for factor in [1e-9_f64, 1.0, 1e9] {
        let actual = build_pca_smooth_basis(
            data.mapv(|value| factor * value).view(),
            &[0, 1],
            &loadings.mapv(|value| value / factor),
            true,
            1.7,
            Some(&mean.mapv(|value| factor * value)),
            None,
            32,
        )
        .unwrap();
        assert_geometry_scaled(&actual, &reference, 1.0, 2e-10);
    }
    eprintln!(
        "#2315 gauge contracts: 2 spherical methods, 3 curvature scales with degrees1/3/5, 3 PCA inverse-loading scales; first_curvature_witness={first_curvature_witness:.6e} second_curvature_witness={second_curvature_witness:.6e}"
    );
}
