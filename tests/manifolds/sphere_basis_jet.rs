//! Finite-difference verification of the analytic spherical-spline DESIGN jets
//! (`∂Φ/∂(lat, lon)`) for all three kernels — Sobolev, pseudo, and harmonic.
//!
//! For each kernel we build the forward design at `points ± h` (central
//! difference) in latitude then longitude and compare the resulting numeric
//! derivative against the analytic jet returned by
//! [`spherical_spline_design_jet`]. The jet must align column-for-column with
//! the forward design (same sum-to-zero transform for the Wahba kernels, same
//! real-spherical-harmonic column order for harmonic), so the comparison is a
//! direct entrywise relative-error check.

use gam::basis::{
    CenterStrategy, SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec,
    SphericalSplineIdentifiability, build_spherical_spline_basis, spherical_spline_design_jet,
};
use ndarray::{Array2, Array3, ArrayView2};

/// Small lat/lon evaluation set (degrees), spread away from the poles so the
/// associated-Legendre / kernel pole handling is exercised only mildly.
fn sample_points() -> Array2<f64> {
    ndarray::array![
        [10.0, 20.0],
        [-25.0, 70.0],
        [40.0, -110.0],
        [-5.0, 150.0],
        [55.0, 5.0],
    ]
}

/// Explicit Wahba centers (degrees) — fixed across point perturbations so the
/// sum-to-zero transform `z` is identical at `points` and `points ± h`.
fn sample_centers() -> Array2<f64> {
    ndarray::array![
        [0.0, 0.0],
        [30.0, 60.0],
        [-45.0, -30.0],
        [60.0, 120.0],
        [-20.0, -150.0],
    ]
}

fn forward_design(points: ArrayView2<'_, f64>, spec: &SphericalSplineBasisSpec) -> Array2<f64> {
    build_spherical_spline_basis(points, spec)
        .expect("forward sphere design")
        .design
        .to_dense()
}

/// Central-difference jet of the forward design: `(N, K, 2)` with the last
/// axis = (∂/∂lat, ∂/∂lon) in the same (degree) angular units as the input.
fn fd_jet(points: &Array2<f64>, spec: &SphericalSplineBasisSpec, h: f64) -> Array3<f64> {
    let base = forward_design(points.view(), spec);
    let (n, k) = (base.nrows(), base.ncols());
    let mut out = Array3::<f64>::zeros((n, k, 2));
    for axis in 0..2 {
        let mut plus = points.clone();
        let mut minus = points.clone();
        for i in 0..n {
            plus[(i, axis)] += h;
            minus[(i, axis)] -= h;
        }
        let dp = forward_design(plus.view(), spec);
        let dm = forward_design(minus.view(), spec);
        for i in 0..n {
            for j in 0..k {
                out[[i, j, axis]] = (dp[(i, j)] - dm[(i, j)]) / (2.0 * h);
            }
        }
    }
    out
}

fn assert_jet_matches_fd(spec: &SphericalSplineBasisSpec, label: &str) {
    let points = sample_points();
    let analytic = spherical_spline_design_jet(points.view(), spec).expect("analytic sphere jet");
    let fd = fd_jet(&points, spec, 1.0e-6);
    assert_eq!(
        analytic.shape(),
        fd.shape(),
        "{label}: jet shape {:?} != fd shape {:?}",
        analytic.shape(),
        fd.shape()
    );
    let mut max_rel = 0.0_f64;
    for (a, f) in analytic.iter().zip(fd.iter()) {
        let denom = f.abs().max(a.abs()).max(1.0e-6);
        let rel = (a - f).abs() / denom;
        max_rel = max_rel.max(rel);
        assert!(
            a.is_finite() && f.is_finite(),
            "{label}: non-finite jet entry analytic={a} fd={f}"
        );
    }
    assert!(
        max_rel < 1.0e-4,
        "{label}: analytic jet disagrees with central-difference (max rel err {max_rel:.3e})"
    );
}

fn wahba_spec(kernel: SphereWahbaKernel, penalty_order: usize) -> SphericalSplineBasisSpec {
    SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::UserProvided(sample_centers()),
        penalty_order,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Wahba,
        max_degree: None,
        wahba_kernel: kernel,
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
    }
}

fn harmonic_spec(max_degree: usize, penalty_order: usize) -> SphericalSplineBasisSpec {
    SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 0 },
        penalty_order,
        double_penalty: false,
        radians: false,
        method: SphereMethod::Harmonic,
        max_degree: Some(max_degree),
        wahba_kernel: SphereWahbaKernel::Sobolev,
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
    }
}

#[test]
fn sobolev_jet_matches_finite_difference_all_orders() {
    for m in [1usize, 2, 3] {
        assert_jet_matches_fd(
            &wahba_spec(SphereWahbaKernel::Sobolev, m),
            &format!("sobolev m={m}"),
        );
    }
}

#[test]
fn pseudo_jet_matches_finite_difference_all_orders() {
    for m in [1usize, 2, 3] {
        assert_jet_matches_fd(
            &wahba_spec(SphereWahbaKernel::Pseudo, m),
            &format!("pseudo m={m}"),
        );
    }
}

#[test]
fn harmonic_jet_matches_finite_difference_multiple_degrees() {
    for l in [2usize, 3] {
        assert_jet_matches_fd(&harmonic_spec(l, 2), &format!("harmonic L={l}"));
    }
}
