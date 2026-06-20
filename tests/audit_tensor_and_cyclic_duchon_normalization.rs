//! Audit (companion to `audit_penalty_normalization_scale_equivariance.rs`):
//! certify the penalty-normalization classification beyond the 1-D B-spline
//! paths.
//!
//! The shipped design penalty is `β'(S/c)β` and the REML smoothing parameter
//! `λ` multiplies that block, so every builder emitting a roughness/difference
//! operator must Frobenius-normalize it (`‖S‖_F ≈ 1`) — see the root-cause class
//! behind #1364 / #1365 / #1366. This file pins two construction-time facts that
//! the sibling test (1-D B-spline) does not reach:
//!
//!   * POSITIVE control — the cyclic (periodic) Duchon roughness penalty is now
//!     Frobenius-normalized (the #1366 fix in
//!     `build_cyclic_duchon_basis_1dwithworkspace`).
//!   * NEGATIVE control — the *harmonic* sphere penalty is INTENTIONALLY left
//!     un-normalized (`normalization_scale = 1.0`): its diagonal entries are the
//!     physical Laplace-Beltrami roughness eigenvalues `[l(l+1)]^m`, which are a
//!     meaningful spectral scale that must NOT be divided away. Asserting
//!     `‖S‖_F ≫ 1` here documents the by-design boundary so the harmonic-sphere
//!     penalty is not mistakenly "fixed" as a leak.

use gam::basis::{
    build_duchon_basiswithworkspace, build_spherical_spline_basis, BasisWorkspace, CenterStrategy,
    DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, OneDimensionalBoundary,
    SpatialIdentifiability, SphereMethod, SphericalSplineBasisSpec,
};
use ndarray::Array2;

fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// POSITIVE control: the cyclic/periodic 1-D Duchon roughness penalty ships
/// Frobenius-normalized after #1366 (was a raw operator with
/// `normalization_scale = 1.0`).
#[test]
fn cyclic_duchon_penalty_is_frobenius_normalized() {
    // One periodic covariate on [0, 1) with user-provided centers, mirroring the
    // periodic-Duchon construction the build path already exercises.
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 0.2, 0.5, 0.8, 1.0]).unwrap();
    let centers =
        Array2::from_shape_vec((6, 1), (0..6).map(|i| i as f64 / 6.0).collect()).unwrap();
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        periodic: None,
        length_scale: Some(0.25),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Zero,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Cyclic { start: 0.0, end: 1.0 },
    };
    let mut workspace = BasisWorkspace::default();
    let built = build_duchon_basiswithworkspace(x.view(), &spec, &mut workspace)
        .expect("cyclic Duchon basis build");
    assert_eq!(
        built.penalties.len(),
        1,
        "cyclic Duchon ships exactly one (roughness) penalty block"
    );
    let n = frob(&built.penalties[0]);
    assert!(
        (n - 1.0).abs() < 1e-9,
        "cyclic Duchon penalty is NOT Frobenius-normalized: ‖S‖_F = {n:.6e} (expected 1.0). \
         The cyclic roughness operator was shipped un-normalized (#1366), putting λ on a \
         basis-dependent scale and mis-calibrating REML's λ-search."
    );
}

/// NEGATIVE control: the harmonic sphere penalty is INTENTIONALLY raw — its
/// diagonal entries are the physical Laplace-Beltrami roughness eigenvalues, a
/// meaningful spectral scale, so `‖S‖_F` is far from 1. This documents the
/// by-design boundary; normalizing it would be a regression, not a fix.
#[test]
fn harmonic_sphere_penalty_is_intentionally_not_frobenius_normalized() {
    // Spread points over the sphere (lat/lon degrees); harmonic method, L = 3.
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![
            -80.0, -170.0, -40.0, -60.0, 0.0, 0.0, 35.0, 80.0, 70.0, 160.0,
        ],
    )
    .unwrap();
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
    assert_eq!(built.penalties.len(), 1, "harmonic sphere ships one penalty");
    let n = frob(&built.penalties[0]);
    // The largest physical eigenvalue at L=3, m=2 is [3*4]^2 = 144, so ‖S‖_F is
    // on the order of 100+. The point is that it is intentionally NOT ~1: the
    // orthonormal-S² basis makes these the physical roughness eigenvalues, which
    // must not be Frobenius-divided away.
    assert!(
        n > 10.0,
        "harmonic sphere penalty unexpectedly looks Frobenius-normalized: ‖S‖_F = {n:.6e}. \
         This basis is CORRECT-BY-DESIGN to ship the raw physical Laplace-Beltrami eigenvalue \
         penalty (un-normalized); a value near 1.0 would mean someone normalized it, which is a \
         regression — the physical spectral scale carries the smoothing semantics here."
    );
}
