//! PROPERTY / INVARIANT tests: penalty and derivative builders must emit
//! all-FINITE output on ordinary small inputs.
//!
//! These tests are the structural complement to the source-text scanners in
//! `build.rs`. A comment can lie and a banned keyword can be reworded around,
//! but a property test that drives the REAL code and asserts a numerical
//! invariant cannot be reworded away: if a penalty / derivative builder ever
//! launders a NaN or an Inf into its output (the "Pattern-1" silent-corruption
//! shortcut), then the finiteness assertion FAILS regardless of how any
//! surrounding comment is phrased.
//!
//! Each property below constructs a representative, low-setup input through the
//! crate's PUBLIC basis API and asserts that the emitted basis / penalty /
//! derivative matrices contain no NaN and no Inf. The invariant is wording-
//! independent: it is a fact about the numbers the code returns.

use gam::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu, build_matern_basis,
    build_matern_basis_log_kappa_derivative, create_thin_plate_spline_basis,
};
use ndarray::{Array2, ArrayView2};

/// Scan a dense matrix and return the first non-finite (NaN or Inf) entry's
/// (row, col, value), or `None` if every entry is finite.
fn first_non_finite(m: ArrayView2<'_, f64>) -> Option<(usize, usize, f64)> {
    for ((r, c), &v) in m.indexed_iter() {
        if !v.is_finite() {
            return Some((r, c, v));
        }
    }
    None
}

/// Assert a matrix is entirely finite, naming the invariant and flagging a
/// non-finite entry as the Pattern-1 silent-corruption signature.
fn assert_all_finite(m: ArrayView2<'_, f64>, what: &str) {
    if let Some((r, c, v)) = first_non_finite(m) {
        panic!(
            "INVARIANT VIOLATED [{what} is all-finite]: entry ({r},{c}) = {v} is NOT finite. \
             A NaN/Inf in a penalty or derivative matrix is silent numerical corruption \
             (the Pattern-1 NaN/zero laundering): downstream REML/IRLS will propagate it \
             into the fit instead of erroring. This invariant fails on BEHAVIOR, so no \
             rewording of any comment near the emission site can hide it."
        );
    }
}

/// A small, well-conditioned 2-D point cloud inside the unit square. Plain
/// finite coordinates: any non-finite OUTPUT is therefore produced by the
/// builder, not inherited from the input.
fn small_cloud_2d() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 2),
        vec![
            0.12, 0.84, 0.55, 0.21, 0.91, 0.47, 0.08, 0.33, 0.62, 0.78, 0.40, 0.05, 0.27, 0.66,
            0.73, 0.19, 0.36, 0.92, 0.84, 0.51,
        ],
    )
    .expect("10x2 cloud")
}

/// PROPERTY 1: the public thin-plate / Duchon penalty constructor returns an
/// all-finite design AND all-finite bending/ridge penalty matrices (and the
/// Wood radial reparameterization) on a normal small 2-D input.
///
/// The thin-plate penalty assembly does eigen-reparameterization and side-
/// constraint projection on a radial Gram — exactly the kind of linear-algebra
/// path where a divide-by-(near)-zero or a `sqrt` of a tiny negative eigenvalue
/// can launder a NaN into the penalty. A correct build is finite; this test
/// fails the moment it is not.
#[test]
fn thin_plate_penalty_is_all_finite() {
    let data = small_cloud_2d();
    // Knots = the data cloud (a standard, valid choice; >= polynomial null-space
    // dimension for d=2, which is 3).
    let knots = data.clone();

    let tps = create_thin_plate_spline_basis(data.view(), knots.view())
        .expect("thin-plate basis builds on a normal 2-D cloud");

    assert_all_finite(tps.basis.view(), "thin-plate design basis");
    assert_all_finite(tps.penalty_bending.view(), "thin-plate bending penalty");
    assert_all_finite(tps.penalty_ridge.view(), "thin-plate ridge penalty");
    assert_all_finite(
        tps.radial_reparam.view(),
        "thin-plate Wood radial reparameterization",
    );
}

/// Build a standard isotropic Matérn spec over a center subset of the cloud.
fn matern_spec() -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
        periodic: None,
        length_scale: 0.5,
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    }
}

/// PROPERTY 2: the public Matérn basis builder returns an all-finite design and
/// all-finite penalty matrices on a normal small 2-D input.
///
/// The Matérn kernel evaluates closed-form half-integer forms with `exp` and
/// polynomial factors of the scaled distance, then applies an identifiability
/// transform and penalty assembly. A non-finite output here would be silent
/// corruption of the model matrix.
#[test]
fn matern_basis_and_penalty_are_all_finite() {
    let data = small_cloud_2d();
    let spec = matern_spec();

    let built = build_matern_basis(data.view(), &spec).expect("matern basis builds on 2-D cloud");

    let dense = built
        .design
        .try_to_dense_by_chunks("matern design finiteness check")
        .expect("matern design densifies");
    assert_all_finite(dense.view(), "matern design matrix");

    assert!(
        !built.penalties.is_empty(),
        "INVARIANT VIOLATED [matern emits at least one penalty]: a penalized smooth with no \
         penalty block silently drops its smoothing constraint."
    );
    for (k, pen) in built.penalties.iter().enumerate() {
        assert_all_finite(pen.view(), &format!("matern penalty block #{k}"));
    }
}

/// PROPERTY 3 (DERIVATIVE PATH): the public Matérn log-κ derivative builder —
/// the analytic ∂(design,penalties)/∂log κ used by the outer REML gradient —
/// returns all-finite derivative matrices on a normal small 2-D input.
///
/// Derivative assembly is a prime Pattern-1 target: it differentiates the
/// kernel and the penalty Gram, multiplying by reciprocals of eigenvalues /
/// distances that can underflow. If any of those slots launders a NaN, the
/// outer gradient is silently poisoned and λ-selection drifts with no error.
/// A correct analytic derivative is finite everywhere; this asserts exactly
/// that, by behavior.
#[test]
fn matern_log_kappa_derivative_is_all_finite() {
    let data = small_cloud_2d();
    let spec = matern_spec();

    let dpsi = build_matern_basis_log_kappa_derivative(data.view(), &spec)
        .expect("matern log-kappa derivative builds on 2-D cloud");

    assert_all_finite(
        dpsi.design_derivative.view(),
        "matern d(design)/d(log kappa)",
    );
    for (k, pen) in dpsi.penalties_derivative.iter().enumerate() {
        assert_all_finite(pen.view(), &format!("matern d(penalty #{k})/d(log kappa)"));
    }
}
