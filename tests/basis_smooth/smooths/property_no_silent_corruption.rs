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

use gam::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu, build_matern_basis};
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

/// Build a standard isotropic Matérn spec over a center subset of the cloud.
fn matern_spec() -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
        periodic: None,
        length_scale: gam::terms::basis::MaternLengthScale::fixed(0.5),
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
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
        !built.active_penalties.is_empty(),
        "INVARIANT VIOLATED [matern emits at least one penalty]: a penalized smooth with no \
         penalty block silently drops its smoothing constraint."
    );
    for (k, active) in built.active_penalties.iter().enumerate() {
        assert_all_finite(active.matrix.view(), &format!("matern penalty block #{k}"));
    }
}

