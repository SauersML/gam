//! PROPERTY / INVARIANT test: a real penalty-construction path must emit
//! all-FINITE output (no NaN, no Inf) on an ordinary small input.
//!
//! This is the BEHAVIOR-level complement to the source-text scanners in
//! `build.rs`. A comment can be reworded and a banned keyword can be spelled
//! around, but a property test that drives the REAL penalty builder and asserts
//! a numerical invariant cannot be reworded away: if the cyclic-Duchon penalty
//! assembly ever launders a NaN or an Inf into its roughness operator (the
//! "Pattern-1" silent NaN/zero-corruption shortcut), the finiteness assertion
//! below FAILS regardless of how any surrounding comment is phrased.
//!
//! The construction here is copied verbatim from the already-compiling
//! `cyclic_duchon_penalty_is_frobenius_normalized` test in
//! `audit_tensor_and_cyclic_duchon_normalization.rs` (same imports, same public
//! `gam::basis` API, same types). Only the assertion changes: instead of a
//! Frobenius-norm check, this asserts every emitted penalty entry is finite.

use gam::basis::{
    BasisWorkspace, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, OneDimensionalBoundary, SpatialIdentifiability,
    build_duchon_basiswithworkspace,
};
use ndarray::Array2;

/// PROPERTY: the public cyclic (periodic) Duchon roughness-penalty constructor
/// returns an all-finite penalty matrix on a normal small periodic input.
///
/// The cyclic-Duchon penalty assembly builds and reparameterizes a radial Gram
/// over a wrapped coordinate — exactly the kind of linear-algebra path where a
/// divide-by-(near)-zero or a `sqrt` of a tiny negative eigenvalue can launder a
/// NaN/Inf into the roughness operator. A correct build is finite; this test
/// fails the moment it is not, on BEHAVIOR, independent of comment wording.
#[test]
fn cyclic_duchon_penalty_output_is_all_finite() {
    // One periodic covariate on [0, 1) with user-provided centers, mirroring the
    // periodic-Duchon construction the build path already exercises.
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 0.2, 0.5, 0.8, 1.0]).unwrap();
    let centers = Array2::from_shape_vec((6, 1), (0..6).map(|i| i as f64 / 6.0).collect()).unwrap();
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
        boundary: OneDimensionalBoundary::Cyclic {
            start: 0.0,
            end: 1.0,
        },
    };
    let mut workspace = BasisWorkspace::default();
    let built = build_duchon_basiswithworkspace(x.view(), &spec, &mut workspace)
        .expect("cyclic Duchon basis build");

    assert!(
        !built.penalties.is_empty(),
        "INVARIANT VIOLATED [cyclic Duchon emits a penalty]: a penalized smooth with no \
         penalty block silently drops its smoothing constraint."
    );
    for (k, pen) in built.penalties.iter().enumerate() {
        if let Some((r, c)) = pen
            .indexed_iter()
            .find(|(_, v)| !v.is_finite())
            .map(|((r, c), _)| (r, c))
        {
            let v = pen[(r, c)];
            panic!(
                "INVARIANT VIOLATED [cyclic Duchon penalty #{k} is all-finite]: entry \
                 ({r},{c}) = {v} is NOT finite. A NaN/Inf in a penalty matrix is silent \
                 numerical corruption (the Pattern-1 NaN/zero laundering): downstream \
                 REML/IRLS will propagate it into the fit instead of erroring. This \
                 invariant fails on BEHAVIOR, so no rewording of any comment near the \
                 emission site can hide it."
            );
        }
    }
}
