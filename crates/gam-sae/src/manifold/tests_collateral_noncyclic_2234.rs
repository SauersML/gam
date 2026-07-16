//! Non-cyclic steering (gam#2234) — the collateral-damage thesis must NOT be
//! circle-specific.
//!
//! The E2 fixture ([`super::tests_collateral_e2_2234`]) proves the on-manifold
//! group action steers with less off-target collateral than a fixed flat
//! direction, but on two PERIODIC circle atoms. The steering primitive and the
//! [`crate::inference::steering::collateral_curve`] readout are axis-general — a
//! chart's periods are per-axis `Option`s and a non-periodic axis simply carries
//! `None`, so no code is circle-specific — but the only landed dominance fixture
//! was cyclic. Per the issue's explicit direction ("do not overfit to cyclic"),
//! this establishes the SAME structural dominance on a genuinely NON-CYCLIC
//! manifold: a curved parabolic arc on the unbounded `Euclidean` line.
//!
//! The atom decodes a degree-2 `EuclideanPatch` coordinate `t` over `[−1, 1]`
//! into an ambient parabolic arc: its degree-1 basis function drives one ambient
//! column and its degree-2 function another, so the image is a genuinely curved
//! arc whose decode tangent ROTATES along the axis exactly as a circle's does —
//! but the coordinate never wraps (no period, no `sin/cos`, no closed loop). A
//! fixed flat decoder direction is off that rotating tangent by an O(1) angle at
//! most rows (near the arc's vertex it is orthogonal to it), while the
//! on-manifold chord tracks it, leaving only its O(δ²) sagitta off-frame — so the
//! flat arm deposits strictly more energy outside the atom's own local
//! decode-tangent frame. The #2234 thesis, decided structurally, with no LLM and
//! no outer-fit convergence, on a manifold with no cyclic symmetry.

use super::*;
use crate::inference::steering::collateral_curve;
use ndarray::Array2;
use std::sync::Arc;

/// A degree-2 `EuclideanPatch` atom whose coordinate `t` decodes into a curved
/// ambient arc: the degree-1 basis function drives column `col_lin` and the
/// degree-2 function drives `col_quad` of `R^p`. Non-periodic (an unbounded
/// `Euclidean` line axis) yet curved — the decode tangent rotates with `t`.
fn parabola_atom(
    name: &str,
    p: usize,
    col_lin: usize,
    col_quad: usize,
    coords: &Array2<f64>,
) -> SaeManifoldAtom {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    // A degree-2 patch basis has three columns: the constant, degree-1, and
    // degree-2 functions of `t` (index 1 is linear, index 2 is quadratic).
    let m = phi.ncols();
    let mut decoder = Array2::<f64>::zeros((m, p));
    decoder[[1, col_lin]] = 1.0; // degree-1 basis → linear ambient column
    decoder[[2, col_quad]] = 1.0; // degree-2 basis → quadratic ambient column
    SaeManifoldAtom::new_with_provided_function_gram(
        name,
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(evaluator)
}

#[test]
fn zz_noncyclic_collateral_on_manifold_beats_flat_at_matched_norm() {
    let n = 240usize;
    let p = 8usize;

    // Two independent parabolic arcs on disjoint ambient planes (atom 0 in
    // span{e0,e1}, atom 1 in span{e2,e3}) over a non-periodic coordinate range
    // that straddles t = 0, where the tangent rotation — and the flat direction's
    // misalignment — is largest. Distinct (affinely reparameterized) schedules
    // keep the two coordinate fields from coinciding.
    let coords0 = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        -1.0 + 2.0 * (row as f64) / (n as f64 - 1.0)
    });
    let coords1 = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        0.8 * (-1.0 + 2.0 * (row as f64) / (n as f64 - 1.0)) + 0.1
    });
    let atom0 = parabola_atom("target-parabola", p, 0, 1, &coords0);
    let atom1 = parabola_atom("other-parabola", p, 2, 3, &coords1);

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 2)),
        vec![coords0.clone(), coords1.clone()],
        // Non-cyclic charts: the Euclidean line has NO period on its axis.
        vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

    let doses = [0.02_f64, 0.05, 0.1, 0.15, 0.2];
    let curve = collateral_curve(&term, 0, 0, &[1usize], &doses)
        .expect("collateral curve must build on the hand-fitted non-cyclic term");

    assert_eq!(curve.manifold.points.len(), doses.len());
    assert_eq!(curve.flat.points.len(), doses.len());

    // (1) The on-manifold arm is a genuine control knob on a non-cyclic axis: it
    //     turns the target feature at every nonzero dose.
    for pt in &curve.manifold.points {
        assert!(
            pt.on_target_effect > 0.0,
            "on-manifold dose {} produced no on-target effect",
            pt.dose
        );
    }

    // (2) Cross-feature leakage onto the geometrically-independent parabola is ≈ 0
    //     in BOTH arms — the disjoint ambient planes make locality exact up to
    //     floating point, cyclic symmetry or not.
    for (m, f) in curve.manifold.points.iter().zip(curve.flat.points.iter()) {
        assert!(
            m.cross_feature < 1.0e-9,
            "on-manifold dose {} leaked onto the independent feature: {:e}",
            m.dose,
            m.cross_feature
        );
        assert!(
            f.cross_feature < 1.0e-9,
            "flat dose {} leaked onto the independent feature: {:e}",
            f.dose,
            f.cross_feature
        );
    }

    // (3) At MATCHED per-row move norm, the on-manifold chord (whose only
    //     off-frame energy is the O(δ²) sagitta) deposits strictly less energy
    //     outside the atom's own rotating tangent frame than the fixed flat
    //     direction (off that tangent by an O(1) angle at most rows) at every
    //     dose — the same dominance the circle shows, on a curve that never wraps.
    for (m, f) in curve.manifold.points.iter().zip(curve.flat.points.iter()) {
        assert!(
            m.collateral < f.collateral,
            "non-cyclic on-manifold collateral {:e} must beat flat {:e} at dose {}",
            m.collateral,
            f.collateral,
            m.dose
        );
    }

    // (4) The aggregate verdict: collateral spent per unit on-target effect is
    //     strictly lower for the on-manifold arm on the non-cyclic manifold too.
    assert!(
        curve.manifold.efficiency.is_finite() && curve.flat.efficiency.is_finite(),
        "both arms must reach finite collateral efficiency"
    );
    assert!(
        curve.manifold_is_cleaner,
        "on-manifold steering must dominate flat on a non-cyclic manifold \
         (manifold={:.4}, flat={:.4})",
        curve.manifold.efficiency, curve.flat.efficiency
    );
    eprintln!(
        "[non-cyclic] collateral efficiency (per unit effect): manifold={:.4}  flat={:.4}  \
         (lower is a cleaner knob)",
        curve.manifold.efficiency, curve.flat.efficiency
    );
}
