//! E2 (gam#2234) — the collateral-damage curve, measured intrinsically in the
//! fitted dictionary's own representation, with NO LLM and NO outer-fit
//! convergence in the loop.
//!
//! The #2234 thesis is that a manifold SAE steers with less collateral than a
//! flat SAE: a flat intervention `x' = x + α·w` adds a FIXED ambient direction,
//! while the on-manifold group action `x' = x + a·(Φ_k(t⊕δ) − Φ_k(t))·B_k`
//! rotates with each row's chart coordinate to stay a chord of the atom's decoder
//! curve. The model-in-the-loop verdict (E1/E2 in `experiments/steering_e1`) reads
//! this off next-token KL and is blocked on real-fit convergence. This local test
//! establishes the SAME dominance STRUCTURALLY: at matched per-row move norm the
//! on-manifold arm deposits far less energy OUTSIDE the target atom's own local
//! decode-tangent frame than a fixed flat direction does.
//!
//! The fixture is two geometrically independent period-1 circle atoms with
//! disjoint ambient image planes (atom 0 in `span{e0,e1}`, atom 1 in
//! `span{e2,e3}`), built by hand — no joint fit is invoked, so this cannot wall
//! on the outer solver. Steering atom 0 must (a) leave the independent atom 1
//! untouched in BOTH arms (cross-feature leakage ≈ 0 — the geometric independence
//! is preserved), and (b) show the on-manifold arm strictly cleaner than flat on
//! the off-target damage that does exist (the flat direction is off the rotating
//! target tangent at most rows; the on-manifold chord's only off-target energy is
//! its second-order sagitta).

use super::*;
use crate::inference::steering::collateral_curve;
use ndarray::Array2;
use std::sync::Arc;

/// Build a hand-specified period-1 circle atom whose `(sin 2πt, cos 2πt)`
/// harmonics decode into ambient columns `col_sin` and `col_cos` of `R^p`.
fn circle_atom(name: &str, p: usize, col_sin: usize, col_cos: usize, coords: &Array2<f64>) -> SaeManifoldAtom {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    // PeriodicHarmonicEvaluator(3) emits [1, sin(2πt), cos(2πt)].
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, col_sin]] = 1.0;
    decoder[[2, col_cos]] = 1.0;
    SaeManifoldAtom::new(
        name,
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator)
}

#[test]
fn zz_e2_collateral_on_manifold_beats_flat_at_matched_norm() {
    let n = 240usize;
    let p = 8usize;

    // Two independent circles with disjoint ambient planes and independent phase
    // schedules (a shift keeps the two fitted-coordinate fields from coinciding).
    let coords0 = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let coords1 = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        ((row as f64 + 0.5) / n as f64 + 0.37).rem_euclid(1.0)
    });
    let atom0 = circle_atom("target-circle", p, 0, 1, &coords0);
    let atom1 = circle_atom("other-circle", p, 2, 3, &coords1);

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 2)),
        vec![coords0.clone(), coords1.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

    // Sweep a dose range; steer atom 0 (axis 0), measure collateral vs atom 1.
    let doses = [0.02_f64, 0.05, 0.1, 0.15, 0.2];
    let curve = collateral_curve(&term, 0, 0, &[1usize], &doses)
        .expect("collateral curve must build on the hand-fitted term");

    assert_eq!(curve.manifold.points.len(), doses.len());
    assert_eq!(curve.flat.points.len(), doses.len());

    // (1) The on-manifold arm is a genuine control knob: it turns the target
    //     feature (nonzero on-target effect) at every nonzero dose.
    for pt in &curve.manifold.points {
        assert!(
            pt.on_target_effect > 0.0,
            "on-manifold dose {} produced no on-target effect",
            pt.dose
        );
    }

    // (2) Cross-feature leakage onto the geometrically-independent atom 1 is ≈ 0
    //     in BOTH arms — steering a feature must not spuriously activate an
    //     orthogonal one. (The atoms' image planes are disjoint, so this is the
    //     structural locality guarantee, exact up to floating point.)
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

    // (3) At MATCHED per-row move norm, the on-manifold arm deposits strictly less
    //     energy outside the target atom's own frame than the fixed flat direction
    //     at every dose — the on-manifold chord tracks the rotating tangent, the
    //     flat direction cannot.
    for (m, f) in curve.manifold.points.iter().zip(curve.flat.points.iter()) {
        assert!(
            m.collateral < f.collateral,
            "on-manifold collateral {:e} must beat flat {:e} at dose {}",
            m.collateral,
            f.collateral,
            m.dose
        );
    }

    // (4) The aggregate E2 verdict: collateral spent per unit on-target effect is
    //     strictly lower for the on-manifold arm.
    assert!(
        curve.manifold.efficiency.is_finite() && curve.flat.efficiency.is_finite(),
        "both arms must reach finite collateral efficiency"
    );
    assert!(
        curve.manifold_is_cleaner,
        "on-manifold steering must dominate flat on collateral efficiency \
         (manifold={:.4}, flat={:.4})",
        curve.manifold.efficiency, curve.flat.efficiency
    );
    eprintln!(
        "[E2] collateral efficiency (per unit effect): manifold={:.4}  flat={:.4}  \
         (lower is a cleaner knob)",
        curve.manifold.efficiency, curve.flat.efficiency
    );
}
