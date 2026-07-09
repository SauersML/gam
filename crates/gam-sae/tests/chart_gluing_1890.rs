//! #1890 — the chart-gluing lane: a second, code-blind fusion trigger for
//! atoms OVER-TILING one manifold.
//!
//! The co-activation fusion lane fires on DEPENDENT (co-firing) codes. Atoms
//! that split one circle into disjoint arcs have DISJOINT supports — hence
//! anti-correlated codes — so that lane is structurally blind to them. The glue
//! lane screens such a pair GEOMETRICALLY (decoder ambient-span principal angles
//! × disjoint supports) and accepts on a seam EQUIVALENCE e-value against the
//! churn-null scatter, not a fit-improvement gate.
//!
//! Two fixtures pin the discriminator:
//!
//!  * ONE circle deliberately over-tiled into K=6 disjoint arc atoms that share
//!    a single decoder (the same circle). The glue lane must PROPOSE gluing with
//!    a certifying (large positive) seam e-value, while the co-activation lane
//!    provably proposes NOTHING (no `Fusion`).
//!  * TWO genuinely distinct circles (orthogonal ambient planes) with adjacent
//!    disjoint supports. Their seam e-value must stay BELOW the acceptance
//!    threshold — the geometry, not the support pattern, is what forbids gluing.

use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};
use gam_sae::structure_harvest::{harvest_move_proposals, HarvestParams};
use gam_solve::structure_search::{MoveProposal, StructureMove};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;

const ON: f64 = 6.0;
const OFF: f64 = -6.0;

/// Build a K-atom periodic SAE term over `n` rows. Atom `j` owns the contiguous
/// arc `arcs[j] = (start, end)` (disjoint supports) and decodes through
/// `decoders[j]` (a `3 × p` periodic-harmonic decoder). Every atom shares the
/// full-circle coordinate `t = row / n` and a `Circle { period: 1.0 }` manifold,
/// so a decoder that maps the fundamental `sin/cos` pair onto two ambient axes
/// traces a unit circle in that plane.
fn build_term(
    n: usize,
    arcs: &[(usize, usize)],
    decoders: &[Array2<f64>],
) -> SaeManifoldTerm {
    let k = arcs.len();
    assert_eq!(decoders.len(), k);
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    let mut atoms = Vec::with_capacity(k);
    let mut coord_blocks = Vec::with_capacity(k);
    for decoder in decoders.iter() {
        let atom = SaeManifoldAtom::new(
            "arc",
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords.clone());
    }

    let mut logits = Array2::<f64>::zeros((n, k));
    for (atom, &(start, end)) in arcs.iter().enumerate() {
        for row in 0..n {
            logits[[row, atom]] = if row >= start && row < end { ON } else { OFF };
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

fn rho_for(k: usize) -> SaeManifoldRho {
    SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k])
}

/// A small deterministic reconstruction residual (the isometry-tolerance band):
/// magnitude ~0.02, well above the curve-grid quantization floor and well below
/// the ambient circle radius.
fn small_residuals(n: usize, p: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        0.02 * ((row * 7 + col * 3) as f64).sin()
    })
}

/// A decoder that traces a unit circle in the ambient plane `(ax_sin, ax_cos)`:
/// fundamental `sin → ax_sin`, `cos → ax_cos` (columns 1 and 2 of the
/// order-3 periodic harmonic basis, matching the seed-dictionary convention).
fn circle_decoder(p: usize, ax_sin: usize, ax_cos: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((3, p));
    d[[1, ax_sin]] = 1.0;
    d[[2, ax_cos]] = 1.0;
    d
}

fn glue_triggers(proposals: &[MoveProposal]) -> Vec<f64> {
    proposals
        .iter()
        .filter_map(|prop| match prop.mv {
            StructureMove::Glue { .. } => Some(prop.trigger),
            _ => None,
        })
        .collect()
}

fn fusion_count(proposals: &[MoveProposal]) -> usize {
    proposals
        .iter()
        .filter(|prop| matches!(prop.mv, StructureMove::Fusion { .. }))
        .count()
}

/// The ledger acceptance threshold: an e-value certifies at α when its log-e
/// clears `ln(1/α)`. A carried seam log-e above this would glue.
fn accept_threshold(alpha: f64) -> f64 {
    (1.0 / alpha).ln()
}

#[test]
fn over_tiled_circle_glues_where_coactivation_is_silent() {
    // One circle split into 6 disjoint arcs, all sharing ONE decoder (the same
    // unit circle in the ambient (0,1) plane).
    let n = 120;
    let k = 6;
    let p = 4;
    let arc = n / k;
    let arcs: Vec<(usize, usize)> = (0..k).map(|j| (j * arc, (j + 1) * arc)).collect();
    let decoders: Vec<Array2<f64>> = (0..k).map(|_| circle_decoder(p, 0, 1)).collect();
    let term = build_term(n, &arcs, &decoders);
    let rho = rho_for(k);
    let residuals = small_residuals(n, p);

    // Enough fusion budget (reused as the glue budget) to expose several pairs;
    // births/fissions off so the report is a clean fusion-vs-glue contrast.
    let params = HarvestParams {
        max_fusions: 8,
        max_fissions: 0,
        max_births: 0,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();

    // The co-activation lane is provably silent on disjoint arcs.
    assert_eq!(
        fusion_count(&report.proposals),
        0,
        "co-activation fusion lane must propose NOTHING on disjoint-support arcs"
    );

    // Every disjoint pair of shared-circle arcs is geometrically screened.
    assert_eq!(
        report.glue_candidates_screened,
        k * (k - 1) / 2,
        "all disjoint aligned pairs should pass the geometric pre-screen"
    );

    // The glue lane fires and its seam e-value certifies (clears the α=0.05
    // acceptance threshold by a wide margin — the arcs ARE one circle).
    let triggers = glue_triggers(&report.proposals);
    assert!(
        !triggers.is_empty(),
        "glue lane must propose over-tiled arcs the co-activation lane cannot see"
    );
    let best = triggers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        best > accept_threshold(0.05),
        "seam equivalence e-value must certify the glue; got log-e {best}"
    );
}

#[test]
fn distinct_circles_do_not_glue() {
    // Two DIFFERENT circles (orthogonal ambient planes) with adjacent disjoint
    // supports: the disjoint-support signature is present, but the geometry is
    // not — they must NOT glue.
    let n = 40;
    let p = 4;
    let arcs = [(0usize, 20usize), (20usize, 40usize)];
    let decoders = [circle_decoder(p, 0, 1), circle_decoder(p, 2, 3)];
    let term = build_term(n, &arcs, &decoders);
    let rho = rho_for(2);
    let residuals = small_residuals(n, p);

    let params = HarvestParams {
        max_fusions: 8,
        max_fissions: 0,
        max_births: 0,
    };
    let report = harvest_move_proposals(&term, &rho, residuals.view(), &params).unwrap();

    // Whether or not the pair is proposed, its seam e-value must stay below the
    // acceptance threshold — the equivalence test rejects distinct manifolds.
    for trigger in glue_triggers(&report.proposals) {
        assert!(
            trigger < accept_threshold(0.05),
            "distinct circles must not certify a glue; got log-e {trigger}"
        );
    }
}
