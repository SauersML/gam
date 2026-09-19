//! gam#2926 / gam#2943 part 3(iii): the rigid Bernoulli intercept is solved in the
//! root slots of the fit's latent law.

use super::family::BernoulliAnchorLaw;
use super::{EmpiricalZGrid, LatentMeasureKind};
use crate::latent_anchor::{anchor_derivatives_in_slot, solve_anchor};
use crate::test_support::skewed_grid;
use gam_math::probability::normal_cdf;

/// `log Σ_k w_k Φ(a + b·u_k) − log Φ(q)`, the Bernoulli calibration residual in
/// linear space, independent of the anchor's log-space residual.
fn calibration_log_residual(a: f64, q: f64, b: f64, nodes: &[f64], weights: &[f64]) -> f64 {
    let mass: f64 = nodes
        .iter()
        .zip(weights)
        .map(|(&u, &w)| w * normal_cdf(a + b * u))
        .sum();
    mass.ln() - normal_cdf(q).ln()
}

fn skewed_global_law() -> (crate::latent_anchor::AnchorGridOwned, LatentMeasureKind) {
    let grid = skewed_grid();
    let kind = LatentMeasureKind::GlobalEmpirical {
        grid: EmpiricalZGrid {
            nodes: grid.nodes.clone(),
            weights: grid.weights.clone(),
        },
    };
    (grid, kind)
}

/// A repeat of a row's equation is answered from its slot bit for bit, a moved
/// iterate is solved from the closed form to bitwise the root a fresh solve
/// finds (gam#2983), and the counts name each.
#[test]
fn rigid_intercept_slots_answer_repeats_and_solve_moved_iterates_from_the_closed_form_2926() {
    let (grid, kind) = skewed_global_law();
    let law = BernoulliAnchorLaw::from_kind(&kind, 4)
        .expect("materialise the law")
        .expect("an empirical law has root slots");
    let context = law.row_context(2, &grid.nodes).expect("the law's own grid");
    let (q, b) = (0.7, 1.1);

    let cold = anchor_derivatives_in_slot(q, b, context, 2, 0)
        .expect("cold solve")
        .alpha;
    assert_eq!(
        cold.to_bits(),
        solve_anchor(q, b, grid.view()).expect("closed-form seed").to_bits()
    );
    let residual = calibration_log_residual(cold, q, b, &grid.nodes, &grid.weights);
    assert!(residual.abs() < 1e-12, "calibration residual {residual:e}");
    let repeat = anchor_derivatives_in_slot(q, b, context, 2, 0)
        .expect("repeat")
        .alpha;
    assert_eq!(repeat.to_bits(), cold.to_bits());

    let (moved_q, moved_b) = (0.74, 1.06);
    let moved = anchor_derivatives_in_slot(moved_q, moved_b, context, 2, 0)
        .expect("moved solve")
        .alpha;
    let residual = calibration_log_residual(moved, moved_q, moved_b, &grid.nodes, &grid.weights);
    assert!(residual.abs() < 1e-12, "moved calibration residual {residual:e}");
    let fresh = solve_anchor(moved_q, moved_b, grid.view()).expect("fresh solve");
    assert_eq!(
        moved.to_bits(),
        fresh.to_bits(),
        "moved {moved:+.17e} vs fresh {fresh:+.17e}"
    );

    let counts = law.counts();
    assert_eq!((counts.hits, counts.cold_solves), (1, 2), "{counts}");
    assert!(counts.cold_evaluations >= 2, "{counts}");
}

/// A row that reads a grid other than the law's is refused rather than answered
/// with the law's roots, and the standard-normal law carries no slots.
#[test]
fn rigid_intercept_slots_refuse_a_grid_that_is_not_the_law_2926() {
    let (grid, kind) = skewed_global_law();
    let law = BernoulliAnchorLaw::from_kind(&kind, 4)
        .expect("materialise the law")
        .expect("an empirical law has root slots");
    assert!(law.row_context(0, &grid.nodes[1..]).is_err());
    assert!(
        BernoulliAnchorLaw::from_kind(&LatentMeasureKind::StandardNormal, 4)
            .expect("standard normal")
            .is_none()
    );
}
