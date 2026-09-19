//! Unit coverage for the certify-last reseed loop (#2374, #2817). The loop
//! re-runs from a strategy change the refused certificate published, and only
//! while the certified value strictly drops between refusals:
//! `take_certify_reseed` decides whether there is a strategy change at all,
//! `certify_reseed_admitted` whether the loop may take it, and
//! `certify_resume_made_progress` is the exact predicate that decides "real
//! descent" vs "genuine floor".
use super::{
    ActiveSetReseed, CERTIFY_RESUME_PROGRESS_REL, CertifyReseedKind, HessianSource, OuterConfig,
    OuterPlan, OuterResult, Solver, certify_reseed_admitted, certify_resume_made_progress,
    OuterProblemSize, outer_criterion_resolution, take_certify_reseed,
};
use ndarray::array;

fn config_with_size(n_obs: Option<usize>, tolerance: f64) -> OuterConfig {
    OuterConfig {
        tolerance,
        problem_size: OuterProblemSize {
            n_obs,
            p_coefficients: n_obs.map(|_| 3),
        },
        ..OuterConfig::default()
    }
}

/// The criterion resolution is the statistical one, `τ_stat = 1/(2n)`, in the
/// criterion's own units: it depends on the declared observation count alone,
/// never on the stationarity tolerance, and a route that declares no size
/// resolves nothing (0), so no resolution-based stop can fire there. Before
/// C3 it was `max(rel, 1e-2·tolerance, 1e-12)·(1 + |V|)`, which grew with the
/// criterion's magnitude and therefore with `n` (#2954).
#[test]
fn criterion_resolution_is_one_over_two_n_and_independent_of_tolerance() {
    for &n in &[1usize, 50, 5_000, 1_000_000] {
        let expected = 1.0 / (2.0 * n as f64);
        for &tolerance in &[1.0e-12, 1.0e-5, 1.0e-2, 10.0] {
            let resolution = outer_criterion_resolution(&config_with_size(Some(n), tolerance));
            assert_eq!(
                resolution, expected,
                "n = {n}, tolerance = {tolerance}: the resolution must be 1/(2n)"
            );
        }
    }
    for &tolerance in &[1.0e-12, 1.0e-5, 1.0e-2] {
        assert_eq!(
            outer_criterion_resolution(&config_with_size(None, tolerance)),
            0.0,
            "an undeclared size must resolve nothing, not fall back to a relative band"
        );
    }
}

// ── Helper math (arbitrary floor) ────────────────────────────────────

#[test]
fn strict_descent_past_the_floor_is_progress() {
    let floor = 1.0e-4;
    // A drop far larger than floor·(1+|cost|)≈0.034 at cost scale ~1e2.
    assert!(certify_resume_made_progress(342.0, 300.0, floor));
    // A drop of many orders is trivially progress.
    assert!(certify_resume_made_progress(1.0e6, 1.0e3, floor));
}

#[test]
fn flat_or_uphill_reseed_is_not_progress() {
    let floor = 1.0e-4;
    // Exactly equal: no descent.
    assert!(!certify_resume_made_progress(100.0, 100.0, floor));
    // Uphill: a metric restart that landed worse is never progress.
    assert!(!certify_resume_made_progress(100.0, 100.5, floor));
    // A reduction SMALLER than the passed floor is within the flat band.
    let cost = 1.0e4;
    let sub_floor = floor * (1.0 + cost) * 0.5;
    assert!(!certify_resume_made_progress(cost, cost - sub_floor, floor));
}

#[test]
fn floor_anchors_on_the_smaller_cost_magnitude() {
    // With prior≈0 and a large-magnitude retried, anchoring on the smaller
    // (prior) magnitude keeps the floor tight so a genuine tiny descent near a
    // small optimum still registers, rather than being swamped by |retried|.
    let floor = 1.0e-4;
    // prior tiny-positive, retried strictly below it by more than floor·(1+0):
    assert!(certify_resume_made_progress(1.0e-2, 1.0e-3, floor));
}

#[test]
fn non_finite_retried_is_never_progress() {
    let floor = 1.0e-4;
    assert!(!certify_resume_made_progress(100.0, f64::NAN, floor));
    assert!(!certify_resume_made_progress(100.0, f64::INFINITY, floor));
    assert!(!certify_resume_made_progress(100.0, f64::NEG_INFINITY, floor));
}

// ── Production gate (roundoff floor) ─────────────────────────────────
//
// These pin the ACTUAL gate the loop runs (`CERTIFY_RESUME_PROGRESS_REL`),
// which must admit the tiny per-reseed descent a flat valley crawls out in
// and reject only numerical noise — the exact distinction the earlier
// cost-stall-floor gate got wrong (#2374: it stopped the survival LAML crawl
// after one hop and refused a well-posed fit).

#[test]
fn roundoff_gate_admits_the_tiny_flat_valley_crawl_step() {
    let rel = CERTIFY_RESUME_PROGRESS_REL;
    // The transformation-survival LAML moves ~4e-5 relative per reseed:
    // cost 342.0730 → 342.0580 is ~4.4e-5 relative, far above roundoff.
    assert!(certify_resume_made_progress(342.0730, 342.0580, rel));
    // Even a 1e-6 relative step at cost ~455 (the two-smooth cohort scale)
    // is real descent under the roundoff gate — the coarse cost-stall floor
    // (~1e-6·456 ≈ 4.6e-4) would have wrongly rejected it.
    assert!(certify_resume_made_progress(455.40, 455.40 - 5.0e-4, rel));
}

#[test]
fn roundoff_gate_rejects_noise_and_non_descent() {
    let rel = CERTIFY_RESUME_PROGRESS_REL;
    // A bitwise-identical reseed (genuine floor: BFGS found no descent from
    // the checkpoint) is not progress.
    assert!(!certify_resume_made_progress(455.40, 455.40, rel));
    // A reduction at the roundoff scale is noise, not descent: a few ULPs at
    // cost ~455 sits below CERTIFY_RESUME_PROGRESS_REL·(1+455).
    let noise = 4.0 * f64::EPSILON * (1.0 + 455.40);
    assert!(!certify_resume_made_progress(455.40, 455.40 - noise, rel));
    // Uphill / non-finite are never progress.
    assert!(!certify_resume_made_progress(455.40, 455.41, rel));
    assert!(!certify_resume_made_progress(455.40, f64::NAN, rel));
}

fn arc_plan_2817() -> OuterPlan {
    OuterPlan {
        solver: Solver::Arc,
        hessian_source: HessianSource::Analytic,
    }
}

/// #2817 (lead ruling 09-12): a refused certificate that publishes no strategy
/// change returns the refusal. The resume at the refused checkpoint re-ran the
/// same search with more iterations under a picked count, so a checkpoint whose
/// solver claimed convergence but carries no reseed yields no re-run at all.
#[test]
fn a_refused_checkpoint_without_a_strategy_change_returns_the_refusal_2817() {
    let mut claimed = OuterResult::new(array![0.25, -1.5], 3.0, 12, true, arc_plan_2817());
    assert!(
        claimed.solver_claimed_convergence(),
        "fixture precondition: the solver claimed convergence at the refused checkpoint"
    );
    assert!(
        take_certify_reseed(&mut claimed).is_none(),
        "a refusal with no published reseed must stand, not re-run the search from the checkpoint"
    );
}

/// The strategy changes a refused certificate can publish are taken in
/// precedence order, and every lower-precedence reseed is dropped with them.
#[test]
fn a_published_reseed_is_taken_in_precedence_order_and_the_rest_dropped_2817() {
    let mut result = OuterResult::new(array![1.0, 2.0], 3.0, 4, false, arc_plan_2817());
    result.tail_snap_reseed = Some(array![1.0, 3.0]);
    result.wrong_rail_reseed = Some(array![1.5, 2.0]);
    result.active_set_reseed = Some(ActiveSetReseed {
        rho: array![1.0, 2.0],
        bounds: (array![-4.0, 2.0], array![4.0, 2.0]),
    });
    let reseed = take_certify_reseed(&mut result)
        .expect("a confirmed-tail snap was published");
    assert_eq!(reseed.kind, CertifyReseedKind::TailSnap);
    assert_eq!(reseed.rho, array![1.0, 3.0]);
    assert!(reseed.search_bounds_override.is_none());
    assert!(
        result.wrong_rail_reseed.is_none() && result.active_set_reseed.is_none(),
        "lower-precedence reseeds are dropped with the one taken"
    );
    assert!(
        take_certify_reseed(&mut result).is_none(),
        "no reseed survives into the next iteration"
    );
}

/// #2817 (lead ruling 09-12): the certify-last loop has no resume count. The first
/// published reseed is taken; a later one only after the certified value strictly
/// dropped since the previous refusal, so a re-run that found no descent returns
/// the refusal instead of taking another reseed.
#[test]
fn a_reseed_without_certified_descent_since_the_last_refusal_returns_the_refusal_2817() {
    assert!(
        certify_reseed_admitted(None, 455.40),
        "the first published reseed is taken"
    );
    assert!(
        certify_reseed_admitted(Some(455.40), 455.40 - 5.0e-4),
        "a reseed after strict certified descent is taken"
    );
    assert!(
        !certify_reseed_admitted(Some(455.40), 455.40),
        "a re-run that certified the same value found no descent"
    );
    assert!(
        !certify_reseed_admitted(Some(455.40), 455.41),
        "a re-run that certified a higher value found no descent"
    );
    assert!(
        !certify_reseed_admitted(Some(455.40), f64::NAN),
        "a non-finite certified value is never descent"
    );
}
