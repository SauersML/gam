//! Unit coverage for the certify-last checkpoint-resume progress gate
//! (#2374). The generalized loop keeps reseeding at the refused checkpoint
//! only while each reseed exploits real descent; `certify_resume_made_progress`
//! is the exact predicate that decides "real descent" vs "genuine floor", so
//! pinning it directly pins the loop's termination contract independent of the
//! solver dynamics that produce the reseeds.
use super::{
    CERTIFY_RESUME_PROGRESS_REL, OuterConfig, certify_resume_made_progress,
    outer_rel_cost_floor,
};

fn config_with_rel_cost(rel_cost: Option<f64>, tolerance: f64) -> OuterConfig {
    OuterConfig {
        tolerance,
        rel_cost_tolerance: rel_cost,
        ..OuterConfig::default()
    }
}

#[test]
fn rel_cost_floor_prefers_explicit_then_scaled_tolerance_never_below_hard_floor() {
    // Explicit relative tolerance wins verbatim.
    let explicit = config_with_rel_cost(Some(1.0e-3), 1.0e-5);
    assert_eq!(outer_rel_cost_floor(&explicit), 1.0e-3);
    // Absent, it derives from a small fraction of the absolute tolerance.
    let derived = config_with_rel_cost(None, 1.0e-2);
    assert!((outer_rel_cost_floor(&derived) - 1.0e-4).abs() <= 1.0e-16);
    // But never below the shared hard floor, however tight the tolerances.
    let tiny = config_with_rel_cost(Some(1.0e-30), 1.0e-30);
    assert_eq!(outer_rel_cost_floor(&tiny), super::COST_STALL_REL_TOL_FLOOR);
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
