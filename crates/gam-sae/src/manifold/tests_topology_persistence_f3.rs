//! Reviewer-F3 persistent-homology topology audit tests.
//!
//! Topology is measured, not latched: these exercise the Vietoris–Rips
//! persistence primitive and the raced-type agreement verdict on synthetic
//! clouds whose true topology is known.
//!
//! * a clean circle → one dominant H₁ loop, one component, agrees with a raced
//!   `Periodic` (circle) type;
//! * a 7-cluster ring forced through a circle fit → 7 persistent H₀ bars, the
//!   `contested` flag raised (disagrees with the connected circle winner);
//! * a straight line → no loop, one component, clean against a raced `Linear`
//!   type, and CONTESTED against a raced circle (a loop predicted where the
//!   data is a line).

use super::*;
use ndarray::Array2;

/// `n` points evenly spaced on a radius-`r` circle in the plane.
fn circle_points(n: usize, r: f64) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
        pts[[i, 0]] = r * theta.cos();
        pts[[i, 1]] = r * theta.sin();
    }
    pts
}

/// `clusters` tight blobs of `per` points each, blob centres evenly spaced on a
/// radius-`r` ring. The within-blob jitter is a deterministic small lattice so
/// the inter-blob gap dominates the within-blob spacing by orders of magnitude.
fn cluster_ring_points(clusters: usize, per: usize, r: f64, jitter: f64) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((clusters * per, 2));
    let mut idx = 0;
    for c in 0..clusters {
        let theta = std::f64::consts::TAU * (c as f64) / (clusters as f64);
        let cx = r * theta.cos();
        let cy = r * theta.sin();
        for j in 0..per {
            // Deterministic tiny offset on a small grid around the centre.
            let a = (j % 3) as f64 - 1.0;
            let b = (j / 3) as f64 - 1.0;
            pts[[idx, 0]] = cx + jitter * a;
            pts[[idx, 1]] = cy + jitter * b;
            idx += 1;
        }
    }
    pts
}

/// `n` points evenly spaced on a straight segment (embedded in the plane).
fn line_points(n: usize, length: f64) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = length * (i as f64) / ((n - 1) as f64);
        pts[[i, 0]] = t;
        pts[[i, 1]] = 0.0;
    }
    pts
}

#[test]
fn vietoris_rips_finds_the_circle_loop() {
    let pts = circle_points(24, 1.0);
    let diagram = vietoris_rips_persistence(pts.view());
    // Exactly one essential H₀ component (VR connects the ring at its diameter).
    let essential_h0 = diagram.h0.iter().filter(|b| b.is_essential()).count();
    assert_eq!(essential_h0, 1, "a circle is one connected component");
    // A dominant H₁ loop exists whose persistence is a large fraction of the
    // diameter — far above the nearest-neighbour spacing.
    let top_h1 = diagram
        .h1
        .iter()
        .map(|b| b.persistence())
        .fold(0.0_f64, f64::max);
    assert!(
        top_h1 > 1.0,
        "the circle's loop must persist well past unit spacing; got {top_h1}"
    );
}

#[test]
fn circle_cloud_agrees_with_a_raced_circle() {
    let pts = circle_points(40, 2.0);
    let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic, 1)
        .expect("periodic atom has a topology prediction");
    assert_eq!(verdict.n_components, 1, "circle is connected");
    assert!(verdict.has_loop, "circle must show a loop");
    assert!(verdict.expected_loop, "periodic type predicts a loop");
    assert!(
        !verdict.contested,
        "a true circle raced as a circle is not contested: {}",
        verdict.note
    );
}

#[test]
fn seven_cluster_ring_forced_through_circle_is_contested() {
    // Seven tight blobs on a ring, but the atom was raced `Periodic` (a circle).
    let pts = cluster_ring_points(7, 6, 3.0, 0.01);
    let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic, 1)
        .expect("periodic atom has a topology prediction");
    assert_eq!(
        verdict.n_components, 7,
        "the seven blobs must register as seven H₀ components: {}",
        verdict.note
    );
    assert!(
        verdict.contested,
        "seven clusters disagree with a connected circle winner: {}",
        verdict.note
    );
}

#[test]
fn line_is_clean_against_a_line_and_contested_against_a_circle() {
    let pts = line_points(40, 5.0);
    // Raced as the (loop-free) linear patch: clean.
    let as_line = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Linear, 1)
        .expect("linear atom has a topology prediction");
    assert_eq!(as_line.n_components, 1, "a line is one component");
    assert!(!as_line.has_loop, "a line has no loop");
    assert!(
        !as_line.contested,
        "a line raced as a line is clean: {}",
        as_line.note
    );

    // The SAME line raced as a circle: the predicted loop is absent → contested.
    let as_circle = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic, 1)
        .expect("periodic atom has a topology prediction");
    assert!(as_circle.expected_loop, "periodic predicts a loop");
    assert!(!as_circle.has_loop, "the line has no loop to find");
    assert!(
        as_circle.contested,
        "a circle fit on a line is contested: {}",
        as_circle.note
    );
}

#[test]
fn precomputed_kind_has_no_prediction_to_contest() {
    let pts = circle_points(20, 1.0);
    let verdict =
        topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Precomputed("x".into()), 1);
    assert!(
        verdict.is_none(),
        "a caller-supplied basis carries no library topology to audit"
    );
}
