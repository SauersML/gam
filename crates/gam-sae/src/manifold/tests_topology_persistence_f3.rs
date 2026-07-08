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

/// Product-circle grid embedded as a flat Clifford torus in R4.
fn torus_points(nu: usize, nv: usize) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((nu * nv, 4));
    let mut row = 0usize;
    for i in 0..nu {
        let u = std::f64::consts::TAU * (i as f64) / (nu as f64);
        for j in 0..nv {
            let v = std::f64::consts::TAU * (j as f64) / (nv as f64);
            pts[[row, 0]] = u.cos();
            pts[[row, 1]] = u.sin();
            pts[[row, 2]] = v.cos();
            pts[[row, 3]] = v.sin();
            row += 1;
        }
    }
    pts
}

/// Standard embedded torus in R3 with ring radius `major` and tube radius
/// `minor`, sampled on a deterministic product grid.
fn embedded_torus_points(nu: usize, nv: usize, major: f64, minor: f64) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((nu * nv, 3));
    let mut row = 0usize;
    for i in 0..nu {
        let u = std::f64::consts::TAU * (i as f64) / (nu as f64);
        for j in 0..nv {
            let v = std::f64::consts::TAU * (j as f64) / (nv as f64);
            let tube = major + minor * v.cos();
            pts[[row, 0]] = tube * u.cos();
            pts[[row, 1]] = tube * u.sin();
            pts[[row, 2]] = minor * v.sin();
            row += 1;
        }
    }
    pts
}

/// Six vertices of the octahedron on S2. Its VR complex has one dominant H2
/// shell before the opposite-vertex edges fill it.
fn octahedron_sphere_points() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ],
    )
    .unwrap()
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
    let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic)
        .expect("periodic atom has a topology prediction");
    assert_eq!(verdict.measured_betti.b0, 1, "circle is connected");
    assert_eq!(verdict.measured_betti.b1, 1, "circle must show one loop");
    assert_eq!(
        verdict.expected_betti.b1, 1,
        "periodic type predicts one loop"
    );
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
    let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic)
        .expect("periodic atom has a topology prediction");
    assert_eq!(
        verdict.measured_betti.b0, 7,
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
    let as_line = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Linear)
        .expect("linear atom has a topology prediction");
    assert_eq!(as_line.measured_betti.b0, 1, "a line is one component");
    assert_eq!(as_line.measured_betti.b1, 0, "a line has no loop");
    assert!(
        !as_line.contested,
        "a line raced as a line is clean: {}",
        as_line.note
    );

    // The SAME line raced as a circle: the predicted loop is absent → contested.
    let as_circle = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic)
        .expect("periodic atom has a topology prediction");
    assert_eq!(as_circle.expected_betti.b1, 1, "periodic predicts a loop");
    assert_eq!(
        as_circle.measured_betti.b1, 0,
        "the line has no loop to find"
    );
    assert!(
        as_circle.contested,
        "a circle fit on a line is contested: {}",
        as_circle.note
    );
}

#[test]
fn torus_signature_requires_two_independent_loops() {
    let pts = torus_points(4, 4);
    let as_torus = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Torus)
        .expect("torus atom has a topology prediction");
    assert_eq!(as_torus.measured_betti.b0, 1, "torus is connected");
    assert_eq!(
        as_torus.measured_betti.b1, 2,
        "torus must show two H1 loops"
    );
    assert_eq!(as_torus.expected_betti.b1, 2, "torus predicts two H1 loops");

    let as_circle = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Periodic)
        .expect("periodic atom has a topology prediction");
    assert_eq!(
        as_circle.measured_betti.b1, 2,
        "same cloud still measures two loops"
    );
    assert_eq!(as_circle.expected_betti.b1, 1, "circle predicts one loop");
    assert!(
        as_circle.contested,
        "a circle candidate on torus support must be contested: {}",
        as_circle.note
    );
}

#[test]
fn embedded_torus_grid_measures_two_h1_generators() {
    let pts = embedded_torus_points(16, 14, 2.5, 1.5);
    let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Torus)
        .expect("torus atom has a topology prediction");
    assert_eq!(verdict.measured_betti.b0, 1, "torus is connected");
    assert_eq!(
        verdict.measured_betti.b1, 2,
        "an embedded torus has two independent H1 generators; note: {}; H1: {:?}",
        verdict.note, verdict.h1
    );
    assert_eq!(
        verdict.measured_betti.b2,
        Some(1),
        "torus encloses one H2 void"
    );
    assert!(
        !verdict.contested,
        "a canonical torus raced as a torus must not be contested: {}",
        verdict.note
    );
}

#[test]
fn sphere_signature_measures_h2_shell() {
    let pts = octahedron_sphere_points();
    let verdict = topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Sphere)
        .expect("sphere atom has a topology prediction");
    assert_eq!(verdict.measured_betti.b0, 1, "sphere is connected");
    assert_eq!(verdict.measured_betti.b1, 0, "sphere has no H1 loop");
    assert_eq!(
        verdict.measured_betti.b2,
        Some(1),
        "sphere has one H2 shell"
    );
    assert!(
        !verdict.contested,
        "octahedron sphere should match the sphere signature: {}",
        verdict.note
    );
}

/// `n` points on a half-circle arc (embedded in the plane).
fn arc_points(n: usize, r: f64) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let theta = std::f64::consts::PI * (i as f64) / ((n - 1) as f64);
        pts[[i, 0]] = r * theta.cos();
        pts[[i, 1]] = r * theta.sin();
    }
    pts
}

#[test]
fn atlas_nerve_recovers_circle_and_arc() {
    // Atlas-first inversion: read topology from the NERVE of a chart cover,
    // never assuming it. A circle's nerve is a cycle (S¹); an arc's is a path.
    let circle = atlas_nerve(circle_points(60, 2.0).view());
    assert!(
        circle.is_circle(),
        "the nerve of a circle cover must recover S¹ (b₁=1, one component): {circle:?}"
    );
    let arc = atlas_nerve(arc_points(60, 2.0).view());
    assert!(
        arc.is_arc(),
        "the nerve of an arc cover must recover a path (b₁=0, one component): {arc:?}"
    );
    assert!(
        !arc.is_circle(),
        "an arc must not be mistaken for a circle: {arc:?}"
    );
}

/// #2159 — a genuine torus must measure `b₁ = 2` ROBUSTLY across sampling
/// densities. The old Pareto-frontier counter reported `{0, 1, 29}` on the same
/// torus at different grid resolutions (dropping the 2nd near-identical H₁
/// generator, or admitting off-staircase noise). The magnitude-based
/// signal/noise split must recover both generators at every resolution, on both
/// the flat Clifford torus in R⁴ (its two circle factors are EXACTLY symmetric —
/// the hardest case for any dominance rule) and the standard embedded torus in
/// R³, without contesting the raced torus type.
#[test]
fn torus_two_h1_generators_resolution_robust_2159() {
    for &(nu, nv) in &[(12usize, 10usize), (14, 12), (16, 14), (10, 8)] {
        let clifford = topology_persistence_verdict(
            torus_points(nu, nv).view(),
            &SaeAtomBasisKind::Torus,
        )
        .expect("torus atom has a topology prediction");
        assert_eq!(
            clifford.measured_betti.b1, 2,
            "Clifford torus {nu}×{nv} must measure two H1 generators; note: {}; H1: {:?}",
            clifford.note, clifford.h1
        );
        assert_eq!(clifford.measured_betti.b0, 1, "torus {nu}×{nv} is connected");

        let embedded = topology_persistence_verdict(
            embedded_torus_points(nu, nv, 2.5, 1.5).view(),
            &SaeAtomBasisKind::Torus,
        )
        .expect("torus atom has a topology prediction");
        assert_eq!(
            embedded.measured_betti.b1, 2,
            "embedded torus {nu}×{nv} must measure two H1 generators; note: {}; H1: {:?}",
            embedded.note, embedded.h1
        );
    }
}

#[test]
fn precomputed_kind_has_no_prediction_to_contest() {
    let pts = circle_points(20, 1.0);
    let verdict =
        topology_persistence_verdict(pts.view(), &SaeAtomBasisKind::Precomputed("x".into()));
    assert!(
        verdict.is_none(),
        "a caller-supplied basis carries no library topology to audit"
    );
}
