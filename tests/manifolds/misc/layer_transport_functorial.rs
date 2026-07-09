//! Functorial layer-transport maps (#1013): synthetic circle→circle ground
//! truth. Self-constructed truth is the assertion target (reference-as-truth
//! paradigm): a known degree-±1 smooth map with noise must yield the planted
//! winding degree, an isometry defect near its analytic value, a passing
//! composition law on consistent triples, and a failing one on a planted
//! inconsistent triple — with rotation-gauge invariance of the verdict.

use gam::inference::layer_transport::{
    ChartTopology, DEFAULT_COMPOSITION_GRID, composition_defect, fit_transport_map,
    transport_ladder,
};
use ndarray::Array1;
use std::f64::consts::TAU;

/// Deterministic splitmix-style generator: reproducible noise, no rand dep.
struct DetNoise(u64);

impl DetNoise {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 11) ^ (self.0 >> 33);
        (bits & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
    }

    /// Uniform noise on `(−scale, scale)`.
    fn jitter(&mut self, scale: f64) -> f64 {
        (self.next_unit() - 0.5) * 2.0 * scale
    }
}

fn uniform_angles(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| TAU * (i as f64 + 0.37) / n as f64 % TAU))
}

const N_OBS: usize = 400;
const NOISE: f64 = 0.02;

#[test]
fn degree_one_map_recovers_degree_and_isometry_defect() {
    let t = uniform_angles(N_OBS);
    let mut rng = DetNoise(0x1013_0001);
    // h(t) = t + 0.4 + 0.3 sin(t + 0.7): degree-1 cover, h' = 1 + 0.3cos > 0.
    // Analytic isometry defect under the uniform density:
    // E[(|h'|−1)²] = E[(0.3 cos)²] = 0.3²/2 = 0.045.
    let target = t.mapv(|v| (v + 0.4 + 0.3 * (v + 0.7).sin()).rem_euclid(TAU));
    let observed = target.mapv(|v| (v + rng.jitter(NOISE)).rem_euclid(TAU));

    let fit = fit_transport_map(
        t.view(),
        observed.view(),
        ChartTopology::Circle,
        ChartTopology::Circle,
    )
    .expect("degree-1 transport fit");

    assert_eq!(
        fit.degree,
        Some(1),
        "planted winding degree must be recovered"
    );
    assert!(
        fit.topology_preserved,
        "a fold-free degree-1 cover preserves the circle topology: min directional h' = {}",
        fit.min_directional_derivative
    );
    assert!(
        fit.min_directional_derivative > 0.0,
        "h' = 1 + 0.3cos never crosses zero; got min {}",
        fit.min_directional_derivative
    );
    let truth = 0.045;
    assert!(
        (fit.isometry_defect - truth).abs() < 0.02,
        "isometry defect {} should be near analytic {}",
        fit.isometry_defect,
        truth
    );
    assert!(
        fit.isometry_defect_se.is_finite() && fit.isometry_defect_se > 0.0,
        "delta-method SE must be a positive finite number, got {}",
        fit.isometry_defect_se
    );
    assert!(
        fit.isometry_defect_se < 0.05,
        "with n=400 and σ=0.02 the defect SE must be small, got {}",
        fit.isometry_defect_se
    );
    assert!(fit.edf.is_finite() && fit.edf >= 1.0, "EDF = {}", fit.edf);
    let conc = fit
        .degree_concentration
        .expect("circle→circle concentration");
    assert!(
        conc > 0.8,
        "de-wound residual must be concentrated, R = {conc}"
    );
}

#[test]
fn reflected_map_recovers_degree_minus_one() {
    let t = uniform_angles(N_OBS);
    let mut rng = DetNoise(0x1013_0002);
    let observed = t.mapv(|v| (-v + 0.2 + 0.2 * v.sin() + rng.jitter(NOISE)).rem_euclid(TAU));

    let fit = fit_transport_map(
        t.view(),
        observed.view(),
        ChartTopology::Circle,
        ChartTopology::Circle,
    )
    .expect("degree -1 transport fit");

    assert_eq!(fit.degree, Some(-1));
    assert!(
        fit.topology_preserved,
        "orientation-reversing homeomorphisms preserve the topology"
    );
}

#[test]
fn collapsed_map_breaks_topology() {
    let t = uniform_angles(N_OBS);
    let mut rng = DetNoise(0x1013_0003);
    // Null-homotopic image: θ ↦ π + 0.9 sin θ wraps the circle onto an arc.
    let observed =
        t.mapv(|v| (std::f64::consts::PI + 0.9 * v.sin() + rng.jitter(NOISE)).rem_euclid(TAU));

    let fit = fit_transport_map(
        t.view(),
        observed.view(),
        ChartTopology::Circle,
        ChartTopology::Circle,
    )
    .expect("collapsed transport fit");

    assert_eq!(fit.degree, Some(0), "an arc image has winding degree 0");
    assert!(
        !fit.topology_preserved,
        "circle→arc must be flagged as topology-breaking"
    );
}

#[test]
fn interval_stretch_map_reports_compute_layer_defect() {
    let n = N_OBS;
    let t = Array1::from_iter((0..n).map(|i| i as f64 / (n - 1) as f64));
    let mut rng = DetNoise(0x1013_0004);
    // h(t) = 2t on [0,1] → [0,2]: monotone (topology preserved) but |h'| = 2
    // everywhere, so the isometry defect is exactly 1 — a COMPUTE-layer map.
    let observed = t.mapv(|v| 2.0 * v + rng.jitter(NOISE));

    let fit = fit_transport_map(
        t.view(),
        observed.view(),
        ChartTopology::Interval { lo: 0.0, hi: 1.0 },
        ChartTopology::Interval { lo: 0.0, hi: 2.0 },
    )
    .expect("interval transport fit");

    assert_eq!(fit.degree, None, "winding degree is circle→circle only");
    assert!(
        fit.topology_preserved,
        "a monotone affine map is a homeomorphism"
    );
    assert!(
        (fit.isometry_defect - 1.0).abs() < 0.3,
        "uniform 2× stretch has defect (2−1)² = 1, got {}",
        fit.isometry_defect
    );
}

/// Build the consistent A→B→C synthetic stack used by the composition tests.
fn composition_stack(seed: u64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let t = uniform_angles(N_OBS);
    let mut rng = DetNoise(seed);
    let f = |v: f64| v + 0.25 * v.sin();
    let g = |u: f64| u + 0.6 + 0.2 * u.sin();
    let coords_b = t.mapv(|v| (f(v) + rng.jitter(NOISE)).rem_euclid(TAU));
    let coords_c = Array1::from_iter(
        t.iter()
            .map(|&v| (g(f(v)) + rng.jitter(NOISE)).rem_euclid(TAU)),
    );
    (t, coords_b, coords_c)
}

#[test]
fn composition_law_passes_on_consistent_triples_and_does_not_fit_target_shift_away() {
    let (t, coords_b, coords_c) = composition_stack(0x1013_0005);
    let circle = ChartTopology::Circle;
    let h_ab = fit_transport_map(t.view(), coords_b.view(), circle, circle).expect("h_ab");
    let h_bc = fit_transport_map(coords_b.view(), coords_c.view(), circle, circle).expect("h_bc");
    let h_ac = fit_transport_map(t.view(), coords_c.view(), circle, circle).expect("h_ac");

    let report = composition_defect(&h_ab, &h_bc, &h_ac, DEFAULT_COMPOSITION_GRID)
        .expect("consistent composition test");
    assert!(
        report.rms_defect < 0.08,
        "consistent triple must have a small RMS defect, got {}",
        report.rms_defect
    );
    assert!(
        report.p_value > 0.01,
        "consistent triple must not reject the composition law: p = {}",
        report.p_value
    );

    // Only the direct route is shifted while the composed route still lands in
    // the original C chart. This is a genuine composition violation; fitting a
    // fresh target rotation here would erase it.
    let rotated_c = coords_c.mapv(|v| (v + 1.3).rem_euclid(TAU));
    let h_ac_rot =
        fit_transport_map(t.view(), rotated_c.view(), circle, circle).expect("rotated h_ac");
    let rotated = composition_defect(&h_ab, &h_bc, &h_ac_rot, DEFAULT_COMPOSITION_GRID)
        .expect("rotated composition test");
    assert!(
        rotated.rms_defect > 1.0,
        "one-route target shift must remain visible; rms defect = {}",
        rotated.rms_defect
    );
    assert!(
        rotated.p_value < 0.01,
        "one-route target shift must reject composition; p = {}",
        rotated.p_value
    );
    assert_eq!(rotated.gauge_rotation, 0.0);
    assert!(!rotated.gauge_reflected);
}

#[test]
fn composition_law_rejects_a_planted_inconsistent_triple() {
    let (t, coords_b, coords_c) = composition_stack(0x1013_0006);
    let circle = ChartTopology::Circle;
    let h_ab = fit_transport_map(t.view(), coords_b.view(), circle, circle).expect("h_ab");
    let h_bc = fit_transport_map(coords_b.view(), coords_c.view(), circle, circle).expect("h_bc");

    // Planted incoherence: the direct A→C map carries an extra 0.5·sin(2t)
    // warp that no circle isometry (rotation/reflection) can remove.
    let mut rng = DetNoise(0x1013_0007);
    let f = |v: f64| v + 0.25 * v.sin();
    let g = |u: f64| u + 0.6 + 0.2 * u.sin();
    let coords_c_bad = Array1::from_iter(
        t.iter()
            .map(|&v| (g(f(v)) + 0.5 * (2.0 * v).sin() + rng.jitter(NOISE)).rem_euclid(TAU)),
    );
    let h_ac_bad =
        fit_transport_map(t.view(), coords_c_bad.view(), circle, circle).expect("bad h_ac");

    let report = composition_defect(&h_ab, &h_bc, &h_ac_bad, DEFAULT_COMPOSITION_GRID)
        .expect("inconsistent composition test");
    assert!(
        report.rms_defect > 0.2,
        "planted 0.5·sin(2t) warp must show as a large defect, got {}",
        report.rms_defect
    );
    assert!(
        report.p_value < 1e-3,
        "the composition law must reject the planted warp: p = {}",
        report.p_value
    );
    assert!(
        report.max_studentized_defect > 3.0,
        "the planted warp must exceed the composed bands, z_max = {}",
        report.max_studentized_defect
    );
}

#[test]
fn transport_ladder_wires_adjacent_two_hop_and_composition_fields() {
    let (t, coords_b, coords_c) = composition_stack(0x1013_0008);
    let layers = [20usize, 21, 22];
    let coords = [t, coords_b, coords_c];
    let topologies = [ChartTopology::Circle; 3];

    let ladder = transport_ladder(&layers, &coords, &topologies).expect("ladder");
    assert_eq!(ladder.adjacent.len(), 2);
    assert_eq!(ladder.two_hop.len(), 1);
    let adj = &ladder.adjacent[0];
    assert_eq!((adj.layer_from, adj.layer_to), (20, 21));
    assert_eq!(adj.degree, Some(1));
    assert!(
        adj.composition_p_value.is_none(),
        "adjacent maps carry no triple"
    );
    let hop = &ladder.two_hop[0];
    assert_eq!((hop.layer_from, hop.layer_to), (20, 22));
    assert_eq!(hop.degree, Some(1));
    assert!(hop.composition_defect.expect("rms defect") < 0.08);
    assert!(hop.composition_p_value.expect("composition p") > 0.01);
    assert!(hop.composition_max_studentized.is_some());
    assert!(hop.composition_gauge_reflected.is_some());
}
