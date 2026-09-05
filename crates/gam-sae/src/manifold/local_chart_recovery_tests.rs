#![cfg(test)]
//! #2818 recovery of #2280 transition contracts through surviving atlas output.

use super::{AtlasOrientability, ChartTransition, LocalAtlas, LocalAtlasConfig};
use crate::manifold::tests_topology_fixtures::{embedded_plane, spherical_band, swiss_roll};

fn genuine_triangle(atlas: &LocalAtlas) -> [&ChartTransition; 3] {
    for ab in atlas.transitions() {
        for bc in atlas
            .transitions()
            .iter()
            .filter(|edge| edge.from_patch == ab.to_patch)
        {
            for ac in atlas
                .transitions()
                .iter()
                .filter(|edge| edge.from_patch == ab.from_patch && edge.to_patch == bc.to_patch)
            {
                // Pairwise overlap alone does not establish a genuine triple.
                // Require a row in all three registered overlap components.
                if ab.shared_rows.iter().any(|row| {
                    bc.shared_rows.binary_search(row).is_ok()
                        && ac.shared_rows.binary_search(row).is_ok()
                }) {
                    return [ab, bc, ac];
                }
            }
        }
    }
    panic!("the deterministic atlas must have a genuine triple overlap");
}

fn triangle_defect(edges: [&ChartTransition; 3]) -> f64 {
    let [ab, bc, ac] = edges;
    let product = ac.rotation.t().dot(&bc.rotation.dot(&ab.rotation));
    product
        .indexed_iter()
        .map(|((row, column), value)| {
            let identity = if row == column { 1.0 } else { 0.0 };
            (value - identity).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn swiss_roll_charts_injective_and_cocycle_closes_2280() {
    let points = swiss_roll(40, 8);
    let atlas = LocalAtlas::build(points.view(), LocalAtlasConfig::balanced(points.nrows(), 2))
        .expect("the original swiss-roll atlas must build");
    assert!(atlas.chart_count() >= 3);
    for chart in atlas.charts() {
        assert!(chart.certificate.min_projection_stretch > 0.0);
        assert!(chart.certificate.captured_variance_fraction > 0.7);
    }
    let edges = genuine_triangle(&atlas);
    let defect = triangle_defect(edges);
    assert!(defect < 0.5, "swiss-roll rotation cocycle defect={defect}");
    assert_eq!(edges.iter().map(|edge| edge.sign).product::<i8>(), 1);
    eprintln!(
        "#2280 swiss-roll: charts={} cocycle_defect={defect:.6e}",
        atlas.chart_count()
    );
}

#[test]
fn embedded_plane_cocycle_closes_to_rounding_2280() {
    let points = embedded_plane(12, 12);
    let atlas = LocalAtlas::build(points.view(), LocalAtlasConfig::balanced(points.nrows(), 2))
        .expect("the original plane atlas must build");
    for chart in atlas.charts() {
        assert!(chart.certificate.captured_variance_fraction > 1.0 - 1e-9);
        assert!((chart.certificate.min_projection_stretch - 1.0).abs() < 1e-6);
    }
    let edges = genuine_triangle(&atlas);
    let defect = triangle_defect(edges);
    assert!(
        defect < 1e-8,
        "exact-plane rotation cocycle defect={defect}"
    );
    assert_eq!(
        atlas.observed_orientability(),
        AtlasOrientability::Orientable
    );
    // On the exact plane the complete affine maps must compose as well as
    // their rotations. This also exercises the live public transition action.
    let [ab, bc, ac] = edges;
    let row = *ab
        .shared_rows
        .iter()
        .find(|row| {
            bc.shared_rows.binary_search(row).is_ok() && ac.shared_rows.binary_search(row).is_ok()
        })
        .expect("the selected edges share a real row");
    let coordinate = atlas.charts()[ab.from_patch].project(points.row(row));
    let via_b = bc.apply(ab.apply(coordinate.view()).view());
    let direct = ac.apply(coordinate.view());
    let observed = atlas.charts()[ac.to_patch].project(points.row(row));
    for (&mapped, &projected) in direct.iter().zip(observed.iter()) {
        assert!(
            (mapped - projected).abs() < 1e-8,
            "the transition must reproduce the observed target coordinate"
        );
    }
    let affine_error = via_b
        .iter()
        .zip(direct.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        affine_error < 1e-8,
        "exact-plane affine cocycle error={affine_error}"
    );
    eprintln!(
        "#2280 plane: charts={} cocycle_defect={defect:.6e} affine_error={affine_error:.6e}",
        atlas.chart_count()
    );
}

#[test]
fn sphere_charts_injective_and_orientable_2280() {
    // The historical fixture is a spherical band with polar caps removed.
    // Its orientability and local transition contract do not imply sphere homology.
    let points = spherical_band(14, 20);
    let atlas = LocalAtlas::build(points.view(), LocalAtlasConfig::balanced(points.nrows(), 2))
        .expect("the original spherical-band atlas must build");
    for chart in atlas.charts() {
        assert!(chart.certificate.min_projection_stretch > 0.0);
    }
    assert_eq!(
        atlas.observed_orientability(),
        AtlasOrientability::Orientable
    );
    let edges = genuine_triangle(&atlas);
    assert_eq!(edges.iter().map(|edge| edge.sign).product::<i8>(), 1);
    let defect = triangle_defect(edges);
    assert!(
        defect < 0.75,
        "spherical-band rotation cocycle defect={defect}"
    );
    eprintln!(
        "#2280 spherical band: charts={} cocycle_defect={defect:.6e}",
        atlas.chart_count()
    );
}
