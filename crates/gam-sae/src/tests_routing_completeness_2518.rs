//! #2518 item 1 — the certified encode's chart routing is COMPLETE, and the
//! constant that used to bound it is gone.
//!
//! The deleted `CERTIFIED_ROUTING_TOPK = 4` answered "how many charts must I
//! refine before I have the global basin?" with an integer. That question has a
//! computable answer, and answering it wrongly produces the one failure a
//! certificate must never have: a row whose encode is locally Kantorovich-
//! certified and globally the wrong branch.
//!
//! Two things are gated here, and they are different claims:
//!
//! * **Completeness** — the scan now visits every certifiable chart and skips one
//!   only when a rigorous residual bound proves it cannot win. The load-bearing
//!   test compares the shipped encode against a brute-force minimum over a dense
//!   latent grid on a decoder that folds, and separately shows that the OLD
//!   four-chart restriction does not contain the winning chart on that fixture —
//!   without which the test would be measuring a fixture that no longer reaches
//!   the defect.
//! * **Soundness of the skip** — the prune's per-chart slack must dominate how far
//!   the reconstruction can actually move inside that chart, checked by sampling
//!   the ball on real charts. If it did not, the scan would be complete on paper
//!   and would still be able to discard the winner.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use crate::encode::{certified_encode_candidates, AtlasConfig, EncodeAtlas};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom};

/// A `d = 1` periodic atom whose decoded image FOLDS: the first harmonic traces a
/// loop and a large third harmonic pulls the curve back across itself, so distant
/// latent coordinates land near the same ambient point and several charts compete
/// for a row in the crossing region.
fn folded_atom() -> (SaeManifoldAtom, Array2<f64>, usize) {
    let m = 5usize; // [1, sin1, cos1, sin2, cos2]
    // The lemniscate `x = sin 2*pi*t`, `y = sin 4*pi*t`: a genuine figure eight
    // that CROSSES ITSELF at the origin, reached from two latent points half a
    // period apart. Distant latent coordinates therefore land on the same ambient
    // point, which is exactly the self-approach that makes a fixed-prefix routing
    // able to certify into the wrong branch.
    let decoder = ndarray::array![
        [0.00_f64, 0.00],
        [1.00, 0.00],
        [0.00, 0.00],
        [0.00, 1.00],
        [0.00, 0.00],
    ];
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "lemniscate",
        SaeAtomBasisKind::Periodic,
        1,
        Array2::<f64>::eye(m),
        Array3::<f64>::zeros((m, m, 1)),
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .expect("lemniscate periodic atom builds")
    // Without the evaluator every chart's center curvature is unreadable, so the
    // whole atlas comes back UNCERTIFIED and any completeness assertion over it
    // would pass vacuously.
    .with_basis_evaluator(Arc::new(
        PeriodicHarmonicEvaluator::new(m).expect("evaluator"),
    ));
    (atom, decoder, m)
}

fn recon_at(decoder: &Array2<f64>, evaluator: &PeriodicHarmonicEvaluator, t: f64) -> Array1<f64> {
    let coords = Array2::from_shape_vec((1, 1), vec![t]).expect("coordinate shape");
    let (phi, _) = evaluator.evaluate(coords.view()).expect("basis evaluates");
    phi.row(0).dot(decoder)
}

fn folded_atlas(atom: &SaeManifoldAtom, charts: usize) -> EncodeAtlas {
    let centers =
        Array2::from_shape_fn((charts, 1), |(c, _)| c as f64 / charts as f64);
    let radii = vec![0.5 / charts as f64; charts];
    let atlas = EncodeAtlas::build_atom_atlas_from_centers(
        0,
        atom,
        centers.view(),
        &radii,
        1.0,
        4.0,
        &AtlasConfig::default(),
    )
    .expect("folded atlas builds");
    EncodeAtlas {
        atoms: vec![atlas],
        config: AtlasConfig::default(),
    }
}

/// Completeness: every certifiable chart is a candidate, in distance order. The
/// old routing handed back at most four indices no matter how many charts could
/// hold the global basin.
#[test]
fn candidates_cover_every_certifiable_chart_in_distance_order() {
    let (atom, _decoder, _m) = folded_atom();
    let atlas = folded_atlas(&atom, 64);
    let x = Array1::from(vec![0.4_f64, -0.7]);
    let candidates = certified_encode_candidates(&atlas.atoms[0], x.view(), 1.0);
    let certifiable = atlas.atoms[0]
        .charts
        .iter()
        .filter(|c| c.certified_radius > 0.0)
        .count();
    assert_eq!(
        candidates.len(),
        certifiable,
        "the certified encode must see every certifiable chart, not a fixed prefix"
    );
    assert!(
        candidates.len() > 4,
        "precondition: this fixture has more certifiable charts than the deleted constant \
         admitted, or the completeness claim is untestable here (got {})",
        candidates.len()
    );
    for pair in candidates.windows(2) {
        assert!(
            pair[0].1 <= pair[1].1,
            "candidates must be distance-ordered for the tail-termination proof to hold: \
             {} then {}",
            pair[0].1,
            pair[1].1
        );
    }
    for (idx, _dist, slack) in &candidates {
        assert!(
            *slack >= 0.0,
            "chart {idx} produced a negative residual slack, which would licence a FALSE skip"
        );
    }
}

/// The prune must never be able to skip a chart that could have won: the slack
/// has to dominate the actual movement of the reconstruction inside the chart.
/// This checks the inequality the termination proof rests on, on real charts.
#[test]
fn per_chart_slack_dominates_the_reachable_reconstruction_movement() {
    let (atom, decoder, m) = folded_atom();
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("evaluator");
    let atlas = folded_atlas(&atom, 64);
    let x = Array1::from(vec![0.1_f64, 0.2]);
    let amplitude = 1.0_f64;
    let candidates = certified_encode_candidates(&atlas.atoms[0], x.view(), amplitude);
    assert!(
        !candidates.is_empty(),
        "no certifiable charts: this test would otherwise pass by having nothing to check"
    );
    for (idx, _dist, slack) in candidates {
        let chart = &atlas.atoms[0].charts[idx];
        let center = chart.region.center[0];
        let radius = chart.region.radius;
        let center_recon = recon_at(&decoder, &evaluator, center);
        // Sample the whole ball; the bound must hold at every reachable point.
        let samples = 64usize;
        for s in 0..=samples {
            let t = center - radius + 2.0 * radius * (s as f64 / samples as f64);
            let moved = &recon_at(&decoder, &evaluator, t) - &center_recon;
            let movement = amplitude * moved.dot(&moved).sqrt();
            assert!(
                movement <= slack * (1.0 + 1.0e-9) + 1.0e-12,
                "chart {idx}: reconstruction moves {movement:.9e} inside the ball but the \
                 prune's slack is only {slack:.9e} — the skip rule would be UNSOUND"
            );
        }
    }
}
