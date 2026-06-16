//! Discovery-recall control for the Fisher-mass enrichment producer
//! (`RowSamplingMeasure`, role (c) of #980).
//!
//! # The scenario this guards
//!
//! A SAE-manifold fit sees many rows. Most are *common* and *behaviorally
//! quiet*: they activate a frequent feature that barely moves the model's
//! output (low output-Fisher mass). A handful of rows are *rare* but
//! *behaviorally loud*: they carry a feature that, when present, strongly drives
//! the output (high Fisher mass). Under plain uniform sampling, the rare-loud
//! rows are drowned — a discovery/seeding batch almost never looks at them, so
//! the important feature is never surfaced. That is the recall failure the
//! Fisher-mass enrichment measure is built to fix: it OVERSAMPLES the live rows.
//!
//! # What we assert
//!
//! 1. **Under-representation baseline.** Uniform sampling gives the rare rows
//!    only their population share of attention — vanishing for a rare feature.
//! 2. **Enrichment lifts recall.** The Fisher-mass measure assigns the rare-loud
//!    rows weight far above their population share, so their expected (and
//!    actually drawn, deterministically) representation in a batch rises
//!    markedly.
//! 3. **No loss is altered.** The enrichment is a SAMPLING measure only: it is
//!    derived purely from the metric's geometry and never multiplies, reweights,
//!    or otherwise touches any per-row loss / quad-form. We assert the per-row
//!    `quad_form` (the criterion-facing residual square) is bit-for-bit
//!    identical whether or not the enrichment measure exists.
//!
//! All draws are deterministic (fixed seeds, index-derived jitter); no clock.

use std::sync::Arc;

use gam::inference::row_measure::{MeasureProvenance, RowSamplingMeasure};
use gam::inference::row_metric::RowMetric;
use ndarray::{Array1, Array2};

/// Plant `n_common` quiet rows (small Fisher factor `quiet_amp`) and `n_rare`
/// loud rows (large factor `loud_amp`). `p = 1`, `rank = 1`, so a row's Fisher
/// mass is exactly `amp²` and the planted contrast is transparent.
fn plant_dataset(
    n_common: usize,
    n_rare: usize,
    quiet_amp: f64,
    loud_amp: f64,
) -> (RowMetric, Vec<usize>) {
    let n = n_common + n_rare;
    let mut u = Array2::<f64>::zeros((n, 1));
    let mut rare_rows = Vec::with_capacity(n_rare);
    for row in 0..n_common {
        u[[row, 0]] = quiet_amp;
    }
    for k in 0..n_rare {
        let row = n_common + k;
        u[[row, 0]] = loud_amp;
        rare_rows.push(row);
    }
    let metric = RowMetric::output_fisher(Arc::new(u), 1, 1).expect("output-fisher metric");
    (metric, rare_rows)
}

#[test]
fn enrichment_oversamples_rare_loud_feature_without_touching_loss() {
    // 500 common quiet rows, 5 rare loud rows. The loud rows each have 100x the
    // Fisher mass of a quiet row (amp 1.0 → mass 1; amp 10.0 → mass 100).
    let n_common = 500;
    let n_rare = 5;
    let quiet_amp = 1.0;
    let loud_amp = 10.0;
    let n = n_common + n_rare;

    let (metric, rare_rows) = plant_dataset(n_common, n_rare, quiet_amp, loud_amp);

    // ---- Baseline: uniform sampling under-represents the rare feature -------
    let uniform = RowSamplingMeasure::uniform(n);
    assert_eq!(uniform.provenance(), MeasureProvenance::Uniform);
    let batch = 200usize;
    let uniform_rep: f64 = uniform
        .expected_representation(batch)
        .iter()
        .enumerate()
        .filter(|(row, _)| rare_rows.contains(row))
        .map(|(_, &r)| r)
        .sum();
    // Population share of the rare rows is 5/505 ≈ 0.0099; in a batch of 200
    // that is < 2 expected hits across ALL five rare rows combined.
    assert!(
        uniform_rep < 2.0,
        "uniform must under-represent the rare feature: got {uniform_rep}"
    );

    // ---- Enrichment: Fisher-mass measure oversamples the rare-loud rows -----
    let enriched = RowSamplingMeasure::from_metric(&metric);
    assert!(
        enriched.is_enriched(),
        "factored metric must produce a real Fisher-mass measure"
    );

    // Total mass = 500 quiet * 1 + 5 loud * 100 = 1000. Rare share = 500/1000 =
    // 0.5. So the rare rows should collectively carry ~half the sampling mass.
    let enriched_rep: f64 = enriched
        .expected_representation(batch)
        .iter()
        .enumerate()
        .filter(|(row, _)| rare_rows.contains(row))
        .map(|(_, &r)| r)
        .sum();
    assert!(
        enriched_rep > 90.0,
        "enrichment must markedly oversample the rare feature: got {enriched_rep}"
    );

    // The lift over uniform is dramatic (≥ 45x here), which is the recall win.
    assert!(
        enriched_rep > 45.0 * uniform_rep.max(1e-9),
        "enrichment recall lift too small: enriched={enriched_rep} uniform={uniform_rep}"
    );

    // ---- Actually-drawn (deterministic) ordering confirms recall -----------
    // Uniform drawn batch: almost never lands on a rare row.
    let uniform_order = uniform.enrichment_order(batch, 1234);
    let uniform_hits = uniform_order
        .iter()
        .filter(|r| rare_rows.contains(r))
        .count();
    // Enriched drawn batch: lands on rare rows many times.
    let enriched_order = enriched.enrichment_order(batch, 1234);
    let enriched_hits = enriched_order
        .iter()
        .filter(|r| rare_rows.contains(r))
        .count();
    assert!(
        enriched_hits > 10 * uniform_hits.max(1),
        "drawn enrichment must visit the rare rows far more often: \
         enriched_hits={enriched_hits} uniform_hits={uniform_hits}"
    );
    // Systematic resampling guarantees each weight-≥1/batch row appears; every
    // rare row carries ~0.1 mass ≫ 1/200, so all five must be drawn.
    for &rr in &rare_rows {
        assert!(
            enriched_order.contains(&rr),
            "rare row {rr} must be discovered (drawn at least once) under enrichment"
        );
    }

    // ---- INVARIANT: the enrichment touches NO per-row loss -----------------
    // The criterion-facing residual square `quad_form(row, r)` must be exactly
    // what the metric reports independent of whether any RowSamplingMeasure was built.
    // We compute it from a fresh metric (no measure) and from the same metric
    // after a measure was produced, and require bit-for-bit equality.
    let (metric_no_measure, _) = plant_dataset(n_common, n_rare, quiet_amp, loud_amp);
    let probe = Array1::from(vec![0.37_f64]);
    for &row in &[0usize, 1, n_common, n_common + n_rare - 1] {
        let with_measure = metric.quad_form(row, probe.view());
        let without_measure = metric_no_measure.quad_form(row, probe.view());
        assert_eq!(
            with_measure, without_measure,
            "row {row}: producing a RowSamplingMeasure must not change any per-row loss"
        );
        // And the measure weight is NOT a factor in that loss: the weight is a
        // sampling probability, categorically separate from the residual square.
        let w = enriched.weights()[row];
        assert!(
            (0.0..=1.0).contains(&w),
            "row {row}: sampling weight {w} must be a probability, not a loss scale"
        );
    }
}

#[test]
fn no_harvest_is_todays_uniform_behavior() {
    // Euclidean metric (no Fisher factors harvested) ⇒ the measure is exactly
    // uniform: every row equal, no enrichment. This is the graceful-degradation
    // guarantee — absent harvest reproduces today's "look at every row equally".
    let metric = RowMetric::euclidean(64, 4).expect("euclidean metric");
    let measure = RowSamplingMeasure::from_metric(&metric);
    assert_eq!(measure.provenance(), MeasureProvenance::Uniform);
    assert!(!measure.is_enriched());
    let expect = 1.0 / 64.0;
    for &w in measure.weights() {
        assert!((w - expect).abs() < 1e-12);
    }
    // The drawn ordering is an even round-robin over all rows: every row appears
    // exactly once in a batch of 64.
    let order = measure.enrichment_order(64, 99);
    let mut seen = vec![0usize; 64];
    for &r in &order {
        seen[r] += 1;
    }
    for (row, &c) in seen.iter().enumerate() {
        assert_eq!(c, 1, "uniform draw must visit row {row} exactly once");
    }
}
