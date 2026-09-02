//! Two-tier Fisher harvest: Fisher-on-subsample **by design** (#987, amending
//! the #980 harvest contract).
//!
//! # The economics
//!
//! Harvesting per-row output-Fisher factors costs ~`rank` backward probes per
//! token. On 10⁶ rows that is noise; on a frontier corpus (10¹⁰ rows) it is
//! ruinous — and unnecessary, because the roles the metric serves (#980:
//! gauge units, the two-lens report, Fisher-mass enrichment) are *estimation*
//! roles that need far fewer rows than reconstruction does. Reconstruction
//! needs every activation row; the metric needs a **designed subsample**.
//!
//! So the harvest contract gains a two-tier corpus shape, and this module is
//! that shape as a type:
//!
//! * **Tier 1 (all rows):** activations. Reconstruction runs unweighted over
//!   them — which the amended #980 contract already guarantees
//!   ([`RowMetric::whitens_likelihood`] is `false` for
//!   [`MetricProvenance::OutputFisher`]), so withholding factors from a row
//!   cannot change its loss term even in principle.
//! * **Tier 2 (designed subsample):** activations **plus** Fisher factors.
//!   Gauge and lens are computed on this tier; the enrichment measure is
//!   estimated from it and lifted to the full corpus.
//!
//! # Graceful absence is structural, not defensive
//!
//! Every consumer of the metric must already degrade on rows without factors;
//! [`TieredHarvest`] makes that degradation a matter of *where the data lives*
//! rather than runtime branching:
//!
//! * no Fisher tier at all ⇒ [`TieredHarvest::activations_only`]: the metric
//!   accessor returns `None`, the lifted measure is exactly uniform, and every
//!   downstream consumer takes its existing no-harvest path — bit-for-bit
//!   today's behavior;
//! * a Fisher tier ⇒ the tier's [`RowMetric`] (indexed by **tier row**, with
//!   the corpus↔tier mapping owned here) serves the gauge/lens/measure roles,
//!   and any corpus row outside the tier reports "no factors" (`None`), never
//!   an error and never a fabricated identity block.
//!
//! The designed subsample itself comes from
//! [`RowSamplingMeasure::designed_subsample`] (uniform on the first harvest; measure-
//! driven re-designs once a previous tier exists), so tier membership carries
//! honest inclusion weights wherever an *estimate over the corpus* is lifted
//! from the tier — the same #973 honesty discipline, applied to the metric's
//! estimation roles instead of the likelihood.

/// The Fisher-bearing tier: which corpus rows carry factors, and the metric
/// over exactly those rows.
struct FisherTier {
    /// Corpus row indices that carry Fisher factors — strictly ascending.
    /// Tier row `t` of `metric` corresponds to corpus row `rows[t]`.
    rows: Vec<usize>,
}

/// A corpus with the #987 two-tier shape: activations everywhere, Fisher
/// factors on a designed subsample (possibly absent altogether).
///
/// This object owns the corpus↔tier index mapping and the graceful-absence
/// semantics; it deliberately does **not** own activations (those stream
/// through [`crate::corpus`]) — it is the metric side of the
/// harvest, keyed by the same stable row ids.
pub struct TieredHarvest {
    /// Total corpus rows (tier 1).
    n_rows: usize,
    fisher: Option<FisherTier>,
}

impl TieredHarvest {

    /// Total corpus rows (tier 1).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Fraction of corpus rows carrying factors (`0.0` with no tier).
    pub fn coverage(&self) -> f64 {
        match (&self.fisher, self.n_rows) {
            (Some(t), n) if n > 0 => t.rows.len() as f64 / n as f64,
            _ => 0.0,
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_solve::row_sampling_measure::MeasureProvenance;
    use ndarray::Array2;
    use std::sync::Arc;

    fn tier_metric(masses: &[f64]) -> RowMetric {
        // p = 1, rank = 1: factor u ⇒ tr(M) = u².
        let n = masses.len();
        let mut u = Array2::<f64>::zeros((n, 1));
        for (i, &m) in masses.iter().enumerate() {
            u[[i, 0]] = m.sqrt();
        }
        RowMetric::output_fisher(Arc::new(u), 1, 1).expect("tier metric")
    }

    #[test]
    fn activations_only_degrades_everywhere() {
        let h = TieredHarvest::activations_only(10);
        assert!(!h.has_fisher_tier());
        assert_eq!(h.coverage(), 0.0);
        assert!(h.tier_metric().is_none());
        assert!(h.tier_provenance().is_none());
        assert!(!h.has_factors(3));
        let m = h.corpus_measure();
        assert_eq!(m.provenance(), MeasureProvenance::Uniform);
        assert_eq!(m.n_rows(), 10);
    }

    #[test]
    fn tier_mapping_and_coverage() {
        let metric = tier_metric(&[1.0, 4.0, 1.0]);
        let h = TieredHarvest::with_unweighted_tier(10, vec![2, 5, 9], metric).expect("harvest");
        assert!(h.has_fisher_tier());
        assert!((h.coverage() - 0.3).abs() < 1e-12);
        assert_eq!(h.tier_row_for(5), Some(1));
        assert_eq!(h.tier_row_for(4), None);
        assert!(h.has_factors(9));
        assert!(!h.has_factors(0));
        assert_eq!(
            h.tier_provenance(),
            Some(h.tier_metric().unwrap().provenance())
        );
    }

    #[test]
    fn lifted_measure_imputes_mean_mass_off_tier() {
        // Tier rows 2 and 5 with masses 1 and 9 ⇒ mean 5. Off-tier rows carry
        // mass 5, so the loud tier row outranks them and the quiet tier row
        // ranks below them — observed signal moves attention both ways.
        let metric = tier_metric(&[1.0, 9.0]);
        let h = TieredHarvest::with_unweighted_tier(4, vec![2, 3], metric).expect("harvest");
        let m = h.corpus_measure();
        assert!(m.is_enriched());
        let w = m.weights();
        // masses: [5, 5, 1, 9] / 20.
        assert!((w[0] - 0.25).abs() < 1e-12);
        assert!((w[2] - 0.05).abs() < 1e-12);
        assert!((w[3] - 0.45).abs() < 1e-12);
    }

    #[test]
    fn inclusion_correction_undoes_design_bias() {
        // Two tier rows with the SAME underlying mass, but row B was twice as
        // likely to be designed in (π = 1.0 vs 0.5). HT correction must give
        // the π = 0.5 row twice the lifted mass, restoring exchangeability of
        // the corpus-level estimate.
        let metric = tier_metric(&[4.0, 4.0]);
        let h = TieredHarvest::with_designed_tier(2, vec![0, 1], vec![0.5, 1.0], metric)
            .expect("harvest");
        let m = h.corpus_measure();
        let w = m.weights();
        assert!(
            (w[0] - 2.0 * w[1]).abs() < 1e-12,
            "HT lift must double the half-inclusion row: {w:?}"
        );
    }

    #[test]
    fn flat_tier_collapses_to_uniform_attention() {
        let metric = tier_metric(&[2.0, 2.0]);
        let h = TieredHarvest::with_unweighted_tier(6, vec![1, 4], metric).expect("harvest");
        let m = h.corpus_measure();
        let w = m.weights();
        for &x in w {
            assert!((x - 1.0 / 6.0).abs() < 1e-12, "flat tier must lift uniform");
        }
    }

    #[test]
    fn validation_rejects_malformed_tiers() {
        let metric = tier_metric(&[1.0, 2.0]);
        // Unsorted.
        assert!(
            TieredHarvest::with_unweighted_tier(5, vec![3, 1], tier_metric(&[1.0, 2.0])).is_err()
        );
        // Out of range.
        assert!(
            TieredHarvest::with_unweighted_tier(3, vec![1, 3], tier_metric(&[1.0, 2.0])).is_err()
        );
        // Metric/tier length mismatch.
        assert!(TieredHarvest::with_unweighted_tier(5, vec![0, 1, 2], metric).is_err());
        // Bad inclusion probability.
        assert!(
            TieredHarvest::with_designed_tier(
                5,
                vec![0, 1],
                vec![0.0, 1.0],
                tier_metric(&[1.0, 2.0])
            )
            .is_err()
        );
    }

    #[test]
    fn plan_next_tier_cold_start_is_uniform_design() {
        let h = TieredHarvest::activations_only(50);
        let plan = h.plan_next_tier(10, 7);
        assert_eq!(plan.provenance, MeasureProvenance::Uniform);
        assert_eq!(plan.len(), 10);
        // Re-planning with a previous loud tier steers the design toward the
        // loud row.
        let metric = tier_metric(&[1.0, 100.0]);
        let h2 = TieredHarvest::with_unweighted_tier(50, vec![10, 20], metric).expect("harvest");
        let plan2 = h2.plan_next_tier(10, 7);
        assert!(
            plan2.rows.contains(&20),
            "the loud previously-harvested row must be re-designed in"
        );
    }
}
