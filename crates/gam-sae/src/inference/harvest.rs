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

use gam_problem::{MetricProvenance, RowMetric};
use gam_solve::row_sampling_measure::{RowSamplingMeasure, per_row_fisher_mass};

/// The Fisher-bearing tier: which corpus rows carry factors, and the metric
/// over exactly those rows.
struct FisherTier {
    /// Corpus row indices that carry Fisher factors — strictly ascending.
    /// Tier row `t` of `metric` corresponds to corpus row `rows[t]`.
    rows: Vec<usize>,
    /// Per-tier-row inclusion probability of the design that picked `rows`
    /// (`1.0` everywhere for an exhaustive or deliberately unweighted tier).
    /// Used to Horvitz–Thompson-correct corpus-level estimates lifted from
    /// the tier, e.g. the total Fisher mass behind the lifted measure.
    inclusion: Vec<f64>,
    /// The metric over the tier rows (n_rows == rows.len()).
    metric: RowMetric,
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
    /// Tier 1 only: a corpus harvested without Fisher factors. Every metric
    /// consumer takes its no-harvest path; [`Self::corpus_measure`] is exactly
    /// uniform. Never an error: absence is a valid, first-class state.
    pub fn activations_only(n_rows: usize) -> Self {
        Self {
            n_rows,
            fisher: None,
        }
    }

    /// Attach a Fisher tier: `tier_rows` are the corpus rows that carry
    /// factors (strictly ascending, in range), `inclusion[t]` the design
    /// inclusion probability of `tier_rows[t]` (all `1.0` for an unweighted
    /// tier — see [`Self::with_unweighted_tier`]), and `metric` the
    /// [`RowMetric`] built over exactly those rows in that order.
    pub fn with_designed_tier(
        n_rows: usize,
        tier_rows: Vec<usize>,
        inclusion: Vec<f64>,
        metric: RowMetric,
    ) -> Result<Self, String> {
        if metric.n_rows() != tier_rows.len() {
            return Err(format!(
                "TieredHarvest: metric covers {} rows but the tier names {}",
                metric.n_rows(),
                tier_rows.len()
            ));
        }
        if inclusion.len() != tier_rows.len() {
            return Err(format!(
                "TieredHarvest: {} inclusion probabilities for {} tier rows",
                inclusion.len(),
                tier_rows.len()
            ));
        }
        for (t, &r) in tier_rows.iter().enumerate() {
            if r >= n_rows {
                return Err(format!(
                    "TieredHarvest: tier row {r} out of corpus range (n_rows = {n_rows})"
                ));
            }
            if t > 0 && tier_rows[t - 1] >= r {
                return Err(
                    "TieredHarvest: tier rows must be strictly ascending (sorted, deduplicated)"
                        .to_string(),
                );
            }
        }
        for (t, &p) in inclusion.iter().enumerate() {
            if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                return Err(format!(
                    "TieredHarvest: tier row {} has invalid inclusion probability {p}",
                    tier_rows[t]
                ));
            }
        }
        Ok(Self {
            n_rows,
            fisher: Some(FisherTier {
                rows: tier_rows,
                inclusion,
                metric,
            }),
        })
    }

    /// Convenience: a Fisher tier whose membership was not importance-designed
    /// (e.g. an exhaustive small-corpus harvest, where the tier IS the corpus,
    /// or a fixed audit slice). All inclusion probabilities are `1.0`, so
    /// lifted estimates apply no correction.
    pub fn with_unweighted_tier(
        n_rows: usize,
        tier_rows: Vec<usize>,
        metric: RowMetric,
    ) -> Result<Self, String> {
        let inclusion = vec![1.0; tier_rows.len()];
        Self::with_designed_tier(n_rows, tier_rows, inclusion, metric)
    }

    /// Total corpus rows (tier 1).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Whether a Fisher tier exists at all.
    pub fn has_fisher_tier(&self) -> bool {
        self.fisher.is_some()
    }

    /// Fraction of corpus rows carrying factors (`0.0` with no tier).
    pub fn coverage(&self) -> f64 {
        match (&self.fisher, self.n_rows) {
            (Some(t), n) if n > 0 => t.rows.len() as f64 / n as f64,
            _ => 0.0,
        }
    }

    /// The corpus rows of the Fisher tier (ascending), empty with no tier.
    pub fn tier_rows(&self) -> &[usize] {
        self.fisher.as_ref().map_or(&[], |t| &t.rows)
    }

    /// The tier metric — `None` when no Fisher tier exists. **Indexed by tier
    /// row**: row `t` of the returned metric is corpus row
    /// `self.tier_rows()[t]`. Consumers serving the gauge/lens roles iterate
    /// the tier, not the corpus; that is the whole point of the shape.
    pub fn tier_metric(&self) -> Option<&RowMetric> {
        self.fisher.as_ref().map(|t| &t.metric)
    }

    /// Provenance of the tier metric, `None` with no tier. A consumer that
    /// certifies "which inner product produced this report" (#980 Object 4)
    /// reads this together with [`Self::coverage`].
    pub fn tier_provenance(&self) -> Option<MetricProvenance> {
        self.fisher.as_ref().map(|t| t.metric.provenance())
    }

    /// Map a corpus row to its tier row, or `None` if the row carries no
    /// factors. O(log tier) binary search; never an error and never a
    /// fabricated factor — `None` IS the graceful-absence answer every metric
    /// consumer must accept.
    pub fn tier_row_for(&self, corpus_row: usize) -> Option<usize> {
        let tier = self.fisher.as_ref()?;
        tier.rows.binary_search(&corpus_row).ok()
    }

    /// Whether a specific corpus row carries Fisher factors.
    pub fn has_factors(&self, corpus_row: usize) -> bool {
        self.tier_row_for(corpus_row).is_some()
    }

    /// Lift the tier's Fisher masses to a full-corpus enrichment measure
    /// (role (c) of #980, served from the subsample as #987 prescribes).
    ///
    /// * Tier rows carry their own `tr(M_t)` mass, Horvitz–Thompson-corrected
    ///   by `1 / inclusion` so a measure-designed tier does not double-count
    ///   the very enrichment that designed it.
    /// * Corpus rows **outside** the tier carry the tier's mean corrected
    ///   mass — the honest "unobserved" imputation: they are neither zeroed
    ///   (which would starve un-harvested rows of all future attention,
    ///   freezing the design) nor boosted. When tier masses are flat this
    ///   collapses the whole measure to uniform, exactly the no-signal
    ///   degeneracy [`RowSamplingMeasure`] already normalizes to.
    /// * No tier ⇒ exactly [`RowSamplingMeasure::uniform`].
    ///
    /// The result obeys every [`RowSamplingMeasure`] invariant — discovery/seeding
    /// attention only, never a loss weight.
    pub fn corpus_measure(&self) -> RowSamplingMeasure {
        let Some(tier) = self.fisher.as_ref() else {
            return RowSamplingMeasure::uniform(self.n_rows);
        };
        if self.n_rows == 0 {
            return RowSamplingMeasure::uniform(0);
        }
        let tier_mass: Vec<f64> = per_row_fisher_mass(&tier.metric);
        let mut corrected = vec![0.0_f64; tier.rows.len()];
        let mut total = 0.0_f64;
        let mut usable = true;
        for (t, &m) in tier_mass.iter().enumerate() {
            if !m.is_finite() {
                usable = false;
                break;
            }
            let v = if m > 0.0 { m / tier.inclusion[t] } else { 0.0 };
            corrected[t] = v;
            total += v;
        }
        if !usable || !(total > 0.0) {
            return RowSamplingMeasure::uniform(self.n_rows);
        }
        let mean = total / tier.rows.len() as f64;
        let mut masses = vec![mean; self.n_rows];
        for (t, &r) in tier.rows.iter().enumerate() {
            masses[r] = corrected[t];
        }
        RowSamplingMeasure::from_masses(tier.metric.provenance(), masses)
    }

    /// Plan the **next** harvest's Fisher tier: a designed subsample of
    /// `budget` corpus rows drawn from this harvest's lifted measure
    /// (uniform on a first harvest with no tier — cold start is just the
    /// degenerate design). Returns the design (rows ascending + inclusion
    /// weights as `1/π` likelihood weights); the caller harvests factors for
    /// exactly those rows and builds the next [`TieredHarvest`] with
    /// `inclusion[t] = 1 / likelihood_weights[t]`.
    pub fn plan_next_tier(
        &self,
        budget: usize,
        seed: u64,
    ) -> gam_solve::row_sampling_measure::DesignedRowSample {
        self.corpus_measure().designed_subsample(budget, seed)
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
