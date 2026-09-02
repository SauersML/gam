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

