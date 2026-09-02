//! Designed corpus target collection — the #991 bridge from a streaming
//! [`CorpusRowSource`] to the in-memory row set + honesty weights the SAE term
//! fits on.
//!
//! # The architecture this realizes
//!
//! At frontier scale the fit never sees the whole corpus: it sees a **designed
//! sample** whose inclusion weights ride into the likelihood so the criterion
//! stays unbiased (#987 / #973). That makes "fit the corpus" a two-step
//! pipeline with a bounded memory footprint by construction:
//!
//! 1. **Design** — a [`RowSamplingMeasure`] over the corpus (uniform on a first
//!    harvest; [`TieredHarvest::corpus_measure`]-driven once Fisher factors
//!    exist) picks `budget` rows via
//!    [`RowSamplingMeasure::designed_subsample`] (deterministic, seeded, honest `1/π`
//!    weights).
//! 2. **Collect** — one deterministic streaming pass over the source
//!    materializes exactly those rows (the only dense `f64` block the fit ever
//!    holds: `budget × p`, not `N × p`), aligned with their weights and global
//!    `row_id`s.
//!
//! The term consumes the result as `(target, set_row_loss_weights)`; the
//! weights enter the objective through the term's single `√w` honesty seam.
//!
//! # Exactness degeneracy (the bit-identity contract)
//!
//! `budget ≥ corpus rows` (always the case below
//! [`designed_sampling_mandatory`]'s threshold unless a caller narrows it)
//! collects **every** row in stream order with weight exactly `1.0` — and the
//! term stores all-equal weights as `None`, so a shard-backed full-budget fit
//! is **bit-for-bit** the in-memory fit of the same rows. Selectivity is then
//! purely a budget decision, not a code path: drivers call this
//! unconditionally and let [`auto_designed_budget`] decide.

use ndarray::Array2;

use gam_solve::row_sampling_measure::MeasureProvenance;

/// Default designed-sample budget once [`designed_sampling_mandatory`] fires.
/// Auto-derived policy, not a knob: 2·10⁶ rows is comfortably in-memory at any
/// realistic activation width (`2e6 × 4096 × 8B ≈ 64 GiB` is the extreme; at
/// GPT-2-small widths it is ~6 GiB), large enough that designed-sample SEs on
/// shared structure are far below fit noise, and small enough that an outer
/// iteration's full pass over the *sample* is minutes, not days.
pub const DESIGNED_SAMPLE_DEFAULT_BUDGET_ROWS: usize = 2_000_000;

/// The collected designed row set: the dense fit target plus everything needed
/// to keep the fit honest and traceable back to the corpus.
#[derive(Debug, Clone)]
pub struct DesignedCorpusTarget {
    /// `(n_selected × p)` upcast activations of exactly the designed rows, in
    /// ascending global row order.
    pub target: Array2<f64>,
    /// Global corpus `row_id` of each target row (ascending). These are the
    /// keys for warm-state reuse ([`super::warm_state`]) and for aligning a
    /// [`TieredHarvest`] Fisher tier with the fitted rows.
    pub row_ids: Vec<u64>,
    /// Per-selected-row Horvitz–Thompson likelihood weight `1/π`, aligned with
    /// `target` rows. Hand to `SaeManifoldTerm::set_row_loss_weights` (which
    /// mean-normalizes; an exact full pass yields all-`1.0` here and the
    /// unweighted path there).
    pub likelihood_weights: Vec<f64>,
    /// Provenance of the measure that shaped the design.
    pub provenance: MeasureProvenance,
    /// Total corpus rows the design was drawn from.
    pub corpus_rows: u64,
}

impl DesignedCorpusTarget {
    /// Number of collected rows.
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

}

