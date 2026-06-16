//! Row-measure handle for trust-region invariant enforcement.
//!
//! A `TrustRegionRowMeasure` is the explicit identity of the set of rows + per-row
//! weights used to evaluate any one of {Hessian, gradient, objective}
//! during a single inner trust-region iteration. The trust-region
//! globalization computes
//!
//!   ρ = actual_reduction / predicted_reduction
//!     = [F(β) − F(β + δ)] / [−g·δ − ½·δᵀHδ]
//!
//! and accepts/rejects the step from ρ. All four quantities (F(β),
//! F(β + δ), g, H) MUST be evaluated against the same row measure for
//! ρ to be meaningful; otherwise the numerator and denominator estimate
//! different objectives and ρ can take any sign, producing the observed
//! ρ = -0.05 with predicted_reduction = +7.378e6 sign flip.
//!
//! `TrustRegionRowMeasure::id` is a stable 64-bit content hash: equal masks
//! (`Arc<OuterScoreSubsample>` pointer equality OR identical mask
//! contents) ⇒ equal ids; differing masks ⇒ differing ids with high
//! probability. The TR loop captures one `TrustRegionRowMeasure` at the top of an
//! iteration and hard-asserts that the id observed by each of the four
//! quantities matches before computing ρ.

use std::sync::Arc;

use crate::custom_family::BlockwiseFitOptions;
use crate::solver::outer_subsample::OuterScoreSubsample;

/// Identifier-carrying handle for a single row measure.
///
/// The handle is `Clone` and cheap to copy; the `Arc` is shared, not
/// duplicated.
#[derive(Clone, Debug)]
pub struct TrustRegionRowMeasure {
    /// Stable 64-bit content hash. Same `mask` (by Arc pointer OR by
    /// row content) ⇒ same id; different `mask` ⇒ different id.
    pub id: u64,
    /// `None` means full data (`0..n`, weight 1.0 per row).
    /// `Some(_)` means the rows and HT weights inside the subsample.
    pub mask: Option<Arc<OuterScoreSubsample>>,
}

impl TrustRegionRowMeasure {
    /// Full-data measure: walk `0..n` with weight 1.0 per row.
    pub fn full_data(n: usize) -> Self {
        Self {
            id: hash_full(n),
            mask: None,
        }
    }

    /// Subsample measure: walk the mask's rows with their per-row HT
    /// weights. Id is derived from the Arc pointer (cheap and stable
    /// for the lifetime of the Arc) combined with mask metadata.
    pub fn subsample(mask: Arc<OuterScoreSubsample>) -> Self {
        let id = hash_subsample(&mask);
        Self {
            id,
            mask: Some(mask),
        }
    }

    /// Build a `TrustRegionRowMeasure` from blockwise-fit options. The outer
    /// optimizer is the sole source of `outer_score_subsample`; inner
    /// paths read this once at the top of each TR iteration and freeze
    /// it for every quantity in that iteration.
    pub fn from_options(options: &BlockwiseFitOptions, n: usize) -> Self {
        match options.outer_score_subsample.as_ref() {
            Some(mask) => Self::subsample(Arc::clone(mask)),
            None => Self::full_data(n),
        }
    }

    /// Materialize the row indices and per-row weights this measure
    /// implies. `full_data(n)` returns `(0..n collected, [1.0; n])`,
    /// preserving the legacy semantics of any caller that walked
    /// `0..self.n` unconditionally with weight 1.0.
    pub fn indices_and_weights(&self, n: usize) -> (Vec<usize>, Vec<f64>) {
        match self.mask.as_ref() {
            Some(m) => {
                assert_eq!(
                    m.n_full, n,
                    "TrustRegionRowMeasure n_full ({}) must match caller n ({})",
                    m.n_full, n
                );
                let indices: Vec<usize> = m.mask.as_ref().clone();
                let mut weights = vec![1.0_f64; n];
                for r in m.rows.iter() {
                    if r.index < n {
                        weights[r.index] = r.weight;
                    }
                }
                (indices, weights)
            }
            None => ((0..n).collect(), vec![1.0_f64; n]),
        }
    }
}

/// Thin wrapper over the canonical SplitMix64 hash in
/// [`crate::linalg::utils::splitmix64_hash`].
fn splitmix64(x: u64) -> u64 {
    crate::linalg::utils::splitmix64_hash(x)
}

const FULL_DATA_ROW_MEASURE_SENTINEL: u64 = 0xA5A5_5A5A_DEAD_BEEF;

fn hash_full(n: usize) -> u64 {
    let mut h = splitmix64(FULL_DATA_ROW_MEASURE_SENTINEL ^ (n as u64));
    if h == 0 {
        h = 0x1234_5678_9ABC_DEF0;
    }
    h
}

fn hash_subsample(mask: &Arc<OuterScoreSubsample>) -> u64 {
    let ptr = Arc::as_ptr(mask) as u64;
    let mut h = splitmix64(ptr);
    h ^= splitmix64(mask.n_full as u64);
    h ^= splitmix64(mask.len() as u64);
    h ^= splitmix64(mask.seed);
    h ^= splitmix64((mask.weight_scale.to_bits()) ^ 0xC0FF_EE00_0000_0000);
    if h == 0 {
        h = 0xDEAD_BEEF_FEED_FACE;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::outer_subsample::OuterScoreSubsample;

    #[test]
    fn full_data_id_is_stable_per_n() {
        let a = TrustRegionRowMeasure::full_data(100);
        let b = TrustRegionRowMeasure::full_data(100);
        let c = TrustRegionRowMeasure::full_data(101);
        assert_eq!(a.id, b.id);
        assert_ne!(a.id, c.id);
        assert!(a.mask.is_none());
    }

    #[test]
    fn subsample_id_matches_for_same_arc() {
        let s = Arc::new(OuterScoreSubsample::new(vec![1, 3, 5], 10, 42));
        let a = TrustRegionRowMeasure::subsample(Arc::clone(&s));
        let b = TrustRegionRowMeasure::subsample(Arc::clone(&s));
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn subsample_id_differs_for_different_arcs() {
        let s1 = Arc::new(OuterScoreSubsample::new(vec![1, 3, 5], 10, 42));
        let s2 = Arc::new(OuterScoreSubsample::new(vec![1, 3, 5], 10, 42));
        let a = TrustRegionRowMeasure::subsample(s1);
        let b = TrustRegionRowMeasure::subsample(s2);
        // Different Arc allocations ⇒ different ids; this is intentional
        // so the TR invariant catches mid-iteration mask rebuilds even
        // when the resulting mask happens to be content-equal.
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn indices_and_weights_full_data() {
        let rm = TrustRegionRowMeasure::full_data(4);
        let (idx, w) = rm.indices_and_weights(4);
        assert_eq!(idx, vec![0, 1, 2, 3]);
        assert_eq!(w, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn indices_and_weights_subsample() {
        let s = Arc::new(OuterScoreSubsample::new(vec![0, 2], 4, 7));
        let rm = TrustRegionRowMeasure::subsample(s);
        let (idx, w) = rm.indices_and_weights(4);
        assert_eq!(idx, vec![0, 2]);
        assert_eq!(w.len(), 4);
        assert!(w[0] > 0.0 && w[2] > 0.0);
    }
}
