//! Outer-loop row subsampling primitive shared across the solver and the
//! family-specific outer-score evaluators.
//!
//! [`OuterScoreSubsample`] and its per-row [`WeightedOuterRow`] are the
//! Horvitz–Thompson row subsample consumed on outer-loop hot paths. They live
//! in the solver layer (below `families`) so that both the solver's row-measure
//! machinery and the family outer-score builders can depend on them downward,
//! without `solver` reaching up into `families`. The stratified *builders* that
//! construct these (`build_outer_score_subsample`, `auto_outer_score_subsample`)
//! remain in `families::marginal_slope_shared`, since they depend on
//! family-specific fit options; they import this type downward.

use std::sync::Arc;

/// Stratified row index subsample shared across outer-loop evaluations.
///
/// `mask` is sorted, deduplicated, and never empty in practice (enforced by
/// `build_outer_score_subsample`).
///
/// Per-row inverse-inclusion weights `w_i = N_h / k_h` (where `h` is the row's
/// stratum) are stored alongside the mask in `rows`. The Horvitz–Thompson
/// estimator for any linear-in-row functional T = Σ_i f_i is
///   T̂ = Σ_{i ∈ mask} w_i · f_i,
/// which is unbiased even when per-stratum sampling fractions differ
/// (the `ceil(k * N_h / n).max(1)` rule in the stratified builder makes
/// rare strata oversample relative to the bulk, so a single global rescale
/// `n_full / |mask|` is biased in those strata).
///
/// `weight_scale` is retained as a *diagnostic* (mean of `w_i` across the
/// mask). It equals the legacy `n_full / |mask|` when all rows share a
/// uniform weight (the common case for caller-supplied masks via
/// [`OuterScoreSubsample::new`]); it can drift from that value under the
/// stratified builder's rare-stratum boost. It is not the per-row scaling
/// factor — consumers must read `rows[i].weight` for HT correctness.
///
/// # Horvitz–Thompson contract
///
/// Per-row weight `rows[i].weight = 1 / π_i`, where `π_i` is the
/// inclusion probability of row `i` under the stratified sampler. Any
/// outer-only score/gradient routine that consumes this subsample must
/// form `Σ_{i ∈ mask} w_i · f_i` so the resulting estimator is unbiased:
///
/// ```text
///   E[ score_subsample ]  =  score_full.
/// ```
///
/// The following families consume this subsample on their outer-loop hot
/// paths: Gaussian-LS, Binomial-LS, the Wiggle variants, CTN, and
/// Survival-LS. Each routes the `rows[i].weight` factor through its
/// per-row accumulator (gradient, Hessian-action, trace probes).
///
/// # Convergence warning
///
/// Subsampled gradients are noisy by construction. The outer driver must
/// **never** declare convergence on a subsampled gradient — near
/// convergence it switches back to the full-data score so that the KKT
/// stopping test sees the unbiased, low-variance signal. New consumers
/// adding subsampled paths must preserve this invariant.
#[derive(Debug, Clone)]
pub struct OuterScoreSubsample {
    pub mask: Arc<Vec<usize>>,
    pub rows: Arc<Vec<WeightedOuterRow>>,
    pub n_full: usize,
    pub weight_scale: f64,
    pub seed: u64,
}

impl OuterScoreSubsample {
    /// Wrap a precomputed mask with the legacy uniform `n_full / m` weight
    /// per row. The caller is responsible for sortedness and uniqueness;
    /// `build_outer_score_subsample` is the canonical (per-stratum HT)
    /// builder.
    pub fn new(mask: Vec<usize>, n_full: usize, seed: u64) -> Self {
        let m = mask.len();
        let w = if m == 0 {
            1.0
        } else {
            n_full as f64 / m as f64
        };
        Self::with_uniform_weight(mask, n_full, seed, w)
    }

    /// Wrap a precomputed mask with an explicit uniform per-row weight.
    /// Useful for tests that need the unrescaled (`weight = 1.0`) sum over a
    /// custom mask, and for callers that already know the desired
    /// rescaling factor and don't want the constructor to derive it from
    /// `n_full / |mask|`.
    pub fn with_uniform_weight(mask: Vec<usize>, n_full: usize, seed: u64, weight: f64) -> Self {
        let rows: Vec<WeightedOuterRow> = mask
            .iter()
            .map(|&index| WeightedOuterRow {
                index,
                weight,
                stratum: 0,
            })
            .collect();
        let weight_scale = if rows.is_empty() { 1.0 } else { weight };
        Self {
            mask: Arc::new(mask),
            rows: Arc::new(rows),
            n_full,
            weight_scale,
            seed,
        }
    }

    /// Wrap a vector of `(index, weight, stratum)` triples. The mask is
    /// derived as the sorted/dedup'd index list. Used by the stratified
    /// builder to install per-row HT weights.
    pub fn from_weighted_rows(mut rows: Vec<WeightedOuterRow>, n_full: usize, seed: u64) -> Self {
        rows.sort_by_key(|r| r.index);
        rows.dedup_by_key(|r| r.index);
        let mask: Vec<usize> = rows.iter().map(|r| r.index).collect();
        let weight_scale = if rows.is_empty() {
            1.0
        } else {
            rows.iter().map(|r| r.weight).sum::<f64>() / rows.len() as f64
        };
        Self {
            mask: Arc::new(mask),
            rows: Arc::new(rows),
            n_full,
            weight_scale,
            seed,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// True when at least two retained rows have different per-row weights.
    /// Consumers that previously applied a single post-sum scalar must
    /// switch to per-row weighting whenever this returns true.
    pub fn has_variable_weights(&self) -> bool {
        let mut iter = self.rows.iter();
        let Some(first) = iter.next() else {
            return false;
        };
        iter.any(|r| (r.weight - first.weight).abs() > 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedOuterRow {
    pub index: usize,
    pub weight: f64,
    /// Stratum identifier the row was drawn from. Pure diagnostic — consumers
    /// must use `weight` for any aggregation.
    pub stratum: u32,
}
