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
/// mask). It equals `n_full / |mask|` when all rows share a uniform inclusion
/// probability (the caller-supplied-mask case represented by
/// [`OuterScoreSubsample::from_uniform_inclusion_mask`]); it can drift from
/// that value under the stratified builder's rare-stratum boost. It is not the
/// per-row scaling factor — consumers must read `rows[i].weight` for HT
/// correctness.
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
    /// Wrap a precomputed mask sampled with a uniform inclusion probability,
    /// assigning each selected row the inverse-inclusion weight `n_full / m`.
    /// The caller is responsible for sortedness and uniqueness;
    /// `build_outer_score_subsample` remains the stratified per-row HT builder.
    pub fn from_uniform_inclusion_mask(mask: Vec<usize>, n_full: usize, seed: u64) -> Self {
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

/// Deterministic row-block tiling constant for the parallel reduction paths.
///
/// All cross-row summations chunk the rows into `ARROW_ROW_CHUNK`-sized tiles
/// and reduce the per-tile partials in tile-index order on the caller thread,
/// so the floating-point reduction tree is fixed across Rayon worker counts and
/// work-stealing decisions. Consumers that require deterministic associativity
/// must keep their tiling a multiple of this constant.
pub const ARROW_ROW_CHUNK: usize = 256;

/// Number of `ARROW_ROW_CHUNK`-sized tiles covering `n_rows`.
#[inline]
pub fn arrow_row_chunk_count(n_rows: usize) -> usize {
    if n_rows == 0 {
        0
    } else {
        (n_rows - 1) / ARROW_ROW_CHUNK + 1
    }
}

/// Row selection for an outer-loop evaluation: either the full data (`All`) or
/// a Horvitz–Thompson [`WeightedOuterRow`] subsample.
///
/// `All` walks rows `0..n_total` with unit weight; `Subsample` walks the stored
/// rows applying each row's inverse-inclusion scale `1/π_i`, so any partial sum
/// `Σ_i w_i · f(row_i)` is an unbiased estimator of the corresponding full-data
/// sum `Σ_{i=1..n_full} f(row_i)`. Inner-PIRLS and final-covariance passes
/// always run with `All`; only outer score / gradient hot loops consume a
/// non-`All` variant.
///
/// Lives in this lower layer (below `families`/`terms`) so the row-kernel
/// consumers and the term hot-paths can name it without the `Subsample` field
/// reaching up into `solver` (#1135). The family-specific constructor
/// (`families::row_kernel::RowSet::from_options`, which reads
/// `custom_family::BlockwiseFitOptions`) stays in `families` as an extension
/// `impl` block.
#[derive(Clone)]
pub enum RowSet {
    All,
    Subsample {
        rows: Arc<Vec<WeightedOuterRow>>,
        n_full: usize,
    },
}

impl RowSet {
    /// Parallel fold-reduce over the row set. `init` produces a fresh
    /// accumulator, `fold` is the per-row update, `reduce` combines two
    /// accumulators.
    ///
    /// Returns the reduced result. Both branches process fixed-size row chunks
    /// in parallel, then combine the chunk accumulators in chunk-index order on
    /// the caller thread. The resulting floating-point reduction tree is fixed
    /// across Rayon worker counts and work-stealing decisions.
    #[inline]
    pub fn par_reduce_fold<T, I, F, R>(&self, n_total: usize, init: I, fold: F, reduce: R) -> T
    where
        T: Send,
        I: Fn() -> T + Send + Sync,
        F: Fn(T, usize, f64) -> T + Send + Sync,
        R: Fn(T, T) -> T + Send + Sync,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        use rayon::slice::ParallelSlice;
        match self {
            Self::All => {
                let chunk_accumulators: Vec<T> = (0..arrow_row_chunk_count(n_total))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * ARROW_ROW_CHUNK;
                        let end = (start + ARROW_ROW_CHUNK).min(n_total);
                        let mut acc = init();
                        for i in start..end {
                            acc = fold(acc, i, 1.0);
                        }
                        acc
                    })
                    .collect();
                let mut total = init();
                for acc in chunk_accumulators {
                    total = reduce(total, acc);
                }
                total
            }
            Self::Subsample { rows, .. } => {
                let chunk_accumulators: Vec<T> = rows
                    .par_chunks(ARROW_ROW_CHUNK)
                    .map(|chunk| {
                        let mut acc = init();
                        for r in chunk {
                            acc = fold(acc, r.index, r.weight);
                        }
                        acc
                    })
                    .collect();
                let mut total = init();
                for acc in chunk_accumulators {
                    total = reduce(total, acc);
                }
                total
            }
        }
    }

    /// Parallel try-fold over fixed-size row chunks, followed by deterministic
    /// chunk-index-order reduction on the caller thread.
    #[inline]
    pub fn par_try_reduce_fold<T, E, I, F, R>(
        &self,
        n_total: usize,
        init: I,
        fold: F,
        reduce: R,
    ) -> Result<T, E>
    where
        T: Send,
        E: Send,
        I: Fn() -> T + Send + Sync,
        F: Fn(T, usize, f64) -> Result<T, E> + Send + Sync,
        R: Fn(T, T) -> Result<T, E> + Send + Sync,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        use rayon::slice::ParallelSlice;
        match self {
            Self::All => {
                let chunk_accumulators: Vec<Result<T, E>> = (0..arrow_row_chunk_count(n_total))
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * ARROW_ROW_CHUNK;
                        let end = (start + ARROW_ROW_CHUNK).min(n_total);
                        let mut acc = init();
                        for i in start..end {
                            acc = fold(acc, i, 1.0)?;
                        }
                        Ok(acc)
                    })
                    .collect();
                let mut total = init();
                for acc in chunk_accumulators {
                    total = reduce(total, acc?)?;
                }
                Ok(total)
            }
            Self::Subsample { rows, .. } => {
                let chunk_accumulators: Vec<Result<T, E>> = rows
                    .par_chunks(ARROW_ROW_CHUNK)
                    .map(|chunk| {
                        let mut acc = init();
                        for r in chunk {
                            acc = fold(acc, r.index, r.weight)?;
                        }
                        Ok(acc)
                    })
                    .collect();
                let mut total = init();
                for acc in chunk_accumulators {
                    total = reduce(total, acc?)?;
                }
                Ok(total)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── arrow_row_chunk_count ─────────────────────────────────────────────────

    #[test]
    fn chunk_count_zero_rows_is_zero() {
        assert_eq!(arrow_row_chunk_count(0), 0);
    }

    #[test]
    fn chunk_count_one_row_is_one() {
        assert_eq!(arrow_row_chunk_count(1), 1);
    }

    #[test]
    fn chunk_count_exact_multiple() {
        assert_eq!(arrow_row_chunk_count(ARROW_ROW_CHUNK), 1);
        assert_eq!(arrow_row_chunk_count(ARROW_ROW_CHUNK * 3), 3);
    }

    #[test]
    fn chunk_count_just_over_boundary() {
        assert_eq!(arrow_row_chunk_count(ARROW_ROW_CHUNK + 1), 2);
    }

    // ── OuterScoreSubsample::from_uniform_inclusion_mask ─────────────────────

    #[test]
    fn uniform_mask_weight_scale_is_n_full_over_m() {
        let mask = vec![0usize, 2, 4];
        let s = OuterScoreSubsample::from_uniform_inclusion_mask(mask, 6, 0);
        assert_eq!(s.len(), 3);
        assert!((s.weight_scale - 2.0).abs() < 1e-14);
        assert!(s.rows.iter().all(|r| (r.weight - 2.0).abs() < 1e-14));
    }

    #[test]
    fn uniform_mask_empty_has_weight_scale_one() {
        let s = OuterScoreSubsample::from_uniform_inclusion_mask(vec![], 10, 0);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.weight_scale, 1.0);
    }

    // ── OuterScoreSubsample::from_weighted_rows ───────────────────────────────

    #[test]
    fn weighted_rows_sorts_and_deduplicates() {
        let rows = vec![
            WeightedOuterRow {
                index: 3,
                weight: 2.0,
                stratum: 0,
            },
            WeightedOuterRow {
                index: 1,
                weight: 1.0,
                stratum: 0,
            },
            WeightedOuterRow {
                index: 3,
                weight: 2.0,
                stratum: 0,
            }, // duplicate
        ];
        let s = OuterScoreSubsample::from_weighted_rows(rows, 10, 42);
        assert_eq!(s.len(), 2);
        assert_eq!(s.mask[0], 1);
        assert_eq!(s.mask[1], 3);
    }

    #[test]
    fn weighted_rows_weight_scale_is_average_weight() {
        let rows = vec![
            WeightedOuterRow {
                index: 0,
                weight: 1.0,
                stratum: 0,
            },
            WeightedOuterRow {
                index: 1,
                weight: 3.0,
                stratum: 0,
            },
        ];
        let s = OuterScoreSubsample::from_weighted_rows(rows, 10, 0);
        assert!((s.weight_scale - 2.0).abs() < 1e-14);
    }

    // ── OuterScoreSubsample::has_variable_weights ─────────────────────────────

    #[test]
    fn has_variable_weights_false_for_uniform() {
        let s = OuterScoreSubsample::with_uniform_weight(vec![0, 1, 2], 3, 0, 1.5);
        assert!(!s.has_variable_weights());
    }

    #[test]
    fn has_variable_weights_true_for_mixed() {
        let rows = vec![
            WeightedOuterRow {
                index: 0,
                weight: 1.0,
                stratum: 0,
            },
            WeightedOuterRow {
                index: 1,
                weight: 2.0,
                stratum: 0,
            },
        ];
        let s = OuterScoreSubsample::from_weighted_rows(rows, 5, 0);
        assert!(s.has_variable_weights());
    }

    // ── RowSet::par_reduce_fold ───────────────────────────────────────────────

    #[test]
    fn row_set_all_sums_indices_zero_to_n() {
        let rs = RowSet::All;
        let sum: f64 =
            rs.par_reduce_fold(5, || 0.0_f64, |acc, i, w| acc + i as f64 * w, |a, b| a + b);
        // 1*0 + 1*1 + 1*2 + 1*3 + 1*4 = 10
        assert!((sum - 10.0).abs() < 1e-14);
    }

    #[test]
    fn row_set_subsample_applies_per_row_weight() {
        let rows = Arc::new(vec![
            WeightedOuterRow {
                index: 2,
                weight: 3.0,
                stratum: 0,
            },
            WeightedOuterRow {
                index: 5,
                weight: 2.0,
                stratum: 0,
            },
        ]);
        let rs = RowSet::Subsample { rows, n_full: 10 };
        let sum: f64 =
            rs.par_reduce_fold(10, || 0.0_f64, |acc, i, w| acc + w * i as f64, |a, b| a + b);
        // 3.0 * 2 + 2.0 * 5 = 6 + 10 = 16
        assert!((sum - 16.0).abs() < 1e-14);
    }
}
