//! Shape and memory planning for high-`K` dictionary score routing.
//!
//! Sparse SAE dictionary routers all have the same hot loop: score a minibatch
//! of `n_rows` residual rows against `n_items` candidate atoms/blocks, keep a
//! tiny online top-`s`, and never materialize the full `n_rows x n_items` score
//! matrix. This module owns the reusable admission and tile-size invariants for
//! that pattern. Domain crates still own their kernels and selection semantics.

/// Minimum `n_rows * n_items` score elements before a cold device route is worth
/// its launch and host/device transfer cost.
pub const DEFAULT_DICTIONARY_SCORE_MIN_ELEMS: usize = 1 << 20;

/// Maximum score elements per device launch. With `f32` scores this is 8 MiB,
/// matching the library row-chunk target and keeping peak score memory bounded
/// independent of dictionary width.
pub const DEFAULT_DICTIONARY_SCORE_TILE_ELEMS: usize =
    gam_runtime::resource::ResourcePolicy::default_library().row_chunk_target_bytes
        / std::mem::size_of::<f32>();

/// Device admission and tile geometry for one minibatch-by-dictionary score
/// route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DictionaryScoreRoutePlan {
    /// Minibatch rows scored together.
    pub n_rows: usize,
    /// Candidate atoms/blocks scored for each row.
    pub n_items: usize,
    /// Dot-product width for one score.
    pub feature_dim: usize,
    /// Minimum `n_rows * n_items` elements required for device admission.
    pub device_min_score_elems: usize,
    /// Maximum `n_rows * tile_items` score elements held by one device launch.
    pub max_tile_score_elems: usize,
    /// Candidate items per launch tile.
    pub tile_items: usize,
    /// Number of candidate tiles covering `0..n_items`.
    pub tile_count: usize,
    /// True when the total route work is large enough to use the device.
    pub device_admitted: bool,
    /// Peak score-block bytes for a full tile.
    pub peak_score_bytes: usize,
    /// Lower-bound arithmetic for dispatch diagnostics: one multiply and one add
    /// per `(row, item, feature)` score term.
    pub dot_flops_lower_bound: u128,
}

impl DictionaryScoreRoutePlan {
    /// Build a plan with explicit thresholds. The function is pure and
    /// allocation-free so call sites can test routing decisions without a CUDA
    /// runtime.
    #[must_use]
    pub fn with_limits(
        n_rows: usize,
        n_items: usize,
        feature_dim: usize,
        device_min_score_elems: usize,
        max_tile_score_elems: usize,
    ) -> Self {
        let total_score_elems = n_rows.saturating_mul(n_items);
        let nondegenerate = n_rows > 0 && n_items > 0 && feature_dim > 0;
        let tile_items = if n_rows == 0 || n_items == 0 {
            0
        } else {
            (max_tile_score_elems / n_rows).clamp(1, n_items)
        };
        let tile_count = if tile_items == 0 {
            0
        } else {
            n_items.div_ceil(tile_items)
        };
        let peak_tile_items = tile_items.min(n_items);
        let peak_score_elems = n_rows.saturating_mul(peak_tile_items);
        let dot_flops_lower_bound = 2u128
            .saturating_mul(n_rows as u128)
            .saturating_mul(n_items as u128)
            .saturating_mul(feature_dim as u128);

        Self {
            n_rows,
            n_items,
            feature_dim,
            device_min_score_elems,
            max_tile_score_elems,
            tile_items,
            tile_count,
            device_admitted: nondegenerate && total_score_elems >= device_min_score_elems,
            peak_score_bytes: peak_score_elems.saturating_mul(std::mem::size_of::<f32>()),
            dot_flops_lower_bound,
        }
    }

    /// Build a plan with the library defaults used by sparse dictionary routers.
    #[must_use]
    pub fn default_for_shape(n_rows: usize, n_items: usize, feature_dim: usize) -> Self {
        Self::with_limits(
            n_rows,
            n_items,
            feature_dim,
            DEFAULT_DICTIONARY_SCORE_MIN_ELEMS,
            DEFAULT_DICTIONARY_SCORE_TILE_ELEMS,
        )
    }

    /// True when the plan covers no route work.
    #[must_use]
    pub const fn is_degenerate(self) -> bool {
        self.n_rows == 0 || self.n_items == 0 || self.feature_dim == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_k32k_shape_is_admitted_and_memory_bounded() {
        let plan = DictionaryScoreRoutePlan::default_for_shape(256, 32_768, 64);
        assert!(plan.device_admitted);
        assert_eq!(plan.tile_items, 8_192);
        assert_eq!(plan.tile_count, 4);
        assert_eq!(plan.peak_score_bytes, 8 * 1024 * 1024);
        assert_eq!(
            plan.dot_flops_lower_bound,
            2u128 * 256u128 * 32_768u128 * 64u128
        );
    }

    #[test]
    fn peak_score_memory_does_not_grow_with_dictionary_width() {
        let small = DictionaryScoreRoutePlan::default_for_shape(512, 4_096, 48);
        let large = DictionaryScoreRoutePlan::default_for_shape(512, 131_072, 48);
        assert_eq!(small.tile_items, large.tile_items);
        assert_eq!(small.peak_score_bytes, large.peak_score_bytes);
        assert!(large.tile_count > small.tile_count);
    }

    #[test]
    fn sub_floor_and_degenerate_shapes_stay_on_host() {
        let tiny = DictionaryScoreRoutePlan::default_for_shape(16, 1024, 64);
        assert!(!tiny.device_admitted);
        assert_eq!(tiny.tile_count, 1);

        for plan in [
            DictionaryScoreRoutePlan::default_for_shape(0, 1024, 64),
            DictionaryScoreRoutePlan::default_for_shape(16, 0, 64),
            DictionaryScoreRoutePlan::default_for_shape(16, 1024, 0),
        ] {
            assert!(plan.is_degenerate());
            assert!(!plan.device_admitted);
            assert_eq!(plan.tile_items, 0);
            assert_eq!(plan.tile_count, 0);
            assert_eq!(plan.peak_score_bytes, 0);
        }
    }

    #[test]
    fn tiny_tile_budget_still_makes_forward_progress() {
        let plan = DictionaryScoreRoutePlan::with_limits(512, 1000, 32, 1, 7);
        assert_eq!(plan.tile_items, 1);
        assert_eq!(plan.tile_count, 1000);
        assert_eq!(plan.peak_score_bytes, 512 * std::mem::size_of::<f32>());
    }
}
