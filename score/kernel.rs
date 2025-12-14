// ========================================================================================
//
//                      The kernel: A pure SIMD execution engine
//
// ========================================================================================
//
// This module contains the final, innermost loop of the compute engine. It is designed
// for maximum, predictable throughput and is 100% allocation-free in the hot path.
// It functions as a "Virtual Machine" that executes a pre-compiled plan, containing
// zero scientific logic, branches, or decisions.

use std::simd::{Simd, cmp::SimdPartialEq, f32x8, u8x8};

// --- Type Aliases for Readability ---
// These types are part of the public API of the kernel.
pub type SimdVec = f32x8;
pub const LANE_COUNT: usize = SimdVec::LEN;

// ========================================================================================
//                            Public API & type definitions
// ========================================================================================

/// A validated, type-safe, zero-cost view over a slice representing a padded,
/// interleaved matrix.
///
/// This struct's constructor guarantees that an instance can only be created if its
/// dimensions and padding are coherent. This makes an invalidly-dimensioned or
/// non-padded matrix an unrepresentable state for the kernel, preventing panics
/// and memory errors in the hot loops. It is used for both the aligned weights
/// and the correction constants matrices.
pub struct PaddedInterleavedWeights<'a> {
    slice: &'a [f32],
    num_scores: usize,
    stride: usize,
}

impl<'a> PaddedInterleavedWeights<'a> {
    /// Creates a new, validated `PaddedInterleavedWeights` view over a slice.
    /// This is the sole entry point for creating this type. It performs a single,
    /// upfront check to ensure the slice length matches the provided dimensions
    /// and the implied padding.
    #[inline]
    pub fn new(slice: &'a [f32], num_rows: usize, num_scores: usize) -> Result<Self, &'static str> {
        // The stride is the width of a single row's data, rounded up to the
        // nearest multiple of the SIMD vector width. This padding is the
        // key to enabling branch-free, "no scalar fallback" SIMD.
        let stride = num_scores.div_ceil(LANE_COUNT) * LANE_COUNT;
        if slice.len() != num_rows * stride {
            return Err(
                "Mismatched matrix data: slice.len() does not equal num_rows * calculated_stride",
            );
        }
        Ok(Self {
            slice,
            num_scores,
            stride,
        })
    }

    /// Returns the original number of scores (K) this matrix was created with.
    #[inline(always)]
    pub fn num_scores(&self) -> usize {
        self.num_scores
    }

    /// Fetches the i-th SIMD vector of data for a given matrix row.
    ///
    /// # Safety
    /// The caller MUST guarantee that `row_idx` is a valid row index for this matrix
    /// and `lane_idx < self.stride / LANE_COUNT`. This contract is upheld by
    /// `accumulate_scores_for_person`, whose loops are correctly bounded.
    #[inline(always)]
    unsafe fn get_simd_lane_unchecked(&self, row_idx: usize, lane_idx: usize) -> SimdVec {
        // The offset calculation uses the pre-computed `stride` to correctly
        // jump between the start of each variant's padded data block.
        let offset = (row_idx * self.stride) + (lane_idx * LANE_COUNT);
        // SAFETY: The `unsafe fn` contract guarantees the offset is in-bounds. The
        // struct's constructor guarantees the data is padded, so reading a full
        // `LANE_COUNT` of floats is always safe.
        unsafe { SimdVec::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT)) }
    }
}

/// A validated, type-safe, zero-cost view over a slice representing a padded,
/// interleaved matrix of `u8` flags.
pub struct PaddedInterleavedFlags<'a> {
    slice: &'a [u8],
    stride: usize,
}

impl<'a> PaddedInterleavedFlags<'a> {
    /// Creates a new, validated `PaddedInterleavedFlags` view over a slice.
    #[inline]
    pub fn new(slice: &'a [u8], num_rows: usize, num_scores: usize) -> Result<Self, &'static str> {
        let stride = num_scores.div_ceil(LANE_COUNT) * LANE_COUNT;
        if slice.len() != num_rows * stride {
            return Err(
                "Mismatched flag matrix data: slice.len() does not equal num_rows * calculated_stride",
            );
        }
        Ok(Self { slice, stride })
    }

    /// Fetches the i-th SIMD vector of flags for a given matrix row.
    ///
    /// # Safety
    /// The caller MUST guarantee that `row_idx` is a valid row index and `lane_idx` is valid.
    #[inline(always)]
    unsafe fn get_simd_lane_unchecked(&self, row_idx: usize, lane_idx: usize) -> Simd<u8, 8> {
        let offset = (row_idx * self.stride) + (lane_idx * LANE_COUNT);
        unsafe { Simd::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT)) }
    }
}

// ========================================================================================
//                              The kernel implementation
// ========================================================================================
/// This constant defines the maximum number of score columns the kernel can handle.
/// It is used to size the kernel's internal accumulator buffer on the stack.
const MAX_KERNEL_ACCUMULATOR_LANES: usize = 100_usize.div_ceil(LANE_COUNT);

/// Calculates the score *adjustments* for a single person over a mini-batch of variants.
/// This kernel is self-contained: it creates its own accumulators, performs the
/// computation, and returns the result. This pure-function approach enhances safety.
/// The internal loop structure is designed to maximize instruction-level parallelism
/// by processing all score columns for a variant before moving to the next.
///
/// # Safety
/// The caller must uphold the contract that all indices in `g1_indices` and `g2_indices`
/// are valid row indices for the `weights` and `flip_flags` matrices, which represent
/// a view of a single mini-batch.
#[inline]
pub fn accumulate_adjustments_for_person(
    weights: &PaddedInterleavedWeights,
    flip_flags: &PaddedInterleavedFlags,
    g1_indices: &[usize],
    g2_indices: &[usize],
) -> [SimdVec; MAX_KERNEL_ACCUMULATOR_LANES] {
    let num_scores = weights.num_scores();
    let num_accumulator_lanes = num_scores.div_ceil(LANE_COUNT);

    // A stack-allocated buffer for this person's mini-batch score adjustments.
    let mut accumulator_buffer = [SimdVec::splat(0.0); MAX_KERNEL_ACCUMULATOR_LANES];

    // --- Loop 1: Dosage=1 Adjustments ---
    let two = SimdVec::splat(2.0);
    for &matrix_row_idx in g1_indices {
        // This inner loop over score columns is the same performant structure as the original kernel.
        for i in 0..num_accumulator_lanes {
            unsafe {
                let weights_vec = weights.get_simd_lane_unchecked(matrix_row_idx, i);
                let flip_mask = flip_flags
                    .get_simd_lane_unchecked(matrix_row_idx, i)
                    .simd_eq(u8x8::splat(1));

                let adj = flip_mask.cast().select(-weights_vec, weights_vec);
                *accumulator_buffer.get_unchecked_mut(i) += adj;
            }
        }
    }

    // --- Loop 2: Dosage=2 Adjustments ---
    for &matrix_row_idx in g2_indices {
        for i in 0..num_accumulator_lanes {
            unsafe {
                let weights_vec = weights.get_simd_lane_unchecked(matrix_row_idx, i);
                let flip_mask = flip_flags
                    .get_simd_lane_unchecked(matrix_row_idx, i)
                    .simd_eq(u8x8::splat(1));

                let adj = flip_mask
                    .cast()
                    .select(-two * weights_vec, two * weights_vec);
                *accumulator_buffer.get_unchecked_mut(i) += adj;
            }
        }
    }

    accumulator_buffer
}
