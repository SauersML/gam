// ========================================================================================
//
//               A tiled, cache-aware, contention-free compute engine
//
// ========================================================================================
//
// This module contains the synchronous, CPU-bound core of the compute pipeline. It is
// designed to be called from a higher-level asynchronous orchestrator within a
// `spawn_blocking` context. Its sole responsibility is to take a raw, variant-major
// chunk of genotype data, pivot it into a person-major tile, generate a sparse
// index of non-zero work, and dispatch it to the kernel. It performs ZERO
// scientific logic or reconciliation. We don't support over 100 scores.
// It will panic / overflow if we try.

use crate::score::kernel;
use crate::score::types::{
    EffectAlleleDosage, OriginalPersonIndex, PersonSubset, PreparationResult,
    ReconciledVariantIndex,
};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::error::Error;
use std::marker::PhantomData;
use std::simd::{Simd, cmp::SimdPartialEq, num::SimdUint};

// --- State Markers (Zero-Sized Types) ---
pub struct Ready;
pub struct Counted;
pub struct Allocated;
pub struct Filled;

// --- Data Structs For Each State ---
pub struct ReadyData<'a> {
    tile: &'a [EffectAlleleDosage],
    num_people: usize,
    mini_batch_size: usize,
    variant_mini_batch_start: usize,
}

pub struct CountedData {
    num_people: usize,
    person_counts: Vec<(u32, u32)>,
    missing_events: Vec<(usize, usize)>,
}

pub struct AllocatedData<'a> {
    // We need to pass the tile reference through for the final fill pass.
    tile: &'a [EffectAlleleDosage],
    variant_mini_batch_start: usize,
    mini_batch_size: usize,
    num_people: usize,
    person_counts: Vec<(u32, u32)>,
    missing_events: Vec<(usize, usize)>,
    g1_offsets: Vec<usize>,
    g2_offsets: Vec<usize>,
    all_g1_indices: Vec<usize>,
    all_g2_indices: Vec<usize>,
}

pub struct FilledData {
    pub person_counts: Vec<(u32, u32)>,
    pub missing_events: Vec<(usize, usize)>,
    pub g1_offsets: Vec<usize>,
    pub g2_offsets: Vec<usize>,
    pub all_g1_indices: Vec<usize>,
    pub all_g2_indices: Vec<usize>,
}

// --- The Main Builder Struct ---
pub struct SparseIndexBuilder<'a, State> {
    state: State,
    marker: PhantomData<&'a ()>,
}

// ========================================================================================
//                             Sparse index builder (typestate)
// ========================================================================================

impl<'a> SparseIndexBuilder<'a, ReadyData<'a>> {
    /// Creates a new builder in the initial `Ready` state.
    pub fn new(data: ReadyData<'a>) -> Self {
        Self {
            state: data,
            marker: PhantomData,
        }
    }

    /// PASS 1: Parallel Count.
    pub fn count_dosages(self) -> SparseIndexBuilder<'a, CountedData> {
        let variants_in_chunk = self.state.tile.len() / self.state.num_people;

        let (person_counts, nested_missing_events): (Vec<(u32, u32)>, Vec<Vec<(usize, usize)>>) =
            (0..self.state.num_people)
                .into_par_iter()
                .map(|person_idx| {
                    let mut g1_count = 0;
                    let mut g2_count = 0;
                    let mut local_missing_events = Vec::new();

                    let genotype_tile_row_offset = person_idx * variants_in_chunk;
                    let mini_batch_genotype_start =
                        genotype_tile_row_offset + self.state.variant_mini_batch_start;
                    let genotype_row = &self.state.tile[mini_batch_genotype_start
                        ..mini_batch_genotype_start + self.state.mini_batch_size];

                    for (i, &dosage) in genotype_row.iter().enumerate() {
                        match dosage.0 {
                            1 => g1_count += 1,
                            2 => g2_count += 1,
                            3 => {
                                let variant_idx_in_chunk = self.state.variant_mini_batch_start + i;
                                local_missing_events.push((person_idx, variant_idx_in_chunk));
                            }
                            _ => (),
                        }
                    }
                    ((g1_count, g2_count), local_missing_events)
                })
                .unzip();

        let missing_events = nested_missing_events.into_iter().flatten().collect();

        SparseIndexBuilder {
            state: CountedData {
                num_people: self.state.num_people,
                person_counts,
                missing_events,
            },
            marker: PhantomData,
        }
    }
}

impl<'a> SparseIndexBuilder<'a, CountedData> {
    /// PASS 2: Serial Allocation & Offset Calculation.
    pub fn allocate_and_prepare(
        self,
        tile: &'a [EffectAlleleDosage],
        variant_mini_batch_start: usize,
        mini_batch_size: usize,
    ) -> SparseIndexBuilder<'a, AllocatedData<'a>> {
        let total_g1: usize = self
            .state
            .person_counts
            .iter()
            .map(|(g1, _)| *g1 as usize)
            .sum();
        let total_g2: usize = self
            .state
            .person_counts
            .iter()
            .map(|(_, g2)| *g2 as usize)
            .sum();

        let mut all_g1_indices = Vec::with_capacity(total_g1);
        let mut all_g2_indices = Vec::with_capacity(total_g2);

        // This is safe because we are creating uninitialized Vecs and will fill them completely.
        unsafe {
            all_g1_indices.set_len(total_g1);
            all_g2_indices.set_len(total_g2);
        }

        let mut g1_offsets = Vec::with_capacity(self.state.num_people);
        let mut g2_offsets = Vec::with_capacity(self.state.num_people);
        let mut g1_running_total = 0;
        let mut g2_running_total = 0;

        for (g1_count, g2_count) in &self.state.person_counts {
            g1_offsets.push(g1_running_total);
            g2_offsets.push(g2_running_total);
            g1_running_total += *g1_count as usize;
            g2_running_total += *g2_count as usize;
        }

        SparseIndexBuilder {
            state: AllocatedData {
                tile,
                variant_mini_batch_start,
                mini_batch_size,
                num_people: self.state.num_people,
                person_counts: self.state.person_counts,
                missing_events: self.state.missing_events,
                g1_offsets,
                g2_offsets,
                all_g1_indices,
                all_g2_indices,
            },
            marker: PhantomData,
        }
    }
}

impl<'a> SparseIndexBuilder<'a, AllocatedData<'a>> {
    /// PASS 3: Parallel Fill. This is where the single `unsafe` block lives.
    pub fn parallel_fill(mut self) -> SparseIndexBuilder<'a, FilledData> {
        let num_people = self.state.num_people;
        let variants_in_chunk = self.state.tile.len() / num_people;

        // Alternative pointer passing: Cast to usize to ensure Send + Sync for the captured variable.
        let g1_ptr_addr = self.state.all_g1_indices.as_mut_ptr() as usize;
        let g2_ptr_addr = self.state.all_g2_indices.as_mut_ptr() as usize;

        // These variables are all thread-safe and can be captured by the closure.
        let tile_ref = self.state.tile;
        let variant_mini_batch_start_val = self.state.variant_mini_batch_start;
        let mini_batch_size_val = self.state.mini_batch_size;
        let person_counts_ref = &self.state.person_counts;
        let g1_offsets_ref = &self.state.g1_offsets;
        let g2_offsets_ref = &self.state.g2_offsets;

        // The `move` closure will now capture `g1_ptr_wrapper` and `g2_ptr_wrapper`,
        // which are Send+Sync, thus satisfying the compiler.
        (0..num_people).into_par_iter().for_each(move |person_idx| {
            let g1_offset = g1_offsets_ref[person_idx];
            let g1_count = person_counts_ref[person_idx].0 as usize;
            let g2_offset = g2_offsets_ref[person_idx];
            let g2_count = person_counts_ref[person_idx].1 as usize;

            // Inside the thread, cast usize back to raw pointer and use it.
            // This is our promise to the compiler that this is safe because the underlying Vec lives.
            let g1_ptr = g1_ptr_addr as *mut usize;
            let g2_ptr = g2_ptr_addr as *mut usize;

            let person_g1_slice =
                unsafe { std::slice::from_raw_parts_mut(g1_ptr.add(g1_offset), g1_count) };
            let person_g2_slice =
                unsafe { std::slice::from_raw_parts_mut(g2_ptr.add(g2_offset), g2_count) };

            let mut g1_written = 0;
            let mut g2_written = 0;

            let genotype_tile_row_offset = person_idx * variants_in_chunk;
            let mini_batch_genotype_start = genotype_tile_row_offset + variant_mini_batch_start_val;
            let genotype_row = &tile_ref
                [mini_batch_genotype_start..mini_batch_genotype_start + mini_batch_size_val];

            for (i, &dosage) in genotype_row.iter().enumerate() {
                match dosage.0 {
                    1 => {
                        person_g1_slice[g1_written] = i;
                        g1_written += 1;
                    }
                    2 => {
                        person_g2_slice[g2_written] = i;
                        g2_written += 1;
                    }
                    _ => (),
                }
            }
        });

        SparseIndexBuilder {
            state: FilledData {
                person_counts: self.state.person_counts,
                missing_events: self.state.missing_events,
                g1_offsets: self.state.g1_offsets,
                g2_offsets: self.state.g2_offsets,
                all_g1_indices: self.state.all_g1_indices,
                all_g2_indices: self.state.all_g2_indices,
            },
            marker: PhantomData,
        }
    }
}

impl<'a> SparseIndexBuilder<'a, FilledData> {
    // Final safe getters
    pub fn g1_slice_for_person(&self, person_idx: usize) -> &[usize] {
        let start = self.state.g1_offsets[person_idx];
        let count = self.state.person_counts[person_idx].0 as usize;
        &self.state.all_g1_indices[start..start + count]
    }
    pub fn g2_slice_for_person(&self, person_idx: usize) -> &[usize] {
        let start = self.state.g2_offsets[person_idx];
        let count = self.state.person_counts[person_idx].1 as usize;
        &self.state.all_g2_indices[start..start + count]
    }
    pub fn missing_events(&self) -> &[(usize, usize)] {
        &self.state.missing_events
    }
    pub fn into_inner(self) -> FilledData {
        self.state
    }
}

// --- SIMD & Engine Tuning Parameters ---
const SIMD_LANES: usize = 8;
type U64xN = Simd<u64, SIMD_LANES>;
type U8xN = Simd<u8, SIMD_LANES>;

/// The number of individuals to process in a single on-the-fly pivoted tile.
/// This value is tuned to ensure the tile fits comfortably within the L3 cache.
const PERSON_BLOCK_SIZE: usize = 4096;

/// The number of variants to process in a single call to the compute kernel. This value
/// controls the frequency of flushing the `f32` accumulators to the `f64` master
/// buffer, which is the primary mechanism for guaranteeing numerical accuracy.
const KERNEL_MINI_BATCH_SIZE: usize = 256;

// ========================================================================================
//                                   Public API
// ========================================================================================

/// Processes one dense, pre-filtered batch of variant-major data using the person-major
/// (pivot) path. This function is called from a parallel context (e.g., Rayon's
/// `par_bridge`), and all of its internal logic is sequential to prevent nested
/// parallelism deadlocks and maximize cache efficiency.
pub fn run_person_major_path(
    variant_major_data: &[u8],
    weights_for_batch: &[f32],
    flips_for_batch: &[u8],
    reconciled_variant_indices_for_batch: &[ReconciledVariantIndex],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f64],
    partial_missing_counts_out: &mut [u32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- Entry Point Validation ---
    let expected_len = prep_result.num_people_to_score * prep_result.score_names.len();
    if partial_scores_out.len() != expected_len {
        return Err(Box::from(format!(
            "Mismatched scores buffer: expected length {}, got {}",
            expected_len,
            partial_scores_out.len()
        )));
    }

    // === Sequential compute within a single parallel task ===
    // This main loop is intentionally sequential. The outer pipeline (in pipeline.rs)
    // is responsible for parallelism by calling this function for different batches
    // on different threads. This avoids thread pool exhaustion and is highly
    // cache-friendly, as each thread works on its own disjoint data blocks.
    let num_scores = prep_result.score_names.len();
    let items_per_block = PERSON_BLOCK_SIZE * num_scores;

    let person_indices_to_process = match &prep_result.person_subset {
        PersonSubset::All => None, // Will generate ranges on the fly.
        PersonSubset::Indices(indices) => Some(indices),
    };

    partial_scores_out
        .chunks_mut(items_per_block)
        .zip(partial_missing_counts_out.chunks_mut(items_per_block))
        .enumerate()
        .for_each(
            |(block_idx, (block_scores_out, block_missing_counts_out))| {
                let person_output_start_idx = block_idx * PERSON_BLOCK_SIZE;
                let person_output_end_idx = (person_output_start_idx + PERSON_BLOCK_SIZE)
                    .min(prep_result.num_people_to_score);

                if person_output_start_idx >= person_output_end_idx {
                    return;
                }

                // This temporary Vec holds the original FAM indices for only the people
                // in the current block. This is a small, fast allocation.
                let person_indices_in_block: Vec<OriginalPersonIndex> =
                    if let Some(indices) = person_indices_to_process {
                        // Case 1: Using a '--keep' file. We take a slice of the pre-sorted
                        // FAM indices that correspond to our current output block.
                        indices[person_output_start_idx..person_output_end_idx]
                            .iter()
                            .map(|&fam_idx| OriginalPersonIndex(fam_idx))
                            .collect()
                    } else {
                        // Case 2: Scoring all people. The FAM index is the same as the
                        // output index. We can generate this range on the fly.
                        (person_output_start_idx as u32..person_output_end_idx as u32)
                            .map(OriginalPersonIndex)
                            .collect()
                    };

                process_block(
                    &person_indices_in_block,
                    prep_result,
                    variant_major_data,
                    weights_for_batch,
                    flips_for_batch,
                    reconciled_variant_indices_for_batch,
                    block_scores_out,
                    block_missing_counts_out,
                    tile_pool,
                );
            },
        );

    Ok(())
}

// ========================================================================================
//                            Public entry points
// ========================================================================================

#[cfg_attr(not(feature = "no-inline-profiling"), inline)]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
pub fn process_tile<'a>(
    tile: &'a [EffectAlleleDosage],
    prep_result: &'a PreparationResult,
    weights_for_batch: &'a [f32],
    flips_for_batch: &'a [u8],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
) {
    process_tile_impl(
        tile,
        prep_result,
        weights_for_batch,
        flips_for_batch,
        reconciled_variant_indices_for_batch,
        block_scores_out,
        block_missing_counts_out,
    );
}

// ========================================================================================
//                            Private implementation
// ========================================================================================

/// Processes a single block of individuals.
#[cfg_attr(not(feature = "no-inline-profiling"), inline)]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn process_block<'a>(
    person_indices_in_block: &'a [OriginalPersonIndex],
    prep_result: &'a PreparationResult,
    variant_major_data: &'a [u8],
    weights: &'a [f32],
    flips: &'a [u8],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
    tile_pool: &'a ArrayQueue<Vec<EffectAlleleDosage>>,
) {
    let variants_in_chunk = reconciled_variant_indices_for_batch.len();
    let tile_size = person_indices_in_block.len() * variants_in_chunk;

    let mut tile = tile_pool.pop().unwrap_or_default();
    tile.clear();
    tile.resize(tile_size, EffectAlleleDosage::default());

    // This pivot function has a single, clear responsibility.
    pivot_tile(
        variant_major_data,
        person_indices_in_block,
        &mut tile,
        prep_result,
    );

    process_tile_impl(
        &tile,
        prep_result,
        weights,
        flips,
        reconciled_variant_indices_for_batch,
        block_scores_out,
        block_missing_counts_out,
    );

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// Accumulates a SIMD lane of `f32` adjustments into a `f64` score slice.
///
/// This function handles both full 8-element lanes and partial tail lanes.
/// After benchmarking, the unrolled scalar loop has been
/// empirically proven to be the fastest implementation for this specific
/// read-modify-write task on the target hardware.
#[inline(always)]
fn accumulate_simd_lane(
    scores_out_slice: &mut [f64],
    adjustments_f32x8: Simd<f32, 8>,
    scores_offset: usize,
    num_scores: usize,
) {
    let adj = adjustments_f32x8.to_array();

    // This is the fast path for full 8-element chunks.
    if scores_offset + SIMD_LANES <= num_scores {
        // Unrolled scalar loop - empirically fastest implementation
        scores_out_slice[scores_offset] += adj[0] as f64;
        scores_out_slice[scores_offset + 1] += adj[1] as f64;
        scores_out_slice[scores_offset + 2] += adj[2] as f64;
        scores_out_slice[scores_offset + 3] += adj[3] as f64;
        scores_out_slice[scores_offset + 4] += adj[4] as f64;
        scores_out_slice[scores_offset + 5] += adj[5] as f64;
        scores_out_slice[scores_offset + 6] += adj[6] as f64;
        scores_out_slice[scores_offset + 7] += adj[7] as f64;
    } else {
        // This is the scalar fallback for the tail end of the data (e.g., if there
        // are 1-7 elements left).
        let end = num_scores;
        for j in 0..(end - scores_offset) {
            scores_out_slice[scores_offset + j] += adj[j] as f64;
        }
    }
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel after
/// calculating a baseline score and pre-computing sparse indices.
#[cfg_attr(not(feature = "no-inline-profiling"), inline)]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
pub(crate) fn process_tile_impl<'a>(
    tile: &'a [EffectAlleleDosage],
    prep_result: &'a PreparationResult,
    weights_for_batch: &'a [f32],
    flips_for_batch: &'a [u8],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
) {
    let variants_in_chunk = reconciled_variant_indices_for_batch.len();
    let num_scores = prep_result.score_names.len();
    let num_people_in_block = if variants_in_chunk > 0 {
        tile.len() / variants_in_chunk
    } else {
        0
    };

    if num_people_in_block == 0 {
        return;
    }

    let stride = prep_result.stride();
    let num_accumulator_lanes = num_scores.div_ceil(SIMD_LANES);

    for variant_mini_batch_start in (0..variants_in_chunk).step_by(KERNEL_MINI_BATCH_SIZE) {
        let mini_batch_size =
            (variants_in_chunk - variant_mini_batch_start).min(KERNEL_MINI_BATCH_SIZE);
        if mini_batch_size == 0 {
            continue;
        }

        // --- Execute the Type-Safe State Machine ---
        let ready_data = ReadyData {
            tile,
            num_people: num_people_in_block,
            mini_batch_size,
            variant_mini_batch_start,
        };

        let filled_builder = SparseIndexBuilder::new(ready_data)
            .count_dosages()
            .allocate_and_prepare(tile, variant_mini_batch_start, mini_batch_size)
            .parallel_fill();

        // --- Create Kernel Input Views ---
        let matrix_slice_start = variant_mini_batch_start * stride;
        let matrix_slice_end = matrix_slice_start + (mini_batch_size * stride);

        // SAFETY: The loop structure and mini-batch calculations ensure that the
        // `matrix_slice_start..matrix_slice_end` range is always within the bounds
        // of `weights_for_batch` and `flips_for_batch`. Using `get_unchecked`
        // bypasses the compiler's bounds checks, which is critical for performance
        // in this hot loop.
        let (weights_chunk, flip_flags_chunk) = unsafe {
            (
                weights_for_batch.get_unchecked(matrix_slice_start..matrix_slice_end),
                flips_for_batch.get_unchecked(matrix_slice_start..matrix_slice_end),
            )
        };

        // SAFETY: The logic of the mini-batch slicing guarantees the chunks passed to
        // the `new` functions have the correct dimensions. Therefore, these operations
        // will never fail, and using `unwrap_unchecked` is safe and removes the
        // performance-killing panic branch from the hot loop.
        let weights = unsafe {
            kernel::PaddedInterleavedWeights::new(weights_chunk, mini_batch_size, num_scores)
                .unwrap_unchecked()
        };
        let flip_flags = unsafe {
            kernel::PaddedInterleavedFlags::new(flip_flags_chunk, mini_batch_size, num_scores)
                .unwrap_unchecked()
        };

        // --- Dispatch to Kernel and Accumulate Results ---
        block_scores_out
            .chunks_exact_mut(num_scores)
            .enumerate()
            .for_each(|(person_idx, scores_out_slice)| {
                let g1_slice = filled_builder.g1_slice_for_person(person_idx);
                let g2_slice = filled_builder.g2_slice_for_person(person_idx);

                let kernel_result_buffer = kernel::accumulate_adjustments_for_person(
                    &weights,
                    &flip_flags,
                    g1_slice,
                    g2_slice,
                );

                for i in 0..num_accumulator_lanes {
                    let scores_offset = i * SIMD_LANES;
                    let adjustments_f32x8 = kernel_result_buffer[i];
                    accumulate_simd_lane(
                        scores_out_slice,
                        adjustments_f32x8,
                        scores_offset,
                        num_scores,
                    );
                }
            });

        // --- Process Deferred Missing Events ---
        for &(person_idx, variant_idx_in_chunk) in filled_builder.missing_events() {
            let global_matrix_row_idx =
                reconciled_variant_indices_for_batch[variant_idx_in_chunk].0 as usize;
            let scores_for_this_variant = &prep_result.variant_to_scores_map[global_matrix_row_idx];
            let weight_row_offset = variant_idx_in_chunk * stride;
            let weight_row = &weights_for_batch[weight_row_offset..weight_row_offset + num_scores];
            let flip_row = &flips_for_batch[weight_row_offset..weight_row_offset + num_scores];
            let person_scores_slice =
                &mut block_scores_out[person_idx * num_scores..(person_idx + 1) * num_scores];
            let person_missing_counts_slice = &mut block_missing_counts_out
                [person_idx * num_scores..(person_idx + 1) * num_scores];

            for &score_idx in scores_for_this_variant {
                let score_col = score_idx.0;
                unsafe {
                    *person_missing_counts_slice.get_unchecked_mut(score_col) += 1;
                    if flip_row[score_col] == 1 {
                        *person_scores_slice.get_unchecked_mut(score_col) -=
                            2.0 * (weight_row[score_col] as f64);
                    }
                }
            }
        }
    }
}

/// A cache-friendly, SIMD-accelerated pivot function using an 8x8 in-register transpose.
/// This function's sole purpose is to pivot raw genotype dosages from the variant-major
/// .bed layout to a person-major tile layout. It performs no reconciliation.
#[cfg_attr(not(feature = "no-inline-profiling"), inline)]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn pivot_tile(
    variant_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
) {
    let num_people_in_block = person_indices_in_block.len();
    let bytes_per_variant = prep_result.bytes_per_variant;
    let variants_in_chunk = if num_people_in_block > 0 {
        tile.len() / num_people_in_block
    } else {
        0
    };

    // Maps a desired sequential variant index (0-7) to its physical source location within
    // the shuffled vector produced by the `transpose_8x8_u8` function. This is used
    // to "un-shuffle" the data into the correct sequential order in the tile.
    const UNSHUFFLE_MAP: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

    for person_chunk_start in (0..num_people_in_block).step_by(SIMD_LANES) {
        let remaining_people = num_people_in_block - person_chunk_start;
        let present_lanes = remaining_people.min(SIMD_LANES);

        let person_indices = U64xN::from_array(core::array::from_fn(|i| {
            if i < present_lanes {
                person_indices_in_block[person_chunk_start + i].0 as u64
            } else {
                0
            }
        }));
        let person_byte_indices = person_indices / U64xN::splat(4);
        let bit_shifts = (person_indices % U64xN::splat(4)) * U64xN::splat(2);

        for variant_chunk_start in (0..variants_in_chunk).step_by(SIMD_LANES) {
            let remaining_variants = variants_in_chunk - variant_chunk_start;
            let present_variants = remaining_variants.min(SIMD_LANES);

            // --- 1. Decode a block of up to 8 variants using SIMD ---
            let mut dosage_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..present_variants {
                let variant_idx_in_batch = variant_chunk_start + i;

                // The offset is simply its index in the batch multiplied by the bytes per variant.
                let variant_byte_offset = variant_idx_in_batch as u64 * bytes_per_variant;
                let source_byte_indices = U64xN::splat(variant_byte_offset) + person_byte_indices;

                let packed_vals =
                    U8xN::gather_or_default(variant_major_data, source_byte_indices.cast());
                let two_bit_genotypes = (packed_vals >> bit_shifts.cast()) & U8xN::splat(0b11);

                let one = U8xN::splat(1);
                let term1 = (two_bit_genotypes >> U8xN::splat(1)) & one;
                let term2 = (two_bit_genotypes & one) + one;
                let initial_dosages = term1 * term2;

                let missing_mask = two_bit_genotypes.simd_eq(U8xN::splat(1));
                dosage_vectors[i] = missing_mask.select(U8xN::splat(3), initial_dosages);
            }

            // --- 2. Transpose the 8x8 block ---
            let person_data_vectors = transpose_8x8_u8(dosage_vectors);

            // --- 3. Write data to the tile, handling full and partial chunks correctly ---
            for i in 0..present_lanes {
                let person_idx_in_block = person_chunk_start + i;
                let dest_offset = person_idx_in_block * variants_in_chunk + variant_chunk_start;
                let shuffled_person_row = person_data_vectors[i].to_array();

                // OPTIMIZATION: Assemble the final row on the stack, then do a single bulk copy.
                // This removes the previous `unsafe` block, replacing it with a safe,
                // bounds-checked copy that is often optimized to a single instruction.

                // 1. Create a small, stack-allocated buffer for the unshuffled row.
                let mut temp_row = [EffectAlleleDosage::default(); SIMD_LANES];

                // 2. Un-shuffle the transposed vector into the temporary buffer.
                for j in 0..present_variants {
                    let dosage_value = shuffled_person_row[UNSHUFFLE_MAP[j]];
                    temp_row[j] = EffectAlleleDosage(dosage_value);
                }

                // 3. Perform a single, bulk copy into the main tile.
                tile[dest_offset..dest_offset + present_variants]
                    .copy_from_slice(&temp_row[..present_variants]);
            }
        }
    }
}

/// Helper function to perform an 8x8 byte matrix transpose using portable `std::simd`.
/// This is a standard, highly-optimized butterfly network algorithm. Please maintain as is.
#[inline(always)]
fn transpose_8x8_u8(matrix: [U8xN; 8]) -> [U8xN; 8] {
    let [m0, m1, m2, m3, m4, m5, m6, m7] = matrix;

    // Stage 1: Interleave 8-bit elements
    let (t0, t1) = m0.interleave(m1);
    let (t2, t3) = m2.interleave(m3);
    let (t4, t5) = m4.interleave(m5);
    let (t6, t7) = m6.interleave(m7);

    // Stage 2: Interleave 16-bit elements
    let (s0, s1) = t0.cast::<u16>().interleave(t2.cast::<u16>());
    let (s2, s3) = t1.cast::<u16>().interleave(t3.cast::<u16>());
    let (s4, s5) = t4.cast::<u16>().interleave(t6.cast::<u16>());
    let (s6, s7) = t5.cast::<u16>().interleave(t7.cast::<u16>());

    // Stage 3: Interleave 32-bit elements
    let (r0, r1) = s0.cast::<u32>().interleave(s4.cast::<u32>());
    let (r2, r3) = s1.cast::<u32>().interleave(s5.cast::<u32>());
    let (r4, r5) = s2.cast::<u32>().interleave(s6.cast::<u32>());
    let (r6, r7) = s3.cast::<u32>().interleave(s7.cast::<u32>());

    [
        r0.cast(),
        r1.cast(),
        r2.cast(),
        r3.cast(),
        r4.cast(),
        r5.cast(),
        r6.cast(),
        r7.cast(),
    ]
}

// ========================================================================================
//                     ADAPTIVE DISPATCHER & variant-MAJOR PATH
// ========================================================================================

/// Calculates a fast proxy for non-reference allele frequency using `popcnt`.
///
/// This function is a key performance enabler. It leverages the `popcnt` (population
/// count) CPU instruction, which is very fast. By viewing the byte slice as
/// `u64` chunks, it minimizes loop iterations and lets the hardware do the heavy
// lifting of counting set bits.
#[inline]
pub fn assess_variant_density(variant_data: &[u8], total_people: usize) -> f32 {
    if total_people == 0 {
        return 0.0;
    }

    const CHUNK_SIZE: usize = std::mem::size_of::<u64>();
    let mut set_bits: u32 = 0;

    // Process full 8-byte chunks using `chunks_exact` for safety and performance.
    let chunks = variant_data.chunks_exact(CHUNK_SIZE);
    let remainder = chunks.remainder();
    for chunk in chunks {
        // This conversion is safe because chunks_exact guarantees the slice length is CHUNK_SIZE.
        let val = u64::from_ne_bytes(chunk.try_into().unwrap());
        set_bits += val.count_ones();
    }

    // Process the remainder byte by byte.
    for &byte in remainder {
        set_bits += byte.count_ones();
    }

    // Normalize by the number of people to get a comparable frequency.
    // The homozygous-reference genotype (0b00) has a popcnt of 0. All others
    // have a popcnt > 0. This gives a reliable, self-contained metric.
    set_bits as f32 / total_people as f32
}

/// Processes a single sparse variant using a direct, pivot-free algorithm.
///
/// This path is optimized for variants where most individuals have the homozygous-
/// reference genotype. It avoids the high overhead of the pivot operation by
// iterating only over the individuals being scored and decoding their genotypes
// on-the-fly, which is allocation-free in the hot path.
pub fn run_variant_major_path(
    variant_data: &[u8],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f64],
    partial_missing_counts_out: &mut [u32],
    reconciled_variant_index: ReconciledVariantIndex,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let num_scores = prep_result.score_names.len();
    let stride = prep_result.stride();

    // Get pointers to the relevant weight/flip data for this variant. This is
    // an efficient calculation of the offset into the global contiguous matrices.
    let matrix_row_offset = reconciled_variant_index.0 as usize * stride;
    let weight_row = &prep_result.weights_matrix()[matrix_row_offset..matrix_row_offset + stride];
    let flip_row = &prep_result.flip_mask_matrix()[matrix_row_offset..matrix_row_offset + stride];
    let affected_scores = &prep_result.variant_to_scores_map[reconciled_variant_index.0 as usize];

    // --- Main Compute Loop ---
    // This single loop iterates only over the individuals we need to score.
    // This is the core optimization that eliminates the massive allocation and
    // redundant work of the previous implementation.
    for out_idx in 0..prep_result.num_people_to_score {
        // Use a pre-computed map to find the original .fam index for this output slot.
        let original_fam_idx = prep_result.output_idx_to_fam_idx[out_idx] as usize;

        // --- On-the-fly Genotype Decoding ---
        // This is very fast (a few bitwise operations) and allocation-free.
        let byte_index = original_fam_idx / 4;
        let bit_offset = (original_fam_idx % 4) * 2;
        let packed_val = (variant_data[byte_index] >> bit_offset) & 0b11;

        // The logic for all dosage states is handled in a single, efficient match.
        match packed_val {
            // Homozygous reference (0b00) or other unknown values. Do nothing.
            // For sparse variants, this branch is highly predictable for the CPU.
            0b00 => (),

            // Missing genotype (0b01).
            0b01 => {
                let scores_offset = out_idx * num_scores;
                for score_col_idx in affected_scores {
                    let col = score_col_idx.0;
                    partial_missing_counts_out[scores_offset + col] += 1;
                    // If the variant was flipped, its contribution (2*W) was added to the
                    // baseline. We must subtract it back out for missing individuals.
                    if flip_row[col] == 1 {
                        partial_scores_out[scores_offset + col] -= 2.0 * weight_row[col] as f64;
                    }
                }
            }

            // Heterozygous (0b10) or Homozygous alternate (0b11).
            0b10 | 0b11 => {
                let dosage = if packed_val == 0b10 { 1.0 } else { 2.0 };
                let scores_offset = out_idx * num_scores;
                for score_col_idx in affected_scores {
                    let col = score_col_idx.0;
                    let weight = weight_row[col] as f64;
                    // The baseline score already assumes a dosage of 0 for non-flipped variants,
                    // and 2 for flipped variants. This calculation applies the correct adjustment.
                    let adjustment = if flip_row[col] == 1 {
                        -weight * dosage
                    } else {
                        weight * dosage
                    };
                    partial_scores_out[scores_offset + col] += adjustment;
                }
            }

            // Should not be reached with valid PLINK data.
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_layout_is_empirically_verified() {
        // NOTE: This test is intentionally verbose with `eprintln!` for one-off
        // visual verification. In a CI/CD environment, these prints would typically be removed.
        const VARIANT_TO_SHUFFLED_POS: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

        let mut variant_major_matrix = [[0u8; 8]; 8];
        for variant_idx in 0..8 {
            for person_idx in 0..8 {
                let val = ((person_idx + 1) * 10 + (variant_idx + 1)) as u8;
                variant_major_matrix[variant_idx][person_idx] = val;
            }
        }

        let input_vectors: [U8xN; 8] =
            core::array::from_fn(|j| U8xN::from_array(variant_major_matrix[j]));

        let transposed_vectors = transpose_8x8_u8(input_vectors);

        eprintln!("\n\n=============== EMPIRICAL TRANSPOSE VERIFICATION ===============");
        eprintln!("Input Matrix (variant-Major): One row per variant");
        for variant_idx in 0..8 {
            eprintln!(
                "  variant {:?}: {:?}",
                variant_idx, variant_major_matrix[variant_idx]
            );
        }
        eprintln!("\n--- Transposed Output Layout (Person-Major) ---");
        eprintln!("Each row represents data for one Person, across all 8 variants...");

        let mut all_tests_passed = true;
        for person_idx in 0..8 {
            let person_row_actual = transposed_vectors[person_idx].to_array();
            eprintln!("  Person {:?}: {:?}", person_idx, person_row_actual);
            for variant_idx in 0..8 {
                let expected_val = ((person_idx + 1) * 10 + (variant_idx + 1)) as u8;
                let val_from_shuffled_pos = person_row_actual[VARIANT_TO_SHUFFLED_POS[variant_idx]];
                if val_from_shuffled_pos != expected_val {
                    all_tests_passed = false;
                    eprintln!(
                        "    -> FAIL for P{},S{}: Expected {}, but value at shuffled pos [{}] was {}.",
                        person_idx,
                        variant_idx,
                        expected_val,
                        VARIANT_TO_SHUFFLED_POS[variant_idx],
                        val_from_shuffled_pos
                    );
                }
            }
        }
        eprintln!("==============================================================\n");

        assert!(
            all_tests_passed,
            "The transpose output layout does not match the expected shuffle pattern."
        );
        eprintln!(
            "✅ SUCCESS: The transpose function shuffles variants within each person-vector as hypothesized."
        );
    }
}
