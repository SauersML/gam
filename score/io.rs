// ========================================================================================
//
//                       The high-performance data producer
//
// ========================================================================================
//
// This module contains the producer logic for the gnomon compute pipeline. Its sole
// responsibility is to read the required variant data from a memory-mapped .bed file
// and send it downstream to the consumer threads for processing. It leverages a
// shared buffer pool to minimize allocations and provides natural backpressure if
// consumers cannot keep up.

use crate::score::decide::ComputePath;
use crate::score::pipeline::PipelineError;
use crate::score::types::{
    BimRowIndex, FilesetBoundary, PreparationResult, ReconciledVariantIndex, WorkItem,
};
pub use crate::shared::files::{
    BedSource, ByteRangeSource, PROGRESS_UPDATE_BATCH_SIZE, TextSource,
    gcs_billing_project_from_env, get_shared_runtime, load_adc_credentials, open_bed_source,
    open_text_source,
};
use ahash::AHashMap;
use crossbeam_channel::Sender;
use crossbeam_queue::ArrayQueue;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct SpoolPlan<'a> {
    pub is_complex_for_required: &'a [u8],
    pub compact_byte_index: &'a [u32],
    pub bytes_per_spooled_variant: u64,
    pub bytes_per_spooled_variant_usize: usize,
    pub scratch: Vec<u8>,
    pub file: &'a mut BufWriter<File>,
    pub offsets: &'a mut AHashMap<BimRowIndex, u64>,
    pub cursor: &'a mut u64,
}

impl<'a> SpoolPlan<'a> {
    #[inline(always)]
    pub fn write_variant(
        &mut self,
        variant_position: usize,
        bim_row_idx: BimRowIndex,
        buffer: &[u8],
    ) -> Result<(), PipelineError> {
        if self
            .is_complex_for_required
            .get(variant_position)
            .copied()
            .unwrap_or(0)
            == 0
        {
            return Ok(());
        }

        let offset_for_variant = *self.cursor;
        self.offsets.insert(bim_row_idx, offset_for_variant);

        if self.bytes_per_spooled_variant_usize == 0 {
            return Ok(());
        }

        debug_assert_eq!(
            self.bytes_per_spooled_variant as usize, self.bytes_per_spooled_variant_usize,
            "cached spool byte counts must match"
        );
        debug_assert_eq!(
            self.scratch.len(),
            self.bytes_per_spooled_variant_usize,
            "scratch buffer must be sized to the spooled variant stride"
        );

        for (dst_idx, &orig_byte_idx) in self.compact_byte_index.iter().enumerate() {
            debug_assert!(
                dst_idx < self.scratch.len(),
                "scratch index {} out of bounds for buffer of length {}",
                dst_idx,
                self.scratch.len()
            );
            let byte_index = orig_byte_idx as usize;
            debug_assert!(
                byte_index < buffer.len(),
                "original byte index {} out of bounds for buffer of length {}",
                byte_index,
                buffer.len()
            );
            // Safety: debug assertion above guarantees the index is within bounds.
            self.scratch[dst_idx] = unsafe { *buffer.get_unchecked(byte_index) };
        }

        self.file
            .write_all(&self.scratch[..self.bytes_per_spooled_variant_usize])
            .map_err(|e| PipelineError::Io(format!("Failed to write complex spool: {e}")))?;

        *self.cursor += self.bytes_per_spooled_variant;
        Ok(())
    }
}

/// The generic entry point for the producer thread.
///
/// This function is a template that iterates through required variants and uses a
/// provided `path_decider` closure to determine the compute path. The compiler will
/// create specialized, optimized versions of this function for each type of closure
/// it is called with, enabling static dispatch and inlining for maximum performance.
///
/// # Arguments
/// * `source`: A shared, thread-safe handle to the underlying .bed byte source.
/// * `prep_result`: The "computation blueprint" that dictates which variants to read.
/// * `sparse_tx`: The channel sender for variants destined for the sparse path.
/// * `dense_tx`: The channel sender for variants destined for the dense path.
/// * `buffer_pool`: A shared pool of reusable byte buffers to eliminate allocation overhead.
/// * `path_decider`: A closure that takes a variant's data and returns the `ComputePath`.
pub fn producer_thread<'a, F>(
    source: Arc<dyn ByteRangeSource>,
    prep_result: Arc<PreparationResult>,
    sparse_tx: Sender<Result<WorkItem, PipelineError>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    variants_processed_count: Arc<AtomicU64>,
    path_decider: F,
    mut spool: Option<SpoolPlan<'a>>,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    let bytes_per_variant = prep_result.bytes_per_variant as usize;
    let bytes_per_variant_u64 = prep_result.bytes_per_variant;
    let mut local_variants_processed: u64 = 0;

    match spool.as_mut() {
        Some(sp) => {
            let sp = sp;
            for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
                let mut buffer = buffer_pool
                    .pop()
                    .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant));
                buffer.clear();
                if buffer.capacity() < bytes_per_variant {
                    buffer.reserve(bytes_per_variant - buffer.capacity());
                }
                buffer.resize(bytes_per_variant, 0);

                let offset = 3 + bim_row_idx.0 * bytes_per_variant_u64;
                let end = offset + bytes_per_variant_u64;

                if end > source.len() {
                    let err = PipelineError::Io(format!(
                        "Fatal: Attempted to read past the end of the .bed source for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                        bim_row_idx.0
                    ));
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    break;
                }

                if let Err(err) = source.read_at(offset, buffer.as_mut_slice()) {
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    break;
                }

                if let Err(err) = sp.write_variant(i, bim_row_idx, &buffer) {
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    break;
                }

                let path = path_decider(&buffer);

                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index: ReconciledVariantIndex(i as u32),
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    &sparse_tx
                };

                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
        None => {
            for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
                let mut buffer = buffer_pool
                    .pop()
                    .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant));
                buffer.clear();
                if buffer.capacity() < bytes_per_variant {
                    buffer.reserve(bytes_per_variant - buffer.capacity());
                }
                buffer.resize(bytes_per_variant, 0);

                let offset = 3 + bim_row_idx.0 * bytes_per_variant_u64;
                let end = offset + bytes_per_variant_u64;

                if end > source.len() {
                    let err = PipelineError::Io(format!(
                        "Fatal: Attempted to read past the end of the .bed source for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                        bim_row_idx.0
                    ));
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    break;
                }

                if let Err(err) = source.read_at(offset, buffer.as_mut_slice()) {
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    break;
                }

                let path = path_decider(&buffer);

                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index: ReconciledVariantIndex(i as u32),
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    &sparse_tx
                };

                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
    }

    if local_variants_processed > 0 {
        variants_processed_count.fetch_add(local_variants_processed, Ordering::Relaxed);
    }
}

/// The producer for the multi-file pipeline. It seamlessly switches between memory-mapped
/// files as it iterates through the globally-indexed list of required variants.
pub fn multi_file_producer_thread<'a, F>(
    prep_result: Arc<PreparationResult>,
    boundaries: &[FilesetBoundary],
    bed_sources: &[BedSource],
    sparse_tx: Sender<Result<WorkItem, PipelineError>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    variants_processed_count: Arc<AtomicU64>,
    path_decider: F,
    mut spool: Option<SpoolPlan<'a>>,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    let mut current_fileset_idx: usize = 0;
    let bytes_per_variant = prep_result.bytes_per_variant;
    let mut local_variants_processed: u64 = 0;

    debug_assert_eq!(boundaries.len(), bed_sources.len());

    let mut current_source = bed_sources[0].byte_source();
    let mut next_boundary_start_idx = if boundaries.len() > 1 {
        boundaries[1].starting_global_index
    } else {
        u64::MAX
    };

    match spool.as_mut() {
        Some(sp) => {
            let sp = sp;
            for (i, &global_bim_row_index) in prep_result.required_bim_indices.iter().enumerate() {
                while global_bim_row_index.0 >= next_boundary_start_idx {
                    current_fileset_idx += 1;
                    current_source = bed_sources[current_fileset_idx].byte_source();
                    next_boundary_start_idx = if boundaries.len() > current_fileset_idx + 1 {
                        boundaries[current_fileset_idx + 1].starting_global_index
                    } else {
                        u64::MAX
                    };
                }

                let local_index =
                    global_bim_row_index.0 - boundaries[current_fileset_idx].starting_global_index;
                let offset = 3 + local_index * bytes_per_variant;
                let end = offset + bytes_per_variant;

                if end > current_source.len() {
                    let err = PipelineError::Io(format!(
                        "Fatal: Read past end of .bed source '{}' for variant with global index {}. Source may be corrupt.",
                        boundaries[current_fileset_idx].bed_path.display(),
                        global_bim_row_index.0
                    ));
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    return;
                }

                let mut buffer = buffer_pool
                    .pop()
                    .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant as usize));
                buffer.clear();
                if buffer.capacity() < bytes_per_variant as usize {
                    buffer.reserve(bytes_per_variant as usize - buffer.capacity());
                }
                buffer.resize(bytes_per_variant as usize, 0);

                if let Err(err) = current_source.read_at(offset, buffer.as_mut_slice()) {
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    return;
                }

                if let Err(err) = sp.write_variant(i, global_bim_row_index, &buffer) {
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    return;
                }

                let path = path_decider(&buffer);
                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index: ReconciledVariantIndex(i as u32),
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    &sparse_tx
                };
                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
        None => {
            for (i, &global_bim_row_index) in prep_result.required_bim_indices.iter().enumerate() {
                while global_bim_row_index.0 >= next_boundary_start_idx {
                    current_fileset_idx += 1;
                    current_source = bed_sources[current_fileset_idx].byte_source();
                    next_boundary_start_idx = if boundaries.len() > current_fileset_idx + 1 {
                        boundaries[current_fileset_idx + 1].starting_global_index
                    } else {
                        u64::MAX
                    };
                }

                let local_index =
                    global_bim_row_index.0 - boundaries[current_fileset_idx].starting_global_index;
                let offset = 3 + local_index * bytes_per_variant;
                let end = offset + bytes_per_variant;

                if end > current_source.len() {
                    let err = PipelineError::Io(format!(
                        "Fatal: Read past end of .bed source '{}' for variant with global index {}. Source may be corrupt.",
                        boundaries[current_fileset_idx].bed_path.display(),
                        global_bim_row_index.0
                    ));
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    return;
                }

                let mut buffer = buffer_pool
                    .pop()
                    .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant as usize));
                buffer.clear();
                if buffer.capacity() < bytes_per_variant as usize {
                    buffer.reserve(bytes_per_variant as usize - buffer.capacity());
                }
                buffer.resize(bytes_per_variant as usize, 0);

                if let Err(err) = current_source.read_at(offset, buffer.as_mut_slice()) {
                    let _ = sparse_tx.send(Err(err.clone()));
                    let _ = dense_tx.send(Err(err));
                    return;
                }

                let path = path_decider(&buffer);
                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index: ReconciledVariantIndex(i as u32),
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    &sparse_tx
                };
                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
    }

    if local_variants_processed > 0 {
        variants_processed_count.fetch_add(local_variants_processed, Ordering::Relaxed);
    }
}
