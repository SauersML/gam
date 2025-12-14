use crate::score::batch;
use crate::score::complex::{ComplexVariantResolver, resolve_complex_variants};
use crate::score::decide::{self, DecisionContext, RunStrategy};
use crate::score::io;
use crate::score::types::{
    BimRowIndex, EffectAlleleDosage, FilesetBoundary, PipelineKind, PreparationResult,
    ReconciledVariantIndex, WorkItem,
};
use ahash::AHashMap;
use crossbeam_channel::{Receiver, bounded};
use crossbeam_queue::ArrayQueue;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use memmap2::{Mmap, MmapOptions};
use num_cpus;
use rand::{RngCore, thread_rng};
use rayon::prelude::*;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// --- Pipeline Tuning Parameters ---

/// The maximum number of sparse work items that can be buffered in the channel.
/// Provides backpressure against a fast producer.
const SPARSE_CHANNEL_BOUND: usize = 8192;
/// The maximum number of dense work items that can be buffered in the channel.
const DENSE_CHANNEL_BOUND: usize = 4096;
/// The number of dense variants to process in a single person-major batch.
/// Tuned for L3 cache efficiency.
const DENSE_BATCH_SIZE: usize = 256;
/// The number of reusable memory buffers for variant data.
const BUFFER_POOL_SIZE: usize = 16384;
/// The buffer size for complex variant spooling.
const SPOOL_BUFFER_SIZE: usize = 8 * 1024 * 1024;

struct SpoolState {
    writer: BufWriter<File>,
    offsets: AHashMap<BimRowIndex, u64>,
    cursor: u64,
}

fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
    let draw_target = if std::io::stderr().is_terminal() {
        ProgressDrawTarget::stderr_with_hz(20)
    } else {
        ProgressDrawTarget::hidden()
    };

    let pb = ProgressBar::with_draw_target(Some(len), draw_target);
    pb.set_style(
        ProgressStyle::with_template(
            "\n> [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message(message.to_string());

    pb
}

// ========================================================================================
//                          Public API, context & error handling
// ========================================================================================

/// An iterator that pulls items from a channel and groups them into batches.
///
/// This is a `Send`-compatible replacement for `itertools::chunks` on a channel
/// iterator, enabling true streaming processing with `rayon::par_bridge`. It is
/// the key to enabling simultaneous I/O and computation for the dense path.
struct ChannelBatcher<T> {
    rx: Receiver<Result<T, PipelineError>>,
    batch_size: usize,
}

impl<T> ChannelBatcher<T> {
    fn new(rx: Receiver<Result<T, PipelineError>>, batch_size: usize) -> Self {
        Self { rx, batch_size }
    }
}

// The implementation of the `Iterator` trait is what allows this to be used in loops
// and with adapters like `par_bridge`.
impl<T: Send> Iterator for ChannelBatcher<T> {
    // The iterator yields a `Result` containing either a `Vec` of items (a batch)
    // or a `PipelineError` if one was sent by the producer.
    type Item = Result<Vec<T>, PipelineError>;

    fn next(&mut self) -> Option<Self::Item> {
        // First, block waiting for one item. If the channel is empty and has been
        // closed by the producer, `recv()` will return an error, and we'll return `None`,
        // ending the iteration. This is the correct way to terminate the stream.
        match self.rx.recv() {
            // Happy path: We received a valid work item from the producer.
            Ok(Ok(first_item)) => {
                let mut batch = Vec::with_capacity(self.batch_size);
                batch.push(first_item);

                // Greedily pull more items from the channel *without blocking* until
                // the batch is full or the channel is momentarily empty.
                while batch.len() < self.batch_size {
                    match self.rx.try_recv() {
                        Ok(Ok(item)) => batch.push(item),
                        // An error was sent down the channel. Stop and propagate it.
                        Ok(Err(e)) => return Some(Err(e)),
                        // The channel is empty or disconnected. The current batch is finished.
                        Err(_) => break,
                    }
                }
                // Return the completed (or partially-filled) batch.
                Some(Ok(batch))
            }
            // An error was sent down the channel as the first item. Propagate it.
            Ok(Err(e)) => Some(Err(e)),
            // The producer has disconnected the channel. End the iteration.
            Err(_) => None,
        }
    }
}

/// A specialized error type for the pipeline, allowing for robust, clonable error
/// propagation from any concurrent stage.
#[derive(Debug, Clone)]
pub enum PipelineError {
    Compute(String),
    Io(String),
    Producer(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Compute(e) => write!(f, "{e}"),
            PipelineError::Io(e) => write!(f, "I/O error during pipeline execution: {e}"),
            PipelineError::Producer(e) => write!(f, "The data producer thread failed: {e}"),
        }
    }
}
impl Error for PipelineError {}

// Enables easy conversion from batch errors into a pipeline error.
impl From<Box<dyn Error + Send + Sync>> for PipelineError {
    fn from(e: Box<dyn Error + Send + Sync>) -> Self {
        PipelineError::Compute(e.to_string())
    }
}

/// Owns shared resource pools and provides a handle to the read-only preparation results.
pub struct PipelineContext {
    pub prep_result: Arc<PreparationResult>,
    pub tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
}

impl PipelineContext {
    /// Creates a new `PipelineContext`, allocating all necessary memory pools.
    pub fn new(prep_result: Arc<PreparationResult>) -> Self {
        Self {
            prep_result,
            tile_pool: Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 4)),
        }
    }
}

/// Executes the entire concurrent compute pipeline.
///
/// This is the primary public entry point. It is synchronous and returns the
/// final aggregated scores and counts upon successful completion.
pub fn run(context: &PipelineContext) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // This match is a zero-cost abstraction. The compiler generates a simple jump
    // to the correct function based on the enum variant, and it's impossible
    // to call the wrong pipeline logic for a given configuration.
    match &context.prep_result.pipeline_kind {
        PipelineKind::SingleFile(bed_path) => run_single_file_pipeline(context, bed_path),
        PipelineKind::MultiFile(boundaries) => run_multi_file_pipeline(context, boundaries),
    }
}

// ========================================================================================
//                        Pipeline stage implementations
// ========================================================================================

/// The pipeline implementation for the common single-fileset case.
/// This function's body is effectively the same as the original `pipeline::run` function,
/// guaranteeing zero performance regression.
fn run_single_file_pipeline(
    context: &PipelineContext,
    bed_path: &Path,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // --- 1. Setup: Memory-map the file, create channels and a shared buffer pool ---
    let bed_source = io::open_bed_source(bed_path)?;
    let shared_source = bed_source.byte_source();

    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(SPARSE_CHANNEL_BOUND);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(DENSE_CHANNEL_BOUND);

    let buffer_pool = Arc::new(ArrayQueue::new(BUFFER_POOL_SIZE));
    for _ in 0..BUFFER_POOL_SIZE {
        buffer_pool
            .push(Vec::with_capacity(
                context.prep_result.bytes_per_variant as usize,
            ))
            .unwrap();
    }

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let variants_processed_count = Arc::new(AtomicU64::new(0));
    let pb = create_progress_bar(variants_to_process, "Computing scores...");

    // --- 2. Pre-computation & STRATEGY SELECTION ---
    let prep_result = &context.prep_result;
    let run_ctx = DecisionContext {
        n_cohort: prep_result.total_people_in_fam as f32,
        k_scores: prep_result.score_names.len() as f32,
        subset_frac: prep_result.num_people_to_score as f32
            / prep_result.total_people_in_fam as f32,
        freq: 0.0,
    };
    let strategy = decide::RunStrategy::UseComplexTree;
    eprintln!("> Decision Engine Strategy: {strategy:?}");

    let num_scores = prep_result.score_names.len();
    let stride = prep_result.stride();
    let master_baseline: Vec<f64> = (0..prep_result.num_reconciled_variants)
        .into_par_iter()
        .fold(
            || vec![0.0f64; num_scores],
            |mut local_baseline, i| {
                let flip_row_offset = i * stride;
                let flip_row =
                    &prep_result.flip_mask_matrix()[flip_row_offset..flip_row_offset + stride];
                let weight_row =
                    &prep_result.weights_matrix()[flip_row_offset..flip_row_offset + stride];
                for k in 0..num_scores {
                    if flip_row[k] == 1 {
                        local_baseline[k] += 2.0 * weight_row[k] as f64;
                    }
                }
                local_baseline
            },
        )
        .reduce(
            || vec![0.0f64; num_scores],
            |mut a, b| {
                a.par_iter_mut().zip(b).for_each(|(v_a, v_b)| *v_a += v_b);
                a
            },
        );

    let has_complex = !prep_result.complex_rules.is_empty();
    let is_remote = bed_source.mmap().is_none();
    let should_spool = has_complex && is_remote;
    let mut spool_state: Option<SpoolState> = None;
    let mut spool_path: Option<PathBuf> = None;
    if should_spool {
        let (spool_dir, spool_stem) = derive_spool_destination(bed_path);
        fs::create_dir_all(&spool_dir).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool directory {}: {e}",
                spool_dir.display()
            ))
        })?;
        let filename = unique_spool_filename(&spool_stem);
        let path = spool_dir.join(&filename);
        let file = File::create(&path).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool file {}: {e}",
                path.display()
            ))
        })?;
        let complex_variant_count = prep_result
            .required_is_complex()
            .iter()
            .filter(|&&flag| flag != 0)
            .count() as u64;
        let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
        let approx_mb = if spool_bytes_per_variant == 0 {
            0.0
        } else {
            (complex_variant_count * spool_bytes_per_variant) as f64 / (1024.0 * 1024.0)
        };
        eprintln!(
            "> Spooling complex genotypes locally: {} variants × {} B ≈ {:.2} MiB to {}",
            complex_variant_count,
            spool_bytes_per_variant,
            approx_mb,
            path.display()
        );
        let offsets_capacity = usize::try_from(complex_variant_count)
            .unwrap_or(usize::MAX / 2)
            .max(1);
        spool_state = Some(SpoolState {
            writer: BufWriter::with_capacity(SPOOL_BUFFER_SIZE, file),
            offsets: AHashMap::with_capacity(offsets_capacity),
            cursor: 0,
        });
        spool_path = Some(path);
    }

    // --- 3. Orchestration: Use a scoped thread for safe producer/consumer execution ---
    let run_ctx_for_closure = run_ctx;
    let strategy_for_closure = strategy;
    let final_result: Result<((Vec<f64>, Vec<u32>), Option<SpoolState>), PipelineError> =
        thread::scope(|s| {
            // Spawn the UI updater thread. This thread is responsible for polling the
            // atomic counter and updating the progress bar on the screen.
            let updater_thread_count = Arc::clone(&variants_processed_count);
            let updater_pb = pb.clone();
            let total_variants = variants_to_process;
            s.spawn(move || {
                // This loop terminates when the number of processed items reaches the
                // total, ensuring this thread finishes before the scope ends
                while updater_thread_count.load(Ordering::Relaxed) < total_variants {
                    let processed = updater_thread_count.load(Ordering::Relaxed);
                    updater_pb.set_position(processed);
                    thread::sleep(Duration::from_millis(200));
                }
                // Perform one final update to ensure the bar shows 100% completion.
                updater_pb.set_position(updater_thread_count.load(Ordering::Relaxed));
            });

            let mut local_spool_state = spool_state.take();
            let producer_logic = {
                let source = Arc::clone(&shared_source);
                let prep_result = Arc::clone(&context.prep_result);
                let buffer_pool = Arc::clone(&buffer_pool);
                let producer_thread_count = Arc::clone(&variants_processed_count);
                let spool_enabled = should_spool;
                let run_ctx = run_ctx_for_closure;
                let strategy = strategy_for_closure;

                move || -> Result<Option<SpoolState>, PipelineError> {
                    match strategy {
                        RunStrategy::UseSimpleTree => {
                            let global_path = decide::decide_path_without_freq(&run_ctx);
                            let path_decider = |_: &[u8]| global_path;
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::producer_thread(
                                Arc::clone(&source),
                                Arc::clone(&prep_result),
                                sparse_tx,
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                            );
                        }
                        RunStrategy::UseComplexTree => {
                            let path_decider = |variant_data: &[u8]| {
                                let current_freq = batch::assess_variant_density(
                                    variant_data,
                                    run_ctx.n_cohort as usize,
                                );
                                let variant_ctx = DecisionContext {
                                    freq: current_freq,
                                    ..run_ctx
                                };
                                decide::decide_path_with_freq(&variant_ctx)
                            };
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::producer_thread(
                                Arc::clone(&source),
                                Arc::clone(&prep_result),
                                sparse_tx,
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                            );
                        }
                    }
                    Ok(local_spool_state)
                }
            };

            let producer_handle = s.spawn(producer_logic);
            let (sparse_result, dense_result) = rayon::join(
                || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
                || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
            );
            let local_spool_state = producer_handle
                .join()
                .map_err(|_| PipelineError::Producer("Producer thread panicked.".to_string()))??;

            // --- 4. Aggregate final results ---
            let (sparse_adjustments, sparse_counts) = sparse_result?;
            let (dense_adjustments, dense_counts) = dense_result?;
            let num_people = prep_result.num_people_to_score;
            let mut final_scores = Vec::with_capacity(num_people * num_scores);
            for _ in 0..num_people {
                final_scores.extend_from_slice(&master_baseline);
            }
            let mut final_counts = vec![0u32; num_people * num_scores];
            final_counts
                .par_iter_mut()
                .zip(sparse_counts)
                .for_each(|(m, p)| *m += p);
            final_counts
                .par_iter_mut()
                .zip(dense_counts)
                .for_each(|(m, p)| *m += p);
            final_scores
                .par_iter_mut()
                .zip(sparse_adjustments)
                .for_each(|(m, p)| *m += p);
            final_scores
                .par_iter_mut()
                .zip(dense_adjustments)
                .for_each(|(m, p)| *m += p);

            pb.finish_with_message("Computation complete.");
            Ok(((final_scores, final_counts), local_spool_state))
        });
    let ((mut final_scores, mut final_counts), mut spool_state) = final_result?;

    if !prep_result.complex_rules.is_empty() {
        let resolver_label = if should_spool {
            "spooled mmap"
        } else {
            "single-file mmap"
        };
        eprintln!(
            "> Resolving {} unique complex variant rule(s) with {} resolver...",
            prep_result.complex_rules.len(),
            resolver_label
        );
        if should_spool {
            let spool_file_path = spool_path
                .clone()
                .expect("spool path missing despite spooling enabled");
            let mut state = spool_state
                .take()
                .expect("spool state missing despite spooling enabled");
            state.writer.flush().map_err(|e| {
                PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
            })?;
            drop(state.writer);
            let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
            let mmap = if spool_bytes_per_variant == 0 {
                let mut anon = MmapOptions::new().len(1).map_anon().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to allocate anonymous mapping for empty complex spool: {e}"
                    ))
                })?;
                anon.copy_from_slice(&[0u8]);
                anon.make_read_only().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to convert anonymous mapping to read-only: {e}"
                    ))
                })?
            } else {
                let spool_file = File::open(&spool_file_path).map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to open complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?;
                unsafe { Mmap::map(&spool_file) }.map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to memory-map complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?
            };
            let dense_map = Arc::new(prep_result.spool_dense_map().to_vec());
            let resolver = ComplexVariantResolver::from_spool(
                Arc::new(mmap),
                state.offsets,
                spool_bytes_per_variant,
                dense_map,
            );
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
            let keep_spool = env::var("GNOMON_KEEP_SPOOL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !keep_spool {
                if let Err(e) = fs::remove_file(&spool_file_path) {
                    eprintln!(
                        "> Warning: Failed to delete complex spool {}: {}",
                        spool_file_path.display(),
                        e
                    );
                }
            } else {
                eprintln!(
                    "> GNOMON_KEEP_SPOOL set; preserving complex spool at {}",
                    spool_file_path.display()
                );
            }
        } else {
            let resolver = ComplexVariantResolver::from_single_source(bed_source.clone());
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

/// The pipeline implementation for the multi-fileset case.
fn run_multi_file_pipeline(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let bed_sources: Vec<io::BedSource> = boundaries
        .iter()
        .map(|b| io::open_bed_source(&b.bed_path))
        .collect::<Result<_, _>>()?;
    let any_remote = bed_sources.iter().any(|s| s.mmap().is_none());
    let shared_sources = Arc::new(bed_sources);

    // --- 1. Setup: No mmap here. Producer manages its own. ---
    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(SPARSE_CHANNEL_BOUND);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(DENSE_CHANNEL_BOUND);
    let buffer_pool = Arc::new(ArrayQueue::new(BUFFER_POOL_SIZE));
    for _ in 0..BUFFER_POOL_SIZE {
        buffer_pool
            .push(Vec::with_capacity(
                context.prep_result.bytes_per_variant as usize,
            ))
            .unwrap();
    }

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let variants_processed_count = Arc::new(AtomicU64::new(0));
    let pb = create_progress_bar(variants_to_process, "Computing scores...");

    // --- 2. Pre-computation (same as single-file) ---
    let prep_result = &context.prep_result;
    let run_ctx = DecisionContext {
        n_cohort: prep_result.total_people_in_fam as f32,
        k_scores: prep_result.score_names.len() as f32,
        subset_frac: prep_result.num_people_to_score as f32
            / prep_result.total_people_in_fam as f32,
        freq: 0.0,
    };
    let strategy = decide::RunStrategy::UseComplexTree;
    eprintln!("> Decision Engine Strategy: {strategy:?}");
    let master_baseline = {
        let num_scores = prep_result.score_names.len();
        let stride = prep_result.stride();
        (0..prep_result.num_reconciled_variants)
            .into_par_iter()
            .fold(
                || vec![0.0f64; num_scores],
                |mut local_baseline, i| {
                    let flip_row_offset = i * stride;
                    let flip_row =
                        &prep_result.flip_mask_matrix()[flip_row_offset..flip_row_offset + stride];
                    let weight_row =
                        &prep_result.weights_matrix()[flip_row_offset..flip_row_offset + stride];
                    for k in 0..num_scores {
                        if flip_row[k] == 1 {
                            local_baseline[k] += 2.0 * weight_row[k] as f64;
                        }
                    }
                    local_baseline
                },
            )
            .reduce(
                || vec![0.0f64; num_scores],
                |mut a, b| {
                    a.par_iter_mut().zip(b).for_each(|(v_a, v_b)| *v_a += v_b);
                    a
                },
            )
    };

    let has_complex = !prep_result.complex_rules.is_empty();
    let should_spool = has_complex && any_remote;
    let mut spool_state: Option<SpoolState> = None;
    let mut spool_path: Option<PathBuf> = None;
    if should_spool {
        let (spool_dir, spool_stem) = derive_spool_destination(&boundaries[0].bed_path);
        fs::create_dir_all(&spool_dir).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool directory {}: {e}",
                spool_dir.display()
            ))
        })?;
        let filename = unique_spool_filename(&spool_stem);
        let path = spool_dir.join(&filename);
        let file = File::create(&path).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool file {}: {e}",
                path.display()
            ))
        })?;
        let complex_variant_count = prep_result
            .required_is_complex()
            .iter()
            .filter(|&&flag| flag != 0)
            .count() as u64;
        let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
        let approx_mb = if spool_bytes_per_variant == 0 {
            0.0
        } else {
            (complex_variant_count * spool_bytes_per_variant) as f64 / (1024.0 * 1024.0)
        };
        eprintln!(
            "> Spooling complex genotypes locally: {} variants × {} B ≈ {:.2} MiB to {}",
            complex_variant_count,
            spool_bytes_per_variant,
            approx_mb,
            path.display()
        );
        let offsets_capacity = usize::try_from(complex_variant_count)
            .unwrap_or(usize::MAX / 2)
            .max(1);
        spool_state = Some(SpoolState {
            writer: BufWriter::with_capacity(SPOOL_BUFFER_SIZE, file),
            offsets: AHashMap::with_capacity(offsets_capacity),
            cursor: 0,
        });
        spool_path = Some(path);
    }

    // --- 3. Orchestration with multi-file producer ---
    let run_ctx_for_closure = run_ctx;
    let strategy_for_closure = strategy;
    let final_result: Result<((Vec<f64>, Vec<u32>), Option<SpoolState>), PipelineError> =
        thread::scope(|s| {
            // Spawn the UI updater thread. This thread is responsible for polling the
            // atomic counter and updating the progress bar on the screen.
            let updater_thread_count = Arc::clone(&variants_processed_count);
            let updater_pb = pb.clone();
            let total_variants = variants_to_process;
            s.spawn(move || {
                // This loop terminates when the number of processed items reaches the
                // total, ensuring this thread finishes before the scope ends
                while updater_thread_count.load(Ordering::Relaxed) < total_variants {
                    let processed = updater_thread_count.load(Ordering::Relaxed);
                    updater_pb.set_position(processed);
                    thread::sleep(Duration::from_millis(200));
                }
                // Perform one final update to ensure the bar shows 100% completion.
                updater_pb.set_position(updater_thread_count.load(Ordering::Relaxed));
            });

            let mut local_spool_state = spool_state.take();
            let producer_logic = {
                let sources = Arc::clone(&shared_sources);
                let prep_result = Arc::clone(&context.prep_result);
                let buffer_pool = Arc::clone(&buffer_pool);
                let producer_thread_count = Arc::clone(&variants_processed_count);
                let spool_enabled = should_spool;
                let run_ctx = run_ctx_for_closure;
                let strategy = strategy_for_closure;

                move || -> Result<Option<SpoolState>, PipelineError> {
                    match strategy {
                        RunStrategy::UseSimpleTree => {
                            let global_path = decide::decide_path_without_freq(&run_ctx);
                            let path_decider = |_: &[u8]| global_path;
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::multi_file_producer_thread(
                                Arc::clone(&prep_result),
                                boundaries,
                                sources.as_ref(),
                                sparse_tx,
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                            );
                        }
                        RunStrategy::UseComplexTree => {
                            let path_decider = |variant_data: &[u8]| {
                                let current_freq = batch::assess_variant_density(
                                    variant_data,
                                    run_ctx.n_cohort as usize,
                                );
                                let variant_ctx = DecisionContext {
                                    freq: current_freq,
                                    ..run_ctx
                                };
                                decide::decide_path_with_freq(&variant_ctx)
                            };
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::multi_file_producer_thread(
                                Arc::clone(&prep_result),
                                boundaries,
                                sources.as_ref(),
                                sparse_tx,
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                            );
                        }
                    }
                    Ok(local_spool_state)
                }
            };

            let producer_handle = s.spawn(producer_logic);
            let (sparse_result, dense_result) = rayon::join(
                || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
                || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
            );
            let local_spool_state = producer_handle
                .join()
                .map_err(|_| PipelineError::Producer("Producer thread panicked.".to_string()))??;

            // --- 4. Aggregate final results (same as single-file) ---
            let (sparse_adjustments, sparse_counts) = sparse_result?;
            let (dense_adjustments, dense_counts) = dense_result?;
            let num_people = prep_result.num_people_to_score;
            let num_scores = prep_result.score_names.len();
            let mut final_scores = Vec::with_capacity(num_people * num_scores);
            for _ in 0..num_people {
                final_scores.extend_from_slice(&master_baseline);
            }
            let mut final_counts = vec![0u32; num_people * num_scores];
            final_counts
                .par_iter_mut()
                .zip(sparse_counts)
                .for_each(|(m, p)| *m += p);
            final_counts
                .par_iter_mut()
                .zip(dense_counts)
                .for_each(|(m, p)| *m += p);
            final_scores
                .par_iter_mut()
                .zip(sparse_adjustments)
                .for_each(|(m, p)| *m += p);
            final_scores
                .par_iter_mut()
                .zip(dense_adjustments)
                .for_each(|(m, p)| *m += p);

            pb.finish_with_message("Computation complete.");
            Ok(((final_scores, final_counts), local_spool_state))
        });
    let ((mut final_scores, mut final_counts), mut spool_state) = final_result?;

    if !prep_result.complex_rules.is_empty() {
        let resolver_label = if should_spool {
            "spooled mmap"
        } else {
            "multi-file mmap"
        };
        eprintln!(
            "> Resolving {} unique complex variant rule(s) with {} resolver...",
            prep_result.complex_rules.len(),
            resolver_label
        );
        if should_spool {
            let spool_file_path = spool_path
                .clone()
                .expect("spool path missing despite spooling enabled");
            let mut state = spool_state
                .take()
                .expect("spool state missing despite spooling enabled");
            state.writer.flush().map_err(|e| {
                PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
            })?;
            drop(state.writer);
            let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
            let mmap = if spool_bytes_per_variant == 0 {
                let mut anon = MmapOptions::new().len(1).map_anon().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to allocate anonymous mapping for empty complex spool: {e}"
                    ))
                })?;
                anon.copy_from_slice(&[0u8]);
                anon.make_read_only().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to convert anonymous mapping to read-only: {e}"
                    ))
                })?
            } else {
                let spool_file = File::open(&spool_file_path).map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to open complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?;
                unsafe { Mmap::map(&spool_file) }.map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to memory-map complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?
            };
            let dense_map = Arc::new(prep_result.spool_dense_map().to_vec());
            let resolver = ComplexVariantResolver::from_spool(
                Arc::new(mmap),
                state.offsets,
                spool_bytes_per_variant,
                dense_map,
            );
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
            let keep_spool = env::var("GNOMON_KEEP_SPOOL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !keep_spool {
                if let Err(e) = fs::remove_file(&spool_file_path) {
                    eprintln!(
                        "> Warning: Failed to delete complex spool {}: {}",
                        spool_file_path.display(),
                        e
                    );
                }
            } else {
                eprintln!(
                    "> GNOMON_KEEP_SPOOL set; preserving complex spool at {}",
                    spool_file_path.display()
                );
            }
        } else {
            let resolver = ComplexVariantResolver::from_multi_sources(
                shared_sources.as_ref().clone(),
                boundaries.to_vec(),
            )?;
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

fn derive_spool_destination(base_path: &Path) -> (PathBuf, String) {
    let stem = base_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "gnomon_results".to_string());
    let path_str = base_path.to_string_lossy();
    if path_str.starts_with("gs://")
        || path_str.starts_with("http://")
        || path_str.starts_with("https://")
    {
        (Path::new(".").to_path_buf(), stem)
    } else {
        let dir = base_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        (dir, stem)
    }
}

fn unique_spool_filename(stem: &str) -> String {
    let pid = process::id();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    let timestamp = now.as_nanos();
    let mut rng = thread_rng();
    let random_component: u32 = rng.next_u32();
    format!(
        "{}.{}.{}.{}.complex_spool.bin",
        stem, pid, timestamp, random_component
    )
}

fn create_spool_plan<'a>(
    prep_result: &'a PreparationResult,
    state: &'a mut SpoolState,
) -> Result<io::SpoolPlan<'a>, PipelineError> {
    let stride = prep_result.spool_bytes_per_variant();
    let stride_usize = usize::try_from(stride).map_err(|_| {
        PipelineError::Compute(format!(
            "spool stride of {} bytes does not fit on this platform",
            stride
        ))
    })?;
    Ok(io::SpoolPlan {
        is_complex_for_required: prep_result.required_is_complex(),
        compact_byte_index: prep_result.spool_compact_byte_index(),
        bytes_per_spooled_variant: stride,
        bytes_per_spooled_variant_usize: stride_usize,
        scratch: vec![0u8; stride_usize],
        file: &mut state.writer,
        offsets: &mut state.offsets,
        cursor: &mut state.cursor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_spool_destination_remote_paths_default_to_current_dir() {
        let (dir, stem) = derive_spool_destination(Path::new("gs://bucket/data/sample.bed"));
        assert_eq!(dir, Path::new("."));
        assert_eq!(stem, "sample");

        let (dir_http, stem_http) =
            derive_spool_destination(Path::new("https://example.com/study/run"));
        assert_eq!(dir_http, Path::new("."));
        assert_eq!(stem_http, "run");
    }

    #[test]
    fn derive_spool_destination_local_paths_use_parent_directory() {
        let path = Path::new("/tmp/project/cohort1.bed");
        let (dir, stem) = derive_spool_destination(path);
        assert_eq!(dir, Path::new("/tmp/project"));
        assert_eq!(stem, "cohort1");
    }
}

/// A RAII guard that ensures a byte buffer is automatically returned to the shared
/// buffer pool when it goes out of scope. This is critical for preventing resource
// leaks in the consumer streams, especially when errors occur.
struct BufferGuard<'a> {
    /// The buffer being managed. Wrapped in an `Option` to allow ownership to be
    /// taken in the `drop` implementation.
    buffer: Option<Vec<u8>>,
    /// A reference to the shared pool where the buffer will be returned.
    pool: &'a ArrayQueue<Vec<u8>>,
}

impl<'a> Drop for BufferGuard<'a> {
    fn drop(&mut self) {
        // When the guard is dropped, it returns its buffer to the pool.
        if let Some(mut buf) = self.buffer.take() {
            buf.clear();
            let _ = self.pool.push(buf);
        }
    }
}

/// A general-purpose RAII guard that executes a closure when it goes out of scope.
///
/// This utility is crucial for ensuring that a specific action, such as releasing a
/// resource or signaling completion, is performed regardless of how a scope is exited
/// (e.g., normal completion, early return, or panic). It holds an optional closure,
/// which is taken and executed in the `drop` implementation, guaranteeing the
/// action runs exactly once.
pub struct ScopeGuard<F: FnOnce()> {
    /// The closure to execute on drop. `Option` is used to allow the closure
    /// to be taken and called, preventing multiple executions.
    action: Option<F>,
}

impl<F: FnOnce()> ScopeGuard<F> {
    /// Creates a new `ScopeGuard` with the given action.
    ///
    /// The action will be executed when the returned guard is dropped.
    #[inline(always)]
    pub fn new(action: F) -> Self {
        Self {
            action: Some(action),
        }
    }
}

impl<F: FnOnce()> Drop for ScopeGuard<F> {
    /// Executes the stored action when the guard goes out of scope.
    ///
    /// This method is called automatically by the Rust compiler. It takes the
    /// action out of the `Option`, ensuring it can only be run once, and then
    /// executes it.
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(action) = self.action.take() {
            action();
        }
    }
}

type ConsumerResult = Result<(Vec<f64>, Vec<u32>), PipelineError>;

/// A contention-free consumer for the sparse variant stream, using Rayon's
/// fold/reduce pattern for maximum parallelism with no locks.
fn process_sparse_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = prep_result.num_people_to_score * prep_result.score_names.len();

    // The fold/reduce pattern creates thread-local accumulators for scores and counts.
    // After processing a work item, its data buffer is immediately returned to the
    // shared pool, creating a true, continuous recycling system.
    let final_result = rx
        .into_iter() // Convert the channel to a blocking iterator.
        .par_bridge() // Bridge it to a Rayon parallel iterator.
        .try_fold(
            || (vec![0.0f64; result_size], vec![0u32; result_size]), // Each thread gets its own accumulator.
            |mut acc, work_result| {
                // The work_item and its buffer are processed within this scope.
                // The `_guard` ensures the buffer is returned to the pool when this
                // scope ends, whether by success or by `?` propagating an error.
                {
                    let work_item = work_result?;
                    let guard = BufferGuard {
                        buffer: Some(work_item.data),
                        pool: &buffer_pool,
                    };

                    batch::run_variant_major_path(
                        // The guard holds the buffer, so we borrow it from there.
                        guard.buffer.as_ref().unwrap(),
                        prep_result,
                        &mut acc.0,
                        &mut acc.1,
                        work_item.reconciled_variant_index,
                    )?;
                }
                Ok::<_, PipelineError>(acc)
            },
        )
        .try_reduce(
            || (vec![0.0f64; result_size], vec![0u32; result_size]), // Identity for the reduction.
            |mut a, b| {
                // Combine accumulators from two threads in parallel.
                a.0.par_iter_mut()
                    .zip(b.0)
                    .for_each(|(v_a, v_b)| *v_a += v_b);
                a.1.par_iter_mut()
                    .zip(b.1)
                    .for_each(|(v_a, v_b)| *v_a += v_b);
                Ok(a)
            },
        )?;

    // `try_reduce` returns `Result<(scores, counts), PipelineError>`.
    // The `?` operator has already unwrapped the Result, leaving just the tuple.
    // With an identity function, try_reduce handles empty streams by returning the identity.
    Ok(final_result)
}

/// A contention-free consumer for the dense variant stream. It uses a custom
/// batching iterator to group items, which are then processed in parallel by Rayon.
/// This implementation allows I/O and computation to run concurrently.
fn process_dense_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = prep_result.num_people_to_score * prep_result.score_names.len();

    // Instantiate our new Send-compatible batching iterator.
    let batch_iterator = ChannelBatcher::new(rx, DENSE_BATCH_SIZE);

    // Use the exact same fold/reduce pattern as the sparse stream, but on batches.
    let final_result = batch_iterator
        .par_bridge() // This is now possible and correct.
        .try_fold(
            || {
                // Per-thread accumulator initializer
                (
                    vec![0.0f64; result_size],
                    vec![0u32; result_size],
                    Vec::with_capacity(DENSE_BATCH_SIZE * (prep_result.bytes_per_variant as usize)),
                )
            },
            |mut acc, batch_result| {
                // The `?` operator handles propagating errors from the channel.
                let batch = batch_result?;
                if batch.is_empty() {
                    return Ok(acc);
                }

                let reconciled_indices: Vec<ReconciledVariantIndex> =
                    batch.iter().map(|wi| wi.reconciled_variant_index).collect();

                let concatenated_data = &mut acc.2;
                concatenated_data.clear();

                // BufferGuard ensures all variant data buffers are returned to the pool,
                // even if an error occurs during computation.
                let guards: Vec<_> = batch
                    .into_iter()
                    .map(|wi| {
                        concatenated_data.extend_from_slice(&wi.data);
                        BufferGuard {
                            buffer: Some(wi.data),
                            pool: &buffer_pool,
                        }
                    })
                    .collect();

                let stride = prep_result.stride();
                let mut weights_for_batch = Vec::with_capacity(reconciled_indices.len() * stride);
                let mut flips_for_batch = Vec::with_capacity(reconciled_indices.len() * stride);
                for &reconciled_idx in &reconciled_indices {
                    let src_offset = reconciled_idx.0 as usize * stride;
                    weights_for_batch.extend_from_slice(
                        &prep_result.weights_matrix()[src_offset..src_offset + stride],
                    );
                    flips_for_batch.extend_from_slice(
                        &prep_result.flip_mask_matrix()[src_offset..src_offset + stride],
                    );
                }

                batch::run_person_major_path(
                    concatenated_data,
                    &weights_for_batch,
                    &flips_for_batch,
                    &reconciled_indices,
                    prep_result,
                    &mut acc.0,
                    &mut acc.1,
                    &context.tile_pool,
                )?;

                drop(guards); // Explicitly drop to return buffers to the pool.

                Ok::<_, PipelineError>(acc)
            },
        )
        .try_reduce(
            || (vec![0.0; result_size], vec![0; result_size], Vec::new()),
            |mut a, b| {
                a.0.par_iter_mut()
                    .zip(b.0)
                    .for_each(|(v_a, v_b)| *v_a += v_b);
                a.1.par_iter_mut()
                    .zip(b.1)
                    .for_each(|(v_a, v_b)| *v_a += v_b);
                Ok(a)
            },
        )?;

    // The `?` operator unwrapped the `Result` from the reduction. If the stream was
    // empty, `try_reduce` (on a `TryFold` iterator) returns the identity value, so
    // `final_result` correctly contains the initial empty vectors. We just need to destructure the tuple.
    let (scores, counts, _) = final_result;
    Ok((scores, counts))
}
