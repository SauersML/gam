// ========================================================================================
//
//        Gnomon compute path performance benchmark (multi-dimensional)
//
// ========================================================================================
//
// This definitive benchmark empirically measures the performance crossover point between
// the "pivot" (person-major) and "no-pivot" (variant-major) compute paths across
// multiple dimensions:
//
// 1. Total Cohort Size (N)
// 2. Number of Scores (K)
// 3. Scored Subset Percentage (simulating --keep)
// 4. Allele Frequency (f)
//
// It uses a realistic genotype generation function that models Hardy-Weinberg
// Equilibrium to provide the most accurate comparison possible.
//
// ========================================================================================

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use gnomon::batch;
use gnomon::kernel;
use gnomon::types::{
    PersonSubset, PipelineKind, PreparationResult, ReconciledVariantIndex, ScoreColumnIndex,
};

use crossbeam_queue::ArrayQueue;
use rand::seq::{SliceRandom, index};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// --- Benchmark Tuning Parameters ---

const NUM_PEOPLE_TO_TEST: &[usize] = &[1, 100, 1_000, 5_000, 10_000, 40_000];
const NUM_SCORES_TO_TEST: &[usize] = &[1, 5, 50, 100];
const SUBSET_PERCENTAGES: &[f32] = &[0.01, 0.05, 0.5, 1.0];
const ALLELE_FREQUENCIES_TO_TEST: &[f32] = &[0.00001, 0.001, 0.02, 0.4];

/// The number of dense variants to process in a single person-major batch.
const PIVOT_PATH_BATCH_SIZE: usize = 256;

/// Creates a realistic, shared "computation blueprint" for the benchmark.
///
/// This function simulates the output of the `prepare` module, providing a valid
/// `PreparationResult` that can handle a SUBSET of people, correctly simulating
/// the state of the program when a --keep file is used.
fn setup_benchmark_context(
    total_num_people: usize,
    num_scores: usize,
    num_variants: usize,
    subset_percentage: f32,
) -> Arc<PreparationResult> {
    // The .bed data is ALWAYS based on the total number of people in the .fam file.
    let bytes_per_variant = ((total_num_people as u64) + 3) / 4;
    let bytes_per_variant_usize = usize::try_from(bytes_per_variant)
        .expect("bytes per variant should fit into usize for benchmarking");
    let stride = (num_scores + kernel::LANE_COUNT - 1) / kernel::LANE_COUNT * kernel::LANE_COUNT;

    let matrix_size = num_variants * stride;
    let weights_matrix = vec![1.0f32; matrix_size];
    let flip_mask_matrix = vec![0u8; matrix_size];
    let variant_to_scores_map = vec![vec![ScoreColumnIndex(0)]; num_variants];

    // --- Subset logic ---
    let num_people_to_score = ((total_num_people as f32) * subset_percentage).round() as usize;

    let (person_subset, final_person_iids, output_idx_to_fam_idx) = if subset_percentage >= 1.0 {
        (
            PersonSubset::All,
            (0..total_num_people)
                .map(|i| format!("IID_{}", i))
                .collect(),
            (0..total_num_people as u32).collect(),
        )
    } else {
        let mut rng = rand::thread_rng();
        let fam_indices_to_keep: Vec<u32> =
            index::sample(&mut rng, total_num_people, num_people_to_score)
                .into_vec()
                .into_iter()
                .map(|i| i as u32)
                .collect();

        (
            PersonSubset::Indices(fam_indices_to_keep.clone()),
            fam_indices_to_keep
                .iter()
                .map(|&i| format!("IID_{}", i))
                .collect(),
            fam_indices_to_keep,
        )
    };

    let mut person_fam_to_output_idx = vec![None; total_num_people];
    for (output_idx, &fam_idx) in output_idx_to_fam_idx.iter().enumerate() {
        person_fam_to_output_idx[fam_idx as usize] = Some(output_idx as u32);
    }
    // --- End subset logic ---

    let (spool_compact_byte_index, spool_dense_map) = match &person_subset {
        PersonSubset::All => {
            let compact: Vec<u32> = (0..bytes_per_variant_usize).map(|i| i as u32).collect();
            let dense: Vec<i32> = (0..bytes_per_variant_usize).map(|i| i as i32).collect();
            (compact, dense)
        }
        PersonSubset::Indices(indices) => {
            let mut unique_bytes = BTreeSet::new();
            for &fam_idx in indices {
                unique_bytes.insert(fam_idx / 4);
            }
            let compact: Vec<u32> = unique_bytes.into_iter().collect();
            let mut dense = vec![-1; bytes_per_variant_usize];
            for (compact_idx, &orig_byte_idx) in compact.iter().enumerate() {
                if let Some(slot) = dense.get_mut(orig_byte_idx as usize) {
                    *slot = compact_idx as i32;
                }
            }
            (compact, dense)
        }
    };
    let spool_bytes_per_variant = spool_compact_byte_index.len() as u64;

    let prep_result = PreparationResult::new(
        weights_matrix,
        flip_mask_matrix,
        stride,
        vec![],
        vec![],
        (0..num_scores).map(|i| format!("score_{}", i)).collect(),
        vec![1; num_scores],
        variant_to_scores_map,
        person_subset,
        final_person_iids,
        num_people_to_score,
        total_num_people,
        num_variants
            .try_into()
            .expect("num_variants should fit into u64 for benchmarking"),
        num_variants,
        bytes_per_variant,
        person_fam_to_output_idx,
        output_idx_to_fam_idx,
        vec![],
        spool_compact_byte_index,
        spool_dense_map,
        spool_bytes_per_variant,
        PipelineKind::SingleFile(PathBuf::from("benchmark")),
    );
    Arc::new(prep_result)
}

/// Generates PLINK .bed format data for a single variant with realistic genotypes.
///
/// This function creates a `Vec<u8>` with genotypes corresponding to Hardy-Weinberg
/// Equilibrium (HWE) for a given allele frequency (q).
/// - p = 1 - q
/// - Freq(Homozygous Reference, 0b00) = p^2
/// - Freq(Heterozygous, 0b10)         = 2pq
/// - Freq(Homozygous Alternate, 0b11)  = q^2
fn generate_variant_data_hwe(num_people: usize, allele_frequency: f32) -> Vec<u8> {
    let q = allele_frequency;
    let p = 1.0 - q;

    let num_hom_alt = (num_people as f32 * q * q).round() as usize;
    let num_het = (num_people as f32 * 2.0 * p * q).round() as usize;
    let num_hom_ref = num_people
        .saturating_sub(num_hom_alt)
        .saturating_sub(num_het);

    let mut genotypes_to_assign = Vec::with_capacity(num_people);
    genotypes_to_assign.extend(std::iter::repeat(0b11).take(num_hom_alt));
    genotypes_to_assign.extend(std::iter::repeat(0b10).take(num_het));
    genotypes_to_assign.extend(std::iter::repeat(0b00).take(num_hom_ref));

    let mut rng = rand::thread_rng();
    genotypes_to_assign.shuffle(&mut rng);

    let bytes_per_variant = (num_people + 3) / 4;
    let mut variant_data = vec![0u8; bytes_per_variant];

    for (person_idx, genotype) in genotypes_to_assign.iter().enumerate() {
        if *genotype != 0b00 {
            let byte_index = person_idx / 4;
            let bit_offset = (person_idx % 4) * 2;
            variant_data[byte_index] |= genotype << bit_offset;
        }
    }

    variant_data
}

/// The master benchmark function that iterates through all specified dimensions.
fn benchmark_the_works(c: &mut Criterion) {
    let mut group = c.benchmark_group("Path Crossover (Multi-Dimensional)");

    for &total_people in NUM_PEOPLE_TO_TEST.iter() {
        for &num_scores in NUM_SCORES_TO_TEST.iter() {
            for &subset_pct in SUBSET_PERCENTAGES.iter() {
                for &freq in ALLELE_FREQUENCIES_TO_TEST.iter() {
                    // --- Setup for this scenario ---
                    let id_str = format!(
                        "N={}_K={}_Subset={:.0}%_Freq={:.3}",
                        total_people,
                        num_scores,
                        subset_pct * 100.0,
                        freq
                    );

                    let prep_result = setup_benchmark_context(
                        total_people,
                        num_scores,
                        PIVOT_PATH_BATCH_SIZE,
                        subset_pct,
                    );

                    let num_people_to_score = prep_result.num_people_to_score;
                    let mut scores_out = vec![0.0f64; num_people_to_score * num_scores];
                    let mut missing_counts_out = vec![0u32; num_people_to_score * num_scores];

                    group.throughput(Throughput::Elements(num_people_to_score as u64));

                    // --- 1. Benchmark variant-major (no-pivot) path ---
                    let variant_data = generate_variant_data_hwe(total_people, freq);
                    group.bench_function(
                        BenchmarkId::new(format!("No-Pivot__{}", id_str), freq),
                        |b| {
                            b.iter(|| {
                                batch::run_variant_major_path(
                                    black_box(&variant_data),
                                    black_box(&prep_result),
                                    black_box(&mut scores_out),
                                    black_box(&mut missing_counts_out),
                                    black_box(ReconciledVariantIndex(0)),
                                )
                                .unwrap();
                            });
                        },
                    );

                    // --- 2. Benchmark person-major (pivot) path ---
                    let tile_pool = Arc::new(ArrayQueue::new(4));
                    let bytes_per_variant: usize = prep_result
                        .bytes_per_variant
                        .try_into()
                        .expect("bytes_per_variant should fit into usize");
                    let mut batch_variant_data =
                        Vec::with_capacity(PIVOT_PATH_BATCH_SIZE * bytes_per_variant);
                    for _ in 0..PIVOT_PATH_BATCH_SIZE {
                        batch_variant_data
                            .extend_from_slice(&generate_variant_data_hwe(total_people, freq));
                    }
                    let reconciled_indices: Vec<_> = (0..PIVOT_PATH_BATCH_SIZE as u32)
                        .map(ReconciledVariantIndex)
                        .collect();
                    let weights_for_batch = prep_result.weights_matrix().to_vec();
                    let flips_for_batch = prep_result.flip_mask_matrix().to_vec();

                    group.bench_function(
                        BenchmarkId::new(format!("Pivot__{}", id_str), freq),
                        |b| {
                            b.iter_custom(|iters| {
                                let start = Instant::now();
                                for _ in 0..iters {
                                    batch::run_person_major_path(
                                        black_box(&batch_variant_data),
                                        black_box(&weights_for_batch),
                                        black_box(&flips_for_batch),
                                        black_box(&reconciled_indices),
                                        black_box(&prep_result),
                                        black_box(&mut scores_out),
                                        black_box(&mut missing_counts_out),
                                        black_box(&tile_pool),
                                    )
                                    .unwrap();
                                }
                                // The `iter_custom` API contract expects this closure to return the total elapsed
                                // time for all `iters` (i.e., `start.elapsed()`). Criterion would then divide
                                // by `iters` to get the average time per batch.
                                //
                                // However, our model needs to compare the AMORTIZED PER-VARIANT cost. To achieve this,
                                // we deliberately break the API contract and pre-divide the total time by the batch
                                // size. The final number reported by Criterion will be:
                                //
                                //   (total_elapsed / PIVOT_PATH_BATCH_SIZE) / iters
                                //
                                // This gives the correct per-variant cost, making it comparable to the No-Pivot path.
                                start.elapsed() / (PIVOT_PATH_BATCH_SIZE as u32)
                            });
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

// Register the master benchmark group with the Criterion runner.
criterion_group!(benches, benchmark_the_works);
criterion_main!(benches);
