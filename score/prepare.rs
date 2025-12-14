// ========================================================================================
//
//               The preparation "compiler"
//
// ========================================================================================
//
// This module transforms raw user inputs into an optimized "computation
// blueprint." It now uses a low-memory, high-throughput streaming merge-join
// algorithm to handle genome-scale data.

use crate::score::io::{TextSource, open_text_source};
use crate::score::pipeline::PipelineError;
use crate::score::reformat;
use crate::score::types::{
    BimRowIndex, FilesetBoundary, GenomicRegion, GroupedComplexRule, PersonSubset, PipelineKind,
    PreparationResult, ScoreColumnIndex, ScoreInfo, parse_chromosome_label,
};
use ahash::{AHashMap, AHashSet};
use bumpalo::Bump;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::str::Utf8Error;
use std::time::Instant;

// The number of SIMD lanes in the kernel. This MUST be kept in sync with kernel.rs.
const LANE_COUNT: usize = 8;

// ========================================================================================
//              Type-driven domain model for streaming
// ========================================================================================

/// The primitive, sortable key used for all merge-join operations.
type VariantKey = (u8, u32);

// --- Zero-Copy Internal Data Structures ---
// These temporary structs hold string data borrowed from an arena, avoiding
// allocations in the hot path. Their lifetime `'arena` is tied to the `Bump`
// arena created at the start of `prepare_for_computation`.

#[derive(Clone)]
struct FilesetPaths {
    bed: PathBuf,
    bim: PathBuf,
    fam: PathBuf,
}

/// A parsed record from a `.bim` file, holding borrowed string slices.
#[derive(Debug, Copy, Clone)]
struct KeyedBimRecord<'arena> {
    key: VariantKey,
    bim_row_index: BimRowIndex,
    allele1: &'arena str,
    allele2: &'arena str,
}

/// A parsed record from a score file, holding a borrowed string slice.
#[derive(Debug, Copy, Clone)]
struct KeyedScoreRecord<'arena> {
    key: VariantKey,
    effect_allele: &'arena str,
    other_allele: &'arena str,
    score_column_index: ScoreColumnIndex,
    weight: f32,
}

// Manual implementation to handle f32 comparison correctly.
impl<'arena> PartialEq for KeyedScoreRecord<'arena> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.weight.to_bits() == other.weight.to_bits()
    }
}
impl<'arena> Eq for KeyedScoreRecord<'arena> {}

/// A self-contained item for the merge heap.
#[derive(Debug, Copy, Clone)]
struct HeapItem<'arena> {
    record: KeyedScoreRecord<'arena>,
    file_idx: usize,
}

impl<'arena> PartialEq for HeapItem<'arena> {
    fn eq(&self, other: &Self) -> bool {
        self.record == other.record
    }
}

impl<'arena> Eq for HeapItem<'arena> {}

impl<'arena> PartialOrd for HeapItem<'arena> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'arena> Ord for HeapItem<'arena> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want a min-heap, so we reverse the comparison on the key.
        // Tie-break by file_idx to ensure deterministic order.
        other
            .record
            .key
            .cmp(&self.record.key)
            .then_with(|| other.file_idx.cmp(&self.file_idx))
    }
}

/// Manages the state of a single file reader in the KWayMergeIterator.
struct FileStream<'arena> {
    reader: BufReader<File>,
    /// A buffer for weights and column indices from the current line being processed.
    line_buffer: std::collections::VecDeque<(f32, ScoreColumnIndex)>,
    /// The key and alleles (borrowed from the arena) for the current buffered line.
    current_line_info: Option<(VariantKey, &'arena str, &'arena str)>,
    // This is a temporary buffer for reading lines from the file before they
    // are allocated into the long-lived arena.
    line_string_buffer: String,
    /// Counter for malformed lines in this specific file stream.
    malformed_lines_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineReadOutcome {
    Pushed,
    Skipped,
    Eof,
}

/// An iterator that merges multiple, sorted score files on the fly.
struct KWayMergeIterator<'arena> {
    streams: Vec<FileStream<'arena>>,
    heap: BinaryHeap<HeapItem<'arena>>,
    file_column_maps: Vec<Vec<ScoreColumnIndex>>,
    // Holds a terminal error. If Some, iteration will stop after yielding the error.
    next_error: Option<PrepError>,
    region_filters: Option<Vec<Option<GenomicRegion>>>,
    region_filter_hits: Option<Vec<bool>>,
    // A reference to the memory arena for zero-copy string allocations.
    bump: &'arena Bump,
}

/// An iterator that streams over one or more `.bim` files.
struct BimIterator<'a, 'arena> {
    filesets: std::slice::Iter<'a, FilesetPaths>,
    current_reader: Option<Box<dyn TextSource + 'a>>,
    global_offset: u64,
    local_line_num: u64,
    current_path: PathBuf,
    // A list of the file boundaries, collected on-the-fly during the single iteration pass.
    boundaries: Vec<FilesetBoundary>,
    // A reference to the memory arena.
    bump: &'arena Bump,
    total_variants: u64,
}

/// Enum to represent the outcome of reconciling one score file line.
enum ReconciliationOutcome<'a, 'arena> {
    Simple(Vec<&'a KeyedBimRecord<'arena>>),
    Complex(Vec<&'a KeyedBimRecord<'arena>>),
    NotFound,
}

// ========================================================================================
//                                  Public API
// ========================================================================================

/// A struct to hold all necessary information to debug a merge-join failure.
/// It is virtually zero-cost in the success case.
#[derive(Debug, Default)]
pub struct MergeDiagnosticInfo {
    total_bim_variants_processed: u64,
    total_score_records_processed: u64,
    // Store the last few raw keys we saw from each stream. Formatting is deferred.
    last_bim_keys_seen: std::collections::VecDeque<VariantKey>,
    last_score_keys_seen: std::collections::VecDeque<VariantKey>,
    /// Track any region filters supplied by the user so we can surface them in diagnostics.
    active_region_filters: Vec<(String, GenomicRegion)>,
    region_filters_without_hits: Vec<String>,
}

/// The number of recent keys to store for diagnostic reporting.
const DIAGNOSTIC_BUFFER_SIZE: usize = 10;

impl MergeDiagnosticInfo {
    /// Adds a key to the circular buffer. This is a very cheap operation.
    fn add_bim_key(&mut self, key: VariantKey) {
        if self.last_bim_keys_seen.len() == DIAGNOSTIC_BUFFER_SIZE {
            self.last_bim_keys_seen.pop_front();
        }
        self.last_bim_keys_seen.push_back(key);
    }

    /// Adds a key to the circular buffer for score file keys.
    fn add_score_key(&mut self, key: VariantKey) {
        if self.last_score_keys_seen.len() == DIAGNOSTIC_BUFFER_SIZE {
            self.last_score_keys_seen.pop_front();
        }
        self.last_score_keys_seen.push_back(key);
    }

    fn record_region_filters(&mut self, filters: &HashMap<String, GenomicRegion>) {
        if filters.is_empty() {
            return;
        }

        self.active_region_filters = filters
            .iter()
            .map(|(name, region)| (name.clone(), *region))
            .collect();
        self.active_region_filters
            .sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }

    fn record_region_without_hits(&mut self, score_name: &str) {
        self.region_filters_without_hits
            .push(score_name.to_string());
    }
}

#[derive(Debug)]
pub enum PrepError {
    Io(io::Error, PathBuf),
    Parse(String),
    Header(String),
    InconsistentKeepId(String),
    PipelineIo {
        path: PathBuf,
        message: String,
    },
    /// An error indicating that no variants from the score files could be matched
    /// to variants in the genotype data.
    NoOverlappingVariants(MergeDiagnosticInfo),
    UnsortedInput {
        source: &'static str,
        path: PathBuf,
        line_number: u64,
        previous_key: VariantKey,
        current_key: VariantKey,
    },
    GenomeBuildMismatch,
    DisjointChromosomes,
    AmbiguousReconciliation(String),
}

enum PostMortemAction {
    None,
    Fatal(PrepError),
    SortAndRetry {
        fileset: FilesetPaths,
        unsorted_error: PrepError,
    },
}

pub fn prepare_for_computation(
    fileset_prefixes: &[PathBuf],
    sorted_score_files: &[PathBuf],
    keep_file: Option<&Path>,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
) -> Result<PreparationResult, PrepError> {
    let max_sort_retries = fileset_prefixes.len().max(1);
    prepare_for_computation_with_retry(
        fileset_prefixes,
        sorted_score_files,
        keep_file,
        score_regions,
        max_sort_retries,
    )
}

fn prepare_for_computation_with_retry(
    fileset_prefixes: &[PathBuf],
    sorted_score_files: &[PathBuf],
    keep_file: Option<&Path>,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    remaining_sort_retries: usize,
) -> Result<PreparationResult, PrepError> {
    // --- Stage 1: Initial setup ---
    eprintln!("> Stage 1: Indexing subject data...");
    let fileset_paths = build_fileset_paths(fileset_prefixes)?;
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fileset_paths)?;
    let total_people_in_fam = all_person_iids.len();

    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();

    // --- Stage 2: Global metadata discovery ---
    eprintln!("> Stage 2: Discovering all score columns...");
    let score_names = parse_score_file_headers_only(sorted_score_files)?;
    let score_name_to_col_index: AHashMap<String, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), ScoreColumnIndex(i)))
        .collect();

    // --- Stage 3: Single-pass data collection ---
    eprintln!("> Stage 3: Streaming and collecting data from all input files...");
    let overall_start_time = Instant::now();

    // The `bump` arena is used for all temporary string allocations during the
    // merge-join. This is the core of the zero-copy optimization.
    let bump = Bump::new();

    let mut diagnostics = MergeDiagnosticInfo::default();
    if let Some(regions) = score_regions {
        diagnostics.record_region_filters(regions);
    }
    let mut seen_invalid_bim_chrs: AHashSet<String> = AHashSet::new();
    let mut seen_invalid_score_chrs: AHashSet<String> = AHashSet::new();

    // Create the iterators, giving them a reference to the arena.
    let mut bim_iterator = BimIterator::new(&fileset_paths, &bump)?;
    let region_filters = score_regions.and_then(|regions| {
        let mut has_any = false;
        let mut filters = Vec::with_capacity(score_names.len());
        for name in &score_names {
            let region = regions.get(name).copied();
            if region.is_some() {
                has_any = true;
            }
            filters.push(region);
        }
        has_any.then_some(filters)
    });

    let mut score_iterator = KWayMergeIterator::new(
        sorted_score_files,
        &score_name_to_col_index,
        region_filters.clone(),
        &bump,
    )?;

    let mut bim_iter = bim_iterator.by_ref().peekable();
    let mut score_iter = score_iterator.by_ref().peekable();

    // These data structures will store the results of the merge-join.
    // Allele strings borrowed from the arena are stored here temporarily.
    let mut simple_path_data: BTreeMap<BimRowIndex, BTreeMap<ScoreColumnIndex, (f32, bool)>> =
        BTreeMap::new();
    // For complex rules, the key is the set of BIM contexts. The value is a tuple
    // containing the canonical chr:pos key for the locus and all score applications.
    let mut intermediate_complex_rules: BTreeMap<
        Vec<(BimRowIndex, &str, &str)>,
        (VariantKey, Vec<(ScoreColumnIndex, f32, &str, &str)>),
    > = BTreeMap::new();

    while bim_iter.peek().is_some() && score_iter.peek().is_some() {
        let bim_key = match bim_iter.peek().unwrap() {
            Ok(rec) => rec.key,
            Err(_) => match bim_iter.next().unwrap().unwrap_err() {
                PrepError::Parse(msg) => {
                    if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                        && seen_invalid_bim_chrs.insert(chr_name.to_string())
                    {
                        eprintln!(
                            "Warning: Skipping variant(s) in BIM file due to unparsable chromosome name: '{chr_name}'."
                        );
                    }
                    continue;
                }
                e => return Err(e),
            },
        };

        let score_key = match score_iter.peek().unwrap() {
            Ok(rec) => rec.key,
            Err(_) => match score_iter.next().unwrap().unwrap_err() {
                PrepError::Parse(msg) => {
                    if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                        && seen_invalid_score_chrs.insert(chr_name.to_string())
                    {
                        eprintln!(
                            "Warning: Skipping variant(s) in score file due to unparsable chromosome name: '{chr_name}'."
                        );
                    }
                    continue;
                }
                e => return Err(e),
            },
        };

        match bim_key.cmp(&score_key) {
            Ordering::Less => {
                diagnostics.add_bim_key(bim_key);
                diagnostics.total_bim_variants_processed += 1;
                bim_iter.next();
            }
            Ordering::Greater => {
                diagnostics.add_score_key(score_key);
                diagnostics.total_score_records_processed += 1;
                score_iter.next();
            }
            Ordering::Equal => {
                let key = bim_key;
                diagnostics.add_bim_key(key);
                diagnostics.add_score_key(key);

                let mut bim_group = Vec::new();
                while let Some(Ok(peek_item)) = bim_iter.peek() {
                    if peek_item.key != key {
                        break;
                    }
                    match bim_iter.next().unwrap() {
                        Ok(item) => bim_group.push(item),
                        Err(PrepError::Parse(msg)) => {
                            if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                                && seen_invalid_bim_chrs.insert(chr_name.to_string())
                            {
                                eprintln!(
                                    "Warning: Skipping variant(s) in BIM file due to unparsable chromosome name: '{chr_name}'."
                                );
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                diagnostics.total_bim_variants_processed += bim_group.len() as u64;

                let mut score_group = Vec::new();
                while let Some(Ok(peek_item)) = score_iter.peek() {
                    if peek_item.key != key {
                        break;
                    }
                    match score_iter.next().unwrap() {
                        Ok(item) => score_group.push(item),
                        Err(PrepError::Parse(msg)) => {
                            if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                                && seen_invalid_score_chrs.insert(chr_name.to_string())
                            {
                                eprintln!(
                                    "Warning: Skipping variant(s) in score file due to unparsable chromosome name: '{chr_name}'."
                                );
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                diagnostics.total_score_records_processed += score_group.len() as u64;

                for score_record in score_group {
                    let outcome = resolve_matches_for_score_line(score_record, &bim_group)?;

                    match outcome {
                        ReconciliationOutcome::Simple(matches) => {
                            for bim_rec in matches {
                                let is_flipped = score_record.effect_allele == bim_rec.allele1;
                                let score_map =
                                    simple_path_data.entry(bim_rec.bim_row_index).or_default();

                                if score_map
                                    .insert(
                                        score_record.score_column_index,
                                        (score_record.weight, is_flipped),
                                    )
                                    .is_some()
                                {
                                    return Err(PrepError::AmbiguousReconciliation(format!(
                                        "Ambiguous input: The same variant-score pair is defined with different weights across input files. Variant BIM index: {}, Score: {}",
                                        bim_rec.bim_row_index.0,
                                        score_names[score_record.score_column_index.0]
                                    )));
                                }
                            }
                        }
                        ReconciliationOutcome::Complex(matches) => {
                            let possible_contexts: Vec<_> = matches
                                .iter()
                                .map(|rec| (rec.bim_row_index, rec.allele1, rec.allele2))
                                .collect();
                            let score_info = (
                                score_record.score_column_index,
                                score_record.weight,
                                score_record.effect_allele,
                                score_record.other_allele,
                            );

                            // The key for the map is the set of BIM contexts. On first
                            // encounter, we store the canonical chr:pos key. Then we
                            // append the score application.
                            let entry = intermediate_complex_rules
                                .entry(possible_contexts)
                                .or_insert_with(|| (key, Vec::new()));

                            // The value is a tuple: (key, scores_vec), so we push to entry.1
                            entry.1.push(score_info);
                        }
                        ReconciliationOutcome::NotFound => {}
                    }
                }
            }
        }
    }
    while let Some(result) = bim_iter.next() {
        match result {
            Ok(record) => {
                diagnostics.add_bim_key(record.key);
                diagnostics.total_bim_variants_processed += 1;
            }
            Err(PrepError::Parse(msg)) => {
                if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                    && seen_invalid_bim_chrs.insert(chr_name.to_string())
                {
                    eprintln!(
                        "Warning: Skipping variant(s) in BIM file due to unparsable chromosome name: '{chr_name}'."
                    );
                }
            }
            Err(e) => return Err(e),
        }
    }

    drop(bim_iter);
    drop(score_iter);

    let region_filter_hits = score_iterator.take_region_filter_hits();
    if let (Some(filters), Some(hit_flags)) = (region_filters.as_ref(), region_filter_hits.as_ref())
    {
        for (idx, region_opt) in filters.iter().enumerate() {
            if let Some(region) = region_opt {
                if !hit_flags.get(idx).copied().unwrap_or(false) {
                    let score_name = &score_names[idx];
                    eprintln!(
                        "Warning: Score '{score_name}' has no variants within the requested region {region}."
                    );
                    diagnostics.record_region_without_hits(score_name);
                }
            }
        }
    }

    let total_variants_in_bim = bim_iterator.total_variants();
    eprintln!(
        "> TIMING: Stage 3 (Data Collection) took {:.2?}",
        overall_start_time.elapsed()
    );

    // After iterating, sum malformed line counts from all streams
    let total_malformed_lines: usize = score_iterator
        .streams
        .iter()
        .map(|s| s.malformed_lines_count)
        .sum();

    if total_malformed_lines > 0 {
        eprintln!(
            "> Warning: Skipped {total_malformed_lines} lines from score files due to missing columns (variant_id, effect_allele, other_allele)."
        );
    }

    // --- Stage 4: In-memory processing and matrix construction ---
    eprintln!("> Stage 4: Verifying data and building final matrices...");

    // The "lifetime escape hatch": convert the temporary, borrowed complex rule data
    // into final, owned `GroupedComplexRule` structs. This is the only place where
    // we perform bulk copies of the allele strings we collected.
    let final_complex_rules: Vec<GroupedComplexRule> = intermediate_complex_rules
        .into_iter()
        .map(|(contexts, (variant_key, scores))| {
            // Convert the numeric chromosome key back into a string representation.
            let chr_str = match variant_key.0 {
                23 => "X".to_string(),
                24 => "Y".to_string(),
                25 => "MT".to_string(),
                n => n.to_string(),
            };

            GroupedComplexRule {
                locus_chr_pos: (chr_str, variant_key.1),
                possible_contexts: contexts
                    .into_iter()
                    .map(|(idx, a1, a2)| (idx, a1.to_string(), a2.to_string()))
                    .collect(),
                score_applications: scores
                    .into_iter()
                    .map(|(sc_idx, weight, ea, oa)| ScoreInfo {
                        effect_allele: ea.to_string(),
                        other_allele: oa.to_string(),
                        weight,
                        score_column_index: sc_idx,
                    })
                    .collect(),
            }
        })
        .collect();

    let mut all_required_indices: BTreeSet<BimRowIndex> = BTreeSet::new();
    all_required_indices.extend(simple_path_data.keys());
    let mut complex_bim_indices: AHashSet<BimRowIndex> = AHashSet::new();
    for rule in &final_complex_rules {
        for (bim_idx, _, _) in &rule.possible_contexts {
            all_required_indices.insert(*bim_idx);
            complex_bim_indices.insert(*bim_idx);
        }
    }

    if all_required_indices.is_empty() {
        match conduct_post_mortem(&fileset_paths, sorted_score_files)? {
            PostMortemAction::Fatal(err) => return Err(err),
            PostMortemAction::SortAndRetry {
                fileset,
                unsorted_error,
            } => {
                if remaining_sort_retries == 0 {
                    return Err(unsorted_error);
                }

                let sorted_bed_path =
                    reformat::sort_plink_fileset(&fileset.bed, &fileset.bim, &fileset.fam)
                        .map_err(|e| PrepError::PipelineIo {
                            path: fileset.bed.clone(),
                            message: e.to_string(),
                        })?;

                eprintln!(
                    "> Detected unsorted genotype data in {}. Sorting into {} and retrying...",
                    fileset.bed.display(),
                    sorted_bed_path.display()
                );

                let mut new_prefixes = fileset_prefixes.to_vec();
                if let Some(idx) = fileset_paths.iter().position(|fs| {
                    fs.bed == fileset.bed && fs.bim == fileset.bim && fs.fam == fileset.fam
                }) {
                    new_prefixes[idx] = sorted_bed_path;
                    return prepare_for_computation_with_retry(
                        &new_prefixes,
                        sorted_score_files,
                        keep_file,
                        score_regions,
                        remaining_sort_retries - 1,
                    );
                }

                return Err(unsorted_error);
            }
            PostMortemAction::None => return Err(PrepError::NoOverlappingVariants(diagnostics)),
        }
    }

    let required_bim_indices: Vec<BimRowIndex> = all_required_indices.into_iter().collect();
    let num_reconciled_variants = required_bim_indices.len();
    let required_is_complex: Vec<u8> = required_bim_indices
        .iter()
        .map(|idx| u8::from(complex_bim_indices.contains(idx)))
        .collect();

    let stride = score_names.len().div_ceil(LANE_COUNT) * LANE_COUNT;
    let mut weights_matrix = vec![0.0f32; num_reconciled_variants * stride];
    let mut flip_mask_matrix = vec![0u8; num_reconciled_variants * stride];
    let mut variant_to_scores_map: Vec<Vec<ScoreColumnIndex>> =
        vec![Vec::new(); num_reconciled_variants];

    weights_matrix
        .par_chunks_mut(stride)
        .zip(flip_mask_matrix.par_chunks_mut(stride))
        .zip(variant_to_scores_map.par_iter_mut())
        .enumerate()
        .for_each(
            |(reconciled_idx, ((weights_chunk, flip_chunk), vtsm_entry))| {
                let bim_row_index = required_bim_indices[reconciled_idx];
                if let Some(score_data_map) = simple_path_data.get(&bim_row_index) {
                    for (&score_col_idx, &(weight, is_flipped)) in score_data_map.iter() {
                        weights_chunk[score_col_idx.0] = weight;
                        flip_chunk[score_col_idx.0] = if is_flipped { 1 } else { 0 };
                    }
                    *vtsm_entry = score_data_map.keys().copied().collect();
                }
            },
        );

    let mut score_variant_counts = vec![0u32; score_names.len()];
    for scores in &variant_to_scores_map {
        for score_col in scores {
            score_variant_counts[score_col.0] += 1;
        }
    }
    for rule in &final_complex_rules {
        let mut counted_cols: AHashSet<ScoreColumnIndex> = AHashSet::new();
        for score_info in &rule.score_applications {
            counted_cols.insert(score_info.score_column_index);
        }
        for score_col in counted_cols {
            score_variant_counts[score_col.0] += 1;
        }
    }

    // --- Stage 5: Final assembly ---
    let bytes_per_variant = (total_people_in_fam as u64).div_ceil(4);
    let bytes_per_variant_usize = bytes_per_variant as usize;
    let (spool_compact_byte_index, spool_dense_map) =
        build_spool_maps(&person_subset, bytes_per_variant_usize);
    let spool_bytes_per_variant = spool_compact_byte_index.len() as u64;
    debug_assert_eq!(
        spool_bytes_per_variant as usize,
        spool_compact_byte_index.len(),
        "spool bytes per variant must equal compact index length"
    );
    let mut output_idx_to_fam_idx = Vec::with_capacity(num_people_to_score);
    let mut person_fam_to_output_idx = vec![None; total_people_in_fam];

    for (output_idx, iid) in final_person_iids.iter().enumerate() {
        let original_fam_idx = *iid_to_original_idx.get(iid).unwrap();
        output_idx_to_fam_idx.push(original_fam_idx);
        person_fam_to_output_idx[original_fam_idx as usize] = Some(output_idx as u32);
    }

    let pipeline_kind = if fileset_paths.len() <= 1 {
        PipelineKind::SingleFile(fileset_paths[0].bed.clone())
    } else {
        PipelineKind::MultiFile(bim_iterator.boundaries)
    };

    Ok(PreparationResult::new(
        weights_matrix,
        flip_mask_matrix,
        stride,
        required_bim_indices,
        final_complex_rules,
        score_names,
        score_variant_counts,
        variant_to_scores_map,
        person_subset,
        final_person_iids,
        num_people_to_score,
        total_people_in_fam,
        total_variants_in_bim,
        num_reconciled_variants,
        bytes_per_variant,
        person_fam_to_output_idx,
        output_idx_to_fam_idx,
        required_is_complex,
        spool_compact_byte_index,
        spool_dense_map,
        spool_bytes_per_variant,
        pipeline_kind,
    ))
}

// ========================================================================================
//                             Private implementation helpers
// ========================================================================================

fn build_fileset_paths(prefixes: &[PathBuf]) -> Result<Vec<FilesetPaths>, PrepError> {
    prefixes
        .iter()
        .map(|prefix| {
            let bed = apply_extension(prefix, "bed")?;
            let bim = apply_extension(prefix, "bim")?;
            let fam = apply_extension(prefix, "fam")?;
            Ok(FilesetPaths { bed, bim, fam })
        })
        .collect()
}

/// Builds the compacted spool index structures for the selected cohort subset.
///
/// The returned `Vec<u32>` is guaranteed to be sorted and unique, providing the
/// exact byte positions (in the original PLINK layout) that contain at least one
/// kept individual. The dense map mirrors the original byte indices and either
/// contains the compacted index or `-1` if no kept person resides in that byte.
fn build_spool_maps(
    person_subset: &PersonSubset,
    bytes_per_variant_usize: usize,
) -> (Vec<u32>, Vec<i32>) {
    match person_subset {
        PersonSubset::All => {
            let mut compact = Vec::with_capacity(bytes_per_variant_usize);
            let mut dense = Vec::with_capacity(bytes_per_variant_usize);
            for i in 0..bytes_per_variant_usize {
                compact.push(
                    u32::try_from(i).expect(
                        "Too many bytes per variant to represent in spool_compact_byte_index",
                    ),
                );
                dense.push(
                    i32::try_from(i)
                        .expect("Too many bytes per variant to represent in spool_dense_map"),
                );
            }
            (compact, dense)
        }
        PersonSubset::Indices(indices) => {
            let mut unique_bytes: BTreeSet<u32> = BTreeSet::new();
            for &fam_idx in indices {
                unique_bytes.insert(fam_idx / 4);
            }
            let compact: Vec<u32> = unique_bytes.into_iter().collect();
            let mut dense = vec![-1i32; bytes_per_variant_usize];
            for (compact_idx, &orig_byte_idx) in compact.iter().enumerate() {
                if let Some(slot) = dense.get_mut(orig_byte_idx as usize) {
                    *slot = i32::try_from(compact_idx)
                        .expect("Too many kept individuals to compact into spool_dense_map");
                }
            }
            (compact, dense)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_spool_maps_all_people_identity_mapping() {
        let bytes_per_variant = 3;
        let (compact, dense) = build_spool_maps(&PersonSubset::All, bytes_per_variant);
        assert_eq!(compact, vec![0, 1, 2]);
        assert_eq!(dense, vec![0, 1, 2]);
    }

    #[test]
    fn build_spool_maps_subset_compacts_sorted_unique_bytes() {
        let indices = vec![0, 1, 8, 9];
        let bytes_per_variant = 4; // enough room for indices up to 3
        let (compact, dense) =
            build_spool_maps(&PersonSubset::Indices(indices.clone()), bytes_per_variant);

        // The compact list must stay sorted and unique regardless of input order.
        assert_eq!(compact, vec![0, 2]);
        assert!(compact.windows(2).all(|w| w[0] < w[1]));

        // Dense map should point to the compact slots for kept bytes and -1 elsewhere.
        assert_eq!(dense.len(), bytes_per_variant);
        assert_eq!(dense[0], 0);
        assert_eq!(dense[1], -1);
        assert_eq!(dense[2], 1);
        assert_eq!(dense[3], -1);

        // Every kept fam index should resolve to a compact entry.
        for fam_idx in indices {
            let orig_byte = (fam_idx / 4) as usize;
            assert!(dense[orig_byte] >= 0);
        }
    }

    #[test]
    fn kway_merge_continues_after_filtered_lines() {
        use std::io::Write;

        let dir = tempfile::tempdir().expect("tempdir");
        let score_path = dir.path().join("score.tsv");
        let mut file = std::fs::File::create(&score_path).expect("create score");
        writeln!(file, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write header");
        writeln!(file, "1:100\tA\tG\t0.5").expect("write first line");
        writeln!(file, "1:150\tC\tT\t0.7").expect("write second line");
        writeln!(file, "1:180\tC\tT\t0.2").expect("write third line");
        drop(file);

        let mut score_name_to_col_index = AHashMap::new();
        score_name_to_col_index.insert("ScoreA".to_string(), ScoreColumnIndex(0));

        let region = GenomicRegion {
            chromosome: 1,
            start: 140,
            end: 200,
        };
        let bump = Bump::new();
        let file_paths = vec![score_path];
        let mut iter = KWayMergeIterator::new(
            &file_paths,
            &score_name_to_col_index,
            Some(vec![Some(region)]),
            &bump,
        )
        .expect("iterator");

        let mut keys = Vec::new();
        while let Some(result) = iter.next() {
            let record = result.expect("record ok");
            keys.push(record.key);
        }

        assert_eq!(keys, vec![(1, 150), (1, 180)]);
    }
}

fn apply_extension(path: &Path, extension: &str) -> Result<PathBuf, PrepError> {
    if let Some(path_str) = path.to_str() {
        if path_str.starts_with("gs://") {
            let mut new_path = path_str.to_string();
            if let Some(dot_pos) = new_path.rfind('.') {
                let slash_pos = new_path.rfind('/');
                if slash_pos.map_or(true, |idx| idx < dot_pos) {
                    new_path.truncate(dot_pos);
                    new_path.push('.');
                    new_path.push_str(extension);
                    return Ok(PathBuf::from(new_path));
                }
            }
            new_path.push('.');
            new_path.push_str(extension);
            Ok(PathBuf::from(new_path))
        } else {
            Ok(path.with_extension(extension))
        }
    } else {
        Err(PrepError::Parse("Invalid UTF-8 in path".to_string()))
    }
}

fn map_pipeline_error(err: PipelineError, path: PathBuf) -> PrepError {
    PrepError::PipelineIo {
        path,
        message: err.to_string(),
    }
}

/// Extracts the malformed chromosome name from a `PrepError::Parse` message.
fn extract_chr_from_parse_error(msg: &str) -> Option<&str> {
    if let Some(rest) = msg.strip_prefix("Invalid chromosome format '")
        && let Some(end_pos) = rest.find('\'')
    {
        return Some(&rest[..end_pos]);
    }
    None
}

fn parse_key(chr_str: &str, pos_str: &str) -> Result<(u8, u32), PrepError> {
    let chr_num = parse_chromosome_label(chr_str).map_err(PrepError::Parse)?;
    let pos_trimmed = pos_str.trim();
    let pos_num: u32 = pos_trimmed
        .parse()
        .map_err(|e| PrepError::Parse(format!("Invalid position '{pos_str}': {e}")))?;

    Ok((chr_num, pos_num))
}

fn conduct_post_mortem(
    fileset_paths: &[FilesetPaths],
    score_files: &[PathBuf],
) -> Result<PostMortemAction, PrepError> {
    let mut bim_chromosomes: AHashSet<u8> = AHashSet::new();
    let mut score_chromosomes: AHashSet<u8> = AHashSet::new();

    let post_mortem_bump = Bump::new();
    let mut bim_iter = BimIterator::new(fileset_paths, &post_mortem_bump)?;
    let mut previous_bim_key: Option<VariantKey> = None;

    while let Some(next_item) = bim_iter.next() {
        match next_item {
            Ok(record) => {
                let current_key = record.key;
                bim_chromosomes.insert(current_key.0);

                if let Some(prev_key) = previous_bim_key {
                    if current_key < prev_key {
                        let unsorted_error = PrepError::UnsortedInput {
                            source: "BIM",
                            path: bim_iter.current_path.clone(),
                            line_number: bim_iter.local_line_num,
                            previous_key: prev_key,
                            current_key,
                        };

                        if let Some(fileset) = fileset_paths
                            .iter()
                            .find(|fs| fs.bim == bim_iter.current_path)
                        {
                            return Ok(PostMortemAction::SortAndRetry {
                                fileset: fileset.clone(),
                                unsorted_error,
                            });
                        }

                        return Ok(PostMortemAction::Fatal(unsorted_error));
                    }
                }

                previous_bim_key = Some(current_key);
            }
            Err(PrepError::Parse(_)) => continue,
            Err(err) => return Err(err),
        }
    }

    for path in score_files {
        let file = File::open(path).map_err(|e| PrepError::Io(e, path.clone()))?;
        let reader = BufReader::new(file);
        let mut previous_key: Option<VariantKey> = None;

        // Score files are already normalized by `reformat_pgs_file`, which guarantees
        // the tab-separated layout: variant_id, effect_allele, other_allele, weight....
        // The quick parser below intentionally relies on that invariant to keep this
        // post-mortem check minimal and fast to implement.
        for (line_index, line_result) in reader.lines().enumerate() {
            let line_number = line_index as u64 + 1;
            let line = line_result.map_err(|e| PrepError::Io(e, path.clone()))?;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.split('\t');
            let variant_id = match parts.next() {
                Some(id) if !id.is_empty() => id,
                _ => continue,
            };

            if variant_id.eq_ignore_ascii_case("variant_id") {
                continue;
            }

            let effect_allele = parts.next();
            let other_allele = parts.next();

            if effect_allele.is_none()
                || other_allele.is_none()
                || effect_allele.is_some_and(|a| a.is_empty())
                || other_allele.is_some_and(|a| a.is_empty())
            {
                continue;
            }

            let mut key_parts = variant_id.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");
            let key = parse_key(chr_str, pos_str)?;

            score_chromosomes.insert(key.0);

            if let Some(prev_key) = previous_key {
                if key < prev_key {
                    return Ok(PostMortemAction::Fatal(PrepError::UnsortedInput {
                        source: "score",
                        path: path.clone(),
                        line_number,
                        previous_key: prev_key,
                        current_key: key,
                    }));
                }
            }

            previous_key = Some(key);
        }
    }

    let has_bim_chromosomes = !bim_chromosomes.is_empty();
    let has_score_chromosomes = !score_chromosomes.is_empty();

    if has_bim_chromosomes
        && has_score_chromosomes
        && bim_chromosomes
            .iter()
            .any(|chr| score_chromosomes.contains(chr))
    {
        return Ok(PostMortemAction::Fatal(PrepError::GenomeBuildMismatch));
    }

    if has_bim_chromosomes && has_score_chromosomes {
        return Ok(PostMortemAction::Fatal(PrepError::DisjointChromosomes));
    }

    Ok(PostMortemAction::None)
}

impl<'a, 'arena> BimIterator<'a, 'arena> {
    fn new(filesets: &'a [FilesetPaths], bump: &'arena Bump) -> Result<Self, PrepError> {
        let mut iter = Self {
            filesets: filesets.iter(),
            current_reader: None,
            global_offset: 0,
            local_line_num: 0,
            current_path: PathBuf::new(),
            boundaries: Vec::with_capacity(filesets.len()),
            bump,
            total_variants: 0,
        };
        iter.next_file()?;
        Ok(iter)
    }

    fn next_file(&mut self) -> Result<bool, PrepError> {
        self.global_offset += self.local_line_num;
        self.local_line_num = 0;

        if let Some(fileset) = self.filesets.next() {
            self.boundaries.push(FilesetBoundary {
                bed_path: fileset.bed.clone(),
                bim_path: fileset.bim.clone(),
                fam_path: fileset.fam.clone(),
                starting_global_index: self.global_offset,
            });

            self.current_path = fileset.bim.clone();
            let reader = open_text_source(&fileset.bim)
                .map_err(|e| map_pipeline_error(e, fileset.bim.clone()))?;
            self.current_reader = Some(reader);
            Ok(true)
        } else {
            self.current_reader = None;
            self.total_variants = self.global_offset;
            Ok(false)
        }
    }

    fn total_variants(&self) -> u64 {
        self.total_variants
    }
}

impl<'a, 'arena> Iterator for BimIterator<'a, 'arena> {
    type Item = Result<KeyedBimRecord<'arena>, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let reader = match self.current_reader.as_mut() {
                Some(reader) => reader,
                None => return None,
            };

            match reader.next_line() {
                Ok(Some(line_bytes)) => {
                    self.local_line_num += 1;
                    self.total_variants = self.global_offset + self.local_line_num;
                    let line_str = match std::str::from_utf8(line_bytes) {
                        Ok(s) => s,
                        Err(e) => {
                            return Some(Err(PrepError::Parse(format!(
                                "Invalid UTF-8 in BIM file '{}': {e}",
                                self.current_path.display()
                            ))));
                        }
                    };

                    let line_in_arena = self.bump.alloc_str(line_str);
                    let mut parts = line_in_arena.split_whitespace();
                    let chr = parts.next();
                    parts.next();
                    parts.next();
                    let pos = parts.next();
                    let a1 = parts.next();
                    let a2 = parts.next();

                    if let (Some(chr_str), Some(pos_str), Some(a1), Some(a2)) = (chr, pos, a1, a2) {
                        match parse_key(chr_str, pos_str) {
                            Ok(key) => {
                                return Some(Ok(KeyedBimRecord {
                                    key,
                                    bim_row_index: BimRowIndex(
                                        self.global_offset + self.local_line_num - 1,
                                    ),
                                    allele1: a1,
                                    allele2: a2,
                                }));
                            }
                            Err(e) => return Some(Err(e)),
                        }
                    }
                }
                Ok(None) => {
                    if let Ok(false) = self.next_file() {
                        return None;
                    }
                }
                Err(err) => {
                    return Some(Err(map_pipeline_error(err, self.current_path.clone())));
                }
            }
        }
    }
}

impl<'arena> KWayMergeIterator<'arena> {
    fn new(
        file_paths: &[PathBuf],
        score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
        region_filters: Option<Vec<Option<GenomicRegion>>>,
        bump: &'arena Bump,
    ) -> Result<Self, PrepError> {
        let mut streams = Vec::with_capacity(file_paths.len());
        let mut file_column_maps = Vec::with_capacity(file_paths.len());

        for path in file_paths {
            let file = File::open(path).map_err(|e| PrepError::Io(e, path.clone()))?;
            let mut reader = BufReader::new(file);
            let mut header_line = String::new();

            loop {
                header_line.clear();
                if reader
                    .read_line(&mut header_line)
                    .map_err(|e| PrepError::Io(e, path.clone()))?
                    == 0
                {
                    break;
                }
                if !header_line.starts_with("##") {
                    break;
                }
            }

            let column_map: Vec<ScoreColumnIndex> = header_line
                .trim()
                .split('\t')
                .skip(3)
                .map(|name| {
                    score_name_to_col_index.get(name).copied().ok_or_else(|| {
                        PrepError::Header(format!(
                            "Score '{name}' from file '{path}' not found in global score list.",
                            name = name,
                            path = path.display()
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;

            file_column_maps.push(column_map);

            streams.push(FileStream {
                reader,
                line_buffer: std::collections::VecDeque::new(),
                current_line_info: None,
                line_string_buffer: String::new(),
                malformed_lines_count: 0,
            });
        }

        let region_filter_hits = region_filters
            .as_ref()
            .map(|_| vec![false; score_name_to_col_index.len()]);

        let mut iter = Self {
            streams,
            heap: BinaryHeap::new(),
            file_column_maps,
            next_error: None,
            region_filters,
            region_filter_hits,
            bump,
        };

        for i in 0..iter.streams.len() {
            iter.replenish_from_stream(i)?
        }

        Ok(iter)
    }

    fn replenish_from_stream(&mut self, file_idx: usize) -> Result<(), PrepError> {
        let column_map = &self.file_column_maps[file_idx];
        let bump = self.bump;

        if self.region_filters.is_none() {
            loop {
                let outcome = {
                    let stream = &mut self.streams[file_idx];
                    if !stream.line_buffer.is_empty() {
                        Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
                        return Ok(());
                    }

                    Self::read_line_into_buffer(stream, column_map, None, None, bump)?
                };

                match outcome {
                    LineReadOutcome::Pushed => {
                        let stream = &mut self.streams[file_idx];
                        Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
                        return Ok(());
                    }
                    LineReadOutcome::Skipped => continue,
                    LineReadOutcome::Eof => return Ok(()),
                }
            }
        }

        let region_filters = self.region_filters.as_deref();
        let mut region_hits = self.region_filter_hits.take();

        loop {
            let outcome = {
                let stream = &mut self.streams[file_idx];
                if !stream.line_buffer.is_empty() {
                    Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
                    self.region_filter_hits = region_hits;
                    return Ok(());
                }

                let region_hits_slice = region_hits.as_mut().map(|vec| vec.as_mut_slice());

                Self::read_line_into_buffer(
                    stream,
                    column_map,
                    region_filters,
                    region_hits_slice,
                    bump,
                )?
            };

            match outcome {
                LineReadOutcome::Pushed => {
                    let stream = &mut self.streams[file_idx];
                    Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
                    self.region_filter_hits = region_hits;
                    return Ok(());
                }
                LineReadOutcome::Skipped => continue,
                LineReadOutcome::Eof => {
                    self.region_filter_hits = region_hits;
                    return Ok(());
                }
            }
        }
    }

    fn read_line_into_buffer(
        stream: &mut FileStream<'arena>,
        column_map: &[ScoreColumnIndex],
        region_filters: Option<&[Option<GenomicRegion>]>,
        mut region_hits: Option<&mut [bool]>,
        bump: &'arena Bump,
    ) -> Result<LineReadOutcome, PrepError> {
        stream.line_buffer.clear();
        stream.current_line_info = None;

        loop {
            stream.line_string_buffer.clear();
            let bytes_read = stream
                .reader
                .read_line(&mut stream.line_string_buffer)
                .map_err(|e| PrepError::Io(e, PathBuf::new()))?;

            if bytes_read == 0 {
                return Ok(LineReadOutcome::Eof);
            }

            if stream.line_string_buffer.trim().is_empty()
                || stream.line_string_buffer.starts_with('#')
            {
                continue;
            }

            // Use the `bump` argument, not `self.bump`.
            let line_in_arena = bump.alloc_str(&stream.line_string_buffer);

            let mut parts = line_in_arena.split('\t');
            let (variant_id, effect_allele, other_allele) =
                match (parts.next(), parts.next(), parts.next()) {
                    (Some(v), Some(e), Some(o))
                        if !v.is_empty() && !e.is_empty() && !o.is_empty() =>
                    {
                        (v, e, o)
                    } // Ensure other_allele is also not empty
                    _ => {
                        stream.malformed_lines_count += 1;
                        continue; // Line doesn't have the required three non-empty columns
                    }
                };

            let mut key_parts = variant_id.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");
            let key = parse_key(chr_str, pos_str)?;
            stream.current_line_info = Some((key, effect_allele, other_allele));

            for (i, weight_str) in parts.enumerate() {
                if let Ok(weight) = weight_str.trim().parse::<f32>()
                    && let Some(&score_column_index) = column_map.get(i)
                {
                    if let Some(filters) = region_filters {
                        if let Some(Some(region)) = filters.get(score_column_index.0) {
                            if !region.contains(key) {
                                continue;
                            }
                        }
                        if let Some(hit_flags) = region_hits.as_deref_mut() {
                            hit_flags[score_column_index.0] = true;
                        }
                    } else if let Some(hit_flags) = region_hits.as_deref_mut() {
                        hit_flags[score_column_index.0] = true;
                    }
                    stream.line_buffer.push_back((weight, score_column_index));
                }
            }
            if stream.line_buffer.is_empty() {
                stream.current_line_info = None;
                return Ok(LineReadOutcome::Skipped);
            } else {
                return Ok(LineReadOutcome::Pushed);
            }
        }
    }

    fn push_next_from_buffer_to_heap(
        stream: &mut FileStream<'arena>,
        file_idx: usize,
        heap: &mut BinaryHeap<HeapItem<'arena>>,
    ) {
        if let Some((weight, score_column_index)) = stream.line_buffer.pop_front() {
            let (key, effect_allele, other_allele) = stream.current_line_info.unwrap();
            let record = KeyedScoreRecord {
                key,
                effect_allele,
                other_allele,
                score_column_index,
                weight,
            };
            heap.push(HeapItem { record, file_idx });
        }
    }

    fn take_region_filter_hits(&mut self) -> Option<Vec<bool>> {
        self.region_filter_hits.take()
    }
}

impl<'arena> Iterator for KWayMergeIterator<'arena> {
    type Item = Result<KeyedScoreRecord<'arena>, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(e) = self.next_error.take() {
            return Some(Err(e));
        }

        let top_item = self.heap.pop()?;
        let record_to_return = top_item.record;
        let file_idx = top_item.file_idx;

        if let Err(e) = self.replenish_from_stream(file_idx) {
            self.next_error = Some(e);
        }

        Some(Ok(record_to_return))
    }
}

fn resolve_matches_for_score_line<'a, 'arena>(
    score_record: KeyedScoreRecord,
    bim_records_for_position: &'a [KeyedBimRecord<'arena>],
) -> Result<ReconciliationOutcome<'a, 'arena>, PrepError> {
    let is_multiallelic_site = bim_records_for_position.len() > 1;

    if is_multiallelic_site {
        let all_contexts_for_locus = bim_records_for_position.iter().collect();
        return Ok(ReconciliationOutcome::Complex(all_contexts_for_locus));
    }

    let mut simple_matches = BTreeMap::new();
    for record_tuple in bim_records_for_position {
        if record_tuple.allele1 == score_record.effect_allele
            || record_tuple.allele2 == score_record.effect_allele
        {
            simple_matches.insert(record_tuple.bim_row_index, record_tuple);
        }
    }

    if simple_matches.is_empty() {
        Ok(ReconciliationOutcome::NotFound)
    } else {
        let matches_as_vec = simple_matches.values().copied().collect();
        Ok(ReconciliationOutcome::Simple(matches_as_vec))
    }
}

fn resolve_person_subset(
    keep_file: Option<&Path>,
    all_person_iids: &[String],
    iid_to_original_idx: &AHashMap<String, u32>,
) -> Result<(PersonSubset, Vec<String>), PrepError> {
    if let Some(path) = keep_file {
        eprintln!(
            "> Subsetting individuals based on keep file: {}",
            path.display()
        );
        let file = File::open(path).map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
        let reader = BufReader::new(file);
        let iids_to_keep: AHashSet<String> = reader
            .lines()
            .filter_map(Result::ok)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut found_people = Vec::with_capacity(iids_to_keep.len());
        let mut missing_ids = Vec::new();

        for iid in iids_to_keep {
            if let Some(&original_idx) = iid_to_original_idx.get(&iid) {
                found_people.push((original_idx, iid));
            } else {
                missing_ids.push(iid);
            }
        }

        if !missing_ids.is_empty() {
            return Err(PrepError::InconsistentKeepId(format_missing_ids_error(
                missing_ids,
            )));
        }

        found_people.sort_unstable_by_key(|(idx, _)| *idx);
        let final_person_iids = found_people.iter().map(|(_, iid)| iid.clone()).collect();
        let subset_indices = found_people.into_iter().map(|(idx, _)| idx).collect();
        Ok((PersonSubset::Indices(subset_indices), final_person_iids))
    } else {
        Ok((PersonSubset::All, all_person_iids.to_vec()))
    }
}

fn parse_fam_and_build_lookup(
    fileset_paths: &[FilesetPaths],
) -> Result<(Vec<String>, AHashMap<String, u32>), PrepError> {
    let mut iid_to_idx = AHashMap::new();
    let mut canonical_iids: Option<Vec<String>> = None;
    let mut canonical_path: Option<PathBuf> = None;
    let mut seen_paths: AHashSet<PathBuf> = AHashSet::new();

    for fileset in fileset_paths {
        if !seen_paths.insert(fileset.fam.clone()) {
            continue;
        }

        let iids = read_fam_file(&fileset.fam)?;
        if let Some(existing) = &canonical_iids {
            if *existing != iids {
                let canonical = canonical_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(PrepError::PipelineIo {
                    path: fileset.fam.clone(),
                    message: format!(
                        "FAM file '{}' does not match canonical FAM '{}'.",
                        fileset.fam.display(),
                        canonical
                    ),
                });
            }
        } else {
            for (idx, iid) in iids.iter().enumerate() {
                iid_to_idx.insert(iid.clone(), idx as u32);
            }
            canonical_path = Some(fileset.fam.clone());
            canonical_iids = Some(iids);
        }
    }

    let person_iids = canonical_iids.ok_or_else(|| {
        PrepError::Parse("No individuals found in provided .fam files.".to_string())
    })?;
    Ok((person_iids, iid_to_idx))
}

fn read_fam_file(path: &Path) -> Result<Vec<String>, PrepError> {
    let mut source =
        open_text_source(path).map_err(|e| map_pipeline_error(e, path.to_path_buf()))?;
    let mut iids = Vec::new();
    let mut line_number = 0usize;

    while let Some(line) = source
        .next_line()
        .map_err(|e| map_pipeline_error(e, path.to_path_buf()))?
    {
        line_number += 1;
        if line.is_empty() {
            continue;
        }
        let line_str = std::str::from_utf8(line).map_err(|e| {
            PrepError::Parse(format!(
                "Invalid UTF-8 in .fam file '{}' on line {}: {e}",
                path.display(),
                line_number
            ))
        })?;
        let iid = line_str
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| {
                PrepError::Parse(format!(
                    "Missing IID in .fam file '{}' on line {}",
                    path.display(),
                    line_number
                ))
            })?
            .to_string();
        iids.push(iid);
    }

    Ok(iids)
}

fn format_missing_ids_error(missing_ids: Vec<String>) -> String {
    let sample_size = 5;
    let sample: Vec<_> = missing_ids.iter().take(sample_size).collect();
    let sample_str = sample
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if missing_ids.len() > sample_size {
        format!(
            "{} individuals from keep file not found. Sample: [{}...]",
            missing_ids.len(),
            sample_str
        )
    } else {
        format!(
            "{} individuals from keep file not found: [{}]",
            missing_ids.len(),
            sample_str
        )
    }
}

/// Parses only the headers of multiple score files to quickly build a complete,
/// sorted, and unique list of all score columns across all files.
pub fn parse_score_file_headers_only(score_files: &[PathBuf]) -> Result<Vec<String>, PrepError> {
    let all_score_names: BTreeSet<String> = score_files
        .par_iter()
        .map(|path| -> Result<Vec<String>, PrepError> {
            let file = File::open(path).map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
            let mut reader = BufReader::new(file);
            let mut header_line = String::new();

            loop {
                header_line.clear();
                if reader
                    .read_line(&mut header_line)
                    .map_err(|e| PrepError::Io(e, path.to_path_buf()))?
                    == 0
                {
                    return Err(PrepError::Header(format!(
                        "Score file \"{}\" is empty or contains only metadata lines.",
                        path.display()
                    )));
                }
                if !header_line.starts_with('#') {
                    break;
                }
            }

            let header_parts: Vec<&str> = header_line.trim().split('\t').collect();
            let expected_prefix = &["variant_id", "effect_allele", "other_allele"];

            if header_parts.len() < 3 || &header_parts[0..3] != expected_prefix {
                return Err(PrepError::Header(format!(
                    "Invalid header in \"{}\": Must start with 'variant_id\teffect_allele\tother_allele'.",
                    path.display()
                )));
            }

            let score_names = header_parts[3..]
                .iter()
                .map(|s| s.to_string())
                .collect();

            Ok(score_names)
        })
        .collect::<Result<Vec<Vec<String>>, PrepError>>()?
        .into_iter()
        .flatten()
        .collect();

    Ok(all_score_names.into_iter().collect())
}

// ========================================================================================
//                                    Error handling
// ========================================================================================

impl Display for PrepError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            PrepError::Io(e, path) => write!(f, "I/O Error for file {}: {}", path.display(), e),
            PrepError::Parse(s) => write!(f, "Parse Error: {s}"),
            PrepError::Header(s) => write!(f, "Invalid Header: {s}"),
            PrepError::InconsistentKeepId(s) => write!(f, "Configuration Error: {s}"),
            PrepError::PipelineIo { path, message } => {
                write!(f, "I/O Error for file {}: {}", path.display(), message)
            }
            PrepError::UnsortedInput {
                source,
                path,
                line_number,
                previous_key,
                current_key,
            } => {
                writeln!(f, "Detected unsorted {source} data in {}.", path.display())?;
                writeln!(
                    f,
                    "Encountered key {}:{} at line {}, which comes after {}:{}.",
                    current_key.0, current_key.1, line_number, previous_key.0, previous_key.1
                )?;
                writeln!(
                    f,
                    "Please sort your input by chromosome and position before running gnomon."
                )
            }
            PrepError::GenomeBuildMismatch => {
                writeln!(
                    f,
                    "No overlapping variants found even though both inputs are sorted and share chromosomes."
                )?;
                writeln!(
                    f,
                    "This strongly suggests a genome build mismatch (for example, GRCh37 vs GRCh38)."
                )
            }
            PrepError::DisjointChromosomes => {
                writeln!(
                    f,
                    "The genotype BIM files and score files do not share any chromosomes."
                )?;
                writeln!(
                    f,
                    "Verify that you are using compatible datasets (e.g., human vs. mouse or differing chromosome naming schemes)."
                )
            }
            PrepError::NoOverlappingVariants(diag) => {
                writeln!(
                    f,
                    "No overlapping variants found between genotype data and score files."
                )?;
                writeln!(
                    f,
                    "This likely means no variant keys (chr:pos) were identical in both sets of files."
                )?;
                writeln!(f, "\n--- DIAGNOSTIC INFORMATION ---")?;
                writeln!(
                    f,
                    "Total variants processed from BIM files: {}",
                    diag.total_bim_variants_processed
                )?;
                writeln!(
                    f,
                    "Total score records processed from Score files: {}",
                    diag.total_score_records_processed
                )?;

                if !diag.active_region_filters.is_empty() {
                    writeln!(f, "\nRegion filters requested:")?;
                    for (score, region) in &diag.active_region_filters {
                        writeln!(f, "  - {score} -> {region}")?;
                    }

                    if !diag.region_filters_without_hits.is_empty() {
                        writeln!(
                            f,
                            "\nNo score records were observed within the requested region for:"
                        )?;
                        for score in &diag.region_filters_without_hits {
                            writeln!(f, "  - {score}")?;
                        }
                    }

                    if diag.total_score_records_processed == 0 {
                        writeln!(
                            f,
                            "\nAll score records were filtered out by the requested region restriction(s)."
                        )?;
                        writeln!(
                            f,
                            "Please verify that the specified coordinates exist in the score file(s)."
                        )?;
                    }
                } else if diag.total_score_records_processed == 0 {
                    writeln!(f, "\nNo score records were processed.")?;
                    writeln!(
                        f,
                        "This can happen if the score files are empty or contain only unparsable entries."
                    )?;
                }

                if !diag.last_bim_keys_seen.is_empty() {
                    writeln!(
                        f,
                        "\nLast {} keys seen from BIM files:",
                        diag.last_bim_keys_seen.len()
                    )?;
                    for key in &diag.last_bim_keys_seen {
                        writeln!(f, "  - {}:{}", key.0, key.1)?;
                    }
                }

                if !diag.last_score_keys_seen.is_empty() {
                    writeln!(
                        f,
                        "\nLast {} keys seen from Score files (does the format match the BIM files?):",
                        diag.last_score_keys_seen.len()
                    )?;
                    for key in &diag.last_score_keys_seen {
                        writeln!(f, "  - {}:{}", key.0, key.1)?;
                    }
                }
                writeln!(
                    f,
                    "\nTIP: Please check for inconsistencies between your files."
                )
            }
            PrepError::AmbiguousReconciliation(s) => write!(f, "{s}"),
        }
    }
}

impl Error for PrepError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PrepError::Io(e, _) => Some(e),
            _ => None,
        }
    }
}

impl From<ParseFloatError> for PrepError {
    fn from(err: ParseFloatError) -> Self {
        PrepError::Parse(format!("Could not parse numeric value: {err}"))
    }
}

impl From<Utf8Error> for PrepError {
    fn from(err: Utf8Error) -> Self {
        PrepError::Parse(format!("Invalid UTF-8 sequence in score file: {err}"))
    }
}
