// ========================================================================================
//                             High-Level Data Contracts
// ========================================================================================

// This file is ONLY for types that are SHARED BETWEEN FILES, not types that only are used in one file.

use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenomicRegion {
    pub chromosome: u8,
    pub start: u32,
    pub end: u32,
}

impl GenomicRegion {
    #[inline]
    pub fn contains(&self, key: (u8, u32)) -> bool {
        key.0 == self.chromosome && key.1 >= self.start && key.1 <= self.end
    }
}

impl fmt::Display for GenomicRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let chr_label = match self.chromosome {
            23 => "X".to_string(),
            24 => "Y".to_string(),
            25 => "MT".to_string(),
            n => format!("{n}"),
        };
        write!(f, "chr{chr_label}:{}-{}", self.start, self.end)
    }
}

pub fn parse_chromosome_label(chr_str: &str) -> Result<u8, String> {
    let mut trimmed = chr_str.trim();

    if trimmed.len() >= 3 && trimmed[..3].eq_ignore_ascii_case("chr") {
        trimmed = &trimmed[3..];
    }

    if trimmed.eq_ignore_ascii_case("X") {
        return Ok(23);
    }
    if trimmed.eq_ignore_ascii_case("Y") {
        return Ok(24);
    }
    if trimmed.eq_ignore_ascii_case("MT") {
        return Ok(25);
    }

    trimmed.parse::<u8>().map_err(|_| {
        format!(
            "Invalid chromosome format '{}'. Expected a number, 'X', 'Y', 'MT', or 'chr' prefix.",
            chr_str.trim()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_chromosome_label_supports_common_variants() {
        assert_eq!(parse_chromosome_label("1").unwrap(), 1);
        assert_eq!(parse_chromosome_label("chr2").unwrap(), 2);
        assert_eq!(parse_chromosome_label("chrX").unwrap(), 23);
        assert_eq!(parse_chromosome_label("MT").unwrap(), 25);
    }

    #[test]
    fn genomic_region_contains_enforces_bounds() {
        let region = GenomicRegion {
            chromosome: 1,
            start: 100,
            end: 200,
        };

        assert!(region.contains((1, 150)));
        assert!(!region.contains((1, 50)));
        assert!(!region.contains((2, 150)));
        assert!(!region.contains((1, 250)));
    }
}

/// The payload sent from the I/O producer to the compute consumers. It contains
/// the raw data for one variant and the necessary metadata to process it.
#[derive(Debug)]
pub struct WorkItem {
    pub data: Vec<u8>,
    pub reconciled_variant_index: ReconciledVariantIndex,
}

/// A boundary marker for a single fileset within a virtually-merged collection.
/// It contains the path to the .bed file and the starting global index for its variants.
#[derive(Debug, Clone)]
pub struct FilesetBoundary {
    pub bed_path: PathBuf,
    pub bim_path: PathBuf,
    pub fam_path: PathBuf,
    pub starting_global_index: u64,
}

/// This enum makes the pipeline choice a type-safe property of the computation.
/// At compile time, the program knows whether it's dealing with one file or many,
/// making it impossible to call the wrong I/O logic.
#[derive(Debug)]
pub enum PipelineKind {
    /// The simple case: one fileset, one .bed file to mmap.
    SingleFile(PathBuf),
    /// The multi-file case: a "virtual" fileset composed of multiple physical files.
    MultiFile(Vec<FilesetBoundary>),
}

/// Contains the score-specific information for a complex variant.
/// This is the part of a rule that changes for each score file.
#[derive(Debug, Clone)]
pub struct ScoreInfo {
    /// The effect allele as defined in the score file. This could theoretically
    /// differ between scores for the same variant.
    pub effect_allele: String,
    /// The other (non-effect) allele for the variant, used for disambiguation.
    pub other_allele: String,
    /// The effect weight for this specific score.
    pub weight: f32,
    /// The global column index for the score this rule applies to.
    pub score_column_index: ScoreColumnIndex,
}

/// A self-contained, grouped unit of work for a single unique complex variant.
/// This struct holds information that is shared across all scores that use this variant.
#[derive(Debug, Clone)]
pub struct GroupedComplexRule {
    /// The canonical chromosome and position for this complex locus. This information
    /// is vital for generating informative error messages.
    pub locus_chr_pos: (String, u32),

    /// A list of all plausible BIM contexts (genotype definitions). This data is
    /// now stored only once per complex variant, not duplicated for every score.
    pub possible_contexts: Vec<(BimRowIndex, String, String)>, // (BimRowIndex, allele1, allele2)

    /// A list of all scores that apply to this set of contexts.
    pub score_applications: Vec<ScoreInfo>,
}
/// Defines the subset of individuals to be processed by the engine.
///
/// This enum makes the program's control flow explicit and type-safe. It is
/// impossible to ambiguously handle the "all vs. subset" cases.
#[derive(Debug)]
pub enum PersonSubset {
    /// Process all individuals from the .fam file.
    All,
    /// Process only a specific subset of individuals.
    /// Contains the sorted, 0-based indices from the original .fam file.
    Indices(Vec<u32>),
}

/// A "proof token" containing the complete, validated, and "compiled" blueprint for a
/// polygenic score calculation.
///
/// The successful creation of this struct is a guarantee that all input files were
/// valid and consistent. It contains the hyper-optimized data matrices that the
/// compute engine will execute.
#[derive(Debug)]
pub struct PreparationResult {
    // --- Private, compiled data matrices ---
    // These fields are private to guarantee their invariants. They are created once
    // by the `prepare` module and can only be read by downstream modules.
    weights_matrix: Vec<f32>,
    flip_mask_matrix: Vec<u8>,
    stride: usize,

    // --- Public metadata & lookup tables ---
    /// The sorted list of original `.bim` row indices for the "fast path."
    /// This is used by the I/O producer to filter the `.bed` file for all
    /// simple, unambiguous variants. The indices are global across all filesets.
    pub required_bim_indices: Vec<BimRowIndex>,
    /// A list of self-contained, grouped rules for variants that require complex,
    /// deferred resolution (the "slow path"). Each rule represents one unique
    /// complex variant and all scores that use it.
    pub complex_rules: Vec<GroupedComplexRule>,
    /// A map from a person's original .fam index to their compact output index.
    /// `None` if the person is not in the scored subset.
    pub person_fam_to_output_idx: Vec<Option<u32>>,
    /// A map from the final, compact output index back to the original .fam file index.
    /// This is a critical optimization for the variant-major path.
    pub output_idx_to_fam_idx: Vec<u32>,
    /// The names of the scores being calculated, corresponding to matrix columns.
    pub score_names: Vec<String>,
    /// A vector containing the total number of variants that contribute to each score.
    /// This is used as the denominator for per-variant-average and missing % calculations.
    /// The order matches `score_names`.
    pub score_variant_counts: Vec<u32>,
    /// A "missingness blueprint" mapping each dense matrix row (variant) to the
    /// score column indices it affects.
    pub variant_to_scores_map: Vec<Vec<ScoreColumnIndex>>,
    /// The exact subset of individuals to be processed.
    pub person_subset: PersonSubset,
    /// The list of Individual IDs (IIDs) for the individuals being scored, in the
    /// final output order.
    pub final_person_iids: Vec<String>,
    /// The number of individuals that will actually be scored in this run.
    pub num_people_to_score: usize,
    /// The total number of individuals found in the original .fam file.
    pub total_people_in_fam: usize,
    /// The total number of variants found across all original .bim files.
    pub total_variants_in_bim: u64,
    /// The number of variants successfully reconciled and included in the matrices.
    /// This corresponds to the number of rows in the compiled matrices.
    pub num_reconciled_variants: usize,
    /// The number of bytes per variant in the .bed file, calculated as CEIL(total_people_in_fam / 4).
    pub bytes_per_variant: u64,
    /// Flags indicating whether the required BIM indices correspond to complex contexts.
    required_is_complex: Vec<u8>,
    /// Mapping of kept byte positions used for complex variant spooling.
    /// Entries are sorted and unique to allow linear compaction during streaming.
    spool_compact_byte_index: Vec<u32>,
    /// Dense lookup table mapping original byte indices to compact spool indices.
    spool_dense_map: Vec<i32>,
    /// The number of bytes written per spooled variant row.
    spool_bytes_per_variant: u64,
    /// The type-safe representation of the I/O pipeline strategy.
    pub pipeline_kind: PipelineKind,
}

impl PreparationResult {
    /// The constructor is crate-private, enforcing the "Airlock" pattern.
    /// Only the `prepare` module can construct this "proof token".
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights_matrix: Vec<f32>,
        flip_mask_matrix: Vec<u8>,
        stride: usize,
        required_bim_indices: Vec<BimRowIndex>,
        complex_rules: Vec<GroupedComplexRule>,
        score_names: Vec<String>,
        score_variant_counts: Vec<u32>,
        variant_to_scores_map: Vec<Vec<ScoreColumnIndex>>,
        person_subset: PersonSubset,
        final_person_iids: Vec<String>,
        num_people_to_score: usize,
        total_people_in_fam: usize,
        total_variants_in_bim: u64,
        num_reconciled_variants: usize,
        bytes_per_variant: u64,
        person_fam_to_output_idx: Vec<Option<u32>>,
        output_idx_to_fam_idx: Vec<u32>,
        required_is_complex: Vec<u8>,
        spool_compact_byte_index: Vec<u32>,
        spool_dense_map: Vec<i32>,
        spool_bytes_per_variant: u64,
        pipeline_kind: PipelineKind,
    ) -> Self {
        Self {
            weights_matrix,
            flip_mask_matrix,
            stride,
            required_bim_indices,
            complex_rules,
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
        }
    }

    // --- Public Getters for Private Data ---

    #[inline(always)]
    pub fn weights_matrix(&self) -> &[f32] {
        &self.weights_matrix
    }

    #[inline(always)]
    pub fn flip_mask_matrix(&self) -> &[u8] {
        &self.flip_mask_matrix
    }

    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline(always)]
    pub fn required_is_complex(&self) -> &[u8] {
        &self.required_is_complex
    }

    #[inline(always)]
    pub fn spool_compact_byte_index(&self) -> &[u32] {
        &self.spool_compact_byte_index
    }

    #[inline(always)]
    pub fn spool_dense_map(&self) -> &[i32] {
        &self.spool_dense_map
    }

    #[inline(always)]
    pub fn spool_bytes_per_variant(&self) -> u64 {
        self.spool_bytes_per_variant
    }
}

// ========================================================================================
//                            Primitive Type Definitions
// ========================================================================================

/// An index into the original, full .fam file (e.g., one of 150,000).
///
/// This newtype prevents confusion between different index spaces at compile time.
/// The `#[repr(transparent)]` attribute guarantees this is a zero-cost abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct OriginalPersonIndex(pub u32);

/// An index into the original, full .bim file.
/// This wraps a u64, which is sufficient for a virtually unlimited number of
/// variants when merging filesets. This is a zero-cost abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BimRowIndex(pub u64);

/// An index into the dense, reconciled matrices (`weights_matrix`, `flip_mask_matrix`).
/// This wraps a u32, which is sufficient for >4 billion variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ReconciledVariantIndex(pub u32);

/// An index corresponding to a specific score (a column in the conceptual matrix).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ScoreColumnIndex(pub usize);

/// A `#[repr(transparent)]` wrapper for a dosage value.
///
/// This type is used in the pivoted tile buffer. Its `u8` representation is
/// compact and efficient for the pivoting process.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectAlleleDosage(pub u8);

impl EffectAlleleDosage {
    /// Creates a new dosage, asserting the value is valid in debug builds.
    #[inline(always)]
    pub fn new(value: u8) -> Self {
        debug_assert!(value <= 3, "Invalid dosage value created: {value}");
        Self(value)
    }
}
