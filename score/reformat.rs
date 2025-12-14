// ========================================================================================
//
//               PGS catalog score file diagnostics & reformatting
//
// ========================================================================================

use flate2::read::MultiGzDecoder;
use memmap2::Mmap;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// ========================================================================================
//                                   Public API
// ========================================================================================

/// Checks if a file appears to be in the gnomon-native format by inspecting its header.
pub fn is_gnomon_native_format(path: &Path) -> io::Result<bool> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(false);
        }
        if !line.starts_with('#') {
            break;
        }
    }

    let header = line.trim();
    Ok(header.starts_with("variant_id\teffect_allele\tother_allele\t"))
}

/// A structured error type for the reformatting process, designed for useful diagnostics.
#[derive(Debug)]
pub enum ReformatError {
    /// An underlying I/O error.
    Io(io::Error),
    /// The file does not contain the expected PGS Catalog signature.
    NotPgsFormat { path: PathBuf },
    /// The header or a data row is missing a required column.
    MissingColumns {
        path: PathBuf,
        line_number: usize,
        line_content: String,
        missing_column_name: String,
    },
    /// A value in the file could not be parsed correctly.
    Parse {
        path: PathBuf,
        line_number: usize,
        line_content: String,
        details: String,
    },
    /// The provided PLINK fileset could not be re-ordered safely.
    InvalidPlinkFileset { path: PathBuf, details: String },
}

impl Display for ReformatError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // This implementation provides a detailed, multi-line diagnostic report to the user.
        writeln!(f, "Failed to reformat PGS Catalog score file.")?;
        writeln!(
            f,
            "\n-- Diagnostic Details ----------------------------------------------------"
        )?;

        match self {
            ReformatError::Io(e) => {
                writeln!(f, "Reason:       An I/O error occurred.")?;
                writeln!(f, "Details:      {e}")?;
            }
            ReformatError::NotPgsFormat { path } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(
                    f,
                    "Reason:       The file is not a valid PGS Catalog scoring file."
                )?;
                writeln!(
                    f,
                    "Details:      The required signature ('###PGS CATALOG SCORING FILE') was not found in the file's metadata header."
                )?;
            }
            ReformatError::MissingColumns {
                path,
                line_number,
                line_content,
                missing_column_name,
            } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(f, "Line Number:  {line_number}")?;
                writeln!(f, "Line Content: \"{}\"", line_content.trim())?;
                writeln!(f, "Reason:       A required column or its data is missing.")?;
                writeln!(
                    f,
                    "Details:      Expected to find column '{missing_column_name}', but it was not found in the header or data for this line."
                )?;
            }
            ReformatError::Parse {
                path,
                line_number,
                line_content,
                details,
            } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(f, "Line Number:  {line_number}")?;
                writeln!(f, "Line Content: \"{}\"", line_content.trim())?;
                writeln!(
                    f,
                    "Reason:       A value in the line could not be parsed correctly."
                )?;
                writeln!(f, "Details:      {details}")?;
            }
            ReformatError::InvalidPlinkFileset { path, details } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(f, "Reason:       Unable to sort PLINK fileset.")?;
                writeln!(f, "Details:      {details}")?;
            }
        }
        write!(
            f,
            "--------------------------------------------------------------------------"
        )
    }
}

impl Error for ReformatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReformatError::Io(e) => Some(e),
            ReformatError::NotPgsFormat { .. } => None,
            ReformatError::MissingColumns { .. } => None,
            ReformatError::Parse { .. } => None,
            ReformatError::InvalidPlinkFileset { .. } => None,
        }
    }
}

impl From<io::Error> for ReformatError {
    fn from(err: io::Error) -> Self {
        ReformatError::Io(err)
    }
}

/// Reformat a PGS Catalog scoring file into a gnomon-native, sorted TSV.
pub fn reformat_pgs_file(input_path: &Path, output_path: &Path) -> Result<(), ReformatError> {
    // --- Define helper types within the function scope ---
    #[derive(Clone, Copy)]
    enum ParsingStrategy {
        OriginalOnly,   // Use chr_name/chr_position, fail if unmappable.
        HarmonizedOnly, // Use hm_chr/hm_pos, fail if unmappable.
        SafeFallback,   // Try hm_chr/hm_pos, fall back to orig on parse failure.
    }

    struct ColumnIndices {
        chr: Option<usize>,
        pos: Option<usize>,
        hm_chr: Option<usize>,
        hm_pos: Option<usize>,
        ea: usize,         // effect_allele is mandatory
        ew: usize,         // effect_weight is mandatory
        oa: Option<usize>, // other_allele is optional
        variant_id: Option<usize>,
        hm_variant_id: Option<usize>,
        rs_id: Option<usize>,
        hm_rs_id: Option<usize>,
    }

    // --- Open file and handle potential GZIP compression ---
    let file = File::open(input_path)?;
    let a_reader: Box<dyn Read + Send> = if input_path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut reader = BufReader::new(a_reader);
    let mut line_buffer = String::new();

    // --- Read and analyze all metadata headers ---
    let mut orig_build_norm: Option<u8> = None;
    let mut hm_build_norm: Option<u8> = None;
    let mut score_id: Option<String> = None;

    let normalize_build = |build_str: &str| -> Option<u8> {
        match build_str.trim().to_lowercase().as_str() {
            "grch37" | "hg19" | "37" => Some(37),
            "grch38" | "hg38" | "38" => Some(38),
            _ => None,
        }
    };

    let mut total_lines_read = 0usize;

    loop {
        line_buffer.clear();
        if reader.read_line(&mut line_buffer)? == 0 {
            // Reached EOF before finding the data header
            return Err(ReformatError::NotPgsFormat {
                path: input_path.to_path_buf(),
            });
        }
        total_lines_read += 1;
        if !line_buffer.starts_with('#') {
            break; // Found the data header line
        }

        let metadata = line_buffer.trim_start_matches('#').trim();
        if let Some(val) = metadata.strip_prefix("genome_build=") {
            orig_build_norm = normalize_build(val);
        } else if let Some(val) = metadata.strip_prefix("HmPOS_build=") {
            hm_build_norm = normalize_build(val);
        } else if let Some(val) = metadata.strip_prefix("pgs_id=") {
            score_id = Some(val.to_string());
        }
    }

    // --- Determine the one, true, safe parsing strategy ---
    let strategy = match (orig_build_norm, hm_build_norm) {
        // Same build (e.g., 38 -> 38) OR unknown builds (can't prove they are different)
        (Some(o), Some(h)) if o == h => ParsingStrategy::SafeFallback,
        (None, None) => ParsingStrategy::SafeFallback, // Can't prove different, assume safe
        (Some(_), None) => ParsingStrategy::OriginalOnly, // Harmonized doesn't exist
        (None, Some(_)) => ParsingStrategy::HarmonizedOnly, // Original doesn't exist

        // Builds are provably different (e.g., 37 -> 38). Liftover occurred.
        (Some(_), Some(_)) => ParsingStrategy::HarmonizedOnly,
    };

    // --- Map header columns to indices for robust, order-independent access ---
    let header_line = line_buffer.trim();
    let header_map: std::collections::HashMap<&str, usize> = header_line
        .split('\t')
        .enumerate()
        .map(|(i, name)| (name, i))
        .collect();

    let column_indices = ColumnIndices {
        chr: header_map.get("chr_name").copied(),
        pos: header_map.get("chr_position").copied(),
        hm_chr: header_map.get("hm_chr").copied(),
        hm_pos: header_map.get("hm_pos").copied(),
        ea: *header_map
            .get("effect_allele")
            .ok_or_else(|| ReformatError::MissingColumns {
                path: input_path.to_path_buf(),
                line_number: 0,
                line_content: header_line.to_string(),
                missing_column_name: "effect_allele".to_string(),
            })?,
        ew: *header_map
            .get("effect_weight")
            .ok_or_else(|| ReformatError::MissingColumns {
                path: input_path.to_path_buf(),
                line_number: 0,
                line_content: header_line.to_string(),
                missing_column_name: "effect_weight".to_string(),
            })?,
        oa: header_map
            .get("other_allele")
            .or_else(|| header_map.get("hm_inferOtherAllele"))
            .copied(),
        variant_id: header_map.get("variant_id").copied(),
        hm_variant_id: header_map
            .get("hm_variant_id")
            .or_else(|| header_map.get("hm_variantID"))
            .copied(),
        rs_id: header_map
            .get("rsID")
            .or_else(|| header_map.get("rsid"))
            .copied(),
        hm_rs_id: header_map
            .get("hm_rsID")
            .or_else(|| header_map.get("hm_rsid"))
            .copied(),
    };

    fn derive_identifier(line: &str, fields: &[&str], columns: &ColumnIndices) -> String {
        fn value_from_idx<'a>(fields: &'a [&'a str], idx: Option<usize>) -> Option<&'a str> {
            idx.and_then(|i| fields.get(i))
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        }

        if let Some(val) = value_from_idx(fields, columns.variant_id) {
            return val.to_string();
        }
        if let Some(val) = value_from_idx(fields, columns.hm_variant_id) {
            return val.to_string();
        }
        if let Some(val) = value_from_idx(fields, columns.rs_id) {
            return val.to_string();
        }
        if let Some(val) = value_from_idx(fields, columns.hm_rs_id) {
            return val.to_string();
        }

        if let (Some(chr), Some(pos)) = (
            value_from_idx(fields, columns.hm_chr).or_else(|| value_from_idx(fields, columns.chr)),
            value_from_idx(fields, columns.hm_pos).or_else(|| value_from_idx(fields, columns.pos)),
        ) {
            return format!("{chr}:{pos}");
        }

        let trimmed = line.trim();
        if !trimmed.is_empty() {
            trimmed.to_string()
        } else {
            "<empty line>".to_string()
        }
    }

    // This is the "hot path". We use the pre-selected strategy and indices to
    // process all data lines in parallel with minimal branching or overhead.
    let score_label = score_id.unwrap_or_else(|| "PGS_SCORE".to_string());

    // --- Define the resolver closure based on the chosen strategy ---
    // The `move` keyword captures the `column_indices` struct by value.
    let resolver = move |line_number: usize, line: &str| -> Result<ResolveOutcome, ReformatError> {
        let fields: Vec<&str> = line.split('\t').collect();

        let make_skip = |reason: String| {
            ResolveOutcome::Skipped(SkipRecord {
                line_number,
                identifier: derive_identifier(line, &fields, &column_indices),
                reason,
            })
        };

        let attempt_coords = |label: &str,
                              c_idx: Option<usize>,
                              p_idx: Option<usize>|
         -> Result<(u8, u32), String> {
            let Some(chr_idx) = c_idx else {
                return Err(format!("{label} chromosome column is missing"));
            };
            let Some(pos_idx) = p_idx else {
                return Err(format!("{label} position column is missing"));
            };
            let chr_raw = fields.get(chr_idx).copied().unwrap_or("").trim();
            if chr_raw.is_empty() {
                return Err(format!("{label} chromosome value is empty"));
            }
            let pos_raw = fields.get(pos_idx).copied().unwrap_or("").trim();
            if pos_raw.is_empty() {
                return Err(format!("{label} position value is empty"));
            }

            parse_key(chr_raw, pos_raw)
                .map_err(|detail| format!("{label} coordinates invalid: {detail}"))
        };

        // This is the core logic, applying the pre-determined strategy.
        let key = match strategy {
            ParsingStrategy::SafeFallback => {
                match attempt_coords("Harmonized", column_indices.hm_chr, column_indices.hm_pos) {
                    Ok(key) => key,
                    Err(h_reason) => {
                        match attempt_coords("Original", column_indices.chr, column_indices.pos) {
                            Ok(key) => key,
                            Err(o_reason) => {
                                return Ok(make_skip(format!(
                                    "Harmonized coordinates unavailable: {h_reason}; Original coordinates unavailable: {o_reason}"
                                )));
                            }
                        }
                    }
                }
            }
            ParsingStrategy::HarmonizedOnly => {
                match attempt_coords("Harmonized", column_indices.hm_chr, column_indices.hm_pos) {
                    Ok(key) => key,
                    Err(reason) => {
                        return Ok(make_skip(format!(
                            "Harmonized coordinates unavailable: {reason}"
                        )));
                    }
                }
            }
            ParsingStrategy::OriginalOnly => {
                match attempt_coords("Original", column_indices.chr, column_indices.pos) {
                    Ok(key) => key,
                    Err(reason) => {
                        return Ok(make_skip(format!(
                            "Original coordinates unavailable: {reason}"
                        )));
                    }
                }
            }
        };

        // Safely extract other mandatory fields.
        let Some(ea_str) = fields
            .get(column_indices.ea)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            return Ok(make_skip("Missing effect_allele value".to_string()));
        };
        let Some(weight_str) = fields
            .get(column_indices.ew)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            return Ok(make_skip("Missing effect_weight value".to_string()));
        };
        let oa_str = column_indices
            .oa
            .and_then(|i| fields.get(i))
            .map_or("N", |s| if s.is_empty() { "N" } else { *s });

        let variant_id = format!("{}:{}", key.0, key.1);
        let line_data = format!("{variant_id}\t{ea_str}\t{oa_str}\t{weight_str}");

        Ok(ResolveOutcome::Resolved(SortableLine { key, line_data }))
    };

    // --- Read all data lines and process them in parallel ---
    let data_lines: Vec<(usize, String)> = reader
        .lines()
        .enumerate()
        .map(|(idx, result)| result.map(|line| (total_lines_read + idx + 1, line)))
        .collect::<Result<Vec<_>, _>>()?;
    let skipped_records = Mutex::new(Vec::new());

    let mut lines_to_sort = data_lines
        .into_par_iter()
        .filter_map(|(line_number, line)| {
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            match resolver(line_number, &line) {
                Ok(ResolveOutcome::Resolved(sortable_line)) => Some(Ok(sortable_line)),
                Ok(ResolveOutcome::Skipped(record)) => {
                    if let Ok(mut guard) = skipped_records.lock() {
                        guard.push(record);
                    }
                    None
                }
                Err(e) => Some(Err(e)), // This is a fatal error.
            }
        })
        .collect::<Result<Vec<_>, ReformatError>>()?;

    // Sort the resolved data and write it to the gnomon-native file.
    lines_to_sort.par_sort_unstable_by_key(|item| item.key);

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);
    writeln!(
        writer,
        "variant_id\teffect_allele\tother_allele\t{score_label}"
    )?;
    for item in lines_to_sort {
        writeln!(writer, "{}", item.line_data)?;
    }
    writer.flush()?;

    // --- Report any non-fatal issues to the user ---
    let mut skipped_records = skipped_records.into_inner().unwrap_or_default();
    if !skipped_records.is_empty() {
        skipped_records.sort_by_key(|record| record.line_number);
        eprintln!(
            "> Warning: Skipped {} variant(s) from '{}'. Details:",
            skipped_records.len(),
            input_path.display()
        );
        for record in skipped_records {
            eprintln!(
                ">   - line {} [{}]: {}",
                record.line_number, record.identifier, record.reason
            );
        }
    }

    Ok(())
}

/// Sorts a gnomon-native file that is not guaranteed to be sorted.
pub fn sort_native_file(input_path: &Path, output_path: &Path) -> Result<(), ReformatError> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    // Separate header lines from data lines to correctly track data line numbers.
    let mut header_lines: Vec<String> = vec![];
    let mut data_lines: Vec<String> = vec![];

    for line_result in reader.lines() {
        let line = line_result.map_err(ReformatError::Io)?;
        if !line.starts_with('#') {
            data_lines.push(line);
        } else {
            header_lines.push(line);
        }
    }

    if data_lines.is_empty() {
        return Ok(());
    }
    // The gnomon-native header is the first data line.
    let header = data_lines.remove(0);

    let mut lines_to_sort: Vec<SortableLine> = data_lines
        .into_par_iter()
        .enumerate()
        .map(|(i, line)| {
            // The line number is relative to the start of the data section.
            let line_number = i + 2;
            if line.is_empty() {
                return Ok(None);
            }

            let mut parts = line.splitn(2, '\t');
            let variant_id_part = parts.next().unwrap_or("");
            let mut key_parts = variant_id_part.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");

            // Call the decoupled parse_key function and map its simple error to our rich error type.
            let key = parse_key(chr_str, pos_str).map_err(|details| ReformatError::Parse {
                path: input_path.to_path_buf(),
                line_number,
                line_content: line.clone(),
                details,
            })?;

            Ok(Some(SortableLine {
                key,
                line_data: line,
            }))
        })
        .filter_map(|result| result.transpose())
        .collect::<Result<_, ReformatError>>()?;

    lines_to_sort.par_sort_unstable_by_key(|item| item.key);

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::new(out_file);
    // Write back any metadata lines that were present.
    for meta_line in header_lines {
        writeln!(writer, "{meta_line}")?;
    }
    writeln!(writer, "{header}")?;
    for item in lines_to_sort {
        writeln!(writer, "{}", item.line_data)?;
    }
    writer.flush()?;
    Ok(())
}

/// Sorts a PLINK binary fileset into a new `{prefix}.sorted.{bed,bim,fam}` trio and returns
/// the sorted BED path (which can be used with `Path::with_extension`).
pub fn sort_plink_fileset(
    bed_path: &Path,
    bim_path: &Path,
    fam_path: &Path,
) -> Result<PathBuf, ReformatError> {
    struct KeyedBimLine {
        key: (u8, u32),
        original_index: usize,
        raw_line: String,
    }

    let bim_file = File::open(bim_path)?;
    let bim_reader = BufReader::new(bim_file);
    let mut keyed_lines: Vec<KeyedBimLine> = Vec::new();

    let mut valid_variant_index: usize = 0;
    for (line_idx, line_result) in bim_reader.lines().enumerate() {
        let line_number = line_idx + 1;
        let line = line_result.map_err(ReformatError::Io)?;
        if line.trim().is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let chr_str = parts.next().unwrap_or("");
        let _ = parts.next();
        let _ = parts.next();
        let pos_str = parts.next().unwrap_or("");

        if chr_str.is_empty() || pos_str.is_empty() {
            return Err(ReformatError::MissingColumns {
                path: bim_path.to_path_buf(),
                line_number,
                line_content: line,
                missing_column_name: "chromosome/position".to_string(),
            });
        }

        let key = parse_key(chr_str, pos_str).map_err(|details| ReformatError::Parse {
            path: bim_path.to_path_buf(),
            line_number,
            line_content: line.clone(),
            details,
        })?;

        keyed_lines.push(KeyedBimLine {
            key,
            original_index: valid_variant_index,
            raw_line: line,
        });
        valid_variant_index += 1;
    }

    if keyed_lines.is_empty() {
        return Err(ReformatError::InvalidPlinkFileset {
            path: bim_path.to_path_buf(),
            details: "BIM file contained no variants.".to_string(),
        });
    }

    keyed_lines.par_sort_unstable_by_key(|record| record.key);

    let fam_file = File::open(fam_path)?;
    let fam_reader = BufReader::new(fam_file);
    let mut fam_line_count: usize = 0;
    for line in fam_reader.lines() {
        let line = line.map_err(ReformatError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        fam_line_count += 1;
    }

    if fam_line_count == 0 {
        return Err(ReformatError::InvalidPlinkFileset {
            path: fam_path.to_path_buf(),
            details: "FAM file contained no individuals.".to_string(),
        });
    }

    let bytes_per_variant = fam_line_count.div_ceil(4);
    let bed_file = File::open(bed_path)?;
    let mmap = unsafe { Mmap::map(&bed_file)? };

    if mmap.len() < 3 {
        return Err(ReformatError::InvalidPlinkFileset {
            path: bed_path.to_path_buf(),
            details: "BED file is smaller than the 3-byte PLINK header.".to_string(),
        });
    }

    let expected_bytes = 3 + keyed_lines.len() * bytes_per_variant;
    if mmap.len() < expected_bytes {
        return Err(ReformatError::InvalidPlinkFileset {
            path: bed_path.to_path_buf(),
            details: format!(
                "BED file is {actual} bytes but expected at least {expected_bytes} bytes for {variants} variant(s) and {people} individual(s).",
                actual = mmap.len(),
                variants = keyed_lines.len(),
                people = fam_line_count,
            ),
        });
    }

    let parent_dir = bed_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = bed_path
        .file_stem()
        .ok_or_else(|| ReformatError::InvalidPlinkFileset {
            path: bed_path.to_path_buf(),
            details: "Could not derive file stem for BED path.".to_string(),
        })?;

    let mut sorted_stem = stem.to_os_string();
    sorted_stem.push(".sorted");
    let mut sorted_bed_name = sorted_stem.clone();
    sorted_bed_name.push(".bed");
    let sorted_bed_path = parent_dir.join(sorted_bed_name);
    let mut sorted_bed = BufWriter::with_capacity(1 << 20, File::create(&sorted_bed_path)?);
    sorted_bed.write_all(&mmap[..3])?;

    for record in &keyed_lines {
        let offset = 3 + record.original_index * bytes_per_variant;
        let end = offset + bytes_per_variant;
        sorted_bed.write_all(&mmap[offset..end])?;
    }
    sorted_bed.flush()?;

    let mut sorted_bim_name = sorted_stem.clone();
    sorted_bim_name.push(".bim");
    let sorted_bim_path = parent_dir.join(sorted_bim_name);
    let mut sorted_bim = BufWriter::with_capacity(1 << 20, File::create(&sorted_bim_path)?);
    for record in &keyed_lines {
        writeln!(sorted_bim, "{}", record.raw_line)?;
    }
    sorted_bim.flush()?;

    let mut sorted_fam_name = sorted_stem.clone();
    sorted_fam_name.push(".fam");
    let sorted_fam_path = parent_dir.join(sorted_fam_name);
    fs::copy(fam_path, &sorted_fam_path).map_err(ReformatError::Io)?;

    Ok(sorted_bed_path)
}

// ========================================================================================
//                        Private types and helpers
// ========================================================================================

struct SortableLine {
    key: (u8, u32),
    line_data: String,
}

enum ResolveOutcome {
    Resolved(SortableLine),
    Skipped(SkipRecord),
}

struct SkipRecord {
    line_number: usize,
    identifier: String,
    reason: String,
}

fn parse_key(chr_str: &str, pos_str: &str) -> Result<(u8, u32), String> {
    // First, check for special, non-numeric chromosome names case-insensitively.
    if chr_str.eq_ignore_ascii_case("X") {
        let pos_num: u32 = pos_str
            .parse()
            .map_err(|e: ParseIntError| format!("Invalid position '{pos_str}': {e}"))?;
        return Ok((23, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("Y") {
        let pos_num: u32 = pos_str
            .parse()
            .map_err(|e: ParseIntError| format!("Invalid position '{pos_str}': {e}"))?;
        return Ok((24, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("MT") {
        let pos_num: u32 = pos_str
            .parse()
            .map_err(|e: ParseIntError| format!("Invalid position '{pos_str}': {e}"))?;
        return Ok((25, pos_num));
    }

    // Next, handle numeric chromosomes, stripping a potential "chr" prefix case-insensitively.
    let number_part = if chr_str.len() >= 3 && chr_str[..3].eq_ignore_ascii_case("chr") {
        &chr_str[3..]
    } else {
        chr_str
    };

    // Now, parse the remaining part.
    let chr_num: u8 = number_part.parse().map_err(|_| {
        format!(
            "Invalid chromosome format '{chr_str}'. Expected a number, 'X', 'Y', 'MT', or 'chr' prefix."
        )
    })?;

    let pos_num: u32 = pos_str
        .parse()
        .map_err(|e: ParseIntError| format!("Invalid position '{pos_str}': {e}"))?;

    Ok((chr_num, pos_num))
}
