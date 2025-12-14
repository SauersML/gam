// ========================================================================================
//
//                               Score file downloader
//
// ========================================================================================

use crate::score::reformat;
use crate::score::types::{GenomicRegion, parse_chromosome_label};
use reqwest::blocking::Client;
use std::io::Write;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;



// ========================================================================================
//                              Public API
// ========================================================================================

/// The primary entry point for the persistent download workflow.
///
/// This function synchronizes a local directory (`scores_dir`) with a list of
/// requested PGS IDs. It performs the following steps:
/// 1. Parses the list of requested PGS IDs.
/// 2. Checks `scores_dir` for any corresponding `.gnomon.tsv` files that already exist.
/// 3. For any missing files, it downloads them in parallel from the PGS Catalog.
/// 4. It reformats the newly downloaded files in parallel into the native format.
/// 5. It cleans up the intermediate downloaded files.
/// 6. It returns a complete list of paths to all requested `.gnomon.tsv` files,
///    including both pre-existing and newly created ones.
///
/// # Arguments
/// * `score_arg` - A comma-separated string of PGS IDs.
/// * `scores_dir` - The path to the permanent directory where score files should be
///   stored. This directory will be created if it does not exist.
///
/// # Returns
/// A `Result` containing a [`ResolvedScoreFiles`] struct on success, or a boxed
/// dynamic error on failure.
pub fn resolve_and_download_scores(
    score_arg: &str,
    scores_dir: &Path,
) -> Result<ResolvedScoreFiles, Box<dyn Error + Send + Sync>> {
    // --- Stage 1: Setup and State Synchronization ---
    fs::create_dir_all(scores_dir)?;
    eprintln!(
        "> Checking for score files in permanent directory: {}",
        scores_dir.display()
    );

    let parsed_specs = parse_requested_scores(score_arg)?;
    if parsed_specs.is_empty() {
        return Err("No valid PGS IDs provided in --score argument.".into());
    }

    let mut requested_pgs_ids = BTreeSet::new();
    let mut region_overrides: HashMap<String, GenomicRegion> = HashMap::new();

    for (id, region) in parsed_specs {
        if !id.starts_with("PGS") {
            return Err(Box::new(DownloadError::InvalidId(id)));
        }

        if !requested_pgs_ids.insert(id.clone()) {
            if let Some(region) = region {
                match region_overrides.get(&id) {
                    Some(existing) if *existing != region => {
                        return Err(Box::new(DownloadError::InvalidRegion(format!(
                            "Score '{id}' specified multiple conflicting regions ({} vs {}).",
                            existing, region
                        ))));
                    }
                    Some(_) => {}
                    None => {
                        region_overrides.insert(id.clone(), region);
                    }
                }
            }
        } else if let Some(region) = region {
            region_overrides.insert(id.clone(), region);
        }
    }

    if requested_pgs_ids.is_empty() {
        return Err("No valid PGS IDs provided in --score argument.".into());
    }

    let mut existing_native_paths = Vec::new();
    let mut files_to_reformat = Vec::new();
    let mut ids_to_download = Vec::new();

    // Check for existing files: final .gnomon.tsv or intermediate .txt.gz
    for id in &requested_pgs_ids {
        let native_path = scores_dir.join(format!("{id}.gnomon.tsv"));
        let temp_gz_path = scores_dir.join(format!("{id}.txt.gz"));

        if native_path.exists() {
            existing_native_paths.push(native_path);
        } else if temp_gz_path.exists() {
            eprintln!("> Found existing downloaded file for {id}. Skipping download.");
            files_to_reformat.push((temp_gz_path, native_path));
        } else {
            ids_to_download.push(id.clone());
        }
    }

    eprintln!(
        "> Found {} existing reformatted score files. Found {} existing downloaded files. Need to download {}.",
        existing_native_paths.len(),
        files_to_reformat.len(),
        ids_to_download.len()
    );

    // --- Stage 2: Conditional Download ---
    if !ids_to_download.is_empty() {
        let newly_downloaded = download_missing_files(&ids_to_download, scores_dir)?;
        files_to_reformat.extend(newly_downloaded);
    }

    // --- Stage 3: Conditional Reformatting ---
    if files_to_reformat.is_empty() {
        eprintln!("> All required score files are already present and converted.");
        return Ok(ResolvedScoreFiles {
            paths: existing_native_paths,
            regions: region_overrides,
        });
    }

    eprintln!(
        "> Reformatting {} score files into gnomon-native format...",
        files_to_reformat.len()
    );

    let new_native_paths = files_to_reformat
        .into_par_iter()
        .map(
            |(input_path, final_native_path)| -> Result<PathBuf, DownloadError> {
                reformat::reformat_pgs_file(&input_path, &final_native_path)
                    .map_err(DownloadError::Reformat)?;

                // Clean up the original downloaded file immediately after successful reformatting.
                fs::remove_file(&input_path)
                    .map_err(|e| DownloadError::Io(e, input_path.clone()))?;

                Ok(final_native_path)
            },
        )
        .collect::<Result<Vec<PathBuf>, _>>()?;

    eprintln!("> Reformatting complete.");

    // --- Stage 4: Final Aggregation ---
    existing_native_paths.extend(new_native_paths);
    Ok(ResolvedScoreFiles {
        paths: existing_native_paths,
        regions: region_overrides,
    })
}

// ========================================================================================
//                             Private implementation
// ========================================================================================

/// A specialized error type for the download and reformat workflow.
#[derive(Debug)]
pub enum DownloadError {
    Io(io::Error, PathBuf),
    Network(String),
    InvalidId(String),
    InvalidRegion(String),
    Reformat(reformat::ReformatError),
    RuntimeCreation(io::Error),
}

impl Display for DownloadError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DownloadError::Io(e, path) => {
                write!(f, "I/O error for '{}': {}", path.display(), e)
            }
            DownloadError::Network(s) => write!(f, "Network download failed: {s}"),
            DownloadError::InvalidId(s) => write!(
                f,
                "Invalid ID format for '{s}'. All IDs must start with 'PGS'."
            ),
            DownloadError::InvalidRegion(s) => write!(f, "Invalid region specification: {s}"),
            DownloadError::Reformat(e) => write!(f, "{e}"),
            DownloadError::RuntimeCreation(e) => write!(f, "Failed to create async runtime: {e}"),
        }
    }
}

impl Error for DownloadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            DownloadError::Io(e, _) => Some(e),
            DownloadError::Reformat(e) => Some(e),
            DownloadError::RuntimeCreation(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_requested_scores_handles_regions() {
        let specs = parse_requested_scores(
            "PGS000001 | chr1:100-200, PGS000002, PGS000003 | chrX:1,000-2,000",
        )
        .unwrap();

        assert_eq!(specs.len(), 3);
        let first_region = specs[0].1.expect("expected region");
        assert!(first_region.contains((1, 150)));
        assert!(specs[1].1.is_none());
        let third_region = specs[2].1.expect("expected region");
        assert_eq!(third_region.chromosome, 23);
        assert_eq!(third_region.start, 1000);
        assert_eq!(third_region.end, 2000);
    }

    #[test]
    fn resolve_and_download_scores_errors_on_conflicting_regions() {
        let dir = tempdir().unwrap();
        let err = resolve_and_download_scores(
            "PGS000010 | chr1:10-20, PGS000010 | chr1:30-40",
            dir.path(),
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("conflicting regions"));
    }
}

/// Orchestrates the parallel download of all specified missing PGS IDs.
fn download_missing_files(
    pgs_ids: &[String],
    target_dir: &Path,
) -> Result<Vec<(PathBuf, PathBuf)>, DownloadError> {
    eprintln!(
        "> Downloading {} missing score files from PGS Catalog...",
        pgs_ids.len()
    );

    let client = Client::builder()
        .user_agent("gnomon-http-client/1.0")
        .build()
        .map_err(|e| DownloadError::Network(format!("Failed to create HTTP client: {e}")))?;

    let multi = Arc::new(MultiProgress::new());
    // Use a progress bar style that does not require the total file size.
    let style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .unwrap()
        .progress_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");

    let results: Result<Vec<_>, DownloadError> = pgs_ids
        .par_iter()
        .map(|id| {
            let temp_gz_path = target_dir.join(format!("{id}.txt.gz"));
            let final_native_path = target_dir.join(format!("{id}.gnomon.tsv"));
            let url = format!(
                "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{id}/ScoringFiles/Harmonized/{id}_hmPOS_GRCh38.txt.gz"
            );

            let pb = multi.add(ProgressBar::new_spinner());
            pb.set_style(style.clone());
            pb.set_message(format!("Downloading {id}..."));
            pb.enable_steady_tick(std::time::Duration::from_millis(100));

            let mut response = client
                .get(&url)
                .send()
                .map_err(|e| DownloadError::Network(format!("Failed to initiate download for {id}: {e}")))?;

            if !response.status().is_success() {
                let status = response.status();
                pb.finish_with_message(format!("Failed: {status}"));
                return Err(DownloadError::Network(format!(
                    "Download failed for {id} with status {status}"
                )));
            }

            let mut file = std::fs::File::create(&temp_gz_path)
                .map_err(|e| DownloadError::Io(e, temp_gz_path.clone()))?;

            // Copy with progress
            let mut buf = [0; 8192];
            use std::io::Read;
            loop {
                let n = response.read(&mut buf).map_err(|e| {
                    DownloadError::Network(format!("Failed reading response for {id}: {e}"))
                })?;
                if n == 0 {
                    break;
                }
                file.write_all(&buf[..n])
                    .map_err(|e| DownloadError::Io(e, temp_gz_path.clone()))?;
                // Since we don't know total size for gzip stream comfortably from all servers, just spinner
            }

            pb.finish_with_message(format!("Downloaded {id}"));
            Ok((temp_gz_path, final_native_path))
        })
        .collect();

    results
}
#[derive(Debug)]
pub struct ResolvedScoreFiles {
    pub paths: Vec<PathBuf>,
    pub regions: HashMap<String, GenomicRegion>,
}

fn parse_position_component(component: &str) -> Result<u32, String> {
    let cleaned: String = component
        .chars()
        .filter(|c| !matches!(c, ',' | '_' | ' '))
        .collect();
    if cleaned.is_empty() {
        return Err(format!("Coordinate '{component}' is empty."));
    }
    let value: u64 = cleaned
        .parse()
        .map_err(|_| format!("Coordinate '{component}' is not a valid integer."))?;
    if value == 0 {
        return Err(format!(
            "Coordinate '{component}' must be greater than zero."
        ));
    }
    if value > u32::MAX as u64 {
        return Err(format!(
            "Coordinate '{component}' exceeds the supported range (>{}).",
            u32::MAX
        ));
    }
    Ok(value as u32)
}

fn parse_region_spec(region_str: &str) -> Result<GenomicRegion, DownloadError> {
    let trimmed = region_str.trim();
    let mut parts = trimmed.splitn(2, ':');
    let chr_part = parts.next().unwrap_or("").trim();
    let range_part = parts
        .next()
        .ok_or_else(|| {
            DownloadError::InvalidRegion(format!("Region '{trimmed}' is missing a ':' separator."))
        })?
        .trim();

    if chr_part.is_empty() {
        return Err(DownloadError::InvalidRegion(format!(
            "Region '{trimmed}' is missing a chromosome label."
        )));
    }
    if range_part.is_empty() {
        return Err(DownloadError::InvalidRegion(format!(
            "Region '{trimmed}' is missing position bounds."
        )));
    }

    let mut bounds = range_part.splitn(2, '-');
    let start_str = bounds.next().unwrap_or("").trim();
    let end_str = bounds
        .next()
        .ok_or_else(|| {
            DownloadError::InvalidRegion(format!(
                "Region '{trimmed}' is missing an end coordinate."
            ))
        })?
        .trim();

    if start_str.is_empty() || end_str.is_empty() {
        return Err(DownloadError::InvalidRegion(format!(
            "Region '{trimmed}' must include both start and end coordinates."
        )));
    }

    let chromosome = parse_chromosome_label(chr_part).map_err(|msg| {
        DownloadError::InvalidRegion(format!("Invalid chromosome in region '{trimmed}': {msg}"))
    })?;
    let start = parse_position_component(start_str).map_err(|msg| {
        DownloadError::InvalidRegion(format!(
            "Invalid start coordinate in region '{trimmed}': {msg}"
        ))
    })?;
    let end = parse_position_component(end_str).map_err(|msg| {
        DownloadError::InvalidRegion(format!(
            "Invalid end coordinate in region '{trimmed}': {msg}"
        ))
    })?;

    if start > end {
        return Err(DownloadError::InvalidRegion(format!(
            "Region '{trimmed}' has start coordinate greater than end coordinate."
        )));
    }

    Ok(GenomicRegion {
        chromosome,
        start,
        end,
    })
}

fn split_score_argument(score_arg: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let bytes = score_arg.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        if bytes[i] == b',' {
            let mut lookahead = i + 1;
            while lookahead < len && bytes[lookahead].is_ascii_whitespace() {
                lookahead += 1;
            }
            if lookahead < len && score_arg[lookahead..].starts_with("PGS") {
                let item = score_arg[start..i].trim();
                if !item.is_empty() {
                    parts.push(item.to_string());
                }
                start = i + 1;
            }
        }
        i += 1;
    }

    let tail = score_arg[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }

    parts
}

fn parse_requested_scores(
    score_arg: &str,
) -> Result<Vec<(String, Option<GenomicRegion>)>, DownloadError> {
    let mut results = Vec::new();

    for raw in split_score_argument(score_arg) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parts = trimmed.split('|');
        let id_part = parts.next().unwrap_or("").trim();
        let region_part = parts.next().map(|s| s.trim());

        if parts.next().is_some() {
            return Err(DownloadError::InvalidRegion(format!(
                "Score specification '{trimmed}' contains multiple '|' separators."
            )));
        }

        if id_part.is_empty() {
            return Err(DownloadError::InvalidId(trimmed.to_string()));
        }

        let region = if let Some(region_str) = region_part {
            if region_str.is_empty() {
                return Err(DownloadError::InvalidRegion(format!(
                    "Region specification for '{id_part}' is empty."
                )));
            }
            Some(parse_region_spec(region_str)?)
        } else {
            None
        };

        results.push((id_part.to_string(), region));
    }

    Ok(results)
}
