// ========================================================================================
//
//                               The strategic orchestrator: Gnomon
//
// ========================================================================================
//
// This module is the central nervous system and active conductor of the application.
// Its sole responsibility is to orchestrate the high-performance pipeline defined in
// the other modules. It owns all major resources and manages the application lifecycle
// from argument parsing to final output.

#![deny(dead_code)]
#![deny(unused_imports)]

use clap::Parser;
use gnomon::score::download;
use gnomon::score::io::{gcs_billing_project_from_env, get_shared_runtime, load_adc_credentials};
use gnomon::score::pipeline::{self, PipelineContext};
use gnomon::score::prepare;
use gnomon::score::reformat;
use gnomon::score::types::{GenomicRegion, PreparationResult};
use natord::compare;
use std::collections::HashMap;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::Instant;

// ========================================================================================
//                              Command-line interface definition
// ========================================================================================

#[derive(Parser, Debug)]
#[clap(
    name = "gnomon",
    version,
    about = "A high-performance engine for polygenic score calculation."
)]
struct Args {
    /// Path to a single score file or a directory containing multiple score files.
    #[clap(value_name = "SCORE_PATH")]
    score: PathBuf,

    /// Path to a file containing a list of individual IDs (IIDs) to include.
    /// If not provided, all individuals in the .fam file will be scored.
    #[clap(long)]
    keep: Option<PathBuf>,

    /// Path to the PLINK .bed file, or a directory containing a single .bed file.
    #[clap(value_name = "GENOTYPE_PATH")]
    input_path: PathBuf,
}

// ========================================================================================
//                              The main orchestration logic
// ========================================================================================

// Main function removed as it's now called through the main binary's subcommand system
// and was causing dead_code warnings.

/// Public interface for calling gnomon with explicit arguments
pub fn run_gnomon_with_args(
    input_path: PathBuf,
    score: PathBuf,
    keep: Option<PathBuf>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args {
        score,
        keep,
        input_path,
    };
    run_gnomon_impl(args)
}

/// The primary application logic
// Function removed to eliminate dead code warnings

/// Core implementation that takes args as parameter
static RAYON_INIT: Once = Once::new();

fn run_gnomon_impl(args: Args) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Initialize the Rayon global thread pool to use all available cores.
    RAYON_INIT.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .build_global()
            .expect("failed to initialize Rayon global thread pool");
    });

    let overall_start_time = Instant::now();
    let fileset_prefixes = resolve_filesets(&args.input_path)?;

    // --- Phase 1a: Score File Resolution ---
    // This block resolves the --score argument into a definitive list of files.
    let (resolved_score_files, score_regions_map): (Vec<PathBuf>, HashMap<String, GenomicRegion>) = {
        let score_arg_str = args.score.to_string_lossy();

        if !args.score.exists() && score_arg_str.contains("PGS") {
            let scores_cache_dir = {
                let p0 = &fileset_prefixes[0];
                let parent_local = p0
                    .to_string_lossy()
                    .starts_with("gs://")
                    .then(|| Path::new(".").to_path_buf())
                    .unwrap_or_else(|| p0.parent().unwrap_or_else(|| Path::new(".")).to_path_buf());
                parent_local.join("gnomon_score_cache")
            };

            let resolved =
                download::resolve_and_download_scores(&score_arg_str, &scores_cache_dir)?;
            (resolved.paths, resolved.regions)
        } else {
            let files = if args.score.is_dir() {
                fs::read_dir(&args.score)?
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|p| p.is_file())
                    .collect()
            } else {
                vec![args.score.clone()]
            };
            (files, HashMap::new())
        }
    };

    if resolved_score_files.is_empty() {
        return Err("No score files were found or resolved.".into());
    }

    // --- Phase 1b: Preparation ---
    // This phase is synchronous and CPU-bound. It parses all input files,
    // reconciles variants, and produces a "computation blueprint".
    let score_regions_ref = if score_regions_map.is_empty() {
        None
    } else {
        Some(&score_regions_map)
    };

    let prep_result = run_preparation_phase(
        &fileset_prefixes,
        &resolved_score_files,
        args.keep.as_deref(),
        score_regions_ref,
    )?;

    // --- Phase 2: Resource Allocation ---
    // A read-only context is created, which allocates all necessary memory pools
    // for the pipeline to use.
    let context = PipelineContext::new(Arc::clone(&prep_result));
    eprintln!("> Resource allocation complete.");

    // --- Phase 3: Pipeline Execution ---
    // This is the primary compute phase. It is a synchronous, blocking call that
    // returns the final, aggregated results upon completion.
    let computation_start = Instant::now();
    let (final_scores, final_counts) = pipeline::run(&context)?;
    eprintln!(
        "> Computation finished. Total pipeline time: {:.2?}",
        computation_start.elapsed()
    );

    // --- Phase 4: Finalization & Output ---
    // After all computation is complete, this synchronous phase writes the
    // final scores to disk. The first fileset's prefix determines the output name.
    finalize_and_write_output(
        &fileset_prefixes[0],
        &prep_result,
        &final_scores,
        &final_counts,
        score_regions_ref,
    )?;

    eprintln!(
        "\nSuccess! Total execution time: {:.2?}",
        overall_start_time.elapsed()
    );
    Ok(())
}

/// **Helper 1:** Encapsulates the entire preparation and file normalization phase.
///
/// This function is synchronous and CPU-bound. It takes a definitive list of
/// resolved score files, normalizes them into a consistent, sorted, gnomon-native
/// format, and then calls the main preparation logic to produce a "computation
// blueprint" (`PreparationResult`). All user-facing console output for this phase
// is handled here.
fn run_preparation_phase(
    fileset_prefixes: &[PathBuf],
    score_files: &[PathBuf],
    keep: Option<&Path>,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
) -> Result<Arc<PreparationResult>, Box<dyn Error + Send + Sync>> {
    if fileset_prefixes.len() > 1 {
        eprintln!(
            "> Found {} PLINK filesets, starting with: {}",
            fileset_prefixes.len(),
            fileset_prefixes[0].display()
        );
    } else {
        eprintln!("> Using PLINK prefix: {}", fileset_prefixes[0].display());
    }

    eprintln!(
        "> Normalizing and preparing {} score file(s)...",
        score_files.len()
    );
    let prep_phase_start = Instant::now();

    let mut native_score_files = Vec::with_capacity(score_files.len());
    for score_file_path in score_files {
        match reformat::is_gnomon_native_format(score_file_path) {
            Ok(true) => {
                // If a file is already in the native format, we assume it is correctly
                // sorted and ready for processing.
                native_score_files.push(score_file_path.clone());
            }
            Ok(false) => {
                // The file is not in native format, so we must convert it.
                // The reformatting function is intelligent and will produce a sorted file.
                let output_dir = score_file_path.parent().unwrap_or_else(|| Path::new("."));
                fs::create_dir_all(output_dir)?;

                eprintln!(
                    "> Info: Score file '{}' is not in native format. Attempting conversion...",
                    score_file_path.display()
                );
                let new_path = score_file_path.with_extension("gnomon.tsv");

                match reformat::reformat_pgs_file(score_file_path, &new_path) {
                    Ok(_) => {
                        eprintln!("> Success: Converted to '{}'.", new_path.display());
                        native_score_files.push(new_path);
                    }
                    Err(e) => {
                        // The reformatting failed. The `ReformatError` type now
                        // contains a rich, detailed diagnostic message. We can
                        // simply convert it into a boxed error and return it.
                        return Err(Box::new(e));
                    }
                }
            }
            Err(e) => {
                // This is a lower-level I/O error from just trying to open the file.
                return Err(format!(
                    "Error reading score file '{}': {}",
                    score_file_path.display(),
                    e
                )
                .into());
            }
        }
    }

    // --- Run the main preparation logic with the fully normalized and sorted files ---
    let prep = prepare::prepare_for_computation(
        fileset_prefixes,
        &native_score_files,
        keep,
        score_regions,
    )
    .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    eprintln!(
        "> Preparation complete in {:.2?}. Found {} individuals to score and {} overlapping variants across {} score(s).",
        prep_phase_start.elapsed(),
        prep.num_people_to_score,
        prep.num_reconciled_variants,
        prep.score_names.len()
    );

    Ok(Arc::new(prep))
}

/// **Helper:** Handles the final file writing.
///
/// This function is synchronous and takes the final results directly.
fn finalize_and_write_output(
    output_prefix: &Path,
    prep_result: &Arc<PreparationResult>,
    final_scores: &[f64],
    final_counts: &[u32],
    score_regions: Option<&HashMap<String, GenomicRegion>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (output_dir, out_stem) = if output_prefix.to_string_lossy().starts_with("gs://") {
        let stem = output_prefix
            .file_name()
            .map_or_else(|| OsString::from("gnomon_results"), OsString::from);
        (Path::new(".").to_path_buf(), stem)
    } else {
        let dir = output_prefix
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let stem = output_prefix
            .file_name()
            .map_or_else(|| OsString::from("gnomon_results"), OsString::from);
        (dir, stem)
    };
    fs::create_dir_all(&output_dir)?;
    let mut out_filename = out_stem;
    out_filename.push(".sscore");
    let out_path = output_dir.join(&out_filename);

    eprintln!(
        "> Writing {} scores per person to {}",
        prep_result.score_names.len(),
        out_path.display()
    );
    let output_start = Instant::now();

    write_scores_to_file(
        &out_path,
        &prep_result.final_person_iids,
        &prep_result.score_names,
        &prep_result.score_variant_counts,
        final_scores,
        final_counts,
        score_regions,
    )?;

    eprintln!("> Final output written in {:.2?}", output_start.elapsed());
    Ok(())
}

// ========================================================================================
//                                  Helper functions
// ========================================================================================

/// Discovers and validates all PLINK filesets from a given path.
///
/// Local (unchanged):
///   1) /path/to/prefix        -> prefix.{bed,bim,fam}
///   2) /path/to/prefix.bed    -> prefix.{bed,bim,fam}
///   3) /path/to/dir/          -> scan for *.bed in that dir, validate triads
///
/// Remote (new):
///   A) gs://bucket/prefix           -> prefix.{bed,bim,fam}
///   B) gs://bucket/prefix.bed       -> prefix.{bed,bim,fam}
///   C) gs://bucket/dir/             -> list all *.bed under that prefix
///   D) gs://bucket/dir/*            -> same as (C) (star is treated as “all under dir/”)
fn resolve_filesets(path: &Path) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    // --- GCS handling first ---
    if let Some(s) = path.to_str() {
        if is_gcs_uri_str(s) {
            return resolve_gcs_filesets(s);
        }
    }

    // --- ORIGINAL LOCAL LOGIC (unchanged) ---
    if !path.is_dir() {
        let prefix = if path.extension().is_some() {
            path.with_extension("")
        } else {
            path.to_path_buf()
        };

        if !prefix.with_extension("bed").is_file()
            || !prefix.with_extension("bim").is_file()
            || !prefix.with_extension("fam").is_file()
        {
            return Err(format!(
                "Input prefix '{}' does not correspond to a complete PLINK fileset (.bed, .bim, .fam).",
                prefix.display()
            )
            .into());
        }
        return Ok(vec![prefix]);
    }

    let mut bed_files: Vec<PathBuf> = fs::read_dir(path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "bed"))
        .collect();

    if bed_files.is_empty() {
        return Err(format!("No .bed files found in directory '{}'.", path.display()).into());
    }

    bed_files.sort_by(|a, b| compare(&a.to_string_lossy(), &b.to_string_lossy()));

    let mut prefixes = Vec::with_capacity(bed_files.len());
    for bed_path in bed_files {
        let prefix = bed_path.with_extension("");
        if !prefix.with_extension("bim").is_file() || !prefix.with_extension("fam").is_file() {
            return Err(format!(
                "Incomplete fileset for prefix '{}'. All .bed files in a directory must have corresponding .bim and .fam files.",
                prefix.display()
            )
            .into());
        }
        prefixes.push(prefix);
    }
    Ok(prefixes)
}

fn is_gcs_uri_str(s: &str) -> bool {
    s.starts_with("gs://")
}

fn split_gcs_uri_dir_and_leaf(raw: &str) -> Result<(String, String), Box<dyn Error + Send + Sync>> {
    let without = raw.trim_start_matches("gs://");
    let mut it = without.splitn(2, '/');
    let bucket = it.next().unwrap_or_default();
    let object = it.next().unwrap_or_default();
    if bucket.is_empty() {
        return Err(format!("Malformed GCS URI '{raw}': missing bucket").into());
    }
    Ok((bucket.to_string(), object.to_string()))
}

/// Robust GCS resolver that supports: exact triad prefix, *.bed in a "directory",
/// trailing slash, and star suffix.
fn resolve_gcs_filesets(uri: &str) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    use google_cloud_auth::credentials::{Credentials, anonymous::Builder as AnonymousCredentials};
    use google_cloud_storage::client::StorageControl;

    let wants_dir_scan = uri.ends_with("/*") || uri.ends_with('/');
    let normalized = if uri.ends_with("/*") {
        &uri[..uri.len() - 1]
    } else {
        uri
    };

    let (bucket, object) = split_gcs_uri_dir_and_leaf(normalized)?;

    let runtime = get_shared_runtime().map_err(|e| format!("{e}"))?;
    let user_project = gcs_billing_project_from_env();

    let make_control =
        |creds: Option<Credentials>| -> Result<StorageControl, Box<dyn Error + Send + Sync>> {
            let credentials = match creds {
                Some(existing) => existing,
                None => load_adc_credentials()
                    .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("{e}").into() })?,
            };

            runtime.block_on(async move {
                StorageControl::builder()
                    .with_credentials(credentials)
                    .build()
                    .await
                    .map_err(|e| -> Box<dyn Error + Send + Sync> {
                        format!("Failed to create Cloud Storage control client: {e}").into()
                    })
            })
        };
    let try_list_objects = |control: &StorageControl,
                            prefix: &str,
                            mut page_token: Option<String>|
     -> Result<
        (Vec<google_cloud_storage::model::Object>, Option<String>),
        Box<dyn Error + Send + Sync>,
    > {
        let mut req = control
            .list_objects()
            .set_parent(format!("projects/_/buckets/{bucket}"))
            .set_prefix(prefix.to_string());
        if let Some(tok) = page_token.take() {
            req = req.set_page_token(tok);
        }
        let resp = runtime
            .block_on(req.send())
            .map_err(|e| {
                let msg = e.to_string();
                if user_project.is_none() && msg.to_lowercase().contains("requester pays") {
                    format!("This is a Requester Pays bucket. Set GOOGLE_PROJECT (or `gcloud config set project ...`) and re-run. Original error while listing gs://{bucket}/{prefix}: {msg}")
                } else {
                    format!("Error listing gs://{bucket}/{prefix}: {msg}")
                }
            })?;
        let next = (!resp.next_page_token.is_empty()).then(|| resp.next_page_token.clone());
        Ok((resp.objects, next))
    };

    let try_head = |control: &StorageControl,
                    object_name: &str|
     -> Result<
        google_cloud_storage::model::Object,
        Box<dyn Error + Send + Sync>,
    > {
        let req = control
            .get_object()
            .set_bucket(format!("projects/_/buckets/{bucket}"))
            .set_object(object_name.to_string());
        runtime
            .block_on(req.send())
            .map_err(|e| {
                let msg = e.to_string();
                if user_project.is_none() && msg.to_lowercase().contains("requester pays") {
                    format!("This is a Requester Pays bucket. Set GOOGLE_PROJECT (or `gcloud config set project ...`) and re-run. Original error while fetching metadata for gs://{bucket}/{object_name}: {msg}").into()
                } else {
                    format!("Failed to fetch metadata for gs://{bucket}/{object_name}: {msg}").into()
                }
            })
    };

    let control = match make_control(None) {
        Ok(control) => control,
        Err(e) => {
            let e_msg = e.to_string();
            let anonymous_creds = AnonymousCredentials::new().build();
            match make_control(Some(anonymous_creds)) {
                Ok(control) => control,
                Err(e2) => {
                    return Err(Box::<dyn Error + Send + Sync>::from(format!(
                        "Unable to initialize Cloud Storage clients: {e_msg} / {e2}"
                    )));
                }
            }
        }
    };

    if !wants_dir_scan && !object.ends_with('/') {
        let triad_prefix = if object.ends_with(".bed") {
            object[..object.len() - 4].to_string()
        } else {
            object.clone()
        };

        for ext in ["bed", "bim", "fam"] {
            let name = format!("{triad_prefix}.{ext}");
            let _ = try_head(&control, &name)?;
        }
        return Ok(vec![PathBuf::from(format!("gs://{bucket}/{triad_prefix}"))]);
    }

    let scan_prefix = if object.is_empty() || object.ends_with('/') {
        object.clone()
    } else {
        format!("{object}/")
    };

    let mut page_token: Option<String> = None;
    let mut objects: Vec<google_cloud_storage::model::Object> = Vec::new();
    loop {
        let (mut items, next) = try_list_objects(&control, &scan_prefix, page_token)?;
        objects.append(&mut items);
        if next.is_none() {
            break;
        }
        page_token = next;
    }

    if objects.is_empty() {
        return Err(format!("No objects found under gs://{bucket}/{scan_prefix}").into());
    }

    use std::collections::{HashMap, HashSet};
    let mut by_prefix: HashMap<String, HashSet<String>> = HashMap::new();
    for obj in objects.into_iter().filter(|o| !o.name.ends_with('/')) {
        let name = obj.name;
        if !name.starts_with(&scan_prefix) {
            continue;
        }
        if let Some((base, ext)) = name.rsplit_once('.') {
            if ["bed", "bim", "fam"].contains(&ext) {
                by_prefix
                    .entry(base.to_string())
                    .or_default()
                    .insert(ext.to_string());
            }
        }
    }

    let mut complete: Vec<String> = by_prefix
        .into_iter()
        .filter_map(|(base, exts)| {
            if exts.contains("bed") && exts.contains("bim") && exts.contains("fam") {
                Some(base)
            } else {
                None
            }
        })
        .collect();

    if complete.is_empty() {
        return Err(format!(
            "No complete PLINK filesets (.bed/.bim/.fam) under gs://{bucket}/{scan_prefix}"
        )
        .into());
    }

    complete.sort_by(|a, b| compare(a, b));

    Ok(complete
        .into_iter()
        .map(|base| PathBuf::from(format!("gs://{bucket}/{base}")))
        .collect())
}

/// Writes the final calculated scores to a self-describing, tab-separated file.
/// This function now calculates the final per-variant average and missing percentage.
fn write_scores_to_file(
    path: &Path,
    person_iids: &[String],
    score_names: &[String],
    score_variant_counts: &[u32],
    sum_scores: &[f64],
    missing_counts: &[u32],
    score_regions: Option<&HashMap<String, GenomicRegion>>,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let num_scores = score_names.len();

    if let Some(regions) = score_regions {
        let mut wrote_metadata_header = false;
        for name in score_names {
            if let Some(region) = regions.get(name) {
                if !wrote_metadata_header {
                    writeln!(writer, "#REGION\tSCORE\tINTERVAL")?;
                    wrote_metadata_header = true;
                }
                writeln!(writer, "#REGION\t{name}\t{region}")?;
            }
        }
    }

    // Write the new, more descriptive, and correctly tab-separated header.
    write!(writer, "#IID")?;
    for name in score_names {
        write!(writer, "\t{name}_AVG\t{name}_MISSING_PCT")?;
    }
    writeln!(writer)?;

    let mut line_buffer = String::with_capacity(
        person_iids
            .first()
            .map_or(128, |s| s.len() + num_scores * 24),
    );
    let mut sum_score_chunks = sum_scores.chunks_exact(num_scores);
    let mut missing_count_chunks = missing_counts.chunks_exact(num_scores);
    let mut ryu_buffer_score = ryu::Buffer::new();
    let mut ryu_buffer_missing = ryu::Buffer::new();

    for iid in person_iids {
        let person_sum_scores = sum_score_chunks.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Mismatched number of persons and score rows during final write.",
            )
        })?;
        let person_missing_counts = missing_count_chunks.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Mismatched number of persons and missing count rows during final write.",
            )
        })?;

        line_buffer.clear();
        write!(&mut line_buffer, "{iid}").unwrap();

        for i in 0..num_scores {
            let final_sum_score = person_sum_scores[i];
            let missing_count = person_missing_counts[i];
            let total_variants_for_score = score_variant_counts[i];

            // The score is calculated based on the number of non-missing variants.
            // This behavior matches standard tools when mean-imputation is disabled.
            let variants_used = total_variants_for_score.saturating_sub(missing_count);

            let avg_score = if variants_used > 0 {
                final_sum_score / (variants_used as f64)
            } else {
                0.0
            };

            let missing_pct = if total_variants_for_score > 0 {
                (missing_count as f32 / total_variants_for_score as f32) * 100.0
            } else {
                0.0
            };

            // Write the correctly tab-separated data columns.
            write!(
                &mut line_buffer,
                "\t{}\t{}",
                ryu_buffer_score.format(avg_score),
                ryu_buffer_missing.format(missing_pct)
            )
            .unwrap();
        }
        writeln!(writer, "{line_buffer}")?;
    }

    writer.flush()
}
