use super::fit::{FitOptions, HwePcaError, HwePcaModel, LdConfig, LdWindow};
use super::io::{
    DatasetOutputError, GenotypeDataset, GenotypeIoError, ProjectionOutputPaths, SelectionPlan,
    load_hwe_model, save_fit_summary, save_hwe_model, save_projection_results,
    save_sample_manifest,
};
use super::progress::{fit_progress, projection_progress};
use super::project::ProjectionOptions;
use super::variant_filter::{VariantFilter, VariantKey, VariantListError};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// High-level commands that can be executed within the `map` module.
#[derive(Debug)]
pub enum MapCommand {
    Fit {
        genotype_path: PathBuf,
        variant_list: Option<PathBuf>,
        components: usize,
        ld: Option<LdWindow>,
    },
    Project {
        genotype_path: PathBuf,
    },
}

/// Errors that can occur when executing `map` commands.
#[derive(Debug)]
pub enum MapDriverError {
    Hwe(HwePcaError),
    Io(std::io::Error),
    Dataset(GenotypeIoError),
    Serialization(serde_json::Error),
    InvalidState(String),
}

impl fmt::Display for MapDriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hwe(err) => write!(f, "HWE PCA error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Dataset(err) => write!(f, "genotype dataset error: {err}"),
            Self::Serialization(err) => write!(f, "serialization error: {err}"),
            Self::InvalidState(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for MapDriverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Hwe(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::Dataset(err) => Some(err),
            Self::Serialization(err) => Some(err),
            Self::InvalidState(_) => None,
        }
    }
}

impl From<HwePcaError> for MapDriverError {
    fn from(value: HwePcaError) -> Self {
        Self::Hwe(value)
    }
}

impl From<std::io::Error> for MapDriverError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<GenotypeIoError> for MapDriverError {
    fn from(value: GenotypeIoError) -> Self {
        Self::Dataset(value)
    }
}

impl From<serde_json::Error> for MapDriverError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value)
    }
}

impl From<DatasetOutputError> for MapDriverError {
    fn from(value: DatasetOutputError) -> Self {
        match value {
            DatasetOutputError::Io(err) => Self::Io(err),
            DatasetOutputError::Serialization(err) => Self::Serialization(err),
            DatasetOutputError::InvalidState(msg) => Self::InvalidState(msg),
        }
    }
}

/// Execute the provided [`MapCommand`].
pub fn run(command: MapCommand) -> Result<(), MapDriverError> {
    match command {
        MapCommand::Fit {
            genotype_path,
            variant_list,
            components,
            ld,
        } => run_fit(&genotype_path, variant_list.as_deref(), components, ld),
        MapCommand::Project { genotype_path } => run_project(&genotype_path),
    }
}

fn run_fit(
    genotype_path: &Path,
    variant_list: Option<&Path>,
    components: usize,
    ld: Option<LdWindow>,
) -> Result<(), MapDriverError> {
    println!("=== HWE PCA model fitting ===");
    println!("Input genotype location: {}", genotype_path.display());

    let dataset = open_dataset(genotype_path)?;
    println!(
        "Resolved genotype data file: {}",
        dataset.data_path().display()
    );
    let variant_hint = dataset.variant_count_hint();
    let variant_display = match variant_hint {
        Some(count) if count > 0 => count.to_string(),
        _ => String::from("unknown"),
    };
    println!(
        "Dataset dimensions: {} samples × {} variants",
        dataset.n_samples(),
        variant_display
    );

    println!("Requested principal components: {components}");
    match &ld {
        Some(LdWindow::Sites(window)) => {
            println!("LD flattening enabled via --ld flag (sites window={window}).");
        }
        Some(LdWindow::BasePairs(bp)) => {
            println!("LD flattening enabled via --ld flag (bp window={bp}).");
        }
        None => println!("LD flattening disabled (default)."),
    }

    let mut variant_keys: Option<Arc<Vec<VariantKey>>> = None;
    let mut selection_plan = SelectionPlan::All;

    if let Some(list_path) = variant_list {
        let filter = VariantFilter::from_file(list_path)
            .map_err(|err| map_variant_list_error(list_path, err))?;

        match &dataset {
            GenotypeDataset::Plink(_) => {
                let selection = dataset.select_variants(&filter)?;
                if selection.indices.is_empty() {
                    return Err(MapDriverError::InvalidState(format!(
                        "Variant list {} did not match any variants in the dataset",
                        list_path.display()
                    )));
                }

                let matched = selection.matched_unique();
                let missing = selection.requested_unique.saturating_sub(matched);
                println!(
                    "Variant list matched {matched} of {} requested variants{}",
                    selection.requested_unique,
                    if missing > 0 {
                        format!(" ({} missing)", missing)
                    } else {
                        String::from("")
                    }
                );

                variant_keys = Some(Arc::new(selection.keys.clone()));
                selection_plan = SelectionPlan::ByIndices(selection.indices);
            }
            GenotypeDataset::Variants(_) => {
                let requested = filter.requested_unique();
                println!(
                    "Variant list contains {requested} unique variants; streaming match will be reported during fit",
                );
                selection_plan = SelectionPlan::ByKeys(Arc::new(filter));
            }
        }
    }

    let mut fit_options = FitOptions::default();
    if let Some(window_spec) = ld.clone() {
        println!("Computing LD normalization weights (local best-diagonal).");
        let mut ld_config = LdConfig {
            window: Some(window_spec.clone()),
            ridge: None,
            variant_keys: None,
        };
        if matches!(window_spec, LdWindow::BasePairs(_)) {
            let keys_arc = if let Some(existing) = &variant_keys {
                Arc::clone(existing)
            } else {
                let keys = dataset.variant_keys_for_plan(&selection_plan)?;
                Arc::new(keys)
            };
            ld_config.variant_keys = Some(keys_arc);
        }
        fit_options.ld = Some(ld_config);
    }

    let mut source = dataset.block_source_with_plan(selection_plan)?;
    let progress = fit_progress();
    let mut model = HwePcaModel::fit_k_with_options_and_progress(
        &mut source,
        components,
        &fit_options,
        &progress,
    )?;

    if let Some(outcome) = source.take_selection_outcome() {
        if outcome.requested_unique > 0 {
            let matched = outcome.matched_keys.len();
            let missing = outcome.missing_keys.len();
            println!(
                "Variant list matched {matched} of {} requested variants{}",
                outcome.requested_unique,
                if missing > 0 {
                    format!(" ({} missing)", missing)
                } else {
                    String::from("")
                }
            );
        }
        if outcome.matched_keys.is_empty() {
            return Err(MapDriverError::InvalidState(
                "Variant list did not match any variants in the dataset".into(),
            ));
        }
        variant_keys = Some(Arc::new(outcome.matched_keys));
    }

    if let Some(keys_arc) = variant_keys {
        let keys = Arc::try_unwrap(keys_arc).unwrap_or_else(|arc| (*arc).clone());
        model.set_variant_keys(Some(keys));
    }

    if let Some(ld) = model.ld() {
        if let Some(bp) = ld.bp_window {
            println!(
                "LD weighting summary: {} variants, sites_window={} bp_window_total_bp={} ridge={:.3e}",
                ld.weights.len(),
                ld.window,
                bp,
                ld.ridge
            );
        } else {
            println!(
                "LD weighting summary: {} variants, window={} ridge={:.3e}",
                ld.weights.len(),
                ld.window,
                ld.ridge
            );
        }
    }

    let retained = model.components();
    if retained == components {
        println!(
            "Retained {retained} principal components across {} samples ({} variants)",
            model.n_samples(),
            model.n_variants()
        );
    } else {
        println!(
            "Retained {retained} principal components (requested {components}) across {} samples ({} variants)",
            model.n_samples(),
            model.n_variants()
        );
        println!("Note: dataset provides only {retained} positive principal components.");
    }

    let model_path = save_hwe_model(&dataset, &model)?;
    let manifest_path = save_sample_manifest(&dataset)?;
    let summary_path = save_fit_summary(&dataset, &model)?;

    println!("Generated output artifacts:");
    println!("  • Model JSON      : {}", model_path.display());
    println!("  • Sample manifest : {}", manifest_path.display());
    println!("  • Fit summary     : {}", summary_path.display());

    Ok(())
}

fn run_project(genotype_path: &Path) -> Result<(), MapDriverError> {
    println!("=== Sample projection into PCA space ===");
    println!("Input genotype location: {}", genotype_path.display());

    let dataset = open_dataset(genotype_path)?;
    println!(
        "Resolved genotype data file: {}",
        dataset.data_path().display()
    );
    let variant_hint = dataset.variant_count_hint();
    let variant_display = match variant_hint {
        Some(count) if count > 0 => count.to_string(),
        _ => String::from("unknown"),
    };
    println!(
        "Dataset dimensions: {} samples × {} variants",
        dataset.n_samples(),
        variant_display
    );

    let model = load_hwe_model(&dataset)?;
    println!(
        "Loaded model with {} principal components spanning {} variants",
        model.components(),
        model.n_variants()
    );

    let mut selection_plan = SelectionPlan::All;

    if let Some(keys) = model.variant_keys() {
        match &dataset {
            GenotypeDataset::Plink(_) => {
                let selection = dataset.select_variants_by_keys(keys)?;
                if !selection.missing.is_empty() {
                    return Err(MapDriverError::InvalidState(
                        "Projection dataset is missing variants required by the model".into(),
                    ));
                }
                if selection.indices.len() != model.n_variants() {
                    return Err(MapDriverError::InvalidState(format!(
                        "Model expects {} variants but matched {} in projection dataset",
                        model.n_variants(),
                        selection.indices.len()
                    )));
                }
                println!(
                    "Using stored variant subset with {} variants for projection",
                    selection.indices.len()
                );
                selection_plan = SelectionPlan::ByIndices(selection.indices);
            }
            GenotypeDataset::Variants(_) => {
                println!(
                    "Using stored variant subset with {} variants for projection",
                    model.n_variants()
                );
                let filter = VariantFilter::from_keys(keys.iter().cloned());
                selection_plan = SelectionPlan::ByKeys(Arc::new(filter));
            }
        }
    } else if let Some(known) = dataset.variant_count_hint() {
        if known > 0 && known != model.n_variants() {
            return Err(MapDriverError::InvalidState(format!(
                "Model expects {} variants but dataset provides {}",
                model.n_variants(),
                known
            )));
        }
    }

    let mut source = dataset.block_source_with_plan(selection_plan)?;
    let options = ProjectionOptions::default();
    let projector = model.projector();
    let progress = projection_progress();
    let result = projector.project_with_options_and_progress(&mut source, &options, &*progress)?;

    if let Some(outcome) = source.take_selection_outcome() {
        if !outcome.missing_keys.is_empty() {
            return Err(MapDriverError::InvalidState(
                "Projection dataset is missing variants required by the model".into(),
            ));
        }
        if outcome.matched_keys.len() != model.n_variants() {
            return Err(MapDriverError::InvalidState(format!(
                "Model expects {} variants but matched {} in projection dataset",
                model.n_variants(),
                outcome.matched_keys.len()
            )));
        }
    }

    let ProjectionOutputPaths { scores, alignment } = save_projection_results(&dataset, &result)?;

    println!("Generated projection outputs:");
    println!("  • Scores    : {}", scores.display());
    if let Some(path) = alignment {
        println!("  • Alignment : {}", path.display());
    }

    println!("Projection complete for {} samples", result.scores.nrows());

    Ok(())
}

fn open_dataset(path: &Path) -> Result<GenotypeDataset, MapDriverError> {
    Ok(GenotypeDataset::open(path)?)
}

fn map_variant_list_error(path: &Path, err: VariantListError) -> MapDriverError {
    match err {
        VariantListError::Io(io_err) => MapDriverError::Io(io_err),
        VariantListError::Parse { line, message } => MapDriverError::InvalidState(format!(
            "Failed to parse variant list {} (line {}): {}",
            path.display(),
            line,
            message
        )),
    }
}

#[cfg(test)]
mod tests {
    use crate::map::fit::HwePcaModel;
    use crate::map::io::{
        GenotypeDataset, ProjectionOutputPaths, load_hwe_model, save_hwe_model,
        save_projection_results,
    };
    use crate::map::project::ProjectionOptions;
    use crate::map::variant_filter::{VariantFilter, VariantKey, VariantSelection};
    use crate::shared::files::{VariantCompression, VariantFormat, open_variant_source};
    use noodles_bcf::io::Reader as BcfReader;
    use noodles_bgzf::io::Reader as BgzfReader;
    use noodles_vcf::variant::RecordBuf;
    use noodles_vcf::variant::record::AlternateBases as _;
    use noodles_vcf::variant::record::samples::keys::key;
    use noodles_vcf::variant::record_buf::samples::sample::Value;
    use noodles_vcf::variant::record_buf::samples::sample::value::Array;
    use noodles_vcf::variant::record_buf::samples::sample::value::genotype::{
        Allele as SampleAllele, Genotype as SampleGenotype,
    };
    use std::error::Error;
    use std::fmt::Write as _;
    use std::fs::{self, File};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::str::FromStr;
    use std::time::Duration;
    use tempfile::{NamedTempFile, tempdir};
    use zip::read::ZipArchive;

    use reqwest::blocking::Client;
    use std::io::{self, Cursor};

    const TEST_COMPONENTS: usize = 8;
    const HGDP_REMOTE_VARIANT_LIST: &str =
        "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv";
    const MANUAL_VARIANT_TARGET: usize = 128;

    fn first_variant_keys(path: &Path, limit: usize) -> Result<Vec<VariantKey>, Box<dyn Error>> {
        let source =
            open_variant_source(path).map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        match (source.format(), source.compression()) {
            (VariantFormat::Bcf, VariantCompression::Plain) => {
                collect_bcf_variant_keys(BcfReader::from(source), limit)
            }
            (VariantFormat::Bcf, VariantCompression::Bgzf) => {
                collect_bcf_variant_keys(BcfReader::from(BgzfReader::new(source)), limit)
            }
            (other_format, other_compression) => Err(format!(
                "unsupported variant source combination: format={other_format:?} compression={other_compression:?}"
            )
            .into()),
        }
    }

    fn collect_bcf_variant_keys<R: Read>(
        mut reader: BcfReader<R>,
        limit: usize,
    ) -> Result<Vec<VariantKey>, Box<dyn Error>> {
        let header = reader
            .read_header()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let mut record = RecordBuf::default();
        let mut keys = Vec::new();

        while keys.len() < limit {
            let bytes = reader
                .read_record_buf(&header, &mut record)
                .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
            if bytes == 0 {
                break;
            }

            if let Some(position) = record.variant_start() {
                let chrom = record.reference_sequence_name().to_string();
                let key = VariantKey::new(&chrom, position.get() as u64);
                keys.push(key);
            }
        }

        Ok(keys)
    }

    fn download_and_extract(
        client: &Client,
        url: &str,
        expected_filename: &str,
        output_dir: &Path,
    ) -> Result<PathBuf, Box<dyn Error>> {
        let response = client.get(url).send()?;
        if !response.status().is_success() {
            return Err(format!("failed to download {url}: status {}", response.status()).into());
        }

        let bytes = response.bytes()?;
        let cursor = Cursor::new(bytes.to_vec());
        let mut archive = ZipArchive::new(cursor)?;
        let mut extracted_path = None;

        for idx in 0..archive.len() {
            let mut entry = archive.by_index(idx)?;
            if !entry.is_file() {
                continue;
            }

            let enclosed = entry
                .enclosed_name()
                .ok_or_else(|| format!("archive entry has invalid name: {}", entry.name()))?
                .to_path_buf();
            let file_name = enclosed
                .file_name()
                .ok_or_else(|| format!("archive entry missing file name: {}", enclosed.display()))?
                .to_owned();
            let out_path = output_dir.join(&file_name);
            let mut out_file = File::create(&out_path)?;
            io::copy(&mut entry, &mut out_file)?;

            let matches_expected = file_name
                .to_str()
                .map(|name| name == expected_filename)
                .unwrap_or(false);
            if matches_expected {
                extracted_path = Some(out_path);
                break;
            }
        }

        extracted_path
            .ok_or_else(|| format!("{expected_filename} not found in archive {url}").into())
    }

    #[test]
    fn fit_and_project_downloaded_plink_dataset() -> Result<(), Box<dyn Error>> {
        let work_dir = tempdir()?;
        let data_dir = work_dir.path().join("data");
        fs::create_dir(&data_dir)?;

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .user_agent("gnomon-test/fit-and-project")
            .build()?;

        let archives = [
            (
                "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.fam.zip",
                "chr22_subset50.fam",
            ),
            (
                "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.bim.zip",
                "chr22_subset50.bim",
            ),
            (
                "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.bed.zip",
                "chr22_subset50.bed",
            ),
        ];

        for (url, expected) in archives {
            download_and_extract(&client, url, expected, &data_dir)?;
        }

        let bed_path = data_dir.join("chr22_subset50.bed");
        let dataset = GenotypeDataset::open(&bed_path)?;

        let mut fit_source = dataset.block_source()?;
        let model = HwePcaModel::fit_k(&mut fit_source, TEST_COMPONENTS)?;
        let model_path = save_hwe_model(&dataset, &model)?;
        assert!(model_path.exists(), "expected saved model to exist");

        let reloaded = load_hwe_model(&dataset)?;
        assert_eq!(reloaded.components(), model.components());

        let mut projection_source = dataset.block_source()?;
        let mut options = ProjectionOptions::default();
        options.return_alignment = true;
        let result = reloaded
            .projector()
            .project_with_options(&mut projection_source, &options)?;

        assert_eq!(result.scores.nrows(), dataset.n_samples());
        assert_eq!(result.scores.ncols(), reloaded.components());

        let ProjectionOutputPaths { scores, alignment } =
            save_projection_results(&dataset, &result)?;

        assert!(scores.exists(), "projection scores were not written");
        if let Some(alignment) = alignment {
            assert!(alignment.exists(), "projection alignment was not written");
        }

        Ok(())
    }

    const HGDP_CHR20_BCF: &str = "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/\
         hgdp1kgp_chr20.filtered.SNV_INDEL.phased.shapeit5.bcf";
    const HGDP_FULL_DATASET: &str =
        "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/";

    fn gnomon_binary_path() -> Result<PathBuf, Box<dyn Error>> {
        use std::env;

        let current_exe = env::current_exe()?;
        let binary_dir = current_exe
            .parent()
            .and_then(|deps| deps.parent())
            .ok_or_else(|| "unable to determine cargo target directory")?;

        let binary_name = format!("gnomon{}", env::consts::EXE_SUFFIX);
        let binary_path = binary_dir.join(&binary_name);

        if binary_path.exists() {
            return Ok(binary_path);
        }

        let target_dir = binary_dir
            .parent()
            .ok_or_else(|| "unable to determine cargo target directory")?;
        let workspace_root = target_dir
            .parent()
            .ok_or_else(|| "unable to determine workspace root for cargo build")?;

        let profile = binary_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "unable to determine cargo profile")?;

        let mut build_cmd = Command::new("cargo");
        build_cmd
            .arg("build")
            .arg("--bin")
            .arg("gnomon")
            .current_dir(workspace_root);

        match profile {
            "debug" => {}
            "release" => {
                build_cmd.arg("--release");
            }
            other => {
                build_cmd.arg("--profile").arg(other);
            }
        }

        let output = build_cmd
            .output()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        if !output.status.success() {
            return Err(format!(
                "failed to build gnomon binary: status={:?} stdout={} stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        if binary_path.exists() {
            Ok(binary_path)
        } else {
            Err(format!("gnomon binary not found at {}", binary_path.display()).into())
        }
    }

    struct LimitedReader<R> {
        inner: R,
        consumed: u64,
        limit: u64,
    }

    impl<R> LimitedReader<R> {
        fn new(inner: R, limit: u64) -> Self {
            Self {
                inner,
                consumed: 0,
                limit,
            }
        }

        fn into_inner(self) -> (R, u64) {
            (self.inner, self.consumed)
        }
    }

    impl<R: Read> Read for LimitedReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            if self.consumed >= self.limit {
                return Ok(0);
            }
            let remaining = (self.limit - self.consumed) as usize;
            let to_read = remaining.min(buf.len());
            let n = self.inner.read(&mut buf[..to_read])?;
            self.consumed += n as u64;
            Ok(n)
        }
    }

    struct RecordingReader<R> {
        inner: R,
        recorded: Vec<u8>,
        max_record: usize,
    }

    impl<R> RecordingReader<R> {
        fn new(inner: R, max_record: usize) -> Self {
            Self {
                inner,
                recorded: Vec::with_capacity(max_record.min(4096)),
                max_record,
            }
        }

        fn recorded_bytes(&self) -> &[u8] {
            &self.recorded
        }
    }

    impl<R: Read> Read for RecordingReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            if n == 0 {
                eprintln!("[partial-test] RecordingReader: inner returned EOF");
            } else {
                eprintln!("[partial-test] RecordingReader: read {n} bytes");
            }
            if n > 0 && self.recorded.len() < self.max_record {
                let remaining = self.max_record - self.recorded.len();
                let to_copy = remaining.min(n);
                self.recorded.extend_from_slice(&buf[..to_copy]);
                eprintln!(
                    "[partial-test] RecordingReader: recorded {} total bytes",
                    self.recorded.len()
                );
            }
            Ok(n)
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum BcfCompression {
        Bgzf,
        Plain,
    }

    fn detect_compression(path: &Path) -> Result<(BcfCompression, Vec<u8>), Box<dyn Error>> {
        let mut source =
            open_variant_source(path).map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let mut prefix = vec![0u8; 4096];
        let mut total = 0usize;
        while total < prefix.len() {
            let read = source.read(&mut prefix[total..])?;
            if read == 0 {
                break;
            }
            total += read;
        }
        prefix.truncate(total);

        let compression = if prefix.starts_with(&[0x1F, 0x8B]) {
            BcfCompression::Bgzf
        } else if prefix.starts_with(b"BCF") {
            BcfCompression::Plain
        } else {
            return Err(format!(
                "Unable to determine compression for {}: prefix {:?}",
                path.display(),
                &prefix[..prefix.len().min(16)]
            )
            .into());
        };

        Ok((compression, prefix))
    }

    fn hexdump(bytes: &[u8]) -> String {
        let mut out = String::new();
        for (idx, byte) in bytes.iter().enumerate() {
            if idx > 0 {
                if idx % 16 == 0 {
                    out.push('\n');
                } else if idx % 4 == 0 {
                    out.push(' ');
                }
            }
            let _ = write!(&mut out, "{:02X}", byte);
        }
        out
    }

    fn dosage_from_value(value: &Value) -> Option<f64> {
        match value {
            Value::Integer(n) => {
                eprintln!("[partial-test] dosage_from_value: integer {n}");
                Some(*n as f64)
            }
            Value::Float(n) => {
                eprintln!("[partial-test] dosage_from_value: float {n}");
                Some(*n as f64)
            }
            Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    eprintln!("[partial-test] dosage_from_value: empty string");
                    None
                } else if let Ok(parsed) = trimmed.parse::<f64>() {
                    eprintln!(
                        "[partial-test] dosage_from_value: parsed numeric string '{trimmed}' -> {parsed}"
                    );
                    Some(parsed)
                } else {
                    eprintln!(
                        "[partial-test] dosage_from_value: attempting genotype parse for '{trimmed}'"
                    );
                    SampleGenotype::from_str(trimmed)
                        .ok()
                        .and_then(|genotype| dosage_from_genotype(genotype.as_ref()))
                }
            }
            Value::Array(Array::Integer(values)) => {
                let result = values.get(0).copied().flatten().map(|n| n as f64);
                eprintln!(
                    "[partial-test] dosage_from_value: integer array {:?} -> {:?}",
                    values, result
                );
                result
            }
            Value::Array(Array::Float(values)) => {
                let result = values.get(0).copied().flatten().map(|n| n as f64);
                eprintln!(
                    "[partial-test] dosage_from_value: float array {:?} -> {:?}",
                    values, result
                );
                result
            }
            Value::Genotype(genotype) => {
                eprintln!("[partial-test] dosage_from_value: genotype {:?}", genotype);
                dosage_from_genotype(genotype.as_ref())
            }
            other => {
                eprintln!("[partial-test] dosage_from_value: unsupported value {other:?}");
                None
            }
        }
    }

    fn dosage_from_genotype(genotype: &[SampleAllele]) -> Option<f64> {
        let mut total = 0.0;
        for (idx, allele) in genotype.iter().enumerate() {
            eprintln!("[partial-test] dosage_from_genotype: allele[{idx}] = {allele:?}");
            match allele.position() {
                Some(0) => {}
                Some(_) => total += 1.0,
                None => return None,
            }
        }
        eprintln!("[partial-test] dosage_from_genotype: total dosage {total}");
        Some(total)
    }

    fn process_reader<R: Read>(
        reader: &mut BcfReader<R>,
    ) -> Result<(usize, Option<f64>), Box<dyn Error>> {
        eprintln!("[partial-test] reading header");
        let header = reader.read_header()?;
        let sample_count = header.sample_names().len();
        eprintln!(
            "[partial-test] header read: samples={} contigs={}",
            sample_count,
            header.contigs().len()
        );
        let sample_preview: Vec<_> = header.sample_names().iter().take(5).collect();
        eprintln!("[partial-test] sample preview: {sample_preview:?}");
        if let Some(contig) = header.contigs().keys().next() {
            eprintln!("[partial-test] first contig key: {contig}");
        }
        assert!(
            !header.sample_names().is_empty(),
            "BCF header should list at least one sample"
        );
        assert!(
            !header.contigs().is_empty(),
            "BCF header should include contig definitions"
        );

        let mut record = RecordBuf::default();
        eprintln!("[partial-test] reading first record");
        let bytes = reader.read_record_buf(&header, &mut record)?;
        eprintln!("[partial-test] record byte count: {bytes}");
        assert!(bytes > 0, "expected to decode the first variant record");
        let variant_start = record
            .variant_start()
            .expect("variant start position should be present");
        eprintln!(
            "[partial-test] record location: {}:{}",
            record.reference_sequence_name(),
            usize::from(variant_start)
        );
        assert!(
            usize::from(variant_start) > 0,
            "variant position must be positive"
        );
        assert!(
            !record.reference_bases().is_empty(),
            "reference allele should not be empty"
        );
        eprintln!(
            "[partial-test] reference bases: {}",
            record.reference_bases()
        );
        assert!(
            !record.alternate_bases().as_ref().is_empty(),
            "alternate allele set should not be empty"
        );
        let mut alt_preview = Vec::new();
        for alt in record.alternate_bases().iter() {
            match alt {
                Ok(allele) => alt_preview.push(allele.to_string()),
                Err(err) => alt_preview.push(format!("<err: {err}>",)),
            }
        }
        eprintln!("[partial-test] alternate alleles: {alt_preview:?}");
        eprintln!("[partial-test] record IDs: {:?}", record.ids());
        eprintln!("[partial-test] record filters: {:?}", record.filters());
        eprintln!("[partial-test] record format keys: {:?}", record.format());

        let samples = record.samples();
        let observed_samples = samples.values().count();
        eprintln!(
            "[partial-test] observed sample records: {} (expected {})",
            observed_samples, sample_count
        );
        assert_eq!(
            observed_samples, sample_count,
            "record should include per-sample data"
        );
        let ds_series = samples.select("DS");
        let gt_series = samples.select(key::GENOTYPE);
        eprintln!("[partial-test] DS present: {}", ds_series.is_some());
        eprintln!("[partial-test] GT present: {}", gt_series.is_some());
        let mut decoded = None;

        for idx in 0..sample_count {
            if let Some(series) = ds_series.as_ref() {
                let raw = series.get(idx);
                eprintln!("[partial-test] sample {idx} DS raw: {raw:?}");
                if let Some(Some(value)) = raw {
                    if let Some(dosage) = dosage_from_value(value) {
                        eprintln!("[partial-test] sample {idx} DS-derived dosage: {dosage}");
                        decoded = Some(dosage);
                        break;
                    }
                }
            }
            if let Some(series) = gt_series.as_ref() {
                let raw = series.get(idx);
                eprintln!("[partial-test] sample {idx} GT raw: {raw:?}");
                if let Some(Some(value)) = raw {
                    if let Some(dosage) = dosage_from_value(value) {
                        eprintln!("[partial-test] sample {idx} GT-derived dosage: {dosage}");
                        decoded = Some(dosage);
                        break;
                    }
                }
            }
        }

        eprintln!("[partial-test] decoded dosage result: {decoded:?}");
        assert!(
            decoded.is_some(),
            "expected to derive a dosage from DS or GT fields"
        );

        Ok((sample_count, decoded))
    }

    #[test]
    fn cli_fit_and_project_full_hgdp_dataset() -> Result<(), Box<dyn Error>> {
        let binary = gnomon_binary_path()?;

        let fit_output = Command::new(&binary)
            .arg("fit")
            .arg(HGDP_FULL_DATASET)
            .arg("--list")
            .arg(HGDP_REMOTE_VARIANT_LIST)
            .arg("--components")
            .arg(TEST_COMPONENTS.to_string())
            .output()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert!(
            fit_output.status.success(),
            "`gnomon fit` failed: status={:?}\nstdout={}\nstderr={}",
            fit_output.status,
            String::from_utf8_lossy(&fit_output.stdout),
            String::from_utf8_lossy(&fit_output.stderr)
        );

        let project_output = Command::new(&binary)
            .arg("project")
            .arg(HGDP_FULL_DATASET)
            .output()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert!(
            project_output.status.success(),
            "`gnomon project` failed: status={:?}\nstdout={}\nstderr={}",
            project_output.status,
            String::from_utf8_lossy(&project_output.stdout),
            String::from_utf8_lossy(&project_output.stderr)
        );

        Ok(())
    }

    #[test]
    fn download_partial_hgdp_chr20_header_and_first_record() -> Result<(), Box<dyn Error>> {
        const MAX_COMPRESSED_BYTES: u64 = 16 * 1024 * 1024;

        eprintln!(
            "[partial-test] starting partial download with limit {} bytes",
            MAX_COMPRESSED_BYTES
        );

        let path = Path::new(HGDP_CHR20_BCF);
        let (compression, probe_prefix) = detect_compression(path)?;
        eprintln!(
            "[partial-test] compression probe: {:?} ({} bytes captured)",
            compression,
            probe_prefix.len()
        );
        if !probe_prefix.is_empty() {
            let preview_len = probe_prefix.len().min(256);
            eprintln!(
                "[partial-test] probe hex preview ({} bytes):\n{}",
                preview_len,
                hexdump(&probe_prefix[..preview_len])
            );
        } else {
            eprintln!("[partial-test] probe produced no bytes");
        }

        match compression {
            BcfCompression::Bgzf => {
                eprintln!("[partial-test] compression detected: BGZF");
                let source = open_variant_source(path).map_err(|err| -> Box<dyn Error> {
                    eprintln!("[partial-test] failed to reopen remote BCF: {err}");
                    Box::new(err)
                })?;
                let recording = RecordingReader::new(source, 4096);
                eprintln!("[partial-test] recording reader initialized (bgzf path)");
                let limited = LimitedReader::new(recording, MAX_COMPRESSED_BYTES);
                eprintln!(
                    "[partial-test] limited reader created with limit {}",
                    MAX_COMPRESSED_BYTES
                );
                let bgzf_reader = BgzfReader::new(limited);
                eprintln!("[partial-test] BGZF reader instantiated");
                let mut reader = BcfReader::from(bgzf_reader);
                eprintln!("[partial-test] BCF reader ready (bgzf path)");

                let (_, decoded) = process_reader(&mut reader)?;

                let bgzf_reader = reader.into_inner();
                let limited = bgzf_reader.into_inner();
                let (recording, consumed) = limited.into_inner();
                let recorded_bytes = recording.recorded_bytes();
                eprintln!(
                    "[partial-test] final recorded compressed bytes: {}",
                    recorded_bytes.len()
                );
                if !recorded_bytes.is_empty() {
                    let preview_len = recorded_bytes.len().min(256);
                    eprintln!(
                        "[partial-test] final compressed prefix hex ({} bytes):\n{}",
                        preview_len,
                        hexdump(&recorded_bytes[..preview_len])
                    );
                }
                eprintln!("[partial-test] total compressed bytes consumed: {consumed}");
                assert!(
                    consumed > 0 && consumed <= MAX_COMPRESSED_BYTES,
                    "expected to read only a small prefix of the remote object (read {consumed} bytes)"
                );
                eprintln!("[partial-test] decoded dosage (bgzf path): {decoded:?}");
            }
            BcfCompression::Plain => {
                eprintln!("[partial-test] compression detected: plain BCF");
                let source = open_variant_source(path).map_err(|err| -> Box<dyn Error> {
                    eprintln!("[partial-test] failed to reopen remote BCF: {err}");
                    Box::new(err)
                })?;
                let recording = RecordingReader::new(source, 4096);
                eprintln!("[partial-test] recording reader initialized (plain path)");
                let limited = LimitedReader::new(recording, MAX_COMPRESSED_BYTES);
                eprintln!(
                    "[partial-test] limited reader created with limit {}",
                    MAX_COMPRESSED_BYTES
                );
                let mut reader = BcfReader::from(limited);
                eprintln!("[partial-test] BCF reader ready (plain path)");

                let (_, decoded) = process_reader(&mut reader)?;

                let limited = reader.into_inner();
                let (recording, consumed) = limited.into_inner();
                let recorded_bytes = recording.recorded_bytes();
                eprintln!(
                    "[partial-test] final recorded bytes (plain path): {}",
                    recorded_bytes.len()
                );
                if !recorded_bytes.is_empty() {
                    let preview_len = recorded_bytes.len().min(256);
                    eprintln!(
                        "[partial-test] final plain prefix hex ({} bytes):\n{}",
                        preview_len,
                        hexdump(&recorded_bytes[..preview_len])
                    );
                }
                eprintln!("[partial-test] total bytes consumed: {consumed}");
                assert!(
                    consumed > 0 && consumed <= MAX_COMPRESSED_BYTES,
                    "expected to read only a small prefix of the remote object (read {consumed} bytes)"
                );
                eprintln!("[partial-test] decoded dosage (plain path): {decoded:?}");
            }
        }

        Ok(())
    }

    fn run_fit_and_project_hgdp_chr20(variant_list: Option<&Path>) -> Result<(), Box<dyn Error>> {
        let dataset = GenotypeDataset::open(Path::new(HGDP_CHR20_BCF))
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let n_samples = dataset.n_samples();
        assert!(n_samples >= 3, "expected at least three samples for PCA");
        assert_eq!(dataset.samples().len(), n_samples);

        let mut selection: Option<VariantSelection> = None;
        let expected_variants: Option<usize> = if let Some(list_path) = variant_list {
            let filter = VariantFilter::from_file(list_path).map_err(|err| {
                let message = format!(
                    "failed to load variant list {}: {err:?}",
                    list_path.display()
                );
                Box::<dyn Error>::from(message)
            })?;
            let chosen = dataset
                .select_variants(&filter)
                .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
            assert!(
                !chosen.indices.is_empty(),
                "variant list {} did not match any variants in the dataset",
                list_path.display()
            );
            selection = Some(chosen);
            selection.as_ref().map(|sel| sel.indices.len())
        } else {
            dataset.variant_count_hint().or_else(|| {
                let count = dataset.n_variants();
                if count > 0 { Some(count) } else { None }
            })
        };

        if let Some(expected) = expected_variants {
            assert!(expected > 0, "expected at least one variant in dataset");
        }

        let mut train_source = dataset
            .block_source_with_selection(selection.as_ref().map(|sel| sel.indices.as_slice()))
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let mut model = HwePcaModel::fit_k(&mut train_source, TEST_COMPONENTS)
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        if let Some(sel) = &selection {
            model.set_variant_keys(Some(sel.keys.clone()));
        }

        assert_eq!(model.n_samples(), n_samples);
        let observed_variants = model.n_variants();
        assert!(
            observed_variants > 0,
            "expected at least one variant in dataset"
        );
        if let Some(expected) = expected_variants {
            assert_eq!(observed_variants, expected);
        } else if variant_list.is_none() {
            if let Some(inferred) = dataset.variant_count_hint() {
                assert_eq!(observed_variants, inferred);
            }
        }
        assert!(model.components() > 0);
        assert_eq!(model.explained_variance().len(), model.components());
        assert!(
            model
                .explained_variance()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(
            model
                .singular_values()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );

        let variance_ratios = model.explained_variance_ratio();
        assert_eq!(variance_ratios.len(), model.components());
        let variance_ratio_sum: f64 = variance_ratios.iter().copied().sum();
        assert!(
            variance_ratio_sum <= 1.0 + 1e-9,
            "explained variance ratios must sum to at most 1 (observed {variance_ratio_sum})"
        );

        drop(train_source);

        let mut inference_source = dataset
            .block_source_with_selection(selection.as_ref().map(|sel| sel.indices.as_slice()))
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let projection = model
            .projector()
            .project_with_options(&mut inference_source, &ProjectionOptions::default())
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert_eq!(projection.scores.nrows(), n_samples);
        assert_eq!(projection.scores.ncols(), model.components());
        assert!(projection.alignment.is_none());

        let scores = projection.scores.as_ref();
        for row in 0..scores.nrows() {
            for col in 0..scores.ncols() {
                let value = scores[(row, col)];
                assert!(
                    value.is_finite(),
                    "projection score for sample {row} PC {col} should be finite"
                );
            }
        }

        if let Some(sel) = &selection {
            if let Some(keys) = model.variant_keys() {
                assert_eq!(keys.len(), sel.indices.len());
            }
        }

        Ok(())
    }

    #[test]
    fn fit_and_project_hgdp_chr20() -> Result<(), Box<dyn Error>> {
        run_fit_and_project_hgdp_chr20(None)
    }

    #[test]
    fn fit_and_project_full_hgdp_chr20_with_manual_variant_list() -> Result<(), Box<dyn Error>> {
        let keys = first_variant_keys(Path::new(HGDP_CHR20_BCF), MANUAL_VARIANT_TARGET)?;
        assert_eq!(
            keys.len(),
            MANUAL_VARIANT_TARGET,
            "expected to collect the configured number of variants from HGDP chr20 dataset"
        );

        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "chrom\tpos")?;
        for key in keys.into_iter().take(MANUAL_VARIANT_TARGET) {
            writeln!(temp_file, "{}\t{}", key.chromosome, key.position)?;
        }
        temp_file.flush()?;

        run_fit_and_project_hgdp_chr20(Some(temp_file.path()))
    }

    #[test]
    fn fit_and_project_hgdp_chr20_with_remote_variant_list() -> Result<(), Box<dyn Error>> {
        run_fit_and_project_hgdp_chr20(Some(Path::new(HGDP_REMOTE_VARIANT_LIST)))
    }
}
