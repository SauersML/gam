use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::str;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::map::fit::{HwePcaModel, VariantBlockSource};
use crate::map::project::ProjectionResult;
use crate::map::variant_filter::{VariantFilter, VariantKey, VariantSelection};
use crate::score::pipeline::PipelineError;
use crate::shared::files::{
    BedSource, ReadMetrics, TextSource, VariantCompression, VariantFormat, VariantSource,
    list_variant_paths, open_bed_source, open_text_source, open_variant_source,
};
use noodles_bcf::Record as BcfRecord;
use noodles_bcf::io::Reader as BcfReader;
use noodles_bgzf::io::Reader as BgzfReader;
use noodles_vcf::io::Reader as VcfReader;
use noodles_vcf::{
    self as vcf, Record as VcfRecord,
    variant::RecordBuf,
    variant::record::samples::{
        keys::key,
        series::{self, Value as SeriesValue, value::Array as SeriesArray},
    },
};
use thiserror::Error;

const PLINK_HEADER_LEN: u64 = 3;

type DynBcfReader = BcfReader<Box<dyn Read + Send>>;
type DynVcfReader = VcfReader<Box<dyn BufRead + Send>>;

enum VariantStreamReader {
    Bcf(DynBcfReader),
    Vcf(DynVcfReader),
}

impl VariantStreamReader {
    fn read_header(&mut self) -> io::Result<vcf::Header> {
        match self {
            Self::Bcf(reader) => reader.read_header(),
            Self::Vcf(reader) => reader.read_header(),
        }
    }

    fn read_record_buf(
        &mut self,
        header: &vcf::Header,
        record: &mut RecordBuf,
    ) -> io::Result<usize> {
        match self {
            Self::Bcf(reader) => reader.read_record_buf(header, record),
            Self::Vcf(reader) => reader.read_record_buf(header, record),
        }
    }
}

#[derive(Debug, Error)]
pub enum PlinkIoError {
    #[error("pipeline I/O error: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("invalid PLINK .bed header: {0}")]
    InvalidHeader(String),
    #[error("unexpected end of .bed payload (expected {expected} bytes, found {actual})")]
    TruncatedBed { expected: u64, actual: u64 },
    #[error("malformed record in {path} at line {line}: {message}")]
    MalformedRecord {
        path: String,
        line: usize,
        message: String,
    },
    #[error("{path} is not valid UTF-8: {source}")]
    Utf8 {
        path: String,
        #[source]
        source: str::Utf8Error,
    },
}

#[derive(Debug, Error)]
pub enum VariantIoError {
    #[error("pipeline I/O error: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Variant dataset is missing sample names")]
    MissingSamples,
    #[error("Variant dataset contains no variant records")]
    NoVariants,
    #[error("Variant decode error: {0}")]
    Decode(String),
    #[error("Variant stream header did not match initial header when reopening")]
    HeaderMismatch,
    #[error("unexpected end of variant stream (expected {expected} variants, read {actual})")]
    UnexpectedEof { expected: usize, actual: usize },
}

#[derive(Debug, Error)]
pub enum GenotypeIoError {
    #[error(transparent)]
    Plink(#[from] PlinkIoError),
    #[error(transparent)]
    Variant(#[from] VariantIoError),
}

#[derive(Debug, Error)]
pub enum DatasetOutputError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("{0}")]
    InvalidState(String),
}

#[derive(Clone, Debug)]
pub struct SampleRecord {
    pub family_id: String,
    pub individual_id: String,
    pub paternal_id: String,
    pub maternal_id: String,
    pub sex: String,
    pub phenotype: String,
}

#[derive(Debug)]
pub enum GenotypeDataset {
    Plink(PlinkDataset),
    Variants(VcfLikeDataset),
}

#[derive(Clone, Debug)]
pub enum SelectionPlan {
    All,
    ByIndices(Vec<usize>),
    ByKeys(Arc<VariantFilter>),
}

#[derive(Clone, Debug, Default)]
pub struct SelectionOutcome {
    pub matched_keys: Vec<VariantKey>,
    pub missing_keys: Vec<VariantKey>,
    pub requested_unique: usize,
}

impl GenotypeDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GenotypeIoError> {
        let path = path.as_ref();
        if guess_is_variant_dataset(path) {
            Ok(Self::Variants(VcfLikeDataset::open(path)?))
        } else {
            Ok(Self::Plink(PlinkDataset::open(path)?))
        }
    }

    pub fn samples(&self) -> &[SampleRecord] {
        match self {
            Self::Plink(dataset) => dataset.samples(),
            Self::Variants(dataset) => dataset.samples(),
        }
    }

    pub fn n_samples(&self) -> usize {
        match self {
            Self::Plink(dataset) => dataset.n_samples(),
            Self::Variants(dataset) => dataset.n_samples(),
        }
    }

    pub fn n_variants(&self) -> usize {
        match self {
            Self::Plink(dataset) => dataset.n_variants(),
            Self::Variants(dataset) => dataset.n_variants(),
        }
    }

    pub fn variant_count_hint(&self) -> Option<usize> {
        match self {
            Self::Plink(dataset) => Some(dataset.n_variants()),
            Self::Variants(dataset) => dataset.variant_count_hint(),
        }
    }

    pub fn block_source(&self) -> Result<DatasetBlockSource, GenotypeIoError> {
        self.block_source_with_plan(SelectionPlan::All)
    }

    pub fn block_source_with_selection(
        &self,
        selection: Option<&[usize]>,
    ) -> Result<DatasetBlockSource, GenotypeIoError> {
        let plan = match selection {
            Some(indices) => SelectionPlan::ByIndices(indices.to_vec()),
            None => SelectionPlan::All,
        };
        self.block_source_with_plan(plan)
    }

    pub fn block_source_with_plan(
        &self,
        plan: SelectionPlan,
    ) -> Result<DatasetBlockSource, GenotypeIoError> {
        match (self, plan) {
            (Self::Plink(dataset), SelectionPlan::All) => {
                Ok(DatasetBlockSource::Plink(dataset.block_source()))
            }
            (Self::Plink(dataset), SelectionPlan::ByIndices(indices)) => Ok(
                DatasetBlockSource::Plink(dataset.block_source_with_selection(Some(indices))),
            ),
            (Self::Plink(dataset), SelectionPlan::ByKeys(filter)) => {
                let selection = dataset
                    .select_variants(filter.as_ref())
                    .map_err(GenotypeIoError::from)?;
                Ok(DatasetBlockSource::Plink(
                    dataset.block_source_with_selection(Some(selection.indices)),
                ))
            }
            (Self::Variants(dataset), plan) => Ok(DatasetBlockSource::Variants(
                dataset.block_source_with_plan(plan)?,
            )),
        }
    }

    pub fn select_variants(
        &self,
        filter: &VariantFilter,
    ) -> Result<VariantSelection, GenotypeIoError> {
        match self {
            Self::Plink(dataset) => dataset
                .select_variants(filter)
                .map_err(GenotypeIoError::from),
            Self::Variants(dataset) => dataset
                .select_variants(filter)
                .map_err(GenotypeIoError::from),
        }
    }

    pub fn select_variants_by_keys(
        &self,
        keys: &[VariantKey],
    ) -> Result<VariantSelection, GenotypeIoError> {
        let filter = VariantFilter::from_keys(keys.iter().cloned());
        let mut selection = self.select_variants(&filter)?;

        let original_indices = std::mem::take(&mut selection.indices);
        let original_keys = std::mem::take(&mut selection.keys);

        let mut matched = HashMap::with_capacity(original_keys.len());
        for (index, key) in original_indices.into_iter().zip(original_keys.into_iter()) {
            matched.insert(key.clone(), (index, key));
        }

        let mut ordered_indices = Vec::with_capacity(matched.len());
        let mut ordered_keys = Vec::with_capacity(matched.len());

        for key in keys {
            if let Some((index, stored_key)) = matched.remove(key) {
                ordered_indices.push(index);
                ordered_keys.push(stored_key);
            }
        }

        selection.indices = ordered_indices;
        selection.keys = ordered_keys;
        Ok(selection)
    }

    pub fn data_path(&self) -> &Path {
        match self {
            Self::Plink(dataset) => dataset.bed_path(),
            Self::Variants(dataset) => dataset.input_path(),
        }
    }

    pub fn output_path(&self, filename: &str) -> PathBuf {
        match self {
            Self::Plink(dataset) => dataset.output_path(filename),
            Self::Variants(dataset) => dataset.output_path(filename),
        }
    }

    pub fn variant_keys_for_plan(
        &self,
        plan: &SelectionPlan,
    ) -> Result<Vec<VariantKey>, GenotypeIoError> {
        match (self, plan) {
            (Self::Plink(dataset), SelectionPlan::All) => Ok(dataset.variant_keys_all()?),
            (Self::Plink(dataset), SelectionPlan::ByIndices(indices)) => {
                let keys = dataset.variant_keys_all()?;
                let mut selected = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if let Some(key) = keys.get(idx) {
                        selected.push(key.clone());
                    } else {
                        return Err(GenotypeIoError::Plink(PlinkIoError::InvalidHeader(
                            format!("variant index {idx} exceeds dataset bounds"),
                        )));
                    }
                }
                Ok(selected)
            }
            (Self::Plink(dataset), SelectionPlan::ByKeys(filter)) => {
                let selection = dataset.select_variants(filter)?;
                Ok(selection.keys)
            }
            (Self::Variants(dataset), SelectionPlan::All) => Ok(dataset.variant_keys_all()?),
            (Self::Variants(dataset), SelectionPlan::ByIndices(indices)) => {
                let keys = dataset.variant_keys_all()?;
                let mut selected = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if let Some(key) = keys.get(idx) {
                        selected.push(key.clone());
                    } else {
                        return Err(GenotypeIoError::Variant(VariantIoError::UnexpectedEof {
                            expected: indices.len(),
                            actual: selected.len(),
                        }));
                    }
                }
                Ok(selected)
            }
            (Self::Variants(dataset), SelectionPlan::ByKeys(filter)) => {
                let selection = dataset.select_variants(filter)?;
                Ok(selection.keys)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionOutputPaths {
    pub scores: PathBuf,
    pub alignment: Option<PathBuf>,
}

pub fn save_hwe_model(
    dataset: &GenotypeDataset,
    model: &HwePcaModel,
) -> Result<PathBuf, DatasetOutputError> {
    let model_path = dataset.output_path("hwe.json");
    prepare_output_path(&model_path)?;

    let file = File::create(&model_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, model)?;
    writer.flush()?;

    Ok(model_path)
}

pub fn load_hwe_model(dataset: &GenotypeDataset) -> Result<HwePcaModel, DatasetOutputError> {
    let model_path = dataset.output_path("hwe.json");
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let model: HwePcaModel = serde_json::from_reader(reader)?;

    if let Some(keys) = model.variant_keys() {
        if keys.len() != model.n_variants() {
            return Err(DatasetOutputError::InvalidState(format!(
                "Model stores {} variant keys but reports {} variants",
                keys.len(),
                model.n_variants()
            )));
        }
    } else if let Some(hint) = dataset.variant_count_hint() {
        if hint > 0 && model.n_variants() != hint {
            return Err(DatasetOutputError::InvalidState(format!(
                "Model expects {} variants but dataset provides {}",
                model.n_variants(),
                hint
            )));
        }
    }

    Ok(model)
}

pub fn save_projection_results(
    dataset: &GenotypeDataset,
    result: &ProjectionResult,
) -> Result<ProjectionOutputPaths, DatasetOutputError> {
    let scores = result.scores.as_ref();
    let samples = dataset.samples();

    if samples.len() != scores.nrows() {
        return Err(DatasetOutputError::InvalidState(format!(
            "Projection scores contain {} rows but dataset has {} samples",
            scores.nrows(),
            samples.len()
        )));
    }

    let scores_path = dataset.output_path("projection.scores.tsv");
    prepare_output_path(&scores_path)?;
    let mut writer = BufWriter::new(File::create(&scores_path)?);

    write!(writer, "FID\tIID")?;
    for idx in 0..scores.ncols() {
        write!(writer, "\tPC{}", idx + 1)?;
    }
    writeln!(writer)?;

    for (row, sample) in samples.iter().enumerate() {
        write!(writer, "{}\t{}", sample.family_id, sample.individual_id)?;
        for col in 0..scores.ncols() {
            let value = scores[(row, col)];
            write!(writer, "\t{}", value)?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;

    let mut alignment_path = None;
    if let Some(alignment) = result.alignment.as_ref() {
        let path = dataset.output_path("projection.alignment.tsv");
        prepare_output_path(&path)?;
        let mut writer = BufWriter::new(File::create(&path)?);

        write!(writer, "FID\tIID")?;
        for idx in 0..alignment.ncols() {
            write!(writer, "\tPC{}", idx + 1)?;
        }
        writeln!(writer)?;

        for (row, sample) in samples.iter().enumerate() {
            write!(writer, "{}\t{}", sample.family_id, sample.individual_id)?;
            for col in 0..alignment.ncols() {
                let value = alignment[(row, col)];
                write!(writer, "\t{}", value)?;
            }
            writeln!(writer)?;
        }

        writer.flush()?;
        alignment_path = Some(path);
    }

    Ok(ProjectionOutputPaths {
        scores: scores_path,
        alignment: alignment_path,
    })
}

pub fn save_sample_manifest(dataset: &GenotypeDataset) -> Result<PathBuf, DatasetOutputError> {
    let manifest_path = dataset.output_path("samples.tsv");
    prepare_output_path(&manifest_path)?;
    let mut writer = BufWriter::new(File::create(&manifest_path)?);

    writeln!(writer, "FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE")?;

    for record in dataset.samples() {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}",
            record.family_id,
            record.individual_id,
            record.paternal_id,
            record.maternal_id,
            record.sex,
            record.phenotype
        )?;
    }

    writer.flush()?;
    Ok(manifest_path)
}

pub fn save_fit_summary(
    dataset: &GenotypeDataset,
    model: &HwePcaModel,
) -> Result<PathBuf, DatasetOutputError> {
    let summary_path = dataset.output_path("hwe.summary.tsv");
    prepare_output_path(&summary_path)?;
    let mut writer = BufWriter::new(File::create(&summary_path)?);

    writeln!(writer, "metric\tvalue")?;
    writeln!(writer, "n_samples\t{}", model.n_samples())?;
    writeln!(writer, "n_variants\t{}", model.n_variants())?;

    for (idx, variance) in model.explained_variance().iter().copied().enumerate() {
        writeln!(writer, "explained_variance_PC{}\t{}", idx + 1, variance)?;
    }

    let ratios = model.explained_variance_ratio();
    for (idx, ratio) in ratios.into_iter().enumerate() {
        writeln!(writer, "explained_variance_ratio_PC{}\t{}", idx + 1, ratio)?;
    }

    writer.flush()?;
    Ok(summary_path)
}

fn prepare_output_path(path: &Path) -> Result<(), io::Error> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum DatasetBlockSource {
    Plink(PlinkVariantBlockSource),
    Variants(VcfLikeVariantBlockSource),
}

impl VariantBlockSource for DatasetBlockSource {
    type Error = GenotypeIoError;

    fn n_samples(&self) -> usize {
        match self {
            Self::Plink(source) => source.n_samples(),
            Self::Variants(source) => source.n_samples(),
        }
    }

    fn n_variants(&self) -> usize {
        match self {
            Self::Plink(source) => source.n_variants(),
            Self::Variants(source) => source.n_variants(),
        }
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        match self {
            Self::Plink(source) => source.reset().map_err(GenotypeIoError::from),
            Self::Variants(source) => source.reset().map_err(GenotypeIoError::from),
        }
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        match self {
            Self::Plink(source) => source
                .next_block_into(max_variants, storage)
                .map_err(GenotypeIoError::from),
            Self::Variants(source) => source
                .next_block_into(max_variants, storage)
                .map_err(GenotypeIoError::from),
        }
    }

    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        match self {
            Self::Plink(_) => None,
            Self::Variants(source) => source.progress_bytes(),
        }
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        match self {
            Self::Plink(source) => source.progress_variants(),
            Self::Variants(source) => source.progress_variants(),
        }
    }
}

impl DatasetBlockSource {
    pub fn take_selection_outcome(&mut self) -> Option<SelectionOutcome> {
        match self {
            Self::Variants(source) => source.take_selection_outcome(),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct PlinkDataset {
    bed: BedSource,
    bed_path: PathBuf,
    bim_path: PathBuf,
    fam_path: PathBuf,
    samples: Vec<SampleRecord>,
    n_variants: usize,
    bytes_per_variant: usize,
}

impl PlinkDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PlinkIoError> {
        let bed_path = normalize_path(path.as_ref(), "bed");
        let bim_path = bed_path.with_extension("bim");
        let fam_path = bed_path.with_extension("fam");

        let bed = open_bed_source(&bed_path)?;
        let mut header = [0u8; PLINK_HEADER_LEN as usize];
        bed.read_at(0, &mut header)?;
        validate_bed_header(&header)?;

        let samples = read_fam_records(&fam_path)?;
        if samples.is_empty() {
            return Err(PlinkIoError::MalformedRecord {
                path: fam_path.display().to_string(),
                line: 0,
                message: "no samples found in .fam".to_string(),
            });
        }
        let n_samples = samples.len();
        let bytes_per_variant = ((n_samples + 3) / 4).max(1);

        let n_variants = count_bim_records(&bim_path)?;
        if n_variants == 0 {
            return Err(PlinkIoError::MalformedRecord {
                path: bim_path.display().to_string(),
                line: 0,
                message: "no variants found in .bim".to_string(),
            });
        }

        let expected_payload = (bytes_per_variant as u64)
            .checked_mul(n_variants as u64)
            .ok_or_else(|| PlinkIoError::TruncatedBed {
                expected: u64::MAX,
                actual: bed.len(),
            })?;
        let actual_payload = bed.len().saturating_sub(PLINK_HEADER_LEN);
        if actual_payload != expected_payload {
            return Err(PlinkIoError::TruncatedBed {
                expected: expected_payload,
                actual: actual_payload,
            });
        }

        Ok(Self {
            bed,
            bed_path,
            bim_path,
            fam_path,
            samples,
            n_variants,
            bytes_per_variant,
        })
    }

    pub fn samples(&self) -> &[SampleRecord] {
        &self.samples
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn n_variants(&self) -> usize {
        self.n_variants
    }

    pub fn bed_path(&self) -> &Path {
        &self.bed_path
    }

    pub fn bim_path(&self) -> &Path {
        &self.bim_path
    }

    pub fn fam_path(&self) -> &Path {
        &self.fam_path
    }

    pub fn variant_records(&self) -> Result<PlinkVariantRecordIter, PlinkIoError> {
        let source = open_text_source(&self.bim_path)?;
        Ok(PlinkVariantRecordIter::new(self.bim_path.clone(), source))
    }

    pub fn variant_keys_all(&self) -> Result<Vec<VariantKey>, PlinkIoError> {
        let mut iter = self.variant_records()?;
        let mut keys = Vec::with_capacity(self.n_variants);
        while let Some(result) = iter.next() {
            let record = result?;
            let position =
                record
                    .position
                    .parse::<u64>()
                    .map_err(|err| PlinkIoError::MalformedRecord {
                        path: iter.path.display().to_string(),
                        line: iter.line,
                        message: format!(
                            "invalid position '{}' for variant {}: {err}",
                            record.position, record.identifier
                        ),
                    })?;
            let key = VariantKey::new(&record.chromosome, position);
            keys.push(key);
        }
        Ok(keys)
    }

    pub fn into_block_source(self) -> PlinkVariantBlockSource {
        PlinkVariantBlockSource::new(
            self.bed,
            self.bytes_per_variant,
            self.samples.len(),
            self.n_variants,
            None,
        )
    }

    pub fn block_source(&self) -> PlinkVariantBlockSource {
        self.block_source_with_selection(None)
    }

    pub fn block_source_with_selection(
        &self,
        selection: Option<Vec<usize>>,
    ) -> PlinkVariantBlockSource {
        PlinkVariantBlockSource::new(
            self.bed.clone(),
            self.bytes_per_variant,
            self.samples.len(),
            self.n_variants,
            selection,
        )
    }

    pub fn select_variants(
        &self,
        filter: &VariantFilter,
    ) -> Result<VariantSelection, PlinkIoError> {
        use std::collections::HashSet;

        let mut iter = self.variant_records()?;
        let mut indices = Vec::new();
        let mut keys = Vec::new();
        let mut matched = HashSet::new();
        let mut index = 0usize;

        while let Some(result) = iter.next() {
            let record = result?;
            let position =
                record
                    .position
                    .parse::<u64>()
                    .map_err(|err| PlinkIoError::MalformedRecord {
                        path: iter.path.display().to_string(),
                        line: iter.line,
                        message: format!(
                            "invalid position '{}' for variant {}: {err}",
                            record.position, record.identifier
                        ),
                    })?;

            let key = VariantKey::new(&record.chromosome, position);
            if filter.contains(&key) && matched.insert(key.clone()) {
                indices.push(index);
                keys.push(key);
            }
            index += 1;
        }

        let missing = filter.missing_keys(&matched);
        Ok(VariantSelection {
            indices,
            keys,
            missing,
            requested_unique: filter.requested_unique(),
        })
    }

    pub fn output_path(&self, filename: &str) -> PathBuf {
        if is_remote_path(&self.bed_path) {
            let stem = self
                .bed_path
                .file_stem()
                .map(|s| s.to_os_string())
                .unwrap_or_else(|| OsString::from("dataset"));
            let mut local = PathBuf::from(stem);
            local.set_extension(filename);
            local
        } else {
            let mut local = self.bed_path.clone();
            local.set_extension(filename);
            local
        }
    }
}

pub struct PlinkVariantRecordIter {
    path: PathBuf,
    reader: Box<dyn TextSource>,
    line: usize,
}

impl PlinkVariantRecordIter {
    fn new(path: PathBuf, reader: Box<dyn TextSource>) -> Self {
        Self {
            path,
            reader,
            line: 0,
        }
    }
}

impl fmt::Debug for PlinkVariantRecordIter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlinkVariantRecordIter")
            .field("path", &self.path)
            .field("line", &self.line)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct PlinkVariantRecord {
    pub chromosome: String,
    pub identifier: String,
    pub genetic_distance: String,
    pub position: String,
    pub allele1: String,
    pub allele2: String,
}

impl Iterator for PlinkVariantRecordIter {
    type Item = Result<PlinkVariantRecord, PlinkIoError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.reader.next_line() {
                Ok(Some(line)) => {
                    self.line += 1;
                    if line.iter().all(u8::is_ascii_whitespace) {
                        continue;
                    }
                    let path = self.path.display().to_string();
                    let text = match str::from_utf8(line) {
                        Ok(s) => s,
                        Err(err) => return Some(Err(PlinkIoError::Utf8 { path, source: err })),
                    };
                    let mut fields = text.split_whitespace();
                    let rec = match (
                        fields.next(),
                        fields.next(),
                        fields.next(),
                        fields.next(),
                        fields.next(),
                        fields.next(),
                    ) {
                        (Some(chr), Some(id), Some(cm), Some(pos), Some(a1), Some(a2)) => {
                            PlinkVariantRecord {
                                chromosome: chr.to_string(),
                                identifier: id.to_string(),
                                genetic_distance: cm.to_string(),
                                position: pos.to_string(),
                                allele1: a1.to_string(),
                                allele2: a2.to_string(),
                            }
                        }
                        _ => {
                            return Some(Err(PlinkIoError::MalformedRecord {
                                path,
                                line: self.line,
                                message: "expected 6 whitespace-delimited fields".to_string(),
                            }));
                        }
                    };
                    return Some(Ok(rec));
                }
                Ok(None) => return None,
                Err(err) => return Some(Err(err.into())),
            }
        }
    }
}

#[derive(Debug)]
pub struct PlinkVariantBlockSource {
    bed: BedSource,
    bytes_per_variant: usize,
    n_samples: usize,
    total_variants: usize,
    selection: Option<Vec<usize>>,
    cursor: usize,
    buffer: Vec<u8>,
}

impl PlinkVariantBlockSource {
    fn new(
        bed: BedSource,
        bytes_per_variant: usize,
        n_samples: usize,
        n_variants: usize,
        selection: Option<Vec<usize>>,
    ) -> Self {
        Self {
            bed,
            bytes_per_variant,
            n_samples,
            total_variants: n_variants,
            selection,
            cursor: 0,
            buffer: Vec::new(),
        }
    }
}

impl VariantBlockSource for PlinkVariantBlockSource {
    type Error = PlinkIoError;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.selection
            .as_ref()
            .map(|indices| indices.len())
            .unwrap_or(self.total_variants)
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.cursor = 0;
        Ok(())
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        if let Some(selection) = &self.selection {
            let remaining = selection.len().saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(0);
            }
            let ncols = remaining.min(max_variants);
            let slice = &selection[self.cursor..self.cursor + ncols];
            let table = decode_table();
            let nrows = self.n_samples;
            let mut emitted = 0usize;

            while emitted < ncols {
                let current_index = slice[emitted];
                if current_index >= self.total_variants {
                    return Err(PlinkIoError::InvalidHeader(format!(
                        "variant index {current_index} exceeds dataset bounds ({})",
                        self.total_variants
                    )));
                }

                let mut run = 1usize;
                while emitted + run < ncols && slice[emitted + run] == current_index + run {
                    run += 1;
                }

                let block_bytes = (self.bytes_per_variant as u64)
                    .checked_mul(run as u64)
                    .ok_or_else(|| PlinkIoError::TruncatedBed {
                        expected: u64::MAX,
                        actual: self.bed.len(),
                    })?;

                let offset = PLINK_HEADER_LEN
                    .checked_add((current_index as u64) * (self.bytes_per_variant as u64))
                    .ok_or_else(|| PlinkIoError::TruncatedBed {
                        expected: u64::MAX,
                        actual: self.bed.len(),
                    })?;

                let end =
                    offset
                        .checked_add(block_bytes)
                        .ok_or_else(|| PlinkIoError::TruncatedBed {
                            expected: u64::MAX,
                            actual: self.bed.len(),
                        })?;
                if end > self.bed.len() {
                    return Err(PlinkIoError::TruncatedBed {
                        expected: block_bytes,
                        actual: self.bed.len().saturating_sub(offset),
                    });
                }

                let needed = block_bytes as usize;
                self.buffer.resize(needed, 0);
                self.bed.read_at(offset, &mut self.buffer[..])?;

                for local in 0..run {
                    let bytes_start = local * self.bytes_per_variant;
                    let bytes_end = bytes_start + self.bytes_per_variant;
                    let bytes = &self.buffer[bytes_start..bytes_end];
                    let dest_offset = (emitted + local) * nrows;
                    let dest = &mut storage[dest_offset..dest_offset + nrows];
                    decode_plink_variant(bytes, dest, nrows, table);
                }

                emitted += run;
            }

            self.cursor += ncols;
            Ok(ncols)
        } else {
            let remaining = self.total_variants.saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(0);
            }

            let ncols = remaining.min(max_variants);
            let block_bytes = (self.bytes_per_variant as u64)
                .checked_mul(ncols as u64)
                .ok_or_else(|| PlinkIoError::TruncatedBed {
                    expected: u64::MAX,
                    actual: self.bed.len(),
                })?;

            let offset = PLINK_HEADER_LEN
                .checked_add((self.cursor as u64) * (self.bytes_per_variant as u64))
                .ok_or_else(|| PlinkIoError::TruncatedBed {
                    expected: u64::MAX,
                    actual: self.bed.len(),
                })?;

            let end =
                offset
                    .checked_add(block_bytes)
                    .ok_or_else(|| PlinkIoError::TruncatedBed {
                        expected: u64::MAX,
                        actual: self.bed.len(),
                    })?;
            if end > self.bed.len() {
                return Err(PlinkIoError::TruncatedBed {
                    expected: block_bytes,
                    actual: self.bed.len().saturating_sub(offset),
                });
            }

            let needed = block_bytes as usize;
            self.buffer.resize(needed, 0);

            self.bed.read_at(offset, &mut self.buffer[..])?;

            let table = decode_table();
            let nrows = self.n_samples;
            for variant_idx in 0..ncols {
                let start = variant_idx * self.bytes_per_variant;
                let end = start + self.bytes_per_variant;
                let bytes = &self.buffer[start..end];
                let dest_offset = variant_idx * nrows;
                let dest = &mut storage[dest_offset..dest_offset + nrows];
                decode_plink_variant(bytes, dest, nrows, table);
            }

            self.cursor += ncols;
            Ok(ncols)
        }
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        let total = self
            .selection
            .as_ref()
            .map(|indices| indices.len())
            .unwrap_or(self.total_variants);
        Some((self.cursor.min(total), Some(total)))
    }
}

#[derive(Debug)]
pub struct VcfLikeDataset {
    input_path: PathBuf,
    parts: Vec<PathBuf>,
    sample_names: Arc<Vec<String>>,
    samples: Vec<SampleRecord>,
    variant_count: Arc<VariantCountTracker>,
    output_hint: OutputHint,
}

#[derive(Debug)]
struct VariantCountTracker {
    known: AtomicBool,
    value: AtomicUsize,
}

impl VariantCountTracker {
    fn new(initial: Option<usize>) -> Self {
        Self {
            known: AtomicBool::new(initial.is_some()),
            value: AtomicUsize::new(initial.unwrap_or(0)),
        }
    }

    fn get(&self) -> Option<usize> {
        if self.known.load(Ordering::Acquire) {
            Some(self.value.load(Ordering::Relaxed))
        } else {
            None
        }
    }

    fn set(&self, value: usize) {
        self.value.store(value, Ordering::Relaxed);
        self.known.store(true, Ordering::Release);
    }
}

#[derive(Clone, Debug)]
enum OutputHint {
    LocalFile(PathBuf),
    LocalDirectory(PathBuf),
    RemoteStem(OsString),
}

impl VcfLikeDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VariantIoError> {
        let path = path.as_ref();
        let input_path = path.to_path_buf();
        let output_hint = compute_output_hint(path);

        let parts = list_variant_paths(path).map_err(VariantIoError::from)?;

        let mut expected_sample_names: Option<Vec<String>> = None;
        let mut samples: Option<Vec<SampleRecord>> = None;

        for part in &parts {
            let (mut reader, compression, format, _) = create_variant_reader_for_file(part)
                .map_err(|err| {
                    print_variant_diagnostics(
                        part,
                        None,
                        None,
                        "initializing variant reader",
                        &err,
                    );
                    err
                })?;
            let header = reader.read_header().map_err(|err| {
                print_variant_diagnostics(
                    part,
                    Some(compression),
                    Some(format),
                    "reading variant header",
                    &err,
                );
                VariantIoError::Io(err)
            })?;

            if expected_sample_names.is_none() {
                let sample_names: Vec<String> = header.sample_names().iter().cloned().collect();
                if sample_names.is_empty() {
                    return Err(VariantIoError::MissingSamples);
                }
                let sample_records = sample_names
                    .iter()
                    .map(|name| SampleRecord {
                        family_id: name.clone(),
                        individual_id: name.clone(),
                        paternal_id: "0".to_string(),
                        maternal_id: "0".to_string(),
                        sex: "0".to_string(),
                        phenotype: "-9".to_string(),
                    })
                    .collect::<Vec<_>>();
                expected_sample_names = Some(sample_names);
                samples = Some(sample_records);
            } else {
                let expected = expected_sample_names
                    .as_ref()
                    .expect("sample names initialized");
                let observed = header.sample_names();
                if observed.len() != expected.len()
                    || !observed.iter().zip(expected.iter()).all(|(a, b)| a == b)
                {
                    return Err(VariantIoError::HeaderMismatch);
                }
            }
        }

        let expected_sample_names = expected_sample_names.expect("sample names captured");
        let samples = samples.expect("sample records captured");

        Ok(Self {
            input_path,
            parts,
            sample_names: Arc::new(expected_sample_names),
            samples,
            variant_count: Arc::new(VariantCountTracker::new(None)),
            output_hint,
        })
    }

    pub fn samples(&self) -> &[SampleRecord] {
        &self.samples
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn n_variants(&self) -> usize {
        self.variant_count.get().unwrap_or(0)
    }

    pub fn variant_count_hint(&self) -> Option<usize> {
        self.variant_count.get()
    }

    pub fn input_path(&self) -> &Path {
        &self.input_path
    }

    pub fn block_source(&self) -> Result<VcfLikeVariantBlockSource, VariantIoError> {
        self.block_source_with_plan(SelectionPlan::All)
    }

    pub fn block_source_with_plan(
        &self,
        plan: SelectionPlan,
    ) -> Result<VcfLikeVariantBlockSource, VariantIoError> {
        VcfLikeVariantBlockSource::new(
            self.parts.clone(),
            Arc::clone(&self.sample_names),
            Arc::clone(&self.variant_count),
            plan,
        )
    }

    pub fn variant_keys_all(&self) -> Result<Vec<VariantKey>, VariantIoError> {
        let mut keys = Vec::new();
        let mut record = RecordBuf::default();

        for part in &self.parts {
            let (mut reader, compression, format, _) = create_variant_reader_for_file(part)
                .map_err(|err| {
                    print_variant_diagnostics(
                        part,
                        None,
                        None,
                        "initializing variant reader for LD window",
                        &err,
                    );
                    err
                })?;
            let header = reader.read_header().map_err(|err| {
                print_variant_diagnostics(
                    part,
                    Some(compression),
                    Some(format),
                    "reading variant header",
                    &err,
                );
                VariantIoError::Io(err)
            })?;

            loop {
                let bytes = reader
                    .read_record_buf(&header, &mut record)
                    .map_err(|err| {
                        print_variant_diagnostics(
                            part,
                            Some(compression),
                            Some(format),
                            "scanning variant records",
                            &err,
                        );
                        VariantIoError::Io(err)
                    })?;
                if bytes == 0 {
                    break;
                }

                let chrom = record.reference_sequence_name().to_string();
                let Some(position) = record.variant_start() else {
                    return Err(VariantIoError::Decode(
                        "variant position missing from record".to_string(),
                    ));
                };
                let pos = position.get() as u64;
                keys.push(VariantKey::new(&chrom, pos));
            }
        }

        Ok(keys)
    }

    pub fn output_path(&self, filename: &str) -> PathBuf {
        match &self.output_hint {
            OutputHint::LocalFile(path) => {
                let mut local = path.clone();
                local.set_extension(filename);
                local
            }
            OutputHint::LocalDirectory(dir) => dir.join(filename),
            OutputHint::RemoteStem(stem) => {
                let mut local = PathBuf::from(stem);
                local.set_extension(filename);
                local
            }
        }
    }

    pub fn select_variants(
        &self,
        filter: &VariantFilter,
    ) -> Result<VariantSelection, VariantIoError> {
        use std::collections::HashSet;

        let mut indices = Vec::new();
        let mut keys = Vec::new();
        let mut matched = HashSet::new();
        let mut record_idx = 0usize;
        let mut record = RecordBuf::default();

        for part in &self.parts {
            let (mut reader, compression, format, _) = create_variant_reader_for_file(part)
                .map_err(|err| {
                    print_variant_diagnostics(
                        part,
                        None,
                        None,
                        "initializing variant reader for variant list",
                        &err,
                    );
                    err
                })?;
            let header = reader.read_header().map_err(|err| {
                print_variant_diagnostics(
                    part,
                    Some(compression),
                    Some(format),
                    "reading variant header",
                    &err,
                );
                VariantIoError::Io(err)
            })?;

            loop {
                let bytes = reader
                    .read_record_buf(&header, &mut record)
                    .map_err(|err| {
                        print_variant_diagnostics(
                            part,
                            Some(compression),
                            Some(format),
                            "scanning variant records for variant list",
                            &err,
                        );
                        VariantIoError::Io(err)
                    })?;
                if bytes == 0 {
                    break;
                }

                let chrom = record.reference_sequence_name().to_string();
                if let Some(position) = record.variant_start() {
                    let pos = position.get() as u64;
                    let key = VariantKey::new(&chrom, pos);
                    if filter.contains(&key) && matched.insert(key.clone()) {
                        indices.push(record_idx);
                        keys.push(key);
                    }
                }
                record_idx += 1;
            }
        }

        let missing = filter.missing_keys(&matched);
        Ok(VariantSelection {
            indices,
            keys,
            missing,
            requested_unique: filter.requested_unique(),
        })
    }
}

pub struct VcfLikeVariantBlockSource {
    parts: Vec<PathBuf>,
    part_idx: usize,
    reader: Option<VariantStreamReader>,
    compression: Option<VariantCompression>,
    format: Option<VariantFormat>,
    header: Option<Arc<vcf::Header>>,
    vcf_record: VcfRecord,
    bcf_record: BcfRecord,
    prefer_ds: bool,
    sample_names: Arc<Vec<String>>,
    n_samples: usize,
    variant_count: Arc<VariantCountTracker>,
    total_variants_hint: Option<usize>,
    filtered_variants_hint: usize,
    selection_plan: SelectionPlan,
    matched_keys: Vec<VariantKey>,
    matched_seen: HashSet<VariantKey>,
    missing_keys: Option<Vec<VariantKey>>,
    requested_unique: usize,
    streamed_variants: usize,
    stream_exhausted: bool,
    selection_finalized: bool,
    metrics_per_part: Vec<Arc<ReadMetrics>>,
    emitted: usize,
    processed: usize,
    spool_entries: Vec<Option<Arc<SpoolEntry>>>,
    spool_root: Option<PathBuf>,
}

struct SpoolEntry {
    final_path: PathBuf,
    temp_path: PathBuf,
    compression: VariantCompression,
    format: VariantFormat,
    metrics: Arc<ReadMetrics>,
    completion: Arc<AtomicBool>,
    state: Arc<PrefetchState>,
    handle: Mutex<Option<thread::JoinHandle<io::Result<()>>>>,
}

impl SpoolEntry {
    fn spawn(
        idx: usize,
        remote_path: &Path,
        spool_root: &Path,
    ) -> Result<Arc<Self>, VariantIoError> {
        let source = open_variant_source(remote_path)?;
        let compression = source.compression();
        let format = source.format();
        let metrics = source.metrics();
        let (final_path, temp_path) = Self::paths_for(idx, remote_path, spool_root);
        let state = Arc::new(PrefetchState::new(PREFETCH_RING_CAPACITY));
        let completion = Arc::new(AtomicBool::new(false));
        let thread_state = Arc::clone(&state);
        let thread_completion = Arc::clone(&completion);
        let thread_final = final_path.clone();
        let thread_temp = temp_path.clone();
        let handle = thread::Builder::new()
            .name("variant-spool".to_string())
            .spawn(move || {
                run_spool_thread(
                    source,
                    thread_temp,
                    thread_final,
                    thread_state,
                    thread_completion,
                )
            })?;

        Ok(Arc::new(Self {
            final_path,
            temp_path,
            compression,
            format,
            metrics,
            completion,
            state,
            handle: Mutex::new(Some(handle)),
        }))
    }

    fn paths_for(idx: usize, remote_path: &Path, spool_root: &Path) -> (PathBuf, PathBuf) {
        let name = remote_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "variants".to_string());
        let final_name = format!("part{}_{}", idx, name);
        let final_path = spool_root.join(final_name);
        let temp_extension = final_path
            .extension()
            .map(|ext| {
                let mut ext = ext.to_os_string();
                ext.push(".spooling");
                ext
            })
            .unwrap_or_else(|| OsString::from("spooling"));
        let temp_path = final_path.with_extension(temp_extension);
        (final_path, temp_path)
    }

    fn is_complete(&self) -> bool {
        self.completion.load(Ordering::Acquire)
    }

    fn error(&self) -> Option<io::Error> {
        self.state.error()
    }

    fn attach_reader(
        &self,
    ) -> io::Result<(
        VariantStreamReader,
        VariantCompression,
        VariantFormat,
        Arc<ReadMetrics>,
    )> {
        if self.is_complete() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "spool already complete",
            ));
        }
        if let Some(err) = self.state.error() {
            return Err(err);
        }
        let tap = self.state.attach()?;
        let reader = match self.format {
            VariantFormat::Bcf => {
                let reader: Box<dyn Read + Send> = match self.compression {
                    VariantCompression::Plain => Box::new(tap),
                    VariantCompression::Bgzf => Box::new(BgzfReader::new(tap)),
                };
                VariantStreamReader::Bcf(BcfReader::from(reader))
            }
            VariantFormat::Vcf => {
                let reader: Box<dyn BufRead + Send> = match self.compression {
                    VariantCompression::Plain => Box::new(BufReader::new(tap)),
                    VariantCompression::Bgzf => Box::new(BgzfReader::new(BufReader::new(tap))),
                };
                VariantStreamReader::Vcf(VcfReader::new(reader))
            }
        };
        Ok((
            reader,
            self.compression,
            self.format,
            Arc::clone(&self.metrics),
        ))
    }

    fn open_local_reader(
        &self,
    ) -> Result<
        (
            VariantStreamReader,
            VariantCompression,
            VariantFormat,
            Arc<ReadMetrics>,
        ),
        VariantIoError,
    > {
        let file = File::open(&self.final_path)?;
        let reader = match self.format {
            VariantFormat::Bcf => {
                let reader: Box<dyn Read + Send> = match self.compression {
                    VariantCompression::Plain => Box::new(file),
                    VariantCompression::Bgzf => Box::new(BgzfReader::new(file)),
                };
                VariantStreamReader::Bcf(BcfReader::from(reader))
            }
            VariantFormat::Vcf => {
                let reader: Box<dyn BufRead + Send> = match self.compression {
                    VariantCompression::Plain => Box::new(BufReader::new(file)),
                    VariantCompression::Bgzf => Box::new(BgzfReader::new(BufReader::new(file))),
                };
                VariantStreamReader::Vcf(VcfReader::new(reader))
            }
        };
        Ok((
            reader,
            self.compression,
            self.format,
            Arc::clone(&self.metrics),
        ))
    }

    fn shutdown(&self) {
        self.state.cancel();
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
        if self.is_complete() {
            return;
        }
        if self.error().is_some() {
            let _ = fs::remove_file(&self.final_path);
        }
        let _ = fs::remove_file(&self.temp_path);
    }
}

impl fmt::Debug for SpoolEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpoolEntry")
            .field("final_path", &self.final_path)
            .field("complete", &self.is_complete())
            .finish()
    }
}

const PREFETCH_RING_CAPACITY: usize = 16 * 1024 * 1024;
const PREFETCH_CHUNK_SIZE: usize = 256 * 1024;

struct PrefetchState {
    inner: Mutex<PrefetchInner>,
    data_ready: Condvar,
    space_available: Condvar,
}

struct PrefetchInner {
    buffer: Vec<u8>,
    head: usize,
    len: usize,
    attached: usize,
    done: bool,
    shutting_down: bool,
    error: Option<(io::ErrorKind, String)>,
}

impl PrefetchState {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(PrefetchInner {
                buffer: vec![0u8; capacity],
                head: 0,
                len: 0,
                attached: 0,
                done: false,
                shutting_down: false,
                error: None,
            }),
            data_ready: Condvar::new(),
            space_available: Condvar::new(),
        }
    }

    fn attach(self: &Arc<Self>) -> io::Result<PrefetchTap> {
        let mut guard = self.inner.lock().unwrap();
        if let Some((kind, message)) = guard.error.clone() {
            return Err(io::Error::new(kind, message));
        }
        if guard.done {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "spool already complete",
            ));
        }
        if guard.attached > 0 {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "spool already has an attached reader",
            ));
        }
        guard.attached = 1;
        guard.head = 0;
        guard.len = 0;
        drop(guard);
        self.space_available.notify_all();
        Ok(PrefetchTap {
            state: Arc::clone(self),
            active: true,
        })
    }

    fn wait_for_first_attachment(&self) -> io::Result<()> {
        let mut guard = self.inner.lock().unwrap();
        while guard.attached == 0 && !guard.shutting_down {
            guard = self.space_available.wait(guard).unwrap();
        }
        if guard.shutting_down {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "spool shutting down",
            ));
        }
        Ok(())
    }

    fn detach(&self) {
        let mut guard = self.inner.lock().unwrap();
        if guard.attached > 0 {
            guard.attached -= 1;
        }
        guard.head = 0;
        guard.len = 0;
        self.space_available.notify_all();
        self.data_ready.notify_all();
    }

    fn push_chunk(&self, data: &[u8]) -> io::Result<()> {
        let mut offset = 0;
        while offset < data.len() {
            let mut guard = self.inner.lock().unwrap();
            while guard.attached > 0 && guard.len == guard.buffer.len() && !guard.shutting_down {
                guard = self.space_available.wait(guard).unwrap();
            }
            if guard.shutting_down {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "spool shutting down",
                ));
            }
            if guard.attached == 0 {
                guard.head = 0;
                guard.len = 0;
                break;
            }
            let capacity = guard.buffer.len();
            let tail = (guard.head + guard.len) % capacity;
            let available = capacity - guard.len;
            let chunk = available.min(data.len() - offset);
            if chunk == 0 {
                continue;
            }
            let first = (capacity - tail).min(chunk);
            guard.buffer[tail..tail + first].copy_from_slice(&data[offset..offset + first]);
            guard.len += first;
            offset += first;

            if first < chunk {
                let second = chunk - first;
                guard.buffer[..second].copy_from_slice(&data[offset..offset + second]);
                guard.len += second;
                offset += second;
            }

            drop(guard);
            self.data_ready.notify_all();
        }
        Ok(())
    }

    fn mark_done(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.done = true;
        self.data_ready.notify_all();
        self.space_available.notify_all();
    }

    fn record_error(&self, kind: io::ErrorKind, message: String) {
        let mut guard = self.inner.lock().unwrap();
        guard.error = Some((kind, message));
        guard.done = true;
        self.data_ready.notify_all();
        self.space_available.notify_all();
    }

    fn cancel(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.shutting_down = true;
        self.data_ready.notify_all();
        self.space_available.notify_all();
    }

    fn error(&self) -> Option<io::Error> {
        let guard = self.inner.lock().unwrap();
        guard
            .error
            .as_ref()
            .map(|(kind, message)| io::Error::new(*kind, message.clone()))
    }
}

struct PrefetchTap {
    state: Arc<PrefetchState>,
    active: bool,
}

impl Read for PrefetchTap {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        loop {
            let mut guard = self.state.inner.lock().unwrap();
            while guard.len == 0 && guard.error.is_none() && !guard.done {
                guard = self.state.data_ready.wait(guard).unwrap();
            }

            if let Some((kind, message)) = guard.error.clone() {
                return Err(io::Error::new(kind, message));
            }

            if guard.len == 0 {
                if guard.done {
                    return Ok(0);
                }
                continue;
            }

            let capacity = guard.buffer.len();
            let to_read = buf.len().min(guard.len);
            let first = (capacity - guard.head).min(to_read);
            buf[..first].copy_from_slice(&guard.buffer[guard.head..guard.head + first]);
            guard.head = (guard.head + first) % capacity;
            guard.len -= first;

            if first < to_read {
                let second = to_read - first;
                buf[first..first + second]
                    .copy_from_slice(&guard.buffer[guard.head..guard.head + second]);
                guard.head = (guard.head + second) % capacity;
                guard.len -= second;
            }

            drop(guard);
            self.state.space_available.notify_all();
            return Ok(to_read);
        }
    }
}

impl Drop for PrefetchTap {
    fn drop(&mut self) {
        if self.active {
            self.state.detach();
            self.active = false;
        }
    }
}

fn run_spool_thread(
    mut source: VariantSource,
    temp_path: PathBuf,
    final_path: PathBuf,
    state: Arc<PrefetchState>,
    completion: Arc<AtomicBool>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    let mut buffer = vec![0u8; PREFETCH_CHUNK_SIZE];

    if let Err(err) = state.wait_for_first_attachment() {
        state.record_error(err.kind(), err.to_string());
        let _ = fs::remove_file(&temp_path);
        return Err(err);
    }

    let result: io::Result<()> = (|| {
        loop {
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            writer.write_all(&buffer[..read])?;
            state.push_chunk(&buffer[..read])?;
        }
        writer.flush()?;
        fs::rename(&temp_path, &final_path)?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            state.mark_done();
            completion.store(true, Ordering::Release);
            Ok(())
        }
        Err(err) => {
            state.record_error(err.kind(), err.to_string());
            let _ = fs::remove_file(&temp_path);
            Err(err)
        }
    }
}

impl std::fmt::Debug for VcfLikeVariantBlockSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VcfLikeVariantBlockSource")
            .field("current_part", &self.parts.get(self.part_idx))
            .field("n_samples", &self.n_samples)
            .field("total_variants_hint", &self.total_variants_hint)
            .field("filtered_variants_hint", &self.filtered_variants_hint)
            .field("emitted", &self.emitted)
            .field("processed", &self.processed)
            .field("streamed_variants", &self.streamed_variants)
            .field("stream_exhausted", &self.stream_exhausted)
            .field("selection_finalized", &self.selection_finalized)
            .field("compression", &self.compression)
            .field("format", &self.format)
            .field("spool_root", &self.spool_root)
            .finish()
    }
}

impl VcfLikeVariantBlockSource {
    fn ensure_spool_root(&mut self) -> Result<PathBuf, io::Error> {
        if let Some(root) = &self.spool_root {
            return Ok(root.clone());
        }

        let mut root = env::temp_dir();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        root.push(format!("gnomon-spool-{}-{}", std::process::id(), unique));
        fs::create_dir_all(&root)?;
        self.spool_root = Some(root.clone());
        Ok(root)
    }

    fn spool_part(&mut self, idx: usize) -> Result<Arc<SpoolEntry>, VariantIoError> {
        let path = self.parts[idx].clone();
        let root = self.ensure_spool_root().map_err(VariantIoError::Io)?;
        SpoolEntry::spawn(idx, &path, &root)
    }

    fn reader_for_part(
        &mut self,
        idx: usize,
    ) -> Result<
        (
            VariantStreamReader,
            VariantCompression,
            VariantFormat,
            Arc<ReadMetrics>,
        ),
        VariantIoError,
    > {
        let path = self.parts[idx].clone();
        if is_http_path(&path) {
            return create_variant_reader_for_file(&path);
        }

        if is_remote_path(&path) {
            if let Ok(result) = create_variant_reader_for_file(&path) {
                return Ok(result);
            }

            if self.spool_entries.len() <= idx {
                self.spool_entries.resize(idx + 1, None);
            }

            if let Some(entry) = self.spool_entries[idx].as_ref() {
                if entry.is_complete() {
                    return entry.open_local_reader();
                }

                match entry.attach_reader() {
                    Ok(result) => return Ok(result),
                    Err(err) => {
                        if entry.is_complete() {
                            return entry.open_local_reader();
                        }
                        if let Some(source_err) = entry.error() {
                            return Err(VariantIoError::Io(source_err));
                        }
                        return Err(VariantIoError::Io(err));
                    }
                }
            }

            let entry = self.spool_part(idx)?;
            let result = entry.attach_reader().map_err(VariantIoError::Io)?;
            self.spool_entries[idx] = Some(entry);
            return Ok(result);
        }

        if self.spool_entries.len() <= idx {
            self.spool_entries.resize(idx + 1, None);
        }

        create_variant_reader_for_file(&path)
    }

    fn new(
        parts: Vec<PathBuf>,
        sample_names: Arc<Vec<String>>,
        variant_count: Arc<VariantCountTracker>,
        selection_plan: SelectionPlan,
    ) -> Result<Self, VariantIoError> {
        let n_samples = sample_names.len();
        let n_variants_hint = variant_count.get();
        let filtered_variants_hint = match &selection_plan {
            SelectionPlan::All => n_variants_hint.unwrap_or(0),
            SelectionPlan::ByIndices(indices) => indices.len(),
            SelectionPlan::ByKeys(filter) => filter.requested_unique(),
        };
        let requested_unique = match &selection_plan {
            SelectionPlan::ByKeys(filter) => filter.requested_unique(),
            SelectionPlan::ByIndices(indices) => indices.len(),
            SelectionPlan::All => n_variants_hint.unwrap_or(0),
        };
        let mut source = Self {
            parts,
            part_idx: 0,
            reader: None,
            compression: None,
            format: None,
            header: None,
            vcf_record: VcfRecord::default(),
            bcf_record: BcfRecord::default(),
            prefer_ds: false,
            sample_names,
            n_samples,
            variant_count,
            total_variants_hint: n_variants_hint,
            filtered_variants_hint,
            selection_plan,
            matched_keys: Vec::new(),
            matched_seen: HashSet::new(),
            missing_keys: None,
            requested_unique,
            streamed_variants: 0,
            stream_exhausted: false,
            selection_finalized: false,
            metrics_per_part: Vec::new(),
            emitted: 0,
            processed: 0,
            spool_entries: Vec::new(),
            spool_root: None,
        };
        if !source.parts.is_empty() {
            source.open_part(0)?;
        }
        Ok(source)
    }

    fn open_part(&mut self, idx: usize) -> Result<(), VariantIoError> {
        self.part_idx = idx;
        self.reader = None;
        self.compression = None;
        self.format = None;
        self.header = None;
        if idx >= self.parts.len() {
            return Ok(());
        }

        let (mut reader, compression, format, metrics) =
            self.reader_for_part(idx).map_err(|err| {
                let path = &self.parts[idx];
                print_variant_diagnostics(path, None, None, "initializing variant reader", &err);
                err
            })?;
        let header = reader.read_header().map_err(|err| {
            print_variant_diagnostics(
                &self.parts[idx],
                Some(compression),
                Some(format),
                "reading variant header",
                &err,
            );
            VariantIoError::Io(err)
        })?;

        let formats = header.formats();
        self.prefer_ds = formats.contains_key("DS");

        let observed = header.sample_names();
        if observed.len() != self.sample_names.len()
            || !observed
                .iter()
                .zip(self.sample_names.iter())
                .all(|(a, b)| a == b)
        {
            return Err(VariantIoError::HeaderMismatch);
        }

        if self.metrics_per_part.len() <= idx {
            self.metrics_per_part.push(metrics);
        } else {
            self.metrics_per_part[idx] = metrics;
        }

        self.reader = Some(reader);
        self.compression = Some(compression);
        self.format = Some(format);
        self.header = Some(Arc::new(header));
        Ok(())
    }

    fn read_next_variant(&mut self) -> Result<Option<usize>, VariantIoError> {
        loop {
            if self.header.is_none() {
                self.stream_exhausted = true;
                return Ok(None);
            }

            let reader = match self.reader.as_mut() {
                Some(reader) => reader,
                None => {
                    self.stream_exhausted = true;
                    return Ok(None);
                }
            };

            let compression = self.compression;
            let format = self.format;
            let path = self.parts.get(self.part_idx).cloned();
            let bytes = match reader {
                VariantStreamReader::Bcf(bcf_reader) => bcf_reader
                    .read_record(&mut self.bcf_record)
                    .map_err(|err| {
                        if let Some(part_path) = path.as_ref() {
                            print_variant_diagnostics(
                                part_path,
                                compression,
                                format,
                                "reading variant record block",
                                &err,
                            );
                        }
                        VariantIoError::Io(err)
                    })?,
                VariantStreamReader::Vcf(vcf_reader) => vcf_reader
                    .read_record(&mut self.vcf_record)
                    .map_err(|err| {
                        if let Some(part_path) = path.as_ref() {
                            print_variant_diagnostics(
                                part_path,
                                compression,
                                format,
                                "reading variant record block",
                                &err,
                            );
                        }
                        VariantIoError::Io(err)
                    })?,
            };

            if bytes == 0 {
                let next_idx = self.part_idx + 1;
                self.open_part(next_idx)?;
                continue;
            }

            let current_index = self.processed;
            self.processed = self.processed.saturating_add(1);
            self.streamed_variants = self.streamed_variants.saturating_add(1);
            return Ok(Some(current_index));
        }
    }
}

impl VariantBlockSource for VcfLikeVariantBlockSource {
    type Error = VariantIoError;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.filtered_variants_hint
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.emitted = 0;
        self.processed = 0;
        self.streamed_variants = 0;
        self.stream_exhausted = false;
        if matches!(self.selection_plan, SelectionPlan::ByKeys(_)) {
            self.matched_seen.clear();
        }
        for metrics in &self.metrics_per_part {
            metrics.reset();
        }
        if self.parts.is_empty() {
            self.reader = None;
            self.compression = None;
            self.format = None;
            self.header = None;
            return Ok(());
        }
        self.open_part(0)
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        if self.stream_exhausted {
            self.finalize_selection();
            return Ok(0);
        }

        let selection_plan = std::mem::replace(&mut self.selection_plan, SelectionPlan::All);
        let (restored_plan, filled_result) = match selection_plan {
            SelectionPlan::All => (
                SelectionPlan::All,
                self.next_block_all(max_variants, storage),
            ),
            SelectionPlan::ByIndices(indices) => {
                let result = self.next_block_indices(&indices, max_variants, storage);
                (SelectionPlan::ByIndices(indices), result)
            }
            SelectionPlan::ByKeys(filter) => {
                let result = self.next_block_keys(filter.as_ref(), max_variants, storage);
                (SelectionPlan::ByKeys(filter), result)
            }
        };
        self.selection_plan = restored_plan;
        let filled = filled_result?;

        if filled == 0 && self.stream_exhausted {
            self.finalize_selection();
        }

        Ok(filled)
    }

    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        VcfLikeVariantBlockSource::progress_bytes(self)
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        let work_done = if matches!(self.selection_plan, SelectionPlan::ByKeys(_)) {
            self.processed
        } else {
            self.emitted
        };
        Some((work_done, self.total_variants_hint))
    }
}

impl VcfLikeVariantBlockSource {
    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        if matches!(self.selection_plan, SelectionPlan::ByKeys(_)) {
            return None;
        }
        if self.metrics_per_part.is_empty() {
            return None;
        }
        let mut total_read = 0u64;
        let mut aggregated_total = Some(0u64);
        for metrics in &self.metrics_per_part {
            let (read, total) = metrics.snapshot();
            total_read = total_read.saturating_add(read);
            match (aggregated_total, total) {
                (Some(acc), Some(value)) => {
                    aggregated_total = Some(acc.saturating_add(value));
                }
                _ => {
                    aggregated_total = None;
                }
            }
        }
        Some((total_read, aggregated_total))
    }

    fn decode_current_variant(&self, dest: &mut [f64]) -> Result<(), VariantIoError> {
        if dest.len() < self.n_samples {
            return Err(VariantIoError::Decode(
                "destination buffer shorter than number of samples".to_string(),
            ));
        }

        match self.format {
            Some(VariantFormat::Vcf) => {
                decode_vcf_record(&self.vcf_record, self.n_samples, self.prefer_ds, dest)
            }
            Some(VariantFormat::Bcf) => {
                let header = self
                    .header
                    .as_ref()
                    .ok_or_else(|| VariantIoError::Decode("BCF header missing".to_string()))?;
                decode_bcf_record(
                    &self.bcf_record,
                    header.as_ref(),
                    self.n_samples,
                    self.prefer_ds,
                    dest,
                )
            }
            None => Err(VariantIoError::Decode(
                "variant stream format unknown".to_string(),
            )),
        }
    }

    fn current_variant_key(&self) -> Result<Option<VariantKey>, VariantIoError> {
        match self.format {
            Some(VariantFormat::Vcf) => {
                let chrom = self.vcf_record.reference_sequence_name().to_string();
                let Some(start) = self.vcf_record.variant_start() else {
                    return Ok(None);
                };
                let position = start.map_err(|err| {
                    VariantIoError::Decode(format!("failed to read VCF position: {err}"))
                })?;
                Ok(Some(VariantKey::new(&chrom, position.get() as u64)))
            }
            Some(VariantFormat::Bcf) => {
                let header = self
                    .header
                    .as_ref()
                    .ok_or_else(|| VariantIoError::Decode("BCF header missing".to_string()))?;
                let chrom = self
                    .bcf_record
                    .reference_sequence_name(header.string_maps())
                    .map_err(|err| {
                        VariantIoError::Decode(format!(
                            "failed to read BCF reference sequence: {err}"
                        ))
                    })?;
                let Some(start) = self.bcf_record.variant_start() else {
                    return Ok(None);
                };
                let position = start.map_err(|err| {
                    VariantIoError::Decode(format!("failed to read BCF position: {err}"))
                })?;
                Ok(Some(VariantKey::new(chrom, position.get() as u64)))
            }
            None => Ok(None),
        }
    }

    fn next_block_all(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, VariantIoError> {
        let mut filled = 0usize;
        while filled < max_variants {
            let Some(_) = self.read_next_variant()? else {
                break;
            };

            let offset = filled * self.n_samples;
            let dest = &mut storage[offset..offset + self.n_samples];
            self.decode_current_variant(dest)?;
            filled += 1;
        }

        if filled == 0 && self.stream_exhausted {
            return Ok(0);
        }

        self.emitted += filled;
        Ok(filled)
    }

    fn next_block_indices(
        &mut self,
        indices: &[usize],
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, VariantIoError> {
        let target_total = indices.len();
        let mut filled = 0usize;

        while filled < max_variants && self.emitted + filled < target_total {
            let Some(current_index) = self.read_next_variant()? else {
                break;
            };

            let target_index = indices[self.emitted + filled];
            if current_index < target_index {
                continue;
            }
            if current_index > target_index {
                return Err(VariantIoError::Decode(format!(
                    "variant index {target_index} requested by PCA list not found in dataset",
                )));
            }

            let offset = filled * self.n_samples;
            let dest = &mut storage[offset..offset + self.n_samples];
            self.decode_current_variant(dest)?;
            filled += 1;
        }

        if filled == 0 && self.stream_exhausted && self.emitted < target_total {
            return Err(VariantIoError::UnexpectedEof {
                expected: target_total,
                actual: self.emitted,
            });
        }

        self.emitted += filled;
        Ok(filled)
    }

    fn next_block_keys(
        &mut self,
        filter: &VariantFilter,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, VariantIoError> {
        let mut filled = 0usize;
        let target_total = if self.selection_finalized {
            self.filtered_variants_hint
        } else {
            self.requested_unique
        };

        while filled < max_variants && (target_total == 0 || self.emitted + filled < target_total) {
            let Some(_) = self.read_next_variant()? else {
                break;
            };

            if let Some(key) = self.current_variant_key()? {
                if filter.contains(&key) {
                    let is_new_match = self.matched_seen.insert(key.clone());
                    if is_new_match {
                        let offset = filled * self.n_samples;
                        let dest = &mut storage[offset..offset + self.n_samples];
                        self.decode_current_variant(dest)?;
                        if !self.selection_finalized {
                            self.matched_keys.push(key);
                        }
                        filled += 1;
                        if target_total > 0 && self.emitted + filled >= target_total {
                            self.stream_exhausted = true;
                            break;
                        }
                        continue;
                    }
                }
            }
        }

        if filled == 0 && self.stream_exhausted {
            // No additional matches in this pass.
        }

        self.emitted += filled;
        Ok(filled)
    }

    fn finalize_selection(&mut self) {
        if self.selection_finalized {
            return;
        }

        match &self.selection_plan {
            SelectionPlan::All => {
                self.filtered_variants_hint = self.emitted;
                self.total_variants_hint = Some(self.processed);
                self.variant_count.set(self.emitted);
            }
            SelectionPlan::ByIndices(_) => {
                self.total_variants_hint = Some(self.processed);
            }
            SelectionPlan::ByKeys(filter) => {
                let missing = filter.missing_keys(&self.matched_seen);
                self.missing_keys = Some(missing);
                self.filtered_variants_hint = self.matched_keys.len();
                self.total_variants_hint = Some(self.processed);
            }
        }

        self.selection_finalized = true;
    }

    pub fn take_selection_outcome(&mut self) -> Option<SelectionOutcome> {
        if !matches!(self.selection_plan, SelectionPlan::ByKeys(_)) {
            return None;
        }
        if !self.selection_finalized {
            self.finalize_selection();
        }
        let matched_keys = std::mem::take(&mut self.matched_keys);
        let missing_keys = self.missing_keys.take().unwrap_or_default();
        Some(SelectionOutcome {
            matched_keys,
            missing_keys,
            requested_unique: self.requested_unique,
        })
    }
}

impl Drop for VcfLikeVariantBlockSource {
    fn drop(&mut self) {
        for entry in self.spool_entries.drain(..).flatten() {
            entry.shutdown();
        }
        if let Some(root) = &self.spool_root {
            let _ = fs::remove_dir(root);
        }
    }
}

fn guess_is_variant_dataset(path: &Path) -> bool {
    if is_remote_path(path) {
        let lower = path.to_string_lossy().to_ascii_lowercase();
        lower.ends_with(".bcf")
            || lower.ends_with(".vcf")
            || lower.ends_with(".vcf.gz")
            || lower.ends_with(".vcf.bgz")
            || lower.ends_with(".bgz")
            || lower.ends_with("/*")
            || lower.ends_with('/')
    } else if path.is_dir() {
        true
    } else {
        let lower = path.to_string_lossy().to_ascii_lowercase();
        lower.ends_with(".bcf")
            || lower.ends_with(".vcf")
            || lower.ends_with(".vcf.gz")
            || lower.ends_with(".vcf.bgz")
    }
}

fn compute_output_hint(path: &Path) -> OutputHint {
    if is_remote_path(path) {
        let mut trimmed = path.to_string_lossy().into_owned();
        while trimmed.ends_with('*') {
            trimmed.pop();
        }
        while trimmed.ends_with('/') {
            trimmed.pop();
        }
        if trimmed.is_empty() {
            return OutputHint::RemoteStem(OsString::from("dataset"));
        }
        let stem = Path::new(&trimmed)
            .file_stem()
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| OsString::from("dataset"));
        OutputHint::RemoteStem(stem)
    } else if path.is_dir() {
        OutputHint::LocalDirectory(path.to_path_buf())
    } else {
        OutputHint::LocalFile(path.to_path_buf())
    }
}

fn create_variant_reader_for_file(
    path: &Path,
) -> Result<
    (
        VariantStreamReader,
        VariantCompression,
        VariantFormat,
        Arc<ReadMetrics>,
    ),
    VariantIoError,
> {
    let source = open_variant_source(path)?;
    let compression = source.compression();
    let format = source.format();
    let metrics = source.metrics();

    let reader = match format {
        VariantFormat::Bcf => {
            let reader: Box<dyn Read + Send> = match compression {
                VariantCompression::Plain => Box::new(source),
                VariantCompression::Bgzf => Box::new(BgzfReader::new(source)),
            };
            VariantStreamReader::Bcf(BcfReader::from(reader))
        }
        VariantFormat::Vcf => {
            let reader: Box<dyn BufRead + Send> = match compression {
                VariantCompression::Plain => Box::new(BufReader::new(source)),
                VariantCompression::Bgzf => Box::new(BgzfReader::new(BufReader::new(source))),
            };
            VariantStreamReader::Vcf(VcfReader::new(reader))
        }
    };

    Ok((reader, compression, format, metrics))
}

fn print_variant_diagnostics(
    path: &Path,
    compression: Option<VariantCompression>,
    format: Option<VariantFormat>,
    stage: &str,
    err: &(dyn std::error::Error + '_),
) {
    let location = if is_remote_path(path) {
        "remote (GCS/HTTP)"
    } else if path.is_dir() {
        "local directory"
    } else {
        "local file"
    };

    eprintln!("Variant diagnostics:");
    eprintln!("  • Path        : {}", path.display());
    eprintln!("  • Location    : {location}");
    match format {
        Some(kind) => eprintln!("  • Format      : {:?}", kind),
        None => eprintln!("  • Format      : unknown (detected before reader initialization)"),
    }
    match compression {
        Some(mode) => eprintln!("  • Compression : {:?}", mode),
        None => eprintln!("  • Compression : unknown (detected before reader initialization)"),
    }
    eprintln!("  • Stage       : {stage}");
    eprintln!("  • Underlying error: {err}");

    if let Some(VariantCompression::Bgzf) = compression {
        if err
            .to_string()
            .to_lowercase()
            .contains("invalid bgzf header")
        {
            eprintln!(
                "  • Hint        : The BGZF stream appears corrupt or truncated; ensure the source is a complete BGZF-compressed .bcf file."
            );
        }
    }
}

fn decode_vcf_record(
    record: &VcfRecord,
    n_samples: usize,
    prefer_ds: bool,
    dest: &mut [f64],
) -> Result<(), VariantIoError> {
    dest[..n_samples].fill(f64::NAN);

    let samples = record.samples();
    if samples.is_empty() {
        return Ok(());
    }

    let mut ds_index = None;
    let mut gt_index = None;
    for (idx, key) in samples.keys().iter().enumerate() {
        if prefer_ds && ds_index.is_none() && key == "DS" {
            ds_index = Some(idx);
        }
        if gt_index.is_none() && key == key::GENOTYPE {
            gt_index = Some(idx);
        }
    }

    let Some(gt_idx) = gt_index else {
        return Err(VariantIoError::Decode(
            "VCF record is missing the required GT FORMAT field".to_string(),
        ));
    };

    for (sample_idx, sample) in samples.iter().enumerate().take(n_samples) {
        let mut ds_field: Option<&str> = None;
        let mut gt_field: Option<&str> = None;

        for (idx, field) in sample.as_ref().split(':').enumerate() {
            if prefer_ds && ds_index == Some(idx) {
                ds_field = Some(field);
            }
            if idx == gt_idx {
                gt_field = Some(field);
            }
        }

        if prefer_ds {
            if let Some(value) = ds_field {
                if let Some(parsed) = parse_numeric_str(value)? {
                    dest[sample_idx] = parsed;
                    continue;
                }
            }
        }

        if let Some(value) = gt_field {
            if let Some(parsed) = parse_vcf_genotype(value)? {
                dest[sample_idx] = parsed;
            }
        }
    }

    Ok(())
}

fn decode_bcf_record(
    record: &BcfRecord,
    header: &vcf::Header,
    n_samples: usize,
    prefer_ds: bool,
    dest: &mut [f64],
) -> Result<(), VariantIoError> {
    dest[..n_samples].fill(f64::NAN);

    let samples = record
        .samples()
        .map_err(|err| VariantIoError::Decode(format!("failed to access BCF samples: {err}")))?;

    if samples.format_count() == 0 {
        return Ok(());
    }

    let mut saw_gt = false;
    let mut used_ds = false;

    for result in samples.series() {
        let series = result
            .map_err(|err| VariantIoError::Decode(format!("failed to read BCF series: {err}")))?;
        let name = series.name(header).map_err(|err| {
            VariantIoError::Decode(format!("failed to resolve BCF FORMAT name: {err}"))
        })?;

        if prefer_ds && name == "DS" {
            decode_bcf_numeric_series(series, header, dest)?;
            used_ds = true;
        } else if name == key::GENOTYPE {
            decode_bcf_genotype_series(series, header, dest)?;
            saw_gt = true;
        }
    }

    if !saw_gt && !(prefer_ds && used_ds) {
        return Err(VariantIoError::Decode(
            "BCF record is missing the required GT FORMAT field".to_string(),
        ));
    }

    Ok(())
}

fn decode_bcf_numeric_series(
    series: noodles_bcf::record::samples::Series<'_>,
    header: &vcf::Header,
    dest: &mut [f64],
) -> Result<(), VariantIoError> {
    for (sample_idx, slot) in dest.iter_mut().enumerate() {
        if let Some(value) = series.get(header, sample_idx) {
            match value {
                Some(Ok(series_value)) => {
                    if let Some(parsed) = numeric_from_series_value(series_value)? {
                        *slot = parsed;
                    }
                }
                Some(Err(err)) => {
                    return Err(VariantIoError::Decode(format!(
                        "failed to decode BCF FORMAT value: {err}"
                    )));
                }
                None => {}
            }
        } else {
            return Err(VariantIoError::Decode(
                "BCF FORMAT series shorter than expected".to_string(),
            ));
        }
    }
    Ok(())
}

fn decode_bcf_genotype_series(
    series: noodles_bcf::record::samples::Series<'_>,
    header: &vcf::Header,
    dest: &mut [f64],
) -> Result<(), VariantIoError> {
    for (sample_idx, slot) in dest.iter_mut().enumerate() {
        if !slot.is_nan() {
            continue;
        }

        let Some(value) = series.get(header, sample_idx) else {
            return Err(VariantIoError::Decode(
                "BCF FORMAT series shorter than expected".to_string(),
            ));
        };

        match value {
            Some(Ok(series_value)) => {
                let parsed = match series_value {
                    SeriesValue::Genotype(genotype) => {
                        dosage_from_series_genotype(genotype.as_ref())?
                    }
                    other => numeric_from_series_value(other)?,
                };
                if let Some(value) = parsed {
                    *slot = value;
                }
            }
            Some(Err(err)) => {
                return Err(VariantIoError::Decode(format!(
                    "failed to decode BCF genotype value: {err}"
                )));
            }
            None => {}
        }
    }
    Ok(())
}

fn numeric_from_series_value(value: SeriesValue<'_>) -> Result<Option<f64>, VariantIoError> {
    match value {
        SeriesValue::Integer(n) => Ok(Some(n as f64)),
        SeriesValue::Float(n) => Ok(Some(n as f64)),
        SeriesValue::String(text) => parse_numeric_str(text.as_ref()),
        SeriesValue::Array(array) => numeric_from_series_array(array),
        SeriesValue::Genotype(genotype) => dosage_from_series_genotype(genotype.as_ref()),
        SeriesValue::Character(_) => Ok(None),
    }
}

fn numeric_from_series_array(array: SeriesArray<'_>) -> Result<Option<f64>, VariantIoError> {
    match array {
        SeriesArray::Integer(values) => match values.iter().next() {
            Some(Ok(Some(value))) => Ok(Some(value as f64)),
            Some(Ok(None)) | None => Ok(None),
            Some(Err(err)) => Err(VariantIoError::Decode(format!(
                "failed to decode BCF integer array: {err}"
            ))),
        },
        SeriesArray::Float(values) => match values.iter().next() {
            Some(Ok(Some(value))) => Ok(Some(value as f64)),
            Some(Ok(None)) | None => Ok(None),
            Some(Err(err)) => Err(VariantIoError::Decode(format!(
                "failed to decode BCF float array: {err}"
            ))),
        },
        SeriesArray::String(values) => match values.iter().next() {
            Some(Ok(Some(value))) => parse_numeric_str(value.as_ref()),
            Some(Ok(None)) | None => Ok(None),
            Some(Err(err)) => Err(VariantIoError::Decode(format!(
                "failed to decode BCF string array: {err}"
            ))),
        },
        SeriesArray::Character(_) => Ok(None),
    }
}

fn dosage_from_series_genotype(
    genotype: &dyn series::value::Genotype,
) -> Result<Option<f64>, VariantIoError> {
    let mut dosage = 0.0f64;
    let mut seen = false;
    for result in genotype.iter() {
        let (position, _) = result.map_err(|err| {
            VariantIoError::Decode(format!("failed to decode genotype allele: {err}"))
        })?;
        match position {
            Some(0) => {
                seen = true;
            }
            Some(_) => {
                dosage += 1.0;
                seen = true;
            }
            None => return Ok(None),
        }
    }
    if seen { Ok(Some(dosage)) } else { Ok(None) }
}

fn parse_numeric_str(text: &str) -> Result<Option<f64>, VariantIoError> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "." {
        Ok(None)
    } else {
        trimmed.parse::<f64>().map(Some).map_err(|err| {
            VariantIoError::Decode(format!("failed to parse numeric string '{trimmed}': {err}"))
        })
    }
}

fn parse_vcf_genotype(field: &str) -> Result<Option<f64>, VariantIoError> {
    if field.is_empty() {
        return Ok(None);
    }

    let mut dosage = 0.0f64;
    let mut seen = false;
    let bytes = field.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'/' | b'|' => {
                idx += 1;
            }
            b'.' => return Ok(None),
            b'0'..=b'9' => {
                let start = idx;
                idx += 1;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                let allele = field[start..idx].parse::<usize>().map_err(|err| {
                    VariantIoError::Decode(format!(
                        "failed to parse genotype allele '{field}': {err}"
                    ))
                })?;
                if allele > 0 {
                    dosage += 1.0;
                }
                seen = true;
            }
            other => {
                return Err(VariantIoError::Decode(format!(
                    "unexpected character '{other}' in genotype field"
                )));
            }
        }
    }

    if seen { Ok(Some(dosage)) } else { Ok(None) }
}

fn decode_plink_variant(bytes: &[u8], dest: &mut [f64], n_samples: usize, table: &[[f64; 4]; 256]) {
    let mut sample_idx = 0usize;
    for &byte in bytes {
        if sample_idx >= n_samples {
            break;
        }
        let decoded = &table[byte as usize];
        let remaining = n_samples - sample_idx;
        let take = remaining.min(4);
        dest[sample_idx..sample_idx + take].copy_from_slice(&decoded[..take]);
        sample_idx += take;
    }
}

fn decode_table() -> &'static [[f64; 4]; 256] {
    static TABLE: OnceLock<[[f64; 4]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0f64; 4]; 256];
        for byte in 0u16..256 {
            for offset in 0..4 {
                let code = ((byte >> (offset * 2)) & 0b11) as u8;
                table[byte as usize][offset] = match code {
                    0 => 0.0,
                    1 => f64::NAN,
                    2 => 1.0,
                    3 => 2.0,
                    _ => unreachable!(),
                };
            }
        }
        table
    })
}

fn normalize_path(path: &Path, extension: &str) -> PathBuf {
    if path.extension().map_or(false, |ext| ext == extension) {
        path.to_owned()
    } else {
        path.with_extension(extension)
    }
}

fn validate_bed_header(header: &[u8]) -> Result<(), PlinkIoError> {
    match header {
        [0x6c, 0x1b, 0x01] => Ok(()),
        [0x6c, 0x1b, mode] => Err(PlinkIoError::InvalidHeader(format!(
            "unsupported mode byte {mode:#04x} (only variant-major mode is supported)"
        ))),
        _ => Err(PlinkIoError::InvalidHeader(
            "missing PLINK magic bytes 0x6c 0x1b".to_string(),
        )),
    }
}

fn read_fam_records(path: &Path) -> Result<Vec<SampleRecord>, PlinkIoError> {
    let mut reader = open_text_source(path)?;
    let mut records = Vec::new();
    let mut line_no = 0usize;

    while let Some(line) = reader.next_line()? {
        line_no += 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let text = str::from_utf8(line).map_err(|err| PlinkIoError::Utf8 {
            path: path.display().to_string(),
            source: err,
        })?;
        let mut fields = text.split_whitespace();
        let (Some(fid), Some(iid), Some(pid), Some(mid), Some(sex), Some(phenotype)) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            return Err(PlinkIoError::MalformedRecord {
                path: path.display().to_string(),
                line: line_no,
                message: "expected 6 whitespace-delimited fields".to_string(),
            });
        };

        records.push(SampleRecord {
            family_id: fid.to_string(),
            individual_id: iid.to_string(),
            paternal_id: pid.to_string(),
            maternal_id: mid.to_string(),
            sex: sex.to_string(),
            phenotype: phenotype.to_string(),
        });
    }

    Ok(records)
}

fn count_bim_records(path: &Path) -> Result<usize, PlinkIoError> {
    let mut reader = open_text_source(path)?;
    let mut count = 0usize;
    while let Some(line) = reader.next_line()? {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        count += 1;
    }
    Ok(count)
}

fn is_remote_path(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.starts_with("gs://") || s.starts_with("http://") || s.starts_with("https://"))
        .unwrap_or(false)
}

fn is_http_path(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.starts_with("http://") || s.starts_with("https://"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::files::VariantSource;
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::Path;
    use tempfile::tempdir;

    fn fake_bcf_header() -> Vec<u8> {
        let header_text =
            b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n\0";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BCF");
        bytes.push(2);
        bytes.push(2);
        bytes.extend_from_slice(&(header_text.len() as u32).to_le_bytes());
        bytes.extend_from_slice(header_text);
        bytes
    }

    fn write_fake_bcf<P: AsRef<Path>>(path: P, payload: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut file = File::create(path)?;
        let header = fake_bcf_header();
        file.write_all(&header)?;
        file.write_all(payload)?;
        Ok(header)
    }

    fn read_all(mut source: VariantSource) -> Vec<u8> {
        let mut data = Vec::new();
        source.read_to_end(&mut data).unwrap();
        data
    }

    #[test]
    fn variant_source_reads_single_file_without_modification() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.bcf");
        let header = write_fake_bcf(&path, &[0x01, 0x02, 0x03]).unwrap();

        let source = open_variant_source(&path).unwrap();
        let bytes = read_all(source);

        let mut expected = header;
        expected.extend_from_slice(&[0x01, 0x02, 0x03]);
        assert_eq!(bytes, expected);
    }

    #[test]
    fn variant_source_skips_headers_when_concatenating_directory() {
        let dir = tempdir().unwrap();
        let files = [
            ("chr2.bcf", vec![0x20]),
            ("chr10.bcf", vec![0x10]),
            ("chr1.bcf", vec![0x01]),
        ];

        for (name, payload) in &files {
            let path = dir.path().join(name);
            write_fake_bcf(&path, payload).unwrap();
        }

        let mut expected = fake_bcf_header();
        expected.extend_from_slice(&[0x01]);
        expected.extend_from_slice(&[0x20]);
        expected.extend_from_slice(&[0x10]);

        let source = open_variant_source(dir.path()).unwrap();
        let bytes = read_all(source);

        assert_eq!(bytes, expected);
    }
}
