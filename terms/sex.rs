use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use infer_sex::{
    Chromosome, GenomeBuild, InferenceConfig, InferenceError, InferenceResult, InferredSex,
    PlatformDefinition, SexInferenceAccumulator, VariantInfo,
};
use thiserror::Error;

use crate::map::fit::VariantBlockSource;
use crate::map::io::{GenotypeDataset, GenotypeIoError, SelectionPlan};
use crate::map::variant_filter::VariantKey;

#[derive(Debug, Error)]
pub enum SexInferenceError {
    #[error("genotype I/O error: {0}")]
    Dataset(#[from] GenotypeIoError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("sex inference error: {0:?}")]
    Inference(InferenceError),
    #[error(
        "variant stream yielded more variants than expected (expected {expected}, observed {observed})"
    )]
    VariantOverflow { expected: usize, observed: usize },
    #[error(
        "variant stream terminated early (processed {observed} of {expected} expected variants)"
    )]
    VariantUnderflow { expected: usize, observed: usize },
    #[error(
        "insufficient sex-informative variants: {autosomes} autosomal and {y_non_par} Y non-PAR loci available"
    )]
    InsufficientInformativeVariants { autosomes: u64, y_non_par: u64 },
}

impl From<InferenceError> for SexInferenceError {
    fn from(value: InferenceError) -> Self {
        SexInferenceError::Inference(value)
    }
}

#[derive(Debug, Clone)]
pub struct SexInferenceRecord {
    pub individual_id: String,
    pub inference: InferenceResult,
}

#[derive(Debug, Clone)]
struct SexVariantSelection {
    keys: Vec<SelectedVariant>,
    plan: SelectionPlan,
    build: GenomeBuild,
}

#[derive(Debug, Clone)]
struct SelectedVariant {
    key: VariantKey,
    chrom: Chromosome,
}

impl SexVariantSelection {
    fn from_all_keys(keys: &[VariantKey], build: GenomeBuild) -> Self {
        const AUTOSOME_SAMPLE_TARGET: usize = 2000;

        let mut selected = Vec::new();
        let mut selected_autosomes = HashSet::new();
        let mut autosome_indices = Vec::new();

        for (index, key) in keys.iter().enumerate() {
            let Some(chrom) = classify_chromosome(&key.chromosome) else {
                continue;
            };

            match chrom {
                Chromosome::Autosome => autosome_indices.push(index),
                Chromosome::X | Chromosome::Y => {
                    selected.push((
                        index,
                        SelectedVariant {
                            key: key.clone(),
                            chrom,
                        },
                    ));
                }
            }
        }

        let autosome_sample_count = AUTOSOME_SAMPLE_TARGET.min(autosome_indices.len());
        for i in 0..autosome_sample_count {
            let source_idx = i * autosome_indices.len() / autosome_sample_count;
            let key_index = autosome_indices[source_idx];
            if selected_autosomes.insert(key_index) {
                selected.push((
                    key_index,
                    SelectedVariant {
                        key: keys[key_index].clone(),
                        chrom: Chromosome::Autosome,
                    },
                ));
            }
        }

        selected.sort_by_key(|entry| entry.0);

        let mut selected_indices = Vec::with_capacity(selected.len());
        let mut selected_keys = Vec::with_capacity(selected.len());
        for (index, variant) in selected.into_iter() {
            selected_indices.push(index);
            selected_keys.push(variant);
        }

        let plan = SelectionPlan::ByIndices(selected_indices);
        Self {
            keys: selected_keys,
            plan,
            build,
        }
    }
}

const SEX_TSV_HEADER: &str = concat!(
    "IID\tBuild\tSex\tY_Density\tX_AutoHet_Ratio\tComposite_Index\tAuto_Valid\tAuto_Het\t",
    "X_NonPAR_Valid\tX_NonPAR_Het\tY_NonPAR_Valid\tY_PAR_Valid",
);

pub fn infer_sex_to_tsv(
    genotype_path: &Path,
    force_build: Option<GenomeBuild>,
) -> Result<PathBuf, SexInferenceError> {
    let dataset = GenotypeDataset::open(genotype_path)?;
    let default_output = dataset.output_path("sex.tsv");

    let variant_keys = dataset.variant_keys_for_plan(&SelectionPlan::All)?;
    let build = force_build.unwrap_or_else(|| {
        let inferred = infer_build_from_keys(&variant_keys);
        eprintln!("Inferred Genome Build: {:?}", inferred);
        inferred
    });
    let selection = SexVariantSelection::from_all_keys(&variant_keys, build);
    let records = collect_inference(&dataset, &selection)?;

    write_results(&default_output, &records, build)?;

    Ok(default_output)
}

fn collect_inference(
    dataset: &GenotypeDataset,
    selection: &SexVariantSelection,
) -> Result<Vec<SexInferenceRecord>, SexInferenceError> {
    let sample_ids: Vec<String> = dataset
        .samples()
        .iter()
        .map(|record| record.individual_id.clone())
        .collect();
    let n_samples = sample_ids.len();

    let build = selection.build;
    let platform = derive_platform_definition(&selection.keys, build);
    ensure_informative_platform(&platform)?;
    let config = InferenceConfig {
        build,
        platform,
        thresholds: None,
    };
    let mut accumulators: Vec<SexInferenceAccumulator> = (0..n_samples)
        .map(|_| SexInferenceAccumulator::new(config))
        .collect();

    let mut block_source = dataset.block_source_with_plan(selection.plan.clone())?;
    let total_variants = selection.keys.len();
    let block_capacity = 256usize;
    let mut storage = vec![f64::NAN; block_capacity * n_samples];
    let mut processed = 0usize;

    while processed < total_variants {
        let capacity = block_capacity.min(total_variants - processed);
        let slice_len = capacity * n_samples;
        let filled = block_source.next_block_into(capacity, &mut storage[..slice_len])?;
        if filled == 0 {
            return Err(SexInferenceError::VariantUnderflow {
                expected: total_variants,
                observed: processed,
            });
        }
        if processed + filled > total_variants {
            return Err(SexInferenceError::VariantOverflow {
                expected: total_variants,
                observed: processed + filled,
            });
        }

        for local_idx in 0..filled {
            let selected = &selection.keys[processed + local_idx];
            let chrom = selected.chrom;
            let pos = selected.key.position;
            let column_offset = local_idx * n_samples;

            for sample_idx in 0..n_samples {
                let dosage = storage[column_offset + sample_idx];
                if dosage.is_nan() {
                    continue;
                }
                let is_het = dosage == 1.0;
                let info = VariantInfo {
                    chrom,
                    pos,
                    is_heterozygous: is_het,
                };
                accumulators[sample_idx].process_variant(&info);
            }
        }

        processed += filled;
    }

    if processed < total_variants {
        return Err(SexInferenceError::VariantUnderflow {
            expected: total_variants,
            observed: processed,
        });
    }

    finalize_records(accumulators, sample_ids)
}

fn finalize_records(
    accumulators: Vec<SexInferenceAccumulator>,
    sample_ids: Vec<String>,
) -> Result<Vec<SexInferenceRecord>, SexInferenceError> {
    accumulators
        .into_iter()
        .zip(sample_ids.into_iter())
        .map(|(acc, individual_id)| {
            let inference = acc.finish()?;
            Ok(SexInferenceRecord {
                individual_id,
                inference,
            })
        })
        .collect()
}

fn write_results(
    path: &Path,
    records: &[SexInferenceRecord],
    build: GenomeBuild,
) -> Result<(), SexInferenceError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", SEX_TSV_HEADER)?;
    for record in records {
        let label = sex_label(record.inference.final_call);
        let report = &record.inference.report;
        let y_density = report
            .y_genome_density
            .map_or("NA".to_string(), |v| format!("{v:.6}"));
        let x_ratio = report
            .x_autosome_het_ratio
            .map_or("NA".to_string(), |v| format!("{v:.6}"));
        let composite = report
            .composite_sex_index
            .map_or("NA".to_string(), |v| format!("{v:.6}"));

        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            record.individual_id,
            format!("{:?}", build),
            label,
            y_density,
            x_ratio,
            composite,
            report.auto_valid_count,
            report.auto_het_count,
            report.x_non_par_valid_count,
            report.x_non_par_het_count,
            report.y_non_par_valid_count,
            report.y_par_valid_count,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn sex_label(sex: InferredSex) -> &'static str {
    match sex {
        InferredSex::Male => "male",
        InferredSex::Female => "female",
    }
}

fn infer_build_from_keys(keys: &[VariantKey]) -> GenomeBuild {
    const GRCH38_THRESHOLD: u64 = 155_700_000;
    const GRCH37_THRESHOLD: u64 = 154_900_000;

    let max_x = keys
        .iter()
        .filter(|key| matches!(classify_chromosome(&key.chromosome), Some(Chromosome::X)))
        .map(|key| key.position)
        .max();

    match max_x {
        Some(pos) if pos >= GRCH38_THRESHOLD => GenomeBuild::Build38,
        Some(pos) if pos >= GRCH37_THRESHOLD => GenomeBuild::Build37,
        Some(_) => GenomeBuild::Build38,
        None => GenomeBuild::Build38,
    }
}

fn classify_chromosome(label: &str) -> Option<Chromosome> {
    let trimmed = label.trim();
    let bytes = trimmed.as_bytes();
    let stripped = if bytes.len() >= 3
        && matches!(bytes[0] | 32, b'c')
        && matches!(bytes[1] | 32, b'h')
        && matches!(bytes[2] | 32, b'r')
    {
        &trimmed[3..]
    } else {
        trimmed
    };

    if stripped.len() == 1 {
        match stripped.as_bytes()[0] {
            b'X' | b'x' => return Some(Chromosome::X),
            b'Y' | b'y' => return Some(Chromosome::Y),
            _ => {}
        }
    }

    match stripped.parse::<u8>().ok()? {
        23 => Some(Chromosome::X),
        24 => Some(Chromosome::Y),
        chrom if (1..=22).contains(&chrom) => Some(Chromosome::Autosome),
        _ => None,
    }
}

fn derive_platform_definition(keys: &[SelectedVariant], build: GenomeBuild) -> PlatformDefinition {
    let mut n_attempted_autosomes = 0u64;
    let mut n_attempted_y_nonpar = 0u64;

    for selected in keys {
        match selected.chrom {
            Chromosome::Autosome => n_attempted_autosomes += 1,
            Chromosome::Y if build.is_in_y_non_par(selected.key.position) => {
                n_attempted_y_nonpar += 1
            }
            _ => {}
        }
    }

    PlatformDefinition {
        n_attempted_autosomes,
        n_attempted_y_nonpar,
    }
}

fn ensure_informative_platform(platform: &PlatformDefinition) -> Result<(), SexInferenceError> {
    if platform.n_attempted_autosomes == 0 {
        return Err(SexInferenceError::InsufficientInformativeVariants {
            autosomes: platform.n_attempted_autosomes,
            y_non_par: platform.n_attempted_y_nonpar,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn classify_chromosome_recognizes_common_labels() {
        assert_eq!(classify_chromosome("X"), Some(Chromosome::X));
        assert_eq!(classify_chromosome("chrX"), Some(Chromosome::X));
        assert_eq!(classify_chromosome("Y"), Some(Chromosome::Y));
        assert_eq!(classify_chromosome("chrY"), Some(Chromosome::Y));
        assert_eq!(classify_chromosome("1"), Some(Chromosome::Autosome));
        assert_eq!(classify_chromosome("chr22"), Some(Chromosome::Autosome));
        assert_eq!(classify_chromosome("MT"), None);
    }

    #[test]
    fn infer_build_from_keys_detects_build_thresholds() {
        let keys = vec![VariantKey::new("chrX", 155_800_000)];
        assert_eq!(infer_build_from_keys(&keys), GenomeBuild::Build38);

        let keys = vec![VariantKey::new("X", 155_000_000)];
        assert_eq!(infer_build_from_keys(&keys), GenomeBuild::Build37);

        let keys: Vec<VariantKey> = Vec::new();
        assert_eq!(infer_build_from_keys(&keys), GenomeBuild::Build38);
    }

    #[test]
    fn platform_definition_counts_autosomes_and_y_nonpar() {
        let build = GenomeBuild::Build38;
        let keys = vec![
            SelectedVariant {
                key: VariantKey::new("1", 1_000),
                chrom: Chromosome::Autosome,
            },
            SelectedVariant {
                key: VariantKey::new("2", 2_000),
                chrom: Chromosome::Autosome,
            },
            SelectedVariant {
                key: VariantKey::new("Y", 3_000_000),
                chrom: Chromosome::Y,
            },
            SelectedVariant {
                key: VariantKey::new("Y", 56_887_902),
                chrom: Chromosome::Y,
            },
            SelectedVariant {
                key: VariantKey::new("Y", 56_887_903),
                chrom: Chromosome::Y,
            },
        ];

        let platform = derive_platform_definition(&keys, build);
        assert_eq!(platform.n_attempted_autosomes, 2);
        assert_eq!(platform.n_attempted_y_nonpar, 2);
    }

    #[test]
    fn ensure_informative_platform_rejects_missing_loci() {
        let ok_platform = PlatformDefinition {
            n_attempted_autosomes: 1,
            n_attempted_y_nonpar: 1,
        };
        ensure_informative_platform(&ok_platform).expect("platform should be accepted");

        let missing_autosomes = PlatformDefinition {
            n_attempted_autosomes: 0,
            n_attempted_y_nonpar: 5,
        };
        let err = ensure_informative_platform(&missing_autosomes)
            .expect_err("should reject missing autosomes");
        match err {
            SexInferenceError::InsufficientInformativeVariants {
                autosomes,
                y_non_par,
            } => {
                assert_eq!(autosomes, 0);
                assert_eq!(y_non_par, 5);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn selection_limits_autosomes_but_keeps_sex_chromosomes() {
        let build = GenomeBuild::Build38;
        let mut keys = Vec::new();
        for i in 0..3000u64 {
            keys.push(VariantKey::new("1", 10_000 + i));
        }
        keys.push(VariantKey::new("X", 3_000_000));
        keys.push(VariantKey::new("Y", 3_000_000));
        keys.push(VariantKey::new("Y", 56_887_903));

        let selection = SexVariantSelection::from_all_keys(&keys, build);

        assert_eq!(selection.keys.len(), 2003);
        if let SelectionPlan::ByIndices(indices) = &selection.plan {
            assert_eq!(indices.len(), selection.keys.len());
            assert!(indices.windows(2).all(|w| w[0] < w[1]));
            assert!(indices.contains(&0));
            assert!(indices.contains(&3000));
            assert!(indices.contains(&3001));
            assert!(indices.contains(&3002));

            let autosome_indices: Vec<_> =
                indices.iter().copied().filter(|idx| *idx < 3000).collect();
            assert_eq!(autosome_indices.len(), 2000);
            assert!(autosome_indices.contains(&0));
            assert!(autosome_indices.contains(&1500));
            assert!(autosome_indices.contains(&2998));
        } else {
            panic!("unexpected selection plan type");
        }
    }

    #[test]
    fn manual_accumulators_produce_expected_calls() {
        let build = GenomeBuild::Build38;
        let platform = PlatformDefinition {
            n_attempted_autosomes: 400,
            n_attempted_y_nonpar: 80,
        };
        let config = InferenceConfig {
            build,
            platform,
            thresholds: None,
        };

        let mut female_acc = SexInferenceAccumulator::new(config);
        let mut male_acc = SexInferenceAccumulator::new(config);

        for i in 0..400u64 {
            let pos = 1_000_000 + i;
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos,
                is_heterozygous: i % 2 == 0,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos,
                is_heterozygous: i % 3 == 0,
            });
        }

        for i in 0..200u64 {
            let pos = 3_000_000 + i;
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos,
                is_heterozygous: true,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos,
                is_heterozygous: false,
            });
        }

        for i in 0..80u64 {
            let pos = 3_000_000 + i;
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Y,
                pos,
                is_heterozygous: false,
            });
        }

        let female_result = female_acc.finish().unwrap();
        let male_result = male_acc.finish().unwrap();

        assert_eq!(female_result.final_call, InferredSex::Female);
        assert_eq!(male_result.final_call, InferredSex::Male);

        assert!(
            female_result
                .report
                .y_genome_density
                .expect("female density")
                < 0.2
        );
        assert!(
            female_result
                .report
                .x_autosome_het_ratio
                .expect("female ratio")
                > 0.5
        );
        assert!(male_result.report.y_genome_density.expect("male density") > 0.5);
        assert!(male_result.report.x_autosome_het_ratio.expect("male ratio") < 0.2);
    }

    #[test]
    fn sex_tsv_includes_metric_columns() -> Result<(), Box<dyn std::error::Error>> {
        let build = GenomeBuild::Build38;
        let platform = PlatformDefinition {
            n_attempted_autosomes: 10,
            n_attempted_y_nonpar: 2,
        };
        let config = InferenceConfig {
            build,
            platform,
            thresholds: None,
        };

        let mut female_acc = SexInferenceAccumulator::new(config);
        let mut male_acc = SexInferenceAccumulator::new(config);

        for i in 0..10u64 {
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos: 1_000 + i,
                is_heterozygous: i % 2 == 0,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos: 1_000 + i,
                is_heterozygous: i % 3 == 0,
            });
        }
        for i in 0..2u64 {
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Y,
                pos: 3_000_000 + i,
                is_heterozygous: false,
            });
        }
        for i in 0..4u64 {
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos: 3_000_000 + i,
                is_heterozygous: true,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos: 3_000_000 + i,
                is_heterozygous: false,
            });
        }

        let female = SexInferenceRecord {
            individual_id: "F1".to_string(),
            inference: female_acc.finish().unwrap(),
        };
        let male = SexInferenceRecord {
            individual_id: "M1".to_string(),
            inference: male_acc.finish().unwrap(),
        };

        let dir = tempdir()?;
        let output_path = dir.path().join("sex.tsv");
        write_results(&output_path, &[female, male], build)?;
        let tsv_contents = std::fs::read_to_string(&output_path)?;
        let mut lines = tsv_contents.lines();
        let header = lines.next().unwrap();
        assert_eq!(header, SEX_TSV_HEADER);

        for line in lines {
            let mut parts = line.split('\t');
            let _ = parts.next().unwrap();
            let build = parts.next().unwrap();
            let sex = parts.next().unwrap();
            let y_density = parts.next().unwrap();
            let x_ratio = parts.next().unwrap();
            let composite = parts.next().unwrap();
            let auto_valid = parts.next().unwrap();
            let auto_het = parts.next().unwrap();
            assert_ne!(y_density, "NA");
            assert_ne!(x_ratio, "NA");
            assert_ne!(composite, "NA");
            assert!(auto_valid.parse::<u64>().unwrap() > 0);
            assert!(auto_het.parse::<u64>().unwrap() > 0);
            let x_valid = parts.next().unwrap().parse::<u64>().unwrap();
            let _ = parts.next().unwrap();
            let y_non_par = parts.next().unwrap().parse::<u64>().unwrap();
            let y_par = parts.next().unwrap().parse::<u64>().unwrap();
            if sex == "female" {
                assert_eq!(y_non_par, 0);
                assert_eq!(y_par, 0);
            } else {
                assert!(y_non_par > 0);
                assert_eq!(y_par, 0);
            }
            assert!(x_valid > 0);
            assert_eq!(build, "Build38");
        }

        Ok(())
    }
}
