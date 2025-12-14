//! A single-file, pure-Rust façade which consumes PLINK 2.0 inputs
//! (`.pgen/.pvar/.psam`) and presents virtual PLINK 1.9 outputs
//! (`.bed/.bim/.fam`) through the same streaming traits your code
//! already uses:
//!   - `.bed` → `ByteRangeSource` (random-access byte ranges)
//!   - `.bim` / `.fam` → `TextSource` (pull-based line iterator)
//!
//! ## Fixed semantics (“best options”, no knobs)
//! - Multiallelic: always split every ALT; never drop. Variant order in
//!   the virtual outputs matches `.pvar`, expanded in ALT order and with
//!   multiallelics deterministically split.
//! - Allele orientation: A1 equals ALT and A2 equals REF (per split
//!   ALT). The virtual `.bim` follows the PLINK 1.9 contract with `cM =`
//!   `0` and synthesised IDs when needed.
//! - Genotype basis: prefer hard-calls; otherwise derive a hard-call from
//!   dosage using nearest-integer with ±0.10 tolerance, else mark the
//!   value as missing.
//! - Ploidy: autosomes plus pseudoautosomal regions are diploid; `X`
//!   (non-PAR), `Y`, and `MT` are treated as haploid for males.
//!   Heterozygotes in these contexts are coerced to missing before PLINK
//!   1.9 packing.
//! - `.bed` encoding: exact PLINK 1.9 2-bit codes (`00` hom ALT, `01`
//!   missing, `10` het, `11` hom REF; least-significant bit first within
//!   each byte).
//! - Split IDs: if `ID != "."` → `ID__ALT=<ALT>`; else use
//!   `chr:pos:ref:alt`.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::str;
use std::sync::{Arc, Mutex};

use crate::files::{
    // Traits
    ByteRangeSource,
    TextSource,
    // Helpers
    open_text_source,
};
/// Bring in your crate-local traits and error type.
/// These are expected to already exist (per your provided infrastructure).
use crate::score::pipeline::PipelineError;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public entrypoints
////////////////////////////////////////////////////////////////////////////////////////////////////

/// A handle which exposes virtual PLINK-1.9 streams backed by PLINK-2.0 inputs.
pub struct VirtualPlink19 {
    /// Random-access virtual `.bed` (PLINK-1.9 bytes).
    pub bed: Arc<dyn ByteRangeSource>,
    /// Streaming virtual `.bim` (PLINK-1.9 tab text).
    pub bim: Box<dyn TextSource>,
    /// Streaming virtual `.fam` (PLINK-1.9 tab text).
    pub fam: Box<dyn TextSource>,
    build: GenomeBuild,
}

impl VirtualPlink19 {
    /// Returns the inferred genome build (`"GRCh37"` or `"GRCh38"`).
    pub fn inferred_genome_build(&self) -> &'static str {
        match self.build {
            GenomeBuild::Grch37 => "GRCh37",
            GenomeBuild::Grch38 => "GRCh38",
        }
    }
}

/// Open from filesystem paths. `.pvar` and `.psam` are opened with your existing
/// `open_text_source`. The `.pgen` is opened as a local file here.
/// If you need remote `.pgen` support, use `open_virtual_plink19_from_sources`
/// and pass an appropriate `ByteRangeSource` for `.pgen`.
pub fn open_virtual_plink19_from_paths(
    pgen_path: &Path,
    pvar_path: &Path,
    psam_path: &Path,
) -> Result<VirtualPlink19, PipelineError> {
    let mut pvar_for_plan = open_text_source(pvar_path)?;
    let mut psam_for_plan = open_text_source(psam_path)?;
    let pgen = Arc::new(LocalFileByteRangeSource::open(pgen_path)?);

    open_virtual_plink19_from_sources(pgen, &mut *pvar_for_plan, &mut *psam_for_plan)
}

/// Open from caller-provided sources. Callers may pass custom/remote-capable
/// `ByteRangeSource` for `.pgen` and `TextSource`s for `.pvar/.psam`.
pub fn open_virtual_plink19_from_sources(
    pgen: Arc<dyn ByteRangeSource>,
    pvar_for_plan: &mut dyn TextSource,
    psam_for_plan: &mut dyn TextSource,
) -> Result<VirtualPlink19, PipelineError> {
    let header = PgenHeader::parse(&*pgen)?;
    let psam_info = PsamInfo::from_psam(psam_for_plan)?;
    let plan = VariantPlan::from_pvar(pvar_for_plan)?;
    let inferred_build = plan.inferred_build();

    if header.m_variants != 0 && header.m_variants as usize != plan.in_variants {
        return Err(PipelineError::Io(format!(
            "Variant count mismatch: .pgen header has {}, .pvar expands to {}",
            header.m_variants, plan.in_variants
        )));
    }

    if header.n_samples != 0 && header.n_samples as usize != psam_info.n_samples {
        return Err(PipelineError::Io(format!(
            "Sample-count mismatch: .pgen header has {}, .psam has {}",
            header.n_samples, psam_info.n_samples
        )));
    }

    let bim_lines = plan.bim_lines().to_vec();
    let fam_rows = psam_info.fam_rows.clone();

    let bed_source: Arc<dyn ByteRangeSource> = match header.mode {
        PgenMode::Bed => {
            if plan.out_variants != plan.in_variants {
                return Err(PipelineError::Io(
                    "Mode 0x01 (.bed) cannot expand multiallelic variants; re-encode the input to mode 0x10/0x11"
                        .into(),
                ));
            }
            pgen.clone()
        }
        _ => {
            let decoder = PgenDecoder::new(
                pgen.clone(),
                header,
                psam_info.n_samples,
                plan.in_variants,
                plan.alts_per_in.clone(),
            )?;
            let sex_by_sample_arc: Arc<[u8]> =
                Arc::from(psam_info.sex_by_sample.clone().into_boxed_slice());
            Arc::new(VirtualBed::new(
                decoder,
                plan.clone(),
                psam_info.n_samples,
                sex_by_sample_arc,
            )?)
        }
    };

    let bim: Box<dyn TextSource> = Box::new(VirtualBim::from_lines(bim_lines));
    let fam: Box<dyn TextSource> = Box::new(VirtualFam::from_rows(fam_rows));

    Ok(VirtualPlink19 {
        bed: bed_source,
        bim,
        fam,
        build: inferred_build,
    })
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PSAM → FAM (header semantics + row mapping)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct PsamInfo {
    n_samples: usize,
    sex_by_sample: Vec<u8>,
    fam_rows: Vec<FamRow>,
}

#[derive(Clone, Default)]
struct PsamColumns {
    fid_idx: Option<usize>,
    iid_idx: Option<usize>,
    pat_idx: Option<usize>,
    mat_idx: Option<usize>,
    sex_idx: Option<usize>,
    pheno_idx: Option<usize>,
    pheno1_idx: Option<usize>,
    sid_idx: Option<usize>,
}

#[derive(Clone, Default)]
struct FamRow {
    fid: String,
    iid: String,
    pat: String,
    mat: String,
    sex: String,
    phe: String,
}

impl FamRow {
    fn as_line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.fid, self.iid, self.pat, self.mat, self.sex, self.phe
        )
    }
}

impl PsamInfo {
    fn from_psam(source: &mut dyn TextSource) -> Result<Self, PipelineError> {
        let mut header_tokens: Option<Vec<String>> = None;
        let mut columns: Option<PsamColumns> = None;
        let mut sex_by_sample: Vec<u8> = Vec::new();
        let mut fam_rows: Vec<FamRow> = Vec::new();

        while let Some(line) = source.next_line()? {
            let s = str::from_utf8(line)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .psam: {e}")))?;

            if s.starts_with('#') {
                let cols = s.trim_start_matches('#').trim();
                if !cols.is_empty() && !cols.starts_with('#') {
                    header_tokens = Some(cols.split_whitespace().map(|t| t.to_string()).collect());
                    columns = None; // the last header wins
                }
                continue;
            }

            let trimmed = s.trim();
            if trimmed.is_empty() {
                continue;
            }

            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }

            if columns.is_none() {
                columns = Some(match header_tokens.as_ref() {
                    Some(tokens) => PsamColumns::from_header(tokens),
                    None => PsamColumns::from_headerless(fields.len()),
                }?);
            }

            let cols = columns.as_ref().unwrap();
            let fam_row = FamRow::from_fields(&fields, cols);
            if fam_row.iid == "0" {
                return Err(PipelineError::Io(
                    "IID must not be '0' (PSAM/FAM contract)".into(),
                ));
            }
            let sex_code = parse_sex_token(&fam_row.sex);
            sex_by_sample.push(sex_code);
            fam_rows.push(fam_row);
        }

        let final_columns = if let Some(cols) = columns {
            cols
        } else {
            let tokens = header_tokens
                .ok_or_else(|| PipelineError::Io("Missing .psam header (#FID/#IID…)".into()))?;
            PsamColumns::from_header(&tokens)?
        };

        drop(final_columns);

        Ok(Self {
            n_samples: sex_by_sample.len(),
            sex_by_sample,
            fam_rows,
        })
    }
}

fn parse_sex_token(token: &str) -> u8 {
    match token.trim() {
        "1" => 1,
        "2" => 2,
        "M" | "m" => 1,
        "F" | "f" => 2,
        t if t.eq_ignore_ascii_case("male") => 1,
        t if t.eq_ignore_ascii_case("female") => 2,
        t if t.eq_ignore_ascii_case("unknown") => 0,
        t if t.eq_ignore_ascii_case("unk") => 0,
        t if t.eq_ignore_ascii_case("u") => 0,
        "0" | "NA" | "na" | "Na" | "nA" | "." | "nan" | "NaN" | "NAN" => 0,
        _ => 0,
    }
}

impl PsamColumns {
    fn from_header(cols: &[String]) -> Result<Self, PipelineError> {
        let mut out = PsamColumns::default();
        for (i, c) in cols.iter().enumerate() {
            match c.to_ascii_uppercase().as_str() {
                "FID" => out.fid_idx = Some(i),
                "IID" => out.iid_idx = Some(i),
                "PAT" => out.pat_idx = Some(i),
                "MAT" => out.mat_idx = Some(i),
                "SEX" => out.sex_idx = Some(i),
                "PHENO" | "PHENOTYPE" => out.pheno_idx = Some(i),
                "PHENO1" => out.pheno1_idx = Some(i),
                "SID" => out.sid_idx = Some(i),
                _ => {}
            }
        }
        if out.iid_idx.is_none() && out.fid_idx.is_none() {
            return Err(PipelineError::Io(
                "Invalid .psam header: need #FID or #IID".to_string(),
            ));
        }
        Ok(out)
    }

    fn from_headerless(field_count: usize) -> Result<Self, PipelineError> {
        if field_count >= 6 {
            Ok(PsamColumns {
                fid_idx: Some(0),
                iid_idx: Some(1),
                pat_idx: Some(2),
                mat_idx: Some(3),
                sex_idx: Some(4),
                pheno_idx: None,
                pheno1_idx: Some(5),
                sid_idx: None,
            })
        } else if field_count == 5 {
            Ok(PsamColumns {
                fid_idx: Some(0),
                iid_idx: Some(1),
                pat_idx: Some(2),
                mat_idx: Some(3),
                sex_idx: Some(4),
                pheno_idx: None,
                pheno1_idx: None,
                sid_idx: None,
            })
        } else {
            Err(PipelineError::Io(
                "Headerless .psam requires 5 or 6 columns".to_string(),
            ))
        }
    }
}

fn coerce_pheno_token(tok: &str) -> String {
    let t = tok.trim();
    if t.is_empty() {
        return "-9".to_string();
    }
    if t.eq_ignore_ascii_case("na")
        || t.eq_ignore_ascii_case("nan")
        || t == "."
        || t.eq_ignore_ascii_case("none")
    {
        return "-9".to_string();
    }
    if t == "0" {
        return "-9".to_string();
    }
    if t.parse::<f64>().is_ok() {
        t.to_string()
    } else {
        "-9".to_string()
    }
}

impl FamRow {
    fn from_fields(fields: &[&str], cols: &PsamColumns) -> FamRow {
        let get_clean = |idx: Option<usize>| -> Option<String> {
            idx.and_then(|i| fields.get(i))
                .map(|s| s.trim())
                .filter(|t| !t.is_empty())
                .map(|t| t.to_string())
        };

        let mut iid = get_clean(cols.iid_idx)
            .or_else(|| get_clean(cols.sid_idx))
            .or_else(|| get_clean(cols.fid_idx))
            .unwrap_or_else(|| "0".to_string());
        if iid == "0" {
            iid = get_clean(cols.sid_idx).unwrap_or_else(|| "0".to_string());
        }

        let fid = get_clean(cols.fid_idx).unwrap_or_else(|| iid.clone());
        let pat = get_clean(cols.pat_idx).unwrap_or_else(|| "0".to_string());
        let mat = get_clean(cols.mat_idx).unwrap_or_else(|| "0".to_string());
        let sex = get_clean(cols.sex_idx).unwrap_or_else(|| "0".to_string());
        // PHENO1 takes precedence when both PHENO and PHENO1 are present.
        let phe = cols
            .pheno1_idx
            .and_then(|i| fields.get(i))
            .or_else(|| cols.pheno_idx.and_then(|i| fields.get(i)))
            .map(|s| coerce_pheno_token(s))
            .unwrap_or_else(|| "-9".to_string());
        FamRow {
            fid,
            iid,
            pat,
            mat,
            sex,
            phe,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PVAR → VariantPlan (always split) + BIM streaming transform
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping from virtual BED variant index (post-split) to PGEN record index
/// and the ALT ordinal within that record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HaploidyKind {
    Diploid,
    HaploidMales,
    HaploidAll,
    HaploidMalesFemalesMissing,
}

#[derive(Clone)]
struct VariantPlan {
    /// Total input variants before splitting (to sanity check decoder bounds).
    in_variants: usize,
    /// Total emitted variants after splitting.
    out_variants: usize,
    /// Dense mapping: out_idx → (in_idx, alt_ordinal_1based).
    out_to_in: Vec<(u32, u16)>,
    /// Per-output variant haploidy behaviour.
    haploidy: Vec<HaploidyKind>,
    /// BIM lines corresponding to each output variant.
    bim_lines: Vec<BimLine>,
    /// ALT allele count per input variant.
    alts_per_in: Vec<u16>,
    build: GenomeBuild,
}

#[derive(Clone)]
struct VariantRangeEntry {
    chrom: String,
    pos: u64,
    out_start: usize,
    out_end: usize,
}

#[derive(Clone, Copy)]
struct PvarCols {
    chrom: usize,
    id: usize,
    pos: usize,
    refa: usize,
    alt: usize,
}

impl PvarCols {
    fn from_header_line(line: &str) -> Result<Self, PipelineError> {
        let body = line.trim_start_matches('#').trim();
        let tokens: Vec<&str> = body.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(PipelineError::Io("Empty .pvar header line".to_string()));
        }
        Self::from_tokens(&tokens)
    }

    fn from_tokens(tokens: &[&str]) -> Result<Self, PipelineError> {
        let mut chrom = None;
        let mut id = None;
        let mut pos = None;
        let mut refa = None;
        let mut alt = None;

        for (i, token) in tokens.iter().enumerate() {
            let upper = token.trim().trim_start_matches('#').to_ascii_uppercase();
            match upper.as_str() {
                "CHROM" => chrom = Some(i),
                "ID" => id = Some(i),
                "POS" | "BP" => pos = Some(i),
                "REF" => refa = Some(i),
                "ALT" => alt = Some(i),
                _ => {}
            }
        }

        let chrom = chrom.ok_or_else(|| PipelineError::Io(".pvar header missing CHROM".into()))?;
        let id = id.ok_or_else(|| PipelineError::Io(".pvar header missing ID".into()))?;
        let pos = pos.ok_or_else(|| PipelineError::Io(".pvar header missing POS".into()))?;
        let refa = refa.ok_or_else(|| PipelineError::Io(".pvar header missing REF".into()))?;
        let alt = alt.ok_or_else(|| PipelineError::Io(".pvar header missing ALT".into()))?;

        Ok(PvarCols {
            chrom,
            id,
            pos,
            refa,
            alt,
        })
    }

    fn from_headerless(field_count: usize) -> Result<Self, PipelineError> {
        if field_count >= 6 {
            Ok(PvarCols {
                chrom: 0,
                id: 1,
                pos: 3,
                refa: 4,
                alt: 5,
            })
        } else if field_count == 5 {
            Ok(PvarCols {
                chrom: 0,
                id: 1,
                pos: 2,
                refa: 3,
                alt: 4,
            })
        } else {
            Err(PipelineError::Io(
                "Headerless .pvar requires ≥5 columns".to_string(),
            ))
        }
    }
}

impl VariantPlan {
    fn from_pvar(pvar: &mut dyn TextSource) -> Result<Self, PipelineError> {
        let mut out_to_in: Vec<(u32, u16)> = Vec::with_capacity(1 << 20);
        let mut haploidy: Vec<HaploidyKind> = Vec::with_capacity(1 << 20);
        let mut per_variant: Vec<VariantRangeEntry> = Vec::with_capacity(1 << 16);
        let mut bim_lines: Vec<BimLine> = Vec::with_capacity(1 << 20);
        let mut alts_per_in: Vec<u16> = Vec::with_capacity(1 << 16);
        let mut header_cols: Option<PvarCols> = None;
        let mut in_idx: u32 = 0;
        let mut in_variants: usize = 0;
        let mut max_x_pos: u64 = 0;
        let mut saw_x_par37_only = false;
        let mut saw_x_par38_only = false;

        while let Some(line) = pvar.next_line()? {
            let s = str::from_utf8(line)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .pvar: {e}")))?;
            let trimmed = s.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with("##") {
                continue;
            }
            if trimmed.starts_with('#') {
                header_cols = Some(PvarCols::from_header_line(trimmed)?);
                continue;
            }

            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }
            let cols = if let Some(cols) = header_cols {
                cols
            } else {
                let derived = PvarCols::from_headerless(fields.len())?;
                header_cols = Some(derived);
                derived
            };

            let chrom_raw = *fields
                .get(cols.chrom)
                .ok_or_else(|| ioerr(".pvar missing CHROM column"))?;
            let pos_raw = *fields
                .get(cols.pos)
                .ok_or_else(|| ioerr(".pvar missing POS column"))?;
            let id_raw = *fields
                .get(cols.id)
                .ok_or_else(|| ioerr(".pvar missing ID column"))?;
            let ref_raw = *fields
                .get(cols.refa)
                .ok_or_else(|| ioerr(".pvar missing REF column"))?;
            let alt_raw = *fields
                .get(cols.alt)
                .ok_or_else(|| ioerr(".pvar missing ALT column"))?;

            let chrom = normalize_chrom(chrom_raw);
            let pos = pos_raw
                .parse::<u64>()
                .map_err(|_| ioerr("Invalid POS in .pvar (expected integer)"))?;
            if chrom == "X" {
                max_x_pos = max_x_pos.max(pos);
                let in37 = in_any_range(pos, GRCH37_X_PAR);
                let in38 = in_any_range(pos, GRCH38_X_PAR);
                if in37 && !in38 {
                    saw_x_par37_only = true;
                }
                if in38 && !in37 {
                    saw_x_par38_only = true;
                }
            }

            let alts: Vec<&str> = alt_raw
                .split(',')
                .map(|a| a.trim())
                .filter(|a| !a.is_empty() && *a != ".")
                .collect();

            for alt in &alts {
                if is_symbolic_alt(alt) {
                    return Err(PipelineError::Io(format!(
                        "Symbolic ALT '{}' unsupported for variant {}:{} (ID {})",
                        alt, chrom, pos, id_raw
                    )));
                }
            }

            let out_start = out_to_in.len();
            let mut alt_ord: u16 = 1;
            for alt in alts.iter() {
                out_to_in.push((in_idx, alt_ord));
                haploidy.push(HaploidyKind::Diploid);
                bim_lines.push(BimLine {
                    chrom: chrom.clone(),
                    pos,
                    id: id_raw.to_string(),
                    refa: ref_raw.to_string(),
                    alt: (*alt).to_string(),
                });
                alt_ord += 1;
            }
            let out_end = out_to_in.len();
            per_variant.push(VariantRangeEntry {
                chrom,
                pos,
                out_start,
                out_end,
            });

            alts_per_in.push(alts.len() as u16);
            in_idx += 1;
            in_variants += 1;
        }

        if header_cols.is_none() {
            return Err(PipelineError::Io(
                "Missing .pvar header or inferable columns".to_string(),
            ));
        }

        if saw_x_par37_only && saw_x_par38_only {
            return Err(PipelineError::Io(
                "Encountered X PAR loci matching both GRCh37-only and GRCh38-only ranges; cannot infer a single build"
                    .to_string(),
            ));
        }

        let build = if saw_x_par37_only {
            GenomeBuild::Grch37
        } else if saw_x_par38_only {
            GenomeBuild::Grch38
        } else {
            infer_genome_build(max_x_pos)
        };
        for entry in per_variant {
            let hap = haploidy_for_variant(&entry.chrom, entry.pos, build);
            for idx in entry.out_start..entry.out_end {
                if let Some(slot) = haploidy.get_mut(idx) {
                    *slot = hap;
                }
            }
        }

        Ok(Self {
            in_variants,
            out_variants: out_to_in.len(),
            out_to_in,
            haploidy,
            bim_lines,
            alts_per_in,
            build,
        })
    }

    #[inline]
    fn mapping(&self, out_idx: usize) -> Option<(u32, u16)> {
        self.out_to_in.get(out_idx).copied()
    }

    #[inline]
    fn haploidy_of(&self, out_idx: usize) -> Option<HaploidyKind> {
        self.haploidy.get(out_idx).copied()
    }

    fn bim_lines(&self) -> &[BimLine] {
        &self.bim_lines
    }

    #[inline]
    fn alt_count_of_in(&self, in_idx: u32) -> u16 {
        self.alts_per_in.get(in_idx as usize).copied().unwrap_or(0)
    }

    #[inline]
    fn inferred_build(&self) -> GenomeBuild {
        self.build
    }
}

fn ioerr(msg: &str) -> PipelineError {
    PipelineError::Io(msg.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenomeBuild {
    Grch37,
    Grch38,
}

fn infer_genome_build(max_x_pos: u64) -> GenomeBuild {
    const GRCH38_THRESHOLD: u64 = 155_700_000;
    const GRCH37_THRESHOLD: u64 = 154_900_000;
    if max_x_pos >= GRCH38_THRESHOLD {
        GenomeBuild::Grch38
    } else if max_x_pos >= GRCH37_THRESHOLD {
        GenomeBuild::Grch37
    } else {
        GenomeBuild::Grch38
    }
}

fn normalize_chrom(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut body = trimmed;
    if trimmed.len() >= 3 && trimmed[..3].eq_ignore_ascii_case("chr") {
        body = &trimmed[3..];
    }
    let upper = body.to_ascii_uppercase();
    match upper.as_str() {
        "M" => "MT".to_string(),
        _ => upper,
    }
}

fn is_symbolic_alt(alt: &str) -> bool {
    let trimmed = alt.trim();
    if trimmed == "*" {
        return true;
    }
    if trimmed.starts_with('<') && trimmed.ends_with('>') {
        return true;
    }
    trimmed.contains('[') || trimmed.contains(']')
}

const GRCH37_X_PAR: &[(u64, u64)] = &[(60_001, 2_699_520), (154_931_044, 155_260_560)];
const GRCH38_X_PAR: &[(u64, u64)] = &[(10_001, 2_781_479), (155_701_383, 156_030_895)];

fn in_any_range(pos: u64, ranges: &[(u64, u64)]) -> bool {
    ranges
        .iter()
        .any(|(start, end)| pos >= *start && pos <= *end)
}

fn haploidy_for_variant(chrom: &str, pos: u64, build: GenomeBuild) -> HaploidyKind {
    match chrom {
        "X" => {
            let ranges = match build {
                GenomeBuild::Grch37 => GRCH37_X_PAR,
                GenomeBuild::Grch38 => GRCH38_X_PAR,
            };
            if in_any_range(pos, ranges) {
                HaploidyKind::Diploid
            } else {
                HaploidyKind::HaploidMales
            }
        }
        "Y" => HaploidyKind::HaploidMalesFemalesMissing,
        "MT" => HaploidyKind::HaploidAll,
        _ => HaploidyKind::Diploid,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .bim (TextSource): split multiallelic, A1=ALT, A2=REF, cM=0, stable IDs
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct BimLine {
    chrom: String,
    pos: u64,
    id: String,
    refa: String,
    alt: String,
}

#[derive(Clone)]
struct VirtualBim {
    lines: Vec<BimLine>,
    next_idx: usize,
    carry: Option<Box<[u8]>>,
}

impl VirtualBim {
    fn from_lines(lines: Vec<BimLine>) -> Self {
        Self {
            lines,
            next_idx: 0,
            carry: None,
        }
    }

    fn format_line(entry: &BimLine) -> String {
        let id_out = if entry.id != "." && !entry.id.is_empty() {
            format!("{}__ALT={}", entry.id, entry.alt)
        } else {
            format!("{}:{}:{}:{}", entry.chrom, entry.pos, entry.refa, entry.alt)
        };
        format!(
            "{}\t{}\t0\t{}\t{}\t{}",
            entry.chrom, id_out, entry.pos, entry.alt, entry.refa
        )
    }
}

impl TextSource for VirtualBim {
    fn len(&self) -> Option<u64> {
        Some(self.lines.len() as u64)
    }

    fn next_line<'a>(&'a mut self) -> Result<Option<&'a [u8]>, PipelineError> {
        if self.next_idx >= self.lines.len() {
            return Ok(None);
        }
        let line = Self::format_line(&self.lines[self.next_idx]);
        self.next_idx += 1;
        self.carry = Some(line.into_bytes().into_boxed_slice());
        Ok(self.carry.as_deref())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .fam (TextSource): map .psam to .fam with fixed defaults
////////////////////////////////////////////////////////////////////////////////////////////////////

struct VirtualFam {
    rows: Vec<FamRow>,
    next_idx: usize,
    carry: Option<Box<[u8]>>,
}

impl VirtualFam {
    fn from_rows(rows: Vec<FamRow>) -> Self {
        Self {
            rows,
            next_idx: 0,
            carry: None,
        }
    }
}

impl TextSource for VirtualFam {
    fn len(&self) -> Option<u64> {
        Some(self.rows.len() as u64)
    }

    fn next_line<'a>(&'a mut self) -> Result<Option<&'a [u8]>, PipelineError> {
        if self.next_idx >= self.rows.len() {
            return Ok(None);
        }
        let line = self.rows[self.next_idx].as_line();
        self.next_idx += 1;
        self.carry = Some(line.into_bytes().into_boxed_slice());
        Ok(self.carry.as_deref())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .bed (ByteRangeSource): fixed 3-byte header + B bytes per variant
////////////////////////////////////////////////////////////////////////////////////////////////////

const BED_MAGIC_0: u8 = 0x6c;
const BED_MAGIC_1: u8 = 0x1b;
const BED_MODE_SNP_MAJOR: u8 = 0x01;

#[derive(Clone)]
struct VirtualBed {
    inner: Arc<Mutex<PgenDecoder>>, // guarded decoder (seekable + scratch buffers)
    plan: VariantPlan,
    n_samples: usize,
    block_bytes: usize, // ceil(n_samples / 4)
    // small LRU of packed blocks by out-variant index
    cache: Arc<Mutex<BlockCache>>,
    sex_by_sample: Arc<[u8]>,
}

impl VirtualBed {
    fn new(
        decoder: PgenDecoder,
        plan: VariantPlan,
        n_samples: usize,
        sex_by_sample: Arc<[u8]>,
    ) -> Result<Self, PipelineError> {
        if sex_by_sample.len() != n_samples {
            return Err(PipelineError::Compute(
                "SEX column count mismatch with sample count".into(),
            ));
        }
        let block_bytes = (n_samples + 3) / 4;
        Ok(Self {
            inner: Arc::new(Mutex::new(decoder)),
            plan,
            n_samples,
            block_bytes,
            cache: Arc::new(Mutex::new(BlockCache::new(256))),
            sex_by_sample,
        })
    }

    #[inline]
    fn total_len(&self) -> u64 {
        3 + (self.plan.out_variants as u64) * (self.block_bytes as u64)
    }

    /// Pack 0/1/2/255 hard-calls (A1 dosage) into PLINK 1.9 2-bit codes.
    /// Codes (LSB-first per 2-bit field):
    ///   00 hom-A1 (A1 dosage 2)
    ///   10 het     (A1 dosage 1)
    ///   11 hom-A2 (A1 dosage 0)
    ///   01 missing
    fn pack_to_block(dst: &mut [u8], hardcalls: &[u8]) {
        debug_assert_eq!(dst.len(), (hardcalls.len() + 3) / 4);
        for chunk_i in 0..dst.len() {
            let base = chunk_i * 4;
            let mut byte = 0u8;
            for j in 0..4 {
                let idx = base + j;
                let code = if idx < hardcalls.len() {
                    match hardcalls[idx] {
                        2 => 0b00,
                        1 => 0b10,
                        0 => 0b11,
                        _ => 0b01, // 255 or anything else → missing
                    }
                } else {
                    0b01
                };
                byte |= code << (2 * j);
            }
            dst[chunk_i] = byte;
        }
    }
}

fn enforce_haploidy(hardcalls: &mut [u8], sex_by_sample: &[u8], kind: HaploidyKind) {
    match kind {
        HaploidyKind::Diploid => {}
        HaploidyKind::HaploidAll => {
            for val in hardcalls.iter_mut() {
                if *val == 1 {
                    *val = 255;
                }
            }
        }
        HaploidyKind::HaploidMales => {
            let n = hardcalls.len().min(sex_by_sample.len());
            for i in 0..n {
                if sex_by_sample[i] == 1 && hardcalls[i] == 1 {
                    hardcalls[i] = 255;
                }
            }
        }
        HaploidyKind::HaploidMalesFemalesMissing => {
            let n = hardcalls.len().min(sex_by_sample.len());
            for i in 0..n {
                let sex = sex_by_sample[i];
                if sex == 2 {
                    hardcalls[i] = 255;
                    continue;
                }
                if hardcalls[i] == 1 {
                    hardcalls[i] = 255;
                }
            }
        }
    }
}

fn fill_sample_ploidy(
    buf: &mut Vec<u8>,
    kind: HaploidyKind,
    sex_by_sample: &[u8],
    n_samples: usize,
) {
    buf.clear();
    buf.resize(n_samples, 2);
    match kind {
        HaploidyKind::Diploid => {}
        HaploidyKind::HaploidAll => {
            for v in buf.iter_mut() {
                *v = 1;
            }
        }
        HaploidyKind::HaploidMales => {
            let limit = sex_by_sample.len().min(n_samples);
            for i in 0..limit {
                if sex_by_sample[i] == 1 {
                    buf[i] = 1;
                }
            }
        }
        HaploidyKind::HaploidMalesFemalesMissing => {
            let limit = sex_by_sample.len().min(n_samples);
            for i in 0..limit {
                match sex_by_sample[i] {
                    1 => buf[i] = 1,
                    2 => buf[i] = 0,
                    _ => {}
                }
            }
        }
    }
}

impl ByteRangeSource for VirtualBed {
    fn len(&self) -> u64 {
        self.total_len()
    }

    fn read_at(&self, mut offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        if dst.is_empty() {
            return Ok(());
        }

        let total = self.total_len();
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| ioerr("Overflow in read_at range"))?;
        if end > total {
            return Err(ioerr("Attempted to read past end of virtual .bed"));
        }

        let mut written = 0usize;
        let mut sample_ploidy_buf: Vec<u8> = Vec::new();

        // 1) Serve the 3-byte header if requested.
        if offset < 3 {
            let hdr = [BED_MAGIC_0, BED_MAGIC_1, BED_MODE_SNP_MAJOR];
            while offset < 3 && written < dst.len() {
                dst[written] = hdr[offset as usize];
                offset += 1;
                written += 1;
            }
            if written == dst.len() {
                return Ok(());
            }
        }

        // 2) Serve the body: contiguous blocks of size self.block_bytes per variant.
        let body_off = offset - 3;
        let mut out_idx = (body_off / (self.block_bytes as u64)) as usize;
        let mut within_block = (body_off % (self.block_bytes as u64)) as usize;

        let mut decoder = self.inner.lock().unwrap();

        while written < dst.len() {
            if out_idx >= self.plan.out_variants {
                break;
            }

            // Copy bytes from this block.
            let remaining_in_block = self.block_bytes - within_block;
            let remaining_in_dst = dst.len() - written;
            let to_copy = remaining_in_block.min(remaining_in_dst);

            // Fetch or produce the packed block for this out-variant.
            let mut cache = self.cache.lock().unwrap();
            if let Some(buf) = cache.get(out_idx) {
                let start = within_block;
                let end = start + to_copy;
                dst[written..written + to_copy].copy_from_slice(&buf[start..end]);
            } else {
                drop(cache); // release lock before decoding

                // Decode hard-calls for this (in_idx, alt_ord) into a scratch buffer.
                let (in_idx, alt_ord) = self
                    .plan
                    .mapping(out_idx)
                    .ok_or_else(|| ioerr("VariantPlan mapping out of bounds"))?;
                let alt_count = self.plan.alt_count_of_in(in_idx);
                if alt_count == 0 {
                    return Err(PipelineError::Io(format!(
                        "ALT count missing for variant {} in .pvar plan",
                        in_idx
                    )));
                }
                if alt_count != 0 && alt_ord > alt_count {
                    return Err(ioerr("ALT ordinal exceeds allele count in .pvar"));
                }
                let haploidy_kind = self.plan.haploidy_of(out_idx);
                let sample_ploidy = haploidy_kind.and_then(|kind| {
                    if matches!(kind, HaploidyKind::Diploid) {
                        None
                    } else {
                        fill_sample_ploidy(
                            &mut sample_ploidy_buf,
                            kind,
                            &self.sex_by_sample,
                            self.n_samples,
                        );
                        Some(sample_ploidy_buf.as_slice())
                    }
                });

                let mut hard = vec![255u8; self.n_samples]; // 255 = missing
                decoder.decode_variant_hardcalls(in_idx, alt_ord, &mut hard, sample_ploidy)?;

                if let Some(kind) = haploidy_kind {
                    if !matches!(kind, HaploidyKind::Diploid) {
                        enforce_haploidy(&mut hard, &self.sex_by_sample, kind);
                    }
                }

                // Pack to bytes and store in cache.
                let mut block = vec![0u8; self.block_bytes];
                Self::pack_to_block(&mut block, &hard);

                let mut cache = self.cache.lock().unwrap();
                cache.put(out_idx, block.clone());

                let start = within_block;
                let end = start + to_copy;
                dst[written..written + to_copy].copy_from_slice(&block[start..end]);
            }

            written += to_copy;
            within_block = 0;
            out_idx += 1;
        }

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tiny LRU for packed blocks
////////////////////////////////////////////////////////////////////////////////////////////////////

struct BlockCache {
    cap: usize,
    // very small MRU list; simple Vec is fine for low capacity
    vals: VecDeque<(usize, Vec<u8>)>, // (out_idx, block)
}

impl BlockCache {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            vals: VecDeque::new(),
        }
    }

    fn get(&mut self, k: usize) -> Option<&Vec<u8>> {
        if let Some(pos) = self.vals.iter().position(|(kk, _)| *kk == k) {
            // move-to-back MRU
            let pair = self.vals.remove(pos).unwrap();
            self.vals.push_back(pair);
            return self.vals.back().map(|(_, v)| v);
        }
        None
    }

    fn put(&mut self, k: usize, v: Vec<u8>) {
        if let Some(pos) = self.vals.iter().position(|(kk, _)| *kk == k) {
            self.vals.remove(pos);
        } else if self.vals.len() == self.cap {
            let _ = self.vals.pop_front();
        }
        self.vals.push_back((k, v));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Minimal local ByteRangeSource for `.pgen` (seekable)
////////////////////////////////////////////////////////////////////////////////////////////////////

struct LocalFileByteRangeSource {
    file: Mutex<File>,
    len: u64,
}

impl LocalFileByteRangeSource {
    fn open(path: &Path) -> Result<Self, PipelineError> {
        let f = File::open(path)
            .map_err(|e| PipelineError::Io(format!("Opening {}: {e}", path.display())))?;
        let len = f
            .metadata()
            .map_err(|e| PipelineError::Io(format!("Metadata {}: {e}", path.display())))?
            .len();
        Ok(Self {
            file: Mutex::new(f),
            len,
        })
    }
}
impl ByteRangeSource for LocalFileByteRangeSource {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        if dst.is_empty() {
            return Ok(());
        }
        if offset.checked_add(dst.len() as u64).unwrap_or(u64::MAX) > self.len {
            return Err(ioerr("Attempted to read past end of local .pgen"));
        }
        let mut f = self.file.lock().unwrap();
        f.seek(SeekFrom::Start(offset))
            .map_err(|e| PipelineError::Io(e.to_string()))?;
        f.read_exact(dst)
            .map_err(|e| PipelineError::Io(e.to_string()))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PGEN decoder (spec-aligned subset)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PgenMode {
    Bed = 0x01,
    FixHard = 0x02,
    FixDosage = 0x03,
    FixPhDosage = 0x04,
    Var = 0x10,
    VarIgnorable = 0x11,
}

fn read_le_u32(src: &dyn ByteRangeSource, off: u64) -> Result<u32, PipelineError> {
    let mut buf = [0u8; 4];
    src.read_at(off, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_le_u64(src: &dyn ByteRangeSource, off: u64) -> Result<u64, PipelineError> {
    let mut buf = [0u8; 8];
    src.read_at(off, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_varint_from_source(
    src: &dyn ByteRangeSource,
    offset: &mut u64,
) -> Result<u64, PipelineError> {
    let mut out: u64 = 0;
    let mut shift = 0;
    loop {
        if *offset >= src.len() {
            return Err(ioerr("Unexpected EOF in header varint"));
        }
        let mut byte = [0u8; 1];
        src.read_at(*offset, &mut byte)?;
        *offset += 1;
        out |= ((byte[0] & 0x7f) as u64) << shift;
        if (byte[0] & 0x80) == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(ioerr("Header varint too large"));
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct PgenHeader {
    mode: PgenMode,
    m_variants: u32,
    n_samples: u32,
    fmt_byte: u8,
    block_offsets: Vec<u64>,
    rec_types: Vec<u8>,
    rec_lens: Vec<u32>,
}

impl PgenHeader {
    fn parse(src: &dyn ByteRangeSource) -> Result<Self, PipelineError> {
        if src.len() < 3 {
            return Err(ioerr("PGEN file too small"));
        }
        let mut magic = [0u8; 3];
        src.read_at(0, &mut magic)?;
        if magic[0] != 0x6c || magic[1] != 0x1b {
            return Err(ioerr("Not a PGEN (bad magic)"));
        }

        let mode = match magic[2] {
            0x01 => PgenMode::Bed,
            0x02 => PgenMode::FixHard,
            0x03 => PgenMode::FixDosage,
            0x04 => PgenMode::FixPhDosage,
            0x10 => PgenMode::Var,
            0x11 => PgenMode::VarIgnorable,
            0x20 | 0x21 => {
                return Err(ioerr(
                    "External index modes (0x20/0x21) unsupported; re-encode with `plink2 --pgen ... --make-pgen`",
                ));
            }
            other => {
                return Err(PipelineError::Io(format!(
                    "Unsupported PGEN mode 0x{other:02x}"
                )));
            }
        };

        if mode == PgenMode::Bed {
            return Ok(Self {
                mode,
                m_variants: 0,
                n_samples: 0,
                fmt_byte: 0,
                block_offsets: vec![],
                rec_types: vec![],
                rec_lens: vec![],
            });
        }

        let m_variants = read_le_u32(src, 3)?;
        let n_samples = read_le_u32(src, 7)?;

        if matches!(
            mode,
            PgenMode::FixHard | PgenMode::FixDosage | PgenMode::FixPhDosage
        ) {
            let mut b = [0u8; 1];
            src.read_at(11, &mut b)?;
            let fmt = b[0];
            return Ok(Self {
                mode,
                m_variants,
                n_samples,
                fmt_byte: fmt,
                block_offsets: vec![],
                rec_types: vec![],
                rec_lens: vec![],
            });
        }

        let fmt = {
            let mut b = [0u8; 1];
            src.read_at(11, &mut b)?;
            b[0]
        };
        let type_bits = if (fmt & 0x0f) <= 3 { 4 } else { 8 };
        let len_bytes = match fmt & 0x07 {
            0 | 4 => 1,
            1 | 5 => 2,
            2 | 6 => 3,
            3 | 7 => 4,
            _ => unreachable!(),
        };
        let ac_bytes = match (fmt >> 4) & 0x03 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 4,
            _ => unreachable!(),
        };
        let ref_flag_mode = (fmt >> 6) & 0x03;

        let blocks = ((m_variants as u64) + ((1u64 << 16) - 1)) >> 16;
        let mut block_offsets = Vec::with_capacity(blocks as usize);
        let mut off = 12u64;
        for _ in 0..blocks {
            block_offsets.push(read_le_u64(src, off)?);
            off += 8;
        }

        let mut rec_types = vec![0u8; m_variants as usize];
        let mut rec_lens = vec![0u32; m_variants as usize];

        let mut idx = 0usize;
        for _ in 0..blocks {
            let remain = (m_variants as usize) - idx;
            let cnt = remain.min(1 << 16);

            // types per block
            if type_bits == 4 {
                let nbytes = (cnt + 1) / 2;
                let mut buf = vec![0u8; nbytes];
                src.read_at(off, &mut buf)?;
                off += nbytes as u64;
                for (i, byte) in buf.into_iter().enumerate() {
                    let base = idx + 2 * i;
                    if base < idx + cnt {
                        rec_types[base] = byte & 0x0f;
                    }
                    if base + 1 < idx + cnt {
                        rec_types[base + 1] = (byte >> 4) & 0x0f;
                    }
                }
            } else {
                let nbytes = cnt;
                src.read_at(off, &mut rec_types[idx..idx + cnt])?;
                off += nbytes as u64;
            }

            if len_bytes > 0 {
                let nbytes = cnt * (len_bytes as usize);
                let mut buf = vec![0u8; nbytes];
                if nbytes > 0 {
                    src.read_at(off, &mut buf)?;
                }
                off += nbytes as u64;
                for i in 0..cnt {
                    let start = i * (len_bytes as usize);
                    let len = match len_bytes {
                        1 => u32::from(buf[start]),
                        2 => u32::from_le_bytes([buf[start], buf[start + 1], 0, 0]),
                        3 => u32::from_le_bytes([buf[start], buf[start + 1], buf[start + 2], 0]),
                        4 => u32::from_le_bytes([
                            buf[start],
                            buf[start + 1],
                            buf[start + 2],
                            buf[start + 3],
                        ]),
                        _ => unreachable!(),
                    };
                    rec_lens[idx + i] = len;
                }
            }

            if ac_bytes > 0 {
                let nbytes = cnt * (ac_bytes as usize);
                off += nbytes as u64;
            }
            if ref_flag_mode == 3 {
                let nbytes = (cnt + 7) / 8;
                off += nbytes as u64;
            }

            idx += cnt;
        }

        if mode == PgenMode::VarIgnorable {
            let mut ext_off = off;
            let header_flags = read_varint_from_source(src, &mut ext_off)?;
            let footer_flags = read_varint_from_source(src, &mut ext_off)?;

            if footer_flags != 0 {
                if ext_off
                    .checked_add(8)
                    .ok_or_else(|| ioerr("Footer offset overflow"))?
                    > src.len()
                {
                    return Err(ioerr("EOF reading footer offset"));
                }
                ext_off += 8;
            }

            let mut lengths = Vec::new();
            let mut mask = header_flags;
            while mask != 0 {
                if (mask & 1) != 0 {
                    lengths.push(read_varint_from_source(src, &mut ext_off)?);
                }
                mask >>= 1;
            }
            let sum: u64 = lengths.into_iter().try_fold(0u64, |acc, x| {
                acc.checked_add(x)
                    .ok_or_else(|| ioerr("Header extension length overflow"))
            })?;
            if ext_off
                .checked_add(sum)
                .ok_or_else(|| ioerr("Header extension overflow"))?
                > src.len()
            {
                return Err(ioerr("Header extensions overrun file"));
            }
        }

        Ok(Self {
            mode,
            m_variants,
            n_samples,
            fmt_byte: fmt,
            block_offsets,
            rec_types,
            rec_lens,
        })
    }
}

#[inline]
fn read_base128_varint(buf: &[u8], cursor: &mut usize) -> Result<u64, PipelineError> {
    let mut out: u64 = 0;
    let mut shift = 0;
    loop {
        if *cursor >= buf.len() {
            return Err(ioerr("Unexpected EOF in varint"));
        }
        let b = buf[*cursor];
        *cursor += 1;
        out |= ((b & 0x7f) as u64) << shift;
        if (b & 0x80) == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(ioerr("Varint too large"));
        }
    }
    Ok(out)
}

#[inline]
fn read_u24_le(buf: &[u8], cursor: &mut usize) -> Result<u32, PipelineError> {
    if *cursor + 3 > buf.len() {
        return Err(ioerr("EOF in u24"));
    }
    let v = u32::from_le_bytes([buf[*cursor], buf[*cursor + 1], buf[*cursor + 2], 0]);
    *cursor += 3;
    Ok(v)
}

#[inline]
fn read_bitarray_indices(
    buf: &[u8],
    cursor: &mut usize,
    nbits: usize,
) -> Result<Vec<usize>, PipelineError> {
    let nbytes = (nbits + 7) / 8;
    if *cursor + nbytes > buf.len() {
        return Err(ioerr("EOF in bitarray"));
    }
    let mut out = Vec::with_capacity(nbits.min(1024));
    for (j, &byte) in buf[*cursor..*cursor + nbytes].iter().enumerate() {
        for b in 0..8 {
            let bit = j * 8 + b;
            if bit >= nbits {
                break;
            }
            if (byte >> b) & 1 == 1 {
                out.push(bit);
            }
        }
    }
    *cursor += nbytes;
    Ok(out)
}

#[inline]
fn read_packed_fixed_width(
    buf: &[u8],
    cursor: &mut usize,
    width_bits: usize,
    count: usize,
) -> Result<Vec<u32>, PipelineError> {
    if width_bits == 0 {
        return Ok(vec![0; count]);
    }
    let total_bits = width_bits * count;
    let nbytes = (total_bits + 7) / 8;
    if *cursor + nbytes > buf.len() {
        return Err(ioerr("EOF in packed values"));
    }
    let slice = &buf[*cursor..*cursor + nbytes];
    let mut out = Vec::with_capacity(count);
    let mut bitpos = 0usize;
    for _ in 0..count {
        let mut acc = 0u32;
        for k in 0..width_bits {
            let bp = bitpos + k;
            let byte = slice[bp >> 3];
            let bit = (byte >> (bp & 7)) & 1;
            acc |= (bit as u32) << k;
        }
        out.push(acc);
        bitpos += width_bits;
    }
    *cursor += nbytes;
    Ok(out)
}

#[inline]
fn sample_id_bytes(n_samples: usize) -> usize {
    if n_samples <= (1 << 8) {
        1
    } else if n_samples <= (1 << 16) {
        2
    } else if n_samples <= (1 << 24) {
        3
    } else {
        4
    }
}

fn difflist_ids(
    buf: &[u8],
    cursor: &mut usize,
    n_samples: usize,
) -> Result<Vec<u32>, PipelineError> {
    let l = read_base128_varint(buf, cursor)? as usize;
    if l == 0 {
        return Ok(vec![]);
    }
    let g = (l + 63) / 64;
    let sid_bytes = sample_id_bytes(n_samples);

    let mut first_ids = Vec::with_capacity(g);
    for _ in 0..g {
        let v = match sid_bytes {
            1 => {
                if *cursor >= buf.len() {
                    return Err(ioerr("EOF in u8 first-ID"));
                }
                let v = buf[*cursor] as u32;
                *cursor += 1;
                v
            }
            2 => {
                if *cursor + 2 > buf.len() {
                    return Err(ioerr("EOF in u16 first-ID"));
                }
                let v = u16::from_le_bytes([buf[*cursor], buf[*cursor + 1]]) as u32;
                *cursor += 2;
                v
            }
            3 => read_u24_le(buf, cursor)?,
            _ => {
                if *cursor + 4 > buf.len() {
                    return Err(ioerr("EOF in u32 first-ID"));
                }
                let v = u32::from_le_bytes([
                    buf[*cursor],
                    buf[*cursor + 1],
                    buf[*cursor + 2],
                    buf[*cursor + 3],
                ]);
                *cursor += 4;
                v
            }
        };
        first_ids.push(v);
    }

    let mut group_delta_bytes = Vec::with_capacity(g.saturating_sub(1));
    if g > 1 {
        if *cursor + (g - 1) > buf.len() {
            return Err(ioerr("EOF in difflist group byte-lengths"));
        }
        for raw in &buf[*cursor..*cursor + g - 1] {
            group_delta_bytes.push((*raw as usize) + 63);
        }
        *cursor += g - 1;
    }

    let mut ids = Vec::with_capacity(l);
    let mut delta_cur = *cursor;

    for gi in 0..g {
        let group_elems = if gi < g - 1 { 64 } else { l - 64 * (g - 1) };
        if group_elems == 0 {
            return Err(ioerr("Empty difflist group"));
        }

        ids.push(first_ids[gi]);

        let start = delta_cur;
        for _ in 1..group_elems {
            let d = read_base128_varint(buf, &mut delta_cur)? as u32;
            let last = *ids.last().unwrap();
            ids.push(
                last.checked_add(d)
                    .ok_or_else(|| ioerr("Difflist delta overflow"))?,
            );
        }
        if gi < g - 1 {
            let used = delta_cur - start;
            let expected = group_delta_bytes[gi];
            if used != expected {
                return Err(ioerr("Difflist group byte-length mismatch"));
            }
        }
    }

    *cursor = delta_cur;
    if ids.len() != l {
        return Err(ioerr("Difflist decode count mismatch"));
    }
    Ok(ids)
}

fn difflist_pairs(
    buf: &[u8],
    cursor: &mut usize,
    n_samples: usize,
) -> Result<Vec<(u32, u8)>, PipelineError> {
    let l = read_base128_varint(buf, cursor)? as usize;
    if l == 0 {
        return Ok(vec![]);
    }
    let g = (l + 63) / 64;
    let sid_bytes = sample_id_bytes(n_samples);

    let mut first_ids = Vec::with_capacity(g);
    for _ in 0..g {
        let v = match sid_bytes {
            1 => {
                if *cursor >= buf.len() {
                    return Err(ioerr("EOF in u8 first-ID"));
                }
                let v = buf[*cursor] as u32;
                *cursor += 1;
                v
            }
            2 => {
                if *cursor + 2 > buf.len() {
                    return Err(ioerr("EOF in u16 first-ID"));
                }
                let v = u16::from_le_bytes([buf[*cursor], buf[*cursor + 1]]) as u32;
                *cursor += 2;
                v
            }
            3 => read_u24_le(buf, cursor)?,
            _ => {
                if *cursor + 4 > buf.len() {
                    return Err(ioerr("EOF in u32 first-ID"));
                }
                let v = u32::from_le_bytes([
                    buf[*cursor],
                    buf[*cursor + 1],
                    buf[*cursor + 2],
                    buf[*cursor + 3],
                ]);
                *cursor += 4;
                v
            }
        };
        first_ids.push(v);
    }

    let mut group_delta_bytes = Vec::with_capacity(g.saturating_sub(1));
    if g > 1 {
        if *cursor + (g - 1) > buf.len() {
            return Err(ioerr("EOF in difflist group byte-lengths"));
        }
        for raw in &buf[*cursor..*cursor + g - 1] {
            group_delta_bytes.push((*raw as usize) + 63);
        }
        *cursor += g - 1;
    }

    let vals_packed = (l + 3) / 4;
    if *cursor + vals_packed > buf.len() {
        return Err(ioerr("EOF in difflist values"));
    }
    let mut vals = Vec::with_capacity(l);
    for i in 0..vals_packed {
        let b = buf[*cursor + i];
        vals.push(b & 0b11);
        if vals.len() == l {
            break;
        }
        vals.push((b >> 2) & 0b11);
        if vals.len() == l {
            break;
        }
        vals.push((b >> 4) & 0b11);
        if vals.len() == l {
            break;
        }
        vals.push((b >> 6) & 0b11);
    }
    *cursor += vals_packed;

    let mut ids = Vec::with_capacity(l);
    let mut delta_cur = *cursor;

    for gi in 0..g {
        let group_elems = if gi < g - 1 { 64 } else { l - 64 * (g - 1) };
        if group_elems == 0 {
            return Err(ioerr("Empty difflist group"));
        }

        ids.push(first_ids[gi]);

        let start = delta_cur;
        for _ in 1..group_elems {
            let d = read_base128_varint(buf, &mut delta_cur)? as u32;
            let last = *ids.last().unwrap();
            ids.push(
                last.checked_add(d)
                    .ok_or_else(|| ioerr("Difflist delta overflow"))?,
            );
        }
        if gi < g - 1 {
            let used = delta_cur - start;
            let expected = group_delta_bytes[gi];
            if used != expected {
                return Err(ioerr("Difflist group byte-length mismatch"));
            }
        }
    }

    *cursor = delta_cur;
    if ids.len() != l || vals.len() != l {
        return Err(ioerr("Difflist decode count mismatch"));
    }
    Ok(ids
        .into_iter()
        .zip(vals.into_iter().map(|v| v as u8))
        .collect())
}

struct PgenDecoder {
    src: Arc<dyn ByteRangeSource>,
    hdr: PgenHeader,
    n: usize,
    scratch: Vec<u8>,
    anchor_cats: Option<Vec<u8>>,
    alt_counts: Vec<u16>,
}

impl PgenDecoder {
    fn new(
        src: Arc<dyn ByteRangeSource>,
        hdr: PgenHeader,
        n_samples_from_psam: usize,
        in_variants: usize,
        alt_counts: Vec<u16>,
    ) -> Result<Self, PipelineError> {
        match hdr.mode {
            PgenMode::Bed => return Err(ioerr("Mode 0x01 passthrough handled elsewhere")),
            PgenMode::FixDosage | PgenMode::FixPhDosage => {
                return Err(ioerr("Fixed-width dosage modes carry no hard-calls"));
            }
            _ => {}
        }

        if hdr.m_variants as usize != in_variants {
            return Err(PipelineError::Io(format!(
                "Variant count mismatch: .pgen header {0} vs .pvar {1}",
                hdr.m_variants, in_variants
            )));
        }

        if hdr.n_samples as usize != n_samples_from_psam && hdr.n_samples != 0 {
            return Err(PipelineError::Io(format!(
                "Sample count mismatch: .pgen {0} vs .psam {1}",
                hdr.n_samples, n_samples_from_psam
            )));
        }

        Ok(Self {
            src,
            hdr,
            n: n_samples_from_psam,
            scratch: Vec::new(),
            anchor_cats: None,
            alt_counts,
        })
    }

    fn record_offset_len(&self, idx: usize) -> Result<(u64, usize, u8), PipelineError> {
        match self.hdr.mode {
            PgenMode::FixHard => {
                let rec_len = (self.n + 3) / 4;
                let ref_flag_mode = (self.hdr.fmt_byte >> 6) & 0x03;
                let mut base = 12u64 + 1; // header + format byte
                if ref_flag_mode == 3 {
                    base = base
                        .checked_add(((self.hdr.m_variants as u64) + 7) / 8)
                        .ok_or_else(|| ioerr("Header overflow"))?;
                }
                Ok((base + (idx as u64) * (rec_len as u64), rec_len, 0))
            }
            PgenMode::Var | PgenMode::VarIgnorable => {
                let block_idx = idx >> 16;
                let block_off = *self
                    .hdr
                    .block_offsets
                    .get(block_idx)
                    .ok_or_else(|| ioerr("Missing block offset"))?;
                let mut off = block_off;
                let mut cursor = block_idx << 16;
                while cursor < idx {
                    off += *self
                        .hdr
                        .rec_lens
                        .get(cursor)
                        .ok_or_else(|| ioerr("Missing rec_len"))? as u64;
                    cursor += 1;
                }
                let rec_len = *self
                    .hdr
                    .rec_lens
                    .get(idx)
                    .ok_or_else(|| ioerr("Missing rec_len"))?
                    as usize;
                let rec_ty = *self
                    .hdr
                    .rec_types
                    .get(idx)
                    .ok_or_else(|| ioerr("Missing rec_type"))?;
                Ok((off, rec_len, rec_ty))
            }
            _ => Err(ioerr("Unsupported PGEN mode")),
        }
    }

    fn decode_variant_hardcalls(
        &mut self,
        in_idx: u32,
        alt_ord_1b: u16,
        dst: &mut [u8],
        sample_ploidy: Option<&[u8]>,
    ) -> Result<(), PipelineError> {
        if dst.len() != self.n {
            return Err(ioerr("Hardcall buffer length must equal sample count"));
        }
        if let Some(ploidy) = sample_ploidy {
            if ploidy.len() != self.n {
                return Err(ioerr("Sample ploidy length mismatch"));
            }
        }
        let idx = in_idx as usize;
        if idx >= self.hdr.m_variants as usize {
            return Err(ioerr("Variant index out of bounds"));
        }

        let (off, len, rec_ty) = self.record_offset_len(idx)?;
        if self.scratch.len() < len {
            self.scratch.resize(len, 0);
        }
        self.src.read_at(off, &mut self.scratch[..len])?;
        let buf = &self.scratch[..len];
        let mut cursor = 0usize;

        let mut cats = vec![3u8; self.n];
        let main_kind = rec_ty & 0x07;
        match main_kind {
            0 => {
                let need = (self.n + 3) / 4;
                if need > len {
                    return Err(ioerr("Truncated type-0 main track"));
                }
                unpack_pgen2bit_to_categories(&buf[cursor..cursor + need], &mut cats, self.n);
                cursor += need;
            }
            1 => {
                if cursor >= len {
                    return Err(ioerr("Truncated type-1 header byte"));
                }
                let pair = buf[cursor];
                cursor += 1;
                let (low, high) = match pair {
                    1 => (0u8, 1),
                    2 => (0, 2),
                    3 => (0, 3),
                    5 => (1, 2),
                    6 => (1, 3),
                    9 => (2, 3),
                    _ => return Err(ioerr("Invalid 1-bit pair code")),
                };
                let idxs = read_bitarray_indices(buf, &mut cursor, self.n)?;
                for i in 0..self.n {
                    cats[i] = low;
                }
                for bit in idxs {
                    if bit < self.n {
                        cats[bit] = high;
                    }
                }
                let pairs = difflist_pairs(buf, &mut cursor, self.n)?;
                for (sid, val) in pairs {
                    if (sid as usize) < self.n {
                        cats[sid as usize] = val;
                    }
                }
            }
            2 | 3 => {
                if (idx & 0xffff) == 0 {
                    return Err(ioerr("LD-compressed record at block start"));
                }
                let anchor = self
                    .anchor_cats
                    .as_ref()
                    .ok_or_else(|| ioerr("Missing LD anchor"))?
                    .clone();
                let mut base = anchor;
                let pairs = difflist_pairs(buf, &mut cursor, self.n)?;
                for (sid, val) in pairs {
                    if (sid as usize) < self.n {
                        base[sid as usize] = val;
                    }
                }
                if main_kind == 3 {
                    for c in &mut base {
                        if *c == 0 {
                            *c = 2;
                        } else if *c == 2 {
                            *c = 0;
                        }
                    }
                }
                cats.copy_from_slice(&base);
            }
            4 | 6 | 7 => {
                let x = match main_kind {
                    4 => 0u8,
                    6 => 2,
                    7 => 3,
                    _ => unreachable!(),
                };
                for i in 0..self.n {
                    cats[i] = x;
                }
                let pairs = difflist_pairs(buf, &mut cursor, self.n)?;
                for (sid, val) in pairs {
                    if (sid as usize) < self.n {
                        cats[sid as usize] = val;
                    }
                }
            }
            _ => {
                return Err(PipelineError::Io(format!(
                    "Unsupported main-track type {}",
                    main_kind
                )));
            }
        }

        let has_multiallelic = (rec_ty & 0b0000_1000) != 0;
        let alt_count = self.alt_counts.get(idx).copied().unwrap_or(0);
        let mut a1dosage = vec![255u8; self.n];
        if has_multiallelic {
            apply_multiallelic_and_project(
                buf,
                &mut cursor,
                self.n,
                &mut cats,
                alt_count,
                alt_ord_1b,
                &mut a1dosage,
            )?;
        } else {
            cats_to_a1dosage(&mut a1dosage, &cats);
        }

        if (rec_ty & 0b0001_0000) != 0 {
            if cursor >= len {
                return Err(PipelineError::Io(format!(
                    "EOF in phase header (variant #{idx})"
                )));
            }
            let start = cursor;
            let mut bit_cursor = 0usize;
            let phasepresent = (buf[start] & 1) == 1;
            bit_cursor += 1;
            let h = cats.iter().filter(|&&c| c == 1).count();
            let mut phased_count = h;
            if phasepresent {
                let mut present_count = 0usize;
                for _ in 0..h {
                    let bit_idx = bit_cursor;
                    let byte_idx = start + (bit_idx >> 3);
                    if byte_idx >= len {
                        return Err(PipelineError::Io(format!(
                            "EOF in phase presence (variant #{idx})"
                        )));
                    }
                    let byte = buf[byte_idx];
                    if (byte >> (bit_idx & 7)) & 1 == 1 {
                        present_count += 1;
                    }
                    bit_cursor += 1;
                }
                phased_count = present_count;
                if bit_cursor & 7 != 0 {
                    bit_cursor += 8 - (bit_cursor & 7);
                }
            }
            for _ in 0..phased_count {
                let bit_idx = bit_cursor;
                let byte_idx = start + (bit_idx >> 3);
                if byte_idx >= len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phase info (variant #{idx})"
                    )));
                }
                bit_cursor += 1;
            }
            let bytes_needed = (bit_cursor + 7) / 8;
            if start + bytes_needed > len {
                return Err(PipelineError::Io(format!(
                    "EOF in phase track (variant #{idx})"
                )));
            }
            cursor = start + bytes_needed;
        }

        let has_dosage = (rec_ty & 0b0110_0000) != 0;

        let mut dosage_entries = 0usize;
        if has_dosage {
            // Multiallelic dosage tracks (#5-#10) are intentionally ignored for
            // alternate alleles beyond the first; keep hard-call derived values
            // (which may remain missing).
            let decode_dosage = alt_count <= 1 || alt_ord_1b == 1;
            let b5 = (rec_ty & 0b0010_0000) != 0;
            let b6 = (rec_ty & 0b0100_0000) != 0;

            if b5 && !b6 {
                let ids = difflist_ids(buf, &mut cursor, self.n)?;
                let cnt = ids.len();
                let need = cnt * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in dosage values (variant #{idx})"
                    )));
                }
                if decode_dosage {
                    for (i, &sid) in ids.iter().enumerate() {
                        if (sid as usize) < self.n {
                            let v =
                                u16::from_le_bytes([buf[cursor + 2 * i], buf[cursor + 2 * i + 1]]);
                            if a1dosage[sid as usize] == 255 && v != 65535 {
                                let pl = sample_ploidy
                                    .and_then(|p| p.get(sid as usize))
                                    .copied()
                                    .unwrap_or(2);
                                let hc = u16_to_hardcall_biallelic(v, pl);
                                if hc != 255 {
                                    a1dosage[sid as usize] = hc;
                                }
                            }
                        }
                    }
                }
                cursor += need;
                dosage_entries = cnt;
            } else if !b5 && b6 {
                let need = self.n * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in dense dosage values (variant #{idx})"
                    )));
                }
                if decode_dosage {
                    for s in 0..self.n {
                        let v = u16::from_le_bytes([buf[cursor + 2 * s], buf[cursor + 2 * s + 1]]);
                        if a1dosage[s] == 255 && v != 65535 {
                            let pl = sample_ploidy.and_then(|p| p.get(s)).copied().unwrap_or(2);
                            let hc = u16_to_hardcall_biallelic(v, pl);
                            if hc != 255 {
                                a1dosage[s] = hc;
                            }
                        }
                    }
                }
                cursor += need;
                dosage_entries = self.n;
            } else {
                let present = read_bitarray_indices(buf, &mut cursor, self.n)?;
                let cnt = present.len();
                let need = cnt * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in sparse dosage values (variant #{idx})"
                    )));
                }
                if decode_dosage {
                    for (i, &s) in present.iter().enumerate() {
                        if s < self.n {
                            let v =
                                u16::from_le_bytes([buf[cursor + 2 * i], buf[cursor + 2 * i + 1]]);
                            if a1dosage[s] == 255 && v != 65535 {
                                let pl = sample_ploidy.and_then(|p| p.get(s)).copied().unwrap_or(2);
                                let hc = u16_to_hardcall_biallelic(v, pl);
                                if hc != 255 {
                                    a1dosage[s] = hc;
                                }
                            }
                        }
                    }
                }
                cursor += need;
                dosage_entries = cnt;
            }
        }

        if (rec_ty & 0b1000_0000) != 0 {
            if !has_dosage {
                return Err(PipelineError::Io(format!(
                    "Phased dosage track present without dosage (variant #{idx})"
                )));
            }
            let b5 = (rec_ty & 0b0010_0000) != 0;
            let b6 = (rec_ty & 0b0100_0000) != 0;
            if !b5 && b6 {
                let need = self.n * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phased dense dosage values (variant #{idx})"
                    )));
                }
                cursor += need;
            } else {
                let d = dosage_entries;
                let nbytes = (d + 7) / 8;
                if cursor + nbytes > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phased dosage presence (variant #{idx})"
                    )));
                }
                let mut present_count = 0usize;
                for bit in 0..d {
                    let byte = buf[cursor + (bit >> 3)];
                    if (byte >> (bit & 7)) & 1 == 1 {
                        present_count += 1;
                    }
                }
                cursor += nbytes;
                let need = present_count * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phased dosage values (variant #{idx})"
                    )));
                }
                cursor += need;
            }
        }

        match len.checked_sub(cursor) {
            Some(0) => {}
            Some(remaining) => {
                return Err(PipelineError::Io(format!(
                    "Trailing data ({remaining} bytes) after variant #{idx}"
                )));
            }
            None => {
                return Err(PipelineError::Io(format!(
                    "Cursor advanced beyond end of record for variant #{idx}"
                )));
            }
        }

        if !matches!(main_kind, 2 | 3) {
            self.anchor_cats = Some(cats.clone());
        }

        dst.copy_from_slice(&a1dosage);
        Ok(())
    }
}

fn unpack_pgen2bit_to_categories(block: &[u8], dst: &mut [u8], n: usize) {
    let mut i = 0usize;
    for &byte in block {
        for shift in 0..4 {
            if i >= n {
                return;
            }
            dst[i] = (byte >> (2 * shift)) & 0b11;
            i += 1;
        }
    }
}

#[inline]
fn cats_to_a1dosage(dst: &mut [u8], cats: &[u8]) {
    for (d, c) in dst.iter_mut().zip(cats) {
        *d = match *c {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 255,
        };
    }
}

#[inline]
fn collect_cat_ids(cats: &[u8], cat: u8) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, &c) in cats.iter().enumerate() {
        if c == cat {
            out.push(i as u32);
        }
    }
    out
}

fn apply_multiallelic_and_project(
    record: &[u8],
    cursor: &mut usize,
    n: usize,
    cats: &mut [u8],
    alt_count: u16,
    alt_ord_1b: u16,
    out: &mut [u8],
) -> Result<(), PipelineError> {
    if alt_count <= 1 {
        cats_to_a1dosage(out, cats);
        return Ok(());
    }

    if *cursor >= record.len() {
        return Err(ioerr("EOF before multiallelic patch header"));
    }
    let fmt_byte = record[*cursor];
    *cursor += 1;
    let cat1_fmt = (fmt_byte & 0x0f) as u8;
    let cat2_fmt = (fmt_byte >> 4) & 0x0f;

    let cat1_ids = collect_cat_ids(cats, 1);
    let cat2_ids = collect_cat_ids(cats, 2);

    let mut cat1_override: Vec<(u32, u16)> = Vec::new();
    if cat1_fmt != 15 {
        match cat1_fmt {
            0 => {
                let set_indices = read_bitarray_indices(record, cursor, cat1_ids.len())?;
                let k = set_indices.len();
                let width = match alt_count {
                    2 => 0,
                    3 => 1,
                    4..=5 => 2,
                    6..=17 => 4,
                    18..=257 => 8,
                    258..=65535 => 16,
                    _ => 24,
                } as usize;
                let vals = read_packed_fixed_width(record, cursor, width, k)?;
                for (idx_in_list, v) in set_indices.into_iter().zip(vals.into_iter()) {
                    let sid = cat1_ids[idx_in_list];
                    let altj = if width == 0 { 2 } else { (v as u16) + 2 };
                    cat1_override.push((sid, altj));
                }
            }
            1 => {
                let sids = difflist_ids(record, cursor, n)?;
                let k = sids.len();
                let width = match alt_count {
                    2 => 0,
                    3 => 1,
                    4..=5 => 2,
                    6..=17 => 4,
                    18..=257 => 8,
                    258..=65535 => 16,
                    _ => 24,
                } as usize;
                let vals = read_packed_fixed_width(record, cursor, width, k)?;
                for (sid, v) in sids.into_iter().zip(vals.into_iter()) {
                    let altj = if width == 0 { 2 } else { (v as u16) + 2 };
                    cat1_override.push((sid, altj));
                }
            }
            _ => return Err(ioerr("Unsupported multiallelic cat1 patch format")),
        }
    }

    let mut cat2_override: Vec<(u32, (u16, u16))> = Vec::new();
    if cat2_fmt != 15 {
        match cat2_fmt {
            0 => {
                let set_indices = read_bitarray_indices(record, cursor, cat2_ids.len())?;
                let k = set_indices.len();
                if alt_count == 2 {
                    let hom2_flags = read_bitarray_indices(record, cursor, k)?;
                    let mut is_hom2 = vec![false; k];
                    for pos in hom2_flags {
                        if pos < k {
                            is_hom2[pos] = true;
                        }
                    }
                    for (flag, idx_in_list) in is_hom2.into_iter().zip(set_indices.into_iter()) {
                        let sid = cat2_ids[idx_in_list];
                        let pair = if flag { (2, 2) } else { (1, 2) };
                        cat2_override.push((sid, pair));
                    }
                } else {
                    let width = match alt_count {
                        3..=4 => 2,
                        5..=16 => 4,
                        17..=256 => 8,
                        257..=65535 => 16,
                        _ => 24,
                    } as usize;
                    let vals = read_packed_fixed_width(record, cursor, width, 2 * k)?;
                    for i in 0..k {
                        let sid = cat2_ids[set_indices[i]];
                        let lo = (vals[2 * i] as u16) + 1;
                        let hi = (vals[2 * i + 1] as u16) + 1;
                        let pair = if lo <= hi { (lo, hi) } else { (hi, lo) };
                        cat2_override.push((sid, pair));
                    }
                }
            }
            1 => {
                let sids = difflist_ids(record, cursor, n)?;
                let k = sids.len();
                if alt_count == 2 {
                    let hom2_flags = read_bitarray_indices(record, cursor, k)?;
                    let mut is_hom2 = vec![false; k];
                    for pos in hom2_flags {
                        if pos < k {
                            is_hom2[pos] = true;
                        }
                    }
                    for (flag, sid) in is_hom2.into_iter().zip(sids.into_iter()) {
                        let pair = if flag { (2, 2) } else { (1, 2) };
                        cat2_override.push((sid, pair));
                    }
                } else {
                    let width = match alt_count {
                        3..=4 => 2,
                        5..=16 => 4,
                        17..=256 => 8,
                        257..=65535 => 16,
                        _ => 24,
                    } as usize;
                    let vals = read_packed_fixed_width(record, cursor, width, 2 * k)?;
                    for i in 0..k {
                        let sid = sids[i];
                        let lo = (vals[2 * i] as u16) + 1;
                        let hi = (vals[2 * i + 1] as u16) + 1;
                        let pair = if lo <= hi { (lo, hi) } else { (hi, lo) };
                        cat2_override.push((sid, pair));
                    }
                }
            }
            _ => return Err(ioerr("Unsupported multiallelic cat2 patch format")),
        }
    }

    cat1_override.sort_unstable_by_key(|x| x.0);
    cat2_override.sort_unstable_by_key(|x| x.0);

    for i in 0..n {
        let c = cats[i];
        out[i] = match c {
            0 => 0,
            3 => 255,
            1 => {
                let mut altj = 1u16;
                if let Ok(pos) = cat1_override.binary_search_by_key(&(i as u32), |(sid, _)| *sid) {
                    altj = cat1_override[pos].1;
                }
                if altj == alt_ord_1b { 1 } else { 0 }
            }
            2 => {
                let mut pair = (1u16, 1u16);
                if let Ok(pos) = cat2_override.binary_search_by_key(&(i as u32), |(sid, _)| *sid) {
                    pair = cat2_override[pos].1;
                }
                let mut dose = 0u8;
                if pair.0 == alt_ord_1b {
                    dose += 1;
                }
                if pair.1 == alt_ord_1b {
                    dose += 1;
                }
                dose
            }
            _ => 255,
        };
    }

    Ok(())
}

fn u16_to_hardcall_biallelic(v: u16, ploidy: u8) -> u8 {
    if v == 65535 || ploidy == 0 {
        return 255;
    }
    if ploidy <= 1 {
        let ds = (v as f32) * (1.0 / 32768.0) * 1.0;
        let candidates = [0.0f32, 1.0];
        let mut best = 255u8;
        let mut best_d = f32::INFINITY;
        for (i, &c) in candidates.iter().enumerate() {
            let d = (ds - c).abs();
            if d < best_d {
                best_d = d;
                best = i as u8;
            }
        }
        if best_d <= 0.10 {
            match best {
                0 => 0,
                1 => 2,
                _ => 255,
            }
        } else {
            255
        }
    } else {
        let ds = (v as f32) * (1.0 / 32768.0) * 2.0;
        let candidates = [0.0f32, 1.0, 2.0];
        let mut best = 255u8;
        let mut best_d = f32::INFINITY;
        for (i, &c) in candidates.iter().enumerate() {
            let d = (ds - c).abs();
            if d < best_d {
                best_d = d;
                best = i as u8;
            }
        }
        if best_d <= 0.10 { best } else { 255 }
    }
}

#[cfg(test)]
fn unpack_plink1_block(block: &[u8], dst: &mut [u8], n: usize) {
    let mut i = 0usize;
    for &byte in block {
        for shift in 0..4 {
            if i >= n {
                return;
            }
            let code = (byte >> (2 * shift)) & 0b11;
            dst[i] = match code {
                0b00 => 2,
                0b10 => 1,
                0b11 => 0,
                _ => 255,
            };
            i += 1;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (subset) – validates packing and basic decode scaffolding behavior
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;
    use std::sync::Arc;

    struct VecSource {
        data: Vec<u8>,
    }

    impl VecSource {
        fn new(data: Vec<u8>) -> Self {
            Self { data }
        }
    }

    impl ByteRangeSource for VecSource {
        fn len(&self) -> u64 {
            self.data.len() as u64
        }

        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
            let off = usize::try_from(offset).map_err(|_| ioerr("Offset too large"))?;
            let end = off + dst.len();
            if end > self.data.len() {
                return Err(ioerr("Read past end"));
            }
            dst.copy_from_slice(&self.data[off..end]);
            Ok(())
        }
    }

    fn encode_varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
        out
    }

    fn pack_twobit_values(values: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                byte |= (val & 0b11) << (2 * i);
            }
            out.push(byte);
        }
        out
    }

    fn push_sid(buf: &mut Vec<u8>, sid: u32, bytes: usize) {
        let le = sid.to_le_bytes();
        buf.extend_from_slice(&le[..bytes]);
    }

    #[test]
    fn pack_contract_smoke() {
        let hard = [0u8, 1, 2, 255, 0, 0, 1, 2, 255, 255];
        let mut block = vec![0u8; (hard.len() + 3) / 4];
        VirtualBed::pack_to_block(&mut block, &hard);
        assert_eq!(block[0], 0x4B);
        assert_eq!(block[1], 0x2F);
        assert_eq!(block[2], 0x55);

        let mut round = vec![0u8; hard.len()];
        unpack_plink1_block(&block, &mut round, hard.len());
        assert_eq!(&hard, &round[..]);
    }

    #[test]
    fn dosage_u16_rounding() {
        let vals = [
            (((0.05 / 2.0) * 32768.0) as u16, 0u8),
            (((1.00 / 2.0) * 32768.0) as u16, 1u8),
            (((1.09 / 2.0) * 32768.0) as u16, 1u8),
            (((2.00 / 2.0) * 32768.0) as u16, 2u8),
        ];
        for (v, expect) in vals {
            assert_eq!(u16_to_hardcall_biallelic(v, 2), expect);
        }

        let hap_vals = [
            (((0.02 / 1.0) * 32768.0) as u16, 0u8),
            (((0.50 / 1.0) * 32768.0) as u16, 255u8),
            (((0.95 / 1.0) * 32768.0) as u16, 2u8),
        ];
        for (v, expect) in hap_vals {
            assert_eq!(u16_to_hardcall_biallelic(v, 1), expect);
        }
    }

    #[test]
    fn difflist_two_group_round_trip() {
        let n_samples = 2_000_000usize;
        let sid_bytes = sample_id_bytes(n_samples);
        assert_eq!(sid_bytes, 3);

        let mut group0_ids = Vec::with_capacity(64);
        let mut group0_deltas = Vec::with_capacity(63);
        let mut cur = 1_000u32;
        group0_ids.push(cur);
        let pattern0 = [1u32, 200, 20_000, 2, 3, 150, 4_000];
        for i in 0..63 {
            let delta = pattern0[i % pattern0.len()];
            cur += delta;
            group0_ids.push(cur);
            group0_deltas.push(delta);
        }

        let mut group1_ids = Vec::with_capacity(15);
        let mut group1_deltas = Vec::with_capacity(14);
        cur = 1_200_000u32;
        group1_ids.push(cur);
        let pattern1 = [
            2u32, 5_000, 180_000, 1, 2, 7_000, 3, 1, 400, 80_000, 2, 1, 1, 2,
        ];
        for &delta in &pattern1 {
            cur += delta;
            group1_ids.push(cur);
            group1_deltas.push(delta);
        }

        assert_eq!(group0_ids.len(), 64);
        assert_eq!(group1_ids.len(), 15);
        assert!(group0_ids.last().unwrap() < group1_ids.first().unwrap());
        assert!(*group1_ids.last().unwrap() < n_samples as u32);

        let expected_ids: Vec<u32> = group0_ids
            .iter()
            .chain(group1_ids.iter())
            .copied()
            .collect();

        let delta_bytes_g0: Vec<u8> = group0_deltas
            .iter()
            .flat_map(|&d| encode_varint(d as u64))
            .collect();
        let delta_bytes_g1: Vec<u8> = group1_deltas
            .iter()
            .flat_map(|&d| encode_varint(d as u64))
            .collect();
        assert!(delta_bytes_g0.len() > 63);
        let sentinel = u8::try_from(delta_bytes_g0.len() - 63).unwrap();

        let mut buf_ids = Vec::new();
        buf_ids.extend_from_slice(&encode_varint(expected_ids.len() as u64));
        push_sid(&mut buf_ids, group0_ids[0], sid_bytes);
        push_sid(&mut buf_ids, group1_ids[0], sid_bytes);
        buf_ids.push(sentinel);
        buf_ids.extend_from_slice(&delta_bytes_g0);
        buf_ids.extend_from_slice(&delta_bytes_g1);

        let mut cursor = 0usize;
        let decoded_ids = difflist_ids(&buf_ids, &mut cursor, n_samples).unwrap();
        assert_eq!(decoded_ids, expected_ids);
        assert_eq!(cursor, buf_ids.len());

        let expected_vals: Vec<u8> = (0..expected_ids.len()).map(|i| (i as u8) & 0b11).collect();
        let mut buf_pairs = Vec::new();
        buf_pairs.extend_from_slice(&encode_varint(expected_ids.len() as u64));
        push_sid(&mut buf_pairs, group0_ids[0], sid_bytes);
        push_sid(&mut buf_pairs, group1_ids[0], sid_bytes);
        buf_pairs.push(sentinel);
        buf_pairs.extend_from_slice(&pack_twobit_values(&expected_vals));
        buf_pairs.extend_from_slice(&delta_bytes_g0);
        buf_pairs.extend_from_slice(&delta_bytes_g1);

        let mut cursor_pairs = 0usize;
        let decoded_pairs = difflist_pairs(&buf_pairs, &mut cursor_pairs, n_samples).unwrap();
        let (ids_again, vals_again): (Vec<_>, Vec<_>) = decoded_pairs.into_iter().unzip();
        assert_eq!(ids_again, expected_ids);
        assert_eq!(vals_again, expected_vals);
        assert_eq!(cursor_pairs, buf_pairs.len());
    }

    #[test]
    fn type3_ld_record_inverts_after_patch() {
        let n = 8usize;
        let anchor_cats = [0u8, 1, 2, 0, 2, 1, 3, 2];
        let rec0 = pack_twobit_values(&anchor_cats);
        assert_eq!(rec0.len(), 2);

        let mut rec1 = Vec::new();
        let difflist_vals = [2u8, 0, 2];
        rec1.extend_from_slice(&encode_varint(difflist_vals.len() as u64));
        rec1.push(0); // first sample ID
        rec1.extend_from_slice(&pack_twobit_values(&difflist_vals));
        rec1.extend_from_slice(&encode_varint(2));
        rec1.extend_from_slice(&encode_varint(3));
        assert_eq!(rec1.len(), 5);

        let mut data = Vec::new();
        data.extend_from_slice(&rec0);
        data.extend_from_slice(&rec1);

        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(data));
        let hdr = PgenHeader {
            mode: PgenMode::Var,
            m_variants: 2,
            n_samples: n as u32,
            fmt_byte: 0,
            block_offsets: vec![0],
            rec_types: vec![0, 3],
            rec_lens: vec![rec0.len() as u32, rec1.len() as u32],
        };
        let mut decoder = PgenDecoder {
            src,
            hdr,
            n,
            scratch: Vec::new(),
            anchor_cats: None,
            alt_counts: vec![1, 1],
        };

        let mut out0 = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(0, 1, &mut out0, None)
            .unwrap();
        assert_eq!(out0, vec![0, 1, 2, 0, 2, 1, 255, 2]);

        let mut out1 = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(1, 1, &mut out1, None)
            .unwrap();
        assert_eq!(out1, vec![0, 1, 2, 2, 0, 0, 255, 0]);

        assert_eq!(decoder.anchor_cats.as_ref().unwrap(), &anchor_cats.to_vec());
    }

    #[test]
    fn multiallelic_alt1_patches_apply() {
        let record = vec![0u8, 0x01, 0x01, 0x01, 0x0A];
        let mut cats = vec![1u8, 2, 2];
        let mut out = vec![255u8; 3];
        let mut cursor = 0usize;
        apply_multiallelic_and_project(&record, &mut cursor, 3, &mut cats, 3, 1, &mut out).unwrap();
        assert_eq!(out, vec![0, 0, 2]);
        assert_eq!(cursor, record.len());
    }

    #[test]
    fn fixed_width_offsets_include_header_components() {
        let n_samples = 10usize;
        let m_variants = 10u32;
        let rec_len = (n_samples + 3) / 4;

        let hdr_plain = PgenHeader {
            mode: PgenMode::FixHard,
            m_variants,
            n_samples: n_samples as u32,
            fmt_byte: 0,
            block_offsets: vec![],
            rec_types: vec![],
            rec_lens: vec![],
        };
        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(vec![]));
        let decoder_plain = PgenDecoder {
            src: Arc::clone(&src),
            hdr: hdr_plain,
            n: n_samples,
            scratch: Vec::new(),
            anchor_cats: None,
            alt_counts: vec![0; m_variants as usize],
        };
        let (off0, len0, ty0) = decoder_plain.record_offset_len(0).unwrap();
        assert_eq!(off0, 12 + 1);
        assert_eq!(len0, rec_len);
        assert_eq!(ty0, 0);
        let (off1, _, _) = decoder_plain.record_offset_len(1).unwrap();
        assert_eq!(off1, 12 + 1 + rec_len as u64);

        let hdr_ref = PgenHeader {
            mode: PgenMode::FixHard,
            m_variants,
            n_samples: n_samples as u32,
            fmt_byte: 0b1100_0000,
            block_offsets: vec![],
            rec_types: vec![],
            rec_lens: vec![],
        };
        let decoder_ref = PgenDecoder {
            src,
            hdr: hdr_ref,
            n: n_samples,
            scratch: Vec::new(),
            anchor_cats: None,
            alt_counts: vec![0; m_variants as usize],
        };
        let (off0_ref, len_ref, _) = decoder_ref.record_offset_len(0).unwrap();
        let expected_base = 12 + 1 + ((m_variants as u64 + 7) / 8);
        assert_eq!(off0_ref, expected_base);
        assert_eq!(len_ref, rec_len);
        let (off2_ref, _, _) = decoder_ref.record_offset_len(2).unwrap();
        assert_eq!(off2_ref, expected_base + (rec_len as u64) * 2);
    }

    #[test]
    fn parse_fixhard_reads_fmt_byte() {
        let m_variants = 3u32;
        let n_samples = 8u32;
        let fmt = 0b1100_0000u8;
        let mut data = vec![0u8; 12];
        data[0] = 0x6c;
        data[1] = 0x1b;
        data[2] = 0x02;
        data[3..7].copy_from_slice(&m_variants.to_le_bytes());
        data[7..11].copy_from_slice(&n_samples.to_le_bytes());
        data[11] = fmt;
        let src = VecSource::new(data);
        let header = PgenHeader::parse(&src).unwrap();
        assert_eq!(header.mode, PgenMode::FixHard);
        assert_eq!(header.m_variants, m_variants);
        assert_eq!(header.n_samples, n_samples);
        assert_eq!(header.fmt_byte, fmt);
    }

    #[test]
    fn fam_row_uses_sid_when_iid_missing() {
        let fields = ["", "unused", "sid123", "1"];
        let cols = PsamColumns {
            fid_idx: Some(0),
            iid_idx: None,
            pat_idx: None,
            mat_idx: None,
            sex_idx: Some(3),
            pheno_idx: None,
            pheno1_idx: None,
            sid_idx: Some(2),
        };
        let fam = FamRow::from_fields(&fields, &cols);
        assert_eq!(fam.iid, "sid123");
        assert_eq!(fam.fid, "sid123");
        assert_eq!(fam.sex, "1");
    }

    #[test]
    fn fam_row_defaults_fid_to_resolved_iid() {
        let fields = ["iid789", ""];
        let cols = PsamColumns {
            fid_idx: Some(1),
            iid_idx: Some(0),
            ..PsamColumns::default()
        };
        let fam = FamRow::from_fields(&fields, &cols);
        assert_eq!(fam.iid, "iid789");
        assert_eq!(fam.fid, "iid789");
    }

    #[test]
    fn parse_sex_token_supports_common_words() {
        assert_eq!(parse_sex_token("male"), 1);
        assert_eq!(parse_sex_token("FEMALE"), 2);
        assert_eq!(parse_sex_token("Unknown"), 0);
        assert_eq!(parse_sex_token("UNK"), 0);
    }

    #[test]
    fn coerce_pheno_token_handles_missing_values() {
        assert_eq!(coerce_pheno_token("   "), "-9");
        assert_eq!(coerce_pheno_token("NaN"), "-9");
        assert_eq!(coerce_pheno_token("1.5"), "1.5");
        assert_eq!(coerce_pheno_token("nonsense"), "-9");
    }
}
