# Terms module

The `terms` crate provides reusable utilities for inferring sample-level
metadata terms from genotype data. It currently powers the `gnomon terms`
command's sex inference pipeline and exposes convenience helpers for writing the
results alongside the source dataset.

## CLI entry point: `gnomon terms`

```
gnomon terms --sex <GENOTYPE_PATH>
```

| Flag | Required | Purpose |
| --- | --- | --- |
| `<GENOTYPE_PATH>` | ✅ | Path or URI identifying the genotype dataset. The loader accepts PLINK `.bed/.bim/.fam` trios, directories of per-chromosome trios, and VCF/BCF files that share the same basename. Remote objects are supported anywhere the standard gnomon genotype I/O layer can reach them. |
| `--sex` | ✅ | Enables sex inference. Additional term inference modes will appear behind their own flags as they are implemented. |

Running the command prints high-level progress and writes a tab-delimited
`sex.tsv` file next to the genotype input (for example, `data/ukb.sex.tsv` when
pointing at `data/ukb.bed`). The command fails fast if no inference mode was
requested, so remember to include the `--sex` flag.

### Output schema

`sex.tsv` captures one row per sample:

| Column | Description |
| --- | --- |
| `IID` | Sample identifier sourced from the PLINK `.fam` file or VCF/BCF header. |
| `Sex` | Final call (`male` or `female`). |
| `Y_Density` | Coverage density estimate across chromosome Y when enough variants were observed; otherwise `NA`. |
| `X_AutoHet_Ratio` | Ratio of X heterozygosity to the autosomal baseline; `NA` when insufficient variants remain. |
| `Composite_Index` | The upstream library's composite sex index when available; otherwise `NA`. |
| `Auto_Valid` | Number of autosomal variants processed. |
| `Auto_Het` | Number of heterozygous autosomal variants processed. |
| `X_NonPAR_Valid` | Number of non-PAR X variants processed. |
| `X_NonPAR_Het` | Number of heterozygous non-PAR X variants processed. |
| `Y_NonPAR_Valid` | Number of non-PAR Y variants processed. |
| `Y_PAR_Valid` | Number of Y PAR variants processed. |

The header is always included and the file is written with Unix newlines. Parent
directories are created automatically when the resolved output lives outside the
current working directory.

### Sex inference algorithm

The implementation streams genotype data in 256-variant blocks, forwarding the
X and Y chromosome slices to the [`infer_sex`](https://docs.rs/infer_sex)
accumulators provided by the upstream crate. Each sample maintains its own
accumulator, which digests heterozygosity on the X chromosome together with Y
coverage statistics to arrive at a final `InferredSex` call. Missing dosages are
preserved as `NaN` by the genotype reader and therefore do not contribute to the
summary counts. Both pseudoautosomal (PAR) and non-PAR Y variants are tracked
and written to `sex.tsv`; the upstream algorithm uses non-PAR Y density for its
decision metrics while still surfacing PAR counts for QC.

Before processing, gnomon inspects the maximum observed X-chromosome position to
automatically select between the GRCh37 and GRCh38 coordinate systems. This is
important because the reference position threshold for pseudoautosomal boundary
detection differs by build. All other variant metadata are streamed directly
from the genotype reader, so the command inherits the same provenance and
validation guarantees as the PCA and scoring pipelines. Autosomal variants are
downsampled to a maximum of 2,000 evenly spaced sites across the input order to
stay fast while avoiding bias toward the earliest chromosomes in the BIM.

### Library API

The CLI experience is backed by two convenience APIs:

* `terms::infer_sex_to_tsv(genotype_path: &Path, force_build: Option<GenomeBuild>)` – Loads the dataset, performs
  inference, writes `sex.tsv`, and returns the resolved output path.
* `terms::SexInferenceRecord` – Bundles the `individual_id` with the raw
  `InferenceResult` produced by the upstream crate, giving downstream code
  access to the detailed evidence that informed the final label written to
  `sex.tsv`.

These helpers make it straightforward to integrate sex inference into larger
pipelines without shelling out to the CLI.

### Failure modes and validation

* **Unexpected variant counts.** The streaming interface expects the genotype
  iterator to yield exactly the advertised number of variants. Overflows or
  underflows raise explicit errors so the CLI exits instead of silently
  skipping data.
* **Insufficient informative loci.** Sex inference requires at least one
  autosomal variant after selection; Y evidence is optional to support X-only
  datasets. Runs missing the autosomal baseline terminate with a clear error
  instead of synthesizing placeholder labels.
* **Unsupported chromosome labels.** Only chromosome labels recognised as X or Y
  contribute to the inference. Everything else—including haploid or mitochondrial
  contigs—is ignored.
* **I/O errors.** Any underlying read/write failures are surfaced as structured
  errors that bubble up to the CLI.

In all cases the CLI exits with a non-zero status and prints the diagnostic so
batch pipelines can react accordingly.
