#!/usr/bin/env python3
"""Example workflow that converts direct-to-consumer genotype exports to VCF and runs
polygenic scores from the PGS catalog with gnomon.

Steps performed:
    1. Download the GRCh37 reference FASTA from Illumina if it is not present.
    2. Use the `convert_genome` Rust CLI to convert the text genomes in ``data`` to VCFs.
    3. Transform each VCF into a PLINK fileset with ``plink2``.
    4. Download a configurable set of PGS catalog scores and score each converted genome
       with gnomon's high-performance pipeline.
    5. Print the final score matrix and assert a few high-level invariants so the script
       can be dropped into continuous integration.

The script is intentionally verbose and defensive.  It prints each shell command prior to
execution, validates that required binaries are available, and refuses to silently skip
steps.  This makes it appropriate for manual runs when experimenting with new scores as
well as for automated checks that should run whenever the ``score`` crate changes.

Usage (after building gnomon with ``cargo build --release``)::

    python examples/convert_score.py \
        --scores PGS004696 PGS003725 \
        --output-dir examples/convert_score_output

The script caches downloaded artifacts inside ``examples/convert_score_output`` so repeat
runs are quick.  Delete that directory to force a full re-run.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# --------------------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "convert_score_output"
DEFAULT_REFERENCE_URL = (
    "https://webdata.illumina.com/downloads/productfiles/"
    "microarray-analytics-array/GRCh37_genome.zip"
)
DEFAULT_PGS_IDS = (
    "PGS000007",
    "PGS000317",
    "PGS004869",
    "PGS000507",
)
PGS_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"

GENOME_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("genome_Joshua_Yoakem_v5_Full_20250129211749.txt", "Joshua_Yoakem"),
    ("autosomal.txt", "LivingDNA_Autosomal"),
)

# --------------------------------------------------------------------------------------
# Dataclasses and small utilities
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolPaths:
    convert_genome: Path
    plink2: Path
    gnomon: Path


@dataclass
class ScoreResult:
    pgs_id: str
    average: float
    missing_pct: float


def debug(msg: str) -> None:
    """Small helper to print progress messages with a common prefix."""

    print(f"[convert_score] {msg}", flush=True)


class CommandError(RuntimeError):
    """Raised when a subprocess finishes unsuccessfully."""


def run_command(argv: Sequence[os.PathLike[str] | str], cwd: Path | None = None) -> None:
    """Execute ``argv`` and raise :class:`CommandError` on failure."""

    printable = " ".join(map(str, argv))
    if cwd is not None:
        printable = f"(cd {cwd} && {printable})"
    debug(f"$ {printable}")
    completed = subprocess.run(list(map(str, argv)), cwd=cwd)
    if completed.returncode != 0:
        raise CommandError(
            f"Command failed with exit code {completed.returncode}: {printable}"
        )


def ensure_binary(name: str, explicit: str | None = None) -> Path:
    """Return the resolved path to ``name`` or raise a clear error."""

    candidate = Path(explicit) if explicit else shutil.which(name)
    if not candidate:
        raise FileNotFoundError(
            f"Required binary '{name}' was not found in PATH. Install it or pass ``--{name.replace('-', '_')}-bin``."
        )
    path = Path(candidate).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Binary '{name}' resolved to {path}, but the file does not exist.")
    return path


# --------------------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------------------

def stream_download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` using urllib with streaming."""

    debug(f"Downloading {url} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response, open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (HTTPError, URLError) as exc:  # pragma: no cover - network failure
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def ensure_reference_fasta(reference_url: str, work_dir: Path) -> Path:
    """Download and extract the FASTA file used by ``convert_genome``."""

    archive_name = Path(reference_url.split("/")[-1])
    archive_path = work_dir / archive_name
    candidate_fastas: List[Path] = []

    for suffix in (".fa", ".fasta", ".fa.gz", ".fasta.gz"):
        candidate_fastas.extend(work_dir.glob(f"*{suffix}"))
    if candidate_fastas:
        debug(f"Using existing reference FASTA: {candidate_fastas[0]}")
        return candidate_fastas[0]

    if not archive_path.exists():
        stream_download(reference_url, archive_path)

    debug(f"Extracting reference from {archive_path}")
    extracted_fasta: Path | None = None
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz")):
                target = work_dir / Path(name).name
                if not target.exists():
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted_fasta = target
                break
    if extracted_fasta is None:
        raise RuntimeError(
            "Reference archive did not contain a .fa or .fasta file; convert_genome cannot run."
        )

    if extracted_fasta.suffixes[-1] == ".gz":
        debug(f"Decompressing {extracted_fasta}")
        decompressed = extracted_fasta.with_suffix("")
        with gzip.open(extracted_fasta, "rb") as src, open(decompressed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted_fasta = decompressed

    debug(f"Reference FASTA ready: {extracted_fasta}")
    return extracted_fasta


def download_pgs_score(
    pgs_id: str, cache_dir: Path, assembly: str = "GRCh37"
) -> Path:
    """Download the harmonized PGS scoring file for ``assembly``."""

    valid_assemblies = {"GRCh37", "GRCh38"}
    if assembly not in valid_assemblies:
        raise ValueError(
            f"Unsupported assembly '{assembly}'. Expected one of: {sorted(valid_assemblies)}."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{pgs_id}_hmPOS_{assembly}.txt.gz"
    url = f"{PGS_BASE_URL}/{pgs_id}/ScoringFiles/Harmonized/{filename}"
    target_gz = cache_dir / filename
    target_txt = cache_dir / filename[:-3]

    if target_txt.exists():
        debug(f"Using cached score {target_txt}")
        return target_txt

    if not target_gz.exists():
        try:
            stream_download(url, target_gz)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Unable to download harmonized score for {pgs_id} ({assembly})."
            ) from exc

    debug(f"Decompressing {target_gz}")
    with gzip.open(target_gz, "rb") as src, open(target_txt, "wb") as dst:
        shutil.copyfileobj(src, dst)
    target_gz.unlink(missing_ok=True)

    if not target_txt.exists():
        raise RuntimeError(
            f"Failed to prepare harmonized score for {pgs_id} ({assembly}); expected {target_txt}."
        )

    return target_txt


# --------------------------------------------------------------------------------------
# Conversion steps
# --------------------------------------------------------------------------------------


def normalize_vcf_genotypes(vcf_path: Path) -> None:
    """Fix genotype fields that some tools emit with leading separators."""

    lines: List[str] = []
    with vcf_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("#"):
                lines.append(raw)
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) > 4 and parts[4] == ".":
                continue
            for idx in range(9, len(parts)):
                field = parts[idx]
                if field and field[0] in {"/", "|"}:
                    parts[idx] = field[1:]
            lines.append("\t".join(parts) + "\n")
    vcf_path.write_text("".join(lines), encoding="utf-8")


def convert_genome_to_vcf(
    tool: Path,
    genome_path: Path,
    sample_id: str,
    reference_fasta: Path,
    output_dir: Path,
    assembly: str,
) -> Path:
    """Invoke the ``convert_genome`` CLI."""

    output_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = output_dir / f"{genome_path.stem}.vcf"
    if vcf_path.exists():
        debug(f"Skipping conversion; VCF already exists: {vcf_path}")
        return vcf_path

    run_command(
        (
            tool,
            genome_path,
            reference_fasta,
            vcf_path,
            "--sample",
            sample_id,
            "--assembly",
            assembly,
        )
    )
    if not vcf_path.exists():
        raise RuntimeError(f"convert_genome reported success but {vcf_path} was not created.")
    normalize_vcf_genotypes(vcf_path)
    return vcf_path


def vcf_to_plink(tool: Path, vcf_path: Path, output_dir: Path, sample_id: str) -> Path:
    """Convert a VCF into a PLINK binary fileset using plink2."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / vcf_path.stem
    bed_path = prefix.with_suffix(".bed")
    psam_path = prefix.with_suffix(".psam")

    psam_path.write_text(
        "#FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE\n"
        f"{sample_id}\t{sample_id}\t0\t0\t0\t-9\n",
        encoding="utf-8",
    )

    if bed_path.exists() and prefix.with_suffix(".bim").exists() and prefix.with_suffix(".fam").exists():
        debug(f"PLINK files already exist for {vcf_path}; skipping conversion")
        return prefix

    run_command(
        (
            tool,
            "--vcf",
            vcf_path,
            "--double-id",
            "--keep-allele-order",
            "--psam",
            psam_path,
            "--allow-no-sex",
            "--split-par",
            "b37",
            "--max-alleles",
            "2",
            "--polyploid-mode",
            "missing",
            "--make-bed",
            "--out",
            prefix,
        )
    )

    for ext in (".bed", ".bim", ".fam"):
        expected = prefix.with_suffix(ext)
        if not expected.exists():
            raise RuntimeError(f"plink2 failed to produce {expected}")
    return prefix


def run_gnomon_score(
    tool: Path,
    score_path: Path,
    plink_prefix: Path,
) -> Path:
    """Execute ``gnomon score`` and return the path to the generated ``.sscore`` file."""

    run_command((tool, "score", score_path, plink_prefix))

    sscore_path = plink_prefix.with_suffix(".sscore")
    if not sscore_path.exists():
        raise RuntimeError(f"gnomon score did not create {sscore_path}")
    return sscore_path


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------

def parse_sscore(path: Path) -> Tuple[List[str], List[ScoreResult]]:
    """Parse a gnomon ``.sscore`` file into a row-wise structure."""

    with path.open("r", encoding="utf-8") as handle:
        header: List[str] | None = None
        rows: List[ScoreResult] = []
        for raw in handle:
            if raw.startswith("#REGION"):
                continue
            if header is None:
                header = [col.lstrip("#") for col in raw.rstrip().split("\t")]
                continue
            values = raw.rstrip().split("\t")
            iid = values[0]
            idx = 1
            while idx < len(values):
                score_name = header[idx]
                if not score_name.endswith("_AVG"):
                    idx += 1
                    continue
                avg = float(values[idx])
                missing = float(values[idx + 1])
                rows.append(
                    ScoreResult(
                        pgs_id=score_name.replace("_AVG", ""),
                        average=avg,
                        missing_pct=missing,
                    )
                )
                idx += 2
        if header is None:
            raise RuntimeError(f"No header found in {path}")
    return header, rows


def derive_sample_id(filename: str, override: str | None = None) -> str:
    """Generate a stable sample ID for the provided genome filename."""

    if override:
        return override
    name = Path(filename).stem
    name = re.sub(r"^genome_", "", name)
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return sanitized or "Sample"


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------

def orchestrate(args: argparse.Namespace) -> None:
    tools = ToolPaths(
        convert_genome=ensure_binary("convert_genome", args.convert_genome_bin),
        plink2=ensure_binary("plink2", args.plink2_bin),
        gnomon=ensure_binary("gnomon", args.gnomon_bin or str(REPO_ROOT / "target" / "release" / "gnomon")),
    )

    output_dir = Path(args.output_dir).resolve()
    vcf_dir = output_dir / "vcf"
    plink_dir = output_dir / "plink"
    score_cache = output_dir / "scores"

    reference_fasta = ensure_reference_fasta(args.reference_url, output_dir / "reference")

    genome_inputs: List[Tuple[Path, str]] = []
    for filename, override in GENOME_SOURCES:
        path = DATA_DIR / filename
        if not path.exists():
            debug(f"Skipping missing genome file: {path}")
            continue
        genome_inputs.append((path, derive_sample_id(filename, override)))

    if len(genome_inputs) < 2:
        raise RuntimeError(
            "Expected at least two genome files in the data directory; cannot proceed."
        )

    all_results: Dict[str, Dict[str, ScoreResult]] = defaultdict(dict)

    for genome_path, sample_id in genome_inputs:
        debug(f"Processing genome {genome_path.name} (sample {sample_id})")
        vcf_path = convert_genome_to_vcf(
            tools.convert_genome,
            genome_path,
            sample_id,
            reference_fasta,
            vcf_dir,
            args.assembly,
        )
        plink_prefix = vcf_to_plink(tools.plink2, vcf_path, plink_dir, sample_id)

        for pgs_id in args.scores:
            score_path = download_pgs_score(pgs_id, score_cache, assembly=args.assembly)
            sscore_path = run_gnomon_score(tools.gnomon, score_path, plink_prefix)
            print()
            print(
                f"==== Contents of {sscore_path.name} for sample {sample_id} / {pgs_id} ===="
            )
            sscore_contents = sscore_path.read_text(encoding="utf-8")
            print(sscore_contents, end="")
            if not sscore_contents.endswith("\n"):
                print()
            print("==== End of", sscore_path.name, "====\n")
            header, scores = parse_sscore(sscore_path)
            assert scores, f"No score rows parsed from {sscore_path}"  # noqa: S101
            for score in scores:
                assert 0.0 <= score.missing_pct <= 100.0  # noqa: S101
                all_results[sample_id][score.pgs_id] = score

    print("\nFinal score summary:\n======================")
    for sample, sample_scores in all_results.items():
        print(f"Sample: {sample}")
        for pgs_id, score in sorted(sample_scores.items()):
            print(
                f"  {pgs_id}: average={score.average:.6f}, missing_pct={score.missing_pct:.2f}"
            )
        print()

    expected_scores = set(args.scores)
    for sample, sample_scores in all_results.items():
        assert (
            expected_scores <= set(sample_scores)
        ), f"Sample {sample} missing expected scores"

    debug("All assertions passed. Conversion and scoring succeeded.")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        nargs="+",
        default=list(DEFAULT_PGS_IDS),
        help="PGS catalog identifiers to download and score (default: %(default)s)",
    )
    parser.add_argument(
        "--assembly",
        default="GRCh37",
        choices=("GRCh37", "GRCh38"),
        help=(
            "Genome assembly to use for convert_genome and PGS downloads "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reference-url",
        default=DEFAULT_REFERENCE_URL,
        help="URL to a ZIP archive containing a GRCh37 FASTA (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Working directory for downloads and intermediate files",
    )
    parser.add_argument(
        "--convert-genome-bin",
        default=None,
        help="Path to the convert_genome executable (defaults to PATH lookup)",
    )
    parser.add_argument(
        "--plink2-bin",
        default=None,
        help="Path to the plink2 executable (defaults to PATH lookup)",
    )
    parser.add_argument(
        "--gnomon-bin",
        default=None,
        help="Path to the gnomon executable (defaults to target/release/gnomon)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrate(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - convenience
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - convenience
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
