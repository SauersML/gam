"""Run PCA on genomic data using Hail."""

from __future__ import annotations

import argparse
import atexit
import fnmatch
import gzip
import logging
import os
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

import hail as hl

try:
    import pysam  # type: ignore
except Exception as exc:  # pragma: no cover - runtime fallback
    raise RuntimeError(
        "pysam is required for handling BCF inputs but could not be imported"
    ) from exc

DEFAULT_DATA_PATH = (
    "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/*.bcf"
)
GCS_CONNECTOR_URL = "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar"
DEFAULT_NUM_PCS = 10
DEFAULT_OUTPUT_PREFIX = "hail_pca"


def _infer_variant_format(path: str) -> str | None:
    """Return the obvious variant format for *path* if one is implied."""

    lowered = path.lower()
    if lowered.endswith((".bcf", ".bcf.bgz", ".bcf.gz")):
        return "bcf"
    if lowered.endswith((".vcf", ".vcf.gz", ".vcf.bgz")):
        return "vcf"
    return None


def _detect_variant_format_from_file(path: str) -> str | None:
    """Return ``"bcf"`` or ``"vcf"`` if the file content makes it obvious."""

    try:
        with hl.hadoop_open(path, "rb") as raw:
            prefix = raw.read(4)
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise RuntimeError(f"Unable to inspect '{path}' to determine its format") from exc

    if prefix.startswith(b"BCF"):
        return "bcf"
    if prefix.startswith(b"##"):
        return "vcf"
    if prefix[:2] != b"\x1f\x8b":
        return None

    try:
        with hl.hadoop_open(path, "rb") as compressed, gzip.GzipFile(fileobj=compressed) as gz:
            header = gz.read(4)
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise RuntimeError(
            f"Unable to inspect compressed file '{path}' to determine its format"
        ) from exc

    if header.startswith(b"BCF"):
        return "bcf"
    if header.startswith(b"##"):
        return "vcf"
    return None


def _resolve_variant_path(path: str) -> tuple[str, str]:
    """Return a glob that selects variants under *path* alongside the format."""

    trimmed = path.rstrip("/")
    inferred = _infer_variant_format(trimmed)

    if "*" in trimmed:
        matches = _list_matching_variant_paths(trimmed)
        if not matches:
            raise FileNotFoundError(f"No files matched pattern '{trimmed}'.")

        try:
            detected = _detect_variant_format_from_file(matches[0])
        except RuntimeError:
            detected = None

        if detected is not None:
            return trimmed, detected

        inferred_match = _infer_variant_format(matches[0])
        if inferred_match is not None:
            return trimmed, inferred_match

        if inferred is not None:
            return trimmed, inferred

        raise ValueError(
            "Unable to determine whether the input files are VCF or BCF. "
            "Please pass a path with an explicit extension."
        )

    if inferred is not None:
        try:
            detected = _detect_variant_format_from_file(trimmed)
        except RuntimeError:
            detected = None

        return trimmed, detected or inferred

    listing_target = path if path.endswith("/") else f"{path}/"
    try:
        entries = hl.hadoop_ls(listing_target)
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise FileNotFoundError(
            f"Failed to list files under '{listing_target}'. Ensure the path exists."
        ) from exc

    files = [entry["path"] for entry in entries if not entry.get("is_dir")]
    if not files:
        raise FileNotFoundError(f"No files were found under '{listing_target}'.")

    for suffix, fmt in (
        (".bcf", "bcf"),
        (".bcf.bgz", "bcf"),
        (".bcf.gz", "bcf"),
        (".vcf.bgz", "vcf"),
        (".vcf.gz", "vcf"),
        (".vcf", "vcf"),
    ):
        matching = [name for name in files if name.lower().endswith(suffix)]
        if not matching:
            continue

        detected = _detect_variant_format_from_file(matching[0]) or fmt
        base = listing_target.rstrip("/")
        return f"{base}/*{suffix}", detected

    raise FileNotFoundError(
        "Unable to determine whether the input files are VCF or BCF. "
        "Please pass a path with an explicit extension."
    )


def _list_matching_variant_paths(pattern: str) -> List[str]:
    """Expand *pattern* into a list of matching variant file paths."""

    if "*" not in pattern:
        return [pattern]

    directory, _, name_pattern = pattern.rpartition("/")
    directory = f"{directory}/" if directory else ""

    if pattern.startswith("gs://"):
        target = directory or pattern
        entries = hl.hadoop_ls(target)
        return [
            entry["path"]
            for entry in entries
            if not entry.get("is_dir")
            and fnmatch.fnmatch(entry["path"].split("/")[-1], name_pattern)
        ]

    base = Path(directory) if directory else Path.cwd()
    return [str(path) for path in base.glob(name_pattern)]


def _copy_variant_to_local(source: str, destination: Path) -> None:
    """Copy a variant file from *source* to *destination*."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.startswith("gs://"):
        hl.hadoop_copy(source, str(destination))
    else:
        shutil.copyfile(source, destination)


def _copy_variant_index_if_available(source: str, destination: Path) -> None:
    """Copy an index sidecar file for *source* into *destination*'s directory."""

    sidecar_suffixes = (".csi", ".tbi", ".idx")
    for suffix in sidecar_suffixes:
        index_source = f"{source}{suffix}"
        if source.startswith("gs://"):
            exists = hl.hadoop_exists(index_source)
        else:
            exists = Path(index_source).exists()
        if not exists:
            continue

        index_destination = destination.with_name(f"{destination.name}{suffix}")
        _copy_variant_to_local(index_source, index_destination)
        break


def _guess_reference_from_header(header: "pysam.libcbcf.VariantHeader") -> Optional[str]:
    """Infer the reference genome from a pysam header if possible."""

    try:
        reference_line = header.get("reference")
    except AttributeError:  # pragma: no cover - pysam interface variations
        reference_line = None

    if reference_line:
        text = str(reference_line[0]).lower()
        if "grch38" in text or "hg38" in text:
            return "GRCh38"
        if "grch37" in text or "hg19" in text:
            return "GRCh37"

    contigs = list(getattr(header, "contigs", []))
    if not contigs:
        return None

    if any(str(name).lower().startswith("chr") for name in contigs):
        return "GRCh38"
    return "GRCh37"


def _convert_single_bcf(
    source: str, destination_directory: Path, index: int
) -> tuple[Path, Optional[str]]:
    """Convert a single BCF *source* into a bgzipped VCF within *destination_directory*."""

    local_bcf = destination_directory / f"input_{index}.bcf"
    logging.info("Copying %s to %s for conversion", source, local_bcf)
    _copy_variant_to_local(source, local_bcf)
    _copy_variant_index_if_available(source, local_bcf)

    dest_path = destination_directory / f"converted_{index}.vcf.bgz"
    logging.info("Converting %s to %s", local_bcf, dest_path)
    header_reference: Optional[str] = None
    with pysam.VariantFile(str(local_bcf)) as reader:
        threads = max(1, min((os.cpu_count() or 1), 8))
        try:
            reader.set_threads(threads)
        except AttributeError:  # pragma: no cover - older pysam versions
            pass
        header_reference = _guess_reference_from_header(reader.header)
        with pysam.VariantFile(str(dest_path), "wz", header=reader.header) as writer:
            try:
                writer.set_threads(threads)
            except AttributeError:  # pragma: no cover - older pysam versions
                pass
            for record in reader:
                writer.write(record)

    logging.info("Indexing converted VCF %s", dest_path)
    pysam.tabix_index(str(dest_path), preset="vcf", force=True)

    for sidecar_suffix in (".csi", ".tbi", ".idx"):
        try:
            (local_bcf.with_name(f"{local_bcf.name}{sidecar_suffix}")).unlink()
        except FileNotFoundError:
            continue

    try:
        local_bcf.unlink()
    except FileNotFoundError:  # pragma: no cover - cleanup race
        pass

    return dest_path, header_reference


def _remove_if_exists(path: Path) -> None:
    """Delete *path* if it exists."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _positive_int_from_env(name: str) -> Optional[int]:
    """Return a positive integer from *name* if it is set and valid."""

    raw = os.environ.get(name)
    if raw is None:
        return None

    try:
        value = int(raw)
    except ValueError:
        logging.warning("Ignoring %s=%r; expected a positive integer", name, raw)
        return None

    if value <= 0:
        logging.warning("Ignoring %s=%r; value must be greater than zero", name, raw)
        return None

    return value


def _conversion_worker_count(total_files: int) -> int:
    """Return the number of concurrent BCF conversions to run."""

    env_value = _positive_int_from_env("GNOMON_BCF_CONVERSION_WORKERS")
    if env_value is not None:
        return min(total_files, env_value)

    cpu_count = os.cpu_count() or 1
    return min(total_files, max(1, min(cpu_count, 8)))


def _conversion_chunk_size(total_files: int, workers: int) -> int:
    """Return how many converted VCFs to import per batch."""

    env_value = _positive_int_from_env("GNOMON_BCF_IMPORT_CHUNK_SIZE")
    if env_value is not None:
        return min(total_files, max(1, env_value))

    # Default to a modest multiple of the worker count to reduce import passes
    default_chunk = max(1, min(total_files, max(workers * 2, 4)))
    return default_chunk


def _import_bcf_as_matrix_table(pattern: str) -> hl.MatrixTable:
    """Import BCF inputs matching *pattern* by converting them incrementally to VCF."""

    matches = _list_matching_variant_paths(pattern)
    if not matches:
        raise FileNotFoundError(f"No BCF files matched pattern '{pattern}'.")

    temp_dir = Path(tempfile.mkdtemp(prefix="hail-bcf-"))
    atexit.register(shutil.rmtree, temp_dir, True)

    reference: Optional[str] = None
    current_checkpoint: Optional[str] = None
    mt: Optional[hl.MatrixTable] = None

    workers = _conversion_worker_count(len(matches))
    chunk_size = _conversion_chunk_size(len(matches), workers)
    logging.info(
        "Preparing to convert %d BCF files with %d workers in chunks of %d",
        len(matches),
        workers,
        chunk_size,
    )

    def schedule_chunk(
        executor: ThreadPoolExecutor, start_index: int
    ) -> list[tuple[str, Future[tuple[Path, Optional[str]]]]]:
        scheduled: list[tuple[str, Future[tuple[Path, Optional[str]]]]] = []
        for offset in range(chunk_size):
            index = start_index + offset
            if index >= len(matches):
                break
            source = matches[index]
            future = executor.submit(_convert_single_bcf, source, temp_dir, index)
            scheduled.append((source, future))
        return scheduled

    next_index = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scheduled = schedule_chunk(executor, next_index)
        next_index += len(scheduled)

        while scheduled:
            converted_results: list[tuple[Path, Optional[str]]] = []
            for source, future in scheduled:
                converted_results.append(future.result())

            # Begin converting the next chunk before triggering the expensive import/checkpoint
            next_batch = schedule_chunk(executor, next_index)
            next_index += len(next_batch)

            chunk_reference = next(
                (ref for _, ref in converted_results if ref is not None), None
            )
            if reference is None and chunk_reference is not None:
                reference = chunk_reference

            converted_paths = [str(path) for path, _ in converted_results]
            import_kwargs = {
                "force_bgz": True,
                "array_elements_required": False,
            }
            if reference is not None:
                import_kwargs["reference_genome"] = reference

            logging.info(
                "Importing a chunk of %d converted VCFs", len(converted_paths)
            )
            piece_mt = hl.import_vcf(converted_paths, **import_kwargs)

            if mt is None:
                mt = piece_mt
            else:
                mt = mt.union_rows(piece_mt)

            checkpoint_target = hl.utils.new_temp_file(
                prefix="hail-bcf-checkpoint", extension="mt"
            )
            logging.info("Checkpointing MatrixTable to %s", checkpoint_target)
            mt = mt.checkpoint(checkpoint_target, overwrite=True)

            if current_checkpoint is not None and current_checkpoint != checkpoint_target:
                logging.info("Removing previous checkpoint %s", current_checkpoint)
                try:
                    hl.hadoop_remove(current_checkpoint, recursive=True)
                except Exception:  # pragma: no cover - best effort cleanup
                    logging.warning(
                        "Failed to remove previous checkpoint %s", current_checkpoint
                    )

            current_checkpoint = checkpoint_target

            logging.info("Cleaning up %d converted VCFs", len(converted_paths))
            for converted_path_str in converted_paths:
                converted_path = Path(converted_path_str)
                _remove_if_exists(converted_path)
                for suffix in (".tbi", ".csi", ".idx"):
                    _remove_if_exists(converted_path.with_name(f"{converted_path.name}{suffix}"))

            scheduled = next_batch

    if mt is None:
        raise RuntimeError("BCF import produced no MatrixTable data")

    return mt


def _infer_reference_genome(pattern: str) -> Optional[str]:
    """Infer a reference genome for the VCF inputs matching *pattern*."""

    matches = _list_matching_variant_paths(pattern)
    if not matches:
        raise FileNotFoundError(f"No variant files matched pattern '{pattern}'.")

    first = matches[0]
    cleanup_dir: Optional[Path] = None
    local_path = Path(first)

    if first.startswith("gs://"):
        cleanup_dir = Path(tempfile.mkdtemp(prefix="hail-ref-"))
        local_path = cleanup_dir / Path(first).name
        _copy_variant_to_local(first, local_path)
        _copy_variant_index_if_available(first, local_path)

    try:
        with pysam.VariantFile(str(local_path)) as reader:
            return _guess_reference_from_header(reader.header)
    finally:
        if cleanup_dir is not None:
            atexit.register(shutil.rmtree, cleanup_dir, True)


def _ensure_gcs_connector() -> str:
    """Download the GCS connector jar if it is not already cached."""

    destination = Path.home() / ".hail" / "gcs-connector-hadoop3-latest.jar"
    if destination.exists():
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")

    try:
        with urlopen(GCS_CONNECTOR_URL) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except URLError as exc:  # pragma: no cover - network failure reporting
        raise RuntimeError("Failed to download the GCS connector required for gs:// support") from exc

    temp_path.replace(destination)
    return str(destination)


def _spark_conf_for_path(path: str) -> dict[str, str]:
    """Return Spark configuration needed for the input path."""

    if path.startswith("gs://"):
        connector_path = _ensure_gcs_connector()
        return {
            "spark.hadoop.fs.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            "spark.hadoop.fs.AbstractFileSystem.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
            "spark.hadoop.fs.gs.auth.service.account.enable": "false",
            "spark.hadoop.fs.gs.auth.null.enable": "true",
            "spark.jars": connector_path,
        }

    return {}


def _read_genotypes(path: str) -> hl.MatrixTable:
    """Read genotype data from either a MatrixTable or variant inputs."""

    if path.endswith(".mt"):
        logging.info("Reading MatrixTable from %s", path)
        return hl.read_matrix_table(path)

    variant_path, variant_format = _resolve_variant_path(path)

    if variant_format == "bcf":
        logging.info("Importing BCFs from %s", variant_path)
        return _import_bcf_as_matrix_table(variant_path)

    logging.info("Importing VCFs from %s", variant_path)
    reference = _infer_reference_genome(variant_path)
    paths = _list_matching_variant_paths(variant_path)
    force_bgz = all(path.endswith((".vcf.bgz", ".vcf.gz")) for path in paths)
    import_kwargs = {"array_elements_required": False, "force_bgz": force_bgz}
    if reference is not None:
        import_kwargs["reference_genome"] = reference
    return hl.import_vcf(variant_path, **import_kwargs)


def _export_scores(scores: hl.Table, output_prefix: str) -> None:
    scores.export(f"{output_prefix}.scores.tsv.bgz")


def _export_loadings(loadings: hl.Table, output_prefix: str) -> None:
    if loadings is None:
        logging.info("No loadings were produced; skipping loadings export")
        return

    loadings.export(f"{output_prefix}.loadings.tsv.bgz")


def _export_eigenvalues(eigenvalues: Iterable[float], output_prefix: str) -> None:
    rows = [
        {"component": i + 1, "eigenvalue": float(value)}
        for i, value in enumerate(eigenvalues)
    ]
    eigenvalue_ht = hl.Table.parallelize(rows).key_by("component")
    eigenvalue_ht.export(f"{output_prefix}.eigenvalues.tsv.bgz")


def run_pca(
    data_path: str,
    output_prefix: str,
    n_pcs: int,
) -> None:
    """Run PCA on the provided data and export the results."""

    logging.info(
        "Starting PCA with data_path=%s, output_prefix=%s, n_pcs=%d",
        data_path,
        output_prefix,
        n_pcs,
    )
    hl.init(spark_conf=_spark_conf_for_path(data_path))

    mt = _read_genotypes(data_path)

    if "GT" not in mt.entry:
        raise ValueError("Genotype MatrixTable must contain a 'GT' entry field")

    eigenvalues, scores, loadings = hl.hwe_normalized_pca(mt.GT, k=n_pcs)

    _export_eigenvalues(eigenvalues, output_prefix)
    _export_scores(scores, output_prefix)
    _export_loadings(loadings, output_prefix)

    logging.info("PCA complete")


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=(
            "Path to a Hail MatrixTable (.mt) or directory/glob of VCF/BCF files. "
            "Defaults to the public HGDP/1kG BCF directory."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Prefix used for the exported eigenvalues, scores, and loadings. "
            f"Defaults to '{DEFAULT_OUTPUT_PREFIX}'."
        ),
    )
    parser.add_argument(
        "--num-pcs",
        type=int,
        default=DEFAULT_NUM_PCS,
        help="Number of principal components to compute.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    return parser.parse_args(args=args)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    run_pca(
        args.data_path,
        args.output_prefix,
        args.num_pcs,
    )


if __name__ == "__main__":
    main()
