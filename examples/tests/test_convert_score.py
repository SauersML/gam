from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from examples import convert_score


def test_download_pgs_score_uses_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path
    cached = cache_dir / "PGS000001_hmPOS_GRCh37.txt"
    cached.write_text("cached", encoding="utf-8")

    called = False

    def fake_download(url: str, destination: Path) -> None:  # pragma: no cover - safety
        nonlocal called
        called = True
        raise AssertionError("download should not be invoked when cache exists")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    result = convert_score.download_pgs_score("PGS000001", cache_dir, assembly="GRCh37")

    assert result == cached
    assert called is False


def test_download_pgs_score_decompresses_fresh_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path
    target_gz = cache_dir / "PGS000002_hmPOS_GRCh37.txt.gz"

    def fake_download(url: str, destination: Path) -> None:
        with gzip.open(destination, "wb") as handle:
            handle.write(b"hm_chr\thm_pos\n")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    result = convert_score.download_pgs_score("PGS000002", cache_dir, assembly="GRCh37")

    assert result.read_text(encoding="utf-8") == "hm_chr\thm_pos\n"
    assert not target_gz.exists()


def test_download_pgs_score_raises_when_assembly_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path

    def fake_download(url: str, destination: Path) -> None:
        raise RuntimeError("404")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    with pytest.raises(RuntimeError) as excinfo:
        convert_score.download_pgs_score("PGS000003", cache_dir, assembly="GRCh37")

    assert "PGS000003" in str(excinfo.value)
    assert "GRCh37" in str(excinfo.value)


def test_download_pgs_score_does_not_mix_cached_assemblies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path
    other_assembly = cache_dir / "PGS000004_hmPOS_GRCh38.txt"
    other_assembly.write_text("other", encoding="utf-8")

    def fake_download(url: str, destination: Path) -> None:
        raise RuntimeError("404")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    with pytest.raises(RuntimeError):
        convert_score.download_pgs_score("PGS000004", cache_dir, assembly="GRCh37")

    assert other_assembly.exists()


def test_convert_genome_to_vcf_passes_requested_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genome_path = tmp_path / "genome.txt"
    genome_path.write_text("genome", encoding="utf-8")
    reference = tmp_path / "ref.fa"
    reference.write_text(">ref\nACGT\n", encoding="utf-8")
    output_dir = tmp_path / "vcf"

    recorded: list[list[str]] = []

    def fake_run_command(argv, cwd=None):
        recorded.append(list(map(str, argv)))
        Path(argv[3]).write_text("##fileformat=VCFv4.2\n", encoding="utf-8")

    monkeypatch.setattr(convert_score, "run_command", fake_run_command)

    vcf_path = convert_score.convert_genome_to_vcf(
        Path("convert_genome"),
        genome_path,
        "Sample",
        reference,
        output_dir,
        "GRCh38",
    )

    assert vcf_path.exists()
    assert recorded
    assert recorded[0][-1] == "GRCh38"


def test_parser_exposes_assembly_flag() -> None:
    parser = convert_score.build_parser()
    default_args = parser.parse_args([])
    assert default_args.assembly == "GRCh37"

    custom_args = parser.parse_args(["--assembly", "GRCh38"])
    assert custom_args.assembly == "GRCh38"

