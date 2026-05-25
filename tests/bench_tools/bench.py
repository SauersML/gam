#!/usr/bin/env python3
"""
End-to-end profiling harness.

Builds the profiling binary, generates one mgcv dataset, trains with perf, and
writes a small HTML report under tests/bench_tools/bench_workdir.
"""

import shutil
import subprocess
import sys
import webbrowser
from html import escape
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import mgcv


WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
WORKDIR = SCRIPT_DIR / "bench_workdir"
WORKDIR.mkdir(exist_ok=True)

EXECUTABLE_PATH = WORKSPACE_ROOT / "target" / "profiling" / "gnomon"
PERF_PERCENT_LIMIT = 5


def run(cmd: list[str], cwd: Path = WORKSPACE_ROOT) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def capture(cmd: list[str], cwd: Path = WORKSPACE_ROOT) -> str:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout


def prepare_training_tsv(df: pd.DataFrame, out_path: Path) -> None:
    required = {"variable_two", "outcome"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Input DF missing required columns: {sorted(missing)}")
    df.rename(
        columns={"variable_two": "PC1", "outcome": "phenotype"},
    ).to_csv(out_path, sep="\t", index=False)


def build_profiling_binary() -> None:
    run(["cargo", "build", "--profile", "profiling"])


def profile_training(train_tsv: Path, tag: str) -> dict[str, object]:
    perf_data = WORKDIR / f"perf_{tag}.data"
    cmd = [
        "perf",
        "record",
        "-e",
        "cycles:u",
        "--call-graph",
        "dwarf,16384",
        "-F",
        "700",
        "-o",
        str(perf_data),
        "--",
        str(EXECUTABLE_PATH),
        "train",
        "--num-pcs",
        "1",
        "--pc-knots",
        "8",
        "--pc-degree",
        "3",
        str(train_tsv),
    ]
    run(cmd)
    report = capture(
        [
            "perf",
            "report",
            "--stdio",
            "--call-graph=graph",
            "--percent-limit",
            str(PERF_PERCENT_LIMIT),
            "--max-stack",
            "1024",
            "-i",
            str(perf_data),
        ]
    )
    return {"tag": tag, "perf_data": perf_data, "report": report}


def generate_flamegraph(perf_data: Path, tag: str) -> Path | None:
    if not shutil.which("inferno-collapse-perf") or not shutil.which("inferno-flamegraph"):
        return None

    collapsed = WORKDIR / f"collapsed_{tag}.txt"
    svg_path = WORKDIR / f"flame_{tag}.svg"
    capture(
        [
            "bash",
            "-lc",
            f"perf script -i {perf_data} | inferno-collapse-perf > {collapsed} && inferno-flamegraph {collapsed} > {svg_path}",
        ]
    )
    return svg_path


def write_report(section: dict[str, object]) -> Path:
    html_path = WORKDIR / "report.html"
    flame_svg = section["flame_svg"]

    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Calibrate Flamegraph</title>\n")
        f.write("<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:18px} h1{margin:0 0 8px} .meta{color:#444;margin:4px 0 16px} .flame svg{width:100%;height:auto;display:block} pre{white-space:pre-wrap;max-height:600px;overflow:auto;border:1px solid #eee;padding:8px;background:#fafafa}</style>\n")
        f.write("<h1>Calibrate Flamegraph</h1>\n")

        if isinstance(flame_svg, Path):
            f.write("<div class='flame'>")
            f.write(flame_svg.read_text(encoding="utf-8"))
            f.write("</div>")
        else:
            f.write("<p>Flamegraph not available. Install inferno-collapse-perf and inferno-flamegraph.</p>")

        f.write("<h2>Perf Text Report</h2>\n")
        f.write(f"<div class='meta'>Showing entries with >= {PERF_PERCENT_LIMIT}%</div>\n")
        f.write("<pre>")
        f.write(escape(str(section["report"])))
        f.write("</pre>")

    return html_path


def main() -> None:
    build_profiling_binary()

    train_tsv = SCRIPT_DIR / "rust_train_nonlinear.tsv"
    df = mgcv.generate_data(
        mgcv.N_SAMPLES_TRAIN,
        mgcv.NOISE_BLEND_FACTOR,
        linear_mode=False,
        noise_mode=False,
    )
    prepare_training_tsv(df, train_tsv)

    section = profile_training(train_tsv, "nonlinear")
    section["flame_svg"] = generate_flamegraph(section["perf_data"], "nonlinear")

    html_path = write_report(section)
    print(f"\nHTML report -> {html_path}")
    webbrowser.open(html_path.resolve().as_uri())
    print("All profiling runs complete.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
