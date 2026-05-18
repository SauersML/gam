"""Biobank-shape margslope-Duchon perf probe.

Mirrors the failing biobank shard `rust_margslope_aniso_duchon16d_rigid`
exactly (16 PCs, 24 centers, order=0, power=8, length_scale=1, with
--scale-dimensions for anisotropy). Runs at multiple sizes with a hard
per-fit wall budget so we can locate the dominant cost without burning
hours.

Outputs:
  - <outdir>/dch_<N>.csv          (synthetic biobank-shape data)
  - <outdir>/dch_<N>.log          (full gam fit stdout/stderr)
  - <outdir>/summary.tsv          (n, total_s, ext_coord_traces, max_per_call)

After the run, parse summary.tsv plus the logs to identify the bottleneck
stages. The dominant cost on the biobank-failing shard is repeated
`DenseSpectralOperator::trace_logdet_operator` calls, one per ψ-axis of
the anisotropy block, each ~50s at n=200/p=95.

Typical use:
    python3 bench/margslope_smoke/run.py \\
        --binary /path/to/gam \\
        --outdir bench/margslope_smoke/runs \\
        --sizes 100 200 500 \\
        --budget-sec 60

The binary can be any prebuilt gam release (same family + flag set as
biobank). When `--binary` is omitted, the script falls back to
`target/release/gam` and refuses to build.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PY = Path(__file__).resolve().parent / "gen.py"

DUCHON_TERM = (
    "duchon(pc1_std, pc2_std, pc3_std, pc4_std, pc5_std, pc6_std, pc7_std, "
    "pc8_std, pc9_std, pc10_std, pc11_std, pc12_std, pc13_std, pc14_std, "
    "pc15_std, pc16_std, centers=24, order=0, power=8, length_scale=1)"
)
MEAN_FORMULA = (
    f"phenotype ~ link(type=probit) + sex + smooth(age_entry_std) + {DUCHON_TERM}"
)
LOGSLOPE_FORMULA = f"smooth(age_entry_std) + {DUCHON_TERM}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", help="Path to gam binary; defaults to target/release/gam")
    ap.add_argument("--outdir", required=True, help="Output dir for CSVs / logs / summary")
    ap.add_argument("--sizes", nargs="+", type=int, required=True, help="n values to probe")
    ap.add_argument("--budget-sec", type=float, default=60.0, help="Per-fit wall-clock budget (default: 60s)")
    ap.add_argument("--seed", type=lambda v: int(v, 0), default=0x5CA1AB1E)
    return ap.parse_args()


def resolve_binary(arg: str | None) -> Path:
    if arg:
        path = Path(arg).resolve()
    else:
        path = (REPO_ROOT / "target" / "release" / "gam").resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"gam binary not found or not executable: {path}")
    return path


def run_one(binary: Path, n: int, outdir: Path, budget: float, seed: int) -> dict[str, Any]:
    data = outdir / f"dch_{n}.csv"
    log = outdir / f"dch_{n}.log"
    model = outdir / f"dch_{n}.model"

    subprocess.check_call(
        [sys.executable, str(GEN_PY), str(n), str(data), str(seed)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    cmd = [
        str(binary), "fit",
        str(data),
        MEAN_FORMULA,
        "--logslope-formula", LOGSLOPE_FORMULA,
        "--z-column", "pgs_ctn_z",
        "--scale-dimensions",
        "--out", str(model),
    ]

    start = time.perf_counter()
    timed_out = False
    with log.open("wb") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        try:
            rc = proc.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                rc = proc.wait()
            timed_out = True
    elapsed = time.perf_counter() - start

    return parse_log(n, elapsed, rc, timed_out, log)


def parse_log(n: int, elapsed: float, rc: int, timed_out: bool, log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(errors="replace")
    trace_lines = [ln for ln in text.splitlines() if "DenseSpectralOperator::trace_logdet_operator" in ln]
    ext_coord_lines = [ln for ln in text.splitlines() if "reml_laml ext_coord_trace" in ln]

    def extract_seconds(ln: str) -> float:
        if "elapsed=" not in ln:
            return 0.0
        tail = ln.split("elapsed=")[-1]
        num = tail.split("s", 1)[0]
        try:
            return float(num)
        except ValueError:
            return 0.0

    trace_durations = [extract_seconds(ln) for ln in trace_lines]
    total_trace_s = sum(trace_durations)
    max_trace_s = max(trace_durations) if trace_durations else 0.0

    converged = "converged=true" in text
    return {
        "n": n,
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "trace_calls": len(trace_lines),
        "trace_total_s": total_trace_s,
        "trace_max_s": max_trace_s,
        "ext_coord_traces": len(ext_coord_lines),
        "converged": converged,
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    binary = resolve_binary(args.binary)
    print(f"# binary: {binary}", file=sys.stderr)
    print(f"# budget: {args.budget_sec}s/fit", file=sys.stderr)

    rows = []
    for n in args.sizes:
        row = run_one(binary, n, outdir, args.budget_sec, args.seed)
        rows.append(row)
        print(
            f"n={row['n']:>5d} rc={row['rc']:>3d} timed_out={row['timed_out']!s:5} "
            f"elapsed={row['elapsed_s']:>7.2f}s "
            f"trace_calls={row['trace_calls']:>4d} "
            f"ext_coord_traces={row['ext_coord_traces']:>4d} "
            f"trace_total={row['trace_total_s']:>7.2f}s "
            f"trace_max={row['trace_max_s']:>5.2f}s "
            f"converged={row['converged']!s}",
            flush=True,
        )

    summary = outdir / "summary.tsv"
    keys = ["n", "rc", "timed_out", "elapsed_s", "trace_calls", "ext_coord_traces", "trace_total_s", "trace_max_s", "converged"]
    with summary.open("w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"# summary: {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
