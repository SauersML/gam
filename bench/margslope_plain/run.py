"""Plain marginal-slope Duchon perf probe.

Target spec (no scale_dimensions, no linkwiggle, no score-warp, no survival):

    train  : columns = [case, sex, prs_z, PC1..PC10]
    formula: case ~ duchon(PC1..PC10, centers=40, order=1, power=2, length_scale=1)
                  + sex
    link   : probit
    logslope: duchon(PC1..PC10, centers=40, order=1, power=2, length_scale=1)
    z-column: prs_z

Outputs:
  - <outdir>/dch_<N>.csv          (synthetic data)
  - <outdir>/dch_<N>.log          (full gam fit stdout/stderr)
  - <outdir>/summary.tsv          (n, elapsed_s, converged, trace_calls, ...)

Per-fit wall budget kills the process at the deadline so we get a clean
total even when convergence is too slow to be useful.

Typical use:
    python3 bench/margslope_plain/run.py \\
        --binary target/release/gam \\
        --outdir bench/margslope_plain/runs \\
        --sizes 200 \\
        --budget-sec 70
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

PC_COLS = ", ".join(f"PC{j}" for j in range(1, 11))
DUCHON_TERM = (
    f"duchon({PC_COLS}, centers=40, order=1, power=2, length_scale=1)"
)
MEAN_FORMULA = f"case ~ link(type=probit) + sex + {DUCHON_TERM}"
LOGSLOPE_FORMULA = DUCHON_TERM


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--binary", help="Path to gam binary; defaults to target/release/gam")
    ap.add_argument("--outdir", required=True, help="Output dir for CSVs / logs / summary")
    ap.add_argument("--sizes", nargs="+", type=int, required=True, help="n values to probe")
    ap.add_argument("--budget-sec", type=float, default=70.0, help="Per-fit wall-clock budget")
    ap.add_argument("--seed", type=lambda v: int(v, 0), default=0xA110CA7E)
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cmd = [
        str(binary), "fit",
        str(data),
        MEAN_FORMULA,
        "--logslope-formula", LOGSLOPE_FORMULA,
        "--z-column", "prs_z",
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
    trace_lines = [
        ln for ln in text.splitlines()
        if "DenseSpectralOperator::trace_logdet_operator" in ln
    ]
    ext_coord_lines = [
        ln for ln in text.splitlines() if "reml_laml ext_coord_trace" in ln
    ]
    outer_iter_lines = [ln for ln in text.splitlines() if "[OUTER]" in ln]
    pirls_iter_lines = [ln for ln in text.splitlines() if "[PIRLS]" in ln]

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
    return {
        "n": n,
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "trace_calls": len(trace_lines),
        "trace_total_s": sum(trace_durations),
        "trace_max_s": max(trace_durations) if trace_durations else 0.0,
        "ext_coord_traces": len(ext_coord_lines),
        "outer_iters": len(outer_iter_lines),
        "pirls_msgs": len(pirls_iter_lines),
        "converged": "converged=true" in text,
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
            f"trace_total={row['trace_total_s']:>7.2f}s "
            f"trace_max={row['trace_max_s']:>5.2f}s "
            f"outer_iters={row['outer_iters']:>3d} "
            f"converged={row['converged']!s}",
            flush=True,
        )

    summary = outdir / "summary.tsv"
    keys = [
        "n", "rc", "timed_out", "elapsed_s", "trace_calls",
        "ext_coord_traces", "trace_total_s", "trace_max_s",
        "outer_iters", "pirls_msgs", "converged",
    ]
    with summary.open("w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"# summary: {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
