"""Table of gpu_bench.sh runs (#2951): one row per card from ~/mpd-data/runpod/NAME/.

    python bench/runpod/gpu_bench_table.py NAME...

Per card: the price per hour (the ledger's dollars over hours) and, for each step configuration
the run measured, the library step with the posterior on the device (seconds, scored tokens per
second, millions of scored tokens per dollar, peak device GiB; scored tokens are the experiments'
tokens, two experiments per base sequence): vpd4l at 8 and 32 sequences of 512 tokens with the
program engine in f32 products (in the first, f32 baseline runs the parts were vpd4l_step_s8 and
vpd4l_step_s32) and with the decoder engine in bfloat16 products, and Qwen3-0.6B at 8 sequences of
512 tokens with the decoder engine in bfloat16. Then the decoder's blocks' product rates (TFLOP/s,
forward and reverse, vpd4l at 32 sequences) and the CUDA parity tests' passed and failed counts.
"""

import csv
import json
import re
import sys
from pathlib import Path

RUNPOD = Path("/Users/user/mpd-data/runpod")
CONFIGS = [
    ("vpd4l f32 s8", ["vpd4l_program_f32_s8", "vpd4l_step_s8"]),
    ("vpd4l f32 s32", ["vpd4l_program_f32_s32", "vpd4l_step_s32"]),
    ("vpd4l bf16 s8", ["vpd4l_decoder_bf16_s8"]),
    ("vpd4l bf16 s32", ["vpd4l_decoder_bf16_s32"]),
    ("qwen3 bf16 s8", ["qwen3_decoder_bf16_s8"]),
]


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def parts(run):
    rows = {}
    path = run / "parts.tsv"
    if path.exists():
        for row in csv.DictReader(path.open(), delimiter="\t"):
            rows[row["part"]] = row
    return rows


def price(name):
    for row in csv.DictReader((RUNPOD / "LEDGER.tsv").open(), delimiter="\t"):
        if row["name"] == name and float(row["hours"]) > 0.02:
            return float(row["dollars"]) / float(row["hours"]), row["gpu"], row["cloud"]
    return None, name, "?"


def step(run, part):
    report = load(run / part / "library_step_bench.json")
    if not report or not report.get("scored"):
        return None
    seconds = report["device_posterior_seconds"]["step"]
    return report, seconds, report["experiments"] * report["context"] / seconds


def tests(run):
    text = "".join(p.read_text() for p in sorted(run.glob("gpu_tests.out")) + sorted(run.glob("test_*.out")))
    found = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed", text)
    if not found:
        return "not run"
    return f"{sum(int(p) for p, _ in found)} passed, {sum(int(f) for _, f in found)} failed"


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def main():
    header = ["card", "cloud", "$/h"]
    for label, _ in CONFIGS:
        header += [f"{label} s", f"{label} tok/s", f"{label} Mtok/$", f"{label} GiB"]
    header += ["blocks fwd/rev TFLOP/s", "parity tests"]
    print("\t".join(header))
    for name in sys.argv[1:]:
        run = RUNPOD / name
        dollars, gpu, cloud = price(name)
        peaks = parts(run)
        row = [gpu, cloud, fmt(dollars, ".2f")]
        blocks = None
        for label, names in CONFIGS:
            found = next(((n, step(run, n)) for n in names if step(run, n)), (None, None))
            part, measured = found
            if measured is None:
                row += ["-"] * 4
                continue
            report, seconds, rate = measured
            if label == "vpd4l bf16 s32" and report.get("decoder_blocks"):
                blocks = report["decoder_blocks"]
            peak = peaks.get(part, {}).get("peak_mib")
            row += [fmt(seconds, ".3f"), fmt(rate, ".0f"), fmt(dollars and rate * 3600 / dollars / 1e6, ".0f"), fmt(peak and float(peak) / 1024, ".1f")]
        row.append("-" if not blocks else f"{blocks['forward_tflops']:.0f}/{blocks['reverse_tflops']:.0f}")
        row.append(tests(run))
        print("\t".join(row))


if __name__ == "__main__":
    main()
