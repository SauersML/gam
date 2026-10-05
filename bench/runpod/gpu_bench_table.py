"""Table of gpu_bench.sh runs (#2951): one row per card from ~/mpd-data/runpod/NAME/.

    python bench/runpod/gpu_bench_table.py NAME...

Per card: the price per hour (the ledger's dollars over hours), the library step on vpd4l with the
posterior on the device at 8 and 32 sequences of 512 tokens (seconds, scored tokens per second and
per dollar; scored tokens are the experiments' tokens, two experiments per base sequence), its peak
device memory, Qwen3-0.6B's forward plus reverse pass at 16 sequences of 128 tokens in f32 and TF32
products (tokens per second), the interchange step on vpd4l and on Qwen3-0.6B, and the CUDA parity
tests' passed and failed counts.
"""

import csv
import json
import re
import sys
from pathlib import Path

RUNPOD = Path("/Users/user/mpd-data/runpod")


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
    ledger = RUNPOD / "LEDGER.tsv"
    for row in csv.DictReader(ledger.open(), delimiter="\t"):
        if row["name"] == name and float(row["hours"]) > 0:
            return float(row["dollars"]) / float(row["hours"]), row["gpu"], row["cloud"]
    pod = load(RUNPOD / name / "pod.json") or {}
    return pod.get("costPerHr"), (pod.get("machine") or {}).get("gpuTypeId"), "?"


def library_step(run, sequences):
    report = load(run / f"vpd4l_step_s{sequences}" / "library_step_bench.json")
    if not report or not report.get("scored"):
        return None
    seconds = report["device_posterior_seconds"]["step"]
    return seconds, report["experiments"] * report["context"] / seconds


def passes(run, arithmetic, sequences=16):
    """Forward plus reverse at SEQUENCES: from step.json, or from the per-batch log lines when a
    larger batch ran out of device memory before the report was written."""
    report = load(run / f"qwen3_passes_{arithmetic}" / "step.json")
    batches = (report or {}).get("batches")
    if batches is None:
        err = run / f"qwen3_passes_{arithmetic}.err"
        text = err.read_text() if err.exists() else ""
        batches = [json.loads(m) for m in re.findall(r"step bench: (\{.*\})", text)]
    for batch in batches:
        if batch["sequences"] == sequences:
            seconds = batch["forward_seconds"] + batch["reverse_seconds"]
            return seconds, batch["tokens"] / seconds, batch["used_bytes_peak"] / 2**30
    return None


def interchange(run, name):
    report = load(run / name / "interchange_bench.json")
    if not report:
        return None
    seconds = report["step_with_gradient"]["median_seconds"]
    return seconds, report["experiments"] * report["context"] / seconds


def tests(run):
    text = (run / "gpu_tests.out").read_text() if (run / "gpu_tests.out").exists() else ""
    found = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed", text)
    if not found:
        return "not run"
    return f"{sum(int(p) for p, _ in found)} passed, {sum(int(f) for _, f in found)} failed"


def cell(value, fmt):
    return "-" if value is None else fmt.format(value)


def main():
    header = ["card", "cloud", "$/h", "vpd4l s8 step s", "s8 tok/s", "vpd4l s32 step s", "s32 tok/s", "s32 Mtok/$",
              "s32 peak GiB", "vpd4l ic s8 tok/s", "qwen3 s16 f32 tok/s", "qwen3 s16 tf32 tok/s", "qwen3 s16 peak GiB",
              "qwen3 ic s4 tok/s", "parity tests"]
    print("\t".join(header))
    for name in sys.argv[1:]:
        run = RUNPOD / name
        dollars, gpu, cloud = price(name)
        rows = parts(run)
        s8, s32 = library_step(run, 8), library_step(run, 32)
        f32, tf32 = passes(run, "f32"), passes(run, "tf32")
        ic_vpd, ic_qwen = interchange(run, "vpd4l_interchange_s8"), interchange(run, "qwen3_interchange_s4")
        peak = rows.get("vpd4l_step_s32", {}).get("peak_mib")
        print("\t".join([
            gpu or name, cloud, cell(dollars, "{:.2f}"),
            cell(s8 and s8[0], "{:.3f}"), cell(s8 and s8[1], "{:.0f}"),
            cell(s32 and s32[0], "{:.3f}"), cell(s32 and s32[1], "{:.0f}"),
            cell(s32 and dollars and s32[1] * 3600 / dollars / 1e6, "{:.1f}"),
            cell(peak and float(peak) / 1024, "{:.1f}"),
            cell(ic_vpd and ic_vpd[1], "{:.0f}"),
            cell(f32 and f32[1], "{:.0f}"), cell(tf32 and tf32[1], "{:.0f}"), cell(f32 and f32[2], "{:.1f}"),
            cell(ic_qwen and ic_qwen[1], "{:.0f}"),
            tests(run),
        ]))


if __name__ == "__main__":
    main()
