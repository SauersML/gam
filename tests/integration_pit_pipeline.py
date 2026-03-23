#!/usr/bin/env python3
"""Fast end-to-end PIT pipeline.

This test exercises:
1. Gaussian location-scale fitting on a reference panel.
2. Prediction on a study cohort.
3. PIT-style latent score construction with explicit empirical standardization.
4. Downstream Bernoulli marginal-slope fitting.

The formulas are intentionally rigid and low-dimensional so this stays fast in
CI while still covering the PIT -> marginal-slope integration path.
"""

import csv
import math
import os
import random
import subprocess
import sys
import tempfile
import time


GAM_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "gam")


def generate_reference(n=40, seed=1):
    """Reference panel with Gaussian location-scale structure."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        pc = [rng.gauss(0, 1) for _ in range(4)]
        mu = 0.3 * pc[0] - 0.2 * pc[1] + 0.1 * pc[2]
        sigma = math.exp(0.15 * pc[0] + 0.1 * pc[3] - 0.5)
        pgs = mu + sigma * rng.gauss(0, 1)
        rows.append(
            {
                "pc1": f"{pc[0]:.9f}",
                "pc2": f"{pc[1]:.9f}",
                "pc3": f"{pc[2]:.9f}",
                "pc4": f"{pc[3]:.9f}",
                "pgs": f"{pgs:.12f}",
            }
        )
    return rows


def generate_study(n=40, seed=2):
    """Study cohort with downstream binary outcome."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        pc = [rng.gauss(0, 1) for _ in range(4)]
        mu = 0.3 * pc[0] - 0.2 * pc[1] + 0.1 * pc[2]
        sigma = math.exp(0.15 * pc[0] + 0.1 * pc[3] - 0.5)
        latent_z = rng.gauss(0, 1)
        pgs = mu + sigma * latent_z

        eta_case = 0.55 * latent_z - 1.0
        p_case = 1.0 / (1.0 + math.exp(-eta_case))
        case = 1 if rng.random() < p_case else 0

        rows.append(
            {
                "pc1": f"{pc[0]:.9f}",
                "pc2": f"{pc[1]:.9f}",
                "pc3": f"{pc[2]:.9f}",
                "pc4": f"{pc[3]:.9f}",
                "pgs": f"{pgs:.12f}",
                "case": str(case),
            }
        )
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pit_transform(study_rows, pred_rows):
    if len(study_rows) != len(pred_rows):
        raise ValueError(
            f"row count mismatch: study={len(study_rows)} predictions={len(pred_rows)}"
        )

    raw_z = []
    merged = []
    for row, pred in zip(study_rows, pred_rows):
        mean = float(pred["mean"])
        sigma = max(float(pred["sigma"]), 1e-12)
        z = (float(row["pgs"]) - mean) / sigma
        raw_z.append(z)
        merged.append((dict(row), mean, sigma, z))

    z_mean = sum(raw_z) / len(raw_z)
    z_var = sum((z - z_mean) ** 2 for z in raw_z) / len(raw_z)
    z_sd = max(math.sqrt(z_var), 1e-12)

    transformed = []
    for row, _mean, _sigma, z in merged:
        row["z"] = repr((z - z_mean) / z_sd)
        transformed.append(row)

    return transformed


def run(args, label, timeout=30):
    cmd = [GAM_BIN] + args
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"  cmd: gam {' '.join(args)}")
    started = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"  TIMEOUT ({timeout}s, elapsed={elapsed:.1f}s)")
        return False, elapsed

    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"  FAIL (exit {result.returncode}, {elapsed:.1f}s)")
        stderr = result.stderr.strip().splitlines()
        for line in stderr[-8:]:
            print(f"  stderr: {line}")
        return False, elapsed

    print(f"  OK ({elapsed:.1f}s)")
    stdout = result.stdout.strip().splitlines()
    for line in stdout[-3:]:
        print(f"  {line}")
    return True, elapsed


def main():
    if not os.path.exists(GAM_BIN):
        print(f"Binary not found: {GAM_BIN}")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="gam_pit_")
    print(f"Workdir: {tmpdir}")

    ref_rows = generate_reference(40, seed=1)
    study_rows = generate_study(40, seed=2)
    ref_csv = os.path.join(tmpdir, "reference.csv")
    study_csv = os.path.join(tmpdir, "study.csv")
    write_csv(ref_rows, ref_csv)
    write_csv(study_rows, study_csv)

    n_case = sum(int(row["case"]) for row in study_rows)
    n_study = len(study_rows)
    print(f"Reference: {len(ref_rows)} rows")
    print(f"Study: {n_study} rows, {n_case} cases ({100.0 * n_case / n_study:.1f}%)")

    timings = {}
    results = {}

    fit1_model = os.path.join(tmpdir, "pgs_dist.json")
    ok, elapsed = run(
        [
            "fit",
            ref_csv,
            "pgs ~ pc1 + pc2 + pc3 + pc4",
            "--predict-noise",
            "pc1 + pc4",
            "--out",
            fit1_model,
        ],
        "Fit 1: Gaussian location-scale reference model",
    )
    results["fit1_locsca"] = ok
    timings["fit1_locsca"] = elapsed

    fit1_pred_csv = os.path.join(tmpdir, "fit1_pred.csv")
    ok, elapsed = run(
        ["predict", fit1_model, study_csv, "--out", fit1_pred_csv],
        "Predict: reference model on study cohort",
    )
    results["pred_fit1"] = ok
    timings["pred_fit1"] = elapsed

    if results["fit1_locsca"] and results["pred_fit1"]:
        study_with_z = pit_transform(study_rows, read_csv(fit1_pred_csv))
        z_values = [float(row["z"]) for row in study_with_z]
        z_mean = sum(z_values) / len(z_values)
        z_sd = math.sqrt(sum((z - z_mean) ** 2 for z in z_values) / len(z_values))
        if abs(z_mean) > 1e-12 or abs(z_sd - 1.0) > 1e-12:
            raise RuntimeError(f"PIT z standardization drifted: mean={z_mean}, sd={z_sd}")
        print(f"PIT z summary: mean={z_mean:.3e}, sd={z_sd:.12f}")
        study_with_z_csv = os.path.join(tmpdir, "study_with_z.csv")
        write_csv(study_with_z, study_with_z_csv)
    else:
        study_with_z_csv = study_csv

    ok, elapsed = run(
        [
            "fit",
            study_with_z_csv,
            "case ~ 1",
            "--logslope-formula",
            "1",
            "--z-column",
            "z",
            "--disable-score-warp",
            "--disable-link-dev",
        ],
        "Fit 2: Bernoulli marginal-slope",
    )
    results["fit2_bern_ms"] = ok
    timings["fit2_bern_ms"] = elapsed

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    total = 0.0
    for name in results:
        passed = results[name]
        elapsed = timings[name]
        total += elapsed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:20s}  {elapsed:7.1f}s")
    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} passed   total={total:.1f}s")
    print(f"\nWorkdir: {tmpdir}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
