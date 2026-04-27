#!/usr/bin/env python3
"""End-to-end PIT pipeline: transformation-normal fit -> predict z -> marginal-slope fits.

Duchon smooths throughout, main-formula linkwiggle, logslope score-warp via
logslope-formula linkwiggle, and timewiggle — no linear terms. PIT is done
entirely by the gam binary via --transformation-normal.
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
        rows.append({
            "pc1": f"{pc[0]:.9f}",
            "pc2": f"{pc[1]:.9f}",
            "pc3": f"{pc[2]:.9f}",
            "pc4": f"{pc[3]:.9f}",
            "pgs": f"{pgs:.12f}",
        })
    return rows


def generate_study(n=40, seed=2):
    """Study cohort with binary + survival outcomes."""
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

        age0 = 40 + rng.random() * 10
        hazard = 0.02 * math.exp(0.05 * (age0 - 45) + 0.4 * latent_z)
        follow = 3 + rng.random() * 7
        cum = hazard * follow
        p_event = 1 - math.exp(-cum)
        event = 1 if rng.random() < p_event else 0
        if event:
            u = rng.random()
            t = -math.log(1 - u * p_event) / max(hazard, 1e-12)
            age1 = age0 + max(min(t, follow - 0.01), 0.01)
        else:
            age1 = age0 + follow

        rows.append({
            "pc1": f"{pc[0]:.9f}",
            "pc2": f"{pc[1]:.9f}",
            "pc3": f"{pc[2]:.9f}",
            "pc4": f"{pc[3]:.9f}",
            "pgs": f"{pgs:.12f}",
            "case": str(case),
            "age0": f"{age0:.4f}",
            "age1": f"{age1:.4f}",
            "event": str(event),
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def merge_z_into_study(study_csv, pred_csv, out_csv):
    """Read predict output (eta column = PIT z), merge as 'z' into study data.

    The merged z scores should be approximately N(0, 1) under conditional
    Gaussianization on the PC manifold. We assert this here so a regression
    that produced biased or improperly scaled z-scores would be flagged
    immediately rather than silently propagating through downstream fits.
    """
    study = read_csv(study_csv)
    preds = read_csv(pred_csv)
    if len(preds) != len(study):
        raise RuntimeError(
            f"PIT prediction row mismatch: got {len(preds)} rows for {len(study)} study rows"
        )
    for s, p in zip(study, preds):
        s["z"] = p["eta"]
    write_csv(study, out_csv)
    zs = [float(p["eta"]) for p in preds]
    n = len(zs)
    z_mean = sum(zs) / n
    z_sd = (sum((z - z_mean) ** 2 for z in zs) / n) ** 0.5
    print(f"  PIT z: n={n}, mean={z_mean:.4f}, sd={z_sd:.4f}, range=[{min(zs):.3f}, {max(zs):.3f}]")

    # Numeric contracts on the PIT z-scores. Failures raise so the pipeline
    # script returns non-zero — silent drift toward a bias of 0.5 or
    # collapse to a near-constant z would otherwise flow into fits 2 and 4
    # and corrupt downstream metrics.
    if not all(_finite(z) for z in zs):
        raise RuntimeError(
            f"PIT z column contained non-finite values; first non-finite: "
            f"{[(i, zs[i]) for i in range(n) if not _finite(zs[i])][:3]}"
        )
    # Mean should be near 0 and standard deviation should be near 1. The
    # tolerance is loose because the test datasets are small (~40 rows),
    # but a runaway bias or near-constant prediction would still fail.
    if abs(z_mean) > 0.5:
        raise RuntimeError(
            f"PIT z mean {z_mean:.4f} drifted too far from zero; expected |mean| < 0.5"
        )
    if not (0.5 < z_sd < 1.8):
        raise RuntimeError(
            f"PIT z sd {z_sd:.4f} outside expected (0.5, 1.8); expected ≈ 1.0"
        )

    # Spread sanity: at least 60% of z values should sit within ±2.5 of zero
    # (true under N(0,1) for >98%, but small-sample noise + tail outliers
    # justify a permissive band).
    inliers = sum(1 for z in zs if abs(z) < 2.5)
    if inliers < 0.6 * n:
        raise RuntimeError(
            f"only {inliers}/{n} PIT z values lie within ±2.5; distribution does "
            "not look like N(0,1)"
        )


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def run(args, label, timeout=300):
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


def skip(label, reason):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"  SKIP ({reason})")
    return False, 0.0


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
    n_event = sum(int(row["event"]) for row in study_rows)
    n_study = len(study_rows)
    print(f"Reference: {len(ref_rows)} rows")
    print(f"Study: {n_study} rows, {n_case} cases ({100.0 * n_case / n_study:.1f}%), {n_event} events ({100.0 * n_event / n_study:.1f}%)")

    timings = {}
    results = {}

    # ── Fit 1: PIT via transformation-normal (Duchon on PCs) ─────────
    fit1_model = os.path.join(tmpdir, "pgs_pit.json")
    ok, elapsed = run([
        "fit", ref_csv,
        "pgs ~ duchon(pc1, pc2, pc3, pc4, centers=20)",
        "--transformation-normal",
        "--scale-dimensions",
        "--out", fit1_model,
    ], "Fit 1: Transformation-normal PIT")
    results["fit1_pit"] = ok
    timings["fit1_pit"] = elapsed

    # ── Predict: PIT z-scores on study data ──────────────────────────
    fit1_pred_csv = os.path.join(tmpdir, "fit1_pred.csv")
    if results["fit1_pit"]:
        ok, elapsed = run([
            "predict", fit1_model, study_csv,
            "--out", fit1_pred_csv,
        ], "Predict: PIT z-scores on study data")
    else:
        ok, elapsed = skip(
            "Predict: PIT z-scores on study data",
            "transformation-normal fit failed",
        )
    results["pred_pit"] = ok
    timings["pred_pit"] = elapsed

    # ── Merge z into study CSV ───────────────────────────────────────
    enriched_csv = os.path.join(tmpdir, "study_with_z.csv")
    if not results["pred_pit"]:
        fit2_model = os.path.join(tmpdir, "bernoulli.json")
        fit4_model = os.path.join(tmpdir, "survival.json")
        results["fit2_bern_ms"], timings["fit2_bern_ms"] = skip(
            "Fit 2: Bernoulli marginal-slope",
            "PIT z-scores were not produced",
        )
        results["fit4_surv_ms"], timings["fit4_surv_ms"] = skip(
            "Fit 4: Survival marginal-slope (Gompertz-Makeham + timewiggle)",
            "PIT z-scores were not produced",
        )
    else:
        merge_z_into_study(study_csv, fit1_pred_csv, enriched_csv)

        # ── Fit 2: Bernoulli marginal-slope (Duchon + linkwiggle + score-warp)
        fit2_model = os.path.join(tmpdir, "bernoulli.json")
        ok, elapsed = run([
            "fit", enriched_csv,
            "case ~ duchon(pc1, pc2, pc3, pc4, centers=20) + linkwiggle(internal_knots=8)",
            "--logslope-formula",
            "duchon(pc1, pc2, pc3, pc4, centers=20) + linkwiggle(internal_knots=8)",
            "--z-column", "z",
            "--scale-dimensions",
            "--out", fit2_model,
        ], "Fit 2: Bernoulli marginal-slope")
        results["fit2_bern_ms"] = ok
        timings["fit2_bern_ms"] = elapsed

        # ── Fit 4: Survival marginal-slope (Duchon + linkwiggle + score-warp + timewiggle)
        fit4_model = os.path.join(tmpdir, "survival.json")
        ok, elapsed = run([
            "fit", enriched_csv,
            "Surv(age0, age1, event) ~ duchon(pc1, pc2, pc3, pc4, centers=20) + survmodel(spec=net, distribution=gaussian) + linkwiggle(internal_knots=8) + timewiggle(internal_knots=8)",
            "--survival-likelihood", "marginal-slope",
            "--baseline-target", "gompertz-makeham",
            "--logslope-formula",
            "duchon(pc1, pc2, pc3, pc4, centers=20) + linkwiggle(internal_knots=8)",
            "--z-column", "z",
            "--scale-dimensions",
            "--out", fit4_model,
        ], "Fit 4: Survival marginal-slope (Gompertz-Makeham + timewiggle)")
        results["fit4_surv_ms"] = ok
        timings["fit4_surv_ms"] = elapsed

    # ── Summary ───────────────────────────────────────────────────────
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
