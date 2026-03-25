#!/usr/bin/env python3
"""End-to-end integration test for marginal-slope and latent survival configs."""

import csv, math, os, random, shutil, subprocess, sys, tempfile

GAM_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "gam")


def generate_data(n=300, seed=42):
    rng = random.Random(seed)
    raw_z = [rng.gauss(0, 1) for _ in range(n)]
    mu_z = sum(raw_z) / n
    sd_z = (sum((x - mu_z)**2 for x in raw_z) / n)**0.5
    zs = [(x - mu_z) / sd_z for x in raw_z]

    rows = []
    for i in range(n):
        z = zs[i]
        bmi = 22 + rng.gauss(0, 3)
        age_entry = 40 + rng.random() * 10
        eta = 0.5 * z + 0.1 * z**2 * (1 if z > 0 else -1) + 0.02 * (bmi - 25) - 0.8
        h = 0.015 * math.exp(0.06 * (age_entry - 45) + eta)
        follow = 3 + rng.random() * 7
        age_exit = age_entry + follow
        cum = h * follow
        p_event = 1 - math.exp(-cum)
        event = 1 if rng.random() < p_event else 0
        if event:
            u = rng.random()
            t = -math.log(1 - u * p_event) / max(h, 1e-12)
            age_exit = age_entry + max(min(t, follow - 0.01), 0.01)
        p_dis = 1 - math.exp(-h * max(60 - age_entry, 0.1))
        disease = 1 if rng.random() < p_dis else 0
        rows.append(dict(z=round(z, 8), bmi=round(bmi, 4),
                         age_entry=round(age_entry, 4),
                         age_exit=round(age_exit, 4),
                         event=event, disease=disease))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(args, label, timeout=240):
    cmd = [GAM_BIN] + args
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({timeout}s)")
        return False
    if r.returncode != 0:
        print(f"  FAIL (exit {r.returncode})")
        for line in r.stderr.strip().split("\n")[-3:]:
            print(f"  {line}")
        return False
    print(f"  OK")
    for line in r.stdout.strip().split("\n")[-2:]:
        print(f"  {line}")
    return True


def main():
    if not os.path.exists(GAM_BIN):
        print(f"Binary not found: {GAM_BIN}")
        sys.exit(1)

    rows = generate_data(300, 42)
    tmpdir = tempfile.mkdtemp(prefix="gam_test_")
    dp = os.path.join(tmpdir, "data.csv")
    write_csv(rows, dp)
    ne = sum(r["event"] for r in rows)
    nd = sum(r["disease"] for r in rows)
    print(f"Data: {len(rows)} rows, {ne} events ({100*ne//len(rows)}%), {nd} disease ({100*nd//len(rows)}%)")

    R = {}
    def mp(name): return os.path.join(tmpdir, f"{name}.gam")

    def bern(name, logslope_formula, extra):
        R[name] = run(
            ["fit", dp, "disease ~ s(bmi)", "--z-column", "z",
             "--logslope-formula", logslope_formula, "--out", mp(name)] + extra, name)

    def surv(name, rhs, extra, logslope_formula=None):
        args = ["fit", dp, f"Surv(age_entry, age_exit, event) ~ {rhs}"]
        if logslope_formula is not None:
            args += ["--logslope-formula", logslope_formula]
        args += ["--out", mp(name)] + extra
        R[name] = run(args, name)

    def pred(name):
        if R.get(name) and os.path.exists(mp(name)):
            out = os.path.join(tmpdir, f"{name}_pred.csv")
            R[f"pred_{name}"] = run(
                ["predict", mp(name), dp, "--out", out], f"predict:{name}")

    # ── Bernoulli marginal-slope ─────────────────────────────────────
    bern("bern_rigid",
         "1",
         ["--disable-score-warp", "--disable-link-dev"])
    bern("bern_scorewarp",
         "1 + linkwiggle(knots=6)",
         ["--disable-link-dev"])
    bern("bern_frailty",
         "1",
         ["--frailty-kind", "gaussian-shift", "--frailty-sd", "0.3",
          "--disable-score-warp", "--disable-link-dev"])
    bern("bern_sw_frailty",
         "1 + linkwiggle(knots=6)",
         ["--frailty-kind", "gaussian-shift", "--frailty-sd", "0.2",
          "--disable-link-dev"])

    # ── Survival marginal-slope ──────────────────────────────────────
    surv("surv_ms_rigid", "s(bmi)",
         ["--z-column", "z",
          "--survival-likelihood", "marginal-slope",
          "--disable-score-warp", "--disable-link-dev"],
         logslope_formula="1")
    surv("surv_ms_scorewarp", "s(bmi)",
         ["--z-column", "z",
          "--survival-likelihood", "marginal-slope",
          "--disable-link-dev"],
         logslope_formula="1 + linkwiggle(knots=6)")
    surv("surv_ms_frailty", "s(bmi)",
         ["--z-column", "z",
          "--survival-likelihood", "marginal-slope",
          "--frailty-kind", "gaussian-shift", "--frailty-sd", "0.3",
          "--disable-score-warp", "--disable-link-dev"],
         logslope_formula="1")

    # ── Latent survival (PH kernel) ──────────────────────────────────
    surv("surv_latent", "z + bmi",
         ["--survival-likelihood", "latent",
          "--frailty-kind", "hazard-multiplier", "--frailty-sd", "0.5",
          "--hazard-loading", "full",
          "--baseline-target", "gompertz"])

    # ── Predict ──────────────────────────────────────────────────────
    for name in [
        "bern_rigid",
        "bern_scorewarp",
        "bern_frailty",
        "bern_sw_frailty",
        "surv_ms_rigid",
        "surv_ms_scorewarp",
        "surv_ms_frailty",
    ]:
        pred(name)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    total = len(R)
    passed = sum(R.values())
    for name, ok in R.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{passed}/{total} passed")

    shutil.rmtree(tmpdir, ignore_errors=True)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
