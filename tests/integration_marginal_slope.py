#!/usr/bin/env python3
"""End-to-end integration test for marginal-slope and latent-variable configs."""

import csv, math, os, random, shutil, subprocess, sys, tempfile

GAM_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "gam")


def generate_data(n=400, seed=42):
    rng = random.Random(seed)
    raw_z = [rng.gauss(0, 1) for _ in range(n)]
    # Exactly standardize z (population sd, not sample sd, to match the check)
    mu_z = sum(raw_z) / n
    sd_z = (sum((x - mu_z)**2 for x in raw_z) / n)**0.5
    zs = [(x - mu_z) / sd_z for x in raw_z]

    rows = []
    for i in range(n):
        z = zs[i]
        sex = 1 if rng.random() < 0.5 else 0
        bmi = 22 + rng.gauss(0, 3)  # continuous covariate
        age_entry = 40 + rng.random() * 10
        eta = 0.5 * z + 0.1 * z**2 * (1 if z > 0 else -1) + 0.3 * sex + 0.02 * (bmi - 25) - 0.8
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
        rows.append(dict(z=round(z, 8), sex=sex, bmi=round(bmi, 4),
                         age_entry=round(age_entry, 4),
                         age_exit=round(age_exit, 4),
                         event=event, disease=disease))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(args, label, expect_ok=True, timeout=240):
    cmd = [GAM_BIN] + args
    short = " ".join(cmd[:7])
    print(f"\n{'='*60}\nTEST: {label}\nCMD:  {short}...\n{'='*60}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return False
    ok = (r.returncode == 0) == expect_ok
    if not ok:
        print(f"{'FAIL' if expect_ok else 'UNEXPECTED'} (exit {r.returncode})")
        for line in r.stderr.strip().split("\n")[-4:]:
            print(f"  {line}")
    else:
        print(f"OK (exit {r.returncode})")
        for line in r.stdout.strip().split("\n")[-2:]:
            print(f"  {line}")
    return ok


def main():
    if not os.path.exists(GAM_BIN):
        print(f"Binary not found: {GAM_BIN}\nRun: cargo build --release")
        sys.exit(1)

    rows = generate_data(300, 42)
    tmpdir = tempfile.mkdtemp(prefix="gam_test_")
    dp = os.path.join(tmpdir, "data.csv")
    write_csv(rows, dp)
    ne = sum(r["event"] for r in rows)
    nd = sum(r["disease"] for r in rows)
    print(f"Data: {len(rows)} rows, {ne} events ({ne*100//len(rows)}%), {nd} disease ({nd*100//len(rows)}%)")

    R = {}
    def mp(name): return os.path.join(tmpdir, f"{name}.gam")

    # ── Bernoulli marginal-slope ─────────────────────────────────────
    def bern(name, extra):
        R[name] = run(
            ["fit", dp, "disease ~ s(bmi)", "--z-column", "z",
             "--logslope-formula", "1",
             "--out", mp(name)] + extra, name)

    bern("bern_rigid", ["--disable-score-warp", "--disable-link-dev"])
    bern("bern_scorewarp", ["--disable-link-dev"])
    bern("bern_frailty03",
         ["--frailty-kind", "gaussian-shift", "--frailty-sd", "0.3",
          "--disable-score-warp", "--disable-link-dev"])
    bern("bern_sw_frailty02",
         ["--frailty-kind", "gaussian-shift", "--frailty-sd", "0.2",
          "--disable-link-dev"])

    # ── Standard binomial ────────────────────────────────────────────
    R["binom_logit"] = run(
        ["fit", dp, "disease ~ s(z) + s(bmi)", "--out", mp("binom_logit")],
        "binom_logit")

    # ── Survival configs ─────────────────────────────────────────────
    surv = "Surv(age_entry, age_exit, event)"

    def srv(name, rhs, extra):
        R[name] = run(
            ["fit", dp, f"{surv} ~ {rhs}", "--out", mp(name)] + extra,
            name, timeout=180)

    srv("surv_locscale", "s(z) + s(bmi)",
        ["--survival-likelihood", "location-scale"])

    srv("surv_ms_rigid", "s(bmi)",
        ["--z-column", "z", "--logslope-formula", "1",
         "--survival-likelihood", "marginal-slope",
         "--disable-score-warp", "--disable-link-dev"])

    srv("surv_ms_frailty03", "s(bmi)",
        ["--z-column", "z", "--logslope-formula", "1",
         "--survival-likelihood", "marginal-slope",
         "--frailty-kind", "gaussian-shift", "--frailty-sd", "0.3",
         "--disable-score-warp", "--disable-link-dev"])

    srv("surv_latent", "s(z) + s(bmi)",
        ["--survival-likelihood", "latent",
         "--frailty-kind", "hazard-multiplier", "--frailty-sd", "0.5",
         "--hazard-loading", "full",
         "--baseline-target", "gompertz",
         "--baseline-rate", "0.001", "--baseline-shape", "0.08"])

    # ── Predict on saved models ──────────────────────────────────────
    for name in list(R.keys()):
        if R[name] and os.path.exists(mp(name)):
            pred_out = os.path.join(tmpdir, f"{name}_pred.csv")
            R[f"pred_{name}"] = run(
                ["predict", mp(name), dp, "--out", pred_out],
                f"predict:{name}")

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
