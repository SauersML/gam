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


def _is_finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _validate_prediction_output(name, path):
    """Read the prediction CSV produced by gam predict and assert it
    looks like a valid probability or survival output. Raises RuntimeError
    on contract violation so the caller treats it as a failed run."""
    import csv as _csv

    with open(path, "r", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        rows_out = list(reader)
    if not rows_out:
        raise RuntimeError(f"{name} produced an empty prediction output")
    header = list(rows_out[0].keys())

    # Pick the most informative column we can verify. Bernoulli marginal
    # slope predicts a probability (`mean` or `prob`); survival marginal
    # slope predicts a survival probability or hazard.
    candidates = ["mean", "prob", "survival_prob"]
    column = next((c for c in candidates if c in header), None)
    if column is None:
        # Nothing we can verify numerically; skip rather than fail because
        # some predict modes emit only `eta`. Print so an integrator can see.
        print(f"  [warn] {name} prediction has columns {header}; no numeric "
              f"contract enforced")
        return

    values = []
    for r in rows_out:
        v = r.get(column)
        if not _is_finite(v):
            raise RuntimeError(
                f"{name} prediction column {column} has non-finite entry: {v}"
            )
        values.append(float(v))

    if not values:
        raise RuntimeError(f"{name} prediction column {column} was empty")

    # Probability columns must sit in [0, 1].
    out_of_band = [(i, v) for i, v in enumerate(values) if v < -1e-9 or v > 1.0 + 1e-9]
    if out_of_band:
        raise RuntimeError(
            f"{name} prediction column {column} has {len(out_of_band)} values "
            f"outside [0,1]; first: {out_of_band[:3]}"
        )

    # Spread sanity: a model that collapsed to a constant prediction is
    # uninformative, even if every value is in [0, 1]. Require a non-trivial
    # range across the test rows for marginal-slope contenders.
    spread = max(values) - min(values)
    if spread < 1e-4:
        raise RuntimeError(
            f"{name} prediction column {column} is essentially constant "
            f"(range={spread:.2e}); model has not learned anything"
        )


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
            # Numeric contract: a successful predict run must produce
            # well-formed probability/survival output. A model that always
            # emits `mean = 0.5` would still exit 0; the structured check
            # below catches that.
            if R[f"pred_{name}"] and os.path.exists(out):
                _validate_prediction_output(name, out)

    # ── Bernoulli marginal-slope ─────────────────────────────────────
    bern("bern_rigid",
         "1",
         ["--disable-score-warp", "--disable-link-dev"])
    bern("bern_scorewarp",
         "1 + linkwiggle(internal_knots=6)",
         ["--disable-link-dev"])
    bern("bern_frailty",
         "1",
         ["--frailty-kind", "gaussian-shift", "--frailty-sd", "0.3",
          "--disable-score-warp", "--disable-link-dev"])
    bern("bern_sw_frailty",
         "1 + linkwiggle(internal_knots=6)",
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
         logslope_formula="1 + linkwiggle(internal_knots=6)")
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
