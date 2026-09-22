#!/usr/bin/env python3
"""Regenerate the gam#2979 / gam#2928 survival marginal-slope workload as a CSV
plus the two fit-request documents the A/B arms use.

EVERY parameter below is quoted from gam#2979's "Workload" section. One thing that
section does NOT state is where each row's uniform draw comes from, and that is the
one degree of freedom here: see FIDELITY in README.md. `--draw` selects it, and the
script prints the event count so the choice is checkable against the issue's 691.

Usage:
    python3 generate_truth2370.py --rows 2000 --out-dir ./t2370
    python3 generate_truth2370.py --rows 300000 --out-dir ./t300k

Writes <out-dir>/{data.csv,request_anchored.json,request_closed.json}.
"""

import argparse
import json
import math
import os

MASK = (1 << 64) - 1
SEED = 0x2370_2941           # gam#2979: "shuffled by SplitMix(0x2370_2941)"
INTERCEPT = -6.99            # q(t, x) = -6.99 + 1.706*ln t + 0.25*sex
LOG_TIME = 1.706
SEX = 0.25
SLOPE = 0.8                  # "and slope 0.8"
GAMMA_SHAPE = 4              # "standardized Gamma(4) mid-quantiles"


def splitmix64(seed):
    state = seed & MASK

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    return nxt


def gamma4_cdf(x):
    return 1.0 - math.exp(-x) * (1.0 + x + x * x / 2.0 + x * x * x / 6.0)


def gamma4_icdf(p):
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gamma4_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_icdf(p):
    """Acklam's inverse normal, refined by one Halley step against erfc."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    err = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = err * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def scores(rows):
    """Standardized Gamma(4) mid-quantiles, shuffled by SplitMix(SEED)."""
    mids = [(i + 0.5) / rows for i in range(rows)]
    z = [(gamma4_icdf(p) - GAMMA_SHAPE) / math.sqrt(GAMMA_SHAPE) for p in mids]
    nxt = splitmix64(SEED)
    for i in range(rows - 1, 0, -1):
        j = nxt() % (i + 1)
        z[i], z[j] = z[j], z[i]
    return z


def uniforms(rows, draw):
    if draw == "midquantile":
        return [(i + 0.5) / rows for i in range(rows)]
    nxt = splitmix64(SEED ^ 0x5851F42D4C957F2D)
    return [((nxt() >> 11) + 0.5) / float(1 << 53) for _ in range(rows)]


def build(rows, draw):
    z = scores(rows)
    u = uniforms(rows, draw)
    out, events = [], 0
    for i in range(rows):
        sex = i % 2
        entry = 20.0 + (i % 5)                 # "Entry ages are 20 + i%5"
        admin = 40.0 + (i % 13)                # "administrative censoring is at 40 + i%13"
        # Probit marginal index: Phi^-1(F(t | x, z)) = q(t, x) + slope*z.
        t = math.exp((normal_icdf(u[i]) - SLOPE * z[i] + (-INTERCEPT) - SEX * sex) / LOG_TIME)
        observed = entry < t <= admin
        exit_age = t if observed else admin
        if exit_age <= entry:                  # left-truncated away; censor at entry+eps
            exit_age = entry + 1.0e-6
            observed = False
        events += int(observed)
        out.append((entry, exit_age, int(observed), sex, z[i], 1.0))
    return out, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--draw", choices=["midquantile", "splitmix"], default="midquantile")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    table, events = build(args.rows, args.draw)

    csv_path = os.path.join(args.out_dir, "data.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("age_entry,age_exit,event,sex,score,w\n")
        for entry, exit_age, event, sex, score, weight in table:
            handle.write(
                f"{entry:.17e},{exit_age:.17e},{event},{sex},{score:.17e},{weight:.17e}\n"
            )

    anchor = min(row[0] for row in table)      # "survival_time_anchor = earliest entry"
    scale = sum(row[1] for row in table) / len(table)   # "scale = mean exit age"

    def request(measure):
        return {
            "schema": "gam.fit-request",
            "schema_version": 1,
            "formula": "Surv(age_entry, age_exit, event) ~ sex",
            "config": {
                "slope_formula": "1",
                "z_column": "score",
                "time_basis": "ispline",
                "time_degree": 3,
                "time_num_internal_knots": 4,
                "baseline_target": "weibull",
                "baseline_scale": scale,
                "baseline_shape": 1.0,
                "survival_time_anchor": anchor,
                "weights": "w",
                "latent_measure": measure,
            },
        }

    for name, measure in (("anchored", "global-empirical"), ("closed", "standard-normal")):
        path = os.path.join(args.out_dir, f"request_{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(request(measure), handle, indent=2)
            handle.write("\n")

    print(f"rows={args.rows} draw={args.draw} events={events} "
          f"anchor={anchor:.6f} baseline_scale={scale:.6f} -> {args.out_dir}")
    print("gam#2979 records 691 events at rows=2000; see FIDELITY in README.md if this differs.")


if __name__ == "__main__":
    main()
