"""The thin Python API preserves Rust's bit-reproducible fit contract."""

from __future__ import annotations

import os
import subprocess
import sys


def test_python_fits_are_bit_identical_across_rayon_thread_counts_and_runs() -> None:
    program = r'''
import json, math
import gamfit

def frame(kind):
    out = {"x": []}
    out.update({"time": [], "event": []} if kind == "survival" else {"y": []})
    for i in range(240):
        x = -2.4 + 4.8 * i / 239.0
        wave = math.sin(2.1 * x) + 0.15 * math.cos(5.3 * x)
        out["x"].append(x)
        if kind == "gaussian": out["y"].append(wave + 0.07 * ((i * 37 % 17) - 8.0))
        elif kind == "binomial": out["y"].append(float((i * 53 % 101) / 101.0 < 1.0 / (1.0 + math.exp(-wave))))
        else:
            latent = math.exp(1.3 - 0.45 * x + 0.12 * wave)
            censor = 5.0 + (i * 29 % 31) / 5.0
            out["time"].append(min(latent, censor)); out["event"].append(float(latent <= censor))
    return out

kind = __import__('os').environ['GAM_REPRO_KIND']
if kind == "survival":
    model = gamfit.fit(frame(kind), "Surv(time, event) ~ s(x, k=8)",
                       survival_likelihood="transformation")
else:
    model = gamfit.fit(frame(kind), "y ~ s(x, k=10)", family=kind)
summary = model.summary()
coef = summary.coefficients_frame()["estimate"].tolist()
print("RESULT " + json.dumps({"coefficients": [v.hex() for v in coef],
    "lambdas": [v.hex() for v in summary.lambdas],
    "log_likelihood": summary.log_likelihood.hex()}, sort_keys=True))
'''
    for kind in ("gaussian", "binomial", "survival"):
        expected = None
        for threads in (1, 2, 8):
            for _run in range(2):
                env = os.environ.copy()
                env.update(RAYON_NUM_THREADS=str(threads), GAM_REPRO_KIND=kind)
                completed = subprocess.run(
                    [sys.executable, "-c", program], env=env, text=True,
                    capture_output=True, check=True,
                )
                result = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT "))
                if expected is None:
                    expected = result
                assert result == expected, f"{kind} changed with RAYON_NUM_THREADS={threads}"
