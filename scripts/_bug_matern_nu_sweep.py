"""Sweep matern nu on sin8 (high-frequency truth) to characterize where nu
collapses to a near-constant fit."""
import subprocess, tempfile
from pathlib import Path
import numpy as np

GAM = Path(__file__).resolve().parents[1] / "target" / "release" / "gam"

rng = np.random.default_rng(11)
n = 240
x = np.sort(rng.uniform(0, 1, n))
x_test = np.linspace(0.001, 0.999, 400)

for freq, label in [(8, "sin8"), (4, "sin4"), (2, "sin2"), (1, "sin1")]:
    y_clean = np.sin(2 * np.pi * freq * x)
    y = y_clean + 0.10 * rng.standard_normal(n)
    y_truth = np.sin(2 * np.pi * freq * x_test)
    tmp = Path(tempfile.mkdtemp())
    train = tmp / "train.csv"; test = tmp / "test.csv"
    train.write_text("x,y\n" + "\n".join(f"{a:.10g},{b:.10g}" for a, b in zip(x, y)))
    test.write_text("x,y\n" + "\n".join(f"{a:.10g},0" for a in x_test))
    print(f"\n=== truth = sin(2π·{freq}·x) ===")
    for nu_str in ["1/2", "1.0", "1.5", "2", "5/2", "3", "7/2", "9/2"]:
        model = tmp / "m.bin"
        fit = subprocess.run(
            [str(GAM), "fit", str(train), f"y ~ matern(x, nu={nu_str})", "--out", str(model)],
            capture_output=True, text=True,
        )
        if fit.returncode != 0:
            print(f"  nu={nu_str:5s} FAIL: {fit.stderr.strip().splitlines()[-1][:120]}")
            continue
        pred_csv = tmp / "p.csv"
        pred = subprocess.run(
            [str(GAM), "predict", str(model), str(test), "--out", str(pred_csv)],
            capture_output=True, text=True,
        )
        lines = pred_csv.read_text().strip().splitlines()
        j = lines[0].split(",").index("mean")
        yhat = np.array([float(ln.split(",")[j]) for ln in lines[1:]])
        rmse = float(np.sqrt(np.mean((yhat - y_truth) ** 2)))
        span = float(yhat.max() - yhat.min())
        flag = ""
        if span < 0.3 * (y_truth.max() - y_truth.min()):
            flag = "  COLLAPSED"
        print(f"  nu={nu_str:5s}  rmse={rmse:.4f}  span={span:.3f}{flag}")
