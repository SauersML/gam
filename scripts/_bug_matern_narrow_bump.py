"""Probe: matern nu=9/2 on a narrow 2D bump collapses with span ≈ 0.008
even after the data-aware length_scale init."""
import subprocess, tempfile
from pathlib import Path
import numpy as np

GAM = Path(__file__).resolve().parents[1] / "target" / "release" / "gam"
rng = np.random.default_rng(13)
n = 500
x1 = rng.uniform(0, 1, n); x2 = rng.uniform(0, 1, n)
y_clean = np.exp(-((x1-0.5)**2 + (x2-0.5)**2) / 0.003)
y = y_clean + 0.10 * rng.standard_normal(n)
g = np.linspace(0.01, 0.99, 30); g1, g2 = np.meshgrid(g, g)
x1t, x2t = g1.ravel(), g2.ravel()
y_truth = np.exp(-((x1t-0.5)**2 + (x2t-0.5)**2) / 0.003)

tmp = Path(tempfile.mkdtemp())
train = tmp / "train.csv"; test = tmp / "test.csv"
train.write_text("x1,x2,y\n" + "\n".join(f"{a:.10g},{b:.10g},{c:.10g}" for a,b,c in zip(x1,x2,y)))
test.write_text("x1,x2,y\n" + "\n".join(f"{a:.10g},{b:.10g},0" for a,b in zip(x1t,x2t)))

for label, formula in [
    ("matern nu=1/2",            "matern(x1, x2, nu=1/2)"),
    ("matern nu=3/2",            "matern(x1, x2, nu=3/2)"),
    ("matern nu=5/2",            "matern(x1, x2, nu=5/2)"),
    ("matern nu=7/2",            "matern(x1, x2, nu=7/2)"),
    ("matern nu=9/2",            "matern(x1, x2, nu=9/2)"),
    ("matern nu=9/2 ls=0.05",    "matern(x1, x2, nu=9/2, length_scale=0.05)"),
    ("matern nu=9/2 ls=0.03",    "matern(x1, x2, nu=9/2, length_scale=0.03)"),
    ("matern nu=9/2 ls=0.10",    "matern(x1, x2, nu=9/2, length_scale=0.10)"),
    ("matern nu=9/2 centers=200","matern(x1, x2, nu=9/2, centers=200)"),
]:
    model = tmp / "m.bin"
    fit = subprocess.run([str(GAM), "fit", str(train), f"y ~ {formula}", "--out", str(model)],
                         capture_output=True, text=True)
    if fit.returncode != 0:
        print(f"{label:30s} FAIL: {fit.stderr.strip().splitlines()[-1][:120]}")
        continue
    pred_csv = tmp / "p.csv"
    p = subprocess.run([str(GAM), "predict", str(model), str(test), "--out", str(pred_csv)],
                       capture_output=True, text=True)
    lines = pred_csv.read_text().strip().splitlines()
    j = lines[0].split(",").index("mean")
    yhat = np.array([float(ln.split(",")[j]) for ln in lines[1:]])
    rmse = float(np.sqrt(np.mean((yhat - y_truth) ** 2)))
    span = float(yhat.max() - yhat.min())
    print(f"{label:30s} rmse={rmse:.4f}  span={span:.4f}  max_yhat={yhat.max():.3f}")
