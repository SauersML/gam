"""Deep probe: 2D Duchon collapses on a high-frequency truth (sin(8π·x1)·cos(8π·x2))."""
import subprocess, tempfile
from pathlib import Path
import numpy as np

GAM = Path(__file__).resolve().parents[1] / "target" / "release" / "gam"
rng = np.random.default_rng(13)
n = 500
x1 = rng.uniform(0, 1, n); x2 = rng.uniform(0, 1, n)
y_clean = np.sin(2*np.pi*4*x1) * np.cos(2*np.pi*4*x2)
y = y_clean + 0.10 * rng.standard_normal(n)
g = np.linspace(0.01, 0.99, 30); g1, g2 = np.meshgrid(g, g)
x1t, x2t = g1.ravel(), g2.ravel()
y_truth = np.sin(2*np.pi*4*x1t) * np.cos(2*np.pi*4*x2t)

tmp = Path(tempfile.mkdtemp())
train = tmp / "train.csv"; test = tmp / "test.csv"
train.write_text("x1,x2,y\n" + "\n".join(f"{a:.10g},{b:.10g},{c:.10g}" for a,b,c in zip(x1,x2,y)))
test.write_text("x1,x2,y\n" + "\n".join(f"{a:.10g},{b:.10g},0" for a,b in zip(x1t,x2t)))

for label, formula in [
    ("duchon default",      "duchon(x1, x2)"),
    ("duchon centers=80",   "duchon(x1, x2, centers=80)"),
    ("duchon centers=150",  "duchon(x1, x2, centers=150)"),
    ("duchon pure=true",    "duchon(x1, x2, pure=true)"),
    ("duchon pure ls=0.05", "duchon(x1, x2, length_scale=0.05)"),
    ("duchon pure ls=0.10", "duchon(x1, x2, length_scale=0.10)"),
    ("duchon pure ls=0.20", "duchon(x1, x2, length_scale=0.20)"),
    ("duchon pure ls=0.50", "duchon(x1, x2, length_scale=0.50)"),
    ("duchon pure ls=1.0",  "duchon(x1, x2, length_scale=1.0)"),
    ("matern centers=150",  "matern(x1, x2, centers=150)"),
    ("matern nu=1/2",       "matern(x1, x2, nu=1/2)"),
    ("matern nu=1/2 k=150", "matern(x1, x2, nu=1/2, k=150)"),
    ("smooth k=12",         "smooth(x1, x2, k=12)"),
    ("smooth k=15",         "smooth(x1, x2, k=15)"),
]:
    model = tmp / "m.bin"
    fit = subprocess.run([str(GAM), "fit", str(train), f"y ~ {formula}", "--out", str(model)],
                         capture_output=True, text=True)
    if fit.returncode != 0:
        print(f"{label:25s} FIT FAILED: {fit.stderr.strip().splitlines()[-1][:120]}")
        continue
    pred_csv = tmp / "p.csv"
    p = subprocess.run([str(GAM), "predict", str(model), str(test), "--out", str(pred_csv)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"{label:25s} PREDICT FAILED")
        continue
    lines = pred_csv.read_text().strip().splitlines()
    j = lines[0].split(",").index("mean")
    yhat = np.array([float(ln.split(",")[j]) for ln in lines[1:]])
    rmse = float(np.sqrt(np.mean((yhat - y_truth) ** 2)))
    span = float(yhat.max() - yhat.min())
    print(f"{label:25s} rmse={rmse:.4f}  span={span:.3f}")
