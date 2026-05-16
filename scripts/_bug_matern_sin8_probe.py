"""Deep dive: matern(x) collapsing to near-constant on a sin(2π·8·x) truth."""
import subprocess, tempfile
from pathlib import Path
import numpy as np

GAM = Path(__file__).resolve().parents[1] / "target" / "release" / "gam"

rng = np.random.default_rng(11)
n = 240
x = np.sort(rng.uniform(0, 1, n))
y_clean = np.sin(2 * np.pi * 8 * x)
y = y_clean + 0.10 * rng.standard_normal(n)

tmp = Path(tempfile.mkdtemp())
train = tmp / "train.csv"
test = tmp / "test.csv"
x_test = np.linspace(0.001, 0.999, 400)
y_truth = np.sin(2 * np.pi * 8 * x_test)

def write_csv(p, cols):
    p.write_text(",".join(cols) + "\n" + "\n".join(
        ",".join(f"{v:.10g}" for v in row) for row in zip(*[globals()[c] for c in cols])
    ))

# train
train.write_text("x,y\n" + "\n".join(f"{a:.10g},{b:.10g}" for a, b in zip(x, y)))
test.write_text("x,y\n" + "\n".join(f"{a:.10g},0" for a in x_test))

for label, formula in [
    ("matern default",         "matern(x)"),
    ("matern k=30",            "matern(x, k=30)"),
    ("matern centers=50",      "matern(x, centers=50)"),
    ("matern centers=80",      "matern(x, centers=80)"),
    ("matern nu=1.5",          "matern(x, nu=1.5)"),
    ("matern nu=2.5 k=50",     "matern(x, nu=2.5, k=50)"),
    ("smooth k=20",            "smooth(x, k=20)"),
    ("smooth k=50",            "smooth(x, k=50)"),
    ("smooth k=80",            "smooth(x, k=80)"),
    ("duchon centers=80",      "duchon(x, centers=80)"),
]:
    model = tmp / "m.bin"
    fit = subprocess.run(
        [str(GAM), "fit", str(train), f"y ~ {formula}", "--out", str(model)],
        capture_output=True, text=True,
    )
    if fit.returncode != 0:
        print(f"{label:25s} -> FIT FAILED:", fit.stderr.strip().splitlines()[-1][:200])
        continue
    pred_csv = tmp / "p.csv"
    pred = subprocess.run(
        [str(GAM), "predict", str(model), str(test), "--out", str(pred_csv)],
        capture_output=True, text=True,
    )
    if pred.returncode != 0:
        print(f"{label:25s} -> PREDICT FAILED")
        continue
    lines = pred_csv.read_text().strip().splitlines()
    j = lines[0].split(",").index("mean")
    yhat = np.array([float(ln.split(",")[j]) for ln in lines[1:]])
    rmse = float(np.sqrt(np.mean((yhat - y_truth) ** 2)))
    print(f"{label:25s} rmse={rmse:.4f}  span={yhat.max()-yhat.min():.3f}  "
          f"min={yhat.min():+.3f} max={yhat.max():+.3f}")
