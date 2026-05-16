"""2D adversarial sweep — drive gam CLI on many (truth, smooth) pairs.

Outputs a per-cell summary; flags candidate failures by RMSE budget."""
from __future__ import annotations

import subprocess, tempfile
from pathlib import Path
import numpy as np

GAM = Path(__file__).resolve().parents[1] / "target" / "release" / "gam"


def truths():
    return {
        "saddle":       lambda x1, x2: 0.7 * (x1 - 0.5) ** 2 - 0.7 * (x2 - 0.5) ** 2,
        "ridge_diag":   lambda x1, x2: np.exp(-((x1 - x2) ** 2) / 0.01),
        "two_bumps":    lambda x1, x2: (np.exp(-((x1 - 0.3)**2 + (x2 - 0.3)**2)/0.02)
                                        - np.exp(-((x1 - 0.7)**2 + (x2 - 0.7)**2)/0.02)),
        "hifreq":       lambda x1, x2: np.sin(2*np.pi*4*x1) * np.cos(2*np.pi*4*x2),
        "step_diag":    lambda x1, x2: (x1 + x2 > 1.0).astype(float),
        "narrow_bump":  lambda x1, x2: np.exp(-((x1-0.5)**2 + (x2-0.5)**2)/0.003),
    }


def smooths():
    return {
        "thinplate":         "thinplate(x1, x2)",
        "matern":            "matern(x1, x2)",
        "matern nu=5/2":     "matern(x1, x2, nu=5/2)",
        "matern nu=9/2":     "matern(x1, x2, nu=9/2)",
        "duchon":            "duchon(x1, x2)",
        "duchon centers=80": "duchon(x1, x2, centers=80)",
        "te k=8":            "te(x1, x2, k=8)",
        "te k=12":           "te(x1, x2, k=12)",
    }


def main():
    rng = np.random.default_rng(13)
    sigma = 0.10
    n_train = 500
    x1_train = rng.uniform(0, 1, n_train)
    x2_train = rng.uniform(0, 1, n_train)

    # test on a uniform grid
    g = np.linspace(0.01, 0.99, 30)
    g1, g2 = np.meshgrid(g, g)
    x1_test = g1.ravel(); x2_test = g2.ravel()

    families = smooths()
    print(f"{'truth':14s} {'family':22s} {'rmse':>9s} {'maxabs':>9s} {'span':>7s}  status")
    failures = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        train = tmp / "train.csv"; test = tmp / "test.csv"
        test.write_text("x1,x2,y\n" +
            "\n".join(f"{a:.10g},{b:.10g},0" for a, b in zip(x1_test, x2_test)))
        for tname, fn in truths().items():
            y_clean = fn(x1_train, x2_train)
            y = y_clean + sigma * rng.standard_normal(n_train)
            train.write_text("x1,x2,y\n" +
                "\n".join(f"{a:.10g},{b:.10g},{c:.10g}"
                         for a, b, c in zip(x1_train, x2_train, y)))
            y_truth = fn(x1_test, x2_test)
            peak = float(y_truth.max() - y_truth.min())
            budget = max(0.30 * peak, 5 * sigma)
            for fname, formula in families.items():
                model = tmp / "m.bin"
                fit = subprocess.run(
                    [str(GAM), "fit", str(train), f"y ~ {formula}", "--out", str(model)],
                    capture_output=True, text=True,
                )
                if fit.returncode != 0:
                    msg = fit.stderr.strip().splitlines()[-1][:140]
                    print(f"{tname:14s} {fname:22s} {'-':>9s} {'-':>9s} {'-':>7s}  CRASH: {msg}")
                    failures.append(f"{tname}/{fname}: CRASH {msg}")
                    continue
                pred_csv = tmp / "p.csv"
                pr = subprocess.run(
                    [str(GAM), "predict", str(model), str(test), "--out", str(pred_csv)],
                    capture_output=True, text=True,
                )
                if pr.returncode != 0:
                    print(f"{tname:14s} {fname:22s} PREDICT FAILED")
                    continue
                lines = pred_csv.read_text().strip().splitlines()
                j = lines[0].split(",").index("mean")
                yhat = np.array([float(ln.split(",")[j]) for ln in lines[1:]])
                rmse = float(np.sqrt(np.mean((yhat - y_truth) ** 2)))
                mx = float(np.max(np.abs(yhat - y_truth)))
                span = float(yhat.max() - yhat.min())
                status = "OK" if mx <= budget else f"!! >{budget:.2f}"
                # Collapse: predicted span << truth span
                if peak > 0.3 and span < 0.2 * peak:
                    status += " COLLAPSE"
                    failures.append(f"{tname}/{fname}: COLLAPSE span={span:.3f} truth_peak={peak:.3f}")
                print(f"{tname:14s} {fname:22s} {rmse:9.4f} {mx:9.4f} {span:7.3f}  {status}")
                if mx > budget and "COLLAPSE" not in status:
                    failures.append(f"{tname}/{fname}: max={mx:.3f} > budget={budget:.3f}")

    print()
    if failures:
        print(f"=== {len(failures)} candidate failures ===")
        for f in failures: print("  -", f)
    else:
        print("=== all combinations within budget ===")


if __name__ == "__main__":
    main()
