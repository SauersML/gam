"""Adversarial 1D smooth quality sweep, driven via the gam CLI.

For each (truth, smooth-family) pair: write train/test CSVs, fit a model, read
predictions on a dense test grid, compute RMSE vs truth. Flag any combination
where RMSE > a per-truth budget — those are candidate bugs.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

GAM = Path(__file__).resolve().parents[1] / "target" / "release" / "gam"
assert GAM.exists(), f"gam CLI not found at {GAM}"

RNG_SEED = 11


def truth_signals():
    return {
        "constant":         lambda x: np.full_like(x, 0.7),
        "linear":           lambda x: 1.5 * x - 0.4,
        "smooth_quad":      lambda x: 1.0 - 2.0 * (x - 0.5) ** 2,
        "sin1":             lambda x: np.sin(2 * np.pi * 1 * x),
        "sin2":             lambda x: np.sin(2 * np.pi * 2 * x),
        "sin4":             lambda x: np.sin(2 * np.pi * 4 * x),
        "sin8":             lambda x: np.sin(2 * np.pi * 8 * x),
        "sharp_bump":       lambda x: np.exp(-((x - 0.5) ** 2) / 0.005),
        "step":             lambda x: (x > 0.5).astype(float),
        "two_bumps":        lambda x: (np.exp(-((x - 0.3) ** 2) / 0.01)
                                       - np.exp(-((x - 0.75) ** 2) / 0.01)),
    }


def smooth_families():
    return {
        "thinplate":          "thinplate(x)",
        "matern":             "matern(x)",
        "duchon":             "duchon(x)",
        "duchon_centers20":   "duchon(x, centers=20)",
        "duchon_centers50":   "duchon(x, centers=50)",
        "smooth":             "smooth(x)",
    }


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    headers = list(columns.keys())
    rows = list(zip(*columns.values()))
    with path.open("w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.10g}" for v in r) + "\n")


def fit_predict(train_csv: Path, test_csv: Path, formula: str, tmp: Path) -> np.ndarray:
    model = tmp / "model.bin"
    out = tmp / "predict.csv"
    fit = subprocess.run(
        [str(GAM), "fit", str(train_csv), formula, "--out", str(model)],
        capture_output=True, text=True,
    )
    if fit.returncode != 0:
        raise RuntimeError(f"fit failed: {fit.stderr[-2000:]}")
    pred = subprocess.run(
        [str(GAM), "predict", str(model), str(test_csv), "--out", str(out)],
        capture_output=True, text=True,
    )
    if pred.returncode != 0:
        raise RuntimeError(f"predict failed: {pred.stderr[-2000:]}")
    # Read prediction column "mean"
    raw = out.read_text().strip().splitlines()
    header = raw[0].split(",")
    if "mean" not in header:
        raise RuntimeError(f"no mean column; header={header}")
    j = header.index("mean")
    vals = np.array([float(line.split(",")[j]) for line in raw[1:]])
    return vals


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sigma = 0.10
    n_train = 240
    x_train = np.sort(rng.uniform(0.0, 1.0, n_train))
    n_test = 400
    x_test = np.linspace(0.001, 0.999, n_test)

    truths = truth_signals()
    families = smooth_families()

    print(f"{'truth':14s} {'family':18s} {'rmse':>9s} {'max':>9s} {'span':>7s}  {'status'}")
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        train_csv = tmp / "train.csv"
        test_csv = tmp / "test.csv"
        # Test CSV needs y column too even though unused
        write_csv(test_csv, {"x": x_test, "y": np.zeros_like(x_test)})

        for tname, fn in truths.items():
            y_truth = fn(x_test)
            y_train_clean = fn(x_train)
            y_noisy = y_train_clean + sigma * rng.standard_normal(n_train)
            write_csv(train_csv, {"x": x_train, "y": y_noisy})

            # Adaptive budget: noise-limited best-case is sigma * sqrt(1 / n / k)
            # but a reasonable max-residual budget for any sane fit is ~5σ for
            # benign truths, 0.3 of truth peak-to-peak for harder ones.
            peak = float(np.max(y_truth) - np.min(y_truth))
            budget = max(0.30 * peak, 5 * sigma)

            for fname, formula_body in families.items():
                try:
                    pred = fit_predict(
                        train_csv, test_csv,
                        f"y ~ {formula_body}",
                        tmp,
                    )
                except Exception as exc:
                    msg = str(exc).splitlines()[-1][:160]
                    print(f"{tname:14s} {fname:18s} {'-':>9s} {'-':>9s} {'-':>7s}  CRASH: {msg}")
                    failures.append(f"{tname}/{fname}: CRASH {msg}")
                    continue
                rmse = float(np.sqrt(np.mean((pred - y_truth) ** 2)))
                mx = float(np.max(np.abs(pred - y_truth)))
                span = float(pred.max() - pred.min())
                status = "OK" if mx <= budget else f"!! >{budget:.3f}"
                print(f"{tname:14s} {fname:18s} {rmse:9.4f} {mx:9.4f} {span:7.3f}  {status}")
                if mx > budget:
                    failures.append(f"{tname}/{fname}: max={mx:.3f} > budget={budget:.3f}")

    print()
    if failures:
        print(f"=== {len(failures)} candidate failures ===")
        for f in failures:
            print("  -", f)
    else:
        print("=== all combinations within budget ===")


if __name__ == "__main__":
    main()
