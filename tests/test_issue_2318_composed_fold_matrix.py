"""Installed-wheel fold matrix for constructive smooth penalties (#2318).

The source-level regression pins the ill-conditioned quadratic itself.  This
public-API matrix exercises the composition boundary that originally exposed
it: every formula is rebuilt on five distinct training subsets with the
default double penalty and REML smoothing selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit

pytest.importorskip("gamfit._rust")


_CASES = (
    ("y ~ smooth(a) + smooth(b) + smooth(c) + d", "add_d"),
    ("y ~ cyclic(b) + smooth(a) + c", "cyclic"),
    ("y ~ smooth(a) + smooth(b) + smooth(c) + e", "add_e"),
    ("y ~ tensor(a, b) + smooth(c) + d", "tensor_smooth"),
    ("y ~ tensor(a, b) + tensor(c, e) + f", "tensor_tensor"),
)


@pytest.fixture(scope="module")
def composed_fold_data_2318() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(2318)
    n = 750
    frame = pd.DataFrame(
        {
            "a": rng.uniform(0.0, 1.0, n),
            "b": rng.uniform(0.0, 1.0, n),
            "c": rng.uniform(0.0, 1.0, n),
            "d": rng.normal(size=n),
            "e": rng.uniform(0.0, 1.0, n),
            "f": rng.normal(size=n),
        }
    )
    a = frame["a"].to_numpy()
    b = frame["b"].to_numpy()
    c = frame["c"].to_numpy()
    d = frame["d"].to_numpy()
    e = frame["e"].to_numpy()
    f = frame["f"].to_numpy()
    truths = {
        "add_d": np.sin(2.0 * np.pi * a)
        + 0.7 * np.cos(2.0 * np.pi * b)
        + 0.6 * (c - 0.5) ** 2
        + 0.3 * d,
        "cyclic": np.sin(2.0 * np.pi * b) + 0.8 * np.sin(2.0 * np.pi * a) + 0.4 * c,
        "add_e": np.sin(2.0 * np.pi * a)
        + 0.7 * np.cos(2.0 * np.pi * b)
        + 0.6 * (c - 0.5) ** 2
        + 0.4 * e,
        "tensor_smooth": np.sin(2.0 * np.pi * a) * np.cos(2.0 * np.pi * b)
        + 0.5 * np.sin(2.0 * np.pi * c)
        + 0.3 * d,
        "tensor_tensor": np.sin(2.0 * np.pi * a) * np.cos(2.0 * np.pi * b)
        + 0.6 * np.sin(2.0 * np.pi * c) * np.cos(2.0 * np.pi * e)
        + 0.2 * f,
    }
    permutation = rng.permutation(n)
    fold = np.empty(n, dtype=np.int64)
    fold[permutation] = np.arange(n, dtype=np.int64) % 5
    return frame, truths, fold


@pytest.mark.parametrize(("formula", "truth_key"), _CASES)
def test_composed_select_true_fits_every_fold_2318(
    formula: str,
    truth_key: str,
    composed_fold_data_2318: tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray],
) -> None:
    frame, truths, fold = composed_fold_data_2318
    truth = truths[truth_key]
    response = truth + np.random.default_rng(10_000 + _CASES.index((formula, truth_key))).normal(
        scale=0.15,
        size=len(frame),
    )
    observed_improvement = []

    for held_out in range(5):
        training_mask = fold != held_out
        validation_mask = ~training_mask
        training = frame.loc[training_mask].copy()
        validation = frame.loc[validation_mask]
        training["y"] = response[training_mask]

        model = gamfit.fit(training, formula, family="gaussian")
        prediction = np.asarray(model.predict(validation), dtype=float)
        target = truth[validation_mask]
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        constant_rmse = float(np.sqrt(np.mean((target - response[training_mask].mean()) ** 2)))

        assert prediction.shape == target.shape
        assert np.all(np.isfinite(prediction))
        assert rmse < 0.45, f"{formula}, fold {held_out}: RMSE {rmse:.4f}"
        observed_improvement.append(rmse / constant_rmse)

    assert max(observed_improvement) < 0.75, (
        f"{formula}: worst fold failed to improve materially over the training-mean predictor: "
        f"ratios={observed_improvement}"
    )
