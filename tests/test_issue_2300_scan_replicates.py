"""Exact observation replicates for saved O(n) spline-scan models (#2300)."""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


SCAN_FORMULA = (
    'y ~ s(x, bs="ps", degree=3, penalty_order=2, double_penalty=false)'
)


def _weighted_scan_rows() -> list[dict[str, float]]:
    rng = np.random.default_rng(2300)
    x = np.linspace(-2.0, 2.0, 180)
    w = np.where(x < 0.0, 1.0, 9.0)
    mean = 0.4 + np.sin(1.3 * x)
    y = mean + rng.normal(0.0, 0.45 / np.sqrt(w))
    return [
        {"y": float(y_i), "x": float(x_i), "w": float(w_i)}
        for x_i, y_i, w_i in zip(x, y, w, strict=True)
    ]


def test_weighted_scan_replicates_use_exact_saved_noise_law_and_seed() -> None:
    model = gamfit.fit(
        _weighted_scan_rows(),
        SCAN_FORMULA,
        family="gaussian",
        weights="w",
    )
    # Equal x makes the saved bridge mean and fitted sigma identical; only the
    # analytic observation weight differs. Therefore sd(w=1)/sd(w=9) is
    # exactly sqrt(9/1)=3 in the generating law.
    probe = [
        {"y": 0.0, "x": 0.15, "w": 1.0},
        {"y": 0.0, "x": 0.15, "w": 9.0},
    ]
    first = np.asarray(model.sample_replicates(probe, 8_000, seed=19), dtype=float)
    again = np.asarray(model.sample_replicates(probe, 8_000, seed=19), dtype=float)
    different = np.asarray(model.sample_replicates(probe, 8_000, seed=20), dtype=float)
    chunks = list(model.iter_replicates(probe, 8_000, chunk_size=517, seed=19))
    streamed = np.concatenate(chunks, axis=0)

    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, streamed)
    assert not np.array_equal(first, different)
    assert first.shape == (8_000, 2)
    assert all(chunk.shape == (517, 2) for chunk in chunks[:-1])
    assert chunks[-1].shape == (245, 2)
    np.testing.assert_allclose(first.mean(axis=0), model.predict(probe), atol=0.025)
    ratio = float(first[:, 0].std() / first[:, 1].std())
    assert 2.85 < ratio < 3.15, ratio


def test_weighted_scan_replicates_round_trip_and_refuse_missing_weights(tmp_path: typing.Any) -> None:
    model = gamfit.fit(
        _weighted_scan_rows(),
        SCAN_FORMULA,
        family="gaussian",
        weights="w",
    )
    probe = [
        {"y": 0.0, "x": -0.35, "w": 0.75},
        {"y": 0.0, "x": 0.65, "w": 4.0},
    ]
    before = np.asarray(model.sample_replicates(probe, 64, seed=71))
    path = tmp_path / "weighted-scan.gam"
    model.save(path)
    restored = gamfit.load(path)
    after = np.asarray(restored.sample_replicates(probe, 64, seed=71))
    np.testing.assert_array_equal(before, after)

    missing = [{"y": row["y"], "x": row["x"]} for row in probe]
    with pytest.raises(Exception, match="weight|weighted"):
        restored.sample_replicates(missing, 4, seed=71)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        restored.iter_replicates(probe, 4, chunk_size=0, seed=71)
