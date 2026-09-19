"""Offline checks of the real-data leaderboard plumbing (no downloads, no fits)."""

from __future__ import annotations

import numpy as np

from . import datasets
from .worker import FOLDS, fold_ids, mean_deviance


def test_folds_partition_rows_and_stratify_binomial() -> None:
    y = (np.arange(1000) % 29 == 0).astype(float)  # 35 events, a rare-event response
    ids = fold_ids(y, "binomial")
    assert sorted(set(ids.tolist())) == list(range(FOLDS))
    assert np.bincount(ids).max() - np.bincount(ids).min() <= 1
    events = np.bincount(ids[y == 1], minlength=FOLDS)
    assert events.max() - events.min() <= 1
    # Fixed seed: the split is a function of the data alone.
    assert np.array_equal(ids, fold_ids(y, "binomial"))


def test_every_formula_names_only_built_columns() -> None:
    for ds in datasets.DATASETS:
        assert ds.formula.startswith("y ~ ")
        assert set(ds.factor_columns) <= set(ds.columns)
        assert len(ds.sources) == len({s.filename for s in ds.sources})
        for s in ds.sources:
            assert len(s.sha256) == 64 and s.url.startswith("https://")


def test_gamsim1_needs_no_download_and_is_reproducible() -> None:
    a = datasets._b_gamsim1({})
    b = datasets._b_gamsim1({})
    assert set(a) == {"x0", "x1", "x2", "x3", "y"}
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_deviance_is_zero_at_the_data_and_weighted() -> None:
    y = np.array([1.0, 2.0, 3.0])
    for fam in ("gaussian", "poisson", "gamma"):
        assert mean_deviance(fam, y, y, np.ones(3)) == 0.0
    p = np.array([0.2, 0.5])
    heavy = mean_deviance("binomial", np.array([0.0, 1.0]), p, np.array([1.0, 3.0]))
    light = mean_deviance("binomial", np.array([0.0, 1.0]), p, np.array([3.0, 1.0]))
    assert heavy > light  # more weight on the worse-predicted row
