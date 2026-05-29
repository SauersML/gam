from __future__ import annotations

import importlib
import os
import tempfile
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def _fit_and_sample():
    rows = [{"y": float(i % 5) + 0.1 * i, "x": float(i)} for i in range(40)]
    model = gamfit.fit(rows, "y ~ x")
    return model.sample(rows, samples=24, warmup=24, chains=1, seed=42)


def test_save_without_extension_returns_loadable_path() -> None:
    """save(path) with no .npz suffix must return the file numpy actually wrote.

    Regression for #391: np.savez appends ".npz" when the target lacks it, but
    save() used to return the un-suffixed path, which does not exist on disk, so
    load_posterior(save_return_value) raised FileNotFoundError.
    """
    ps = _fit_and_sample()

    base = os.path.join(tempfile.mkdtemp(), "post1")  # natural no-extension call
    returned = ps.save(base)

    # The returned path is the one actually written and ends with the npz suffix.
    assert returned.endswith(".npz"), f"expected .npz path, got {returned!r}"
    assert os.path.exists(returned), f"returned path does not exist on disk: {returned!r}"

    # Round-trip via the returned value must succeed (this is the bug repro).
    loaded = gamfit.load_posterior(returned)
    np.testing.assert_allclose(
        np.asarray(loaded.samples),
        np.asarray(ps.samples),
        atol=0.0,
        err_msg="round-trip via save() return value must reproduce samples bit-exactly",
    )


def test_load_tolerates_npz_auto_suffix() -> None:
    """load() accepts the un-suffixed path even though the file is at path.npz.

    Mirrors numpy's savez auto-suffix so callers who kept their original
    (extension-less) path string can still load what they saved.
    """
    ps = _fit_and_sample()

    base = os.path.join(tempfile.mkdtemp(), "post2")
    ps.save(base)

    assert not os.path.exists(base)
    assert os.path.exists(base + ".npz")

    # Passing the original extension-less path must resolve to the .npz file.
    loaded = gamfit.load_posterior(base)
    np.testing.assert_allclose(
        np.asarray(loaded.mean),
        np.asarray(ps.mean),
        atol=0.0,
        err_msg="load() must resolve the numpy .npz auto-suffix from the bare path",
    )


def test_explicit_npz_path_is_unchanged() -> None:
    """An explicit .npz path is returned and loaded verbatim (no double suffix)."""
    ps = _fit_and_sample()

    base = os.path.join(tempfile.mkdtemp(), "post3.npz")
    returned = ps.save(base)

    assert returned == base, f"explicit .npz path must round-trip unchanged: {returned!r}"
    assert not os.path.exists(base + ".npz"), "must not append a second .npz suffix"
    loaded = gamfit.load_posterior(returned)
    np.testing.assert_allclose(
        np.asarray(loaded.std),
        np.asarray(ps.std),
        atol=0.0,
        err_msg="explicit .npz path must round-trip std bit-exactly",
    )
