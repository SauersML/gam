"""Regression for issue #399: `Model.sample` must reject degenerate draw counts
with a clean, typed configuration error *before* the sampling engine is invoked
— never a Rust panic across the FFI boundary.

The pre-fix behaviour was:

    samples=0 -> GamError: sample_table panicked inside Rust boundary:
                 expected thread to succeed in generating observation.

i.e. the panic-payload was caught at the FFI boundary and surfaced as a
"panicked inside Rust boundary" message. The fix validates `n_samples` up front
(mirroring the existing `target_accept` guard), so it now raises a clean message
that does NOT contain "panicked inside Rust boundary". The chain count is no
longer configurable: every run uses two chains.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def _gaussian_model():
    rows = [{"y": float(i % 5) + 0.1 * i, "x": float(i)} for i in range(40)]
    return rows, gamfit.fit(rows, "y ~ x", family="gaussian")


def _logit_model():
    # Unit-weight Bernoulli-logit GAM: this auto-selects the Pólya-Gamma Gibbs
    # sampler in the Rust core, NOT the general-mcmc NUTS engine. The original
    # #399 fix left this path unguarded.
    rng = np.random.default_rng(0)
    xs = np.linspace(-2.0, 2.0, 60)
    p = 1.0 / (1.0 + np.exp(-(0.5 + 1.3 * xs)))
    ys = (rng.uniform(0.0, 1.0, xs.size) < p).astype(float)
    rows = [{"y": float(y), "x": float(x)} for x, y in zip(xs, ys)]
    return rows, gamfit.fit(rows, "y ~ x", family="binomial")


def _assert_clean_config_error(rows, model, **bad_cfg) -> None:
    with pytest.raises(gamfit.errors.GamError) as exc_info:
        model.sample(rows, **bad_cfg)
    message = str(exc_info.value)
    assert "panicked inside Rust boundary" not in message, (
        "degenerate sample config must be rejected by an up-front validator, "
        f"not surface as a caught Rust panic: {message!r}"
    )
    # The validator names the offending field so the user can fix it.
    assert "n_samples" in message, (
        f"validation error should name the bad field, got: {message!r}"
    )


def test_zero_samples_raises_clean_config_error_gaussian() -> None:
    rows, model = _gaussian_model()
    _assert_clean_config_error(rows, model, samples=0, seed=1)


def test_too_few_samples_raises_clean_config_error_gaussian() -> None:
    # samples in {1, 2, 3} crash the engine's split-R-hat path; must be rejected.
    rows, model = _gaussian_model()
    for bad in (1, 2, 3):
        _assert_clean_config_error(rows, model, samples=bad, seed=1)


def test_zero_samples_raises_clean_config_error_logit() -> None:
    # Missed path in the original fix: the unit-weight binomial-logit surface
    # routes through Pólya-Gamma Gibbs, which previously returned a silently
    # empty posterior for samples=0 instead of a typed error.
    rows, model = _logit_model()
    _assert_clean_config_error(rows, model, samples=0, seed=1)


def test_minimum_samples_return_draws_from_both_chains() -> None:
    # Four draws per chain is the fewest split R-hat accepts, and every run uses
    # two chains, on the NUTS and the Pólya-Gamma Gibbs surfaces alike.
    for rows, model in (_gaussian_model(), _logit_model()):
        draws = model.sample(rows, samples=4, seed=123)
        assert np.asarray(draws.samples).shape[0] == 8, (
            "four draws per chain over two chains must return eight draws"
        )
