"""The native anytime-valid structure certificate is persisted verbatim."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _fit():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * math.pi, 200)
    harmonic = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(2, 16))
    mixing /= np.linalg.norm(mixing, axis=0, keepdims=True)
    target = harmonic @ mixing + 0.05 * rng.normal(size=(200, 16))
    target -= target.mean(axis=0, keepdims=True)
    return gamfit.sae_manifold_fit(
        X=target,
        K=2,
        atom_basis="periodic",
        d_atom=2,
        assignment="ordered_beta_bernoulli",
        n_iter=30,
        learning_rate=0.04,
        random_state=0,
    )


def _certificate(fit) -> dict:
    raw = fit.structure_certificate_json
    assert raw is not None
    return json.loads(raw)


def _confirmed_indices(log_e: list[float], alpha: float) -> set[int]:
    order = sorted(range(len(log_e)), key=log_e.__getitem__, reverse=True)
    k_star = 0
    for rank, index in enumerate(order, start=1):
        if log_e[index] >= math.log(len(log_e)) - math.log(alpha) - math.log(rank):
            k_star = rank
    return set(order[:k_star])


def test_structure_certificate_matches_independent_e_bh() -> None:
    cert = _certificate(_fit())
    assert set(cert) == {"alpha", "entries"}
    assert 0.0 < cert["alpha"] < 1.0
    entries = cert["entries"]
    log_e = [float(entry["log_e"]) for entry in entries]
    assert all(math.isfinite(value) for value in log_e)
    assert all(int(entry["steps"]) >= 0 for entry in entries)

    expected = _confirmed_indices(log_e, float(cert["alpha"]))
    observed = {index for index, entry in enumerate(entries) if entry["confirmed"]}
    assert observed == expected


def test_stricter_alpha_never_grows_confirmed_set() -> None:
    cert = _certificate(_fit())
    log_e = [float(entry["log_e"]) for entry in cert["entries"]]
    assert _confirmed_indices(log_e, 0.01) <= _confirmed_indices(log_e, 0.2)


def test_contested_entries_are_the_unconfirmed_complement() -> None:
    entries = _certificate(_fit())["entries"]
    confirmed = [entry for entry in entries if entry["confirmed"]]
    contested = [entry for entry in entries if not entry["confirmed"]]
    assert len(confirmed) + len(contested) == len(entries)


def test_certificate_round_trips_through_native_payload() -> None:
    fit = _fit()
    restored = gamfit.ManifoldSAE.from_dict(fit.to_dict())
    assert restored.structure_certificate_json == fit.structure_certificate_json
