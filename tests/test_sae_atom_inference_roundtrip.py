"""Per-atom post-PIRLS inference reports: contents and JSON round-trip.

The Rust SAE fit emits, for every fitted atom, an ``atom_inference`` report
holding the #1097 penalty-debiased decoder-functional point summaries. It
carries no non-constancy e-value: every row of the inner-fit snapshot comes
from the full-data fit (the latent coordinate ``t_i`` is itself fitted to
``Z_i``), so no split of those rows is an evaluation fold independent of the
alternative, and a split-likelihood-ratio statistic on them does not satisfy
``E_{H0}[E] <= 1`` (#3929). These tests pin that the report reaches a Python
caller through ``ManifoldSAE.atom_inference_reports``, publishes no such
e-value, and survives a ``to_json``/``from_json`` round-trip exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import ManifoldSAE  # noqa: E402

# Both tests drive a full SAE-manifold atom-smooth REML fit (~2-3 min), so they
# are @slow: excluded from the default `-m "not slow"` run and run by the
# python-populations job.
pytestmark = pytest.mark.slow

_FUNCTIONALS = ("peak_contrast", "average_value", "decoder_variation_norm")
_ESTIMATE_KEYS = ("theta_plugin", "theta_onestep", "penalty_bias")


def _circle_data(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    """A single circular harmonic mixed into ``p`` output dims: a genuinely
    curved 1-atom truth, so the atom's inner smooth is harvested."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _fit_one_atom_periodic(seed: int = 0):
    z = _circle_data(n=300, p=48, noise=0.04, seed=seed)
    return gamfit.sae.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ordered_beta_bernoulli",
        n_iter=50,
        learning_rate=0.04,
        random_state=seed,
    )


def test_atom_inference_reports_functionals_and_no_e_value():
    fit = _fit_one_atom_periodic()
    reports = fit.atom_inference_reports
    assert isinstance(reports, list) and len(reports) == len(fit.atoms), (
        f"atom_inference_reports must hold one entry per atom; got {reports!r} "
        f"for {len(fit.atoms)} atoms"
    )
    report = reports[0]
    assert set(report) == {"atom_index", "atom_name", "functionals"}, (
        "the per-atom report carries only the functional point summaries; a "
        "non-constancy e-value on the full-data snapshot is not valid under H0 "
        f"(#3929). keys present: {sorted(report)}"
    )
    functionals = report["functionals"]
    assert functionals is not None, (
        "a harvested curved atom must carry its functional point summaries"
    )
    for name in _FUNCTIONALS:
        estimate = functionals[name]
        assert estimate is not None, f"functional {name} must be reported"
        for key in _ESTIMATE_KEYS:
            assert math.isfinite(float(estimate[key])), (
                f"{name}.{key} must be finite; got {estimate[key]!r}"
            )
        # One-step debiasing removes exactly the reported penalty bias.
        assert math.isclose(
            float(estimate["theta_onestep"]),
            float(estimate["theta_plugin"]) - float(estimate["penalty_bias"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ), f"{name}: theta_onestep != theta_plugin - penalty_bias ({estimate})"


def test_atom_inference_reports_survive_json_roundtrip():
    fit = _fit_one_atom_periodic(seed=1)
    before = fit.atom_inference_reports
    reloaded = ManifoldSAE.from_json(fit.to_json())
    after = reloaded.atom_inference_reports
    assert after == before, (
        "atom_inference_reports must survive a JSON round-trip exactly; "
        f"before={before}, after={after}"
    )
