"""End-to-end WP-D output-Fisher shard -> SAE fit round-trip (#980).

Exercises the full magic-by-default wiring: a tiny synthetic torch model is
harvested for per-token output-Fisher factors (``gamfit.torch.harvest``), and the
resulting ``(X, U, mass_residual)`` shard is fed to the public
``gamfit.sae_manifold_fit`` via ``fisher_factors=``. The *presence* of the shard
activates ``RowMetric::OutputFisher`` in the Rust core -- there is no flag.

The amended contract (#980) says the metric is installed for the gauge / lens
only: ``RowMetric::OutputFisher`` does NOT whiten the reconstruction likelihood
(``whitens_likelihood() == False``), so with the isometry gauge off (the default
``isometry_weight=0.0``) the optimized objective and the fitted reconstruction
are bit-for-bit identical to the no-shard Euclidean fit. We assert exactly that:

* the fit runs and returns a ``ManifoldSAE``;
* its provenance is reported as ``"OutputFisher"`` (no-shard run reports
  ``"Euclidean"``);
* the per-row ``mass_residual`` truncation diagnostic rides into the report;
* the DATA-FIT (fitted reconstruction + penalized quasi-Laplace criterion) is identical to the no-shard
  run -- the gauge/lens metric leaves the likelihood untouched.

Fixed seeds throughout; no clock entropy. Requires torch (skipped otherwise).
"""

from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
torch = pytest.importorskip("torch")

from gamfit.torch.harvest import (  # noqa: E402
    HarvestShard,
    harvest_output_fisher_factors,
    load_harvest_shard,
    save_harvest_shard,
)


class _LinearHead(torch.nn.Module):
    """``logits = x @ Wᵀ`` with hook site = the identity-passed input ``x``.

    ``feature`` is an ``nn.Identity`` whose output is the hook-site activation
    ``x_n`` (the SAE response row); ``head`` is a fixed linear map so the output
    Jacobian ``∂logits/∂x_n = W`` is exact and token-independent.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.feature = torch.nn.Identity()
        self.head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature(x))


def _harvest_shard(n: int, p: int, classes: int, rank: int) -> HarvestShard:
    """A deterministic synthetic shard: activations on a planted circle so the
    SAE has real low-dimensional structure to reconstruct, plus the exact
    output-Fisher factors of a fixed linear head over those activations.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # Planted 1D circular structure in the activations (the SAE response rows).
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.standard_normal((2, p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    x_np = (harm @ mixing + 0.02 * rng.standard_normal((n, p))).astype(np.float64)
    x_np -= x_np.mean(axis=0, keepdims=True)

    w_np = rng.standard_normal((classes, p)).astype(np.float64)
    model = _LinearHead(torch.from_numpy(w_np)).to(torch.float64)
    shard = harvest_output_fisher_factors(
        model,
        model.feature,
        torch.from_numpy(x_np).to(torch.float64),
        rank=rank,
        oversample=3,
        n_iter=4,
        trace_probes=p,
        seed=0,
    )
    assert isinstance(shard, HarvestShard)
    assert shard.X.shape == (n, p)
    assert shard.U.shape == (n, p, rank)
    assert shard.mass_residual.shape == (n,)
    return shard


def _fit(x: np.ndarray, fisher_factors=None):
    # Default isometry_weight=0.0 (gauge off): the output-Fisher metric drives
    # only the gauge, so the DATA-FIT is identical to the Euclidean run.
    return gamfit.sae_manifold_fit(
        X=x,
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=8,
        random_state=0,
        fisher_factors=fisher_factors,
    )


def test_fisher_shard_roundtrip_installs_metric_without_touching_data_fit() -> None:
    n, p, classes, rank = 60, 8, 5, 3
    shard = _harvest_shard(n, p, classes, rank)
    x = np.asarray(shard.X, dtype=np.float64)

    # Baseline: no shard -> Euclidean metric, today's behaviour.
    fit_base = _fit(x, fisher_factors=None)
    # With the harvested shard passed straight through (HarvestShard accepted).
    fit_fisher = _fit(x, fisher_factors=shard)

    # 1) The fit runs and returns a model both ways.
    assert fit_base is not None
    assert fit_fisher is not None

    # 2) Provenance flips to OutputFisher exactly when a shard is supplied.
    assert fit_base.metric_provenance == "Euclidean"
    assert fit_fisher.metric_provenance == "OutputFisher"

    # 3) The per-row truncation diagnostic rides into the report (and only then).
    assert fit_base.fisher_mass_residual is None
    assert fit_fisher.fisher_mass_residual is not None
    np.testing.assert_allclose(
        fit_fisher.fisher_mass_residual,
        np.asarray(shard.mass_residual, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    # 4) DATA-FIT IDENTICAL: the gauge/lens metric leaves the likelihood
    #    untouched (OutputFisher.whitens_likelihood() == False) and the gauge is
    #    off by default, so the fitted reconstruction + REML score are bit-for-bit
    #    the same as the no-shard Euclidean fit.
    np.testing.assert_array_equal(fit_fisher.fitted, fit_base.fitted)
    assert fit_fisher.penalized_quasi_laplace_criterion == fit_base.penalized_quasi_laplace_criterion
    assert fit_fisher.reconstruction_r2 == fit_base.reconstruction_r2


def test_fisher_shard_roundtrip_accepts_loaded_npz_dict(tmp_path) -> None:
    """The ``load_harvest_shard`` f64 dict form is accepted identically and
    produces the same OutputFisher provenance + identical data-fit."""
    n, p, classes, rank = 48, 7, 4, 2
    shard = _harvest_shard(n, p, classes, rank)
    x = np.asarray(shard.X, dtype=np.float64)

    save_harvest_shard(shard, tmp_path / "shard.npz")
    loaded = load_harvest_shard(tmp_path / "shard.npz")  # f64 dict {X,U,mass_residual,rank}

    fit_base = _fit(x, fisher_factors=None)
    fit_dict = _fit(x, fisher_factors=loaded)

    assert fit_dict.metric_provenance == "OutputFisher"
    assert fit_dict.fisher_mass_residual is not None
    np.testing.assert_array_equal(fit_dict.fitted, fit_base.fitted)
    assert fit_dict.penalized_quasi_laplace_criterion == fit_base.penalized_quasi_laplace_criterion
