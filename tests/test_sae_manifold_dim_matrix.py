from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

# #1512 triage: the SAE-manifold dim-matrix sweep fits many configurations and
# runs past the standard Python-API CI runner budget (>240s in triage), so it
# is tagged slow and excluded from the directory-level `-m "not slow"` CI step
# (still collected, and run by a bare `pytest tests/` locally).
pytestmark = pytest.mark.slow


TOPOLOGIES = ("circle", "euclidean", "torus", "sphere")
D_ATOMS = (1, 2)
MIN_R2 = 0.05


def _mixed_signal_data(d_atom: int, *, n: int = 96, p: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_atom == 1:
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        latent = np.column_stack(
            [
                np.sin(2.0 * math.pi * t),
                np.cos(2.0 * math.pi * t),
                np.sin(4.0 * math.pi * t),
            ]
        )
    else:
        u = rng.uniform(0.0, 1.0, size=(n, 2))
        latent = np.column_stack(
            [
                np.sin(2.0 * math.pi * u[:, 0]),
                np.cos(2.0 * math.pi * u[:, 0]),
                np.sin(2.0 * math.pi * u[:, 1]),
                np.cos(2.0 * math.pi * u[:, 1]),
                u[:, 0] - u[:, 1],
            ]
        )
    mixing = rng.normal(size=(latent.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = latent @ mixing + 0.015 * rng.standard_normal((n, p))
    return z - z.mean(axis=0, keepdims=True)


def _fit_or_fail(z: np.ndarray, *, atom_topology: str, d_atom: int):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = gamfit.sae_manifold_fit(
                X=z,
                K=1,
                d_atom=d_atom,
                atom_topology=atom_topology,
                assignment="softmax",
                isometry_weight=0.0,
                ard_per_atom=False,
                sparsity_weight=0.01,
                smoothness_weight=0.01,
                decoder_incoherence_weight=0.0,
                n_iter=30,
                learning_rate=1.0,
                random_state=13,
            )
    except Exception as exc:
        pytest.fail(
            "SAE manifold topology x d_atom matrix documents the current "
            f"d=1 solver gap and related convergence work: "
            f"atom_topology={atom_topology!r}, d_atom={d_atom}, "
            f"{type(exc).__name__}: {exc}"
        )
    if (
        not np.all(np.isfinite(fit.fitted))
        or not np.isfinite(fit.reconstruction_r2)
        or fit.reconstruction_r2 < MIN_R2
    ):
        pytest.fail(
            "SAE manifold topology x d_atom matrix did not pass the "
            f"convergence guard for atom_topology={atom_topology!r}, "
            f"d_atom={d_atom}: reconstruction_r2={fit.reconstruction_r2!r}"
        )
    return fit


@pytest.mark.parametrize("atom_topology", TOPOLOGIES)
@pytest.mark.parametrize("d_atom", D_ATOMS)
def test_sae_manifold_fits_each_topology_dimension_pair(atom_topology: str, d_atom: int):
    z = _mixed_signal_data(d_atom, seed=100 + d_atom)
    fit = _fit_or_fail(z, atom_topology=atom_topology, d_atom=d_atom)

    assert fit.fitted.shape == z.shape
    assert fit.assignments.shape == (z.shape[0], 1)
    assert len(fit.atoms) == len(fit.coords) == 1
    assert fit.coords[0].shape == (z.shape[0], d_atom)
    assert fit.atoms[0].coords.shape == (z.shape[0], d_atom)
    assert fit.atom_topology == atom_topology
    assert fit.atom_topologies == [atom_topology]
    assert np.all(np.isfinite(fit.assignments))
    assert np.all(np.isfinite(fit.coords[0]))
