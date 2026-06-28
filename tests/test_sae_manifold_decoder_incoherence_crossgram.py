from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _redundant_two_atom_data(n: int = 160, p: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(n, 2))
    latent = np.column_stack(
        [
            np.sin(2.0 * math.pi * t[:, 0]),
            np.cos(2.0 * math.pi * t[:, 0]),
            np.sin(2.0 * math.pi * t[:, 1]),
            np.cos(2.0 * math.pi * t[:, 1]),
            np.sin(2.0 * math.pi * (t[:, 0] + t[:, 1])),
        ]
    )
    mixing = rng.normal(size=(latent.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = latent @ mixing + 0.02 * rng.standard_normal((n, p))
    return z - z.mean(axis=0, keepdims=True)


def _fit(z: np.ndarray, *, decoder_incoherence_weight: float, seed: int):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = gamfit.sae_manifold_fit(
                X=z,
                K=2,
                d_atom=2,
                atom_topology="euclidean",
                assignment="softmax",
                isometry_weight=0.0,
                ard_per_atom=False,
                sparsity_weight=0.01,
                smoothness_weight=0.01,
                decoder_incoherence_weight=decoder_incoherence_weight,
                n_iter=35,
                learning_rate=1.0,
                random_state=seed,
            )
    except Exception as exc:
        pytest.fail(
            "decoder_incoherence cross-Gram comparison requires the multi-atom "
            "solver to converge: "
            f"weight={decoder_incoherence_weight}, {type(exc).__name__}: {exc}"
        )
    if (
        not np.all(np.isfinite(fit.fitted))
        or not np.isfinite(fit.reconstruction_r2)
        or fit.reconstruction_r2 < 0.05
    ):
        pytest.fail(
            "decoder_incoherence cross-Gram comparison produced a degenerate "
            f"fit for weight={decoder_incoherence_weight}: "
            f"reconstruction_r2={fit.reconstruction_r2!r}"
        )
    return fit


def _decoder_cross_gram_energy(fit) -> float:
    blocks = [np.asarray(block, dtype=float) for block in fit.decoder_blocks]
    total = 0.0
    for left in range(len(blocks)):
        for right in range(left + 1, len(blocks)):
            # Stored blocks are (basis_rows, p_out). Cross-atom decoder
            # directions live in output space, so compare row-space overlap.
            cross = blocks[left] @ blocks[right].T
            total += float(np.sum(cross * cross))
    return total


@pytest.mark.slow
def test_decoder_incoherence_reduces_recovered_cross_atom_decoder_cross_gram():
    z = _redundant_two_atom_data()

    fit_off = _fit(z, decoder_incoherence_weight=0.0, seed=33)
    fit_on = _fit(z, decoder_incoherence_weight=50.0, seed=33)

    off_energy = _decoder_cross_gram_energy(fit_off)
    on_energy = _decoder_cross_gram_energy(fit_on)
    if not (np.isfinite(off_energy) and np.isfinite(on_energy)) or off_energy <= 1e-10:
        pytest.fail(
            "decoder_incoherence cross-Gram comparison produced a degenerate "
            f"off fit: off_energy={off_energy!r}, on_energy={on_energy!r}"
        )

    assert on_energy < 0.90 * off_energy, (
        "decoder_incoherence_weight=50.0 should reduce the recovered "
        "cross-atom decoder cross-Gram relative to decoder_incoherence_weight=0.0; "
        f"off_energy={off_energy:.6g}, on_energy={on_energy:.6g}"
    )
