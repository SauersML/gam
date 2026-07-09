from __future__ import annotations

import numpy as np
import pytest

import gamfit

torch = pytest.importorskip("torch")


def _planted_circle(n: int, p: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.05, 0.95, size=n)
    base = np.column_stack(
        [
            np.cos(2.0 * np.pi * t),
            np.sin(2.0 * np.pi * t),
            0.5 * np.cos(4.0 * np.pi * t),
            0.5 * np.sin(4.0 * np.pi * t),
        ]
    )
    mix = rng.standard_normal((base.shape[1], p))
    mix /= np.linalg.norm(mix, axis=0, keepdims=True) + 1.0e-12
    return base @ mix + 0.01 * rng.standard_normal((n, p))


def _circle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raw = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.abs((raw + 0.5) % 1.0 - 0.5)


def test_distilled_encoder_reports_fallback_and_keeps_heldout_circle_gap_small() -> None:
    x_train = _planted_circle(96, seed=10)
    x_test = _planted_circle(32, seed=11)
    fit = gamfit.sae_manifold_fit(
        X=x_train,
        K=1,
        d_atom=1,
        atom_topology="circle",
        atom_basis="periodic",
        assignment="softmax",
        n_iter=25,
        random_state=2,
    )

    encoder = fit.distill_encoder(
        x_train,
        hidden=(48, 48),
        epochs=250,
        batch_size=32,
        validation_fraction=0.2,
        random_state=7,
        tolerance_multiplier=1.5,
    )

    t_init, _logits = encoder.predict_initializers(x_test)
    exact = fit.converged_latents(x_test)
    coord_gap = float(np.mean(_circle_delta(t_init[0, :, :1], exact["coords"][0])))
    assert coord_gap < 0.05

    encoded, stats = fit.encode(x_test, encoder=encoder, return_stats=True)
    exact_assign = np.asarray(exact["assignments"], dtype=float)
    rel_assignment_gap = np.linalg.norm(encoded - exact_assign) / max(
        np.linalg.norm(exact_assign), 1.0e-12
    )
    assert rel_assignment_gap < 0.05
    assert stats["rows"] == x_test.shape[0]
    assert stats["accepted_rows"] + stats["fallback_rows"] == x_test.shape[0]
    assert stats["fallback_rate"] == pytest.approx(
        stats["fallback_rows"] / x_test.shape[0]
    )
    assert encoder.last_stats is not None
    assert encoder.last_stats.to_dict() == stats


class _FakeModel:
    assignment = "softmax"
    tau = 1.0
    alpha = 1.0
    jumprelu_threshold = 0.0
    _atom_dims = [1]
    _basis_kinds = ["euclidean"]

    def __init__(self) -> None:
        self.calls = 0

    def converged_latents(self, X):
        x = np.asarray(X, dtype=float)
        t = x[:, :1].copy()
        logits = np.zeros((x.shape[0], 1), dtype=float)
        return {
            "coords": [t],
            "assignments": np.ones((x.shape[0], 1), dtype=float),
            "logits": logits,
            "fitted": x.copy(),
        }

    def _oos_payload(self, X, *, t_init=None, a_init=None):
        self.calls += 1
        x = np.asarray(X, dtype=float)
        return {
            "atoms": [
                {
                    "on_atom_coords_t": x[:, :1].copy(),
                }
            ],
            "assignments_z": np.ones((x.shape[0], 1), dtype=float),
            "logits": np.zeros((x.shape[0], 1), dtype=float),
            "fitted": x.copy(),
        }


def test_distill_module_uses_exact_probe_for_fallback_stats() -> None:
    from gamfit.distill import distill_encoder, encode_with_fallback

    x = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    model = _FakeModel()
    encoder = distill_encoder(
        model,
        x,
        hidden=8,
        epochs=20,
        batch_size=4,
        validation_fraction=0.25,
        random_state=0,
    )
    encoded, stats = encode_with_fallback(model, x, encoder)
    np.testing.assert_allclose(encoded, np.ones((x.shape[0], 1)))
    assert stats.rows == x.shape[0]
    assert stats.exact_probe_rows == x.shape[0]
    assert model.calls == 1
