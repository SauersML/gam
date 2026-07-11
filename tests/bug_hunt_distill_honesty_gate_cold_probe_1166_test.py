"""Regression for #1166 — the distilled-encoder honesty gate must measure the
encoder against the COLD canonical feature map, not a probe warm-started from
the encoder's own guess.

The acceptance gate in ``gamfit.distill.encode_with_fallback`` compares the
encoder's fast prediction against an "exact" frozen-decoder solve. If that solve
is warm-started from the encoder's own initializers, the finite-iteration Newton
refinement is biased toward the encoder guess, so the measured error is
systematically too small and rows that DISAGREE with the canonical ``encode(X)``
feature map can still pass. The fix solves the reference probe cold
(``_oos_payload(x)`` with no ``t_init``/``a_init``), which is the same solve the
public ``encode(X)`` returns and that ``distill_encoder`` calibrates against.

This test is intentionally torch-free: ``encode_with_fallback`` only duck-types
the encoder, so a fixed fake encoder + a model whose warm/cold solves diverge
pins the behavior without building the Rust extension or torch.
"""

from __future__ import annotations

import numpy as np

from gamfit.distill import encode_with_fallback


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class _FixedEncoder:
    """Duck-typed :class:`DistilledEncoder` returning controlled initializers."""

    assignment = "softmax"
    tau = 1.0
    alpha = 1.0
    threshold_gate_threshold = 0.0
    atom_dims = (1, 1)
    coord_periods = ((None,), (None,))
    assignment_tolerance = 0.05
    coord_tolerance = 0.05

    def __init__(self, coords_knm: np.ndarray, logits_nk: np.ndarray) -> None:
        self._coords = np.ascontiguousarray(np.asarray(coords_knm, dtype=np.float64))
        self._logits = np.ascontiguousarray(np.asarray(logits_nk, dtype=np.float64))
        self.last_stats = None

    def predict_initializers(self, X):  # noqa: ARG002 - X unused; fixed output
        return self._coords, self._logits


class _DivergentModel:
    """Frozen-decoder stand-in whose warm probe echoes its seed while the cold
    canonical solve lands somewhere different.

    The warm path (``t_init``/``a_init`` given) returns the seed verbatim — the
    pathological "biased toward the encoder guess" behavior. The cold path
    (no seed) returns the canonical ``cold_*`` solution. A correct gate compares
    the encoder guess against the cold solution and therefore rejects.
    """

    def __init__(self, cold_coords_knm: np.ndarray, cold_logits_nk: np.ndarray) -> None:
        self._cold_coords = np.ascontiguousarray(np.asarray(cold_coords_knm, dtype=np.float64))
        self._cold_logits = np.ascontiguousarray(np.asarray(cold_logits_nk, dtype=np.float64))
        self.warm_calls = 0
        self.cold_calls = 0

    def _payload(self, coords_knm: np.ndarray, logits_nk: np.ndarray) -> dict:
        atoms = [
            {"on_atom_coords_t": np.ascontiguousarray(coords_knm[k])}
            for k in range(coords_knm.shape[0])
        ]
        return {
            "atoms": atoms,
            "assignments_z": _softmax(logits_nk),
            "logits": np.ascontiguousarray(logits_nk),
            "fitted": np.zeros((logits_nk.shape[0], 1), dtype=np.float64),
        }

    def _oos_payload(self, X, *, t_init=None, a_init=None):  # noqa: ARG002
        if t_init is None and a_init is None:
            self.cold_calls += 1
            return self._payload(self._cold_coords, self._cold_logits)
        self.warm_calls += 1
        return self._payload(
            np.asarray(t_init, dtype=np.float64),
            np.asarray(a_init, dtype=np.float64),
        )


def test_honesty_gate_uses_cold_canonical_probe_not_warm_self_probe() -> None:
    n = 4
    # Encoder guess: coords at 0, logits favor atom 0.
    enc_coords = np.zeros((2, n, 1), dtype=np.float64)
    enc_logits = np.tile(np.array([[3.0, 0.0]]), (n, 1))
    encoder = _FixedEncoder(enc_coords, enc_logits)

    # Canonical cold solve: coords far from 0, logits favor atom 1 — i.e. the
    # encoder is genuinely wrong about these rows under the real feature map.
    cold_coords = np.full((2, n, 1), 1.0, dtype=np.float64)
    cold_logits = np.tile(np.array([[0.0, 3.0]]), (n, 1))
    model = _DivergentModel(cold_coords, cold_logits)

    encoded, stats = encode_with_fallback(model, np.zeros((n, 3)), encoder)

    # The gate must have consulted ONLY the cold canonical solve.
    assert model.cold_calls == 1
    assert model.warm_calls == 0

    # Every row disagrees with the canonical solve, so every row must fall back.
    # On the buggy warm-probe gate the probe echoes the encoder guess, the error
    # reads ~0, and all rows are wrongly accepted (fallback_rows == 0).
    assert stats.fallback_rows == n
    assert stats.accepted_rows == 0
    assert stats.exact_probe_rows == n

    # Returned assignments reproduce the canonical cold solve (favoring atom 1),
    # NOT the encoder's fast guess (which favored atom 0).
    cold_assign = _softmax(cold_logits)
    np.testing.assert_allclose(encoded, cold_assign)
    fast_assign = _softmax(enc_logits)
    assert np.max(np.abs(encoded - fast_assign)) > 0.5


def test_honesty_gate_accepts_when_encoder_matches_cold_solve() -> None:
    # Discriminator sibling: when the encoder guess agrees with the cold
    # canonical solve, the gate accepts and returns the fast values.
    n = 5
    enc_coords = np.zeros((2, n, 1), dtype=np.float64)
    enc_logits = np.tile(np.array([[3.0, 0.0]]), (n, 1))
    encoder = _FixedEncoder(enc_coords, enc_logits)

    # Cold solve matches the encoder guess exactly.
    model = _DivergentModel(enc_coords.copy(), enc_logits.copy())

    encoded, stats = encode_with_fallback(model, np.zeros((n, 3)), encoder)

    assert model.cold_calls == 1
    assert model.warm_calls == 0
    assert stats.accepted_rows == n
    assert stats.fallback_rows == 0
    np.testing.assert_allclose(encoded, _softmax(enc_logits))
