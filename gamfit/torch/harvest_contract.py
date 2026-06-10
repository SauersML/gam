"""The two-tier harvest contract — activations for all, Fisher factors for few.

At 10^10-token scale the harvest is *not* uniform. Two costs are wildly
asymmetric:

* **Activations** ``x_n`` are one forward pass per token — cheap, and the SAE
  reconstruction objective genuinely needs *all* of them.
* **Output-Fisher factors** ``U_n`` (the low-rank ``J_nᵀ F J_n`` shard that
  :mod:`gamfit.torch.harvest` produces) cost ``~r`` backward passes per token.
  Materializing them for every one of 10^10 tokens is ruinous — and unnecessary:
  the *lens* (the pullback metric used to whiten the geometry) and the *gauge*
  (the output-Fisher used to gauge-fix the SAE) need far fewer rows than
  reconstruction does. A few well-chosen rows pin the metric; the rest only ever
  feed the Euclidean recon.

So the contract is **two-tier by design**: every row carries an activation; a
*designed importance subsample* additionally carries a Fisher factor. The pieces:

* :func:`select_importance_subsample` — a deterministic, seeded, importance-
  weighted selector. Given per-row importance scores it returns the row indices
  that get factors. Deterministic across calls with the same seed; no
  wall-clock, no arrival-order dependence (the selection is a hash-stable
  weighted draw without replacement).
* :class:`HarvestedRow` — a row that **may or may not** carry a Fisher factor.
  ``factor`` is ``None`` on the un-sampled majority.
* :func:`row_metric` / :class:`MetricConsumerContract` — the **graceful
  degradation** contract: every metric consumer must produce a valid metric for
  a factor-less row by falling back to the Euclidean / identity metric
  (``W_n = I``), and use the output-Fisher metric ``W_n = U_n U_nᵀ`` only when a
  factor is present. A missing factor is never an error; it is the common case.
* :func:`stream_harvest_with_selectivity` — the explicit streaming path: rows
  stream from a store, and *only the selected rows* invoke the (expensive)
  factor computation. The un-selected rows pass through carrying their
  activation alone.

The factor convention matches :mod:`gamfit.torch.harvest`: ``U_n`` is
``(p, r)`` with ``U_n U_nᵀ`` the rank-r truncation of the pullback ``G_n``, so a
present factor plugs straight into ``RowMetric::output_fisher``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "HarvestedRow",
    "MetricConsumerContract",
    "row_metric",
    "metric_quadratic_form",
    "select_importance_subsample",
    "stream_harvest_with_selectivity",
]


# ---------------------------------------------------------------------------
# Deterministic importance subsample selector
# ---------------------------------------------------------------------------


def _stable_uniform(key: int, seed: int) -> float:
    """A deterministic uniform-in-(0,1) draw keyed by ``(key, seed)``.

    Uses a BLAKE2b digest of the ``(seed, key)`` pair rather than a stateful RNG
    so the draw for a given row is independent of how many rows preceded it (no
    arrival-order / streaming-order dependence) and is bit-reproducible across
    processes. The top 53 bits of the digest become the mantissa of the uniform.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(int(seed).to_bytes(8, "little", signed=False))
    h.update(int(key).to_bytes(8, "little", signed=True))
    word = int.from_bytes(h.digest(), "little")
    # 53-bit mantissa → uniform in [0, 1); +0.5 then /2^53 keeps it in (0,1).
    mantissa = (word >> 11) & ((1 << 53) - 1)
    return (mantissa + 0.5) / float(1 << 53)


def select_importance_subsample(
    importance: Sequence[float] | np.ndarray,
    n_select: int,
    *,
    seed: int = 0,
    row_keys: Sequence[int] | None = None,
) -> np.ndarray:
    """Deterministically select ``n_select`` row indices for Fisher factors.

    This is the *designed* importance subsample: rows with larger ``importance``
    (e.g. the pullback trace estimate, the residual norm, or any saliency the
    caller supplies) are more likely to be chosen, but the draw is a seeded,
    hash-stable, weighted sampling *without replacement* so it is fully
    reproducible and order-independent.

    The mechanism is the Efraimidis–Spirakis weighted-reservoir key: each row
    ``i`` with weight ``w_i = max(importance_i, 0) + eps`` is assigned the key
    ``k_i = u_i ** (1 / w_i)`` with ``u_i`` a *deterministic* uniform keyed by
    ``(row_key_i, seed)``; the ``n_select`` rows with the largest keys are kept.
    A higher weight pushes ``k_i`` toward 1, so heavy rows are preferentially
    selected, exactly proportional to weight in expectation. Determinism comes
    from ``u_i`` being a hash of ``(row_key, seed)`` — not a sequential RNG — so
    two calls with the same seed (and same row keys) return identical indices.

    Parameters
    ----------
    importance
        Per-row non-negative importance scores, length ``n``.
    n_select
        Number of rows to select. Clamped to ``[0, n]``.
    seed
        Fixed seed for the deterministic uniform draws.
    row_keys
        Optional stable per-row identity keys (e.g. global token ids) so the
        selection is reproducible even when ``importance`` is computed over a
        shuffled / sharded stream. Defaults to the positional index ``0..n-1``.

    Returns
    -------
    np.ndarray
        Sorted ``int64`` array of selected row indices (positions into
        ``importance``), length ``min(n_select, n)``.
    """
    w = np.asarray(importance, dtype=np.float64).reshape(-1)
    n = int(w.shape[0])
    if row_keys is not None:
        keys = [int(k) for k in row_keys]
        if len(keys) != n:
            raise ValueError(
                f"row_keys length {len(keys)} must match importance length {n}"
            )
    else:
        keys = list(range(n))
    k = max(0, min(int(n_select), n))
    if k == 0 or n == 0:
        return np.empty((0,), dtype=np.int64)
    eps = 1.0e-12
    es_keys = np.empty((n,), dtype=np.float64)
    for i in range(n):
        weight = max(float(w[i]), 0.0) + eps
        u = _stable_uniform(keys[i], seed)
        # Efraimidis–Spirakis: key = u ** (1 / weight); larger weight ⇒ key → 1.
        es_keys[i] = u ** (1.0 / weight)
    # Largest keys win. Tie-break by row key for full determinism (np.argsort is
    # not stable across the descending flip, so break ties on the stable key).
    order = sorted(range(n), key=lambda i: (es_keys[i], keys[i]), reverse=True)
    chosen = np.array(sorted(order[:k]), dtype=np.int64)
    return chosen


# ---------------------------------------------------------------------------
# A harvested row that may or may not carry a Fisher factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarvestedRow:
    """One harvested token: an activation, optionally a Fisher factor.

    Attributes
    ----------
    activation
        ``(p,)`` hook-site activation ``x_n`` — present for *every* row.
    factor
        ``(p, r)`` output-Fisher factor ``U_n`` with ``U_n U_nᵀ`` the rank-r
        truncation of the pullback ``G_n``; ``None`` on the un-sampled majority.
        Matches the :mod:`gamfit.torch.harvest` convention so a present factor
        feeds ``RowMetric::output_fisher`` directly.
    mass_residual
        Optional ``trace(G_n) − Σ_{k≤r} λ_k`` truncation diagnostic; only
        meaningful when ``factor`` is present.
    row_key
        Optional stable identity (e.g. global token id) used by the deterministic
        selector and for reproducible streaming.
    """

    activation: np.ndarray
    factor: np.ndarray | None = None
    mass_residual: float | None = None
    row_key: int | None = None

    def __post_init__(self) -> None:
        act = np.asarray(self.activation, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "activation", act)
        if self.factor is not None:
            fac = np.asarray(self.factor, dtype=np.float64)
            if fac.ndim != 2 or fac.shape[0] != act.shape[0]:
                raise ValueError(
                    f"factor must be (p, r) with p={act.shape[0]}; got shape {fac.shape}"
                )
            object.__setattr__(self, "factor", fac)

    @property
    def has_factor(self) -> bool:
        return self.factor is not None

    @property
    def rank(self) -> int:
        """``r`` when a factor is present, else ``0`` (Euclidean fallback)."""
        return 0 if self.factor is None else int(self.factor.shape[1])


# ---------------------------------------------------------------------------
# Graceful-degradation metric contract
# ---------------------------------------------------------------------------


def row_metric(row: HarvestedRow) -> np.ndarray:
    """Dense ``(p, p)`` metric ``W_n`` for ``row`` — degrades gracefully.

    The contract every metric consumer must honor:

    * **factor present** → ``W_n = U_n U_nᵀ`` (the output-Fisher pullback metric,
      the rank-r truncation of ``G_n``);
    * **factor absent** → ``W_n = I_p`` (the Euclidean / identity metric).

    A factor-less row is *not* an error: it yields a valid, well-defined metric.
    This keeps the two-tier harvest sound — the recon-only majority of rows still
    have a usable metric, just the Euclidean one, and only the importance-sampled
    minority get the sharper output-Fisher geometry.
    """
    p = int(row.activation.shape[0])
    if row.factor is None:
        return np.eye(p, dtype=np.float64)
    U = row.factor
    return U @ U.T


def metric_quadratic_form(row: HarvestedRow, v: np.ndarray) -> float:
    """``vᵀ W_n v`` for a vector ``v``, matrix-free in the factor case.

    Mirrors :func:`row_metric`'s degradation but never forms ``W_n``:

    * factor present → ``‖U_nᵀ v‖²`` (so ``W_n = U_n U_nᵀ`` implicitly);
    * factor absent → ``‖v‖²`` (the identity metric).
    """
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    if vv.shape[0] != row.activation.shape[0]:
        raise ValueError(
            f"v has length {vv.shape[0]} but activation dim is "
            f"{row.activation.shape[0]}"
        )
    if row.factor is None:
        return float(vv @ vv)
    projected = row.factor.T @ vv
    return float(projected @ projected)


@dataclass(frozen=True)
class MetricConsumerContract:
    """Bundled graceful-degradation metric API a downstream consumer binds to.

    A consumer (lens whitening, gauge-fixing, a per-row reweighting) takes this
    contract instead of reaching into :class:`HarvestedRow` directly, so the
    factor-present / factor-absent branch lives in exactly one place. Both
    methods are total — they return a valid metric for *every* row.
    """

    def dense_metric(self, row: HarvestedRow) -> np.ndarray:
        """``(p, p)`` metric ``W_n`` (output-Fisher or identity fallback)."""
        return row_metric(row)

    def quadratic_form(self, row: HarvestedRow, v: np.ndarray) -> float:
        """``vᵀ W_n v`` (matrix-free; identity fallback when factor-less)."""
        return metric_quadratic_form(row, v)

    def is_degraded(self, row: HarvestedRow) -> bool:
        """``True`` when the row falls back to the Euclidean metric."""
        return not row.has_factor


# ---------------------------------------------------------------------------
# Streaming-from-store-with-selectivity
# ---------------------------------------------------------------------------


def stream_harvest_with_selectivity(
    rows: Iterable[tuple[int, np.ndarray]],
    *,
    importance: Sequence[float] | np.ndarray,
    n_select: int,
    compute_factor: Callable[[int, np.ndarray], tuple[np.ndarray, float]],
    seed: int = 0,
    row_keys: Sequence[int] | None = None,
) -> Iterator[HarvestedRow]:
    """Stream ``(row_index, activation)`` pairs, computing factors selectively.

    This is the explicit two-tier streaming path. The ``importance`` scores (one
    per row, indexed by the streamed ``row_index``) drive
    :func:`select_importance_subsample`, which is evaluated **once, up front**,
    so the selected set is fixed and deterministic before streaming begins —
    independent of the order rows arrive from the store.

    For each streamed row:

    * the activation is always carried (the cheap, all-rows tier);
    * **only if** the row's index is in the selected subsample does
      ``compute_factor(row_index, activation)`` run (the expensive ``~r``
      backward-pass tier), yielding ``(U_n, mass_residual)``;
    * otherwise the row is yielded factor-less and the metric consumer degrades
      to the Euclidean fallback.

    Parameters
    ----------
    rows
        Iterable of ``(row_index, activation)`` pairs streamed from a store.
        ``row_index`` indexes into ``importance`` / ``row_keys``.
    importance
        Per-row importance scores, length ``n`` (the full row population).
    n_select
        Number of rows that get factors.
    compute_factor
        Called only for selected rows: ``(row_index, activation) -> (U_n,
        mass_residual)``. Encapsulates the expensive per-row factor harvest.
    seed
        Fixed seed for the deterministic importance subsample.
    row_keys
        Optional stable per-row identity keys passed to the selector.

    Yields
    ------
    HarvestedRow
        In streamed order, each carrying its activation and — for the selected
        minority — its Fisher factor and mass residual.
    """
    selected = set(
        int(i)
        for i in select_importance_subsample(
            importance, n_select, seed=seed, row_keys=row_keys
        )
    )
    keys = [int(k) for k in row_keys] if row_keys is not None else None
    for row_index, activation in rows:
        idx = int(row_index)
        act = np.asarray(activation, dtype=np.float64).reshape(-1)
        key = keys[idx] if keys is not None else idx
        if idx in selected:
            factor, mass_residual = compute_factor(idx, act)
            yield HarvestedRow(
                activation=act,
                factor=np.asarray(factor, dtype=np.float64),
                mass_residual=float(mass_residual),
                row_key=key,
            )
        else:
            yield HarvestedRow(activation=act, factor=None, row_key=key)
