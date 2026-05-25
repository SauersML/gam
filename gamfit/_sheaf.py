"""Cellular-sheaf consistency penalty.

Python wrapper around the Rust ``SheafConsistencyPenalty`` analytic primitive
(see ``src/terms/sheaf.rs``). The Rust core operates on a flat stacked-stalk
vector ``s ∈ R^{Σ d_v}``; this wrapper accepts the more ergonomic per-layer
input formats (``dict[int, ndarray]``, ``list[ndarray]``, or a single 1-D
``ndarray``) and routes through ``__call__`` / ``gradient`` / ``hessian_diag``
/ ``hvp`` / ``harmonic_modes``.

Mathematical contract:

    P(s) = ½ · weight · ∑_e ‖δs[e]‖²
    δs[e] = R_e^{(u→e)}(s_{u_e}) − R_e^{(v→e)}(s_{v_e})

The sheaf Laplacian ``L = δᵀ δ`` is never materialised — gradients and HVPs
route through two matvecs (apply δ then δᵀ). ``harmonic_modes(tol)`` returns
the number of eigenvalues of ``L`` strictly below ``tol``; this generalises
the connected-component count of a graph Laplacian and quantifies the space
of globally consistent sections.

Reference: Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves",
J. Appl. Comput. Topol. 3 (2019).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._binding import rust_module as _rust_module


def _rust_sheaf_class() -> type[Any]:
    module = _rust_module()
    cls = getattr(module, "SheafConsistencyPenalty", None)
    if cls is None:
        raise AttributeError(
            "gamfit._rust does not expose SheafConsistencyPenalty; "
            "rebuild the local Rust extension"
        )
    return cls


def _normalize_edges(edges: Sequence[Any]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for edge in edges:
        pair = tuple(int(x) for x in edge)
        if len(pair) != 2:
            raise ValueError(
                f"SheafConsistencyPenalty.edges entries must be (u, v) pairs, got {edge!r}"
            )
        out.append((pair[0], pair[1]))
    return out


def _normalize_restrictions(
    restriction_ops: Sequence[Any],
) -> list[Any]:
    """Coerce restriction-op entries to numpy float64 arrays or (W_uv, W_vu) pairs.

    Accepts:
        * a 2-D array-like → ``EdgeRestriction::single(W_uv)``
        * a tuple/list ``(W_uv, W_vu)`` with ``W_vu`` optionally ``None``
    """
    out: list[Any] = []
    for entry in restriction_ops:
        if isinstance(entry, np.ndarray):
            out.append(np.ascontiguousarray(entry, dtype=np.float64))
            continue
        if isinstance(entry, (tuple, list)):
            if len(entry) == 1:
                w_uv = np.ascontiguousarray(entry[0], dtype=np.float64)
                out.append(w_uv)
                continue
            if len(entry) == 2:
                w_uv = np.ascontiguousarray(entry[0], dtype=np.float64)
                second = entry[1]
                if second is None:
                    out.append((w_uv, None))
                else:
                    w_vu = np.ascontiguousarray(second, dtype=np.float64)
                    out.append((w_uv, w_vu))
                continue
            raise ValueError(
                "SheafConsistencyPenalty.restriction_ops entries as tuples must have length 1 or 2"
            )
        # Last-resort coerce: treat as a 2-D array-like.
        arr = np.ascontiguousarray(entry, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                "SheafConsistencyPenalty.restriction_ops entries must be 2-D arrays or "
                "(W_uv, W_vu) tuples"
            )
        out.append(arr)
    return out


def _infer_stalk_dims(
    edges: Sequence[tuple[int, int]],
    restrictions: Sequence[Any],
    stalk_dims: Sequence[int] | None,
) -> list[int]:
    if stalk_dims is not None:
        return [int(d) for d in stalk_dims]
    # Auto-derive from restrictions: r_uv.ncols == d_u, r_vu.ncols == d_v (or
    # for single-restriction edges, r_uv.nrows == d_v since R_uv: R^{d_u} → R^{d_v}).
    if not edges:
        raise ValueError(
            "SheafConsistencyPenalty: stalk_dims is required when edges is empty"
        )
    max_vertex = max(max(u, v) for u, v in edges)
    dims: list[int | None] = [None] * (max_vertex + 1)
    for (u, v), restriction in zip(edges, restrictions):
        if isinstance(restriction, tuple):
            r_uv, r_vu = restriction
            if dims[u] is None:
                dims[u] = int(r_uv.shape[1])
            if r_vu is None:
                if dims[v] is None:
                    dims[v] = int(r_uv.shape[0])
            elif dims[v] is None:
                dims[v] = int(r_vu.shape[1])
        else:
            r_uv = restriction
            if dims[u] is None:
                dims[u] = int(r_uv.shape[1])
            if dims[v] is None:
                dims[v] = int(r_uv.shape[0])
    for vtx, d in enumerate(dims):
        if d is None:
            raise ValueError(
                f"SheafConsistencyPenalty: could not infer stalk dim for vertex {vtx}; "
                "pass stalk_dims explicitly"
            )
    return [int(d) for d in dims]


class SheafConsistencyPenalty:
    """Cellular-sheaf consistency penalty (public Python wrapper).

    Parameters
    ----------
    edges : sequence of ``(u, v)`` pairs
        Directed edges over vertex indices ``0..K``.
    restriction_ops : sequence
        One entry per edge. Each entry is either a 2-D array ``W_uv`` (single
        restriction; the head side is implicitly identity) or a tuple
        ``(W_uv, W_vu)`` (both sides explicit). ``W_vu`` may be ``None`` for
        the single-restriction form.
    stalk_dims : sequence of int, optional
        Per-vertex stalk dimensions ``d_v``. Inferred from ``restriction_ops``
        when omitted.
    weight : float, default 1.0
        Penalty multiplier on ``½ · ‖δs‖²``.
    target : str, default ``"z"``
        Slot name in the term registry. Diagnostic only; does not participate
        in the math.

    Examples
    --------
    >>> import numpy as np
    >>> import gamfit
    >>> sheaf = gamfit.SheafConsistencyPenalty(
    ...     edges=[(0, 1), (1, 2)],
    ...     restriction_ops=[np.eye(3), np.eye(3)],
    ...     weight=0.1,
    ... )
    >>> z = {0: np.zeros(3), 1: np.zeros(3), 2: np.zeros(3)}
    >>> float(sheaf(z))
    0.0
    >>> sheaf.harmonic_modes(1e-10)
    3
    """

    def __init__(
        self,
        edges: Sequence[Any],
        restriction_ops: Sequence[Any],
        *,
        stalk_dims: Sequence[int] | None = None,
        weight: float = 1.0,
        target: str | int = "z",
    ) -> None:
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"SheafConsistencyPenalty.weight must be finite and > 0, got {weight}"
            )
        edges_norm = _normalize_edges(edges)
        restrictions_norm = _normalize_restrictions(restriction_ops)
        if len(edges_norm) != len(restrictions_norm):
            raise ValueError(
                f"edge count {len(edges_norm)} != restriction count {len(restrictions_norm)}"
            )
        dims = _infer_stalk_dims(edges_norm, restrictions_norm, stalk_dims)
        cls = _rust_sheaf_class()
        self._rust = cls(
            edges_norm,
            restrictions_norm,
            dims,
            float(weight),
            target=target,
        )
        self._stalk_dims = tuple(dims)
        self._weight = float(weight)
        self._target = target

    # -- attribute passthrough --------------------------------------------

    @property
    def stalk_dims(self) -> tuple[int, ...]:
        return self._stalk_dims

    @property
    def total_dim(self) -> int:
        result = self._rust.total_dim
        return int(result)

    @property
    def num_edges(self) -> int:
        return int(self._rust.num_edges)

    @property
    def num_vertices(self) -> int:
        return int(self._rust.num_vertices)

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def target(self) -> Any:
        return self._target

    # -- core methods -----------------------------------------------------

    def _stack(self, z: Any) -> np.ndarray:
        if isinstance(z, np.ndarray) and z.ndim == 1:
            arr = np.ascontiguousarray(z, dtype=np.float64)
            if arr.shape[0] != self.total_dim:
                raise ValueError(
                    f"SheafConsistencyPenalty: input length {arr.shape[0]} "
                    f"!= total stalk dim {self.total_dim}"
                )
            return arr
        if isinstance(z, Mapping):
            parts = []
            for v in range(self.num_vertices):
                if v not in z:
                    raise KeyError(
                        f"SheafConsistencyPenalty: z_per_layer missing vertex {v}"
                    )
                parts.append(np.ascontiguousarray(z[v], dtype=np.float64).ravel())
        elif isinstance(z, Sequence):
            if len(z) != self.num_vertices:
                raise ValueError(
                    f"SheafConsistencyPenalty: expected {self.num_vertices} stalks, got {len(z)}"
                )
            parts = [np.ascontiguousarray(zv, dtype=np.float64).ravel() for zv in z]
        else:
            raise TypeError(
                "SheafConsistencyPenalty input must be a 1-D ndarray, a dict[int, ndarray], "
                f"or a list[ndarray]; got {type(z).__name__}"
            )
        for v, (part, expected) in enumerate(zip(parts, self._stalk_dims)):
            if part.shape[0] != expected:
                raise ValueError(
                    f"SheafConsistencyPenalty: stalk[{v}] has length {part.shape[0]}, "
                    f"expected {expected}"
                )
        return np.concatenate(parts).astype(np.float64, copy=False)

    def __call__(self, z: Any) -> float:
        flat = self._stack(z)
        return float(self._rust(flat))

    def value(self, z: Any) -> float:
        return self.__call__(z)

    def gradient(self, z: Any) -> np.ndarray:
        flat = self._stack(z)
        return np.asarray(self._rust.gradient(flat), dtype=np.float64)

    def hessian_diag(self, z: Any) -> np.ndarray:
        flat = self._stack(z)
        return np.asarray(self._rust.hessian_diag(flat), dtype=np.float64)

    def hvp(self, z: Any, v: Any) -> np.ndarray:
        flat = self._stack(z)
        v_flat = self._stack(v)
        return np.asarray(self._rust.hvp(flat, v_flat), dtype=np.float64)

    def harmonic_modes(self, tol: float = 1e-8) -> int:
        return int(self._rust.harmonic_modes(float(tol)))

    def __repr__(self) -> str:
        return repr(self._rust)


__all__ = ["SheafConsistencyPenalty"]
