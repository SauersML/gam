"""Partial-supervision gauge-fix example — thin Python wrapper around the
Rust ``gam::identifiability::sae::partial_supervision_solve`` primitive.

All numerical linear algebra (Procrustes / anchor / soft-L2 ridge / QR
orthogonalization) lives in Rust; this module only handles argument
marshaling, shape validation that doesn't duplicate the Rust checks, and
result wrapping into the :class:`PartialSupervisionFit` dataclass.

Color-specific auxiliaries (HSV/RGB/LCh) reuse the existing
:class:`gamfit.GaugeCompanion` scorer; its loss is also a Rust pyfunction
(``equivariant_gauge_companion_loss``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, Any

import numpy as np

from .._binding import rust_module


SupMethod = Literal["procrustes", "anchor", "soft_l2"]
FreeConstraint = Literal["orthogonal_to_sup"] | None
AuxColorName = Literal["HSV", "RGB", "LCh"]


@dataclass(slots=True)
class PartialSupervisionFit:
    """Result of running :meth:`PartialSupervisionExample.fit`.

    Attributes
    ----------
    T_supervised : (N, d_supervised) ndarray
        Latent block tied to ``aux`` via the selected ``sup_method``.
    T_free : (N, d_free) ndarray
        Latent block decorrelated from ``T_supervised`` per
        ``free_constraint``.
    alignment_score : float
        ``1 - ‖T_sup_aligned - aux‖_F² / ‖aux‖_F²`` for every method
        (1.0 = perfect, 0.0 = no better than the constant-zero predictor).
    sup_method : str
        Echoes the ``sup_method`` argument used.
    free_constraint : str | None
        Echoes the ``free_constraint`` argument used.
    selected_weight : float | None
        REML-selected soft-L2 weight (only set for ``soft_l2``).
    map_R : (d_supervised, d_supervised) ndarray | None
        Procrustes rotation. ``None`` for the other methods.
    map_A : ndarray | None
        Affine slope (anchor) or ridge map (soft_l2). ``None`` for
        procrustes.
    map_b : ndarray | None
        Anchor affine intercept. ``None`` for the other methods.
    aux_score : float | None
        Optional :class:`gamfit.GaugeCompanion` loss when a color
        auxiliary was supplied; ``None`` otherwise.
    warnings : list[str]
        Identifiability-theorem precondition issues from
        ``gamfit.identifiability.check`` (empty list when every
        applicable theorem passed or when the check was disabled).
    report : IdentifiabilityReport | None
        Structured identifiability report. ``None`` when the check was
        disabled by passing ``check_identifiability=False``.
    """

    T_supervised: np.ndarray
    T_free: np.ndarray
    alignment_score: float
    sup_method: str
    free_constraint: str | None
    selected_weight: float | None = None
    map_R: np.ndarray | None = None
    map_A: np.ndarray | None = None
    map_b: np.ndarray | None = None
    aux_score: float | None = None
    warnings: list[str] = field(default_factory=list)
    report: Any = None

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Apply the fitted gauge to a new T-block of shape ``(N', T_dim)``.

        Routes through the same Rust primitive used by :meth:`fit`,
        passing the fitted map as the input ``T_supervised`` slice. For
        ``procrustes`` and ``soft_l2`` the supervised slice is
        right-multiplied by ``map_R`` / ``map_A``; for ``anchor`` the
        affine map is applied. The free slice is orthogonalized against
        the new supervised slice when a free constraint is configured.

        Returns
        -------
        ndarray
            Concatenated supervised and free latent block.

        Raises
        ------
        ValueError
            If ``X_new`` is not 2D or its width does not equal ``T_dim``.
        RuntimeError
            If the fitted map needed by ``sup_method`` is missing.
        """
        X = np.ascontiguousarray(np.asarray(X_new, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError(f"predict expects a 2D array, got shape {X.shape}")
        d_sup = self.T_supervised.shape[1]
        d_free = self.T_free.shape[1]
        if X.shape[1] != d_sup + d_free:
            raise ValueError(
                f"predict expects T_dim={d_sup + d_free} columns, got {X.shape[1]}"
            )
        T_sup_raw = X[:, :d_sup]
        T_free_raw = X[:, d_sup:]
        if self.sup_method == "procrustes" and self.map_R is not None:
            T_sup_new = T_sup_raw @ self.map_R
        elif self.sup_method == "anchor" and self.map_A is not None and self.map_b is not None:
            T_sup_new = T_sup_raw @ self.map_A + self.map_b
        elif self.sup_method == "soft_l2" and self.map_A is not None:
            T_sup_new = T_sup_raw @ self.map_A
        else:
            raise RuntimeError(
                f"predict: missing fitted map for sup_method={self.sup_method!r}"
            )
        # Use the Rust primitive's free-block projection so that the
        # orthogonalization stays in one place. We re-pass the (already
        # aligned) sup slice as both `t_sup` and `aux`; with method
        # "procrustes" and that input the SVD step returns R=I and leaves
        # T_sup_new unchanged, while the free-block QR projection runs.
        if self.free_constraint == "orthogonal_to_sup" and d_free > 0:
            result = rust_module().partial_supervision_solve(
                np.ascontiguousarray(T_sup_new),
                np.ascontiguousarray(T_sup_new),
                np.ascontiguousarray(T_free_raw),
                "procrustes",
                [],
                "orthogonal_to_sup",
            )
            T_free_new = np.asarray(result["t_free"], dtype=np.float64)
        else:
            T_free_new = T_free_raw.copy()
        return np.concatenate([T_sup_new, T_free_new], axis=1)


@dataclass(slots=True)
class PartialSupervisionExample:
    """Gauge-fix example for partial supervision.

    See :func:`partial_supervision` for the user-facing entry point.
    """

    T_dim: int
    aux: np.ndarray
    d_supervised: int
    d_free: int
    sup_method: SupMethod = "procrustes"
    free_constraint: FreeConstraint = "orthogonal_to_sup"
    anchor_idx: Sequence[int] = field(default_factory=lambda: (0,))
    aux_name: AuxColorName | None = None

    def __post_init__(self) -> None:
        if self.d_supervised < 0 or self.d_free < 0:
            raise ValueError("d_supervised and d_free must be non-negative")
        if self.d_supervised + self.d_free != self.T_dim:
            raise ValueError(
                f"d_supervised + d_free must equal T_dim; got "
                f"{self.d_supervised} + {self.d_free} != {self.T_dim}"
            )
        if self.sup_method not in ("procrustes", "anchor", "soft_l2"):
            raise ValueError(
                f"sup_method must be 'procrustes', 'anchor' or 'soft_l2'; "
                f"got {self.sup_method!r}"
            )
        if self.free_constraint not in ("orthogonal_to_sup", None):
            raise ValueError(
                f"free_constraint must be 'orthogonal_to_sup' or None; "
                f"got {self.free_constraint!r}"
            )
        aux_arr = np.ascontiguousarray(np.asarray(self.aux, dtype=np.float64))
        if aux_arr.ndim != 2:
            raise ValueError(
                f"aux must be a 2D array (N, d_supervised); got shape {aux_arr.shape}"
            )
        if aux_arr.shape[1] != self.d_supervised:
            raise ValueError(
                f"aux must have d_supervised={self.d_supervised} columns; "
                f"got {aux_arr.shape[1]}"
            )
        self.aux = aux_arr

    def fit(
        self,
        X: np.ndarray,
        T_init: np.ndarray | None = None,
        *,
        check_identifiability: bool = True,
    ) -> PartialSupervisionFit:
        """Run the gauge-fix example.

        Parameters
        ----------
        X : (N, p) ndarray
            Predictor block. Used to derive ``T_init`` (thin SVD on the
            centred predictor matrix) when ``T_init`` is not provided.
        T_init : (N, T_dim) ndarray, optional
            Initial latent block. Required when ``X.shape[1] < T_dim``.
        check_identifiability : bool, default True
            Run ``gamfit.identifiability.check`` on the result.

        Returns
        -------
        PartialSupervisionFit
            Aligned supervised/free latent blocks and fitted gauge maps.

        Raises
        ------
        ValueError
            If shapes are inconsistent or an initializer is required but not
            supplied.
        """
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        N = X.shape[0]
        if self.aux.shape[0] != N:
            raise ValueError(
                f"aux has {self.aux.shape[0]} rows but X has {N}; mismatch"
            )
        T = self._resolve_T_init(X, T_init, N)
        T_sup = np.ascontiguousarray(T[:, : self.d_supervised])
        T_free_raw = np.ascontiguousarray(T[:, self.d_supervised :])

        free_arg = (
            "orthogonal_to_sup" if self.free_constraint == "orthogonal_to_sup" else "none"
        )
        anchor_arg = [int(i) for i in self.anchor_idx]

        result = rust_module().partial_supervision_solve(
            T_sup,
            self.aux,
            T_free_raw,
            str(self.sup_method),
            anchor_arg,
            free_arg,
        )

        sup_aligned = np.ascontiguousarray(np.asarray(result["t_supervised"], dtype=np.float64))
        free_aligned = np.ascontiguousarray(np.asarray(result["t_free"], dtype=np.float64))
        alignment_score = float(result["alignment_score"])
        sw = result["selected_weight"]
        selected_weight = None if sw is None else float(sw)
        mr = result["map_r"]
        map_R = None if mr is None else np.ascontiguousarray(np.asarray(mr, dtype=np.float64))
        ma = result["map_a"]
        map_A = None if ma is None else np.ascontiguousarray(np.asarray(ma, dtype=np.float64))
        mb = result["map_b"]
        map_b = None if mb is None else np.ascontiguousarray(np.asarray(mb, dtype=np.float64))

        aux_score: float | None = None
        if self.aux_name is not None:
            # Reuse the existing GaugeCompanion scorer (also Rust-backed).
            from .._equivariant import GaugeCompanion
            companion = GaugeCompanion(
                aux=self.aux_name, d_aux=self.d_supervised, aux_values=self.aux,
            )
            aux_score = companion.loss(sup_aligned)

        fit_result = PartialSupervisionFit(
            T_supervised=sup_aligned,
            T_free=free_aligned,
            alignment_score=alignment_score,
            sup_method=self.sup_method,
            free_constraint=self.free_constraint,
            selected_weight=selected_weight,
            map_R=map_R,
            map_A=map_A,
            map_b=map_b,
            aux_score=aux_score,
        )

        if bool(check_identifiability):
            from ..identifiability import check as _check_identifiability

            report = _check_identifiability(fit_result, aux=self.aux)
            fit_result.report = report
            fit_result.warnings = report.as_warnings()

        return fit_result

    def _resolve_T_init(
        self, X: np.ndarray, T_init: np.ndarray | None, N: int
    ) -> np.ndarray:
        if T_init is not None:
            T = np.ascontiguousarray(np.asarray(T_init, dtype=np.float64))
            if T.shape != (N, self.T_dim):
                raise ValueError(
                    f"T_init must have shape ({N}, {self.T_dim}); got {T.shape}"
                )
            return T
        if X.shape[1] < self.T_dim:
            raise ValueError(
                f"X has {X.shape[1]} columns but T_dim={self.T_dim}; "
                f"pass T_init=... to initialize a wider latent block"
            )
        # Centre and thin-SVD via the Rust faer bridge.
        scores = rust_module().thin_svd_scores(np.ascontiguousarray(X), int(self.T_dim))
        return np.ascontiguousarray(np.asarray(scores, dtype=np.float64))


def partial_supervision(
    T_dim: int,
    aux: np.ndarray,
    d_supervised: int,
    d_free: int,
    sup_method: SupMethod = "procrustes",
    free_constraint: FreeConstraint = "orthogonal_to_sup",
    anchor_idx: Sequence[int] = (0,),
    aux_name: AuxColorName | None = None,
) -> PartialSupervisionExample:
    """Build a partial-supervision gauge-fix example.

    See module docstring and :class:`PartialSupervisionFit` for the
    semantics. All numerical work happens in Rust via
    ``gam::identifiability::sae::partial_supervision_solve``; this Python
    layer is a marshal-only wrapper.

    Examples
    --------
    >>> import numpy as np, gamfit
    >>> rng = np.random.default_rng(0)
    >>> hsv = rng.standard_normal((200, 3))
    >>> example = gamfit.examples.partial_supervision(
    ...     T_dim=6, aux=hsv, d_supervised=3, d_free=3,
    ...     sup_method='procrustes',
    ...     free_constraint='orthogonal_to_sup',
    ... )
    >>> fit = example.fit(rng.standard_normal((200, 8)))
    >>> fit.T_supervised.shape, fit.T_free.shape
    ((200, 3), (200, 3))

    Returns
    -------
    PartialSupervisionExample
        Example object; call ``.fit(X, T_init=None)`` to run the Rust solve.

    Raises
    ------
    ValueError
        Raised during example construction for invalid dimensions, methods, or
        auxiliary array shape.
    """
    return PartialSupervisionExample(
        T_dim=T_dim,
        aux=np.asarray(aux),
        d_supervised=d_supervised,
        d_free=d_free,
        sup_method=sup_method,
        free_constraint=free_constraint,
        anchor_idx=tuple(anchor_idx),
        aux_name=aux_name,
    )


__all__ = [
    "partial_supervision",
    "PartialSupervisionExample",
    "PartialSupervisionFit",
]
