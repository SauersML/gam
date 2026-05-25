"""Partial-supervision gauge-fix recipe.

`partial_supervision(T_dim, aux, d_supervised, d_free, sup_method,
free_constraint)` ties the first `d_supervised` columns of a latent
factor block to a numeric auxiliary signal (Procrustes / anchor /
soft-L2), and projects the remaining `d_free` columns onto the
orthogonal complement of the supervised block.

The recipe is a *runner*: it owns the gauge-fix step in latent
T-space and produces a `PartialSupervisionFit` dataclass with the
aligned `T_supervised`, decorrelated `T_free`, and an alignment
score documented in :class:`PartialSupervisionFit`.

Color-specific auxiliaries (HSV/RGB/LCh) are supported via the
existing :class:`gamfit.GaugeCompanion` scoring path when ``aux`` is
passed as one of the strings ``"HSV"``, ``"RGB"``, ``"LCh"`` with a
companion ``aux_values=`` array — the recipe then reuses
``GaugeCompanion.loss`` to populate ``aux_score`` alongside the
numeric Procrustes alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, Any

import numpy as np


SupMethod = Literal["procrustes", "anchor", "soft_l2"]
FreeConstraint = Literal["orthogonal_to_sup"] | None
AuxColorName = Literal["HSV", "RGB", "LCh"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _orthogonal_procrustes(T_sup: np.ndarray, aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve min ‖T_sup R - aux‖_F² s.t. RᵀR = I via SVD of T_supᵀ aux.

    Returns ``(R, T_sup @ R)``. Squared dimensions required.
    """
    if T_sup.shape != aux.shape:
        raise ValueError(
            f"orthogonal_procrustes requires T_sup.shape == aux.shape; "
            f"got T_sup={T_sup.shape}, aux={aux.shape}"
        )
    M = T_sup.T @ aux
    U, _s, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return R, T_sup @ R


def _anchor_affine(
    T_sup: np.ndarray, aux: np.ndarray, anchor_idx: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares affine map A, b minimizing ‖T_sup A + b - aux‖² over anchors.

    Pins the supervised block at the anchor rows (exactly when len(anchor)
    >= d_sup + 1 and the anchor T-block has full column rank); otherwise
    delivers the least-squares projection that minimizes anchor residuals
    and applies it to every row.
    """
    if len(anchor_idx) == 0:
        raise ValueError("anchor method requires at least one anchor row")
    Ta = T_sup[list(anchor_idx), :]
    Aa = aux[list(anchor_idx), :]
    # Solve [Ta | 1] @ [A; b] = Aa in least-squares sense.
    ones = np.ones((Ta.shape[0], 1), dtype=Ta.dtype)
    design = np.concatenate([Ta, ones], axis=1)
    coef, *_ = np.linalg.lstsq(design, Aa, rcond=None)
    A = coef[:-1, :]
    b = coef[-1, :]
    fitted = T_sup @ A + b
    return A, b, fitted


def _soft_l2_select_weight(T_sup: np.ndarray, aux: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """REML-style 1D selection of the soft-L2 weight λ.

    Minimizes a GCV-style criterion over a log-spaced grid of λ values.
    For a fixed orthogonal R, the soft-L2 objective is
        ‖T_sup R - aux‖_F² + λ ‖R‖_F² = ‖T_sup R - aux‖_F² + λ d.
    The minimizing R for the data-fit term is the Procrustes solution,
    so we use a ridge-on-the-affine-map proxy:
        argmin_A ‖T_sup A - aux‖² + λ ‖A‖² → A_λ = (TᵀT + λI)⁻¹ Tᵀ aux.
    The GCV score is ‖T_sup A_λ - aux‖² / (N - tr(H_λ))² with
    H_λ = T_sup (TᵀT + λI)⁻¹ T_supᵀ.
    """
    N, d = T_sup.shape
    G = T_sup.T @ T_sup
    rhs = T_sup.T @ aux
    eigvals, eigvecs = np.linalg.eigh(G)
    Ut_aux = eigvecs.T @ rhs
    # Avoid log(0); floor eigenvalues at a tiny positive number for the grid.
    floor = float(max(1e-12, eigvals[-1] * 1e-10))
    grid = np.geomspace(floor, max(eigvals[-1] * 1e3, floor * 1e6), num=64)
    best_score = np.inf
    best_lam = float(grid[0])
    best_A: np.ndarray = np.zeros_like(rhs)
    best_resid: np.ndarray = aux.copy()
    aux_norm_sq = float(np.sum(aux * aux))
    for lam in grid:
        denom = eigvals + lam
        A_eig = Ut_aux / denom[:, None]
        A_lam = eigvecs @ A_eig
        fitted = T_sup @ A_lam
        resid = fitted - aux
        rss = float(np.sum(resid * resid))
        trace_H = float(np.sum(eigvals / denom))
        gcv_denom = N - trace_H
        if gcv_denom <= 0.0:
            continue
        score = rss / (gcv_denom * gcv_denom)
        if score < best_score:
            best_score = score
            best_lam = float(lam)
            best_A = A_lam
            best_resid = resid
    if not np.isfinite(best_score):
        raise RuntimeError(
            "soft_l2: GCV selection did not find a finite-score weight; "
            "check that T_sup has nonzero variance"
        )
    # Inverse-scaled residual norm (as documented in the public docstring).
    score = 1.0 - float(np.sum(best_resid * best_resid)) / max(aux_norm_sq, 1e-300)
    return best_lam, best_A, np.asarray([score], dtype=np.float64)


def _orthogonal_complement_projection(T_free: np.ndarray, T_sup: np.ndarray) -> np.ndarray:
    """QR-based projection of T_free onto the orthogonal complement of col(T_sup).

    Computes Q from the thin QR of T_sup (numerically tighter than naive
    Gram–Schmidt), then returns ``T_free - Q (Qᵀ T_free)``.
    """
    if T_sup.size == 0 or T_sup.shape[1] == 0:
        return T_free.copy()
    Q, _ = np.linalg.qr(T_sup, mode="reduced")
    return T_free - Q @ (Q.T @ T_free)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PartialSupervisionFit:
    """Result of running :meth:`PartialSupervisionRecipe.fit`.

    Attributes
    ----------
    T_supervised : (N, d_supervised) ndarray
        Latent block tied to ``aux`` via the selected ``sup_method``.
    T_free : (N, d_free) ndarray
        Latent block decorrelated from ``T_supervised`` per
        ``free_constraint``.
    alignment_score : float
        Convention: ``1 - ‖T_sup R - aux‖_F² / ‖aux‖_F²`` for
        ``procrustes`` / ``anchor`` (1.0 = perfect alignment, 0.0 =
        no better than predicting 0). For ``soft_l2`` the same
        scaled-residual form is used with the REML-selected weight's
        ridge map A_λ in place of R.
    sup_method : str
        Echoes the ``sup_method`` argument used.
    free_constraint : str | None
        Echoes the ``free_constraint`` argument used.
    selected_weight : float | None
        REML-selected soft-L2 weight (only set when
        ``sup_method == 'soft_l2'``).
    map_R : (d_supervised, d_supervised) ndarray | None
        Procrustes rotation when ``sup_method == 'procrustes'``.
    map_A : ndarray | None
        Affine slope (anchor) or ridge map (soft_l2). ``None`` for
        procrustes.
    map_b : ndarray | None
        Anchor affine intercept. ``None`` for the other methods.
    aux_score : float | None
        Optional :class:`gamfit.GaugeCompanion` loss when a color
        auxiliary was supplied; ``None`` otherwise.
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

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Project a new T-block ``X_new`` (N', T_dim) through the fitted gauge.

        For ``procrustes`` / ``soft_l2`` the supervised slice is
        right-multiplied by the stored map; for ``anchor`` the affine
        map is applied. The free slice is orthogonalized against the
        new supervised slice when a free constraint is configured.
        """
        if X_new.ndim != 2:
            raise ValueError(f"predict expects a 2D array, got shape {X_new.shape}")
        d_sup = self.T_supervised.shape[1]
        d_free = self.T_free.shape[1]
        if X_new.shape[1] != d_sup + d_free:
            raise ValueError(
                f"predict expects T_dim={d_sup + d_free} columns, got {X_new.shape[1]}"
            )
        T_sup_raw = X_new[:, :d_sup]
        T_free_raw = X_new[:, d_sup:]
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
        if self.free_constraint == "orthogonal_to_sup":
            T_free_new = _orthogonal_complement_projection(T_free_raw, T_sup_new)
        else:
            T_free_new = T_free_raw.copy()
        return np.concatenate([T_sup_new, T_free_new], axis=1)


@dataclass(slots=True)
class PartialSupervisionRecipe:
    """Gauge-fix recipe for partial supervision.

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

    def fit(self, X: np.ndarray, T_init: np.ndarray | None = None) -> PartialSupervisionFit:
        """Run the gauge-fix recipe.

        Parameters
        ----------
        X : (N, p) ndarray
            Predictor block. Used only for shape consistency when
            ``T_init`` is not given; in that case ``T_init`` is
            derived from the leading ``T_dim`` PCA scores of ``X``.
        T_init : (N, T_dim) ndarray, optional
            Initial latent block. Required when ``X.shape[1] <
            T_dim`` because PCA can't recover that many components.
        """
        X = np.asarray(X, dtype=np.float64)
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

        aux_norm_sq = float(np.sum(self.aux * self.aux))
        if aux_norm_sq <= 0.0:
            raise ValueError("aux has zero Frobenius norm; alignment is undefined")

        sup_aligned: np.ndarray
        score: float
        selected_weight: float | None = None
        map_R: np.ndarray | None = None
        map_A: np.ndarray | None = None
        map_b: np.ndarray | None = None

        if self.sup_method == "procrustes":
            R, sup_aligned = _orthogonal_procrustes(T_sup, self.aux)
            resid = sup_aligned - self.aux
            score = 1.0 - float(np.sum(resid * resid)) / aux_norm_sq
            map_R = R
        elif self.sup_method == "anchor":
            A, b, sup_aligned = _anchor_affine(T_sup, self.aux, self.anchor_idx)
            resid = sup_aligned - self.aux
            score = 1.0 - float(np.sum(resid * resid)) / aux_norm_sq
            map_A = A
            map_b = b
        else:  # soft_l2
            lam, A_lam, score_arr = _soft_l2_select_weight(T_sup, self.aux)
            sup_aligned = T_sup @ A_lam
            score = float(score_arr[0])
            selected_weight = lam
            map_A = A_lam

        if self.free_constraint == "orthogonal_to_sup":
            free_aligned = _orthogonal_complement_projection(T_free_raw, sup_aligned)
        else:
            free_aligned = T_free_raw.copy()

        aux_score: float | None = None
        if self.aux_name is not None:
            # Reuse the existing GaugeCompanion scorer for color auxiliaries.
            from .._equivariant import GaugeCompanion  # local import; avoid cycle.
            companion = GaugeCompanion(
                aux=self.aux_name, d_aux=self.d_supervised, aux_values=self.aux,
            )
            aux_score = companion.loss(sup_aligned)

        return PartialSupervisionFit(
            T_supervised=sup_aligned,
            T_free=free_aligned,
            alignment_score=score,
            sup_method=self.sup_method,
            free_constraint=self.free_constraint,
            selected_weight=selected_weight,
            map_R=map_R,
            map_A=map_A,
            map_b=map_b,
            aux_score=aux_score,
        )

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
        Xc = X - X.mean(axis=0, keepdims=True)
        # Thin SVD: leading T_dim left singular vectors scaled by singular values.
        U, s, _Vt = np.linalg.svd(Xc, full_matrices=False)
        k = self.T_dim
        return U[:, :k] * s[:k]


def partial_supervision(
    T_dim: int,
    aux: np.ndarray,
    d_supervised: int,
    d_free: int,
    sup_method: SupMethod = "procrustes",
    free_constraint: FreeConstraint = "orthogonal_to_sup",
    anchor_idx: Sequence[int] = (0,),
    aux_name: AuxColorName | None = None,
) -> PartialSupervisionRecipe:
    """Build a partial-supervision gauge-fix recipe.

    Parameters
    ----------
    T_dim : int
        Total latent dimension (must equal ``d_supervised + d_free``).
    aux : (N, d_supervised) array_like
        Numeric auxiliary signal that the supervised block is tied to.
    d_supervised : int
        Number of latent columns supervised against ``aux``.
    d_free : int
        Number of latent columns left unsupervised. After
        ``free_constraint='orthogonal_to_sup'`` they are projected onto
        the orthogonal complement of the supervised column space.
    sup_method : {'procrustes', 'anchor', 'soft_l2'}, default 'procrustes'
        Solve method for the supervised block. See module docstring and
        :class:`PartialSupervisionFit` for the alignment-score
        conventions.
    free_constraint : {'orthogonal_to_sup', None}, default 'orthogonal_to_sup'
        Decorrelation rule for the free block. ``None`` skips
        projection and lets the free block be penalty-regularized
        downstream.
    anchor_idx : sequence of int, default ``(0,)``
        Row indices used by the ``'anchor'`` method.
    aux_name : {'HSV','RGB','LCh', None}, default None
        Optional color-auxiliary name. When set, the recipe also
        evaluates :class:`gamfit.GaugeCompanion` loss on the aligned
        supervised block and stores it in ``fit.aux_score``.

    Returns
    -------
    PartialSupervisionRecipe
        Call ``recipe.fit(X, T_init=...)`` to run the recipe.

    Examples
    --------
    >>> import numpy as np, gamfit
    >>> rng = np.random.default_rng(0)
    >>> hsv = rng.standard_normal((200, 3))
    >>> recipe = gamfit.recipes.partial_supervision(
    ...     T_dim=6, aux=hsv, d_supervised=3, d_free=3,
    ...     sup_method='procrustes',
    ...     free_constraint='orthogonal_to_sup',
    ... )
    >>> fit = recipe.fit(rng.standard_normal((200, 8)))
    >>> fit.T_supervised.shape, fit.T_free.shape
    ((200, 3), (200, 3))
    """
    return PartialSupervisionRecipe(
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
    "PartialSupervisionRecipe",
    "PartialSupervisionFit",
]
