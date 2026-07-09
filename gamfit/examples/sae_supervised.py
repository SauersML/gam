"""SAE + supervised-head example.

`sae_supervised(X, Y, supervised_mask, ...)` is pure orchestration:

1. Fit a manifold SAE on the **full** ``X`` (supervised + unsupervised
   rows) via the Rust kernel ``gamfit.sae_manifold_fit``.
2. Extract the per-row latent assignments produced by that kernel.
3. Fit a GLM head on the supervised slice of those latents via
   ``gamfit.fit`` against ``Y[supervised_mask]``.
4. Return a uniform :class:`SaeSupervisedFit` result with ``.sae``,
   ``.model``, ``.report()``, and ``.predict(X)`` semantics.

Every numerical step delegates to a Rust kernel: SAE fit, R² scoring,
GLM solve, prediction. This module only sequences those calls and
shapes the inputs / outputs. No new math lives here.

Out-of-sample prediction on genuinely new ``X`` is fully supported: the Rust
OOS predict path returns converged per-token assignments, which ``predict()``
feeds into the GLM head via :meth:`ManifoldSAE.encode`. The underlying
:class:`ManifoldSAE` remains available as ``.sae`` for assignment diagnostics,
per-atom decoder covariances, and posterior shape bands when the Rust fit
produced those fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .._sae_manifold import sae_manifold_fit, ManifoldSAE
from .._api import fit as gamfit_fit
from .._model import Model


__all__ = [
    "SaeSupervisedFit",
    "sae_supervised",
]


@dataclass(slots=True)
class SaeSupervisedFit:
    """Result of :func:`sae_supervised`.

    Attributes
    ----------
    sae : ManifoldSAE
        Manifold SAE fit on the full ``X``. Owns the encoder / decoder,
        per-row ``assignments``, per-atom fits, and any decoder-covariance /
        shape-band fields produced by ``sae_manifold_fit``.
    model : gamfit.Model
        GLM head fit on ``(latents[supervised_mask], Y[supervised_mask])``.
    supervised_mask : (N,) bool ndarray
        Boolean mask of supervised rows (copy of the input mask).
    latent_names : tuple of str
        Column names used for the GLM head (e.g. ``("t_0", ..., "t_{K-1}")``).
    response_name : str
        Column name used for the response (default ``"y"``).
    n_train : int
        Total training rows (``X.shape[0]``).
    n_supervised : int
        Number of supervised rows (``supervised_mask.sum()``).
    """

    sae: ManifoldSAE
    model: Model
    supervised_mask: np.ndarray
    latent_names: tuple[str, ...]
    response_name: str
    n_train: int
    n_supervised: int

    def report(self) -> dict[str, Any]:
        """Combined example-level summary.

        Returns the SAE summary, the GLM head summary, and row counts. All fit
        statistics come from the underlying SAE and GAM/GLM objects.
        """
        return {
            "example": "sae_supervised",
            "n_train": self.n_train,
            "n_supervised": self.n_supervised,
            "latent_dim": len(self.latent_names),
            "sae": self.sae.summary(),
            "head": self.model.summary().to_dict()
            if hasattr(self.model.summary(), "to_dict")
            else dict(self.model.summary()),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the response for ``X``.

        The SAE's frozen-decoder encoder computes an ``(N, K)`` assignment
        matrix for ``X`` via :meth:`ManifoldSAE.encode`; those assignment
        columns are fed straight into the fitted GAM/GLM head. No Python-side
        re-derivation of the SAE encoder is performed.

        Returns
        -------
        ndarray
            Response predictions from the supervised head.

        Raises
        ------
        ValueError
            If ``X`` is not a 2D numeric matrix.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"predict expects a 2D X, got shape {X_arr.shape}")
        latents = np.asarray(self.sae.encode(X_arr), dtype=np.float64)
        table = _assignments_to_table(latents, self.latent_names)
        return np.asarray(self.model.predict(table), dtype=np.float64)


def _validate_supervised_mask(mask: Any, n: int) -> np.ndarray:
    """Validate ``mask`` and return a contiguous bool ndarray of length ``n``."""
    mask_arr = np.ascontiguousarray(np.asarray(mask))
    if mask_arr.dtype != np.bool_:
        if not np.issubdtype(mask_arr.dtype, np.integer):
            raise TypeError(
                f"supervised_mask must be a boolean (or 0/1 integer) array; "
                f"got dtype {mask_arr.dtype!r}"
            )
        unique = np.unique(mask_arr)
        if not np.all((unique == 0) | (unique == 1)):
            raise ValueError(
                "supervised_mask integer values must be 0 or 1; "
                f"got unique values {unique.tolist()!r}"
            )
        mask_arr = mask_arr.astype(np.bool_, copy=False)
    if mask_arr.ndim != 1:
        raise ValueError(
            f"supervised_mask must be 1D; got shape {mask_arr.shape}"
        )
    if mask_arr.shape[0] != n:
        raise ValueError(
            f"supervised_mask has length {mask_arr.shape[0]} but X has {n} rows"
        )
    if not mask_arr.any():
        raise ValueError(
            "supervised_mask selects zero rows; sae_supervised needs at least "
            "one supervised observation to fit the GLM head"
        )
    return mask_arr


def _latent_names(k: int) -> tuple[str, ...]:
    return tuple(f"t_{i}" for i in range(k))


def _assignments_to_table(
    latents: np.ndarray, names: Sequence[str]
) -> dict[str, np.ndarray]:
    if latents.ndim != 2 or latents.shape[1] != len(names):
        raise ValueError(
            f"latents shape {latents.shape} does not match {len(names)} columns"
        )
    return {name: latents[:, i].copy() for i, name in enumerate(names)}


def _supervised_table(
    latents: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    latent_names: Sequence[str],
    response_name: str,
) -> dict[str, np.ndarray]:
    table = _assignments_to_table(latents[mask], latent_names)
    table[response_name] = y[mask].astype(np.float64, copy=False)
    return table


def _default_k(n: int, p: int) -> int:
    """Pick a sane default atom count from problem characteristics.

    Heuristic: K = min(p, max(2, floor(sqrt(n)/2))). Keeps K below the
    "n > K" guard in ``sae_manifold_fit`` while scaling with the data.
    """
    n_root = int(np.floor(np.sqrt(max(n, 4)) / 2))
    k = max(2, n_root)
    # Stay strictly below n (Rust guard requires n > K).
    return int(min(k, max(2, p), n - 1))


def sae_supervised(
    X: np.ndarray,
    Y: np.ndarray,
    supervised_mask: np.ndarray,
    *,
    K: int | None = None,
    d_atom: int = 2,
    atom_topology: str = "circle",
    family: str = "auto",
    head_formula: str | None = None,
    sae_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> SaeSupervisedFit:
    """Fit a manifold SAE on ``X`` then a GLM head on supervised rows.

    Parameters
    ----------
    X : (N, p) array_like
        Feature matrix, used for the SAE fit. All rows participate in
        the unsupervised SAE step.
    Y : (N,) array_like
        Response vector. Only rows where ``supervised_mask`` is True
        are used for the GLM head; the remaining rows contribute only
        to the SAE step.
    supervised_mask : (N,) bool array_like
        Boolean (or 0/1 integer) mask flagging the supervised rows.
        Must select at least one row.
    K : int, optional
        Number of SAE atoms. When ``None``, the example picks a sane
        default from ``(n, p)``.
    d_atom : int, default 2
        Latent atom dimension passed to ``sae_manifold_fit``.
    atom_topology : str, default ``"circle"``
        Atom manifold topology (``"circle"``, ``"sphere"``, ``"torus"``,
        ...). Forwarded to ``sae_manifold_fit``.
    family : str, default ``"auto"``
        GLM family for the head. Forwarded to ``gamfit.fit``.
    head_formula : str, optional
        Override the GLM head formula. Defaults to a linear-in-latents
        Gaussian-style RHS ``"y ~ t_0 + t_1 + ... + t_{K-1}"``.
    sae_kwargs : dict, optional
        Extra keyword arguments forwarded to ``sae_manifold_fit``.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``gamfit.fit`` for the head.

    Returns
    -------
    SaeSupervisedFit
        Uniform example result with ``.sae``, ``.model``, ``.report()``, and
        ``.predict(X)``. Inspect ``fit.sae.assignments`` for the learned
        assignment matrix, ``fit.sae.atoms[i].decoder_covariance``, and
        ``fit.sae.shape_uncertainty(atom=i)`` when present.

    Raises
    ------
    ValueError
        If input shapes are inconsistent, no supervised rows are selected, or
        ``K`` is outside the accepted range for the SAE fit.
    """
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X_arr.shape}")
    n, p = X_arr.shape
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    if Y_arr.ndim != 1 or Y_arr.shape[0] != n:
        raise ValueError(
            f"Y must be a 1D array of length N={n}; got shape {Y_arr.shape}"
        )
    mask_arr = _validate_supervised_mask(supervised_mask, n)

    k_atoms = int(K) if K is not None else _default_k(n, p)
    if k_atoms < 2:
        raise ValueError(
            f"K must be >= 2 to form a non-degenerate latent block; got {k_atoms}"
        )
    if k_atoms >= n:
        raise ValueError(
            f"K={k_atoms} must be strictly less than N={n} (sae_manifold_fit guard)"
        )

    sae_call_kwargs: dict[str, Any] = dict(sae_kwargs or {})
    # Magic-by-default: pick a small enough atom dim for circular bases.
    sae = sae_manifold_fit(
        X=X_arr,
        K=k_atoms,
        d_atom=int(d_atom),
        atom_topology=str(atom_topology),
        **sae_call_kwargs,
    )
    latents = np.asarray(sae.assignments, dtype=np.float64)
    if latents.ndim != 2 or latents.shape[0] != n:
        raise RuntimeError(
            f"sae_manifold_fit returned assignments of shape {latents.shape}; "
            f"expected (N={n}, K={k_atoms})"
        )

    latent_names = _latent_names(latents.shape[1])
    response_name = "y"
    table = _supervised_table(latents, Y_arr, mask_arr, latent_names, response_name)
    formula = head_formula
    if formula is None:
        rhs = " + ".join(latent_names)
        formula = f"{response_name} ~ {rhs}"

    fit_call_kwargs: dict[str, Any] = dict(fit_kwargs or {})
    fit_call_kwargs.setdefault("family", family)
    head_model = gamfit_fit(table, formula, **fit_call_kwargs)

    return SaeSupervisedFit(
        sae=sae,
        model=head_model,
        supervised_mask=mask_arr,
        latent_names=latent_names,
        response_name=response_name,
        n_train=n,
        n_supervised=int(mask_arr.sum()),
    )
