"""Sparse Identification of Nonlinear Dynamics (Brunton 2016) as a gamfit atom.

Implements :class:`SINDyAtoms`, a thin Python orchestration layer over the
Rust :func:`gamfit._rust.sindy_stlsq_solve_array` solver. The numerics
(Sequential Thresholded Least Squares, SCAD / MCP local-quadratic
re-weighting, BIC-based ``lam='auto'`` selection) live entirely in Rust per
the project's no-math-in-Python rule. This module only:

  * builds the library design matrix Θ from a trajectory and a list of
    library term descriptors;
  * computes ``dz/dt`` via centered finite differences when the user does
    not supply derivatives;
  * marshals (Θ, Ẋ) into Rust and unpacks the coefficient matrix;
  * pretty-prints the learned ODE system.

Example
-------
>>> import gamfit
>>> sindy = gamfit.SINDyAtoms(
...     library=['const', 'id', 'product'],
...     sparsity={'kind': 'scad', 'a': 3.7, 'lam': 'auto'},
...     threshold={'kind': 'stlsq', 'tol': 0.05, 'max_rounds': 10},
...     state_dim=3,
... )
>>> sindy.fit(z_trajectory, dz_dt=z_dot)               # doctest: +SKIP
>>> sindy.equations_human_readable(['x', 'y', 'z'])    # doctest: +SKIP
['dx/dt = -10.0x + 10.0y', ...]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ._binding import rust_module

__all__ = ["SINDyAtoms"]


# ---------------------------------------------------------------------------
# Library-term descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LibraryTerm:
    """One column of the library design matrix Θ."""

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    """``fn(z) -> (n,)`` where ``z`` has shape ``(n, state_dim)``."""


def _expand_library(
    library: Sequence,
    state_dim: int,
    state_names: Sequence[str],
) -> list[_LibraryTerm]:
    """Expand the user-supplied library spec into one term per output column.

    Strings select canonical SINDy library families; callables become a single
    user-named column. The expansion is deterministic and matches the Brunton
    2016 conventions:

    * ``'const'``           → ``1``
    * ``'id'`` / ``'linear'`` → ``z_i`` for each state dim ``i``
    * ``'square'``          → ``z_i**2`` for each ``i``
    * ``'cube'``            → ``z_i**3`` for each ``i``
    * ``'product'``         → ``z_i z_j`` for all unique pairs ``i < j``
    * ``'sin'``             → ``sin(z_i)`` for each ``i``
    * ``'cos'``             → ``cos(z_i)`` for each ``i``
    """
    terms: list[_LibraryTerm] = []
    for entry in library:
        if callable(entry):
            name = getattr(entry, "__name__", "user_term")
            terms.append(
                _LibraryTerm(
                    name=name,
                    fn=lambda z, f=entry: np.asarray(f(z), dtype=np.float64).reshape(-1),
                )
            )
            continue
        if not isinstance(entry, str):
            raise TypeError(
                f"SINDyAtoms: library entries must be strings or callables; got {type(entry).__name__}"
            )
        key = entry.strip().lower()
        if key == "const":
            terms.append(
                _LibraryTerm(name="1", fn=lambda z: np.ones(z.shape[0], dtype=np.float64))
            )
        elif key in ("id", "linear"):
            for i in range(state_dim):
                terms.append(
                    _LibraryTerm(
                        name=state_names[i],
                        fn=lambda z, i=i: z[:, i].astype(np.float64, copy=False),
                    )
                )
        elif key == "square":
            for i in range(state_dim):
                terms.append(
                    _LibraryTerm(
                        name=f"{state_names[i]}^2",
                        fn=lambda z, i=i: (z[:, i] * z[:, i]).astype(np.float64, copy=False),
                    )
                )
        elif key == "cube":
            for i in range(state_dim):
                terms.append(
                    _LibraryTerm(
                        name=f"{state_names[i]}^3",
                        fn=lambda z, i=i: (z[:, i] ** 3).astype(np.float64, copy=False),
                    )
                )
        elif key == "product":
            for i in range(state_dim):
                for j in range(i, state_dim):
                    if i == j:
                        continue  # squares belong to the 'square' family
                    name_i, name_j = state_names[i], state_names[j]
                    terms.append(
                        _LibraryTerm(
                            name=f"{name_i}{name_j}",
                            fn=lambda z, i=i, j=j: (z[:, i] * z[:, j]).astype(
                                np.float64, copy=False
                            ),
                        )
                    )
        elif key == "sin":
            for i in range(state_dim):
                terms.append(
                    _LibraryTerm(
                        name=f"sin({state_names[i]})",
                        fn=lambda z, i=i: np.sin(z[:, i]).astype(np.float64, copy=False),
                    )
                )
        elif key == "cos":
            for i in range(state_dim):
                terms.append(
                    _LibraryTerm(
                        name=f"cos({state_names[i]})",
                        fn=lambda z, i=i: np.cos(z[:, i]).astype(np.float64, copy=False),
                    )
                )
        else:
            raise ValueError(
                f"SINDyAtoms: unknown library term {entry!r}. Built-ins are "
                "'const', 'id', 'square', 'cube', 'product', 'sin', 'cos'; "
                "pass a callable for custom features."
            )
    if not terms:
        raise ValueError("SINDyAtoms: library expanded to zero terms")
    return terms


def _format_three_sig_figs(value: float) -> str:
    """Mirrors :func:`gam::solver::reml_compare::format_three_significant`.

    Three significant figures, decimal for ``|v| < 1000``, scientific otherwise.
    Trailing zeros are preserved (so ``10.0`` stays ``"10.0"``, not ``"10"``).
    """
    if value == 0.0:
        return "0"
    if not np.isfinite(value):
        return str(value)
    abs_v = abs(value)
    exponent = int(np.floor(np.log10(abs_v)))
    if exponent >= 3:
        return f"{value:.2e}"
    decimals = max(2 - exponent, 0)
    scale = 10 ** decimals
    # Decimal-domain half-away-from-zero rounding so 1.005 -> "1.01"
    # (matches src/solver/reml_compare.rs::format_three_significant).
    rounded_mag = np.floor(abs_v * scale + 0.5) / scale
    rounded = rounded_mag if value > 0 else -rounded_mag
    return f"{rounded:.{decimals}f}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SINDyAtoms:
    """SINDy (Brunton 2016) as a gamfit atom family.

    The Rust kernel runs Sequential Thresholded Least Squares (Brunton 2016
    Algorithm 1) on a library design matrix Θ built from the user's library
    spec. The optional SCAD / MCP penalty uses local quadratic approximation
    (Fan-Li 2001) inside each STLSQ round.

    Parameters
    ----------
    library : sequence of strings or callables
        Library term descriptors. Strings select canonical SINDy families
        (``'const'``, ``'id'``, ``'square'``, ``'cube'``, ``'product'``,
        ``'sin'``, ``'cos'``). Callables produce one custom column each;
        each callable receives the full trajectory ``z`` of shape
        ``(n_samples, state_dim)`` and must return a ``(n_samples,)`` array.
    sparsity : dict or string
        Concave penalty family. Either ``'stlsq'`` (plain ridge backbone) or
        a dict like ``{'kind': 'scad' | 'mcp', 'a': 3.7, 'lam': 'auto' | float}``.
        ``lam='auto'`` triggers BIC-based grid selection in Rust — see Notes.
    threshold : dict
        STLSQ thresholding parameters: ``{'kind': 'stlsq', 'tol': float | 'auto',
        'max_rounds': int}``. ``tol='auto'`` resolves to ``0.05 * std(Ẋ)``.
    state_dim : int
        Number of state variables ``d``.

    Notes
    -----
    ``lam='auto'`` uses a BIC sweep over a logarithmic λ grid in Rust. The
    REML / LAML ``'auto'`` route used by e.g. :class:`AdaptiveTopK` does not
    apply here: SINDy is a one-shot sparse-regression problem with no
    underlying ``S_θ`` penalty design block — there is nothing for the outer
    REML loop to differentiate. BIC is the SINDy literature's default
    complexity criterion and matches the original Brunton 2016 hyper-sweep.

    Attributes
    ----------
    theta : ndarray of shape ``(state_dim, n_library_terms)``
        Sparse coefficient matrix populated by :meth:`fit`. Rows index state
        variables; columns index library terms. **Mathematical convention:
        each row is the SINDy coefficient vector for one state-variable's
        time derivative**, matching the public-API spec in this module's
        docstring (the Rust kernel internally uses the transposed shape and
        the row/column orientation is reversed at marshal time).
    """

    def __init__(
        self,
        library: Sequence,
        sparsity: dict | str = "stlsq",
        threshold: dict | None = None,
        *,
        state_dim: int,
    ) -> None:
        if not isinstance(state_dim, int) or state_dim < 1:
            raise ValueError(
                f"SINDyAtoms: state_dim must be a positive int, got {state_dim!r}"
            )
        self._library_spec = list(library)
        self._state_dim = int(state_dim)

        if isinstance(sparsity, str):
            sparsity = {"kind": sparsity}
        if not isinstance(sparsity, dict):
            raise TypeError(
                f"SINDyAtoms: sparsity must be a dict or string, got {type(sparsity).__name__}"
            )
        kind = str(sparsity.get("kind", "stlsq")).lower()
        if kind in ("stlsq", "ridge", "l2"):
            self._penalty_kind = "ridge"
            self._concave_a = 0.0
        elif kind == "scad":
            self._penalty_kind = "scad"
            self._concave_a = float(sparsity.get("a", 3.7))
            if not (self._concave_a > 2.0):
                raise ValueError(
                    f"SINDyAtoms: SCAD requires a > 2, got {self._concave_a}"
                )
        elif kind in ("mcp", "scad_mcp"):
            self._penalty_kind = "mcp"
            self._concave_a = float(sparsity.get("a", 3.0))
            if not (self._concave_a > 1.0):
                raise ValueError(
                    f"SINDyAtoms: MCP requires a > 1, got {self._concave_a}"
                )
        else:
            raise ValueError(
                f"SINDyAtoms: sparsity.kind must be 'stlsq', 'scad', or 'mcp'; got {kind!r}"
            )
        lam_raw = sparsity.get("lam", 1.0e-3)
        if isinstance(lam_raw, str) and lam_raw.lower() == "auto":
            self._auto_lam = True
            self._lam = 0.0
        else:
            self._auto_lam = False
            self._lam = float(lam_raw)
            if not (self._lam >= 0.0):
                raise ValueError(
                    f"SINDyAtoms: sparsity.lam must be non-negative, got {self._lam}"
                )

        if threshold is None:
            threshold = {"kind": "stlsq", "tol": "auto", "max_rounds": 10}
        thr_kind = str(threshold.get("kind", "stlsq")).lower()
        if thr_kind != "stlsq":
            raise ValueError(
                f"SINDyAtoms: only threshold.kind='stlsq' is supported; got {thr_kind!r}"
            )
        self._tol_raw = threshold.get("tol", "auto")
        self._max_rounds = int(threshold.get("max_rounds", 10))
        if self._max_rounds < 1:
            raise ValueError(
                f"SINDyAtoms: threshold.max_rounds must be >= 1, got {self._max_rounds}"
            )

        # Populated by fit():
        self.theta: np.ndarray | None = None
        self._term_names: list[str] | None = None
        self._state_names: list[str] | None = None
        self._rounds_used: int | None = None
        self._converged: bool | None = None
        self._lam_used: float | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        z_trajectory: np.ndarray,
        dz_dt: np.ndarray | None = None,
        *,
        dt: float = 1.0,
        state_names: Sequence[str] | None = None,
    ) -> "SINDyAtoms":
        """Fit the sparse coefficient matrix.

        Parameters
        ----------
        z_trajectory : ndarray of shape ``(n_samples, state_dim)``
        dz_dt : ndarray of shape ``(n_samples, state_dim)`` or None
            Time derivatives. If ``None``, computed by centered finite
            differences with spacing ``dt`` (endpoints use one-sided
            differences). The standard SINDy convention.
        dt : float
            Sample spacing for the finite-difference fallback. Ignored when
            ``dz_dt`` is supplied.
        state_names : sequence of str, optional
            Names for the state variables. Defaults to ``['x', 'y', 'z',
            'w', 's1', 's2', ...]``.

        Returns
        -------
        self
        """
        z = np.asarray(z_trajectory, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self._state_dim:
            raise ValueError(
                f"SINDyAtoms.fit: z_trajectory must be (n, state_dim={self._state_dim}); "
                f"got shape {z.shape}"
            )
        n = z.shape[0]
        if n < 4:
            raise ValueError(
                f"SINDyAtoms.fit: need at least 4 trajectory rows, got {n}"
            )

        if dz_dt is None:
            if not (np.isfinite(dt) and dt > 0):
                raise ValueError(
                    f"SINDyAtoms.fit: dt must be a positive finite float when dz_dt is None, got {dt}"
                )
            # Centered differences, one-sided at endpoints. Keep all n rows.
            dz = np.empty_like(z)
            dz[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2.0 * dt)
            dz[0, :] = (z[1, :] - z[0, :]) / dt
            dz[-1, :] = (z[-1, :] - z[-2, :]) / dt
        else:
            dz = np.asarray(dz_dt, dtype=np.float64)
            if dz.shape != z.shape:
                raise ValueError(
                    f"SINDyAtoms.fit: dz_dt shape {dz.shape} must match z_trajectory shape {z.shape}"
                )

        names = self._resolve_state_names(state_names)
        terms = _expand_library(self._library_spec, self._state_dim, names)
        theta_design = np.column_stack([term.fn(z) for term in terms]).astype(
            np.float64, copy=False
        )
        if theta_design.shape[0] != n:
            raise ValueError(
                "SINDyAtoms.fit: library term produced wrong row count "
                f"({theta_design.shape[0]} vs trajectory length {n})"
            )

        tol = self._resolve_tol(dz)
        rust = rust_module()
        coefs_kxp, rounds_used, converged, lam_used = rust.sindy_stlsq_solve_array(
            theta_design,
            dz,
            float(tol),
            int(self._max_rounds),
            float(self._lam),
            self._penalty_kind,
            float(self._concave_a) if self._concave_a > 0.0 else 3.7,
            bool(self._auto_lam),
        )
        # Rust returns Ξ ∈ (p, d). The public API contract puts state-vars on
        # the rows and library terms on the columns, so transpose at the
        # marshaling boundary.
        self.theta = np.asarray(coefs_kxp, dtype=np.float64).T.copy()
        self._term_names = [term.name for term in terms]
        self._state_names = list(names)
        self._rounds_used = int(rounds_used)
        self._converged = bool(converged)
        self._lam_used = float(lam_used)
        return self

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def equations_human_readable(
        self, state_names: Sequence[str] | None = None
    ) -> list[str]:
        """Return one human-readable ODE per state variable.

        Coefficients are formatted to three significant figures (matching
        the Rust-side ``format_three_significant`` contract), and library
        terms with ``|coef| < tol`` are dropped. The first term in each
        equation has its sign inlined (``-10.0x``), subsequent terms use
        a separator (`` - 3.00y``).
        """
        if self.theta is None or self._term_names is None or self._state_names is None:
            raise RuntimeError("SINDyAtoms.equations_human_readable: call fit() first")
        names = (
            list(state_names) if state_names is not None else list(self._state_names)
        )
        if len(names) != self._state_dim:
            raise ValueError(
                f"SINDyAtoms.equations_human_readable: expected {self._state_dim} state names, "
                f"got {len(names)}"
            )
        lines: list[str] = []
        for i in range(self._state_dim):
            row = self.theta[i]
            parts: list[str] = []
            for coef, term_name in zip(row, self._term_names):
                if coef == 0.0:
                    continue
                sign = "-" if coef < 0 else "+"
                magnitude = _format_three_sig_figs(abs(coef))
                if term_name == "1":
                    rendered = magnitude
                else:
                    rendered = f"{magnitude}{term_name}"
                if not parts:
                    parts.append(f"-{rendered}" if sign == "-" else rendered)
                else:
                    parts.append(f" {sign} {rendered}")
            rhs = "".join(parts) if parts else "0"
            lines.append(f"d{names[i]}/dt = {rhs}")
        return lines

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def converged(self) -> bool:
        if self._converged is None:
            raise RuntimeError("SINDyAtoms.converged: call fit() first")
        return self._converged

    @property
    def rounds_used(self) -> int:
        if self._rounds_used is None:
            raise RuntimeError("SINDyAtoms.rounds_used: call fit() first")
        return self._rounds_used

    @property
    def lam_used(self) -> float:
        if self._lam_used is None:
            raise RuntimeError("SINDyAtoms.lam_used: call fit() first")
        return self._lam_used

    @property
    def term_names(self) -> list[str]:
        if self._term_names is None:
            raise RuntimeError("SINDyAtoms.term_names: call fit() first")
        return list(self._term_names)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_state_names(
        self, state_names: Sequence[str] | None
    ) -> list[str]:
        if state_names is not None:
            names = list(state_names)
            if len(names) != self._state_dim:
                raise ValueError(
                    f"SINDyAtoms: expected {self._state_dim} state names, got {len(names)}"
                )
            return names
        default = ["x", "y", "z", "w"]
        if self._state_dim <= len(default):
            return default[: self._state_dim]
        return default + [f"s{i + 1}" for i in range(self._state_dim - len(default))]

    def _resolve_tol(self, dz: np.ndarray) -> float:
        if isinstance(self._tol_raw, str):
            if self._tol_raw.lower() != "auto":
                raise ValueError(
                    f"SINDyAtoms: threshold.tol must be 'auto' or a float; got {self._tol_raw!r}"
                )
            std = float(np.std(dz))
            return max(0.05 * std, 1.0e-6)
        tol = float(self._tol_raw)
        if not (tol >= 0.0):
            raise ValueError(f"SINDyAtoms: threshold.tol must be >= 0, got {tol}")
        return tol
