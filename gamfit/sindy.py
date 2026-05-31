"""Sparse Identification of Nonlinear Dynamics (Brunton 2016) as a gamfit atom.

Implements :class:`SINDyAtoms`, a thin Python orchestration layer over the
Rust :func:`gamfit._rust.sindy_stlsq_solve_array` solver. The numerics
(Sequential Thresholded Least Squares, SCAD / MCP local-quadratic
re-weighting, BIC-based ``lam='auto'`` selection) live entirely in Rust per
the project's no-math-in-Python rule. The candidate-library Θ builder
(``gam::solver::sindy::sindy_library``) and the finite-difference derivative
(``gam::solver::sindy::sindy_finite_difference``) also live in Rust. This
module only:

  * parses the user library spec and evaluates any **callable** custom terms
    (genuinely Python-only glue), marshaling the ordered spec into Rust which
    owns the built-in monomial/trig column math and naming;
  * requests ``dz/dt`` via the Rust finite-difference kernel when the user
    does not supply derivatives;
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

from typing import Sequence

import numpy as np

from ._binding import rust_module

__all__ = ["SINDyAtoms"]


# ---------------------------------------------------------------------------
# Library-term spec marshaling
# ---------------------------------------------------------------------------

# Built-in family tokens recognised by the Rust library builder
# (`gam::solver::sindy::sindy_library`). Aliases collapse onto these.
_BUILTIN_LIBRARY_TOKENS = {
    "const": "const",
    "id": "id",
    "linear": "id",
    "square": "square",
    "cube": "cube",
    "product": "product",
    "sin": "sin",
    "cos": "cos",
}


def _build_library_spec(
    library: Sequence,
    z: np.ndarray,
) -> list[tuple[str, np.ndarray | None, str | None]]:
    """Marshal the user library spec into the ordered Rust-FFI spec list.

    Returns one ``(token, custom_column, custom_name)`` tuple per library entry,
    in spec order. Built-in string families (``'const'``, ``'id'``/``'linear'``,
    ``'square'``, ``'cube'``, ``'product'``, ``'sin'``, ``'cos'``) carry just a
    token; the actual column math (powers, pairwise products, trig) and column
    naming live in Rust. **Callable** terms are genuinely Python-only glue: each
    is evaluated here and its ``(n,)`` column passed through as a ``'custom'``
    entry so Rust can splice it into Θ at the right position.
    """
    spec: list[tuple[str, np.ndarray | None, str | None]] = []
    for entry in library:
        if callable(entry):
            name = getattr(entry, "__name__", "user_term")
            column = np.asarray(entry(z), dtype=np.float64).reshape(-1)
            spec.append(("custom", column, str(name)))
            continue
        if not isinstance(entry, str):
            raise TypeError(
                f"SINDyAtoms: library entries must be strings or callables; got {type(entry).__name__}"
            )
        token = _BUILTIN_LIBRARY_TOKENS.get(entry.strip().lower())
        if token is None:
            raise ValueError(
                f"SINDyAtoms: unknown library term {entry!r}. Built-ins are "
                "'const', 'id', 'square', 'cube', 'product', 'sin', 'cos'; "
                "pass a callable for custom features."
            )
        spec.append((token, None, None))
    if not spec:
        raise ValueError("SINDyAtoms: library expanded to zero terms")
    return spec


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

        rust = rust_module()
        if dz_dt is None:
            if not (np.isfinite(dt) and dt > 0):
                raise ValueError(
                    f"SINDyAtoms.fit: dt must be a positive finite float when dz_dt is None, got {dt}"
                )
            # Centered differences, one-sided at endpoints (Rust core). Keep all
            # n rows. The standard SINDy differentiation step.
            dz = np.asarray(
                rust.sindy_finite_difference_array(z, float(dt)), dtype=np.float64
            )
        else:
            dz = np.asarray(dz_dt, dtype=np.float64)
            if dz.shape != z.shape:
                raise ValueError(
                    f"SINDyAtoms.fit: dz_dt shape {dz.shape} must match z_trajectory shape {z.shape}"
                )

        names = self._resolve_state_names(state_names)
        # Built-in monomial/trig library columns + naming are owned by Rust;
        # only user callable terms are evaluated in Python and spliced in.
        library_spec = _build_library_spec(self._library_spec, z)
        theta_design, term_names = rust.sindy_library_array(z, list(names), library_spec)
        theta_design = np.asarray(theta_design, dtype=np.float64)
        if theta_design.shape[0] != n:
            raise ValueError(
                "SINDyAtoms.fit: library term produced wrong row count "
                f"({theta_design.shape[0]} vs trajectory length {n})"
            )

        tol = self._resolve_tol(dz)
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
        self._term_names = [str(name) for name in term_names]
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
