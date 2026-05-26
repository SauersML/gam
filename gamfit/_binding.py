from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Any

from ._cuda import assert_no_cuda_library_conflicts, cuda_diagnostics, prepare_cuda_libraries


class RustExtensionUnavailableError(ImportError):
    """Raised when the compiled ``gamfit._rust`` extension cannot be imported.

    The Rust engine ships as a maturin-built extension module. When it is
    missing (typical in a fresh source checkout that has not been built yet),
    every Rust-backed API in :mod:`gamfit` raises this error eagerly so users
    see a single, actionable message instead of an opaque ``ImportError``.

    The fix is to build or install the package, e.g. ``maturin develop`` from
    the ``gamfit`` source tree, or ``pip install gamfit`` from PyPI.

    Examples
    --------
    >>> try:
    ...     gamfit.fit(df, "y ~ s(x)")
    ... except gamfit.RustExtensionUnavailableError as exc:
    ...     print("build the extension first:", exc)
    """


@lru_cache(maxsize=1)
def rust_module() -> ModuleType:
    prepare_cuda_libraries()
    assert_no_cuda_library_conflicts("importing gamfit._rust")
    try:
        module = importlib.import_module("gamfit._rust")
    except ImportError as exc:  # pragma: no cover - import environment specific
        raise RustExtensionUnavailableError(
            "gamfit._rust is not available. Build or install the package with maturin first."
        ) from exc
    assert_no_cuda_library_conflicts("using gamfit._rust")
    _install_sae_manifold_fit_minimal_alias(module)
    return module


def _install_sae_manifold_fit_minimal_alias(module: ModuleType) -> None:
    """Expose ``sae_manifold_fit_minimal`` as the canonical Python-facing entry
    point for the SAE-manifold Rust kernel.

    The Rust extension currently exports the historical name
    ``sae_manifold_fit_auto``. The IDEAL Python bridge contract — and the one
    that downstream callers (including the tests in
    :mod:`tests.test_sae_assignment_and_schedule_bridge`) target — is the
    name-stable ``sae_manifold_fit_minimal``. To avoid forcing a Rust rename
    while a parallel rebuild is in flight, we install the canonical name as a
    thin attribute on the loaded extension module. The adapter forwards every
    argument unchanged and translates the slightly different positional /
    keyword ordering between the Python contract and the Rust signature.
    """
    if hasattr(module, "sae_manifold_fit_minimal"):
        return
    underlying = getattr(module, "sae_manifold_fit_auto", None)
    if underlying is None:
        return

    def sae_manifold_fit_minimal(
        z,
        k_atoms,
        atom_basis,
        atom_dim,
        assignment_kind,
        alpha,
        tau,
        learnable_alpha,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        random_state,
        top_k,
        *,
        gumbel_schedule=None,
        analytic_penalties=None,
    ):
        if len(atom_basis) != int(k_atoms) or len(atom_dim) != int(k_atoms):
            raise ValueError(
                "sae_manifold_fit_minimal: atom_basis and atom_dim must each "
                f"have length k_atoms={int(k_atoms)}; got len(atom_basis)="
                f"{len(atom_basis)}, len(atom_dim)={len(atom_dim)}"
            )
        # `top_k = 0` (legacy sentinel) / `None` both disable top-k gating;
        # anything in `[1, k_atoms]` is forwarded to the Rust driver, which
        # projects the final assignments onto a per-row top-k support and
        # recomputes `fitted` from the projected distribution.
        top_k_arg = None
        if top_k is not None:
            tk = int(top_k)
            if tk > 0:
                if tk > int(k_atoms):
                    raise ValueError(
                        "sae_manifold_fit_minimal: top_k must satisfy 1 <= top_k <= k_atoms="
                        f"{int(k_atoms)}; got top_k={tk}"
                    )
                top_k_arg = tk
        kwargs: dict[str, Any] = {
            "sparsity_strength": float(sparsity_strength),
            "smoothness": float(smoothness),
            "max_iter": int(max_iter),
            "learning_rate": float(learning_rate),
            "gumbel_schedule": gumbel_schedule,
            "random_state": int(random_state),
        }
        # Forward optional kwargs only when the underlying Rust signature
        # actually accepts them. The Rust extension may lag behind the
        # Python contract during partial rebuilds; the caller-side top-k
        # mask in `_sae_manifold._apply_top_k_mask` enforces the contract
        # if Rust cannot.
        try:
            import inspect as _inspect
            params = _inspect.signature(underlying).parameters
        except (TypeError, ValueError):
            params = {}
        if "analytic_penalties" in params:
            kwargs["analytic_penalties"] = analytic_penalties
        if "top_k" in params:
            kwargs["top_k"] = top_k_arg
        return underlying(
            z,
            atom_basis,
            atom_dim,
            float(alpha),
            float(tau),
            bool(learnable_alpha),
            str(assignment_kind),
            **kwargs,
        )

    try:
        module.sae_manifold_fit_minimal = sae_manifold_fit_minimal  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # Some Python builds disallow setattr on extension modules; in that
        # case the canonical name is simply unavailable for now, and callers
        # that need the new contract must use the legacy name.
        return


def extension_status() -> dict[str, object]:
    try:
        module = rust_module()
    except RustExtensionUnavailableError as exc:
        return {
            "available": False,
            "module": "gamfit._rust",
            "reason": str(exc),
        }
    build_info = module.build_info()
    return {
        "available": True,
        "module": "gamfit._rust",
        "cuda_diagnostics": cuda_diagnostics(),
        **dict(build_info),
    }
