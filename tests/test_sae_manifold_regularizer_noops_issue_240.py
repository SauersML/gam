"""RED tests pinning contract for issue #240.

`gamfit.sae_manifold_fit` advertises 5 regularizer / topology parameters that
are currently silent no-ops. Each test fits twice with `random_state` fixed —
once with the parameter at its default and once with a value that should
visibly alter the fit — then asserts the resulting arrays differ.

If a future fix instead raises `NotImplementedError`, these tests still pin
the contract correctly: silent acceptance with no effect is the bug.
"""
from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _data(seed: int = 0, n: int = 32, d: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _baseline(**overrides):
    """Common kwargs for sae_manifold_fit."""
    kwargs = dict(
        n_atoms=2,
        atom_basis="periodic",
        atom_dim=1,
        assignment="softmax",
        max_iter=5,
        random_state=7,
    )
    kwargs.update(overrides)
    return kwargs


def _differs(a, b, *, atol: float = 1e-8) -> bool:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return True
    return not np.allclose(a_arr, b_arr, atol=atol, rtol=0.0)


def _fit_must_react(param_name: str, on_value, off_value=None, *, n: int = 32):
    """Fit with parameter on vs off; assert fitted or assignments differ.

    A NotImplementedError on the `on_value` call is an acceptable contract
    (it's the alternative principled fix); silent equality is the bug.
    """
    X = _data(seed=1, n=n)
    base_kwargs = _baseline()
    off_kwargs = dict(base_kwargs)
    if off_value is not None:
        off_kwargs[param_name] = off_value
    fit_off = gamfit.sae_manifold_fit(Z=X, **off_kwargs)

    on_kwargs = dict(base_kwargs)
    on_kwargs[param_name] = on_value
    try:
        fit_on = gamfit.sae_manifold_fit(Z=X, **on_kwargs)
    except NotImplementedError:
        # Principled rejection: parameter is honestly declared unsupported.
        return

    differs_fitted = _differs(fit_on.fitted, fit_off.fitted)
    differs_assign = _differs(fit_on.assignments, fit_off.assignments)
    assert differs_fitted or differs_assign, (
        f"`{param_name}` is a silent no-op: fitting with on={on_value!r} vs "
        f"off={off_value!r} produces identical fitted and assignments arrays. "
        "Either wire the parameter into the Rust objective or raise "
        "NotImplementedError at the wrapper boundary."
    )


def test_isometry_weight_is_not_a_silent_noop():
    _fit_must_react("isometry_weight", on_value=100.0, off_value=0.0)


def test_ard_per_atom_is_not_a_silent_noop():
    _fit_must_react("ard_per_atom", on_value=False, off_value=True)


def test_block_orthogonality_weight_is_not_a_silent_noop():
    _fit_must_react("block_orthogonality_weight", on_value=100.0, off_value=0.0)


def test_mechanism_sparsity_groups_is_not_a_silent_noop():
    _fit_must_react(
        "mechanism_sparsity_groups",
        on_value=[[0], [1]],
        off_value=None,
    )


def test_topology_selector_is_not_a_silent_noop():
    """`topology_selector` appears only in the function signature — it must
    either alter the fit or be rejected. A unique sentinel object is used so
    the test fails even if the implementation grows e.g. a string parser.
    """
    sentinel = object()
    _fit_must_react("topology_selector", on_value=sentinel, off_value=None)


def test_primitive_names_metadata_is_not_a_substitute_for_effect():
    """The current code only updates `primitive_names` when these toggles
    change. Confirm metadata flips but fits stay identical — captures the
    exact pathological state from issue #240."""
    X = _data(seed=2)
    base_kwargs = _baseline()
    fit_off = gamfit.sae_manifold_fit(
        Z=X, **base_kwargs, isometry_weight=0.0, block_orthogonality_weight=0.0
    )
    fit_on = gamfit.sae_manifold_fit(
        Z=X, **base_kwargs, isometry_weight=10.0, block_orthogonality_weight=10.0
    )
    metadata_changed = set(fit_on.primitive_names) != set(fit_off.primitive_names)
    fit_arrays_unchanged = np.allclose(
        fit_on.fitted, fit_off.fitted, atol=1e-8, rtol=0.0
    ) and np.allclose(
        fit_on.assignments, fit_off.assignments, atol=1e-8, rtol=0.0
    )
    # The bug: metadata flips but math doesn't. Pinning that this combination
    # must not persist — fix must either change the math or remove the
    # metadata advertising.
    assert not (metadata_changed and fit_arrays_unchanged), (
        "Metadata `primitive_names` flipped but `fitted` and `assignments` "
        "are bit-identical: regularizers are advertised but not applied."
    )
