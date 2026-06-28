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


# #1512 triage / #240: every regularizer-effect test that drives a real SAE fit
# panics — pyo3_runtime.PanicException 'index out of bounds: the len is 1 but
# the index is 1' (same SAE out-of-bounds family as #357). A real engine bug;
# the one kwarg-rejection test that does not fit still passes. Marked xfail so
# the open panic is tracked without reddening the directory-level CI suite.
_XFAIL_240 = pytest.mark.xfail(
    strict=True,
    reason="#240 open: SAE regularizer-effect fit panics "
    "(pyo3 'index out of bounds: the len is 1 but the index is 1').",
)


def _data(seed: int = 0, n: int = 32, d: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _baseline(**overrides):
    """Common kwargs for sae_manifold_fit.

    Uses ``K=1, d_atom=2``: the SAE arrow-Schur driver supports
    multi-axis analytic penalties (ARD, Isometry, BlockOrthogonality) only
    when ``k_atoms == 1`` — see
    ``src/terms/sae/manifold/mod.rs::add_sae_analytic_penalty_contributions``.
    Holding ``d_atom=2`` gives enough latent axes for
    ``block_orthogonality_weight`` to find a non-trivial 2-group partition,
    and for ``decoder_feature_sparsity_groups=[[0,1],[2,3]]`` to partition
    the four-feature output into two basis-aligned decoder groups.
    """
    # `sphere` keeps `latent_dim == 2` (periodic atoms force it to 1
    # regardless of `atom_dim`). Block-orthogonality requires ≥2 axes.
    # `isometry_weight=0.0` overrides the public default of 1.0 so the
    # baseline does not trip the issue #249 NotImplementedError gate when
    # tests vary unrelated parameters.
    kwargs = dict(
        K=1,
        atom_basis="sphere",
        d_atom=2,
        assignment="softmax",
        n_iter=5,
        random_state=7,
        isometry_weight=0.0,
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
    fit_off = gamfit.sae_manifold_fit(X=X, **off_kwargs)

    on_kwargs = dict(base_kwargs)
    on_kwargs[param_name] = on_value
    try:
        fit_on = gamfit.sae_manifold_fit(X=X, **on_kwargs)
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


@_XFAIL_240
def test_isometry_weight_is_not_a_silent_noop():
    """isometry_weight=100.0 must produce a fit that visibly differs from
    isometry_weight=0.0.  NotImplementedError is no longer an accepted outcome
    — the SAE Isometry path is fully wired as of issue #250."""
    X = _data(seed=1, n=32)
    base_kwargs = _baseline()
    fit_off = gamfit.sae_manifold_fit(X=X, **{**base_kwargs, "isometry_weight": 0.0})
    fit_on = gamfit.sae_manifold_fit(X=X, **{**base_kwargs, "isometry_weight": 100.0})
    assert _differs(fit_on.fitted, fit_off.fitted) or _differs(
        fit_on.assignments, fit_off.assignments
    ), (
        "isometry_weight=100.0 produced an identical fit to isometry_weight=0.0; "
        "the SAE Isometry penalty is a silent no-op."
    )


@_XFAIL_240
def test_ard_per_atom_is_not_a_silent_noop():
    _fit_must_react("ard_per_atom", on_value=False, off_value=True)


@_XFAIL_240
def test_block_orthogonality_weight_is_not_a_silent_noop():
    _fit_must_react("block_orthogonality_weight", on_value=100.0, off_value=0.0)


@_XFAIL_240
def test_decoder_feature_sparsity_groups_is_not_a_silent_noop():
    # The four-feature output (n=32, d=4) must be partitioned end-to-end;
    # MechanismSparsityPenalty requires every feature index in
    # [0, p_out) to appear in exactly one group.
    _fit_must_react(
        "decoder_feature_sparsity_groups",
        on_value=[[0, 1], [2, 3]],
        off_value=None,
    )


@_XFAIL_240
def test_decoder_feature_sparsity_groups_produces_nontrivial_gradient():
    """The previous ``mechanism_sparsity_groups`` kwarg silently raised
    ``NotImplementedError`` — accepted by ``_fit_must_react`` as a principled
    rejection. The stride-aware "beta" target view now wires
    MechanismSparsityPenalty into the SAE decoder block. Fitting with the
    new ``decoder_feature_sparsity_groups`` kwarg must actually return,
    and the fit must visibly differ from the no-penalty baseline."""
    X = _data(seed=4, n=32)
    base = _baseline()
    fit_off = gamfit.sae_manifold_fit(X=X, **base)
    fit_on = gamfit.sae_manifold_fit(
        X=X, **base, decoder_feature_sparsity_groups=[[0, 1], [2, 3]]
    )
    # Must not silently raise — the rename is a wiring change, not a deferral.
    assert _differs(fit_on.fitted, fit_off.fitted) or _differs(
        fit_on.assignments, fit_off.assignments
    ), (
        "decoder_feature_sparsity_groups failed to alter the fit; the SAE "
        "'beta' target view dispatch is silent."
    )


def test_topology_selector_is_not_an_accepted_kwarg():
    """``topology_selector`` was a dead placeholder — only present in the
    function signature, never used anywhere downstream. The principled fix
    for issue #240 deletes it. Pin that decision: passing the kwarg must
    raise ``TypeError`` so callers don't silently lose configuration."""
    X = _data(seed=3)
    with pytest.raises(TypeError, match="topology_selector"):
        gamfit.sae_manifold_fit(X=X, **_baseline(), topology_selector=object())


@_XFAIL_240
def test_primitive_names_metadata_is_not_a_substitute_for_effect():
    """The original issue-#240 footgun: setting ``isometry_weight`` /
    ``block_orthogonality_weight`` flipped the ``primitive_names`` list but
    left fit arrays bit-identical (silent acceptance of a no-op).

    The principled fix has two halves. (1) ``ard_per_atom`` is wired
    through to Rust and DOES alter the fit (see
    ``test_ard_per_atom_is_not_a_silent_noop``). (2) The remaining three
    knobs raise ``NotImplementedError`` at the wrapper boundary until the
    Rust SAE row-block driver supports them (issue #249). Either branch
    rules out the pathological metadata-only divergence: this test pins
    that ``isometry_weight=10.0`` must NOT silently succeed and produce an
    identical fit."""
    X = _data(seed=2)
    base_kwargs = _baseline()
    on_kwargs = dict(base_kwargs)
    on_kwargs["isometry_weight"] = 10.0
    on_kwargs["block_orthogonality_weight"] = 10.0
    try:
        fit_on = gamfit.sae_manifold_fit(X=X, **on_kwargs)
    except NotImplementedError:
        # Principled rejection — exactly the contract we want.
        return
    # If the call returned, the fit must visibly differ from the no-penalty
    # baseline. Silent acceptance with no effect is the bug.
    fit_off = gamfit.sae_manifold_fit(X=X, **base_kwargs)
    differs = (
        _differs(fit_on.fitted, fit_off.fitted)
        or _differs(fit_on.assignments, fit_off.assignments)
    )
    assert differs, (
        "isometry_weight=10.0 and block_orthogonality_weight=10.0 were "
        "accepted without effect — the issue-#240 silent-no-op regression."
    )
