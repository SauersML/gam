"""Status flag for per-smooth λ in the torch additive REML path.

The closed-form Gaussian REML kernel used by
:func:`gamfit.torch._reml.gaussian_reml_fit_additive` lives in Rust at::

    crates/gam-core/src/solver/gaussian_reml.rs

That kernel is structurally **single-λ**: it exploits the joint generalised
eigendecomposition of the pair ``(S, X'WX)`` to reduce the REML score and
its gradient to a rational function of a single scalar ``λ`` (so each probe
costs O(1) after one eigendecomposition). Extending it to per-block ``λ_k``
requires a multi-dimensional outer optimiser over ``log λ_k``, an analytic
VJP through the F×F Hessian of the REML criterion, and per-block routing
through the eigendecomposition step — i.e. a substantial multi-day Rust
refactor. Until that lands, the torch additive path only supports a single
shared ``λ`` across all smooths.

For per-smooth ``λ`` today, use the formula API::

    model = gamfit.fit(df, 'y ~ s(x1) + s(x2) + ...')

which drives the PIRLS workflow and does the full multi-block REML/LAML
outer optimisation. It returns a serialised :class:`gamfit.Model` rather
than differentiable tensors.

This module exists so library code and tests can branch on the limitation
without duplicating the explanation. When the multi-block closed-form
kernel ships in Rust, flip :data:`MULTI_LAMBDA_SUPPORTED` to ``True`` and
delete the placeholder test in ``tests/test_python_api.py``.
"""

from __future__ import annotations

#: ``True`` once the torch additive REML path supports independent λ_k per
#: smooth block. Currently ``False``: the closed-form Rust kernel is
#: single-λ. See the module docstring for the recommended workaround.
MULTI_LAMBDA_SUPPORTED: bool = False

__all__ = ["MULTI_LAMBDA_SUPPORTED"]
