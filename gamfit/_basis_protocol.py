"""Uniform callable-basis protocol for gamfit descriptors.

Every basis descriptor in the public surface — :class:`gamfit.Duchon`,
:class:`gamfit.BSpline`, :class:`gamfit.Matern`, :class:`gamfit.Pca`,
:class:`gamfit.TensorBSpline`, :class:`gamfit.PeriodicSplineCurve` — gains
three methods with a single uniform contract:

* ``evaluate(*coords) -> Tensor`` of shape ``(B, M)``.
* ``jacobian(*coords) -> Tensor`` of shape ``(B, M, d)``.
* ``hessian(*coords)  -> Tensor`` of shape ``(B, M, d, d)``.

``d`` is the *intrinsic* dimensionality of the descriptor's domain.
``B`` is the common length of the input coordinate tensors (one positional
argument per intrinsic axis). ``M`` is the basis-column count
(``basis_size``).

Cross-frame interop
-------------------

One optimized Rust implementation drives every numerical frontend the user
might already be using. :meth:`evaluate` accepts an explicit ``backend=``
keyword or auto-detects from the input array's framework:

* ``backend="torch"`` (default for ``torch.Tensor`` inputs) — autograd-
  connected torch output.
* ``backend="numpy"`` (default for plain ``numpy.ndarray`` inputs) — pure
  FFI returning ``numpy.ndarray``; no gradient.
* ``backend="jax"`` (default for ``jax.Array`` inputs) — Rust FFI wrapped
  in ``jax.pure_callback`` so ``jit`` / ``vmap`` work end-to-end. ``grad``
  routes through a ``jax.custom_jvp`` whose tangent rule consults the
  descriptor's analytic Jacobian when one is available.

Lazy imports: importing ``gamfit`` never imports torch, jax, or any
frontend. Each backend module is imported only on first use of that
backend, so users with only one frontend installed pay zero startup cost
for the others.

Capability matrix
-----------------

Not every descriptor supports every backend. Each concrete descriptor
declares its supported backends via :attr:`BasisDescriptor.SUPPORTED_BACKENDS`
(a frozenset). Backends absent from that set raise
:class:`NotImplementedError` with a clear message at call time; the
descriptor does NOT silently fall back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch as _torch_t

    TensorLike = _torch_t.Tensor


# ---------------------------------------------------------------------------
# Lazy framework loaders
# ---------------------------------------------------------------------------

_TORCH_MODULE: Any = None
_JAX_MODULE: Any = None
_JNP_MODULE: Any = None


def _torch() -> Any:
    """Lazy-import torch with a clear error when it is missing."""
    global _TORCH_MODULE
    if _TORCH_MODULE is None:
        try:
            import torch as _t
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "gamfit basis-descriptor evaluation with backend='torch' "
                "requires torch. Install torch (e.g. `pip install torch`) "
                "or pass backend='numpy' / backend='jax'."
            ) from exc
        _TORCH_MODULE = _t
    return _TORCH_MODULE


def _jax() -> tuple[Any, Any]:
    """Lazy-import jax + jax.numpy with a clear error when missing."""
    global _JAX_MODULE, _JNP_MODULE
    if _JAX_MODULE is None:
        try:
            import jax as _j
            import jax.numpy as _jnp
        except ImportError as exc:  # pragma: no cover - exercised only without jax
            raise ImportError(
                "gamfit basis-descriptor evaluation with backend='jax' "
                "requires jax. Install jax (e.g. `pip install jax`) "
                "or pass backend='numpy' / backend='torch'."
            ) from exc
        _JAX_MODULE = _j
        _JNP_MODULE = _jnp
    return _JAX_MODULE, _JNP_MODULE


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_VALID_BACKENDS = frozenset({"torch", "numpy", "jax"})


def _detect_backend(coords: Sequence[Any]) -> str:
    """Detect the framework of the first array-like coordinate.

    Detection is *non-importing*: we inspect the object's class module
    string before importing the framework. This means a user who only has
    numpy installed never accidentally triggers a torch or jax import.

    Cross-frame dispatch contract (see :mod:`gamfit._frame`):

    * Pure NumPy inputs return :class:`numpy.ndarray` — the default frame
      is NumPy so ``import gamfit`` works with just NumPy installed.
    * Torch inputs return :class:`torch.Tensor` with autograd connected.
    * JAX inputs return :class:`jax.Array`, safe under ``jit``/``vmap``.
    * Mixed frames (a torch tensor and a jax array in the same call) raise
      :class:`TypeError` with a clear "inputs must be in the same frame"
      message — there is no implicit conversion that would silently drop
      one frame's autograd graph.
    """
    from ._frame import detect_frame

    return detect_frame(*coords).value


def _normalize_backend(backend: str | None, coords: Sequence[Any]) -> str:
    if backend is None:
        return _detect_backend(coords)
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of "
            f"{sorted(_VALID_BACKENDS)}"
        )
    return backend


# ---------------------------------------------------------------------------
# Per-backend coordinate coercion
# ---------------------------------------------------------------------------
#
# The 1-D / equal-length validation and the per-backend stack live once, in
# the frame adapters (``gamfit._frame_<frame>``). The protocol just routes the
# already-normalized backend to the matching adapter.


def _stack_coords_torch(coords: Sequence[Any]) -> Any:
    """Stack 1D torch coordinates into a (B, d) float64 tensor."""
    from ._frame_torch import stack_coords

    return stack_coords(coords)


def _stack_coords_for(backend: str, coords: Sequence[Any]) -> Any:
    """Stack ``coords`` via the frame adapter for ``backend``.

    ``backend`` is already normalized to one of ``"torch"``/``"numpy"``/
    ``"jax"`` by :func:`_normalize_backend`.
    """
    if backend == "torch":
        return _stack_coords_torch(coords)
    if backend == "numpy":
        from ._frame_numpy import stack_coords_f64

        return stack_coords_f64(coords)
    from ._frame_jax import stack_coords

    return stack_coords(coords)


# ---------------------------------------------------------------------------
# Main mixin
# ---------------------------------------------------------------------------


class BasisDescriptor:
    """Mixin endowing a smooth descriptor with callable basis methods.

    Concrete descriptors implement at least one backend-specific evaluator:

    * :meth:`_evaluate_torch` — returns ``(B, M)`` torch tensor with autograd.
    * :meth:`_evaluate_numpy` — returns ``(B, M)`` numpy ndarray (no grad).
    * :meth:`_evaluate_jax`   — returns ``(B, M)`` jax array (jit/vmap-safe).

    The protocol also derives :meth:`jacobian` / :meth:`hessian` from the
    torch implementation via ``torch.autograd.functional``. Descriptors are
    free to override either with an analytic closed form — the contract is
    that autograd of evaluate matches the override to numerical eps.
    """

    #: Backends this descriptor supports. Concrete subclasses override.
    #: Membership is enforced at :meth:`evaluate` dispatch time.
    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})

    # ------------------------------------------------------------------ shape

    @property
    def intrinsic_dim(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} must implement intrinsic_dim"
        )

    @property
    def basis_size(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} must implement basis_size"
        )

    # ------------------------------------------------------------------ core

    def _evaluate_torch(self, coords: Any) -> Any:
        """Torch backend implementation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the torch backend"
        )

    def _evaluate_numpy(self, coords: Any) -> Any:
        """NumPy backend implementation.

        Concrete descriptors that support ``backend='numpy'`` override this
        method to call the Rust FFI directly and return a ``(B, M)``
        ``numpy.ndarray``. Descriptors that do not support numpy leave it
        raising :class:`NotImplementedError` AND drop ``"numpy"`` from
        :attr:`SUPPORTED_BACKENDS`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the numpy backend"
        )

    def _evaluate_jax(self, coords: Any) -> Any:
        """JAX backend implementation.

        Default: wrap :meth:`_evaluate_numpy` (Rust FFI) in a
        ``jax.pure_callback`` and attach a ``jax.custom_jvp`` so ``jit``,
        ``vmap``, and ``grad`` all work. The JVP rule consults the
        descriptor's :meth:`_jacobian_numpy` when implemented; otherwise
        ``grad`` raises a clear error and the user is told to route through
        a torch path or supply an analytic Jacobian.

        ``coords`` is a ``(B, d)`` jax.numpy array.
        """
        jax, jnp = _jax()
        B = int(coords.shape[0])
        M = int(self.basis_size)
        d = int(self.intrinsic_dim)
        out_shape = jax.ShapeDtypeStruct((B, M), coords.dtype)

        def _host(x):
            import numpy as np

            arr = np.asarray(x, dtype=np.float64)
            res = self._evaluate_numpy(arr)
            return np.asarray(res, dtype=np.asarray(x).dtype)

        @jax.custom_jvp
        def _fwd(x):
            return jax.pure_callback(_host, out_shape, x)

        @_fwd.defjvp
        def _fwd_jvp(primals, tangents):
            (x,) = primals
            (xdot,) = tangents
            primal_out = _fwd(x)
            if not hasattr(self, "_jacobian_numpy"):
                raise NotImplementedError(
                    f"{type(self).__name__} does not implement an analytic "
                    "jacobian for the JAX backend. Use backend='torch' for "
                    "grad/vjp until the Rust value_grad FFI lands."
                )
                # (unreachable below by design; raised above)
            jac_shape = jax.ShapeDtypeStruct((B, M, d), coords.dtype)

            def _host_jac(z):
                import numpy as np

                arr = np.asarray(z, dtype=np.float64)
                return np.asarray(
                    self._jacobian_numpy(arr), dtype=np.asarray(z).dtype
                )

            jac = jax.pure_callback(_host_jac, jac_shape, x)
            # tangent_out[b, m] = sum_k jac[b, m, k] * xdot[b, k]
            tangent_out = jnp.einsum("bmk,bk->bm", jac, xdot)
            return primal_out, tangent_out

        return _fwd(coords)

    def evaluate(self, *coords: Any, backend: str | None = None) -> Any:
        """Evaluate the basis at ``coords``. Returns ``(B, M)``.

        Parameters
        ----------
        *coords
            One positional 1D array per intrinsic axis. Each element may
            be a ``numpy.ndarray``, ``torch.Tensor``, or ``jax.Array``.
        backend
            ``"torch"`` / ``"numpy"`` / ``"jax"`` / ``None``. ``None``
            auto-detects from the framework of ``coords[0]``. Auto-detect
            does NOT import a framework just to check — class-module
            string is inspected first.

        Returns
        -------
        Array
            ``(B, M)`` array in the requested backend's native type.
        """
        if len(coords) != self.intrinsic_dim:
            raise ValueError(
                f"{type(self).__name__}.evaluate expected "
                f"{self.intrinsic_dim} coordinate argument(s), got {len(coords)}"
            )
        chosen = _normalize_backend(backend, coords)
        if chosen not in self.SUPPORTED_BACKENDS:
            raise NotImplementedError(
                f"{type(self).__name__} does not support backend={chosen!r}; "
                f"supported backends: {sorted(self.SUPPORTED_BACKENDS)}"
            )
        if chosen == "torch":
            stacked = _stack_coords_torch(coords)
            return self._evaluate_torch(stacked)
        if chosen == "numpy":
            stacked = _stack_coords_for("numpy", coords)
            return self._evaluate_numpy(stacked)
        # jax
        stacked = _stack_coords_for("jax", coords)
        return self._evaluate_jax(stacked)

    # -------------------------------------------------------------- jacobian

    def jacobian(self, *coords: Any) -> Any:
        """Per-row Jacobian ``∂Φ/∂x``; shape ``(B, M, d)``.

        Implemented via ``torch.autograd.functional.jacobian`` on a
        decoupled per-row evaluation, which is independent of batch index.
        Concrete descriptors may override with an analytic form.
        """
        torch = _torch()
        if len(coords) != self.intrinsic_dim:
            raise ValueError(
                f"{type(self).__name__}.jacobian expected "
                f"{self.intrinsic_dim} coordinate argument(s), got {len(coords)}"
            )
        stacked = _stack_coords_torch(coords).detach().requires_grad_(True)
        # Per-row jacobian: differentiate evaluate w.r.t. coords; rows are
        # independent across B so the cross-batch Jacobian block is zero.
        out = self._evaluate_torch(stacked)
        B, M = int(out.shape[0]), int(out.shape[1])
        d = self.intrinsic_dim
        jac = torch.zeros((B, M, d), dtype=out.dtype, device=out.device)
        for m in range(M):
            grads = torch.autograd.grad(
                out[:, m].sum(),
                stacked,
                retain_graph=True,
                create_graph=False,
            )[0]
            jac[:, m, :] = grads
        return jac

    # --------------------------------------------------------------- hessian

    def hessian(self, *coords: Any) -> Any:
        """Per-row Hessian ``∂²Φ/∂x∂xᵀ``; shape ``(B, M, d, d)``.

        Implemented by differentiating :meth:`jacobian` once more via
        autograd. Concrete descriptors may override with an analytic form.
        """
        torch = _torch()
        if len(coords) != self.intrinsic_dim:
            raise ValueError(
                f"{type(self).__name__}.hessian expected "
                f"{self.intrinsic_dim} coordinate argument(s), got {len(coords)}"
            )
        stacked = _stack_coords_torch(coords).detach().requires_grad_(True)
        out = self._evaluate_torch(stacked)
        B, M = int(out.shape[0]), int(out.shape[1])
        d = self.intrinsic_dim
        hess = torch.zeros((B, M, d, d), dtype=out.dtype, device=out.device)
        # First derivatives, with create_graph=True so we can differentiate again.
        first_grads = []
        for m in range(M):
            g = torch.autograd.grad(
                out[:, m].sum(),
                stacked,
                retain_graph=True,
                create_graph=True,
            )[0]
            first_grads.append(g)
        for m in range(M):
            for j in range(d):
                gj = first_grads[m][:, j]
                g2 = torch.autograd.grad(
                    gj.sum(),
                    stacked,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                hess[:, m, j, :] = g2
        return hess

    # -------------------------------------------------------------- protocol

    def __call__(self, *coords: Any, backend: str | None = None) -> Any:
        return self.evaluate(*coords, backend=backend)
