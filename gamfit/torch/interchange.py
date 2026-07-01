"""Distributed Alignment Search (DAS) interchange swap decoder.

This module is the PyTorch-facing thin wrapper for the Rust primitive
``gam::terms::decoders::interchange_decoder``. The decoder is *not* gamfit-specific
— the same Rust primitive is callable from the CLI and the Rust library.
This file exists only because PyTorch needs a custom
:class:`torch.autograd.Function` to register the Rust forward and the
Rust analytic gradients with the autograd tape; the actual arithmetic
(both forward and backward) lives in Rust.

DAS (Geiger, Wu, Potts, Icard, Goodman, CLeaR 2024) evaluates a causal
hypothesis by transplanting the latent atoms hypothesized to encode a
target concept from one input ``a`` into another input ``b`` and
back-propagating a swap-reconstruction error. The Anthropic-style gated
decoder ties gate and magnitude in a single ``(F, F)`` matrix and is
therefore unsuitable; this decoder gives the per-feature scalar gate
its own parameter space so it has something to transplant.
"""
from __future__ import annotations

from typing import Any, Final

import numpy as np
import torch
from torch import nn

from .._binding import rust_module


_VALID_SWAP_MODES: Final[frozenset[str]] = frozenset({"scalar_mask"})


def _as_f64_cpu(t: torch.Tensor) -> np.ndarray:
    # Move host first, then cast: a fused ``.to(dtype=float64, device="cpu")``
    # forces the float64 cast on the source device, which MPS cannot do.
    return np.ascontiguousarray(t.detach().cpu().to(dtype=torch.float64).numpy())


def _as_f64_2d(t: torch.Tensor) -> np.ndarray:
    return _as_f64_cpu(t)


def _as_f64_1d(t: torch.Tensor) -> np.ndarray:
    return _as_f64_cpu(t)


def _as_bool_1d(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(device="cpu").numpy().astype(np.bool_))


def _from_numpy_like(arr: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=ref.dtype, device=ref.device)


class _InterchangeDecodeFn(torch.autograd.Function):
    """Plain gated-decode autograd shim around the Rust primitive."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        z: torch.Tensor,
        weights: torch.Tensor,
        gate: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        rm = rust_module()
        z_np = _as_f64_2d(z)
        w_np = _as_f64_2d(weights)
        g_np = _as_f64_1d(gate)
        b_np = None if bias is None else _as_f64_1d(bias)
        out_np = np.asarray(rm.interchange_decode_forward(z_np, w_np, g_np, b_np))
        ctx.save_for_backward(z, weights, gate)
        ctx.has_bias = bias is not None
        return _from_numpy_like(out_np, z)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        z, weights, gate = ctx.saved_tensors
        rm = rust_module()
        grad_z_np, grad_w_np, grad_g_np, grad_b_np = rm.interchange_decode_backward(
            _as_f64_2d(z),
            _as_f64_2d(weights),
            _as_f64_1d(gate),
            _as_f64_2d(grad_out),
            bool(ctx.has_bias),
        )
        grad_z = _from_numpy_like(np.asarray(grad_z_np), z)
        grad_w = _from_numpy_like(np.asarray(grad_w_np), weights)
        grad_g = _from_numpy_like(np.asarray(grad_g_np), gate)
        grad_b: torch.Tensor | None
        if grad_b_np is None:
            grad_b = None
        else:
            grad_b = _from_numpy_like(np.asarray(grad_b_np), gate)
        return grad_z, grad_w, grad_g, grad_b


class _InterchangeSwapFn(torch.autograd.Function):
    """Masked-swap autograd shim around the Rust primitive."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
        gate: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        rm = rust_module()
        out_np = np.asarray(
            rm.interchange_swap_forward(
                _as_f64_2d(z_a),
                _as_f64_2d(z_b),
                _as_bool_1d(mask),
                _as_f64_2d(weights),
                _as_f64_1d(gate),
                None if bias is None else _as_f64_1d(bias),
            )
        )
        ctx.save_for_backward(z_a, z_b, mask, weights, gate)
        ctx.has_bias = bias is not None
        return _from_numpy_like(out_np, z_a)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        z_a, z_b, mask, weights, gate = ctx.saved_tensors
        rm = rust_module()
        (
            grad_za_np,
            grad_zb_np,
            grad_w_np,
            grad_g_np,
            grad_b_np,
        ) = rm.interchange_swap_backward(
            _as_f64_2d(z_a),
            _as_f64_2d(z_b),
            _as_bool_1d(mask),
            _as_f64_2d(weights),
            _as_f64_1d(gate),
            _as_f64_2d(grad_out),
            bool(ctx.has_bias),
        )
        grad_za = _from_numpy_like(np.asarray(grad_za_np), z_a)
        grad_zb = _from_numpy_like(np.asarray(grad_zb_np), z_b)
        grad_w = _from_numpy_like(np.asarray(grad_w_np), weights)
        grad_g = _from_numpy_like(np.asarray(grad_g_np), gate)
        grad_b: torch.Tensor | None
        if grad_b_np is None:
            grad_b = None
        else:
            grad_b = _from_numpy_like(np.asarray(grad_b_np), gate)
        # mask is bool, no gradient.
        return grad_za, grad_zb, None, grad_w, grad_g, grad_b


class InterchangeSwapDecoder(nn.Module):
    """Decoder with a per-feature scalar gate decoupled from reconstruction.

    Thin owner of the trainable parameters. All math is dispatched to the
    shared Rust primitive ``gam::terms::decoders::interchange_decoder`` via a
    :class:`torch.autograd.Function` that registers Rust-computed
    gradients with PyTorch's autograd tape.

    Forward
    -------
    ``x_hat[i, d] = sum_f gate[f] * z[i, f] * W_dec[d, f] + bias[d]``

    Interchange swap
    ----------------
    ``swap_decode(z_a, z_b, atom_mask)`` composes the latent atom-by-atom:
    where ``atom_mask`` is ``True`` the column of ``z_a`` is used, else
    the column of ``z_b``. Reconstruction weights and gate are shared.
    The composition is differentiable end-to-end so a swap-R^2 term can
    drive training.

    Parameters
    ----------
    D:
        Output (reconstruction) width.
    F:
        Latent / atom width.
    swap_mode:
        Only ``'scalar_mask'`` is currently supported. The argument
        stays in the signature so richer composition rules can extend
        later without breaking callers.

    Examples
    --------
    >>> import torch
    >>> from gamfit import InterchangeSwapDecoder
    >>> dec = InterchangeSwapDecoder(D=8, F=4)
    >>> x_hat = dec(torch.randn(3, 4))
    >>> z_a, z_b = torch.randn(3, 4), torch.randn(3, 4)
    >>> mask = torch.tensor([True, False, True, False])
    >>> swapped = dec.swap_decode(z_a, z_b, atom_mask=mask)
    >>> loss = ((swapped - x_hat) ** 2).mean()
    >>> loss.backward()
    """

    def __init__(
        self,
        D: int,
        F: int,
        swap_mode: str = "scalar_mask",
        *,
        bias: bool = True,
        init_scale: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if not isinstance(D, int) or not isinstance(F, int):
            raise TypeError("D and F must be ints")
        if D <= 0 or F <= 0:
            raise ValueError("D and F must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be > 0")
        if swap_mode not in _VALID_SWAP_MODES:
            raise ValueError(
                f"unknown swap_mode {swap_mode!r}; "
                f"supported: {sorted(_VALID_SWAP_MODES)}"
            )

        self.D = int(D)
        self.F = int(F)
        self.swap_mode = swap_mode

        self.W_dec = nn.Parameter(torch.empty(self.D, self.F, device=device, dtype=dtype))
        # Gate initialised to 1.0 so a freshly-built decoder behaves like a
        # plain linear decoder (no gating bias).
        self.gate = nn.Parameter(torch.ones(self.F, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.D, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        nn.init.normal_(self.W_dec, mean=0.0, std=float(init_scale))

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------

    def _check_latent(self, z: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if z.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(z.shape)}")
        if z.shape[1] != self.F:
            raise ValueError(
                f"{name} has width {z.shape[1]}, expected F={self.F}"
            )
        if not torch.is_floating_point(z):
            raise TypeError(f"{name} must be a floating-point tensor")
        return z

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Standard gated decode via the Rust primitive."""
        z = self._check_latent(z, "z")
        return _InterchangeDecodeFn.apply(z, self.W_dec, self.gate, self.bias)

    def swap_decode(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Interchange-intervention decode via the Rust primitive.

        For atoms with ``atom_mask[f]`` true, use ``z_a[:, f]``; otherwise
        ``z_b[:, f]``. Returns a tensor with autograd wired through to
        ``z_a``, ``z_b``, ``W_dec``, ``gate``, and ``bias``.
        """
        z_a = self._check_latent(z_a, "z_a")
        z_b = self._check_latent(z_b, "z_b")
        if z_a.shape != z_b.shape:
            raise ValueError(
                f"z_a and z_b must have the same shape, "
                f"got {tuple(z_a.shape)} and {tuple(z_b.shape)}"
            )
        if not isinstance(atom_mask, torch.Tensor):
            raise TypeError("atom_mask must be a torch.Tensor")
        if atom_mask.dtype != torch.bool:
            raise TypeError(
                f"atom_mask must be a bool tensor, got dtype {atom_mask.dtype}"
            )
        if atom_mask.dim() != 1 or atom_mask.shape[0] != self.F:
            raise ValueError(
                f"atom_mask must be 1-D of length F={self.F}, "
                f"got shape {tuple(atom_mask.shape)}"
            )
        if self.swap_mode != "scalar_mask":
            # Enforced in __init__; this guards against post-construction mutation.
            raise ValueError(f"unsupported swap_mode {self.swap_mode!r}")
        return _InterchangeSwapFn.apply(
            z_a, z_b, atom_mask, self.W_dec, self.gate, self.bias
        )


__all__ = ["InterchangeSwapDecoder"]
