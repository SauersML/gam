"""Distributed Alignment Search (DAS) interchange swap decoder.

This module implements :class:`InterchangeSwapDecoder`, a decoder primitive
designed for Geiger-style Distributed Alignment Search (DAS) causal
abstraction analysis. DAS evaluates a causal hypothesis by performing
*interchange interventions*: take two inputs ``a`` and ``b``, transplant
the latent atoms hypothesized to encode a target concept (e.g. "hue")
from ``a`` into ``b``, decode, and check whether the reconstruction
swaps along the predicted causal dimension. The differentiable
swap-reconstruction error (``swap-R^2``) becomes a loss that aligns
atoms with interpretable concepts.

The standard Anthropic-style gated decoder ties the per-feature gate
and reconstruction magnitude into a single ``(F, F)`` matrix — there is
no isolated gate parameter to swap. :class:`InterchangeSwapDecoder`
deliberately decouples them: a per-feature scalar gate ``(F,)`` lives
in its OWN parameter space, separate from the ``(D, F)`` reconstruction
weights. Interchange interventions transplant gate values atom-by-atom
while reconstruction weights stay fixed, which is what makes
``swap_decode`` a meaningful causal probe.

Reference
---------
Geiger, Wu, Potts, Icard, Goodman. "Finding Alignments Between
Interpretable Causal Variables and Distributed Neural Representations."
CLeaR 2024.
"""
from __future__ import annotations

from typing import Final

import torch
from torch import nn


_VALID_SWAP_MODES: Final[frozenset[str]] = frozenset({"scalar_mask"})


class InterchangeSwapDecoder(nn.Module):
    """Decoder with a per-feature scalar gate decoupled from reconstruction.

    The reconstruction is ``x_hat = (g * z) @ W_dec^T + b``, where the
    scalar gate ``g`` is a length-``F`` learnable vector that lives in
    its own parameter space (independent of the ``(D, F)`` decoder
    matrix ``W_dec``). Decoupling the gate from the reconstruction
    weights is what enables interchange interventions: in
    :meth:`swap_decode`, the gate vector is composed atom-by-atom from
    two source latents while the reconstruction weights remain shared.

    Parameters
    ----------
    D:
        Output (reconstruction) width.
    F:
        Latent / atom width.
    swap_mode:
        Interchange composition rule. Only ``'scalar_mask'`` is
        currently supported: gate values for atoms in ``atom_mask`` are
        taken from ``z_a``, the rest from ``z_b``. The argument exists
        so the API stays stable when richer composition rules
        (e.g. per-token mask, soft mask) land later.
    bias:
        Whether to include a learnable output bias.
    init_scale:
        Stddev for the ``W_dec`` Gaussian initializer.

    Examples
    --------
    >>> import torch
    >>> from gamfit import InterchangeSwapDecoder
    >>> dec = InterchangeSwapDecoder(D=8, F=4)
    >>> z = torch.randn(3, 4)
    >>> x_hat = dec(z)
    >>> z_a = torch.randn(3, 4)
    >>> z_b = torch.randn(3, 4)
    >>> hue_mask = torch.tensor([True, False, True, False])
    >>> swapped = dec.swap_decode(z_a, z_b, atom_mask=hue_mask)
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

        # Reconstruction weights: x_hat = (g * z) @ W_dec^T + b
        self.W_dec = nn.Parameter(torch.empty(self.D, self.F, device=device, dtype=dtype))
        # Per-feature scalar gate in its OWN parameter space.
        # Init at 1.0 so a freshly-built decoder is a plain linear decoder
        # (gate is a no-op), making downstream tests / DAS warm-starts
        # interpretable from epoch 0.
        self.gate = nn.Parameter(torch.ones(self.F, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.D, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        nn.init.normal_(self.W_dec, mean=0.0, std=float(init_scale))

    # ------------------------------------------------------------------
    # internals
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

    def _decode_with_gate(
        self, z: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        # gate broadcasts over the batch dimension.
        gated = z * gate.to(dtype=z.dtype, device=z.device)
        out = gated @ self.W_dec.to(dtype=z.dtype, device=z.device).t()
        if self.bias is not None:
            out = out + self.bias.to(dtype=z.dtype, device=z.device)
        return out

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Standard gated decode: ``x_hat = (gate * z) @ W_dec^T + b``."""
        z = self._check_latent(z, "z")
        return self._decode_with_gate(z, self.gate)

    def swap_decode(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode an interchange-intervention latent.

        For atoms where ``atom_mask`` is True, take values from ``z_a``;
        for atoms where it is False, take values from ``z_b``. The gate
        and reconstruction weights are shared between the two
        reconstructions — only the latent activations are
        interchanged. This is the gradient-friendly form of a DAS
        causal-abstraction intervention.

        Parameters
        ----------
        z_a, z_b:
            Latent codes for the two source inputs. Both must be
            ``(B, F)`` with matching shapes.
        atom_mask:
            1-D bool tensor of length ``F``. ``True`` selects the
            corresponding atom from ``z_a``, ``False`` from ``z_b``.

        Returns
        -------
        torch.Tensor
            Reconstruction of the interchange-intervened latent.
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

        if self.swap_mode == "scalar_mask":
            mask = atom_mask.to(device=z_a.device).view(1, self.F)
            z_swapped = torch.where(mask, z_a, z_b)
            return self._decode_with_gate(z_swapped, self.gate)

        # _VALID_SWAP_MODES is enforced in __init__, so this is unreachable
        # unless the attribute is mutated after construction.
        raise ValueError(f"unsupported swap_mode {self.swap_mode!r}")


__all__ = ["InterchangeSwapDecoder"]
