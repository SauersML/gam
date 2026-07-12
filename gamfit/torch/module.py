"""GAM as a torch ``nn.Module`` — for embedding inside neural networks.

The :class:`GAM` module holds smooth-term specs as its architecture. Each
forward pass refits coefficients via Gaussian REML against the provided
response (training mode), or evaluates persistent coefficient buffers at the
provided points (inference mode after ``.freeze()``).

This is the drop-in for sparse-coding, manifold-discovery, and similar
architectures where positions are emitted by upstream neural layers and
the GAM coefficients should be fit fresh each batch with autograd
flowing back to the encoder.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ..smooth import Smooth
from .fit import FitResult, fit


class GAM(nn.Module):
    """Multi-smooth additive GAM as a ``nn.Module``.

    Construction takes a list of :class:`Smooth` specs defining the module
    architecture. Coefficients flow from each REML fit and are not parameters;
    :meth:`freeze` snapshots them as persistent buffers saved by ``state_dict``
    and migrated by :meth:`~torch.nn.Module.to`.

    Forward modes
    -------------
    Training (``self.training == True``):
        ``forward(points, response)`` refits all smooths' coefficients
        against ``response`` and returns the fitted values plus the
        ``FitResult`` (latter via ``self.last_fit``).

    Inference (after ``self.freeze(...)``):
        ``forward(points)`` evaluates the cached coefficients at the
        given points. Feedforward, no REML, no response needed.

    Parameters
    ----------
    smooths : list of :class:`Smooth`. The additive structure.

    Notes
    -----
    Backward routes through the gamfit engine VJP w.r.t. ``points`` and
    each smooth's ``by`` gate. Smooth ``centers`` / ``knots`` are
    structural (no gradient propagated).
    """

    def __init__(self, smooths: Sequence[Smooth]) -> None:
        super().__init__()
        self.smooths: list[Smooth] = list(smooths)
        if not self.smooths:
            raise ValueError("GAM needs at least one Smooth")
        # Keep the specs themselves on self for dispatch.  A frozen coefficient
        # block is persistent module state, not an ordinary Python attribute:
        # registration makes state_dict(), Module.to(), DDP replication, and
        # device/dtype transforms obey the standard nn.Module contract.  Empty
        # buffers denote the unfrozen state and give load_state_dict stable keys;
        # _load_from_state_dict resizes them before PyTorch copies a frozen state.
        for index in range(len(self.smooths)):
            self.register_buffer(
                self._coefficient_buffer_name(index),
                torch.empty(0),
                persistent=True,
            )
        self.last_fit: FitResult | None = None

    @staticmethod
    def _coefficient_buffer_name(index: int) -> str:
        return f"_frozen_coefficient_{index}"

    def _frozen_coefficients(self) -> list[torch.Tensor] | None:
        coefficients = [
            getattr(self, self._coefficient_buffer_name(index))
            for index in range(len(self.smooths))
        ]
        populated = [coefficient.numel() > 0 for coefficient in coefficients]
        if not any(populated):
            return None
        if not all(populated):
            raise RuntimeError(
                "GAM has a partially populated frozen coefficient state; "
                "every smooth block must be present"
            )
        return coefficients

    def _install_frozen_coefficients(
        self, coefficient_blocks: Sequence[torch.Tensor]
    ) -> None:
        if len(coefficient_blocks) != len(self.smooths):
            raise RuntimeError(
                "fit returned "
                f"{len(coefficient_blocks)} coefficient blocks for "
                f"{len(self.smooths)} smooths"
            )

        snapshots: list[torch.Tensor] = []
        output_dim: int | None = None
        for index, coefficient in enumerate(coefficient_blocks):
            if not isinstance(coefficient, torch.Tensor):
                raise TypeError(
                    f"coefficient block {index} must be a torch.Tensor; "
                    f"got {type(coefficient).__name__}"
                )
            if coefficient.dim() != 2 or 0 in coefficient.shape:
                raise RuntimeError(
                    f"coefficient block {index} must have non-empty shape "
                    f"(basis, output); got {tuple(coefficient.shape)}"
                )
            if not coefficient.is_floating_point():
                raise TypeError(
                    f"coefficient block {index} must be floating point; "
                    f"got {coefficient.dtype}"
                )
            block_output_dim = int(coefficient.shape[1])
            if output_dim is None:
                output_dim = block_output_dim
            elif block_output_dim != output_dim:
                raise RuntimeError(
                    "all frozen coefficient blocks must share one output "
                    f"dimension; block 0 has {output_dim}, block {index} has "
                    f"{block_output_dim}"
                )
            # freeze() promises a snapshot.  clone() prevents later mutation of
            # the FitResult tensor from silently changing the deployed module.
            snapshots.append(coefficient.detach().clone().contiguous())

        # Install only after every block validates, so a failed re-freeze cannot
        # leave a half-updated model.
        for index, snapshot in enumerate(snapshots):
            setattr(self, self._coefficient_buffer_name(index), snapshot)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        points: torch.Tensor | Sequence[torch.Tensor],
        response: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the additive fitted values at ``points``.

        Training mode: requires ``response``; refits coefficients via REML.
        Eval mode (after ``.freeze``): ignores ``response``; uses cached
        coefficients.
        """
        frozen_coefficients = self._frozen_coefficients()
        if self.training and frozen_coefficients is None:
            if response is None:
                raise ValueError("GAM in training mode requires response= for REML fit")
            result = fit(points, response, self.smooths)
            self.last_fit = result
            return result.fitted

        if frozen_coefficients is None:
            raise RuntimeError(
                "GAM is in eval mode but no frozen coefficients; "
                "call .freeze(points, response) first or set .train()"
            )
        return self._evaluate_frozen(points)

    # ------------------------------------------------------------------
    # Freeze (lock-and-cache)
    # ------------------------------------------------------------------

    def freeze(
        self,
        points: torch.Tensor | Sequence[torch.Tensor],
        response: torch.Tensor,
    ) -> None:
        """Run one REML fit on ``(points, response)`` and snapshot the
        coefficients as buffers. After this, ``forward(points)`` evaluates
        the cached coefficients feedforward — no gamfit call at inference.
        """
        result = fit(points, response, self.smooths)
        coefficient_blocks = (
            result.coefficients
            if isinstance(result.coefficients, list)
            else [result.coefficients]
        )
        self._install_frozen_coefficients(coefficient_blocks)
        self.last_fit = result
        self.eval()

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Frozen coefficient shapes are learned at fit time, so a fresh GAM has
        # zero-length registered buffers.  Resize each destination to the saved
        # block before nn.Module performs its ordinary checked copy.  The current
        # module device is retained; dtype follows the saved fit exactly.
        saved_blocks = [
            state_dict.get(prefix + self._coefficient_buffer_name(index))
            for index in range(len(self.smooths))
        ]
        provided_blocks = sum(block is not None for block in saved_blocks)
        if 0 < provided_blocks < len(self.smooths):
            error_msgs.append(
                "A GAM state_dict must provide every frozen coefficient buffer "
                f"or none of them; found {provided_blocks} of {len(self.smooths)}."
            )
        nonempty_blocks = [
            block
            for block in saved_blocks
            if isinstance(block, torch.Tensor) and block.numel() > 0
        ]
        if nonempty_blocks and len(nonempty_blocks) != len(self.smooths):
            error_msgs.append(
                "A frozen GAM state must contain one non-empty coefficient "
                f"block per smooth; found {len(nonempty_blocks)} for "
                f"{len(self.smooths)} smooths."
            )
        output_dims = {
            int(block.shape[1])
            for block in nonempty_blocks
            if block.dim() == 2 and 0 not in block.shape
        }
        if len(output_dims) > 1:
            error_msgs.append(
                "Frozen GAM coefficient blocks in state_dict do not share one "
                f"output dimension: {sorted(output_dims)}."
            )

        for index in range(len(self.smooths)):
            name = self._coefficient_buffer_name(index)
            key = prefix + name
            incoming = state_dict.get(key)
            if incoming is None:
                continue
            if not isinstance(incoming, torch.Tensor):
                error_msgs.append(
                    f'While loading {key!r}, expected a tensor but found '
                    f'{type(incoming).__name__}.'
                )
                continue
            current = getattr(self, name)
            if incoming.numel() == 0:
                setattr(
                    self,
                    name,
                    torch.empty(0, dtype=incoming.dtype, device=current.device),
                )
                continue
            if incoming.dim() != 2 or 0 in incoming.shape:
                error_msgs.append(
                    f'While loading {key!r}, expected a non-empty 2-D frozen '
                    f'coefficient block but found shape {tuple(incoming.shape)}.'
                )
                continue
            setattr(
                self,
                name,
                torch.empty(
                    incoming.shape,
                    dtype=incoming.dtype,
                    device=current.device,
                ),
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # ------------------------------------------------------------------
    # Internal: evaluate at frozen coefficients
    # ------------------------------------------------------------------

    def _evaluate_frozen(
        self, points: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Eval-mode forward: design @ frozen_coef per smooth, summed."""
        from .fit import _build_design_penalty

        frozen_coefficients = self._frozen_coefficients()
        if frozen_coefficients is None:
            raise RuntimeError("frozen_coefs is None")
        points_list = (
            list(points) if isinstance(points, (list, tuple))
            else [points] * len(self.smooths)
        )
        if len(points_list) != len(self.smooths):
            raise ValueError(
                f"got {len(points_list)} points tensors for "
                f"{len(self.smooths)} smooths"
            )
        if len(frozen_coefficients) != len(self.smooths):
            raise RuntimeError(
                f"GAM has {len(frozen_coefficients)} frozen coefficient blocks for "
                f"{len(self.smooths)} smooths"
            )
        fitted_parts: list[torch.Tensor] = []
        for s, pts, coef in zip(
            self.smooths,
            points_list,
            frozen_coefficients,
            strict=True,
        ):
            design, _ = _build_design_penalty(s, pts)
            if s.by is not None:
                by_t = (
                    s.by if isinstance(s.by, torch.Tensor)
                    else torch.as_tensor(s.by, dtype=torch.float64, device=design.device)
                ).reshape(-1)
                design = design * by_t.unsqueeze(1).to(
                    device=design.device, dtype=design.dtype
                )
            fitted_parts.append(design @ coef.to(design.dtype))
        return torch.stack(fitted_parts, dim=0).sum(dim=0)


__all__ = ["GAM"]
