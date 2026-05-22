"""GAM as a torch ``nn.Module`` — for embedding inside neural networks.

The :class:`GAM` module holds smooth-term specs and their structural
parameters (centers, knots) as buffers. Each forward pass refits the
coefficients via Gaussian REML against the provided response (training
mode), or evaluates cached coefficients at the provided points
(inference mode after ``.freeze()``).

This is the drop-in for sparse-coding, manifold-discovery, and similar
architectures where positions are emitted by upstream neural layers and
the GAM coefficients should be fit fresh each batch with autograd
flowing back to the encoder.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..smooth import Smooth
from .fit import FitResult, fit


class GAM(nn.Module):
    """Multi-smooth additive GAM as a ``nn.Module``.

    Construction takes a list of :class:`Smooth` specs. Their structural
    parameters (centers / knots) become buffers (saved/loaded with
    ``state_dict``, not trained by Adam). Coefficients are runtime — they
    flow from each REML fit and are not parameters.

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
        if len(smooths) == 0:
            raise ValueError("GAM needs at least one Smooth")
        # Keep the specs themselves on self for dispatch.
        self.smooths: list[Smooth] = list(smooths)
        # Frozen coefficients (set by .freeze()).
        self._frozen_coefs: list[torch.Tensor] | None = None
        self._frozen_lambdas: torch.Tensor | None = None
        self.last_fit: FitResult | None = None

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
        if self.training and self._frozen_coefs is None:
            if response is None:
                raise ValueError("GAM in training mode requires response= for REML fit")
            result = fit(points, response, self.smooths)
            self.last_fit = result
            return result.fitted

        if self._frozen_coefs is None:
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
        if isinstance(result.coefficients, list):
            self._frozen_coefs = [c.detach() for c in result.coefficients]
        else:
            self._frozen_coefs = [result.coefficients.detach()]
        self._frozen_lambdas = result.lambdas.detach()
        self.last_fit = result
        self.eval()

    # ------------------------------------------------------------------
    # Internal: evaluate at frozen coefficients
    # ------------------------------------------------------------------

    def _evaluate_frozen(
        self, points: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Eval-mode forward: design @ frozen_coef per smooth, summed."""
        from .fit import _build_design_penalty

        assert self._frozen_coefs is not None
        points_list = (
            list(points) if isinstance(points, (list, tuple))
            else [points] * len(self.smooths)
        )
        fitted_parts: list[torch.Tensor] = []
        for s, pts, coef in zip(self.smooths, points_list, self._frozen_coefs):
            design, _ = _build_design_penalty(s, pts)
            if s.by is not None:
                by_t = (
                    s.by if isinstance(s.by, torch.Tensor)
                    else torch.as_tensor(s.by, dtype=torch.float64, device=design.device)
                ).reshape(-1)
                design = design * by_t.unsqueeze(1).to(design.dtype)
            fitted_parts.append(design @ coef.to(design.dtype))
        return torch.stack(fitted_parts, dim=0).sum(dim=0)


__all__ = ["GAM"]
