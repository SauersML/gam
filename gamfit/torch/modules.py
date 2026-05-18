"""Wrap a fitted ``gamfit.Model`` as a frozen ``torch.nn.Module``.

The :func:`from_fitted` loader is the canonical way to embed an already-trained
gamfit GAM into a larger torch graph. The returned module routes the input
tensor through the saved model via the engine's array-input prediction path
and returns predictions on the user's original device and dtype.

Limitation. The forward pass crosses the numpy / Rust boundary, so PyTorch's
autograd cannot construct an analytic gradient with respect to the input
features. Gradients flowing back to ``X`` are therefore *not* supported in
this v1 implementation; the module is frozen by design (model coefficients are
registered as buffers, not parameters). A future revision can re-evaluate the
smooths inside torch to enable input-gradient flow; until then, use this when
the GAM is a fixed feature transform and you want a small torch ``nn.Module``
that can sit alongside other modules in your training loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from ._coerce import from_numpy_like, to_numpy_f64

if TYPE_CHECKING:
    from .._model import Model


class _FittedGamModule(nn.Module):
    """Frozen torch wrapper for a fitted :class:`gamfit.Model`.

    The module accepts a ``(N, F)`` tensor of features in the column order used
    at training time (the engine names columns ``x0``, ``x1``, ...) and returns
    a ``(N, P)`` tensor of predictions where ``P`` is the number of prediction
    columns the underlying model class emits (typically ``eta`` followed by
    ``mean``).
    """

    def __init__(self, model: "Model") -> None:
        super().__init__()
        self._model = model
        # A dummy buffer pins the module's device / dtype so ``model.to(...)``
        # affects the returned predictions. We do not register the model's
        # learned coefficients as torch buffers because the Rust engine holds
        # the canonical serialized state; mirroring them would duplicate
        # storage without enabling autograd through the basis evaluations.
        self.register_buffer("_anchor", torch.zeros((), dtype=torch.float64))

    @property
    def model(self) -> "Model":
        """The wrapped fitted gamfit model."""
        return self._model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError("from_fitted module expects a torch.Tensor input")
        if X.dim() != 2:
            raise ValueError(
                f"from_fitted module expects a 2-D (N, F) input, got shape {tuple(X.shape)}"
            )
        x_np = to_numpy_f64(X)
        out_np = self._model.predict_array(x_np)
        out_np = np.ascontiguousarray(np.asarray(out_np, dtype=np.float64))
        return from_numpy_like(out_np, X)


def from_fitted(model: "Model") -> nn.Module:
    """Wrap a fitted ``gamfit.Model`` as a frozen torch ``nn.Module``.

    Parameters
    ----------
    model:
        A fitted :class:`gamfit.Model` returned by ``gamfit.fit`` / ``fit_array``.

    Returns
    -------
    torch.nn.Module
        A module whose ``forward(X)`` accepts a ``(N, F)`` tensor and returns a
        ``(N, P)`` tensor of predictions on the same device and dtype as ``X``.
        The wrapped model is treated as frozen: it has no trainable parameters,
        and gradients do not flow back through ``X`` in this v1 implementation
        because the prediction path crosses the numpy / Rust boundary. See the
        module docstring for details and the planned follow-up direction.

    Examples
    --------
    >>> import gamfit
    >>> from gamfit.torch import from_fitted
    >>> model = gamfit.fit_array(X_train, y_train, formula="y ~ s(x0)")
    >>> wrapped = from_fitted(model)
    >>> preds = wrapped(torch.as_tensor(X_test))
    """
    if not hasattr(model, "predict_array"):
        raise TypeError(
            "from_fitted expects a fitted gamfit.Model with a predict_array method"
        )
    return _FittedGamModule(model)
