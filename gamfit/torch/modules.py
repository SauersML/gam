"""Wrap a fitted ``gamfit.Model`` as a frozen ``torch.nn.Module``.

The :func:`from_fitted` loader is the canonical way to embed an already-trained
gamfit GAM into a larger torch graph. The returned module routes the input
tensor through the saved model via the engine's array-input prediction path
and returns predictions on the user's original device and dtype.

Limitation. The forward pass crosses the numpy / Rust boundary, so PyTorch's
autograd cannot construct an analytic gradient with respect to the input
features. Gradients flowing back to ``X`` are therefore *not* supported in
this v1 implementation; the module is frozen by design because it exposes no
trainable parameters and delegates prediction to the fitted model. A future
revision can re-evaluate the smooths inside torch to enable input-gradient
flow; until then, use this when the GAM is a fixed feature transform and you
want a small torch ``nn.Module`` that can sit alongside other modules in your
training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F_torch
from torch import nn

from .._binding import rust_module
from ._coerce import from_numpy_like, to_numpy_f64

if TYPE_CHECKING:
    from .._model import Model


@dataclass(frozen=True, slots=True)
class ManifoldSAEConfig:
    """Configuration for :class:`ManifoldSAE`."""

    D: int
    F: int
    K: int
    bias: bool = True
    init_scale: float = 0.02


@dataclass(frozen=True, slots=True)
class ManifoldSAEOutput:
    """Output bundle returned by :class:`ManifoldSAE.forward`."""

    z: torch.Tensor
    x_hat: torch.Tensor
    theta: torch.Tensor
    amp: torch.Tensor
    gate: torch.Tensor


def _check_2d_float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _hard_topk_mask(values: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k >= values.shape[-1]:
        return torch.ones_like(values, dtype=torch.bool)
    _, idx = torch.topk(values.abs(), k=top_k, dim=-1)
    return torch.zeros_like(values, dtype=torch.bool).scatter_(-1, idx, True)


def _masked_ste(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hard = values * mask.to(dtype=values.dtype)
    return values + (hard - values).detach()


class ManifoldSAE(nn.Module):
    """Trainable torch mirror of the SAE manifold primitive.

    >>> import torch
    >>> from gamfit.torch import ManifoldSAE, ManifoldSAEConfig
    >>> out = ManifoldSAE(ManifoldSAEConfig(D=8, F=4, K=2))(torch.randn(3, 8))
    """

    def __init__(self, cfg: ManifoldSAEConfig) -> None:
        super().__init__()
        if not isinstance(cfg, ManifoldSAEConfig):
            raise TypeError("ManifoldSAE expects a ManifoldSAEConfig")
        if cfg.D <= 0 or cfg.F <= 0:
            raise ValueError("ManifoldSAEConfig.D and F must be positive")
        if cfg.K <= 0 or cfg.K > cfg.F:
            raise ValueError("ManifoldSAEConfig.K must satisfy 0 < K <= F")
        if cfg.init_scale <= 0.0:
            raise ValueError("ManifoldSAEConfig.init_scale must be > 0")
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.D, cfg.F, bias=cfg.bias)
        self.decoder = nn.Linear(cfg.F, cfg.D, bias=cfg.bias)
        self.gate_bias = nn.Parameter(torch.zeros(cfg.F))
        self.reset_parameters()
        self._snapshot: dict[str, torch.Tensor | ManifoldSAEConfig] = {}

    def reset_parameters(self) -> None:
        nn.init.normal_(self.encoder.weight, mean=0.0, std=self.cfg.init_scale)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=self.cfg.init_scale)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        nn.init.zeros_(self.gate_bias)

    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        x = _check_2d_float_tensor(x, "x")
        if x.shape[1] != self.cfg.D:
            raise ValueError(f"ManifoldSAE expected input width {self.cfg.D}, got {x.shape[1]}")
        theta = self.encoder(x)
        gate = torch.sigmoid(theta + self.gate_bias.to(dtype=theta.dtype, device=theta.device))
        amp = F_torch.softplus(theta)
        z_pre = gate * amp
        z = _masked_ste(z_pre, _hard_topk_mask(z_pre, self.cfg.K))
        x_hat = self.decoder(z)
        return ManifoldSAEOutput(z=z, x_hat=x_hat, theta=theta, amp=amp, gate=gate)

    @torch.no_grad()
    def update_snapshot(self) -> None:
        self._snapshot = {
            "cfg": replace(self.cfg),
            "encoder_weight": self.encoder.weight.detach().cpu().clone(),
            "decoder_weight": self.decoder.weight.detach().cpu().clone(),
            "gate_bias": self.gate_bias.detach().cpu().clone(),
        }
        if self.encoder.bias is not None:
            self._snapshot["encoder_bias"] = self.encoder.bias.detach().cpu().clone()
        if self.decoder.bias is not None:
            self._snapshot["decoder_bias"] = self.decoder.bias.detach().cpu().clone()

    @torch.no_grad()
    def extract_feature_curves(self) -> dict[str, object]:
        decoder_weight = self.decoder.weight.detach().cpu().clone()
        encoder_weight = self.encoder.weight.detach().cpu().clone()
        gate_bias = self.gate_bias.detach().cpu()
        features = [
            {
                "feature": j,
                "decoder": decoder_weight[:, j].clone(),
                "encoder": encoder_weight[j, :].clone(),
                "gate_bias": gate_bias[j].clone(),
            }
            for j in range(self.cfg.F)
        ]
        return {
            "cfg": replace(self.cfg),
            "features": features,
            "decoder_weight": decoder_weight,
            "encoder_weight": encoder_weight,
            "snapshot": dict(self._snapshot),
        }


class TopKActivationPenalty(nn.Module):
    """Hard top-k activation with straight-through gradients.

    >>> import torch
    >>> from gamfit.torch import TopKActivationPenalty
    >>> z = TopKActivationPenalty(top_k=2, F=4)(torch.randn(3, 4))
    """

    def __init__(self, top_k: int, F: int) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.F = int(F)
        if self.F <= 0:
            raise ValueError("F must be positive")
        if self.top_k <= 0 or self.top_k > self.F:
            raise ValueError("top_k must satisfy 0 < top_k <= F")
        self.register_buffer("_last_k_pred", torch.tensor(float(self.top_k)), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _check_2d_float_tensor(z, "z")
        if z.shape[1] != self.F:
            raise ValueError(f"TopKActivationPenalty expected width {self.F}, got {z.shape[1]}")
        mask = _hard_topk_mask(z, self.top_k)
        self._last_k_pred = mask.sum(dim=-1).to(dtype=z.dtype).mean().detach()
        return _masked_ste(z, mask)

    def penalty(self) -> torch.Tensor:
        return self._last_k_pred


class GatedSAEDecoder(nn.Module):
    """Gated SAE decoder with separate gate and magnitude matrices.

    >>> import torch
    >>> from gamfit.torch import GatedSAEDecoder
    >>> x_hat = GatedSAEDecoder(F=4, D=8)(torch.randn(3, 4))
    """

    def __init__(
        self,
        F: int,
        D: int,
        *,
        bias: bool = True,
        init_scale: float = 0.02,
        ste: bool = True,
    ) -> None:
        super().__init__()
        self.F = int(F)
        self.D = int(D)
        self.ste = bool(ste)
        if self.F <= 0 or self.D <= 0:
            raise ValueError("F and D must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be > 0")
        self.w_gate = nn.Parameter(torch.empty(self.F, self.F))
        self.w_amp = nn.Parameter(torch.empty(self.D, self.F))
        self.bias = nn.Parameter(torch.zeros(self.D)) if bias else None
        nn.init.normal_(self.w_gate, mean=0.0, std=float(init_scale))
        nn.init.normal_(self.w_amp, mean=0.0, std=float(init_scale))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _check_2d_float_tensor(z, "z")
        if z.shape[1] != self.F:
            raise ValueError(f"GatedSAEDecoder expected width {self.F}, got {z.shape[1]}")
        gate_logits = z @ self.w_gate.to(dtype=z.dtype, device=z.device).t()
        soft_gate = torch.sigmoid(gate_logits)
        if self.ste:
            hard_gate = (gate_logits > 0).to(dtype=z.dtype)
            gate = soft_gate + (hard_gate - soft_gate).detach()
        else:
            gate = soft_gate
        out = (gate * z) @ self.w_amp.to(dtype=z.dtype, device=z.device).t()
        if self.bias is not None:
            out = out + self.bias.to(dtype=z.dtype, device=z.device)
        return out


class SparsityPenalty(nn.Module):
    """Scalar sparsity penalty for SAE activations.

    >>> import torch
    >>> from gamfit.torch import SparsityPenalty
    >>> loss = SparsityPenalty("l1", 0.01)(torch.randn(3, 4))
    """

    def __init__(
        self,
        kind: Literal["l1", "l0", "hoyer"],
        weight: float,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if kind not in {"l1", "l0", "hoyer"}:
            raise ValueError("kind must be one of 'l1', 'l0', or 'hoyer'")
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        self.kind = kind
        self.weight = float(weight)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _check_2d_float_tensor(z, "z")
        if self.kind == "l1":
            value = z.abs().mean()
        elif self.kind == "l0":
            soft = 1.0 - torch.exp(-z.abs() / self.eps)
            hard = (z != 0).to(dtype=z.dtype)
            value = (soft + (hard - soft).detach()).mean()
        else:
            flat = z.reshape(z.shape[0], -1)
            l1 = flat.abs().sum(dim=-1)
            l2 = torch.sqrt((flat * flat).sum(dim=-1) + self.eps)
            denom = flat.shape[1] ** 0.5 - 1.0
            if denom <= 0.0:
                value = z.new_zeros(())
            else:
                value = ((l1 / l2 - 1.0) / denom).clamp_min(0.0).mean()
        return z.new_tensor(self.weight) * value


class SoftmaxAssignmentSparsityPenalty(nn.Module):
    """Softmax-relaxed top-k assignment sparsity.

    >>> import torch
    >>> from gamfit.torch import SoftmaxAssignmentSparsityPenalty
    >>> loss = SoftmaxAssignmentSparsityPenalty(F=4, target_k=2)(torch.randn(3, 4))
    """

    def __init__(
        self,
        F: int,
        target_k: int,
        *,
        weight: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.F = int(F)
        self.target_k = int(target_k)
        self.weight = float(weight)
        self.temperature = float(temperature)
        if self.F <= 0:
            raise ValueError("F must be positive")
        if self.target_k <= 0 or self.target_k > self.F:
            raise ValueError("target_k must satisfy 0 < target_k <= F")
        if self.weight < 0.0:
            raise ValueError("weight must be non-negative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = _check_2d_float_tensor(logits, "logits")
        if logits.shape[1] != self.F:
            raise ValueError(
                f"SoftmaxAssignmentSparsityPenalty expected width {self.F}, got {logits.shape[1]}"
            )
        probs = torch.softmax(logits / self.temperature, dim=-1)
        sorted_probs = probs.sort(dim=-1, descending=True).values
        tail_mass = sorted_probs[:, self.target_k :].sum(dim=-1)
        return logits.new_tensor(self.weight) * tail_mass.mean()


class _FittedGamModule(nn.Module):
    """Frozen torch wrapper for a fitted :class:`gamfit.Model`.

    The module accepts a ``(N, F)`` tensor of features in the column order used
    at training time (the engine names columns ``x0``, ``x1``, ...) and returns
    a ``(N, P)`` tensor of predictions where ``P`` is the number of prediction
    columns the underlying model class emits (typically ``eta`` followed by
    ``mean``).
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 2:
            raise ValueError(
                f"from_fitted module expects a 2-D (N, F) input, got shape {tuple(X.shape)}"
            )
        out_np = self._model.predict_array(to_numpy_f64(X))
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
    return rust_module().torch_from_fitted(_FittedGamModule, model)
