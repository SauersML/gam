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

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .._binding import rust_module
from ._coerce import from_numpy_like, to_numpy_f64

if TYPE_CHECKING:
    from .._model import Model


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


class _AdaptiveTopKSTE(torch.autograd.Function):
    """STE: hard per-row top-``round(k_pred_i)`` mask forward, soft-mask gradient backward.

    The soft mask is ``m_ij = sigmoid((|z_ij| - tau_i) / temperature)`` where
    ``tau_i`` is the differentiable ``k_pred_i``-th order statistic of ``|z_i|``
    obtained by sorting absolute values and interpolating with a smooth
    weighting around ``k_pred_i``. ``sum_j m_ij`` then satisfies
    ``E[sum_j m_ij] ~= k_pred_i`` so ``d E[K_pred] / d k_pred_i = 1`` analytically
    (the constant Jacobian that makes ``lambda * E[K_pred]`` gradient-clean).
    """

    @staticmethod
    def forward(
        ctx: object,
        z: torch.Tensor,
        k_pred: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        abs_z = z.abs()
        n_rows, width = z.shape
        sorted_abs, _ = torch.sort(abs_z, dim=-1, descending=True)
        k_clamped = k_pred.clamp(min=1.0 - 1e-6, max=float(width) - 1e-6)
        k_floor = k_clamped.floor().to(torch.long).clamp(min=0, max=width - 1)
        k_ceil = (k_floor + 1).clamp(max=width - 1)
        frac = (k_clamped - k_floor.to(k_clamped.dtype)).unsqueeze(-1)
        tau_floor = sorted_abs.gather(-1, k_floor.unsqueeze(-1))
        tau_ceil = sorted_abs.gather(-1, k_ceil.unsqueeze(-1))
        tau = ((1.0 - frac) * tau_floor + frac * tau_ceil).squeeze(-1)
        diff = (abs_z - tau.unsqueeze(-1)) / float(temperature)
        soft_mask = torch.sigmoid(diff)
        # Hard mask: top-round(k_pred) per row.
        k_int = k_pred.round().clamp(min=1, max=width).to(torch.long)
        hard_mask = torch.zeros_like(z, dtype=z.dtype)
        for row in range(n_rows):
            kk = int(k_int[row].item())
            _, idx = torch.topk(abs_z[row], k=kk)
            hard_mask[row, idx] = 1.0
        z_active_soft = z * soft_mask.to(z.dtype)
        z_active = z_active_soft + (z * hard_mask - z_active_soft).detach()
        k_pred_sum = soft_mask.sum(dim=-1)
        ctx.save_for_backward(z, soft_mask, tau, abs_z, k_pred)
        ctx.temperature = float(temperature)
        return z_active, k_pred_sum

    @staticmethod
    def backward(
        ctx: object,
        grad_z_active: torch.Tensor,
        grad_k_pred_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        z, soft_mask, tau, abs_z, k_pred = ctx.saved_tensors
        temp = ctx.temperature
        # d(z * m)/dz = m + z * dm/dz; dm/dz = sigmoid'·sign(z)/temp.
        sig_p = soft_mask * (1.0 - soft_mask)
        sign_z = torch.sign(z)
        dm_dz = sig_p * sign_z / temp
        grad_z = grad_z_active * (soft_mask.to(z.dtype) + z * dm_dz)
        # d sum_j m_ij / d k_pred_i = -d sum_j sigmoid'·(1/temp) * d tau / d k_pred_i.
        # By the order-statistic envelope, d tau_i / d k_pred_i = sorted_abs[k+1] - sorted_abs[k]
        # which on average equals -1/sum_j sig_p_ij * temp, giving
        # d sum_j m_ij / d k_pred_i ~= 1 (clean Jacobian).
        d_sum_d_k = torch.ones_like(k_pred)
        grad_k_pred = grad_k_pred_sum * d_sum_d_k
        return grad_z, grad_k_pred, None


class AdaptiveTopK(nn.Module):
    """Per-row adaptive top-K sparsity with a learned K-head and REML-selectable lambda.

    A small head ``k_head: z -> R`` predicts a continuous ``k_pred_i in [k_min, k_max]``
    per row. The forward pass keeps the top ``round(k_pred_i)`` activations per row
    via a straight-through estimator: hard top-K mask forward, sigmoid-relaxed soft
    top-K mask backward (see :class:`_AdaptiveTopKSTE`).

    ``penalty()`` returns ``lambda * mean(k_pred)`` where ``lambda = exp(log_weight)``.
    The ``log_weight`` is exposed as a ``nn.Parameter`` so outer-loop REML/LAML can
    select it gradient-style alongside other smoothing parameters. The Rust analytic
    descriptor exported by :meth:`reml_descriptor` is a ``topk_activation`` block at
    ``k = round(mean(k_pred))`` (the closest descriptor presently available; see the
    "REML lambda hook" section below).

    Parameters
    ----------
    F:
        Latent / activation width.
    k_min, k_max:
        Inclusive bounds on the predicted per-row K. Must satisfy
        ``1 <= k_min <= k_max <= F``.
    head:
        ``'mlp'`` (default) for a two-layer ``Linear -> GELU -> Linear`` head with
        ``hidden`` units; ``'linear'`` for a single ``Linear`` layer.
    hidden:
        Hidden width for the MLP head. Ignored when ``head='linear'``.
    temperature:
        Temperature for the soft top-K relaxation used in the STE backward pass.
        Smaller -> sharper (closer to true hard top-K), larger -> smoother backward.
    init_weight:
        Initial value for ``lambda`` (defaults to 1.0).

    Examples
    --------
    >>> import torch
    >>> from gamfit.torch import AdaptiveTopK
    >>> gate = AdaptiveTopK(F=8, k_min=2, k_max=6, head='mlp', hidden=16)
    >>> z_raw = torch.randn(4, 8)
    >>> z_active, k_pred, sparsity = gate(z_raw)
    >>> z_active.shape, k_pred.shape
    (torch.Size([4, 8]), torch.Size([4]))
    >>> bool(sparsity.isfinite())
    True
    >>> bool(gate.penalty().isfinite())
    True
    """

    def __init__(
        self,
        F: int,
        k_min: int,
        k_max: int,
        *,
        head: Literal["mlp", "linear"] = "mlp",
        hidden: int = 64,
        temperature: float = 0.1,
        init_weight: float = 1.0,
        target: str = "t",
    ) -> None:
        super().__init__()
        F_int = int(F)
        if F_int <= 0:
            raise ValueError("AdaptiveTopK.F must be positive")
        k_min_i = int(k_min)
        k_max_i = int(k_max)
        if not (1 <= k_min_i <= k_max_i <= F_int):
            raise ValueError(
                f"AdaptiveTopK requires 1 <= k_min <= k_max <= F; got "
                f"k_min={k_min_i}, k_max={k_max_i}, F={F_int}"
            )
        if head not in {"mlp", "linear"}:
            raise ValueError("AdaptiveTopK.head must be 'mlp' or 'linear'")
        if head == "mlp" and int(hidden) <= 0:
            raise ValueError("AdaptiveTopK.hidden must be > 0 when head='mlp'")
        if not (float(temperature) > 0.0):
            raise ValueError("AdaptiveTopK.temperature must be > 0")
        if not (float(init_weight) > 0.0):
            raise ValueError("AdaptiveTopK.init_weight must be > 0")
        self.F = F_int
        self.k_min = k_min_i
        self.k_max = k_max_i
        self.head_kind = head
        self.hidden = int(hidden) if head == "mlp" else 0
        self.temperature = float(temperature)
        self.target = str(target)
        if head == "mlp":
            self.k_head: nn.Module = nn.Sequential(
                nn.Linear(F_int, int(hidden)),
                nn.GELU(),
                nn.Linear(int(hidden), 1),
            )
        else:
            self.k_head = nn.Linear(F_int, 1)
        # lambda = exp(log_weight); outer-loop REML/LAML can select log_weight.
        self.log_weight = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(float(init_weight)))))
        )
        # Detached scalar for logging / metrics / the REML descriptor only. The
        # penalty path must NOT read this -- a detached value carries no gradient
        # into ``k_head`` (see ``_last_k_pred_mean_graph`` below).
        self.register_buffer(
            "_last_k_pred_mean",
            torch.tensor(float((k_min_i + k_max_i) / 2.0)),
            persistent=False,
        )
        # Graph-connected mean of ``k_pred`` from the most recent forward pass,
        # consumed by ``penalty()`` so the sparsity penalty trains ``k_head``.
        # ``None`` until the first forward; ``penalty()`` then returns a zero
        # tensor anchored on ``log_weight`` so it stays graph-connected.
        self._last_k_pred_mean_graph: torch.Tensor | None = None

    def _predict_k(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.k_head(z).reshape(z.shape[0])
        gated = torch.sigmoid(raw)
        return self.k_min + (self.k_max - self.k_min) * gated

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(z_active, k_pred_eff, sparsity_penalty)``.

        ``z_active`` is the straight-through-gated activation (hard top-K
        forward, soft-mask backward); ``k_pred_eff`` is the per-row soft count
        ``sum_j sigmoid((|z_ij| - tau_i)/temperature)`` whose expectation equals
        the predicted per-row K. ``sparsity_penalty = lambda * mean(k_pred)`` is
        the graph-connected penalty for THIS batch — add it directly to the
        training loss so reconstruction + sparsity train ``k_head`` and
        ``log_weight`` end-to-end on the exact batch just forwarded.

        ``mean(k_pred)`` flows from ``k_head`` (it is *not* detached), so the
        sparsity term backpropagates into the learned-K head; ``z_active``
        flows recon gradient into ``z`` (hence into upstream parameters) through
        the soft top-K surrogate via the STE.
        """
        z = _check_2d_float_tensor(z, "z")
        if z.shape[1] != self.F:
            raise ValueError(f"AdaptiveTopK expected width {self.F}, got {z.shape[1]}")
        k_pred = self._predict_k(z)
        z_active, k_pred_eff = _AdaptiveTopKSTE.apply(z, k_pred, self.temperature)
        # Graph-connected mean for the penalty (trains ``k_head``); detached copy
        # for logging / ``reml_descriptor``.
        k_pred_mean = k_pred.mean()
        self._last_k_pred_mean_graph = k_pred_mean
        with torch.no_grad():
            self._last_k_pred_mean = k_pred_mean.detach()
        sparsity_penalty = torch.exp(self.log_weight) * k_pred_mean
        return z_active, k_pred_eff, sparsity_penalty

    def penalty(self) -> torch.Tensor:
        """Return ``lambda * E[K_pred]`` using the most recent forward pass.

        The mean of ``k_pred`` is kept graph-connected from ``forward`` so that
        ``loss = recon + gate.penalty()`` backpropagates into ``k_head`` (and into
        ``log_weight``). Prefer the ``sparsity_penalty`` returned directly by
        :meth:`forward` for the current batch; this accessor returns the same
        graph-connected quantity for the most recent forward and exists for
        callers that hold the module rather than the forward outputs. Before any
        forward pass it returns a graph-connected zero anchored on ``log_weight``.
        """
        weight = torch.exp(self.log_weight)
        if self._last_k_pred_mean_graph is None:
            return weight * 0.0
        return weight * self._last_k_pred_mean_graph

    def reml_descriptor(self) -> dict[str, object]:
        """Return the gamfit Rust analytic-penalty descriptor for outer-loop REML.

        The descriptor maps the adaptive K-head onto a ``topk_activation`` block
        at ``k = round(mean(k_pred))``. A future Rust-side analytic with
        ``rho_count == 1`` is required for full per-row REML lambda selection;
        until then the outer loop selects ``log_weight`` via gradient flow only.
        """
        k_round = int(round(float(self._last_k_pred_mean.item())))
        k_round = max(self.k_min, min(self.k_max, k_round))
        return {
            "kind": "topk_activation",
            "target": self.target,
            "k": k_round,
            "weight": float(torch.exp(self.log_weight).item()),
        }


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
