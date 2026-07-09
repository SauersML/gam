"""Post-hoc distilled encoders for fitted :class:`gamfit.ManifoldSAE` models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise ModuleNotFoundError(
            "ManifoldSAE.distill_encoder requires torch. Install a torch build "
            "compatible with this Python environment."
        ) from exc
    return torch


def _as_2d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D numeric array; got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _pad_coords(coords: Sequence[np.ndarray], atom_dims: Sequence[int]) -> np.ndarray:
    if len(coords) != len(atom_dims):
        raise ValueError(
            f"expected {len(atom_dims)} coordinate blocks, got {len(coords)}"
        )
    max_dim = max(int(dim) for dim in atom_dims)
    n_rows = int(np.asarray(coords[0]).shape[0])
    out = np.zeros((n_rows, len(atom_dims), max_dim), dtype=np.float64)
    for atom, (block, dim) in enumerate(zip(coords, atom_dims)):
        arr = np.asarray(block, dtype=np.float64)
        if arr.shape != (n_rows, int(dim)):
            raise ValueError(
                f"coords[{atom}] must have shape {(n_rows, int(dim))}; got {arr.shape}"
            )
        out[:, atom, : int(dim)] = arr
    return out


def _flatten_targets(coords_nkd: np.ndarray, logits: np.ndarray) -> np.ndarray:
    n_rows = coords_nkd.shape[0]
    return np.concatenate(
        [coords_nkd.reshape(n_rows, -1), np.asarray(logits, dtype=np.float64)],
        axis=1,
    )


def _split_targets(
    values: np.ndarray,
    *,
    k_atoms: int,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    coord_width = int(k_atoms) * int(max_dim)
    coords = values[:, :coord_width].reshape(values.shape[0], int(k_atoms), int(max_dim))
    logits = values[:, coord_width : coord_width + int(k_atoms)]
    return coords, logits


def _scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = values.mean(axis=0)
    spread = values.std(axis=0)
    spread = np.where(spread > 1.0e-12, spread, 1.0)
    return center.astype(np.float64), spread.astype(np.float64)


def _activation_from_logits(
    logits: np.ndarray,
    *,
    assignment: str,
    tau: float,
    alpha: float,
    jumprelu_threshold: float,
) -> np.ndarray:
    tau = float(tau)
    if tau <= 0.0 or not np.isfinite(tau):
        raise ValueError(f"tau must be finite and positive; got {tau}")
    name = str(assignment)
    z = np.asarray(logits, dtype=np.float64)
    if name == "softmax":
        shifted = (z - np.max(z, axis=1, keepdims=True)) / tau
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)
    if name == "ibp_map":
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"alpha must be finite and positive; got {alpha}")
        k_atoms = z.shape[1]
        ratio = float(alpha) / (float(alpha) + 1.0)
        prior = np.exp(
            np.arange(1, k_atoms + 1, dtype=np.float64) * np.log(ratio)
        )
        prior = np.maximum(prior, np.finfo(np.float64).tiny)
        sig = 1.0 / (1.0 + np.exp(-np.clip(z / tau, -709.0, 709.0)))
        return sig * prior.reshape(1, -1)
    # #1777 — the hard-sigmoid gate's primary token is "threshold_gate"; the
    # legacy "jumprelu" spelling is still accepted as a deprecated alias.
    if name in ("threshold_gate", "jumprelu"):
        shifted = (z - float(jumprelu_threshold)) / tau
        sig = 1.0 / (1.0 + np.exp(-np.clip(shifted, -709.0, 709.0)))
        return np.where(z > float(jumprelu_threshold), sig, 0.0)
    raise ValueError(f"unsupported assignment kind {assignment!r}")


@dataclass(slots=True)
class EncoderFallbackStats:
    """Honest accounting for one encoder-gated encode call."""

    rows: int
    accepted_rows: int
    fallback_rows: int
    fallback_rate: float
    assignment_linf_mean: float
    assignment_linf_max: float
    coord_linf_mean: float
    coord_linf_max: float
    assignment_tolerance: float
    coord_tolerance: float
    exact_probe_rows: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rows": self.rows,
            "accepted_rows": self.accepted_rows,
            "fallback_rows": self.fallback_rows,
            "fallback_rate": self.fallback_rate,
            "assignment_linf_mean": self.assignment_linf_mean,
            "assignment_linf_max": self.assignment_linf_max,
            "coord_linf_mean": self.coord_linf_mean,
            "coord_linf_max": self.coord_linf_max,
            "assignment_tolerance": self.assignment_tolerance,
            "coord_tolerance": self.coord_tolerance,
            "exact_probe_rows": self.exact_probe_rows,
        }


@dataclass(slots=True)
class DistilledEncoder:
    """Small MLP distilled from exact frozen-decoder SAE solves."""

    module: Any
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    atom_dims: tuple[int, ...]
    assignment: str
    tau: float
    alpha: float
    jumprelu_threshold: float
    assignment_tolerance: float
    coord_tolerance: float
    training_history: dict[str, list[float] | float | int]
    last_stats: EncoderFallbackStats | None = None

    @property
    def k_atoms(self) -> int:
        return len(self.atom_dims)

    @property
    def max_dim(self) -> int:
        return max(self.atom_dims)

    @property
    def input_dim(self) -> int:
        return int(self.x_mean.shape[0])

    def predict_initializers(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        torch = _torch()
        x = _as_2d_float(X, "X")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"encoder expected X with {self.input_dim} columns; got {x.shape[1]}"
            )
        x_std = (x - self.x_mean.reshape(1, -1)) / self.x_scale.reshape(1, -1)
        device = next(self.module.parameters()).device
        dtype = next(self.module.parameters()).dtype
        with torch.no_grad():
            tensor = torch.as_tensor(x_std, dtype=dtype, device=device)
            pred_std = self.module(tensor).detach().cpu().numpy()
        pred = pred_std * self.y_scale.reshape(1, -1) + self.y_mean.reshape(1, -1)
        coords_nkd, logits = _split_targets(
            pred,
            k_atoms=self.k_atoms,
            max_dim=self.max_dim,
        )
        coords_knm = np.transpose(coords_nkd, (1, 0, 2))
        return np.ascontiguousarray(coords_knm), np.ascontiguousarray(logits)

    def encode_fast(self, X: Any) -> np.ndarray:
        _, logits = self.predict_initializers(X)
        return _activation_from_logits(
            logits,
            assignment=self.assignment,
            tau=self.tau,
            alpha=self.alpha,
            jumprelu_threshold=self.jumprelu_threshold,
        )


def _build_module(
    *,
    input_dim: int,
    output_dim: int,
    hidden: int | Sequence[int],
    dtype: Any,
) -> Any:
    torch = _torch()
    widths = [int(hidden)] if isinstance(hidden, int) else [int(v) for v in hidden]
    if any(width <= 0 for width in widths):
        raise ValueError(f"hidden widths must be positive; got {widths}")
    layers: list[Any] = []
    prev = int(input_dim)
    for width in widths:
        layers.append(torch.nn.Linear(prev, width, dtype=dtype))
        layers.append(torch.nn.GELU())
        prev = width
    layers.append(torch.nn.Linear(prev, int(output_dim), dtype=dtype))
    return torch.nn.Sequential(*layers)


def distill_encoder(
    model: Any,
    X: Any,
    *,
    hidden: int | Sequence[int] = (64, 64),
    epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    tolerance_multiplier: float = 1.25,
) -> DistilledEncoder:
    """Train a post-hoc MLP encoder from exact SAE OOS teacher solves."""

    torch = _torch()
    x = _as_2d_float(X, "X")
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1; got {epochs}")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if not (0.0 <= float(validation_fraction) < 1.0):
        raise ValueError(
            f"validation_fraction must be in [0, 1); got {validation_fraction}"
        )
    exact = model.converged_latents(x)
    atom_dims = tuple(int(dim) for dim in model._atom_dims)
    coords_nkd = _pad_coords(exact["coords"], atom_dims)
    logits = np.asarray(exact["logits"], dtype=np.float64)
    if logits.shape != (x.shape[0], len(atom_dims)):
        raise ValueError(
            f"teacher logits must have shape {(x.shape[0], len(atom_dims))}; got {logits.shape}"
        )
    y = _flatten_targets(coords_nkd, logits)
    x_mean, x_scale = _scale(x)
    y_mean, y_scale = _scale(y)
    x_std = (x - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)
    y_std = (y - y_mean.reshape(1, -1)) / y_scale.reshape(1, -1)

    rng = np.random.default_rng(int(random_state))
    order = rng.permutation(x.shape[0])
    val_count = int(round(float(validation_fraction) * x.shape[0]))
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    if train_idx.size == 0:
        raise ValueError("distill_encoder needs at least one training row")

    torch.manual_seed(int(random_state))
    dtype = torch.float64
    module = _build_module(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden=hidden,
        dtype=dtype,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=float(learning_rate))
    x_tensor = torch.as_tensor(x_std, dtype=dtype)
    y_tensor = torch.as_tensor(y_std, dtype=dtype)
    train_losses: list[float] = []
    val_losses: list[float] = []
    for _epoch in range(int(epochs)):
        module.train()
        epoch_order = train_idx[rng.permutation(train_idx.size)]
        total = 0.0
        seen = 0
        for start in range(0, epoch_order.size, int(batch_size)):
            batch_idx = epoch_order[start : start + int(batch_size)]
            xb = x_tensor[batch_idx]
            yb = y_tensor[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            pred = module(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(batch_idx.size)
            seen += int(batch_idx.size)
        train_losses.append(total / max(seen, 1))
        if val_idx.size:
            module.eval()
            with torch.no_grad():
                val_pred = module(x_tensor[val_idx])
                val_loss = torch.mean((val_pred - y_tensor[val_idx]) ** 2)
            val_losses.append(float(val_loss.detach()))

    module.eval()
    with torch.no_grad():
        pred_std = module(x_tensor).detach().cpu().numpy()
    pred = pred_std * y_scale.reshape(1, -1) + y_mean.reshape(1, -1)
    pred_coords, pred_logits = _split_targets(
        pred,
        k_atoms=len(atom_dims),
        max_dim=max(atom_dims),
    )
    pred_assign = _activation_from_logits(
        pred_logits,
        assignment=str(model.assignment),
        tau=float(model.tau),
        alpha=float(model.alpha),
        jumprelu_threshold=float(model.jumprelu_threshold),
    )
    exact_assign = np.asarray(exact["assignments"], dtype=np.float64)
    coord_err = np.max(np.abs(pred_coords - coords_nkd), axis=(1, 2))
    assign_err = np.max(np.abs(pred_assign - exact_assign), axis=1)
    calibration_idx = val_idx if val_idx.size else train_idx
    coord_tol = float(np.max(coord_err[calibration_idx]) * float(tolerance_multiplier))
    assign_tol = float(np.max(assign_err[calibration_idx]) * float(tolerance_multiplier))
    coord_tol = max(coord_tol, 1.0e-10)
    assign_tol = max(assign_tol, 1.0e-10)
    history: dict[str, list[float] | float | int] = {
        "train_loss": train_losses,
        "validation_loss": val_losses,
        "assignment_calibration_linf": float(np.max(assign_err[calibration_idx])),
        "coord_calibration_linf": float(np.max(coord_err[calibration_idx])),
        "training_rows": int(train_idx.size),
        "validation_rows": int(val_idx.size),
    }
    return DistilledEncoder(
        module=module,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        atom_dims=atom_dims,
        assignment=str(model.assignment),
        tau=float(model.tau),
        alpha=float(model.alpha),
        jumprelu_threshold=float(model.jumprelu_threshold),
        assignment_tolerance=assign_tol,
        coord_tolerance=coord_tol,
        training_history=history,
    )


def encode_with_fallback(
    model: Any,
    X: Any,
    encoder: DistilledEncoder,
) -> tuple[np.ndarray, EncoderFallbackStats]:
    """Run an encoder-gated encode and fall back rowwise to exact solves."""

    x = _as_2d_float(X, "X")
    t_init, logits_init = encoder.predict_initializers(x)
    fast_assign = _activation_from_logits(
        logits_init,
        assignment=encoder.assignment,
        tau=encoder.tau,
        alpha=encoder.alpha,
        jumprelu_threshold=encoder.jumprelu_threshold,
    )
    # #1166 — the acceptance gate MUST be cold-started. The "exact" reference
    # probe is solved with NO `t_init`/`a_init`, so it is the canonical
    # feature map that the public `encode(X)` returns and that
    # `distill_encoder` calibrated the tolerances against (`converged_latents`,
    # itself a cold `_oos_payload`). A warm-started probe seeded from the
    # encoder's own guess (`t_init=t_init, a_init=logits_init`) would bias the
    # finite-iteration Newton refinement toward that guess, so `assign_err` /
    # `coord_err` would be measured against a moving reference and systematically
    # under-estimated — the self-referential gate of #1166. Cold-starting keeps
    # the gate measuring the encoder against the same fixed solve everywhere.
    exact_payload = model._oos_payload(x)
    exact_assign = np.asarray(exact_payload["assignments_z"], dtype=np.float64)
    exact_coords = _pad_coords(
        [np.asarray(atom["on_atom_coords_t"], dtype=np.float64) for atom in exact_payload["atoms"]],
        encoder.atom_dims,
    )
    pred_coords_nkd = np.transpose(t_init, (1, 0, 2))
    assign_err = np.max(np.abs(fast_assign - exact_assign), axis=1)
    coord_err = np.max(np.abs(pred_coords_nkd - exact_coords), axis=(1, 2))
    accepted = (assign_err <= encoder.assignment_tolerance) & (
        coord_err <= encoder.coord_tolerance
    )
    encoded = exact_assign.copy()
    encoded[accepted] = fast_assign[accepted]
    fallback_rows = int(np.count_nonzero(~accepted))
    rows = int(x.shape[0])
    stats = EncoderFallbackStats(
        rows=rows,
        accepted_rows=int(np.count_nonzero(accepted)),
        fallback_rows=fallback_rows,
        fallback_rate=float(fallback_rows / rows),
        assignment_linf_mean=float(np.mean(assign_err)),
        assignment_linf_max=float(np.max(assign_err)),
        coord_linf_mean=float(np.mean(coord_err)),
        coord_linf_max=float(np.max(coord_err)),
        assignment_tolerance=float(encoder.assignment_tolerance),
        coord_tolerance=float(encoder.coord_tolerance),
        exact_probe_rows=rows,
    )
    encoder.last_stats = stats
    return encoded, stats


__all__ = [
    "DistilledEncoder",
    "EncoderFallbackStats",
    "distill_encoder",
    "encode_with_fallback",
]
