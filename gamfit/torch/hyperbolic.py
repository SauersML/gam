"""Hyperbolic atom dictionary in the Poincaré ball.

Thin torch wrapper over the Rust geometry primitives in
``gam::geometry::poincare``. All math (Möbius addition, log/exp at the
origin, tangent-space decoder forward and analytic backward, Lorentz
model arithmetic, and numerical safeguards) lives in Rust; this module
only owns the ``nn.Parameter`` storage and routes tensors across the
Python/numpy boundary into the Rust kernel via a ``torch.autograd.Function``.

The Rust kernel is also reachable directly from Rust or the CLI — see
``gam::geometry::poincare`` — so this feature is not torch-specific.

Decoder convention
------------------
``forward(z)`` decodes via tangent-space aggregation at the origin
followed by the ball exponential map::

    v       = sum_f z_f * log_0(a_f)              (in T_0 B = R^d)
    x_hat   = exp_0(v)

Closed-form and fully differentiable; reduces to ``z @ atoms`` in the
Euclidean (``c -> 0``) limit; equivariant under Möbius isometries
fixing the origin. The analytic Jacobian is implemented in Rust
(``poincare_tangent_decode_backward``) and surfaced through
:class:`_PoincareTangentDecode` below.

References
----------
* Nickel, Kiela. *Poincaré Embeddings for Learning Hierarchical
  Representations.* NeurIPS 2017. arXiv:1705.08039.
* Ganea, Bécigneul, Hofmann. *Hyperbolic Neural Networks.* NeurIPS 2018.
  arXiv:1805.09112.
* Hyperbolic-Mamba, arXiv:2505.18973 (2025).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .._binding import rust_module


def _np_f64(t: torch.Tensor) -> np.ndarray:
    """Detach to a contiguous numpy float64 array for the Rust bridge."""

    return np.ascontiguousarray(t.detach().cpu().to(torch.float64).numpy())


def _from_np(values: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    """Promote a numpy array back to a torch tensor matching ``ref``."""

    return torch.from_numpy(np.ascontiguousarray(values)).to(
        dtype=ref.dtype, device=ref.device
    )


class _PoincareTangentDecode(torch.autograd.Function):
    """Autograd shim around the Rust tangent-space-at-origin decoder.

    Forward calls ``poincare_tangent_decode_forward`` in Rust and stashes
    the (atoms_projected, gates, v, tangents) state needed by the analytic
    backward; backward calls ``poincare_tangent_decode_backward`` to obtain
    gradients for both atoms and gates.
    """

    @staticmethod
    def forward(
        ctx,
        atoms: torch.Tensor,
        gates: torch.Tensor,
        curvature: float,
    ) -> torch.Tensor:
        rust = rust_module()
        atoms_np = _np_f64(atoms)
        gates_np = _np_f64(gates)
        (
            x_hat_np,
            atoms_proj_np,
            v_np,
            tangents_np,
            proj_scale_np,
        ) = rust.poincare_tangent_decode_forward(atoms_np, gates_np, float(curvature))
        ctx.curvature = float(curvature)
        ctx.save_for_backward(
            torch.from_numpy(np.ascontiguousarray(atoms_proj_np)),
            torch.from_numpy(np.ascontiguousarray(gates_np)),
            torch.from_numpy(np.ascontiguousarray(v_np)),
            torch.from_numpy(np.ascontiguousarray(tangents_np)),
            torch.from_numpy(np.ascontiguousarray(proj_scale_np)),
        )
        ctx.atoms_dtype = atoms.dtype
        ctx.gates_dtype = gates.dtype
        ctx.atoms_device = atoms.device
        ctx.gates_device = gates.device
        return _from_np(x_hat_np, atoms)

    @staticmethod
    def backward(ctx, grad_x_hat: torch.Tensor):
        rust = rust_module()
        atoms_p, gates, v, tangents, proj_scale = ctx.saved_tensors
        grad_np = _np_f64(grad_x_hat)
        grad_gates_np, grad_atoms_np = rust.poincare_tangent_decode_backward(
            np.ascontiguousarray(atoms_p.numpy()),
            np.ascontiguousarray(gates.numpy()),
            np.ascontiguousarray(v.numpy()),
            np.ascontiguousarray(tangents.numpy()),
            np.ascontiguousarray(proj_scale.numpy()),
            grad_np,
            ctx.curvature,
        )
        grad_atoms = torch.from_numpy(np.ascontiguousarray(grad_atoms_np)).to(
            dtype=ctx.atoms_dtype, device=ctx.atoms_device
        )
        grad_gates = torch.from_numpy(np.ascontiguousarray(grad_gates_np)).to(
            dtype=ctx.gates_dtype, device=ctx.gates_device
        )
        return grad_atoms, grad_gates, None


class _PoincareLorentzDecode(torch.autograd.Function):
    """Autograd shim around the Rust Lorentz-path decoder.

    Both forward (``poincare_lorentz_decode_forward``) and backward
    (``poincare_lorentz_decode_backward``) live in Rust. The Lorentz and
    Poincaré tangent-space-at-origin decoders satisfy the algebraic
    identity ``y_Lorentz(z; A) == y_Poincare(z; A)`` exactly (see the
    derivation in ``src/geometry/poincare.rs``: the factor-of-two
    discrepancies in the Lorentz log/exp at the origin cancel inside the
    linear mix), so the analytic Jacobian of the Lorentz forward is the
    same closed form derived for the Poincaré path. A dedicated Rust unit
    test finite-differences the *Lorentz* forward against this backward to
    pin the agreement at 1e-5.
    """

    @staticmethod
    def forward(
        ctx,
        atoms: torch.Tensor,
        gates: torch.Tensor,
        curvature: float,
    ) -> torch.Tensor:
        rust = rust_module()
        atoms_np = _np_f64(atoms)
        gates_np = _np_f64(gates)
        x_hat_np = rust.poincare_lorentz_decode_forward(atoms_np, gates_np, float(curvature))
        # Cache Poincaré forward state for the analytic backward.
        (
            _,
            atoms_proj_np,
            v_np,
            tangents_np,
            proj_scale_np,
        ) = rust.poincare_tangent_decode_forward(atoms_np, gates_np, float(curvature))
        ctx.curvature = float(curvature)
        ctx.save_for_backward(
            torch.from_numpy(np.ascontiguousarray(atoms_proj_np)),
            torch.from_numpy(np.ascontiguousarray(gates_np)),
            torch.from_numpy(np.ascontiguousarray(v_np)),
            torch.from_numpy(np.ascontiguousarray(tangents_np)),
            torch.from_numpy(np.ascontiguousarray(proj_scale_np)),
        )
        ctx.atoms_dtype = atoms.dtype
        ctx.gates_dtype = gates.dtype
        ctx.atoms_device = atoms.device
        ctx.gates_device = gates.device
        return _from_np(x_hat_np, atoms)

    @staticmethod
    def backward(ctx, grad_x_hat: torch.Tensor):
        rust = rust_module()
        atoms_p, gates, v, tangents, proj_scale = ctx.saved_tensors
        grad_np = _np_f64(grad_x_hat)
        grad_gates_np, grad_atoms_np = rust.poincare_lorentz_decode_backward(
            np.ascontiguousarray(atoms_p.numpy()),
            np.ascontiguousarray(gates.numpy()),
            np.ascontiguousarray(v.numpy()),
            np.ascontiguousarray(tangents.numpy()),
            np.ascontiguousarray(proj_scale.numpy()),
            grad_np,
            ctx.curvature,
        )
        grad_atoms = torch.from_numpy(np.ascontiguousarray(grad_atoms_np)).to(
            dtype=ctx.atoms_dtype, device=ctx.atoms_device
        )
        grad_gates = torch.from_numpy(np.ascontiguousarray(grad_gates_np)).to(
            dtype=ctx.gates_dtype, device=ctx.gates_device
        )
        return grad_atoms, grad_gates, None


class PoincareAtoms(nn.Module):
    """A learnable dictionary of ``F`` atoms in the Poincaré ball ``B^d_c``.

    Parameters
    ----------
    F:
        Number of atoms.
    ball_dim:
        Ambient dimension ``d`` of the ball.
    curvature:
        Sectional curvature ``c``, must be strictly negative.
    lorentz:
        If ``True`` use the Rust Lorentz-model forward (boundary-safe).
        Backward gradients are routed through the (isometric) Poincaré
        analytic backward — see :class:`_PoincareLorentzDecode`.
    init_scale:
        Standard deviation of the Gaussian initialiser for atom positions.
        Atoms are projected into the ball after sampling via the Rust
        projection primitive.
    device, dtype:
        Standard torch placement arguments.

    Examples
    --------
    >>> import torch
    >>> from gamfit.torch.hyperbolic import PoincareAtoms
    >>> atoms = PoincareAtoms(F=4, ball_dim=3)
    >>> z = torch.randn(2, 4)
    >>> x_hat = atoms(z)
    >>> x_hat.shape
    torch.Size([2, 3])
    """

    def __init__(
        self,
        F: int,
        ball_dim: int,
        curvature: float = -1.0,
        *,
        lorentz: bool = False,
        init_scale: float = 0.01,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(F, int) or F <= 0:
            raise ValueError(f"F must be a positive int, got {F!r}")
        if not isinstance(ball_dim, int) or ball_dim <= 0:
            raise ValueError(f"ball_dim must be a positive int, got {ball_dim!r}")
        if not isinstance(curvature, (int, float)):
            raise TypeError("curvature must be a real number")
        curvature = float(curvature)
        if not math.isfinite(curvature) or curvature >= 0.0:
            raise ValueError(
                "PoincareAtoms requires negative curvature (c < 0); "
                f"got curvature={curvature!r}"
            )
        if init_scale <= 0.0:
            raise ValueError("init_scale must be > 0")
        if dtype is None:
            dtype = torch.get_default_dtype()

        self.F = F
        self.ball_dim = ball_dim
        self.curvature = curvature
        self.lorentz = bool(lorentz)
        self.init_scale = init_scale

        atoms_init = torch.randn(F, ball_dim, device=device, dtype=dtype) * init_scale
        rust = rust_module()
        projected_rows = [
            rust.poincare_project_into_ball(np.ascontiguousarray(row), curvature)
            for row in atoms_init.detach().cpu().to(torch.float64).numpy()
        ]
        atoms_init = torch.from_numpy(
            np.ascontiguousarray(np.stack(projected_rows, axis=0))
        ).to(dtype=dtype, device=device)
        self.atoms = nn.Parameter(atoms_init)

    # ------------------------------------------------------------------ utils

    def project_into_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` strictly inside the ball via the Rust primitive."""

        rust = rust_module()
        if x.dim() == 1:
            out = rust.poincare_project_into_ball(_np_f64(x), self.curvature)
            return _from_np(out, x)
        flat = x.reshape(-1, x.shape[-1]).detach().cpu().to(torch.float64).numpy()
        rows = [
            rust.poincare_project_into_ball(np.ascontiguousarray(r), self.curvature)
            for r in flat
        ]
        stacked = np.stack(rows, axis=0).reshape(x.shape)
        return _from_np(stacked, x)

    def mobius_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Möbius addition ``u ⊕_c v`` via the Rust primitive (row-wise)."""

        if u.shape != v.shape:
            raise ValueError(
                f"mobius_add inputs must share shape; got {tuple(u.shape)} and {tuple(v.shape)}"
            )
        rust = rust_module()
        if u.dim() == 1:
            out = rust.poincare_mobius_add(_np_f64(u), _np_f64(v), self.curvature)
            return _from_np(out, u)
        u_flat = u.reshape(-1, u.shape[-1]).detach().cpu().to(torch.float64).numpy()
        v_flat = v.reshape(-1, v.shape[-1]).detach().cpu().to(torch.float64).numpy()
        rows = [
            rust.poincare_mobius_add(
                np.ascontiguousarray(u_flat[i]),
                np.ascontiguousarray(v_flat[i]),
                self.curvature,
            )
            for i in range(u_flat.shape[0])
        ]
        stacked = np.stack(rows, axis=0).reshape(u.shape)
        return _from_np(stacked, u)

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geodesic distance ``d_c(a, b)`` via the Rust primitive (row-wise)."""

        if a.shape != b.shape:
            raise ValueError(
                f"distance inputs must share shape; got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        if a.shape[-1] != self.ball_dim:
            raise ValueError(
                f"distance expects last dim = {self.ball_dim}; got {tuple(a.shape)}"
            )
        rust = rust_module()
        if a.dim() == 1:
            d_scalar = rust.poincare_distance(_np_f64(a), _np_f64(b), self.curvature)
            return torch.as_tensor(d_scalar, dtype=a.dtype, device=a.device)
        a_flat = a.reshape(-1, a.shape[-1]).detach().cpu().to(torch.float64).numpy()
        b_flat = b.reshape(-1, b.shape[-1]).detach().cpu().to(torch.float64).numpy()
        out = np.empty(a_flat.shape[0], dtype=np.float64)
        for i in range(a_flat.shape[0]):
            out[i] = rust.poincare_distance(
                np.ascontiguousarray(a_flat[i]),
                np.ascontiguousarray(b_flat[i]),
                self.curvature,
            )
        out = out.reshape(a.shape[:-1])
        return torch.from_numpy(out).to(dtype=a.dtype, device=a.device)

    # ------------------------------------------------------------------ forward

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode gate vector ``z`` (shape ``(..., F)``) to a ball point."""

        if z.shape[-1] != self.F:
            raise ValueError(
                f"PoincareAtoms.forward expects z.shape[-1] = {self.F}; "
                f"got {tuple(z.shape)}"
            )
        lead_shape = z.shape[:-1]
        if not lead_shape:
            z2 = z.unsqueeze(0)
            collapsed = True
        else:
            z2 = z.reshape(-1, self.F)
            collapsed = False

        op = _PoincareLorentzDecode if self.lorentz else _PoincareTangentDecode
        x2 = op.apply(self.atoms, z2, self.curvature)
        if collapsed:
            return x2.squeeze(0)
        return x2.reshape(*lead_shape, self.ball_dim)


__all__ = ["PoincareAtoms"]
