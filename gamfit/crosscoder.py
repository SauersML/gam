"""Anthropic-2024 shared-encoder, per-layer-decoder Sparse Autoencoder ("Crosscoder").

A Crosscoder ties a single shared encoder over the *concatenation* of activations
from multiple layers to a *family* of per-layer decoders, one ``Linear(n_atoms, d_l)``
for each layer dimension ``d_l``. Atoms (latent codes) are forced to be shared across
layers by construction, so the per-layer decoder column norms ``||W_dec[l][:, k]||_2``
characterise how strongly atom ``k`` is expressed in layer ``l``. This makes the
Crosscoder a natural tool for **cross-layer interpretability**: tracking how a feature
evolves through a transformer's residual stream.

The "decoder-weighted L1" trick (Anthropic 2024) penalises each activation ``|z_k|``
by the *sum of L2 norms of that atom's decoder columns across layers*::

    L1_loss = sum_n sum_k |z[n, k]| * (sum_l ||W_dec[l][:, k]||_2)

rather than the naive ``sum_n sum_k |z[n, k]|``. This prevents the published SAE
failure mode where the encoder learns to shrink ``z`` while the decoder compensates
with a large ``W_dec``, gaming the L1 penalty without producing genuinely sparse codes.

Honest caveat on REML integration
---------------------------------
The Crosscoder objective is ``reconstruction MSE + lambda * decoder-weighted L1``.
The L1 term is non-quadratic in ``z`` so it does *not* slot into gamfit's analytic
REML / LAML quadratic-penalty pipeline the way smooth-spline penalties do. The
training loop here is a standard Adam optimiser over the joint encoder / decoder
parameters. ``lambda`` is a user-supplied hyper-parameter, not REML-selected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._binding import rust_module


def _parse_shared_encoder(spec: str) -> list[int]:
    """Parse a shared-encoder spec into a list of hidden widths.

    ``'linear'`` returns an empty list. ``'mlp[H1, H2, ...]'`` returns the
    list of hidden widths. Any other spec is an error.
    """
    text = str(spec).strip().lower()
    if text == "linear":
        return []
    match = re.fullmatch(r"mlp\[\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\]", text)
    if match is None:
        raise ValueError(
            f"shared_encoder={spec!r} must be 'linear' or 'mlp[H1, H2, ...]' "
            "with positive integer widths"
        )
    widths = [int(piece) for piece in match.group(1).split(",")]
    if any(w <= 0 for w in widths):
        raise ValueError(
            f"shared_encoder={spec!r} hidden widths must all be positive"
        )
    return widths


@dataclass(slots=True)
class CrosscoderFitDiagnostics:
    """Per-epoch training diagnostics for :class:`Crosscoder`."""

    losses: np.ndarray
    recon_losses: np.ndarray
    l1_losses: np.ndarray


class Crosscoder:
    """Anthropic-2024 shared-encoder, per-layer-decoder cross-layer SAE.

    Parameters
    ----------
    layer_dims : sequence of int
        Per-layer activation widths ``[d_0, d_1, ..., d_{L-1}]``.
    n_atoms : int
        Number of latent atoms (shared across layers).
    decoder_weighted_l1 : bool, default True
        If True, weight each atom's L1 by the sum of L2 norms of its
        per-layer decoder columns (Anthropic 2024). If False, fall back to
        the naive ``mean(|z|)`` sparsity penalty.
    shared_encoder : str, default ``'mlp[1024]'``
        Either ``'linear'`` for a single ``Linear(sum(layer_dims), n_atoms)``
        encoder, or ``'mlp[H1, H2, ...]'`` for an MLP with GELU activations.

    Examples
    --------
    >>> import numpy as np, gamfit
    >>> rng = np.random.default_rng(0)
    >>> X_stack = [rng.normal(size=(64, d)) for d in (16, 32, 16)]
    >>> cc = gamfit.Crosscoder(layer_dims=[16, 32, 16], n_atoms=8,
    ...                        shared_encoder='linear')
    >>> _ = cc.fit(X_stack, epochs=5, lr=1e-2)
    >>> cc.per_layer_r2().shape
    (3,)
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        n_atoms: int,
        *,
        decoder_weighted_l1: bool = True,
        shared_encoder: str = "mlp[1024]",
        l1_weight: float = 1e-3,
    ) -> None:
        dims = [int(d) for d in layer_dims]
        if len(dims) == 0:
            raise ValueError("layer_dims must contain at least one layer")
        if any(d <= 0 for d in dims):
            raise ValueError("each entry of layer_dims must be positive")
        n_atoms_i = int(n_atoms)
        if n_atoms_i <= 0:
            raise ValueError("n_atoms must be positive")
        if float(l1_weight) < 0.0:
            raise ValueError("l1_weight must be non-negative")

        self.layer_dims: list[int] = dims
        self.n_atoms: int = n_atoms_i
        self.decoder_weighted_l1: bool = bool(decoder_weighted_l1)
        self.shared_encoder_spec: str = str(shared_encoder)
        self.l1_weight: float = float(l1_weight)
        self._hidden_widths: list[int] = _parse_shared_encoder(shared_encoder)

        self._module = None  # built lazily on .fit() to keep import torch-free
        self._fitted: bool = False
        self._diagnostics: CrosscoderFitDiagnostics | None = None
        self._per_layer_r2: np.ndarray | None = None

    # ------------------------------------------------------------------ build

    def _build_module(self):
        import torch
        from torch import nn

        widths = self._hidden_widths
        in_dim = int(sum(self.layer_dims))
        out_dim = self.n_atoms

        class _CrosscoderModule(nn.Module):
            def __init__(
                self_inner,
                hidden_widths: list[int],
                encoder_in: int,
                encoder_out: int,
                layer_dims: list[int],
            ) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                prev = encoder_in
                for width in hidden_widths:
                    layers.append(nn.Linear(prev, width))
                    layers.append(nn.GELU())
                    prev = width
                layers.append(nn.Linear(prev, encoder_out))
                # ReLU keeps activations non-negative; standard SAE convention.
                layers.append(nn.ReLU())
                self_inner.encoder = nn.Sequential(*layers)
                self_inner.decoders = nn.ModuleList(
                    [nn.Linear(encoder_out, d, bias=False) for d in layer_dims]
                )

            def forward(
                self_inner, x_concat: "torch.Tensor"
            ) -> tuple["torch.Tensor", list["torch.Tensor"]]:
                z = self_inner.encoder(x_concat)
                recons = [dec(z) for dec in self_inner.decoders]
                return z, recons

        module = _CrosscoderModule(widths, in_dim, out_dim, self.layer_dims)
        return module

    # -------------------------------------------------------------------- fit

    def fit(
        self,
        X_stack: Sequence[np.ndarray],
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int | None = None,
        seed: int = 0,
    ) -> "Crosscoder":
        """Fit the Crosscoder by Adam on the joint encoder + per-layer decoders.

        Parameters
        ----------
        X_stack : list of ``np.ndarray``
            One ``(N, d_l)`` matrix per layer, with the same ``N`` across layers
            (per-row alignment is the cross-layer correspondence).
        epochs : int
            Number of full-batch (or mini-batch) passes.
        lr : float
            Adam learning rate.
        batch_size : int or None
            Mini-batch size. ``None`` uses full-batch training.
        seed : int
            Seed for torch RNG.
        """
        import torch

        if len(X_stack) != len(self.layer_dims):
            raise ValueError(
                f"X_stack has {len(X_stack)} layers but layer_dims has "
                f"{len(self.layer_dims)}"
            )
        arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in X_stack]
        n_rows = arrays[0].shape[0]
        for li, a in enumerate(arrays):
            if a.ndim != 2:
                raise ValueError(
                    f"X_stack[{li}] must be 2-D, got shape {a.shape}"
                )
            if a.shape[0] != n_rows:
                raise ValueError(
                    f"X_stack[{li}] has N={a.shape[0]} but X_stack[0] has "
                    f"N={n_rows}; per-row alignment is required"
                )
            if a.shape[1] != self.layer_dims[li]:
                raise ValueError(
                    f"X_stack[{li}] has width {a.shape[1]} but layer_dims[{li}]"
                    f"={self.layer_dims[li]}"
                )

        # nn.Linear constructors consume the process-global torch RNG. Isolate
        # module construction so fitting a Crosscoder is reproducible without
        # perturbing the caller's stochastic program.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            module = self._build_module()
        module.train()

        tensors = [torch.as_tensor(a, dtype=torch.float32) for a in arrays]
        x_concat = torch.cat(tensors, dim=1)

        optim = torch.optim.Adam(module.parameters(), lr=float(lr))
        n_epochs = int(epochs)
        if n_epochs <= 0:
            raise ValueError("epochs must be positive")
        bs = int(batch_size) if batch_size is not None else n_rows
        if bs <= 0 or bs > n_rows:
            raise ValueError(f"batch_size must satisfy 0 < bs <= N={n_rows}")

        loss_log: list[float] = []
        recon_log: list[float] = []
        l1_log: list[float] = []
        permutation_generator = torch.Generator(device=x_concat.device).manual_seed(
            int(seed)
        )
        for _epoch in range(n_epochs):
            perm = torch.randperm(
                n_rows,
                generator=permutation_generator,
                device=x_concat.device,
            )
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0
            n_batches = 0
            for start in range(0, n_rows, bs):
                idx = perm[start : start + bs]
                xb = x_concat.index_select(0, idx)
                tb = [t.index_select(0, idx) for t in tensors]
                z, recons = module(xb)
                # Reconstruction MSE summed over layers, mean over (rows, dims).
                recon_loss = z.new_zeros(())
                for tl, rl in zip(tb, recons):
                    recon_loss = recon_loss + ((tl - rl) ** 2).mean()
                # Decoder-weighted L1: |z_{n,k}| weighted by sum_l ||W_dec[l][:, k]||_2.
                if self.decoder_weighted_l1:
                    col_norms = z.new_zeros(self.n_atoms)
                    for dec in module.decoders:
                        # dec.weight shape: (d_l, n_atoms); column norm is L2 over dim=0.
                        col_norms = col_norms + dec.weight.pow(2).sum(dim=0).sqrt()
                    l1_loss = (z.abs() * col_norms.unsqueeze(0)).sum(dim=1).mean()
                else:
                    l1_loss = z.abs().mean()
                loss = recon_loss + self.l1_weight * l1_loss
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().item())
                epoch_recon += float(recon_loss.detach().item())
                epoch_l1 += float(l1_loss.detach().item())
                n_batches += 1
            loss_log.append(epoch_loss / max(n_batches, 1))
            recon_log.append(epoch_recon / max(n_batches, 1))
            l1_log.append(epoch_l1 / max(n_batches, 1))

        module.eval()
        with torch.no_grad():
            z_full, recons_full = module(x_concat)
            per_layer = np.empty(len(self.layer_dims), dtype=np.float64)
            for li, (tl, rl) in enumerate(zip(tensors, recons_full)):
                tl_np = tl.detach().cpu().numpy().astype(np.float64)
                rl_np = rl.detach().cpu().numpy().astype(np.float64)
                # Reconstruction R^2 is a numeric kernel owned by the Rust core
                # (column-mean-centered SST, residual SSR, 1 - SSR/SST, with the
                # SST==0 -> NaN and non-finite guards). Route through the FFI
                # rather than re-deriving the formula here.
                per_layer[li] = float(
                    rust_module().sae_manifold_reconstruction_r2(tl_np, rl_np)
                )

        self._module = module
        self._fitted = True
        self._per_layer_r2 = per_layer
        self._diagnostics = CrosscoderFitDiagnostics(
            losses=np.asarray(loss_log, dtype=np.float64),
            recon_losses=np.asarray(recon_log, dtype=np.float64),
            l1_losses=np.asarray(l1_log, dtype=np.float64),
        )
        return self

    # ----------------------------------------------------------- diagnostics

    def _require_fitted(self) -> None:
        if not self._fitted or self._module is None:
            raise RuntimeError("Crosscoder must be fit before calling diagnostics")

    def per_layer_r2(self) -> np.ndarray:
        """Return per-layer reconstruction R^2 evaluated on the training data."""
        self._require_fitted()
        if self._per_layer_r2 is None:
            raise RuntimeError("per-layer R^2 was not computed during fit")
        return self._per_layer_r2.copy()

    def _decoder_column_norms(self) -> np.ndarray:
        """Return the ``(n_atoms, n_layers)`` matrix of decoder column L2 norms."""
        self._require_fitted()
        import torch

        out = np.empty((self.n_atoms, len(self.layer_dims)), dtype=np.float64)
        with torch.no_grad():
            for li, dec in enumerate(self._module.decoders):
                # dec.weight: (d_l, n_atoms).
                norms = dec.weight.pow(2).sum(dim=0).sqrt().detach().cpu().numpy()
                out[:, li] = norms.astype(np.float64)
        return out

    def atom_layer_affinity(self) -> np.ndarray:
        """Return ``(n_atoms, n_layers)`` normalised decoder column-norm affinity.

        Entry ``[k, l]`` is ``||W_dec[l][:, k]||_2`` divided by atom ``k``'s
        maximum across layers (so each row's max is 1.0). Atoms with zero
        decoder norm across every layer are reported as all zeros.
        """
        norms = self._decoder_column_norms()
        row_max = norms.max(axis=1, keepdims=True)
        out = np.zeros_like(norms)
        nonzero = (row_max > 0.0).reshape(-1)
        out[nonzero] = norms[nonzero] / row_max[nonzero]
        return out

    def harmonic_atoms(self, tol: float = 0.05) -> np.ndarray:
        """Return indices of atoms whose per-layer affinity is >= ``tol`` everywhere.

        These are the atoms that are present (non-trivially) in every layer,
        i.e. the cross-layer "shared" features.
        """
        tol_f = float(tol)
        if not (0.0 <= tol_f <= 1.0):
            raise ValueError("tol must lie in [0, 1]")
        affinity = self.atom_layer_affinity()
        mask = (affinity >= tol_f).all(axis=1)
        return np.nonzero(mask)[0].astype(np.int64)

    @property
    def diagnostics(self) -> CrosscoderFitDiagnostics:
        """Per-epoch training loss curves recorded during :meth:`fit`."""
        self._require_fitted()
        if self._diagnostics is None:
            raise RuntimeError("diagnostics were not recorded during fit")
        return self._diagnostics
