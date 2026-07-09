#!/usr/bin/env python3
"""Manifold-zoo arena: the BSF toy protocol, scored with OUR manifold SAE.

The BSF paper ("Structuring Sparsity: Block-Sparse Featurizers Capture Visual
Concept Manifolds", Goodfire 2026) evaluates featurizers on a controlled
additive mixture of manifolds: ``M`` primitive factors (a mix of 1-D concept
segments and curved manifolds -- circles, spheres, tori, Mobius bands, swiss
rolls, helices) embedded in ``R^d`` through random orthonormal maps, summed
``|S| = L0`` at a time. Their block-sparse featurizers reach per-factor
contribution R^2 of 0.93-0.97 against an oracle ceiling of 0.99, while the flat
TopK SAE shatters to ~0.53 (their Fig. "leaderboard").

This bench reproduces that data-generating process EXACTLY (their Appendix
"Toy model of Manifold Superposition", including the per-instance center+RMS
normalization and the clean, noise-free mixtures) and scores:

* ``ours_rust``  -- ``gamfit.sae_manifold_fit`` (the production Rust REML path)
  at its DEFAULTS. Atoms are genuinely curved charts, so a circle costs ONE
  intrinsic coordinate where a block-sparse featurizer spends a 2-D block, and
  the recovered coordinate IS the factor's intrinsic parameter.
* ``ours_torch`` -- ``gamfit.torch.ManifoldSAE`` trained with Adam (the GPU
  lane for paper-scale N).
* ``flat``       -- the TopK SAE contrast (``bench/synth_sae_compare.TopKSAE``),
  granted the paper's generous reading: each factor is matched to its
  ``span_dim`` best-fitting latents (greedy forward selection on actual
  contribution R^2), i.e. exactly the ambient dimension the factor needs.
* ``oracle``     -- per-row least squares on the TRUE active subspaces (the
  recovery ceiling, their protocol).

Scoring follows their protocol: each factor is matched to the single atom whose
firing best predicts the factor's presence (point-biserial correlation of
|gate| against the active mask over held-out rows), and the matched atom ALONE
reconstructs the factor's per-row contribution; we report per-factor R^2 on the
rows where the factor is active. On top we report the two currencies the paper
argues from:

* Description length (their Eq. 4, uniform across featurizers): support bits
  ``log2 C(G, L0)`` + water-filled code bits from the per-atom active-code
  covariance spectra + water-filled residual bits + the dictionary amortized
  over N tokens -- read at distortion floors ``1 - R2 in {.01,.05,.10,.20}``.
  For ``ours_rust`` the fit's NATIVE bits/token (the repo's headline currency)
  is reported alongside.
* Active coordinates per token: ours = sum of (intrinsic dim + amplitude) over
  active atoms; flat = k. The "a circle is 1 coordinate" headline.
* Per-atom stable rank of contributions (their concept-dimensionality read).

Results append as JSONL (a crash loses nothing); ``--dump-clouds`` writes an
NPZ with per-factor true/recovered contribution clouds + intrinsic-coordinate
hue for the figure gallery (``bench/bsf_zoo_figures.py``).

Paper-matched invocation (their toy: M=128, d=128, L0=4, N=3e5):

    python3 bsf_manifold_zoo.py --factors 128 --ambient 128 --l0 4 \
        --n-train 300000 --n-test 100000 --featurizers ours_torch,flat,oracle

Local smoke:

    python3 bsf_manifold_zoo.py --factors 12 --ambient 48 --l0 3 \
        --n-train 6000 --n-test 3000 --atoms 12 --featurizers ours_rust,flat,oracle
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gamfit._description_length import FittedFeaturizer, description_length

# --------------------------------------------------------------------------- #
# The manifold zoo (their Appendix table, verbatim parametrizations)          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ZooType:
    name: str
    intrinsic_dim: int
    span_dim: int
    sampler: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]
    """(rng, n) -> (points (n, span_dim), intrinsic coords (n, intrinsic_dim))."""


def _segment(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    t = rng.uniform(-1.0, 1.0, size=n)
    return t[:, None], t[:, None]


def _circle(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack([np.cos(th), np.sin(th)]), th[:, None]


def _disk(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    return np.column_stack([r * np.cos(th), r * np.sin(th)]), np.column_stack([r, th])


def _sphere(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi = np.arccos(rng.uniform(-1.0, 1.0, size=n))  # uniform on the sphere
    pts = np.column_stack(
        [np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)]
    )
    return pts, np.column_stack([phi, th])


def _torus(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    big_r, small_r = 1.0, 0.4
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    ring = big_r + small_r * np.cos(phi)
    pts = np.column_stack([ring * np.cos(th), ring * np.sin(th), small_r * np.sin(phi)])
    return pts, np.column_stack([th, phi])


def _mobius(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    t = rng.uniform(-0.5, 0.5, size=n)
    a = 1.0 + t * np.cos(phi / 2.0)
    pts = np.column_stack([a * np.cos(phi), a * np.sin(phi), t * np.sin(phi / 2.0)])
    return pts, np.column_stack([phi, t])


def _swiss(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    th = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    h = rng.uniform(-1.0, 1.0, size=n)
    pts = np.column_stack([th * np.cos(th), h, th * np.sin(th)])
    return pts, np.column_stack([th, h])


def _helix(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    th = rng.uniform(0.0, 4.0 * np.pi, size=n)
    pts = np.column_stack([np.cos(th), np.sin(th), 0.25 * th])
    return pts, th[:, None]


ZOO: dict[str, ZooType] = {
    z.name: z
    for z in (
        ZooType("segment", 1, 1, _segment),
        ZooType("circle", 1, 2, _circle),
        ZooType("disk", 2, 2, _disk),
        ZooType("sphere", 2, 3, _sphere),
        ZooType("torus", 2, 3, _torus),
        ZooType("mobius", 2, 3, _mobius),
        ZooType("swiss", 2, 3, _swiss),
        ZooType("helix", 1, 3, _helix),
    )
}
CURVED_CYCLE = ["circle", "disk", "sphere", "torus", "mobius", "swiss", "helix"]
_CALIBRATION_N = 50_000  # their per-instance center+RMS calibration sample size


@dataclass(frozen=True)
class FactorInstance:
    kind: str
    frame: np.ndarray  # (span_dim, d) orthonormal rows
    mu: np.ndarray  # (span_dim,) calibration mean
    sigma: float  # calibration RMS norm

    def draw(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(contribution in R^d (n, d), intrinsic coords (n, intrinsic_dim))."""
        raw, theta = ZOO[self.kind].sampler(rng, n)
        local = (raw - self.mu[None, :]) / self.sigma
        return local @ self.frame, theta


class ZooData:
    """Additive mixture of manifolds: their toy DGP, exactly."""

    def __init__(self, m_factors: int, d_ambient: int, l0: int, seed: int, *,
                 curved_fraction: float = 0.5) -> None:
        self.m_factors = int(m_factors)
        self.d_ambient = int(d_ambient)
        self.l0 = int(l0)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.factors: list[FactorInstance] = []
        n_curved = int(round(curved_fraction * m_factors))
        kinds = ["segment"] * (m_factors - n_curved) + [
            CURVED_CYCLE[i % len(CURVED_CYCLE)] for i in range(n_curved)
        ]
        rng.shuffle(kinds)
        for kind in kinds:
            zoo = ZOO[kind]
            raw, _ = zoo.sampler(rng, _CALIBRATION_N)
            mu = raw.mean(axis=0)
            centered = raw - mu[None, :]
            sigma = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
            gauss = rng.standard_normal((d_ambient, zoo.span_dim))
            q, _ = np.linalg.qr(gauss)
            self.factors.append(
                FactorInstance(kind=kind, frame=np.ascontiguousarray(q.T), mu=mu,
                               sigma=max(sigma, 1e-12))
            )
        self.kinds = kinds

    def sample(
        self, n: int, seed: int, *, keep_contributions: bool
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]] | None]:
        """Draw ``n`` clean mixtures.

        Returns ``(x (n, d), active (n, M) bool, contribs)`` where ``contribs``
        (kept only for eval splits -- it is the ground truth being recovered)
        is a per-factor list of ``{"rows": idx, "m": contribution, "theta": t}``
        restricted to the rows where that factor is active.
        """
        rng = np.random.default_rng(seed)
        x = np.zeros((n, self.d_ambient))
        active = np.zeros((n, self.m_factors), dtype=bool)
        # Uniform-without-replacement active sets, vectorized via argsort.
        order = np.argsort(rng.random((n, self.m_factors)), axis=1)[:, : self.l0]
        rows = np.repeat(np.arange(n), self.l0)
        active[rows, order.ravel()] = True
        contribs: list[dict[str, np.ndarray]] | None = [] if keep_contributions else None
        for i, factor in enumerate(self.factors):
            idx = np.flatnonzero(active[:, i])
            if idx.size == 0:
                if contribs is not None:
                    contribs.append({"rows": idx, "m": np.zeros((0, self.d_ambient)),
                                     "theta": np.zeros((0, 1))})
                continue
            m_i, theta = factor.draw(rng, idx.size)
            x[idx] += m_i
            if contribs is not None:
                contribs.append({"rows": idx, "m": m_i, "theta": theta})
        return np.ascontiguousarray(x), active, contribs


# --------------------------------------------------------------------------- #
# Featurizer adapters: everything scoring needs from one fitted featurizer    #
# --------------------------------------------------------------------------- #


def _fit_oracle(data: ZooData, test_x: np.ndarray, active: np.ndarray) -> FittedFeaturizer:
    """Least squares on the TRUE active frames -- the recovery ceiling."""
    n, d = test_x.shape
    contrib_hat = np.zeros((data.m_factors, n, d), dtype=np.float32)
    recon = np.zeros_like(test_x)
    t0 = time.perf_counter()
    # Group rows by active set to solve each unique stacked-frame LS once.
    keys = np.packbits(active, axis=1).tobytes()
    row_key = [keys[i * ((data.m_factors + 7) // 8):(i + 1) * ((data.m_factors + 7) // 8)]
               for i in range(n)]
    by_key: dict[bytes, list[int]] = {}
    for i, key in enumerate(row_key):
        by_key.setdefault(key, []).append(i)
    for rows in by_key.values():
        rows_arr = np.asarray(rows)
        acts = np.flatnonzero(active[rows_arr[0]])
        frames = [data.factors[j].frame for j in acts]
        stack = np.vstack(frames)  # (sum b_j, d)
        gram = stack @ stack.T
        coef = np.linalg.solve(gram, stack @ test_x[rows_arr].T).T  # (rows, sum b)
        start = 0
        for j, frame in zip(acts, frames):
            b = frame.shape[0]
            part = coef[:, start:start + b] @ frame
            contrib_hat[j, rows_arr] = part
            recon[rows_arr] += part
            start += b
    gate = active.astype(float)
    return FittedFeaturizer(
        name="oracle",
        gate=gate,
        atom_contribution=lambda g: contrib_hat[g].astype(float),
        code_dims=np.array([data.factors[j].frame.shape[0] for j in range(data.m_factors)]),
        dictionary_params=sum(f.frame.size for f in data.factors),
        recon=recon,
        fit_seconds=time.perf_counter() - t0,
    )


def _fit_flat_topk(
    train_x: np.ndarray, test_x: np.ndarray, *, width: int, k: int, steps: int,
    batch_size: int, lr: float, seed: int, device: str,
) -> FittedFeaturizer:
    """The TopK SAE contrast (bench/synth_sae_compare.TopKSAE recipe)."""
    import torch

    from synth_sae_compare import TopKSAE

    torch.manual_seed(seed)
    model = TopKSAE(train_x.shape[1], width, k, seed).to(device)
    x_all = torch.as_tensor(np.asarray(train_x, dtype=np.float32), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(steps):
        idx = rng.integers(0, x_all.shape[0], size=min(batch_size, x_all.shape[0]))
        batch = x_all[idx]
        recon, _z = model(batch)
        loss = torch.mean((recon - batch) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    fit_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(test_x, dtype=np.float32), device=device)
        codes = []
        recons = []
        for start in range(0, xt.shape[0], 16384):
            chunk = xt[start:start + 16384]
            recon_chunk, z_chunk = model(chunk)
            codes.append(z_chunk.cpu().numpy())
            recons.append(recon_chunk.cpu().numpy())
        z = np.concatenate(codes, axis=0)
        recon = np.concatenate(recons, axis=0)
        dec = model.decoder.weight.detach().cpu().numpy().T  # (width, d)
        dec_bias = model.decoder.bias.detach().cpu().numpy()
    return FittedFeaturizer(
        name="flat_topk",
        gate=np.abs(z),
        atom_contribution=lambda g: np.outer(z[:, g], dec[g]),
        code_dims=np.ones(dec.shape[0], dtype=int),
        dictionary_params=dec.size,
        recon=recon,
        fit_seconds=fit_seconds,
        extras={"decoder": dec, "decoder_bias": dec_bias, "codes": z},
    )


def _fit_ours_rust(
    train_x: np.ndarray, test_x: np.ndarray, *, atoms: int, top_k: int | None,
    n_iter: int, seed: int,
) -> FittedFeaturizer:
    """The production Rust REML path at its DEFAULTS (magic by default)."""
    import gamfit

    from synth_sae_bench_manifold import _basis_values

    t0 = time.perf_counter()
    fit = gamfit.sae_manifold_fit(
        X=train_x,
        n_atoms=atoms,
        top_k=top_k,
        n_iter=n_iter,
        random_state=seed,
    )
    fit_seconds = time.perf_counter() - t0

    def _r2_of(x: np.ndarray, recon: np.ndarray) -> float:
        ss_res = float(np.sum((x - recon) ** 2))
        ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    # Three-way state-vs-encoder discriminator:
    #   native   -- fit.fitted, the terminal state's own train reconstruction;
    #   re-enc(train) -- converged_latents on the SAME train rows (re-encode);
    #   re-enc(test)  -- the held-out encode the arena scores.
    # native >> re-enc(train) localizes the loss to the OOS ENCODE path itself
    # (row re-encoding, not data novelty); re-enc(train) >> re-enc(test) is a
    # genuine generalization gap.
    native_train_r2 = _r2_of(train_x, np.asarray(fit.fitted, dtype=float))
    # COLD arm must run the GENUINE frozen-decoder OOS solve on the training
    # rows. `converged_latents(train_x)` short-circuits on a bit-exact training
    # matrix (`_is_training_data`) and returns the STORED training latents —
    # which would make this arm vacuously equal to `native` and mislocalize the
    # OOS collapse. Call the underlying OOS payload directly so the cold
    # seed/routing path actually executes.
    train_payload = fit._oos_payload(train_x, t_init=None, a_init=None)
    reenc_train_r2 = _r2_of(train_x, np.asarray(train_payload["fitted"], dtype=float))
    # Warm re-encode: seed the SAME frozen-decoder OOS solve with the training
    # fit's own converged coords/logits. Recovering the native number here
    # proves the solver is sound and localizes the cold-path loss to
    # seeding/routing; staying collapsed indicts the solve itself.
    d_max = max(int(np.asarray(c).shape[1]) for c in fit.coords)
    t_init = np.zeros((len(fit.coords), train_x.shape[0], d_max))
    for k, c in enumerate(fit.coords):
        c = np.asarray(c, dtype=float)
        t_init[k, :, : c.shape[1]] = c
    warm_payload = fit.converged_latents(
        train_x, t_init=t_init, a_init=np.asarray(fit.low_level_logits, dtype=float)
    )
    warm_train_r2 = _r2_of(train_x, np.asarray(warm_payload["fitted"], dtype=float))
    print(
        f"[ours_rust] native train r2 {native_train_r2:.4f} | "
        f"re-encode(train) COLD r2 {reenc_train_r2:.4f} | "
        f"re-encode(train) WARM r2 {warm_train_r2:.4f}",
        flush=True,
    )
    payload = fit.converged_latents(test_x)
    assignments = np.asarray(payload["assignments"], dtype=float)
    recon = np.asarray(payload["fitted"], dtype=float)
    coords = [np.asarray(c, dtype=float) for c in payload["coords"]]
    blocks = [np.asarray(b, dtype=float) for b in fit.decoder_blocks]

    def contribution(g: int) -> np.ndarray:
        basis = fit.basis_specs[g]
        n_harm = fit._n_harmonics[g] if g < len(fit._n_harmonics) else 1
        centers = fit._duchon_centers[g] if g < len(fit._duchon_centers) else None
        phi = _basis_values(basis, coords[g], n_harm, centers)
        rows = min(phi.shape[1], blocks[g].shape[0])
        return assignments[:, g:g + 1] * (phi[:, :rows] @ blocks[g][:rows])

    d_atoms = [np.asarray(c).shape[1] if np.asarray(c).ndim == 2 else 1 for c in coords]
    native_bpt = None
    dl = getattr(fit, "description_length", None)
    if isinstance(dl, dict) and "bits_per_token" in dl:
        native_bpt = float(dl["bits_per_token"])
    return FittedFeaturizer(
        name="ours_rust",
        gate=np.abs(assignments),
        atom_contribution=contribution,
        code_dims=np.asarray([d + 1 for d in d_atoms]),  # coords + amplitude
        dictionary_params=int(sum(b.size for b in blocks)),
        recon=recon,
        fit_seconds=fit_seconds,
        native_bits_per_token=native_bpt,
        atom_intrinsic_coords=lambda g: coords[g],
        extras={"basis_specs": list(fit.basis_specs)},
    )


def _fit_ours_torch(
    train_x: np.ndarray, test_x: np.ndarray, *, atoms: int, top_k: int, steps: int,
    batch_size: int, lr: float, seed: int, device: str, intrinsic_rank: int,
    atom_manifold: str, atom_basis: str,
) -> FittedFeaturizer:
    """The torch lane (GPU scale): gamfit.torch.ManifoldSAE + Adam."""
    import torch

    from gamfit.torch.manifold_sae import ManifoldSAE, ManifoldSAEConfig

    torch.manual_seed(seed)
    cfg = ManifoldSAEConfig(
        input_dim=train_x.shape[1],
        n_atoms=atoms,
        intrinsic_rank=intrinsic_rank,
        atom_manifold=atom_manifold,
        atom_basis=atom_basis,
        sparsity={"kind": "softmax_topk", "target_k": top_k},
        # fp32 for the GPU TRAINING lane: the library default is float64
        # (f64-first everywhere else), but B200-class GPUs have token FP64
        # throughput — the fp64 default measured 0.07 steps/s on the paper
        # config (~24 h/run) with the SAME math available ~40x faster in
        # fp32. The FFI basis kernels already compute their jets in f64
        # internally and cast back, and all downstream RD/EV metrics are
        # f64 numpy on the exported arrays, so lane accuracy is unaffected.
        dtype=torch.float32,
    )
    model = ManifoldSAE(cfg).to(device)
    x_all = torch.as_tensor(np.asarray(train_x, dtype=np.float32), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    model.train()
    last_log = t0
    for step in range(steps):
        idx = rng.integers(0, x_all.shape[0], size=min(batch_size, x_all.shape[0]))
        batch = x_all[idx]
        out = model(batch)
        # Reconstruction + sparsity, plus the encoder position-alignment penalty:
        # under the Rust E-step coordinate path the reconstruction is built at
        # solved coordinates that are tape constants, so the encoder position head
        # gets no reconstruction gradient. The alignment penalty (mean squared
        # circular distance between encoder-predicted and E-step-solved
        # coordinates) keeps that head learning. It is a zero scalar for lanes
        # that do not run the E-step (non-circle / non-softmax_topk).
        loss = (
            torch.mean((out.x_hat - batch) ** 2)
            + model.sparsity.penalty(out.gate)
            + model.position_alignment_penalty()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        now = time.perf_counter()
        # Time-based heartbeat (plus the periodic/final step): a slow lane must
        # never be log-silent — the fp64 paper lane sat 7 h with one line
        # because step 201 (the next `% 200` hit) was ~48 min away at best, so
        # "stalled" and "alive but slow" were indistinguishable from the log.
        if step % 200 == 0 or step == steps - 1 or now - last_log >= 30.0:
            rate = (step + 1) / max(now - t0, 1e-9)
            print(
                f"[ours_torch] step {step + 1}/{steps} loss={float(loss.detach()):.5f} "
                f"({rate:.2f} steps/s)",
                flush=True,
            )
            last_log = now
    fit_seconds = time.perf_counter() - t0
    model.eval()
    zs, curves_all, recons, positions = [], [], [], []
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(test_x, dtype=np.float32), device=device)
        for start in range(0, xt.shape[0], 8192):
            out = model(xt[start:start + 8192])
            zs.append(out.z.cpu().numpy())
            curves_all.append(out.curves.cpu().numpy())
            recons.append(out.x_hat.cpu().numpy())
            positions.append(out.positions.cpu().numpy())
    # Metrics stay f64: lift the fp32 lane exports once, here, so every
    # downstream EV/bits accumulation runs in float64 numpy as before.
    z = np.concatenate(zs, axis=0).astype(np.float64)
    curves = np.concatenate(curves_all, axis=0).astype(np.float64)  # (N, F, n_basis)
    recon = np.concatenate(recons, axis=0).astype(np.float64)
    pos = np.concatenate(positions, axis=0).astype(np.float64)  # (N, F, rank_embed)
    dec = model.decoder_blocks.detach().cpu().numpy().astype(np.float64)  # (F, n_basis, d)

    def contribution(g: int) -> np.ndarray:
        return z[:, g:g + 1] * (curves[:, g, :] @ dec[g])

    return FittedFeaturizer(
        name="ours_torch",
        gate=np.abs(z),
        atom_contribution=contribution,
        code_dims=np.full(atoms, intrinsic_rank + 1),
        dictionary_params=int(dec.size),
        recon=recon,
        fit_seconds=fit_seconds,
        atom_intrinsic_coords=lambda g: pos[:, g, :],
        extras={"positions_shape": list(pos.shape)},
    )


# --------------------------------------------------------------------------- #
# Their scoring protocol                                                      #
# --------------------------------------------------------------------------- #


def _point_biserial(gate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Correlation of each gate column with a boolean mask (vectorized)."""
    g = gate - gate.mean(axis=0, keepdims=True)
    m = mask.astype(float) - float(mask.mean())
    num = g.T @ m
    den = np.sqrt(np.sum(g * g, axis=0) * float(np.sum(m * m)))
    return num / np.maximum(den, 1e-12)


def _contribution_r2(m_true: np.ndarray, m_hat: np.ndarray) -> float:
    center = m_true.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum((m_true - center) ** 2))
    ss_res = float(np.sum((m_true - m_hat) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _greedy_flat_match(
    fitted: FittedFeaturizer, rows: np.ndarray, m_true: np.ndarray, span_dim: int,
) -> tuple[list[int], np.ndarray]:
    """The paper's generous flat reading: span_dim best-fitting latents, greedy."""
    codes = fitted.extras["codes"][rows]  # type: ignore[index]
    dec = fitted.extras["decoder"]  # type: ignore[index]
    chosen: list[int] = []
    m_hat = np.zeros_like(m_true)
    resid = m_true.copy()
    fired = np.flatnonzero(np.any(np.abs(codes) > 0, axis=0))
    for _ in range(span_dim):
        best_j, best_gain = -1, -np.inf
        for j in fired:
            if j in chosen:
                continue
            cand = np.outer(codes[:, j], dec[j])
            gain = -float(np.sum((resid - cand) ** 2))
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j < 0:
            break
        chosen.append(int(best_j))
        m_hat += np.outer(codes[:, best_j], dec[best_j])
        resid = m_true - m_hat
    return chosen, m_hat


def score_recovery(
    data: ZooData,
    fitted: FittedFeaturizer,
    active: np.ndarray,
    contribs: list[dict[str, np.ndarray]],
    *,
    flat_multi_atom: bool,
) -> dict[str, Any]:
    """Per-factor contribution R^2 under their block<->factor matching."""
    per_factor: list[dict[str, Any]] = []
    corr = np.stack(
        [_point_biserial(fitted.gate, active[:, i]) for i in range(data.m_factors)]
    )  # (M, G)
    matched_atoms = np.argmax(np.abs(corr), axis=1)
    contrib_cache: dict[int, np.ndarray] = {}
    for i, factor in enumerate(data.factors):
        rows = contribs[i]["rows"]
        if rows.size < 8:
            continue
        m_true = contribs[i]["m"]
        if flat_multi_atom:
            chosen, m_hat = _greedy_flat_match(
                fitted, rows, m_true, ZOO[factor.kind].span_dim
            )
            atom_label: Any = chosen
            active_coords = len(chosen)
        else:
            g = int(matched_atoms[i])
            if g not in contrib_cache:
                contrib_cache[g] = fitted.atom_contribution(g)
            m_hat = contrib_cache[g][rows]
            atom_label = g
            active_coords = int(fitted.code_dims[g])
        per_factor.append(
            {
                "factor": i,
                "kind": factor.kind,
                "intrinsic_dim": ZOO[factor.kind].intrinsic_dim,
                "span_dim": ZOO[factor.kind].span_dim,
                "matched_atom": atom_label,
                "r2": _contribution_r2(m_true, m_hat),
                "match_corr": float(np.max(np.abs(corr[i]))),
                "coords_per_activation": active_coords,
                "n_rows": int(rows.size),
            }
        )
    r2s = np.asarray([p["r2"] for p in per_factor])
    by_kind: dict[str, float] = {}
    for kind in sorted({p["kind"] for p in per_factor}):
        by_kind[kind] = float(np.mean([p["r2"] for p in per_factor if p["kind"] == kind]))
    return {
        "per_factor": per_factor,
        "r2_mean": float(np.mean(r2s)) if r2s.size else float("nan"),
        "r2_by_kind": by_kind,
        "coords_per_activation_mean": float(
            np.mean([p["coords_per_activation"] for p in per_factor])
        ) if per_factor else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Description length (their Eq. 4, uniform estimator) + dimensionality        #
# --------------------------------------------------------------------------- #


def stable_ranks(fitted: FittedFeaturizer) -> dict[str, Any]:
    gate_active = fitted.gate > 1e-10
    ranks: list[float] = []
    for g in range(fitted.gate.shape[1]):
        rows = np.flatnonzero(gate_active[:, g])
        if rows.size < 8:
            continue
        take = rows if rows.size <= 2048 else rows[:: max(rows.size // 2048, 1)]
        m_g = fitted.atom_contribution(g)[take]
        m_g = m_g - m_g.mean(axis=0, keepdims=True)
        sing = np.linalg.svd(m_g, compute_uv=False)
        if sing.size == 0 or sing[0] <= 1e-12:
            continue
        ranks.append(float(np.sum(sing**2) / (sing[0] ** 2)))
    return {
        "stable_rank_mean": float(np.mean(ranks)) if ranks else float("nan"),
        "stable_rank_p90": float(np.percentile(ranks, 90)) if ranks else float("nan"),
        "n_live_atoms": len(ranks),
    }


def _test_r2(fitted: FittedFeaturizer, test_x: np.ndarray) -> float:
    ss_res = float(np.sum((test_x - fitted.recon) ** 2))
    ss_tot = float(np.sum((test_x - test_x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# --------------------------------------------------------------------------- #
# Cloud dumps for the figure gallery                                          #
# --------------------------------------------------------------------------- #


def dump_clouds(
    path: Path,
    data: ZooData,
    fitted: FittedFeaturizer,
    contribs: list[dict[str, np.ndarray]],
    recovery: dict[str, Any],
    *,
    max_factors: int = 8,
    max_points: int = 1500,
) -> None:
    # Deterministic by factor index so every featurizer dumps the SAME factors
    # and the gallery columns are directly comparable.
    curved = [p for p in recovery["per_factor"] if p["kind"] != "segment"]
    curved.sort(key=lambda p: p["factor"])
    seen_kinds: dict[str, int] = {}
    picks = []
    for p in curved:  # first instance of each curved kind, then wrap around
        if seen_kinds.get(p["kind"], 0) == 0:
            picks.append(p)
            seen_kinds[p["kind"]] = 1
    picks = (picks + [p for p in curved if p not in picks])[:max_factors]
    payload: dict[str, np.ndarray] = {}
    meta: list[dict[str, Any]] = []
    for p in picks:
        i = p["factor"]
        rows = contribs[i]["rows"]
        take = np.arange(rows.size)
        if rows.size > max_points:
            take = np.random.default_rng(0).choice(rows.size, size=max_points, replace=False)
        m_true = contribs[i]["m"][take]
        if isinstance(p["matched_atom"], list):
            _, m_hat_all = _greedy_flat_match(
                fitted, rows, contribs[i]["m"], ZOO[p["kind"]].span_dim
            )
            m_hat = m_hat_all[take]
        else:
            m_hat = fitted.atom_contribution(int(p["matched_atom"]))[rows][take]
        # Project both into the TRUE factor's principal frame (their rendering).
        center = m_true.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(m_true - center, full_matrices=False)
        frame = vt[:3]
        payload[f"true_{i}"] = ((m_true - center) @ frame.T).astype(np.float32)
        payload[f"rec_{i}"] = ((m_hat - center) @ frame.T).astype(np.float32)
        payload[f"theta_{i}"] = contribs[i]["theta"][take].astype(np.float32)
        if fitted.atom_intrinsic_coords is not None and not isinstance(p["matched_atom"], list):
            coords = fitted.atom_intrinsic_coords(int(p["matched_atom"]))
            payload[f"learned_theta_{i}"] = np.asarray(coords)[rows][take].astype(np.float32)
        meta.append({"factor": i, "kind": p["kind"], "r2": p["r2"]})
    payload["meta_json"] = np.frombuffer(
        json.dumps({"featurizer": fitted.name, "factors": meta}).encode(), dtype=np.uint8
    )
    np.savez_compressed(path, **payload)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--ambient", type=int, default=128)
    parser.add_argument("--l0", type=int, default=4)
    parser.add_argument("--curved-fraction", type=float, default=0.5)
    parser.add_argument("--n-train", type=int, default=300_000)
    parser.add_argument("--n-test", type=int, default=100_000)
    parser.add_argument("--atoms", type=int, default=None,
                        help="Atom count for our SAE (default: M factors).")
    parser.add_argument("--flat-width", type=int, default=None,
                        help="Flat dictionary width (default: 2 * sum of span dims).")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Active atoms for our SAE (default: L0).")
    parser.add_argument("--flat-k", type=int, default=None,
                        help="Active latents for the flat SAE (default: L0 * mean span).")
    parser.add_argument("--rust-iters", type=int, default=40)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--intrinsic-rank", type=int, default=2)
    parser.add_argument("--atom-manifold", default="product")
    parser.add_argument("--atom-basis", default="fourier")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--featurizers", default="ours_rust,flat,oracle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="bsf_zoo_results.jsonl")
    parser.add_argument("--dump-clouds", default=None,
                        help="Directory for per-featurizer NPZ cloud dumps.")
    args = parser.parse_args()

    data = ZooData(args.factors, args.ambient, args.l0, args.seed,
                   curved_fraction=args.curved_fraction)
    train_x, _train_active, _ = data.sample(args.n_train, args.seed + 1,
                                            keep_contributions=False)
    test_x, test_active, contribs = data.sample(args.n_test, args.seed + 2,
                                                keep_contributions=True)
    assert contribs is not None
    span_sum = sum(ZOO[k].span_dim for k in data.kinds)
    atoms = args.atoms or args.factors
    top_k = args.top_k or args.l0
    flat_width = args.flat_width or 2 * span_sum
    flat_k = args.flat_k or max(args.l0 * int(round(span_sum / args.factors)), args.l0)
    out_path = Path(args.out)
    clouds_dir = Path(args.dump_clouds) if args.dump_clouds else None
    if clouds_dir is not None:
        clouds_dir.mkdir(parents=True, exist_ok=True)

    header = {
        "record": "config",
        "factors": args.factors, "ambient": args.ambient, "l0": args.l0,
        "curved_fraction": args.curved_fraction, "kinds": data.kinds,
        "n_train": args.n_train, "n_test": args.n_test,
        "atoms": atoms, "top_k": top_k, "flat_width": flat_width, "flat_k": flat_k,
        "seed": args.seed, "span_sum": span_sum,
    }
    with out_path.open("a") as fh:
        fh.write(json.dumps(header) + "\n")

    wanted = [w.strip() for w in args.featurizers.split(",") if w.strip()]
    for which in wanted:
        if which == "oracle":
            fitted = _fit_oracle(data, test_x, test_active)
        elif which == "flat":
            fitted = _fit_flat_topk(
                train_x, test_x, width=flat_width, k=flat_k, steps=args.steps,
                batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                device=args.device,
            )
        elif which == "ours_rust":
            fitted = _fit_ours_rust(
                train_x, test_x, atoms=atoms, top_k=top_k,
                n_iter=args.rust_iters, seed=args.seed,
            )
        elif which == "ours_torch":
            fitted = _fit_ours_torch(
                train_x, test_x, atoms=atoms, top_k=top_k, steps=args.steps,
                batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                device=args.device, intrinsic_rank=args.intrinsic_rank,
                atom_manifold=args.atom_manifold, atom_basis=args.atom_basis,
            )
        else:
            raise SystemExit(f"unknown featurizer {which!r}")
        recovery = score_recovery(
            data, fitted, test_active, contribs,
            flat_multi_atom=(which == "flat"),
        )
        record = {
            "record": "result",
            "featurizer": fitted.name,
            "seed": args.seed,
            "test_r2": _test_r2(fitted, test_x),
            "fit_seconds": fitted.fit_seconds,
            "recovery_r2_mean": recovery["r2_mean"],
            "recovery_r2_by_kind": recovery["r2_by_kind"],
            "coords_per_activation_mean": recovery["coords_per_activation_mean"],
            "mdl": description_length(fitted, test_x),
            "dimensionality": stable_ranks(fitted),
            "per_factor": recovery["per_factor"],
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"[{fitted.name}] recovery R2 {recovery['r2_mean']:.4f} "
            f"(test R2 {record['test_r2']:.4f}, "
            f"coords/act {recovery['coords_per_activation_mean']:.2f}, "
            f"fit {fitted.fit_seconds:.1f}s)"
        )
        for kind, val in recovery["r2_by_kind"].items():
            print(f"    {kind:>8s}: {val:.4f}")
        if clouds_dir is not None:
            dump_clouds(
                clouds_dir / f"clouds_{fitted.name}_seed{args.seed}.npz",
                data, fitted, contribs, recovery,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
