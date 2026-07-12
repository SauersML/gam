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

This bench uses exact analytic defining equations for all eight native objects,
including a full-height 2-D Swiss-roll sheet, then applies the paper's
per-instance center/RMS normalization and clean additive-superposition protocol.
It scores:

* ``ours_rust``  -- ``gamfit.sae_manifold_fit`` (the production Rust REML path)
  at its DEFAULTS. Atoms are genuinely curved charts, so a circle costs ONE
  intrinsic coordinate where a block-sparse featurizer spends a 2-D block, and
  the recovered coordinate IS the factor's intrinsic parameter.
* ``flat``       -- the TopK SAE contrast (``bench/synth_sae_compare.TopKSAE``),
  granted the paper's generous reading: each factor is matched to its
  ``span_dim`` best-fitting latents (greedy forward selection on actual
  contribution R^2), i.e. exactly the ambient dimension the factor needs.
* ``oracle``     -- per-row least squares on the TRUE active subspaces (the
  recovery ceiling, their protocol).

Scoring follows their protocol: factors and atoms receive the exact optimal
one-to-one Hungarian match by point-biserial correlation of |gate| against the
held-out active masks. The matched atom ALONE reconstructs the factor's per-row
contribution; we report per-factor R^2 on the rows where the factor is active.
This prevents a collapsed dictionary from claiming several planted factors
with the same atom. On top we report the two currencies the paper argues from:

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

Large joint invocation (M=128, d=128, L0=4, N=3e5):

    python3 -m bench.bsf_manifold_zoo --factors 128 --ambient 128 --l0 4 \
        --n-train 300000 --n-test 100000 --featurizers ours_rust,flat,oracle

Local smoke:

    python3 -m bench.bsf_manifold_zoo --factors 12 --ambient 48 --l0 3 \
        --n-train 6000 --n-test 3000 --atoms 12 --featurizers ours_rust,flat,oracle
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench._synth_sae_metrics import _hungarian_max_numpy
from bench.manifold_zoo_geometry import (
    CURVED_CYCLE,
    GEOMETRY_REVISION,
    ZOO,
    validate_analytic_sample,
)
from gamfit._description_length import FittedFeaturizer, description_length

_CALIBRATION_N = 50_000  # their per-instance center+RMS calibration sample size


@dataclass(frozen=True)
class FactorInstance:
    kind: str
    frame: np.ndarray  # (span_dim, d) orthonormal rows
    mu: np.ndarray  # (span_dim,) calibration mean
    sigma: float  # calibration RMS norm

    def draw(
        self, rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ambient contribution, intrinsic parameters, and raw native points."""
        raw, theta = ZOO[self.kind].sampler(rng, n)
        validate_analytic_sample(self.kind, raw, theta)
        local = (raw - self.mu[None, :]) / self.sigma
        return local @ self.frame, theta, raw


class ZooData:
    """Additive mixture of manifolds.

    ``dgp="toy"`` is their toy DGP exactly: every factor unit-importance,
    uniform-without-replacement active sets of fixed size ``l0``, unit
    amplitudes, no noise floor.

    ``dgp="llm"`` reshapes the SAME manifold mixture into the statistics real
    LLM residual-stream features show:

    * **Zipfian firing** — feature ``i`` fires independently with probability
      ``∝ (i+1)^-zipf`` (mean row support ``l0``), so rows have variable L0
      with a few frequent features and a long tail.
    * **Context-correlated co-occurrence** — each row draws one of a few
      latent contexts; a context doubles the firing rate of its feature
      subset (topics make features co-fire; marginal rates preserved).
    * **Power-law importance** — feature ``i`` contributes at scale
      ``∝ (i+1)^-importance`` (a few dominant directions, a long tail).
    * **Heavy-tailed amplitudes** — every firing scales its contribution by a
      mean-one lognormal, so features have continuous strength, not 0/1.
    * **Unstructured noise floor** — dense isotropic Gaussian residual the
      dictionary should NOT explain (the unexplained-variance floor every
      real SAE faces).

    The curved minority (circles / helices — the day-of-week and numeric
    features found in real models) is controlled by ``curved_fraction`` as
    before; realistic runs want it low (~0.25) with the rest segments.
    """

    def __init__(self, m_factors: int, d_ambient: int, l0: int, seed: int, *,
                 curved_fraction: float = 0.5, kinds: list[str] | None = None,
                 dgp: str = "toy",
                 zipf: float = 1.0, importance: float = 0.5,
                 amp_sigma: float = 0.6, noise: float = 0.25,
                 n_contexts: int = 4) -> None:
        self.m_factors = int(m_factors)
        self.d_ambient = int(d_ambient)
        self.l0 = int(l0)
        self.seed = int(seed)
        if dgp not in ("toy", "llm"):
            raise ValueError(f"dgp must be 'toy' or 'llm'; got {dgp!r}")
        self.dgp = dgp
        self.noise = float(noise)
        self.amp_sigma = float(amp_sigma)
        self.n_contexts = int(n_contexts)
        rng = np.random.default_rng(seed)
        self.factors: list[FactorInstance] = []
        if kinds is None:
            n_curved = int(round(curved_fraction * m_factors))
            kinds = ["segment"] * (m_factors - n_curved) + [
                CURVED_CYCLE[i % len(CURVED_CYCLE)] for i in range(n_curved)
            ]
            rng.shuffle(kinds)
        else:
            kinds = [str(kind).strip().lower() for kind in kinds]
            if len(kinds) != m_factors:
                raise ValueError(
                    f"explicit kinds has {len(kinds)} entries but m_factors={m_factors}"
                )
            unknown = sorted(set(kinds) - ZOO.keys())
            if unknown:
                raise ValueError(
                    f"unknown zoo kinds {unknown}; expected a subset of {sorted(ZOO)}"
                )
        for kind in kinds:
            zoo = ZOO[kind]
            if d_ambient < zoo.span_dim:
                raise ValueError(
                    f"ambient dimension {d_ambient} cannot isometrically embed "
                    f"{kind} with native span {zoo.span_dim}"
                )
            raw, calibration_parameters = zoo.sampler(rng, _CALIBRATION_N)
            validate_analytic_sample(kind, raw, calibration_parameters)
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
        if dgp == "llm":
            ranks = np.arange(1, self.m_factors + 1, dtype=float)
            probs = ranks ** -float(zipf)
            self.base_probs = np.minimum(probs * (self.l0 / probs.sum()), 0.95)
            weights = ranks ** -float(importance)
            self.importance = weights / weights.mean()
            # Each context DOUBLES the rate of a random third of the features
            # and scales the rest so every feature's marginal rate (averaged
            # over contexts) is exactly its Zipf base rate.
            boost, frac = 2.0, 1.0 / 3.0
            off_gain = (1.0 - frac * boost) / (1.0 - frac)
            self.context_gain = np.full((self.n_contexts, self.m_factors), off_gain)
            for ctx in range(self.n_contexts):
                subset = rng.choice(self.m_factors,
                                    size=max(1, int(round(frac * self.m_factors))),
                                    replace=False)
                self.context_gain[ctx, subset] = boost

    def sample(
        self, n: int, seed: int, *, keep_contributions: bool
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]] | None]:
        """Draw ``n`` clean mixtures.

        Returns ``(x (n, d), active (n, M) bool, contribs)`` where ``contribs``
        (kept only for eval splits -- it is the ground truth being recovered)
        is a per-factor list of ambient contribution, intrinsic parameters, and
        exact native-span coordinates, restricted to active rows.
        """
        rng = np.random.default_rng(seed)
        x = np.zeros((n, self.d_ambient))
        active = np.zeros((n, self.m_factors), dtype=bool)
        if self.dgp == "llm":
            # Zipfian per-feature Bernoulli firing, correlated through a
            # per-row latent context; empty rows fire their most likely
            # feature so every row carries signal.
            ctx = rng.integers(0, self.n_contexts, size=n)
            probs = np.clip(self.base_probs[None, :] * self.context_gain[ctx], 0.0, 0.95)
            active = rng.random((n, self.m_factors)) < probs
            empty = ~active.any(axis=1)
            if empty.any():
                active[np.flatnonzero(empty), np.argmax(probs[empty], axis=1)] = True
        else:
            # Uniform-without-replacement active sets. Partial selection avoids
            # sorting all M random keys when only L0 << M entries are retained.
            order = np.argpartition(
                rng.random((n, self.m_factors)), self.l0 - 1, axis=1
            )[:, : self.l0]
            rows = np.repeat(np.arange(n), self.l0)
            active[rows, order.ravel()] = True
        contribs: list[dict[str, np.ndarray]] | None = [] if keep_contributions else None
        for i, factor in enumerate(self.factors):
            idx = np.flatnonzero(active[:, i])
            if idx.size == 0:
                if contribs is not None:
                    contribs.append(
                        {
                            "rows": idx,
                            "m": np.zeros((0, self.d_ambient)),
                            "theta": np.zeros((0, ZOO[factor.kind].intrinsic_dim)),
                            "native": np.zeros((0, ZOO[factor.kind].span_dim)),
                        }
                    )
                continue
            m_i, theta, native = factor.draw(rng, idx.size)
            if self.dgp == "llm":
                # Mean-one lognormal amplitude per firing, scaled by the
                # feature's power-law importance; the ground-truth
                # contribution INCLUDES both (that is what recovery targets).
                amps = self.importance[i] * rng.lognormal(
                    mean=-0.5 * self.amp_sigma ** 2, sigma=self.amp_sigma,
                    size=idx.size,
                )
                m_i = amps[:, None] * m_i
            x[idx] += m_i
            if contribs is not None:
                contribs.append(
                    {
                        "rows": idx,
                        "m": m_i,
                        "theta": theta,
                        "native": native,
                    }
                )
        if self.dgp == "llm" and self.noise > 0.0:
            # Dense unstructured residual — deliberately NOT part of any
            # factor's ground truth; it is the unexplained-variance floor.
            x += self.noise * rng.standard_normal(x.shape)
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

    from bench.synth_sae_compare import TopKSAE

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

    fit_start = time.perf_counter()
    fit = gamfit.sae_manifold_fit(
        X=train_x,
        K=atoms,
        assignment="topk" if top_k is not None else "softmax",
        top_k=top_k,
        n_iter=n_iter,
        random_state=seed,
    )
    fit_seconds = time.perf_counter() - fit_start

    oos_start = time.perf_counter()
    payload = fit.converged_latents(test_x)
    oos_seconds = time.perf_counter() - oos_start
    assignments = np.asarray(payload["assignments"], dtype=float)
    recon = np.asarray(payload["fitted"], dtype=float)
    coords = [np.asarray(c, dtype=float) for c in payload["coords"]]
    atom_images = [np.asarray(image, dtype=float) for image in payload["atom_images"]]
    blocks = [np.asarray(b, dtype=float) for b in fit.decoder_blocks]
    if len(atom_images) != assignments.shape[1]:
        raise RuntimeError(
            "native OOS atom-image count does not match the assignment width: "
            f"{len(atom_images)} != {assignments.shape[1]}"
        )

    contribution_cache: dict[int, np.ndarray] = {}
    def contribution(g: int) -> np.ndarray:
        if g not in contribution_cache:
            contribution_cache[g] = assignments[:, g:g + 1] * atom_images[g]
        return contribution_cache[g]

    d_atoms = [np.asarray(c).shape[1] if np.asarray(c).ndim == 2 else 1 for c in coords]
    native_bpt = None
    dl = fit.description_length()
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
        extras={
            "basis_kinds": list(fit.basis_kinds),
            "atom_dims": list(fit.atom_dims),
            "fit_config": {
                "atoms": atoms,
                "top_k": top_k,
                "assignment": "topk" if top_k is not None else "softmax",
            },
            "profile": {
                "native_fit_seconds": fit_seconds,
                "heldout_oos_seconds": oos_seconds,
            },
        },
    )


# --------------------------------------------------------------------------- #
# Their scoring protocol                                                      #
# --------------------------------------------------------------------------- #


def _point_biserial_matrix(gate: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """All factor-mask × gate correlations in one matrix multiply."""
    g = gate - gate.mean(axis=0, keepdims=True)
    m = masks.astype(float) - masks.mean(axis=0, keepdims=True)
    num = m.T @ g
    den = np.sqrt(
        np.sum(m * m, axis=0)[:, None] * np.sum(g * g, axis=0)[None, :]
    )
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
        remaining = np.asarray([j for j in fired if int(j) not in chosen], dtype=int)
        if remaining.size == 0:
            break
        remaining_codes = codes[:, remaining]
        residual_projections = resid @ dec[remaining].T
        improvements = (
            2.0 * np.sum(remaining_codes * residual_projections, axis=0)
            - np.sum(remaining_codes * remaining_codes, axis=0)
            * np.sum(dec[remaining] * dec[remaining], axis=1)
        )
        best_j = int(remaining[int(np.argmax(improvements))])
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
    corr = _point_biserial_matrix(fitted.gate, active)  # (M, G)
    matched_atoms: np.ndarray | None = None
    matching = "flat-greedy-multi-atom"
    if not flat_multi_atom:
        if fitted.gate.shape[1] < data.m_factors:
            raise ValueError(
                "one-to-one recovery requires at least as many learned atoms as "
                f"planted factors; got {fitted.gate.shape[1]} < {data.m_factors}"
            )
        factor_rows, atom_columns = _hungarian_max_numpy(np.abs(corr))
        if factor_rows.size != data.m_factors:
            raise RuntimeError(
                "exact atom/factor matching did not cover every planted factor"
            )
        matched_atoms = np.full(data.m_factors, -1, dtype=int)
        matched_atoms[factor_rows] = atom_columns
        matching = "hungarian-exact-one-to-one"
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
            assert matched_atoms is not None
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
        "matching": matching,
        "n_unique_matched_atoms": len(
            {
                int(p["matched_atom"])
                for p in per_factor
                if not isinstance(p["matched_atom"], list)
            }
        ),
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
    if data.dgp != "toy":
        raise ValueError("native analytic cloud dumps require the nuisance-free toy DGP")
    # Deterministic by factor index so every featurizer dumps the SAME factors
    # and the gallery columns are directly comparable.
    candidates = list(recovery["per_factor"])
    candidates.sort(key=lambda p: p["factor"])
    seen_kinds: dict[str, int] = {}
    picks = []
    for p in candidates:  # first instance of each kind, then wrap around
        if seen_kinds.get(p["kind"], 0) == 0:
            picks.append(p)
            seen_kinds[p["kind"]] = 1
    picks = (picks + [p for p in candidates if p not in picks])[:max_factors]
    payload: dict[str, np.ndarray] = {}
    meta: list[dict[str, Any]] = []
    for p in picks:
        i = p["factor"]
        rows = contribs[i]["rows"]
        take = np.arange(rows.size)
        if rows.size > max_points:
            take = np.random.default_rng(0).choice(rows.size, size=max_points, replace=False)
        factor = data.factors[i]
        true_native = contribs[i]["native"][take]
        if isinstance(p["matched_atom"], list):
            _, m_hat_all = _greedy_flat_match(
                fitted, rows, contribs[i]["m"], ZOO[p["kind"]].span_dim
            )
            m_hat = m_hat_all[take]
        else:
            m_hat = fitted.atom_contribution(int(p["matched_atom"]))[rows][take]
        # frame.T exactly inverts the random isometry. Undoing the deterministic
        # center/RMS calibration then restores the sampler's original native
        # analytic coordinates. No displayed-point PCA, SVD, UMAP, or learned
        # projection is involved.
        recovered_local = m_hat @ factor.frame.T
        recovered_native = recovered_local * factor.sigma + factor.mu[None, :]
        validate_analytic_sample(p["kind"], true_native, contribs[i]["theta"][take])
        payload[f"true_{i}"] = np.ascontiguousarray(true_native, dtype=np.float64)
        payload[f"rec_{i}"] = np.ascontiguousarray(recovered_native, dtype=np.float64)
        payload[f"theta_{i}"] = np.ascontiguousarray(
            contribs[i]["theta"][take], dtype=np.float64
        )
        if fitted.atom_intrinsic_coords is not None and not isinstance(p["matched_atom"], list):
            coords = fitted.atom_intrinsic_coords(int(p["matched_atom"]))
            payload[f"learned_theta_{i}"] = np.ascontiguousarray(
                np.asarray(coords)[rows][take], dtype=np.float64
            )
        meta.append(
            {
                "factor": i,
                "kind": p["kind"],
                "matched_atom": p["matched_atom"],
                "r2": p["r2"],
            }
        )
    payload["meta_json"] = np.frombuffer(
        json.dumps(
            {
                "schema": "joint-manifold-sae-analytic-clouds-v3",
                "coordinate_space": "native-analytic",
                "coordinate_dtype": "float64",
                "geometry_revision": GEOMETRY_REVISION,
                "featurizer": fitted.name,
                "joint_fit": fitted.name == "ours_rust",
                "fit_config": (fitted.extras or {}).get("fit_config"),
                "data_config": {
                    "factors": data.m_factors,
                    "ambient": data.d_ambient,
                    "l0": data.l0,
                    "kinds": data.kinds,
                    "dgp": data.dgp,
                },
                "matching": recovery["matching"],
                "n_unique_matched_atoms": recovery["n_unique_matched_atoms"],
                "factors": meta,
            }
        ).encode(),
        dtype=np.uint8,
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
    parser.add_argument(
        "--kinds",
        default=None,
        help=(
            "comma-separated planted factor kinds; when supplied, its length "
            "sets --factors (useful for focused per-kind acceptance runs)"
        ),
    )
    parser.add_argument("--dgp", choices=("toy", "llm"), default="toy",
                        help="'llm' = Zipf firing, context co-occurrence, power-law "
                             "importance, lognormal amplitudes, noise floor.")
    parser.add_argument("--llm-zipf", type=float, default=1.0)
    parser.add_argument("--llm-importance", type=float, default=0.5)
    parser.add_argument("--llm-amp-sigma", type=float, default=0.6)
    parser.add_argument("--llm-noise", type=float, default=0.25)
    parser.add_argument("--llm-contexts", type=int, default=4)
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--featurizers", default="ours_rust,flat,oracle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="bsf_zoo_results.jsonl")
    parser.add_argument("--dump-clouds", default=None,
                        help="Directory for per-featurizer NPZ cloud dumps.")
    args = parser.parse_args()

    selected_kinds = (
        [kind.strip().lower() for kind in args.kinds.split(",") if kind.strip()]
        if args.kinds is not None
        else None
    )
    factors = len(selected_kinds) if selected_kinds is not None else args.factors
    if factors == 0:
        raise SystemExit("--kinds must name at least one zoo kind")
    if args.l0 > factors:
        raise SystemExit(f"--l0={args.l0} cannot exceed the {factors} planted factors")
    data = ZooData(factors, args.ambient, args.l0, args.seed,
                   curved_fraction=args.curved_fraction, kinds=selected_kinds, dgp=args.dgp,
                   zipf=args.llm_zipf, importance=args.llm_importance,
                   amp_sigma=args.llm_amp_sigma, noise=args.llm_noise,
                   n_contexts=args.llm_contexts)
    train_x, _train_active, _ = data.sample(args.n_train, args.seed + 1,
                                            keep_contributions=False)
    test_x, test_active, contribs = data.sample(args.n_test, args.seed + 2,
                                                keep_contributions=True)
    assert contribs is not None
    span_sum = sum(ZOO[k].span_dim for k in data.kinds)
    atoms = args.atoms or factors
    top_k = args.top_k or args.l0
    flat_width = args.flat_width or 2 * span_sum
    flat_k = args.flat_k or max(args.l0 * int(round(span_sum / factors)), args.l0)
    out_path = Path(args.out)
    clouds_dir = Path(args.dump_clouds) if args.dump_clouds else None
    if clouds_dir is not None:
        clouds_dir.mkdir(parents=True, exist_ok=True)

    header = {
        "record": "config",
        "factors": factors, "ambient": args.ambient, "l0": args.l0,
        "curved_fraction": args.curved_fraction, "kinds": data.kinds,
        "dgp": args.dgp,
        "llm_zipf": args.llm_zipf,
        "llm_importance": args.llm_importance,
        "llm_amp_sigma": args.llm_amp_sigma,
        "llm_noise": args.llm_noise,
        "llm_contexts": args.llm_contexts,
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
            try:
                fitted = _fit_ours_rust(
                    train_x, test_x, atoms=atoms, top_k=top_k,
                    n_iter=args.rust_iters, seed=args.seed,
                )
            except Exception as error:
                # A failed/non-converged manifold arm is scientific output, not
                # an absent row. Persist the typed failure before re-raising so
                # the process remains fail-loud and the JSONL distinguishes
                # "ran and refused" from "never started" (#2230/#2134).
                failure = {
                    "record": "error",
                    "featurizer": "ours_rust",
                    "seed": args.seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                with out_path.open("a") as fh:
                    fh.write(json.dumps(failure) + "\n")
                print(
                    f"[ours_rust] failed: {failure['error_type']}: {failure['error']}",
                    flush=True,
                )
                raise
        else:
            raise SystemExit(f"unknown featurizer {which!r}")
        scoring_start = time.perf_counter()
        recovery = score_recovery(
            data, fitted, test_active, contribs,
            flat_multi_atom=(which == "flat"),
        )
        scoring_seconds = time.perf_counter() - scoring_start

        mdl_start = time.perf_counter()
        mdl = description_length(fitted, test_x)
        mdl_seconds = time.perf_counter() - mdl_start

        dimensionality_start = time.perf_counter()
        dimensionality = stable_ranks(fitted)
        dimensionality_seconds = time.perf_counter() - dimensionality_start

        dump_seconds = 0.0
        if clouds_dir is not None:
            dump_start = time.perf_counter()
            dump_clouds(
                clouds_dir / f"clouds_{fitted.name}_seed{args.seed}.npz",
                data, fitted, contribs, recovery,
            )
            dump_seconds = time.perf_counter() - dump_start

        profile = dict((fitted.extras or {}).get("profile", {}))
        profile.update(
            recovery_scoring_seconds=scoring_seconds,
            description_length_seconds=mdl_seconds,
            dimensionality_seconds=dimensionality_seconds,
            cloud_dump_seconds=dump_seconds,
        )
        record = {
            "record": "result",
            "featurizer": fitted.name,
            "seed": args.seed,
            "test_r2": _test_r2(fitted, test_x),
            "fit_seconds": fitted.fit_seconds,
            "recovery_r2_mean": recovery["r2_mean"],
            "recovery_r2_by_kind": recovery["r2_by_kind"],
            "matching": recovery["matching"],
            "n_unique_matched_atoms": recovery["n_unique_matched_atoms"],
            "coords_per_activation_mean": recovery["coords_per_activation_mean"],
            "mdl": mdl,
            "dimensionality": dimensionality,
            "per_factor": recovery["per_factor"],
            "profile": profile,
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
        print(
            "    profile: "
            + ", ".join(f"{name}={seconds:.2f}s" for name, seconds in profile.items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
