#!/usr/bin/env python
"""Protocol-grade held-out topology race on real LLM activations.

The driver fits every learned arm on a deterministic fit split, evaluates all
reported EV/log-density numbers on the held-out eval split, and emits a single
JSON report with a reproducible protocol block. It also reports active scalar
channels per eval row and total parameter counts so manifold, vanilla SAE,
official SAE, union-flat null, and PCA arms can be compared by matched capacity.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import traceback
from typing import Any

import numpy as np
import torch


TOPOLOGY_DEFAULT_DIM = {"circle": 1, "sphere": 2, "torus": 2, "euclidean": 3}


def _estimated_basis_size(topology: str, d_atom: int, n_obs: int) -> int:
    if topology in {"circle", "periodic"}:
        return 2 * max(1, int(d_atom)) + 1
    if topology == "sphere":
        return 7
    if topology == "torus":
        return (2 * 3 + 1) ** max(1, int(d_atom))
    center_floor = max(8, int(d_atom) + 2)
    return min(max(center_floor, 32), max(1, int(n_obs)))


def sae_candidate_plan(n_obs: int, p_out: int, k: int, topology: str, d_atom: int) -> dict[str, Any]:
    import gamfit

    total_basis = int(k) * _estimated_basis_size(topology, d_atom, n_obs)
    border_dim = total_basis * int(p_out)
    plan = gamfit._sae_manifold.rust_module().sae_streaming_plan(
        int(n_obs),
        total_basis,
        int(k),
        int(d_atom),
        border_dim,
    )
    return dict(plan)


def plan_peak_bytes(plan: dict[str, Any]) -> int:
    if plan.get("direct_admitted"):
        return int(plan.get("estimated_direct_peak_bytes") or 0)
    if plan.get("matrix_free_admitted"):
        return int(plan.get("estimated_matrix_free_peak_bytes") or 0)
    return max(
        int(plan.get("estimated_direct_peak_bytes") or 0),
        int(plan.get("estimated_matrix_free_peak_bytes") or 0),
    )


def load_activations(path: str, n: int | None, npy_key: str | None,
                     shuffle_seed: int | None) -> torch.Tensor:
    if path.endswith(".npy"):
        x = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        x = payload[npy_key or "X"]
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        x = x[torch.randperm(x.shape[0], generator=g)]
    if n is not None and n > 0:
        x = x[:n]
    return x.float()


def deterministic_split(n_rows: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < eval_fraction < 1.0:
        raise SystemExit("--eval-frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_eval = max(1, int(round(n_rows * eval_fraction)))
    n_eval = min(n_eval, n_rows - 1)
    eval_idx = np.sort(order[:n_eval])
    fit_idx = np.sort(order[n_eval:])
    return fit_idx, eval_idx


def normalize_from_fit(x_fit_raw: torch.Tensor, x_eval_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x_fit_raw.mean(0, keepdim=True)
    sigma = x_fit_raw.std(0, keepdim=True).clamp(min=1e-6)
    return (x_fit_raw - mu) / sigma, (x_eval_raw - mu) / sigma, mu, sigma


def explained_variance(x: np.ndarray, xhat: np.ndarray) -> float:
    num = float(np.square(x - xhat).sum())
    den = float(np.square(x - x.mean(axis=0, keepdims=True)).sum())
    return 1.0 - num / den if den > 0.0 else 0.0


def fit_dispersion(x_fit: np.ndarray, xhat_fit: np.ndarray) -> float:
    return max(float(np.square(x_fit - xhat_fit).mean()), 1e-12)


def gaussian_log_density_per_scalar(x: np.ndarray, xhat: np.ndarray, dispersion: float) -> float:
    mse = float(np.square(x - xhat).mean())
    return -0.5 * (math.log(2.0 * math.pi * dispersion) + mse / dispersion)


def add_eval_metrics(entry: dict[str, Any], x_eval: np.ndarray, xhat_eval: np.ndarray,
                     dispersion: float, *, x_eval_raw: np.ndarray | None = None,
                     mu: np.ndarray | None = None, sigma: np.ndarray | None = None) -> None:
    entry["eval_ev"] = explained_variance(x_eval, xhat_eval)
    entry["eval_mse"] = float(np.square(x_eval - xhat_eval).mean())
    entry["fit_dispersion"] = dispersion
    entry["eval_log_density_per_scalar"] = gaussian_log_density_per_scalar(
        x_eval, xhat_eval, dispersion,
    )
    entry["eval_log_density_total"] = float(entry["eval_log_density_per_scalar"] * x_eval.size)
    if x_eval_raw is not None and mu is not None and sigma is not None:
        entry["eval_ev_raw"] = explained_variance(x_eval_raw, xhat_eval * sigma + mu)


def active_threshold(assignment: str, k: int) -> float:
    if assignment == "softmax":
        return 1.0 / max(1, k)
    if assignment == "jumprelu":
        return 0.0
    return 0.5


def manifold_active_channels(assignments: np.ndarray, dims: list[int], assignment: str) -> float:
    cut = active_threshold(assignment, assignments.shape[1])
    active = assignments > cut if assignment != "jumprelu" else assignments > 0.0
    charges = np.asarray([1 + int(d) for d in dims], dtype=np.float64)
    return float((active * charges.reshape(1, -1)).sum(axis=1).mean())


def manifold_parameter_count(model: Any) -> tuple[int, dict[str, int]]:
    decoder = int(sum(np.asarray(block).size for block in model.decoder_blocks))
    coords = int(sum(np.asarray(coord).size for coord in model.coords))
    assignments = int(np.asarray(model.assignments).size)
    logits = int(np.asarray(getattr(model, "low_level_logits", np.zeros((0, 0)))).size)
    total = decoder + coords + assignments + logits
    return total, {
        "decoder_coefficients": decoder,
        "fit_coordinates": coords,
        "fit_assignments": assignments,
        "fit_logits": logits,
    }


class _VanillaSAE(torch.nn.Module):
    """MLP encoder, hard top-k, linear decoder baseline."""

    def __init__(self, d_model: int, features: int, top_k: int) -> None:
        super().__init__()
        self.F = features
        self.top_k = top_k
        hidden = 4 * d_model
        self.norm = torch.nn.LayerNorm(d_model)
        self.fc1 = torch.nn.Linear(d_model, hidden)
        self.act = torch.nn.GELU()
        self.head = torch.nn.Linear(hidden, features)
        self.W_dec = torch.nn.Parameter(torch.randn(features, d_model) / d_model**0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.nn.functional.relu(self.head(self.act(self.fc1(self.norm(x)))))
        vals, idx = torch.topk(z, self.top_k, dim=1)
        gate = torch.zeros_like(z).scatter_(1, idx, vals)
        return gate @ self.W_dec, gate


def vanilla_parameter_count(sae: _VanillaSAE) -> int:
    return int(sum(p.numel() for p in sae.parameters()))


def vanilla_eval(sae: _VanillaSAE, x_fit: torch.Tensor, x_eval: torch.Tensor,
                 label: str) -> dict[str, Any]:
    sae.eval()
    with torch.no_grad():
        xhat_fit, _gate_fit = sae(x_fit)
        xhat_eval, gate_eval = sae(x_eval)
    out: dict[str, Any] = {
        "method": "vanilla_topk_sae",
        "label": label,
        "F": sae.F,
        "top_k": sae.top_k,
        "active_scalar_channels_per_row": float(sae.top_k),
        "total_parameter_count": vanilla_parameter_count(sae),
        "alive_atoms_eval": int((gate_eval.abs().sum(0) > 0).sum().item()),
        "mean_active_per_eval_row": float((gate_eval > 0).float().sum(1).mean().item()),
    }
    dispersion = fit_dispersion(x_fit.numpy(), xhat_fit.numpy())
    add_eval_metrics(out, x_eval.numpy(), xhat_eval.numpy(), dispersion)
    return out


def vanilla_from_ckpt(ckpt_path: str, x_fit: torch.Tensor, x_eval: torch.Tensor) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    sae = _VanillaSAE(x_fit.shape[1], int(sig.get("F", 16)), int(sig.get("top_k", 2)))
    sae.load_state_dict(ckpt["sae"])
    out = vanilla_eval(sae, x_fit, x_eval, "vanilla_topk_ckpt")
    out["ckpt"] = ckpt_path
    return out


def vanilla_train(x_fit: torch.Tensor, x_eval: torch.Tensor, features: int, top_k: int,
                  steps: int, lr: float, batch: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sae = _VanillaSAE(x_fit.shape[1], features, top_k).to(dev)
    x_dev = x_fit.to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.randint(0, x_dev.shape[0], (min(batch, x_dev.shape[0]),), device=dev)
        xb = x_dev[idx]
        xhat, _gate = sae(xb)
        loss = torch.square(xb - xhat).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"    [vanilla step {step}] mse={loss.item():.4e}", flush=True)
    out = vanilla_eval(sae.cpu(), x_fit, x_eval, "vanilla_topk_trained")
    out["steps"] = steps
    return out


def official_topk_sae_eval(ckpt_path: str, x_fit_raw: torch.Tensor, x_eval_raw: torch.Tensor,
                           mu: torch.Tensor, sigma: torch.Tensor) -> dict[str, Any]:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    k = 100
    low = ckpt_path.lower()
    for cand in (25, 50, 100, 200):
        if f"k{cand}" in low or f"l0_{cand}" in low:
            k = cand
    w_enc, w_dec = sd["W_enc"].float(), sd["W_dec"].float()
    b_enc, b_dec = sd["b_enc"].float(), sd["b_dec"].float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_enc, w_dec, b_enc, b_dec = (t.to(dev) for t in (w_enc, w_dec, b_enc, b_dec))

    def reconstruct(raw: torch.Tensor) -> tuple[torch.Tensor, float]:
        outs = []
        actives = []
        with torch.no_grad():
            for chunk in torch.split(raw.to(dev), 8192):
                pre = (chunk - b_dec) @ w_enc.T + b_enc
                vals, idx = torch.topk(torch.relu(pre), k, dim=1)
                z = torch.zeros_like(pre).scatter_(1, idx, vals)
                actives.append((z > 0).sum(1).float().mean().item())
                outs.append((z @ w_dec.T + b_dec).cpu())
        return torch.cat(outs), float(np.mean(actives))

    xhat_fit_raw, _fit_active = reconstruct(x_fit_raw)
    xhat_eval_raw, eval_active = reconstruct(x_eval_raw)
    x_fit = ((x_fit_raw - mu) / sigma).numpy()
    x_eval = ((x_eval_raw - mu) / sigma).numpy()
    xhat_fit = ((xhat_fit_raw - mu) / sigma).numpy()
    xhat_eval = ((xhat_eval_raw - mu) / sigma).numpy()
    out: dict[str, Any] = {
        "method": "official_topk_sae",
        "label": "official_topk_sae",
        "ckpt": ckpt_path,
        "d_sae": int(w_enc.shape[0]),
        "k": k,
        "active_scalar_channels_per_row": float(k),
        "total_parameter_count": int(w_enc.numel() + w_dec.numel() + b_enc.numel() + b_dec.numel()),
        "mean_active_per_eval_row": eval_active,
    }
    add_eval_metrics(
        out,
        x_eval,
        xhat_eval,
        fit_dispersion(x_fit, xhat_fit),
        x_eval_raw=x_eval_raw.numpy(),
        mu=mu.numpy(),
        sigma=sigma.numpy(),
    )
    return out


def worker_gamfit():
    import gamfit

    return gamfit


def run_candidate(x_fit: np.ndarray, x_eval: np.ndarray, k: int, topology: str, d_atom: int,
                  n_iter: int, seed: int, assignment: str, top_k: int | None,
                  mu: np.ndarray, sigma: np.ndarray, x_eval_raw: np.ndarray,
                  solver_plan: dict[str, Any]) -> dict[str, Any]:
    gamfit = worker_gamfit()
    t0 = time.time()
    role = "union_flat_patches_null" if topology == "euclidean" else "manifold_atom"
    out: dict[str, Any] = {
        "method": "gam_manifold_sae",
        "role": role,
        "topology": topology,
        "K": k,
        "d_atom": d_atom,
        "assignment": assignment,
        "n_iter": n_iter,
        "seed": seed,
        "solver_plan": solver_plan,
    }
    try:
        model = gamfit.sae_manifold_fit(
            x_fit, K=k, d_atom=d_atom, atom_topology=topology,
            assignment=assignment, n_iter=n_iter, random_state=seed,
            top_k=top_k,
        )
        latents = model.converged_latents(x_eval)
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
        out["seconds"] = time.time() - t0
        return out

    out["status"] = "ok"
    out["seconds"] = time.time() - t0
    out["solver_plan"] = getattr(model, "solver_plan", None) or solver_plan
    out["reml_score_fit"] = float(model.reml_score) if model.reml_score is not None else None
    xhat_fit = np.asarray(model.fitted, dtype=np.float64)
    xhat_eval = np.asarray(latents["fitted"], dtype=np.float64)
    assignments_eval = np.asarray(latents["assignments"], dtype=np.float64)
    dims = [int(getattr(atom, "active_dim", None) or d_atom) for atom in model.atoms]
    params, param_breakdown = manifold_parameter_count(model)
    out["active_scalar_channels_per_row"] = manifold_active_channels(assignments_eval, dims, assignment)
    out["total_parameter_count"] = params
    out["parameter_count_breakdown"] = param_breakdown
    out["mean_active_atoms_per_eval_row"] = float(
        (assignments_eval > active_threshold(assignment, assignments_eval.shape[1])).sum(axis=1).mean()
    )
    out["alive_atoms_eval"] = int(
        ((assignments_eval > active_threshold(assignment, assignments_eval.shape[1])).sum(axis=0) > 0).sum()
    )
    out["atoms"] = [
        {
            "atom": i,
            "evidence": getattr(atom, "evidence", None),
            "active_dim": getattr(atom, "active_dim", None),
            "eval_mass": float(assignments_eval[:, i].sum() / max(assignments_eval.sum(), 1e-12)),
        }
        for i, atom in enumerate(model.atoms)
    ]
    add_eval_metrics(
        out,
        x_eval,
        xhat_eval,
        fit_dispersion(x_fit, xhat_fit),
        x_eval_raw=x_eval_raw,
        mu=mu,
        sigma=sigma,
    )
    for field in ("residual_gauge", "atom_two_lens", "metric_provenance"):
        value = getattr(model, field, None)
        if value is not None:
            try:
                json.dumps(value)
                out[field] = value
            except TypeError:
                out[field] = repr(value)[:4000]
    return out


def pca_eval(x_fit: np.ndarray, x_eval: np.ndarray, rank: int) -> dict[str, Any]:
    rank = max(1, min(int(rank), min(x_fit.shape)))
    mean = x_fit.mean(axis=0, keepdims=True)
    centered_fit = x_fit - mean
    _u, _s, vt = np.linalg.svd(centered_fit, full_matrices=False)
    components = vt[:rank]
    xhat_fit = (centered_fit @ components.T) @ components + mean
    centered_eval = x_eval - mean
    xhat_eval = (centered_eval @ components.T) @ components + mean
    out: dict[str, Any] = {
        "method": "pca_rank_matched",
        "role": "pca_rank_matched_linear_null",
        "rank": rank,
        "active_scalar_channels_per_row": float(rank),
        "total_parameter_count": int(mean.size + components.size),
    }
    add_eval_metrics(out, x_eval, xhat_eval, fit_dispersion(x_fit, xhat_fit))
    return out


def collect_capacity_ranks(entries: list[dict[str, Any]], max_rank: int) -> list[int]:
    ranks = set()
    for entry in entries:
        if entry.get("status", "ok") != "ok":
            continue
        active = entry.get("active_scalar_channels_per_row")
        if active is None:
            continue
        ranks.add(max(1, min(int(round(float(active))), max_rank)))
    return sorted(ranks)


def comparison_table(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        if entry.get("status", "ok") != "ok":
            continue
        active = entry.get("active_scalar_channels_per_row")
        if active is None:
            continue
        rows.append({
            "capacity_group_active_channels": int(round(float(active))),
            "active_scalar_channels_per_row": float(active),
            "method": entry.get("method"),
            "role": entry.get("role", entry.get("label")),
            "topology": entry.get("topology"),
            "rank": entry.get("rank"),
            "total_parameter_count": entry.get("total_parameter_count"),
            "eval_ev": entry.get("eval_ev"),
            "eval_log_density_per_scalar": entry.get("eval_log_density_per_scalar"),
        })
    return sorted(rows, key=lambda row: (
        row["capacity_group_active_channels"],
        str(row["method"]),
        str(row.get("topology")),
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--npy-key", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle rows with this seed before --n truncation")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--topologies", default="euclidean,circle,sphere")
    ap.add_argument("--d-atom", type=int, default=None,
                    help="override per-topology default intrinsic dim")
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assignment", default="ibp_map")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--vanilla-ckpt", default=None)
    ap.add_argument("--official-sae", default=None, action="append",
                    help="published top-k SAE checkpoint(s) to evaluate on the same split")
    ap.add_argument("--train-vanilla", action="store_true",
                    help="train a fresh vanilla top-k baseline on the fit split")
    ap.add_argument("--vanilla-f", type=int, default=None,
                    help="dictionary size for --train-vanilla (default: --k)")
    ap.add_argument("--vanilla-topk", type=int, default=2)
    ap.add_argument("--vanilla-steps", type=int, default=500)
    ap.add_argument("--vanilla-lr", type=float, default=1e-3)
    ap.add_argument("--vanilla-batch", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_seed = args.seed if args.split_seed is None else args.split_seed
    x_raw = load_activations(args.acts, args.n, args.npy_key, args.shuffle_seed)
    fit_idx, eval_idx = deterministic_split(x_raw.shape[0], args.eval_frac, split_seed)
    x_fit_raw = x_raw[torch.as_tensor(fit_idx)]
    x_eval_raw = x_raw[torch.as_tensor(eval_idx)]
    x_fit, x_eval, mu, sigma = normalize_from_fit(x_fit_raw, x_eval_raw)

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if "euclidean" not in topologies:
        topologies.append("euclidean")

    report: dict[str, Any] = {
        "protocol": {
            "name": "W2 held-out matched-capacity topology race",
            "acts": args.acts,
            "n_total": int(x_raw.shape[0]),
            "p": int(x_raw.shape[1]),
            "split": {
                "kind": "deterministic_random_holdout",
                "seed": int(split_seed),
                "eval_fraction": float(args.eval_frac),
                "fit_n": int(fit_idx.size),
                "eval_n": int(eval_idx.size),
                "fit_indices": fit_idx.tolist(),
                "eval_indices": eval_idx.tolist(),
            },
            "normalization": "mean/std fit on fit split only and reused for eval",
            "fit_budgets": {
                "seed": int(args.seed),
                "k": int(args.k),
                "n_iter": int(args.n_iter),
                "assignment": args.assignment,
                "top_k": args.top_k,
                "topologies": topologies,
                "d_atom_override": args.d_atom,
                "vanilla_steps": int(args.vanilla_steps),
            },
            "reported_ev_log_density": "eval split only; Gaussian dispersion estimated on fit residuals",
            "capacity": "active scalar channels per eval row; manifold charges gate plus active d_k per active atom",
            "nulls": ["union_flat_patches_null: gam atom_topology=euclidean", "pca_rank_matched_linear_null"],
        },
        "candidates": [],
        "baselines": [],
        "pca": [],
    }
    print(
        f"[setup] fit {tuple(x_fit.shape)} eval {tuple(x_eval.shape)} "
        f"var_eval={float(x_eval.var()):.3f}",
        flush=True,
    )

    for ckpt in args.official_sae or []:
        entry = official_topk_sae_eval(ckpt, x_fit_raw, x_eval_raw, mu, sigma)
        report["baselines"].append(entry)
        print(f"[baseline] official k={entry['k']} eval_ev={entry['eval_ev']}", flush=True)

    if args.vanilla_ckpt:
        entry = vanilla_from_ckpt(args.vanilla_ckpt, x_fit, x_eval)
        report["baselines"].append(entry)
        print(f"[baseline] vanilla ckpt eval_ev={entry['eval_ev']}", flush=True)
    elif args.train_vanilla:
        entry = vanilla_train(
            x_fit, x_eval, args.vanilla_f or args.k, args.vanilla_topk,
            args.vanilla_steps, args.vanilla_lr, args.vanilla_batch, args.seed,
        )
        report["baselines"].append(entry)
        print(f"[baseline] vanilla trained eval_ev={entry['eval_ev']}", flush=True)

    x_fit64 = x_fit.double().numpy()
    x_eval64 = x_eval.double().numpy()
    mu64 = mu.numpy().astype(np.float64)
    sigma64 = sigma.numpy().astype(np.float64)
    x_eval_raw64 = x_eval_raw.numpy().astype(np.float64)
    candidates = []
    for topo in topologies:
        d = args.d_atom if args.d_atom is not None else TOPOLOGY_DEFAULT_DIM.get(topo, 2)
        plan = sae_candidate_plan(x_fit64.shape[0], x_fit64.shape[1], args.k, topo, d)
        candidates.append((topo, d, plan))

    budget = min(int(c[2].get("in_core_budget_bytes", 0)) for c in candidates) if candidates else 0
    batches: list[list[tuple[str, int, dict[str, Any]]]] = []
    current: list[tuple[str, int, dict[str, Any]]] = []
    current_peak = 0
    for candidate in sorted(candidates, key=lambda item: plan_peak_bytes(item[2]), reverse=True):
        peak = plan_peak_bytes(candidate[2])
        if current and budget > 0 and current_peak + peak > budget:
            batches.append(current)
            current = []
            current_peak = 0
        current.append(candidate)
        current_peak += peak
    if current:
        batches.append(current)
    report["protocol"]["fit_budgets"]["solver_plan_batches"] = [
        {
            "candidates": [topo for topo, _d, _plan in batch],
            "predicted_peak_bytes": int(sum(plan_peak_bytes(plan) for _topo, _d, plan in batch)),
            "budget_bytes": int(budget),
        }
        for batch in batches
    ]

    for batch in batches:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {}
            for topo, d, plan in batch:
                print(
                    f"[fit] topology={topo} K={args.k} d={d} n_iter={args.n_iter} "
                    f"route={plan.get('route')} predicted_peak_gib={plan_peak_bytes(plan) / 1024**3:.2f}",
                    flush=True,
                )
                future = executor.submit(
                    run_candidate,
                    x_fit64,
                    x_eval64,
                    args.k,
                    topo,
                    d,
                    args.n_iter,
                    args.seed,
                    args.assignment,
                    args.top_k,
                    mu64,
                    sigma64,
                    x_eval_raw64,
                    plan,
                )
                futures[future] = (topo, d, plan)

            for future in concurrent.futures.as_completed(futures):
                topo, d, plan = futures[future]
                try:
                    res = future.result()
                except Exception as exc:
                    res = {
                        "method": "gam_manifold_sae",
                        "topology": topo,
                        "K": args.k,
                        "d_atom": d,
                        "assignment": args.assignment,
                        "n_iter": args.n_iter,
                        "seed": args.seed,
                        "solver_plan": plan,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:],
                    }
                print(
                    f"[fit] -> topology={topo} status={res['status']} "
                    f"eval_ev={res.get('eval_ev')} reml_fit={res.get('reml_score_fit')}",
                    flush=True,
                )
                report["candidates"].append(res)
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=2, default=str)

    comparable = [*report["candidates"], *report["baselines"]]
    max_rank = max(1, min(x_fit64.shape))
    for rank in collect_capacity_ranks(comparable, max_rank):
        entry = pca_eval(x_fit64, x_eval64, rank)
        report["pca"].append(entry)
        print(f"[baseline] pca rank={rank} eval_ev={entry['eval_ev']}", flush=True)

    report["comparison_table"] = comparison_table([*report["candidates"], *report["baselines"], *report["pca"]])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
