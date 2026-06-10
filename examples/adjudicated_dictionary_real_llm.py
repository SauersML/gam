#!/usr/bin/env python
"""Evidence-adjudicated manifold dictionary on real LLM activations.

Feeds real residual-stream activations into gam's native multi-atom joint
manifold-SAE fit (:func:`gamfit.sae_manifold_fit`) and reports, per topology
candidate: REML evidence, reconstruction EV (the same explained-variance
metric vanilla-SAE baselines report), per-atom ARD active dimensions, the
residual-gauge identifiability certificate, and assignment sparsity.

The point of the exercise (issue #977's real-model rung): a vanilla SAE
asserts "every feature is a direction" and the older curve-SAE experiments
asserted "every feature is a 1D curve" — both architecture fiat. Here the
atom dimension is ARD-selected, the topology is raced on evidence, and every
verdict ships with an identifiability certificate, on activations from a real
model rather than planted synthetic data.

Baselines: pass ``--vanilla-ckpt`` to re-evaluate an existing vanilla top-k
checkpoint on exactly the rows being fit, or ``--train-vanilla`` to train one
from scratch on those rows (same architecture/recipe as the Manifold-SAE
``llm_sweep`` harness).

Usage examples:
  python adjudicated_dictionary_real_llm.py --acts acts.pt --n 2000 --k 4 \
      --topologies euclidean --n-iter 10 --out smoke.json
  python adjudicated_dictionary_real_llm.py --acts acts.pt --k 16 \
      --topologies euclidean,circle,sphere --train-vanilla \
      --out dictionary_k16.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import torch


def load_normalized(path: str, n: int | None, npy_key: str | None = None,
                    shuffle_seed: int | None = None):
    """Load activations; return (z-scored X, raw X, mu, sigma)."""
    if path.endswith(".npy"):
        X = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
    else:
        d = torch.load(path, map_location="cpu", weights_only=False)
        X = d[npy_key or "X"]
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        X = X[torch.randperm(X.shape[0], generator=g)]
    if n is not None and n > 0:
        X = X[:n]
    X = X.float()
    mu = X.mean(0, keepdim=True)
    sigma = X.std(0).clamp(min=1e-6)
    return (X - mu) / sigma, X, mu, sigma


def explained_variance(X: np.ndarray, Xhat: np.ndarray) -> float:
    num = float(((X - Xhat) ** 2).sum())
    den = float(((X - X.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - num / den


class _VanillaSAE(torch.nn.Module):
    """The llm_sweep vanilla baseline: MLP encoder, hard top-k, linear decoder."""

    def __init__(self, D: int, F: int, top_k: int) -> None:
        super().__init__()
        self.F = F
        self.top_k = top_k
        H = 4 * D
        self.norm = torch.nn.LayerNorm(D)
        self.fc1 = torch.nn.Linear(D, H)
        self.act = torch.nn.GELU()
        self.head = torch.nn.Linear(H, F)
        self.W_dec = torch.nn.Parameter(torch.randn(F, D) / D**0.5)

    def forward(self, x):
        z = torch.nn.functional.relu(self.head(self.act(self.fc1(self.norm(x)))))
        vals, idx = torch.topk(z, self.top_k, dim=1)
        gate = torch.zeros_like(z).scatter_(1, idx, vals)
        return gate @ self.W_dec, gate


def vanilla_eval(sae: "_VanillaSAE", Xn: torch.Tensor) -> dict:
    sae.eval()
    with torch.no_grad():
        Xhat, gate = sae(Xn)
    return {
        "F": sae.F,
        "top_k": sae.top_k,
        "ev": explained_variance(Xn.numpy(), Xhat.numpy()),
        "alive_atoms": int((gate.abs().sum(0) > 0).sum().item()),
        "mean_active_per_row": float((gate > 0).float().sum(1).mean().item()),
    }


def vanilla_from_ckpt(ckpt_path: str, Xn: torch.Tensor) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sig = ckpt.get("sig", {})
    sae = _VanillaSAE(Xn.shape[1], int(sig.get("F", 16)), int(sig.get("top_k", 2)))
    sae.load_state_dict(ckpt["sae"])
    out = vanilla_eval(sae, Xn)
    out["label"] = "vanilla_topk_ckpt"
    return out


def vanilla_train(Xn: torch.Tensor, F: int, top_k: int, steps: int, lr: float,
                  batch: int, seed: int) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sae = _VanillaSAE(Xn.shape[1], F, top_k).to(dev)
    X = Xn.to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.randint(0, X.shape[0], (min(batch, X.shape[0]),), device=dev)
        xb = X[idx]
        xhat, _ = sae(xb)
        loss = ((xb - xhat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"    [vanilla step {step}] mse={loss.item():.4e}", flush=True)
    out = vanilla_eval(sae.cpu(), Xn)
    out["label"] = "vanilla_topk_trained"
    out["steps"] = steps
    return out


def official_topk_sae_eval(ckpt_path: str, X_raw: torch.Tensor,
                           mu: torch.Tensor, sigma: torch.Tensor) -> dict:
    """Evaluate a published top-k SAE (Qwen SAE-Res format: W_enc/W_dec/b_enc/b_dec).

    Decode convention: z = topk(W_enc @ (x - b_dec) + b_enc, k);
    xhat = W_dec @ z + b_dec. ``k`` is taken from the filename convention
    (``k100``/``k50``) or defaults to 100.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    k = 100
    low = ckpt_path.lower()
    for cand in (25, 50, 100, 200):
        if f"k{cand}" in low or f"l0_{cand}" in low:
            k = cand
    W_enc, W_dec = sd["W_enc"].float(), sd["W_dec"].float()
    b_enc, b_dec = sd["b_enc"].float(), sd["b_dec"].float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W_enc, W_dec, b_enc, b_dec = (t.to(dev) for t in (W_enc, W_dec, b_enc, b_dec))
    outs = []
    actives = []
    with torch.no_grad():
        for chunk in torch.split(X_raw.to(dev), 8192):
            pre = (chunk - b_dec) @ W_enc.T + b_enc
            vals, idx = torch.topk(torch.relu(pre), k, dim=1)
            z = torch.zeros_like(pre).scatter_(1, idx, vals)
            actives.append((z > 0).sum(1).float().mean().item())
            outs.append((z @ W_dec.T + b_dec).cpu())
    xhat_raw = torch.cat(outs)
    xhat_z = (xhat_raw - mu) / sigma
    Xn = ((X_raw - mu) / sigma).numpy()
    return {
        "label": "official_topk_sae",
        "ckpt": ckpt_path,
        "d_sae": int(W_enc.shape[0]),
        "k": k,
        "ev_raw": explained_variance(X_raw.numpy(), xhat_raw.numpy()),
        "ev": explained_variance(Xn, xhat_z.numpy()),
        "mean_active_per_row": float(np.mean(actives)),
    }


TOPOLOGY_DEFAULT_DIM = {"circle": 1, "sphere": 2, "torus": 2, "euclidean": 3}


def run_candidate(Xn64: np.ndarray, k: int, topology: str, d_atom: int,
                  n_iter: int, seed: int, assignment: str, top_k: int | None,
                  mu: np.ndarray | None = None, sigma: np.ndarray | None = None,
                  X_raw: np.ndarray | None = None) -> dict:
    import gamfit

    t0 = time.time()
    out: dict = {"topology": topology, "K": k, "d_atom": d_atom,
                 "assignment": assignment, "n_iter": n_iter, "seed": seed}
    try:
        m = gamfit.sae_manifold_fit(
            Xn64, K=k, d_atom=d_atom, atom_topology=topology,
            assignment=assignment, n_iter=n_iter, random_state=seed,
            top_k=top_k,
        )
    except Exception as e:  # record honest failures, keep the ladder going
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        out["seconds"] = time.time() - t0
        return out
    out["status"] = "ok"
    out["seconds"] = time.time() - t0
    out["reml_score"] = float(m.reml_score) if m.reml_score is not None else None
    out["reconstruction_r2"] = (
        float(m.reconstruction_r2) if m.reconstruction_r2 is not None else None
    )
    fitted = np.asarray(m.fitted)
    out["ev"] = explained_variance(Xn64, fitted)
    if mu is not None and sigma is not None and X_raw is not None:
        out["ev_raw"] = explained_variance(X_raw, fitted * sigma + mu)
    asg = np.asarray(m.assignments)
    out["mean_active_per_row"] = float((asg > 1e-3).sum(axis=1).mean())
    out["alive_atoms"] = int(((asg > 1e-3).sum(axis=0) > 0).sum())
    out["atoms"] = [
        {
            "atom": i,
            "evidence": getattr(a, "evidence", None),
            "active_dim": getattr(a, "active_dim", None),
            "mass": float(asg[:, i].sum() / max(asg.sum(), 1e-12)),
        }
        for i, a in enumerate(m.atoms)
    ]
    for field in ("residual_gauge", "atom_two_lens", "metric_provenance"):
        v = getattr(m, field, None)
        if v is not None:
            try:
                json.dumps(v)
                out[field] = v
            except TypeError:
                out[field] = repr(v)[:4000]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--npy-key", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle rows with this seed before --n truncation")
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
                    help="published top-k SAE checkpoint(s) to evaluate on the same rows")
    ap.add_argument("--train-vanilla", action="store_true",
                    help="train a fresh vanilla top-k baseline on the same rows")
    ap.add_argument("--vanilla-f", type=int, default=None,
                    help="dictionary size for --train-vanilla (default: --k)")
    ap.add_argument("--vanilla-topk", type=int, default=2)
    ap.add_argument("--vanilla-steps", type=int, default=500)
    ap.add_argument("--vanilla-lr", type=float, default=1e-3)
    ap.add_argument("--vanilla-batch", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xn, X_raw, mu, sigma = load_normalized(args.acts, args.n, args.npy_key,
                                           args.shuffle_seed)
    report: dict = {
        "acts": args.acts,
        "n": int(Xn.shape[0]),
        "p": int(Xn.shape[1]),
        "candidates": [],
    }
    print(f"[setup] X {tuple(Xn.shape)} var={float(Xn.var()):.3f}", flush=True)

    for ckpt in args.official_sae or []:
        entry = official_topk_sae_eval(ckpt, X_raw, mu, sigma)
        report.setdefault("official", []).append(entry)
        print(f"[baseline] {entry}", flush=True)

    if args.vanilla_ckpt:
        report["vanilla"] = vanilla_from_ckpt(args.vanilla_ckpt, Xn)
        print(f"[baseline] {report['vanilla']}", flush=True)
    elif args.train_vanilla:
        report["vanilla"] = vanilla_train(
            Xn, args.vanilla_f or args.k, args.vanilla_topk,
            args.vanilla_steps, args.vanilla_lr, args.vanilla_batch, args.seed,
        )
        print(f"[baseline] {report['vanilla']}", flush=True)

    Xn64 = Xn.double().numpy()
    for topo in [t.strip() for t in args.topologies.split(",") if t.strip()]:
        d = args.d_atom if args.d_atom is not None else TOPOLOGY_DEFAULT_DIM.get(topo, 2)
        print(f"[fit] topology={topo} K={args.k} d={d} n_iter={args.n_iter}", flush=True)
        res = run_candidate(Xn64, args.k, topo, d, args.n_iter, args.seed,
                            args.assignment, args.top_k,
                            mu=mu.numpy().astype(np.float64),
                            sigma=sigma.numpy().astype(np.float64),
                            X_raw=X_raw.numpy().astype(np.float64))
        print(f"[fit] -> status={res['status']} ev={res.get('ev')} "
              f"reml={res.get('reml_score')} {res['seconds']:.1f}s", flush=True)
        report["candidates"].append(res)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
