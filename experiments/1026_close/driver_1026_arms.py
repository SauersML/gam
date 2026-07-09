#!/usr/bin/env python3
"""#1026 close-bar arms — manifold/hybrid SAE vs traditional TopK at 32K on creditscope.

The #1026 close bar (owner comment): "close if manifold SAE beats traditional SAE
at 32K dictionary size" on the creditscope Qwen3.5-35B-A3B activation set, held-out
EV, identical split. This driver runs ONE arm per invocation so every arm is a
short, independently schedulable GPU/CPU job; all arms derive the identical
train/test split from --seed, so numbers are directly comparable across jobs.

Arms:
  external_topk  — Gao-et-al. TopK SAE (torch, GPU), the "traditional SAE" bar.
  gam_flat       — gamfit.sae_manifold_fit sparse-code lane (our linear lane;
                   K>P routes to the linear REML schedule internally), held-out EV.
  curved_topk    — gamfit.sae_manifold_fit(assignment='topk') (CPU Rust core).
  torch_manifold — gamfit.torch.ManifoldSAE trained with Adam on the GPU.
  hybrid         — flat TopK at a reduced active budget + torch manifold on the
                   flat residual; combined recon scored at the MATCHED total
                   per-token active-scalar budget (flat k_lin scalars + t·(1+d)
                   curved scalars == --top-k). This is the dominance-argument
                   contestant: it strictly generalizes the flat dictionary.

EV = 1 − ||X_te − recon||²_F / ||X_te − mean_tr||²_F, train-mean baseline, always
scored on the untouched held-out split (never train — no overfit reporting).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def held_out_ev(x_test: np.ndarray, recon: np.ndarray, mean_train: np.ndarray) -> float:
    ssr = float(np.sum((x_test.astype(np.float64) - recon.astype(np.float64)) ** 2))
    sst = float(np.sum((x_test.astype(np.float64) - mean_train.astype(np.float64)[None, :]) ** 2))
    return 1.0 - ssr / max(sst, 1e-300)


def load_chunk_dir(chunk_dir: str, max_rows: int, seed: int) -> np.ndarray:
    """mmap chunk_*.npy float16 shards; deterministic max_rows subsample (the
    #1893 loader, verbatim semantics so splits agree with that harness)."""
    import glob

    files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npy")))
    if not files:
        raise SystemExit(f"no chunk_*.npy shards in {chunk_dir}")
    mms = [np.load(f, mmap_mode="r") for f in files]
    p = int(mms[0].shape[1])
    for m in mms:
        if int(m.shape[1]) != p:
            raise SystemExit(f"inconsistent p across shards: {p} vs {m.shape[1]}")
    counts = [int(m.shape[0]) for m in mms]
    offsets = np.cumsum([0] + counts)
    rng = np.random.default_rng(seed)
    take = min(max_rows, int(offsets[-1]))
    idx = np.sort(rng.choice(int(offsets[-1]), take, replace=False))
    parts = []
    for c, m in enumerate(mms):
        sel = idx[(idx >= offsets[c]) & (idx < offsets[c + 1])] - offsets[c]
        if sel.size:
            parts.append(np.asarray(m[sel], dtype=np.float32))
    out = np.concatenate(parts, axis=0)
    rng.shuffle(out)
    return np.ascontiguousarray(out, dtype=np.float32)


def make_split(X: np.ndarray, test_frac: float, seed: int):
    rng = np.random.default_rng(seed + 1)  # split stream separate from subsample stream
    n = X.shape[0]
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    te, tr = perm[:n_test], perm[n_test:]
    return np.ascontiguousarray(X[tr]), np.ascontiguousarray(X[te])


# --------------------------------------------------------------------------- #
def fit_external_topk(x_tr, x_te, mean_tr, *, K, top_k, steps, lr, bs, seed,
                      return_model=False, cosine_lr=False, collect=None):
    """Gao-et-al. TopK SAE with the standard training refinements (tied init,
    unit-norm decoder columns, pre-bias). The traditional-SAE bar."""
    import torch

    torch.set_float32_matmul_precision("high")  # TF32 on B200; applied to BOTH torch arms
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = x_tr.shape[1]
    xtr = torch.from_numpy(x_tr).to(dev)
    xte = torch.from_numpy(x_te).to(dev)
    b_dec = torch.nn.Parameter(xtr.mean(0).clone())
    W_enc = torch.nn.Parameter(torch.randn(K, p, device=dev) / p ** 0.5)
    W_dec = torch.nn.Parameter(W_enc.detach().clone())
    with torch.no_grad():
        W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    opt = torch.optim.Adam([W_enc, W_dec, b_dec], lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
             if cosine_lr else None)

    def encode(x):
        pre = (x - b_dec) @ W_enc.t()
        topv, topi = pre.topk(top_k, dim=1)
        return torch.relu(topv), topi

    def decode(vals, idx):
        return torch.einsum("bk,bkp->bp", vals, W_dec[idx]) + b_dec

    n = xtr.shape[0]
    t0 = time.perf_counter()
    last = t0
    for step in range(steps):
        i = torch.randint(0, n, (min(bs, n),), device=dev)
        xb = xtr[i]
        vals, idx = encode(xb)
        loss = ((decode(vals, idx) - xb) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        now = time.perf_counter()
        if step % 500 == 0 or step == steps - 1 or now - last >= 30.0:
            print(f"[external_topk] step {step+1}/{steps} loss={float(loss):.5f} "
                  f"({(step+1)/max(now-t0,1e-9):.1f} steps/s)", flush=True)
            last = now
    with torch.no_grad():
        outs = []
        for s in range(0, xte.shape[0], 8192):
            vals, idx = encode(xte[s:s+8192])
            outs.append(decode(vals, idx).float().cpu().numpy())
    recon = np.concatenate(outs, 0)
    ev = held_out_ev(x_te, recon, mean_tr)
    if collect is not None:
        # Stash the flat weights as numpy so the Eq-4 bits builder re-encodes
        # x_bits host-side (torch-free) with the SAME dictionary.
        collect["W_enc"] = W_enc.detach().float().cpu().numpy()
        collect["W_dec"] = W_dec.detach().float().cpu().numpy()
        collect["b_dec"] = b_dec.detach().float().cpu().numpy()
        collect["top_k_flat"] = int(top_k)
    if return_model:
        return ev, (encode, decode, dev)
    return ev


def fit_pca_bar(x_tr, x_te, mean_tr, *, ranks):
    """Affine PCA held-out EV at each rank — the linear-optimum yardstick
    (rank-k PCA is EV-optimal among ALL linear rank-k reconstructions)."""
    xc = x_tr - mean_tr[None, :]
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(w)[::-1]
    v = v[:, order]
    out = {}
    tc = x_te - mean_tr[None, :]
    for r in ranks:
        vr = v[:, :r]
        recon = (tc @ vr) @ vr.T + mean_tr[None, :]
        out[f"pca_ev_r{r}"] = held_out_ev(x_te, recon, mean_tr)
    return out


def fit_gam_flat(x_tr, x_te, mean_tr, *, K, top_k, max_epochs, seed, collect=None):
    import gamfit

    fit = gamfit.sae_manifold_fit(
        x_tr, K=K, assignment="softmax", top_k=top_k, n_iter=max_epochs)
    tr = fit.transform(x_te)
    recon = fit.reconstruct(tr.indices, tr.codes)
    if collect is not None:
        collect["flat_fit"] = fit
    return held_out_ev(x_te, recon, mean_tr), fit.explained_variance


def fit_curved_topk(x_tr, x_te, mean_tr, *, K, top_k, d_atom, topology, seed):
    import gamfit

    model = gamfit.sae_manifold_fit(
        x_tr, K=K, d_atom=d_atom, atom_topology=topology,
        assignment="topk", top_k=top_k, random_state=seed)
    recon = np.asarray(model.reconstruct(x_te), dtype=np.float32)
    return held_out_ev(x_te, recon, mean_tr)


def fit_torch_manifold(x_tr, x_te, mean_tr, *, atoms, target_k, d, steps, lr, bs,
                       seed, manifold, basis):
    """The GPU manifold lane (BSF-arena winner lane) fit directly on the data."""
    recon, _ = _torch_manifold_recon(
        x_tr, x_te, atoms=atoms, target_k=target_k, d=d, steps=steps, lr=lr,
        bs=bs, seed=seed, manifold=manifold, basis=basis)
    return held_out_ev(x_te, recon, mean_tr)


def _torch_manifold_recon(x_tr, x_te, *, atoms, target_k, d, steps, lr, bs, seed,
                          manifold, basis):
    import torch
    from gamfit.torch.manifold_sae import ManifoldSAE, ManifoldSAEConfig

    # Deploy-skew preflight: this arm needs API that older installed gamfit
    # wheels lack. Fail at entry with an actionable message instead of an
    # AttributeError minutes into a GPU job (node1 4c04d9ffeb94).
    for required in ("position_alignment_penalty",):
        if not hasattr(ManifoldSAE, required):
            import gamfit
            raise SystemExit(
                f"[torch_manifold] installed gamfit "
                f"{getattr(gamfit, '__version__', '?')} at "
                f"{os.path.dirname(gamfit.__file__)} predates "
                f"ManifoldSAE.{required}; upgrade the venv wheel to the current "
                f"build before submitting this arm."
            )

    torch.set_float32_matmul_precision("high")  # TF32 on B200; applied to BOTH torch arms
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if d == 1 and manifold == "product":
        manifold = "circle"  # product requires rank >= 2; rank-1 atoms are circles
    cfg = ManifoldSAEConfig(
        input_dim=x_tr.shape[1], n_atoms=atoms, intrinsic_rank=d,
        atom_manifold=manifold, atom_basis=basis,
        sparsity={"kind": "softmax_topk", "target_k": target_k},
        dtype=torch.float32,  # fp32 GPU training lane; metrics stay f64 numpy
    )
    model = ManifoldSAE(cfg).to(dev)
    xtr = torch.as_tensor(x_tr, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    last = t0
    model.train()
    for step in range(steps):
        i = rng.integers(0, xtr.shape[0], size=min(bs, xtr.shape[0]))
        batch = xtr[i]
        out = model(batch)
        loss = (torch.mean((out.x_hat - batch) ** 2)
                + model.sparsity.penalty(out.gate)
                + model.position_alignment_penalty())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        now = time.perf_counter()
        if step % 200 == 0 or step == steps - 1 or now - last >= 30.0:
            print(f"[torch_manifold] step {step+1}/{steps} loss={float(loss.detach()):.5f} "
                  f"({(step+1)/max(now-t0,1e-9):.2f} steps/s)", flush=True)
            last = now
    model.eval()
    outs = []
    with torch.no_grad():
        xte = torch.as_tensor(x_te, device=dev)
        for s in range(0, xte.shape[0], 8192):
            outs.append(model(xte[s:s+8192]).x_hat.float().cpu().numpy())
    return np.concatenate(outs, 0), model


def fit_hybrid(x_tr, x_te, mean_tr, *, K, top_k, curved_atoms, curved_k, d,
               flat_steps, curved_steps, lr, bs, seed, manifold, basis, collect=None):
    """Flat TopK at reduced budget k_lin + torch manifold on the flat residual.

    Matched per-token active-scalar budget: k_lin + curved_k·(1+d) == top_k.
    """
    k_lin = top_k - curved_k * (1 + d)
    if k_lin < 1:
        raise SystemExit(f"hybrid budget infeasible: top_k={top_k} curved_k={curved_k} d={d}")
    print(f"[hybrid] k_lin={k_lin} + curved_k={curved_k}*(1+{d}) == {top_k}", flush=True)
    ev_flat, (encode, decode, dev) = fit_external_topk(
        x_tr, x_te, mean_tr, K=K, top_k=k_lin, steps=flat_steps, lr=lr, bs=bs,
        seed=seed, return_model=True, collect=collect)
    print(f"[hybrid] flat tier ev={ev_flat:.4f} (k_lin={k_lin})", flush=True)
    import torch
    with torch.no_grad():
        def resid(x_np):
            parts = []
            xt = torch.from_numpy(x_np).to(dev)
            for s in range(0, xt.shape[0], 8192):
                vals, idx = encode(xt[s:s+8192])
                parts.append((xt[s:s+8192] - decode(vals, idx)).float().cpu().numpy())
            return np.concatenate(parts, 0)
        r_tr, r_te = resid(x_tr), resid(x_te)
    flat_recon_te = x_te - r_te
    curved_recon_r, curved_model = _torch_manifold_recon(
        r_tr, r_te, atoms=curved_atoms, target_k=curved_k, d=d, steps=curved_steps,
        lr=lr, bs=bs, seed=seed, manifold=manifold, basis=basis)
    combined = flat_recon_te + curved_recon_r
    if collect is not None:
        collect["k_lin"] = k_lin
        collect["r_te"] = r_te
        collect["curved_model"] = curved_model
        collect["dev"] = dev
        collect["recon_full"] = combined
    return held_out_ev(x_te, combined, mean_tr), ev_flat


def fit_hybrid_rust(x_tr, x_te, mean_tr, *, K, top_k, curved_K, curved_k, d,
                    topology, max_epochs, curved_rows, seed, collect=None):
    """ALL-RUST hybrid: gam sae_manifold_fit sparse-code (linear) tier at reduced
    actives + gam sae_manifold_fit curved TopK tier on the linear residual. Matched
    per-token active-scalar budget: k_lin + curved_k·(1+d) == top_k."""
    import gamfit

    k_lin = top_k - curved_k * (1 + d)
    if k_lin < 1:
        raise SystemExit(f"hybrid_rust budget infeasible: top_k={top_k} curved_k={curved_k} d={d}")
    print(f"[hybrid_rust] k_lin={k_lin} + curved_k={curved_k}*(1+{d}) == {top_k}", flush=True)
    t0 = time.perf_counter()
    flat = gamfit.sae_manifold_fit(
        x_tr, K=K, assignment="softmax", top_k=k_lin, n_iter=max_epochs)
    tr_tr = flat.transform(x_tr)
    tr_te = flat.transform(x_te)
    flat_recon_tr = flat.reconstruct(tr_tr.indices, tr_tr.codes)
    flat_recon_te = flat.reconstruct(tr_te.indices, tr_te.codes)
    ev_flat = held_out_ev(x_te, flat_recon_te, mean_tr)
    print(f"[hybrid_rust] flat tier ev={ev_flat:.4f} (k_lin={k_lin}, "
          f"{time.perf_counter()-t0:.0f}s)", flush=True)
    r_tr = np.ascontiguousarray(x_tr - flat_recon_tr)
    r_te = np.ascontiguousarray(x_te - flat_recon_te)
    rows = min(curved_rows, r_tr.shape[0])
    sub = np.random.default_rng(seed).choice(r_tr.shape[0], rows, replace=False)
    t1 = time.perf_counter()
    curved = gamfit.sae_manifold_fit(
        r_tr[sub], K=curved_K, d_atom=d, atom_topology=topology,
        assignment="topk", top_k=curved_k, random_state=seed)
    print(f"[hybrid_rust] curved tier fit {time.perf_counter()-t1:.0f}s", flush=True)
    curved_recon_te = np.asarray(curved.reconstruct(r_te), dtype=np.float32)
    combined = flat_recon_te + curved_recon_te
    if collect is not None:
        collect["flat_fit"] = flat
        collect["curved_model"] = curved
        collect["r_te"] = r_te
        collect["recon_full"] = combined
    ev = held_out_ev(x_te, combined, mean_tr)
    return ev, ev_flat


# --------------------------------------------------------------------------- #
def score_bits_for_arm(arm, collect, x_te, bits_max_rows, seed):
    """Build the arm's FittedFeaturizer on a test subsample and score Eq-4 bits.

    Returns a dict of ``bits_at_r2_*`` / ``code_bits_*`` / ``resid_bits_*`` /
    ``support_bits`` (namespaced ``bits_<key>``) plus the scorer provenance, or
    ``None`` for arms with no bits builder wired (gam#2233 task 3 wires the four
    dominance-argument contestants: external_topk, gam_flat, hybrid, hybrid_rust).
    """
    # Local imports so a non-bits run never pays the sibling-module import.
    import bits_eq4
    import arm_featurizers as af

    n_te = x_te.shape[0]
    rng = np.random.default_rng(seed + 7)  # bits stream separate from fit/split
    take = min(int(bits_max_rows), n_te)
    idx = np.sort(rng.choice(n_te, take, replace=False))
    x_bits = np.ascontiguousarray(x_te[idx])

    if arm == "external_topk":
        fitted = af.build_external_topk(
            x_bits, W_enc=collect["W_enc"], W_dec=collect["W_dec"],
            b_dec=collect["b_dec"], top_k=collect.get("top_k_flat"))
    elif arm == "gam_flat":
        fitted = af.build_gam_flat(x_bits, fit=collect["flat_fit"])
    elif arm == "hybrid":
        r_bits = np.ascontiguousarray(collect["r_te"][idx])
        recon_bits = np.ascontiguousarray(collect["recon_full"][idx])
        fitted = af.build_hybrid_torch(
            x_bits, r_bits, W_enc=collect["W_enc"], W_dec=collect["W_dec"],
            b_dec=collect["b_dec"], k_lin=collect["k_lin"],
            curved_model=collect["curved_model"], dev=collect["dev"],
            recon_full=recon_bits)
    elif arm == "hybrid_rust":
        r_bits = np.ascontiguousarray(collect["r_te"][idx])
        recon_bits = np.ascontiguousarray(collect["recon_full"][idx])
        fitted = af.build_hybrid_rust(
            x_bits, r_bits, flat_fit=collect["flat_fit"],
            curved_model=collect["curved_model"], recon_full=recon_bits)
    else:
        return None

    dl = bits_eq4.description_length(fitted, x_bits.astype(np.float64))
    out = {f"bits_{k}": v for k, v in dl.items()}
    out["bits_scorer"] = bits_eq4.scorer_source()
    out["bits_rows"] = int(take)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["external_topk", "gam_flat", "curved_topk",
                             "torch_manifold", "hybrid", "pca_bar", "hybrid_rust"])
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--max-rows", type=int, default=120_000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=32768)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--d-atom", type=int, default=2)
    ap.add_argument("--atom-topology", default="circle")
    ap.add_argument("--atom-manifold", default="product")
    ap.add_argument("--atom-basis", default="fourier")
    ap.add_argument("--curved-atoms", type=int, default=256)
    ap.add_argument("--curved-k", type=int, default=2)
    ap.add_argument("--curved-steps", type=int, default=6000)
    ap.add_argument("--curved-rows", type=int, default=20000,
                    help="hybrid_rust: row subsample for the curved-tier Rust fit")
    ap.add_argument("--cosine-lr", action="store_true",
                    help="cosine LR decay for the external bar (stronger baseline)")
    ap.add_argument("--bits", action="store_true",
                    help="also score gam#2233 Eq-4 description-length bits at fixed "
                         "R2 on the test split (the MDL scoreboard the crossover "
                         "theorem says curved atoms win by a wide margin)")
    ap.add_argument("--bits-max-rows", type=int, default=8192,
                    help="test-row subsample for Eq-4 bits (bounds the per-atom "
                         "SVD sweep at K=32768); the gate/contrib/recon are all "
                         "rebuilt on this same subsample")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results_1026.jsonl")
    args = ap.parse_args()

    X = load_chunk_dir(args.chunk_dir, args.max_rows, args.seed)
    x_tr, x_te = make_split(X, args.test_frac, args.seed)
    mean_tr = x_tr.mean(0)
    n, p = X.shape
    print(f"[#1026] arm={args.arm} N={n} p={p} train={x_tr.shape[0]} test={x_te.shape[0]} "
          f"K={args.K} top_k={args.top_k}", flush=True)

    t0 = time.perf_counter()
    extra: dict = {}
    # Handles for the Eq-4 bits scorer (gam#2233); only the four wired arms
    # populate it, and only when --bits is requested.
    collect: dict | None = {} if args.bits else None
    if args.arm == "external_topk":
        ev = fit_external_topk(x_tr, x_te, mean_tr, K=args.K, top_k=args.top_k,
                               steps=args.steps, lr=args.lr, bs=args.batch_size,
                               seed=args.seed, cosine_lr=args.cosine_lr,
                               collect=collect)
    elif args.arm == "gam_flat":
        ev, ev_train = fit_gam_flat(x_tr, x_te, mean_tr, K=args.K, top_k=args.top_k,
                                    max_epochs=args.max_epochs, seed=args.seed,
                                    collect=collect)
        extra["ev_train"] = ev_train
    elif args.arm == "curved_topk":
        ev = fit_curved_topk(x_tr, x_te, mean_tr, K=args.K, top_k=args.top_k,
                             d_atom=args.d_atom, topology=args.atom_topology,
                             seed=args.seed)
    elif args.arm == "torch_manifold":
        ev = fit_torch_manifold(x_tr, x_te, mean_tr, atoms=args.curved_atoms,
                                target_k=args.curved_k, d=args.d_atom,
                                steps=args.steps, lr=args.lr, bs=args.batch_size,
                                seed=args.seed, manifold=args.atom_manifold,
                                basis=args.atom_basis)
    elif args.arm == "hybrid_rust":
        ev, ev_flat = fit_hybrid_rust(x_tr, x_te, mean_tr, K=args.K, top_k=args.top_k,
                                      curved_K=args.curved_atoms, curved_k=args.curved_k,
                                      d=args.d_atom, topology=args.atom_topology,
                                      max_epochs=args.max_epochs,
                                      curved_rows=args.curved_rows, seed=args.seed,
                                      collect=collect)
        extra["ev_flat_tier"] = ev_flat
    elif args.arm == "pca_bar":
        extra = fit_pca_bar(x_tr, x_te, mean_tr, ranks=[16, 32, 64, 128, 512])
        ev = extra[f"pca_ev_r{args.top_k}"] if f"pca_ev_r{args.top_k}" in extra else extra["pca_ev_r32"]
    else:  # hybrid
        ev, ev_flat = fit_hybrid(x_tr, x_te, mean_tr, K=args.K, top_k=args.top_k,
                                 curved_atoms=args.curved_atoms, curved_k=args.curved_k,
                                 d=args.d_atom, flat_steps=args.steps,
                                 curved_steps=args.curved_steps, lr=args.lr,
                                 bs=args.batch_size, seed=args.seed,
                                 manifold=args.atom_manifold, basis=args.atom_basis,
                                 collect=collect)
        extra["ev_flat_tier"] = ev_flat

    if args.bits and collect is not None:
        bits = score_bits_for_arm(args.arm, collect, x_te, args.bits_max_rows, args.seed)
        if bits is not None:
            extra.update(bits)

    wall = time.perf_counter() - t0
    rec = {"issue": 1026, "arm": args.arm, "tag": args.tag, "N": n, "p": p,
           "K": args.K, "top_k": args.top_k, "d_atom": args.d_atom,
           "curved_atoms": args.curved_atoms, "curved_k": args.curved_k,
           "steps": args.steps, "seed": args.seed, "ev": ev, "wall_s": round(wall, 1),
           **extra}
    print("[#1026] RESULT " + json.dumps(rec), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
