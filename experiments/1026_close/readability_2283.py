#!/usr/bin/env python3
"""#2283 — is the Eq-4 bits-at-R2 comparison READABLE at the theorem's margin?

The crossover corollary predicts a ~20.1-bit advantage for the curved hybrid at
K=32768/top_k=32, and that margin lives ENTIRELY in the support term (the
faithful config makes the dictionary terms equal and a circle atom's span
s = d+1 zeroes the theorem's code term).  The scoreboard the margin is read on
also carries a residual term and a code term that the theorem says nothing
about, and both move with fit quality.

So before any hybrid row can decide the corollary, one number has to exist:
how much does bits@R2=0.99 move for reasons that are NOT the support term?

Two independent readings, both on the DECLARED #2502 document split:

  (A) SEED SPREAD.  Refit the same external TopK architecture at several seeds.
      Same data, same split, same rows scored, same everything except the torch
      seed.  The spread of bits@R2 is a lower bound on the noise any single
      paired row carries.

  (B) FIT-QUALITY SENSITIVITY.  Hold one fit fixed and rescale its codes by
      alpha, so the dictionary, the support pattern and the code dimensions are
      bit-identical and ONLY the reconstruction quality moves.  d(bits)/d(EV)
      converts "how close must two arms' EV be" into bits.

Everything is scored by the package's own Rust-backed Eq-4 core through
gamfit._description_length, i.e. the same scorer that produced the
authoritative external row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from crossover_theorem_check import support_code_bits


def _sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                        sort_keys=True, separators=(",", ":")).encode())
    h.update(b"\0")
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def held_out_ev(x_te: np.ndarray, recon: np.ndarray, mean_tr: np.ndarray) -> float:
    num = float(np.sum((x_te - recon) ** 2))
    den = float(np.sum((x_te - mean_tr[None, :]) ** 2))
    return 1.0 - num / den


# --------------------------------------------------------------------------- #
# Eq-4 featurizer for a flat TopK dictionary, with an explicit code scale.
# Mirrors experiments/1026_close/arm_featurizers.build_external_topk; the alpha
# knob multiplies the CODES, so recon, gate and every atom contribution move
# together and the support pattern / dictionary / code_dims are untouched.
# --------------------------------------------------------------------------- #
class _RowLazy:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, take):
        return self._fn(np.asarray(take))


def build_external_topk(x_bits, *, W_enc, W_dec, b_dec, top_k, alpha=1.0,
                        make_fitted_featurizer):
    pre = (x_bits.astype(np.float32) - b_dec[None, :]) @ W_enc.T
    K = W_dec.shape[0]
    top_k = min(int(top_k), K)
    topi = np.argpartition(-pre, top_k - 1, axis=1)[:, :top_k]
    rows = np.arange(pre.shape[0])[:, None]
    topv = np.maximum(pre[rows, topi], 0.0).astype(np.float32)
    del pre
    topv = topv * np.float32(alpha)

    n = topi.shape[0]
    gate = np.zeros((n, K), dtype=np.float32)
    np.add.at(gate, (rows, topi), np.abs(topv))

    def atom_contribution(g: int):
        dec_g = W_dec[g]

        def selected(take, g=g, dec_g=dec_g):
            ri = topi[take]
            rc = topv[take]
            signed = np.where(ri == g, rc, 0.0).sum(axis=1)
            return np.outer(signed, dec_g)

        return _RowLazy(selected)

    recon = np.einsum("nk,nkp->np", topv, W_dec[topi]) + b_dec[None, :]
    return make_fitted_featurizer(
        name="external_topk",
        gate=gate,
        atom_contribution=atom_contribution,
        code_dims=np.ones(K, dtype=int),
        dictionary_params=int(W_dec.size),
        recon=recon.astype(np.float64),
        fit_seconds=0.0,
    ), recon


# --------------------------------------------------------------------------- #
def fit_external_topk(x_tr, *, K, top_k, steps, lr, bs, seed, log):
    """Gao-et-al. TopK SAE: tied init, unit-norm decoder rows, pre-bias."""
    import torch

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = x_tr.shape[1]
    xtr = torch.from_numpy(x_tr).to(dev)
    b_dec = torch.nn.Parameter(xtr.mean(0).clone())
    W_enc = torch.nn.Parameter(torch.randn(K, p, device=dev) / p ** 0.5)
    W_dec = torch.nn.Parameter(W_enc.detach().clone())
    with torch.no_grad():
        W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    opt = torch.optim.Adam([W_enc, W_dec, b_dec], lr=lr)

    def encode(x):
        pre = (x - b_dec) @ W_enc.t()
        topv, topi = pre.topk(top_k, dim=1)
        return torch.relu(topv), topi

    def decode(vals, idx):
        return torch.einsum("bk,bkp->bp", vals, W_dec[idx]) + b_dec

    n = xtr.shape[0]
    t0 = time.perf_counter()
    for step in range(steps):
        i = torch.randint(0, n, (min(bs, n),), device=dev)
        xb = xtr[i]
        vals, idx = encode(xb)
        loss = ((decode(vals, idx) - xb) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        if step % 1000 == 0 or step == steps - 1:
            log(f"[seed {seed}] step {step+1}/{steps} loss={float(loss):.6f} "
                f"({(step+1)/max(time.perf_counter()-t0,1e-9):.1f} steps/s)")
    wall = time.perf_counter() - t0
    out = {
        "W_enc": W_enc.detach().float().cpu().numpy(),
        "W_dec": W_dec.detach().float().cpu().numpy(),
        "b_dec": b_dec.detach().float().cpu().numpy(),
        "fit_seconds": wall,
    }
    del xtr, W_enc, W_dec, b_dec, opt
    torch.cuda.empty_cache()
    return out


def encode_recon_host(x, *, W_enc, W_dec, b_dec, top_k, block=8192):
    """TopK reconstruction of arbitrarily many rows.

    The encode is a (rows x p) @ (p x K) product with K = 32768; on a contended
    host that is the single most expensive step in the run, and it is the same
    arithmetic the fit just did on the device. Route it to the GPU when one is
    there, and fall back to numpy otherwise so the function still runs on a
    device-free box.
    """
    try:
        import torch

        if torch.cuda.is_available():
            dev = "cuda"
            We = torch.from_numpy(W_enc).to(dev)
            Wd = torch.from_numpy(W_dec).to(dev)
            bd = torch.from_numpy(b_dec).to(dev)
            outs = []
            with torch.no_grad():
                for s in range(0, x.shape[0], block):
                    xb = torch.from_numpy(
                        np.ascontiguousarray(x[s:s + block],
                                             dtype=np.float32)).to(dev)
                    pre = (xb - bd) @ We.t()
                    topv, topi = pre.topk(top_k, dim=1)
                    topv = torch.relu(topv)
                    rec = torch.einsum("bk,bkp->bp", topv, Wd[topi]) + bd
                    outs.append(rec.float().cpu().numpy())
                    del xb, pre, topv, topi, rec
            del We, Wd, bd
            torch.cuda.empty_cache()
            return np.concatenate(outs, 0)
    except Exception:  # noqa: BLE001
        pass
    outs = []
    for s in range(0, x.shape[0], block):
        xb = x[s:s + block].astype(np.float32)
        pre = (xb - b_dec[None, :]) @ W_enc.T
        topi = np.argpartition(-pre, top_k - 1, axis=1)[:, :top_k]
        rows = np.arange(xb.shape[0])[:, None]
        topv = np.maximum(pre[rows, topi], 0.0)
        outs.append(np.einsum("nk,nkp->np", topv, W_dec[topi]) + b_dec[None, :])
    return np.concatenate(outs, 0)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--split-module",
                    default=os.path.expanduser("~/i2502-baselines"))
    ap.add_argument("--repo", default=os.path.expanduser("~/saefrontier2"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--K", type=int, default=32768)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=120000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--bits-rows", type=int, default=24000)
    ap.add_argument("--full-split-seed", type=int, default=0,
                    help="which seed additionally gets a whole-held-out-split score")
    ap.add_argument("--alphas", default="0.90,0.95,0.99,1.0,1.02")
    ap.add_argument("--step-ladder", default="",
                    help="extra REAL fits at seed --full-split-seed with these "
                         "training budgets; gives d(bits)/d(EV) from genuine "
                         "refits rather than an analytic code rescale")
    ap.add_argument("--amortization-horizon", type=int, default=120000)
    ap.add_argument("--skip-whole-split-score", action="store_true",
                    help="the whole-split score costs one more full Eq-4 pass "
                         "at 2.6x the rows; the seed spread does not need it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    sys.path.insert(0, args.split_module)
    from issue_2502_doc_split import (SPLIT_VERSION, row_side, split_manifest)

    from gamfit._description_length import FittedFeaturizer, description_length

    def mk(**kw):
        return FittedFeaturizer(**kw)

    import gamfit
    import torch

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    prov = {
        "node": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "gamfit": gamfit.__version__,
        "CUDA_VISIBLE_DEVICES": cuda_visible,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "cuda_runtime": torch.version.cuda,
    }
    log(f"provenance {json.dumps(prov)}")

    # ---------------- data + declared split ----------------
    H = Path(args.harvest)
    doc_ids = np.load(H / "doc_ids.npy")
    man = split_manifest(doc_ids)
    log(f"split {SPLIT_VERSION} hash={man['split_hash']} "
        f"test_rows={man['n_rows_test']} train_rows={man['n_rows_train']}")

    side = row_side(doc_ids)                      # True == held out
    acts = np.load(H / f"resid_L{args.layer}.npy", mmap_mode="r")
    log(f"harvest resid_L{args.layer} shape={acts.shape} dtype={acts.dtype}")

    train_idx_all = np.flatnonzero(~side)
    test_idx = np.flatnonzero(side)
    rng = np.random.default_rng(12345)            # fixed, NOT the fit seed
    take = min(args.n_train, train_idx_all.size)
    train_idx = np.sort(rng.choice(train_idx_all, take, replace=False))

    x_tr = np.ascontiguousarray(acts[train_idx], dtype=np.float32)
    x_te = np.ascontiguousarray(acts[test_idx], dtype=np.float32)
    mean_tr = x_tr.mean(0)
    log(f"train {x_tr.shape} test {x_te.shape}")

    brng = np.random.default_rng(7)
    bsel = np.sort(brng.choice(x_te.shape[0],
                               min(args.bits_rows, x_te.shape[0]),
                               replace=False))
    x_bits = np.ascontiguousarray(x_te[bsel])

    data_ident = {
        "split_version": SPLIT_VERSION,
        "split_hash": man["split_hash"],
        "harvest_doc_digest": man.get("harvest_doc_digest"),
        "layer": args.layer,
        "train_row_ids_sha256": _sha256_array(train_idx),
        "test_row_ids_sha256": _sha256_array(test_idx),
        "bits_row_ids_sha256": _sha256_array(test_idx[bsel]),
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "n_bits_rows": int(x_bits.shape[0]),
        "p": int(x_tr.shape[1]),
    }
    log(f"data identity {json.dumps(data_ident)}")

    r2_targets = (0.8, 0.9, 0.95, 0.99)
    records = []

    def emit(rec):
        rec["provenance"] = prov
        rec["data_identity"] = data_ident
        rec["config"] = {"K": args.K, "top_k": args.top_k,
                         "steps": args.steps, "lr": args.lr, "bs": args.bs,
                         "amortization_horizon": args.amortization_horizon}
        records.append(rec)
        with open(outp, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # Checkpoint/resume. This box has killed the run twice (one CUDA OOM from a
    # foreign tenant, one unattributed SIGTERM), each time AFTER paying for a
    # fit. The fits are the expensive head and the scoring is the cheap tail, so
    # the head gets persisted: a restart reloads any fit it already paid for and
    # goes straight to the part that was interrupted.
    ckdir = outp.parent / "fits"
    ckdir.mkdir(parents=True, exist_ok=True)

    def fit_cached(seed, steps):
        path = ckdir / (f"topk_K{args.K}_k{args.top_k}_n{args.n_train}"
                        f"_s{steps}_seed{seed}.npz")
        if path.exists():
            z = np.load(path)
            log(f"resumed fit from {path.name}")
            return {"W_enc": z["W_enc"], "W_dec": z["W_dec"],
                    "b_dec": z["b_dec"], "fit_seconds": float(z["fit_seconds"])}
        w = fit_external_topk(x_tr, K=args.K, top_k=args.top_k, steps=steps,
                              lr=args.lr, bs=args.bs, seed=seed, log=log)
        # np.savez APPENDS ".npz" unless the name already ends in it, so a
        # ".npz.part" staging name becomes ".npz.part.npz" and the rename below
        # then fails on a file that was never created. Stage as ".part.npz".
        tmp = path.with_suffix(".part.npz")
        np.savez(tmp, W_enc=w["W_enc"], W_dec=w["W_dec"], b_dec=w["b_dec"],
                 fit_seconds=w["fit_seconds"])
        tmp.replace(path)
        return w

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fits = {}
    for seed in seeds:
        log(f"=== external TopK seed {seed} ===")
        w = fit_cached(seed, args.steps)
        recon_te = encode_recon_host(x_te, W_enc=w["W_enc"], W_dec=w["W_dec"],
                                     b_dec=w["b_dec"], top_k=args.top_k)
        ev = held_out_ev(x_te, recon_te, mean_tr)
        del recon_te
        log(f"[seed {seed}] held-out EV (whole split) = {ev:.6f} "
            f"fit {w['fit_seconds']:.1f}s")

        fitted, recon_b = build_external_topk(
            x_bits, W_enc=w["W_enc"], W_dec=w["W_dec"], b_dec=w["b_dec"],
            top_k=args.top_k, alpha=1.0, make_fitted_featurizer=mk)
        ev_bits = held_out_ev(x_bits, recon_b, mean_tr)
        t0 = time.perf_counter()
        bits = description_length(
            fitted, x_bits, amortization_horizon=args.amortization_horizon,
            r2_targets=r2_targets)
        log(f"[seed {seed}] scored in {time.perf_counter()-t0:.1f}s "
            f"bits@0.99={bits.get('bits_at_r2_0.99')}")
        emit({"record": "seed_row", "seed": seed, "alpha": 1.0,
              "ev_whole_split": ev, "ev_bits_rows": ev_bits,
              "fit_seconds": w["fit_seconds"],
              "bits_rows": int(x_bits.shape[0]), "bits": bits})
        del fitted, recon_b
        if seed == args.full_split_seed:
            fits[seed] = w
        else:
            del w

    # ---------------- (B1) fit-quality ladder, REAL refits ----------------
    seed0 = args.full_split_seed
    ladder = [int(s) for s in args.step_ladder.split(",") if s.strip()]
    for budget in ladder:
        log(f"=== ladder fit seed {seed0} steps {budget} ===")
        w = fit_cached(seed0, budget)
        recon_te = encode_recon_host(x_te, W_enc=w["W_enc"], W_dec=w["W_dec"],
                                     b_dec=w["b_dec"], top_k=args.top_k)
        ev = held_out_ev(x_te, recon_te, mean_tr)
        del recon_te
        fitted, recon_b = build_external_topk(
            x_bits, W_enc=w["W_enc"], W_dec=w["W_dec"], b_dec=w["b_dec"],
            top_k=args.top_k, alpha=1.0, make_fitted_featurizer=mk)
        ev_bits = held_out_ev(x_bits, recon_b, mean_tr)
        bits = description_length(
            fitted, x_bits, amortization_horizon=args.amortization_horizon,
            r2_targets=r2_targets)
        log(f"[ladder {budget}] EV(whole)={ev:.6f} EV(bits rows)={ev_bits:.6f} "
            f"bits@0.99={bits.get('bits_at_r2_0.99')}")
        emit({"record": "ladder_row", "seed": seed0, "steps": budget,
              "ev_whole_split": ev, "ev_bits_rows": ev_bits,
              "fit_seconds": w["fit_seconds"],
              "bits_rows": int(x_bits.shape[0]), "bits": bits})
        del fitted, recon_b, w

    # ---------------- (B2) analytic code-scale control ----------------
    if seed0 in fits:
        w = fits[seed0]
        for alpha in [float(a) for a in args.alphas.split(",") if a.strip()]:
            fitted, recon_a = build_external_topk(
                x_bits, W_enc=w["W_enc"], W_dec=w["W_dec"], b_dec=w["b_dec"],
                top_k=args.top_k, alpha=alpha, make_fitted_featurizer=mk)
            ev_bits = held_out_ev(x_bits, recon_a, mean_tr)
            bits = description_length(
                fitted, x_bits, amortization_horizon=args.amortization_horizon,
                r2_targets=r2_targets)
            log(f"[alpha {alpha}] ev(bits rows)={ev_bits:.6f} "
                f"bits@0.99={bits.get('bits_at_r2_0.99')}")
            emit({"record": "alpha_row", "seed": seed0, "alpha": alpha,
                  "ev_bits_rows": ev_bits, "bits_rows": int(x_bits.shape[0]),
                  "bits": bits})
            del fitted, recon_a

        # ---------------- whole-held-out-split score ----------------
        if not args.skip_whole_split_score:
            log("=== whole held-out split score (seed %d) ===" % seed0)
            fitted, _ = build_external_topk(
                x_te, W_enc=w["W_enc"], W_dec=w["W_dec"], b_dec=w["b_dec"],
                top_k=args.top_k, alpha=1.0, make_fitted_featurizer=mk)
            t0 = time.perf_counter()
            bits = description_length(
                fitted, x_te, amortization_horizon=args.amortization_horizon,
                r2_targets=r2_targets)
            log(f"[whole split] scored in {time.perf_counter()-t0:.1f}s "
                f"bits@0.99={bits.get('bits_at_r2_0.99')}")
            emit({"record": "whole_split_row", "seed": seed0, "alpha": 1.0,
                  "bits_rows": int(x_te.shape[0]), "bits": bits})
            del fitted

    # ---------------- the theorem's own margin at this config ----------------
    margins = {}
    for charts in (8, 32, 256):
        k_flat = args.K - 3 * charts       # faithful: k_flat*P + charts*3*P == K*P
        curved_k = 2
        margins[str(charts)] = (
            support_code_bits(args.K, args.top_k)
            - support_code_bits(k_flat + charts, args.top_k - curved_k))
    emit({"record": "theorem_margin", "support_bits_external":
          support_code_bits(args.K, args.top_k), "margin_by_charts": margins})
    log(f"theorem margin by chart count: {json.dumps(margins)}")

    log(f"wrote {len(records)} records to {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
