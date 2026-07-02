"""32K-dictionary head-to-head: block-sparse tiered dictionary vs a matched
TopK-SAE, at deployment scale, over a sharded residual-stream corpus.

Three arms, all at MATCHED decoder-scalar budget (``K·d``) and matched active-code
count (``L0``), same epochs, same held-out split:

  * **ARM A — block-sparse T1** (:class:`gamfit.BlockSparseDictStream`): ``n_blocks``
    blocks of ``block_size`` orthonormal atoms (``K = n_blocks·block_size``),
    streamed multi-epoch over the shards, then ``to_fit(sample)`` →
    ``seed_manifest`` (the Tier-2 hand-off).
  * **ARM B — TopK-SAE** (``K`` atoms, minimal in-script torch, decoder rows
    unit-norm, top-``L0`` activation): the direction baseline at the SAME budget.
  * **ARM C — curved Tier-2** on the top-``N`` evidence blocks from Arm A's
    manifest (one K=1 chart per block via the ``compose_tiers`` recipe, guarded
    with a linear-lift fallback).

Reported to one results JSON: **held-out EV** per arm, **bits/token** (MDL, from
the manifest's featurizer rows), per-block **stable rank + utilisation**, and — the
number that settles the deployment-scale ``f*`` claim (Manifold-SAE#5) — the
**per-feature FIRING COUNTS** distribution. A **cross-corpus transfer** row scores
the frozen dictionaries on a second (e.g. creditscope) shard set.

HONEST SCOPING (baked into the JSON): at ~1.4M tokens a 32K dictionary sees only
~``1.4e6·L0/K`` firings per feature (≈ tens) — an UNDERTRAINED-but-SYMMETRIC regime
(both arms pay it equally); the head-to-head is a fair relative comparison, not a
converged-dictionary claim.

Usage (on MSI, after ``maturin develop --release``)::

    python dict32k_head_to_head.py --shards-dir $SCRATCH/qwen_l17_shards \
        --eval-shards-dir $SCRATCH/creditscope_shards \
        --n-blocks 16384 --block-size 2 --block-topk 4 --epochs 4 \
        --out $SCRATCH/dict32k_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_shard_io import load_shards  # noqa: E402


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def explained_variance(x: np.ndarray, recon: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0 else 0.0


def firing_summary(counts: np.ndarray, n_tokens: int) -> dict:
    """Per-feature firing-count distribution — the deployment-scale f* evidence."""
    c = np.asarray(counts, dtype=np.float64)
    nz = c[c > 0]
    return {
        "n_features": int(c.size),
        "n_tokens": int(n_tokens),
        "total_firings": int(c.sum()),
        "mean_firings_per_feature": round(float(c.mean()), 3),
        "median_firings_per_feature": round(float(np.median(c)), 3),
        "p10": round(float(np.percentile(c, 10)), 3),
        "p90": round(float(np.percentile(c, 90)), 3),
        "max": int(c.max()) if c.size else 0,
        "frac_dead": round(float((c == 0).mean()), 4),
        "frac_ge_10": round(float((c >= 10).mean()), 4),
        "frac_ge_100": round(float((c >= 100).mean()), 4),
        "alive_mean_firings": round(float(nz.mean()) if nz.size else 0.0, 3),
    }


# --------------------------------------------------------------------------- #
# Streaming split (train rows before cutoff; held-out test rows after)
# --------------------------------------------------------------------------- #
def _total_rows(reader) -> int:
    return int(getattr(reader, "total_tokens", len(reader)))


def train_batches(reader, batch: int, cutoff: int) -> Iterator[np.ndarray]:
    """Yield train batches (rows ``[0, cutoff)``) from a fresh pass over the shards."""
    seen = 0
    for b in reader.batches(batch):
        if seen >= cutoff:
            break
        if seen + b.shape[0] <= cutoff:
            yield b
            seen += b.shape[0]
        else:
            yield b[: cutoff - seen]
            seen = cutoff
            break


def read_rows_after(reader, batch: int, cutoff: int, cap: int | None = None) -> np.ndarray:
    """Read held-out rows ``[cutoff, cutoff+cap)`` into memory (the test set)."""
    seen = 0
    parts: list[np.ndarray] = []
    got = 0
    for b in reader.batches(batch):
        if seen + b.shape[0] <= cutoff:
            seen += b.shape[0]
            continue
        start = max(0, cutoff - seen)
        take = b[start:]
        if cap is not None and got + take.shape[0] > cap:
            take = take[: cap - got]
        parts.append(np.ascontiguousarray(take, dtype=np.float32))
        got += take.shape[0]
        seen += b.shape[0]
        if cap is not None and got >= cap:
            break
    return np.concatenate(parts, 0) if parts else np.zeros((0, reader.d_model), np.float32)


def read_sample(reader, batch: int, cap: int) -> np.ndarray:
    """First ``cap`` rows (frame seed + to_fit sample + TopK train sample)."""
    return read_rows_after(reader, batch, 0, cap)


# --------------------------------------------------------------------------- #
# ARM A — block-sparse tiered dictionary
# --------------------------------------------------------------------------- #
def run_arm_a(reader, test: np.ndarray, sample: np.ndarray, cfg: dict, out_prefix: str) -> dict:
    import gamfit

    t0 = time.time()
    stream = gamfit.block_sparse_dictionary_fit_begin(
        sample,
        cfg["n_blocks"],
        block_size=cfg["block_size"],
        block_topk=cfg["block_topk"],
        max_epochs=cfg["epochs"],
        minibatch=cfg["minibatch"],
        block_tile=cfg["block_tile"],
        aux_k=cfg["aux_k"],
    )
    cutoff = cfg["train_cutoff"]
    for epoch in range(cfg["epochs"]):
        rows = 0
        for b in train_batches(reader, cfg["batch"], cutoff):
            stream.partial_fit(b)
            rows += b.shape[0]
        ep = stream.end_epoch()
        print(f"[A] epoch {ep['epoch']} rows={rows} EV={ep['explained_variance']:.4f} "
              f"gamma={ep['gamma']:.3f} dead={ep['dead']} revived={ep['revived']}", flush=True)
        if ep["converged"]:
            break
    art = stream.finalize()

    # Held-out reconstruction + firing counts via the frozen frames.
    fit_test = art.to_fit(test)
    ev_test = explained_variance(test, fit_test.fitted)
    blocks_test = np.asarray(fit_test.blocks)  # (n_test, block_topk)
    counts = np.bincount(blocks_test.reshape(-1), minlength=art.n_blocks).astype(np.int64)

    # Seed manifest (Tier-2 hand-off) from a sample; MDL from its featurizer rows.
    fit_sample = art.to_fit(sample)
    manifest = fit_sample.seed_manifest(sample, n_basis_chart=cfg["n_basis_chart"])
    _save_manifest(manifest, fit_sample, sample, out_prefix)
    mdl = _score_mdl(manifest)

    return {
        "arm": "A_block_sparse",
        "n_features": int(art.n_blocks),
        "K_atoms": int(art.n_blocks * art.block_size),
        "ev_test": round(ev_test, 4),
        "ev_train_final": round(float(art.explained_variance), 4),
        "gamma": round(float(art.gamma), 4),
        "epochs": int(art.epochs),
        "converged": bool(art.converged),
        "firing": firing_summary(counts, test.shape[0]),
        "block_utilization_mean": round(float(np.mean(art.block_utilization)), 5),
        "block_stable_rank_mean": round(float(np.mean(art.block_stable_rank)), 4),
        "block_stable_rank_p90": round(float(np.percentile(art.block_stable_rank, 90)), 4),
        "mdl_bits_per_token": mdl,
        "seeds_prefix": out_prefix,
        "wall_s": round(time.time() - t0, 1),
        "_artifact": art,  # in-memory only; stripped before JSON dump
    }


def _save_manifest(manifest: dict, fit, sample: np.ndarray, out_prefix: str) -> None:
    try:
        from compose_tiers import emit_block_seed_manifest

        emit_block_seed_manifest(fit, sample, out_prefix)
    except Exception as exc:  # pragma: no cover
        print(f"[A] manifest emit skipped: {type(exc).__name__}: {exc}", flush=True)


def _score_mdl(manifest: dict) -> float | None:
    for cand in ("/Users/user/Manifold-SAE/experiments/mdl_ladder",
                 str(Path(__file__).resolve().parent.parent.parent / "Manifold-SAE"
                     / "experiments" / "mdl_ladder")):
        if os.path.exists(os.path.join(cand, "mdl.py")):
            sys.path.insert(0, cand)
            try:
                import mdl

                resp = mdl.score_json({"delta2": None, "l_param_bits": None,
                                       "featurizers": manifest["mdl_featurizers"]})
                return round(float(sum(r.get("bits_per_token", 0.0) for r in resp.get("rows", []))), 4)
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# ARM B — TopK-SAE at the matched budget (minimal in-script torch)
# --------------------------------------------------------------------------- #
def run_arm_b(reader, test: np.ndarray, sample: np.ndarray, cfg: dict) -> dict:
    import torch

    torch.manual_seed(cfg["seed"])
    d = int(reader.d_model)
    k_atoms = cfg["n_blocks"] * cfg["block_size"]  # matched decoder scalars K·d
    l0 = cfg["block_topk"] * cfg["block_size"]     # matched active-code count
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = _TopKSAE(d, k_atoms, l0).to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    b_dec_init = torch.tensor(sample.mean(0), dtype=torch.float64, device=dev)
    with torch.no_grad():
        model.b_dec.copy_(b_dec_init)

    t0 = time.time()
    cutoff = cfg["train_cutoff"]
    for epoch in range(cfg["epochs"]):
        rows, loss_acc = 0, 0.0
        for b in train_batches(reader, cfg["batch"], cutoff):
            xb = torch.tensor(np.ascontiguousarray(b, dtype=np.float64), device=dev)
            xhat, _ = model(xb)
            loss = ((xhat - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_decoder()
            rows += b.shape[0]
            loss_acc = float(loss.detach())
        print(f"[B] epoch {epoch} rows={rows} recon={loss_acc:.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(np.ascontiguousarray(test, dtype=np.float64), device=dev)
        xhat, topi = model(xt)
        ev_test = explained_variance(test, xhat.cpu().numpy())
        counts = np.bincount(topi.cpu().numpy().reshape(-1), minlength=k_atoms).astype(np.int64)
    mdl = _topk_mdl(cfg, test, k_atoms, l0, d, counts)
    return {
        "arm": "B_topk_sae",
        "n_features": int(k_atoms),
        "K_atoms": int(k_atoms),
        "L0": int(l0),
        "ev_test": round(ev_test, 4),
        "firing": firing_summary(counts, test.shape[0]),
        "mdl_bits_per_token": mdl,
        "wall_s": round(time.time() - t0, 1),
        "_model": model,  # in-memory only
    }


class _TopKSAE:  # thin torch module built lazily so import works without torch
    def __new__(cls, d, k, l0):
        import torch
        import torch.nn as nn

        class TopKSAE(nn.Module):
            def __init__(self, d, k, l0):
                super().__init__()
                self.l0 = l0
                g = torch.Generator().manual_seed(0)
                w = torch.randn(k, d, generator=g, dtype=torch.float64)
                w = w / w.norm(dim=1, keepdim=True)
                self.W_dec = nn.Parameter(w)            # (K, d) unit-norm rows
                self.W_enc = nn.Parameter(w.t().clone())  # (d, K) tied-ish init
                self.b_dec = nn.Parameter(torch.zeros(d, dtype=torch.float64))

            def forward(self, x):
                pre = torch.relu((x - self.b_dec) @ self.W_enc)  # (n, K)
                topv, topi = pre.topk(self.l0, dim=1)
                z = torch.zeros_like(pre).scatter_(1, topi, topv)
                xhat = z @ self.W_dec + self.b_dec
                return xhat, topi

            @torch.no_grad()
            def normalize_decoder(self):
                self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

        return TopKSAE(d, k, l0)


def _topk_mdl(cfg, test, k_atoms, l0, d, counts) -> float | None:
    """Bits/token for the TopK-SAE via the MDL scorer: one direction featurizer per
    firing budget (uniform), at the same task floor the block arm uses."""
    for cand in ("/Users/user/Manifold-SAE/experiments/mdl_ladder",):
        if os.path.exists(os.path.join(cand, "mdl.py")):
            sys.path.insert(0, cand)
            try:
                import mdl

                total_var = float(((test - test.mean(0)) ** 2).sum() / max(test.shape[0], 1))
                feats = [{
                    "name": "topk-sae", "kind": "direction",
                    "total_var": max(total_var, 1e-9), "n_tokens": int(test.shape[0]),
                    "n_firings": int(counts.sum()), "n_params": int(k_atoms * d),
                    "g_dict": int(k_atoms), "k_active": int(l0),
                    "coded_dim": int(l0), "ev": 0.5,
                }]
                resp = mdl.score_json({"delta2": None, "l_param_bits": None, "featurizers": feats})
                return round(float(sum(r.get("bits_per_token", 0.0) for r in resp.get("rows", []))), 4)
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# ARM C — curved Tier-2 on the top-N evidence blocks
# --------------------------------------------------------------------------- #
def run_arm_c(sample: np.ndarray, seeds_prefix: str, cfg: dict, arm_a: dict) -> dict:
    t0 = time.time()
    try:
        from compose_tiers import compose_charts_from_manifest

        report, _ = compose_charts_from_manifest(
            seeds_prefix, X=sample, chart_n_iter=cfg["chart_n_iter"],
            chart_topology="circle", chart_d_atom=2, random_state=cfg["seed"],
        )
        n_charts = sum(1 for r in report["per_block"] if r.get("chart_status") == "CONVERGED")
        return {
            "arm": "C_curved_t2",
            "n_blocks_charted": int(report["n_blocks"]),
            "n_charts_converged": int(n_charts),
            "composed_ambient_ev": report.get("composed_ambient_ev"),
            "wall_s": round(time.time() - t0, 1),
        }
    except Exception as exc:
        return {"arm": "C_curved_t2", "status": f"{type(exc).__name__}",
                "error": str(exc)[:200], "wall_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# Cross-corpus transfer
# --------------------------------------------------------------------------- #
def transfer_eval(arm_a: dict, arm_b: dict, eval_dir: str, cfg: dict) -> dict:
    reader = load_shards(eval_dir)
    x = read_sample(reader, cfg["batch"], cfg["transfer_cap"])
    out: dict = {"eval_dir": eval_dir, "n_eval_rows": int(x.shape[0])}
    if x.shape[0] == 0:
        return out
    art = arm_a.get("_artifact")
    if art is not None and x.shape[1] == art.decoder.shape[1]:
        out["arm_A_transfer_ev"] = round(explained_variance(x, art.to_fit(x).fitted), 4)
    model = arm_b.get("_model")
    if model is not None:
        import torch

        with torch.no_grad():
            xt = torch.tensor(np.ascontiguousarray(x, dtype=np.float64),
                              device=next(model.parameters()).device)
            xhat, _ = model(xt)
        out["arm_B_transfer_ev"] = round(explained_variance(x, xhat.cpu().numpy()), 4)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def drive(args) -> dict:
    reader = load_shards(args.shards_dir)
    total = _total_rows(reader)
    d_model = int(reader.d_model)
    test_cap = min(args.held_out, max(1, int(total * 0.1)))
    train_cutoff = max(1, total - test_cap)
    print(f"[driver] corpus {total} tokens, d_model={d_model}; "
          f"train<{train_cutoff}, held-out {test_cap}", flush=True)

    test = read_rows_after(reader, args.batch, train_cutoff, test_cap)
    sample = read_sample(reader, args.batch, args.sample_rows)

    l0 = args.block_topk * args.block_size
    cfg = {
        "n_blocks": args.n_blocks, "block_size": args.block_size,
        "block_topk": args.block_topk, "epochs": args.epochs, "batch": args.batch,
        "minibatch": args.minibatch, "block_tile": args.block_tile, "aux_k": args.aux_k,
        "n_basis_chart": args.n_basis_chart, "chart_n_iter": args.chart_n_iter,
        "lr": args.lr, "seed": args.seed, "train_cutoff": train_cutoff,
        "transfer_cap": args.transfer_cap,
    }
    results: dict = {
        "config": {**vars(args), "L0": l0, "d_model": d_model, "total_tokens": total,
                   "train_cutoff": train_cutoff, "held_out_rows": int(test.shape[0]),
                   "K_atoms": args.n_blocks * args.block_size},
        "honest_scoping": {
            "total_tokens": total,
            "K_features": args.n_blocks * args.block_size,
            "L0": l0,
            "mean_firings_per_feature_expected": round(total * l0 / max(args.n_blocks * args.block_size, 1), 2),
            "note": "Deployment-scale but UNDERTRAINED: ~tens of firings/feature at "
                    "32K. Both arms pay this equally (matched budget/L0/epochs), so "
                    "the head-to-head is a fair relative comparison, not a converged "
                    "dictionary. Firing counts are reported so the f* claim "
                    "(Manifold-SAE#5) is evidenced, not assumed.",
        },
        "arms": {},
    }
    seeds_prefix = os.path.join(os.path.dirname(args.out) or ".",
                                f"dict32k_seeds_{os.getpid()}")

    def _save():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dump = dict(results)
        dump["arms"] = {k: _strip(v) for k, v in results["arms"].items()}
        Path(args.out).write_text(json.dumps(dump, indent=2, default=float))

    for name, fn in (
        ("A", lambda: run_arm_a(reader, test, sample, cfg, seeds_prefix)),
        ("B", lambda: run_arm_b(reader, test, sample, cfg)),
    ):
        try:
            rec = fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rec = {"arm": name, "status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        results["arms"][rec.get("arm", name)] = rec
        _save()
        print(f"[driver] arm {name}: EV_test={rec.get('ev_test')} "
              f"bits/tok={rec.get('mdl_bits_per_token')}", flush=True)

    # Arm C consumes Arm A's manifest.
    arm_a = results["arms"].get("A_block_sparse", {})
    if arm_a.get("seeds_prefix"):
        results["arms"]["C_curved_t2"] = run_arm_c(sample, arm_a["seeds_prefix"], cfg, arm_a)
        _save()

    # Cross-corpus transfer.
    if args.eval_shards_dir:
        try:
            results["transfer"] = transfer_eval(arm_a, results["arms"].get("B_topk_sae", {}),
                                                args.eval_shards_dir, cfg)
        except Exception as exc:
            results["transfer"] = {"status": f"{type(exc).__name__}", "error": str(exc)[:200]}
        _save()

    _save()
    print(f"[driver] DONE -> {args.out}", flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards-dir", required=True, help="residual_shard train corpus")
    ap.add_argument("--eval-shards-dir", default=None, help="cross-corpus transfer set")
    ap.add_argument("--out", required=True, help="results JSON path")
    ap.add_argument("--n-blocks", type=int, default=16384)
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--block-topk", type=int, default=4, help="active blocks/token (L0=topk*b)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=65536, help="rows per streamed shard batch")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--block-tile", type=int, default=2048)
    ap.add_argument("--aux-k", type=int, default=256, help="AuxK dead-block revival budget")
    ap.add_argument("--held-out", type=int, default=50000, help="held-out test rows")
    ap.add_argument("--sample-rows", type=int, default=32768, help="frame-seed + to_fit sample")
    ap.add_argument("--transfer-cap", type=int, default=50000)
    ap.add_argument("--n-basis-chart", type=int, default=4)
    ap.add_argument("--chart-n-iter", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> dict:
    return drive(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
