"""#2951 probe: is a pretrained LM's weight action determined by the LOCAL geometry of its input manifold?

Analysis under SPEC 8's exception (torch execution of a measurement; it decides whether the chart-restricted
decomposition belongs in the engine, it is not that decomposition).

At a site (the post-norm input of layer ``L``'s MLP, shared by gate and up) every held-out token's input ``x`` is
replaced by a projection and the next-token KL to the clean model is measured:

* ``global r``: ``mu + P_r (x - mu)``, the top-``r`` principal subspace of a training bank (one chart);
* ``atlas C x d``: ``x`` is assigned to its nearest of ``C`` k-means centres and replaced by
  ``mu_c + P_{c,d} (x - mu_c)``, the chart's own top-``d`` local principal subspace;
* ``random C x d``: the same with a RANDOM partition of the bank into ``C`` cells (equal parameter count, no
  geometry) — separates "more subspaces" from "the manifold's local tangent";
* ``VQ C``: ``d = 0``, the centre alone.

Every projection is fitted on the training bank and applied to disjoint held-out tokens. If the local dimension at
fine scale is small, ``atlas C x d`` reaches a KL that ``global r`` needs ``r >> d`` for, and ``random`` does not.
The per-token count of rank-1 weight pieces the restriction uses at this site is ``d`` per matrix reading ``x``.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F


def token_batches(tok, seq_len, batch, seed, skip):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    buf, made = [], 0
    for row in ds:
        buf.extend(tok(row["text"]).input_ids + [tok.eos_token_id])
        while len(buf) >= seq_len * batch:
            chunk = torch.tensor(buf[: seq_len * batch]).view(batch, seq_len)
            buf = buf[seq_len * batch:]
            made += 1
            if made > skip:
                yield chunk


def kmeans(X, C, iters, gen):
    centres = X[torch.randperm(X.shape[0], generator=gen, device="cpu")[:C].to(X.device)].clone()
    for _ in range(iters):
        assign = nearest(X, centres)
        sums = torch.zeros_like(centres).index_add_(0, assign, X)
        counts = torch.bincount(assign, minlength=C).clamp_min(1).to(X.dtype)[:, None]
        centres = sums / counts
    return centres, nearest(X, centres)


def nearest(X, centres, chunk=16384):
    out = []
    cn = centres.pow(2).sum(-1)
    for s in range(0, X.shape[0], chunk):
        out.append((cn[None] - 2 * X[s:s + chunk] @ centres.T).argmin(-1))
    return torch.cat(out)


def local_frames(X, assign, C, dmax):
    mus = torch.zeros(C, X.shape[1], device=X.device)
    frames = torch.zeros(C, dmax, X.shape[1], device=X.device)
    for c in range(C):
        pts = X[assign == c]
        if pts.shape[0] == 0:
            continue
        mus[c] = pts.mean(0)
        _, _, Vt = torch.linalg.svd(pts - mus[c], full_matrices=False)
        k = min(dmax, Vt.shape[0])
        frames[c, :k] = Vt[:k]
    return mus, frames


def chart_project(x, c, mus, frames, chunk=512):
    out = torch.empty_like(x)
    for s in range(0, x.shape[0], chunk):
        cc = c[s:s + chunk]
        Fd = frames[cc]  # n x d x p
        u = x[s:s + chunk] - mus[cc]
        out[s:s + chunk] = mus[cc] + torch.einsum("nd,ndp->np", torch.einsum("ndp,np->nd", Fd, u), Fd)
    return out


def fisher_metric(args, model, tok, mlp, dev):
    """Mean output Fisher pulled back to the site: E[g g^T], g = d/dx of the log-probability of a label SAMPLED from
    the model's own next-token distribution, summed over every position (a site perturbation reaches every later
    prediction through attention). E_y[g g^T] at fixed input is the Fisher pullback of the summed next-token KL."""
    p = model.config.hidden_size
    Fbar = torch.zeros(p, p, device=dev, dtype=torch.float64)
    held = {}

    def keep(_m, inputs):
        x = inputs[0]
        x.retain_grad()
        held["x"] = x

    stream = token_batches(tok, args.seq_len, args.eval_batch, args.seed + 2, 0)
    seen = 0
    for prm in model.parameters():
        prm.requires_grad_(False)
    h = mlp.register_forward_pre_hook(keep)
    emb = model.get_input_embeddings()
    try:
        while seen < args.fisher_tokens:
            ids = next(stream).to(dev)
            with torch.enable_grad():
                e = emb(ids).detach().requires_grad_(True)
                logits = model(inputs_embeds=e).logits.float()
                y = torch.distributions.Categorical(logits=logits.detach()).sample()
                lp = torch.log_softmax(logits, -1).gather(-1, y[..., None]).sum()
                lp.backward()
            g = held["x"].grad.double().reshape(-1, p)
            Fbar += g.T @ g
            seen += g.shape[0]
            held.clear()
    finally:
        h.remove()
    return (Fbar / seen).float(), seen


def fisher_arms(args, model, tok, mlp, X, mu, ranks, dims, state, kl_under, dev, layer):
    """Restriction minimizing the second-order KL E[(1/2) d^T Fbar d], d = (I - P)(x - mu), over rank-r oblique P:
    with Fbar = L L^T, P = L^{-T} Pi_r L^T and Pi_r the top-r left singular subspace of L^T (X - mu)^T
    (reduced-rank regression). The same per chart with the chart's own rows."""
    Fbar, seen = fisher_metric(args, model, tok, mlp, dev)
    evals, evecs = torch.linalg.eigh(Fbar.double())
    floor = evals.max() * 1e-8
    Lt = (evecs * evals.clamp_min(floor).sqrt()).T.float()  # L^T with L = V diag(sqrt(lambda))
    Lt_inv = (evecs * evals.clamp_min(floor).rsqrt()).float()  # L^{-T}
    out = {"fisher_tokens": seen, "fisher_eigen_top": evals.flip(0)[:64].tolist(), "fisher_global": {}, "fisher_atlas": {}}
    Y = (X - mu) @ Lt.T  # rows L^T (x - mu)
    _, _, Vt = torch.linalg.svd(Y, full_matrices=False)
    for r in ranks:
        Pi = Vt[:r]
        out["fisher_global"][r] = kl_under(lambda x, Pi=Pi: mu + (((x - mu) @ Lt.T) @ Pi.T @ Pi) @ Lt_inv.T)
        print(f"[chart] layer {layer} fisher-global r={r}: KL {out['fisher_global'][r]:.4f}", flush=True)
    C = args.fisher_charts
    gen = torch.Generator().manual_seed(args.seed)
    centres, assign = kmeans(X, C, 25, gen)
    dmax = max(dims)
    mus = torch.zeros(C, X.shape[1], device=dev)
    frames = torch.zeros(C, dmax, X.shape[1], device=dev)
    for c in range(C):
        pts = X[assign == c]
        if pts.shape[0] == 0:
            continue
        mus[c] = pts.mean(0)
        _, _, Vc = torch.linalg.svd((pts - mus[c]) @ Lt.T, full_matrices=False)
        k = min(dmax, Vc.shape[0])
        frames[c, :k] = Vc[:k]
    for d in dims:
        def fatlas(x, d=d):
            c = nearest(x, centres)
            y = (x - mus[c]) @ Lt.T
            return mus[c] + chart_project(y, c, torch.zeros_like(mus), frames[:, :d]) @ Lt_inv.T
        out["fisher_atlas"][f"{C}x{d}"] = kl_under(fatlas)
        print(f"[chart] layer {layer} fisher-atlas C={C} d={d}: KL {out['fisher_atlas'][f'{C}x{d}']:.4f}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--bank-tokens", type=int, required=True)
    parser.add_argument("--eval-batches", type=int, required=True)
    parser.add_argument("--charts", required=True)
    parser.add_argument("--dims", required=True)
    parser.add_argument("--global-ranks", required=True)
    parser.add_argument("--fisher-tokens", type=int, default=0,
                        help="tokens whose sampled-label gradients estimate the output Fisher metric at the site (0: skip)")
    parser.add_argument("--fisher-charts", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    stream = token_batches(tok, args.seq_len, args.eval_batch, args.seed, 0)
    held = [next(stream) for _ in range(args.eval_batches)]
    charts = [int(c) for c in args.charts.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    ranks = [int(r) for r in args.global_ranks.split(",")]
    dmax = max(dims)
    state = {"mode": None}
    report = {"args": vars(args), "layers": {}}

    with torch.inference_mode():
        clean = [torch.log_softmax(model(input_ids=h.to(dev)).logits.float(), -1).cpu() for h in held]

    def kl_under(project):
        state["mode"] = project
        tot, n = 0.0, 0
        with torch.inference_mode():
            for h, ref in zip(held, clean):
                lp = torch.log_softmax(model(input_ids=h.to(dev)).logits.float(), -1)
                r = ref.to(dev)
                tot += (r.exp() * (r - lp)).sum().item()
                n += r.shape[0] * r.shape[1]
        state["mode"] = None
        return tot / n

    for layer in [int(x) for x in args.layers.split(",")]:
        mlp = model.model.layers[layer].mlp
        cap = []

        def grab(_m, inputs):
            if state["mode"] == "grab":
                cap.append(inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]))

        def splice(_m, inputs):
            if callable(state["mode"]):
                x = inputs[0]
                y = state["mode"](x.float().reshape(-1, x.shape[-1])).reshape(x.shape).to(x.dtype)
                return (y,)
            return None

        h1 = mlp.register_forward_pre_hook(grab)
        h2 = mlp.register_forward_pre_hook(splice)
        state["mode"] = "grab"
        bank_stream = token_batches(tok, args.seq_len, args.batch, args.seed + 1, 0)
        t0 = time.time()
        with torch.inference_mode():
            while sum(c.shape[0] for c in cap) < args.bank_tokens:
                model(input_ids=next(bank_stream).to(dev))
        state["mode"] = None
        X = torch.cat(cap)[: args.bank_tokens]
        cap.clear()
        print(f"[chart] layer {layer} bank {tuple(X.shape)} in {time.time() - t0:.0f}s", flush=True)
        entry = {"global": {}, "atlas": {}, "random": {}, "vq": {}}
        mu = X.mean(0)
        _, _, Vt = torch.linalg.svd(X - mu, full_matrices=False)
        entry["zero_mean_input"] = kl_under(lambda x: mu.expand_as(x))
        print(f"[chart] layer {layer} mean input: KL {entry['zero_mean_input']:.4f}", flush=True)
        for r in ranks:
            P = Vt[:r]
            entry["global"][r] = kl_under(lambda x, P=P: mu + ((x - mu) @ P.T) @ P)
            print(f"[chart] layer {layer} global r={r}: KL {entry['global'][r]:.4f}", flush=True)
        del Vt
        if args.fisher_tokens:
            entry.update(fisher_arms(args, model, tok, mlp, X, mu, ranks, dims, state, kl_under, dev, layer))
        gen = torch.Generator().manual_seed(args.seed)
        for C in charts:
            centres, assign = kmeans(X, C, 25, gen)
            mus, frames = local_frames(X, assign, C, dmax)
            rand_assign = torch.randint(0, C, (X.shape[0],), generator=gen).to(dev)
            rmus, rframes = local_frames(X, rand_assign, C, dmax)
            entry["vq"][C] = kl_under(lambda x, ce=centres: ce[nearest(x, ce)])
            for d in dims:
                def atlas(x, d=d, ce=centres, m=mus, fr=frames):
                    return chart_project(x, nearest(x, ce), m, fr[:, :d])

                def random_cells(x, d=d, m=rmus, fr=rframes):
                    # a random cell has no locality: each token goes to a random cell, as the bank did
                    return chart_project(x, torch.randint(0, C, (x.shape[0],), device=x.device), m, fr[:, :d])

                entry["atlas"][f"{C}x{d}"] = kl_under(atlas)
                entry["random"][f"{C}x{d}"] = kl_under(random_cells)
                print(f"[chart] layer {layer} C={C} d={d}: atlas KL {entry['atlas'][f'{C}x{d}']:.4f}  "
                      f"random-cells KL {entry['random'][f'{C}x{d}']:.4f}  (VQ {entry['vq'][C]:.4f})", flush=True)
            del frames, rframes
        report["layers"][layer] = entry
        h1.remove()
        h2.remove()
        del X
        torch.cuda.empty_cache()
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)
    print(f"[chart] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
