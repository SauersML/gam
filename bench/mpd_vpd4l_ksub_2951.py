"""#2951: manifold (atlas) parameter decomposition of VPD's own 4-layer Pile target, against VPD's logged evals.

Analysis under SPEC 8's exception (torch execution; the fits are closed-form).

Every weight matrix ``W`` of the target reads one input stream ``z`` (``q/k/v`` share the post-``rms_1`` stream,
``o_proj`` reads the concatenated head outputs, ``c_fc`` the post-``rms_2`` stream, ``down_proj`` the GELU hidden).
The decomposition of ``W`` is an atlas of that stream's data manifold: ``C`` charts ``(mu_c, F_c)``, each an affine
``d``-dimensional patch (k-means cell, its own local principal frame). On a token in chart ``c`` the weights used are
the chart's slice ``z -> W mu_c + W F_c^T F_c (z - mu_c)``; ``W = sum_c 1[z in c] W (mu_c + P_c(. - mu_c))`` off the
normal residual. A token's active parameters at a site are one ``d``-dimensional patch (``d`` directions + one offset).

Evaluation is VPD's own: KL to the target (and CE difference) with EVERY decomposed site restricted at once, each
reading the already-modified stream (error propagation), on the pre-tokenized Pile-uncopyrighted TEST split the
target's run used (danbraunai/pile-uncopyrighted-tok). VPD's logged numbers for the same target (W&B summaries):
paper run s-55ea3f9b KL 0.313 (rounded masks), 0.214 (stochastic), total L0 201; newest run p-8383f5e5 KL 0.291
(rounded), 0.207 (stochastic), total L0 180.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from mpd_llm_chart_restriction_2951 import chart_project, kmeans, local_frames, nearest

# VPD p-8383f5e5 eval L0 per site (CI > 0), for the budget-matched arm
VPD_L0 = {
    0: {"q": 1.05, "k": 1.50, "v": 2.24, "o": 2.64, "fc": 12.81, "down": 12.14},
    1: {"q": 1.09, "k": 1.48, "v": 3.35, "o": 5.73, "fc": 6.99, "down": 5.96},
    2: {"q": 4.20, "k": 4.13, "v": 7.27, "o": 13.09, "fc": 10.34, "down": 9.12},
    3: {"q": 2.28, "k": 2.18, "v": 6.58, "o": 10.44, "fc": 23.23, "down": 30.57},
}
STREAMS = ("attn_in", "o_in", "mlp_in", "down_in")


class Target:
    def __init__(self, path, dev):
        from safetensors.torch import load_file
        self.w = {k: v.to(dev) for k, v in load_file(path).items()}
        self.n_layer, self.n_head, self.d, self.eps, self.base = 4, 6, 768, 1e-6, 10000.0
        self.hd = self.d // self.n_head
        self.edit = None  # (layer, stream, z) -> z_hat
        self.grab = None

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x):
        T = x.shape[-2]
        inv = 1.0 / (self.base ** (torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd))
        ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def site(self, l, stream, z):
        shape = z.shape
        flat = z.reshape(-1, shape[-1])
        if self.grab is not None:
            self.grab[l][stream].append(flat)
        if self.edit is not None:
            flat = self.edit(l, stream, flat)
        return flat.reshape(shape)

    def __call__(self, ids):
        w = self.w
        h = w["wte.weight"][ids]
        B, T, _ = h.shape
        for l in range(self.n_layer):
            p = f"h.{l}."
            x = self.site(l, "attn_in", self.rms(h, w[p + "rms_1.weight"]))
            q, k, v = ((x @ w[p + f"attn.{n}_proj.weight"].T).view(B, T, self.n_head, self.hd).transpose(1, 2)
                       for n in ("q", "k", "v"))
            o = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
            o = self.site(l, "o_in", o.transpose(1, 2).reshape(B, T, self.d))
            h = h + o @ w[p + "attn.o_proj.weight"].T
            x = self.site(l, "mlp_in", self.rms(h, w[p + "rms_2.weight"]))
            a = x @ w[p + "mlp.c_fc.weight"].T
            hid = self.site(l, "down_in", F.gelu(a, approximate="tanh"))
            h = h + hid @ w[p + "mlp.down_proj.weight"].T
        return self.rms(h, w["ln_f.weight"]) @ w["wte.weight"].T


def weighted_frames(Z, assign, C, dmax, W):
    """Per chart, the rank-d oblique slice minimizing E||W(z - z_hat)||^2 (reduced-rank regression):
    U = top-d left singular vectors of W (Z_c - mu_c)^T; read R = U^T W (d x in), write A = W^+ U (in x d)."""
    p = Z.shape[1]
    mus = torch.zeros(C, p, device=Z.device)
    reads = torch.zeros(C, dmax, p, device=Z.device)
    writes = torch.zeros(C, dmax, p, device=Z.device)
    Wpinv = torch.linalg.pinv(W.double()).float()  # in x out
    for c in range(C):
        pts = Z[assign == c]
        if pts.shape[0] == 0:
            continue
        mus[c] = pts.mean(0)
        Y = (pts - mus[c]) @ W.T  # n x out
        _, _, Vt = torch.linalg.svd(Y, full_matrices=False)
        k = min(dmax, Vt.shape[0])
        U = Vt[:k]  # k x out, orthonormal rows in output space
        reads[c, :k] = U @ W
        writes[c, :k] = (Wpinv @ U.T).T
    return mus, reads, writes


def oblique_project(z, c, mus, reads, writes, chunk=1024):
    out = torch.empty_like(z)
    for s in range(0, z.shape[0], chunk):
        cc = c[s:s + chunk]
        u = z[s:s + chunk] - mus[cc]
        coef = torch.einsum("ndp,np->nd", reads[cc], u)
        out[s:s + chunk] = mus[cc] + torch.einsum("nd,ndp->np", coef, writes[cc])
    return out


def pgd_adversary(model, held, clean, atlas, dsel, dmax, steps, step_size, dev, seed):
    """VPD's PGDReconLoss, transcribed to charts: per site, the directions a token's chart DROPS (frames d..dmax-1 of
    its W-weighted oblique slice, plus the remainder beyond them) are partially restored by masks t in [0,1], one
    mask per (site, dropped direction) shared by every token and position (VPD's ``source_shape: c``); random init,
    ``steps`` sign-gradient ascent steps of ``step_size`` on the mean KL, projected to [0,1]."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    masks = {(l, s): torch.rand(dmax - dsel(l, s) + 1, generator=gen, device=dev).requires_grad_(True)
             for l in range(model.n_layer) for s in STREAMS}

    def edit(l, s, z):
        centres, m, fr, wr, ww = atlas[l][s]
        d = dsel(l, s)
        c = nearest(z.detach(), centres)
        u = z - m[c]
        coef = torch.einsum("ndp,np->nd", wr[c], u)  # all dmax coordinates
        kept = m[c] + torch.einsum("nd,ndp->np", coef[:, :d], ww[c, :d])
        dropped_dirs = torch.einsum("nd,ndp->ndp", coef[:, d:], ww[c, d:])  # n x (dmax-d) x p
        full = m[c] + torch.einsum("nd,ndp->np", coef, ww[c])
        remainder = z - full
        t = masks[(l, s)]
        return kept + torch.einsum("k,nkp->np", t[:-1], dropped_dirs) + t[-1] * remainder

    def mean_kl():
        model.edit = edit
        tot, n = 0.0, 0
        for ids, ref in zip(held, clean):
            ids = ids.to(dev)
            lp = torch.log_softmax(model(ids[:, :-1]).float(), -1)
            r = ref.to(dev)
            tot = tot + (r.exp() * (r - lp)).sum()
            n += r.shape[0] * r.shape[1]
        model.edit = None
        return tot / n

    with torch.enable_grad():
        for _ in range(steps):
            loss = mean_kl()
            grads = torch.autograd.grad(loss, list(masks.values()))
            with torch.no_grad():
                for t, g in zip(masks.values(), grads):
                    t.add_(step_size * g.sign()).clamp_(0.0, 1.0)
        with torch.no_grad():
            final = mean_kl().item()
    return final


def residual_assign(z, m, reads, W, chunk=2048):
    """Each row's chart = argmin_c ||W(z - mu_c)||^2 - ||O_c^T W (z - mu_c)||^2, the output error of chart c's
    slice on that row (the read rows are O_c^T W with orthonormal O_c, so the kept energy is ||R_c (z - mu_c)||^2)."""
    C, d, p = reads.shape
    Wm = m @ W.T  # C x q
    Rm = torch.einsum("cdp,cp->cd", reads, m)  # C x d
    Rflat = reads.reshape(C * d, p)
    out = torch.empty(z.shape[0], dtype=torch.long, device=z.device)
    for s in range(0, z.shape[0], chunk):
        zz = z[s:s + chunk]
        Wz = zz @ W.T  # n x q
        total = (Wz.pow(2).sum(1, keepdim=True) - 2 * Wz @ Wm.T + Wm.pow(2).sum(1)[None])  # n x C
        kept = ((zz @ Rflat.T).view(-1, C, d) - Rm[None]).pow(2).sum(-1)  # n x C
        out[s:s + chunk] = (total - kept).argmin(1)
    return out


def k_subspaces(Z, W, assign, C, d, max_sweeps):
    """Alternate: fit each cell's rank-d W-weighted slice; reassign every row to the chart whose slice carries its
    output best. The summed output error never increases; stop at a fixed point (or max_sweeps, reported)."""
    history = []
    for sweep in range(max_sweeps):
        m, reads, writes = weighted_frames(Z, assign, C, d, W)
        new = residual_assign(Z, m, reads, W)
        changed = int((new != assign).sum())
        history.append(changed)
        assign = new
        if changed == 0:
            break
    m, reads, writes = weighted_frames(Z, assign, C, d, W)
    return m, reads, writes, history


def rows(path, n, offset=0):
    t = pq.read_table(path).slice(offset, n)
    return torch.tensor(t.column("input_ids").to_pylist(), dtype=torch.long)


def site_weight(model, l, s):
    p = f"h.{l}."
    w = model.w
    if s == "attn_in":
        return torch.cat([w[p + f"attn.{n}_proj.weight"] for n in ("q", "k", "v")])
    return {"o_in": w[p + "attn.o_proj.weight"], "mlp_in": w[p + "mlp.c_fc.weight"],
            "down_in": w[p + "mlp.down_proj.weight"]}[s]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--bank-rows", type=int, required=True)
    parser.add_argument("--eval-rows", type=int, required=True)
    parser.add_argument("--charts", required=True)
    parser.add_argument("--dims", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--max-sweeps", type=int, default=30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Target(args.weights, dev)
    test = rows(args.test, args.eval_rows)
    held = [test[i:i + args.batch] for i in range(0, len(test), args.batch)]
    report = {"args": vars(args), "vpd": {"s-55ea3f9b": {"kl_rounded": 0.31295, "kl_stoch": 0.21356, "l0": 201.45},
                                          "p-8383f5e5": {"kl_rounded": 0.29135, "kl_stoch": 0.20661, "l0": 180.41}}}
    clean = []
    ce = 0.0
    with torch.inference_mode():
        for ids in held:
            ids = ids.to(dev)
            logits = model(ids[:, :-1]).float()
            clean.append(torch.log_softmax(logits, -1).cpu())
            ce += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()
    report["clean_ce"] = ce / len(held)
    print(f"[atlas4l] clean CE {report['clean_ce']:.4f} (run's logged val loss 2.7075)", flush=True)

    def evaluate(edit):
        model.edit = edit
        kl = ce = 0.0
        n = 0
        with torch.inference_mode():
            for ids, ref in zip(held, clean):
                ids = ids.to(dev)
                logits = model(ids[:, :-1]).float()
                lp = torch.log_softmax(logits, -1)
                r = ref.to(dev)
                kl += (r.exp() * (r - lp)).sum().item()
                n += r.shape[0] * r.shape[1]
                ce += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()
        model.edit = None
        return {"kl": kl / n, "ce_difference": ce / len(held) - report["clean_ce"]}

    bank_ids = rows(args.train, args.bank_rows)
    model.grab = [{s: [] for s in STREAMS} for _ in range(model.n_layer)]
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(bank_ids), args.batch):
            model(bank_ids[i:i + args.batch, :-1].to(dev))
    Z = [{s: torch.cat(model.grab[l][s]) for s in STREAMS} for l in range(model.n_layer)]
    model.grab = None
    print(f"[atlas4l] banks {[(s, tuple(Z[0][s].shape)) for s in STREAMS]} in {time.time() - t0:.0f}s", flush=True)
    mus = [{s: Z[l][s].mean(0) for s in STREAMS} for l in range(model.n_layer)]
    report["mean_everywhere"] = evaluate(lambda l, s, z: mus[l][s].expand_as(z))
    print(f"[atlas4l] every site at its mean input: {report['mean_everywhere']}", flush=True)
    dims = [int(d) for d in args.dims.split(",")]
    dmax = max(max(dims), 32)
    budget = {l: {"attn_in": math.ceil(max(VPD_L0[l]["q"], VPD_L0[l]["k"], VPD_L0[l]["v"])),
                  "o_in": math.ceil(VPD_L0[l]["o"]), "mlp_in": math.ceil(VPD_L0[l]["fc"]),
                  "down_in": math.ceil(VPD_L0[l]["down"])} for l in range(4)}
    report["budget_per_site"] = budget
    report["arms"] = {}
    gen = torch.Generator().manual_seed(args.seed)
    for C in [int(c) for c in args.charts.split(",")]:
        t0 = time.time()
        atlas = [{} for _ in range(model.n_layer)]
        for l in range(model.n_layer):
            for s in STREAMS:
                if C == 1:
                    centres = mus[l][s][None]
                    assign = torch.zeros(Z[l][s].shape[0], dtype=torch.long, device=dev)
                else:
                    centres, assign = kmeans(Z[l][s], C, 20, gen)
                m, fr = local_frames(Z[l][s], assign, C, dmax)
                W = site_weight(model, l, s)
                wm, wr, ww = weighted_frames(Z[l][s], assign, C, dmax, W)
                atlas[l][s] = (centres, m, fr, wr, ww)
        fit_s = time.time() - t0
        print(f"[atlas4l] C={C}: fitted {4 * len(STREAMS)} atlases in {fit_s:.1f}s", flush=True)

        def make(dsel, weighted):
            def edit(l, s, z):
                centres, m, fr, wr, ww = atlas[l][s]
                d = dsel(l, s)
                if weighted:
                    return oblique_project(z, nearest(z, centres), m, wr[:, :d], ww[:, :d])
                return chart_project(z, nearest(z, centres), m, fr[:, :d])
            return edit
        arms = [(f"uniform_d{d}", (lambda d: (lambda l, s: d))(d),
                 sum(d * (3 if s == "attn_in" else 1) for s in STREAMS) * 4) for d in dims]
        arms.append(("vpd_budget", lambda l, s: budget[l][s],
                     sum(budget[l][s] * (3 if s == "attn_in" else 1) for l in range(4) for s in STREAMS)))
        ksub = {}
        for d in dims:
            ksub[d] = [{} for _ in range(model.n_layer)]
            t1 = time.time()
            for l in range(model.n_layer):
                for st in STREAMS:
                    W = site_weight(model, l, st)
                    _, assign0 = (None, nearest(Z[l][st], atlas[l][st][0]))
                    m, rd, wr, hist = k_subspaces(Z[l][st], W, assign0, C, d, args.max_sweeps)
                    ksub[d][l][st] = (m, rd, wr, W)
                    if l == 3 and st == "down_in":
                        print(f"[atlas4l] C={C} d={d} k-subspaces sweeps changed {hist}", flush=True)
            def kedit(l, st, z, d=d):
                m, rd, wr, W = ksub[d][l][st]
                return oblique_project(z, residual_assign(z, m, rd, W), m, rd, wr)
            res = evaluate(kedit)
            res.update({"charts": C, "fit_seconds": time.time() - t1 + fit_s,
                        "active_directions_per_token": d * 6 * 4})
            report["arms"][f"C{C}:ksub_d{d}"] = res
            print(f"[atlas4l] C={C} k-subspace uniform_d{d}: KL {res['kl']:.4f}  dCE {res['ce_difference']:.4f}  "
                  f"(active directions/token {d * 24})", flush=True)
        del ksub
        for name, dsel, pieces in arms:
            if C > 1 and args.pgd_steps > 0:
                adv = pgd_adversary(model, held[:1], clean[:1], atlas, dsel, dmax, args.pgd_steps, 0.1, dev, args.seed)
                report["arms"][f"C{C}:W-weighted {name}:pgd{args.pgd_steps}"] = adv
                print(f"[atlas4l] C={C} W-weighted {name}: PGD-{args.pgd_steps} shared-mask KL {adv:.4f} "
                      f"(VPD p-8383f5e5 PGDReconLoss_20step 0.6045)", flush=True)
            for weighted in (False, True):
                tag = ("W-weighted " if weighted else "input-PCA ") + name
                res = evaluate(make(dsel, weighted))
                res.update({"charts": C, "fit_seconds": fit_s, "active_directions_per_token": pieces})
                report["arms"][f"C{C}:{tag}"] = res
                print(f"[atlas4l] C={C} {tag}: KL {res['kl']:.4f}  dCE {res['ce_difference']:.4f}  "
                      f"(active directions/token across all 24 matrices {pieces})", flush=True)
        del atlas
        torch.cuda.empty_cache()
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)
    print(f"[atlas4l] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
