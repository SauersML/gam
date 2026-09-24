"""#2951: Parseval-frame parameter decomposition on the Stiefel manifold, on VPD's own 4-layer Pile target.

Analysis under SPEC 8's exception (torch execution of the target and of the Riemannian optimization).

For every one of the 24 weight matrices ``W`` (``q x p``) of the target, the decomposition is a Parseval frame of its
output space: ``K`` atoms ``O_k`` (``q x m``) whose stack ``X = [O_1; ...; O_K]^T``-transposed has orthonormal
columns, ``X^T X = I_q``. Then ``sum_k O_k O_k^T = I`` EXACTLY, so the pieces ``B_k = O_k O_k^T W`` (rank ``m``) sum to
``W`` with zero faithfulness error for every frame: faithfulness is the geometry of the Stiefel manifold ``St(Km, q)``
the frame lives on, not a penalty. ``Km > q`` makes it overcomplete (room for superposition).

A token uses the ``L`` pieces carrying the most of its own output energy ``||O_k^T W z||^2`` (a parameter-free
selector: no causal-importance network, no penalty; ``L`` is a hard constraint). The frame is fitted by Riemannian
Adam on the product of Stiefel manifolds against the model's own end-to-end KL with all 24 matrices restricted at
once (errors propagating): tangent projection ``G - X sym(X^T G)``, QR retraction.

Evaluation is VPD's: KL to the target (rounded masks = selected pieces on, the rest off), and VPD's PGDReconLoss
adversary (inactive pieces re-added with masks in [0,1] shared by all tokens, 20 sign-gradient steps of 0.1), on the
target's own Pile-uncopyrighted test split.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from mpd_vpd4l_atlas_2951 import rows

MATS = ("q", "k", "v", "o", "fc", "down")
# VPD p-8383f5e5 final eval L0 per matrix (CI > 0)
VPD_L0 = {0: {"q": 1.05, "k": 1.50, "v": 2.24, "o": 2.64, "fc": 12.81, "down": 12.14},
          1: {"q": 1.09, "k": 1.48, "v": 3.35, "o": 5.73, "fc": 6.99, "down": 5.96},
          2: {"q": 4.20, "k": 4.13, "v": 7.27, "o": 13.09, "fc": 10.34, "down": 9.12},
          3: {"q": 2.28, "k": 2.18, "v": 6.58, "o": 10.44, "fc": 23.23, "down": 30.57}}
VPD = {"p-8383f5e5": {"kl_rounded": 0.29135, "kl_stoch": 0.20661, "kl_ci": 0.31372, "l0": 180.41,
                      "pgd20": 0.6045}}


class Target:
    def __init__(self, path, dev):
        from safetensors.torch import load_file
        self.w = {k: v.to(dev) for k, v in load_file(path).items()}
        self.n_layer, self.n_head, self.d, self.eps, self.base = 4, 6, 768, 1e-6, 10000.0
        self.hd = self.d // self.n_head
        self.edit = None  # (layer, mat, y) -> y_hat on the matrix output

    def W(self, l, m):
        p = f"h.{l}."
        return {"q": p + "attn.q_proj.weight", "k": p + "attn.k_proj.weight", "v": p + "attn.v_proj.weight",
                "o": p + "attn.o_proj.weight", "fc": p + "mlp.c_fc.weight", "down": p + "mlp.down_proj.weight"}[m]

    def lin(self, l, m, z):
        y = z @ self.w[self.W(l, m)].T
        if self.edit is None:
            return y
        shape = y.shape
        return self.edit(l, m, y.reshape(-1, shape[-1]), z.reshape(-1, z.shape[-1])).reshape(shape)

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x):
        T = x.shape[-2]
        inv = 1.0 / (self.base ** (torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd))
        ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def __call__(self, ids):
        w = self.w
        h = w["wte.weight"][ids]
        B, T, _ = h.shape
        for l in range(self.n_layer):
            p = f"h.{l}."
            x = self.rms(h, w[p + "rms_1.weight"])
            q, k, v = (self.lin(l, n, x).view(B, T, self.n_head, self.hd).transpose(1, 2) for n in ("q", "k", "v"))
            o = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
            h = h + self.lin(l, "o", o.transpose(1, 2).reshape(B, T, self.d))
            x = self.rms(h, w[p + "rms_2.weight"])
            h = h + self.lin(l, "down", F.gelu(self.lin(l, "fc", x), approximate="tanh"))
        return self.rms(h, w["ln_f.weight"]) @ w["wte.weight"].T


ADAPTIVE = {"on": False}


def select(energy, L):
    """Per-row top-L, or (adaptive) the top n*L energies of the whole batch at this matrix: the same mean L0 per
    token, spent where the tokens need it (VPD's L0 is likewise a mean over tokens)."""
    e = energy.detach()
    if not ADAPTIVE["on"]:
        return torch.zeros_like(e).scatter_(1, e.topk(L, dim=1).indices, 1.0)
    flat = e.reshape(-1)
    keep = torch.zeros_like(flat)
    keep[flat.topk(min(flat.numel(), e.shape[0] * L)).indices] = 1.0
    return keep.view_as(e)


def frame_restrict(y, X, K, m, L, masks=None):
    """y: n x q; X: (K m) x q with orthonormal columns. Keep the L atoms with the most energy; with ``masks``
    (K,), the inactive atoms are re-added with those weights (VPD's adversarial source on inactive components)."""
    c = (y @ X.T).view(-1, K, m)  # atom coordinates
    energy = c.pow(2).sum(-1)
    keep = select(energy, L)
    weight = keep if masks is None else keep + (1 - keep) * masks[None]
    return (c * weight[..., None]).reshape(-1, K * m) @ X


def frame_restrict_input(z, X, W, K, m, L, masks=None):
    """Input-side frame: pieces W O_k O_k^T with sum_k O_k O_k^T = I on the input space. The token keeps the L
    atoms whose pieces write the most output energy ||W O_k c_k||^2 = c_k^T (O_k^T W^T W O_k) c_k."""
    c = (z @ X.T).view(-1, K, m)
    WO = (W @ X.T).T.reshape(K, m, -1)  # K x m x q
    gram = torch.einsum("kaq,kbq->kab", WO, WO)
    energy = torch.einsum("nka,kab,nkb->nk", c, gram, c)
    keep = select(energy, L)
    weight = keep if masks is None else keep + (1 - keep) * masks[None]
    zhat = (c * weight[..., None]).reshape(-1, K * m) @ X
    return zhat @ W.T


def neuron_frame(K, m, q, overcomplete, dev, gen):
    """Parseval frame whose first block is the standard basis grouped m at a time; with overcompleteness 2 a
    random rotation of it is stacked and the whole scaled by 1/sqrt(2), which keeps X^T X = I exactly."""
    eye = torch.eye(q)
    if overcomplete <= 1:
        X = eye
    else:
        R = torch.linalg.qr(torch.randn(q, q, generator=gen))[0]
        X = torch.cat([eye, R.T]) / math.sqrt(2.0)
    return X.to(dev)


def biorth_restrict(v, V, K, m, L, W=None, masks=None):
    """Biorthogonal (oblique) frame: V (d x d) invertible, a_k = columns of V grouped m at a time, b_k = columns of
    V^{-T}; sum_k b_k a_k^T = I EXACTLY for every invertible V. Coordinates c = v V, reconstruction from the kept
    groups c_A (V^{-1})_A. With W (input side) the energy of piece k is ||W B_k c_k||^2, else ||B_k c_k||^2."""
    Vinv = torch.linalg.inv(V)  # rows of V^{-1} = columns of V^{-T} = b_k
    c = (v @ V).view(-1, K, m)
    B = Vinv.view(K, m, -1)  # K x m x d
    WB = B if W is None else torch.einsum("kmd,qd->kmq", B, W)
    gram = torch.einsum("kaq,kbq->kab", WB, WB)
    energy = torch.einsum("nka,kab,nkb->nk", c, gram, c)
    keep = select(energy, L)
    weight = keep if masks is None else keep + (1 - keep) * masks[None]
    vhat = (c * weight[..., None]).reshape(-1, K * m) @ Vinv
    return vhat if W is None else vhat @ W.T


def stiefel(K, m, q, dev, gen):
    G = torch.randn(K * m, q, generator=gen).to(dev)
    return torch.linalg.qr(G)[0]  # (K m) x q, orthonormal columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--eval-rows", type=int, required=True)
    parser.add_argument("--overcomplete", type=float, required=True, help="K m / q")
    parser.add_argument("--atom-dim", type=int, required=True)
    parser.add_argument("--active", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--init", choices=("neuron", "random", "svd"), default="neuron")
    parser.add_argument("--attn-side", choices=("input", "output"), default="output")
    parser.add_argument("--opt", choices=("adam", "nsgd"), default="adam")
    parser.add_argument("--frame-kind", choices=("parseval", "biorth"), default="parseval",
                        help="biorth: an invertible V on GL(d), pieces b_k a_k^T with B = V^{-T}; exact for every V")
    parser.add_argument("--adaptive", action="store_true",
                        help="batch-level top-(n L) selection per matrix: same mean pieces per token, adaptive per token")
    parser.add_argument("--resume-frames", default="", help="load frames saved by an earlier run (skips ODL)")
    parser.add_argument("--tf32-train", action="store_true",
                        help="TF32 matmuls in training steps only; every evaluation stays full fp32")
    parser.add_argument("--odl-iters", type=int, default=0)
    parser.add_argument("--odl-rows", type=int, default=64)
    parser.add_argument("--only-site", default="", help="restrict only this matrix, as layer:mat (diagnosis)")
    parser.add_argument("--only", choices=("all", "mlp", "attn"), default="all",
                        help="restrict only these matrices (diagnosis); the rest run exactly")
    parser.add_argument("--budget-scale", type=float, default=1.0,
                        help="with --vpd-budget: pieces per matrix scaled from VPD's final per-matrix L0")
    parser.add_argument("--retract-every", type=int, default=1,
                        help="QR retraction every this many steps; between, tangent steps leave X^T X = I + O(lr^2)")
    parser.add_argument("--test-every", type=int, default=0, help="test evaluations every this many steps (0: 4 per run)")
    parser.add_argument("--vpd-budget", action="store_true",
                        help="per-matrix pieces L = ceil(VPD's own per-matrix eval L0 / m) instead of a uniform L")
    parser.add_argument("--stochastic", action="store_true",
                        help="also train on inactive pieces re-added with uniform [0,1] masks (VPD's stochastic claim)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    ADAPTIVE["on"] = args.adaptive
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Target(args.weights, dev)
    test = rows(args.test, args.eval_rows)
    held = [test[i:i + args.batch] for i in range(0, len(test), args.batch)]
    train = rows(args.train, args.train_rows)
    m, L = args.atom_dim, args.active
    actives = {(l, mat): (max(1, math.ceil(args.budget_scale * VPD_L0[l][mat] / m)) if args.vpd_budget else L)
               for l in range(4) for mat in MATS}
    gen = torch.Generator().manual_seed(args.seed)
    frames, shapes, sides = {}, {}, {}
    for l in range(model.n_layer):
        for mat in MATS:
            Wm = model.w[model.W(l, mat)]
            if mat in ("fc", "down"):
                side = "input" if mat == "down" else "output"  # the neuron side of each MLP matrix
            else:
                side = args.attn_side  # the residual-stream (feature) side for q/k/v is their input
            dim = Wm.shape[1] if side == "input" else Wm.shape[0]
            K = math.ceil(args.overcomplete * dim / m)
            shapes[(l, mat)] = K
            sides[(l, mat)] = side
            if args.init in ("neuron", "svd") and mat in ("fc", "down"):
                frames[(l, mat)] = neuron_frame(K, m, dim, args.overcomplete, dev, gen)
            elif args.init == "svd" and args.overcomplete <= 1:
                # the weight's own rank-revealing read (seed.rs): right singular vectors on the input side, left on
                # the output side, as an orthogonal complete frame; atoms pair consecutive singular directions
                U, _, Vh = torch.linalg.svd(Wm.double(), full_matrices=True)
                frames[(l, mat)] = (Vh if side == "input" else U.T).float().contiguous()
            else:
                frames[(l, mat)] = stiefel(K, m, dim, dev, gen)
    if args.frame_kind == "biorth":
        # atoms are the COLUMNS of V; every initial frame is orthogonal, so the transpose is exact and B = V^{-T} = V
        frames = {k: v.T.contiguous() for k, v in frames.items()}
    report = {"args": vars(args), "vpd": VPD, "pieces_per_token": sum(actives.values()),
              "matrix_slices_per_token": m * sum(actives.values()), "atoms_per_matrix": {f"{k[0]}:{k[1]}": v for k, v in shapes.items()}}
    print(f"[frame4l] {sum(actives.values())} pieces of rank {m} per token ({m * sum(actives.values())} matrix slices; "
          f"VPD L0 180 rank-1)", flush=True)

    def edit_with(masks=None):
        def edit(l, mat, y, z=None):
            if args.only == "mlp" and mat not in ("fc", "down") or args.only == "attn" and mat in ("fc", "down"):
                return y
            if args.only_site and args.only_site != f"{l}:{mat}":
                return y
            mk = None if masks is None else masks[(l, mat)]
            if args.frame_kind == "biorth":
                if sides[(l, mat)] == "input":
                    return biorth_restrict(z, frames[(l, mat)], shapes[(l, mat)], m, actives[(l, mat)],
                                           model.w[model.W(l, mat)], mk)
                return biorth_restrict(y, frames[(l, mat)], shapes[(l, mat)], m, actives[(l, mat)], None, mk)
            if sides[(l, mat)] == "input":
                return frame_restrict_input(z, frames[(l, mat)], model.w[model.W(l, mat)], shapes[(l, mat)], m,
                                            actives[(l, mat)], mk)
            return frame_restrict(y, frames[(l, mat)], shapes[(l, mat)], m, actives[(l, mat)], mk)
        return edit

    def evaluate():
        kl = ce = 0.0
        n = 0
        cce = 0.0
        with torch.inference_mode():
            for ids in held:
                ids = ids.to(dev)
                model.edit = None
                ref_logits = model(ids[:, :-1]).float()
                ref = torch.log_softmax(ref_logits, -1)
                model.edit = edit_with()
                logits = model(ids[:, :-1]).float()
                model.edit = None
                lp = torch.log_softmax(logits, -1)
                kl += (ref.exp() * (ref - lp)).sum().item()
                n += ref.shape[0] * ref.shape[1]
                tgt = ids[:, 1:].reshape(-1)
                ce += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt).item()
                cce += F.cross_entropy(ref_logits.reshape(-1, ref_logits.shape[-1]), tgt).item()
        return {"kl": kl / n, "ce_difference": (ce - cce) / len(held)}

    def batch_kl(ids, masks=None):
        with torch.no_grad():
            model.edit = None
            ref = torch.log_softmax(model(ids[:, :-1]).float(), -1)
        model.edit = edit_with(masks)
        lp = torch.log_softmax(model(ids[:, :-1]).float(), -1)
        model.edit = None
        return (ref.exp() * (ref - lp)).sum(-1).mean()

    if args.resume_frames:
        saved = torch.load(args.resume_frames, map_location=dev)
        for key in frames:
            frames[key] = saved[f"{key[0]}:{key[1]}"].to(dev)
        print(f"[frame4l] resumed frames from {args.resume_frames}", flush=True)
    if args.odl_iters > 0 and not args.resume_frames:
        # Sequential orthogonal dictionary learning: in network order, on the stream as it arrives with every
        # upstream matrix already restricted, alternate (a) each row coded by its top-L atom groups of the current
        # orthogonal frame and (b) the frame updated by orthogonal Procrustes, X = polar(S^T D), which minimizes
        # ||D - S X|| over orthogonal X for the fixed codes (D the rows of the frame's side: the input z on the input
        # side, the output y on the output side). Monotone in the reconstruction error; closed form per step.
        odl_ids = train[: args.odl_rows].to(dev)
        fitted = set()
        order = [(l, mat) for l in range(model.n_layer) for mat in MATS]
        for key in order:
            bank = {}

            def grab_edit(l, mat, y, z=None):
                if (l, mat) == key:
                    bank["d"] = (z if sides[key] == "input" else y).detach()
                if (l, mat) in fitted:
                    return edit_with()(l, mat, y, z)
                return y
            model.edit = grab_edit
            with torch.no_grad():
                model(odl_ids[:, :-1])
            model.edit = None
            D = bank["d"]
            X = frames[key]
            K = shapes[key]
            Lk = actives[key]
            for _ in range(args.odl_iters):
                c = (D @ X.T).view(-1, K, m)
                energy = c.pow(2).sum(-1)
                keep = torch.zeros_like(energy).scatter_(1, energy.topk(Lk, dim=1).indices, 1.0)
                S = (c * keep[..., None]).reshape(-1, K * m)
                U, _, Vh = torch.linalg.svd(S.T @ D, full_matrices=False)
                X = U @ Vh
            frames[key] = X
            fitted.add(key)
        print(f"[frame4l] sequential orthogonal dictionary learning done ({args.odl_iters} iterations per matrix)",
              flush=True)
    report["init"] = evaluate()
    print(f"[frame4l] random Stiefel init: {report['init']}", flush=True)
    keys = list(frames)
    mom = {k: torch.zeros_like(frames[k]) for k in keys}
    sq = {k: torch.zeros_like(frames[k]) for k in keys}
    trace = []
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, len(train), (args.batch,), generator=gen)
        ids = train[idx].to(dev)
        for k in keys:
            frames[k].requires_grad_(True)
        torch.backends.cuda.matmul.allow_tf32 = args.tf32_train
        with torch.enable_grad():
            loss = batch_kl(ids)
            if args.stochastic:
                smask = {k: torch.rand(shapes[k], device=dev) for k in keys}
                loss = 0.5 * loss + 0.5 * batch_kl(ids, smask)
            grads = torch.autograd.grad(loss, [frames[k] for k in keys])
        torch.backends.cuda.matmul.allow_tf32 = False  # every evaluation runs in full fp32
        trace.append(loss.item())
        with torch.no_grad():
            for k, g in zip(keys, grads):
                X = frames[k].detach()
                if args.frame_kind == "biorth":
                    mom[k].mul_(0.9).add_(0.1 * g)
                    sq[k].mul_(0.999).add_(0.001 * g.pow(2))
                    upd = (mom[k] / (1 - 0.9 ** (step + 1))) / ((sq[k] / (1 - 0.999 ** (step + 1))).sqrt() + 1e-12)
                    lr_t = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
                    frames[k] = (X - lr_t * upd).detach()
                    continue
                sym = 0.5 * (X.T @ g + g.T @ X)
                tang = g - X @ sym
                if args.opt == "nsgd":
                    # rotation-equivariant normalized Riemannian momentum: the step moves the frame by a fraction
                    # lr of its own Frobenius norm sqrt(d), in the direction of the transported momentum
                    mom[k].mul_(0.9).add_(tang)
                    mom[k] = mom[k] - X @ (0.5 * (X.T @ mom[k] + mom[k].T @ X))
                    upd = mom[k] * (math.sqrt(X.shape[1]) / (mom[k].norm() + 1e-30))
                else:
                    mom[k].mul_(0.9).add_(0.1 * tang)
                    sq[k].mul_(0.999).add_(0.001 * tang.pow(2))
                    upd = (mom[k] / (1 - 0.9 ** (step + 1))) / ((sq[k] / (1 - 0.999 ** (step + 1))).sqrt() + 1e-12)
                    upd = upd - X @ (0.5 * (X.T @ upd + upd.T @ X))
                lr_t = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
                moved = X - lr_t * upd
                if (step + 1) % args.retract_every == 0 or step == args.steps - 1:
                    Q, Rr = torch.linalg.qr(moved)
                    moved = Q * torch.sign(torch.diagonal(Rr))[None]
                frames[k] = moved.detach()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"[frame4l] step {step}: train KL {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if (step + 1) % (args.test_every or max(1, args.steps // 4)) == 0:
            res = evaluate()
            print(f"[frame4l] step {step + 1}: TEST {res}", flush=True)
    report["trace"] = trace[::10]
    report["train_seconds"] = time.time() - t0
    report["final"] = evaluate()
    # exact faithfulness: sum of all pieces = W
    if args.frame_kind == "biorth":
        worst = max((frames[k] @ torch.linalg.inv(frames[k]) - torch.eye(frames[k].shape[0], device=dev)).abs().max().item()
                    for k in keys)
        report["frame_condition_max"] = max(torch.linalg.cond(frames[k]).item() for k in keys)
        print(f"[frame4l] max cond(V) {report['frame_condition_max']:.3e}", flush=True)
    else:
        worst = max((frames[k].T @ frames[k] - torch.eye(frames[k].shape[1], device=dev)).abs().max().item() for k in keys)
    report["frame_orthonormality_defect_max"] = worst
    print(f"[frame4l] FINAL {report['final']} (VPD rounded 0.291, stochastic 0.207); max |X^T X - I| {worst:.2e}",
          flush=True)
    if args.pgd_steps > 0:
        g2 = torch.Generator(device=dev).manual_seed(args.seed)
        masks = {k: torch.rand(shapes[k], generator=g2, device=dev) for k in keys}
        ids = held[0].to(dev)
        for _ in range(args.pgd_steps):
            for k in keys:
                masks[k].requires_grad_(True)
            with torch.enable_grad():
                loss = batch_kl(ids, masks)
                grads = torch.autograd.grad(loss, [masks[k] for k in keys])
            with torch.no_grad():
                for k, g in zip(keys, grads):
                    masks[k] = (masks[k] + 0.1 * g.sign()).clamp(0, 1).detach()
        with torch.no_grad():
            report["pgd"] = batch_kl(ids, masks).item()
        print(f"[frame4l] PGD-{args.pgd_steps} (inactive pieces re-added adversarially, shared masks): "
              f"KL {report['pgd']:.4f} (VPD 0.6045)", flush=True)
    torch.save({f"{k[0]}:{k[1]}": frames[k].cpu() for k in keys}, args.out.replace(".json", "_frames.pt"))
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[frame4l] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
