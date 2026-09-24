"""#2951: a sparse union of m-dimensional subspace atoms per weight, on VPD's own 4-layer Pile target.

Analysis under SPEC 8's exception (torch execution; every fitting step is a closed-form selection or SVD).

For a weight ``W`` (``q x p``) reading stream ``z``, work in its output space: ``y = W (z - mu)``. A dictionary of ``K``
atoms, each an orthonormal ``m``-dimensional subspace ``O_k`` of ``range(W)``. A token uses ``L`` atoms, chosen by
greedy subspace pursuit on its own ``y`` (pick the atom carrying the most residual energy, remove its projection,
repeat), so its used weights are ``W`` restricted to the span of ``L`` planes: ``L m`` directions, chosen per token
from ``C(K, L)`` combinations. Execution replaces the input by ``mu + W^+ y_hat``, whose image under ``W`` is exactly
``W mu + y_hat``. VPD's rank-1 subcomponents are the ``m = 1`` case of this object.

Dictionary fitting alternates two closed-form steps to a fixed point of the summed output error: (1) per-token
selection; (2) each atom refitted as the top-``m`` left singular subspace of its tokens' residuals with its own
contribution added back (the K-SVD update lifted from lines to subspaces).

Evaluation is VPD's: KL to the target with every one of the 24 matrices restricted at once, each reading the
already-modified stream, on the target's own Pile-uncopyrighted test split.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from mpd_vpd4l_atlas_2951 import STREAMS, VPD_L0, Target, rows, site_weight


def range_basis(W):
    U, S, _ = torch.linalg.svd(W.double(), full_matrices=False)
    band = max(W.shape) * torch.finfo(torch.float64).eps * S[0]
    r = int((S > band).sum())
    return U[:, :r].float()


def pursue(Y, atoms, L):
    """Greedy subspace pursuit. Y: n x q residual targets; atoms: K x q x m orthonormal columns.
    Returns the selected atom indices (n x L) and the reconstruction."""
    K, q, m = atoms.shape
    flat = atoms.permute(0, 2, 1).reshape(K * m, q)  # rows = atom directions
    r = Y.clone()
    chosen = torch.empty(Y.shape[0], L, dtype=torch.long, device=Y.device)
    taken = torch.zeros(Y.shape[0], K, dtype=torch.bool, device=Y.device)
    for j in range(L):
        coef = (r @ flat.T).view(-1, K, m)  # n x K x m
        energy = coef.pow(2).sum(-1).masked_fill(taken, -1.0)
        k = energy.argmax(1)
        chosen[:, j] = k
        taken[torch.arange(Y.shape[0], device=Y.device), k] = True
        c = coef[torch.arange(Y.shape[0], device=Y.device), k]  # n x m
        r = r - torch.einsum("nm,nqm->nq", c, atoms[k])
    return chosen, Y - r


def fit_dictionary(Y, K, m, L, sweeps, gen):
    n, q = Y.shape
    idx = torch.randperm(n, generator=gen)[: K * m].to(Y.device)
    seed = Y[idx].view(K, m, q).permute(0, 2, 1)  # K x q x m
    atoms = torch.linalg.qr(seed)[0]
    history = []
    for _ in range(sweeps):
        chosen, recon = pursue(Y, atoms, L)
        err = (Y - recon).pow(2).sum().item()
        history.append(err)
        resid = Y - recon
        for k in range(K):
            rows_k, slot = (chosen == k).nonzero(as_tuple=True)
            if rows_k.numel() < m:
                # unused atom: re-seed on the worst-fitted rows (the residual's own top directions)
                worst = resid.pow(2).sum(1).topk(max(m, 64)).indices
                U, _, _ = torch.linalg.svd(resid[worst].T, full_matrices=False)
                atoms[k] = U[:, :m]
                continue
            back = resid[rows_k] + torch.einsum("nq,qm,pm->np", Y[rows_k], atoms[k], atoms[k])
            U, _, _ = torch.linalg.svd(back.T, full_matrices=False)
            new = U[:, :m]
            # keep the residual consistent for the next atom: swap this atom's projection
            resid[rows_k] = back - (back @ new) @ new.T
            atoms[k] = new
    chosen, recon = pursue(Y, atoms, L)
    history.append((Y - recon).pow(2).sum().item())
    return atoms, history


def pgd_masks(model, held, clean, dicts, mus, Ws, pinvs, L, steps, step_size, dev, seed):
    """VPD's PGDReconLoss on the atom decomposition: every atom of every site carries one mask t_k in [0,1], shared
    by all tokens and positions (VPD's ``source_shape: c``). A token's selected atoms contribute t_k times their
    pursuit coefficients (an atom's mask is t I on its subspace, the only gauge-invariant mask, P1); unselected atoms
    stay off. Random init; ``steps`` sign-gradient ascent steps on the mean KL, projected to [0,1]."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    masks = {key: torch.rand(atoms.shape[0], generator=gen, device=dev).requires_grad_(True)
             for key, atoms in ((k, v) for l in range(len(dicts)) for k, v in
                                (((l, s), dicts[l][s]) for s in dicts[l]))}

    def edit(l, s, z):
        atoms = dicts[l][s]
        y = (z - mus[l][s]) @ Ws[l][s].T
        with torch.no_grad():
            chosen, _ = pursue(y.detach(), atoms, L)
        r = y
        yhat = torch.zeros_like(y)
        t = masks[(l, s)]
        rows = torch.arange(y.shape[0], device=y.device)
        for j in range(L):
            k = chosen[:, j]
            c = torch.einsum("nq,nqm->nm", r, atoms[k])
            part = torch.einsum("nm,nqm->nq", c, atoms[k])
            yhat = yhat + t[k][:, None] * part
            r = r - part
        return mus[l][s] + yhat @ pinvs[l][s].T

    def mean_kl():
        model.edit = edit
        tot, n = 0.0, 0
        for ids, ref in zip(held, clean):
            ids = ids.to(dev)
            lp = torch.log_softmax(model(ids[:, :-1]).float(), -1)
            rr = ref.to(dev)
            tot = tot + (rr.exp() * (rr - lp)).sum()
            n += rr.shape[0] * rr.shape[1]
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
            return mean_kl().item()


def restricted(y, atoms, L):
    """Pursuit with the selection taken as fixed (no gradient through the argmax) and the reconstruction
    differentiable in the atoms: returns y_hat."""
    with torch.no_grad():
        chosen, _ = pursue(y.detach(), atoms.detach(), L)
    r = y
    yhat = torch.zeros_like(y)
    for j in range(L):
        a = atoms[chosen[:, j]]  # n x q x m
        part = torch.einsum("nm,nqm->nq", torch.einsum("nq,nqm->nm", r, a), a)
        yhat = yhat + part
        r = r - part
    return yhat


def grassmann_polish(model, bank_ids, dicts, mus, Ws, pinvs, L, steps, lr, batch, dev, seed):
    """Riemannian descent of the end-to-end KL (every site restricted at once, errors propagating) over the product
    of Grassmannians the atoms live on. Euclidean gradient G -> tangent G - O O^T G; Adam moments on the tangent
    vectors; QR retraction. No penalty: L is a hard constraint and the objective is the model's own KL."""
    params = [(l, s) for l in range(len(dicts)) for s in dicts[l]]
    for key in params:
        l, s = key
        dicts[l][s] = dicts[l][s].clone().requires_grad_(True)
    m1 = {key: torch.zeros_like(dicts[key[0]][key[1]]) for key in params}
    m2 = {key: torch.zeros_like(dicts[key[0]][key[1]]) for key in params}
    gen = torch.Generator().manual_seed(seed)
    trace = []

    def edit(l, s, z):
        y = (z - mus[l][s]) @ Ws[l][s].T
        return mus[l][s] + restricted(y, dicts[l][s], L) @ pinvs[l][s].T

    for step in range(steps):
        idx = torch.randint(0, len(bank_ids), (batch,), generator=gen)
        ids = bank_ids[idx].to(dev)
        with torch.no_grad():
            model.edit = None
            ref = torch.log_softmax(model(ids[:, :-1]).float(), -1)
        with torch.enable_grad():
            model.edit = edit
            lp = torch.log_softmax(model(ids[:, :-1]).float(), -1)
            model.edit = None
            loss = (ref.exp() * (ref - lp)).sum(-1).mean()
            grads = torch.autograd.grad(loss, [dicts[l][s] for l, s in params])
        trace.append(loss.item())
        with torch.no_grad():
            for key, g in zip(params, grads):
                l, s = key
                O = dicts[l][s]
                tang = g - torch.einsum("kqm,kpm,kpn->kqn", O, O, g)
                m1[key].mul_(0.9).add_(0.1 * tang)
                m2[key].mul_(0.999).add_(0.001 * tang.pow(2))
                upd = (m1[key] / (1 - 0.9 ** (step + 1))) / ((m2[key] / (1 - 0.999 ** (step + 1))).sqrt() + 1e-12)
                upd = upd - torch.einsum("kqm,kpm,kpn->kqn", O, O, upd)
                new = torch.linalg.qr(O - lr * upd)[0]
                dicts[l][s] = new.detach().requires_grad_(True)
        if step % 25 == 0:
            print(f"[dict4l] polish step {step}: train KL {loss.item():.4f}", flush=True)
    for key in params:
        l, s = key
        dicts[l][s] = dicts[l][s].detach()
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--bank-rows", type=int, required=True)
    parser.add_argument("--eval-rows", type=int, required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--dims", required=True, help="comma list of m (atom dimension)")
    parser.add_argument("--actives", required=True, help="comma list of L (atoms per token)")
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequential", action="store_true",
                        help="fit sites in network order on the stream already restricted upstream")
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--polish-steps", type=int, default=0)
    parser.add_argument("--polish-lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Target(args.weights, dev)
    test = rows(args.test, args.eval_rows)
    held = [test[i:i + args.batch] for i in range(0, len(test), args.batch)]
    report = {"args": vars(args), "vpd": {"p-8383f5e5": {"kl_rounded": 0.29135, "kl_stoch": 0.20661, "l0": 180.41}}}
    clean, ce = [], 0.0
    with torch.inference_mode():
        for ids in held:
            ids = ids.to(dev)
            logits = model(ids[:, :-1]).float()
            clean.append(torch.log_softmax(logits, -1).cpu())
            ce += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()
    report["clean_ce"] = ce / len(held)
    print(f"[dict4l] clean CE {report['clean_ce']:.4f}", flush=True)

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
    with torch.inference_mode():
        for i in range(0, len(bank_ids), args.batch):
            model(bank_ids[i:i + args.batch, :-1].to(dev))
    Z = [{s: torch.cat(model.grab[l][s]) for s in STREAMS} for l in range(model.n_layer)]
    model.grab = None
    mus = [{s: Z[l][s].mean(0) for s in STREAMS} for l in range(model.n_layer)]
    Ws = [{s: site_weight(model, l, s) for s in STREAMS} for l in range(model.n_layer)]
    pinvs = [{s: torch.linalg.pinv(Ws[l][s].double()).float() for s in STREAMS} for l in range(model.n_layer)]
    gen = torch.Generator().manual_seed(args.seed)
    report["arms"] = {}
    for m in [int(v) for v in args.dims.split(",")]:
        for L in [int(v) for v in args.actives.split(",")]:
            t0 = time.time()
            dicts = [{} for _ in range(model.n_layer)]
            fitted = set()

            def partial_edit(l, st, z):
                if (l, st) not in fitted:
                    return z
                y = (z - mus[l][st]) @ Ws[l][st].T
                _, yhat = pursue(y, dicts[l][st], L)
                return mus[l][st] + yhat @ pinvs[l][st].T

            for l in range(model.n_layer):
                for st in STREAMS:
                    if args.sequential:
                        model.grab = [{x: [] for x in STREAMS} for _ in range(model.n_layer)]
                        model.edit = partial_edit
                        with torch.inference_mode():
                            for i in range(0, len(bank_ids), args.batch):
                                model(bank_ids[i:i + args.batch, :-1].to(dev))
                        model.edit = None
                        Zs = torch.cat(model.grab[l][st])
                        model.grab = None
                    else:
                        Zs = Z[l][st]
                    Y = (Zs - mus[l][st]) @ Ws[l][st].T
                    atoms, hist = fit_dictionary(Y, args.atoms, m, L, args.sweeps, gen)
                    dicts[l][st] = atoms
                    fitted.add((l, st))
                    if l == 3 and st == "down_in":
                        tot = Y.pow(2).sum().item()
                        print(f"[dict4l] m={m} L={L} layer3 down: unexplained share by sweep "
                              f"{[round(h / tot, 4) for h in hist]}", flush=True)
            fit_s = time.time() - t0

            def edit(l, s, z):
                y = (z - mus[l][s]) @ Ws[l][s].T
                _, yhat = pursue(y, dicts[l][s], L)
                return mus[l][s] + yhat @ pinvs[l][s].T

            res = evaluate(edit)
            per_site = {}
            for l0 in range(model.n_layer):
                for s0 in STREAMS:
                    def only(l, s, z, l0=l0, s0=s0):
                        return edit(l, s, z) if (l, s) == (l0, s0) else z
                    per_site[f"{l0}:{s0}"] = evaluate(only)["kl"]
            res["per_site_alone_kl"] = per_site
            print(f"[dict4l] m={m} L={L} one site at a time (KL): "
                  f"{ {k: round(v, 4) for k, v in per_site.items()} }", flush=True)
            torch.save({f"{l}:{s}": dicts[l][s].cpu() for l in range(model.n_layer) for s in STREAMS},
                       args.out.replace(".json", f"_m{m}_L{L}_closed.pt"))
            if args.polish_steps > 0:
                closed = dict(res)
                print(f"[dict4l] m={m} L={L} closed form: KL {closed['kl']:.4f}", flush=True)
                t1 = time.time()
                trace = grassmann_polish(model, bank_ids, dicts, mus, Ws, pinvs, L, args.polish_steps,
                                         args.polish_lr, 4, dev, args.seed)
                res = evaluate(edit)
                res["closed_form"] = closed
                res["polish_seconds"] = time.time() - t1
                res["polish_trace"] = trace[::10]
            res.update({"atom_dim": m, "atoms_per_token": L, "atoms": args.atoms, "fit_seconds": fit_s,
                        "active_directions_per_token": m * L * 16, "active_matrix_slices_per_token": m * L * 24})
            if args.pgd_steps > 0:
                res["pgd_kl"] = pgd_masks(model, held[:1], clean[:1], dicts, mus, Ws, pinvs, L, args.pgd_steps, 0.1,
                                          dev, args.seed)
                print(f"[dict4l] m={m} L={L}: PGD-{args.pgd_steps} shared-mask KL {res['pgd_kl']:.4f} "
                      f"(VPD PGDReconLoss_20step 0.6045)", flush=True)
            report["arms"][f"m{m}_L{L}{'_seq' if args.sequential else ''}"] = res
            print(f"[dict4l] m={m} L={L}: KL {res['kl']:.4f}  dCE {res['ce_difference']:.4f}  "
                  f"(per token: {m * L} directions per site x 16 sites = {m * L * 16}; fit {fit_s:.0f}s)", flush=True)
            with open(args.out, "w") as handle:
                json.dump(report, handle, indent=1)
    print(f"[dict4l] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
