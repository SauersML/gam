"""#2951 on the trained modular-addition transformer: the structure the plane decomposition exposes.

Analysis under SPEC 8's exception (benchmark evaluation, not an MPD input). Reads checkpoints from
``bench/mpd_modadd_2951.py train``; writes one JSON receipt. Every quantity is an exact evaluation
over a stated finite family unless it says otherwise.

1. Redundancy (P12). For the top-``R`` planes of ``W_U`` and of ``W_E`` (the table at every use), every
   subset is deleted in turn, the rest kept. Per test pair: which kept sets preserve the argmax and
   stay within ``--eps`` nats of KL; the minimal such sets (the antichain a hitting-set support is
   read from); whether any single plane is necessary. Deleting ``W_U`` planes is convex in the masks,
   so for ``W_U`` a binary kept set that fails also fails for every interior mask below it.
2. Margin law. Per test pair, the exact worst deletion of the ``F`` largest non-key ``W_U`` planes
   (vertex enumeration, convex), against the pair's logit margin.
3. Interactions. Embedding plane ``k`` of operand ``a`` alone is turned by ``theta`` or scaled by
   ``lambda``; the response of every key output harmonic ``k'`` is the median complex ratio of the
   output's Fourier coefficient (in ``c - a - b``) to the unedited one.
4. Formation. Items 1 and 3's diagonal at every stored checkpoint.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import torch

from mpd_modadd_2951 import build_model, all_pairs
from mpd_modadd_adversary_2951 import final_residual, kl_rows, unembed_planes
from mpd_modadd_paths_2951 import planes_of, spectrum_of_logits


def subsets(r, dev):
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=r)), dtype=torch.float64, device=dev)


def minimal_sets(ok_rows, kept):
    """``ok_rows``: n x V bool; ``kept``: V x r (1 = kept). Per row, the inclusion-minimal kept sets."""
    sizes = kept.sum(1)
    out = []
    for row in ok_rows:
        good = [i for i in torch.nonzero(row).flatten().tolist()]
        good.sort(key=lambda i: sizes[i].item())
        mins = []
        for i in good:
            s = kept[i].bool()
            if not any(bool((m <= s).all()) for m in mins):
                mins.append(s)
        out.append(mins)
    return out


def summarize_minimal(mins, labels):
    sizes = [min(int(m.sum()) for m in ms) if ms else -1 for ms in mins]
    counts = [len(ms) for ms in mins]
    necessary = torch.zeros(len(labels))
    for ms in mins:
        if ms:
            inter = torch.stack(ms).all(0)
            necessary += inter.double().cpu()
    freq = {}
    for ms in mins:
        for m in ms:
            key = tuple(l for l, b in zip(labels, m.tolist()) if b)
            freq[key] = freq.get(key, 0) + 1
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:12]
    hist = {}
    for c in counts:
        hist[c] = hist.get(c, 0) + 1
    return {
        "rows": len(mins),
        "rows_with_no_sufficient_set": sum(1 for s in sizes if s < 0),
        "smallest_sufficient_size_hist": {str(k): sizes.count(k) for k in sorted(set(sizes))},
        "number_of_minimal_sets_hist": {str(k): v for k, v in sorted(hist.items())},
        "planes_necessary_fraction": dict(zip([str(l) for l in labels], (necessary / len(mins)).tolist())),
        "most_common_minimal_sets": [[list(k), v] for k, v in top],
    }


def redundancy_unembed(h, z0, log_p0, waves, order, R, eps, dev):
    top = order[:R]
    kept = subsets(R, dev)
    W = waves[:, top]
    label = z0.argmax(-1)
    oks = []
    for s in range(0, W.shape[0], 512):
        logits = z0[s:s + 512, None, :] - torch.einsum("vf,nfp->nvp", 1.0 - kept, W[s:s + 512])
        kl = kl_rows(log_p0[s:s + 512, None, :], logits)
        oks.append((kl <= eps) & (logits.argmax(-1) == label[s:s + 512, None]))
    return summarize_minimal(minimal_sets(torch.cat(oks).cpu(), kept.cpu()), (top + 1).tolist())


def redundancy_embed(model, pairs, table, mean, planes, order, R, eps, dev, p):
    top = order[:R]
    kept = subsets(R, dev)
    positions = torch.arange(p, dtype=torch.float64, device=dev)
    ang = 2 * math.pi * (top + 1).double()[:, None] * positions[None, :] / p  # R x p
    content = torch.cos(ang)[:, :, None] * planes[top][:, None, :, 0] + torch.sin(ang)[:, :, None] * planes[top][:, None, :, 1]
    with torch.inference_mode():
        base = model(pairs)
        log_p0 = torch.log_softmax(base, -1)
        label = base.argmax(-1)
        cols = []
        for v in range(kept.shape[0]):
            edited = table.clone()
            edited[:p] -= torch.einsum("r,rpd->pd", 1.0 - kept[v], content)
            logits = model(pairs, embed_at={u: edited for u in (0, 1, 2)})
            kl = kl_rows(log_p0, logits)
            cols.append((kl <= eps) & (logits.argmax(-1) == label))
    ok = torch.stack(cols, 1)
    return summarize_minimal(minimal_sets(ok.cpu(), kept.cpu()), (top + 1).tolist())


def interactions(model, pairs, table, planes, order_out, keys_in, p, dev):
    """R[k'][k] for a turn of pi/2 and the gain curve of a scale, operand a only (use site pos0)."""
    with torch.inference_mode():
        base_spec = spectrum_of_logits(model(pairs).cpu(), pairs.cpu(), p)[:, order_out]
        positions = torch.arange(p, dtype=torch.float64, device=dev)
        turn, gains = {}, {}
        for k in keys_in.tolist():
            ang = 2 * math.pi * (k + 1) * positions / p
            u = planes[k]  # d x 2
            content = torch.cos(ang)[:, None] * u[None, :, 0] + torch.sin(ang)[:, None] * u[None, :, 1]
            rotated = -torch.sin(ang)[:, None] * u[None, :, 0] + torch.cos(ang)[:, None] * u[None, :, 1]  # turn by +pi/2
            edited = table.clone()
            edited[:p] += rotated - content
            spec = spectrum_of_logits(model(pairs, embed_at={0: edited}).cpu(), pairs.cpu(), p)[:, order_out]
            ratio = spec / base_spec
            turn[str(k + 1)] = {"abs": ratio.abs().median(0).values.tolist(), "angle": torch.angle(ratio).median(0).values.tolist()}
            curve = []
            for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                edited = table.clone()
                edited[:p] += (lam - 1.0) * content
                spec = spectrum_of_logits(model(pairs, embed_at={0: edited}).cpu(), pairs.cpu(), p)[:, order_out]
                curve.append({"lambda": lam, "amplitude_ratio": (spec / base_spec).abs().median(0).values.tolist()})
            gains[str(k + 1)] = curve
    return {"output_frequencies": (order_out + 1).tolist(), "turn_pi_over_2": turn, "gain": gains}


def margin_law(z0, log_p0, waves, rest, F, dev):
    W = waves[:, rest[:F]]
    verts = subsets(F, dev)
    best = torch.empty(W.shape[0], dtype=torch.float64, device=dev)
    for s in range(0, W.shape[0], 64):
        logits = z0[s:s + 64, None, :] - torch.einsum("vf,nfp->nvp", verts, W[s:s + 64])
        best[s:s + 64] = kl_rows(log_p0[s:s + 64, None, :], logits).max(1).values
    top2 = z0.topk(2, -1).values
    margin = top2[:, 0] - top2[:, 1]
    lb, lm = best.clamp_min(1e-300).log(), margin
    rank = lambda x: x.argsort().argsort().double()
    rb, rm = rank(lb), rank(lm)
    spearman = (((rb - rb.mean()) * (rm - rm.mean())).mean() / (rb.std() * rm.std())).item()
    A = torch.stack([torch.ones_like(lm), lm], 1)
    coef = torch.linalg.lstsq(A, lb[:, None]).solution.flatten().tolist()
    bins = torch.quantile(margin, torch.linspace(0, 1, 11, dtype=torch.float64, device=dev))
    binned = []
    for i in range(10):
        sel = (margin >= bins[i]) & (margin <= bins[i + 1])
        binned.append({"margin_lo": bins[i].item(), "margin_hi": bins[i + 1].item(),
                       "sup_median": best[sel].median().item(), "sup_max": best[sel].max().item()})
    return {"F": F, "spearman_logsup_vs_margin": spearman, "fit_log_sup_eq_a_plus_b_margin": coef, "deciles": binned,
            "sup_max": best.max().item(), "margin_at_argmax_sup": margin[best.argmax()].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--redundancy-planes", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--margin-free", type=int, required=True)
    parser.add_argument("--keys", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dev = torch.device(args.device)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    p = config["p"]
    pairs = all_pairs(p)[run["test_idx"]].to(dev)
    report = {"p": p, "eps": args.eps, "checkpoints": {}}
    last = max(run["checkpoints"])
    for step in sorted(run["checkpoints"]):
        model = build_model(config)
        model.load_state_dict(run["checkpoints"][step])
        model = model.double().eval().to(dev)
        with torch.inference_mode():
            h = final_residual(model, pairs)
            z0 = h @ model.W_U.T
            log_p0 = torch.log_softmax(z0, -1)
            acc = (z0.argmax(-1) == (pairs[:, 0] + pairs[:, 1]) % p).double().mean().item()
            umean, uplanes, cos, sin = unembed_planes(model.W_U.detach(), p)
            upower = uplanes.pow(2).sum((1, 2))
            uorder = torch.argsort(upower, descending=True)
            coeff = torch.einsum("kdj,nd->nkj", uplanes, h)
            waves = coeff[..., 0, None] * cos[None] + coeff[..., 1, None] * sin[None]
            table = model.W_E.detach().clone()
            emean, eplanes = planes_of(table.cpu(), p)
            eplanes = eplanes.to(dev)
            epower = eplanes.pow(2).sum((1, 2))
            eorder = torch.argsort(epower, descending=True).to(dev)
        entry = {
            "test_acc": acc,
            "unembed_power_top": [[int(k) + 1, upower[k].item()] for k in uorder[:10]],
            "embed_power_top": [[int(k) + 1, epower[k].item()] for k in eorder[:10]],
            "unembed_power_share_top5": (upower[uorder[:5]].sum() / upower.sum()).item(),
            "embed_power_share_top5": (epower[eorder[:5]].sum() / epower.sum()).item(),
            "redundancy_unembed": redundancy_unembed(h, z0, log_p0, waves, uorder, args.redundancy_planes, args.eps, dev),
            "redundancy_embed": redundancy_embed(model, pairs, table.double(), emean, eplanes, eorder,
                                                 args.redundancy_planes, args.eps, dev, p),
        }
        if step == last:
            entry["margin_law"] = margin_law(z0, log_p0, waves, uorder[args.keys:], args.margin_free, dev)
            entry["interactions"] = interactions(model, pairs, table.double(), eplanes, uorder[:8].cpu(), eorder[:8], p, dev)
        report["checkpoints"][str(step)] = entry
        r = entry["redundancy_unembed"]
        e = entry["redundancy_embed"]
        print(f"[step {step}] acc={acc:.4f} U top5 share={entry['unembed_power_share_top5']:.3f} E top5 share={entry['embed_power_share_top5']:.3f} "
              f"U min-size hist={r['smallest_sufficient_size_hist']} U #minimal={r['number_of_minimal_sets_hist']} "
              f"E min-size hist={e['smallest_sufficient_size_hist']} E #minimal={e['number_of_minimal_sets_hist']}", flush=True)
        print(f"   U necessary={ {k: round(v, 3) for k, v in r['planes_necessary_fraction'].items()} } top sets={r['most_common_minimal_sets'][:4]}", flush=True)
        print(f"   E necessary={ {k: round(v, 3) for k, v in e['planes_necessary_fraction'].items()} } top sets={e['most_common_minimal_sets'][:4]}", flush=True)
        if "margin_law" in entry:
            m = entry["margin_law"]
            print(f"   margin law: spearman={m['spearman_logsup_vs_margin']:.3f} fit={m['fit_log_sup_eq_a_plus_b_margin']} sup_max={m['sup_max']:.3g} at margin {m['margin_at_argmax_sup']:.3g}", flush=True)
            it = entry["interactions"]
            print(f"   output freqs {it['output_frequencies']}", flush=True)
            for k, v in it["turn_pi_over_2"].items():
                print(f"   turn plane {k}: |R|={[round(x, 3) for x in v['abs']]} arg={[round(x, 3) for x in v['angle']]}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[structure] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
