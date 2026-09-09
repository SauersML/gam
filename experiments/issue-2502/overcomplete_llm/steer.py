"""Causal steering with the Rust dictionary's atoms (issue #2502, criterion 3).

Split-half design, so the tokens a block is claimed to control are never the
tokens the claim is read off:

* **discovery (half A)** — at a fixed unit dose, add the block's own mean firing
  direction to the layer-`L` residual stream and record the mean change in
  next-token log-probability for every vocabulary item. The block's target set is
  the top of that list.
* **confirmation (half B)** — sweep the dose and report the mean log-probability
  change on the target set against a frequency-matched control set, on
  *different* sequences. A direction that steers shows a monotone dose response
  on its targets and a flat one on the controls.
* **ablation** — at positions where the block actually fires, remove its own
  subspace component and read the same statistic.

The dose is expressed in the block's own units: `alpha x mean_gate x u`, where
`mean_gate` is the block's mean held-out `||z_g||` and `u` its unit mean firing
direction. Nothing here is a free constant.

Thin PyTorch wrapper: the directions and the codes come from the Rust fit.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--atoms", type=int, required=True)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--blocks", type=int, default=12, help="how many blocks to steer")
    ap.add_argument("--targets", type=int, default=10)
    ap.add_argument("--seqs-per-half", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--doses", default="-2,-1,-0.5,0,0.5,1,2,4")
    ap.add_argument("--angle-blocks", type=int, default=4,
                    help="b=2 blocks whose chart angle is swept causally")
    ap.add_argument("--angle-bins", type=int, default=24)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    meta = json.load(open(os.path.join(args.acts_dir, "meta.json")))
    layer, skip, seq_len, p = (
        meta["layer"],
        meta["skip_positions"],
        meta["seq_len"],
        meta["hidden_size"],
    )
    keep = seq_len - skip
    b, k = args.block_size, args.topk
    g = args.atoms // b

    decoder = np.fromfile(os.path.join(args.fit_dir, "decoder.f32"), dtype=np.float32)
    decoder = decoder.reshape(args.atoms, p)
    blocks = np.fromfile(os.path.join(args.fit_dir, "eval_blocks.u32"), dtype=np.uint32)
    codes = np.fromfile(os.path.join(args.fit_dir, "eval_codes.f32"), dtype=np.float32)
    n = blocks.size // k
    blocks = blocks.reshape(n, k)
    codes = codes.reshape(n, k, b)
    seqs = np.load(os.path.join(args.acts_dir, "eval_seqs.npy"))
    tokens = np.load(os.path.join(args.acts_dir, "eval_tokens.npy"))[:n]

    counts = np.bincount(blocks.reshape(-1), minlength=g)
    chosen = np.argsort(-counts)[: args.blocks]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval().cuda()
    layers = model.model.layers

    n_seq = n // keep
    half = min(args.seqs_per_half, n_seq // 2)
    seqs_a = list(range(half))
    seqs_b = list(range(n_seq // 2, n_seq // 2 + half))
    doses = [float(x) for x in args.doses.split(",")]
    print(
        f"[steer] blocks={len(chosen)} half_A={len(seqs_a)} half_B={len(seqs_b)} "
        f"doses={doses}",
        flush=True,
    )

    intervention = {"vec": None, "ablate": None}

    def hook(_module, _inputs, output):
        was_tuple = isinstance(output, tuple)
        h = output[0] if was_tuple else output
        if intervention["vec"] is None and intervention["ablate"] is None:
            return output
        h = h.clone()
        if intervention["ablate"] is not None:
            frame = intervention["ablate"]  # (b, p) orthonormal
            proj = torch.einsum("btp,rp->btr", h[:, skip:, :].float(), frame)
            h[:, skip:, :] -= torch.einsum("btr,rp->btp", proj, frame).to(h.dtype)
        if intervention["vec"] is not None:
            h[:, skip:, :] += intervention["vec"].to(h.dtype)
        return (h,) + output[1:] if was_tuple else h

    handle = layers[layer].register_forward_hook(hook)

    def logprobs(seq_ids):
        """Mean next-token log-probs over the spliced positions, summed over the batch."""
        out = None
        cnt = 0
        for start in range(0, len(seq_ids), args.batch):
            batch_ids = seq_ids[start : start + args.batch]
            ids = torch.tensor(seqs[batch_ids].astype(np.int64), device="cuda")
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits[:, skip:-1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                s = lp.reshape(-1, lp.shape[-1]).sum(0)
                cnt += lp.shape[0] * lp.shape[1]
            out = s if out is None else out + s
        return (out / cnt).cpu().numpy()

    intervention["vec"] = None
    intervention["ablate"] = None
    clean_a = logprobs(seqs_a)
    clean_b = logprobs(seqs_b)
    # Frequency table over the model's own logit width, so a control token can be
    # matched to any target the sweep surfaces.
    vocab = clean_a.size
    base_counts = np.bincount(tokens, minlength=vocab)[:vocab].astype(np.float64)

    report = {"layer": layer, "doses": doses, "blocks": []}
    for gid in chosen:
        gid = int(gid)
        hit_r, hit_j = np.nonzero(blocks == gid)
        if hit_r.size < 50:
            continue
        z = codes[hit_r, hit_j, :]
        frame = decoder[gid * b : (gid + 1) * b, :]  # (b, p), orthonormal rows
        firing_dirs = z @ frame  # (m, p)
        mean_dir = firing_dirs.mean(axis=0)
        norm = float(np.linalg.norm(mean_dir))
        if norm == 0.0:
            continue
        unit = mean_dir / norm
        mean_gate = float(np.linalg.norm(z, axis=1).mean())
        unit_t = torch.from_numpy(unit).cuda()
        frame_t = torch.from_numpy(np.ascontiguousarray(frame)).cuda()

        # Discovery on half A at unit dose.
        intervention["vec"] = (mean_gate * unit_t).view(1, 1, p)
        dlp_a = logprobs(seqs_a) - clean_a
        intervention["vec"] = None
        targets = np.argsort(-dlp_a)[: args.targets]
        # Frequency-matched controls: for each target, the token with the closest
        # held-out count that is not itself a target.
        controls = []
        used = set(int(t) for t in targets)
        order = np.argsort(base_counts)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        for t in targets:
            r = rank[t]
            for step in range(1, 400):
                for cand_rank in (r + step, r - step):
                    if 0 <= cand_rank < order.size:
                        cand = int(order[cand_rank])
                        if cand not in used:
                            controls.append(cand)
                            used.add(cand)
                            break
                else:
                    continue
                break
        controls = np.array(controls, dtype=np.int64)

        curve = []
        for alpha in doses:
            intervention["vec"] = (
                None if alpha == 0.0 else (alpha * mean_gate * unit_t).view(1, 1, p)
            )
            dlp_b = logprobs(seqs_b) - clean_b
            curve.append(
                {
                    "dose": alpha,
                    "target_mean_dlogp": float(dlp_b[targets].mean()),
                    "control_mean_dlogp": float(dlp_b[controls].mean()),
                }
            )
            intervention["vec"] = None

        # Ablation: remove the block's own subspace everywhere on half B.
        intervention["ablate"] = frame_t
        dlp_abl = logprobs(seqs_b) - clean_b
        intervention["ablate"] = None

        report["blocks"].append(
            {
                "block": gid,
                "firings": int(hit_r.size),
                "mean_gate": mean_gate,
                "targets": [
                    {"token_id": int(t), "token": tok.decode([int(t)])} for t in targets
                ],
                "controls": [int(c) for c in controls],
                "dose_curve": curve,
                "ablation_target_mean_dlogp": float(dlp_abl[targets].mean()),
                "ablation_control_mean_dlogp": float(dlp_abl[controls].mean()),
            }
        )
        print(
            f"[steer] block {gid}: targets={[tok.decode([int(t)]) for t in targets[:5]]} "
            f"dose+1 target={curve[doses.index(1.0)]['target_mean_dlogp']:.4f} "
            f"control={curve[doses.index(1.0)]['control_mean_dlogp']:.4f} "
            f"ablate target={float(dlp_abl[targets].mean()):.4f}",
            flush=True,
        )

    # Angular sweep: walk the CHART of a b=2 block -- inject r_bar*(cos(t)*d0 +
    # sin(t)*d1) at every angle -- and read which tokens the model boosts. If the
    # block were a direction detector the answer would not depend on t.
    angle_report = []
    for e in report["blocks"][: args.angle_blocks]:
        gid = e["block"]
        frame = decoder[gid * b : (gid + 1) * b, :]
        if frame.shape[0] != 2:
            break
        mean_gate = e["mean_gate"]
        bins = []
        for t in np.linspace(-np.pi, np.pi, args.angle_bins, endpoint=False):
            direction = np.cos(t) * frame[0] + np.sin(t) * frame[1]
            vec = torch.from_numpy(np.ascontiguousarray(mean_gate * direction)).cuda()
            intervention["vec"] = vec.view(1, 1, p)
            d = logprobs(seqs_b) - clean_b
            intervention["vec"] = None
            top = np.argsort(-d)[:8]
            bins.append(
                {
                    "theta": float(t),
                    "top_tokens": [tok.decode([int(x)]) for x in top],
                    "top_token_ids": [int(x) for x in top],
                    "top_dlogp": [float(d[int(x)]) for x in top],
                }
            )
        distinct = len({bb["top_tokens"][0] for bb in bins})
        angle_report.append(
            {"block": gid, "bins": bins, "distinct_top1_tokens_over_angles": distinct}
        )
        print(
            f"[steer] block {gid} angular sweep: {distinct}/{args.angle_bins} distinct "
            f"top-1 tokens across the chart",
            flush=True,
        )
    report["angle_sweep"] = angle_report

    handle.remove()
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[steer] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
