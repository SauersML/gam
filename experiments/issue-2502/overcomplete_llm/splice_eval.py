"""Causal judge for issue #2502: splice a dictionary's reconstruction back into
the model's residual stream and read the damage on the model's own loss.

Thin PyTorch wrapper: it substitutes bytes produced by the Rust fit into a
forward pass and reports cross-entropy. No modelling math.

For every arm, layer-`L` output at held-out position `t` is replaced by
`train_mean + reconstruction[row(t)]`, and the model's next-token cross-entropy
is measured over exactly the spliced positions. The `identity` arm splices the
harvested activation itself and must land at 0.000000 -- it is the positive
control that the splice path is wired correctly.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=RECON.f32")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-seqs", type=int, default=0, help="0 = all held-out sequences")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    meta = json.load(open(os.path.join(args.acts_dir, "meta.json")))
    layer = meta["layer"]
    skip = meta["skip_positions"]
    seq_len = meta["seq_len"]
    p = meta["hidden_size"]
    keep = seq_len - skip

    seqs = np.load(os.path.join(args.acts_dir, "eval_seqs.npy"))
    ctx = np.load(os.path.join(args.acts_dir, "eval_ctx.npy"))
    n_rows = ctx.shape[0]
    # Only whole sequences are spliceable: the final flush can truncate.
    n_seq_full = n_rows // keep
    for s in range(n_seq_full):
        if int(ctx[s * keep, 0]) != s or int(ctx[s * keep, 1]) != skip:
            raise SystemExit(f"row layout broken at sequence {s}: ctx={ctx[s * keep]}")
    n_seq = n_seq_full if args.max_seqs == 0 else min(args.max_seqs, n_seq_full)
    print(f"[splice] rows={n_rows} sequences={n_seq}/{n_seq_full} keep={keep} p={p}", flush=True)

    acts = np.load(os.path.join(args.acts_dir, "eval.npy"), mmap_mode="r")

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[splice] arm {name}: {path} missing, skipped", flush=True)
            continue
        mm = np.memmap(path, dtype=np.float32, mode="r")
        if mm.size % p != 0:
            raise SystemExit(f"{path}: {mm.size} floats is not a multiple of p={p}")
        arms[name] = (mm.reshape(-1, p), os.path.dirname(path))
        print(f"[splice] arm {name}: {arms[name][0].shape} from {path}", flush=True)

    mean = None
    for _name, (_mat, d) in arms.items():
        mpath = os.path.join(d, "train_mean.f32")
        if os.path.exists(mpath):
            mean = np.fromfile(mpath, dtype=np.float32)
            break
    if mean is None or mean.size != p:
        raise SystemExit("could not load a train_mean.f32 of the right width")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval().cuda()
    blocks = model.model.layers

    patch = {"tensor": None}

    def hook(_module, _inputs, output):
        if patch["tensor"] is None:
            return output
        was_tuple = isinstance(output, tuple)
        h = output[0] if was_tuple else output
        h = h.clone()
        h[:, skip:, :] = patch["tensor"].to(h.dtype)
        return (h,) + output[1:] if was_tuple else h

    handle = blocks[layer].register_forward_hook(hook)

    # `identity` is not an arm file: it splices the harvested activation itself
    # and is the positive control that the splice path is wired correctly.
    arm_names = ["identity"] + list(arms)
    results = {name: 0.0 for name in arm_names}
    results["clean"] = 0.0
    counts = 0

    with torch.no_grad():
        for start in range(0, n_seq, args.batch):
            stop = min(start + args.batch, n_seq)
            ids = torch.tensor(seqs[start:stop].astype(np.int64), device="cuda")
            # Rows for these sequences are contiguous: sequence s owns rows
            # [s*keep, (s+1)*keep) by construction of the harvest.
            row_slices = [slice(s * keep, (s + 1) * keep) for s in range(start, stop)]
            targets = ids[:, skip + 1 :]

            def score():
                logits = model(input_ids=ids, use_cache=False).logits[:, skip:-1, :]
                return torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()

            patch["tensor"] = None
            clean = score()
            results["clean"] += clean
            counts += targets.numel()

            for name in arm_names:
                if name == "identity":
                    block = np.stack([np.asarray(acts[sl]) for sl in row_slices])
                else:
                    mat = arms[name][0]
                    block = np.stack([np.asarray(mat[sl]) + mean for sl in row_slices])
                patch["tensor"] = torch.from_numpy(np.ascontiguousarray(block)).cuda()
                results[name] += score()
            patch["tensor"] = None
            if (start // args.batch) % 10 == 0:
                print(f"[splice] {stop}/{n_seq} sequences", flush=True)

    handle.remove()
    clean_ce = results["clean"] / counts
    report = {
        "model": args.model,
        "layer": layer,
        "sequences": n_seq,
        "scored_tokens": counts,
        "clean_ce": clean_ce,
        "arms": {
            name: {
                "ce": results[name] / counts,
                "delta_ce": results[name] / counts - clean_ce,
            }
            for name in arm_names
        },
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("[splice] " + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
