"""Measured removal effect of every VPD subcomponent of vpd4l on induction (repeated Pile text):
KL(M || M_e) in bits per target token, full vocabulary, M_e = M with W - u_i v_i^T at one site (every
token). Sequences: held-out val rows 1024.. (the contracts' held-out rows), first HALF tokens, repeated;
targets = second-half positions predicting the repeated next token.

  mem-lease 8 ~/mpd-data/venv/bin/python vpd_induction_removal.py OUT.json
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.home() / "gam/bench/oracle"))
import vpd_labels as VL  # noqa: E402
import vpd_model as VM  # noqa: E402

HALF, B, C = 48, 8, 8


@torch.no_grad()
def main():
    out = Path(sys.argv[1])
    dev = VL.device()
    model = VL.Model(dev)
    uv = VL.load_uv(dev, Path.home() / "mpd-data/oracle/vpd/uv.safetensors")
    rows = VM.val_tokens(B, seq=HALF, offset=1024)
    ids = torch.cat([rows, rows], 1).to(dev)
    targets = torch.arange(HALF, 2 * HALF - 1, device=dev)  # position t predicts ids[t + 1] = ids[t + 1 - HALF]
    entering, final = model.clean(ids)
    lp0 = model.log_probs(final[:, targets])  # [B, P, V]
    p0 = lp0.exp()
    acc = (lp0.argmax(-1) == ids[:, targets + 1]).float().mean().item()
    nll = -lp0.gather(-1, ids[:, targets + 1, None]).mean().item() / math.log(2)
    print(f"clean: next-token accuracy {acc:.3f}, {nll:.3f} bits per target token", flush=True)
    result = {"half": HALF, "sequences": B, "offset": 1024, "clean_accuracy": acc, "clean_bits": nll, "sites": {}}
    t0 = time.time()
    names = VM.site_names(model.t.n_layer)
    for name in [n for n in names if ".attn." in n] + [n for n in names if ".mlp." in n]:  # attention first
        layer = int(name.split(".")[1])
        U, V = uv[name]  # U [C, d_out], V [d_in, C]
        effects, drops = [], []
        for s in range(0, U.shape[0], C):
            idx = torch.arange(s, min(s + C, U.shape[0]), device=dev)
            n = len(idx)
            v = V[:, idx].T.repeat_interleave(B, 0)
            u = U[idx].repeat_interleave(B, 0)
            coef = -torch.ones(n * B, device=dev)
            x = entering[layer].repeat(n, 1, 1)
            h, _ = model.from_layer(layer, x, (name, v, u, coef))
            lp = model.log_probs(h[:, targets]).view(n, B, len(targets), -1)
            kl = (p0[None] * (lp0[None] - lp)).sum(-1) / math.log(2)  # [n, B, P]
            effects += kl.mean((1, 2)).tolist()
            gold = ids[:, targets + 1][None, :, :, None].expand(n, -1, -1, 1)
            drops += ((lp0.gather(-1, ids[:, targets + 1, None])[None] - lp.gather(-1, gold)).mean((1, 2, 3))
                      / math.log(2)).tolist()
        order = sorted(range(len(effects)), key=lambda i: -effects[i])
        result["sites"][name] = {"kl_bits": effects, "gold_drop_bits": drops, "top": order[:32]}
        print(f"{name}: top {[(i, round(effects[i], 4)) for i in order[:6]]} ({time.time() - t0:.0f} s)", flush=True)
        out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
