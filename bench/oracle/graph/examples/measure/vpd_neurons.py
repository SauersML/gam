"""Which MLP neurons of one vpd4l layer carry a behavior's answer, and how many pay for themselves:
each neuron's clean activation patched alone into the counterfactual run (all positions) gives its
recovery, KL(M_clean || M_cf) - KL(M_clean || M_patched) in bits per target token; then the top-k
neurons by recovery are patched together for k = 1, 2, 4, ..., and the k minimizing
KL(M_clean || M_patched top-k) + k * numbers per neuron * 1/2 log2 N / N (the opaque price per
token of declaring them, N = 2^24 scored tokens) is reported. The clean family is a proxy for the
checker's whole score.

  MPD_MEM_GIB=4 mem-lease 4 ~/mpd-data/venv/bin/python vpd_neurons.py OUT_DIR BEHAVIOR.json:LAYER [...]
(OUT_DIR/neurons_<behavior id>_l<LAYER>.json each; vpd4l is loaded once)
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # bench/oracle
import vpd_labels as VL  # noqa: E402
import vpd_model as VM  # noqa: E402

N = 2**24
CHUNK = 16


@torch.no_grad()
def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = VL.device()
    t = VL.Model(dev).t
    for job in sys.argv[2:]:
        if not job.endswith(".json") and ":" in job:
            path, layer = job.rsplit(":", 1)
            beh = json.loads(Path(path).read_text())
            neurons(t, dev, beh, int(layer), out_dir / f"neurons_{beh['id']}_l{layer}.json")


def neurons(t, dev, beh: dict, layer: int, out: Path):
    H, D = t.n_head, t.hd
    prompts = [p for p in beh["prompts"] if p.get("counterfactual")]
    T = max(len(p["token_ids"]) for p in prompts)

    def batch(key):
        ids = torch.zeros(len(prompts), T, dtype=torch.long)
        for i, p in enumerate(prompts):
            x = (p if key == "clean" else p["counterfactual"])["token_ids"]
            ids[i, : len(x)] = torch.tensor(x)
        return ids.to(dev)

    clean, counterfactual = batch("clean"), batch("cf")
    rows = torch.tensor([i for i, p in enumerate(prompts) for _ in p["target_positions"]], device=dev)
    cols = torch.tensor([c for p in prompts for c in p["target_positions"]], device=dev)
    B = len(prompts)

    def site(i, kind, h):
        return t.site(f"h.{i}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}")(h)

    def attention(i, x):
        Bx, L, _ = x.shape
        h = VM.rms(x, t.norms[2 * i], t.eps)
        q = site(i, "q_proj", h).view(Bx, L, H, D).transpose(1, 2)
        k = site(i, "k_proj", h).view(Bx, L, H, D).transpose(1, 2)
        v = site(i, "v_proj", h).view(Bx, L, H, D).transpose(1, 2)
        q, k = t._rope(q, L), t._rope(k, L)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(Bx, L, -1)
        return x + site(i, "o_proj", y)

    def hidden(i, x):
        return VM.gelu_tanh(site(i, "c_fc", VM.rms(x, t.norms[2 * i + 1], t.eps)))

    def finish(i, x, hid, reps):
        """The rest of the model from layer i's MLP output on, for `reps` copies of the batch."""
        x = x + site(i, "down_proj", hid)
        for j in range(i + 1, t.n_layer):
            x = attention(j, x)
            x = x + site(j, "down_proj", hidden(j, x))
        r = torch.cat([rows + b * B for b in range(reps)])
        c = cols.repeat(reps)
        return torch.log_softmax((VM.rms(x, t.ln_f, t.eps)[r, c] @ t.wte.T).float(), -1)

    def until(ids):
        x = t.wte[ids]
        for i in range(layer):
            x = attention(i, x)
            x = x + site(i, "down_proj", hidden(i, x))
        x = attention(layer, x)
        return x, hidden(layer, x)

    xc, hc = until(clean)
    xf, hf = until(counterfactual)
    lp_clean = finish(layer, xc, hc, 1)
    p_clean = lp_clean.exp()
    targets = len(rows)

    def kl(lp, reps):
        per = (p_clean.repeat(reps, 1) * (lp_clean.repeat(reps, 1) - lp)).sum(-1) / math.log(2)
        return per.view(reps, targets).mean(1)

    base = kl(finish(layer, xf, hf, 1), 1).item()
    width = hf.shape[-1]
    recovery = torch.zeros(width)
    for s in range(0, width, CHUNK):
        idx = torch.arange(s, min(s + CHUNK, width), device=dev)
        n = len(idx)
        hid = hf.repeat(n, 1, 1)
        for j, neuron in enumerate(idx):
            hid[j * B : (j + 1) * B, :, neuron] = hc[:, :, neuron]
        recovery[s : s + n] = base - kl(finish(layer, xf.repeat(n, 1, 1), hid, n), n).cpu()
    order = torch.argsort(recovery, descending=True)
    numbers = 2 * t.wte.shape[1]  # a vpd4l neuron: its c_fc row and its down_proj column
    price = numbers * 0.5 * math.log2(N) / N  # bits per token per declared neuron
    curve = []
    k = 1
    while k <= width:
        hid = hf.clone()
        top = order[:k].to(dev)
        hid[:, :, top] = hc[:, :, top]
        left = kl(finish(layer, xf, hid, 1), 1).item()
        curve.append({"k": k, "kl_bits": left, "opaque_bits_per_token": k * price, "total": left + k * price})
        k *= 2
    best = min(curve, key=lambda c: c["total"])
    print(f"{beh['id']} layer {layer}: base {base:.3f} bits; best k {best['k']} (KL {best['kl_bits']:.3f} + opaque "
          f"{best['opaque_bits_per_token']:.3f}); top neurons {order[:8].tolist()}", flush=True)
    out.write_text(json.dumps({"behavior": beh["id"], "layer": layer, "base_bits": base, "price_per_neuron": price,
                               "recovery": recovery.tolist(), "order": order.tolist(), "curve": curve, "best": best}))


if __name__ == "__main__":
    main()
