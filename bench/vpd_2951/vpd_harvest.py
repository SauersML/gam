"""Harvest the 4L Pile target's next-token behaviour for behaviour discovery (#2951,
crates/gam-sae/examples/mpd_behaviors_vpd_2951.rs).

usage: vpd_harvest.py N_DOCS OUT_DIR

A row is (doc, t) for t in 2..511 of the first 512 tokens of each doc, docs in file order, so
the first n rows are a prefix of whole documents. Pass 1 accumulates u, the family's mean
next-token distribution; pass 2 streams per row: feat (x_t, x_{t-1}, x_{t-2}) u32, top-64 ids
u32 and probabilities f32, tail mass f64, H(p) f64 nats, -sum p log u f64 nats, and per layer-head
the argmax key offset t - argmax_k (u8, clipped at 255). base.bin is u (f64).
"""

import json
import sys
import time
from pathlib import Path

import torch

from vpd_model import load_target, val_tokens

TOP = 64
MB = 4
T0 = 2

n_docs, out = int(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
dev = "mps"
target = load_target(dev)
L, H = target.n_layer, target.n_head
vocab = target.wte.shape[0]
for i in range(L):
    for k in ("q_proj", "k_proj"):
        target.site(f"h.{i}.attn.{k}").cache_output = True


def batches():
    for d in range(0, n_docs, MB):
        yield val_tokens(min(MB, n_docs - d), offset=d)


t0 = time.time()
u = torch.zeros(vocab, dtype=torch.float64)
rows = 0
with torch.no_grad():
    for ids in batches():
        p = target(ids.to(dev))[:, T0:].float().softmax(-1)
        u += p.sum((0, 1)).cpu().double()
        rows += p.shape[0] * p.shape[1]
        del p
u /= rows
print(f"[{time.time() - t0:.0f}s] pass 1: {rows} rows", flush=True)
log_u = u.clamp_min(1e-300).log().float().to(dev)
u.numpy().astype("<f8").tofile(out / "base.bin")

files = {n: open(out / f"{n}.bin", "wb") for n in ("feat", "top_ids", "top_p", "tail", "ent", "xent", "heads")}
with torch.no_grad():
    for ids in batches():
        B, T = ids.shape
        logp = target(ids.to(dev))[:, T0:].float().log_softmax(-1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)
        xent = -(p * log_u).sum(-1)
        tp, ti = p.topk(TOP, -1)
        tail = (1.0 - tp.cpu().double().sum(-1)).clamp_min(0.0)
        offs = []
        for i in range(L):
            a = target.attention_pattern(
                target.site(f"h.{i}.attn.q_proj").last_output, target.site(f"h.{i}.attn.k_proj").last_output
            )[:, :, T0:]
            k = a.argmax(-1)
            offs.append((torch.arange(T0, T, device=dev)[None, None, :] - k).clamp(0, 255))
            del a
        heads = torch.stack(offs, 1).reshape(B, L * H, T - T0).permute(0, 2, 1).to(torch.uint8)
        feat = torch.stack([ids[:, T0:], ids[:, T0 - 1:T - 1], ids[:, :T - T0]], -1).to(torch.int64)
        files["feat"].write(feat.numpy().astype("<u4").tobytes())
        files["top_ids"].write(ti.cpu().numpy().astype("<u4").tobytes())
        files["top_p"].write(tp.cpu().numpy().astype("<f4").tobytes())
        files["tail"].write(tail.numpy().astype("<f8").tobytes())
        files["ent"].write(ent.cpu().double().numpy().astype("<f8").tobytes())
        files["xent"].write(xent.cpu().double().numpy().astype("<f8").tobytes())
        files["heads"].write(heads.cpu().numpy().tobytes())
        del logp, p
for f in files.values():
    f.close()
(out / "meta.json").write_text(json.dumps({"rows": rows, "top": TOP, "vocab": vocab, "heads": L * H, "docs": n_docs}))
print(f"[{time.time() - t0:.0f}s] pass 2 done: {rows} rows -> {out}", flush=True)
