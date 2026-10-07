"""Counterfactual quality of each behavior (MPD #2951 graph oracle).

Undeclared pieces of a program stand in with their values on counterfactual prompts, so a behavior whose counterfactual
barely moves the model gives the empty program almost no error to explain. For every prompt pair (x, x') this measures,
at the target positions, whether the model's top token changes and KL(M(x) || M(x')) in bits; per behavior it reports
the fraction of prompts whose top answer changes at any target and the mean KL over targets. Results go into each
behavior file ("counterfactual_quality") and into <root>/cf_quality.tsv.

    mem-lease 6 ~/mpd-data/venv/bin/python bench/oracle/graph/behaviors/cf_quality.py --model qwen3-0.6b
    mem-lease 2 ~/mpd-data/venv/bin/python bench/oracle/graph/behaviors/cf_quality.py --model vpd4l --device cpu --max-tokens 4096
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import HF, OUT, Model  # noqa: E402


def additive_mask(lengths: list[int], T: int, blocks: list[list[list[int]]], dtype) -> torch.Tensor:
    """[B, 1, T, T] additive mask: causal within each sequence's real tokens, padding masked, attention_block ranges
    masked (queries in [q0, q1) do not see keys in [k0, k1))."""
    neg = torch.finfo(dtype).min
    m = torch.full((len(lengths), 1, T, T), neg, dtype=dtype)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for i, (n, blk) in enumerate(zip(lengths, blocks)):
        allow = causal.clone()
        allow[:, n:] = False
        for q0, q1, k0, k1 in blk:
            allow[q0:q1, k0:k1] = False
        allow[n:, :] = torch.eye(T, dtype=torch.bool)[n:, :]  # padding queries see themselves (no NaN rows)
        m[i, 0][allow] = 0
    return m


@torch.no_grad()
def pair_stats(model: Model, ps: list[dict], max_tokens: int) -> tuple[list[bool], list[float]]:
    """Per prompt whether the top token changes at any target, and per target KL(M(x) || M(x')) in bits. Each prompt's
    x and x' run in the same forward pass, so only these numbers leave the device."""
    changed, kls = [], []
    T = max(len(p["token_ids"]) for p in ps)
    chunk = max(1, max_tokens // (2 * T))
    for c0 in range(0, len(ps), chunk):
        part = ps[c0:c0 + chunk]
        seqs = [p["token_ids"] for p in part] + [p["counterfactual"]["token_ids"] for p in part]
        blocks = [p.get("attention_block", []) for p in part] + [p["counterfactual"].get("attention_block", p.get("attention_block", [])) for p in part]
        Tc = max(map(len, seqs))
        ids = torch.zeros(len(seqs), Tc, dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = torch.tensor(x)
        ids = ids.to(model.device)
        if model.model in HF:
            if any(blocks):
                dtype = next(model.m.parameters()).dtype
                mask = additive_mask([len(x) for x in seqs], Tc, blocks, dtype).to(model.device)
            else:
                mask = torch.tensor([[1] * len(x) + [0] * (Tc - len(x)) for x in seqs], device=model.device)
            h, head = model.m.model(input_ids=ids, attention_mask=mask).last_hidden_state, model.m.lm_head
        else:
            h, head = model.m.hidden(ids), lambda x: x @ model.m.wte.T
        n = len(part)
        rows_a = [(r, t) for r, p in enumerate(part) for t in p["target_positions"]]
        bi = torch.tensor([r for r, _ in rows_a] + [n + r for r, _ in rows_a], device=h.device)
        ti = torch.tensor([t for _, t in rows_a] * 2, device=h.device)
        lp = torch.log_softmax(head(h[bi, ti]).float(), -1)
        a, b = lp[:len(rows_a)], lp[len(rows_a):]
        kl = ((a.exp() * (a - b)).sum(-1) / math.log(2)).cpu().tolist()
        diff = (a.argmax(-1) != b.argmax(-1)).cpu().tolist()
        k = 0
        for p in part:
            m = len(p["target_positions"])
            changed.append(any(diff[k:k + m]))
            kls += kl[k:k + m]
            k += m
        del h, lp, a, b
    if model.device == "mps":
        torch.mps.empty_cache()
    return changed, kls


def quality(model: Model, beh: dict, max_tokens: int) -> dict:
    changed, kls = pair_stats(model, beh["prompts"], max_tokens)
    n = len(changed)
    kls_sorted = sorted(kls)
    return {"changed_fraction": round(sum(changed) / n, 4), "mean_kl_bits": round(sum(kls) / len(kls), 4),
            "median_kl_bits": round(kls_sorted[len(kls_sorted) // 2], 4), "prompts": n, "targets": len(kls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*HF, "vpd4l"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--root", default=str(OUT))
    ap.add_argument("--families", default="")
    a = ap.parse_args()
    root = Path(a.root)
    model = Model(a.model, a.device)
    rows = []
    for f in sorted((root / a.model).glob("*.json")):
        beh = json.loads(f.read_text())
        if a.families and beh["family"] not in a.families.split(","):
            continue
        q = quality(model, beh, a.max_tokens)
        beh["counterfactual_quality"] = q
        f.write_text(json.dumps(beh))
        rows.append({"model": a.model, "id": beh["id"], "family": beh["family"], "split": beh["split"], "keep": beh.get("keep", ""), **q})
        print(f"{beh['id']:40s} changed {q['changed_fraction']:.3f}  KL {q['mean_kl_bits']:.3f} bits (median {q['median_kl_bits']:.3f})", flush=True)
    path = root / "cf_quality.tsv"
    old = []
    if path.exists():
        with path.open() as fh:
            old = [r for r in csv.DictReader(fh, delimiter="\t") if not (r["model"] == a.model and r["id"] in {x["id"] for x in rows})]
    cols = ["model", "id", "family", "split", "keep", "prompts", "targets", "changed_fraction", "mean_kl_bits", "median_kl_bits"]
    with path.open("w") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in sorted(old + rows, key=lambda r: (r["model"], r["id"])):
            w.writerow(r)


if __name__ == "__main__":
    main()
