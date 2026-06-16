#!/usr/bin/env python3
"""Harvest residual-stream activations from a strong open LLM over a diverse text
corpus, save per-layer .npy banks + a manifest, and compute PCA / effective-rank
stats for the #1026 EV-vs-K parity program.

Self-contained: only needs torch + transformers + datasets + numpy.
"""
import os, sys, json, time, argparse
import numpy as np

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")         # comma idx; empty => auto early/mid/late
    ap.add_argument("--n_docs", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=300000)  # cap collected token-rows per layer
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.out, exist_ok=True)

    log("loading tokenizer/model", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True, attn_implementation="sdpa")
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    log("model loaded: n_layers", n_layers, "hidden", hidden,
        "devices", torch.cuda.device_count())

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = sorted(set([max(1, n_layers//6), n_layers//2, max(1, n_layers - n_layers//6)]))
    log("harvest layers (hidden_states idx)", layers)

    # ---- corpus ----
    log("loading corpus", args.dataset, args.dataset_config)
    from datasets import load_dataset
    candidates = [
        (args.dataset, args.dataset_config),
        ("Salesforce/wikitext", "wikitext-103-raw-v1"),
        ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        ("stas/openwebtext-10k", None),
        ("wikimedia/wikipedia", "20231101.en"),
    ]
    texts = []
    manifest_dataset = f"{args.dataset}/{args.dataset_config}"
    for name, cfg in candidates:
        try:
            if cfg:
                ds = load_dataset(name, cfg, split="train", streaming=True)
            else:
                ds = load_dataset(name, split="train", streaming=True)
            it = iter(ds)
            while len(texts) < args.n_docs:
                try:
                    row = next(it)
                except StopIteration:
                    break
                t = row.get("text", "") or row.get("content", "")
                if t and len(t.strip()) > 200:
                    texts.append(t.strip())
            log("corpus", name, cfg, "-> collected", len(texts), "docs")
            if len(texts) >= max(200, args.n_docs // 4):
                manifest_dataset = f"{name}/{cfg}" if cfg else name
                break
        except Exception as e:
            log("corpus", name, cfg, "FAILED:", repr(e)[:200])
            texts = []
            continue
    else:
        raise RuntimeError("no corpus loaded")
    log("collected", len(texts), "docs total")

    banks = {L: [] for L in layers}
    counts = {L: 0 for L in layers}
    tot_tokens = 0
    done = False
    for bs in range(0, len(texts), args.batch):
        if done:
            break
        chunk = texts[bs:bs+args.batch]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=args.seq_len, padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        mask = enc["attention_mask"].bool()
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states  # tuple len n_layers+1
        for L in layers:
            h = hs[L]                       # (B, T, H)
            sel = h[mask]                   # (n_real_tokens, H)
            sel = sel.float().cpu().numpy().astype(np.float32)
            if counts[L] < args.max_tokens:
                take = min(len(sel), args.max_tokens - counts[L])
                banks[L].append(sel[:take])
                counts[L] += take
        tot_tokens += int(mask.sum().item())
        if bs % (args.batch*20) == 0:
            log("batch", bs, "tokens~", counts[layers[0]])
        if all(counts[L] >= args.max_tokens for L in layers):
            done = True

    # ---- save banks ----
    manifest = {"model": args.model, "n_layers": n_layers, "hidden": hidden,
                "seq_len": args.seq_len, "dataset": manifest_dataset,
                "n_docs": len(texts), "layers": {}, "ts": time.time()}
    for L in layers:
        arr = np.concatenate(banks[L], axis=0)
        fn = os.path.join(args.out, f"resid_L{L}.npy")
        np.save(fn, arr)
        # ---- stats: PCA spectrum, effective rank ----
        # Eigendecompose the d x d covariance (d=3584) instead of n x d SVD:
        # far faster + numerically robust (avoids slow LAPACK gesdd path).
        Xc = arr - arr.mean(0, keepdims=True)
        m = min(len(Xc), 20000)
        idx = np.random.RandomState(0).choice(len(Xc), m, replace=False) if len(Xc) > m else np.arange(len(Xc))
        Xs = Xc[idx].astype(np.float64)
        C = (Xs.T @ Xs) / max(1, (m-1))
        var = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
        ev = var / var.sum()
        cum = np.cumsum(ev)
        # effective rank (entropy of normalized spectrum)
        p = var / var.sum()
        eff_rank = float(np.exp(-(p*np.log(p+1e-12)).sum()))
        # participation ratio
        pr = float((var.sum()**2) / (var**2).sum())
        def k_for(frac):
            return int(np.searchsorted(cum, frac) + 1)
        stat = {"file": os.path.basename(fn), "shape": list(arr.shape),
                "n_tokens": int(arr.shape[0]),
                "ev_top1": float(ev[0]), "ev_top10": float(cum[min(9,len(cum)-1)]),
                "ev_top50": float(cum[min(49,len(cum)-1)]),
                "k90": k_for(0.90), "k95": k_for(0.95), "k99": k_for(0.99),
                "eff_rank_entropy": eff_rank, "participation_ratio": pr,
                "ev_curve_first64": [float(x) for x in cum[:64]]}
        manifest["layers"][str(L)] = stat
        log(f"L{L}", "shape", arr.shape, "k90", stat["k90"], "k95", stat["k95"],
            "k99", stat["k99"], "eff_rank", round(eff_rank,1), "PR", round(pr,1),
            "ev_top1", round(stat["ev_top1"],4))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("DONE total_tokens_processed", tot_tokens, "manifest written", args.out)

if __name__ == "__main__":
    main()
