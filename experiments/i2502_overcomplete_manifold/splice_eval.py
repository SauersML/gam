"""#2502 splice benchmark: ΔCE when replacing the L16 residual stream with each
method's reconstruction, on FRESH held-out text (wikitext validation).

Arms: base (no patch), chart (PCA-512 projection only), manifold (gam K=32k
circle dictionary), gam_linear (matched TopK linear lane), torch_topk (standard
Adam TopK SAE), mean_ablate (constant c0 — the floor).

Loss-recovered metric per SAE arm: (CE_ablate - CE_arm) / (CE_ablate - CE_base).
Two-pass design: pass 1 captures activations; reconstructions are computed
offline (Rust/torch); pass 2 patches them back in. All arms share the identical
token batch (common random numbers).
"""
import argparse, json, os, pickle, time
import numpy as np


def resolve_decoder_layers(model):
    import torch
    best = None
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (best is None or len(mod) > len(best)):
            best = mod
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16"))
    ap.add_argument("--fits", default=os.path.expanduser("~/i2502/fits"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502/fits/splice.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda:0")
    model.eval()
    layers = resolve_decoder_layers(model)
    layer = layers[args.layer]

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="validation", streaming=True)
    buf, seqs = [], []
    for ex in ds:
        if not ex["text"].strip():
            continue
        buf.extend(tok.encode(ex["text"], add_special_tokens=False))
        while len(buf) >= args.seq_len:
            seqs.append(buf[: args.seq_len])
            buf = buf[args.seq_len:]
        if len(seqs) >= args.n_seqs:
            break
    ids = torch.tensor(seqs[: args.n_seqs], dtype=torch.long, device="cuda:0")
    B, T = ids.shape
    print(f"[splice] {B} seqs x {T}", flush=True)

    lift = np.load(f"{args.prep}/lift.npy")          # (P, D)
    c0 = np.load(f"{args.prep}/c0.npy")              # (D,)
    c1 = np.load(f"{args.prep}/c1.npy")

    # ---- pass 1: capture h at layer output, batched ----
    captured = []

    def cap_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float().cpu())

    def ce_of_logits(logits):
        import torch.nn.functional as F
        return float(F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1)).item())

    handle = layer.register_forward_hook(cap_hook)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False)
    handle.remove()
    ce_base = ce_of_logits(out.logits)
    H = torch.cat(captured, 0).numpy().astype(np.float64)   # (B, T, D)
    del captured, out
    D = H.shape[-1]
    flat = H.reshape(-1, D)
    posflags = np.zeros((B, T), dtype=bool)
    posflags[:, 0] = True
    cvec = np.where(posflags.reshape(-1, 1), c1[None, :], c0[None, :])
    Z = np.ascontiguousarray((flat - cvec) @ lift.T)        # chart coords (N, P)

    def lift_back(Zr):
        return Zr @ lift + cvec

    arms = {}
    arms["chart"] = lift_back(Z)
    arms["mean_ablate"] = np.broadcast_to(cvec, flat.shape).copy()

    # gam manifold + linear reconstructions (frozen-decoder OOS solve, chunked)
    import gamfit
    for name, pklname in (("manifold", f"manifold_k{args.k}.pkl"),
                          ("gam_linear", f"linear_k{args.k}.pkl")):
        path = os.path.join(args.fits, pklname)
        if not os.path.exists(path):
            print(f"[splice] skip {name}: {path} missing", flush=True)
            continue
        with open(path, "rb") as f:
            # Dispatch on the payload's own schema tag: the flagship fits here
            # are overcomplete (K > P) and carry the support tag, which the
            # /v6-pinned ManifoldSAE parser rejects outright (#2567).
            m = gamfit.sae.model_from_dict(pickle.load(f))
        t0 = time.time()
        R = np.empty_like(Z)
        step = 4096
        for i in range(0, len(Z), step):
            R[i:i + step] = np.asarray(m.reconstruct(np.ascontiguousarray(Z[i:i + step])))
        print(f"[splice] {name} recon {len(Z)} rows in {time.time()-t0:.0f}s", flush=True)
        arms[name] = lift_back(R)
        del m

    # torch TopK reconstruction
    npz = os.path.join(args.fits, f"torch_topk_k{args.k}.npz")
    if os.path.exists(npz):
        w = np.load(npz)
        pre = (Z - w["b_pre"]) @ w["W_enc"] + w["b_enc"]
        kk = 8
        idx = np.argpartition(pre, -kk, axis=1)[:, -kk:]
        vals = np.take_along_axis(pre, idx, 1)
        vals = np.maximum(vals, 0.0)
        R = np.zeros_like(Z)
        Zr = np.zeros_like(pre)
        np.put_along_axis(Zr, idx, vals, 1)
        R = Zr @ w["W_dec"] + w["b_pre"]
        arms["torch_topk"] = lift_back(R)
        del pre, Zr

    # ---- pass 2: patched forwards ----
    results = {"base": ce_base, "meta": dict(model=args.model, layer=args.layer,
               n_seqs=B, seq_len=T, k=args.k)}
    for name, recon in arms.items():
        rt = torch.from_numpy(recon.reshape(B, T, D).astype(np.float32))

        def patch_hook(_m, _i, output, rt=rt):
            h = output[0] if isinstance(output, tuple) else output
            e = rt.to(device=h.device, dtype=h.dtype)
            if isinstance(output, tuple):
                return (e,) + output[1:]
            return e

        handle = layer.register_forward_hook(patch_hook)
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False)
        handle.remove()
        results[name] = ce_of_logits(out.logits)
        print(f"[splice] {name}: CE={results[name]:.4f} (base {ce_base:.4f})", flush=True)
        del out

    if "mean_ablate" in results:
        floor, base = results["mean_ablate"], results["base"]
        for name in ("chart", "manifold", "gam_linear", "torch_topk"):
            if name in results and floor > base:
                results[f"loss_recovered_{name}"] = (floor - results[name]) / (floor - base)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("[splice] DONE", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
