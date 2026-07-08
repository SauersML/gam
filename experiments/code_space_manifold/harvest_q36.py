"""Robust weekday-battery harvest for Qwen3.6-35B-A3B (MoE, multimodal class).
70 prompts (10 templates x 7 weekdays), last-token residual at requested layers.
CPU. Handles the ForConditionalGeneration class (routes text through .model)."""
import argparse, json, time
import numpy as np

WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
TEMPLATES = ["Today is {w}","The day after tomorrow is {w}","My favorite day of the week is {w}",
    "The meeting is scheduled for {w}","Yesterday was {w}","We will travel on {w}",
    "The store is closed on {w}","Her birthday falls on {w}","The exam takes place on {w}",
    "It always rains on {w}"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    prompts, labels, tids = [], [], []
    for ti, t in enumerate(TEMPLATES):
        for wi, w in enumerate(WEEKDAYS):
            prompts.append(t.format(w=w)); labels.append(wi); tids.append(ti)
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), trust_remote_code=True,
        low_cpu_mem_usage=True).eval()
    print(f"[harvest] model loaded ({time.time()-t0:.0f}s) type={type(model).__name__}", flush=True)
    per = {l: [] for l in args.layers}
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states
            for l in args.layers:
                per[l].append(hs[l][0, -1, :].float().cpu().numpy())
            if (i+1) % 10 == 0:
                print(f"[harvest] {i+1}/70 ({time.time()-t0:.0f}s)", flush=True)
    payload = {f"acts_L{l}": np.asarray(v, np.float64) for l, v in per.items()}
    payload["labels"] = np.asarray(labels); payload["template_ids"] = np.asarray(tids)
    np.savez(args.out, **payload)
    json.dump(prompts, open(args.out + ".prompts.json", "w"), indent=1)
    print(f"[harvest] -> {args.out}  layers={args.layers}  n_layers_hs={len(hs)}", flush=True)

if __name__ == "__main__":
    main()
