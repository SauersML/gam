"""Which transcoder features of Qwen3-0.6B's MLPs carry a behavior's answer (for PD.tc programs): at
each given layer, the ReLU transcoder (~/mpd-data/transcoders/qwen3-0.6b-lowl0, a_i = relu(W_enc[i] . x
+ b_enc[i]) on the MLP's input x, writing a_i W_dec[i]) is run on the clean and the counterfactual
prompts; the CANDIDATES features whose write changes most between them (sum over positions of
|a_i clean - a_i counterfactual| x |W_dec[i]|) are each patched alone: the counterfactual run's MLP
output at that layer gets (a_i clean - a_i counterfactual) W_dec[i] added at every position, and the
recovery is KL(M_clean || M_cf) - KL(M_clean || M_patched) in bits per target token.

  HF_HUB_OFFLINE=1 MPD_MEM_GIB=8 mem-lease 8 ~/mpd-data/venv/bin/python qwen_tc_patch.py BEHAVIOR.json LAYERS OUT.json
  (LAYERS comma-separated, e.g. 16,18,19)
"""

import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

TRANSCODERS = Path.home() / "mpd-data/transcoders/qwen3-0.6b-lowl0"
CANDIDATES = 32


@torch.no_grad()
def main():
    behavior, layers, out = Path(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], Path(sys.argv[3])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(dev).eval()
    beh = json.loads(behavior.read_text())
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
    state = {"layer": None, "record": None, "add": None}
    seen = {}

    def hook(layer):
        def fn(module, args, output):
            if state["record"] is not None and layer == state["layer"]:
                seen[state["record"]] = (args[0].clone(), output.clone())
            if state["add"] is not None and layer == state["layer"]:
                return output + state["add"]
        return fn

    for l, block in enumerate(model.model.layers):
        block.mlp.register_forward_hook(hook(l))

    def log_probs(ids):
        hidden = model.model(ids).last_hidden_state[rows, cols]
        return torch.log_softmax(model.lm_head(hidden).float(), -1)

    lp_clean = log_probs(clean)
    p_clean = lp_clean.exp()

    def kl(lp):
        return ((p_clean * (lp_clean - lp)).sum(-1).mean() / math.log(2)).item()

    base = kl(log_probs(counterfactual))
    result = {"behavior": beh["id"], "base_bits": base, "layers": {}}
    for layer in layers:
        path = TRANSCODERS / f"layer_{layer}.safetensors"
        if not path.exists():  # a pod: fetch just this layer from the Hugging Face repo
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download("mwhanna/qwen3-0.6b-transcoders-lowl0", f"layer_{layer}.safetensors"))
        tc = load_file(str(path))
        enc, b_enc, dec = (tc[k].to(dev, torch.float32) for k in ("W_enc", "b_enc", "W_dec"))
        state.update(layer=layer, add=None)
        for key, ids in (("clean", clean), ("cf", counterfactual)):
            state["record"] = key
            model.model(ids)
        state["record"] = None
        (xc, _), (xf, _) = seen["clean"], seen["cf"]
        ac, af = torch.relu(xc @ enc.T + b_enc), torch.relu(xf @ enc.T + b_enc)  # [B, T, F]
        delta = ac - af
        size = delta.abs().sum((0, 1)) * dec.norm(dim=1)
        candidates = torch.argsort(size, descending=True)[:CANDIDATES].tolist()
        features = []
        for i in candidates:
            state["add"] = delta[..., i : i + 1] * dec[i]
            features.append({"feature": i, "recovery_bits": base - kl(log_probs(counterfactual)),
                             "write_change": size[i].item()})
        state["add"] = None
        features.sort(key=lambda f: -f["recovery_bits"])
        result["layers"][str(layer)] = features
        print(f"layer {layer}:", [(f["feature"], round(f["recovery_bits"], 3)) for f in features[:8]], flush=True)
        out.write_text(json.dumps(result))
        del enc, b_enc, dec, ac, af, delta


if __name__ == "__main__":
    main()
