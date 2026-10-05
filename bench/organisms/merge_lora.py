"""Stage published model organisms under neutral IDs (#2951): merge each LoRA adapter into its base
weights, or take a full fine-tune's weights as they are, and write a plain Hugging Face directory that
carries nothing of the source: no adapter files, model card, trainer state or repository name.

  merge_lora.py --map MAP.json --base-repo ORG/NAME --base-revision REV --out DIR [--cache DIR] [--only ID ...]

MAP.json: {ID: {"repo", "revision", "kind": "adapter" | "full"}} (other keys ignored). For an adapter,
every adapted matrix becomes W + (lora_alpha / r) B A, computed in float32 and stored in bfloat16 (the
base checkpoint's dtype); rsLoRA, DoRA and extra trained modules are refused, not approximated. A full
fine-tune must have the base's tensor names and shapes. DIR/ID gets model.safetensors, the base's
config.json (without its source path) and generation_config.json, and the base's tokenizer files.
Downloads go to --cache and each organism's are deleted once it is written. Prints only IDs and times.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def base_state(base_dir):
    state = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        state.update(load_file(f))
    return state


def merged_adapter(base, repo, revision, cache):
    cfg = json.load(open(hf_hub_download(repo, "adapter_config.json", revision=revision, cache_dir=cache)))
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("modules_to_save"):
        raise SystemExit("adapter uses rsLoRA, DoRA or extra trained modules; not handled")
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(hf_hub_download(repo, "adapter_model.safetensors", revision=revision, cache_dir=cache))
    out = dict(base)
    for key, a in lora.items():
        if ".lora_A." not in key:
            if ".lora_B." in key:
                continue
            raise SystemExit("adapter holds a tensor that is not a LoRA factor")
        name = key.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
        b = lora[key.replace(".lora_A.", ".lora_B.")]
        w = base[name]
        out[name] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
    return out


def full_weights(base, repo, revision, cache):
    d = snapshot_download(repo, revision=revision, cache_dir=cache, allow_patterns=["*.safetensors"])
    state = base_state(d)
    if "lm_head.weight" in state and "lm_head.weight" not in base:
        if torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            del state["lm_head.weight"]
    if set(state) != set(base) or any(state[k].shape != base[k].shape for k in base):
        raise SystemExit("full fine-tune does not match the base's tensors")
    return {k: v.to(base[k].dtype) for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--base-repo", required=True)
    ap.add_argument("--base-revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    entries = json.load(open(args.map))
    base_dir = snapshot_download(args.base_repo, revision=args.base_revision)
    base = base_state(base_dir)
    config = json.load(open(Path(base_dir) / "config.json"))
    config.pop("_name_or_path", None)
    for oid, e in entries.items():
        if args.only and oid not in args.only:
            continue
        started = time.time()
        cache = Path(args.cache) / oid
        state = (merged_adapter if e["kind"] == "adapter" else full_weights)(base, e["repo"], e["revision"], str(cache))
        dst = Path(args.out) / oid
        dst.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in state.items()}, str(dst / "model.safetensors"), metadata={"format": "pt"})
        json.dump(config, open(dst / "config.json", "w"), indent=2)
        for f in ("generation_config.json",) + TOKENIZER_FILES:
            if (Path(base_dir) / f).exists():
                shutil.copyfile(Path(base_dir) / f, dst / f)
        shutil.rmtree(cache, ignore_errors=True)
        print(f"{oid} written in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
