"""An oracle adapter as a plain HF checkpoint (#2951 graph oracle): the base with the adapter merged into its weights,
and the part tokens (added to the tokenizer) with their rows from the adapter's part-token maps in the extended
vocabulary, the output layer untied (part_vocab.materialize). An RL stack without our part-token modules (prime-rl)
starts from it, a new LoRA on top and the part rows fixed.

  export_hf.py ADAPTER REGISTRY OUT [--base Qwen/Qwen3-4B]     (REGISTRY: part_tokens.py build's output)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter")
    ap.add_argument("registry")
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import part_vocab
    import train

    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    parts = train.load_parts(a.registry, a.adapter, base, len(tok), torch.device("cpu"))
    first = part_vocab.install(base, tok, parts)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    out = part_vocab.materialize(merged, tok, parts, first, a.base, a.out)
    print(f"{out}: {a.base} with {a.adapter} merged, {len(parts.tokens())} part tokens from id {first}")


if __name__ == "__main__":
    main()
