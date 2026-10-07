"""The graph oracle's input (#2951): a behavior file rendered as text, after a short `mech` reference
and example programs. No weights or vectors: the behavior's description, a few of its prompts with
the model M's top next tokens and probabilities (the behavior file's `model_top`, measured when the
behavior was built), and the views of M with their sizes.

  prompt.py BEHAVIOR.json [--prompts 4] [--shots 1]      prints the prompt
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

VPD4L_TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"

REFERENCE = """\
Write ONE Python file that explains how the model produces the behavior below. It may import only
`from mech import node, edges, L, PD, embed, logits`.
- node(*pieces) declares a node: pieces of the model's weights that compute with their actual inputs.
{pieces}  Indices may be several ints, slices or ranges; a site without indices (L[3].mlp) is all of
  its units. A node's pieces lie in one layer's attention or one layer's MLP.
- writer >> reader declares an edge, listed in edges(...). The writer is a node or embed; the reader is
  node.query, node.key or node.value (attention), node.input or the node itself (all of its reads), or
  logits. Writers must come before readers.
- Anything you do not declare behaves as it would on the counterfactual prompt, so declare the pieces
  and connections that carry the information that decides the answer.
- The score, in bits (lower is better), adds: the code's Python tokens; every weight number of the
  declared pieces ({prices}); the error of the program against the model under random experiments
  applied identically to both (prompt edits, weight edits, node value swaps, edge cuts); and the error
  of a reader that predicts the model from the program's comments and docstrings alone. Comments and docstrings cost nothing: say in English what each node
  computes and why the edges are there."""


@lru_cache(None)
def tokenizer(model: str):
    """The target's tokenizer: vpd4l's file, or Qwen3's tokenizer.json (every size shares it) through the
    Hugging Face cache (HF_HOME / HF_HUB_CACHE honored; downloaded when missing)."""
    import tokenizers

    if model == "vpd4l":
        return tokenizers.Tokenizer.from_file(str(VPD4L_TOKENIZER))
    from huggingface_hub import hf_hub_download

    return tokenizers.Tokenizer.from_file(hf_hub_download(mech.QWEN3[model], "tokenizer.json"))


def views(model: str) -> tuple[str, str]:
    """(the model's sizes, the piece addresses its views offer)."""
    s = mech.shapes(model)
    groups = s["heads"] // s["kv_heads"]
    heads = f"{s['heads']} attention heads" + (f" ({s['kv_heads']} key-value groups of {groups})" if groups > 1 else "")
    sizes = (f"Model {model}: {s['layers']} layers (0..{s['layers'] - 1}); per layer {heads} and {s['d_mlp']} MLP "
             f"neurons; vocabulary {s['vocab']} tokens.")
    pieces = ["  L[l].head[h] (head h of layer l: its query, key, value and output weights), "
              "L[l].mlp[i, ...] (MLP neurons)"]
    if s["views"].get("vpd"):
        sites = ", ".join(f"{k} {v}" for k, v in s["views"]["vpd"][0].items())
        pieces.append(f"  PD.vpd[l].<site>[i, ...] (rank-one subcomponents of VPD's decomposition of a weight matrix; "
                      f"per layer {sites})")
    if s["views"].get("library"):
        n = s["views"]["library"]["parts"][0]
        pieces.append(f"  PD.lib[l].attn[i, ...] and PD.lib[l].mlp[i, ...] (parts of our decomposition of layer l's "
                      f"attention or MLP, which may overlap; layer 0 has {n['attn']} and {n['mlp']})")
    if s["views"].get("transcoder"):
        pieces.append(f"  PD.tc[l][i, ...] (transcoder features replacing layer l's MLP; "
                      f"{s['views']['transcoder'][0]} per layer)")
    return sizes, ",\n".join(pieces) + ".\n"


def prices(model: str) -> str:
    """How many weight numbers the common pieces of `model` use (each costs 1/2 log2 N bits)."""
    s = mech.shapes(model)
    d, hd, gated = s["d_model"], s["head_dim"], model.startswith("qwen")
    head = (2 + 2 * s["kv_heads"] / s["heads"]) * d * hd  # q rows and o columns, plus its share of k and v
    neuron = (3 if gated else 2) * d
    out = f"a head uses about {head / 1e3:.0f}k, an MLP neuron {neuron:,}"
    if s["views"].get("transcoder"):
        out += f", a transcoder feature {2 * d:,}"
    if s["views"].get("vpd"):
        out += f", a VPD subcomponent {2 * d:,} to {d + s['d_mlp']:,}"
    return out + "; each costs 1/2 log2 N bits, N the behavior's scored tokens"


def show(model: str, ids: list[int], t: int, top) -> str:
    context = tokenizer(model).decode(ids[: t + 1])
    return f"{context!r} -> " + ", ".join(f"{tok!r} {p:.2f}" for tok, p in top)


def behavior_text(behavior: dict, prompts: int) -> str:
    model = behavior["model"]
    lines = [f"Behavior {behavior['id']}: {behavior['description']}",
             f"Examples (the model's top next tokens with probabilities):"]
    for p in behavior["prompts"][:prompts]:
        for k, t in enumerate(p["target_positions"]):
            lines.append("  " + show(model, p["token_ids"], t, p["model_top"][k]))
            cf = p.get("counterfactual")
            if cf and cf.get("model_top"):
                lines.append("    edited: " + show(model, cf["token_ids"], t, cf["model_top"][k]))
    return "\n".join(lines)


SHOT_CHARS = 6000  # few-shot programs longer than this (long neuron lists) are left out of prompts


def examples(behavior: dict, shots: int) -> list[tuple[str, dict, str]]:
    """Up to `shots` example programs (name, index entry, source) for `behavior`: train-split examples
    of other families only (examples/index.json; families sharing their first word count as one), so a
    prompt never shows a program for its own behavior family or a held-out one; the target model's first, then by the index's "priority" (the
    hand-written ones first), then shortest."""
    index = json.loads((HERE / "examples/index.json").read_text())
    text = {n: (HERE / "examples" / f"{n}.py").read_text() for n in index}
    kin = (behavior.get("family") or "").split("_")[0]  # induction_random and induction_phrase are kin
    names = sorted((n for n, e in index.items() if e["split"] == "train" and e["family"].split("_")[0] != kin
                    and len(text[n]) <= SHOT_CHARS),
                   key=lambda n: (index[n]["model"] != behavior["model"], index[n].get("priority", 9), len(text[n])))
    return [(n, index[n], text[n]) for n in names[:shots]]


def render(behavior: dict, prompts: int = 4, shots: int = 1) -> str:
    model = behavior["model"]
    sizes, pieces = views(model)
    parts = [REFERENCE.format(pieces=pieces, prices=prices(model))]
    for name, entry, source in examples(behavior, shots):
        parts.append(f"Example program ({entry['model']}; its docstring states the behavior):\n"
                     f"```python\n{source.strip()}\n```")
    parts.append(f"{sizes}\n{behavior_text(behavior, prompts)}\n\nWrite the program.")
    return "\n\n".join(parts)


def program_of(answer: str) -> str:
    """The program in an oracle's answer: the last ```python block that parses, else the answer."""
    blocks = [b.split("\n", 1)[1] if "\n" in b else "" for b in answer.split("```")[1::2]]
    for b in reversed(blocks):
        try:
            ast.parse(b)
            return b
        except SyntaxError:
            continue
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--shots", type=int, default=1)
    a = ap.parse_args()
    print(render(json.loads(a.behavior.read_text()), a.prompts, a.shots))


if __name__ == "__main__":
    main()
