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

TOKENIZERS = {
    "vpd4l": Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json",
    "qwen3-0.6b": Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots",
}

REFERENCE = """\
Write ONE Python file that explains how the model produces the behavior below. It may import only
`from mech import node, edges, L, PD, embed, logits`.
- node(*pieces) declares a node: pieces of the model's weights that compute with their actual inputs.
{pieces}  Indices may be several ints, slices or ranges.
- writer >> reader declares an edge, listed in edges(...). The writer is a node or embed; the reader is
  node.query, node.key or node.value (attention), node.input or the node itself (all of its reads), or
  logits. Writers must come before readers.
- Every piece you do not declare is replaced by its average over the behavior's prompts, and every
  edge you do not declare carries the writer's average write.
- The score, in bits (lower is better), adds: the code's Python tokens; the error of the program
  against the model under random experiments applied identically to both (prompt edits, weight edits,
  node value swaps, edge cuts); and the error of a reader that predicts the model from the program's
  comments and docstrings alone. Comments and docstrings cost nothing: say in English what each node
  computes and why the edges are there."""


@lru_cache(None)
def tokenizer(model: str):
    import tokenizers

    path = TOKENIZERS[model]
    if path.is_dir():
        path = next(path.glob("*/tokenizer.json"))
    return tokenizers.Tokenizer.from_file(str(path))


def views(model: str) -> tuple[str, str]:
    """(the model's sizes, the piece addresses its views offer)."""
    s = mech.shapes(model)
    groups = s["heads"] // s["kv_heads"]
    heads = f"{s['heads']} attention heads" + (f" ({s['kv_heads']} key-value groups of {groups})" if groups > 1 else "")
    sizes = (f"Model {model}: {s['layers']} layers (0..{s['layers'] - 1}); per layer {heads} and {s['d_mlp']} MLP "
             f"neurons; vocabulary {s['vocab']} tokens.")
    pieces = [f"  L[l].head[h] (head h of layer l: its query, key, value and output weights), "
              f"L[l].mlp[i, ...] (MLP neurons),"]
    if s["views"].get("vpd"):
        sites = ", ".join(f"{k} {v}" for k, v in s["views"]["vpd"][0].items())
        pieces.append(f"  PD.vpd[l].<site>[i, ...] (rank-one subcomponents of VPD's decomposition of a weight matrix; "
                      f"per layer {sites}),")
    if s["views"].get("library"):
        pieces.append("  PD.lib[l].<site>[i, ...] (parts of our decomposition),")
    if s["views"].get("transcoder"):
        pieces.append(f"  PD.tc[l][i, ...] (transcoder features replacing layer l's MLP; "
                      f"{s['views']['transcoder'][0]} per layer),")
    return sizes, "\n".join(pieces) + "\n"


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


def examples(behavior: dict, shots: int) -> list[tuple[str, dict, str]]:
    """Up to `shots` example programs (name, index entry, source) for `behavior`: train-split examples
    of other families only (examples/index.json), so a prompt never shows a program for its own
    behavior family or a held-out one; the target model's first."""
    index = json.loads((HERE / "examples/index.json").read_text())
    names = sorted((n for n, e in index.items() if e["split"] == "train" and e["family"] != behavior.get("family")),
                   key=lambda n: (index[n]["model"] != behavior["model"], n))
    return [(n, index[n], (HERE / "examples" / f"{n}.py").read_text()) for n in names[:shots]]


def render(behavior: dict, prompts: int = 4, shots: int = 1) -> str:
    model = behavior["model"]
    sizes, pieces = views(model)
    parts = [REFERENCE.format(pieces=pieces)]
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
