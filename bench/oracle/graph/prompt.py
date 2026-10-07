"""The graph oracle's input (#2951): a behavior file rendered as text, after a short `mech` reference
and example answers (a program, then its plain-English explanation). The oracle answers in the same form:
split_answer separates the program (scored by the checker) from the explanation (read by the reader). No weights or vectors: the behavior's description, a few of its prompts with
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
Explain how the model produces the behavior below. Answer with ONE Python program in a ```python block,
then the explanation, in plain English, after the block. The program may import only
`from mech import node, edges, L, PD, embed, logits, attend, tokens, shift`.
- node(*pieces) declares a node: pieces of the model's weights that compute with their actual inputs.
{pieces}  A node's pieces lie in one layer's attention or one layer's MLP.
- node(L[l].head[h], rule=attend(...)) gives heads an attention rule in place of their query and key
  weights (which are then not charged): attend(offset=k) (the position k back), attend(query=tokens,
  key=shift(tokens, 1)) (every earlier position whose previous token is the current token), or
  attend(first=True) (the first position). A ruled head reads only its value.
- writer >> reader declares an edge, listed in edges(...). The writer is a node or embed; the reader is
  node.query, node.key or node.value (attention), node.input or the node itself (all of its reads), or
  logits. Writers must come before readers.
- Anything you do not declare behaves as it would on the counterfactual prompt, so declare the pieces
  and connections that carry the information that decides the answer, and nothing more: write the
  smallest program that explains.
- Comments and docstrings are your working notes: they cost nothing and nobody else reads them.
- The explanation after the code says what each part does and how the parts connect, in a few plain
  sentences. A reader that never sees the code predicts the model under experiments from it alone.
- The score, in bits (lower is better), adds: the code's Python tokens; every weight number of the
  declared pieces ({prices}); the error of the program against the model under random experiments
  applied identically to both (prompt edits, weight edits, node value swaps, edge cuts); and the error
  of the reader's predictions from your explanation."""


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
    """(the model's sizes, the piece addresses the oracle may use: heads, and VPD subcomponents or
    transcoder features where the target has them; neuron lists are not part of its vocabulary)."""
    s = mech.shapes(model)
    groups = s["heads"] // s["kv_heads"]
    heads = f"{s['heads']} attention heads" + (f" ({s['kv_heads']} key-value groups of {groups})" if groups > 1 else "")
    sizes = (f"Model {model}: {s['layers']} layers (0..{s['layers'] - 1}); per layer {heads} and an MLP of {s['d_mlp']} "
             f"hidden units; vocabulary {s['vocab']} tokens.")
    pieces = ["  L[l].head[h] (head h of layer l: its query, key, value and output weights)"]
    if s["views"].get("vpd"):
        sites = ", ".join(f"{k} {v}" for k, v in s["views"]["vpd"][0].items())
        pieces.append(f"  PD.vpd[l].<site>[i, ...] (rank-one subcomponents of VPD's decomposition of a weight matrix; "
                      f"per layer {sites}), and PD.vpd[l].<site>.rest (what that matrix holds beyond its subcomponents)")
    if s["views"].get("transcoder"):
        pieces.append(f"  PD.tc[l][i, ...] (transcoder features replacing layer l's MLP; "
                      f"{s['views']['transcoder'][0]} per layer)")
    return sizes, ",\n".join(pieces) + ".\n"


def prices(model: str) -> str:
    """How many weight numbers the oracle's pieces of `model` use (each costs at most 1/2 log2 N bits)."""
    s = mech.shapes(model)
    d, hd = s["d_model"], s["head_dim"]
    head = (2 + 2 * s["kv_heads"] / s["heads"]) * d * hd  # q rows and o columns, plus its share of k and v
    out = f"a head uses about {head / 1e3:.0f}k"
    if s["views"].get("transcoder"):
        out += f", a transcoder feature {2 * d:,}"
    if s["views"].get("vpd"):
        out += f", a VPD subcomponent {2 * d:,} to {d + s['d_mlp']:,}"
    return out + "; each costs at most 1/2 log2 N bits, N the behavior's scored tokens"


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


def examples(behavior: dict, shots: int) -> list[tuple[str, dict, str, str]]:
    """Up to `shots` example answers (name, index entry, source, explanation) for `behavior`: train-split
    examples of other families only (examples/index.json; families sharing their first word count as one),
    so a prompt never shows a program for its own behavior family or a held-out one; only programs in the
    oracle's vocabulary (no neuron lists) that have an explanation (examples/<name>.explanation.txt); the
    target model's first, then by the index's "priority", then shortest."""
    index = json.loads((HERE / "examples/index.json").read_text())
    kin = (behavior.get("family") or "").split("_")[0]  # induction_random and induction_phrase are kin
    found = {}
    for n, e in index.items():
        note = HERE / "examples" / f"{n}.explanation.txt"
        if e["split"] != "train" or e["family"].split("_")[0] == kin or not note.exists():
            continue
        source = (HERE / "examples" / f"{n}.py").read_text()
        ir = mech.trace_inline(source, e["model"])
        if ir["valid"] and not any(p["view"] == "native" and p["kind"] == "mlp" for n_ in ir["nodes"] for p in n_["pieces"]):
            found[n] = (source, note.read_text().strip())
    names = sorted(found, key=lambda n: (index[n]["model"] != behavior["model"], index[n].get("priority", 9), len(found[n][0])))
    return [(n, index[n], *found[n]) for n in names[:shots]]


def render(behavior: dict, prompts: int = 4, shots: int = 1) -> str:
    model = behavior["model"]
    sizes, pieces = views(model)
    parts = [REFERENCE.format(pieces=pieces, prices=prices(model))]
    for name, entry, source, explanation in examples(behavior, shots):
        parts.append(f"Example answer ({entry['model']}):\n```python\n{source.strip()}\n```\n\n{explanation}")
    parts.append(f"{sizes}\n{behavior_text(behavior, prompts)}\n\nWrite the program, then the explanation.")
    return "\n\n".join(parts)


def split_answer(answer: str) -> tuple[str, str]:
    """(program, explanation) of an oracle's answer: the last fenced block that parses as Python, and the
    text after that block (a leading "Explanation:" label dropped). Without such a block: (answer, "")."""
    parts = answer.split("```")
    for k in range(len(parts) - 2, 0, -2):  # fenced blocks are the odd parts; take the last one that parses
        block = parts[k].split("\n", 1)[1] if "\n" in parts[k] else ""
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        explanation = "```".join(parts[k + 1 :]).strip()
        if explanation.lower().startswith("explanation:"):
            explanation = explanation[len("explanation:") :].strip()
        return block, explanation
    return answer, ""


def program_of(answer: str) -> str:
    """The program in an oracle's answer (split_answer's first part)."""
    return split_answer(answer)[0]


def explanation_of(answer: str) -> str:
    """The English explanation after the program (split_answer's second part)."""
    return split_answer(answer)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--shots", type=int, default=1)
    a = ap.parse_args()
    print(render(json.loads(a.behavior.read_text()), a.prompts, a.shots))


if __name__ == "__main__":
    main()
