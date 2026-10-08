"""Reference programs for the graph oracle's end-to-end checks (#2951): mech sources in native units (traced
and scored with no decomposition attached, decomposition "native") that every score must order the same
way on one behavior.

  empty(model)                no node: every piece a stand-in
  hand(model)                 the hand-written example program of the model (examples/<model>_*.py)
  random_heads(model, k, s)   k heads drawn with seed s from the heads the hand program does not use,
                              wired like an induction circuit (embed into every read, each head into
                              the next, the last into the logits)
  full(model)                 every piece declared: one node per layer's heads and one per layer's MLP,
                              every causal edge declared (route input), so the program is M itself
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPH = HERE.parent
sys.path.insert(0, str(GRAPH))

import mech  # noqa: E402

HEADER = "from mech import node, edges, L, embed, logits\n"


def empty(model: str) -> str:
    return '"""The empty program: every piece of the model is replaced by its average over the prompts."""\n' + HEADER


def hand_for(model: str, behavior: str | None = None, family: str | None = None) -> str | None:
    """The hand-written example program (examples/index.json) for a behavior: the one written for that
    behavior id, else one of the model's examples whose family shares the behavior family's first word
    (induction_random ~ induction_text), native pieces first; None when there is none."""
    index = json.loads((GRAPH / "examples/index.json").read_text())
    mine = {k: v for k, v in index.items() if v["model"] == model and (GRAPH / f"examples/{k}.py").exists()}
    exact = [k for k, v in mine.items() if behavior and v.get("behavior") == behavior]
    near = sorted((k for k, v in mine.items() if family and v["family"].split("_")[0] == family.split("_")[0]),
                  key=lambda k: ("native" not in k and "heads" not in k, k))
    for k in exact + near:
        return (GRAPH / f"examples/{k}.py").read_text()
    return None


def hand(model: str, behavior: str | None = None, family: str = "induction") -> str:
    found = hand_for(model, behavior, family)
    if found is None:
        raise FileNotFoundError(f"no hand-written example program for {model} {behavior or family}")
    return found


def hand_heads(model: str) -> set[tuple[int, int]]:
    """The (layer, head) pairs the hand program declares (none without a native hand program)."""
    source = hand_for(model, family="induction")
    ir = mech.trace_inline(source, model, decomposition="native") if source else {"valid": False}
    if not ir["valid"]:
        return set()
    return {(p["layer"], i) for n in ir["nodes"] for p in n["pieces"] if p["kind"] == "head"
            for i in ([p["index"]] if isinstance(p["index"], int) else p["index"])}


def random_heads(model: str, k: int = 3, seed: int = 0) -> str:
    s = mech.shapes(model)
    used = hand_heads(model)
    pool = [(l, h) for l in range(s["layers"]) for h in range(s["heads"]) if (l, h) not in used]
    picked = sorted(random.Random(seed).sample(pool, k))
    names = [f"h{i}" for i in range(k)]
    lines = ['"""Heads drawn at random from those the hand-written program does not use."""', HEADER.rstrip()]
    lines += [f"{n} = node(L[{l}].head[{h}])" for n, (l, h) in zip(names, picked)]
    wires = [f"    embed >> {n}.{r}," for n in names for r in ("query", "key", "value")]
    wires += [f"    {a} >> {b}.key," for (a, (la, _)), (b, (lb, _)) in zip(zip(names, picked), zip(names[1:], picked[1:])) if la < lb]
    wires.append(f"    {names[-1]} >> logits,")
    return "\n".join(lines + ["edges("] + wires + [")"]) + "\n"


def full(model: str) -> str:
    s = mech.shapes(model)
    lines = ['"""Every piece of the model declared and every causal connection listed: the model itself."""',
             HEADER.rstrip()]
    writers = ["embed"]
    wires = []
    for l in range(s["layers"]):
        for name, piece in ((f"attn{l}", f"L[{l}].head[0:{s['heads']}]"), (f"mlp{l}", f"L[{l}].mlp[0:{s['d_mlp']}]")):
            lines.append(f"{name} = node({piece})")
            wires += [f"    {w} >> {name}," for w in writers]
            writers.append(name)
    wires += [f"    {w} >> logits," for w in writers]
    return "\n".join(lines + ["edges("] + wires + [")"]) + "\n"


def references(model: str, seed: int = 0) -> dict[str, str]:
    """empty, random and full, and the hand program when the model has a native one."""
    found = hand_for(model, family="induction")
    native = found is not None and mech.trace_inline(found, model, decomposition="native")["valid"]
    return {**({"hand": found} if native else {}), "empty": empty(model), "random": random_heads(model, 3, seed), "full": full(model)}
