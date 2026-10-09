"""Format v3 explanations for the searches (#2951 graph oracle): groups of VPD subcomponents as source text and as the
checker's IR.

A unit is (layer, site, index): site a VPD site name (q_proj ... down_proj), index an int or "rest". A group is
{"name", "units", "reads", "label", "writes"}: reads lists "input" and group names, label names the behavior variable
the group carries (or None), writes is "output" or None.

  source(groups)           the explanation's Python: its `groups` statement
  ir(units)                the IR of one group naming `units`, reading the input and writing the output, built
                           without mech's validation (a search measures sets that may hold no residual writer)
  units_of(text)           the units of every part token in a text
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import edits  # noqa: E402
import mech  # noqa: E402

CODE = {site: code for code, site in mech.SITES.items()}


def token(unit) -> str:
    layer, site, index = unit
    return f"<p:{layer}.{CODE[site]}.{index}>"


def units_of(text: str) -> list[tuple]:
    out = []
    for m in mech.PART.finditer(text):
        u = (int(m[1]), mech.SITES[m[2]], m[3] if m[3] == "rest" else int(m[3]))
        if u not in out:
            out.append(u)
    return out


def order(units) -> list[tuple]:
    sites = list(mech.SITES.values())
    return sorted(units, key=lambda u: (u[0], sites.index(u[1]), -1 if u[2] == "rest" else u[2]))


def source(groups: list[dict]) -> str:
    """The explanation's text: its `groups` statement, one group per line, subcomponents in model order."""
    return edits.Answer((), tuple(edits.Statement(-1, g["name"], tuple(token(u) for u in order(g["units"])), tuple(g.get("reads", ["input"])),
                                                  g.get("label"), g.get("writes")) for g in groups), ()).source()


def ir(units, model: str = "vpd4l") -> dict:
    """The IR of one group naming `units` that reads the input and writes the output (mech.build's edges: the
    embedding into each subcomponent that reads, each earlier block into each later one, each residual writer
    into the logits, and the embedding into the logits)."""
    layers = mech.shapes(model)["layers"]
    blocks: dict[tuple, dict] = {}
    for layer, site, index in units:
        block = "mlp" if site in ("c_fc", "down_proj") else "attn"
        blocks.setdefault((layer, block), {}).setdefault(site, set()).add(index)
    nodes = [mech.Node(f"g.{l}.{b}", l, b, blocks[(l, b)]) for l, b in sorted(blocks)]
    embed, logits = [("resid", -1)], [("resid", 2 * layers, ("input",))]
    edges = []
    for k, n in enumerate(nodes):
        for name, writes in [("embed", embed)] + [(m.id, m.writes()) for m in nodes[:k]]:
            if mech._connects(writes, n.reads(), "input"):
                edges.append({"from": name, "to": n.id, "route": "input"})
    for name, writes in [(n.id, n.writes()) for n in nodes] + [("embed", embed)]:
        if mech._connects(writes, logits, "input"):
            edges.append({"from": name, "to": "logits", "route": "input"})
    return {"model": model, "decomposition": "vpd", "nodes": [n.ir() for n in nodes], "edges": edges, "alignments": [],
            "groups": [], "python_tokens": 0, "token_types": 0, "source": "", "valid": True, "error": None}
