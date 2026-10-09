"""Format v4 explanations for the searches (#2951 graph oracle): groups of VPD subcomponents as source text and as the
checker's IR.

A unit is (layer, site, index): site a VPD site name (q_proj ... down_proj), index an int or "rest". A node is
{"name", "units", "at"}; an edge (writer, reader[, route]); labels {node: behavior variable}.

  source(nodes, edges, labels, functions)   the explanation's Python in the canonical form
  chain(units)             a set of units as nodes per block with the edges of `ir`
  ir(units, model, standin) the IR of one node naming `units`, reading the input and writing the output, built
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


def source(nodes: list[dict], edges: list[tuple], labels: dict | None = None, functions: str = "") -> str:
    """The explanation's text in edits.Answer's canonical form: `functions` (code the nodes' "at" names), then the nodes
    ({"name", "units", "at"}; subcomponents in model order), the edges and the labels."""
    return edits.Answer(tuple(functions.strip().split("\n")) if functions.strip() else (),
                        tuple(edits.Statement(-1, n["name"], tuple(token(u) for u in order(n["units"])), n.get("at", "all")) for n in nodes),
                        tuple(tuple(e) for e in edges), tuple((labels or {}).items())).source()


def block(u: tuple) -> tuple[int, int]:
    """A unit's block in execution order: (layer, 0 for attention or 1 for the MLP)."""
    return u[0], int(u[1] in ("c_fc", "down_proj"))


def wire(nodes: list[dict]) -> list[tuple]:
    """Every edge the nodes allow: the input into each node that reads the residual stream, a node into another when
    one of its residual writers comes before one of the other's readers, and each node that writes into the output."""
    readers = {n["name"]: [block(u) for u in n["units"] if u[1] in ("q_proj", "k_proj", "v_proj", "c_fc")] for n in nodes}
    writers = {n["name"]: [block(u) for u in n["units"] if u[1] in ("o_proj", "down_proj")] for n in nodes}
    edges = []
    for n in nodes:
        r = n["name"]
        if readers[r]:
            edges += [("input", r)] + [(w["name"], r) for w in nodes if w["name"] != r and writers[w["name"]]
                                       and min(writers[w["name"]]) < max(readers[r])]
    return edges + [(n["name"], "output") for n in nodes if writers[n["name"]]]


def chain(units: list[tuple], at: str = "all") -> tuple[list[dict], list[tuple]]:
    """A set of units as one node per block, each reading the input and every earlier block, the residual writers
    writing the output (the edges of e2e/explain.ir, as nodes and edges)."""
    blocks: dict[tuple, list[tuple]] = {}
    for u in units:
        blocks.setdefault(block(u), []).append(u)
    nodes = [{"name": f"{'mlp' if k[1] else 'attn'}{k[0]}", "units": blocks[k], "at": at} for k in sorted(blocks)]
    return nodes, wire(nodes)


def writes(units) -> bool:
    """Whether units write the residual stream (an o or down subcomponent)."""
    return any(u[1] in ("o_proj", "down_proj") for u in units)


def ir(units, model: str = "vpd4l", standin: str = "counterfactual") -> dict:
    """The IR of one node naming `units` that reads the input and writes the output (mech.build's edges: the
    embedding into each subcomponent that reads, each earlier block into each later one, each residual writer
    into the logits, and the embedding into the logits); everything else `standin` ("counterfactual": its values on
    the changed prompt, or "delete")."""
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
    return {"model": model, "decomposition": "vpd", "standin": standin, "nodes": [n.ir() for n in nodes], "edges": edges,
            "alignments": [], "groups": [], "python_tokens": 0, "token_types": 0, "source": "", "valid": True, "error": None}
