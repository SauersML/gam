"""The `mech` library of the graph oracle (#2951): the only module an oracle program may import.

A program declares nodes bound to pieces of the target model M's weights and the edges between them:

    from mech import node, edges, L, PD, embed, logits
    prev  = node(L[1].head[1])                 # a native head: its rows of q, k, v and columns of o
    match = node(L[2].head[4])
    boost = node(L[3].mlp[118, 2051])          # native MLP neurons: gate/up rows, down columns
    part  = node(PD.vpd[2].c_fc[1534], PD.vpd[2].down_proj[77])   # VPD subcomponents (vpd4l)
    edges(
        prev >> match.key,                     # routes: .query .key .value (heads), .input (any read)
        embed >> match.query,
        match >> logits,
    )

Addresses (layer l, indices i, j, ... from 0):
  L[l].head[h, ...]          native attention heads (Qwen3: query heads; a head's k/v rows are those of
                             its key-value group, shared with the other heads of the group)
  L[l].mlp[i, ...]           native MLP neurons (L[l].attn: all heads of layer l)
  PD.vpd[l].<site>[i, ...]   VPD subcomponents U_i V_i^T, site in q_proj k_proj v_proj o_proj c_fc down_proj
  PD.lib[l].attn[i, ...]     our library's parts of layer l's attention (.mlp[i]: of its MLP); parts may
                             overlap, so a part may sit in several nodes (vpd4l: decomp's start, arm
                             LIBRARY_ARM; part i = the i-th component of that block in the file)
  PD.tc[l][i, ...]           transcoder features of layer l's MLP (Qwen3-0.6B)
Indices may be ints, slices or ranges; a site without indices (L[3].mlp) is all of its units.
`node(*pieces)` makes one node (its pieces in one layer's attention or one layer's MLP); `writer >> reader` declares an
edge (writer: a node or embed; reader: a route handle, a node = all of its reads, or logits) and
`edges(...)` lists them. node(L[1].head[1], rule=attend(offset=1)) replaces a head's query and key by
an attention rule (attend(offset=k), attend(query=tokens, key=shift(tokens, 1)), attend(first=True)).
Undeclared pieces and edges carry the model's values on the prompt's counterfactual. Comments and docstrings are free text: the English of the explanation.

trace(source, model) checks a program and runs it in a sandboxed child process, returning the IR the
checker reads (design.txt section 5); code_length counts its Python tokens; english extracts its
comments and docstrings; shapes(model) is the registry of model sizes (shapes.json, built from the
weight and config files by `mech.py shapes --write`).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import inspect
import io
import json
import os
import signal
import subprocess
import sys
import threading
import tokenize
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES_FILE = HERE / "shapes.json"
MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-8b", "vpd4l")
QWEN3 = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3-8b": "Qwen/Qwen3-8B"}
VIEWS = ("native", "vpd", "library", "transcoder")
SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")
ROUTES = ("query", "key", "value", "input")
EXPORTS = ("node", "edges", "L", "PD", "embed", "logits", "attend", "tokens", "shift")
ATTRIBUTES = ("head", "attn", "mlp", "vpd", "lib", "tc", "query", "key", "value", "input") + SITES
LIBRARY_ARM = "grouped_own"  # the arm of decomp's start that PD.lib addresses
DEFAULT_STANDIN = "counterfactual"


class MechError(Exception):
    """An invalid program: a bad address, edge or construct."""


# ---------------------------------------------------------------------------------------------------
# Model shapes

_SHAPES: dict | None = None


def shapes(model: str) -> dict:
    """The registry entry of `model`: layers, heads, kv_heads, head_dim, d_model, d_mlp, vocab and,
    per view, its size per layer (vpd/library: subcomponents per site; transcoder: features)."""
    global _SHAPES
    if _SHAPES is None:
        _SHAPES = json.loads(SHAPES_FILE.read_text())
    if model not in _SHAPES:
        raise MechError(f"unknown model {model!r}; models: {', '.join(sorted(_SHAPES))}")
    return _SHAPES[model]


def build_shapes(data: Path) -> dict:
    """Reads the registry from the weight and config files under the data root `data` (~/mpd-data)."""
    import struct

    engine = json.loads((data / "engine/vpd4l/export.json").read_text())["config"]
    decomposition = json.loads((data / "engine/vpd4l_decomposition/export.json").read_text())["config"]
    vpd = [{site: decomposition["subcomponents"][f"h.{l}.{'mlp' if site in ('c_fc', 'down_proj') else 'attn'}.{site}"]
            for site in SITES} for l in range(engine["n_layers"])]
    library = None
    start = data / "decomp/start.components.json"
    if start.exists():  # decomp's exact all-on start: part i of layer l's attention or MLP, in file order
        arm = next(r for r in json.loads(start.read_text()) if r["arm"] == LIBRARY_ARM)
        parts = [{"attn": 0, "mlp": 0} for _ in range(engine["n_layers"])]
        for c in arm["components"]:
            site = c["read"]["own"][0] if "own" in c["read"] else c["read"]["direction"]["site"]
            parts[site // len(SITES)]["attn" if site % len(SITES) < 4 else "mlp"] += 1
        library = {"source": "decomp/start.components.json", "arm": LIBRARY_ARM, "parts": parts}
    from huggingface_hub import hf_hub_download

    transcoders = data / "transcoders/qwen3-0.6b-lowl0"
    qwens = {}
    for name, repo in QWEN3.items():
        qwen = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        widths = None
        if name == "qwen3-0.6b":
            widths = []
            for l in range(qwen["num_hidden_layers"]):
                with open(transcoders / f"layer_{l}.safetensors", "rb") as f:
                    header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                widths.append(header["W_enc"]["shape"][0])
        qwens[name] = {
            "layers": qwen["num_hidden_layers"], "heads": qwen["num_attention_heads"],
            "kv_heads": qwen["num_key_value_heads"], "head_dim": qwen["head_dim"],
            "d_model": qwen["hidden_size"], "d_mlp": qwen["intermediate_size"], "vocab": qwen["vocab_size"],
            "views": {"native": True, "vpd": None, "library": None, "transcoder": widths},
            "source": [f"huggingface {repo} config.json"] + (["transcoders/qwen3-0.6b-lowl0/layer_*.safetensors"] if widths else []),
        }
    return {
        "vpd4l": {
            "layers": engine["n_layers"], "heads": engine["n_heads"], "kv_heads": engine["n_kv_heads"],
            "head_dim": engine["head_dim"], "d_model": engine["d_model"], "d_mlp": engine["d_mlp"],
            "vocab": engine["vocab"],
            "views": {"native": True, "vpd": vpd, "library": library, "transcoder": None},
            "source": ["engine/vpd4l/export.json", "engine/vpd4l_decomposition/export.json"],
        },
        **qwens,
    }


# ---------------------------------------------------------------------------------------------------
# Program objects

_PROGRAM: "_Program | None" = None


class _Program:
    def __init__(self, model: str):
        self.model = model
        self.shape = shapes(model)
        self.nodes: list[Node] = []
        self.edges: dict[tuple, Edge] = {}


def _shape() -> dict | None:
    return None if _PROGRAM is None else _PROGRAM.shape


def _indices(key, size: int | None, what: str) -> tuple[int, ...]:
    """The indices an address selects: ints, slices, ranges or lists of them, in range [0, size)."""
    out: list[int] = []
    for k in key if isinstance(key, tuple) else (key,):
        if isinstance(k, slice):
            if size is None:
                raise MechError(f"{what}: a slice needs the model's size")
            out.extend(range(*k.indices(size)))
        elif isinstance(k, (range, list)):
            out.extend(_indices(tuple(k), size, what))
        elif isinstance(k, int) and not isinstance(k, bool):
            if k < 0 or (size is not None and k >= size):
                raise MechError(f"{what}[{k}] is out of range 0..{size - 1 if size else '?'}")
            out.append(k)
        else:
            raise MechError(f"{what}: index {k!r} is not an int, slice or range")
    if not out:
        raise MechError(f"{what}: empty selection")
    return tuple(sorted(set(out)))


class Piece:
    """Pieces of one of M's sites: (view, layer, kind, indices)."""

    def __init__(self, view: str, layer: int, kind: str, index: tuple[int, ...], size: int | None = None):
        self.view, self.layer, self.kind, self.index, self.size = view, layer, kind, index, size

    def name(self) -> str:
        i = ", ".join(map(str, self.index[:4])) + (", ..." if len(self.index) > 4 else "")
        if self.view == "native":
            return f"L[{self.layer}].{self.kind}[{i}]"
        if self.view == "transcoder":
            return f"PD.tc[{self.layer}][{i}]"
        if self.view == "library":
            return f"PD.lib[{self.layer}].{self.kind}[{i}]"
        return f"PD.{'vpd' if self.view == 'vpd' else 'lib'}[{self.layer}].{self.kind}[{i}]"

    def block(self) -> str:
        return "mlp" if self.kind in ("mlp", "c_fc", "down_proj", "feature") else "attn"

    def reads(self) -> list[tuple]:
        """What the piece reads: ("resid", position, routes) or (internal stream, layer, routes).
        Residual positions: layer l's attention reads and writes at 2l, its MLP at 2l + 1."""
        l = self.layer
        if self.kind in ("head", "attn"):  # a native head, or a library part of the attention
            return [("resid", 2 * l, ("query", "key", "value", "input"))]
        if self.kind in ("mlp", "c_fc", "feature"):
            return [("resid", 2 * l + 1, ("input",))]
        if self.kind in ("q_proj", "k_proj", "v_proj"):
            return [("resid", 2 * l, ({"q_proj": "query", "k_proj": "key", "v_proj": "value"}[self.kind], "input"))]
        if self.kind == "o_proj":
            return [("attn", l, ("input",))]
        return [("hidden", l, ("input",))]  # down_proj reads its layer's MLP hidden activation

    def writes(self) -> list[tuple]:
        l = self.layer
        if self.kind in ("head", "attn", "o_proj"):
            return [("resid", 2 * l)]
        if self.kind in ("mlp", "down_proj", "feature"):  # "mlp": native neurons or a library part
            return [("resid", 2 * l + 1)]
        if self.kind in ("q_proj", "k_proj", "v_proj"):
            return [("attn", l)]
        return [("hidden", l)]  # c_fc writes its layer's MLP hidden pre-activation

    def __rshift__(self, other):
        raise MechError(f"{self.name()} is a piece; make it a node, node({self.name()}), before connecting it")

    def ir(self) -> dict:
        """index: one int, a sorted list, or null for every unit of the site."""
        whole = self.size is not None and len(self.index) == self.size
        return {"view": self.view, "layer": self.layer, "kind": self.kind,
                "index": None if whole else self.index[0] if len(self.index) == 1 else list(self.index)}


class _Site:
    def __init__(self, view: str, layer: int, kind: str, size: int | None):
        self.view, self.layer, self.kind, self.size = view, layer, kind, size

    def __getitem__(self, key) -> Piece:
        what = Piece(self.view, self.layer, self.kind, (0,)).name().rsplit("[", 1)[0]
        return Piece(self.view, self.layer, self.kind, _indices(key, self.size, what), self.size)

    def whole(self) -> Piece:
        """Every unit of the site, e.g. node(L[3].mlp)."""
        if self.size is None:
            raise MechError("a whole site needs the model's size")
        return Piece(self.view, self.layer, self.kind, tuple(range(self.size)), self.size)


def _layer(l, what: str) -> int:
    shape = _shape()
    if not isinstance(l, int) or isinstance(l, bool):
        raise MechError(f"{what}[{l!r}]: the layer must be an int")
    if l < 0 or (shape is not None and l >= shape["layers"]):
        raise MechError(f"{what}[{l}]: layers are 0..{shape['layers'] - 1 if shape else '?'}")
    return l


class _NativeLayer:
    def __init__(self, l: int):
        self._l = l

    def __getattr__(self, name: str):
        raise MechError(f"L[{self._l}].{name}: a layer has .head[h], .attn (all heads) and .mlp[i]")

    @property
    def head(self) -> _Site:
        shape = _shape()
        return _Site("native", self._l, "head", shape and shape["heads"])

    @property
    def attn(self) -> _Site:
        """The layer's attention: all of its heads, e.g. node(L[2].attn)."""
        return self.head

    @property
    def mlp(self) -> _Site:
        shape = _shape()
        return _Site("native", self._l, "mlp", shape and shape["d_mlp"])


class _Layers:
    def __getitem__(self, l) -> _NativeLayer:
        return _NativeLayer(_layer(l, "L"))


def _view(view: str, name: str):
    shape = _shape()
    if shape is not None and not shape["views"].get(view):
        raise MechError(f"PD.{name}: the {view} view is not available for {_PROGRAM.model}")
    return None if shape is None else shape["views"][view]


class _DecompLayer:
    def __init__(self, view: str, name: str, l: int):
        self._view, self._name, self._l = view, name, l

    def __getattr__(self, site: str) -> _Site:
        if site not in SITES:
            raise MechError(f"PD.{self._name}[{self._l}].{site}: sites are {', '.join(SITES)}")
        sizes = _view(self._view, self._name)
        return _Site(self._view, self._l, site, sizes and sizes[self._l][site])


class _LibLayer:
    """Our library's parts of layer l: .attn[i] (a part of the attention) and .mlp[i] (of the MLP)."""

    def __init__(self, l: int):
        self._l = l

    def _site(self, block: str) -> _Site:
        counts = _view("library", "lib")
        return _Site("library", self._l, block, counts and counts["parts"][self._l][block])

    attn = property(lambda self: self._site("attn"))
    mlp = property(lambda self: self._site("mlp"))

    def __getattr__(self, name: str):
        raise MechError(f"PD.lib[{self._l}].{name}: library parts are PD.lib[l].attn[i] and PD.lib[l].mlp[i]")


class _Decomp:
    def __init__(self, view: str, name: str):
        self._view, self._name = view, name

    def __getitem__(self, l):
        l = _layer(l, f"PD.{self._name}")
        if self._view == "transcoder":
            widths = _view(self._view, self._name)
            return _Site("transcoder", l, "feature", widths and widths[l])
        _view(self._view, self._name)
        return _LibLayer(l) if self._view == "library" else _DecompLayer(self._view, self._name, l)


class _PD:
    vpd = _Decomp("vpd", "vpd")
    lib = _Decomp("library", "lib")
    tc = _Decomp("transcoder", "tc")


class Node:
    """A node: a set of pieces that compute with their actual inputs, routed by declared edges."""

    def __init__(self, pieces: tuple[Piece, ...]):
        self.pieces = pieces
        self.id: str | None = None
        self.rule: Rule | None = None
        if _PROGRAM is not None:
            _PROGRAM.nodes.append(self)

    def _route(self, route: str) -> "Route":
        if self.rule is not None and route in ("query", "key"):
            raise MechError(f"node {self.label()}'s rule replaces its {route}; route its value or input")
        if not any(route in r for p in self.pieces for (_, _, r) in p.reads()):
            raise MechError(f"node {self.label()} has no {route} read (query/key/value need a head or a q/k/v_proj piece)")
        return Route(self, route)

    query = property(lambda self: self._route("query"))
    key = property(lambda self: self._route("key"))
    value = property(lambda self: self._route("value"))
    input = property(lambda self: self._route("input"))

    def label(self) -> str:
        return self.id or "(" + ", ".join(p.name() for p in self.pieces) + ")"

    def __rshift__(self, other) -> "Edge":
        return _edge(self, other)


class Route:
    """One read of a node: query, key, value, or input (all of its reads)."""

    def __init__(self, node: Node, route: str):
        self.node, self.route = node, route

    def __rshift__(self, other):
        raise MechError(f"{self.node.label()}.{self.route} is a read; write `{self.node.label()} >> ...`")


class _Embed:
    def __rshift__(self, other) -> "Edge":
        return _edge(self, other)

    def __repr__(self) -> str:
        return "embed"


class _Logits:
    def __rshift__(self, other):
        raise MechError("logits is a reader; it cannot write")

    def __repr__(self) -> str:
        return "logits"


L = _Layers()
PD = _PD()
embed = _Embed()
logits = _Logits()


class Edge:
    """writer >> reader: the writer's actual write reaches the reader's route."""

    def __init__(self, src, dst, route: str):
        self.src, self.dst, self.route = src, dst, route

    def __rshift__(self, other) -> "Edge":
        if not isinstance(self.dst, Node):
            raise MechError("logits cannot write")
        return _edge(self.dst, other)


def _edge(src, dst) -> Edge:
    if isinstance(dst, Node):
        dst = dst.input
    if isinstance(dst, Route):
        target, route = dst.node, dst.route
    elif dst is logits:
        target, route = logits, "input"
    else:
        raise MechError(f"{dst!r} is not a reader (a node, a route handle such as node.key, or logits)")
    if src is target:
        raise MechError(f"node {src.label()} cannot read itself")
    writes = [("resid", -1)] if src is embed else [w for p in src.pieces for w in p.writes()]
    reads = ([("resid", 2 * _PROGRAM.shape["layers"] if _PROGRAM else 10**9, ("input",))] if target is logits
             else [r for p in target.pieces for r in p.reads()])
    ok = any((ws == rs == "resid" and w < r or ws == rs != "resid" and w == r) and route in routes
             for ws, w in writes for rs, r, routes in reads)
    if not ok:
        name = "embed" if src is embed else src.label()
        reader = "logits" if target is logits else f"{target.label()}.{route}"
        raise MechError(f"edge {name} >> {reader} connects nothing: no piece of the writer writes "
                        f"where a piece of the reader reads it later (writers must precede readers)")
    edge = Edge(src, target, route)
    if _PROGRAM is not None:
        _PROGRAM.edges.setdefault((id(src), id(target), route), edge)
    return edge


class Expr:
    """A token-level quantity a rule compares: `tokens` (the token at a position) or shift(expr, k)
    (expr at the position k earlier)."""

    def __init__(self, ir: dict):
        self.ir = ir


tokens = Expr({"op": "tokens"})


def shift(expr: Expr, k: int) -> Expr:
    """`expr` read k positions earlier: shift(tokens, 1) is the previous token."""
    if not isinstance(expr, Expr) or not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise MechError("shift(expr, k): expr is tokens or a shift, k a positive int")
    return Expr({"op": "shift", "arg": expr.ir, "by": k})


class Rule:
    def __init__(self, ir: dict):
        self.ir = ir


def attend(offset: int | None = None, query: Expr | None = None, key: Expr | None = None,
           first: bool = False) -> Rule:
    """An attention rule for a head node, replacing its query and key computation: the head attends
    uniformly to the earlier positions j that satisfy the rule, and computes its value and output with
    the model's own weights on its actual (routed) value input. attend(offset=k): j = t - k;
    attend(query=q, key=k): every j < t with k at j equal to q at t (induction: query=tokens,
    key=shift(tokens, 1)); attend(first=True): j = 0. With no such j the head attends to position 0."""
    given = (offset is not None) + (query is not None or key is not None) + bool(first)
    if given != 1:
        raise MechError("attend(): give exactly one of offset=k, query=... with key=..., or first=True")
    if offset is not None:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise MechError("attend(offset=k): k is an int >= 0")
        return Rule({"op": "attend", "offset": offset})
    if first:
        return Rule({"op": "attend", "first": True})
    if not isinstance(query, Expr) or not isinstance(key, Expr):
        raise MechError("attend(query=..., key=...): both are tokens or shift(tokens, k)")
    return Rule({"op": "attend", "query": query.ir, "key": key.ir})


def node(*pieces, rule=None) -> Node:
    """One node made of `pieces` (addresses such as L[1].head[1] or PD.vpd[2].c_fc[5, 9]); `rule`: an
    attention rule (attend(...)) for a node of native heads."""
    if rule is not None and not isinstance(rule, Rule):
        raise MechError("rule=: an attention rule such as attend(offset=1)")
    pieces = tuple(q for p in pieces for q in (p if isinstance(p, (list, tuple)) else (p,)))  # node([a, b]) too
    if not pieces:
        raise MechError("node() needs at least one piece")
    pieces = tuple(p.whole() if isinstance(p, _Site) else p for p in pieces)
    for p in pieces:
        if not isinstance(p, Piece):
            raise MechError(f"node(): {p!r} is not a piece address such as L[1].head[1]")
    if len({(p.layer, p.block()) for p in pieces}) > 1:
        raise MechError("a node's pieces must lie in one layer's attention or one layer's MLP; "
                        "make one node per site and connect them with edges")
    if rule is not None and any(p.kind != "head" for p in pieces):
        raise MechError("an attention rule applies to a node of native heads (L[l].head[...])")
    merged: dict[tuple, set] = {}
    for p in pieces:
        merged.setdefault((p.view, p.layer, p.kind, p.size), set()).update(p.index)
    made = Node(tuple(Piece(v, l, k, tuple(sorted(i)), n) for (v, l, k, n), i in merged.items()))
    made.rule = rule
    return made


def edges(*declared) -> None:
    """Lists the program's edges (each `writer >> reader` is declared where it is written)."""
    for e in (f for d in declared for f in (d if isinstance(d, (list, tuple)) else (d,))):  # edges([...]) too
        if not isinstance(e, Edge):
            raise MechError(f"edges(): {e!r} is not an edge `writer >> reader`")


# ---------------------------------------------------------------------------------------------------
# Sandbox and tracer

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Attribute, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Call, ast.keyword, ast.Starred, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context, ast.FunctionDef, ast.Lambda,
    ast.arguments, ast.arg, ast.Return, ast.Pass, ast.For, ast.If, ast.Break, ast.Continue,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.JoinedStr,
    ast.FormattedValue, ast.ImportFrom, ast.alias, ast.Assert, ast.NamedExpr,
)
BANNED_ATTRIBUTES = {
    "format", "format_map", "mro", "gi_frame", "gi_code", "gi_yieldfrom", "gi_running", "cr_frame",
    "cr_code", "cr_await", "ag_frame", "ag_code", "ag_await", "f_back", "f_globals", "f_locals",
    "f_builtins", "f_code", "f_trace", "tb_frame", "tb_next", "co_code", "co_consts", "co_names",
}
SAFE_BUILTINS = {name: getattr(builtins, name)
                 for name in ("range", "len", "list", "tuple", "dict", "set", "frozenset", "int", "float",
                              "bool", "str", "enumerate", "zip", "min", "max", "sum", "abs", "sorted",
                              "reversed", "any", "all", "round", "divmod", "map", "filter", "isinstance",
                              "repr", "chr", "ord", "iter", "next")}
SAFE_BUILTINS["print"] = lambda *a, **k: None
MAX_SOURCE = 200_000


def _docstring_ranges(tree: ast.AST, source: str) -> list[tuple]:
    """Source ranges (lines, character columns) of bare string statements: docstrings and string
    comments."""
    lines = source.splitlines(keepends=True)

    def column(line: int, offset: int) -> int:  # ast offsets count UTF-8 bytes, tokenize characters
        return len(lines[line - 1].encode()[:offset].decode("utf-8", errors="ignore"))

    return [((n.lineno, column(n.lineno, n.col_offset)), (n.end_lineno, column(n.end_lineno, n.end_col_offset)))
            for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]


def check(tree: ast.AST) -> list[tuple[str, str]]:
    """What the program imports from mech, as (export or "*" or "mech", bound name); raises MechError on
    a forbidden construct."""
    imported: list[tuple[str, str]] = []
    docs = {id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    for n in ast.walk(tree):
        line = getattr(n, "lineno", "?")
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name != "mech":
                    raise MechError(f"line {line}: only `from mech import ...` (or `import mech`) is allowed")
                imported.append(("mech", a.asname or "mech"))
            continue
        if not isinstance(n, ALLOWED_NODES):
            raise MechError(f"line {line}: {type(n).__name__} is not allowed in a mech program")
        if isinstance(n, ast.ImportFrom):
            if n.module != "mech" or n.level:
                raise MechError(f"line {line}: only `from mech import ...` is allowed")
            for a in n.names:
                if a.name != "*" and a.name not in EXPORTS:
                    raise MechError(f"line {line}: mech has no {a.name!r}; it exports {', '.join(EXPORTS)}")
                imported.append((a.name, a.asname or a.name))
        elif isinstance(n, ast.Attribute) and (n.attr.startswith("_") or n.attr in BANNED_ATTRIBUTES):
            raise MechError(f"line {line}: attribute {n.attr!r} is not allowed")
        elif isinstance(n, ast.Name) and n.id.startswith("__"):
            raise MechError(f"line {line}: name {n.id!r} is not allowed")
        elif isinstance(n, (ast.FunctionDef, ast.Lambda)) and getattr(n, "decorator_list", None):
            raise MechError(f"line {line}: decorators are not allowed")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and "__" in n.value:
            raise MechError(f"line {line}: strings containing '__' are not allowed outside docstrings")
    return imported


class _MechModule:
    """`import mech` inside a program: the exports as attributes."""

    def __init__(self, exports: dict):
        for name, value in exports.items():
            setattr(self, name, value)


class _NoImports(ast.NodeTransformer):
    """Replaces the (checked) import statements by `pass`: their names are bound beforehand."""

    def visit_Import(self, n):
        return ast.copy_location(ast.Pass(), n)

    visit_ImportFrom = visit_Import


def _line_of(exc: BaseException) -> int | None:
    lines = [f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<program>"]
    return lines[-1] if lines else None


def trace_inline(source: str, model: str) -> dict:
    """Checks and runs `source` in this process (trusted sources only; `trace` sandboxes) -> IR."""
    global _PROGRAM
    ir = {"model": model, "standin": DEFAULT_STANDIN, "nodes": [], "edges": [], "python_tokens": 0,
          "token_types": 0, "source": source, "valid": False, "error": None}
    try:
        ir["python_tokens"], ir["token_types"] = code_length(source)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        pass
    try:
        if model not in MODELS:
            raise MechError(f"unknown model {model!r}; models: {', '.join(MODELS)}")
        if len(source) > MAX_SOURCE:
            raise MechError(f"program longer than {MAX_SOURCE} characters")
        tree = ast.parse(source, "<program>")
        imported = check(tree)
        exports = {name: globals()[name] for name in EXPORTS}
        namespace = {"__builtins__": SAFE_BUILTINS, "__name__": "program"}
        for name, bound in imported:
            if name == "*":
                namespace.update(exports)
            elif name == "mech":
                namespace[bound] = _MechModule(exports)
            else:
                namespace[bound] = exports[name]
        tree = ast.fix_missing_locations(_NoImports().visit(tree))
        _PROGRAM = program = _Program(model)
        try:
            exec(compile(tree, "<program>", "exec"), namespace)
        finally:
            _PROGRAM = None
        _validate(program, namespace, ir)
        ir["valid"] = True
    except MechError as e:
        line = _line_of(e)
        ir["error"] = (f"line {line}: " if line else "") + str(e)
    except SyntaxError as e:
        ir["error"] = f"line {e.lineno}: syntax error: {e.msg}"
    except RecursionError:
        ir["error"] = "recursion too deep"
    except Exception as e:
        line = _line_of(e)
        ir["error"] = (f"line {line}: " if line else "") + f"{type(e).__name__}: {e}"
    if not ir["valid"]:
        ir["nodes"], ir["edges"] = [], []
    return ir


def _validate(program: _Program, namespace: dict, ir: dict) -> None:
    named = set()
    for name, value in namespace.items():
        if isinstance(value, Node) and value.id is None and not name.startswith("_"):
            value.id = name
            named.add(name)
    k = 0
    for n in program.nodes:
        while n.id is None:
            if f"node{k}" not in named:
                n.id = f"node{k}"
            k += 1
    owner: dict[tuple, str] = {}
    view_of: dict[tuple, str] = {}
    for n in program.nodes:
        for p in n.pieces:
            v = view_of.setdefault((p.layer, p.block()), p.view)
            if v != p.view:
                raise MechError(f"layer {p.layer}'s {p.block()} is read through two views ({v} and {p.view}); "
                                f"use one view per layer's attention and MLP")
            if p.view == "library":
                continue  # library parts may overlap; the checker takes the union per node
            for i in p.index:
                o = owner.setdefault((p.view, p.layer, p.kind, i), n.id)
                if o != n.id:
                    raise MechError(f"{Piece(p.view, p.layer, p.kind, (i,)).name()} is in nodes {o} and {n.id}; "
                                    f"a piece belongs to one node")
    ir["standin"] = DEFAULT_STANDIN
    ir["nodes"] = [{"id": n.id, "pieces": [p.ir() for p in n.pieces], "rule": n.rule and n.rule.ir}
                   for n in program.nodes]
    ir["edges"] = [{"from": "embed" if e.src is embed else e.src.id,
                    "to": "logits" if e.dst is logits else e.dst.id, "route": e.route}
                   for e in program.edges.values()]


MEMORY = 1 << 30  # bytes a traced program may use


def _limit(seconds: float) -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (int(seconds) + 1, int(seconds) + 2))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY + (1 << 30), MEMORY + (1 << 30)))
    except (ValueError, OSError):
        pass  # macOS does not enforce address-space limits; trace() watches the footprint instead


_LIBC = None


def _footprint(pid: int) -> int:
    """macOS: the process's physical footprint (rusage_info_v2 ri_phys_footprint), 0 if unknown."""
    global _LIBC
    import ctypes
    import ctypes.util
    import struct

    if _LIBC is None:
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
    buf = ctypes.create_string_buffer(256)
    if _LIBC.proc_pid_rusage(pid, 2, buf) != 0:
        return 0
    return struct.unpack_from("Q", buf.raw, 72)[0]


def _invalid(source: str, model: str, error: str) -> dict:
    try:
        tokens, types = code_length(source)
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        tokens, types = 0, 0
    return {"model": model, "standin": DEFAULT_STANDIN, "nodes": [], "edges": [], "python_tokens": tokens,
            "token_types": types, "source": source, "valid": False, "error": error}


def _traced_child(source: str, model: str, timeout: float) -> dict:
    """Forks a child that traces `source` under CPU/memory limits; waits for it with a wall-clock
    deadline, watching its footprint on macOS (RLIMIT_AS is not enforced there)."""
    import select
    import time

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(r)
            _limit(timeout)
            sys.setrecursionlimit(500)
            data = json.dumps(trace_inline(source, model)).encode()
            view = memoryview(data)
            while view:
                view = view[os.write(w, view):]
        finally:
            os._exit(0)
    os.close(w)
    chunks, deadline, over, late = [], time.monotonic() + timeout + 2, False, False
    try:
        while True:
            ready, _, _ = select.select([r], [], [], 0.005)
            if ready:
                chunk = os.read(r, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
            elif sys.platform == "darwin" and _footprint(pid) > MEMORY:
                over = True
                os.kill(pid, signal.SIGKILL)
                break
            elif time.monotonic() > deadline:
                late = True
                os.kill(pid, signal.SIGKILL)
                break
    finally:
        os.close(r)
        _, status = os.waitpid(pid, 0)
    if over or (os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL and not late):
        return _invalid(source, model, f"memory limit of {MEMORY >> 20} MiB exceeded")
    if late or (os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGXCPU):
        return _invalid(source, model, f"time limit of {timeout} s exceeded")
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        return _invalid(source, model, f"the program crashed the tracer (status {status})")


def serve() -> None:
    """JSON lines on stdin {"source", "model", "timeout"} -> IR lines on stdout, one forked child each."""
    for model in MODELS:
        shapes(model)
    for line in sys.stdin:
        req = json.loads(line)
        print(json.dumps(_traced_child(req["source"], req["model"], req.get("timeout", 10.0))), flush=True)


_SERVERS = threading.local()


def trace(source: str, model: str, timeout: float = 10.0) -> dict:
    """Checks and runs `source` sandboxed (restricted names and builtins; a forked child with CPU and
    memory limits and a wall-clock deadline) -> IR dict (design.txt section 5). Each thread keeps one
    tracer server (`mech.py serve`, started with -I -S: no site hooks, the standard library only, no
    memory-ledger reservation), so a trace costs a fork, a few milliseconds."""
    for attempt in range(2):
        server = getattr(_SERVERS, "proc", None)
        if server is None or server.poll() is not None or _SERVERS.pid != os.getpid():  # a forked caller starts its own
            _SERVERS.pid = os.getpid()
            server = _SERVERS.proc = subprocess.Popen(
                [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                env={**os.environ, "MPD_MEM_GIB": "1"})
        try:
            server.stdin.write(json.dumps({"source": source, "model": model, "timeout": timeout}) + "\n")
            server.stdin.flush()
            line = server.stdout.readline()
            if line:
                return json.loads(line)
        except (BrokenPipeError, OSError, ValueError):
            pass
        server.kill()
        _SERVERS.proc = None
    return _invalid(source, model, "the tracer server failed")


def trace_many(sources: list[str], model: str, timeout: float = 10.0, workers: int = 8) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda s: trace(s, model, timeout), sources))


# ---------------------------------------------------------------------------------------------------
# Code length and English

KEYWORDS = ("False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
            "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
            "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield")
OPERATORS = ("!", "!=", "%", "%=", "&", "&=", "(", ")", "*", "**", "**=", "*=", "+", "+=", ",", "-", "-=",
             "->", ".", "...", "/", "//", "//=", "/=", ":", ":=", ";", "<", "<<", "<<=", "<=", "=", "==", ">",
             ">=", ">>", ">>=", "@", "@=", "[", "]", "^", "^=", "{", "|", "|=", "}", "~")
LITERAL_CHARACTERS = {chr(c) for c in range(32, 127)} | {"\t", "\n"}
FIXED_NAMES = set(EXPORTS) | set(ATTRIBUTES) | set(SAFE_BUILTINS) | {"mech"}
SKIPPED = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
           tokenize.ENCODING, tokenize.ENDMARKER}


def code_length(source: str) -> tuple[int, int]:
    """(python_tokens, token_types). Tokens of the code without comments and docstrings: a name or
    keyword or operator is one token, a number or string literal one token per character as written.
    token_types = keywords + operators + mech and builtin names + names the program defines +
    literal characters (printable ASCII, tab, newline and any other character the literals use)."""
    tree = ast.parse(source)
    docs = _docstring_ranges(tree, source)

    def in_doc(start, end) -> bool:
        return any(a <= start and end <= b for a, b in docs)

    tokens = 0
    names: set[str] = set()
    characters = set(LITERAL_CHARACTERS)
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type in SKIPPED or not t.string:
            continue
        literal = t.type in (tokenize.NUMBER, tokenize.STRING) or "STRING" in tokenize.tok_name[t.type]
        if literal:
            if t.type == tokenize.STRING and in_doc(t.start, t.end):
                continue
            tokens += len(t.string)
            characters.update(t.string)
            continue
        tokens += 1
        if t.type == tokenize.NAME and t.string not in KEYWORDS and t.string not in FIXED_NAMES:
            names.add(t.string)
    return tokens, len(KEYWORDS) + len(OPERATORS) + len(FIXED_NAMES) + len(names) + len(characters)


def english(source: str) -> str:
    """The program's comments and docstrings (bare string statements) in source order, one per line."""
    tree = ast.parse(source)
    items = [((n.lineno, n.col_offset), inspect.cleandoc(n.value.value)) for n in ast.walk(tree)
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    items += [(t.start, t.string[1:].strip()) for t in tokenize.generate_tokens(io.StringIO(source).readline)
              if t.type == tokenize.COMMENT]
    return "\n".join(text for _, text in sorted(items) if text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trace", help="stdin program -> IR JSON on stdout (sandboxed child entry point)")
    t.add_argument("--model", required=True)
    t.add_argument("--timeout", type=float, default=10.0)
    t.add_argument("--inline", action="store_true", help="no CPU/memory limits (trusted programs)")
    s = sub.add_parser("shapes", help="print or rebuild the model shape registry")
    s.add_argument("--data", type=Path, default=Path.home() / "mpd-data")
    s.add_argument("--write", action="store_true")
    sub.add_parser("english", help="stdin program -> its comments and docstrings")
    sub.add_parser("serve", help="the tracer server (JSON lines; trace() starts it)")
    a = ap.parse_args()
    if a.command == "trace":
        source = sys.stdin.read()
        if not a.inline:
            _limit(a.timeout)
        sys.setrecursionlimit(500)
        print(json.dumps(trace_inline(source, a.model)))
    elif a.command == "serve":
        serve()
    elif a.command == "shapes":
        built = build_shapes(a.data)
        if a.write:
            SHAPES_FILE.write_text(json.dumps(built, indent=1) + "\n")
        print(json.dumps(built, indent=1))
    else:
        print(english(sys.stdin.read()))


if __name__ == "__main__":
    main()
