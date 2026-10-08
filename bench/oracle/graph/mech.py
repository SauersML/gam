"""The `mech` library of the graph oracle (#2951): the only module an oracle program may import.

A program (design_v2 section 1) is an ALGORITHM in plain Python over the prompt's tokens, BINDINGS that
name the parts of the target model M holding its variables, and optional attention CLAIMS. Induction on
vpd4l with VPD attached ("A B ... A -> B"):

    from mech import bind, claim

    def back(tokens):
        # each position attends to the one before it
        return [[t - 1] if t else [0] for t in range(len(tokens))]

    def prev(tokens, back):
        # the token before each position, which layer 1 moves forward one position
        return [tokens[js[0]] if t else None for t, js in enumerate(back)]

    def match(tokens, prev):
        # each position attends to the earlier positions whose previous token is its own token
        return [[j for j in range(t) if prev[j] == tokens[t]] for t in range(len(tokens))]

    def answer(tokens, match):
        # the token at the latest such position: the one that followed the current token before
        return [tokens[js[-1]] if js else None for js in match]

    claim(back, <p:1.q.316>, <p:1.k.329>)
    bind(prev, <p:1.v.228>, <p:1.v.346>, <p:1.o.311>, <p:1.o.340>)
    claim(match, <p:2.q.335>, <p:2.k.206>)
    bind(answer, <p:2.v.559>, <p:2.o.735>, <p:3.v.677>, <p:3.o.806>)

The algorithm. A variable is a top-level function, named by its name, that takes `tokens` (the prompt as
M's token strings, such as " cat") and other variables (by parameter name) and returns one value per
position: a string, number, bool or None, or a list or tuple of them. Its value at position t may use
tokens 0..t only. Other functions are helpers. Comments and docstrings are free working notes.
bind(variable, parts...): what the parts write into the residual stream holds the variable. A variable
may span layers (one node per layer's attention or MLP); a part belongs to one node.
claim(pattern, parts...): `pattern`'s value at t lists the positions 0..t that the parts' attention at
query t attends to, uniformly (none: position 0), or maps positions to weights; the parts are q_proj
and k_proj parts (or native heads), in one layer or several (the pattern holds in each). A claim states
what the parts compute; they still compute it with M's weights.
The answer is the one bound variable that no variable reads: its value at t is the token M predicts
after position t (a longer string: its first token).
Edges follow the data flow: a variable reading `tokens` reads the token embedding; one reading another
variable reads the writes of that variable's parts (through unbound steps); a variable spanning layers
feeds its own later parts; the answer's parts and the embedding write the logits.

Parts (layer L, index I, from 0). The oracle writes one part token per part; text spellings in brackets:
  <p:L.S.I>     [PD[L].<site>[I, ...]] part I of the attached decomposition at layer L's site S:
                VPD: S in q k v o fc down (sites q_proj k_proj v_proj o_proj c_fc down_proj);
                our library: S in attn mlp; transcoders: S = mlp (the features of layer L's MLP)
  <p:L.S.rest>  [PD[L].<site>.rest] a VPD site's remainder W - sum of its subcomponents
  <p:L.h.I>     [L[L].head[I, ...]] a native attention head; <p:L.a> [L[L].attn] all heads of layer L
  <p:L.m>       [L[L].mlp] a native MLP
Native units appear only where the attached decomposition leaves a block uncovered: none with VPD or the
library, heads with transcoders. Text indices may be ints, slices or ranges.

The low-level form, without an algorithm: node(*parts) makes one node; `writer >> reader` is an edge
(writer: a node or embed; reader: a node, a route node.query/.key/.value/.input, or logits); edges(...)
lists them.

trace(source, model, behavior=...) checks a program, runs it in a sandboxed child process and, given
the behavior, evaluates the algorithm on its prompts: per bound variable, interchange pairs (prompt i
with the variable's value from prompt j, the next prompt of i's length, and the answer at each of i's
targets), and each claim's pattern on every prompt and counterfactual. It returns the IR the checker
reads. Parts a program leaves out are the checker's stand-ins (deleted, with a decomposition attached).
code_length counts Python tokens (a part token is one); english extracts comments and docstrings;
shapes(model) is the registry of model sizes (shapes.json, `mech.py shapes --write`).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import inspect
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import tokenize
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES_FILE = HERE / "shapes.json"
MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-8b", "vpd4l")
QWEN3 = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3-8b": "Qwen/Qwen3-8B"}
VPD4L_TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
VIEWS = ("native", "vpd", "library", "transcoder")
SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")
# Per decomposition: its sites (PD[l].<site>) and the blocks it covers (no native units there).
DECOMPOSITIONS = {"vpd": SITES, "library": ("attn", "mlp"), "transcoder": ("mlp",)}
COVERS = {"vpd": ("attn", "mlp"), "library": ("attn", "mlp"), "transcoder": ("mlp",)}
DEFAULT_DECOMPOSITION = {"vpd4l": "vpd", "qwen3-0.6b": "transcoder"}
CODES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down",
         "attn": "attn", "mlp": "mlp"}  # a site's code in part tokens
SITE_OF = {code: site for site, code in CODES.items()}
PART = re.compile(r"<p:(\d+)\.(?:(q|k|v|o|fc|down|attn|mlp)\.(\d+|rest)|h\.(\d+)|(a|m))>")
ROUTES = ("query", "key", "value", "input")
EXPORTS = ("bind", "claim", "node", "edges", "L", "PD", "embed", "logits")
ATTRIBUTES = ("head", "attn", "mlp", "rest", "query", "key", "value", "input") + SITES
LIBRARY_ARM = "grouped_own"  # the arm of decomp's start that the library view addresses


class MechError(Exception):
    """An invalid program: a bad address, edge, binding or construct."""


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
    def __init__(self, model: str, decomposition: str | None, namespace: dict):
        self.model = model
        self.shape = shapes(model)
        self.decomposition = decomposition
        self.namespace = namespace
        self.nodes: list[Node] = []
        self.edges: dict[tuple, Edge] = {}
        self.bound: dict[str, list[Piece]] = {}
        self.claimed: dict[str, list[Piece]] = {}


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
    """Parts of one of M's sites: (view, layer, kind, indices). The view is "native" or the attached
    decomposition; kind is the IR's: head, mlp (native), a VPD site, attn/mlp (library), feature
    (transcoder)."""

    def __init__(self, view: str, layer: int, kind: str, index: tuple[int, ...], size: int | None = None,
                 rest: bool = False):
        self.view, self.layer, self.kind, self.index, self.size, self.rest = view, layer, kind, index, size, rest

    def site(self) -> str:
        """The site's name in PD[l].<site>."""
        return "mlp" if self.kind == "feature" else self.kind

    def whole(self) -> bool:
        return self.size is not None and len(self.index) == self.size

    def name(self) -> str:
        l = self.layer
        if self.rest:
            return f"PD[{l}].{self.site()}.rest"
        if self.view == "native" and self.whole():
            return f"L[{l}].{'attn' if self.kind == 'head' else 'mlp'}"
        i = ", ".join(map(str, self.index[:4])) + (", ..." if len(self.index) > 4 else "")
        if self.view == "native":
            return f"L[{l}].{self.kind}[{i}]"
        return f"PD[{l}].{self.site()}[{i}]"

    def tokens(self) -> list[str]:
        """The part tokens of its units."""
        l = self.layer
        if self.rest:
            return [f"<p:{l}.{CODES[self.site()]}.rest>"]
        if self.view == "native":
            if self.whole():
                return [f"<p:{l}.{'a' if self.kind == 'head' else 'm'}>"]
            if self.kind != "head":
                raise MechError(f"{self.name()}: MLP neurons have no part tokens")
            return [f"<p:{l}.h.{i}>" for i in self.index]
        return [f"<p:{l}.{CODES[self.site()]}.{i}>" for i in self.index]

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
        raise MechError(f"{self.name()} is a part; make it a node, node({self.name()}), before connecting it")

    def ir(self) -> dict:
        """index: one int, a sorted list, null for every unit of the site, or "rest" for a VPD site's
        remainder W - sum of its subcomponents."""
        if self.rest:
            return {"view": self.view, "layer": self.layer, "kind": self.kind, "index": "rest"}
        return {"view": self.view, "layer": self.layer, "kind": self.kind,
                "index": None if self.whole() else self.index[0] if len(self.index) == 1 else list(self.index)}


class _Site:
    def __init__(self, view: str, layer: int, kind: str, size: int | None):
        self.view, self.layer, self.kind, self.size = view, layer, kind, size

    def _what(self) -> str:
        return Piece(self.view, self.layer, self.kind, (0,)).name().rsplit("[", 1)[0]

    def __getitem__(self, key) -> Piece:
        return Piece(self.view, self.layer, self.kind, _indices(key, self.size, self._what()), self.size)

    @property
    def rest(self) -> Piece:
        """A VPD site's remainder W - sum of its subcomponents, e.g. PD[2].q_proj.rest."""
        if self.view != "vpd":
            raise MechError(f"{self._what()}.rest: only a VPD site has a remainder")
        return Piece(self.view, self.layer, self.kind, (), self.size, rest=True)

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
        raise MechError(f"L[{self._l}].{name}: a layer has .head[h], .attn (all heads) and .mlp")

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


def _decomposition(what: str) -> str:
    if _PROGRAM is None or _PROGRAM.decomposition is None:
        raise MechError(f"{what}: no decomposition is attached" + (f" for {_PROGRAM.model}" if _PROGRAM else ""))
    return _PROGRAM.decomposition


def _site(decomposition: str, l: int, site: str) -> _Site:
    """Layer l's site of the attached decomposition."""
    if site not in DECOMPOSITIONS[decomposition]:
        raise MechError(f"PD[{l}].{site}: the {decomposition} decomposition's sites are "
                        f"{', '.join(DECOMPOSITIONS[decomposition])}")
    sizes = _PROGRAM.shape["views"].get(decomposition)
    if not sizes:
        raise MechError(f"PD[{l}].{site}: no {decomposition} decomposition exists for {_PROGRAM.model}")
    if decomposition == "vpd":
        return _Site("vpd", l, site, sizes[l][site])
    if decomposition == "library":
        return _Site("library", l, site, sizes["parts"][l][site])
    return _Site("transcoder", l, "feature", sizes[l])


class _PDLayer:
    """Layer l's parts of the attached decomposition: PD[l].<site>[i, ...] and PD[l].<site>.rest."""

    def __init__(self, l: int):
        self._l = l

    def __getattr__(self, site: str) -> _Site:
        return _site(_decomposition(f"PD[{self._l}].{site}"), self._l, site)


class _PD:
    def __getitem__(self, l) -> _PDLayer:
        return _PDLayer(_layer(l, "PD"))

    def __getattr__(self, name: str):
        raise MechError(f"PD.{name}: PD[l].<site>[i] names part i of the attached decomposition (or a part token "
                        "such as <p:2.v.559>)")


def _part(token: str) -> Piece:
    """The part a part token names (<p:2.v.559>, <p:1.q.rest>, <p:3.h.4>, <p:0.a>, <p:1.m>)."""
    m = PART.fullmatch(token)
    if m is None:
        raise MechError(f"{token!r} is not a part token (<p:L.S.I>, <p:L.S.rest>, <p:L.h.I>, <p:L.a>, <p:L.m>)")
    l = _layer(int(m[1]), "<p:")
    if m[2]:
        site = _site(_decomposition(token), l, SITE_OF[m[2]])
        return site.rest if m[3] == "rest" else site[int(m[3])]
    native = _NativeLayer(l)
    if m[4]:
        return native.head[int(m[4])]
    return (native.attn if m[5] == "a" else native.mlp).whole()


def _covered(p: Piece) -> None:
    """Native units only where the attached decomposition leaves the block uncovered."""
    decomposition = _PROGRAM and _PROGRAM.decomposition
    if p.view == "native" and decomposition and p.block() in COVERS[decomposition]:
        sites = ", ".join(f"PD[{p.layer}].{s}" for s in DECOMPOSITIONS[decomposition]
                          if decomposition != "vpd" or (s in ("c_fc", "down_proj")) == (p.block() == "mlp"))
        raise MechError(f"{p.name()}: layer {p.layer}'s {p.block()} is decomposed by {decomposition}; "
                        f"name its parts ({sites}) instead of native units")


def _pieces(parts, what: str) -> tuple[Piece, ...]:
    """Part arguments (pieces, whole sites, part tokens, or lists of them) as pieces."""
    out = []
    for p in (q for p in parts for q in (p if isinstance(p, (list, tuple)) else (p,))):
        if isinstance(p, str):
            p = _part(p)
        elif isinstance(p, _Site):
            p = p.whole()
        if not isinstance(p, Piece):
            raise MechError(f"{what}: {p!r} is not a part (a part token such as <p:2.v.559>, or PD[2].v_proj[559])")
        _covered(p)
        out.append(p)
    if not out:
        raise MechError(f"{what} needs at least one part")
    return tuple(out)


def _merged(pieces) -> tuple[Piece, ...]:
    merged: dict[tuple, set] = {}
    for p in pieces:
        merged.setdefault((p.view, p.layer, p.kind, p.size, p.rest), set()).update(p.index)
    return tuple(Piece(v, l, k, tuple(sorted(i)), n, r) for (v, l, k, n, r), i in merged.items())


class Node:
    """A node: a set of pieces that compute with their actual inputs, routed by declared edges."""

    def __init__(self, pieces: tuple[Piece, ...], register: bool = True):
        self.pieces = pieces
        self.id: str | None = None
        self.claim: dict | None = None
        if _PROGRAM is not None and register:
            _PROGRAM.nodes.append(self)

    def _route(self, route: str) -> "Route":
        if not any(route in r for p in self.pieces for (_, _, r) in p.reads()):
            raise MechError(f"node {self.label()} has no {route} read (query/key/value need a head or a q/k/v_proj part)")
        return Route(self, route)

    query = property(lambda self: self._route("query"))
    key = property(lambda self: self._route("key"))
    value = property(lambda self: self._route("value"))
    input = property(lambda self: self._route("input"))

    def label(self) -> str:
        return self.id or "(" + ", ".join(p.name() for p in self.pieces) + ")"

    def site(self) -> tuple[int, str]:
        p = self.pieces[0]
        return p.layer, p.block()

    def writes_residual(self) -> bool:
        return any(w[0] == "resid" for p in self.pieces for w in p.writes())

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


def _connects(src, target, route: str) -> bool:
    """Whether a piece of `src` (a node or embed) writes where a piece of `target` (a node or logits)
    reads it later through `route`."""
    writes = [("resid", -1)] if src is embed else [w for p in src.pieces for w in p.writes()]
    reads = ([("resid", 2 * _PROGRAM.shape["layers"] if _PROGRAM else 10**9, ("input",))] if target is logits
             else [r for p in target.pieces for r in p.reads()])
    return any((ws == rs == "resid" and w < r or ws == rs != "resid" and w == r) and route in routes
               for ws, w in writes for rs, r, routes in reads)


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
    if not _connects(src, target, route):
        name = "embed" if src is embed else src.label()
        reader = "logits" if target is logits else f"{target.label()}.{route}"
        raise MechError(f"edge {name} >> {reader} connects nothing: no piece of the writer writes "
                        f"where a piece of the reader reads it later (writers must precede readers)")
    edge = Edge(src, target, route)
    if _PROGRAM is not None:
        _PROGRAM.edges.setdefault((id(src), id(target), route), edge)
    return edge


def node(*parts) -> Node:
    """One node made of `parts` (part tokens such as <p:2.v.559>, or PD[2].c_fc[5, 9], L[1].head[1])."""
    pieces = _pieces(parts, "node()")
    if len({(p.layer, p.block()) for p in pieces}) > 1:
        raise MechError("a node's parts must lie in one layer's attention or one layer's MLP; "
                        "make one node per site and connect them with edges")
    return Node(_merged(pieces))


def edges(*declared) -> None:
    """Lists the program's edges (each `writer >> reader` is declared where it is written)."""
    for e in (f for d in declared for f in (d if isinstance(d, (list, tuple)) else (d,))):  # edges([...]) too
        if not isinstance(e, Edge):
            raise MechError(f"edges(): {e!r} is not an edge `writer >> reader`")


def _variable(fn, what: str) -> str:
    """The name of a variable: a function the program defines with `def`."""
    if _PROGRAM is None:
        raise MechError(f"{what}() runs inside a program")
    if not isinstance(fn, types.FunctionType) or fn.__name__ == "<lambda>" or _PROGRAM.namespace.get(fn.__name__) is not fn:
        raise MechError(f"{what}(): the first argument is a variable, a function the program defines with def "
                        f"(its name is the variable's name), not {fn!r}")
    return fn.__name__


def bind(variable, *parts) -> None:
    """The variable `variable` (a function of the algorithm) is held by what `parts` write into the
    residual stream."""
    name = _variable(variable, "bind")
    _PROGRAM.bound.setdefault(name, []).extend(_pieces(parts, f"bind({name}, ...)"))


def claim(pattern, *parts) -> None:
    """The attention of `parts` (q_proj and k_proj parts, or native heads, in each of their layers)
    follows the pattern variable `pattern`: at query t, the positions its value lists (uniformly) or
    weighs."""
    name = _variable(pattern, "claim")
    _PROGRAM.claimed.setdefault(name, []).extend(_pieces(parts, f"claim({name}, ...)"))


# ---------------------------------------------------------------------------------------------------
# The algorithm

class _Algorithm:
    """A program's variables: functions of `tokens` and of each other, evaluated on token sequences."""

    def __init__(self, namespace: dict, roots):
        self.functions: dict[str, types.FunctionType] = {}
        self.params: dict[str, tuple[str, ...]] = {}
        todo = list(roots)
        while todo:
            name = todo.pop()
            if name in self.functions:
                continue
            fn = self.functions[name] = namespace[name]
            params = []
            for p in inspect.signature(fn).parameters.values():
                if p.kind is not p.POSITIONAL_OR_KEYWORD or p.default is not p.empty:
                    raise MechError(f"variable {name}: parameter {p.name} is not a plain parameter (a variable "
                                    "takes tokens and other variables by name)")
                if p.name != "tokens":
                    if not isinstance(namespace.get(p.name), types.FunctionType) or namespace[p.name].__name__ != p.name:
                        raise MechError(f"variable {name}: parameter {p.name} names no variable (a variable takes "
                                        "tokens and other variables, functions of the program, by name)")
                    todo.append(p.name)
                params.append(p.name)
            self.params[name] = tuple(params)
        self.readers = {n: [m for m in self.params if n in self.params[m]] for n in self.params}
        for name in self.params:  # rejects cycles
            self.upstream(name)

    def upstream(self, name: str, stack: tuple = ()) -> set:
        """Every variable `name` depends on, and "tokens" if it reads them."""
        if name in stack:
            raise MechError("variables " + " -> ".join(stack[stack.index(name):] + (name,)) + " read each other")
        out = set()
        for p in self.params[name]:
            out.add(p)
            if p != "tokens":
                out |= self.upstream(p, stack + (name,))
        return out

    def sources(self, name: str, held) -> set:
        """What `name` reads through steps that no part holds: variables in `held` and "tokens"."""
        out = set()
        for p in self.params[name]:
            out |= {p} if p == "tokens" or p in held else self.sources(p, held)
        return out

    def values(self, tokens: list, wanted, fixed: dict | None = None) -> dict:
        """The values of the variables `wanted` on `tokens` ({name: list}); `fixed` sets variables'
        values instead of computing them."""
        values = dict(fixed or {})

        def value(name):
            if name not in values:
                args = [list(tokens) if p == "tokens" else list(value(p)) for p in self.params[name]]
                try:
                    out = self.functions[name](*args)
                except MechError:
                    raise
                except RecursionError:
                    raise
                except Exception as e:
                    line = _line_of(e)
                    raise MechError((f"line {line}: " if line else "") + f"variable {name}: {type(e).__name__}: {e}") from None
                if not isinstance(out, (list, tuple)) or len(out) != len(tokens):
                    raise MechError(f"variable {name} returned {len(out) if isinstance(out, (list, tuple)) else repr(out)} "
                                    f"values for {len(tokens)} positions (a variable returns one value per position)")
                values[name] = list(out)
            return values[name]

        return {name: value(name) for name in wanted}


def _pattern_row(value, t: int, what: str) -> list[float]:
    """One query position's claimed attention (positions, or {position: weight}) as weights 0..t."""
    row = [0.0] * (t + 1)
    items = value.items() if isinstance(value, dict) else [(j, 1.0) for j in (value or [])]
    for j, w in items:
        if not isinstance(j, int) or isinstance(j, bool) or not 0 <= j <= t:
            raise MechError(f"{what} at position {t}: {j!r} is not a position 0..{t}")
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not w >= 0:
            raise MechError(f"{what} at position {t}: weight {w!r} is not a number >= 0")
        row[j] += float(w)
    if not sum(row):
        row[0] = 1.0  # no position: the head rests on position 0
    return row


def _evaluate(program: _Program, algorithm: _Algorithm, answer: str, behavior: dict, ir: dict) -> None:
    """The algorithm on the behavior: interchange pairs per bound variable, claimed patterns, and how
    often the answer is the prompt's next token."""
    prompts, targets = behavior["prompts"], behavior["targets"]
    cache: dict[tuple, dict] = {}

    def at(i: int, t: int, name: str) -> list:  # a variable on prompt i cut after position t
        key = (i, t)
        if name not in cache.setdefault(key, {}):
            cache[key].update(algorithm.values(prompts[i][: t + 1], [name], cache[key]))
        return cache[key][name]

    ir["clean"] = []  # per target: the answer on the clean prompt, and the prompt's next token
    for i, ts in enumerate(targets):
        for t in ts:
            a = at(i, t, answer)[t]
            if a is not None and not isinstance(a, str):
                raise MechError(f"the answer {answer} at position {t} of prompt {i} is {a!r}, not a token string")
            ir["clean"].append([i, t, a, prompts[i][t + 1] if t + 1 < len(prompts[i]) else None])

    def swapped(i: int, j: int, v: str) -> list | None:  # the answers at i's targets with v from prompt j
        out = []
        for t in targets[i]:
            a = algorithm.values(prompts[i][: t + 1], [answer], {v: at(j, t, v)})[answer][t]
            if a is None:
                return None
            if not isinstance(a, str):
                raise MechError(f"the answer {answer} at position {t} of prompt {i} is {a!r}, not a token string")
            out.append(a)
        return out

    n = len(prompts)
    for b in ir["bindings"]:
        v = b["variable"]
        for i, ts in enumerate(targets):
            # the source: the first later prompt of i's length (cyclically) under which the answer is
            # defined and changes, else the first under which it is defined
            clean, first = [at(i, t, answer)[t] for t in ts], None
            for k in range(i + 1, i + n):
                j = k % n
                if not ts or len(prompts[j]) != len(prompts[i]):
                    continue
                answers = swapped(i, j, v)
                if answers is not None:
                    first = first or (j, answers)
                    if answers != clean:
                        first = (j, answers)
                        break
            if first:
                b["pairs"].append({"base": i, "source": first[0], "answer_text": first[1]})
    for n in program.nodes:
        if n.claim is None:
            continue
        name, rows = n.claim["variable"], {}
        for side in ("prompts", "counterfactuals"):
            rows[side] = []
            for tokens in behavior.get(side) or []:
                value = algorithm.values(tokens, [name])[name]
                rows[side].append([_pattern_row(value[t], t, f"claimed pattern {name}") for t in range(len(tokens))])
        n.claim = {"op": "pattern", "variable": name, **rows}


def _validate(program: _Program, namespace: dict, ir: dict, behavior: dict | None) -> None:
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
    bound, claimed = program.bound, program.claimed
    if set(bound) & set(claimed):
        raise MechError(f"{', '.join(sorted(set(bound) & set(claimed)))}: a variable is bound (a value parts write) "
                        "or claimed (an attention pattern), not both")
    algorithm = _Algorithm(namespace, list(bound) + list(claimed)) if bound or claimed else None
    taken = {n.id for n in program.nodes}
    held: dict[str, list[Node]] = {}
    for name, pieces in bound.items():  # one node per layer's attention or MLP
        sites = sorted({(p.layer, p.block()) for p in pieces})
        for layer, block in sites:
            made = Node(_merged(p for p in pieces if (p.layer, p.block()) == (layer, block)))
            made.id = name if len(sites) == 1 else f"{name}.{layer}.{block}"
            if not made.writes_residual():
                raise MechError(f"bind({name}, ...): its parts in layer {layer}'s {block} write no residual stream "
                                f"(q/k/v_proj and c_fc parts write their own site's stream); bind the "
                                f"{'o_proj' if block == 'attn' else 'down_proj'} parts that carry the variable, "
                                f"or claim the attention pattern")
            held.setdefault(name, []).append(made)
    owner = {}
    for n in [m for ms in held.values() for m in ms]:
        for p in n.pieces:
            for i in (("rest",) if p.rest else p.index):
                owner[(p.view, p.layer, p.kind, i)] = n
    for name, pieces in claimed.items():  # per layer, the node of the claimed parts
        if any(p.block() != "attn" for p in pieces):
            raise MechError(f"claim({name}, ...): a claim is about attention parts")
        layers = sorted({p.layer for p in pieces})
        for layer in layers:
            group = [p for p in pieces if p.layer == layer]
            if not any(p.kind in ("q_proj", "k_proj", "head", "attn") for p in group):
                raise MechError(f"claim({name}, ...): name layer {layer}'s q_proj and k_proj parts (or heads) whose "
                                "queries and keys produce the pattern")
            homes = {owner.get((p.view, p.layer, p.kind, i)) for p in group for i in (("rest",) if p.rest else p.index)}
            if len(homes) > 1:
                raise MechError(f"claim({name}, ...): layer {layer}'s claimed parts are bound to a variable in part; "
                                "claim parts that are all bound to one variable in a layer, or none bound")
            home = homes.pop()
            if home is None:
                home = Node(_merged(group))
                home.id = name if len(layers) == 1 else f"{name}.{layer}.attn"
            elif home.claim is not None:
                raise MechError(f"claim({name}, ...): node {home.id} already carries the claim {home.claim['variable']}")
            home.claim = {"variable": name}
            held.setdefault(name, []).append(home)
    for name, ns in held.items():
        for n in ns:
            if n not in program.nodes:
                if n.id in taken:
                    raise MechError(f"node {n.id} of variable {name} is also a node the program names; rename one")
                program.nodes.append(n)
    owner = {}
    for n in program.nodes:
        for p in n.pieces:
            if p.view == "library":
                continue  # library parts may overlap; the checker takes the union per node
            for i in (("rest",) if p.rest else p.index):
                o = owner.setdefault((p.view, p.layer, p.kind, i), n.id)
                if o != n.id:
                    raise MechError(f"{Piece(p.view, p.layer, p.kind, (i,), rest=p.rest).name()} is in nodes {o} and "
                                    f"{n.id}; a part belongs to one node")
    ir["bindings"], ir["variables"], ir["answer"] = [], [], None
    if algorithm is not None:
        sinks = [v for v in bound if not algorithm.readers[v]]
        if len(sinks) != 1:
            raise MechError("the answer is the one bound variable no variable reads; " +
                            (f"{', '.join(sinks)} are all unread" if sinks else "bind the answer's parts"))
        answer = ir["answer"] = sinks[0]

        def connect(src, dst) -> bool:
            if src is dst or not _connects(src, dst, "input"):
                return False
            program.edges.setdefault((id(src), id(dst), "input"), Edge(src, dst, "input"))
            return True

        line = {v: algorithm.functions[v].__code__.co_firstlineno for v in algorithm.functions}
        for name, ns in held.items():
            for s in sorted(algorithm.sources(name, held), key=lambda s: (s != "tokens", line.get(s, 0))):
                writers = [embed] if s == "tokens" else held[s]
                if not any([w is n or connect(w, n) for w in writers for n in ns]):  # every pair connected
                    raise MechError(f"variable {name} reads {s}, but no part of " + ("the token embedding" if s == "tokens" else s) +
                                    f" writes where a part of {name} reads it later")
            for a in ns:  # a variable spanning layers feeds its own later parts
                for b in ns:
                    if a.site() < b.site():
                        connect(a, b)
        for n in held[answer]:
            connect(n, logits)
        connect(embed, logits)
        ir["variables"] = [{"name": v, "reads": list(algorithm.params[v]),
                            "role": "bound" if v in bound else "claimed" if v in claimed else "step",
                            "nodes": [n.id for n in held.get(v, [])],
                            "pieces": [p.ir() for p in _merged(bound.get(v) or claimed.get(v) or [])]}
                           for v in sorted(algorithm.params, key=line.get)]
        ir["bindings"] = [{"variable": v, "nodes": [n.id for n in held[v]], "pairs": []} for v in bound]
        if behavior is not None:
            _evaluate(program, algorithm, answer, behavior, ir)
    ir["nodes"] = [{"id": n.id, "pieces": [p.ir() for p in n.pieces],
                    "claim": n.claim if n.claim and "op" in n.claim else None} for n in program.nodes]
    ir["claims"] = {n.id: n.claim["variable"] for n in program.nodes if n.claim}
    ir["edges"] = [{"from": "embed" if e.src is embed else e.src.id,
                    "to": "logits" if e.dst is logits else e.dst.id, "route": e.route}
                   for e in program.edges.values()]


# ---------------------------------------------------------------------------------------------------
# Sandbox and tracer

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Attribute, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Call, ast.keyword, ast.Starred, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context, ast.FunctionDef, ast.Lambda,
    ast.arguments, ast.arg, ast.Return, ast.Pass, ast.For, ast.While, ast.If, ast.Break, ast.Continue,
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


def quote_parts(source: str) -> str:
    """`source` with each bare part token (<p:2.v.559>) outside strings and comments written as a string
    literal ("<p:2.v.559>"), so that it parses as Python."""
    if "<p:" not in source:
        return source
    try:
        skip = [(t.start, t.end) for t in tokenize.generate_tokens(io.StringIO(source).readline)
                if t.type in (tokenize.STRING, tokenize.COMMENT) or "FSTRING" in tokenize.tok_name[t.type]]
    except (tokenize.TokenError, SyntaxError, IndentationError):
        skip = []
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    spans = [(starts[a[0] - 1] + a[1], starts[b[0] - 1] + b[1]) for a, b in skip]
    out, last = [], 0
    for m in PART.finditer(source):
        if any(a <= m.start() < b for a, b in spans):
            continue
        out += [source[last:m.start()], f'"{m[0]}"']
        last = m.end()
    return "".join(out) + source[last:]


def _empty(source: str, model: str, decomposition: str | None) -> dict:
    return {"model": model, "decomposition": decomposition, "nodes": [], "edges": [], "bindings": [],
            "variables": [], "answer": None, "claims": {}, "python_tokens": 0, "token_types": 0,
            "source": source, "valid": False, "error": None}


def _trace(source: str, model: str, behavior: dict | None = None, decomposition: str | None = None) -> dict:
    """Checks and runs `source` in this process -> IR, with answers as token strings (`behavior`: the
    prompts, counterfactuals and targets as token strings, behavior_tokens()). The sandboxed child's
    entry point."""
    global _PROGRAM
    decomposition = DEFAULT_DECOMPOSITION.get(model) if decomposition is None else decomposition
    decomposition = None if decomposition == "native" else decomposition
    ir = _empty(source, model, decomposition)
    quoted = quote_parts(source)
    try:
        ir["python_tokens"], ir["token_types"] = code_length(quoted)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        pass
    try:
        if model not in MODELS:
            raise MechError(f"unknown model {model!r}; models: {', '.join(MODELS)}")
        if decomposition is not None and decomposition not in DECOMPOSITIONS:
            raise MechError(f"unknown decomposition {decomposition!r}: {', '.join(DECOMPOSITIONS)} or native")
        if len(source) > MAX_SOURCE:
            raise MechError(f"program longer than {MAX_SOURCE} characters")
        tree = ast.parse(quoted, "<program>")
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
        _PROGRAM = program = _Program(model, decomposition, namespace)
        try:
            exec(compile(tree, "<program>", "exec"), namespace)
            _validate(program, namespace, ir, behavior)
        finally:
            _PROGRAM = None
        ir["valid"] = True
    except MechError as e:
        line = _line_of(e)
        ir["error"] = (f"line {line}: " if line and not str(e).startswith("line ") else "") + str(e)
    except SyntaxError as e:
        ir["error"] = f"line {e.lineno}: syntax error: {e.msg}"
    except RecursionError:
        ir["error"] = "recursion too deep"
    except Exception as e:
        line = _line_of(e)
        ir["error"] = (f"line {line}: " if line else "") + f"{type(e).__name__}: {e}"
    if not ir["valid"]:
        ir.update(nodes=[], edges=[], bindings=[], variables=[], answer=None, claims={})
    return ir


_TOKENIZERS: dict = {}


def tokenizer(model: str):
    """The target's tokenizer (the `tokenizers` library): vpd4l's file, or Qwen3's tokenizer.json (every
    size shares it) through the Hugging Face cache (downloaded when missing)."""
    if model not in _TOKENIZERS:
        import tokenizers

        if model == "vpd4l":
            _TOKENIZERS[model] = tokenizers.Tokenizer.from_file(str(VPD4L_TOKENIZER))
        else:
            from huggingface_hub import hf_hub_download

            _TOKENIZERS[model] = tokenizers.Tokenizer.from_file(hf_hub_download(QWEN3[model], "tokenizer.json"))
    return _TOKENIZERS[model]


_VOCABULARIES: dict = {}


def _token_id(model: str, text: str, known: dict) -> int | None:
    """The token an answer `text` starts with: the token it is (one of the behavior's own, else the lowest
    id of M's vocabulary that decodes to it), else the first token M's tokenizer splits it into (an answer
    may spell out a longer continuation)."""
    if text in known:
        return known[text]
    if model not in _VOCABULARIES:
        tk = tokenizer(model)
        strings = tk.decode_batch([[i] for i in range(tk.get_vocab_size())], skip_special_tokens=False)
        vocabulary: dict[str, int] = {}
        for i, s in enumerate(strings):
            vocabulary.setdefault(s, i)
        _VOCABULARIES[model] = vocabulary
    if text in _VOCABULARIES[model]:
        return _VOCABULARIES[model][text]
    ids = tokenizer(model).encode(text, add_special_tokens=False).ids if text else []
    return ids[0] if ids else None


def behavior_tokens(behavior, model: str) -> tuple[dict, dict]:
    """(what the algorithm runs on: {"prompts", "counterfactuals", "targets"}, each token as M's string;
    {token string: id} of the behavior's tokens). `behavior`: a behavior record or its file."""
    if not isinstance(behavior, dict):
        behavior = json.loads(Path(behavior).expanduser().read_text())
    tk = tokenizer(model)
    prompts = [p["token_ids"] for p in behavior["prompts"]]
    counterfactuals = [p["counterfactual"]["token_ids"] for p in behavior["prompts"] if p.get("counterfactual")]
    ids = sorted({i for row in prompts + counterfactuals for i in row})
    strings = dict(zip(ids, tk.decode_batch([[i] for i in ids], skip_special_tokens=True)))  # BOS: ""
    known: dict[str, int] = {}
    for i in ids:
        known.setdefault(strings[i], i)
    payload = {"prompts": [[strings[i] for i in row] for row in prompts],
               "counterfactuals": [[strings[i] for i in row] for row in counterfactuals],
               "targets": [list(p["target_positions"]) for p in behavior["prompts"]]}
    if len(payload["counterfactuals"]) != len(prompts):
        payload["counterfactuals"] = []
    return payload, known


def _answer_ids(ir: dict, model: str, known: dict | None) -> dict:
    """Adds each interchange answer's token id (the checker's "answer"; a string that starts with no token
    makes the program invalid) and the algorithm's accuracy: the share of targets whose clean answer starts
    with the prompt's next token."""
    clean = ir.pop("clean", None)
    if clean is not None:
        hits = [a is not None and n is not None and _token_id(model, a, known or {}) == _token_id(model, n, known or {})
                for _, _, a, n in clean]
        ir["algorithm_accuracy"] = sum(hits) / len(hits) if hits else None
    for b in ir.get("bindings") or []:
        for pair in b["pairs"]:
            pair["answer"] = []
            for text in pair["answer_text"]:
                i = _token_id(model, text, known or {})
                if i is None:
                    ir.update(nodes=[], edges=[], bindings=[], variables=[], answer=None, claims={}, valid=False,
                              error=f"the answer {text!r} (variable {ir.get('answer')}, interchanging {b['variable']}, "
                                    f"prompt {pair['base']}) starts with no token of {model}")
                    return ir
                pair["answer"].append(i)
    return ir


def trace_inline(source: str, model: str, behavior=None, decomposition: str | None = None) -> dict:
    """Checks and runs `source` in this process (trusted sources only; `trace` sandboxes) -> IR."""
    payload, known = behavior_tokens(behavior, model) if behavior is not None else (None, None)
    return _answer_ids(_trace(source, model, payload, decomposition), model, known)


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


def _invalid(source: str, model: str, decomposition: str | None, error: str) -> dict:
    ir = _empty(source, model, DEFAULT_DECOMPOSITION.get(model) if decomposition is None else decomposition)
    try:
        ir["python_tokens"], ir["token_types"] = code_length(quote_parts(source))
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        pass
    ir["error"] = error
    return ir


def _traced_child(req: dict) -> dict:
    """Forks a child that traces the request's program under CPU/memory limits; waits for it with a
    wall-clock deadline, watching its footprint on macOS (RLIMIT_AS is not enforced there)."""
    import select
    import time

    source, model, timeout = req["source"], req["model"], req.get("timeout", 10.0)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(r)
            _limit(timeout)
            sys.setrecursionlimit(500)
            data = json.dumps(_trace(source, model, req.get("behavior"), req.get("decomposition"))).encode()
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
    decomposition = req.get("decomposition")
    if over or (os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL and not late):
        return _invalid(source, model, decomposition, f"memory limit of {MEMORY >> 20} MiB exceeded")
    if late or (os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGXCPU):
        return _invalid(source, model, decomposition, f"time limit of {timeout} s exceeded")
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        return _invalid(source, model, decomposition, f"the program crashed the tracer (status {status})")


def serve() -> None:
    """JSON lines on stdin {"source", "model", "timeout", "behavior", "decomposition"} -> IR lines on
    stdout, one forked child each."""
    for model in MODELS:
        shapes(model)
    for line in sys.stdin:
        print(json.dumps(_traced_child(json.loads(line))), flush=True)


_SERVERS = threading.local()


def trace(source: str, model: str, timeout: float = 10.0, behavior=None, decomposition: str | None = None) -> dict:
    """Checks and runs `source` sandboxed (restricted names and builtins; a forked child with CPU and
    memory limits and a wall-clock deadline) -> IR dict. `behavior` (a record or its file): evaluate the
    algorithm's interchange pairs and claims on its prompts; `decomposition`: what PD names ("vpd",
    "library", "transcoder", or "native" for none; the model's default when None). Each thread keeps one
    tracer server (`mech.py serve`, started with -I -S: no site hooks, the standard library only, no
    memory-ledger reservation), so a trace costs a fork, a few milliseconds."""
    payload, known = behavior_tokens(behavior, model) if behavior is not None else (None, None)
    request = json.dumps({"source": source, "model": model, "timeout": timeout, "behavior": payload,
                          "decomposition": decomposition})
    for attempt in range(2):
        server = getattr(_SERVERS, "proc", None)
        if server is None or server.poll() is not None or _SERVERS.pid != os.getpid():  # a forked caller starts its own
            _SERVERS.pid = os.getpid()
            server = _SERVERS.proc = subprocess.Popen(
                [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                env={**os.environ, "MPD_MEM_GIB": "1"})
        try:
            server.stdin.write(request + "\n")
            server.stdin.flush()
            line = server.stdout.readline()
            if line:
                return _answer_ids(json.loads(line), model, known)
        except (BrokenPipeError, OSError, ValueError):
            pass
        server.kill()
        _SERVERS.proc = None
    return _invalid(source, model, decomposition, "the tracer server failed")


def trace_many(sources: list[str], model: str, timeout: float = 10.0, workers: int = 8, behavior=None,
               decomposition: str | None = None) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    if behavior is not None and not isinstance(behavior, dict):
        behavior = json.loads(Path(behavior).expanduser().read_text())
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda s: trace(s, model, timeout, behavior, decomposition), sources))


# ---------------------------------------------------------------------------------------------------
# Code length and English

KEYWORDS = ("False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
            "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
            "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield")
OPERATORS = ("!", "!=", "%", "%=", "&", "&=", "(", ")", "*", "**", "**=", "*=", "+", "+=", ",", "-", "-=",
             "->", ".", "...", "/", "//", "//=", "/=", ":", ":=", ";", "<", "<<", "<<=", "<=", "=", "==", ">",
             ">=", ">>", ">>=", "@", "@=", "[", "]", "^", "^=", "{", "|", "|=", "}", "~")
LITERAL_CHARACTERS = {chr(c) for c in range(32, 127)} | {"\t", "\n"}
FIXED_NAMES = set(EXPORTS) | set(ATTRIBUTES) | set(SAFE_BUILTINS) | {"mech", "tokens"}
SKIPPED = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
           tokenize.ENCODING, tokenize.ENDMARKER}


def code_length(source: str) -> tuple[int, int]:
    """(python_tokens, token_types) of a source whose part tokens are quoted (quote_parts). Tokens of the
    code without comments and docstrings: a name, keyword, operator or part token is one token, a number
    or string literal one token per character as written. token_types = keywords + operators + mech and
    builtin names + names the program defines + part tokens + literal characters (printable ASCII, tab,
    newline and any other character the literals use)."""
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
        if t.type == tokenize.STRING and PART.fullmatch(t.string[1:-1]) and t.string[0] == t.string[-1]:
            tokens += 1
            names.add(t.string[1:-1])
            continue
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
    source = quote_parts(source)
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
    t.add_argument("--decomposition", help="vpd, library, transcoder or native (default: the model's)")
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
        print(json.dumps(_trace(source, a.model, None, a.decomposition)))
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
