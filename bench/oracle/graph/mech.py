"""The loader of graph-oracle explanations (#2951, format v4): plain Python, no imports.

An explanation is a causal graph of the target model's VPD subcomponents: named nodes, each a set of subcomponents
acting at some positions, and edges saying which node reads which. The model's own weights do every computation.
Closing quotes on vpd4l (one behavior variable, inside: whether a quotation is open):

    def quote_marks(tokens):
        return [t == '"' for t in tokens]

    nodes = {
        "mark":  {"subcomponents": ["<p:0.fc.225>", "<p:0.down.663>"], "at": quote_marks},
        "carry": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>"], "at": "targets"},
        "close": {"subcomponents": ["<p:3.fc.1013>", "<p:3.down.885>"], "at": "targets"},
    }
    edges = [("input", "mark"), ("mark", "carry", "value"), ("input", "carry", "query"), ("input", "carry", "key"),
             ("carry", "close"), ("close", "output")]
    labels = {"carry": "inside"}

  nodes   name -> {"subcomponents": [...], "at": where}. A subcomponent is "<p:L.S.I>": layer L, site S (q k v o fc
          down: q_proj k_proj v_proj o_proj c_fc down_proj), subcomponent I; "<p:L.S.rest>" is the site's remainder
          W - sum of its subcomponents. A subcomponent belongs to one node. "at" (default "all"): "all", "targets"
          (the positions whose next token the behavior asks for), "last", or a function of the file taking `tokens`
          (the sequence as the model's token strings) and returning a list of bools or of positions. A node acts
          only there; elsewhere its subcomponents carry their values on the changed prompt, and so does what a
          reader at another position reads from them through attention.
  edges   (writer, reader) or (writer, reader, route): the writer a node or "input" (the token embedding), the
          reader a node or "output" (the next-token logits), the route "query", "key" or "value" for an attention
          reader (default: every input the reader has). A writer must write before the reader reads. A node
          spanning several blocks feeds its own later blocks.
  labels  optional: node -> the behavior variable it carries (each variable at most one node). A variable is
          tested on the prompts whose changed prompt changes it, by swapping the node's output from the changed
          prompt; a variable no node carries pays its whole signal.
Everything an explanation leaves out runs on the changed prompt (only what it names sees the prompt), and the
embedding always reaches the output.

trace(source, model, behavior=...) runs a source in a sandboxed child (restricted syntax and builtins, CPU and
memory limits) and returns the IR the checker reads: nodes (one per node and block, a block being one layer's
attention or MLP, with its positions per prompt and changed prompt), edges, alignments (one per behavior variable),
and python_tokens and token_types of the code beside the nodes, edges and labels statements (the structure, which
the checker prices). code_length and explanation_length count what the score charges.
"""

from __future__ import annotations

import argparse
import ast
import builtins
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES_FILE = HERE / "shapes.json"
QWEN3 = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3-8b": "Qwen/Qwen3-8B"}
VPD4L_TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
SITES = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "fc": "c_fc", "down": "down_proj"}
PART = re.compile(r"<p:(\d+)\.(q|k|v|o|fc|down)\.(\d+|rest)>")
ROUTES = ("query", "key", "value")
INPUT, OUTPUT = "input", "output"


class MechError(Exception):
    """An invalid explanation."""


_SHAPES: dict | None = None


def shapes(model: str) -> dict:
    """The model's registry entry (shapes.json): layers, sizes, and per layer its VPD subcomponents per site."""
    global _SHAPES
    if _SHAPES is None:
        _SHAPES = json.loads(SHAPES_FILE.read_text())
    if model not in _SHAPES or not (_SHAPES[model].get("views") or {}).get("vpd"):
        known = sorted(m for m, s in _SHAPES.items() if (s.get("views") or {}).get("vpd"))
        raise MechError(f"no VPD decomposition of {model!r}; models: {', '.join(known)}")
    return _SHAPES[model]


# ---------------------------------------------------------------------------------------------------
# Groups -> nodes and edges

class Node:
    """One group's subcomponents in one block (a layer's attention or MLP)."""

    def __init__(self, id: str, layer: int, block: str, parts: dict[str, set]):
        self.id, self.layer, self.block, self.parts = id, layer, block, parts  # parts: site -> indices ("rest" too)

    def reads(self) -> list[tuple]:
        """(stream, place, routes): the residual stream at place 2l (attention) / 2l + 1 (MLP), or a block's own
        stream (attention values "attn", MLP hidden "hidden") of layer l."""
        l, out = self.layer, []
        for site in self.parts:
            if site in ("q_proj", "k_proj", "v_proj"):
                out.append(("resid", 2 * l, ({"q_proj": "query", "k_proj": "key", "v_proj": "value"}[site], INPUT)))
            elif site == "o_proj":
                out.append(("attn", l, (INPUT,)))
            elif site == "c_fc":
                out.append(("resid", 2 * l + 1, (INPUT,)))
            else:
                out.append(("hidden", l, (INPUT,)))
        return out

    def writes(self) -> list[tuple]:
        l, out = self.layer, []
        for site in self.parts:
            if site in ("q_proj", "k_proj", "v_proj"):
                out.append(("attn", l))
            elif site == "o_proj":
                out.append(("resid", 2 * l))
            elif site == "c_fc":
                out.append(("hidden", l))
            else:
                out.append(("resid", 2 * l + 1))
        return out

    def ir(self) -> dict:
        pieces = []
        for site in sorted(self.parts, key=list(SITES.values()).index):
            index = self.parts[site]
            if "rest" in index:
                pieces.append({"view": "vpd", "layer": self.layer, "kind": site, "index": "rest"})
            numbers = sorted(i for i in index if i != "rest")
            if numbers:
                pieces.append({"view": "vpd", "layer": self.layer, "kind": site,
                               "index": numbers[0] if len(numbers) == 1 else numbers})
        return {"id": self.id, "pieces": pieces, "claim": None}


def _connects(writes: list[tuple], reads: list[tuple], route: str) -> bool:
    """Whether a write reaches a read later through `route`."""
    return any((ws == rs == "resid" and w < r or ws == rs != "resid" and w == r) and route in routes
               for ws, w in writes for rs, r, routes in reads)


def _part(token, group: str, shape: dict) -> tuple[int, str, object]:
    if not isinstance(token, str) or not PART.fullmatch(token):
        raise MechError(f"group {group}: {token!r} is not a subcomponent \"<p:L.S.I>\" (S one of q k v o fc down)")
    m = PART.fullmatch(token)
    layer, site = int(m[1]), SITES[m[2]]
    if layer >= shape["layers"]:
        raise MechError(f"group {group}: {token}: layers are 0..{shape['layers'] - 1}")
    if m[3] == "rest":
        return layer, site, "rest"
    size = shape["views"]["vpd"][layer][site]
    if int(m[3]) >= size:
        raise MechError(f"group {group}: {token}: layer {layer}'s {site} has subcomponents 0..{size - 1}")
    return layer, site, int(m[3])


def _where(at, name: str, namespace: dict):
    """A node's "at" as a function (tokens, targets) -> positions."""
    if at is None or at == "all":
        return lambda tokens, targets: range(len(tokens))
    if at == "targets":
        return lambda tokens, targets: targets
    if at == "last":
        return lambda tokens, targets: [len(tokens) - 1] if tokens else []
    if callable(at) and getattr(at, "__name__", "") in namespace and namespace[at.__name__] is at:
        def positions(tokens, targets):
            try:
                out = at(list(tokens))
            except MechError:
                raise
            except RecursionError:
                raise
            except Exception as e:
                line = _line_of(e)
                raise MechError((f"line {line}: " if line else "") + f"node {name}: at {at.__name__}: {type(e).__name__}: {e}") from None
            if not isinstance(out, (list, tuple)):
                raise MechError(f"node {name}: at {at.__name__} returned {out!r}, not a list of bools or positions")
            if out and all(isinstance(x, bool) for x in out):
                if len(out) != len(tokens):
                    raise MechError(f"node {name}: at {at.__name__} returned {len(out)} bools for {len(tokens)} tokens")
                return [t for t, on in enumerate(out) if on]
            if not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < len(tokens) for x in out):
                raise MechError(f"node {name}: at {at.__name__} returned {out!r}, not positions 0..{len(tokens) - 1}")
            return sorted(set(out))
        return positions
    raise MechError(f"node {name}: \"at\" is \"all\", \"targets\", \"last\" or a function the file defines with def")


def build(nodes_value, edges_value, labels_value, namespace: dict, model: str, behavior: dict | None = None) -> tuple[dict, dict[str, list[Node]], dict[str, str]]:
    """The IR's nodes (with their positions on the behavior's prompts and changed prompts) and edges, the IR nodes
    per explanation node, and the variable each labeled node carries (checked against the behavior's)."""
    shape = shapes(model)
    if not isinstance(nodes_value, dict) or not nodes_value:
        raise MechError("`nodes` must be a non-empty dict {name: {\"subcomponents\": [...], \"at\": ...}}")
    nodes: dict[str, list[Node]] = {}
    where: dict[str, object] = {}
    owner: dict[tuple, str] = {}
    for name, g in nodes_value.items():
        if not isinstance(name, str) or not name.isidentifier() or name in (INPUT, OUTPUT):
            raise MechError(f"node name {name!r}: an identifier other than input and output")
        if not isinstance(g, dict) or set(g) - {"subcomponents", "at"}:
            raise MechError(f"node {name}: a dict with keys subcomponents and at")
        parts = g.get("subcomponents")
        if not isinstance(parts, (list, tuple)) or not parts:
            raise MechError(f"node {name}: \"subcomponents\" must be a non-empty list")
        blocks: dict[tuple, dict[str, set]] = {}
        for token in parts:
            layer, site, index = _part(token, name, shape)
            o = owner.setdefault((layer, site, index), name)
            if o != name:
                raise MechError(f"{token} is in nodes {o} and {name}; a subcomponent belongs to one node")
            block = "mlp" if site in ("c_fc", "down_proj") else "attn"
            blocks.setdefault((layer, block), {}).setdefault(site, set()).add(index)
        keys = sorted(blocks)
        nodes[name] = [Node(name if len(keys) == 1 else f"{name}.{l}.{b}", l, b, blocks[(l, b)]) for l, b in keys]
        where[name] = None if g.get("at", "all") == "all" else _where(g.get("at"), name, namespace)
    edges: dict[tuple, dict] = {}

    def edge(src: str, writes: list[tuple], dst: Node | None, route: str) -> bool:
        reads = [("resid", 2 * shape["layers"], (INPUT,))] if dst is None else dst.reads()
        if not _connects(writes, reads, route):
            return False
        key = (src, "logits" if dst is None else dst.id, route)
        edges.setdefault(key, {"from": key[0], "to": key[1], "route": route})
        return True

    if not isinstance(edges_value, (list, tuple)) or not edges_value:
        raise MechError("`edges` must be a non-empty list of (writer, reader) or (writer, reader, route)")
    for e in edges_value:
        if not isinstance(e, (list, tuple)) or len(e) not in (2, 3) or not all(isinstance(x, str) for x in e):
            raise MechError(f"edge {e!r}: (writer, reader) or (writer, reader, route)")
        src, dst, route = e[0], e[1], (e[2] if len(e) == 3 else INPUT)
        if src != INPUT and src not in nodes or dst != OUTPUT and dst not in nodes or route not in ROUTES + (INPUT,):
            raise MechError(f"edge {e!r}: the writer a node or \"input\", the reader a node or \"output\", the route "
                            "\"query\", \"key\" or \"value\"")
        if src == dst:
            raise MechError(f"edge {e!r}: a node cannot read itself")
        writers = [("embed", [("resid", -1)])] if src == INPUT else [(n.id, n.writes()) for n in nodes[src]]
        readers = [None] if dst == OUTPUT else nodes[dst]
        if not any([edge(w, ws, n, route) for w, ws in writers for n in readers]):
            raise MechError(f"edge {e!r} connects nothing: no subcomponent of the writer writes where a subcomponent "
                            "of the reader reads it later (writers must come first; only o and down subcomponents "
                            "write the residual stream)")
    if not any(k[1] == "logits" and k[0] != "embed" for k in edges):
        raise MechError("no node writes the output; add an edge (node, \"output\")")
    for ns in nodes.values():  # a node spanning blocks feeds its own later subcomponents
        for a in ns:
            for b in ns:
                if (a.layer, a.block) < (b.layer, b.block):
                    edge(a.id, a.writes(), b, INPUT)
    edge("embed", [("resid", -1)], None, INPUT)  # the embedding is not decomposed: it always reaches the output
    labels: dict[str, str] = {}
    if labels_value is not None:
        if not isinstance(labels_value, dict):
            raise MechError("`labels` must be a dict {node: behavior variable}")
        variables = behavior["variables"] if behavior is not None else None
        for name, v in labels_value.items():
            if name not in nodes:
                raise MechError(f"labels: {name!r} is not a node")
            if not isinstance(v, str) or variables is not None and v not in variables:
                raise MechError(f"labels: {v!r} is not a variable of this behavior; its variables: " + (", ".join(variables or []) or "none"))
            if v in labels.values():
                raise MechError(f"labels: variable {v} is carried by two nodes")
            for n in nodes[name]:  # a label test swaps what the node writes into the residual stream
                if not any(w[0] == "resid" for w in n.writes()):
                    raise MechError(f"labels: node {name} carries {v}, but its subcomponents in layer {n.layer}'s "
                                    f"{'attention' if n.block == 'attn' else 'MLP'} include no "
                                    f"{'o' if n.block == 'attn' else 'down'} subcomponent, so they write nothing a swap can carry")
            labels[name] = v
    ir_nodes = []
    for name, ns in nodes.items():
        at = []
        if where[name] is not None and behavior is not None:
            for ids, strings, targets in behavior["sequences"]:
                at.append({"tokens": ids, "positions": list(where[name](strings, targets))})
        for n in ns:
            ir_nodes.append({**n.ir(), "at": at})
    return {"nodes": ir_nodes, "edges": list(edges.values())}, nodes, labels


# ---------------------------------------------------------------------------------------------------
# Variables on a behavior

def alignments(behavior: dict | None, nodes: dict[str, list[str]], labels: dict[str, str]) -> list[dict]:
    """One alignment per behavior variable: the IR nodes of the node carrying it (none when no node does) and the
    prompts whose changed prompt changes it, each paired with that changed prompt."""
    if behavior is None:
        return [{"variable": v, "nodes": nodes[g], "pairs": []} for g, v in labels.items()]
    carrier = {v: g for g, v in labels.items()}
    out = []
    for v in behavior["variables"]:
        pairs = [{"base": i, "changed": True} for i, (varies, tokens, changed, targets) in
                 enumerate(zip(behavior["varies"], behavior["prompts"], behavior["counterfactuals"], behavior["targets"]))
                 if v in varies and targets and len(changed) == len(tokens)]
        if pairs:
            out.append({"variable": v, "nodes": nodes[carrier[v]] if v in carrier else [], "pairs": pairs})
    return out


# ---------------------------------------------------------------------------------------------------
# Sandbox and tracer

ALLOWED = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Attribute, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Call, ast.keyword, ast.Starred, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context, ast.FunctionDef, ast.Lambda,
    ast.arguments, ast.arg, ast.Return, ast.Pass, ast.For, ast.While, ast.If, ast.Break, ast.Continue,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.JoinedStr,
    ast.FormattedValue, ast.Assert, ast.NamedExpr,
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


def check(tree: ast.AST) -> None:
    """Raises MechError on a construct an explanation may not use (imports, dunder names, decorators, ...)."""
    docs = {id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    for n in ast.walk(tree):
        line = getattr(n, "lineno", "?")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            raise MechError(f"line {line}: an explanation imports nothing")
        if not isinstance(n, ALLOWED):
            raise MechError(f"line {line}: {type(n).__name__} is not allowed")
        if isinstance(n, ast.Attribute) and (n.attr.startswith("_") or n.attr in BANNED_ATTRIBUTES):
            raise MechError(f"line {line}: attribute {n.attr!r} is not allowed")
        if isinstance(n, ast.Name) and n.id.startswith("__"):
            raise MechError(f"line {line}: name {n.id!r} is not allowed")
        if isinstance(n, (ast.FunctionDef, ast.Lambda)) and getattr(n, "decorator_list", None):
            raise MechError(f"line {line}: decorators are not allowed")
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and "__" in n.value:
            raise MechError(f"line {line}: strings containing '__' are not allowed outside docstrings")


def _line_of(exc: BaseException) -> int | None:
    lines = [f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<explanation>"]
    return lines[-1] if lines else None


def _empty(source: str, model: str) -> dict:
    return {"model": model, "decomposition": "vpd", "standin": "counterfactual", "nodes": [], "edges": [], "alignments": [],
            "groups": [], "node_ids": {}, "labels": {}, "python_tokens": 0, "token_types": 0, "source": source,
            "valid": False, "error": None}


def _trace(source: str, model: str, behavior: dict | None = None) -> dict:
    """Checks and runs `source` in this process -> IR (`behavior`: behavior_tokens()'s payload, for the label
    pairs). The sandboxed child's entry point."""
    ir = _empty(source, model)
    try:
        ir["python_tokens"], ir["token_types"] = code_length(source)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        pass
    try:
        if len(source) > MAX_SOURCE:
            raise MechError(f"explanation longer than {MAX_SOURCE} characters")
        tree = ast.parse(source, "<explanation>")
        check(tree)
        namespace = {"__builtins__": SAFE_BUILTINS, "__name__": "explanation"}
        exec(compile(tree, "<explanation>", "exec"), namespace)
        if "nodes" not in namespace or "edges" not in namespace:
            raise MechError("the explanation defines no `nodes` dict and `edges` list")
        built, nodes, labels = build(namespace["nodes"], namespace["edges"], namespace.get("labels"), namespace, model, behavior)
        ir.update(built)
        ir["node_ids"] = {g: [n.id for n in ns] for g, ns in nodes.items()}
        ir["labels"] = labels
        ir["alignments"] = alignments(behavior, ir["node_ids"], labels)
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
        ir.update(nodes=[], edges=[], alignments=[], node_ids={}, labels={})
    return ir


_TOKENIZERS: dict = {}


def tokenizer(model: str):
    """The target's tokenizer (the `tokenizers` library): vpd4l's file, or Qwen3's tokenizer.json through the
    Hugging Face cache."""
    if model not in _TOKENIZERS:
        import tokenizers

        if model == "vpd4l":
            _TOKENIZERS[model] = tokenizers.Tokenizer.from_file(str(VPD4L_TOKENIZER))
        else:
            from huggingface_hub import hf_hub_download

            _TOKENIZERS[model] = tokenizers.Tokenizer.from_file(hf_hub_download(QWEN3[model], "tokenizer.json"))
    return _TOKENIZERS[model]


def behavior_tokens(behavior, model: str) -> dict:
    """What the tracer needs of a behavior: {"prompts", "counterfactuals", "targets", "varies", "variables",
    "sequences"}, tokens as the model's strings (a prompt without a changed prompt gets an empty one), each
    prompt's variables its changed prompt changes, the behavior's variables, and every prompt and changed prompt as
    (token ids, token strings, target positions) for node positions. `behavior`: a behavior record or its file."""
    if not isinstance(behavior, dict):
        behavior = json.loads(Path(behavior).expanduser().read_text())
    tk = tokenizer(model)
    prompts = [p["token_ids"] for p in behavior["prompts"]]
    changed = [(p.get("counterfactual") or {}).get("token_ids") or [] for p in behavior["prompts"]]
    ids = sorted({i for row in prompts + changed for i in row})
    strings = dict(zip(ids, tk.decode_batch([[i] for i in ids], skip_special_tokens=True)))  # BOS: ""
    varies = [[v for v in p.get("varies") or [] if v != "tokens"] for p in behavior["prompts"]]
    targets = [list(p["target_positions"]) for p in behavior["prompts"]]
    sequences = [(row, [strings[i] for i in row], t) for rows in (prompts, changed) for row, t in zip(rows, targets) if row]
    return {"prompts": [[strings[i] for i in row] for row in prompts],
            "counterfactuals": [[strings[i] for i in row] for row in changed],
            "targets": targets, "varies": varies, "variables": sorted({v for row in varies for v in row}),
            "sequences": sequences}


def trace_inline(source: str, model: str, behavior=None) -> dict:
    """Checks and runs `source` in this process (trusted sources only; trace() sandboxes) -> IR."""
    return _trace(source, model, behavior_tokens(behavior, model) if behavior is not None else None)


MEMORY = 1 << 30  # bytes a traced explanation may use


def _limit(seconds: float) -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (int(seconds) + 1, int(seconds) + 2))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY + (1 << 30), MEMORY + (1 << 30)))
    except (ValueError, OSError):
        pass  # macOS does not enforce address-space limits; the parent watches the footprint instead


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
    ir = _empty(source, model)
    try:
        ir["python_tokens"], ir["token_types"] = code_length(source)
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        pass
    ir["error"] = error
    return ir


def _traced_child(req: dict) -> dict:
    """Forks a child that traces the request under CPU/memory limits; waits with a wall-clock deadline,
    watching its footprint on macOS (RLIMIT_AS is not enforced there)."""
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
            data = json.dumps(_trace(source, model, req.get("behavior"))).encode()
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
        return _invalid(source, model, f"the explanation crashed the tracer (status {status})")


def serve() -> None:
    """JSON lines on stdin {"source", "model", "timeout", "behavior"} -> IR lines on stdout, one forked child each."""
    shapes("vpd4l")
    for line in sys.stdin:
        print(json.dumps(_traced_child(json.loads(line))), flush=True)


_SERVERS = threading.local()


def trace(source: str, model: str, timeout: float = 10.0, behavior=None) -> dict:
    """Checks and runs `source` sandboxed -> IR. `behavior` (a record or its file): the label pairs on its
    prompts. Each thread keeps one tracer server (`mech.py serve`, started with -I -S), so a trace costs a fork."""
    payload = behavior_tokens(behavior, model) if behavior is not None else None
    request = json.dumps({"source": source, "model": model, "timeout": timeout, "behavior": payload})
    for _ in range(2):
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
                return json.loads(line)
        except (BrokenPipeError, OSError, ValueError):
            pass
        server.kill()
        _SERVERS.proc = None
    return _invalid(source, model, "the tracer server failed")


def trace_many(sources: list[str], model: str, timeout: float = 10.0, workers: int = 8, behavior=None) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    if behavior is not None and not isinstance(behavior, dict):
        behavior = json.loads(Path(behavior).expanduser().read_text())
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda s: trace(s, model, timeout, behavior), sources))


# ---------------------------------------------------------------------------------------------------
# What the score charges

KEYWORDS = ("False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
            "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
            "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield")
OPERATORS = ("!", "!=", "%", "%=", "&", "&=", "(", ")", "*", "**", "**=", "*=", "+", "+=", ",", "-", "-=",
             "->", ".", "...", "/", "//", "//=", "/=", ":", ":=", ";", "<", "<<", "<<=", "<=", "=", "==", ">",
             ">=", ">>", ">>=", "@", "@=", "[", "]", "^", "^=", "{", "|", "|=", "}", "~")
LITERAL_CHARACTERS = {chr(c) for c in range(32, 127)} | {"\t", "\n"}
STRUCTURE = ("nodes", "edges", "labels")  # the statements the checker prices as structure
FIXED_NAMES = set(SAFE_BUILTINS) | set(STRUCTURE) | {"tokens"}
SKIPPED = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
           tokenize.ENCODING, tokenize.ENDMARKER}


def code_length(source: str) -> tuple[int, int]:
    """(python_tokens, token_types) of the code a reader must read beyond the structure: every statement but
    the nodes, edges and labels assignments (the checker prices nodes, edges and labels), without comments and
    docstrings. A name, keyword or operator is one token, a number or string literal one token per character
    as written. token_types = keywords + operators + builtin names + names the file defines + literal
    characters."""
    tree = ast.parse(source)
    docs = [((n.lineno, n.col_offset), (n.end_lineno, n.end_col_offset)) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    structure = [(n.lineno, n.end_lineno) for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in STRUCTURE for t in n.targets)]
    tokens, names, characters = 0, set(), set(LITERAL_CHARACTERS)
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type in SKIPPED or not t.string or any(a <= t.start[0] <= b for a, b in structure):
            continue
        if t.type == tokenize.STRING and any(a <= t.start and t.end <= b for a, b in docs):
            continue
        if t.type in (tokenize.NUMBER, tokenize.STRING) or "STRING" in tokenize.tok_name[t.type]:
            tokens += len(t.string)
            characters.update(t.string)
            continue
        tokens += 1
        if t.type == tokenize.NAME and t.string not in KEYWORDS and t.string not in FIXED_NAMES:
            names.add(t.string)
    return tokens, len(KEYWORDS) + len(OPERATORS) + len(FIXED_NAMES) + len(names) + len(characters)


READER = "qwen3-0.6b"  # every Qwen3 size shares this tokenizer


def explanation_length(text: str) -> tuple[int, int]:
    """(tokens, token types) of the English explanation under the reader's tokenizer: its tokens, and the
    size of the reader's vocabulary (a uniform code over it costs log2 of that per token)."""
    tk = tokenizer(READER)
    return (len(tk.encode(text, add_special_tokens=False).ids) if text else 0), tk.get_vocab_size()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trace", help="stdin explanation -> IR JSON on stdout")
    t.add_argument("--model", default="vpd4l")
    t.add_argument("--behavior", type=Path, help="a behavior file: compute the label pairs on it")
    sub.add_parser("serve", help="the tracer server (JSON lines; trace() starts it)")
    a = ap.parse_args()
    if a.command == "trace":
        print(json.dumps(trace(sys.stdin.read(), a.model, behavior=a.behavior), indent=1))
    else:
        serve()


if __name__ == "__main__":
    main()
