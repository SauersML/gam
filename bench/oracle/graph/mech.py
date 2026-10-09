"""The loader of graph-oracle explanations (#2951, format v3): plain Python, no imports.

An explanation names groups of the target model's VPD subcomponents and which groups read which; the model's
own weights do every computation. Closing quotes on vpd4l, whose behavior has one variable, inside (whether a
quotation is open):

    groups = {
        "quote": {"subcomponents": ["<p:0.fc.225>", "<p:0.down.663>"], "reads": ["input"], "label": "inside"},
        "answer": {"subcomponents": ["<p:3.fc.1013>", "<p:3.down.885>"], "reads": ["quote"], "writes": "output"},
    }

`groups` maps a name to:
  subcomponents  VPD subcomponents, "<p:L.S.I>": layer L, site S (q k v o fc down: q_proj k_proj v_proj o_proj
                 c_fc down_proj), subcomponent I; "<p:L.S.rest>" is the site's remainder W - sum of its
                 subcomponents. A subcomponent belongs to one group.
  reads          "input" (the token embedding) and other groups, a group's attention read as "name:query",
                 "name:key" or "name:value". A read connects every pair of subcomponents where the writer
                 writes before the reader reads.
  writes         "output": the group writes the next-token logits.
  label          the behavior variable the group carries. Every group that does not write the output carries
                 one, a variable belongs to at most one group, and a group that writes the output carries none.
Everything an explanation leaves out runs on the prompt's changed prompt (the checker's default stand-ins).

A behavior's variables are what its changed prompts change (each prompt's "varies"; "tokens", the input itself,
is not one). A variable is tested on the prompts whose changed prompt changes it: the group's output on the
changed prompt is swapped into the model's run on the prompt, and the model's output there should become its
output on the changed prompt (the checker's alignment term). A variable no group carries costs its whole signal.

trace(source, model, behavior=...) runs a source in a sandboxed child (restricted syntax and builtins, CPU and
memory limits) and returns the IR the checker reads: nodes (one per group and block, a block being one layer's
attention or MLP), edges, alignments (one per behavior variable, with its prompt pairs), and python_tokens and
token_types of any code beside the `groups` statement (the structure, which the checker prices). code_length
and explanation_length count what the score charges.
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
KEYS = {"subcomponents", "reads", "writes", "label"}
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


def build(groups, model: str, variables: list[str] | None = None) -> tuple[dict, dict[str, list[Node]], dict[str, str]]:
    """The IR's nodes and edges of a `groups` value, its nodes per group, and the variable each labeled group
    carries (`variables`: the behavior's, checked when given)."""
    shape = shapes(model)
    if not isinstance(groups, dict) or not groups:
        raise MechError("`groups` must be a non-empty dict {name: {\"subcomponents\": [...], ...}}")
    nodes: dict[str, list[Node]] = {}
    labels: dict[str, str] = {}
    owner: dict[tuple, str] = {}
    for name, g in groups.items():
        if not isinstance(name, str) or not name.isidentifier() or name in (INPUT, OUTPUT):
            raise MechError(f"group name {name!r}: an identifier other than input and output")
        if not isinstance(g, dict):
            raise MechError(f"group {name}: a dict with keys {', '.join(sorted(KEYS))}")
        if set(g) - KEYS:
            raise MechError(f"group {name}: unknown keys {', '.join(sorted(map(str, set(g) - KEYS)))}; keys: {', '.join(sorted(KEYS))}")
        parts = g.get("subcomponents")
        if not isinstance(parts, (list, tuple)) or not parts:
            raise MechError(f"group {name}: \"subcomponents\" must be a non-empty list")
        blocks: dict[tuple, dict[str, set]] = {}
        for token in parts:
            layer, site, index = _part(token, name, shape)
            o = owner.setdefault((layer, site, index), name)
            if o != name:
                raise MechError(f"{token} is in groups {o} and {name}; a subcomponent belongs to one group")
            block = "mlp" if site in ("c_fc", "down_proj") else "attn"
            blocks.setdefault((layer, block), {}).setdefault(site, set()).add(index)
        keys = sorted(blocks)
        nodes[name] = [Node(name if len(keys) == 1 else f"{name}.{l}.{b}", l, b, blocks[(l, b)]) for l, b in keys]
        label = g.get("label")
        if label is not None:
            if not isinstance(label, str) or not label:
                raise MechError(f"group {name}: \"label\" is the name of the behavior variable it carries")
            if variables is not None and label not in variables:
                raise MechError(f"group {name}: {label!r} is not a variable of this behavior; its variables: "
                                + (", ".join(variables) or "none"))
            if label in labels.values():
                raise MechError(f"variable {label} is carried by two groups; a variable belongs to one group")
            for n in nodes[name]:  # a label test swaps what the group writes into the residual stream
                if not any(w[0] == "resid" for w in n.writes()):
                    raise MechError(f"group {name} carries {label}, but its subcomponents in layer {n.layer}'s "
                                    f"{'attention' if n.block == 'attn' else 'MLP'} include no "
                                    f"{'o' if n.block == 'attn' else 'down'} subcomponent, so they write nothing a "
                                    "swap can carry; add one or move them to another group")
            labels[name] = label
        writes = g.get("writes")
        if writes not in (None, OUTPUT):
            raise MechError(f"group {name}: \"writes\" can only be \"output\"")
        if writes is None and label is None:
            raise MechError(f"group {name} does not write the output, so it carries a behavior variable (\"label\")")
        if writes == OUTPUT and label is not None:
            raise MechError(f"group {name} writes the output, so it carries no variable; put the subcomponents "
                            f"that carry {label} in a group of their own")
    if not any(g.get("writes") == OUTPUT for g in groups.values()):
        raise MechError("no group writes the output; mark the group that writes the next-token logits with \"writes\": \"output\"")
    ids = {n.id for ns in nodes.values() for n in ns}
    if len(ids) != sum(map(len, nodes.values())):
        raise MechError("two groups make the same node name; rename one")
    edges: dict[tuple, dict] = {}

    def edge(src: str, writes: list[tuple], dst: Node | None, route: str) -> bool:
        reads = [("resid", 2 * shape["layers"], (INPUT,))] if dst is None else dst.reads()
        if not _connects(writes, reads, route):
            return False
        key = (src, "logits" if dst is None else dst.id, route)
        edges.setdefault(key, {"from": key[0], "to": key[1], "route": route})
        return True

    for name, g in groups.items():
        reads = g.get("reads") or []
        if not isinstance(reads, (list, tuple)):
            raise MechError(f"group {name}: \"reads\" must be a list")
        for r in reads:
            source, _, route = (r if isinstance(r, str) else "").partition(":")
            route = route or INPUT
            if source != INPUT and source not in groups or route not in ROUTES + (INPUT,) or source == INPUT and route != INPUT:
                raise MechError(f"group {name} reads {r!r}: \"input\", a group's name, or a group's \"name:query\", "
                                "\"name:key\" or \"name:value\"")
            if source == name:
                raise MechError(f"group {name} cannot read itself")
            writers = [("embed", [("resid", -1)])] if source == INPUT else [(n.id, n.writes()) for n in nodes[source]]
            if not any([edge(w, ws, n, route) for w, ws in writers for n in nodes[name]]):
                raise MechError(f"group {name} reads {r}, but no subcomponent of " +
                                ("the input" if source == INPUT else source) +
                                f" writes where a subcomponent of {name} reads it later (writers must come first)")
        ns = sorted(nodes[name], key=lambda n: (n.layer, n.block))
        for a in ns:  # a group spanning blocks feeds its own later subcomponents
            for b in ns:
                if (a.layer, a.block) < (b.layer, b.block):
                    edge(a.id, a.writes(), b, INPUT)
        if g.get("writes") == OUTPUT and not any([edge(n.id, n.writes(), None, INPUT) for n in ns]):
            raise MechError(f"group {name} writes the output, but none of its subcomponents writes the residual "
                            "stream (an o or down subcomponent)")
    edge("embed", [("resid", -1)], None, INPUT)
    ir = {"nodes": [n.ir() for ns in nodes.values() for n in ns], "edges": list(edges.values())}
    return ir, nodes, labels


# ---------------------------------------------------------------------------------------------------
# Variables on a behavior

def alignments(behavior: dict | None, nodes: dict[str, list[str]], labels: dict[str, str]) -> list[dict]:
    """One alignment per behavior variable: the nodes of the group carrying it (none when no group does) and the
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
    return {"model": model, "decomposition": "vpd", "nodes": [], "edges": [], "alignments": [], "groups": [],
            "group_nodes": {}, "labels": {}, "python_tokens": 0, "token_types": 0, "source": source,
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
        if "groups" not in namespace:
            raise MechError("the explanation defines no `groups` dict")
        built, nodes, labels = build(namespace["groups"], model, behavior["variables"] if behavior is not None else None)
        ir.update(built)
        ir["group_nodes"] = {g: [n.id for n in ns] for g, ns in nodes.items()}
        ir["labels"] = labels
        ir["alignments"] = alignments(behavior, ir["group_nodes"], labels)
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
        ir.update(nodes=[], edges=[], alignments=[], group_nodes={}, labels={})
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
    """What the tracer needs of a behavior: {"prompts", "counterfactuals", "targets", "varies", "variables"},
    tokens as the model's strings (a prompt without a changed prompt gets an empty one), each prompt's variables
    its changed prompt changes, and the behavior's variables. `behavior`: a behavior record or its file."""
    if not isinstance(behavior, dict):
        behavior = json.loads(Path(behavior).expanduser().read_text())
    tk = tokenizer(model)
    prompts = [p["token_ids"] for p in behavior["prompts"]]
    changed = [(p.get("counterfactual") or {}).get("token_ids") or [] for p in behavior["prompts"]]
    ids = sorted({i for row in prompts + changed for i in row})
    strings = dict(zip(ids, tk.decode_batch([[i] for i in ids], skip_special_tokens=True)))  # BOS: ""
    varies = [[v for v in p.get("varies") or [] if v != "tokens"] for p in behavior["prompts"]]
    return {"prompts": [[strings[i] for i in row] for row in prompts],
            "counterfactuals": [[strings[i] for i in row] for row in changed],
            "targets": [list(p["target_positions"]) for p in behavior["prompts"]],
            "varies": varies, "variables": sorted({v for row in varies for v in row})}


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
FIXED_NAMES = set(SAFE_BUILTINS) | {"groups"}
SKIPPED = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
           tokenize.ENCODING, tokenize.ENDMARKER}


def code_length(source: str) -> tuple[int, int]:
    """(python_tokens, token_types) of the code a reader must read beyond the structure: every statement but
    the `groups` assignment (the checker prices groups, reads and labels as structure), without comments and
    docstrings. A name, keyword or operator is one token, a number or string literal one token per character
    as written. token_types = keywords + operators + builtin names + names the file defines + literal
    characters."""
    tree = ast.parse(source)
    docs = [((n.lineno, n.col_offset), (n.end_lineno, n.end_col_offset)) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    structure = [(n.lineno, n.end_lineno) for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "groups" for t in n.targets)]
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
