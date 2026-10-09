"""The loader of graph-oracle answers (#2951): plain Python, no imports.

An answer explains the target model's prediction of the next token after a text: a function graph(tokens, targets)
of the sequence (the model's token strings) and the positions whose next token is explained. Its docstring is the
plain-English explanation. It returns a list of steps, most important first; each step is a dict
{(position, reader): parents, ..., "out": parents, "uses": [library entries]} adding subcomponents and connections to
the graph of the steps before it, so the first k steps are themselves a complete, smaller explanation. A comment line
above a step says in English what it adds. A reader is a subcomponent "<p:L.S.I>" (layer L, weight matrix S in q k v
o fc down: query, key, value, attention output, MLP input, MLP output; subcomponent I) at a position; its parents are
the subcomponents whose outputs it reads, written as one string of subcomponents at the reader's own position, or as
{position: string} (an attention output reading values at positions). "out" lists what the prediction at the targets
reads. Every edge must be a connection the model has (connects()). Nodes are the readers and their parents. "uses"
lists library entries (score.py's LIBRARY: recurring mechanisms, sets of edges at positions relative to the target)
the step includes without writing them out. A single dict is an answer of one step.

    def graph(tokens, targets):
        '''At the last position the attention output reads the value written at position 3, which ...'''
        t = targets[0]
        return [
            # the attention output that writes the prediction reads position 3
            {(t, "<p:3.o.281>"): {3: "<p:3.v.676>"}, "out": "<p:3.o.281>"},
            # what the value at position 3 reads
            {(3, "<p:3.v.676>"): "<p:0.down.3473>", "out": "<p:2.down.773>"},
        ]

trace(source, model, behavior=...) runs a source in a sandboxed child (restricted syntax and builtins, CPU and memory
limits) and returns the IR: {"graph": {"nodes": [[layer, matrix, position, index], ...], "parents": [[reader, writer],
...], "out": [writer, ...], "uses": [entry, ...], "node_step", "parent_step", "out_step", "uses_step": the step that
added each, "explanation": the docstring, "notes": the comment lines in order}} with nodes as indices, and "valid" /
"error".
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import re
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES_FILE = HERE / "shapes.json"
QWEN3 = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3-8b": "Qwen/Qwen3-8B"}
VPD4L_TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
SITES = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "fc": "c_fc", "down": "down_proj"}
PART = re.compile(r"<p:(\d+)\.(q|k|v|o|fc|down)\.(\d+|rest)>")


RESID_WRITERS = ("o_proj", "down_proj")
RESID_READERS = ("q_proj", "k_proj", "v_proj", "c_fc")


def stage(layer: int, kind: str) -> float:
    """A residual writer's place in its position's residual stream (attention output of layer l at l + 0.5, MLP output
    at l + 1) or a reader's (query, key, value at l, MLP input at l + 0.5): a reader reads what was written before."""
    return layer + (0.5 if kind in ("o_proj", "c_fc") else (1.0 if kind == "down_proj" else 0.0))


def connects(wl: int, wk: str, wt: int, rl: int | None, rk: str | None, rt: int | None, targets: list[int]) -> bool:
    """Whether the model connects a writer subcomponent (layer wl, matrix wk, position wt) to a reader (rl, rk, rt; rk
    None: the prediction at the targets): the residual stream at one position (an attention or MLP output into a later
    query, key, value or MLP input, or into the prediction), attention (a value into the same layer's attention output
    at that or a later position), or one MLP (an MLP input into the same MLP's output)."""
    if rk is None:
        return wk in RESID_WRITERS and wt in targets
    if wk in RESID_WRITERS and rk in RESID_READERS:
        return wt == rt and stage(wl, wk) <= stage(rl, rk)
    if wk == "v_proj" and rk == "o_proj":
        return wl == rl and wt <= rt
    if wk == "c_fc" and rk == "down_proj":
        return wl == rl and wt == rt
    return False


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


def _part(token, shape: dict) -> tuple[int, str, object]:
    if not isinstance(token, str) or not PART.fullmatch(token):
        raise MechError(f"{token!r} is not a subcomponent \"<p:L.S.I>\" (S one of q k v o fc down)")
    m = PART.fullmatch(token)
    layer, site = int(m[1]), SITES[m[2]]
    if layer >= shape["layers"]:
        raise MechError(f"{token}: layers are 0..{shape['layers'] - 1}")
    if m[3] == "rest":
        return layer, site, "rest"
    size = shape["views"]["vpd"][layer][site]
    if int(m[3]) >= size:
        raise MechError(f"{token}: layer {layer}'s {site} has subcomponents 0..{size - 1}")
    return layer, site, int(m[3])


def _names(parts) -> list:
    """Subcomponents written as a list, or as one string of them written together."""
    if isinstance(parts, str):
        if PART.sub("", parts).strip():
            raise MechError(f"{parts[:80]!r}: a string of subcomponents holds only \"<p:L.S.I>\" tokens")
        return [m[0] for m in PART.finditer(parts)]
    if not isinstance(parts, (list, tuple, set, frozenset)):
        raise MechError(f"{parts!r}: parents are a string of subcomponents, a list of them, or {{position: string}}")
    return list(parts)


def graph(fn, model: str, behavior: dict | None) -> dict:
    """The IR graph of an answer: graph(tokens, targets) run on the task's sequence (behavior_tokens()'s first
    "sequences" entry)."""
    if not behavior or not behavior.get("sequences"):
        raise MechError("an answer needs the task's sequence to run on")
    shape = shapes(model)
    _, strings, targets = behavior["sequences"][0]
    T = len(strings)
    out = fn(list(strings), list(targets))
    steps = [out] if isinstance(out, dict) else out
    if not isinstance(steps, (list, tuple)) or not all(isinstance(st, dict) for st in steps):
        raise MechError(f"graph() returned {type(out).__name__}: a list of steps, each a dict {{(position, reader): parents, \"out\": parents}}")
    index: dict[tuple, int] = {}
    nodes: list[list] = []
    node_step: list[int] = []
    step = 0

    def node(position, token) -> int:
        if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position < T:
            raise MechError(f"position {position!r}: positions are 0..{T - 1}")
        layer, kind, i = _part(token, shape)
        if i == "rest":
            raise MechError(f"{token}: a remainder is not a graph node")
        key = (layer, kind, position, i)
        if key not in index:
            index[key] = len(nodes)
            nodes.append(list(key))
            node_step.append(step)
        return index[key]

    def parents(value, at: list[int]) -> list[int]:
        if isinstance(value, dict):
            return [node(p, tok) for p, names in value.items() for tok in _names(names)]
        return [node(p, tok) for p in at for tok in _names(value)]

    edges, reads, uses = [], [], []
    edge_step, read_step, uses_step = [], [], []
    seen_edges, seen_reads = set(), set()
    for step, st in enumerate(steps):
        for key, value in st.items():
            if key == "uses":
                if not (isinstance(value, (list, tuple)) and all(isinstance(u, str) for u in value)):
                    raise MechError("uses: a list of library entry names")
                for u in value:
                    if u not in uses:
                        uses.append(u)
                        uses_step.append(step)
                continue
            if key == "out":
                for w in parents(value, list(targets)):
                    wl, wk, wt, _ = nodes[w]
                    if not connects(wl, wk, wt, None, None, None, list(targets)):
                        raise MechError(f"\"out\": the prediction reads attention and MLP outputs at the targets {list(targets)}, not layer {wl}'s {wk} at {wt}")
                    if w not in seen_reads:
                        seen_reads.add(w)
                        reads.append(w)
                        read_step.append(step)
                continue
            if not (isinstance(key, tuple) and len(key) == 2):
                raise MechError(f"key {key!r}: (position, subcomponent), \"out\" or \"uses\"")
            r = node(*key)
            rl, rk, rt, _ = nodes[r]
            for w in parents(value, [rt]):
                wl, wk, wt, _ = nodes[w]
                if not connects(wl, wk, wt, rl, rk, rt, list(targets)):
                    raise MechError(f"{key!r} reads layer {wl}'s {wk} at {wt}: the model has no such connection (an attention or MLP output "
                                    "into a later query, key, value or MLP input at its position; a value into the same layer's attention "
                                    "output at that or a later position; an MLP input into the same MLP's output)")
                if (r, w) not in seen_edges:
                    seen_edges.add((r, w))
                    edges.append([r, w])
                    edge_step.append(step)
    return {"nodes": nodes, "parents": edges, "out": reads, "uses": uses, "node_step": node_step, "parent_step": edge_step,
            "out_step": read_step, "uses_step": uses_step, "steps": len(steps), "explanation": (getattr(fn, "__doc__", None) or "").strip()}


def notes(source: str) -> list[str]:
    """The comment lines of a source, in order."""
    import io
    import tokenize

    try:
        return [t.string.lstrip("#").strip() for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


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


EMPTY_GRAPH = {"nodes": [], "parents": [], "out": [], "uses": [], "node_step": [], "parent_step": [], "out_step": [], "uses_step": [], "steps": 0,
               "explanation": "", "notes": []}


def _empty(source: str, model: str) -> dict:
    return {"model": model, "source": source, "graph": dict(EMPTY_GRAPH), "valid": False, "error": None}


def _trace(source: str, model: str, behavior: dict | None = None) -> dict:
    """Checks and runs `source` in this process -> IR (`behavior`: behavior_tokens()'s payload, the sequences on()
    runs on). The sandboxed child's entry point."""
    ir = _empty(source, model)
    try:
        if len(source) > MAX_SOURCE:
            raise MechError(f"explanation longer than {MAX_SOURCE} characters")
        tree = ast.parse(source, "<explanation>")
        check(tree)
        namespace = {"__builtins__": SAFE_BUILTINS, "__name__": "explanation"}
        exec(compile(tree, "<explanation>", "exec"), namespace)
        if not callable(namespace.get("graph")):
            raise MechError("the answer defines no function graph(tokens, targets)")
        ir.update(graph={**graph(namespace["graph"], model, behavior), "notes": notes(source)}, valid=True)
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
        ir["graph"] = dict(EMPTY_GRAPH)
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
    """What the tracer needs of a behavior: {"sequences"}, every prompt as (token ids, token strings, target
    positions). `behavior`: a behavior record or its file."""
    if not isinstance(behavior, dict):
        behavior = json.loads(Path(behavior).expanduser().read_text())
    tk = tokenizer(model)
    rows = [p["token_ids"] for p in behavior["prompts"]]
    ids = sorted({i for row in rows for i in row})
    strings = dict(zip(ids, tk.decode_batch([[i] for i in ids], skip_special_tokens=True)))  # BOS: ""
    return {"sequences": [(row, [strings[i] for i in row], list(p["target_positions"])) for row, p in zip(rows, behavior["prompts"]) if row]}


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
    """Checks and runs `source` sandboxed -> IR. `behavior` (a record or its file): the sequences on() runs on. Each
    thread keeps one tracer server (`mech.py serve`, started with -I -S), so a trace costs a fork."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trace", help="stdin explanation -> IR JSON on stdout")
    t.add_argument("--model", default="vpd4l")
    t.add_argument("--behavior", type=Path, help="a behavior file: the sequences on() runs on")
    sub.add_parser("serve", help="the tracer server (JSON lines; trace() starts it)")
    a = ap.parse_args()
    if a.command == "trace":
        print(json.dumps(trace(sys.stdin.read(), a.model, behavior=a.behavior), indent=1))
    else:
        serve()


if __name__ == "__main__":
    main()
