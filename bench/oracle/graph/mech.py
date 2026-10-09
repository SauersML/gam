"""The loader of graph-oracle explanations (#2951): plain Python, no imports.

An explanation is a gate program: a function on(tokens, targets) of the sequence (the model's token strings) and the
positions whose next token is explained, returning which of the target model's VPD subcomponents act at each position,
{position: subcomponents} or [(position, subcomponent), ...]. A position's subcomponents are a list of names or one
string of names written together. A name is "<p:L.S.I>": layer L, site S (q k v o fc down: q_proj k_proj v_proj
o_proj c_fc down_proj), subcomponent I; "<p:L.S.rest>" is the site's remainder W - sum of its subcomponents. The
model's own weights do every computation, connected as in the model; what the program names acts where it says, and
the scorer decides what every other subcomponent carries.

    def on(tokens, targets):
        return {t: "<p:0.fc.225><p:2.v.80><p:2.o.63>" for t in targets}

trace(source, model, behavior=...) runs a source in a sandboxed child (restricted syntax and builtins, CPU and memory
limits) and returns the IR the checker reads: one node per block (a layer's attention or MLP) and set of positions,
with its positions per sequence.
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


class Node:
    """Subcomponents of one block (a layer's attention or MLP)."""

    def __init__(self, id: str, layer: int, block: str, parts: dict[str, set]):
        self.id, self.layer, self.block, self.parts = id, layer, block, parts  # parts: site -> indices ("rest" too)

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
    """A position's subcomponents: a list of names, or one string of names written together."""
    if isinstance(parts, str):
        if PART.sub("", parts).strip():
            raise MechError(f"{parts[:80]!r}: a string of subcomponents holds only names \"<p:L.S.I>\"")
        return [m[0] for m in PART.finditer(parts)]
    if not isinstance(parts, (list, tuple, set, frozenset)):
        raise MechError(f"on() returned {parts!r} for a position: a list of subcomponents or a string of them")
    return list(parts)


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


def gates(on, model: str, behavior: dict | None) -> list[dict]:
    """The IR nodes of a gate program: `on(tokens, targets)` run on every sequence of the behavior
    (behavior_tokens()'s "sequences") returns, per position, the subcomponents that act there; subcomponents of one
    block acting at the same positions of every sequence are one node, which acts there and nowhere else."""
    if not behavior or not behavior.get("sequences"):
        raise MechError("a gate program needs the behavior's sequences to run on")
    shape = shapes(model)
    where: dict[tuple, list[set]] = {}
    sequences = behavior["sequences"]
    for k, (ids, strings, targets) in enumerate(sequences):
        out = on(list(strings), list(targets))
        pairs = [(p, part) for p, parts in out.items() for part in _names(parts)] if isinstance(out, dict) else list(out)
        for item in pairs:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise MechError(f"on() returned {item!r}: positions map to lists of subcomponents, or (position, subcomponent) pairs")
            position, token = item
            if not isinstance(position, int) or not 0 <= position < len(strings):
                raise MechError(f"on() returned position {position!r} for a sequence of {len(strings)} tokens")
            unit = _part(token, shape)
            where.setdefault(unit, [set() for _ in sequences])[k].add(position)
    grouped: dict[tuple, dict[str, set]] = {}
    for (layer, site, index), at in where.items():
        key = (layer, "mlp" if site in ("c_fc", "down_proj") else "attn", tuple(tuple(sorted(a)) for a in at))
        grouped.setdefault(key, {}).setdefault(site, set()).add(index)
    nodes = []
    for j, ((layer, block, at), parts) in enumerate(sorted(grouped.items(), key=lambda kv: kv[0][:2])):
        node = Node(f"n{j}", layer, block, parts).ir()
        node["at"] = [{"tokens": list(ids), "positions": list(a)} for (ids, _, _), a in zip(sequences, at)]
        nodes.append(node)
    return nodes


def _empty(source: str, model: str) -> dict:
    return {"model": model, "decomposition": "vpd", "nodes": [], "edges": [], "wiring": "model", "standin": None,
            "source": source, "valid": False, "error": None}


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
        if not callable(namespace.get("on")):
            raise MechError("the explanation defines no function on(tokens, targets)")
        nodes = gates(namespace["on"], model, behavior)
        ir.update(nodes=nodes, edges=[] if nodes else [{"from": "embed", "to": "logits", "route": "input"}], valid=True)
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
        ir.update(nodes=[], edges=[])
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
