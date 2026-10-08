"""One call: a program (Python source through mech's tracer, or its IR) and a behavior -> every score term in bits.

The execution, code and opaque-number terms come from the Rust checker (crates/gam-mpd/examples/mpd_graph_2951.rs,
a JSON-lines server); the reader term from reader_score.py (g-reader), fed the checker's per-experiment items
(the experiment in words, M_e's probabilities of M's clean top-k tokens and of everything else).

    from score import Checker
    with Checker("vpd4l") as c:
        c.behavior("~/mpd-data/graph_oracle/behaviors/vpd4l/induction_0.json")
        print(c.score(open("prog.py").read()))
"""
import json
import os
import subprocess
import sys
from pathlib import Path

EXPORTS = {
    "vpd4l": Path.home() / "mpd-data/engine/vpd4l",
    "qwen3-0.6b": Path.home() / "mpd-data/engine/qwen3_0p6b_heldout64",
}
# The latest checker build (g-exec2 copies each release build there), else g-exec's first build.
PUBLISHED = Path.home() / "mpd-data/graph_oracle/bin/mpd_graph_2951"
BINARY = Path(os.environ.get("GRAPH_CHECKER") or (PUBLISHED if PUBLISHED.exists() else Path.home() / "mpd-data/scratch/g-exec/bin/mpd_graph_2951"))
HERE = Path(__file__).resolve().parent
# Each model's published shared base (g-exec2): generic machinery declared once per model, a program IR whose nodes
# the checker adds to every scored program (always on, connected to every node, priced apart in base_bits).
BASES = {"vpd4l": Path.home() / "mpd-data/graph_oracle/base_vpd4l.json"}


def trace(source, model, behavior=None, decomposition=None):
    """The IR of a program's source, through mech's sandboxed tracer (g-mech's bench/oracle/graph/mech.py):
    its algorithm evaluated on `behavior`'s prompts, PD bound to `decomposition`."""
    sys.path.insert(0, str(HERE))
    import mech
    return mech.trace(source, model, behavior=behavior, decomposition=decomposition)


def tokenizer(model):
    """The target model's decoder: token ids -> text."""
    if model == "vpd4l":
        import tokenizers
        tk = tokenizers.Tokenizer.from_file(str(Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"))
        return lambda ids: tk.decode(list(ids), skip_special_tokens=False)
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return lambda ids: hf.decode(list(ids))


def reader_request(address, message):
    """One request to reader_score.py's server (JSON lines over TCP) -> the program's result."""
    import socket
    host, port = address.rsplit(":", 1)
    with socket.create_connection((host, int(port))) as s:
        s.sendall((json.dumps(message) + "\n").encode())
        reply = json.loads(s.makefile().readline())
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["ok"]["results"][0]


class Checker:
    def __init__(self, model, export=None, memory_gib=None, threads=None, views=None, device=None, memo_dir=None, base=None):
        """memory_gib: the server's mem-lease (vpd4l: a batch of 8 programs at 8 threads ran under 12 GiB and
        was killed under 8 GiB); threads: its rayon threads (RAYON_NUM_THREADS when unset, 6 by default: runs
        in parallel each hold their own streams and log-probabilities); views: decomposition views to attach,
        {"vpd": DIR, "transcoders": DIR} (the server's load keys); device: "gpu" runs the large products on
        the single-precision device (the server's load key; float32, so compare scores within one device);
        memo_dir: the server's --memo-dir (GRAPH_MEMO_DIR when unset), small per-behavior memos of the targets
        and native bit widths that later runs on the same behavior reuse (builds from a6a063a9c4 on); base: the
        shared base every program is scored with (a base IR file, or True for the model's published BASES entry;
        GRAPH_BASE when unset; builds from 880cfaa196 on)."""
        # Qwen3-0.6B's load in float64 passed 16.3 GiB and was killed under a 16 GiB lease.
        memory_gib = memory_gib or (28 if model.startswith("qwen3") else 12)
        env = dict(os.environ)
        env.setdefault("RAYON_NUM_THREADS", str(threads or 6))
        self.model = model
        export = Path(export or EXPORTS[model]).expanduser()
        command = [str(BINARY)]
        memo_dir = memo_dir or os.environ.get("GRAPH_MEMO_DIR")
        if memo_dir:
            command += ["--memo-dir", str(Path(memo_dir).expanduser())]
        if not os.environ.get("MEM_LEASE_GIB"):
            command = ["mem-lease", str(memory_gib)] + command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.request({"op": "load", "export": str(export), **{k: str(Path(v).expanduser()) for k, v in (views or {}).items()}, **({"device": device} if device else {})})
        self.behavior_record = None
        base = base if base is not None else os.environ.get("GRAPH_BASE")
        self.base = str(BASES[model] if base is True or base == "1" else Path(base).expanduser()) if base else None
        # What a program's PD names: the attached decomposition (the library over VPD when both are).
        views = views or {}
        self.decomposition = "library" if "library" in views else "vpd" if "vpd" in views else "transcoder" if "transcoders" in views else "native"

    def request(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("the checker exited")
        answer = json.loads(line)
        if answer.get("ok") is False:
            raise RuntimeError(answer.get("error"))
        return answer

    def behavior(self, path):
        path = Path(path).expanduser()
        self.behavior_record = json.loads(path.read_text())
        answer = self.request({"op": "behavior", "path": str(path)})
        if self.base:
            answer["base"] = self.request({"op": "base", "path": self.base})
        return answer

    def score(self, program, experiments=32, seed=0, routing="edges", N=None, reader=True, reader_top=8, stand_in="input"):
        """Every score term (design.txt section 5). `program` is Python source, an IR dict, or {"source",
        "explanation"} (the oracle's answer split by prompt.split_answer); the reader reads the explanation
        alone (none when absent). An untraceable source is scored as the empty program, flagged invalid. The reader term comes from the reader_score
        server at GRAPH_READER (host:port); without one it is left out (reader_error_bits None) and the items
        are returned for a later reader pass."""
        return self.score_batch([program], experiments, seed, routing, N, reader, reader_top, stand_in)[0]

    def score_batch(self, programs, experiments=32, seed=0, routing="edges", N=None, reader=True, reader_top=8, stand_in="input", uniform_seeds=None, options=None):
        """score() for many programs of the current behavior under one seed, in one checker request (the server
        runs M once per experiment it has not cached and the programs in parallel). uniform_seeds m: the
        experiments are drawn from seed mod m, so m collections recur across a caller's seeds (58119de9b3).
        options: further request keys passed to the server as they are (e.g. experiment families)."""
        irs = [self.ir(p) for p in programs]
        request = {"op": "score", "programs": irs, "experiments": experiments, "seed": seed,
                   "routing": routing, "N": N, "reader_top": reader_top if reader else 0, "stand_in": stand_in, **(options or {})}
        if uniform_seeds:
            request["uniform_seeds"] = uniform_seeds
        answer = self.request(request)
        return [self.finish(ir, a, reader) for ir, a in zip(irs, answer["scores"])]

    def ir(self, program):
        """The IR of a program (source, IR, or {"source", "explanation"}), carrying its explanation."""
        if isinstance(program, dict) and "nodes" in program:
            return program
        source, explanation = (program, "") if isinstance(program, str) else (program["source"], program.get("explanation") or "")
        try:
            ir = trace(source, self.model, self.behavior_record, self.decomposition)
        except Exception as e:  # the tracer's error is the program's error
            ir = {"model": self.model, "nodes": [], "edges": [], "python_tokens": 0, "token_types": 0,
                  "source": source, "valid": False, "error": f"{type(e).__name__}: {e}"}
        ir["explanation"] = explanation
        sys.path.insert(0, str(HERE))
        import mech
        ir["explanation_tokens"], ir["explanation_token_types"] = mech.explanation_length(explanation)
        return ir

    def finish(self, ir, answer, reader):
        """Adds the reader term to one program's checker answer."""
        if "binding_error_bits" in answer:  # it would score no alignment: the IR's "alignments" key is new to it
            raise RuntimeError(f"the checker {BINARY} predates the alignment rename (binding -> alignment); use a build that reads \"alignments\"")
        items = answer.pop("items", None)
        answer["reader_error_bits"] = None
        if reader and items:
            items = self.texts(items)
            address = os.environ.get("GRAPH_READER")
            if address:
                result = reader_request(address, {"op": "score", "N": int(answer["N"]), "items": items,
                                                  "programs": [{"id": "p", "explanation": ir.get("explanation", ""), "valid": ir.get("valid", True)}]})
                answer["reader"] = result
                answer["reader_error_bits"] = result["reader_error_bits"]
                answer["total_bits"] += answer["reader_error_bits"]
            else:
                answer["items"] = items
                answer["explanation"] = ir.get("explanation", "")  # what a later reader pass reads
        return answer

    def texts(self, items):
        """Adds the text M reads and each candidate's string (the model's own tokenizer)."""
        decode = tokenizer(self.model)
        for it in items:
            it["text"] = decode(it["token_ids"])
            for c in it["candidates"]:
                c["text"] = decode([c["token_id"]])
        return items

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
            self.process.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def score(program, behavior_path, model="vpd4l", **kw):
    with Checker(model) as c:
        c.behavior(behavior_path)
        return c.score(program, **kw)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("program", help="program .py (traced by mech) or IR .json")
    p.add_argument("behavior")
    p.add_argument("--model", default="vpd4l")
    p.add_argument("--experiments", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--routing", default="edges", choices=["edges", "nodes"])
    p.add_argument("--no-reader", action="store_true")
    a = p.parse_args()
    text = Path(a.program).read_text()
    program = json.loads(text) if a.program.endswith(".json") else text
    print(json.dumps(score(program, a.behavior, model=a.model, experiments=a.experiments, seed=a.seed,
                           routing=a.routing, reader=not a.no_reader), indent=1))
