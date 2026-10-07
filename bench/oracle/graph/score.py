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
}
BINARY = Path(os.environ.get("GRAPH_CHECKER", Path.home() / "mpd-data/scratch/g-exec/bin/mpd_graph_2951"))
HERE = Path(__file__).resolve().parent


def trace(source, model):
    """The IR of a program's source, through mech's sandboxed tracer (g-mech's bench/oracle/graph/mech.py)."""
    sys.path.insert(0, str(HERE))
    import mech
    return mech.trace(source, model)


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
    def __init__(self, model, export=None, memory_gib=6):
        self.model = model
        export = Path(export or EXPORTS[model]).expanduser()
        command = [str(BINARY)]
        if not os.environ.get("MEM_LEASE_GIB"):
            command = ["mem-lease", str(memory_gib)] + command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.request({"op": "load", "export": str(export)})
        self.behavior_record = None

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
        return self.request({"op": "behavior", "path": str(path)})

    def score(self, program, experiments=32, seed=0, routing="edges", N=None, reader=True, reader_top=8):
        """Every score term (design.txt section 5). `program` is Python source or an IR dict. An untraceable
        source is scored as the empty program, flagged invalid. The reader term comes from the reader_score
        server at GRAPH_READER (host:port); without one it is left out (reader_error_bits None) and the items
        are returned for a later reader pass."""
        if isinstance(program, str):
            try:
                ir = trace(program, self.model)
            except Exception as e:  # the tracer's error is the program's error
                ir = {"model": self.model, "nodes": [], "edges": [], "python_tokens": 0, "token_types": 0,
                      "source": program, "valid": False, "error": f"{type(e).__name__}: {e}"}
        else:
            ir = program
        answer = self.request({"op": "score", "program": ir, "experiments": experiments, "seed": seed,
                               "routing": routing, "N": N, "reader_top": reader_top if reader else 0})
        items = answer.pop("items", None)
        answer["reader_error_bits"] = None
        if reader and items:
            items = self.texts(items)
            address = os.environ.get("GRAPH_READER")
            if address:
                result = reader_request(address, {"op": "score", "N": int(answer["N"]), "items": items,
                                                  "programs": [{"id": "p", "source": ir.get("source", ""), "valid": ir.get("valid", True)}]})
                answer["reader"] = result
                answer["reader_error_bits"] = result["reader_error_bits"]
                answer["total_bits"] += answer["reader_error_bits"]
            else:
                answer["items"] = items
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
