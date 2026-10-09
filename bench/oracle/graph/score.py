"""One call: an explanation (a gate program through mech's tracer, or its IR) and a task -> its score in bits, from the
Rust checker (crates/gam-mpd/examples/mpd_graph_2951.rs, a JSON-lines server).

An explanation names a circuit: which subcomponents act at which positions (mech.gates). Every subcomponent it leaves
out is deleted (VPD's ablation), and it is judged by the KL in bits of the model's next-token distribution at the
task's targets from the circuit's (exec_error_bits, summed over targets) and by the (subcomponent, position) pairs it
names (pairs), against the task's reference, VPD's own answer (text.py) scored alongside it: an explanation no less
faithful than the reference is better the fewer pairs it names, one less faithful worse by its excess KL (order).

    from score import Checker
    with Checker("vpd4l") as c:
        c.behavior("~/mpd-data/graph_oracle/texts/vpd4l/text9000.json")
        print(c.score(open("explanation.py").read()))
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORTS = {"vpd4l": Path.home() / "mpd-data/engine/vpd4l"}
VIEWS = {"vpd4l": {"vpd": Path.home() / "mpd-data/engine/vpd4l_decomposition"}}
METRIC = "full"  # the model's whole next-token distribution
# The published checker build (each release build of the graph checker is copied there).
PUBLISHED = Path.home() / "mpd-data/graph_oracle/bin/mpd_graph_2951"
# On MATS (mats-run with MATS_BUILD=1) or a pod: the job's own build, $MPD_BIN/mpd_graph_2951 (Linux, CUDA with device "gpu").
BINARY = Path(os.environ.get("GRAPH_CHECKER") or (Path(os.environ["MPD_BIN"]) / "mpd_graph_2951" if os.environ.get("MPD_BIN") else PUBLISHED))


def order(s: dict, reference: dict | None = None) -> tuple:
    """How a score ranks, lower first: valid, then its KL above the reference's (a score of VPD's answer on the same
    text, from the same checker; without one, its KL), then fewer pairs."""
    if not s.get("valid", True):
        return (1, math.inf, math.inf)
    ref = reference["exec_error_bits"] if reference and reference.get("valid", True) else 0.0
    return (0, max(0.0, s["exec_error_bits"] - ref), s["pairs"])


def pairs(ir: dict) -> int:
    """The (subcomponent, position) pairs an IR's circuit names: per node its subcomponents times the positions it acts
    at over every sequence (a node without positions acts at all of them, counted as one each)."""
    total = 0
    for node in ir.get("nodes", []):
        count = sum(len(p["index"]) if isinstance(p.get("index"), list) else 1 for p in node["pieces"])
        at = node.get("at") or []
        total += count * (sum(len(s["positions"]) for s in at) if at else 1)
    return total


class Checker:
    def __init__(self, model, export=None, memory_gib=None, threads=None, views=None, device=None, memo_dir=None):
        """memory_gib: the server's mem-lease (GRAPH_MEM_GIB, else 16; vpd4l: a batch of 8 programs at 8 threads ran
        under 12 GiB and was killed under 8 GiB; one 255-subcomponent program at 64 experiments reached 12.5); threads: its rayon threads (RAYON_NUM_THREADS when unset, 6 by default); views: decomposition
        views to attach ({"vpd": DIR}, the model's VPD decomposition by default); device: "gpu" runs the large products
        on the single-precision device (float32, so compare scores within one device); memo_dir: the server's
        --memo-dir (GRAPH_MEMO_DIR when unset), per-behavior memos of the targets that later runs reuse."""
        memory_gib = int(memory_gib or os.environ.get("GRAPH_MEM_GIB", 16))  # mem-lease takes whole GiB
        env = dict(os.environ)
        env.setdefault("RAYON_NUM_THREADS", str(threads or 6))
        self.model = model
        command = [str(BINARY)]
        memo_dir = memo_dir or os.environ.get("GRAPH_MEMO_DIR")
        if memo_dir:
            command += ["--memo-dir", str(Path(memo_dir).expanduser())]
        if not os.environ.get("MEM_LEASE_GIB") and shutil.which("mem-lease"):  # the Mac's memory guard where it exists
            command = ["mem-lease", str(memory_gib)] + command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        views = VIEWS.get(model, {}) if views is None else views
        self.request({"op": "load", "export": str(Path(export or EXPORTS[model]).expanduser()),
                      **{k: str(Path(v).expanduser()) for k, v in views.items()}, **({"device": device} if device else {})})
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

    def score(self, program, experiments=0, seed=0, N=None, stand_in="delete", metric=None):
        """Every score term. `program` is a source, an IR dict, or {"source", "explanation"} (the oracle's answer split
        by prompt.split_answer). An untraceable source is scored as naming nothing, flagged invalid."""
        return self.score_batch([program], experiments, seed, N, stand_in, metric=metric)[0]

    def score_batch(self, programs, experiments=0, seed=0, N=None, stand_in="delete", uniform_seeds=None, options=None, metric=None):
        """score() for many programs of the current behavior under one seed, in one checker request (the server runs M
        once per experiment it has not cached and the programs in parallel). uniform_seeds m: the experiments are drawn
        from seed mod m, so m collections recur across a caller's seeds. options: further request keys passed to the
        server as they are). stand_in: what the subcomponents a program does not name carry: "delete" (VPD's ablation)
        or "counterfactual". metric: "full" (METRIC) or the checker's others. Each score gains pairs; necessity runs
        are off unless options turn them on."""
        irs = [self.ir(p) for p in programs]
        request = {"op": "score", "programs": irs, "experiments": experiments, "seed": seed, "routing": "edges", "N": N,
                   "reader_top": 0, "stand_in": stand_in, "metric": metric or METRIC, "necessity": False, **(options or {})}
        if uniform_seeds:
            request["uniform_seeds"] = uniform_seeds
        scores = self.request(request)["scores"]
        for s, ir in zip(scores, irs):
            s["pairs"] = pairs(ir)
        return scores

    def ir(self, program):
        """The IR of a program (source, IR, or {"source", ...})."""
        if isinstance(program, dict) and "nodes" in program:
            return program
        sys.path.insert(0, str(HERE))
        import mech

        source = program if isinstance(program, str) else program["source"]
        try:
            return mech.trace(source, self.model, behavior=self.behavior_record)
        except Exception as e:  # the tracer's error is the program's error
            return {"model": self.model, "nodes": [], "edges": [], "source": source, "valid": False, "error": f"{type(e).__name__}: {e}"}

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
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("program", help="an explanation .py (traced by mech) or an IR .json")
    p.add_argument("behavior")
    p.add_argument("--model", default="vpd4l")
    p.add_argument("--experiments", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    text = Path(a.program).read_text()
    program = json.loads(text) if a.program.endswith(".json") else text
    print(json.dumps(score(program, a.behavior, model=a.model, experiments=a.experiments, seed=a.seed), indent=1))
