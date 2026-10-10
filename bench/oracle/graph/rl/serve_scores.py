"""The verifier as a service (#2951 graph oracle): persistent scoring processes on the GPU, each holding a score.Scorer
(vpd4l and VPD's subcomponents loaded once, its per-text work cached), behind HTTP, so an RL stack's reward function
(prime-rl's verifiers environment, or rl/train.py) posts answers and gets score dicts back. A text's answers always go
to the same process, so its changed prompts and baselines are computed once; a process scores together every queued
answer to the same (text, seed, necessity).

  serve_scores.py [--port 8765] [--workers 4] [--device cuda] [--texts DIR]
  POST /score {"task": TASK_ID, "sources": [SRC, ...] | "answers": [REPLY, ...], "seed": 0, "necessity": false}
      -> {"scores": [{"valid", "error", "curve", "lo", "hi", "kl_bits", "bits", "steps", "area", "reward"}, ...]}
  A reply is an oracle's whole answer, its program taken by prompt.split_answer. "reward" is minus the curve area, an
  answer that cannot run counting as the empty answer (score.EMPTY, scored with the request and cached).
  GET /health -> {"workers": N, "pending": requests in flight}
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import queue
import sys
import threading
import zlib
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))

KEEP = ("valid", "error", "curve", "lo", "hi", "kl_bits", "bits", "steps", "nodes", "edges", "explanation", "notes")


def work(device: str | None, texts: str, inbox, outbox) -> None:
    """One scoring process: take a request, then every request already queued, score them grouped by (text, seed,
    necessity), and answer each with its own scores in order."""
    import score
    from prompt import split_answer

    sc = score.Scorer(device)
    tasks = {}

    def task(tid: str) -> dict:
        if tid not in tasks:
            p = Path(texts) / f"{tid}.json"
            tasks[tid] = {**json.loads(p.read_text()), "path": str(p)}
        return tasks[tid]

    while True:
        batch = [inbox.get()]
        while True:
            try:
                batch.append(inbox.get_nowait())
            except queue.Empty:
                break
        groups = {}
        for rid, req in batch:
            srcs = req["sources"] if "sources" in req else [split_answer(a if "```" in a else "```python\n" + a)[0] for a in req["answers"]]
            groups.setdefault((req["task"], int(req.get("seed", 0)), bool(req.get("necessity"))), []).append((rid, srcs))
        for (tid, seed, nec), reqs in groups.items():
            sources = [s for _, srcs in reqs for s in srcs]
            try:
                got = sc.score(task(tid), sources + [score.EMPTY], seed, necessity=nec)
                empty = score.key(got.pop())[1]
                got = [{**{k: s[k] for k in KEEP if k in s}, "area": score.key(s)[1]} for s in got]
                for s in got:  # an answer that cannot run: no area (JSON has no infinity), the empty answer's reward
                    ran = s.get("valid", True) and s["area"] < float("inf")
                    s["area"], s["reward"] = (s["area"], -s["area"]) if ran else (None, -empty)
                err = None
            except Exception as e:  # a bad request answers with its error; the process keeps serving
                got, err = None, f"{type(e).__name__}: {e}"
            at = 0
            for rid, srcs in reqs:
                outbox.put((rid, got[at:at + len(srcs)] if got is not None else None, err))
                at += len(srcs)


class Pool:
    """The scoring processes, a request id counter, and the futures of the requests in flight."""

    def __init__(self, workers: int, device: str | None, texts: str):
        ctx = mp.get_context("spawn")  # CUDA in each process
        self.out = ctx.Queue()
        self.inboxes = [ctx.Queue() for _ in range(workers)]
        self.procs = [ctx.Process(target=work, args=(device, texts, q, self.out), daemon=True) for q in self.inboxes]
        for p in self.procs:
            p.start()
        self.ids, self.waiting, self.lock = itertools.count(), {}, threading.Lock()
        threading.Thread(target=self.collect, daemon=True).start()

    def collect(self) -> None:
        while True:
            rid, scores, err = self.out.get()
            with self.lock:
                fut = self.waiting.pop(rid)
            if err:
                fut.set_exception(RuntimeError(err))
            else:
                fut.set_result(scores)

    def submit(self, req: dict) -> Future:
        fut, rid = Future(), next(self.ids)
        with self.lock:
            self.waiting[rid] = fut
        self.inboxes[zlib.crc32(req["task"].encode()) % len(self.inboxes)].put((rid, req))  # a text's answers to one process
        return fut


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device")
    ap.add_argument("--texts", default=str(Path.home() / "mpd-data/graph_oracle/texts/vpd4l"))
    a = ap.parse_args()
    pool = Pool(a.workers, a.device, a.texts)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"workers": len(pool.procs), "alive": sum(p.is_alive() for p in pool.procs), "pending": len(pool.waiting)})
            else:
                self.reply(404, {"error": "GET /health only"})

        def do_POST(self):
            if self.path != "/score":
                return self.reply(404, {"error": "POST /score only"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(200, {"scores": pool.submit(req).result()})
            except Exception as e:
                self.reply(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    print(f"serve_scores: {a.workers} scoring processes, http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
