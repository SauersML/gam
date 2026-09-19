"""Benchmark driver: every measurement in a fresh subprocess, >=3 reps.

Usage: driver.py OUT.jsonl PLAN [--reps 3] [--timeout 600] [--memcap-mb 5000] [--taskset CPUS]
  PLAN is a python-literal list of (lib, family, n, design) tuples, or one of the
  named plans in PLANS below.

Each rep records loadavg / nproc at start and end, wall time, the worker's
JSON (phase timings, ru_maxrss), the driver-polled max RSS and max thread count,
and a status: ok | timeout | memcap | error.  If a (lib,family,design) config
fails with timeout/memcap at some n, larger n for that config are skipped and
recorded as status=skipped_after_<status>.
"""
import ast
import json
import os
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NS = [1000, 10000, 100000, 1000000]

PLANS = {}
PLANS["gaussian_all"] = [(lib, "gaussian", n, d) for d in ["p1", "p5", "p20", "te"]
                         for n in NS for lib in ["gamfit", "gamfit_k20", "pygam", "pygam_gs"]]
PLANS["glm_all"] = [(lib, fam, n, d) for fam in ["binomial", "poisson"] for d in ["p1", "p5", "p20", "te"]
                    for n in NS for lib in ["gamfit", "gamfit_k20", "pygam", "pygam_gs"]]
LIBS = ["gamfit", "gamfit_k20", "pygam", "pygam_gs"]
FAMS = ["gaussian", "binomial", "poisson"]
DES = ["p1", "p5", "p20", "te"]
# Split plans so a loaded shared box can run them in pieces.
for _n, _tag in [(1000, "n1e3"), (10000, "n1e4"), (100000, "n1e5"), (1000000, "n1e6")]:
    PLANS[_tag] = [(lib, fam, _n, d) for fam in FAMS for d in DES for lib in LIBS]
# n=1e5 without the (more expensive) matched-k gamfit variant except for p1/te
PLANS["n1e5_core"] = [(lib, fam, 100000, d) for fam in FAMS for d in DES
                      for lib in (LIBS if d in ("p1", "te") else ["gamfit", "pygam", "pygam_gs"])]
PLANS["n1e4_core"] = [(lib, fam, 10000, d) for fam in FAMS for d in ["p1", "p5", "te"]
                      for lib in ["gamfit", "pygam", "pygam_gs"]]
PLANS["n1e6_core"] = [(lib, fam, 1000000, d) for fam in FAMS for d in ["p1", "p5"]
                      for lib in ["gamfit", "pygam"]]


def run_one(cfg, seed, timeout, memcap_mb, taskset, env_extra):
    lib, fam, n, design = cfg
    cmd = [PY, os.path.join(HERE, "worker.py"), lib, fam, str(n), design, str(seed)]
    if taskset:
        cmd = ["taskset", "-c", taskset] + cmd
    env = dict(os.environ)
    env.update(env_extra)
    rec = dict(lib=lib, family=fam, n=n, design=design, seed=seed, taskset=taskset,
               nproc=os.cpu_count(), load_start=os.getloadavg(), env_extra=env_extra)
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                         cwd="/tmp")
    ps = psutil.Process(p.pid)
    max_rss = 0.0
    max_thr = 0
    status = None
    while p.poll() is None:
        try:
            procs = [ps] + ps.children(recursive=True)
            rss = sum(q.memory_info().rss for q in procs) / 2**20
            thr = sum(q.num_threads() for q in procs)
            max_rss = max(max_rss, rss)
            max_thr = max(max_thr, thr)
        except psutil.Error:
            pass
        if max_rss > memcap_mb:
            status = "memcap"
            p.kill()
            break
        if time.perf_counter() - t0 > timeout:
            status = "timeout"
            p.kill()
            break
        time.sleep(0.05)
    out, err = p.communicate()
    rec["wall_s"] = time.perf_counter() - t0
    rec["load_end"] = os.getloadavg()
    rec["poll_max_rss_mb"] = max_rss
    rec["poll_max_threads"] = max_thr
    res = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if res is not None:
        rec.update({k: v for k, v in res.items() if k not in rec})
        status = status or "ok"
    else:
        status = status or "error"
        rec["stderr_tail"] = err[-2000:]
    rec["status"] = status
    # Count noisy stderr lines (gamfit prints solver chatter)
    rec["stderr_lines"] = len(err.splitlines())
    return rec


def main():
    args = sys.argv[1:]
    outp = args[0]
    plan = args[1]
    reps = 3
    timeout = 600
    memcap = 5000
    taskset = None
    env_extra = {}
    i = 2
    while i < len(args):
        if args[i] == "--reps":
            reps = int(args[i + 1]); i += 2
        elif args[i] == "--timeout":
            timeout = float(args[i + 1]); i += 2
        elif args[i] == "--memcap-mb":
            memcap = float(args[i + 1]); i += 2
        elif args[i] == "--taskset":
            taskset = args[i + 1]; i += 2
        elif args[i] == "--env":
            k, v = args[i + 1].split("=", 1); env_extra[k] = v; i += 2
        else:
            raise SystemExit(f"bad arg {args[i]}")
    cfgs = PLANS[plan] if plan in PLANS else [tuple(c) for c in ast.literal_eval(plan)]
    failed = {}
    with open(outp, "a") as fh:
        for cfg in cfgs:
            key = (cfg[0], cfg[1], cfg[3])
            if key in failed and cfg[2] > failed[key][0]:
                rec = dict(lib=cfg[0], family=cfg[1], n=cfg[2], design=cfg[3],
                           status="skipped_after_" + failed[key][1])
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(json.dumps(rec), flush=True)
                continue
            for r in range(reps):
                rec = run_one(cfg, r, timeout, memcap, taskset, env_extra)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                short = {k: rec.get(k) for k in ["lib", "family", "n", "design", "seed", "status", "wall_s",
                                                  "import_s", "fit1_s", "fit2_s", "pred1_s", "peak_rss_mb",
                                                  "rmse_mu", "edf", "load_start"]}
                print(json.dumps(short), flush=True)
                if rec["status"] in ("timeout", "memcap"):
                    failed[key] = (cfg[2], rec["status"])
                    break


if __name__ == "__main__":
    main()
