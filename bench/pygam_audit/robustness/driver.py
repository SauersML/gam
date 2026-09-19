import json, os, subprocess, sys
from cases import CASES

PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bvenv/bin/python"))
HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = int(os.environ.get("CASE_TIMEOUT", "180"))
names = sys.argv[1:] or list(CASES)
env = dict(os.environ, RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
results = {}
path = os.path.join(HERE, "results.json")
if os.path.exists(path):
    results = json.load(open(path))
for n in names:
    results.setdefault(n, {})
    for lib in ("gamfit", "pygam"):
        try:
            p = subprocess.run([PY, "run_one.py", lib, n], cwd=HERE, capture_output=True,
                               text=True, timeout=TIMEOUT, env=env)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                r = json.loads(line[-1][7:])
            else:
                r = {"fit": "CRASH", "rc": p.returncode, "stderr": p.stderr[-800:]}
        except subprocess.TimeoutExpired:
            r = {"fit": "HANG", "timeout": TIMEOUT}
        results[n][lib] = r
        print(n, lib, json.dumps(r, default=str)[:1500], flush=True)
    json.dump(results, open(path, "w"), indent=1, default=str)
