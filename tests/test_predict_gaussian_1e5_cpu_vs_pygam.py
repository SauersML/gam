"""Gaussian ``s(x)`` predict on 1e5 fresh rows is within 1.2x of pyGAM's CPU.

The pyGAM audit (bench/pygam_audit/speed.md, table 3.3) recorded Gaussian p=1
predict on 1e5 fresh rows at 0.28 s CPU for gamfit against 0.11 s for pyGAM.
Most of the gap was transport: the prediction columns crossed the FFI as one
JSON document that Python decoded, re-encoded, decoded again, and then walked
value by value into arrays. They now cross as float64 arrays.

The measurement follows the audit's method: CPU time (``time.process_time``)
of a predict call on a column mapping of fresh rows, in a fresh single-threaded
process, so the figure is comparable to the recorded single-threaded pyGAM
number.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

# pyGAM's recorded CPU time for Gaussian p=1 predict on 1e5 fresh rows
# (bench/pygam_audit/speed.md table 3.3) and the allowed ratio to it.
PYGAM_PREDICT_CPU_S = 0.11
ALLOWED_RATIO_TO_PYGAM = 1.2
PREDICT_ROWS = 100_000

_WORKER = textwrap.dedent(
    """
    import json, sys, time
    import numpy as np
    import gamfit

    rows = int(sys.argv[1])
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 10_000)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.5, x.size)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="gaussian")
    fresh = {"x": rng.uniform(0.0, 1.0, rows)}
    cpu = []
    for _ in range(3):
        start = time.process_time()
        prediction = model.predict(fresh)
        cpu.append(time.process_time() - start)
    assert prediction.shape == (rows,)
    print(json.dumps({"cpu_s": min(cpu)}))
    """
)


def test_gaussian_predict_1e5_rows_cpu_is_within_ratio_of_pygam() -> None:
    env = dict(os.environ)
    for name in ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER, str(PREDICT_ROWS)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    cpu_s = json.loads(completed.stdout.strip().splitlines()[-1])["cpu_s"]
    budget = ALLOWED_RATIO_TO_PYGAM * PYGAM_PREDICT_CPU_S
    assert cpu_s <= budget, (
        f"Gaussian predict on {PREDICT_ROWS} rows took {cpu_s:.3f} s CPU; "
        f"allowed {budget:.3f} s ({ALLOWED_RATIO_TO_PYGAM}x pyGAM's {PYGAM_PREDICT_CPU_S} s)"
    )
