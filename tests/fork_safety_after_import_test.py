"""Fits keep working in processes forked after ``import gamfit``.

A ``fork()``ed child inherits the parent's memory but none of its threads. A
worker pool built in the parent is therefore a pool with no workers in the
child, and a fit there used to wait forever for them: every fork-based caller
(a ``multiprocessing`` fork pool, joblib's fork backend, a bare ``os.fork()``)
hung once gamfit was imported. The fit now runs on a pool that belongs to the
current process, so a forked child builds its own.

The scenario runs in a subprocess so that a regression is a bounded failure
instead of a hung test run. Every process also reports the unnamed threads
that appeared while it fitted: there must be none, so no computation reached
rayon's global pool or a library's private thread pool (neither of which a
forked child can use) instead of gam's own named workers.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import gamfit
import pytest

# A test-only bound on the whole scenario. A fixed fork scenario of small fits
# finishes in seconds; the regression this guards against never finishes.
HANG_BOUND_SECONDS = 300

_SCENARIO = r'''
import json, math, multiprocessing, os, sys
import gamfit


# Enough rows that the fit's dense products are large enough for a library to
# split them over threads of its own (matrixmultiply's thread tree did).
ROWS = 2000

with open("/proc/self/comm") as handle:
    INTERPRETER_THREAD = handle.read().strip()


def frame():
    out = {"x": [], "y": []}
    for i in range(ROWS):
        x = -2.4 + 4.8 * i / (ROWS - 1)
        out["x"].append(x)
        out["y"].append(math.sin(2.1 * x) + 0.07 * ((i * 37 % 17) - 8.0))
    return out


def thread_names():
    names = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as handle:
                names[tid] = handle.read().strip()
        except FileNotFoundError:
            pass
    return names


def fit_once(_=None):
    before = thread_names()
    data = frame()
    model = gamfit.fit(data, "y ~ s(x, k=10)")
    coef = model.summary().coefficients_frame()["estimate"].tolist()
    prediction = [float(v) for v in model.predict(data)]
    # Threads that the fit started without naming them. The process pool
    # names each of its workers; a thread nobody names (rayon's global pool, a
    # library's private pool) keeps the name of the thread that started it, so
    # it shows up under the interpreter's name or as a second copy of a gam
    # worker's name. Threads that a library names itself (an allocator's
    # background thread) are that library's to manage across fork.
    after = thread_names()
    counts = {}
    for name in after.values():
        counts[name] = counts.get(name, 0) + 1
    unnamed = sorted(
        name
        for tid, name in after.items()
        if tid not in before
        and (name == INTERPRETER_THREAD or (name.startswith("gam-") and counts[name] > 1))
    )
    return {
        "fit": {
            "coefficients": [v.hex() for v in coef],
            "prediction": [v.hex() for v in prediction],
        },
        "unnamed_threads": unnamed,
    }


def fit_in_forked_child():
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_end)
        status = 1
        try:
            with os.fdopen(write_end, "w") as out:
                json.dump(fit_once(), out)
            status = 0
        finally:
            os._exit(status)
    os.close(write_end)
    with os.fdopen(read_end) as source:
        payload = source.read()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, "forked child failed"
    return json.loads(payload)


if __name__ == "__main__":
    report = {}
    # Forked before this process has fitted anything: import alone must not
    # leave state a child cannot use.
    report["fork_before_parent_fit"] = fit_in_forked_child()
    report["parent"] = fit_once()
    with multiprocessing.get_context("fork").Pool(2) as pool:
        report["fork_pool"] = pool.map(fit_once, range(2))
    report["os_fork"] = fit_in_forked_child()
    report["parent_again"] = fit_once()
    print("RESULT " + json.dumps(report, sort_keys=True), flush=True)
'''


def _run_scenario(tmp_path: Path) -> dict[str, Any]:
    script = tmp_path / "fork_scenario.py"
    script.write_text(_SCENARIO)
    # Run next to the package under test so the child imports the same gamfit.
    package_parent = Path(gamfit.__file__).resolve().parent.parent
    child = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=package_parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=HANG_BOUND_SECONDS)
    except subprocess.TimeoutExpired:
        # Take the forked children down with the scenario process.
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate()
        pytest.fail(
            f"fitting in forked processes did not finish in {HANG_BOUND_SECONDS}s "
            f"(deadlock after fork)\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    assert child.returncode == 0, f"scenario failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    line = next(line for line in stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line.removeprefix("RESULT "))


def test_fits_in_forked_children_match_the_parent(tmp_path: Path) -> None:
    report = _run_scenario(tmp_path)
    runs = {
        "fork_before_parent_fit": report["fork_before_parent_fit"],
        "parent": report["parent"],
        "fork_pool[0]": report["fork_pool"][0],
        "fork_pool[1]": report["fork_pool"][1],
        "os_fork": report["os_fork"],
        "parent_again": report["parent_again"],
    }
    expected = report["parent"]["fit"]
    for name, run in runs.items():
        assert run["fit"] == expected, f"{name} fit differs from the parent's"
        unnamed = run["unnamed_threads"]
        assert not unnamed, f"{name} started threads outside gam's pool: {unnamed}"
