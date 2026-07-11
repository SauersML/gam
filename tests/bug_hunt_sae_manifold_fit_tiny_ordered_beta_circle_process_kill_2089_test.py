"""Bug hunt / regression for issue #2089: a tiny ``sae_manifold_fit`` ordered independent Beta--Bernoulli
circle fit TERMINATED THE HOST PYTHON PROCESS (exit 137 / SIGKILL) instead of
returning a model or raising a Python exception.

The stale-binary failure mode was a K>=2 dictionary co-collapse whose inner fit
diverged (reconstruction EV walking to large negative values across multi-start
reseeds), driving a runaway working set that the OOM reaper killed. A library
must never terminate its host: the call must either complete or raise a
diagnosable Python exception.

Because the historical failure was a *process kill*, this regression MUST run in
an isolated subprocess with a timeout — an in-process assertion cannot survive a
SIGKILL of its own interpreter. The test is fix-agnostic: it asserts only that
the child is NOT killed by a signal (no negative return code, no 137) within the
timeout. A clean completion (rc 0) or a raised Python exception (rc 1 with a
traceback) both pass.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_REPRO = textwrap.dedent(
    """
    import numpy as np
    import gamfit

    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 32))
    try:
        gamfit.sae_manifold_fit(
            X=X,
            K=6,
            d_atom=1,
            atom_topology="circle",
            assignment="ordered_beta_bernoulli",
            n_iter=3,
            sparsity_weight=0.01,
            smoothness_weight=0.01,
            isometry_weight=0.1,
            ard_per_atom=False,
            decoder_incoherence_weight=0.1,
            nuclear_norm_weight=0.0,
            random_state=0,
            alpha="auto",
        )
        print("COMPLETED_OK")
    except Exception as exc:  # a diagnosable Python exception is acceptable
        print("RAISED", type(exc).__name__)
    """
)


def test_tiny_ordered_beta_circle_fit_does_not_kill_the_process() -> None:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _REPRO],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:  # pragma: no cover - hang is also a failure
        pytest.fail("sae_manifold_fit tiny ordered independent Beta--Bernoulli circle fit hung (>300s)")

    assert proc.returncode >= 0, (
        "sae_manifold_fit terminated the host process with signal "
        f"{-proc.returncode}; a library must raise, not kill the interpreter. "
        f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode != 137, (
        "sae_manifold_fit was OOM/SIGKILLed (exit 137) on a tiny circle fit. "
        f"stdout tail:\n{proc.stdout[-2000:]}"
    )
    assert ("COMPLETED_OK" in proc.stdout) or ("RAISED" in proc.stdout), (
        "expected either a completed fit or a raised Python exception; got "
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
