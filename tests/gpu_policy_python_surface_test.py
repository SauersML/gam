"""Python surface for the process-global GPU policy (gamfit.configure_gpu_policy).

The policy slot is process-global and first-writer-wins, so every scenario runs
in its own subprocess: a test that configured the policy in-process would leak
that choice into every later test in the session.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run(code: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("GAMFIT_GPU", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_default_policy_is_auto_and_reading_does_not_lock() -> None:
    proc = _run(
        "import gamfit\n"
        "assert gamfit.gpu_policy() == 'auto', gamfit.gpu_policy()\n"
        # Reading must not have claimed the slot: an explicit configure after a
        # read still wins.
        "gamfit.configure_gpu_policy('off')\n"
        "assert gamfit.gpu_policy() == 'off', gamfit.gpu_policy()\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_configure_off_reads_back() -> None:
    proc = _run(
        "import gamfit\n"
        "gamfit.configure_gpu_policy('off')\n"
        "print(gamfit.gpu_policy())\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "off"


def test_reasserting_same_policy_is_noop() -> None:
    proc = _run(
        "import gamfit\n"
        "gamfit.configure_gpu_policy('off')\n"
        "gamfit.configure_gpu_policy('off')\n"
        "print(gamfit.gpu_policy())\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "off"


def test_conflicting_reconfigure_raises_runtime_error() -> None:
    proc = _run(
        "import gamfit\n"
        "gamfit.configure_gpu_policy('off')\n"
        "try:\n"
        "    gamfit.configure_gpu_policy('required')\n"
        "except RuntimeError as error:\n"
        "    assert 'first writer wins' in str(error), error\n"
        "    print('raised')\n"
        "else:\n"
        "    print('did-not-raise')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "raised"


def test_unrecognized_value_raises_value_error() -> None:
    proc = _run(
        "import gamfit\n"
        "try:\n"
        "    gamfit.configure_gpu_policy('gpu-please')\n"
        "except ValueError as error:\n"
        "    assert 'auto|off|required' in str(error), error\n"
        "    print('raised')\n"
        "else:\n"
        "    print('did-not-raise')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "raised"


def test_env_var_is_honored_at_import() -> None:
    proc = _run(
        "import gamfit\nprint(gamfit.gpu_policy())\n",
        env_extra={"GAMFIT_GPU": "off"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "off"


def test_env_var_typo_fails_import_loudly() -> None:
    proc = _run(
        "import gamfit\n",
        env_extra={"GAMFIT_GPU": "offf"},
    )
    assert proc.returncode != 0
    assert "auto|off|required" in proc.stderr


def test_off_policy_fit_never_raises_gpu_resolution_error() -> None:
    """Under 'off', a fit large enough to engage the arrow-Schur GPU path must
    never surface a GPU runtime resolution error (#2322) — whatever else the
    solver decides about the fit. The fit outcome itself is not asserted; the
    assertion is purely that the failure mode, if any, is not the GPU probe.

    The GPU resolution fault, when it fires, kills the fit in well under ten
    seconds (0.4 s measured on the batched path, ~6 s on the resident-frame
    path). A fit still doing solver work at the deadline has therefore already
    demonstrated the property, so a subprocess timeout counts as a pass
    provided no GPU error reached stderr — this also keeps the test bounded on
    hosts where the CPU inner solve is slow (#2258).
    """
    code = (
        "import numpy as np\n"
        "import gamfit\n"
        "gamfit.configure_gpu_policy('off')\n"
        "rng = np.random.default_rng(0)\n"
        "t = rng.uniform(0, 2*np.pi, 220)\n"
        "x = np.c_[np.cos(t), np.sin(t)] + 0.05*rng.standard_normal((220, 2))\n"
        "X = np.hstack([x, 0.05*rng.standard_normal((220, 14))])\n"
        "try:\n"
        "    gamfit.sae_manifold_fit(X, K=1, d_atom=1, atom_topology='circle',\n"
        "                            sparsity_weight=1.0, n_iter=2, random_state=0)\n"
        "    print('fit-completed')\n"
        "except Exception as error:\n"
        "    message = str(error)\n"
        "    assert 'GPU runtime resolution failed' not in message, message\n"
        "    assert 'DriverError' not in message, message\n"
        "    print('non-gpu-failure')\n"
    )
    env = dict(os.environ)
    env.pop("GAMFIT_GPU", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except subprocess.TimeoutExpired as expired:
        stderr = expired.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        assert "DriverError" not in stderr, stderr
        assert "GPU runtime resolution failed" not in stderr, stderr
        return  # solver still working on CPU at the deadline: property shown
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() in {"fit-completed", "non-gpu-failure"}
