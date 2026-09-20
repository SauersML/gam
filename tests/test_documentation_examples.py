"""Executable contracts for the Markdown examples shipped to users.

A ``python`` fence is executable documentation, not pseudocode.  API signatures
belong in ``text`` fences.  Examples which require external models, accelerators,
or large user-owned activation arrays use ``python no-exec`` and therefore are
not executable examples; their number is capped so the tag stays an exception.

Every ``python`` fence runs in a fresh namespace holding only the builtins, as
it would when a reader pastes it into a new interpreter: it imports, loads or
simulates everything it uses.  The harness injects no data and no model.
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
READMES = (ROOT / "README.md", ROOT / "README_PYPI.md")
DOCS = (*READMES, *sorted((ROOT / "docs").glob("*.md")))
FENCE = re.compile(r"^```(?P<language>python|bash|sh)[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
# Every fence opening whose info string names Python, at any indentation.
PYTHON_INFO = re.compile(r"^(?P<indent>[ \t]*)```(?P<info>py(?:thon|con)?\b[^\n]*)$", re.MULTILINE)
NOT_EXECUTED = "python no-exec"
# The fences that cannot run here: pyGAM itself, external models, accelerators
# and user-owned activation arrays.  Raising this number needs a reason.
NOT_EXECUTED_CAP = 25


def examples(language: str):
    for path in DOCS:
        source = path.read_text(encoding="utf-8")
        for match in FENCE.finditer(source):
            if match["language"] != language:
                continue
            line = source.count("\n", 0, match.start()) + 2
            yield pytest.param(path, line, match["body"], id=f"{path.relative_to(ROOT)}:{line}")


def python_fence_infos():
    for path in DOCS:
        source = path.read_text(encoding="utf-8")
        for match in PYTHON_INFO.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            yield f"{path.relative_to(ROOT)}:{line}", match["indent"], match["info"].rstrip()


def test_python_fences_are_executed_or_explicitly_not():
    """A Python fence either runs or says it does not, and few say it does not.

    An unrecognised info string such as ``python skip``, or an indented
    ``python`` fence, would otherwise drop an example from the run without
    anyone deciding so.
    """
    infos = list(python_fence_infos())
    unknown = [
        (where, indent + info) for where, indent, info in infos
        if info != NOT_EXECUTED and (info != "python" or indent)
    ]
    assert not unknown, f"python fences that neither run nor say they do not: {unknown}"
    not_executed = [where for where, _, info in infos if info == NOT_EXECUTED]
    assert len(not_executed) <= NOT_EXECUTED_CAP, (
        f"{len(not_executed)} `{NOT_EXECUTED}` fences exceed the cap of {NOT_EXECUTED_CAP}; "
        f"make the new ones runnable: {not_executed}"
    )


@pytest.mark.parametrize("path,line,code", list(examples("python")))
def test_python_documentation_example(path, line, code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__", "__builtins__": builtins}
    compiled = compile(code, f"{path.relative_to(ROOT)}:{line}", "exec")
    exec(compiled, namespace)


def synthetic_data() -> pd.DataFrame:
    """One small, deterministic frame behind the CSV files the CLI examples read.

    Every response carries signal and noise, so an example's fit has something to
    identify. A noiseless response, a binary label with no covariate effect or a
    constant composition makes an example refuse because of the data, not because
    of the API it documents.
    """
    n = 120
    rng = np.random.default_rng(20260917)
    x = np.linspace(0.1, 4.0, n)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    risk = 1.0 / (1.0 + np.exp(-(x - 2.0)))
    sand = 0.2 + 0.08 * np.sin(x) + rng.uniform(0.0, 0.02, n)
    silt = 0.3 + 0.08 * np.cos(x) + rng.uniform(0.0, 0.02, n)
    direction = np.column_stack([np.sin(theta), np.cos(theta), 0.4 * np.sin(x)])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    frame = pd.DataFrame({
        "x": x, "x1": x, "x2": np.sin(x), "x3": np.cos(x), "x4": x / 4,
        "age": 30 + x * 10, "bmi": 20 + x, "hba1c": 4 + x / 2,
        "dose": x / 4, "prop": x / 5, "dow": np.arange(n) % 7,
        "hour": np.arange(n) % 24, "theta": theta, "h": x / 4,
        "u": theta, "v": np.resize(theta[::-1], n), "space": x, "time": np.arange(n) % 12,
        "lat": np.linspace(-1, 1, n), "lon": np.linspace(-3, 3, n),
        "pc1": x, "pc2": np.sin(x), "pc3": np.cos(x), "pc4": x * x / 16,
        "raw_score": np.sin(x) + x / 4 + rng.normal(0.0, 0.3, n),
        "PGS": np.sin(x) + x / 4 + rng.normal(0.0, 0.3, n),
        "pgs": np.sin(x) + x / 4 + rng.normal(0.0, 0.3, n),
        "z": np.sin(x) + rng.normal(0.0, 0.2, n), "prev": np.linspace(.1, .9, n),
        "entry": np.zeros(n), "exit": 1.0 + rng.exponential(8.0 / (0.5 + risk)),
        "event": (rng.uniform(size=n) < 0.7).astype(float),
        "case": (rng.uniform(size=n) < risk).astype(float),
        "disease": (rng.uniform(size=n) < risk).astype(float),
        "y": 1 + np.sin(x) + x / 3 + rng.normal(0.0, 0.3, n),
        "outcome": 1 + np.sin(x) + rng.normal(0.0, 0.3, n),
        "site": np.resize(["A", "B", "C"], n), "site_id": np.resize(["A", "B"], n),
        "family_id": np.arange(n) % 10,
        "group": np.resize(["control", "treated"], n), "treatment": np.arange(n) % 2,
        "sand": sand, "silt": silt, "clay": 1.0 - sand - silt,
        "nx": direction[:, 0], "ny": direction[:, 1], "nz": direction[:, 2],
    })
    # Drawn after every column above, so adding a name never moves an existing column's values.
    exposure = rng.uniform(0.5, 2.0, n)
    nb_mean = np.exp(1.0 + 0.3 * np.cos(x))
    frame["patient"] = [f"P{row:03d}" for row in range(n)]
    frame["death"] = (rng.uniform(size=n) < 0.2 + 0.4 * (1.0 - risk)).astype(float)
    frame["cause"] = np.where(frame["event"] > 0, np.where(rng.uniform(size=n) < risk, 1.0, 2.0), 0.0)
    frame["log_exposure"] = np.log(exposure)
    frame["count"] = rng.poisson(exposure * np.exp(0.5 + 0.4 * np.sin(x))).astype(float)
    frame["freq"] = rng.integers(1, 4, n).astype(float)
    frame["rate"] = rng.negative_binomial(5, 5.0 / (5.0 + nb_mean)).astype(float)
    frame["claim"] = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.gamma(2.0, np.exp(0.2 * x) / 2.0))
    frame["year"] = 2000.0 + np.arange(n) % 12
    frame["rare_event"] = (rng.uniform(size=n) < 0.2 * risk).astype(float)
    frame["left"] = frame["exit"] * rng.uniform(0.3, 0.9, n)
    frame["right"] = frame["exit"]
    return frame


@pytest.fixture(scope="session")
def cli_inputs():
    """The rows and the saved model behind the files the CLI examples name."""
    import gamfit

    frame = synthetic_data()
    return frame, gamfit.fit(frame, "y ~ x")


def _cli_binary() -> Path:
    configured = os.environ.get("GAM_CLI_BIN")
    candidates = [Path(configured)] if configured else []
    candidates += [ROOT / "target/release/gam", ROOT / "target/debug/gam"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AssertionError("build the CLI first with `cargo build --release -p gam-cli`")


def cli_examples():
    for language in ("bash", "sh"):
        for parameter in examples(language):
            # pytest.param stores positional values in ``values``.
            path, line, code = parameter.values
            if code.lstrip().startswith("gam "):
                yield pytest.param(path, line, code, id=parameter.id)


@pytest.mark.parametrize("path,line,code", list(cli_examples()))
def test_cli_documentation_example(path, line, code, tmp_path, cli_inputs):
    frame, model = cli_inputs
    for name in ("train.csv", "data.csv", "new.csv", "new_data.csv", "labelled.csv", "held_out.csv"):
        frame.to_csv(tmp_path / name, index=False)
    for name in ("model.gam", "model.json"):
        model.save(tmp_path / name)
    np.save(tmp_path / "layer0.npy", np.arange(48, dtype=float).reshape(12, 4))
    np.save(tmp_path / "layer1.npy", np.arange(48, dtype=float).reshape(12, 4) + 0.1)
    pd.DataFrame({"id": ["subject-17", "subject-18"], "entry": [0.0, 0.0], "exit": [5.0, 5.0]}).to_csv(
        tmp_path / "s.csv", index=False
    )
    pd.DataFrame({
        "id": ["subject-17", "subject-17", "subject-18"],
        "time": [1.0, 4.0, 2.0], "mark": ["relapse", "death", "death"],
    }).to_csv(tmp_path / "e.csv", index=False)
    pd.DataFrame({
        "id": ["subject-17", "subject-18"], "start": [0.0, 0.0], "x": [0.0, 1.0],
    }).to_csv(tmp_path / "c.csv", index=False)
    env = os.environ.copy()
    env["PATH"] = f"{_cli_binary().parent}{os.pathsep}{env.get('PATH', '')}"
    completed = subprocess.run(
        code, shell=True, executable="/bin/bash", cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=120,
    )
    assert completed.returncode == 0, (
        f"{path.relative_to(ROOT)}:{line}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
