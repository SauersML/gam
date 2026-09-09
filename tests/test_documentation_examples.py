"""Executable contracts for the Markdown examples shipped to users.

A ``python`` fence is executable documentation, not pseudocode.  API signatures
belong in ``text`` fences.  Examples which require external models, accelerators,
or large user-owned activation arrays use ``python no-exec`` and therefore are
not executable examples.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
FENCE = re.compile(r"^```(?P<language>python|bash|sh)[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def examples(language: str):
    for path in DOCS:
        source = path.read_text(encoding="utf-8")
        for match in FENCE.finditer(source):
            if match["language"] != language:
                continue
            line = source.count("\n", 0, match.start()) + 2
            yield pytest.param(path, line, match["body"], id=f"{path.relative_to(ROOT)}:{line}")


def synthetic_data() -> pd.DataFrame:
    """One small, deterministic frame covering names used by introductory snippets."""
    n = 40
    x = np.linspace(0.1, 4.0, n)
    frame = pd.DataFrame({
        "x": x, "x1": x, "x2": np.sin(x), "x3": np.cos(x), "x4": x / 4,
        "age": 30 + x * 10, "bmi": 20 + x, "hba1c": 4 + x / 2,
        "dose": x / 4, "prop": x / 5, "dow": np.arange(n) % 7,
        "hour": np.arange(n) % 24, "theta": np.linspace(0, 2 * np.pi, n, endpoint=False),
        "lat": np.linspace(-1, 1, n), "lon": np.linspace(-3, 3, n),
        "pc1": x, "pc2": np.sin(x), "pc3": np.cos(x), "pc4": x * x / 16,
        "raw_score": np.sin(x) + x / 4, "PGS": np.sin(x) + x / 4,
        "pgs": np.sin(x) + x / 4, "z": np.sin(x), "prev": np.linspace(.1, .9, n),
        "entry": np.zeros(n), "exit": np.arange(1, n + 1, dtype=float),
        "event": (np.arange(n) % 3 == 0).astype(float),
        "case": (np.arange(n) % 2).astype(float), "disease": (np.arange(n) % 2).astype(float),
        "y": 1 + np.sin(x) + x / 3, "outcome": 1 + np.sin(x),
        "site": np.resize(["A", "B", "C"], n), "site_id": np.resize(["A", "B"], n),
        "group": np.resize(["control", "treated"], n), "treatment": np.arange(n) % 2,
        "sand": np.full(n, .2), "silt": np.full(n, .3), "clay": np.full(n, .5),
        "nx": np.sin(x), "ny": np.cos(x), "nz": np.zeros(n),
    })
    return frame


@pytest.fixture(scope="session")
def python_context():
    import gamfit

    df = synthetic_data()
    train = df.copy()
    test = df.head(4).drop(columns=["y", "outcome", "case", "disease", "event"])
    model = gamfit.fit(train, "y ~ x")
    posterior = model.sample(train, samples=20, warmup=20, chains=1, seed=42)
    x_matrix = train[["x", "x2"]]
    return {
        "gamfit": gamfit, "np": np, "pd": pd, "df": df, "data": df,
        "train": train, "train_df": train, "test": test, "test_df": test,
        "cal_df": train.head(10), "model": model, "model_a": model, "model_b": model,
        "loaded": model, "posterior": posterior,
        "X": x_matrix, "X_train": x_matrix.to_numpy(), "X_test": x_matrix.head(4).to_numpy(),
        "y": train["y"].to_numpy(),
        "consume": lambda value: None, "process": lambda value: None,
    }


@pytest.mark.parametrize("path,line,code", list(examples("python")))
def test_python_documentation_example(path, line, code, python_context, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = dict(python_context)
    compiled = compile(code, f"{path.relative_to(ROOT)}:{line}", "exec")
    exec(compiled, namespace)


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
            if parameter.values[2].lstrip().startswith("gam "):
                yield parameter


@pytest.mark.parametrize("path,line,code", list(cli_examples()))
def test_cli_documentation_example(path, line, code, tmp_path, python_context):
    frame = python_context["train"]
    for name in ("train.csv", "data.csv", "new.csv", "new_data.csv", "labelled.csv"):
        frame.to_csv(tmp_path / name, index=False)
    for name in ("model.gam", "model.json"):
        python_context["model"].save(tmp_path / name)
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
