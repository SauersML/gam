"""Executable contracts for the Markdown examples shipped to users.

A ``python`` fence is executable documentation, not pseudocode.  API signatures
belong in ``text`` fences.  Examples which require external models, accelerators,
or large user-owned activation arrays use ``python no-exec`` and therefore are
not executable examples.

Most fences run against a shared namespace of synthetic data (``df``,
``model``, ...).  The pages a reader copies whole, and the first example of each
README, run in an empty namespace instead: they must import and load everything
they use.
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
READMES = (ROOT / "README.md", ROOT / "README_PYPI.md")
DOCS = (*READMES, *sorted((ROOT / "docs").glob("*.md")))
# Every fence on these pages is a complete program.
SELF_CONTAINED_PAGES = frozenset({ROOT / "docs" / "tour.md", ROOT / "docs" / "migrating-from-pygam.md"})
FENCE = re.compile(r"^```(?P<language>python|bash|sh)[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def examples(language: str):
    for path in DOCS:
        source = path.read_text(encoding="utf-8")
        first_python = True
        for match in FENCE.finditer(source):
            if match["language"] != language:
                continue
            self_contained = path in SELF_CONTAINED_PAGES or (path in READMES and first_python)
            first_python = False
            line = source.count("\n", 0, match.start()) + 2
            yield pytest.param(path, line, match["body"], self_contained, id=f"{path.relative_to(ROOT)}:{line}")


def synthetic_data() -> pd.DataFrame:
    """One small, deterministic frame covering names used by introductory snippets.

    Every response carries signal and noise, so a snippet's fit has something to
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
def python_context():
    import gamfit

    df = synthetic_data()
    train = df.copy()
    test = df.head(4).drop(columns=[
        "y", "outcome", "case", "disease", "event", "death", "cause", "count", "rate", "claim", "rare_event",
    ])
    model = gamfit.fit(train, "y ~ x")
    posterior = model.sample(train, samples=20, seed=42)
    x_matrix = train[["x", "x2"]]
    return {
        "gamfit": gamfit, "np": np, "pd": pd, "df": df, "data": df,
        "train": train, "train_df": train, "test": test, "test_df": test,
        "cal_df": train.head(10), "held_out": train.head(10),
        "model": model, "model_a": model, "model_b": model,
        "loaded": model, "posterior": posterior,
        "X": x_matrix, "X_train": x_matrix.to_numpy(), "X_test": x_matrix.head(4).to_numpy(),
        "y": train["y"].to_numpy(),
        "consume": lambda value: None, "process": lambda value: None,
    }


@pytest.mark.parametrize("path,line,code,self_contained", list(examples("python")))
def test_python_documentation_example(path, line, code, self_contained, python_context, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"} if self_contained else dict(python_context)
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
            path, line, code, _ = parameter.values
            if code.lstrip().startswith("gam "):
                yield pytest.param(path, line, code, id=parameter.id)


@pytest.mark.parametrize("path,line,code", list(cli_examples()))
def test_cli_documentation_example(path, line, code, tmp_path, python_context):
    frame = python_context["train"]
    for name in ("train.csv", "data.csv", "new.csv", "new_data.csv", "labelled.csv", "held_out.csv"):
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
