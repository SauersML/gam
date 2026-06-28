"""Contract: SAE manifold fake Rust modules expose the production method name.

Production `gamfit/_sae_manifold.py` calls
`rust_module().sae_manifold_fit_minimal(...)` directly (the Rust
`#[pyfunction]` is named `sae_manifold_fit_minimal`; there is no alias
layer). The fakes in
`tests/test_sae_manifold_softmax_dispatch.py::_FakeRustModule` and
`tests/test_sae_manifold_ibp_refresh.py::_FakeRustModule` must therefore wire
`sae_manifold_fit_minimal` with the real signature for the bridge to drive
them.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_TEST_FILES = (
    REPO_ROOT / "tests" / "test_sae_manifold_softmax_dispatch.py",
    REPO_ROOT / "tests" / "test_sae_manifold_ibp_refresh.py",
)


def _load_fake_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_bug_j_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", FAKE_TEST_FILES, ids=lambda p: p.name)
def test_fake_rust_module_exposes_sae_manifold_fit_minimal(path: Path) -> None:
    mod = _load_fake_module(path)
    fake = mod._FakeRustModule()
    assert hasattr(fake, "sae_manifold_fit_minimal"), (
        f"{path.name}::_FakeRustModule must expose `sae_manifold_fit_minimal` "
        f"(production gamfit/_sae_manifold.py calls it). Currently exposes: "
        f"{sorted(n for n in dir(fake) if not n.startswith('_'))}."
    )


@pytest.mark.parametrize("path", FAKE_TEST_FILES, ids=lambda p: p.name)
def test_existing_fake_module_tests_pass(path: Path) -> None:
    # Run from a directory without the source ./gamfit package so the installed
    # compiled wheel (which carries `gamfit._rust`) wins the import, mirroring how
    # the CI `python-tests` job runs pytest from `runner.temp`. Running from
    # REPO_ROOT would let the source ./gamfit (no `_rust`) shadow the wheel and
    # fail with ModuleNotFoundError: gamfit._rust.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q", "--no-header"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{path.name} is failing.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
