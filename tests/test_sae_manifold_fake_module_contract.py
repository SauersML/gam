"""RED test for bug J: SAE manifold fake Rust modules use old method names.

Production `gamfit/_sae_manifold.py:432` calls
`rust_module().sae_manifold_fit_minimal(...)`. The fakes in
`tests/test_sae_manifold_softmax_dispatch.py::_FakeRustModule` and
`tests/test_sae_manifold_ibp_refresh.py::_FakeRustModule` only expose
`sae_manifold_fit` (and `sae_manifold_fit_ibp`). They never wire
`sae_manifold_fit_minimal` (nor the `sae_manifold_fit_auto` underlying that the
real bridge would alias from).

Production callsite: `gamfit/_sae_manifold.py:432`.
Bridge alias install: `gamfit/_binding.py:45-107` — only triggers if the module
exposes `sae_manifold_fit_auto`.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
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
        f"(production gamfit/_sae_manifold.py:432 calls it). Currently exposes: "
        f"{sorted(n for n in dir(fake) if not n.startswith('_'))}."
    )


@pytest.mark.parametrize("path", FAKE_TEST_FILES, ids=lambda p: p.name)
def test_existing_fake_module_tests_pass(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q", "--no-header"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{path.name} is failing.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
