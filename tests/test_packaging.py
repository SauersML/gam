"""Packaging discipline tests for the gamfit PyPI wheel.

These are RED tests that lock down the public contract that `pip install gamfit`
without any extras should produce a small, importable package. They cover:

  * Issue #218 — numpy is required by top-level `import gamfit` and must be a
    base dependency, not an extra.
  * Issue #220 — the six `nvidia-*-cu12` packages are >1.4 GB of mandatory
    downloads on Linux x86_64 and must live behind an opt-in extra
    (e.g. `gamfit[cuda]`), not in base `[project] dependencies`.
  * Family check — no other heavy optional runtime (pyarrow, polars, pandas,
    matplotlib, scikit-learn, torch) is imported unconditionally on the
    `import gamfit` path, so the contract above stays meaningful.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
GAMFIT_PKG = REPO_ROOT / "gamfit"


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _base_deps() -> list[str]:
    return list(_load_pyproject()["project"].get("dependencies", []))


def _extras() -> dict:
    return dict(_load_pyproject()["project"].get("optional-dependencies", {}))


def _req_name(spec: str) -> str:
    # strip env markers and version pin, lowercase, normalize underscores.
    bare = spec.split(";", 1)[0].strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        bare = bare.split(sep, 1)[0]
    return bare.strip().lower().replace("_", "-")


# --- Issue #218 --------------------------------------------------------------


def test_numpy_is_a_base_dependency():
    """`import gamfit` requires numpy unconditionally, so it must be base."""
    names = {_req_name(s) for s in _base_deps()}
    assert "numpy" in names, (
        "numpy must appear in [project].dependencies because "
        "gamfit/__init__.py -> gamfit/_penalties.py imports numpy at "
        "module load. Found base deps: " + ", ".join(sorted(names))
    )


# --- Issue #220 --------------------------------------------------------------


NVIDIA_CUDA_PACKAGES = (
    "nvidia-cublas-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nvjitlink-cu12",
)


def test_no_cuda_wheels_in_base_dependencies():
    """The 1.4GB nvidia-*-cu12 stack must not be mandatory."""
    base_names = {_req_name(s) for s in _base_deps()}
    leaked = sorted(base_names & set(NVIDIA_CUDA_PACKAGES))
    assert not leaked, (
        "These CUDA wheels are in base [project].dependencies and force "
        "every Linux x86_64 install to download >1.4 GB of nvidia-*-cu12 "
        "wheels (see issue #220). Move them to an opt-in extra such as "
        "[project.optional-dependencies].cuda. Leaked: " + ", ".join(leaked)
    )


def test_cuda_extra_exists_and_lists_the_cuda_wheels():
    """Once CUDA is moved out of base, it must still be installable as an extra."""
    extras = _extras()
    assert "cuda" in extras, (
        "Expected a [project.optional-dependencies].cuda extra so users can "
        "opt into the nvidia-*-cu12 stack via `pip install gamfit[cuda]`. "
        "Found extras: " + ", ".join(sorted(extras))
    )
    cuda_names = {_req_name(s) for s in extras["cuda"]}
    missing = sorted(set(NVIDIA_CUDA_PACKAGES) - cuda_names)
    assert not missing, (
        "[project.optional-dependencies].cuda is missing wheels that used to "
        "be mandatory: " + ", ".join(missing)
    )


# --- Family check: lazy-import discipline ------------------------------------

# Modules that `import gamfit` reaches at load time via the chain in
# gamfit/__init__.py. Keep this list narrow — adding a heavy import below
# without also adding the dep to base is what got us into #218 in the first
# place. The intent: if any of these starts importing pyarrow/polars/etc.
# unconditionally, this test forces a deliberate choice (move to base deps,
# or lazy-import it).
LOAD_PATH_MODULES = (
    "gamfit/__init__.py",
    "gamfit/_api.py",
    "gamfit/_penalties.py",
    "gamfit/_binding.py",
    "gamfit/_compare.py",
    "gamfit/smooth.py",
    "gamfit/topology.py",
    "gamfit/_basis_descriptors.py",
    "gamfit/_basis_eval.py",
    "gamfit/_basis_protocol.py",
    "gamfit/_composite_penalty.py",
    "gamfit/_diagnostics.py",
    "gamfit/_equivariant.py",
    "gamfit/_protocol.py",
    "gamfit/_select_topology.py",
    "gamfit/_sheaf.py",
    "gamfit/identifiability.py",
    "gamfit/diagnostics.py",
    "gamfit/manifolds.py",
    "gamfit/kernels.py",
)

# Anything in here is a heavy/optional dep. Top-level imports of these in
# load-path modules are forbidden unless the package is also listed in base
# [project].dependencies.
HEAVY_OPTIONAL_DEPS = frozenset(
    {
        "pyarrow",
        "polars",
        "pandas",
        "matplotlib",
        "sklearn",  # the import name for scikit-learn
        "torch",
    }
)


def _top_level_imports(path: Path) -> set[str]:
    """Return the set of top-level module names imported at module load.

    Excludes imports nested inside functions, classes, or `if TYPE_CHECKING`
    blocks — those are lazy and do not affect `import gamfit`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".", 1)[0])
    return found


@pytest.mark.parametrize("relpath", LOAD_PATH_MODULES)
def test_no_unconditional_heavy_optional_imports(relpath: str):
    """No load-path module may import a heavy optional dep at module scope."""
    path = REPO_ROOT / relpath
    if not path.exists():
        pytest.skip(f"{relpath} not present in this checkout")
    imports = _top_level_imports(path)
    base_names = {_req_name(s) for s in _base_deps()}
    offenders = sorted(
        name
        for name in imports & HEAVY_OPTIONAL_DEPS
        # sklearn -> scikit-learn dist name; if user lists scikit-learn in
        # base deps we accept it. Same idea for the others.
        if {
            "pyarrow": "pyarrow",
            "polars": "polars",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "sklearn": "scikit-learn",
            "torch": "torch",
        }[name]
        not in base_names
    )
    assert not offenders, (
        f"{relpath} unconditionally imports heavy optional dep(s) "
        f"{offenders} at module scope. Either lazy-import inside the "
        "function that needs them, or promote the dist to base "
        "[project].dependencies (this is the failure mode behind #218)."
    )


# --- Sanity: the package is importable in this checkout ---------------------


def test_gamfit_package_dir_exists():
    """If this fails the other tests are meaningless — fail loudly."""
    assert GAMFIT_PKG.is_dir(), f"expected {GAMFIT_PKG} to be a directory"
    assert (GAMFIT_PKG / "__init__.py").is_file()


if __name__ == "__main__":  # pragma: no cover - manual invocation aid
    sys.exit(pytest.main([__file__, "-v"]))
