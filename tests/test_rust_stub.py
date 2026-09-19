"""The ``gamfit._rust`` stub is generated from the Rust source and matches the binary.

``scripts/gen_rust_stub.py`` derives ``gamfit/_rust.pyi`` (and the
``gamfit/_rust_module.pyi`` protocol that ``rust_module()`` returns) from the
PyO3 declarations in ``crates/gam-pyffi``. Two independent checks keep it true:

* the committed stubs equal a fresh generation, so a ``#[pyfunction]`` change
  that is not followed by a regeneration fails here;
* ``mypy.stubtest`` imports the compiled extension and fails on any symbol the
  stub declares but the module lacks, any public symbol the module exports but
  the stub omits, and any signature that disagrees with the runtime one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gamfit
import gamfit._rust  # noqa: F401  - a missing build fails here, not inside stubtest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GENERATOR = _REPO_ROOT / "scripts" / "gen_rust_stub.py"
# The package actually imported: the source tree after ``maturin develop``, the
# installed wheel in CI. Its stub is the one users type-check against.
_PACKAGE_DIR = Path(gamfit.__file__).resolve().parent


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def test_committed_stubs_match_the_rust_source() -> None:
    result = _run(str(_GENERATOR), "--check", cwd=_REPO_ROOT)
    assert result.returncode == 0, (
        "gamfit/_rust.pyi is stale; run `python scripts/gen_rust_stub.py`\n"
        + result.stdout
        + result.stderr
    )


def test_imported_package_ships_its_type_information() -> None:
    for name in ("py.typed", "_rust.pyi", "_rust_module.pyi"):
        assert (_PACKAGE_DIR / name).is_file(), f"{_PACKAGE_DIR} does not ship {name}"


def test_stub_matches_the_compiled_module(tmp_path: Path) -> None:
    # stubtest resolves both the runtime module and its stub from the working
    # directory, so running it next to the imported package checks the pair a
    # user gets. An empty config keeps the repo's mypy settings (and its
    # ``files`` list) out of the comparison.
    empty_config = tmp_path / "mypy.ini"
    empty_config.write_text("[mypy]\n", encoding="utf-8")
    result = _run(
        "-m",
        "mypy.stubtest",
        "gamfit._rust",
        "--mypy-config-file",
        str(empty_config),
        cwd=_PACKAGE_DIR.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr
