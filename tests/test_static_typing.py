"""``mypy --strict`` accepts gamfit and a typical user of it (audit PKG-05).

gamfit ships ``py.typed``, so its annotations are part of the public API: a
strict user must be able to call ``gamfit.fit``, ``gamfit.load`` and every other
name in ``gamfit.__all__`` and its public submodules without "does not explicitly export" or untyped-call
errors. Checking the package itself keeps those annotations honest, and the
deliberately wrong snippet proves the check sees real types rather than ``Any``.

Type-checking the whole package needs the optional frameworks it integrates
with (torch, jax) and the third-party stub packages installed, so these tests
carry the ``static_typing`` marker and run in the dedicated CI lane that
installs them. They do not need the compiled extension: mypy reads
``gamfit/_rust.pyi``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.static_typing

_REPO_ROOT = Path(__file__).resolve().parent.parent
_USAGE_EXAMPLE = Path("tests") / "typing_usage_example.py"


def _mypy(*targets: str) -> subprocess.CompletedProcess[str]:
    # Run from the checkout so ``gamfit`` resolves to the source package, whose
    # stubs are the ones under review; ``pyproject.toml`` supplies the strict
    # configuration, and ``--strict`` is repeated so the check cannot weaken
    # silently if that configuration changes.
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", *targets],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_mypy_strict_accepts_gamfit_and_a_user_of_it() -> None:
    result = _mypy("gamfit", "scripts/gen_rust_stub.py", str(_USAGE_EXAMPLE))
    assert result.returncode == 0, result.stdout + result.stderr


def test_mypy_strict_rejects_a_misuse_of_the_public_api(tmp_path: Path) -> None:
    misuse = tmp_path / "misuse.py"
    misuse.write_text(
        "import gamfit\n"
        "\n"
        "model: gamfit.Model = gamfit.fit({'y': [1.0]}, 3)\n"
        "gamfit.sae.adjudicate_atom_shape(model, folds='five')\n",
        encoding="utf-8",
    )
    result = _mypy(str(misuse))
    assert result.returncode == 1, result.stdout + result.stderr
    # (line, error code) of every error: an integer formula matches no ``fit``
    # overload, and both ``adjudicate_atom_shape`` arguments have the wrong type.
    errors = sorted(
        (int(match.group(1)), match.group(2))
        for match in re.finditer(r"misuse\.py:(\d+): error: .*\[([a-z-]+)\]$", result.stdout, re.M)
    )
    assert errors == [(3, "call-overload"), (4, "arg-type"), (4, "arg-type")], result.stdout
