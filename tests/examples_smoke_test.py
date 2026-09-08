"""Keep the user-facing Python examples executable as standalone programs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = tuple(sorted((REPOSITORY_ROOT / "examples").glob("*.py")))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_python_example_runs(example: Path) -> None:
    environment = os.environ.copy()
    # Plotting demonstrations must also run on headless CI workers.
    environment.setdefault("MPLBACKEND", "Agg")

    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{example.name} exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
