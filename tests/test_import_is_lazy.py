"""``import gamfit`` defers work that a fit does not need.

The package version is read from the installed distribution metadata, and
loading ``importlib.metadata`` (plus the distribution scan behind
``metadata.version``) is a fixed import-time cost on every process. It is paid
lazily, the first time ``gamfit.__version__`` is read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import gamfit

_PROBE = """
import json, sys
import numpy  # gamfit's own import-time dependency, loaded first
before = set(sys.modules)
import gamfit
loaded_by_import = sorted(set(sys.modules) - before)
version = gamfit.__version__
print(json.dumps({
    "loaded_by_import": loaded_by_import,
    "version": version,
    "cached": "__version__" in vars(gamfit),
}))
"""


def _probe() -> dict[str, object]:
    # Run from the directory that holds the package under test so the child
    # imports the same gamfit as this process.
    package_parent = Path(gamfit.__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=package_parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_does_not_load_distribution_metadata() -> None:
    result = _probe()
    loaded = result["loaded_by_import"]
    assert isinstance(loaded, list)
    assert "importlib.metadata" not in loaded


def test_version_resolves_on_first_access_and_is_cached() -> None:
    result = _probe()
    assert isinstance(result["version"], str) and result["version"]
    assert result["cached"] is True
    assert gamfit.__version__ == result["version"]
