"""Recognising the input table kind must not import a dataframe library.

`detect_table_kind` used to `import pandas`, `polars` and `pyarrow` to run its
`isinstance` checks, so the first fit on a dict or numpy table paid a cold
pandas import (~0.3 s, several times the whole n=1000 fit). An object of a
library's class can exist only once that library is imported, so the probe
reads `sys.modules` instead. The check runs in a fresh interpreter: the test
session itself may already have pandas loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys

_PROBE = r"""
import json, sys
import numpy as np
import gamfit
from gamfit._tables import detect_table_kind

libraries = ("pandas", "polars", "pyarrow")
before = {name for name in libraries if name in sys.modules}
rng = np.random.default_rng(0)
x = rng.uniform(0.0, 1.0, 200)
y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, 200)
kinds = [detect_table_kind({"x": x, "y": y}), detect_table_kind(np.column_stack([x, y]))]
gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="gaussian")
after = {name for name in libraries if name in sys.modules}
print(json.dumps({"kinds": kinds, "imported": sorted(after - before)}))
"""


def _run_probe() -> dict[str, object]:
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _PROBE],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_dict_and_numpy_tables_fit_without_importing_dataframe_libraries() -> None:
    probe = _run_probe()
    assert probe["kinds"] == ["unknown", "numpy"]
    assert probe["imported"] == [], (
        f"fitting a dict table imported {probe['imported']}"
    )


def test_loaded_library_tables_are_still_recognised() -> None:
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    from gamfit._tables import detect_table_kind

    assert detect_table_kind(pd.DataFrame({"x": [1.0]})) == "pandas"
    assert detect_table_kind(pa.table({"x": [1.0]})) == "pyarrow"
    assert detect_table_kind(np.zeros((3, 2))) == "numpy"
    assert detect_table_kind({"x": [1.0]}) == "unknown"
    assert detect_table_kind(None) == "unknown"
