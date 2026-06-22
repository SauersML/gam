from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

def test_multinomial_summary_norm_strides_correctly() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.0, 1.0, 300)
    cls = rng.choice(["A", "B", "C"], size=300)
    df = pd.DataFrame({"x": x, "y": cls})
    model = gamfit.fit(df, "y ~ x", family="multinomial")
    
    # Overwrite the coefficients with a known matrix whose contiguous and
    # strided slices produce visibly different norms.
    # P = 2 (intercept, x), K = 3 (m = 2 active classes).
    # Matrix shape (P, m):
    # [[ 1.0, 10.0],
    #  [ 2.0, 20.0]]
    # Flat row-major: [1.0, 10.0, 2.0, 20.0]
    model._metadata["coefficients_flat"] = [1.0, 10.0, 2.0, 20.0]
    
    summary = model.summary()
    
    # If correctly strided (column-wise extraction):
    # Class 0: [1.0, 2.0] -> norm = sqrt(1+4) = sqrt(5) = 2.236
    # Class 1: [10.0, 20.0] -> norm = sqrt(100+400) = sqrt(500) = 22.36
    #
    # If incorrectly contiguous:
    # Class 0: [1.0, 10.0] -> norm = sqrt(101) = 10.05
    # Class 1: [2.0, 20.0] -> norm = sqrt(404) = 20.1
    
    assert "‖β_a‖₂ = 2.236" in summary
    assert "‖β_a‖₂ = 22.36" in summary
    assert "10.05" not in summary
    assert "20.1" not in summary
