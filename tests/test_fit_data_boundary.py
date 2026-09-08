import numpy as np
import pandas as pd
import pytest

import gamfit


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ([0.0, np.nan, 1.0], "non-finite value NaN"),
        ([0.0, np.inf, 1.0], "non-finite value inf"),
        ([0.0, -np.inf, 1.0], "non-finite value -inf"),
        ([4.0, 4.0, 4.0], "is constant"),
        ([np.nan, 4.0, np.nan], "only one non-missing value"),
    ],
)
def test_fit_rejects_numeric_degeneracy_with_typed_column_error(values, problem):
    frame = pd.DataFrame({"y": [0.0, 1.0, 2.0], "offender": values})
    with pytest.raises(gamfit.DataError, match="column 'offender'.*" + problem):
        gamfit.fit(frame, "y ~ offender", family="gaussian")


@pytest.mark.parametrize(
    "frame, column, problem",
    [
        (pd.DataFrame({"y": [], "x": []}), "<table>", "no observations"),
        (pd.DataFrame([[0, 1, 2], [1, 2, 3]], columns=["y", "x", "x"]), "x", "duplicate name"),
        (pd.DataFrame({"y": [0.0, 1.0, 2.0], "group": pd.Categorical(["only"] * 3)}), "group", "fewer than two levels"),
    ],
)
def test_fit_rejects_structural_degeneracy_with_typed_column_error(frame, column, problem):
    with pytest.raises(gamfit.DataError, match=f"column '{column}'.*{problem}"):
        gamfit.fit(frame, "y ~ group" if "group" in frame.columns else "y ~ x", family="gaussian")
