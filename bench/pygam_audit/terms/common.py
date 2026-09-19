import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import gamfit


def gpred(m, df):
    out = m.predict(df)
    if isinstance(out, np.ndarray):
        return out
    if isinstance(out, pd.DataFrame):
        for c in ("mean", "eta", "fit", "prediction"):
            if c in out.columns:
                return out[c].to_numpy()
        return out.iloc[:, 0].to_numpy()
    if isinstance(out, dict):
        for c in ("mean", "eta"):
            if c in out:
                return np.asarray(out[c])
    try:
        return np.asarray(out.mean)
    except AttributeError:
        return np.asarray(out)


def check_shape(f, grid, kind):
    """Return worst violation (positive = violated) on dense grid."""
    d = np.diff(f)
    if kind == "inc":
        return max(0.0, -d.min())
    if kind == "dec":
        return max(0.0, d.max())
    d2 = np.diff(f, 2)
    if kind == "convex":
        return max(0.0, -d2.min())
    if kind == "concave":
        return max(0.0, d2.max())
