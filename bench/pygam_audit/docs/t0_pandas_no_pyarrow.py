import numpy as np, pandas as pd, gamfit, importlib.util
print("pandas", pd.__version__, "pyarrow installed:", importlib.util.find_spec("pyarrow") is not None)
rng = np.random.default_rng(0); x = rng.uniform(0, 10, 200)
df = pd.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, .3, 200)})
try:
    gamfit.fit(df, "y ~ s(x)"); print("pandas fit OK")
except Exception as e:
    print("pandas fit FAILED:", type(e).__name__, e)
m = gamfit.fit({"x": x, "y": df.y.values}, "y ~ s(x)"); print("dict fit OK")
