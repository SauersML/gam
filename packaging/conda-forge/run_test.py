"""conda-build runs this after installing the package: one small fit and predict."""

import numpy as np

import gamfit

x = np.linspace(0.0, 1.0, 200)
model = gamfit.fit({"x": x, "y": np.sin(6.0 * x)}, "y ~ s(x)")
mean = np.asarray(model.predict({"x": x}), dtype=float)
assert mean.shape == (200,) and np.all(np.isfinite(mean)), mean
