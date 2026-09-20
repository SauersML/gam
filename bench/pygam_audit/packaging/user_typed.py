import numpy as np
import gamfit
import gamfit.sklearn
import pygam

x = np.linspace(0, 1, 100)
m = gamfit.fit({"x": x, "y": x}, "y ~ s(x)")
reveal_type(m)
p = m.predict({"x": x})
reveal_type(p)
s = m.summary()
reveal_type(s)
reveal_type(gamfit.load)
reveal_type(gamfit.adjudicate_atom_shape)
est = gamfit.sklearn.GAMRegressor("y ~ s(x)")
reveal_type(est.fit)
g = pygam.LinearGAM(pygam.s(0)).fit(x[:, None], x)
reveal_type(g)
