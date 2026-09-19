import numpy as np, warnings, traceback
warnings.simplefilter("ignore")
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from gamfit.sklearn import GAMRegressor, GAMClassifier
# sklearn _regression_dataset
X, y = make_regression(n_samples=200, n_features=10, n_informative=1, bias=5.0, noise=20, random_state=42)
X = StandardScaler().fit_transform(X); y = StandardScaler().fit_transform(y.reshape(-1,1)).ravel()
print("coef informative col:", np.argmax(np.abs(np.corrcoef(np.c_[X,y].T)[-1,:-1])))
for f in ["s(x0)+s(x1)", "s(x0)", "s(x1)", "x0+x1"]:
    try:
        m = GAMRegressor(formula=f).fit(X, y); print(f, "OK R2", round(m.score(X,y),3))
    except Exception as e: print(f, "FAIL", type(e).__name__, str(e)[:200])
# pure noise
rng = np.random.default_rng(0)
Xn = rng.normal(size=(200,2)); yn = rng.normal(size=200)
for seed in range(3):
    rng = np.random.default_rng(seed); Xn = rng.normal(size=(200,2)); yn = rng.normal(size=200)
    try:
        m = GAMRegressor(formula="s(x0)+s(x1)").fit(Xn, yn); print("noise seed",seed,"OK")
    except Exception as e: print("noise seed",seed,"FAIL", type(e).__name__, str(e)[:150])
# unfitted
try: GAMClassifier(formula="s(x0)").predict(np.zeros((3,1)))
except Exception as e: print("unfitted clf:", type(e).__name__, str(e)[:150])
try: GAMRegressor(formula="s(x0)").predict(np.zeros((3,1)))
except Exception as e: print("unfitted reg:", type(e).__name__, str(e)[:150])
# NaN in column not in formula
rng = np.random.default_rng(1); X3 = rng.uniform(size=(40,3)); y3 = X3[:,0]+0.1*rng.normal(size=40)
X3[0,2] = np.nan
try: GAMRegressor(formula="s(x0)+s(x1)").fit(X3,y3); print("NaN in unused col: accepted")
except Exception as e: print("NaN unused col:", type(e).__name__, str(e)[:150])
# extra feature at predict
m = GAMRegressor(formula="s(x0)").fit(X[:, :2], y)
try: print("predict with 5 cols (fit had 2):", m.predict(X[:, :5])[:2])
except Exception as e: print("extra col predict:", type(e).__name__, str(e)[:150])
print("n_features_in_", getattr(m, "n_features_in_", None))
