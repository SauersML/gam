"""What gamfit can do today on inverse-Gaussian data: gamma(log) as a variance-misspecified proxy."""
import numpy as np, warnings
from compare_fits import make, gam_fit_predict
warnings.filterwarnings("ignore")
for r in range(2):
    rng = np.random.default_rng(1000 + r)
    train, test, truth, _, _ = make("inv_gauss", rng, 600)
    mu, lo, hi, tt = gam_fit_predict(train, "gamma", test)
    print(f"rep {r} gamfit gamma(log) on IG data: RMSE {np.sqrt(np.mean((mu-truth)**2)):.4f} cover95 {np.mean((lo<=truth)&(truth<=hi)):.3f} time {tt:.1f}s")
