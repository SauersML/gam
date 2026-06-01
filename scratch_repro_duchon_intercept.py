"""Reproduce the duchon-nullspace-constant vs GLM-intercept rank-1 deficiency.

Probe A: plain probit GLM `y ~ duchon(x1,x2,x3)`  -> isolates whether the
         general smooth-vs-intercept centering fails for pure Duchon.
Probe B: bernoulli-marginal-slope (matches gnomon biobank binary path).
Probe C: gnomon's proposed fix -> add explicit linear PC terms.

Run: ~/gam/.venv/bin/python scratch_repro_duchon_intercept.py
"""

import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
N = 4000
PCs = rng.standard_normal((N, 3))
z = rng.standard_normal(N)
sex = rng.integers(0, 2, N).astype(float)
entry_age_z = rng.standard_normal(N)
age_ns = rng.standard_normal((N, 4))
eta = 0.3 * PCs[:, 0] - 0.2 * PCs[:, 1] + 0.5 * z + 0.1 * sex
p = 1.0 / (1.0 + np.exp(-eta))
y = (rng.uniform(size=N) < p).astype(float)

df = pd.DataFrame(
    {
        "event": y,
        "sex": sex,
        "prs_z": z,
        "entry_age_z": entry_age_z,
        "current_age_ns_1": age_ns[:, 0],
        "current_age_ns_2": age_ns[:, 1],
        "current_age_ns_3": age_ns[:, 2],
        "current_age_ns_4": age_ns[:, 3],
        "PC1": PCs[:, 0],
        "PC2": PCs[:, 1],
        "PC3": PCs[:, 2],
    }
)

DUCHON = "duchon(PC1, PC2, PC3, centers=10, order=1)"
AGE = "entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4"


def run(label, **kwargs):
    print(f"\n===== {label} =====")
    try:
        m = gamfit.fit(**kwargs)
        print(f"  OK -> fit succeeded ({label})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL [{type(exc).__name__}]: {str(exc)[:400]}")
        return False


# Probe A: plain probit GLM with a pure Duchon over the 3 PCs + implicit intercept.
run(
    "A: plain probit GLM  y ~ duchon(PCs)",
    data=df[["event", "PC1", "PC2", "PC3"]],
    formula=f"event ~ {DUCHON}",
    family="bernoulli",
    link="probit",
)

# Probe B: BMS marginal-slope, mirrors gnomon's failing binary formula.
run(
    "B: bernoulli-marginal-slope (gnomon binary, pre-fix)",
    data=df,
    formula=f"event ~ {DUCHON} + sex + {AGE} + linkwiggle()",
    family="bernoulli-marginal-slope",
    link="probit",
    z_column="prs_z",
    logslope_formula=f"{DUCHON} + linkwiggle()",
)

# Probe C: gnomon's proposed fix -> explicit linear PC terms in the marginal formula.
run(
    "C: bernoulli-marginal-slope + explicit PC linear terms (gnomon fix)",
    data=df,
    formula=f"event ~ {DUCHON} + PC1 + PC2 + PC3 + sex + {AGE} + linkwiggle()",
    family="bernoulli-marginal-slope",
    link="probit",
    z_column="prs_z",
    logslope_formula=f"{DUCHON} + linkwiggle()",
)
