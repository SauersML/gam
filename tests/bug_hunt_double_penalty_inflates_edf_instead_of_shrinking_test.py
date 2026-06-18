import numpy as np
import pandas as pd

import gamfit


def test_bspline_double_penalty_does_not_inflate_linear_edf():
    n = 800
    x = np.linspace(0.0, 1.0, n)
    on = []
    off = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"x": x, "y": 2.0 + 3.0 * x + rng.normal(0.0, 0.15, n)})
        on.append(
            float(
                gamfit.fit(
                    df, "y ~ s(x, k=20, bs=ps, double_penalty=True)"
                ).summary().edf_total
            )
        )
        off.append(
            float(
                gamfit.fit(
                    df, "y ~ s(x, k=20, bs=ps, double_penalty=False)"
                ).summary().edf_total
            )
        )

    assert np.mean(on) <= np.mean(off) + 1e-8, (on, off)


def test_default_double_penalty_does_not_inflate_irrelevant_smooth_edf():
    n = 800
    on = []
    off = []
    for seed in range(100, 105):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n)
        z = rng.uniform(0.0, 1.0, n)
        df = pd.DataFrame({"x": x, "z": z, "y": np.sin(6.0 * x) + rng.normal(0.0, 0.3, n)})
        m_on = gamfit.fit(df, "y ~ s(x) + s(z)")
        m_off = gamfit.fit(
            df, "y ~ s(x, double_penalty=False) + s(z, double_penalty=False)"
        )
        z_on = next(t for t in m_on.summary().smooth_terms if "z" in t["name"])
        z_off = next(t for t in m_off.summary().smooth_terms if "z" in t["name"])
        on.append(float(z_on["edf"]))
        off.append(float(z_off["edf"]))

    assert np.mean(on) <= np.mean(off) + 1e-8, (on, off)
