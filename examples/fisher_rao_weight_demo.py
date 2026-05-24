import numpy as np

import gamfit


def make_data(seed=4, n=180):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    behavior = 0.75 * np.sin(2.0 * np.pi * x)
    template = 1.35 * np.sin(10.0 * np.pi * x)
    angle = 0.55 + behavior + template

    y = np.column_stack([np.cos(angle), np.sin(angle)])
    burst = np.abs(x - 0.35) < 0.035
    burst |= np.abs(x - 0.72) < 0.04
    y[burst] += rng.normal(scale=[0.45, 0.05], size=(burst.sum(), 2))
    y += rng.normal(scale=0.035, size=y.shape)
    y /= np.linalg.norm(y, axis=1, keepdims=True)

    grad = np.column_stack([-np.sin(0.55 + behavior), np.cos(0.55 + behavior)])
    W = np.zeros((n, 2, 2))
    for i, g in enumerate(grad):
        W[i] = 0.08 * np.eye(2) + 3.5 * np.outer(g, g)
    return {"x": x, "y0": y[:, 0], "y1": y[:, 1]}, W


def fit_shared(data, fisher_rao_w):
    return gamfit.fit(
        data,
        "y0 ~ s(x, k=36)",
        family="gaussian",
        response_geometry="spherical",
        response_columns=["y0", "y1"],
        fisher_rao_w=fisher_rao_w,
    )


def main():
    data, W = make_data()
    identity = fit_shared(data, fisher_rao_w=None)
    behavioral = fit_shared(data, fisher_rao_w=W)

    s0 = identity.summary()["shared_fit"]
    s1 = behavioral.summary()["shared_fit"]
    print("W=None lambda:", round(float(s0["lambda"]), 4), "edf:", round(float(s0["edf"]), 2))
    print("W=Fisher-Rao lambda:", round(float(s1["lambda"]), 4), "edf:", round(float(s1["edf"]), 2))


if __name__ == "__main__":
    main()
