"""Seeded Monte-Carlo calibration of multi-predictor smooth-term p-values.

Each scenario simulates a dataset, fits it through the public ``gamfit`` API,
and records the p-value of every test of the target term. Under a null scenario
the target covariate has no effect on the tested predictor, so a valid test has
P(p <= a) <= a; under a power scenario it has a real effect.

    python calibrate.py SCENARIO N REPS SEED OUT.json

Scenarios
---------
multinomial_null     3 classes, class probabilities depend on z only; x is null
                     for every class. Records the per-class s(x) tests and the
                     all-classes (joint) s(x) test.
multinomial_power    as above plus a smooth x effect on class b only.
ls_scale_only        Gaussian location-scale: mean depends on z, the noise scale
                     depends on x. Records the mean-predictor s(x) test, whose
                     null holds although x is in the model through the scale.
ls_power             as above plus a smooth x effect on the mean.
weibull_null         Weibull survival: hazard depends on age, noise is null.
                     Records the s(noise) test on the covariate predictor.
weibull_power        as above; records the s(age) test.
transformation_null  transformation (Royston-Parmar) survival, whose covariates
                     share one predictor block with the time basis: as
                     weibull_null.
transformation_power as above; records the s(age) test.
survls_null          location-scale survival (threshold + scale blocks): as
                     weibull_null, tested on the threshold predictor.
survls_power         as above; records the s(age) test.

The latent survival likelihood is not calibrated: a single n = 300 fit takes
minutes, so 500 replications are infeasible (see README.md).

Only fields are read here; every p-value comes from the Rust kernels.
"""

import json
import sys
import time

import numpy as np
import pandas as pd

import gamfit

CLASSES = np.array(["a", "b", "c"])


def multinomial_data(rng, n, x_effect):
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    eta = np.stack(
        [np.zeros(n), np.sin(2 * np.pi * z) + x_effect * np.sin(np.pi * x), 1.5 * (z - 0.5)],
        axis=1,
    )
    prob = np.exp(eta)
    prob /= prob.sum(axis=1, keepdims=True)
    u = rng.uniform(size=n)[:, None]
    y = (u > np.cumsum(prob, axis=1)).sum(axis=1)
    return pd.DataFrame({"x": x, "z": z, "y": CLASSES[y]})


def multinomial_rep(rng, n, x_effect):
    df = multinomial_data(rng, n, x_effect)
    model = gamfit.fit(df, "y ~ s(x) + s(z)", family="multinomial")
    tests = {}
    for row in model.smooth_significance():
        if row["term"].startswith("s(x"):
            tests[f"per-class {row['class']} vs c"] = row.get("p_value")
    for row in model.joint_smooth_significance():
        if row["term"].startswith("s(x"):
            tests["joint (all classes)"] = row.get("p_value")
    return tests


def ls_rep(rng, n, mean_x_effect):
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    mean = np.sin(2 * np.pi * z) + mean_x_effect * np.sin(np.pi * x)
    y = mean + np.exp(-1.0 + 1.2 * x) * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    model = gamfit.fit(df, "y ~ s(x) + s(z)", noise_formula="s(x)")
    summary = model.summary()
    if summary.smooth_terms_unavailable is not None:
        raise RuntimeError(f"smooth table unavailable: {summary.smooth_terms_unavailable}")
    tests = {}
    for row in summary.smooth_terms:
        if row["name"].startswith("s(x") or row["name"] == "x":
            tests["mean s(x)"] = row.get("p_value")
    return tests


def survival_rep(rng, n, likelihood, term):
    age = rng.uniform(40, 75, n)
    noise = rng.uniform(0, 1, n)
    eta = 0.05 * (age - 57.5)
    u = rng.uniform(1e-12, 1, n)
    # Weibull(shape 1.5, scale 10) baseline with a proportional-hazards age effect.
    t = 10.0 * (-np.log(u) * np.exp(-eta)) ** (1 / 1.5)
    c = np.minimum(rng.exponential(30.0, n), 25.0)
    event = (t <= c).astype(int)
    df = pd.DataFrame(
        {"entry": np.zeros(n), "exit": np.minimum(t, c), "event": event, "age": age, "noise": noise}
    )
    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ s(age) + s(noise)",
        survival_likelihood=likelihood,
    )
    summary = model.summary()
    if summary.smooth_terms_unavailable is not None:
        raise RuntimeError(f"smooth table unavailable: {summary.smooth_terms_unavailable}")
    tests = {}
    for row in summary.smooth_terms:
        if row["name"].startswith(f"s({term}"):
            tests[f"s({term})"] = row.get("p_value")
    return tests


SCENARIOS = {
    "multinomial_null": lambda rng, n: multinomial_rep(rng, n, 0.0),
    "multinomial_power": lambda rng, n: multinomial_rep(rng, n, 0.8),
    "ls_scale_only": lambda rng, n: ls_rep(rng, n, 0.0),
    "ls_power": lambda rng, n: ls_rep(rng, n, 0.3),
    "weibull_null": lambda rng, n: survival_rep(rng, n, "weibull", "noise"),
    "weibull_power": lambda rng, n: survival_rep(rng, n, "weibull", "age"),
    "transformation_null": lambda rng, n: survival_rep(rng, n, "transformation", "noise"),
    "transformation_power": lambda rng, n: survival_rep(rng, n, "transformation", "age"),
    "survls_null": lambda rng, n: survival_rep(rng, n, "location-scale", "noise"),
    "survls_power": lambda rng, n: survival_rep(rng, n, "location-scale", "age"),
}


def main():
    scenario, n, reps, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rng = np.random.default_rng(seed)
    records = []
    started = time.time()
    for rep in range(reps):
        try:
            records.append({"rep": rep, "tests": SCENARIOS[scenario](rng, n)})
        except Exception as exc:  # a failed replication is recorded, never dropped silently
            records.append({"rep": rep, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    with open(out, "w") as handle:
        json.dump(
            {
                "scenario": scenario,
                "n": n,
                "reps": reps,
                "seed": seed,
                "seconds": round(time.time() - started, 1),
                "records": records,
            },
            handle,
        )


if __name__ == "__main__":
    main()
