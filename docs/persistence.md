# Save and load

`gamfit` persists fitted models (`.gam`).

## Models — `.gam`

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test = {"x": np.array([1.5, 2.5, 3.5])}

model = gamfit.fit(train, "y ~ s(x)")
model.save("model.gam")

loaded = gamfit.load("model.gam")
preds  = loaded.predict(test)
```

In-memory transport:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train, "y ~ s(x)")

blob   = model.dumps()              # bytes
loaded = gamfit.loads(blob)
```

The `.gam` file is the JSON serialisation of the Rust `FittedModel`
enum, which has the five variants `standard`, `location-scale`,
`marginal-slope`, `survival`, and `transformation-normal` (`model_type`
plus `payload`; the Rust save path writes it with
`serde_json::to_writer`). It contains:

- coefficients, smoothing parameters, basis specifications;
- formula and family / link metadata;
- the joint posterior state needed by the model's default point estimand;
- the data schema used by `Model.check(...)` and `Model.predict(...)`.

It does not contain the training data, with one exception stated below.

The default point estimate is always the posterior mean. For a curved response
map this differs from plugging the fitted coefficient mode into the inverse
link, so a mode alone is not a complete saved model. Such models persist either
the conditional coefficient covariance in the saved coefficient frame or a
same-frame strictly positive-definite penalized precision from which it can be
reconstructed. Save and load reject a curved-link payload lacking that state;
they never silently substitute a plug-in/MAP prediction.

**The multinomial payload additionally bundles its training rows** — the design
in the saved coefficient basis, the class index, the case weights, and the
coupled joint penalty. This is not redundancy with the covariance: the
covariance IS the Laplace quadratic model of the log-posterior, and integrating
`softmax` against that quadratic model is not an approximation of the posterior
mean at all — it keeps the curvature half of the leading correction and drops
the skewness half, which on a near-separated softmax is neither small nor the
same sign. Computing the published estimand honestly therefore requires
evaluating the posterior away from its mode, which requires the likelihood,
which requires the rows. `mgcv` makes the same choice in keeping the model
frame with the fitted object.

Saved model payloads cover the current Python-facing model classes:
standard scalar GAMs (the `standard` variant), Gaussian / binomial /
dispersion location-scale fits (`location-scale`), Bernoulli
marginal-slope models (`marginal-slope`), survival transformation /
Weibull / location-scale / marginal-slope / latent models (`survival`),
and transformation-normal models (`transformation-normal`). The saved
schema also records the training table kind so a loaded model preserves
the same default prediction container policy for ambiguous dict/list
inputs.

After loading, every method works as on the original model:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test = {"x": np.array([1.5, 2.5, 3.5])}
gamfit.fit(train, "y ~ s(x)").save("model.gam")
loaded = gamfit.load("model.gam")

loaded.predict(test, interval=0.95)
loaded.summary()
loaded.diagnose(train)             # diagnostics need the response column
loaded.sample(test, seed=42)
```

## Posterior samples

Posterior draws have no gamfit file format. `posterior.to_numpy()` and
`posterior.to_pandas()` hand the draw matrix to NumPy or pandas, and the saved
model is what `model.sample(...)` draws from again.

## Version compatibility

`.gam` payloads are version-gated by the Rust loader
(`validate_for_persistence`). The payload `version` must match the
current `MODEL_PAYLOAD_VERSION`; a schema mismatch fails to load. For
long-term archival, pin the `gamfit` version or refit after upgrades.

## Patterns

### Inspect a model without the training data

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train, "y ~ s(x)")

model.save("model.gam")            # a model saved earlier
m = gamfit.load("model.gam")
print(m.summary())
print(m.model_class, m.formula)
```

### Round-trip in tests

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test = {"x": np.array([1.5, 2.5, 3.5])}
model = gamfit.fit(train, "y ~ s(x)")

blob = model.dumps()
assert (gamfit.loads(blob).predict(test)
        == model.predict(test)).all()
```
