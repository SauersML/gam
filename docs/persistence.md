# Save and load

`gamfit` persists two things: fitted **models** and **posterior samples**.
Both round-trip exactly.

## Models — `.gam`

A fitted `Model` saves to a single binary file:

```python
model = gamfit.fit(train, "y ~ s(x)")
model.save("model.gam")

loaded = gamfit.load("model.gam")
preds  = loaded.predict(test)
```

Or as in-memory bytes:

```python
blob   = model.dumps()                     # bytes
loaded = gamfit.loads(blob)
```

The `.gam` file is a binary blob produced by the Rust engine. It contains:

- coefficients, smoothing parameters, basis specifications
- formula and family/link metadata
- the schema needed by `Model.check(...)` and `Model.predict(...)`

It does not contain:

- training data
- the Python version or package version it was created with. Minor version
  bumps usually load; major bumps are not guaranteed.

After loading, every method works as on the original model:

```python
loaded.predict(test, interval=0.95)
loaded.summary()
loaded.diagnose(test)
loaded.sample(test, seed=42)
```

## Posterior samples — `.npz`

`PosteriorSamples.save(...)` writes a NumPy `.npz` archive:

```python
posterior = model.sample(train, seed=42)
posterior.save("posterior.npz")

restored = gamfit.load_posterior("posterior.npz")
# or:
restored = gamfit.PosteriorSamples.load("posterior.npz")
```

The archive includes:

| Array | Shape | Meaning |
| --- | --- | --- |
| `samples` | `(n_draws, n_coeffs) float64` | Raw draws. |
| `mean`, `std` | `(n_coeffs,) float64` | Per-coeff posterior moments. |
| `rhat`, `ess` | scalar | Convergence summaries. |
| `converged` | scalar bool | `rhat < 1.1`. |
| `model_bytes` | 1-D uint8 | The fitted model (so `predict(...)` works after a load). |
| `metadata` | 0-D object (JSON) | `config`, `coefficient_names`, `method`, `model_class`, `family_kind`. |

Because the model bytes are bundled, `predict()` and `predict_draws()`
work after a round-trip:

```python
restored = gamfit.load_posterior("posterior.npz")
bands    = restored.predict(test, level=0.95)
```

## Version compatibility

`.gam` files and `.npz` posteriors written by `gamfit` `0.1.x` are not
guaranteed to load with `0.2.x` or later. For long-term archival, either:

- pin the gamfit version used to produce artefacts, or
- re-fit on the new version when you upgrade.

Save and reload within the same session work across all supported
families.

## Patterns

### Save a model and a sampled posterior together

```python
model.save("model.gam")
model.sample(train, seed=42).save("posterior.npz")
```

Load them independently later, or load the posterior, which carries the
model bytes.

### Inspect a model on a machine without the training data

```python
m = gamfit.load("model.gam")
print(m.summary())            # coefficients, family, etc.
print(m.model_class, m.formula)
```

Lets you share a model for reproducibility without sharing data.

### Programmatic round-trip in tests

```python
import io
blob = model.dumps()
assert gamfit.loads(blob).predict(test)["mean"] == model.predict(test)["mean"]
```
