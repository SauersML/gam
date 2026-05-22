# Save and load

`gamfit` persists fitted models (`.gam`) and posterior samples (`.npz`).

## Models — `.gam`

```python
model = gamfit.fit(train, "y ~ s(x)")
model.save("model.gam")

loaded = gamfit.load("model.gam")
preds  = loaded.predict(test)
```

In-memory transport:

```python
blob   = model.dumps()              # bytes
loaded = gamfit.loads(blob)
```

The `.gam` file is the JSON serialisation of the Rust `Model` struct
(produced by `serde_json::to_writer`). It contains:

- coefficients, smoothing parameters, basis specifications;
- formula and family / link metadata;
- the data schema used by `Model.check(...)` and `Model.predict(...)`.

It does not contain the training data.

After loading, every method works as on the original model:

```python
loaded.predict(test, interval=0.95)
loaded.summary()
loaded.diagnose(test)
loaded.sample(test, seed=42)
```

## Posterior samples — `.npz`

```python
posterior = model.sample(train, seed=42)
posterior.save("posterior.npz")

restored = gamfit.load_posterior("posterior.npz")
# equivalent:
restored = gamfit.PosteriorSamples.load("posterior.npz")
```

The archive (written via `numpy.savez`) contains:

| Array | Shape / dtype | Meaning |
| --- | --- | --- |
| `samples` | `(n_draws, n_coeffs)` float64 | Raw draws. |
| `mean` | `(n_coeffs,)` float64 | Per-coefficient posterior mean. |
| `std` | `(n_coeffs,)` float64 | Per-coefficient posterior std. |
| `rhat` | scalar float64 | Convergence diagnostic. |
| `ess` | scalar float64 | Effective sample size. |
| `converged` | scalar bool | Convergence flag. |
| `model_bytes` | 1-D uint8 | The fitted model's serialised bytes. |
| `metadata` | 0-D object (JSON) | `coefficient_names`, `method`, `model_class`, `family_kind`, `config`. |

`PosteriorSamples.load` requires `allow_pickle=True` to round-trip the
metadata object array; only load archives produced by `save`.

Because the model bytes are bundled, `predict()` and `predict_draws()`
work after a round-trip:

```python
restored = gamfit.load_posterior("posterior.npz")
bands    = restored.predict(test, level=0.95)
```

## Version compatibility

`.gam` payloads are version-gated by the Rust loader
(`validate_for_persistence`). Models written by an older release may
fail to load in a newer release if a required field has been added. For
long-term archival, pin the `gamfit` version or refit after upgrades.

## Patterns

### Save model and posterior together

```python
model.save("model.gam")
model.sample(train, seed=42).save("posterior.npz")
```

The posterior archive carries the model bytes, so loading just the
posterior is enough to call `predict()`.

### Inspect a model without the training data

```python
m = gamfit.load("model.gam")
print(m.summary())
print(m.model_class, m.formula)
```

### Round-trip in tests

```python
blob = model.dumps()
assert (gamfit.loads(blob).predict(test)["mean"]
        == model.predict(test)["mean"]).all()
```
