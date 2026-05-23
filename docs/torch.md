# PyTorch integration

`gamfit.torch` exposes gamfit's analytic primitives to PyTorch's autograd.
Each function wraps the corresponding NumPy entry point in `gamfit._api`;
no derivative math is reimplemented in Python. For primitives with an
analytic Rust backward, gradients flow through `loss.backward()`.

## Installation

The torch dependency is optional:

```bash
pip install gamfit[torch]
```

Importing `gamfit` does not import torch. Importing `gamfit.torch` without
torch installed raises `ImportError` with an install hint.

## Differentiable Gaussian REML

The four closed-form REML primitives have analytic Rust VJPs. Each returns
a `GaussianRemlOutput` (`coefficients`, `fitted`, `lam`, `reml_score`) and
routes upstream gradients into the matching Rust backward:

```python
import torch
import gamfit.torch as gt

x = torch.randn(40, 4, dtype=torch.float64, requires_grad=True)
y = torch.randn(40, 1, dtype=torch.float64)
penalty = torch.eye(4, dtype=torch.float64)

out = gt.gaussian_reml_fit(x, y, penalty)
loss = (out.fitted - y).pow(2).mean() + 0.01 * out.reml_score
loss.backward()
```

The same pattern applies to `gaussian_reml_fit_batched`,
`gaussian_reml_fit_additive`, and `gaussian_reml_fit_blocks`
(the additive variant returns `AdditiveRemlOutput`). Saved tensors
are version-checked; in-place mutation between forward and backward raises
`RuntimeError`.

## Embedding a fitted model

`from_fitted` wraps a fitted `gamfit.Model` as a frozen `nn.Module`:

```python
import gamfit
import gamfit.torch as gt

model = gamfit.fit_array(X_train, y_train, "y ~ s(x0) + x1 + x2")
frozen = gt.from_fitted(model)

X = torch.as_tensor(X_test, dtype=torch.float64)
preds = frozen(X)   # (N, P): typically [eta, mean]
```

The forward pass crosses the NumPy / Rust boundary. Gradients do not flow
back through the inputs. The wrapped model has no trainable parameters
(coefficients live in the saved bytes, not in `nn.Parameter`s).

## Response geometry

Pure-torch implementations of the simplex and unit-sphere transforms:

`closure`, `clr`, `alr`, `inverse_alr`, `simplex_log_map`,
`simplex_exp_map`, `simplex_frechet_mean`, `sphere_log_map`,
`sphere_exp_map`, `sphere_frechet_mean`. Autograd flows through these
because they are written directly in torch.

## Device and dtype

The gamfit Rust backend operates on f64 CPU buffers. Tensors are moved to
CPU f64 for the call and returned on the caller's original device and
dtype. For training-loop workloads, prefer the batched and additive
primitives (`gaussian_reml_fit_batched`, `gaussian_reml_fit_additive`,
`gaussian_reml_fit_blocks`) over per-feature Python iteration.

## Public API

| Group | Symbols |
| --- | --- |
| Closed-form REML | `gaussian_reml_fit`, `gaussian_reml_fit_batched`, `gaussian_reml_fit_additive`, `gaussian_reml_fit_blocks`, `GaussianRemlOutput`, `AdditiveRemlOutput` |
| Basis evaluations | `bspline_basis`, `bspline_basis_derivative`, `duchon_basis`, `periodic_spline_curve_basis`, `sphere_basis` |
| Penalty / ridge | `smoothness_penalty`, `gaussian_weighted_ridge`, `gaussian_weighted_ridge_batch` |
| Response geometry | `closure`, `clr`, `alr`, `inverse_alr`, `simplex_log_map`, `simplex_exp_map`, `simplex_frechet_mean`, `sphere_log_map`, `sphere_exp_map`, `sphere_frechet_mean` |
| Fitted-model loader | `from_fitted` |

Value-producing primitives are bit-exact equal to their NumPy
counterparts. Primitives with an analytic Rust backward are covered by
`torch.autograd.gradcheck`.
