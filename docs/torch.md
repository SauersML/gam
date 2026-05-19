# PyTorch integration

`gamfit.torch` is a thin bridge that lets you compose gamfit's analytic
primitives with PyTorch's autograd. Every function in the bridge is a one-call
wrapper around the corresponding NumPy entry point in `gamfit._api`; no
derivative math is rewritten in Python or torch. The closed-form Gaussian REML
gradients computed by the Rust engine flow straight into `loss.backward()`.

## When to use it

Reach for `gamfit.torch` when you want a gamfit component to live inside a
larger torch model:

- A neural encoder produces design coordinates that flow through a fitted
  smooth and back, as in manifold-style sparse autoencoders.
- A torch training loop needs gradients through a closed-form Gaussian REML
  fit (for example, to learn the positions, weights, or `by`-variable feeding
  the fit).
- A previously trained gamfit model needs to act as a frozen, differentiable
  building block alongside other `nn.Module` components.

If you only want to *call* a gamfit primitive on a NumPy array, keep using
`gamfit._api` directly — the torch bridge adds nothing for that case.

## Installation

The torch dependency is optional. To install:

```bash
pip install gamfit[torch]
```

Importing `gamfit` does not pull in torch. The first import of `gamfit.torch`
checks that torch is available and raises a clear `ImportError` with install
instructions if it is not.

## Differentiable closed-form Gaussian REML

The four closed-form REML primitives have analytic VJPs in Rust. Each wrapper
returns a `GaussianRemlOutput` named tuple of four torch tensors —
`coefficients`, `fitted`, `lam`, `reml_score` — and routes upstream gradients
into the matching Rust backward:

```python
import torch
import gamfit.torch as gt

x = torch.randn(40, 4, dtype=torch.float64, requires_grad=True)
y = torch.randn(40, 1, dtype=torch.float64)
penalty = torch.eye(4, dtype=torch.float64)

out = gt.gaussian_reml_fit(x, y, penalty)
loss = (out.fitted - y).pow(2).mean() + 0.01 * out.reml_score
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
```

The same pattern applies to the ragged-batched, position-based, and
position-batched variants. Every input tensor flows through PyTorch's
`save_for_backward` machinery, so any in-place mutation between forward and
backward raises a `RuntimeError` instead of producing a silently wrong
gradient.

## Embedding a fitted model

`gamfit.torch.from_fitted` wraps a fitted `gamfit.Model` as a frozen
`nn.Module`. The module is intended for use as a fixed feature transform
inside a larger learned model:

```python
import gamfit
import gamfit.torch as gt

model = gamfit.fit_array(X_train, y_train, "y ~ s(x0) + x1 + x2")
frozen = gt.from_fitted(model)

# Drop into a torch graph and continue training on top of the GAM's outputs.
head = torch.nn.Linear(2, 1, dtype=torch.float64)
preds = head(frozen(torch.as_tensor(X_test, dtype=torch.float64))[:, :2])
```

The forward pass crosses the NumPy / Rust boundary; gradients **do not flow
back** through the inputs in this version. That is appropriate for the
"frozen GAM as a building block" use case. For full joint training of a
smooth from scratch, train with `gamfit.fit` and embed the result here.

## Response geometry

Tensor-in / tensor-out wrappers exist for the simplex, ALR/CLR, and unit-sphere
transforms (`closure`, `clr`, `alr`, `inverse_alr`, `simplex_log_map`,
`simplex_exp_map`, `simplex_frechet_mean`, `sphere_log_map`, `sphere_exp_map`,
`sphere_frechet_mean`). All are pure torch implementations, so autograd flows
through them naturally. NumPy callers use `gamfit._response_geometry` directly.

## Device and dtype

The gamfit Rust backend runs on f64 CPU. Tensors arriving from any device or
dtype are detached, moved to a CPU f64 buffer, processed in Rust, and wrapped
back as tensors on the caller's original device and dtype. This means a single
host↔device transfer per primitive call. For training-loop workloads, prefer
the batched primitives (`gaussian_reml_fit_batched`,
`gaussian_reml_fit_positions_batched`) over per-feature Python iteration.

## Public API

| Surface | Symbols |
| --- | --- |
| Closed-form REML | `gaussian_reml_fit`, `gaussian_reml_fit_batched`, `gaussian_reml_fit_positions`, `gaussian_reml_fit_positions_batched`, `GaussianRemlOutput` |
| Basis evaluations | `bspline_basis`, `bspline_basis_derivative`, `duchon_basis_1d`, `duchon_basis_1d_derivative` |
| Penalty and ridge | `smoothness_penalty`, `gaussian_weighted_ridge`, `gaussian_weighted_ridge_batch` |
| Response geometry | `closure`, `clr`, `alr`, `inverse_alr`, `simplex_log_map`, `simplex_exp_map`, `simplex_frechet_mean`, `sphere_log_map`, `sphere_exp_map`, `sphere_frechet_mean` |
| Model loader | `from_fitted` |

Every value-producing primitive is bit-exact equal to its NumPy counterpart on
identical inputs. Every primitive with an analytic Rust backward is covered by
a `torch.autograd.gradcheck` against finite differences.
