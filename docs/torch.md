# PyTorch integration

`gamfit.torch` exposes gamfit's analytic primitives and smooth-fit helpers
to PyTorch. The low-level REML and basis wrappers call the corresponding
NumPy / Rust entry points; for primitives with an analytic Rust backward,
gradients flow through `loss.backward()`.

## Installation

The torch dependency is optional:

```bash
pip install torch
```

Importing `gamfit` does not import torch. Importing `gamfit.torch` without
torch installed raises `ImportError` with an install hint. The package
metadata does not define a `gamfit[torch]` extra; choose the PyTorch wheel
that matches your CPU/CUDA environment.

## Differentiable Gaussian REML

The closed-form Gaussian REML primitives have analytic Rust VJPs.
`gaussian_reml_fit` and `gaussian_reml_fit_batched` return a
`GaussianRemlOutput` (`coefficients`, `fitted`, `lam`, `reml_score`, `edf`)
and route upstream gradients into the matching Rust backward:

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
`gaussian_reml_fit_additive`, and `gaussian_reml_fit_blocks`. The additive
and block variants return `AdditiveRemlOutput` (`coefficients`, `fitted`,
`lambdas`, `log_lambdas`, `reml_score`, `edf`); single-response additive
fits use exact dense multi-block REML, while multi-output additive fits use
the shared-scale block-orthogonal estimator with per-smooth lambdas. Saved
tensors are version-checked;
in-place mutation between forward and backward raises `RuntimeError`.

## Embedding a fitted model

`from_fitted` wraps a fitted `gamfit.Model` as a frozen `nn.Module`:

```python
import gamfit
import gamfit.torch as gt

model = gamfit.fit_array(X_train, y_train, "y ~ s(x0) + x1 + x2")
frozen = gt.from_fitted(model)

X = torch.as_tensor(X_test, dtype=torch.float64)
preds = frozen(X)   # response-scale point predictions
```

The forward pass crosses the NumPy / Rust boundary. Gradients do not flow
back through the inputs. The wrapped model has no trainable parameters
(coefficients live in the saved bytes, not in `nn.Parameter`s).

## Response geometry

Torch implementations of the simplex and unit-sphere transforms:

`gamfit.torch` exports `closure`, `clr`, `alr`, `inverse_alr`, `simplex_log_map`,
`simplex_exp_map`, `simplex_frechet_mean`, `sphere_log_map`,
`sphere_exp_map`, `sphere_frechet_mean`. Autograd flows through these
except `sphere_frechet_mean`, which calls the Rust spherical mean routine
and is forward-only.

`gamfit.torch.geometry` also exposes `ilr`, `inverse_ilr`, and
`aitchison_metric`. Its simplex log/exp maps default to the isometric ILR chart
(`coordinates="ilr"` / `"simplex"`), while `coordinates="alr"` remains a
non-isometric chart whose Aitchison Gram is `aitchison_metric(d)`.

## Device and dtype

The gamfit Rust backend operates on f64 CPU buffers. Autograd REML wrappers
move tensors to CPU f64 for the call and return tensors on the caller's
original device and dtype. Some structural basis / penalty builders return
float64 tensors because the penalty algebra is f64. For training-loop
workloads, prefer `gamfit.torch.fit` or the batched and additive primitives
(`gaussian_reml_fit_batched`, `gaussian_reml_fit_additive`,
`gaussian_reml_fit_blocks`) over per-feature Python iteration.

## Output-Fisher harvesting and attention backends

`harvest_output_fisher_factors` and
`harvest_downstream_output_fisher_factors` apply the output-Fisher pullback
matrix-free with both JVPs and VJPs. PyTorch SDPA and flash-attention operators
can support reverse AD while lacking the forward-mode JVP these two harvests
require. For Hugging Face models, select the differentiable attention path when
the model is loaded:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="eager",
)
```

If an unsupported attention operator is reached, gamfit raises an actionable
error at the JVP boundary. It does not mutate an already-loaded model or change
the estimator. `harvest_behavioral_fisher_probes` is VJP-only, but it emits the
distinct likelihood-weighting `behavioral_fisher` sketch; it is not a silent
replacement for either top-eigenfactor/dosimetry harvest.

## Public API

| Group | Symbols |
| --- | --- |
| High-level fitting | `fit`, `FitResult`, `GAM` |
| Smooth specs | `Smooth`, `Duchon`, `BSpline`, `TensorBSpline`, `Matern`, `Pca`, `PeriodicSplineCurve`, `ShapeConstraintLiteral`, `Sphere`, `Categorical` |
| Closed-form REML | `gaussian_reml_fit`, `gaussian_reml_fit_batched`, `gaussian_reml_fit_additive`, `gaussian_reml_fit_blocks`, `GaussianRemlOutput`, `AdditiveRemlOutput` |
| Basis evaluations | `bspline_basis`, `bspline_basis_derivative`, `duchon_basis`, `periodic_spline_curve_basis`, `sphere_basis` |
| Penalty / ridge | `smoothness_penalty`, `gaussian_weighted_ridge`, `gaussian_weighted_ridge_batch` |
| Penalty modules | `AdaptiveTopK`, `ARDPenalty`, `BlockOrthogonalityPenalty`, `GatedSAEDecoder`, `GumbelTemperatureSchedule`, `IBPAssignmentPenalty`, `IsometryPenalty`, `IvaeRidgeMeanGauge`, `JumpReLUPenalty`, `LazyPcaBasis`, `MechanismSparsityPenalty`, `MonotonicityPenalty`, `RiemannianRetraction`, `SoftmaxAssignmentSparsityPenalty`, `SparsityPenalty`, `TopKActivationPenalty`, `TopologyAutoSelector` |
| Manifold SAE | `ManifoldSAE`, `ManifoldSAEConfig`, `DecoderConfig`, `RemlConfig`, `SparsityConfig`, `ManifoldSAEOutput`, `circular_concordance`, `CircularConcordanceReport` |
| Harvest / Fisher factors | `HarvestShard`, `harvest_output_fisher_factors`, `harvest_downstream_output_fisher_factors`, `save_harvest_shard`, `load_harvest_shard` |
| Hyperbolic / interchange | `PoincareAtoms`, `InterchangeSwapDecoder` |
| Skip transcoder | `SkipAffineSmooth`, `skip_transcoder`, `select_skip_transcoder`, `SkipTranscoderSelectionResult` |
| Response geometry | `closure`, `clr`, `alr`, `inverse_alr`, `simplex_log_map`, `simplex_exp_map`, `simplex_frechet_mean`, `sphere_log_map`, `sphere_exp_map`, `sphere_frechet_mean`; in `gamfit.torch.geometry`: `ilr`, `inverse_ilr`, `aitchison_metric` |
| Fitted-model loader | `from_fitted` |

Low-level value-producing wrappers are bit-exact equal to their NumPy
counterparts. Primitives with an analytic Rust backward are covered by
`torch.autograd.gradcheck`.
