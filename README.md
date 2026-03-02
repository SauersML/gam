# gam

Generalized penalized likelihood engine.

The crate exposes flat `src/*.rs` modules for spline basis generation, reparameterization,
PIRLS/REML optimization, joint models, diagnostics/ALO, and HMC.

High-level smooth term support includes:
- Isotropic radial smooths (`ThinPlate`, `Matern`, `Duchon`) for same-scale feature spaces.
- Tensor-product smooths (`TensorBSpline`) for mixed-scale axes (e.g., space x time).
- Random effects via categorical dummy blocks with identity (ridge) penalties.
- Bounded parametric coefficients via `bounded(x, min=..., max=...)`, with explicit optional priors on the bounded user-scale coefficient.
  `prior="uniform"` is the log-Jacobian correction for a flat prior on the bounded coefficient scale.
- Generic linear coefficient constraints via `linear(x, min=..., max=...)`, `constrain(x, ...)`, `nonnegative(x)`, and `nonpositive(x)`.
- Unpenalized linear columns are automatically centered/scaled internally during fitting for solver conditioning, then mapped back to the original coefficient scale for summaries, prediction, and saved models.

Formula examples:
- `y ~ age + smooth(bmi) + group(site)`
- `y ~ nonnegative(effect) + smooth(bmi)`
- `y ~ bounded(mu_hat, min=0, max=1) + matern(pc1, pc2, pc3)`
- `y ~ bounded(mu_hat, min=0, max=1, prior="log-jacobian") + z`
- `y ~ bounded(log_v_hat, min=0, max=2, target=1, strength=5) + x`
