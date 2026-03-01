# gam

Generalized penalized likelihood engine.

The crate exposes flat `src/*.rs` modules for spline basis generation, reparameterization,
PIRLS/REML optimization, joint models, diagnostics/ALO, and HMC.

High-level smooth term support includes:
- Isotropic radial smooths (`ThinPlate`, `Matern`, `Duchon`) for same-scale feature spaces.
- Tensor-product smooths (`TensorBSpline`) for mixed-scale axes (e.g., space x time).
- Random effects via categorical dummy blocks with identity (ridge) penalties.
- Bounded parametric coefficients via `bounded(x, min=..., max=...)`, with optional interior Beta priors.

Formula examples:
- `y ~ age + smooth(bmi) + group(site)`
- `y ~ bounded(mu_hat, min=0, max=1) + matern(pc1, pc2, pc3)`
- `y ~ bounded(log_v_hat, min=0, max=2, target=1, strength=5) + x`
