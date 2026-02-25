# gam

Generalized penalized likelihood engine.

The crate exposes flat `src/*.rs` modules for spline basis generation, reparameterization,
PIRLS/REML optimization, joint models, diagnostics/ALO, and HMC.

High-level smooth term support includes:
- Isotropic radial smooths (`ThinPlate`, `Matern`, `Duchon`) for same-scale feature spaces.
- Tensor-product smooths (`TensorBSpline`) for mixed-scale axes (e.g., space x time).
- Random effects via categorical dummy blocks with identity (ridge) penalties.
