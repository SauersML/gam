# gam

Generalized penalized likelihood engine extracted from gnomon.

The crate exposes flat `src/*.rs` modules for spline basis generation, reparameterization,
PIRLS/REML optimization, joint models, diagnostics/ALO, and HMC.

## Boundary contract

- Engine repo path: `/Users/user/gam`
- Adapter repo path: `/Users/user/gnomon/calibrate`
- Dependency direction: `gnomon -> gam` only.

`gam` owns numeric estimation primitives and family-dispatched solvers. Domain
feature engineering (`PGS/PC/sex` schema policy, file parsing policy, and model
artifact mapping for gnomon outputs) is owned by `gnomon/calibrate`.
