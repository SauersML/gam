# gam

Generalized penalized likelihood engine.

The crate exposes flat `src/*.rs` modules for spline basis generation, reparameterization,
PIRLS/REML optimization, joint models, diagnostics/ALO, and HMC.

## Boundary contract

- Repository: [`SauersML/gam`](https://github.com/SauersML/gam)

`gam` owns numeric estimation primitives and family-dispatched solvers. Domain
feature engineering, file/schema policy, and artifact mapping belong in caller
applications.
