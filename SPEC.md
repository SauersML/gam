# Project Specification

These are project-level engineering constraints. They are intentionally
stricter than the public docs because they guide implementation choices.

- Autodiff is never used outside of tests as hand-derived derivatives
  enable performance optimizations. Exception: external libraries may use
  autodiff internally, for example PyTorch.
- Finite differences are never used outside of tests. The only exception
  is differentiating observed data when analytic differentiation is not
  mathematically available.
- Posterior mean is the default prediction target, never MAP.
- Analytic closed forms should be supported in general for all model
  types.
- Penalties must be on the final function itself, not on model
  coefficients. Exception: coefficient-space penalties are allowed when
  they are exactly equivalent to the function-space penalty.
- Fitting and inference must be fast across data scales, especially
  large-scale data where solver strategy matters most.
- Never vendor external software.
- Python should be a thin wrapper over Rust and should avoid containing
  math or substantial engine logic. Python, Rust, and CLI behavior should
  stay in feature parity. Exception: Python logic needed for external
  integrations such as PyTorch.
- The program must not run out of memory on reasonably resourced
  computers.
- REML or LAML is used for fitting, never GCV.
- Penalties, as priors toward no effect, should usually be present except
  for obvious unpenalized terms such as intercepts.
- Defaults should give the best model for the user.
- Defaults should generally allow recovery of the null, empirical-Bayes
  style. Users must opt into overfitting potential.
- Large changes are acceptable when they materially improve the code.
- XFAIL tests are not allowed. A failing test should always indicate
  problematic behavior.
- The goal of this project is never to copy existing reference implementations.
