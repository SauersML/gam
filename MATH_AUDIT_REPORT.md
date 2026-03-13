# Mathematical Mistakes Audit

This file records only defects that are provably wrong from the code itself.
No proposed fixes are included here. Each item has a direct counterexample and
at least one intentionally failing test that makes the defect visible.

## 1. `safe_exp` derivatives are not derivatives of `safe_exp`

Code:

- `src/families/sigma_link.rs:43`
- `src/families/sigma_link.rs:48`
- `src/families/sigma_link.rs:66`
- `src/families/sigma_link.rs:95`

Definition in code:

`safe_exp(eta) = exp(clamp(eta, -700, 700))`

That means `safe_exp` is exactly constant on both plateaus:

- for every `eta > 700`, `safe_exp(eta) = exp(700)`
- for every `eta < -700`, `safe_exp(eta) = exp(-700)`

So on either plateau:

- `d/deta safe_exp(eta) = 0`
- `d²/deta² safe_exp(eta) = 0`
- `d³/deta³ safe_exp(eta) = 0`
- `d⁴/deta⁴ safe_exp(eta) = 0`

But the code returns:

- `d1 = safe_exp(eta)`
- `d2 = safe_exp(eta)`
- `d3 = safe_exp(eta)`
- `d4 = safe_exp(eta)`

Counterexample:

- At `eta = 701`, the implemented function is already on the constant upper plateau.
- Finite differences of `safe_exp` at `701` are exactly zero.
- The analytic helpers still return `exp(700)`, which is nonzero by over 300 orders of magnitude.

Failing test:

- `exp_sigma_derivatives_follow_the_clamped_safe_exp_definition`

## 2. `q_chain_derivs_scalar` differentiates through a floored denominator that is locally constant

Code:

- `src/families/survival_location_scale.rs:813`
- `src/families/survival_location_scale.rs:814`
- `src/families/survival_location_scale.rs:962`

Implemented predictor:

`q(eta_t, eta_ls) = -eta_t / max(sigma(eta_ls), 1e-12)`

When `sigma(eta_ls) < 1e-12`, the denominator is frozen at `1e-12`, so `q` is locally constant in
`eta_ls`. Therefore every derivative with respect to `eta_ls` must vanish on the active floor branch:

- `dq/deta_ls = 0`
- `d²q/(deta_t deta_ls) = 0`
- `d²q/deta_ls² = 0`
- `d³q/(deta_t deta_ls²) = 0`
- `d³q/deta_ls³ = 0`

But `q_chain_derivs_scalar` computes those derivatives using the unfloored
`dsigma`, `d2sigma`, and `d3sigma`, even when `sigma` has already been replaced by
`max(sigma, 1e-12)` in the actual predictor.

Counterexample:

- Let `eta_t = 2` and `eta_ls = -30`.
- Then `sigma = exp(-30) < 1e-12`, so the implemented predictor is
  `q = -2 / 1e-12 = -2e12`, which is locally constant in `eta_ls`.
- The correct `eta_ls` derivatives are all zero.
- The code instead returns nonzero values, for example the failing test observes
  `dq/deta_ls = 0.1871524593768035` instead of `0`.

Failing test:

- `q_chain_derivatives_vanish_when_sigma_floor_is_active`

## 3. CLogLog and LogLog inverse-link derivatives overflow to `NaN` instead of decaying to zero

Code:

- `src/solver/mixture_link.rs:436`
- `src/solver/mixture_link.rs:473`
- `src/solver/mixture_link.rs:145`
- `src/solver/mixture_link.rs:163`

### CLogLog

The code uses:

- `t = exp(eta)`
- `s = exp(-t)`
- `d1 = t * s`

For large finite positive `eta`, `t` overflows to `inf` and `s` becomes `0`, so
the product becomes `inf * 0 = NaN`.

But mathematically:

`d1 = exp(eta - exp(eta))`

and this tends to `0`, not `NaN`. The same problem propagates into `d2`, `d3`,
and the fourth derivative helper.

Counterexample:

- At `eta = 800`, the exact floating-point result should be:
  - `mu = 1`
  - `d1 = 0`
  - `d2 = 0`
  - `d3 = 0`
  - `d4 = 0`
- The implementation produces `NaN` derivatives.

Failing test:

- `cloglog_large_finite_eta_should_saturate_without_nan_derivatives`

### LogLog

The code uses:

- `r = exp(-eta)`
- `mu = exp(-r)`
- `d1 = mu * r`

For large finite negative `eta`, `r` overflows to `inf` and `mu` becomes `0`, so
again the product becomes `0 * inf = NaN`.

But mathematically:

`d1 = exp(-eta - exp(-eta))`

which also tends to `0`, not `NaN`.

Counterexample:

- At `eta = -800`, the exact floating-point result should be:
  - `mu = 0`
  - `d1 = 0`
  - `d2 = 0`
  - `d3 = 0`
- The implementation produces `NaN` derivatives.

Failing test:

- `loglog_large_negative_finite_eta_should_saturate_without_nan_derivatives`

## 4. The closed-form Matérn kernel can return `NaN` at huge finite distances even though the kernel must decay to zero

Code:

- `src/terms/basis.rs:2411`
- `src/terms/basis.rs:2507`

For half-integer Matérn kernels, the implementation uses formulas of the form:

`poly(a) * exp(-a)`

with `a = sqrt(2 nu) * r / length_scale`.

For sufficiently large finite `r`, the polynomial part overflows to `inf` before
the exponential is multiplied in, so the final result becomes `inf * 0 = NaN`.

But for every supported half-integer `nu`, the Matérn kernel is exponentially
decaying in `r`, so at huge finite distance its floating-point value should be
exactly `0`, not `NaN`. The same issue contaminates the log-kappa derivative helper.

Counterexample:

- With `nu = 9/2`, `length_scale = 1`, and `r = 1e308`, the exact kernel is far below
  the smallest positive `f64`, so the correct floating-point result is `0`.
- The current formula can produce `NaN` because the polynomial factor overflows first.

Failing test:

- `matern_closed_form_should_decay_to_zero_not_nan_at_huge_distance`
