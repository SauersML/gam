# Mathematical Mistakes Audit

This file records only defects that are provably wrong from the code itself.
No proposed fixes are included here. Each item has a direct counterexample and
at least one intentionally failing test that makes the defect visible.

The strongest findings are derivation-level mismatches:

- the code differentiates through active floors or plateaus even though the
  coded scalar function is locally constant there
- or the code injects `exp(eta)` derivatives into exact-Newton paths even
  though the actual sigma link is `safe_exp(eta) = exp(clamp(eta, -700, 700))`

Those are mathematical errors in the implemented derivatives themselves, not
just floating-point tail accuracy problems.

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

## 5. `BinomialLocationScaleFamily` builds a nonzero log-sigma working weight on a branch where the coded mean is exactly constant

Code:

- `src/families/gamlss.rs:3236`
- `src/families/gamlss.rs:3253`

Diagonal working-set logic:

- `q0(eta_t, eta_ls) = -eta_t / max(sigma(eta_ls), 1e-12)`
- `chain_ls = -q0 * dsigma/deta_ls / max(sigma, 1e-12)`
- `dmu_ls = dmu/dq * chain_ls`
- `w_ls = weight * dmu_ls^2 / var`

But once `sigma(eta_ls) < 1e-12`, the actual coded predictor `q0` is frozen:

- `q0(eta_ls) = -eta_t / 1e-12`

So the coded mean `mu(eta_ls)` is locally constant in `eta_ls`, and therefore:

- `dmu/deta_ls = 0`
- the log-sigma working weight must be `0`

Counterexample:

- Take `eta_t = 1e-13` and `eta_ls = -30`.
- Then `sigma = exp(-30) < 1e-12`, so `q0 = -0.1` is exactly constant under
  small perturbations of `eta_ls`.
- A central finite difference of the coded mean gives `dmu/deta_ls = 0`.
- The implementation still produces a positive log-sigma working weight:
  `0.000055543424612589786`.

Failing test:

- `binomial_location_scale_log_sigma_working_weight_should_vanish_on_sigma_floor_branch`

## 6. `BinomialLocationScaleFamily` exact-Newton log-sigma curvature differentiates `exp(eta_ls)` instead of the coded `safe_exp(eta_ls)`

Code:

- `src/families/gamlss.rs:6792`
- `src/families/gamlss.rs:6793`

The exact-Newton non-wiggle path injects:

`d2sigma_deta2 = exp(eta_ls)`

But the actual sigma link used throughout the family is:

`sigma(eta_ls) = safe_exp(eta_ls) = exp(clamp(eta_ls, -700, 700))`

So for every `eta_ls > 700`, the coded sigma is constant and the coded
log-likelihood is locally flat in `eta_ls`. On that plateau:

- `d/deta_ls log L = 0`
- `d²/deta_ls² log L = 0`

Counterexample:

- Let `eta_ls = 701` and `eta_t = 0.1 * exp(700)`, so the coded quotient `q0`
  stays moderate while `safe_exp` is already on its upper plateau.
- The coded scalar log-likelihood at `701 - 1e-4`, `701`, and `701 + 1e-4` is
  exactly the same, so finite-difference score and curvature are both `0`.
- The exact-Newton log-sigma information returned by the code is `NaN`.

Failing test:

- `binomial_location_scale_exact_log_sigma_block_should_be_flat_on_safe_exp_plateau`

## 7. The same exact-Newton plateau derivation error is duplicated in `BinomialLocationScaleWiggleFamily`

Code:

- `src/families/gamlss.rs:9335`
- `src/families/gamlss.rs:9355`

The wiggle exact-Newton path independently injects:

`d2sigma_deta2 = exp(eta_ls)`

even though its sigma link is the same clamped `safe_exp`.

Counterexample:

- Use the wiggle family with `betaw = 0`, `beta_log_sigma = 701`, and
  `beta_t = 0.1 * exp(700)`.
- Because `safe_exp(701 ± 1e-4) = exp(700)`, both `q0` and the coded
  log-likelihood are locally constant in `beta_log_sigma`.
- Finite-difference score and curvature are both `0`.
- The exact-Newton log-sigma information returned by the code is `NaN`.

Failing test:

- `binomial_location_scalewiggle_exact_log_sigma_block_should_be_flat_on_safe_exp_plateau`

## 8. `SurvivalLocationScaleFamily` exact-Newton log-sigma curvature differentiates through the `safe_exp` plateau

Code:

- `src/families/survival_location_scale.rs:551`
- `src/families/survival_location_scale.rs:553`

The exact joint-Hessian path takes `(sigma, ds, d2s, d3s)` from
`exp_sigma_derivs_up_to_third(eta_ls_exit)`. But those helpers treat the clamped
`safe_exp` link as if all derivatives still equaled `sigma`, even when
`eta_ls_exit > 700` and the coded sigma is constant.

Therefore on the plateau the coded survival log-likelihood is locally flat in
the log-sigma coefficient, so its exact-Newton log-sigma score and information
must both be zero.

Counterexample:

- Let `beta_log_sigma = 701` and choose `beta_threshold = 0.1 * exp(700)` so
  the coded transformed predictor remains moderate while `safe_exp` is already
  constant.
- The coded scalar survival log-likelihood at `701 - 1e-4`, `701`, and
  `701 + 1e-4` is identical, so finite-difference score and curvature are `0`.
- The exact-Newton log-sigma information returned by the code is `NaN`.

Failing test:

- `joint_exact_newton_log_sigma_block_should_be_flat_on_safe_exp_plateau`

## 9. `GaussianLocationScaleFamily` has a separate sigma helper that does not even evaluate the same sigma function as the family itself

Code:

- `src/families/gamlss.rs:3761`
- `src/families/gamlss.rs:3770`

The helper

`gaussian_sigma_derivs_up_to_fourth(eta)`

returns

- `sigma = exp(eta)`
- `d1 = exp(eta)`
- `d2 = exp(eta)`
- `d3 = exp(eta)`
- `d4 = exp(eta)`

But the Gaussian family evaluation code uses the coded sigma link

`sigma(eta) = max(safe_exp(eta), 1e-12) = max(exp(clamp(eta, -700, 700)), 1e-12)`.

So the helper is not merely giving the wrong derivatives on the plateau; it is
evaluating a different scalar function altogether.

Counterexample:

- At `eta = 701`, the coded family sigma is `exp(700)`.
- `gaussian_sigma_derivs_up_to_fourth` instead returns `exp(701)`.
- Because the coded sigma link is already constant on that plateau, all
  finite-difference derivatives of the coded sigma are `0`.
- The helper still returns nonzero first through fourth derivatives.

Failing test:

- `gaussian_sigma_helper_uses_a_different_function_than_the_coded_sigma_link`
