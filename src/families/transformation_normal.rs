//! Conditional transformation model: estimate h(y|x) such that h(Y|x) ~ N(0,1).
//!
//! Given a response variable y and covariates x with a pre-built covariate design
//! operator, this family estimates a smooth monotone transformation h(y | x) mapping
//! the conditional distribution of Y|x onto a standard normal.
//!
//! The response-direction basis is `[1, I_1(y), ..., I_K(y)]`, tensored with an
//! arbitrary covariate design operator. Column 0 is an unconstrained location
//! component `b(x)`. The I-spline columns are shape components with squared
//! covariate-side coefficients, giving the SCOP representation
//! `h(y, x) = b(x) + ε·(y−median_y) + Σ_k I_k(y) γ_k(x)^2` and
//! `h'(y, x) = ε + Σ_k M_k(y) γ_k(x)^2`. Monotonicity is structural:
//! the fixed derivative floor `ε` keeps the change-of-variables log-density
//! away from the `log(0)` singularity, while the non-negative M-spline basis
//! and squared covariate-side coefficients supply the learned shape.
//!
//! The log-likelihood per observation is the finite-support normalized
//! change-of-variables density for a standard normal target:
//!
//!   ℓ_i = -½ h_i² + log(h'_i) - log(Φ(h_U(x_i)) - Φ(h_L(x_i)))
//!
//! where `h_i = b(x_i) + ε·(y_i−median_y) + Σ_k I_k(y_i) γ_k(x_i)^2`
//! and `h'_i = ε + Σ_k M_k(y_i) γ_k(x_i)^2`. The endpoint normalizer is
//! required because the I-spline response basis saturates at finite support
//! values rather than mapping onto the full real line.

// Split from the original oversized module; keep included in order.
include!("transformation_normal/imports.rs");

mod endpoint_normalizer;

include!("transformation_normal/family_core.rs");
include!("transformation_normal/custom_family_impl.rs");
