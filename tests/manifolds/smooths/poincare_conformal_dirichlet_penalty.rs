//! Property + discrimination gate for the hyperbolic-SAE roughness operator
//! `poincare::conformal_dirichlet_penalty` (the conformal-reweighted Dirichlet
//! Gram that turns a flat tangent patch into a hyperbolic patch).
//!
//! The operator is the single source of truth for hyperbolic atom smoothness.
//! These tests pin the properties that make it a *correct* and *useful*
//! roughness penalty, with truth we construct ourselves (no reference tool):
//!
//! 1. **Symmetric PSD** — a roughness Gram must be a valid quadratic form: it
//!    is symmetric and `βᵀSβ ≥ 0` for every coefficient vector.
//! 2. **Closed-form pullback (`d = 2`, `c = −1`)** — the penalised field is a
//!    function of the *tangent* coordinate `t`, and `p = exp₀(t)` is not a
//!    conformal chart, so the energy must integrate against the pullback metric
//!    `h(t) = J(t)ᵀ λ² J(t)`, which in polar tangent coordinates is
//!    `4 dr² + sinh²(2r) dθ²`. The assembled Gram must equal
//!    `Σ_n Φ'(t_n)ᵀ (√det h · h⁻¹) Φ'(t_n)` built from that closed form — and it
//!    must **not** equal the flat tangent-coordinate Dirichlet Gram (the old
//!    implementation, which conflated the ball and tangent charts).
//! 3. **`d = 1` half-speed tangent coordinate** — a 1-D hyperbolic manifold is
//!    intrinsically flat, but the tangent coordinate runs at half arc-length
//!    (`geodesic dist = 2‖t‖`), so the per-row weight the operator applies is the
//!    exact constant `1/2` at every radius, independent of the boundary
//!    proximity. We pin that constant.

