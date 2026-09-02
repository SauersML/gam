//! Owed-work regression gate for GitHub issue #1418.
//!
//! ISSUE: the implicit-function (IFT) correction in the SAE outer-ρ REML
//! gradient is `−½·Γᵀ·θ̂_ρ` with the sensitivity `θ̂_ρ = −A⁻¹ ∂g/∂ρ`, where
//! `A = ∇²_θθ L` is the EXACT stationarity Jacobian of the inner fit. The matrix
//! the inner solve actually factors is a stability-conditioned surrogate `B`
//! (Gauss-Newton data curvature with the residual-curvature term `⟨r, ∂²f⟩`
//! dropped; the softmax entropy Hessian replaced by its Gershgorin/Fisher PSD
//! majorizer; the periodic ARD curvature `V''` replaced by `max(V'',0)`). The
//! `½log|B|` Laplace value term is consistent with `Γ = ½tr(B⁻¹ ∂B/∂θ)`, but the
//! implicit STEP must use the exact `A`, not `B`. Using `B` for the IFT solve
//! biases the correction by `(B⁻¹ − A⁻¹)`, which is nonzero exactly when the
//! dropped curvature `ΔC = A − B` is nonzero (large residual, indefinite entropy,
//! periodic ARD past a quarter period).
//!
//! FIX (landed in `analytic_outer_rho_gradient_components`,
//! `solve_exact_stationarity`, `apply_exact_hessian_minus_b` in
//! `src/terms/sae/manifold/construction.rs`): the IFT correction now applies the
//! TRUE `A⁻¹` via a `B⁻¹`-preconditioned Neumann fixed point
//! (`A = B + ΔC`, `x = B⁻¹ rhs − B⁻¹(ΔC x)`), with `ΔC` assembled matrix-free
//! over all three dropped channels. `B` survives only as the preconditioner.
//!
//! CERTIFICATE (public API only, survives refactors of the private solver): the
//! full analytic outer-ρ gradient — explicit + direct log-det traces + Occam +
//! the implicit-state correction — must match a centered finite difference of the
//! actual custom quasi-Laplace criterion (the inner problem is re-solved at each perturbed ρ, so
//! the FD carries the true envelope/IFT terms governed by `A`). The fixture is
//! deliberately built with a LARGE, unmodellable residual at the inner optimum on
//! a curved (periodic-harmonic) basis, so the dropped residual curvature
//! `⟨r, ∂²f⟩` is genuinely large: were the implicit step still using `B`, the
//! analytic gradient would deviate from the FD by `O(‖B⁻¹ − A⁻¹‖)` and the bound
//! below would fail.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

