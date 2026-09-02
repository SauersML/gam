//! Regression: a Gaussian REML fit must be INVARIANT to a global rescaling of
//! the prior weights by a positive constant `c`.
//!
//! `gamfit` treats `weights` as inverse-variance weights with a *profiled*
//! dispersion: `Var(yᵢ) = φ / wᵢ` with `φ` estimated, exactly as mgcv does.
//! Under that convention the absolute magnitude of the weights carries no
//! information — only their ratios do. Multiplying every weight by the same
//! `c > 0` is absorbed entirely by the profiled scale `φ̂ → c·φ̂`, so the
//! selected smoothing parameters, the effective degrees of freedom, and the
//! fitted coefficients (hence every prediction) must be unchanged. (The
//! library already honours this for the *uncertainty*: the conditional SEs are
//! invariant to weight rescaling, because `φ̂·(XᵀWX)⁻¹` is — `XᵀWX` scales by
//! `c`, `φ̂` scales by `c`, the product is fixed.)
//!
//! The same invariance is visible directly in the Gaussian REML objective the
//! engine minimises (`src/solver/reml/unified.rs:7248`, the
//! `DispersionHandling::ProfiledGaussian` arm):
//!
//!   V(ρ) = D_p/(2φ̂) + ½·log|H| − ½·log|S|₊ + ((n−M_p)/2)·log(2π φ̂),
//!   with  D_p = deviance + penalty,  φ̂ = D_p/(n − M_p).
//!
//! Send `W → c·W` and the invariance-preserving smoothing `λ → c·λ` (so the
//! penalised Hessian `H = XᵀWX + λS → c·H` and β̂ is fixed). Then:
//!   • `D_p → c·D_p`, `φ̂ → c·φ̂`  ⇒  `D_p/(2φ̂)` is unchanged;
//!   • `½(log|H| − log|S|₊) → ½(p − r)·log c = ½·M_p·log c`;
//!   • `((n−M_p)/2)·log(2π φ̂) → +((n−M_p)/2)·log c`.
//! The two `log c` pieces sum to `(n/2)·log c`, a constant **independent of ρ**.
//! A constant offset cannot move an argmin, so the minimiser obeys
//! `λ̂(c·W) = c·λ̂(W)` exactly and the fit (β̂, EDF, predictions) is invariant.
//!
//! Observed (this fixture, debug build): rescaling all weights from 1 to
//! `c = 1000` moves the selected λ by a factor ≈ 810 (not 1000), shifts the
//! effective dof by ≈ 1.6e-2, and moves the fitted coefficients by ≈ 4e-3 —
//! a genuine change in the fitted function from a no-op rescaling. The
//! `(n−M_p)/2·log(2π φ̂)` term grows like `(n/2)·log c` with the weight scale,
//! inflating the absolute objective value while leaving its shape fixed; the
//! outer optimiser then stops short of the (shifted) optimum, so the
//! invariance that holds in exact arithmetic is broken in the implementation.
//!
//! This test fits the SAME `y ~ s(x)` Gaussian model twice — once with all
//! weights `1`, once with all weights `c = 1000` — and asserts the selected
//! smoothing parameter scales exactly by `c` and that the EDF and the fitted
//! coefficients are invariant. It fails today and will pass once the Gaussian
//! REML smoothing selection is made weight-scale invariant.

