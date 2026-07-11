//! Regression: `SmoothThresholdPenalty::psd_majorizer_diag` majorizes the
//! exact Hessian for any coordinate below (or just at) its threshold, violating
//! the `AnalyticPenalty::psd_majorizer_diag` contract.
//!
//! The trait documents the majorizer as a PSD *upper bound* on the exact
//! curvature (src/terms/analytic_penalties/mod.rs:410-418):
//!
//!     "Diagonal of a **PSD majorizer** of the Hessian — the positive
//!      re-weighted-ℓ₂ / MM surrogate `diag(B)` with `B ⪰ ∂²P/∂target²`
//!      everywhere and `B ⪰ 0`."
//!
//! That `B ⪰ ∂²P` is the property the inner Newton / PIRLS MM step relies on: a
//! surrogate quadratic with curvature `B` upper-bounds `P` only when `B`
//! dominates the true Hessian, which is what guarantees the monotone-decrease
//! of majorization-minimization.
//!
//! For the smooth threshold prior the two diagonal entries are
//! 3060-3068), with `g = sigmoid((x - tau)/eps)` and `C = weight*tau/eps² > 0`:
//!
//!     exact Hessian   h(g) = C · g(1-g)(1-2g)        (`true_hessian_diag_entry`)
//!     majorizer       m(g) = C · [g(1-g)]²           (`psd_hessian_diag_entry`)
//!
//! The majorizer is always ≥ 0, so it correctly dominates the *concave* region
//! `g > 1/2` (where `h < 0`). But in the *convex* region `g < 1/2` the exact
//! curvature is positive, and
//!
//!     m(g) ≥ h(g)  ⟺  [g(1-g)]² ≥ g(1-g)(1-2g)  ⟺  g(1-g) ≥ 1-2g
//!                  ⟺  g² - 3g + 1 ≤ 0            ⟺  g ≥ (3-√5)/2 ≈ 0.3820.
//!
//! So for every coordinate with gate `g < 0.382` — i.e. `x < tau - 0.481·eps`,
//! the entire "comfortably below threshold" region — the majorizer is strictly
//! **less** than the positive exact Hessian (≈7× smaller at `g = 0.12`). There
//! the MM surrogate *under*-estimates curvature, so it is not an upper bound and
//! the majorization guarantee is lost for exactly the inactive latent
//! coordinates the smooth threshold prior is meant to keep suppressed.
//!
//! Reproduction is closed-form: pick `tau = 1`, `eps = 0.5`, and `x = 0`
//! (`g ≈ 0.119`). The exact Hessian diagonal is positive (`≈ 0.416·weight`) but
//! the majorizer (`≈ 0.057·weight`) is far below it. The assertion below encodes
//! the trait contract `majorizer ≥ exact` at that point; it fails today and will
//! pass once `psd_majorizer_diag` returns a genuine upper bound on `h(g)` over
//! the convex region.
//!
//! Root cause: `psd_hessian_diag_entry` (src/terms/analytic_penalties/mod.rs:3065)
//! only guarantees non-negativity, not domination of the exact Hessian.
//! Related: #794 (sibling SAE-penalty curvature defect: MonotonicityPenalty::hvp
//! magnitude), #793.

use gam::terms::analytic_penalties::{AnalyticPenalty, PsiSlice, SmoothThresholdPenalty};
use ndarray::Array1;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[test]
fn smooth_threshold_psd_majorizer_dominates_exact_hessian_below_threshold() {
    let weight = 1.3_f64;
    let eps = 0.5_f64;
    let tau = 1.0_f64;

    // latent_dim = 1; four coordinates spanning below / at / above threshold.
    let p = SmoothThresholdPenalty::new(
        PsiSlice::full(4, Some(1)),
        Array1::from(vec![tau]),
        weight,
        eps,
    )
    .expect("construct smooth threshold penalty");
    // rho_count == latent_dim; rho = 0 keeps the threshold at its base `tau`.
    let rho = Array1::from(vec![0.0_f64]);
    let target = Array1::from(vec![0.0, 0.7, 1.0, 2.0]);

    let exact = p
        .hessian_diag(target.view(), rho.view())
        .expect("smooth threshold has a closed-form diagonal Hessian");
    let majorizer = p
        .psd_majorizer_diag(target.view(), rho.view())
        .expect("smooth threshold exposes a PSD majorizer diagonal");

    // Contract: the majorizer must dominate the exact Hessian at EVERY
    // coordinate (B ⪰ ∂²P), and in particular wherever the exact curvature is
    // positive (the convex, below-threshold region).
    let tol = 1e-12;
    for i in 0..target.len() {
        let g = sigmoid((target[i] - tau) / eps);
        assert!(
            majorizer[i] + tol >= exact[i],
            "PSD majorizer fails to dominate the exact Hessian at x = {} (gate g = {g:.4}): \
             majorizer = {:.6} < exact = {:.6}. The trait requires B ⪰ ∂²P everywhere.",
            target[i],
            majorizer[i],
            exact[i]
        );
    }
}
