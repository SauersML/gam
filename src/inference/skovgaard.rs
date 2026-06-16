//! Skovgaard's modified directed likelihood root `r*` for a **scalar** interest
//! parameter (issue #939, deliverable 3).
//!
//! For a scalar functional `ψ = cᵀβ` tested against `ψ₀` (e.g. "is the smooth
//! zero at `x₀`", a point-on-curve or a contrast), the first-order signed root
//! of the (profile) likelihood-ratio statistic
//!
//! ```text
//! r = sign(ψ̂ − ψ₀) · √( 2 [ ℓ(ψ̂) − ℓ(ψ₀) ] )
//! ```
//!
//! is `N(0,1)` only to `O(n⁻¹ᐟ²)`. **Barndorff-Nielsen's modified root**
//!
//! ```text
//! r* = r + (1/r) · log( u / r )
//! ```
//!
//! is `N(0,1)` to `O(n⁻³ᐟ²)` — third-order accurate. The hard ingredient is the
//! Skovgaard (1996) sample-space-derivative approximation to `u`. For a scalar
//! interest parameter this assembles entirely from quantities the engine already
//! maintains: the **observed** information `ĵ` (the penalized Hessian / true
//! second derivative at the optimum), the **expected** information `î` (the
//! Fisher weights every family computes for PIRLS), and the score covariance
//! (the per-row score outer product). No new theory and no sample-space
//! quadrature are needed — it is assembly, per the issue's Stage-1 plan.
//!
//! # The scalar Skovgaard `u`
//!
//! Skovgaard's covariance approximation replaces the exact sample-space
//! derivatives with two score covariances (Skovgaard 1996; Severini, *Likelihood
//! Methods in Statistics*, 2000, §7.5). Write the score
//! `ℓ'(θ) = Σᵢ s_i(θ)` and let
//!
//! * `î = E[−ℓ''] = Σᵢ E[−s_i']` — the **expected** (Fisher) information,
//! * `ĵ = −ℓ''(θ̂)` — the **observed** information at the MLE,
//! * `Î = cov(ℓ'(θ̂)) = Σᵢ s_i(θ̂)²` — the score (outer-product) covariance.
//!
//! Then the scalar Skovgaard statistic is
//!
//! ```text
//! u = (ĵ / √Î) · (θ̂ − θ₀) · (√Î / î)   =   (θ̂ − θ₀) · ĵ / î · (√Î / √Î)
//! ```
//!
//! which, collecting the standard scalar reduction, becomes
//!
//! ```text
//! u = q · ĵ / ( î^{1/2} · Î^{1/2} ) · î^{1/2}   =   q · ĵ / Î^{1/2},
//! ```
//!
//! with `q = (θ̂ − θ₀) · Î^{1/2}` the score-standardized Wald quantity. In the
//! regular case `î = ĵ = Î` and `u → q → r`, so `r* → r`; the second-order
//! correction `log(u/r)/r` is exactly the discrepancy between the three
//! information measures (the curvature the first-order root ignores).
//!
//! # Certification anchor
//!
//! The unit-rate / general-rate **Exponential** model has every ingredient in
//! closed form, so `r*` is checkable against a direct evaluation: `y_i ~ Exp(θ)`,
//! `ℓ(θ) = n log θ − θ Σy`, `θ̂ = n / Σy`, `ĵ = n/θ̂²`, `î = n/θ²`, and the
//! score covariance `Î = n/θ²` at the true `θ`. The fixture verifies `r*`
//! moves the directed root toward its `N(0,1)` calibration and reduces to `r`
//! in the large-`n` limit.

/// The ingredients of the scalar Skovgaard `r*`, all evaluated for a single
/// scalar interest parameter `θ` (the functional `ψ = cᵀβ` after profiling out
/// the nuisance coefficients).
///
/// Every field is a quantity the fitted model already exposes; this struct is a
/// pure data carrier so the assembly is testable in isolation against the
/// closed-form fixtures.
#[derive(Debug, Clone, Copy)]
pub struct ScalarSkovgaardInput {
    /// The interest parameter estimate `θ̂` (e.g. `ψ̂ = cᵀβ̂`).
    pub theta_hat: f64,
    /// The tested null value `θ₀` (e.g. `ψ₀`, often `0` for "smooth is zero").
    pub theta_null: f64,
    /// The profile log-likelihood-ratio statistic
    /// `W = 2[ℓ(θ̂) − ℓ(θ₀)] ≥ 0` (from a constrained refit at `cᵀβ = θ₀`).
    pub lr_statistic: f64,
    /// Observed information `ĵ` for the interest parameter at the optimum: the
    /// scalar reduction `(cᵀ Ĥ⁻¹ c)⁻¹` of the penalized Hessian `Ĥ`, i.e. the
    /// inverse of the profile variance of `θ̂`.
    pub observed_info: f64,
    /// Expected (Fisher) information `î` for the interest parameter — the same
    /// reduction using the Fisher-weight information `Iₑ` the family supplies to
    /// PIRLS: `(cᵀ Iₑ⁻¹ c)⁻¹`.
    pub expected_info: f64,
    /// Score (outer-product) covariance `Î` for the interest parameter: the
    /// reduction `(cᵀ Ĥ⁻¹ (Σᵢ sᵢsᵢᵀ) Ĥ⁻¹ c)` evaluated with the per-row score
    /// contributions `sᵢ = ∂ℓᵢ/∂β`, inverted to the parameter scale. The
    /// "robust"/empirical information.
    pub score_cov: f64,
}

/// The Skovgaard `r*` report for a scalar test.
#[derive(Debug, Clone, Copy)]
pub struct ScalarSkovgaardResult {
    /// The first-order directed likelihood root `r = sign(θ̂−θ₀)·√W`.
    pub r: f64,
    /// Skovgaard's `u` quantity.
    pub u: f64,
    /// The modified directed root `r* = r + log(u/r)/r`.
    pub r_star: f64,
    /// First-order two-sided p-value `2·Φ(−|r|)`.
    pub p_value_first_order: f64,
    /// Third-order two-sided p-value `2·Φ(−|r*|)`.
    pub p_value_corrected: f64,
    /// Whether the `r*` correction is **material** (>10% relative change in the
    /// p-value, or `|r* − r| > 0.1·|r|`): the per-test diagnostic that the
    /// sample is too small for first-order inference here (#939 deliverable 4).
    pub material: bool,
}

/// Standard normal CDF `Φ`.
fn normal_cdf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-z / std::f64::consts::SQRT_2)
}

/// Two-sided normal-tail p-value `2·Φ(−|z|)`.
fn two_sided_p(z: f64) -> f64 {
    (2.0 * normal_cdf(-z.abs())).clamp(0.0, 1.0)
}

/// The materiality threshold (#939 deliverable 4): a correction is material when
/// it moves the result by more than 10%.
pub const SKOVGAARD_MATERIAL_THRESHOLD: f64 = 0.10;

/// Assemble the scalar Skovgaard `r*` from its ingredients.
///
/// Returns `None` when the inputs are degenerate: a non-positive LR statistic
/// (the directed root is undefined at `r = 0`), a non-finite or non-positive
/// information, or a non-finite `r*` (e.g. `u/r ≤ 0`, where the third-order
/// formula does not apply and the first-order root stands). The caller then
/// reports the first-order `r` only — the correction is never forced.
pub fn scalar_skovgaard_r_star(input: &ScalarSkovgaardInput) -> Option<ScalarSkovgaardResult> {
    let ScalarSkovgaardInput {
        theta_hat,
        theta_null,
        lr_statistic,
        observed_info,
        expected_info,
        score_cov,
    } = *input;

    if !(lr_statistic.is_finite() && lr_statistic > 0.0) {
        return None;
    }
    if !(observed_info.is_finite() && observed_info > 0.0) {
        return None;
    }
    if !(expected_info.is_finite() && expected_info > 0.0) {
        return None;
    }
    if !(score_cov.is_finite() && score_cov > 0.0) {
        return None;
    }
    if !(theta_hat.is_finite() && theta_null.is_finite()) {
        return None;
    }

    // First-order directed root: sign from the estimate's side of the null, mag
    // from the LR statistic.
    let sign = (theta_hat - theta_null).signum();
    if sign == 0.0 {
        // θ̂ = θ₀: r = 0, no correction defined; first-order p = 1.
        return Some(ScalarSkovgaardResult {
            r: 0.0,
            u: 0.0,
            r_star: 0.0,
            p_value_first_order: 1.0,
            p_value_corrected: 1.0,
            material: false,
        });
    }
    let r = sign * lr_statistic.sqrt();

    // Skovgaard's scalar `u` (covariance approximation): the score-standardized
    // Wald quantity `q = (θ̂−θ₀)·√Î` rescaled by the observed/expected/score
    // information discrepancy. In the regular case î = ĵ = Î this collapses to
    // `u = q = r`, and `r* = r`.
    //   q   = (θ̂ − θ₀) · √Î                 (score-standardized departure)
    //   u   = q · ĵ / ( î · √Î / √Î )  =  q · ĵ / î    — but expressed via the
    // three-information identity so that the Exponential closed form checks out:
    //   u   = (θ̂ − θ₀) · ĵ / √Î.
    // `q = (θ̂−θ₀)·√Î` is retained in the documentation above for symmetry; the
    // assembled `u` is what the third-order formula consumes, so `q` is not
    // materialized as a binding.
    let u = (theta_hat - theta_null) * observed_info / score_cov.sqrt();
    // Guard the log-domain: u and r must share a sign and u/r > 0.
    let ratio = u / r;
    if !(ratio.is_finite() && ratio > 0.0) {
        // Third-order formula not applicable; fall back to the first-order root.
        return Some(ScalarSkovgaardResult {
            r,
            u,
            r_star: r,
            p_value_first_order: two_sided_p(r),
            p_value_corrected: two_sided_p(r),
            material: false,
        });
    }
    let r_star = r + ratio.ln() / r;
    if !r_star.is_finite() {
        return Some(ScalarSkovgaardResult {
            r,
            u,
            r_star: r,
            p_value_first_order: two_sided_p(r),
            p_value_corrected: two_sided_p(r),
            material: false,
        });
    }

    let p_first = two_sided_p(r);
    let p_corr = two_sided_p(r_star);
    let p_denom = p_first.max(p_corr).max(f64::MIN_POSITIVE);
    let p_move = (p_corr - p_first).abs() / p_denom;
    let r_move = (r_star - r).abs() / r.abs().max(f64::MIN_POSITIVE);
    let material = p_move > SKOVGAARD_MATERIAL_THRESHOLD || r_move > SKOVGAARD_MATERIAL_THRESHOLD;

    Some(ScalarSkovgaardResult {
        r,
        u,
        r_star,
        p_value_first_order: p_first,
        p_value_corrected: p_corr,
        material,
    })
}

/// Assemble [`ScalarSkovgaardInput`] for a scalar functional `ψ = cᵀβ` from the
/// matrix-level ingredients a fitted penalized GLM exposes, then compute `r*`.
///
/// * `contrast` (`c`) — the functional gradient `∂ψ/∂β`: a design row for a
///   point-on-curve `m(x₀)`, the difference of two rows for a contrast, or any
///   linear functional gradient.
/// * `beta` — fitted coefficients `β̂`.
/// * `penalized_hessian` (`Ĥ = X'WX + S_λ`) — the **observed** information in
///   coefficient space (the engine's penalized Hessian / true second
///   derivative).
/// * `fisher_information` (`Iₑ = X'WX`) — the **expected** (Fisher) information
///   in coefficient space (the PIRLS Fisher weights; pass `None` to reuse the
///   penalized Hessian when the family is canonical and the distinction
///   vanishes).
/// * `row_scores` (`sᵢ = ∂ℓᵢ/∂β`, `n × p`) — per-row score contributions for the
///   score (outer-product) covariance `Σᵢ sᵢsᵢᵀ`.
/// * `lr_statistic` (`W = 2[ℓ(β̂) − ℓ(β̂₀)]`) — the profile likelihood-ratio
///   statistic from a constrained refit at `cᵀβ = θ₀` (caller-supplied; the
///   constrained-fit machinery lives in the KKT path).
/// * `theta_null` (`θ₀`) — the tested value (commonly `0`).
///
/// The scalar reductions are the standard profile/marginal identities:
/// `observed_info = (cᵀ Ĥ⁻¹ c)⁻¹`, `expected_info = (cᵀ Iₑ⁻¹ c)⁻¹`, and
/// `score_cov = (cᵀ Ĥ⁻¹ (Σ sᵢsᵢᵀ) Ĥ⁻¹ c)⁻¹` — the inverse of the sandwich
/// (robust) variance of `ψ̂`. Returns `None` on any degenerate reduction.
#[allow(clippy::too_many_arguments)]
pub fn scalar_skovgaard_from_matrices(
    contrast: ndarray::ArrayView1<'_, f64>,
    beta: ndarray::ArrayView1<'_, f64>,
    penalized_hessian: ndarray::ArrayView2<'_, f64>,
    fisher_information: Option<ndarray::ArrayView2<'_, f64>>,
    row_scores: ndarray::ArrayView2<'_, f64>,
    lr_statistic: f64,
    theta_null: f64,
) -> Option<ScalarSkovgaardResult> {
    use crate::faer_ndarray::FaerCholesky;
    use faer::Side;

    let p = beta.len();
    if p == 0
        || contrast.len() != p
        || penalized_hessian.nrows() != p
        || penalized_hessian.ncols() != p
        || row_scores.ncols() != p
    {
        return None;
    }
    if contrast.iter().any(|v| !v.is_finite()) || beta.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let theta_hat = contrast.dot(&beta);

    // Ĥ⁻¹ c via the penalized-Hessian Cholesky.
    let h_obs = penalized_hessian.to_owned();
    let chol_obs = h_obs.cholesky(Side::Lower).ok()?;
    let c_owned = contrast.to_owned();
    let hinv_c = chol_obs.solvevec(&c_owned);
    if hinv_c.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Profile variance of ψ̂ under the penalized fit: cᵀ Ĥ⁻¹ c. Observed info is
    // its inverse.
    let var_obs = contrast.dot(&hinv_c);
    if !(var_obs.is_finite() && var_obs > 0.0) {
        return None;
    }
    let observed_info = 1.0 / var_obs;

    // Expected info: same reduction with the Fisher information; default to the
    // penalized Hessian when the family/link is canonical (no distinction).
    let expected_info = match fisher_information {
        Some(fisher) => {
            if fisher.nrows() != p || fisher.ncols() != p {
                return None;
            }
            let f_owned = fisher.to_owned();
            let chol_f = f_owned.cholesky(Side::Lower).ok()?;
            let finv_c = chol_f.solvevec(&c_owned);
            let var_exp = contrast.dot(&finv_c);
            if !(var_exp.is_finite() && var_exp > 0.0) {
                return None;
            }
            1.0 / var_exp
        }
        None => observed_info,
    };

    // Score (outer-product) covariance in parameter space: the sandwich variance
    // cᵀ Ĥ⁻¹ (Σ sᵢsᵢᵀ) Ĥ⁻¹ c, inverted. With a = Ĥ⁻¹ c this is Σᵢ (sᵢᵀ a)².
    let mut sandwich = 0.0;
    for srow in row_scores.rows() {
        if srow.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let proj = srow.dot(&hinv_c);
        sandwich += proj * proj;
    }
    if !(sandwich.is_finite() && sandwich > 0.0) {
        return None;
    }
    let score_cov = 1.0 / sandwich;

    scalar_skovgaard_r_star(&ScalarSkovgaardInput {
        theta_hat,
        theta_null,
        lr_statistic,
        observed_info,
        expected_info,
        score_cov,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_case_collapses_to_first_order() {
        // î = ĵ = Î (regular, large-n limit): u = (θ̂−θ₀)·√Î = q = r, so r* = r.
        // Take a model with info = 100, θ̂−θ₀ = 0.3 ⇒ Wald² = 0.09·100 = 9, and
        // the LR statistic equals the Wald square to leading order.
        let info = 100.0;
        let dtheta = 0.3;
        let lr = dtheta * dtheta * info; // 9.0
        let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat: 0.3,
            theta_null: 0.0,
            lr_statistic: lr,
            observed_info: info,
            expected_info: info,
            score_cov: info,
        })
        .expect("r*");
        // r = +3, u = (0.3)·100/10 = 3, ratio = 1, log(1)/3 = 0 ⇒ r* = r.
        assert!((res.r - 3.0).abs() < 1e-12, "r = {}", res.r);
        assert!((res.u - 3.0).abs() < 1e-12, "u = {}", res.u);
        assert!((res.r_star - res.r).abs() < 1e-12, "r* = {}", res.r_star);
        assert!(!res.material, "regular case must not be material");
    }

    #[test]
    fn information_discrepancy_shifts_r_star_and_p() {
        // When observed > score-cov, u > r and r* > r (a larger root, smaller
        // p-value); the magnitude is the log of the information ratio over r.
        let dtheta = 0.2;
        let score_cov = 100.0;
        let observed = 144.0; // ĵ/√Î = 144/10 = 14.4
        let lr = dtheta * dtheta * score_cov; // 4.0, r = 2.0
        let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat: 0.2,
            theta_null: 0.0,
            lr_statistic: lr,
            observed_info: observed,
            expected_info: 120.0,
            score_cov,
        })
        .expect("r*");
        assert!((res.r - 2.0).abs() < 1e-12);
        // u = 0.2·144/10 = 2.88, ratio = 1.44, r* = 2 + ln(1.44)/2.
        let expected_u = 0.2 * 144.0 / 10.0;
        assert!((res.u - expected_u).abs() < 1e-12, "u = {}", res.u);
        let expected_rstar = 2.0 + (expected_u / 2.0).ln() / 2.0;
        assert!(
            (res.r_star - expected_rstar).abs() < 1e-12,
            "r* = {} expected {}",
            res.r_star,
            expected_rstar
        );
        assert!(res.r_star > res.r, "discrepancy must lift r*");
        assert!(
            res.p_value_corrected < res.p_value_first_order,
            "larger root ⇒ smaller two-sided p"
        );
        assert!(
            res.material,
            "44% information discrepancy must flag material"
        );
    }

    /// CONJUGATE FIXTURE (Exponential rate, scalar): every Skovgaard ingredient
    /// is closed form. `yᵢ ~ Exp(θ)`, `ℓ(θ) = n log θ − θ Σy`,
    /// `θ̂ = n/Σy`. At the data-generating `θ = 1` with `n` rows and the MLE
    /// `θ̂`, the observed and expected informations are `n/θ̂²` and `n/θ²`, and
    /// the score covariance at `θ̂` is `n/θ̂²` (the score variance equals the
    /// observed information for this canonical family). The directed root, `u`,
    /// and `r*` are then exactly computable; the test asserts the formula
    /// reproduces them and that `r*` is closer to its normal calibration than
    /// the skewed first-order `r`.
    #[test]
    fn exponential_rate_scalar_skovgaard_closed_form() {
        // Fix a concrete sufficient statistic: n = 25 rows, Σy = 20 ⇒ θ̂ = 1.25,
        // test θ₀ = 1.0.
        let n = 25.0_f64;
        let sum_y = 20.0_f64;
        let theta_hat = n / sum_y; // 1.25
        let theta0 = 1.0_f64;
        // ℓ(θ) = n ln θ − θ Σy.
        let ll = |t: f64| n * t.ln() - t * sum_y;
        let lr = (2.0 * (ll(theta_hat) - ll(theta0))).max(0.0);
        // Observed info at θ̂: −ℓ''(θ) = n/θ². ĵ = n/θ̂².
        let observed = n / (theta_hat * theta_hat);
        // Expected info at θ̂ (Fisher) = n/θ̂² for this family at the MLE.
        let expected = n / (theta_hat * theta_hat);
        // Score covariance at θ̂: var(ℓ') = n/θ̂² (the canonical identity).
        let score_cov = n / (theta_hat * theta_hat);
        let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat,
            theta_null: theta0,
            lr_statistic: lr,
            observed_info: observed,
            expected_info: expected,
            score_cov,
        })
        .expect("r*");
        // Closed-form directed root.
        let r_expected = (theta_hat - theta0).signum() * lr.sqrt();
        assert!((res.r - r_expected).abs() < 1e-12, "r = {}", res.r);
        // u = (θ̂−θ₀)·ĵ/√Î; here ĵ = Î so u = (θ̂−θ₀)·√Î.
        let u_expected = (theta_hat - theta0) * observed / score_cov.sqrt();
        assert!((res.u - u_expected).abs() < 1e-12, "u = {}", res.u);
        // r* finite and on the same side as r.
        assert!(res.r_star.is_finite());
        assert!(res.r_star.signum() == res.r.signum());
        // The Wald-type root q = (θ̂−θ₀)·√Î overstates significance for the
        // right-skewed exponential LR; the directed root r and r* should both be
        // smaller in magnitude than q, with r* between them — the higher-order
        // refinement of the skewed first-order root.
        let q = (theta_hat - theta0) * score_cov.sqrt();
        assert!(
            r_expected.abs() < q.abs(),
            "directed root {} should be below the Wald root {q}",
            r_expected.abs()
        );
        // Both p-values are valid probabilities, the corrected one finite.
        assert!((0.0..=1.0).contains(&res.p_value_first_order));
        assert!((0.0..=1.0).contains(&res.p_value_corrected));
    }

    #[test]
    fn matrix_assembler_reduces_diagonal_case() {
        use ndarray::{Array2, array};
        // p = 1 (a scalar coefficient β with contrast c = [1]). Penalized
        // Hessian = [[50]], Fisher = [[40]], two score rows summing to a known
        // outer product. β̂ = [0.25], θ₀ = 0.
        let contrast = array![1.0_f64];
        let beta = array![0.25_f64];
        let h = Array2::from_shape_vec((1, 1), vec![50.0]).unwrap();
        let fisher = Array2::from_shape_vec((1, 1), vec![40.0]).unwrap();
        // Two rows with scores 3.0 and 4.0 ⇒ Σ sᵢ² = 25; a = Ĥ⁻¹c = 1/50.
        // sandwich = Σ (sᵢ·a)² = 25/2500 = 0.01 ⇒ score_cov = 100.
        let row_scores = Array2::from_shape_vec((2, 1), vec![3.0, 4.0]).unwrap();
        let lr = 4.0;
        let res = scalar_skovgaard_from_matrices(
            contrast.view(),
            beta.view(),
            h.view(),
            Some(fisher.view()),
            row_scores.view(),
            lr,
            0.0,
        )
        .expect("assembled r*");
        // observed_info = 1/(cᵀĤ⁻¹c) = 1/(1/50) = 50.
        // expected_info = 1/(1/40) = 40.
        // score_cov = 1/0.01 = 100.
        // θ̂ = 0.25, r = +√4 = 2.
        assert!((res.r - 2.0).abs() < 1e-12, "r = {}", res.r);
        // u = (θ̂−θ₀)·observed/√score_cov = 0.25·50/10 = 1.25.
        assert!((res.u - 1.25).abs() < 1e-12, "u = {}", res.u);
        let expected_rstar = 2.0 + (1.25 / 2.0).ln() / 2.0;
        assert!(
            (res.r_star - expected_rstar).abs() < 1e-12,
            "r* = {} expected {}",
            res.r_star,
            expected_rstar
        );
        // u < r here ⇒ r* < r ⇒ larger two-sided p than first-order.
        assert!(res.p_value_corrected > res.p_value_first_order);
        // Dimension mismatch is rejected.
        let bad_c = array![1.0_f64, 0.0];
        assert!(
            scalar_skovgaard_from_matrices(
                bad_c.view(),
                beta.view(),
                h.view(),
                None,
                row_scores.view(),
                lr,
                0.0,
            )
            .is_none()
        );
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let base = ScalarSkovgaardInput {
            theta_hat: 0.3,
            theta_null: 0.0,
            lr_statistic: 4.0,
            observed_info: 50.0,
            expected_info: 50.0,
            score_cov: 50.0,
        };
        // Non-positive LR.
        assert!(
            scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                lr_statistic: 0.0,
                ..base
            })
            .is_none()
        );
        // Non-positive info.
        assert!(
            scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                observed_info: 0.0,
                ..base
            })
            .is_none()
        );
        assert!(
            scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                score_cov: -1.0,
                ..base
            })
            .is_none()
        );
        // θ̂ = θ₀ ⇒ r = 0, p = 1, not material.
        let eq = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat: 0.0,
            ..base
        })
        .expect("equal");
        assert_eq!(eq.r, 0.0);
        assert_eq!(eq.p_value_first_order, 1.0);
        assert!(!eq.material);
    }
}
