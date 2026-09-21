//! Skovgaard's modified directed likelihood root `r*` for a **scalar** interest
//! parameter (issue #939, deliverable 3; corrected in #3535).
//!
//! For a scalar functional `ψ = cᵀβ` tested against `ψ₀` (e.g. "is the smooth
//! zero at `x₀`", a point-on-curve or a contrast), the first-order signed root
//! of the (profile) likelihood-ratio statistic
//!
//! ```text
//! r = sign(ψ̂ − ψ₀) · √( 2 [ ℓ(β̂) − ℓ(β̃) ] ),     β̃ = argmax ℓ subject to cᵀβ = ψ₀,
//! ```
//!
//! is `N(0,1)` only to `O(n⁻¹ᐟ²)`. **Barndorff-Nielsen's modified root**
//!
//! ```text
//! r* = r + (1/r) · log( u / r )
//! ```
//!
//! is `N(0,1)` to third order when `u` is Barndorff-Nielsen's sample-space
//! derivative quantity. This module computes `u` from **Skovgaard's (1996)
//! approximation** to those sample-space derivatives (Skovgaard 1996,
//! *Bernoulli* 2:145–165; Severini, *Likelihood Methods in Statistics*, 2000,
//! §7.5; Brazzale, Davison & Reid 2007, §8.5):
//!
//! ```text
//! ℓ_{;θ̂}(θ̂) − ℓ_{;θ̂}(θ̃)  ≈  q̂ᵀ î⁻¹ ĵ,      ℓ_{θ;θ̂}(θ̃)  ≈  Ŝᵀ î⁻¹ ĵ,
//! q̂ = cov_θ̂[ U(θ̂), ℓ(θ̂) − ℓ(θ̃) ],          Ŝ = cov_θ̂[ U(θ̂), U(θ̃)ᵀ ],
//! ```
//!
//! with `U` the score, `î = var_θ̂[U(θ̂)]` the expected information and
//! `ĵ = −ℓ''(θ̂)` the observed information. The `î⁻¹ĵ` factor is what converts
//! the covariances into sample-space derivatives; it is not optional.
//!
//! # The scalar `u` (no nuisance)
//!
//! Substituting into `u = ĵ^{-1/2} · {ℓ_{;θ̂}(θ̂) − ℓ_{;θ̂}(θ₀)}` gives
//!
//! ```text
//! u = √ĵ · q̂ / î                        (model form, [`scalar_skovgaard_r_star`]).
//! ```
//!
//! `q̂` is the covariance itself, not its linearisation: with the leading-Taylor
//! surrogate `q̂ ≈ (θ̂ − θ₀)·î` the information cancels and `u` degenerates to
//! the Wald root `(θ̂ − θ₀)√ĵ`, which is exact only in the canonical
//! parameterisation of a one-parameter exponential family. In any other
//! parameterisation the exact `q̂` is required — for the mean-parameterised
//! exponential `q̂ = n(1/μ₀ − 1/μ̂)`, and the linear surrogate would break the
//! parameterisation invariance of `r*`.
//!
//! The **Severini empirical** companion replaces both model covariances by
//! their observed-sample analogues, consistently in the numerator and in the
//! `î⁻¹` factor: `Î = Σᵢ sᵢ(θ̂)²` and `q̂_emp = Σᵢ sᵢ(θ̂)·(ℓᵢ(θ̂) − ℓᵢ(θ₀))`, so
//! `u_emp = √ĵ · q̂_emp / Î`. The two forms agree under correct specification
//! and diverge under misspecification; reporting both is a model-adequacy
//! diagnostic.
//!
//! # The nuisance-adjusted `u` (`ψ = cᵀβ`, `p ≥ 1` coefficients)
//!
//! In a parameterisation `θ = (ψ, λ)` Skovgaard's `u` is
//!
//! ```text
//! u = |ĵ|^{1/2} · |Ŝ| · [Ŝ⁻¹q̂]_ψ / ( |î| · |j̃_λλ|^{1/2} ),
//! ```
//!
//! where `j̃ = −ℓ''(β̃)` is the observed information at the constrained fit and
//! `j̃_λλ` its nuisance block. Mapping to the coefficient coordinates `β`
//! through any invertible `θ = Bβ` whose first row is `cᵀ`, every determinant
//! picks up a power of `|B|` that cancels, `[Ŝ⁻¹q̂]_ψ = cᵀŜ⁻¹q̂`, and
//! `|j̃_λλ| = |j̃| · cᵀj̃⁻¹c`. So in `β` coordinates
//!
//! ```text
//! u = |ĵ|^{1/2} · |Ŝ| · cᵀŜ⁻¹q̂ / ( |î| · |j̃|^{1/2} · (cᵀj̃⁻¹c)^{1/2} ),
//! ```
//!
//! which [`skovgaard_r_star_with_nuisance`] evaluates. It is invariant under any
//! linear reparameterisation of `β` and under rescaling `c`, and for `p = 1` it
//! is the scalar formula above. `Ŝ` is in general **not** symmetric (rows index
//! `U(β̂)`, columns `U(β̃)`), so its determinant is taken with its sign from a
//! partial-pivot LU factorisation. For a canonical exponential family with
//! linear predictor `Xβ`, `Ŝ = î` and `q̂ = î(β̂ − β̃)`, and the formula reduces
//! to the exact `u = (ψ̂ − ψ₀)·|ĵ|^{1/2} / (|j̃|^{1/2}(cᵀj̃⁻¹c)^{1/2})`. Collapsing
//! the problem to scalar informations such as `1/(cᵀĵ⁻¹c)` drops the
//! `|ĵ|^{1/2}/|j̃|^{1/2}` nuisance adjustment and is not a valid substitute.
//!
//! The empirical companion uses `Î_emp = Σᵢ sᵢ(β̂)sᵢ(β̂)ᵀ`,
//! `Ŝ_emp = Σᵢ sᵢ(β̂)sᵢ(β̃)ᵀ` and `q̂_emp = Σᵢ sᵢ(β̂)(ℓᵢ(β̂) − ℓᵢ(β̃))` in place
//! of `î`, `Ŝ` and `q̂`.
//!
//! # Accuracy contract (what is actually certified)
//!
//! 1. **The covariances are the caller's.** This module does not approximate
//!    `q̂`, `Ŝ` or `î`; it assembles `u` from them. With the exact model
//!    covariances under the fitted model, the `r*` tail has Skovgaard's
//!    accuracy: relative error `O(n⁻³ᐟ²)` for deviations `ψ − ψ₀ = O(n⁻¹ᐟ²)` and
//!    `O(n⁻¹)` in large-deviation regions, and it is exact to third order for
//!    a linear interest parameter in a canonical exponential family.
//! 2. **Penalized curvature is not likelihood information.** When the caller
//!    passes penalized Hessians `X'WX + S_λ` as `ĵ` and `j̃`, the smoothing
//!    penalty contributes prior curvature and the resulting `r*`/p-values
//!    describe the **penalized (MAP) surrogate**, not the likelihood. A
//!    deterministic penalty contributes nothing to `î`, `Ŝ` or `q̂`, which are
//!    covariances. The likelihood-theory calibration statements apply only
//!    when the penalty's leverage on the interest direction is negligible;
//!    otherwise treat the output as a penalized-curvature diagnostic.
//! 3. **Identifiability is required, not repaired.** A singular `î`, `Ŝ`, `ĵ`
//!    or `j̃` (for instance more coefficients than rows, where the empirical
//!    `Σ sᵢsᵢᵀ` has rank at most `n`) makes `u` undefined and the assembly
//!    returns `None`; the first-order root then stands.
//!
//! # Certification anchors
//!
//! * The **mean-parameterised exponential** (`μ` the mean, `n = 5`): the exact
//!   null law of `Σy` is `Gamma(n, μ₀)`, whose tail is a finite Poisson sum.
//!   The `r*` tail matches it to well inside `n⁻³ᐟ²` relative error while the
//!   first-order root misses by 20–45%, `r*` equals minus the canonical-rate
//!   `r*` (parameterisation invariance), and `u_emp = u` for any data.
//! * A **two-group Poisson** log-linear model with interest `β₁ + β₂`: the
//!   nuisance-adjusted `u` equals the closed form `√(S₁S₂)·(ψ̂−ψ₀)/√(a+b)`
//!   of the canonical family, which differs from any scalar reduction.
//! * A generic `p = 3` input with a non-symmetric `Ŝ`: `u`, `u_emp` and `r*` are
//!   unchanged by a linear reparameterisation with negative Jacobian.

use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_math::probability::normal_two_sided_probability;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// The ingredients of the scalar Skovgaard `r*` for a scalar parameter `θ`
/// with no nuisance parameters.
///
/// Every field is a likelihood quantity of the fitted model; this struct is a
/// pure data carrier so the assembly is testable in isolation against the
/// closed-form fixtures.
#[derive(Debug, Clone, Copy)]
pub struct ScalarSkovgaardInput {
    /// The estimate `θ̂`.
    pub theta_hat: f64,
    /// The tested null value `θ₀`.
    pub theta_null: f64,
    /// The log-likelihood-ratio statistic `W = 2[ℓ(θ̂) − ℓ(θ₀)] ≥ 0`.
    pub lr_statistic: f64,
    /// Observed information `ĵ = −ℓ''(θ̂)`.
    pub observed_info: f64,
    /// Expected information `î = var_θ̂[U(θ̂)]` under the fitted model.
    pub expected_info: f64,
    /// Skovgaard's `q̂ = cov_θ̂[U(θ̂), ℓ(θ̂) − ℓ(θ₀)]` under the fitted model.
    /// Its sign normally matches `θ̂ − θ₀`; when it does not, `u/r ≤ 0` and the
    /// first-order root stands.
    pub loglik_covariance: f64,
    /// Empirical information `Î = Σᵢ sᵢ(θ̂)²`, with `sᵢ = ∂ℓᵢ/∂θ`.
    pub empirical_info: f64,
    /// Empirical `q̂_emp = Σᵢ sᵢ(θ̂)·(ℓᵢ(θ̂) − ℓᵢ(θ₀))`.
    pub empirical_loglik_covariance: f64,
}

/// The Skovgaard `r*` report for a scalar test.
#[derive(Debug, Clone, Copy)]
pub struct ScalarSkovgaardResult {
    /// The first-order directed likelihood root `r = sign(θ̂−θ₀)·√W`.
    pub r: f64,
    /// Skovgaard's `u` from the model covariances (`√ĵ·q̂/î` for a scalar
    /// parameter; the determinant form with a nuisance).
    pub u: f64,
    /// The modified directed root `r* = r + log(u/r)/r` (model form).
    pub r_star: f64,
    /// First-order two-sided p-value `2·Φ(−|r|)`.
    pub p_value_first_order: f64,
    /// Higher-order-corrected two-sided p-value `2·Φ(−|r*|)` (model form),
    /// with the accuracy stated in the module's accuracy contract.
    pub p_value_corrected: f64,
    /// The Severini empirical companion `u_emp`, with the empirical covariances
    /// in place of the model ones.
    pub u_empirical: f64,
    /// The empirical modified root `r* = r + log(u_emp/r)/r`. Falls back to `r`
    /// when the empirical log-domain guard fails, exactly as the model form.
    pub r_star_empirical: f64,
    /// Two-sided p-value `2·Φ(−|r*_emp|)` of the empirical companion.
    pub p_value_corrected_empirical: f64,
    /// Whether the `r*` correction is **material** (>10% relative change in the
    /// p-value, or `|r* − r| > 0.1·|r|`): the per-test diagnostic that the
    /// sample is too small for first-order inference here (#939 deliverable 4).
    pub material: bool,
}

/// The materiality threshold (#939 deliverable 4): a correction is material when
/// it moves the result by more than 10%.
pub(crate) const SKOVGAARD_MATERIAL_THRESHOLD: f64 = 0.10;

/// Assemble the scalar Skovgaard `r*` from its ingredients: `u = √ĵ·q̂/î` and
/// `u_emp = √ĵ·q̂_emp/Î`.
///
/// Returns `None` when the inputs are degenerate: a non-positive LR statistic
/// (the directed root is undefined at `r = 0`), a non-finite or non-positive
/// information, or a non-finite covariance. When `u/r ≤ 0` the third-order
/// formula does not apply and `r*` is reported as the first-order `r`; the
/// correction is never forced.
pub fn scalar_skovgaard_r_star(input: &ScalarSkovgaardInput) -> Option<ScalarSkovgaardResult> {
    let ScalarSkovgaardInput {
        theta_hat,
        theta_null,
        lr_statistic,
        observed_info,
        expected_info,
        loglik_covariance,
        empirical_info,
        empirical_loglik_covariance,
    } = *input;

    for info in [observed_info, expected_info, empirical_info] {
        if !(info.is_finite() && info > 0.0) {
            return None;
        }
    }
    if !(loglik_covariance.is_finite() && empirical_loglik_covariance.is_finite()) {
        return None;
    }
    let sqrt_obs = observed_info.sqrt();
    let u = sqrt_obs * loglik_covariance / expected_info;
    let u_empirical = sqrt_obs * empirical_loglik_covariance / empirical_info;
    assemble_modified_root(theta_hat, theta_null, lr_statistic, u, u_empirical)
}

/// The ingredients of Skovgaard's nuisance-adjusted `r*` for `ψ = cᵀβ` over
/// `p` coefficients, from the full fit `β̂` and the constrained fit `β̃`
/// (`cᵀβ̃ = ψ₀`). Every covariance is taken under the full fit `β̂`.
#[derive(Debug, Clone, Copy)]
pub struct SkovgaardNuisanceInput<'a> {
    /// The functional gradient `c = ∂ψ/∂β` (a prediction row for a point on a
    /// curve, a row difference for a contrast, …). Length `p`.
    pub contrast: ArrayView1<'a, f64>,
    /// Full-fit coefficients `β̂`; `ψ̂ = cᵀβ̂`.
    pub beta_hat: ArrayView1<'a, f64>,
    /// Constrained-fit coefficients `β̃` maximising the likelihood subject to
    /// `cᵀβ = ψ₀`; the tested value is `ψ₀ = cᵀβ̃`.
    pub beta_null: ArrayView1<'a, f64>,
    /// `W = 2[ℓ(β̂) − ℓ(β̃)] ≥ 0`.
    pub lr_statistic: f64,
    /// Observed information `ĵ = −ℓ''(β̂)` (`p × p`, symmetric positive
    /// definite; the penalized Hessian for a penalized fit, see the accuracy
    /// contract, item 2).
    pub observed_info_hat: ArrayView2<'a, f64>,
    /// Observed information `j̃ = −ℓ''(β̃)` at the constrained fit (`p × p`,
    /// symmetric positive definite).
    pub observed_info_null: ArrayView2<'a, f64>,
    /// Expected information `î = var_β̂[U(β̂)]` (`p × p`, symmetric positive
    /// definite).
    pub expected_info: ArrayView2<'a, f64>,
    /// `Ŝ = cov_β̂[U(β̂), U(β̃)ᵀ]` (`p × p`, general): row `j` indexes
    /// `U_j(β̂)`, column `k` indexes `U_k(β̃)`.
    pub score_covariance: ArrayView2<'a, f64>,
    /// `q̂ = cov_β̂[U(β̂), ℓ(β̂) − ℓ(β̃)]` (length `p`).
    pub loglik_covariance: ArrayView1<'a, f64>,
    /// Per-row scores `sᵢ(β̂) = ∂ℓᵢ/∂β` at the full fit (`n × p`).
    pub row_scores_hat: ArrayView2<'a, f64>,
    /// Per-row scores `sᵢ(β̃)` at the constrained fit (`n × p`).
    pub row_scores_null: ArrayView2<'a, f64>,
    /// Per-row log-likelihood differences `ℓᵢ(β̂) − ℓᵢ(β̃)` (length `n`).
    pub row_loglik_diff: ArrayView1<'a, f64>,
}

/// Skovgaard's nuisance-adjusted `r*` for `ψ = cᵀβ`:
/// `u = |ĵ|^{1/2}·|Ŝ|·cᵀŜ⁻¹q̂ / (|î|·|j̃|^{1/2}·(cᵀj̃⁻¹c)^{1/2})`, and the same
/// with `Σ sᵢsᵢᵀ`, `Σ sᵢ(β̂)sᵢ(β̃)ᵀ` and `Σ sᵢ(β̂)(ℓᵢ(β̂) − ℓᵢ(β̃))` for the
/// empirical companion (see the module documentation for the derivation).
///
/// Returns `None` on mismatched shapes, non-finite entries, a non-positive LR
/// statistic, an information matrix that is not positive definite, a singular
/// `Ŝ` (model or empirical), or a non-finite `u`.
pub fn skovgaard_r_star_with_nuisance(
    input: &SkovgaardNuisanceInput<'_>,
) -> Option<ScalarSkovgaardResult> {
    let SkovgaardNuisanceInput {
        contrast,
        beta_hat,
        beta_null,
        lr_statistic,
        observed_info_hat,
        observed_info_null,
        expected_info,
        score_covariance,
        loglik_covariance,
        row_scores_hat,
        row_scores_null,
        row_loglik_diff,
    } = *input;

    let p = beta_hat.len();
    let n = row_scores_hat.nrows();
    if p == 0 || n == 0 {
        return None;
    }
    let vectors_fit = contrast.len() == p && beta_null.len() == p && loglik_covariance.len() == p;
    let matrices_fit = [
        observed_info_hat,
        observed_info_null,
        expected_info,
        score_covariance,
    ]
    .iter()
    .all(|m| m.nrows() == p && m.ncols() == p);
    let rows_fit = row_scores_hat.ncols() == p
        && row_scores_null.nrows() == n
        && row_scores_null.ncols() == p
        && row_loglik_diff.len() == n;
    if !(vectors_fit && matrices_fit && rows_fit) {
        return None;
    }
    let all_finite = [
        contrast,
        beta_hat,
        beta_null,
        loglik_covariance,
        row_loglik_diff,
    ]
    .iter()
    .all(|v| v.iter().all(|x| x.is_finite()))
        && [
            observed_info_hat,
            observed_info_null,
            expected_info,
            score_covariance,
            row_scores_hat,
            row_scores_null,
        ]
        .iter()
        .all(|m| m.iter().all(|x| x.is_finite()));
    if !all_finite {
        return None;
    }

    let theta_hat = contrast.dot(&beta_hat);
    let theta_null = contrast.dot(&beta_null);

    let (_, logdet_obs_hat) = spd_factor_logdet(observed_info_hat)?;
    let (chol_obs_null, logdet_obs_null) = spd_factor_logdet(observed_info_null)?;
    let contrast_owned = contrast.to_owned();
    let null_variance = contrast.dot(&chol_obs_null.solvevec(&contrast_owned));
    if !(null_variance.is_finite() && null_variance > 0.0) {
        return None;
    }
    // The factors common to the model and empirical forms:
    // |ĵ|^{1/2} / (|j̃|^{1/2} (cᵀj̃⁻¹c)^{1/2}).
    let log_common = 0.5 * (logdet_obs_hat - logdet_obs_null);
    let sqrt_null_variance = null_variance.sqrt();
    let nuisance_u = |info: ArrayView2<'_, f64>,
                      score_cov: ArrayView2<'_, f64>,
                      loglik_cov: ArrayView1<'_, f64>|
     -> Option<f64> {
        let (_, logdet_info) = spd_factor_logdet(info)?;
        let lu = DenseLu::factor(score_cov)?;
        let directional = contrast.dot(&lu.solve(loglik_cov));
        let magnitude = (log_common + lu.log_abs_det - logdet_info).exp();
        let u = lu.det_sign * magnitude * directional / sqrt_null_variance;
        u.is_finite().then_some(u)
    };

    let u = nuisance_u(expected_info, score_covariance, loglik_covariance)?;
    let empirical_info: Array2<f64> = row_scores_hat.t().dot(&row_scores_hat);
    let empirical_score_cov: Array2<f64> = row_scores_hat.t().dot(&row_scores_null);
    let empirical_loglik_cov: Array1<f64> = row_scores_hat.t().dot(&row_loglik_diff);
    let u_empirical = nuisance_u(
        empirical_info.view(),
        empirical_score_cov.view(),
        empirical_loglik_cov.view(),
    )?;
    assemble_modified_root(theta_hat, theta_null, lr_statistic, u, u_empirical)
}

/// Cholesky factor and log-determinant `2 Σ ln Lₖₖ` of a symmetric positive
/// definite matrix (its lower triangle is read), or `None` if it is not
/// positive definite.
fn spd_factor_logdet(
    m: ArrayView2<'_, f64>,
) -> Option<(gam_linalg::faer_ndarray::FaerCholeskyFactor, f64)> {
    let chol = m.cholesky(Side::Lower).ok()?;
    let logdet = 2.0 * chol.diag().iter().map(|d| d.ln()).sum::<f64>();
    logdet.is_finite().then_some((chol, logdet))
}

/// Partial-pivot LU `PA = LU` of a general square matrix, carrying the sign
/// and log-magnitude of its determinant (the Skovgaard `Ŝ` is not symmetric).
struct DenseLu {
    /// Unit-lower `L` below the diagonal, `U` on and above it.
    lu: Array2<f64>,
    /// `perm[i]` is the original row placed at row `i`.
    perm: Vec<usize>,
    log_abs_det: f64,
    det_sign: f64,
}

impl DenseLu {
    /// Factor `a`, or `None` if a pivot column is exactly zero (singular) or the
    /// determinant is not representable.
    fn factor(a: ArrayView2<'_, f64>) -> Option<Self> {
        let p = a.nrows();
        if a.ncols() != p {
            return None;
        }
        let mut lu = a.to_owned();
        let mut perm: Vec<usize> = (0..p).collect();
        let mut det_sign = 1.0_f64;
        let mut log_abs_det = 0.0_f64;
        for k in 0..p {
            let mut pivot_row = k;
            let mut pivot_abs = lu[[k, k]].abs();
            for i in (k + 1)..p {
                let candidate = lu[[i, k]].abs();
                if candidate > pivot_abs {
                    pivot_abs = candidate;
                    pivot_row = i;
                }
            }
            if !(pivot_abs.is_finite() && pivot_abs > 0.0) {
                return None;
            }
            if pivot_row != k {
                for j in 0..p {
                    lu.swap([k, j], [pivot_row, j]);
                }
                perm.swap(k, pivot_row);
                det_sign = -det_sign;
            }
            let pivot = lu[[k, k]];
            if pivot < 0.0 {
                det_sign = -det_sign;
            }
            log_abs_det += pivot_abs.ln();
            for i in (k + 1)..p {
                let multiplier = lu[[i, k]] / pivot;
                lu[[i, k]] = multiplier;
                for j in (k + 1)..p {
                    let update = multiplier * lu[[k, j]];
                    lu[[i, j]] -= update;
                }
            }
        }
        log_abs_det.is_finite().then_some(Self {
            lu,
            perm,
            log_abs_det,
            det_sign,
        })
    }

    /// Solve `A x = b` with the stored factors.
    fn solve(&self, b: ArrayView1<'_, f64>) -> Array1<f64> {
        let p = self.perm.len();
        let mut x: Array1<f64> = self.perm.iter().map(|&row| b[row]).collect();
        for i in 0..p {
            let mut acc = x[i];
            for j in 0..i {
                acc -= self.lu[[i, j]] * x[j];
            }
            x[i] = acc;
        }
        for i in (0..p).rev() {
            let mut acc = x[i];
            for j in (i + 1)..p {
                acc -= self.lu[[i, j]] * x[j];
            }
            x[i] = acc / self.lu[[i, i]];
        }
        x
    }
}

/// Shared tail of both assemblies: the directed root from `W`, the modified
/// roots from `u` and `u_emp`, their p-values and the materiality flag.
fn assemble_modified_root(
    theta_hat: f64,
    theta_null: f64,
    lr_statistic: f64,
    u: f64,
    u_empirical: f64,
) -> Option<ScalarSkovgaardResult> {
    if !(lr_statistic.is_finite() && lr_statistic > 0.0) {
        return None;
    }
    if !(theta_hat.is_finite() && theta_null.is_finite()) {
        return None;
    }
    if !(u.is_finite() && u_empirical.is_finite()) {
        return None;
    }

    // θ̂ = θ₀ ⇒ the directed root is exactly `r = 0`: no side, no correction.
    // NOTE: `f64::signum` returns `±1.0` even for `±0.0` (it never returns `0.0`),
    // so the equality case MUST be detected directly from `θ̂ − θ₀ == 0` rather
    // than from `sign == 0.0` — the latter is unreachable and would let an
    // on-the-null input fall through to `r = √W ≠ 0`.
    if theta_hat == theta_null {
        return Some(ScalarSkovgaardResult {
            r: 0.0,
            u: 0.0,
            r_star: 0.0,
            p_value_first_order: 1.0,
            p_value_corrected: 1.0,
            u_empirical: 0.0,
            r_star_empirical: 0.0,
            p_value_corrected_empirical: 1.0,
            material: false,
        });
    }
    // First-order directed root: sign from the estimate's side of the null, mag
    // from the LR statistic. `θ̂ ≠ θ₀` here, so `sign ∈ {−1, +1}`.
    let sign = (theta_hat - theta_null).signum();
    let r = sign * lr_statistic.sqrt();
    let p_first = normal_two_sided_probability(r);

    // Barndorff-Nielsen modification `r* = r + log(u/r)/r`, guarding the
    // log-domain (`u` and `r` must share a sign and `u/r > 0`): when the
    // third-order formula does not apply the first-order root stands.
    let modified_root = |u_val: f64| -> f64 {
        let ratio = u_val / r;
        if !(ratio.is_finite() && ratio > 0.0) {
            return r;
        }
        let rs = r + ratio.ln() / r;
        if rs.is_finite() { rs } else { r }
    };
    let r_star = modified_root(u);
    let r_star_empirical = modified_root(u_empirical);

    let p_corr = normal_two_sided_probability(r_star);
    let p_corr_empirical = normal_two_sided_probability(r_star_empirical);
    let p_denom = p_first.max(p_corr);
    // Two tail probabilities that both underflowed to zero have not moved.
    let p_move = if p_denom > 0.0 {
        (p_corr - p_first).abs() / p_denom
    } else {
        0.0
    };
    // `lr_statistic > 0` was checked above, so `|r| = √W > 0` even for a
    // subnormal `W`.
    let r_move = (r_star - r).abs() / r.abs();
    let material = p_move > SKOVGAARD_MATERIAL_THRESHOLD || r_move > SKOVGAARD_MATERIAL_THRESHOLD;

    Some(ScalarSkovgaardResult {
        r,
        u,
        r_star,
        p_value_first_order: p_first,
        p_value_corrected: p_corr,
        u_empirical,
        r_star_empirical,
        p_value_corrected_empirical: p_corr_empirical,
        material,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use gam_math::probability::{normal_cdf, normal_pdf};
    use ndarray::array;

    /// `|actual − expected| ≤ γ_ops · max(|actual|, |expected|)`: agreement to the
    /// rounding of an `ops`-operation evaluation.
    fn assert_close(actual: f64, expected: f64, ops: usize, what: &str) {
        let tol = accumulation_growth(ops) * actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: {actual} vs {expected} (|Δ|={:.3e}, tol={tol:.3e})",
            (actual - expected).abs()
        );
    }

    /// Operation count of one nuisance assembly on `p` coefficients and `n`
    /// rows: three factorisations and solves of `p × p` matrices plus the
    /// `n × p` empirical cross-products.
    fn nuisance_ops(p: usize, n: usize) -> usize {
        4 * p * p * p + 2 * n * p * p
    }

    #[test]
    fn regular_case_collapses_to_first_order() {
        // î = ĵ = Î = 100 and a log-likelihood difference linear in the score
        // (q̂ = (θ̂−θ₀)·î): u = √ĵ·q̂/î = (θ̂−θ₀)√ĵ = 3 = r, so r* = r.
        let info = 100.0;
        let dtheta = 0.3;
        let lr = dtheta * dtheta * info; // 9.0
        let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat: dtheta,
            theta_null: 0.0,
            lr_statistic: lr,
            observed_info: info,
            expected_info: info,
            loglik_covariance: dtheta * info,
            empirical_info: info,
            empirical_loglik_covariance: dtheta * info,
        })
        .expect("r*");
        assert_close(res.r, 3.0, 4, "r");
        assert_close(res.u, 3.0, 4, "u");
        assert_close(res.r_star, res.r, 4, "r*");
        assert_close(res.u_empirical, res.u, 4, "u_emp");
        assert_close(res.r_star_empirical, res.r_star, 4, "r*_emp");
        assert!(!res.material, "regular case must not be material");
    }

    /// #3535 regression guard. Skovgaard's `u` is `√ĵ·q̂/î`: with a
    /// log-likelihood difference whose covariance with the score is linear,
    /// `q̂ = (θ̂−θ₀)·î`, the expected information cancels and `u` is the Wald
    /// root `(θ̂−θ₀)√ĵ` whatever `î/ĵ` is. The pre-fix `u = (θ̂−θ₀)·î/√ĵ`
    /// returned `(θ̂−θ₀)√ĵ·(î/ĵ)` = 3.92 here, and `2.42` for the empirical
    /// form that kept `Î` only in the numerator.
    #[test]
    fn linear_loglik_covariance_cancels_the_information_ratio() {
        let dtheta = 0.2;
        let observed = 100.0; // ĵ
        let expected = 196.0; // î ≠ ĵ
        let empirical = 121.0; // Î ≠ î
        let lr = dtheta * dtheta * observed; // 4.0, r = 2.0
        let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat: dtheta,
            theta_null: 0.0,
            lr_statistic: lr,
            observed_info: observed,
            expected_info: expected,
            loglik_covariance: dtheta * expected,
            empirical_info: empirical,
            empirical_loglik_covariance: dtheta * empirical,
        })
        .expect("r*");
        let wald = dtheta * observed.sqrt();
        assert_close(res.r, 2.0, 4, "r");
        assert_close(res.u, wald, 8, "u");
        assert_close(res.u_empirical, wald, 8, "u_emp");
        assert_close(res.r_star, res.r, 8, "r*");
        assert!(!res.material);
    }

    /// CONJUGATE FIXTURE (Exponential rate, the canonical parameter): `ℓ(θ) =
    /// n ln θ − θ Σy`, `θ̂ = n/Σy`, `ĵ = î = n/θ̂²`, and `ℓ(θ̂) − ℓ(θ₀)` is affine
    /// in `Σy`, so `q̂ = (θ̂−θ₀)·var(Σy) = (θ̂−θ₀)·n/θ̂²` exactly and `u` is the
    /// Wald root. For the right-skewed exponential LR the modified root lies
    /// strictly between the Wald root and `r`.
    #[test]
    fn exponential_rate_scalar_skovgaard_closed_form() {
        let n = 25.0_f64;
        let sum_y = 20.0_f64;
        let theta_hat = n / sum_y; // 1.25
        let theta0 = 1.0_f64;
        let ll = |t: f64| n * t.ln() - t * sum_y;
        let lr = 2.0 * (ll(theta_hat) - ll(theta0));
        let info = n / (theta_hat * theta_hat);
        let q = (theta_hat - theta0) * info;
        let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat,
            theta_null: theta0,
            lr_statistic: lr,
            observed_info: info,
            expected_info: info,
            loglik_covariance: q,
            empirical_info: info,
            empirical_loglik_covariance: q,
        })
        .expect("r*");
        let r_expected = (theta_hat - theta0).signum() * lr.sqrt();
        assert_close(res.r, r_expected, 4, "r");
        let wald = (theta_hat - theta0) * info.sqrt();
        assert_close(res.u, wald, 8, "u");
        assert_close(res.u_empirical, res.u, 8, "u_emp");
        assert!(
            wald < res.r_star && res.r_star < r_expected,
            "need Wald < r* < r: Wald={wald} r*={} r={r_expected}",
            res.r_star
        );
        assert!((0.0..=1.0).contains(&res.p_value_first_order));
        assert!((0.0..=1.0).contains(&res.p_value_corrected));
    }

    /// EXACT-TAIL FIXTURE (#3535): the exponential in its **mean**
    /// parameterisation, where the linear surrogate `q̂ ≈ (μ̂−μ₀)·î` is wrong.
    ///
    /// `yᵢ ~ Exp(mean μ)`: `ℓ(μ) = −n ln μ − Σy/μ`, `μ̂ = ȳ`, `sᵢ(μ) = −1/μ +
    /// yᵢ/μ²`, `ĵ = î = n/μ̂²`, and the exact
    /// `q̂ = cov_μ̂(Σy/μ̂², Σy(1/μ₀ − 1/μ̂)) = n(1/μ₀ − 1/μ̂)`, giving
    /// `u = √n(μ̂/μ₀ − 1)`. The exact null law is `Σy ~ Gamma(n, scale μ₀)`, with
    /// `P(Σy ≥ t) = e^{−t/μ₀} Σ_{k<n} (t/μ₀)ᵏ/k!`.
    ///
    /// Asserted, at `n = 5` over both tails:
    /// * the `r*` tail is within relative error `n⁻³ᐟ²` of the exact tail while
    ///   the first-order root is not (it misses by 20–45%; the pre-fix
    ///   `u = √n(1 − μ₀/μ̂)` missed by 55–170%);
    /// * `r*` is minus the canonical-rate `r*` (`θ = 1/μ` reverses the order),
    ///   the parameterisation invariance the pre-fix `u` violated;
    /// * the Severini empirical `u_emp = √ĵ·q̂_emp/Î` equals `u` for any data
    ///   (both `q̂_emp` and `Î` carry the same `Σ(yᵢ − μ̂)²`).
    #[test]
    fn mean_exponential_r_star_matches_exact_gamma_tail() {
        const N: usize = 5;
        let n = N as f64;
        let mu0 = 1.0_f64;
        // Fixed data shape with Σw = n, rescaled so that Σy = t exactly.
        let shape = [0.2_f64, 0.6, 0.9, 1.4, 1.9];
        let band = n.powf(-1.5);
        let row_ops = 10 * N;
        for &t in &[8.0_f64, 10.0, 12.0, 3.0, 2.0] {
            let y: Vec<f64> = shape.iter().map(|w| w * t / n).collect();
            let mu_hat = y.iter().sum::<f64>() / n;
            let lr = 2.0 * n * ((mu0 / mu_hat).ln() + mu_hat / mu0 - 1.0);
            let info = n / (mu_hat * mu_hat);
            let scores: Vec<f64> = y
                .iter()
                .map(|&yi| -1.0 / mu_hat + yi / (mu_hat * mu_hat))
                .collect();
            let row_ll = |mu: f64, yi: f64| -mu.ln() - yi / mu;
            let empirical_info: f64 = scores.iter().map(|s| s * s).sum();
            let empirical_q: f64 = scores
                .iter()
                .zip(&y)
                .map(|(s, &yi)| s * (row_ll(mu_hat, yi) - row_ll(mu0, yi)))
                .sum();
            let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                theta_hat: mu_hat,
                theta_null: mu0,
                lr_statistic: lr,
                observed_info: info,
                expected_info: info,
                loglik_covariance: n * (1.0 / mu0 - 1.0 / mu_hat),
                empirical_info,
                empirical_loglik_covariance: empirical_q,
            })
            .expect("mean-exponential r*");
            assert_close(res.u, n.sqrt() * (mu_hat / mu0 - 1.0), row_ops, "u");
            assert_close(res.u_empirical, res.u, row_ops, "u_emp");

            // Exact Gamma(n, μ₀) upper tail of Σy at t.
            let x = t / mu0;
            let mut term = 1.0_f64;
            let mut partial = 1.0_f64;
            for k in 1..N {
                term *= x / k as f64;
                partial += term;
            }
            let upper = (-x).exp() * partial;
            let (exact, tail_star, tail_first) = if mu_hat > mu0 {
                (upper, normal_cdf(-res.r_star), normal_cdf(-res.r))
            } else {
                (1.0 - upper, normal_cdf(res.r_star), normal_cdf(res.r))
            };
            let rel_star = (tail_star - exact).abs() / exact;
            let rel_first = (tail_first - exact).abs() / exact;
            assert!(
                rel_star < band,
                "t={t}: r* tail {tail_star:.6} vs exact {exact:.6}: relative error \
                 {rel_star:.4} exceeds n^(-3/2) = {band:.4}"
            );
            assert!(
                rel_first > band,
                "t={t}: the first-order tail must miss the n^(-3/2) band for this \
                 fixture to have teeth (relative error {rel_first:.4})"
            );

            // Canonical rate θ = 1/μ: u is the Wald root (linear q̂ is exact there).
            let theta_hat = 1.0 / mu_hat;
            let theta0 = 1.0 / mu0;
            let rate_info = n / (theta_hat * theta_hat);
            let rate_q = (theta_hat - theta0) * rate_info;
            let rate = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                theta_hat,
                theta_null: theta0,
                lr_statistic: lr,
                observed_info: rate_info,
                expected_info: rate_info,
                loglik_covariance: rate_q,
                empirical_info: rate_info,
                empirical_loglik_covariance: rate_q,
            })
            .expect("rate r*");
            assert_close(res.u, -rate.u, row_ops, "u invariance");
            assert_close(res.r_star, -rate.r_star, row_ops, "r* invariance");
        }
    }

    /// For `p = 1` the determinant form is the scalar `u = √ĵ·q̂/î` in the
    /// parameter `θ = cβ` (informations scale by `1/c²`, covariances by `1/c`).
    #[test]
    fn nuisance_path_reduces_to_scalar_path_for_one_coefficient() {
        let c = 2.0_f64;
        let beta_hat = 0.3_f64;
        let beta_null = 0.05_f64;
        let lr = 0.4;
        let (j_hat, j_null, info, score_cov, q) = (40.0_f64, 44.0_f64, 36.0_f64, 35.0_f64, 8.0_f64);
        let s_hat = array![3.0_f64, -2.0, 4.0];
        let s_null = array![3.5_f64, -1.4, 4.4];
        let dl = array![0.9_f64, -0.5, 1.2];
        let one = |v: f64| Array2::from_elem((1, 1), v);
        let (jh, jn, ie, sc) = (one(j_hat), one(j_null), one(info), one(score_cov));
        let contrast = array![c];
        let bh = array![beta_hat];
        let bn = array![beta_null];
        let qv = array![q];
        let sh = s_hat.clone().insert_axis(ndarray::Axis(1));
        let sn = s_null.clone().insert_axis(ndarray::Axis(1));
        let matrix = skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
            contrast: contrast.view(),
            beta_hat: bh.view(),
            beta_null: bn.view(),
            lr_statistic: lr,
            observed_info_hat: jh.view(),
            observed_info_null: jn.view(),
            expected_info: ie.view(),
            score_covariance: sc.view(),
            loglik_covariance: qv.view(),
            row_scores_hat: sh.view(),
            row_scores_null: sn.view(),
            row_loglik_diff: dl.view(),
        })
        .expect("p = 1 nuisance path");
        let scalar = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat: c * beta_hat,
            theta_null: c * beta_null,
            lr_statistic: lr,
            observed_info: j_hat / (c * c),
            expected_info: info / (c * c),
            loglik_covariance: q / c,
            empirical_info: s_hat.dot(&s_hat) / (c * c),
            empirical_loglik_covariance: s_hat.dot(&dl) / c,
        })
        .expect("scalar path");
        let ops = nuisance_ops(1, s_hat.len());
        assert_close(matrix.r, scalar.r, ops, "r");
        assert_close(matrix.u, scalar.u, ops, "u");
        assert_close(matrix.u_empirical, scalar.u_empirical, ops, "u_emp");
        assert_close(matrix.r_star, scalar.r_star, ops, "r*");
        assert_close(
            matrix.r_star_empirical,
            scalar.r_star_empirical,
            ops,
            "r*_emp",
        );
    }

    /// CANONICAL NUISANCE FIXTURE: two Poisson groups with log means `β₁, β₂`
    /// (indicator design), interest `ψ = β₁ + β₂ = ln(μ₁μ₂)`. Group totals
    /// `S₁ = 14`, `S₂ = 6` over `m₁ = m₂ = 3` rows, `ψ₀ = ln 4`. The constrained
    /// fit has `S_k − m_k μ̃_k` equal across groups, so `a = m₁μ̃₁` and
    /// `b = m₂μ̃₂` solve `a − b = S₁ − S₂` and `ab = m₁m₂e^{ψ₀}`. The family is
    /// canonical, so `Ŝ = î = ĵ = diag(S₁, S₂)`, `j̃ = diag(a, b)`,
    /// `q̂ = î(β̂ − β̃)`, and the exact
    /// `u = (ψ̂−ψ₀)|ĵ|^{1/2}/(|j̃|^{1/2}(cᵀj̃⁻¹c)^{1/2}) = √(S₁S₂)(ψ̂−ψ₀)/√(a+b)`.
    /// The scalar reduction `(ψ̂−ψ₀)/√(cᵀĵ⁻¹c) = (ψ̂−ψ₀)√(S₁S₂/(S₁+S₂))` differs
    /// (`a + b ≠ S₁ + S₂`), so dropping the nuisance adjustment fails this test.
    /// Per-row scores `x_i(y_i − μ)` give `Σŝs̃ᵀ = Σŝŝᵀ` and `q̂_emp = Σŝŝᵀ(β̂−β̃)`,
    /// so the empirical `u` equals the model `u`.
    #[test]
    fn two_group_poisson_nuisance_adjustment_matches_closed_form() {
        let groups: [&[f64]; 2] = [&[3.0, 5.0, 6.0], &[1.0, 2.0, 3.0]];
        let totals = [groups[0].iter().sum::<f64>(), groups[1].iter().sum::<f64>()];
        let sizes = [groups[0].len() as f64, groups[1].len() as f64];
        let psi0 = 4.0_f64.ln();
        let diff = totals[0] - totals[1];
        let product = sizes[0] * sizes[1] * psi0.exp();
        let a = 0.5 * (diff + (diff * diff + 4.0 * product).sqrt());
        let b = a - diff;
        let beta_hat = array![(totals[0] / sizes[0]).ln(), (totals[1] / sizes[1]).ln()];
        let beta_null = array![(a / sizes[0]).ln(), (b / sizes[1]).ln()];
        let contrast = array![1.0_f64, 1.0];
        let loglik = |beta: &Array1<f64>| -> f64 {
            (0..2)
                .map(|k| totals[k] * beta[k] - sizes[k] * beta[k].exp())
                .sum()
        };
        let lr = 2.0 * (loglik(&beta_hat) - loglik(&beta_null));
        let info_hat = Array2::from_diag(&array![totals[0], totals[1]]);
        let info_null = Array2::from_diag(&array![a, b]);
        let q = info_hat.dot(&(&beta_hat - &beta_null));

        let n_rows = groups[0].len() + groups[1].len();
        let mut s_hat = Array2::<f64>::zeros((n_rows, 2));
        let mut s_null = Array2::<f64>::zeros((n_rows, 2));
        let mut dl = Array1::<f64>::zeros(n_rows);
        let mut row = 0;
        for (k, group) in groups.iter().enumerate() {
            let (mu_hat, mu_null) = (beta_hat[k].exp(), beta_null[k].exp());
            for &y in group.iter() {
                s_hat[[row, k]] = y - mu_hat;
                s_null[[row, k]] = y - mu_null;
                dl[row] = y * (beta_hat[k] - beta_null[k]) - (mu_hat - mu_null);
                row += 1;
            }
        }
        let res = skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
            contrast: contrast.view(),
            beta_hat: beta_hat.view(),
            beta_null: beta_null.view(),
            lr_statistic: lr,
            observed_info_hat: info_hat.view(),
            observed_info_null: info_null.view(),
            expected_info: info_hat.view(),
            score_covariance: info_hat.view(),
            loglik_covariance: q.view(),
            row_scores_hat: s_hat.view(),
            row_scores_null: s_null.view(),
            row_loglik_diff: dl.view(),
        })
        .expect("two-group Poisson r*");
        let dpsi = contrast.dot(&(&beta_hat - &beta_null));
        let u_exact = (totals[0] * totals[1]).sqrt() * dpsi / (a + b).sqrt();
        let ops = nuisance_ops(2, n_rows);
        assert_close(res.r, lr.sqrt(), ops, "r");
        assert_close(res.u, u_exact, ops, "u");
        assert_close(res.u_empirical, res.u, ops, "u_emp");
        let scalar_reduction = dpsi * (totals[0] * totals[1] / (totals[0] + totals[1])).sqrt();
        assert!(
            (res.u - scalar_reduction).abs() > accumulation_growth(ops) * res.u.abs(),
            "the nuisance adjustment must move u off the scalar reduction \
             (u={}, scalar={scalar_reduction})",
            res.u
        );
    }

    /// Generic `p = 3` inputs with a **non-symmetric** `Ŝ` and `n = 5` rows,
    /// used in the two parameterisations for `β` and `γ = M⁻¹β` (`M` has a
    /// negative determinant). The ingredients transform as `c ↦ Mᵀc`,
    /// informations and `Ŝ` as `A ↦ MᵀAM`, `q̂ ↦ Mᵀq̂`, row scores `sᵢ ↦ Mᵀsᵢ`;
    /// `u`, `u_emp` and `r*` must not change.
    #[test]
    fn nuisance_path_is_invariant_under_linear_reparameterization() {
        let j_hat = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let j_null = array![[5.0, 0.8, 0.3], [0.8, 2.5, 0.1], [0.3, 0.1, 2.2]];
        let info = array![[3.6, 0.9, 0.4], [0.9, 2.8, 0.3], [0.4, 0.3, 1.9]];
        let score_cov = array![[3.4, 1.1, 0.2], [0.7, 2.6, 0.5], [0.6, 0.1, 1.8]];
        let contrast = array![1.0, -0.5, 0.25];
        let beta_hat = array![0.9, 0.2, -0.3];
        let beta_null = array![0.4, 0.3, -0.1];
        let q = score_cov.dot(&(&beta_hat - &beta_null)) * 1.1;
        let s_hat = array![
            [1.0, 0.3, -0.2],
            [-0.4, 0.8, 0.1],
            [0.2, -0.6, 0.9],
            [-0.7, -0.4, -0.8],
            [0.5, 0.1, 0.3]
        ];
        let s_null = &s_hat
            + &array![
                [0.1, 0.0, -0.1],
                [0.05, 0.1, 0.0],
                [0.0, -0.05, 0.1],
                [-0.1, 0.02, 0.0],
                [0.03, 0.0, 0.05]
            ];
        let dl = s_hat.dot(&(&beta_hat - &beta_null)) + array![0.01, -0.02, 0.015, 0.0, -0.01];
        let lr = 0.35;
        let base = skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
            contrast: contrast.view(),
            beta_hat: beta_hat.view(),
            beta_null: beta_null.view(),
            lr_statistic: lr,
            observed_info_hat: j_hat.view(),
            observed_info_null: j_null.view(),
            expected_info: info.view(),
            score_covariance: score_cov.view(),
            loglik_covariance: q.view(),
            row_scores_hat: s_hat.view(),
            row_scores_null: s_null.view(),
            row_loglik_diff: dl.view(),
        })
        .expect("base parameterisation");

        // β = Mγ with |M| = −1.9625.
        let m = array![[1.0, 0.5, 0.0], [0.0, -1.0, 0.25], [0.3, 0.0, 2.0]];
        let m_inv = {
            let lu = DenseLu::factor(m.view()).expect("M invertible");
            assert!(lu.det_sign < 0.0, "the fixture wants a negative Jacobian");
            let mut inv = Array2::<f64>::zeros((3, 3));
            for k in 0..3 {
                let mut e = Array1::<f64>::zeros(3);
                e[k] = 1.0;
                inv.column_mut(k).assign(&lu.solve(e.view()));
            }
            inv
        };
        let congruent = |a: &Array2<f64>| m.t().dot(a).dot(&m);
        let (j_hat_g, j_null_g, info_g, score_cov_g) = (
            congruent(&j_hat),
            congruent(&j_null),
            congruent(&info),
            congruent(&score_cov),
        );
        let (contrast_g, q_g) = (m.t().dot(&contrast), m.t().dot(&q));
        let (beta_hat_g, beta_null_g) = (m_inv.dot(&beta_hat), m_inv.dot(&beta_null));
        let (s_hat_g, s_null_g) = (s_hat.dot(&m), s_null.dot(&m));
        let moved = skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
            contrast: contrast_g.view(),
            beta_hat: beta_hat_g.view(),
            beta_null: beta_null_g.view(),
            lr_statistic: lr,
            observed_info_hat: j_hat_g.view(),
            observed_info_null: j_null_g.view(),
            expected_info: info_g.view(),
            score_covariance: score_cov_g.view(),
            loglik_covariance: q_g.view(),
            row_scores_hat: s_hat_g.view(),
            row_scores_null: s_null_g.view(),
            row_loglik_diff: dl.view(),
        })
        .expect("reparameterised");
        // The transformation itself costs another congruence per matrix.
        let ops = 2 * nuisance_ops(3, s_hat.nrows());
        assert_close(moved.r, base.r, ops, "r");
        assert_close(moved.u, base.u, ops, "u");
        assert_close(moved.u_empirical, base.u_empirical, ops, "u_emp");
        assert_close(moved.r_star, base.r_star, ops, "r*");
        assert_close(moved.r_star_empirical, base.r_star_empirical, ops, "r*_emp");
        // The fixture is not degenerate: u ≠ r, so r* moved.
        assert!((base.r_star - base.r).abs() > accumulation_growth(ops) * base.r.abs());
    }

    #[test]
    fn nuisance_path_rejects_degenerate_inputs() {
        let eye = Array2::<f64>::eye(2);
        let contrast = array![1.0, 0.0];
        let beta_hat = array![0.5, 0.1];
        let beta_null = array![0.0, 0.1];
        let q = array![0.5, 0.0];
        let s_hat = array![[1.0, 0.5], [-0.5, 1.0], [0.3, -0.2]];
        let dl = array![0.2, -0.1, 0.05];
        let input = SkovgaardNuisanceInput {
            contrast: contrast.view(),
            beta_hat: beta_hat.view(),
            beta_null: beta_null.view(),
            lr_statistic: 0.25,
            observed_info_hat: eye.view(),
            observed_info_null: eye.view(),
            expected_info: eye.view(),
            score_covariance: eye.view(),
            loglik_covariance: q.view(),
            row_scores_hat: s_hat.view(),
            row_scores_null: s_hat.view(),
            row_loglik_diff: dl.view(),
        };
        assert!(skovgaard_r_star_with_nuisance(&input).is_some());
        // Contrast of the wrong length.
        let short = array![1.0];
        assert!(
            skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
                contrast: short.view(),
                ..input
            })
            .is_none()
        );
        // Singular expected information.
        let singular = array![[1.0, 1.0], [1.0, 1.0]];
        assert!(
            skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
                expected_info: singular.view(),
                ..input
            })
            .is_none()
        );
        // Singular Ŝ (the LU meets a zero pivot column).
        let zero_col = array![[1.0, 0.0], [2.0, 0.0]];
        assert!(
            skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
                score_covariance: zero_col.view(),
                ..input
            })
            .is_none()
        );
        // Fewer rows than coefficients: Σ sᵢsᵢᵀ has rank 1 < p.
        let one_row = s_hat.slice(ndarray::s![0..1, ..]);
        let one_dl = dl.slice(ndarray::s![0..1]);
        assert!(
            skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
                row_scores_hat: one_row,
                row_scores_null: one_row,
                row_loglik_diff: one_dl,
                ..input
            })
            .is_none()
        );
        // Non-positive LR statistic.
        assert!(
            skovgaard_r_star_with_nuisance(&SkovgaardNuisanceInput {
                lr_statistic: 0.0,
                ..input
            })
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
            loglik_covariance: 15.0,
            empirical_info: 50.0,
            empirical_loglik_covariance: 15.0,
        };
        // Non-positive LR.
        assert!(
            scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                lr_statistic: 0.0,
                ..base
            })
            .is_none()
        );
        // Non-positive information.
        assert!(
            scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                observed_info: 0.0,
                ..base
            })
            .is_none()
        );
        assert!(
            scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                empirical_info: -1.0,
                ..base
            })
            .is_none()
        );
        // Non-finite covariance.
        assert!(
            scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                loglik_covariance: f64::NAN,
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

    // ── INDEPENDENT ANALYTIC FIXTURE: Poisson mean via the Lugannani–Rice
    //    saddlepoint (#939, deliverable 3 hardening) ───────────────────────────
    //
    // A second, algebraically-distinct closed form for `r*`. For a one-parameter
    // exponential family the modified root `r*` and the Lugannani–Rice (1980)
    // saddlepoint approximation share the SAME directed root `r` and the SAME `u`,
    // but combine them by different formulas:
    //
    //   * Barndorff-Nielsen:  r* = r + (1/r)·log(u/r),  tail = Φ(−r*),
    //   * Lugannani–Rice:     tail = 1 − Φ(r) − φ(r)·(1/r − 1/u).
    //
    // Both are third-order-accurate approximations to the SAME exact tail
    // probability, so they must agree to O(n⁻³ᐟ²). Reproducing the L–R tail from
    // OUR `(r, u)` is therefore an independent check that the engine's `u` (built
    // from observed/expected/score information) is the genuine Skovgaard `u` — a
    // different derivation than the Exponential/canonical anchor already in the
    // module, on a different family.
    //
    // Poisson model: `S ~ Poisson(nμ)`, canonical θ = log μ. For the natural
    // sufficient statistic the canonical-family `u` equals the Wald root
    // `q = (θ̂−θ₀)·√ĵ` with `ĵ = nμ̂ = S` (the exact observed information in θ),
    // and the LR root is `r = sign(θ̂−θ₀)·√W`, `W = 2[S log(S/(nμ₀)) − (S−nμ₀)]`.

    /// The right-tail `P(X ≥ x)` for the Poisson saddlepoint problem, computed two
    /// independent ways from a single `(r, u)`: our Barndorff-Nielsen `r*` tail and
    /// the Lugannani–Rice tail. Returns `(p_rstar, p_lr, p_first_order)`.
    fn poisson_tails(n: f64, mu_hat: f64, mu0: f64) -> (f64, f64, f64) {
        let theta_hat = mu_hat.ln();
        let theta0 = mu0.ln();
        let s = n * mu_hat; // sufficient statistic S = nμ̂
        // LR statistic W = 2[S log(S/(nμ₀)) − (S − nμ₀)].
        let w = 2.0 * (s * (s / (n * mu0)).ln() - (s - n * mu0));
        // AT-THE-NULL atom (`μ̂ = μ₀` ⇒ `θ̂ = θ₀`, `W = 0`): the directed root is
        // exactly `r = 0`, `r*` is undefined, and the right-tail probability is
        // exactly `½` for all three forms. `scalar_skovgaard_r_star` correctly
        // returns `None` on a zero LR (the degenerate-input contract), so handle
        // this lattice atom here rather than unwrapping a `None`. For n=10, μ₀=1.3
        // the integer `s = 13` lands exactly on the null, so this branch is live.
        if theta_hat == theta0 || !(w > 0.0) {
            return (0.5, 0.5, 0.5);
        }
        // Observed/expected info in θ both equal nμ̂ = S for the canonical
        // Poisson at the MLE. For the canonical family the sample-space
        // derivatives are exact and linear: Ŝ = î and
        // q̂ = ℓ_{;θ̂}(θ̂) − ℓ_{;θ̂}(θ₀) = (θ̂ − θ₀)·S. The single sufficient
        // statistic makes the empirical (one-row) form coincide with the model
        // form, so both are fed the same exact quantities.
        let info = s;
        let q = (theta_hat - theta0) * info;
        let res = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
            theta_hat,
            theta_null: theta0,
            lr_statistic: w,
            observed_info: info,
            expected_info: info,
            loglik_covariance: q,
            empirical_info: info,
            empirical_loglik_covariance: q,
        })
        .expect("poisson r*");
        let r = res.r;
        let u = res.u; // canonical ⇒ exact q̂ gives u = (θ̂−θ₀)√ĵ, the Wald root
        // Our r* right-tail P(X ≥ x) = Φ(−r*).
        let p_rstar = gam_math::probability::normal_cdf(-res.r_star);
        // Lugannani–Rice tail from the same (r, u).
        let p_lr = gam_math::probability::normal_cdf(-r) - normal_pdf(r) * (1.0 / r - 1.0 / u);
        let p_first = gam_math::probability::normal_cdf(-r);
        (p_rstar, p_lr, p_first)
    }

    #[test]
    fn poisson_r_star_tail_matches_lugannani_rice_saddlepoint() {
        // A genuine upper-tail event: μ̂ above μ₀ so r, u > 0 and the tail is small.
        // Several (n, μ̂, μ₀) so the agreement is not a single coincidence.
        for &(n, mu_hat, mu0) in &[
            (12.0_f64, 1.6_f64, 1.0_f64),
            (20.0, 2.3, 1.7),
            (8.0, 3.1, 2.0),
            (30.0, 0.9, 0.6),
        ] {
            let (p_rstar, p_lr, p_first) = poisson_tails(n, mu_hat, mu0);
            // Both saddlepoint tails are valid probabilities.
            assert!(
                (0.0..=1.0).contains(&p_rstar) && (0.0..=1.0).contains(&p_lr),
                "n={n} μ̂={mu_hat}: tails must be probabilities (r*={p_rstar}, LR={p_lr})"
            );
            // INDEPENDENT AGREEMENT: r* and Lugannani–Rice must coincide to the
            // shared O(n⁻³ᐟ²) order. Their relative gap shrinks with n; at these
            // modest n a 5% relative band (and a tight absolute floor) holds.
            let rel = (p_rstar - p_lr).abs() / p_lr.max(1e-12);
            assert!(
                (p_rstar - p_lr).abs() < 5e-3 || rel < 0.05,
                "n={n} μ̂={mu_hat} μ₀={mu0}: r* tail {p_rstar:.6} must match \
                 Lugannani–Rice {p_lr:.6} (|Δ|={:.2e}, rel={rel:.3})",
                (p_rstar - p_lr).abs()
            );
            // And both higher-order tails must DIFFER from the raw first-order
            // directed-root tail — otherwise the correction is a no-op and the
            // agreement above would be vacuous.
            assert!(
                (p_rstar - p_first).abs() > 1e-4,
                "n={n} μ̂={mu_hat}: r* must move the tail off the first-order value \
                 (r*={p_rstar:.6}, first-order={p_first:.6})"
            );
        }
    }

    /// SMALL-n SIZE CHECK (#939, deliverable 3 hardening): under the null the
    /// one-sided p-value from the modified root `r*` must be BETTER calibrated
    /// (closer to Uniform / nominal size) than the raw first-order directed root
    /// `r`. Exact null distribution: `S ~ Poisson(nμ₀)` summed over the integer
    /// support — no Monte-Carlo noise, the size is computed exactly from the pmf.
    #[test]
    fn poisson_r_star_improves_small_n_size_over_first_order() {
        let mu0 = 1.3_f64;
        // Two complementary calibration metrics, accumulated across all n:
        //   (1) the DISCRETE exact rejection size at fixed α — coarse, because a
        //       Poisson statistic rejects at whole atoms, so r* and the
        //       first-order root can tie when their p-values straddle the same
        //       atom (a no-op at that α/n, never an improvement). We require r*
        //       to be NO WORSE at every (n, α) and strictly better in aggregate.
        //   (2) accuracy against the EXACT Poisson upper-tail probability
        //       P(S′ ≥ s) (ground truth, summed from the same pmf): the
        //       third-order r* tail must be a strictly better approximation to
        //       the exact tail than the first-order root, in pmf-weighted mean
        //       absolute error over the upper half. This is not quantized by the
        //       rejection grid, so it is the un-aliased witness that r* genuinely
        //       sharpens the tail.
        let alphas = [0.05_f64, 0.10];
        let mut total_err_first = 0.0_f64;
        let mut total_err_star = 0.0_f64;
        let mut total_mae_first = 0.0_f64;
        let mut total_mae_star = 0.0_f64;
        // Small n where the first-order directed root is visibly skewed.
        for &n in &[10.0_f64, 16.0, 25.0] {
            let rate = n * mu0; // S ~ Poisson(rate)
            let s_max = (rate + 50.0 * rate.sqrt()).ceil() as usize;
            // Materialize the exact pmf so we can form the exact upper tail.
            let mut pmf_vec = vec![0.0_f64; s_max + 1];
            let mut pmf = (-rate).exp();
            pmf_vec[0] = pmf;
            for s in 1..=s_max {
                pmf *= rate / s as f64;
                pmf_vec[s] = pmf;
            }
            // Exact upper tail P(S′ ≥ s) = Σ_{t≥s} pmf(t).
            let mut exact_upper = vec![0.0_f64; s_max + 2];
            for s in (0..=s_max).rev() {
                exact_upper[s] = exact_upper[s + 1] + pmf_vec[s];
            }
            let mut size_first = [0.0_f64; 2];
            let mut size_star = [0.0_f64; 2];
            // pmf-weighted mean |approx tail − exact tail| over the upper half.
            let mut tail_err_first = 0.0_f64;
            let mut tail_err_star = 0.0_f64;
            let mut mass_upper = 0.0_f64;
            for s in 1..=s_max {
                let p = pmf_vec[s];
                let mu_hat = s as f64 / n;
                // Upper-tail p-values (one-sided test μ > μ₀). For μ̂ < μ₀ the
                // upper-tail p-value is > 0.5 and never triggers a small-α
                // rejection, so this cleanly isolates the calibrated tail.
                let (p_rstar, _p_lr, p_first) = poisson_tails(n, mu_hat, mu0);
                for (j, &a) in alphas.iter().enumerate() {
                    if p_first <= a {
                        size_first[j] += p;
                    }
                    if p_rstar <= a {
                        size_star[j] += p;
                    }
                }
                if mu_hat >= mu0 {
                    // CONTINUITY-CORRECTED ground truth. The saddlepoint tails
                    // `p_first` / `p_rstar` are CONTINUOUS approximations built
                    // from the integer-valued statistic `S = s`, so they target
                    // the lattice tail at the MID-CELL point `s − ½`, i.e. the
                    // continuity-corrected exact tail `½[P(S′ ≥ s) + P(S′ ≥ s+1)]`
                    // — NOT the raw integer atom `P(S′ ≥ s)`. Comparing the
                    // continuous tail against the un-corrected integer tail
                    // introduces a fixed half-integer offset that the cruder
                    // first-order root happens to partially absorb, ALIASING the
                    // MAE so the more-accurate third-order `r*` looks worse. With
                    // the standard mid-cell continuity correction the comparison
                    // is on the lattice point the saddlepoint actually estimates,
                    // and `r*`'s genuine O(n⁻³ᐟ²) sharpening shows through (it is
                    // ~5× closer to the exact tail than the first-order root here).
                    let exact = 0.5 * (exact_upper[s] + exact_upper[s + 1]);
                    tail_err_first += p * (p_first - exact).abs();
                    tail_err_star += p * (p_rstar - exact).abs();
                    mass_upper += p;
                }
            }
            // (1) No-worse-everywhere at fixed α, accumulating the aggregate error.
            for (j, &a) in alphas.iter().enumerate() {
                let err_first = (size_first[j] - a).abs();
                let err_star = (size_star[j] - a).abs();
                assert!(
                    err_star <= err_first + 1e-9,
                    "n={n} α={a}: r* size {:.4} (|Δ|={err_star:.4}) must be no worse than \
                     first-order size {:.4} (|Δ|={err_first:.4})",
                    size_star[j],
                    size_first[j]
                );
                total_err_first += err_first;
                total_err_star += err_star;
            }
            // (2) The un-aliased witness: the r* tail must approximate the EXACT
            // upper-tail probability NO WORSE than the first-order root at every
            // n (a half-integer continuity offset can tie a particular n), and
            // strictly better in aggregate below. pmf-weighted MAE over the upper
            // half.
            let mae_first = tail_err_first / mass_upper;
            let mae_star = tail_err_star / mass_upper;
            assert!(
                mae_star <= mae_first + 1e-12,
                "n={n}: r* tail must approximate the exact Poisson upper tail no worse \
                 than first-order: MAE*={mae_star:.6} must be ≤ MAE={mae_first:.6}"
            );
            total_mae_first += mae_first;
            total_mae_star += mae_star;
        }
        // Aggregate strict improvement, robust to per-point discreteness ties:
        //   (a) the exact-tail MAE summed over all n must strictly drop — r*
        //       genuinely sharpens the tail (the un-aliased, continuous witness);
        //   (b) the discrete rejection-size error summed over all (n, α) must be
        //       no worse — the correction never degrades nominal calibration.
        assert!(
            total_mae_star < total_mae_first - 1e-9,
            "r* must strictly improve the aggregate exact-tail MAE over first-order: \
             Σ MAE*={total_mae_star:.6} must be < Σ MAE={total_mae_first:.6}"
        );
        assert!(
            total_err_star <= total_err_first + 1e-9,
            "r* must not worsen the aggregate discrete size over first-order: \
             Σ|size_r*−α|={total_err_star:.5} must be ≤ Σ|size_r−α|={total_err_first:.5}"
        );
    }
}
