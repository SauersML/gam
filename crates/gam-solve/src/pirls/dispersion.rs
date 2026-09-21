//! Exact nuisance-scale estimation from a certified linear predictor.
//!
//! Every estimator consumes the same inverse-link surface as the PIRLS working
//! state.  Eta projection, mean floors, neutral fallback values, and silently
//! returned parameter-band endpoints are forbidden: if a required statistic or
//! a finite interior nuisance estimate cannot be represented, the fit fails
//! closed.

use super::*;

/// The NB2 profile score `∂ℓ/∂θ`, its observed information `−∂²ℓ/∂θ²`, and the
/// rounding band of the score's accumulation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NegbinThetaScore {
    pub(crate) score: f64,
    pub(crate) info: f64,
    pub(crate) band: f64,
}

impl NegbinThetaScore {
    /// Whether the score is positive by more than its own rounding band.
    pub(crate) fn resolvably_positive(&self) -> bool {
        self.score > self.band
    }
}

fn certified_log_means(eta: &Array1<f64>) -> Result<Vec<f64>, EstimationError> {
    super::par_certified_rows(eta.len(), |i| crate::mixture_link::log_link_solver_exp(eta[i]))
}

/// Per-row means read from the same inverse-link surface as the PIRLS working
/// state: the generic variance × link cell's link, a reciprocal power
/// `μ = η^{−a}`, the log link, or the Gaussian identity. An `η` outside the
/// link domain fails exactly as the PIRLS row does.
fn certified_link_means(
    response: &ResponseFamily,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
) -> Result<Vec<f64>, EstimationError> {
    if let Some(cell) = GenericEdmCell::classify(response, inverse_link) {
        return super::par_certified_rows(eta.len(), |i| generic_edm_mean(cell, i, eta[i]));
    }
    if let Some((link, exponent)) = reciprocal_power_link(inverse_link) {
        return super::par_certified_rows(eta.len(), |i| {
            require_reciprocal_link_domain(link, eta[i])?;
            let mu = (-exponent * eta[i].ln()).exp();
            if mu.is_finite() && mu > 0.0 {
                Ok(mu)
            } else {
                Err(EstimationError::pirls_row_geometry_unrepresentable(i, "mean", eta[i], mu))
            }
        });
    }
    match (response, inverse_link) {
        (_, InverseLink::Standard(StandardLink::Log)) => certified_log_means(eta),
        (ResponseFamily::Gaussian, InverseLink::Standard(StandardLink::Identity)) => {
            super::par_certified_rows(eta.len(), |i| {
                if eta[i].is_finite() {
                    Ok(eta[i])
                } else {
                    Err(EstimationError::pirls_row_geometry_unrepresentable(i, "mean", eta[i], eta[i]))
                }
            })
        }
        (_, other) => crate::bail_invalid_estim!(
            "nuisance-scale estimation has no inverse link surface for {response:?} with {other:?}"
        ),
    }
}

#[inline]
fn certified_prior_weight(row: usize, eta: f64, weight: f64) -> Result<f64, EstimationError> {
    if weight.is_finite() && weight >= 0.0 {
        Ok(weight)
    } else {
        Err(EstimationError::pirls_row_geometry_unrepresentable(row, "prior weight", eta, weight))
    }
}

fn certified_pairs_sum(rows: &[(f64, f64)]) -> Result<(f64, f64), EstimationError> {
    let sum = gam_linalg::pairwise_reduce::par_pairwise_map_reduce(
        rows.len(),
        |i| rows[i],
        |(a, b), (c, d)| (a + c, b + d),
        (0.0, 0.0),
    );
    if sum.0.is_finite() && sum.1.is_finite() {
        Ok(sum)
    } else {
        Err(EstimationError::InvalidInput(
            "nuisance-profile reduction exceeded the finite f64 range".to_string(),
        ))
    }
}

#[inline]
pub(crate) fn gamma_shape_score(shape: f64, target: f64) -> f64 {
    let log_minus_digamma = if shape >= 32.0 {
        // ln(a)-psi(a) = 1/(2a)+1/(12a²)-1/(120a⁴)+... .
        // Subtracting the two logarithmic-size values loses the entire
        // score at large a. The first omitted term here is O(a^-12),
        // below one ulp of the result throughout this branch.
        let inv = shape.recip();
        let inv2 = inv * inv;
        0.5 * inv
            + inv2
                * (1.0 / 12.0
                    + inv2
                        * (-1.0 / 120.0
                            + inv2 * (1.0 / 252.0 + inv2 * (-1.0 / 240.0 + inv2 / 132.0))))
    } else {
        shape.ln() - digamma(shape)
    };
    log_minus_digamma - target
}

#[inline]
fn gamma_shape_statistic(response: f64, mean: f64) -> f64 {
    let relative_residual = (response - mean) / mean;
    if relative_residual.abs() <= 0.5 {
        // r-ln(r)-1 = d-ln(1+d), d=(y-mu)/mu. The subtraction
        // y-mu preserves nearby represented responses before division.
        -gam_math::special::log1p_minus_x(relative_residual)
    } else {
        // The log ratio remains finite when y/mu underflows to zero.
        gam_math::special::expm1_minus_x(response.ln() - mean.ln())
    }
}

/// Residual degrees of freedom `n₊ − edf` left to a scale estimate once the
/// mean model has spent `mean_model_edf` effective degrees of freedom on the
/// `n₊` positive-weight rows.
///
/// `mean_model_edf = 0` is the plug-in reading (the likelihood maximized in the
/// scale at a fixed `η`). The fitted `edf = tr(F)` of the penalized fit
/// behind `η` is the residual-df reading. It is the stationary point of the
/// Laplace-approximate marginal likelihood in the scale at fixed `λ`: with
/// `H = XᵀW₀X/φ + S_λ`, the term `−½ log|H|` contributes
/// `∂/∂φ = tr(H⁻¹XᵀW₀X/φ)/(2φ) = edf/(2φ)` to the scale score.
fn residual_degrees_of_freedom(
    positive_rows: f64,
    mean_model_edf: f64,
    what: &str,
) -> Result<f64, EstimationError> {
    if !(mean_model_edf.is_finite() && mean_model_edf >= 0.0) {
        crate::bail_invalid_estim!(
            "{what}: mean-model effective degrees of freedom must be finite and nonnegative, got {mean_model_edf:?}"
        );
    }
    let residual_df = positive_rows - mean_model_edf;
    if !(residual_df > 0.0) {
        crate::bail_invalid_estim!(
            "{what}: no residual degrees of freedom are left for the scale ({positive_rows} positive-weight rows, mean-model edf {mean_model_edf})"
        );
    }
    Ok(residual_df)
}

/// Weighted, residual-df-corrected Gamma shape score
/// `Σ_g f_g·[ln(w_g α) − ψ(w_g α) − κ/(2 w_g α) − t̄]`.
///
/// `groups` holds each distinct positive prior weight `w_g` with its share
/// `f_g = (count_g · w_g) / Σ w` of the total prior weight, so `Σ f_g = 1`.
/// `κ = edf / n₊` is the fraction of the rows the mean model spent. The score
/// then equals `(Σ w)⁻¹ · (∂ℓ/∂α − edf/(2α))`: the shape derivative of the
/// precision-weighted Gamma log-likelihood (row shape `wᵢ α`) plus the
/// `−½ log|H|` term of the Laplace marginal. `H` carries the Fisher weight
/// `∝ α`, so that term contributes `−tr(H⁻¹XᵀWX)/(2α) = −edf/(2α)`, spread
/// evenly as `−κ/(2α)` per row.
///
/// For `κ < 1` every term is strictly decreasing in `α`. The bound
/// `ψ'(x) > 1/x + 1/(2x²)` gives `d/dx[ln x − ψ(x) − κ/(2x)] < −(1 − κ)/(2x²)`.
/// Each term falls from `(1 − κ/2)/x → +∞` as `x → 0⁺` to `(1 − κ)/(2x) → 0⁺`,
/// so a positive `t̄` has exactly one root. A product `w_g α` that underflows
/// to zero takes its `α → 0⁺` limit `+∞`, as does a row whose `ln x − ψ(x)`
/// has already overflowed (the `κ` term is at most half of it there). One that
/// overflows takes the `α → ∞` limit `0 − t̄`, which the asymptotic branch of
/// [`gamma_shape_score`] already returns at `+∞`.
fn weighted_gamma_shape_score(groups: &[(f64, f64)], shape: f64, target: f64, kappa: f64) -> f64 {
    let mut score = 0.0;
    for &(weight, share) in groups {
        let row_shape = weight * shape;
        if row_shape == 0.0 {
            return f64::INFINITY;
        }
        let row_score = gamma_shape_score(row_shape, target);
        if row_score == f64::INFINITY {
            return f64::INFINITY;
        }
        score += share * (row_score - 0.5 * kappa / row_shape);
    }
    score
}

/// Exact Gamma shape MLE at a certified linear predictor, with `μ` read from
/// the fit's own inverse link.
///
/// Prior weights are precisions, exactly as in the reported log-likelihood
/// (`gamma_saturated_log_normalizer` evaluates row `i` at shape `wᵢ α`) and in
/// the Gaussian identity scale `φ̂ = Σ wᵢ rᵢ² / (n₊ − edf)`: row `i` is
/// `Gamma(shape = wᵢ α, mean = μᵢ)`. The shape score is therefore
/// `∂ℓ/∂α = Σ wᵢ [ln(wᵢ α) − ψ(wᵢ α) − tᵢ]`, `tᵢ = yᵢ/μᵢ − ln(yᵢ/μᵢ) − 1`,
/// not the frequency-weight score `Σ wᵢ [ln α − ψ(α) − tᵢ]`. Only weight
/// ratios carry information: a global rescale `w → c·w` maps the MLE to
/// `α̂/c`, so every row shape `wᵢ α̂` — hence `β̂`, its covariance and the
/// log-likelihood — is unchanged. With unit weights both scores coincide.
///
/// `mean_model_edf` is the effective degrees of freedom of the fit behind `η`
/// (see [`residual_degrees_of_freedom`]). The score solved is
/// `∂ℓ/∂α − edf/(2α)`, so with unit weights and a large shape the root is
/// `α̂ ≈ (n₊ − edf)/D` with `D = 2 Σ tᵢ` the Gamma deviance. That is
/// `φ̂ = D/(n₊ − edf)`, the residual-df scale; `edf = 0` gives the plug-in
/// `D/n₊`, which is biased low by `(n₊ − edf)/n₊`.
pub(crate) fn estimate_gamma_shape_from_eta(
    inverse_link: &InverseLink,
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    mean_model_edf: f64,
) -> Result<f64, EstimationError> {
    let means = certified_link_means(&ResponseFamily::Gamma, inverse_link, eta)?;
    let rows: Vec<(f64, f64)> = super::par_certified_rows(eta.len(), |i| {
        let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
        if wi == 0.0 {
            return Ok((0.0, 0.0));
        }
        if !(y[i].is_finite() && y[i] > 0.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Gamma response", eta[i], y[i]));
        }
        let target = gamma_shape_statistic(y[i], means[i]);
        let contribution = wi * target;
        if !(target.is_finite() && target >= 0.0 && contribution.is_finite()) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "Gamma shape statistic",
                eta[i],
                contribution,
            ));
        }
        Ok((contribution, wi))
    })?;
    let (weighted_target, total_weight) = certified_pairs_sum(&rows)?;
    if !(total_weight > 0.0) {
        crate::bail_invalid_estim!("Gamma shape profiling requires positive total prior weight");
    }
    let target = weighted_target / total_weight;
    // Every row is now a nonnegative, cancellation-free statistic. A small
    // positive value is real dispersion; only zero has no finite shape MLE.
    if !(target.is_finite() && target > 0.0) {
        crate::bail_invalid_estim!(
            "Gamma shape MLE is not finite: the dispersion statistic is {target:?}"
        );
    }

    // Group the positive prior weights by value: the score depends on a row
    // only through its weight, and a uniform-weight design (the common case)
    // collapses to a single scalar term.
    let mut positive_weights: Vec<f64> = priorweights.iter().copied().filter(|&w| w > 0.0).collect();
    positive_weights.sort_by(f64::total_cmp);
    let mut groups: Vec<(f64, f64)> = Vec::new();
    let mut start = 0;
    while start < positive_weights.len() {
        let weight = positive_weights[start];
        let mut end = start + 1;
        while end < positive_weights.len() && positive_weights[end] == weight {
            end += 1;
        }
        let share = ((end - start) as f64 * weight) / total_weight;
        // A share below the smallest subnormal carries no representable part
        // of the normalized score; keeping it would only turn `0 · ∞` into NaN
        // at the `α → 0⁺` edge.
        if share > 0.0 {
            groups.push((weight, share));
        }
        start = end;
    }
    let positive_rows = positive_weights.len() as f64;
    let residual_df =
        residual_degrees_of_freedom(positive_rows, mean_model_edf, "Gamma shape profiling")?;
    let kappa = mean_model_edf / positive_rows;
    // The closed-form approximation solves the unit-weight plug-in score
    // `g(a) = t̄`. The weighted root sits near that row shape divided by the
    // mean weight, and the residual-df term scales the large-shape asymptote
    // `(1 − κ)/(2a) = t̄` by `1 − κ = (n₊ − edf)/n₊`.
    let mean_weight = total_weight / positive_rows;

    // Each corrected row term falls from `+∞` as `α → 0⁺` to `0⁺` as `α → ∞`,
    // so for a positive statistic the score has exactly one root. Bracket it by
    // halving and doubling outward from the closed-form approximation; the only
    // way out of either walk is the representable range itself.
    let row_shape_approx = if target < 3.0 {
        let delta = 3.0 - target;
        (delta + (delta * delta + 24.0 * target).sqrt()) / 12.0 / target
    } else {
        // Rationalize the numerator and divide the radical by target:
        // 2/(sqrt(t²+18t+9)+t-3). Neither t² nor 12t is formed.
        let inv = target.recip();
        (2.0 * inv) / ((1.0 + 18.0 * inv + 9.0 * inv * inv).sqrt() + 1.0 - 3.0 * inv)
    };
    let approx = row_shape_approx * (residual_df / positive_rows) / mean_weight;
    if !(approx.is_finite() && approx > 0.0) {
        crate::bail_invalid_estim!(
            "Gamma shape approximation is not representable (profile target={target:?}, approximation={approx:?}, mean prior weight={mean_weight:?})"
        );
    }
    let score = |shape: f64| weighted_gamma_shape_score(&groups, shape, target, kappa);
    let mut lo = approx;
    let mut hi = approx;
    while score(lo) <= 0.0 {
        lo *= 0.5;
        if !(lo > 0.0) {
            crate::bail_invalid_estim!(
                "Gamma shape MLE lies below the representable range (profile target={target:?})"
            );
        }
    }
    while score(hi) > 0.0 {
        if hi == f64::MAX {
            crate::bail_invalid_estim!(
                "Gamma shape MLE exceeds the representable range (profile target={target:?})"
            );
        }
        hi = if hi <= 0.5 * f64::MAX { 2.0 * hi } else { f64::MAX };
    }
    // Bisect until no representable shape lies strictly inside the bracket.
    loop {
        let mid = lo + 0.5 * (hi - lo);
        if !(mid > lo && mid < hi) {
            break;
        }
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let shape = lo + 0.5 * (hi - lo);
    if shape.is_finite() && shape > 0.0 {
        Ok(shape)
    } else {
        crate::bail_invalid_estim!("Gamma shape solve produced {shape:?}")
    }
}

/// Exact method-of-moments Beta precision on the represented logit surface.
pub(crate) fn estimate_beta_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<f64, EstimationError> {
    let rows: Vec<(f64, f64)> = super::par_certified_rows(eta.len(), |i| {
        let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
        if wi == 0.0 {
            return Ok((0.0, 0.0));
        }
        if !(y[i].is_finite() && y[i] > 0.0 && y[i] < 1.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Beta response", eta[i], y[i]));
        }
        if !eta[i].is_finite() {
            return Err(EstimationError::InverseLinkDomainViolation {
                link: "standard logit inverse link",
                eta: eta[i],
                lower: -f64::MAX,
                upper: f64::MAX,
            });
        }
        // The exact logit pair, as in the working-state row: the rounded mean
        // reaches 1 at eta ~ 36.7 while 1 - mu is still a normal number, so
        // neither the residual nor the variance may be formed from the rounded
        // mean.
        let (mu, one_minus_mu) = logit_probability_pair(eta[i]);
        let variance = mu * one_minus_mu;
        if !(variance.is_finite() && variance > 0.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Beta mean/variance", eta[i], variance));
        }
        // In the upper half take y - mu = (1 - mu) - (1 - y), so the residual
        // carries the complement's full relative precision (1 - y is exact
        // for y >= 1/2).
        let resid = if mu > 0.5 { one_minus_mu - (1.0 - y[i]) } else { y[i] - mu };
        let statistic = wi * (resid / variance) * resid;
        if !(statistic.is_finite() && statistic >= 0.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "Beta precision statistic",
                eta[i],
                statistic,
            ));
        }
        Ok((statistic, wi))
    })?;
    let (weighted_pearson, total_weight) = certified_pairs_sum(&rows)?;
    if !(total_weight > 0.0 && weighted_pearson > 0.0) {
        crate::bail_invalid_estim!(
            "Beta precision MLE is not finite and positive (Pearson={weighted_pearson:?}, weight={total_weight:?})"
        );
    }
    let phi = total_weight / weighted_pearson - 1.0;
    if phi.is_finite() && phi > 0.0 {
        Ok(phi)
    } else {
        crate::bail_invalid_estim!("Beta precision estimate is not finite and positive: {phi:?}")
    }
}

/// Exact Pearson Tweedie dispersion on the represented log-link surface.
///
/// Prior weights are precisions, `Var(yᵢ) = φ μᵢ^p / wᵢ` — the same convention
/// as the reported Tweedie log-likelihood (evaluated at dispersion `φ/wᵢ`) and
/// the Gaussian identity scale. The moment identity `E[wᵢ (yᵢ−μᵢ)²/μᵢ^p] = φ`
/// holds row by row, so `φ̂ = Σ wᵢ (yᵢ−μᵢ)²/μᵢ^p / n₊` over the `n₊`
/// positive-weight rows. Dividing by `Σ wᵢ` instead would read the weights as
/// replicate counts: a global rescale `w → c·w` would then leave `φ̂` fixed
/// while the working weights `w/φ̂` grow by `c`, shrinking every SE by `√c`.
///
/// At the fitted `μ̂` of a penalized fit with `edf = tr(F)`, the Pearson sum
/// has `E[Σ wᵢ (yᵢ−μ̂ᵢ)²/μ̂ᵢ^p] ≈ φ (n₊ − edf)`, so the scale divides by the
/// residual degrees of freedom `n₊ − mean_model_edf` (see
/// [`residual_degrees_of_freedom`]; `0` is the plug-in moment at fixed `η`).
pub(crate) fn estimate_tweedie_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    p: f64,
    mean_model_edf: f64,
) -> Result<f64, EstimationError> {
    if !is_valid_tweedie_power(p) {
        crate::bail_invalid_estim!("invalid Tweedie variance power {p:?}");
    }
    let means = certified_log_means(eta)?;
    let rows: Vec<(f64, f64)> = super::par_certified_rows(eta.len(), |i| {
        let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
        if wi == 0.0 {
            return Ok((0.0, 0.0));
        }
        if !(y[i].is_finite() && y[i] >= 0.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Tweedie response", eta[i], y[i]));
        }
        let resid = y[i] - means[i];
        // Form the complete Pearson term before exponentiating. Both
        // residual² and mu^p may exceed the float range while their
        // weighted ratio remains finite and informative.
        let statistic = if resid == 0.0 {
            0.0
        } else {
            (wi.ln() + 2.0 * resid.abs().ln() - p * means[i].ln()).exp()
        };
        if !(statistic.is_finite() && statistic >= 0.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "Tweedie dispersion statistic",
                eta[i],
                statistic,
            ));
        }
        Ok((statistic, 1.0))
    })?;
    let (weighted_pearson, positive_rows) = certified_pairs_sum(&rows)?;
    if !(positive_rows > 0.0 && weighted_pearson > 0.0) {
        crate::bail_invalid_estim!(
            "Tweedie dispersion is not finite and positive (Pearson={weighted_pearson:?}, positive-weight rows={positive_rows:?})"
        );
    }
    let residual_df =
        residual_degrees_of_freedom(positive_rows, mean_model_edf, "Tweedie dispersion")?;
    let phi = weighted_pearson / residual_df;
    if phi.is_finite() && phi > 0.0 {
        Ok(phi)
    } else {
        crate::bail_invalid_estim!("Tweedie dispersion estimate is invalid: {phi:?}")
    }
}

/// Dispersion `φ̂ = Σ wᵢ dᵢ / (n₊ − edf)` for the families whose
/// log-likelihood is `−dᵢ/(2φ/wᵢ) − ½ log(φ/wᵢ) + c(yᵢ)`: the Gaussian
/// (`d = (y−μ)²`) and the inverse Gaussian (`d = (y−μ)²/(y μ²)`).
///
/// Prior weights are precisions (`Var(yᵢ) ∝ φ/wᵢ`), exactly as in the reported
/// log-likelihood and the Gaussian identity scale `Σ wᵢ rᵢ² / (n₊ − edf)`.
/// `∂ℓ/∂φ = Σᵢ [wᵢ dᵢ/(2φ²) − 1/(2φ)]` over the `n₊` positive-weight rows, so
/// the stationary point divides by `n₊`, not by `Σ wᵢ` (which would be the MLE
/// of the frequency-weight likelihood `Σ wᵢ [−dᵢ/(2φ) − ½ log φ]`). A global
/// rescale `w → c·w` maps `φ̂ → c·φ̂` and leaves `β̂` and its covariance fixed.
///
/// `mean_model_edf = 0` returns that plug-in MLE at the given `η`. The fitted
/// `edf` of the penalized fit behind `η` returns the stationary point
/// `Σ wᵢ dᵢ / (n₊ − edf)` of the Laplace marginal in `φ` (see
/// [`residual_degrees_of_freedom`]), the same residual-df scale as the
/// Gaussian identity `Σ wᵢ rᵢ² / (n₊ − edf)`: `E[Σ wᵢ dᵢ(μ̂)] ≈ φ (n₊ − edf)`.
///
/// `μ` is read from the same inverse-link surface as the working state
/// (the generic variance × link cell's link, a reciprocal power
/// `μ = η^{−a}`, or the log link), so an `η` outside the link domain fails
/// exactly as the PIRLS row does.
pub(crate) fn estimate_dispersion_phi_from_eta(
    response: &ResponseFamily,
    inverse_link: &InverseLink,
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    mean_model_edf: f64,
) -> Result<f64, EstimationError> {
    let inverse_gaussian = match response {
        ResponseFamily::Gaussian => false,
        ResponseFamily::InverseGaussian => true,
        other => crate::bail_invalid_estim!(
            "dispersion φ̂ is defined for the Gaussian and inverse Gaussian families, not {other:?}"
        ),
    };
    let means = certified_link_means(response, inverse_link, eta)?;
    let rows: Vec<(f64, f64)> = super::par_certified_rows(eta.len(), |i| {
        let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
        if wi == 0.0 {
            return Ok((0.0, 0.0));
        }
        let mu = means[i];
        let statistic = if inverse_gaussian {
            if !(y[i].is_finite() && y[i] > 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    i,
                    "inverse Gaussian response",
                    eta[i],
                    y[i],
                ));
            }
            let resid = y[i] - mu;
            // w (y−μ)²/(y μ²) = w · y · (1/μ − 1/y)²; assembled in logs so
            // neither the squared residual nor μ² has to be representable.
            if resid == 0.0 {
                0.0
            } else {
                (wi.ln() + 2.0 * resid.abs().ln() - y[i].ln() - 2.0 * mu.ln()).exp()
            }
        } else {
            if !y[i].is_finite() {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
                    i,
                    "Gaussian response",
                    eta[i],
                    y[i],
                ));
            }
            let resid = y[i] - mu;
            wi * resid * resid
        };
        if !(statistic.is_finite() && statistic >= 0.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "dispersion statistic",
                eta[i],
                statistic,
            ));
        }
        Ok((statistic, 1.0))
    })?;
    let (weighted_deviance, positive_rows) = certified_pairs_sum(&rows)?;
    if !(positive_rows > 0.0 && weighted_deviance > 0.0) {
        crate::bail_invalid_estim!(
            "dispersion MLE is not finite and positive (deviance={weighted_deviance:?}, positive-weight rows={positive_rows:?})"
        );
    }
    let residual_df = residual_degrees_of_freedom(positive_rows, mean_model_edf, "dispersion")?;
    let phi = weighted_deviance / residual_df;
    if phi.is_finite() && phi > 0.0 {
        Ok(phi)
    } else {
        crate::bail_invalid_estim!("dispersion estimate is invalid: {phi:?}")
    }
}

#[cfg(test)]
mod gamma_tweedie_profile_math_tests {
    use super::*;

    #[test]
    fn gamma_statistic_retains_near_unit_ratios_and_underflowed_ratios() {
        for response in [1.0_f64 - 1.0e-8, 1.0 + 1.0e-8] {
            let delta = response - 1.0;
            let expected = 0.5 * delta * delta - delta.powi(3) / 3.0;
            let actual = gamma_shape_statistic(response, 1.0);
            assert!(actual > 0.0);
            assert!((actual / expected - 1.0).abs() < 1.0e-14);
        }
        let target = gamma_shape_statistic(1.0e-300, 1.0e300);
        assert!((target - (600.0 * std::f64::consts::LN_10 - 1.0)).abs() < 1.0e-12);
        assert_eq!(gamma_shape_statistic(1.0, 1.0), 0.0);
    }

    #[test]
    fn gamma_shape_score_retains_large_shape_information() {
        for shape in [1.0e16_f64, 1.0e200, f64::MAX] {
            let score = gamma_shape_score(shape, 0.0);
            assert!(score > 0.0);
            assert!((score / (0.5 / shape) - 1.0).abs() < 1.0e-14);
        }
    }

    #[test]
    fn gamma_profile_fits_small_nonzero_dispersion_and_large_targets() {
        let y = Array1::from(vec![1.0 - 1.0e-8, 1.0 + 1.0e-8]);
        let eta = Array1::zeros(2);
        let weights = Array1::ones(2);
        let shape = estimate_gamma_shape_from_eta(&InverseLink::Standard(StandardLink::Log), y.view(), &eta, weights.view(), 0.0)
            .expect("a nonzero dispersion has a finite Gamma shape");
        let mean_square = 0.5 * ((y[0] - 1.0).powi(2) + (y[1] - 1.0).powi(2));
        assert!((shape * mean_square - 1.0).abs() < 1.0e-7);
        let target = 0.5 * (gamma_shape_statistic(y[0], 1.0) + gamma_shape_statistic(y[1], 1.0));
        assert!(gamma_shape_score(0.99 * shape, target) > 0.0);
        assert!(gamma_shape_score(1.01 * shape, target) < 0.0);

        let large_y = Array1::from(vec![1.0e200]);
        let shape = estimate_gamma_shape_from_eta(&InverseLink::Standard(StandardLink::Log), large_y.view(), &Array1::zeros(1), Array1::ones(1).view(), 0.0)
            .expect("a large profile target has a small finite Gamma shape");
        assert!((shape * large_y[0] - 1.0).abs() < 1.0e-12);

        assert!(estimate_gamma_shape_from_eta(&InverseLink::Standard(StandardLink::Log), Array1::ones(2).view(), &eta, weights.view(), 0.0).is_err());
    }

    /// Prior weights are precisions (row shape `wᵢ α`): the shape MLE is the
    /// root of `Σ wᵢ [ln(wᵢ α) − ψ(wᵢ α) − tᵢ]`, and a global rescale
    /// `w → c·w` maps it to `α̂/c`. A zero-weight row carries no information.
    #[test]
    fn gamma_shape_reads_prior_weights_as_precisions() {
        let log = InverseLink::Standard(StandardLink::Log);
        let y = Array1::from(vec![0.4, 1.3, 0.9, 2.2, 0.7, 1.6, 3.0]);
        let eta = Array1::from(vec![-0.3, 0.1, 0.0, 0.5, -0.2, 0.4, 0.6]);
        let weights = Array1::from(vec![0.5, 2.0, 1.0, 3.5, 0.25, 1.5, 0.0]);
        let shape = estimate_gamma_shape_from_eta(&log, y.view(), &eta, weights.view(), 0.0)
            .expect("weighted Gamma shape is finite");
        let precision_score = |alpha: f64| -> f64 {
            (0..y.len())
                .filter(|&i| weights[i] > 0.0)
                .map(|i| {
                    let w = weights[i];
                    let t = gamma_shape_statistic(y[i], eta[i].exp());
                    w * ((w * alpha).ln() - digamma(w * alpha) - t)
                })
                .sum()
        };
        assert!(precision_score(shape * (1.0 - 1.0e-6)) > 0.0);
        assert!(precision_score(shape * (1.0 + 1.0e-6)) < 0.0);

        for c in [1.0e-3_f64, 7.0, 1.0e3] {
            let scaled = weights.mapv(|w| c * w);
            let scaled_shape = estimate_gamma_shape_from_eta(&log, y.view(), &eta, scaled.view(), 0.0)
                .expect("rescaled weighted Gamma shape is finite");
            assert!(
                (scaled_shape * c / shape - 1.0).abs() < 1.0e-12,
                "shape must scale as 1/c under w -> c w: c={c}, {scaled_shape} vs {shape}"
            );
        }
    }

    /// Tweedie Pearson φ̂ and the Gaussian / inverse-Gaussian dispersion MLE
    /// divide by the positive-weight row count: `E[wᵢ dᵢ] = φ` per row when
    /// `Var(yᵢ) ∝ φ/wᵢ`, so a global rescale `w → c·w` maps `φ̂ → c·φ̂`.
    #[test]
    fn dispersion_estimates_read_prior_weights_as_precisions() {
        let log = InverseLink::Standard(StandardLink::Log);
        let y = Array1::from(vec![0.4, 1.3, 0.9, 2.2, 0.7, 1.6, 3.0]);
        let eta = Array1::from(vec![-0.3, 0.1, 0.0, 0.5, -0.2, 0.4, 0.6]);
        let weights = Array1::from(vec![0.5, 2.0, 1.0, 3.5, 0.25, 1.5, 0.0]);
        let positive_rows = weights.iter().filter(|&&w| w > 0.0).count() as f64;
        let p = 1.5;
        let tweedie_expected = (0..y.len())
            .map(|i| {
                let mu = eta[i].exp();
                weights[i] * (y[i] - mu).powi(2) / mu.powf(p)
            })
            .sum::<f64>()
            / positive_rows;
        let gaussian_expected = (0..y.len())
            .map(|i| weights[i] * (y[i] - eta[i].exp()).powi(2))
            .sum::<f64>()
            / positive_rows;
        let inverse_gaussian_expected = (0..y.len())
            .map(|i| {
                let mu = eta[i].exp();
                weights[i] * (y[i] - mu).powi(2) / (y[i] * mu * mu)
            })
            .sum::<f64>()
            / positive_rows;
        for c in [1.0_f64, 1.0e-3, 7.0, 1.0e3] {
            let scaled = weights.mapv(|w| c * w);
            let tweedie = estimate_tweedie_phi_from_eta(y.view(), &eta, scaled.view(), p, 0.0)
                .expect("weighted Tweedie phi is finite");
            assert!((tweedie / (c * tweedie_expected) - 1.0).abs() < 1.0e-12);
            let gaussian = estimate_dispersion_phi_from_eta(
                &ResponseFamily::Gaussian,
                &log,
                y.view(),
                &eta,
                scaled.view(),
                0.0,
            )
            .expect("weighted Gaussian phi is finite");
            assert!((gaussian / (c * gaussian_expected) - 1.0).abs() < 1.0e-12);
            let inverse_gaussian = estimate_dispersion_phi_from_eta(
                &ResponseFamily::InverseGaussian,
                &log,
                y.view(),
                &eta,
                scaled.view(),
                0.0,
            )
            .expect("weighted inverse Gaussian phi is finite");
            assert!((inverse_gaussian / (c * inverse_gaussian_expected) - 1.0).abs() < 1.0e-12);
        }
    }

    /// With the fitted mean model's `edf`, the Pearson / deviance scales divide
    /// by the residual degrees of freedom `n₊ − edf` (#4075). With no residual
    /// degrees of freedom there is no scale estimate at all.
    #[test]
    fn dispersion_estimates_divide_by_residual_degrees_of_freedom() {
        let log = InverseLink::Standard(StandardLink::Log);
        let y = Array1::from(vec![0.4, 1.3, 0.9, 2.2, 0.7, 1.6, 3.0]);
        let eta = Array1::from(vec![-0.3, 0.1, 0.0, 0.5, -0.2, 0.4, 0.6]);
        let weights = Array1::from(vec![0.5, 2.0, 1.0, 3.5, 0.25, 1.5, 0.0]);
        let positive_rows = weights.iter().filter(|&&w| w > 0.0).count() as f64;
        let edf = 2.5;
        let inflation = positive_rows / (positive_rows - edf);
        let tweedie = |df: f64| estimate_tweedie_phi_from_eta(y.view(), &eta, weights.view(), 1.5, df);
        let dispersion = |response: ResponseFamily, df: f64| {
            estimate_dispersion_phi_from_eta(&response, &log, y.view(), &eta, weights.view(), df)
        };
        let tweedie_plugin = tweedie(0.0).expect("plug-in Tweedie phi");
        let tweedie_corrected = tweedie(edf).expect("residual-df Tweedie phi");
        assert!((tweedie_corrected / (inflation * tweedie_plugin) - 1.0).abs() < 1.0e-12);
        for response in [ResponseFamily::Gaussian, ResponseFamily::InverseGaussian] {
            let plugin = dispersion(response.clone(), 0.0).expect("plug-in dispersion");
            let corrected = dispersion(response.clone(), edf).expect("residual-df dispersion");
            assert!((corrected / (inflation * plugin) - 1.0).abs() < 1.0e-12);
        }
        for bad_df in [positive_rows, positive_rows + 1.0, -1.0, f64::NAN] {
            assert!(tweedie(bad_df).is_err(), "edf {bad_df} must be refused");
            assert!(dispersion(ResponseFamily::Gaussian, bad_df).is_err());
            assert!(
                estimate_gamma_shape_from_eta(&log, y.view(), &eta, weights.view(), bad_df).is_err()
            );
        }
    }

    /// The residual-df Gamma shape is the root of the Laplace-marginal score
    /// `Σ wᵢ [ln(wᵢ α) − ψ(wᵢ α) − tᵢ] − edf/(2α)`. It keeps the precision
    /// reading (`α̂(c·w) = α̂(w)/c`). At a small dispersion it reduces to
    /// `φ̂ = D/(n₊ − edf)`, the residual-df deviance scale.
    #[test]
    fn gamma_shape_divides_by_residual_degrees_of_freedom() {
        let log = InverseLink::Standard(StandardLink::Log);
        let y = Array1::from(vec![0.4, 1.3, 0.9, 2.2, 0.7, 1.6, 3.0]);
        let eta = Array1::from(vec![-0.3, 0.1, 0.0, 0.5, -0.2, 0.4, 0.6]);
        let weights = Array1::from(vec![0.5, 2.0, 1.0, 3.5, 0.25, 1.5, 0.0]);
        let edf = 2.5;
        let shape = estimate_gamma_shape_from_eta(&log, y.view(), &eta, weights.view(), edf)
            .expect("residual-df Gamma shape is finite");
        let marginal_score = |alpha: f64| -> f64 {
            (0..y.len())
                .filter(|&i| weights[i] > 0.0)
                .map(|i| {
                    let w = weights[i];
                    let t = gamma_shape_statistic(y[i], eta[i].exp());
                    w * ((w * alpha).ln() - digamma(w * alpha) - t)
                })
                .sum::<f64>()
                - edf / (2.0 * alpha)
        };
        assert!(marginal_score(shape * (1.0 - 1.0e-6)) > 0.0);
        assert!(marginal_score(shape * (1.0 + 1.0e-6)) < 0.0);
        let plugin = estimate_gamma_shape_from_eta(&log, y.view(), &eta, weights.view(), 0.0)
            .expect("plug-in Gamma shape is finite");
        assert!(shape < plugin, "spending edf must lower the shape (raise φ): {shape} vs {plugin}");
        for c in [1.0e-3_f64, 7.0, 1.0e3] {
            let scaled = weights.mapv(|w| c * w);
            let scaled_shape =
                estimate_gamma_shape_from_eta(&log, y.view(), &eta, scaled.view(), edf)
                    .expect("rescaled residual-df Gamma shape is finite");
            assert!((scaled_shape * c / shape - 1.0).abs() < 1.0e-12);
        }

        // Small dispersion, unit weights: ln a − ψ(a) = 1/(2a) + 1/(12a²) + …,
        // so the root is (n − edf)/D up to a relative 1/(6 a (1 − κ)) ≈ 1e-9.
        let y = Array1::from(vec![1.0 - 1.0e-4, 1.0 + 2.0e-4, 1.0 - 3.0e-4, 1.0 + 1.5e-4]);
        let eta = Array1::zeros(y.len());
        let weights = Array1::ones(y.len());
        let edf = 1.5;
        let deviance: f64 = y.iter().map(|&yi| 2.0 * gamma_shape_statistic(yi, 1.0)).sum();
        let shape = estimate_gamma_shape_from_eta(&log, y.view(), &eta, weights.view(), edf)
            .expect("small-dispersion residual-df Gamma shape is finite");
        let expected = (y.len() as f64 - edf) / deviance;
        assert!((shape / expected - 1.0).abs() < 1.0e-7, "{shape} vs (n − edf)/D = {expected}");
    }

    #[test]
    fn tweedie_pearson_statistic_preserves_representable_extreme_ratios() {
        for log_mean in [-600.0_f64, 600.0] {
            let y = Array1::zeros(1);
            let eta = Array1::from(vec![log_mean]);
            let weights = Array1::ones(1);
            let phi = estimate_tweedie_phi_from_eta(y.view(), &eta, weights.view(), 1.5, 0.0)
                .expect("Pearson ratio is representable despite overflowed squared terms");
            // With y=0, (y-mu)^2/mu^p = mu^(2-p).
            let expected = (0.5 * log_mean).exp();
            assert!((phi / expected - 1.0).abs() < 1.0e-12);
        }
    }
}

fn negbin_theta_score_and_info_from_means(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    means: &[f64],
    priorweights: ArrayView1<'_, f64>,
    theta: f64,
) -> Result<NegbinThetaScore, EstimationError> {
    if !(theta.is_finite() && theta > 0.0) {
        crate::bail_invalid_estim!("negative-binomial theta must be finite and positive");
    }
    let psi_theta = digamma(theta);
    let trigamma_theta = trigamma(theta);
    let ln_theta = theta.ln();
    let inv_theta = theta.recip();
    let rows: Vec<(f64, f64, f64)> = super::par_certified_rows(eta.len(), |i| {
        let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
        if wi == 0.0 {
            return Ok((0.0, 0.0, 0.0));
        }
        let yi = y[i];
        if !valid_count_response(yi) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "negative-binomial response",
                eta[i],
                yi,
            ));
        }
        let theta_plus_mu = theta + means[i];
        let theta_plus_y = theta + yi;
        let digamma_y = digamma(yi + theta);
        let ln_theta_plus_mu = theta_plus_mu.ln();
        let ratio = theta_plus_y / theta_plus_mu;
        let s = digamma_y - psi_theta + ln_theta + 1.0 - ln_theta_plus_mu - ratio;
        // Avoid forming `(theta + mu)^2`, which can overflow even when the
        // information term itself is representable.
        let info_row = -trigamma(yi + theta) + trigamma_theta - inv_theta + 2.0 / theta_plus_mu
            - (theta_plus_y / theta_plus_mu) / theta_plus_mu;
        let score = wi * s;
        let info = wi * info_row;
        let magnitude = wi
            * (digamma_y.abs()
                + psi_theta.abs()
                + ln_theta.abs()
                + 1.0
                + ln_theta_plus_mu.abs()
                + ratio.abs());
        if !(score.is_finite() && info.is_finite() && magnitude.is_finite()) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "negative-binomial theta score/information",
                eta[i],
                score,
            ));
        }
        Ok((score, info, magnitude))
    })?;
    let score_info: Vec<(f64, f64)> = rows.iter().map(|&(score, info, _)| (score, info)).collect();
    let (score, info) = certified_pairs_sum(&score_info)?;
    let magnitudes: Vec<(f64, f64)> =
        rows.iter().map(|&(_, _, magnitude)| (magnitude, 0.0)).collect();
    let (magnitude, _) = certified_pairs_sum(&magnitudes)?;
    // Each row's score is formed with fourteen rounded operations (three argument
    // sums, two digammas, two logarithms, the quotient, five additions and the
    // weight) over a cancelling sum of magnitude `magnitude`, and the rows are
    // reduced pairwise.
    let reduction_depth = 14 + (usize::BITS - eta.len().leading_zeros()) as usize;
    let band = gam_linalg::roundoff::accumulation_band(reduction_depth, magnitude);
    Ok(NegbinThetaScore { score, info, band })
}

pub(crate) fn negbin_theta_score_and_info(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    theta: f64,
) -> Result<NegbinThetaScore, EstimationError> {
    let means = certified_log_means(eta)?;
    negbin_theta_score_and_info_from_means(y, eta, &means, priorweights, theta)
}

/// The linear predictor's pull on the NB2 profile score: `∂(∂ℓ/∂θ)/∂η_i`
/// under the log link, `w_i μ_i (y_i − μ_i) / (θ + μ_i)²`, from the per-row
/// score `ψ(y+θ) − ψ(θ) + ln θ + 1 − ln(θ+μ) − (θ+y)/(θ+μ)` whose
/// `μ`-derivative is `(y − μ)/(θ + μ)²` and `dμ/dη = μ`.
pub(crate) fn negbin_theta_score_eta_gradient(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    theta: f64,
) -> Result<Array1<f64>, EstimationError> {
    if !(theta.is_finite() && theta > 0.0) {
        crate::bail_invalid_estim!("negative-binomial theta must be finite and positive");
    }
    let means = certified_log_means(eta)?;
    let rows = super::par_certified_rows(eta.len(), |i| {
        let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
        if wi == 0.0 {
            return Ok(0.0);
        }
        let yi = y[i];
        if !valid_count_response(yi) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "negative-binomial response",
                eta[i],
                yi,
            ));
        }
        let theta_plus_mu = theta + means[i];
        // Two bounded ratios instead of `(θ + μ)²`, which can overflow when
        // the derivative itself is representable.
        let pull = wi * (means[i] / theta_plus_mu) * ((yi - means[i]) / theta_plus_mu);
        if pull.is_finite() {
            Ok(pull)
        } else {
            Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "negative-binomial theta-score eta derivative",
                eta[i],
                pull,
            ))
        }
    })?;
    Ok(Array1::from_vec(rows))
}

/// Profile the NB2 theta: the smallest representable theta at which the profile
/// score is not resolvably positive.
///
/// `∂ℓ/∂θ` is positive below the maximizer. For overdispersed data it changes
/// sign at a finite root; for equidispersed or underdispersed data it decays to
/// zero without changing sign as θ grows, the Poisson limit. Either way the
/// estimate is where the score stops being positive beyond its own rounding band:
/// at a root that is the root to the arithmetic's resolution, and in the Poisson
/// limit it is the θ past which the data cannot distinguish NB2 from its limit.
/// The bracket is found by doubling or halving outward from a data-scale seed,
/// and the only way out of either walk is the representable range itself.
pub(crate) fn estimate_negbin_theta_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<f64, EstimationError> {
    let means = certified_log_means(eta)?;
    let seed_rows: Vec<(f64, f64)> = super::par_certified_rows(eta.len(), |i| {
        let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
        if wi == 0.0 {
            return Ok((0.0, 0.0));
        }
        if !valid_count_response(y[i]) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "negative-binomial response",
                eta[i],
                y[i],
            ));
        }
        let resid = y[i] - means[i];
        let pearson = wi * resid * resid / means[i];
        let weighted_mu = wi * means[i];
        if !(pearson.is_finite() && pearson >= 0.0 && weighted_mu.is_finite()) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "negative-binomial seed statistic",
                eta[i],
                pearson,
            ));
        }
        Ok((weighted_mu, pearson))
    })?;
    let (wmu, wpearson) = certified_pairs_sum(&seed_rows)?;
    let total_weight = priorweights.iter().try_fold(0.0, |sum, &w| {
        if w.is_finite() && w >= 0.0 {
            let next = sum + w;
            if next.is_finite() { Ok(next) } else { Err(()) }
        } else {
            Err(())
        }
    });
    let total_weight = total_weight.map_err(|_| {
        EstimationError::InvalidInput(
            "negative-binomial total prior weight is invalid or unrepresentable".to_string(),
        )
    })?;
    if !(total_weight > 0.0) {
        crate::bail_invalid_estim!("negative-binomial profiling requires positive total weight");
    }
    let mu_bar = wmu / total_weight;
    let pearson_ratio = wpearson / total_weight;
    // The method-of-moments θ when the Pearson statistic shows overdispersion,
    // otherwise the mean count, which is the data's own scale for θ. This is only
    // where the outward walk starts; it is never returned as an estimate.
    let moment = mu_bar / (pearson_ratio - 1.0);
    let seed = if pearson_ratio > 1.0 && moment.is_finite() && moment > 0.0 {
        moment
    } else {
        mu_bar
    };
    if !(seed.is_finite() && seed > 0.0) {
        crate::bail_invalid_estim!(
            "negative-binomial theta seed is not representable (mean count {mu_bar:?}, Pearson ratio {pearson_ratio:?})"
        );
    }
    let profile =
        |theta: f64| negbin_theta_score_and_info_from_means(y, eta, &means, priorweights, theta);

    let mut lo;
    let mut hi;
    if profile(seed)?.resolvably_positive() {
        lo = seed;
        hi = 2.0 * seed;
        loop {
            if !hi.is_finite() {
                crate::bail_invalid_estim!(
                    "negative-binomial theta profile score stays resolvably positive past the representable range (seed {seed:?})"
                );
            }
            if !profile(hi)?.resolvably_positive() {
                break;
            }
            lo = hi;
            hi *= 2.0;
        }
    } else {
        hi = seed;
        lo = 0.5 * seed;
        loop {
            if !(lo > 0.0) {
                crate::bail_invalid_estim!(
                    "negative-binomial theta MLE lies below the representable range: the profile score is not resolvably positive at any positive theta below {seed:?}"
                );
            }
            if profile(lo)?.resolvably_positive() {
                break;
            }
            hi = lo;
            lo *= 0.5;
        }
    }
    // Safeguarded Newton inside `[lo, hi]`. A Newton step is taken only when it
    // lands strictly inside the bracket and at most halves the previous step;
    // otherwise the bracket is bisected. Steps therefore shrink geometrically or
    // the bracket halves, and the loop ends when no representable theta lies
    // strictly inside the bracket or the step no longer moves theta.
    let mut theta = hi;
    let mut previous_step = hi - lo;
    loop {
        let at = profile(theta)?;
        if at.resolvably_positive() {
            lo = theta;
        } else {
            hi = theta;
        }
        let bisection = lo + 0.5 * (hi - lo);
        let newton = theta + at.score / at.info;
        let next = if at.info > 0.0
            && newton > lo
            && newton < hi
            && (newton - theta).abs() <= 0.5 * previous_step.abs()
        {
            newton
        } else {
            bisection
        };
        if !(next > lo && next < hi) || next == theta {
            break;
        }
        previous_step = next - theta;
        theta = next;
    }
    Ok(hi)
}

#[cfg(test)]
mod negbin_theta_profile_tests {
    use super::*;

    fn profile_at(y: &Array1<f64>, eta: &Array1<f64>, w: &Array1<f64>, theta: f64) -> NegbinThetaScore {
        negbin_theta_score_and_info(y.view(), eta, w.view(), theta)
            .expect("the NB2 profile is representable at a positive theta")
    }

    /// An extremely overdispersed sample has its theta MLE far below any fixed
    /// profiling rail: the estimate is where the score changes sign, not the
    /// endpoint of a declared interval (#2469).
    #[test]
    fn overdispersed_theta_is_the_root_of_the_profile_score() {
        let n = 1000;
        let mut y = Array1::<f64>::zeros(n);
        y[n - 1] = 1.0e6;
        let eta = Array1::<f64>::from_elem(n, 1000.0_f64.ln());
        let w = Array1::<f64>::ones(n);
        let theta = estimate_negbin_theta_from_eta(y.view(), &eta, w.view())
            .expect("an overdispersed sample has a finite theta MLE");
        assert!(theta.is_finite() && theta > 0.0, "theta={theta:e}");
        let below = profile_at(&y, &eta, &w, 0.5 * theta);
        let above = profile_at(&y, &eta, &w, 2.0 * theta);
        assert!(
            below.score > 0.0 && above.score < 0.0,
            "the score must change sign across [theta/2, 2 theta] at theta={theta:e}: \
             below={below:?} above={above:?}"
        );
    }

    /// An underdispersed sample has no finite theta MLE: the profile score stays
    /// positive and decays toward zero, the Poisson limit. The estimate is the
    /// smallest theta at which that score is no longer resolvably positive, so the
    /// score there is inside its band and above it at half that theta (#2469).
    #[test]
    fn underdispersed_theta_is_where_the_score_stops_being_resolvable() {
        let y = Array1::from(vec![2.0, 3.0, 4.0, 3.0, 2.0, 4.0, 3.0, 3.0]);
        let eta = Array1::<f64>::from_elem(y.len(), 3.0_f64.ln());
        let w = Array1::<f64>::ones(y.len());
        let theta = estimate_negbin_theta_from_eta(y.view(), &eta, w.view())
            .expect("the Poisson limit is represented by a finite theta");
        assert!(theta.is_finite() && theta > 0.0, "theta={theta:e}");
        let at = profile_at(&y, &eta, &w, theta);
        let half = profile_at(&y, &eta, &w, 0.5 * theta);
        assert!(
            !at.resolvably_positive() && half.resolvably_positive(),
            "theta={theta:e} must be the resolution limit of the profile score: at={at:?} half={half:?}"
        );
    }
}
