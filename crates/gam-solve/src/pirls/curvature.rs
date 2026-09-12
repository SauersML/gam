//! Curvature primitives: the variance-function jet, observed-information
//! Hessian weights, and the weight-family / weight-link classification used to
//! choose between Fisher and observed curvature per family.

use super::*;

pub struct VarianceJet {
    pub v: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
}

impl VarianceJet {
    /// Bernoulli / binomial variance V(μ) = μ(1−μ).
    #[inline]
    pub fn bernoulli(mu: f64) -> Self {
        Self::bernoulli_with_complement(mu, 1.0 - mu)
    }

    /// Bernoulli variance jet from the (μ, 1−μ) PAIR (#2273).
    ///
    /// `1.0 - mu` is a hard zero once `mu` rounds to exactly `1.0`, which a
    /// bounded inverse link reaches far inside its tail — cloglog at `η ≈ 3.62`,
    /// probit at `η ≈ 8.29` — while the true complement is still a perfectly
    /// representable `~1e-18`. `V = μ(1−μ)` then collapses to `0` and every
    /// observed-information quantity divides by it, so a row whose Fisher weight
    /// is a healthy `1e-14` is refused as unrepresentable instead of evaluated.
    /// `crate::mixture_link::inverse_link_complement_for_inverse_link` recovers
    /// the complement from `η` without cancellation; this constructor is how it
    /// reaches the variance.
    ///
    /// Only `v` changes: `v1 = 1 − 2μ` needs no complement (it is `−1` to full
    /// precision wherever `μ` has saturated, from either expression), and
    /// rewriting it would move existing well-conditioned fits by an ulp for no
    /// accuracy gained.
    #[inline]
    pub fn bernoulli_with_complement(mu: f64, one_minus_mu: f64) -> Self {
        Self {
            v: mu * one_minus_mu,
            v1: 1.0 - 2.0 * mu,
            v2: -2.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Poisson variance V(μ) = μ.
    #[inline]
    pub fn poisson(mu: f64) -> Self {
        Self {
            v: mu,
            v1: 1.0,
            v2: 0.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Gamma variance V(μ) = μ².
    #[inline]
    pub fn gamma(mu: f64) -> Self {
        Self {
            v: mu * mu,
            v1: 2.0 * mu,
            v2: 2.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Tweedie variance V(μ) = μ^p.
    #[inline]
    pub fn tweedie(mu: f64, p: f64) -> Self {
        Self {
            v: mu.powf(p),
            v1: p * mu.powf(p - 1.0),
            v2: p * (p - 1.0) * mu.powf(p - 2.0),
            v3: p * (p - 1.0) * (p - 2.0) * mu.powf(p - 3.0),
            v4: p * (p - 1.0) * (p - 2.0) * (p - 3.0) * mu.powf(p - 4.0),
        }
    }

    /// Negative-binomial variance V(μ) = μ + μ² / theta.
    #[inline]
    pub fn negative_binomial(mu: f64, theta: f64) -> Self {
        let inv_theta = if valid_negbin_theta(theta) {
            1.0 / theta
        } else {
            f64::NAN
        };
        Self {
            v: mu + mu * mu * inv_theta,
            v1: 1.0 + 2.0 * mu * inv_theta,
            v2: 2.0 * inv_theta,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Gaussian (identity) variance V(μ) = 1.
    #[inline]
    pub fn gaussian() -> Self {
        Self {
            v: 1.0,
            v1: 0.0,
            v2: 0.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Binomial(n, p) variance V(p) = p(1−p), identical to Bernoulli.
    ///
    /// The trial count `n` enters as a prior-weight multiplier, not through
    /// the variance function itself.
    #[inline]
    pub fn binomial_n(mu: f64, one_minus_mu: f64) -> Self {
        // V(μ) = μ(1−μ), same jet as Bernoulli
        Self::bernoulli_with_complement(mu, one_minus_mu)
    }

    /// Beta-regression variance V(μ) = μ(1−μ)/(1+φ).
    #[inline]
    pub fn beta(mu: f64, one_minus_mu: f64, phi: f64) -> Self {
        let scale = 1.0 / (1.0 + phi);
        let base = Self::bernoulli_with_complement(mu, one_minus_mu);
        Self {
            v: base.v * scale,
            v1: base.v1 * scale,
            v2: base.v2 * scale,
            v3: 0.0,
            v4: 0.0,
        }
    }
}

/// Certify and return the exact statistical `(W, dW/deta, d2W/deta2)` surface.
/// Positive-definiteness stabilization belongs to the assembled matrix/ridge
/// layer; changing individual row weights would change the likelihood Hessian.
pub fn exact_hessian_surface_arrays(
    hessian_weights: gam_linalg::matrix::SignedWeightsView<'_>,
    c_array: &Array1<f64>,
    d_array: &Array1<f64>,
    eta: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
    let hessian_view = hessian_weights.view();
    let n = hessian_view.len();
    if c_array.len() != n || d_array.len() != n || eta.len() != n {
        crate::bail_invalid_estim!(
            "exact Hessian surface length mismatch: W={}, c={}, d={}, eta={}",
            n,
            c_array.len(),
            d_array.len(),
            eta.len()
        );
    }
    for i in 0..n {
        for (quantity, value) in [
            ("observed Hessian weight", hessian_view[i]),
            ("observed Hessian dW/deta", c_array[i]),
            ("observed Hessian d2W/deta2", d_array[i]),
        ] {
            if !value.is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity,
                    eta: eta[i],
                    value,
                });
            }
        }
    }
    Ok((
        hessian_view.to_owned(),
        c_array.to_owned(),
        d_array.to_owned(),
    ))
}

#[inline]
pub(crate) fn fixed_glm_dispersion(
    likelihood: &GlmLikelihoodSpec,
) -> Result<f64, EstimationError> {
    use gam_problem::ResolvedLikelihoodScale as Scale;

    let scale = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    match scale {
        // The profiled Gaussian working geometry is intentionally scale-free.
        Scale::ProfiledGaussian | Scale::Unit | Scale::NegativeBinomial { .. } => Ok(1.0),
        Scale::FixedGaussian { phi } | Scale::Tweedie { phi, .. } => Ok(phi.value()),
        Scale::Gamma { .. } => scale
            .gamma_phi()
            .map_err(|error| EstimationError::InvalidInput(error.to_string())),
        // Beta precision is already inside its variance/Fisher geometry; it is
        // not an exponential-dispersion phi multiplier.
        Scale::BetaPrecision { .. } => Ok(1.0),
        Scale::Unspecified => Err(EstimationError::InvalidInput(
            "family has no fixed GLM dispersion".to_string(),
        )),
    }
}

/// The constant dispersion factor `k` the inner IRLS working weight carries but
/// the (post-#2126/#2131 *unscaled*) `calculate_deviance` does **not**.
///
/// The inner P-IRLS builds its gradient and Hessian from the working weight,
/// which for the dispersion families is `prior · k` (Gamma) or `prior · … / φ`
/// (Tweedie / fixed-φ Gaussian) — i.e. the Newton/LM step is computed for the
/// penalized objective `k·D(β) + βᵀSβ`, whose argmin is the true penalized MLE
/// (`max ℓ − ½βᵀSβ`, since the Gamma/Tweedie log-likelihood is `−½·k·D`). But
/// `loglik_deviance` returns the *reported* deviance `D` (φ ≡ 1), so the LM
/// gain-ratio would compare the *actual* reduction in `D + βᵀSβ` against a
/// *predicted* reduction built for `k·D + βᵀSβ`. When `k ≠ 1` those two
/// objectives have different minima; at a heavily-penalized ρ every step that
/// lowers `k·D + penalty` *raises* `D + penalty`, so no step is ever accepted,
/// the solve freezes with a non-zero (k-scaled) residual gradient, and the outer
/// REML sees a non-finite cost for every seed (issue #2128). Scaling the
/// gain-ratio objective's deviance by `k` realigns it with the step, the
/// gradient certificate, and the outer objective (which already carries the same
/// `k`; see `calculate_loglikelihood_omitting_constants_from_eta`).
///
///  * Gamma:  weight `prior·shape` ⇒ `k = shape` (`= 1/φ`).
///  * Tweedie: weight `prior·μ^{2−p}/φ` ⇒ `k = 1/φ` (the μ-power is already in
///    the deviance's η-derivative, so only the constant `1/φ` is missing from D).
///  * Gaussian with an explicitly fixed `φ ≠ 1`: weight `prior/φ` ⇒ `k = 1/φ`.
///  * Every other family (Poisson, Binomial, negative-binomial, Beta, profiled
///    Gaussian): the working weight carries no constant dispersion factor absent
///    from D, so `k = 1` and the objective is already self-consistent.
#[inline]
pub(crate) fn penalized_objective_deviance_scale(
    likelihood: &GlmLikelihoodSpec,
) -> Result<f64, EstimationError> {
    let resolved = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let k = match likelihood.spec.response {
        ResponseFamily::Gamma => resolved
            .gamma_shape()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        ResponseFamily::Tweedie { .. } => {
            let phi = resolved
                .tweedie_phi()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            1.0 / phi
        }
        ResponseFamily::Gaussian => match resolved {
            gam_problem::ResolvedLikelihoodScale::ProfiledGaussian => 1.0,
            gam_problem::ResolvedLikelihoodScale::FixedGaussian { phi } => 1.0 / phi.value(),
            _ => {
                return Err(EstimationError::InvalidInput(
                    "resolved Gaussian scale has the wrong family variant".to_string(),
                ));
            }
        },
        _ => 1.0,
    };
    if k.is_finite() && k > 0.0 {
        Ok(k)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "penalized objective deviance scale is not representable: {k:?}"
        )))
    }
}

#[inline]
pub fn weight_family_for_glm_likelihood(
    likelihood: &GlmLikelihoodSpec,
) -> Result<WeightFamily, EstimationError> {
    let resolved = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => Ok(WeightFamily::Gaussian),
        ResponseFamily::Poisson => Ok(WeightFamily::Poisson),
        ResponseFamily::Tweedie { p } => Ok(WeightFamily::Tweedie { p: *p }),
        ResponseFamily::NegativeBinomial { .. } => Ok(WeightFamily::NegativeBinomial {
            theta: resolved
                .negative_binomial_theta()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        }),
        ResponseFamily::Beta { .. } => Ok(WeightFamily::Beta {
            phi: resolved
                .beta_precision()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        }),
        ResponseFamily::Gamma => Ok(WeightFamily::Gamma),
        ResponseFamily::Binomial => Ok(WeightFamily::Binomial),
        ResponseFamily::RoystonParmar => Err(EstimationError::InvalidInput(
            "Royston-Parmar is not a GLM weight family".to_string(),
        )),
    }
}

#[inline]
pub(crate) fn weight_link_for_inverse_link(inverse_link: &InverseLink) -> WeightLink {
    match inverse_link {
        InverseLink::Standard(StandardLink::Log) => WeightLink::Log,
        InverseLink::Standard(StandardLink::Identity)
        | InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
        | InverseLink::Standard(StandardLink::LogLog)
        | InverseLink::Standard(StandardLink::Cauchit)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => WeightLink::Other,
    }
}

#[inline]
pub(crate) fn supports_observed_hessian_curvature_for_likelihood(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
) -> bool {
    let spec = &likelihood.spec;
    if matches!(spec.response, ResponseFamily::NegativeBinomial { .. }) {
        return matches!(inverse_link, InverseLink::Standard(StandardLink::Log));
    }
    if matches!(spec.response, ResponseFamily::Gamma) {
        return true;
    }
    if !matches!(spec.response, ResponseFamily::Binomial) {
        return false;
    }
    matches!(
        spec.link,
        InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::CLogLog)
            | InverseLink::Standard(StandardLink::LogLog)
            | InverseLink::Standard(StandardLink::Cauchit)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    )
}

/// Compute vectorised observed-information curvature arrays (w_obs, c_obs, d_obs)
/// for the Hessian surface at the mode.
///
/// This function is the primary entry point for obtaining the observed weights
/// that flow into the outer REML/LAML Hessian H_obs = X' W_obs X + S. The
/// observed corrections include residual-dependent terms that vanish for
/// canonical links but are nonzero for probit, cloglog, SAS, mixture, Gamma-log,
/// and other flexible links.
///
/// The output arrays are:
/// - `hessian_weights`: W_obs per observation (exact; matrix ridge applied separately).
/// - `hessian_c`: c_obs = dW_obs/deta per observation (for outer gradient C[v]).
/// - `hessian_d`: d_obs = d^2W_obs/deta^2 per observation (for outer Hessian Q[v_k,v_l]).
///
/// See `observed_weight_noncanonical` for the per-observation formulas and
/// response.md Section 3 for the mathematical justification of why observed
/// (not Fisher) information is required.
pub(crate) fn compute_observed_hessian_curvature_arrays_into(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    y: ArrayView1<'_, f64>,
    fisher_weights: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    hessian_weights: &mut Array1<f64>,
    hessian_c: &mut Array1<f64>,
    hessian_d: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    assert!(supports_observed_hessian_curvature_for_likelihood(
        likelihood,
        inverse_link
    ));
    let n = eta.len();
    if hessian_weights.len() != n {
        *hessian_weights = Array1::<f64>::zeros(n);
    }
    if hessian_c.len() != n {
        *hessian_c = Array1::<f64>::zeros(n);
    }
    if hessian_d.len() != n {
        *hessian_d = Array1::<f64>::zeros(n);
    }

    let weight_family = weight_family_for_glm_likelihood(likelihood)?;
    let weight_link = weight_link_for_inverse_link(inverse_link);
    let phi = fixed_glm_dispersion(likelihood)?;

    // Compute into an indexed certificate buffer before touching caller-owned
    // arrays.  Parallel evaluation stays O(n), while the ordered scan below
    // deterministically reports the smallest bad row and guarantees atomic
    // output on error.
    let certified: Vec<Result<(f64, f64, f64), EstimationError>> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(f64, f64, f64), EstimationError> {
            let eta_used = eta[i];
            if !(priorweights[i].is_finite() && priorweights[i] >= 0.0) {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "prior weight",
                    eta: eta_used,
                    value: priorweights[i],
                });
            }
            if priorweights[i] == 0.0 {
                return Ok((0.0, 0.0, 0.0));
            }
            // Every jet and every variance carrier is evaluated at this exact
            // eta.  A non-representable tail is refused below rather than
            // projected onto a different Hessian surface.
            let jet =
                crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta_used)?;
            let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                inverse_link,
                eta_used,
            )?;
            let one_minus_mu = crate::mixture_link::inverse_link_complement_for_inverse_link(
                inverse_link,
                eta_used,
                jet.mu,
            );
            let (w_obs, c_obs, d_obs) = observed_weight_dispatch(
                weight_family,
                weight_link,
                y[i],
                jet.mu,
                one_minus_mu,
                phi,
                priorweights[i],
                jet,
                h4,
            );
            // A *finite* but non-positive observed weight is NOT a failure: the
            // observed information `W_obs = W_Fisher - (y-μ)·B` legitimately goes
            // indefinite on individual rows for a non-canonical link (probit,
            // cloglog, SAS, and — critically for #1598 — a blended/mixture link)
            // whenever a residual flips the correction's sign.  Signed row
            // weights are assembled exactly; the matrix-level ridge handles a
            // non-PD aggregate without modifying these statistical carriers.
            if !w_obs.is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "observed Hessian weight",
                    eta: eta_used,
                    value: w_obs,
                });
            }
            if !c_obs.is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "observed Hessian dW/deta",
                    eta: eta_used,
                    value: c_obs,
                });
            }
            if !d_obs.is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "observed Hessian d2W/deta2",
                    eta: eta_used,
                    value: d_obs,
                });
            }
            Ok((w_obs, c_obs, d_obs))
        })
        .collect();
    let certified: Vec<(f64, f64, f64)> = certified.into_iter().collect::<Result<_, _>>()?;
    for (i, &(w, c, d)) in certified.iter().enumerate() {
        hessian_weights[i] = w;
        hessian_c[i] = c;
        hessian_d[i] = d;
    }
    // The caller supplies Fisher weights for the observed-vs-Fisher contract;
    // certify that this parallel surface has the same row cardinality.
    if fisher_weights.len() != n {
        crate::bail_invalid_estim!(
            "observed Hessian Fisher-weight length mismatch: expected {n}, got {}",
            fisher_weights.len()
        );
    }
    Ok(())
}

pub(crate) fn compute_observed_hessian_curvature_arrays(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    y: ArrayView1<'_, f64>,
    fisher_weights: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
    let n = eta.len();
    let mut hessian_weights = Array1::<f64>::zeros(n);
    let mut hessian_c = Array1::<f64>::zeros(n);
    let mut hessian_d = Array1::<f64>::zeros(n);
    compute_observed_hessian_curvature_arrays_into(
        likelihood,
        inverse_link,
        eta,
        y,
        fisher_weights,
        priorweights,
        &mut hessian_weights,
        &mut hessian_c,
        &mut hessian_d,
    )?;
    Ok((hessian_weights, hessian_c, hessian_d))
}

/// Per-observation observed-information weights and their first two
/// eta-derivatives for a general exponential-dispersion family with a
/// noncanonical link.
///
/// The observed weight differs from the Fisher (expected) weight by a
/// residual-dependent correction (see response.md Section 3):
///
///   W_obs = W_Fisher - (y - mu) * B
///   B = (h'' V - h'^2 V') / (phi V^2)
///
///   c_obs = c_Fisher + h' * B - (y - mu) * B_eta
///   d_obs = d_Fisher + h'' * B + 2*h' * B_eta - (y - mu) * B_etaeta
///
/// For canonical links (for example logit-Binomial and log-Poisson), B = 0
/// so observed = Fisher and no correction is needed.
///
/// These observed quantities are required for:
/// 1. The outer REML/LAML Hessian H_obs = X' W_obs X + S (log|H| term).
/// 2. The outer gradient's C\[v\] correction (uses c_obs).
/// 3. The outer Hessian's Q[v_k, v_l] correction (uses d_obs).
///
/// Using Fisher weights in the outer REML would yield a PQL-type surrogate
/// rather than the exact Laplace approximation.
///
/// # Arguments
/// * `resid` -- `y - mu`, formed by the caller so a saturated Bernoulli row can
///   supply the link's tail complement instead of a cancelled zero (#2273)
/// * `h1`...`h4` -- inverse-link derivatives h'(eta) ... h''''(eta)
/// * `vj`  -- variance-function jet (V, V', V'', V''') evaluated at mu
/// * `phi` -- dispersion parameter (1.0 for Bernoulli/Poisson)
/// * `pw`  -- prior weight for this observation
///
/// # Returns
/// `(w_obs, c_obs, d_obs)` -- the observed weight and its first two
/// eta-derivatives, all pre-multiplied by `pw`.
#[inline]
pub fn observed_weight_noncanonical(
    resid: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> (f64, f64, f64) {
    // The whole tower is `T = h₁/(φV)` and its η-derivatives: `B = T₁`,
    // `B_η = T₂`, `B_ηη = T₃`, and `W_F = h₁·T₀` with its derivatives by the
    // product rule. `weight_ratio_tower` builds `T` by Leibniz on `T·Q = h₁`,
    // dividing by `Q = φV` once per order.
    //
    // #2273 — that is the whole point. The closed forms this replaced carried
    // `φV²`, `φV³` and `φV⁴` denominators, and for a saturating Bernoulli row
    // those UNDERFLOW while every quantity that matters stays representable: at
    // `η = 5.26` a cloglog row has `V = 3.6e-84`, so `V⁴ = 1.7e-334` is zero in
    // f64 and `B_ηη` came out NaN, refusing the row as "not representable" —
    // when the correct `B_ηη` is a perfectly ordinary number. The tower never
    // forms a power of `V`, so the same row evaluates. It is also the identical
    // derivation `e_obs_from_jets` uses one order higher, so the third and
    // fourth derivatives of one object are now built by one recurrence instead
    // of two independently-maintained algebraic expansions.
    //
    // `h5` is not needed for orders <= 3, so `T₄` is not read.
    let t = weight_ratio_tower(h1, h2, h3, h4, 0.0, vj, phi);
    let (t0, t1, t2, t3) = (t[0], t[1], t[2], t[3]);

    // W_F = h₁·T and its derivatives by the product rule.
    let w_f = h1 * t0;
    let c_f = h2 * t0 + h1 * t1;
    let d_f = h3 * t0 + 2.0 * h2 * t1 + h1 * t2;

    // W_obs = W_F − (y−μ)·T₁, differentiated twice with `(y−μ)' = −h₁`.
    //
    // `resid = y − μ` is supplied rather than formed here (#2273): for a
    // saturated Bernoulli row `y − μ` cancels to exactly zero while the true
    // residual is the link's representable tail complement, and only the caller
    // knows the link.
    let w_obs = w_f - resid * t1;
    let c_obs = c_f + h1 * t1 - resid * t2;
    let d_obs = d_f + h2 * t1 + 2.0 * h1 * t2 - resid * t3;

    (pw * w_obs, pw * c_obs, pw * d_obs)
}

/// `T = h₁/(φV)` and its first four η-derivatives, by Leibniz on `T·Q = h₁`
/// with `Q = φ·V(μ(η))`.
///
/// One recurrence, dividing by `Q` once per order and never forming a power of
/// `V`. Every noncanonical observed-information quantity is a polynomial in this
/// tower and the inverse-link jet, so this is the single place the algebra
/// lives.
///
/// `T₄` requires `h5`; callers that only need orders up to 3 may pass any value
/// for it and ignore the last entry.
#[inline]
pub fn weight_ratio_tower(
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    h5: f64,
    vj: VarianceJet,
    phi: f64,
) -> [f64; 5] {
    let VarianceJet { v, v1, v2, v3, v4 } = vj;
    let q = phi * v;
    let h1_sq = h1 * h1;
    let h1_cu = h1_sq * h1;
    let h1_qu = h1_sq * h1_sq;

    // Q = phi*V and its eta-derivatives (chain rule d/deta = h1 * d/dmu on V):
    //   Q'    = phi V1 h1
    //   Q''   = phi (V1 h2 + V2 h1^2)
    //   Q'''  = phi (V1 h3 + 3 V2 h1 h2 + V3 h1^3)
    //   Q'''' = phi (V1 h4 + 4 V2 h1 h3 + 3 V2 h2^2 + 6 V3 h1^2 h2 + V4 h1^4)
    let q1 = phi * v1 * h1;
    let q2 = phi * (v1 * h2 + v2 * h1_sq);
    let q3 = phi * (v1 * h3 + 3.0 * v2 * h1 * h2 + v3 * h1_cu);
    let q4 = phi
        * (v1 * h4 + 4.0 * v2 * h1 * h3 + 3.0 * v2 * h2 * h2 + 6.0 * v3 * h1_sq * h2 + v4 * h1_qu);

    //   T'    = (h2 - T Q')/Q
    //   T''   = (h3 - 2 T' Q' - T Q'')/Q
    //   T'''  = (h4 - 3 T'' Q' - 3 T' Q'' - T Q''')/Q
    //   T'''' = (h5 - 4 T''' Q' - 6 T'' Q'' - 4 T' Q''' - T Q'''')/Q
    let t0 = h1 / q;
    let t1 = (h2 - t0 * q1) / q;
    let t2 = (h3 - 2.0 * t1 * q1 - t0 * q2) / q;
    let t3 = (h4 - 3.0 * t2 * q1 - 3.0 * t1 * q2 - t0 * q3) / q;
    let t4 = (h5 - 4.0 * t3 * q1 - 6.0 * t2 * q2 - 4.0 * t1 * q3 - t0 * q4) / q;
    [t0, t1, t2, t3, t4]
}

/// Per-observation third η-derivative of the observed-information weight,
/// `e_obs := ∂³W_obs/∂η³`, for a general exponential-dispersion family with
/// any (canonical or non-canonical) link.
///
/// Closed-form derivation:
///   Define `T(η) := h₁(η)/(φ V(μ(η)))`. Then
///   * Fisher weight `W_F = h₁ · T`
///   * Observed correction `B = T'`, so `B_η = T''`, `B_ηη = T'''`,
///     `B_ηηη = T''''`
///   * `W_obs = W_F − (y−μ) · T'`
///
/// Differentiating three times:
///   `∂³W_obs/∂η³ = W_F''' + h₃·T' + 3 h₂·T'' + 3 h₁·T''' − (y−μ)·T''''`
///
/// `T` is computed via Leibniz on `T·Q = h₁` with `Q = φV`; `W_F` via
/// Leibniz on `W_F·1 = h₁·T` (product rule).
///
/// All inverse-link derivatives `h₁..h₅` and variance-function derivatives
/// `V..V₄` are required as inputs. Caller supplies them.
///
/// Returns `pw * e_obs` (pre-multiplied by the prior weight) so the result
/// scales identically to `(w_obs, c_obs, d_obs)` from
/// `observed_weight_noncanonical`.
#[inline]
pub fn e_obs_from_jets(
    resid: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    h5: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> f64 {
    // One recurrence, shared with `observed_weight_noncanonical` (#2273): the
    // two used to expand the same `T`-tower independently, and the lower-order
    // one expanded it into `φV²/φV³/φV⁴` closed forms that underflow on a
    // saturated Bernoulli row while this one does not.
    let t = weight_ratio_tower(h1, h2, h3, h4, h5, vj, phi);
    let (t0, t1, t2, t3, t4) = (t[0], t[1], t[2], t[3], t[4]);

    // Fisher weight derivatives via product rule on W_F = h₁·T.
    //   W_F^(0) = h₁ T
    //   W_F^(1) = h₁ T₁ + h₂ T
    //   W_F^(2) = h₁ T₂ + 2 h₂ T₁ + h₃ T
    //   W_F^(3) = h₁ T₃ + 3 h₂ T₂ + 3 h₃ T₁ + h₄ T
    let w_f3 = h1 * t3 + 3.0 * h2 * t2 + 3.0 * h3 * t1 + h4 * t0;

    // Observed third derivative: differentiate W_obs = W_F − (y−μ)·T₁ thrice.
    // (resid)' = −h₁, so iterating product rule yields
    //   ∂³((y−μ)·T₁)/∂η³ = −h₃·T₁ − 3 h₂·T₂ − 3 h₁·T₃ + (y−μ)·T₄
    let e_obs = w_f3 + h3 * t1 + 3.0 * h2 * t2 + 3.0 * h1 * t3 - resid * t4;

    pw * e_obs
}

// Closed-form observed-information weights for the log-link Gamma and
// negative-binomial pairs: algebraically identical to the generic tower, but
// free of its large-η intermediates (each item states its algebra).

/// Gamma family with log link: `V(μ)=μ²`, `μ=exp(η)`.
///
/// For a Gamma exponential-dispersion model the negative log-likelihood
/// second derivative with respect to `η` is exactly `y / (φ μ)`.  The generic
/// observed-information formula computes the same value as
/// `W_Fisher - (y - μ)·B`; with `V=μ²` and log-link derivatives this subtracts
/// two `1/φ`-scale terms to leave the small positive `y/(φμ)` tail, and its
/// intermediate `V²`/`V³` products overflow for large trial `η`.  This
/// closed form is algebraically identical but cancellation- and overflow-free.
///
/// ```text
/// w_obs =  ω y / (φ μ)
/// c_obs = -ω y / (φ μ)
/// d_obs =  ω y / (φ μ)
/// ```
#[inline]
pub fn observed_weight_gamma_log(y: f64, mu: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let w = (pw / phi) * (y / mu);
    (w, -w, w)
}

/// NB2 observed information under the log link, evaluated through bounded
/// ratios.  With `r = theta/(theta+mu)` and `s = 1-r`,
/// `W_obs = prior (y+theta) r s`, `W' = W(r-s)`, and
/// `W'' = W((r-s)^2 - 2rs)`.
#[inline]
pub fn observed_weight_negative_binomial_log(
    y: f64,
    mu: f64,
    theta: f64,
    prior_weight: f64,
) -> (f64, f64, f64) {
    let r = if theta >= mu {
        1.0 / (1.0 + mu / theta)
    } else {
        let theta_over_mu = theta / mu;
        theta_over_mu / (1.0 + theta_over_mu)
    };
    let s = 1.0 - r;
    let w = prior_weight * (y + theta) * r * s;
    let c = w * (r - s);
    let d = w * ((r - s) * (r - s) - 2.0 * r * s);
    (w, c, d)
}

/// Family tag for the observed-information weight dispatch.
///
/// This is a simplified family tag that identifies the variance function,
/// independent of the link function. It is used by [`observed_weight_dispatch`]
/// to select closed-form weight specializations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFamily {
    Gaussian,
    Binomial,
    Poisson,
    Tweedie { p: f64 },
    NegativeBinomial { theta: f64 },
    Beta { phi: f64 },
    Gamma,
}

/// Link tag for the observed-information weight dispatch.
///
/// Identifies the link function for selecting closed-form weight
/// specializations in [`observed_weight_dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLink {
    Log,
    /// Any other link — falls back to the generic noncanonical formula.
    Other,
}

/// `one_minus_mu` is the cancellation-free complement of `mu` (#2273). It is a
/// required argument rather than a derived one because only the caller knows the
/// inverse link, and only the link can produce `1 − μ` without cancellation once
/// `μ` has saturated to exactly `1.0`; the families whose variance does not
/// involve the complement ignore it.
#[inline]
pub fn variance_jet_for_weight_family(
    family: WeightFamily,
    mu: f64,
    one_minus_mu: f64,
) -> VarianceJet {
    match family {
        WeightFamily::Gaussian => VarianceJet::gaussian(),
        WeightFamily::Binomial => VarianceJet::binomial_n(mu, one_minus_mu),
        WeightFamily::Poisson => VarianceJet::poisson(mu),
        WeightFamily::Tweedie { p } => VarianceJet::tweedie(mu, p),
        WeightFamily::NegativeBinomial { theta } => VarianceJet::negative_binomial(mu, theta),
        WeightFamily::Beta { phi } => VarianceJet::beta(mu, one_minus_mu, phi),
        WeightFamily::Gamma => VarianceJet::gamma(mu),
    }
}

/// Dispatch to closed-form observed-information weights for known family-link
/// combinations, falling back to the generic noncanonical formula.
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight.
///
/// For the `Binomial + Logit` case, `n_trials` is passed as `phi` (dispersion
/// slot is unused for binomial) and the prior weight controls the
/// observation-level scaling. For all other cases, `phi` is the dispersion
/// parameter.
///
/// `jet` and `h4` are the inverse-link derivatives used by the generic
/// noncanonical fallback path. They may be zero for the specialized paths.
/// `y − μ` formed from the (μ, 1−μ) pair for a two-point response (#2273).
///
/// For a saturated Bernoulli row `y − μ` cancels to exactly `0` at `y = 1`,
/// silently degrading the observed information back to Fisher on precisely the
/// rows where the two differ most. The pair carries the residual exactly:
/// `y = 1 ⇒ y − μ = 1 − μ`, `y = 0 ⇒ y − μ = −μ`. Grouped-binomial proportions
/// and every other family fall through to the ordinary difference, which does
/// not cancel for them.
#[inline]
pub fn bernoulli_pair_residual(family: WeightFamily, y: f64, mu: f64, one_minus_mu: f64) -> f64 {
    if matches!(family, WeightFamily::Binomial) {
        if y == 1.0 {
            return one_minus_mu;
        }
        if y == 0.0 {
            return -mu;
        }
    }
    y - mu
}

pub fn observed_weight_dispatch(
    family: WeightFamily,
    link: WeightLink,
    y: f64,
    mu: f64,
    one_minus_mu: f64,
    phi: f64,
    prior_weight: f64,
    jet: MixtureInverseLinkJet,
    h4: f64,
) -> (f64, f64, f64) {
    match (family, link) {
        (WeightFamily::Gamma, WeightLink::Log) => {
            observed_weight_gamma_log(y, mu, phi, prior_weight)
        }
        (WeightFamily::NegativeBinomial { theta }, WeightLink::Log) => {
            observed_weight_negative_binomial_log(y, mu, theta, prior_weight)
        }
        _ => {
            // Generic noncanonical path via the full variance-function jet.
            let vj = variance_jet_for_weight_family(family, mu, one_minus_mu);
            let resid = bernoulli_pair_residual(family, y, mu, one_minus_mu);
            observed_weight_noncanonical(
                resid,
                jet.d1,
                jet.d2,
                jet.d3,
                h4,
                vj,
                phi,
                prior_weight,
            )
        }
    }
}

#[derive(Clone)]
pub enum DirectionalWorkingCurvature {
    /// Directional derivative of the PIRLS curvature when the working
    /// curvature is diagonal in observation space:
    ///   W_τ = diag(w_τ).
    Diagonal(Array1<f64>),
}

pub fn directionalworking_curvature_from_c_array(
    c_array: &Array1<f64>,
    eta_direction: &Array1<f64>,
) -> DirectionalWorkingCurvature {
    DirectionalWorkingCurvature::Diagonal(c_array * eta_direction)
}
