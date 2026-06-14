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
    /// Lower floor on μ before evaluating power-law variance functions, so that
    /// `μ^(p−k)` derivatives stay finite as μ → 0 instead of producing inf/NaN.
    pub(crate) const VARIANCE_MU_FLOOR: f64 = 1e-10;

    /// Bernoulli / binomial variance V(μ) = μ(1−μ).
    #[inline]
    pub fn bernoulli(mu: f64) -> Self {
        Self {
            v: mu * (1.0 - mu),
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
        let mu = mu.max(Self::VARIANCE_MU_FLOOR);
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
        let mu = mu.max(Self::VARIANCE_MU_FLOOR);
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
    pub fn binomial_n(mu: f64) -> Self {
        // V(μ) = μ(1−μ), same jet as Bernoulli
        Self::bernoulli(mu)
    }

    /// Beta-regression variance V(μ) = μ(1−μ)/(1+φ).
    #[inline]
    pub fn beta(mu: f64, phi: f64) -> Self {
        let scale = 1.0 / (1.0 + phi.max(1e-12));
        let base = Self::bernoulli(mu);
        Self {
            v: base.v * scale,
            v1: base.v1 * scale,
            v2: base.v2 * scale,
            v3: 0.0,
            v4: 0.0,
        }
    }
}


pub(crate) const OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC: f64 = 1e-6;

pub(crate) const OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR: f64 = 1e-12;


/// Returns the per-row floor `max(fisher · 1e-6, 1e-12)` used by PIRLS to
/// stabilize the observed-information Hessian H = X' W X + S. Saturated
/// rows where W_obs ≤ floor were silently raised to `floor` when PIRLS
/// built the inner Hessian; outer REML/LAML derivatives must use the
/// **same** floored W to keep `H` and `dH/dψ` on one surface.
///
/// This is the single source of truth for the floor formula. Both the
/// inner solver (`solver_hessian_weights_into`) and the outer derivative
/// path (`outer_hessian_curvature_arrays`) route through this helper so
/// the inner-stabilized H and the outer dH/dψ cannot drift apart.
#[inline]
pub fn solver_hessian_weight_floor(fisher_weight: f64) -> f64 {
    (fisher_weight.max(0.0) * OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC)
        .max(OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR)
}


/// Build the (W, c, d) triple that matches PIRLS's stabilized H = X' W X + S.
///
/// PIRLS internally uses `W[i] = max(W_obs[i], floor(W_F[i]))` to keep H PD,
/// but `pirls_result.finalweights` stores the **unfloored** observed weights.
/// Reusing those directly in `∂H/∂ψ = X_τ' W X + … + X' diag(c · X_τ β̂) X`
/// produces an operator that disagrees with `H` at every saturated row — a
/// 5%-Frobenius bias that `tr(G_ε(H) · op)` amplifies by O(1/σ_min(H)),
/// driving the analytic gradient off by orders of magnitude.
///
/// This helper returns the floored W, plus c and d masked to zero wherever
/// the floor is active (so `∂W/∂η` is zero on the constant-floor branch).
pub fn outer_hessian_curvature_arrays(
    hessian_weights: crate::matrix::SignedWeightsView<'_>,
    fisher_weights: crate::matrix::PsdWeightsView<'_>,
    c_array: &Array1<f64>,
    d_array: &Array1<f64>,
    eta: &Array1<f64>,
    inverse_link: &InverseLink,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let hessian_view = hessian_weights.view();
    let fisher_view = fisher_weights.view();
    let n = hessian_view.len();
    let mut w_out = Array1::<f64>::zeros(n);
    let mut c_out = Array1::<f64>::zeros(n);
    let mut d_out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let floor = solver_hessian_weight_floor(fisher_view[i]);
        let w = hessian_view[i];
        let clamp_active = eta_clamp_active(inverse_link, eta[i]);
        let w_below_floor = !(w.is_finite() && w > floor);
        if w_below_floor {
            w_out[i] = floor;
            c_out[i] = 0.0;
            d_out[i] = 0.0;
        } else if clamp_active {
            w_out[i] = w;
            c_out[i] = 0.0;
            d_out[i] = 0.0;
        } else {
            w_out[i] = w;
            c_out[i] = c_array[i];
            d_out[i] = d_array[i];
        }
    }
    (w_out, c_out, d_out)
}


#[inline]
pub(crate) fn fixed_glm_dispersion(likelihood: &GlmLikelihoodSpec) -> f64 {
    likelihood.fixed_phi().unwrap_or(1.0)
}


#[inline]
pub fn weight_family_for_glm_likelihood(likelihood: &GlmLikelihoodSpec) -> WeightFamily {
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => WeightFamily::Gaussian,
        ResponseFamily::Poisson => WeightFamily::Poisson,
        ResponseFamily::Tweedie { p } => WeightFamily::Tweedie { p: *p },
        ResponseFamily::NegativeBinomial { theta, .. } => {
            WeightFamily::NegativeBinomial { theta: *theta }
        }
        ResponseFamily::Beta { phi } => WeightFamily::Beta { phi: *phi },
        ResponseFamily::Gamma => WeightFamily::Gamma,
        ResponseFamily::Binomial => WeightFamily::Binomial,
        ResponseFamily::RoystonParmar => WeightFamily::Gaussian,
    }
}


#[inline]
pub(crate) fn weight_link_for_inverse_link(inverse_link: &InverseLink) -> WeightLink {
    match inverse_link {
        InverseLink::Standard(StandardLink::Identity) => WeightLink::Identity,
        InverseLink::Standard(StandardLink::Log) => WeightLink::Log,
        InverseLink::Standard(StandardLink::Logit) => WeightLink::Logit,
        InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
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
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    )
}


#[inline]
pub(crate) fn eta_for_observed_hessian_jet(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        // Why: canonical links keep V(mu) representable across the full f64 eta range; only guard against inf.
        InverseLink::Standard(StandardLink::Logit | StandardLink::Log) => {
            eta.clamp(-ETA_CLAMP, ETA_CLAMP)
        }
        InverseLink::Standard(StandardLink::Identity) => eta,
        // Why: probit mu=Phi(eta) saturates to 1.0 in f64 by |eta|~8.3; +/-6 keeps V=mu(1-mu) ~ 1e-9 representable.
        InverseLink::Standard(StandardLink::Probit) => eta.clamp(-6.0, 6.0),
        // Why: cloglog has mu~exp(eta) for eta<<0 (underflows below ~-23) and 1-mu~exp(-exp(eta)) collapses by eta=3.
        InverseLink::Standard(StandardLink::CLogLog) | InverseLink::LatentCLogLog(_) => {
            eta.clamp(-23.0, 3.0)
        }
        // Why: SAS / beta-logistic / mixture compose logistic-like sigmoids that saturate by |eta|~20 (logistic(20)~1-2e-9).
        InverseLink::Sas(_) | InverseLink::BetaLogistic(_) | InverseLink::Mixture(_) => {
            eta.clamp(-20.0, 20.0)
        }
    }
}


/// Returns true at rows where PIRLS clamped η (so the observed-info weights
/// were computed at the clamped value, making `∂W/∂η` zero w.r.t. the
/// **unclamped** η).  Outer REML/LAML derivative formulas must mask `c_obs`
/// and `d_obs` to zero on these rows or the analytic ∂H/∂ψ disagrees with
/// the H whose log-det we differentiate.
#[inline]
pub fn eta_clamp_active(inverse_link: &InverseLink, eta: f64) -> bool {
    let clamped = eta_for_observed_hessian_jet(inverse_link, eta);
    clamped != eta
}


/// Build solver-conditioned weights from the exact hessian weights.
///
/// The returned array applies a solver-only floor per observation so the
/// Newton linear system X'W X + S stays numerically usable. This floor is
/// purely a linear-algebra concern: the exact statistical weights stored in
/// `lasthessian_weights` / `finalweights` are not affected.
pub(crate) fn solver_hessian_weights_into(
    hessian_weights: &Array1<f64>,
    fisher_weights: &Array1<f64>,
    out: &mut Array1<f64>,
) {
    if out.len() != hessian_weights.len() {
        *out = Array1::<f64>::zeros(hessian_weights.len());
    }
    ndarray::Zip::from(out)
        .and(hessian_weights)
        .and(fisher_weights)
        .par_for_each(|o, &w, &fw| {
            let floor = solver_hessian_weight_floor(fw);
            *o = if w.is_finite() && w > floor { w } else { floor };
        });
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
/// - `hessian_weights`: W_obs per observation (exact; solver floor applied separately).
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

    let weight_family = weight_family_for_glm_likelihood(likelihood);
    let weight_link = weight_link_for_inverse_link(inverse_link);
    let phi = fixed_glm_dispersion(likelihood);

    // Parallel per-row weight assembly. At large scale (n = 320k) this loop
    // dominates non-canonical paths because each row independently evaluates
    // inverse-link jets and residual-dependent observed curvature. Write
    // directly into reusable output slices rather than collecting row tuples,
    // which removes an O(n) temporary allocation on every PIRLS update.
    hessian_weights
        .as_slice_mut()
        .expect("hessian weights must be contiguous")
        .par_iter_mut()
        .zip(
            hessian_c
                .as_slice_mut()
                .expect("hessian c must be contiguous")
                .par_iter_mut(),
        )
        .zip(
            hessian_d
                .as_slice_mut()
                .expect("hessian d must be contiguous")
                .par_iter_mut(),
        )
        .enumerate()
        .try_for_each(|(i, ((w_out, c_out), d_out))| -> Result<(), EstimationError> {
            let eta_used = eta_for_observed_hessian_jet(inverse_link, eta[i]);
            // Why: closed-form observed_weight_noncanonical requires (mu, d1..d3, h4) at one consistent eta;
            // mixing PIRLS-state jets at unclamped eta with h4 at eta_used produced 0/0 in phi_v* divisions,
            // surfacing as: "observed Hessian curvature is not positive finite at row N: observed=NaN, fisher=0".
            let jet =
                crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta_used)?;
            let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                inverse_link, eta_used,
            )?;
            let (w_obs, c_obs, d_obs) = observed_weight_dispatch(
                weight_family,
                weight_link,
                eta_used,
                y[i],
                jet.mu,
                phi,
                priorweights[i].max(0.0),
                jet,
                h4,
            );
            let fisher_weight = fisher_weights[i].max(0.0);
            if !(w_obs.is_finite() && w_obs > 0.0) {
                crate::bail_invalid_estim!(
                    "observed Hessian curvature is not positive finite at row {i}: observed={w_obs}, fisher={fisher_weight}"
                );
            }
            if !c_obs.is_finite() || !d_obs.is_finite() {
                crate::bail_invalid_estim!(
                    "observed Hessian curvature derivatives are non-finite at row {i}: c={c_obs}, d={d_obs}"
                );
            }
            *w_out = w_obs;
            *c_out = c_obs;
            *d_out = d_obs;
            Ok(())
        })
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
/// 2. The outer gradient's C[v] correction (uses c_obs).
/// 3. The outer Hessian's Q[v_k, v_l] correction (uses d_obs).
///
/// Using Fisher weights in the outer REML would yield a PQL-type surrogate
/// rather than the exact Laplace approximation.
///
/// # Arguments
/// * `y`   -- response value
/// * `mu`  -- fitted mean h(eta)
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
    y: f64,
    mu: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> (f64, f64, f64) {
    let VarianceJet {
        v,
        v1,
        v2,
        v3,
        v4: _,
    } = vj;
    let phi_v = phi * v;
    let phi_v2 = phi * v * v;
    let phi_v3 = phi * v * v * v;

    // ---- Fisher weight and derivatives ----
    let h1_sq = h1 * h1;
    let w_f = h1_sq / phi_v;

    // c_F = (2 h₁ h₂ V − h₁³ V₁) / (φ V²)
    let n0 = h1_sq; // numerator of w_F
    let n1 = 2.0 * h1 * h2; // ∂(h₁²)/∂η
    let n2 = 2.0 * (h2 * h2 + h1 * h3); // ∂²(h₁²)/∂η²
    let vd1 = h1 * v1; // ∂V/∂η = V'·h'
    let vd2 = h2 * v1 + h1_sq * v2; // ∂²V/∂η²

    let c_f = (n1 * v - n0 * vd1) / phi_v2;

    // d_F = ∂c_F/∂η via quotient rule on c_F = (n1·v − n0·vd1) / (φ·v²)
    // numerator of c_F and its η-derivative (cross terms cancel):
    let numer_cf = n1 * v - n0 * vd1;
    let dnumer_cf = n2 * v - n0 * vd2;
    let d_f = (dnumer_cf * v - 2.0 * numer_cf * vd1) / (phi_v3);

    // ---- Observed correction term B and its η-derivatives ----
    // B = (h₂ V − h₁² V₁) / (φ V²)
    let b_num = h2 * v - h1_sq * v1;
    let b = b_num / phi_v2;

    // B_η = (h₃ V² − 3 h₁ h₂ V V₁ − h₁³ V V₂ + 2 h₁³ V₁²) / (φ V³)
    let b_eta_num =
        h3 * v * v - 3.0 * h1 * h2 * v * v1 - h1_sq * h1 * v * v2 + 2.0 * h1_sq * h1 * v1 * v1;
    let b_eta = b_eta_num / phi_v3;

    // B_ηη = ∂B_η/∂η.
    //
    // We differentiate b_eta_num / (φ V³) using the quotient rule.
    //
    // Numerator derivative of b_eta_num w.r.t. η, using chain rule ∂/∂η = h₁·∂/∂μ
    // for the V-dependent parts:
    //
    //   ∂/∂η [h₃ V²]               = h₄ V² + 2 h₃ V h₁ V₁
    //   ∂/∂η [3 h₁ h₂ V V₁]        = 3(h₂² + h₁ h₃)V V₁ + 3 h₁ h₂(h₁ V₁² + V h₁ V₂)
    //   ∂/∂η [h₁³ V V₂]            = 3 h₁² h₂ V V₂ + h₁³(h₁ V₁ V₂ + V h₁ V₃)
    //   ∂/∂η [2 h₁³ V₁²]           = 6 h₁² h₂ V₁² + 4 h₁³ V₁ h₁ V₂
    //                                = 6 h₁² h₂ V₁² + 4 h1_sq * h1_sq * v1 * v2
    //
    // Denominator derivative: ∂/∂η [φ V³] = 3 φ V² h₁ V₁.

    let h1_cu = h1_sq * h1;
    let h1_qu = h1_sq * h1_sq;

    let db_eta_num = h4 * v * v + 2.0 * h3 * v * h1 * v1
        - 3.0 * (h2 * h2 + h1 * h3) * v * v1
        - 3.0 * h1 * h2 * (h1 * v1 * v1 + v * h1 * v2)
        - 3.0 * h1_sq * h2 * v * v2
        - h1_cu * (h1 * v1 * v2 + v * h1 * v3)
        + 6.0 * h1_sq * h2 * v1 * v1
        + 4.0 * h1_qu * v1 * v2;

    let phi_v4 = phi_v3 * v;
    let b_etaeta = (db_eta_num * v - 3.0 * b_eta_num * h1 * v1) / phi_v4;

    // ---- Assemble observed quantities ----
    let resid = y - mu;

    let w_obs = w_f - resid * b;
    let c_obs = c_f + h1 * b - resid * b_eta;
    let d_obs = d_f + h2 * b + 2.0 * h1 * b_eta - resid * b_etaeta;

    (pw * w_obs, pw * c_obs, pw * d_obs)
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
    y: f64,
    mu: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    h5: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> f64 {
    let VarianceJet { v, v1, v2, v3, v4 } = vj;
    let q = phi * v;

    // Q = φV and its η-derivatives.
    //   Q'    = φ V₁ h₁
    //   Q''   = φ (V₁ h₂ + V₂ h₁²)
    //   Q'''  = φ (V₁ h₃ + 3 V₂ h₁ h₂ + V₃ h₁³)
    //   Q'''' = φ (V₁ h₄ + 4 V₂ h₁ h₃ + 3 V₂ h₂² + 6 V₃ h₁² h₂ + V₄ h₁⁴)
    let h1_sq = h1 * h1;
    let h1_cu = h1_sq * h1;
    let h1_qu = h1_sq * h1_sq;

    let q1 = phi * v1 * h1;
    let q2 = phi * (v1 * h2 + v2 * h1_sq);
    let q3 = phi * (v1 * h3 + 3.0 * v2 * h1 * h2 + v3 * h1_cu);
    let q4 = phi
        * (v1 * h4 + 4.0 * v2 * h1 * h3 + 3.0 * v2 * h2 * h2 + 6.0 * v3 * h1_sq * h2 + v4 * h1_qu);

    // T = h₁/Q and T', T'', T''', T'''' via Leibniz on T·Q = h₁.
    //   T'    = (h₂  − T·Q')/Q
    //   T''   = (h₃  − 2 T'·Q' − T·Q'')/Q
    //   T'''  = (h₄  − 3 T''·Q' − 3 T'·Q'' − T·Q''')/Q
    //   T'''' = (h₅  − 4 T'''·Q' − 6 T''·Q'' − 4 T'·Q''' − T·Q'''')/Q
    let t0 = h1 / q;
    let t1 = (h2 - t0 * q1) / q;
    let t2 = (h3 - 2.0 * t1 * q1 - t0 * q2) / q;
    let t3 = (h4 - 3.0 * t2 * q1 - 3.0 * t1 * q2 - t0 * q3) / q;
    let t4 = (h5 - 4.0 * t3 * q1 - 6.0 * t2 * q2 - 4.0 * t1 * q3 - t0 * q4) / q;

    // Fisher weight derivatives via product rule on W_F = h₁·T.
    //   W_F^(0) = h₁ T
    //   W_F^(1) = h₁ T₁ + h₂ T
    //   W_F^(2) = h₁ T₂ + 2 h₂ T₁ + h₃ T
    //   W_F^(3) = h₁ T₃ + 3 h₂ T₂ + 3 h₃ T₁ + h₄ T
    let w_f3 = h1 * t3 + 3.0 * h2 * t2 + 3.0 * h3 * t1 + h4 * t0;

    // Observed third derivative: differentiate W_obs = W_F − (y−μ)·T₁ thrice.
    // (resid)' = −h₁, so iterating product rule yields
    //   ∂³((y−μ)·T₁)/∂η³ = −h₃·T₁ − 3 h₂·T₂ − 3 h₁·T₃ + (y−μ)·T₄
    let resid = y - mu;
    let e_obs = w_f3 + h3 * t1 + 3.0 * h2 * t2 + 3.0 * h1 * t3 - resid * t4;

    pw * e_obs
}


// Direct (closed-form) observed-information weights for specific family-link
// combinations.  These avoid the overhead of the generic noncanonical formula
// when the algebra simplifies.

/// Gaussian family with log link: y ~ N(μ, φ), μ = exp(η).
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight `pw`.
///
/// ```text
/// w_obs = ω μ(2μ − y) / φ
/// c_obs = ω μ(4μ − y) / φ
/// d_obs = ω μ(8μ − y) / φ
/// ```
#[inline]
pub fn observed_weight_gaussian_log(y: f64, mu: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let inv_phi = pw / phi;
    let w = inv_phi * mu * (2.0 * mu - y);
    let c = inv_phi * mu * (4.0 * mu - y);
    let d = inv_phi * mu * (8.0 * mu - y);
    (w, c, d)
}


/// Gaussian family with inverse link: y ~ N(μ, φ), μ = 1/η.
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight `pw`.
///
/// ```text
/// w_obs = ω (3 − 2ηy) / (φ η⁴)
/// c_obs = 6ω (ηy − 2) / (φ η⁵)
/// d_obs = 12ω (5 − 2ηy) / (φ η⁶)
/// ```
#[inline]
pub fn observed_weight_gaussian_inverse(y: f64, eta: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let eta2 = eta * eta;
    let eta4 = eta2 * eta2;
    let eta5 = eta4 * eta;
    let eta6 = eta4 * eta2;
    let ey = eta * y;
    let inv_phi = pw / phi;
    let w = inv_phi * (3.0 - 2.0 * ey) / eta4;
    let c = inv_phi * 6.0 * (ey - 2.0) / eta5;
    let d = inv_phi * 12.0 * (5.0 - 2.0 * ey) / eta6;
    (w, c, d)
}


#[inline]
pub(crate) fn observed_weight_binomial_logit_from_jet(
    n_trials: f64,
    jet: MixtureInverseLinkJet,
    pw: f64,
) -> (f64, f64, f64) {
    let scale = pw * n_trials;
    (scale * jet.d1, scale * jet.d2, scale * jet.d3)
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
    Identity,
    Log,
    Logit,
    Inverse,
    /// Any other link — falls back to the generic noncanonical formula.
    Other,
}


#[inline]
pub fn variance_jet_for_weight_family(family: WeightFamily, mu: f64) -> VarianceJet {
    match family {
        WeightFamily::Gaussian => VarianceJet::gaussian(),
        WeightFamily::Binomial => VarianceJet::binomial_n(mu),
        WeightFamily::Poisson => VarianceJet::poisson(mu),
        WeightFamily::Tweedie { p } => VarianceJet::tweedie(mu, p),
        WeightFamily::NegativeBinomial { theta } => VarianceJet::negative_binomial(mu, theta),
        WeightFamily::Beta { phi } => VarianceJet::beta(mu, phi),
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
pub fn observed_weight_dispatch(
    family: WeightFamily,
    link: WeightLink,
    eta: f64,
    y: f64,
    mu: f64,
    phi: f64,
    prior_weight: f64,
    jet: MixtureInverseLinkJet,
    h4: f64,
) -> (f64, f64, f64) {
    match (family, link) {
        (WeightFamily::Gaussian, WeightLink::Log) => {
            observed_weight_gaussian_log(y, mu, phi, prior_weight)
        }
        (WeightFamily::Gaussian, WeightLink::Inverse) => {
            observed_weight_gaussian_inverse(y, eta, phi, prior_weight)
        }
        (WeightFamily::Binomial, WeightLink::Logit) => {
            observed_weight_binomial_logit_from_jet(1.0, jet, prior_weight)
        }
        _ => {
            // Generic noncanonical path via the full variance-function jet.
            let vj = variance_jet_for_weight_family(family, mu);
            observed_weight_noncanonical(y, mu, jet.d1, jet.d2, jet.d3, h4, vj, phi, prior_weight)
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
    hessian_weights: &Array1<f64>,
    eta_direction: &Array1<f64>,
) -> DirectionalWorkingCurvature {
    let mut w_direction = c_array * eta_direction;
    for i in 0..w_direction.len() {
        if hessian_weights[i] <= 0.0 || !w_direction[i].is_finite() {
            w_direction[i] = 0.0;
        }
    }
    DirectionalWorkingCurvature::Diagonal(w_direction)
}
