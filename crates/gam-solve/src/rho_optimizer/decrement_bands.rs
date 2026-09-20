//! The bands a Newton-decrement stationarity verdict is decided against, the
//! face it is taken on, and the gradient bound it publishes (#2954).

use super::run::{OuterConfig, StationarityBoundSource};
use crate::estimate::outer_eval_capture::CertificateEvidence;
use ndarray::{Array1, Array2};

/// The error bands of a Newton-decrement stationarity certificate at a point
/// whose curvature is in hand (#2954), over `coordinates` gradient components,
/// together with the objective band by term.
///
/// Every band is the forward error of the quantity it bands, charged with the
/// u-based growth factor `γ_m = m·u/(1 − m·u)` (`accumulation_growth`) at the
/// sequential count `m = n + p²` over the declared `n` rows and `p`
/// coefficients, the largest count any accumulation in these channels can have:
///
/// * each gradient component's band is `γ_m·(|fixed_beta_k| + |logdet_h_k| +
///   |logdet_s_k|) + |kkt_k|`. The magnitudes are read from the evaluation's own
///   [`RhoGradientParts`](crate::estimate::outer_eval_capture::RhoGradientParts),
///   where `kkt_k = total − (fixed_beta + logdet_h + logdet_s)` is the applied
///   implicit-function correction. Charging the assembled `|g_k|` instead can
///   only understate `band_λ²` when the channels cancel, and so only certify more;
/// * the Hessian's band is `γ_m·‖H‖_F` at the same count;
/// * `band_f = B_channels + B_factor + |E_r|`, the error the evaluated `V` itself
///   carries. A decrease below it is one the evaluator cannot resolve, so
///   demanding it would refuse points the criterion cannot tell apart:
///   * the channel term `B_channels = γ_m·(|fixed_beta| + |logdet_h| +
///     |logdet_s| + |kkt|)` charges each additive channel of `V`
///     ([`CertificateCriterion`]) on its own magnitude, because the assembled
///     `|V|` understates the rounding whenever the channels cancel. A route that
///     publishes no channels is charged `γ_1·|V|`;
///   * the factor term `B_factor = ½·δ_logdet` is the first-order forward error
///     the inner factorization's own backward error carries into the
///     `½·log|H_β|` channel ([`InnerFactorCondition`]), with the `O(‖δH‖²)`
///     remainder dropped. The factor in hand derives it for its own kernel: for
///     the symmetric eigensolver (Weyl, a normwise backward error `‖E‖₂ ≤
///     p·ε·‖H‖₂`) `δ_logdet = p·ε·‖H‖₂·Σ_active 1/σ_i`; for Cholesky (Higham
///     Thm 10.3, componentwise `|δH| ≤ γ_(p+1)·|L||Lᵀ|`) `δ_logdet =
///     p·γ_(p+1)·‖H̃⁻¹‖_F` with `H̃ = D⁻¹HD⁻¹`, `D = diag(H)^(1/2)`, the
///     equilibrated condition van der Sluis makes the right one there. No
///     equilibration tightens the eigensolver's normwise error. The eigenvalues
///     summed are the regularized `r_ε(σ_i)` the operator prices `log|H_β|₊`
///     with, so the floor that regularization applies bounds each term. Where
///     the channel is nonzero and the factor derives no bound, no verdict is
///     taken;
///   * the inner-mode term `E_r = ½·rᵀH_β⁻¹r` is the error `V` carries because
///     its inner mode stops at a residual `r` rather than at the exact mode, in
///     `V`'s units ([`InnerResidualCharge`]): the iterative inner Newton's own
///     final penalized gradient, or a direct solve's normal-equation residual
///     `(XᵀWX + S_λ)β̂ − XᵀWz`, including the moving-Hessian log-det response
///     where the IFT correction formed it. Where the evaluation can form no
///     residual, no verdict is taken;
///
/// `Err` names why no bands can be formed, and the caller's first-order ladder
/// decides instead: the route declares no problem size (no formation count to
/// charge), the evaluation published no parts for some coordinate (no term
/// magnitudes to charge), its `log|H_β|` came from a factor that derives no
/// forward error for it, it formed no inner residual, or `band_f` exceeds
/// `τ_stat` or is not finite.
///
/// The verdict's tolerance is [`DecrementTolerance::value`], `max(τ_stat −
/// band_f, band_f)` over the statistical resolution `τ_stat = 1/(2n)` (C3,
/// boundary-probability §3.6). It replaces `rel_cost_floor·(1 + |V|)`, which
/// moved with the units of `y` and with any additive constant in `V`.
///
/// [`CertificateCriterion`]: crate::estimate::outer_eval_capture::CertificateCriterion
/// [`InnerFactorCondition`]: crate::estimate::outer_eval_capture::InnerFactorCondition
/// [`InnerResidualCharge`]: crate::estimate::outer_eval_capture::InnerResidualCharge
pub(crate) fn outer_decrement_bands(
    config: &OuterConfig,
    hessian: &Array2<f64>,
    coordinates: usize,
    cost: f64,
    evidence: &CertificateEvidence,
) -> Result<(opt::DecrementBands, ObjectiveBand, DecrementTolerance), DecrementVerdictNotTaken> {
    let size = &config.problem_size;
    let (Some(n_obs), Some(p_coefficients)) = (size.n_obs, size.p_coefficients) else {
        return Err(DecrementVerdictNotTaken::NoProblemSize);
    };
    let growth = gam_linalg::roundoff::accumulation_growth(n_obs + p_coefficients * p_coefficients);
    let mut gradient_band = Array1::<f64>::zeros(coordinates);
    for (k, band) in gradient_band.iter_mut().enumerate() {
        let Some(part) = evidence.parts.iter().find(|part| part.index == k) else {
            return Err(DecrementVerdictNotTaken::NoGradientParts { coordinate: k });
        };
        let envelope = part.fixed_beta + part.logdet_h + part.logdet_s;
        let channels = part.fixed_beta.abs() + part.logdet_h.abs() + part.logdet_s.abs();
        *band = growth * channels + (part.total - envelope).abs();
    }
    let frobenius = hessian
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let objective_band = outer_objective_band(config, cost, evidence)?;
    let Some(tau_stat) = size.statistical_resolution() else {
        return Err(DecrementVerdictNotTaken::NoProblemSize);
    };
    // The decrease the certificate may leave is the statistical resolution
    // less the error `V` itself carries, `τ_stat − band_f`: `λ̂² + band_λ² ≤
    // τ_stat − band_f` bounds the true decrease left, rounding included, by
    // `τ_stat`. Where `½τ_stat < band_f ≤ τ_stat` the arithmetic cannot
    // resolve that standard, and the verdict decides at the arithmetic's own
    // resolution `band_f`, the tightest decidable one, as the per-coordinate
    // band decides at `max(ε, τ − ε)` (#2954); the two meet at `band_f =
    // ½τ_stat`, and the decrease a certificate leaves stays within `τ_stat`.
    // Past `τ_stat` the evaluator cannot resolve a statistically meaningful
    // decrease at all, so no verdict is taken there (T1; #3192 makes it a hard
    // typed failure once the running-error band replaces the a priori one).
    if !(objective_band.total() <= tau_stat) {
        return Err(DecrementVerdictNotTaken::ObjectiveNotResolvable {
            band_f: objective_band.total(),
            tau_stat,
        });
    }
    let tolerance = DecrementTolerance {
        tau_stat,
        band_f: objective_band.total(),
    };
    // Two quantities, two fields (#3012): the verdict's bar is the tolerance,
    // and `objective` is the rounding band, which opt echoes as the evidence's
    // `band_f` for every reader that asks whether a change in `V` is real.
    let bands = opt::DecrementBands {
        objective: objective_band.total(),
        tolerance: tolerance.value(),
        gradient: gradient_band,
        hessian: growth * frobenius,
    };
    Ok((bands, objective_band, tolerance))
}

/// The error `band_f` the evaluated `V` carries, by term (#2954), formed from
/// the evaluation's own [`CertificateEvidence`] exactly as
/// [`outer_decrement_bands`] forms it, without the gradient and Hessian bands or
/// the certificate's tolerance.
///
/// This is the band a comparison of two evaluated values needs (#3018): two
/// values differ resolvably exactly when they are further apart than the sum of
/// their bands. The count `m = n + p²` is needed only to charge published
/// channels; a route that publishes none is charged `γ_1·|V|`. `Err` names why
/// no band can be formed: channels on a route that declares no size, a nonzero
/// `log|H_β|` channel whose factor derives no forward error, or no inner residual.
pub(crate) fn outer_objective_band(
    config: &OuterConfig,
    cost: f64,
    evidence: &CertificateEvidence,
) -> Result<ObjectiveBand, DecrementVerdictNotTaken> {
    let criterion = evidence.criterion;
    // A `log|H_β|` channel whose factor derives no forward error would be
    // charged nothing for it, so no band is formed there.
    if criterion.is_some_and(|criterion| criterion.logdet_h != 0.0)
        && evidence.inner_factor.is_none()
    {
        return Err(DecrementVerdictNotTaken::NoLogdetForwardError);
    }
    // An inner mode whose residual the evaluation cannot form carries an error
    // the band would charge nothing for, so no band is formed there.
    let Some(inner_residual) = evidence
        .inner_residual
        .map(|charge| charge.energy.abs())
        .filter(|energy| energy.is_finite())
    else {
        return Err(DecrementVerdictNotTaken::NoInnerResidual);
    };
    let channels = match criterion {
        None => gam_linalg::roundoff::accumulation_growth(1) * cost.abs(),
        Some(criterion) => {
            let size = &config.problem_size;
            let (Some(n_obs), Some(p_coefficients)) = (size.n_obs, size.p_coefficients) else {
                return Err(DecrementVerdictNotTaken::NoProblemSize);
            };
            gam_linalg::roundoff::accumulation_growth(n_obs + p_coefficients * p_coefficients)
                * (criterion.fixed_beta.abs()
                    + criterion.logdet_h.abs()
                    + criterion.logdet_s.abs()
                    + criterion.kkt.abs())
        }
    };
    Ok(ObjectiveBand {
        channels,
        factor: criterion
            .filter(|criterion| criterion.logdet_h != 0.0)
            .zip(evidence.inner_factor)
            .map(|(_, factor)| 0.5 * factor.logdet_forward_error.abs())
            .filter(|band| band.is_finite())
            .unwrap_or(0.0),
        inner_residual,
    })
}

/// The Newton-decrement stationarity verdict at a point whose curvature is in
/// hand, decided against the bands [`outer_decrement_bands`] forms (#2954):
/// [`opt::newton_decrement_verdict`] certifies iff `λ̂² + band_λ² ≤
/// max(τ_stat − band_f, band_f)` ([`DecrementTolerance::value`]).
///
/// Every gradient standard the certificate used to apply grew with `n`: the
/// arithmetic floor `n·√ε`, the declared-scale rung `τ·(1 + n)`, the
/// point-anchored widening `τ·(1 + |V|)`, and the curvature rung's decrement
/// tolerance `rel_cost_floor·(1 + |V|)`. At `n = 300,000` the declared band was
/// `6.0` and a seed certified in zero iterations. The decrement bounds the decrease
/// left to the minimum, in the criterion's own units, so it needs no
/// scale anchor; judged against `τ_stat = 1/(2n)` less the arithmetic's
/// resolution it is independent of `outer_tol`, of the units of `y` and of any
/// additive constant in `V` too.
///
/// Railed coordinates go in the active set, where the projected gradient's sign
/// decides them: a railed coordinate stays on the face only while its projected
/// gradient, the inward component the box projection keeps, is within its own
/// band. One whose inward descent the arithmetic resolves fails its bound's KKT
/// condition, so it is released and judged free with the rest.
///
/// `Err` is [`outer_decrement_bands`]'s: no verdict is taken.
pub(crate) fn outer_decrement_verdict(
    config: &OuterConfig,
    hessian: &Array2<f64>,
    projected_gradient: &Array1<f64>,
    railed: &[usize],
    cost: f64,
    evidence: &CertificateEvidence,
) -> Result<OuterDecrementDecision, DecrementVerdictNotTaken> {
    let (bands, objective_band, tolerance) =
        outer_decrement_bands(config, hessian, projected_gradient.len(), cost, evidence)?;
    let mut face = Vec::new();
    let mut released = Vec::new();
    for &k in railed {
        if k >= projected_gradient.len() || face.contains(&k) || released.contains(&k) {
            continue;
        }
        if projected_gradient[k].abs() <= bands.gradient[k] {
            face.push(k);
        } else {
            released.push(k);
        }
    }
    let mut active = vec![false; projected_gradient.len()];
    for &k in &face {
        active[k] = true;
    }
    Ok(OuterDecrementDecision {
        verdict: opt::newton_decrement_verdict(hessian, projected_gradient, Some(&active), &bands),
        face,
        released,
        objective_band,
        tolerance,
    })
}

/// Why no Newton-decrement verdict is taken at a point (#2954). The caller's
/// first-order ladder decides there instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DecrementVerdictNotTaken {
    /// The route declares no problem size, so there is no formation count to
    /// charge the bands at.
    NoProblemSize,
    /// The evaluation published no gradient parts for `coordinate`, so its band
    /// has no term magnitudes to charge.
    NoGradientParts { coordinate: usize },
    /// The criterion's `log|H_β|` channel came from an inner factor that derives
    /// no forward error for its log-determinant.
    NoLogdetForwardError,
    /// The evaluation formed no residual for its inner mode, so the error its
    /// value carries from stopping short of the exact mode is unknown.
    NoInnerResidual,
    /// The objective band `band_f` exceeds the statistical resolution
    /// `τ_stat = 1/(2n)` (or is not finite): the evaluator cannot resolve a
    /// decrease the data could distinguish, so a verdict at `band_f` would
    /// certify a point the criterion cannot tell from a better one (T1, #3192).
    ObjectiveNotResolvable { band_f: f64, tau_stat: f64 },
}

impl std::fmt::Display for DecrementVerdictNotTaken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProblemSize => f.write_str("the route declares no problem size"),
            Self::NoGradientParts { coordinate } => write!(
                f,
                "the evaluation published no gradient parts for coordinate {coordinate}"
            ),
            Self::NoLogdetForwardError => {
                f.write_str("the inner factor derives no forward error for its log-determinant")
            }
            Self::NoInnerResidual => {
                f.write_str("the evaluation formed no residual for its inner mode")
            }
            Self::ObjectiveNotResolvable { band_f, tau_stat } => write!(
                f,
                "objective not resolvable: band_f {band_f:.3e} exceeds the statistical \
                 resolution τ_stat {tau_stat:.3e}"
            ),
        }
    }
}

/// The objective band a decrement verdict was decided against, by term (#2954).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ObjectiveBand {
    /// `γ_m·(|fixed_beta| + |logdet_h| + |logdet_s| + |kkt|)`, the rounding of
    /// `V`'s channels (`γ_1·|V|` where the route publishes none).
    pub(crate) channels: f64,
    /// `½·δ_logdet`, the inner factor's forward error in the `½·log|H_β|` channel.
    pub(crate) factor: f64,
    /// `|½·rᵀH_β⁻¹r|`, the error the inner mode's residual leaves in `V`.
    pub(crate) inner_residual: f64,
}

impl ObjectiveBand {
    pub(crate) fn total(self) -> f64 {
        self.channels + self.factor + self.inner_residual
    }
}

/// The decrease a decrement verdict may leave to the minimum (C3,
/// boundary-probability §3.6): the statistical resolution less the objective
/// band, `max(τ_stat − band_f, band_f)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DecrementTolerance {
    /// `τ_stat = 1/(2n)` ([`OuterProblemSize::statistical_resolution`]).
    ///
    /// [`OuterProblemSize::statistical_resolution`]: super::run::OuterProblemSize::statistical_resolution
    pub(crate) tau_stat: f64,
    /// [`ObjectiveBand::total`], the error the evaluated `V` carries.
    pub(crate) band_f: f64,
}

impl DecrementTolerance {
    /// The tolerance the verdict certifies `λ̂² + band_λ²` against.
    pub(crate) fn value(self) -> f64 {
        (self.tau_stat - self.band_f).max(self.band_f)
    }

    /// Whether the arithmetic cannot resolve the statistical resolution, so the
    /// verdict decides at `band_f` instead (#3192).
    pub(crate) fn arithmetic_limited(self) -> bool {
        2.0 * self.band_f > self.tau_stat
    }
}

/// The criterion's resolution at a comparison whose evaluated values carry the
/// summed band `band` (#3286, #3192): the certificate's own tolerance
/// [`DecrementTolerance::value`], `max(τ_stat − band, band)`, over the criterion's
/// statistical resolution `tau_stat` ([`outer_criterion_resolution`]).
///
/// A decrease is of consequence exactly when it exceeds this. A measured decrease
/// `d` between two values within `b_a` and `b_b` of the exact criterion proves no
/// true decrease above `τ_stat` once `d + b_a + b_b ≤ τ_stat`, so the upper arm is
/// `τ_stat − (b_a + b_b)`; and no comparison can claim a resolution finer than the
/// arithmetic's own `b_a + b_b`, the lower arm. Every consumer asking whether a
/// decrease matters reads this one number: the cost-stall guard's resolved-descent
/// and stall tests, the fixed-point progress certificate, the online decrement
/// stops, and the certificate's rungs.
///
/// A route that declares no observation count has `τ_stat = 0`, no statistical
/// slack, and decides at its arithmetic resolution `band`. Reading the bare
/// `τ_stat` there resolved every difference and so switched both of ARC's stops
/// off (#3286).
///
/// [`outer_criterion_resolution`]: super::run::outer_criterion_resolution
pub(crate) fn outer_resolution(tau_stat: f64, band: f64) -> f64 {
    DecrementTolerance {
        tau_stat,
        band_f: band,
    }
    .value()
}

/// The evaluation band `b` of a computed criterion value (#3286): the objective
/// band `band_f` its evaluation's evidence forms ([`outer_objective_band`]), or,
/// where that evidence forms none, the value's own representation error
/// ([`value_representation_band`]).
pub(crate) fn outer_value_band(
    config: &OuterConfig,
    cost: f64,
    evidence: Option<&CertificateEvidence>,
) -> f64 {
    evidence
        .and_then(|evidence| outer_objective_band(config, cost, evidence).ok())
        .map(ObjectiveBand::total)
        .filter(|band| band.is_finite())
        .unwrap_or_else(|| value_representation_band(cost))
}

/// `γ₁·|V|`, the representation error every computed criterion value carries: the
/// band of a value whose evaluation publishes no evidence (#3286).
pub(crate) fn value_representation_band(cost: f64) -> f64 {
    gam_linalg::roundoff::accumulation_growth(1) * cost.abs()
}

/// A Newton-decrement verdict with the face it was taken on (#2954).
#[derive(Debug, Clone)]
pub(crate) struct OuterDecrementDecision {
    pub(crate) verdict: opt::DecrementVerdict,
    /// The railed coordinates held on the face: each one's projected gradient
    /// is within its own band, so its bound's KKT condition holds.
    pub(crate) face: Vec<usize>,
    /// The railed coordinates released from the face because their inward
    /// descent is resolvable. The verdict judged them free.
    pub(crate) released: Vec<usize>,
    pub(crate) objective_band: ObjectiveBand,
    pub(crate) tolerance: DecrementTolerance,
}

/// The stationarity bound a decrement verdict publishes, with its rung (#2954).
///
/// A certificate records a gradient bound beside `|Pg|`, so the verdict is
/// rendered as the gradient norm along the measured direction at which the
/// decrement reaches the verdict's tolerance with the measured rounding held
/// fixed: `|Pg|·√((tol − band_λ²)/λ̂²)`, where `tol` is the evidence's `tolerance`
/// field, the tolerance [`DecrementTolerance::value`] handed to the verdict. It clears `|Pg|` exactly when the verdict
/// certifies and falls strictly below it on `DecrementAboveTolerance`. A verdict
/// that cannot decide publishes `0`, so the point refuses unless large-step
/// flatness removes its flat coordinates and a second verdict certifies.
///
/// `None` for a genuine saddle: stationarity stays on the first-order ladder
/// there, so the negative-curvature adjudication still runs on it.
pub(crate) fn decrement_stationarity_bound(
    projected_grad_norm: f64,
    verdict: &opt::DecrementVerdict,
) -> Option<(f64, StationarityBoundSource)> {
    let along_direction = |evidence: &opt::DecrementEvidence| {
        let headroom = evidence.tolerance - evidence.band_lambda_sq;
        if !(headroom > 0.0) {
            0.0
        } else if evidence.lambda_sq > 0.0 {
            projected_grad_norm * (headroom / evidence.lambda_sq).sqrt()
        } else {
            projected_grad_norm
        }
    };
    match verdict {
        opt::DecrementVerdict::Certified(evidence) => Some((
            along_direction(evidence).max(projected_grad_norm),
            StationarityBoundSource::NewtonDecrement,
        )),
        opt::DecrementVerdict::DecrementAboveTolerance(evidence) => Some((
            along_direction(evidence).min(projected_grad_norm.next_down()),
            StationarityBoundSource::NewtonDecrement,
        )),
        opt::DecrementVerdict::NotPositiveDefinite { .. } => None,
        _ => Some((0.0, StationarityBoundSource::NewtonDecrementUndecided)),
    }
}
