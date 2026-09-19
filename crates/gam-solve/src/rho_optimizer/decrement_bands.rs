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
/// forward error for it, it formed no inner residual, or `band_f` exceeds the
/// objective resolution the
/// certificate asserts, `τ = rel_cost_floor·(1 + |V|)` (the cost-stall guard's
/// and the curvature rung's): a band that wide would read any decrease as noise,
/// so it certifies nothing.
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
) -> Result<(opt::DecrementBands, ObjectiveBand), DecrementVerdictNotTaken> {
    let size = &config.rho_uncertainty_problem_size;
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
    let criterion = evidence.criterion;
    // A `log|H_β|` channel whose factor derives no forward error would be
    // charged nothing for it, so the verdict is not taken there.
    if criterion.is_some_and(|criterion| criterion.logdet_h != 0.0)
        && evidence.inner_factor.is_none()
    {
        return Err(DecrementVerdictNotTaken::NoLogdetForwardError);
    }
    // An inner mode whose residual the evaluation cannot form carries an error
    // the band would charge nothing for, so the verdict is not taken there.
    let Some(inner_residual) = evidence
        .inner_residual
        .map(|charge| charge.energy.abs())
        .filter(|energy| energy.is_finite())
    else {
        return Err(DecrementVerdictNotTaken::NoInnerResidual);
    };
    let objective_band = ObjectiveBand {
        channels: criterion.map_or_else(
            || gam_linalg::roundoff::accumulation_growth(1) * cost.abs(),
            |criterion| {
                growth
                    * (criterion.fixed_beta.abs()
                        + criterion.logdet_h.abs()
                        + criterion.logdet_s.abs()
                        + criterion.kkt.abs())
            },
        ),
        factor: criterion
            .filter(|criterion| criterion.logdet_h != 0.0)
            .zip(evidence.inner_factor)
            .map(|(_, factor)| 0.5 * factor.logdet_forward_error.abs())
            .filter(|band| band.is_finite())
            .unwrap_or(0.0),
        inner_residual,
    };
    // A band past the objective resolution the certificate itself asserts would
    // read any decrease as noise, so the verdict is not taken there.
    let resolution = super::run::outer_rel_cost_floor(config) * (1.0 + cost.abs());
    if !(objective_band.total() <= resolution) {
        return Err(DecrementVerdictNotTaken::ObjectiveNotResolvable {
            band_f: objective_band.total(),
            tau: resolution,
        });
    }
    let bands = opt::DecrementBands {
        objective: objective_band.total(),
        gradient: gradient_band,
        hessian: growth * frobenius,
    };
    Ok((bands, objective_band))
}

/// The Newton-decrement stationarity verdict at a point whose curvature is in
/// hand, decided against the bands [`outer_decrement_bands`] forms (#2954):
/// [`opt::newton_decrement_verdict`] certifies iff `½λ̂² + band_λ² ≤ band_f`.
///
/// Every gradient standard the certificate used to apply grew with `n`: the
/// arithmetic floor `n·√ε`, the declared-scale rung `τ·(1 + n)`, the
/// point-anchored widening `τ·(1 + |V|)`, and the curvature rung's decrement
/// tolerance `rel_cost_floor·(1 + |V|)`. At `n = 300,000` the declared band was
/// `6.0` and a seed certified in zero iterations. The decrement is the decrease a
/// Newton step would still buy, in the criterion's own units, so it needs no
/// scale anchor; judged at the arithmetic's resolution it is independent of
/// `outer_tol` too.
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
    let (bands, objective_band) =
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
    /// `band_f` exceeds the objective resolution `τ = rel_cost_floor·(1 + |V|)`
    /// the certificate asserts, so any decrease would read as noise.
    ObjectiveNotResolvable { band_f: f64, tau: f64 },
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
            Self::ObjectiveNotResolvable { band_f, tau } => {
                write!(
                    f,
                    "objective not resolvable: band_f {band_f:.3e} > τ {tau:.3e}"
                )
            }
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
}

/// The stationarity bound a decrement verdict publishes, with its rung (#2954).
///
/// A certificate records a gradient bound beside `|Pg|`, so the verdict is
/// rendered as the gradient norm along the measured direction at which the
/// decrement reaches the objective band with the measured rounding held fixed:
/// `|Pg|·√((band_f − band_λ²)/(½λ̂²))`. It clears `|Pg|` exactly when the verdict
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
        let headroom = evidence.band_f - evidence.band_lambda_sq;
        let half_decrement = 0.5 * evidence.lambda_sq;
        if !(headroom > 0.0) {
            0.0
        } else if half_decrement > 0.0 {
            projected_grad_norm * (headroom / half_decrement).sqrt()
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
