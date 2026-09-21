//! Response moments of a location-scale survival fit whose coefficients carry
//! an inequality cone.
//!
//! # What this replaces, and why it is not a refinement of it
//!
//! [`super::moments::exact_survival_response_moments_row`] integrates the
//! predictor against a JOINT NORMAL. That is exact whenever the posterior is
//! Gaussian, and it is what this model's non-wiggle fits get. A fit that
//! carries the monotone link-wiggle block does not have a Gaussian posterior:
//! `β_w ≥ 0` is enforced during the fit
//! (`family_solver::block_linear_constraints`, one row per wiggle coefficient)
//! and the reported posterior is the Gaussian truncated to that cone. The
//! Gaussian rule was reaching it through the moment-matched normal — the normal
//! carrying `E_π[β]` and `Σ_π`.
//!
//! Matching two moments of a cone-truncated law is not an approximation that
//! gets better with work. For `q > 1` retained rows the pushforward of the
//! truncated joint through a basis row is not normal at all, so the error is a
//! FLOOR, and the normal puts a measurable share of its mass — a few percent,
//! measured — on coefficient vectors the fit excluded. Both properties are
//! wrong in kind, not in size: no node count, tolerance, or correction factor
//! moves them.
//!
//! # The rule
//!
//! The truncated posterior factorizes exactly. Write `u = Aβ − b` for the
//! retained constraint-normal coordinates and `G = ΣAᵀW⁻¹` for the lift the
//! correction already stores. Then
//!
//! ```text
//! β = β_unc + G(u − E_untrunc[u]) + ε,   ε ~ N(0, Σ_res),  Σ_res = Σ − GWGᵀ,
//! ```
//!
//! with `ε` exactly Gaussian and exactly independent of `u` — conditioning a
//! Gaussian on `Aβ` leaves a Gaussian whatever the truncation then does to
//! `Aβ`. Note `AΣ_resAᵀ = W − W = 0`: the tangent moves no constraint normal,
//! so a draw built this way is feasible for every `u` the cone admits.
//!
//! So the law needs one rule over `(u, ε)` jointly, which is what
//! [`gam_solve::constrained_posterior::ConstrainedPosteriorJointRule`] serves: a
//! single low-discrepancy rule whose first `q` coordinates run the
//! separation-of-variables map into the cone and whose remaining coordinates
//! carry the standardized tangent. The per-row cost is one evaluation per node,
//! with no factor of the cubature's node count in it — the nested alternative
//! costs `nodes × outer × inner`, up to `4096 × 3375 × 21` on the shipped rules,
//! which is why this was never simply cut over.
//!
//! # What this rule gives up, and how its accuracy is certified
//!
//! A lattice rule on the smooth tangent block does not have the spectral
//! accuracy of the Gauss-Hermite tensor rule it replaces. So the node count is
//! not fixed: every replicate lattice of the rule is extended until the
//! replicate standard error of each row's response moments is within the law's
//! own certified relative accuracy (#2917), and the rule is gated against a
//! reference built from the DENSITY rather than from the cubature
//! (`truncated_response_moments_beat_the_moment_matched_normal_2679`). The
//! refinement this replaced compared consecutive doublings of ONE lattice, which
//! cannot see bias.

use super::*;
use gam_solve::constrained_posterior::{
    ConstrainedPosteriorCorrection, ConstrainedPosteriorGeometry, ConstrainedPosteriorJointRule,
    constrained_posterior_correction_from_covariance,
};

/// Slack allowed when checking the stored removed variance against the
/// constraint-normal variance reconstructed here.
///
/// Both are outputs of the same `1e-3`-relative orthant cubature, one saved at
/// fit time and one rebuilt from the reported covariance, so they can disagree
/// by that much without anything being wrong. This is a frame/scale guard, not
/// a precision claim: a covariance that arrived on the wrong scale misses by a
/// FACTOR, not by a part in a thousand.
const ORTHANT_REMOVED_VARIANCE_SLACK: f64 = 1e-2;

/// How far the truncated law's own mean may sit from the reported coefficient
/// vector before the frames are declared misaligned, in POSTERIOR STANDARD
/// DEVIATIONS of the coefficient in question.
///
/// Denominated in the posterior's own spread because that is the resolution the
/// reported coefficients carry: the two sides are the same estimand computed
/// two ways, and the cubature that produced the stored mean shift is certified
/// to `1e-3` relative. A wrong gauge or a wrong centre misses by an appreciable
/// fraction of a standard deviation, which this catches; quadrature noise does
/// not.
const COEFFICIENT_RECONSTRUCTION_SD_TOLERANCE: f64 = 1e-2;

/// The inequality-truncated coefficient posterior, expressed in the RAW
/// (reported) coefficient frame the response-moment rule works in.
pub(crate) struct TruncatedCoefficientLaw {
    /// `β_unc` in raw coordinates — the centre of the AMBIENT Gaussian, which
    /// is not the reported coefficient vector: the reported one is `E_π[β]`.
    center: Array1<f64>,
    /// `T G`, the raw displacement per unit of `u − E_untrunc[u]`.
    lift: Array2<f64>,
    /// `E_untrunc[u] = Aβ_unc − b` on the retained rows.
    normal_center: Array1<f64>,
    /// `Σ_res` in raw coordinates.
    residual_covariance: Array2<f64>,
    /// Joint rule over `(u, tangent)`; the tangent block has
    /// [`Self::tangent_dimension`] coordinates.
    rule: ConstrainedPosteriorJointRule,
    tangent_dimension: usize,
}

/// Build the truncated law for a fit, or `None` when the law whose second
/// moment is `covariance` retains no constraint row.
///
/// `covariance` must be one of the fit's own reported covariances: the
/// conditional `Σ_π`, or the smoothing-corrected `V_c`, whose truncated law is
/// rebuilt from the ρ-marginal ambient `Σ + C` (#3524). Any other matrix is
/// refused. It is never given the Gaussian rule, which would integrate against
/// coefficient vectors the fit excluded.
pub(crate) fn build_truncated_coefficient_law(
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    tangent_dimension: usize,
) -> Result<Option<TruncatedCoefficientLaw>, String> {
    let Some(pieces) = truncated_law_pieces(fit, covariance)? else {
        return Ok(None);
    };
    pieces.into_law(tangent_dimension).map(Some)
}

/// The truncated law as a rule over WHOLE coefficient vectors: every node of
/// the joint rule maps to one raw coefficient vector
/// `β = β_unc + TG(u − E_untrunc[u]) + L_res z`, with `L_res` a factor of
/// `Σ_res` and `z` the node's tangent coordinates, so the tangent block has the
/// rank of `Σ_res` coordinates.
///
/// This is the form a consumer needs when its functional of `β` does not
/// factor through a few linear predictors per evaluation point, as the plug-in
/// survival surfaces do not (a time grid, a hazard read from the time channel's
/// rate, a link wiggle evaluated at the realized location). Every node is
/// feasible: `u` lies inside the cone by construction and `AL_res = 0`
/// because `AΣ_resAᵀ = 0`.
pub(crate) struct TruncatedCoefficientDraws {
    law: TruncatedCoefficientLaw,
    /// `L_res` with `L_res L_resᵀ = Σ_res`, raw × rank.
    residual_factor: Array2<f64>,
}

impl TruncatedCoefficientDraws {
    /// The joint rule the draws are served on.
    pub(crate) fn rule(&self) -> &ConstrainedPosteriorJointRule {
        &self.law.rule
    }

    /// The raw coefficient vector at one node of [`Self::rule`].
    pub(crate) fn coefficients(
        &self,
        normal_coordinates: &Array1<f64>,
        tangent: &[f64],
    ) -> Result<Array1<f64>, String> {
        if normal_coordinates.len() != self.law.normal_center.len()
            || tangent.len() != self.residual_factor.ncols()
        {
            return Err(format!(
                "truncated coefficient draw: node has {} constraint-normal and {} tangent \
                 coordinates, the law has {} and {}",
                normal_coordinates.len(),
                tangent.len(),
                self.law.normal_center.len(),
                self.residual_factor.ncols()
            ));
        }
        let displacement = normal_coordinates - &self.law.normal_center;
        Ok(&self.law.center
            + &self.law.lift.dot(&displacement)
            + &self
                .residual_factor
                .dot(&ArrayView1::from(tangent)))
    }
}

/// [`TruncatedCoefficientDraws`] for a fit, or `None` under the same conditions
/// as [`build_truncated_coefficient_law`].
pub(crate) fn build_truncated_coefficient_draws(
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
) -> Result<Option<TruncatedCoefficientDraws>, String> {
    let Some(pieces) = truncated_law_pieces(fit, covariance)? else {
        return Ok(None);
    };
    let residual_factor = factorize_psd_covariance(
        &pieces.residual_covariance,
        "survival location-scale truncated residual covariance",
    )?
    .factor;
    let law = pieces.into_law(residual_factor.ncols())?;
    Ok(Some(TruncatedCoefficientDraws {
        law,
        residual_factor,
    }))
}

/// Everything of the truncated law but its rule, whose tangent dimension is the
/// consumer's.
struct TruncatedLawPieces {
    center: Array1<f64>,
    lift: Array2<f64>,
    normal_center: Array1<f64>,
    normal_covariance: Array2<f64>,
    residual_covariance: Array2<f64>,
    upper_limits: Vec<f64>,
}

impl TruncatedLawPieces {
    fn into_law(self, tangent_dimension: usize) -> Result<TruncatedCoefficientLaw, String> {
        let rule = ConstrainedPosteriorJointRule::new(
            &self.normal_center,
            &self.normal_covariance,
            &self.upper_limits,
            tangent_dimension,
        )?;
        Ok(TruncatedCoefficientLaw {
            center: self.center,
            lift: self.lift,
            normal_center: self.normal_center,
            residual_covariance: self.residual_covariance,
            rule,
            tangent_dimension,
        })
    }
}

/// The pieces of the truncated law whose reported second moment is
/// `covariance`, or `None` when that law retains no constraint row, so that its
/// ambient Gaussian IS the posterior.
///
/// Two of the fit's reported covariances describe a cone-truncated law. Each is
/// recognized by identity with the fit's own matrix, never by shape:
///
/// * The conditional `Σ_π = Σ − GΔGᵀ`, whose truncation moments the geometry
///   stores.
/// * The smoothing-corrected `V_c`, the truncation of the ρ-MARGINAL ambient
///   `Σ + C`. The feasible set constrains β and says nothing about ρ, so the
///   β-marginal of the truncated joint posterior is exactly the truncation of
///   `N(β_unc, Σ + C)` (#2705). The stored moments are not its moments. The
///   lift `G_c = (Σ+C)AᵀW_c⁻¹` and the orthant moments at `W_c = A(Σ+C)Aᵀ` are
///   functions of the covariance, so they are rebuilt here from `Σ + C` by the
///   same construction the fit publishes `V_c` with (#3524).
///
/// Any other matrix is the second moment of no law this fit defines, and is
/// refused. It is never answered with the Gaussian rule, because that would put
/// mass on coefficient vectors the fit excluded.
fn truncated_law_pieces(
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
) -> Result<Option<TruncatedLawPieces>, String> {
    let Some(geometry) = fit.geometry.as_ref() else {
        return Ok(None);
    };
    let Some(constrained) = geometry.constrained_posterior.as_ref() else {
        return Ok(None);
    };
    let stored = constrained.correction()?;
    let conditional = fit.beta_covariance();

    if conditional.is_some_and(|conditional| conditional == covariance) {
        let Some(correction) = stored else {
            // Every constraint row is slack at f64 resolution: the truncation is
            // invisible and the ambient Gaussian IS the posterior.
            return Ok(None);
        };
        let frame = ConeFrame::new(geometry, constrained, covariance.nrows())?;
        let lift = frame.lift(correction)?;
        frame.certify_conditional_mean(fit, covariance, Some((&lift, correction)))?;
        let ambient = restore_removed_variance(covariance, &lift, correction);
        return frame.pieces(constrained, correction, lift, ambient).map(Some);
    }

    if !fit
        .beta_covariance_corrected()
        .is_some_and(|corrected| corrected == covariance)
    {
        return Err(
            "survival location-scale truncated response moments: the covariance reaching this \
             rule is neither the fit's conditional nor its smoothing-corrected covariance, so \
             it is the second moment of no cone-truncated law this fit defines"
                .to_string(),
        );
    }
    let conditional = conditional.ok_or_else(|| {
        "survival location-scale truncated response moments: the fit reports a \
         smoothing-corrected covariance but no conditional one, and the corrected truncated \
         law is built on the conditional ambient covariance"
            .to_string()
    })?;
    let smoothing = fit.smoothing_correction().ok_or_else(|| {
        match fit.smoothing_correction_absence() {
            Some(absence) => format!(
                "survival location-scale truncated response moments: the fit carries no \
                 smoothing correction to build the corrected truncated law from: {absence}"
            ),
            None => "survival location-scale truncated response moments: the fit carries no \
                     smoothing correction to build the corrected truncated law from"
                .to_string(),
        }
    })?;
    let raw_dimension = covariance.nrows();
    if conditional.dim() != (raw_dimension, raw_dimension)
        || smoothing.dim() != (raw_dimension, raw_dimension)
    {
        return Err(format!(
            "survival location-scale truncated response moments: the smoothing-corrected \
             covariance is {raw_dimension}x{raw_dimension} but the conditional covariance is \
             {:?} and the smoothing correction {:?}",
            conditional.dim(),
            smoothing.dim()
        ));
    }
    let frame = ConeFrame::new(geometry, constrained, raw_dimension)?;
    // Recover the conditional ambient `Σ` from `Σ_π` exactly as the conditional
    // branch does, certifying the frame against the reported coefficients first.
    // The corrected law's own mean is not the reported vector, so this
    // certification is the one the corrected law inherits.
    let conditional_ambient = match stored {
        Some(correction) => {
            let lift = frame.lift(correction)?;
            frame.certify_conditional_mean(fit, conditional, Some((&lift, correction)))?;
            restore_removed_variance(conditional, &lift, correction)
        }
        None => {
            frame.certify_conditional_mean(fit, conditional, None)?;
            conditional.clone()
        }
    };
    // `V_c = V_cond + C` is published on the same gauge lift as the conditional
    // covariance, so the raw `C` adds to the raw ambient directly and reduces
    // to the active frame through the same exact pseudo-inverse.
    let ambient = &conditional_ambient + smoothing;
    let unconstrained_center = constrained.unconstrained_center()?;
    let marginal = constrained_posterior_correction_from_covariance(
        &frame.reduce(&ambient),
        unconstrained_center,
        &constrained.constraints,
    )?;
    let Some(correction) = marginal else {
        if stored.is_some() {
            // `C = J·V_ρ·Jᵀ` is PSD, so `W_c ⪰ W`, and a row retained at the
            // conditional law's standardized slack is retained at the corrected
            // one. Reaching this means the correction is not what `V_c` was built
            // from, and the Gaussian rule the caller holds is centred on the
            // conditional truncated mean, which is not this law's mean.
            return Err(
                "survival location-scale truncated response moments: the conditional law \
                 retains constraint rows but the smoothing-corrected law, whose ambient \
                 covariance is wider, retains none; the smoothing correction is not the one \
                 the corrected covariance was built from"
                    .to_string(),
            );
        }
        return Ok(None);
    };
    let lift = frame.lift(&correction)?;
    frame.pieces(constrained, &correction, lift, ambient).map(Some)
}

/// `Σ = Σ_π + GΔGᵀ` in RAW coordinates, with `lift = TG`. This is the identity
/// the correction stores, and both `G` and `Δ` push forward through the gauge
/// with the covariance, so this inverts it exactly. It does not recompute a
/// second ambient from a precision that may not be the one the correction was
/// built against.
fn restore_removed_variance(
    truncated: &Array2<f64>,
    lift: &Array2<f64>,
    correction: &ConstrainedPosteriorCorrection,
) -> Array2<f64> {
    let raw_dimension = truncated.nrows();
    let scaled = lift.dot(&correction.removed_normal_variance);
    let mut out = truncated.clone();
    for i in 0..raw_dimension {
        for j in 0..=i {
            let restored = scaled.row(i).dot(&lift.row(j));
            out[[i, j]] += restored;
            if i != j {
                out[[j, i]] = out[[i, j]];
            }
        }
    }
    out
}

/// The map between the ACTIVE frame the cone lives in and the RAW frame the
/// reported covariances and the response-moment rule live in.
struct ConeFrame<'a> {
    transform: &'a Array2<f64>,
    /// `(TᵀT)⁻¹`, exact because the gauge has full column rank.
    gram_inverse: Array2<f64>,
    /// `β_unc` in raw coordinates.
    center: Array1<f64>,
}

impl<'a> ConeFrame<'a> {
    fn new(
        geometry: &'a FitGeometry,
        constrained: &ConstrainedPosteriorGeometry,
        raw_dimension: usize,
    ) -> Result<Self, String> {
        let transform = &geometry.coefficient_gauge.t_full;
        if transform.nrows() != raw_dimension {
            return Err(format!(
                "survival location-scale truncated response moments: the coefficient gauge lifts \
                 into {} raw coordinates but the reported covariance is \
                 {raw_dimension}x{raw_dimension}",
                transform.nrows()
            ));
        }
        let active_dimension = transform.ncols();
        let unconstrained_center = constrained.unconstrained_center()?;
        if unconstrained_center.len() != active_dimension {
            return Err(format!(
                "survival location-scale truncated response moments: the ambient centre has {} \
                 coordinates but the gauge reduces to {active_dimension}",
                unconstrained_center.len()
            ));
        }
        let center =
            transform.dot(unconstrained_center) + &geometry.coefficient_gauge.affine_shift;

        let gram = transform.t().dot(transform);
        let (eigenvalues, eigenvectors) = gram.eigh(faer::Side::Lower).map_err(|e| {
            format!("survival location-scale gauge Gram eigendecomposition failed: {e}")
        })?;
        let max_eigenvalue = eigenvalues
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
        // A Gram eigenvalue inside the eigensolver's rounding band `γ_p·max|λ|`
        // is a rank-deficient gauge direction.
        let floor = gam_linalg::roundoff::accumulation_growth(eigenvalues.len()) * max_eigenvalue;
        if eigenvalues.iter().any(|&value| value <= floor) {
            return Err(format!(
                "survival location-scale truncated response moments: the coefficient gauge is \
                 rank deficient (smallest Gram eigenvalue {:.3e} against floor {floor:.3e}), so \
                 the constraint rows have no raw representative",
                eigenvalues
                    .iter()
                    .fold(f64::INFINITY, |acc, &value| acc.min(value))
            ));
        }
        let mut gram_inverse = Array2::<f64>::zeros((active_dimension, active_dimension));
        for (column, &eigenvalue) in eigenvalues.iter().enumerate() {
            let vector = eigenvectors.column(column);
            for i in 0..active_dimension {
                for j in 0..active_dimension {
                    gram_inverse[[i, j]] += vector[i] * vector[j] / eigenvalue;
                }
            }
        }
        Ok(Self {
            transform,
            gram_inverse,
            center,
        })
    }

    /// `TG`, the raw displacement per unit of constraint-normal displacement.
    fn lift(&self, correction: &ConstrainedPosteriorCorrection) -> Result<Array2<f64>, String> {
        if correction.lift.nrows() != self.transform.ncols() {
            return Err(format!(
                "survival location-scale truncated response moments: the correction lift has {} \
                 rows but the gauge reduces to {}",
                correction.lift.nrows(),
                self.transform.ncols()
            ));
        }
        Ok(self.transform.dot(&correction.lift))
    }

    /// `T⁺ M T⁺ᵀ` with `T⁺ = (TᵀT)⁻¹Tᵀ`: a raw bilinear form carried back into
    /// the active frame. This is exact for every form saved as the gauge
    /// congruence `T M_active Tᵀ`, which is how every covariance-like matrix
    /// of this fit is published.
    fn reduce(&self, raw: &Array2<f64>) -> Array2<f64> {
        let pseudo_inverse = self.gram_inverse.dot(&self.transform.t());
        let reduced = pseudo_inverse.dot(raw).dot(&pseudo_inverse.t());
        // The congruence of a symmetric form is symmetric; averaging with the
        // transpose only removes the rounding of the two products.
        (&reduced + &reduced.t()) * 0.5
    }

    /// `E_π[β] = β_unc + TG·(E[u] − E_untrunc[u])` must reproduce the
    /// coefficient vector the fit reports, in the RAW frame, under the
    /// conditional law the fit reports it for. This certifies the gauge lift of
    /// the centre and of the correction lift together. It is an absolute
    /// magnitude, not a ratio.
    fn certify_conditional_mean(
        &self,
        fit: &UnifiedFitResult,
        conditional: &Array2<f64>,
        truncation: Option<(&Array2<f64>, &ConstrainedPosteriorCorrection)>,
    ) -> Result<(), String> {
        let reported = &fit.beta;
        let raw_dimension = self.center.len();
        if reported.len() != raw_dimension {
            return Ok(());
        }
        let reconstructed = match truncation {
            Some((lift, correction)) => &self.center + &lift.dot(&correction.normal_mean_shift),
            None => self.center.clone(),
        };
        let mut worst = 0.0f64;
        for j in 0..raw_dimension {
            let scale = conditional[[j, j]].max(0.0).sqrt().max(f64::MIN_POSITIVE);
            worst = worst.max((reported[j] - reconstructed[j]).abs() / scale);
        }
        if worst > COEFFICIENT_RECONSTRUCTION_SD_TOLERANCE {
            return Err(format!(
                "survival location-scale truncated response moments: the truncated law's mean \
                 disagrees with the reported coefficient vector by {worst:.3e} posterior standard \
                 deviations, above {COEFFICIENT_RECONSTRUCTION_SD_TOLERANCE:.1e}; the ambient \
                 centre and the correction lift are not in the frame the reported coefficients are"
            ));
        }
        Ok(())
    }

    /// The law's pieces from its truncation `correction`, the correction's raw
    /// `lift` and the law's raw AMBIENT covariance.
    fn pieces(
        &self,
        constrained: &ConstrainedPosteriorGeometry,
        correction: &ConstrainedPosteriorCorrection,
        lift: Array2<f64>,
        ambient: Array2<f64>,
    ) -> Result<TruncatedLawPieces, String> {
        let raw_dimension = ambient.nrows();
        let active_dimension = self.transform.ncols();
        let unconstrained_center = constrained.unconstrained_center()?;
        let retained = correction.rows.len();

        let mut normal_center = Array1::<f64>::zeros(retained);
        let mut constraint_rows = Array2::<f64>::zeros((retained, active_dimension));
        for (position, &row) in correction.rows.iter().enumerate() {
            let a = constrained.constraints.a.row(row);
            normal_center[position] =
                a.dot(unconstrained_center) - constrained.constraints.b[row];
            constraint_rows.row_mut(position).assign(&a);
        }

        // `W = AΣAᵀ` lives in the ACTIVE frame while the covariance we hold is
        // the raw lift of it. `Tᵀ` is surjective onto the active frame (the
        // gauge has full column rank), so every active row `a` has an exact raw
        // representative `c = T(TᵀT)⁻¹a` with `Tᵀc = a`, and then
        // `aᵀΣ_active a = cᵀ(TΣ_activeTᵀ)c`. This is a change of representative,
        // not an approximation.
        let pulled_rows = constraint_rows
            .dot(&self.gram_inverse)
            .dot(&self.transform.t());
        let normal_covariance = pulled_rows.dot(&ambient).dot(&pulled_rows.t());

        // `Σ_res = Σ − GWGᵀ`. The constraint-normal block is removed exactly,
        // so the tangent moves no constraint normal and every point of the rule
        // below is feasible.
        let residual_covariance = {
            let scaled = lift.dot(&normal_covariance);
            let mut out = ambient;
            for i in 0..raw_dimension {
                for j in 0..=i {
                    let removed = scaled.row(i).dot(&lift.row(j));
                    out[[i, j]] -= removed;
                    if i != j {
                        out[[j, i]] = out[[i, j]];
                    }
                }
            }
            symmetrize_and_clip_covariance(&out)
        };

        // `Δ = W − Cov[u]` with both PSD, so `0 ≤ Δ_kk ≤ W_kk`. `Δ` is READ from
        // the correction while `W` is RECONSTRUCTED here from the raw ambient
        // covariance, so this compares two independently-produced numbers. A
        // covariance that reached us on a different scale than the one the
        // correction was built on breaks it immediately, where a ratio-shaped
        // check would cancel the scale and read clean.
        for k in 0..retained {
            let removed = correction.removed_normal_variance[[k, k]];
            let total = normal_covariance[[k, k]];
            let slack = ORTHANT_REMOVED_VARIANCE_SLACK * total.abs().max(removed.abs());
            if !(removed >= -slack && removed <= total + slack) {
                return Err(format!(
                    "survival location-scale truncated response moments: the removed variance \
                     {removed:.6e} on retained row {k} is not within [0, {total:.6e}], the \
                     constraint-normal variance reconstructed from the reported covariance; the \
                     covariance reaching this rule is not the one the cone correction was built \
                     against"
                ));
            }
        }

        Ok(TruncatedLawPieces {
            center: self.center.clone(),
            lift,
            normal_center,
            normal_covariance,
            residual_covariance,
            upper_limits: correction.upper_limits(),
        })
    }
}

/// Everything one row's response moment reads that does not move with the node:
/// the AMBIENT predictor centre of the three channels, how one unit of each
/// constraint-normal coordinate and of each tangent coordinate moves them, and
/// the link-wiggle block's affine conditional law.
///
/// The structure mirrors the Gaussian rule exactly — the same projected
/// covariance on `(h, threshold, log σ)`, the same affine conditional
/// regression of the link-wiggle block onto it — with two differences, and only
/// two: the covariance those blocks are read from is `Σ_res` rather than `Σ_π`,
/// and the location is the node's `β_unc + G(u − E_untrunc[u])` rather than a
/// single moment-matched centre.
struct TruncatedResponseRow {
    /// `(h, threshold, log σ)` at the ambient centre `β_unc`, offsets included.
    mu: [f64; 3],
    /// `C G`: how one unit of each constraint-normal coordinate moves each of
    /// the three predictor channels.
    channel_lift: Array2<f64>,
    /// Factor of the channels' residual covariance: how one unit of each tangent
    /// coordinate moves each channel.
    htl_factor: Array2<f64>,
    wiggle: Option<TruncatedWiggleRow>,
}

/// One row's link-wiggle block, conditioned on the realized channels.
struct TruncatedWiggleRow {
    regression: Array2<f64>,
    cov_cond: Array2<f64>,
    knots: Array1<f64>,
    degree: usize,
    center: Array1<f64>,
    lift: Array2<f64>,
}

impl TruncatedResponseRow {
    fn new(
        input: &SurvivalLocationScalePredictInput,
        fit: &UnifiedFitResult,
        law: &TruncatedCoefficientLaw,
        x_threshold_dense: &Array2<f64>,
        x_log_sigma_dense: &Array2<f64>,
        row: usize,
    ) -> Result<Self, String> {
        let beta_time = fit.beta_time();
        let beta_threshold = fit.beta_threshold();
        let beta_log_sigma = fit.beta_log_sigma();
        let beta_link_wiggle = fit.beta_link_wiggle();
        let p_time = beta_time.len();
        let p_t = beta_threshold.len();
        let p_ls = beta_log_sigma.len();
        let pw = beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
        let (time, threshold, log_sigma, wiggle_range) =
            survival_response_moment_block_ranges(p_time, p_t, p_ls, pw);

        let a_h = input.x_time_exit.row(row).to_owned();
        let a_t = x_threshold_dense.row(row).to_owned();
        let a_ls = x_log_sigma_dense.row(row).to_owned();

        // The AMBIENT centre of the predictor blocks: `β_unc`, not `E_π[β]`. The
        // node displacement is measured from it, and adding it to the weighted
        // node displacement reproduces `E_π[β]` by construction.
        let center_time = law.center.slice(s![time.start..time.end]).to_owned();
        let center_threshold = law
            .center
            .slice(s![threshold.start..threshold.end])
            .to_owned();
        let center_log_sigma = law
            .center
            .slice(s![log_sigma.start..log_sigma.end])
            .to_owned();
        let mu = [
            a_h.dot(&center_time) + input.eta_time_offset_exit[row],
            a_t.dot(&center_threshold) + input.eta_threshold_offset[row],
            a_ls.dot(&center_log_sigma) + input.eta_log_sigma_offset[row],
        ];

        let lift = &law.lift;
        let retained = law.normal_center.len();
        let mut channel_lift = Array2::<f64>::zeros((3, retained));
        for k in 0..retained {
            let mut h = 0.0;
            for (j, &value) in a_h.iter().enumerate() {
                h += value * lift[[time.start + j, k]];
            }
            let mut t = 0.0;
            for (j, &value) in a_t.iter().enumerate() {
                t += value * lift[[threshold.start + j, k]];
            }
            let mut l = 0.0;
            for (j, &value) in a_ls.iter().enumerate() {
                l += value * lift[[log_sigma.start + j, k]];
            }
            channel_lift[[0, k]] = h;
            channel_lift[[1, k]] = t;
            channel_lift[[2, k]] = l;
        }

        let cov_htl = projected_survival_response_moment_covariance(
            &law.residual_covariance,
            &a_h,
            &a_t,
            &a_ls,
            p_time,
            p_t,
            p_ls,
        );
        let htl_factor = factorize_psd_covariance(
            &covariance3_to_array2(cov_htl),
            "survival response-moment residual covariance",
        )?;
        let htl_rank = htl_factor.factor.ncols();
        if htl_rank + usize::from(pw > 0) > law.tangent_dimension {
            return Err(format!(
                "survival location-scale truncated response moments: row {row} needs {} tangent \
                 coordinates but the joint rule carries {}",
                htl_rank + usize::from(pw > 0),
                law.tangent_dimension
            ));
        }

        // The link-wiggle block, when present, is conditioned on the realized
        // `(h, threshold, log σ)` exactly as the Gaussian rule conditions it — the
        // affine conditional mean of a joint Gaussian — because conditional on `u`
        // the law IS a joint Gaussian with covariance `Σ_res`.
        let wiggle = match (beta_link_wiggle.as_ref(), wiggle_range) {
            (Some(_), Some(wiggle_range)) => {
                let cov_wy = {
                    let mut out = Array2::<f64>::zeros((pw, 3));
                    let cov_wh = law
                        .residual_covariance
                        .slice(s![
                            wiggle_range.start..wiggle_range.end,
                            time.start..time.end
                        ])
                        .to_owned();
                    let cov_wt = law
                        .residual_covariance
                        .slice(s![
                            wiggle_range.start..wiggle_range.end,
                            threshold.start..threshold.end
                        ])
                        .to_owned();
                    let cov_wl = law
                        .residual_covariance
                        .slice(s![
                            wiggle_range.start..wiggle_range.end,
                            log_sigma.start..log_sigma.end
                        ])
                        .to_owned();
                    out.column_mut(0).assign(&cov_wh.dot(&a_h));
                    out.column_mut(1).assign(&cov_wt.dot(&a_t));
                    out.column_mut(2).assign(&cov_wl.dot(&a_ls));
                    out
                };
                let cov_ww = law
                    .residual_covariance
                    .slice(s![
                        wiggle_range.start..wiggle_range.end,
                        wiggle_range.start..wiggle_range.end
                    ])
                    .to_owned();
                let mut regression = cov_wy.dot(&htl_factor.eigenvectors);
                for column in 0..regression.ncols() {
                    let scale = htl_factor.inv_sqrt_eigenvalues[column];
                    regression
                        .column_mut(column)
                        .mapv_inplace(|value| value * scale);
                }
                let cov_cond = symmetrize_and_clip_covariance(
                    &(cov_ww - regression.dot(&regression.t().to_owned())),
                );
                let knots = input
                    .link_wiggle_knots
                    .as_ref()
                    .or(fit.artifacts.survival_link_wiggle_knots.as_ref())
                    .ok_or_else(|| {
                        "predict_survival_location_scale: link-wiggle coefficients are missing \
                         knot metadata"
                            .to_string()
                    })?
                    .clone();
                let degree = input
                    .link_wiggle_degree
                    .or(fit.artifacts.survival_link_wiggle_degree)
                    .ok_or_else(|| {
                        "predict_survival_location_scale: link-wiggle coefficients are missing \
                         degree metadata"
                            .to_string()
                    })?;
                let center = law
                    .center
                    .slice(s![wiggle_range.start..wiggle_range.end])
                    .to_owned();
                let mut wiggle_lift = Array2::<f64>::zeros((pw, retained));
                for j in 0..pw {
                    for k in 0..retained {
                        wiggle_lift[[j, k]] = lift[[wiggle_range.start + j, k]];
                    }
                }
                Some(TruncatedWiggleRow {
                    regression,
                    cov_cond,
                    knots,
                    degree,
                    center,
                    lift: wiggle_lift,
                })
            }
            _ => None,
        };
        Ok(Self {
            mu,
            channel_lift,
            htl_factor: htl_factor.factor,
            wiggle,
        })
    }

    /// The survival probability at one node, from the node's displacement
    /// `u − E_untrunc[u]` and its tangent coordinates.
    fn survival_probability(
        &self,
        input: &SurvivalLocationScalePredictInput,
        law: &TruncatedCoefficientLaw,
        displacement: &Array1<f64>,
        tangent: &[f64],
    ) -> Result<f64, String> {
        let retained = displacement.len();
        let htl_rank = self.htl_factor.ncols();
        let mut x = self.mu;
        for (channel, value) in x.iter_mut().enumerate() {
            for k in 0..retained {
                *value += self.channel_lift[[channel, k]] * displacement[k];
            }
            for column in 0..htl_rank {
                *value += self.htl_factor[[channel, column]] * tangent[column];
            }
        }
        let q0 = survival_q0_from_eta(x[1], x[2]);
        // The scale divides the time transform too (#2695).
        let time_share = x[0] * exp_sigma_inverse_from_eta_scalar(x[2]);
        let eta = match self.wiggle.as_ref() {
            None => time_share + q0,
            Some(wiggle) => {
                let q0_arr = Array1::from_vec(vec![q0]);
                let basis = survival_wiggle_basis_with_options(
                    q0_arr.view(),
                    &wiggle.knots,
                    wiggle.degree,
                    BasisOptions::value(),
                )?;
                if basis.ncols() != wiggle.center.len() {
                    return Err(SurvivalLocationScaleError::DimensionMismatch {
                        reason: format!(
                            "predict_survival_location_scale: link-wiggle basis/beta mismatch: \
                             {} vs {}",
                            basis.ncols(),
                            wiggle.center.len()
                        ),
                    }
                    .into());
                }
                let b = basis.row(0).to_owned();
                let mut conditional_mean = wiggle.center.clone();
                for j in 0..conditional_mean.len() {
                    let mut shift = 0.0;
                    for k in 0..retained {
                        shift += wiggle.lift[[j, k]] * displacement[k];
                    }
                    for column in 0..htl_rank {
                        shift += wiggle.regression[[j, column]] * tangent[column];
                    }
                    conditional_mean[j] += shift;
                }
                let w_mean = b.dot(&conditional_mean);
                let w_variance = b.dot(&wiggle.cov_cond.dot(&b)).max(0.0);
                let w = w_mean + w_variance.sqrt() * tangent[law.tangent_dimension - 1];
                time_share + q0 + w
            }
        };
        let probability = inverse_link_survival_prob_checked(&input.inverse_link, eta)?;
        Ok(probability)
    }
}

/// Response moments for every row under the truncated law, certified on the
/// spread of the joint rule's replicate lattices (#2917).
///
/// Every replicate is extended from `N` to `2N` nodes, visiting only the new
/// nodes. A row is read and retired once the replicate standard error of its
/// `E[S]` and of its response standard error `sqrt(E[S²] − E[S]²)`, less the
/// integrand's f64 rounding, is within the law's certified relative accuracy of
/// `sqrt(E[S]·(1 − E[S]))`; the rows still uncertified are evaluated on the next
/// doubling.
///
/// That scale is the largest standard deviation a probability with this mean can
/// have (Bhatia–Davis on `[0, 1]`). It plays the role the pre-truncation standard
/// deviation plays for the law's own moments: a bound the posterior spread only
/// shrinks below, so the certificate is stated at a scale the row's moments are
/// read at, not at a scale the posterior happens to reach. Measuring against the
/// posterior spread itself does not terminate on a far-tail row: there `1 − S`
/// varies over orders of magnitude across the posterior, and the replicate
/// spread of its standard deviation falls at nearly the Monte Carlo rate with a
/// constant set by that heavy tail.
///
/// The rule this replaced stopped when two consecutive doublings of ONE lattice
/// moved no moment by more than an absolute tolerance, which cannot see bias: a
/// proposal dominated by a handful of nodes moves slowly between doublings while
/// being nowhere near the answer. Past the rule's maximum node count the moments
/// are refused, not reported.
pub(crate) fn truncated_survival_response_moments(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    x_threshold_dense: &Array2<f64>,
    x_log_sigma_dense: &Array2<f64>,
) -> Result<Option<(Array1<f64>, Array1<f64>)>, String> {
    let n = input.x_time_exit.nrows();
    let tangent_dimension = 3 + usize::from(fit.beta_link_wiggle().is_some());
    let Some(law) = build_truncated_coefficient_law(fit, covariance, tangent_dimension)? else {
        return Ok(None);
    };
    let rows = (0..n)
        .map(|row| {
            TruncatedResponseRow::new(input, fit, &law, x_threshold_dense, x_log_sigma_dense, row)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let ambient_displacement = Array1::<f64>::zeros(law.normal_center.len());
    let ambient_tangent = vec![0.0; tangent_dimension];
    let ambient_probabilities = rows
        .iter()
        .map(|row| row.survival_probability(input, &law, &ambient_displacement, &ambient_tangent))
        .collect::<Result<Vec<_>, String>>()?;
    let integrand = ResponseMomentIntegrand {
        input,
        law: &law,
        rows,
        ambient_probabilities,
    };
    let replicates = law.rule.replicates();
    if replicates < 2 {
        return Err(format!(
            "survival location-scale truncated response moments need at least two replicate \
             lattices to certify on; the joint rule carries {replicates}"
        ));
    }
    let tolerance = law.rule.relative_tolerance();
    let mut accumulators: Vec<ResponseMomentAccumulator> =
        std::iter::repeat_with(|| ResponseMomentAccumulator::new(n))
            .take(replicates)
            .collect();
    let mut first = Array1::<f64>::zeros(n);
    let mut second = Array1::<f64>::zeros(n);
    let mut active: Vec<usize> = (0..n).collect();
    let mut evaluated = 0usize;
    while !active.is_empty() {
        let target = if evaluated == 0 {
            law.rule.initial_points()
        } else {
            2 * evaluated
        };
        accumulators
            .as_mut_slice()
            .into_par_iter()
            .enumerate()
            .try_for_each(|(replicate, accumulator)| {
                law.rule.visit_nodes(
                    replicate,
                    evaluated,
                    target,
                    |log_weight, normal_coordinates, tangent| {
                        accumulator.push(&integrand, &active, log_weight, normal_coordinates, tangent)
                    },
                )
            })?;
        evaluated = target;
        let mut worst: Option<RowCertificate> = None;
        let mut uncertified = Vec::with_capacity(active.len());
        for certificate in
            certify_response_moments(&accumulators, &active, &integrand.ambient_probabilities)?
        {
            if certificate.error <= tolerance {
                first[certificate.row] = certificate.first;
                second[certificate.row] = certificate.second;
            } else {
                uncertified.push(certificate.row);
                if worst
                    .as_ref()
                    .is_none_or(|current| certificate.error > current.error)
                {
                    worst = Some(certificate);
                }
            }
        }
        active = uncertified;
        let nodes = evaluated * replicates;
        if let Some(worst) = worst
            && nodes >= law.rule.maximum_points()
        {
            return Err(format!(
                "survival location-scale truncated response moments did not certify: after \
                 {nodes} joint cubature nodes over {replicates} replicate lattices, the replicate \
                 standard error of row {}'s response moments, less the integrand's rounding, is \
                 {:.3e} of sqrt(E[S](1 - E[S])), above the law's certified relative accuracy \
                 {tolerance:.1e}",
                worst.row, worst.error
            ));
        }
    }
    Ok(Some((first, second)))
}

/// What one node's response evaluation reads, shared by every replicate.
struct ResponseMomentIntegrand<'a> {
    input: &'a SurvivalLocationScalePredictInput,
    law: &'a TruncatedCoefficientLaw,
    rows: Vec<TruncatedResponseRow>,
    /// Each row's survival probability at the ambient centre (zero displacement
    /// and zero tangent coordinates): the reference its sums run about.
    ambient_probabilities: Vec<f64>,
}

/// One replicate lattice's running sums `Σw`, and `Σw·d` and `Σw·d²` for every
/// row, where `d = S − S_ambient` is the node's survival probability less the
/// row's probability at the ambient centre, all on ONE log scale.
///
/// Every row sees the same node weights, so one rescale serves them all. The
/// weights span hundreds of decades between a barely-truncated face and a deeply
/// pinned one, so the sums carry an explicit log scale and are rescaled whenever
/// a heavier node arrives; accumulating the weights directly would underflow the
/// face to zero and leave the normalized moments as `0/0`.
///
/// The sums run about `S_ambient` rather than over `S` because the standard
/// error is read by subtraction. Running sums of `w·S` and `w·S²` each carry
/// rounding of order `ε·√N` relative to `S`, and on a row whose survival is
/// nearly certain `E[S²] − E[S]²` is then that rounding rather than the variance:
/// the replicate spread stops falling with `N` and the row cannot certify. About
/// the ambient centre both sums are of the variance's own order, so their
/// rounding is relative to it.
struct ResponseMomentAccumulator {
    log_scale: f64,
    weight_sum: f64,
    deviation_sum: Array1<f64>,
    deviation_square_sum: Array1<f64>,
}

impl ResponseMomentAccumulator {
    fn new(n: usize) -> Self {
        Self {
            log_scale: f64::NEG_INFINITY,
            weight_sum: 0.0,
            deviation_sum: Array1::zeros(n),
            deviation_square_sum: Array1::zeros(n),
        }
    }

    /// Fold one node into the sums of every row in `active`.
    fn push(
        &mut self,
        integrand: &ResponseMomentIntegrand<'_>,
        active: &[usize],
        log_weight: f64,
        normal_coordinates: &Array1<f64>,
        tangent: &[f64],
    ) -> Result<(), String> {
        if log_weight > self.log_scale {
            let rescale = (self.log_scale - log_weight).exp();
            self.weight_sum *= rescale;
            self.deviation_sum *= rescale;
            self.deviation_square_sum *= rescale;
            self.log_scale = log_weight;
        }
        let weight = (log_weight - self.log_scale).exp();
        let displacement = normal_coordinates - &integrand.law.normal_center;
        self.weight_sum += weight;
        for &row in active {
            let deviation = integrand.rows[row].survival_probability(
                integrand.input,
                integrand.law,
                &displacement,
                tangent,
            )? - integrand.ambient_probabilities[row];
            self.deviation_sum[row] += weight * deviation;
            self.deviation_square_sum[row] += weight * deviation * deviation;
        }
        Ok(())
    }
}

/// One row's pooled response moments, and the replicate standard error they rest
/// on.
struct RowCertificate {
    row: usize,
    /// `E[S]`.
    first: f64,
    /// `E[S²]`.
    second: f64,
    /// The larger replicate standard error of `E[S]` and of the response standard
    /// error, less the integrand's rounding, as a fraction of
    /// `sqrt(E[S]·(1 − E[S]))`.
    error: f64,
}

fn certify_response_moments(
    accumulators: &[ResponseMomentAccumulator],
    active: &[usize],
    ambient_probabilities: &[f64],
) -> Result<Vec<RowCertificate>, String> {
    if let Some(accumulator) = accumulators
        .iter()
        .find(|accumulator| !(accumulator.weight_sum.is_finite() && accumulator.weight_sum > 0.0))
    {
        return Err(format!(
            "survival location-scale truncated response moments: a replicate lattice \
             accumulated no finite node weight (weight sum {})",
            accumulator.weight_sum
        ));
    }
    // Pool every replicate on the heaviest replicate's scale.
    let top = accumulators
        .iter()
        .map(|accumulator| accumulator.log_scale)
        .fold(f64::NEG_INFINITY, f64::max);
    let pooling_scales: Vec<f64> = accumulators
        .iter()
        .map(|accumulator| (accumulator.log_scale - top).exp())
        .collect();
    let pooled_weight = accumulators
        .iter()
        .zip(&pooling_scales)
        .map(|(accumulator, scale)| scale * accumulator.weight_sum)
        .sum::<f64>();
    let mut means = vec![0.0; accumulators.len()];
    let mut standard_errors = vec![0.0; accumulators.len()];
    let mut certificates = Vec::with_capacity(active.len());
    for &row in active {
        let mut pooled_deviation = 0.0;
        let mut pooled_square = 0.0;
        for (replicate, (accumulator, scale)) in
            accumulators.iter().zip(&pooling_scales).enumerate()
        {
            let mean = accumulator.deviation_sum[row] / accumulator.weight_sum;
            let square = accumulator.deviation_square_sum[row] / accumulator.weight_sum;
            means[replicate] = mean;
            standard_errors[replicate] = (square - mean * mean).max(0.0).sqrt();
            pooled_deviation += scale * accumulator.deviation_sum[row];
            pooled_square += scale * accumulator.deviation_square_sum[row];
        }
        let mean_deviation = pooled_deviation / pooled_weight;
        let variance = (pooled_square / pooled_weight - mean_deviation * mean_deviation).max(0.0);
        // A weighted mean of probabilities lies in [0, 1]; the bound only removes
        // the rounding of the division.
        let first = (ambient_probabilities[row] + mean_deviation).clamp(0.0, 1.0);
        let second = (variance + first * first).clamp(0.0, 1.0);
        let spread =
            replicate_standard_error(&means).max(replicate_standard_error(&standard_errors));
        // `S = 1 − F(η)` is evaluated in f64, which resolves a probability to
        // `f64::EPSILON` absolute: a spread within that is the integrand's own
        // rounding, which no node count removes.
        let excess = spread - f64::EPSILON;
        let largest_spread = (first * (1.0 - first)).sqrt();
        let error = if excess <= 0.0 {
            0.0
        } else if largest_spread > 0.0 {
            excess / largest_spread
        } else {
            f64::INFINITY
        };
        certificates.push(RowCertificate {
            row,
            first,
            second,
            error,
        });
    }
    Ok(certificates)
}

/// Standard error of the mean of independent replicate estimates of one quantity.
pub(crate) fn replicate_standard_error(estimates: &[f64]) -> f64 {
    let replicates = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / replicates;
    let spread = estimates
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (replicates - 1.0);
    (spread / replicates).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::gauge::Gauge;
    use ndarray::array;

    /// Knots and degree for a link-wiggle block of the requested width, taken
    /// from the production knot and block builders so the basis is the one
    /// predict uses.
    fn wiggle_metadata(width: usize) -> (Array1<f64>, usize) {
        let seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        for degree in [2usize, 3, 1] {
            for num_internal_knots in 0..=8 {
                if let Ok(knots) = crate::wiggle::monotone_warp_knots_from_seed(
                    seed.view(),
                    degree,
                    num_internal_knots,
                ) && let Ok(block) = crate::wiggle::buildwiggle_block_input_from_orders(
                    seed.view(),
                    &knots,
                    degree,
                    &[2],
                    false,
                ) && block.design.ncols() == width
                {
                    return (knots, degree);
                }
            }
        }
        panic!("could not synthesize link-wiggle metadata for {width} coefficients");
    }

    /// #2679: the response-moment integral must be taken against the
    /// CONE-TRUNCATED posterior, not against the normal that carries its first
    /// two moments.
    ///
    /// The fixture is the regime the defect lives in and nothing else: a
    /// two-coefficient monotone link-wiggle block whose ambient centre straddles
    /// both walls, so BOTH constraint rows are retained and the pushforward of
    /// the truncated joint through the basis row is genuinely non-normal. The
    /// threshold and log-sigma blocks carry no variance, which makes `q0` and
    /// the basis row `b` deterministic — that is what lets the reference be
    /// built in closed form, and it costs nothing, because the quantity under
    /// test is the WARP's law.
    ///
    /// The reference integrates the exact truncated density directly:
    /// tensor Simpson over the retained cone for `β_w`, and, inside it, a
    /// Gauss-Legendre rule for the conditional Gaussian law of `h | β_w`. It
    /// calls neither cubature. The comparison arm is the shipped rule, obtained
    /// by handing the SAME coefficients and the SAME reported covariance to a
    /// fit carrying no cone geometry — one variable changed, nothing else.
    #[test]
    fn truncated_response_moments_beat_the_moment_matched_normal_2679() {
        let (base_knots, degree) = wiggle_metadata(2);

        // Blocks: time(0..2), threshold(2..4), log_sigma(4..6), wiggle(6..8).
        // The `(time, wiggle)` sub-block is `F Fᵀ`; the threshold and log-sigma
        // rows are exactly zero.
        let f = array![
            [0.30, 0.00, 0.00, 0.00],
            [0.10, 0.25, 0.00, 0.00],
            [0.22, 0.12, 0.26, 0.00],
            [0.18, 0.14, 0.10, 0.24],
        ];
        let block = f.dot(&f.t());
        let mut ambient = Array2::<f64>::zeros((8, 8));
        for i in 0..2 {
            for j in 0..2 {
                ambient[[i, j]] = block[[i, j]];
                ambient[[i, 6 + j]] = block[[i, 2 + j]];
                ambient[[6 + j, i]] = block[[2 + j, i]];
                ambient[[6 + i, 6 + j]] = block[[2 + i, 2 + j]];
            }
        }

        // Ambient centre. The wiggle coordinates straddle the wall in units of
        // their own standard deviations, which is what retains both rows.
        let center = array![0.40, -0.10, 0.20, 0.30, -0.50, 0.10, 0.04, -0.09];
        let constraints = gam_problem::LinearInequalityConstraints::new(
            array![
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ],
            array![0.0, 0.0],
        )
        .expect("two-row link-wiggle non-negativity cone");
        let correction =
            constrained_posterior_correction_from_covariance(&ambient, &center, &constraints)
                .expect("the correction is computable on this face")
                .expect("a centre straddling both walls must retain the face");
        let mut retained = correction.rows.clone();
        retained.sort_unstable();
        assert_eq!(
            retained,
            vec![0, 1],
            "the fixture must retain BOTH wiggle rows; a one-row face has a closed-form \
             pushforward and would measure nothing about normality"
        );

        let sigma_pi = correction.apply_to_covariance(&ambient);
        let beta_pi = correction.posterior_mean(&center);
        let beta_w = beta_pi.slice(s![6..8]).to_owned();
        assert!(
            beta_w.iter().all(|&value| value > 0.0),
            "the truncated posterior mean must be interior to the cone, got {beta_w:?}"
        );

        let a_h = array![1.0, 0.5];
        // Row 0 is the fixture. Rows 1-3 are row 0 with its exit predictor moved
        // 2.7, 4.7 and 6.7 units into the survival tail, where `1 − S` is
        // near-certain and varies over orders of magnitude across the posterior
        // (#2917).
        let x_threshold_dense = array![[1.0, -0.2], [1.0, -0.2], [1.0, -0.2], [1.0, -0.2]];
        let x_log_sigma_dense = array![[1.0, 0.3], [1.0, 0.3], [1.0, 0.3], [1.0, 0.3]];
        let eta_time_offset_exit = array![0.2, -2.5, -4.5, -6.5];
        let eta_threshold_offset = array![0.7, 0.7, 0.7, 0.7];
        let eta_log_sigma_offset = array![0.4, 0.4, 0.4, 0.4];
        let mu_t =
            x_threshold_dense.row(0).dot(&beta_pi.slice(s![2..4])) + eta_threshold_offset[0];
        let mu_ls =
            x_log_sigma_dense.row(0).dot(&beta_pi.slice(s![4..6])) + eta_log_sigma_offset[0];
        let q0 = survival_q0_from_eta(mu_t, mu_ls);

        // Re-centre the knots on the realized `q0` so BOTH I-spline columns
        // carry weight there; a zero basis row would make the warp inert.
        let lo = base_knots.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = base_knots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let knots = base_knots.mapv(|knot| knot + (q0 - 0.5 * (lo + hi)));
        let basis = survival_wiggle_basis_with_options(
            Array1::from_vec(vec![q0]).view(),
            &knots,
            degree,
            BasisOptions::value(),
        )
        .expect("link wiggle basis");
        let b = basis.row(0).to_owned();
        assert!(
            b[0] > 1.0e-3 && b[1] > 1.0e-3,
            "both wiggle coordinates must carry basis weight at q0, got {b:?}"
        );

        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5]],
            eta_time_offset_exit,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                x_threshold_dense.clone(),
            )),
            eta_threshold_offset,
            x_log_sigma: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                x_log_sigma_dense.clone(),
            )),
            eta_log_sigma_offset,
            x_link_wiggle: Some(DesignMatrix::Dense(
                gam_linalg::matrix::DenseDesignMatrix::from(Array2::from_shape_fn(
                    (4, basis.ncols()),
                    |index| basis[[0, index.1]],
                )),
            )),
            link_wiggle_knots: Some(knots.clone()),
            link_wiggle_degree: Some(degree),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };

        let make_fit = |geometry: Option<FitGeometry>| -> UnifiedFitResult {
            let mut fit = survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
                training_sample_size: 32,
                log_lambdas: Array1::zeros(0),
                beta_time: beta_pi.slice(s![0..2]).to_owned(),
                beta_threshold: beta_pi.slice(s![2..4]).to_owned(),
                beta_log_sigma: beta_pi.slice(s![4..6]).to_owned(),
                beta_link_wiggle: Some(beta_w.clone()),
                link_wiggle_knots: Some(knots.clone()),
                link_wiggle_degree: Some(degree),
                lambdas_time: Array1::zeros(0),
                lambdas_threshold: Array1::zeros(0),
                lambdas_log_sigma: Array1::zeros(0),
                lambdas_linkwiggle: Some(Array1::zeros(0)),
                log_likelihood: 0.0,
                reml_score: Some(0.0),
                stable_penalty_term: 0.0,
                penalized_objective: Some(0.0),
                used_device: false,
                outer_iterations: 0,
                outer_gradient_norm: None,
                criterion_certificate: None,
                outer_converged: true,
                covariance_conditional: Some(sigma_pi.clone()),
                covariance_corrected: None,
                smoothing_correction: None,
                smoothing_correction_absence: None,
                geometry,
                penalty_block_trace: Vec::new(),
                edf_by_block: Vec::new(),
                edf_rank_bound: Vec::new(),
                coefficient_mode_selection: Default::default(),
            })
            .expect("valid survival test fit");
            fit.covariance_conditional = Some(sigma_pi.clone());
            fit
        };

        let cone_geometry = FitGeometry {
            coefficient_gauge: Gauge::identity(&[2, 2, 2, 2]),
            penalized_hessian: Array2::<f64>::eye(8).into(),
            constrained_posterior: Some(ConstrainedPosteriorGeometry::with_moments(
                constraints,
                beta_pi.clone(),
                center.clone(),
                Some(correction.clone()),
            )),
            working: None,
        };
        let truncated_fit = make_fit(Some(cone_geometry));
        let gaussian_fit = make_fit(None);

        // Reference. `β_w` is truncated to the cone; `h | β_w` is Gaussian with
        // the affine conditional mean of the AMBIENT joint; `η = s·h + q0 + bᵀβ_w`
        // with the deterministic scale `s = e^{−μ_ls}`, which divides the time
        // transform too (#2695).
        let time_scale = exp_sigma_inverse_from_eta_scalar(mu_ls);
        let sigma_ww = ambient.slice(s![6..8, 6..8]).to_owned();
        let sigma_hw = a_h.dot(&ambient.slice(s![0..2, 6..8]));
        let var_h_ambient = a_h.dot(&ambient.slice(s![0..2, 0..2]).dot(&a_h));
        let det = sigma_ww[[0, 0]] * sigma_ww[[1, 1]] - sigma_ww[[0, 1]] * sigma_ww[[1, 0]];
        let inverse_ww = array![
            [sigma_ww[[1, 1]] / det, -sigma_ww[[0, 1]] / det],
            [-sigma_ww[[1, 0]] / det, sigma_ww[[0, 0]] / det]
        ];
        let regression = inverse_ww.dot(&sigma_hw);
        let conditional_var = (var_h_ambient - sigma_hw.dot(&regression)).max(0.0);
        assert!(
            conditional_var > 0.0,
            "the fixture must leave `h` a live tangent direction after conditioning on the cone"
        );
        // `mu_h` is centred on the REPORTED mean, so put the reference on the
        // ambient centre it belongs to before adding the conditional shift.
        let mu_h_ambient = a_h.dot(&center.slice(s![0..2])) + input.eta_time_offset_exit[0];
        let center_w = center.slice(s![6..8]).to_owned();
        let (gl_nodes, gl_weights) = gam_math::special::gauss_legendre(32);
        let conditional_sd = conditional_var.sqrt();
        let inner = |eta_center: f64| -> (f64, f64) {
            let half = 10.0 * conditional_sd;
            let mut first = 0.0;
            let mut second = 0.0;
            let mut mass = 0.0;
            for (node, weight) in gl_nodes.iter().zip(gl_weights.iter()) {
                let offset = half * node;
                let standardized = offset / conditional_sd;
                let quadrature = half * weight * (-0.5 * standardized * standardized).exp();
                let probability = inverse_link_survival_prob_checked(
                    &input.inverse_link,
                    eta_center + time_scale * offset,
                )
                .expect("inverse link");
                first += quadrature * probability;
                second += quadrature * probability * probability;
                mass += quadrature;
            }
            (first / mass, second / mass)
        };
        let density = |w0: f64, w1: f64| -> f64 {
            let d0 = w0 - center_w[0];
            let d1 = w1 - center_w[1];
            let quadratic = inverse_ww[[0, 0]] * d0 * d0
                + 2.0 * inverse_ww[[0, 1]] * d0 * d1
                + inverse_ww[[1, 1]] * d1 * d1;
            (-0.5 * quadratic).exp()
        };
        let upper0 = center_w[0].max(0.0) + 12.0 * sigma_ww[[0, 0]].sqrt();
        let upper1 = center_w[1].max(0.0) + 12.0 * sigma_ww[[1, 1]].sqrt();
        let grid = 401;
        let mut reference_mass = 0.0;
        let mut reference_first = 0.0;
        let mut reference_second = 0.0;
        {
            let mut accumulate = |scale: f64, w0: f64, w1: f64| {
                let value = density(w0, w1);
                let shift = regression[0] * (w0 - center_w[0]) + regression[1] * (w1 - center_w[1]);
                let eta_center = time_scale * (mu_h_ambient + shift) + q0 + b[0] * w0 + b[1] * w1;
                let (m1, m2) = inner(eta_center);
                reference_mass += scale * value;
                reference_first += scale * value * m1;
                reference_second += scale * value * m2;
            };
            let step0 = upper0 / ((grid - 1) as f64);
            let step1 = upper1 / ((grid - 1) as f64);
            for i in 0..grid {
                let weight0 = if i == 0 || i == grid - 1 {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                for j in 0..grid {
                    let weight1 = if j == 0 || j == grid - 1 {
                        1.0
                    } else if j % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    accumulate(
                        weight0 * weight1,
                        step0 * (i as f64),
                        step1 * (j as f64),
                    );
                }
            }
        }
        let reference_first = reference_first / reference_mass;
        let reference_second = reference_second / reference_mass;

        let (gaussian_mean, gaussian_second) =
            exact_survival_response_moments(&input, &gaussian_fit, &sigma_pi)
                .expect("moment-matched normal response moments");
        let (truncated_mean, truncated_second) =
            exact_survival_response_moments(&input, &truncated_fit, &sigma_pi)
                .expect("truncated response moments");

        let gaussian_error = (gaussian_mean[0] - reference_first).abs();
        let truncated_error = (truncated_mean[0] - reference_first).abs();
        let gaussian_error_second = (gaussian_second[0] - reference_second).abs();
        let truncated_error_second = (truncated_second[0] - reference_second).abs();
        eprintln!(
            "[2679] reference E[S]={reference_first:.12e} E[S^2]={reference_second:.12e}; \
             moment-matched normal {:.12e}/{:.12e} (err {gaussian_error:.3e}/\
             {gaussian_error_second:.3e}); truncated law {:.12e}/{:.12e} (err \
             {truncated_error:.3e}/{truncated_error_second:.3e})",
            gaussian_mean[0], gaussian_second[0], truncated_mean[0], truncated_second[0]
        );

        // Non-vacuity: the two arms must be free to disagree, and the arm under
        // test must be reached at all. If the shipped rule already matched the
        // exact pushforward here, the fixture would prove nothing.
        assert!(
            gaussian_error > 1.0e-4,
            "the moment-matched normal must be measurably wrong on this fixture or the \
             comparison is vacuous; got {gaussian_error:.3e}"
        );
        assert!(
            (truncated_mean[0] - gaussian_mean[0]).abs() > 1.0e-6,
            "the two arms produced the same number, so the truncated rule was not reached"
        );

        assert!(
            truncated_error < 0.25 * gaussian_error,
            "the truncated rule must be decisively closer to the exact pushforward than the \
             moment-matched normal: {truncated_error:.3e} vs {gaussian_error:.3e} against \
             reference {reference_first:.12e}"
        );
        assert!(
            truncated_error_second < 0.25 * gaussian_error_second,
            "the same must hold for the second moment, which is what \
             `response_standard_error` reports: {truncated_error_second:.3e} vs \
             {gaussian_error_second:.3e} against reference {reference_second:.12e}"
        );

        // The tail rows certified rather than refused: the `expect` above is where
        // a refusal would surface. Each is a probability that survives at least as
        // surely as row 0, whose `1 − E[S]` is orders of magnitude larger than any
        // integration error the certificate admits.
        for row in 1..4 {
            let response_standard_error = (truncated_second[row]
                - truncated_mean[row] * truncated_mean[row])
                .max(0.0)
                .sqrt();
            eprintln!(
                "[2917] row {row}: truncated E[S]={:.15e} 1-E[S]={:.3e} sd={:.3e}; \
                 moment-matched normal E[S]={:.15e}",
                truncated_mean[row],
                1.0 - truncated_mean[row],
                response_standard_error,
                gaussian_mean[row]
            );
            assert!(
                truncated_mean[row] <= 1.0 && truncated_mean[row] > truncated_mean[0],
                "row {row}: E[S]={:.15e} must lie in (E[S] of row 0 = {:.15e}, 1]",
                truncated_mean[row],
                truncated_mean[0]
            );
        }
    }

    /// #3524: under the smoothing-corrected covariance a cone-truncated fit's
    /// coefficient posterior is the truncation of the ρ-marginal
    /// `N(β_unc, Σ + C)`. The truncated rule must be built for that law. It
    /// used to be declined, which sent `--covariance-mode corrected` to the
    /// Gaussian rule on a law the fit truncated.
    ///
    /// The pieces must be that law's own. Its lift `G_c = (Σ+C)AᵀW_c⁻¹` and its
    /// constraint-normal spread `W_c = A(Σ+C)Aᵀ` are rebuilt at `Σ + C`, not
    /// read from the stored conditional moments. The certificate is the law's
    /// second moment: `Σ_res + G_c(W_c − Δ_c)G_cᵀ` must reproduce the published
    /// `V_c` for the corrected law, exactly as it reproduces `Σ_π` for the
    /// conditional one. A covariance that is neither reported matrix is the
    /// second moment of no law this fit defines, and must be refused rather
    /// than integrated as a Gaussian.
    #[test]
    fn smoothing_corrected_covariance_gets_its_own_truncated_law_3524() {
        let (knots, degree) = wiggle_metadata(2);
        let f = array![
            [0.30, 0.00, 0.00, 0.00],
            [0.10, 0.25, 0.00, 0.00],
            [0.22, 0.12, 0.26, 0.00],
            [0.18, 0.14, 0.10, 0.24],
        ];
        let block = f.dot(&f.t());
        let mut ambient = Array2::<f64>::zeros((8, 8));
        for i in 0..2 {
            for j in 0..2 {
                ambient[[i, j]] = block[[i, j]];
                ambient[[i, 6 + j]] = block[[i, 2 + j]];
                ambient[[6 + j, i]] = block[[2 + j, i]];
                ambient[[6 + i, 6 + j]] = block[[2 + i, 2 + j]];
            }
        }
        let center = array![0.40, -0.10, 0.20, 0.30, -0.50, 0.10, 0.04, -0.09];
        let constraints = gam_problem::LinearInequalityConstraints::new(
            array![
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ],
            array![0.0, 0.0],
        )
        .expect("two-row link-wiggle non-negativity cone");
        let conditional =
            constrained_posterior_correction_from_covariance(&ambient, &center, &constraints)
                .expect("the conditional correction is computable on this face")
                .expect("a centre straddling both walls must retain the face");
        let sigma_pi = conditional.apply_to_covariance(&ambient);
        let beta_pi = conditional.posterior_mean(&center);

        // `C = K Kᵀ`, PSD as the first-order `J·V_ρ·Jᵀ` is, loading on the time
        // AND the wiggle blocks so that it moves both the lift and the
        // constraint-normal spread.
        let k = array![
            [0.20, 0.00],
            [0.05, 0.10],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.12, 0.04],
            [0.03, 0.15],
        ];
        let smoothing = k.dot(&k.t());
        let marginal_ambient = &ambient + &smoothing;
        let marginal = constrained_posterior_correction_from_covariance(
            &marginal_ambient,
            &center,
            &constraints,
        )
        .expect("the corrected correction is computable on this face")
        .expect("a wider ambient keeps both rows retained");
        let corrected = marginal.apply_to_covariance(&marginal_ambient);

        let mut fit = survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
            training_sample_size: 32,
            log_lambdas: Array1::zeros(0),
            beta_time: beta_pi.slice(s![0..2]).to_owned(),
            beta_threshold: beta_pi.slice(s![2..4]).to_owned(),
            beta_log_sigma: beta_pi.slice(s![4..6]).to_owned(),
            beta_link_wiggle: Some(beta_pi.slice(s![6..8]).to_owned()),
            link_wiggle_knots: Some(knots),
            link_wiggle_degree: Some(degree),
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_linkwiggle: Some(Array1::zeros(0)),
            log_likelihood: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            outer_iterations: 0,
            outer_gradient_norm: None,
            criterion_certificate: None,
            outer_converged: true,
            covariance_conditional: Some(sigma_pi.clone()),
            covariance_corrected: Some(corrected.clone()),
            smoothing_correction: Some((
                smoothing.clone(),
                gam_solve::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                    active_rank: 2,
                    rho_dimension: 2,
                },
            )),
            smoothing_correction_absence: None,
            geometry: Some(FitGeometry {
                coefficient_gauge: Gauge::identity(&[2, 2, 2, 2]),
                penalized_hessian: Array2::<f64>::eye(8).into(),
                constrained_posterior: Some(ConstrainedPosteriorGeometry::with_moments(
                    constraints.clone(),
                    beta_pi.clone(),
                    center.clone(),
                    Some(conditional.clone()),
                )),
                working: None,
            }),
            penalty_block_trace: Vec::new(),
            edf_by_block: Vec::new(),
            edf_rank_bound: Vec::new(),
            coefficient_mode_selection: Default::default(),
        })
        .expect("valid survival test fit");
        fit.covariance_conditional = Some(sigma_pi.clone());
        fit.covariance_corrected = Some(corrected.clone());

        let max_gap = |left: &Array2<f64>, right: &Array2<f64>| {
            left.iter()
                .zip(right.iter())
                .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
        };
        // `Σ_res + G(W − Δ)Gᵀ`, the second moment the pieces carry.
        let second_moment = |pieces: &TruncatedLawPieces, removed: &Array2<f64>| {
            let spread = &pieces.normal_covariance - removed;
            &pieces.residual_covariance + &pieces.lift.dot(&spread).dot(&pieces.lift.t())
        };

        let conditional_pieces = truncated_law_pieces(&fit, &sigma_pi)
            .expect("conditional law pieces")
            .expect("the conditional law retains the face");
        let corrected_pieces = truncated_law_pieces(&fit, &corrected)
            .expect("corrected law pieces")
            .expect(
                "the smoothing-corrected law of a cone-truncated fit is truncated too; it must \
                 not be declined into the Gaussian rule",
            );

        let conditional_gap = max_gap(
            &second_moment(&conditional_pieces, &conditional.removed_normal_variance),
            &sigma_pi,
        );
        let corrected_gap = max_gap(
            &second_moment(&corrected_pieces, &marginal.removed_normal_variance),
            &corrected,
        );
        assert!(
            conditional_gap <= 1e-12 && corrected_gap <= 1e-12,
            "each law's pieces must reproduce its own published covariance: conditional gap \
             {conditional_gap:.3e}, corrected gap {corrected_gap:.3e}"
        );

        // The corrected law carries its own lift and constraint-normal spread,
        // and they are not the stored conditional ones.
        let lift_gap = max_gap(&corrected_pieces.lift, &marginal.lift);
        assert!(
            lift_gap <= 1e-12,
            "the corrected lift must be G_c = (Σ+C)AᵀW_c⁻¹, off by {lift_gap:.3e}"
        );
        let expected_spread = constraints
            .a
            .dot(&marginal_ambient)
            .dot(&constraints.a.t());
        let spread_gap = max_gap(&corrected_pieces.normal_covariance, &expected_spread);
        assert!(
            spread_gap <= 1e-12,
            "the corrected constraint-normal spread must be A(Σ+C)Aᵀ, off by {spread_gap:.3e}"
        );
        let moved = max_gap(&corrected_pieces.lift, &conditional_pieces.lift);
        assert!(
            moved > 1e-2,
            "the fixture's C must move the lift, or the test measures nothing (moved {moved:.3e})"
        );

        assert!(
            build_truncated_coefficient_draws(&fit, &corrected)
                .expect("corrected draws")
                .is_some(),
            "the corrected covariance must reach the truncated-law survival rule"
        );

        let foreign = &corrected * 1.5;
        let refusal = truncated_law_pieces(&fit, &foreign).err().expect(
            "a covariance that is neither the conditional nor the corrected one must be refused",
        );
        assert!(
            refusal.contains("neither the fit's conditional nor its smoothing-corrected"),
            "unexpected refusal: {refusal}"
        );
    }
}
