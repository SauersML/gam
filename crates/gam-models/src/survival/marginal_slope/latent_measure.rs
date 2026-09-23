//! The automatic latent-score measure gate for the survival marginal-slope
//! family (gam#2768).
//!
//! The Bernoulli marginal-slope family has run an automatic gate on its latent
//! score since #905: a Rao score test on `E[z|C]` and `Var(z|C)` over the
//! marginal-index span, escalating to the conditional location-scale correction
//! `ζ = (z − m(C))/√v(C)` when it fires, with rank inverse-normal and empirical
//! fallbacks below it. The survival marginal-slope family ran *none* of it. It
//! called `standardize_latent_z_with_policy` and nothing else, and under the
//! default policy — `Frozen { mean: 0, sd: 1 }` — that transform is the
//! identity: it checked, it warned, and it passed z through unchanged.
//!
//! That is not a cosmetic gap. The survival row index is
//! `η = q·c(g) + s(g)·z`, so a conditional shift `E[z|C] = m(C) ≠ 0` puts
//! `s(g(C))·m(C)` into the *influence* channel `q` — the same `b(C)·m(C)`
//! leakage the Bernoulli gate exists to remove, in a model whose whole point is
//! that `q` is the marginal index. The pooled marginal gate cannot see it (the
//! marginal law of z can be exactly N(0,1) while every conditional law is
//! shifted), and rank-INT provably cannot fix it (no transform depending only on
//! the marginal `F_Z` can enforce `E[T(Z)|C] ≡ const`).
//!
//! This module is the survival caller of the *shared* gate
//! ([`build_latent_measure_decision`]), not a second copy of it. The one thing
//! it declares that BMS does not is the family's kernel capability. Since
//! gam#2923 the survival row program owns a declared-law branch — the anchored
//! frame ([`AnchoredStaticSlopeGeometry`]), which solves the marginal identity
//! on a finite law instead of lowering it in closed form — so on every
//! configuration that frame serves the gate is asked for
//! [`EmpiricalLatentMeasureSupport::Available`] exactly as the Bernoulli family
//! asks, and a sample no pre-transform makes adequately normal reaches the
//! kernel as the exact empirical latent measure. On the configurations the
//! anchored frame does not yet serve (listed by
//! [`anchored_kernel_unavailable_reason`]) the gate is still asked for
//! [`EmpiricalLatentMeasureSupport::StandardNormalOnly`] and its residual
//! verdict is routed through the spec's own [`LatentZCheckMode`].

use super::*;

use crate::bms::{
    EmpiricalLatentMeasureSupport, LatentLawConsumed, LatentMeasureCalibration, LatentMeasureKind,
    LatentMeasureSpec, LatentZConditionalCalibration, MovingLawArm, build_latent_measure_decision,
    estimated_latent_law,
};
use crate::inference::predict_io::FittedLatentScoreMap;

/// Everything the fit and its persistence need from the latent-law gate: the
/// per-coordinate decisions, the law the primary score consumed, the score the
/// gate saw *before* calibrating it, and the conditioning block it conditioned
/// on.
///
/// The raw score and the conditioning block are not diagnostics. The
/// Murphy-Topel generated-regressor correction needs both — it differentiates
/// the first stage, which regressed the RAW score on that block — and the
/// conditioning block is also what the persistence gate compares against the
/// finally-resolved marginal design to decide whether prediction can reproduce
/// the map at all.
pub(crate) struct SurvivalLatentScoreCalibration {
    /// One decision per latent-score column, in column order.
    pub(crate) per_score: Vec<LatentMeasureCalibration>,
    /// The measure the kernel integrates against, one per latent-score column
    /// (gam#2923). `StandardNormal` is the Gaussian closed form; an empirical
    /// kind is the finite law the anchored frame solves the identity on.
    pub(crate) per_score_measure: Vec<LatentMeasureKind>,
    /// Which law the PRIMARY score consumed (gam#2926), persisted with the model.
    pub(crate) consumed: LatentLawConsumed,
    /// Each score's estimated law on its own axis where its provisional closed
    /// form is certified at the converged fit (gam#2926), one per column; `None`
    /// for every other law. With one score the certificate is taken on this law;
    /// with several it is taken on their joint law, and these are what a re-solve
    /// persists per score.
    pub(crate) certificate_laws: Vec<Option<crate::bms::EmpiricalZGrid>>,
    /// With one score whose law moves on the span, the moving-law certificate's
    /// candidates (gam#2926), taken at the converged fit; `None` otherwise.
    pub(crate) moving_law: Option<Box<crate::bms::moving_law_rule::MovingLawCandidates>>,
    /// The (normalised, pre-calibration) latent scores the gate was handed.
    pub(crate) raw_scores: Array2<f64>,
    /// The conditioning span `a(C)` the conditional branch used, when it was
    /// built. `None` when the CTN Stage-1 absorber suppressed it.
    pub(crate) conditioning: Option<std::sync::Arc<Array2<f64>>>,
}

impl SurvivalLatentScoreCalibration {
    /// The conditional location-scale calibration on the PRIMARY score, if the
    /// Rao gate escalated to one. This is the only branch with a generated
    /// regressor: rank-INT is a fixed monotone map of the marginal ECDF and the
    /// identity is not estimated at all.
    pub(crate) fn primary_conditional(&self) -> Option<&LatentZConditionalCalibration> {
        match self.per_score.first() {
            Some(LatentMeasureCalibration::ConditionalLocationScale(cal)) => Some(cal),
            _ => None,
        }
    }

    /// The measure of the PRIMARY score: the object the fit persists and the
    /// anchored frame is built on.
    pub(crate) fn primary_measure(&self) -> &LatentMeasureKind {
        self.per_score_measure
            .first()
            .expect("a latent-score calibration carries at least one column")
    }
}

/// Why the anchored frame cannot serve a term spec, if it cannot.
///
/// Each item is a real combination to support and a separate piece of chain
/// rule; refusing by name is honest, whereas running the closed form under a
/// declared law would silently fit a different model from the one requested.
pub(crate) fn anchored_kernel_unavailable_reason(
    spec: &SurvivalMarginalSlopeTermSpec,
) -> Option<&'static str> {
    let score_dim = spec.z.ncols();
    if score_dim != 1 {
        // gam#2929: with K ≥ 2 scores the anchor reads the law of the drive
        // `rᵀz`, which the fit builds as the joint law of the score vector and
        // projects onto each row's own slope vector.
        if spec.declared_latent_law.is_some() {
            return Some(
                "a declared latent law is a finite law of ONE score: with K ≥ 2 scores the \
                 anchor needs the joint law of the score vector, which latent_measure = \
                 global-empirical builds from the training scores",
            );
        }
        let per_score = spec
            .slopespecs
            .as_ref()
            .is_some_and(|specs| specs.len() == score_dim);
        if !per_score {
            return Some(
                "a latent law over K ≥ 2 scores is served on the per-score slope topology (one \
                 slope surface per score): a slope shared across the scores reads only their \
                 sum, and the anchored scalar frame does not build that sum's law",
            );
        }
        // gam#2948: the flex row program solves each timepoint's intercept on the
        // family's own law, and it reads that law as ONE score's scalar grid
        // (`SurvivalLatentLaw::scalar_grid`). A K ≥ 2 fit anchors on the joint law
        // of the score vector, which carries no scalar grid, so the combination is
        // refused here rather than at the first row solve.
        if spec.score_warp.is_some() || spec.link_dev.is_some() {
            return Some(
                "a score-warp or link-deviation flex block anchors each timepoint on the \
                 scalar law of ONE score: with K ≥ 2 scores the fit anchors on the joint law \
                 of the score vector, which carries no scalar grid for the flex row program",
            );
        }
    }
    if spec
        .score_influence_jacobian
        .as_ref()
        .is_some_and(|jacobian| jacobian.ncols() > 0)
    {
        return Some(
            "a declared latent law is not yet supported together with a CTN Stage-1 \
             influence absorber: the absorber shifts the de-nested index through the \
             closed-form frame's trailing primary",
        );
    }
    if !matches!(
        spec.slope_template,
        SurvivalCovariateTermBlockTemplate::Static
    ) {
        return Some(
            "a declared latent law is not yet supported together with a follow-up-varying \
             slope: the anchored frame carries the slope on one channel, and the rate of the \
             anchor along follow-up under a moving slope is its own piece of chain rule",
        );
    }
    if spec.timewiggle_block.is_some() {
        return Some(
            "a declared latent law is not yet supported together with a time-wiggle \
             baseline: the wiggle's coefficient calculus runs the flex row program",
        );
    }
    None
}

/// Run the latent-law gate over every latent-score coordinate and replace
/// `spec.z` by the calibrated score in place (only the declared conditional
/// location-scale law calibrates; every other law leaves the score as given).
///
/// Returns one [`LatentMeasureCalibration`] per z column, in column order, for
/// persistence: prediction MUST apply the identical map, so the fit's decision
/// travels with the model rather than being re-derived from the prediction
/// sample.
///
/// # Why per coordinate
///
/// With `K > 1` latent scores the row index is `η = q·c + Σ_k s(g_k)·z_k`, so
/// the leakage is `Σ_k s(g_k(C))·m_k(C)` — a sum of per-coordinate conditional
/// shifts, and each coordinate's law is judged on the same span `a(C)`. (A
/// conditional `Σ(C)` is a different and larger question; it is gam#2766.)
pub(crate) fn resolve_survival_latent_score_calibration(
    spec: &mut SurvivalMarginalSlopeTermSpec,
    marginal_design: &TermCollectionDesign,
    data: ArrayView2<'_, f64>,
) -> Result<SurvivalLatentScoreCalibration, String> {
    // #461 seam, mirrored from BMS: when a CTN Stage-1 influence absorber is
    // active the conditional leakage is already absorbed by the absorber's own
    // orthogonalisation, and replacing z here would perturb the widened-marginal
    // predict seam. Neither the span test nor a local law is then engaged; the
    // pooled checks still are, because they are about the marginal law of z and
    // the absorber says nothing about that.
    let absorber_active = spec
        .score_influence_jacobian
        .as_ref()
        .is_some_and(|jacobian| jacobian.ncols() > 0);
    // gam#2923/gam#2926: a declared finite law, the pooled empirical law and the
    // location-scale law are finite laws the anchored frame carries, and the
    // default is one only where the score's law departs from the Gaussian law.
    // Where the frame does not serve the spec a finite law is refused by the
    // frame's own reason, never replaced by the closed form.
    let finite_law_requested = spec.declared_latent_law.is_some()
        || !matches!(
            spec.latent_z_policy.latent_measure,
            LatentMeasureSpec::StandardNormal | LatentMeasureSpec::Auto { .. }
        );
    let frame_unavailable = anchored_kernel_unavailable_reason(spec);
    let support = match frame_unavailable {
        None => EmpiricalLatentMeasureSupport::Available,
        Some(reason) => {
            if finite_law_requested {
                return Err(format!(
                    "survival marginal-slope was asked to anchor on a finite latent law, but \
                     {reason} (gam#2926)"
                ));
            }
            EmpiricalLatentMeasureSupport::StandardNormalOnly
        }
    };
    if let Some(grid) = spec.declared_latent_law.as_ref() {
        // A declared law is the caller's statement about the score AS GIVEN:
        // the gate does not run, no pre-transform is fitted, and the law is
        // what the fit anchors on and persists.
        let kind = LatentMeasureKind::GlobalEmpirical { grid: grid.clone() };
        kind.validate("survival marginal-slope declared latent law")?;
        log::debug!(
            "[survival-marginal-slope latent-z] the row index is anchored on a DECLARED latent \
             law of {} nodes; the automatic gate is not run (gam#2923)",
            grid.nodes.len(),
        );
        let k = spec.z.ncols();
        return Ok(SurvivalLatentScoreCalibration {
            per_score: vec![LatentMeasureCalibration::None; k],
            per_score_measure: vec![kind; k],
            consumed: LatentLawConsumed::DeclaredFiniteLaw {
                nodes: grid.nodes.len(),
            },
            certificate_laws: vec![None; k],
            moving_law: None,
            raw_scores: spec.z.clone(),
            conditioning: None,
        });
    }
    let context_cols =
        estimated_latent_law::marginal_formula_context_columns(&spec.marginalspec, data.ncols())?;
    let context_features = data.select(ndarray::Axis(1), &context_cols);
    let local_context = (!absorber_active && !context_cols.is_empty()).then(|| {
        estimated_latent_law::LocalLawContext {
            features: context_features.view(),
            feature_cols: context_cols.clone(),
        }
    });
    let resolved = resolve_latent_score_calibration_from_parts(
        &spec.z,
        &spec.weights,
        &spec.latent_z_policy,
        absorber_active,
        &marginal_design.design,
        local_context.as_ref(),
        support,
    )
    .map_err(|error| match frame_unavailable {
        Some(reason) => format!("{error}; the anchored frame is unavailable here because {reason}"),
        None => error,
    })?;
    spec.z = resolved.calibrated_scores;
    let consumed = resolved
        .per_score_consumed
        .into_iter()
        .next()
        .ok_or_else(|| "survival marginal-slope latent-law gate saw no score column".to_string())?;
    Ok(SurvivalLatentScoreCalibration {
        per_score: resolved.per_score,
        per_score_measure: resolved.per_score_measure,
        consumed,
        certificate_laws: resolved.per_score_certificate_law,
        moving_law: resolved.moving_law,
        raw_scores: resolved.raw_scores,
        conditioning: resolved.conditioning,
    })
}

/// The gate itself, over exactly the things it reads.
///
/// Split out from the spec-shaped entry point above so it is directly
/// exercisable: the decision is about `z`, the weights, the policy, the
/// absorber flag, the marginal design and the context covariates, and nothing
/// else about a survival term spec bears on it.
pub(crate) struct ResolvedLatentScoreCalibration {
    pub(crate) per_score: Vec<LatentMeasureCalibration>,
    pub(crate) per_score_measure: Vec<LatentMeasureKind>,
    pub(crate) per_score_consumed: Vec<LatentLawConsumed>,
    pub(crate) per_score_certificate_law: Vec<Option<crate::bms::EmpiricalZGrid>>,
    /// The moving-law certificate's candidates of a single moving score.
    pub(crate) moving_law: Option<Box<crate::bms::moving_law_rule::MovingLawCandidates>>,
    pub(crate) raw_scores: Array2<f64>,
    pub(crate) calibrated_scores: Array2<f64>,
    pub(crate) conditioning: Option<std::sync::Arc<Array2<f64>>>,
}

pub(crate) fn resolve_latent_score_calibration_from_parts(
    scores: &Array2<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
    absorber_active: bool,
    marginal_design: &DesignMatrix,
    local_context: Option<&estimated_latent_law::LocalLawContext<'_>>,
    support: EmpiricalLatentMeasureSupport,
) -> Result<ResolvedLatentScoreCalibration, String> {
    let k = scores.ncols();
    let raw_scores = scores.clone();
    let conditioning = if absorber_active {
        None
    } else {
        Some(marginal_design.try_to_dense_arc("survival marginal-slope conditional latent-z gate")?)
    };

    let mut calibrations = Vec::with_capacity(k);
    let mut measures = Vec::with_capacity(k);
    let mut consumed = Vec::with_capacity(k);
    let mut certificate_laws = Vec::with_capacity(k);
    let mut moving_law = None;
    let mut calibrated_scores = scores.clone();
    for col in 0..k {
        let raw = scores.column(col).to_owned();
        let mut decision = build_latent_measure_decision(
            &raw,
            weights,
            policy,
            conditioning.as_ref().map(|design| design.view()),
            local_context,
            support,
            "survival-marginal-slope",
        )?;
        // The moving-law certificate covers ONE score: with several, the anchor
        // reads the drive `rᵀz`, whose law is the joint law, and the arms are
        // not scored on a scalar axis here (gam#2949). The arm the gate FITTED
        // is still the law this column's own conditional evidence chose, and
        // the joint law transports it exactly unless it is a LOCAL law:
        //
        //   * a location-scale arm's calibration is applied to the score below,
        //     BEFORE any downstream consumer — the joint law included — sees
        //     it, so the moving mean and variance are divided out and what the
        //     joint law compresses is the standardised residual, whose law does
        //     not move. The map travels with the law
        //     (`SurvivalJointLatentLaw::score_calibrations`) so prediction reads
        //     a new score on the same axis;
        //   * a pooled or location-scale-empirical arm is a finite law of that
        //     same residual, which the joint law's own compression carries.
        //
        // Only the LOCAL arm — the gate's finding that the column's SHAPE moves
        // on the span — has nothing that `μ + L(a)·ε` can follow, and that
        // column alone still sends every column back to the closed form. The
        // certificate itself is not taken at `K ≥ 2`, and the label says so
        // through `uncertified`, the field that exists for an anchor the
        // certificate does not evaluate.
        if k >= 2 && decision.moving_law.take().is_some() {
            if matches!(decision.kind, LatentMeasureKind::LocalEmpirical { .. }) {
                decision.kind = LatentMeasureKind::StandardNormal;
                decision.calibration = LatentMeasureCalibration::None;
                decision.empirical_build = None;
            } else if let LatentLawConsumed::EstimatedMovingLaw {
                arm, uncertified, ..
            } = &mut decision.consumed
            {
                *uncertified = Some(format!(
                    "the cross-fitted moving-law certificate scores the arms of ONE score's own \
                     axis; with K={k} scores the anchor reads the drive rᵀz on the joint latent \
                     law, so the {} arm this column's conditional evidence chose is fitted and \
                     carried without being scored against the other arms (gam#2949)",
                    arm.label(),
                ));
            }
        } else if decision.moving_law.is_some() {
            moving_law = decision.moving_law.take();
        }
        if support == EmpiricalLatentMeasureSupport::StandardNormalOnly
            && !matches!(decision.kind, LatentMeasureKind::StandardNormal)
        {
            // Unreachable by construction — `StandardNormalOnly` never returns
            // another kind — but the closed-form kernel rests on it, so the
            // invariant is checked rather than assumed.
            return Err(
                "survival marginal-slope latent-measure gate returned a non-standard-normal \
                 measure for a standard-normal-only kernel"
                    .to_string(),
            );
        }
        log::debug!(
            "[survival-marginal-slope latent-z] score column {col}: the row index consumed the \
             {} latent law (gam#2926)",
            decision.consumed.label(),
        );
        let calibrated = match &decision.calibration {
            LatentMeasureCalibration::None => raw,
            LatentMeasureCalibration::ConditionalLocationScale(cal) => {
                // The conditional branch is only reachable when the gate had a
                // conditioning block to fire on, so it is present here.
                let a_block = conditioning.as_ref().ok_or_else(|| {
                    "survival marginal-slope conditional latent calibration requires the \
                     marginal conditioning block"
                        .to_string()
                })?;
                FittedLatentScoreMap::conditional_only(cal)
                    .calibrate(raw.view(), Some(a_block.view()))?
            }
        };
        if !matches!(decision.calibration, LatentMeasureCalibration::None) {
            log::debug!(
                "[survival-marginal-slope latent-z] score column {col}: applied the {} \
                 calibration before any downstream consumer saw the score",
                calibration_label(&decision.calibration),
            );
        }
        calibrated_scores.column_mut(col).assign(&calibrated);
        calibrations.push(decision.calibration);
        measures.push(decision.kind);
        consumed.push(decision.consumed);
        certificate_laws.push(decision.certificate_law);
    }
    if k >= 2 {
        route_multi_score_latent_laws(&mut measures, &mut consumed, &mut certificate_laws);
    }
    Ok(ResolvedLatentScoreCalibration {
        per_score: calibrations,
        per_score_measure: measures,
        per_score_consumed: consumed,
        per_score_certificate_law: certificate_laws,
        moving_law,
        raw_scores,
        calibrated_scores,
        conditioning,
    })
}

/// Settle what a fit on `K ≥ 2` scores anchors on, from each column's decision.
///
/// With several scores the anchor reads the law of the drive `rᵀz`, and any
/// column that is not the standard normal sends the fit to the joint latent law
/// (gam#2929), which transports one pooled residual law by `μ + L(a)·ε`.
///
/// That transport follows a moving MEAN and covariance — a location-scale arm's
/// map is divided out of the score before the law is built, and travels with the
/// law so prediction reads a new score on the same axis (gam#2949) — but not a
/// moving SHAPE. So a column whose fitted arm is the LOCAL one has nothing to
/// anchor on yet: every column keeps the closed form, labelled
/// `gaussian-uncertified` and naming that score. No default fit reaches the
/// joint law's refusal of a local law.
///
/// Otherwise, when some column departs, the fit anchors on the joint law for all
/// columns. A column the adequacy screen passed is then not anchored on its
/// closed form, so it is relabelled `estimated-global`, and no column carries a
/// closed-form certificate law. When every column passes, the closed form is
/// provisional, and the converged fit certifies it on the joint law.
fn route_multi_score_latent_laws(
    measures: &mut [LatentMeasureKind],
    consumed: &mut [LatentLawConsumed],
    certificate_laws: &mut [Option<crate::bms::EmpiricalZGrid>],
) {
    // A Gaussian declaration over several scores is one declaration about the
    // score vector, and the fit records it on the primary score. When a later
    // score fails the shape screen the record carries that score's ledger, so
    // the declaration's excess anchoring loss is measured on the joint law.
    if let LatentLawConsumed::DeclaredGaussian { adequacy: None, .. } = &consumed[0]
        && let Some((failing, ledger)) =
            consumed
                .iter()
                .enumerate()
                .skip(1)
                .find_map(|(col, decision)| match decision {
                    LatentLawConsumed::DeclaredGaussian {
                        adequacy: Some(adequacy),
                        ..
                    } => Some((col, adequacy.clone())),
                    _ => None,
                })
    {
        if let LatentLawConsumed::DeclaredGaussian { adequacy, .. } = &mut consumed[0] {
            *adequacy = Some(ledger);
        }
        certificate_laws[0] = certificate_laws[failing].clone();
    }
    let evidence_of = |decision: &LatentLawConsumed| match decision {
        LatentLawConsumed::EstimatedGaussianAdequate { evidence, .. }
        | LatentLawConsumed::EstimatedGlobalByResidual { evidence, .. }
        | LatentLawConsumed::GaussianUncertified { evidence, .. }
        | LatentLawConsumed::EstimatedGlobal { evidence }
        | LatentLawConsumed::EstimatedMovingLaw { evidence, .. }
        | LatentLawConsumed::ConditionalLocationScale { evidence, .. }
        | LatentLawConsumed::DeclaredGaussian { evidence, .. } => Some(evidence.clone()),
        LatentLawConsumed::RequestedGlobalEmpirical | LatentLawConsumed::DeclaredFiniteLaw { .. } => {
            None
        }
    };
    let k = measures.len();
    // gam#2949: only a column whose fitted arm is LOCAL is beyond the joint
    // law's transport. A location-scale arm is divided out of the score before
    // the joint law sees it, and a pooled or residual finite law is a law of
    // that same residual, so those columns are carried on the joint law and are
    // no longer a reason to put every column back on the closed form.
    if let Some(moving) = consumed.iter().position(|decision| {
        matches!(
            decision,
            LatentLawConsumed::EstimatedMovingLaw {
                arm: MovingLawArm::Local,
                ..
            }
        )
    }) {
        let moving_summary = evidence_of(&consumed[moving])
            .map(|evidence| evidence.summary())
            .unwrap_or_default();
        let missing = format!(
            "the conditional law of score column {moving} moves in SHAPE on the marginal-index \
             span ({moving_summary}), and with K={k} scores the joint latent law transports one \
             pooled residual law by μ + L(a)·ε, which follows a moving mean and covariance but \
             not a moving shape (gam#2949)"
        );
        log::debug!(
            "[survival-marginal-slope latent-z] every score column keeps the closed form, \
             uncertified: {missing} (gam#2926)"
        );
        for col in 0..k {
            measures[col] = LatentMeasureKind::StandardNormal;
            certificate_laws[col] = None;
            // One policy governs every column, so a local law sits only beside
            // other automatic decisions, and each of those carries its evidence.
            let Some(evidence) = evidence_of(&consumed[col]) else {
                continue;
            };
            let adequacy = match &consumed[col] {
                LatentLawConsumed::EstimatedGaussianAdequate { adequacy, .. } => {
                    Some(adequacy.clone())
                }
                LatentLawConsumed::GaussianUncertified { adequacy, .. } => adequacy.clone(),
                _ => None,
            };
            consumed[col] = LatentLawConsumed::GaussianUncertified {
                evidence,
                adequacy,
                certificate: None,
                missing: missing.clone(),
            };
        }
        return;
    }
    if measures
        .iter()
        .any(|measure| !matches!(measure, LatentMeasureKind::StandardNormal))
    {
        for col in 0..k {
            if let LatentLawConsumed::EstimatedGaussianAdequate { evidence, .. } = &consumed[col] {
                consumed[col] = LatentLawConsumed::EstimatedGlobal {
                    evidence: evidence.clone(),
                };
            }
            certificate_laws[col] = None;
        }
    }
}

/// Decide the score-covariance field the fit will consume (gam#2766).
///
/// Split out from [`fit_survival_marginal_slope_terms_impl`] for the same reason
/// [`resolve_latent_score_calibration_from_parts`] is: the decision is about the
/// calibrated scores, the weights and the conditioning block, and nothing else
/// about a survival term spec bears on it — so it should be exercisable without
/// standing up a whole fit.
///
/// `scores` must be the axis the row kernel will see, i.e. AFTER
/// [`resolve_survival_latent_score_calibration`]. `Σ` is the covariance of the
/// score the likelihood integrates over, so running this on the raw axis would
/// model a different object than the one `c(a)` has to correct for.
///
/// Returns the pooled field when there is no conditioning block (the #461
/// absorber suppressed it — there is then no span to condition on and the
/// pooled matrix is the only defined answer), and when the pair-wise Rao gate
/// does not fire.
pub(crate) fn resolve_score_covariance_field(
    scores: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    conditioning: Option<ArrayView2<'_, f64>>,
) -> Result<ScoreCovarianceField, String> {
    let pooled = marginal_slope_covariance_from_scores(scores, &weights.to_owned())?;
    let Some(conditioning) = conditioning else {
        return Ok(ScoreCovarianceField::pooled(pooled));
    };
    match crate::bms::ConditionalScoreCovariance::fit(scores, weights, conditioning)? {
        Some(model) => {
            log::debug!(
                "[survival-marginal-slope] conditional score covariance ENGAGED on K={} scores: \
                 pair Rao p-values {:?}",
                model.score_dim,
                model.pair_pvalues,
            );
            ScoreCovarianceField::conditional(pooled, model, conditioning)
        }
        None => Ok(ScoreCovarianceField::pooled(pooled)),
    }
}

fn calibration_label(calibration: &LatentMeasureCalibration) -> &'static str {
    match calibration {
        LatentMeasureCalibration::None => "identity",
        LatentMeasureCalibration::ConditionalLocationScale(_) => "conditional location-scale",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bms::{LatentZCheckMode, LatentZNormalizationMode};
    use gam_linalg::matrix::DenseDesignMatrix;

    /// Deterministic standard normals (Box–Muller over splitmix64).
    fn gaussians(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut unit = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            // `unit` draws from the open interval (0, 1), so `ln u1` is finite.
            let u1 = unit();
            let u2 = unit();
            let r = (-2.0 * u1.ln()).sqrt();
            out.push(r * (std::f64::consts::TAU * u2).cos());
            out.push(r * (std::f64::consts::TAU * u2).sin());
        }
        out.truncate(n);
        out
    }

    fn standardized(mut v: Vec<f64>) -> Vec<f64> {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
        for value in v.iter_mut() {
            *value = (*value - mean) / sd;
        }
        v
    }

    /// `z = m·x + √(1−m²)·ζ`, exactly standard normal marginally and
    /// conditionally shifted on the marginal design's `x` column.
    fn shifted_fixture(n: usize, m: f64) -> (Array2<f64>, Array1<f64>, DesignMatrix, Vec<f64>) {
        let x = standardized(gaussians(n, 0x2768_A1));
        let zeta = standardized(gaussians(n, 0x2768_B2));
        let residual_sd = (1.0 - m * m).sqrt();
        let mut z = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            z[[row, 0]] = m * x[row] + residual_sd * zeta[row];
        }
        let mut design = Array2::<f64>::ones((n, 2));
        for row in 0..n {
            design[[row, 1]] = x[row];
        }
        (
            z,
            Array1::<f64>::ones(n),
            DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            zeta,
        )
    }

    fn policy(latent_measure: LatentMeasureSpec) -> LatentZPolicy {
        LatentZPolicy {
            check_mode: LatentZCheckMode::WarnOnly,
            normalization: LatentZNormalizationMode::Frozen { mean: 0.0, sd: 1.0 },
            latent_measure,
            ..LatentZPolicy::frozen_transformation_normal()
        }
    }

    fn location_scale() -> LatentMeasureSpec {
        LatentMeasureSpec::ConditionalLocationScale {
            grid_size: crate::bms::DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
        }
    }

    /// Under the declared conditional location-scale law the gate fires on a
    /// conditionally shifted score, and the score it hands the kernel is the
    /// CLEAN one — not merely centred (gam#2768, gam#2926).
    ///
    /// Recovering `ζ` up to sign and a common scale is the whole claim: the fit's
    /// coefficients are then the ones the outcome was generated with, and the
    /// influence channel `q` no longer carries `b(C)·m(C)`.
    #[test]
    fn survival_gate_recovers_the_conditionally_standardized_score() {
        let n = 4000;
        let m = 0.6;
        let (z, weights, design, zeta_truth) = shifted_fixture(n, m);
        let resolved = resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &policy(location_scale()),
            false,
            &design,
            None,
            EmpiricalLatentMeasureSupport::Available,
        )
        .expect("gate");

        assert_eq!(resolved.per_score.len(), 1, "one decision per score column");
        assert!(
            matches!(
                resolved.per_score[0],
                LatentMeasureCalibration::ConditionalLocationScale(_)
            ),
            "the E[z|C] Rao gate must escalate to the conditional location-scale \
             correction at Corr(z, x) = {m} and n = {n}; got {}",
            calibration_label(&resolved.per_score[0])
        );

        // The calibrated column must be ζ itself. Correlation, because the
        // calibration is fit rather than known and carries estimation error in
        // m̂ and in the residual scale.
        let calibrated = resolved.calibrated_scores.column(0);
        let dot: f64 = calibrated
            .iter()
            .zip(zeta_truth.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_cal = calibrated.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_truth = zeta_truth.iter().map(|v| v * v).sum::<f64>().sqrt();
        let correlation = dot / (norm_cal * norm_truth);
        assert!(
            correlation > 0.9995,
            "the calibrated score must BE the clean score; correlation {correlation:.6}"
        );

        // Unit variance, which is what keeps `q` the marginal index.
        let mean = calibrated.iter().sum::<f64>() / n as f64;
        let sd = (calibrated.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        assert!(
            mean.abs() < 0.02 && (sd - 1.0).abs() < 0.05,
            "calibrated score must be standardised; mean={mean:.4} sd={sd:.4}"
        );

        // The raw score is kept for the generated-regressor correction, which
        // differentiates the FIRST stage and therefore needs the axis that stage
        // regressed — not the one the fit consumed.
        assert_eq!(resolved.raw_scores, z, "the raw score must survive the gate");
        assert!(
            resolved.conditioning.is_some(),
            "the conditional branch must retain the block it conditioned on"
        );
    }

    /// The gate must NOT fire on a score that is already conditionally standard
    /// normal. A trigger-happy gate would redefine the latent axis of every
    /// clean fit, which is a worse failure than the one it exists to prevent.
    #[test]
    fn survival_gate_stays_quiet_on_an_unshifted_score() {
        let n = 4000;
        let (_, weights, design, zeta) = shifted_fixture(n, 0.6);
        let mut clean = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            clean[[row, 0]] = zeta[row];
        }
        let resolved = resolve_latent_score_calibration_from_parts(
            &clean,
            &weights,
            &policy(location_scale()),
            false,
            &design,
            None,
            EmpiricalLatentMeasureSupport::Available,
        )
        .expect("gate");
        assert!(
            matches!(resolved.per_score[0], LatentMeasureCalibration::None),
            "got {}",
            calibration_label(&resolved.per_score[0])
        );
        assert_eq!(
            resolved.calibrated_scores, clean,
            "an unfired gate must leave the score untouched, byte for byte"
        );

        // The default on the same clean score: no structure on the span and a
        // score the adequacy check cannot tell from N(0, 1), so the closed form,
        // chosen by that evidence and recorded (gam#2926).
        let estimated = resolve_latent_score_calibration_from_parts(
            &clean,
            &weights,
            &policy(LatentMeasureSpec::auto_default()),
            false,
            &design,
            None,
            EmpiricalLatentMeasureSupport::Available,
        )
        .expect("gate");
        let LatentLawConsumed::EstimatedGaussianAdequate { adequacy, .. } =
            &estimated.per_score_consumed[0]
        else {
            panic!(
                "a conditionally standard normal score must reach the closed form by evidence; got {:?}",
                estimated.per_score_consumed[0]
            )
        };
        assert!(adequacy.passes(), "the recorded evidence must be the passing ledger");
        assert!(matches!(
            estimated.per_score_measure[0],
            LatentMeasureKind::StandardNormal
        ));
        assert_eq!(estimated.calibrated_scores, clean);
    }

    /// The default on a conditionally shifted score with no context covariates
    /// is refused by name; a closed-form-only kernel keeps the closed form where
    /// the default law departs, recorded as uncertified with what is missing, and
    /// fits a provisional closed form where the screen cannot tell the law from
    /// N(0, 1) (gam#2926).
    #[test]
    fn survival_default_law_refuses_or_stays_uncertified_by_name() {
        let n = 4000;
        let (z, weights, design, zeta) = shifted_fixture(n, 0.6);
        let error = match resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &policy(LatentMeasureSpec::auto_default()),
            false,
            &design,
            None,
            EmpiricalLatentMeasureSupport::Available,
        ) {
            Ok(_) => panic!("a moving conditional law without context covariates must refuse"),
            Err(error) => error,
        };
        assert!(
            error.contains("conditional law moves on the marginal-index span"),
            "got {error}"
        );
        let uncertified = resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &policy(LatentMeasureSpec::auto_default()),
            false,
            &design,
            None,
            EmpiricalLatentMeasureSupport::StandardNormalOnly,
        )
        .expect("a closed-form-only kernel keeps the closed form where the default law moves");
        let LatentLawConsumed::GaussianUncertified { missing, .. } = &uncertified.per_score_consumed[0]
        else {
            panic!(
                "a moving law on a closed-form-only kernel must be recorded as uncertified; got {:?}",
                uncertified.per_score_consumed[0]
            )
        };
        assert!(
            missing.contains("moves on the marginal-index span")
                && missing.contains("evaluates only the closed form"),
            "the record must say how the law departs and why the kernel cannot carry it; got {missing}"
        );
        assert!(matches!(
            uncertified.per_score_measure[0],
            LatentMeasureKind::StandardNormal
        ));
        let mut clean = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            clean[[row, 0]] = zeta[row];
        }
        let adequate = resolve_latent_score_calibration_from_parts(
            &clean,
            &weights,
            &policy(LatentMeasureSpec::auto_default()),
            false,
            &design,
            None,
            EmpiricalLatentMeasureSupport::StandardNormalOnly,
        )
        .expect("a Gaussian-adequate score reaches the closed form on a closed-form-only kernel");
        assert!(matches!(
            adequate.per_score_consumed[0],
            LatentLawConsumed::EstimatedGaussianAdequate { .. }
        ));
    }

    /// `K = 2` scores whose conditional CORRELATION is `amplitude·x/√(1+x²)` and
    /// whose conditional marginals are exactly `N(0,1)`. At `amplitude = 0` the
    /// conditional covariance is constant. This is the gam#2766 shape: nothing
    /// for the per-coordinate gate above to correct, everything in the
    /// off-diagonal.
    fn varying_correlation_fixture(
        n: usize,
        amplitude: f64,
    ) -> (Array2<f64>, Array1<f64>, DesignMatrix) {
        let x = standardized(gaussians(n, 0x2766_11));
        let e0 = standardized(gaussians(n, 0x2766_22));
        let e1 = standardized(gaussians(n, 0x2766_33));
        let mut scores = Array2::<f64>::zeros((n, 2));
        let mut design = Array2::<f64>::ones((n, 2));
        for row in 0..n {
            design[[row, 1]] = x[row];
            let phi = amplitude * x[row] / (1.0 + x[row] * x[row]).sqrt();
            scores[[row, 0]] = e0[row];
            scores[[row, 1]] = phi * e0[row] + (1.0 - phi * phi).max(0.0).sqrt() * e1[row];
        }
        (
            scores,
            Array1::<f64>::ones(n),
            DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        )
    }

    /// The fit installs a per-row `Σ(a_i)` when the pair gate fires, and the
    /// rows genuinely differ. Without this the model and the row program could
    /// both be right while the entry point never wires them together.
    #[test]
    fn the_fit_installs_a_conditional_field_when_the_pair_gate_fires() {
        let n = 20_000;
        let (scores, weights, design) = varying_correlation_fixture(n, 0.8);
        let conditioning = design
            .try_to_dense_arc("test conditioning")
            .expect("dense conditioning");
        let field = resolve_score_covariance_field(
            scores.view(),
            weights.view(),
            Some(conditioning.view()),
        )
        .expect("field");
        assert!(field.is_conditional(), "a varying Cov(z₀,z₁|a) must escalate");
        assert_eq!(field.materialised_rows(), Some(n));
        assert_eq!(field.dim(), 2);
        // The correlation must track the planted sign across the covariate.
        let correlation = |row: usize| {
            let sigma = field.at_row(row).to_dense();
            sigma[[0, 1]] / (sigma[[0, 0]] * sigma[[1, 1]]).sqrt()
        };
        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for row in 0..n {
            let value = correlation(row);
            lowest = lowest.min(value);
            highest = highest.max(value);
        }
        assert!(
            lowest < -0.4 && highest > 0.4,
            "the installed field must span the planted correlation range; got [{lowest:.3}, {highest:.3}]"
        );
    }

    /// A constant conditional covariance leaves the pooled object in place, and
    /// so does a suppressed conditioning block: with no span to condition on the
    /// pooled matrix is the only defined answer, which is the #461 absorber
    /// seam one level up.
    #[test]
    fn the_fit_keeps_the_pooled_field_without_a_trigger_or_a_span() {
        let n = 8_000;
        let (scores, weights, design) = varying_correlation_fixture(n, 0.0);
        let conditioning = design
            .try_to_dense_arc("test conditioning")
            .expect("dense conditioning");
        let constant = resolve_score_covariance_field(
            scores.view(),
            weights.view(),
            Some(conditioning.view()),
        )
        .expect("field");
        assert!(
            !constant.is_conditional(),
            "a constant Cov(z₀,z₁|a) must leave the pooled Σ in place"
        );

        // Same scores, no conditioning block at all.
        let (varying, weights, _) = varying_correlation_fixture(n, 0.8);
        let suppressed =
            resolve_score_covariance_field(varying.view(), weights.view(), None).expect("field");
        assert!(
            !suppressed.is_conditional(),
            "with no conditioning span there is nothing to condition Σ on"
        );
        // And the pooled object it keeps is the one the fit would have built.
        let expected = marginal_slope_covariance_from_scores(varying.view(), &weights)
            .expect("pooled Σ")
            .to_dense();
        assert_eq!(suppressed.pooled_covariance().to_dense(), expected);
    }

    /// The #461 absorber seam: with a CTN Stage-1 influence absorber active the
    /// conditional leakage is already absorbed, and replacing z would perturb the
    /// widened-marginal predict seam. Neither the span test nor a calibration may
    /// then engage, even on a score that would otherwise move on the span.
    #[test]
    fn survival_gate_does_not_condition_behind_an_active_influence_absorber() {
        let n = 4000;
        let (z, weights, design, _) = shifted_fixture(n, 0.6);
        let resolved = resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &policy(LatentMeasureSpec::auto_default()),
            true,
            &design,
            None,
            EmpiricalLatentMeasureSupport::Available,
        )
        .expect("gate");
        assert!(
            matches!(resolved.per_score[0], LatentMeasureCalibration::None),
            "no calibration may be fitted behind an active absorber"
        );
        let LatentLawConsumed::EstimatedGaussianAdequate { evidence, .. } =
            &resolved.per_score_consumed[0]
        else {
            panic!(
                "behind an absorber the default has no span to test, and this score is marginally \
                 N(0, 1), so the closed form by evidence; got {:?}",
                resolved.per_score_consumed[0]
            )
        };
        assert!(
            evidence.mean_p_value.is_none(),
            "the span test must not run behind an absorber"
        );
        assert!(
            resolved.conditioning.is_none(),
            "no conditioning block may be built when the branch is suppressed"
        );
        // The declared location-scale law needs the span, and says so.
        let error = match resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &policy(location_scale()),
            true,
            &design,
            None,
            EmpiricalLatentMeasureSupport::Available,
        ) {
            Ok(_) => panic!("the declared location-scale law must refuse behind an absorber"),
            Err(error) => error,
        };
        assert!(error.contains("no span is available"), "got {error}");
    }

    /// Every latent-score column is gated, not just the primary one: with `K > 1`
    /// the leakage is a SUM of per-coordinate conditional shifts, so a gate that
    /// only looked at column 0 would leave the rest of it in `q`.
    #[test]
    fn survival_gate_covers_every_score_column() {
        let n = 4000;
        let m = 0.6;
        let (first, weights, design, zeta) = shifted_fixture(n, m);
        let mut two = Array2::<f64>::zeros((n, 2));
        two.column_mut(0).assign(&first.column(0));
        // A second column shifted the OTHER way, so a gate that reused column
        // 0's calibration would leave a visible residual correlation.
        let residual_sd = (1.0 - m * m).sqrt();
        let x: Vec<f64> = (0..n)
            .map(|row| (first[[row, 0]] - residual_sd * zeta[row]) / m)
            .collect();
        let other = standardized(gaussians(n, 0x2768_C3));
        for row in 0..n {
            two[[row, 1]] = -m * x[row] + residual_sd * other[row];
        }
        let resolved = resolve_latent_score_calibration_from_parts(
            &two,
            &weights,
            &policy(location_scale()),
            false,
            &design,
            None,
            EmpiricalLatentMeasureSupport::Available,
        )
        .expect("gate");
        assert_eq!(resolved.per_score.len(), 2);
        for (column, calibration) in resolved.per_score.iter().enumerate() {
            assert!(
                matches!(
                    calibration,
                    LatentMeasureCalibration::ConditionalLocationScale(_)
                ),
                "column {column} must be gated on its own conditional moments; got {}",
                calibration_label(calibration)
            );
        }
        // Both calibrated columns must be conditionally uncorrelated with x.
        for column in 0..2 {
            let calibrated = resolved.calibrated_scores.column(column);
            let cov: f64 = calibrated
                .iter()
                .zip(x.iter())
                .map(|(z, x)| z * x)
                .sum::<f64>()
                / n as f64;
            assert!(
                cov.abs() < 0.03,
                "column {column} still carries a conditional shift: Cov(ζ, x) = {cov:.4}"
            );
        }
    }

    /// With `K = 2` scores the default never reaches the joint law's refusal of a
    /// local law. A column whose law moves in location and scale carries its
    /// fitted arm onto the joint law, which transports it once the map is divided
    /// out (gam#2949, 7e641c78b9): the arm is recorded, uncertified at `K ≥ 2`
    /// and naming the issue, and no column is sent back to the closed form, which
    /// only a LOCAL arm still forces. A column that departs without moving
    /// sends the fit to the joint law, and the column the screen passed is then
    /// labelled by the law the fit anchors on, not by a closed form it never
    /// consumes (gam#2926, gam#2929, gam#2949).
    #[test]
    fn multi_score_default_routes_around_the_joint_law_refusal() {
        let n = 4000;
        let (first, weights, design, zeta) = shifted_fixture(n, 0.6);
        let x: Vec<f64> = design_column(&design, 1);
        let features = Array2::from_shape_vec((n, 1), x.clone()).expect("features");
        let context = estimated_latent_law::LocalLawContext {
            features: features.view(),
            feature_cols: vec![0],
        };

        let mut moving = Array2::<f64>::zeros((n, 2));
        moving.column_mut(0).assign(&first.column(0));
        moving.column_mut(1).assign(&Array1::from(zeta.clone()));
        let routed = resolve_latent_score_calibration_from_parts(
            &moving,
            &weights,
            &policy(LatentMeasureSpec::auto_default()),
            false,
            &design,
            Some(&context),
            EmpiricalLatentMeasureSupport::Available,
        )
        .expect("a moving column at K = 2 carries its arm rather than refusing");
        let LatentLawConsumed::EstimatedMovingLaw { arm, uncertified, .. } =
            &routed.per_score_consumed[0]
        else {
            panic!(
                "the moving column must carry its fitted arm; got {:?}",
                routed.per_score_consumed[0]
            )
        };
        assert!(
            !matches!(arm, MovingLawArm::Local),
            "the shifted fixture moves in location, not shape; got the {} arm",
            arm.label()
        );
        let missing = uncertified
            .as_deref()
            .expect("a K = 2 arm is carried without the one-score certificate");
        assert!(
            missing.contains("gam#2949"),
            "the record must name the transport issue; got {missing}"
        );
        assert!(routed.per_score_certificate_law[0].is_none());
        for col in 0..2 {
            assert!(
                !matches!(
                    routed.per_score_consumed[col],
                    LatentLawConsumed::GaussianUncertified { .. }
                ),
                "column {col}: only a LOCAL arm sends the columns back to the closed form; \
                 got {:?}",
                routed.per_score_consumed[col]
            );
        }
        assert!(
            joint_latent_law_measure_refusal(2, &routed.per_score_measure).is_none(),
            "the routed decision must not reach the joint law's refusal"
        );

        let skewed = standardized(
            gaussians(n, 0x2926_5C)
                .into_iter()
                .map(|g| (0.8 * g).exp())
                .collect(),
        );
        let mut departing = Array2::<f64>::zeros((n, 2));
        departing.column_mut(0).assign(&Array1::from(zeta));
        departing.column_mut(1).assign(&Array1::from(skewed));
        let joint = resolve_latent_score_calibration_from_parts(
            &departing,
            &weights,
            &policy(LatentMeasureSpec::auto_default()),
            false,
            &design,
            Some(&context),
            EmpiricalLatentMeasureSupport::Available,
        )
        .expect("a departing column at K = 2 anchors on the joint law");
        assert!(
            matches!(
                joint.per_score_measure[1],
                LatentMeasureKind::GlobalEmpirical { .. }
            ),
            "the skewed column must depart; got {:?}",
            joint.per_score_consumed[1]
        );
        assert!(
            matches!(
                joint.per_score_consumed[0],
                LatentLawConsumed::EstimatedGlobal { .. }
            ),
            "the column the screen passed anchors on the joint law; got {:?}",
            joint.per_score_consumed[0]
        );
        assert!(joint.per_score_certificate_law.iter().all(Option::is_none));
        assert!(joint_latent_law_measure_refusal(2, &joint.per_score_measure).is_none());
    }

    fn design_column(design: &DesignMatrix, col: usize) -> Vec<f64> {
        design
            .try_to_dense_arc("test design column")
            .expect("dense design")
            .column(col)
            .to_vec()
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::super::spec::split_persisted_latent_calibrations;
    use crate::bms::{LatentMeasureCalibration, LatentZConditionalCalibration};
    use ndarray::Array2;

    fn conditional() -> LatentMeasureCalibration {
        LatentMeasureCalibration::ConditionalLocationScale(LatentZConditionalCalibration {
            mean_coeffs: vec![0.0, 0.5],
            log_var_coeffs: Vec::new(),
            basis_ncols: 1,
            homoskedastic_var: 0.75,
            post_mean: 0.0,
            post_sd: 1.0,
            theta1_cov: Array2::<f64>::zeros((3, 3)),
        })
    }

    #[test]
    fn an_unfired_gate_persists_nothing() {
        let cond = split_persisted_latent_calibrations(&[LatentMeasureCalibration::None], true, false)
            .expect("no calibration is always persistable");
        assert!(cond.is_none());
    }

    #[test]
    fn a_conditional_calibration_persists_into_its_field() {
        let cond = split_persisted_latent_calibrations(&[conditional()], true, false)
            .expect("conditional persists");
        assert!(cond.is_some());
    }

    /// A conditioning span the resolved marginal spec would NOT rebuild is a
    /// refusal, not a warning: the saved model would apply a different latent map
    /// than its coefficients were fitted under, and nothing downstream could tell.
    #[test]
    fn an_unreproducible_conditioning_span_refuses_to_persist() {
        let error = split_persisted_latent_calibrations(&[conditional()], false, false)
            .expect_err("an unreproducible span must refuse");
        assert!(
            error.contains("RESOLVED marginal spec"),
            "the refusal must name what prediction would rebuild instead; got {error}"
        );
        // A fit that calibrated nothing conditions on nothing, so the same state
        // must NOT refuse it.
        split_persisted_latent_calibrations(&[LatentMeasureCalibration::None], false, false)
            .expect("an uncalibrated score is unaffected by the span");
    }

    /// The saved contract holds one score surface. A K>1 fit whose SECOND score
    /// was calibrated cannot be represented, and the loss is named at the point
    /// of loss rather than truncated.
    #[test]
    fn a_calibrated_secondary_score_refuses_to_persist() {
        let error = split_persisted_latent_calibrations(
            &[LatentMeasureCalibration::None, conditional()],
            true,
            false,
        )
        .expect_err("a calibrated second score must refuse");
        assert!(
            error.contains("column 1"),
            "the refusal must name the column that cannot be carried; got {error}"
        );
        // An uncalibrated second score carries nothing to lose.
        split_persisted_latent_calibrations(
            &[conditional(), LatentMeasureCalibration::None],
            true,
            false,
        )
        .expect("only the persisted surface was calibrated");
    }

    /// gam#2949: when the joint latent law carries one map per coordinate, the
    /// single-surface payload field is EMPTY — one owner for the map a score is
    /// read on — and a calibrated second score is no longer a refusal. The
    /// reproducibility refusal still binds, because the span is rebuilt from the
    /// resolved marginal spec whichever object holds the map.
    #[test]
    fn a_joint_law_that_carries_the_maps_persists_none_of_them_in_the_scalar_field_2949() {
        let carried = split_persisted_latent_calibrations(
            &[LatentMeasureCalibration::None, conditional()],
            true,
            true,
        )
        .expect("a calibrated second score persists once the law carries its map");
        assert!(
            carried.is_none(),
            "the scalar field must stay empty so no coordinate is mapped twice"
        );
        let first_only = split_persisted_latent_calibrations(&[conditional()], true, true)
            .expect("the primary score's map travels with the law too");
        assert!(first_only.is_none());
        let error =
            split_persisted_latent_calibrations(&[conditional()], false, true).expect_err(
                "an unreproducible span refuses whichever object carries the map",
            );
        assert!(
            error.contains("RESOLVED marginal spec"),
            "the refusal must name what prediction would rebuild instead; got {error}"
        );
    }
}
