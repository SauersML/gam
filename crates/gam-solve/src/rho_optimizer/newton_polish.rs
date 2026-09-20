//! The mint's Newton polish (#2954): which verdicts it polishes, the damped step
//! on the free coordinates, the settling step where the decrease left is
//! resolvable but a step's is not, when it tries a limit face first (#3012), and
//! the coordinate a stopped polish rails at a limit-model bound.

use super::capability::OuterThetaLayout;
use super::decrement_bands::OuterDecrementDecision;
use super::run::{
    CertificationFidelity, OuterConfig, StationarityBound, StationarityBoundSource,
    certify_outer_optimality_at_terminal_fidelity, outer_nonconvergence_error,
};
use super::{OuterEvalOrder, OuterObjective, OuterResult, native_coordinate};
use crate::estimate::EstimationError;
use crate::model_types::{NewtonPolishRecord, OuterCriterionCertificate, RailFaceKind};
use ndarray::{Array1, Array2};

/// What the mint's Newton polish reads from the certificate that decided to
/// take it (#2954).
pub(super) struct MintPolish<'a> {
    pub(super) allow_certify_reseed: bool,
    pub(super) fidelity: CertificationFidelity,
    /// The polish this walk has taken so far, `None` on the first verdict.
    pub(super) polish: Option<PolishWalk>,
    pub(super) decision: &'a OuterDecrementDecision,
    pub(super) evidence: &'a opt::DecrementEvidence,
    /// The criterion at the judged point.
    pub(super) cost: f64,
    pub(super) analytic_hessian: &'a Option<Array2<f64>>,
    pub(super) projected_gradient: &'a Array1<f64>,
    pub(super) projected_grad_norm: f64,
    pub(super) bounds: &'a (Array1<f64>, Array1<f64>),
    pub(super) layout: OuterThetaLayout,
    pub(super) stationarity_bound: f64,
    pub(super) bound_source: StationarityBoundSource,
}

/// #2954 Newton polish. The mint's decrement verdict is stricter than any
/// search stop: a search told to stop on a gradient band hands over a point
/// where a Newton step still buys a decrease the arithmetic resolves, and the
/// verdict refuses it. The analytic Hessian that decided is in hand, so the
/// mint takes that Newton step on the free coordinates, clamped to the model
/// box, and re-certifies the point it reaches by the same verdict: the
/// recursion re-owns it with a value, gradient and Hessian evaluation. A step
/// is kept only when it lowers the criterion by more than `band_f`, and a full
/// step that does not is damped (`damped_newton_step`). Where the decrease left,
/// `λ̂²`, is resolvable but the full step's own `½λ̂²` is not, the full step is a
/// settling step whose point must certify. The walk ends because the criterion
/// is bounded below and each kept step buys more than `band_f` of it, not
/// because the decrement contracts: a resolvable decrease is progress even
/// where `λ̂²` grew after it. Where the decrement contracts slower than Newton's
/// quadratic rate the polish tries the limit faces its step heads to first
/// (#3012).
///
/// When polishing stops with the decrement still above its band, the optimum
/// may lie on the box instead: along an exponential tail `V ≈ V∞ + a·e^(−ρ)`
/// every Newton step moves ρ by exactly one, `λ̂²` contracts by `e^(−1)`, and
/// the infimum is at the bound. That is projected Newton's case, so the
/// coordinate carrying the step toward its bound is railed there when that
/// lowers the criterion by more than `band_f`, as a kept Newton step must, and
/// the recursion judges the free coordinates by the decrement and the railed
/// one by its bound's KKT sign (`outer_decrement_verdict` releases it again if
/// its inward descent is resolvable). Otherwise the point is refused by name
/// and never reaches the first-order ladder below.
///
/// Called where the verdict's decrement is resolvable
/// ([`resolvable_decrease_evidence`]). It returns the certificate the recursion
/// mints at the point the polish reaches, or the named refusal.
pub(super) fn polish_the_mint(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    inputs: MintPolish<'_>,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let MintPolish {
        allow_certify_reseed,
        fidelity,
        polish,
        decision,
        evidence,
        cost,
        analytic_hessian,
        projected_gradient,
        projected_grad_norm,
        bounds,
        layout,
        stationarity_bound,
        bound_source,
    } = inputs;
    let mut record = match polish {
        Some(walk) => walk.record,
        None => crate::model_types::NewtonPolishRecord {
            lambda_sq_before: evidence.lambda_sq,
            lambda_sq_after: evidence.lambda_sq,
            decreases: Vec::new(),
            settled: false,
            rails: Vec::new(),
            entry: result.rho.to_vec(),
        },
    };
    // A rail changes the face, so the rate is measured from the first verdict on
    // the new face.
    let face_start = record.rails.last().map_or(0, |rail| rail.steps_before);
    // The decrement where the last Newton step on this face began, when one did.
    // A decrement that grew after a step is no reason to stop: that step lowered
    // the criterion by more than `band_f`, and the walk's bound is the criterion's
    // lower bound, not the decrement's contraction (#3012).
    let previous_lambda_sq =
        (record.decreases.len() > face_start).then_some(record.lambda_sq_after);
    // Newton's quadratic rate, `λ₊ ≤ 2λ²`: the rate of a criterion
    // self-concordant with constant 2 (`|V'''| ≤ 2·V''^(3/2)`) once `λ ≤ 1/4`
    // (Boyd & Vandenberghe, *Convex Optimization*, §9.6.3), measured on the step
    // just taken rather than predicted from the first decrement. An exponential
    // tail `V∞ + a·e^(−ρ)` has `|V'''|/V''^(3/2) = V''^(−1/2)`, large exactly
    // where the tail is flat, and there Newton contracts `λ̂²` by `e^(−1)` per
    // step, a linear rate, with the infimum at the bound.
    let quadratic = previous_lambda_sq
        .is_none_or(|previous| evidence.lambda_sq.sqrt() <= 2.0 * previous);
    let settled = record.settled;
    record.lambda_sq_after = evidence.lambda_sq;
    record.settled = false;
    let newton_step = analytic_hessian
        .as_ref()
        .and_then(|hessian| free_newton_step(hessian, &projected_gradient, &decision.face));
    let mut trial_ran = false;
    // Whether a backtrack along the Newton step found no step that lowers the
    // criterion by more than `band_f`: the decrement promises a decrease the
    // criterion does not deliver along its Newton direction, and the refusal is
    // typed by it (#3012).
    let mut backtrack_unresolved = false;
    let mut stopped = if settled {
        format!(
            "the settling step's point does not certify (λ̂² {:.3e} there, {:.3e} before it)",
            evidence.lambda_sq,
            previous_lambda_sq.unwrap_or(f64::NAN),
        )
    } else if !quadratic {
        format!(
            "the Newton decrement contracted slower than Newton's quadratic rate λ₊ ≤ 2λ² \
             (λ̂² {:.3e} after the last step, {:.3e} before it)",
            evidence.lambda_sq,
            previous_lambda_sq.unwrap_or(f64::NAN),
        )
    } else if let Some(step) = newton_step.as_ref() {
        trial_ran = true;
        match damped_newton_step(obj, config, &result.rho, step, bounds, cost, evidence) {
            Ok(taken) => {
                return take_polish_step(
                    obj,
                    config,
                    context,
                    result,
                    allow_certify_reseed,
                    fidelity,
                    record,
                    taken,
                    cost,
                    evidence,
                );
            }
            Err(reason) => {
                backtrack_unresolved = true;
                reason
            }
        }
    } else {
        "the free Hessian is not positive definite".to_string()
    };
    let mut refusal_source = bound_source;
    let plan = newton_step.as_ref().map_or(RailPlan::None, |step| {
        polish_rail_plan(
            &result.rho,
            &projected_gradient,
            step,
            &bounds,
            &decision.face,
            &record.rails,
            layout,
            config,
        )
    });
    match plan {
        RailPlan::Faces(faces) => {
            for (face, path) in faces {
                let mut railed = path.unwrap_or_else(|| result.rho.clone());
                for &(k, bound) in &face {
                    railed[k] = bound;
                }
                trial_ran = true;
                let natives = face
                    .iter()
                    .map(|&(k, _)| native_coordinate(config.native_coordinate_order.as_deref(), k))
                    .collect::<Vec<_>>();
                match judged_cost(obj, &railed) {
                    Ok((rail_cost, judged))
                        if rail_cost.is_finite() && cost - rail_cost > evidence.band_f =>
                    {
                        log::debug!(
                            "[CERTIFICATE] {context}: Newton-decrement polish rails coordinate(s) \
                             {natives:?} at their limit-model bounds ({stopped}); the criterion \
                             there is {rail_cost:.9e} against {cost:.9e}, and the railed point is \
                             judged instead (#2954)",
                        );
                        for &(k, bound) in &face {
                            record.rails.push(crate::model_types::NewtonPolishRail {
                                index: k,
                                from: result.rho[k],
                                to: bound,
                                decrease: cost - rail_cost,
                                steps_before: record.decreases.len(),
                                face: RailFaceKind::LimitModel,
                            });
                        }
                        result.rho = railed;
                        result.final_value = rail_cost;
                        return certify_outer_optimality_at_terminal_fidelity(
                            obj,
                            config,
                            context,
                            result,
                            allow_certify_reseed,
                            fidelity,
                            Some(PolishWalk { record, judged }),
                        );
                    }
                    Ok((rail_cost, _)) => {
                        stopped = format!(
                            "{stopped}; railing coordinate(s) {natives:?} at their bounds changes \
                             the criterion by {:.3e}",
                            rail_cost - cost,
                        );
                    }
                    Err(error) => {
                        stopped = format!(
                            "{stopped}; railing coordinate(s) {natives:?} at their bounds failed \
                             to evaluate ({error})"
                        );
                    }
                }
            }
        }
        RailPlan::Representability { index, bound } => {
            // Box-KKT certifies only at a face that belongs to the model (#2627):
            // a literal bound says where the arithmetic stops, not where the
            // term's limit holds, so the mint refuses by that type.
            let native = native_coordinate(config.native_coordinate_order.as_deref(), index);
            refusal_source = StationarityBoundSource::RepresentabilityFace;
            stopped = format!(
                "{stopped}; coordinate {native} heads to its bound {bound:.6e}, a \
                 representability face rather than the term's derived limit model, so it \
                 is not railed"
            );
        }
        RailPlan::None => {}
    }
    // A linear rate ends the quadratic phase, not the polish (#3012): on an
    // exact-fit `y ~ s(x)` a polish that stopped there refused a sequence that one
    // more step certified. So once no limit face lowers the criterion, the polish
    // keeps taking Newton steps, each of which must again lower the criterion by
    // more than `band_f`, or settle where only the
    // decrease left is resolvable. That bounds the walk, since the criterion is
    // bounded below and a settling step's point must certify. Along a tail `λ̂²`
    // is the whole decrease left, so the walk certifies once that decrease is
    // below the band. The walk is never moved onto a representability face
    // (`damped_newton_step`), so box-KKT is never taken at one.
    if !quadratic
        && !settled
        && let Some(step) = newton_step.as_ref()
    {
        trial_ran = true;
        match damped_newton_step(obj, config, &result.rho, step, bounds, cost, evidence) {
            Ok(taken) => {
                log::debug!(
                    "[CERTIFICATE] {context}: Newton-decrement polish continues past a \
                     sub-quadratic rate ({stopped}): λ̂² went from {:.3e} to {:.3e} on this \
                     face (#3012)",
                    previous_lambda_sq.unwrap_or(f64::NAN),
                    evidence.lambda_sq,
                );
                return take_polish_step(
                    obj,
                    config,
                    context,
                    result,
                    allow_certify_reseed,
                    fidelity,
                    record,
                    taken,
                    cost,
                    evidence,
                );
            }
            Err(reason) => {
                backtrack_unresolved = true;
                stopped = format!("{stopped}; continuing past it, {reason}");
            }
        }
    }
    if trial_ran {
        // The trial moved the evaluator off the judged point. Re-own it, so
        // the checkpoint this refusal carries is the point it judged.
        obj.eval_with_order(&result.rho, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| {
                EstimationError::RemlOptimizationFailed(format!(
                    "{context}: failed to re-own the judged point after a stopped Newton \
                     polish: {err}"
                ))
            })?;
    }
    if backtrack_unresolved
        && !matches!(
            refusal_source,
            StationarityBoundSource::RepresentabilityFace
        )
    {
        refusal_source = StationarityBoundSource::NewtonBacktrackUnresolved;
    }
    result.final_hessian = analytic_hessian.clone();
    Err(outer_nonconvergence_error(
        context,
        &format!(
            "Newton-decrement above tolerance after polish: λ̂²={:.3e} > band_f={:.3e} \
             after {} Newton step(s) on the current face (ΔV per step {:?}; {} \
             rail(s) {:?}; λ̂² {:.3e} before, {:.3e} now): {stopped}",
            evidence.lambda_sq,
            evidence.band_f,
            record.decreases.len() - face_start,
            record.decreases,
            record.rails.len(),
            record
                .rails
                .iter()
                .map(|rail| rail.index)
                .collect::<Vec<_>>(),
            record.lambda_sq_before,
            evidence.lambda_sq,
        ),
        result,
        Some(projected_grad_norm),
        StationarityBound::from_ladder(stationarity_bound, refusal_source),
    ))
}

/// The evidence of a verdict whose Newton decrement the arithmetic resolves as
/// a decrease still to buy (#2954): `DecrementAboveTolerance`, and a
/// `DecrementUnresolved` whose `λ̂²` exceeds its own rounding `band_λ²` by more
/// than `band_f`. The latter's certificate is undecidable, but its decrease is
/// not: the decrease left to the minimum is more than either band. So both are polished, or
/// refused by name, and neither certifies.
pub(super) fn resolvable_decrease_evidence(
    verdict: &opt::DecrementVerdict,
) -> Option<&opt::DecrementEvidence> {
    match verdict {
        opt::DecrementVerdict::DecrementAboveTolerance(evidence) => Some(evidence),
        opt::DecrementVerdict::DecrementUnresolved(evidence)
            if evidence.lambda_sq - evidence.band_lambda_sq > evidence.band_f =>
        {
            Some(evidence)
        }
        _ => None,
    }
}

/// A polish step the criterion accepted: the point `t·p` along the Newton step
/// `p` reached, the criterion there, that evaluation's certificate evidence, and
/// whether it was a settling step.
struct DampedNewtonStep {
    point: Array1<f64>,
    cost: f64,
    judged: JudgedEvidence,
    t: f64,
    settling: bool,
}

/// The first step along the Newton step `p`, halving `t` from one, that lowers
/// the criterion by more than `band_f` (#3012).
///
/// A verdict refuses on the decrease left to the minimum, `λ̂²`, while the full
/// step's quadratic model promises `½λ̂²`. Where `band_f < λ̂² ≤ 2·band_f` the
/// decrease left is resolvable but no step's decrease is, so no step can show
/// the decrease `band_f` a kept step must. There the full step is a settling
/// step: it is taken where it does not raise the criterion by more than
/// `band_f`, and the point it reaches must certify (`polish_the_mint` refuses a
/// settled point that does not).
///
/// A full step outside the quadratic region can raise the criterion while the
/// decrement still promises a resolvable decrease; it is damped rather than
/// refused. The quadratic model predicts the decrease `t·(1 − t/2)·λ̂²` along the
/// Newton step, and once that prediction itself reaches `band_f` no shorter step
/// is predicted to buy a resolvable decrease, so the halving ends there: the
/// decrement against the band sets how far it goes. Each trial is judged at the
/// certificate's order ([`judged_cost`]). A trial whose evaluation fails is a
/// step too long for the objective, and is halved like one that raises the
/// criterion. So is one that would move a coordinate onto a representability
/// face: that would rail it at a face that is not the term's limit model, which
/// the polish never does (#2627).
fn damped_newton_step(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    rho: &Array1<f64>,
    step: &Array1<f64>,
    bounds: &(Array1<f64>, Array1<f64>),
    cost: f64,
    evidence: &opt::DecrementEvidence,
) -> Result<DampedNewtonStep, String> {
    let (lower, upper) = bounds;
    let settling = !(0.5 * evidence.lambda_sq > evidence.band_f);
    let mut t = 1.0_f64;
    let mut full_step = None;
    loop {
        let unclamped = rho + &(step * t);
        let point =
            Array1::from_iter((0..rho.len()).map(|k| unclamped[k].max(lower[k]).min(upper[k])));
        // A coordinate within its rail margin of a bound is railed there by the
        // certificate (`coordinate_rail_margin`), so entering that margin is
        // moving onto the face.
        let literal = (0..rho.len()).find(|&k| {
            let margin = super::bridges::coordinate_rail_margin(lower[k], upper[k]);
            let toward_upper = step[k] > 0.0;
            let reaches = if toward_upper {
                unclamped[k] >= upper[k] - margin
            } else {
                step[k] < 0.0 && unclamped[k] <= lower[k] + margin
            };
            reaches && rail_face_kind(config, k, toward_upper) == RailFaceKind::Representability
        });
        let outcome = if let Some(k) = literal {
            format!(
                "reaches coordinate {}'s representability face",
                native_coordinate(config.native_coordinate_order.as_deref(), k),
            )
        } else {
            match judged_cost(obj, &point) {
                Ok((trial_cost, judged))
                    if trial_cost.is_finite()
                        && if settling {
                            trial_cost - cost <= evidence.band_f
                        } else {
                            cost - trial_cost > evidence.band_f
                        } =>
                {
                    return Ok(DampedNewtonStep {
                        point,
                        cost: trial_cost,
                        judged,
                        t,
                        settling,
                    });
                }
                Ok((trial_cost, _)) => {
                    format!("changes the criterion by {:.3e}", trial_cost - cost)
                }
                Err(error) => format!("fails to evaluate ({error})"),
            }
        };
        if settling {
            return Err(format!(
                "the settling step, where λ̂²={:.3e} is resolvable against band_f={:.3e} but \
                 the full step's own ½λ̂² is not, {outcome}",
                evidence.lambda_sq, evidence.band_f,
            ));
        }
        let full = full_step.get_or_insert(outcome);
        let next = 0.5 * t;
        if !(next * (1.0 - 0.5 * next) * evidence.lambda_sq > evidence.band_f) {
            return Err(format!(
                "no step along the Newton step lowers the criterion by more than band_f: \
                 the full step {full}, and halving it to t={t:.3e} brings the quadratic \
                 model's own decrease t(1 − t/2)·λ̂² to band_f"
            ));
        }
        t = next;
    }
}

/// Take an accepted polish step and re-certify the point it reaches (#2954): the
/// recursion re-owns that point with a value, gradient and Hessian evaluation.
fn take_polish_step(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    allow_certify_reseed: bool,
    fidelity: CertificationFidelity,
    mut record: NewtonPolishRecord,
    taken: DampedNewtonStep,
    cost: f64,
    evidence: &opt::DecrementEvidence,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let DampedNewtonStep {
        point,
        cost: trial_cost,
        judged,
        t,
        settling,
    } = taken;
    log::debug!(
        "[CERTIFICATE] {context}: Newton-decrement polish {} {}: λ̂²={:.3e} is \
         resolvable against band_f={:.3e}; the Newton step at t={t:.3e} moves the \
         criterion from {cost:.9e} to {trial_cost:.9e}, and the point it reaches is judged \
         instead (#2954)",
        if settling { "settling step" } else { "step" },
        record.decreases.len() + 1,
        evidence.lambda_sq,
        evidence.band_f,
    );
    record.decreases.push(cost - trial_cost);
    record.settled = settling;
    result.rho = point;
    result.final_value = trial_cost;
    certify_outer_optimality_at_terminal_fidelity(
        obj,
        config,
        context,
        result,
        allow_certify_reseed,
        fidelity,
        Some(PolishWalk { record, judged }),
    )
}

/// The Newton step `−H_F⁻¹·g_F` on the coordinates `railed` leaves free, zero on
/// the railed ones, through a lower Cholesky factor of the symmetrized free
/// block (#2954). `None` when the free block is empty, not positive definite, or
/// a quantity is not finite.
pub(super) fn free_newton_step(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    railed: &[usize],
) -> Option<Array1<f64>> {
    let n = gradient.len();
    if hessian.nrows() != n || hessian.ncols() != n {
        return None;
    }
    let free: Vec<usize> = (0..n).filter(|k| !railed.contains(k)).collect();
    let q = free.len();
    if q == 0 {
        return None;
    }
    let mut factor = Array2::<f64>::zeros((q, q));
    for (a, &i) in free.iter().enumerate() {
        for (b, &j) in free.iter().enumerate() {
            factor[[a, b]] = 0.5 * (hessian[[i, j]] + hessian[[j, i]]);
        }
    }
    for j in 0..q {
        let mut pivot = factor[[j, j]];
        for k in 0..j {
            pivot -= factor[[j, k]] * factor[[j, k]];
        }
        if !(pivot.is_finite() && pivot > 0.0) {
            return None;
        }
        let root = pivot.sqrt();
        factor[[j, j]] = root;
        for i in (j + 1)..q {
            let mut value = factor[[i, j]];
            for k in 0..j {
                value -= factor[[i, k]] * factor[[j, k]];
            }
            factor[[i, j]] = value / root;
        }
    }
    let mut forward = Array1::<f64>::zeros(q);
    for a in 0..q {
        let mut value = -gradient[free[a]];
        for k in 0..a {
            value -= factor[[a, k]] * forward[k];
        }
        forward[a] = value / factor[[a, a]];
    }
    let mut reduced = Array1::<f64>::zeros(q);
    for a in (0..q).rev() {
        let mut value = forward[a];
        for k in (a + 1)..q {
            value -= factor[[k, a]] * reduced[k];
        }
        reduced[a] = value / factor[[a, a]];
    }
    let mut step = Array1::<f64>::zeros(n);
    for (a, &i) in free.iter().enumerate() {
        step[i] = reduced[a];
    }
    step.iter().all(|value| value.is_finite()).then_some(step)
}

/// The faces a stopped Newton polish tries to rail, in order (#2954).
pub(super) enum RailPlan {
    /// No free ρ coordinate's step heads to a bound it descends toward.
    None,
    /// The coordinate carrying the largest share of the step heads to a
    /// representability face.
    Representability { index: usize, bound: f64 },
    /// Sets of coordinates to move to their limit-model bounds together, each
    /// with the point the other coordinates move to (`None`: they stay).
    Faces(Vec<(Vec<(usize, f64)>, Option<Array1<f64>>)>),
}

/// Which faces a stopped Newton polish tries to rail (#2954).
///
/// A coordinate is eligible when it is a free log-smoothing coordinate this
/// polish has not railed before, whose step heads to one of its box bounds with
/// the criterion descending that way (`−g_k·p_k > 0`). The one carrying the
/// largest share `−g_k·p_k` of the Newton decrease `λ̂² = −gᵀp` leads: when its
/// bound is a representability face the polish refuses by that type. Otherwise
/// the faces tried are, in order: the leader alone; projected Newton's path
/// `P[ρ + t·p]` to the first limit-model face an outgoing coordinate reaches,
/// railing the coordinates that reach it and moving the rest along the step;
/// every eligible coordinate heading to its λ → ∞ limit together; and those
/// without the leader. An exponential tail can be carried by several penalties
/// at once (the margins of one tensor term shrink away together), and railing
/// one of them alone moves the criterion off that tail (#2349 round 7); an
/// interior coordinate with a large share can lead while the tail is the rest. Each face costs one
/// evaluation, and the recursion's KKT check releases any coordinate a face
/// railed that its bound does not hold.
///
/// Only a ρ coordinate is eligible: its box is a proxy for `λ = ∞` (or `0`),
/// which an exponential tail approaches without reaching, while a ψ box is a
/// real constraint the ordinary projection already decides (#2453). A
/// coordinate railed once is not railed again, so a rail released by its
/// bound's KKT sign cannot cycle.
pub(super) fn polish_rail_plan(
    rho: &Array1<f64>,
    projected_gradient: &Array1<f64>,
    step: &Array1<f64>,
    bounds: &(Array1<f64>, Array1<f64>),
    face: &[usize],
    rails: &[crate::model_types::NewtonPolishRail],
    layout: OuterThetaLayout,
    config: &OuterConfig,
) -> RailPlan {
    let (lower, upper) = bounds;
    let eligible: Vec<(usize, f64, bool, f64, RailFaceKind)> = (0..rho.len())
        .filter(|&k| {
            layout.coordinate_is_log_smoothing(k)
                && !face.contains(&k)
                && !rails.iter().any(|rail| rail.index == k)
                && k < lower.len()
                && k < upper.len()
                && k < step.len()
                && k < projected_gradient.len()
        })
        .filter_map(|k| {
            let share = -projected_gradient[k] * step[k];
            let (bound, at_upper) = if step[k] > 0.0 {
                (upper[k], true)
            } else if step[k] < 0.0 {
                (lower[k], false)
            } else {
                return None;
            };
            (share.is_finite() && share > 0.0 && bound.is_finite() && bound != rho[k]).then(|| {
                (
                    k,
                    bound,
                    at_upper,
                    share,
                    rail_face_kind(config, k, at_upper),
                )
            })
        })
        .collect();
    let Some(&(leader, leader_bound, _, _, leader_face)) =
        eligible.iter().max_by(|a, b| a.3.total_cmp(&b.3))
    else {
        return RailPlan::None;
    };
    if leader_face != RailFaceKind::LimitModel {
        return RailPlan::Representability {
            index: leader,
            bound: leader_bound,
        };
    }
    let infinite: Vec<(usize, f64)> = eligible
        .iter()
        .filter(|&&(_, _, at_upper, _, kind)| at_upper && kind == RailFaceKind::LimitModel)
        .map(|&(k, bound, ..)| (k, bound))
        .collect();
    let without_leader: Vec<(usize, f64)> = infinite
        .iter()
        .copied()
        .filter(|&(k, _)| k != leader)
        .collect();
    // Projected Newton's path `P[ρ + t·p]` to the first limit-model face an
    // outgoing coordinate reaches: every coordinate moves along the step, and
    // those that reach their bound there are railed on it.
    let limit_faces: Vec<(usize, f64)> = eligible
        .iter()
        .filter(|&&(.., kind)| kind == RailFaceKind::LimitModel)
        .map(|&(k, bound, ..)| (k, bound))
        .collect();
    let reach = limit_faces
        .iter()
        .map(|&(k, bound)| (bound - rho[k]) / step[k])
        .filter(|t| t.is_finite() && *t > 0.0)
        .fold(f64::INFINITY, f64::min);
    let path = reach.is_finite().then(|| {
        let point = Array1::from_iter(
            (0..rho.len()).map(|k| (rho[k] + reach * step[k]).max(lower[k]).min(upper[k])),
        );
        let reached: Vec<(usize, f64)> = limit_faces
            .iter()
            .copied()
            .filter(|&(k, bound)| (bound - rho[k]) / step[k] <= reach)
            .collect();
        (reached, Some(point))
    });
    let mut faces = vec![(vec![(leader, leader_bound)], None)];
    let candidates = path.into_iter().chain(
        [infinite, without_leader]
            .into_iter()
            .map(|face| (face, None)),
    );
    for candidate in candidates {
        if !candidate.0.is_empty() && !faces.contains(&candidate) {
            faces.push(candidate);
        }
    }
    RailPlan::Faces(faces)
}

/// One optimizer walk's Newton polish so far (#2954): the record its
/// certificate publishes, and the evidence the trial that reached the current
/// point published. The certificate recursion of that one walk owns it and drops
/// it on return, so no other walk, concurrent or later, can read it, whatever
/// point either visits.
pub(super) struct PolishWalk {
    pub(super) record: NewtonPolishRecord,
    judged: JudgedEvidence,
}

/// The certificate evidence a polish trial's evaluation published, with the
/// point it was published at.
struct JudgedEvidence {
    point: Array1<u64>,
    evidence: crate::estimate::outer_eval_capture::CertificateEvidence,
}

/// The criterion at a polish trial, from an evaluation of the order the
/// certificate judged the current point at (#2954). `band_f` bounds the error of
/// that evaluation's channels, factor and inner mode; a value-only probe may take
/// other paths (the Gram-statistic Gaussian value lane, its own inner warm start)
/// whose error `band_f` does not bound, so a decrease is read only between
/// evaluations of one kind.
///
/// The evaluation's certificate evidence comes back with it for
/// [`terminal_certificate_evidence`]: an objective that caches its evaluations
/// answers the certificate's own evaluation at an accepted trial from that cache
/// and publishes nothing the second time.
fn judged_cost(
    obj: &mut dyn OuterObjective,
    point: &Array1<f64>,
) -> Result<(f64, JudgedEvidence), EstimationError> {
    crate::estimate::outer_eval_capture::begin_certificate_parts_capture();
    let evaluation = obj.eval_with_order(point, OuterEvalOrder::ValueGradientHessian);
    let evidence = crate::estimate::outer_eval_capture::take_certificate_evidence();
    evaluation.map(|evaluation| {
        (
            evaluation.cost,
            JudgedEvidence {
                point: point.mapv(f64::to_bits),
                evidence,
            },
        )
    })
}

/// The evidence the certificate's own evaluation at `rho` stands on (#2954): what
/// that evaluation published, or, when it published nothing because it was
/// answered from the objective's cache, what this walk's polish trial published
/// evaluating the same point at the same order.
pub(super) fn terminal_certificate_evidence(
    rho: &Array1<f64>,
    walk: Option<&PolishWalk>,
    published: crate::estimate::outer_eval_capture::CertificateEvidence,
) -> crate::estimate::outer_eval_capture::CertificateEvidence {
    let published_nothing = published.parts.is_empty()
        && published.criterion.is_none()
        && published.inner_factor.is_none()
        && published.inner_residual.is_none();
    match walk.map(|walk| &walk.judged) {
        Some(judged) if published_nothing && judged.point == rho.mapv(f64::to_bits) => {
            judged.evidence.clone()
        }
        _ => published,
    }
}

/// The kind of face coordinate `k` sits on at its `upper` or lower bound
/// (#2954): the route's declared limit model, or a representability face.
pub(crate) fn rail_face_kind(config: &OuterConfig, k: usize, upper: bool) -> RailFaceKind {
    let is_limit = config
        .model_domain_limit_faces
        .as_ref()
        .and_then(|(lower_faces, upper_faces)| {
            if upper {
                upper_faces.get(k)
            } else {
                lower_faces.get(k)
            }
        })
        .copied()
        .unwrap_or(false);
    if is_limit {
        RailFaceKind::LimitModel
    } else {
        RailFaceKind::Representability
    }
}
