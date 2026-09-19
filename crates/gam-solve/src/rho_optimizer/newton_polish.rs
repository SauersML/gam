//! The mint's Newton polish (#2954): which verdicts it polishes, the step budget
//! quadratic convergence allows, the step on the free coordinates, and the
//! coordinate a stopped polish rails at a limit-model bound.

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
    pub(super) allow_tail_snap: bool,
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
/// is kept only when it lowers the criterion by more than `band_f`, and the
/// steps on one face are bounded by `newton_polish_step_budget`.
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
        allow_tail_snap,
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
            step_budget: newton_polish_step_budget(evidence.lambda_sq, evidence.band_f),
            rails: Vec::new(),
            entry: result.rho.to_vec(),
        },
    };
    // A rail changes the face, so the first verdict on the new face sets the
    // budget its own decrement allows.
    let face_start = record.rails.last().map_or(0, |rail| rail.steps_before);
    if !record.rails.is_empty() && record.decreases.len() == face_start {
        record.step_budget = newton_polish_step_budget(evidence.lambda_sq, evidence.band_f);
    }
    record.lambda_sq_after = evidence.lambda_sq;
    let newton_step = analytic_hessian
        .as_ref()
        .and_then(|hessian| free_newton_step(hessian, &projected_gradient, &decision.face));
    let mut trial_ran = false;
    let mut stopped = if record.decreases.len() - face_start >= record.step_budget {
        "the step budget quadratic convergence allows is spent".to_string()
    } else if let Some(step) = newton_step.as_ref() {
        let (lower, upper) = &bounds;
        let trial = Array1::from_iter(
            (0..result.rho.len()).map(|k| (result.rho[k] + step[k]).max(lower[k]).min(upper[k])),
        );
        trial_ran = true;
        match judged_cost(obj, &trial) {
            Ok((trial_cost, judged))
                if trial_cost.is_finite() && cost - trial_cost > evidence.band_f =>
            {
                log::info!(
                    "[CERTIFICATE] {context}: Newton-decrement polish step {}: ½λ̂²={:.3e} \
                     is resolvable against band_f={:.3e}; the Newton step lowers the \
                     criterion from {:.9e} to {trial_cost:.9e}, and the point it reaches \
                     is judged instead (#2954)",
                    record.decreases.len() + 1,
                    0.5 * evidence.lambda_sq,
                    evidence.band_f,
                    cost,
                );
                record.decreases.push(cost - trial_cost);
                result.rho = trial;
                result.final_value = trial_cost;
                return certify_outer_optimality_at_terminal_fidelity(
                    obj,
                    config,
                    context,
                    result,
                    allow_tail_snap,
                    fidelity,
                    Some(PolishWalk { record, judged }),
                );
            }
            Ok((trial_cost, _)) => format!(
                "the Newton step changes the criterion by {:.3e}, which is not a decrease \
                 above band_f",
                trial_cost - cost,
            ),
            Err(error) => format!("the Newton step's evaluation failed ({error})"),
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
                        log::info!(
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
                            allow_tail_snap,
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
    result.final_hessian = analytic_hessian.clone();
    Err(outer_nonconvergence_error(
        context,
        &format!(
            "Newton-decrement above tolerance after polish: ½λ̂²={:.3e} > band_f={:.3e} \
             after {} of {} Newton step(s) on the current face (ΔV per step {:?}; {} \
             rail(s) {:?}; λ̂² {:.3e} before, {:.3e} now): {stopped}",
            0.5 * evidence.lambda_sq,
            evidence.band_f,
            record.decreases.len() - face_start,
            record.step_budget,
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
/// `DecrementUnresolved` whose `½λ̂²` exceeds its own rounding `band_λ²` by more
/// than `band_f`. The latter's certificate is undecidable, but its decrease is
/// not: a Newton step buys more than either band. So both are polished, or
/// refused by name, and neither certifies.
pub(super) fn resolvable_decrease_evidence(
    verdict: &opt::DecrementVerdict,
) -> Option<&opt::DecrementEvidence> {
    match verdict {
        opt::DecrementVerdict::DecrementAboveTolerance(evidence) => Some(evidence),
        opt::DecrementVerdict::DecrementUnresolved(evidence)
            if 0.5 * evidence.lambda_sq - evidence.band_lambda_sq > evidence.band_f =>
        {
            Some(evidence)
        }
        _ => None,
    }
}

/// The Newton-step budget a mint's polish may spend, from quadratic convergence
/// (#2954).
///
/// For a self-concordant criterion the Newton decrement satisfies `λ₊ ≤ 2λ²`
/// once `λ ≤ 1/4` (Boyd & Vandenberghe, *Convex Optimization*, §9.6.3). So
/// `2λ_t ≤ (2λ_0)^(2^t)`, and `½λ_t² ≤ band_f` as soon as
/// `2^(t+1) ≥ ln(8·band_f)/ln(2λ_0)`. The budget is the smallest such `t`, at
/// least one step. Outside `λ_0 ≤ 1/4` the bound gives no step count and the
/// budget is zero. A criterion that needs more steps than this converges slower
/// than Newton's quadratic rate, and its mint is refused by name.
pub(super) fn newton_polish_step_budget(lambda_sq: f64, band_f: f64) -> usize {
    let lambda = lambda_sq.sqrt();
    if !(lambda.is_finite() && lambda > 0.0 && lambda <= 0.25 && band_f.is_finite() && band_f > 0.0)
    {
        return 0;
    }
    let ratio = (8.0 * band_f).ln() / (2.0 * lambda).ln();
    if !(ratio.is_finite() && ratio > 1.0) {
        return 1;
    }
    (ratio.log2() - 1.0).ceil().max(1.0) as usize
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
