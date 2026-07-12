//! Reactive three-leg continuation for a non-finite K≥2 SAE outer seed.
//!
//! # Three endpoint legs coupled by one scalar parameter
//!
//! One objective evaluation installs and solves all three homotopy legs at the
//! same `s`:
//!
//! 1. **log-ρ** — the objective's legal upper-box endpoint down to its literal
//!    target. At large penalty strength the penalized Hessian dominates the
//!    likelihood Hessian.
//! 2. **Assignment temperature τ** — diffuse softmax / IBP relaxation (high τ)
//!    sharpened toward the objective's literal target. High τ makes the
//!    assignment map smooth and far from the combinatorial argmax cliff.
//! 3. **Isometry weights** — zero entry weights ramped to the objective's
//!    literal per-penalty vector. A loose gauge leaves the decoder free to find
//!    a good fit before the gauge pins it.
//!
//! [`ContinuationPath`] advances all three **in lockstep** along a single
//! scalar path parameter `s ∈ [1 → 0]`. `s = 1` is the *entry regime*: legal
//! upper-box ρ, high τ, and loose isometry. `s = 0` is the real objective: target ρ\*,
//! sharp τ, tight isometry. The path walks `s` monotonically down, advancing
//! the three literal waypoint values together.
//!
//! # Entry is always the heavy-smoothing regime
//!
//! There is no "solve cold at the real objective" entry. The only entry is
//! `s = 1`, where every leg is at its smoothing extreme. A K≥2 SAE joint fit
//! either reaches the literal target through accepted coupled waypoints or
//! returns a typed refusal; it never fabricates arrival.
//!
//! # The accepted path is monotone; failed attempts refine their distance
//!
//! If a downward step's inner solve struggles, the last accepted waypoint is
//! retained and only the next attempted distance is halved. No independent
//! waypoint count, wall-clock deadline, or evaluation ceiling can fabricate an
//! arrival. A successful solve at the literal `s = 0` target is the only
//! [`ContinuationStep::Arrived`] value. If repeated refinement can no longer
//! produce a strictly smaller representable waypoint, [`ContinuationPath::step`]
//! returns a typed non-convergence error.
//!
//! Full objective checkpoints, rather than coefficient-only fallback state,
//! make each accepted waypoint independent of mutations from refused trials.

use ndarray::Array1;

use crate::estimate::reml::continuation::{ContinuationState, eval_step};
use crate::inner_status::InnerFailure;
use crate::rho_optimizer::{OuterEvalOrder, OuterObjective};

/// The endpoints of one coupled annealing leg, in path-parameter terms.
/// `at_entry` is the value at `s = 1` (heavy-smoothing regime); `at_target`
/// is the value at `s = 0` (real objective). Interpolation is in the leg's
/// coordinate: ρ is already stored as log-precision, while temperature and
/// isometry weights are literal objective scalars.
#[derive(Debug, Clone, Copy)]
struct LegEndpoints {
    /// Value at `s = 1`: the smoothing-extreme entry regime.
    at_entry: f64,
    /// Value at `s = 0`: the real-objective target.
    at_target: f64,
}

impl LegEndpoints {
    /// Construct from an entry value and a target value.
    #[must_use]
    fn new(at_entry: f64, at_target: f64) -> Self {
        Self {
            at_entry,
            at_target,
        }
    }

    /// Linear interpolation in the leg's natural coordinate at path parameter
    /// `s ∈ [0, 1]`: `s = 1 → at_entry`, `s = 0 → at_target`. The caller passes
    /// values in the coordinate the objective consumes, so a convex blend is
    /// also the literal waypoint value installed or evaluated.
    #[must_use]
    fn at(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, 1.0);
        // Endpoint waypoints are literal objective state, not merely values
        // numerically close to it. The affine expression below can lose an
        // endpoint bit through cancellation (for example, target +
        // (entry - target) need not be bitwise-identical to entry), which
        // breaks transactional restoration of the exact accepted waypoint.
        if s.to_bits() == 0.0_f64.to_bits() {
            return self.at_target;
        }
        if s.to_bits() == 1.0_f64.to_bits() {
            return self.at_entry;
        }
        self.at_target + s * (self.at_entry - self.at_target)
    }
}

/// Literal scalar state of an objective at one coupled-domain waypoint.
///
/// The assignment temperature is a single global scalar. Isometry weights are
/// retained one-per-registered penalty: collapsing them into one synthetic
/// number would lose the objective's actual target state when several
/// isometry penalties carry different weights.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationScalarState {
    pub assignment_temperature: f64,
    pub isometry_weights: Vec<f64>,
}

impl ContinuationScalarState {
    pub fn new(assignment_temperature: f64, isometry_weights: Vec<f64>) -> Result<Self, String> {
        if !(assignment_temperature.is_finite() && assignment_temperature > 0.0) {
            return Err(format!(
                "continuation assignment temperature must be finite and positive; got \
                 {assignment_temperature}"
            ));
        }
        if let Some((index, weight)) = isometry_weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| !(weight.is_finite() && *weight >= 0.0))
        {
            return Err(format!(
                "continuation isometry weight[{index}] must be finite and non-negative; got \
                 {weight}"
            ));
        }
        Ok(Self {
            assignment_temperature,
            isometry_weights,
        })
    }

    #[must_use]
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        self.assignment_temperature.to_bits() == other.assignment_temperature.to_bits()
            && self.isometry_weights.len() == other.isometry_weights.len()
            && self
                .isometry_weights
                .iter()
                .zip(&other.isometry_weights)
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }
}

/// Objective-owned scalar endpoints for reactive domain entry. The target is
/// the literal state of the real objective; the entry is a smoother state
/// derived by that objective from its own routing and penalty geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationScalarContract {
    entry: ContinuationScalarState,
    target: ContinuationScalarState,
}

impl ContinuationScalarContract {
    pub fn new(
        entry: ContinuationScalarState,
        target: ContinuationScalarState,
    ) -> Result<Self, String> {
        if entry.isometry_weights.len() != target.isometry_weights.len() {
            return Err(format!(
                "continuation entry/target isometry dimensions differ: {} != {}",
                entry.isometry_weights.len(),
                target.isometry_weights.len()
            ));
        }
        if entry.assignment_temperature < target.assignment_temperature {
            return Err(format!(
                "continuation entry temperature {} is sharper than literal target {}",
                entry.assignment_temperature, target.assignment_temperature
            ));
        }
        if let Some((index, (&entry_weight, &target_weight))) = entry
            .isometry_weights
            .iter()
            .zip(&target.isometry_weights)
            .enumerate()
            .find(|(_, (entry_weight, target_weight))| entry_weight > target_weight)
        {
            return Err(format!(
                "continuation entry isometry weight[{index}]={entry_weight} exceeds literal target {target_weight}"
            ));
        }
        Ok(Self { entry, target })
    }

    #[must_use]
    pub fn entry(&self) -> &ContinuationScalarState {
        &self.entry
    }

    #[must_use]
    pub fn target(&self) -> &ContinuationScalarState {
        &self.target
    }

    #[must_use]
    pub fn at(&self, s: f64) -> ContinuationScalarState {
        let s = s.clamp(0.0, 1.0);
        let temperature = LegEndpoints::new(
            self.entry.assignment_temperature,
            self.target.assignment_temperature,
        )
        .at(s);
        let isometry_weights = self
            .entry
            .isometry_weights
            .iter()
            .zip(&self.target.isometry_weights)
            .map(|(&entry, &target)| LegEndpoints::new(entry, target).at(s))
            .collect();
        ContinuationScalarState {
            assignment_temperature: temperature,
            isometry_weights,
        }
    }
}

/// Endpoint state that [`ContinuationPath`] owns. It computes every leg from
/// the same `s`; there are no independently advancing schedule objects.
#[derive(Debug, Clone)]
struct CoupledSchedules {
    /// Log-ρ endpoints, **per-component** (one entry per outer coordinate).
    /// Entry is the objective's legal upper box and target is literal ρ\*.
    rho_entry: Array1<f64>,
    /// ρ\* — the real-objective smoothing vector at `s = 0`.
    rho_target: Array1<f64>,
    /// Typed scalar entry/target state supplied by the objective itself.
    scalars: ContinuationScalarContract,
}

impl CoupledSchedules {
    /// The coupled lockstep target value of every scalar leg at path parameter
    /// `s`. ρ is a vector and is returned by [`Self::rho_target_at`].
    #[must_use]
    fn scalar_targets_at(&self, s: f64) -> ContinuationScalarState {
        self.scalars.at(s)
    }

    /// Exact log-ρ waypoint at path parameter `s`: a convex blend per
    /// component from the legal upper-box entry to literal ρ\*.
    #[must_use]
    fn rho_target_at(&self, s: f64) -> Array1<f64> {
        assert_eq!(
            self.rho_entry.len(),
            self.rho_target.len(),
            "ContinuationPath: ρ entry/target dimension mismatch"
        );
        let s = s.clamp(0.0, 1.0);
        // Preserve the literal box and target vectors at the two contract
        // endpoints. Besides making the scalar and rho legs symmetric, this
        // prevents affine-rounding from changing a bound bit at entry.
        if s.to_bits() == 0.0_f64.to_bits() {
            return self.rho_target.clone();
        }
        if s.to_bits() == 1.0_f64.to_bits() {
            return self.rho_entry.clone();
        }
        let mut out = self.rho_target.clone();
        for i in 0..out.len() {
            out[i] = self.rho_target[i] + s * (self.rho_entry[i] - self.rho_target[i]);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Typed outcomes for accepted progress and attempted-distance refinement.
// ─────────────────────────────────────────────────────────────────────────

/// Outcome of one [`ContinuationPath`] waypoint step. The defining structural
/// property: **there is no false-arrival arm.** A successful step enters,
/// descends, arrives with its solved target state, or reports that the next
/// attempted distance was refined. Non-convergence is returned by
/// [`ContinuationPath::step`] as a typed error rather than encoded as arrival.
#[derive(Debug, Clone)]
pub(crate) enum ContinuationStep {
    /// The objective installed and solved the literal heavy scalar entry at
    /// `s = 1`. Carries the accepted waypoint state that warms the first descent.
    Entered { state: ContinuationState },
    /// `s` was lowered toward `0` and the inner solve at the new waypoint
    /// succeeded. Carries the accepted waypoint state and the new `s`.
    Descended { s: f64, state: ContinuationState },
    /// `s` reached `0`: the path arrived at the real objective (ρ\*, τ_min,
    /// tight isometry). Terminal-but-successful; the criterion is the real
    /// objective's, identical for cold and warm entry (#969).
    Arrived { state: ContinuationState },
    /// The attempted waypoint did not solve, so the accepted `s` remains
    /// unchanged and the next attempted distance is smaller. Carries the last
    /// accepted `s` and the evidence that requested refinement.
    Refined { s: f64, reason: RefinementReason },
}

/// Why the next waypoint distance was refined. Purely diagnostic: the last
/// accepted waypoint remains installed and no progress is fabricated.
#[derive(Debug, Clone)]
pub(crate) enum RefinementReason {
    /// The exact coupled waypoint evaluation did not converge. The underlying
    /// failure is kept for logging; the path retries from the last accepted
    /// state with a smaller attempted distance.
    WaypointStruggled(InnerFailure),
}

// ─────────────────────────────────────────────────────────────────────────
//  The ContinuationPath object.
// ─────────────────────────────────────────────────────────────────────────

/// Coupled continuation path. Owns the three endpoint legs and the scalar path
/// parameter `s`, and drives the K≥2 SAE joint fit down the coupled
/// homotopy. Entry is always the legal heavy endpoint at `s = 1`, and
/// only a solved literal target can produce arrival.
///
/// [`ContinuationPath::step`] transactionally installs and evaluates one exact
/// waypoint at a time. The path itself owns the accepted warm trajectory.
#[derive(Debug, Clone)]
pub struct ContinuationPath {
    schedules: CoupledSchedules,
    /// Last successfully solved path parameter. Starts at `1.0` (entry regime)
    /// and walks monotonically toward `0.0`.
    s: f64,
    /// Current descent step in `s`. A failed attempted waypoint halves it; a
    /// successful waypoint doubles it up to the remaining distance. There is
    /// no numeric floor: inability to form a strictly smaller representable
    /// waypoint is a typed non-convergence result.
    s_step: f64,
    /// The most recent converged waypoint state. `None` until the first leg
    /// converges; every later waypoint starts from this accepted full objective
    /// state and beta hint. A failed attempted waypoint never overwrites it.
    warm: Option<ContinuationState>,
}

impl ContinuationPath {
    /// Build at the legal heavy endpoint `s = 1`. Reactive continuation cannot
    /// begin cold at the literal target; failure to solve this entry is typed.
    #[must_use]
    fn enter(schedules: CoupledSchedules) -> Self {
        Self {
            schedules,
            s: 1.0,
            s_step: 1.0,
            warm: None,
        }
    }

    /// Build from a concrete log-ρ target and the objective's legal upper box.
    /// The upper box is the heavy entry endpoint; the scalar endpoints come
    /// from the typed objective contract. Every endpoint must be finite, and
    /// each upper entry must be at least its target.
    pub fn heavy_entry_for_rho(
        rho_target: Array1<f64>,
        bounds_upper: Array1<f64>,
        scalars: ContinuationScalarContract,
    ) -> Result<Self, gam_problem::EstimationError> {
        if rho_target.len() != bounds_upper.len() {
            return Err(gam_problem::EstimationError::RemlOptimizationFailed(
                format!(
                    "reactive continuation rho target/bounds dimensions differ: {} != {}",
                    rho_target.len(),
                    bounds_upper.len()
                ),
            ));
        }
        for (index, (&target, &entry)) in
            rho_target.iter().zip(bounds_upper.iter()).enumerate()
        {
            if !(target.is_finite() && entry.is_finite() && entry >= target) {
                return Err(gam_problem::EstimationError::RemlOptimizationFailed(
                    format!(
                        "reactive continuation rho endpoint[{index}] must be finite with legal \
                         upper entry >= target; entry={entry}, target={target}"
                    ),
                ));
            }
        }
        // The legal upper box is the literal heavy-penalty endpoint. This uses
        // objective-owned geometry rather than a private log-offset heuristic,
        // and makes rho move under the same `s` as temperature and isometry.
        let schedules = couple_schedules(
            bounds_upper.clone(),
            rho_target,
            scalars,
        );
        Ok(Self::enter(schedules))
    }

    /// Current path parameter `s ∈ [0, 1]`.
    #[must_use]
    pub fn s(&self) -> f64 {
        self.s
    }

    /// The scalar leg targets (τ, isometry weight) at the current `s`. The
    /// wiring agent installs these before the inner solve at this waypoint.
    #[must_use]
    pub fn current_scalar_targets(&self) -> ContinuationScalarState {
        self.schedules.scalar_targets_at(self.s)
    }

    /// Refine the next descent after a failed attempted waypoint. `s` remains
    /// the last successfully solved waypoint; only the attempted distance is
    /// halved, so the accepted path is monotone and never fabricates progress.
    fn refine_step(&mut self) {
        self.s_step *= 0.5;
    }

    /// Take one waypoint step down the coupled homotopy.
    ///
    /// 1. Lower `s` by the current step toward `0`.
    /// 2. Install the exact scalar state and evaluate the exact log-ρ waypoint,
    ///    with the last accepted inner β carried warm.
    /// 3. On evaluation success: [`ContinuationStep::Descended`] (or
    ///    [`ContinuationStep::Arrived`] if `s` reached `0`).
    /// 4. On error or non-finite evidence: retain the last successful waypoint and halve the
    ///    attempted distance. If no strictly smaller representable waypoint
    ///    remains, return a typed non-convergence error.
    ///
    /// `obj` is the SAE joint outer objective (`SaeManifoldOuterObjective`,
    /// which is an [`OuterObjective`]). `initial_beta` warms the inner solve;
    /// pass the empty array for a cold entry.
    pub(crate) fn step(
        &mut self,
        obj: &mut dyn OuterObjective,
        initial_beta: &Array1<f64>,
    ) -> Result<ContinuationStep, gam_problem::EstimationError> {
        // The cold leg solves the literal heavy entry at s=1 before any
        // descent. Every later leg lowers s by one path step. This makes the
        // heavy entry an evaluated waypoint rather than an unevaluated
        // endpoint that the old implementation skipped on its first call.
        let entering = self.warm.is_none();
        let s_next = if entering {
            1.0
        } else {
            (self.s - self.s_step).max(0.0)
        };
        if !entering && !(s_next < self.s) {
            return Err(gam_problem::EstimationError::RemlOptimizationFailed(
                format!(
                    "reactive domain continuation cannot form a smaller representable waypoint \
                     below accepted s={:.17e}",
                    self.s
                ),
            ));
        }

        // Snapshot the complete accepted objective state before installing a
        // trial. A coefficient hint alone cannot restore latent coordinates,
        // routing logits, or decoder frames after a failed inner solve.
        obj.begin_reactive_domain_waypoint()?;

        // Install the objective-owned scalar state before evaluating rho. This
        // is the actual coupling seam: mutating private schedule copies would
        // leave the evaluated objective unchanged.
        let scalar_state = self.schedules.scalar_targets_at(s_next);
        if let Err(install_error) = obj.install_reactive_domain_scalar_state(&scalar_state) {
            return match obj.rollback_reactive_domain_waypoint() {
                Ok(()) => Err(install_error),
                Err(rollback_error) => Err(
                    gam_problem::EstimationError::RemlOptimizationFailed(format!(
                        "reactive coupled waypoint installation failed ({install_error}); \
                         full-state rollback also failed ({rollback_error})"
                    )),
                ),
            };
        }

        // One path attempt is exactly one objective evaluation at the shared
        // `(rho(s), scalar(s))` waypoint. There is no nested rho scheduler: its
        // step floors, retry counts, and private clock would decouple rho from
        // the scalar legs. The last accepted beta is the only warm payload.
        let rho_waypoint = self.schedules.rho_target_at(s_next);
        let (beta_seed, accepted_steps) = match self.warm.as_ref() {
            Some(start) => (
                start
                    .last_eval
                    .inner_beta_hint
                    .clone()
                    .unwrap_or_else(|| start.last_beta.clone()),
                start.steps_accepted,
            ),
            None => (initial_beta.clone(), 0),
        };
        let evaluation = eval_step(
            obj,
            &rho_waypoint,
            &beta_seed,
            OuterEvalOrder::Value,
        )
        .and_then(|eval| {
            if eval.cost.is_finite() {
                Ok(eval)
            } else {
                Err(InnerFailure::LikelihoodFailure(format!(
                    "reactive coupled waypoint at s={s_next:.17e} returned non-finite evidence {}",
                    eval.cost
                )))
            }
        });

        Ok(match evaluation {
            Ok(eval) => {
                if let Err(commit_error) = obj.commit_reactive_domain_waypoint(&rho_waypoint) {
                    return match obj.rollback_reactive_domain_waypoint() {
                        Ok(()) => Err(commit_error),
                        Err(rollback_error) => Err(
                            gam_problem::EstimationError::RemlOptimizationFailed(format!(
                                "reactive coupled waypoint commit failed ({commit_error}); \
                                 full-state rollback also failed ({rollback_error})"
                            )),
                        ),
                    };
                }
                let state = ContinuationState {
                    last_rho: rho_waypoint,
                    last_eval: eval,
                    last_beta: beta_seed,
                    steps_accepted: accepted_steps + 1,
                };
                self.warm = Some(state.clone());
                self.s = s_next;
                // Successful progress permits a naturally larger next step,
                // bounded only by the distance remaining to the literal target.
                self.s_step = (self.s_step * 2.0).min(self.s);
                if entering {
                    ContinuationStep::Entered { state }
                } else if self.s <= 0.0 {
                    ContinuationStep::Arrived { state }
                } else {
                    ContinuationStep::Descended { s: self.s, state }
                }
            }
            Err(failure) => {
                obj.rollback_reactive_domain_waypoint().map_err(|rollback_error| {
                    gam_problem::EstimationError::RemlOptimizationFailed(format!(
                        "reactive coupled waypoint failed ({}), and full-state rollback failed \
                         ({rollback_error})",
                        failure.message()
                    ))
                })?;
                if entering {
                    return Err(gam_problem::EstimationError::RemlOptimizationFailed(
                        format!(
                            "reactive domain entry failed at the objective-owned scalar entry: {}",
                            failure.message()
                        ),
                    ));
                }
                self.refine_step();
                let refined_next = (self.s - self.s_step).max(0.0);
                if !(refined_next < self.s) {
                    return Err(gam_problem::EstimationError::RemlOptimizationFailed(
                        format!(
                            "reactive domain continuation cannot form a smaller representable \
                             waypoint below s={:.17e} after coupled-waypoint non-convergence: {}",
                            self.s,
                            failure.message()
                        ),
                    ));
                }
                ContinuationStep::Refined {
                    s: self.s,
                    reason: RefinementReason::WaypointStruggled(failure),
                }
            }
        })
    }
}

/// Build a coupled path from the rho box and the typed scalar contract supplied
/// by the objective.
///
/// `rho_target` is ρ\* (the real objective); `rho_entry` is the objective's
/// legal upper-box endpoint. Each is evaluated at the same `s` as the scalar
/// contract.
#[must_use]
fn couple_schedules(
    rho_entry: Array1<f64>,
    rho_target: Array1<f64>,
    scalars: ContinuationScalarContract,
) -> CoupledSchedules {
    CoupledSchedules {
        rho_entry,
        rho_target,
        scalars,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_contract() -> ContinuationScalarContract {
        ContinuationScalarContract::new(
            ContinuationScalarState::new(2.0, vec![0.01, 0.02]).expect("valid entry"),
            ContinuationScalarState::new(0.1, vec![1.0, 2.0]).expect("valid target"),
        )
        .expect("matching scalar dimensions")
    }

    fn schedules() -> CoupledSchedules {
        couple_schedules(
            Array1::from_vec(vec![5.0, 5.0]),
            Array1::from_vec(vec![0.0, 0.0]),
            scalar_contract(),
        )
    }

    #[test]
    fn literal_endpoint_bits_survive_without_affine_rounding() {
        let scalar_entry =
            ContinuationScalarState::new(2.0, vec![-0.0, 0.01]).expect("valid entry");
        let scalar_target =
            ContinuationScalarState::new(0.1, vec![1.0, 2.0]).expect("valid target");
        let contract =
            ContinuationScalarContract::new(scalar_entry.clone(), scalar_target.clone())
                .expect("ordered scalar endpoints");
        assert!(contract.at(1.0).bitwise_eq(&scalar_entry));
        assert!(contract.at(0.0).bitwise_eq(&scalar_target));

        let rho_entry = Array1::from_vec(vec![0.01, 0.02]);
        let rho_target = Array1::from_vec(vec![-0.0, -0.1]);
        let schedules = couple_schedules(rho_entry.clone(), rho_target.clone(), contract);
        assert!(
            schedules
                .rho_target_at(1.0)
                .iter()
                .zip(rho_entry.iter())
                .all(|(actual, literal)| actual.to_bits() == literal.to_bits())
        );
        assert!(
            schedules
                .rho_target_at(0.0)
                .iter()
                .zip(rho_target.iter())
                .all(|(actual, literal)| actual.to_bits() == literal.to_bits())
        );
    }

    #[test]
    fn entry_is_the_heavy_smoothing_regime() {
        let path = ContinuationPath::enter(schedules());
        assert_eq!(
            path.s(),
            1.0,
            "entry must be s = 1 (heavy-smoothing regime)"
        );
        let targets = path.current_scalar_targets();
        assert!(targets.bitwise_eq(scalar_contract().entry()));
        // Log-ρ at s = 1 is the supplied legal upper-box endpoint.
        let rho = path.schedules.rho_target_at(path.s());
        assert!(
            rho.iter()
                .all(|entry| entry.to_bits() == 5.0_f64.to_bits())
        );
    }

    #[test]
    fn target_endpoint_is_the_real_objective() {
        let sch = schedules();
        let targets0 = sch.scalar_targets_at(0.0);
        assert!(targets0.bitwise_eq(scalar_contract().target()));
        let rho0 = sch.rho_target_at(0.0);
        assert!(
            (rho0[0]).abs() < 1e-12 && (rho0[1]).abs() < 1e-12,
            "s=0 ρ = ρ*"
        );
    }

    #[test]
    fn legs_move_in_lockstep_along_s() {
        let sch = schedules();
        // Halfway down the path, every leg is halfway (in its natural coord)
        // between entry and target.
        let mid = sch.scalar_targets_at(0.5);
        assert!((mid.assignment_temperature - (0.1 + 0.5 * (2.0 - 0.1))).abs() < 1e-12);
        assert!((mid.isometry_weights[0] - (1.0 + 0.5 * (0.01 - 1.0))).abs() < 1e-12);
        assert!((mid.isometry_weights[1] - (2.0 + 0.5 * (0.02 - 2.0))).abs() < 1e-12);
        let rho_mid = sch.rho_target_at(0.5);
        assert!((rho_mid[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn heavy_entry_starts_in_the_heavy_regime() {
        let path = ContinuationPath::heavy_entry_for_rho(
            Array1::zeros(1),
            Array1::from_elem(1, 10.0),
            scalar_contract(),
        )
        .expect("finite ordered rho endpoints");
        assert_eq!(path.s(), 1.0, "heavy_entry must enter at s = 1");
    }

    #[test]
    fn heavy_entry_refuses_nonfinite_or_reversed_rho_endpoints() {
        let nonfinite = ContinuationPath::heavy_entry_for_rho(
            Array1::zeros(1),
            Array1::from_vec(vec![f64::INFINITY]),
            scalar_contract(),
        )
        .expect_err("non-finite legal entry must be typed refusal");
        assert!(nonfinite.to_string().contains("must be finite"));

        let reversed = ContinuationPath::heavy_entry_for_rho(
            Array1::from_vec(vec![2.0]),
            Array1::from_vec(vec![1.0]),
            scalar_contract(),
        )
        .expect_err("entry below target must be typed refusal");
        assert!(reversed.to_string().contains("entry >= target"));
    }

    #[derive(Default)]
    struct RecordingObjective {
        orders: Vec<OuterEvalOrder>,
        rho_evaluated: Vec<Array1<f64>>,
        seed_count: usize,
        installed: Vec<ContinuationScalarState>,
        installed_current: Option<ContinuationScalarState>,
        full_state_marker: usize,
        checkpoint_full_state_marker: Option<usize>,
        checkpoint_installed_current: Option<Option<ContinuationScalarState>>,
        fail_literal_target_once: bool,
        trial_entry_markers: Vec<usize>,
    }

    impl OuterObjective for RecordingObjective {
        fn capability(&self) -> crate::rho_optimizer::OuterCapability {
            crate::rho_optimizer::OuterCapability {
                gradient: gam_problem::Derivative::Analytic,
                hessian: crate::rho_optimizer::DeclaredHessianForm::Unavailable,
                n_params: 2,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }

        fn eval_cost(
            &mut self,
            rho: &Array1<f64>,
        ) -> Result<f64, crate::model_types::EstimationError> {
            Ok(rho.iter().map(|v| v * v).sum())
        }

        fn eval(
            &mut self,
            rho: &Array1<f64>,
        ) -> Result<gam_problem::OuterEval, crate::model_types::EstimationError> {
            Ok(gam_problem::OuterEval {
                cost: self.eval_cost(rho)?,
                gradient: Array1::zeros(rho.len()),
                hessian: gam_problem::HessianValue::Unavailable,
                inner_beta_hint: Some(Array1::from_vec(vec![1.0, self.seed_count as f64])),
            })
        }

        fn eval_with_order(
            &mut self,
            rho: &Array1<f64>,
            order: OuterEvalOrder,
        ) -> Result<gam_problem::OuterEval, crate::model_types::EstimationError> {
            self.orders.push(order);
            self.rho_evaluated.push(rho.clone());
            self.trial_entry_markers.push(self.full_state_marker);
            if self.fail_literal_target_once
                && self
                    .installed_current
                    .as_ref()
                    .is_some_and(|state| state.bitwise_eq(scalar_contract().target()))
            {
                self.fail_literal_target_once = false;
                self.full_state_marker = usize::MAX;
                return Ok(gam_problem::OuterEval::infeasible(rho.len()));
            }
            let mut eval = self.eval(rho)?;
            self.full_state_marker += 1;
            if matches!(order, OuterEvalOrder::Value) {
                eval.gradient = Array1::zeros(0);
                eval.hessian = gam_problem::HessianValue::Unavailable;
            }
            Ok(eval)
        }

        fn reset(&mut self) {}

        fn seed_inner_state(
            &mut self,
            beta: &Array1<f64>,
        ) -> Result<crate::rho_optimizer::SeedOutcome, crate::model_types::EstimationError>
        {
            self.seed_count += beta.len().max(1);
            Ok(crate::rho_optimizer::SeedOutcome::Installed)
        }

        fn reactive_domain_scalar_contract(
            &self,
        ) -> Result<Option<ContinuationScalarContract>, crate::model_types::EstimationError>
        {
            Ok(Some(scalar_contract()))
        }

        fn install_reactive_domain_scalar_state(
            &mut self,
            state: &ContinuationScalarState,
        ) -> Result<(), crate::model_types::EstimationError> {
            self.installed.push(state.clone());
            self.installed_current = Some(state.clone());
            Ok(())
        }

        fn begin_reactive_domain_waypoint(
            &mut self,
        ) -> Result<(), crate::model_types::EstimationError> {
            assert!(self.checkpoint_full_state_marker.is_none());
            self.checkpoint_full_state_marker = Some(self.full_state_marker);
            self.checkpoint_installed_current = Some(self.installed_current.clone());
            Ok(())
        }

        fn commit_reactive_domain_waypoint(
            &mut self,
            _: &Array1<f64>,
        ) -> Result<(), crate::model_types::EstimationError> {
            self.checkpoint_full_state_marker
                .take()
                .expect("active waypoint checkpoint");
            self.checkpoint_installed_current
                .take()
                .expect("active scalar checkpoint");
            Ok(())
        }

        fn rollback_reactive_domain_waypoint(
            &mut self,
        ) -> Result<(), crate::model_types::EstimationError> {
            self.full_state_marker = self
                .checkpoint_full_state_marker
                .take()
                .expect("active waypoint checkpoint");
            self.installed_current = self
                .checkpoint_installed_current
                .take()
                .expect("active scalar checkpoint");
            Ok(())
        }
    }

    #[test]
    fn coupled_path_prewarm_requests_value_only_evals() {
        let mut path = ContinuationPath::enter(schedules());
        let mut obj = RecordingObjective::default();
        let initial_beta = Array1::zeros(0);

        let step = path
            .step(&mut obj, &initial_beta)
            .expect("the heavy entry must solve");
        assert!(
            matches!(step, ContinuationStep::Entered { .. }),
            "the first coupled-path call must solve the literal entry waypoint"
        );
        assert!(
            obj.installed
                .first()
                .expect("entry installation")
                .bitwise_eq(scalar_contract().entry())
        );
        assert_eq!(obj.rho_evaluated.first(), Some(&Array1::from_vec(vec![5.0, 5.0])));
        assert!(
            !obj.orders.is_empty(),
            "the coupled path should evaluate at least one rho waypoint"
        );
        assert!(
            obj.orders
                .iter()
                .all(|order| matches!(order, OuterEvalOrder::Value)),
            "coupled continuation pre-warm must not request outer gradients: {:?}",
            obj.orders
        );
    }

    #[test]
    fn refinement_halves_distance_without_moving_the_accepted_waypoint() {
        let mut path = ContinuationPath::enter(schedules());
        path.s = 0.5;
        path.s_step = 0.25;
        path.refine_step();
        assert_eq!(path.s.to_bits(), 0.5_f64.to_bits());
        assert_eq!(path.s_step.to_bits(), 0.125_f64.to_bits());
    }

    #[test]
    fn arrival_is_a_successful_literal_target_solve() {
        let mut path = ContinuationPath::enter(schedules());
        let mut obj = RecordingObjective::default();
        let initial_beta = Array1::zeros(0);
        assert!(matches!(
            path.step(&mut obj, &initial_beta).expect("entry solve"),
            ContinuationStep::Entered { .. }
        ));
        let arrived = path
            .step(&mut obj, &initial_beta)
            .expect("literal target solve");
        let state = match arrived {
            ContinuationStep::Arrived { state } => state,
            other => panic!("expected exact-target arrival, got {other:?}"),
        };
        assert_eq!(path.s().to_bits(), 0.0_f64.to_bits());
        assert!(state.last_eval.cost.is_finite());
        assert!(
            obj.installed
                .last()
                .expect("target installation")
                .bitwise_eq(scalar_contract().target())
        );
        assert_eq!(obj.rho_evaluated.last(), Some(&Array1::zeros(2)));
    }

    #[test]
    fn failed_waypoint_rolls_back_full_state_before_refined_retry() {
        let mut path = ContinuationPath::enter(schedules());
        let mut obj = RecordingObjective {
            full_state_marker: 7,
            fail_literal_target_once: true,
            ..Default::default()
        };
        let initial_beta = Array1::zeros(0);

        assert!(matches!(
            path.step(&mut obj, &initial_beta).expect("entry solve"),
            ContinuationStep::Entered { .. }
        ));
        assert_eq!(obj.full_state_marker, 8, "entry state must commit");

        assert!(matches!(
            path.step(&mut obj, &initial_beta)
                .expect("non-finite target must refine"),
            ContinuationStep::Refined {
                reason: RefinementReason::WaypointStruggled(_),
                ..
            }
        ));
        assert_eq!(path.s().to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            obj.full_state_marker, 8,
            "failed trial mutation must roll back the complete accepted state"
        );
        assert!(
            obj.installed_current
                .as_ref()
                .expect("restored accepted scalar")
                .bitwise_eq(scalar_contract().entry()),
            "rollback must restore the accepted scalar state too"
        );

        assert!(matches!(
            path.step(&mut obj, &initial_beta).expect("refined midpoint"),
            ContinuationStep::Descended { s, .. } if s.to_bits() == 0.5_f64.to_bits()
        ));
        assert_eq!(
            obj.trial_entry_markers.last(),
            Some(&8),
            "refined retry must start from the restored accepted state"
        );
        assert_eq!(obj.full_state_marker, 9, "refined midpoint must commit");
    }

    #[test]
    fn unrepresentable_descent_is_typed_without_evaluating_again() {
        let mut path = ContinuationPath::enter(schedules());
        let mut obj = RecordingObjective::default();
        let initial_beta = Array1::zeros(0);
        path.step(&mut obj, &initial_beta).expect("entry solve");
        let evals_after_entry = obj.rho_evaluated.len();
        let installs_after_entry = obj.installed.len();
        path.s_step = f64::MIN_POSITIVE;

        let error = path
            .step(&mut obj, &initial_beta)
            .expect_err("rounded-away descent must be a typed refusal");
        assert!(error.to_string().contains("smaller representable waypoint"));
        assert_eq!(obj.rho_evaluated.len(), evals_after_entry);
        assert_eq!(obj.installed.len(), installs_after_entry);
    }
}
