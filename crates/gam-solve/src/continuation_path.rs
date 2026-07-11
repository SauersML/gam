//! Object 1 — the `ContinuationPath`: one object that couples the three
//! annealing schedules that today live separately and uncoupled, so a K≥2 SAE
//! joint fit always arrives via a regime where the inner problem is a
//! contraction — never solved cold.
//!
//! # The three schedules, coupled along one scalar path parameter `s`
//!
//! Three homotopy legs presently advance on their own clocks:
//!
//! 1. **ρ-anneal** — heavy oversmoothing penalty ρ₀ ≫ ρ\* down to the target
//!    ρ\*. Owned by the spine
//!    [`crate::estimate::reml::continuation::fit_with_continuation`]
//!    (the callable ρ-anneal primitive, promoted from a private warm-start
//!    helper). At large ρ the penalized Hessian dominates the likelihood
//!    Hessian, so the inner P-IRLS / arrow-Schur solve is strongly convex —
//!    a contraction.
//! 2. **Assignment temperature τ** — diffuse softmax / IBP relaxation (high τ)
//!    sharpened toward the near-discrete MAP active set (low τ). Owned by
//!    `gam_sae::manifold::GumbelTemperatureSchedule`. High τ makes
//!    the assignment map smooth and far from the combinatorial argmax cliff.
//! 3. **Isometry weight** — loose analytic isometry gauge (small w) ramped to
//!    the tight target weight. Owned by
//!    [`gam_terms::analytic_penalties::ScalarWeightSchedule`] on
//!    [`gam_terms::analytic_penalties::IsometryPenalty`]. A loose gauge
//!    leaves the decoder free to find a good fit before the gauge pins it.
//!
//! [`ContinuationPath`] advances all three **in lockstep** along a single
//! scalar path parameter `s ∈ [1 → 0]`. `s = 1` is the *entry regime*: large
//! ρ, high τ, loose-but-rising isometry — the regime in which the joint inner
//! solve is provably a contraction. `s = 0` is the *real objective*: target ρ\*,
//! sharp τ, tight isometry. The path walks `s` monotonically down, advancing
//! the underlying schedules at each waypoint so the inner problem is never
//! asked to jump from cold to the real objective.
//!
//! # Entry is always the heavy-smoothing regime
//!
//! There is no "solve cold at the real objective" entry. The only entry is
//! `s = 1`, where every leg is at its smoothing extreme and the inner solve is
//! a contraction. A K≥2 SAE joint fit therefore always *arrives* at ρ\* / τ_min
//! / tight-isometry along a continuous descent from a regime where convergence
//! is guaranteed.
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
//! # How this absorbs #969 (warm-invariance) and #976 (hardening)
//!
//! * **#969 — warm-invariance.** A cold entry (no warm β) and a warm entry
//!   (β carried from a previous fit / cache) both enter at `s = 1`, where the
//!   inner solve is a contraction with a *unique* fixed point. A contraction
//!   forgets its initial condition, so both entries are funneled to the SAME
//!   `s = 1` iterate, and from there walk the SAME coupled schedule to the
//!   SAME criterion at `s = 0`. Warm entry only *shortens* the walk (its β is
//!   already near the `s = 1` fixed point); it cannot change the destination.
//!   The path therefore makes "cold and warm reach the same criterion" a
//!   structural property rather than a tolerance the caller must check.
//! * **#976 — hardening.** Two hooks the wiring agent (editing
//!   `rho_optimizer.rs` / `atom_selection.rs`) calls per inner iteration:
//!   a **trust-region cap on the assignment logits**
//!   ([`LogitTrustRegion`]) so a single Newton step can never fling the
//!   relaxed assignment across the argmax cliff; and an **active-mass-floor
//!   breach signal** ([`ActiveMassFloor`] / [`MassFloorBreach`]) that, when the
//!   per-row active mass collapses toward the uniform saddle, triggers a
//!   *re-seed from the scaffold* (the pristine seeded geometry) — recorded in
//!   the [`ReseedLedger`], **never fatal**. A breach is a ledger entry and a
//!   smaller next attempted distance, not an error return.
//!
//! This module owns the coupling object and the hook *interfaces / return
//! types*. The wiring agent implements the call sites against these types.

use ndarray::{Array1, ArrayView2};

use crate::estimate::reml::continuation::{ContinuationState, eval_step};
use crate::inner_status::InnerFailure;
use crate::rho_optimizer::{OuterEvalOrder, OuterObjective};

/// The endpoints of one coupled annealing leg, in path-parameter terms.
/// `at_entry` is the value at `s = 1` (heavy-smoothing regime); `at_target`
/// is the value at `s = 0` (real objective). Interpolation is in the leg's
/// own natural geometry (log-space for ρ and τ, linear-in-weight for the
/// isometry gauge, matching each schedule's `current_*` law).
#[derive(Debug, Clone, Copy)]
pub struct LegEndpoints {
    /// Value at `s = 1`: the smoothing-extreme entry regime.
    pub at_entry: f64,
    /// Value at `s = 0`: the real-objective target.
    pub at_target: f64,
}

impl LegEndpoints {
    /// Construct from an entry value and a target value.
    #[must_use]
    pub fn new(at_entry: f64, at_target: f64) -> Self {
        Self {
            at_entry,
            at_target,
        }
    }

    /// Linear interpolation in the leg's natural coordinate at path parameter
    /// `s ∈ [0, 1]`: `s = 1 → at_entry`, `s = 0 → at_target`. The caller passes
    /// values already in the leg's natural geometry (e.g. log τ, log λ), so a
    /// plain convex blend is the right law and matches the schedules'
    /// `current_*` interpolation.
    #[must_use]
    pub fn at(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, 1.0);
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

/// The coupled schedule state that [`ContinuationPath`] owns. Each leg is the
/// concrete schedule object the rest of the codebase already advances; the
/// path holds them so they can only ever move together.
#[derive(Debug, Clone)]
pub struct CoupledSchedules {
    /// ρ-anneal endpoints, **per-component** in ρ-space (one entry per
    /// smoothing parameter). The entry vector is the oversmoothed ρ₀; the
    /// target is ρ\*. The actual descent is executed by the ρ-anneal spine
    /// ([`fit_with_continuation`]); these endpoints fix where `s` places the
    /// spine's `target` waypoint along the coupled walk.
    pub rho_entry: Array1<f64>,
    /// ρ\* — the real-objective smoothing vector at `s = 0`.
    pub rho_target: Array1<f64>,
    /// Typed scalar entry/target state supplied by the objective itself.
    pub scalars: ContinuationScalarContract,
}

impl CoupledSchedules {
    /// The coupled lockstep target value of every scalar leg at path parameter
    /// `s`. ρ is a vector and rides the spine, so it is not returned here; the
    /// two scalar legs (τ, isometry weight) are.
    #[must_use]
    pub fn scalar_targets_at(&self, s: f64) -> ContinuationScalarState {
        self.scalars.at(s)
    }

    /// The ρ target the spine should anneal toward at path parameter `s`:
    /// a convex blend (per component) of the oversmoothed entry ρ₀ and ρ\*.
    /// At `s = 1` this is ρ₀ itself (so the spine's own oversmoothing offset
    /// stacks the path into the deepest contraction); at `s = 0` it is ρ\*.
    #[must_use]
    pub fn rho_target_at(&self, s: f64) -> Array1<f64> {
        assert_eq!(
            self.rho_entry.len(),
            self.rho_target.len(),
            "ContinuationPath: ρ entry/target dimension mismatch"
        );
        let s = s.clamp(0.0, 1.0);
        let mut out = self.rho_target.clone();
        for i in 0..out.len() {
            out[i] = self.rho_target[i] + s * (self.rho_entry[i] - self.rho_target[i]);
        }
        out
    }
}

/// The lockstep target values of the two scalar legs at a given `s`. Handed to
/// the wiring agent so it can install τ on the SAE term and the isometry weight
/// on the gauge penalty before the spine pass at this waypoint.
// ─────────────────────────────────────────────────────────────────────────
//  Hardening hook interfaces (#976). Defined here; implemented at the call
//  sites by the wiring agent (rho_optimizer.rs / atom_selection.rs).
// ─────────────────────────────────────────────────────────────────────────

/// Per-iteration trust-region cap on the assignment logits.
///
/// The wiring agent calls [`LogitTrustRegion::cap_step`] on each candidate
/// Newton step in assignment-logit space before it is applied, so a single
/// step can never fling the relaxed assignment across the argmax cliff (the
/// discontinuity the τ anneal exists to avoid). The cap is an ∞-norm radius on
/// the logit increment, tied to the current τ: hotter τ (diffuse) tolerates a
/// larger logit move; colder τ (sharp) clamps tighter, because near the cliff a
/// small logit change is a large assignment change.
#[derive(Debug, Clone, Copy)]
pub struct LogitTrustRegion {
    /// ∞-norm radius on the logit increment at the current waypoint.
    pub radius: f64,
}

/// Outcome of applying the logit trust-region cap to a proposed step. The
/// wiring agent applies the returned (possibly shrunk) step. There is no
/// "reject" outcome — the cap only *scales* the step.
#[derive(Debug, Clone, Copy)]
pub enum LogitStepCap {
    /// The proposed step was within the radius; apply it unchanged.
    Within,
    /// The proposed step exceeded the radius; scale it by `scale ∈ (0, 1)` so
    /// its ∞-norm equals `radius`, then apply.
    Scaled { scale: f64 },
}

impl LogitTrustRegion {
    /// Build the per-waypoint logit trust region from the current τ. Hotter τ
    /// ⇒ larger radius (the assignment map is gentle); colder τ ⇒ tighter
    /// radius (near the argmax cliff). The radius is `τ · LOGIT_TR_TAU_GAIN`
    /// clamped to `[LOGIT_TR_MIN, LOGIT_TR_MAX]`.
    #[must_use]
    pub fn for_tau(tau: f64) -> Self {
        const LOGIT_TR_TAU_GAIN: f64 = 4.0;
        const LOGIT_TR_MIN: f64 = 1.0e-2;
        const LOGIT_TR_MAX: f64 = 8.0;
        let radius = (tau * LOGIT_TR_TAU_GAIN).clamp(LOGIT_TR_MIN, LOGIT_TR_MAX);
        Self { radius }
    }

    /// Decide how to cap a proposed logit increment given its ∞-norm. The
    /// wiring agent passes the step's ∞-norm; this returns whether to apply it
    /// unchanged or scaled to the radius. Never rejects.
    #[must_use]
    pub fn cap_step(&self, step_inf_norm: f64) -> LogitStepCap {
        if !step_inf_norm.is_finite() || step_inf_norm <= self.radius || step_inf_norm == 0.0 {
            LogitStepCap::Within
        } else {
            LogitStepCap::Scaled {
                scale: self.radius / step_inf_norm,
            }
        }
    }
}

/// Active-mass-floor watcher (#976). The wiring agent calls
/// [`ActiveMassFloor::check`] with the per-row mean active assignment mass each
/// inner iteration. When the mass collapses toward the uniform saddle (below
/// the floor), `check` returns a [`MassFloorBreach`] the caller records in the
/// [`ReseedLedger`] and acts on by re-seeding from the scaffold. A breach is
/// **never fatal** — there is no error return.
#[derive(Debug, Clone, Copy)]
pub struct ActiveMassFloor {
    /// Mean active mass below which the assignment is judged to have collapsed
    /// toward the near-uniform saddle and a scaffold re-seed is triggered.
    pub floor: f64,
}

impl ActiveMassFloor {
    /// Default floor: the **failure boundary**, not the healthy operating
    /// point. The SAE routing-collapse quality oracle plants a healthy
    /// codes'-units active mass of ~`0.2` and asserts recovery of at least
    /// half of it; the floor therefore sits at `0.5 × 0.2 = 0.1` — breach
    /// exactly when the fit enters the region the quality assertion already
    /// calls collapsed. Placing the floor *at* the healthy operating mass
    /// (the previous `0.2`) made a healthy converging IBP-MAP fit oscillate
    /// across the floor, and every spurious breach re-seeds from the scaffold
    /// (`obj.reset()`) — re-seed thrash that discards converged routing mass and
    /// pins the fit near the cold seed: itself a collapse mechanism. Genuine saddle collapse
    /// (~`0.03` observed mass) is still far below this floor.
    pub const DEFAULT_FLOOR: f64 = 0.1;

    #[must_use]
    pub fn default_floor() -> Self {
        Self {
            floor: Self::DEFAULT_FLOOR,
        }
    }

    /// Check the observed mean active mass against the floor. Returns
    /// `Some(MassFloorBreach)` when collapsed (caller re-seeds from scaffold +
    /// logs to the ledger), `None` when healthy. Never an error.
    #[must_use]
    pub fn check(&self, mean_active_mass: f64) -> Option<MassFloorBreach> {
        if mean_active_mass.is_finite() && mean_active_mass >= self.floor {
            None
        } else {
            Some(MassFloorBreach {
                observed_mean_mass: mean_active_mass,
                floor: self.floor,
            })
        }
    }
}

/// A recorded active-mass-floor breach. Carries the observed mass and the floor
/// it fell below. The wiring agent's response is a re-seed-from-scaffold plus a
/// smaller next attempted path distance: the event is appended to the
/// [`ReseedLedger`] without moving the last accepted waypoint.
#[derive(Debug, Clone, Copy)]
pub struct MassFloorBreach {
    pub observed_mean_mass: f64,
    pub floor: f64,
}

/// Append-only ledger of scaffold re-seeds triggered by active-mass-floor
/// breaches. Non-fatal by construction: the ledger only *records*; it never
/// holds a terminal/abort state. The wiring agent threads one ledger through
/// the joint fit and queries [`ReseedLedger::reseed_count`] for diagnostics.
#[derive(Debug, Clone, Default)]
pub struct ReseedLedger {
    entries: Vec<ReseedEvent>,
}

/// One scaffold re-seed event: the path parameter `s` at which the breach was
/// observed and the breach payload. Lets diagnostics see whether re-seeds
/// cluster at sharp-τ waypoints (the expected near-cliff regime).
#[derive(Debug, Clone, Copy)]
pub struct ReseedEvent {
    pub s: f64,
    pub breach: MassFloorBreach,
}

impl ReseedLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record a scaffold re-seed triggered at path parameter `s`. Returns
    /// nothing fatal — recording a breach is routine homotopy bookkeeping.
    pub fn record(&mut self, s: f64, breach: MassFloorBreach) {
        self.entries.push(ReseedEvent { s, breach });
    }

    #[must_use]
    pub fn reseed_count(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn events(&self) -> &[ReseedEvent] {
        &self.entries
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Coarse diagnostic view of the last accepted scalar regime.
//
//  The seed cascade in `rho_optimizer.rs` observes the path through a coarser
//  lens than the per-waypoint `s`: it only needs to know which scalar regime
//  the path most recently solved. `PathRegime` is that coarse view — a band of
//  the path parameter `s`.
// ─────────────────────────────────────────────────────────────────────────

/// Coarse band of the last accepted path parameter. Banded from
/// `s ∈ [0, 1]`: larger `s` means deeper in the contraction basin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathRegime {
    /// `s` near the real objective (`s ≤ 1/4`): the path is at or close to ρ*,
    /// the lightest smoothing the path ever sits at. The nominal entry band
    /// only on a fully-descended path.
    Target,
    /// Mid-path (`1/4 < s ≤ 3/4`): partially annealed, intermediate smoothing.
    Annealing,
    /// Heavy-smoothing entry band (`s > 3/4`): the deepest contraction regime,
    /// where the joint inner solve is provably a contraction. The band a fresh
    /// `heavy_entry` starts in.
    Heavy,
}

impl PathRegime {
    /// Band the live path parameter `s` into the coarse regime the seed cascade
    /// reports. Monotone in `s`: larger `s` ⇒ heavier regime.
    #[must_use]
    fn from_s(s: f64) -> Self {
        let s = s.clamp(0.0, 1.0);
        if s > 0.75 {
            PathRegime::Heavy
        } else if s > 0.25 {
            PathRegime::Annealing
        } else {
            PathRegime::Target
        }
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
    /// `s = 1`. Carries the accepted spine state that warms the first descent.
    Entered { state: ContinuationState },
    /// `s` was lowered toward `0` and the inner solve at the new waypoint
    /// succeeded. Carries the accepted spine state and the new `s`.
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
    /// The active-mass floor was breached at this waypoint; a scaffold re-seed
    /// was recorded and the next attempted distance is refined.
    MassFloorBreached(MassFloorBreach),
}

// ─────────────────────────────────────────────────────────────────────────
//  The ContinuationPath object.
// ─────────────────────────────────────────────────────────────────────────

/// Object 1 — the coupled continuation path. Owns the three schedules and the
/// scalar path parameter `s`, and drives the K≥2 SAE joint fit down the coupled
/// homotopy. Entry is always `s = 1` (heavy-smoothing contraction regime), and
/// only a solved literal target can produce arrival.
///
/// The outer driver advances the path one waypoint at a time:
/// `let step = path.step(obj, &warm_beta)?;` and, per [`ContinuationStep`],
/// installs the next waypoint's [`ContinuationScalarState`] (τ on the SAE term,
/// one weight per isometry penalty) and applies the [`LogitTrustRegion`] /
/// [`ActiveMassFloor`] hooks inside the inner solve.
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
    /// Logit trust region and active-mass floor recomputed per waypoint from
    /// the current τ.
    logit_tr: LogitTrustRegion,
    mass_floor: ActiveMassFloor,
    /// Path-owned re-seed ledger for breaches reported through the bare,
    /// no-ledger hardening hook ([`ContinuationPath::note_active_mass_breach`]).
    /// The richer ledger-threading API ([`ContinuationPath::note_mass_breach`])
    /// is unchanged; this internal ledger backs the inner-loop call site that
    /// does not thread its own ledger. Append-only, never fatal.
    reseed_ledger: ReseedLedger,
    /// The most recent converged waypoint state. `None` until the first leg
    /// converges (that leg runs the full cold spine); every later waypoint is
    /// a WARM leg from here — the structural fix for the per-waypoint
    /// cold-spine cost blowup. A failed attempted waypoint never overwrites it.
    warm: Option<ContinuationState>,
}

impl ContinuationPath {
    /// Build the coupled path. `s` is initialized to `1.0` — the heavy-smoothing
    /// entry regime where the joint inner solve is a contraction. The path can
    /// **only** be entered here; there is no constructor that starts cold at the
    /// real objective. This is what makes warm-invariance (#969) structural: any
    /// entry, warm or cold, funnels through the `s = 1` contraction fixed point.
    #[must_use]
    pub fn enter(schedules: CoupledSchedules) -> Self {
        let entry_targets = schedules.scalar_targets_at(1.0);
        let logit_tr = LogitTrustRegion::for_tau(entry_targets.assignment_temperature);
        Self {
            schedules,
            s: 1.0,
            s_step: 1.0,
            logit_tr,
            mass_floor: ActiveMassFloor::default_floor(),
            reseed_ledger: ReseedLedger::new(),
            warm: None,
        }
    }

    /// Heavy-smoothing entry coupled to a CONCRETE ρ target and legal box. The
    /// seed cascade rebuilds the path per-seed with this once it knows the
    /// objective's real ρ dimension (the no-argument [`ContinuationPath::heavy_entry`]
    /// is a dimension-1 placeholder used only before the seed is in hand). The
    /// ρ leg rides the spine from the spine's own oversmoothed ρ₀ down to
    /// `rho_target` (the real objective ρ\*); `bounds_upper` is the legal ρ box.
    /// The scalar legs use the typed entry/target contract supplied by the
    /// objective; no private default may replace its literal target. Enters at
    /// `s = 1`. `rho_target` and `bounds_upper` must share length.
    #[must_use]
    pub fn heavy_entry_for_rho(
        rho_target: Array1<f64>,
        bounds_upper: Array1<f64>,
        scalars: ContinuationScalarContract,
    ) -> Self {
        assert_eq!(
            rho_target.len(),
            bounds_upper.len(),
            "ContinuationPath::heavy_entry_for_rho: ρ target/bounds dim mismatch"
        );
        // The legal upper box is the literal heavy-penalty endpoint. This uses
        // objective-owned geometry rather than a private log-offset heuristic,
        // and makes rho move under the same `s` as temperature and isometry.
        let schedules = couple_schedules(
            bounds_upper.clone(),
            rho_target,
            scalars,
        );
        Self::enter(schedules)
    }

    /// Coarse diagnostic band of the last successfully solved waypoint. A fresh
    /// path is in [`PathRegime::Heavy`].
    #[must_use]
    pub fn current_regime(&self) -> PathRegime {
        PathRegime::from_s(self.s)
    }

    /// The base radius the per-iteration assignment-logit trust region is built
    /// from (`rho_optimizer.rs` / `atom_selection.rs` hardening hook). This is
    /// the ∞-norm logit step radius at the current waypoint; larger accepted
    /// `s` values cool τ and hand back a tighter radius.
    #[must_use]
    pub fn logit_step_radius(&self) -> f64 {
        self.logit_tr.radius
    }

    /// Bare active-mass-floor breach hook for the inner-loop call site that does
    /// not thread its own [`ReseedLedger`]. Records the breach in the
    /// path-owned ledger at the current `s` and refines the next distance —
    /// the same non-fatal response as [`ContinuationPath::note_mass_breach`],
    /// without requiring the caller to carry a ledger. Returns the unchanged
    /// accepted [`PathRegime`] so the call site can report it.
    pub fn note_active_mass_breach(&mut self) -> PathRegime {
        let breach = MassFloorBreach {
            observed_mean_mass: self.mass_floor.floor,
            floor: self.mass_floor.floor,
        };
        // Single source of truth for the breach response: route through
        // `note_mass_breach` so the record-then-refine logic is not
        // duplicated. The bare hook differs only in *which* ledger it threads —
        // the path-owned one — so we lend that ledger to the shared driver and
        // hand it back afterwards.
        let mut owned = std::mem::take(&mut self.reseed_ledger);
        let step = self.note_mass_breach(breach, &mut owned);
        self.reseed_ledger = owned;
        // The shared driver retains the last accepted waypoint and refines the
        // attempted distance. Match the
        // step exhaustively so the outcome is observed (no silent discard) and
        // the "every breach is progress" invariant is documented at the use
        // site: every arm resolves to the accepted live regime.
        match step {
            ContinuationStep::Refined { .. }
            | ContinuationStep::Entered { .. }
            | ContinuationStep::Descended { .. }
            | ContinuationStep::Arrived { .. } => self.current_regime(),
        }
    }

    /// Number of scaffold re-seeds recorded through the bare
    /// [`ContinuationPath::note_active_mass_breach`] hook (diagnostics).
    #[must_use]
    pub fn reseed_count(&self) -> usize {
        self.reseed_ledger.reseed_count()
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

    /// The ρ target the spine should anneal toward at the current `s`.
    #[must_use]
    pub fn current_rho_target(&self) -> Array1<f64> {
        self.schedules.rho_target_at(self.s)
    }

    /// The per-waypoint logit trust region (from the current τ). The wiring
    /// agent caps each assignment-logit Newton step with this.
    #[must_use]
    pub fn logit_trust_region(&self) -> LogitTrustRegion {
        self.logit_tr
    }

    /// The active-mass floor for this path. The wiring agent calls
    /// [`ActiveMassFloor::check`] with the observed mean active mass each inner
    /// iteration and, on breach, records a scaffold re-seed in the ledger and
    /// reports it back via [`ContinuationPath::note_mass_breach`].
    #[must_use]
    pub fn active_mass_floor(&self) -> ActiveMassFloor {
        self.mass_floor
    }

    /// Record an active-mass-floor breach into the ledger and refine the next
    /// attempted distance. Returns the [`ContinuationStep::Refined`] the wiring
    /// agent should act on. This is the hook the wiring agent calls when
    /// [`ActiveMassFloor::check`] returns `Some` from inside the inner solve.
    pub(crate) fn note_mass_breach(
        &mut self,
        breach: MassFloorBreach,
        ledger: &mut ReseedLedger,
    ) -> ContinuationStep {
        ledger.record(self.s, breach);
        self.refine_step();
        ContinuationStep::Refined {
            s: self.s,
            reason: RefinementReason::MassFloorBreached(breach),
        }
    }

    /// Refine the next descent after a failed attempted waypoint. `s` remains
    /// the last successfully solved waypoint; only the attempted distance is
    /// halved, so the accepted path is monotone and never fabricates progress.
    fn refine_step(&mut self) {
        self.s_step *= 0.5;
        self.logit_tr = LogitTrustRegion::for_tau(
            self.schedules
                .scalar_targets_at(self.s)
                .assignment_temperature,
        );
    }

    /// Take one waypoint step down the coupled homotopy.
    ///
    /// 1. Lower `s` by the current step toward `0`.
    /// 2. Advance the τ and isometry schedules to the new waypoint (lockstep).
    /// 3. Run the ρ-anneal **spine** ([`fit_with_continuation`]) toward the new
    ///    `s`'s ρ target, with the inner β carried warm.
    /// 4. On spine success: [`ContinuationStep::Descended`] (or
    ///    [`ContinuationStep::Arrived`] if `s` reached `0`).
    /// 5. On spine struggle: retain the last successful waypoint and halve the
    ///    attempted distance. If no strictly smaller representable waypoint
    ///    remains, return a typed non-convergence error.
    ///
    /// `obj` is the SAE joint outer objective (`SaeManifoldOuterObjective`,
    /// which is an [`OuterObjective`]). `initial_beta` warms the inner solve;
    /// pass the empty array for cold entry (warm-invariance, #969, guarantees
    /// the same destination either way).
    pub(crate) fn step(
        &mut self,
        obj: &mut dyn OuterObjective,
        initial_beta: &Array1<f64>,
    ) -> Result<ContinuationStep, gam_problem::EstimationError> {
        // The cold leg solves the literal heavy entry at s=1 before any
        // descent. Every later leg lowers s by one path step. This makes the
        // contraction entry an evaluated waypoint rather than an unevaluated
        // endpoint that the old implementation skipped on its first call.
        let entering = self.warm.is_none();
        let s_next = if entering {
            1.0
        } else {
            (self.s - self.s_step).max(0.0)
        };

        // Install the objective-owned scalar state before evaluating rho. This
        // is the actual coupling seam: mutating private schedule copies would
        // leave the evaluated objective unchanged.
        let scalar_state = self.schedules.scalar_targets_at(s_next);
        obj.install_reactive_domain_scalar_state(&scalar_state)?;
        self.logit_tr = LogitTrustRegion::for_tau(scalar_state.assignment_temperature);

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
                self.logit_tr = LogitTrustRegion::for_tau(
                    self.schedules
                        .scalar_targets_at(self.s)
                        .assignment_temperature,
                );
                if entering {
                    ContinuationStep::Entered { state }
                } else if self.s <= 0.0 {
                    ContinuationStep::Arrived { state }
                } else {
                    ContinuationStep::Descended { s: self.s, state }
                }
            }
            Err(failure) => {
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
                             waypoint below s={:.17e} after spine non-convergence: {}",
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
pub fn couple_schedules(
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

/// View helper: the wiring agent passes the SAE assignment matrix (rows ×
/// atoms) to compute the mean active mass for the [`ActiveMassFloor`] check.
/// Defined here so the floor's input convention has one owner.
#[must_use]
pub fn mean_active_mass(assignments: ArrayView2<'_, f64>) -> f64 {
    let n = assignments.nrows();
    if n == 0 {
        return 0.0;
    }
    // Per-row active mass = max assignment weight in the row (how concentrated
    // the routing is); the saddle is uniform (~1/K), a routed fit is ~1.
    let mut acc = 0.0;
    for row in assignments.rows() {
        let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if row_max.is_finite() {
            acc += row_max;
        }
    }
    acc / n as f64
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
    fn entry_is_the_heavy_smoothing_regime() {
        let path = ContinuationPath::enter(schedules());
        assert_eq!(
            path.s(),
            1.0,
            "entry must be s = 1 (heavy-smoothing regime)"
        );
        let targets = path.current_scalar_targets();
        assert_eq!(targets.assignment_temperature.to_bits(), 2.0_f64.to_bits());
        assert_eq!(targets.isometry_weights, vec![0.01, 0.02]);
        // ρ target at s = 1 is the oversmoothed entry ρ₀.
        let rho = path.current_rho_target();
        assert!((rho[0] - 5.0).abs() < 1e-12 && (rho[1] - 5.0).abs() < 1e-12);
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
    fn logit_trust_region_tightens_as_tau_cools() {
        let hot = LogitTrustRegion::for_tau(2.0);
        let cold = LogitTrustRegion::for_tau(0.05);
        assert!(
            cold.radius < hot.radius,
            "colder τ must give a tighter logit trust region"
        );
        // A step within the radius is applied unchanged.
        assert!(matches!(
            cold.cap_step(cold.radius * 0.5),
            LogitStepCap::Within
        ));
        // A step past the radius is scaled down, never rejected.
        match cold.cap_step(cold.radius * 4.0) {
            LogitStepCap::Scaled { scale } => {
                assert!(scale > 0.0 && scale < 1.0);
                assert!((scale - 0.25).abs() < 1e-12);
            }
            LogitStepCap::Within => panic!("expected the over-radius step to be scaled"),
        }
    }

    #[test]
    fn active_mass_floor_breach_is_recorded_never_fatal() {
        let floor = ActiveMassFloor::default_floor();
        assert!(floor.check(0.9).is_none(), "healthy routing → no breach");
        let breach = floor.check(0.05).expect("collapsed routing → breach");
        let mut ledger = ReseedLedger::new();
        ledger.record(0.3, breach);
        assert_eq!(ledger.reseed_count(), 1);
        assert!((ledger.events()[0].s - 0.3).abs() < 1e-12);
    }

    #[test]
    fn note_mass_breach_refines_distance_without_moving_accepted_waypoint() {
        let mut path = ContinuationPath::enter(schedules());
        path.s = 0.5;
        path.s_step = 0.25;
        let accepted_before = path.s;
        let distance_before = path.s_step;
        let mut ledger = ReseedLedger::new();
        let breach = MassFloorBreach {
            observed_mean_mass: 0.05,
            floor: ActiveMassFloor::DEFAULT_FLOOR,
        };
        let step = path.note_mass_breach(breach, &mut ledger);
        assert!(matches!(
            step,
            ContinuationStep::Refined {
                reason: RefinementReason::MassFloorBreached(_),
                ..
            }
        ));
        assert_eq!(
            path.s().to_bits(),
            accepted_before.to_bits(),
            "a failed attempt must retain the last accepted waypoint"
        );
        assert_eq!(
            path.s_step.to_bits(),
            (distance_before * 0.5).to_bits(),
            "a mass breach must halve only the next attempted distance"
        );
        assert_eq!(ledger.reseed_count(), 1);
    }

    #[test]
    fn continuation_step_has_no_reject_arm() {
        // Compile-time + exhaustiveness witness: successful target arrival is
        // distinct from refinement and carries its solved state. Typed
        // non-convergence is returned outside this enum.
        fn is_progress(step: &ContinuationStep) -> bool {
            match step {
                ContinuationStep::Entered { .. }
                | ContinuationStep::Descended { .. }
                | ContinuationStep::Arrived { .. }
                | ContinuationStep::Refined { .. } => true,
            }
        }
        let breach = MassFloorBreach {
            observed_mean_mass: 0.0,
            floor: 0.2,
        };
        assert!(is_progress(&ContinuationStep::Refined {
            s: 1.0,
            reason: RefinementReason::MassFloorBreached(breach),
        }));
    }

    #[test]
    fn mean_active_mass_distinguishes_routed_from_saddle() {
        use ndarray::array;
        // Two rows, K=2. Routed: one weight near 1. Saddle: uniform 0.5.
        let routed = array![[0.95, 0.05], [0.9, 0.1]];
        let saddle = array![[0.5, 0.5], [0.5, 0.5]];
        assert!(mean_active_mass(routed.view()) > 0.85);
        assert!((mean_active_mass(saddle.view()) - 0.5).abs() < 1e-12);
        assert!(
            ActiveMassFloor::default_floor()
                .check(mean_active_mass(saddle.view()))
                .is_none(),
            "uniform 0.5 is above the floor — saddle detection is about \
             collapse below the failure boundary (0.5× the planted healthy \
             mass), not the healthy operating point"
        );
    }

    #[test]
    fn heavy_entry_starts_in_the_heavy_regime() {
        let path = ContinuationPath::heavy_entry_for_rho(
            Array1::zeros(1),
            Array1::from_elem(1, 10.0),
            scalar_contract(),
        );
        assert_eq!(path.s(), 1.0, "heavy_entry must enter at s = 1");
        assert_eq!(
            path.current_regime(),
            PathRegime::Heavy,
            "a fresh heavy_entry is in the heavy-smoothing regime"
        );
        assert!(
            path.logit_step_radius().is_finite() && path.logit_step_radius() > 0.0,
            "logit step radius must be finite and positive at entry"
        );
    }

    #[test]
    fn bare_active_mass_breach_records_and_refines() {
        let mut path = ContinuationPath::enter(schedules());
        path.s = 0.4;
        path.s_step = 0.2;
        assert_eq!(path.reseed_count(), 0);
        let accepted_before = path.s;
        let distance_before = path.s_step;
        let regime = path.note_active_mass_breach();
        assert_eq!(
            path.reseed_count(),
            1,
            "breach must be recorded in the path ledger"
        );
        assert_eq!(
            path.s.to_bits(),
            accepted_before.to_bits(),
            "breach must retain the accepted waypoint"
        );
        assert_eq!(
            path.s_step.to_bits(),
            (distance_before * 0.5).to_bits(),
            "breach must refine the next attempted distance"
        );
        assert_eq!(regime, path.current_regime());
    }

    #[test]
    fn path_regime_bands_are_monotone_in_s() {
        assert_eq!(PathRegime::from_s(0.0), PathRegime::Target);
        assert_eq!(PathRegime::from_s(0.2), PathRegime::Target);
        assert_eq!(PathRegime::from_s(0.5), PathRegime::Annealing);
        assert_eq!(PathRegime::from_s(0.9), PathRegime::Heavy);
        assert_eq!(PathRegime::from_s(1.0), PathRegime::Heavy);
    }

    #[derive(Default)]
    struct RecordingObjective {
        orders: Vec<OuterEvalOrder>,
        seed_count: usize,
        installed: Vec<ContinuationScalarState>,
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
            let mut eval = self.eval(rho)?;
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
            .expect("the entry contraction must solve");
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
    }
}
