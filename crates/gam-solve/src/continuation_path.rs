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
//! # The tail is a homotopy FLOOR, not a gate
//!
//! If a downward step's inner solve struggles, the path does **not** reject:
//! it re-enters a heavier regime (raises `s` back toward 1 by a back-off
//! fraction) and re-descends with a finer step. This is a *floor* the iterate
//! bounces off, never a trapdoor it falls through. The structural guarantee is
//! encoded in the type: [`ContinuationStep`] — the per-step outcome enum — has
//! **no `Reject` / `Failed` / `NoUsableSeed` arm**. The worst a step can report
//! is [`ContinuationStep::Reentered`] (bounced off the floor, re-descending),
//! which is progress toward, not abandonment of, the fit. There is no value of
//! the outcome type that means "give up".
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
//!   regime re-entry, not an error return.
//!
//! This module owns the coupling object and the hook *interfaces / return
//! types*. The wiring agent implements the call sites against these types.

use ndarray::{Array1, ArrayView2};

use crate::estimate::reml::continuation::{
    ContinuationFailure, ContinuationState, PATH_BUDGET, continue_path_from, fit_with_continuation,
};
use crate::rho_optimizer::{OuterEvalOrder, OuterObjective};
use gam_terms::analytic_penalties::ScalarWeightSchedule;
use gam_problem::schedule::{GumbelTemperatureSchedule, ScheduleKind};

/// Number of lockstep waypoints the path visits as `s` walks `1 → 0`. Each
/// waypoint advances every leg one notch and runs one ρ-anneal spine pass.
/// Chosen so the geometric schedules have room to descend an order of
/// magnitude or two per leg without a single step that crosses the contraction
/// boundary; the homotopy floor absorbs any waypoint that still over-reaches.
pub const CONTINUATION_WAYPOINTS: usize = 8;

/// Back-off fraction applied to `s` when a waypoint's inner solve struggles:
/// `s ← min(1, s + REENTRY_BACKOFF)`. Re-entering a heavier regime and
/// re-descending with a halved step is the *floor* behavior — there is no
/// rejection alternative.
///
/// Exactly **one waypoint** of the lockstep grid (`1/CONTINUATION_WAYPOINTS`):
/// a bounce off the homotopy floor re-enters the *previous* waypoint's heavier
/// regime, the lightest regime already proven solvable on this walk. Combined
/// with the halved re-descent step, a one-notch bounce costs ~2 walk legs
/// (the re-entry plus one finer re-descent). The previous two-notch back-off
/// (0.25) cost ~4 legs per bounce, which starved the bounded walk budget under
/// repeated mass-floor bounces and left the K≥2 joint fit stranded mid-path —
/// handed to the solver half-annealed, the routing-collapse signature.
pub const REENTRY_BACKOFF: f64 = 1.0 / CONTINUATION_WAYPOINTS as f64;

/// Total leg budget for one coupled walk (`rho_optimizer.rs` drives
/// [`ContinuationPath::step`] at most this many times per seed). Two legs per
/// waypoint: a clean walk uses `CONTINUATION_WAYPOINTS` descents, and each
/// homotopy-floor bounce costs ~2 extra legs at the one-notch
/// [`REENTRY_BACKOFF`] (the re-entry leg plus one finer re-descent leg), so a
/// 2× budget tolerates ~`CONTINUATION_WAYPOINTS/2` bounces before the walk is
/// cut off — enough for the expected near-cliff re-entries while keeping the
/// total inner-solve count bounded. The previous 1.5× budget tolerated only
/// ~1 two-notch bounce, so any mass-floor bounce ended the walk un-arrived.
pub const CONTINUATION_WALK_BUDGET: usize = 2 * CONTINUATION_WAYPOINTS;

/// Eval budget for one **warm** waypoint leg. A warm leg starts at the
/// previous waypoint's converged state and walks one waypoint of ρ, so it
/// needs a handful of evals, not the full cold spine: the coupled path's
/// waypoints ARE the anneal. (Re-running the whole ρ₀→target spine per
/// waypoint multiplies the walk's cost by the spine budget — the K=2 existence
/// fixture burned 7 CPU-hours exactly that way before warm legs existed.)
pub const WARM_LEG_EVAL_BUDGET: usize = 8;

/// Hard ceiling on *budgeted* spine evals across one coupled walk — the #968
/// termination guarantee made structural. A clean walk budgets
/// `PATH_BUDGET + (CONTINUATION_WAYPOINTS − 1) · WARM_LEG_EVAL_BUDGET` (one
/// cold entry spine, then warm legs); the ceiling leaves ~3× that for
/// homotopy-floor bounces. At the ceiling the path **arrives with its best
/// converged state** instead of spending another leg: a walk cannot spin.
pub const WALK_EVAL_CEILING: usize =
    3 * (PATH_BUDGET + CONTINUATION_WAYPOINTS * WARM_LEG_EVAL_BUDGET);

/// Floor on the per-waypoint descent step in `s`. Below this the path is
/// taking near-zero steps; it does not give up — it pins `s` at its current
/// (heavier) regime and keeps re-descending from there. The floor is a
/// *behavior*, never an exit.
pub const S_STEP_FLOOR: f64 = 1.0 / 256.0;

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
    /// Legal upper bound on ρ (the spine clamps ρ₀ into this box).
    pub rho_bounds_upper: Array1<f64>,
    /// Assignment-temperature schedule (τ leg). Consumed, not re-implemented:
    /// the path reads `tau_start` / `tau_min` as its τ endpoints and advances
    /// the schedule in lockstep with `s`.
    pub temperature: GumbelTemperatureSchedule,
    /// Isometry-weight schedule (gauge leg). Consumed: `w_start` / `w_end` are
    /// the isometry endpoints; advanced in lockstep with `s`.
    pub isometry: ScalarWeightSchedule,
}

impl CoupledSchedules {
    /// τ endpoints as `LegEndpoints` in the schedule's natural coordinate.
    /// `s = 1` → `tau_start` (diffuse), `s = 0` → `tau_min` (sharp).
    #[must_use]
    pub fn temperature_endpoints(&self) -> LegEndpoints {
        LegEndpoints::new(self.temperature.tau_start, self.temperature.tau_min)
    }

    /// Isometry-weight endpoints. `s = 1` → `w_start` (loose), `s = 0` →
    /// `w_end` (tight).
    #[must_use]
    pub fn isometry_endpoints(&self) -> LegEndpoints {
        LegEndpoints::new(self.isometry.w_start, self.isometry.w_end)
    }

    /// The coupled lockstep target value of every scalar leg at path parameter
    /// `s`. ρ is a vector and rides the spine, so it is not returned here; the
    /// two scalar legs (τ, isometry weight) are.
    #[must_use]
    pub fn scalar_targets_at(&self, s: f64) -> ScalarLegTargets {
        ScalarLegTargets {
            tau: self.temperature_endpoints().at(s),
            isometry_weight: self.isometry_endpoints().at(s),
        }
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
#[derive(Debug, Clone, Copy)]
pub struct ScalarLegTargets {
    /// Assignment temperature τ at this waypoint.
    pub tau: f64,
    /// Isometry gauge weight at this waypoint.
    pub isometry_weight: f64,
}

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
    /// (`obj.reset()`) and re-enters a heavier regime — re-seed thrash that
    /// discards converged routing mass each bounce and pins the fit near the
    /// cold seed: itself a collapse mechanism. Genuine saddle collapse
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
/// it fell below. The wiring agent's response is a re-seed-from-scaffold, not a
/// failure: this is appended to the [`ReseedLedger`] and the path re-enters a
/// heavier regime.
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
//  Regime-escalation view of re-entry (#969 seed-cascade demotion).
//
//  The seed cascade in `rho_optimizer.rs` observes the path through a coarser
//  lens than the per-waypoint `s`: it only needs to know "which heavier regime
//  did this seed get demoted to". `PathRegime` is that coarse view — a band of
//  the path parameter `s` — and `PathDemotionReason` records *why* the cascade
//  asked for the demotion. A demotion is exactly a re-entry into a heavier
//  regime (it routes onto the same `reenter_heavier` mechanism as a spine
//  struggle or a mass-floor breach); there is NO rejection / disqualification
//  arm, mirroring `ContinuationStep`.
// ─────────────────────────────────────────────────────────────────────────

/// The coarse "which heavy-smoothing regime is the path currently entering at"
/// view the seed cascade reports against. Banded from the live path parameter
/// `s ∈ [0, 1]`: heavier regime ⇒ larger `s` ⇒ deeper into the contraction
/// basin. Every variant is a *re-entry* the cascade re-evaluates a seed at;
/// none of them is a rejection.
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
    /// `heavy_entry` starts in and the band repeated demotions converge toward.
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

/// Why the seed cascade asked the path to demote a seed to a heavier regime.
/// Purely a diagnostic tag carried into the demotion ledger — every variant
/// resolves to "re-enter the same seed at a heavier `s`", never to a rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathDemotionReason {
    /// A uniform structural diagnosis (rank / alias / active-set defect seen
    /// consistently across seeds) that the legacy contract would have used to
    /// short-circuit the cascade. For a continuation-entry objective it instead
    /// demotes to a heavier regime and keeps evaluating.
    UniformStructural,
    /// The continuation pre-warm refused to reach a seed at the current regime
    /// (a structural refusal of the seed's joint design). Demoted to a heavier
    /// regime so the joint solver gets a feasible basin the current regime could
    /// not reach.
    PrewarmStructural,
}

// ─────────────────────────────────────────────────────────────────────────
//  The per-step outcome enum. Note: NO Reject / Failed / NoUsableSeed arm.
// ─────────────────────────────────────────────────────────────────────────

/// Outcome of one [`ContinuationPath`] waypoint step. The defining structural
/// property: **there is no rejection arm.** A step either descends, arrives, or
/// bounces off the homotopy floor back into a heavier regime. None of these
/// means "give up"; the tail is a floor, not a gate. The absence of a `Reject`
/// variant is the whole point — the type cannot represent "no usable seed".
#[derive(Debug, Clone)]
pub(crate) enum ContinuationStep {
    /// `s` was lowered toward `0` and the inner solve at the new waypoint
    /// succeeded. Carries the accepted spine state and the new `s`.
    Descended { s: f64, state: ContinuationState },
    /// `s` reached `0`: the path arrived at the real objective (ρ\*, τ_min,
    /// tight isometry). Terminal-but-successful; the criterion is the real
    /// objective's, identical for cold and warm entry (#969).
    Arrived { state: ContinuationState },
    /// The inner solve at the attempted waypoint struggled, so the path
    /// re-entered a heavier regime (`s` raised back toward `1` by the back-off
    /// fraction) and will re-descend with a finer step. This is the homotopy
    /// floor in action — progress toward the fit, never abandonment. Carries
    /// the heavier `s` to descend from next and the underlying spine signal
    /// that prompted the back-off (for diagnostics only; it is **not** an
    /// error the path surfaces upward).
    Reentered { s: f64, reason: ReentryReason },
}

/// Why a waypoint re-entered a heavier regime. Purely diagnostic — every
/// variant resolves to "re-descend from a heavier `s`", never to a rejection.
#[derive(Debug, Clone)]
pub(crate) enum ReentryReason {
    /// The ρ-anneal spine could not complete the descent to this waypoint's ρ
    /// target from the current regime. The underlying `ContinuationFailure` is
    /// kept for logging; the path's response is unconditionally to re-enter a
    /// heavier regime, because at the heaviest regime the inner solve is a
    /// contraction and *must* converge.
    SpineStruggled(ContinuationFailure),
    /// The active-mass floor was breached at this waypoint; a scaffold re-seed
    /// was recorded and the path re-enters a heavier regime to let τ re-diffuse
    /// the assignment before re-sharpening.
    MassFloorBreached(MassFloorBreach),
    /// The descent step in `s` underflowed `S_STEP_FLOOR`; the path pins `s` at
    /// the current heavier regime and keeps re-descending from there rather
    /// than taking vanishing steps. Still not a rejection — the floor holds.
    StepUnderflow,
}

// ─────────────────────────────────────────────────────────────────────────
//  The ContinuationPath object.
// ─────────────────────────────────────────────────────────────────────────

/// Object 1 — the coupled continuation path. Owns the three schedules and the
/// scalar path parameter `s`, and drives the K≥2 SAE joint fit down the coupled
/// homotopy. Entry is always `s = 1` (heavy-smoothing contraction regime); the
/// tail is a homotopy floor with no rejection exit.
///
/// The wiring agent drives the path one waypoint at a time:
/// `let step = path.step(obj, &mut ledger);` and, per [`ContinuationStep`],
/// installs the next waypoint's [`ScalarLegTargets`] (τ on the SAE term,
/// isometry weight on the gauge penalty) and applies the [`LogitTrustRegion`] /
/// [`ActiveMassFloor`] hooks inside the inner solve.
#[derive(Debug, Clone)]
pub struct ContinuationPath {
    schedules: CoupledSchedules,
    /// Current path parameter. Starts at `1.0` (entry regime) and walks toward
    /// `0.0`. Re-entry raises it back toward `1.0`; descent lowers it.
    s: f64,
    /// Current descent step in `s`. Halved on re-entry, restored on a clean
    /// descent. Floored at [`S_STEP_FLOOR`] (a behavior, not an exit).
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
    /// cold-spine cost blowup. Kept across re-entries (a heavier waypoint is
    /// still downstream of a converged lighter-ρ state in walk distance).
    warm: Option<ContinuationState>,
    /// Budgeted spine evals spent so far (cold legs budget the full spine,
    /// warm legs budget [`WARM_LEG_EVAL_BUDGET`]). Compared against
    /// [`WALK_EVAL_CEILING`] for the #968 structural-termination guarantee.
    evals_budgeted: usize,
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
        let logit_tr = LogitTrustRegion::for_tau(entry_targets.tau);
        Self {
            schedules,
            s: 1.0,
            s_step: 1.0 / CONTINUATION_WAYPOINTS as f64,
            logit_tr,
            mass_floor: ActiveMassFloor::default_floor(),
            reseed_ledger: ReseedLedger::new(),
            warm: None,
            evals_budgeted: 0,
        }
    }

    /// No-argument heavy-smoothing entry for a continuation-entry objective
    /// (the seed cascade ctor). Builds the default coupled schedules — a
    /// single-component oversmoothed ρ leg, the standard diffuse→sharp τ leg and
    /// the loose→tight isometry gauge leg — and enters at `s = 1`, the
    /// heavy-smoothing contraction regime. The seed cascade only reads the
    /// coarse [`PathRegime`] and the logit step radius from the path; the
    /// concrete ρ vector is replaced by the spine's own per-component target at
    /// each waypoint via [`ContinuationPath::current_rho_target`], so the
    /// single-component default here is the entry placeholder, not a constraint
    /// on the real fit's dimensionality.
    #[must_use]
    pub fn heavy_entry() -> Self {
        Self::enter(default_coupled_schedules())
    }

    /// Heavy-smoothing entry coupled to a CONCRETE ρ target and legal box. The
    /// seed cascade rebuilds the path per-seed with this once it knows the
    /// objective's real ρ dimension (the no-argument [`ContinuationPath::heavy_entry`]
    /// is a dimension-1 placeholder used only before the seed is in hand). The
    /// ρ leg rides the spine from the spine's own oversmoothed ρ₀ down to
    /// `rho_target` (the real objective ρ\*); `bounds_upper` is the legal ρ box.
    /// The τ / isometry legs use the standard diffuse→sharp / loose→tight
    /// default endpoints. Enters at `s = 1`, the heavy-smoothing contraction
    /// regime. `rho_target` and `bounds_upper` must share length.
    #[must_use]
    pub fn heavy_entry_for_rho(rho_target: Array1<f64>, bounds_upper: Array1<f64>) -> Self {
        assert_eq!(
            rho_target.len(),
            bounds_upper.len(),
            "ContinuationPath::heavy_entry_for_rho: ρ target/bounds dim mismatch"
        );
        // Passing `rho_target` as both entry and target lets the spine own the
        // entire oversmoothing offset (it builds ρ₀ = ρ* + OVERSMOOTH_OFFSET_INIT
        // internally and anneals down), while the path simply rides at `s` along
        // ρ*. This keeps a single source of truth for the ρ anneal — the spine —
        // and the path couples the τ / isometry legs against that shared walk.
        let schedules = couple_schedules(
            rho_target.clone(),
            rho_target,
            bounds_upper,
            default_temperature_schedule(),
            default_isometry_schedule(),
        );
        Self::enter(schedules)
    }

    /// The coarse heavy-smoothing regime the path is currently entering at. The
    /// seed cascade reports this in its demotion ledger and final diagnosis. A
    /// fresh [`ContinuationPath::heavy_entry`] is in [`PathRegime::Heavy`].
    #[must_use]
    pub fn enter_regime(&self) -> PathRegime {
        PathRegime::from_s(self.s)
    }

    /// Demote the seed cascade to a heavier path regime with a recorded reason
    /// and return the regime re-entered at. This is the regime-escalation view
    /// of re-entry: it routes onto the same [`ContinuationPath::reenter_heavier`]
    /// mechanism a spine struggle or a mass-floor breach uses (raise `s` toward
    /// the entry regime, refine the step), so a structural diagnosis becomes a
    /// heavier-regime RE-ENTRY of the same seed — **never** a rejection. The
    /// `reason` is a diagnostic tag the caller records alongside the returned
    /// regime; the demotion mechanism is identical for every reason.
    pub fn demote_with_reason(&mut self, reason: PathDemotionReason) -> PathRegime {
        // The reason is diagnostic only: every demotion is a re-entry into a
        // heavier regime. Naming it explicitly keeps the value live (no silent
        // discard) while documenting that the escalation path is reason-agnostic.
        match reason {
            PathDemotionReason::UniformStructural | PathDemotionReason::PrewarmStructural => {
                self.reenter_heavier();
            }
        }
        self.enter_regime()
    }

    /// The base radius the per-iteration assignment-logit trust region is built
    /// from (`rho_optimizer.rs` / `atom_selection.rs` hardening hook). This is
    /// the ∞-norm logit step radius at the current waypoint; heavier regimes
    /// (after a demotion / re-entry) cool τ and so hand back a tighter radius,
    /// shrinking every atom's logit cap with no separate knob.
    #[must_use]
    pub fn logit_step_radius(&self) -> f64 {
        self.logit_tr.radius
    }

    /// Bare active-mass-floor breach hook for the inner-loop call site that does
    /// not thread its own [`ReseedLedger`]. Records the breach in the
    /// path-owned ledger at the current `s` and re-enters a heavier regime —
    /// the same non-fatal response as [`ContinuationPath::note_mass_breach`],
    /// without requiring the caller to carry a ledger. Returns the heavier
    /// [`PathRegime`] re-entered at so the call site can report it. **Never
    /// fatal** — a breach is a re-entry, never a rejection.
    pub fn note_active_mass_breach(&mut self) -> PathRegime {
        let breach = MassFloorBreach {
            observed_mean_mass: self.mass_floor.floor,
            floor: self.mass_floor.floor,
        };
        // Single source of truth for the breach response: route through
        // `note_mass_breach` so the record-then-re-enter logic is not
        // duplicated. The bare hook differs only in *which* ledger it threads —
        // the path-owned one — so we lend that ledger to the shared driver and
        // hand it back afterwards.
        let mut owned = std::mem::take(&mut self.reseed_ledger);
        let step = self.note_mass_breach(breach, &mut owned);
        self.reseed_ledger = owned;
        // The shared driver always re-enters a heavier regime (never rejects);
        // the bare hook's contract is the coarse regime it landed in. Match the
        // step exhaustively so the outcome is observed (no silent discard) and
        // the "every breach is progress" invariant is documented at the use
        // site: every arm resolves to the heavier live regime.
        match step {
            ContinuationStep::Reentered { .. }
            | ContinuationStep::Descended { .. }
            | ContinuationStep::Arrived { .. } => self.enter_regime(),
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
    pub fn current_scalar_targets(&self) -> ScalarLegTargets {
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

    /// Record an active-mass-floor breach into the ledger and re-enter a
    /// heavier regime. Returns the [`ContinuationStep::Reentered`] the wiring
    /// agent should act on. **Never fatal** — a breach is a re-entry, never a
    /// rejection. This is the hook the wiring agent calls when
    /// [`ActiveMassFloor::check`] returns `Some` from inside the inner solve.
    pub(crate) fn note_mass_breach(
        &mut self,
        breach: MassFloorBreach,
        ledger: &mut ReseedLedger,
    ) -> ContinuationStep {
        ledger.record(self.s, breach);
        self.reenter_heavier();
        ContinuationStep::Reentered {
            s: self.s,
            reason: ReentryReason::MassFloorBreached(breach),
        }
    }

    /// Raise `s` back toward the entry regime by the back-off fraction and
    /// halve the descent step (finer re-descent). Floors the step at
    /// [`S_STEP_FLOOR`]; underflow does not abandon the path, it pins the
    /// heavier regime. Recomputes the τ-tied logit trust region for the
    /// heavier regime.
    fn reenter_heavier(&mut self) {
        self.s = (self.s + REENTRY_BACKOFF).min(1.0);
        self.s_step = (self.s_step * 0.5).max(S_STEP_FLOOR);
        self.logit_tr = LogitTrustRegion::for_tau(self.schedules.scalar_targets_at(self.s).tau);
    }

    /// Whether the path has arrived at (or below) the real objective `s = 0`.
    /// The outer driver stops driving [`ContinuationPath::step`] once this is
    /// true and hands the warm iterate to the normal optimizer at ρ\*.
    #[must_use]
    pub fn arrived(&self) -> bool {
        self.s <= 0.0
    }

    /// Take one waypoint step down the coupled homotopy.
    ///
    /// 1. Lower `s` by the current step toward `0`.
    /// 2. Advance the τ and isometry schedules to the new waypoint (lockstep).
    /// 3. Run the ρ-anneal **spine** ([`fit_with_continuation`]) toward the new
    ///    `s`'s ρ target, with the inner β carried warm.
    /// 4. On spine success: [`ContinuationStep::Descended`] (or
    ///    [`ContinuationStep::Arrived`] if `s` reached `0`).
    /// 5. On spine struggle: re-enter a heavier regime and return
    ///    [`ContinuationStep::Reentered`]. **No rejection branch exists.**
    ///
    /// `obj` is the SAE joint outer objective (`SaeManifoldOuterObjective`,
    /// which is an [`OuterObjective`]). `initial_beta` warms the inner solve;
    /// pass the empty array for cold entry (warm-invariance, #969, guarantees
    /// the same destination either way).
    pub(crate) fn step(
        &mut self,
        obj: &mut dyn OuterObjective,
        initial_beta: &Array1<f64>,
    ) -> ContinuationStep {
        // #968 hard ceiling: total budgeted spine evals across the walk are
        // bounded. At the ceiling the path hands its best converged state to
        // the real optimizer (legs advanced to the target regime) instead of
        // spending another leg — termination is structural, not statistical.
        // With no converged state yet the walk keeps trying (the consumer's
        // own `CONTINUATION_WALK_BUDGET` loop bounds that case).
        if self.evals_budgeted >= WALK_EVAL_CEILING {
            if let Some(state) = self.warm.clone() {
                log::warn!(
                    "[PATH] walk eval ceiling {WALK_EVAL_CEILING} reached at s={:.4}; arriving \
                     with the best converged waypoint state (scalar legs advanced to target)",
                    self.s
                );
                self.advance_scalar_legs_to(0.0);
                self.s = 0.0;
                return ContinuationStep::Arrived { state };
            }
        }

        // Descent step in s, floored. If the step has already underflowed, the
        // path pins the heavier regime and re-descends from there — still no
        // rejection.
        if self.s_step < S_STEP_FLOOR {
            self.reenter_heavier();
            return ContinuationStep::Reentered {
                s: self.s,
                reason: ReentryReason::StepUnderflow,
            };
        }

        let s_next = (self.s - self.s_step).max(0.0);

        // Advance the coupled scalar legs to the new waypoint. The schedule
        // objects are stepped in lockstep so τ and the isometry weight track
        // exactly the same path parameter the ρ leg is about to anneal to.
        self.advance_scalar_legs_to(s_next);

        // The ρ leg rides the spine: anneal from the spine's own oversmoothed
        // ρ₀ down to this waypoint's ρ target. At s = 1 the waypoint ρ target
        // is ρ₀ itself, so the spine's oversmoothing stacks into the deepest
        // contraction; at s = 0 it is ρ*.
        let rho_target = self.schedules.rho_target_at(s_next);
        // First leg (no converged waypoint yet): the full oversmoothed spine —
        // the documented deepest-contraction entry. Every later waypoint is a
        // WARM leg from the previous waypoint's converged state. The coupled
        // path's waypoints ARE the anneal; re-running the whole ρ₀→target
        // spine per waypoint multiplies the walk's cost by the spine budget
        // (the K=2 existence fixture burned 7 CPU-hours exactly that way).
        let spine = match self.warm.clone() {
            Some(start) => {
                self.evals_budgeted += WARM_LEG_EVAL_BUDGET;
                // The coupled path is still a continuation pre-warm: its only
                // payload is the converged inner coefficient state carried by
                // `inner_beta_hint`.  Requesting an outer gradient here repeats
                // the expensive REML/LAML derivative assembly at every waypoint
                // even though no line search or stationarity test consumes it;
                // the real outer optimizer asks for the gradient once after the
                // path arrives.  `Value` preserves the same inner solve and warm
                // beta propagation while removing that redundant derivative work.
                continue_path_from(
                    obj,
                    start,
                    &rho_target,
                    OuterEvalOrder::Value,
                    WARM_LEG_EVAL_BUDGET,
                )
            }
            None => {
                self.evals_budgeted += PATH_BUDGET;
                fit_with_continuation(
                    obj,
                    &rho_target,
                    &self.schedules.rho_bounds_upper,
                    initial_beta,
                    OuterEvalOrder::Value,
                )
            }
        };

        match spine {
            Ok(state) => {
                self.warm = Some(state.clone());
                self.s = s_next;
                // Clean descent: restore the nominal step (grow back toward the
                // coarse schedule) and refresh the τ-tied logit trust region.
                self.s_step = (1.0 / CONTINUATION_WAYPOINTS as f64).min(self.s.max(S_STEP_FLOOR));
                self.logit_tr =
                    LogitTrustRegion::for_tau(self.schedules.scalar_targets_at(self.s).tau);
                if self.s <= 0.0 {
                    ContinuationStep::Arrived { state }
                } else {
                    ContinuationStep::Descended { s: self.s, state }
                }
            }
            Err(failure) => {
                // The homotopy FLOOR: never reject. Re-enter a heavier regime
                // and re-descend with a finer step. At the heaviest regime the
                // inner solve is a contraction and must converge, so the floor
                // is reachable in finitely many back-offs.
                self.reenter_heavier();
                ContinuationStep::Reentered {
                    s: self.s,
                    reason: ReentryReason::SpineStruggled(failure),
                }
            }
        }
    }

    /// Advance the τ and isometry schedule objects so their live values match
    /// the lockstep targets at `s_next`. Consumes the schedules' own
    /// `current_*` laws by selecting the schedule iteration whose output is
    /// closest to the coupled target, keeping a single source of truth for each
    /// leg's interpolation (no parallel re-derivation of the decay law).
    fn advance_scalar_legs_to(&mut self, s_next: f64) {
        let targets = self.schedules.scalar_targets_at(s_next);
        // τ: walk the schedule's iteration counter to the step whose
        // `current_tau` first reaches (≤) the coupled target, so the live τ on
        // the SAE term equals the coupled-path value. Monotone-decreasing, so a
        // forward scan from the current count is correct and terminates at
        // tau_min.
        Self::advance_temperature_to(&mut self.schedules.temperature, targets.tau);
        Self::advance_isometry_to(&mut self.schedules.isometry, targets.isometry_weight);
        self.logit_tr = LogitTrustRegion::for_tau(targets.tau);
    }

    /// Step `schedule.iter_count` forward until `current_tau` is ≤ `target_tau`
    /// (τ is monotone non-increasing in iter). Leaves the counter pointing at
    /// the waypoint so the SAE term reads the coupled τ. Bounded by the
    /// schedule's own `tau_min` floor — never spins past it.
    fn advance_temperature_to(schedule: &mut GumbelTemperatureSchedule, target_tau: f64) {
        // Guard: a malformed schedule can't make progress; clamp to one step so
        // the live τ is still the schedule's current value, never NaN.
        let max_scan = temperature_scan_budget(schedule);
        let mut scanned = 0;
        while scanned < max_scan && schedule.current_tau(schedule.iter_count) > target_tau {
            schedule.iter_count += 1;
            scanned += 1;
        }
    }

    /// Step `schedule.iter_count` forward until `current_weight` is ≥
    /// `target_weight` (isometry weight is monotone non-decreasing in iter when
    /// `w_end ≥ w_start`, the tightening direction). Bounded by `w_end`.
    fn advance_isometry_to(schedule: &mut ScalarWeightSchedule, target_weight: f64) {
        let max_scan = isometry_scan_budget(schedule);
        let mut scanned = 0;
        while scanned < max_scan && schedule.current_weight(schedule.iter_count) < target_weight {
            schedule.iter_count += 1;
            scanned += 1;
        }
    }
}

/// Scan budget for advancing the temperature schedule. For a `Linear` schedule
/// the number of steps is known; for geometric / reciprocal it is bounded by a
/// generous waypoint multiple so the lockstep scan always terminates.
fn temperature_scan_budget(schedule: &GumbelTemperatureSchedule) -> usize {
    const GEOMETRIC_SCAN_CAP: usize = 4096;
    match &schedule.decay {
        ScheduleKind::Linear { steps } => *steps + 1,
        ScheduleKind::Geometric { .. } | ScheduleKind::ReciprocalIter => GEOMETRIC_SCAN_CAP,
    }
}

/// Scan budget for advancing the isometry-weight schedule (mirrors
/// [`temperature_scan_budget`]).
fn isometry_scan_budget(schedule: &ScalarWeightSchedule) -> usize {
    const GEOMETRIC_SCAN_CAP: usize = 4096;
    match &schedule.kind {
        ScheduleKind::Linear { steps } => *steps + 1,
        ScheduleKind::Geometric { .. } | ScheduleKind::ReciprocalIter => GEOMETRIC_SCAN_CAP,
    }
}

/// Convenience: build the standard coupled schedules for a K≥2 SAE joint fit
/// from the ρ box and the τ / isometry schedules the term already carries.
///
/// `rho_target` is ρ\* (the real objective); `rho_entry` is the oversmoothed
/// entry ρ₀ (caller supplies, or the spine derives its own offset on top —
/// passing `rho_target` here lets the spine own the entire oversmoothing and
/// the path simply rides at `s` along ρ\*). `rho_bounds_upper` is the legal box.
#[must_use]
pub fn couple_schedules(
    rho_entry: Array1<f64>,
    rho_target: Array1<f64>,
    rho_bounds_upper: Array1<f64>,
    temperature: GumbelTemperatureSchedule,
    isometry: ScalarWeightSchedule,
) -> CoupledSchedules {
    CoupledSchedules {
        rho_entry,
        rho_target,
        rho_bounds_upper,
        temperature,
        isometry,
    }
}

/// Default coupled schedules for a no-argument [`ContinuationPath::heavy_entry`].
///
/// Builds the standard three legs at their smoothing-extreme entry values:
/// * ρ — a single-component oversmoothed entry `ρ₀` descending to `ρ* = 0`,
///   inside a generous legal box. The seed cascade's spine replaces this with
///   the real per-component ρ target at each waypoint, so the single component
///   here is only the entry placeholder.
/// * τ — the diffuse→sharp assignment-temperature leg (`DEFAULT_ENTRY_TAU` down
///   to `DEFAULT_TARGET_TAU`) over the standard waypoint count.
/// * isometry — the loose→tight gauge leg (`DEFAULT_ENTRY_ISOMETRY` up to the
///   tight target weight) over the same waypoint count.
///
/// These endpoints match the smoothing-extreme entry regime every leg is at at
/// `s = 1`; the path walks them down in lockstep exactly as a caller-supplied
/// [`CoupledSchedules`] would.
#[must_use]
fn default_coupled_schedules() -> CoupledSchedules {
    /// Oversmoothed entry ρ₀ for the single-component placeholder leg.
    const DEFAULT_ENTRY_RHO: f64 = 5.0;
    /// Legal ρ upper bound for the placeholder leg.
    const DEFAULT_RHO_UPPER: f64 = 10.0;

    couple_schedules(
        Array1::from_elem(1, DEFAULT_ENTRY_RHO),
        Array1::zeros(1),
        Array1::from_elem(1, DEFAULT_RHO_UPPER),
        default_temperature_schedule(),
        default_isometry_schedule(),
    )
}

/// The standard diffuse→sharp assignment-temperature leg (`DEFAULT_ENTRY_TAU`
/// down to `DEFAULT_TARGET_TAU`) over the standard waypoint count. Shared by
/// both [`ContinuationPath::heavy_entry`] and
/// [`ContinuationPath::heavy_entry_for_rho`] so the τ leg has one source.
#[must_use]
fn default_temperature_schedule() -> GumbelTemperatureSchedule {
    /// Diffuse entry τ (the schedule's `tau_start`) at `s = 1`. Entry
    /// heaviness is tied to the cold-seed logit scale: the production IBP
    /// residual-energy seed emits logits at gain `4.0`
    /// (`SAE_RESIDUAL_SEED_GAIN` in `gam-pyffi`), so seeded logits span
    /// roughly `±4`. Entry τ ≥ that gain keeps every seeded row in the
    /// near-linear band of the gate (`|logit|/τ ≤ 1`), where the assignment
    /// map is smooth and contractive — no row enters pre-saturated against
    /// the argmax cliff. The previous `2.0` entry let ±4-gain seeds start at
    /// `|logit|/τ = 2`, already in the saturated tail.
    const DEFAULT_ENTRY_TAU: f64 = 4.0;
    /// Target τ (`tau_min`) at `s = 0`. The `s = 0` endpoint of every leg
    /// must be the REAL objective's value, and the production IBP-MAP
    /// assignment temperature (gamfit `sae_manifold_fit`, `ibp_map` path) is
    /// `τ = 0.5`. The previous `0.1` target over-sharpened the leg *past*
    /// the real objective, which tightened the τ-tied logit trust region at
    /// arrival to radius `0.4` (vs `2.0` at the true operating τ) — choking
    /// exactly the late-walk logit growth the routing mass needs to climb
    /// from the diffuse entry to the planted level.
    const DEFAULT_TARGET_TAU: f64 = 0.5;
    GumbelTemperatureSchedule::new(
        DEFAULT_ENTRY_TAU,
        DEFAULT_TARGET_TAU,
        ScheduleKind::Linear {
            steps: CONTINUATION_WAYPOINTS,
        },
    )
    .expect("default continuation temperature schedule must be valid")
}

/// The standard loose→tight isometry gauge leg (`DEFAULT_ENTRY_ISOMETRY` up to
/// `DEFAULT_TARGET_ISOMETRY`) over the standard waypoint count. Shared source
/// for the isometry leg across both heavy-entry constructors.
#[must_use]
fn default_isometry_schedule() -> ScalarWeightSchedule {
    /// Entry isometry weight (`w_start`) at `s = 1`; the chart pin starts fully
    /// off and ramps after the anchor has settled.
    const DEFAULT_ENTRY_ISOMETRY: f64 = 0.0;
    /// Tight target isometry weight (`w_end`) at `s = 0`.
    const DEFAULT_TARGET_ISOMETRY: f64 = 1.0;
    ScalarWeightSchedule::new(
        DEFAULT_ENTRY_ISOMETRY,
        DEFAULT_TARGET_ISOMETRY,
        ScheduleKind::Linear {
            steps: CONTINUATION_WAYPOINTS,
        },
    )
    .expect("default continuation isometry schedule must be valid")
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

    fn lin_temp() -> GumbelTemperatureSchedule {
        GumbelTemperatureSchedule::new(2.0, 0.1, ScheduleKind::Linear { steps: 8 })
            .expect("valid temperature schedule")
    }

    fn lin_iso() -> ScalarWeightSchedule {
        ScalarWeightSchedule::new(0.01, 1.0, ScheduleKind::Linear { steps: 8 })
            .expect("valid isometry schedule")
    }

    fn schedules() -> CoupledSchedules {
        couple_schedules(
            Array1::from_vec(vec![5.0, 5.0]),
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![10.0, 10.0]),
            lin_temp(),
            lin_iso(),
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
        // τ at entry is the diffuse extreme (tau_start), isometry is loose
        // (w_start).
        assert!((targets.tau - 2.0).abs() < 1e-12, "entry τ = tau_start");
        assert!(
            (targets.isometry_weight - 0.01).abs() < 1e-12,
            "entry isometry = w_start"
        );
        // ρ target at s = 1 is the oversmoothed entry ρ₀.
        let rho = path.current_rho_target();
        assert!((rho[0] - 5.0).abs() < 1e-12 && (rho[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn target_endpoint_is_the_real_objective() {
        let sch = schedules();
        let targets0 = sch.scalar_targets_at(0.0);
        assert!(
            (targets0.tau - 0.1).abs() < 1e-12,
            "s=0 τ = tau_min (sharp)"
        );
        assert!(
            (targets0.isometry_weight - 1.0).abs() < 1e-12,
            "s=0 isometry = w_end (tight)"
        );
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
        assert!((mid.tau - (0.1 + 0.5 * (2.0 - 0.1))).abs() < 1e-12);
        assert!((mid.isometry_weight - (0.01 + 0.5 * (1.0 - 0.01))).abs() < 1e-12);
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
    fn note_mass_breach_reenters_heavier_and_logs() {
        let mut path = ContinuationPath::enter(schedules());
        // Walk s down a bit first so a re-entry visibly raises it.
        path.s = 0.5;
        let mut ledger = ReseedLedger::new();
        let breach = MassFloorBreach {
            observed_mean_mass: 0.05,
            floor: ActiveMassFloor::DEFAULT_FLOOR,
        };
        let step = path.note_mass_breach(breach, &mut ledger);
        assert!(matches!(
            step,
            ContinuationStep::Reentered {
                reason: ReentryReason::MassFloorBreached(_),
                ..
            }
        ));
        assert!(
            path.s() > 0.5,
            "re-entry must raise s toward the entry regime"
        );
        assert_eq!(ledger.reseed_count(), 1);
    }

    #[test]
    fn continuation_step_has_no_reject_arm() {
        // Compile-time + exhaustiveness witness: every ContinuationStep value
        // resolves to a heavier-regime re-entry. There is no rejection arm, so a
        // `match` over the enum cannot bind a "give up" case. If a Reject variant
        // were ever added, this match would fail to compile against the
        // documented invariant.
        fn is_progress(step: &ContinuationStep) -> bool {
            match step {
                ContinuationStep::Descended { .. }
                | ContinuationStep::Arrived { .. }
                | ContinuationStep::Reentered { .. } => true,
            }
        }
        let breach = MassFloorBreach {
            observed_mean_mass: 0.0,
            floor: 0.2,
        };
        assert!(is_progress(&ContinuationStep::Reentered {
            s: 1.0,
            reason: ReentryReason::MassFloorBreached(breach),
        }));
        assert!(is_progress(&ContinuationStep::Reentered {
            s: 1.0,
            reason: ReentryReason::StepUnderflow,
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
        let path = ContinuationPath::heavy_entry();
        assert_eq!(path.s(), 1.0, "heavy_entry must enter at s = 1");
        assert_eq!(
            path.enter_regime(),
            PathRegime::Heavy,
            "a fresh heavy_entry is in the heavy-smoothing regime"
        );
        assert!(
            path.logit_step_radius().is_finite() && path.logit_step_radius() > 0.0,
            "logit step radius must be finite and positive at entry"
        );
    }

    #[test]
    fn demote_with_reason_reenters_heavier_never_rejects() {
        let mut path = ContinuationPath::heavy_entry();
        // Walk down so a demotion visibly raises s back toward the entry regime.
        path.s = 0.3;
        path.s_step = 0.1;
        let before = path.s;
        let regime = path.demote_with_reason(PathDemotionReason::UniformStructural);
        assert!(
            path.s > before,
            "demotion must raise s toward the entry regime"
        );
        // The returned regime is the coarse band of the (heavier) live s.
        assert_eq!(regime, path.enter_regime());
        // A second reason demotes the same way — reason-agnostic escalation.
        let regime2 = path.demote_with_reason(PathDemotionReason::PrewarmStructural);
        assert_eq!(regime2, path.enter_regime());
        assert!(path.s >= before, "repeated demotions never lower s");
    }

    #[test]
    fn bare_active_mass_breach_records_and_reenters() {
        let mut path = ContinuationPath::heavy_entry();
        path.s = 0.4;
        assert_eq!(path.reseed_count(), 0);
        let before = path.s;
        let regime = path.note_active_mass_breach();
        assert_eq!(
            path.reseed_count(),
            1,
            "breach must be recorded in the path ledger"
        );
        assert!(path.s > before, "breach must re-enter a heavier regime");
        assert_eq!(regime, path.enter_regime());
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
        ) -> Result<crate::rho_optimizer::SeedOutcome, crate::model_types::EstimationError> {
            self.seed_count += beta.len().max(1);
            Ok(crate::rho_optimizer::SeedOutcome::Installed)
        }
    }

    #[test]
    fn coupled_path_prewarm_requests_value_only_evals() {
        let mut path = ContinuationPath::enter(schedules());
        let mut obj = RecordingObjective::default();
        let initial_beta = Array1::zeros(0);

        let step = path.step(&mut obj, &initial_beta);
        assert!(
            matches!(
                step,
                ContinuationStep::Descended { .. } | ContinuationStep::Arrived { .. }
            ),
            "recording objective should accept the first coupled-path waypoint"
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
    fn reentry_floors_step_but_never_exits() {
        let mut path = ContinuationPath::enter(schedules());
        path.s = 0.5;
        // Force many re-entries; s_step must floor, s stays in [0,1], and the
        // path never produces a non-progress outcome.
        for _ in 0..50 {
            path.reenter_heavier();
            assert!(path.s_step >= S_STEP_FLOOR);
            assert!((0.0..=1.0).contains(&path.s));
        }
    }
}
