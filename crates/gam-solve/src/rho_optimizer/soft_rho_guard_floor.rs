//! Decide, from a gradient ladder alone, whether an outer objective's criterion
//! carries the soft ρ-guard barrier (#2629 scope item 2).
//!
//! # Why this exists
//!
//! [`OuterObjective::soft_rho_guard_gradient`] returns `None` by default, and
//! that default is *correct* for most objective families and *wrong* for the
//! ones built on a `RemlState`. The two are byte-identical at the trait: an
//! objective that carries the barrier and says nothing looks exactly like one
//! that has no barrier at all. #2545 fixed one family, #2629 fixed the second,
//! and the table of "which of the remaining families carry it" in #2629 was
//! **grep evidence** — `RemlState::build_prior` is the only site that adds
//! [`SoftRhoGuardPriorAtom`]'s gradient to a criterion, so a family that never
//! reaches it cannot carry the barrier.
//!
//! Grep evidence is an argument about call graphs. This module is the
//! measurement that settles it, and it is the same instrument #2545's
//! acceptance used, lifted out of that one test so every family can be put
//! through it:
//!
//! > A REML/LAML criterion's λ→∞ face gives `∂V/∂ρ = c·e^{−ρ}`. The barrier's
//! > gradient `w·a·tanh(a·ρ̃)` **saturates** at `w·a` instead of decaying. So on
//! > a ladder of saturated ρ the two hypotheses are trivially separable:
//! > exactly one of `g` and `g − guard` has a constant pencil `c = r·e^{ρ}`.
//!
//! At `ρ ∈ {21, 24, 27, 30}` the discrimination is not marginal — it is four
//! orders of magnitude. `e^{30}/e^{21} ≈ 8100`, so a floor mistaken for a face
//! makes `ĉ` grow by that factor across the ladder, while a face mistaken for a
//! floor leaves a residual of the wrong sign at every rung.
//!
//! # What it is NOT
//!
//! It is not a stationarity test, a convergence test, or a tolerance. It answers
//! one yes/no question about an objective's *construction*, from data the
//! certificate's tail probe already collects. Its verdicts carry the numbers
//! they were reached on, so a refusal is readable without re-running anything.
//!
//! [`OuterObjective::soft_rho_guard_gradient`]: super::OuterObjective::soft_rho_guard_gradient
//! [`SoftRhoGuardPriorAtom`]: crate::estimate::reml::atoms::SoftRhoGuardPriorAtom

/// The saturated-ρ ladder #2450 established and #2545/#2629 measure on.
///
/// `ρ ≥ 21` is where the REML part's own ρ-derivative was measured below
/// `1e-10` on the reference fixture, so anything left there is not the REML
/// tail; `RHO_BOUND = 30` is the deepest point the box admits. Four rungs is
/// the minimum that makes the pencil's *constancy* (three independent ratios)
/// an observation rather than a definition.
pub const SATURATED_RHO_LADDER: [f64; 4] = [21.0, 24.0, 27.0, 30.0];

/// How constant `ĉ = r·e^{ρ}` must be across the ladder for a hypothesis to
/// hold, as a relative spread `(max − min)/max`.
///
/// #2545's acceptance measured `87.512 / 87.512 / 87.511 / 87.474` across the
/// four rungs — a spread of `4.3e-4`. One percent is two orders of headroom
/// over that and still refuses the divergent `ĉ` a saturating floor produces
/// (which is off by a factor `e^9 ≈ 8100`, five orders past the bar).
pub const PENCIL_CONSTANCY_TOL: f64 = 1.0e-2;

/// Below this fraction of the barrier's own emission, a gradient cannot be
/// hiding the floor.
///
/// The floor is a fixed positive constant at every rung. For a criterion that
/// carries it to nevertheless present `|g| < f·guard`, its own face tail would
/// have to cancel the barrier to within `f` at *every* rung — i.e. the tail
/// would have to track `−guard`, a constant, which is the one thing a decaying
/// face cannot do. `f = 1/2` makes that argument with a factor of two to spare.
pub const ABSENCE_MAGNITUDE_FRACTION: f64 = 0.5;

/// One rung of a saturated-ρ gradient ladder: the outer ρ-gradient of ONE
/// coordinate, at one saturated ρ.
///
/// `rho_gradient` is **signed** and is the coordinate's own `∂V/∂ρ_i` as the
/// objective reports it — not `max_j |g_j|`, and not the certificate's
/// barrier-subtracted view. The sign matters: the face tail `c·e^{−ρ}` may
/// approach the rail from either side (measured `c = +87.5` on the #2450
/// Matérn/Gaussian fixture and `c = −22.8` on the #2629 SAS/binomial one), and
/// a classifier fed `|g|` cannot tell a sign flip from a floor.
///
/// To classify a whole objective, build one ladder per ρ-coordinate. The
/// barrier is added to every coordinate identically by `build_prior`, so the
/// verdicts must agree; a disagreement is itself a finding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuardLadderRung {
    /// The ρ this coordinate was held at. Must be a saturated value — deep
    /// enough that the criterion's own tail is small next to the floor.
    pub rho: f64,
    /// The coordinate's signed `∂V/∂ρ_i` there, barrier included if the
    /// criterion adds one.
    pub rho_gradient: f64,
}

/// The fit of one hypothesis (`floor present` / `floor absent`) to a ladder.
///
/// Carried by every verdict so a refusal reports the numbers it was reached on.
#[derive(Clone, Debug, PartialEq)]
pub struct PencilFit {
    /// `ĉ_j = r_j · e^{ρ_j}` at each rung, `r` being the hypothesis' residual.
    pub pencil: Vec<f64>,
    /// The mean of `pencil`, the hypothesis' estimate of the face constant.
    pub constant: f64,
    /// `(max|ĉ| − min|ĉ|)/max|ĉ|`; `f64::INFINITY` when the pencil is not a
    /// single-signed nonzero run (so no constant exists to be spread).
    pub spread: f64,
    /// Whether this hypothesis holds: a finite, single-signed, nonzero pencil
    /// whose spread is within [`PENCIL_CONSTANCY_TOL`].
    pub holds: bool,
}

/// What a ladder says about the objective that produced it.
#[derive(Clone, Debug, PartialEq)]
pub enum SoftRhoGuardFloor {
    /// The criterion ADDS the barrier. `g − guard` is a clean `c·e^{−ρ}` face
    /// tail while `g` itself is pinned near the saturating floor.
    ///
    /// An objective in this state MUST install `with_soft_rho_guard_gradient`
    /// (or override `soft_rho_guard_gradient`); if it does not, every railed
    /// coordinate of every fit it produces carries a standing `|Pg| ≥ w·a`
    /// stationarity residual that no amount of convergence can clear. That is
    /// precisely the #2545/#2629 defect.
    Carried {
        /// The face constant `ĉ` under the barrier.
        face: PencilFit,
        /// The barrier's own emission at each rung.
        guard: Vec<f64>,
    },
    /// The criterion does NOT add the barrier: `g` is itself a clean
    /// `c·e^{−ρ}` face tail with no floor under it. `None` is the correct
    /// answer for this objective at the trait.
    AbsentDecayingFace {
        /// The face constant `ĉ` measured on the raw gradient.
        face: PencilFit,
        /// The barrier's emission at each rung, i.e. what the floor WOULD have
        /// been. Reported so the margin is readable.
        guard: Vec<f64>,
    },
    /// The criterion does not add the barrier, established by magnitude rather
    /// than by shape: the gradient is below what the floor alone would be, so
    /// there is no floor to be found regardless of what the tail is doing.
    ///
    /// This is the verdict for a family whose face has decayed all the way into
    /// roundoff — where the pencil is noise and only the magnitude argument is
    /// available. It is no weaker than [`Self::AbsentDecayingFace`]: it says the
    /// floor is *not present*, which is the whole question.
    AbsentBelowTheFloor {
        /// `max_j |g_j|` over the ladder.
        max_abs_gradient: f64,
        /// `min_j guard(ρ_j)` over the ladder — the floor's smallest value.
        min_guard: f64,
    },
    /// Neither hypothesis holds. The ladder does not answer the question; say
    /// so rather than pick the nearer one.
    ///
    /// Real causes, in the order they are worth checking: the ladder is not
    /// deep enough (the criterion's own tail is still comparable to the floor),
    /// the fit is not converged at these ρ so the "gradient" is not a tail at
    /// all, or the objective carries a DIFFERENT non-decaying term (a
    /// configured ρ-prior, for instance) that neither hypothesis models.
    Indeterminate {
        /// Fit of "the criterion adds the barrier".
        with_floor: PencilFit,
        /// Fit of "the criterion adds no barrier".
        without_floor: PencilFit,
        /// Why no verdict was reached, in a form that names the numbers.
        reason: String,
    },
}

impl SoftRhoGuardFloor {

    /// A one-line rendering suitable for a refusal message or a test failure.
    pub fn summary(&self) -> String {
        match self {
            Self::Carried { face, guard } => format!(
                "CARRIED: the criterion adds the soft rho-guard barrier \
                 (guard {:.6e} at the deepest rung) over a clean face tail \
                 c={:.6e} (pencil spread {:.3e})",
                guard.last().copied().unwrap_or(f64::NAN),
                face.constant,
                face.spread
            ),
            Self::AbsentDecayingFace { face, guard } => format!(
                "ABSENT: the gradient is itself a clean c*e^-rho face tail \
                 c={:.6e} (pencil spread {:.3e}); a floor would have been \
                 {:.6e} at the deepest rung",
                face.constant,
                face.spread,
                guard.last().copied().unwrap_or(f64::NAN)
            ),
            Self::AbsentBelowTheFloor {
                max_abs_gradient,
                min_guard,
            } => format!(
                "ABSENT: max|g| over the ladder is {max_abs_gradient:.6e}, below \
                 the floor's own smallest value {min_guard:.6e} — there is no \
                 floor under a gradient smaller than the floor"
            ),
            Self::Indeterminate {
                with_floor,
                without_floor,
                reason,
            } => format!(
                "INDETERMINATE: {reason} (with-floor pencil spread {:.3e}, \
                 without-floor pencil spread {:.3e})",
                with_floor.spread, without_floor.spread
            ),
        }
    }
}

#[cfg(test)]
#[path = "soft_rho_guard_floor_tests.rs"]
mod soft_rho_guard_floor_tests;
