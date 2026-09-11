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

use ndarray::Array1;

use crate::estimate::reml::atoms::SoftRhoGuardPriorAtom;
use crate::estimate::{RHO_BOUND, RHO_SOFT_PRIOR_SHARPNESS, RHO_SOFT_PRIOR_WEIGHT};

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

impl PencilFit {
    fn from_residuals(ladder: &[GuardLadderRung], residual: impl Fn(&GuardLadderRung) -> f64) -> Self {
        let pencil: Vec<f64> = ladder
            .iter()
            .map(|rung| residual(rung) * rung.rho.exp())
            .collect();
        let constant = if pencil.is_empty() {
            0.0
        } else {
            pencil.iter().sum::<f64>() / pencil.len() as f64
        };
        let all_finite = pencil.iter().all(|c| c.is_finite());
        let all_positive = pencil.iter().all(|c| *c > 0.0);
        let all_negative = pencil.iter().all(|c| *c < 0.0);
        let single_signed = all_finite && (all_positive || all_negative);
        let (spread, holds) = if !single_signed {
            (f64::INFINITY, false)
        } else {
            let max = pencil.iter().fold(0.0f64, |acc, c| acc.max(c.abs()));
            let min = pencil.iter().fold(f64::MAX, |acc, c| acc.min(c.abs()));
            let spread = (max - min) / max;
            (spread, spread <= PENCIL_CONSTANCY_TOL)
        };
        Self {
            pencil,
            constant,
            spread,
            holds,
        }
    }
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
    /// `true` for both absence verdicts — the question the trait's `None`
    /// default answers.
    pub fn is_absent(&self) -> bool {
        matches!(
            self,
            Self::AbsentDecayingFace { .. } | Self::AbsentBelowTheFloor { .. }
        )
    }

    /// `true` when the criterion demonstrably adds the barrier and therefore
    /// owes the seam a publication.
    pub fn is_carried(&self) -> bool {
        matches!(self, Self::Carried { .. })
    }

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

/// The soft ρ-guard barrier's own gradient emission at `rho`, from the SAME
/// atom `RemlState::build_prior` adds.
///
/// Not a re-derivation: this calls [`SoftRhoGuardPriorAtom::evaluate_anchored`]
/// with the shipped policy constants, so a change to the weight, the sharpness,
/// or the bound moves the criterion and this classifier together. A closed form
/// written here would be a second copy of the thing #931 collapsed into one
/// atom, and it would silently disagree the moment the anchor is nonzero.
///
/// `anchor` is `RemlState::rho_weight_anchor()` — exactly `0.0` for an
/// unweighted fit, `log g(w)` otherwise (#877). Passing `0.0` for a weighted
/// state measures the wrong function, and does so quietly: that is the failure
/// mode #2545's doc calls out, so it is a required argument rather than a
/// default.
pub fn soft_rho_guard_emission_at(rho: f64, anchor: f64) -> f64 {
    SoftRhoGuardPriorAtom::evaluate_anchored(
        &Array1::from_elem(1, rho),
        RHO_SOFT_PRIOR_WEIGHT,
        RHO_SOFT_PRIOR_SHARPNESS,
        RHO_BOUND,
        anchor,
    )
    .gradient()[0]
}

/// Classify one ρ-coordinate's saturated ladder.
///
/// See the module docs for the discrimination. In short: fit `ĉ = r·e^{ρ}` under
/// both hypotheses (`r = g − guard` and `r = g`) and accept the one whose pencil
/// is a single-signed constant. If the gradient is smaller than the floor
/// itself, report absence on that ground alone. If neither hypothesis holds,
/// report [`SoftRhoGuardFloor::Indeterminate`] with both fits rather than
/// rounding to the nearer one.
///
/// `anchor` is the weight anchor the criterion's barrier was evaluated at; see
/// [`soft_rho_guard_emission_at`].
pub fn classify_soft_rho_guard_floor(
    ladder: &[GuardLadderRung],
    anchor: f64,
) -> SoftRhoGuardFloor {
    let empty = || PencilFit {
        pencil: Vec::new(),
        constant: 0.0,
        spread: f64::INFINITY,
        holds: false,
    };
    if ladder.len() < 3 {
        return SoftRhoGuardFloor::Indeterminate {
            with_floor: empty(),
            without_floor: empty(),
            reason: format!(
                "a ladder of {} rung(s) cannot show a pencil to be CONSTANT — \
                 three rungs are the minimum that makes constancy an observation \
                 rather than a definition",
                ladder.len()
            ),
        };
    }
    if let Some(bad) = ladder
        .iter()
        .find(|rung| !rung.rho.is_finite() || !rung.rho_gradient.is_finite())
    {
        return SoftRhoGuardFloor::Indeterminate {
            with_floor: empty(),
            without_floor: empty(),
            reason: format!(
                "the ladder carries a non-finite rung (rho={}, g={})",
                bad.rho, bad.rho_gradient
            ),
        };
    }

    let guard: Vec<f64> = ladder
        .iter()
        .map(|rung| soft_rho_guard_emission_at(rung.rho, anchor))
        .collect();
    let with_floor = PencilFit::from_residuals(ladder, |rung| {
        rung.rho_gradient - soft_rho_guard_emission_at(rung.rho, anchor)
    });
    let without_floor = PencilFit::from_residuals(ladder, |rung| rung.rho_gradient);

    // Order matters, and it is not arbitrary. `with_floor` is checked first
    // because it is the *stronger* claim — it requires the residual under a
    // known constant to be a clean face — and because a criterion that carries
    // the floor cannot also satisfy `without_floor` (its raw pencil grows by
    // `e^{Δρ}` across the ladder, a factor 8100 here, five orders past the bar).
    match (with_floor.holds, without_floor.holds) {
        (true, false) => SoftRhoGuardFloor::Carried {
            face: with_floor,
            guard,
        },
        (false, true) => SoftRhoGuardFloor::AbsentDecayingFace {
            face: without_floor,
            guard,
        },
        (true, true) => SoftRhoGuardFloor::Indeterminate {
            with_floor,
            without_floor,
            reason: "both hypotheses fit, which is geometrically impossible for a \
                     nonzero floor (the floor is constant, so it cannot be a \
                     c*e^-rho tail as well) — the ladder is degenerate"
                .to_string(),
        },
        (false, false) => {
            // Neither SHAPE fits. The magnitude argument is independent of shape
            // and is the one that survives when the face has decayed into
            // roundoff, where the raw pencil is noise times `e^{30}`.
            let max_abs_gradient = ladder
                .iter()
                .fold(0.0f64, |acc, rung| acc.max(rung.rho_gradient.abs()));
            let min_guard = guard.iter().fold(f64::MAX, |acc, g| acc.min(g.abs()));
            if max_abs_gradient <= ABSENCE_MAGNITUDE_FRACTION * min_guard {
                return SoftRhoGuardFloor::AbsentBelowTheFloor {
                    max_abs_gradient,
                    min_guard,
                };
            }
            SoftRhoGuardFloor::Indeterminate {
                reason: format!(
                    "neither pencil is a single-signed constant, and max|g|={max_abs_gradient:.6e} \
                     is not below the floor's own {min_guard:.6e} either — this ladder does not \
                     answer the question. Check that the ladder is deep enough (the criterion's \
                     own tail must be small next to the floor), that the inner solve converged at \
                     every rung, and that no OTHER non-decaying term (a configured rho-prior) is \
                     in the criterion"
                ),
                with_floor,
                without_floor,
            }
        }
    }
}

#[cfg(test)]
#[path = "soft_rho_guard_floor_tests.rs"]
mod soft_rho_guard_floor_tests;
